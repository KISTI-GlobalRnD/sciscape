"""Hierarchical clustering with multi-layer consensus and auto-γ.

Builds nano → micro → meso (→ ...) hierarchy:
  1. Combine multi-layer edges (consensus weighting)
  2. Auto-γ → Leiden → postprocess at each level
  3. Contract graph after each level
  4. Stop when cluster count is small enough

Each level uses auto-gamma independently — contraction changes
edge density, so γ must be re-calibrated.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import polars as pl

from .auto_gamma import find_gamma
from .leiden_rust import run_leiden_rust, postprocess_small_clusters_rust, RUST_AVAILABLE
from ..linkage.combine import combine_edge_layers

log = logging.getLogger(__name__)

LEVEL_NAMES = ["nano", "micro", "meso", "macro", "mega"]

# Default: each level ~1/3 of previous cluster count
# For 100k papers: nano ~900, micro ~300, meso ~100, macro ~30
DEFAULT_TARGETS = {
    "nano": 0.5,        # ~900 clusters (avg ~110)
    "micro": 1.5,       # ~300 clusters (avg ~330)
    "meso": 5.0,        # ~100 clusters (avg ~1000)
    "macro": 15.0,      # ~30 clusters (avg ~3300)
    "mega": 40.0,
}

DEFAULT_MIN_SIZE = {
    "nano": 30,
    "micro": 100,
    "meso": 300,
    "macro": 1000,
    "mega": 3000,
}


@dataclass
class HierarchyLevel:
    """Result for one level of the hierarchy."""
    name: str
    gamma: float
    n_clusters: int
    max_pct: float
    avg_size: int
    top5: List[int]
    membership: np.ndarray  # original node indices
    elapsed: float


@dataclass
class HierarchyResult:
    """Complete hierarchical clustering result."""
    levels: List[HierarchyLevel]
    n_nodes: int

    @property
    def memberships_by_level(self) -> Dict[str, np.ndarray]:
        return {level.name: level.membership for level in self.levels}

    def to_dataframe(self, uids: Sequence[str]) -> pl.DataFrame:
        """Convert to DataFrame with uid + cluster columns per level."""
        data = {"uid": list(uids)}
        for level in self.levels:
            data[f"cluster_{level.name}"] = level.membership.tolist()
        return pl.DataFrame(data)


def build_hierarchy(
    edges: pl.DataFrame | None = None,
    *,
    layer_paths: Dict[str, Path] | None = None,
    layers: Dict[str, pl.DataFrame] | None = None,
    n_levels: int = 3,
    targets: Dict[str, float] | None = None,
    min_sizes: Dict[str, int] | None = None,
    combine_strategy: str = "consensus",
    combine_top_k: int = 30,
    seed: int = 42,
    stop_at_clusters: int = 10,
    cached_levels: List[HierarchyLevel] | None = None,
    progress: callable | None = None,
) -> HierarchyResult:
    """Build hierarchical clustering with auto-γ per level.

    Parameters
    ----------
    edges : pl.DataFrame, optional
        Pre-combined edge table. Used if layers/layer_paths not provided.
    layer_paths : dict, optional
        {name: Path} for multi-layer combination.
    layers : dict, optional
        {name: pl.DataFrame} for multi-layer combination.
    n_levels : int
        Maximum hierarchy depth (default 3: nano/micro/meso).
    targets : dict, optional
        {level_name: target_max_pct} for auto-gamma per level.
    min_sizes : dict, optional
        {level_name: min_cluster_size} for postprocess per level.
    combine_strategy : str
        Edge combination strategy (default "consensus").
    combine_top_k : int
        Per-node top-k filter.
    stop_at_clusters : int
        Stop adding levels when cluster count <= this.

    Returns
    -------
    HierarchyResult
    """
    if not RUST_AVAILABLE:
        raise ImportError("Rust Leiden backend required for hierarchical clustering")

    targets = targets or DEFAULT_TARGETS
    min_sizes = min_sizes or DEFAULT_MIN_SIZE

    def _log(msg):
        log.info(msg)
        if progress:
            progress(msg)

    # ── Step 1: Combine edges (if multi-layer) ──
    if layers or layer_paths:
        if layers is None:
            layers = {}
            for name, path in layer_paths.items():
                path = Path(path)
                if path.exists():
                    layers[name] = pl.read_parquet(path)
                    _log(f"Loaded {name}: {layers[name].height:,} edges")

        _log(f"Combining {len(layers)} layers (strategy={combine_strategy}, top_k={combine_top_k})")
        combined = combine_edge_layers(
            layers, strategy=combine_strategy, gcc=True, top_k=combine_top_k,
        )
    elif edges is not None:
        combined = edges
    else:
        raise ValueError("Provide edges, layers, or layer_paths")

    n_total = pl.concat([combined["uid1"], combined["uid2"]]).n_unique()
    _log(f"Combined: {n_total:,} nodes, {combined.height:,} edges")

    # ── Step 2: Integer remap (once) ──
    import tempfile
    from .integer_remap import integer_remap, join_back_uids

    _tmpdir = tempfile.mkdtemp()
    remap = integer_remap(combined, Path(_tmpdir) / "remap")
    ie = pl.read_parquet(remap.int_edges_path)
    src = ie["src"].to_numpy().astype(np.uint32)
    dst = ie["dst"].to_numpy().astype(np.uint32)
    w = ie["weight"].to_numpy().astype(np.float64)
    n_nodes = remap.n_nodes

    # uid mapping for back-projection
    uid_mem = join_back_uids(np.arange(n_nodes), remap.node_manifest_path)
    uids = uid_mem["uid"].to_list()

    # ── Step 3: Hierarchical levels ──
    levels: List[HierarchyLevel] = []
    cur_src, cur_dst, cur_w = src, dst, w
    cur_n = n_nodes
    prev_membership_original: np.ndarray | None = None
    node_sizes: np.ndarray | None = None

    # Resume from cached levels (skip re-computation)
    start_level = 0
    if cached_levels:
        for i, cached in enumerate(cached_levels):
            levels.append(cached)
            _log(f"Cached {cached.name}: {cached.n_clusters} cl, γ={cached.gamma:.2e}")

            # Sequentially contract for each cached level
            # Need to get per-level membership relative to current graph
            if i == 0:
                # First level: membership is on original graph nodes
                level_mem = cached.membership[:n_nodes].astype(np.uint64)
            else:
                # Higher level: membership is relative to previous contraction
                # Map original → current super-node → this level's cluster
                level_mem = cached.membership[prev_membership_original].astype(np.uint64)

            new_src, new_dst, new_w, new_n, new_sizes = _contract_edges(
                cur_src, cur_dst, cur_w, level_mem, node_sizes,
            )
            # Apply 1/rank + top-k on contracted edges
            contracted_top_k = min(max(3, new_n // 3), 30)
            contracted_df = pl.DataFrame({
                "uid1": new_src.astype(str), "uid2": new_dst.astype(str), "rel_sum2": new_w,
            })
            if contracted_df.height > contracted_top_k * new_n:
                from ..linkage.filters import filter_top_k
                contracted_df = filter_top_k(contracted_df, contracted_top_k)
            contracted_df = (
                contracted_df.sort("rel_sum2", descending=True)
                .with_row_index("_r")
                .with_columns((1.0 / (pl.col("_r") + 1).cast(pl.Float64)).alias("rel_sum2"))
                .drop("_r")
            )
            cur_src = contracted_df["uid1"].to_numpy().astype(np.uint32)
            cur_dst = contracted_df["uid2"].to_numpy().astype(np.uint32)
            cur_w = contracted_df["rel_sum2"].to_numpy().astype(np.float64)
            cur_n = new_n
            node_sizes = new_sizes
            prev_membership_original = cached.membership.astype(np.uint64)

        start_level = len(cached_levels)
        _log(f"Resumed from {start_level} cached levels, contracted to {cur_n} nodes")

    for level_idx in range(start_level, n_levels):
        level_name = LEVEL_NAMES[level_idx] if level_idx < len(LEVEL_NAMES) else f"level_{level_idx}"
        target_pct = targets.get(level_name, 30.0)
        min_size = min_sizes.get(level_name, 2)

        _log(f"\n{'='*50}")
        _log(f"Level {level_idx}: {level_name} (target_max<{target_pct}%, min_size={min_size})")
        _log(f"{'='*50}")
        t0 = time.perf_counter()

        # Find γ and run Leiden
        nw = node_sizes.astype(np.float64) if node_sizes is not None else None

        if level_idx == 0 and prev_membership_original is None:
            # First level on full graph: use auto_gamma with integer_remap
            gamma_result = find_gamma(
                _edges_to_df(cur_src, cur_dst, cur_w, cur_n),
                target_max_pct=target_pct,
                min_size=min_size,
                postprocess=True,
                progress=_log,
            )
            gamma = gamma_result.gamma
            _log(f"Auto-γ: {gamma:.2e} ({gamma_result.n_clusters} cl, max={gamma_result.max_pct}%)")
        else:
            # Contracted graph: direct γ sweep (small graph, fast)
            gamma = _sweep_gamma_direct(
                cur_src, cur_dst, cur_w, cur_n, nw,
                target_max_pct=target_pct, min_size=min_size, seed=seed,
                _log=_log,
            )

        r = run_leiden_rust(
            edges_src=cur_src, edges_dst=cur_dst, edges_weight=cur_w,
            resolution=gamma, n_nodes=cur_n, seed=seed, n_iterations=10,
            node_weights=nw,
        )
        # Postprocess: use node_weights for min_size on contracted graphs
        # On contracted graphs, "size" means original paper count (via node_weights)
        if node_sizes is not None:
            # Contracted: use min_weight (doc count sum) instead of min_size (super-node count)
            p = postprocess_small_clusters_rust(
                resolution=gamma, min_size=0,
                min_weight=float(min_size),
                membership=r.membership,
                edges_src=cur_src, edges_dst=cur_dst, edges_weight=cur_w,
                node_weights=nw, n_nodes=cur_n, seed=seed,
                gamma_decay=0.5, max_rounds=3,
                use_greedy=True, use_component_merge=True,
            )
        else:
            p = postprocess_small_clusters_rust(
                resolution=gamma, min_size=min_size,
                membership=r.membership,
                edges_src=cur_src, edges_dst=cur_dst, edges_weight=cur_w,
                n_nodes=cur_n, seed=seed,
                gamma_decay=0.5, max_rounds=5,
                use_greedy=True, use_component_merge=True,
            )
        mem = p.membership

        # Map back to original nodes
        if prev_membership_original is None:
            original_mem = mem.copy()
        else:
            original_mem = mem[prev_membership_original]

        sizes = Counter(original_mem.tolist())
        mx = max(sizes.values())
        top5 = sorted(sizes.values(), reverse=True)[:5]
        avg = n_total // len(sizes) if sizes else 0
        elapsed = time.perf_counter() - t0

        level = HierarchyLevel(
            name=level_name, gamma=gamma,
            n_clusters=len(sizes), max_pct=round(100 * mx / n_total, 1),
            avg_size=avg, top5=top5,
            membership=original_mem, elapsed=round(elapsed, 1),
        )
        levels.append(level)
        _log(f"→ {level_name}: {level.n_clusters} cl, max={level.max_pct}%, avg={avg}, {elapsed:.0f}s")

        # Stop condition
        if len(sizes) <= stop_at_clusters:
            _log(f"Stopping: {len(sizes)} clusters ≤ {stop_at_clusters}")
            break

        # Contract for next level
        prev_membership_original = original_mem.astype(np.uint64)

        new_src, new_dst, new_w, new_n, new_sizes = _contract_edges(
            cur_src, cur_dst, cur_w, mem, node_sizes,
        )

        # Re-normalize contracted edges: 1/rank + top-k pruning
        # Without this, contracted graph is near-complete and CPM can't split
        contracted_df = pl.DataFrame({
            "uid1": new_src.astype(str),
            "uid2": new_dst.astype(str),
            "rel_sum2": new_w,
        })
        contracted_top_k = min(max(3, new_n // 3), 30)
        from ..linkage.filters import filter_top_k
        if contracted_df.height > contracted_top_k * new_n:
            contracted_df = filter_top_k(contracted_df, contracted_top_k)
        # 1/rank re-normalization
        contracted_df = (
            contracted_df.sort("rel_sum2", descending=True)
            .with_row_index("_r")
            .with_columns((1.0 / (pl.col("_r") + 1).cast(pl.Float64)).alias("rel_sum2"))
            .drop("_r")
        )
        cur_src = contracted_df["uid1"].to_numpy().astype(np.uint32)
        cur_dst = contracted_df["uid2"].to_numpy().astype(np.uint32)
        cur_w = contracted_df["rel_sum2"].to_numpy().astype(np.float64)
        cur_n = new_n
        node_sizes = new_sizes
        _log(f"Contracted: {new_n} super-nodes, {contracted_df.height} edges (top-{contracted_top_k} + 1/rank)")

    return HierarchyResult(levels=levels, n_nodes=n_total)


def _sweep_gamma_direct(
    src, dst, w, n_nodes, node_weights,
    *,
    target_max_pct: float = 10.0,
    min_size: int = 3,
    seed: int = 42,
    _log=None,
) -> float:
    """Direct γ sweep on small contracted graph (no integer_remap)."""
    import math

    # Estimate γ range from edge weights
    w_median = float(np.median(w)) if len(w) > 0 else 1.0
    w_max = float(w.max()) if len(w) > 0 else 1.0
    lo = max(1e-6, w_median * 0.01)
    hi = w_max * 100

    n_probes = 8
    gammas = [10 ** (math.log10(lo) + i * (math.log10(hi) - math.log10(lo)) / max(n_probes - 1, 1))
              for i in range(n_probes)]

    best_gamma = gammas[0]
    best_pct = 100.0

    for gamma in gammas:
        r = run_leiden_rust(
            edges_src=src, edges_dst=dst, edges_weight=w,
            resolution=gamma, n_nodes=n_nodes, seed=seed,
            node_weights=node_weights,
        )
        from collections import Counter
        sizes = Counter(r.membership.tolist())
        mx = max(sizes.values())
        pct = 100 * mx / n_nodes
        n_cl = len(sizes)
        if _log:
            _log(f"  γ={gamma:.2e}: {n_cl} cl, max={pct:.1f}%")
        if pct <= target_max_pct and (pct > best_pct or best_pct > target_max_pct):
            best_pct = pct
            best_gamma = gamma
        elif pct > target_max_pct and pct < best_pct:
            best_pct = pct
            best_gamma = gamma

    if _log:
        _log(f"  Selected γ={best_gamma:.2e} (max={best_pct:.1f}%)")
    return best_gamma


def _edges_to_df(src, dst, w, n) -> pl.DataFrame:
    """Convert numpy edge arrays back to DataFrame for auto_gamma."""
    return pl.DataFrame({
        "uid1": src.astype(str),
        "uid2": dst.astype(str),
        "rel_sum2": w,
    })


def _contract_edges(src, dst, weight, membership, prev_node_sizes):
    """Contract edges via membership (same as pipeline._contract_edges)."""
    from scipy.sparse import coo_matrix

    mem = membership.astype(np.int64)
    new_src = mem[src.astype(np.int64)]
    new_dst = mem[dst.astype(np.int64)]
    mask = new_src != new_dst
    new_src, new_dst = new_src[mask], new_dst[mask]
    new_weight = weight[mask]

    n_clusters = int(mem.max()) + 1
    mat = coo_matrix((new_weight, (new_src, new_dst)), shape=(n_clusters, n_clusters))
    sym = (mat + mat.T).tocoo()
    upper = sym.row < sym.col
    out_src = sym.row[upper].astype(np.uint32)
    out_dst = sym.col[upper].astype(np.uint32)
    out_w = sym.data[upper]

    if prev_node_sizes is not None:
        node_sizes = np.zeros(n_clusters, dtype=np.int64)
        for v in range(len(mem)):
            node_sizes[mem[v]] += prev_node_sizes[v]
    else:
        node_sizes = np.bincount(mem, minlength=n_clusters)

    return out_src, out_dst, out_w, n_clusters, node_sizes


__all__ = ["build_hierarchy", "HierarchyResult", "HierarchyLevel"]
