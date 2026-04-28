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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import polars as pl

from .auto_gamma import find_gamma
from .leiden_rust import (
    build_leiden_graph,
    project_membership_rust,
    run_leiden_rust,
    postprocess_small_clusters_rust,
    RUST_AVAILABLE,
)
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
    uids: List[str] | None = None  # UID list in remap order (authoritative)

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
    n_levels: int = 4,
    targets: Dict[str, float] | None = None,
    min_sizes: Dict[str, int] | None = None,
    combine_strategy: str = "consensus",
    combine_top_k: int | str = "auto",
    seed: int = 42,
    stop_at_clusters: int = 5,
    cached_levels: List[HierarchyLevel] | None = None,
    cache_dir: Path | None = None,
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

    # ── Output directory setup ──
    out = Path(cache_dir) if cache_dir else None
    if out:
        out.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Combine edges (saved/loaded) ──
    combined_path = out / "combined_edges.parquet" if out else None
    if combined_path and combined_path.exists():
        _log(f"Loading combined edges: {combined_path}")
        combined = pl.read_parquet(combined_path)
    elif layers or layer_paths:
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
        if combined_path:
            combined.write_parquet(combined_path)
            _log(f"Saved combined edges → {combined_path}")
    elif edges is not None:
        combined = edges
    else:
        raise ValueError("Provide edges, layers, or layer_paths")

    # Validate edge columns
    required_cols = {"uid1", "uid2", "rel_sum2"}
    missing = required_cols - set(combined.columns)
    if missing:
        raise ValueError(f"Edge DataFrame missing columns: {missing}")

    n_total = pl.concat([combined["uid1"], combined["uid2"]]).n_unique()
    _log(f"Combined: {n_total:,} nodes, {combined.height:,} edges")

    # ── Step 2: Integer remap (in-memory, no disk I/O) ──
    from .integer_remap import integer_remap_memory

    src, dst, w, n_nodes, uids = integer_remap_memory(combined)

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
            # Need per-level membership relative to CURRENT graph nodes
            if i == 0:
                # First level: current graph = original graph
                level_mem = cached.membership[:n_nodes].astype(np.uint64)
            else:
                # Higher level: current graph has cur_n super-nodes
                # Build super-node → cluster mapping from cached original-node membership
                # Each super-node inherits its cluster from any member original node
                # prev_membership_original[v] = super-node ID for original node v
                # cached.membership[v] = cluster at this level for original node v
                # We need: level_mem[super_node_s] = cluster of any node in super_node_s
                cached_mem = cached.membership.astype(np.uint64)
                prev_mem = prev_membership_original.astype(np.uint64)
                if int(prev_mem.max()) >= cur_n:
                    log.warning("Cached level %s: prev_mem max %d >= cur_n %d, re-indexing",
                                cached.name, int(prev_mem.max()), cur_n)
                    cur_n = int(prev_mem.max()) + 1
                level_mem = np.zeros(cur_n, dtype=np.uint64)
                # Direct scatter: last write wins (all nodes in same super-node → same cluster)
                level_mem[prev_mem] = cached_mem[:len(prev_mem)]

            cur_src, cur_dst, cur_w, cur_n, node_sizes = _contract_and_normalize(
                cur_src, cur_dst, cur_w, level_mem, node_sizes,
            )
            prev_membership_original = cached.membership.astype(np.uint64)

        start_level = len(cached_levels)
        _log(f"Resumed from {start_level} cached levels, contracted to {cur_n} nodes")

    for level_idx in range(start_level, n_levels):
        level_name = LEVEL_NAMES[level_idx] if level_idx < len(LEVEL_NAMES) else f"level_{level_idx}"
        target_pct = targets.get(level_name, 30.0)
        min_size = min_sizes.get(level_name, 2)

        # Check saved level result
        level_dir = out / level_name if out else None
        level_mem_path = level_dir / "membership.parquet" if level_dir else None
        level_meta_path = level_dir / "meta.json" if level_dir else None

        if level_mem_path and level_mem_path.exists() and level_meta_path and level_meta_path.exists():
            import json as _json
            level_mem_df = pl.read_parquet(level_mem_path)
            with open(level_meta_path) as _f:
                level_meta = _json.load(_f)
            original_mem = np.array(level_mem_df["cluster"].to_list(), dtype=np.uint64)
            gamma = level_meta["gamma"]
            size_arr = np.bincount(original_mem.astype(np.int32))
            size_arr = size_arr[size_arr > 0]
            if len(size_arr) == 0:
                _log(f"WARNING: empty cached clustering at {level_name}, skipping")
                continue
            mx = int(size_arr.max())
            n_cl = len(size_arr)
            level = HierarchyLevel(
                name=level_name, gamma=gamma,
                n_clusters=n_cl, max_pct=round(100 * mx / max(n_total, 1), 1),
                avg_size=n_total // n_cl, top5=sorted(size_arr.tolist(), reverse=True)[:5],
                membership=original_mem, elapsed=0,
            )
            levels.append(level)
            _log(f"Loaded {level_name}: {level.n_clusters} cl, γ={gamma:.2e}")

            # Contract for next level
            if prev_membership_original is None:
                # First level: mem is on original graph
                level_mem_for_contract = original_mem[:n_nodes].astype(np.uint64)
            else:
                # Subsequent: build super-node membership from original-node membership
                pm = prev_membership_original.astype(np.uint64)
                alloc_n = max(cur_n, int(pm.max()) + 1) if len(pm) > 0 else cur_n
                level_mem_for_contract = np.zeros(alloc_n, dtype=np.uint64)
                level_mem_for_contract[pm] = original_mem[:len(pm)]
            prev_membership_original = original_mem.astype(np.uint64)
            cur_src, cur_dst, cur_w, cur_n, node_sizes = _contract_and_normalize(
                cur_src, cur_dst, cur_w, level_mem_for_contract, node_sizes,
            )
            continue

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

        try:
            graph = build_leiden_graph(
                edges_src=cur_src, edges_dst=cur_dst, edges_weight=cur_w,
                n_nodes=cur_n, node_weights=nw,
            )
            r = graph.run_leiden(
                resolution=gamma, seed=seed, n_iterations=10,
            )
        except AttributeError:
            graph = None
            r = run_leiden_rust(
                edges_src=cur_src, edges_dst=cur_dst, edges_weight=cur_w,
                resolution=gamma, n_nodes=cur_n, seed=seed, n_iterations=10,
                node_weights=nw,
            )
        # Postprocess: use node_weights for min_size on contracted graphs
        # On contracted graphs, "size" means original paper count (via node_weights)
        if node_sizes is not None:
            # Contracted: use min_weight (doc count sum) instead of min_size (super-node count)
            if graph is not None:
                p = graph.postprocess_small_clusters(
                    resolution=gamma, min_size=0,
                    min_weight=float(min_size),
                    membership=r.membership,
                    seed=seed,
                    gamma_decay=0.5, max_rounds=3,
                    use_greedy=True, use_component_merge=True,
                )
            else:
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
            if graph is not None:
                p = graph.postprocess_small_clusters(
                    resolution=gamma, min_size=min_size,
                    membership=r.membership,
                    seed=seed,
                    gamma_decay=0.5, max_rounds=5,
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
            original_mem = mem
        else:
            original_mem = project_membership_rust(mem, prev_membership_original)

        size_arr = np.bincount(original_mem.astype(np.int32))
        size_arr = size_arr[size_arr > 0]
        if len(size_arr) == 0:
            _log(f"WARNING: empty clustering at {level_name}, stopping")
            break
        mx = int(size_arr.max())
        top5 = sorted(size_arr.tolist(), reverse=True)[:5]
        n_cl = len(size_arr)
        avg = n_total // n_cl
        elapsed = time.perf_counter() - t0

        level = HierarchyLevel(
            name=level_name, gamma=gamma,
            n_clusters=n_cl, max_pct=round(100 * mx / max(n_total, 1), 1),
            avg_size=avg, top5=top5,
            membership=original_mem, elapsed=round(elapsed, 1),
        )
        levels.append(level)
        _log(f"→ {level_name}: {level.n_clusters} cl, max={level.max_pct}%, avg={avg}, {elapsed:.0f}s")

        # Save level result
        if level_dir:
            import json as _json
            level_dir.mkdir(parents=True, exist_ok=True)
            pl.DataFrame({"uid": uids, "cluster": original_mem.tolist()}).write_parquet(level_mem_path)
            meta_dict = {
                "gamma": gamma, "n_clusters": level.n_clusters,
                "max_pct": level.max_pct, "avg_size": level.avg_size,
                "top5": level.top5, "target_pct": target_pct, "min_size": min_size,
            }
            with open(level_meta_path, "w") as _f:
                _json.dump(meta_dict, _f, indent=2)
            _log(f"Saved → {level_dir}/")

        # Stop condition
        if n_cl <= stop_at_clusters:
            _log(f"Stopping: {n_cl} clusters ≤ {stop_at_clusters}")
            break

        # Contract for next level
        prev_membership_original = original_mem.astype(np.uint64)

        cur_src, cur_dst, cur_w, cur_n, node_sizes = _contract_and_normalize(
            cur_src, cur_dst, cur_w, mem, node_sizes,
        )
        _log(f"Contracted: {cur_n} super-nodes, {len(cur_w)} edges")

    result = HierarchyResult(levels=levels, n_nodes=n_total, uids=uids)

    # Save full hierarchy
    if out and uids:
        hierarchy_df = result.to_dataframe(uids)
        hierarchy_path = out / "hierarchy.parquet"
        hierarchy_df.write_parquet(hierarchy_path)
        _log(f"Saved hierarchy → {hierarchy_path}")

    return result


def _sweep_gamma_direct(
    src, dst, w, n_nodes, node_weights,
    *,
    target_max_pct: float = 10.0,
    min_size: int = 3,
    seed: int = 42,
    _log=None,
) -> float:
    """Direct γ sweep on small contracted graph (no integer_remap).

    Uses density-aware range estimation + binary refinement for
    contracted graphs that tend to be near-complete after contraction.
    """
    import math

    n_edges = len(w)
    w_median = float(np.median(w)) if n_edges > 0 else 1.0
    w_max = float(w.max()) if n_edges > 0 else 1.0

    # Density-aware bounds (contracted graphs are often dense)
    max_possible = n_nodes * (n_nodes - 1) / 2 if n_nodes > 1 else 1
    density = n_edges / max_possible
    density_boost = 1.0 + 50.0 * density

    lo = max(1e-6, w_median * 0.01 * density_boost)
    hi = w_max * 100

    # Phase 1: coarse sweep (12 probes for better coverage)
    n_probes = 12
    gammas = [10 ** (math.log10(lo) + i * (math.log10(hi) - math.log10(lo)) / max(n_probes - 1, 1))
              for i in range(n_probes)]
    try:
        graph = build_leiden_graph(
            edges_src=src, edges_dst=dst, edges_weight=w,
            n_nodes=n_nodes, node_weights=node_weights,
        )
    except AttributeError:
        graph = None

    cache = {}
    for gamma in gammas:
        if graph is not None:
            r = graph.run_leiden(resolution=gamma, seed=seed)
        else:
            r = run_leiden_rust(
                edges_src=src, edges_dst=dst, edges_weight=w,
                resolution=gamma, n_nodes=n_nodes, seed=seed,
                node_weights=node_weights,
            )
        mem = np.asarray(r.membership, dtype=np.int64)
        n_cl = len(set(mem.tolist()))
        if n_cl == 0:
            cache[gamma] = (0, 100.0)
            continue
        # Use node_weights-aware pct if available
        if node_weights is not None:
            total_weight = float(node_weights.sum())
            cluster_weights = np.bincount(mem, weights=node_weights.astype(np.float64))
            max_weight = float(cluster_weights.max())
            pct = 100 * max_weight / max(total_weight, 1)
        else:
            sizes = np.bincount(mem)
            mx = int(sizes.max())
            pct = 100 * mx / max(n_nodes, 1)
        cache[gamma] = (n_cl, pct)
        if _log:
            _log(f"  γ={gamma:.2e}: {n_cl} cl, max={pct:.1f}%")

    # Phase 2: binary refinement between best bracket
    sorted_gammas = sorted(cache.keys())
    for _ in range(4):
        below = [(g, cache[g]) for g in sorted_gammas if cache[g][1] <= target_max_pct]
        above = [(g, cache[g]) for g in sorted_gammas if cache[g][1] > target_max_pct]

        if not below or not above:
            break

        # Bracket: highest γ above target, lowest γ below target
        hi_g = max(above, key=lambda x: x[0])[0]
        lo_g = min(below, key=lambda x: x[0])[0]

        if lo_g / hi_g < 1.5:
            break  # close enough

        mid_g = 10 ** ((math.log10(hi_g) + math.log10(lo_g)) / 2)
        if graph is not None:
            r = graph.run_leiden(resolution=mid_g, seed=seed)
        else:
            r = run_leiden_rust(
                edges_src=src, edges_dst=dst, edges_weight=w,
                resolution=mid_g, n_nodes=n_nodes, seed=seed,
                node_weights=node_weights,
            )
        mem = np.asarray(r.membership, dtype=np.int64)
        n_cl = len(set(mem.tolist()))
        if n_cl == 0:
            cache[mid_g] = (0, 100.0)
            sorted_gammas = sorted(cache.keys())
            continue
        if node_weights is not None:
            total_weight = float(node_weights.sum())
            cluster_weights = np.bincount(mem, weights=node_weights.astype(np.float64))
            pct = 100 * float(cluster_weights.max()) / max(total_weight, 1)
        else:
            sizes = np.bincount(mem)
            pct = 100 * int(sizes.max()) / max(n_nodes, 1)
        cache[mid_g] = (n_cl, pct)
        sorted_gammas = sorted(cache.keys())
        if _log:
            _log(f"  γ={mid_g:.2e}: {n_cl} cl, max={pct:.1f}% (refine)")

    # Select best: closest to target from below
    candidates = [(g, info) for g, info in cache.items() if info[1] <= target_max_pct]
    if candidates:
        best_gamma, (_, best_pct) = max(candidates, key=lambda x: x[1][1])
    else:
        best_gamma, (_, best_pct) = min(cache.items(), key=lambda x: x[1][1])

    if _log:
        _log(f"  Selected γ={best_gamma:.2e} (max={best_pct:.1f}%)")
    return best_gamma


def _contract_and_normalize(cur_src, cur_dst, cur_w, mem, node_sizes):
    """Contract graph, apply top-k pruning, and 1/rank re-normalization.

    Returns (new_src, new_dst, new_w, n_clusters, node_sizes).
    """
    new_src, new_dst, new_w, new_n, new_sizes = _contract_edges(
        cur_src, cur_dst, cur_w, mem, node_sizes,
    )
    contracted_top_k = _adaptive_contracted_k(new_n)
    if len(new_w) > contracted_top_k * new_n:
        cdf = pl.DataFrame({
            "uid1": new_src.astype(str), "uid2": new_dst.astype(str), "rel_sum2": new_w,
        })
        from ..linkage.filters import filter_top_k
        cdf = filter_top_k(cdf, contracted_top_k)
        new_src = cdf["uid1"].to_numpy().astype(np.uint32)
        new_dst = cdf["uid2"].to_numpy().astype(np.uint32)
        new_w = cdf["rel_sum2"].to_numpy().astype(np.float64)
    # 1/rank re-normalization
    if len(new_w) > 0:
        order = np.argsort(-new_w)
        ranked_w = np.empty_like(new_w)
        ranked_w[order] = 1.0 / np.arange(1, len(new_w) + 1, dtype=np.float64)
    else:
        ranked_w = new_w
    return new_src, new_dst, ranked_w, new_n, new_sizes


def _adaptive_contracted_k(n_nodes: int) -> int:
    """Adaptive top-k for contracted graphs.

    Same sqrt-based formula as combine.py's compute_adaptive_k,
    but with lower floor (3) since contracted graphs are small.
    """
    import math
    return max(3, min(30, int(math.sqrt(n_nodes))))


def _edges_to_df(src, dst, w, n) -> pl.DataFrame:
    """Convert numpy edge arrays back to DataFrame for auto_gamma."""
    return pl.DataFrame({
        "uid1": src.astype(str),
        "uid2": dst.astype(str),
        "rel_sum2": w,
    })


def _contract_edges(src, dst, weight, membership, prev_node_sizes):
    """Contract edges via membership. Uses Rust when available."""
    # ── Rust fast path (only for large graphs where overhead is worthwhile) ──
    if RUST_AVAILABLE and len(weight) > 500_000:
        try:
            from sciscape_leiden import rust_contract_edges
            mem64 = membership.astype(np.uint64)
            pns = prev_node_sizes.astype(np.int64) if prev_node_sizes is not None else None
            out_src, out_dst, out_w, n_cl, sizes = rust_contract_edges(
                src.astype(np.uint32), dst.astype(np.uint32), weight.astype(np.float64),
                mem64, pns,
            )
            return out_src, out_dst, out_w, n_cl, sizes
        except Exception:
            pass  # fallback to Python

    from scipy.sparse import coo_matrix
    mem = np.array(membership, dtype=np.int64)
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
        # Vectorized weighted bincount: sum prev_node_sizes per cluster
        node_sizes = np.bincount(mem, weights=prev_node_sizes.astype(np.float64),
                                 minlength=n_clusters).astype(np.int64)
    else:
        node_sizes = np.bincount(mem, minlength=n_clusters)

    return out_src, out_dst, out_w, n_clusters, node_sizes


__all__ = ["build_hierarchy", "HierarchyResult", "HierarchyLevel"]
