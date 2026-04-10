"""End-to-end Leiden clustering pipeline."""

from __future__ import annotations

from pathlib import Path
import time

from collections import OrderedDict

import numpy as np
import polars as pl

import logging

from .clustering import attach_uids
from .config import ClusterTables, HierarchyConfig, HierarchyLevelConfig, LeidenConfig
from .graph import build_graph, giant_component
from .hierarchy import build_cluster_tables
from .hierarchy_builder import HierarchyBuilder
from .integer_remap import integer_remap, join_back_uids
from .leiden_java import run_leiden_java
from .leiden_rust import RUST_AVAILABLE, run_leiden_rust, postprocess_small_clusters_rust
from .postprocess import merge_small_clusters
from .io import load_edge_table
from .logging import (
    DEFAULT_LOG_FILE,
    LogMetadata,
    PROGRESS_LOG_FILE,
    _now_iso,
    resolve_log_path,
    write_history_entry,
    write_progress_event,
)
from .tuning import resolve_resolution_schedule, scan_resolution_grid


_log_module = logging.getLogger(__name__)


def _select_backend(edges: pl.DataFrame, config: LeidenConfig) -> str:
    """Determine which backend to use: 'rust', 'igraph', or 'java'.

    Auto priority: rust (if available) > java (large graphs) > igraph.

    Rust is preferred because it:
    - Has no JVM startup cost (~2s saved per run)
    - Passes numpy arrays directly via PyO3 (zero-copy, no file I/O)
    - Uses native memory layout with cache-friendly CSR + unsafe hot paths
    - Avoids Python/igraph C-extension overhead for leidenalg
    - Runs contraction, refinement, and postprocess entirely in compiled code
    """
    if config.backend in ("igraph", "java", "rust"):
        if config.backend == "rust" and not RUST_AVAILABLE:
            _log_module.warning(
                "rust backend requested but sciscape-leiden not installed, "
                "falling back to auto selection"
            )
        else:
            return config.backend
    # auto: prefer rust when available
    if RUST_AVAILABLE:
        _log_module.info("auto-selected rust backend (sciscape-leiden available)")
        return "rust"
    # fallback: java for large graphs, igraph for small
    n_nodes = pl.concat([edges["uid1"], edges["uid2"]]).n_unique()
    if n_nodes >= config.auto_backend_threshold:
        _log_module.info(
            "auto-selected java backend (%d nodes >= %d threshold)",
            n_nodes, config.auto_backend_threshold,
        )
        return "java"
    return "igraph"


def _run_java_backend(
    edges: pl.DataFrame,
    config: LeidenConfig,
    *,
    output_dir: Path,
    progress_log_path: Path,
) -> ClusterTables:
    """Run Leiden via the Java backend (no igraph)."""
    if not config.resolutions:
        raise ValueError(
            "Java backend requires explicit resolutions in LeidenConfig.resolutions. "
            "level_constraints (binary search) is not supported with the Java backend."
        )

    t0 = time.perf_counter()

    # Integer remap
    t_remap = time.perf_counter()
    remap = integer_remap(edges, output_dir / "remap")
    _log(config, f"integer remap in {time.perf_counter() - t_remap:.2f}s "
         f"({remap.n_nodes} nodes, {remap.n_edges} edges)",
         progress_log_path=progress_log_path)

    # Run Leiden at each resolution
    memberships_by_level = OrderedDict()
    resolutions_map = OrderedDict()
    qualities_map = OrderedDict()

    for level_name, gamma in config.resolutions.items():
        t_leiden = time.perf_counter()
        result = run_leiden_java(
            remap.int_edges_path,
            resolution=float(gamma),
            n_nodes=remap.n_nodes,
            jar_path=config.jar_path,
            seed=config.seed or 0,
            iterations=config.leiden_iterations or 10,
            java_heap=config.java_heap,
        )
        _log(config,
             f"level {level_name}: java leiden in {time.perf_counter() - t_leiden:.2f}s "
             f"(gamma={gamma:.6g}, {result.n_clusters} clusters)",
             progress_log_path=progress_log_path)

        memberships_by_level[level_name] = result.membership.tolist()
        resolutions_map[level_name] = float(gamma)
        qualities_map[level_name] = 0.0  # Java backend doesn't return quality

    # Join back UIDs
    t_join = time.perf_counter()
    uid_cluster = join_back_uids(
        memberships_by_level[next(iter(memberships_by_level))],
        remap.node_manifest_path,
    )
    # Build membership DataFrame with all levels
    membership_df = uid_cluster.select("uid")
    for level_name, mem in memberships_by_level.items():
        membership_df = membership_df.with_columns(
            pl.Series(f"cluster_{level_name}", mem),
        )
    _log(config, f"uid join-back in {time.perf_counter() - t_join:.2f}s",
         progress_log_path=progress_log_path)

    levels = tuple(memberships_by_level.keys())
    tables = build_cluster_tables(
        membership_df,
        levels=levels,
        resolutions=resolutions_map,
        qualities=qualities_map,
    )

    _log(config, f"java pipeline finished in {time.perf_counter() - t0:.2f}s",
         progress_log_path=progress_log_path)
    return tables


def _run_rust_backend(
    edges: pl.DataFrame,
    config: LeidenConfig,
    *,
    output_dir: Path,
    progress_log_path: Path,
) -> ClusterTables:
    """Run Leiden via the Rust backend (numpy arrays, no file I/O).

    Supports hierarchical multi-level clustering with graph contraction
    between levels, warm-start via initial_membership, and optional
    postprocessing with cascading γ.
    """
    if not config.resolutions:
        raise ValueError(
            "Rust backend requires explicit resolutions in LeidenConfig.resolutions. "
            "level_constraints (binary search) is not supported with the Rust backend."
        )

    t0 = time.perf_counter()

    # Integer remap (reuse existing infra for UID → int mapping)
    t_remap = time.perf_counter()
    remap = integer_remap(edges, output_dir / "remap")
    _log(config, f"integer remap in {time.perf_counter() - t_remap:.2f}s "
         f"({remap.n_nodes} nodes, {remap.n_edges} edges)",
         progress_log_path=progress_log_path)

    # Load edges into numpy once (shared across all levels)
    int_edges = pl.read_parquet(remap.int_edges_path)
    edges_src = int_edges["src"].to_numpy().astype(np.uint32)
    edges_dst = int_edges["dst"].to_numpy().astype(np.uint32)
    edges_weight = int_edges["weight"].to_numpy().astype(np.float64)
    n_nodes = remap.n_nodes

    memberships_by_level = OrderedDict()
    resolutions_map = OrderedDict()
    qualities_map = OrderedDict()

    # State for hierarchical contraction
    cur_src, cur_dst, cur_weight = edges_src, edges_dst, edges_weight
    cur_n_nodes = n_nodes
    prev_membership_original: np.ndarray | None = None  # map back to original nodes
    prev_membership_graph: np.ndarray | None = None  # for warm-start on contracted graph
    node_sizes: np.ndarray | None = None  # per-supernode original counts

    seed = config.seed or 0
    n_iter = config.leiden_iterations or 10

    for level_name, gamma in config.resolutions.items():
        t_leiden = time.perf_counter()
        result = run_leiden_rust(
            edges_src=cur_src,
            edges_dst=cur_dst,
            edges_weight=cur_weight,
            resolution=float(gamma),
            n_nodes=cur_n_nodes,
            seed=seed,
            n_iterations=n_iter,
            initial_membership=prev_membership_graph,
        )
        _log(config,
             f"level {level_name}: rust leiden in {time.perf_counter() - t_leiden:.2f}s "
             f"(γ={gamma:.6g}, {result.n_clusters} clusters, Q={result.quality:.4f})",
             progress_log_path=progress_log_path)

        membership = result.membership

        # Postprocess
        if config.postprocess is not None:
            min_size, _ = config.postprocess.resolve_thresholds(has_node_weights=False)
            if min_size is not None and min_size > 1:
                t_post = time.perf_counter()
                post = postprocess_small_clusters_rust(
                    resolution=float(gamma),
                    min_size=min_size,
                    membership=membership,
                    edges_src=cur_src,
                    edges_dst=cur_dst,
                    edges_weight=cur_weight,
                    n_nodes=cur_n_nodes,
                    seed=seed,
                    n_iterations=n_iter,
                )
                _log(config,
                     f"level {level_name}: rust postprocess in "
                     f"{time.perf_counter() - t_post:.2f}s "
                     f"({result.n_clusters}→{post.n_clusters} clusters, "
                     f"{len(post.rounds)} rounds)",
                     progress_log_path=progress_log_path)
                membership = post.membership

        # Map back to original node indices
        if prev_membership_original is None:
            original_membership = membership.copy()
        else:
            original_membership = membership[prev_membership_original]

        memberships_by_level[level_name] = original_membership.tolist()
        resolutions_map[level_name] = float(gamma)
        qualities_map[level_name] = float(result.quality)
        prev_membership_original = np.asarray(
            memberships_by_level[level_name], dtype=np.uint64
        )

        # Contract graph for next level
        n_clusters = int(membership.max()) + 1
        if n_clusters > 1:
            t_contract = time.perf_counter()
            cur_src, cur_dst, cur_weight, cur_n_nodes, node_sizes = \
                _contract_edges(cur_src, cur_dst, cur_weight, membership, node_sizes)
            prev_membership_graph = None  # singleton start on contracted graph
            _log(config,
                 f"level {level_name}: contracted to {cur_n_nodes} nodes in "
                 f"{time.perf_counter() - t_contract:.2f}s",
                 progress_log_path=progress_log_path)

    # Join back UIDs
    t_join = time.perf_counter()
    first_level = next(iter(memberships_by_level))
    uid_cluster = join_back_uids(
        memberships_by_level[first_level],
        remap.node_manifest_path,
    )
    membership_df = uid_cluster.select("uid")
    for level_name, mem in memberships_by_level.items():
        membership_df = membership_df.with_columns(
            pl.Series(f"cluster_{level_name}", mem),
        )
    _log(config, f"uid join-back in {time.perf_counter() - t_join:.2f}s",
         progress_log_path=progress_log_path)

    levels = tuple(memberships_by_level.keys())
    tables = build_cluster_tables(
        membership_df,
        levels=levels,
        resolutions=resolutions_map,
        qualities=qualities_map,
    )

    _log(config, f"rust pipeline finished in {time.perf_counter() - t0:.2f}s",
         progress_log_path=progress_log_path)
    return tables


def _contract_edges(
    src: np.ndarray,
    dst: np.ndarray,
    weight: np.ndarray,
    membership: np.ndarray,
    prev_node_sizes: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]:
    """Contract edges via membership, aggregate weights, remove self-loops.

    Returns (new_src, new_dst, new_weight, n_clusters, node_sizes).
    """
    mem = membership.astype(np.int64)
    new_src = mem[src.astype(np.int64)]
    new_dst = mem[dst.astype(np.int64)]

    # Remove self-loops
    mask = new_src != new_dst
    new_src = new_src[mask]
    new_dst = new_dst[mask]
    new_weight = weight[mask]

    n_clusters = int(mem.max()) + 1

    # Aggregate duplicate edges using sparse matrix
    from scipy.sparse import coo_matrix
    mat = coo_matrix(
        (new_weight, (new_src, new_dst)),
        shape=(n_clusters, n_clusters),
    )
    # Sum duplicates and extract upper triangle (undirected)
    mat = mat.tocsr()
    # Symmetrize: take upper triangle of (mat + mat.T)
    sym = mat + mat.T
    sym = sym.tocoo()
    # Keep only upper triangle
    upper = sym.row < sym.col
    out_src = sym.row[upper].astype(np.uint32)
    out_dst = sym.col[upper].astype(np.uint32)
    out_weight = sym.data[upper]

    # Compute node_sizes for contracted graph
    if prev_node_sizes is not None:
        node_sizes = np.zeros(n_clusters, dtype=np.int64)
        for v in range(len(mem)):
            node_sizes[mem[v]] += prev_node_sizes[v]
    else:
        node_sizes = np.bincount(mem, minlength=n_clusters)

    return out_src, out_dst, out_weight, n_clusters, node_sizes


def _log(config: LeidenConfig, message: str, *, progress_log_path: Path) -> None:
    if config.progress:
        config.progress(message)
    if config.log_history:
        write_progress_event(message, path=progress_log_path)


def _resolve_stability_seeds(config: LeidenConfig) -> tuple[int, ...]:
    if config.stability_seeds:
        return tuple(dict.fromkeys(int(s) for s in config.stability_seeds))
    if config.seed is not None:
        base = int(config.seed)
        return (base, base + 1, base + 2)
    return (0, 1, 2)


def _run_hierarchy_with_explicit_resolutions(
    giant,
    config: LeidenConfig,
):
    levels = [
        HierarchyLevelConfig(
            name=str(level_name),
            resolution=float(gamma),
            objective=config.objective,
            seeds=(),
            iterations=config.leiden_iterations,
            postprocess=None,
        )
        for level_name, gamma in config.resolutions.items()
    ]
    hierarchy_cfg = HierarchyConfig(
        levels=levels,
        reuse_membership=True,
        contract_weights="sum",
        contract_loops=True,
    )
    builder = HierarchyBuilder(
        giant,
        objective=config.objective,
        default_iterations=config.leiden_iterations,
        default_seed=config.seed,
        default_postprocess=config.postprocess,
    )
    return builder.build(hierarchy_cfg)


def run_pipeline(
    zip_path: Path,
    inner_name: str,
    config: LeidenConfig,
) -> ClusterTables:
    """Run the complete Leiden clustering workflow and return result tables."""
    progress_log_path = resolve_log_path(
        default_path=PROGRESS_LOG_FILE,
        explicit_path=config.progress_log_path,
        log_dir=config.log_dir,
        run_id=config.run_id,
    )
    history_log_path = resolve_log_path(
        default_path=DEFAULT_LOG_FILE,
        explicit_path=config.history_log_path,
        log_dir=config.log_dir,
        run_id=config.run_id,
    )

    t0 = time.perf_counter()
    edges = load_edge_table(zip_path, inner_name)
    _log(config, f"loaded edges in {time.perf_counter() - t0:.2f}s", progress_log_path=progress_log_path)

    # ── Backend dispatch ──────────────────────────────────────
    backend = _select_backend(edges, config)
    if backend == "rust":
        output_dir = Path(config.log_dir or ".") / (config.run_id or "leiden_rust")
        return _run_rust_backend(
            edges, config,
            output_dir=output_dir,
            progress_log_path=progress_log_path,
        )
    if backend == "java":
        output_dir = Path(config.log_dir or ".") / (config.run_id or "leiden_java")
        return _run_java_backend(
            edges, config,
            output_dir=output_dir,
            progress_log_path=progress_log_path,
        )

    # ── igraph path (fallback) ───────────────────────────────
    t_graph = time.perf_counter()
    edge_count = edges.height
    graph = build_graph(edges)
    _log(config, f"built graph in {time.perf_counter() - t_graph:.2f}s", progress_log_path=progress_log_path)

    t_giant = time.perf_counter()
    giant = giant_component(graph)
    giant_build_time = time.perf_counter() - t_giant
    total_nodes = graph.vcount()
    node_count = giant.vcount()
    coverage = (node_count / total_nodes) if total_nodes else 1.0
    _log(
        config,
        (
            f"extracted giant component in {giant_build_time:.2f}s "
            f"({node_count}/{total_nodes} nodes, {coverage:.2%} coverage)"
        ),
        progress_log_path=progress_log_path,
    )

    resolutions_map = OrderedDict()
    cluster_counts = OrderedDict()
    qualities_map = OrderedDict()

    memberships_by_level = OrderedDict()

    progress_cb = config.progress
    if config.log_history:

        def progress_with_log(message: str) -> None:
            write_progress_event(message, path=progress_log_path)
            if config.progress:
                config.progress(message)

        progress_cb = progress_with_log

    if config.resolutions:
        hierarchy = _run_hierarchy_with_explicit_resolutions(giant, config)
        for layer in hierarchy.layers:
            memberships_by_level[layer.name] = hierarchy.memberships_by_level[layer.name]
            resolutions_map[layer.name] = float(layer.resolution)
            cluster_counts[layer.name] = int(layer.cluster_count)
            qualities_map[layer.name] = float(layer.quality)
    elif config.level_constraints:
        t_search = time.perf_counter()
        schedule = resolve_resolution_schedule(
            giant,
            config.level_constraints,
            config.objective,
            config.resolution_bounds,
            config.max_iterations,
            progress=progress_cb,
            n_iterations=config.leiden_iterations,
            seed=config.seed,
        )
        _log(
            config,
            f"resolved resolutions in {time.perf_counter() - t_search:.2f}s",
            progress_log_path=progress_log_path,
        )
        for level, result in schedule.items():
            resolutions_map[level] = result.resolution
            membership = list(result.partition.membership)
            if config.postprocess is not None:
                node_weights = giant.vs["weight"] if "weight" in giant.vs.attributes() else None
                min_size, min_weight = config.postprocess.resolve_thresholds(
                    has_node_weights=node_weights is not None
                )
                post_result = merge_small_clusters(
                    giant,
                    membership,
                    min_size=min_size,
                    min_weight=min_weight,
                    node_weights=node_weights,
                    max_passes=max(config.postprocess.max_passes, 1),
                )
                membership = post_result.membership
            memberships_by_level[level] = membership
            cluster_counts[level] = len(set(membership))
            qualities_map[level] = float(result.quality)
    else:
        raise ValueError(
            "LeidenConfig must specify either explicit resolutions or level_constraints"
        )

    if config.stability_metric:
        seeds = _resolve_stability_seeds(config)
        if len(seeds) < 2:
            _log(
                config,
                f"stability skipped: need >=2 seeds, got {len(seeds)}",
                progress_log_path=progress_log_path,
            )
        else:
            t_stability = time.perf_counter()
            scan = scan_resolution_grid(
                giant,
                list(resolutions_map.values()),
                seeds=seeds,
                objective=config.objective,
                n_iterations=config.leiden_iterations,
                postprocess=config.postprocess,
                stability_metric=config.stability_metric,
                parallel=False,
            )
            _log(
                config,
                f"computed stability in {time.perf_counter() - t_stability:.2f}s "
                f"(metric={config.stability_metric}, seeds={list(seeds)})",
                progress_log_path=progress_log_path,
            )
            if scan.stability:
                for level, gamma in resolutions_map.items():
                    score = scan.stability.get(float(gamma))
                    if score is None:
                        continue
                    _log(
                        config,
                        f"{level}: stability_{config.stability_metric}={score:.6f} (gamma={float(gamma):.6g})",
                        progress_log_path=progress_log_path,
                    )

    t_tables = time.perf_counter()
    membership = pl.DataFrame({
        f"cluster_{level}": labels
        for level, labels in memberships_by_level.items()
    })
    membership_with_uids = attach_uids(membership, giant)

    levels = tuple(memberships_by_level.keys())
    tables = build_cluster_tables(
        membership_with_uids,
        levels=levels,
        resolutions=resolutions_map,
        qualities=qualities_map,
    )
    _log(config, f"built tables in {time.perf_counter() - t_tables:.2f}s", progress_log_path=progress_log_path)

    _log(config, f"pipeline finished in {time.perf_counter() - t0:.2f}s", progress_log_path=progress_log_path)

    if config.log_history:
        metadata = LogMetadata(
            source=str(zip_path),
            node_count=node_count,
            edge_count=edge_count,
            timestamp=_now_iso(),
        )
        try:
            write_history_entry(
                history_log_path,
                metadata=metadata,
                levels=levels,
                resolutions=resolutions_map,
                cluster_counts=cluster_counts,
                coverage=coverage,
                qualities=qualities_map,
            )
        except Exception as exc:  # pragma: no cover - logging should not crash pipeline
            if config.progress:
                config.progress(f"[warning] failed to write history log: {exc}")

    return tables


__all__ = ["run_pipeline"]
