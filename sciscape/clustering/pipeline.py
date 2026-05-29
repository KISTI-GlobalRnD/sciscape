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
from .integer_remap import integer_remap
from .leiden_java import run_leiden_java
from .leiden_rust import (
    RUST_AVAILABLE,
    project_membership_rust,
    remap_parquet_to_leiden_graph,
    run_leiden_rust,
    postprocess_small_clusters_rust,
)
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
from .tuning import _search_resolution_rust, resolve_resolution_schedule, scan_resolution_grid


_log_module = logging.getLogger(__name__)


def _compact_membership_array(values: np.ndarray | list[int]) -> np.ndarray:
    """Store non-negative cluster IDs in the narrowest safe integer dtype."""
    arr = np.asarray(values)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    if arr.size == 0:
        return np.ascontiguousarray(arr, dtype=np.uint32)

    if np.issubdtype(arr.dtype, np.signedinteger) and int(arr.min()) < 0:
        return np.ascontiguousarray(arr, dtype=np.int64)

    if int(arr.max()) <= np.iinfo(np.uint32).max:
        return np.ascontiguousarray(arr, dtype=np.uint32)
    return np.ascontiguousarray(arr, dtype=np.uint64)


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
    edges: pl.DataFrame | Path,
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
            iterations=10 if config.leiden_iterations is None else config.leiden_iterations,
            java_heap=config.java_heap,
        )
        _log(config,
             f"level {level_name}: java leiden in {time.perf_counter() - t_leiden:.2f}s "
             f"(gamma={gamma:.6g}, {result.n_clusters} clusters)",
             progress_log_path=progress_log_path)

        memberships_by_level[level_name] = _compact_membership_array(result.membership)
        resolutions_map[level_name] = float(gamma)
        qualities_map[level_name] = 0.0  # Java backend doesn't return quality

    # Join back UIDs
    t_join = time.perf_counter()
    # Build membership DataFrame with all levels
    membership_df = pl.read_parquet(remap.node_manifest_path, columns=["uid"]).with_columns(
        [
            pl.Series(f"cluster_{level_name}", mem)
            for level_name, mem in memberships_by_level.items()
        ]
    )
    _log(config, f"uid join-back in {time.perf_counter() - t_join:.2f}s",
         progress_log_path=progress_log_path)

    levels = tuple(memberships_by_level.keys())
    t_tables = time.perf_counter()
    tables = build_cluster_tables(
        membership_df,
        levels=levels,
        resolutions=resolutions_map,
        qualities=qualities_map,
    )
    _log(config, f"cluster table build in {time.perf_counter() - t_tables:.2f}s",
         progress_log_path=progress_log_path)

    _log(config, f"java pipeline finished in {time.perf_counter() - t0:.2f}s",
         progress_log_path=progress_log_path)
    return tables


def _run_rust_backend(
    edges: pl.DataFrame | Path,
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
    if not config.resolutions and not config.level_constraints:
        raise ValueError(
            "Rust backend requires either explicit resolutions or level_constraints."
        )

    t0 = time.perf_counter()

    # Integer remap + initial graph build.  For parquet input, prefer the Rust
    # direct path so we do not write raw sidecars only to read them back.
    from .runner import RustLeidenRunner

    seed = config.seed or 0
    n_iter = 10 if config.leiden_iterations is None else config.leiden_iterations
    remap = None
    rust_runner = None
    if isinstance(edges, (str, Path)) and Path(edges).suffix.lower() == ".parquet":
        t_remap_graph = time.perf_counter()
        try:
            direct = remap_parquet_to_leiden_graph(Path(edges), output_dir / "remap")
        except Exception as exc:
            _log_module.warning(
                "rust direct remap+graph build failed; falling back to sidecar path: %s",
                exc,
            )
        else:
            if direct is not None:
                remap, graph = direct
                rust_runner = RustLeidenRunner.from_graph(
                    graph,
                    default_iterations=n_iter,
                    default_seed=seed,
                )
                _log(
                    config,
                    f"integer remap + direct graph build in "
                    f"{time.perf_counter() - t_remap_graph:.2f}s "
                    f"({remap.n_nodes} nodes, {remap.n_edges} edges)",
                    progress_log_path=progress_log_path,
                )

    if remap is None or rust_runner is None:
        t_remap = time.perf_counter()
        remap = integer_remap(edges, output_dir / "remap", write_int_edges=False)
        _log(config, f"integer remap in {time.perf_counter() - t_remap:.2f}s "
             f"({remap.n_nodes} nodes, {remap.n_edges} edges)",
             progress_log_path=progress_log_path)
        rust_runner = RustLeidenRunner.from_edge_path(
            remap.int_edges_path,
            remap.n_nodes,
            default_iterations=n_iter,
            default_seed=seed,
        )

    memberships_by_level = OrderedDict()
    resolutions_map = OrderedDict()
    qualities_map = OrderedDict()

    # State for hierarchical contraction
    prev_membership_original: np.ndarray | None = None  # map back to original nodes
    prev_membership_graph: np.ndarray | None = None  # for warm-start on contracted graph

    if config.resolutions:
        level_plan = [
            (level_name, float(gamma), None)
            for level_name, gamma in config.resolutions.items()
        ]
    else:
        level_plan = [
            (f"level-{idx}", None, constraint)
            for idx, constraint in enumerate(config.level_constraints or (), start=1)
        ]

    n_levels = len(level_plan)
    for level_idx, (level_name, explicit_gamma, constraint) in enumerate(level_plan, start=1):
        search_membership = None
        if explicit_gamma is None:
            min_clusters, max_clusters = constraint
            t_search = time.perf_counter()
            search_result = _search_resolution_rust(
                rust_runner,
                level_name,
                min_clusters,
                max_clusters,
                config.resolution_bounds,
                config.max_iterations,
                progress=config.progress,
                n_iterations=n_iter,
                seed=seed,
                cache=None,
            )
            gamma = float(search_result.resolution)
            result_quality = float(search_result.quality)
            raw_n_clusters = int(search_result.cluster_count)
            search_membership = getattr(search_result, "_membership", None)
            _log(config,
                 f"level {level_name}: resolved rust gamma in "
                 f"{time.perf_counter() - t_search:.2f}s "
                 f"(γ={gamma:.6g}, {raw_n_clusters} clusters)",
                 progress_log_path=progress_log_path)
        else:
            gamma = float(explicit_gamma)
            result_quality = 0.0
            raw_n_clusters = 0

        if (
            search_membership is not None
            and len(search_membership) == rust_runner.n_nodes
            and prev_membership_graph is None
        ):
            membership = np.asarray(search_membership, dtype=np.uint64)
            _log(config,
                 f"level {level_name}: reused rust native search membership "
                 f"(γ={gamma:.6g}, {raw_n_clusters} clusters, Q={result_quality:.4f})",
                 progress_log_path=progress_log_path)
        else:
            t_leiden = time.perf_counter()
            result = rust_runner.run(
                gamma,
                seed=seed,
                n_iterations=n_iter,
                initial_membership=prev_membership_graph,
            )
            raw_n_clusters = int(result.n_clusters)
            result_quality = float(result.quality)
            _log(config,
                 f"level {level_name}: rust leiden in {time.perf_counter() - t_leiden:.2f}s "
                 f"(γ={gamma:.6g}, {raw_n_clusters} clusters, Q={result_quality:.4f})",
                 progress_log_path=progress_log_path)
            membership = np.asarray(result.membership, dtype=np.uint64)

        # Postprocess
        if config.postprocess is not None:
            min_size_val, min_weight_val = config.postprocess.resolve_thresholds(
                has_node_weights=rust_runner.has_node_weights
            )
            do_post = (
                (min_weight_val is not None and min_weight_val > 0)
                or (min_size_val is not None and min_size_val > 1)
            )
            if do_post:
                t_post = time.perf_counter()
                post = rust_runner.postprocess(
                    resolution=float(gamma),
                    min_size=int(min_size_val or 0),
                    min_weight=float(min_weight_val or 0.0),
                    membership=membership,
                    seed=seed,
                    n_iterations=n_iter,
                )
                _log(config,
                     f"level {level_name}: rust postprocess in "
                     f"{time.perf_counter() - t_post:.2f}s "
                     f"({raw_n_clusters}→{post.n_clusters} clusters, "
                     f"{len(post.rounds)} rounds)",
                     progress_log_path=progress_log_path)
                membership = post.membership

        # Map back to original node indices
        if prev_membership_original is None:
            original_membership = membership
        else:
            original_membership = project_membership_rust(
                membership,
                prev_membership_original,
            )

        stored_membership = _compact_membership_array(original_membership)
        memberships_by_level[level_name] = stored_membership
        resolutions_map[level_name] = float(gamma)
        qualities_map[level_name] = float(result_quality)
        prev_membership_original = stored_membership

        # Contract graph for next level
        n_clusters = int(membership.max()) + 1
        if level_idx < n_levels and n_clusters > 1:
            t_contract = time.perf_counter()
            rust_runner = rust_runner.contract(membership)
            prev_membership_graph = None  # singleton start on contracted graph
            _log(config,
                 f"level {level_name}: contracted to {rust_runner.n_nodes} nodes in "
                 f"{time.perf_counter() - t_contract:.2f}s",
                 progress_log_path=progress_log_path)

    # Join back UIDs
    t_join = time.perf_counter()
    membership_df = pl.read_parquet(remap.node_manifest_path, columns=["uid"]).with_columns(
        [
            pl.Series(f"cluster_{level_name}", mem)
            for level_name, mem in memberships_by_level.items()
        ]
    )
    _log(config, f"uid join-back in {time.perf_counter() - t_join:.2f}s",
         progress_log_path=progress_log_path)

    levels = tuple(memberships_by_level.keys())
    t_tables = time.perf_counter()
    tables = build_cluster_tables(
        membership_df,
        levels=levels,
        resolutions=resolutions_map,
        qualities=qualities_map,
    )
    _log(config, f"cluster table build in {time.perf_counter() - t_tables:.2f}s",
         progress_log_path=progress_log_path)

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
    if RUST_AVAILABLE and len(weight) > 500_000:
        try:
            from sciscape_leiden import rust_contract_edges

            pns = (
                np.ascontiguousarray(prev_node_sizes, dtype=np.int64)
                if prev_node_sizes is not None else None
            )
            return rust_contract_edges(
                np.ascontiguousarray(src, dtype=np.uint32),
                np.ascontiguousarray(dst, dtype=np.uint32),
                np.ascontiguousarray(weight, dtype=np.float64),
                np.ascontiguousarray(membership, dtype=np.uint64),
                pns,
            )
        except Exception as exc:
            _log_module.warning(
                "rust_contract_edges failed; falling back to scipy contraction: %s",
                exc,
            )

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
        node_sizes = np.bincount(
            mem,
            weights=prev_node_sizes.astype(np.float64),
            minlength=n_clusters,
        ).astype(np.int64)
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
    input_path = Path(zip_path)
    can_defer_parquet_load = input_path.suffix.lower() == ".parquet" and inner_name is None
    deferred_backend: str | None = None
    if can_defer_parquet_load:
        if config.backend == "java":
            deferred_backend = "java"
        elif config.backend in ("auto", "rust") and RUST_AVAILABLE:
            deferred_backend = "rust"

    if deferred_backend is not None:
        edges: pl.DataFrame | Path = input_path
        backend = deferred_backend
        _log(
            config,
            f"loaded edges path in {time.perf_counter() - t0:.2f}s",
            progress_log_path=progress_log_path,
        )
    else:
        edges = load_edge_table(zip_path, inner_name)
        _log(
            config,
            f"loaded edges in {time.perf_counter() - t0:.2f}s",
            progress_log_path=progress_log_path,
        )
        backend = _select_backend(edges, config)

    # ── Backend dispatch ──────────────────────────────────────
    if backend == "rust":
        output_dir = Path(config.log_dir or "workspace/output") / (config.run_id or "leiden_rust")
        return _run_rust_backend(
            edges, config,
            output_dir=output_dir,
            progress_log_path=progress_log_path,
        )
    if backend == "java":
        output_dir = Path(config.log_dir or "workspace/output") / (config.run_id or "leiden_java")
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
