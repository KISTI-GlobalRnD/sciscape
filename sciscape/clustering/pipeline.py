"""End-to-end Leiden clustering pipeline."""

from __future__ import annotations

from pathlib import Path
import time

from collections import OrderedDict

import polars as pl

from .clustering import attach_uids
from .config import ClusterTables, HierarchyConfig, HierarchyLevelConfig, LeidenConfig
from .graph import build_graph, giant_component
from .hierarchy import build_cluster_tables
from .hierarchy_builder import HierarchyBuilder
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
