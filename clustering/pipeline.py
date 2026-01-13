"""End-to-end Leiden clustering pipeline."""

from __future__ import annotations

from pathlib import Path
import time

from collections import OrderedDict

import polars as pl

from .clustering import attach_uids
from .config import ClusterTables, LeidenConfig
from .graph import build_graph, giant_component
from .hierarchy import build_cluster_tables
from .postprocess import merge_small_clusters
from .io import load_edge_table
from .logging import (
    DEFAULT_LOG_FILE,
    LogMetadata,
    PROGRESS_LOG_FILE,
    _now_iso,
    write_history_entry,
    write_progress_event,
)
from .runner import LeidenRunner
from .tuning import resolve_resolution_schedule


def _log(config: LeidenConfig, message: str) -> None:
    if config.progress:
        config.progress(message)
    if config.log_history:
        write_progress_event(message, path=PROGRESS_LOG_FILE)


def run_pipeline(
    zip_path: Path,
    inner_name: str,
    config: LeidenConfig,
) -> ClusterTables:
    """Run the complete Leiden clustering workflow and return result tables."""

    t0 = time.perf_counter()
    edges = load_edge_table(zip_path, inner_name)
    _log(config, f"loaded edges in {time.perf_counter() - t0:.2f}s")

    t_graph = time.perf_counter()
    edge_count = edges.height
    graph = build_graph(edges)
    _log(config, f"built graph in {time.perf_counter() - t_graph:.2f}s")

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
    )

    resolutions_map = OrderedDict()
    cluster_counts = OrderedDict()
    qualities_map = OrderedDict()

    memberships_by_level = OrderedDict()

    progress_cb = config.progress
    if config.log_history:
        def progress_with_log(message: str) -> None:
            write_progress_event(message, path=PROGRESS_LOG_FILE)
            if config.progress:
                config.progress(message)

        progress_cb = progress_with_log

    if config.resolutions:
        runner = LeidenRunner(
            giant,
            objective=config.objective,
            default_iterations=config.leiden_iterations,
            default_seed=config.seed,
        )
        initial_membership = None
        for level, gamma in config.resolutions.items():
            result = runner.run(
                gamma,
                seed=config.seed,
                n_iterations=config.leiden_iterations,
                initial_membership=initial_membership,
            )
            membership = result.membership
            if config.postprocess is not None:
                node_weights = giant.vs["weight"] if "weight" in giant.vs.attributes() else None
                post_result = merge_small_clusters(
                    giant,
                    membership,
                    min_size=config.postprocess.min_size,
                    min_weight=config.postprocess.min_weight,
                    node_weights=node_weights,
                    max_passes=max(config.postprocess.max_passes, 1),
                )
                membership = post_result.membership

            memberships_by_level[level] = membership
            resolutions_map[level] = gamma
            cluster_counts[level] = len(set(membership))
            qualities_map[level] = float(result.quality)
            initial_membership = membership
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
        _log(config, f"resolved resolutions in {time.perf_counter() - t_search:.2f}s")
        for level, result in schedule.items():
            resolutions_map[level] = result.resolution
            membership = list(result.partition.membership)
            if config.postprocess is not None:
                node_weights = giant.vs["weight"] if "weight" in giant.vs.attributes() else None
                post_result = merge_small_clusters(
                    giant,
                    membership,
                    min_size=config.postprocess.min_size,
                    min_weight=config.postprocess.min_weight,
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
    _log(config, f"built tables in {time.perf_counter() - t_tables:.2f}s")

    _log(config, f"pipeline finished in {time.perf_counter() - t0:.2f}s")

    if config.log_history:
        metadata = LogMetadata(
            source=str(zip_path),
            node_count=node_count,
            edge_count=edge_count,
            timestamp=_now_iso(),
        )
        try:
            write_history_entry(
                DEFAULT_LOG_FILE,
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
