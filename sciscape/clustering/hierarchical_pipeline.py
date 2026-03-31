"""High-level pipeline for building hierarchical Leiden classifications."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import igraph as ig
import polars as pl

from .config import ClusterTables, HierarchyConfig, LeidenConfig
from .graph import build_graph, giant_component
from .hierarchy import build_cluster_tables
from .hierarchy_builder import HierarchyBuilder, HierarchyBuildResult
from .io import load_edge_table


@dataclass(frozen=True)
class HierarchyPipelineResult:
    """Bundle containing the hierarchy metadata and the result tables."""

    hierarchy: HierarchyBuildResult
    tables: ClusterTables


def _build_result_for_graph(
    graph: ig.Graph,
    *,
    leiden_config: LeidenConfig,
    hierarchy_config: HierarchyConfig,
) -> HierarchyPipelineResult:
    builder = HierarchyBuilder(
        graph,
        objective=leiden_config.objective,
        default_iterations=leiden_config.leiden_iterations,
        default_seed=leiden_config.seed,
        default_postprocess=leiden_config.postprocess,
    )
    hierarchy = builder.build(hierarchy_config)

    if "uid" in graph.vs.attributes():
        uids = list(graph.vs["uid"])
    else:
        uids = [str(i) for i in range(graph.vcount())]

    membership_columns = {
        f"cluster_{name}": labels
        for name, labels in hierarchy.memberships_by_level.items()
    }
    membership_df = pl.DataFrame(membership_columns)
    membership_with_uids = membership_df.with_columns(
        pl.Series("uid", uids)
    ).select("uid", *membership_df.columns)

    tables = build_cluster_tables(
        membership_with_uids,
        levels=tuple(hierarchy.memberships_by_level.keys()),
        resolutions={layer.name: layer.resolution for layer in hierarchy.layers},
        qualities={layer.name: layer.quality for layer in hierarchy.layers},
    )

    return HierarchyPipelineResult(hierarchy=hierarchy, tables=tables)


def run_hierarchy_pipeline_from_graph(
    graph: ig.Graph,
    leiden_config: LeidenConfig,
    hierarchy_config: HierarchyConfig,
) -> HierarchyPipelineResult:
    """Construct a hierarchy from an already prepared graph."""

    return _build_result_for_graph(
        graph,
        leiden_config=leiden_config,
        hierarchy_config=hierarchy_config,
    )


def run_hierarchy_pipeline(
    zip_path: Path,
    inner_name: str,
    leiden_config: LeidenConfig,
    hierarchy_config: HierarchyConfig,
) -> HierarchyPipelineResult:
    """Construct a hierarchy using the configured levels and post-processing rules."""

    edges = load_edge_table(zip_path, inner_name)
    graph = build_graph(edges)
    giant = giant_component(graph)
    return _build_result_for_graph(
        giant,
        leiden_config=leiden_config,
        hierarchy_config=hierarchy_config,
    )


__all__ = [
    "HierarchyPipelineResult",
    "run_hierarchy_pipeline",
    "run_hierarchy_pipeline_from_graph",
]
