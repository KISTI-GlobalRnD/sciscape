"""High-level pipeline for building hierarchical Leiden classifications."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

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

    builder = HierarchyBuilder(
        giant,
        objective=leiden_config.objective,
        default_iterations=leiden_config.leiden_iterations,
        default_seed=leiden_config.seed,
        default_postprocess=leiden_config.postprocess,
    )
    hierarchy = builder.build(hierarchy_config)

    membership_columns = {
        f"cluster_{name}": labels
        for name, labels in hierarchy.memberships_by_level.items()
    }
    membership_df = pl.DataFrame(membership_columns)
    membership_with_uids = membership_df.with_columns(
        pl.Series("uid", giant.vs["uid"])
    ).select("uid", *membership_df.columns)

    tables = build_cluster_tables(
        membership_with_uids,
        levels=tuple(hierarchy.memberships_by_level.keys()),
        resolutions={layer.name: layer.resolution for layer in hierarchy.layers},
        qualities={layer.name: layer.quality for layer in hierarchy.layers},
    )

    return HierarchyPipelineResult(hierarchy=hierarchy, tables=tables)


__all__ = ["HierarchyPipelineResult", "run_hierarchy_pipeline"]
