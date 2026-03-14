"""Public API for the SciScape Leiden clustering module."""

from __future__ import annotations

from .config import (
    ClusterTables,
    LeidenConfig,
    EnsembleConfig,
    PostprocessConfig,
    HierarchyLevelConfig,
    HierarchyConfig,
)
from .io import load_edge_table
from .graph import build_graph, giant_component
from .clustering import attach_uids, partitions_to_polars, run_leiden_levels
from .hierarchy import build_cluster_tables, get_cluster_hierarchy
from .hierarchical_pipeline import (
    HierarchyPipelineResult,
    run_hierarchy_pipeline,
    run_hierarchy_pipeline_from_graph,
)
from .pipeline import run_pipeline
from .tuning import (
    ResolutionResult,
    ResolutionScanEntry,
    ResolutionScanResult,
    resolve_resolution_schedule,
    scan_resolution_grid,
)
from .logging import LogMetadata, DEFAULT_LOG_FILE
from .postprocess import PostprocessResult, merge_small_clusters
from .partitioning import partition_class
from .runner import LeidenRunner, LeidenRunResult
from .core_documents import (
    ClusterDocument,
    ClusterDBConfig,
    load_db_config,
    build_field_map,
    fetch_records,
    UIDDataExtractor,
    select_core_documents,
    build_core_documents,
)
from .cluster_naming import (
    ClusterSummary,
    create_client,
    detect_and_translate,
    summarise_cluster,
    summarise_clusters,
)
from .ensemble import (
    EnsembleMembership,
    EnsembleResult,
    run_ensemble_pipeline,
)

__all__ = [
    "ClusterTables",
    "LeidenConfig",
    "PostprocessConfig",
    "HierarchyLevelConfig",
    "HierarchyConfig",
    "load_edge_table",
    "build_graph",
    "giant_component",
    "run_leiden_levels",
    "partitions_to_polars",
    "attach_uids",
    "build_cluster_tables",
    "get_cluster_hierarchy",
    "run_pipeline",
    "run_hierarchy_pipeline",
    "run_hierarchy_pipeline_from_graph",
    "HierarchyPipelineResult",
    "partition_class",
    "LeidenRunner",
    "LeidenRunResult",
    "merge_small_clusters",
    "PostprocessResult",
    "resolve_resolution_schedule",
    "ResolutionResult",
    "ResolutionScanEntry",
    "ResolutionScanResult",
    "scan_resolution_grid",
    "LogMetadata",
    "DEFAULT_LOG_FILE",
    "ClusterDocument",
    "ClusterDBConfig",
    "load_db_config",
    "build_field_map",
    "fetch_records",
    "UIDDataExtractor",
    "select_core_documents",
    "build_core_documents",
    "ClusterSummary",
    "create_client",
    "detect_and_translate",
    "summarise_cluster",
    "summarise_clusters",
    "EnsembleConfig",
    "EnsembleMembership",
    "EnsembleResult",
    "run_ensemble_pipeline",
]
