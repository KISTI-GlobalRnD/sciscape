"""Visualization package for SciScape keyword extraction results.

Modules:
    _data_prep          Data preparation helpers
    _charts             Individual Plotly chart functions
    _dashboard_template HTML template constant
    _dashboard          Dashboard export (assembles data + template)
    _network_map        Cluster network map (MDS / spring layout)
    _hierarchy          Hierarchical treemap / sunburst visualization
    _temporal           Temporal comparison (heatmap, trend lines)
"""

from ._charts import (
    plot_cluster_keywords,
    plot_cross_cluster_terms,
    plot_depth_distribution,
    plot_score_distribution,
    plot_temporal_trends,
)
from ._dashboard import export_dashboard, export_report, export_viewer
from ._hierarchy import plot_cluster_sunburst, plot_cluster_treemap
from ._network_map import plot_cluster_map, plot_cluster_map_with_keywords
from ._temporal import plot_cluster_trend_comparison, plot_temporal_heatmap

__all__ = [
    "export_dashboard",
    "export_report",
    "export_viewer",
    "plot_cluster_keywords",
    "plot_cluster_map",
    "plot_cluster_map_with_keywords",
    "plot_cluster_sunburst",
    "plot_cluster_treemap",
    "plot_cluster_trend_comparison",
    "plot_cross_cluster_terms",
    "plot_depth_distribution",
    "plot_score_distribution",
    "plot_temporal_heatmap",
    "plot_temporal_trends",
]
