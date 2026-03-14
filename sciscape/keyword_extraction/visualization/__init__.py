"""Visualization package for SciScape keyword extraction results.

Modules:
    _data_prep          Data preparation helpers
    _charts             Individual Plotly chart functions
    _dashboard_template HTML template constant
    _dashboard          Dashboard export (assembles data + template)
    _network_map        (planned) Cluster network map
    _hierarchy          (planned) Hierarchical zoom visualization
"""

from ._charts import (
    plot_cluster_keywords,
    plot_cross_cluster_terms,
    plot_depth_distribution,
    plot_score_distribution,
    plot_temporal_trends,
)
from ._dashboard import export_dashboard

__all__ = [
    "export_dashboard",
    "plot_cluster_keywords",
    "plot_cross_cluster_terms",
    "plot_depth_distribution",
    "plot_score_distribution",
    "plot_temporal_trends",
]
