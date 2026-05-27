"""Public API for the SciScape keyword extraction module."""

from __future__ import annotations

from .config import CORE_COLUMNS, TIER2_COLUMNS, TIER3_COLUMNS, KeywordExtractionConfig, KeywordRecord, VocabMergeConfig
from .cooccurrence import collect_cooccurrence
from .depth import DepthConfig, estimate_depth
from .diagnostics import KeywordDiagnostics, keyword_diagnostics, score_before_after
from .keyword_extraction import KeywordExtractionPipeline, run_keyword_pipeline
from .normalization import normalize_keywords
from .quality import annotate_keyword_quality, quality_flag_counts
from .term_network import TermNetwork, TermNetworkConfig
from .vocab_cleansing import VocabSimGraph
from .visualization import (
    export_dashboard,
    plot_cluster_keywords,
    plot_cross_cluster_terms,
    plot_depth_distribution,
    plot_score_distribution,
    plot_temporal_trends,
)

__all__ = [
    "CORE_COLUMNS",
    "TIER2_COLUMNS",
    "TIER3_COLUMNS",
    "DepthConfig",
    "KeywordDiagnostics",
    "KeywordExtractionConfig",
    "KeywordExtractionPipeline",
    "KeywordRecord",
    "TermNetwork",
    "TermNetworkConfig",
    "VocabMergeConfig",
    "VocabSimGraph",
    "annotate_keyword_quality",
    "collect_cooccurrence",
    "estimate_depth",
    "keyword_diagnostics",
    "normalize_keywords",
    "export_dashboard",
    "plot_cluster_keywords",
    "plot_cross_cluster_terms",
    "plot_depth_distribution",
    "plot_score_distribution",
    "plot_temporal_trends",
    "quality_flag_counts",
    "run_keyword_pipeline",
    "score_before_after",
]
