"""Public API for the SciScape keyword extraction module."""

from __future__ import annotations

from .config import CORE_COLUMNS, TIER2_COLUMNS, TIER3_COLUMNS, KeywordExtractionConfig, KeywordRecord, VocabMergeConfig
from .abbreviations import build_abbreviation_lookup, extract_parenthetical_abbreviations
from .cooccurrence import collect_cooccurrence
from .cluster_sharded import adaptive_candidate_cap, build_cluster_shard_manifest, run_cluster_sharded_preflight
from .depth import DepthConfig, estimate_depth
from .diagnostics import KeywordDiagnostics, keyword_diagnostics, score_before_after
from .keyword_extraction import KeywordExtractionPipeline, run_keyword_pipeline
from .normalization import normalize_keywords
from .quality import annotate_keyword_quality, keyword_quality_residual_report, quality_flag_counts, write_keyword_quality_residual_report
from .rule_artifact import build_keyword_rule_artifact_inputs, write_keyword_cleaning_rule_artifacts
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
    "adaptive_candidate_cap",
    "build_abbreviation_lookup",
    "build_cluster_shard_manifest",
    "build_keyword_rule_artifact_inputs",
    "collect_cooccurrence",
    "estimate_depth",
    "extract_parenthetical_abbreviations",
    "keyword_diagnostics",
    "keyword_quality_residual_report",
    "normalize_keywords",
    "export_dashboard",
    "plot_cluster_keywords",
    "plot_cross_cluster_terms",
    "plot_depth_distribution",
    "plot_score_distribution",
    "plot_temporal_trends",
    "quality_flag_counts",
    "run_cluster_sharded_preflight",
    "run_keyword_pipeline",
    "score_before_after",
    "write_keyword_quality_residual_report",
    "write_keyword_cleaning_rule_artifacts",
]
