"""Public API for the SciScape keyword extraction module."""

from __future__ import annotations

from .config import CORE_COLUMNS, KeywordExtractionConfig, KeywordRecord
from .diagnostics import KeywordDiagnostics, keyword_diagnostics, score_before_after
from .keyword_extraction import KeywordExtractionPipeline, run_keyword_pipeline

__all__ = [
    "CORE_COLUMNS",
    "KeywordDiagnostics",
    "KeywordExtractionConfig",
    "KeywordExtractionPipeline",
    "KeywordRecord",
    "keyword_diagnostics",
    "run_keyword_pipeline",
    "score_before_after",
]
