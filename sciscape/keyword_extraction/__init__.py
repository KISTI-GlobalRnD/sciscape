"""Public API for the SciScape keyword extraction module."""

from __future__ import annotations

from .diagnostics import KeywordDiagnostics, keyword_diagnostics, score_before_after
from .keyword_extraction import KeywordExtractionConfig, KeywordExtractionPipeline, run_keyword_pipeline

__all__ = [
    "KeywordDiagnostics",
    "KeywordExtractionConfig",
    "KeywordExtractionPipeline",
    "keyword_diagnostics",
    "run_keyword_pipeline",
    "score_before_after",
]
