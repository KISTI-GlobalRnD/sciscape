"""Backward-compatible re-exports from the refactored keyword extraction package.

All public symbols are now defined in their respective submodules:
- config.py: KeywordExtractionConfig, KeywordRecord, CORE_COLUMNS
- extraction.py: utilities and _DataSource
- pipeline.py: KeywordExtractionPipeline, run_keyword_pipeline
- llm_canonicalize.py: LLMCanonicalizeMixin
- temporal.py: TemporalMixin
"""

from .pipeline import KeywordExtractionPipeline, run_keyword_pipeline

__all__ = ["KeywordExtractionPipeline", "run_keyword_pipeline"]
