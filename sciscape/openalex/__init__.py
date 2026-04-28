"""OpenAlex API integration for sciscape.

Query → fetch → build edges → analyze, all from a search string.
"""

from .client import OpenAlexClient
from .edges import build_citation_edges
from .pipeline import run_openalex_pipeline, OpenAlexPipelineConfig

__all__ = [
    "OpenAlexClient",
    "build_citation_edges",
    "run_openalex_pipeline",
    "OpenAlexPipelineConfig",
]
