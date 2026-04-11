"""End-to-end OpenAlex query → analysis pipeline.

    query → fetch works → build edges → landscape (cluster + keywords + report)

Designed as a library function callable from CLI, web, or notebooks.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence

import polars as pl

from .client import OpenAlexClient, WorkRecord
from .edges import build_citation_edges, works_to_abstracts

log = logging.getLogger(__name__)


@dataclass
class OpenAlexPipelineConfig:
    """Configuration for the query → analyze pipeline."""

    # ── Query ──
    query: str = ""
    filters: Dict[str, str] = field(default_factory=dict)
    max_works: int = 10000

    # ── API ──
    email: str | None = None

    # ── Edge building ──
    edge_types: Sequence[str] = ("dc", "bc")
    normalization: str = "fractional"
    bc_topk: int = 50
    min_shared_refs: int = 1

    # ── Output ──
    output_dir: Path = Path("openalex_output")

    # ── Landscape (forwarded) ──
    run_landscape: bool = True

    # ── Callbacks ──
    progress: Callable[[str], None] | None = None


@dataclass
class OpenAlexPipelineResult:
    """Result of the pipeline."""
    n_works: int
    n_edges: Dict[str, int]
    works: list  # List[WorkRecord]
    abstracts_path: Path | None
    edges_path: Path | None
    landscape_dir: Path | None


def run_openalex_pipeline(
    config: OpenAlexPipelineConfig,
) -> OpenAlexPipelineResult:
    """Run the full query → fetch → edges → landscape pipeline.

    Parameters
    ----------
    config : OpenAlexPipelineConfig
        Pipeline configuration.

    Returns
    -------
    OpenAlexPipelineResult
    """
    t0 = time.perf_counter()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _log(msg: str) -> None:
        log.info(msg)
        if config.progress:
            config.progress(msg)

    # ── Step 1: Query OpenAlex ──
    _log(f"Querying OpenAlex: '{config.query}' (max {config.max_works} works)")
    client = OpenAlexClient(
        email=config.email,
        progress=config.progress,
    )
    works = client.search_works(
        config.query,
        filters=config.filters,
        max_results=config.max_works,
    )
    _log(f"Fetched {len(works)} works in {time.perf_counter() - t0:.1f}s")

    if not works:
        _log("No works found. Aborting.")
        return OpenAlexPipelineResult(
            n_works=0, n_edges={}, works=[],
            abstracts_path=None, edges_path=None, landscape_dir=None,
        )

    # ── Step 2: Save abstracts ──
    abstracts_df = works_to_abstracts(works)
    abstracts_path = output_dir / "abstracts.parquet"
    abstracts_df.write_parquet(abstracts_path)
    _log(f"Saved {abstracts_df.height} abstracts → {abstracts_path}")

    # ── Step 3: Build citation edges ──
    t_edges = time.perf_counter()
    edge_tables = build_citation_edges(
        works,
        normalization=config.normalization,
        bc="bc" in config.edge_types,
        bc_topk=config.bc_topk,
        min_shared_refs=config.min_shared_refs,
    )

    # Combine edge types
    edge_dfs = []
    for etype in config.edge_types:
        if etype in edge_tables and edge_tables[etype].height > 0:
            edge_dfs.append(edge_tables[etype])

    n_edges = {}
    if edge_dfs:
        combined = pl.concat(edge_dfs).group_by(["uid1", "uid2"]).agg(
            pl.col("rel_sum2").sum()
        )
        edges_path = output_dir / "edges.parquet"
        combined.write_parquet(edges_path)
        for etype, df in edge_tables.items():
            n_edges[etype] = df.height
        _log(f"Built edges in {time.perf_counter() - t_edges:.1f}s: "
             f"{combined.height} combined ({n_edges})")
    else:
        edges_path = None
        _log("No edges built (insufficient citation data)")

    # ── Step 4: Run landscape pipeline (optional) ──
    landscape_dir = None
    if config.run_landscape and edges_path is not None:
        try:
            from ..landscape import run_landscape, LandscapeConfig

            landscape_dir = output_dir / "landscape"
            landscape_dir.mkdir(exist_ok=True)

            _log("Running landscape pipeline...")
            run_landscape(
                edge_path=edges_path,
                abstract_path=abstracts_path,
                output_dir=landscape_dir,
                config=LandscapeConfig(
                    progress=config.progress,
                ),
            )
            _log(f"Landscape complete → {landscape_dir}")
        except ImportError as e:
            _log(f"Landscape skipped (missing dependency: {e})")
        except Exception as e:
            _log(f"Landscape failed: {e}")

    total_time = time.perf_counter() - t0
    _log(f"Pipeline complete in {total_time:.1f}s: "
         f"{len(works)} works, {sum(n_edges.values())} edges")

    return OpenAlexPipelineResult(
        n_works=len(works),
        n_edges=n_edges,
        works=works,
        abstracts_path=abstracts_path,
        edges_path=edges_path,
        landscape_dir=landscape_dir,
    )


__all__ = ["run_openalex_pipeline", "OpenAlexPipelineConfig", "OpenAlexPipelineResult"]
