"""End-to-end OpenAlex query → analysis pipeline.

    query → fetch works → build edges → landscape (cluster + keywords + report)

Designed as a library function callable from CLI, web, or notebooks.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

import polars as pl

from .client import (
    BACKOFF_BASE,
    BACKOFF_MAX,
    MAX_RETRIES,
    REQUEST_TIMEOUT,
    OpenAlexClient,
    WorkRecord,
)
from .edges import build_citation_edges, works_to_abstracts

log = logging.getLogger(__name__)


def _is_control_flow_exception(exc: Exception) -> bool:
    """Return true for caller-raised control-flow exceptions.

    The web app passes a cancellation checkpoint/progress callback that raises
    its own JobCancelled type. OpenAlex keeps landscape/report failures
    non-fatal, but cancellation must not be swallowed by those broad guards.
    """
    return exc.__class__.__name__ in {"JobCancelled", "OpenAlexPipelineCancelled"}


@dataclass
class OpenAlexPipelineConfig:
    """Configuration for the query → analyze pipeline."""

    # ── Query ──
    query: str = ""
    filters: Dict[str, str] = field(default_factory=dict)
    max_works: int = 10000

    # ── API ──
    email: str | None = None
    request_timeout: float = REQUEST_TIMEOUT
    max_retries: int = MAX_RETRIES
    backoff_base: float = BACKOFF_BASE
    backoff_max: float = BACKOFF_MAX
    api_attempt_budget: int | None = None
    retry_wait_budget_seconds: float | None = None
    interruptible_requests: bool = False
    request_poll_interval: float = 0.25

    # ── Edge building ──
    edge_types: Sequence[str] = ("dc", "bc")
    normalization: str = "fractional"
    bc_topk: int = 50
    min_shared_refs: int = 1

    # ── Output ──
    output_dir: Path = Path("workspace/openalex_output")

    # ── Landscape (forwarded) ──
    run_landscape: bool = True
    combine_strategy: str = "consensus"
    combine_top_k: int | str = "auto"
    auto_gamma: bool = True
    auto_gamma_target: float = 3.0

    # ── Callbacks ──
    progress: Callable[[str], None] | None = None
    checkpoint: Callable[[], None] | None = None
    api_telemetry: Callable[[dict[str, Any]], None] | None = None


@dataclass
class OpenAlexPipelineResult:
    """Result of the pipeline."""
    n_works: int
    n_edges: Dict[str, int]
    works: list  # List[WorkRecord]
    abstracts_path: Path | None
    edges_path: Path | None
    landscape_dir: Path | None
    api_telemetry: dict[str, Any] | None = None


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

    def _checkpoint() -> None:
        if config.checkpoint is not None:
            config.checkpoint()

    latest_api_telemetry: dict[str, Any] | None = None

    def _api_telemetry(snapshot: dict[str, Any]) -> None:
        nonlocal latest_api_telemetry
        latest_api_telemetry = dict(snapshot)
        if config.api_telemetry is not None:
            config.api_telemetry(dict(snapshot))

    # ── Step 1: Query OpenAlex ──
    _checkpoint()
    _log(f"Querying OpenAlex: '{config.query}' (max {config.max_works} works)")
    client = OpenAlexClient(
        email=config.email,
        progress=config.progress,
        checkpoint=_checkpoint,
        request_timeout=config.request_timeout,
        max_retries=config.max_retries,
        backoff_base=config.backoff_base,
        backoff_max=config.backoff_max,
        api_attempt_budget=config.api_attempt_budget,
        retry_wait_budget_seconds=config.retry_wait_budget_seconds,
        interruptible_requests=config.interruptible_requests,
        request_poll_interval=config.request_poll_interval,
        telemetry=_api_telemetry,
    )
    works = client.search_works(
        config.query,
        filters=config.filters,
        max_results=config.max_works,
    )
    _checkpoint()
    _log(f"Fetched {len(works)} works in {time.perf_counter() - t0:.1f}s")

    if not works:
        _log("No works found. Aborting.")
        return OpenAlexPipelineResult(
            n_works=0, n_edges={}, works=[],
            abstracts_path=None, edges_path=None, landscape_dir=None,
            api_telemetry=latest_api_telemetry,
        )

    # ── Step 2: Save abstracts ──
    _checkpoint()
    abstracts_df = works_to_abstracts(works)
    abstracts_path = output_dir / "abstracts.parquet"
    abstracts_df.write_parquet(abstracts_path)
    _checkpoint()
    _log(f"Saved {abstracts_df.height} abstracts → {abstracts_path}")

    # ── Step 3: Build citation edges ──
    _checkpoint()
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

    # Save per-layer edge files for network visualization
    n_edges = {}
    for etype, df in edge_tables.items():
        if df.height > 0:
            layer_path = output_dir / f"edges_{etype}.parquet"
            df.write_parquet(layer_path)

    _checkpoint()
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
        _checkpoint()
        try:
            from ..landscape import run_landscape, LandscapeConfig

            landscape_dir = output_dir / "landscape"
            landscape_dir.mkdir(exist_ok=True)

            # Use per-layer edges for multi-layer combination
            layer_paths = {}
            for etype in config.edge_types:
                lp = output_dir / f"edges_{etype}.parquet"
                if lp.exists():
                    layer_paths[etype] = lp

            _log("Running landscape pipeline...")
            lcfg = LandscapeConfig(progress=config.progress)
            # Always enable auto-gamma + consensus for multi-layer
            lcfg.auto_gamma = config.auto_gamma
            lcfg.auto_gamma_target = config.auto_gamma_target
            if len(layer_paths) >= 2:
                lcfg.layer_paths = layer_paths
                lcfg.combine_strategy = config.combine_strategy
                lcfg.combine_top_k = config.combine_top_k

            run_landscape(
                edge_path=edges_path,
                abstract_path=abstracts_path,
                output_dir=landscape_dir,
                config=lcfg,
            )
            _log(f"Landscape complete → {landscape_dir}")
        except ImportError as e:
            _log(f"Landscape skipped (missing dependency: {e})")
        except Exception as e:
            if _is_control_flow_exception(e):
                raise
            _log(f"Landscape failed: {e}")
        _checkpoint()

    _checkpoint()
    try:
        from ..artifacts import write_result_manifest

        write_result_manifest(
            output_dir,
            mode="live_query",
            source_overrides={
                "source_type": "openalex_query",
                "query": config.query,
                "filters": dict(config.filters),
                "record_count": len(works),
                "api_telemetry": latest_api_telemetry,
            },
        )
        _log(f"Saved result manifest → {output_dir / 'result_manifest.json'}")
    except Exception as e:
        if _is_control_flow_exception(e):
            raise
        _log(f"Result manifest skipped: {e}")

    _checkpoint()
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
        api_telemetry=latest_api_telemetry,
    )


__all__ = ["run_openalex_pipeline", "OpenAlexPipelineConfig", "OpenAlexPipelineResult"]
