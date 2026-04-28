"""Shared helpers for web API endpoints.

Reduces duplication of job-loading, file-finding, and error-checking
patterns across route modules.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Tuple

from fastapi import HTTPException

log = logging.getLogger(__name__)


def require_done_job(jobs, job_id: str) -> Tuple[dict, dict]:
    """Load a job and verify it's complete.

    Returns (job_dict, result_dict).
    Raises HTTPException 404/400 on failure.
    """
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    if job["status"] != "done":
        raise HTTPException(status_code=400, detail="job not done")
    result = job.get("result", {})
    return job, result


def get_landscape_dir(result: dict) -> Path | None:
    """Extract landscape directory from job result."""
    ld = result.get("landscape_dir")
    return Path(ld) if ld else None


def find_file(directory: Path | None, pattern: str) -> Path | None:
    """Find first file matching glob pattern in directory."""
    if not directory or not directory.exists():
        return None
    for f in directory.glob(pattern):
        return f
    return None


def find_membership(result: dict) -> Path | None:
    """Find membership parquet in landscape output."""
    ld = get_landscape_dir(result)
    return find_file(ld, "**/membership*.parquet")


def find_keywords(result: dict) -> Path | None:
    """Find keywords parquet in landscape output."""
    ld = get_landscape_dir(result)
    return find_file(ld, "**/keywords*.parquet")


def get_edges_path(result: dict) -> Path | None:
    """Get edges parquet path from result."""
    ep = result.get("edges_path")
    return Path(ep) if ep and Path(ep).exists() else None


def get_abstracts_path(result: dict) -> Path | None:
    """Get abstracts parquet path from result."""
    ap = result.get("abstracts_path")
    return Path(ap) if ap and Path(ap).exists() else None


def get_cluster_column(mem_df) -> str | None:
    """Find the first cluster_ column in a membership DataFrame."""
    cols = [c for c in mem_df.columns if c.startswith("cluster_")]
    return cols[0] if cols else None
