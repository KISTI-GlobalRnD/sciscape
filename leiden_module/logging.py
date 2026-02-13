"""Utilities for recording resolution search history."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


DEFAULT_LOG_FILE = Path("resolution_history.jsonl")
PROGRESS_LOG_FILE = Path("pipeline_progress.log")


@dataclass
class LogMetadata:
    source: str
    node_count: int
    edge_count: int
    timestamp: str


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def write_history_entry(
    path: Path,
    *,
    metadata: LogMetadata,
    levels: Iterable[str],
    resolutions: Dict[str, float],
    cluster_counts: Dict[str, int],
    coverage: Optional[float] = None,
    qualities: Optional[Dict[str, float]] = None,
) -> None:
    import json

    entry: Dict[str, Any] = {
        "timestamp": metadata.timestamp,
        "source": metadata.source,
        "node_count": metadata.node_count,
        "edge_count": metadata.edge_count,
        "levels": list(levels),
        "resolutions": {level: float(gamma) for level, gamma in resolutions.items()},
        "cluster_counts": {level: int(count) for level, count in cluster_counts.items()},
    }
    if coverage is not None:
        entry["giant_component_coverage"] = coverage
    if qualities:
        entry["qualities"] = {level: float(val) for level, val in qualities.items()}

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def write_progress_event(
    message: str,
    *,
    step: Optional[str] = None,
    path: Path = PROGRESS_LOG_FILE,
) -> None:
    import json

    entry = {
        "timestamp": _now_iso(),
        "step": step or "",
        "message": message,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


__all__ = [
    "LogMetadata",
    "DEFAULT_LOG_FILE",
    "PROGRESS_LOG_FILE",
    "write_history_entry",
    "write_progress_event",
    "_now_iso",
]
