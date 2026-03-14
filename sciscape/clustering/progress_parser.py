"""Utilities to parse pipeline progress logs and inspect resolution searches."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Sequence

from .logging import PROGRESS_LOG_FILE

LEVEL_PATTERN = re.compile(
    r"^(?P<level>level-\d+):\s+(?P<stage>.+?)\s+gamma=(?P<gamma>[0-9.eE+-]+)\s+->\s+(?P<count>\d+)\s+clusters(?:\s+\(quality=(?P<quality>[0-9.eE+-]+)\))?$"
)


@dataclass
class GammaEvent:
    level: str
    stage: str
    gamma: float
    cluster_count: int
    quality: Optional[float] = None
    timestamp: Optional[str] = None


@dataclass
class EtaEstimate:
    observed_events: int
    remaining_events: int
    elapsed_seconds: float
    rate_events_per_minute: float
    eta_seconds: float
    eta_timestamp: str


def _parse_timestamp(raw: Optional[str]) -> Optional[datetime]:
    if not raw:
        return None
    try:
        normalized = raw.replace("Z", "+00:00")
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def parse_progress_log(
    path: Path = PROGRESS_LOG_FILE,
    *,
    level: Optional[str] = None,
) -> List[GammaEvent]:
    """Parse the progress log and return gamma evaluation events."""

    events: List[GammaEvent] = []
    if not path.exists():
        return events

    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue

            message = payload.get("message", "")
            match = LEVEL_PATTERN.match(message)
            if not match:
                continue

            lvl = match.group("level")
            if level and lvl != level:
                continue

            quality = match.group("quality")
            events.append(
                GammaEvent(
                    level=lvl,
                    stage=match.group("stage"),
                    gamma=float(match.group("gamma")),
                    cluster_count=int(match.group("count")),
                    quality=float(quality) if quality is not None else None,
                    timestamp=str(payload.get("timestamp")) if payload.get("timestamp") else None,
                )
            )

    return events


def last_gamma_for_level(
    level: str,
    *,
    path: Path = PROGRESS_LOG_FILE,
) -> Optional[GammaEvent]:
    """Return the most recent gamma event for the given level, if any."""

    for event in reversed(parse_progress_log(path=path, level=level)):
        return event
    return None


def estimate_eta(
    events: Sequence[GammaEvent],
    *,
    total_expected_events: Optional[int] = None,
    remaining_events: Optional[int] = None,
) -> Optional[EtaEstimate]:
    if len(events) < 2:
        return None

    if remaining_events is None:
        if total_expected_events is None:
            return None
        remaining = max(int(total_expected_events) - int(len(events)), 0)
    else:
        remaining = max(int(remaining_events), 0)

    first_ts = _parse_timestamp(events[0].timestamp)
    last_ts = _parse_timestamp(events[-1].timestamp)
    if first_ts is None or last_ts is None:
        return None

    elapsed = (last_ts - first_ts).total_seconds()
    if elapsed <= 0:
        return None

    observed_intervals = len(events) - 1
    rate_per_sec = observed_intervals / elapsed
    if rate_per_sec <= 0:
        return None

    eta_seconds = 0.0 if remaining == 0 else float(remaining / rate_per_sec)
    eta_ts = last_ts + timedelta(seconds=eta_seconds)
    return EtaEstimate(
        observed_events=int(len(events)),
        remaining_events=int(remaining),
        elapsed_seconds=float(elapsed),
        rate_events_per_minute=float(rate_per_sec * 60.0),
        eta_seconds=float(eta_seconds),
        eta_timestamp=eta_ts.isoformat(),
    )


def estimate_eta_from_progress_log(
    *,
    path: Path = PROGRESS_LOG_FILE,
    level: Optional[str] = None,
    total_expected_events: Optional[int] = None,
    remaining_events: Optional[int] = None,
) -> Optional[EtaEstimate]:
    events = parse_progress_log(path=path, level=level)
    return estimate_eta(
        events,
        total_expected_events=total_expected_events,
        remaining_events=remaining_events,
    )


__all__ = [
    "EtaEstimate",
    "GammaEvent",
    "estimate_eta",
    "estimate_eta_from_progress_log",
    "parse_progress_log",
    "last_gamma_for_level",
]
