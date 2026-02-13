"""Utilities to parse pipeline progress logs and inspect resolution searches."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from .logging import PROGRESS_LOG_FILE

LEVEL_PATTERN = re.compile(
    r"^(?P<level>level-\d+):\s+(?P<stage>\w+)\s+gamma=(?P<gamma>[0-9.eE+-]+)\s+->\s+(?P<count>\d+)\s+clusters(?:\s+\(quality=(?P<quality>[0-9.eE+-]+)\))?$"
)


@dataclass
class GammaEvent:
    level: str
    stage: str
    gamma: float
    cluster_count: int
    quality: Optional[float] = None


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


__all__ = ["GammaEvent", "parse_progress_log", "last_gamma_for_level"]
