"""Compatibility shim for SciScape.

Prefer importing from `sciscape`:
  - `sciscape.clustering`
  - `sciscape.keyword_extraction`
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from sciscape import __version__

_SUBMODULES = ("clustering", "keyword_extraction")

__all__ = ["__version__", *_SUBMODULES]


def __getattr__(name: str) -> Any:  # pragma: no cover
    if name in _SUBMODULES:
        return import_module(f"sciscape.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

