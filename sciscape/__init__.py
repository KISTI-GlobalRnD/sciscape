"""SciScape module: Leiden clustering + keyword extraction."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "0.2.0"
_SUBMODULES = ("clustering", "keyword_extraction", "landscape", "linkage")

__all__ = ["__version__", *_SUBMODULES]


def __getattr__(name: str) -> Any:  # pragma: no cover
    if name in _SUBMODULES:
        return import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
