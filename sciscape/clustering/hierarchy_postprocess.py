"""Compatibility import path for hierarchy oversize postprocess helpers.

New code should import from ``sciscape.clustering.hierarchy_oversize_postprocess``.
This module remains to avoid breaking older research scripts and notebooks.
"""

from __future__ import annotations

from . import hierarchy_oversize_postprocess as _impl
from .hierarchy_oversize_postprocess import *  # noqa: F401,F403
from .hierarchy_oversize_postprocess import __all__


def __getattr__(name: str):
    return getattr(_impl, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_impl)))
