#!/usr/bin/env python3
"""Compatibility shim for reusable Leiden basin profiling helpers.

The implementation lives in :mod:`sciscape.clustering.leiden_basin_profile`.
Research scripts keep importing this file by basename, so this shim preserves
that path while avoiding another script-local implementation.
"""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").exists()
)
SCRIPT_ROOT = REPO_ROOT / "research/consensus/scripts"
_SCRIPT_PATHS = [REPO_ROOT, SCRIPT_ROOT]
_SCRIPT_PATHS.extend(path for path in SCRIPT_ROOT.rglob("*") if path.is_dir())
for _script_path in reversed(_SCRIPT_PATHS):
    _script_path_str = str(_script_path)
    if _script_path_str not in sys.path:
        sys.path.insert(0, _script_path_str)


from sciscape.clustering.leiden_basin_profile import *  # noqa: F401,F403,E402
