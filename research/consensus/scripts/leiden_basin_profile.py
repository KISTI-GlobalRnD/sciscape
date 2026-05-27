#!/usr/bin/env python3
"""Compatibility shim for reusable Leiden basin profiling helpers.

The implementation lives in :mod:`sciscape.clustering.leiden_basin_profile`.
Research scripts keep importing this file by basename, so this shim preserves
that path while avoiding another script-local implementation.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sciscape.clustering.leiden_basin_profile import *  # noqa: F401,F403,E402
