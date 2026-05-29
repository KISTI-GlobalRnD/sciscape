"""Tests for Dongdaemun safe-fast interpretability helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "research"
    / "consensus"
    / "scripts"
    / "dongdaemun_hierarchy"
    / "refinement_runs"
    / "run_dongdaemun_safe_fast_interpretability.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "run_dongdaemun_safe_fast_interpretability_for_test",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_tokens_drop_markup_and_common_words():
    module = _load_module()

    tokens = module._tokens("<i>Transdermal</i> drug delivery and skin penetration")

    assert "transdermal" in tokens
    assert "delivery" in tokens
    assert "skin" in tokens
    assert "and" not in tokens


def test_overlap_rows_reports_fragmentation():
    module = _load_module()
    standard = np.asarray([0, 0, 0, 0, 1], dtype=np.uint64)
    safe = np.asarray([2, 2, 3, 3, 4], dtype=np.uint64)
    weights = np.ones(5, dtype=np.float64)

    rows = module._overlap_rows(
        standard_membership=standard,
        safe_membership=safe,
        node_weights=weights,
    )
    by_standard = {row["standard_cluster"]: row for row in rows}

    assert by_standard[0]["largest_overlap_fraction"] == 0.5
    assert by_standard[0]["fragment_count_ge_1pct"] == 2
    assert by_standard[0]["children"][0]["overlap_nodes"] == 2
