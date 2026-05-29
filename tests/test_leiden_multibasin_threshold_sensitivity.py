"""Tests for Leiden coarse-basin threshold sensitivity analysis."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "research" / "consensus" / "scripts"
ANALYSIS_PATH = Path(__file__).resolve().parents[1] / "research/consensus/scripts/leiden_basin/basin_signatures/signature_detection/analyze_leiden_multibasin_threshold_sensitivity.py"


def _load_script(module_name: str):
    if str(ANALYSIS_PATH.parent) not in sys.path:
        sys.path.insert(0, str(ANALYSIS_PATH.parent))
    spec = importlib.util.spec_from_file_location(module_name, ANALYSIS_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_threshold_sensitivity_reports_coarse_count_changes():
    module = _load_script("leiden_multibasin_threshold_sensitivity_for_test")
    candidates = pd.DataFrame(
        [
            _candidate(0, signature="basin-a", sketch_membership="0;0;1;1"),
            _candidate(1, signature="basin-b", sketch_membership="0;1;1;1"),
            _candidate(2, signature="basin-c", sketch_membership="0;1;0;1"),
        ]
    )

    sensitivity = module.build_threshold_sensitivity(
        candidates,
        endpoint_taus=[0.0, 1.0],
        support_taus=[0.0, 1.0],
        material_delta_q=1.0,
        material_relative_ppm=10.0,
        iso_q_delta=10.0,
        iso_q_relative_ppm=10.0,
    )

    assert len(sensitivity) == 4
    strict = sensitivity[
        (sensitivity["endpoint_tau"] == 0.0) & (sensitivity["support_tau"] == 0.0)
    ].iloc[0]
    loose = sensitivity[
        (sensitivity["endpoint_tau"] == 1.0) & (sensitivity["support_tau"] == 1.0)
    ].iloc[0]
    assert strict["coarse_basin_count"] == 3
    assert loose["coarse_basin_count"] == 1


def _candidate(
    candidate_index: int,
    *,
    signature: str,
    sketch_membership: str,
) -> dict[str, object]:
    return {
        "candidate_eval_mode": "multifidelity_label",
        "case": "case",
        "seed": 11,
        "candidate_budget": 3,
        "max_group_candidates": 3,
        "candidate_index": candidate_index,
        "p1_delta_q": 3.0 - candidate_index,
        "p5_delta_q": 10.0 + candidate_index,
        "p5_relative_delta_q_ppm": 100.0 + candidate_index,
        "p5_basin_signature": signature,
        "p5_changed_fraction_vs_baseline": 0.001 * (candidate_index + 1),
        "p5_changed_nodes_vs_baseline": candidate_index + 1,
        "p5_basin_sketch_node_hash": "sample-hash",
        "p5_basin_sketch_baseline_membership": "0;0;1;1",
        "p5_basin_sketch_membership": sketch_membership,
    }
