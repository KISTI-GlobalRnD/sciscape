"""Tests for Dongdaemun resolution perturbation robustness summaries."""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "research"
    / "consensus"
    / "scripts"
    / "dongdaemun_hierarchy"
    / "trajectory_analysis"
    / "analyze_dongdaemun_resolution_robustness.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "analyze_dongdaemun_resolution_robustness_for_test",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_resolution_robustness_groups_down_and_up_perturbations(tmp_path):
    module = _load_module()
    events = [
        _candidate("run-a", 0.98, "qpos_spos", "selected_by_policy", True, 2.0),
        _candidate("run-a", 0.98, "qneg_spos", "policy_rejected", True, -1.0),
        _candidate("run-a", 1.02, "qpos_sneg", "invalid", False, 3.0),
        _candidate("run-a", 1.0, "qpos_spos", "selected_by_policy", True, 9.0),
    ]

    rows = module.build_resolution_robustness_rows(events)

    assert len(rows) == 2
    down = next(row for row in rows if row["direction"] == "down")
    up = next(row for row in rows if row["direction"] == "up")
    assert down["gamma_multiplier"] == 0.98
    assert down["n_candidates"] == 2
    assert down["selected_count"] == 1
    assert down["qpos_spos_count"] == 1
    assert up["gamma_multiplier"] == 1.02
    assert up["valid_count"] == 0
    assert up["qpos_sneg_count"] == 1


def test_resolution_robustness_writes_report_and_csv(tmp_path):
    module = _load_module()
    paths = module.analyze_dongdaemun_resolution_robustness_for_test(
        candidate_events=[
            _candidate("run-a", 0.95, "qpos_spos", "selected_by_policy", True, 1.0),
            _candidate("run-a", 1.05, "qneg_sneg", "policy_rejected", True, -2.0),
        ],
        output_dir=tmp_path,
    )

    assert paths["report"].exists()
    assert "Down selected candidates: 1" in paths["report"].read_text()
    with paths["summary_csv"].open(newline="") as fh:
        assert list(csv.DictReader(fh).fieldnames or []) == module.SUMMARY_COLUMNS


def _candidate(
    run_id: str,
    gamma_multiplier: float,
    quadrant: str,
    decision: str,
    valid: bool,
    delta_q: float,
) -> dict[str, object]:
    return {
        "event": "candidate_profile",
        "run_id": run_id,
        "source": "high_gamma",
        "gamma_multiplier": gamma_multiplier,
        "quadrant": quadrant,
        "decision": decision,
        "valid": valid,
        "quality_passes": delta_q > 0.0,
        "candidate_delta_q": delta_q,
        "largest_child_fraction": 0.4,
        "largest_child_fraction_improvement": 0.1,
    }
