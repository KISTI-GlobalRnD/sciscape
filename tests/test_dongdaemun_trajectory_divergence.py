"""Tests for Dongdaemun trajectory divergence analysis."""

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
    / "analyze_dongdaemun_trajectory_divergence.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "analyze_dongdaemun_trajectory_divergence_for_test",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_first_divergence_phase_uses_first_membership_hash_change():
    module = _load_module()
    events = [
        _checkpoint("a", 1, 0, "after_local_move", "h0", 10.0),
        _checkpoint("b", 1, 0, "after_local_move", "h0", 10.0),
        _checkpoint("a", 1, 0, "after_refinement", "h1", 10.5),
        _checkpoint("b", 1, 0, "after_refinement", "h2", 10.4),
        _checkpoint("a", 1, 0, "final", "h1", 10.5),
        _checkpoint("b", 1, 0, "final", "h2", 10.4),
    ]

    rows = module.build_first_divergence_rows(events, [("a", "b")])

    assert rows[0]["first_divergence_phase"] == "after_refinement"
    assert rows[0]["first_divergence_iteration"] == 1
    assert rows[0]["first_divergence_depth"] == 0
    assert rows[0]["left_membership_hash"] == "h1"
    assert rows[0]["right_membership_hash"] == "h2"


def test_report_and_csv_outputs_have_required_columns(tmp_path):
    module = _load_module()
    trajectory_events = [
        _checkpoint("a", 1, 0, "after_local_move", "h0", 10.0),
        _checkpoint("b", 1, 0, "after_local_move", "h9", 9.0),
        {
            "event": "local_merge_margin_summary",
            "run_id": "a",
            "depth": 0,
            "parent_id": 3,
            "parent_visit_index": 1,
            "source": "near_tie_refinement_probe",
            "decision_count": 5,
            "low_margin_decision_count": 2,
            "changed_decision_count": 1,
            "min_margin": 0.01,
            "p10_margin": 0.01,
            "p50_margin": 0.02,
            "selected_child_count": 4,
            "largest_child_fraction": 0.4,
        },
    ]
    candidate_events = [
        {
            "event": "adaptive_probe_candidate",
            "run_id": "a",
            "source": "near_tie_refinement_probe",
            "mode": "candidate",
            "committed": True,
            "candidate_delta_q": 1.0,
            "baseline_candidate_delta_q": 0.1,
            "gain_vs_baseline": 0.9,
            "near_tie_low_margin_decision_count": 2,
            "near_tie_changed_decision_count": 1,
        },
        {
            "event": "adaptive_local_shake_candidate",
            "run_id": "a",
            "depth": 0,
            "parent_id": 3,
            "parent_visit_index": 1,
            "candidate_index": 0,
            "mode": "qf_replace",
            "arm": "resolution_up",
            "candidate_delta_q": 1.1,
            "current_candidate_delta_q": 1.0,
            "gain_vs_current": 0.1,
            "distinct": True,
            "valid": True,
            "quality_passes": True,
            "commit_eligible": True,
            "commit_block_reason": "eligible",
        },
        {
            "event": "adaptive_local_shake_decision",
            "run_id": "a",
            "depth": 0,
            "parent_id": 3,
            "parent_visit_index": 1,
            "selected_candidate_index": 0,
            "committed": True,
        },
    ]

    paths = module.analyze_dongdaemun_trajectory_divergence_for_test(
        trajectory_events=trajectory_events,
        candidate_events=candidate_events,
        pairs=[("a", "b")],
        output_dir=tmp_path,
    )

    assert paths["report"].exists()
    assert "Divergent pairs: 1" in paths["report"].read_text()
    assert _fieldnames(paths["first_divergence"]) == module.FIRST_DIVERGENCE_COLUMNS
    assert _fieldnames(paths["margin_summary"]) == module.MARGIN_SUMMARY_COLUMNS
    assert _fieldnames(paths["policy_comparison"]) == module.POLICY_COMPARISON_COLUMNS
    assert _fieldnames(paths["local_shake"]) == module.LOCAL_SHAKE_COLUMNS
    assert "Local-shake commits: 1" in paths["report"].read_text()


def _checkpoint(
    run_id: str,
    iteration: int,
    depth: int,
    phase: str,
    membership_hash: str,
    quality: float,
) -> dict[str, object]:
    return {
        "event": "phase_checkpoint",
        "run_id": run_id,
        "iteration": iteration,
        "depth": depth,
        "phase": phase,
        "membership_hash": membership_hash,
        "n_clusters": 3,
        "quality": quality,
    }


def _fieldnames(path: Path) -> list[str]:
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader.fieldnames or [])
