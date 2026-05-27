from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "research"
    / "consensus"
    / "scripts"
    / "evaluate_leiden_basin_branch_candidate_controls.py"
)


def _load_module():
    script_dir = SCRIPT_PATH.parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    spec = importlib.util.spec_from_file_location(
        "evaluate_leiden_basin_branch_candidate_controls_for_test",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_select_branch_candidate_prefers_recovered_support_gate_score():
    module = _load_module()
    rows = pd.DataFrame(
        [
            {
                "pair_id": "p0",
                "path_final_state_id": "weak",
                "path_selection_policy": "fixed_tail_backfill",
                "path_final_support_distance_to_vanilla": 0.02,
                "path_final_target_progress_from_vanilla": 0.01,
                "path_final_target_coverage_fraction": 0.1,
                "path_final_delta_q_vs_start": 2.0,
                "path_q_wall": 0.0,
                "path_final_mutable_node_count": 10,
            },
            {
                "pair_id": "p0",
                "path_final_state_id": "good",
                "path_selection_policy": "fixed_tail_backfill",
                "path_final_support_distance_to_vanilla": 0.08,
                "path_final_target_progress_from_vanilla": 0.03,
                "path_final_target_coverage_fraction": 0.3,
                "path_final_delta_q_vs_start": 1.0,
                "path_q_wall": 1.0,
                "path_final_mutable_node_count": 20,
            },
        ]
    )

    selected = module.select_branch_candidate_row(
        rows,
        pair_id="p0",
        selection_policy="fixed_tail_backfill",
        support_gate=0.05,
    )

    assert selected["path_final_state_id"] == "good"


def test_control_summary_rows_marks_seed_control_dominance():
    module = _load_module()
    rows = pd.DataFrame(
        [
            {
                "row_type": "branch_candidate",
                "run_id": "branch",
                "quality": 10.0,
                "support_distance_to_vanilla": 0.10,
                "target_progress_from_vanilla": 0.04,
                "q_wall": 1.0,
                "mutable_node_count": 20,
            },
            {
                "row_type": "control",
                "run_id": "control_dominates",
                "quality": 11.0,
                "support_distance_to_vanilla": 0.11,
                "target_progress_from_vanilla": 0.05,
            },
        ]
    )

    summary = module.control_summary_rows(rows)

    assert summary.iloc[0]["verdict"] == "seed_control_dominates_branch"
    assert int(summary.iloc[0]["control_dominates_branch_rows"]) == 1


def test_control_summary_rows_marks_support_progress_tradeoff():
    module = _load_module()
    rows = pd.DataFrame(
        [
            {
                "row_type": "branch_candidate",
                "run_id": "branch",
                "quality": 10.5,
                "support_distance_to_vanilla": 0.12,
                "target_progress_from_vanilla": 0.05,
                "q_wall": 1.0,
                "mutable_node_count": 20,
            },
            {
                "row_type": "control",
                "run_id": "control",
                "quality": 11.0,
                "support_distance_to_vanilla": 0.08,
                "target_progress_from_vanilla": 0.03,
            },
        ]
    )

    summary = module.control_summary_rows(rows)

    assert summary.iloc[0]["verdict"] == "branch_support_progress_tradeoff"


def test_control_summary_rows_marks_unique_candidate_directed_quality_lag():
    module = _load_module()
    rows = pd.DataFrame(
        [
            {
                "row_type": "branch_candidate",
                "run_id": "branch",
                "quality": 10.0,
                "support_distance_to_vanilla": 0.10,
                "target_progress_from_vanilla": 0.04,
                "q_wall": 1.0,
                "mutable_node_count": 20,
            },
            {
                "row_type": "control",
                "run_id": "quality_control",
                "quality": 11.2,
                "support_distance_to_vanilla": 0.80,
                "target_progress_from_vanilla": -0.20,
            },
        ]
    )

    summary = module.control_summary_rows(rows, material_delta_q=1.0)

    assert summary.iloc[0]["verdict"] == "branch_unique_candidate_directed_quality_lag"
    assert int(summary.iloc[0]["candidate_directed_control_rows"]) == 0
