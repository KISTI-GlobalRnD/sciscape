from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "research"
    / "consensus"
    / "scripts"
    / "profile_leiden_basin_ordered_flips_batch.py"
)


def _load_module():
    script_dir = SCRIPT_PATH.parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    spec = importlib.util.spec_from_file_location(
        "profile_leiden_basin_ordered_flips_batch_for_test",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_selected_target_rows_filters_priority_and_pair_ids():
    module = _load_module()
    rows = pd.DataFrame(
        [
            {"pair_id": "a", "recommended_priority": 1},
            {"pair_id": "b", "recommended_priority": 3},
            {"pair_id": "c", "recommended_priority": 5},
        ]
    )

    selected = module.selected_target_rows(
        rows,
        max_priority=4,
        pair_ids=("b",),
    )

    assert selected["pair_id"].tolist() == ["b"]


def test_first_step_rows_selects_policy_specific_frontier():
    module = _load_module()
    frontier = pd.DataFrame(
        [
            {
                "scoring_policy": "q_first",
                "step_index": 1,
                "unit_id": "q",
                "unit_node_count": 1,
                "delta_q_immediate": 1.0,
                "incremental_progress_fraction": 0.1,
                "raw_barrier_if_chosen": 0.0,
                "candidate_label": 10,
                "vanilla_label": 20,
                "candidate_label_closure_extra_count": 0,
                "q_first_score": 1.0,
                "progress_first_score": 0.1,
                "balanced_score": 0.1,
            },
            {
                "scoring_policy": "progress_first",
                "step_index": 1,
                "unit_id": "p",
                "unit_node_count": 5,
                "delta_q_immediate": -2.0,
                "incremental_progress_fraction": 0.5,
                "raw_barrier_if_chosen": 2.0,
                "candidate_label": 11,
                "vanilla_label": 21,
                "candidate_label_closure_extra_count": 3,
                "q_first_score": -2.0,
                "progress_first_score": 0.5,
                "balanced_score": 0.25,
            },
        ]
    )

    selected = module._first_step_rows(frontier)

    assert selected.set_index("scoring_policy").loc["q_first", "first_unit_id"] == "q"
    assert (
        selected.set_index("scoring_policy").loc[
            "progress_first",
            "first_unit_id",
        ]
        == "p"
    )


def test_summarize_profile_dir_returns_case_and_policy_rows(tmp_path: Path):
    module = _load_module()
    profile_dir = tmp_path / "p"
    profile_dir.mkdir()
    summary = {
        "pair_id": "p",
        "candidate_index": 2,
        "vanilla_seed": 11,
        "vanilla_randomness": 0.0,
        "vanilla_minus_candidate_quality": 1.2,
        "candidate_support_size": 3,
        "vanilla_support_size": 10,
        "v_only_support_size": 7,
        "unit_count": 2,
        "frontier_rows": 2,
        "beam_rows": 2,
    }
    (profile_dir / module.SINGLE_SUMMARY_FILENAME).write_text(json.dumps(summary))
    pd.DataFrame(
        [
            {
                "scoring_policy": "q_first",
                "step_index": 1,
                "unit_id": "q",
                "unit_node_count": 1,
                "delta_q_immediate": 1.0,
                "incremental_progress_fraction": 0.1,
                "raw_barrier_if_chosen": 0.0,
                "candidate_label": 10,
                "vanilla_label": 20,
                "candidate_label_closure_extra_count": 0,
                "q_first_score": 1.0,
                "progress_first_score": 0.1,
                "balanced_score": 0.1,
            },
            {
                "scoring_policy": "progress_first",
                "step_index": 1,
                "unit_id": "p",
                "unit_node_count": 3,
                "delta_q_immediate": -2.0,
                "incremental_progress_fraction": 0.5,
                "raw_barrier_if_chosen": 2.0,
                "candidate_label": 11,
                "vanilla_label": 21,
                "candidate_label_closure_extra_count": 3,
                "q_first_score": -2.0,
                "progress_first_score": 0.5,
                "balanced_score": 0.25,
            },
        ]
    ).to_csv(profile_dir / module.FRONTIER_ROWS_FILENAME, index=False)
    pd.DataFrame(
        [
            {
                "scoring_policy": "q_first",
                "step_index": 1,
                "raw_barrier_so_far": 0.0,
                "result_support_distance_to_candidate": 0.9,
                "delta_q_vs_start": 1.0,
                "flipped_node_count": 1,
            },
            {
                "scoring_policy": "progress_first",
                "step_index": 1,
                "raw_barrier_so_far": 2.0,
                "result_support_distance_to_candidate": 0.5,
                "delta_q_vs_start": -2.0,
                "flipped_node_count": 3,
            },
        ]
    ).to_csv(profile_dir / module.BEAM_ROWS_FILENAME, index=False)

    case_row, policy_rows = module.summarize_profile_dir(
        profile_dir=profile_dir,
        target_row=pd.Series(
            {
                "inspection_role": "role",
                "recommended_priority": 1,
            }
        ),
    )

    assert case_row["pair_id"] == "p"
    assert case_row["first_q_progress_same_unit"] is False
    assert len(policy_rows) == 2
