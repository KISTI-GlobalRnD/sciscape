from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "research" / "consensus" / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import evaluate_leiden_basin_target_elbow_polish as elbow  # noqa: E402


def test_guarded_escalation_starts_guarded():
    policy, reason = elbow._target_selection_policy(
        path_policy=elbow.PATH_POLICY_GUARDED_ESCALATE,
        target_stage_index=1,
        parent_row={"state_support_distance_to_vanilla": 0.0},
        min_support_shift_from_vanilla=0.05,
    )

    assert policy == elbow.PATH_POLICY_GUARDED_ELBOW
    assert reason == "initial_guarded"


def test_guarded_escalation_uses_fixed_after_support_stall():
    policy, reason = elbow._target_selection_policy(
        path_policy=elbow.PATH_POLICY_GUARDED_ESCALATE,
        target_stage_index=2,
        parent_row={"state_support_distance_to_vanilla": 0.049},
        min_support_shift_from_vanilla=0.05,
    )

    assert policy == elbow.PATH_POLICY_FIXED_CAP
    assert reason == "below_support_gate"


def test_guarded_escalation_stays_guarded_after_support_gate():
    policy, reason = elbow._target_selection_policy(
        path_policy=elbow.PATH_POLICY_GUARDED_ESCALATE,
        target_stage_index=2,
        parent_row={"state_support_distance_to_vanilla": 0.051},
        min_support_shift_from_vanilla=0.05,
    )

    assert policy == elbow.PATH_POLICY_GUARDED_ELBOW
    assert reason == "support_gate_reached"


def test_guarded_backfill_shares_escalation_gate_rule():
    policy, reason = elbow._target_selection_policy(
        path_policy=elbow.PATH_POLICY_GUARDED_BACKFILL,
        target_stage_index=2,
        parent_row={"state_support_distance_to_vanilla": 0.049},
        min_support_shift_from_vanilla=0.05,
    )

    assert policy == elbow.PATH_POLICY_FIXED_CAP
    assert reason == "below_support_gate"


def test_selection_context_marks_fixed_tail_backfill_as_escalated():
    context = elbow._selection_context(
        path_policy=elbow.PATH_POLICY_GUARDED_BACKFILL,
        selection_policy=elbow.SELECTION_FIXED_TAIL_BACKFILL,
        escalation_reason="below_support_gate_backfill",
        target_stage_index=2,
        selected=[4, 2],
    )

    assert context["escalated_to_fixed"] is True
    assert context["selected_node_ids"] == "2,4"


def test_rank_and_filter_prefix_rows_preserves_original_selected_rank():
    rows = pd.DataFrame(
        [
            {"pair_id": "p1", "prefix_unit_ids": "a"},
            {"pair_id": "p1", "prefix_unit_ids": "b"},
            {"pair_id": "p1", "prefix_unit_ids": "c"},
            {"pair_id": "p2", "prefix_unit_ids": "d"},
            {"pair_id": "p2", "prefix_unit_ids": "e"},
        ]
    )

    filtered = elbow._rank_and_filter_prefix_rows(
        rows,
        selected_prefix_ranks=(2,),
    )

    assert filtered["prefix_unit_ids"].tolist() == ["b", "e"]
    assert filtered["selected_prefix_rank"].tolist() == [2, 2]
