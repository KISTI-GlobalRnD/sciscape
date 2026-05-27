from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "sciscape"
    / "clustering"
    / "leiden_basin_profile.py"
)


class FakeGraph:
    def cpm_quality(self, membership, *, resolution):
        del resolution
        membership = np.asarray(membership, dtype=np.uint64)
        score = 0.0
        if membership[3] != np.uint64(2):
            score += 5.0
        if membership[4] != np.uint64(3):
            score -= 1.0
        if membership[5] != np.uint64(4):
            score -= 1.0
        return score


def _load_module():
    script_dir = SCRIPT_PATH.parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    spec = importlib.util.spec_from_file_location(
        "leiden_basin_profile_for_test",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_label_intersection_units_groups_v_only_nodes():
    module = _load_module()
    baseline = np.asarray([0, 0, 1, 1, 1, 1], dtype=np.uint64)
    candidate = baseline.copy()
    vanilla = np.asarray([0, 0, 1, 2, 3, 1], dtype=np.uint64)
    src = np.asarray([2, 3, 4], dtype=np.uint32)
    dst = np.asarray([3, 4, 5], dtype=np.uint32)
    weight = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)

    units, summary = module.build_label_intersection_units(
        baseline_membership=baseline,
        candidate_membership=candidate,
        vanilla_membership=vanilla,
        src=src,
        dst=dst,
        weight=weight,
        node_weights=np.ones(6, dtype=np.float64),
        context={"case": "toy"},
    )

    assert summary["v_only_support_size"] == 2
    assert units["unit_node_count"].tolist() == [1, 1]
    assert units["candidate_label"].tolist() == [1, 1]
    assert units["vanilla_label"].tolist() == [2, 3]
    assert units.loc[0, "boundary_edge_weight"] == 3.0


def test_policy_score_keeps_q_first_and_progress_first_distinct():
    module = _load_module()

    assert module.policy_score(policy="q_first", delta_q=-1.0, progress=0.9) == -1.0
    assert module.policy_score(policy="progress_first", delta_q=-1.0, progress=0.9) == 0.9
    assert module.policy_score(policy="balanced", delta_q=-2.0, progress=0.5) == 0.25


def test_ordered_flip_beam_can_choose_different_first_blocks_by_policy():
    module = _load_module()
    baseline = np.asarray([0, 0, 1, 1, 1, 1], dtype=np.uint64)
    candidate = baseline.copy()
    vanilla = np.asarray([0, 0, 1, 2, 3, 4], dtype=np.uint64)
    units = pd.DataFrame(
        [
            {
                "unit_id": "good_q",
                "unit_type": module.UNIT_TYPE_LABEL_INTERSECTION,
                "candidate_label": 1,
                "vanilla_label": 2,
                "unit_node_count": 1,
                "unit_node_weight": 1.0,
                "candidate_label_closure_node_count": 4,
                "candidate_label_closure_extra_count": 3,
                "boundary_edge_weight": 1.0,
                "incident_edge_weight": 1.0,
                "node_ids": "3",
            },
            {
                "unit_id": "good_progress",
                "unit_type": module.UNIT_TYPE_LABEL_INTERSECTION,
                "candidate_label": 1,
                "vanilla_label": 3,
                "unit_node_count": 2,
                "unit_node_weight": 2.0,
                "candidate_label_closure_node_count": 4,
                "candidate_label_closure_extra_count": 2,
                "boundary_edge_weight": 2.0,
                "incident_edge_weight": 2.0,
                "node_ids": "4,5",
            },
        ]
    )

    frontier, beam = module.run_ordered_flip_beam(
        graph=FakeGraph(),
        units=units,
        baseline_membership=baseline,
        candidate_membership=candidate,
        vanilla_membership=vanilla,
        start_quality=0.0,
        candidate_quality=0.0,
        vanilla_quality=0.0,
        sketch_nodes=np.asarray([0, 1, 2, 3, 4, 5], dtype=np.uint32),
        resolution=0.01,
        beam_width=1,
        max_steps=1,
        scoring_policies=("q_first", "progress_first"),
        context={"case": "toy"},
    )

    first_choices = beam.set_index("scoring_policy")["chosen_unit_id"].to_dict()
    assert first_choices["q_first"] == "good_q"
    assert first_choices["progress_first"] == "good_progress"
    assert len(frontier) == 4


def test_barrier_aware_prefix_annotation_marks_non_greedy_rows():
    module = _load_module()
    frontier = pd.DataFrame(
        [
            {
                "parent_state_id": "q_first:0:root",
                "scoring_policy": "q_first",
                "step_index": 1,
                "unit_id": "q",
                "unit_node_count": 1,
                "candidate_label_closure_extra_count": 0,
                "candidate_progress_fraction": 0.1,
                "incremental_progress_fraction": 0.1,
                "delta_q_immediate": 1.0,
                "raw_barrier_if_chosen": 0.0,
                "q_first_score": 1.0,
                "progress_first_score": 0.1,
                "balanced_score": 0.1,
            },
            {
                "parent_state_id": "q_first:0:root",
                "scoring_policy": "q_first",
                "step_index": 1,
                "unit_id": "hidden",
                "unit_node_count": 4,
                "candidate_label_closure_extra_count": 20,
                "candidate_progress_fraction": 0.4,
                "incremental_progress_fraction": 0.4,
                "delta_q_immediate": -2.0,
                "raw_barrier_if_chosen": 2.0,
                "q_first_score": -2.0,
                "progress_first_score": 0.4,
                "balanced_score": 0.2,
            },
        ]
    )
    beam = pd.DataFrame(
        columns=[
            "state_id",
            "selected_unit_ids",
            "selected_unit_count",
            "flipped_node_count",
        ]
    )

    annotated = module.annotate_barrier_aware_prefixes(
        frontier_rows=frontier,
        beam_rows=beam,
        v_only_support_size=10,
    )
    hidden = annotated.set_index("unit_id").loc["hidden"]

    assert hidden["prefix_unit_ids"] == "hidden"
    assert hidden["prefix_flipped_node_count_estimate"] == 4
    assert hidden["q_rank_within_parent"] == 2
    assert module.BARRIER_Q_GREEDY_MISS in hidden["greedy_failure_labels"]
    assert module.BARRIER_CLOSURE_COMPOUND_MISS in hidden["greedy_failure_labels"]
    assert module.BARRIER_POLISH_RECOVERY_MISS in hidden["greedy_failure_labels"]


def test_barrier_progress_frontier_keeps_progress_improving_rows():
    module = _load_module()
    prefixes = pd.DataFrame(
        [
            {
                "unit_id": "low",
                "peak_raw_barrier": 0.0,
                "prefix_flipped_node_count_estimate": 1,
                "support_progress_fraction": 0.1,
                "barrier_aware_score": 0.1,
            },
            {
                "unit_id": "dominated",
                "peak_raw_barrier": 1.0,
                "prefix_flipped_node_count_estimate": 2,
                "support_progress_fraction": 0.05,
                "barrier_aware_score": 0.05,
            },
            {
                "unit_id": "higher",
                "peak_raw_barrier": 2.0,
                "prefix_flipped_node_count_estimate": 4,
                "support_progress_fraction": 0.4,
                "barrier_aware_score": 0.2,
            },
        ]
    )

    selected = module.select_barrier_progress_frontier(prefixes, max_rows=10)

    assert selected["unit_id"].tolist() == ["higher", "low"]


def test_apply_prefix_units_uses_stable_fresh_labels_across_units():
    module = _load_module()
    start = np.asarray([0, 0, 1, 2, 3], dtype=np.uint64)
    donor = np.asarray([0, 0, 1, 1, 1], dtype=np.uint64)
    units = pd.DataFrame(
        [
            {"unit_id": "a", "node_ids": "3"},
            {"unit_id": "b", "node_ids": "4"},
        ]
    )

    edited, mutable = module.apply_prefix_units(
        membership=start,
        donor_membership=donor,
        units=units,
        prefix_unit_ids="a,b",
    )

    assert mutable.tolist() == [3, 4]
    assert edited[3] == edited[4]
    assert edited[3] not in set(start.tolist())


def test_compact_membership_remaps_sparse_labels():
    module = _load_module()

    compacted = module.compact_membership(np.asarray([10, 10, 42], dtype=np.uint64))

    assert compacted.tolist() == [0, 0, 1]


def test_polish_recovery_classifier_requires_quality_and_support_retention():
    module = _load_module()

    assert (
        module.classify_polish_recovery(
            raw_delta_q_vs_start=-1.0,
            polish_delta_q_vs_start=0.5,
            raw_progress_from_vanilla=0.2,
            polish_progress_from_vanilla=0.15,
            polish_support_distance_to_vanilla=0.08,
        )
        == module.POLISH_RESULT_RECOVERED_SHIFT
    )
    assert (
        module.classify_polish_recovery(
            raw_delta_q_vs_start=-1.0,
            polish_delta_q_vs_start=0.5,
            raw_progress_from_vanilla=0.2,
            polish_progress_from_vanilla=0.01,
            polish_support_distance_to_vanilla=0.0,
        )
        == module.POLISH_RESULT_RECOVERED_VANILLA_NEAR
    )
    assert (
        module.classify_polish_recovery(
            raw_delta_q_vs_start=1.0,
            polish_delta_q_vs_start=0.5,
            raw_progress_from_vanilla=0.2,
            polish_progress_from_vanilla=0.2,
            polish_support_distance_to_vanilla=0.1,
        )
        == module.POLISH_RESULT_QUALITY_LOSS
    )
