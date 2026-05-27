from __future__ import annotations

import numpy as np
import pandas as pd

from sciscape.clustering import leiden_basin_search as search


def test_prefix_direct_nodes_collects_units_in_order_independent_set():
    units = pd.DataFrame(
        [
            {"unit_id": "a", "node_ids": "3,1"},
            {"unit_id": "b", "node_ids": "2,3"},
        ]
    )

    nodes = search.prefix_direct_nodes(units, "b,a")

    assert nodes.tolist() == [1, 2, 3]


def test_context_actions_select_candidate_closure_and_boundary_shell_by_pull():
    state = search.make_prefix_state(
        state_id="s0",
        prefix_rank=1,
        prefix_unit_ids="a",
        membership=np.asarray([0, 1, 2, 3, 4], dtype=np.uint64),
        quality=0.0,
        direct_nodes=np.asarray([1], dtype=np.uint32),
        mutable_nodes=np.asarray([1], dtype=np.uint32),
    )
    candidate = np.asarray([0, 7, 7, 8, 9], dtype=np.uint64)
    vanilla = np.asarray([0, 1, 2, 3, 4], dtype=np.uint64)
    src = np.asarray([1, 1, 3], dtype=np.uint32)
    dst = np.asarray([2, 3, 4], dtype=np.uint32)
    weight = np.asarray([5.0, 2.0, 9.0], dtype=np.float64)

    actions = search.build_context_actions(
        state=state,
        candidate_membership=candidate,
        vanilla_membership=vanilla,
        src=src,
        dst=dst,
        weight=weight,
        node_count=5,
        action_types=(
            search.ACTION_CANDIDATE_CLOSURE_TOPK,
            search.ACTION_BOUNDARY_SHELL_TOPK,
        ),
        context_multiplier=2.0,
        max_context_nodes=2,
    )

    by_type = {action.action_type: action for action in actions}
    assert by_type[search.ACTION_CANDIDATE_CLOSURE_TOPK].context_nodes.tolist() == [2]
    assert by_type[search.ACTION_BOUNDARY_SHELL_TOPK].context_nodes.tolist() == [2, 3]


def test_remaining_target_action_selects_uncovered_nodes_by_pull():
    state = search.make_prefix_state(
        state_id="s0",
        prefix_rank=1,
        prefix_unit_ids="a",
        membership=np.asarray([0, 1, 2, 3, 4], dtype=np.uint64),
        quality=0.0,
        direct_nodes=np.asarray([1], dtype=np.uint32),
        target_nodes=np.asarray([1, 2, 3, 4], dtype=np.uint32),
        mutable_nodes=np.asarray([1], dtype=np.uint32),
    )
    src = np.asarray([1, 1, 3], dtype=np.uint32)
    dst = np.asarray([2, 3, 4], dtype=np.uint32)
    weight = np.asarray([5.0, 2.0, 9.0], dtype=np.float64)

    actions = search.build_remaining_target_actions(
        state=state,
        src=src,
        dst=dst,
        weight=weight,
        node_count=5,
        target_action_multiplier=2.0,
        max_target_action_nodes=2,
    )

    assert len(actions) == 1
    assert actions[0].action_type == search.ACTION_REMAINING_TARGET_TOPK
    assert actions[0].action_nodes.tolist() == [2, 3]
    assert actions[0].context_nodes.tolist() == []


def test_remaining_target_pull_elbow_summary_finds_score_drop():
    state = search.make_prefix_state(
        state_id="s0",
        prefix_rank=1,
        prefix_unit_ids="a",
        membership=np.asarray([0, 1, 2, 3, 4], dtype=np.uint64),
        quality=0.0,
        direct_nodes=np.asarray([1], dtype=np.uint32),
        target_nodes=np.asarray([1, 2, 3, 4], dtype=np.uint32),
        mutable_nodes=np.asarray([1], dtype=np.uint32),
    )
    src = np.asarray([1, 1, 1], dtype=np.uint32)
    dst = np.asarray([2, 3, 4], dtype=np.uint32)
    weight = np.asarray([10.0, 8.0, 1.0], dtype=np.float64)

    frame = search.remaining_target_pull_frame(
        state=state,
        src=src,
        dst=dst,
        weight=weight,
        node_count=5,
    )
    summary = search.remaining_target_elbow_summary(frame, fixed_k=3)

    assert frame["node"].tolist() == [2, 3, 4]
    assert frame["pull_score"].tolist() == [10.0, 8.0, 1.0]
    assert summary["gap_elbow_k"] == 2
    assert summary["cumulative_elbow_k"] == 2
    assert summary["guarded_elbow_k"] == 2
    assert np.isclose(summary["guarded_elbow_pull_fraction"], 18.0 / 19.0)


def test_remaining_target_pull_elbow_summary_guards_tiny_first_gap():
    pull_frame = pd.DataFrame(
        {
            "rank": [1, 2, 3, 4],
            "node": [10, 11, 12, 13],
            "pull_score": [10.0, 5.0, 4.0, 3.0],
            "cumulative_pull": [10.0, 15.0, 19.0, 22.0],
            "cumulative_pull_fraction": [
                10.0 / 22.0,
                15.0 / 22.0,
                19.0 / 22.0,
                1.0,
            ],
            "score_fraction_of_top": [1.0, 0.5, 0.4, 0.3],
            "next_pull_score": [5.0, 4.0, 3.0, 0.0],
            "next_gap": [5.0, 1.0, 1.0, 3.0],
            "next_gap_fraction_of_top": [0.5, 0.1, 0.1, 0.3],
        }
    )

    summary = search.remaining_target_elbow_summary(
        pull_frame,
        fixed_k=4,
        cumulative_fraction=0.80,
        min_gap_fraction=0.25,
        min_guarded_pull_fraction=0.50,
    )

    assert summary["gap_elbow_k"] == 1
    assert summary["cumulative_elbow_k"] == 3
    assert summary["guarded_elbow_k"] == 3
    assert summary["guarded_elbow_reason"] == "cumulative"


def test_branching_target_growth_actions_keep_guarded_fixed_and_tail_visible():
    state = search.make_prefix_state(
        state_id="s0",
        prefix_rank=1,
        prefix_unit_ids="a",
        membership=np.asarray([0, 1, 2, 3, 4, 5], dtype=np.uint64),
        quality=0.0,
        direct_nodes=np.asarray([1], dtype=np.uint32),
        target_nodes=np.asarray([1, 2, 3, 4, 5], dtype=np.uint32),
        mutable_nodes=np.asarray([1], dtype=np.uint32),
    )
    src = np.asarray([1, 1, 1, 1], dtype=np.uint32)
    dst = np.asarray([2, 3, 4, 5], dtype=np.uint32)
    weight = np.asarray([10.0, 9.0, 2.0, 1.0], dtype=np.float64)

    actions = search.build_branching_target_growth_actions(
        state=state,
        src=src,
        dst=dst,
        weight=weight,
        node_count=6,
        target_stage_index=1,
        target_action_multiplier=4.0,
        max_target_action_nodes=4,
        selection_policies=(
            search.TARGET_SELECTION_GUARDED_ELBOW,
            search.TARGET_SELECTION_FIXED_CAP,
            search.TARGET_SELECTION_FIXED_TAIL_BACKFILL,
        ),
        cumulative_fraction=0.80,
        min_gap_fraction=0.25,
        min_guarded_pull_fraction=0.50,
    )

    by_policy = {item.selection_policy: item for item in actions}
    assert by_policy[search.TARGET_SELECTION_GUARDED_ELBOW].selected_nodes.tolist() == [2, 3]
    assert by_policy[search.TARGET_SELECTION_FIXED_CAP].selected_nodes.tolist() == [2, 3, 4, 5]
    assert by_policy[search.TARGET_SELECTION_FIXED_TAIL_BACKFILL].selected_nodes.tolist() == [4, 5]
    context = search.branch_target_action_context(
        by_policy[search.TARGET_SELECTION_FIXED_CAP]
    )
    assert context["selection_policy"] == search.TARGET_SELECTION_FIXED_CAP
    assert context["target_stage_index"] == 1
    assert context["fixed_effective_k"] == 4
    assert context["guarded_elbow_k"] == 2


def test_remaining_target_unit_action_selects_coherent_uncovered_unit():
    state = search.make_prefix_state(
        state_id="s0",
        prefix_rank=1,
        prefix_unit_ids="a",
        membership=np.asarray([0, 1, 2, 3, 4, 5], dtype=np.uint64),
        quality=0.0,
        direct_nodes=np.asarray([1], dtype=np.uint32),
        target_nodes=np.asarray([1, 2, 3, 4, 5], dtype=np.uint32),
        mutable_nodes=np.asarray([1], dtype=np.uint32),
    )
    src = np.asarray([1, 1, 1, 4], dtype=np.uint32)
    dst = np.asarray([2, 3, 5, 5], dtype=np.uint32)
    weight = np.asarray([5.0, 4.0, 0.1, 7.0], dtype=np.float64)
    target_units = pd.DataFrame(
        [
            {
                "unit_type": search.TARGET_UNIT_LABEL_INTERSECTION_BLOCK,
                "unit_id": "strong",
                "node_ids": "1,2,3",
                "unit_density": 0.7,
                "triangle_edge_fraction": 1.0,
            },
            {
                "unit_type": search.TARGET_UNIT_LABEL_INTERSECTION_BLOCK,
                "unit_id": "weak",
                "node_ids": "4,5",
                "unit_density": 1.0,
                "triangle_edge_fraction": 0.0,
            },
        ]
    )

    actions = search.build_remaining_target_unit_actions(
        state=state,
        target_unit_rows=target_units,
        src=src,
        dst=dst,
        weight=weight,
        node_count=6,
        target_unit_types=(search.TARGET_UNIT_LABEL_INTERSECTION_BLOCK,),
        max_target_unit_actions=1,
        max_target_unit_nodes=4,
    )

    assert len(actions) == 1
    assert actions[0].action_type == search.ACTION_REMAINING_TARGET_UNIT_TOPK
    assert actions[0].action_nodes.tolist() == [2, 3]
    assert "unit_id=strong" in actions[0].action_params


def test_transplant_action_nodes_reuses_reference_donor_label_mapping():
    membership = np.asarray([10, 20, 30, 40], dtype=np.uint64)
    donor = np.asarray([7, 7, 8, 8], dtype=np.uint64)

    out = search.transplant_action_nodes(
        membership=membership,
        donor_membership=donor,
        action_nodes=np.asarray([1, 2], dtype=np.uint32),
        reference_nodes=np.asarray([0], dtype=np.uint32),
    )

    assert int(out[1]) == 10
    assert int(out[2]) == 41
    assert int(out[0]) == 10
    assert int(out[3]) == 40


def test_target_edge_support_counts_common_target_neighbors():
    src = np.asarray([0, 1, 0, 3], dtype=np.uint32)
    dst = np.asarray([1, 2, 2, 4], dtype=np.uint32)
    weight = np.ones(4, dtype=np.float64)

    rows = search.target_edge_support_rows(
        src=src,
        dst=dst,
        weight=weight,
        target_nodes=np.asarray([0, 1, 2, 3, 4], dtype=np.uint32),
    )

    support = {
        (int(row["src"]), int(row["dst"])): int(row["edge_support"])
        for _, row in rows.iterrows()
    }
    assert support[(0, 1)] == 1
    assert support[(0, 2)] == 1
    assert support[(1, 2)] == 1
    assert support[(3, 4)] == 0


def test_build_target_unit_rows_emits_connected_and_triangle_units():
    src = np.asarray([0, 1, 0, 3], dtype=np.uint32)
    dst = np.asarray([1, 2, 2, 4], dtype=np.uint32)
    weight = np.ones(4, dtype=np.float64)
    baseline = np.asarray([0, 0, 0, 1, 1], dtype=np.uint64)
    candidate = np.asarray([2, 2, 2, 3, 3], dtype=np.uint64)
    vanilla = np.asarray([4, 4, 5, 6, 6], dtype=np.uint64)

    rows = search.build_target_unit_rows(
        target_nodes=np.asarray([0, 1, 2, 3, 4], dtype=np.uint32),
        candidate_support_nodes=np.asarray([0], dtype=np.uint32),
        baseline_membership=baseline,
        candidate_membership=candidate,
        vanilla_membership=vanilla,
        src=src,
        dst=dst,
        weight=weight,
        node_count=5,
        unit_types=(
            search.TARGET_UNIT_LABEL_INTERSECTION_BLOCK,
            search.TARGET_UNIT_CONNECTED_COMPONENT,
            search.TARGET_UNIT_TRIANGLE_SUPPORTED_COMPONENT,
        ),
        triangle_support_min=1,
    )

    by_type = {
        unit_type: group.sort_values("unit_node_count", ascending=False)
        for unit_type, group in rows.groupby("unit_type")
    }
    assert by_type[search.TARGET_UNIT_CONNECTED_COMPONENT].iloc[0]["node_ids"] == "0,1,2"
    assert by_type[search.TARGET_UNIT_CONNECTED_COMPONENT].iloc[1]["node_ids"] == "3,4"
    triangle_nodes = by_type[search.TARGET_UNIT_TRIANGLE_SUPPORTED_COMPONENT][
        "node_ids"
    ].tolist()
    assert "0,1,2" in triangle_nodes
    assert "3" in triangle_nodes
    assert "4" in triangle_nodes
    label_rows = by_type[search.TARGET_UNIT_LABEL_INTERSECTION_BLOCK]
    assert int(label_rows["baseline_label_count"].max()) == 1
    assert float(rows["triangle_edge_fraction"].max()) == 1.0


def test_classify_search_state_requires_q_and_support_shift():
    assert (
        search.classify_search_state(
            delta_q_vs_start=0.1,
            candidate_progress_from_vanilla=0.2,
            support_distance_to_vanilla=0.08,
        )
        == search.SEARCH_LABEL_SUPPORT_SHIFT_Q_RECOVERED
    )
    assert (
        search.classify_search_state(
            delta_q_vs_start=0.1,
            candidate_progress_from_vanilla=0.2,
            support_distance_to_vanilla=0.01,
        )
        == search.SEARCH_LABEL_VANILLA_COLLAPSE
    )
    assert (
        search.classify_search_state(
            delta_q_vs_start=-0.2,
            candidate_progress_from_vanilla=0.2,
            support_distance_to_vanilla=0.08,
        )
        == search.SEARCH_LABEL_QUALITY_LOSS
    )


def test_classify_reachability_state_ignores_quality_gates():
    assert (
        search.classify_reachability_state(
            target_progress_from_vanilla=-0.5,
            support_distance_to_vanilla=0.08,
            target_coverage_fraction=0.0,
        )
        == search.REACHABILITY_LABEL_SUPPORT_GATE_REACHED
    )
    assert (
        search.classify_reachability_state(
            target_progress_from_vanilla=0.02,
            support_distance_to_vanilla=0.01,
            target_coverage_fraction=0.0,
        )
        == search.REACHABILITY_LABEL_TARGET_PROGRESS
    )
    assert (
        search.classify_reachability_state(
            target_progress_from_vanilla=0.0,
            support_distance_to_vanilla=0.01,
            target_coverage_fraction=0.0,
        )
        == search.REACHABILITY_LABEL_SOURCE_ESCAPE
    )
    assert (
        search.classify_reachability_state(
            target_progress_from_vanilla=0.0,
            support_distance_to_vanilla=0.0,
            target_coverage_fraction=0.3,
        )
        == search.REACHABILITY_LABEL_COVERAGE_ONLY
    )
    assert (
        search.classify_reachability_state(
            target_progress_from_vanilla=0.0,
            support_distance_to_vanilla=0.0,
            target_coverage_fraction=0.0,
        )
        == search.REACHABILITY_LABEL_STALLED
    )


def test_state_distance_combines_support_and_endpoint_terms():
    assert np.isclose(
        search.state_distance(
            support_distance_value=0.5,
            endpoint_distance_value=0.2,
            support_weight=1.0,
            endpoint_weight=0.25,
        ),
        0.55,
    )


def test_search_state_metrics_reports_target_action_coverage():
    state = search.make_prefix_state(
        state_id="s0",
        prefix_rank=1,
        prefix_unit_ids="a",
        membership=np.asarray([5, 3, 1, 4, 2], dtype=np.uint64),
        quality=1.0,
        direct_nodes=np.asarray([1, 3, 4], dtype=np.uint32),
        target_nodes=np.asarray([1, 2, 3], dtype=np.uint32),
        mutable_nodes=np.asarray([1, 3, 4], dtype=np.uint32),
    )

    metrics = search.search_state_metrics(
        state=state,
        baseline_membership=np.asarray([0, 0, 1, 1, 2], dtype=np.uint64),
        candidate_membership=np.asarray([0, 3, 1, 4, 2], dtype=np.uint64),
        vanilla_membership=np.asarray([5, 0, 1, 4, 2], dtype=np.uint64),
        sketch_nodes=np.asarray([0, 1, 2, 3, 4], dtype=np.uint32),
        start_quality=0.0,
        candidate_quality=0.0,
        vanilla_quality=0.0,
        vanilla_support_distance_to_candidate=0.5,
    )

    assert metrics["target_node_count"] == 3
    assert metrics["action_node_count"] == 3
    assert metrics["action_target_node_count"] == 2
    assert metrics["action_off_target_node_count"] == 1
    assert metrics["covered_target_count"] == 2
    assert metrics["remaining_target_count"] == 1
    assert np.isclose(metrics["target_coverage_fraction"], 2.0 / 3.0)


def test_pathway_marginal_metrics_compares_parent_state():
    parent = {
        "state_target_distance": 0.8,
        "state_q_debt_vs_start": 0.2,
        "mutable_node_count": 10,
        "covered_target_count": 4,
    }
    row = {
        "state_target_distance": 0.7,
        "state_target_progress_from_vanilla": 0.0,
        "state_q_debt_vs_start": 0.35,
        "mutable_node_count": 13,
        "covered_target_count": 5,
    }

    metrics = search.pathway_marginal_metrics(row, parent_row=parent)

    assert np.isclose(metrics["marginal_target_distance_reduction"], 0.1)
    assert np.isclose(metrics["marginal_q_debt"], 0.15)
    assert metrics["marginal_mutable_node_count"] == 3
    assert metrics["marginal_covered_target_count"] == 1
    assert np.isclose(metrics["marginal_cost_per_target_node"], 3.0)


def test_select_search_beam_honors_state_greedy_policy():
    states = [
        search.make_prefix_state(
            state_id="quality",
            prefix_rank=1,
            prefix_unit_ids="a",
            membership=np.asarray([0, 1], dtype=np.uint64),
            quality=0.0,
            direct_nodes=np.asarray([0], dtype=np.uint32),
            mutable_nodes=np.asarray([0], dtype=np.uint32),
        ),
        search.make_prefix_state(
            state_id="state",
            prefix_rank=2,
            prefix_unit_ids="b",
            membership=np.asarray([0, 1], dtype=np.uint64),
            quality=0.0,
            direct_nodes=np.asarray([1], dtype=np.uint32),
            mutable_nodes=np.asarray([1], dtype=np.uint32),
        ),
    ]
    rows = pd.DataFrame(
        [
            {
                "state_id": "quality",
                "state_greedy_score": 0.1,
                "quality_search_score": 10.0,
                "state_target_progress_from_vanilla": 0.1,
                "state_support_distance_to_vanilla": 0.1,
                "state_delta_q_vs_start": 10.0,
                "mutable_node_count": 1,
            },
            {
                "state_id": "state",
                "state_greedy_score": 1.0,
                "quality_search_score": 0.0,
                "state_target_progress_from_vanilla": 0.8,
                "state_support_distance_to_vanilla": 0.5,
                "state_delta_q_vs_start": 0.0,
                "mutable_node_count": 1,
            },
        ]
    )

    selected = search.select_search_beam(
        states,
        rows,
        beam_width=1,
        search_policy=search.SEARCH_POLICY_STATE_GREEDY,
    )

    assert selected[0].state_id == "state"


def test_select_search_beam_reachability_first_keeps_bad_q_source_escape():
    states = [
        search.make_prefix_state(
            state_id="q_good",
            prefix_rank=1,
            prefix_unit_ids="a",
            membership=np.asarray([0, 1], dtype=np.uint64),
            quality=0.0,
            direct_nodes=np.asarray([0], dtype=np.uint32),
            mutable_nodes=np.asarray([0], dtype=np.uint32),
        ),
        search.make_prefix_state(
            state_id="escape",
            prefix_rank=2,
            prefix_unit_ids="b",
            membership=np.asarray([0, 1], dtype=np.uint64),
            quality=0.0,
            direct_nodes=np.asarray([1], dtype=np.uint32),
            mutable_nodes=np.asarray([1], dtype=np.uint32),
        ),
    ]
    rows = pd.DataFrame(
        [
            {
                "state_id": "q_good",
                "state_greedy_score": 1.0,
                "quality_search_score": 5.0,
                "reachability_search_score": 0.1,
                "state_target_progress_from_vanilla": 0.0,
                "state_support_distance_to_vanilla": 0.01,
                "target_coverage_fraction": 0.0,
                "state_delta_q_vs_start": 5.0,
                "mutable_node_count": 1,
            },
            {
                "state_id": "escape",
                "state_greedy_score": -10.0,
                "quality_search_score": -10.0,
                "reachability_search_score": 0.8,
                "state_target_progress_from_vanilla": 0.0,
                "state_support_distance_to_vanilla": 0.2,
                "target_coverage_fraction": 0.0,
                "state_delta_q_vs_start": -10.0,
                "mutable_node_count": 1,
            },
        ]
    )

    selected = search.select_search_beam(
        states,
        rows,
        beam_width=1,
        search_policy=search.SEARCH_POLICY_REACHABILITY_FIRST,
    )

    assert search.search_policy_score_column(
        search.SEARCH_POLICY_REACHABILITY_FIRST
    ) == "reachability_search_score"
    assert selected[0].state_id == "escape"


def test_select_pareto_rows_drops_dominated_search_states():
    rows = pd.DataFrame(
        [
            {
                "state_id": "good",
                "state_delta_q_vs_start": 1.0,
                "state_candidate_progress_from_vanilla": 0.2,
                "state_target_progress_from_vanilla": 0.2,
                "state_support_distance_to_vanilla": 0.1,
                "mutable_node_count": 10,
                "search_score": 2.0,
                "state_greedy_score": 2.0,
            },
            {
                "state_id": "dominated",
                "state_delta_q_vs_start": 0.5,
                "state_candidate_progress_from_vanilla": 0.1,
                "state_target_progress_from_vanilla": 0.1,
                "state_support_distance_to_vanilla": 0.05,
                "mutable_node_count": 12,
                "search_score": 1.0,
                "state_greedy_score": 1.0,
            },
            {
                "state_id": "tradeoff",
                "state_delta_q_vs_start": 0.1,
                "state_candidate_progress_from_vanilla": 0.4,
                "state_target_progress_from_vanilla": 0.4,
                "state_support_distance_to_vanilla": 0.2,
                "mutable_node_count": 30,
                "search_score": 1.5,
                "state_greedy_score": 1.5,
            },
        ]
    )

    selected = search.select_pareto_rows(rows)

    assert selected["state_id"].tolist() == ["good", "tradeoff"]


def test_select_pareto_rows_reachability_first_does_not_dominate_by_quality():
    rows = pd.DataFrame(
        [
            {
                "state_id": "q_good",
                "state_delta_q_vs_start": 2.0,
                "state_candidate_progress_from_vanilla": 0.0,
                "state_target_progress_from_vanilla": 0.0,
                "state_support_distance_to_vanilla": 0.01,
                "target_coverage_fraction": 0.0,
                "mutable_node_count": 10,
                "search_score": 2.0,
                "state_greedy_score": 2.0,
                "reachability_search_score": 0.01,
            },
            {
                "state_id": "bad_q_escape",
                "state_delta_q_vs_start": -5.0,
                "state_candidate_progress_from_vanilla": 0.0,
                "state_target_progress_from_vanilla": 0.0,
                "state_support_distance_to_vanilla": 0.2,
                "target_coverage_fraction": 0.0,
                "mutable_node_count": 12,
                "search_score": -5.0,
                "state_greedy_score": -5.0,
                "reachability_search_score": 0.2,
            },
        ]
    )

    selected = search.select_pareto_rows(
        rows,
        search_policy=search.SEARCH_POLICY_REACHABILITY_FIRST,
    )

    assert selected["state_id"].tolist() == ["bad_q_escape", "q_good"]


def test_compute_pathway_wall_rows_tracks_intermediate_qf_debt():
    rows = pd.DataFrame(
        [
            {
                "state_id": "root",
                "parent_state_id": "",
                "pair_id": "p0",
                "depth": 0,
                "prefix_rank": 1,
                "prefix_unit_ids": "u1",
                "action_type": search.ACTION_PREFIX_ONLY,
                "applied_actions": "prefix_only",
                "elapsed_sec": 0.1,
                "peak_raw_barrier_input": 5.0,
                "state_delta_q_vs_start": -2.0,
                "state_q_debt_vs_start": 2.0,
                "state_support_distance_to_vanilla": 0.02,
                "state_support_distance_to_candidate": 0.8,
                "state_target_progress_from_vanilla": 0.01,
                "state_candidate_progress_from_vanilla": 0.01,
                "target_coverage_fraction": 0.2,
                "covered_target_count": 2,
                "remaining_target_count": 8,
                "mutable_node_count": 2,
            },
            {
                "state_id": "child",
                "parent_state_id": "root",
                "pair_id": "p0",
                "depth": 1,
                "prefix_rank": 1,
                "prefix_unit_ids": "u1",
                "action_type": search.ACTION_REMAINING_TARGET_TOPK,
                "applied_actions": "prefix_only,remaining_target_topk",
                "elapsed_sec": 0.2,
                "peak_raw_barrier_input": np.nan,
                "state_delta_q_vs_start": 1.0,
                "state_q_debt_vs_start": 0.0,
                "state_support_distance_to_vanilla": 0.06,
                "state_support_distance_to_candidate": 0.7,
                "state_target_progress_from_vanilla": 0.03,
                "state_candidate_progress_from_vanilla": 0.03,
                "target_coverage_fraction": 0.5,
                "covered_target_count": 5,
                "remaining_target_count": 5,
                "mutable_node_count": 5,
            },
        ]
    )

    paths = search.compute_pathway_wall_rows(
        rows,
        source_label="unit",
        support_gate=0.05,
        barrier_floor=1.0,
    )
    child = paths.set_index("path_final_state_id").loc["child"]

    assert child["source_label"] == "unit"
    assert child["path_root_state_id"] == "root"
    assert child["path_state_count"] == 2
    assert np.isclose(child["path_q_wall"], 2.0)
    assert np.isclose(child["path_min_delta_q_vs_start"], -2.0)
    assert np.isclose(child["path_final_delta_q_vs_start"], 1.0)
    assert np.isclose(child["path_q_recovery_from_wall"], 3.0)
    assert np.isclose(child["path_wall_reduction_vs_prefix_raw_barrier"], 3.0)
    assert bool(child["path_support_gate_reached"])
    assert bool(child["path_support_gate_q_recovered"])
    assert np.isclose(child["path_target_progress_gain_from_root"], 0.02)
    assert np.isclose(child["path_elapsed_sec_sum"], 0.3)


def test_annotate_pathway_debt_area_rows_tracks_shortcut_cost():
    states = pd.DataFrame(
        [
            {
                "state_id": "root",
                "parent_state_id": "",
                "pair_id": "p0",
                "depth": 0,
                "prefix_rank": 1,
                "elapsed_sec": 0.1,
                "state_delta_q_vs_start": -2.0,
                "state_q_debt_vs_start": 2.0,
                "state_support_distance_to_vanilla": 0.01,
                "state_support_distance_to_candidate": 0.8,
                "state_target_progress_from_vanilla": 0.01,
                "target_coverage_fraction": 0.2,
                "mutable_node_count": 2,
            },
            {
                "state_id": "child",
                "parent_state_id": "root",
                "pair_id": "p0",
                "depth": 1,
                "prefix_rank": 1,
                "elapsed_sec": 0.2,
                "state_delta_q_vs_start": -1.0,
                "state_q_debt_vs_start": 1.0,
                "state_support_distance_to_vanilla": 0.04,
                "state_support_distance_to_candidate": 0.7,
                "state_target_progress_from_vanilla": 0.02,
                "target_coverage_fraction": 0.4,
                "mutable_node_count": 5,
            },
            {
                "state_id": "final",
                "parent_state_id": "child",
                "pair_id": "p0",
                "depth": 2,
                "prefix_rank": 1,
                "elapsed_sec": 0.3,
                "state_delta_q_vs_start": 1.0,
                "state_q_debt_vs_start": 0.0,
                "state_support_distance_to_vanilla": 0.06,
                "state_support_distance_to_candidate": 0.6,
                "state_target_progress_from_vanilla": 0.03,
                "target_coverage_fraction": 0.5,
                "mutable_node_count": 6,
            },
        ]
    )
    paths = search.compute_pathway_wall_rows(states, support_gate=0.05)

    annotated = search.annotate_pathway_debt_area_rows(
        paths,
        state_rows=states,
        support_gate=0.05,
    )
    final = annotated.set_index("path_final_state_id").loc["final"]

    assert bool(final["path_chain_parent_complete"])
    assert final["path_debt_below_start_state_count"] == 2
    assert np.isclose(final["path_debt_below_start_fraction"], 2 / 3)
    assert np.isclose(final["path_q_debt_area_step"], 3.0)
    assert np.isclose(final["path_q_debt_area_elapsed"], 0.4)
    assert np.isclose(final["path_q_debt_area_mutable"], 7.0)
    assert final["path_wall_duration_steps"] == 2
    assert final["path_post_wall_steps"] == 2
    assert np.isclose(final["path_post_wall_elapsed_sec"], 0.5)
    assert np.isclose(final["path_post_wall_marginal_mutable"], 4.0)
    assert np.isclose(final["path_q_recovery_from_wall_area"], 3.0)
    assert np.isclose(final["path_recovery_slope_per_step"], 1.5)
    assert np.isclose(final["path_recovery_slope_per_mutable"], 0.75)
    assert np.isclose(final["path_support_per_debt_area_step"], 0.02)
    assert np.isclose(final["path_progress_per_debt_area_step"], 0.01)
    assert bool(final["path_gate_quality_recovered_by_area"])


def test_annotate_tunneling_evidence_rows_labels_recovered_and_detour():
    paths = pd.DataFrame(
        [
            {
                "artifact_label": "branch",
                "path_final_state_id": "recovered",
                "path_q_wall": 2.0,
                "path_q_debt_area_step": 2.0,
                "path_q_debt_area_mutable": 20.0,
                "path_final_delta_q_vs_start": 1.0,
                "path_final_support_distance_to_vanilla": 0.08,
                "path_final_target_progress_from_vanilla": 0.02,
            },
            {
                "artifact_label": "side",
                "path_final_state_id": "detour",
                "path_q_wall": 0.4,
                "path_q_debt_area_step": 0.6,
                "path_q_debt_area_mutable": 6.0,
                "path_final_delta_q_vs_start": -0.2,
                "path_final_support_distance_to_vanilla": 0.07,
                "path_final_target_progress_from_vanilla": 0.02,
            },
        ]
    )

    annotated = search.annotate_tunneling_evidence_rows(paths, support_gate=0.05)
    labels = annotated.set_index("path_final_state_id")["tunnel_route_label"]

    assert labels["recovered"] == search.TUNNEL_ROUTE_RECOVERABLE
    assert labels["detour"] == search.TUNNEL_ROUTE_UNRECOVERED_DETOUR
    recovered = annotated.set_index("path_final_state_id").loc["recovered"]
    assert bool(recovered["tunnel_recoverable"])
    assert np.isclose(recovered["tunnel_q_recovery_per_debt_area_step"], 0.5)
    assert np.isclose(recovered["tunnel_wall_concentration"], 1.0)


def test_trace_tunneling_path_states_marks_wall_gate_and_recovery_steps():
    states = pd.DataFrame(
        [
            {
                "state_id": "root",
                "parent_state_id": "",
                "action_type": "raw",
                "state_delta_q_vs_start": -2.0,
                "state_q_debt_vs_start": 2.0,
                "state_support_distance_to_vanilla": 0.01,
                "state_support_distance_to_candidate": 0.8,
                "state_target_progress_from_vanilla": 0.01,
                "mutable_node_count": 2,
                "marginal_mutable_node_count": 2,
                "elapsed_sec": 0.1,
            },
            {
                "state_id": "gate",
                "parent_state_id": "root",
                "action_type": search.ACTION_REMAINING_TARGET_TOPK,
                "state_delta_q_vs_start": -1.0,
                "state_q_debt_vs_start": 1.0,
                "state_support_distance_to_vanilla": 0.06,
                "state_support_distance_to_candidate": 0.7,
                "state_target_progress_from_vanilla": 0.02,
                "mutable_node_count": 5,
                "marginal_mutable_node_count": 3,
                "elapsed_sec": 0.2,
            },
            {
                "state_id": "final",
                "parent_state_id": "gate",
                "action_type": search.ACTION_REMAINING_TARGET_TOPK,
                "state_delta_q_vs_start": 1.0,
                "state_q_debt_vs_start": 0.0,
                "state_support_distance_to_vanilla": 0.07,
                "state_support_distance_to_candidate": 0.6,
                "state_target_progress_from_vanilla": 0.03,
                "mutable_node_count": 6,
                "marginal_mutable_node_count": 1,
                "elapsed_sec": 0.3,
            },
        ]
    )
    paths = search.compute_pathway_wall_rows(states, support_gate=0.05)
    paths = search.annotate_pathway_debt_area_rows(
        paths,
        state_rows=states,
        support_gate=0.05,
    )
    paths = search.annotate_tunneling_evidence_rows(paths, support_gate=0.05)
    final_path = paths[paths["path_final_state_id"].eq("final")]

    trace = search.trace_tunneling_path_states(
        final_path,
        state_rows=states,
        support_gate=0.05,
    )

    assert trace["trace_state_id"].tolist() == ["root", "gate", "final"]
    assert trace["trace_first_support_gate_step"].unique().tolist() == [1]
    assert trace["trace_first_candidate_directed_step"].unique().tolist() == [1]
    assert trace["trace_first_q_recovered_step"].unique().tolist() == [2]
    root = trace.set_index("trace_state_id").loc["root"]
    gate = trace.set_index("trace_state_id").loc["gate"]
    final = trace.set_index("trace_state_id").loc["final"]
    assert bool(root["trace_is_wall_peak"])
    assert gate["trace_phase"] == "under_q_debt"
    assert final["trace_phase"] == "candidate_recovered"


def test_summarize_post_gate_recovery_paths_separates_near_miss_and_plateau():
    trace = pd.DataFrame(
        [
            {
                "artifact_label": "side",
                "pair_id": "p0",
                "path_final_state_id": "near",
                "tunnel_route_label": search.TUNNEL_ROUTE_UNRECOVERED_DETOUR,
                "path_prefix_rank": 8,
                "path_selection_policy": "fixed_cap",
                "path_policy": "fixed_cap",
                "trace_step_index": 0,
                "trace_delta_q_vs_start": 0.2,
                "trace_q_debt": 0.0,
                "trace_support_distance_to_vanilla": 0.03,
                "trace_target_progress_from_vanilla": 0.01,
                "trace_first_candidate_directed_step": 1,
                "trace_first_support_gate_step": 1,
                "trace_wall_step_index": 1,
                "trace_candidate_directed": False,
                "trace_q_recovered": True,
            },
            {
                "artifact_label": "side",
                "pair_id": "p0",
                "path_final_state_id": "near",
                "tunnel_route_label": search.TUNNEL_ROUTE_UNRECOVERED_DETOUR,
                "path_prefix_rank": 8,
                "path_selection_policy": "fixed_cap",
                "path_policy": "fixed_cap",
                "trace_step_index": 1,
                "trace_delta_q_vs_start": -0.3,
                "trace_q_debt": 0.3,
                "trace_support_distance_to_vanilla": 0.06,
                "trace_target_progress_from_vanilla": 0.02,
                "trace_first_candidate_directed_step": 1,
                "trace_first_support_gate_step": 1,
                "trace_wall_step_index": 1,
                "trace_candidate_directed": True,
                "trace_q_recovered": False,
            },
            {
                "artifact_label": "side",
                "pair_id": "p0",
                "path_final_state_id": "near",
                "tunnel_route_label": search.TUNNEL_ROUTE_UNRECOVERED_DETOUR,
                "path_prefix_rank": 8,
                "path_selection_policy": "fixed_cap",
                "path_policy": "fixed_cap",
                "trace_step_index": 2,
                "trace_delta_q_vs_start": -0.2,
                "trace_q_debt": 0.2,
                "trace_support_distance_to_vanilla": 0.07,
                "trace_target_progress_from_vanilla": 0.03,
                "trace_first_candidate_directed_step": 1,
                "trace_first_support_gate_step": 1,
                "trace_wall_step_index": 1,
                "trace_candidate_directed": True,
                "trace_q_recovered": False,
            },
            {
                "artifact_label": "side",
                "pair_id": "p0",
                "path_final_state_id": "plateau",
                "tunnel_route_label": search.TUNNEL_ROUTE_UNRECOVERED_DETOUR,
                "path_prefix_rank": 6,
                "path_selection_policy": "fixed_cap",
                "path_policy": "fixed_cap",
                "trace_step_index": 0,
                "trace_delta_q_vs_start": -0.4,
                "trace_q_debt": 0.4,
                "trace_support_distance_to_vanilla": 0.06,
                "trace_target_progress_from_vanilla": 0.02,
                "trace_first_candidate_directed_step": 0,
                "trace_first_support_gate_step": 0,
                "trace_wall_step_index": 0,
                "trace_candidate_directed": True,
                "trace_q_recovered": False,
            },
            {
                "artifact_label": "side",
                "pair_id": "p0",
                "path_final_state_id": "plateau",
                "tunnel_route_label": search.TUNNEL_ROUTE_UNRECOVERED_DETOUR,
                "path_prefix_rank": 6,
                "path_selection_policy": "fixed_cap",
                "path_policy": "fixed_cap",
                "trace_step_index": 1,
                "trace_delta_q_vs_start": -0.4,
                "trace_q_debt": 0.4,
                "trace_support_distance_to_vanilla": 0.06,
                "trace_target_progress_from_vanilla": 0.02,
                "trace_first_candidate_directed_step": 0,
                "trace_first_support_gate_step": 0,
                "trace_wall_step_index": 0,
                "trace_candidate_directed": True,
                "trace_q_recovered": False,
            },
        ]
    )

    steps = search.annotate_post_gate_recovery_step_rows(trace)
    near_steps = steps[steps["path_final_state_id"].eq("near")]
    assert near_steps["post_gate_step_label"].tolist() == [
        search.POST_GATE_STEP_PRE_GATE,
        search.POST_GATE_STEP_GATE_ENTRY,
        search.POST_GATE_STEP_RECOVERY_TREND,
    ]

    summary = search.summarize_post_gate_recovery_paths(trace)
    verdicts = summary.set_index("path_final_state_id")["post_gate_verdict"]
    assert verdicts["near"] == search.POST_GATE_VERDICT_NEAR_MISS
    assert verdicts["plateau"] == search.POST_GATE_VERDICT_PLATEAU
    near = summary.set_index("path_final_state_id").loc["near"]
    assert np.isclose(near["post_gate_best_delta_q_gain_from_gate"], 0.1)
    assert near["post_gate_recovery_step_count"] == 1


def test_build_post_gate_recovery_actions_emits_context_and_transplant_probes():
    state = search.make_prefix_state(
        state_id="root",
        prefix_rank=1,
        prefix_unit_ids="u",
        membership=np.asarray([0, 0, 1, 1, 2, 2], dtype=np.uint64),
        quality=1.0,
        direct_nodes=np.asarray([1, 2], dtype=np.uint32),
        action_nodes=np.asarray([1, 2], dtype=np.uint32),
        mutable_nodes=np.asarray([1, 2], dtype=np.uint32),
        target_nodes=np.asarray([1, 2, 3, 4], dtype=np.uint32),
    )
    candidate = np.asarray([0, 3, 3, 3, 4, 4], dtype=np.uint64)
    vanilla = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.uint64)
    src = np.asarray([1, 2, 2, 3, 4], dtype=np.uint32)
    dst = np.asarray([3, 3, 4, 5, 5], dtype=np.uint32)
    weight = np.ones(src.size, dtype=np.float64)

    actions = search.build_post_gate_recovery_actions(
        state=state,
        candidate_membership=candidate,
        vanilla_membership=vanilla,
        src=src,
        dst=dst,
        weight=weight,
        node_count=6,
        action_types=(
            search.ACTION_CANDIDATE_CLOSURE_TOPK,
            search.ACTION_BOUNDARY_SHELL_TOPK,
        ),
        context_multiplier=2.0,
        max_context_nodes=4,
        include_context_only=True,
        include_candidate_transplant=True,
        include_boundary_transplant=False,
    )

    by_type = {candidate.action.action_type: candidate for candidate in actions}
    assert search.ACTION_RECOVERY_CANDIDATE_CONTEXT_TOPK in by_type
    assert search.ACTION_RECOVERY_CANDIDATE_TRANSPLANT_TOPK in by_type
    assert search.ACTION_RECOVERY_BOUNDARY_CONTEXT_TOPK in by_type
    context = by_type[search.ACTION_RECOVERY_CANDIDATE_CONTEXT_TOPK].action
    transplant = by_type[search.ACTION_RECOVERY_CANDIDATE_TRANSPLANT_TOPK].action
    assert context.action_nodes is None
    assert transplant.action_nodes is not None
    assert set(transplant.action_nodes.astype(int)) == {3}


def test_classify_post_gate_recovery_move_rows_marks_q_gain_and_tradeoff():
    rows = pd.DataFrame(
        [
            {
                "state_id": "q_gain",
                "state_delta_q_vs_start": -0.2,
                "state_support_distance_to_vanilla": 0.07,
                "state_target_progress_from_vanilla": 0.03,
            },
            {
                "state_id": "tradeoff",
                "state_delta_q_vs_start": -0.4,
                "state_support_distance_to_vanilla": 0.09,
                "state_target_progress_from_vanilla": 0.04,
            },
            {
                "state_id": "recovered",
                "state_delta_q_vs_start": 0.1,
                "state_support_distance_to_vanilla": 0.061,
                "state_target_progress_from_vanilla": 0.03,
            },
        ]
    )

    classified = search.classify_post_gate_recovery_move_rows(
        rows,
        target_delta_q=-0.3,
        target_support=0.06,
        target_progress=0.02,
        support_gate=0.05,
        progress_margin=0.005,
    )
    verdicts = classified.set_index("state_id")["post_gate_move_verdict"]

    assert verdicts["q_gain"] == search.POST_GATE_RECOVERY_MOVE_Q_GAIN
    assert verdicts["tradeoff"] == search.POST_GATE_RECOVERY_MOVE_SUPPORT_TRADEOFF
    assert verdicts["recovered"] == search.POST_GATE_RECOVERY_MOVE_RECOVERED


def test_rank_tunneling_operator_candidates_prioritizes_recovered_seed():
    paths = pd.DataFrame(
        [
            {
                "tunnel_route_label": search.TUNNEL_ROUTE_UNRECOVERED_DETOUR,
                "path_final_state_id": "detour",
                "path_q_wall": 0.2,
                "path_q_debt_area_step": 0.2,
                "path_q_debt_area_mutable": 2.0,
                "path_final_delta_q_vs_start": -0.1,
                "path_final_support_distance_to_vanilla": 0.08,
                "path_final_target_progress_from_vanilla": 0.03,
                "path_final_mutable_node_count": 20,
            },
            {
                "tunnel_route_label": search.TUNNEL_ROUTE_RECOVERABLE,
                "path_final_state_id": "tunnel",
                "path_q_wall": 1.0,
                "path_q_debt_area_step": 1.0,
                "path_q_debt_area_mutable": 5.0,
                "path_final_delta_q_vs_start": 1.0,
                "path_final_support_distance_to_vanilla": 0.07,
                "path_final_target_progress_from_vanilla": 0.02,
                "path_final_mutable_node_count": 10,
            },
            {
                "tunnel_route_label": search.TUNNEL_ROUTE_PARTIAL_PROGRESS,
                "path_final_state_id": "partial",
                "path_q_wall": 0.0,
                "path_q_debt_area_step": 0.0,
                "path_q_debt_area_mutable": 0.0,
                "path_final_delta_q_vs_start": 0.2,
                "path_final_support_distance_to_vanilla": 0.03,
                "path_final_target_progress_from_vanilla": 0.02,
                "path_final_mutable_node_count": 5,
            },
        ]
    )

    ranked = search.rank_tunneling_operator_candidates(paths, support_gate=0.05)

    assert ranked["path_final_state_id"].tolist() == ["tunnel", "detour", "partial"]
    assert (
        ranked.iloc[0]["tunnel_operator_category"]
        == search.TUNNEL_OPERATOR_RECOVERABLE_SEED
    )
    assert (
        ranked.iloc[1]["tunnel_operator_category"]
        == search.TUNNEL_OPERATOR_RECOVERY_TARGET
    )
    assert bool(ranked.iloc[0]["tunnel_operator_acceptance_ready"])


def test_summarize_pathway_wall_rows_reports_lowest_wall_gate_path():
    paths = pd.DataFrame(
        [
            {
                "source_label": "a",
                "pair_id": "p0",
                "path_final_state_id": "wide",
                "path_q_wall": 2.0,
                "path_wall_crossed": True,
                "path_final_delta_q_vs_start": 1.0,
                "path_final_support_distance_to_vanilla": 0.2,
                "path_final_target_progress_from_vanilla": 0.03,
                "path_final_target_coverage_fraction": 0.5,
                "path_final_mutable_node_count": 10,
            },
            {
                "source_label": "a",
                "pair_id": "p0",
                "path_final_state_id": "cheap_gate",
                "path_q_wall": 0.5,
                "path_wall_crossed": True,
                "path_final_delta_q_vs_start": -0.1,
                "path_final_support_distance_to_vanilla": 0.06,
                "path_final_target_progress_from_vanilla": 0.02,
                "path_final_target_coverage_fraction": 0.4,
                "path_final_mutable_node_count": 6,
            },
            {
                "source_label": "a",
                "pair_id": "p0",
                "path_final_state_id": "near",
                "path_q_wall": 0.0,
                "path_wall_crossed": False,
                "path_final_delta_q_vs_start": 2.0,
                "path_final_support_distance_to_vanilla": 0.01,
                "path_final_target_progress_from_vanilla": 0.0,
                "path_final_target_coverage_fraction": 0.1,
                "path_final_mutable_node_count": 2,
            },
        ]
    )

    summary = search.summarize_pathway_wall_rows(paths, support_gate=0.05)

    row = summary.iloc[0]
    assert row["support_gate_rows"] == 2
    assert row["support_gate_q_recovered_rows"] == 1
    assert row["min_wall_gate_state_id"] == "cheap_gate"
    assert np.isclose(row["support_gate_q_wall_min"], 0.5)


def test_select_qf_wall_frontier_keeps_wall_progress_tradeoff():
    paths = pd.DataFrame(
        [
            {
                "path_final_state_id": "cheap",
                "path_q_wall": 0.0,
                "path_final_target_progress_from_vanilla": 0.01,
                "path_final_support_distance_to_vanilla": 0.02,
                "path_final_target_coverage_fraction": 0.1,
                "path_final_delta_q_vs_start": 1.0,
                "path_final_mutable_node_count": 5,
            },
            {
                "path_final_state_id": "dominated",
                "path_q_wall": 1.0,
                "path_final_target_progress_from_vanilla": 0.005,
                "path_final_support_distance_to_vanilla": 0.01,
                "path_final_target_coverage_fraction": 0.05,
                "path_final_delta_q_vs_start": 0.5,
                "path_final_mutable_node_count": 8,
            },
            {
                "path_final_state_id": "progress",
                "path_q_wall": 2.0,
                "path_final_target_progress_from_vanilla": 0.08,
                "path_final_support_distance_to_vanilla": 0.2,
                "path_final_target_coverage_fraction": 0.6,
                "path_final_delta_q_vs_start": -0.5,
                "path_final_mutable_node_count": 12,
            },
        ]
    )

    frontier = search.select_qf_wall_frontier(paths)

    assert set(frontier["path_final_state_id"]) == {"cheap", "progress"}


def test_select_branch_path_rows_preserves_selection_policy_diversity():
    paths = pd.DataFrame(
        [
            {
                "path_final_state_id": "fixed_best",
                "path_selection_policy": search.TARGET_SELECTION_FIXED_CAP,
                "path_q_wall": 1.0,
                "path_final_delta_q_vs_start": 1.0,
                "path_final_support_distance_to_vanilla": 0.20,
                "path_final_target_progress_from_vanilla": 0.04,
                "path_final_target_coverage_fraction": 0.4,
                "path_final_mutable_node_count": 100,
            },
            {
                "path_final_state_id": "fixed_second",
                "path_selection_policy": search.TARGET_SELECTION_FIXED_CAP,
                "path_q_wall": 1.1,
                "path_final_delta_q_vs_start": 1.0,
                "path_final_support_distance_to_vanilla": 0.19,
                "path_final_target_progress_from_vanilla": 0.04,
                "path_final_target_coverage_fraction": 0.4,
                "path_final_mutable_node_count": 100,
            },
            {
                "path_final_state_id": "guarded",
                "path_selection_policy": search.TARGET_SELECTION_GUARDED_ELBOW,
                "path_q_wall": 0.4,
                "path_final_delta_q_vs_start": 0.5,
                "path_final_support_distance_to_vanilla": 0.08,
                "path_final_target_progress_from_vanilla": 0.02,
                "path_final_target_coverage_fraction": 0.2,
                "path_final_mutable_node_count": 40,
            },
        ]
    )

    selected = search.select_branch_path_rows(paths, beam_width=2)

    assert selected["path_final_state_id"].tolist() == ["fixed_best", "guarded"]
    assert "path_branch_discovery_score" in selected.columns


def test_classify_branch_greedy_failure_rows_combines_prefix_wall_and_controls():
    paths = pd.DataFrame(
        [
            {
                "case": "case-a",
                "pair_id": "p1",
                "path_root_state_id": "root",
                "path_final_state_id": "child",
                "path_state_ids": "root|child",
                "path_selection_policy": search.TARGET_SELECTION_FIXED_TAIL_BACKFILL,
                "path_q_wall": 1.5,
                "path_q_recovery_from_wall": 2.0,
                "path_final_delta_q_vs_start": 0.5,
                "path_final_support_distance_to_vanilla": 0.08,
                "path_final_target_progress_from_vanilla": 0.02,
                "path_final_target_coverage_fraction": 0.3,
                "path_final_mutable_node_count": 20,
                "path_support_gate_q_recovered": True,
            }
        ]
    )
    states = pd.DataFrame(
        [
            {
                "state_id": "root",
                "greedy_failure_labels": (
                    "q_greedy_miss;progress_greedy_miss;"
                    "closure_compound_miss;polish_recovery_miss"
                ),
                "context_to_action_ratio": 0.0,
                "marginal_q_debt": 1.5,
            },
            {
                "state_id": "child",
                "greedy_failure_labels": "",
                "context_to_action_ratio": 0.0,
                "marginal_q_debt": -1.5,
            },
        ]
    )
    controls = pd.DataFrame(
        [
            {
                "row_type": "control",
                "pair_id": "p1",
                "delta_q_vs_vanilla": 2.0,
                "target_progress_from_vanilla": -0.2,
                "support_distance_to_vanilla": 0.6,
            }
        ]
    )

    classified = search.classify_branch_greedy_failure_rows(
        paths,
        state_rows=states,
        control_rows=controls,
        material_delta_q=1.0,
    )

    row = classified.iloc[0]
    assert row["path_candidate_directed"]
    assert row["q_greedy_miss"]
    assert row["progress_greedy_miss"]
    assert row["closure_compound_miss"]
    assert row["polish_recovery_miss"]
    assert row["candidate_directed_control_count"] == 0
    assert row["control_comparison_status"] == (
        search.GREEDY_CONTROL_BRANCH_UNIQUE_QUALITY_LAG
    )
    assert np.isclose(row["branch_delta_q_minus_best_control"], -1.5)


def test_summarize_greedy_failure_rows_reports_best_and_counts():
    classified = pd.DataFrame(
        [
            {
                "case": "case-a",
                "pair_id": "p1",
                "path_final_state_id": "weak",
                "path_candidate_directed": False,
                "path_support_gate_q_recovered": False,
                "q_greedy_miss": False,
                "progress_greedy_miss": False,
                "closure_compound_miss": False,
                "polish_recovery_miss": False,
                "control_comparison_status": search.GREEDY_CONTROL_NOT_CANDIDATE_DIRECTED,
                "failure_labels": "greedy_visible",
                "path_branch_discovery_score": 0.1,
                "path_final_support_distance_to_vanilla": 0.01,
                "path_final_delta_q_vs_start": 0.0,
                "path_final_target_progress_from_vanilla": 0.0,
                "path_q_wall": 0.0,
                "path_final_mutable_node_count": 1,
            },
            {
                "case": "case-a",
                "pair_id": "p1",
                "path_final_state_id": "best",
                "path_candidate_directed": True,
                "path_support_gate_q_recovered": True,
                "q_greedy_miss": True,
                "progress_greedy_miss": True,
                "closure_compound_miss": True,
                "polish_recovery_miss": True,
                "control_comparison_status": (
                    search.GREEDY_CONTROL_BRANCH_UNIQUE_QUALITY_LAG
                ),
                "failure_labels": (
                    "q_greedy_miss;progress_greedy_miss;"
                    "closure_compound_miss;polish_recovery_miss"
                ),
                "path_branch_discovery_score": 2.0,
                "path_final_support_distance_to_vanilla": 0.1,
                "path_final_delta_q_vs_start": 1.0,
                "path_final_target_progress_from_vanilla": 0.03,
                "path_q_wall": 1.2,
                "path_final_mutable_node_count": 12,
            },
        ]
    )

    summary = search.summarize_greedy_failure_rows(classified)

    row = summary.iloc[0]
    assert row["path_rows"] == 2
    assert row["candidate_directed_rows"] == 1
    assert row["support_gate_q_recovered_rows"] == 1
    assert row["q_greedy_miss_rows"] == 1
    assert row["unique_candidate_directed_quality_lag_rows"] == 1
    assert row["best_state_id"] == "best"


def test_summarize_wall_route_families_separates_side_routes_from_gate_wall():
    rows = pd.DataFrame(
        [
            {
                "case": "case-a",
                "pair_id": "p1",
                "path_prefix_rank": 1,
                "path_final_state_id": "side",
                "path_candidate_directed": False,
                "path_q_recovered": True,
                "path_final_support_distance_to_vanilla": 0.04,
                "path_final_target_progress_from_vanilla": 0.02,
                "path_final_delta_q_vs_start": 0.2,
                "path_q_wall": 0.2,
                "path_final_mutable_node_count": 10,
                "path_branch_discovery_score": 0.3,
                "failure_labels": "q_greedy_miss",
                "control_comparison_status": "not_candidate_directed",
            },
            {
                "case": "case-a",
                "pair_id": "p1",
                "path_prefix_rank": 9,
                "path_final_state_id": "gate",
                "path_candidate_directed": True,
                "path_q_recovered": True,
                "path_final_support_distance_to_vanilla": 0.08,
                "path_final_target_progress_from_vanilla": 0.03,
                "path_final_delta_q_vs_start": 1.0,
                "path_q_wall": 1.5,
                "path_final_mutable_node_count": 30,
                "path_branch_discovery_score": 1.0,
                "failure_labels": "q_greedy_miss;polish_recovery_miss",
                "control_comparison_status": (
                    search.GREEDY_CONTROL_BRANCH_UNIQUE_QUALITY_LAG
                ),
            },
        ]
    )

    family_rows, prefix_rows, summary_rows = search.summarize_wall_route_families(
        rows,
        support_gate=0.05,
        progress_margin=0.005,
        side_support_fraction=0.75,
    )

    summary = summary_rows.iloc[0]
    assert summary["candidate_directed_wall_entries"] == 1
    assert summary["lower_wall_side_route_rows"] == 1
    assert summary["wall_route_verdict"] == (
        "single_observed_candidate_wall_with_lower_wall_side_routes"
    )
    assert set(family_rows["prefix_rank"]) == {1, 9}
    side_prefix = prefix_rows[prefix_rows["prefix_rank"].eq(1)].iloc[0]
    assert side_prefix["side_route_candidate_rows"] == 1
