import math

import pytest

from sciscape.clustering.branch_adaptive import (
    best_match_child_jaccard,
    child_size_diagnostics,
    cpm_merge_gain,
    mean_pairwise_ami,
    rank_branch_split_candidates,
    source_max_ratio_delta_if_applied,
    split_accounting_from_between,
    split_accounting_from_child_weights,
    split_accounting_from_delta_cut,
    split_accounting_from_edges,
)


def test_three_node_gamma_star_counterexample_is_not_ultrametric():
    gamma_12 = split_accounting_from_between(
        e_between=0.30,
        w_between=1.0,
        gamma=0.5,
    ).gamma_star_split
    gamma_23 = split_accounting_from_between(
        e_between=0.40,
        w_between=1.0,
        gamma=0.5,
    ).gamma_star_split
    gamma_13 = split_accounting_from_between(
        e_between=0.35,
        w_between=1.0,
        gamma=0.5,
    ).gamma_star_split

    assert [gamma_12, gamma_13, gamma_23] == pytest.approx([0.30, 0.35, 0.40])
    assert len({gamma_12, gamma_13, gamma_23}) == 3


def test_cpm_merge_gain_sign_changes_at_gamma_star():
    edge_weight = 0.9
    weight_a = 40.0
    weight_b = 20.0
    gamma_star = edge_weight / (weight_a * weight_b)

    assert cpm_merge_gain(edge_weight, weight_a, weight_b, gamma_star - 1e-4) > 0.0
    assert cpm_merge_gain(edge_weight, weight_a, weight_b, gamma_star + 1e-4) < 0.0


def test_multiway_split_gain_uses_pair_weight_scale():
    accounting = split_accounting_from_child_weights(
        [2.0, 3.0, 5.0],
        e_between=0.7,
        gamma=0.1,
    )

    assert accounting.w_between == pytest.approx(31.0)
    assert accounting.gamma_star_split == pytest.approx(0.7 / 31.0)
    assert accounting.delta_q_split_original == pytest.approx(2.4)
    assert accounting.raw_split_gap == pytest.approx(0.1 - 0.7 / 31.0)
    assert accounting.normalized_split_gain == pytest.approx(
        accounting.delta_q_split_original / accounting.w_between
    )


def test_split_accounting_from_probe_delta_matches_direct_formula():
    direct = split_accounting_from_between(e_between=1.25, w_between=240.0, gamma=0.01)
    recovered = split_accounting_from_delta_cut(
        delta_q_split_original=direct.delta_q_split_original,
        e_between=direct.e_between,
        gamma=0.01,
    )

    assert recovered.w_between == pytest.approx(direct.w_between)
    assert recovered.gamma_star_split == pytest.approx(direct.gamma_star_split)
    assert recovered.normalized_split_gain == pytest.approx(direct.normalized_split_gain)


def test_original_edge_accounting_ignores_external_edges_for_parent_split():
    accounting = split_accounting_from_edges(
        src=[0, 1, 0, 2, 1, 2],
        dst=[1, 2, 2, 3, 3, 3],
        edge_weight=[0.8, 0.2, 0.1, 99.0, 99.0, 99.0],
        nodes=[0, 1, 2],
        child_labels=[0, 0, 1],
        gamma=0.5,
    )

    assert accounting.e_between == pytest.approx(0.3)
    assert accounting.w_between == pytest.approx(2.0)
    assert accounting.delta_q_split_original == pytest.approx(0.7)


def test_original_edge_accounting_remaps_arbitrary_child_labels():
    accounting = split_accounting_from_edges(
        src=[0, 1, 0, 2],
        dst=[1, 2, 2, 3],
        edge_weight=[0.8, 0.2, 0.1, 99.0],
        nodes=[0, 1, 2],
        child_labels=[10, 10, 20],
        gamma=0.5,
    )

    assert accounting.e_between == pytest.approx(0.3)
    assert accounting.w_between == pytest.approx(2.0)
    assert accounting.delta_q_split_original == pytest.approx(0.7)


def test_child_size_diagnostics_and_source_pressure_delta():
    diagnostics = child_size_diagnostics([100.0, 300.0, 600.0], min_doc_weight=250.0)

    assert diagnostics["n_children_below_min"] == 1
    assert diagnostics["largest_child_fraction"] == pytest.approx(0.6)
    assert 0.0 < diagnostics["child_weight_entropy"] < 1.0
    assert source_max_ratio_delta_if_applied(
        parent_doc_weight=1000.0,
        largest_child_fraction=0.6,
        target_max_doc_weight=500.0,
    ) == pytest.approx(-0.8)


def test_rank_branch_split_candidates_is_quality_first_and_parent_exclusive():
    rows = [
        {
            "field": 30,
            "sample": "field30",
            "source_seed": 11,
            "parent_cluster": 5,
            "alpha": 1.05,
            "local_seed": 11,
            "k_children": 2,
            "delta_q_split_original": 1.0,
            "w_between": 1000.0,
            "normalized_split_gain": 0.001,
            "source_max_ratio_delta_if_applied": -0.5,
            "child_weight_entropy": 0.9,
            "n_children_below_min": 0,
            "largest_child_fraction": 0.5,
            "status": "ok",
        },
        {
            "field": 30,
            "sample": "field30",
            "source_seed": 11,
            "parent_cluster": 5,
            "alpha": 1.15,
            "local_seed": 11,
            "k_children": 3,
            "delta_q_split_original": 2.0,
            "w_between": 1000.0,
            "normalized_split_gain": 0.002,
            "source_max_ratio_delta_if_applied": -0.4,
            "child_weight_entropy": 0.8,
            "n_children_below_min": 1,
            "largest_child_fraction": 0.6,
            "status": "ok",
        },
        {
            "field": 30,
            "sample": "field30",
            "source_seed": 11,
            "parent_cluster": 6,
            "alpha": 1.15,
            "local_seed": 11,
            "k_children": 2,
            "delta_q_split_original": -0.1,
            "w_between": 1000.0,
            "normalized_split_gain": -0.0001,
            "source_max_ratio_delta_if_applied": -0.7,
            "child_weight_entropy": 0.9,
            "n_children_below_min": 0,
            "largest_child_fraction": 0.5,
            "status": "ok",
        },
    ]

    ranked = rank_branch_split_candidates(
        rows,
        gamma=0.01,
        tau_split_ratio=0.0,
        epsilon_q=0.0,
    )

    selected = [row for row in ranked if row["selected_for_apply"]]
    assert len(selected) == 1
    assert selected[0]["parent_cluster"] == 5
    assert selected[0]["alpha"] == 1.15
    assert any(row["conflict_reason"] == "parent_already_selected" for row in ranked)
    assert any(row["rejection_reason"] == "quality_regression" for row in ranked)


def test_stability_helpers():
    assert mean_pairwise_ami([[0, 0, 1], [0, 0, 1], [1, 1, 0]]) == pytest.approx(1.0)
    assert best_match_child_jaccard([0, 0, 1, 1], [0, 1, 1, 1]) == pytest.approx(
        (1 / 2 + 2 / 3) / 2
    )
    assert math.isfinite(mean_pairwise_ami([[0, 1, 2], [0, 0, 1]]))
