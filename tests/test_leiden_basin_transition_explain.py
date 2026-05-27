from __future__ import annotations

import numpy as np

from sciscape.clustering import leiden_basin_transition_explain as explain


def test_membership_change_summary_separates_label_permutation_from_partition_change():
    reference = np.asarray([0, 0, 1, 1], dtype=np.uint64)
    permuted = np.asarray([7, 7, 8, 8], dtype=np.uint64)

    summary = explain.membership_change_summary(
        reference_membership=reference,
        membership=permuted,
    )

    assert summary["exact_changed_node_count"] == 4
    assert summary["aligned_changed_node_count"] == 0
    assert summary["exact_only_changed_node_count"] == 4


def test_change_node_rows_marks_bundle_roles_and_hop_distance():
    reference = np.asarray([0, 0, 1, 1, 2], dtype=np.uint64)
    current = np.asarray([0, 1, 1, 1, 2], dtype=np.uint64)
    baseline = reference.copy()
    vanilla = reference.copy()
    candidate = np.asarray([0, 1, 1, 1, 2], dtype=np.uint64)
    src = np.asarray([0, 1, 2, 3], dtype=np.uint32)
    dst = np.asarray([1, 2, 3, 4], dtype=np.uint32)
    weight = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float64)

    rows = explain.build_change_node_rows(
        reference_membership=reference,
        membership=current,
        baseline_membership=baseline,
        vanilla_membership=vanilla,
        candidate_membership=candidate,
        src=src,
        dst=dst,
        weight=weight,
        target_nodes=np.asarray([1], dtype=np.uint32),
        context_nodes=np.asarray([2], dtype=np.uint32),
        bundle_nodes=np.asarray([1, 2], dtype=np.uint32),
        source_action_nodes=np.asarray([0], dtype=np.uint32),
        source_mutable_nodes=np.asarray([0], dtype=np.uint32),
        include_nodes=np.asarray([1, 2, 3], dtype=np.uint32),
    ).set_index("node")

    assert bool(rows.loc[1, "in_selected_target"])
    assert bool(rows.loc[2, "in_context"])
    assert rows.loc[3, "hop_to_bundle"] == 1
    assert rows.loc[3, "pull_to_bundle"] == 3.0


def test_change_shell_rows_aggregates_exact_only_changes():
    reference = np.asarray([0, 0, 1, 1], dtype=np.uint64)
    permuted = np.asarray([7, 7, 8, 8], dtype=np.uint64)
    src = np.asarray([0, 1, 2], dtype=np.uint32)
    dst = np.asarray([1, 2, 3], dtype=np.uint32)

    rows = explain.build_change_shell_rows(
        reference_membership=reference,
        membership=permuted,
        src=src,
        dst=dst,
        target_nodes=np.asarray([0], dtype=np.uint32),
        bundle_nodes=np.asarray([0], dtype=np.uint32),
    )

    exact_only = rows[rows["change_kind"].eq("exact_only_label_changed")]
    assert int(exact_only["node_count"].sum()) == 4
    aligned = rows[rows["change_kind"].eq("aligned_partition_changed")]
    assert int(aligned["node_count"].sum()) == 0
