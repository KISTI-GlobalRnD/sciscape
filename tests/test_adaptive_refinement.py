import json
import csv

import numpy as np

from sciscape.clustering.adaptive_refinement import (
    BoundaryCandidatePolicy,
    MacroMergePolicy,
    run_macro_merge_policy_ensemble,
    score_boundary_candidates,
    simulate_macro_merge_policy,
    summarize_boundary_candidate_policies,
    summarize_boundary_group_probes,
    summarize_boundary_move_probes,
    summarize_cluster_graph_stats,
    summarize_multi_core_split_probes,
    summarize_split_merge_repair_probes,
    write_adaptive_refinement_report,
    write_boundary_candidate_report,
    write_boundary_group_probe_report,
    write_boundary_move_probe_report,
    write_macro_merge_ensemble_report,
    write_multi_core_split_probe_report,
    write_split_merge_repair_probe_report,
)
from sciscape.clustering.leiden_rust import (
    RustBoundaryGroupProbes,
    RustBoundaryMoveProbes,
    RustClusterGraphStats,
    RustMultiCoreSplitProbes,
    RustSplitMergeRepairProbes,
)


def _stats() -> RustClusterGraphStats:
    return RustClusterGraphStats(
        block_count=np.array([2, 3, 1], dtype=np.uint64),
        doc_weight=np.array([20.0, 120.0, 400.0], dtype=np.float64),
        internal_weight=np.array([5.0, 20.0, 80.0], dtype=np.float64),
        external_weight=np.array([10.0, 15.0, 5.0], dtype=np.float64),
        degree=np.array([2, 2, 1], dtype=np.uint64),
        top_neighbor=np.array([1, 0, 1], dtype=np.int64),
        top_neighbor_weight=np.array([7.0, 7.0, 5.0], dtype=np.float64),
        second_neighbor=np.array([2, 2, -1], dtype=np.int64),
        second_neighbor_weight=np.array([3.0, 5.0, 0.0], dtype=np.float64),
        neighbor_weight_ratio=np.array([3.0 / 7.0, 5.0 / 7.0, 0.0], dtype=np.float64),
        conductance=np.array([0.5, 15.0 / 55.0, 5.0 / 165.0], dtype=np.float64),
        leafness=np.array([0.7, 7.0 / 15.0, 1.0], dtype=np.float64),
        band_distance=np.array([30.0, 0.0, 150.0], dtype=np.float64),
        candidate_source=np.array([0, 1], dtype=np.uint64),
        candidate_target=np.array([1, 2], dtype=np.uint64),
        candidate_edge_weight=np.array([7.0, 5.0], dtype=np.float64),
        candidate_delta_q=np.array([1.0, -2.0], dtype=np.float64),
        candidate_merged_weight=np.array([140.0, 520.0], dtype=np.float64),
        candidate_size_band_gain=np.array([30.0, -120.0], dtype=np.float64),
    )


def _move_probes() -> RustBoundaryMoveProbes:
    return RustBoundaryMoveProbes(
        cluster=np.array([10, 20], dtype=np.uint64),
        block_count=np.array([3, 2], dtype=np.uint64),
        doc_weight=np.array([120.0, 80.0], dtype=np.float64),
        internal_weight=np.array([20.0, 10.0], dtype=np.float64),
        external_weight=np.array([30.0, 20.0], dtype=np.float64),
        conductance=np.array([30.0 / 70.0, 20.0 / 40.0], dtype=np.float64),
        leafness=np.array([0.5, 0.8], dtype=np.float64),
        top_neighbor=np.array([1, 2], dtype=np.int64),
        top_neighbor_weight=np.array([15.0, 16.0], dtype=np.float64),
        second_neighbor=np.array([2, 3], dtype=np.int64),
        second_neighbor_weight=np.array([10.0, 4.0], dtype=np.float64),
        neighbor_weight_ratio=np.array([2.0 / 3.0, 0.25], dtype=np.float64),
        positive_move_count=np.array([2, 0], dtype=np.uint64),
        positive_move_weight=np.array([50.0, 0.0], dtype=np.float64),
        positive_delta_q=np.array([1.5, 0.0], dtype=np.float64),
        near_neutral_move_count=np.array([2, 1], dtype=np.uint64),
        near_neutral_move_weight=np.array([50.0, 10.0], dtype=np.float64),
        near_neutral_delta_q=np.array([1.5, -0.01], dtype=np.float64),
        best_move_delta_q=np.array([1.0, -0.01], dtype=np.float64),
        best_move_node=np.array([7, 8], dtype=np.uint64),
        best_move_target=np.array([1, 3], dtype=np.int64),
        top_move_count=np.array([1, 0], dtype=np.uint64),
        second_move_count=np.array([1, 0], dtype=np.uint64),
    )


def _group_probes() -> RustBoundaryGroupProbes:
    return RustBoundaryGroupProbes(
        cluster=np.array([10, 20], dtype=np.uint64),
        block_count=np.array([3, 2], dtype=np.uint64),
        doc_weight=np.array([120.0, 80.0], dtype=np.float64),
        top_neighbor=np.array([1, 2], dtype=np.int64),
        second_neighbor=np.array([2, 3], dtype=np.int64),
        top_group_count=np.array([2, 1], dtype=np.uint64),
        top_group_weight=np.array([50.0, 10.0], dtype=np.float64),
        top_group_to_target_weight=np.array([30.0, 2.0], dtype=np.float64),
        top_group_cut_weight=np.array([5.0, 1.0], dtype=np.float64),
        top_group_move_delta_q=np.array([2.0, -0.5], dtype=np.float64),
        top_group_split_delta_q=np.array([1.0, -1.0], dtype=np.float64),
        top_group_is_full_cluster=np.array([False, False], dtype=bool),
        second_group_count=np.array([1, 1], dtype=np.uint64),
        second_group_weight=np.array([20.0, 20.0], dtype=np.float64),
        second_group_to_target_weight=np.array([10.0, 3.0], dtype=np.float64),
        second_group_cut_weight=np.array([4.0, 2.0], dtype=np.float64),
        second_group_move_delta_q=np.array([0.5, -0.2], dtype=np.float64),
        second_group_split_delta_q=np.array([0.25, -0.4], dtype=np.float64),
        second_group_is_full_cluster=np.array([False, False], dtype=bool),
        best_delta_q=np.array([2.0, -0.2], dtype=np.float64),
        best_action=np.array([1, 0], dtype=np.uint8),
    )


def _multi_core_split_probes() -> RustMultiCoreSplitProbes:
    return RustMultiCoreSplitProbes(
        cluster=np.array([10, 10, 20], dtype=np.uint64),
        gamma_multiplier=np.array([1.5, 2.0, 1.5], dtype=np.float64),
        probe_resolution=np.array([0.01275, 0.017, 0.01275], dtype=np.float64),
        block_count=np.array([100, 100, 80], dtype=np.uint64),
        doc_weight=np.array([300.0, 300.0, 120.0], dtype=np.float64),
        internal_weight=np.array([500.0, 500.0, 90.0], dtype=np.float64),
        induced_directed_edges=np.array([1000, 1000, 250], dtype=np.uint64),
        n_parts=np.array([3, 5, 1], dtype=np.uint64),
        non_singleton_parts=np.array([3, 4, 1], dtype=np.uint64),
        singleton_parts=np.array([0, 1, 0], dtype=np.uint64),
        singleton_weight=np.array([0.0, 1.0, 0.0], dtype=np.float64),
        core_part_count=np.array([3, 4, 1], dtype=np.uint64),
        core_part_weight=np.array([300.0, 299.0, 120.0], dtype=np.float64),
        largest_part_weight=np.array([140.0, 100.0, 120.0], dtype=np.float64),
        second_part_weight=np.array([90.0, 80.0, 0.0], dtype=np.float64),
        largest_part_fraction=np.array([0.4667, 0.3333, 1.0], dtype=np.float64),
        cut_weight=np.array([10.0, 40.0, 0.0], dtype=np.float64),
        split_delta_q_base=np.array([2.0, -1.0, 0.0], dtype=np.float64),
        split_delta_q_probe=np.array([8.0, 5.0, 0.0], dtype=np.float64),
        hysteresis_only=np.array([False, True, False], dtype=bool),
    )


def _split_merge_repair_probes() -> RustSplitMergeRepairProbes:
    return RustSplitMergeRepairProbes(
        cluster=np.array([10, 20], dtype=np.uint64),
        gamma_multiplier=np.array([1.05, 1.10], dtype=np.float64),
        probe_resolution=np.array([0.008925, 0.00935], dtype=np.float64),
        block_count=np.array([100, 80], dtype=np.uint64),
        doc_weight=np.array([300.0, 120.0], dtype=np.float64),
        n_parts=np.array([5, 3], dtype=np.uint64),
        core_part_count=np.array([2, 1], dtype=np.uint64),
        singleton_weight=np.array([20.0, 50.0], dtype=np.float64),
        cut_weight=np.array([10.0, 8.0], dtype=np.float64),
        split_delta_q_base=np.array([-5.0, -3.0], dtype=np.float64),
        split_delta_q_probe=np.array([4.0, 2.0], dtype=np.float64),
        repair_merge_count=np.array([3, 1], dtype=np.uint64),
        repair_delta_q=np.array([8.0, 1.0], dtype=np.float64),
        net_delta_q=np.array([3.0, -2.0], dtype=np.float64),
        final_source_units=np.array([2, 1], dtype=np.uint64),
        retained_source_units=np.array([1, 1], dtype=np.uint64),
        escaped_source_units=np.array([1, 0], dtype=np.uint64),
        escaped_source_weight=np.array([80.0, 0.0], dtype=np.float64),
        final_small_source_units=np.array([0, 1], dtype=np.uint64),
        final_small_source_weight=np.array([0.0, 10.0], dtype=np.float64),
        largest_source_unit_fraction=np.array([0.7, 1.0], dtype=np.float64),
        restored_source_cluster=np.array([False, True], dtype=bool),
    )


def test_summarize_cluster_graph_stats():
    summary = summarize_cluster_graph_stats(_stats(), min_weight=50.0, max_weight=250.0)

    assert summary["n_active_clusters"] == 3
    assert summary["n_below_min_weight"] == 1
    assert summary["n_above_max_weight"] == 1
    assert summary["n_within_weight_band"] == 1
    assert summary["n_merge_candidates"] == 2
    assert summary["n_positive_delta_candidates"] == 1
    assert summary["n_positive_and_band_improving_candidates"] == 1


def test_write_adaptive_refinement_report(tmp_path):
    paths = write_adaptive_refinement_report(
        _stats(),
        tmp_path,
        min_weight=50.0,
        max_weight=250.0,
        top_candidates=1,
    )

    summary = json.loads((tmp_path / "cluster_graph_summary.json").read_text())
    assert summary["n_merge_candidates"] == 2
    assert set(paths) == {"summary", "merge_candidates", "cluster_arrays"}

    candidate_lines = (tmp_path / "macro_merge_candidates.csv").read_text().splitlines()
    assert len(candidate_lines) == 2
    row = next(csv.DictReader(candidate_lines))
    assert row["rank"] == "1"
    assert row["source"] == "0"
    assert row["target"] == "1"
    assert float(row["source_doc_weight"]) == 20.0
    assert float(row["target_doc_weight"]) == 120.0
    assert int(row["source_degree"]) == 2
    assert int(row["target_degree"]) == 2
    assert float(row["source_leafness"]) == 0.7

    arrays = np.load(tmp_path / "cluster_graph_stats.npz")
    np.testing.assert_array_equal(arrays["block_count"], np.array([2, 3, 1], dtype=np.uint64))
    np.testing.assert_array_equal(arrays["candidate_source"], np.array([0, 1], dtype=np.uint64))
    np.testing.assert_array_equal(arrays["second_neighbor"], np.array([2, 2, -1], dtype=np.int64))


def test_simulate_macro_merge_policy_greedy_non_conflicting():
    stats = RustClusterGraphStats(
        block_count=np.array([2, 3, 1, 4], dtype=np.uint64),
        doc_weight=np.array([20.0, 120.0, 40.0, 100.0], dtype=np.float64),
        internal_weight=np.array([5.0, 20.0, 2.0, 10.0], dtype=np.float64),
        external_weight=np.array([10.0, 15.0, 8.0, 12.0], dtype=np.float64),
        degree=np.array([2, 2, 1, 1], dtype=np.uint64),
        top_neighbor=np.array([1, 0, 3, 2], dtype=np.int64),
        top_neighbor_weight=np.array([7.0, 7.0, 5.0, 5.0], dtype=np.float64),
        second_neighbor=np.array([2, 2, -1, -1], dtype=np.int64),
        second_neighbor_weight=np.array([3.0, 2.0, 0.0, 0.0], dtype=np.float64),
        neighbor_weight_ratio=np.array([3.0 / 7.0, 2.0 / 7.0, 0.0, 0.0], dtype=np.float64),
        conductance=np.array([0.5, 0.4, 0.9, 0.6], dtype=np.float64),
        leafness=np.array([0.7, 0.5, 0.2, 0.3], dtype=np.float64),
        band_distance=np.array([30.0, 0.0, 10.0, 0.0], dtype=np.float64),
        candidate_source=np.array([0, 1, 2], dtype=np.uint64),
        candidate_target=np.array([1, 2, 3], dtype=np.uint64),
        candidate_edge_weight=np.array([7.0, 6.0, 5.0], dtype=np.float64),
        candidate_delta_q=np.array([-1e-5, -2e-5, -1e-5], dtype=np.float64),
        candidate_merged_weight=np.array([140.0, 160.0, 140.0], dtype=np.float64),
        candidate_size_band_gain=np.array([30.0, 10.0, 10.0], dtype=np.float64),
    )

    result = simulate_macro_merge_policy(
        stats,
        MacroMergePolicy(name="test", epsilon=1e-4),
        min_weight=50.0,
        max_weight=250.0,
    )

    assert result.n_candidates_after_filters == 3
    assert result.n_selected == 2
    assert result.estimated_active_clusters_after == 2
    assert result.below_min_delta == -2
    assert result.within_band_delta == 0
    assert result.q_debt == 2e-5
    assert result.singleton_endpoint_pairs == 1


def test_simulate_macro_merge_policy_filters_singletons():
    result = simulate_macro_merge_policy(
        _stats(),
        MacroMergePolicy(
            name="no-singletons",
            epsilon=2.0,
            allow_singleton_endpoint=False,
        ),
        min_weight=50.0,
        max_weight=250.0,
    )

    assert result.n_candidates_after_filters == 1
    assert result.n_selected == 1
    assert result.singleton_endpoint_pairs == 0


def test_write_macro_merge_ensemble_report(tmp_path):
    paths = write_macro_merge_ensemble_report(
        _stats(),
        tmp_path,
        policies=[MacroMergePolicy(name="probe", epsilon=1.0)],
        min_weight=50.0,
        max_weight=250.0,
    )

    assert set(paths) == {"json", "csv"}
    data = json.loads((tmp_path / "macro_merge_policy_ensemble.json").read_text())
    assert data[0]["policy"]["name"] == "probe"
    rows = list(csv.DictReader((tmp_path / "macro_merge_policy_ensemble.csv").open()))
    assert rows[0]["policy"] == "probe"
    assert int(rows[0]["n_selected"]) == 1


def test_run_macro_merge_policy_ensemble_default_policies():
    results = run_macro_merge_policy_ensemble(_stats(), min_weight=50.0, max_weight=250.0)

    assert results
    assert all(result.policy.name for result in results)


def test_score_boundary_candidates():
    candidate_ids, scores = score_boundary_candidates(
        _stats(),
        BoundaryCandidatePolicy(
            name="boundary",
            min_block_count=2,
            min_doc_weight=10.0,
            min_degree=2,
            min_conductance=0.2,
            max_leafness=0.8,
            min_neighbor_weight_ratio=0.4,
        ),
    )

    np.testing.assert_array_equal(candidate_ids, np.array([1, 0]))
    assert scores[0] > scores[1]


def test_summarize_boundary_candidate_policies():
    rows = summarize_boundary_candidate_policies(
        _stats(),
        [
            BoundaryCandidatePolicy(
                name="boundary",
                min_block_count=2,
                min_doc_weight=10.0,
                min_degree=2,
                min_conductance=0.2,
                max_leafness=0.8,
                min_neighbor_weight_ratio=0.4,
            )
        ],
    )

    assert rows[0]["policy"]["name"] == "boundary"
    assert rows[0]["n_candidates_after_filters"] == 2
    assert rows[0]["n_exported"] == 2


def test_write_boundary_candidate_report(tmp_path):
    paths = write_boundary_candidate_report(
        _stats(),
        tmp_path,
        [
            BoundaryCandidatePolicy(
                name="boundary",
                min_block_count=2,
                min_doc_weight=10.0,
                min_degree=2,
                min_conductance=0.2,
                max_leafness=0.8,
                min_neighbor_weight_ratio=0.4,
            )
        ],
    )

    assert set(paths) == {"summary", "candidates"}
    data = json.loads((tmp_path / "boundary_candidate_policy_summary.json").read_text())
    assert data[0]["n_candidates_after_filters"] == 2
    rows = list(csv.DictReader((tmp_path / "boundary_candidates.csv").open()))
    assert rows[0]["policy"] == "boundary"
    assert rows[0]["cluster"] == "1"
    assert rows[0]["second_neighbor"] == "2"


def test_summarize_boundary_move_probes():
    summary = summarize_boundary_move_probes(_move_probes())

    assert summary["n_probes"] == 2
    assert summary["n_with_positive_moves"] == 1
    assert summary["n_with_near_neutral_moves"] == 2
    assert summary["total_positive_move_count"] == 2
    assert summary["total_positive_delta_q"] == 1.5


def test_write_boundary_move_probe_report(tmp_path):
    paths = write_boundary_move_probe_report(_move_probes(), tmp_path)

    assert set(paths) == {"summary", "probes"}
    summary = json.loads((tmp_path / "boundary_move_probe_summary.json").read_text())
    assert summary["n_with_positive_moves"] == 1
    rows = list(csv.DictReader((tmp_path / "boundary_move_probes.csv").open()))
    assert rows[0]["cluster"] == "10"
    assert rows[0]["best_move_node"] == "7"
    assert rows[1]["cluster"] == "20"


def test_summarize_boundary_group_probes():
    summary = summarize_boundary_group_probes(_group_probes())

    assert summary["n_probes"] == 2
    assert summary["n_positive_best"] == 1
    assert summary["n_positive_top_group_move"] == 1
    assert summary["best_action_counts"]["top_move"] == 1
    assert summary["best_action_counts"]["none"] == 1


def test_write_boundary_group_probe_report(tmp_path):
    paths = write_boundary_group_probe_report(_group_probes(), tmp_path)

    assert set(paths) == {"summary", "probes"}
    summary = json.loads((tmp_path / "boundary_group_probe_summary.json").read_text())
    assert summary["n_positive_best"] == 1
    rows = list(csv.DictReader((tmp_path / "boundary_group_probes.csv").open()))
    assert rows[0]["cluster"] == "10"
    assert rows[0]["best_action"] == "1"
    assert rows[1]["cluster"] == "20"
    assert rows[1]["best_action"] == "0"


def test_summarize_multi_core_split_probes():
    summary = summarize_multi_core_split_probes(_multi_core_split_probes())

    assert summary["n_probes"] == 3
    assert summary["n_split"] == 2
    assert summary["n_base_positive"] == 1
    assert summary["n_probe_positive"] == 2
    assert summary["n_hysteresis_only"] == 1
    assert summary["n_meaningful_core_split"] == 2


def test_write_multi_core_split_probe_report(tmp_path):
    paths = write_multi_core_split_probe_report(_multi_core_split_probes(), tmp_path)

    assert set(paths) == {"summary", "probes"}
    summary = json.loads((tmp_path / "multi_core_split_probe_summary.json").read_text())
    assert summary["n_hysteresis_only"] == 1
    rows = list(csv.DictReader((tmp_path / "multi_core_split_probes.csv").open()))
    assert rows[0]["cluster"] == "10"
    assert rows[0]["split_delta_q_base"] == "2.0"
    assert any(row["hysteresis_only"] == "True" for row in rows)


def test_summarize_split_merge_repair_probes():
    summary = summarize_split_merge_repair_probes(_split_merge_repair_probes())

    assert summary["n_probes"] == 2
    assert summary["n_net_positive"] == 1
    assert summary["n_with_repair_merges"] == 2
    assert summary["n_with_escaped_source"] == 1
    assert summary["n_restored_source_cluster"] == 1


def test_write_split_merge_repair_probe_report(tmp_path):
    paths = write_split_merge_repair_probe_report(_split_merge_repair_probes(), tmp_path)

    assert set(paths) == {"summary", "probes"}
    summary = json.loads((tmp_path / "split_merge_repair_probe_summary.json").read_text())
    assert summary["n_net_positive"] == 1
    rows = list(csv.DictReader((tmp_path / "split_merge_repair_probes.csv").open()))
    assert rows[0]["cluster"] == "10"
    assert rows[0]["net_delta_q"] == "3.0"
    assert rows[0]["escaped_source_units"] == "1"
