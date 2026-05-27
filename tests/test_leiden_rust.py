"""Tests for the Rust Leiden Python wrapper."""

import json

import numpy as np
import polars as pl
import pytest

import sciscape.clustering.leiden_rust as leiden_rust
from sciscape.clustering.leiden_rust import (
    RUST_AVAILABLE,
    RUST_DONGDAEMUN_AVAILABLE,
    RUST_DONGDAEMUN_REFINEMENT_AVAILABLE,
    build_leiden_graph,
    dongdaemun_refine_rust,
    postprocess_small_clusters_rust,
    project_membership_rust,
    remap_parquet_to_leiden_graph,
    run_leiden_rust,
)

pytestmark = pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust backend required")
pytestmark_dongdaemun = pytest.mark.skipif(
    not RUST_DONGDAEMUN_AVAILABLE,
    reason="Rust Dongdaemun binding required",
)
pytestmark_dongdaemun_refinement = pytest.mark.skipif(
    not RUST_DONGDAEMUN_REFINEMENT_AVAILABLE,
    reason="Rust Dongdaemun refinement binding required",
)


def _two_clique_edges():
    src = []
    dst = []
    w = []
    for offset in (0, 4):
        for i in range(4):
            for j in range(i + 1, 4):
                src.append(offset + i)
                dst.append(offset + j)
                w.append(1.0)
    src.append(0)
    dst.append(4)
    w.append(0.01)
    return (
        np.asarray(src, dtype=np.uint32),
        np.asarray(dst, dtype=np.uint32),
        np.asarray(w, dtype=np.float64),
    )


def test_cached_graph_matches_run_leiden_wrapper():
    src, dst, w = _two_clique_edges()
    graph = build_leiden_graph(
        edges_src=src,
        edges_dst=dst,
        edges_weight=w,
        n_nodes=8,
    )

    cached = graph.run_leiden(resolution=0.1, seed=7, n_iterations=3)
    wrapper = run_leiden_rust(
        edges_src=src,
        edges_dst=dst,
        edges_weight=w,
        n_nodes=8,
        resolution=0.1,
        seed=7,
        n_iterations=3,
    )

    assert graph.n_nodes == 8
    assert graph.n_edges == len(src) * 2
    assert cached.n_clusters == wrapper.n_clusters
    np.testing.assert_array_equal(cached.membership, wrapper.membership)


def test_cached_graph_run_leiden_can_return_uint32_membership():
    src, dst, w = _two_clique_edges()
    graph = build_leiden_graph(
        edges_src=src,
        edges_dst=dst,
        edges_weight=w,
        n_nodes=8,
    )

    as_u64 = graph.run_leiden(resolution=0.1, seed=7, n_iterations=3)
    as_u32 = graph.run_leiden(
        resolution=0.1,
        seed=7,
        n_iterations=3,
        membership_dtype=np.uint32,
    )

    assert as_u32.membership.dtype == np.uint32
    assert as_u32.n_clusters == as_u64.n_clusters
    assert as_u32.quality == pytest.approx(as_u64.quality)
    np.testing.assert_array_equal(
        as_u32.membership.astype(np.uint64), as_u64.membership
    )


def test_cached_graph_randomness_schedule_matches_wrapper():
    src, dst, w = _two_clique_edges()
    graph = build_leiden_graph(
        edges_src=src,
        edges_dst=dst,
        edges_weight=w,
        n_nodes=8,
    )
    schedule = [0.02, 0.01, 0.005]

    cached = graph.run_leiden(
        resolution=0.1,
        seed=7,
        n_iterations=3,
        randomness_schedule=schedule,
    )
    wrapper = run_leiden_rust(
        edges_src=src,
        edges_dst=dst,
        edges_weight=w,
        n_nodes=8,
        resolution=0.1,
        seed=7,
        n_iterations=3,
        randomness_schedule=schedule,
    )

    assert cached.n_clusters == wrapper.n_clusters
    assert cached.quality == pytest.approx(wrapper.quality)
    np.testing.assert_array_equal(cached.membership, wrapper.membership)


def test_cached_graph_cpm_quality():
    graph = build_leiden_graph(
        edges_src=np.asarray([0, 1, 2], dtype=np.uint32),
        edges_dst=np.asarray([1, 2, 0], dtype=np.uint32),
        edges_weight=np.asarray([1.0, 1.0, 1.0], dtype=np.float64),
        n_nodes=3,
    )

    quality = graph.cpm_quality(
        np.asarray([0, 0, 0], dtype=np.uint64),
        resolution=0.5,
    )

    assert quality == pytest.approx(1.5)


def test_cached_graph_cluster_graph_stats():
    graph = build_leiden_graph(
        edges_src=np.asarray([0, 0, 1, 2], dtype=np.uint32),
        edges_dst=np.asarray([1, 2, 3, 3], dtype=np.uint32),
        edges_weight=np.asarray([2.0, 0.5, 0.5, 3.0], dtype=np.float64),
        n_nodes=4,
    )
    membership = np.asarray([0, 0, 1, 1], dtype=np.uint64)

    stats = graph.cluster_graph_stats(
        membership,
        resolution=0.1,
        min_weight=3.0,
        max_weight=10.0,
        top_k=4,
    )

    np.testing.assert_array_equal(
        stats.block_count, np.asarray([2, 2], dtype=np.uint64)
    )
    np.testing.assert_allclose(stats.doc_weight, np.asarray([2.0, 2.0]))
    np.testing.assert_allclose(stats.internal_weight, np.asarray([2.0, 3.0]))
    np.testing.assert_allclose(stats.external_weight, np.asarray([1.0, 1.0]))
    np.testing.assert_array_equal(
        stats.top_neighbor, np.asarray([1, 0], dtype=np.int64)
    )
    np.testing.assert_allclose(stats.band_distance, np.asarray([1.0, 1.0]))
    assert stats.n_candidates == 1
    assert int(stats.candidate_source[0]) == 0
    assert int(stats.candidate_target[0]) == 1
    assert float(stats.candidate_delta_q[0]) == pytest.approx(0.6)
    assert float(stats.candidate_size_band_gain[0]) == pytest.approx(2.0)


def test_cached_graph_external_grain_probes_returns_selection_arrays():
    graph = build_leiden_graph(
        edges_src=np.asarray([0, 0, 1], dtype=np.uint32),
        edges_dst=np.asarray([1, 2, 3], dtype=np.uint32),
        edges_weight=np.asarray([0.1, 10.0, 9.0], dtype=np.float64),
        n_nodes=4,
    )
    membership = np.asarray([0, 0, 1, 2], dtype=np.uint64)

    probes = graph.external_grain_probes(
        membership,
        np.asarray([0], dtype=np.uint64),
        resolution=0.1,
    )

    assert probes.n_probes == 1
    assert bool(probes.recommended_for_split_repair[0]) is True
    assert probes.recommended_for_split_repair.dtype == np.bool_
    np.testing.assert_allclose(
        probes.priority,
        probes.best_group_delta_q
        / np.maximum(probes.incident_directed_edges.astype(np.float64), 1.0),
    )

    filtered = graph.external_grain_probes(
        membership,
        np.asarray([0], dtype=np.uint64),
        resolution=0.1,
        min_doc_weight=3.0,
    )
    assert bool(filtered.recommended_for_split_repair[0]) is False
    np.testing.assert_allclose(filtered.priority, probes.priority)


def test_cached_graph_external_grain_priority_clusters_accepts_uint32_membership():
    graph = build_leiden_graph(
        edges_src=np.asarray([0, 0, 1], dtype=np.uint32),
        edges_dst=np.asarray([1, 2, 3], dtype=np.uint32),
        edges_weight=np.asarray([0.1, 10.0, 9.0], dtype=np.float64),
        n_nodes=4,
    )
    membership_u64 = np.asarray([0, 0, 1, 2], dtype=np.uint64)
    membership_u32 = membership_u64.astype(np.uint32)
    candidate_clusters = np.asarray([0, 1, 2], dtype=np.uint64)

    as_u64 = graph.external_grain_priority_clusters(
        membership_u64,
        candidate_clusters,
        resolution=0.1,
        count=2,
    )
    as_u32 = graph.external_grain_priority_clusters(
        membership_u32,
        candidate_clusters,
        resolution=0.1,
        count=2,
    )

    assert as_u32 == as_u64
    assert len(as_u32) == 2


def test_cached_graph_non_monotone_group_escape_accepts_non_loss_candidate():
    graph = build_leiden_graph(
        edges_src=np.asarray([0, 1], dtype=np.uint32),
        edges_dst=np.asarray([1, 2], dtype=np.uint32),
        edges_weight=np.asarray([0.1, 10.0], dtype=np.float64),
        n_nodes=3,
    )
    membership = np.asarray([0, 0, 1], dtype=np.uint64)

    result = graph.non_monotone_group_escape_probe(
        membership,
        np.asarray([0], dtype=np.uint64),
        resolution=0.1,
        max_candidates=3,
        polish_iterations=0,
        randomness=0.0,
        seed=11,
    )

    assert result.accepted is True
    assert result.quality >= result.baseline_quality
    assert result.best_delta_q > 0.0
    np.testing.assert_array_equal(
        result.membership, np.asarray([0, 1, 1], dtype=np.uint64)
    )
    assert result.candidate_rows
    assert result.candidate_rows[0]["accepted_by_quality"] is True
    assert result.candidate_rows[0]["pre_polish_delta_q"] > 0.0


def test_cached_graph_non_monotone_group_escape_rejects_loss_candidate():
    graph = build_leiden_graph(
        edges_src=np.asarray([0, 1], dtype=np.uint32),
        edges_dst=np.asarray([1, 2], dtype=np.uint32),
        edges_weight=np.asarray([10.0, 0.1], dtype=np.float64),
        n_nodes=3,
    )
    membership = np.asarray([0, 0, 1], dtype=np.uint64)

    result = graph.non_monotone_group_escape_probe(
        membership,
        np.asarray([0], dtype=np.uint64),
        resolution=0.1,
        max_candidates=3,
        polish_iterations=0,
        randomness=0.0,
        seed=11,
    )

    assert result.accepted is False
    assert result.quality == pytest.approx(result.baseline_quality)
    assert result.best_delta_q < 0.0
    np.testing.assert_array_equal(result.membership, membership)
    assert result.candidate_rows
    assert result.candidate_rows[0]["accepted_by_quality"] is False
    assert result.candidate_rows[0]["post_polish_delta_q"] < 0.0


def test_cached_graph_non_monotone_group_escape_max_candidates_zero_noop():
    graph = build_leiden_graph(
        edges_src=np.asarray([0, 1], dtype=np.uint32),
        edges_dst=np.asarray([1, 2], dtype=np.uint32),
        edges_weight=np.asarray([0.1, 10.0], dtype=np.float64),
        n_nodes=3,
    )
    membership = np.asarray([0, 0, 1], dtype=np.uint64)

    result = graph.non_monotone_group_escape_probe(
        membership,
        np.asarray([0], dtype=np.uint64),
        resolution=0.1,
        max_candidates=0,
        polish_iterations=5,
        randomness=0.0,
        seed=11,
    )

    assert result.accepted is False
    assert result.quality == pytest.approx(result.baseline_quality)
    assert result.best_delta_q == pytest.approx(0.0)
    assert result.candidate_rows == []
    np.testing.assert_array_equal(result.membership, membership)


def test_cached_graph_non_monotone_group_escape_can_skip_return_membership():
    graph = build_leiden_graph(
        edges_src=np.asarray([0, 1], dtype=np.uint32),
        edges_dst=np.asarray([1, 2], dtype=np.uint32),
        edges_weight=np.asarray([0.1, 10.0], dtype=np.float64),
        n_nodes=3,
    )
    membership = np.asarray([0, 0, 1], dtype=np.uint64)
    kwargs = {
        "resolution": 0.1,
        "max_candidates": 3,
        "polish_iterations": 0,
        "randomness": 0.0,
        "seed": 11,
    }

    with_membership = graph.non_monotone_group_escape_probe(
        membership,
        np.asarray([0], dtype=np.uint64),
        **kwargs,
    )
    without_membership = graph.non_monotone_group_escape_probe(
        membership,
        np.asarray([0], dtype=np.uint64),
        return_membership=False,
        **kwargs,
    )

    assert without_membership.membership.dtype == np.uint64
    assert without_membership.membership.shape == (0,)
    assert without_membership.accepted == with_membership.accepted
    assert without_membership.quality == pytest.approx(with_membership.quality)
    assert without_membership.best_delta_q == pytest.approx(
        with_membership.best_delta_q
    )
    assert len(without_membership.candidate_rows) == len(with_membership.candidate_rows)


def test_cached_graph_non_monotone_group_escape_accepts_uint32_membership():
    graph = build_leiden_graph(
        edges_src=np.asarray([0, 1], dtype=np.uint32),
        edges_dst=np.asarray([1, 2], dtype=np.uint32),
        edges_weight=np.asarray([0.1, 10.0], dtype=np.float64),
        n_nodes=3,
    )
    membership_u64 = np.asarray([0, 0, 1], dtype=np.uint64)
    membership_u32 = membership_u64.astype(np.uint32)
    kwargs = {
        "resolution": 0.1,
        "max_candidates": 3,
        "polish_iterations": 0,
        "randomness": 0.0,
        "seed": 11,
        "return_membership": False,
    }

    as_u64 = graph.non_monotone_group_escape_probe(
        membership_u64,
        np.asarray([0], dtype=np.uint64),
        **kwargs,
    )
    as_u32 = graph.non_monotone_group_escape_probe(
        membership_u32,
        np.asarray([0], dtype=np.uint64),
        **kwargs,
    )

    assert as_u32.membership.shape == (0,)
    assert as_u32.accepted == as_u64.accepted
    assert as_u32.quality == pytest.approx(as_u64.quality)
    assert as_u32.best_delta_q == pytest.approx(as_u64.best_delta_q)
    assert len(as_u32.candidate_rows) == len(as_u64.candidate_rows)


def test_cached_graph_non_monotone_group_escape_parallel_matches_serial():
    graph = build_leiden_graph(
        edges_src=np.asarray([0, 1, 1, 3], dtype=np.uint32),
        edges_dst=np.asarray([1, 2, 3, 4], dtype=np.uint32),
        edges_weight=np.asarray([0.1, 7.0, 6.0, 0.2], dtype=np.float64),
        n_nodes=5,
    )
    membership = np.asarray([0, 0, 1, 2, 2], dtype=np.uint64)
    kwargs = {
        "resolution": 0.1,
        "max_candidates": 5,
        "polish_iterations": 1,
        "randomness": 0.0,
        "seed": 99,
    }

    serial = graph.non_monotone_group_escape_probe(
        membership,
        np.asarray([0, 2], dtype=np.uint64),
        parallel_candidates=False,
        **kwargs,
    )
    parallel = graph.non_monotone_group_escape_probe(
        membership,
        np.asarray([0, 2], dtype=np.uint64),
        parallel_candidates=True,
        **kwargs,
    )

    serial_keys = [
        (
            row["candidate_index"],
            row["source_cluster"],
            row["target_cluster"],
            row["group_kind"],
        )
        for row in serial.candidate_rows
    ]
    parallel_keys = [
        (
            row["candidate_index"],
            row["source_cluster"],
            row["target_cluster"],
            row["group_kind"],
        )
        for row in parallel.candidate_rows
    ]
    assert serial_keys == parallel_keys
    assert parallel.quality == pytest.approx(serial.quality)
    assert parallel.best_delta_q == pytest.approx(serial.best_delta_q)
    np.testing.assert_array_equal(parallel.membership, serial.membership)
    assert parallel.candidate_eval_cpu_sum_elapsed_ms >= 0.0
    assert parallel.candidate_eval_wall_elapsed_ms >= 0.0
    if len(parallel.candidate_rows) > 1:
        assert parallel.candidate_eval_parallel is True
        assert parallel.candidate_eval_parallel_workers >= 1


def test_trajectory_local_move_focus_trace_filters_nodes_and_roles(
    tmp_path, monkeypatch
):
    trace_path = tmp_path / "trajectory_trace.jsonl"
    monkeypatch.setenv("SCISCAPE_DDM_TRAJECTORY_TRACE_PATH", str(trace_path))
    monkeypatch.setenv("SCISCAPE_DDM_TRAJECTORY_TRACE_RUN_ID", "focus-run")
    monkeypatch.setenv("SCISCAPE_DDM_TRAJECTORY_TRACE_EPOCH", "focus-epoch")
    monkeypatch.setenv("SCISCAPE_DDM_LOCAL_MOVE_FOCUS_NODES", "0,2")
    monkeypatch.setenv("SCISCAPE_DDM_LOCAL_MOVE_NEIGHBOR_NODES", "1,2,bad")
    graph = build_leiden_graph(
        edges_src=np.asarray([0, 1, 2], dtype=np.uint32),
        edges_dst=np.asarray([1, 2, 3], dtype=np.uint32),
        edges_weight=np.asarray([5.0, 5.0, 5.0], dtype=np.float64),
        n_nodes=4,
    )

    graph.run_leiden(resolution=0.1, seed=7, n_iterations=1, randomness=0.0)

    events = [json.loads(line) for line in trace_path.read_text().splitlines()]
    focus_events = [
        event
        for event in events
        if event["event"] == "local_move_focus_node"
        and event["run_id"] == "focus-run"
        and event["depth"] == 0
    ]
    assert focus_events
    assert {event["node"] for event in focus_events} <= {0, 1, 2}
    assert not any(event["node"] == 3 for event in focus_events)
    assert any(
        event["node"] == 2 and event["role"] == "target" for event in focus_events
    )
    assert not any(
        event["node"] == 2 and event["role"] == "neighbor" for event in focus_events
    )
    assert {
        "current_cluster",
        "best_cluster",
        "best_increment",
        "margin",
        "moved",
    } <= set(focus_events[0])


def test_cached_graph_non_monotone_group_escape_multifidelity_returns_rows():
    graph = build_leiden_graph(
        edges_src=np.asarray([0, 1], dtype=np.uint32),
        edges_dst=np.asarray([1, 2], dtype=np.uint32),
        edges_weight=np.asarray([0.1, 10.0], dtype=np.float64),
        n_nodes=3,
    )
    membership = np.asarray([0, 0, 1], dtype=np.uint64)

    result = graph.non_monotone_group_escape_multifidelity_probe(
        membership,
        np.asarray([0], dtype=np.uint64),
        resolution=0.1,
        max_candidates=3,
        prescreen_iterations=0,
        final_iterations=0,
        finalists=1,
        label_full_p5=True,
        randomness=0.0,
        seed=11,
    )

    assert result.selected_policy == "p1_top1_then_p5"
    assert result.accepted is True
    assert result.candidate_rows
    assert result.policy_rows
    row = result.candidate_rows[0]
    assert {"pre_delta_q", "p1_delta_q", "p5_delta_q", "selected_by_p1_top1"} <= set(
        row
    )
    policies = {row["policy"]: row for row in result.policy_rows}
    assert policies["full_top3_p5"]["available"] is True
    assert policies["p1_top1_then_p5"]["p1_evaluated"] == len(result.candidate_rows)
    np.testing.assert_array_equal(
        result.membership, np.asarray([0, 1, 1], dtype=np.uint64)
    )


def test_cached_graph_non_monotone_group_escape_multifidelity_returns_approx_labels():
    graph = build_leiden_graph(
        edges_src=np.asarray([0, 1, 1, 2], dtype=np.uint32),
        edges_dst=np.asarray([1, 2, 3, 3], dtype=np.uint32),
        edges_weight=np.asarray([0.1, 10.0, 8.0, 1.0], dtype=np.float64),
        n_nodes=4,
    )
    membership = np.asarray([0, 0, 1, 1], dtype=np.uint64)

    result = graph.non_monotone_group_escape_multifidelity_probe(
        membership,
        np.asarray([0], dtype=np.uint64),
        resolution=0.1,
        max_candidates=3,
        prescreen_iterations=0,
        final_iterations=1,
        finalists=1,
        label_full_p5=True,
        randomness=0.0,
        seed=11,
        return_membership=False,
        approx_polish_labels=True,
    )

    assert result.membership.shape == (0,)
    assert result.candidate_rows
    row = result.candidate_rows[0]
    assert {
        "localized_quality",
        "localized_delta_q",
        "localized_elapsed_ms",
        "localized_active_nodes",
        "localized_active_clusters",
        "localized_rank",
        "quotient_quality",
        "quotient_delta_q",
        "quotient_elapsed_ms",
        "quotient_supernodes",
        "quotient_rank",
        "ub_delta_q",
        "ub_elapsed_ms",
        "ub_covers_p5",
        "ub_violation",
        "ub_rank",
    } <= set(row)
    assert np.isfinite(row["localized_delta_q"])
    assert np.isfinite(row["quotient_delta_q"])
    assert np.isfinite(row["ub_delta_q"])
    assert row["localized_rank"] >= 1
    assert row["quotient_rank"] >= 1
    assert row["ub_rank"] >= 1


def test_cached_graph_non_monotone_group_escape_multifidelity_returns_basin_signatures():
    graph = build_leiden_graph(
        edges_src=np.asarray([0, 1], dtype=np.uint32),
        edges_dst=np.asarray([1, 2], dtype=np.uint32),
        edges_weight=np.asarray([0.1, 10.0], dtype=np.float64),
        n_nodes=3,
    )
    membership = np.asarray([0, 0, 1], dtype=np.uint64)

    result = graph.non_monotone_group_escape_multifidelity_probe(
        membership,
        np.asarray([0], dtype=np.uint64),
        resolution=0.05,
        max_candidates=3,
        prescreen_iterations=0,
        final_iterations=0,
        finalists=1,
        label_full_p5=True,
        randomness=0.0,
        seed=11,
        return_membership=False,
        basin_signatures=True,
    )

    assert result.membership.shape == (0,)
    assert result.candidate_rows
    row = result.candidate_rows[0]
    assert {
        "p5_basin_signature",
        "p5_basin_cluster_count",
        "p5_changed_nodes_vs_baseline",
        "p5_baseline_fragmentation_nodes",
        "p5_baseline_mixing_nodes",
        "p5_changed_fraction_vs_baseline",
        "p5_relative_delta_q_ppm",
        "p5_basin_sketch_sample_size",
        "p5_basin_sketch_node_hash",
        "p5_basin_sketch_baseline_membership",
        "p5_basin_sketch_membership",
        "p5_basin_changed_support_node_count",
        "p5_basin_changed_support_sketch_sample_size",
        "p5_basin_changed_support_node_hash",
        "p5_basin_changed_support_nodes",
    } <= set(row)
    assert row["p5_basin_signature"]
    assert row["p5_basin_cluster_count"] == 2
    assert row["p5_changed_nodes_vs_baseline"] == 1
    assert row["p5_baseline_fragmentation_nodes"] == 1
    assert row["p5_baseline_mixing_nodes"] == 1
    assert row["p5_changed_fraction_vs_baseline"] == pytest.approx(1 / 3)
    assert np.isfinite(row["p5_relative_delta_q_ppm"])
    assert row["p5_basin_sketch_sample_size"] == 3
    assert row["p5_basin_sketch_node_hash"]
    assert row["p5_basin_sketch_baseline_membership"] == "0;0;1"
    assert row["p5_basin_sketch_membership"] == "0;1;1"
    assert row["p5_basin_changed_support_node_count"] == 2
    assert row["p5_basin_changed_support_sketch_sample_size"] == 2
    assert row["p5_basin_changed_support_node_hash"]
    assert row["p5_basin_changed_support_nodes"] == "1;2"


def test_cached_graph_non_monotone_group_escape_multifidelity_can_skip_return_membership():
    graph = build_leiden_graph(
        edges_src=np.asarray([0, 1], dtype=np.uint32),
        edges_dst=np.asarray([1, 2], dtype=np.uint32),
        edges_weight=np.asarray([0.1, 10.0], dtype=np.float64),
        n_nodes=3,
    )
    membership = np.asarray([0, 0, 1], dtype=np.uint64)
    kwargs = {
        "resolution": 0.1,
        "max_candidates": 3,
        "prescreen_iterations": 0,
        "final_iterations": 0,
        "finalists": 1,
        "label_full_p5": True,
        "randomness": 0.0,
        "seed": 11,
    }

    with_membership = graph.non_monotone_group_escape_multifidelity_probe(
        membership,
        np.asarray([0], dtype=np.uint64),
        **kwargs,
    )
    without_membership = graph.non_monotone_group_escape_multifidelity_probe(
        membership,
        np.asarray([0], dtype=np.uint64),
        return_membership=False,
        **kwargs,
    )

    assert without_membership.membership.dtype == np.uint64
    assert without_membership.membership.shape == (0,)
    assert without_membership.accepted == with_membership.accepted
    assert without_membership.quality == pytest.approx(with_membership.quality)
    assert without_membership.selected_policy == with_membership.selected_policy
    assert (
        without_membership.selected_candidate_index
        == with_membership.selected_candidate_index
    )
    assert len(without_membership.candidate_rows) == len(with_membership.candidate_rows)
    assert len(without_membership.policy_rows) == len(with_membership.policy_rows)


def test_cached_graph_non_monotone_group_escape_multifidelity_accepts_uint32_membership():
    graph = build_leiden_graph(
        edges_src=np.asarray([0, 1], dtype=np.uint32),
        edges_dst=np.asarray([1, 2], dtype=np.uint32),
        edges_weight=np.asarray([0.1, 10.0], dtype=np.float64),
        n_nodes=3,
    )
    membership_u64 = np.asarray([0, 0, 1], dtype=np.uint64)
    membership_u32 = membership_u64.astype(np.uint32)
    kwargs = {
        "resolution": 0.1,
        "max_candidates": 3,
        "prescreen_iterations": 0,
        "final_iterations": 0,
        "finalists": 1,
        "label_full_p5": True,
        "randomness": 0.0,
        "seed": 11,
        "return_membership": False,
    }

    as_u64 = graph.non_monotone_group_escape_multifidelity_probe(
        membership_u64,
        np.asarray([0], dtype=np.uint64),
        **kwargs,
    )
    as_u32 = graph.non_monotone_group_escape_multifidelity_probe(
        membership_u32,
        np.asarray([0], dtype=np.uint64),
        **kwargs,
    )

    assert as_u32.membership.shape == (0,)
    assert as_u32.accepted == as_u64.accepted
    assert as_u32.quality == pytest.approx(as_u64.quality)
    assert as_u32.selected_policy == as_u64.selected_policy
    assert as_u32.selected_candidate_index == as_u64.selected_candidate_index
    assert len(as_u32.candidate_rows) == len(as_u64.candidate_rows)
    assert len(as_u32.policy_rows) == len(as_u64.policy_rows)


def test_cached_graph_apply_split_repair_candidates_moves_escaped_fragment():
    graph = build_leiden_graph(
        edges_src=np.asarray([0, 1], dtype=np.uint32),
        edges_dst=np.asarray([1, 2], dtype=np.uint32),
        edges_weight=np.asarray([0.1, 10.0], dtype=np.float64),
        n_nodes=3,
    )
    membership = np.asarray([0, 0, 1], dtype=np.uint64)

    result = graph.apply_split_merge_repair_candidates(
        membership,
        np.asarray([0], dtype=np.uint64),
        np.asarray([0], dtype=np.uint64),
        [10.0],
        resolution=0.1,
        gamma_multipliers=[10.0],
        min_core_weight=1.0,
        randomness=0.0,
        seed=42,
    )

    np.testing.assert_array_equal(
        result.membership, np.asarray([0, 1, 1], dtype=np.uint64)
    )
    assert result.n_applied == 1
    assert int(result.changed_nodes[0]) == 1
    assert int(result.moved_to_existing_cluster_nodes[0]) == 1
    exact_delta = graph.cpm_quality(
        result.membership, resolution=0.1
    ) - graph.cpm_quality(
        membership,
        resolution=0.1,
    )
    assert exact_delta == pytest.approx(float(result.predicted_net_delta_q.sum()))


def test_cached_graph_pair_seeded_apply_matches_selected_probe():
    graph = build_leiden_graph(
        edges_src=np.asarray([0, 1, 1, 2, 3], dtype=np.uint32),
        edges_dst=np.asarray([1, 2, 3, 3, 4], dtype=np.uint32),
        edges_weight=np.asarray([0.1, 10.0, 0.5, 0.2, 8.0], dtype=np.float64),
        n_nodes=5,
    )
    membership = np.asarray([0, 0, 0, 1, 1], dtype=np.uint64)
    gamma_multipliers = [1.05, 10.0]

    probes = graph.split_merge_repair_probes(
        membership,
        np.asarray([0], dtype=np.uint64),
        resolution=0.1,
        gamma_multipliers=gamma_multipliers,
        min_core_weight=1.0,
        randomness=0.01,
        seed=42,
        pair_seeded=True,
    )
    selected = np.flatnonzero(np.isclose(probes.gamma_multiplier, 10.0))[0]
    result = graph.apply_split_merge_repair_candidates(
        membership,
        np.asarray([0], dtype=np.uint64),
        np.asarray([0], dtype=np.uint64),
        [10.0],
        resolution=0.1,
        gamma_multipliers=gamma_multipliers,
        min_core_weight=1.0,
        randomness=0.01,
        seed=42,
        pair_seeded=True,
    )

    assert result.n_applied == 1
    assert int(result.n_parts[0]) == int(probes.n_parts[selected])
    assert float(result.split_delta_q_base[0]) == pytest.approx(
        float(probes.split_delta_q_base[selected])
    )
    assert float(result.repair_delta_q[0]) == pytest.approx(
        float(probes.repair_delta_q[selected])
    )
    assert float(result.predicted_net_delta_q[0]) == pytest.approx(
        float(probes.net_delta_q[selected])
    )


@pytestmark_dongdaemun
def test_cached_graph_dongdaemun_refine_commits_escaped_fragment():
    graph = build_leiden_graph(
        edges_src=np.asarray([0, 1], dtype=np.uint32),
        edges_dst=np.asarray([1, 2], dtype=np.uint32),
        edges_weight=np.asarray([0.1, 10.0], dtype=np.float64),
        n_nodes=3,
    )
    membership = np.asarray([0, 0, 1], dtype=np.uint64)

    result = graph.dongdaemun_refine(
        membership,
        resolution=0.1,
        target_max_weight=1.5,
        gamma_multipliers=[10.0],
        min_core_weight=1.0,
        randomness=0.0,
        repair_epsilon=0.0,
        apply_iterations=1,
        seed=42,
        pair_seeded=False,
    )

    np.testing.assert_array_equal(
        result.membership, np.asarray([0, 1, 1], dtype=np.uint64)
    )
    assert result.n_clusters == 2
    assert result.diagnostic_membership is None
    assert result.audit.accepted is True
    assert result.audit.status == "committed"
    assert result.audit.split_iteration.tolist() == [1]
    assert result.audit.split_n_selected.tolist() == [1]
    assert result.audit.split_n_applied.tolist() == [1]
    assert result.audit.effective_delta_q > 0.0


@pytestmark_dongdaemun
def test_dongdaemun_refine_rust_standalone_builds_graph_and_commits():
    membership = np.asarray([0, 0, 1], dtype=np.uint64)

    result = dongdaemun_refine_rust(
        membership=membership,
        edges_src=np.asarray([0, 1], dtype=np.uint32),
        edges_dst=np.asarray([1, 2], dtype=np.uint32),
        edges_weight=np.asarray([0.1, 10.0], dtype=np.float64),
        n_nodes=3,
        resolution=0.1,
        target_max_weight=1.5,
        gamma_multipliers=[10.0],
        min_core_weight=1.0,
        randomness=0.0,
        repair_epsilon=0.0,
        apply_iterations=1,
        seed=42,
        pair_seeded=False,
    )

    np.testing.assert_array_equal(
        result.membership, np.asarray([0, 1, 1], dtype=np.uint64)
    )
    assert result.audit.status == "committed"
    assert result.audit.accepted is True


def test_dongdaemun_refine_rust_reports_stale_binding(monkeypatch):
    monkeypatch.setattr(leiden_rust, "RUST_DONGDAEMUN_AVAILABLE", False)

    with pytest.raises(AttributeError, match="maturin develop"):
        leiden_rust.dongdaemun_refine_rust(
            membership=np.asarray([0], dtype=np.uint64),
            resolution=0.1,
            target_max_weight=1.0,
        )


@pytestmark_dongdaemun_refinement
def test_cached_graph_run_leiden_dongdaemun_refinement_reports_audit():
    graph = build_leiden_graph(
        edges_src=np.asarray([0, 2, 0, 0, 1, 1], dtype=np.uint32),
        edges_dst=np.asarray([1, 3, 2, 3, 2, 3], dtype=np.uint32),
        edges_weight=np.asarray([10.0, 10.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64),
        n_nodes=4,
    )
    result = graph.run_leiden_dongdaemun_refinement(
        target_max_weight=2.5,
        resolution=0.000001,
        n_iterations=1,
        randomness=0.0,
        seed=11,
        initial_membership=np.zeros(4, dtype=np.uint64),
        max_extra_parents_per_iteration=1,
        max_singleton_weight_fraction=1.0,
        gamma_multipliers=[20_000_000.0],
    )

    assert result.membership.shape == (4,)
    assert np.isfinite(result.quality)
    assert result.audit.enabled is True
    assert result.audit.selected_parent_count_total >= 1
    assert result.audit.applied_parent_count_total >= 1
    assert result.audit.high_gamma_candidates_total >= 1
    assert result.audit.high_gamma_applied_total >= 1
    assert result.audit.same_gamma_candidates_total == 0
    assert result.audit.same_gamma_applied_total == 0
    assert result.audit.baseline_repair_candidates_total == 0
    assert result.audit.baseline_repair_improved_candidates_total == 0
    assert result.audit.baseline_repair_selected_total == 0
    assert result.audit.baseline_repair_merge_count_total == 0
    assert result.audit.baseline_repair_delta_sum == 0.0
    assert result.audit.final_quality_guard_enabled is False
    assert result.audit.final_quality_guard_triggered is False
    assert result.audit.final_quality_guard_standard_quality == 0.0
    assert result.audit.final_quality_guard_pre_guard_quality == pytest.approx(
        result.quality
    )
    assert result.audit.final_quality_delta_vs_guard_standard == 0.0
    assert np.isfinite(result.audit.candidate_quality_delta_sum)
    assert result.audit.high_gamma_quality_delta_sum == pytest.approx(
        result.audit.candidate_quality_delta_sum
    )
    assert result.audit.candidate_rejected_by_quality_total == 0
    quadrant_total = (
        result.audit.candidate_qpos_spos_total
        + result.audit.candidate_qpos_sneg_total
        + result.audit.candidate_qneg_spos_total
        + result.audit.candidate_qneg_sneg_total
    )
    assert quadrant_total == (
        result.audit.same_gamma_candidates_total
        + result.audit.high_gamma_candidates_total
    )
    assert (
        result.audit.candidate_true_positive_total
        + result.audit.candidate_false_positive_total
    ) == result.audit.applied_parent_count_total
    assert result.audit.iteration_selected_parents.shape[0] >= 1
    assert result.audit.iteration_high_gamma_candidates.shape[0] >= 1
    assert result.audit.iteration_same_gamma_candidates.shape[0] >= 1
    assert result.audit.iteration_candidate_quality_delta_sum.shape[0] >= 1
    assert result.audit.iteration_candidate_qpos_spos.shape[0] >= 1
    assert result.audit.iteration_candidate_true_positive.shape[0] >= 1
    assert result.audit.iteration_baseline_repair_candidates.shape[0] >= 1
    assert np.all(result.audit.iteration_baseline_repair_candidates == 0)


@pytestmark_dongdaemun_refinement
def test_dongdaemun_refinement_candidate_trace_includes_run_id(tmp_path, monkeypatch):
    trace_path = tmp_path / "candidate_trace.jsonl"
    monkeypatch.setenv("SCISCAPE_DDM_CANDIDATE_TRACE_PATH", str(trace_path))
    monkeypatch.setenv("SCISCAPE_DDM_CANDIDATE_TRACE_RUN_ID", "trace-row-1")
    graph = build_leiden_graph(
        edges_src=np.asarray([0, 2, 0, 0, 1, 1], dtype=np.uint32),
        edges_dst=np.asarray([1, 3, 2, 3, 2, 3], dtype=np.uint32),
        edges_weight=np.asarray([10.0, 10.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64),
        n_nodes=4,
    )

    graph.run_leiden_dongdaemun_refinement(
        target_max_weight=2.5,
        resolution=0.000001,
        n_iterations=1,
        randomness=0.0,
        seed=11,
        initial_membership=np.zeros(4, dtype=np.uint64),
        max_extra_parents_per_iteration=1,
        max_singleton_weight_fraction=1.0,
        gamma_multipliers=[20_000_000.0],
    )

    events = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert {event["run_id"] for event in events} == {"trace-row-1"}
    assert any(event["event"] == "candidate_profile" for event in events)
    assert any(event["event"] == "candidate_decision" for event in events)
    profile = next(event for event in events if event["event"] == "candidate_profile")
    assert "adaptive_diagnostic_score" in profile
    assert "adaptive_quality_band" in profile
    assert "adaptive_plateau_compared" in profile
    assert profile["parent_visit_index"] == 1


def test_dongdaemun_refinement_adaptive_probe_trace_only_records_probe(
    tmp_path, monkeypatch
):
    trace_path = tmp_path / "candidate_trace.jsonl"
    monkeypatch.setenv("SCISCAPE_DDM_CANDIDATE_TRACE_PATH", str(trace_path))
    monkeypatch.setenv("SCISCAPE_DDM_CANDIDATE_TRACE_RUN_ID", "probe-row-1")
    graph = build_leiden_graph(
        edges_src=np.asarray([0, 2, 0, 0, 1, 1], dtype=np.uint32),
        edges_dst=np.asarray([1, 3, 2, 3, 2, 3], dtype=np.uint32),
        edges_weight=np.asarray([10.0, 10.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64),
        n_nodes=4,
    )

    baseline = graph.run_leiden_dongdaemun_refinement(
        target_max_weight=2.5,
        resolution=0.000001,
        n_iterations=1,
        randomness=0.01,
        seed=11,
        initial_membership=np.zeros(4, dtype=np.uint64),
        max_extra_parents_per_iteration=1,
        max_singleton_weight_fraction=1.0,
        gamma_multipliers=[],
        seed_perturbations=1,
        candidate_quality_policy="quality_first",
    )
    probed = graph.run_leiden_dongdaemun_refinement(
        target_max_weight=2.5,
        resolution=0.000001,
        n_iterations=1,
        randomness=0.01,
        seed=11,
        initial_membership=np.zeros(4, dtype=np.uint64),
        max_extra_parents_per_iteration=1,
        max_singleton_weight_fraction=1.0,
        gamma_multipliers=[],
        seed_perturbations=1,
        candidate_quality_policy="quality_first",
        adaptive_probe_mode="trace_only",
        adaptive_probe_perturbations=2,
        adaptive_probe_targets=("0:0:1",),
        adaptive_probe_include_node_order_control=True,
    )

    assert np.array_equal(probed.membership, baseline.membership)
    events = [json.loads(line) for line in trace_path.read_text().splitlines()]
    probe_events = [
        event for event in events if event["event"] == "adaptive_probe_candidate"
    ]
    assert len(probe_events) == 4
    assert {event["source"] for event in probe_events} == {
        "same_gamma_probe",
        "node_order_control",
    }
    assert {event["mode"] for event in probe_events} == {"trace_only"}
    assert all(event["parent_visit_index"] == 1 for event in probe_events)
    assert all(event["committed"] is False for event in probe_events)
    assert all(event["commit_block_reason"] == "trace_only" for event in probe_events)
    assert all(event["commit_strategy"] == "online_first" for event in probe_events)
    assert all("commit_gain_parent_weight" in event for event in probe_events)


def test_dongdaemun_refinement_adaptive_probe_rejects_unknown_commit_source():
    graph = build_leiden_graph(
        edges_src=np.asarray([0, 2, 0, 0, 1, 1], dtype=np.uint32),
        edges_dst=np.asarray([1, 3, 2, 3, 2, 3], dtype=np.uint32),
        edges_weight=np.asarray([10.0, 10.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64),
        n_nodes=4,
    )

    with pytest.raises(ValueError, match="adaptive_probe_commit_sources"):
        graph.run_leiden_dongdaemun_refinement(
            target_max_weight=2.5,
            resolution=0.000001,
            n_iterations=1,
            randomness=0.01,
            seed=11,
            initial_membership=np.zeros(4, dtype=np.uint64),
            max_extra_parents_per_iteration=1,
            max_singleton_weight_fraction=1.0,
            gamma_multipliers=[],
            seed_perturbations=1,
            adaptive_probe_mode="conservative_apply",
            adaptive_probe_perturbations=1,
            adaptive_probe_targets=("0:0:1",),
            adaptive_probe_commit_sources=("bad_source",),
        )


def test_dongdaemun_refinement_adaptive_probe_rejects_unknown_commit_strategy():
    graph = build_leiden_graph(
        edges_src=np.asarray([0, 2, 0, 0, 1, 1], dtype=np.uint32),
        edges_dst=np.asarray([1, 3, 2, 3, 2, 3], dtype=np.uint32),
        edges_weight=np.asarray([10.0, 10.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64),
        n_nodes=4,
    )

    with pytest.raises(ValueError, match="adaptive_probe_commit_strategy"):
        graph.run_leiden_dongdaemun_refinement(
            target_max_weight=2.5,
            resolution=0.000001,
            n_iterations=1,
            randomness=0.01,
            seed=11,
            initial_membership=np.zeros(4, dtype=np.uint64),
            max_extra_parents_per_iteration=1,
            max_singleton_weight_fraction=1.0,
            gamma_multipliers=[],
            seed_perturbations=1,
            adaptive_probe_mode="conservative_apply",
            adaptive_probe_perturbations=1,
            adaptive_probe_targets=("0:0:1",),
            adaptive_probe_commit_strategy="unknown",
        )


def test_dongdaemun_refinement_forwards_near_tie_probe_kwargs():
    class FakeGraph:
        def __init__(self):
            self.kwargs = None

        def run_leiden_dongdaemun_refinement(self, **kwargs):
            self.kwargs = kwargs
            raise RuntimeError("captured")

    fake_backend = FakeGraph()
    graph = leiden_rust.RustLeidenGraph(graph=fake_backend, n_nodes=2, n_edges=0)

    with pytest.raises(RuntimeError, match="captured"):
        graph.run_leiden_dongdaemun_refinement(
            target_max_weight=1.0,
            resolution=0.1,
            adaptive_near_tie_probe_mode="qf_replace",
            adaptive_near_tie_margin_parent_weight=1e-4,
            adaptive_near_tie_randomness=0.05,
            adaptive_near_tie_max_decisions_per_parent=8,
        )

    assert fake_backend.kwargs["adaptive_near_tie_probe_mode"] == "qf_replace"
    assert fake_backend.kwargs[
        "adaptive_near_tie_margin_parent_weight"
    ] == pytest.approx(1e-4)
    assert fake_backend.kwargs["adaptive_near_tie_randomness"] == pytest.approx(0.05)
    assert fake_backend.kwargs["adaptive_near_tie_max_decisions_per_parent"] == 8


def test_dongdaemun_refinement_rejects_invalid_near_tie_probe_mode():
    class FakeGraph:
        def run_leiden_dongdaemun_refinement(self, **kwargs):
            raise AssertionError("backend should not be called")

    graph = leiden_rust.RustLeidenGraph(graph=FakeGraph(), n_nodes=2, n_edges=0)

    with pytest.raises(ValueError, match="adaptive_near_tie_probe_mode"):
        graph.run_leiden_dongdaemun_refinement(
            target_max_weight=1.0,
            resolution=0.1,
            adaptive_near_tie_probe_mode="bad",
        )


def test_dongdaemun_refinement_forwards_local_shake_kwargs():
    class FakeGraph:
        def __init__(self):
            self.kwargs = None

        def run_leiden_dongdaemun_refinement(self, **kwargs):
            self.kwargs = kwargs
            raise RuntimeError("captured")

    fake_backend = FakeGraph()
    graph = leiden_rust.RustLeidenGraph(graph=fake_backend, n_nodes=2, n_edges=0)

    with pytest.raises(RuntimeError, match="captured"):
        graph.run_leiden_dongdaemun_refinement(
            target_max_weight=1.0,
            resolution=0.1,
            adaptive_local_shake_mode="qf_replace",
            adaptive_local_shake_arms=(
                "near_tie_refinement",
                "resolution_up",
                "resolution_down",
                "seed_local_refinement",
            ),
            adaptive_local_shake_max_arms_per_parent=2,
            adaptive_local_shake_max_candidates_per_parent=4,
            adaptive_local_shake_resolution_up_multipliers=(1.02,),
            adaptive_local_shake_resolution_down_multipliers=(0.98,),
            adaptive_local_shake_seed_perturbations=1,
            adaptive_local_shake_near_tie_margin_parent_weight=1e-4,
            adaptive_local_shake_near_tie_randomness=0.05,
            adaptive_local_shake_final_guard_mode="runner_audit",
        )

    assert fake_backend.kwargs["adaptive_local_shake_mode"] == "qf_replace"
    assert fake_backend.kwargs["adaptive_local_shake_arms"] == [
        "near_tie_refinement",
        "resolution_up",
        "resolution_down",
        "seed_local_refinement",
    ]
    assert fake_backend.kwargs["adaptive_local_shake_max_arms_per_parent"] == 2
    assert fake_backend.kwargs["adaptive_local_shake_max_candidates_per_parent"] == 4
    assert fake_backend.kwargs["adaptive_local_shake_resolution_up_multipliers"] == [
        pytest.approx(1.02)
    ]
    assert fake_backend.kwargs["adaptive_local_shake_resolution_down_multipliers"] == [
        pytest.approx(0.98)
    ]
    assert fake_backend.kwargs["adaptive_local_shake_seed_perturbations"] == 1
    assert fake_backend.kwargs[
        "adaptive_local_shake_near_tie_margin_parent_weight"
    ] == pytest.approx(1e-4)
    assert fake_backend.kwargs[
        "adaptive_local_shake_near_tie_randomness"
    ] == pytest.approx(0.05)
    assert (
        fake_backend.kwargs["adaptive_local_shake_final_guard_mode"] == "runner_audit"
    )


def test_dongdaemun_refinement_rejects_invalid_local_shake_options():
    class FakeGraph:
        def run_leiden_dongdaemun_refinement(self, **kwargs):
            raise AssertionError("backend should not be called")

    graph = leiden_rust.RustLeidenGraph(graph=FakeGraph(), n_nodes=2, n_edges=0)

    with pytest.raises(ValueError, match="adaptive_local_shake_mode"):
        graph.run_leiden_dongdaemun_refinement(
            target_max_weight=1.0,
            resolution=0.1,
            adaptive_local_shake_mode="bad",
        )
    with pytest.raises(ValueError, match="adaptive_local_shake_arms"):
        graph.run_leiden_dongdaemun_refinement(
            target_max_weight=1.0,
            resolution=0.1,
            adaptive_local_shake_mode="qf_replace",
        )
    with pytest.raises(ValueError, match="adaptive_local_shake_arms"):
        graph.run_leiden_dongdaemun_refinement(
            target_max_weight=1.0,
            resolution=0.1,
            adaptive_local_shake_mode="qf_replace",
            adaptive_local_shake_arms=("bad_arm",),
        )
    with pytest.raises(ValueError, match="quality_guard"):
        graph.run_leiden_dongdaemun_refinement(
            target_max_weight=1.0,
            resolution=0.1,
            adaptive_local_shake_final_guard_mode="quality_guard",
        )


@pytestmark_dongdaemun_refinement
def test_dongdaemun_refinement_local_shake_trace_only_records_candidates(
    tmp_path, monkeypatch
):
    trace_path = tmp_path / "candidate_trace.jsonl"
    monkeypatch.setenv("SCISCAPE_DDM_CANDIDATE_TRACE_PATH", str(trace_path))
    monkeypatch.setenv("SCISCAPE_DDM_CANDIDATE_TRACE_RUN_ID", "local-shake-row")
    graph = build_leiden_graph(
        edges_src=np.asarray([0, 2, 0, 0, 1, 1], dtype=np.uint32),
        edges_dst=np.asarray([1, 3, 2, 3, 2, 3], dtype=np.uint32),
        edges_weight=np.asarray([10.0, 10.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64),
        n_nodes=4,
    )

    result = graph.run_leiden_dongdaemun_refinement(
        target_max_weight=2.5,
        resolution=0.000001,
        n_iterations=1,
        randomness=0.0,
        seed=11,
        initial_membership=np.zeros(4, dtype=np.uint64),
        max_extra_parents_per_iteration=1,
        max_singleton_weight_fraction=1.0,
        min_largest_child_fraction_improvement=0.0,
        adaptive_local_shake_mode="trace_only",
        adaptive_local_shake_arms=("resolution_up",),
        adaptive_local_shake_resolution_up_multipliers=(1.02,),
    )

    events = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert any(event["event"] == "adaptive_local_shake_trigger" for event in events)
    assert any(event["event"] == "adaptive_local_shake_candidate" for event in events)
    assert any(event["event"] == "adaptive_local_shake_decision" for event in events)
    assert result.audit.adaptive_local_shake_candidates_total >= 1
    assert result.audit.adaptive_local_shake_commits_total == 0


@pytestmark_dongdaemun_refinement
def test_dongdaemun_refinement_quality_trace_records_checkpoints(tmp_path, monkeypatch):
    trace_path = tmp_path / "quality_trace.jsonl"
    monkeypatch.setenv("SCISCAPE_DDM_QUALITY_TRACE_PATH", str(trace_path))
    monkeypatch.setenv("SCISCAPE_DDM_QUALITY_TRACE_RUN_ID", "quality-row-1")
    monkeypatch.setenv("SCISCAPE_DDM_QUALITY_TRACE_EPOCH", "epoch-1")
    graph = build_leiden_graph(
        edges_src=np.asarray([0, 2, 0, 0, 1, 1], dtype=np.uint32),
        edges_dst=np.asarray([1, 3, 2, 3, 2, 3], dtype=np.uint32),
        edges_weight=np.asarray([10.0, 10.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64),
        n_nodes=4,
    )

    graph.run_leiden_dongdaemun_refinement(
        target_max_weight=2.5,
        resolution=0.000001,
        n_iterations=1,
        randomness=0.0,
        seed=11,
        initial_membership=np.zeros(4, dtype=np.uint64),
        max_extra_parents_per_iteration=1,
        max_singleton_weight_fraction=1.0,
        gamma_multipliers=[20_000_000.0],
        use_final_quality_guard=True,
    )

    events = [json.loads(line) for line in trace_path.read_text().splitlines()]
    phases = [event["phase"] for event in events]
    assert phases == ["start", "after_iteration", "pre_final_guard", "final"]
    assert {event["run_id"] for event in events} == {"quality-row-1"}
    assert [event["checkpoint_index"] for event in events] == list(range(len(events)))
    start_quality = events[0]["quality"]
    assert events[0]["quality_delta_vs_start"] == pytest.approx(0.0)
    assert events[0]["max_doc_weight"] == pytest.approx(4.0)
    assert events[0]["max_doc_weight_ratio"] == pytest.approx(1.6)
    assert events[0]["n_above_max_doc_weight"] == 1
    for event in events:
        assert event["quality_delta_vs_start"] == pytest.approx(
            event["quality"] - start_quality
        )
    assert events[-1]["selected_parent_count_total"] >= 1
    assert events[-1]["applied_parent_count_total"] >= 0
    assert all("elapsed_ms_since_run_start" in event for event in events)
    assert all("iteration_elapsed_ms" in event for event in events)
    assert all("moved_nodes" in event for event in events)


def test_standard_leiden_quality_trace_records_checkpoints(tmp_path, monkeypatch):
    trace_path = tmp_path / "standard_quality_trace.jsonl"
    monkeypatch.setenv("SCISCAPE_LEIDEN_QUALITY_TRACE_PATH", str(trace_path))
    monkeypatch.setenv("SCISCAPE_LEIDEN_QUALITY_TRACE_RUN_ID", "standard-row-1")
    monkeypatch.setenv("SCISCAPE_LEIDEN_QUALITY_TRACE_EPOCH", "epoch-1")
    monkeypatch.setenv("SCISCAPE_LEIDEN_QUALITY_TRACE_TARGET_MAX_WEIGHT", "2.5")
    graph = build_leiden_graph(
        edges_src=np.asarray([0, 2, 0, 0, 1, 1], dtype=np.uint32),
        edges_dst=np.asarray([1, 3, 2, 3, 2, 3], dtype=np.uint32),
        edges_weight=np.asarray([10.0, 10.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64),
        n_nodes=4,
    )

    result = graph.run_leiden(
        resolution=0.000001,
        n_iterations=2,
        randomness=0.0,
        seed=11,
        initial_membership=np.zeros(4, dtype=np.uint64),
    )

    events = [json.loads(line) for line in trace_path.read_text().splitlines()]
    phases = [event["phase"] for event in events]
    assert phases[0] == "start"
    assert "after_iteration" in phases
    assert phases[-1] == "final"
    assert {event["run_id"] for event in events} == {"standard-row-1"}
    assert [event["checkpoint_index"] for event in events] == list(range(len(events)))
    assert events[-1]["quality"] == pytest.approx(result.quality)
    start_quality = events[0]["quality"]
    for event in events:
        assert event["schema"] == "leiden_quality_checkpoint.v1"
        assert event["quality_delta_vs_start"] == pytest.approx(
            event["quality"] - start_quality
        )
        assert event["elapsed_ms_since_run_start"] >= 0.0
        assert event["iteration_elapsed_ms"] >= 0.0
        assert "moved_nodes" in event
    assert events[0]["max_doc_weight"] == pytest.approx(4.0)
    assert events[0]["max_doc_weight_ratio"] == pytest.approx(1.6)
    assert events[0]["n_above_max_doc_weight"] == 1


def test_standard_leiden_quality_trace_randomness_zero_is_deterministic(
    tmp_path, monkeypatch
):
    graph = build_leiden_graph(
        edges_src=np.asarray([0, 2, 0, 0, 1, 1], dtype=np.uint32),
        edges_dst=np.asarray([1, 3, 2, 3, 2, 3], dtype=np.uint32),
        edges_weight=np.asarray([10.0, 10.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64),
        n_nodes=4,
    )

    traces = []
    results = []
    for index in range(2):
        trace_path = tmp_path / f"standard_quality_trace_{index}.jsonl"
        monkeypatch.setenv("SCISCAPE_LEIDEN_QUALITY_TRACE_PATH", str(trace_path))
        monkeypatch.setenv("SCISCAPE_LEIDEN_QUALITY_TRACE_RUN_ID", f"run-{index}")
        monkeypatch.setenv("SCISCAPE_LEIDEN_QUALITY_TRACE_EPOCH", f"epoch-{index}")
        monkeypatch.setenv("SCISCAPE_LEIDEN_QUALITY_TRACE_TARGET_MAX_WEIGHT", "2.5")
        results.append(
            graph.run_leiden(
                resolution=0.000001,
                n_iterations=2,
                randomness=0.0,
                seed=11,
                initial_membership=np.zeros(4, dtype=np.uint64),
            )
        )
        traces.append(
            [json.loads(line) for line in trace_path.read_text().splitlines()]
        )

    assert np.array_equal(results[0].membership, results[1].membership)
    assert results[0].quality == pytest.approx(results[1].quality)

    comparable_fields = (
        "phase",
        "iteration",
        "quality",
        "quality_delta_vs_start",
        "n_clusters",
        "max_doc_weight",
        "max_doc_weight_ratio",
        "n_above_max_doc_weight",
        "moved_nodes",
    )
    left = [{field: event[field] for field in comparable_fields} for event in traces[0]]
    right = [
        {field: event[field] for field in comparable_fields} for event in traces[1]
    ]
    assert left == right


@pytestmark_dongdaemun_refinement
def test_cached_graph_run_leiden_dongdaemun_auto_fast_skips_low_pressure():
    src, dst, w = _two_clique_edges()
    graph = build_leiden_graph(
        edges_src=src,
        edges_dst=dst,
        edges_weight=w,
        n_nodes=8,
    )

    result = graph.run_leiden_dongdaemun_auto_fast_refinement(
        target_max_weight=100.0,
        resolution=0.1,
        seed=7,
        n_iterations=3,
        trigger_max_doc_weight_ratio=1.03,
        trigger_min_above_max_doc_weight=2,
    )

    assert result.selected_variant == "standard"
    assert result.triggered is False
    assert result.fallback_triggered is True
    assert result.fallback_reason == "trigger_not_met"
    assert result.repair_off is None
    np.testing.assert_array_equal(result.membership, result.standard.membership)


@pytestmark_dongdaemun_refinement
def test_cached_graph_run_leiden_dongdaemun_safe_fast_skips_low_pressure():
    src, dst, w = _two_clique_edges()
    graph = build_leiden_graph(
        edges_src=src,
        edges_dst=dst,
        edges_weight=w,
        n_nodes=8,
    )

    result = graph.run_leiden_dongdaemun_safe_fast_refinement(
        target_max_weight=100.0,
        resolution=0.1,
        seed=7,
        n_iterations=3,
    )

    assert result.selected_variant == "standard"
    assert result.triggered is False
    assert result.fallback_triggered is True
    assert result.fallback_reason == "trigger_not_met"
    assert result.max_extra_parents_per_iteration == 4
    assert result.max_extra_children_per_parent == 16
    assert result.repair_off is None
    np.testing.assert_array_equal(result.membership, result.standard.membership)


@pytestmark_dongdaemun_refinement
def test_cached_graph_run_leiden_dongdaemun_safe_fast_matches_explicit_defaults():
    graph = build_leiden_graph(
        edges_src=np.asarray([0, 2, 0, 0, 1, 1], dtype=np.uint32),
        edges_dst=np.asarray([1, 3, 2, 3, 2, 3], dtype=np.uint32),
        edges_weight=np.asarray([10.0, 10.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64),
        n_nodes=4,
    )
    common = {
        "target_max_weight": 2.5,
        "resolution": 0.000001,
        "n_iterations": 1,
        "randomness": 0.0,
        "seed": 11,
        "initial_membership": np.zeros(4, dtype=np.uint64),
        "trigger_max_doc_weight_ratio": None,
        "trigger_min_above_max_doc_weight": None,
        "accept_max_doc_weight_ratio": 10.0,
    }

    safe = graph.run_leiden_dongdaemun_safe_fast_refinement(**common)
    explicit = graph.run_leiden_dongdaemun_auto_fast_refinement(
        **common,
        accept_min_quality_delta=0.0,
        accept_min_quality_delta_ratio=None,
        max_extra_parents_per_iteration=4,
        max_extra_children_per_parent=16,
        parent_selection_policy="weight",
        gamma_multipliers=(1.02, 1.05),
        use_quotient_diagnostic=True,
        candidate_quality_policy="structural",
        allow_repair_escalation=False,
        baseline_repair_policy="adaptive",
    )

    assert safe.selected_variant == explicit.selected_variant
    assert safe.triggered == explicit.triggered
    assert safe.fallback_triggered == explicit.fallback_triggered
    assert safe.fallback_reason == explicit.fallback_reason
    assert safe.quality == pytest.approx(explicit.quality)
    assert (
        safe.max_extra_parents_per_iteration == explicit.max_extra_parents_per_iteration
    )
    assert safe.max_extra_children_per_parent == explicit.max_extra_children_per_parent
    np.testing.assert_array_equal(safe.membership, explicit.membership)


@pytestmark_dongdaemun_refinement
def test_cached_graph_run_leiden_dongdaemun_refinement_accepts_parent_policy():
    src, dst, w = _two_clique_edges()
    graph = build_leiden_graph(
        edges_src=src,
        edges_dst=dst,
        edges_weight=w,
        n_nodes=8,
    )

    result = graph.run_leiden_dongdaemun_refinement(
        target_max_weight=3.0,
        resolution=0.1,
        seed=7,
        n_iterations=3,
        max_extra_parents_per_iteration=2,
        max_extra_children_per_parent=8,
        parent_selection_policy="pressure_boundary",
        gamma_multipliers=[1.02],
    )

    assert result.membership.shape == (8,)
    assert isinstance(result.quality, float)


@pytestmark_dongdaemun_refinement
def test_cached_graph_run_leiden_dongdaemun_auto_fast_quality_guard_falls_back():
    graph = build_leiden_graph(
        edges_src=np.asarray([0, 2, 0, 0, 1, 1], dtype=np.uint32),
        edges_dst=np.asarray([1, 3, 2, 3, 2, 3], dtype=np.uint32),
        edges_weight=np.asarray([10.0, 10.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64),
        n_nodes=4,
    )

    result = graph.run_leiden_dongdaemun_auto_fast_refinement(
        target_max_weight=2.5,
        resolution=0.000001,
        n_iterations=1,
        randomness=0.0,
        seed=11,
        initial_membership=np.zeros(4, dtype=np.uint64),
        trigger_max_doc_weight_ratio=None,
        trigger_min_above_max_doc_weight=None,
        accept_max_doc_weight_ratio=10.0,
        accept_min_quality_delta=1.0e9,
        max_extra_parents_per_iteration=1,
        max_extra_children_per_parent=8,
        max_singleton_weight_fraction=1.0,
        gamma_multipliers=[20_000_000.0],
    )

    assert result.selected_variant == "standard"
    assert result.triggered is True
    assert result.fallback_triggered is True
    assert result.fallback_reason == "quality_guard"
    assert result.repair_off is not None
    np.testing.assert_array_equal(result.membership, result.standard.membership)


@pytestmark_dongdaemun_refinement
def test_cached_graph_run_leiden_dongdaemun_auto_fast_quality_ratio_guard_falls_back():
    graph = build_leiden_graph(
        edges_src=np.asarray([0, 2, 0, 0, 1, 1], dtype=np.uint32),
        edges_dst=np.asarray([1, 3, 2, 3, 2, 3], dtype=np.uint32),
        edges_weight=np.asarray([10.0, 10.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64),
        n_nodes=4,
    )

    result = graph.run_leiden_dongdaemun_auto_fast_refinement(
        target_max_weight=2.5,
        resolution=0.000001,
        n_iterations=1,
        randomness=0.0,
        seed=11,
        initial_membership=np.zeros(4, dtype=np.uint64),
        trigger_max_doc_weight_ratio=None,
        trigger_min_above_max_doc_weight=None,
        accept_max_doc_weight_ratio=10.0,
        accept_min_quality_delta_ratio=1.0e6,
        max_extra_parents_per_iteration=1,
        max_extra_children_per_parent=8,
        max_singleton_weight_fraction=1.0,
        gamma_multipliers=[20_000_000.0],
    )

    assert result.selected_variant == "standard"
    assert result.fallback_triggered is True
    assert result.fallback_reason == "quality_guard"


@pytestmark_dongdaemun_refinement
def test_cached_graph_run_leiden_dongdaemun_auto_fast_runs_tier_and_escalation():
    graph = build_leiden_graph(
        edges_src=np.asarray([0, 2, 0, 0, 1, 1], dtype=np.uint32),
        edges_dst=np.asarray([1, 3, 2, 3, 2, 3], dtype=np.uint32),
        edges_weight=np.asarray([10.0, 10.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64),
        n_nodes=4,
    )

    result = graph.run_leiden_dongdaemun_auto_fast_refinement(
        target_max_weight=2.5,
        resolution=0.000001,
        n_iterations=1,
        randomness=0.0,
        seed=11,
        initial_membership=np.zeros(4, dtype=np.uint64),
        trigger_max_doc_weight_ratio=None,
        trigger_min_above_max_doc_weight=None,
        accept_max_doc_weight_ratio=10.0,
        max_extra_parents_per_iteration=1,
        max_extra_children_per_parent=8,
        parent_selection_policy="pressure_boundary",
        severe_trigger_max_doc_weight_ratio=0.0,
        severe_max_extra_parents_per_iteration=2,
        severe_max_extra_children_per_parent=16,
        max_singleton_weight_fraction=1.0,
        gamma_multipliers=[20_000_000.0],
        allow_repair_escalation=True,
        repair_escalation_accept_max_doc_weight_ratio=10.0,
    )

    assert result.triggered is True
    assert result.fallback_triggered is False
    assert result.severe_tier_triggered is True
    assert result.max_extra_parents_per_iteration == 2
    assert result.max_extra_children_per_parent == 16
    assert result.repair_off is not None
    assert result.repair_escalated is True
    assert result.repair_on is not None
    assert result.selected_variant in {"refine_repair_off", "refine_repair_on"}
    assert result.membership.shape == (4,)


@pytestmark_dongdaemun_refinement
def test_cached_graph_run_leiden_dongdaemun_refinement_final_quality_guard_falls_back():
    graph = build_leiden_graph(
        edges_src=np.asarray([0, 2, 0, 0, 1, 1], dtype=np.uint32),
        edges_dst=np.asarray([1, 3, 2, 3, 2, 3], dtype=np.uint32),
        edges_weight=np.asarray([10.0, 10.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64),
        n_nodes=4,
    )
    standard = graph.run_leiden(
        resolution=0.000001,
        n_iterations=1,
        randomness=0.0,
        seed=11,
        initial_membership=np.zeros(4, dtype=np.uint64),
    )
    guarded = graph.run_leiden_dongdaemun_refinement(
        target_max_weight=2.5,
        resolution=0.000001,
        n_iterations=1,
        randomness=0.0,
        seed=11,
        initial_membership=np.zeros(4, dtype=np.uint64),
        max_extra_parents_per_iteration=1,
        max_singleton_weight_fraction=1.0,
        gamma_multipliers=[20_000_000.0],
        use_final_quality_guard=True,
        min_final_quality_delta=1.0e9,
    )

    assert guarded.audit.final_quality_guard_enabled is True
    assert guarded.audit.final_quality_guard_triggered is True
    assert guarded.audit.final_quality_guard_standard_quality == pytest.approx(
        standard.quality
    )
    assert np.isfinite(guarded.audit.final_quality_guard_pre_guard_quality)
    assert guarded.quality == pytest.approx(standard.quality)
    assert np.array_equal(guarded.membership, standard.membership)
    assert guarded.audit.applied_parent_count_total >= 1


@pytestmark_dongdaemun_refinement
def test_cached_graph_run_leiden_dongdaemun_refinement_reports_quotient_audit():
    graph = build_leiden_graph(
        edges_src=np.asarray([0, 2, 0, 0, 1, 1], dtype=np.uint32),
        edges_dst=np.asarray([1, 3, 2, 3, 2, 3], dtype=np.uint32),
        edges_weight=np.asarray([10.0, 10.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64),
        n_nodes=4,
    )
    result = graph.run_leiden_dongdaemun_refinement(
        target_max_weight=2.5,
        resolution=0.000001,
        n_iterations=1,
        randomness=0.0,
        seed=11,
        initial_membership=np.zeros(4, dtype=np.uint64),
        max_extra_parents_per_iteration=1,
        max_singleton_weight_fraction=1.0,
        gamma_multipliers=[20_000_000.0],
        use_quotient_diagnostic=True,
    )

    assert result.membership.shape == (4,)
    assert result.audit.quotient_candidates_total >= 1
    assert result.audit.quotient_positive_candidates_total == 0
    assert result.audit.quotient_selected_total == 0
    assert result.audit.quotient_score_sum == 0.0
    assert result.audit.iteration_quotient_candidates.shape[0] >= 1
    assert result.audit.iteration_quotient_positive_candidates.shape[0] >= 1
    assert result.audit.iteration_quotient_selected.shape[0] >= 1
    assert result.audit.iteration_quotient_score_sum.shape[0] >= 1


@pytestmark_dongdaemun_refinement
def test_cached_graph_run_leiden_dongdaemun_refinement_reports_baseline_repair_audit():
    graph = build_leiden_graph(
        edges_src=np.asarray([0, 2], dtype=np.uint32),
        edges_dst=np.asarray([1, 3], dtype=np.uint32),
        edges_weight=np.asarray([10.0, 10.0], dtype=np.float64),
        n_nodes=4,
    )
    result = graph.run_leiden_dongdaemun_refinement(
        target_max_weight=1.5,
        resolution=1.0,
        n_iterations=1,
        randomness=0.0,
        seed=11,
        initial_membership=np.zeros(4, dtype=np.uint64),
        max_extra_parents_per_iteration=2,
        max_extra_children_per_parent=8,
        max_singleton_weight_fraction=0.0,
        min_largest_child_fraction_improvement=0.0,
        gamma_multipliers=[100.0],
        use_baseline_repair=True,
    )

    assert result.membership.shape == (4,)
    assert result.audit.baseline_repair_candidates_total >= 1
    assert result.audit.baseline_repair_improved_candidates_total >= 1
    assert result.audit.baseline_repair_merge_count_total >= 1
    assert result.audit.baseline_repair_delta_sum > 0.0
    assert result.audit.iteration_baseline_repair_candidates.shape[0] >= 1
    assert result.audit.iteration_baseline_repair_improved_candidates.shape[0] >= 1
    assert result.audit.iteration_baseline_repair_selected.shape[0] >= 1
    assert result.audit.iteration_baseline_repair_merge_count.shape[0] >= 1
    assert result.audit.iteration_baseline_repair_delta_sum.shape[0] >= 1


@pytestmark_dongdaemun_refinement
def test_cached_graph_run_leiden_dongdaemun_refinement_reports_seed_perturbation_audit():
    graph = build_leiden_graph(
        edges_src=np.asarray([0, 2, 0, 0, 1, 1], dtype=np.uint32),
        edges_dst=np.asarray([1, 3, 2, 3, 2, 3], dtype=np.uint32),
        edges_weight=np.asarray([10.0, 10.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64),
        n_nodes=4,
    )
    result = graph.run_leiden_dongdaemun_refinement(
        target_max_weight=2.5,
        resolution=0.000001,
        n_iterations=1,
        randomness=0.01,
        seed=11,
        initial_membership=np.zeros(4, dtype=np.uint64),
        max_extra_parents_per_iteration=1,
        max_singleton_weight_fraction=1.0,
        gamma_multipliers=[],
        seed_perturbations=2,
    )

    assert result.membership.shape == (4,)
    assert result.audit.enabled is True
    assert result.audit.same_gamma_candidates_total >= 2
    assert result.audit.high_gamma_candidates_total == 0
    assert result.audit.iteration_same_gamma_candidates.shape[0] >= 1


@pytestmark_dongdaemun_refinement
def test_cached_graph_run_leiden_dongdaemun_refinement_accepts_selective_policy():
    graph = build_leiden_graph(
        edges_src=np.asarray([0, 2, 0, 0, 1, 1], dtype=np.uint32),
        edges_dst=np.asarray([1, 3, 2, 3, 2, 3], dtype=np.uint32),
        edges_weight=np.asarray([10.0, 10.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64),
        n_nodes=4,
    )
    result = graph.run_leiden_dongdaemun_refinement(
        target_max_weight=2.5,
        resolution=0.000001,
        n_iterations=1,
        randomness=0.0,
        seed=11,
        initial_membership=np.zeros(4, dtype=np.uint64),
        max_extra_parents_per_iteration=1,
        max_singleton_weight_fraction=1.0,
        min_largest_child_fraction_improvement=0.05,
        gamma_multipliers=[20_000_000.0],
        candidate_quality_policy="selective",
        min_candidate_delta_q=1.0e-12,
    )

    assert result.membership.shape == (4,)
    assert result.audit.enabled is True


@pytestmark_dongdaemun_refinement
def test_cached_graph_run_leiden_dongdaemun_refinement_accepts_pressure_aware_policy():
    graph = build_leiden_graph(
        edges_src=np.asarray([0, 2, 0, 0, 1, 1], dtype=np.uint32),
        edges_dst=np.asarray([1, 3, 2, 3, 2, 3], dtype=np.uint32),
        edges_weight=np.asarray([10.0, 10.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64),
        n_nodes=4,
    )
    result = graph.run_leiden_dongdaemun_refinement(
        target_max_weight=2.5,
        resolution=0.000001,
        n_iterations=1,
        randomness=0.0,
        seed=11,
        initial_membership=np.zeros(4, dtype=np.uint64),
        max_extra_parents_per_iteration=1,
        max_singleton_weight_fraction=1.0,
        min_largest_child_fraction_improvement=0.05,
        gamma_multipliers=[20_000_000.0],
        candidate_quality_policy="pressure_aware",
        min_candidate_delta_q=-1.0,
    )

    assert result.membership.shape == (4,)
    assert result.audit.enabled is True


@pytestmark_dongdaemun_refinement
def test_cached_graph_run_leiden_dongdaemun_refinement_accepts_adaptive_plateau_policy():
    graph = build_leiden_graph(
        edges_src=np.asarray([0, 2, 0, 0, 1, 1], dtype=np.uint32),
        edges_dst=np.asarray([1, 3, 2, 3, 2, 3], dtype=np.uint32),
        edges_weight=np.asarray([10.0, 10.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64),
        n_nodes=4,
    )
    result = graph.run_leiden_dongdaemun_refinement(
        target_max_weight=2.5,
        resolution=0.000001,
        n_iterations=1,
        randomness=0.0,
        seed=11,
        initial_membership=np.zeros(4, dtype=np.uint64),
        max_extra_parents_per_iteration=1,
        max_singleton_weight_fraction=1.0,
        min_largest_child_fraction_improvement=0.05,
        gamma_multipliers=[20_000_000.0],
        candidate_quality_policy="adaptive_plateau",
        min_candidate_delta_q=-1.0,
        adaptive_plateau_quality_band=1.0,
    )

    assert result.membership.shape == (4,)
    assert result.audit.enabled is True


def test_dongdaemun_refinement_reports_stale_binding(monkeypatch):
    monkeypatch.setattr(leiden_rust, "RUST_DONGDAEMUN_REFINEMENT_AVAILABLE", False)

    with pytest.raises(AttributeError, match="maturin develop"):
        leiden_rust._check_dongdaemun_refinement_available()


def test_cached_graph_trim_oversize_boundary_moves_reduces_source():
    graph = build_leiden_graph(
        edges_src=np.asarray([0, 1, 2], dtype=np.uint32),
        edges_dst=np.asarray([1, 2, 3], dtype=np.uint32),
        edges_weight=np.asarray([3.0, 0.1, 4.0], dtype=np.float64),
        n_nodes=4,
    )
    membership = np.asarray([0, 0, 0, 1], dtype=np.uint64)

    quality_before = graph.cpm_quality(membership, resolution=0.1)
    result = graph.trim_oversize_boundary_moves(
        membership,
        np.asarray([0], dtype=np.uint64),
        resolution=0.1,
        target_max_weight=2.0,
        min_delta_q=0.0,
    )
    quality_after = graph.cpm_quality(result.membership, resolution=0.1)

    np.testing.assert_array_equal(
        result.membership, np.asarray([0, 0, 1, 1], dtype=np.uint64)
    )
    assert result.n_moves == 1
    assert int(result.source[0]) == 0
    assert int(result.target[0]) == 1
    assert int(result.node[0]) == 2
    assert float(result.source_weight_after[0]) == pytest.approx(2.0)
    assert float(result.target_weight_after[0]) == pytest.approx(2.0)
    assert quality_after - quality_before == pytest.approx(float(result.delta_q.sum()))


def test_cached_graph_postprocess_shape_matches_wrapper():
    src, dst, w = _two_clique_edges()
    graph = build_leiden_graph(
        edges_src=src,
        edges_dst=dst,
        edges_weight=w,
        n_nodes=8,
    )
    initial = np.arange(8, dtype=np.uint64)

    cached = graph.postprocess_small_clusters(
        resolution=0.1,
        min_size=2,
        membership=initial,
        seed=3,
        n_iterations=2,
    )
    wrapper = postprocess_small_clusters_rust(
        edges_src=src,
        edges_dst=dst,
        edges_weight=w,
        n_nodes=8,
        resolution=0.1,
        min_size=2,
        membership=initial,
        seed=3,
        n_iterations=2,
    )

    assert cached.membership.shape == wrapper.membership.shape == (8,)
    assert cached.changed_at_round.shape == wrapper.changed_at_round.shape == (8,)
    assert cached.n_clusters == wrapper.n_clusters


def test_graph_from_edge_path_matches_arrays(tmp_path):
    src, dst, w = _two_clique_edges()
    edge_path = tmp_path / "int_edges.parquet"
    pl.DataFrame({"src": src, "dst": dst, "weight": w}).write_parquet(edge_path)

    from_path = build_leiden_graph(edge_path=edge_path, n_nodes=8)
    from_arrays = build_leiden_graph(
        edges_src=src,
        edges_dst=dst,
        edges_weight=w,
        n_nodes=8,
    )

    path_result = from_path.run_leiden(resolution=0.1, seed=11, n_iterations=3)
    array_result = from_arrays.run_leiden(resolution=0.1, seed=11, n_iterations=3)

    assert from_path.n_edges == from_arrays.n_edges == len(src) * 2
    assert path_result.n_clusters == array_result.n_clusters
    np.testing.assert_array_equal(path_result.membership, array_result.membership)


def test_project_membership_rust_compacts_to_uint32():
    membership = np.array([10, 20, 30], dtype=np.uint64)
    previous = np.array([2, 0, 1, 2], dtype=np.uint32)

    projected = project_membership_rust(membership, previous)

    assert projected.dtype == np.uint32
    np.testing.assert_array_equal(
        projected, np.array([30, 10, 20, 30], dtype=np.uint32)
    )


def test_project_membership_rust_preserves_wide_cluster_ids():
    wide = np.iinfo(np.uint32).max + 1
    membership = np.array([0, wide], dtype=np.uint64)
    previous = np.array([1, 0], dtype=np.uint64)

    projected = project_membership_rust(membership, previous)

    assert projected.dtype == np.uint64
    np.testing.assert_array_equal(projected, np.array([wide, 0], dtype=np.uint64))


def test_project_membership_rust_rejects_out_of_bounds_index():
    membership = np.array([0, 1], dtype=np.uint64)
    previous = np.array([0, 2], dtype=np.uint32)

    with pytest.raises(ValueError, match="out of bounds"):
        project_membership_rust(membership, previous)


def test_graph_from_edge_path_with_node_weights_leaves_no_temp_sidecar(tmp_path):
    src, dst, w = _two_clique_edges()
    edge_path = tmp_path / "int_edges.parquet"
    pl.DataFrame({"src": src, "dst": dst, "weight": w}).write_parquet(edge_path)

    graph = build_leiden_graph(
        edge_path=edge_path,
        n_nodes=8,
        node_weights=np.arange(1, 9, dtype=np.float64),
    )

    assert graph.node_weights is not None
    assert not list(tmp_path.glob("node_weights.*.f64.bin"))


def test_graph_from_edge_path_with_node_weights_path_leaves_no_temp_sidecar(tmp_path):
    src, dst, w = _two_clique_edges()
    edge_path = tmp_path / "int_edges.parquet"
    node_weights_path = tmp_path / "node_weights.f64.bin"
    weights = np.arange(1, 9, dtype=np.float64)
    pl.DataFrame({"src": src, "dst": dst, "weight": w}).write_parquet(edge_path)
    weights.tofile(node_weights_path)

    graph = build_leiden_graph(
        edge_path=edge_path,
        n_nodes=8,
        node_weights_path=node_weights_path,
    )

    assert graph.node_weights is not None
    np.testing.assert_array_equal(np.asarray(graph.node_weights), weights)
    assert not list(tmp_path.glob("node_weights.*.f64.bin"))


def test_graph_from_edge_path_rejects_node_weights_and_path(tmp_path):
    src, dst, w = _two_clique_edges()
    edge_path = tmp_path / "int_edges.parquet"
    node_weights_path = tmp_path / "node_weights.f64.bin"
    pl.DataFrame({"src": src, "dst": dst, "weight": w}).write_parquet(edge_path)
    np.ones(8, dtype=np.float64).tofile(node_weights_path)

    with pytest.raises(ValueError, match="node_weights"):
        build_leiden_graph(
            edge_path=edge_path,
            n_nodes=8,
            node_weights=np.ones(8, dtype=np.float64),
            node_weights_path=node_weights_path,
        )


def test_remap_parquet_to_graph_skips_edge_files(tmp_path):
    src, dst, w = _two_clique_edges()
    uids = [f"n{i}" for i in range(8)]
    edge_path = tmp_path / "string_edges.parquet"
    pl.DataFrame(
        {
            "uid1": [uids[int(i)] for i in src],
            "uid2": [uids[int(i)] for i in dst],
            "rel_sum2": w,
        }
    ).write_parquet(edge_path)

    direct = remap_parquet_to_leiden_graph(edge_path, tmp_path / "remap")

    assert direct is not None
    remap, graph = direct
    assert remap.n_nodes == 8
    assert remap.n_edges == len(src)
    assert graph.n_edges == len(src) * 2
    assert remap.node_manifest_path.exists()
    assert not remap.int_edges_path.exists()
    assert not (tmp_path / "remap" / "src.u32.bin").exists()
    assert not (tmp_path / "remap" / "dst.u32.bin").exists()
    assert not (tmp_path / "remap" / "weight.f64.bin").exists()

    result = graph.run_leiden(resolution=0.1, seed=11, n_iterations=3)
    assert result.membership.shape == (8,)


def test_cached_graph_contract_returns_reusable_graph():
    src, dst, w = _two_clique_edges()
    graph = build_leiden_graph(
        edges_src=src,
        edges_dst=dst,
        edges_weight=w,
        n_nodes=8,
    )
    membership = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.uint64)

    contracted = graph.contract(membership)
    result = contracted.run_leiden(resolution=0.1, seed=13, n_iterations=2)

    assert contracted.n_nodes == 2
    assert contracted.node_weights is not None
    np.testing.assert_array_equal(contracted.node_weights, np.array([4.0, 4.0]))
    assert result.membership.shape == (2,)


def test_cached_graph_search_resolution_returns_stats_without_membership():
    src, dst, w = _two_clique_edges()
    graph = build_leiden_graph(
        edges_src=src,
        edges_dst=dst,
        edges_weight=w,
        n_nodes=8,
    )

    result = graph.search_resolution(
        min_clusters=2,
        max_clusters=3,
        lower_bound=0.001,
        upper_bound=1.0,
        max_iterations=8,
        n_iterations=3,
        seed=17,
    )

    assert result.cluster_count in {2, 3}
    assert result.resolution > 0
    assert np.isfinite(result.quality)
    assert result.eval_count >= 2
    assert result.membership.shape == (8,)
    assert result.membership.dtype == np.uint64
