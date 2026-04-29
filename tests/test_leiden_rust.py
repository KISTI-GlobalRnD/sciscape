"""Tests for the Rust Leiden Python wrapper."""

import numpy as np
import polars as pl
import pytest

from sciscape.clustering.leiden_rust import (
    RUST_AVAILABLE,
    build_leiden_graph,
    postprocess_small_clusters_rust,
    project_membership_rust,
    remap_parquet_to_leiden_graph,
    run_leiden_rust,
)

pytestmark = pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust backend required")


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

    np.testing.assert_array_equal(stats.block_count, np.asarray([2, 2], dtype=np.uint64))
    np.testing.assert_allclose(stats.doc_weight, np.asarray([2.0, 2.0]))
    np.testing.assert_allclose(stats.internal_weight, np.asarray([2.0, 3.0]))
    np.testing.assert_allclose(stats.external_weight, np.asarray([1.0, 1.0]))
    np.testing.assert_array_equal(stats.top_neighbor, np.asarray([1, 0], dtype=np.int64))
    np.testing.assert_allclose(stats.band_distance, np.asarray([1.0, 1.0]))
    assert stats.n_candidates == 1
    assert int(stats.candidate_source[0]) == 0
    assert int(stats.candidate_target[0]) == 1
    assert float(stats.candidate_delta_q[0]) == pytest.approx(0.6)
    assert float(stats.candidate_size_band_gain[0]) == pytest.approx(2.0)


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
    np.testing.assert_array_equal(projected, np.array([30, 10, 20, 30], dtype=np.uint32))


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


def test_remap_parquet_to_graph_skips_edge_files(tmp_path):
    src, dst, w = _two_clique_edges()
    uids = [f"n{i}" for i in range(8)]
    edge_path = tmp_path / "string_edges.parquet"
    pl.DataFrame({
        "uid1": [uids[int(i)] for i in src],
        "uid2": [uids[int(i)] for i in dst],
        "rel_sum2": w,
    }).write_parquet(edge_path)

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
