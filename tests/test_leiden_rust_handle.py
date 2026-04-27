"""Tests for Rust graph-handle Leiden/postprocess APIs."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from sciscape.clustering.leiden_rust import (
    RUST_AVAILABLE,
    _load_membership_path,
    write_membership_raw_sidecar,
    write_membership_sidecars_for_dataframe,
    contract_graph_rust_handle,
    load_graph_rust,
    postprocess_small_clusters_rust,
    postprocess_small_clusters_rust_handle,
    run_leiden_rust,
    run_leiden_rust_handle,
    summarize_membership_rust_handle,
)
from sciscape.clustering.runner import RustLeidenRunner
from sciscape.clustering.integer_remap import integer_remap

pytestmark = pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust backend required")


def _two_clique_arrays():
    src = np.array([0, 1, 2, 3, 4, 5, 2], dtype=np.uint32)
    dst = np.array([1, 2, 0, 4, 5, 3, 3], dtype=np.uint32)
    w = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.01], dtype=np.float64)
    return src, dst, w, 6


def _singleton_merge_arrays():
    src = np.array([0], dtype=np.uint32)
    dst = np.array([1], dtype=np.uint32)
    w = np.array([10.0], dtype=np.float64)
    return src, dst, w, 2


def _postprocess_renumber_arrays():
    src = np.array([1, 3, 4, 5, 0], dtype=np.uint32)
    dst = np.array([2, 4, 5, 3, 3], dtype=np.uint32)
    w = np.array([1.0, 1.0, 1.0, 1.0, 5.0], dtype=np.float64)
    return src, dst, w, 6


def test_graph_handle_matches_direct_leiden():
    src, dst, w, n_nodes = _two_clique_arrays()

    direct = run_leiden_rust(
        edges_src=src,
        edges_dst=dst,
        edges_weight=w,
        n_nodes=n_nodes,
        resolution=0.5,
        seed=42,
    )
    handle = load_graph_rust(
        edges_src=src,
        edges_dst=dst,
        edges_weight=w,
        n_nodes=n_nodes,
    )
    via_handle = run_leiden_rust_handle(
        handle,
        resolution=0.5,
        seed=42,
    )

    assert handle.n_nodes == n_nodes
    assert np.array_equal(direct.membership, via_handle.membership)
    assert via_handle.n_clusters == direct.n_clusters
    assert via_handle.quality == pytest.approx(direct.quality)


def test_rust_runner_run_array_keeps_numpy_membership():
    src, dst, w, n_nodes = _two_clique_arrays()
    runner = RustLeidenRunner(src, dst, w, n_nodes, default_seed=42)

    array_result = runner.run_array(0.5, seed=42)
    list_result = runner.run(0.5, seed=42)

    assert isinstance(array_result.membership, np.ndarray)
    assert array_result.membership.dtype == np.uint64
    assert isinstance(list_result.membership, list)
    assert np.array_equal(array_result.membership, np.asarray(list_result.membership, dtype=np.uint64))


def test_rust_runner_run_array_accepts_initial_membership_path(tmp_path: Path):
    src, dst, w, n_nodes = _two_clique_arrays()
    runner = RustLeidenRunner(src, dst, w, n_nodes, default_seed=42)
    init = np.array([0, 0, 0, 1, 1, 1], dtype=np.uint64)
    init_path = tmp_path / "init.npy"
    np.save(init_path, init)

    result = runner.run_array(0.5, seed=42, initial_membership_path=str(init_path))

    assert isinstance(result.membership, np.ndarray)
    assert len(result.membership) == n_nodes


def test_postprocess_handle_matches_direct():
    src, dst, w, n_nodes = _two_clique_arrays()
    init_membership = np.array([0, 0, 0, 1, 1, 2], dtype=np.uint64)

    direct = postprocess_small_clusters_rust(
        resolution=0.1,
        min_size=3,
        membership=init_membership,
        edges_src=src,
        edges_dst=dst,
        edges_weight=w,
        n_nodes=n_nodes,
        seed=42,
    )
    handle = load_graph_rust(
        edges_src=src,
        edges_dst=dst,
        edges_weight=w,
        n_nodes=n_nodes,
    )
    via_handle = postprocess_small_clusters_rust_handle(
        handle=handle,
        resolution=0.1,
        min_size=3,
        membership=init_membership,
        seed=42,
    )

    assert np.array_equal(direct.membership, via_handle.membership)
    assert via_handle.n_clusters == direct.n_clusters
    assert np.array_equal(direct.changed_at_round, via_handle.changed_at_round)


def test_postprocess_handle_accepts_membership_path(tmp_path: Path):
    src, dst, w, n_nodes = _two_clique_arrays()
    membership = np.array([0, 0, 0, 1, 1, 2], dtype=np.uint64)
    membership_path = tmp_path / "membership.npy"
    np.save(membership_path, membership)

    handle = load_graph_rust(
        edges_src=src,
        edges_dst=dst,
        edges_weight=w,
        n_nodes=n_nodes,
    )
    result = postprocess_small_clusters_rust_handle(
        handle=handle,
        resolution=0.1,
        min_size=3,
        membership_path=membership_path,
        seed=42,
    )

    assert len(result.membership) == n_nodes
    assert result.n_clusters >= 1


def test_load_membership_path_memmaps_npy(tmp_path: Path):
    membership = np.array([0, 1, 1, 2], dtype=np.uint64)
    membership_path = tmp_path / "membership.npy"
    np.save(membership_path, membership)

    loaded = _load_membership_path(membership_path)

    assert isinstance(loaded, np.memmap)
    assert np.array_equal(loaded, membership)


def test_load_membership_path_builds_parquet_sidecar(tmp_path: Path):
    membership = np.array([0, 1, 1, 2], dtype=np.uint64)
    membership_path = tmp_path / "membership.parquet"
    pl.DataFrame({"membership": membership}).write_parquet(membership_path)

    loaded = _load_membership_path(membership_path)
    sidecar_path = membership_path.with_name(f"{membership_path.name}.u64.bin")

    assert sidecar_path.exists()
    assert isinstance(loaded, np.memmap)
    assert np.array_equal(loaded, membership)


def test_write_membership_raw_sidecar_roundtrip(tmp_path: Path):
    membership = np.array([0, 1, 1, 2], dtype=np.uint64)
    membership_path = tmp_path / "membership.parquet"

    sidecar_path = write_membership_raw_sidecar(membership_path, membership)
    loaded = _load_membership_path(sidecar_path)

    assert sidecar_path == membership_path.with_name(f"{membership_path.name}.u64.bin")
    assert isinstance(loaded, np.memmap)
    assert np.array_equal(loaded, membership)


def test_write_membership_sidecars_for_dataframe_with_column_selector(tmp_path: Path):
    membership_path = tmp_path / "membership.parquet"
    frame = pl.DataFrame(
        {
            "uid": ["a", "b", "c"],
            "cluster_nano": [0, 0, 1],
            "cluster_micro": [0, 1, 1],
        }
    )
    frame.write_parquet(membership_path)

    written = write_membership_sidecars_for_dataframe(membership_path, frame)
    nano = _load_membership_path(f"{membership_path}#cluster_nano")
    micro = _load_membership_path(f"{membership_path}#cluster_micro")

    assert len(written) == 2
    assert np.array_equal(nano, np.array([0, 0, 1], dtype=np.uint64))
    assert np.array_equal(micro, np.array([0, 1, 1], dtype=np.uint64))


def test_run_leiden_handle_accepts_raw_binary_membership_path(tmp_path: Path):
    src, dst, w, n_nodes = _two_clique_arrays()
    init_membership = np.array([0, 0, 0, 1, 1, 1], dtype=np.uint64)
    membership_path = tmp_path / "initial.u64.bin"
    init_membership.tofile(membership_path)

    handle = load_graph_rust(
        edges_src=src,
        edges_dst=dst,
        edges_weight=w,
        n_nodes=n_nodes,
    )
    result = run_leiden_rust_handle(
        handle,
        resolution=0.5,
        seed=42,
        initial_membership_path=membership_path,
    )

    assert len(result.membership) == n_nodes
    assert result.n_clusters >= 1


def test_postprocess_handle_accepts_raw_binary_membership_path(tmp_path: Path):
    src, dst, w, n_nodes = _two_clique_arrays()
    membership = np.array([0, 0, 0, 1, 1, 2], dtype=np.uint64)
    membership_path = tmp_path / "membership.u64.bin"
    membership.tofile(membership_path)

    handle = load_graph_rust(
        edges_src=src,
        edges_dst=dst,
        edges_weight=w,
        n_nodes=n_nodes,
    )
    result = postprocess_small_clusters_rust_handle(
        handle=handle,
        resolution=0.1,
        min_size=3,
        membership_path=membership_path,
        seed=42,
    )

    assert len(result.membership) == n_nodes
    assert result.n_clusters >= 1


def test_postprocess_handle_can_disable_changed_round_trace():
    src, dst, w, n_nodes = _two_clique_arrays()
    membership = np.array([0, 0, 0, 1, 1, 2], dtype=np.uint64)

    handle = load_graph_rust(
        edges_src=src,
        edges_dst=dst,
        edges_weight=w,
        n_nodes=n_nodes,
    )
    result = postprocess_small_clusters_rust_handle(
        handle=handle,
        resolution=0.1,
        min_size=3,
        membership=membership,
        seed=42,
        track_changed_rounds=False,
    )

    assert len(result.membership) == n_nodes
    assert result.changed_at_round.size == 0


def test_run_leiden_handle_accepts_initial_membership_path(tmp_path: Path):
    src, dst, w, n_nodes = _two_clique_arrays()
    init_membership = np.array([0, 0, 0, 1, 1, 1], dtype=np.uint64)
    membership_path = tmp_path / "initial.npy"
    np.save(membership_path, init_membership)

    handle = load_graph_rust(
        edges_src=src,
        edges_dst=dst,
        edges_weight=w,
        n_nodes=n_nodes,
    )
    result = run_leiden_rust_handle(
        handle,
        resolution=0.5,
        seed=42,
        initial_membership_path=membership_path,
    )

    assert len(result.membership) == n_nodes
    assert result.n_clusters >= 1


def test_run_leiden_handle_applies_fixed_nodes_without_initial_membership():
    src, dst, w, n_nodes = _singleton_merge_arrays()
    handle = load_graph_rust(
        edges_src=src,
        edges_dst=dst,
        edges_weight=w,
        n_nodes=n_nodes,
    )

    unfixed = run_leiden_rust_handle(handle, resolution=0.0, seed=42)
    fixed = run_leiden_rust_handle(
        handle,
        resolution=0.0,
        seed=42,
        fixed_nodes=np.array([True, True], dtype=bool),
    )

    assert unfixed.n_clusters == 1
    assert fixed.n_clusters == 2
    assert fixed.membership[0] != fixed.membership[1]


def test_runner_run_array_preserves_initial_membership_path_in_weighted_rebuild(monkeypatch):
    src, dst, w, n_nodes = _two_clique_arrays()
    runner = RustLeidenRunner(src, dst, w, n_nodes, default_seed=42)
    captured = {}

    def fake_load_membership_path(path):
        captured["path"] = str(path)
        return np.array([0, 0, 0, 1, 1, 1], dtype=np.uint64)

    def fake_run_leiden_rust(**kwargs):
        captured["initial_membership"] = kwargs.get("initial_membership")
        return SimpleNamespace(
            membership=np.array([0, 0, 0, 1, 1, 1], dtype=np.uint64),
            quality=1.0,
            n_clusters=2,
        )

    monkeypatch.setattr("sciscape.clustering.leiden_rust._load_membership_path", fake_load_membership_path)
    monkeypatch.setattr("sciscape.clustering.leiden_rust.run_leiden_rust", fake_run_leiden_rust)

    result = runner.run_array(
        0.5,
        seed=42,
        initial_membership_path="init.npy",
        node_sizes=np.ones(n_nodes, dtype=np.float64),
    )

    assert captured["path"] == "init.npy"
    assert np.array_equal(
        captured["initial_membership"],
        np.array([0, 0, 0, 1, 1, 1], dtype=np.uint64),
    )
    assert isinstance(result.membership, np.ndarray)


def test_graph_handle_prefers_raw_sidecars_for_edge_path(tmp_path: Path):
    src, dst, w, n_nodes = _two_clique_arrays()
    edges = pl.DataFrame({"src": src, "dst": dst, "weight": w})
    edge_path = tmp_path / "int_edges.parquet"
    edges.write_parquet(edge_path)
    manifest = pl.DataFrame({"node_idx": np.arange(n_nodes, dtype=np.int32), "uid": [f"N{i}" for i in range(n_nodes)]})
    manifest.write_parquet(tmp_path / "node_manifest.parquet")
    integer_remap(
        pl.DataFrame({
            "uid1": ["N0", "N1", "N2", "N3", "N4", "N5", "N2"],
            "uid2": ["N1", "N2", "N0", "N4", "N5", "N3", "N3"],
            "rel_sum2": w,
        }),
        tmp_path,
        overwrite=True,
    )

    handle = load_graph_rust(edge_path=edge_path, n_nodes=n_nodes)
    assert handle.loaded_from == "raw_files"
    result = run_leiden_rust_handle(handle, resolution=0.5, seed=42)
    assert len(result.membership) == n_nodes


def test_contracted_handle_can_skip_materialized_node_weights():
    src, dst, w, n_nodes = _two_clique_arrays()
    handle = load_graph_rust(
        edges_src=src,
        edges_dst=dst,
        edges_weight=w,
        n_nodes=n_nodes,
    )
    membership = np.array([0, 0, 0, 1, 1, 1], dtype=np.uint64)

    reduced_handle, reduced_node_weights = contract_graph_rust_handle(handle, membership)
    n_cl, max_size, max_weight, total_weight = summarize_membership_rust_handle(
        reduced_handle,
        membership=np.array([0, 1], dtype=np.uint64),
    )

    assert reduced_node_weights is None
    assert reduced_handle.loaded_from == "contracted_handle"
    assert n_cl == 2
    assert max_size == 1
    assert max_weight == pytest.approx(3.0)
    assert total_weight == pytest.approx(6.0)


def test_handle_membership_length_validation():
    src, dst, w, n_nodes = _two_clique_arrays()
    handle = load_graph_rust(
        edges_src=src,
        edges_dst=dst,
        edges_weight=w,
        n_nodes=n_nodes,
    )

    with pytest.raises(ValueError, match="initial_membership length mismatch"):
        run_leiden_rust_handle(
            handle,
            resolution=0.5,
            seed=42,
            initial_membership=np.array([0, 1], dtype=np.uint64),
        )

    with pytest.raises(ValueError, match="fixed_nodes length mismatch"):
        run_leiden_rust_handle(
            handle,
            resolution=0.5,
            seed=42,
            fixed_nodes=np.array([True, False], dtype=bool),
        )

    with pytest.raises(ValueError, match="membership length mismatch"):
        contract_graph_rust_handle(handle, np.array([0, 1], dtype=np.uint64))

    with pytest.raises(ValueError, match="membership length mismatch"):
        summarize_membership_rust_handle(handle, membership=np.array([0, 1], dtype=np.uint64))

    with pytest.raises(ValueError, match="membership length mismatch"):
        postprocess_small_clusters_rust_handle(
            handle=handle,
            resolution=0.1,
            min_size=2,
            membership=np.array([0, 1], dtype=np.uint64),
            seed=42,
        )


def test_postprocess_changed_at_ignores_pure_cluster_renumbering():
    src, dst, w, n_nodes = _postprocess_renumber_arrays()
    handle = load_graph_rust(
        edges_src=src,
        edges_dst=dst,
        edges_weight=w,
        n_nodes=n_nodes,
    )
    membership = np.array([0, 1, 1, 2, 2, 2], dtype=np.uint64)

    result = postprocess_small_clusters_rust_handle(
        handle=handle,
        resolution=0.1,
        min_size=2,
        membership=membership,
        seed=42,
        track_changed_rounds=True,
    )

    # Cluster 1 stays intact even if cluster IDs are compacted after cluster 0 merges away.
    assert result.changed_at_round[1] == -1
    assert result.changed_at_round[2] == -1


def test_contracted_runner_tracks_internal_node_weights_without_array():
    src, dst, w, n_nodes = _two_clique_arrays()
    runner = RustLeidenRunner(src, dst, w, n_nodes, default_seed=42)
    contracted = runner.contract(np.array([0, 0, 0, 1, 1, 1], dtype=np.uint64))

    assert contracted.has_node_weights is True
    assert contracted.node_weights is None
