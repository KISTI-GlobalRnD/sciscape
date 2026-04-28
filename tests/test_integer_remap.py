"""Tests for sciscape.clustering.integer_remap."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from sciscape.clustering.integer_remap import (
    RemapResult,
    ensure_int_edge_sidecars,
    int_edge_sidecar_paths,
    integer_remap,
    join_back_uids,
    load_manifest,
)
from sciscape.clustering.leiden_rust import RUST_AVAILABLE


@pytest.fixture
def sample_edges() -> pl.DataFrame:
    return pl.DataFrame({
        "uid1": ["A", "B", "C", "A"],
        "uid2": ["B", "C", "D", "D"],
        "rel_sum2": [1.0, 2.0, 3.0, 0.5],
    })


class TestIntegerRemap:
    def test_returns_remap_result(self, sample_edges, tmp_path):
        result = integer_remap(sample_edges, tmp_path)
        assert isinstance(result, RemapResult)

    def test_node_count(self, sample_edges, tmp_path):
        result = integer_remap(sample_edges, tmp_path)
        assert result.n_nodes == 4  # A, B, C, D

    def test_edge_count(self, sample_edges, tmp_path):
        result = integer_remap(sample_edges, tmp_path)
        assert result.n_edges == 4

    def test_manifest_parquet_written(self, sample_edges, tmp_path):
        result = integer_remap(sample_edges, tmp_path)
        assert result.node_manifest_path.exists()
        manifest = pl.read_parquet(result.node_manifest_path)
        assert set(manifest.columns) == {"node_idx", "uid"}
        assert manifest.height == 4

    def test_int_edges_parquet_written(self, sample_edges, tmp_path):
        result = integer_remap(sample_edges, tmp_path)
        assert result.int_edges_path.exists()
        edges = pl.read_parquet(result.int_edges_path)
        assert set(edges.columns) == {"src", "dst", "weight"}
        assert edges.height == 4
        assert edges["src"].dtype == pl.UInt32
        assert edges["dst"].dtype == pl.UInt32
        assert edges["weight"].dtype == pl.Float64

    def test_int_edge_sidecars_written(self, sample_edges, tmp_path):
        result = integer_remap(sample_edges, tmp_path)
        src_path, dst_path, weight_path = int_edge_sidecar_paths(result.int_edges_path)

        assert src_path.exists()
        assert dst_path.exists()
        assert weight_path.exists()
        assert src_path.stat().st_size == result.n_edges * np.dtype(np.uint32).itemsize
        assert dst_path.stat().st_size == result.n_edges * np.dtype(np.uint32).itemsize
        assert weight_path.stat().st_size == result.n_edges * np.dtype(np.float64).itemsize

    def test_invalid_int_edge_sidecar_is_regenerated(self, sample_edges, tmp_path):
        result = integer_remap(sample_edges, tmp_path)
        src_path, dst_path, weight_path = int_edge_sidecar_paths(result.int_edges_path)
        src_path.write_bytes(b"bad")

        ensure_int_edge_sidecars(result.int_edges_path)

        assert src_path.stat().st_size == result.n_edges * np.dtype(np.uint32).itemsize
        assert dst_path.stat().st_size == result.n_edges * np.dtype(np.uint32).itemsize
        assert weight_path.stat().st_size == result.n_edges * np.dtype(np.float64).itemsize

    def test_integer_indices_are_contiguous(self, sample_edges, tmp_path):
        integer_remap(sample_edges, tmp_path)
        manifest = pl.read_parquet(tmp_path / "node_manifest.parquet")
        indices = manifest["node_idx"].to_numpy()
        assert np.array_equal(indices, np.arange(4))

    def test_edge_indices_reference_valid_nodes(self, sample_edges, tmp_path):
        result = integer_remap(sample_edges, tmp_path)
        edges = pl.read_parquet(result.int_edges_path)
        assert edges["src"].min() >= 0
        assert edges["dst"].min() >= 0
        assert edges["src"].max() < result.n_nodes
        assert edges["dst"].max() < result.n_nodes

    def test_cache_hit(self, sample_edges, tmp_path):
        r1 = integer_remap(sample_edges, tmp_path)
        r2 = integer_remap(sample_edges, tmp_path)
        assert r1.n_nodes == r2.n_nodes
        assert r1.n_edges == r2.n_edges

    def test_overwrite_forces_recompute(self, sample_edges, tmp_path):
        integer_remap(sample_edges, tmp_path)
        result = integer_remap(sample_edges, tmp_path, overwrite=True)
        assert result.n_nodes == 4

    def test_from_parquet_path(self, sample_edges, tmp_path):
        parquet_path = tmp_path / "edges.parquet"
        sample_edges.write_parquet(parquet_path)
        result = integer_remap(parquet_path, tmp_path / "out")
        assert result.n_nodes == 4

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust backend required")
    def test_from_parquet_path_writes_valid_rust_remap(self, sample_edges, tmp_path):
        parquet_path = tmp_path / "edges.parquet"
        sample_edges.write_parquet(parquet_path)

        result = integer_remap(parquet_path, tmp_path / "out")
        manifest = pl.read_parquet(result.node_manifest_path)
        int_edges = pl.read_parquet(result.int_edges_path)
        src_path, dst_path, weight_path = int_edge_sidecar_paths(result.int_edges_path)

        assert manifest["node_idx"].dtype == pl.UInt32
        assert int_edges["src"].dtype == pl.UInt32
        assert int_edges["dst"].dtype == pl.UInt32
        assert int_edges["weight"].dtype == pl.Float64
        assert src_path.stat().st_size == result.n_edges * np.dtype(np.uint32).itemsize
        assert dst_path.stat().st_size == result.n_edges * np.dtype(np.uint32).itemsize
        assert weight_path.stat().st_size == result.n_edges * np.dtype(np.float64).itemsize

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust backend required")
    def test_dataframe_input_uses_rust_remap_without_leaving_temp_input(self, sample_edges, tmp_path):
        result = integer_remap(sample_edges, tmp_path)

        assert result.n_nodes == 4
        assert result.n_edges == 4
        assert not list(tmp_path.glob("_rust_remap_input_*.parquet"))

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust backend required")
    def test_rust_sidecar_only_remap_skips_int_edges_parquet(self, sample_edges, tmp_path):
        parquet_path = tmp_path / "edges.parquet"
        sample_edges.write_parquet(parquet_path)

        result = integer_remap(parquet_path, tmp_path / "out", write_int_edges=False)
        src_path, dst_path, weight_path = result.sidecar_paths

        assert result.n_nodes == 4
        assert result.n_edges == 4
        assert result.node_manifest_path.exists()
        assert not result.int_edges_path.exists()
        assert src_path.exists()
        assert dst_path.exists()
        assert weight_path.exists()

        cached = integer_remap(parquet_path, tmp_path / "out", write_int_edges=False)
        assert cached.n_edges == result.n_edges
        assert ensure_int_edge_sidecars(result.int_edges_path) == result.sidecar_paths

    def test_custom_column_names(self, tmp_path):
        edges = pl.DataFrame({
            "source": ["X", "Y"],
            "target": ["Y", "Z"],
            "w": [1.0, 2.0],
        })
        result = integer_remap(
            edges, tmp_path,
            uid1_col="source", uid2_col="target", weight_col="w",
        )
        assert result.n_nodes == 3
        assert result.n_edges == 2


class TestJoinBackUids:
    def test_basic(self, sample_edges, tmp_path):
        result = integer_remap(sample_edges, tmp_path)
        membership = [0, 0, 1, 1]  # 4 nodes
        joined = join_back_uids(membership, result.node_manifest_path)
        assert set(joined.columns) == {"uid", "cluster"}
        assert joined.height == 4

    def test_membership_values_preserved(self, sample_edges, tmp_path):
        result = integer_remap(sample_edges, tmp_path)
        membership = [10, 20, 30, 40]
        joined = join_back_uids(membership, result.node_manifest_path)
        assert set(joined["cluster"].to_list()) == {10, 20, 30, 40}

    def test_accepts_dataframe(self, sample_edges, tmp_path):
        result = integer_remap(sample_edges, tmp_path)
        manifest = pl.read_parquet(result.node_manifest_path)
        membership = np.array([0, 1, 0, 1])
        joined = join_back_uids(membership, manifest)
        assert joined.height == 4


class TestLoadManifest:
    def test_load(self, sample_edges, tmp_path):
        result = integer_remap(sample_edges, tmp_path)
        manifest = load_manifest(result.node_manifest_path)
        assert isinstance(manifest, pl.DataFrame)
        assert manifest.height == 4
