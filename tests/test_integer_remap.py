"""Tests for sciscape.clustering.integer_remap."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from sciscape.clustering.integer_remap import (
    RemapResult,
    integer_remap,
    join_back_uids,
    load_binary_edge_arrays,
    load_manifest,
)


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
        assert result.src_bin_path.exists()
        assert result.dst_bin_path.exists()
        assert result.weight_bin_path.exists()

    def test_binary_edge_sidecars_roundtrip(self, sample_edges, tmp_path):
        result = integer_remap(sample_edges, tmp_path)
        src, dst, weight = load_binary_edge_arrays(result)
        edges = pl.read_parquet(result.int_edges_path)
        assert np.array_equal(src, edges["src"].to_numpy().astype(np.uint32))
        assert np.array_equal(dst, edges["dst"].to_numpy().astype(np.uint32))
        assert np.allclose(weight, edges["weight"].to_numpy().astype(np.float64))

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
