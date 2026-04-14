"""Tests for prepartition module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import igraph as ig
import pytest

from sciscape.clustering.runner import LeidenRunner
from sciscape.clustering.prepartition import (
    PrepartitionResult,
    CascadeResult,
    prepartition,
    contract_graph,
    cascade_search,
    save_prepartition,
    load_prepartition,
    load_prepartition_metadata,
    is_cache_valid,
)


def _make_graph(n_cliques: int = 4, clique_size: int = 20, bridge_weight: float = 0.1):
    """Create a synthetic graph with clear community structure."""
    g = ig.Graph()
    g.add_vertices(n_cliques * clique_size)
    edges = []
    weights = []
    # Dense intra-clique edges
    for c in range(n_cliques):
        base = c * clique_size
        for i in range(clique_size):
            for j in range(i + 1, clique_size):
                edges.append((base + i, base + j))
                weights.append(1.0)
    # Sparse inter-clique bridges
    for c in range(n_cliques - 1):
        edges.append((c * clique_size, (c + 1) * clique_size))
        weights.append(bridge_weight)
    g.add_edges(edges)
    g.es["weight"] = weights
    return g


@pytest.fixture
def graph():
    return _make_graph()


@pytest.fixture
def runner(graph):
    return LeidenRunner(graph, objective="cpm", default_seed=42)


class TestPrepartition:
    def test_basic(self, runner):
        result = prepartition(runner, gamma_pre=0.1, seed=42)
        assert isinstance(result, PrepartitionResult)
        assert result.n_nodes == 80
        assert result.n_parts > 0
        assert len(result.pre_membership) == 80
        assert result.gamma_pre == 0.1

    def test_membership_contiguous(self, runner):
        result = prepartition(runner, gamma_pre=0.1, seed=42)
        ids = set(result.pre_membership)
        assert ids == set(range(result.n_parts))

    def test_node_sizes_list(self, runner):
        result = prepartition(runner, gamma_pre=0.1, seed=42)
        sizes = result.node_sizes_list
        assert len(sizes) == result.n_parts
        assert sum(sizes) == result.n_nodes

    def test_high_gamma_more_parts(self, runner):
        low = prepartition(runner, gamma_pre=0.01, seed=42)
        high = prepartition(runner, gamma_pre=0.5, seed=42)
        assert high.n_parts >= low.n_parts


class TestContractGraph:
    def test_contraction(self, runner):
        blocks = prepartition(runner, gamma_pre=0.1, seed=42)
        contracted, c_runner = contract_graph(runner, blocks)
        assert contracted.vcount() == blocks.n_parts
        assert contracted.vcount() < runner.graph.vcount()

    def test_no_self_loops(self, runner):
        blocks = prepartition(runner, gamma_pre=0.1, seed=42)
        contracted, _ = contract_graph(runner, blocks)
        for e in contracted.es:
            assert e.source != e.target


class TestCascadeSearch:
    def test_basic(self, runner):
        blocks = prepartition(runner, gamma_pre=0.1, seed=42)
        result = cascade_search(
            runner, blocks,
            gamma_targets=[0.05, 0.01],
            seed=42,
        )
        assert isinstance(result, CascadeResult)
        assert len(result.membership) == 80
        assert result.n_clusters > 0
        assert result.hot_started is True

    def test_no_hot_start(self, runner):
        blocks = prepartition(runner, gamma_pre=0.1, seed=42)
        result = cascade_search(
            runner, blocks,
            gamma_targets=[0.05, 0.01],
            seed=42,
            hot_start=False,
        )
        assert result.hot_started is False

    def test_skips_high_gamma(self, runner):
        blocks = prepartition(runner, gamma_pre=0.1, seed=42)
        result = cascade_search(
            runner, blocks,
            gamma_targets=[0.5, 0.05, 0.01],  # 0.5 > 0.1, should skip
            seed=42,
        )
        assert 0.5 not in result.cascade_path

    def test_all_above_gamma_pre_raises(self, runner):
        blocks = prepartition(runner, gamma_pre=0.1, seed=42)
        with pytest.raises(ValueError, match="No valid"):
            cascade_search(runner, blocks, gamma_targets=[0.5, 1.0])

    def test_cascade_path_descending(self, runner):
        blocks = prepartition(runner, gamma_pre=0.5, seed=42)
        result = cascade_search(
            runner, blocks,
            gamma_targets=[0.1, 0.01, 0.05],  # unordered input
            seed=42,
        )
        # cascade_path should be descending (execution order)
        assert result.cascade_path == sorted(result.cascade_path, reverse=True)


class TestPersistence:
    def test_save_load_roundtrip(self, runner):
        blocks = prepartition(runner, gamma_pre=0.1, seed=42)
        uids = [f"W{i}" for i in range(80)]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "parts.parquet"
            save_prepartition(blocks, path, uids, source="/data/test.parquet")

            loaded = load_prepartition(path)
            assert loaded is not None
            assert loaded.gamma_pre == blocks.gamma_pre
            assert loaded.seed == blocks.seed
            assert loaded.n_parts == blocks.n_parts
            assert loaded.n_nodes == blocks.n_nodes
            assert loaded.pre_membership == blocks.pre_membership

    def test_load_nonexistent(self):
        result = load_prepartition(Path("/nonexistent/blocks.parquet"))
        assert result is None

    def test_metadata(self, runner):
        blocks = prepartition(runner, gamma_pre=0.1, seed=42)
        uids = [f"W{i}" for i in range(80)]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "parts.parquet"
            save_prepartition(blocks, path, uids, source="/data/test.parquet")

            meta = load_prepartition_metadata(path)
            assert meta is not None
            assert meta["gamma_pre"] == "0.1"
            assert meta["source"] == "/data/test.parquet"
            assert "created" in meta

    def test_cache_valid(self, runner):
        blocks = prepartition(runner, gamma_pre=0.1, seed=42)
        uids = [f"W{i}" for i in range(80)]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "parts.parquet"
            save_prepartition(blocks, path, uids, source="/data/test.parquet")

            assert is_cache_valid(path, 0.1, 80, "/data/test.parquet") is True
            assert is_cache_valid(path, 0.2, 80, "/data/test.parquet") is False  # gamma mismatch
            assert is_cache_valid(path, 0.1, 100, "/data/test.parquet") is False  # n_nodes mismatch
            assert is_cache_valid(path, 0.1, 80, "/data/other.parquet") is False  # source mismatch
            assert is_cache_valid(path, 0.1, 80) is True  # no source check

    def test_cache_nonexistent(self):
        assert is_cache_valid(Path("/nope"), 0.1, 80) is False
