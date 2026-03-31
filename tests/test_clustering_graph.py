"""Tests for sciscape.clustering.graph — build_graph and giant_component."""

from __future__ import annotations

import igraph as ig
import polars as pl
import pytest

from sciscape.clustering.graph import build_graph, giant_component


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def simple_edges() -> pl.DataFrame:
    """5-node path graph: A-B-C-D-E with unit weights."""
    return pl.DataFrame({
        "uid1": ["A", "B", "C", "D"],
        "uid2": ["B", "C", "D", "E"],
        "rel_sum2": [1.0, 1.0, 1.0, 1.0],
    })


@pytest.fixture
def weighted_edges() -> pl.DataFrame:
    """Triangle with varying weights."""
    return pl.DataFrame({
        "uid1": ["A", "B", "A"],
        "uid2": ["B", "C", "C"],
        "rel_sum2": [0.5, 1.5, 2.0],
    })


@pytest.fixture
def disconnected_edges() -> pl.DataFrame:
    """Two disconnected components: {A,B} and {C,D}."""
    return pl.DataFrame({
        "uid1": ["A", "C"],
        "uid2": ["B", "D"],
        "rel_sum2": [1.0, 1.0],
    })


# ── build_graph basic ────────────────────────────────────────


class TestBuildGraphBasic:
    def test_node_count(self, simple_edges):
        g = build_graph(simple_edges)
        assert g.vcount() == 5

    def test_edge_count(self, simple_edges):
        g = build_graph(simple_edges)
        assert g.ecount() == 4

    def test_undirected(self, simple_edges):
        g = build_graph(simple_edges)
        assert not g.is_directed()

    def test_uid_attribute(self, simple_edges):
        g = build_graph(simple_edges)
        uids = set(g.vs["uid"])
        assert uids == {"A", "B", "C", "D", "E"}

    def test_weight_attribute(self, simple_edges):
        g = build_graph(simple_edges)
        assert all(w == 1.0 for w in g.es["weight"])


# ── Weights ──────────────────────────────────────────────────


class TestBuildGraphWeights:
    def test_weights_preserved(self, weighted_edges):
        g = build_graph(weighted_edges)
        # Map edges to weight dict keyed by sorted uid pair
        edge_weights = {}
        for e in g.es:
            pair = tuple(sorted([g.vs[e.source]["uid"], g.vs[e.target]["uid"]]))
            edge_weights[pair] = e["weight"]

        assert edge_weights[("A", "B")] == pytest.approx(0.5)
        assert edge_weights[("B", "C")] == pytest.approx(1.5)
        assert edge_weights[("A", "C")] == pytest.approx(2.0)

    def test_min_weight_filters(self, weighted_edges):
        g = build_graph(weighted_edges, min_weight=1.0)
        # Edge A-B (0.5) should be filtered out
        assert g.ecount() == 2
        # All remaining weights >= 1.0
        assert all(w >= 1.0 for w in g.es["weight"])

    def test_min_weight_filters_all_raises(self, weighted_edges):
        """Filtering all edges should raise a clear ValueError."""
        with pytest.raises(ValueError, match="No edges remain after filtering"):
            build_graph(weighted_edges, min_weight=10.0)

    def test_min_weight_none_keeps_all(self, weighted_edges):
        g = build_graph(weighted_edges, min_weight=None)
        assert g.ecount() == 3


# ── Edge cases ───────────────────────────────────────────────


class TestBuildGraphEdgeCases:
    def test_empty_edges_raises(self):
        """Empty edge DataFrame should raise a clear ValueError."""
        edges = pl.DataFrame({
            "uid1": pl.Series([], dtype=pl.Utf8),
            "uid2": pl.Series([], dtype=pl.Utf8),
            "rel_sum2": pl.Series([], dtype=pl.Float64),
        })
        with pytest.raises(ValueError, match="No edges remain after filtering"):
            build_graph(edges)

    def test_self_loop_edge(self):
        """Self-loops should be preserved (igraph allows them)."""
        edges = pl.DataFrame({
            "uid1": ["A", "A"],
            "uid2": ["B", "A"],
            "rel_sum2": [1.0, 0.5],
        })
        g = build_graph(edges)
        assert g.vcount() == 2
        assert g.ecount() == 2

    def test_duplicate_edges(self):
        """Duplicate edges are kept as multi-edges (igraph default)."""
        edges = pl.DataFrame({
            "uid1": ["A", "A"],
            "uid2": ["B", "B"],
            "rel_sum2": [1.0, 2.0],
        })
        g = build_graph(edges)
        assert g.vcount() == 2
        assert g.ecount() == 2

    def test_single_edge(self):
        edges = pl.DataFrame({
            "uid1": ["X"],
            "uid2": ["Y"],
            "rel_sum2": [3.14],
        })
        g = build_graph(edges)
        assert g.vcount() == 2
        assert g.ecount() == 1
        assert g.es[0]["weight"] == pytest.approx(3.14)


# ── Missing columns ─────────────────────────────────────────


class TestBuildGraphValidation:
    def test_missing_uid1(self):
        edges = pl.DataFrame({"uid2": ["B"], "rel_sum2": [1.0]})
        with pytest.raises(ValueError, match="uid1"):
            build_graph(edges)

    def test_missing_uid2(self):
        edges = pl.DataFrame({"uid1": ["A"], "rel_sum2": [1.0]})
        with pytest.raises(ValueError, match="uid2"):
            build_graph(edges)

    def test_missing_weight(self):
        edges = pl.DataFrame({"uid1": ["A"], "uid2": ["B"]})
        with pytest.raises(ValueError, match="rel_sum2"):
            build_graph(edges)

    def test_extra_columns_ignored(self):
        edges = pl.DataFrame({
            "uid1": ["A"],
            "uid2": ["B"],
            "rel_sum2": [1.0],
            "extra": [42],
        })
        g = build_graph(edges)
        assert g.vcount() == 2


# ── giant_component ──────────────────────────────────────────


class TestGiantComponent:
    def test_connected_graph_unchanged(self, simple_edges):
        g = build_graph(simple_edges)
        gc = giant_component(g)
        assert gc.vcount() == g.vcount()

    def test_disconnected_returns_largest(self, disconnected_edges):
        g = build_graph(disconnected_edges)
        gc = giant_component(g)
        # Both components have 2 nodes; giant picks one
        assert gc.vcount() == 2

    def test_empty_graph(self):
        g = ig.Graph(n=0, directed=False)
        gc = giant_component(g)
        assert gc.vcount() == 0

    def test_larger_component_selected(self):
        """Giant component should be the larger of unequal components."""
        edges = pl.DataFrame({
            "uid1": ["A", "B", "D"],
            "uid2": ["B", "C", "E"],
            "rel_sum2": [1.0, 1.0, 1.0],
        })
        g = build_graph(edges)
        gc = giant_component(g)
        # {A,B,C} has 3 nodes, {D,E} has 2 — giant is 3
        assert gc.vcount() == 3
