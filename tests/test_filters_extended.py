"""Tests for adaptive top-k, Rust fast paths, and integer_remap_memory."""

import numpy as np
import polars as pl
import pytest

from sciscape.linkage.filters import compute_adaptive_k, filter_top_k, filter_giant_component
from sciscape.clustering.integer_remap import integer_remap_memory


# ── compute_adaptive_k ───────────────────────────────────────

class TestAdaptiveK:

    def test_small_graph(self):
        assert compute_adaptive_k(50) == 7

    def test_medium_graph(self):
        assert compute_adaptive_k(300) == 17

    def test_large_graph_capped(self):
        assert compute_adaptive_k(1000) == 30
        assert compute_adaptive_k(100000) == 30

    def test_tiny_graph_floor(self):
        assert compute_adaptive_k(1) == 5
        assert compute_adaptive_k(0) == 5

    def test_custom_bounds(self):
        assert compute_adaptive_k(100, k_min=3, k_max=15) == 10
        assert compute_adaptive_k(4, k_min=3, k_max=15) == 3


# ── filter_top_k with auto top_k ────────────────────────────

class TestFilterTopK:

    def _make_star(self, n=50):
        """Star graph: node 0 connected to all others."""
        return pl.DataFrame({
            "uid1": ["0"] * (n - 1),
            "uid2": [str(i) for i in range(1, n)],
            "rel_sum2": np.random.exponential(1.0, n - 1).tolist(),
        })

    def test_basic_filter(self):
        df = self._make_star(50)
        result = filter_top_k(df, 10)
        assert result.height <= df.height
        assert result.height > 0

    def test_k_larger_than_degree(self):
        """If k > max degree, all edges should survive."""
        df = self._make_star(10)
        result = filter_top_k(df, 100)
        assert result.height == df.height

    def test_symmetric_mode(self):
        df = self._make_star(20)
        sym = filter_top_k(df, 5, mode="symmetric")
        mut = filter_top_k(df, 5, mode="mutual")
        # symmetric keeps more edges than mutual
        assert sym.height >= mut.height

    def test_empty_edges(self):
        df = pl.DataFrame({"uid1": [], "uid2": [], "rel_sum2": []})
        result = filter_top_k(df, 10)
        assert result.height == 0

    def test_columns_preserved(self):
        df = self._make_star(20)
        result = filter_top_k(df, 5)
        assert set(result.columns) == {"uid1", "uid2", "rel_sum2"}


# ── filter_giant_component ───────────────────────────────────

class TestFilterGCC:

    def test_connected_graph(self):
        df = pl.DataFrame({
            "uid1": ["A", "B", "C"],
            "uid2": ["B", "C", "A"],
            "rel_sum2": [1.0, 1.0, 1.0],
        })
        result = filter_giant_component(df)
        assert result.height == 3

    def test_disconnected_graph(self):
        df = pl.DataFrame({
            "uid1": ["A", "B", "X"],
            "uid2": ["B", "C", "Y"],
            "rel_sum2": [1.0, 1.0, 1.0],
        })
        result = filter_giant_component(df)
        # GCC is {A,B,C} = 2 edges
        assert result.height == 2

    def test_empty_graph(self):
        df = pl.DataFrame({"uid1": [], "uid2": [], "rel_sum2": []})
        result = filter_giant_component(df)
        assert result.height == 0


# ── integer_remap_memory ─────────────────────────────────────

class TestIntegerRemapMemory:

    def test_basic(self):
        df = pl.DataFrame({
            "uid1": ["A", "B", "C"],
            "uid2": ["B", "C", "A"],
            "rel_sum2": [1.0, 2.0, 3.0],
        })
        src, dst, w, n, uids = integer_remap_memory(df)
        assert n == 3
        assert len(uids) == 3
        assert len(src) == 3
        assert len(dst) == 3
        assert w.dtype == np.float64
        assert src.dtype == np.uint32

    def test_node_count(self):
        df = pl.DataFrame({
            "uid1": ["A", "A", "B"],
            "uid2": ["B", "C", "C"],
            "rel_sum2": [1.0, 1.0, 1.0],
        })
        _, _, _, n, uids = integer_remap_memory(df)
        assert n == 3
        assert set(uids) == {"A", "B", "C"}

    def test_weight_preservation(self):
        df = pl.DataFrame({
            "uid1": ["A"], "uid2": ["B"], "rel_sum2": [3.14],
        })
        _, _, w, _, _ = integer_remap_memory(df)
        assert abs(w[0] - 3.14) < 1e-10

    def test_empty_edges(self):
        df = pl.DataFrame({"uid1": [], "uid2": [], "rel_sum2": []})
        src, dst, w, n, uids = integer_remap_memory(df)
        assert n == 0
        assert len(src) == 0
