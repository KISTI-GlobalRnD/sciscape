"""Tests for edge landscape visualization (edge_landscape.py)."""

import numpy as np
import polars as pl
import pytest

from sciscape.visualization.edge_landscape import (
    compute_edge_year_matrix,
    compute_multilayer_year_matrices,
)


# ── Helpers ──────────────────────────────────────────────────

def _make_edges_with_years():
    """Edges with known year mapping for testing."""
    edges = pl.DataFrame({
        "uid1": ["A", "A", "B", "C"],
        "uid2": ["B", "C", "D", "D"],
        "rel_sum2": [1.0, 2.0, 3.0, 4.0],
    })
    year_map = {"A": 2020, "B": 2021, "C": 2020, "D": 2022}
    return edges, year_map


# ── Tests: compute_edge_year_matrix ──────────────────────────

class TestEdgeYearMatrix:

    def test_basic_output(self):
        edges, year_map = _make_edges_with_years()
        result = compute_edge_year_matrix(edges, year_map)
        assert "count_matrix" in result
        assert "weight_sum_matrix" in result
        assert "years" in result
        assert result["years"] == [2020, 2021, 2022]

    def test_matrix_dimensions(self):
        edges, year_map = _make_edges_with_years()
        result = compute_edge_year_matrix(edges, year_map)
        n = len(result["years"])
        assert len(result["count_matrix"]) == n
        assert len(result["count_matrix"][0]) == n

    def test_symmetry(self):
        """Count matrix should be symmetric."""
        edges, year_map = _make_edges_with_years()
        result = compute_edge_year_matrix(edges, year_map)
        mat = np.array(result["count_matrix"])
        np.testing.assert_array_equal(mat, mat.T)

    def test_weight_symmetry(self):
        edges, year_map = _make_edges_with_years()
        result = compute_edge_year_matrix(edges, year_map)
        mat = np.array(result["weight_sum_matrix"])
        np.testing.assert_array_almost_equal(mat, mat.T)

    def test_total_edges(self):
        edges, year_map = _make_edges_with_years()
        result = compute_edge_year_matrix(edges, year_map)
        # Each edge counted twice (symmetric), so total / 2 = n_edges
        total = result.get("total_edges", sum(sum(r) for r in result["count_matrix"]) // 2)
        assert total == 4

    def test_explicit_year_range(self):
        edges, year_map = _make_edges_with_years()
        result = compute_edge_year_matrix(edges, year_map, year_range=(2019, 2023))
        assert result["years"] == [2019, 2020, 2021, 2022, 2023]
        assert len(result["count_matrix"]) == 5

    def test_missing_years(self):
        """Nodes not in year_map should be filtered out."""
        edges = pl.DataFrame({
            "uid1": ["A", "X"],
            "uid2": ["B", "Y"],
            "rel_sum2": [1.0, 2.0],
        })
        year_map = {"A": 2020, "B": 2021}  # X, Y not in map
        result = compute_edge_year_matrix(edges, year_map)
        total = sum(sum(r) for r in result["count_matrix"]) // 2
        assert total == 1  # only A-B counted

    def test_empty_edges(self):
        edges = pl.DataFrame({"uid1": [], "uid2": [], "rel_sum2": []})
        result = compute_edge_year_matrix(edges, {})
        assert "error" in result

    def test_float_years(self):
        """Float years should be cast to int."""
        edges = pl.DataFrame({
            "uid1": ["A"], "uid2": ["B"], "rel_sum2": [1.0],
        })
        year_map = {"A": 2020.5, "B": 2021.7}
        result = compute_edge_year_matrix(edges, year_map)
        assert all(isinstance(y, int) for y in result["years"])

    def test_no_year_data(self):
        edges = pl.DataFrame({"uid1": ["A"], "uid2": ["B"], "rel_sum2": [1.0]})
        year_map = {"A": None, "B": 0}
        result = compute_edge_year_matrix(edges, year_map)
        assert "error" in result


class TestMultilayerYearMatrices:

    def test_multiple_layers(self):
        edges_bc = pl.DataFrame({
            "uid1": ["A", "B"], "uid2": ["B", "C"], "rel_sum2": [1.0, 2.0],
        })
        edges_cc = pl.DataFrame({
            "uid1": ["A"], "uid2": ["C"], "rel_sum2": [3.0],
        })
        year_map = {"A": 2020, "B": 2021, "C": 2022}
        layers = {"bc": edges_bc, "cc": edges_cc}

        result = compute_multilayer_year_matrices(layers, year_map)
        assert "bc" in result
        assert "cc" in result
        assert result["bc"]["years"] == [2020, 2021, 2022]
