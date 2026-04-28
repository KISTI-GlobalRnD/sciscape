"""Tests for consensus visualization (consensus.py)."""

import polars as pl
import pytest

from sciscape.visualization.consensus import (
    compute_consensus_stats,
    compute_consensus_vs_cluster,
    format_consensus_report,
    consensus_to_plotly,
)


# ── Helpers ──────────────────────────────────────────────────

def _make_layer(pairs, weight=1.0):
    return pl.DataFrame({
        "uid1": [p[0] for p in pairs],
        "uid2": [p[1] for p in pairs],
        "rel_sum2": [weight] * len(pairs),
    })


# ── Tests: compute_consensus_stats ───────────────────────────

class TestConsensusStats:

    def test_single_layer(self):
        layers = {"bc": _make_layer([("A", "B"), ("B", "C")])}
        stats = compute_consensus_stats(layers, top_k=0)
        assert stats["n_layers"] == 1
        assert stats["total_edges"] == 2
        assert 1 in stats["n_layers_distribution"]

    def test_two_layers_overlap(self):
        layers = {
            "bc": _make_layer([("A", "B"), ("B", "C"), ("C", "D")]),
            "cc": _make_layer([("A", "B"), ("C", "D"), ("D", "E")]),
        }
        stats = compute_consensus_stats(layers, top_k=0)
        assert stats["n_layers"] == 2
        # A-B and C-D appear in both layers
        assert stats["n_layers_distribution"].get(2, 0) == 2
        assert stats["backbone_size"] == 2

    def test_no_overlap(self):
        layers = {
            "bc": _make_layer([("A", "B")]),
            "cc": _make_layer([("C", "D")]),
        }
        stats = compute_consensus_stats(layers, top_k=0)
        assert stats["backbone_size"] == 0
        assert stats["n_layers_distribution"].get(1, 0) == 2

    def test_overlap_matrix(self):
        layers = {
            "bc": _make_layer([("A", "B"), ("B", "C")]),
            "cc": _make_layer([("A", "B"), ("D", "E")]),
        }
        stats = compute_consensus_stats(layers, top_k=0)
        assert "overlap_matrix" in stats
        assert "bc_cc" in stats["overlap_matrix"]
        assert stats["overlap_matrix"]["bc_cc"] == 1  # A-B shared

    def test_per_layer_coverage(self):
        layers = {
            "bc": _make_layer([("A", "B"), ("B", "C")]),
            "cc": _make_layer([("X", "Y")]),
        }
        stats = compute_consensus_stats(layers, top_k=0)
        assert stats["per_layer_coverage"]["bc"] == 3  # A, B, C
        assert stats["per_layer_coverage"]["cc"] == 2  # X, Y

    def test_empty_layers(self):
        empty = pl.DataFrame({"uid1": [], "uid2": [], "rel_sum2": []})
        layers = {"bc": empty, "cc": empty}
        stats = compute_consensus_stats(layers, top_k=0)
        assert stats["total_edges"] == 0

    def test_with_top_k(self):
        # Many edges, top_k filters to fewer
        pairs = [(f"n{i}", f"n{j}") for i in range(10) for j in range(i+1, 10)]
        layers = {"bc": _make_layer(pairs)}
        stats = compute_consensus_stats(layers, top_k=3)
        assert stats["total_edges"] <= len(pairs)


# ── Tests: compute_consensus_vs_cluster ──────────────────────

class TestConsensusVsCluster:

    def test_intra_vs_cross(self):
        layers = {
            "bc": _make_layer([("A", "B"), ("A", "C"), ("B", "D")]),
        }
        membership = {"A": 0, "B": 0, "C": 0, "D": 1}
        result = compute_consensus_vs_cluster(layers, membership, top_k=0)
        assert 1 in result
        assert result[1]["intra"] >= 1  # A-B, A-C are intra
        assert result[1]["cross"] >= 1  # B-D is cross

    def test_empty(self):
        layers = {"bc": _make_layer([])}
        result = compute_consensus_vs_cluster(layers, {}, top_k=0)
        assert result == {}


# ── Tests: format_consensus_report ───────────────────────────

class TestFormatReport:

    def test_produces_string(self):
        stats = {
            "n_layers": 2,
            "total_edges": 100,
            "backbone_size": 10,
            "n_layers_distribution": {1: 60, 2: 40},
            "per_layer_coverage": {"bc": 500, "cc": 400},
            "per_layer_edges": {"bc": 1000, "cc": 800},
        }
        report = format_consensus_report(stats)
        assert "Consensus" in report
        assert "backbone" in report.lower() or "Backbone" in report


# ── Tests: consensus_to_plotly ───────────────────────────────

class TestPlotly:

    def test_produces_figures(self):
        stats = {
            "n_layers_distribution": {1: 60, 2: 40},
            "overlap_layer_names": ["bc", "cc"],
            "overlap_matrix": {"bc_bc": 100, "bc_cc": 20, "cc_bc": 20, "cc_cc": 80},
        }
        figs = consensus_to_plotly(stats)
        assert "distribution" in figs
        assert "overlap_heatmap" in figs
