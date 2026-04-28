"""Tests for sciscape.clustering.pipeline — run_pipeline orchestrator."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import polars as pl
import pytest

from sciscape.clustering.config import ClusterTables, LeidenConfig
from sciscape.clustering.hierarchy import build_cluster_tables
from sciscape.clustering.pipeline import (
    _compact_membership_array,
    _resolve_stability_seeds,
    run_pipeline,
)
from sciscape.clustering.leiden_rust import RUST_AVAILABLE


class TestCompactMembershipArray:
    def test_downcasts_nonnegative_cluster_ids_to_uint32(self):
        arr = np.array([0, 2, 1, 2], dtype=np.uint64)

        compact = _compact_membership_array(arr)

        assert compact.dtype == np.uint32
        assert compact.tolist() == [0, 2, 1, 2]

    def test_preserves_cluster_ids_beyond_uint32(self):
        wide = np.array([0, np.iinfo(np.uint32).max + 1], dtype=np.uint64)

        compact = _compact_membership_array(wide)

        assert compact.dtype == np.uint64
        assert compact.tolist() == wide.tolist()


class TestBuildClusterTables:
    def test_assigns_sorted_dense_indices_per_hierarchy_level(self):
        df = pl.DataFrame(
            {
                "uid": ["A", "B", "C", "D", "E"],
                "cluster_macro": [20, 10, 20, 10, 10],
                "cluster_nano": [3, 5, 2, 5, 7],
            }
        )

        tables = build_cluster_tables(df, levels=("macro", "nano"))
        rows = (
            tables.membership
            .sort("uid")
            .select("uid", "macro", "nano", "total_index")
            .to_dicts()
        )

        assert rows == [
            {"uid": "A", "macro": 2, "nano": 2, "total_index": "2.2"},
            {"uid": "B", "macro": 1, "nano": 1, "total_index": "1.1"},
            {"uid": "C", "macro": 2, "nano": 1, "total_index": "2.1"},
            {"uid": "D", "macro": 1, "nano": 1, "total_index": "1.1"},
            {"uid": "E", "macro": 1, "nano": 2, "total_index": "1.2"},
        ]

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust backend required")
    def test_assigns_dense_indices_with_rust_u32_helper(self):
        df = pl.DataFrame(
            {
                "uid": ["A", "B", "C", "D", "E"],
                "cluster_macro": pl.Series([20, 10, 20, 10, 10], dtype=pl.UInt32),
                "cluster_nano": pl.Series([3, 5, 2, 5, 7], dtype=pl.UInt32),
            }
        )

        tables = build_cluster_tables(df, levels=("macro", "nano"))
        rows = (
            tables.membership
            .sort("uid")
            .select("uid", "macro", "nano", "total_index")
            .to_dicts()
        )

        assert rows == [
            {"uid": "A", "macro": 2, "nano": 2, "total_index": "2.2"},
            {"uid": "B", "macro": 1, "nano": 1, "total_index": "1.1"},
            {"uid": "C", "macro": 2, "nano": 1, "total_index": "2.1"},
            {"uid": "D", "macro": 1, "nano": 1, "total_index": "1.1"},
            {"uid": "E", "macro": 1, "nano": 2, "total_index": "1.2"},
        ]


# ── _resolve_stability_seeds ─────────────────────────────────


class TestResolveStabilitySeeds:
    def test_default_no_seed(self):
        cfg = LeidenConfig()
        seeds = _resolve_stability_seeds(cfg)
        assert seeds == (0, 1, 2)

    def test_explicit_seed_generates_triple(self):
        cfg = LeidenConfig(seed=10)
        seeds = _resolve_stability_seeds(cfg)
        assert seeds == (10, 11, 12)

    def test_explicit_stability_seeds(self):
        cfg = LeidenConfig(stability_seeds=[5, 10, 15])
        seeds = _resolve_stability_seeds(cfg)
        assert seeds == (5, 10, 15)

    def test_stability_seeds_deduplicated(self):
        cfg = LeidenConfig(stability_seeds=[1, 1, 2, 2, 3])
        seeds = _resolve_stability_seeds(cfg)
        assert seeds == (1, 2, 3)

    def test_stability_seeds_override_seed(self):
        cfg = LeidenConfig(seed=42, stability_seeds=[7, 8])
        seeds = _resolve_stability_seeds(cfg)
        assert seeds == (7, 8)


# ── Config validation ────────────────────────────────────────


class TestConfigValidation:
    def test_no_resolutions_no_constraints_raises(self, tmp_path):
        """Pipeline requires either resolutions or level_constraints."""
        edges = pl.DataFrame({
            "uid1": ["A", "B"],
            "uid2": ["B", "C"],
            "rel_sum2": [1.0, 1.0],
        })
        edge_path = tmp_path / "edges.parquet"
        edges.write_parquet(edge_path)

        cfg = LeidenConfig()  # neither resolutions nor level_constraints
        with pytest.raises(ValueError, match="resolutions.*level_constraints"):
            run_pipeline(edge_path, None, cfg)


# ── Synthetic integration ────────────────────────────────────


def _make_clique_edges(n: int = 8) -> pl.DataFrame:
    """Build a fully connected clique of *n* nodes as an edge DataFrame."""
    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            rows.append({"uid1": f"N{i}", "uid2": f"N{j}", "rel_sum2": 1.0})
    return pl.DataFrame(rows)


def _make_two_clique_edges(
    clique_size: int = 6,
    bridge_weight: float = 0.05,
) -> pl.DataFrame:
    """Two cliques connected by a weak bridge -- produces 2 communities."""
    rows = []
    for i in range(clique_size):
        for j in range(i + 1, clique_size):
            rows.append({"uid1": f"N{i}", "uid2": f"N{j}", "rel_sum2": 1.0})
    offset = clique_size
    for i in range(clique_size):
        for j in range(i + 1, clique_size):
            rows.append({
                "uid1": f"N{offset + i}",
                "uid2": f"N{offset + j}",
                "rel_sum2": 1.0,
            })
    # Bridge between the two cliques
    rows.append({"uid1": "N0", "uid2": f"N{offset}", "rel_sum2": bridge_weight})
    return pl.DataFrame(rows)


class TestRunPipelineIntegration:
    def test_explicit_resolutions_single_level(self, tmp_path):
        """Single-level explicit resolution on a small clique."""
        edges = _make_clique_edges(8)
        edge_path = tmp_path / "edges.parquet"
        edges.write_parquet(edge_path)

        cfg = LeidenConfig(
            resolutions=OrderedDict({"nano": 0.01}),
            objective="cpm",
            seed=42,
        )
        tables = run_pipeline(edge_path, None, cfg)

        assert isinstance(tables, ClusterTables)
        assert "uid" in tables.membership.columns
        assert "cluster_nano" in tables.membership.columns
        assert tables.membership.height == 8
        assert tables.levels == ("nano",)

    def test_explicit_resolutions_two_levels(self, tmp_path):
        """Two-level explicit resolutions on a graph with community structure."""
        edges = _make_two_clique_edges(clique_size=6, bridge_weight=0.05)
        edge_path = tmp_path / "edges.parquet"
        edges.write_parquet(edge_path)

        cfg = LeidenConfig(
            resolutions=OrderedDict({"macro": 0.01, "nano": 0.5}),
            objective="cpm",
            seed=42,
        )
        tables = run_pipeline(edge_path, None, cfg)

        # Both levels should appear in the membership table
        assert "cluster_macro" in tables.membership.columns
        assert "cluster_nano" in tables.membership.columns
        assert tables.levels == ("macro", "nano")

    def test_output_description_table(self, tmp_path):
        """Description table should list cluster sizes."""
        edges = _make_clique_edges(8)
        edge_path = tmp_path / "edges.parquet"
        edges.write_parquet(edge_path)

        cfg = LeidenConfig(
            resolutions=OrderedDict({"nano": 0.01}),
            objective="cpm",
            seed=42,
        )
        tables = run_pipeline(edge_path, None, cfg)

        assert "number_of_nodes" in tables.description.columns
        total_nodes = tables.description["number_of_nodes"].sum()
        assert total_nodes == 8

    def test_resolutions_stored(self, tmp_path):
        edges = _make_clique_edges(6)
        edge_path = tmp_path / "edges.parquet"
        edges.write_parquet(edge_path)

        cfg = LeidenConfig(
            resolutions=OrderedDict({"nano": 0.05}),
            objective="cpm",
            seed=0,
        )
        tables = run_pipeline(edge_path, None, cfg)

        assert tables.resolutions is not None
        assert "nano" in tables.resolutions
        assert tables.resolutions["nano"] == pytest.approx(0.05)

    def test_progress_callback(self, tmp_path):
        """Progress callback receives messages during the run."""
        edges = _make_clique_edges(6)
        edge_path = tmp_path / "edges.parquet"
        edges.write_parquet(edge_path)

        messages = []
        cfg = LeidenConfig(
            resolutions=OrderedDict({"nano": 0.01}),
            objective="cpm",
            seed=42,
            progress=messages.append,
        )
        run_pipeline(edge_path, None, cfg)

        assert len(messages) > 0
        assert any("loaded edges" in m for m in messages)
        assert any("pipeline finished" in m for m in messages)

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust backend required")
    def test_rust_parquet_path_defers_eager_edge_load(self, tmp_path, monkeypatch):
        edges = _make_clique_edges(8)
        edge_path = tmp_path / "edges.parquet"
        edges.write_parquet(edge_path)

        import sciscape.clustering.pipeline as pipeline

        def fail_load(*args, **kwargs):
            raise AssertionError("load_edge_table should not be called")

        monkeypatch.setattr(pipeline, "load_edge_table", fail_load)
        cfg = LeidenConfig(
            resolutions=OrderedDict({"nano": 0.01}),
            objective="cpm",
            seed=42,
            backend="rust",
            log_dir=tmp_path,
            run_id="rust_direct_graph",
        )

        tables = pipeline.run_pipeline(edge_path, None, cfg)

        assert tables.membership.height == 8
        remap_dir = tmp_path / "rust_direct_graph" / "remap"
        assert not (remap_dir / "int_edges.parquet").exists()
        assert not (remap_dir / "src.u32.bin").exists()
        assert not (remap_dir / "dst.u32.bin").exists()
        assert not (remap_dir / "weight.f64.bin").exists()

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust backend required")
    def test_rust_level_constraints_reuse_native_search_membership(self, tmp_path):
        edges = _make_two_clique_edges(clique_size=4, bridge_weight=0.01)
        edge_path = tmp_path / "edges.parquet"
        edges.write_parquet(edge_path)

        messages = []
        cfg = LeidenConfig(
            level_constraints=[(2, 3)],
            resolution_bounds=(0.001, 1.0),
            max_iterations=8,
            objective="cpm",
            seed=42,
            backend="rust",
            progress=messages.append,
        )

        tables = run_pipeline(edge_path, None, cfg)

        assert tables.membership.height == 8
        assert any("reused rust native search membership" in msg for msg in messages)
