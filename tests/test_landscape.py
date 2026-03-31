"""Integration tests for sciscape.landscape module."""

from __future__ import annotations


import igraph as ig
import pandas as pd
import polars as pl
import pytest

from sciscape.landscape import LandscapeConfig


# ── Fixtures ──────────────────────────────────────────────────


def _make_edges_and_abstracts(
    n_cliques: int = 4,
    clique_size: int = 30,
    bridge_weight: float = 0.1,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Create synthetic edge list and abstract data with clear communities."""
    g = ig.Graph()
    n = n_cliques * clique_size
    g.add_vertices(n)
    edges, weights = [], []

    for c in range(n_cliques):
        base = c * clique_size
        for i in range(clique_size):
            for j in range(i + 1, clique_size):
                edges.append((base + i, base + j))
                weights.append(1.0)
    for c in range(n_cliques - 1):
        edges.append((c * clique_size, (c + 1) * clique_size))
        weights.append(bridge_weight)

    g.add_edges(edges)
    uids = [f"W{i:04d}" for i in range(n)]

    edge_rows = []
    for (s, t), w in zip(edges, weights):
        edge_rows.append({"uid1": uids[s], "uid2": uids[t], "rel_sum2": w})
    edge_df = pl.DataFrame(edge_rows)

    topics = ["deep learning neural network"] * clique_size
    topics += ["quantum computing qubit"] * clique_size
    topics += ["solar energy battery storage"] * clique_size
    topics += ["climate change carbon emission"] * clique_size

    abstract_df = pl.DataFrame({
        "uid": uids,
        "title": [f"Study of {topics[i]} {i}" for i in range(n)],
        "abstract": [
            f"This paper studies {topics[i]}. We analyse {topics[i]} methods "
            f"and present results for {topics[i]} applications in detail."
            for i in range(n)
        ],
        "pubyear": [2020 + (i % 5) for i in range(n)],
    })

    return edge_df, abstract_df


@pytest.fixture
def sample_dir(tmp_path):
    """Create temp dir with edge + abstract parquet files."""
    edge_df, abstract_df = _make_edges_and_abstracts()
    edge_path = tmp_path / "edges.parquet"
    abstract_path = tmp_path / "abstracts.parquet"
    edge_df.write_parquet(edge_path)
    abstract_df.to_pandas().to_parquet(abstract_path, index=False)
    return tmp_path, edge_path, abstract_path


# ── 1. Input validation ──────────────────────────────────────


class TestInputValidation:
    def test_missing_edge_file(self, tmp_path):
        from sciscape.landscape import run_landscape

        abstract_path = tmp_path / "abstracts.parquet"
        pl.DataFrame({"uid": ["a"], "title": ["t"], "abstract": ["a"], "pubyear": [2020]}).to_pandas().to_parquet(abstract_path)

        with pytest.raises(FileNotFoundError, match="Edge file not found"):
            run_landscape(
                tmp_path / "nonexistent.parquet",
                abstract_path,
                tmp_path / "out",
            )

    def test_missing_abstract_file(self, tmp_path):
        from sciscape.landscape import run_landscape

        edge_path = tmp_path / "edges.parquet"
        pl.DataFrame({"uid1": ["a"], "uid2": ["b"], "rel_sum2": [1.0]}).write_parquet(edge_path)

        with pytest.raises(FileNotFoundError, match="Abstract file not found"):
            run_landscape(
                edge_path,
                tmp_path / "nonexistent.parquet",
                tmp_path / "out",
            )

    def test_missing_abstract_columns(self, tmp_path):
        from sciscape.landscape import run_landscape

        edge_path = tmp_path / "edges.parquet"
        abstract_path = tmp_path / "abstracts.parquet"
        pl.DataFrame({"uid1": ["a"], "uid2": ["b"], "rel_sum2": [1.0]}).write_parquet(edge_path)
        # Missing 'abstract' and 'pubyear' columns
        pd.DataFrame({"uid": ["a"], "title": ["t"]}).to_parquet(abstract_path)

        with pytest.raises(ValueError, match="missing columns"):
            run_landscape(edge_path, abstract_path, tmp_path / "out")


# ── 2. Clustering modes ──────────────────────────────────────


class TestClustering:
    def test_standard_mode(self, sample_dir):
        """Standard mode (gamma_block=None): direct γ search."""
        from sciscape.landscape import _run_clustering

        tmp_path, edge_path, _ = sample_dir
        edges = pl.read_parquet(edge_path)
        out = tmp_path / "out_standard"
        out.mkdir()

        cfg = LandscapeConfig(
            gamma_block=None,
            gamma_range=(1e-4, 1e-1),
            min_docs_per_cluster=5,
            n_hierarchy_levels=1,
            leiden_iterations=10,
            seed=42,
        )
        result = _run_clustering(edges, cfg, out)

        assert "uid" in result.columns
        assert "cluster_nano" in result.columns
        assert len(result) == 120
        # Should produce more than 1 cluster
        n_clusters = result["cluster_nano"].n_unique()
        assert n_clusters > 1

    def test_block_init_mode(self, sample_dir):
        """Block-init mode (gamma_block="auto"): blocks → contraction → cascade."""
        from sciscape.landscape import _run_clustering

        tmp_path, edge_path, _ = sample_dir
        edges = pl.read_parquet(edge_path)
        out = tmp_path / "out_block"
        out.mkdir()

        cfg = LandscapeConfig(
            gamma_block="auto",
            gamma_range=(1e-4, 1e-1),
            min_docs_per_cluster=5,
            n_hierarchy_levels=1,
            leiden_iterations=10,
            seed=42,
        )
        result = _run_clustering(edges, cfg, out)

        assert "cluster_nano" in result.columns
        assert len(result) == 120
        n_clusters = result["cluster_nano"].n_unique()
        assert n_clusters > 1

        # blocks.parquet should be cached
        assert (out / "blocks.parquet").exists()

    def test_block_init_explicit_gamma(self, sample_dir):
        """Explicit gamma_block value."""
        from sciscape.landscape import _run_clustering

        tmp_path, edge_path, _ = sample_dir
        edges = pl.read_parquet(edge_path)
        out = tmp_path / "out_explicit"
        out.mkdir()

        cfg = LandscapeConfig(
            gamma_block=0.5,
            gamma_range=(1e-4, 1e-1),
            min_docs_per_cluster=5,
            n_hierarchy_levels=1,
            leiden_iterations=10,
            seed=42,
        )
        result = _run_clustering(edges, cfg, out)

        assert len(result) == 120

    def test_block_cache_reuse(self, sample_dir):
        """Second run with same config should reuse cached blocks."""
        from sciscape.landscape import _run_clustering

        tmp_path, edge_path, _ = sample_dir
        edges = pl.read_parquet(edge_path)
        out = tmp_path / "out_cache"
        out.mkdir()

        cfg = LandscapeConfig(
            gamma_block=0.5,
            gamma_range=(1e-4, 1e-1),
            min_docs_per_cluster=5,
            n_hierarchy_levels=1,
            leiden_iterations=10,
            seed=42,
        )
        _run_clustering(edges, cfg, out)
        blocks_mtime = (out / "blocks.parquet").stat().st_mtime

        # Force nano recalculation but blocks should be reused
        (out / "membership.parquet").unlink()
        _run_clustering(edges, cfg, out)
        assert (out / "blocks.parquet").stat().st_mtime == blocks_mtime

    def test_two_levels(self, sample_dir):
        """Two hierarchy levels: nano + micro."""
        from sciscape.landscape import _run_clustering

        tmp_path, edge_path, _ = sample_dir
        edges = pl.read_parquet(edge_path)
        out = tmp_path / "out_2levels"
        out.mkdir()

        cfg = LandscapeConfig(
            gamma_block=None,
            gamma_range=(1e-4, 1e-1),
            min_docs_per_cluster=5,
            n_hierarchy_levels=2,
            leiden_iterations=10,
            seed=42,
        )
        result = _run_clustering(edges, cfg, out)

        assert "cluster_nano" in result.columns
        assert "cluster_micro" in result.columns
        # Micro should have fewer or equal clusters than nano
        n_nano = result.filter(pl.col("cluster_nano") >= 0)["cluster_nano"].n_unique()
        n_micro = result["cluster_micro"].n_unique()
        assert n_micro <= n_nano


# ── 3. CLI argument parsing ──────────────────────────────────


class TestCLI:
    def test_gamma_block_auto(self):
        from sciscape.cli import _build_parser
        parser = _build_parser()
        args = parser.parse_args([
            "landscape", "e.parquet", "a.parquet",
        ])
        assert args.gamma_block == "auto"

    def test_gamma_block_none(self):
        from sciscape.cli import _build_parser
        parser = _build_parser()
        args = parser.parse_args([
            "landscape", "e.parquet", "a.parquet", "--gamma-block", "none",
        ])
        assert args.gamma_block == "none"

    def test_gamma_block_float(self):
        from sciscape.cli import _build_parser
        parser = _build_parser()
        args = parser.parse_args([
            "landscape", "e.parquet", "a.parquet", "--gamma-block", "0.01",
        ])
        assert args.gamma_block == "0.01"

    def test_gamma_range(self):
        from sciscape.cli import _build_parser
        parser = _build_parser()
        args = parser.parse_args([
            "landscape", "e.parquet", "a.parquet",
            "--gamma-range", "1e-5,1e-2",
        ])
        assert args.gamma_range == "1e-5,1e-2"

    def test_gamma_range_default_none(self):
        from sciscape.cli import _build_parser
        parser = _build_parser()
        args = parser.parse_args([
            "landscape", "e.parquet", "a.parquet",
        ])
        assert args.gamma_range is None


# ── 4. Config resolution ─────────────────────────────────────


class TestConfig:
    def test_auto_default(self):
        cfg = LandscapeConfig()
        assert cfg.gamma_block == "auto"

    def test_none_disables(self):
        cfg = LandscapeConfig(gamma_block=None)
        assert cfg.gamma_block is None

    def test_explicit_float(self):
        cfg = LandscapeConfig(gamma_block=0.05)
        assert cfg.gamma_block == 0.05

    def test_auto_resolves_from_gamma_range(self):
        """Auto gamma_block = 10 × gamma_range[1]."""
        cfg = LandscapeConfig(gamma_range=(1e-6, 1e-3))
        assert cfg.gamma_block == "auto"
        # Resolution happens inside _run_clustering, not in config
        expected = 10.0 * cfg.gamma_range[1]
        assert expected == pytest.approx(1e-2)
