"""Tests for multi-layer edge combination pipeline."""

import numpy as np
import polars as pl
import pytest

from sciscape.linkage.combine import combine_edge_layers


def _make_edges(pairs, weight=1.0):
    """Helper: create edge DataFrame from (uid1, uid2) pairs."""
    return pl.DataFrame({
        "uid1": [p[0] for p in pairs],
        "uid2": [p[1] for p in pairs],
        "rel_sum2": [weight] * len(pairs),
    })


class TestCombineEdgeLayers:

    def test_sum_adds_weights(self):
        bc = _make_edges([("A", "B"), ("B", "C")], weight=1.0)
        cc = _make_edges([("A", "B"), ("C", "D")], weight=2.0)
        result = combine_edge_layers({"bc": bc, "cc": cc}, strategy="sum", gcc=False, top_k=0)
        ab = result.filter((pl.col("uid1") == "A") & (pl.col("uid2") == "B"))
        assert ab.height == 1
        assert ab["rel_sum2"][0] == pytest.approx(3.0)  # 1 + 2

    def test_boosted_multiplies_by_n_layers(self):
        bc = _make_edges([("A", "B")], weight=1.0)
        cc = _make_edges([("A", "B")], weight=1.0)
        dc = _make_edges([("A", "B")], weight=1.0)
        result = combine_edge_layers(
            {"bc": bc, "cc": cc, "dc": dc},
            strategy="consensus", gcc=False, top_k=0,
        )
        ab = result.filter((pl.col("uid1") == "A") & (pl.col("uid2") == "B"))
        # sum = 3.0, n_layers = 3, boosted = 3 × 3 = 9
        assert ab["rel_sum2"][0] == pytest.approx(9.0)

    def test_boosted_single_layer_no_boost(self):
        bc = _make_edges([("A", "B")], weight=2.0)
        cc = _make_edges([("C", "D")], weight=3.0)
        result = combine_edge_layers(
            {"bc": bc, "cc": cc},
            strategy="consensus", gcc=False, top_k=0,
        )
        ab = result.filter((pl.col("uid1") == "A") & (pl.col("uid2") == "B"))
        # only in bc → n_layers=1, boosted = 2.0 × 1 = 2.0
        assert ab["rel_sum2"][0] == pytest.approx(2.0)

    def test_max_keeps_maximum(self):
        bc = _make_edges([("A", "B")], weight=1.0)
        cc = _make_edges([("A", "B")], weight=5.0)
        result = combine_edge_layers({"bc": bc, "cc": cc}, strategy="max", gcc=False, top_k=0)
        ab = result.filter((pl.col("uid1") == "A") & (pl.col("uid2") == "B"))
        assert ab["rel_sum2"][0] == pytest.approx(5.0)

    def test_vote_counts_layers(self):
        bc = _make_edges([("A", "B")], weight=100.0)
        cc = _make_edges([("A", "B")], weight=0.001)
        dc = _make_edges([("A", "B")], weight=50.0)
        result = combine_edge_layers(
            {"bc": bc, "cc": cc, "dc": dc},
            strategy="vote", gcc=False, top_k=0,
        )
        ab = result.filter((pl.col("uid1") == "A") & (pl.col("uid2") == "B"))
        assert ab["rel_sum2"][0] == pytest.approx(3.0)  # 3 layers

    def test_rank_normalization(self):
        # 3 edges: weights 10, 5, 1 → ranks 1, 2, 3 → 1/rank: 1.0, 0.5, 0.333
        bc = pl.DataFrame({
            "uid1": ["A", "B", "C"],
            "uid2": ["B", "C", "D"],
            "rel_sum2": [10.0, 5.0, 1.0],
        })
        result = combine_edge_layers({"bc": bc}, strategy="rank", gcc=False, top_k=0)
        # After 1/rank: AB=1.0, BC=0.5, CD=0.333
        ab = result.filter((pl.col("uid1") == "A") & (pl.col("uid2") == "B"))
        assert ab["rel_sum2"][0] == pytest.approx(1.0, abs=0.01)

    def test_gcc_removes_isolated(self):
        # A-B-C connected, D-E isolated
        edges = _make_edges([("A", "B"), ("B", "C"), ("D", "E")])
        result = combine_edge_layers({"l1": edges}, strategy="sum", gcc=True, top_k=0)
        uids = set(result["uid1"].to_list()) | set(result["uid2"].to_list())
        assert "D" not in uids
        assert "A" in uids

    def test_empty_layers(self):
        empty = pl.DataFrame({"uid1": [], "uid2": [], "rel_sum2": []})
        result = combine_edge_layers({"l1": empty}, strategy="sum", gcc=False, top_k=0)
        assert result.height == 0

    def test_top_k_filters(self):
        # Dense graph: A-B, A-C, A-D, B-C, B-D, C-D (complete K4)
        # Each node has degree 3, top-1 should remove some
        edges = pl.DataFrame({
            "uid1": ["A", "A", "A", "B", "B", "C"],
            "uid2": ["B", "C", "D", "C", "D", "D"],
            "rel_sum2": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        })
        result_full = combine_edge_layers({"l1": edges}, strategy="sum", gcc=False, top_k=0)
        result_k1 = combine_edge_layers({"l1": edges}, strategy="sum", gcc=False, top_k=1)
        assert result_k1.height < result_full.height


class TestAutoGamma:

    def test_import(self):
        from sciscape.clustering.auto_gamma import find_gamma, AutoGammaResult
        assert find_gamma is not None

    def test_small_graph(self):
        """Auto-gamma on tiny graph should return a valid γ."""
        from sciscape.clustering.auto_gamma import find_gamma
        from sciscape.clustering.leiden_rust import RUST_AVAILABLE
        if not RUST_AVAILABLE:
            pytest.skip("Rust Leiden not available")

        # 30 nodes, 3 clusters of 10
        edges = []
        for c in range(3):
            for i in range(c * 10, (c + 1) * 10):
                for j in range(i + 1, (c + 1) * 10):
                    edges.append((f"N{i}", f"N{j}", np.random.uniform(0.5, 1.0)))
        # Weak cross-cluster
        for _ in range(5):
            edges.append((f"N{np.random.randint(0,10)}", f"N{np.random.randint(10,20)}", 0.01))

        df = pl.DataFrame({
            "uid1": [e[0] for e in edges],
            "uid2": [e[1] for e in edges],
            "rel_sum2": [e[2] for e in edges],
        })

        result = find_gamma(df, target_max_pct=50.0, gamma_range=(1e-4, 1.0),
                            n_coarse=4, max_refine=2, min_size=5, postprocess=False)
        assert result.gamma > 0
        assert result.n_clusters >= 1
        assert result.max_pct <= 60  # roughly within target


class TestLandscapeConfig:

    def test_multilayer_config(self):
        from sciscape.landscape import LandscapeConfig
        cfg = LandscapeConfig(
            layer_paths={"bc": "bc.parquet", "cc": "cc.parquet"},
            combine_strategy="consensus",
            auto_gamma=True,
        )
        assert cfg.layer_paths is not None
        assert len(cfg.layer_paths) == 2
        assert cfg.combine_strategy == "consensus"
        assert cfg.auto_gamma is True

    def test_default_no_layers(self):
        from sciscape.landscape import LandscapeConfig
        cfg = LandscapeConfig()
        assert cfg.layer_paths is None
        assert cfg.auto_gamma is False
