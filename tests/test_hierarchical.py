"""Tests for hierarchical clustering (hierarchical.py)."""

import numpy as np
import polars as pl
import pytest
import tempfile
from pathlib import Path

from sciscape.clustering.leiden_rust import (
    RUST_AVAILABLE,
    RUST_DONGDAEMUN_AVAILABLE,
    build_leiden_graph,
)

pytestmark = pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust backend required")

from sciscape.clustering.hierarchical import (
    build_hierarchy,
    HierarchyResult,
    HierarchyLevel,
    _contract_and_normalize,
    _contract_edges,
    _adaptive_contracted_k,
)
from sciscape.clustering.hierarchy_oversize_postprocess import (
    HierarchyPostprocessConfig,
    hierarchy_target_max_doc_weight,
    postprocess_config_hash,
    run_hierarchy_level_postprocess,
    trim_min_delta_q_for_policy,
)


# ── Helpers ──────────────────────────────────────────────────

def _make_hierarchical_graph(n_macro=3, n_micro_per_macro=4, n_per_micro=15):
    """Create graph with known 3-level structure.

    n_macro groups, each with n_micro_per_macro sub-groups,
    each sub-group has n_per_micro nodes. Dense within sub-group,
    medium within macro, weak across macro.
    """
    edges = []
    node_id = 0
    for macro in range(n_macro):
        macro_start = node_id
        for micro in range(n_micro_per_macro):
            micro_start = node_id
            for i in range(n_per_micro):
                for j in range(i + 1, n_per_micro):
                    edges.append((str(micro_start + i), str(micro_start + j), 1.0))
            node_id += n_per_micro
            # Medium edges between micro groups within same macro
            if micro > 0:
                for k in range(3):
                    edges.append((str(micro_start + k), str(micro_start - n_per_micro + k), 0.1))
        # Weak edges between macro groups
        if macro > 0:
            edges.append((str(macro_start), str(macro_start - n_macro * n_micro_per_macro * n_per_micro + 1), 0.01))

    return pl.DataFrame({
        "uid1": [e[0] for e in edges],
        "uid2": [e[1] for e in edges],
        "rel_sum2": [e[2] for e in edges],
    })


def _make_simple_graph(n=60):
    """Simple graph: 3 cliques of n/3 nodes."""
    edges = []
    k = n // 3
    for group in range(3):
        start = group * k
        for i in range(k):
            for j in range(i + 1, k):
                edges.append((str(start + i), str(start + j), 1.0))
        # Weak cross-edges
        if group > 0:
            edges.append((str(start), str(start - k), 0.01))
    return pl.DataFrame({
        "uid1": [e[0] for e in edges],
        "uid2": [e[1] for e in edges],
        "rel_sum2": [e[2] for e in edges],
    })


# ── Tests: build_hierarchy ───────────────────────────────────

class TestBuildHierarchy:

    def test_returns_hierarchy_result(self):
        df = _make_simple_graph(60)
        result = build_hierarchy(edges=df, n_levels=1,
                                 targets={"nano": 50.0}, min_sizes={"nano": 2})
        assert isinstance(result, HierarchyResult)
        assert result.n_nodes > 0
        assert len(result.levels) >= 1

    def test_single_level(self):
        df = _make_simple_graph(60)
        result = build_hierarchy(edges=df, n_levels=1,
                                 targets={"nano": 50.0}, min_sizes={"nano": 2})
        assert len(result.levels) == 1
        level = result.levels[0]
        assert level.name == "nano"
        assert level.n_clusters >= 2
        assert level.gamma > 0
        assert len(level.membership) == result.n_nodes

    def test_two_levels(self):
        df = _make_simple_graph(90)
        result = build_hierarchy(edges=df, n_levels=2,
                                 targets={"nano": 30.0, "micro": 60.0},
                                 min_sizes={"nano": 2, "micro": 2},
                                 stop_at_clusters=1)
        assert len(result.levels) >= 1
        # Second level should have fewer or equal clusters
        if len(result.levels) >= 2:
            assert result.levels[1].n_clusters <= result.levels[0].n_clusters

    def test_stop_at_clusters(self):
        df = _make_simple_graph(60)
        result = build_hierarchy(edges=df, n_levels=5, stop_at_clusters=10,
                                 targets={"nano": 50.0, "micro": 80.0},
                                 min_sizes={"nano": 2, "micro": 2})
        # Should stop early when cluster count is low enough
        last = result.levels[-1]
        assert last.n_clusters >= 1

    def test_to_dataframe(self):
        df = _make_simple_graph(60)
        result = build_hierarchy(edges=df, n_levels=1,
                                 targets={"nano": 50.0}, min_sizes={"nano": 2})
        n = result.n_nodes
        uids = [str(i) for i in range(n)]
        out_df = result.to_dataframe(uids)
        assert "uid" in out_df.columns
        assert "cluster_nano" in out_df.columns
        assert out_df.height == n

    def test_memberships_by_level(self):
        df = _make_simple_graph(60)
        result = build_hierarchy(edges=df, n_levels=1,
                                 targets={"nano": 50.0}, min_sizes={"nano": 2})
        mbl = result.memberships_by_level
        assert "nano" in mbl
        assert len(mbl["nano"]) == result.n_nodes

    def test_with_cache_dir(self):
        df = _make_simple_graph(60)
        with tempfile.TemporaryDirectory() as td:
            result1 = build_hierarchy(edges=df, n_levels=1, cache_dir=Path(td),
                                      targets={"nano": 50.0}, min_sizes={"nano": 2})
            # Verify files saved
            assert (Path(td) / "nano" / "membership.parquet").exists()
            assert (Path(td) / "nano" / "meta.json").exists()
            assert (Path(td) / "hierarchy.parquet").exists()

            # Second run should load from cache
            result2 = build_hierarchy(edges=df, n_levels=1, cache_dir=Path(td),
                                      targets={"nano": 50.0}, min_sizes={"nano": 2})
            assert result2.levels[0].n_clusters == result1.levels[0].n_clusters

    def test_postprocess_enabled_writes_summary_and_meta(self):
        import json

        df = _make_simple_graph(60)
        cfg = HierarchyPostprocessConfig(enabled=True)
        with tempfile.TemporaryDirectory() as td:
            result = build_hierarchy(
                edges=df,
                n_levels=1,
                cache_dir=Path(td),
                targets={"nano": 50.0},
                min_sizes={"nano": 2},
                hierarchy_postprocess=cfg,
            )

            assert len(result.levels) == 1
            summary_path = Path(td) / "nano" / "postprocess" / "summary.json"
            moves_path = Path(td) / "nano" / "postprocess" / "oversize_boundary_trim_moves.csv"
            meta_path = Path(td) / "nano" / "meta.json"
            assert summary_path.exists()
            assert moves_path.exists()
            meta = json.loads(meta_path.read_text())
            summary = json.loads(summary_path.read_text())
            assert meta["hierarchy_postprocess_enabled"] is True
            assert meta["oversize_policy"] == "quality_first"
            assert meta["postprocess_backend"] == "python"
            assert meta["postprocess_config_hash"] == postprocess_config_hash(cfg)
            assert meta["target_min_doc_weight"] == 2.0
            assert meta["target_max_doc_weight"] == pytest.approx(30.0)
            assert summary["backend"] == "python"
            assert "small_cluster_summary" in summary
            assert "oversize_summary" in summary
            assert "final_summary" in summary

    def test_postprocess_cache_hash_mismatch_recomputes_level(self):
        import json

        df = _make_simple_graph(60)
        cfg = HierarchyPostprocessConfig(enabled=True)
        with tempfile.TemporaryDirectory() as td:
            build_hierarchy(
                edges=df,
                n_levels=1,
                cache_dir=Path(td),
                targets={"nano": 50.0},
                min_sizes={"nano": 2},
                hierarchy_postprocess=cfg,
            )
            meta_path = Path(td) / "nano" / "meta.json"
            meta = json.loads(meta_path.read_text())
            meta["postprocess_config_hash"] = "stale"
            meta_path.write_text(json.dumps(meta))

            build_hierarchy(
                edges=df,
                n_levels=1,
                cache_dir=Path(td),
                targets={"nano": 50.0},
                min_sizes={"nano": 2},
                hierarchy_postprocess=cfg,
            )
            refreshed = json.loads(meta_path.read_text())
            assert refreshed["postprocess_config_hash"] == postprocess_config_hash(cfg)

    def test_with_layers(self):
        """Test multi-layer combination path."""
        n = 30
        edges_a = []
        edges_b = []
        for i in range(n):
            for j in range(i + 1, n):
                if abs(i - j) < 10:
                    edges_a.append((str(i), str(j), 1.0))
                if (i // 10) == (j // 10):
                    edges_b.append((str(i), str(j), 1.0))
        layer_a = pl.DataFrame({"uid1": [e[0] for e in edges_a],
                                "uid2": [e[1] for e in edges_a],
                                "rel_sum2": [e[2] for e in edges_a]})
        layer_b = pl.DataFrame({"uid1": [e[0] for e in edges_b],
                                "uid2": [e[1] for e in edges_b],
                                "rel_sum2": [e[2] for e in edges_b]})
        result = build_hierarchy(layers={"a": layer_a, "b": layer_b}, n_levels=1,
                                 targets={"nano": 50.0}, min_sizes={"nano": 2})
        assert result.n_nodes > 0
        assert len(result.levels) >= 1


class TestHierarchyPostprocess:

    def test_config_defaults_are_disabled_quality_first(self):
        cfg = HierarchyPostprocessConfig()
        assert cfg.enabled is False
        assert cfg.use_rust_dongdaemun is False
        assert cfg.oversize_policy == "quality_first"
        assert trim_min_delta_q_for_policy(cfg) == 0.0

    def test_target_max_uses_level_target_percentage(self):
        assert hierarchy_target_max_doc_weight(250.0, 20.0) == 50.0

    def test_hard_cap_uses_negative_trim_bound(self):
        cfg = HierarchyPostprocessConfig(enabled=True, oversize_policy="hard_cap")
        assert trim_min_delta_q_for_policy(cfg) == -1.0

    def test_hard_cap_failure_falls_back_to_small_membership(self, tmp_path):
        class FakeGraph:
            def cpm_quality(self, membership, *, resolution):
                return 10.0

            def split_merge_repair_probes(self, *args, **kwargs):
                return {"cluster": np.asarray([], dtype=np.uint64)}

            def trim_oversize_boundary_moves(self, membership, candidate_clusters, **kwargs):
                return {
                    "membership": np.asarray(membership, dtype=np.uint64),
                    "source": np.asarray([], dtype=np.uint64),
                    "target": np.asarray([], dtype=np.uint64),
                    "node": np.asarray([], dtype=np.uint64),
                    "node_weight": np.asarray([], dtype=np.float64),
                    "delta_q": np.asarray([], dtype=np.float64),
                    "source_weight_before": np.asarray([], dtype=np.float64),
                    "source_weight_after": np.asarray([], dtype=np.float64),
                    "target_weight_before": np.asarray([], dtype=np.float64),
                    "target_weight_after": np.asarray([], dtype=np.float64),
                }

        small_membership = np.asarray([0, 0, 0, 1], dtype=np.uint64)
        result = run_hierarchy_level_postprocess(
            FakeGraph(),
            raw_membership=small_membership,
            small_membership=small_membership,
            node_weights=np.ones(4, dtype=np.float64),
            resolution=0.1,
            min_doc_weight=1.0,
            target_max_doc_weight=2.0,
            config=HierarchyPostprocessConfig(enabled=True, oversize_policy="hard_cap"),
            seed=0,
            output_dir=tmp_path,
        )

        assert result.accepted is False
        assert result.status == "hard_cap_not_satisfied"
        assert result.oversize_summary["target_max_satisfied"] is False
        np.testing.assert_array_equal(result.membership, small_membership)
        assert (tmp_path / "summary.json").exists()
        assert (tmp_path / "oversize_boundary_trim_moves.csv").exists()

    def test_rust_dongdaemun_opt_in_keeps_python_backend_for_artifacts(self, tmp_path):
        import json

        class FakeGraph:
            def cpm_quality(self, membership, *, resolution):
                return 0.0

            def dongdaemun_refine(self, *args, **kwargs):
                raise AssertionError("artifact-writing runs must not use Rust fast path")

        membership = np.asarray([0, 1, 2], dtype=np.uint64)
        result = run_hierarchy_level_postprocess(
            FakeGraph(),
            raw_membership=membership,
            small_membership=membership,
            node_weights=np.ones(3, dtype=np.float64),
            resolution=0.1,
            min_doc_weight=1.0,
            target_max_doc_weight=2.0,
            config=HierarchyPostprocessConfig(
                enabled=True,
                use_rust_dongdaemun=True,
            ),
            seed=0,
            output_dir=tmp_path,
        )

        summary = json.loads((tmp_path / "summary.json").read_text())
        assert result.backend == "python"
        assert result.status == "no_current_oversize_candidates"
        assert summary["backend"] == "python"
        assert "rust_audit" not in summary["oversize_summary"]
        assert (tmp_path / "oversize_boundary_trim_moves.csv").exists()

    def test_rust_dongdaemun_opt_in_requires_rust_graph_binding(self):
        class FakeGraph:
            def cpm_quality(self, membership, *, resolution):
                return 0.0

            def dongdaemun_refine(self, *args, **kwargs):
                raise AssertionError("non-Rust graphs must stay on the Python path")

        membership = np.asarray([0, 1, 2], dtype=np.uint64)
        result = run_hierarchy_level_postprocess(
            FakeGraph(),
            raw_membership=membership,
            small_membership=membership,
            node_weights=np.ones(3, dtype=np.float64),
            resolution=0.1,
            min_doc_weight=1.0,
            target_max_doc_weight=2.0,
            config=HierarchyPostprocessConfig(
                enabled=True,
                use_rust_dongdaemun=True,
                write_artifacts=False,
            ),
            seed=0,
            output_dir=None,
        )

        assert result.backend == "python"
        assert result.status == "no_current_oversize_candidates"
        assert "rust_audit" not in result.oversize_summary
        np.testing.assert_array_equal(result.membership, membership)

    def test_rust_dongdaemun_opt_in_uses_fast_path_without_artifacts(self):
        if not RUST_DONGDAEMUN_AVAILABLE:
            pytest.skip("Rust Dongdaemun binding not available")
        graph = build_leiden_graph(
            edges_src=np.asarray([0, 1], dtype=np.uint32),
            edges_dst=np.asarray([1, 2], dtype=np.uint32),
            edges_weight=np.asarray([1.0, 1.0], dtype=np.float64),
            n_nodes=3,
        )

        membership = np.asarray([0, 1, 2], dtype=np.uint64)
        result = run_hierarchy_level_postprocess(
            graph,
            raw_membership=membership,
            small_membership=membership,
            node_weights=np.ones(3, dtype=np.float64),
            resolution=0.1,
            min_doc_weight=1.0,
            target_max_doc_weight=2.0,
            config=HierarchyPostprocessConfig(
                enabled=True,
                use_rust_dongdaemun=True,
                write_artifacts=False,
            ),
            seed=0,
            output_dir=None,
        )

        assert result.backend == "rust_dongdaemun"
        assert result.oversize_summary["backend"] == "rust_dongdaemun"
        assert result.oversize_summary["rust_audit"]["status"] == result.status
        assert result.status == "no_current_oversize_candidates"
        assert result.accepted is True
        np.testing.assert_array_equal(result.membership, membership)
        assert result.paths == {}


# ── Tests: helper functions ──────────────────────────────────

class TestContractEdges:

    def test_basic_contraction(self):
        src = np.array([0, 0, 1, 2], dtype=np.uint32)
        dst = np.array([1, 2, 3, 3], dtype=np.uint32)
        w = np.array([1.0, 2.0, 3.0, 1.0])
        mem = np.array([0, 0, 1, 1], dtype=np.uint64)

        out_src, out_dst, out_w, n_cl, sizes = _contract_edges(src, dst, w, mem, None)
        assert n_cl == 2
        assert len(out_src) == 1  # one inter-cluster edge
        assert sizes[0] == 2 and sizes[1] == 2

    def test_all_same_cluster(self):
        src = np.array([0, 1], dtype=np.uint32)
        dst = np.array([1, 2], dtype=np.uint32)
        w = np.array([1.0, 1.0])
        mem = np.array([0, 0, 0], dtype=np.uint64)

        out_src, out_dst, out_w, n_cl, sizes = _contract_edges(src, dst, w, mem, None)
        assert n_cl == 1
        assert len(out_w) == 0  # no inter-cluster edges

    def test_with_node_sizes(self):
        src = np.array([0, 1], dtype=np.uint32)
        dst = np.array([1, 2], dtype=np.uint32)
        w = np.array([1.0, 1.0])
        mem = np.array([0, 0, 1], dtype=np.uint64)
        prev_sizes = np.array([10, 20, 30], dtype=np.int64)

        _, _, _, n_cl, sizes = _contract_edges(src, dst, w, mem, prev_sizes)
        assert sizes[0] == 30  # 10 + 20
        assert sizes[1] == 30

    def test_empty_edges(self):
        src = np.array([], dtype=np.uint32)
        dst = np.array([], dtype=np.uint32)
        w = np.array([], dtype=np.float64)
        mem = np.array([0, 1], dtype=np.uint64)

        out_src, out_dst, out_w, n_cl, sizes = _contract_edges(src, dst, w, mem, None)
        assert len(out_w) == 0


class TestContractAndNormalize:

    def test_produces_ranked_weights(self):
        src = np.array([0, 0, 1], dtype=np.uint32)
        dst = np.array([1, 2, 2], dtype=np.uint32)
        w = np.array([1.0, 2.0, 3.0])
        mem = np.array([0, 1, 2], dtype=np.uint64)

        out_src, out_dst, out_w, n_cl, sizes = _contract_and_normalize(
            src, dst, w, mem, None)
        # 1/rank normalization: weights should be 1/1, 1/2, 1/3
        assert n_cl == 3
        if len(out_w) > 0:
            assert out_w.max() <= 1.0
            assert out_w.min() > 0

    def test_zero_edges(self):
        src = np.array([0, 1], dtype=np.uint32)
        dst = np.array([1, 0], dtype=np.uint32)
        w = np.array([1.0, 1.0])
        mem = np.array([0, 0], dtype=np.uint64)

        out_src, out_dst, out_w, n_cl, sizes = _contract_and_normalize(
            src, dst, w, mem, None)
        assert len(out_w) == 0


class TestAdaptiveContractedK:

    def test_small_graph(self):
        assert _adaptive_contracted_k(9) == 3

    def test_medium_graph(self):
        assert _adaptive_contracted_k(100) == 10

    def test_large_graph(self):
        assert _adaptive_contracted_k(1000) == 30

    def test_min_floor(self):
        assert _adaptive_contracted_k(1) == 3
