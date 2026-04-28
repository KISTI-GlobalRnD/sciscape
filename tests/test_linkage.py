"""Tests for sciscape.linkage — DC, BC, CC construction from citation data."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from sciscape.linkage import (
    CombineMethod,
    DCNormalization,
    LinkageConfig,
    Normalization,
    WeightNorm,
    build_bc,
    build_cc,
    build_dc,
    combine_edges,
    priority_fill_edges,
    filter_giant_component,
    filter_min_weight,
    filter_top_k,
    normalize_weights,
)


# ── Fixtures ──────────────────────────────────────────────────────

def _triangle_citations() -> tuple[pl.DataFrame, set[str]]:
    """Three papers forming a citation triangle + external references.

    Papers: A, B, C  (focal set)
    External: X, Y, Z (out-of-field references)

    Citations (→ = cites):
        A → B, A → X, A → Y
        B → C, B → X, B → Z
        C → A, C → Y, C → Z

    DC (in-set, undirected): A-B (A→B + C→A?), A-C (C→A), B-C (B→C)
    BC shared refs:
        A∩B = {X}         → 1 shared
        A∩C = {Y}         → 1 shared
        B∩C = {Z}         → 1 shared
    CC shared citers (who cites A,B,C from the full set):
        Citers of A: C (C→A)
        Citers of B: A (A→B)
        Citers of C: B (B→C)
        A∩B shared citers: none
        A∩C shared citers: none
        B∩C shared citers: none
    """
    citations = pl.DataFrame({
        "citing_work_id": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
        "cited_work_id":  ["B", "X", "Y", "C", "X", "Z", "A", "Y", "Z"],
        "cited_in_set":   [  1,   0,   0,   1,   0,   0,   1,   0,   0],
    })
    node_ids = {"A", "B", "C"}
    return citations, node_ids


def _diamond_citations() -> tuple[pl.DataFrame, set[str]]:
    """Four papers where A,B both cite C,D → strong BC coupling.

    Papers: A, B, C, D  (all focal)

    Citations:
        A → C, A → D
        B → C, B → D
        C → D

    BC: A∩B share refs {C, D} → count=2
    CC: C∩D share citers {A, B} → count=2
    """
    citations = pl.DataFrame({
        "citing_work_id": ["A", "A", "B", "B", "C"],
        "cited_work_id":  ["C", "D", "C", "D", "D"],
        "cited_in_set":   [  1,   1,   1,   1,   1],
    })
    node_ids = {"A", "B", "C", "D"}
    return citations, node_ids


# ── DC Tests ──────────────────────────────────────────────────────

class TestBuildDC:
    def test_binary_weights(self):
        cit, nodes = _triangle_citations()
        result = build_dc(cit, nodes, norms=[DCNormalization.BINARY])
        assert "dc_binary" in result
        df = result["dc_binary"]
        assert set(df.columns) == {"uid1", "uid2", "rel_sum2"}
        # A→B, B→C, C→A → 3 undirected edges
        assert df.height == 3
        # Binary: each edge weight = 1.0
        assert all(w == 1.0 for w in df["rel_sum2"].to_list())

    def test_fractional_weights(self):
        cit, nodes = _triangle_citations()
        result = build_dc(cit, nodes, norms=[DCNormalization.FRACTIONAL])
        df = result["dc_fractional"]
        assert df.height == 3
        # A has 3 total refs → weight = 1/3 for A→B
        # B has 3 total refs → weight = 1/3 for B→C
        # C has 3 total refs → weight = 1/3 for C→A
        weights = sorted(df["rel_sum2"].to_list())
        for w in weights:
            assert w == pytest.approx(1.0 / 3.0, abs=1e-6)

    def test_both_norms(self):
        cit, nodes = _triangle_citations()
        result = build_dc(cit, nodes)
        assert "dc_binary" in result
        assert "dc_fractional" in result

    def test_empty_citations(self):
        """No in-set citations → empty DataFrames."""
        cit = pl.DataFrame({
            "citing_work_id": ["A"],
            "cited_work_id": ["X"],
            "cited_in_set": [0],
        })
        result = build_dc(cit, {"A", "X"}, norms=[DCNormalization.BINARY])
        assert result["dc_binary"].height == 0

    def test_custom_column_names(self):
        cit = pl.DataFrame({
            "src": ["A", "B"],
            "dst": ["B", "A"],
        })
        cfg = LinkageConfig(
            citing_col="src", cited_col="dst", cited_in_set_col=None,
        )
        result = build_dc(cit, {"A", "B"}, config=cfg, norms=[DCNormalization.BINARY])
        df = result["dc_binary"]
        assert df.height == 1  # A-B undirected
        assert df["rel_sum2"][0] == 2.0  # A→B + B→A


# ── BC Tests ──────────────────────────────────────────────────────

class TestBuildBC:
    def test_diamond_raw(self):
        """A and B share 2 references (C, D) → bc_raw weight = 2."""
        cit, nodes = _diamond_citations()
        cfg = LinkageConfig(bc_min_shared=1, cc_min_shared=1)
        result = build_bc(cit, nodes, config=cfg, norms=[Normalization.RAW])
        df = result["bc_raw"]

        # Find A-B edge
        ab = df.filter(
            ((pl.col("uid1") == "A") & (pl.col("uid2") == "B"))
            | ((pl.col("uid1") == "B") & (pl.col("uid2") == "A"))
        )
        assert ab.height == 1
        assert ab["rel_sum2"][0] == 2.0

    def test_diamond_cosine(self):
        """A∩B=2, deg(A)=2, deg(B)=2 → cosine = 2/sqrt(2*2) = 1.0."""
        cit, nodes = _diamond_citations()
        cfg = LinkageConfig(bc_min_shared=1, cc_min_shared=1)
        result = build_bc(cit, nodes, config=cfg, norms=[Normalization.COSINE])
        df = result["bc_cosine"]

        ab = df.filter(
            ((pl.col("uid1") == "A") & (pl.col("uid2") == "B"))
            | ((pl.col("uid1") == "B") & (pl.col("uid2") == "A"))
        )
        assert ab["rel_sum2"][0] == pytest.approx(1.0)

    def test_diamond_assoc_strength(self):
        """A∩B=2, deg(A)=2, deg(B)=2 → assoc = 2/(2*2) = 0.5."""
        cit, nodes = _diamond_citations()
        cfg = LinkageConfig(bc_min_shared=1, cc_min_shared=1)
        result = build_bc(cit, nodes, config=cfg, norms=[Normalization.ASSOC_STRENGTH])
        df = result["bc_assoc_strength"]

        ab = df.filter(
            ((pl.col("uid1") == "A") & (pl.col("uid2") == "B"))
            | ((pl.col("uid1") == "B") & (pl.col("uid2") == "A"))
        )
        assert ab["rel_sum2"][0] == pytest.approx(0.5)

    def test_min_shared_filter(self):
        """bc_min_shared=2 should filter edges with only 1 shared ref."""
        cit, nodes = _triangle_citations()
        cfg = LinkageConfig(bc_min_shared=2, cc_min_shared=2)
        result = build_bc(cit, nodes, config=cfg, norms=[Normalization.RAW])
        # triangle: each pair shares exactly 1 ref → all filtered out
        assert result["bc_raw"].height == 0

    def test_out_of_field_refs_included(self):
        """BC uses ALL refs (including out-of-field X, Y, Z)."""
        cit, nodes = _triangle_citations()
        cfg = LinkageConfig(bc_min_shared=1, cc_min_shared=1)
        result = build_bc(cit, nodes, config=cfg, norms=[Normalization.RAW])
        # A refs: {B, X, Y}, B refs: {C, X, Z}, C refs: {A, Y, Z}
        # A∩B={X}, A∩C={Y}, B∩C={Z} → 3 edges, all weight 1
        df = result["bc_raw"]
        assert df.height == 3


# ── CC Tests ──────────────────────────────────────────────────────

class TestBuildCC:
    def test_diamond_raw(self):
        """C and D are both cited by A and B → cc_raw = 2."""
        cit, nodes = _diamond_citations()
        cfg = LinkageConfig(bc_min_shared=1, cc_min_shared=1)
        result = build_cc(cit, nodes, config=cfg, norms=[Normalization.RAW])
        df = result["cc_raw"]

        cd = df.filter(
            ((pl.col("uid1") == "C") & (pl.col("uid2") == "D"))
            | ((pl.col("uid1") == "D") & (pl.col("uid2") == "C"))
        )
        assert cd.height == 1
        assert cd["rel_sum2"][0] == 2.0

    def test_diamond_cosine(self):
        """C∩D=2, citers(C)=2={A,B}, citers(D)=3={A,B,C} → 2/sqrt(2*3)."""
        cit, nodes = _diamond_citations()
        cfg = LinkageConfig(bc_min_shared=1, cc_min_shared=1)
        result = build_cc(cit, nodes, config=cfg, norms=[Normalization.COSINE])
        df = result["cc_cosine"]

        cd = df.filter(
            ((pl.col("uid1") == "C") & (pl.col("uid2") == "D"))
            | ((pl.col("uid1") == "D") & (pl.col("uid2") == "C"))
        )
        expected = 2.0 / np.sqrt(2 * 3)
        assert cd["rel_sum2"][0] == pytest.approx(expected)

    def test_min_shared_filter(self):
        """cc_min_shared=3 filters out edges with < 3 shared citers."""
        cit, nodes = _diamond_citations()
        cfg = LinkageConfig(bc_min_shared=3, cc_min_shared=3)
        result = build_cc(cit, nodes, config=cfg, norms=[Normalization.RAW])
        assert result["cc_raw"].height == 0

    def test_external_citers_counted(self):
        """Citers outside the focal set should be counted in CC."""
        # E (non-focal) cites A and B → A∩B share citer E
        citations = pl.DataFrame({
            "citing_work_id": ["E", "E"],
            "cited_work_id":  ["A", "B"],
            "cited_in_set":   [  1,   1],
        })
        cfg = LinkageConfig(bc_min_shared=1, cc_min_shared=1)
        result = build_cc(citations, {"A", "B"}, config=cfg, norms=[Normalization.RAW])
        df = result["cc_raw"]
        assert df.height == 1
        assert df["rel_sum2"][0] == 1.0


# ── Combination Tests ─────────────────────────────────────────────

class TestCombineEdges:
    def _two_edge_sets(self) -> dict[str, pl.DataFrame]:
        """Two overlapping edge sets."""
        bc = pl.DataFrame({
            "uid1": ["A", "A"],
            "uid2": ["B", "C"],
            "rel_sum2": [0.8, 0.4],
        })
        cc = pl.DataFrame({
            "uid1": ["A", "B"],
            "uid2": ["B", "C"],
            "rel_sum2": [0.6, 0.3],
        })
        return {"bc": bc, "cc": cc}

    def test_sum(self):
        sets = self._two_edge_sets()
        combined = combine_edges(sets, CombineMethod.SUM)
        assert "uid1" in combined.columns
        assert "uid2" in combined.columns
        assert "rel_sum2" in combined.columns
        # A-B exists in both → should have combined weight
        ab = combined.filter(
            ((pl.col("uid1") == "A") & (pl.col("uid2") == "B"))
            | ((pl.col("uid1") == "B") & (pl.col("uid2") == "A"))
        )
        assert ab.height == 1
        # bc weight: 0.8/0.8=1.0, cc weight: 0.6/0.6=1.0 → sum=2.0
        assert ab["rel_sum2"][0] == pytest.approx(2.0)

    def test_max(self):
        sets = self._two_edge_sets()
        combined = combine_edges(sets, CombineMethod.MAX)
        ab = combined.filter(
            ((pl.col("uid1") == "A") & (pl.col("uid2") == "B"))
            | ((pl.col("uid1") == "B") & (pl.col("uid2") == "A"))
        )
        # Both normalized to 1.0 → max=1.0
        assert ab["rel_sum2"][0] == pytest.approx(1.0)

    def test_noisy_or(self):
        sets = self._two_edge_sets()
        combined = combine_edges(sets, CombineMethod.NOISY_OR)
        ab = combined.filter(
            ((pl.col("uid1") == "A") & (pl.col("uid2") == "B"))
            | ((pl.col("uid1") == "B") & (pl.col("uid2") == "A"))
        )
        # Both normalized to 1.0 → noisy_or = 1 - (1-1)(1-1) = 1.0
        assert ab["rel_sum2"][0] == pytest.approx(1.0)

    def test_single_set_passthrough(self):
        """Single edge set → returned as-is."""
        df = pl.DataFrame({
            "uid1": ["A"], "uid2": ["B"], "rel_sum2": [0.5],
        })
        result = combine_edges({"only": df})
        assert result.height == 1

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            combine_edges({})

    def test_disjoint_edges_sum(self):
        """Two edge sets with no overlap → union of edges."""
        s1 = pl.DataFrame({"uid1": ["A"], "uid2": ["B"], "rel_sum2": [1.0]})
        s2 = pl.DataFrame({"uid1": ["C"], "uid2": ["D"], "rel_sum2": [1.0]})
        combined = combine_edges({"s1": s1, "s2": s2}, CombineMethod.SUM)
        assert combined.height == 2


# ── Config Tests ──────────────────────────────────────────────────

class TestLinkageConfig:
    def test_defaults(self):
        cfg = LinkageConfig()
        assert cfg.bc_min_shared == 3
        assert cfg.cc_min_shared == 2
        assert cfg.citing_col == "citing_work_id"
        assert cfg.cited_col == "cited_work_id"

    def test_custom(self):
        cfg = LinkageConfig(bc_min_shared=5, cc_min_shared=3, citing_col="src", cited_col="dst")
        assert cfg.bc_min_shared == 5
        assert cfg.citing_col == "src"


# ── Integration: DC → BC/CC pipeline ─────────────────────────────

class TestIntegration:
    def test_full_pipeline(self):
        """Build DC + BC + CC + combine into a single edge set."""
        cit, nodes = _diamond_citations()
        cfg = LinkageConfig(bc_min_shared=1, cc_min_shared=1)

        dc = build_dc(cit, nodes, config=cfg, norms=[DCNormalization.FRACTIONAL])
        bc = build_bc(cit, nodes, config=cfg, norms=[Normalization.COSINE])
        cc = build_cc(cit, nodes, config=cfg, norms=[Normalization.COSINE])

        # All should produce DataFrames
        assert dc["dc_fractional"].height > 0
        assert bc["bc_cosine"].height > 0
        assert cc["cc_cosine"].height > 0

        # Combine BC + CC
        combined = combine_edges(
            {"bc": bc["bc_cosine"], "cc": cc["cc_cosine"]},
            CombineMethod.SUM,
        )
        assert combined.height > 0
        assert set(combined.columns) == {"uid1", "uid2", "rel_sum2"}

    def test_output_compatible_with_build_graph(self):
        """Combined edges should be directly usable by build_graph()."""
        from sciscape.clustering.graph import build_graph

        cit, nodes = _diamond_citations()
        cfg = LinkageConfig(bc_min_shared=1, cc_min_shared=1)
        bc = build_bc(cit, nodes, config=cfg, norms=[Normalization.COSINE])

        graph = build_graph(bc["bc_cosine"])
        # D has no outgoing citations → no BC edges → 3 nodes in graph
        assert graph.vcount() >= 3
        assert graph.ecount() > 0
        assert "weight" in graph.es.attributes()


# ── Filter Tests ──────────────────────────────────────────────────

def _sample_edges() -> pl.DataFrame:
    """6-edge star: A connected to B,C,D,E,F with varying weights."""
    return pl.DataFrame({
        "uid1": ["A", "A", "A", "A", "A", "B"],
        "uid2": ["B", "C", "D", "E", "F", "C"],
        "rel_sum2": [1.0, 0.8, 0.5, 0.3, 0.1, 0.6],
    })


class TestFilterMinWeight:
    def test_basic(self):
        df = _sample_edges()
        result = filter_min_weight(df, 0.5)
        assert result.height == 4  # 1.0, 0.8, 0.5, 0.6

    def test_filter_all(self):
        df = _sample_edges()
        result = filter_min_weight(df, 2.0)
        assert result.height == 0

    def test_filter_none(self):
        df = _sample_edges()
        result = filter_min_weight(df, 0.0)
        assert result.height == 6


class TestFilterTopK:
    def test_k2_symmetric(self):
        """Keep top-2 per node on a dense graph (all degree > 2)."""
        # Complete K5: every node has degree 4 → k=2 should reduce edges
        df = pl.DataFrame({
            "uid1": ["A", "A", "A", "A", "B", "B", "B", "C", "C", "D"],
            "uid2": ["B", "C", "D", "E", "C", "D", "E", "D", "E", "E"],
            "rel_sum2": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        })
        result = filter_top_k(df, k=2)
        # 10 edges → should drop some (every node keeps top-2)
        assert result.height < 10
        assert result.height >= 5  # at least 5 nodes × 1 edge

    def test_k1(self):
        """k=1: each node keeps only its strongest edge."""
        # K5 graph
        df = pl.DataFrame({
            "uid1": ["A", "A", "A", "A", "B", "B", "B", "C", "C", "D"],
            "uid2": ["B", "C", "D", "E", "C", "D", "E", "D", "E", "E"],
            "rel_sum2": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        })
        result = filter_top_k(df, k=1)
        assert result.height >= 1
        assert result.height < 10

    def test_k_large(self):
        """k >= max_degree → no filtering."""
        df = _sample_edges()
        result = filter_top_k(df, k=100)
        assert result.height == 6

    def test_invalid_k(self):
        df = _sample_edges()
        with pytest.raises(ValueError, match="k must be >= 1"):
            filter_top_k(df, k=0)

    def test_mutual_mode(self):
        """Mutual mode: keep only if both endpoints agree."""
        df = _sample_edges()
        mutual = filter_top_k(df, k=2, mode="mutual")
        symmetric = filter_top_k(df, k=2, mode="symmetric")
        # mutual ⊆ symmetric
        assert mutual.height <= symmetric.height

    def test_empty(self):
        empty = pl.DataFrame({"uid1": [], "uid2": [], "rel_sum2": []}).cast(
            {"uid1": pl.Utf8, "uid2": pl.Utf8, "rel_sum2": pl.Float64}
        )
        result = filter_top_k(empty, k=5)
        assert result.height == 0


class TestFilterGiantComponent:
    def test_keeps_connected(self):
        """All edges in one component → nothing filtered."""
        df = _sample_edges()
        result = filter_giant_component(df)
        assert result.height == 6

    def test_removes_isolated_component(self):
        """Add a disconnected pair → should be filtered if smaller."""
        df = pl.concat([
            _sample_edges(),
            pl.DataFrame({"uid1": ["X"], "uid2": ["Y"], "rel_sum2": [0.5]}),
        ])
        result = filter_giant_component(df)
        # Giant component is A-B-C-D-E-F (6 edges), X-Y is smaller
        assert result.height == 6


# ── Normalization Tests ───────────────────────────────────────────

class TestNormalizeWeights:
    def test_max(self):
        df = _sample_edges()
        result = normalize_weights(df, WeightNorm.MAX)
        assert result["rel_sum2"].max() == pytest.approx(1.0)
        assert result["rel_sum2"].min() == pytest.approx(0.1)  # 0.1/1.0

    def test_minmax(self):
        df = _sample_edges()
        result = normalize_weights(df, WeightNorm.MINMAX)
        assert result["rel_sum2"].max() == pytest.approx(1.0)
        assert result["rel_sum2"].min() == pytest.approx(0.0)

    def test_rank(self):
        df = _sample_edges()
        result = normalize_weights(df, WeightNorm.RANK)
        # Ranks: 0.1→1, 0.3→2, 0.5→3, 0.6→4, 0.8→5, 1.0→6 → /6
        assert result["rel_sum2"].max() == pytest.approx(1.0)
        assert result["rel_sum2"].min() == pytest.approx(1.0 / 6)

    def test_zscore(self):
        df = _sample_edges()
        result = normalize_weights(df, WeightNorm.ZSCORE)
        # Z-score: mean ≈ 0
        assert result["rel_sum2"].mean() == pytest.approx(0.0, abs=1e-10)

    def test_quantile(self):
        df = _sample_edges()
        result = normalize_weights(df, WeightNorm.QUANTILE)
        assert result["rel_sum2"].max() == pytest.approx(1.0)

    def test_empty(self):
        empty = pl.DataFrame({"uid1": [], "uid2": [], "rel_sum2": []}).cast(
            {"uid1": pl.Utf8, "uid2": pl.Utf8, "rel_sum2": pl.Float64}
        )
        result = normalize_weights(empty, WeightNorm.MAX)
        assert result.height == 0


# ── Extended Combination Tests ────────────────────────────────────

class TestCombineExtended:
    def _overlapping_sets(self) -> dict[str, pl.DataFrame]:
        """bc has A-B and A-C; cc has A-B and B-C."""
        bc = pl.DataFrame({
            "uid1": ["A", "A"],
            "uid2": ["B", "C"],
            "rel_sum2": [0.8, 0.4],
        })
        cc = pl.DataFrame({
            "uid1": ["A", "B"],
            "uid2": ["B", "C"],
            "rel_sum2": [0.6, 0.3],
        })
        return {"bc": bc, "cc": cc}

    def test_weighted_sum(self):
        sets = self._overlapping_sets()
        combined = combine_edges(
            sets, CombineMethod.WEIGHTED_SUM,
            weights={"bc": 0.7, "cc": 0.3},
        )
        ab = combined.filter(
            ((pl.col("uid1") == "A") & (pl.col("uid2") == "B"))
            | ((pl.col("uid1") == "B") & (pl.col("uid2") == "A"))
        )
        assert ab.height == 1
        # bc: 0.8/0.8=1.0 * 0.7 = 0.7; cc: 0.6/0.6=1.0 * 0.3 = 0.3 → 1.0
        assert ab["rel_sum2"][0] == pytest.approx(1.0)

    def test_weighted_sum_missing_weights_raises(self):
        sets = self._overlapping_sets()
        with pytest.raises(ValueError, match="requires"):
            combine_edges(sets, CombineMethod.WEIGHTED_SUM)

    def test_weighted_sum_incomplete_weights_raises(self):
        sets = self._overlapping_sets()
        with pytest.raises(ValueError, match="missing keys"):
            combine_edges(sets, CombineMethod.WEIGHTED_SUM, weights={"bc": 1.0})

    def test_min(self):
        """MIN: only edges in both layers survive (with smaller weight)."""
        sets = self._overlapping_sets()
        combined = combine_edges(sets, CombineMethod.MIN)
        # A-B in both → min(1.0, 1.0) = 1.0 (both normalized to 1.0)
        # A-C in bc only, B-C in cc only → min with 0 = 0 → filtered
        ab = combined.filter(
            ((pl.col("uid1") == "A") & (pl.col("uid2") == "B"))
            | ((pl.col("uid1") == "B") & (pl.col("uid2") == "A"))
        )
        assert ab.height == 1

    def test_product(self):
        """PRODUCT: edge absent in one layer → 0."""
        sets = self._overlapping_sets()
        combined = combine_edges(sets, CombineMethod.PRODUCT)
        # A-B: 1.0 * 1.0 = 1.0
        # A-C: 0.5 * 0 = 0 (absent in cc) → filtered
        ab = combined.filter(
            ((pl.col("uid1") == "A") & (pl.col("uid2") == "B"))
            | ((pl.col("uid1") == "B") & (pl.col("uid2") == "A"))
        )
        assert ab.height == 1
        assert ab["rel_sum2"][0] == pytest.approx(1.0)

    def test_geometric_mean(self):
        """GEOMETRIC_MEAN: sqrt(product)."""
        bc = pl.DataFrame({"uid1": ["A"], "uid2": ["B"], "rel_sum2": [0.9]})
        cc = pl.DataFrame({"uid1": ["A"], "uid2": ["B"], "rel_sum2": [0.4]})
        combined = combine_edges(
            {"bc": bc, "cc": cc}, CombineMethod.GEOMETRIC_MEAN,
        )
        # Both normalized to 1.0 → gmean(1.0, 1.0) = 1.0
        assert combined["rel_sum2"][0] == pytest.approx(1.0)

    def test_geometric_mean_different_weights(self):
        """GEOMETRIC_MEAN with pre_normalize=False."""
        bc = pl.DataFrame({"uid1": ["A"], "uid2": ["B"], "rel_sum2": [0.9]})
        cc = pl.DataFrame({"uid1": ["A"], "uid2": ["B"], "rel_sum2": [0.4]})
        combined = combine_edges(
            {"bc": bc, "cc": cc}, CombineMethod.GEOMETRIC_MEAN,
            pre_normalize=False,
        )
        expected = np.sqrt(0.9 * 0.4)
        assert combined["rel_sum2"][0] == pytest.approx(expected)

    def test_harmonic_mean(self):
        """HARMONIC_MEAN: only edges in ALL layers."""
        bc = pl.DataFrame({"uid1": ["A"], "uid2": ["B"], "rel_sum2": [0.8]})
        cc = pl.DataFrame({"uid1": ["A"], "uid2": ["B"], "rel_sum2": [0.4]})
        combined = combine_edges(
            {"bc": bc, "cc": cc}, CombineMethod.HARMONIC_MEAN,
            pre_normalize=False,
        )
        expected = 2.0 / (1.0 / 0.8 + 1.0 / 0.4)
        assert combined["rel_sum2"][0] == pytest.approx(expected)

    def test_harmonic_mean_disjoint(self):
        """HARMONIC_MEAN: disjoint sets → no edges."""
        s1 = pl.DataFrame({"uid1": ["A"], "uid2": ["B"], "rel_sum2": [1.0]})
        s2 = pl.DataFrame({"uid1": ["C"], "uid2": ["D"], "rel_sum2": [1.0]})
        combined = combine_edges(
            {"s1": s1, "s2": s2}, CombineMethod.HARMONIC_MEAN,
        )
        assert combined.height == 0

    def test_consensus(self):
        """CONSENSUS: (avg_weight) × consensus_count."""
        sets = self._overlapping_sets()
        combined = combine_edges(
            sets, CombineMethod.CONSENSUS, pre_normalize=False,
        )
        # A-B in both layers: avg=(0.8+0.6)/2=0.7, cons=2 → 1.4
        ab = combined.filter(
            ((pl.col("uid1") == "A") & (pl.col("uid2") == "B"))
            | ((pl.col("uid1") == "B") & (pl.col("uid2") == "A"))
        )
        assert ab.height == 1
        assert ab["rel_sum2"][0] == pytest.approx(0.7 * 2)

        # A-C in bc only: avg=0.4/2=0.2, cons=1 → 0.2
        ac = combined.filter(
            ((pl.col("uid1") == "A") & (pl.col("uid2") == "C"))
            | ((pl.col("uid1") == "C") & (pl.col("uid2") == "A"))
        )
        assert ac.height == 1
        assert ac["rel_sum2"][0] == pytest.approx(0.4 / 2 * 1)

    def test_consensus_normalized(self):
        """CONSENSUS with pre_normalize=True (default)."""
        sets = self._overlapping_sets()
        combined = combine_edges(sets, CombineMethod.CONSENSUS)
        # After normalization: bc A-B=1.0, A-C=0.5; cc A-B=1.0, B-C=1.0
        # A-B: avg=(1.0+1.0)/2=1.0, cons=2 → 2.0
        ab = combined.filter(
            ((pl.col("uid1") == "A") & (pl.col("uid2") == "B"))
            | ((pl.col("uid1") == "B") & (pl.col("uid2") == "A"))
        )
        assert ab.height == 1
        assert ab["rel_sum2"][0] == pytest.approx(2.0)

    def test_consensus_three_layers(self):
        """CONSENSUS with 3 layers: edge in all 3 gets 3× bonus."""
        s1 = pl.DataFrame({"uid1": ["A", "A"], "uid2": ["B", "C"], "rel_sum2": [0.9, 0.3]})
        s2 = pl.DataFrame({"uid1": ["A"], "uid2": ["B"], "rel_sum2": [0.6]})
        s3 = pl.DataFrame({"uid1": ["A"], "uid2": ["B"], "rel_sum2": [0.3]})
        combined = combine_edges(
            {"s1": s1, "s2": s2, "s3": s3},
            CombineMethod.CONSENSUS, pre_normalize=False,
        )
        # A-B in all 3: avg=(0.9+0.6+0.3)/3=0.6, cons=3 → 1.8
        ab = combined.filter(
            ((pl.col("uid1") == "A") & (pl.col("uid2") == "B"))
            | ((pl.col("uid1") == "B") & (pl.col("uid2") == "A"))
        )
        assert ab.height == 1
        assert ab["rel_sum2"][0] == pytest.approx(1.8)

    def test_pre_normalize_false(self):
        """pre_normalize=False: raw weights used directly."""
        bc = pl.DataFrame({"uid1": ["A"], "uid2": ["B"], "rel_sum2": [0.5]})
        cc = pl.DataFrame({"uid1": ["A"], "uid2": ["B"], "rel_sum2": [0.3]})
        combined = combine_edges(
            {"bc": bc, "cc": cc}, CombineMethod.SUM, pre_normalize=False,
        )
        assert combined["rel_sum2"][0] == pytest.approx(0.8)


# ── Priority Fill ────────────────────────────────────────────────

class TestPriorityFill:
    def _two_layer_sets(self) -> dict[str, pl.DataFrame]:
        """bc has A-B, A-C, A-D; cc has A-B, B-C, B-D."""
        bc = pl.DataFrame({
            "uid1": ["A", "A", "A"],
            "uid2": ["B", "C", "D"],
            "rel_sum2": [0.9, 0.6, 0.3],
        })
        cc = pl.DataFrame({
            "uid1": ["A", "B", "B"],
            "uid2": ["B", "C", "D"],
            "rel_sum2": [0.8, 0.5, 0.4],
        })
        return {"bc": bc, "cc": cc}

    def test_consensus_edges_prioritized(self):
        """Consensus edges (A-B in both layers) get highest weight."""
        sets = self._two_layer_sets()
        combined = priority_fill_edges(sets, k=5)
        ab = combined.filter(
            ((pl.col("uid1") == "A") & (pl.col("uid2") == "B"))
            | ((pl.col("uid1") == "B") & (pl.col("uid2") == "A"))
        )
        assert ab.height == 1
        assert ab["rel_sum2"][0] == 2.0  # consensus count = 2

    def test_single_layer_edges_included(self):
        """Single-layer edges fill remaining slots."""
        sets = self._two_layer_sets()
        combined = priority_fill_edges(sets, k=5)
        # A-D is in bc only → weight 1.0
        ad = combined.filter(
            ((pl.col("uid1") == "A") & (pl.col("uid2") == "D"))
            | ((pl.col("uid1") == "D") & (pl.col("uid2") == "A"))
        )
        assert ad.height == 1
        assert ad["rel_sum2"][0] == 1.0

    def test_k_limits_neighbors(self):
        """With small k, only the best edges survive per node."""
        sets = self._two_layer_sets()
        combined = priority_fill_edges(sets, k=1)
        # Each node picks only 1 neighbor — consensus A-B should dominate
        ab = combined.filter(
            ((pl.col("uid1") == "A") & (pl.col("uid2") == "B"))
            | ((pl.col("uid1") == "B") & (pl.col("uid2") == "A"))
        )
        assert ab.height == 1

    def test_layer_priority_tiebreak(self):
        """layer_priority determines tie-breaking order."""
        # Two disjoint layers, same node A
        s1 = pl.DataFrame({"uid1": ["A"], "uid2": ["X"], "rel_sum2": [1.0]})
        s2 = pl.DataFrame({"uid1": ["A"], "uid2": ["Y"], "rel_sum2": [1.0]})
        combined = priority_fill_edges(
            {"bc": s1, "emb": s2}, k=1,
            layer_priority={"bc": 0, "emb": 1},  # bc preferred
        )
        # A should pick X (from bc, priority 0) over Y (from emb, priority 1)
        ax = combined.filter(
            ((pl.col("uid1") == "A") & (pl.col("uid2") == "X"))
            | ((pl.col("uid1") == "X") & (pl.col("uid2") == "A"))
        )
        assert ax.height == 1

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            priority_fill_edges({}, k=5)

    def test_three_layers(self):
        """Three layers: edge in all 3 gets consensus_count=3."""
        s1 = pl.DataFrame({"uid1": ["A"], "uid2": ["B"], "rel_sum2": [0.9]})
        s2 = pl.DataFrame({"uid1": ["A"], "uid2": ["B"], "rel_sum2": [0.6]})
        s3 = pl.DataFrame({"uid1": ["A"], "uid2": ["B"], "rel_sum2": [0.3]})
        combined = priority_fill_edges({"s1": s1, "s2": s2, "s3": s3}, k=5)
        assert combined["rel_sum2"][0] == 3.0


# ── Full Pipeline: build → filter → normalize → combine ──────────

class TestFullPipeline:
    def test_pipeline(self):
        """End-to-end: citations → BC/CC → filter → normalize → combine → graph."""
        from sciscape.clustering.graph import build_graph

        cit, nodes = _diamond_citations()
        cfg = LinkageConfig(bc_min_shared=1, cc_min_shared=1)

        bc = build_bc(cit, nodes, config=cfg, norms=[Normalization.COSINE])["bc_cosine"]
        cc = build_cc(cit, nodes, config=cfg, norms=[Normalization.COSINE])["cc_cosine"]

        # Filter
        bc = filter_min_weight(bc, 0.01)
        cc = filter_min_weight(cc, 0.01)

        # Normalize
        bc = normalize_weights(bc, WeightNorm.MAX)
        cc = normalize_weights(cc, WeightNorm.MAX)

        # Combine
        combined = combine_edges(
            {"bc": bc, "cc": cc},
            CombineMethod.SUM,
        )
        assert combined.height > 0

        # Build graph
        graph = build_graph(combined)
        assert graph.vcount() >= 2
        assert graph.ecount() > 0
