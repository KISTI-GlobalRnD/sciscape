"""Tests for keyword extraction diagnostics module."""


import pandas as pd
import pytest

from sciscape.keyword_extraction.diagnostics import (
    KeywordDiagnostics,
    _clamp,
    _quantiles,
    _safe_year_series,
    _sample_cluster_ids,
    _subphrase_redundancy_ratio,
    _token_jaccard_redundancy_ratio,
    keyword_diagnostics,
    score_before_after,
)


# ---------------------------------------------------------------------------
# _clamp
# ---------------------------------------------------------------------------

class TestClamp:
    def test_within_range(self):
        assert _clamp(0.5, 0.0, 1.0) == 0.5

    def test_below_lo(self):
        assert _clamp(-1.0, 0.0, 1.0) == 0.0

    def test_above_hi(self):
        assert _clamp(2.0, 0.0, 1.0) == 1.0

    def test_at_boundaries(self):
        assert _clamp(0.0, 0.0, 1.0) == 0.0
        assert _clamp(1.0, 0.0, 1.0) == 1.0


# ---------------------------------------------------------------------------
# _quantiles
# ---------------------------------------------------------------------------

class TestQuantiles:
    def test_basic(self):
        s = pd.Series([1, 2, 3, 4, 5])
        result = _quantiles(s, qs=(0.5,))
        assert "p50" in result
        assert result["p50"] == pytest.approx(3.0)

    def test_multiple_quantiles(self):
        s = pd.Series(range(100))
        result = _quantiles(s, qs=(0.1, 0.5, 0.9))
        assert "p10" in result
        assert "p50" in result
        assert "p90" in result
        assert result["p10"] < result["p50"] < result["p90"]

    def test_empty_series(self):
        assert _quantiles(pd.Series([], dtype=float), qs=(0.5,)) == {}

    def test_na_values_ignored(self):
        s = pd.Series([1.0, None, 3.0, None, 5.0])
        result = _quantiles(s, qs=(0.5,))
        assert "p50" in result
        assert result["p50"] == pytest.approx(3.0)

    def test_all_na(self):
        s = pd.Series([None, None, None])
        assert _quantiles(s, qs=(0.5,)) == {}


# ---------------------------------------------------------------------------
# _safe_year_series
# ---------------------------------------------------------------------------

class TestSafeYearSeries:
    def test_dict_input(self):
        assert _safe_year_series({2020: 3, 2021: 5}) == {2020: 3, 2021: 5}

    def test_json_string(self):
        assert _safe_year_series('{"2020": 3, "2021": 5}') == {2020: 3, 2021: 5}

    def test_none(self):
        assert _safe_year_series(None) == {}

    def test_nan(self):
        assert _safe_year_series(float("nan")) == {}

    def test_empty_string(self):
        assert _safe_year_series("") == {}
        assert _safe_year_series("   ") == {}

    def test_invalid_json(self):
        assert _safe_year_series("{bad json}") == {}

    def test_zero_counts_excluded(self):
        assert _safe_year_series({2020: 0, 2021: 5}) == {2021: 5}

    def test_non_numeric_keys_skipped(self):
        assert _safe_year_series({"abc": 3, "2021": 5}) == {2021: 5}

    def test_non_dict_type(self):
        assert _safe_year_series(42) == {}
        assert _safe_year_series([1, 2, 3]) == {}


# ---------------------------------------------------------------------------
# _sample_cluster_ids
# ---------------------------------------------------------------------------

class TestSampleClusterIds:
    def test_no_sampling(self):
        ids = [0, 1, 2, 3]
        result = _sample_cluster_ids(ids, sample_clusters=None, seed=0)
        assert result == [0, 1, 2, 3]

    def test_sample_fewer(self):
        ids = list(range(10))
        result = _sample_cluster_ids(ids, sample_clusters=3, seed=42)
        assert len(result) == 3
        assert all(c in ids for c in result)
        assert result == sorted(result)  # should be sorted

    def test_sample_more_than_available(self):
        ids = [0, 1, 2]
        result = _sample_cluster_ids(ids, sample_clusters=10, seed=0)
        assert result == [0, 1, 2]

    def test_duplicates_in_input(self):
        ids = [0, 0, 1, 1, 2, 2]
        result = _sample_cluster_ids(ids, sample_clusters=None, seed=0)
        assert result == [0, 1, 2]

    def test_deterministic_seed(self):
        ids = list(range(20))
        r1 = _sample_cluster_ids(ids, sample_clusters=5, seed=42)
        r2 = _sample_cluster_ids(ids, sample_clusters=5, seed=42)
        assert r1 == r2


# ---------------------------------------------------------------------------
# _subphrase_redundancy_ratio
# ---------------------------------------------------------------------------

class TestSubphraseRedundancy:
    def test_no_redundancy(self):
        red, total = _subphrase_redundancy_ratio(["alpha", "beta", "gamma"])
        assert red == 0
        assert total == 3

    def test_subphrase_detected(self):
        red, total = _subphrase_redundancy_ratio(["machine learning", "learning", "quantum"])
        assert red == 1  # "learning" is subphrase of "machine learning"
        assert total == 3

    def test_empty(self):
        assert _subphrase_redundancy_ratio([]) == (0, 0)

    def test_duplicates_collapsed(self):
        red, total = _subphrase_redundancy_ratio(["abc", "abc", "def"])
        assert total == 2  # deduped

    def test_case_insensitive(self):
        red, total = _subphrase_redundancy_ratio(["Machine Learning", "machine learning model"])
        assert red == 1  # "machine learning" is subphrase of "machine learning model"


# ---------------------------------------------------------------------------
# _token_jaccard_redundancy_ratio
# ---------------------------------------------------------------------------

class TestTokenJaccardRedundancy:
    def test_no_overlap(self):
        red, total = _token_jaccard_redundancy_ratio(
            ["alpha beta", "gamma delta"], threshold=0.5
        )
        assert red == 0

    def test_high_overlap(self):
        red, total = _token_jaccard_redundancy_ratio(
            ["machine learning", "machine learning model", "quantum bit"],
            threshold=0.5,
        )
        # "machine learning" and "machine learning model" share 2/3 Jaccard ≈ 0.67
        assert red >= 2  # both flagged as part of a redundant pair

    def test_empty(self):
        assert _token_jaccard_redundancy_ratio([], threshold=0.5) == (0, 0)

    def test_threshold_boundary(self):
        # Jaccard of {"a","b"} vs {"a","c"} = 1/3 ≈ 0.33
        red, _ = _token_jaccard_redundancy_ratio(["a b", "a c"], threshold=0.5)
        assert red == 0  # below threshold
        red2, _ = _token_jaccard_redundancy_ratio(["a b", "a c"], threshold=0.3)
        assert red2 == 2  # above threshold


# ---------------------------------------------------------------------------
# KeywordDiagnostics.to_dict
# ---------------------------------------------------------------------------

class TestKeywordDiagnosticsDataclass:
    def test_to_dict_roundtrip(self):
        diag = KeywordDiagnostics(
            n_rows=10,
            n_clusters=2,
            terms_per_cluster={"p50": 5.0, "mean": 5.0},
            doc_coverage={"p50": 3.0},
            score={"p50": 1.5},
            years_per_term={"p50": 3.0},
            single_year_term_ratio=0.1,
            redundancy_subphrase_ratio=0.05,
            redundancy_token_jaccard_ratio=0.03,
        )
        d = diag.to_dict()
        assert d["n_rows"] == 10
        assert d["n_clusters"] == 2
        assert isinstance(d["terms_per_cluster"], dict)
        assert d["single_year_term_ratio"] == pytest.approx(0.1)
        assert d["scope_counts"] == {}
        assert d["unresolved_short_form_ratio"] is None

    def test_to_dict_none_values(self):
        diag = KeywordDiagnostics(
            n_rows=0, n_clusters=0,
            terms_per_cluster={}, doc_coverage={}, score={},
            years_per_term={},
            single_year_term_ratio=None,
            redundancy_subphrase_ratio=None,
            redundancy_token_jaccard_ratio=None,
        )
        d = diag.to_dict()
        assert d["single_year_term_ratio"] is None
        assert d["redundancy_subphrase_ratio"] is None


# ---------------------------------------------------------------------------
# keyword_diagnostics (integration)
# ---------------------------------------------------------------------------

class TestKeywordDiagnostics:
    def test_empty_dataframe(self):
        diag = keyword_diagnostics(pd.DataFrame())
        assert diag.n_rows == 0
        assert diag.n_clusters == 0

    def test_none_input(self):
        diag = keyword_diagnostics(None)
        assert diag.n_rows == 0

    def test_minimal_dataframe(self):
        df = pd.DataFrame({
            "cluster_id": [0, 0, 1],
            "term": ["alpha", "beta", "gamma"],
            "score": [1.0, 2.0, 1.5],
        })
        diag = keyword_diagnostics(df, sample_clusters=None)
        assert diag.n_rows == 3
        assert diag.n_clusters == 2
        assert "p50" in diag.score

    def test_with_doc_coverage(self):
        df = pd.DataFrame({
            "cluster_id": [0, 0],
            "term": ["alpha", "beta"],
            "score": [1.0, 2.0],
            "doc_coverage": [10, 20],
        })
        diag = keyword_diagnostics(df, sample_clusters=None)
        assert "p50" in diag.doc_coverage
        assert "mean" in diag.doc_coverage

    def test_with_year_series(self):
        df = pd.DataFrame({
            "cluster_id": [0, 0],
            "term": ["alpha", "beta"],
            "score": [1.0, 2.0],
            "pub_year_series": [{2020: 3, 2021: 5}, {2021: 2}],
        })
        diag = keyword_diagnostics(df, sample_clusters=None)
        assert diag.single_year_term_ratio is not None
        assert "p50" in diag.years_per_term

    def test_no_cluster_column(self):
        df = pd.DataFrame({
            "term": ["alpha", "beta"],
            "score": [1.0, 2.0],
        })
        diag = keyword_diagnostics(df, sample_clusters=None)
        assert diag.n_clusters == 0
        assert diag.redundancy_subphrase_ratio is None

    def test_quality_refinement_diagnostics(self):
        df = pd.DataFrame({
            "cluster_id": [0, 0, 0, 1, 1],
            "term": [
                "traffic flow",
                "traffic flow prediction",
                "eeg",
                "drug drug interaction",
                "ddi",
            ],
            "display_label": [
                "traffic flow",
                "traffic flow prediction",
                "eeg",
                "drug drug interaction",
                "drug drug interaction",
            ],
            "score": [5.0, 4.0, 3.0, 5.0, 1.0],
            "keyword_scope": [
                "cluster_specific",
                "cluster_specific",
                "cluster_specific",
                "cluster_specific",
                "cluster_specific",
            ],
            "abbreviation_status": [
                "not_abbreviation",
                "not_abbreviation",
                "unlinked_short_form",
                "not_abbreviation",
                "duplicate_expansion",
            ],
            "quality_flags": [
                "phrase",
                "phrase",
                "unlinked_short_form|acronym_like",
                "phrase",
                "duplicate_expansion|acronym_like",
            ],
            "representative_role": [
                "representative_phrase",
                "representative_phrase",
                "review_short_form",
                "representative_phrase",
                "duplicate_expansion",
            ],
            "representative_rank": [1, 2, 3, 1, 2],
        })

        diag = keyword_diagnostics(df, sample_clusters=None)

        assert diag.scope_counts == {"cluster_specific": 5}
        assert diag.abbreviation_status_counts["unlinked_short_form"] == 1
        assert diag.unresolved_short_form_ratio == pytest.approx(0.2)
        assert diag.review_flag_ratio == pytest.approx(0.2)
        assert diag.representative_role_counts["representative_phrase"] == 3
        assert diag.representative_diversity_ratio == pytest.approx(4 / 5)
        assert diag.family_compression_ratio == pytest.approx(1 / 6)


# ---------------------------------------------------------------------------
# score_before_after
# ---------------------------------------------------------------------------

class TestScoreBeforeAfter:
    def test_identical_gives_baseline(self):
        df = pd.DataFrame({
            "cluster_id": [0, 0],
            "term": ["alpha", "beta"],
            "score": [1.0, 2.0],
            "doc_coverage": [5, 10],
            "pub_year_series": [{2020: 3}, {2020: 2, 2021: 1}],
        })
        result = score_before_after(df, df, sample_clusters=None)
        # Identical before/after → score near baseline (50)
        assert result["total_score"] == pytest.approx(50.0, abs=1.0)

    def test_empty_before_after(self):
        empty = pd.DataFrame()
        result = score_before_after(empty, empty, sample_clusters=None)
        assert result["total_score"] == pytest.approx(50.0)

    def test_custom_weights(self):
        before = pd.DataFrame({
            "cluster_id": [0], "term": ["x"], "score": [1.0],
            "doc_coverage": [5], "pub_year_series": [{2020: 1}],
        })
        after = pd.DataFrame({
            "cluster_id": [0], "term": ["x"], "score": [1.0],
            "doc_coverage": [10], "pub_year_series": [{2020: 1}],
        })
        result = score_before_after(
            before, after, sample_clusters=None,
            weights={"coverage": 40.0},
        )
        assert "components" in result
        assert result["components"]["coverage"]["weight"] == 40.0
        assert "review_load" in result["components"]
        assert "representative_diversity" in result["components"]

    def test_output_structure(self):
        df = pd.DataFrame({
            "cluster_id": [0], "term": ["x"], "score": [1.0],
        })
        result = score_before_after(df, df, sample_clusters=None)
        assert "total_score" in result
        assert "baseline" in result
        assert "components" in result
        assert "before" in result
        assert "after" in result
        assert "params" in result
        assert 0.0 <= result["total_score"] <= 100.0
