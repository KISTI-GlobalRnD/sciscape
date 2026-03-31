"""Tests for post-top-K keyword normalization (Stage 5)."""

import pandas as pd

from sciscape.keyword_extraction.normalization import (
    _edit_distance,
    _expand_abbreviations,
    _normalize_notation,
    normalize_keywords,
)


class TestExpandAbbreviations:
    def test_known_abbreviation(self):
        aliases = {"bq": "becquerel", "sv": "sievert"}
        assert _expand_abbreviations("bq", aliases) == "becquerel"
        assert _expand_abbreviations("Bq", aliases) == "becquerel"

    def test_unknown_term_unchanged(self):
        aliases = {"bq": "becquerel"}
        assert _expand_abbreviations("neutron", aliases) == "neutron"

    def test_empty_aliases(self):
        assert _expand_abbreviations("bq", {}) == "bq"


class TestNormalizeNotation:
    def test_greek_letter(self):
        assert _normalize_notation("γ-ray") == "gamma ray"
        assert _normalize_notation("α particle") == "alpha particle"

    def test_hyphen_to_space(self):
        assert _normalize_notation("machine-learning") == "machine learning"

    def test_no_change(self):
        assert _normalize_notation("neutron") == "neutron"

    def test_multiple_greek(self):
        assert _normalize_notation("α-β decay") == "alpha beta decay"


class TestEditDistance:
    def test_identical(self):
        assert _edit_distance("hello", "hello") == 0

    def test_one_char_diff(self):
        assert _edit_distance("cat", "bat") == 1

    def test_insertion(self):
        assert _edit_distance("cat", "cats") == 1

    def test_empty(self):
        assert _edit_distance("", "abc") == 3
        assert _edit_distance("abc", "") == 3


class TestNormalizeKeywords:
    def _make_df(self, rows):
        return pd.DataFrame(rows, columns=["cluster_id", "term", "score", "frequency"])

    def test_abbreviation_expansion(self):
        df = self._make_df([
            (0, "bq", 1.0, 100),
            (0, "neutron", 2.0, 200),
        ])
        aliases = {"bq": "becquerel"}
        result = normalize_keywords(df, aliases)
        terms = result["term"].tolist()
        assert "becquerel" in terms
        assert "bq" not in terms

    def test_greek_normalization(self):
        df = self._make_df([
            (0, "γ-ray", 1.5, 150),
            (0, "neutron", 2.0, 200),
        ])
        result = normalize_keywords(df, {})
        terms = result["term"].tolist()
        assert "gamma ray" in terms

    def test_edit_distance_merge(self):
        df = self._make_df([
            (0, "machine learning", 2.0, 5000),
            (0, "machien learning", 0.5, 3),
        ])
        result = normalize_keywords(df, {}, max_edit_distance=2, min_frequency_ratio=0.01)
        # "machien learning" should be merged into "machine learning"
        assert len(result) == 1
        assert result.iloc[0]["term"] == "machine learning"
        assert result.iloc[0]["frequency"] == 5003

    def test_no_merge_high_frequency(self):
        # Both terms have significant frequency — don't merge
        df = self._make_df([
            (0, "model", 2.0, 500),
            (0, "modal", 1.5, 400),
        ])
        result = normalize_keywords(df, {}, max_edit_distance=2, min_frequency_ratio=0.01)
        # 400 > 0.01 * 500, so no merge
        assert len(result) == 2

    def test_empty_df(self):
        df = self._make_df([])
        result = normalize_keywords(df, {})
        assert result.empty

    def test_preserves_cluster_grouping(self):
        df = self._make_df([
            (0, "bq", 1.0, 100),
            (1, "bq", 1.5, 200),
        ])
        aliases = {"bq": "becquerel"}
        result = normalize_keywords(df, aliases)
        assert len(result) == 2
        assert set(result["cluster_id"]) == {0, 1}

    def test_preserves_extra_columns(self):
        df = pd.DataFrame({
            "cluster_id": [0, 0],
            "term": ["neutron", "proton"],
            "score": [2.0, 1.5],
            "frequency": [200, 150],
            "doc_coverage": [50, 40],
        })
        result = normalize_keywords(df, {})
        assert "doc_coverage" in result.columns
