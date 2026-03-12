"""Tests for extraction utilities and shared utils module."""

import math

import numpy as np
import pandas as pd
import pytest
from scipy import sparse as sp

from sciscape.keyword_extraction.extraction import (
    _argpartition_topk,
    _effective_n_jobs,
    _group_sum_by_cluster,
    _llr_2x2,
    _mmr_jaccard_select,
    _suppress_subphrases,
)
from sciscape.keyword_extraction.utils import (
    _edit_distance,
    _normalize_text_basic,
)


# ---------------------------------------------------------------------------
# _effective_n_jobs
# ---------------------------------------------------------------------------

class TestEffectiveNJobs:
    def test_none_returns_1(self):
        assert _effective_n_jobs(None) == 1

    def test_zero_returns_1(self):
        assert _effective_n_jobs(0) == 1

    def test_positive(self):
        assert _effective_n_jobs(4) == 4

    def test_negative_one_uses_cpus(self):
        import os
        result = _effective_n_jobs(-1)
        assert result >= 1
        assert result == max(1, os.cpu_count() or 1)

    def test_negative_clamps_to_1(self):
        assert _effective_n_jobs(-99) == 1

    def test_invalid_type(self):
        assert _effective_n_jobs("abc") == 1


# ---------------------------------------------------------------------------
# _argpartition_topk
# ---------------------------------------------------------------------------

class TestArgpartitionTopk:
    def test_basic(self):
        values = np.array([0.1, 0.5, 0.3, 0.9, 0.2])
        indices = np.array([10, 20, 30, 40, 50])
        result = _argpartition_topk(values, indices, k=3)
        # Should return top-3 sorted descending
        assert len(result) == 3
        assert result[0] == (40, pytest.approx(0.9))
        assert result[1] == (20, pytest.approx(0.5))
        assert result[2] == (30, pytest.approx(0.3))

    def test_k_larger_than_array(self):
        values = np.array([0.5, 0.1])
        indices = np.array([0, 1])
        result = _argpartition_topk(values, indices, k=10)
        assert len(result) == 2

    def test_empty(self):
        values = np.array([], dtype=float)
        indices = np.array([], dtype=int)
        assert _argpartition_topk(values, indices, k=5) == []

    def test_single_element(self):
        values = np.array([3.14])
        indices = np.array([42])
        result = _argpartition_topk(values, indices, k=1)
        assert result == [(42, pytest.approx(3.14))]


# ---------------------------------------------------------------------------
# _group_sum_by_cluster
# ---------------------------------------------------------------------------

class TestGroupSumByCluster:
    def test_basic_aggregation(self):
        # 4 docs, 3 features, 2 clusters
        X = sp.csr_matrix(np.array([
            [1, 0, 2],
            [0, 3, 0],
            [2, 1, 0],
            [0, 0, 4],
        ]))
        codes = np.array([0, 0, 1, 1])
        result = _group_sum_by_cluster(X, codes, K=2)
        assert result.shape == (2, 3)
        dense = result.toarray()
        np.testing.assert_array_equal(dense[0], [1, 3, 2])
        np.testing.assert_array_equal(dense[1], [2, 1, 4])

    def test_empty_matrix(self):
        X = sp.csr_matrix((0, 5))
        codes = np.array([], dtype=int)
        result = _group_sum_by_cluster(X, codes, K=3)
        assert result.shape == (3, 5)
        assert result.nnz == 0


# ---------------------------------------------------------------------------
# _llr_2x2
# ---------------------------------------------------------------------------

class TestLLR2x2:
    def test_zero_table(self):
        assert _llr_2x2(0, 0, 0, 0) == 0.0

    def test_positive_association(self):
        # Term highly concentrated in cluster
        score = _llr_2x2(50, 10, 5, 935)
        assert score > 0

    def test_no_association(self):
        # Uniform distribution
        score = _llr_2x2(25, 25, 25, 25)
        assert score == pytest.approx(0.0, abs=0.1)

    def test_symmetry(self):
        # Swapping rows should give same score
        s1 = _llr_2x2(50, 10, 5, 935)
        s2 = _llr_2x2(5, 935, 50, 10)
        assert s1 == pytest.approx(s2, abs=0.01)


# ---------------------------------------------------------------------------
# _mmr_jaccard_select
# ---------------------------------------------------------------------------

class TestMMRJaccardSelect:
    def test_lambda_zero_returns_original_order(self):
        cands = ["a b", "c d", "e f"]
        scores = {"a b": 3.0, "c d": 2.0, "e f": 1.0}
        result = _mmr_jaccard_select(cands, scores, lambda_=0.0, top_k=2)
        assert result == ["a b", "c d"]

    def test_diversity_penalty(self):
        # "machine learning" and "machine vision" share "machine" (Jaccard 1/3)
        # With high diversity weight, "quantum bit" (no overlap) should beat
        # "machine vision" even with lower relevance
        cands = ["machine learning", "machine vision", "quantum bit"]
        scores = {"machine learning": 3.0, "machine vision": 2.0, "quantum bit": 1.9}
        # lambda=0.3 → heavy diversity weight (1-0.3=0.7)
        result = _mmr_jaccard_select(cands, scores, lambda_=0.3, top_k=2)
        assert result[0] == "machine learning"
        assert result[1] == "quantum bit"

    def test_topk_limit(self):
        cands = ["a", "b", "c", "d"]
        scores = {c: float(i) for i, c in enumerate(cands)}
        result = _mmr_jaccard_select(cands, scores, lambda_=0.5, top_k=2)
        assert len(result) == 2

    def test_empty_candidates(self):
        assert _mmr_jaccard_select([], {}, lambda_=0.5, top_k=5) == []


# ---------------------------------------------------------------------------
# _suppress_subphrases
# ---------------------------------------------------------------------------

class TestSuppressSubphrases:
    def test_subphrase_removed(self):
        terms = ["machine learning model", "machine learning", "quantum"]
        result = _suppress_subphrases(terms, max_keep=10)
        assert "machine learning model" in result
        assert "machine learning" not in result  # subphrase of the first
        assert "quantum" in result

    def test_no_subphrases(self):
        terms = ["alpha", "beta", "gamma"]
        result = _suppress_subphrases(terms, max_keep=10)
        assert result == terms

    def test_max_keep(self):
        terms = ["a", "b", "c", "d", "e"]
        result = _suppress_subphrases(terms, max_keep=3)
        assert len(result) == 3

    def test_empty(self):
        assert _suppress_subphrases([], max_keep=10) == []


# ---------------------------------------------------------------------------
# _normalize_text_basic
# ---------------------------------------------------------------------------

class TestNormalizeTextBasic:
    def test_html_tags_removed(self):
        assert _normalize_text_basic("hello <b>world</b>") == "hello world"

    def test_whitespace_collapsed(self):
        assert _normalize_text_basic("  hello   world  ") == "hello world"

    def test_newlines_removed(self):
        assert _normalize_text_basic("line1\nline2\rline3") == "line1 line2 line3"

    def test_non_string_returns_empty(self):
        assert _normalize_text_basic(None) == ""
        assert _normalize_text_basic(42) == ""

    def test_normal_text_unchanged(self):
        assert _normalize_text_basic("hello world") == "hello world"


# ---------------------------------------------------------------------------
# _edit_distance
# ---------------------------------------------------------------------------

class TestEditDistanceUtils:
    def test_identical(self):
        assert _edit_distance("hello", "hello") == 0

    def test_one_edit(self):
        assert _edit_distance("cat", "bat") == 1

    def test_insertion(self):
        assert _edit_distance("cat", "cats") == 1

    def test_deletion(self):
        assert _edit_distance("cats", "cat") == 1

    def test_empty_strings(self):
        assert _edit_distance("", "") == 0
        assert _edit_distance("abc", "") == 3
        assert _edit_distance("", "abc") == 3

    def test_early_termination(self):
        # Length diff > 3 → returns length diff directly
        assert _edit_distance("a", "abcde") == 4

    def test_symmetric(self):
        assert _edit_distance("kitten", "sitting") == _edit_distance("sitting", "kitten")
