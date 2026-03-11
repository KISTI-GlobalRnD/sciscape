"""Tests for vocabulary-level merge (Stage 2 post-vectorizer)."""

import numpy as np
import pytest
from scipy import sparse as sp

from sciscape.keyword_extraction.config import VocabMergeConfig
from sciscape.keyword_extraction.vocab_merge import (
    _simple_singular,
    apply_merge_map,
    build_merge_map,
)


class TestSimpleSingular:
    def test_regular_plural(self):
        assert _simple_singular("neurons") == "neuron"
        assert _simple_singular("models") == "model"
        assert _simple_singular("results") == "result"

    def test_ies_plural(self):
        assert _simple_singular("batteries") == "battery"
        assert _simple_singular("strategies") == "strategy"

    def test_es_plural(self):
        assert _simple_singular("processes") == "process"
        assert _simple_singular("boxes") == "box"
        assert _simple_singular("matches") == "match"

    def test_short_words_skipped(self):
        assert _simple_singular("as") is None
        assert _simple_singular("us") is None
        assert _simple_singular("yes") is None

    def test_double_s_skipped(self):
        assert _simple_singular("stress") is None
        assert _simple_singular("loss") is None


class TestBuildMergeMap:
    def test_plural_merge(self):
        names = np.array(["neuron", "neurons", "model", "models", "result"])
        cfg = VocabMergeConfig(enabled=True, plural_to_singular=True, hyphen_normalize=False)
        merge = build_merge_map(names, cfg)
        # neurons(1) -> neuron(0), models(3) -> model(2)
        assert merge == {1: 0, 3: 2}

    def test_hyphen_merge(self):
        names = np.array(["machine-learning", "machine learning", "deep learning"])
        cfg = VocabMergeConfig(enabled=True, plural_to_singular=False, hyphen_normalize=True)
        merge = build_merge_map(names, cfg)
        # machine-learning(0) -> machine learning(1)
        assert merge == {0: 1}

    def test_no_merge_when_target_missing(self):
        names = np.array(["neurons", "model"])
        cfg = VocabMergeConfig(enabled=True, plural_to_singular=True, hyphen_normalize=False)
        merge = build_merge_map(names, cfg)
        # "neuron" not in vocabulary, so no merge
        assert merge == {}

    def test_empty_merge_map(self):
        names = np.array(["alpha", "beta", "gamma"])
        cfg = VocabMergeConfig(enabled=True)
        merge = build_merge_map(names, cfg)
        assert merge == {}

    def test_disabled_returns_empty(self):
        names = np.array(["neuron", "neurons"])
        cfg = VocabMergeConfig(enabled=True, plural_to_singular=False, hyphen_normalize=False)
        merge = build_merge_map(names, cfg)
        assert merge == {}


class TestApplyMergeMap:
    def test_merge_columns(self):
        # 2 documents, 3 features: [neuron, neurons, model]
        X = sp.csr_matrix(np.array([
            [5, 3, 2],
            [1, 7, 4],
        ]))
        names = np.array(["neuron", "neurons", "model"])
        merge = {1: 0}  # neurons -> neuron

        X_merged, names_merged = apply_merge_map(X, names, merge)

        assert X_merged.shape == (2, 2)
        np.testing.assert_array_equal(names_merged, ["neuron", "model"])
        # neuron column = original neuron + neurons
        assert X_merged[0, 0] == 8  # 5 + 3
        assert X_merged[1, 0] == 8  # 1 + 7
        assert X_merged[0, 1] == 2
        assert X_merged[1, 1] == 4

    def test_empty_merge_map_noop(self):
        X = sp.csr_matrix(np.array([[1, 2], [3, 4]]))
        names = np.array(["a", "b"])

        X_merged, names_merged = apply_merge_map(X, names, {})

        assert X_merged.shape == X.shape
        np.testing.assert_array_equal(names_merged, names)

    def test_preserves_sparsity(self):
        # Large sparse matrix
        X = sp.random(100, 500, density=0.01, format="csr", dtype=np.float64)
        names = np.array([f"term_{i}" for i in range(500)])
        merge = {}  # no merges

        X_merged, names_merged = apply_merge_map(X, names, merge)
        assert sp.issparse(X_merged)
        assert X_merged.shape == X.shape
