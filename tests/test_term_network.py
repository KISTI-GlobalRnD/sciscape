"""Tests for multi-layer term similarity network (Stage 7)."""

import numpy as np
import pytest
from scipy import sparse as sp

from sciscape.keyword_extraction.term_network import (
    TermNetwork,
    TermNetworkConfig,
    _build_blocks,
    _char_ngrams,
    _edit_distance,
    _jaccard,
)


class TestCharNgrams:
    def test_basic(self):
        ngrams = _char_ngrams("hello", 3)
        assert "hel" in ngrams
        assert "ell" in ngrams
        assert "llo" in ngrams

    def test_short_string(self):
        ngrams = _char_ngrams("ab", 3)
        assert ngrams == {"ab"}


class TestJaccard:
    def test_identical(self):
        assert _jaccard({"a", "b"}, {"a", "b"}) == 1.0

    def test_disjoint(self):
        assert _jaccard({"a"}, {"b"}) == 0.0

    def test_partial(self):
        assert _jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)

    def test_empty(self):
        assert _jaccard(set(), set()) == 0.0


class TestEditDistance:
    def test_identical(self):
        assert _edit_distance("hello", "hello") == 0

    def test_one_edit(self):
        assert _edit_distance("cat", "bat") == 1


class TestBuildBlocks:
    def test_token_blocking(self):
        terms = ["machine learning", "deep learning", "machine vision"]
        blocks = _build_blocks(terms, "token")
        # "learning" block should contain indices 0 and 1
        assert 0 in blocks.get("learning", [])
        assert 1 in blocks.get("learning", [])
        # "machine" block should contain indices 0 and 2
        assert 0 in blocks.get("machine", [])
        assert 2 in blocks.get("machine", [])

    def test_prefix_blocking(self):
        terms = ["neural", "neuron", "network"]
        blocks = _build_blocks(terms, "prefix", prefix_length=3)
        assert 0 in blocks.get("neu", [])
        assert 1 in blocks.get("neu", [])
        assert 2 in blocks.get("net", [])


class TestTermNetworkStringLayer:
    def test_similar_terms_detected(self):
        cfg = TermNetworkConfig(
            enabled=True,
            max_edit_distance=2,
            min_char_ngram_sim=0.3,
            blocking_strategy="prefix",
            prefix_length=3,
        )
        net = TermNetwork(cfg)
        terms = ["neuron", "neurons", "network", "deep"]
        layer = net.build_layer_string(terms)

        assert layer.shape == (4, 4)
        # neuron and neurons should have high similarity
        assert layer[0, 1] > 0.5

    def test_dissimilar_terms_zero(self):
        cfg = TermNetworkConfig(enabled=True, min_char_ngram_sim=0.3)
        net = TermNetwork(cfg)
        terms = ["alpha", "zzzzz"]
        layer = net.build_layer_string(terms)
        # These should not match (no shared tokens for blocking)
        assert layer.nnz == 0 or layer[0, 1] == 0

    def test_empty_terms(self):
        cfg = TermNetworkConfig(enabled=True)
        net = TermNetwork(cfg)
        layer = net.build_layer_string([])
        assert layer.shape == (0, 0)


class TestTermNetworkTokenLayer:
    def test_abbreviation_detection(self):
        cfg = TermNetworkConfig(
            enabled=True,
            min_token_overlap=0.3,
            blocking_strategy="token",
        )
        net = TermNetwork(cfg)
        terms = ["ml", "machine learning", "deep learning"]
        layer = net.build_layer_token(terms)

        # "ml" should match "machine learning" via abbreviation
        assert layer[0, 1] > 0.5

    def test_containment(self):
        cfg = TermNetworkConfig(enabled=True, min_token_overlap=0.3)
        net = TermNetwork(cfg)
        terms = ["learning", "machine learning", "deep learning"]
        layer = net.build_layer_token(terms)

        # "learning" is contained in "machine learning"
        assert layer[0, 1] > 0.0

    def test_token_overlap(self):
        cfg = TermNetworkConfig(enabled=True, min_token_overlap=0.3)
        net = TermNetwork(cfg)
        terms = ["machine learning model", "machine learning algorithm"]
        layer = net.build_layer_token(terms)

        # 2/4 Jaccard overlap = 0.5
        assert layer[0, 1] >= 0.5


class TestTermNetworkCooccurrenceLayer:
    def test_normalization(self):
        cfg = TermNetworkConfig(enabled=True)
        net = TermNetwork(cfg)
        cooc = sp.csr_matrix(np.array([
            [0, 10, 2],
            [10, 0, 1],
            [2, 1, 0],
        ], dtype=np.int64))
        layer = net.build_layer_cooccurrence(cooc)

        assert layer.shape == (3, 3)
        # Values should be in [0, 1]
        assert layer.data.max() <= 1.0
        assert layer.data.min() >= 0.0


class TestCombineAndMerge:
    def test_combine_layers(self):
        cfg = TermNetworkConfig(enabled=True)
        net = TermNetwork(cfg)

        layer1 = sp.csr_matrix(np.array([[0, 0.8], [0.8, 0]], dtype=np.float32))
        layer2 = sp.csr_matrix(np.array([[0, 0.6], [0.6, 0]], dtype=np.float32))

        combined = net.combine_layers([layer1, layer2], [1.0, 1.0])
        # Weighted average: (0.8 + 0.6) / 2 = 0.7
        assert combined[0, 1] == pytest.approx(0.7, abs=0.01)

    def test_find_merge_groups(self):
        cfg = TermNetworkConfig(enabled=True, merge_threshold=0.4)
        net = TermNetwork(cfg)

        # 4 terms: 0-1 similar, 2-3 similar, no cross-group edges
        combined = sp.csr_matrix(np.array([
            [0, 0.8, 0, 0],
            [0.8, 0, 0, 0],
            [0, 0, 0, 0.9],
            [0, 0, 0.9, 0],
        ], dtype=np.float32))

        terms = ["neuron", "neurons", "model", "models"]
        groups = net.find_merge_groups(combined, terms)

        assert len(groups) == 2
        group_sets = [set(g) for g in groups]
        assert {"neuron", "neurons"} in group_sets
        assert {"model", "models"} in group_sets

    def test_no_groups_below_threshold(self):
        cfg = TermNetworkConfig(enabled=True, merge_threshold=0.9)
        net = TermNetwork(cfg)

        combined = sp.csr_matrix(np.array([
            [0, 0.3],
            [0.3, 0],
        ], dtype=np.float32))

        terms = ["alpha", "beta"]
        groups = net.find_merge_groups(combined, terms)
        assert len(groups) == 0

    def test_generate_candidate_sets(self):
        cfg = TermNetworkConfig(enabled=True)
        net = TermNetwork(cfg)

        groups = [["neuron", "neurons"], ["model", "models"]]
        candidates = net.generate_candidate_sets(groups)

        assert len(candidates) == 2
        assert candidates[0]["size"] == 2
        assert "neuron" in candidates[0]["terms"]
