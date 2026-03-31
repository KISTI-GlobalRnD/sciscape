"""Edge case tests for keyword extraction pipeline stages."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse as sp

from sciscape.keyword_extraction.config import VocabMergeConfig
from sciscape.keyword_extraction.depth import DepthConfig, estimate_depth
from sciscape.keyword_extraction.normalization import (
    _build_norm_blocks,
    normalize_keywords,
)
from sciscape.keyword_extraction.term_network import TermNetwork, TermNetworkConfig
from sciscape.keyword_extraction.vocab_merge import _simple_singular, build_merge_map


# ---- Depth edge cases ----

class TestDepthEdgeCases:
    def _make_df(self, rows):
        return pd.DataFrame(rows, columns=["cluster_id", "term", "score", "frequency", "doc_coverage"])

    def test_single_term(self):
        df = self._make_df([(0, "radiation", 2.0, 500, 100)])
        result = estimate_depth(df, config=DepthConfig(enabled=True, n_levels=3))
        assert len(result) == 1
        assert "depth_score" in result.columns

    def test_all_equal_scores(self):
        df = self._make_df([
            (0, "alpha", 1.0, 100, 50),
            (0, "beta", 1.0, 100, 50),
            (0, "gamma", 1.0, 100, 50),
        ])
        result = estimate_depth(df, config=DepthConfig(enabled=True))
        # All equal inputs should produce equal depth scores
        assert result["depth_score"].nunique() == 1

    def test_nan_doc_coverage(self):
        df = pd.DataFrame({
            "cluster_id": [0, 0],
            "term": ["a_term", "b_term"],
            "score": [1.0, 2.0],
            "frequency": [10, 20],
            "doc_coverage": [float("nan"), 5.0],
        })
        # Should not crash
        result = estimate_depth(df, config=DepthConfig(enabled=True, weight_doc_coverage=1.0))
        assert len(result) == 2


# ---- Normalization edge cases ----

class TestNormalizationEdgeCases:
    def _make_df(self, rows):
        return pd.DataFrame(rows, columns=["cluster_id", "term", "score", "frequency"])

    def test_zero_frequency_terms(self):
        df = self._make_df([(0, "alpha", 1.0, 0), (0, "alphaa", 0.5, 0)])
        result = normalize_keywords(df, {}, max_edit_distance=2)
        assert len(result) >= 1

    def test_unicode_terms(self):
        df = self._make_df([(0, "réseau neuronal", 1.0, 100)])
        result = normalize_keywords(df, {})
        assert len(result) == 1

    def test_single_char_terms(self):
        df = self._make_df([(0, "x", 1.0, 50), (0, "y", 0.8, 40)])
        result = normalize_keywords(df, {}, max_edit_distance=1)
        # Short terms (<=3 chars) should not be edit-distance merged
        assert len(result) == 2

    def test_short_term_blocking(self):
        """Short terms should be placed in _short block."""
        blocks = _build_norm_blocks(["ai", "ml", "network"], max_edit_distance=2)
        assert "_short" in blocks
        assert len(blocks["_short"]) == 2  # ai and ml


# ---- Term Network edge cases ----

class TestTermNetworkEdgeCases:
    def test_single_term(self):
        cfg = TermNetworkConfig(enabled=True, layers=["string"])
        network = TermNetwork(cfg)
        layer = network.build_layer_string(["hello"])
        assert layer.shape == (1, 1)

    def test_empty_terms(self):
        cfg = TermNetworkConfig(enabled=True, layers=["string"])
        network = TermNetwork(cfg)
        layer = network.build_layer_string([])
        assert layer.shape == (0, 0)

    def test_unicode_in_terms(self):
        cfg = TermNetworkConfig(enabled=True, layers=["string", "token"])
        network = TermNetwork(cfg)
        terms = ["réseau", "reseau", "network"]
        layer = network.build_layer_string(terms)
        assert layer.shape == (3, 3)


# ---- Vocab Merge edge cases ----

class TestVocabMergeEdgeCases:
    def test_all_zeros_count_matrix(self):
        names = np.array(["neuron", "neurons"])
        C = sp.csr_matrix(np.zeros((2, 2)))
        cfg = VocabMergeConfig(enabled=True, merge_frequency_ratio=0.01)
        merge = build_merge_map(names, cfg, C=C)
        # Both have 0 frequency, major=0 → freq_ok returns True
        assert merge == {1: 0}

    def test_singular_ambiguous(self):
        """'lives' could be plural of 'life' or verb 'live'."""
        # _simple_singular doesn't handle irregular plurals
        result = _simple_singular("lives")
        # Should return "live" (regular rule), not "life"
        assert result == "live"

    def test_merge_preserves_order(self):
        names = np.array(["network", "networks", "model", "models", "deep"])
        cfg = VocabMergeConfig(enabled=True, plural_to_singular=True)
        merge = build_merge_map(names, cfg)
        # Should consistently produce networks→network, models→model
        assert 1 in merge and merge[1] == 0
        assert 3 in merge and merge[3] == 2
