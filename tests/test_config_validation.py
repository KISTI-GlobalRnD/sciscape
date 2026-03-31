"""Tests for KeywordExtractionConfig parameter validation."""

from pathlib import Path

import pytest

from sciscape.keyword_extraction.config import KeywordExtractionConfig
from sciscape.keyword_extraction.depth import DepthConfig
from sciscape.keyword_extraction.term_network import TermNetworkConfig


# Use dummy paths (validation doesn't check file existence)
_DUMMY = dict(abstract_path=Path("/tmp/a.parquet"), membership_path=Path("/tmp/m.parquet"))


class TestConfigValidation:
    def test_valid_defaults(self):
        """Default parameters should pass validation."""
        cfg = KeywordExtractionConfig(**_DUMMY)
        assert cfg.top_n_keywords == 100

    def test_ngram_min_gt_max(self):
        with pytest.raises(ValueError, match="ngram_min"):
            KeywordExtractionConfig(**_DUMMY, ngram_min=3, ngram_max=2)

    def test_top_n_keywords_zero(self):
        with pytest.raises(ValueError, match="top_n_keywords"):
            KeywordExtractionConfig(**_DUMMY, top_n_keywords=0)

    def test_top_n_unigrams_negative(self):
        with pytest.raises(ValueError, match="top_n_unigrams"):
            KeywordExtractionConfig(**_DUMMY, top_n_unigrams=-1)

    def test_mmr_lambda_out_of_range(self):
        with pytest.raises(ValueError, match="mmr_jaccard_lambda"):
            KeywordExtractionConfig(**_DUMMY, mmr_jaccard_lambda=1.5)

    def test_mmr_pool_factor_below_one(self):
        with pytest.raises(ValueError, match="mmr_pool_factor"):
            KeywordExtractionConfig(**_DUMMY, mmr_pool_factor=0.5)

    def test_norm_edit_distance_negative(self):
        with pytest.raises(ValueError, match="norm_max_edit_distance"):
            KeywordExtractionConfig(**_DUMMY, norm_max_edit_distance=-1)

    def test_norm_freq_ratio_out_of_range(self):
        with pytest.raises(ValueError, match="norm_min_frequency_ratio"):
            KeywordExtractionConfig(**_DUMMY, norm_min_frequency_ratio=2.0)

    def test_cross_cluster_min_count_zero(self):
        with pytest.raises(ValueError, match="cross_cluster_penalty_min_count"):
            KeywordExtractionConfig(**_DUMMY, cross_cluster_penalty_min_count=0)

    def test_cross_cluster_fn_invalid(self):
        with pytest.raises(ValueError, match="cross_cluster_penalty_fn"):
            KeywordExtractionConfig(**_DUMMY, cross_cluster_penalty_fn="sqrt")

    def test_expansion_mode_invalid(self):
        with pytest.raises(ValueError, match="short_term_expansion_mode"):
            KeywordExtractionConfig(**_DUMMY, short_term_expansion_mode="delete")

    def test_auto_merge_sim_out_of_range(self):
        with pytest.raises(ValueError, match="auto_merge_min_similarity"):
            KeywordExtractionConfig(**_DUMMY, auto_merge_min_similarity=-0.1)

    def test_candidate_max_zero(self):
        with pytest.raises(ValueError, match="alias_candidate_max"):
            KeywordExtractionConfig(**_DUMMY, alias_candidate_max=0)

    def test_scoring_weights_both_zero(self):
        with pytest.raises(ValueError, match="w_ctfidf.*w_llr"):
            KeywordExtractionConfig(**_DUMMY, w_ctfidf=0.0, w_llr=0.0)

    def test_boundary_values_pass(self):
        """Edge values at boundaries should pass."""
        cfg = KeywordExtractionConfig(
            **_DUMMY,
            mmr_jaccard_lambda=0.0,
            mmr_pool_factor=1.0,
            norm_max_edit_distance=0,
            norm_min_frequency_ratio=0.0,
            auto_merge_min_similarity=1.0,
        )
        assert cfg.mmr_jaccard_lambda == 0.0


class TestDepthConfigValidation:
    def test_valid_defaults(self):
        cfg = DepthConfig(enabled=True)
        assert cfg.n_levels == 4

    def test_n_levels_too_small(self):
        with pytest.raises(ValueError, match="n_levels"):
            DepthConfig(n_levels=1)

    def test_temporal_fraction_out_of_range(self):
        with pytest.raises(ValueError, match="temporal_recent_fraction"):
            DepthConfig(temporal_recent_fraction=1.5)


class TestTermNetworkConfigValidation:
    def test_valid_defaults(self):
        cfg = TermNetworkConfig(enabled=True)
        assert cfg.max_group_size == 5

    def test_max_block_size_zero(self):
        with pytest.raises(ValueError, match="max_block_size"):
            TermNetworkConfig(max_block_size=0)

    def test_max_group_size_one(self):
        with pytest.raises(ValueError, match="max_group_size"):
            TermNetworkConfig(max_group_size=1)

    def test_merge_threshold_out_of_range(self):
        with pytest.raises(ValueError, match="merge_threshold"):
            TermNetworkConfig(merge_threshold=2.0)

    def test_invalid_blocking_strategy(self):
        with pytest.raises(ValueError, match="blocking_strategy"):
            TermNetworkConfig(blocking_strategy="random")
