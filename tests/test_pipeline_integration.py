"""Integration tests for the full pipeline with new stages enabled."""

from pathlib import Path

import pandas as pd
import pytest

from sciscape.keyword_extraction import (
    KeywordExtractionConfig,
    VocabMergeConfig,
    run_keyword_pipeline,
)
from sciscape.keyword_extraction.depth import DepthConfig
from sciscape.keyword_extraction.term_network import TermNetworkConfig


@pytest.fixture
def sample_data(tmp_path):
    """Create minimal sample dataset for pipeline testing."""
    abstracts = pd.DataFrame({
        "uid": [f"D{i}" for i in range(8)],
        "title": [
            "Deep learning neural networks",
            "Neural network architectures",
            "Machine learning models for prediction",
            "Machine-learning optimization algorithms",
            "Quantum computing quantum bits",
            "Quantum bit error correction",
            "Solar energy battery storage",
            "Battery systems for solar energy",
        ],
        "abstract": [
            "Deep learning with neural networks enables pattern recognition in images.",
            "Neural network architecture design improves classification accuracy.",
            "Machine learning models predict material properties using features.",
            "Machine-learning optimization algorithms accelerate training convergence.",
            "Quantum computing with quantum bits offers parallel computation.",
            "Quantum bit error correction ensures reliable quantum computation.",
            "Solar energy combined with battery storage provides grid resilience.",
            "Battery systems integrated with solar energy panels reduce costs.",
        ],
        "pubyear": [2018, 2019, 2020, 2021, 2018, 2019, 2020, 2021],
    })
    membership = pd.DataFrame({
        "uid": [f"D{i}" for i in range(8)],
        "cluster": [0, 0, 0, 0, 1, 1, 2, 2],
    })

    abstract_path = tmp_path / "abstracts.parquet"
    membership_path = tmp_path / "membership.parquet"
    abstracts.to_parquet(abstract_path, index=False)
    membership.to_parquet(membership_path, index=False)

    return abstract_path, membership_path


def _base_config(abstract_path, membership_path, **overrides):
    defaults = dict(
        abstract_path=abstract_path,
        membership_path=membership_path,
        cluster_level="cluster",
        include_title=True,
        title_weight=1.0,
        min_df_unigram=1,
        min_df_phrase=1,
        phrase_min_count_per_cluster=1,
        top_n_keywords=5,
        ngram_min=2,
        ngram_max=2,
        use_phrase_vectorizer=True,
        n_jobs=1,
    )
    defaults.update(overrides)
    return KeywordExtractionConfig(**defaults)


class TestPipelineWithVocabMerge:
    def test_vocab_merge_enabled(self, sample_data):
        cfg = _base_config(*sample_data, vocab_merge=VocabMergeConfig(enabled=True))
        keywords = run_keyword_pipeline(cfg)
        assert not keywords.empty
        assert "term" in keywords.columns

    def test_vocab_merge_disabled_is_default(self, sample_data):
        cfg = _base_config(*sample_data)
        keywords = run_keyword_pipeline(cfg)
        assert not keywords.empty


class TestPipelineWithNormalization:
    def test_normalization_enabled(self, sample_data):
        cfg = _base_config(*sample_data, normalization_enabled=True)
        keywords = run_keyword_pipeline(cfg)
        assert not keywords.empty

    def test_normalization_disabled(self, sample_data):
        cfg = _base_config(*sample_data, normalization_enabled=False)
        keywords = run_keyword_pipeline(cfg)
        assert not keywords.empty


class TestPipelineWithCooccurrence:
    def test_cooccurrence_enabled(self, sample_data):
        cfg = _base_config(*sample_data, cooccurrence_enabled=True, cooccurrence_min_count=1)
        from sciscape.keyword_extraction.pipeline import KeywordExtractionPipeline
        pipeline = KeywordExtractionPipeline(cfg)
        keywords = pipeline.run()
        assert not keywords.empty
        # co-occurrence matrix should have been computed
        assert pipeline.cooc_matrix is not None
        assert pipeline.cooc_matrix.shape[0] > 0


class TestPipelineWithTermNetwork:
    def test_term_network_enabled(self, sample_data):
        tn_cfg = TermNetworkConfig(enabled=True, layers=["string", "token"], merge_threshold=0.5)
        cfg = _base_config(
            *sample_data,
            cooccurrence_enabled=True,
            cooccurrence_min_count=1,
            term_network=tn_cfg,
        )
        from sciscape.keyword_extraction.pipeline import KeywordExtractionPipeline
        pipeline = KeywordExtractionPipeline(cfg)
        keywords = pipeline.run()
        assert not keywords.empty

    def test_merge_candidates_injected_as_column(self, sample_data):
        """Stage 7→8 bridge: merge_candidates become per-term candidates column."""
        tn_cfg = TermNetworkConfig(enabled=True, layers=["string", "token"], merge_threshold=0.3)
        cfg = _base_config(
            *sample_data,
            cooccurrence_enabled=True,
            cooccurrence_min_count=1,
            term_network=tn_cfg,
        )
        from sciscape.keyword_extraction.pipeline import KeywordExtractionPipeline
        pipeline = KeywordExtractionPipeline(cfg)
        keywords = pipeline.run()
        assert not keywords.empty
        # candidates column should always exist after bridge (possibly empty lists)
        assert "candidates" in keywords.columns
        # Each entry should be a list
        for cands in keywords["candidates"]:
            assert isinstance(cands, list)


class TestPipelineWithDepth:
    def test_depth_enabled(self, sample_data):
        depth_cfg = DepthConfig(enabled=True, n_levels=3)
        cfg = _base_config(*sample_data, depth=depth_cfg)
        keywords = run_keyword_pipeline(cfg)
        assert not keywords.empty
        assert "depth_score" in keywords.columns
        assert "depth_level" in keywords.columns

    def test_depth_with_cooccurrence(self, sample_data):
        depth_cfg = DepthConfig(enabled=True, n_levels=3, weight_cooc_asymmetry=0.5)
        cfg = _base_config(
            *sample_data,
            cooccurrence_enabled=True,
            cooccurrence_min_count=1,
            depth=depth_cfg,
        )
        from sciscape.keyword_extraction.pipeline import KeywordExtractionPipeline
        pipeline = KeywordExtractionPipeline(cfg)
        keywords = pipeline.run()
        assert not keywords.empty
        assert "depth_score" in keywords.columns


class TestPipelineAllStages:
    def test_all_stages_enabled(self, sample_data):
        """Smoke test: run pipeline with all optional stages enabled."""
        cfg = _base_config(
            *sample_data,
            vocab_merge=VocabMergeConfig(enabled=True),
            normalization_enabled=True,
            cooccurrence_enabled=True,
            cooccurrence_min_count=1,
            term_network=TermNetworkConfig(enabled=True, layers=["string", "token"]),
            depth=DepthConfig(enabled=True, n_levels=3),
        )
        from sciscape.keyword_extraction.pipeline import KeywordExtractionPipeline
        pipeline = KeywordExtractionPipeline(cfg)
        keywords = pipeline.run()

        assert not keywords.empty
        assert "depth_score" in keywords.columns
        assert "depth_level" in keywords.columns
        assert "cross_cluster_count" in keywords.columns
        assert pipeline.cooc_matrix is not None


class TestCheckpointSystem:
    def test_save_and_resume_from_scoring(self, sample_data, tmp_path):
        """Save checkpoint after scoring, resume and get same final shape."""
        cfg = _base_config(*sample_data)
        from sciscape.keyword_extraction.pipeline import KeywordExtractionPipeline
        pipeline = KeywordExtractionPipeline(cfg)

        # Full run for reference
        full_result = pipeline.run()

        # Save checkpoint after scoring
        ckpt_dir = tmp_path / "ckpt_scoring"
        top_df = pipeline._stage_scores_and_topk()
        pipeline.save_checkpoint(ckpt_dir, top_df, stage="scoring")

        # Resume from checkpoint
        pipeline2 = KeywordExtractionPipeline(cfg)
        resumed = pipeline2.run_from_checkpoint(ckpt_dir)

        assert not resumed.empty
        assert set(full_result.columns) == set(resumed.columns)

    def test_backward_compat_stage2_snapshot(self, sample_data, tmp_path):
        """Legacy save_stage2_snapshot / run_from_stage2_snapshot still works."""
        cfg = _base_config(*sample_data)
        from sciscape.keyword_extraction.pipeline import KeywordExtractionPipeline
        pipeline = KeywordExtractionPipeline(cfg)
        pipeline.run()

        # Use legacy API
        top_df = pipeline._stage_scores_and_topk()
        ckpt_dir = tmp_path / "legacy_ckpt"
        pipeline.save_stage2_snapshot(ckpt_dir, top_df)

        pipeline2 = KeywordExtractionPipeline(cfg)
        result = pipeline2.run_from_stage2_snapshot(ckpt_dir)
        assert not result.empty
        assert "term" in result.columns

    def test_checkpoint_preserves_cooc_matrix(self, sample_data, tmp_path):
        """Co-occurrence matrix is saved and restored from checkpoint."""
        cfg = _base_config(
            *sample_data,
            cooccurrence_enabled=True,
            cooccurrence_min_count=1,
        )
        from sciscape.keyword_extraction.pipeline import KeywordExtractionPipeline
        pipeline = KeywordExtractionPipeline(cfg)
        keywords = pipeline.run()

        # Save checkpoint after cooccurrence
        ckpt_dir = tmp_path / "ckpt_cooc"
        pipeline.save_checkpoint(ckpt_dir, keywords, stage="cooccurrence")

        # Resume — should have cooc_matrix restored
        pipeline2 = KeywordExtractionPipeline(cfg)
        pipeline2.load_checkpoint(ckpt_dir)
        assert pipeline2.cooc_matrix is not None
