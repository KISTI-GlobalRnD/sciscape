"""Tests for temporal metrics computation (Stage 10)."""

from pathlib import Path

import pandas as pd
import pytest

from sciscape.keyword_extraction import KeywordExtractionConfig, run_keyword_pipeline
from sciscape.keyword_extraction.pipeline import KeywordExtractionPipeline


@pytest.fixture
def sample_data(tmp_path):
    abstracts = pd.DataFrame({
        "uid": [f"D{i}" for i in range(6)],
        "title": [
            "Deep learning neural networks",
            "Neural network architectures",
            "Machine learning models",
            "Deep learning optimization",
            "Quantum computing bits",
            "Quantum error correction",
        ],
        "abstract": [
            "Deep learning with neural networks enables pattern recognition.",
            "Neural network architecture design improves classification.",
            "Machine learning models predict material properties.",
            "Deep learning optimization algorithms accelerate training.",
            "Quantum computing with quantum bits offers parallel computation.",
            "Quantum error correction ensures reliable computation.",
        ],
        "pubyear": [2018, 2019, 2020, 2021, 2019, 2020],
    })
    membership = pd.DataFrame({
        "uid": [f"D{i}" for i in range(6)],
        "cluster": [0, 0, 0, 0, 1, 1],
    })
    abs_path = tmp_path / "abstracts.parquet"
    mem_path = tmp_path / "membership.parquet"
    abstracts.to_parquet(abs_path, index=False)
    membership.to_parquet(mem_path, index=False)
    return abs_path, mem_path


def _cfg(abs_path, mem_path, **overrides):
    defaults = dict(
        abstract_path=abs_path,
        membership_path=mem_path,
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


class TestTemporalOutput:
    def test_year_series_columns_present(self, sample_data):
        cfg = _cfg(*sample_data)
        result = run_keyword_pipeline(cfg)
        assert not result.empty
        for col in ("pub_year_series", "ppm_series", "loglift_series",
                     "bayesian_log_odds_series", "year_denominators"):
            assert col in result.columns, f"Missing column: {col}"

    def test_pub_year_series_has_year_keys(self, sample_data):
        cfg = _cfg(*sample_data)
        result = run_keyword_pipeline(cfg)
        for series in result["pub_year_series"]:
            if series:
                for key in series.keys():
                    assert isinstance(key, int), f"Year key should be int, got {type(key)}"

    def test_ppm_values_positive(self, sample_data):
        cfg = _cfg(*sample_data)
        result = run_keyword_pipeline(cfg)
        for ppm in result["ppm_series"]:
            for year, val in ppm.items():
                if not (val != val):  # skip NaN
                    assert val >= 0, f"PPM should be non-negative, got {val}"

    def test_multi_cluster_year_series(self, sample_data):
        """Year series are computed per-cluster, not globally."""
        cfg = _cfg(*sample_data)
        result = run_keyword_pipeline(cfg)
        # Cluster 0 has years 2018-2021, cluster 1 has years 2019-2020
        c1_rows = result[result["cluster_id"] == 1]
        for series in c1_rows["pub_year_series"]:
            if series:
                for year in series.keys():
                    assert 2019 <= year <= 2020

    def test_single_year_data(self, tmp_path):
        """All documents from same year — temporal metrics should still work."""
        abstracts = pd.DataFrame({
            "uid": ["D0", "D1", "D2"],
            "title": ["neural network", "deep learning", "machine learning"],
            "abstract": [
                "neural network for classification",
                "deep learning for recognition",
                "machine learning for prediction",
            ],
            "pubyear": [2020, 2020, 2020],
        })
        membership = pd.DataFrame({
            "uid": ["D0", "D1", "D2"],
            "cluster": [0, 0, 0],
        })
        abs_path = tmp_path / "single_year_abs.parquet"
        mem_path = tmp_path / "single_year_mem.parquet"
        abstracts.to_parquet(abs_path, index=False)
        membership.to_parquet(mem_path, index=False)
        cfg = _cfg(abs_path, mem_path)
        result = run_keyword_pipeline(cfg)
        assert not result.empty
        # All year series should have exactly one year
        for series in result["pub_year_series"]:
            if series:
                assert len(series) == 1
                assert 2020 in series
