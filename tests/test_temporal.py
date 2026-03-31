"""Tests for temporal metrics computation (Stage 10)."""

import math
from collections import Counter, defaultdict

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


def _make_temporal_stub(term_year, cluster_denoms):
    """Create a minimal TemporalMixin stub for _build_time_series_metrics."""
    pipeline = object.__new__(KeywordExtractionPipeline)
    pipeline.cluster_year_token_denoms = cluster_denoms

    def _log(msg, *args):
        pass
    pipeline._log = _log
    return pipeline


# ---------------------------------------------------------------------------
# Unit tests for _build_time_series_metrics
# ---------------------------------------------------------------------------

class TestBuildTimeSeriesMetrics:
    def test_basic_metrics(self):
        """Basic case: one cluster, one term, two years."""
        term_year = defaultdict(lambda: defaultdict(Counter))
        term_year[0]["alpha"] = Counter({2020: 10, 2021: 20})
        denoms = {0: Counter({2020: 100, 2021: 200})}

        stub = _make_temporal_stub(term_year, denoms)
        top_df = pd.DataFrame({
            "cluster_id": [0],
            "term": ["alpha"],
            "score": [1.0],
            "frequency": [30],
        })
        result = stub._build_time_series_metrics(top_df, term_year)
        assert "ppm_series" in result.columns
        assert "loglift_series" in result.columns
        assert "bayesian_log_odds_series" in result.columns
        assert "year_denominators" in result.columns

        ppm = result["ppm_series"].iloc[0]
        assert 2020 in ppm
        assert ppm[2020] == pytest.approx(1e6 * 10 / 100)

    def test_empty_dataframe(self):
        """Empty top_df returns empty columns."""
        stub = _make_temporal_stub({}, {})
        top_df = pd.DataFrame(columns=["cluster_id", "term", "score", "frequency"])
        result = stub._build_time_series_metrics(top_df, {})
        assert "ppm_series" in result.columns
        assert len(result) == 0

    def test_zero_denominator_gives_nan(self):
        """PPM should be NaN when denominator is zero."""
        term_year = defaultdict(lambda: defaultdict(Counter))
        term_year[0]["alpha"] = Counter({2020: 5})
        denoms = {0: Counter()}  # no denom for 2020

        stub = _make_temporal_stub(term_year, denoms)
        top_df = pd.DataFrame({
            "cluster_id": [0], "term": ["alpha"],
            "score": [1.0], "frequency": [5],
        })
        result = stub._build_time_series_metrics(top_df, term_year)
        ppm = result["ppm_series"].iloc[0]
        assert math.isnan(ppm[2020])

    def test_multi_cluster_isolation(self):
        """Metrics use per-cluster denominators, not global."""
        term_year = defaultdict(lambda: defaultdict(Counter))
        term_year[0]["alpha"] = Counter({2020: 10})
        term_year[1]["alpha"] = Counter({2020: 5})
        denoms = {0: Counter({2020: 100}), 1: Counter({2020: 50})}

        stub = _make_temporal_stub(term_year, denoms)
        top_df = pd.DataFrame({
            "cluster_id": [0, 1],
            "term": ["alpha", "alpha"],
            "score": [1.0, 1.0],
            "frequency": [10, 5],
        })
        result = stub._build_time_series_metrics(top_df, term_year)
        ppm_c0 = result["ppm_series"].iloc[0][2020]
        ppm_c1 = result["ppm_series"].iloc[1][2020]
        # Both should be 100,000 PPM (10/100 and 5/50 = same ratio)
        assert ppm_c0 == pytest.approx(1e5)
        assert ppm_c1 == pytest.approx(1e5)

    def test_loglift_positive_when_concentrated(self):
        """When a term is more concentrated in a cluster than globally, loglift > 0."""
        term_year = defaultdict(lambda: defaultdict(Counter))
        term_year[0]["alpha"] = Counter({2020: 50})
        term_year[1]["beta"] = Counter({2020: 5})
        denoms = {0: Counter({2020: 100}), 1: Counter({2020: 1000})}

        stub = _make_temporal_stub(term_year, denoms)
        top_df = pd.DataFrame({
            "cluster_id": [0],
            "term": ["alpha"],
            "score": [1.0],
            "frequency": [50],
        })
        result = stub._build_time_series_metrics(top_df, term_year)
        loglift = result["loglift_series"].iloc[0]
        # alpha: 50/100 in cluster, 50/1100 globally → loglift > 0
        assert loglift[2020] > 0

    def test_term_not_in_year_data(self):
        """A term with no year data should get empty series."""
        term_year = defaultdict(lambda: defaultdict(Counter))
        denoms = {0: Counter({2020: 100})}

        stub = _make_temporal_stub(term_year, denoms)
        top_df = pd.DataFrame({
            "cluster_id": [0], "term": ["unknown"],
            "score": [1.0], "frequency": [10],
        })
        result = stub._build_time_series_metrics(top_df, term_year)
        assert result["ppm_series"].iloc[0] == {}
        assert result["loglift_series"].iloc[0] == {}

    def test_pre_existing_pub_year_series(self):
        """If top_df already has pub_year_series, use it."""
        term_year = defaultdict(lambda: defaultdict(Counter))
        denoms = {0: Counter({2020: 100})}

        stub = _make_temporal_stub(term_year, denoms)
        top_df = pd.DataFrame({
            "cluster_id": [0], "term": ["alpha"],
            "score": [1.0], "frequency": [10],
            "pub_year_series": [{2020: 10}],
        })
        result = stub._build_time_series_metrics(top_df, term_year)
        ppm = result["ppm_series"].iloc[0]
        assert 2020 in ppm
        assert ppm[2020] == pytest.approx(1e5)


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
