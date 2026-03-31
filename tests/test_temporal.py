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


# ---------------------------------------------------------------------------
# PPM calculation: exact numeric verification
# ---------------------------------------------------------------------------

class TestPPMCalculation:
    """Verify PPM = 1e6 * count / denom for known inputs."""

    def test_ppm_exact_multi_year(self):
        """PPM is computed independently per year."""
        term_year = defaultdict(lambda: defaultdict(Counter))
        term_year[0]["alpha"] = Counter({2020: 5, 2021: 20, 2022: 0})
        denoms = {0: Counter({2020: 500, 2021: 1000, 2022: 200})}

        stub = _make_temporal_stub(term_year, denoms)
        top_df = pd.DataFrame({
            "cluster_id": [0], "term": ["alpha"],
            "score": [1.0], "frequency": [25],
        })
        result = stub._build_time_series_metrics(top_df, term_year)
        ppm = result["ppm_series"].iloc[0]

        assert ppm[2020] == pytest.approx(1e6 * 5 / 500)      # 10_000
        assert ppm[2021] == pytest.approx(1e6 * 20 / 1000)     # 20_000
        # year 2022 has count=0 so it is excluded from year_counts_sorted
        assert 2022 not in ppm

    def test_ppm_fractional_values(self):
        """PPM can be fractional when count is small relative to denom."""
        term_year = defaultdict(lambda: defaultdict(Counter))
        term_year[0]["beta"] = Counter({2020: 1})
        denoms = {0: Counter({2020: 3})}

        stub = _make_temporal_stub(term_year, denoms)
        top_df = pd.DataFrame({
            "cluster_id": [0], "term": ["beta"],
            "score": [1.0], "frequency": [1],
        })
        result = stub._build_time_series_metrics(top_df, term_year)
        ppm = result["ppm_series"].iloc[0]
        assert ppm[2020] == pytest.approx(1e6 / 3)

    def test_ppm_nan_when_zero_denom_per_year(self):
        """When cluster denominator is 0 for a specific year, PPM is NaN."""
        term_year = defaultdict(lambda: defaultdict(Counter))
        term_year[0]["alpha"] = Counter({2020: 5, 2021: 10})
        # denom exists for 2020 but is 0 for 2021
        denoms = {0: Counter({2020: 100})}

        stub = _make_temporal_stub(term_year, denoms)
        top_df = pd.DataFrame({
            "cluster_id": [0], "term": ["alpha"],
            "score": [1.0], "frequency": [15],
        })
        result = stub._build_time_series_metrics(top_df, term_year)
        ppm = result["ppm_series"].iloc[0]
        assert ppm[2020] == pytest.approx(1e6 * 5 / 100)
        assert math.isnan(ppm[2021])


# ---------------------------------------------------------------------------
# Log-lift (smoothed log-odds ratio): exact numeric verification
# ---------------------------------------------------------------------------

class TestLogLiftCalculation:
    """Verify loglift = log((count+alpha)/(denom_c+alpha)) - log((global+alpha)/(global_denom+alpha))."""

    def test_loglift_exact_values(self):
        """Compute loglift for known inputs and compare to hand-calculated result."""
        import numpy as _np

        alpha = 0.5
        # Single cluster: cluster counts = global counts
        term_year = defaultdict(lambda: defaultdict(Counter))
        term_year[0]["alpha"] = Counter({2020: 40})
        denoms = {0: Counter({2020: 200})}

        stub = _make_temporal_stub(term_year, denoms)
        top_df = pd.DataFrame({
            "cluster_id": [0], "term": ["alpha"],
            "score": [1.0], "frequency": [40],
        })
        result = stub._build_time_series_metrics(top_df, term_year)
        loglift = result["loglift_series"].iloc[0]

        # Only one cluster, so global = cluster.
        # p_cluster = (40+0.5)/(200+0.5), p_global = same => loglift = 0
        p_cluster = (40 + alpha) / (200 + alpha)
        p_global = (40 + alpha) / (200 + alpha)
        expected = float(_np.log(p_cluster) - _np.log(p_global))
        assert loglift[2020] == pytest.approx(expected, abs=1e-10)

    def test_loglift_two_clusters_asymmetric(self):
        """When a term is more concentrated in cluster 0 than globally, loglift > 0 in cluster 0."""
        import numpy as _np

        alpha = 0.5
        term_year = defaultdict(lambda: defaultdict(Counter))
        term_year[0]["alpha"] = Counter({2020: 80})
        term_year[1]["alpha"] = Counter({2020: 10})
        denoms = {0: Counter({2020: 200}), 1: Counter({2020: 800})}

        stub = _make_temporal_stub(term_year, denoms)
        top_df = pd.DataFrame({
            "cluster_id": [0, 1],
            "term": ["alpha", "alpha"],
            "score": [1.0, 1.0],
            "frequency": [80, 10],
        })
        result = stub._build_time_series_metrics(top_df, term_year)

        # global: count=80+10=90, denom=200+800=1000
        p_c0 = (80 + alpha) / (200 + alpha)
        p_c1 = (10 + alpha) / (800 + alpha)
        p_global = (90 + alpha) / (1000 + alpha)

        ll_c0 = result["loglift_series"].iloc[0][2020]
        ll_c1 = result["loglift_series"].iloc[1][2020]

        assert ll_c0 == pytest.approx(float(_np.log(p_c0) - _np.log(p_global)))
        assert ll_c1 == pytest.approx(float(_np.log(p_c1) - _np.log(p_global)))
        assert ll_c0 > 0  # concentrated
        assert ll_c1 < 0  # dilute

    def test_loglift_nan_when_denom_zero(self):
        """Loglift should be NaN when cluster denominator is 0."""
        term_year = defaultdict(lambda: defaultdict(Counter))
        term_year[0]["alpha"] = Counter({2020: 5})
        denoms = {0: Counter()}  # no denom

        stub = _make_temporal_stub(term_year, denoms)
        top_df = pd.DataFrame({
            "cluster_id": [0], "term": ["alpha"],
            "score": [1.0], "frequency": [5],
        })
        result = stub._build_time_series_metrics(top_df, term_year)
        ll = result["loglift_series"].iloc[0]
        assert math.isnan(ll[2020])


# ---------------------------------------------------------------------------
# Bayesian log-odds: exact numeric verification
# ---------------------------------------------------------------------------

class TestBayesianLogOdds:
    """Verify bayesian_log_odds_series against the formula in temporal.py."""

    def test_bayesian_logodds_exact(self):
        """Compare bayesian log-odds to hand-calculated result for known inputs."""
        import numpy as _np

        alpha = 0.5
        prior = 0.5

        term_year = defaultdict(lambda: defaultdict(Counter))
        term_year[0]["alpha"] = Counter({2020: 30})
        term_year[1]["alpha"] = Counter({2020: 10})
        denoms = {0: Counter({2020: 100}), 1: Counter({2020: 400})}

        stub = _make_temporal_stub(term_year, denoms)
        top_df = pd.DataFrame({
            "cluster_id": [0], "term": ["alpha"],
            "score": [1.0], "frequency": [30],
        })
        result = stub._build_time_series_metrics(top_df, term_year)
        bayes = result["bayesian_log_odds_series"].iloc[0]

        # global: count=30+10=40, denom=100+400=500
        global_count = 40
        global_denom = 500

        p_year = (global_count + alpha) / (global_denom + alpha)
        theta_cluster = (30 + prior * p_year) / (100 + prior)
        theta_global = (global_count + prior * p_year) / (global_denom + prior)
        theta_cluster = float(_np.clip(theta_cluster, 1e-9, 1 - 1e-9))
        theta_global = float(_np.clip(theta_global, 1e-9, 1 - 1e-9))
        expected = float(
            (_np.log(theta_cluster) - _np.log(1 - theta_cluster))
            - (_np.log(theta_global) - _np.log(1 - theta_global))
        )
        assert bayes[2020] == pytest.approx(expected)

    def test_bayesian_logodds_nan_when_denom_zero(self):
        """Bayesian log-odds is NaN when cluster denominator is 0."""
        term_year = defaultdict(lambda: defaultdict(Counter))
        term_year[0]["alpha"] = Counter({2020: 5})
        denoms = {0: Counter()}

        stub = _make_temporal_stub(term_year, denoms)
        top_df = pd.DataFrame({
            "cluster_id": [0], "term": ["alpha"],
            "score": [1.0], "frequency": [5],
        })
        result = stub._build_time_series_metrics(top_df, term_year)
        bayes = result["bayesian_log_odds_series"].iloc[0]
        assert math.isnan(bayes[2020])

    def test_bayesian_logodds_single_cluster_near_zero(self):
        """With one cluster, bayesian log-odds should be near 0 (cluster ~ global)."""
        term_year = defaultdict(lambda: defaultdict(Counter))
        term_year[0]["alpha"] = Counter({2020: 50})
        denoms = {0: Counter({2020: 1000})}

        stub = _make_temporal_stub(term_year, denoms)
        top_df = pd.DataFrame({
            "cluster_id": [0], "term": ["alpha"],
            "score": [1.0], "frequency": [50],
        })
        result = stub._build_time_series_metrics(top_df, term_year)
        bayes = result["bayesian_log_odds_series"].iloc[0]
        # Single cluster: cluster == global, so bayesian logodds ~ 0
        assert bayes[2020] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestTemporalEdgeCases:
    """Edge cases for _build_time_series_metrics."""

    def test_term_in_only_one_year(self):
        """A term appearing in exactly 1 year produces single-entry series."""
        term_year = defaultdict(lambda: defaultdict(Counter))
        term_year[0]["alpha"] = Counter({2022: 15})
        denoms = {0: Counter({2022: 300})}

        stub = _make_temporal_stub(term_year, denoms)
        top_df = pd.DataFrame({
            "cluster_id": [0], "term": ["alpha"],
            "score": [1.0], "frequency": [15],
        })
        result = stub._build_time_series_metrics(top_df, term_year)

        ppm = result["ppm_series"].iloc[0]
        ll = result["loglift_series"].iloc[0]
        bayes = result["bayesian_log_odds_series"].iloc[0]

        assert len(ppm) == 1
        assert 2022 in ppm
        assert ppm[2022] == pytest.approx(1e6 * 15 / 300)
        # Single cluster + single year: loglift ~ 0
        assert ll[2022] == pytest.approx(0.0, abs=1e-10)
        assert len(bayes) == 1

    def test_term_with_zero_frequency_everywhere(self):
        """A term with zero count in every year gets empty series (zeros are filtered)."""
        term_year = defaultdict(lambda: defaultdict(Counter))
        term_year[0]["alpha"] = Counter({2020: 0, 2021: 0})
        denoms = {0: Counter({2020: 100, 2021: 200})}

        stub = _make_temporal_stub(term_year, denoms)
        top_df = pd.DataFrame({
            "cluster_id": [0], "term": ["alpha"],
            "score": [1.0], "frequency": [0],
        })
        result = stub._build_time_series_metrics(top_df, term_year)
        ppm = result["ppm_series"].iloc[0]
        assert ppm == {}

    def test_large_year_range_2000_to_2025(self):
        """Term spanning 26 years produces correct series length and no errors."""
        years = list(range(2000, 2026))
        counts = {y: y - 1999 for y in years}  # 1, 2, ..., 26
        denom_counts = {y: 1000 for y in years}

        term_year = defaultdict(lambda: defaultdict(Counter))
        term_year[0]["trend"] = Counter(counts)
        denoms = {0: Counter(denom_counts)}

        stub = _make_temporal_stub(term_year, denoms)
        top_df = pd.DataFrame({
            "cluster_id": [0], "term": ["trend"],
            "score": [1.0], "frequency": [sum(counts.values())],
        })
        result = stub._build_time_series_metrics(top_df, term_year)
        ppm = result["ppm_series"].iloc[0]

        assert len(ppm) == 26
        assert ppm[2000] == pytest.approx(1e6 * 1 / 1000)
        assert ppm[2025] == pytest.approx(1e6 * 26 / 1000)
        # Series should be sorted by year
        assert list(ppm.keys()) == years

    def test_all_zero_counts_no_crash(self):
        """Counter with all-zero entries does not cause division errors."""
        term_year = defaultdict(lambda: defaultdict(Counter))
        term_year[0]["alpha"] = Counter({2020: 0})
        term_year[1]["beta"] = Counter({2020: 0})
        denoms = {0: Counter({2020: 0}), 1: Counter({2020: 0})}

        stub = _make_temporal_stub(term_year, denoms)
        top_df = pd.DataFrame({
            "cluster_id": [0, 1],
            "term": ["alpha", "beta"],
            "score": [1.0, 1.0],
            "frequency": [0, 0],
        })
        # Should not raise any exception
        result = stub._build_time_series_metrics(top_df, term_year)
        assert len(result) == 2
        # All series should be empty (zero counts filtered out)
        for idx in range(2):
            assert result["ppm_series"].iloc[idx] == {}

    def test_multiple_terms_same_cluster(self):
        """Multiple terms in the same cluster get independent series."""
        term_year = defaultdict(lambda: defaultdict(Counter))
        term_year[0]["alpha"] = Counter({2020: 10, 2021: 20})
        term_year[0]["beta"] = Counter({2020: 5})
        denoms = {0: Counter({2020: 100, 2021: 200})}

        stub = _make_temporal_stub(term_year, denoms)
        top_df = pd.DataFrame({
            "cluster_id": [0, 0],
            "term": ["alpha", "beta"],
            "score": [1.0, 0.5],
            "frequency": [30, 5],
        })
        result = stub._build_time_series_metrics(top_df, term_year)

        ppm_alpha = result["ppm_series"].iloc[0]
        ppm_beta = result["ppm_series"].iloc[1]

        assert len(ppm_alpha) == 2
        assert len(ppm_beta) == 1
        assert ppm_alpha[2020] == pytest.approx(1e6 * 10 / 100)
        assert ppm_alpha[2021] == pytest.approx(1e6 * 20 / 200)
        assert ppm_beta[2020] == pytest.approx(1e6 * 5 / 100)

    def test_year_denominators_output_matches_years(self):
        """year_denominators column has the same year keys as pub_year_series."""
        term_year = defaultdict(lambda: defaultdict(Counter))
        term_year[0]["alpha"] = Counter({2019: 3, 2021: 7})
        denoms = {0: Counter({2019: 50, 2020: 60, 2021: 70})}

        stub = _make_temporal_stub(term_year, denoms)
        top_df = pd.DataFrame({
            "cluster_id": [0], "term": ["alpha"],
            "score": [1.0], "frequency": [10],
        })
        result = stub._build_time_series_metrics(top_df, term_year)
        yr_denoms = result["year_denominators"].iloc[0]
        pub_years = result["pub_year_series"].iloc[0]

        # year_denominators should have exactly the same year keys as pub_year_series
        assert set(yr_denoms.keys()) == set(pub_years.keys())
        assert yr_denoms[2019] == 50
        assert yr_denoms[2021] == 70


# ---------------------------------------------------------------------------
# Integration: missing pubyear column
# ---------------------------------------------------------------------------

class TestMissingYearColumn:
    def test_missing_pubyear_still_runs(self, tmp_path):
        """When pubyear column is absent, the pipeline should handle it gracefully."""
        abstracts = pd.DataFrame({
            "uid": ["D0", "D1", "D2"],
            "title": ["neural network", "deep learning", "machine learning"],
            "abstract": [
                "neural network for classification",
                "deep learning for recognition",
                "machine learning for prediction",
            ],
            # no pubyear column
        })
        membership = pd.DataFrame({
            "uid": ["D0", "D1", "D2"],
            "cluster": [0, 0, 0],
        })
        abs_path = tmp_path / "no_year_abs.parquet"
        mem_path = tmp_path / "no_year_mem.parquet"
        abstracts.to_parquet(abs_path, index=False)
        membership.to_parquet(mem_path, index=False)
        cfg = _cfg(abs_path, mem_path)

        # The pipeline may raise an error or produce empty temporal columns.
        # Either outcome is acceptable; it must not crash with an unhandled exception.
        try:
            result = run_keyword_pipeline(cfg)
            # If it succeeds, temporal columns should be present (possibly empty series)
            assert "ppm_series" in result.columns
        except (KeyError, ValueError):
            # An explicit error about missing year column is also acceptable
            pass
