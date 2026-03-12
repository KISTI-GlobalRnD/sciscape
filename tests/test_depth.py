"""Tests for conceptual depth estimation (Stage 9)."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse as sp

from sciscape.keyword_extraction.depth import (
    DepthConfig,
    _compute_cross_cluster_counts,
    _compute_temporal_trend,
    _normalize_signal,
    estimate_depth,
)


def _make_df(rows, extra_cols=None):
    cols = ["cluster_id", "term", "score", "frequency", "doc_coverage"]
    df = pd.DataFrame(rows, columns=cols)
    if extra_cols:
        for k, v in extra_cols.items():
            df[k] = v
    return df


class TestNormalizeSignal:
    def test_basic(self):
        s = pd.Series([0, 5, 10])
        result = _normalize_signal(s)
        assert result.iloc[0] == 0.0
        assert result.iloc[2] == 1.0

    def test_constant(self):
        s = pd.Series([5, 5, 5])
        result = _normalize_signal(s)
        assert (result == 0.5).all()


class TestCrossClusterCounts:
    def test_multi_cluster_term(self):
        df = _make_df([
            (0, "neutron", 2.0, 200, 50),
            (1, "neutron", 1.5, 150, 40),
            (0, "proton", 1.0, 100, 30),
        ])
        counts = _compute_cross_cluster_counts(df)
        assert counts["neutron"] == 2
        assert counts["proton"] == 1


class TestTemporalTrend:
    def test_rising_trend(self):
        df = pd.DataFrame({
            "cluster_id": [0],
            "term": ["emerging"],
            "pub_year_series": [{2015: 5, 2016: 10, 2017: 20, 2018: 40, 2019: 80}],
        })
        trend = _compute_temporal_trend(df, recent_fraction=0.4)
        assert trend.iloc[0] > 1.0  # recent > early

    def test_declining_trend(self):
        df = pd.DataFrame({
            "cluster_id": [0],
            "term": ["declining"],
            "pub_year_series": [{2015: 80, 2016: 40, 2017: 20, 2018: 10, 2019: 5}],
        })
        trend = _compute_temporal_trend(df, recent_fraction=0.4)
        assert trend.iloc[0] < 1.0

    def test_no_year_data(self):
        df = pd.DataFrame({
            "cluster_id": [0],
            "term": ["nodata"],
            "pub_year_series": [{}],
        })
        trend = _compute_temporal_trend(df)
        assert trend.iloc[0] == 0.0

    def test_missing_column(self):
        df = pd.DataFrame({"cluster_id": [0], "term": ["x"]})
        trend = _compute_temporal_trend(df)
        assert trend.iloc[0] == 0.0


class TestEstimateDepth:
    def test_basic_depth_estimation(self):
        df = _make_df([
            (0, "radiation", 2.0, 500, 100),  # broad: high coverage
            (0, "nuclear energy", 1.5, 200, 50),  # medium
            (0, "cesium 137 contamination", 1.0, 30, 10),  # specific: low coverage, long
        ])
        config = DepthConfig(
            enabled=True,
            n_levels=3,
            weight_doc_coverage=0.5,
            weight_ngram_length=0.5,
            weight_cross_cluster=0.0,
            weight_cooc_asymmetry=0.0,
        )
        result = estimate_depth(df, config=config)

        assert "depth_score" in result.columns
        assert "depth_level" in result.columns
        # "radiation" should be shallower (lower depth) than "cesium 137 contamination"
        assert result.iloc[0]["depth_score"] < result.iloc[2]["depth_score"]

    def test_cross_cluster_signal(self):
        df = _make_df([
            (0, "model", 2.0, 300, 80),
            (1, "model", 1.8, 250, 70),
            (0, "lstm network", 1.5, 50, 15),
        ])
        config = DepthConfig(
            enabled=True,
            weight_doc_coverage=0.0,
            weight_cross_cluster=1.0,
            weight_ngram_length=0.0,
            weight_cooc_asymmetry=0.0,
        )
        result = estimate_depth(df, config=config)

        # "model" appears in 2 clusters → broader → lower depth
        # "lstm network" appears in 1 cluster → more specific → higher depth
        model_rows = result[result["term"] == "model"]
        lstm_rows = result[result["term"] == "lstm network"]
        assert model_rows["depth_score"].iloc[0] < lstm_rows["depth_score"].iloc[0]

    def test_with_cooccurrence(self):
        terms = ["radiation", "nuclear", "cesium"]
        term_to_idx = {t: i for i, t in enumerate(terms)}

        # cesium co-occurs with radiation (asymmetric: radiation is broader)
        cooc = sp.csr_matrix(np.array([
            [0, 50, 30],  # radiation: high total
            [50, 0, 20],  # nuclear: medium
            [30, 20, 0],  # cesium: low total
        ], dtype=np.float64))

        df = _make_df([
            (0, "radiation", 2.0, 500, 100),
            (0, "nuclear", 1.5, 300, 60),
            (0, "cesium", 1.0, 50, 10),
        ])

        config = DepthConfig(
            enabled=True,
            weight_cooc_asymmetry=1.0,
            weight_doc_coverage=0.0,
            weight_cross_cluster=0.0,
            weight_ngram_length=0.0,
            asymmetry_threshold=0.0,
        )
        result = estimate_depth(df, cooc_matrix=cooc, selected_terms=terms, config=config)
        assert "depth_score" in result.columns

    def test_cross_cluster_count_in_output(self):
        df = _make_df([
            (0, "model", 2.0, 300, 80),
            (1, "model", 1.8, 250, 70),
            (0, "lstm", 1.5, 50, 15),
        ])
        result = estimate_depth(df, config=DepthConfig(enabled=True))
        assert "cross_cluster_count" in result.columns
        # "model" appears in 2 clusters
        model_ccc = result[result["term"] == "model"]["cross_cluster_count"].iloc[0]
        assert model_ccc == 2
        # "lstm" appears in 1 cluster
        lstm_ccc = result[result["term"] == "lstm"]["cross_cluster_count"].iloc[0]
        assert lstm_ccc == 1

    def test_empty_df(self):
        df = _make_df([])
        result = estimate_depth(df)
        assert result.empty
        assert "depth_score" in result.columns
        assert "depth_level" in result.columns
        assert "cross_cluster_count" in result.columns

    def test_n_levels(self):
        df = _make_df([
            (0, f"term_{i}", 1.0, 100 - i * 10, 50 - i * 5)
            for i in range(10)
        ])
        config = DepthConfig(enabled=True, n_levels=4)
        result = estimate_depth(df, config=config)

        # Should have levels 0-3
        assert result["depth_level"].min() >= 0
        assert result["depth_level"].max() <= 3

    def test_temporal_trend_signal(self):
        """Rising terms get higher depth score than declining ones."""
        df = _make_df([
            (0, "established", 2.0, 500, 100),
            (0, "emerging", 1.5, 200, 50),
        ])
        # established: mostly early publications
        df["pub_year_series"] = [
            {2015: 100, 2016: 90, 2017: 80, 2018: 70, 2019: 60},
            {2015: 10, 2016: 20, 2017: 40, 2018: 80, 2019: 150},
        ]
        config = DepthConfig(
            enabled=True,
            weight_doc_coverage=0.0,
            weight_cross_cluster=0.0,
            weight_ngram_length=0.0,
            weight_cooc_asymmetry=0.0,
            weight_temporal_trend=1.0,
        )
        result = estimate_depth(df, config=config)
        # "emerging" has rising trend → higher depth
        assert result.iloc[1]["depth_score"] > result.iloc[0]["depth_score"]

    def test_temporal_trend_no_data(self):
        """Without pub_year_series, temporal signal is skipped gracefully."""
        df = _make_df([
            (0, "alpha", 2.0, 500, 100),
            (0, "beta", 1.5, 200, 50),
        ])
        config = DepthConfig(
            enabled=True,
            weight_temporal_trend=1.0,
            weight_doc_coverage=0.0,
            weight_cross_cluster=0.0,
            weight_ngram_length=0.0,
            weight_cooc_asymmetry=0.0,
        )
        result = estimate_depth(df, config=config)
        assert "depth_score" in result.columns

    def test_no_doc_coverage_column(self):
        """Should work even without doc_coverage column."""
        df = pd.DataFrame({
            "cluster_id": [0, 0, 0],
            "term": ["alpha", "beta gamma", "delta epsilon zeta"],
            "score": [2.0, 1.5, 1.0],
            "frequency": [300, 150, 50],
        })
        config = DepthConfig(
            enabled=True,
            weight_doc_coverage=0.0,
            weight_ngram_length=1.0,
            weight_cross_cluster=0.0,
            weight_cooc_asymmetry=0.0,
        )
        result = estimate_depth(df, config=config)
        # Longer ngrams should get higher depth
        assert result.iloc[2]["depth_score"] > result.iloc[0]["depth_score"]
