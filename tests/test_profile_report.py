from pathlib import Path

import numpy as np

from sciscape.clustering.profile_report import (
    ClusterSummary,
    PhaseMetric,
    ProfileCase,
    ProfileResult,
    render_html_report,
    summarize_membership,
    write_artifacts,
)


def test_summarize_membership_counts_small_and_singletons():
    summary = summarize_membership(np.array([0, 0, 1, 2, 2, 2]), 3, min_size=3)

    assert summary.n_clusters == 3
    assert summary.n_small == 2
    assert summary.n_singletons == 1
    assert summary.max_size == 3
    assert summary.size_bins["1"] == 1
    assert summary.size_bins["2-4"] == 2


def test_render_and_write_profile_report(tmp_path):
    result = ProfileResult(
        case=ProfileCase(
            name="toy",
            edge_path=Path("toy.parquet"),
            uid1_col="src",
            uid2_col="dst",
            weight_col="weight",
        ),
        n_nodes=6,
        directed_edges=12,
        undirected_edges=6,
        phases=[
            PhaseMetric("remap", 0.1, rss_mb=10.0, hwm_mb=12.0),
            PhaseMetric("graph_build", 0.2, rss_mb=11.0, hwm_mb=12.0),
            PhaseMetric("leiden", 0.3, rss_mb=11.0, hwm_mb=13.0),
            PhaseMetric("postprocess", 0.05, rss_mb=11.0, hwm_mb=13.0),
        ],
        raw_clusters=ClusterSummary(3, 2, 1, 1, 2.0, 3.0, 3.0, 3),
        post_clusters=ClusterSummary(1, 0, 0, 6, 6.0, 6.0, 6.0, 6),
        quality_raw=1.0,
        quality_post=1.2,
        changed_nodes=3,
        postprocess_rounds=[
            {
                "round": 0,
                "method": "leiden",
                "gamma": 0.01,
                "n_small_before": 2,
                "n_small_after": 0,
                "n_merged": 2,
                "n_total_clusters": 1,
            }
        ],
    )

    html = render_html_report([result])
    assert "SciScape Leiden Profile Report" in html
    assert "toy" in html
    assert "Cluster Size Distribution" in html

    paths = write_artifacts([result], tmp_path)
    assert paths["report_html"].exists()
    assert paths["metrics_csv"].exists()
    assert paths["rounds_csv"].exists()
    assert paths["metrics_json"].exists()
