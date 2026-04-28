from pathlib import Path

import pytest

from sciscape.clustering.crosscheck_report import (
    CrosscheckRun,
    parse_seeds,
    render_html_report,
    write_artifacts,
)
from sciscape.clustering.profile_report import ClusterSummary, ProfileCase


def _summary(n_clusters: int, singletons: int) -> ClusterSummary:
    return ClusterSummary(
        n_clusters=n_clusters,
        n_small=singletons,
        n_singletons=singletons,
        min_size=1,
        median_size=3.0,
        p90_size=10.0,
        p99_size=20.0,
        max_size=30,
    )


def test_parse_seeds():
    assert parse_seeds("0,1, 7") == [0, 1, 7]
    with pytest.raises(Exception, match="seeds"):
        parse_seeds("")


def test_render_and_write_crosscheck_report(tmp_path):
    run = CrosscheckRun(
        case=ProfileCase(
            name="toy",
            edge_path=Path("toy.parquet"),
            uid1_col="src",
            uid2_col="dst",
            weight_col="weight",
        ),
        seed=3,
        n_nodes=6,
        directed_edges=12,
        undirected_edges=6,
        compare_nodes=6,
        rust_sec=0.2,
        java_sec=1.0,
        rust_quality=10.5,
        java_quality=10.0,
        rust_clusters=_summary(2, 1),
        java_clusters=_summary(3, 2),
        nmi=0.8,
        ari=0.7,
    )

    html = render_html_report([run])
    assert "SciScape Leiden Java/Rust Cross-check" in html
    assert "toy" in html
    assert "5.00x" in html

    paths = write_artifacts([run], tmp_path)
    assert paths["report_html"].exists()
    assert paths["metrics_csv"].exists()
    assert paths["metrics_json"].exists()
