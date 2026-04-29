import json
import csv

import numpy as np

from sciscape.clustering.adaptive_refinement import (
    summarize_cluster_graph_stats,
    write_adaptive_refinement_report,
)
from sciscape.clustering.leiden_rust import RustClusterGraphStats


def _stats() -> RustClusterGraphStats:
    return RustClusterGraphStats(
        block_count=np.array([2, 3, 1], dtype=np.uint64),
        doc_weight=np.array([20.0, 120.0, 400.0], dtype=np.float64),
        internal_weight=np.array([5.0, 20.0, 80.0], dtype=np.float64),
        external_weight=np.array([10.0, 15.0, 5.0], dtype=np.float64),
        degree=np.array([2, 2, 1], dtype=np.uint64),
        top_neighbor=np.array([1, 0, 1], dtype=np.int64),
        top_neighbor_weight=np.array([7.0, 7.0, 5.0], dtype=np.float64),
        conductance=np.array([0.5, 15.0 / 55.0, 5.0 / 165.0], dtype=np.float64),
        leafness=np.array([0.7, 7.0 / 15.0, 1.0], dtype=np.float64),
        band_distance=np.array([30.0, 0.0, 150.0], dtype=np.float64),
        candidate_source=np.array([0, 1], dtype=np.uint64),
        candidate_target=np.array([1, 2], dtype=np.uint64),
        candidate_edge_weight=np.array([7.0, 5.0], dtype=np.float64),
        candidate_delta_q=np.array([1.0, -2.0], dtype=np.float64),
        candidate_merged_weight=np.array([140.0, 520.0], dtype=np.float64),
        candidate_size_band_gain=np.array([30.0, -120.0], dtype=np.float64),
    )


def test_summarize_cluster_graph_stats():
    summary = summarize_cluster_graph_stats(_stats(), min_weight=50.0, max_weight=250.0)

    assert summary["n_active_clusters"] == 3
    assert summary["n_below_min_weight"] == 1
    assert summary["n_above_max_weight"] == 1
    assert summary["n_within_weight_band"] == 1
    assert summary["n_merge_candidates"] == 2
    assert summary["n_positive_delta_candidates"] == 1
    assert summary["n_positive_and_band_improving_candidates"] == 1


def test_write_adaptive_refinement_report(tmp_path):
    paths = write_adaptive_refinement_report(
        _stats(),
        tmp_path,
        min_weight=50.0,
        max_weight=250.0,
        top_candidates=1,
    )

    summary = json.loads((tmp_path / "cluster_graph_summary.json").read_text())
    assert summary["n_merge_candidates"] == 2
    assert set(paths) == {"summary", "merge_candidates", "cluster_arrays"}

    candidate_lines = (tmp_path / "macro_merge_candidates.csv").read_text().splitlines()
    assert len(candidate_lines) == 2
    row = next(csv.DictReader(candidate_lines))
    assert row["rank"] == "1"
    assert row["source"] == "0"
    assert row["target"] == "1"
    assert float(row["source_doc_weight"]) == 20.0
    assert float(row["target_doc_weight"]) == 120.0
    assert int(row["source_degree"]) == 2
    assert int(row["target_degree"]) == 2
    assert float(row["source_leafness"]) == 0.7

    arrays = np.load(tmp_path / "cluster_graph_stats.npz")
    np.testing.assert_array_equal(arrays["block_count"], np.array([2, 3, 1], dtype=np.uint64))
