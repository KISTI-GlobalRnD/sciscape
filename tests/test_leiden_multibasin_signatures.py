"""Tests for Leiden p5 basin signature analysis."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "research" / "consensus" / "scripts"
ANALYSIS_PATH = SCRIPT_DIR / "analyze_leiden_multibasin_signatures.py"


def _load_script(module_name: str):
    if str(ANALYSIS_PATH.parent) not in sys.path:
        sys.path.insert(0, str(ANALYSIS_PATH.parent))
    spec = importlib.util.spec_from_file_location(module_name, ANALYSIS_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_multibasin_signature_analysis_summarizes_material_basin_coverage():
    module = _load_script("leiden_multibasin_signatures_for_test")
    candidates = pd.DataFrame(
        [
            _candidate(0, signature="basin-a", p1=4.0, p5=1.0, rel=20.0),
            _candidate(1, signature="basin-b", p1=3.0, p5=5.0, rel=50.0),
            _candidate(2, signature="basin-a", p1=2.0, p5=2.0, rel=20.0),
            _candidate(3, signature="basin-c", p1=1.0, p5=0.1, rel=0.5),
        ]
    )

    basin_rows, summary, coverage = module.analyze_signatures(
        candidates,
        material_delta_q=1.0,
        material_relative_ppm=10.0,
    )

    assert len(basin_rows) == 3
    row = summary.iloc[0]
    assert row["dongdaemun_family"] == "diagnostic"
    assert row["distinct_basin_count"] == 3
    assert row["distinct_material_basin_count"] == 2
    assert row["distinct_meso_or_macro_material_basin_count"] == 2
    assert row["distinct_macro_material_basin_count"] == 0
    assert row["full_p5_best_candidate_index"] == 1
    assert row["best_basin_signature"] == "basin-b"
    assert bool(row["best_basin_is_material"]) is True
    assert row["low_roi_positive_count"] == 1
    assert row["mean_alignment_error_fraction_vs_baseline"] == pytest.approx(0.0025)
    assert row["max_alignment_error_nodes_vs_baseline"] == 4
    assert row["mean_aligned_changed_support_node_count"] == pytest.approx(2.5)

    top1 = coverage[coverage["top_k"] == 1].iloc[0]
    top2 = coverage[coverage["top_k"] == 2].iloc[0]
    assert bool(top1["best_basin_hit_at_k"]) is False
    assert top1["best_quality_regret_at_k"] == 4.0
    assert top1["distinct_basin_coverage_at_k"] == 1 / 3
    assert top1["distinct_material_basin_coverage_at_k"] == 0.5
    assert bool(top2["best_basin_hit_at_k"]) is True
    assert top2["best_quality_regret_at_k"] == 0.0
    assert set(basin_rows["basin_scale_tier"]) == {"meso"}
    assert "mean_alignment_error_fraction_vs_baseline" in basin_rows.columns
    assert "max_alignment_error_nodes_vs_baseline" in basin_rows.columns


def test_multibasin_signature_analysis_writes_expected_artifacts(tmp_path):
    module = _load_script("leiden_multibasin_signatures_artifacts_for_test")
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    pd.DataFrame(
        [
            _candidate(0, signature="basin-a", p1=1.0, p5=1.5, rel=15.0),
            _candidate(1, signature="basin-b", p1=0.5, p5=0.2, rel=2.0),
        ]
    ).to_csv(input_dir / "candidate_level_rows.csv", index=False)

    candidates = module._read_csvs(input_dir, "candidate_level_rows.csv")
    basin_rows, summary, coverage = module.analyze_signatures(candidates)
    output_dir.mkdir()
    basin_rows.to_csv(output_dir / "leiden_multibasin_basin_rows.csv", index=False)
    summary.to_csv(output_dir / "leiden_multibasin_basin_summary.csv", index=False)
    coverage.to_csv(output_dir / "leiden_multibasin_coverage_curves.csv", index=False)
    module.write_report(output_dir, summary, coverage)

    assert (output_dir / "leiden_multibasin_basin_rows.csv").exists()
    assert (output_dir / "leiden_multibasin_basin_summary.csv").exists()
    assert (output_dir / "leiden_multibasin_coverage_curves.csv").exists()
    report = (output_dir / "leiden_multibasin_signature_report.md").read_text(
        encoding="utf-8"
    )
    assert "Dongdaemun diagnostic artifact" in report


def test_multibasin_signature_analysis_builds_pairwise_coarse_matrix():
    module = _load_script("leiden_multibasin_pairwise_for_test")
    candidates = pd.DataFrame(
        [
            _candidate(
                0,
                signature="basin-a",
                p1=3.0,
                p5=10.0,
                rel=100.0,
                sketch_membership="0;0;1;1",
                changed_support_nodes="0;1",
            ),
            _candidate(
                1,
                signature="basin-b",
                p1=2.0,
                p5=10.5,
                rel=100.5,
                sketch_membership="0;1;1;1",
                changed_support_nodes="1;2",
            ),
            _candidate(
                2,
                signature="basin-c",
                p1=1.0,
                p5=30.0,
                rel=300.0,
                sketch_membership="0;1;0;1",
                changed_support_nodes="2;3",
            ),
        ]
    )
    signature_rows = module._mark_material_gain(
        module._signature_frame(candidates),
        material_delta_q=1.0,
        material_relative_ppm=10.0,
    )

    pairwise = module.build_pairwise_basin_matrix(
        signature_rows,
        coarse_endpoint_tau=0.1,
        coarse_support_tau=0.5,
        iso_q_delta=1.0,
        iso_q_relative_ppm=1.0,
    )
    coarse = module.build_coarse_basin_rows(signature_rows, pairwise)

    assert len(pairwise) == 3
    first = pairwise[
        (pairwise["left_candidate_index"] == 0)
        & (pairwise["right_candidate_index"] == 1)
    ].iloc[0]
    assert first["sample_coassignment_distance"] == 0.5
    assert first["changed_node_support_jaccard_distance"] == pytest.approx(2 / 3)
    assert first["coarse_support_distance_source"] == "changed_node_support"
    assert bool(first["same_coarse_basin"]) is False
    assert bool(first["iso_q_pair"]) is True
    assert bool(first["partition_distinct_iso_q_pair"]) is True
    assert len(coarse) == 3


def _candidate(
    candidate_index: int,
    *,
    signature: str,
    p1: float,
    p5: float,
    rel: float,
    sketch_membership: str = "",
    changed_support_nodes: str = "",
) -> dict[str, object]:
    return {
        "candidate_eval_mode": "multifidelity_label",
        "case": "case",
        "seed": 11,
        "candidate_budget": 4,
        "max_group_candidates": 4,
        "candidate_index": candidate_index,
        "p1_delta_q": p1,
        "p5_delta_q": p5,
        "p5_relative_delta_q_ppm": rel,
        "p5_basin_signature": signature,
        "p5_changed_fraction_vs_baseline": 0.001 * (candidate_index + 1),
        "p5_changed_nodes_vs_baseline": candidate_index + 1,
        "p5_alignment_error_fraction_vs_baseline": 0.001 * (candidate_index + 1),
        "p5_alignment_error_nodes_vs_baseline": candidate_index + 1,
        "p5_aligned_changed_support_node_count": candidate_index + 1,
        "p5_basin_sketch_node_hash": "sample-hash",
        "p5_basin_sketch_baseline_membership": "0;0;1;1" if sketch_membership else "",
        "p5_basin_sketch_membership": sketch_membership,
        "p5_basin_changed_support_nodes": changed_support_nodes,
        "p5_aligned_changed_support_nodes": changed_support_nodes,
    }
