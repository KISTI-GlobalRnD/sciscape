"""Tests for Dongdaemun better/similar basin portfolio evidence."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "research" / "consensus" / "scripts"
ANALYSIS_PATH = Path(__file__).resolve().parents[1] / "research/consensus/scripts/leiden_basin/evidence_panels/portfolio_evidence/analyze_leiden_basin_portfolio_evidence.py"


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


def test_basin_portfolio_rows_capture_better_and_similar_basin_need():
    module = _load_script("leiden_basin_portfolio_for_test")
    candidates = pd.DataFrame(
        [
            _candidate(0, p1=40.0, p5=10.0, support_nodes="0;1"),
            _candidate(1, p1=30.0, p5=11.0, support_nodes="0;2"),
            _candidate(2, p1=20.0, p5=35.0, support_nodes="0;1"),
            _candidate(3, p1=10.0, p5=34.5, support_nodes="2;3"),
        ]
    )

    rows = module.build_basin_portfolio_rows(
        candidates,
        material_regret_q=10.0,
        near_best_delta_q=1.0,
        support_distinct_tau=0.5,
    )

    row = rows.iloc[0]
    assert row["portfolio_need_label"] == "search_better_and_review_similar"
    assert row["quality_first_candidate_index"] == 2
    assert row["quality_first_p1_rank"] == 3
    assert row["better_basin_premium_q"] == 25.0
    assert bool(row["better_basin_material_premium"]) is True
    assert row["near_best_candidate_count"] == 2
    assert row["support_distinct_iso_q_pair_count"] >= 1


def test_lookalike_pair_rows_keep_similar_q_distinct_support_pairs():
    module = _load_script("leiden_basin_portfolio_pairs_for_test")
    candidates = pd.DataFrame(
        [
            _candidate(0, p1=40.0, p5=10.0, support_nodes="0;1"),
            _candidate(1, p1=30.0, p5=35.0, support_nodes="0;1"),
            _candidate(2, p1=20.0, p5=35.5, support_nodes="2;3"),
        ]
    )

    pairs = module.build_lookalike_pair_rows(
        candidates,
        support_distinct_tau=0.5,
        iso_q_delta=1.0,
        iso_q_relative_ppm=10.0,
    )

    assert len(pairs) == 1
    pair = pairs.iloc[0]
    assert pair["left_candidate_index"] == 1
    assert pair["right_candidate_index"] == 2
    assert pair["q_delta_abs"] == 0.5
    assert pair["coarse_support_distance"] == 1.0
    assert pair["lookalike_reason"] == "similar QF but partition/support-distinct"


def test_basin_portfolio_summary_and_report(tmp_path):
    module = _load_script("leiden_basin_portfolio_report_for_test")
    candidates = pd.DataFrame(
        [
            _candidate(0, p1=40.0, p5=10.0, support_nodes="0;1"),
            _candidate(1, p1=30.0, p5=35.0, support_nodes="0;1"),
            _candidate(2, p1=20.0, p5=35.5, support_nodes="2;3"),
        ]
    )
    rows = module.build_basin_portfolio_rows(candidates, material_regret_q=10.0)
    pairs = module.build_lookalike_pair_rows(candidates)
    summary = module.build_portfolio_summary(rows)

    all_row = summary[summary["group"] == "all"].iloc[0]
    assert all_row["case_count"] == 1
    assert all_row["material_better_basin_count"] == 1
    assert all_row["cases_with_support_distinct_iso_q_pairs"] == 1

    module.write_report(tmp_path, rows, pairs, summary)
    report = (tmp_path / "dongdaemun_basin_portfolio_evidence_report.md").read_text(
        encoding="utf-8"
    )
    assert "Dongdaemun Basin Portfolio Evidence" in report
    assert "Better-basin search is justified" in report


def _candidate(
    candidate_index: int,
    *,
    p1: float,
    p5: float,
    support_nodes: str,
    rel: float = 100.0,
) -> dict[str, object]:
    return {
        "candidate_eval_mode": "multifidelity_label",
        "case": "adaptive_refinement_field12_gcc_emb_full_knn30_bc_cosine",
        "seed": 11,
        "candidate_budget": 4,
        "max_group_candidates": 4,
        "candidate_index": candidate_index,
        "p1_delta_q": p1,
        "p5_delta_q": p5,
        "p5_relative_delta_q_ppm": rel + candidate_index,
        "p5_elapsed_ms": 10.0 + candidate_index,
        "p5_basin_signature": f"basin-{candidate_index}",
        "p5_changed_fraction_vs_baseline": 0.001 * (candidate_index + 1),
        "p5_changed_nodes_vs_baseline": candidate_index + 1,
        "p5_basin_sketch_node_hash": "sample-hash",
        "p5_basin_sketch_baseline_membership": "0;0;1;1",
        "p5_basin_sketch_membership": "0;0;1;1",
        "p5_basin_changed_support_node_count": 2,
        "p5_basin_changed_support_sketch_sample_size": 2,
        "p5_basin_changed_support_nodes": support_nodes,
    }
