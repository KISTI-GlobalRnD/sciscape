"""Tests for Dongdaemun quality-first candidate choice diagnostics."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "research" / "consensus" / "scripts"
ANALYSIS_PATH = SCRIPT_DIR / "analyze_leiden_quality_first_choice.py"


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


def test_quality_first_ledger_selects_best_endpoint_not_p1():
    module = _load_script("leiden_quality_first_choice_for_test")
    candidates = pd.DataFrame(
        [
            _candidate(0, p1=30.0, p5=10.0),
            _candidate(1, p1=20.0, p5=11.0),
            _candidate(2, p1=10.0, p5=35.0),
        ]
    )

    rows = module.build_quality_first_choice_rows(candidates, material_regret_q=10.0)

    row = rows.iloc[0]
    assert row["selection_principle"] == "evaluate_candidate_budget_then_choose_max_p5_delta_q"
    assert row["quality_first_frame"] == "delayed_best_shallow"
    assert row["p1_candidate_index"] == 0
    assert row["quality_first_candidate_index"] == 2
    assert row["quality_first_p1_rank"] == 3
    assert row["quality_first_premium_over_p1_q"] == 25.0
    assert bool(row["quality_first_material_premium"]) is True


def test_quality_frontier_marks_materially_short_prefixes():
    module = _load_script("leiden_quality_first_frontier_for_test")
    candidates = pd.DataFrame(
        [
            _candidate(0, p1=30.0, p5=10.0),
            _candidate(1, p1=20.0, p5=11.0),
            _candidate(2, p1=10.0, p5=35.0),
        ]
    )

    frontier = module.build_quality_frontier_rows(
        candidates,
        acceptable_regret_q=1.0,
        material_regret_q=10.0,
    )

    top1 = frontier[frontier["top_k"] == 1].iloc[0]
    top3 = frontier[frontier["top_k"] == 3].iloc[0]
    assert top1["frontier_state"] == "materially_short"
    assert bool(top1["quality_first_hit"]) is False
    assert top3["frontier_state"] == "oracle_recovered"
    assert bool(top3["quality_first_hit"]) is True


def test_quality_first_summary_counts_premium_frames():
    module = _load_script("leiden_quality_first_summary_for_test")
    candidates = pd.DataFrame(
        [
            _candidate(0, p1=30.0, p5=10.0),
            _candidate(1, p1=20.0, p5=35.0),
        ]
    )
    rows = module.build_quality_first_choice_rows(candidates, material_regret_q=10.0)

    summary = module.build_quality_first_summary(rows)

    all_row = summary[summary["group"] == "all"].iloc[0]
    assert all_row["case_count"] == 1
    assert all_row["delayed_best_shallow_count"] == 1
    assert all_row["material_premium_count"] == 1
    assert all_row["premium_q_sum"] == 25.0


def test_quality_first_report_is_written(tmp_path):
    module = _load_script("leiden_quality_first_report_for_test")
    candidates = pd.DataFrame(
        [
            _candidate(0, p1=10.0, p5=20.0),
            _candidate(1, p1=9.0, p5=21.0),
        ]
    )
    choice = module.build_quality_first_choice_rows(candidates)
    frontier = module.build_quality_frontier_rows(candidates)
    summary = module.build_quality_first_summary(choice)

    module.write_report(tmp_path, choice, frontier, summary)

    report = (tmp_path / "dongdaemun_quality_first_choice_report.md").read_text(
        encoding="utf-8"
    )
    assert "Dongdaemun Quality-First Choice Review" in report
    assert "quality-first endpoint" in report


def _candidate(
    candidate_index: int,
    *,
    p1: float,
    p5: float,
    rel: float = 100.0,
) -> dict[str, object]:
    return {
        "candidate_eval_mode": "multifidelity_label",
        "case": "adaptive_refinement_field12_gcc_emb_full_knn30_bc_cosine",
        "seed": 11,
        "candidate_budget": 3,
        "max_group_candidates": 3,
        "candidate_index": candidate_index,
        "p1_delta_q": p1,
        "p5_delta_q": p5,
        "p5_relative_delta_q_ppm": rel,
        "p5_elapsed_ms": 10.0 + candidate_index,
        "p5_basin_signature": f"basin-{candidate_index}",
        "p5_changed_fraction_vs_baseline": 0.001 * (candidate_index + 1),
        "p5_changed_nodes_vs_baseline": candidate_index + 1,
        "p5_basin_sketch_node_hash": "sample-hash",
        "p5_basin_sketch_baseline_membership": "0;0;1;1",
        "p5_basin_sketch_membership": "0;0;1;1",
        "p5_basin_changed_support_node_count": 2,
        "p5_basin_changed_support_sketch_sample_size": 2,
        "p5_basin_changed_support_nodes": "0;1",
    }
