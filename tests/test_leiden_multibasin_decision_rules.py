"""Tests for Dongdaemun greedy-failure decision rule diagnostics."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "research" / "consensus" / "scripts"
ANALYSIS_PATH = SCRIPT_DIR / "analyze_leiden_multibasin_decision_rules.py"


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


def test_decision_rules_identify_top3_guard():
    module = _load_script("leiden_multibasin_decision_rules_top3_for_test")
    candidates = pd.DataFrame(
        [
            _candidate(0, p1=30.0, p5=10.0),
            _candidate(1, p1=20.0, p5=11.0),
            _candidate(2, p1=10.0, p5=35.0),
        ]
    )

    decisions = module.build_case_decision_rows(
        candidates,
        acceptable_regret_q=1.0,
        material_regret_q=10.0,
    )

    row = decisions.iloc[0]
    assert row["decision_label"] == "top3_guard"
    assert row["best_p1_rank"] == 3
    assert row["min_k_to_acceptable_regret"] == 3
    assert bool(row["p1_top1_material_miss"]) is True
    assert bool(row["support_sketch_exact"]) is True


def test_decision_rules_do_not_promote_low_roi_gains():
    module = _load_script("leiden_multibasin_decision_rules_low_roi_for_test")
    candidates = pd.DataFrame(
        [
            _candidate(0, p1=10.0, p5=0.1, rel=0.5),
            _candidate(1, p1=9.0, p5=0.2, rel=0.8),
        ]
    )

    decisions = module.build_case_decision_rows(candidates)

    row = decisions.iloc[0]
    assert row["decision_label"] == "low_roi_skip_expansion"
    assert bool(row["best_gain_material"]) is False


def test_decision_rules_report_truncated_support_sketch():
    module = _load_script("leiden_multibasin_decision_rules_support_for_test")
    candidates = pd.DataFrame(
        [
            _candidate(0, p1=10.0, p5=2.0, support_count=5, support_sample=5),
            _candidate(1, p1=9.0, p5=3.0, support_count=10, support_sample=5),
        ]
    )

    decisions = module.build_case_decision_rows(candidates)

    assert bool(decisions.iloc[0]["support_sketch_exact"]) is False


def test_decision_rules_write_report(tmp_path):
    module = _load_script("leiden_multibasin_decision_rules_report_for_test")
    candidates = pd.DataFrame(
        [
            _candidate(0, p1=10.0, p5=20.0),
            _candidate(1, p1=9.0, p5=21.0),
        ]
    )
    decisions = module.build_case_decision_rows(candidates)
    summary = module.build_decision_summary(decisions)

    module.write_report(tmp_path, decisions, summary)

    report = (tmp_path / "dongdaemun_greedy_failure_decision_report.md").read_text(
        encoding="utf-8"
    )
    assert "Dongdaemun Greedy Failure Decision Review" in report
    assert "p1 sufficient cases" in report


def _candidate(
    candidate_index: int,
    *,
    p1: float,
    p5: float,
    rel: float = 100.0,
    support_count: int = 2,
    support_sample: int = 2,
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
        "p5_basin_signature": f"basin-{candidate_index}",
        "p5_changed_fraction_vs_baseline": 0.001 * (candidate_index + 1),
        "p5_changed_nodes_vs_baseline": candidate_index + 1,
        "p5_basin_sketch_node_hash": "sample-hash",
        "p5_basin_sketch_baseline_membership": "0;0;1;1",
        "p5_basin_sketch_membership": "0;0;1;1",
        "p5_basin_changed_support_node_count": support_count,
        "p5_basin_changed_support_sketch_sample_size": support_sample,
        "p5_basin_changed_support_nodes": "0;1",
    }
