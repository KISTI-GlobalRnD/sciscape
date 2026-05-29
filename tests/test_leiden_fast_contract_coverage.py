"""Tests for Dongdaemun fast-prefix contract coverage diagnostics."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "research" / "consensus" / "scripts"
ANALYSIS_PATH = Path(__file__).resolve().parents[1] / "research/consensus/scripts/leiden_basin/basin_signatures/portfolio_contracts/analyze_leiden_fast_contract_coverage.py"


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


def test_fast_contract_coverage_requires_endpoint_and_portfolio_pairs():
    module = _load_script("leiden_fast_contract_coverage_for_test")
    candidates = pd.DataFrame(
        [
            _candidate(0, p1=40.0, p5=10.0, support_nodes="0;1"),
            _candidate(1, p1=30.0, p5=35.0, support_nodes="0;1"),
            _candidate(2, p1=20.0, p5=35.5, support_nodes="2;3"),
        ]
    )

    rows = module.build_fast_contract_coverage_rows(
        candidates,
        material_regret_q=10.0,
        near_best_delta_q=1.0,
        support_distinct_tau=0.5,
    )

    fast_p1 = rows[rows["mode_name"] == "fast_p1"].iloc[0]
    assert bool(fast_p1["endpoint_obligation_covered"]) is False
    assert bool(fast_p1["contract_fully_covered"]) is False
    assert fast_p1["coverage_class"] == "quality_and_portfolio_missed"
    assert fast_p1["quality_regret_q"] == 25.5
    assert fast_p1["near_qf_candidate_missed_count"] == 1
    assert fast_p1["support_distinct_iso_q_pair_missed_count"] == 1

    fast_top3 = rows[rows["mode_name"] == "fast_top3"].iloc[0]
    assert bool(fast_top3["endpoint_obligation_covered"]) is True
    assert bool(fast_top3["contract_fully_covered"]) is True
    assert fast_top3["support_distinct_iso_q_pair_covered_count"] == 1
    assert fast_top3["coverage_class"] == "contract_covered"


def test_fast_contract_summary_counts_missed_obligations():
    module = _load_script("leiden_fast_contract_summary_for_test")
    candidates = pd.DataFrame(
        [
            _candidate(0, p1=40.0, p5=10.0, support_nodes="0;1"),
            _candidate(1, p1=30.0, p5=35.0, support_nodes="0;1"),
            _candidate(2, p1=20.0, p5=35.5, support_nodes="2;3"),
        ]
    )
    rows = module.build_fast_contract_coverage_rows(
        candidates,
        material_regret_q=10.0,
        near_best_delta_q=1.0,
        support_distinct_tau=0.5,
    )

    summary = module.build_fast_contract_summary(rows)

    p1 = summary[summary["mode_name"] == "fast_p1"].iloc[0]
    top3 = summary[summary["mode_name"] == "fast_top3"].iloc[0]
    assert p1["case_count"] == 1
    assert p1["contract_fully_covered_count"] == 0
    assert p1["material_regret_count"] == 1
    assert p1["quality_regret_q_sum"] == 25.5
    assert p1["support_distinct_iso_q_pair_missed_sum"] == 1
    assert top3["contract_fully_covered_count"] == 1
    assert top3["support_distinct_iso_q_pair_missed_sum"] == 0


def test_fast_contract_report_is_written(tmp_path):
    module = _load_script("leiden_fast_contract_report_for_test")
    candidates = pd.DataFrame(
        [
            _candidate(0, p1=40.0, p5=10.0, support_nodes="0;1"),
            _candidate(1, p1=30.0, p5=35.0, support_nodes="0;1"),
            _candidate(2, p1=20.0, p5=35.5, support_nodes="2;3"),
        ]
    )
    rows = module.build_fast_contract_coverage_rows(
        candidates,
        material_regret_q=10.0,
        near_best_delta_q=1.0,
        support_distinct_tau=0.5,
    )
    summary = module.build_fast_contract_summary(rows)

    module.write_report(tmp_path, rows, summary)

    report = (tmp_path / "dongdaemun_fast_contract_coverage_report.md").read_text(
        encoding="utf-8"
    )
    assert "Dongdaemun Fast Contract Coverage" in report
    assert "Fast mode should be evaluated against explicit output obligations" in report
    assert "fast_p1: full contract coverage 0/1" in report


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
        "candidate_budget": 3,
        "max_group_candidates": 3,
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
