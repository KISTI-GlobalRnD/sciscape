"""Tests for the Dongdaemun accurate-mode basin portfolio contract."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "research" / "consensus" / "scripts"
ANALYSIS_PATH = Path(__file__).resolve().parents[1] / "research/consensus/scripts/leiden_basin/basin_signatures/portfolio_contracts/analyze_leiden_accurate_basin_portfolio_contract.py"


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


def test_accurate_contract_requires_portfolio_for_quality_and_support_risk():
    module = _load_script("leiden_accurate_contract_for_test")
    candidates = pd.DataFrame(
        [
            _candidate(0, p1=40.0, p5=10.0, support_nodes="0;1"),
            _candidate(1, p1=30.0, p5=35.0, support_nodes="0;1"),
            _candidate(2, p1=20.0, p5=35.5, support_nodes="2;3"),
        ]
    )

    rows = module.build_accurate_contract_rows(
        candidates,
        material_regret_q=10.0,
        near_best_delta_q=1.0,
        support_distinct_tau=0.5,
        iso_q_delta=1.0,
        iso_q_relative_ppm=10.0,
    )

    row = rows.iloc[0]
    assert row["contract_version"] == module.CONTRACT_VERSION
    assert (
        row["accurate_mode_output_contract"]
        == "best_plus_near_qf_support_distinct_portfolio"
    )
    assert row["winner_only_risk"] == "high_quality_and_interpretation_risk"
    assert row["selection_principle"] == "choose_max_p5_then_attach_basin_portfolio"
    assert row["quality_first_candidate_index"] == 2
    assert row["quality_first_p1_rank"] == 3
    assert row["quality_first_premium_over_p1_q"] == 25.5
    assert bool(row["quality_first_material_premium"]) is True
    assert row["near_qf_alternative_count"] == 1
    assert row["support_distinct_iso_q_pair_count"] == 1
    assert "return_best_endpoint" in row["output_obligations"]
    assert "report_p1_quality_premium" in row["output_obligations"]
    assert "return_near_qf_alternatives" in row["output_obligations"]
    assert "return_support_distinct_iso_q_pairs" in row["output_obligations"]


def test_accurate_portfolio_member_rows_label_best_p1_and_lookalikes():
    module = _load_script("leiden_accurate_member_contract_for_test")
    candidates = pd.DataFrame(
        [
            _candidate(0, p1=40.0, p5=10.0, support_nodes="0;1"),
            _candidate(1, p1=30.0, p5=35.0, support_nodes="0;1"),
            _candidate(2, p1=20.0, p5=35.5, support_nodes="2;3"),
        ]
    )

    rows = module.build_accurate_portfolio_member_rows(
        candidates,
        near_best_delta_q=1.0,
        support_distinct_tau=0.5,
        iso_q_delta=1.0,
        iso_q_relative_ppm=10.0,
    )

    by_candidate = {
        int(row["candidate_index"]): row for _, row in rows.iterrows()
    }
    assert set(by_candidate) == {0, 1, 2}
    assert by_candidate[0]["portfolio_role"] == "p1_choice"
    assert "near_qf_alternative" in by_candidate[1]["portfolio_role"]
    assert "support_distinct_lookalike" in by_candidate[1]["portfolio_role"]
    assert "quality_first_best" in by_candidate[2]["portfolio_role"]
    assert by_candidate[2]["q_gap_to_best"] == 0.0
    assert by_candidate[1]["q_gap_to_best"] == 0.5
    assert by_candidate[1]["distance_to_best_support"] == 1.0
    assert bool(by_candidate[1]["same_coarse_as_best"]) is False


def test_accurate_contract_summary_and_report(tmp_path):
    module = _load_script("leiden_accurate_contract_report_for_test")
    candidates = pd.DataFrame(
        [
            _candidate(0, p1=40.0, p5=10.0, support_nodes="0;1"),
            _candidate(1, p1=30.0, p5=35.0, support_nodes="0;1"),
            _candidate(2, p1=20.0, p5=35.5, support_nodes="2;3"),
        ]
    )
    contract = module.build_accurate_contract_rows(
        candidates,
        material_regret_q=10.0,
        near_best_delta_q=1.0,
        support_distinct_tau=0.5,
        iso_q_delta=1.0,
        iso_q_relative_ppm=10.0,
    )
    members = module.build_accurate_portfolio_member_rows(
        candidates,
        near_best_delta_q=1.0,
        support_distinct_tau=0.5,
        iso_q_delta=1.0,
        iso_q_relative_ppm=10.0,
    )
    pairs = module.build_lookalike_pair_rows(
        candidates,
        support_distinct_tau=0.5,
        iso_q_delta=1.0,
        iso_q_relative_ppm=10.0,
    )
    summary = module.build_contract_summary(contract)

    all_row = summary[summary["group"] == "all"].iloc[0]
    assert all_row["case_count"] == 1
    assert all_row["best_endpoint_only_count"] == 0
    assert all_row["best_plus_near_qf_support_distinct_portfolio_count"] == 1
    assert all_row["high_quality_and_interpretation_risk_count"] == 1
    assert all_row["quality_first_premium_q_sum"] == 25.5
    assert all_row["support_distinct_iso_q_pair_count"] == 1

    module.write_report(tmp_path, contract, members, pairs, summary)
    report = (
        tmp_path / "dongdaemun_accurate_basin_portfolio_contract_report.md"
    ).read_text(encoding="utf-8")
    assert "Dongdaemun Accurate Basin Portfolio Contract" in report
    assert "Accurate mode must return a structured portfolio" in report
    assert "winner-only sufficient cases: 0/1" in report


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
