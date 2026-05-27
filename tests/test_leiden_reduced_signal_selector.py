"""Tests for attribution-derived reduced non-p1 selector diagnostics."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "research" / "consensus" / "scripts"
ANALYSIS_PATH = SCRIPT_DIR / "analyze_leiden_reduced_signal_selector.py"


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


def test_reduced_nonp1_selector_recovers_hard_and_core_without_p1_rank2_leak():
    module = _load_script("leiden_reduced_selector_for_test")
    rows = module.build_reduced_selector_rows(
        _candidate_frame(),
        material_regret_q=10.0,
        near_best_delta_q=1.0,
        support_distinct_tau=0.5,
    )

    p1_hard = rows[
        (rows["selector_name"] == "p1_top3") & (rows["contract_tier"] == "hard")
    ].iloc[0]
    reduced_hard = rows[
        (rows["selector_name"] == "reduced_nonp1_top1_union")
        & (rows["contract_tier"] == "hard")
    ].iloc[0]
    hybrid_core = rows[
        (rows["selector_name"] == "p1_top3_plus_reduced_nonp1_top1")
        & (rows["contract_tier"] == "core")
    ].iloc[0]

    assert bool(p1_hard["tier_covered"]) is False
    assert bool(p1_hard["material_regret"]) is True
    assert bool(reduced_hard["tier_covered"]) is False
    assert bool(reduced_hard["endpoint_obligation_covered"]) is True
    assert bool(reduced_hard["material_regret"]) is False
    assert reduced_hard["selected_candidate_indices"] == "1;2"
    assert bool(hybrid_core["tier_covered"]) is True
    assert hybrid_core["selected_candidate_indices"] == "0;1;2;3;4"


def test_reduced_selector_summary_and_reference_comparison():
    module = _load_script("leiden_reduced_selector_summary_for_test")
    rows = module.build_reduced_selector_rows(
        _candidate_frame(),
        material_regret_q=10.0,
        near_best_delta_q=1.0,
        support_distinct_tau=0.5,
    )

    summary = module.build_reduced_selector_summary(rows)
    comparison = module.build_reference_comparison_rows(summary)

    reduced_hard = summary[
        (summary["selector_name"] == "reduced_nonp1_top1_union")
        & (summary["contract_tier"] == "hard")
    ].iloc[0]
    pair = comparison[
        (comparison["reference_selector"] == "cheap_metric_top1_union")
        & (comparison["reduced_selector"] == "reduced_nonp1_top1_union")
        & (comparison["contract_tier"] == "hard")
    ].iloc[0]
    assert reduced_hard["tier_covered_count"] == 0
    assert reduced_hard["material_regret_count"] == 0
    assert pair["reduced_tier_covered_count"] == 0
    assert pair["tier_covered_count_delta"] == -1


def test_reduced_signal_selector_report_is_written(tmp_path):
    module = _load_script("leiden_reduced_selector_report_for_test")
    rows = module.build_reduced_selector_rows(
        _candidate_frame(),
        material_regret_q=10.0,
        near_best_delta_q=1.0,
        support_distinct_tau=0.5,
    )
    summary = module.build_reduced_selector_summary(rows)
    comparison = module.build_reference_comparison_rows(summary)

    module.write_report(tmp_path, rows, summary, comparison)

    report = (tmp_path / "dongdaemun_reduced_signal_selector_report.md").read_text(
        encoding="utf-8"
    )
    assert "Dongdaemun Reduced Signal Selector" in report
    assert "Reduced Selector Summary" in report
    assert "Reference Comparison" in report


def _candidate_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _candidate(
                0,
                p1=50.0,
                p5=0.0,
                support_nodes="0;1",
                priority=0.1,
                split=0.0,
                weight=1.0,
                target=1.0,
                edges=1.0,
            ),
            _candidate(
                3,
                p1=40.0,
                p5=20.0,
                support_nodes="4;5",
                priority=0.9,
                split=1.0,
                weight=2.0,
                target=2.0,
                edges=2.0,
            ),
            _candidate(
                4,
                p1=30.0,
                p5=20.5,
                support_nodes="6;7",
                priority=0.8,
                split=2.0,
                weight=3.0,
                target=3.0,
                edges=3.0,
            ),
            _candidate(
                1,
                p1=20.0,
                p5=99.5,
                support_nodes="0;1",
                priority=0.2,
                split=100.0,
                weight=100.0,
                target=10.0,
                edges=10.0,
            ),
            _candidate(
                2,
                p1=10.0,
                p5=100.0,
                support_nodes="2;3",
                priority=1.0,
                split=10.0,
                weight=10.0,
                target=100.0,
                edges=100.0,
            ),
        ]
    )


def _candidate(
    candidate_index: int,
    *,
    p1: float,
    p5: float,
    support_nodes: str,
    priority: float,
    split: float,
    weight: float,
    target: float,
    edges: float,
    rel: float = 100.0,
) -> dict[str, object]:
    return {
        "candidate_eval_mode": "multifidelity_label",
        "case": "adaptive_refinement_field12_gcc_emb_full_knn30_bc_cosine",
        "seed": 11,
        "candidate_budget": 5,
        "max_group_candidates": 5,
        "candidate_index": candidate_index,
        "priority": priority,
        "p1_delta_q": p1,
        "group_delta_q": p1,
        "group_move_delta_q": p1,
        "group_split_delta_q": split,
        "group_weight": weight,
        "group_fraction": priority,
        "group_cut_weight": weight,
        "group_to_target_weight": target,
        "localized_delta_q": split,
        "quotient_delta_q": split,
        "pre_delta_q": split,
        "ub_delta_q": split,
        "incident_directed_edges": edges,
        "p5_delta_q": p5,
        "p5_relative_delta_q_ppm": rel + candidate_index,
        "p5_elapsed_ms": 10.0 + candidate_index,
        "p5_basin_signature": f"basin-{candidate_index}",
        "p5_changed_fraction_vs_baseline": 0.001 * (candidate_index + 1),
        "p5_changed_nodes_vs_baseline": candidate_index + 1,
        "p5_basin_sketch_node_hash": "sample-hash",
        "p5_basin_sketch_baseline_membership": "0;0;1;1;2;2;3;3",
        "p5_basin_sketch_membership": "0;0;1;1;2;2;3;3",
        "p5_basin_changed_support_node_count": 2,
        "p5_basin_changed_support_sketch_sample_size": 2,
        "p5_basin_changed_support_nodes": support_nodes,
    }
