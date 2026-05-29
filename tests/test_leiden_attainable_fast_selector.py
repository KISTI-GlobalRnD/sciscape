"""Tests for pre-p5 attainable Dongdaemun fast selector diagnostics."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "research" / "consensus" / "scripts"
ANALYSIS_PATH = Path(__file__).resolve().parents[1] / "research/consensus/scripts/leiden_basin/operator_probes/selector_signals/analyze_leiden_attainable_fast_selector.py"


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


def test_attainable_selector_rows_compare_p1_prefix_and_cheap_union():
    module = _load_script("leiden_attainable_selector_for_test")
    candidates = _candidate_frame()

    rows = module.build_attainable_selector_rows(
        candidates,
        material_regret_q=10.0,
        near_best_delta_q=1.0,
        support_distinct_tau=0.5,
    )

    p1_hard = rows[
        (rows["selector_name"] == "p1_top1") & (rows["contract_tier"] == "hard")
    ].iloc[0]
    assert bool(p1_hard["tier_covered"]) is False
    assert bool(p1_hard["endpoint_obligation_covered"]) is False
    assert p1_hard["required_candidate_indices"] == "0;2"
    assert p1_hard["selected_candidate_indices"] == "0"
    assert p1_hard["required_candidate_missed_count"] == 1
    assert p1_hard["quality_regret_q"] == 100.0

    union_core = rows[
        (rows["selector_name"] == "cheap_metric_top1_union")
        & (rows["contract_tier"] == "core")
    ].iloc[0]
    assert bool(union_core["tier_covered"]) is True
    assert bool(union_core["endpoint_obligation_covered"]) is True
    assert union_core["required_candidate_indices"] == "0;1;2"
    assert union_core["selected_candidate_indices"] == "0;1;2"
    assert union_core["selected_candidate_count"] == 3
    assert union_core["quality_regret_q"] == 0.0


def test_attainable_selector_summary_counts_coverage_and_regret():
    module = _load_script("leiden_attainable_selector_summary_for_test")
    candidates = _candidate_frame()
    rows = module.build_attainable_selector_rows(
        candidates,
        material_regret_q=10.0,
        near_best_delta_q=1.0,
        support_distinct_tau=0.5,
    )

    summary = module.build_attainable_selector_summary(rows)

    p1_hard = summary[
        (summary["selector_name"] == "p1_top1")
        & (summary["contract_tier"] == "hard")
    ].iloc[0]
    union_core = summary[
        (summary["selector_name"] == "cheap_metric_top1_union")
        & (summary["contract_tier"] == "core")
    ].iloc[0]
    assert p1_hard["tier_covered_count"] == 0
    assert p1_hard["material_regret_count"] == 1
    assert p1_hard["quality_regret_q_sum"] == 100.0
    assert union_core["tier_covered_count"] == 1
    assert union_core["material_regret_count"] == 0
    assert union_core["selected_candidate_count_mean"] == 3.0


def test_attainable_selector_report_is_written(tmp_path):
    module = _load_script("leiden_attainable_selector_report_for_test")
    candidates = _candidate_frame()
    rows = module.build_attainable_selector_rows(
        candidates,
        material_regret_q=10.0,
        near_best_delta_q=1.0,
        support_distinct_tau=0.5,
    )
    summary = module.build_attainable_selector_summary(rows)

    module.write_report(tmp_path, rows, summary)

    report = (tmp_path / "dongdaemun_attainable_fast_selector_report.md").read_text(
        encoding="utf-8"
    )
    assert "Dongdaemun Attainable Fast Selector" in report
    assert "cheap/pre-p5 selector families" in report
    assert "Cheap-metric union selectors are diagnostic prototypes" in report


def _candidate_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _candidate(
                0,
                p1=50.0,
                p5=0.0,
                support_nodes="0;1",
                priority=0.1,
                localized=0.0,
                ub=0.0,
                weight=1.0,
            ),
            _candidate(
                1,
                p1=40.0,
                p5=99.5,
                support_nodes="0;1",
                priority=0.2,
                localized=100.0,
                ub=100.0,
                weight=100.0,
            ),
            _candidate(
                2,
                p1=30.0,
                p5=100.0,
                support_nodes="2;3",
                priority=1.0,
                localized=10.0,
                ub=10.0,
                weight=10.0,
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
    localized: float,
    ub: float,
    weight: float,
    rel: float = 100.0,
) -> dict[str, object]:
    return {
        "candidate_eval_mode": "multifidelity_label",
        "case": "adaptive_refinement_field12_gcc_emb_full_knn30_bc_cosine",
        "seed": 11,
        "candidate_budget": 3,
        "max_group_candidates": 3,
        "candidate_index": candidate_index,
        "priority": priority,
        "p1_delta_q": p1,
        "group_delta_q": p1,
        "group_move_delta_q": p1,
        "group_split_delta_q": p1,
        "group_weight": weight,
        "group_fraction": priority,
        "group_cut_weight": weight,
        "group_to_target_weight": weight,
        "localized_delta_q": localized,
        "quotient_delta_q": localized,
        "pre_delta_q": ub,
        "ub_delta_q": ub,
        "incident_directed_edges": weight,
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
