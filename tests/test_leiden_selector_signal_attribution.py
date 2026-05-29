"""Tests for cheap/pre-p5 selector signal attribution diagnostics."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "research" / "consensus" / "scripts"
ANALYSIS_PATH = Path(__file__).resolve().parents[1] / "research/consensus/scripts/leiden_basin/operator_probes/selector_signals/analyze_leiden_selector_signal_attribution.py"


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


def test_signal_candidate_rows_mark_non_p1_recovery():
    module = _load_script("leiden_selector_signal_attribution_for_test")
    candidates = _candidate_frame()

    rows = module.build_signal_candidate_rows(
        candidates,
        material_regret_q=10.0,
        near_best_delta_q=1.0,
        support_distinct_tau=0.5,
    )

    priority_best = rows[
        (rows["metric_name"] == "priority")
        & (rows["metric_rank"] == 1)
        & (rows["candidate_index"] == 2)
    ].iloc[0]
    assert priority_best["requirement_tier"] == "hard"
    assert bool(priority_best["required_hard"]) is True
    assert bool(priority_best["metric_recovers_hard_vs_p1_top1"]) is True
    assert bool(priority_best["is_quality_first_candidate"]) is True

    localized_near = rows[
        (rows["metric_name"] == "localized_delta_q")
        & (rows["metric_rank"] == 1)
        & (rows["candidate_index"] == 1)
    ].iloc[0]
    assert localized_near["requirement_tier"] == "core"
    assert bool(localized_near["required_core"]) is True
    assert bool(localized_near["metric_recovers_core_vs_p1_top3"]) is True


def test_metric_signal_summary_counts_recovery_potential():
    module = _load_script("leiden_selector_signal_summary_for_test")
    rows = module.build_signal_candidate_rows(
        _candidate_frame(),
        material_regret_q=10.0,
        near_best_delta_q=1.0,
        support_distinct_tau=0.5,
    )

    summary = module.build_metric_signal_summary(rows)

    priority_top1 = summary[
        (summary["metric_name"] == "priority")
        & (summary["metric_top_n"] == 1)
    ].iloc[0]
    localized_top1 = summary[
        (summary["metric_name"] == "localized_delta_q")
        & (summary["metric_top_n"] == 1)
    ].iloc[0]
    assert priority_top1["hard_recover_vs_p1_top1_case_count"] == 1
    assert priority_top1["endpoint_recover_vs_p1_top1_case_count"] == 1
    assert localized_top1["core_recover_vs_p1_top3_case_count"] == 1


def test_selector_delta_rows_attribute_cover_flips_to_signals():
    module = _load_script("leiden_selector_signal_delta_for_test")
    rows = module.build_selector_delta_rows(
        _candidate_frame(),
        material_regret_q=10.0,
        near_best_delta_q=1.0,
        support_distinct_tau=0.5,
    )

    hard = rows[
        (rows["base_selector"] == "p1_top1")
        & (rows["expanded_selector"] == "cheap_metric_top1_union")
        & (rows["contract_tier"] == "hard")
    ].iloc[0]
    core = rows[
        (rows["base_selector"] == "p1_top3")
        & (rows["expanded_selector"] == "p1_top3_plus_metric_top1")
        & (rows["contract_tier"] == "core")
    ].iloc[0]
    assert bool(hard["tier_cover_flip"]) is True
    assert bool(hard["material_regret_fixed"]) is True
    assert hard["newly_covered_required_candidate_indices"] == "2"
    assert "priority@rank1" in hard["responsible_signal_sources"]

    assert bool(core["tier_cover_flip"]) is True
    assert core["newly_covered_required_candidate_indices"] == "1;2"
    assert "localized_delta_q@rank1" in core["responsible_signal_sources"]

    gain_summary = module.build_selector_gain_summary(rows)
    realized = module.build_realized_signal_summary(rows)
    assert (
        gain_summary[
            (gain_summary["expanded_selector"] == "cheap_metric_top1_union")
            & (gain_summary["contract_tier"] == "hard")
        ]["tier_cover_flip_count"].iloc[0]
        == 1
    )
    assert "priority" in set(realized["signal_name"])


def test_selector_signal_attribution_report_is_written(tmp_path):
    module = _load_script("leiden_selector_signal_report_for_test")
    candidates = _candidate_frame()
    candidate_rows = module.build_signal_candidate_rows(
        candidates,
        material_regret_q=10.0,
        near_best_delta_q=1.0,
        support_distinct_tau=0.5,
    )
    metric_summary = module.build_metric_signal_summary(candidate_rows)
    delta_rows = module.build_selector_delta_rows(
        candidates,
        material_regret_q=10.0,
        near_best_delta_q=1.0,
        support_distinct_tau=0.5,
    )
    gain_summary = module.build_selector_gain_summary(delta_rows)
    realized_signal_summary = module.build_realized_signal_summary(delta_rows)

    module.write_report(
        tmp_path,
        candidate_rows,
        metric_summary,
        delta_rows,
        gain_summary,
        realized_signal_summary,
    )

    report = (
        tmp_path / "dongdaemun_selector_signal_attribution_report.md"
    ).read_text(encoding="utf-8")
    assert "Dongdaemun Selector Signal Attribution" in report
    assert "Metric Recovery Potential" in report
    assert "Signal Sources Behind Gains" in report


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
                3,
                p1=40.0,
                p5=20.0,
                support_nodes="4;5",
                priority=0.9,
                localized=1.0,
                ub=1.0,
                weight=2.0,
            ),
            _candidate(
                4,
                p1=30.0,
                p5=20.5,
                support_nodes="6;7",
                priority=0.8,
                localized=2.0,
                ub=2.0,
                weight=3.0,
            ),
            _candidate(
                1,
                p1=20.0,
                p5=99.5,
                support_nodes="0;1",
                priority=0.2,
                localized=100.0,
                ub=100.0,
                weight=100.0,
            ),
            _candidate(
                2,
                p1=10.0,
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
        "candidate_budget": 5,
        "max_group_candidates": 5,
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
        "p5_basin_sketch_baseline_membership": "0;0;1;1;2;2;3;3",
        "p5_basin_sketch_membership": "0;0;1;1;2;2;3;3",
        "p5_basin_changed_support_node_count": 2,
        "p5_basin_changed_support_sketch_sample_size": 2,
        "p5_basin_changed_support_nodes": support_nodes,
    }
