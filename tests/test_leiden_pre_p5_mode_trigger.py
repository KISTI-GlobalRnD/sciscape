"""Tests for Dongdaemun pre-p5 mode trigger diagnostics."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "research" / "consensus" / "scripts"
ANALYSIS_PATH = Path(__file__).resolve().parents[1] / "research/consensus/scripts/leiden_basin/basin_signatures/local_modes/analyze_leiden_pre_p5_mode_trigger.py"


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


def test_pre_p5_features_exclude_p5_oracle_columns():
    module = _load_script("leiden_pre_p5_features_for_test")
    candidates = pd.DataFrame(
        [
            _candidate(0, field=12, p1=30.0, p5=10.0, priority=0.8),
            _candidate(1, field=12, p1=20.0, p5=35.0, priority=0.2),
        ]
    )

    features = module.build_pre_p5_feature_rows(candidates)

    row = features.iloc[0]
    assert row["p1_candidate_index"] == 0
    assert "p5_delta_q" not in features.columns
    assert "oracle_fast_p1_quality_regret_q" not in features.columns
    assert row["p1_gap_1_2_abs"] == 10.0
    assert row["cheap_metric_count"] > 0


def test_pre_p5_oracle_rows_attach_quality_and_portfolio_labels():
    module = _load_script("leiden_pre_p5_oracle_for_test")
    candidates = pd.DataFrame(
        [
            _candidate(0, field=12, p1=40.0, p5=10.0, support_nodes="0;1"),
            _candidate(1, field=12, p1=30.0, p5=35.0, support_nodes="0;1"),
            _candidate(2, field=12, p1=20.0, p5=35.5, support_nodes="2;3"),
        ]
    )

    rows = module.build_pre_p5_oracle_rows(candidates, material_regret_q=10.0)

    row = rows.iloc[0]
    assert bool(row["oracle_accurate_for_quality"]) is True
    assert bool(row["oracle_accurate_for_portfolio"]) is True
    assert bool(row["oracle_accurate_for_final"]) is True
    assert row["oracle_fast_p1_quality_regret_q"] == 25.5
    assert row["oracle_support_distinct_iso_q_pair_count"] == 1


def test_leave_field_out_trigger_reports_missed_regret():
    module = _load_script("leiden_pre_p5_lfo_for_test")
    candidates = pd.DataFrame(
        [
            _candidate(0, field=12, p1=40.0, p5=10.0),
            _candidate(1, field=12, p1=30.0, p5=35.0),
            _candidate(0, field=26, p1=40.0, p5=10.0),
            _candidate(1, field=26, p1=30.0, p5=35.0),
        ]
    )
    rows = module.build_pre_p5_oracle_rows(candidates, material_regret_q=10.0)

    lfo = module.build_leave_field_out_trigger_rows(
        rows,
        target_columns=("oracle_accurate_for_quality",),
    )

    assert set(lfo["heldout_field"]) == {12, 26}
    assert lfo["required_count"].sum() == 2
    assert "missed_quality_regret_q" in lfo.columns


def test_pre_p5_report_is_written(tmp_path):
    module = _load_script("leiden_pre_p5_report_for_test")
    candidates = pd.DataFrame(
        [
            _candidate(0, field=12, p1=40.0, p5=10.0),
            _candidate(1, field=12, p1=30.0, p5=35.0),
        ]
    )
    rows = module.build_pre_p5_oracle_rows(candidates)
    lfo = module.build_leave_field_out_trigger_rows(rows)
    baseline = module.build_baseline_policy_rows(rows)
    summary = module.build_trigger_summary(pd.concat([baseline, lfo], ignore_index=True))

    module.write_report(tmp_path, rows, lfo, baseline, summary)

    report = (tmp_path / "dongdaemun_pre_p5_mode_trigger_report.md").read_text(
        encoding="utf-8"
    )
    assert "Dongdaemun Pre-p5 Mode Trigger Review" in report
    assert "pre-p5 candidate features" in report


def _candidate(
    candidate_index: int,
    *,
    field: int,
    p1: float,
    p5: float,
    priority: float = 1.0,
    support_nodes: str = "0;1",
) -> dict[str, object]:
    return {
        "candidate_eval_mode": "multifidelity_label",
        "case": f"adaptive_refinement_field{field}_gcc_emb_full_knn30_bc_cosine",
        "seed": 11,
        "candidate_budget": 3,
        "max_group_candidates": 3,
        "candidate_index": candidate_index,
        "priority": priority,
        "p1_delta_q": p1,
        "group_delta_q": p1 * 0.5,
        "group_weight": 10.0 + candidate_index,
        "group_fraction": 0.1 * (candidate_index + 1),
        "localized_delta_q": p1 * 0.25,
        "quotient_delta_q": p1 * 0.2,
        "pre_delta_q": p1 * 0.1,
        "ub_delta_q": p1 * 0.3,
        "incident_directed_edges": 100 + candidate_index,
        "p5_delta_q": p5,
        "p5_relative_delta_q_ppm": 100.0 + candidate_index,
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
