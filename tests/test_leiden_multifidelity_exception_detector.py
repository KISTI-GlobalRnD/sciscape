"""Tests for p1-visible Leiden multifidelity exception detector analysis."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "research" / "consensus" / "scripts"
ANALYSIS_PATH = Path(__file__).resolve().parents[1] / "research/consensus/scripts/leiden_basin/basin_signatures/trajectory_failure/analyze_leiden_multifidelity_exception_detector.py"


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


def test_feature_rows_mark_cc11_style_top2_exception_and_structural_rescue():
    module = _load_script("leiden_multifidelity_exception_detector_features")
    diagnostics = module.build_candidate_rank_diagnostics(
        pd.DataFrame(
            [
                _candidate(0, p1=2.0, p5=2.1, group_count=7, target_weight=8.0),
                _candidate(1, p1=1.0, p5=1.1, group_count=1, target_weight=0.2),
                _candidate(2, p1=0.8, p5=3.2, group_count=2, target_weight=0.7),
            ]
        )
    )
    diagnostics.insert(0, "dataset", "synthetic")

    features = module.build_exception_feature_rows(
        diagnostics,
        pd.DataFrame(),
        pd.DataFrame(),
    )

    row = features.iloc[0]
    assert bool(row["missed_by_p1_top2"]) is True
    assert bool(row["needs_p1_top3"]) is True
    assert int(row["structural_rescue_candidate_index"]) == 2
    assert bool(row["structural_rescue_candidate_is_full_p5_winner"]) is True
    assert round(float(row["p1_top2_top3_gap"]), 6) == 0.2


def test_policy_scorecard_rewards_structural_rescue_without_always_top3_cost():
    module = _load_script("leiden_multifidelity_exception_detector_policy")
    diagnostics = module.build_candidate_rank_diagnostics(
        pd.DataFrame(
            [
                _candidate(0, p1=2.0, p5=2.1, p1_elapsed=1.0, p5_elapsed=10.0, group_count=7, target_weight=8.0),
                _candidate(1, p1=1.0, p5=1.1, p1_elapsed=1.0, p5_elapsed=10.0, group_count=1, target_weight=0.2),
                _candidate(2, p1=0.8, p5=3.2, p1_elapsed=1.0, p5_elapsed=10.0, group_count=2, target_weight=0.7),
            ]
        )
    )
    diagnostics.insert(0, "dataset", "synthetic")
    features = module.build_exception_feature_rows(diagnostics, pd.DataFrame(), pd.DataFrame())

    policy_rows = module.build_policy_evaluation_rows(diagnostics, features)
    scorecard = module.build_policy_scorecard(policy_rows).set_index("policy")

    assert int(scorecard.loc["always_p1_top2", "n_policy_misses"]) == 1
    assert int(scorecard.loc["p1_top2_with_structural_rescue", "n_policy_misses"]) == 0
    assert scorecard.loc["p1_top2_with_structural_rescue", "miss_recall"] == 1.0
    assert scorecard.loc["p1_top2_with_structural_rescue", "mean_p5_evaluated"] == 3.0


def test_analyze_input_dirs_writes_feature_scorecard_and_report(tmp_path):
    module = _load_script("leiden_multifidelity_exception_detector_end_to_end")
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    pd.DataFrame(
        [
            _candidate(0, p1=2.0, p5=2.1),
            _candidate(1, p1=1.0, p5=1.1),
            _candidate(2, p1=0.8, p5=3.2, group_count=2, target_weight=0.7),
        ]
    ).to_csv(input_dir / "candidate_level_rows.csv", index=False)
    pd.DataFrame(
        [
            _policy("full_top3_p5", selected=2, total=30.0, final_delta=3.2, matches=True),
            _policy("p1_top2_then_p5", selected=0, total=23.0, final_delta=2.1, matches=False),
        ]
    ).to_csv(input_dir / "policy_comparison_rows.csv", index=False)
    pd.DataFrame(
        [
            {
                "case": "case",
                "seed": 11,
                "candidate_budget": 3,
                "target_policy": "extra_p5_final",
                "extra_tau_status": "reached",
                "perturb_tau_status": "reached",
                "extra_p5_final_target_conclusion_changed": False,
            }
        ]
    ).to_csv(input_dir / "work_acceleration_monitor_scorecard.csv", index=False)

    paths = module.analyze_input_dirs([input_dir], output_dir)

    assert paths["feature_rows"].exists()
    assert paths["policy_scorecard"].exists()
    assert paths["report"].exists()
    features = pd.read_csv(paths["feature_rows"])
    assert bool(features.iloc[0]["missed_by_p1_top2"]) is True


def _candidate(
    candidate_index: int,
    *,
    p1: float,
    p5: float,
    p1_elapsed: float = 1.0,
    p5_elapsed: float = 5.0,
    group_kind: str = "best",
    group_count: int | None = None,
    target_weight: float | None = None,
    cut_weight: float | None = None,
) -> dict[str, object]:
    group_count = candidate_index + 1 if group_count is None else group_count
    target_weight = 0.2 * (candidate_index + 1) if target_weight is None else target_weight
    cut_weight = 0.3 * (candidate_index + 1) if cut_weight is None else cut_weight
    return {
        "case": "case",
        "seed": 11,
        "candidate_budget": 3,
        "candidate_eval_mode": "multifidelity_label",
        "selected_policy": "p1_top1_then_p5",
        "candidate_index": candidate_index,
        "source_cluster": 10 + candidate_index,
        "target_cluster": 20 + candidate_index,
        "p1_delta_q": p1,
        "p5_delta_q": p5,
        "p1_quality": 100.0 + p1,
        "p5_quality": 100.0 + p5,
        "p1_elapsed_ms": p1_elapsed,
        "p5_elapsed_ms": p5_elapsed,
        "pre_delta_q": 0.1 * candidate_index,
        "group_count": group_count,
        "group_weight": float(group_count),
        "group_fraction": 0.1 * group_count,
        "group_to_target_weight": target_weight,
        "group_cut_weight": cut_weight,
        "priority": 0.01 * (candidate_index + 1),
        "best_group_delta_q": 0.4 * (candidate_index + 1),
        "group_kind": group_kind,
        "recommended_for_split_repair": True,
    }


def _policy(
    policy: str,
    *,
    selected: int,
    total: float,
    final_delta: float,
    matches: bool,
) -> dict[str, object]:
    return {
        "case": "case",
        "seed": 11,
        "candidate_budget": 3,
        "candidate_eval_mode": "multifidelity_label",
        "policy": policy,
        "selected_candidate_index": selected,
        "candidate_count": 3,
        "p1_evaluated": 3,
        "p5_evaluated": 3,
        "p1_elapsed_ms": 0.0,
        "p5_elapsed_ms": total,
        "total_elapsed_ms": total,
        "final_delta_q": final_delta,
        "quality": 100.0 + final_delta,
        "accepted": True,
        "available": True,
        "matches_full_p5": matches,
    }
