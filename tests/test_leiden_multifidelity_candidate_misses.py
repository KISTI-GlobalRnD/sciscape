"""Tests for Leiden multi-fidelity candidate miss attribution."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "research" / "consensus" / "scripts"
ANALYSIS_PATH = Path(__file__).resolve().parents[1] / "research/consensus/scripts/leiden_basin/basin_signatures/trajectory_failure/analyze_leiden_multifidelity_candidate_misses.py"


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


def test_candidate_rank_tie_break_is_deterministic():
    module = _load_script("leiden_multifidelity_candidate_misses_tie")
    rows = pd.DataFrame(
        [
            _candidate(2, p1=5.0, p5=7.0),
            _candidate(1, p1=5.0, p5=7.0),
            _candidate(0, p1=4.0, p5=9.0),
        ]
    )

    diagnostics = module.build_candidate_rank_diagnostics(rows)
    by_candidate = diagnostics.set_index("candidate_index")

    assert by_candidate.loc[1, "p1_rank"] == 1
    assert by_candidate.loc[2, "p1_rank"] == 2
    assert by_candidate.loc[0, "p5_rank"] == 1
    assert by_candidate.loc[1, "p5_rank"] == 2
    assert by_candidate.loc[2, "p5_rank"] == 3


def test_synthetic_cc11_style_miss_marks_full_winner_missed_by_p1_top2():
    module = _load_script("leiden_multifidelity_candidate_misses_miss")
    rows = pd.DataFrame(
        [
            _candidate(0, p1=2.0, p5=2.1),
            _candidate(1, p1=1.0, p5=1.1),
            _candidate(2, p1=0.5, p5=3.2),
        ]
    )

    diagnostics = module.build_candidate_rank_diagnostics(rows)
    winner = diagnostics[diagnostics["candidate_index"] == 2].iloc[0]

    assert winner["p1_rank"] == 3
    assert winner["p5_rank"] == 1
    assert bool(winner["missed_by_p1_top2"]) is True


def test_analyzer_compares_policy_decisions_and_preserves_unreached_target(tmp_path):
    module = _load_script("leiden_multifidelity_candidate_misses_end_to_end")
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    pd.DataFrame(
        [
            _candidate(0, p1=3.0, p5=2.0, p1_elapsed=0.0, p5_elapsed=10.0),
            _candidate(1, p1=2.0, p5=1.0, p1_elapsed=0.0, p5_elapsed=20.0),
            _candidate(2, p1=1.0, p5=4.0, p1_elapsed=0.0, p5_elapsed=30.0),
        ]
    ).to_csv(input_dir / "candidate_level_rows.csv", index=False)
    pd.DataFrame(
        [
            _policy("full_top3_p5", selected=2, total=60.0, final_delta=4.0, matches=True),
            _policy("p1_top2_then_p5", selected=0, total=30.0, final_delta=2.0, matches=False),
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
                "perturb_tau_status": "did_not_reach_target",
                "k_work_saving_pct": float("nan"),
                "net_elapsed_saving_pct": 10.0,
            }
        ]
    ).to_csv(input_dir / "work_acceleration_monitor_scorecard.csv", index=False)

    paths = module.analyze_input_dir(input_dir)
    summary = pd.read_csv(paths["policy_decision_summary"])

    row = summary.iloc[0]
    assert bool(row["p1_top2_then_p5_same_decision_as_full_top3_p5"]) is False
    assert bool(row["p1_top3_then_p5_synthesized"]) is True
    assert bool(row["p1_top3_then_p5_same_decision_as_full_top3_p5"]) is True
    assert row["extra_p5_final_perturb_tau_status"] == "did_not_reach_target"
    assert bool(row["extra_p5_final_target_conclusion_changed"]) is True


def test_structural_rescue_selects_cc11_style_rank3_full_winner():
    module = _load_script("leiden_multifidelity_candidate_misses_rescue_cc11")
    diagnostics = module.build_candidate_rank_diagnostics(
        pd.DataFrame(
            [
                _candidate(0, p1=2.0, p5=2.1, group_count=7, target_weight=8.0),
                _candidate(1, p1=1.0, p5=1.1, group_count=1, target_weight=0.2),
                _candidate(2, p1=0.5, p5=3.2, group_count=2, target_weight=0.7),
            ]
        )
    )
    policy_rows = module.build_structural_rescue_policy_rows(diagnostics)
    summary = module.build_structural_rescue_summary(
        diagnostics,
        module.augment_policy_rows(_full_policy_rows(diagnostics), diagnostics),
        policy_rows,
    )

    policy = policy_rows.iloc[0]
    row = summary.iloc[0]
    assert int(policy["rescue_candidate_index"]) == 2
    assert bool(policy["rescue_candidate_is_full_p5_winner"]) is True
    assert int(policy["selected_candidate_index"]) == 2
    assert bool(row["structural_rescue_same_decision_as_full_top3_p5"]) is True


def test_structural_rescue_does_not_break_when_top2_already_has_winner():
    module = _load_script("leiden_multifidelity_candidate_misses_rescue_no_regress")
    diagnostics = module.build_candidate_rank_diagnostics(
        pd.DataFrame(
            [
                _candidate(0, p1=3.0, p5=5.0, group_count=7, target_weight=8.0),
                _candidate(1, p1=2.0, p5=1.0, group_count=1, target_weight=0.2),
                _candidate(2, p1=1.0, p5=3.0, group_count=2, target_weight=0.7),
            ]
        )
    )
    policy_rows = module.build_structural_rescue_policy_rows(diagnostics)
    summary = module.build_structural_rescue_summary(
        diagnostics,
        module.augment_policy_rows(_full_policy_rows(diagnostics), diagnostics),
        policy_rows,
    )

    policy = policy_rows.iloc[0]
    row = summary.iloc[0]
    assert int(policy["rescue_candidate_index"]) == 2
    assert int(policy["selected_candidate_index"]) == 0
    assert bool(row["structural_rescue_same_decision_as_full_top3_p5"]) is True


def test_structural_rescue_tie_break_is_deterministic():
    module = _load_script("leiden_multifidelity_candidate_misses_rescue_tie")
    diagnostics = module.build_candidate_rank_diagnostics(
        pd.DataFrame(
            [
                _candidate(0, p1=4.0, p5=4.0, group_count=3, target_weight=3.0),
                _candidate(1, p1=3.0, p5=3.0, group_count=3, target_weight=3.0),
                _candidate(2, p1=2.0, p5=2.0, group_count=2, target_weight=0.8, cut_weight=0.4),
                _candidate(3, p1=1.0, p5=5.0, group_count=2, target_weight=0.8, cut_weight=0.2),
            ]
        )
    )

    policy = module.build_structural_rescue_policy_rows(diagnostics).iloc[0]

    assert int(policy["rescue_candidate_index"]) == 3


def test_structural_rescue_rejects_high_target_weight_per_node_loser():
    module = _load_script("leiden_multifidelity_candidate_misses_rescue_reject_high_target")
    diagnostics = module.build_candidate_rank_diagnostics(
        pd.DataFrame(
            [
                _candidate(0, p1=3.0, p5=2.0, group_count=7, target_weight=8.0),
                _candidate(1, p1=2.0, p5=4.0, group_count=1, target_weight=0.2),
                _candidate(2, p1=1.0, p5=1.0, group_count=2, target_weight=2.0),
            ]
        )
    )

    policy = module.build_structural_rescue_policy_rows(diagnostics).iloc[0]

    assert int(policy["rescue_candidate_index"]) == -1
    assert bool(policy["rescue_selected"]) is False
    assert policy["finalist_candidate_indices"] == "0,1"


def test_structural_rescue_is_unavailable_when_rescued_p5_label_is_missing():
    module = _load_script("leiden_multifidelity_candidate_misses_rescue_missing_p5")
    diagnostics = module.build_candidate_rank_diagnostics(
        pd.DataFrame(
            [
                _candidate(0, p1=3.0, p5=2.0, group_count=7, target_weight=8.0),
                _candidate(1, p1=2.0, p5=1.0, group_count=1, target_weight=0.2),
                _candidate(
                    2,
                    p1=1.0,
                    p5=float("nan"),
                    group_count=2,
                    target_weight=0.7,
                ),
            ]
        )
    )

    policy = module.build_structural_rescue_policy_rows(diagnostics).iloc[0]

    assert int(policy["rescue_candidate_index"]) == 2
    assert bool(policy["available"]) is False
    assert int(policy["p5_evaluated"]) == 2


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
        "group_count": group_count,
        "group_weight": float(group_count),
        "group_fraction": 0.1 * group_count,
        "group_to_target_weight": target_weight,
        "group_cut_weight": cut_weight,
        "priority": 0.01 * (candidate_index + 1),
        "best_group_delta_q": 0.4 * (candidate_index + 1),
        "group_kind": group_kind,
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


def _full_policy_rows(diagnostics: pd.DataFrame) -> pd.DataFrame:
    winner = diagnostics[diagnostics["is_full_p5_winner"]].iloc[0]
    return pd.DataFrame(
        [
            _policy(
                "full_top3_p5",
                selected=int(winner["candidate_index"]),
                total=float(diagnostics["p5_elapsed_ms"].sum()),
                final_delta=float(winner["p5_delta_q"]),
                matches=True,
            )
        ]
    )
