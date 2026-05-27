"""Tests for approximate polish candidate analysis."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "research" / "consensus" / "scripts"
ANALYSIS_PATH = SCRIPT_DIR / "analyze_leiden_approx_polish_candidates.py"


def _load_script(path: Path, module_name: str):
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_approx_polish_analysis_summarizes_recall_and_policy_rows(tmp_path):
    module = _load_script(ANALYSIS_PATH, "leiden_approx_polish_analysis_for_test")
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    pd.DataFrame(
        [
            {
                "candidate_eval_mode": "localized_label",
                "case": "case",
                "seed": 11,
                "candidate_budget": 3,
                "max_group_candidates": 3,
                "candidate_index": 0,
                "p5_delta_q": 1.0,
                "localized_delta_q": 3.0,
                "localized_rank": 1,
                "quotient_delta_q": 1.0,
                "quotient_rank": 3,
                "ub_delta_q": 5.0,
                "ub_rank": 1,
                "ub_covers_p5": True,
                "ub_violation": 0.0,
            },
            {
                "candidate_eval_mode": "localized_label",
                "case": "case",
                "seed": 11,
                "candidate_budget": 3,
                "max_group_candidates": 3,
                "candidate_index": 1,
                "p5_delta_q": 3.0,
                "localized_delta_q": 2.0,
                "localized_rank": 2,
                "quotient_delta_q": 3.0,
                "quotient_rank": 1,
                "ub_delta_q": 4.0,
                "ub_rank": 2,
                "ub_covers_p5": True,
                "ub_violation": 0.0,
            },
        ]
    ).to_csv(input_dir / "candidate_level_rows.csv", index=False)
    pd.DataFrame(
        [
            {
                "candidate_eval_mode": "localized_label",
                "policy": "localized_top2_then_p5",
                "available": True,
                "matches_full_p5": True,
                "total_elapsed_ms": 10.0,
                "p5_evaluated": 2,
                "candidate_count": 3,
            }
        ]
    ).to_csv(input_dir / "policy_comparison_rows.csv", index=False)

    candidates = module._read_csvs(input_dir, "candidate_level_rows.csv")
    policies = module._read_csvs(input_dir, "policy_comparison_rows.csv")
    candidate_summary = module._candidate_summary(candidates)
    policy_summary = module._policy_summary(policies)
    output_dir.mkdir()
    module._write_report(output_dir, candidate_summary, policy_summary)

    localized = candidate_summary[candidate_summary["approach"] == "localized"].iloc[0]
    quotient = candidate_summary[candidate_summary["approach"] == "quotient"].iloc[0]
    assert bool(localized["recall_at_1"]) is False
    assert bool(localized["recall_at_2"]) is True
    assert bool(quotient["recall_at_1"]) is True
    assert policy_summary.iloc[0]["matches_full_p5_rate"] == 1.0
    assert (output_dir / "approx_polish_report.md").exists()


def test_approx_polish_analysis_keeps_candidate_eval_modes_separate(tmp_path):
    module = _load_script(ANALYSIS_PATH, "leiden_approx_polish_analysis_modes_for_test")
    rows = []
    for mode in ("localized_label", "quotient_label"):
        rows.extend(
            [
                {
                    "candidate_eval_mode": mode,
                    "case": "case",
                    "seed": 11,
                    "candidate_budget": 3,
                    "max_group_candidates": 3,
                    "candidate_index": 0,
                    "p5_delta_q": 5.0,
                    "localized_delta_q": 2.0,
                    "localized_rank": 1,
                },
                {
                    "candidate_eval_mode": mode,
                    "case": "case",
                    "seed": 11,
                    "candidate_budget": 3,
                    "max_group_candidates": 3,
                    "candidate_index": 1,
                    "p5_delta_q": 1.0,
                    "localized_delta_q": 1.0,
                    "localized_rank": 2,
                },
            ]
        )
    summary = module._candidate_summary(pd.DataFrame(rows))
    localized = summary[summary["approach"] == "localized"]

    assert set(localized["candidate_eval_mode"]) == {"localized_label", "quotient_label"}
    assert localized["candidate_count"].tolist() == [2, 2]
