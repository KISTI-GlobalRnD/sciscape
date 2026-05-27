"""Tests for Dongdaemun fast/accurate mode tradeoff diagnostics."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "research" / "consensus" / "scripts"
ANALYSIS_PATH = SCRIPT_DIR / "analyze_leiden_two_mode_tradeoff.py"


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


def test_two_mode_rows_compare_fast_and_accurate_modes():
    module = _load_script("leiden_two_mode_tradeoff_for_test")
    candidates = pd.DataFrame(
        [
            _candidate(0, p1=30.0, p5=10.0, elapsed=10.0),
            _candidate(1, p1=20.0, p5=11.0, elapsed=20.0),
            _candidate(2, p1=10.0, p5=35.0, elapsed=30.0),
        ]
    )

    rows = module.build_two_mode_rows(candidates, material_regret_q=10.0)

    fast = rows[rows["mode_name"] == "fast_p1"].iloc[0]
    accurate = rows[rows["mode_name"] == "accurate_full_budget"].iloc[0]
    assert fast["mode_family"] == "fast"
    assert fast["quality_regret_q"] == 25.0
    assert bool(fast["material_regret"]) is True
    assert fast["estimated_p5_elapsed_ms"] == 10.0
    assert fast["elapsed_ratio_vs_accurate"] == 10.0 / 60.0
    assert accurate["mode_family"] == "accurate"
    assert accurate["quality_regret_q"] == 0.0
    assert bool(accurate["quality_first_hit"]) is True


def test_two_mode_summary_explains_both_mode_needs():
    module = _load_script("leiden_two_mode_need_for_test")
    candidates = pd.DataFrame(
        [
            _candidate(0, p1=30.0, p5=10.0, elapsed=10.0),
            _candidate(1, p1=20.0, p5=11.0, elapsed=20.0),
            _candidate(2, p1=10.0, p5=35.0, elapsed=30.0),
        ]
    )
    rows = module.build_two_mode_rows(candidates, material_regret_q=10.0)
    summary = module.build_two_mode_summary(rows)

    need_rows = module.build_mode_need_rows(summary)

    assert set(need_rows["need"]) >= {"fast_mode", "accurate_mode", "mode_pair"}
    fast_summary = summary[summary["mode_name"] == "fast_p1"].iloc[0]
    assert fast_summary["material_regret_count"] == 1
    assert fast_summary["quality_regret_q_sum"] == 25.0


def test_two_mode_report_is_written(tmp_path):
    module = _load_script("leiden_two_mode_report_for_test")
    candidates = pd.DataFrame(
        [
            _candidate(0, p1=10.0, p5=20.0, elapsed=10.0),
            _candidate(1, p1=9.0, p5=21.0, elapsed=10.0),
        ]
    )
    rows = module.build_two_mode_rows(candidates)
    summary = module.build_two_mode_summary(rows)
    need_rows = module.build_mode_need_rows(summary)

    module.write_report(tmp_path, rows, summary, need_rows)

    report = (tmp_path / "dongdaemun_two_mode_tradeoff_report.md").read_text(
        encoding="utf-8"
    )
    assert "Dongdaemun Two-Mode Tradeoff Review" in report
    assert "Fast mode is justified by cost" in report


def _candidate(
    candidate_index: int,
    *,
    p1: float,
    p5: float,
    elapsed: float,
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
        "p5_relative_delta_q_ppm": rel,
        "p5_elapsed_ms": elapsed,
        "p5_basin_signature": f"basin-{candidate_index}",
        "p5_changed_fraction_vs_baseline": 0.001 * (candidate_index + 1),
        "p5_changed_nodes_vs_baseline": candidate_index + 1,
        "p5_basin_sketch_node_hash": "sample-hash",
        "p5_basin_sketch_baseline_membership": "0;0;1;1",
        "p5_basin_sketch_membership": "0;0;1;1",
        "p5_basin_changed_support_node_count": 2,
        "p5_basin_changed_support_sketch_sample_size": 2,
        "p5_basin_changed_support_nodes": "0;1",
    }
