"""Tests for local Dongdaemun candidate beam analysis."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "research"
    / "consensus"
    / "scripts"
    / "dongdaemun_hierarchy"
    / "trajectory_analysis"
    / "analyze_dongdaemun_local_candidate_beam.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "analyze_dongdaemun_local_candidate_beam_for_test",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _profile(
    *,
    run_id: str,
    parent_id: int,
    candidate_id: int,
    quality: float,
    pressure: float,
    max_ratio: float,
    largest: float,
    n_clusters: int = 4,
    valid: bool = True,
    quality_passes: bool = True,
    decision: str = "policy_rejected",
) -> dict[str, object]:
    return {
        "event": "candidate_profile",
        "run_id": run_id,
        "depth": 0,
        "parent_id": parent_id,
        "candidate_id": candidate_id,
        "source": "same_gamma_seed" if candidate_id < 2 else "high_gamma",
        "source_index": candidate_id,
        "gamma_multiplier": 1.0 if candidate_id < 2 else 1.05,
        "repaired": False,
        "parent_size": 100,
        "parent_weight": 100.0,
        "standard_n_clusters": 8,
        "candidate_n_clusters": n_clusters,
        "standard_largest_child_fraction": 0.7,
        "largest_child_fraction": largest,
        "largest_child_fraction_improvement": 0.7 - largest,
        "standard_max_child_weight_ratio": 1.4,
        "candidate_max_child_weight_ratio": max_ratio,
        "pressure_reduction": pressure,
        "singleton_weight_fraction": 0.0,
        "candidate_delta_q": quality,
        "adaptive_diagnostic_score": pressure + quality / 100.0,
        "adaptive_quality_band": 0.2,
        "adaptive_plateau_compared": True,
        "quadrant": "qpos_spos",
        "valid": valid,
        "quality_passes": quality_passes,
        "decision": decision,
        "quotient_score": pressure,
        "baseline_repair_merge_count": 0,
        "baseline_repair_delta_sum": 0.0,
    }


def _decision(run_id: str, parent_id: int, candidate_id: int) -> dict[str, object]:
    return {
        "event": "candidate_decision",
        "run_id": run_id,
        "depth": 0,
        "parent_id": parent_id,
        "candidate_id": candidate_id,
        "decision": "selected_applied",
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_pressure_band_policy_selects_local_pressure_candidate(tmp_path):
    module = _load_module()
    run_id = "run-a"
    events = [
        _profile(
            run_id=run_id,
            parent_id=1,
            candidate_id=0,
            quality=10.0,
            pressure=0.2,
            max_ratio=1.2,
            largest=0.60,
            decision="selected_by_policy",
        ),
        _profile(
            run_id=run_id,
            parent_id=1,
            candidate_id=1,
            quality=9.9,
            pressure=1.1,
            max_ratio=0.7,
            largest=0.35,
        ),
        _profile(
            run_id=run_id,
            parent_id=1,
            candidate_id=2,
            quality=8.0,
            pressure=2.0,
            max_ratio=0.4,
            largest=0.20,
        ),
        _profile(
            run_id=run_id,
            parent_id=1,
            candidate_id=3,
            quality=20.0,
            pressure=2.0,
            max_ratio=0.4,
            largest=0.20,
            valid=False,
            quality_passes=False,
            decision="invalid",
        ),
        _decision(run_id, 1, 0),
    ]

    rows = module.build_parent_policy_rows(
        events=events,
        run_metadata={run_id: {"sample": "tiny", "candidate_quality_policy": "selective"}},
        policy_names=(
            "current_greedy",
            "quality_top1",
            "pressure_within_quality_band",
            "quality_top3_pressure",
            "balanced_norm_v1",
        ),
        beam_width=3,
        quality_band=0.2,
        pressure_weight=0.5,
    )

    by_policy = {row["policy_name"]: row for row in rows}
    assert by_policy["current_greedy"]["selected_candidate_id"] == 0
    assert by_policy["quality_top1"]["selected_candidate_id"] == 0
    assert by_policy["pressure_within_quality_band"]["selected_candidate_id"] == 1
    assert by_policy["quality_top3_pressure"]["selected_candidate_id"] == 1
    assert by_policy["balanced_norm_v1"]["selected_candidate_id"] == 1
    assert by_policy["pressure_within_quality_band"]["candidate_delta_q_vs_current"] < 0
    assert (
        by_policy["pressure_within_quality_band"][
            "pressure_reduction_delta_vs_current"
        ]
        > 0
    )
    assert by_policy["pressure_within_quality_band"]["n_selectable_candidates"] == 3


def test_seed_consensus_lite_prefers_repeated_structural_signature(tmp_path):
    module = _load_module()
    run_id = "run-b"
    events = [
        _profile(
            run_id=run_id,
            parent_id=2,
            candidate_id=0,
            quality=10.0,
            pressure=0.3,
            max_ratio=1.1,
            largest=0.50,
            n_clusters=5,
            decision="selected_by_policy",
        ),
        _profile(
            run_id=run_id,
            parent_id=2,
            candidate_id=1,
            quality=9.95,
            pressure=0.7,
            max_ratio=0.9,
            largest=0.40,
            n_clusters=4,
        ),
        _profile(
            run_id=run_id,
            parent_id=2,
            candidate_id=2,
            quality=9.94,
            pressure=0.8,
            max_ratio=0.8,
            largest=0.4002,
            n_clusters=4,
        ),
        _decision(run_id, 2, 0),
    ]

    choice = module.select_policy_candidate(
        "seed_consensus_lite",
        events[:-1],
        current=events[0],
        beam_width=5,
        quality_band=0.2,
        pressure_weight=0.35,
        signature_precision=2,
    )

    assert choice.selected["candidate_id"] == 2
    assert sorted(choice.retained_ids()) == [0, 1, 2]
    assert "selected_signature" in choice.reasons[2]


def test_repeated_parent_id_is_split_into_visit_blocks(tmp_path):
    module = _load_module()
    run_id = "run-repeat"
    events = [
        _profile(
            run_id=run_id,
            parent_id=7,
            candidate_id=0,
            quality=1.0,
            pressure=0.1,
            max_ratio=1.0,
            largest=0.5,
            decision="selected_by_policy",
        ),
        _decision(run_id, 7, 0),
        _profile(
            run_id=run_id,
            parent_id=7,
            candidate_id=0,
            quality=2.0,
            pressure=0.2,
            max_ratio=0.9,
            largest=0.4,
            decision="selected_by_policy",
        ),
        _decision(run_id, 7, 0),
    ]

    rows = module.build_parent_policy_rows(
        events=events,
        run_metadata={},
        policy_names=("current_greedy",),
    )

    assert len(rows) == 2
    assert [row["parent_visit_index"] for row in rows] == [1, 2]
    assert [row["current_candidate_delta_q"] for row in rows] == [1.0, 2.0]


def test_analyze_candidate_trace_writes_outputs_and_summary(tmp_path):
    module = _load_module()
    run_id = "run-c"
    trace_path = tmp_path / "candidate_trace.jsonl"
    runs_path = tmp_path / "candidate_trace_runs.jsonl"
    output_dir = tmp_path / "analysis"
    _write_jsonl(
        trace_path,
        [
            _profile(
                run_id=run_id,
                parent_id=1,
                candidate_id=0,
                quality=10.0,
                pressure=0.2,
                max_ratio=1.2,
                largest=0.60,
                decision="selected_by_policy",
            ),
            _profile(
                run_id=run_id,
                parent_id=1,
                candidate_id=1,
                quality=9.9,
                pressure=1.1,
                max_ratio=0.7,
                largest=0.35,
            ),
            _decision(run_id, 1, 0),
        ],
    )
    _write_jsonl(
        runs_path,
        [
            {
                "schema": "dongdaemun_refinement_candidate_trace_run.v1",
                "run_id": run_id,
                "sample": "tiny",
                "variant": "refine",
                "config_id": "local_beam",
                "gamma_preset": "mild",
                "seed_perturbations": 2,
                "parent_selection_policy": "weight",
                "candidate_quality_policy": "selective",
                "adaptive_plateau_quality_band": 0.2,
            }
        ],
    )

    payload = module.analyze_candidate_trace(
        trace_path=trace_path,
        runs_path=runs_path,
        output_dir=output_dir,
        policy_names=(
            "current_greedy",
            "pressure_within_quality_band",
            "mixed_local_beam_v1",
        ),
        beam_width=5,
        quality_band=0.2,
        pressure_weight=0.35,
    )

    assert payload["n_parent_policy_rows"] == 3
    for path in payload["paths"].values():
        assert Path(path).exists()
    assert "pressure_within_quality_band" in Path(
        payload["paths"]["report"]
    ).read_text(encoding="utf-8")

    with Path(payload["paths"]["missed_cases"]).open(encoding="utf-8") as fh:
        missed_rows = list(csv.DictReader(fh))
    assert any(
        row["policy_name"] == "pressure_within_quality_band"
        and row["selected_candidate_id"] == "1"
        for row in missed_rows
    )

    with Path(payload["paths"]["policy_summary"]).open(encoding="utf-8") as fh:
        summary_rows = list(csv.DictReader(fh))
    pressure_summary = next(
        row for row in summary_rows if row["policy_name"] == "pressure_within_quality_band"
    )
    assert pressure_summary["sample"] == "tiny"
    assert pressure_summary["n_differs_from_current"] == "1"
