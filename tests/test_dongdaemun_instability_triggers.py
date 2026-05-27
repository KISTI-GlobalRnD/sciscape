"""Tests for Dongdaemun instability trigger analysis."""

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
    / "analyze_dongdaemun_instability_triggers.py"
)


def _load_module():
    if str(SCRIPT_PATH.parent) not in sys.path:
        sys.path.insert(0, str(SCRIPT_PATH.parent))
    spec = importlib.util.spec_from_file_location(
        "analyze_dongdaemun_instability_triggers_for_test",
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


def test_build_instability_rows_flags_local_disagreement_and_retains_beam():
    module = _load_module()
    run_id = "run-unstable"
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
            quality=9.95,
            pressure=0.8,
            max_ratio=0.8,
            largest=0.35,
        ),
        _profile(
            run_id=run_id,
            parent_id=1,
            candidate_id=2,
            quality=9.85,
            pressure=1.5,
            max_ratio=0.5,
            largest=0.20,
            n_clusters=6,
        ),
        _decision(run_id, 1, 0),
    ]

    rows = module.build_instability_rows(
        events=events,
        run_metadata={run_id: {"sample": "tiny", "candidate_quality_policy": "selective"}},
        quality_margin_abs=0.2,
        quality_band_abs=0.2,
        beam_width=5,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["unstable"] is True
    assert "low_quality_margin" in row["unstable_reasons"]
    assert "quality_pressure_disagree" in row["unstable_reasons"]
    assert "signature_diverse_in_band" in row["unstable_reasons"]
    assert row["quality_top1_candidate_id"] == 0
    assert row["pressure_top1_candidate_id"] == 2
    assert sorted(row["retained_candidate_ids"]) == [0, 1, 2]
    assert row["estimated_extra_replays"] == 2


def test_stable_parent_visit_does_not_trigger_lookahead():
    module = _load_module()
    run_id = "run-stable"
    events = [
        _profile(
            run_id=run_id,
            parent_id=2,
            candidate_id=0,
            quality=10.0,
            pressure=2.0,
            max_ratio=0.6,
            largest=0.30,
            decision="selected_by_policy",
        ),
        _profile(
            run_id=run_id,
            parent_id=2,
            candidate_id=1,
            quality=7.0,
            pressure=0.2,
            max_ratio=1.2,
            largest=0.60,
        ),
        _decision(run_id, 2, 0),
    ]

    rows = module.build_instability_rows(
        events=events,
        run_metadata={},
        quality_margin_abs=0.2,
        quality_band_abs=0.2,
        beam_width=5,
    )

    assert len(rows) == 1
    assert rows[0]["unstable"] is False
    assert rows[0]["retained_candidate_ids"] == []
    assert rows[0]["estimated_extra_replays"] == 0


def test_analyze_instability_triggers_writes_outputs(tmp_path):
    module = _load_module()
    run_id = "run-output"
    trace_path = tmp_path / "candidate_trace.jsonl"
    runs_path = tmp_path / "candidate_trace_runs.jsonl"
    output_dir = tmp_path / "instability"
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
                quality=9.95,
                pressure=0.8,
                max_ratio=0.8,
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
                "config_id": "instability",
                "gamma_preset": "mild",
                "seed_perturbations": 2,
                "parent_selection_policy": "weight",
                "candidate_quality_policy": "selective",
                "adaptive_plateau_quality_band": 0.2,
            }
        ],
    )

    payload = module.analyze_instability_triggers(
        trace_path=trace_path,
        runs_path=runs_path,
        output_dir=output_dir,
        quality_margin_abs=0.2,
        quality_band_abs=0.2,
        beam_width=5,
    )

    assert payload["n_parent_rows"] == 1
    assert payload["n_unstable_parent_rows"] == 1
    for path in payload["paths"].values():
        assert Path(path).exists()
    assert "Unstable parent visits" in Path(payload["paths"]["report"]).read_text(
        encoding="utf-8"
    )

    with Path(payload["paths"]["lookahead_candidates"]).open(encoding="utf-8") as fh:
        lookahead_rows = list(csv.DictReader(fh))
    assert len(lookahead_rows) == 1
    assert lookahead_rows[0]["sample"] == "tiny"
    assert lookahead_rows[0]["quality_pressure_disagree"] == "True"
