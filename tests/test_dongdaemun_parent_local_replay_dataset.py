"""Tests for parent-local replay dataset collection."""

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
    / "datasets"
    / "collect_dongdaemun_parent_local_replay_dataset.py"
)


def _load_module():
    if str(SCRIPT_PATH.parent) not in sys.path:
        sys.path.insert(0, str(SCRIPT_PATH.parent))
    spec = importlib.util.spec_from_file_location(
        "collect_dongdaemun_parent_local_replay_dataset_for_test",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _parent_row(
    *,
    run_id: str,
    parent_id: int,
    visit: int,
    unstable: bool,
    weight: float = 100.0,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "depth": 0,
        "parent_id": parent_id,
        "parent_visit_index": visit,
        "sample": "tiny",
        "variant": "refine_repair_off",
        "config_id": "cfg",
        "seed_perturbations": 1,
        "candidate_quality_policy": "quality_first",
        "parent_weight": weight,
        "n_profiles": 2,
        "unstable": unstable,
        "unstable_reasons": '["quality_pressure_disagree"]' if unstable else "[]",
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_select_replay_targets_pairs_trigger_with_matched_random():
    module = _load_module()
    rows = [
        _parent_row(run_id="r1", parent_id=1, visit=1, unstable=True),
        _parent_row(run_id="r1", parent_id=2, visit=1, unstable=False),
        _parent_row(run_id="r1", parent_id=3, visit=1, unstable=False),
    ]

    targets = module.select_replay_targets(rows, max_trigger_rows=10, random_seed=1)

    assert [row["target_subset"] for row in targets].count("trigger") == 1
    assert [row["target_subset"] for row in targets].count("random_matched") == 1
    assert all(row["probe_target"].count(":") == 2 for row in targets)


def test_replay_rows_compute_local_win_and_lift():
    module = _load_module()
    targets = [
        {**_parent_row(run_id="r1", parent_id=1, visit=1, unstable=True), "target_subset": "trigger"},
        {**_parent_row(run_id="r1", parent_id=2, visit=1, unstable=False), "target_subset": "random_matched"},
    ]
    for target in targets:
        target["probe_target"] = module._target_string(target)
    events = [
        {
            "event": "adaptive_probe_candidate",
            "run_id": "r1",
            "depth": 0,
            "parent_id": 1,
            "parent_visit_index": 1,
            "source": "same_gamma_probe",
            "gain_vs_baseline": 0.5,
            "local_win": True,
        },
        {
            "event": "adaptive_probe_candidate",
            "run_id": "r1",
            "depth": 0,
            "parent_id": 2,
            "parent_visit_index": 1,
            "source": "node_order_control",
            "gain_vs_baseline": -0.1,
            "local_win": False,
        },
    ]

    rows = module.build_replay_rows(targets, events)
    summary, gains = module.summarize_replay_rows(rows)

    assert rows[0]["local_win"] is True
    assert rows[0]["best_probe_source"] == "same_gamma_probe"
    assert rows[1]["local_win"] is False
    trigger = next(row for row in summary if row["target_subset"] == "trigger")
    random = next(row for row in summary if row["target_subset"] == "random_matched")
    assert trigger["local_win_rate"] == 1.0
    assert random["local_win_rate"] == 0.0
    trigger_gain = next(row for row in gains if row["target_subset"] == "trigger")
    assert trigger_gain["p50_gain_per_win"] == 0.5


def test_collect_replay_dataset_writes_outputs_without_execute(tmp_path):
    module = _load_module()
    parent_rows_path = tmp_path / "parents.csv"
    runs_path = tmp_path / "runs.jsonl"
    output_dir = tmp_path / "out"
    _write_csv(
        parent_rows_path,
        [
            _parent_row(run_id="r1", parent_id=1, visit=1, unstable=True),
            _parent_row(run_id="r1", parent_id=2, visit=1, unstable=False),
        ],
    )
    runs_path.write_text(
        json.dumps({"run_id": "r1", "summary_path": "missing.json"}) + "\n",
        encoding="utf-8",
    )

    payload = module.collect_parent_local_replay_dataset(
        parent_rows_path=parent_rows_path,
        runs_path=runs_path,
        output_dir=output_dir,
        execute=False,
        max_trigger_rows=10,
    )

    assert payload["n_targets"] == 2
    for path in payload["paths"].values():
        if path.endswith("parent_local_replay_trace.jsonl"):
            continue
        assert Path(path).exists()
    report = Path(payload["paths"]["report"]).read_text(encoding="utf-8")
    assert "Lift threshold" in report
