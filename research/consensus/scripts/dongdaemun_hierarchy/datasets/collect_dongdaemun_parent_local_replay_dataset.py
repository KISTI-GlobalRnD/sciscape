"""Collect parent-local adaptive greedy replay labels.

The dataset tests whether existing instability triggers predict profitable
local stochastic wins.  It compares trigger rows against a matched random
stable subset and records adaptive probe candidates emitted by Rust.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable
import sys

REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").exists()
)
SCRIPT_ROOT = REPO_ROOT / "research/consensus/scripts"
_SCRIPT_PATHS = [REPO_ROOT, SCRIPT_ROOT]
_SCRIPT_PATHS.extend(path for path in SCRIPT_ROOT.rglob("*") if path.is_dir())
for _script_path in reversed(_SCRIPT_PATHS):
    _script_path_str = str(_script_path)
    if _script_path_str not in sys.path:
        sys.path.insert(0, _script_path_str)


import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
import evaluate_dongdaemun_refinement_slice4 as pilot  # noqa: E402
import run_dongdaemun_refinement_slice4_quality_sweep as sweep  # noqa: E402

DEFAULT_PARENT_ROWS = (
    Path("research/consensus/results/adaptive_refinement")
    / "dongdaemun_instability_triggers_20260511"
    / "representative_sources"
    / "instability_parent_rows.csv"
)
DEFAULT_RUNS_PATH = (
    Path("research/consensus/results/adaptive_refinement")
    / "dongdaemun_refinement_qs_profile"
    / "selective_benchmark_20260508"
    / "representative_sources_p4_c16_current_sp0_1_2_policy_core"
    / "candidate_trace_runs.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    Path("research/consensus/results/adaptive_refinement")
    / "dongdaemun_parent_local_replay_dataset_20260511"
)

TARGETS_FILENAME = "parent_local_replay_targets.csv"
ROWS_FILENAME = "parent_local_replay_rows.csv"
RUN_ROWS_FILENAME = "parent_local_replay_run_rows.csv"
SUMMARY_FILENAME = "parent_local_replay_summary.csv"
GAIN_FILENAME = "parent_local_replay_gain_distribution.csv"
REPORT_FILENAME = "parent_local_replay_report.md"
SUMMARY_JSON_FILENAME = "parent_local_replay_summary.json"
TRACE_FILENAME = "parent_local_replay_trace.jsonl"
SCHEMA_VERSION = 1

def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))

def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fieldnames})

def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value

def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes"}

def _float_value(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default

def _int_value(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default

def _target_key(row: dict[str, Any]) -> tuple[str, int, int, int]:
    return (
        str(row.get("run_id")),
        _int_value(row.get("depth")),
        _int_value(row.get("parent_id")),
        _int_value(row.get("parent_visit_index"), 1),
    )

def _target_string(row: dict[str, Any]) -> str:
    return "{depth}:{parent}:{visit}".format(
        depth=_int_value(row.get("depth")),
        parent=_int_value(row.get("parent_id")),
        visit=_int_value(row.get("parent_visit_index"), 1),
    )

def _match_key(row: dict[str, Any]) -> tuple[Any, ...]:
    weight = _float_value(row.get("parent_weight"))
    if weight <= 0:
        weight_bin = "unknown"
    else:
        weight_bin = int(math.log10(max(1.0, weight)) * 2)
    return (
        row.get("sample"),
        row.get("variant"),
        row.get("config_id"),
        row.get("seed_perturbations"),
        row.get("candidate_quality_policy"),
        weight_bin,
    )

def select_replay_targets(
    parent_rows: list[dict[str, Any]],
    *,
    max_trigger_rows: int = 150,
    random_seed: int = 13,
) -> list[dict[str, Any]]:
    rng = random.Random(int(random_seed))
    eligible = [
        row
        for row in parent_rows
        if row.get("run_id") and _int_value(row.get("n_profiles")) > 0
    ]
    trigger_rows = [row for row in eligible if _bool_value(row.get("unstable"))]
    stable_rows = [row for row in eligible if not _bool_value(row.get("unstable"))]
    trigger_rows = sorted(
        trigger_rows,
        key=lambda row: (
            str(row.get("run_id")),
            _int_value(row.get("parent_id")),
            _int_value(row.get("parent_visit_index")),
        ),
    )[: int(max_trigger_rows)]

    stable_by_key: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in stable_rows:
        stable_by_key[_match_key(row)].append(row)
    for rows in stable_by_key.values():
        rng.shuffle(rows)
    stable_fallback = stable_rows[:]
    rng.shuffle(stable_fallback)

    selected_stable: list[dict[str, Any]] = []
    used: set[tuple[str, int, int, int]] = set()
    for trigger in trigger_rows:
        bucket = stable_by_key.get(_match_key(trigger), [])
        candidate = None
        while bucket and candidate is None:
            item = bucket.pop()
            if _target_key(item) not in used:
                candidate = item
        while stable_fallback and candidate is None:
            item = stable_fallback.pop()
            if _target_key(item) not in used:
                candidate = item
        if candidate is not None:
            used.add(_target_key(candidate))
            selected_stable.append(candidate)

    targets: list[dict[str, Any]] = []
    for label, rows in (("trigger", trigger_rows), ("random_matched", selected_stable)):
        for row in rows:
            target = dict(row)
            target["target_subset"] = label
            target["probe_target"] = _target_string(row)
            targets.append(target)
    return targets

def _quantile(values: list[float], q: float) -> float | None:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return None
    if len(finite) == 1:
        return finite[0]
    pos = (len(finite) - 1) * float(q)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return finite[lo]
    frac = pos - lo
    return finite[lo] * (1.0 - frac) + finite[hi] * frac

def build_replay_rows(
    targets: list[dict[str, Any]],
    probe_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    events_by_key: dict[tuple[str, int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for event in probe_events:
        if event.get("event") != "adaptive_probe_candidate":
            continue
        events_by_key[_target_key(event)].append(event)

    rows: list[dict[str, Any]] = []
    for target in targets:
        events = events_by_key.get(_target_key(target), [])
        gains = [_float_value(event.get("gain_vs_baseline")) for event in events]
        win_events = [event for event in events if _bool_value(event.get("local_win"))]
        source_counts: dict[str, int] = defaultdict(int)
        for event in events:
            source_counts[str(event.get("source"))] += 1
        row = {
            "target_subset": target.get("target_subset"),
            "run_id": target.get("run_id"),
            "depth": _int_value(target.get("depth")),
            "parent_id": _int_value(target.get("parent_id")),
            "parent_visit_index": _int_value(target.get("parent_visit_index"), 1),
            "probe_target": target.get("probe_target") or _target_string(target),
            "sample": target.get("sample"),
            "variant": target.get("variant"),
            "config_id": target.get("config_id"),
            "seed_perturbations": target.get("seed_perturbations"),
            "candidate_quality_policy": target.get("candidate_quality_policy"),
            "parent_weight": _float_value(target.get("parent_weight")),
            "trigger_unstable": _bool_value(target.get("unstable")),
            "trigger_reasons": target.get("unstable_reasons"),
            "n_probe_events": len(events),
            "n_local_win_events": len(win_events),
            "local_win": bool(win_events),
            "best_probe_gain": None if not gains else max(gains),
            "best_probe_source": None
            if not events
            else max(events, key=lambda event: _float_value(event.get("gain_vs_baseline"))).get(
                "source"
            ),
            "probe_source_counts": dict(sorted(source_counts.items())),
        }
        rows.append(row)
    return rows

def summarize_replay_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    gain_rows: list[dict[str, Any]] = []
    by_subset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_subset[str(row.get("target_subset"))].append(row)
    rates: dict[str, float] = {}
    for subset, subset_rows in sorted(by_subset.items()):
        wins = [row for row in subset_rows if _bool_value(row.get("local_win"))]
        gains = [_float_value(row.get("best_probe_gain")) for row in wins]
        win_rate = 0.0 if not subset_rows else len(wins) / len(subset_rows)
        rates[subset] = win_rate
        summary_rows.append(
            {
                "target_subset": subset,
                "n_targets": len(subset_rows),
                "n_with_probe_events": sum(_int_value(row.get("n_probe_events")) > 0 for row in subset_rows),
                "n_local_win": len(wins),
                "local_win_rate": win_rate,
                "mean_gain_per_win": None if not gains else sum(gains) / len(gains),
                "max_gain": None if not gains else max(gains),
            }
        )
        gain_rows.append(
            {
                "target_subset": subset,
                "p50_gain_per_win": _quantile(gains, 0.50),
                "p90_gain_per_win": _quantile(gains, 0.90),
                "p99_gain_per_win": _quantile(gains, 0.99),
            }
        )
    trigger_rate = rates.get("trigger", 0.0)
    random_rate = rates.get("random_matched", 0.0)
    lift = None if random_rate <= 0.0 else trigger_rate / random_rate
    summary_rows.append(
        {
            "target_subset": "lift",
            "n_targets": None,
            "n_with_probe_events": None,
            "n_local_win": None,
            "local_win_rate": lift,
            "mean_gain_per_win": None,
            "max_gain": None,
        }
    )
    return summary_rows, gain_rows

def _build_report(
    *,
    rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    gain_rows: list[dict[str, Any]],
    lift_threshold: float,
) -> str:
    lift_row = next((row for row in summary_rows if row["target_subset"] == "lift"), {})
    lift = lift_row.get("local_win_rate")
    passed = lift is not None and float(lift) >= float(lift_threshold)
    lines = [
        "# Dongdaemun Parent-Local Replay Dataset",
        "",
        f"- Targets: {len(rows)}",
        f"- Lift threshold: {lift_threshold:g}",
        f"- Lift passed: {passed}",
        "",
        "## Summary",
        "",
        "| subset | targets | with probes | wins | win rate / lift | mean gain | max gain |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| {subset} | {targets} | {probes} | {wins} | {rate} | {mean} | {max_gain} |".format(
                subset=row.get("target_subset"),
                targets="" if row.get("n_targets") is None else row.get("n_targets"),
                probes="" if row.get("n_with_probe_events") is None else row.get("n_with_probe_events"),
                wins="" if row.get("n_local_win") is None else row.get("n_local_win"),
                rate="" if row.get("local_win_rate") is None else f"{float(row['local_win_rate']):.4g}",
                mean="" if row.get("mean_gain_per_win") is None else f"{float(row['mean_gain_per_win']):.4g}",
                max_gain="" if row.get("max_gain") is None else f"{float(row['max_gain']):.4g}",
            )
        )
    lines.extend(["", "## Gain Distribution", "", "| subset | p50 | p90 | p99 |", "| --- | ---: | ---: | ---: |"])
    for row in gain_rows:
        lines.append(
            "| {subset} | {p50} | {p90} | {p99} |".format(
                subset=row.get("target_subset"),
                p50="" if row.get("p50_gain_per_win") is None else f"{float(row['p50_gain_per_win']):.4g}",
                p90="" if row.get("p90_gain_per_win") is None else f"{float(row['p90_gain_per_win']):.4g}",
                p99="" if row.get("p99_gain_per_win") is None else f"{float(row['p99_gain_per_win']):.4g}",
            )
        )
    lines.append("")
    return "\n".join(lines)

def _run_metadata_by_id(runs_path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["run_id"]): row for row in _read_jsonl(runs_path) if row.get("run_id")}

def _execute_probe_runs(
    *,
    targets: list[dict[str, Any]],
    run_metadata: dict[str, dict[str, Any]],
    output_dir: Path,
    adaptive_probe_mode: str,
    adaptive_probe_perturbations: int,
    adaptive_probe_tolerance_parent_weight: float,
    include_node_order_control: bool,
    adaptive_probe_commit_min_gain_parent_weight: float,
    adaptive_probe_max_commits_total: int,
    adaptive_probe_max_commits_per_depth: int,
    adaptive_probe_commit_sources: tuple[str, ...],
    adaptive_probe_commit_strategy: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    trace_path = output_dir / TRACE_FILENAME
    if trace_path.exists():
        trace_path.unlink()
    targets_by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for target in targets:
        targets_by_run[str(target.get("run_id"))].append(target)

    run_rows: list[dict[str, Any]] = []
    for run_id, run_targets in sorted(targets_by_run.items()):
        metadata = run_metadata.get(run_id)
        if metadata is None:
            continue
        input_cfg = pilot._resolve_input_from_summary(
            Path(metadata["summary_path"]),
            sample=metadata.get("sample"),
            seed=_int_value(metadata.get("seed"), 42),
        )
        n_nodes = pilot._infer_n_nodes(input_cfg)
        node_weights = pilot._load_node_weights(input_cfg.node_weights_path, n_nodes)
        graph = pilot._load_graph(input_cfg, node_weights)
        run_config = pilot.Slice4RunConfig(
            gamma_multipliers=tuple(float(x) for x in metadata.get("gamma_multipliers", [])),
            seed_perturbations=_int_value(metadata.get("seed_perturbations")),
            max_extra_parents_per_iteration=_int_value(
                metadata.get("max_extra_parents_per_iteration"), 16
            ),
            max_extra_children_per_parent=_int_value(
                metadata.get("max_extra_children_per_parent"), 64
            ),
            parent_selection_policy=str(metadata.get("parent_selection_policy") or "weight"),
            candidate_quality_policy=str(metadata.get("candidate_quality_policy") or "structural"),
            min_candidate_delta_q=_float_value(metadata.get("min_candidate_delta_q")),
            adaptive_plateau_quality_band=_float_value(
                metadata.get("adaptive_plateau_quality_band")
            ),
            use_final_quality_guard=_bool_value(metadata.get("use_final_quality_guard")),
            min_final_quality_delta=_float_value(metadata.get("min_final_quality_delta")),
            baseline_repair_policy=str(metadata.get("baseline_repair_policy") or "replace"),
            baseline_repair_replace_min_parent_ratio=_float_value(
                metadata.get("baseline_repair_replace_min_parent_ratio"), 1.05
            ),
            adaptive_probe_mode=adaptive_probe_mode,
            adaptive_probe_perturbations=int(adaptive_probe_perturbations),
            adaptive_probe_targets=tuple(_target_string(target) for target in run_targets),
            adaptive_probe_tolerance_parent_weight=float(
                adaptive_probe_tolerance_parent_weight
            ),
            adaptive_probe_include_node_order_control=bool(include_node_order_control),
            adaptive_probe_commit_min_gain_parent_weight=float(
                adaptive_probe_commit_min_gain_parent_weight
            ),
            adaptive_probe_max_commits_total=int(adaptive_probe_max_commits_total),
            adaptive_probe_max_commits_per_depth=int(
                adaptive_probe_max_commits_per_depth
            ),
            adaptive_probe_commit_sources=tuple(adaptive_probe_commit_sources),
            adaptive_probe_commit_strategy=str(adaptive_probe_commit_strategy),
        )
        probe_run_id = run_id
        with sweep._candidate_trace_path_context(trace_path, explicit=True, resume=True):
            with sweep._candidate_trace_context(probe_run_id):
                row, _membership = pilot._run_variant(
                    graph=graph,
                    input_cfg=input_cfg,
                    run_config=run_config,
                    node_weights=node_weights,
                    variant=str(metadata.get("variant") or pilot.VARIANT_REPAIR_OFF),
                    standard_membership=None,
                    standard_quality=None,
                )
        row.update(
            {
                "run_id": run_id,
                "config_id": metadata.get("config_id"),
                "seed": _int_value(metadata.get("seed"), 42),
                "seed_perturbations": _int_value(metadata.get("seed_perturbations")),
                "adaptive_probe_mode": adaptive_probe_mode,
                "adaptive_probe_perturbations": int(adaptive_probe_perturbations),
                "adaptive_probe_include_node_order_control": bool(
                    include_node_order_control
                ),
                "adaptive_probe_tolerance_parent_weight": float(
                    adaptive_probe_tolerance_parent_weight
                ),
                "adaptive_probe_commit_min_gain_parent_weight": float(
                    adaptive_probe_commit_min_gain_parent_weight
                ),
                "adaptive_probe_max_commits_total": int(adaptive_probe_max_commits_total),
                "adaptive_probe_max_commits_per_depth": int(
                    adaptive_probe_max_commits_per_depth
                ),
                "adaptive_probe_commit_sources": tuple(adaptive_probe_commit_sources),
                "adaptive_probe_commit_strategy": str(adaptive_probe_commit_strategy),
                "n_probe_targets": len(run_targets),
                "n_trigger_targets": sum(
                    1 for target in run_targets if target.get("target_subset") == "trigger"
                ),
                "n_random_matched_targets": sum(
                    1
                    for target in run_targets
                    if target.get("target_subset") == "random_matched"
                ),
            }
        )
        run_rows.append(row)
    return _read_jsonl(trace_path), run_rows

def collect_parent_local_replay_dataset(
    *,
    parent_rows_path: Path,
    runs_path: Path,
    output_dir: Path,
    execute: bool = False,
    max_trigger_rows: int = 150,
    random_seed: int = 13,
    adaptive_probe_mode: str = "trace_only",
    adaptive_probe_perturbations: int = 5,
    adaptive_probe_tolerance_parent_weight: float = 1e-6,
    include_node_order_control: bool = True,
    adaptive_probe_commit_min_gain_parent_weight: float = 0.0,
    adaptive_probe_max_commits_total: int = 0,
    adaptive_probe_max_commits_per_depth: int = 0,
    adaptive_probe_commit_sources: tuple[str, ...] = (),
    adaptive_probe_commit_strategy: str = "online_first",
    execution_target_subset: str = "all",
    lift_threshold: float = 3.0,
) -> dict[str, Any]:
    parent_rows = _read_csv(parent_rows_path)
    targets = select_replay_targets(
        parent_rows,
        max_trigger_rows=max_trigger_rows,
        random_seed=random_seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    run_metadata = _run_metadata_by_id(runs_path)
    if execution_target_subset == "all":
        execution_targets = targets
    else:
        execution_targets = [
            target
            for target in targets
            if target.get("target_subset") == execution_target_subset
        ]
    run_rows: list[dict[str, Any]]
    if execute:
        probe_events, run_rows = _execute_probe_runs(
            targets=execution_targets,
            run_metadata=run_metadata,
            output_dir=output_dir,
            adaptive_probe_mode=adaptive_probe_mode,
            adaptive_probe_perturbations=adaptive_probe_perturbations,
            adaptive_probe_tolerance_parent_weight=adaptive_probe_tolerance_parent_weight,
            include_node_order_control=include_node_order_control,
            adaptive_probe_commit_min_gain_parent_weight=adaptive_probe_commit_min_gain_parent_weight,
            adaptive_probe_max_commits_total=adaptive_probe_max_commits_total,
            adaptive_probe_max_commits_per_depth=adaptive_probe_max_commits_per_depth,
            adaptive_probe_commit_sources=tuple(adaptive_probe_commit_sources),
            adaptive_probe_commit_strategy=adaptive_probe_commit_strategy,
        )
    else:
        probe_events = _read_jsonl(output_dir / TRACE_FILENAME)
        run_rows = _read_csv(output_dir / RUN_ROWS_FILENAME)
    rows = build_replay_rows(targets, probe_events)
    summary_rows, gain_rows = summarize_replay_rows(rows)

    targets_path = output_dir / TARGETS_FILENAME
    rows_path = output_dir / ROWS_FILENAME
    run_rows_path = output_dir / RUN_ROWS_FILENAME
    summary_path = output_dir / SUMMARY_FILENAME
    gain_path = output_dir / GAIN_FILENAME
    report_path = output_dir / REPORT_FILENAME
    summary_json_path = output_dir / SUMMARY_JSON_FILENAME
    _write_csv(targets_path, targets)
    _write_csv(rows_path, rows)
    _write_csv(run_rows_path, run_rows)
    _write_csv(summary_path, summary_rows)
    _write_csv(gain_path, gain_rows)
    report_path.write_text(
        _build_report(
            rows=rows,
            summary_rows=summary_rows,
            gain_rows=gain_rows,
            lift_threshold=lift_threshold,
        ),
        encoding="utf-8",
    )
    payload = {
        "schema": "dongdaemun_parent_local_replay_dataset.v1",
        "schema_version": SCHEMA_VERSION,
        "parent_rows_path": str(parent_rows_path),
        "runs_path": str(runs_path),
        "execute": bool(execute),
        "n_targets": len(targets),
        "n_execution_targets": len(execution_targets),
        "execution_target_subset": execution_target_subset,
        "adaptive_probe_commit_min_gain_parent_weight": float(
            adaptive_probe_commit_min_gain_parent_weight
        ),
        "adaptive_probe_max_commits_total": int(adaptive_probe_max_commits_total),
        "adaptive_probe_max_commits_per_depth": int(
            adaptive_probe_max_commits_per_depth
        ),
        "adaptive_probe_commit_sources": tuple(adaptive_probe_commit_sources),
        "adaptive_probe_commit_strategy": str(adaptive_probe_commit_strategy),
        "n_run_rows": len(run_rows),
        "n_probe_events": sum(_int_value(row.get("n_probe_events")) for row in rows),
        "paths": {
            "targets": str(targets_path),
            "rows": str(rows_path),
            "run_rows": str(run_rows_path),
            "summary": str(summary_path),
            "gain_distribution": str(gain_path),
            "report": str(report_path),
            "summary_json": str(summary_json_path),
            "trace": str(output_dir / TRACE_FILENAME),
        },
    }
    _write_json(summary_json_path, payload)
    return payload

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-rows", type=Path, default=DEFAULT_PARENT_ROWS)
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-trigger-rows", type=int, default=150)
    parser.add_argument("--random-seed", type=int, default=13)
    parser.add_argument("--adaptive-probe-mode", default="trace_only")
    parser.add_argument("--adaptive-probe-perturbations", type=int, default=5)
    parser.add_argument("--adaptive-probe-tolerance-parent-weight", type=float, default=1e-6)
    parser.add_argument("--no-node-order-control", action="store_true")
    parser.add_argument("--adaptive-probe-commit-min-gain-parent-weight", type=float, default=0.0)
    parser.add_argument("--adaptive-probe-max-commits-total", type=int, default=0)
    parser.add_argument("--adaptive-probe-max-commits-per-depth", type=int, default=0)
    parser.add_argument(
        "--adaptive-probe-commit-sources",
        default="",
        help="Comma-separated conservative commit sources: same_gamma_probe,node_order_control",
    )
    parser.add_argument(
        "--adaptive-probe-commit-strategy",
        choices=("online_first", "best_qf", "risk_adjusted"),
        default="online_first",
    )
    parser.add_argument(
        "--execution-target-subset",
        choices=("all", "trigger", "random_matched"),
        default="all",
    )
    parser.add_argument("--lift-threshold", type=float, default=3.0)
    return parser

def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    payload = collect_parent_local_replay_dataset(
        parent_rows_path=args.parent_rows,
        runs_path=args.runs,
        output_dir=args.output_dir,
        execute=bool(args.execute),
        max_trigger_rows=int(args.max_trigger_rows),
        random_seed=int(args.random_seed),
        adaptive_probe_mode=str(args.adaptive_probe_mode),
        adaptive_probe_perturbations=int(args.adaptive_probe_perturbations),
        adaptive_probe_tolerance_parent_weight=float(
            args.adaptive_probe_tolerance_parent_weight
        ),
        include_node_order_control=not bool(args.no_node_order_control),
        adaptive_probe_commit_min_gain_parent_weight=float(
            args.adaptive_probe_commit_min_gain_parent_weight
        ),
        adaptive_probe_max_commits_total=int(args.adaptive_probe_max_commits_total),
        adaptive_probe_max_commits_per_depth=int(
            args.adaptive_probe_max_commits_per_depth
        ),
        adaptive_probe_commit_sources=tuple(
            source.strip()
            for source in str(args.adaptive_probe_commit_sources).split(",")
            if source.strip()
        ),
        adaptive_probe_commit_strategy=str(args.adaptive_probe_commit_strategy),
        execution_target_subset=str(args.execution_target_subset),
        lift_threshold=float(args.lift_threshold),
    )
    print(f"Saved parent-local replay dataset to {payload['paths']['summary_json']}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
