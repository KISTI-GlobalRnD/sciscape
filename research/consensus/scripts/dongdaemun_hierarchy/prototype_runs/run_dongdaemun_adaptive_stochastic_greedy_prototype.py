"""Run trace-only and apply-if-win adaptive stochastic greedy prototypes."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any
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


SCRIPT_DIR = Path(__file__).resolve().parent
from collect_dongdaemun_parent_local_replay_dataset import (  # noqa: E402
    DEFAULT_PARENT_ROWS,
    DEFAULT_RUNS_PATH,
    collect_parent_local_replay_dataset,
)

DEFAULT_OUTPUT_DIR = (
    Path("research/consensus/results/adaptive_refinement")
    / "dongdaemun_adaptive_stochastic_greedy_prototype_20260511"
)
SUMMARY_JSON_FILENAME = "adaptive_stochastic_greedy_prototype_summary.json"
CONSERVATIVE_SUMMARY_FILENAME = "conservative_policy_summary.csv"
CONSERVATIVE_REPORT_FILENAME = "conservative_policy_report.md"
SCHEMA_VERSION = 1
CONSERVATIVE_POLICIES = (
    {
        "name": "r05_total1_all",
        "min_gain_parent_weight": 0.5,
        "max_commits_total": 1,
        "max_commits_per_depth": 1,
        "commit_sources": (),
        "commit_strategy": "online_first",
    },
    {
        "name": "r10_total1_all",
        "min_gain_parent_weight": 1.0,
        "max_commits_total": 1,
        "max_commits_per_depth": 1,
        "commit_sources": (),
        "commit_strategy": "online_first",
    },
    {
        "name": "r10_total2_all",
        "min_gain_parent_weight": 1.0,
        "max_commits_total": 2,
        "max_commits_per_depth": 1,
        "commit_sources": (),
        "commit_strategy": "online_first",
    },
    {
        "name": "r10_total1_same",
        "min_gain_parent_weight": 1.0,
        "max_commits_total": 1,
        "max_commits_per_depth": 1,
        "commit_sources": ("same_gamma_probe",),
        "commit_strategy": "online_first",
    },
    {
        "name": "r10_total1_node",
        "min_gain_parent_weight": 1.0,
        "max_commits_total": 1,
        "max_commits_per_depth": 1,
        "commit_sources": ("node_order_control",),
        "commit_strategy": "online_first",
    },
    {
        "name": "r05_total1_all_best_qf",
        "min_gain_parent_weight": 0.5,
        "max_commits_total": 1,
        "max_commits_per_depth": 1,
        "commit_sources": (),
        "commit_strategy": "best_qf",
    },
    {
        "name": "r05_total1_all_risk",
        "min_gain_parent_weight": 0.5,
        "max_commits_total": 1,
        "max_commits_per_depth": 1,
        "commit_sources": (),
        "commit_strategy": "risk_adjusted",
    },
    {
        "name": "r10_total1_node_best_qf",
        "min_gain_parent_weight": 1.0,
        "max_commits_total": 1,
        "max_commits_per_depth": 1,
        "commit_sources": ("node_order_control",),
        "commit_strategy": "best_qf",
    },
    {
        "name": "r10_total1_node_risk",
        "min_gain_parent_weight": 1.0,
        "max_commits_total": 1,
        "max_commits_per_depth": 1,
        "commit_sources": ("node_order_control",),
        "commit_strategy": "risk_adjusted",
    },
)

def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))

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
        writer.writerows(rows)

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

def _summarize_against_trace(
    *, policy_name: str, trace_run_rows_path: Path, policy_run_rows_path: Path
) -> dict[str, Any]:
    trace_by_run = {row["run_id"]: row for row in _read_csv(trace_run_rows_path)}
    policy_by_run = {row["run_id"]: row for row in _read_csv(policy_run_rows_path)}
    common_ids = sorted(set(trace_by_run) & set(policy_by_run))
    quality_deltas: list[float] = []
    trace_elapsed = 0.0
    policy_elapsed = 0.0
    n_above_delta = 0
    applied_parent_delta = 0
    for run_id in common_ids:
        trace = trace_by_run[run_id]
        policy = policy_by_run[run_id]
        quality_deltas.append(
            _float_value(policy.get("quality")) - _float_value(trace.get("quality"))
        )
        trace_elapsed += _float_value(trace.get("elapsed_sec"))
        policy_elapsed += _float_value(policy.get("elapsed_sec"))
        n_above_delta += _int_value(policy.get("n_above_max_doc_weight")) - _int_value(
            trace.get("n_above_max_doc_weight")
        )
        applied_parent_delta += _int_value(
            policy.get("applied_parent_count_total")
        ) - _int_value(trace.get("applied_parent_count_total"))
    return {
        "policy": policy_name,
        "runs": len(common_ids),
        "quality_delta_sum": sum(quality_deltas),
        "quality_delta_mean": None
        if not quality_deltas
        else sum(quality_deltas) / len(quality_deltas),
        "quality_wins": sum(delta > 1e-9 for delta in quality_deltas),
        "quality_losses": sum(delta < -1e-9 for delta in quality_deltas),
        "quality_equal": sum(abs(delta) <= 1e-9 for delta in quality_deltas),
        "elapsed_trace_sum": trace_elapsed,
        "elapsed_policy_sum": policy_elapsed,
        "elapsed_ratio_total": None
        if trace_elapsed <= 0.0
        else policy_elapsed / trace_elapsed,
        "n_above_delta_sum": n_above_delta,
        "applied_parent_count_delta_sum": applied_parent_delta,
    }

def _build_conservative_report(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Conservative Adaptive Probe Policy Summary",
        "",
        "| policy | runs | qf delta | wins/loss/equal | elapsed ratio | n above delta | applied parent delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {policy} | {runs} | {qf:.6g} | {wins}/{losses}/{equal} | {elapsed} | {above} | {applied} |".format(
                policy=row.get("policy"),
                runs=row.get("runs"),
                qf=float(row.get("quality_delta_sum") or 0.0),
                wins=row.get("quality_wins"),
                losses=row.get("quality_losses"),
                equal=row.get("quality_equal"),
                elapsed=""
                if row.get("elapsed_ratio_total") is None
                else f"{float(row['elapsed_ratio_total']):.4g}",
                above=row.get("n_above_delta_sum"),
                applied=row.get("applied_parent_count_delta_sum"),
            )
        )
    lines.append("")
    return "\n".join(lines)

def run_adaptive_stochastic_greedy_prototype(
    *,
    parent_rows_path: Path,
    runs_path: Path,
    output_dir: Path,
    execute: bool = False,
    max_trigger_rows: int = 150,
    random_seed: int = 13,
    adaptive_probe_perturbations: int = 5,
    adaptive_probe_tolerance_parent_weight: float = 1e-6,
    include_node_order_control: bool = True,
    apply_execution_target_subset: str = "trigger",
    lift_threshold: float = 3.0,
) -> dict[str, Any]:
    trace_only = collect_parent_local_replay_dataset(
        parent_rows_path=parent_rows_path,
        runs_path=runs_path,
        output_dir=output_dir / "trace_only",
        execute=execute,
        max_trigger_rows=max_trigger_rows,
        random_seed=random_seed,
        adaptive_probe_mode="trace_only",
        adaptive_probe_perturbations=adaptive_probe_perturbations,
        adaptive_probe_tolerance_parent_weight=adaptive_probe_tolerance_parent_weight,
        include_node_order_control=include_node_order_control,
        execution_target_subset="all",
        lift_threshold=lift_threshold,
    )
    apply_if_win = collect_parent_local_replay_dataset(
        parent_rows_path=parent_rows_path,
        runs_path=runs_path,
        output_dir=output_dir / "apply_if_win",
        execute=execute,
        max_trigger_rows=max_trigger_rows,
        random_seed=random_seed,
        adaptive_probe_mode="apply_if_win",
        adaptive_probe_perturbations=adaptive_probe_perturbations,
        adaptive_probe_tolerance_parent_weight=adaptive_probe_tolerance_parent_weight,
        include_node_order_control=include_node_order_control,
        execution_target_subset=apply_execution_target_subset,
        lift_threshold=lift_threshold,
    )
    conservative_summaries: dict[str, str] = {}
    conservative_rows: list[dict[str, Any]] = []
    trace_run_rows_path = Path(trace_only["paths"]["run_rows"])
    for policy in CONSERVATIVE_POLICIES:
        policy_output_dir = output_dir / f"conservative_{policy['name']}"
        policy_result = collect_parent_local_replay_dataset(
            parent_rows_path=parent_rows_path,
            runs_path=runs_path,
            output_dir=policy_output_dir,
            execute=execute,
            max_trigger_rows=max_trigger_rows,
            random_seed=random_seed,
            adaptive_probe_mode="conservative_apply",
            adaptive_probe_perturbations=adaptive_probe_perturbations,
            adaptive_probe_tolerance_parent_weight=adaptive_probe_tolerance_parent_weight,
            include_node_order_control=include_node_order_control,
            adaptive_probe_commit_min_gain_parent_weight=float(
                policy["min_gain_parent_weight"]
            ),
            adaptive_probe_max_commits_total=int(policy["max_commits_total"]),
            adaptive_probe_max_commits_per_depth=int(policy["max_commits_per_depth"]),
            adaptive_probe_commit_sources=tuple(policy["commit_sources"]),
            adaptive_probe_commit_strategy=str(policy["commit_strategy"]),
            execution_target_subset="trigger",
            lift_threshold=lift_threshold,
        )
        conservative_summaries[str(policy["name"])] = policy_result["paths"][
            "summary_json"
        ]
        conservative_rows.append(
            {
                **policy,
                **_summarize_against_trace(
                    policy_name=str(policy["name"]),
                    trace_run_rows_path=trace_run_rows_path,
                    policy_run_rows_path=Path(policy_result["paths"]["run_rows"]),
                ),
                "summary_json": policy_result["paths"]["summary_json"],
            }
        )
    conservative_summary_path = output_dir / CONSERVATIVE_SUMMARY_FILENAME
    conservative_report_path = output_dir / CONSERVATIVE_REPORT_FILENAME
    _write_csv(conservative_summary_path, conservative_rows)
    conservative_report_path.write_text(
        _build_conservative_report(conservative_rows),
        encoding="utf-8",
    )
    payload = {
        "schema": "dongdaemun_adaptive_stochastic_greedy_prototype.v1",
        "schema_version": SCHEMA_VERSION,
        "execute": bool(execute),
        "trace_only_summary": trace_only["paths"]["summary_json"],
        "apply_if_win_summary": apply_if_win["paths"]["summary_json"],
        "lift_threshold": float(lift_threshold),
        "adaptive_probe_perturbations": int(adaptive_probe_perturbations),
        "adaptive_probe_tolerance_parent_weight": float(
            adaptive_probe_tolerance_parent_weight
        ),
        "include_node_order_control": bool(include_node_order_control),
        "apply_execution_target_subset": apply_execution_target_subset,
        "conservative_summaries": conservative_summaries,
        "conservative_policy_summary": str(conservative_summary_path),
        "conservative_policy_report": str(conservative_report_path),
    }
    summary_path = output_dir / SUMMARY_JSON_FILENAME
    _write_json(summary_path, payload)
    payload["paths"] = {"summary_json": str(summary_path)}
    _write_json(summary_path, payload)
    return payload

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-rows", type=Path, default=DEFAULT_PARENT_ROWS)
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-trigger-rows", type=int, default=150)
    parser.add_argument("--random-seed", type=int, default=13)
    parser.add_argument("--adaptive-probe-perturbations", type=int, default=5)
    parser.add_argument("--adaptive-probe-tolerance-parent-weight", type=float, default=1e-6)
    parser.add_argument("--no-node-order-control", action="store_true")
    parser.add_argument(
        "--apply-execution-target-subset",
        choices=("all", "trigger", "random_matched"),
        default="trigger",
    )
    parser.add_argument("--lift-threshold", type=float, default=3.0)
    return parser

def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    payload = run_adaptive_stochastic_greedy_prototype(
        parent_rows_path=args.parent_rows,
        runs_path=args.runs,
        output_dir=args.output_dir,
        execute=bool(args.execute),
        max_trigger_rows=int(args.max_trigger_rows),
        random_seed=int(args.random_seed),
        adaptive_probe_perturbations=int(args.adaptive_probe_perturbations),
        adaptive_probe_tolerance_parent_weight=float(
            args.adaptive_probe_tolerance_parent_weight
        ),
        include_node_order_control=not bool(args.no_node_order_control),
        apply_execution_target_subset=str(args.apply_execution_target_subset),
        lift_threshold=float(args.lift_threshold),
    )
    print(f"Saved prototype summary to {payload['paths']['summary_json']}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
