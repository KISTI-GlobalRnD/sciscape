#!/usr/bin/env python3
"""Run resumable Leiden perturbation-portfolio monitor batches."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
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


import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
MONITOR_SCRIPT = SCRIPT_DIR / "run_leiden_hysteresis_work_acceleration_monitor.py"
DEFAULT_MANIFEST = (
    REPO_ROOT
    / "research/consensus/results/adaptive_refinement/"
    "leiden_hysteresis_exception_detector_graphs_20260514/graph_manifest.csv"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "research/consensus/results/adaptive_refinement/"
    "leiden_hysteresis_parallel_portfolio_batch_20260514"
)

def _parse_csv(value: str | None) -> list[str]:
    if value is None or not str(value).strip():
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]

def _parse_int_csv(value: str | None, *, default: list[int]) -> list[int]:
    parts = _parse_csv(value)
    if not parts:
        return list(default)
    return [int(part) for part in parts]

def _safe_slug(value: Any) -> str:
    text = str(value).strip()
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)

def _filter_manifest(
    manifest: pd.DataFrame,
    *,
    fields: list[int],
    methods: list[str],
    limit: int | None,
) -> pd.DataFrame:
    frame = manifest.copy()
    if fields:
        frame = frame[frame["field"].astype(int).isin(fields)]
    if methods:
        frame = frame[frame["method"].astype(str).isin(methods)]
    frame = frame.sort_values(["field", "method", "graph_dir"]).reset_index(drop=True)
    if limit is not None:
        frame = frame.head(limit)
    return frame

def _case_slug(row: pd.Series, *, mode: str, seed: int, budget: int, probe_only: bool) -> str:
    sample = _safe_slug(row.get("sample", f"field{int(row['field'])}"))
    method = _safe_slug(row["method"])
    mode_slug = _safe_slug(mode)
    suffix = "_probe_only" if probe_only else ""
    return f"{sample}_{method}_seed{seed}_budget{budget}_{mode_slug}{suffix}"

def _completion_marker(case_dir: Path) -> Path:
    return case_dir / "portfolio_batch_case_complete.json"

def _is_completed(case_dir: Path) -> bool:
    marker = _completion_marker(case_dir)
    if not marker.exists():
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return payload.get("status") == "completed"

def _monitor_command(
    *,
    graph_dir: Path,
    output_dir: Path,
    mode: str,
    seed: int,
    budget: int,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        sys.executable,
        str(MONITOR_SCRIPT),
        "--graph-dirs",
        str(graph_dir),
        "--output-dir",
        str(output_dir),
        "--seeds",
        str(seed),
        "--candidate-eval-mode",
        mode,
        "--candidate-budgets",
        str(budget),
        "--baseline-iterations",
        str(args.baseline_iterations),
        "--polish-iterations",
        str(args.polish_iterations),
        "--prescreen-iterations",
        str(args.prescreen_iterations),
        "--final-iterations",
        str(args.final_iterations),
        "--multifidelity-finalists",
        str(args.multifidelity_finalists),
        "--local-merge-summary-mode",
        args.local_merge_summary_mode,
    ]
    if args.keep_raw_trajectory:
        command.append("--keep-raw-trajectory")
    if args.probe_only:
        command.append("--probe-only")
    if args.basin_signatures:
        command.append("--basin-signatures")
    return command

def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()

def _available_memory_gb() -> float:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return math.nan
    for line in meminfo.read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            parts = line.split()
            if len(parts) >= 2:
                return float(parts[1]) / (1024.0 * 1024.0)
    return math.nan

def _parallel_worker_limit(args: argparse.Namespace) -> int | None:
    limits: list[int] = []
    explicit = int(getattr(args, "max_parallel_candidate_workers", 0) or 0)
    if explicit > 0:
        limits.append(explicit)

    estimated = float(getattr(args, "estimated_candidate_worker_gb", 0.0) or 0.0)
    if estimated > 0.0:
        budget = getattr(args, "memory_budget_gb", None)
        if budget is None:
            available = _available_memory_gb()
            reserve = float(getattr(args, "memory_reserve_gb", 16.0) or 0.0)
            budget = available - reserve if math.isfinite(available) else math.nan
        budget = float(budget)
        if math.isfinite(budget):
            limits.append(max(1, int(math.floor(max(0.0, budget) / estimated))))

    if not limits:
        return None
    return max(1, min(limits))

def _subprocess_env_for_mode(
    mode: str,
    args: argparse.Namespace,
) -> tuple[dict[str, str], int | None]:
    env = os.environ.copy()
    if mode != "parallel_full_p5_portfolio":
        return env, None
    limit = _parallel_worker_limit(args)
    if limit is None:
        return env, None
    existing = env.get("RAYON_NUM_THREADS")
    if existing:
        try:
            limit = min(limit, max(1, int(existing)))
        except ValueError:
            pass
    env["RAYON_NUM_THREADS"] = str(limit)
    env["SCISCAPE_PARALLEL_CANDIDATE_WORKER_LIMIT"] = str(limit)
    return env, limit

def _write_report(
    *,
    out_dir: Path,
    case_rows: list[dict[str, Any]],
    failures: pd.DataFrame,
    scorecard: pd.DataFrame,
) -> None:
    completed = sum(1 for row in case_rows if row["status"] == "completed")
    skipped = sum(1 for row in case_rows if row["status"] == "skipped")
    failed = sum(1 for row in case_rows if row["status"] == "failed")
    lines = [
        "# Leiden Parallel Portfolio Batch Report",
        "",
        f"- Cases visited: {len(case_rows)}",
        f"- Completed: {completed}",
        f"- Skipped: {skipped}",
        f"- Failed: {failed}",
        "",
        "## Resource Guard",
        "",
        f"- Parallel worker limits used: {sorted({row.get('parallel_candidate_worker_limit') for row in case_rows if row.get('parallel_candidate_worker_limit')})}",
        "",
        "## Scorecard Snapshot",
        "",
    ]
    if scorecard.empty:
        lines.append("- No scorecard rows were aggregated.")
    else:
        target_rows = scorecard[scorecard["target_policy"] == "extra_p5_final"].copy()
        if target_rows.empty:
            target_rows = scorecard.head(10).copy()
        lines.extend(
            [
                "| case | seed | budget | mode | k_work saving % | net elapsed saving % | role |",
                "|---|---:|---:|---|---:|---:|---|",
            ]
        )
        for _, row in target_rows.head(20).iterrows():
            lines.append(
                "| {case} | {seed} | {budget} | {mode} | {work:.1f} | {elapsed:.1f} | {role} |".format(
                    case=row.get("case", ""),
                    seed=int(row.get("seed", 0)),
                    budget=int(row.get("candidate_budget", 0)),
                    mode=row.get("candidate_eval_mode", ""),
                    work=float(row.get("k_work_saving_pct", math.nan)),
                    elapsed=float(row.get("net_elapsed_saving_pct", math.nan)),
                    role=row.get("acceleration_role", ""),
                )
            )
    if not failures.empty:
        lines.extend(["", "## Failures", ""])
        for _, row in failures.iterrows():
            lines.append(f"- {row['case_slug']}: exit {row['returncode']}")
    (out_dir / "portfolio_batch_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

def _aggregate_outputs(out_dir: Path, case_rows: list[dict[str, Any]]) -> None:
    run_frames: list[pd.DataFrame] = []
    score_frames: list[pd.DataFrame] = []
    failure_rows = [row for row in case_rows if row["status"] == "failed"]
    for row in case_rows:
        case_dir = Path(row["case_dir"])
        for filename, frames in (
            ("monitor_run_rows.csv", run_frames),
            ("work_acceleration_monitor_scorecard.csv", score_frames),
        ):
            frame = _read_csv_if_exists(case_dir / filename)
            if frame.empty:
                continue
            frame = frame.copy()
            frame["batch_case_slug"] = row["case_slug"]
            frame["candidate_eval_mode"] = row["candidate_eval_mode"]
            frame["probe_only"] = bool(row.get("probe_only", False))
            frame["graph_field"] = row["field"]
            frame["graph_method"] = row["method"]
            frames.append(frame)
    run_rows = pd.concat(run_frames, ignore_index=True) if run_frames else pd.DataFrame()
    scorecard = pd.concat(score_frames, ignore_index=True) if score_frames else pd.DataFrame()
    failures = pd.DataFrame(failure_rows)
    run_rows.to_csv(out_dir / "portfolio_batch_run_rows.csv", index=False)
    scorecard.to_csv(out_dir / "portfolio_batch_scorecard.csv", index=False)
    failures.to_csv(out_dir / "portfolio_batch_failures.csv", index=False)
    pd.DataFrame(case_rows).to_csv(out_dir / "portfolio_batch_cases.csv", index=False)
    _write_report(
        out_dir=out_dir,
        case_rows=case_rows,
        failures=failures,
        scorecard=scorecard,
    )

def run_batch(args: argparse.Namespace) -> None:
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.graph_manifest.expanduser().resolve())
    fields = _parse_int_csv(args.fields, default=[])
    methods = _parse_csv(args.methods)
    modes = _parse_csv(args.candidate_eval_modes) or ["parallel_full_p5_portfolio"]
    seeds = _parse_int_csv(args.seeds, default=[11])
    budgets = _parse_int_csv(args.candidate_budgets, default=[3])
    frame = _filter_manifest(
        manifest,
        fields=fields,
        methods=methods,
        limit=args.limit,
    )

    case_rows: list[dict[str, Any]] = []
    for _, graph_row in frame.iterrows():
        graph_dir = Path(str(graph_row["graph_dir"])).expanduser()
        if not graph_dir.is_absolute():
            graph_dir = (REPO_ROOT / graph_dir).resolve()
        for mode in modes:
            for seed in seeds:
                for budget in budgets:
                    case_slug = _case_slug(
                        graph_row,
                        mode=mode,
                        seed=seed,
                        budget=budget,
                        probe_only=bool(args.probe_only),
                    )
                    case_dir = out_dir / case_slug
                    case_dir.mkdir(parents=True, exist_ok=True)
                    base = {
                        "case_slug": case_slug,
                        "case_dir": str(case_dir),
                        "field": int(graph_row["field"]),
                        "method": str(graph_row["method"]),
                        "graph_dir": str(graph_dir),
                        "candidate_eval_mode": mode,
                        "probe_only": bool(args.probe_only),
                        "basin_signatures": bool(args.basin_signatures),
                        "seed": seed,
                        "candidate_budget": budget,
                        "memory_budget_gb": getattr(args, "memory_budget_gb", None),
                        "estimated_candidate_worker_gb": getattr(
                            args,
                            "estimated_candidate_worker_gb",
                            0.0,
                        ),
                    }
                    if args.resume and _is_completed(case_dir):
                        case_rows.append({**base, "status": "skipped", "returncode": 0})
                        continue
                    command = _monitor_command(
                        graph_dir=graph_dir,
                        output_dir=case_dir,
                        mode=mode,
                        seed=seed,
                        budget=budget,
                        args=args,
                    )
                    env, worker_limit = _subprocess_env_for_mode(mode, args)
                    base["parallel_candidate_worker_limit"] = worker_limit
                    t0 = time.perf_counter()
                    completed = subprocess.run(
                        command,
                        cwd=REPO_ROOT,
                        env=env,
                        text=True,
                        capture_output=True,
                        check=False,
                    )
                    elapsed = time.perf_counter() - t0
                    (case_dir / "monitor_stdout.log").write_text(
                        completed.stdout,
                        encoding="utf-8",
                    )
                    (case_dir / "monitor_stderr.log").write_text(
                        completed.stderr,
                        encoding="utf-8",
                    )
                    status = "completed" if completed.returncode == 0 else "failed"
                    marker = {
                        **base,
                        "status": status,
                        "returncode": completed.returncode,
                        "elapsed_sec": elapsed,
                        "command": command,
                    }
                    if status == "completed":
                        _completion_marker(case_dir).write_text(
                            json.dumps(marker, indent=2, sort_keys=True),
                            encoding="utf-8",
                        )
                    case_rows.append(marker)
    _aggregate_outputs(out_dir, case_rows)
    print(
        json.dumps(
            {
                "cases": len(case_rows),
                "completed": sum(1 for row in case_rows if row["status"] == "completed"),
                "skipped": sum(1 for row in case_rows if row["status"] == "skipped"),
                "failed": sum(1 for row in case_rows if row["status"] == "failed"),
                "output_dir": str(out_dir),
            },
            indent=2,
            sort_keys=True,
        )
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fields", type=str, default=None)
    parser.add_argument("--methods", type=str, default=None)
    parser.add_argument(
        "--candidate-eval-modes",
        type=str,
        default="parallel_full_p5_portfolio",
        help=(
            "Comma-separated monitor modes, e.g. full_p5,parallel_full_p5_portfolio,"
            "localized_label,quotient_label,upper_bound_label."
        ),
    )
    parser.add_argument("--seeds", type=str, default="11")
    parser.add_argument("--candidate-budgets", type=str, default="3")
    parser.add_argument("--baseline-iterations", type=int, default=10)
    parser.add_argument("--polish-iterations", type=int, default=5)
    parser.add_argument("--prescreen-iterations", type=int, default=1)
    parser.add_argument("--final-iterations", type=int, default=5)
    parser.add_argument("--multifidelity-finalists", type=int, default=1)
    parser.add_argument(
        "--local-merge-summary-mode",
        choices=("compact", "focused", "full"),
        default="compact",
    )
    parser.add_argument("--keep-raw-trajectory", action="store_true")
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument(
        "--basin-signatures",
        action="store_true",
        help="Forward --basin-signatures to monitor candidate rows.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--max-parallel-candidate-workers",
        type=int,
        default=0,
        help="Set RAYON_NUM_THREADS for parallel_full_p5_portfolio cases; 0 leaves Rayon default.",
    )
    parser.add_argument(
        "--memory-budget-gb",
        type=float,
        default=None,
        help="Optional memory budget used with --estimated-candidate-worker-gb to cap parallel workers.",
    )
    parser.add_argument(
        "--estimated-candidate-worker-gb",
        type=float,
        default=0.0,
        help="Estimated per-candidate parallel worker memory. 0 disables memory-derived capping.",
    )
    parser.add_argument(
        "--memory-reserve-gb",
        type=float,
        default=16.0,
        help="Reserve to subtract from MemAvailable when --memory-budget-gb is omitted.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    run_batch(parse_args())
