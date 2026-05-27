#!/usr/bin/env python3
"""Run long-polish guards for monitored Leiden work-acceleration candidates."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_leiden_hysteresis_shatter_smoke import _case_name, _load_graph_arrays  # noqa: E402
from run_leiden_hysteresis_work_acceleration_monitor import (  # noqa: E402
    _compact_membership,
    _reconstruct_external_group,
)
from sciscape.clustering.leiden_rust import build_leiden_graph  # noqa: E402


DEFAULT_MONITOR_DIR = (
    REPO_ROOT
    / "research/consensus/results/adaptive_refinement/"
    "leiden_hysteresis_work_acceleration_monitor_20260513"
)
DEFAULT_OUTPUT_DIR = DEFAULT_MONITOR_DIR / "long_polish_guard"


def _load_graph_map(summary_path: Path) -> dict[str, Path]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    out: dict[str, Path] = {}
    for rel in payload.get("graph_dirs", []):
        path = (REPO_ROOT / rel).resolve()
        out[_case_name(path)] = path
    return out


def _ppm(delta: float, baseline_quality: float) -> float:
    return delta / baseline_quality * 1_000_000.0 if baseline_quality else 0.0


def _run_guard_for_row(
    *,
    row: pd.Series,
    graph_dir: Path,
    resolution: float,
    baseline_iterations: int,
    long_polish_iterations: int,
    randomness: float,
    perturb_seed_offset: int,
) -> dict[str, Any]:
    arrays = _load_graph_arrays(graph_dir)
    graph = build_leiden_graph(
        edges_src=arrays.src,
        edges_dst=arrays.dst,
        edges_weight=arrays.weight,
        n_nodes=int(arrays.node_weights.shape[0]),
        node_weights=arrays.node_weights,
    )
    seed = int(row["seed"])
    case = str(row["case"])
    baseline_start = time.perf_counter()
    baseline = graph.run_leiden(
        resolution=resolution,
        seed=seed,
        n_iterations=baseline_iterations,
        randomness=randomness,
    )
    baseline_elapsed = time.perf_counter() - baseline_start
    baseline_membership = np.asarray(baseline.membership, dtype=np.uint64)

    group_nodes, reconstruction = _reconstruct_external_group(
        src=arrays.src,
        dst=arrays.dst,
        weight=arrays.weight,
        membership=baseline_membership,
        node_weights=arrays.node_weights,
        source_cluster=int(row["source_cluster"]),
        target_cluster=int(row["target_cluster"]),
    )
    perturbed = baseline_membership.copy()
    perturbed[group_nodes] = np.uint64(int(row["target_cluster"]))
    perturbed = _compact_membership(perturbed)

    extra_seed = seed + perturb_seed_offset
    perturb_seed = seed + perturb_seed_offset + int(row["candidate_index"])

    extra_start = time.perf_counter()
    extra = graph.run_leiden(
        resolution=resolution,
        seed=extra_seed,
        n_iterations=long_polish_iterations,
        randomness=randomness,
        initial_membership=baseline_membership,
    )
    extra_elapsed = time.perf_counter() - extra_start

    perturb_start = time.perf_counter()
    perturb = graph.run_leiden(
        resolution=resolution,
        seed=perturb_seed,
        n_iterations=long_polish_iterations,
        randomness=randomness,
        initial_membership=perturbed,
    )
    perturb_elapsed = time.perf_counter() - perturb_start

    extra_delta = float(extra.quality - baseline.quality)
    perturb_delta = float(perturb.quality - baseline.quality)
    advantage = perturb_delta - extra_delta
    out = {
        "case": case,
        "seed": seed,
        "candidate_budget": int(row["candidate_budget"]) if "candidate_budget" in row else 0,
        "source_cluster": int(row["source_cluster"]),
        "target_cluster": int(row["target_cluster"]),
        "group_kind": row["group_kind"],
        "group_count": int(row["group_count"]),
        "group_weight": float(row["group_weight"]),
        "reconstructed_group_count": int(reconstruction["reconstructed_group_count"]),
        "reconstructed_group_weight": float(reconstruction["reconstructed_group_weight"]),
        "candidate_index": int(row["candidate_index"]),
        "baseline_quality": float(baseline.quality),
        "baseline_elapsed_sec": baseline_elapsed,
        "extra_long_quality": float(extra.quality),
        "perturb_long_quality": float(perturb.quality),
        "extra_long_delta_q": extra_delta,
        "perturb_long_delta_q": perturb_delta,
        "perturb_minus_extra_long_delta_q": advantage,
        "extra_long_delta_ppm": _ppm(extra_delta, float(baseline.quality)),
        "perturb_long_delta_ppm": _ppm(perturb_delta, float(baseline.quality)),
        "perturb_minus_extra_long_ppm": _ppm(advantage, float(baseline.quality)),
        "extra_long_n_clusters": int(extra.n_clusters),
        "perturb_long_n_clusters": int(perturb.n_clusters),
        "extra_long_elapsed_sec": extra_elapsed,
        "perturb_long_elapsed_sec": perturb_elapsed,
        "long_guard_passed": bool(advantage >= 0.0),
    }
    return out


def _write_report(rows: pd.DataFrame, out_dir: Path) -> None:
    lines = [
        "# Leiden Hysteresis Work Acceleration Long-Polish Guard",
        "",
        "Guard check for monitored acceleration candidates. Positive long ppm means perturb still beats ordinary extra polish after the longer polish budget.",
        "",
        "| case | seed | budget | group | p20 adv ppm | p20 adv q | guard |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in rows.iterrows():
        lines.append(
            "| {case} | {seed} | {budget} | {group} | {ppm:.1f} | {dq:.3f} | {guard} |".format(
                case=row["case"],
                seed=int(row["seed"]),
                budget=int(row.get("candidate_budget", 0)),
                group=int(row["group_count"]),
                ppm=float(row["perturb_minus_extra_long_ppm"]),
                dq=float(row["perturb_minus_extra_long_delta_q"]),
                guard="pass" if bool(row["long_guard_passed"]) else "fail",
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Passing this guard supports a durable quality-neutral-or-better speed claim.",
            "- Failing this guard can still be a shortcut signal, but it should not be treated as a durable quality improvement.",
            "- This guard does not measure k_work at p20; it only checks whether long polish catches or reverses the short-run perturb advantage.",
        ]
    )
    (out_dir / "long_polish_guard_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monitor-dir", type=Path, default=DEFAULT_MONITOR_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--resolution", type=float, default=0.01)
    parser.add_argument("--baseline-iterations", type=int, default=10)
    parser.add_argument("--long-polish-iterations", type=int, default=20)
    parser.add_argument("--randomness", type=float, default=0.01)
    parser.add_argument("--perturb-seed-offset", type=int, default=5000)
    parser.add_argument(
        "--roles",
        type=str,
        default="short_run_work_acceleration_candidate_needs_long,work_acceleration_quality_neutral",
        help="Comma-separated acceleration_role values to guard; use all for every row.",
    )
    parser.add_argument(
        "--target-policy",
        type=str,
        default="extra_p5_final",
        help="Target policy to guard for v2 scorecards; use all to disable this filter.",
    )
    parser.add_argument(
        "--allow-nonpositive-k-work-saving",
        action="store_true",
        help="Do not require positive k_work saving when selecting v2 rows.",
    )
    parser.add_argument(
        "--guard-all-budgets",
        action="store_true",
        help="Guard every selected candidate budget instead of the lowest selected budget per case/seed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    monitor_dir = args.monitor_dir.expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    graph_map = _load_graph_map(monitor_dir / "work_acceleration_monitor_summary.json")
    run_rows = pd.read_csv(monitor_dir / "monitor_run_rows.csv")
    scorecard = pd.read_csv(monitor_dir / "work_acceleration_monitor_scorecard.csv")
    selected = scorecard.copy()
    if "target_policy" in selected and args.target_policy != "all":
        selected = selected[selected["target_policy"] == args.target_policy]
    if "k_work_saving_pct" in selected and not args.allow_nonpositive_k_work_saving:
        selected = selected[pd.to_numeric(selected["k_work_saving_pct"], errors="coerce") > 0.0]
    roles = {part.strip() for part in args.roles.split(",") if part.strip()}
    if roles and "all" not in roles:
        selected = selected[selected["acceleration_role"].isin(roles)]
    if "candidate_budget" not in selected:
        selected["candidate_budget"] = 0
    if not args.guard_all_budgets and not selected.empty:
        selected = selected.sort_values(
            ["case", "seed", "candidate_budget", "k_work_saving_pct"],
            ascending=[True, True, True, False],
        ).drop_duplicates(["case", "seed"], keep="first")
    selected_keys = {
        (row["case"], int(row["seed"]), int(row["candidate_budget"]))
        for _, row in selected.iterrows()
    }
    perturb_rows = run_rows[run_rows["branch"] == "perturb"].copy()
    if "candidate_budget" not in perturb_rows:
        perturb_rows["candidate_budget"] = 0
    perturb_rows = perturb_rows[
        perturb_rows.apply(
            lambda row: (
                row["case"],
                int(row["seed"]),
                int(row["candidate_budget"]),
            )
            in selected_keys,
            axis=1,
        )
    ]

    rows: list[dict[str, Any]] = []
    for _, row in perturb_rows.sort_values(["case", "seed"]).iterrows():
        case = str(row["case"])
        graph_dir = graph_map.get(case)
        if graph_dir is None:
            raise KeyError(f"no graph_dir for case {case!r}")
        print(
            f"[long-guard] {case} seed={int(row['seed'])} budget={int(row.get('candidate_budget', 0))}",
            flush=True,
        )
        rows.append(
            _run_guard_for_row(
                row=row,
                graph_dir=graph_dir,
                resolution=float(args.resolution),
                baseline_iterations=int(args.baseline_iterations),
                long_polish_iterations=int(args.long_polish_iterations),
                randomness=float(args.randomness),
                perturb_seed_offset=int(args.perturb_seed_offset),
            )
        )

    frame = pd.DataFrame(rows)
    csv_path = out_dir / "long_polish_guard_rows.csv"
    frame.to_csv(csv_path, index=False)
    _write_report(frame, out_dir)
    summary = {
        "schema": "leiden_hysteresis_work_acceleration_long_guard.v2",
        "monitor_dir": str(monitor_dir.relative_to(REPO_ROOT)),
        "long_polish_iterations": int(args.long_polish_iterations),
        "target_policy": args.target_policy,
        "guard_all_budgets": bool(args.guard_all_budgets),
        "n_rows": int(len(frame)),
        "n_pass": int(frame["long_guard_passed"].sum()) if not frame.empty else 0,
        "paths": {
            "rows_csv": str(csv_path.relative_to(REPO_ROOT)),
            "report_md": str((out_dir / "long_polish_guard_report.md").relative_to(REPO_ROOT)),
        },
    }
    (out_dir / "long_polish_guard_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary["paths"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
