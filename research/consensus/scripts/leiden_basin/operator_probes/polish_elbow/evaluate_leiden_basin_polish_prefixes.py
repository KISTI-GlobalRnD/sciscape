#!/usr/bin/env python3
"""Evaluate bounded local polish on selected barrier-aware pathway prefixes."""

from __future__ import annotations

import argparse
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


import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent

from collect_leiden_vanilla_reachability_sweep import (  # noqa: E402
    _load_graph,
    _read_candidate_rows,
    compatible_sketch_nodes,
)
from profile_leiden_basin_ordered_flips import (  # noqa: E402
    DEFAULT_OUTPUT_DIR as DEFAULT_SINGLE_PROFILE_DIR,
    UNIT_ROWS_FILENAME,
    _markdown_table,
)
from profile_leiden_basin_ordered_flips_batch import (  # noqa: E402
    DEFAULT_OUTPUT_DIR as DEFAULT_PROFILE_BATCH_DIR,
)
from analyze_leiden_basin_barrier_aware_pathways import (  # noqa: E402
    DEFAULT_OUTPUT_DIR as DEFAULT_PREFIX_DIR,
    PREFIX_ROWS_FILENAME as BARRIER_PREFIX_ROWS_FILENAME,
)
from run_leiden_basin_transition_operator_pilot import (  # noqa: E402
    DEFAULT_CANDIDATE_DIRS,
    DEFAULT_VANILLA_DIR,
    VANILLA_ROWS_FILENAME,
    _find_candidate_row,
    _find_vanilla_row,
    _recreate_candidate,
    _run_leiden,
    _safe_int,
)
from sciscape.clustering.leiden_basin_profile import (  # noqa: E402
    apply_prefix_units,
    classify_polish_recovery,
    compact_membership,
    fixed_outside,
    membership_metric_row,
    score_membership,
    support_distance,
    support_progress_from_vanilla,
    v_only_support_nodes,
)

DEFAULT_OUTPUT_DIR = DEFAULT_SINGLE_PROFILE_DIR.parent / (
    "pathway_polish_aware_prefix_field34_cc_v1"
)
ROWS_FILENAME = "polish_aware_prefix_rows.csv"
CASE_ROWS_FILENAME = "polish_aware_case_rows.csv"
SUMMARY_FILENAME = "polish_aware_summary.json"
REPORT_FILENAME = "polish_aware_report.md"

def _parse_csv_tuple(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())

def select_prefix_rows(
    prefix_rows: pd.DataFrame,
    *,
    pair_ids: tuple[str, ...],
    top_prefixes_per_case: int,
) -> pd.DataFrame:
    rows = prefix_rows.copy()
    if pair_ids:
        rows = rows[rows["pair_id"].astype(str).isin(set(pair_ids))].copy()
    if rows.empty:
        return rows
    return (
        rows.sort_values(
            [
                "pair_id",
                "barrier_aware_score",
                "support_progress_fraction",
                "peak_raw_barrier",
            ],
            ascending=[True, False, False, True],
        )
        .groupby("pair_id", as_index=False)
        .head(int(top_prefixes_per_case))
        .reset_index(drop=True)
    )

def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None

def _evaluate_case_prefixes(
    *,
    case_prefix_rows: pd.DataFrame,
    profile_batch_dir: Path,
    candidate_dirs: tuple[Path, ...],
    vanilla_dir: Path,
    baseline_iterations: int,
    candidate_polish_iterations: int,
    local_polish_iterations: int,
    resolution: float,
    randomness: float,
    perturb_seed_offset: int,
    polish_seed_offset: int,
) -> pd.DataFrame:
    first = case_prefix_rows.iloc[0]
    case = str(first["case"])
    pair_id = str(first["pair_id"])
    candidate_index = int(first["candidate_index"])
    vanilla_seed = int(first["vanilla_seed"])
    vanilla_randomness = float(first["vanilla_randomness"])
    vanilla_n = str(first["vanilla_requested_n_iterations"])
    profile_dir = profile_batch_dir / pair_id
    units = pd.read_csv(profile_dir / UNIT_ROWS_FILENAME)
    candidate_rows = _read_candidate_rows(candidate_dirs)
    vanilla_rows = pd.read_csv(vanilla_dir / VANILLA_ROWS_FILENAME)
    candidate_row = _find_candidate_row(
        candidate_rows,
        case=case,
        candidate_index=candidate_index,
    )
    vanilla_row = _find_vanilla_row(
        vanilla_rows,
        case=case,
        seed=vanilla_seed,
        randomness=vanilla_randomness,
        n_iterations=vanilla_n,
    )
    graph_dir = Path(str(vanilla_row["graph_dir"]))
    graph, node_weights, arrays = _load_graph(graph_dir)
    baseline = _run_leiden(
        graph,
        resolution=resolution,
        seed=int(candidate_row.get("seed", 0)),
        n_iterations=baseline_iterations,
        randomness=randomness,
    )
    candidate = _recreate_candidate(
        graph=graph,
        arrays=arrays,
        node_weights=node_weights,
        baseline_membership=baseline.membership,
        baseline_quality=baseline.quality,
        row=candidate_row,
        resolution=resolution,
        randomness=randomness,
        perturb_seed_offset=perturb_seed_offset,
        polish_iterations=candidate_polish_iterations,
    )
    vanilla = _run_leiden(
        graph,
        resolution=resolution,
        seed=vanilla_seed,
        n_iterations=int(_safe_int(vanilla_n, baseline_iterations) or baseline_iterations),
        randomness=vanilla_randomness,
    )
    sketch_nodes, sketch_context = compatible_sketch_nodes(
        arrays=arrays,
        baseline_membership=baseline.membership,
        node_weights=node_weights,
        candidate_rows=candidate_rows[candidate_rows["case"].astype(str) == case],
    )
    if not bool(sketch_context.get("sketch_context_hash_matches_candidate", False)):
        raise RuntimeError(f"sketch context mismatch for {case}")
    candidate_support, vanilla_support, _ = v_only_support_nodes(
        baseline.membership,
        candidate.recreated.membership,
        vanilla.membership,
    )
    vanilla_support_distance_to_candidate = support_distance(
        vanilla_support,
        candidate_support,
    )[0]
    out_rows: list[dict[str, Any]] = []
    for prefix_rank, (_, prefix_row) in enumerate(case_prefix_rows.iterrows(), start=1):
        raw_membership, mutable_nodes = apply_prefix_units(
            membership=vanilla.membership,
            donor_membership=candidate.recreated.membership,
            units=units,
            prefix_unit_ids=prefix_row["prefix_unit_ids"],
        )
        raw_membership = compact_membership(raw_membership)
        raw_quality = score_membership(
            graph,
            raw_membership,
            resolution=resolution,
        )
        raw_metrics = membership_metric_row(
            membership=raw_membership,
            quality=raw_quality,
            baseline_membership=baseline.membership,
            candidate_membership=candidate.recreated.membership,
            vanilla_membership=vanilla.membership,
            sketch_nodes=sketch_nodes,
            start_quality=vanilla.quality,
            candidate_quality=candidate.recreated.quality,
            vanilla_quality=vanilla.quality,
            prefix="raw",
        )
        raw_progress = support_progress_from_vanilla(
            support_distance_to_candidate=raw_metrics[
                "raw_support_distance_to_candidate"
            ],
            vanilla_support_distance_to_candidate=vanilla_support_distance_to_candidate,
        )
        if int(local_polish_iterations) > 0 and mutable_nodes.size:
            polished = _run_leiden(
                graph,
                resolution=resolution,
                seed=int(polish_seed_offset) + int(prefix_rank),
                n_iterations=local_polish_iterations,
                randomness=randomness,
                initial_membership=raw_membership,
                fixed_nodes=fixed_outside(int(raw_membership.size), mutable_nodes),
            )
            polish_membership = polished.membership
            polish_quality = polished.quality
            polish_elapsed = polished.elapsed_sec
        else:
            polish_membership = raw_membership
            polish_quality = raw_quality
            polish_elapsed = 0.0
        polish_metrics = membership_metric_row(
            membership=polish_membership,
            quality=polish_quality,
            baseline_membership=baseline.membership,
            candidate_membership=candidate.recreated.membership,
            vanilla_membership=vanilla.membership,
            sketch_nodes=sketch_nodes,
            start_quality=vanilla.quality,
            candidate_quality=candidate.recreated.quality,
            vanilla_quality=vanilla.quality,
            prefix="polish",
        )
        polish_progress = support_progress_from_vanilla(
            support_distance_to_candidate=polish_metrics[
                "polish_support_distance_to_candidate"
            ],
            vanilla_support_distance_to_candidate=vanilla_support_distance_to_candidate,
        )
        raw_debt = float(raw_metrics["raw_q_debt_vs_start"])
        polish_debt = float(polish_metrics["polish_q_debt_vs_start"])
        q_debt_recovery_fraction = (
            (raw_debt - polish_debt) / raw_debt if raw_debt > 0.0 else math.nan
        )
        progress_retention_fraction = (
            polish_progress / raw_progress if raw_progress > 0.0 else math.nan
        )
        row = {
            "case": case,
            "field": first.get("field", ""),
            "method": first.get("method", ""),
            "pair_id": pair_id,
            "candidate_index": candidate_index,
            "vanilla_seed": vanilla_seed,
            "vanilla_randomness": vanilla_randomness,
            "vanilla_requested_n_iterations": vanilla_n,
            "prefix_rank": int(prefix_rank),
            "prefix_unit_ids": prefix_row["prefix_unit_ids"],
            "prefix_unit_count": int(prefix_row["prefix_unit_count"]),
            "prefix_flipped_node_count_estimate": int(
                prefix_row["prefix_flipped_node_count_estimate"]
            ),
            "mutable_node_count": int(mutable_nodes.size),
            "barrier_aware_score": float(prefix_row["barrier_aware_score"]),
            "peak_raw_barrier_input": float(prefix_row["peak_raw_barrier"]),
            "support_progress_fraction_input": float(
                prefix_row["support_progress_fraction"]
            ),
            "greedy_failure_labels": prefix_row["greedy_failure_labels"],
            "candidate_quality": float(candidate.recreated.quality),
            "vanilla_quality": float(vanilla.quality),
            "vanilla_support_distance_to_candidate": float(
                vanilla_support_distance_to_candidate
            ),
            "local_polish_iterations": int(local_polish_iterations),
            "local_polish_elapsed_sec": float(polish_elapsed),
            **raw_metrics,
            **polish_metrics,
            "raw_candidate_progress_from_vanilla": float(raw_progress),
            "polish_candidate_progress_from_vanilla": float(polish_progress),
            "q_recovery": float(polish_quality - raw_quality),
            "q_debt_recovery_fraction": _finite_or_none(q_debt_recovery_fraction),
            "candidate_progress_retention_fraction": _finite_or_none(
                progress_retention_fraction
            ),
            "polish_recovery_label": classify_polish_recovery(
                raw_delta_q_vs_start=float(raw_metrics["raw_delta_q_vs_start"]),
                polish_delta_q_vs_start=float(
                    polish_metrics["polish_delta_q_vs_start"]
                ),
                raw_progress_from_vanilla=float(raw_progress),
                polish_progress_from_vanilla=float(polish_progress),
                polish_support_distance_to_vanilla=float(
                    polish_metrics["polish_support_distance_to_vanilla"]
                ),
            ),
        }
        out_rows.append(row)
    return pd.DataFrame(out_rows)

def _case_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    grouped = rows.groupby("pair_id", sort=True)
    case_rows: list[dict[str, Any]] = []
    for pair_id, group in grouped:
        best = group.sort_values(
            [
                "polish_delta_q_vs_start",
                "polish_candidate_progress_from_vanilla",
                "candidate_progress_retention_fraction",
            ],
            ascending=[False, False, False],
        ).iloc[0]
        labels = group["polish_recovery_label"].value_counts().to_dict()
        case_rows.append(
            {
                "pair_id": pair_id,
                "evaluated_prefix_rows": int(len(group)),
                "recovered_support_shift_rows": int(
                    labels.get("recovered_support_shift", 0)
                ),
                "recovered_vanilla_near_rows": int(
                    labels.get("recovered_vanilla_near", 0)
                ),
                "quality_loss_rows": int(labels.get("quality_loss", 0)),
                "raw_only_rows": int(labels.get("raw_only", 0)),
                "best_prefix_rank": int(best["prefix_rank"]),
                "best_polish_delta_q_vs_start": float(
                    best["polish_delta_q_vs_start"]
                ),
                "best_q_recovery": float(best["q_recovery"]),
                "best_q_debt_recovery_fraction": _finite_or_none(
                    best["q_debt_recovery_fraction"]
                ),
                "best_polish_candidate_progress_from_vanilla": float(
                    best["polish_candidate_progress_from_vanilla"]
                ),
                "best_candidate_progress_retention_fraction": _finite_or_none(
                    best["candidate_progress_retention_fraction"]
                ),
                "best_polish_support_distance_to_vanilla": float(
                    best["polish_support_distance_to_vanilla"]
                ),
                "best_polish_recovery_label": best["polish_recovery_label"],
            }
        )
    return pd.DataFrame(case_rows)

def write_report(
    path: Path,
    *,
    rows: pd.DataFrame,
    case_rows: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    lines = [
        "# Polish-Aware Pathway Prefix Evaluation",
        "",
        "This diagnostic applies selected barrier-aware prefixes, then runs bounded local polish with only prefix nodes mutable.",
        "",
        "It tests recoverability and support retention. It does not promote an operator.",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "| --- | --- |",
    ]
    for key in [
        "prefix_dir",
        "profile_batch_dir",
        "evaluated_prefix_rows",
        "top_prefixes_per_case",
        "local_polish_iterations",
    ]:
        lines.append(f"| {key} | {summary.get(key, '')} |")
    lines.extend(["", "## Case Rows", ""])
    case_cols = [
        "pair_id",
        "evaluated_prefix_rows",
        "recovered_support_shift_rows",
        "recovered_vanilla_near_rows",
        "quality_loss_rows",
        "best_prefix_rank",
        "best_polish_delta_q_vs_start",
        "best_q_recovery",
        "best_q_debt_recovery_fraction",
        "best_polish_candidate_progress_from_vanilla",
        "best_candidate_progress_retention_fraction",
        "best_polish_support_distance_to_vanilla",
        "best_polish_recovery_label",
    ]
    lines.extend(
        _markdown_table(
            case_rows[[c for c in case_cols if c in case_rows.columns]],
            max_rows=40,
        )
    )
    lines.extend(["", "## Prefix Rows", ""])
    row_cols = [
        "pair_id",
        "prefix_rank",
        "prefix_unit_count",
        "mutable_node_count",
        "peak_raw_barrier_input",
        "raw_delta_q_vs_start",
        "polish_delta_q_vs_start",
        "q_recovery",
        "q_debt_recovery_fraction",
        "raw_candidate_progress_from_vanilla",
        "polish_candidate_progress_from_vanilla",
        "candidate_progress_retention_fraction",
        "polish_support_distance_to_vanilla",
        "polish_recovery_label",
        "prefix_unit_ids",
    ]
    display = rows.sort_values(
        [
            "pair_id",
            "polish_delta_q_vs_start",
            "polish_candidate_progress_from_vanilla",
        ],
        ascending=[True, False, False],
    )
    lines.extend(
        _markdown_table(display[[c for c in row_cols if c in display.columns]], max_rows=80)
    )
    lines.extend(
        [
            "",
            "## Guardrail",
            "",
            "- `recovered_support_shift` is only a diagnostic label. It still needs seed and multi-start controls before any operator claim.",
            "- `recovered_vanilla_near` means polish improved or preserved QF but did not retain enough support movement away from vanilla.",
            "- If most rows collapse to vanilla-near, the next change should target larger closure/context prefixes rather than another threshold sweep.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def run_evaluation(
    *,
    prefix_dir: Path,
    profile_batch_dir: Path,
    output_dir: Path,
    candidate_dirs: tuple[Path, ...],
    vanilla_dir: Path,
    pair_ids: tuple[str, ...],
    top_prefixes_per_case: int,
    baseline_iterations: int,
    candidate_polish_iterations: int,
    local_polish_iterations: int,
    resolution: float,
    randomness: float,
    perturb_seed_offset: int,
    polish_seed_offset: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefixes = select_prefix_rows(
        pd.read_csv(prefix_dir / BARRIER_PREFIX_ROWS_FILENAME),
        pair_ids=pair_ids,
        top_prefixes_per_case=top_prefixes_per_case,
    )
    if prefixes.empty:
        raise ValueError("No prefix rows selected for polish evaluation")
    frames: list[pd.DataFrame] = []
    for _, case_prefixes in prefixes.groupby("pair_id", sort=True):
        frames.append(
            _evaluate_case_prefixes(
                case_prefix_rows=case_prefixes,
                profile_batch_dir=profile_batch_dir,
                candidate_dirs=candidate_dirs,
                vanilla_dir=vanilla_dir,
                baseline_iterations=baseline_iterations,
                candidate_polish_iterations=candidate_polish_iterations,
                local_polish_iterations=local_polish_iterations,
                resolution=resolution,
                randomness=randomness,
                perturb_seed_offset=perturb_seed_offset,
                polish_seed_offset=polish_seed_offset,
            )
        )
    rows = pd.concat(frames, ignore_index=True)
    case_rows = _case_rows(rows)
    rows.to_csv(output_dir / ROWS_FILENAME, index=False)
    case_rows.to_csv(output_dir / CASE_ROWS_FILENAME, index=False)
    summary = {
        "schema": "leiden_basin_polish_aware_prefixes.v1",
        "prefix_dir": str(prefix_dir),
        "profile_batch_dir": str(profile_batch_dir),
        "output_dir": str(output_dir),
        "pair_ids": list(pair_ids),
        "evaluated_prefix_rows": int(len(rows)),
        "top_prefixes_per_case": int(top_prefixes_per_case),
        "baseline_iterations": int(baseline_iterations),
        "candidate_polish_iterations": int(candidate_polish_iterations),
        "local_polish_iterations": int(local_polish_iterations),
        "resolution": float(resolution),
        "randomness": float(randomness),
        "perturb_seed_offset": int(perturb_seed_offset),
        "polish_seed_offset": int(polish_seed_offset),
    }
    (output_dir / SUMMARY_FILENAME).write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_report(
        output_dir / REPORT_FILENAME,
        rows=rows,
        case_rows=case_rows,
        summary=summary,
    )
    return summary

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix-dir", type=Path, default=DEFAULT_PREFIX_DIR)
    parser.add_argument(
        "--profile-batch-dir",
        type=Path,
        default=DEFAULT_PROFILE_BATCH_DIR,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        action="append",
        default=None,
    )
    parser.add_argument("--vanilla-dir", type=Path, default=DEFAULT_VANILLA_DIR)
    parser.add_argument("--pair-ids", default="")
    parser.add_argument("--top-prefixes-per-case", type=int, default=10)
    parser.add_argument("--baseline-iterations", type=int, default=10)
    parser.add_argument("--candidate-polish-iterations", type=int, default=5)
    parser.add_argument("--local-polish-iterations", type=int, default=3)
    parser.add_argument("--resolution", type=float, default=0.01)
    parser.add_argument("--randomness", type=float, default=0.01)
    parser.add_argument("--perturb-seed-offset", type=int, default=5000)
    parser.add_argument("--polish-seed-offset", type=int, default=9000)
    return parser

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_evaluation(
        prefix_dir=args.prefix_dir,
        profile_batch_dir=args.profile_batch_dir,
        output_dir=args.output_dir,
        candidate_dirs=tuple(args.candidate_dir or DEFAULT_CANDIDATE_DIRS),
        vanilla_dir=args.vanilla_dir,
        pair_ids=_parse_csv_tuple(args.pair_ids),
        top_prefixes_per_case=args.top_prefixes_per_case,
        baseline_iterations=args.baseline_iterations,
        candidate_polish_iterations=args.candidate_polish_iterations,
        local_polish_iterations=args.local_polish_iterations,
        resolution=args.resolution,
        randomness=args.randomness,
        perturb_seed_offset=args.perturb_seed_offset,
        polish_seed_offset=args.polish_seed_offset,
    )
    print(summary)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
