#!/usr/bin/env python3
"""Summarize approximate Full Leiden polish candidate labels."""

from __future__ import annotations

import argparse
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


import pandas as pd

APPROACHES = {
    "localized": {
        "metric": "localized_delta_q",
        "rank": "localized_rank",
        "topk": (1, 2, 3),
    },
    "quotient": {
        "metric": "quotient_delta_q",
        "rank": "quotient_rank",
        "topk": (1, 3, 5),
    },
    "upper_bound": {
        "metric": "ub_delta_q",
        "rank": "ub_rank",
        "topk": (1, 2, 3, 5),
    },
}

GROUP_COLUMNS = [
    "candidate_eval_mode",
    "case",
    "seed",
    "candidate_budget",
    "max_group_candidates",
]

def _read_csvs(input_dir: Path, filename: str) -> pd.DataFrame:
    paths = sorted(input_dir.glob(f"**/{filename}"))
    frames: list[pd.DataFrame] = []
    for path in paths:
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if frame.empty:
            continue
        frame["source_path"] = str(path.relative_to(input_dir))
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)

def _finite_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default

def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    try:
        return bool(value)
    except ValueError:
        return False

def _group_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in GROUP_COLUMNS if column in frame.columns]

def _rank_by_metric(group: pd.DataFrame, metric: str) -> pd.Series:
    values = pd.to_numeric(group.get(metric), errors="coerce")
    candidates = pd.to_numeric(group.get("candidate_index"), errors="coerce").fillna(0)
    order = pd.DataFrame(
        {
            "row_index": group.index,
            "metric": values,
            "candidate_index": candidates,
        }
    )
    order = order.sort_values(
        ["metric", "candidate_index"],
        ascending=[False, True],
        na_position="last",
    )
    ranks = pd.Series(index=group.index, dtype="float64")
    for rank, row_index in enumerate(order["row_index"], start=1):
        if math.isfinite(_finite_float(group.loc[row_index, metric])):
            ranks.loc[row_index] = rank
    return ranks

def _candidate_summary(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty or "p5_delta_q" not in candidates.columns:
        return pd.DataFrame()
    group_cols = _group_columns(candidates)
    if not group_cols:
        candidates = candidates.copy()
        candidates["_all"] = "all"
        group_cols = ["_all"]
    rows: list[dict[str, Any]] = []
    for group_key, group in candidates.groupby(group_cols, dropna=False):
        group_key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        base = dict(zip(group_cols, group_key_values, strict=False))
        p5_values = pd.to_numeric(group["p5_delta_q"], errors="coerce")
        if p5_values.notna().sum() == 0:
            continue
        p5_order = group.assign(_p5=p5_values).sort_values(
            ["_p5", "candidate_index"],
            ascending=[False, True],
            na_position="last",
        )
        full_winner = p5_order.iloc[0]
        full_winner_index = int(full_winner.get("candidate_index", -1))
        full_best_delta = _finite_float(full_winner.get("p5_delta_q"))
        for approach, spec in APPROACHES.items():
            metric = spec["metric"]
            if metric not in group.columns:
                continue
            metric_values = pd.to_numeric(group[metric], errors="coerce")
            if metric_values.notna().sum() == 0:
                continue
            ranks = (
                pd.to_numeric(group[spec["rank"]], errors="coerce")
                if spec["rank"] in group.columns
                else _rank_by_metric(group, metric)
            )
            winner_rows = group[group["candidate_index"].astype(int) == full_winner_index]
            winner_rank = math.nan
            if not winner_rows.empty:
                winner_rank = _finite_float(ranks.loc[winner_rows.index[0]])
            summary = {
                **base,
                "approach": approach,
                "candidate_count": int(len(group)),
                "full_p5_winner_candidate_index": full_winner_index,
                "full_p5_winner_approx_rank": winner_rank,
                "full_p5_best_delta_q": full_best_delta,
                "approx_best_delta_q": _finite_float(metric_values.max()),
            }
            for top_k in spec["topk"]:
                top_rows = group.loc[ranks[ranks <= top_k].index]
                top_p5 = pd.to_numeric(top_rows.get("p5_delta_q"), errors="coerce")
                best_top_p5 = _finite_float(top_p5.max())
                summary[f"recall_at_{top_k}"] = bool(
                    math.isfinite(winner_rank) and winner_rank <= top_k
                )
                summary[f"quality_gap_at_{top_k}"] = (
                    full_best_delta - best_top_p5
                    if math.isfinite(full_best_delta) and math.isfinite(best_top_p5)
                    else math.nan
                )
            if approach == "upper_bound" and "ub_violation" in group.columns:
                violations = pd.to_numeric(group["ub_violation"], errors="coerce")
                covers = group.get("ub_covers_p5", pd.Series(dtype=object)).map(_truthy)
                summary["ub_coverage_rate"] = float(covers.mean()) if len(covers) else math.nan
                summary["ub_max_violation"] = _finite_float(violations.max())
            rows.append(summary)
    return pd.DataFrame(rows)

def _policy_summary(policies: pd.DataFrame) -> pd.DataFrame:
    if policies.empty or "policy" not in policies.columns:
        return pd.DataFrame()
    group_cols = ["candidate_eval_mode", "policy"]
    group_cols = [column for column in group_cols if column in policies.columns]
    rows: list[dict[str, Any]] = []
    for key, group in policies.groupby(group_cols, dropna=False):
        values = key if isinstance(key, tuple) else (key,)
        base = dict(zip(group_cols, values, strict=False))
        available = group.get("available", pd.Series(dtype=object)).map(_truthy)
        matches = group.get("matches_full_p5", pd.Series(dtype=object)).map(_truthy)
        total_elapsed = pd.to_numeric(group.get("total_elapsed_ms"), errors="coerce")
        p5_evaluated = pd.to_numeric(group.get("p5_evaluated"), errors="coerce")
        candidate_count = pd.to_numeric(group.get("candidate_count"), errors="coerce")
        rows.append(
            {
                **base,
                "rows": int(len(group)),
                "available_rate": float(available.mean()) if len(available) else math.nan,
                "matches_full_p5_rate": float(matches.mean()) if len(matches) else math.nan,
                "mean_total_elapsed_ms": _finite_float(total_elapsed.mean()),
                "mean_p5_evaluated": _finite_float(p5_evaluated.mean()),
                "mean_candidate_count": _finite_float(candidate_count.mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)

def _write_report(
    output_dir: Path,
    candidate_summary: pd.DataFrame,
    policy_summary: pd.DataFrame,
) -> None:
    lines = [
        "# Approximate Polish Candidate Review",
        "",
        "## Candidate Recall",
        "",
    ]
    if candidate_summary.empty:
        lines.append("- No approximate candidate rows were available.")
    else:
        aggregate_rows = []
        for approach, group in candidate_summary.groupby("approach"):
            row: dict[str, Any] = {"approach": approach, "groups": len(group)}
            for column in group.columns:
                if column.startswith("recall_at_"):
                    row[column] = float(group[column].mean())
                if column.startswith("quality_gap_at_"):
                    row[f"max_{column}"] = _finite_float(group[column].max())
            if approach == "upper_bound" and "ub_coverage_rate" in group:
                row["mean_ub_coverage_rate"] = _finite_float(group["ub_coverage_rate"].mean())
                row["max_ub_violation"] = _finite_float(group["ub_max_violation"].max())
            aggregate_rows.append(row)
        aggregate = pd.DataFrame(aggregate_rows)
        lines.extend(_markdown_table(aggregate).splitlines())
    lines.extend(["", "## Policy Rows", ""])
    if policy_summary.empty:
        lines.append("- No policy rows were available.")
    else:
        display = policy_summary.copy()
        lines.extend(_markdown_table(display).splitlines())
    (output_dir / "approx_polish_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    columns = list(frame.columns)
    out = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                if math.isnan(value):
                    values.append("")
                else:
                    values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        out.append("| " + " | ".join(values) + " |")
    return "\n".join(out)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = _read_csvs(input_dir, "candidate_level_rows.csv")
    policies = _read_csvs(input_dir, "policy_comparison_rows.csv")
    candidate_summary = _candidate_summary(candidates)
    policy_summary = _policy_summary(policies)
    candidate_summary.to_csv(output_dir / "approx_polish_candidate_summary.csv", index=False)
    policy_summary.to_csv(output_dir / "approx_polish_policy_summary.csv", index=False)
    _write_report(output_dir, candidate_summary, policy_summary)
    print(
        {
            "candidate_rows": int(len(candidates)),
            "policy_rows": int(len(policies)),
            "output_dir": str(output_dir),
        }
    )

if __name__ == "__main__":
    main()
