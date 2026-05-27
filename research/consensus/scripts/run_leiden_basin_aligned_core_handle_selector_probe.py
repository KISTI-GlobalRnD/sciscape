#!/usr/bin/env python3
"""Replay aligned-core handle selectors against subset outcomes.

This diagnostic asks whether local/proxy handle features can recover the
minimal sufficient handle subset that was found by exhaustive subset probing.
It does not run Leiden again; it grades selector top-k prefixes against the
already computed subset and stability artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT))

from run_leiden_basin_aligned_core_boundary_operator_probe import (  # noqa: E402
    DEFAULT_FRONTIER_ROWS,
)
from run_leiden_basin_aligned_core_handle_stability_probe import (  # noqa: E402
    DEFAULT_OUTPUT_DIR as DEFAULT_STABILITY_DIR,
    SUMMARY_ROWS_FILENAME as STABILITY_SUMMARY_ROWS_FILENAME,
)
from run_leiden_basin_aligned_core_handle_subset_probe import (  # noqa: E402
    DEFAULT_OUTPUT_DIR as DEFAULT_SUBSET_DIR,
    ROWS_FILENAME as SUBSET_ROWS_FILENAME,
)
from sciscape.clustering.leiden_basin_search import (  # noqa: E402
    ALIGNED_CORE_HANDLE_SELECTOR_POLICIES,
    build_aligned_core_handle_selector_plan_rows,
)


COMBINED_DIR = DEFAULT_SUBSET_DIR.parent
DEFAULT_SUBSET_ROWS = DEFAULT_SUBSET_DIR / SUBSET_ROWS_FILENAME
DEFAULT_STABILITY_SUMMARY_ROWS = DEFAULT_STABILITY_DIR / STABILITY_SUMMARY_ROWS_FILENAME
DEFAULT_OUTPUT_DIR = COMBINED_DIR / (
    "basin_transition_aligned_core_handle_selector_field34_cc_c0_p8_v0"
)

PLAN_ROWS_FILENAME = "aligned_core_handle_selector_plan_rows.csv"
NODE_SCORE_ROWS_FILENAME = "aligned_core_handle_selector_node_score_rows.csv"
ROWS_FILENAME = "aligned_core_handle_selector_rows.csv"
SUMMARY_ROWS_FILENAME = "aligned_core_handle_selector_summary_rows.csv"
CONFIG_FILENAME = "aligned_core_handle_selector_config.json"
SUMMARY_FILENAME = "aligned_core_handle_selector_summary.json"
REPORT_FILENAME = "aligned_core_handle_selector_report.md"

SUFFICIENT_VERDICT = "sufficient_full_core_quality_match"


def _parse_policy_tuple(value: str) -> tuple[str, ...]:
    text = str(value).strip()
    if not text:
        return ALIGNED_CORE_HANDLE_SELECTOR_POLICIES
    parsed = tuple(part.strip() for part in text.split(",") if part.strip())
    return parsed or ALIGNED_CORE_HANDLE_SELECTOR_POLICIES


def _markdown_table(frame: pd.DataFrame, *, max_rows: int = 40) -> list[str]:
    if frame.empty:
        return []
    display = frame.head(max_rows)
    columns = list(display.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in display.iterrows():
        values: list[str] = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append("" if math.isnan(value) else f"{value:.6g}")
            else:
                values.append("" if pd.isna(value) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _minimal_sufficient_subset_id(subset_rows: pd.DataFrame) -> str:
    sufficient = subset_rows[
        subset_rows["handle_subset_verdict"].astype(str).eq(SUFFICIENT_VERDICT)
    ].copy()
    if sufficient.empty:
        return ""
    sufficient["_subset_size"] = pd.to_numeric(
        sufficient["subset_size"],
        errors="coerce",
    ).fillna(math.inf)
    sufficient["_quality_gain"] = pd.to_numeric(
        sufficient["operator_delta_q_gain_vs_source"],
        errors="coerce",
    ).fillna(-math.inf)
    row = sufficient.sort_values(
        ["_subset_size", "_quality_gain"],
        ascending=[True, False],
    ).iloc[0]
    return str(row["subset_node_ids"])


def _with_selector_outcomes(
    plans: pd.DataFrame,
    *,
    subset_rows: pd.DataFrame,
    stability_summary_rows: pd.DataFrame | None,
) -> pd.DataFrame:
    if plans.empty:
        return pd.DataFrame()
    subset_keep = [
        "subset_node_ids",
        "handle_subset_verdict",
        "recovers_required_aligned_core",
        "required_aligned_core_hit_count",
        "required_aligned_core_hit_fraction",
        "operator_delta_q_gain_vs_source",
        "quality_gap_vs_full_handle_set",
        "quality_gain_per_bundle_node",
        "state_delta_q_vs_vanilla",
        "operator_final_aligned_changed_support_node_ids",
    ]
    missing = set(subset_keep) - set(subset_rows.columns)
    if missing:
        raise ValueError(f"subset rows are missing required columns: {sorted(missing)}")
    out = plans.merge(
        subset_rows[subset_keep],
        on="subset_node_ids",
        how="left",
        validate="many_to_one",
    )
    if stability_summary_rows is not None and not stability_summary_rows.empty:
        stability_keep = [
            "subset_node_ids",
            "evaluation_count",
            "stable_sufficient_count",
            "stable_sufficient_fraction",
            "mean_delta_q_gain_vs_source",
            "mean_quality_gap_vs_full_handle_set",
            "mean_required_core_hit_fraction",
            "verdict_counts",
        ]
        available = [column for column in stability_keep if column in stability_summary_rows]
        out = out.merge(
            stability_summary_rows[available].drop_duplicates("subset_node_ids"),
            on="subset_node_ids",
            how="left",
        )
    minimal_id = _minimal_sufficient_subset_id(subset_rows)
    out["matches_minimal_sufficient_subset"] = out["subset_node_ids"].astype(str).eq(
        minimal_id
    )
    out["selector_reaches_sufficient"] = out["handle_subset_verdict"].astype(str).eq(
        SUFFICIENT_VERDICT
    )
    return out


def _summary_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    summaries: list[dict[str, Any]] = []
    for policy, group in rows.groupby("selector_policy", sort=False):
        group = group.sort_values("subset_size")
        sufficient = group[group["selector_reaches_sufficient"].astype(bool)]
        first = sufficient.iloc[0] if not sufficient.empty else None
        k4 = group[group["subset_size"].astype(int).eq(4)]
        k4_row = k4.iloc[0] if not k4.empty else None
        best = group.sort_values(
            [
                "operator_delta_q_gain_vs_source",
                "required_aligned_core_hit_fraction",
                "subset_size",
            ],
            ascending=[False, False, True],
        ).iloc[0]
        summaries.append(
            {
                "selector_policy": policy,
                "selector_feature_family": str(group.iloc[0]["selector_feature_family"]),
                "selector_uses_replay_features": bool(
                    group.iloc[0]["selector_uses_replay_features"]
                ),
                "evaluated_prefix_count": int(len(group)),
                "first_sufficient_k": (
                    int(first["subset_size"]) if first is not None else math.nan
                ),
                "first_sufficient_subset_node_ids": (
                    str(first["subset_node_ids"]) if first is not None else ""
                ),
                "first_sufficient_delta_q_gain_vs_source": (
                    float(first["operator_delta_q_gain_vs_source"])
                    if first is not None
                    else math.nan
                ),
                "k4_subset_node_ids": (
                    str(k4_row["subset_node_ids"]) if k4_row is not None else ""
                ),
                "k4_verdict": (
                    str(k4_row["handle_subset_verdict"]) if k4_row is not None else ""
                ),
                "k4_delta_q_gain_vs_source": (
                    float(k4_row["operator_delta_q_gain_vs_source"])
                    if k4_row is not None
                    else math.nan
                ),
                "k4_quality_gap_vs_full_handle_set": (
                    float(k4_row["quality_gap_vs_full_handle_set"])
                    if k4_row is not None
                    else math.nan
                ),
                "k4_stable_sufficient_fraction": (
                    float(k4_row["stable_sufficient_fraction"])
                    if k4_row is not None
                    and "stable_sufficient_fraction" in k4_row
                    and not pd.isna(k4_row["stable_sufficient_fraction"])
                    else math.nan
                ),
                "k4_matches_minimal_sufficient_subset": (
                    bool(k4_row["matches_minimal_sufficient_subset"])
                    if k4_row is not None
                    else False
                ),
                "best_quality_subset_size": int(best["subset_size"]),
                "best_quality_subset_node_ids": str(best["subset_node_ids"]),
                "best_delta_q_gain_vs_source": float(
                    best["operator_delta_q_gain_vs_source"]
                ),
                "best_quality_gap_vs_full_handle_set": float(
                    best["quality_gap_vs_full_handle_set"]
                ),
            }
        )
    return pd.DataFrame(summaries)


def _write_report(
    path: Path,
    *,
    rows: pd.DataFrame,
    score_rows: pd.DataFrame,
    summary_rows: pd.DataFrame,
) -> None:
    summary_cols = [
        "selector_policy",
        "selector_feature_family",
        "selector_uses_replay_features",
        "first_sufficient_k",
        "first_sufficient_subset_node_ids",
        "first_sufficient_delta_q_gain_vs_source",
        "k4_subset_node_ids",
        "k4_verdict",
        "k4_delta_q_gain_vs_source",
        "k4_quality_gap_vs_full_handle_set",
        "k4_stable_sufficient_fraction",
        "k4_matches_minimal_sufficient_subset",
        "best_quality_subset_size",
        "best_quality_subset_node_ids",
    ]
    row_cols = [
        "selector_policy",
        "subset_size",
        "selector_ordered_node_ids",
        "subset_node_ids",
        "handle_subset_verdict",
        "required_aligned_core_hit_fraction",
        "operator_delta_q_gain_vs_source",
        "quality_gap_vs_full_handle_set",
        "stable_sufficient_fraction",
        "matches_minimal_sufficient_subset",
    ]
    score_cols = [
        "selector_policy",
        "selector_rank",
        "node",
        "selector_feature_family",
        "aligned_change_count",
        "selected_target_count",
        "source_action_count",
        "source_mutable_count",
        "max_pull_to_context",
        "candidate_label",
    ]
    lines = [
        "# Aligned-Core Handle Selector Probe",
        "",
        "This diagnostic grades top-k handle selectors against the exhaustive",
        "subset outcome table. It is a selector replay, not a new Leiden run.",
        "",
        "## Selector Summary",
        "",
    ]
    lines.extend(
        _markdown_table(
            summary_rows[[c for c in summary_cols if c in summary_rows]],
            max_rows=20,
        )
    )
    lines.extend(["", "## Selector Prefix Rows", ""])
    lines.extend(_markdown_table(rows[[c for c in row_cols if c in rows]], max_rows=80))
    lines.extend(["", "## Node Scores", ""])
    lines.extend(
        _markdown_table(
            score_rows[[c for c in score_cols if c in score_rows]],
            max_rows=80,
        )
    )
    lines.extend(
        [
            "",
            "## Interpretation Guardrail",
            "",
            "- Replay-frontier policies use prior diagnostic outcomes and are not operator-ready.",
            "- Local-graph-proxy policies are closer to deployable selectors, but still need cross-case validation.",
            "- A selector is interesting only if it reaches the sufficient subset with small k and keeps material quality, not merely positive delta_q.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_probe(
    *,
    frontier_rows_path: Path,
    subset_rows_path: Path,
    stability_summary_rows_path: Path | None,
    output_dir: Path,
    selector_policies: tuple[str, ...],
    min_target_change_count: int,
    min_subset_size: int,
    max_subset_size: int | None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frontier_rows = pd.read_csv(frontier_rows_path)
    subset_rows = pd.read_csv(subset_rows_path)
    stability_summary_rows = (
        pd.read_csv(stability_summary_rows_path)
        if stability_summary_rows_path is not None
        and stability_summary_rows_path.exists()
        else None
    )
    plans, score_rows = build_aligned_core_handle_selector_plan_rows(
        frontier_rows,
        selector_policies=selector_policies,
        min_target_change_count=min_target_change_count,
        min_subset_size=min_subset_size,
        max_subset_size=max_subset_size,
    )
    rows = _with_selector_outcomes(
        plans,
        subset_rows=subset_rows,
        stability_summary_rows=stability_summary_rows,
    )
    summary_rows = _summary_rows(rows)
    plans.to_csv(output_dir / PLAN_ROWS_FILENAME, index=False)
    score_rows.to_csv(output_dir / NODE_SCORE_ROWS_FILENAME, index=False)
    rows.to_csv(output_dir / ROWS_FILENAME, index=False)
    summary_rows.to_csv(output_dir / SUMMARY_ROWS_FILENAME, index=False)
    run_config = {
        "frontier_rows_path": str(frontier_rows_path),
        "subset_rows_path": str(subset_rows_path),
        "stability_summary_rows_path": (
            str(stability_summary_rows_path) if stability_summary_rows_path else ""
        ),
        "output_dir": str(output_dir),
        "selector_policies": list(selector_policies),
        "min_target_change_count": int(min_target_change_count),
        "min_subset_size": int(min_subset_size),
        "max_subset_size": int(max_subset_size) if max_subset_size is not None else 0,
    }
    (output_dir / CONFIG_FILENAME).write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    payload = {
        "schema": "leiden_basin_aligned_core_handle_selector_probe.v0",
        "output_dir": str(output_dir),
        "selector_policy_count": int(len(selector_policies)),
        "plan_count": int(len(plans)),
        "row_count": int(len(rows)),
        "node_score_row_count": int(len(score_rows)),
        "summary_row_count": int(len(summary_rows)),
        "local_graph_proxy_k4_success_count": int(
            summary_rows[
                summary_rows["selector_feature_family"].astype(str).eq(
                    "local_graph_proxy"
                )
                & summary_rows["k4_matches_minimal_sufficient_subset"].astype(bool)
            ].shape[0]
        )
        if not summary_rows.empty
        else 0,
        "paths": {
            "plan_rows": str(output_dir / PLAN_ROWS_FILENAME),
            "node_score_rows": str(output_dir / NODE_SCORE_ROWS_FILENAME),
            "rows": str(output_dir / ROWS_FILENAME),
            "summary_rows": str(output_dir / SUMMARY_ROWS_FILENAME),
            "report": str(output_dir / REPORT_FILENAME),
        },
    }
    (output_dir / SUMMARY_FILENAME).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(
        output_dir / REPORT_FILENAME,
        rows=rows,
        score_rows=score_rows,
        summary_rows=summary_rows,
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontier-rows", type=Path, default=DEFAULT_FRONTIER_ROWS)
    parser.add_argument("--subset-rows", type=Path, default=DEFAULT_SUBSET_ROWS)
    parser.add_argument(
        "--stability-summary-rows",
        type=Path,
        default=DEFAULT_STABILITY_SUMMARY_ROWS,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--selector-policies",
        default=",".join(ALIGNED_CORE_HANDLE_SELECTOR_POLICIES),
    )
    parser.add_argument("--min-target-change-count", type=int, default=5)
    parser.add_argument("--min-subset-size", type=int, default=1)
    parser.add_argument("--max-subset-size", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_probe(
        frontier_rows_path=args.frontier_rows,
        subset_rows_path=args.subset_rows,
        stability_summary_rows_path=args.stability_summary_rows,
        output_dir=args.output_dir,
        selector_policies=_parse_policy_tuple(args.selector_policies),
        min_target_change_count=args.min_target_change_count,
        min_subset_size=args.min_subset_size,
        max_subset_size=(None if int(args.max_subset_size) <= 0 else int(args.max_subset_size)),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
