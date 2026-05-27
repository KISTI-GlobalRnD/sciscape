#!/usr/bin/env python3
"""Probe stability of compact aligned-core handle subsets.

This diagnostic replays selected handle subsets across source cases and polish
seed offsets. It tests whether the minimal sufficient subset discovered in one
p8 run is stable or only a single-seed artifact.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT))

from run_leiden_basin_aligned_core_handle_subset_probe import (  # noqa: E402
    DEFAULT_FRONTIER_ROWS,
    ROWS_FILENAME as SUBSET_ROWS_FILENAME,
    _add_full_set_comparisons,
    _evaluate_subsets,
    _markdown_table,
)
from run_leiden_basin_attachment_margin_cross_prefix_probe import (  # noqa: E402
    DEFAULT_SOURCE_RECOVERY_POLICY,
    SUMMARY_ROWS_FILENAME as ATTACHMENT_SUMMARY_ROWS_FILENAME,
)
from run_leiden_basin_attachment_margin_joint_bundle_probe import (  # noqa: E402
    DEFAULT_ATTACHMENT_DIR,
)
from run_leiden_basin_attachment_margin_stage2_recovery import (  # noqa: E402
    DEFAULT_CONTROL_DIR,
    _control_context,
    _load_control_summary,
    _rebuild_source_state,
)
from sciscape.clustering.leiden_basin_profile import parse_node_ids  # noqa: E402
from sciscape.clustering.leiden_basin_search import (  # noqa: E402
    node_csv,
    select_aligned_core_boundary_nodes,
    unique_sorted_u32,
)


COMBINED_DIR = DEFAULT_ATTACHMENT_DIR.parent
DEFAULT_SUBSET_DIR = COMBINED_DIR / "basin_transition_aligned_core_handle_subset_field34_cc_c0_p8_v0"
DEFAULT_SUBSET_ROWS = DEFAULT_SUBSET_DIR / SUBSET_ROWS_FILENAME
DEFAULT_OUTPUT_DIR = COMBINED_DIR / (
    "basin_transition_aligned_core_handle_stability_field34_cc_c0_p6_p8_p10_v0"
)

PLAN_ROWS_FILENAME = "aligned_core_handle_stability_plan_rows.csv"
ROWS_FILENAME = "aligned_core_handle_stability_rows.csv"
SUMMARY_ROWS_FILENAME = "aligned_core_handle_stability_summary_rows.csv"
CONFIG_FILENAME = "aligned_core_handle_stability_config.json"
SUMMARY_FILENAME = "aligned_core_handle_stability_summary.json"
REPORT_FILENAME = "aligned_core_handle_stability_report.md"


def _parse_csv_tuple(value: str, default: tuple[str, ...]) -> tuple[str, ...]:
    text = str(value).strip()
    if not text:
        return default
    parsed = tuple(part.strip() for part in text.split(",") if part.strip())
    return parsed or default


def _parse_int_tuple(value: str, default: tuple[int, ...]) -> tuple[int, ...]:
    text = str(value).strip()
    if not text:
        return default
    parsed = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    return parsed or default


def select_stability_subsets(
    subset_rows: pd.DataFrame,
    *,
    max_partial_rows: int = 3,
) -> pd.DataFrame:
    """Select sufficient and near-miss subsets for stability replay."""
    if subset_rows.empty:
        return pd.DataFrame()
    required = {
        "subset_node_ids",
        "subset_size",
        "handle_subset_verdict",
        "operator_delta_q_gain_vs_source",
        "required_aligned_core_hit_count",
        "quality_gap_vs_full_handle_set",
    }
    missing = required - set(subset_rows.columns)
    if missing:
        raise ValueError(f"subset rows are missing required columns: {sorted(missing)}")
    rows = subset_rows.copy()
    rows["_subset_size"] = pd.to_numeric(rows["subset_size"], errors="coerce").fillna(0)
    rows["_hit_count"] = pd.to_numeric(
        rows["required_aligned_core_hit_count"],
        errors="coerce",
    ).fillna(0)
    rows["_quality_gap"] = pd.to_numeric(
        rows["quality_gap_vs_full_handle_set"],
        errors="coerce",
    ).fillna(-math.inf)
    rows["_quality_gain"] = pd.to_numeric(
        rows["operator_delta_q_gain_vs_source"],
        errors="coerce",
    ).fillna(-math.inf)
    full_size = int(rows["_subset_size"].max())
    selected: list[pd.Series] = []
    sufficient = rows[
        rows["handle_subset_verdict"].astype(str).eq("sufficient_full_core_quality_match")
    ].sort_values(["_subset_size", "_quality_gain"], ascending=[True, False])
    selected.extend(row for _, row in sufficient.iterrows())
    partial = rows[
        ~rows["handle_subset_verdict"].astype(str).eq("sufficient_full_core_quality_match")
    ].sort_values(
        ["_hit_count", "_quality_gap", "_quality_gain", "_subset_size"],
        ascending=[False, False, False, True],
    )
    selected.extend(row for _, row in partial.head(int(max_partial_rows)).iterrows())

    deduped: dict[str, pd.Series] = {}
    for row in selected:
        deduped.setdefault(str(row["subset_node_ids"]), row)
    out_rows: list[dict[str, Any]] = []
    for rank, row in enumerate(deduped.values(), start=1):
        subset_size = int(row["_subset_size"])
        verdict = str(row["handle_subset_verdict"])
        role = "full_handle_set" if subset_size == full_size else verdict
        if verdict == "sufficient_full_core_quality_match" and subset_size < full_size:
            role = "minimal_sufficient"
        elif verdict != "sufficient_full_core_quality_match":
            role = f"near_miss_size_{subset_size}"
        out_rows.append(
            {
                "plan_rank": int(rank),
                "subset_role": role,
                "subset_size": subset_size,
                "subset_node_ids": str(row["subset_node_ids"]),
                "baseline_verdict": verdict,
                "baseline_required_hit_count": int(row["_hit_count"]),
                "baseline_quality_gap_vs_full_handle_set": float(row["_quality_gap"]),
                "baseline_delta_q_gain_vs_source": float(row["_quality_gain"]),
            }
        )
    return pd.DataFrame(out_rows)


def _plans_for_evaluation(selected: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    full_nodes = unique_sorted_u32(
        np.concatenate(
            [
                unique_sorted_u32(parse_node_ids(value))
                for value in selected["subset_node_ids"].astype(str)
            ]
        )
    )
    full_ids = node_csv(full_nodes)
    for _, row in selected.iterrows():
        subset = unique_sorted_u32(parse_node_ids(row["subset_node_ids"]))
        rows.append(
            {
                "plan_rank": int(row["plan_rank"]),
                "plan_kind": "stability_handle_subset",
                "subset_size": int(subset.size),
                "target_node_count": int(full_nodes.size),
                "full_target_node_ids": full_ids,
                "subset_node_ids": node_csv(subset),
                "bundle_node_count": int(subset.size),
                "bundle_node_ids": node_csv(subset),
            }
        )
    return pd.DataFrame(rows)


def _with_stability_verdicts(
    rows: pd.DataFrame,
    *,
    quality_tolerance: float,
) -> pd.DataFrame:
    if rows.empty:
        return rows
    out = rows.copy()
    out["stability_verdict"] = [
        "stable_sufficient"
        if bool(row["recovers_required_aligned_core"])
        and float(row["quality_gap_vs_full_handle_set"]) >= -float(quality_tolerance)
        else "stable_core_quality_lag"
        if bool(row["recovers_required_aligned_core"])
        else "partial_core"
        if float(row["operator_delta_q_gain_vs_source"]) > 0.0
        else "no_core_gain"
        for _, row in out.iterrows()
    ]
    return out


def _summary_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    summary: list[dict[str, Any]] = []
    for (subset_role, subset_ids), group in rows.groupby(
        ["subset_role", "subset_node_ids"],
        sort=False,
    ):
        stable = group[group["stability_verdict"].astype(str).eq("stable_sufficient")]
        summary.append(
            {
                "subset_role": subset_role,
                "subset_node_ids": subset_ids,
                "subset_size": int(group["subset_size"].iloc[0]),
                "evaluation_count": int(len(group)),
                "stable_sufficient_count": int(len(stable)),
                "stable_sufficient_fraction": float(len(stable)) / float(max(1, len(group))),
                "source_case_count": int(group["source_case"].astype(str).nunique()),
                "polish_seed_offset_count": int(
                    group["polish_seed_offset"].astype(int).nunique()
                ),
                "mean_delta_q_gain_vs_source": float(
                    group["operator_delta_q_gain_vs_source"].mean()
                ),
                "min_delta_q_gain_vs_source": float(
                    group["operator_delta_q_gain_vs_source"].min()
                ),
                "mean_quality_gap_vs_full_handle_set": float(
                    group["quality_gap_vs_full_handle_set"].mean()
                ),
                "min_quality_gap_vs_full_handle_set": float(
                    group["quality_gap_vs_full_handle_set"].min()
                ),
                "mean_required_core_hit_fraction": float(
                    group["required_aligned_core_hit_fraction"].mean()
                ),
                "verdict_counts": json.dumps(
                    group["stability_verdict"].value_counts().to_dict(),
                    sort_keys=True,
                ),
            }
        )
    return pd.DataFrame(summary)


def _write_report(
    path: Path,
    *,
    selected: pd.DataFrame,
    rows: pd.DataFrame,
    summary_rows: pd.DataFrame,
) -> None:
    selected_cols = [
        "plan_rank",
        "subset_role",
        "subset_size",
        "subset_node_ids",
        "baseline_verdict",
        "baseline_required_hit_count",
        "baseline_quality_gap_vs_full_handle_set",
    ]
    summary_cols = [
        "subset_role",
        "subset_node_ids",
        "evaluation_count",
        "stable_sufficient_count",
        "stable_sufficient_fraction",
        "source_case_count",
        "polish_seed_offset_count",
        "mean_delta_q_gain_vs_source",
        "min_delta_q_gain_vs_source",
        "mean_quality_gap_vs_full_handle_set",
        "min_quality_gap_vs_full_handle_set",
        "mean_required_core_hit_fraction",
        "verdict_counts",
    ]
    row_cols = [
        "source_case",
        "polish_seed_offset",
        "subset_role",
        "subset_node_ids",
        "required_aligned_core_hit_count",
        "required_aligned_core_hit_fraction",
        "recovers_required_aligned_core",
        "operator_delta_q_gain_vs_source",
        "quality_gap_vs_full_handle_set",
        "state_delta_q_vs_vanilla",
        "quality_minus_best_same_randomness_control",
        "stability_verdict",
        "operator_final_aligned_changed_support_node_ids",
    ]
    lines = [
        "# Aligned-Core Handle Stability Probe",
        "",
        "This diagnostic replays selected handle subsets across source cases and",
        "polish seed offsets. It tests whether the minimal sufficient subset is",
        "stable or only a single-run artifact.",
        "",
        "## Selected Subsets",
        "",
    ]
    lines.extend(_markdown_table(selected[[c for c in selected_cols if c in selected]], max_rows=20))
    lines.extend(["", "## Stability Summary", ""])
    lines.extend(
        _markdown_table(
            summary_rows[[c for c in summary_cols if c in summary_rows]],
            max_rows=20,
        )
    )
    lines.extend(["", "## Evaluation Rows", ""])
    display = rows.sort_values(
        [
            "source_case",
            "polish_seed_offset",
            "subset_size",
            "operator_delta_q_gain_vs_source",
        ],
        ascending=[True, True, True, False],
    )
    lines.extend(_markdown_table(display[[c for c in row_cols if c in display]], max_rows=120))
    lines.extend(
        [
            "",
            "## Interpretation Guardrail",
            "",
            "- Stability across polish seeds is still diagnostic, not an operator claim.",
            "- Cross-source rows are meaningful only as nearby-source pressure tests for the same node handles.",
            "- A subset is stable only when it recovers the required aligned core and matches the full-handle quality within tolerance in the same source/seed group.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_probe(
    *,
    subset_rows_path: Path,
    frontier_rows_path: Path,
    attachment_dir: Path,
    control_dir: Path | None,
    output_dir: Path,
    source_cases: tuple[str, ...],
    polish_seed_offsets: tuple[int, ...],
    source_recovery_policy: str,
    recovery_seed: int,
    max_partial_rows: int,
    min_target_change_count: int,
    min_boundary_change_count: int,
    quality_tolerance: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = select_stability_subsets(
        pd.read_csv(subset_rows_path),
        max_partial_rows=max_partial_rows,
    )
    if selected.empty:
        raise ValueError("No stability subsets selected")
    plans = _plans_for_evaluation(selected)
    frontier_rows = pd.read_csv(frontier_rows_path)
    selection = select_aligned_core_boundary_nodes(
        frontier_rows,
        min_target_change_count=min_target_change_count,
        min_boundary_change_count=min_boundary_change_count,
        max_context_core_nodes=0,
    )
    required_nodes = unique_sorted_u32(
        np.concatenate([selection.target_nodes, selection.boundary_core_nodes])
    )
    stage1_rows = pd.read_csv(attachment_dir / ATTACHMENT_SUMMARY_ROWS_FILENAME)
    control_rows = _load_control_summary(control_dir)
    all_rows: list[pd.DataFrame] = []
    for source_case in source_cases:
        stage1 = stage1_rows[stage1_rows["source_case"].astype(str).eq(str(source_case))]
        if stage1.empty:
            raise ValueError(f"No attachment summary row for source_case={source_case}")
        config, case_ctx, source_state, source_row, effective_seed, meta = _rebuild_source_state(
            source_move_dir=Path(str(stage1.iloc[0]["source_move_dir"])),
            source_case=source_case,
            source_recovery_policy=source_recovery_policy,
            requested_recovery_seed=recovery_seed,
        )
        for offset in polish_seed_offsets:
            rows = _evaluate_subsets(
                plans=plans,
                config=config,
                case_ctx=case_ctx,
                source_state=source_state,
                source_row=source_row,
                meta=meta,
                control_ctx=_control_context(control_rows, source_case),
                recovery_seed=effective_seed,
                polish_seed_offset=int(offset),
            )
            rows["polish_seed_offset"] = int(offset)
            rows = rows.merge(
                selected[["plan_rank", "subset_role", "baseline_verdict"]],
                on="plan_rank",
                how="left",
            )
            rows = _add_full_set_comparisons(
                rows,
                required_nodes=required_nodes,
                quality_tolerance=quality_tolerance,
            )
            rows = _with_stability_verdicts(
                rows,
                quality_tolerance=quality_tolerance,
            )
            all_rows.append(rows)
    rows = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    summary_rows = _summary_rows(rows)
    selected.to_csv(output_dir / PLAN_ROWS_FILENAME, index=False)
    rows.to_csv(output_dir / ROWS_FILENAME, index=False)
    summary_rows.to_csv(output_dir / SUMMARY_ROWS_FILENAME, index=False)
    run_config = {
        "subset_rows_path": str(subset_rows_path),
        "frontier_rows_path": str(frontier_rows_path),
        "attachment_dir": str(attachment_dir),
        "control_dir": str(control_dir) if control_dir is not None else "",
        "output_dir": str(output_dir),
        "source_cases": list(source_cases),
        "polish_seed_offsets": list(polish_seed_offsets),
        "source_recovery_policy": source_recovery_policy,
        "requested_recovery_seed": int(recovery_seed),
        "max_partial_rows": int(max_partial_rows),
        "required_aligned_core_node_ids": node_csv(required_nodes),
        "quality_tolerance": float(quality_tolerance),
    }
    (output_dir / CONFIG_FILENAME).write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    stable_count = int(
        rows["stability_verdict"].astype(str).eq("stable_sufficient").sum()
    ) if not rows.empty else 0
    payload = {
        "schema": "leiden_basin_aligned_core_handle_stability_probe.v0",
        "output_dir": str(output_dir),
        "selected_subset_count": int(len(selected)),
        "row_count": int(len(rows)),
        "summary_row_count": int(len(summary_rows)),
        "stable_sufficient_count": stable_count,
        "paths": {
            "plan_rows": str(output_dir / PLAN_ROWS_FILENAME),
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
        selected=selected,
        rows=rows,
        summary_rows=summary_rows,
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-rows", type=Path, default=DEFAULT_SUBSET_ROWS)
    parser.add_argument("--frontier-rows", type=Path, default=DEFAULT_FRONTIER_ROWS)
    parser.add_argument("--attachment-dir", type=Path, default=DEFAULT_ATTACHMENT_DIR)
    parser.add_argument("--control-dir", type=Path, default=DEFAULT_CONTROL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source-cases", default="p6_wide,p8_fullctx,p10_wide")
    parser.add_argument("--polish-seed-offsets", default="2000,3000,4000")
    parser.add_argument("--source-recovery-policy", default=DEFAULT_SOURCE_RECOVERY_POLICY)
    parser.add_argument("--recovery-seed", type=int, default=0)
    parser.add_argument("--max-partial-rows", type=int, default=3)
    parser.add_argument("--min-target-change-count", type=int, default=5)
    parser.add_argument("--min-boundary-change-count", type=int, default=5)
    parser.add_argument("--quality-tolerance", type=float, default=1e-9)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_probe(
        subset_rows_path=args.subset_rows,
        frontier_rows_path=args.frontier_rows,
        attachment_dir=args.attachment_dir,
        control_dir=args.control_dir,
        output_dir=args.output_dir,
        source_cases=_parse_csv_tuple(
            args.source_cases,
            default=("p6_wide", "p8_fullctx", "p10_wide"),
        ),
        polish_seed_offsets=_parse_int_tuple(
            args.polish_seed_offsets,
            default=(2000, 3000, 4000),
        ),
        source_recovery_policy=args.source_recovery_policy,
        recovery_seed=args.recovery_seed,
        max_partial_rows=args.max_partial_rows,
        min_target_change_count=args.min_target_change_count,
        min_boundary_change_count=args.min_boundary_change_count,
        quality_tolerance=args.quality_tolerance,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
