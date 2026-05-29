#!/usr/bin/env python3
"""Evaluate source-local handle selectors without aligned-core frontier input.

This diagnostic selects top-k handles from source-local attachment/gate-pull
features, then runs candidate-label transplant plus bounded polish. The p8
aligned-core frontier is used only as an evaluation target for the current c0
slice, not as an input to the selector.
"""

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

from run_leiden_basin_aligned_core_boundary_operator_probe import (  # noqa: E402
    DEFAULT_FRONTIER_ROWS,
)
from run_leiden_basin_aligned_core_handle_subset_probe import (  # noqa: E402
    _evaluate_subsets,
)
from run_leiden_basin_attachment_margin_cross_prefix_probe import (  # noqa: E402
    DEFAULT_OUTPUT_DIR as DEFAULT_ATTACHMENT_DIR,
    DEFAULT_SOURCE_RECOVERY_POLICY,
    SCORE_ROWS_FILENAME as ATTACHMENT_SCORE_ROWS_FILENAME,
    SUMMARY_ROWS_FILENAME as ATTACHMENT_SUMMARY_ROWS_FILENAME,
)
from run_leiden_basin_attachment_margin_stage2_recovery import (  # noqa: E402
    DEFAULT_CONTROL_DIR,
    _control_context,
    _load_control_summary,
    _rebuild_source_state,
)
from sciscape.clustering.leiden_basin_profile import (  # noqa: E402
    parse_node_ids,
)
from sciscape.clustering.leiden_basin_search import (  # noqa: E402
    LOCAL_HANDLE_SELECTOR_POLICIES,
    build_local_handle_selector_plan_rows,
    intersect_sorted_u32,
    node_csv,
    select_aligned_core_boundary_nodes,
    unique_sorted_u32,
)

COMBINED_DIR = DEFAULT_ATTACHMENT_DIR.parent
DEFAULT_ATTACHMENT_SCORE_ROWS = DEFAULT_ATTACHMENT_DIR / ATTACHMENT_SCORE_ROWS_FILENAME
DEFAULT_ATTACHMENT_SUMMARY_ROWS = DEFAULT_ATTACHMENT_DIR / ATTACHMENT_SUMMARY_ROWS_FILENAME
DEFAULT_OUTPUT_DIR = COMBINED_DIR / (
    "basin_transition_local_handle_selector_field34_cc_c0_p6_p8_p10_coherent_v0"
)

PLAN_ROWS_FILENAME = "local_handle_selector_plan_rows.csv"
NODE_SCORE_ROWS_FILENAME = "local_handle_selector_node_score_rows.csv"
ROWS_FILENAME = "local_handle_selector_rows.csv"
SUMMARY_ROWS_FILENAME = "local_handle_selector_summary_rows.csv"
CONFIG_FILENAME = "local_handle_selector_config.json"
SUMMARY_FILENAME = "local_handle_selector_summary.json"
REPORT_FILENAME = "local_handle_selector_report.md"

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

def _required_core_from_frontier(
    frontier_rows_path: Path,
    *,
    min_target_change_count: int,
    min_boundary_change_count: int,
) -> np.ndarray:
    frontier_rows = pd.read_csv(frontier_rows_path)
    selection = select_aligned_core_boundary_nodes(
        frontier_rows,
        min_target_change_count=min_target_change_count,
        min_boundary_change_count=min_boundary_change_count,
        max_context_core_nodes=0,
    )
    return unique_sorted_u32(
        np.concatenate([selection.target_nodes, selection.boundary_core_nodes])
    )

def _with_required_core_verdicts(
    rows: pd.DataFrame,
    *,
    required_nodes: np.ndarray,
    min_material_q_gain: float,
) -> pd.DataFrame:
    if rows.empty:
        return rows
    required = unique_sorted_u32(required_nodes)
    out = rows.copy()
    if required.size == 0:
        out["evaluation_required_core_node_ids"] = ""
        out["evaluation_required_core_count"] = 0
        out["evaluation_required_core_hit_count"] = 0
        out["evaluation_required_core_hit_fraction"] = math.nan
        out["evaluation_required_core_hit_node_ids"] = ""
        out["evaluation_recovers_required_core"] = False
        out["local_selector_verdict"] = [
            "local_quality_same_randomness_win"
            if float(row.get("quality_minus_best_same_randomness_control", -math.inf)) >= 0.0
            and float(row["operator_delta_q_gain_vs_source"]) >= float(min_material_q_gain)
            else "local_material_quality_gain"
            if float(row["operator_delta_q_gain_vs_source"]) >= float(min_material_q_gain)
            else "local_no_material_gain"
            for _, row in out.iterrows()
        ]
        return out
    hit_counts: list[int] = []
    hit_ids: list[str] = []
    for value in out["operator_final_aligned_changed_support_node_ids"]:
        aligned = unique_sorted_u32(parse_node_ids(value))
        hits = intersect_sorted_u32(aligned, required)
        hit_counts.append(int(hits.size))
        hit_ids.append(node_csv(hits))
    out["evaluation_required_core_node_ids"] = node_csv(required)
    out["evaluation_required_core_count"] = int(required.size)
    out["evaluation_required_core_hit_count"] = hit_counts
    out["evaluation_required_core_hit_fraction"] = (
        out["evaluation_required_core_hit_count"].astype(float)
        / float(max(1, int(required.size)))
    )
    out["evaluation_required_core_hit_node_ids"] = hit_ids
    out["evaluation_recovers_required_core"] = out[
        "evaluation_required_core_hit_count"
    ].astype(int).eq(int(required.size))
    out["local_selector_verdict"] = [
        "local_required_core_same_randomness_win"
        if bool(row["evaluation_recovers_required_core"])
        and float(row.get("quality_minus_best_same_randomness_control", -math.inf)) >= 0.0
        else "local_required_core_quality_gain"
        if bool(row["evaluation_recovers_required_core"])
        and float(row["operator_delta_q_gain_vs_source"]) >= float(min_material_q_gain)
        else "local_partial_core_quality_gain"
        if float(row["operator_delta_q_gain_vs_source"]) >= float(min_material_q_gain)
        else "local_no_material_gain"
        for _, row in out.iterrows()
    ]
    return out

def _summary_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    summaries: list[dict[str, Any]] = []
    group_cols = ["source_case", "selector_policy"]
    for (source_case, policy), group in rows.groupby(group_cols, sort=False):
        group = group.sort_values("selected_k")
        required = group[group["evaluation_recovers_required_core"].astype(bool)]
        same_randomness = group[
            group["local_selector_verdict"]
            .astype(str)
            .eq("local_required_core_same_randomness_win")
        ]
        k4 = group[group["selected_k"].astype(int).eq(4)]
        k4_row = k4.iloc[0] if not k4.empty else None
        best = group.sort_values(
            [
                "state_quality",
                "evaluation_required_core_hit_fraction",
                "selected_k",
            ],
            ascending=[False, False, True],
        ).iloc[0]
        summaries.append(
            {
                "source_case": source_case,
                "selector_policy": policy,
                "evaluated_k_count": int(len(group)),
                "first_required_core_k": (
                    int(required.iloc[0]["selected_k"]) if not required.empty else math.nan
                ),
                "first_required_core_subset_node_ids": (
                    str(required.iloc[0]["subset_node_ids"]) if not required.empty else ""
                ),
                "first_same_randomness_win_k": (
                    int(same_randomness.iloc[0]["selected_k"])
                    if not same_randomness.empty
                    else math.nan
                ),
                "k4_subset_node_ids": (
                    str(k4_row["subset_node_ids"]) if k4_row is not None else ""
                ),
                "k4_required_core_hit_fraction": (
                    float(k4_row["evaluation_required_core_hit_fraction"])
                    if k4_row is not None
                    else math.nan
                ),
                "k4_delta_q_gain_vs_source": (
                    float(k4_row["operator_delta_q_gain_vs_source"])
                    if k4_row is not None
                    else math.nan
                ),
                "k4_state_delta_q_vs_vanilla": (
                    float(k4_row["state_delta_q_vs_vanilla"])
                    if k4_row is not None
                    else math.nan
                ),
                "k4_quality_minus_same_randomness_control": (
                    float(k4_row.get("quality_minus_best_same_randomness_control", math.nan))
                    if k4_row is not None
                    else math.nan
                ),
                "k4_verdict": (
                    str(k4_row["local_selector_verdict"]) if k4_row is not None else ""
                ),
                "best_selected_k": int(best["selected_k"]),
                "best_subset_node_ids": str(best["subset_node_ids"]),
                "best_delta_q_gain_vs_source": float(
                    best["operator_delta_q_gain_vs_source"]
                ),
                "best_required_core_hit_fraction": float(
                    best["evaluation_required_core_hit_fraction"]
                ),
                "best_quality_minus_same_randomness_control": float(
                    best.get("quality_minus_best_same_randomness_control", math.nan)
                ),
                "best_verdict": str(best["local_selector_verdict"]),
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
        "source_case",
        "selector_policy",
        "first_required_core_k",
        "first_required_core_subset_node_ids",
        "first_same_randomness_win_k",
        "k4_subset_node_ids",
        "k4_required_core_hit_fraction",
        "k4_delta_q_gain_vs_source",
        "k4_state_delta_q_vs_vanilla",
        "k4_quality_minus_same_randomness_control",
        "k4_verdict",
        "best_selected_k",
        "best_subset_node_ids",
        "best_delta_q_gain_vs_source",
        "best_required_core_hit_fraction",
        "best_verdict",
    ]
    row_cols = [
        "source_case",
        "selector_policy",
        "selector_candidate_label",
        "selected_k",
        "selector_ordered_node_ids",
        "subset_node_ids",
        "evaluation_required_core_hit_fraction",
        "operator_delta_q_gain_vs_source",
        "state_delta_q_vs_vanilla",
        "state_target_progress_from_vanilla",
        "quality_minus_best_same_randomness_control",
        "local_selector_verdict",
    ]
    score_cols = [
        "source_case",
        "selector_policy",
        "selector_rank",
        "selector_candidate_label",
        "node",
        "gate_pull_margin_vs_current_source",
        "pull_to_gate_context",
        "in_source_action",
        "in_source_mutable",
        "candidate_label",
        "vanilla_label",
    ]
    lines = [
        "# Local Handle Selector Probe",
        "",
        "This diagnostic ranks handles from source-local attachment/gate-pull",
        "features and evaluates candidate-label transplant plus bounded polish.",
        "The aligned-core frontier is used only when requested as a c0 evaluation target.",
        "",
        "## Summary",
        "",
    ]
    lines.extend(
        _markdown_table(
            summary_rows[[c for c in summary_cols if c in summary_rows]],
            max_rows=80,
        )
    )
    lines.extend(["", "## Evaluation Rows", ""])
    display = rows.sort_values(
        ["source_case", "selector_policy", "selected_k"],
        ascending=[True, True, True],
    )
    lines.extend(_markdown_table(display[[c for c in row_cols if c in display]], max_rows=160))
    lines.extend(["", "## Top Node Scores", ""])
    top_scores = score_rows[score_rows["selector_rank"].astype(int).le(10)].copy()
    lines.extend(
        _markdown_table(
            top_scores[[c for c in score_cols if c in top_scores]],
            max_rows=160,
        )
    )
    lines.extend(
        [
            "",
            "## Interpretation Guardrail",
            "",
            "- Selector input is local graph/proxy feature rows, not the p8 aligned-core frontier.",
            "- The known six-node aligned core is used only for current-slice evaluation when `evaluation_core_mode=frontier`.",
            "- With `evaluation_core_mode=none`, verdicts are quality/control smoke labels rather than core-recovery labels.",
            "- A useful operator candidate must keep material QF and cost evidence beside support/core recovery.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def run_probe(
    *,
    attachment_score_rows_path: Path,
    attachment_summary_rows_path: Path,
    frontier_rows_path: Path,
    evaluation_core_mode: str,
    control_dir: Path | None,
    output_dir: Path,
    source_cases: tuple[str, ...],
    selector_policies: tuple[str, ...],
    selected_ks: tuple[int, ...],
    source_recovery_policy: str,
    recovery_seed: int,
    polish_seed_offset: int,
    min_target_change_count: int,
    min_boundary_change_count: int,
    min_material_q_gain: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    score_rows_all = pd.read_csv(attachment_score_rows_path)
    stage1_rows = pd.read_csv(attachment_summary_rows_path)
    control_rows = _load_control_summary(control_dir)
    if evaluation_core_mode == "frontier":
        required_core = _required_core_from_frontier(
            frontier_rows_path,
            min_target_change_count=min_target_change_count,
            min_boundary_change_count=min_boundary_change_count,
        )
    elif evaluation_core_mode == "none":
        required_core = np.asarray([], dtype=np.uint32)
    else:
        raise ValueError(f"Unknown evaluation core mode: {evaluation_core_mode}")
    all_plans: list[pd.DataFrame] = []
    all_scores: list[pd.DataFrame] = []
    all_rows: list[pd.DataFrame] = []
    for source_case in source_cases:
        plans, scores = build_local_handle_selector_plan_rows(
            score_rows_all,
            selector_policies=selector_policies,
            selected_ks=selected_ks,
            source_case=source_case,
        )
        if plans.empty:
            continue
        stage1 = stage1_rows[stage1_rows["source_case"].astype(str).eq(str(source_case))]
        if stage1.empty:
            raise ValueError(f"No attachment summary row for source_case={source_case}")
        config, case_ctx, source_state, source_row, effective_seed, meta = _rebuild_source_state(
            source_move_dir=Path(str(stage1.iloc[0]["source_move_dir"])),
            source_case=source_case,
            source_recovery_policy=source_recovery_policy,
            requested_recovery_seed=recovery_seed,
        )
        rows = _evaluate_subsets(
            plans=plans,
            config=config,
            case_ctx=case_ctx,
            source_state=source_state,
            source_row=source_row,
            meta=meta,
            control_ctx=_control_context(control_rows, source_case),
            recovery_seed=effective_seed,
            polish_seed_offset=polish_seed_offset,
        )
        rows["polish_seed_offset"] = int(polish_seed_offset)
        rows = rows.merge(
            plans[
                [
                    "plan_rank",
                    "selector_policy",
                    "selector_feature_family",
                    "selector_uses_replay_features",
                    "selector_candidate_label",
                    "selected_k",
                    "selector_ordered_node_ids",
                ]
            ],
            on="plan_rank",
            how="left",
        )
        rows = _with_required_core_verdicts(
            rows,
            required_nodes=required_core,
            min_material_q_gain=min_material_q_gain,
        )
        all_plans.append(plans)
        all_scores.append(scores)
        all_rows.append(rows)
    plan_rows = pd.concat(all_plans, ignore_index=True) if all_plans else pd.DataFrame()
    score_rows = pd.concat(all_scores, ignore_index=True) if all_scores else pd.DataFrame()
    rows = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    summary_rows = _summary_rows(rows)
    plan_rows.to_csv(output_dir / PLAN_ROWS_FILENAME, index=False)
    score_rows.to_csv(output_dir / NODE_SCORE_ROWS_FILENAME, index=False)
    rows.to_csv(output_dir / ROWS_FILENAME, index=False)
    summary_rows.to_csv(output_dir / SUMMARY_ROWS_FILENAME, index=False)
    run_config = {
        "attachment_score_rows_path": str(attachment_score_rows_path),
        "attachment_summary_rows_path": str(attachment_summary_rows_path),
        "frontier_rows_path": str(frontier_rows_path),
        "evaluation_core_mode": str(evaluation_core_mode),
        "control_dir": str(control_dir) if control_dir is not None else "",
        "output_dir": str(output_dir),
        "source_cases": list(source_cases),
        "selector_policies": list(selector_policies),
        "selected_ks": list(selected_ks),
        "source_recovery_policy": source_recovery_policy,
        "requested_recovery_seed": int(recovery_seed),
        "polish_seed_offset": int(polish_seed_offset),
        "evaluation_required_core_node_ids": node_csv(required_core),
        "min_material_q_gain": float(min_material_q_gain),
    }
    (output_dir / CONFIG_FILENAME).write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    payload = {
        "schema": "leiden_basin_local_handle_selector_probe.v0",
        "output_dir": str(output_dir),
        "source_case_count": int(len(source_cases)),
        "selector_policy_count": int(len(selector_policies)),
        "plan_count": int(len(plan_rows)),
        "row_count": int(len(rows)),
        "node_score_row_count": int(len(score_rows)),
        "summary_row_count": int(len(summary_rows)),
        "required_core_recovery_row_count": int(
            rows["evaluation_recovers_required_core"].astype(bool).sum()
        )
        if not rows.empty
        else 0,
        "same_randomness_win_row_count": int(
            rows["local_selector_verdict"]
            .astype(str)
            .isin(
                {
                    "local_required_core_same_randomness_win",
                    "local_quality_same_randomness_win",
                }
            )
            .sum()
        )
        if not rows.empty
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
    parser.add_argument("--attachment-score-rows", type=Path, default=DEFAULT_ATTACHMENT_SCORE_ROWS)
    parser.add_argument(
        "--attachment-summary-rows",
        type=Path,
        default=DEFAULT_ATTACHMENT_SUMMARY_ROWS,
    )
    parser.add_argument("--frontier-rows", type=Path, default=DEFAULT_FRONTIER_ROWS)
    parser.add_argument(
        "--evaluation-core-mode",
        choices=("frontier", "none"),
        default="frontier",
        help=(
            "Use the c0 frontier-derived required core for evaluation, or skip "
            "required-core evaluation for cross-case smoke probes."
        ),
    )
    parser.add_argument("--control-dir", type=Path, default=DEFAULT_CONTROL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source-cases", default="p6_wide,p8_fullctx,p10_wide")
    parser.add_argument(
        "--selector-policies",
        default="candidate_label_margin_coherent",
    )
    parser.add_argument("--selected-ks", default="1,2,3,4")
    parser.add_argument("--source-recovery-policy", default=DEFAULT_SOURCE_RECOVERY_POLICY)
    parser.add_argument("--recovery-seed", type=int, default=0)
    parser.add_argument("--polish-seed-offset", type=int, default=2000)
    parser.add_argument("--min-target-change-count", type=int, default=5)
    parser.add_argument("--min-boundary-change-count", type=int, default=5)
    parser.add_argument("--min-material-q-gain", type=float, default=0.01)
    return parser

def main() -> None:
    args = build_parser().parse_args()
    result = run_probe(
        attachment_score_rows_path=args.attachment_score_rows,
        attachment_summary_rows_path=args.attachment_summary_rows,
        frontier_rows_path=args.frontier_rows,
        evaluation_core_mode=args.evaluation_core_mode,
        control_dir=args.control_dir,
        output_dir=args.output_dir,
        source_cases=_parse_csv_tuple(
            args.source_cases,
            default=("p6_wide", "p8_fullctx", "p10_wide"),
        ),
        selector_policies=_parse_csv_tuple(
            args.selector_policies,
            default=LOCAL_HANDLE_SELECTOR_POLICIES,
        ),
        selected_ks=_parse_int_tuple(args.selected_ks, default=(1, 2, 3, 4, 5, 8)),
        source_recovery_policy=args.source_recovery_policy,
        recovery_seed=args.recovery_seed,
        polish_seed_offset=args.polish_seed_offset,
        min_target_change_count=args.min_target_change_count,
        min_boundary_change_count=args.min_boundary_change_count,
        min_material_q_gain=args.min_material_q_gain,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))

if __name__ == "__main__":
    main()
