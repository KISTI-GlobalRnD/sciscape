#!/usr/bin/env python3
"""Audit methodology-v0 partial-wall references against route trace evidence.

This is M4a in the Leiden basin methodology-v0 sequence. It audits the two
existing partial-wall protocol references selected by M3 against existing
uniform route-runner artifacts. It does not execute routes, load memberships,
promote walls, inspect quality/cost, or claim a directed-search method.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").exists()
)
BASE_RESULT_DIR = REPO_ROOT / "research/consensus/results/adaptive_refinement"
DEFAULT_METHOD_DIR = BASE_RESULT_DIR / "leiden_basin_methodology_v0_20260529"
DEFAULT_ROUTE_RUNNER_DIR = (
    BASE_RESULT_DIR / "leiden_basin_uniform_wall_probe_runner_clean_distinct_after_gap_fill_20260528"
)

M3_ROWS_CSV = "methodology_v0_wall_pathway_schema_review_rows.csv"
ROUTE_LABEL_CSV = "uniform_route_label_rows.csv"
ROUTE_SCHEDULE_CLAIM_CSV = "uniform_route_schedule_claim_rows.csv"
DIRECT_ROUTE_CSV = "uniform_direct_pair_route_rows.csv"
OBJECTIVE_WALL_CSV = "uniform_objective_wall_rows.csv"
SUPPORT_MOVEMENT_CSV = "uniform_support_movement_rows.csv"
POLISH_REVERSION_CSV = "uniform_polish_reversion_rows.csv"

SCHEDULE_ROWS_CSV = "methodology_v0_partial_wall_trace_audit_schedule_rows.csv"
PAIR_ROWS_CSV = "methodology_v0_partial_wall_trace_audit_pair_rows.csv"
SUMMARY_JSON = "methodology_v0_partial_wall_trace_audit_summary.json"
REPORT_MD = "methodology_v0_partial_wall_trace_audit_report.md"
CONFIG_JSON = "methodology_v0_partial_wall_trace_audit_config.json"

CLAIM_BOUNDARY = (
    "Methodology-v0 M4a partial-wall trace audit only; no route execution, "
    "wall-promotion change, basin-quality claim, cost claim, or directed-search "
    "claim."
)
QUALITY_COST_STATUS = "excluded_by_methodology_v0"
ROUTE_EXECUTION_STATUS = "not_executed_m4a_trace_audit_only"
WALL_PROMOTION_STATUS = "not_promoted_m4a_trace_audit_only"

TARGET_SUPPORT_THRESHOLD = 0.5
DIRECT_TARGET_SUPPORT_THRESHOLD = 0.05
TARGET_ENDPOINT_DISTANCE_THRESHOLD = 1e-3


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"empty CSV: {path}") from exc


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _count(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in frame:
        return {}
    return {str(k): int(v) for k, v in frame[column].value_counts(dropna=False).to_dict().items()}


def _safe_float(value: Any) -> float:
    try:
        if pd.isna(value):
            return math.nan
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def _safe_int(value: Any) -> int | None:
    try:
        if pd.isna(value):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _safe_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def _bool_text(value: Any) -> bool:
    text = _safe_text(value).lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no", ""}:
        return False
    return bool(value)


def _unique_join(values: pd.Series) -> str:
    texts = sorted({_safe_text(value) for value in values if _safe_text(value)})
    return "|".join(texts)


def _selected_pairs(m3_rows: pd.DataFrame) -> pd.DataFrame:
    selected = m3_rows[
        m3_rows["m3_wall_pathway_schema_status"].eq(
            "existing_partial_wall_protocol_reference_needs_trace_audit"
        )
    ].copy()
    if selected.empty:
        raise ValueError("no M3 partial-wall protocol references found")
    return selected.sort_values(["field", "method", "panel_pair_id"]).reset_index(drop=True)


def _route_schedule_claim_lookup(claim_rows: pd.DataFrame) -> pd.DataFrame:
    if claim_rows.empty:
        return pd.DataFrame(columns=["panel_pair_id"])
    return claim_rows.drop_duplicates(subset=["panel_pair_id"], keep="first").copy()


def _schedule_trace_status(*, label: pd.Series, direct: pd.DataFrame, objective: pd.DataFrame,
                           support: pd.DataFrame, polish: pd.DataFrame) -> dict[str, Any]:
    direct_count = int(len(direct))
    objective_count = int(len(objective))
    support_count = int(len(support))
    polish_count = int(len(polish))

    route_completion_status = _unique_join(direct.get("route_completion_status", pd.Series(dtype=str)))
    step_statuses = _unique_join(direct.get("step_status", pd.Series(dtype=str)))

    objective_debt_max = (
        float(objective["objective_debt_from_start"].max()) if objective_count else math.nan
    )
    objective_debt_final = (
        float(objective["objective_debt_from_start"].iloc[-1]) if objective_count else math.nan
    )
    objective_recovery_max = (
        float(objective["objective_recovery_from_min"].max()) if objective_count else math.nan
    )
    wall_step_count = int(objective["wall_step_flag"].sum()) if objective_count else 0

    support_to_target_start = (
        float(support["support_distance_to_target"].iloc[0]) if support_count else math.nan
    )
    support_to_target_min = (
        float(support["support_distance_to_target"].min()) if support_count else math.nan
    )
    support_to_target_final = (
        float(support["support_distance_to_target"].iloc[-1]) if support_count else math.nan
    )
    support_to_source_final = (
        float(support["support_distance_to_source"].iloc[-1]) if support_count else math.nan
    )
    endpoint_to_target_final = (
        float(support["endpoint_distance_to_target"].iloc[-1]) if support_count else math.nan
    )

    polish_row = polish.iloc[0] if polish_count else pd.Series(dtype=object)
    post_polish_assignment = _safe_text(polish_row.get("post_polish_endpoint_assignment"))
    reversion_status = _safe_text(polish_row.get("reversion_status"))
    post_polish_support_to_target = _safe_float(
        polish_row.get("post_polish_support_distance_to_target")
    )
    post_polish_support_to_source = _safe_float(
        polish_row.get("post_polish_support_distance_to_source")
    )
    post_polish_endpoint_to_target = _safe_float(
        polish_row.get("post_polish_endpoint_distance_to_target")
    )
    post_polish_endpoint_to_source = _safe_float(
        polish_row.get("post_polish_endpoint_distance_to_source")
    )

    w1_status = (
        "direct_trace_present_complete_target_scope"
        if direct_count > 0 and "complete_target_scope" in route_completion_status
        else "direct_trace_missing_or_incomplete"
    )
    w2_status = (
        "objective_debt_and_recovery_observed"
        if objective_debt_max > 0 and objective_recovery_max > 0 and wall_step_count > 0
        else "objective_debt_recovery_missing_or_flat"
    )
    w3_status = (
        "pre_polish_direct_trace_reaches_target_support"
        if support_to_target_final <= DIRECT_TARGET_SUPPORT_THRESHOLD
        and endpoint_to_target_final <= TARGET_ENDPOINT_DISTANCE_THRESHOLD
        else "pre_polish_direct_trace_not_target_assigned"
    )
    w4_status = (
        "post_polish_endpoint_stays_at_target"
        if post_polish_assignment == "target_endpoint" and reversion_status == "stays_at_target"
        else "post_polish_assignment_not_target_stable"
    )
    w5_status = (
        "crosses_reference_schedule_row"
        if _safe_text(label.get("route_label")) == "direct_route_reaches_target_and_polish_stays"
        and _safe_text(label.get("support_assignment_status")) == "target_endpoint"
        else "unknown_or_unassigned_schedule_row"
    )
    support_incompatibility_status = (
        "support_incompatibility_not_observed_post_polish_target_like"
        if post_polish_support_to_target <= TARGET_SUPPORT_THRESHOLD
        else "support_incompatibility_or_boundary_loss_after_polish"
    )
    support_exactness_status = (
        "post_polish_support_not_exact_target"
        if post_polish_support_to_target > DIRECT_TARGET_SUPPORT_THRESHOLD
        else "post_polish_support_exact_or_near_exact_target"
    )

    return {
        "route_schedule": _safe_text(label.get("route_schedule")),
        "route_id": _safe_text(label.get("route_id")),
        "route_label": _safe_text(label.get("route_label")),
        "route_label_confidence": _safe_text(label.get("route_label_confidence")),
        "wall_assignment_status": _safe_text(label.get("wall_assignment_status")),
        "support_assignment_status": _safe_text(label.get("support_assignment_status")),
        "direct_trace_row_count_observed": direct_count,
        "objective_trace_row_count_observed": objective_count,
        "support_trace_row_count_observed": support_count,
        "polish_trace_row_count_observed": polish_count,
        "route_completion_status": route_completion_status,
        "step_statuses": step_statuses,
        "objective_debt_max": objective_debt_max,
        "objective_debt_final": objective_debt_final,
        "objective_recovery_max": objective_recovery_max,
        "wall_step_count": wall_step_count,
        "support_to_target_start": support_to_target_start,
        "support_to_target_min": support_to_target_min,
        "support_to_target_final": support_to_target_final,
        "support_to_source_final": support_to_source_final,
        "endpoint_to_target_final": endpoint_to_target_final,
        "post_polish_endpoint_assignment": post_polish_assignment,
        "reversion_status": reversion_status,
        "post_polish_support_to_target": post_polish_support_to_target,
        "post_polish_support_to_source": post_polish_support_to_source,
        "post_polish_endpoint_to_target": post_polish_endpoint_to_target,
        "post_polish_endpoint_to_source": post_polish_endpoint_to_source,
        "w1_direct_trace_status": w1_status,
        "w2_objective_debt_recovery_status": w2_status,
        "w3_support_movement_status": w3_status,
        "w4_polish_assignment_status": w4_status,
        "w5_route_label_status": w5_status,
        "support_incompatibility_status": support_incompatibility_status,
        "support_exactness_status": support_exactness_status,
    }


def _schedule_rows(
    *,
    selected_pairs: pd.DataFrame,
    route_labels: pd.DataFrame,
    direct_routes: pd.DataFrame,
    objective_rows: pd.DataFrame,
    support_rows: pd.DataFrame,
    polish_rows: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    selected_lookup = selected_pairs.set_index("panel_pair_id").to_dict(orient="index")
    for _, label in route_labels[
        route_labels["panel_pair_id"].isin(selected_lookup.keys())
    ].sort_values(["panel_pair_id", "route_schedule"]).iterrows():
        pair_id = _safe_text(label.get("panel_pair_id"))
        route_id = _safe_text(label.get("route_id"))
        base = selected_lookup[pair_id]
        status = _schedule_trace_status(
            label=label,
            direct=direct_routes[direct_routes["route_id"].eq(route_id)].sort_values("step_index"),
            objective=objective_rows[objective_rows["route_id"].eq(route_id)].sort_values(
                "step_index"
            ),
            support=support_rows[support_rows["route_id"].eq(route_id)].sort_values("step_index"),
            polish=polish_rows[polish_rows["route_id"].eq(route_id)],
        )
        rows.append(
            {
                "panel_pair_id": pair_id,
                "case_id": base.get("case_id", ""),
                "field": base.get("field", ""),
                "method": base.get("method", ""),
                "panel_role": base.get("panel_role", ""),
                "left_endpoint_identity_id": base.get("left_endpoint_identity_id", ""),
                "right_endpoint_identity_id": base.get("right_endpoint_identity_id", ""),
                **status,
                "quality_cost_status": QUALITY_COST_STATUS,
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows)


def _all_eq(values: pd.Series, expected: str) -> bool:
    return bool(len(values)) and set(values.astype(str)) == {expected}


def _pair_rows(
    *,
    selected_pairs: pd.DataFrame,
    schedule_rows: pd.DataFrame,
    schedule_claim_rows: pd.DataFrame,
) -> pd.DataFrame:
    claim_lookup = _route_schedule_claim_lookup(schedule_claim_rows).set_index("panel_pair_id")
    rows: list[dict[str, Any]] = []
    for _, selected in selected_pairs.iterrows():
        pair_id = _safe_text(selected.get("panel_pair_id"))
        schedules = schedule_rows[schedule_rows["panel_pair_id"].eq(pair_id)].copy()
        claim = claim_lookup.loc[pair_id] if pair_id in claim_lookup.index else pd.Series(dtype=object)
        schedule_count = int(len(schedules))
        w1_ok = _all_eq(schedules["w1_direct_trace_status"], "direct_trace_present_complete_target_scope")
        w2_ok = _all_eq(
            schedules["w2_objective_debt_recovery_status"],
            "objective_debt_and_recovery_observed",
        )
        w3_ok = _all_eq(
            schedules["w3_support_movement_status"],
            "pre_polish_direct_trace_reaches_target_support",
        )
        w4_ok = _all_eq(schedules["w4_polish_assignment_status"], "post_polish_endpoint_stays_at_target")
        w5_ok = _all_eq(schedules["w5_route_label_status"], "crosses_reference_schedule_row")
        schedule_stable = all(
            _bool_text(claim.get(col))
            for col in [
                "schedule_replicated",
                "stable_route_label",
                "stable_wall_assignment",
                "stable_support_assignment",
            ]
        )
        support_target_max = (
            float(schedules["post_polish_support_to_target"].max()) if schedule_count else math.nan
        )
        support_target_min = (
            float(schedules["post_polish_support_to_target"].min()) if schedule_count else math.nan
        )
        objective_debt_max = (
            float(schedules["objective_debt_max"].max()) if schedule_count else math.nan
        )
        objective_recovery_max = (
            float(schedules["objective_recovery_max"].max()) if schedule_count else math.nan
        )
        wall_step_min = int(schedules["wall_step_count"].min()) if schedule_count else 0
        wall_step_max = int(schedules["wall_step_count"].max()) if schedule_count else 0

        if all([w1_ok, w2_ok, w3_ok, w4_ok, w5_ok, schedule_stable]):
            route_label_rule = "crosses_reference_schedule_stable_target_polish"
            pathway_status = "audited_crosses_protocol_reference"
        else:
            route_label_rule = "unknown_trace_audit_incomplete_or_unstable"
            pathway_status = "unknown_trace_audit_incomplete_or_unstable"

        if support_target_max <= TARGET_SUPPORT_THRESHOLD:
            support_status = "support_incompatibility_absent_target_like_after_polish"
        else:
            support_status = "support_incompatibility_or_boundary_loss_present_after_polish"

        if support_target_max > DIRECT_TARGET_SUPPORT_THRESHOLD:
            support_exactness = "post_polish_support_target_like_but_not_exact"
        else:
            support_exactness = "post_polish_support_exact_or_near_exact"

        wall_status = "not_promoted_constructed_pathway_reference_only"
        if pathway_status == "audited_crosses_protocol_reference":
            audit_status = "trace_audit_complete_partial_pathway_reference"
            next_action = (
                "freeze_route_label_rule_then_compare_against_full_cache_not_routed_candidates"
            )
        else:
            audit_status = "trace_audit_incomplete"
            next_action = "repair_trace_audit_inputs_before_any_probe"

        rows.append(
            {
                "panel_pair_id": pair_id,
                "case_id": selected.get("case_id", ""),
                "field": selected.get("field", ""),
                "method": selected.get("method", ""),
                "panel_role": selected.get("panel_role", ""),
                "left_endpoint_identity_id": selected.get("left_endpoint_identity_id", ""),
                "right_endpoint_identity_id": selected.get("right_endpoint_identity_id", ""),
                "pair_evidence_grade": selected.get("pair_evidence_grade", ""),
                "calibrated_relation": selected.get("calibrated_relation", ""),
                "support_distance_max": selected.get("support_distance_max", ""),
                "schedule_count": schedule_count,
                "schedule_replicated": _bool_text(claim.get("schedule_replicated")),
                "stable_route_label": _bool_text(claim.get("stable_route_label")),
                "stable_wall_assignment": _bool_text(claim.get("stable_wall_assignment")),
                "stable_support_assignment": _bool_text(claim.get("stable_support_assignment")),
                "w1_pair_status": "pass" if w1_ok else "fail_or_missing",
                "w2_pair_status": "pass" if w2_ok else "fail_or_missing",
                "w3_pair_status": "pass" if w3_ok else "fail_or_missing",
                "w4_pair_status": "pass" if w4_ok else "fail_or_missing",
                "w5_pair_status": "pass" if w5_ok else "fail_or_missing",
                "w6_schedule_invariance_status": "pass" if schedule_stable else "fail_or_missing",
                "route_label_rule_v0_1": route_label_rule,
                "m4a_pathway_audit_status": pathway_status,
                "m4a_wall_claim_status": wall_status,
                "m4a_trace_audit_status": audit_status,
                "support_incompatibility_audit_status": support_status,
                "support_exactness_status": support_exactness,
                "post_polish_support_to_target_min": support_target_min,
                "post_polish_support_to_target_max": support_target_max,
                "objective_debt_max": objective_debt_max,
                "objective_recovery_max": objective_recovery_max,
                "wall_step_count_min": wall_step_min,
                "wall_step_count_max": wall_step_max,
                "allowed_claim": (
                    "schedule-stable constructed pathway reference reaches target endpoint "
                    "after objective debt/recovery under v0 trace audit"
                ),
                "forbidden_claim": (
                    "supported wall claim; basin-quality claim; cost claim; directed-search "
                    "or basin-tunneling operator claim"
                ),
                "m4a_next_action": next_action,
                "quality_cost_status": QUALITY_COST_STATUS,
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows)


def _summary(
    *,
    pair_rows: pd.DataFrame,
    schedule_rows: pd.DataFrame,
    m3_rows: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    full_cache_not_routed = m3_rows[
        m3_rows["m3_next_action"].eq("prepare_predeclared_wall_trace_schema_before_any_probe")
    ]
    trace_complete = int(
        pair_rows["m4a_trace_audit_status"].eq(
            "trace_audit_complete_partial_pathway_reference"
        ).sum()
    )
    wall_promoted = int(pair_rows["m4a_wall_claim_status"].ne(
        "not_promoted_constructed_pathway_reference_only"
    ).sum())
    return {
        "status": "methodology_v0_partial_wall_trace_audit_complete",
        "date": "2026-05-29",
        "script": _rel(Path(__file__).resolve()),
        "output_dir": _rel(output_dir),
        "audited_pair_count": int(len(pair_rows)),
        "audited_schedule_row_count": int(len(schedule_rows)),
        "trace_audit_complete_pair_count": trace_complete,
        "new_route_execution_count": 0,
        "wall_promotion_count": wall_promoted,
        "full_cache_not_routed_future_candidate_count": int(len(full_cache_not_routed)),
        "route_label_rule_v0_1_counts": _count(pair_rows, "route_label_rule_v0_1"),
        "m4a_pathway_audit_status_counts": _count(pair_rows, "m4a_pathway_audit_status"),
        "m4a_wall_claim_status_counts": _count(pair_rows, "m4a_wall_claim_status"),
        "support_incompatibility_audit_status_counts": _count(
            pair_rows, "support_incompatibility_audit_status"
        ),
        "w1_direct_trace_status_counts": _count(schedule_rows, "w1_direct_trace_status"),
        "w2_objective_debt_recovery_status_counts": _count(
            schedule_rows, "w2_objective_debt_recovery_status"
        ),
        "w3_support_movement_status_counts": _count(schedule_rows, "w3_support_movement_status"),
        "w4_polish_assignment_status_counts": _count(schedule_rows, "w4_polish_assignment_status"),
        "w5_route_label_status_counts": _count(schedule_rows, "w5_route_label_status"),
        "quality_cost_excluded": bool(pair_rows["quality_cost_status"].eq(QUALITY_COST_STATUS).all()),
        "route_execution_not_run": bool(pair_rows["route_execution_status"].eq(
            ROUTE_EXECUTION_STATUS
        ).all()),
        "wall_promotion_not_run": bool(pair_rows["wall_promotion_status"].eq(
            WALL_PROMOTION_STATUS
        ).all()),
        "decision": (
            "M4a audits two field26 partial-wall references as schedule-stable "
            "constructed pathway references, not supported wall claims. Objective "
            "debt/recovery and target endpoint assignment are present across schedules, "
            "but support incompatibility is not observed and post-polish support remains "
            "target-like rather than exact."
        ),
        "next_step": (
            "Freeze route-label v0.1 around schedule-stable target-polish versus "
            "schedule-dependent polish loss. Only after that, consider a predeclared "
            "micro-probe for the three full-cache not-routed pairs."
        ),
        "paths": {
            "schedule_rows": _rel(output_dir / SCHEDULE_ROWS_CSV),
            "pair_rows": _rel(output_dir / PAIR_ROWS_CSV),
            "summary": _rel(output_dir / SUMMARY_JSON),
            "report": _rel(output_dir / REPORT_MD),
            "config": _rel(output_dir / CONFIG_JSON),
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _write_report(path: Path, summary: dict[str, Any], pair_rows: pd.DataFrame) -> None:
    lines = [
        "# Methodology v0 Partial-Wall Trace Audit",
        "",
        "Date: 2026-05-29",
        "",
        "## Scope",
        "",
        "This artifact is M4a of the methodology-v0 sequence. It audits existing",
        "partial-wall protocol references against W1-W6 trace fields. It does not",
        "execute routes, promote wall claims, load memberships, or inspect",
        "quality/cost.",
        "",
        "## Decision",
        "",
        str(summary["decision"]),
        "",
        "## Counts",
        "",
        f"- audited pairs: `{summary['audited_pair_count']}`",
        f"- audited schedule rows: `{summary['audited_schedule_row_count']}`",
        f"- trace-audit complete pairs: `{summary['trace_audit_complete_pair_count']}`",
        f"- new route executions: `{summary['new_route_execution_count']}`",
        f"- wall promotions: `{summary['wall_promotion_count']}`",
        f"- full-cache not-routed future candidates: "
        f"`{summary['full_cache_not_routed_future_candidate_count']}`",
        "",
        "## Pair Results",
        "",
        "| pair | route label rule | pathway audit | wall status | post-polish support-to-target max |",
        "| --- | --- | --- | --- | ---: |",
    ]
    for _, row in pair_rows.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['panel_pair_id']}`",
                    f"`{row['route_label_rule_v0_1']}`",
                    f"`{row['m4a_pathway_audit_status']}`",
                    f"`{row['m4a_wall_claim_status']}`",
                    f"`{float(row['post_polish_support_to_target_max']):.3f}`",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Route Label Rule v0.1",
            "",
            "A schedule-stable target-polish row may be called a constructed pathway",
            "reference when all schedules pass W1-W6 and finish at the target",
            "endpoint after objective debt/recovery. It remains a partial protocol",
            "reference, not a supported wall claim.",
            "",
            "A schedule-dependent polish-loss row stays `unknown` even if some",
            "schedules reach the target endpoint.",
            "",
            "## No-Leak Checks",
            "",
            f"- quality/cost excluded: `{str(summary['quality_cost_excluded']).lower()}`",
            f"- route execution not run: `{str(summary['route_execution_not_run']).lower()}`",
            f"- wall promotion not run: `{str(summary['wall_promotion_not_run']).lower()}`",
            "",
            "## Next Step",
            "",
            str(summary["next_step"]),
            "",
            "Claim boundary: " + CLAIM_BOUNDARY,
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(*, methodology_dir: Path, route_runner_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    m3_rows = _read_csv(methodology_dir / M3_ROWS_CSV)
    route_labels = _read_csv(route_runner_dir / ROUTE_LABEL_CSV)
    schedule_claims = _read_csv(route_runner_dir / ROUTE_SCHEDULE_CLAIM_CSV)
    direct_routes = _read_csv(route_runner_dir / DIRECT_ROUTE_CSV)
    objective_rows = _read_csv(route_runner_dir / OBJECTIVE_WALL_CSV)
    support_rows = _read_csv(route_runner_dir / SUPPORT_MOVEMENT_CSV)
    polish_rows = _read_csv(route_runner_dir / POLISH_REVERSION_CSV)

    selected_pairs = _selected_pairs(m3_rows)
    schedule_audit = _schedule_rows(
        selected_pairs=selected_pairs,
        route_labels=route_labels,
        direct_routes=direct_routes,
        objective_rows=objective_rows,
        support_rows=support_rows,
        polish_rows=polish_rows,
    )
    pair_audit = _pair_rows(
        selected_pairs=selected_pairs,
        schedule_rows=schedule_audit,
        schedule_claim_rows=schedule_claims,
    )
    summary = _summary(
        pair_rows=pair_audit,
        schedule_rows=schedule_audit,
        m3_rows=m3_rows,
        output_dir=output_dir,
    )

    _write_csv(schedule_audit, output_dir / SCHEDULE_ROWS_CSV)
    _write_csv(pair_audit, output_dir / PAIR_ROWS_CSV)
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(
            {
                "methodology_dir": _rel(methodology_dir),
                "route_runner_dir": _rel(route_runner_dir),
                "output_dir": _rel(output_dir),
                "target_support_threshold": TARGET_SUPPORT_THRESHOLD,
                "direct_target_support_threshold": DIRECT_TARGET_SUPPORT_THRESHOLD,
                "target_endpoint_distance_threshold": TARGET_ENDPOINT_DISTANCE_THRESHOLD,
                "quality_cost_status": QUALITY_COST_STATUS,
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_report(output_dir / REPORT_MD, summary, pair_audit)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methodology-dir", type=Path, default=DEFAULT_METHOD_DIR)
    parser.add_argument("--route-runner-dir", type=Path, default=DEFAULT_ROUTE_RUNNER_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_METHOD_DIR)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = run(
        methodology_dir=args.methodology_dir,
        route_runner_dir=args.route_runner_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
