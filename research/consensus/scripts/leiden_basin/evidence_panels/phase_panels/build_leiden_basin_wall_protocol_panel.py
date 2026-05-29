#!/usr/bin/env python3
"""Build a representative basin-pair panel for uniform wall-evidence testing.

This script steps back from the c0-c2 direct audit and restores a wider Track C
surface. It selects representative calibrated basin-pair relations across
fields, sources, and relation zones, then attaches a uniform evidence protocol
for future wall tests. It does not run a new route, compare basin quality, or
make a wall claim.
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
SCRIPT_ROOT = REPO_ROOT / "research/consensus/scripts"
BASE_RESULT_DIR = REPO_ROOT / "research/consensus/results/adaptive_refinement"
DEFAULT_CALIBRATION_DIR = BASE_RESULT_DIR / "leiden_basin_definition_calibration_20260528"
DEFAULT_ROUTE_JOIN_DIR = BASE_RESULT_DIR / "leiden_basin_route_wall_evidence_join_20260528"
DEFAULT_DIRECT_AUDIT_DIR = BASE_RESULT_DIR / "direct_pair_route_audit_field34_cc_c0_c2_20260528"
DEFAULT_OUTPUT_DIR = BASE_RESULT_DIR / "leiden_basin_wall_protocol_panel_20260528"

IDENTITY_PAIR_RELATIONS = "identity_pair_relation_rows.csv"
ENDPOINT_IDENTITY_ROWS = "endpoint_identity_rows.csv"
CALIBRATED_CASE_SUMMARY = "calibrated_basin_case_summary.csv"
ROUTE_JOIN_CANDIDATE_PAIRS = "route_join_candidate_pair_rows.csv"
ROUTE_WALL_PAIR_SUMMARY = "route_wall_pair_summary.csv"
DIRECT_PAIR_ROUTE_SUMMARY = "direct_pair_route_summary.csv"

PANEL_CSV = "basin_pair_wall_protocol_panel.csv"
PROTOCOL_STEPS_CSV = "uniform_wall_protocol_steps.csv"
PAIR_REQUIREMENTS_CSV = "wall_protocol_pair_requirements.csv"
SUMMARY_JSON = "wall_protocol_panel_summary.json"
REPORT_MD = "basin_wall_protocol_panel_report.md"
CONFIG_JSON = "wall_protocol_panel_config.json"

SAME_SUPPORT_MAX = 0.5
DISTINCT_SUPPORT_MIN = 0.75


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _safe_float(value: Any, default: float = math.nan) -> float:
    try:
        if pd.isna(value):
            return default
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _fmt_float(value: float) -> str:
    return "" if not math.isfinite(value) else f"{value:.10g}"


def _identity_key(case_id: str, left_id: str, right_id: str) -> tuple[str, str, str]:
    left, right = sorted((str(left_id), str(right_id)))
    return str(case_id), left, right


def _pair_id(case_id: str, left_idx: int, right_idx: int) -> str:
    lo, hi = sorted((left_idx, right_idx))
    return f"{case_id}:c{lo}-c{hi}"


def _load_identity_pairs(calibration_dir: Path, route_join_dir: Path, direct_audit_dir: Path) -> pd.DataFrame:
    pairs = _read_csv(calibration_dir / IDENTITY_PAIR_RELATIONS)
    endpoints = _read_csv(calibration_dir / ENDPOINT_IDENTITY_ROWS)
    case_summary = _read_csv(calibration_dir / CALIBRATED_CASE_SUMMARY)
    route_candidates = _read_csv(calibration_dir / ROUTE_JOIN_CANDIDATE_PAIRS)
    route_summary = _read_csv(route_join_dir / ROUTE_WALL_PAIR_SUMMARY)
    direct_summary = _read_csv(direct_audit_dir / DIRECT_PAIR_ROUTE_SUMMARY)
    if pairs.empty:
        return pd.DataFrame()

    endpoint_lookup: dict[str, dict[str, Any]] = {}
    for _, row in endpoints.iterrows():
        endpoint_lookup[str(row["endpoint_identity_id"])] = row.to_dict()

    route_candidate_keys = {
        _identity_key(row["case_id"], row["left_endpoint_identity_id"], row["right_endpoint_identity_id"])
        for _, row in route_candidates.iterrows()
    } if not route_candidates.empty else set()

    route_summary_lookup = {
        str(row["pair_id"]): row.to_dict()
        for _, row in route_summary.iterrows()
    } if not route_summary.empty else {}

    direct_summary_lookup = {
        str(row["pair_id"]): row.to_dict()
        for _, row in direct_summary.iterrows()
    } if not direct_summary.empty else {}

    route_source_lookup: dict[str, str] = {}
    if not case_summary.empty:
        for _, row in case_summary.iterrows():
            route_source_lookup[str(row["case_id"])] = str(row.get("has_route_trace_source", ""))

    rows: list[dict[str, Any]] = []
    for _, row in pairs.iterrows():
        case_id = str(row["case_id"])
        left_id = str(row["left_endpoint_identity_id"])
        right_id = str(row["right_endpoint_identity_id"])
        left_endpoint = endpoint_lookup.get(left_id, {})
        right_endpoint = endpoint_lookup.get(right_id, {})
        left_idx = _safe_int(left_endpoint.get("representative_candidate_index"))
        right_idx = _safe_int(right_endpoint.get("representative_candidate_index"))
        panel_pair_id = _pair_id(case_id, left_idx, right_idx)
        route_pair = route_summary_lookup.get(panel_pair_id, {})
        direct_pair = direct_summary_lookup.get(panel_pair_id, {})
        route_key = _identity_key(case_id, left_id, right_id)
        rows.append(
            {
                "panel_pair_id": panel_pair_id,
                "source_label": str(row["source_label"]),
                "case_id": case_id,
                "field": str(row["field"]),
                "method": str(row["method"]),
                "candidate_budget": _safe_int(row["candidate_budget"]),
                "left_endpoint_identity_id": left_id,
                "right_endpoint_identity_id": right_id,
                "left_representative_candidate_index": left_idx,
                "right_representative_candidate_index": right_idx,
                "calibrated_relation": str(row["calibrated_relation"]),
                "candidate_pair_count": _safe_int(row.get("candidate_pair_count"), 0),
                "endpoint_distance_min": _fmt_float(_safe_float(row.get("endpoint_distance_min"))),
                "endpoint_distance_max": _fmt_float(_safe_float(row.get("endpoint_distance_max"))),
                "support_distance_min": _fmt_float(_safe_float(row.get("support_distance_min"))),
                "support_distance_max": _fmt_float(_safe_float(row.get("support_distance_max"))),
                "support_distance_max_num": _safe_float(row.get("support_distance_max")),
                "support_distance_min_num": _safe_float(row.get("support_distance_min")),
                "has_existing_route_source": route_source_lookup.get(case_id, "no"),
                "existing_route_join_candidate": "yes" if route_key in route_candidate_keys else "no",
                "existing_route_join_status": str(route_pair.get("wall_evidence_status", "")),
                "existing_wall_claim_status": str(route_pair.get("wall_claim_status", "")),
                "direct_route_audit_status": str(direct_pair.get("verdict", "")),
                "direct_cross_route_rows": _safe_int(direct_pair.get("direct_cross_route_rows"), 0),
                "self_endpoint_route_rows": _safe_int(direct_pair.get("self_endpoint_route_rows"), 0),
            }
        )
    return pd.DataFrame(rows)


def _add_rows(
    selected: list[pd.Series],
    seen: set[str],
    candidates: pd.DataFrame,
    *,
    role: str,
    priority: int,
    rule: str,
) -> None:
    for _, row in candidates.iterrows():
        pair_id = str(row["panel_pair_id"])
        if pair_id in seen:
            continue
        out = row.copy()
        out["panel_role"] = role
        out["protocol_priority"] = priority
        out["selection_rule"] = rule
        seen.add(pair_id)
        selected.append(out)


def _group_take(frame: pd.DataFrame, n: int) -> pd.DataFrame:
    if frame.empty:
        return frame
    return frame.groupby(["source_label", "field"], group_keys=False, sort=True).head(n)


def _select_panel(pairs: pd.DataFrame) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame()
    selected: list[pd.Series] = []
    seen: set[str] = set()

    route_candidates = pairs[
        pairs["existing_route_join_candidate"].eq("yes")
        & pairs["calibrated_relation"].eq("distinct_support_local")
    ].sort_values(["support_distance_max_num", "panel_pair_id"], ascending=[False, True])
    _add_rows(
        selected,
        seen,
        route_candidates,
        role="existing_route_diagnostic_control",
        priority=1,
        rule="all calibrated route-join candidates are retained as controls",
    )

    distinct = pairs[pairs["calibrated_relation"].eq("distinct_support_local")].sort_values(
        ["source_label", "field", "support_distance_max_num", "panel_pair_id"],
        ascending=[True, True, False, True],
    )
    _add_rows(
        selected,
        seen,
        _group_take(distinct, 2),
        role="distinct_high_support_representative",
        priority=2,
        rule="top support-distance distinct pairs per source and field",
    )

    ambiguous = pairs[pairs["calibrated_relation"].eq("ambiguous_support_local")].copy()
    ambiguous["distance_to_distinct_boundary"] = (DISTINCT_SUPPORT_MIN - ambiguous["support_distance_max_num"]).abs()
    near_distinct = ambiguous.sort_values(
        ["source_label", "field", "distance_to_distinct_boundary", "panel_pair_id"],
        ascending=[True, True, True, True],
    )
    _add_rows(
        selected,
        seen,
        _group_take(near_distinct, 1),
        role="ambiguous_near_distinct_boundary",
        priority=3,
        rule="ambiguous pairs closest to the provisional distinct threshold",
    )

    ambiguous["distance_to_same_boundary"] = (ambiguous["support_distance_min_num"] - SAME_SUPPORT_MAX).abs()
    near_same = ambiguous.sort_values(
        ["source_label", "field", "distance_to_same_boundary", "panel_pair_id"],
        ascending=[True, True, True, True],
    )
    _add_rows(
        selected,
        seen,
        _group_take(near_same, 1),
        role="ambiguous_near_same_boundary",
        priority=4,
        rule="ambiguous pairs closest to the same-support threshold",
    )

    same = pairs[pairs["calibrated_relation"].isin({"same_endpoint_identity", "same_support_local"})].sort_values(
        ["source_label", "field", "support_distance_max_num", "panel_pair_id"],
        ascending=[True, True, True, True],
    )
    _add_rows(
        selected,
        seen,
        _group_take(same, 1),
        role="same_or_identity_control",
        priority=5,
        rule="same-zone controls for calibration sanity checks",
    )

    if not selected:
        return pd.DataFrame()
    panel = pd.DataFrame([row.to_dict() for row in selected])
    panel["claim_boundary"] = (
        "Panel membership is for uniform wall-evidence testing only; no wall, "
        "basin evaluation, or directed-search claim is made."
    )
    public_cols = [
        "panel_pair_id",
        "panel_role",
        "protocol_priority",
        "selection_rule",
        "source_label",
        "case_id",
        "field",
        "method",
        "candidate_budget",
        "left_endpoint_identity_id",
        "right_endpoint_identity_id",
        "left_representative_candidate_index",
        "right_representative_candidate_index",
        "calibrated_relation",
        "candidate_pair_count",
        "endpoint_distance_min",
        "endpoint_distance_max",
        "support_distance_min",
        "support_distance_max",
        "has_existing_route_source",
        "existing_route_join_candidate",
        "existing_route_join_status",
        "existing_wall_claim_status",
        "direct_route_audit_status",
        "direct_cross_route_rows",
        "self_endpoint_route_rows",
        "claim_boundary",
    ]
    return panel[public_cols].sort_values(["protocol_priority", "source_label", "field", "panel_pair_id"])


def _protocol_steps() -> pd.DataFrame:
    rows = [
        {
            "protocol_step_id": "W0",
            "protocol_step": "endpoint_identity_confirmation",
            "purpose": "Confirm that both endpoints are accepted calibrated endpoint identities.",
            "required_input": "endpoint identity rows and calibrated pair relation",
            "required_output": "left and right endpoint identities with representative candidate indices",
            "acceptance_status_rule": "available for every retained panel row",
            "unknown_status_rule": "mark pair unavailable if either endpoint identity is missing",
            "claim_boundary": "This step defines pair eligibility only.",
        },
        {
            "protocol_step_id": "W1",
            "protocol_step": "direct_pair_route_trace",
            "purpose": "Create or locate a route trace directly between the paired endpoint identities.",
            "required_input": "left endpoint, right endpoint, fixed case metadata",
            "required_output": "stepwise route trace with source and target endpoint references",
            "acceptance_status_rule": "direct trace exists and names both endpoint identities",
            "unknown_status_rule": "existing self-endpoint or one-sided traces are partial, not accepted",
            "claim_boundary": "This step cannot rank or prefer an endpoint.",
        },
        {
            "protocol_step_id": "W2",
            "protocol_step": "objective_wall_trace",
            "purpose": "Measure objective drop, debt, and recovery along the direct route.",
            "required_input": "direct pair route trace",
            "required_output": "wall height, debt duration, debt area, and recovery status",
            "acceptance_status_rule": "objective metrics are attached to the same direct route",
            "unknown_status_rule": "candidate-side metrics do not establish pair-level wall evidence",
            "claim_boundary": "Objective wall evidence is independent of basin evaluation.",
        },
        {
            "protocol_step_id": "W3",
            "protocol_step": "support_movement_trace",
            "purpose": "Measure support movement toward source and target endpoint identities.",
            "required_input": "direct pair route trace and endpoint support signatures",
            "required_output": "support distance to source and target at each route step",
            "acceptance_status_rule": "support movement is measured on the same direct route",
            "unknown_status_rule": "one-sided candidate movement remains partial evidence",
            "claim_boundary": "Support movement cannot redefine endpoint identity.",
        },
        {
            "protocol_step_id": "W4",
            "protocol_step": "polish_reversion_check",
            "purpose": "Test whether polishing returns the path to source, target, or another endpoint identity.",
            "required_input": "direct route states before and after polish",
            "required_output": "post-polish endpoint identity assignment",
            "acceptance_status_rule": "post-polish endpoint is assigned by the same basin definition",
            "unknown_status_rule": "raw label movement without aligned identity is not accepted",
            "claim_boundary": "Reversion labels are wall-route labels, not evaluation labels.",
        },
        {
            "protocol_step_id": "W5",
            "protocol_step": "route_label_assignment",
            "purpose": "Assign crosses, bounces, collapses, absent, or unknown using W1-W4.",
            "required_input": "endpoint identity, direct route, objective wall trace, support trace, polish result",
            "required_output": "one route label with supported, partial, or unknown confidence",
            "acceptance_status_rule": "label cites endpoint assignment and wall evidence",
            "unknown_status_rule": "missing direct route keeps the label unknown or absent",
            "claim_boundary": "Route labels precede any basin evaluation.",
        },
    ]
    return pd.DataFrame(rows)


def _available_status(row: pd.Series, step_id: str) -> tuple[str, str, str]:
    role = str(row["panel_role"])
    route_join = str(row["existing_route_join_candidate"])
    direct_rows = _safe_int(row.get("direct_cross_route_rows"), 0)
    self_rows = _safe_int(row.get("self_endpoint_route_rows"), 0)
    if step_id == "W0":
        return "available_from_calibration", "endpoint_identity_rows.csv", "no new artifact"
    if role == "same_or_identity_control" and step_id in {"W1", "W2", "W3", "W4", "W5"}:
        return "control_optional", "same-zone control; not a wall-candidate pair", "optional same-control trace"
    if step_id == "W1":
        if direct_rows > 0:
            return "direct_trace_available", "direct audit reports direct cross-route rows", "no new artifact"
        if route_join == "yes" and self_rows > 0:
            return "partial_self_routes_only", "existing route artifacts are self-endpoint traces", "uniform direct pair-route trace"
        if route_join == "yes":
            return "partial_route_context_only", "route-join candidate exists without direct trace", "uniform direct pair-route trace"
        return "missing_uniform_artifact", "no existing direct pair-route source", "uniform direct pair-route trace"
    if step_id in {"W2", "W3"}:
        if direct_rows > 0:
            return "requires_direct_trace_metric_check", "direct route exists but metrics must be verified", "same-route metric table"
        if route_join == "yes":
            return "partial_candidate_side_metrics_only", "existing metrics are not tied to a direct pair route", "same-route metric table"
        return "missing_uniform_artifact", "no same-route wall/support metric source", "same-route metric table"
    if step_id == "W4":
        if direct_rows > 0:
            return "requires_polish_identity_check", "direct route exists but post-polish identity must be assigned", "post-polish identity table"
        return "blocked_until_direct_trace", "no accepted direct pair route trace", "post-polish identity table"
    if step_id == "W5":
        if direct_rows > 0:
            return "blocked_until_w2_w4_review", "route label requires wall, support, and polish evidence", "route label table"
        return "blocked_until_direct_trace", "no accepted direct pair route trace", "route label table"
    return "unknown", "", ""


def _pair_requirements(panel: pd.DataFrame, steps: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, pair in panel.iterrows():
        for _, step in steps.iterrows():
            status, available, missing = _available_status(pair, str(step["protocol_step_id"]))
            rows.append(
                {
                    "panel_pair_id": str(pair["panel_pair_id"]),
                    "case_id": str(pair["case_id"]),
                    "panel_role": str(pair["panel_role"]),
                    "calibrated_relation": str(pair["calibrated_relation"]),
                    "protocol_step_id": str(step["protocol_step_id"]),
                    "protocol_step": str(step["protocol_step"]),
                    "requirement_status": status,
                    "available_evidence_status": available,
                    "missing_artifact": missing,
                    "claim_boundary": "Requirement status is protocol readiness only; no wall claim is made.",
                }
            )
    return pd.DataFrame(rows)


def _summary(calibration_dir: Path, panel: pd.DataFrame, requirements: pd.DataFrame) -> dict[str, Any]:
    all_pairs = _read_csv(calibration_dir / IDENTITY_PAIR_RELATIONS)
    relation_counts = all_pairs["calibrated_relation"].value_counts().to_dict() if not all_pairs.empty else {}
    role_counts = panel["panel_role"].value_counts().to_dict() if not panel.empty else {}
    status_counts = requirements["requirement_status"].value_counts().to_dict() if not requirements.empty else {}
    return {
        "status": "wall_protocol_panel",
        "date": "2026-05-28",
        "calibration_dir": _rel(calibration_dir),
        "identity_pair_rows": int(len(all_pairs)),
        "distinct_support_local_rows": int(relation_counts.get("distinct_support_local", 0)),
        "ambiguous_support_local_rows": int(relation_counts.get("ambiguous_support_local", 0)),
        "same_or_identity_rows": int(
            relation_counts.get("same_support_local", 0)
            + relation_counts.get("same_endpoint_identity", 0)
        ),
        "panel_pair_count": int(len(panel)),
        "panel_role_counts": role_counts,
        "field_count": int(panel["field"].nunique()) if not panel.empty else 0,
        "source_label_count": int(panel["source_label"].nunique()) if not panel.empty else 0,
        "existing_route_control_count": int(panel["existing_route_join_candidate"].eq("yes").sum()) if not panel.empty else 0,
        "requirement_status_counts": status_counts,
        "decision": "use this panel to design uniform wall-evidence artifacts before any further c0-c2 replay",
        "claim_boundary": (
            "The panel defines a testing surface and protocol requirements only; "
            "no wall, basin evaluation, or directed-search claim is made."
        ),
    }


def _write_report(path: Path, summary: dict[str, Any], panel: pd.DataFrame, requirements: pd.DataFrame) -> None:
    lines = [
        "# Leiden Basin Wall Protocol Panel",
        "",
        "Status: representative panel for uniform wall-evidence testing",
        "Date: 2026-05-28",
        "",
        "This report deliberately steps back from the c0-c2 direct audit. It selects a broader calibrated basin-pair panel and attaches the same wall-evidence protocol to every retained pair. It does not run a route, compare basin evaluation fields, or make a wall claim.",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "| --- | --- |",
    ]
    for key in (
        "identity_pair_rows",
        "distinct_support_local_rows",
        "ambiguous_support_local_rows",
        "same_or_identity_rows",
        "panel_pair_count",
        "field_count",
        "source_label_count",
        "existing_route_control_count",
    ):
        lines.append(f"| {key} | {summary.get(key, '')} |")

    lines.extend(["", "## Panel Roles", "", "| role | pairs |", "| --- | ---: |"])
    for role, count in sorted(summary.get("panel_role_counts", {}).items()):
        lines.append(f"| {role} | {count} |")

    lines.extend(
        [
            "",
            "## Requirement Status",
            "",
            "| requirement_status | rows |",
            "| --- | ---: |",
        ]
    )
    for status, count in sorted(summary.get("requirement_status_counts", {}).items()):
        lines.append(f"| {status} | {count} |")

    lines.extend(
        [
            "",
            "## Panel Preview",
            "",
            "| pair_id | role | relation | support_max | route_source | route_join | direct_audit |",
            "| --- | --- | --- | ---: | --- | --- | --- |",
        ]
    )
    for _, row in panel.head(40).iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["panel_pair_id"]),
                    str(row["panel_role"]),
                    str(row["calibrated_relation"]),
                    str(row["support_distance_max"]),
                    str(row["has_existing_route_source"]),
                    str(row["existing_route_join_candidate"]),
                    str(row["direct_route_audit_status"]),
                ]
            )
            + " |"
        )

    missing_direct = requirements[
        requirements["protocol_step_id"].eq("W1")
        & requirements["requirement_status"].isin(
            {"missing_uniform_artifact", "partial_self_routes_only", "partial_route_context_only"}
        )
    ]
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- c0-c2 is retained only as an existing-route diagnostic control.",
            "- The next research unit is the representative panel, not a single c0-c2 replay.",
            f"- {len(missing_direct)} panel wall-candidate rows still need a uniform direct pair-route trace before wall claims can be compared.",
            "- Basin evaluation and directed-search claims remain blocked until the uniform protocol produces supported wall/route labels.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    calibration_dir: Path,
    route_join_dir: Path,
    direct_audit_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs = _load_identity_pairs(calibration_dir, route_join_dir, direct_audit_dir)
    panel = _select_panel(pairs)
    steps = _protocol_steps()
    requirements = _pair_requirements(panel, steps)
    summary = _summary(calibration_dir, panel, requirements)

    _write_csv(panel, output_dir / PANEL_CSV)
    _write_csv(steps, output_dir / PROTOCOL_STEPS_CSV)
    _write_csv(requirements, output_dir / PAIR_REQUIREMENTS_CSV)
    (output_dir / SUMMARY_JSON).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(
            {
                "script": _rel(Path(__file__)),
                "calibration_dir": _rel(calibration_dir),
                "route_join_dir": _rel(route_join_dir),
                "direct_audit_dir": _rel(direct_audit_dir),
                "same_support_max": SAME_SUPPORT_MAX,
                "distinct_support_min": DISTINCT_SUPPORT_MIN,
                "scope": "panel selection and protocol readiness only",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_report(output_dir / REPORT_MD, summary, panel, requirements)
    return {"output_dir": _rel(output_dir), **summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir", type=Path, default=DEFAULT_CALIBRATION_DIR)
    parser.add_argument("--route-join-dir", type=Path, default=DEFAULT_ROUTE_JOIN_DIR)
    parser.add_argument("--direct-audit-dir", type=Path, default=DEFAULT_DIRECT_AUDIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result = run(args.calibration_dir, args.route_join_dir, args.direct_audit_dir, args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
