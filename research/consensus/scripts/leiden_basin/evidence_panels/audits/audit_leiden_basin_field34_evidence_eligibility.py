#!/usr/bin/env python3
"""Audit field34 evidence eligibility for Leiden basin cartography.

This is a source-level and fixture-level hygiene audit. It decides how field34
can be used as evidence in Track C before any route execution. It uses endpoint
identity, support size, duplicate/no-op status, and support-source provenance.
It does not use basin quality, cost, ranking, or operator-success fields.
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
CROSSFIELD_ROOT = BASE_RESULT_DIR / "leiden_multibasin_crossfield_budget12_support_20260519"
DEFAULT_CALIBRATION_DIR = BASE_RESULT_DIR / "leiden_basin_definition_calibration_20260528"
DEFAULT_BLOCKER_TRIAGE_DIR = BASE_RESULT_DIR / "leiden_basin_route_label_blocker_triage_20260529"
DEFAULT_COVERAGE_DIR = (
    BASE_RESULT_DIR / "leiden_basin_wall_panel_context_coverage_after_clean_distinct_route_gate_20260528"
)
DEFAULT_OUTPUT_DIR = BASE_RESULT_DIR / "leiden_basin_field34_evidence_eligibility_audit_20260529"

ENDPOINT_IDENTITY_ROWS = "endpoint_identity_rows.csv"
CANDIDATE_PAIR_RELATION_ROWS = "candidate_pair_relation_rows.csv"
FIELD34_QUEUE = "field34_hygiene_queue.csv"
COVERAGE_ROWS = "wall_panel_context_coverage_rows.csv"

ENDPOINT_ROWS_CSV = "field34_endpoint_universe_rows.csv"
METHOD_ROWS_CSV = "field34_method_eligibility_rows.csv"
PAIR_ROWS_CSV = "field34_pair_support_source_rows.csv"
QUEUE_ROWS_CSV = "field34_queue_projection_rows.csv"
SUMMARY_JSON = "field34_evidence_eligibility_summary.json"
REPORT_MD = "field34_evidence_eligibility_report.md"
CONFIG_JSON = "field34_evidence_eligibility_config.json"

TINY_SUPPORT_MAX = 20
SMALL_SUPPORT_MAX = 200
CLAIM_BOUNDARY = (
    "Field34 evidence eligibility audit only; no route execution, wall-promotion "
    "change, basin-quality claim, cost claim, or directed-search claim."
)


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


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = math.nan) -> float:
    try:
        if pd.isna(value):
            return default
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _case_tail_from_case(case: str) -> str:
    marker = "20260514_"
    return case.split(marker, 1)[1] if marker in case else case


def _case_id_from_case(case: str, budget: int) -> str:
    tail = _case_tail_from_case(case)
    return f"{tail}_budget{budget}"


def _field_method_from_case(case: str) -> tuple[str, str]:
    tail = _case_tail_from_case(case)
    field, _, method = tail.partition("_")
    if not method:
        raise ValueError(f"cannot parse field/method from case: {case}")
    return field, method


def _support_size_class(count: int) -> str:
    if count <= 0:
        return "zero_support"
    if count <= TINY_SUPPORT_MAX:
        return "tiny_support"
    if count <= SMALL_SUPPORT_MAX:
        return "small_support"
    return "moderate_support"


def _endpoint_hygiene_status(
    *,
    endpoint_filter_status: str,
    support_count: int,
    duplicate_signature_count: int,
    identity_member_count: int,
) -> str:
    if endpoint_filter_status == "excluded_zero_support" or support_count <= 0:
        return "zero_or_noop_filtered"
    if duplicate_signature_count > 1 or identity_member_count > 1:
        if support_count <= TINY_SUPPORT_MAX:
            return "duplicate_tiny_support_reference_only"
        return "duplicate_endpoint_reference_only"
    if support_count <= TINY_SUPPORT_MAX:
        return "tiny_support_reference_only"
    if support_count <= SMALL_SUPPORT_MAX:
        return "small_support_diagnostic_reference"
    return "moderate_support_candidate"


def _method_role(row: pd.Series) -> str:
    endpoint_rows = _safe_int(row["endpoint_rows"])
    accepted = _safe_int(row["accepted_endpoint_rows"])
    zero = _safe_int(row["zero_support_rows"])
    duplicate = _safe_int(row["duplicate_signature_rows"])
    tiny = _safe_int(row["tiny_support_endpoint_rows"])
    small = _safe_int(row["small_support_endpoint_rows"])
    max_support = _safe_int(row["accepted_support_max"])
    median_support = _safe_float(row["accepted_support_median"])
    if accepted == 0:
        return "field34_filtered_no_accepted_endpoint"
    if endpoint_rows and zero / endpoint_rows >= 0.5:
        return "field34_mostly_filtered_source"
    if duplicate:
        return "field34_duplicate_mixed_reference_only"
    if tiny and accepted and tiny / accepted >= 0.5:
        return "field34_tiny_support_reference_only"
    if math.isfinite(median_support) and median_support <= TINY_SUPPORT_MAX:
        return "field34_tiny_support_reference_only"
    if max_support <= SMALL_SUPPORT_MAX or small == accepted:
        return "field34_small_support_diagnostic_reference"
    return "field34_selective_reference_only"


def _calibration_eligibility(method_role: str) -> str:
    if method_role == "field34_filtered_no_accepted_endpoint":
        return "exclude_from_basin_definition_calibration"
    return "not_clean_calibration_source"


def _load_field34_candidates() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(CROSSFIELD_ROOT.glob("field34_*_seed11_budget12_multifidelity_label_probe_only/candidate_level_rows.csv")):
        frame = _read_csv(path)
        frame = frame.copy()
        frame["candidate_source_artifact"] = _rel(path)
        parsed = frame["case"].map(lambda value: _field_method_from_case(str(value)))
        frame["field"] = parsed.map(lambda item: item[0])
        frame["method"] = parsed.map(lambda item: item[1])
        frame["case_id"] = frame.apply(
            lambda row: _case_id_from_case(str(row["case"]), _safe_int(row["candidate_budget"])),
            axis=1,
        )
        frames.append(frame)
    if not frames:
        raise ValueError("no field34 candidate rows found")
    return pd.concat(frames, ignore_index=True, sort=False)


def _endpoint_rows(calibration_dir: Path) -> pd.DataFrame:
    candidates = _load_field34_candidates()
    identity = _read_csv(calibration_dir / ENDPOINT_IDENTITY_ROWS)
    identity = identity[identity["field"].astype(str).eq("field34")].copy()
    key_cols = ["case_id", "candidate_index"]
    rows = candidates.merge(
        identity[
            [
                "case_id",
                "candidate_index",
                "endpoint_identity_id",
                "endpoint_filter_status",
                "support_node_count",
                "support_node_hash",
                "endpoint_signature",
                "identity_member_count",
                "representative_candidate_index",
            ]
        ],
        on=key_cols,
        how="left",
        suffixes=("", "_identity"),
    )
    rows["support_node_count"] = pd.to_numeric(
        rows["support_node_count"].fillna(rows["p5_basin_changed_support_node_count"]),
        errors="coerce",
    ).fillna(0).astype(int)
    rows["endpoint_filter_status"] = rows["endpoint_filter_status"].fillna(
        rows["support_node_count"].map(lambda count: "excluded_zero_support" if count <= 0 else "accepted")
    )
    rows["endpoint_signature"] = rows["endpoint_signature"].fillna(rows["p5_basin_signature"])
    rows["support_node_hash"] = rows["support_node_hash"].fillna(
        rows["p5_basin_changed_support_node_hash"]
    )
    signature_counts = (
        rows.groupby(["case_id", "endpoint_signature"], dropna=False)["candidate_index"]
        .transform("count")
        .astype(int)
    )
    rows["duplicate_signature_count"] = signature_counts
    rows["identity_member_count"] = pd.to_numeric(
        rows["identity_member_count"],
        errors="coerce",
    ).fillna(signature_counts).astype(int)
    rows["support_size_class"] = rows["support_node_count"].map(_support_size_class)
    rows["endpoint_hygiene_status"] = rows.apply(
        lambda row: _endpoint_hygiene_status(
            endpoint_filter_status=str(row["endpoint_filter_status"]),
            support_count=int(row["support_node_count"]),
            duplicate_signature_count=int(row["duplicate_signature_count"]),
            identity_member_count=int(row["identity_member_count"]),
        ),
        axis=1,
    )
    rows["field34_evidence_role"] = rows["endpoint_hygiene_status"].map(
        lambda status: (
            "filtered"
            if status == "zero_or_noop_filtered"
            else "diagnostic_reference_only"
            if "reference" in status or "small_support" in status
            else "candidate_with_caution"
        )
    )
    out_cols = [
        "case_id",
        "case",
        "field",
        "method",
        "candidate_budget",
        "candidate_index",
        "endpoint_identity_id",
        "endpoint_filter_status",
        "endpoint_hygiene_status",
        "field34_evidence_role",
        "support_size_class",
        "support_node_count",
        "support_node_hash",
        "endpoint_signature",
        "duplicate_signature_count",
        "identity_member_count",
        "representative_candidate_index",
        "p5_basin_sketch_node_hash",
        "p5_basin_sketch_sample_size",
        "candidate_source_artifact",
    ]
    out = rows[out_cols].copy()
    out["claim_boundary"] = CLAIM_BOUNDARY
    return out.sort_values(["method", "candidate_index"]).reset_index(drop=True)


def _method_rows(endpoint_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (case_id, method), group in endpoint_rows.groupby(["case_id", "method"], sort=True):
        accepted = group[~group["endpoint_hygiene_status"].eq("zero_or_noop_filtered")]
        support = pd.to_numeric(accepted["support_node_count"], errors="coerce")
        row = {
            "case_id": case_id,
            "field": "field34",
            "method": method,
            "endpoint_rows": int(len(group)),
            "accepted_endpoint_rows": int(len(accepted)),
            "zero_support_rows": int(group["endpoint_hygiene_status"].eq("zero_or_noop_filtered").sum()),
            "duplicate_signature_rows": int(group["duplicate_signature_count"].gt(1).sum()),
            "tiny_support_endpoint_rows": int(
                accepted["support_size_class"].eq("tiny_support").sum()
            ),
            "small_support_endpoint_rows": int(
                accepted["support_size_class"].isin(["tiny_support", "small_support"]).sum()
            ),
            "accepted_support_min": "" if support.empty else int(support.min()),
            "accepted_support_median": "" if support.empty else float(support.median()),
            "accepted_support_max": "" if support.empty else int(support.max()),
            "endpoint_hygiene_statuses": "|".join(sorted(group["endpoint_hygiene_status"].unique())),
        }
        role = _method_role(pd.Series(row))
        row["method_evidence_role"] = role
        row["basin_definition_calibration_eligibility"] = _calibration_eligibility(role)
        row["route_gate_policy"] = (
            "no_broad_field34_route_gate; project pair-level hygiene only"
        )
        row["claim_boundary"] = CLAIM_BOUNDARY
        rows.append(row)
    return pd.DataFrame(rows)


def _pair_rows(calibration_dir: Path, endpoint_rows: pd.DataFrame) -> pd.DataFrame:
    pairs = _read_csv(calibration_dir / CANDIDATE_PAIR_RELATION_ROWS)
    pairs = pairs[pairs["field"].astype(str).eq("field34")].copy()
    left = endpoint_rows.add_prefix("left_")
    right = endpoint_rows.add_prefix("right_")
    pairs = pairs.merge(
        left[
            [
                "left_case_id",
                "left_candidate_index",
                "left_endpoint_hygiene_status",
                "left_field34_evidence_role",
                "left_support_size_class",
                "left_support_node_count",
            ]
        ],
        left_on=["case_id", "left_candidate_index"],
        right_on=["left_case_id", "left_candidate_index"],
        how="left",
    ).merge(
        right[
            [
                "right_case_id",
                "right_candidate_index",
                "right_endpoint_hygiene_status",
                "right_field34_evidence_role",
                "right_support_size_class",
                "right_support_node_count",
            ]
        ],
        left_on=["case_id", "right_candidate_index"],
        right_on=["right_case_id", "right_candidate_index"],
        how="left",
    )
    pairs["pair_hygiene_status"] = pairs.apply(_pair_hygiene_status, axis=1)
    pairs["claim_boundary"] = CLAIM_BOUNDARY
    cols = [
        "case_id",
        "field",
        "method",
        "left_candidate_index",
        "right_candidate_index",
        "left_endpoint_status",
        "right_endpoint_status",
        "left_endpoint_hygiene_status",
        "right_endpoint_hygiene_status",
        "left_support_node_count",
        "right_support_node_count",
        "support_distance",
        "support_distance_source",
        "support_relation",
        "relation_reason",
        "pair_hygiene_status",
        "claim_boundary",
    ]
    return pairs[cols].sort_values(["method", "left_candidate_index", "right_candidate_index"])


def _pair_hygiene_status(row: pd.Series) -> str:
    statuses = {
        str(row.get("left_endpoint_hygiene_status", "")),
        str(row.get("right_endpoint_hygiene_status", "")),
    }
    support_source = str(row.get("support_distance_source", ""))
    left_count = _safe_int(row.get("left_support_node_count"))
    right_count = _safe_int(row.get("right_support_node_count"))
    min_count = min(left_count, right_count)
    if "zero_or_noop_filtered" in statuses or str(row.get("support_relation")) == "excluded_hygiene":
        return "pair_filtered_zero_or_noop_endpoint"
    if "changed_pair_support" in support_source:
        return "pair_hold_fallback_support_source"
    if any("duplicate" in status for status in statuses):
        return "pair_reference_only_duplicate_endpoint"
    if min_count <= TINY_SUPPORT_MAX:
        return "pair_hold_tiny_support"
    if min_count <= SMALL_SUPPORT_MAX:
        return "pair_reference_only_small_support"
    return "pair_candidate_with_field34_caution"


def _queue_projection(
    *,
    blocker_triage_dir: Path,
    coverage_dir: Path,
    endpoint_rows: pd.DataFrame,
    pair_rows: pd.DataFrame,
    method_rows: pd.DataFrame,
) -> pd.DataFrame:
    queue = _read_csv(blocker_triage_dir / FIELD34_QUEUE)
    coverage = _read_csv(coverage_dir / COVERAGE_ROWS)
    coverage = coverage[coverage["field"].astype(str).eq("field34")].copy()
    rows = queue.merge(
        coverage[
            [
                "panel_pair_id",
                "left_representative_candidate_index",
                "right_representative_candidate_index",
                "left_support_node_count",
                "right_support_node_count",
                "field_hygiene_status",
                "runner_preflight_status",
                "runner_context_status",
                "existing_route_order_sensitivity_status",
                "existing_wall_claim_gate_status",
            ]
        ],
        on="panel_pair_id",
        how="left",
    )
    endpoint_lookup = endpoint_rows.set_index(["case_id", "candidate_index"]).to_dict("index")
    pair_lookup: dict[tuple[str, int, int], pd.Series] = {}
    for _, row in pair_rows.iterrows():
        key = (
            str(row["case_id"]),
            int(row["left_candidate_index"]),
            int(row["right_candidate_index"]),
        )
        pair_lookup[key] = row
        pair_lookup[(key[0], key[2], key[1])] = row
    method_lookup = method_rows.set_index(["case_id", "method"]).to_dict("index")
    projected: list[dict[str, Any]] = []
    for _, row in rows.iterrows():
        case_id = str(row["case_id"])
        method = str(case_id).replace("field34_", "").replace("_budget12", "")
        if "method" in row and pd.notna(row.get("method")):
            method = str(row.get("method"))
        left_idx = _safe_int(row.get("left_representative_candidate_index"))
        right_idx = _safe_int(row.get("right_representative_candidate_index"))
        left = endpoint_lookup.get((case_id, left_idx), {})
        right = endpoint_lookup.get((case_id, right_idx), {})
        pair = pair_lookup.get((case_id, left_idx, right_idx))
        method_meta = method_lookup.get((case_id, method), {})
        pair_status = (
            "pair_identity_or_self_control"
            if left_idx == right_idx
            else str(pair.get("pair_hygiene_status", "")) if pair is not None else "pair_relation_not_found"
        )
        projected_decision = _queue_decision(row, pair_status, left, right)
        projected.append(
            {
                "panel_pair_id": row["panel_pair_id"],
                "field": row["field"],
                "case_id": case_id,
                "method": method,
                "panel_role": row.get("panel_role", ""),
                "calibrated_relation": row.get("calibrated_relation", ""),
                "relation_taxonomy_v0_1": row.get("relation_taxonomy_v0_1", ""),
                "route_label_interpretation_v0": row.get("route_label_interpretation_v0", ""),
                "blocker_priority": row.get("blocker_priority", ""),
                "left_candidate_index": left_idx,
                "right_candidate_index": right_idx,
                "left_endpoint_hygiene_status": left.get("endpoint_hygiene_status", ""),
                "right_endpoint_hygiene_status": right.get("endpoint_hygiene_status", ""),
                "left_support_node_count": left.get("support_node_count", ""),
                "right_support_node_count": right.get("support_node_count", ""),
                "support_distance_source": "" if pair is None else pair.get("support_distance_source", ""),
                "support_distance": "" if pair is None else pair.get("support_distance", ""),
                "pair_hygiene_status": pair_status,
                "method_evidence_role": method_meta.get("method_evidence_role", ""),
                "basin_definition_calibration_eligibility": method_meta.get(
                    "basin_definition_calibration_eligibility",
                    "",
                ),
                "runner_preflight_status": row.get("runner_preflight_status", ""),
                "existing_route_order_sensitivity_status": row.get(
                    "existing_route_order_sensitivity_status",
                    "",
                ),
                "existing_wall_claim_gate_status": row.get("existing_wall_claim_gate_status", ""),
                "field34_fixture_decision": projected_decision,
                "wall_promotion_status_after_hygiene": "no_wall_promotion",
                "route_execution_status_after_hygiene": "not_recommended",
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(projected).sort_values(["blocker_priority", "panel_pair_id"])


def _queue_decision(row: pd.Series, pair_status: str, left: dict[str, Any], right: dict[str, Any]) -> str:
    route_label = str(row.get("route_label_interpretation_v0", ""))
    if pair_status in {"pair_identity_or_self_control", "pair_filtered_zero_or_noop_endpoint"}:
        return "hygiene_filtered_no_op_or_duplicate"
    if pair_status == "pair_hold_fallback_support_source":
        return "hygiene_hold_fallback_support_source"
    if pair_status == "pair_hold_tiny_support":
        return "hygiene_hold_tiny_support_reference_only"
    if route_label and route_label != "nan":
        return "hygiene_pass_reference_only_existing_route_label"
    if pair_status == "pair_reference_only_duplicate_endpoint":
        return "hygiene_pass_reference_only_duplicate_endpoint"
    if pair_status == "pair_reference_only_small_support":
        return "hygiene_pass_reference_only_small_support"
    left_count = _safe_int(left.get("support_node_count"))
    right_count = _safe_int(right.get("support_node_count"))
    if min(left_count, right_count) <= SMALL_SUPPORT_MAX:
        return "hygiene_pass_reference_only_small_support"
    return "hygiene_pass_route_gate_candidate_with_field34_caution"


def _summary(
    *,
    endpoint_rows: pd.DataFrame,
    method_rows: pd.DataFrame,
    pair_rows: pd.DataFrame,
    queue_rows: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    return {
        "status": "field34_evidence_eligibility_audit_prepared",
        "date": "2026-05-29",
        "script": "research/consensus/scripts/audit_leiden_basin_field34_evidence_eligibility.py",
        "output_dir": _rel(output_dir),
        "endpoint_row_count": int(len(endpoint_rows)),
        "method_count": int(len(method_rows)),
        "pair_row_count": int(len(pair_rows)),
        "queue_row_count": int(len(queue_rows)),
        "endpoint_hygiene_status_counts": {
            str(k): int(v)
            for k, v in endpoint_rows["endpoint_hygiene_status"].value_counts().to_dict().items()
        },
        "method_evidence_role_counts": {
            str(k): int(v)
            for k, v in method_rows["method_evidence_role"].value_counts().to_dict().items()
        },
        "pair_hygiene_status_counts": {
            str(k): int(v)
            for k, v in pair_rows["pair_hygiene_status"].value_counts().to_dict().items()
        },
        "queue_fixture_decision_counts": {
            str(k): int(v)
            for k, v in queue_rows["field34_fixture_decision"].value_counts().to_dict().items()
        },
        "route_gate_candidate_count": int(
            queue_rows["field34_fixture_decision"]
            .eq("hygiene_pass_route_gate_candidate_with_field34_caution")
            .sum()
        ),
        "promoted_wall_claim_count": int(
            queue_rows["wall_promotion_status_after_hygiene"].ne("no_wall_promotion").sum()
        ),
        "immediate_route_execution_count": int(
            queue_rows["route_execution_status_after_hygiene"].eq("ready").sum()
        ),
        "decision": (
            "Field34 is not a clean basin-definition calibration source. Use it as "
            "filtered or diagnostic/reference evidence unless a row survives explicit "
            "support-size, duplicate, and support-source hygiene gates."
        ),
        "next_step": (
            "Do not run a field34 route batch from this audit alone. Use the fixture "
            "decisions to keep field34 rows as reference/hold/filtered evidence, then "
            "reassess whether any non-field34 wall-evidence question remains."
        ),
        "paths": {
            "endpoint_rows": _rel(output_dir / ENDPOINT_ROWS_CSV),
            "method_rows": _rel(output_dir / METHOD_ROWS_CSV),
            "pair_rows": _rel(output_dir / PAIR_ROWS_CSV),
            "queue_rows": _rel(output_dir / QUEUE_ROWS_CSV),
            "summary": _rel(output_dir / SUMMARY_JSON),
            "report": _rel(output_dir / REPORT_MD),
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _write_report(path: Path, summary: dict[str, Any], method_rows: pd.DataFrame, queue_rows: pd.DataFrame) -> None:
    lines = [
        "# Field34 Evidence Eligibility Audit",
        "",
        "Date: 2026-05-29",
        "",
        "## Scope",
        "",
        "This artifact audits field34 as an evidence source before route execution.",
        "It separates source-level endpoint hygiene, support-source provenance, and",
        "fixture eligibility. It does not inspect basin quality/cost or promote wall",
        "claims.",
        "",
        "## Decision",
        "",
        str(summary["decision"]),
        "",
        "## Method Eligibility",
        "",
        "| method | endpoints | zero | duplicates | accepted support range | evidence role | calibration |",
        "| --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in method_rows.itertuples(index=False):
        support_range = f"{row.accepted_support_min}-{row.accepted_support_max}"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.method),
                    str(row.endpoint_rows),
                    str(row.zero_support_rows),
                    str(row.duplicate_signature_rows),
                    support_range,
                    str(row.method_evidence_role),
                    str(row.basin_definition_calibration_eligibility),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Queue Projection",
            "",
            "| panel_pair_id | pair hygiene | fixture decision | route execution |",
            "| --- | --- | --- | --- |",
        ]
    )
    for row in queue_rows.itertuples(index=False):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.panel_pair_id),
                    str(row.pair_hygiene_status),
                    str(row.field34_fixture_decision),
                    str(row.route_execution_status_after_hygiene),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Summary Counts",
            "",
            f"- endpoint hygiene: `{summary['endpoint_hygiene_status_counts']}`",
            f"- method roles: `{summary['method_evidence_role_counts']}`",
            f"- pair hygiene: `{summary['pair_hygiene_status_counts']}`",
            f"- queue decisions: `{summary['queue_fixture_decision_counts']}`",
            f"- route-gate candidate count: `{summary['route_gate_candidate_count']}`",
            f"- promoted wall claims: `{summary['promoted_wall_claim_count']}`",
            f"- immediate route execution rows: `{summary['immediate_route_execution_count']}`",
            "",
            "Next step: " + str(summary["next_step"]),
            "",
            "Claim boundary: " + CLAIM_BOUNDARY,
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(
    *,
    calibration_dir: Path,
    blocker_triage_dir: Path,
    coverage_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    endpoint_rows = _endpoint_rows(calibration_dir)
    method_rows = _method_rows(endpoint_rows)
    pair_rows = _pair_rows(calibration_dir, endpoint_rows)
    queue_rows = _queue_projection(
        blocker_triage_dir=blocker_triage_dir,
        coverage_dir=coverage_dir,
        endpoint_rows=endpoint_rows,
        pair_rows=pair_rows,
        method_rows=method_rows,
    )
    summary = _summary(
        endpoint_rows=endpoint_rows,
        method_rows=method_rows,
        pair_rows=pair_rows,
        queue_rows=queue_rows,
        output_dir=output_dir,
    )

    _write_csv(endpoint_rows, output_dir / ENDPOINT_ROWS_CSV)
    _write_csv(method_rows, output_dir / METHOD_ROWS_CSV)
    _write_csv(pair_rows, output_dir / PAIR_ROWS_CSV)
    _write_csv(queue_rows, output_dir / QUEUE_ROWS_CSV)
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(
            {
                "calibration_dir": _rel(calibration_dir),
                "blocker_triage_dir": _rel(blocker_triage_dir),
                "coverage_dir": _rel(coverage_dir),
                "output_dir": _rel(output_dir),
                "tiny_support_max": TINY_SUPPORT_MAX,
                "small_support_max": SMALL_SUPPORT_MAX,
                "claim_boundary": CLAIM_BOUNDARY,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_report(output_dir / REPORT_MD, summary, method_rows, queue_rows)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir", type=Path, default=DEFAULT_CALIBRATION_DIR)
    parser.add_argument("--blocker-triage-dir", type=Path, default=DEFAULT_BLOCKER_TRIAGE_DIR)
    parser.add_argument("--coverage-dir", type=Path, default=DEFAULT_COVERAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = run(
        calibration_dir=args.calibration_dir,
        blocker_triage_dir=args.blocker_triage_dir,
        coverage_dir=args.coverage_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
