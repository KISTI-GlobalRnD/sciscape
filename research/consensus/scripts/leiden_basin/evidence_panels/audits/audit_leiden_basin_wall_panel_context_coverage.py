#!/usr/bin/env python3
"""Audit context coverage for the Leiden basin wall-protocol panel.

This script prepares the next Track C step after the 7-pair route-gate panel.
It does not run routes and does not evaluate basin value. It only answers:

- which of the 23 wall-panel pairs have source endpoint artifacts;
- which pairs have a matching vanilla graph context for the uniform runner;
- which pairs already have route-schedule gate output;
- which distinct pairs are runnable next;
- which ambiguous pairs need relation refinement before wall promotion.
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
COMBINED_DIR = (
    BASE_RESULT_DIR
    / "leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30"
)
DEFAULT_PANEL_DIR = BASE_RESULT_DIR / "leiden_basin_wall_protocol_panel_20260528"
DEFAULT_CALIBRATION_DIR = BASE_RESULT_DIR / "leiden_basin_definition_calibration_20260528"
DEFAULT_GATE_DIR = BASE_RESULT_DIR / "leiden_basin_uniform_wall_probe_runner_expanded_controls_20260528"
DEFAULT_OUTPUT_DIR = BASE_RESULT_DIR / "leiden_basin_wall_panel_context_coverage_20260528"

PANEL_CSV = "basin_pair_wall_protocol_panel.csv"
ENDPOINT_ROWS_CSV = "endpoint_identity_rows.csv"
CLAIM_PANEL_CSV = "uniform_route_schedule_claim_panel_summary.csv"

COVERAGE_ROWS_CSV = "wall_panel_context_coverage_rows.csv"
CASE_REQUIREMENTS_CSV = "wall_panel_context_case_requirements.csv"
AMBIGUOUS_QUEUE_CSV = "ambiguous_relation_refinement_queue.csv"
DISTINCT_QUEUE_CSV = "runnable_distinct_pair_queue.csv"
SUMMARY_JSON = "wall_panel_context_coverage_summary.json"
REPORT_MD = "wall_panel_context_coverage_report.md"
CONFIG_JSON = "wall_panel_context_coverage_config.json"

VANILLA_FIELD12_26 = COMBINED_DIR / "vanilla_reachability_sweep_field12_field26_material_core_exact"
VANILLA_FIELD30 = COMBINED_DIR / "vanilla_reachability_sweep_field30_material_core_exact"
VANILLA_FIELD34_CORE = COMBINED_DIR / "vanilla_reachability_sweep_field34_core_exact"
VANILLA_FIELD34_CC = COMBINED_DIR / "vanilla_reachability_sweep_field34_cc_n10_compatible_sketch"

RUNNER_CANDIDATE_REQUIRED_COLUMNS = ("case", "candidate_index", "source_cluster", "target_cluster")
RUNNER_VANILLA_REQUIRED_COLUMNS = ("case", "graph_dir")
SAME_SUPPORT_MAX = 0.5
DISTINCT_SUPPORT_MIN = 0.75
AMBIGUOUS_NEAR_SAME_MAX = 0.60
AMBIGUOUS_NEAR_DISTINCT_MIN = 0.70


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


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


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


def _path_exists(path_text: str) -> bool:
    if not path_text or path_text == "nan":
        return False
    path = _resolve_path(path_text)
    return path.exists()


def _resolve_path(path_text: Any) -> Path:
    path = Path(str(path_text))
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _case_tail(case_id: str) -> str:
    return str(case_id).removesuffix("_budget12").removesuffix("_budget15")


def _case_rows(frame: pd.DataFrame, case_id: str) -> pd.DataFrame:
    if frame.empty or "case" not in frame:
        return pd.DataFrame()
    tail = _case_tail(case_id)
    return frame[frame["case"].fillna("").astype(str).str.endswith(tail)].copy()


def _vanilla_dir_for_field(field: str, case_id: str) -> Path:
    if field in {"field12", "field26"}:
        return VANILLA_FIELD12_26
    if field == "field30":
        return VANILLA_FIELD30
    if field == "field34" and "all_edges_cc_cosine" in case_id:
        return VANILLA_FIELD34_CC
    if field == "field34":
        return VANILLA_FIELD34_CORE
    return Path("")


def _csv_has_case(path: Path, case_id: str) -> bool:
    if not path.exists():
        return False
    try:
        rows = pd.read_csv(path, usecols=["case"])
    except (pd.errors.EmptyDataError, ValueError):
        return False
    return not _case_rows(rows, case_id).empty


def _vanilla_context_status(
    field: str,
    case_id: str,
    extra_vanilla_dirs: tuple[Path, ...] = (),
) -> tuple[str, str]:
    vanilla_dir = _vanilla_dir_for_field(field, case_id)
    if not vanilla_dir:
        return "", "missing"
    rows_path = vanilla_dir / "vanilla_basin_rows.csv"
    if _csv_has_case(rows_path, case_id):
        return _rel(vanilla_dir), "available"
    for extra_dir in extra_vanilla_dirs:
        extra_rows_path = extra_dir / "vanilla_basin_rows.csv"
        if _csv_has_case(extra_rows_path, case_id):
            return _rel(extra_dir), "available"
    return _rel(vanilla_dir), "missing"


def _load_candidate_rows_for_preflight(
    left_source: str,
    right_source: str,
) -> tuple[pd.DataFrame, list[str]]:
    frames: list[pd.DataFrame] = []
    notes: list[str] = []
    seen_paths: set[Path] = set()
    for label, source in (("left", left_source), ("right", right_source)):
        if not source or source == "nan":
            notes.append(f"{label}_source_missing")
            continue
        path = _resolve_path(source)
        if not path.exists():
            notes.append(f"{label}_source_file_missing")
            continue
        resolved = path.resolve()
        if resolved in seen_paths:
            continue
        seen_paths.add(resolved)
        frame = _read_csv(path)
        if frame.empty:
            notes.append(f"{label}_source_empty")
            continue
        missing = [col for col in RUNNER_CANDIDATE_REQUIRED_COLUMNS if col not in frame.columns]
        if missing:
            notes.append(f"{label}_source_missing_columns:{','.join(missing)}")
            continue
        frame = frame.copy()
        frame["candidate_source_artifact"] = _rel(path)
        frame["candidate_index"] = pd.to_numeric(frame["candidate_index"], errors="coerce")
        frames.append(frame)
    if not frames:
        return pd.DataFrame(), notes
    rows = pd.concat(frames, ignore_index=True, sort=False)
    return rows.drop_duplicates(
        subset=["candidate_source_artifact", "case", "candidate_index"],
        keep="first",
    ), notes


def _candidate_match_status(
    *,
    candidates: pd.DataFrame,
    case_id: str,
    left_index: Any,
    right_index: Any,
) -> tuple[str, str]:
    if candidates.empty:
        return "missing_candidate_rows", "candidate rows unavailable"
    case_rows = _case_rows(candidates, case_id)
    if case_rows.empty:
        return "missing_candidate_case_rows", "candidate source has no strict case-tail match"
    notes: list[str] = []
    for label, index in (("left", left_index), ("right", right_index)):
        try:
            idx = int(float(index))
        except (TypeError, ValueError):
            notes.append(f"{label}_candidate_index_invalid")
            continue
        rows = case_rows[pd.to_numeric(case_rows["candidate_index"], errors="coerce").eq(idx)]
        if rows.empty:
            notes.append(f"{label}_candidate_index_missing:{idx}")
    if notes:
        return "missing_candidate_index", ";".join(notes)
    return "candidate_rows_ready", "candidate rows and endpoint indices found"


def _select_runner_vanilla_row(vanilla_dir: Path, case_id: str) -> tuple[str, str, str]:
    rows_path = vanilla_dir / "vanilla_basin_rows.csv"
    rows = _read_csv(rows_path)
    if rows.empty:
        return "missing_vanilla_rows", "vanilla_basin_rows missing or empty", ""
    missing = [col for col in RUNNER_VANILLA_REQUIRED_COLUMNS if col not in rows.columns]
    if missing:
        return "missing_vanilla_columns", f"missing columns:{','.join(missing)}", ""
    case_rows = _case_rows(rows, case_id)
    if case_rows.empty:
        return "missing_vanilla_case_rows", "vanilla rows have no strict case-tail match", ""
    case_rows = case_rows.copy()
    case_rows["_seed_pref"] = (
        pd.to_numeric(case_rows.get("seed"), errors="coerce").fillna(-1).astype(int).eq(11)
    )
    case_rows["_n10_pref"] = case_rows.get("requested_n_iterations", "").astype(str).eq("10")
    case_rows["_randomness_abs"] = pd.to_numeric(
        case_rows.get("randomness"),
        errors="coerce",
    ).fillna(math.inf).abs()
    case_rows = case_rows.sort_values(
        ["_seed_pref", "_n10_pref", "_randomness_abs", "graph_dir"],
        ascending=[False, False, True, True],
    )
    graph_dir_text = str(case_rows.iloc[0].get("graph_dir", "")).strip()
    if not graph_dir_text or graph_dir_text == "nan":
        return "missing_graph_dir", "selected vanilla row has no graph_dir", ""
    graph_dir = _resolve_path(graph_dir_text)
    if not graph_dir.exists():
        return "missing_graph_dir", f"graph_dir missing:{graph_dir}", _rel(graph_dir)
    return "vanilla_graph_ready", "vanilla row and graph_dir found", _rel(graph_dir)


def _runner_preflight(
    *,
    row: pd.Series,
    left_source: str,
    right_source: str,
    vanilla_dir_text: str,
) -> tuple[str, str, str, str, str]:
    case_id = str(row["case_id"])
    candidates, candidate_notes = _load_candidate_rows_for_preflight(left_source, right_source)
    candidate_status, candidate_note = _candidate_match_status(
        candidates=candidates,
        case_id=case_id,
        left_index=row.get("left_representative_candidate_index"),
        right_index=row.get("right_representative_candidate_index"),
    )
    if candidate_notes:
        candidate_note = candidate_note + ";" + ";".join(candidate_notes)

    vanilla_dir = _resolve_path(vanilla_dir_text) if vanilla_dir_text else Path("")
    if not vanilla_dir_text:
        vanilla_status, vanilla_note, graph_dir = "missing_vanilla_context", "no vanilla dir", ""
    else:
        vanilla_status, vanilla_note, graph_dir = _select_runner_vanilla_row(vanilla_dir, case_id)

    ready = candidate_status == "candidate_rows_ready" and vanilla_status == "vanilla_graph_ready"
    if ready:
        return (
            "runner_preflight_ready",
            "candidate rows, endpoint indices, vanilla row, and graph_dir are present",
            graph_dir,
            candidate_status,
            vanilla_status,
        )
    issue_statuses = [status for status in (candidate_status, vanilla_status) if not status.endswith("_ready")]
    if len(issue_statuses) > 1:
        status = "runner_preflight_failed_multiple_inputs"
    else:
        status = f"runner_preflight_failed_{issue_statuses[0]}"
    return (
        status,
        f"{candidate_note};{vanilla_note}",
        graph_dir,
        candidate_status,
        vanilla_status,
    )


def _endpoint_lookup(endpoint_rows: pd.DataFrame) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for _, row in endpoint_rows.iterrows():
        endpoint_id = str(row["endpoint_identity_id"])
        lookup[endpoint_id] = {
            "source_artifact": str(row.get("source_artifact", "")),
            "support_node_count": row.get("support_node_count", ""),
            "candidate_index": row.get("candidate_index", ""),
            "representative_candidate_index": row.get("representative_candidate_index", ""),
        }
    return lookup


def _gate_lookup(claim_panel: pd.DataFrame) -> dict[str, dict[str, str]]:
    if claim_panel.empty:
        return {}
    lookup: dict[str, dict[str, str]] = {}
    for _, row in claim_panel.iterrows():
        pair_id = str(row["panel_pair_id"])
        lookup[pair_id] = {
            "route_order_sensitivity_status": str(
                row.get("route_order_sensitivity_status", "not_run")
            ),
            "wall_claim_gate_status": str(row.get("wall_claim_gate_status", "not_run")),
            "source_output": str(row.get("source_output", "")),
        }
    return lookup


def _candidate_source_status(left_source: str, right_source: str) -> str:
    left_ok = _path_exists(left_source)
    right_ok = _path_exists(right_source)
    if left_ok and right_ok:
        return "both_available"
    if left_ok or right_ok:
        return "partial"
    return "missing"


def _runner_context_status(vanilla_status: str, source_status: str) -> str:
    if vanilla_status == "available" and source_status == "both_available":
        return "runnable"
    if vanilla_status != "available" and source_status != "both_available":
        return "missing_both"
    if vanilla_status != "available":
        return "missing_vanilla_context"
    return "missing_candidate_source"


def _runner_context_status_from_preflight(preflight_status: str) -> str:
    if preflight_status == "runner_preflight_ready":
        return "runnable"
    if "candidate" in preflight_status:
        return "missing_candidate_context"
    if "vanilla" in preflight_status:
        return "missing_vanilla_context"
    if "graph_dir" in preflight_status:
        return "missing_graph_context"
    return "missing_or_invalid_runner_context"


def _field_hygiene_status(field: str, relation: str, left_support: Any, right_support: Any) -> str:
    flags: list[str] = []
    if relation == "same_endpoint_identity":
        flags.append("same_endpoint_identity_control_do_not_route")
    if field == "field34":
        flags.append("field34_hygiene_review_required")
    left = _safe_float(left_support)
    right = _safe_float(right_support)
    counts = [value for value in (left, right) if math.isfinite(value)]
    if counts and min(counts) <= 10:
        flags.append("tiny_support_endpoint")
    elif counts and min(counts) <= 50:
        flags.append("small_support_endpoint")
    return "|".join(flags) if flags else "standard"


def _next_action(
    relation: str,
    runner_status: str,
    route_status: str,
    gate_status: str,
    field_hygiene_status: str,
) -> str:
    if relation == "ambiguous_support_local":
        return "refine_ambiguous_relation_before_wall_claim"
    if relation.startswith("same"):
        return "keep_as_control"
    if gate_status == "passes_schedule_invariance_distinct_partial_wall_evidence":
        return "retain_as_current_distinct_partial_wall_gate"
    if route_status == "route_order_sensitive":
        return "keep_as_route_order_sensitive_control"
    if relation == "distinct_support_local" and runner_status == "runnable":
        if field_hygiene_status != "standard":
            return "review_hygiene_before_route_gate"
        return "run_w1_w6_route_order_gate"
    if relation == "distinct_support_local":
        return "locate_or_generate_context_before_route_gate"
    return "hold_for_manual_review"


def _coverage_rows(
    panel: pd.DataFrame,
    endpoint_rows: pd.DataFrame,
    claim_panel: pd.DataFrame,
    extra_vanilla_dirs: tuple[Path, ...] = (),
) -> pd.DataFrame:
    endpoints = _endpoint_lookup(endpoint_rows)
    gates = _gate_lookup(claim_panel)
    rows: list[dict[str, Any]] = []
    for _, row in panel.iterrows():
        pair_id = str(row["panel_pair_id"])
        field = str(row["field"])
        case_id = str(row["case_id"])
        left_id = str(row["left_endpoint_identity_id"])
        right_id = str(row["right_endpoint_identity_id"])
        left = endpoints.get(left_id, {})
        right = endpoints.get(right_id, {})
        left_source = str(left.get("source_artifact", ""))
        right_source = str(right.get("source_artifact", ""))
        source_status = _candidate_source_status(left_source, right_source)
        vanilla_dir, vanilla_status = _vanilla_context_status(
            field,
            case_id,
            extra_vanilla_dirs=extra_vanilla_dirs,
        )
        (
            runner_preflight_status,
            runner_preflight_notes,
            runner_preflight_graph_dir,
            candidate_preflight_status,
            vanilla_preflight_status,
        ) = _runner_preflight(
            row=row,
            left_source=left_source,
            right_source=right_source,
            vanilla_dir_text=vanilla_dir,
        )
        runner_status = _runner_context_status_from_preflight(runner_preflight_status)
        gate = gates.get(pair_id, {})
        route_status = gate.get("route_order_sensitivity_status", "not_run")
        gate_status = gate.get("wall_claim_gate_status", "not_run")
        relation = str(row["calibrated_relation"])
        field_hygiene_status = _field_hygiene_status(
            field,
            relation,
            left.get("support_node_count", ""),
            right.get("support_node_count", ""),
        )
        rows.append(
            {
                "panel_pair_id": pair_id,
                "panel_role": str(row["panel_role"]),
                "protocol_priority": row.get("protocol_priority", ""),
                "source_label": str(row["source_label"]),
                "case_id": case_id,
                "field": field,
                "method": str(row["method"]),
                "candidate_budget": row.get("candidate_budget", ""),
                "left_endpoint_identity_id": left_id,
                "right_endpoint_identity_id": right_id,
                "left_representative_candidate_index": row.get(
                    "left_representative_candidate_index", ""
                ),
                "right_representative_candidate_index": row.get(
                    "right_representative_candidate_index", ""
                ),
                "calibrated_relation": relation,
                "endpoint_distance_min": row.get("endpoint_distance_min", ""),
                "endpoint_distance_max": row.get("endpoint_distance_max", ""),
                "support_distance_min": row.get("support_distance_min", ""),
                "support_distance_max": row.get("support_distance_max", ""),
                "left_support_node_count": left.get("support_node_count", ""),
                "right_support_node_count": right.get("support_node_count", ""),
                "left_endpoint_source_artifact": left_source,
                "right_endpoint_source_artifact": right_source,
                "candidate_source_status": source_status,
                "vanilla_context_dir": vanilla_dir,
                "vanilla_context_status": vanilla_status,
                "runner_preflight_status": runner_preflight_status,
                "runner_preflight_notes": runner_preflight_notes,
                "runner_preflight_graph_dir": runner_preflight_graph_dir,
                "candidate_preflight_status": candidate_preflight_status,
                "vanilla_preflight_status": vanilla_preflight_status,
                "runner_context_status": runner_status,
                "field_hygiene_status": field_hygiene_status,
                "existing_route_order_sensitivity_status": route_status,
                "existing_wall_claim_gate_status": gate_status,
                "existing_gate_source_output": gate.get("source_output", ""),
                "next_action": _next_action(
                    relation,
                    runner_status,
                    route_status,
                    gate_status,
                    field_hygiene_status,
                ),
                "claim_boundary": (
                    "Context coverage only; no basin ranking, wall promotion, "
                    "or route execution is made."
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["protocol_priority", "panel_pair_id"])


def _case_requirements(coverage: pd.DataFrame) -> pd.DataFrame:
    grouped = coverage.groupby(["field", "case_id"], dropna=False)
    rows: list[dict[str, Any]] = []
    for (field, case_id), group in grouped:
        relation_counts = group["calibrated_relation"].value_counts().to_dict()
        action_counts = group["next_action"].value_counts().to_dict()
        rows.append(
            {
                "field": field,
                "case_id": case_id,
                "panel_pair_count": int(len(group)),
                "distinct_pair_count": int(relation_counts.get("distinct_support_local", 0)),
                "ambiguous_pair_count": int(relation_counts.get("ambiguous_support_local", 0)),
                "same_pair_count": int(
                    sum(
                        count
                        for relation, count in relation_counts.items()
                        if str(relation).startswith("same")
                    )
                ),
                "runnable_pair_count": int(group["runner_context_status"].eq("runnable").sum()),
                "missing_vanilla_context_count": int(
                    group["runner_context_status"].eq("missing_vanilla_context").sum()
                ),
                "missing_candidate_source_count": int(
                    group["runner_context_status"].eq("missing_candidate_source").sum()
                ),
                "missing_both_count": int(group["runner_context_status"].eq("missing_both").sum()),
                "missing_graph_context_count": int(
                    group["runner_context_status"].eq("missing_graph_context").sum()
                ),
                "missing_or_invalid_runner_context_count": int(
                    group["runner_context_status"].eq("missing_or_invalid_runner_context").sum()
                ),
                "existing_gate_pair_count": int(
                    group["existing_wall_claim_gate_status"].ne("not_run").sum()
                ),
                "queued_distinct_run_count": int(
                    action_counts.get("run_w1_w6_route_order_gate", 0)
                ),
                "ambiguous_refinement_count": int(
                    action_counts.get("refine_ambiguous_relation_before_wall_claim", 0)
                ),
                "hygiene_review_before_route_count": int(
                    action_counts.get("review_hygiene_before_route_gate", 0)
                ),
                "vanilla_context_statuses": "|".join(
                    sorted(set(group["vanilla_context_status"].astype(str)))
                ),
                "runner_preflight_statuses": "|".join(
                    sorted(set(group["runner_preflight_status"].astype(str)))
                ),
                "field_hygiene_statuses": "|".join(
                    sorted(set(group["field_hygiene_status"].astype(str)))
                ),
                "runner_context_statuses": "|".join(
                    sorted(set(group["runner_context_status"].astype(str)))
                ),
                "next_actions": "|".join(sorted(set(group["next_action"].astype(str)))),
            }
        )
    return pd.DataFrame(rows).sort_values(["field", "case_id"])


def _ambiguous_band(row: pd.Series) -> str:
    support_max = _safe_float(row.get("support_distance_max"))
    if not math.isfinite(support_max):
        return "unknown"
    if support_max >= AMBIGUOUS_NEAR_DISTINCT_MIN:
        return "near_distinct"
    if support_max <= AMBIGUOUS_NEAR_SAME_MAX:
        return "near_same"
    return "middle"


def _relation_refinement_need(row: pd.Series) -> str:
    band = _ambiguous_band(row)
    if row.get("existing_wall_claim_gate_status") == (
        "stable_route_evidence_basin_relation_ambiguous_no_supported_wall_claim"
    ):
        return "stronger_basin_identity_evidence_for_stable_route"
    if band == "near_distinct":
        return "threshold_sensitivity_or_full_membership_check"
    if band == "near_same":
        return "same_zone_control_rule_check"
    return "stronger_signature_or_membership_check"


def _ambiguous_queue(coverage: pd.DataFrame) -> pd.DataFrame:
    queue = coverage[coverage["calibrated_relation"].eq("ambiguous_support_local")].copy()
    if queue.empty:
        return pd.DataFrame()
    queue["ambiguous_band"] = queue.apply(_ambiguous_band, axis=1)
    queue["relation_refinement_need"] = queue.apply(_relation_refinement_need, axis=1)
    queue["has_stable_route_evidence"] = queue["existing_wall_claim_gate_status"].eq(
        "stable_route_evidence_basin_relation_ambiguous_no_supported_wall_claim"
    )
    queue["next_relation_action"] = queue.apply(
        lambda row: (
            "prioritize_relation_refinement"
            if bool(row["has_stable_route_evidence"])
            else (
                "hold_route_gate_until_relation_rule_fixed"
                if row["runner_context_status"] == "runnable"
                else "resolve_context_then_recheck_relation"
            )
        ),
        axis=1,
    )
    cols = [
        "panel_pair_id",
        "panel_role",
        "field",
        "case_id",
        "method",
        "left_endpoint_identity_id",
        "right_endpoint_identity_id",
        "support_distance_min",
        "support_distance_max",
        "ambiguous_band",
        "runner_context_status",
        "runner_preflight_status",
        "field_hygiene_status",
        "existing_route_order_sensitivity_status",
        "existing_wall_claim_gate_status",
        "has_stable_route_evidence",
        "relation_refinement_need",
        "next_relation_action",
    ]
    return queue[cols].sort_values(
        ["has_stable_route_evidence", "ambiguous_band", "support_distance_max", "panel_pair_id"],
        ascending=[False, True, False, True],
    )


def _distinct_queue(coverage: pd.DataFrame) -> pd.DataFrame:
    queue = coverage[coverage["calibrated_relation"].eq("distinct_support_local")].copy()
    if queue.empty:
        return pd.DataFrame()
    queue["distinct_route_gate_status"] = queue.apply(
        lambda row: (
            "current_distinct_partial_wall_gate"
            if row["existing_wall_claim_gate_status"]
            == "passes_schedule_invariance_distinct_partial_wall_evidence"
            else (
                "route_order_sensitive_control"
                if row["existing_route_order_sensitivity_status"] == "route_order_sensitive"
                else (
                "field_hygiene_review_before_route_gate"
                if row["next_action"] == "review_hygiene_before_route_gate"
                else (
                    "queued_for_w1_w6_route_order_gate"
                    if row["runner_context_status"] == "runnable"
                    else "context_required_before_route_gate"
                    )
                )
            )
        ),
        axis=1,
    )
    cols = [
        "panel_pair_id",
        "panel_role",
        "field",
        "case_id",
        "method",
        "left_endpoint_identity_id",
        "right_endpoint_identity_id",
        "support_distance_min",
        "support_distance_max",
        "runner_context_status",
        "runner_preflight_status",
        "field_hygiene_status",
        "candidate_source_status",
        "vanilla_context_status",
        "existing_route_order_sensitivity_status",
        "existing_wall_claim_gate_status",
        "distinct_route_gate_status",
        "next_action",
    ]
    return queue[cols].sort_values(
        ["distinct_route_gate_status", "runner_context_status", "support_distance_max", "panel_pair_id"],
        ascending=[True, True, False, True],
    )


def _gate_input_metadata(gate_dir: Path, claim_panel: pd.DataFrame) -> dict[str, Any]:
    config = _read_json(gate_dir / "uniform_wall_probe_runner_config.json")
    summary = _read_json(gate_dir / "uniform_wall_probe_runner_summary.json")
    if claim_panel.empty:
        status = "missing_or_empty_gate_claim_panel"
    else:
        status = "available"
    return {
        "gate_input_status": status,
        "gate_dir": _rel(gate_dir),
        "gate_claim_pair_count": int(len(claim_panel)),
        "gate_runner_status": str(summary.get("status", "")),
        "gate_runner_pair_count": summary.get("pair_count", ""),
        "gate_runner_error_count": summary.get("error_count", ""),
        "gate_route_schedules": config.get("route_schedules", summary.get("route_schedules", [])),
        "gate_selected_pair_ids": config.get("pair_ids", summary.get("selected_pair_ids", [])),
    }


def _summary(
    coverage: pd.DataFrame,
    case_requirements: pd.DataFrame,
    ambiguous_queue: pd.DataFrame,
    distinct_queue: pd.DataFrame,
    gate_metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "status": "wall_panel_context_coverage_prepared",
        "date": "2026-05-28",
        "panel_pair_count": int(len(coverage)),
        "case_count": int(len(case_requirements)),
        "relation_counts": coverage["calibrated_relation"].value_counts().to_dict(),
        "runner_context_status_counts": coverage["runner_context_status"].value_counts().to_dict(),
        "runner_preflight_status_counts": coverage[
            "runner_preflight_status"
        ].value_counts().to_dict(),
        "field_hygiene_status_counts": coverage["field_hygiene_status"].value_counts().to_dict(),
        "next_action_counts": coverage["next_action"].value_counts().to_dict(),
        "existing_wall_claim_gate_status_counts": coverage[
            "existing_wall_claim_gate_status"
        ].value_counts().to_dict(),
        "runnable_not_run_distinct_pair_count": int(
            coverage["next_action"].eq("run_w1_w6_route_order_gate").sum()
        ),
        "hygiene_review_distinct_pair_count": int(
            coverage["next_action"].eq("review_hygiene_before_route_gate").sum()
        ),
        "ambiguous_refinement_pair_count": int(len(ambiguous_queue)),
        "stable_ambiguous_route_evidence_count": int(
            ambiguous_queue["has_stable_route_evidence"].sum()
        )
        if not ambiguous_queue.empty
        else 0,
        "distinct_queue_status_counts": distinct_queue["distinct_route_gate_status"]
        .value_counts()
        .to_dict()
        if not distinct_queue.empty
        else {},
        **gate_metadata,
        "decision": (
            "Use the context coverage rows to choose the next route-gate batch; "
            "refine ambiguous basin relations before wall promotion."
        ),
        "claim_boundary": (
            "This is a preparation artifact only. It does not rank basins, "
            "promote wall claims, or run new routes."
        ),
    }


def _write_report(
    path: Path,
    summary: dict[str, Any],
    coverage: pd.DataFrame,
    ambiguous_queue: pd.DataFrame,
    distinct_queue: pd.DataFrame,
) -> None:
    lines = [
        "# Leiden Basin Wall Panel Context Coverage",
        "",
        "Status: 23-pair wall panel context coverage prepared",
        "Date: 2026-05-28",
        "",
        "This artifact prepares the next route-gate batch. It does not run routes, rank basins, or promote wall claims.",
        "",
        "## Coverage Summary",
        "",
        f"- panel pairs: {summary['panel_pair_count']}",
        f"- cases: {summary['case_count']}",
        f"- runnable not-run distinct pairs: {summary['runnable_not_run_distinct_pair_count']}",
        f"- distinct pairs needing hygiene review before route gate: {summary['hygiene_review_distinct_pair_count']}",
        f"- ambiguous pairs needing relation refinement: {summary['ambiguous_refinement_pair_count']}",
        f"- stable ambiguous route-evidence rows: {summary['stable_ambiguous_route_evidence_count']}",
        f"- gate input status: {summary['gate_input_status']} ({summary['gate_claim_pair_count']} claim rows)",
        "",
        "## Runner Context Status",
        "",
        "| status | pairs |",
        "| --- | ---: |",
    ]
    for status, count in sorted(summary["runner_context_status_counts"].items()):
        lines.append(f"| {status} | {count} |")
    lines.extend(
        [
            "",
            "## Runner Preflight Status",
            "",
            "| status | pairs |",
            "| --- | ---: |",
        ]
    )
    for status, count in sorted(summary["runner_preflight_status_counts"].items()):
        lines.append(f"| {status} | {count} |")
    lines.extend(
        [
            "",
            "## Field Hygiene Status",
            "",
            "| status | pairs |",
            "| --- | ---: |",
        ]
    )
    for status, count in sorted(summary["field_hygiene_status_counts"].items()):
        lines.append(f"| {status} | {count} |")
    lines.extend(["", "## Next Actions", "", "| action | pairs |", "| --- | ---: |"])
    for action, count in sorted(summary["next_action_counts"].items()):
        lines.append(f"| {action} | {count} |")

    lines.extend(
        [
            "",
            "## Distinct Pair Queue",
            "",
            "| pair_id | field | route_gate_status | runner_context | next_action |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for _, row in distinct_queue.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["panel_pair_id"]),
                    str(row["field"]),
                    str(row["distinct_route_gate_status"]),
                    str(row["runner_context_status"]),
                    str(row["next_action"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Ambiguous Relation Queue",
            "",
            "| pair_id | field | band | stable_route_evidence | next_relation_action |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for _, row in ambiguous_queue.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["panel_pair_id"]),
                    str(row["field"]),
                    str(row["ambiguous_band"]),
                    str(row["has_stable_route_evidence"]),
                    str(row["next_relation_action"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- Keep the current distinct partial-wall gate as provisional route evidence, not as basin evaluation.",
            "- Run new W1-W6 route-order gates only for distinct pairs whose runner preflight is ready and whose field hygiene does not require review.",
            "- Treat field34 or tiny-support pairs as hygiene-review candidates before route execution.",
            "- Treat stable ambiguous route rows as basin-relation refinement targets before any wall promotion.",
            "- Resolve missing vanilla graph context before expanding route execution to unavailable cases.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    panel_dir: Path,
    calibration_dir: Path,
    gate_dir: Path,
    output_dir: Path,
    extra_vanilla_dirs: tuple[Path, ...] = (),
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    panel = _read_csv(panel_dir / PANEL_CSV)
    endpoint_rows = _read_csv(calibration_dir / ENDPOINT_ROWS_CSV)
    claim_panel = _read_csv(gate_dir / CLAIM_PANEL_CSV)
    if panel.empty:
        raise FileNotFoundError(panel_dir / PANEL_CSV)
    if endpoint_rows.empty:
        raise FileNotFoundError(calibration_dir / ENDPOINT_ROWS_CSV)

    normalized_extra_vanilla_dirs = tuple(_resolve_path(path) for path in extra_vanilla_dirs)
    coverage = _coverage_rows(
        panel,
        endpoint_rows,
        claim_panel,
        extra_vanilla_dirs=normalized_extra_vanilla_dirs,
    )
    case_requirements = _case_requirements(coverage)
    ambiguous = _ambiguous_queue(coverage)
    distinct = _distinct_queue(coverage)
    gate_metadata = _gate_input_metadata(gate_dir, claim_panel)
    summary = _summary(coverage, case_requirements, ambiguous, distinct, gate_metadata)

    _write_csv(coverage, output_dir / COVERAGE_ROWS_CSV)
    _write_csv(case_requirements, output_dir / CASE_REQUIREMENTS_CSV)
    _write_csv(ambiguous, output_dir / AMBIGUOUS_QUEUE_CSV)
    _write_csv(distinct, output_dir / DISTINCT_QUEUE_CSV)
    (output_dir / SUMMARY_JSON).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(
            {
                "script": _rel(Path(__file__)),
                "panel_dir": _rel(panel_dir),
                "calibration_dir": _rel(calibration_dir),
                "gate_dir": _rel(gate_dir),
                "extra_vanilla_dirs": [
                    _rel(path)
                    for path in normalized_extra_vanilla_dirs
                ],
                "gate_input_status": gate_metadata["gate_input_status"],
                "same_support_max": SAME_SUPPORT_MAX,
                "distinct_support_min": DISTINCT_SUPPORT_MIN,
                "ambiguous_near_same_max": AMBIGUOUS_NEAR_SAME_MAX,
                "ambiguous_near_distinct_min": AMBIGUOUS_NEAR_DISTINCT_MIN,
                "scope": "context coverage and runner preflight only; no route execution",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_report(output_dir / REPORT_MD, summary, coverage, ambiguous, distinct)
    return {"output_dir": _rel(output_dir), **summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-dir", type=Path, default=DEFAULT_PANEL_DIR)
    parser.add_argument("--calibration-dir", type=Path, default=DEFAULT_CALIBRATION_DIR)
    parser.add_argument("--gate-dir", type=Path, default=DEFAULT_GATE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--extra-vanilla-dir",
        dest="extra_vanilla_dirs",
        action="append",
        type=Path,
        default=[],
        help=(
            "Additional directory containing vanilla_basin_rows.csv. "
            "Used only when the default field-level vanilla context lacks the case."
        ),
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.panel_dir,
                args.calibration_dir,
                args.gate_dir,
                args.output_dir,
                extra_vanilla_dirs=tuple(args.extra_vanilla_dirs),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
