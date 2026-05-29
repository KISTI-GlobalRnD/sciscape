#!/usr/bin/env python3
"""Audit direct route evidence for a single calibrated Leiden basin pair.

The default target is `field34_all_edges_cc_cosine_budget12:c0-c2`, the only
pair with route metrics on both endpoint sides after the narrow Phase 2 join.
This script checks whether existing artifacts contain a direct cross route
between the two calibrated endpoint identities. It does not run a new operator.
"""

from __future__ import annotations

import argparse
import json
import math
import re
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

BASE_RESULT_DIR = REPO_ROOT / "research/consensus/results/adaptive_refinement"
DEFAULT_JOIN_DIR = BASE_RESULT_DIR / "leiden_basin_route_wall_evidence_join_20260528"
DEFAULT_OUTPUT_DIR = BASE_RESULT_DIR / "direct_pair_route_audit_field34_cc_c0_c2_20260528"
DEFAULT_PAIR_ID = "field34_all_edges_cc_cosine_budget12:c0-c2"

PAIR_CONTEXT_CSV = "route_join_pair_context.csv"
DIRECT_CONTEXT_CSV = "direct_pair_route_context.csv"
DIRECT_ROUTE_CANDIDATE_ROWS = "direct_route_candidate_rows.csv"
DIRECT_PAIR_WALL_EVIDENCE_ROWS = "direct_pair_wall_evidence_rows.csv"
DIRECT_PAIR_SUMMARY = "direct_pair_route_summary.csv"
SUMMARY_JSON = "direct_pair_route_audit_summary.json"
REPORT_MD = "direct_pair_route_audit_report.md"
CONFIG_JSON = "direct_pair_route_audit_config.json"

PAIR_ID_RE = re.compile(r"candidate:[^:]+:(\d+)")

SCAN_COLUMN_TOKENS = (
    "candidate_index",
    "left_node_id",
    "right_node_id",
    "source_case",
    "source_label",
    "artifact_label",
    "prefix_rank",
    "path_prefix_rank",
    "path_policy",
    "path_selection_policy",
    "q_wall",
    "quality_barrier",
    "debt",
    "q_recovery",
    "q_recovered",
    "support_gate",
    "support_progress",
    "target_progress",
    "coverage_fraction",
    "support_distance",
    "endpoint_distance",
    "collapse",
    "label",
    "status",
)

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

def _fmt_float(value: float) -> str:
    return "" if not math.isfinite(value) else f"{value:.10g}"

def _split_dirs(value: Any) -> list[Path]:
    if pd.isna(value):
        return []
    return [Path(part) for part in str(value).split(";") if part]

def _read_relevant_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(
            path,
            usecols=lambda column: any(token in column.lower() for token in SCAN_COLUMN_TOKENS),
        )
    except (pd.errors.EmptyDataError, ValueError):
        return pd.DataFrame()

def _source_side_from_artifact_name(name: str) -> str:
    padded = f"_{name}_"
    if "_c0_" in padded:
        return "left_c0"
    if "_c2_" in padded:
        return "right_c2"
    return "unknown"

def _route_family(name: str) -> str:
    families = (
        "aligned_core",
        "attachment_margin",
        "boundary",
        "branch_target_growth",
        "closure",
        "gate_release",
        "greedy_failure",
        "label_internal",
        "landscape",
        "local_handle_selector",
        "minimal_pathway",
        "post_gate",
        "search",
        "selector_source_screen",
        "side_route",
        "target_elbow",
        "target_units",
        "tunneling",
        "wall_route",
    )
    for family in families:
        if family in name:
            return family
    return "other_route_artifact"

def _metric_columns(frame: pd.DataFrame, *tokens: str) -> list[str]:
    out: list[str] = []
    for column in frame.columns:
        lower = column.lower()
        if any(token in lower for token in tokens):
            out.append(column)
    return out

def _max_metric(frame: pd.DataFrame, columns: list[str]) -> float:
    values: list[float] = []
    for column in columns:
        series = pd.to_numeric(frame[column], errors="coerce")
        if series.notna().any():
            values.append(float(series.max()))
    return max(values) if values else math.nan

def _min_metric(frame: pd.DataFrame, columns: list[str]) -> float:
    values: list[float] = []
    for column in columns:
        series = pd.to_numeric(frame[column], errors="coerce")
        if series.notna().any():
            values.append(float(series.min()))
    return min(values) if values else math.nan

def _truthy_count(frame: pd.DataFrame, columns: list[str]) -> int:
    count = 0
    for column in columns:
        values = frame[column]
        if pd.api.types.is_bool_dtype(values):
            count += int(values.fillna(False).sum())
        else:
            text = values.fillna("").astype(str).str.lower()
            count += int(text.isin({"true", "1", "yes", "recovered", "reached"}).sum())
    return count

def _label_counts(frame: pd.DataFrame) -> str:
    label_columns = [
        column for column in frame.columns if any(token in column.lower() for token in ("label", "status"))
    ]
    counts: dict[str, int] = {}
    for column in label_columns:
        for value, count in frame[column].fillna("").astype(str).value_counts().items():
            if not value:
                continue
            key = f"{column}={value}"
            counts[key] = counts.get(key, 0) + int(count)
    return ";".join(f"{key}:{value}" for key, value in sorted(counts.items())[:20])

def _candidate_ids_from_nodes(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for side in ("left", "right"):
        column = f"{side}_node_id"
        if column not in out:
            out[f"{side}_candidate_index"] = pd.NA
            continue
        out[f"{side}_candidate_index"] = (
            out[column].fillna("").astype(str).str.extract(PAIR_ID_RE, expand=False)
        )
        out[f"{side}_candidate_index"] = pd.to_numeric(out[f"{side}_candidate_index"], errors="coerce")
    return out

def _load_pair_context(join_dir: Path, pair_id: str) -> pd.Series:
    context = _read_csv(join_dir / PAIR_CONTEXT_CSV)
    if context.empty:
        raise FileNotFoundError(join_dir / PAIR_CONTEXT_CSV)
    matches = context[context["pair_id"].eq(pair_id)]
    if matches.empty:
        raise ValueError(f"pair_id not found: {pair_id}")
    return matches.iloc[0]

def _direct_context_row(pair: pd.Series) -> pd.DataFrame:
    columns = [
        "pair_id",
        "case_id",
        "left_candidate_index",
        "right_candidate_index",
        "left_endpoint_identity_id",
        "right_endpoint_identity_id",
        "left_endpoint_signature",
        "right_endpoint_signature",
        "left_support_node_count",
        "right_support_node_count",
        "left_support_node_hash",
        "right_support_node_hash",
        "endpoint_distance",
        "support_distance",
        "calibrated_relation",
    ]
    return pd.DataFrame([{column: pair.get(column, "") for column in columns}])

def _direction_class(source_side: str, candidate_index: int, left_idx: int, right_idx: int) -> str:
    if source_side == "left_c0" and candidate_index == right_idx:
        return "direct_c0_to_c2"
    if source_side == "right_c2" and candidate_index == left_idx:
        return "direct_c2_to_c0"
    if source_side == "left_c0" and candidate_index == left_idx:
        return "source_self_c0"
    if source_side == "right_c2" and candidate_index == right_idx:
        return "source_self_c2"
    if candidate_index in {left_idx, right_idx}:
        return "endpoint_side_context"
    return "unrelated_candidate"

def _evidence_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    objective_barrier = _max_metric(frame, _metric_columns(frame, "q_wall", "quality_barrier"))
    objective_debt = _max_metric(frame, _metric_columns(frame, "debt"))
    objective_recovery = _max_metric(frame, _metric_columns(frame, "q_recovery"))
    support_progress = _max_metric(frame, _metric_columns(frame, "support_progress", "target_progress", "coverage_fraction"))
    support_to_candidate_min = _min_metric(
        frame,
        _metric_columns(
            frame,
            "support_distance_to_candidate",
            "final_support_distance_to_candidate",
            "result_support_distance_to_candidate",
        ),
    )
    support_to_vanilla_min = _min_metric(
        frame,
        _metric_columns(
            frame,
            "support_distance_to_vanilla",
            "final_support_distance_to_vanilla",
            "result_support_distance_to_vanilla",
        ),
    )
    endpoint_distance_min = _min_metric(frame, _metric_columns(frame, "endpoint_distance"))
    endpoint_distance_max = _max_metric(frame, _metric_columns(frame, "endpoint_distance"))
    recovered_count = _truthy_count(frame, _metric_columns(frame, "q_recovered", "support_gate"))
    return {
        "objective_barrier_max": _fmt_float(objective_barrier),
        "objective_debt_max": _fmt_float(objective_debt),
        "objective_recovery_max": _fmt_float(objective_recovery),
        "support_progress_max": _fmt_float(support_progress),
        "support_distance_to_candidate_min": _fmt_float(support_to_candidate_min),
        "support_distance_to_vanilla_min": _fmt_float(support_to_vanilla_min),
        "endpoint_distance_min": _fmt_float(endpoint_distance_min),
        "endpoint_distance_max": _fmt_float(endpoint_distance_max),
        "recovered_or_gate_count": recovered_count,
        "outcome_label_counts": _label_counts(frame),
    }

def _audit_rows(pair: pd.Series) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    left_idx = int(pair["left_candidate_index"])
    right_idx = int(pair["right_candidate_index"])
    route_dirs = _split_dirs(pair["route_trace_source_dirs"])

    for artifact_dir in sorted(route_dirs):
        if not artifact_dir.exists():
            continue
        artifact_source_side = _source_side_from_artifact_name(artifact_dir.name)
        route_family = _route_family(artifact_dir.name)
        for csv_path in sorted(artifact_dir.glob("*.csv")):
            frame = _read_relevant_csv(csv_path)
            if frame.empty:
                continue

            if "left_node_id" in frame and "right_node_id" in frame:
                with_ids = _candidate_ids_from_nodes(frame)
                left_values = pd.to_numeric(with_ids["left_candidate_index"], errors="coerce")
                right_values = pd.to_numeric(with_ids["right_candidate_index"], errors="coerce")
                mask = (
                    (left_values.eq(left_idx) & right_values.eq(right_idx))
                    | (left_values.eq(right_idx) & right_values.eq(left_idx))
                )
                direct = with_ids[mask].copy()
                if not direct.empty:
                    rows.append(
                        {
                            "pair_id": str(pair["pair_id"]),
                            "artifact_dir": _rel(artifact_dir),
                            "artifact_name": artifact_dir.name,
                            "route_family": route_family,
                            "source_csv": _rel(csv_path),
                            "source_csv_name": csv_path.name,
                            "artifact_source_side": artifact_source_side,
                            "candidate_index": "",
                            "direction_class": "direct_pair_context",
                            "evidence_role": "pair_context_only",
                            "source_row_count": len(direct),
                            **_evidence_metrics(direct),
                        }
                    )

            if "candidate_index" not in frame:
                continue
            candidate_values = pd.to_numeric(frame["candidate_index"], errors="coerce")
            for candidate_index in (left_idx, right_idx):
                subset = frame[candidate_values.eq(candidate_index)].copy()
                if subset.empty:
                    continue
                direction_class = _direction_class(
                    artifact_source_side,
                    candidate_index,
                    left_idx,
                    right_idx,
                )
                if direction_class.startswith("direct_"):
                    evidence_role = "direct_cross_route"
                elif direction_class.startswith("source_self_"):
                    evidence_role = "self_endpoint_route"
                else:
                    evidence_role = "endpoint_side_context"
                rows.append(
                    {
                        "pair_id": str(pair["pair_id"]),
                        "artifact_dir": _rel(artifact_dir),
                        "artifact_name": artifact_dir.name,
                        "route_family": route_family,
                        "source_csv": _rel(csv_path),
                        "source_csv_name": csv_path.name,
                        "artifact_source_side": artifact_source_side,
                        "candidate_index": candidate_index,
                        "direction_class": direction_class,
                        "evidence_role": evidence_role,
                        "source_row_count": len(subset),
                        **_evidence_metrics(subset),
                    }
                )

    return pd.DataFrame(rows)

def _wall_evidence_rows(candidate_rows: pd.DataFrame) -> pd.DataFrame:
    if candidate_rows.empty:
        return pd.DataFrame()
    metric_cols = [
        "objective_barrier_max",
        "objective_debt_max",
        "objective_recovery_max",
        "support_progress_max",
        "support_distance_to_candidate_min",
        "support_distance_to_vanilla_min",
    ]
    rows = candidate_rows[candidate_rows["evidence_role"].isin({"direct_cross_route", "self_endpoint_route"})].copy()
    if rows.empty:
        return rows
    has_metric = pd.Series(False, index=rows.index)
    for column in metric_cols:
        has_metric = has_metric | pd.to_numeric(rows[column], errors="coerce").notna()
    return rows[has_metric].copy()

def _summary(pair: pd.Series, candidate_rows: pd.DataFrame, wall_rows: pd.DataFrame) -> pd.DataFrame:
    direct_cross = candidate_rows[candidate_rows["evidence_role"].eq("direct_cross_route")]
    self_routes = candidate_rows[candidate_rows["evidence_role"].eq("self_endpoint_route")]
    pair_context = candidate_rows[candidate_rows["evidence_role"].eq("pair_context_only")]
    endpoint_context = candidate_rows[candidate_rows["evidence_role"].eq("endpoint_side_context")]
    objective_rows = wall_rows[
        pd.to_numeric(wall_rows.get("objective_barrier_max", pd.Series(dtype=object)), errors="coerce").notna()
        | pd.to_numeric(wall_rows.get("objective_debt_max", pd.Series(dtype=object)), errors="coerce").notna()
    ] if not wall_rows.empty else pd.DataFrame()
    support_rows = wall_rows[
        pd.to_numeric(wall_rows.get("support_progress_max", pd.Series(dtype=object)), errors="coerce").notna()
        | pd.to_numeric(wall_rows.get("support_distance_to_candidate_min", pd.Series(dtype=object)), errors="coerce").notna()
    ] if not wall_rows.empty else pd.DataFrame()

    if not direct_cross.empty:
        verdict = "partial_wall_direct_route_rows_present"
        claim = "partial_wall"
        next_step = "inspect direct cross route rows before supported claim"
    elif not self_routes.empty:
        verdict = "no_direct_pair_route_self_routes_only"
        claim = "no_wall_claim"
        next_step = "run minimal direct c0-c2 route replay or construct direct pair-route artifact"
    elif not pair_context.empty:
        verdict = "pair_context_only"
        claim = "no_wall_claim"
        next_step = "direct route evidence absent in existing artifacts"
    else:
        verdict = "no_existing_evidence"
        claim = "no_wall_claim"
        next_step = "new direct pair-route artifact required"

    return pd.DataFrame(
        [
            {
                "pair_id": str(pair["pair_id"]),
                "case_id": str(pair["case_id"]),
                "left_candidate_index": int(pair["left_candidate_index"]),
                "right_candidate_index": int(pair["right_candidate_index"]),
                "support_distance": str(pair["support_distance"]),
                "endpoint_distance": str(pair["endpoint_distance"]),
                "candidate_row_count": len(candidate_rows),
                "direct_cross_route_rows": len(direct_cross),
                "self_endpoint_route_rows": len(self_routes),
                "pair_context_rows": len(pair_context),
                "endpoint_side_context_rows": len(endpoint_context),
                "wall_metric_rows": len(wall_rows),
                "objective_wall_or_debt_rows": len(objective_rows),
                "support_movement_rows": len(support_rows),
                "verdict": verdict,
                "wall_claim_status": claim,
                "next_step": next_step,
            }
        ]
    )

def _write_report(path: Path, summary: pd.DataFrame) -> None:
    row = summary.iloc[0].to_dict()
    lines = [
        "# Direct Pair Route Audit: field34/cc c0-c2",
        "",
        "Status: existing-artifact direct route audit",
        "Date: 2026-05-28",
        "",
        "This audit checks whether existing route artifacts directly connect the calibrated c0-c2 endpoint pair. It does not run a new operator and does not compare basin quality or cost.",
        "",
        "## Result",
        "",
        "| metric | value |",
        "| --- | --- |",
    ]
    for key in (
        "pair_id",
        "support_distance",
        "endpoint_distance",
        "candidate_row_count",
        "direct_cross_route_rows",
        "self_endpoint_route_rows",
        "pair_context_rows",
        "wall_metric_rows",
        "objective_wall_or_debt_rows",
        "support_movement_rows",
        "verdict",
        "wall_claim_status",
        "next_step",
    ):
        lines.append(f"| {key} | {row.get(key, '')} |")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- Existing artifacts do not contain a direct cross route for c0-c2.",
            "- The available rows are self-endpoint route traces and direct pair context, so this remains `no_wall_claim`.",
            "- The next experiment should be a minimal direct c0-c2 route replay, not a broad replay over all distinct pairs.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def run(join_dir: Path, output_dir: Path, pair_id: str) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pair = _load_pair_context(join_dir, pair_id)
    context = _direct_context_row(pair)
    candidate_rows = _audit_rows(pair)
    wall_rows = _wall_evidence_rows(candidate_rows)
    summary = _summary(pair, candidate_rows, wall_rows)

    _write_csv(context, output_dir / DIRECT_CONTEXT_CSV)
    _write_csv(candidate_rows, output_dir / DIRECT_ROUTE_CANDIDATE_ROWS)
    _write_csv(wall_rows, output_dir / DIRECT_PAIR_WALL_EVIDENCE_ROWS)
    _write_csv(summary, output_dir / DIRECT_PAIR_SUMMARY)

    result = {
        "status": "direct_pair_route_audit",
        "date": "2026-05-28",
        "join_dir": _rel(join_dir),
        "pair_id": pair_id,
        "candidate_row_count": int(summary["candidate_row_count"].iloc[0]),
        "direct_cross_route_rows": int(summary["direct_cross_route_rows"].iloc[0]),
        "self_endpoint_route_rows": int(summary["self_endpoint_route_rows"].iloc[0]),
        "pair_context_rows": int(summary["pair_context_rows"].iloc[0]),
        "wall_metric_rows": int(summary["wall_metric_rows"].iloc[0]),
        "wall_claim_status": str(summary["wall_claim_status"].iloc[0]),
        "verdict": str(summary["verdict"].iloc[0]),
        "claim_boundary": (
            "Existing-artifact audit only; no supported wall, basin-quality, cost, "
            "or directed-search claim is made."
        ),
    }
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(
            {
                "script": _rel(Path(__file__)),
                "join_dir": _rel(join_dir),
                "pair_id": pair_id,
                "audit_scope": "existing_artifacts_only",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / SUMMARY_JSON).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _write_report(output_dir / REPORT_MD, summary)
    return result

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--join-dir", type=Path, default=DEFAULT_JOIN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pair-id", default=DEFAULT_PAIR_ID)
    args = parser.parse_args()
    summary = run(args.join_dir, args.output_dir, args.pair_id)
    print(json.dumps({"output_dir": _rel(args.output_dir), **summary}, indent=2))

if __name__ == "__main__":
    main()
