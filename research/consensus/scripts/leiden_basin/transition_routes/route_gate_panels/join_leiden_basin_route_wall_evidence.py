#!/usr/bin/env python3
"""Join calibrated basin-pair candidates to existing route/wall artifacts.

This is a narrow Phase 2 diagnostic. It starts only from
`route_join_candidate_pair_rows.csv` and reports whether existing route
artifacts provide wall evidence for those pairs. It does not rank basin quality
or make a directed-search claim.
"""

from __future__ import annotations

import argparse
import json
import math
import re
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
DEFAULT_OUTPUT_DIR = BASE_RESULT_DIR / "leiden_basin_route_wall_evidence_join_20260528"

PAIR_CONTEXT_CSV = "route_join_pair_context.csv"
ARTIFACT_INVENTORY_CSV = "route_wall_artifact_inventory.csv"
EVIDENCE_ROWS_CSV = "route_wall_evidence_rows.csv"
PAIR_SUMMARY_CSV = "route_wall_pair_summary.csv"
SUMMARY_JSON = "wall_evidence_join_summary.json"
REPORT_MD = "basin_wall_evidence_join_report.md"
CONFIG_JSON = "wall_evidence_join_config.json"

ROUTE_JOIN_CANDIDATE_PAIRS = "route_join_candidate_pair_rows.csv"
ENDPOINT_IDENTITY_ROWS = "endpoint_identity_rows.csv"
CANDIDATE_PAIR_RELATIONS = "candidate_pair_relation_rows.csv"

PAIR_ID_RE = re.compile(r"candidate:[^:]+:(\d+)")

SCAN_COLUMN_TOKENS = (
    "candidate_index",
    "left_node_id",
    "right_node_id",
    "node_id",
    "source_case",
    "source_label",
    "artifact_label",
    "prefix_rank",
    "path_prefix_rank",
    "path_policy",
    "path_selection_policy",
    "q_wall",
    "wall",
    "debt",
    "q_recovery",
    "q_recovered",
    "support_gate",
    "support_progress",
    "target_progress",
    "coverage_fraction",
    "support_distance",
    "endpoint_distance",
    "quality_barrier",
    "quality_debt",
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


def _candidate_from_identity(identity_id: str) -> str:
    return identity_id.rsplit(":", 1)[-1] if ":" in identity_id else identity_id


def _pair_id(case_id: str, left_idx: int, right_idx: int) -> str:
    lo, hi = sorted((left_idx, right_idx))
    return f"{case_id}:c{lo}-c{hi}"


def _route_family(path: Path) -> str:
    name = path.name
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


def _read_relevant_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(
            path,
            usecols=lambda column: any(token in column.lower() for token in SCAN_COLUMN_TOKENS),
        )
    except (pd.errors.EmptyDataError, ValueError):
        return pd.DataFrame()


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
        column
        for column in frame.columns
        if any(token in column.lower() for token in ("label", "status"))
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
            out[column]
            .fillna("")
            .astype(str)
            .str.extract(PAIR_ID_RE, expand=False)
        )
        out[f"{side}_candidate_index"] = pd.to_numeric(
            out[f"{side}_candidate_index"],
            errors="coerce",
        )
    return out


def _build_pair_context(calibration_dir: Path) -> pd.DataFrame:
    pairs = _read_csv(calibration_dir / ROUTE_JOIN_CANDIDATE_PAIRS)
    endpoint_rows = _read_csv(calibration_dir / ENDPOINT_IDENTITY_ROWS)
    candidate_pairs = _read_csv(calibration_dir / CANDIDATE_PAIR_RELATIONS)
    rows: list[dict[str, Any]] = []
    for _, pair in pairs.iterrows():
        left_id = str(pair["left_endpoint_identity_id"])
        right_id = str(pair["right_endpoint_identity_id"])
        case_id = str(pair["case_id"])
        match = candidate_pairs[
            (
                candidate_pairs["left_endpoint_identity_id"].eq(left_id)
                & candidate_pairs["right_endpoint_identity_id"].eq(right_id)
            )
            | (
                candidate_pairs["left_endpoint_identity_id"].eq(right_id)
                & candidate_pairs["right_endpoint_identity_id"].eq(left_id)
            )
        ]
        if match.empty:
            continue
        match_row = match.iloc[0]
        left_idx = int(match_row["left_candidate_index"])
        right_idx = int(match_row["right_candidate_index"])
        left_endpoint = endpoint_rows[
            endpoint_rows["endpoint_identity_id"].eq(match_row["left_endpoint_identity_id"])
        ].iloc[0]
        right_endpoint = endpoint_rows[
            endpoint_rows["endpoint_identity_id"].eq(match_row["right_endpoint_identity_id"])
        ].iloc[0]
        rows.append(
            {
                "pair_id": _pair_id(case_id, left_idx, right_idx),
                "case_id": case_id,
                "left_candidate_index": left_idx,
                "right_candidate_index": right_idx,
                "left_endpoint_identity_id": str(match_row["left_endpoint_identity_id"]),
                "right_endpoint_identity_id": str(match_row["right_endpoint_identity_id"]),
                "left_endpoint_short": _candidate_from_identity(str(match_row["left_endpoint_identity_id"])),
                "right_endpoint_short": _candidate_from_identity(str(match_row["right_endpoint_identity_id"])),
                "left_endpoint_signature": str(left_endpoint["endpoint_signature"]),
                "right_endpoint_signature": str(right_endpoint["endpoint_signature"]),
                "left_support_node_count": int(left_endpoint["support_node_count"]),
                "right_support_node_count": int(right_endpoint["support_node_count"]),
                "left_support_node_hash": str(left_endpoint["support_node_hash"]),
                "right_support_node_hash": str(right_endpoint["support_node_hash"]),
                "endpoint_distance": _fmt_float(_safe_float(match_row.get("endpoint_distance"))),
                "support_distance": _fmt_float(_safe_float(match_row.get("support_distance"))),
                "calibrated_relation": str(match_row["support_relation"]),
                "route_trace_source_dirs": str(pair.get("route_trace_source_dirs", "")),
            }
        )
    return pd.DataFrame(rows)


def _direct_pair_evidence(
    frame: pd.DataFrame,
    *,
    pair: pd.Series,
    artifact_dir: Path,
    csv_path: Path,
) -> list[dict[str, Any]]:
    if "left_node_id" not in frame or "right_node_id" not in frame:
        return []
    with_ids = _candidate_ids_from_nodes(frame)
    left_idx = int(pair["left_candidate_index"])
    right_idx = int(pair["right_candidate_index"])
    lo, hi = sorted((left_idx, right_idx))
    left_values = pd.to_numeric(with_ids["left_candidate_index"], errors="coerce")
    right_values = pd.to_numeric(with_ids["right_candidate_index"], errors="coerce")
    mask = (
        (left_values.eq(lo) & right_values.eq(hi))
        | (left_values.eq(hi) & right_values.eq(lo))
    )
    direct = with_ids[mask].copy()
    if direct.empty:
        return []
    rows: list[dict[str, Any]] = []
    rows.append(
        _evidence_record(
            pair=pair,
            artifact_dir=artifact_dir,
            csv_path=csv_path,
            frame=direct,
            evidence_scope="direct_pair_context",
            endpoint_side="both",
            candidate_index="",
        )
    )
    return rows


def _candidate_side_evidence(
    frame: pd.DataFrame,
    *,
    pair: pd.Series,
    artifact_dir: Path,
    csv_path: Path,
) -> list[dict[str, Any]]:
    if "candidate_index" not in frame:
        return []
    rows: list[dict[str, Any]] = []
    candidate_values = pd.to_numeric(frame["candidate_index"], errors="coerce")
    side_by_index = {
        int(pair["left_candidate_index"]): "left",
        int(pair["right_candidate_index"]): "right",
    }
    for candidate_index, endpoint_side in side_by_index.items():
        subset = frame[candidate_values.eq(candidate_index)].copy()
        if subset.empty:
            continue
        rows.append(
            _evidence_record(
                pair=pair,
                artifact_dir=artifact_dir,
                csv_path=csv_path,
                frame=subset,
                evidence_scope="candidate_route_trace",
                endpoint_side=endpoint_side,
                candidate_index=candidate_index,
            )
        )
    return rows


def _evidence_record(
    *,
    pair: pd.Series,
    artifact_dir: Path,
    csv_path: Path,
    frame: pd.DataFrame,
    evidence_scope: str,
    endpoint_side: str,
    candidate_index: int | str,
) -> dict[str, Any]:
    objective_wall = _max_metric(frame, _metric_columns(frame, "q_wall", "quality_barrier"))
    objective_debt = _max_metric(frame, _metric_columns(frame, "debt"))
    objective_recovery = _max_metric(frame, _metric_columns(frame, "q_recovery"))
    support_progress = _max_metric(frame, _metric_columns(frame, "support_progress", "target_progress", "coverage_fraction"))
    support_to_candidate_min = _min_metric(frame, _metric_columns(frame, "support_distance_to_candidate", "final_support_distance_to_candidate", "result_support_distance_to_candidate"))
    support_to_candidate_max = _max_metric(frame, _metric_columns(frame, "support_distance_to_candidate", "final_support_distance_to_candidate", "result_support_distance_to_candidate"))
    support_to_vanilla_min = _min_metric(frame, _metric_columns(frame, "support_distance_to_vanilla", "final_support_distance_to_vanilla", "result_support_distance_to_vanilla"))
    support_to_vanilla_max = _max_metric(frame, _metric_columns(frame, "support_distance_to_vanilla", "final_support_distance_to_vanilla", "result_support_distance_to_vanilla"))
    endpoint_distance_min = _min_metric(frame, _metric_columns(frame, "endpoint_distance"))
    endpoint_distance_max = _max_metric(frame, _metric_columns(frame, "endpoint_distance"))
    recovered_count = _truthy_count(frame, _metric_columns(frame, "q_recovered", "support_gate"))
    label_counts = _label_counts(frame)

    evidence_types: list[str] = []
    if math.isfinite(objective_wall) or math.isfinite(objective_debt):
        evidence_types.append("objective_wall_or_debt")
    if math.isfinite(objective_recovery) or recovered_count:
        evidence_types.append("objective_recovery_or_gate")
    if math.isfinite(support_progress) or math.isfinite(support_to_candidate_min):
        evidence_types.append("support_movement")
    if evidence_scope == "direct_pair_context":
        evidence_types.append("pair_distance_context")
    if label_counts:
        evidence_types.append("route_outcome_labels")
    if not evidence_types:
        evidence_types.append("trace_presence_only")

    if evidence_scope == "direct_pair_context":
        strength = "context_only"
    elif "objective_wall_or_debt" in evidence_types and "support_movement" in evidence_types:
        strength = "route_metric_present"
    elif "objective_wall_or_debt" in evidence_types:
        strength = "objective_metric_present"
    elif "support_movement" in evidence_types:
        strength = "support_metric_present"
    else:
        strength = "trace_presence_only"

    return {
        "pair_id": str(pair["pair_id"]),
        "case_id": str(pair["case_id"]),
        "left_candidate_index": int(pair["left_candidate_index"]),
        "right_candidate_index": int(pair["right_candidate_index"]),
        "endpoint_side": endpoint_side,
        "candidate_index": candidate_index,
        "artifact_dir": _rel(artifact_dir),
        "artifact_name": artifact_dir.name,
        "route_family": _route_family(artifact_dir),
        "source_csv": _rel(csv_path),
        "source_csv_name": csv_path.name,
        "source_row_count": int(len(frame)),
        "evidence_scope": evidence_scope,
        "evidence_strength": strength,
        "evidence_types": ";".join(evidence_types),
        "objective_wall_max": _fmt_float(objective_wall),
        "objective_debt_max": _fmt_float(objective_debt),
        "objective_recovery_max": _fmt_float(objective_recovery),
        "support_progress_max": _fmt_float(support_progress),
        "support_distance_to_candidate_min": _fmt_float(support_to_candidate_min),
        "support_distance_to_candidate_max": _fmt_float(support_to_candidate_max),
        "support_distance_to_vanilla_min": _fmt_float(support_to_vanilla_min),
        "support_distance_to_vanilla_max": _fmt_float(support_to_vanilla_max),
        "endpoint_distance_min": _fmt_float(endpoint_distance_min),
        "endpoint_distance_max": _fmt_float(endpoint_distance_max),
        "recovered_or_gate_count": recovered_count,
        "outcome_label_counts": label_counts,
    }


def _artifact_inventory(pair_context: pd.DataFrame) -> pd.DataFrame:
    dirs: set[Path] = set()
    for value in pair_context["route_trace_source_dirs"].dropna().unique():
        dirs.update(_split_dirs(value))
    rows: list[dict[str, Any]] = []
    for artifact_dir in sorted(dirs):
        csv_paths = sorted(path for path in artifact_dir.glob("*.csv") if path.is_file())
        json_paths = sorted(path for path in artifact_dir.glob("*.json") if path.is_file())
        rows.append(
            {
                "artifact_dir": _rel(artifact_dir),
                "artifact_name": artifact_dir.name,
                "route_family": _route_family(artifact_dir),
                "csv_file_count": len(csv_paths),
                "json_file_count": len(json_paths),
                "csv_files": ";".join(path.name for path in csv_paths),
            }
        )
    return pd.DataFrame(rows)


def _join_evidence(pair_context: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for _, pair in pair_context.iterrows():
        for artifact_dir in _split_dirs(pair["route_trace_source_dirs"]):
            if not artifact_dir.exists():
                continue
            for csv_path in sorted(artifact_dir.glob("*.csv")):
                frame = _read_relevant_csv(csv_path)
                if frame.empty:
                    continue
                for record in _direct_pair_evidence(
                    frame,
                    pair=pair,
                    artifact_dir=artifact_dir,
                    csv_path=csv_path,
                ) + _candidate_side_evidence(
                    frame,
                    pair=pair,
                    artifact_dir=artifact_dir,
                    csv_path=csv_path,
                ):
                    key = (
                        str(record["pair_id"]),
                        str(record["source_csv"]),
                        str(record["evidence_scope"]),
                        str(record["endpoint_side"]),
                        str(record["candidate_index"]),
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append(record)
    return pd.DataFrame(rows)


def _pair_summary(pair_context: pd.DataFrame, evidence: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, pair in pair_context.iterrows():
        pair_rows = evidence[evidence["pair_id"].eq(pair["pair_id"])] if not evidence.empty else pd.DataFrame()
        route_rows = pair_rows[pair_rows["evidence_scope"].eq("candidate_route_trace")] if not pair_rows.empty else pd.DataFrame()
        left_metric = route_rows[
            route_rows["endpoint_side"].eq("left")
            & route_rows["evidence_strength"].isin({"route_metric_present", "objective_metric_present", "support_metric_present"})
        ]
        right_metric = route_rows[
            route_rows["endpoint_side"].eq("right")
            & route_rows["evidence_strength"].isin({"route_metric_present", "objective_metric_present", "support_metric_present"})
        ]
        direct_context = pair_rows[pair_rows["evidence_scope"].eq("direct_pair_context")] if not pair_rows.empty else pd.DataFrame()
        objective_rows = route_rows[
            route_rows["evidence_types"].fillna("").str.contains("objective_wall_or_debt", regex=False)
        ]
        support_rows = route_rows[
            route_rows["evidence_types"].fillna("").str.contains("support_movement", regex=False)
        ]

        if not left_metric.empty and not right_metric.empty:
            status = "partial_both_endpoint_route_evidence"
            wall_claim = "partial"
            note = "route metrics exist for both endpoints, but no direct pair route is established"
        elif not left_metric.empty or not right_metric.empty:
            status = "partial_single_endpoint_route_evidence"
            wall_claim = "ambiguous"
            note = "route metrics exist for only one endpoint in the calibrated pair"
        elif not direct_context.empty:
            status = "context_only"
            wall_claim = "absent"
            note = "pair distance context exists, but no route metric row is joined"
        else:
            status = "no_joined_evidence"
            wall_claim = "absent"
            note = "no joined route or pair-context row found"

        rows.append(
            {
                "pair_id": str(pair["pair_id"]),
                "case_id": str(pair["case_id"]),
                "left_candidate_index": int(pair["left_candidate_index"]),
                "right_candidate_index": int(pair["right_candidate_index"]),
                "left_endpoint_identity_id": str(pair["left_endpoint_identity_id"]),
                "right_endpoint_identity_id": str(pair["right_endpoint_identity_id"]),
                "calibrated_support_distance": str(pair["support_distance"]),
                "calibrated_endpoint_distance": str(pair["endpoint_distance"]),
                "joined_evidence_rows": len(pair_rows),
                "direct_pair_context_rows": len(direct_context),
                "left_route_metric_rows": len(left_metric),
                "right_route_metric_rows": len(right_metric),
                "objective_wall_or_debt_rows": len(objective_rows),
                "support_movement_rows": len(support_rows),
                "route_family_count": pair_rows["route_family"].nunique() if not pair_rows.empty else 0,
                "wall_evidence_status": status,
                "wall_claim_status": wall_claim,
                "interpretation_note": note,
            }
        )
    return pd.DataFrame(rows)


def _write_report(path: Path, summary: dict[str, Any], pair_summary: pd.DataFrame) -> None:
    lines = [
        "# Leiden Basin Route-Wall Evidence Join",
        "",
        "Status: narrow Phase 2 evidence join",
        "Date: 2026-05-28",
        "",
        "This report joins existing route artifacts to the 3 calibrated route-join candidate pairs. It does not compare basin quality or cost.",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "| --- | --- |",
    ]
    for key in (
        "pair_count",
        "artifact_inventory_rows",
        "joined_evidence_rows",
        "direct_pair_context_rows",
        "candidate_route_trace_rows",
        "partial_pair_count",
        "ambiguous_pair_count",
        "supported_pair_count",
    ):
        lines.append(f"| {key} | {summary.get(key, '')} |")
    lines.extend(
        [
            "",
            "## Pair Summary",
            "",
            "| pair_id | support_distance | context | left_metrics | right_metrics | objective_rows | support_rows | wall_status | claim |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for _, row in pair_summary.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["pair_id"]),
                    str(row["calibrated_support_distance"]),
                    str(row["direct_pair_context_rows"]),
                    str(row["left_route_metric_rows"]),
                    str(row["right_route_metric_rows"]),
                    str(row["objective_wall_or_debt_rows"]),
                    str(row["support_movement_rows"]),
                    str(row["wall_evidence_status"]),
                    str(row["wall_claim_status"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- No pair is promoted to a supported wall claim in this join.",
            "- A pair with route metrics on both endpoint sides is still only partial unless a direct route between the calibrated endpoint identities is demonstrated.",
            "- The next useful experiment is a direct pair-route audit for the strongest partial pair, not a broad replay over all distinct pairs.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(calibration_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_context = _build_pair_context(calibration_dir)
    inventory = _artifact_inventory(pair_context)
    evidence = _join_evidence(pair_context)
    pair_summary = _pair_summary(pair_context, evidence)

    summary = {
        "status": "route_wall_evidence_join",
        "date": "2026-05-28",
        "calibration_dir": _rel(calibration_dir),
        "pair_count": int(len(pair_context)),
        "artifact_inventory_rows": int(len(inventory)),
        "joined_evidence_rows": int(len(evidence)),
        "direct_pair_context_rows": int(evidence["evidence_scope"].eq("direct_pair_context").sum()) if not evidence.empty else 0,
        "candidate_route_trace_rows": int(evidence["evidence_scope"].eq("candidate_route_trace").sum()) if not evidence.empty else 0,
        "partial_pair_count": int(pair_summary["wall_claim_status"].eq("partial").sum()) if not pair_summary.empty else 0,
        "ambiguous_pair_count": int(pair_summary["wall_claim_status"].eq("ambiguous").sum()) if not pair_summary.empty else 0,
        "supported_pair_count": int(pair_summary["wall_claim_status"].eq("supported").sum()) if not pair_summary.empty else 0,
        "claim_boundary": (
            "Joined route metrics are wall evidence candidates only; no basin-quality, "
            "cost, or directed-search claim is made."
        ),
    }

    _write_csv(pair_context, output_dir / PAIR_CONTEXT_CSV)
    _write_csv(inventory, output_dir / ARTIFACT_INVENTORY_CSV)
    _write_csv(evidence, output_dir / EVIDENCE_ROWS_CSV)
    _write_csv(pair_summary, output_dir / PAIR_SUMMARY_CSV)
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(
            {
                "script": _rel(Path(__file__)),
                "calibration_dir": _rel(calibration_dir),
                "input_pair_file": ROUTE_JOIN_CANDIDATE_PAIRS,
                "candidate_pair_count": int(len(pair_context)),
                "join_scope": "route_join_candidate_pairs_only",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / SUMMARY_JSON).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_report(output_dir / REPORT_MD, summary, pair_summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir", type=Path, default=DEFAULT_CALIBRATION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    summary = run(args.calibration_dir, args.output_dir)
    print(json.dumps({"output_dir": _rel(args.output_dir), **summary}, indent=2))


if __name__ == "__main__":
    main()
