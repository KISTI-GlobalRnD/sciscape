#!/usr/bin/env python3
"""Build Phase 1 basin-only indexes from existing Leiden artifacts.

The generated tables intentionally avoid quality, cost, materiality, ranking,
and operator-success fields. They describe observed endpoint identities,
support-local grouping availability, metric hygiene, and route/wall source
availability only.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
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
DEFAULT_OUTPUT_DIR = BASE_RESULT_DIR / "leiden_basin_phase1_index_20260528"
COMBINED_ROOT = (
    BASE_RESULT_DIR
    / "leiden_multibasin_crossfield_budget12_support_20260519"
    / "combined_with_field30"
)
COMBINED_SIGNATURE_DIR = COMBINED_ROOT / "signature_review"
CROSSFIELD_ROOT = (
    BASE_RESULT_DIR / "leiden_multibasin_crossfield_budget12_support_20260519"
)
STRICT_FIELD30_ROOT = (
    BASE_RESULT_DIR / "leiden_multibasin_signature_field30_budget12_support_20260519"
)
STRICT_FIELD26_ROOT = (
    BASE_RESULT_DIR
    / "leiden_multibasin_signature_field26_citation_embedding_budget15_support_20260519"
)

ENDPOINT_TAU = 0.02
SUPPORT_TAU = 0.5

LANDSCAPE_CASE_INDEX = "landscape_case_index.csv"
METRIC_HYGIENE_AUDIT = "metric_hygiene_audit.csv"
BASIN_CARTOGRAPHY_CASE_INDEX = "basin_cartography_case_index.csv"
WALL_EVIDENCE_ROWS = "wall_evidence_rows.csv"
ROUTE_TAXONOMY_ROWS = "route_taxonomy_rows.csv"
SUMMARY_JSON = "basin_cartography_summary.json"
REPORT_MD = "leiden_landscape_diagnostics_report.md"
CONFIG_JSON = "phase1_config.json"


ROUTE_SOURCE_PATTERNS = (
    ("basin_transition_branch_target_growth_field34_cc_c0_v0", "branch_target_growth", "c0"),
    ("basin_transition_branch_target_growth_field34_cc_c2_v0", "branch_target_growth", "c2"),
    ("basin_transition_boundary_analysis_field34_cc", "boundary_analysis", "field34_cc"),
    ("basin_transition_boundary_calibration_field34_cc", "boundary_calibration", "field34_cc"),
    ("basin_transition_post_gate_recovery_moves_field34_cc_", "post_gate_recovery_moves", "field34_cc"),
    ("basin_transition_attachment_margin_", "attachment_margin", "field34_cc"),
    ("basin_transition_search_field34_cc_", "transition_search", "field34_cc"),
    ("basin_transition_closure_operator_pilot_field34_cc", "closure_operator_negative_control", "field34_cc"),
    (
        "basin_transition_label_internal_repair_pilot_field34_cc",
        "label_internal_repair_negative_control",
        "field34_cc",
    ),
    (
        "basin_transition_attachment_margin_stage2_recovery_field34_cc_",
        "stage2_recovery_negative_control",
        "field34_cc",
    ),
)


QUALITY_LIKE_COLUMNS = {
    "quality",
    "delta_q",
    "p5_delta_q",
    "best_p5_delta_q",
    "basin_best_delta_q",
    "full_p5_best_delta_q",
    "relative_delta_q_ppm",
    "p5_relative_delta_q_ppm",
    "material_basin",
    "material_candidate_count",
    "cost",
    "elapsed_sec",
    "wall_time",
    "regret",
    "best_quality_regret",
    "best_quality_regret_at_k",
    "selected_by_full_p5",
    "selected_by_p1_top1",
    "selected_by_p1_top2",
    "operator_success",
}


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


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


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


def _fmt_float(value: float) -> str:
    if not math.isfinite(value):
        return ""
    return f"{value:.10g}"


def _case_tail(case: str) -> str:
    marker = "20260514_"
    return case.split(marker, 1)[1] if marker in case else case


def _case_field_method(case: str) -> tuple[str, str]:
    tail = _case_tail(case)
    parts = tail.split("_")
    field = parts[0] if parts else ""
    method = "_".join(parts[1:]) if len(parts) > 1 else ""
    return field, method


def _case_slug(case: str, candidate_budget: int | None = None) -> str:
    slug = _case_tail(case)
    if candidate_budget is not None:
        return f"{slug}_budget{candidate_budget}"
    return slug


def _classify_graph_kind(method: str) -> str:
    if method.startswith("all_edges_"):
        return "all_edges"
    if method.startswith("gcc_"):
        return "gcc"
    return "unknown"


def _method_family(method: str) -> str:
    for prefix in ("gcc_emb_full_knn30_", "all_edges_"):
        if method.startswith(prefix):
            return method.removeprefix(prefix)
    return method


def _case_source_root(case: str, source_label: str) -> str:
    if source_label:
        return source_label
    field, method = _case_field_method(case)
    if field == "field30":
        return "strict_field30_support"
    if field == "field26" and method == "gcc_emb_full_knn30_citation_embedding":
        return "strict_field26_support_and_crossfield"
    if field in {"field12", "field26", "field34"}:
        return "crossfield_budget12_support"
    return "combined_signature_review"


def _candidate_key(case: str, candidate_budget: int) -> tuple[str, int]:
    return (case, candidate_budget)


def _candidate_rows_by_case() -> dict[tuple[str, int], pd.DataFrame]:
    out: dict[tuple[str, int], pd.DataFrame] = {}
    for path in sorted(CROSSFIELD_ROOT.glob("*/candidate_level_rows.csv")):
        frame = _read_csv(path)
        if frame.empty or "case" not in frame:
            continue
        case = str(frame["case"].iloc[0])
        candidate_budget = _safe_int(frame["candidate_budget"].iloc[0])
        frame = frame.copy()
        frame["source_artifact"] = _rel(path)
        out[_candidate_key(case, candidate_budget)] = frame
    for root in (STRICT_FIELD30_ROOT, STRICT_FIELD26_ROOT):
        for path in sorted(root.glob("*/candidate_level_rows.csv")):
            frame = _read_csv(path)
            if frame.empty or "case" not in frame:
                continue
            case = str(frame["case"].iloc[0])
            candidate_budget = _safe_int(frame["candidate_budget"].iloc[0])
            key = _candidate_key(case, candidate_budget)
            if key in out:
                continue
            frame = frame.copy()
            frame["source_artifact"] = _rel(path)
            out[key] = frame
    return out


def _path_for_strict_signature_dir(case: str, candidate_budget: int) -> Path | None:
    field, method = _case_field_method(case)
    if field == "field30" and candidate_budget == 12:
        return STRICT_FIELD30_ROOT / "signature_review"
    if (
        field == "field26"
        and method == "gcc_emb_full_knn30_citation_embedding"
        and candidate_budget == 15
    ):
        return STRICT_FIELD26_ROOT / "signature_review"
    return None


def _support_count_stats(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty or "p5_basin_changed_support_node_count" not in frame:
        return {
            "support_count_min": "",
            "support_count_max": "",
            "zero_support_rows": 0,
            "exact_support_capture": "unknown",
        }
    counts = pd.to_numeric(frame["p5_basin_changed_support_node_count"], errors="coerce")
    sizes = pd.to_numeric(
        frame.get("p5_basin_changed_support_sketch_sample_size"),
        errors="coerce",
    )
    valid = counts.dropna()
    zero_support_rows = int((counts.fillna(0) == 0).sum())
    exact = "unknown"
    if not valid.empty and len(sizes) == len(counts):
        exact = "yes" if bool((counts.fillna(-1) == sizes.fillna(-2)).all()) else "no"
    return {
        "support_count_min": int(valid.min()) if not valid.empty else "",
        "support_count_max": int(valid.max()) if not valid.empty else "",
        "zero_support_rows": zero_support_rows,
        "exact_support_capture": exact,
    }


def _duplicate_signature_stats(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty or "p5_basin_signature" not in frame:
        return {
            "endpoint_identity_count_from_rows": 0,
            "duplicate_signature_groups": 0,
            "duplicate_endpoint_rows": 0,
        }
    signatures = frame["p5_basin_signature"].fillna("").astype(str)
    signatures = signatures[signatures.str.len() > 0]
    counts = Counter(signatures)
    duplicate_groups = sum(1 for value in counts.values() if value > 1)
    duplicate_rows = sum(value - 1 for value in counts.values() if value > 1)
    return {
        "endpoint_identity_count_from_rows": len(counts),
        "duplicate_signature_groups": duplicate_groups,
        "duplicate_endpoint_rows": duplicate_rows,
    }


def _pairwise_stats(pairwise: pd.DataFrame, case: str) -> dict[str, Any]:
    if pairwise.empty:
        return {
            "pairwise_rows": 0,
            "support_distance_source": "",
            "has_pairwise_endpoint_distance": "no",
            "endpoint_distance_metric": "sample_coassignment_distance",
            "endpoint_distance_max": "",
            "support_distance_metric": "coarse_support_distance",
            "support_distance_min": "",
            "support_distance_mean": "",
            "support_distance_max": "",
            "same_coarse_pairs": 0,
        }
    rows = pairwise[pairwise["case"].astype(str) == case].copy()
    if rows.empty:
        return _pairwise_stats(pd.DataFrame(), case)
    endpoint = pd.to_numeric(rows.get("sample_coassignment_distance"), errors="coerce")
    support = pd.to_numeric(rows.get("coarse_support_distance"), errors="coerce")
    same_coarse = rows.get("same_coarse_basin", pd.Series([], dtype=object))
    sources = sorted(
        str(value)
        for value in rows.get("coarse_support_distance_source", pd.Series([], dtype=object))
        .dropna()
        .unique()
    )
    return {
        "pairwise_rows": len(rows),
        "support_distance_source": ";".join(sources),
        "has_pairwise_endpoint_distance": "yes" if endpoint.notna().any() else "no",
        "endpoint_distance_metric": "sample_coassignment_distance",
        "endpoint_distance_max": _fmt_float(float(endpoint.max()))
        if endpoint.notna().any()
        else "",
        "support_distance_metric": "coarse_support_distance",
        "support_distance_min": _fmt_float(float(support.min()))
        if support.notna().any()
        else "",
        "support_distance_mean": _fmt_float(float(support.mean()))
        if support.notna().any()
        else "",
        "support_distance_max": _fmt_float(float(support.max()))
        if support.notna().any()
        else "",
        "same_coarse_pairs": int(
            same_coarse.astype(str).str.lower().isin({"true", "1"}).sum()
        )
        if not same_coarse.empty
        else 0,
    }


def _threshold_stats(case: str, candidate_budget: int) -> dict[str, Any]:
    strict_dir = _path_for_strict_signature_dir(case, candidate_budget)
    if strict_dir is None:
        return {
            "has_threshold_sensitivity": "no",
            "threshold_source_artifact": "",
            "threshold_coarse_count_min": "",
            "threshold_coarse_count_max": "",
            "threshold_exact_count_min": "",
            "threshold_exact_count_max": "",
        }
    threshold_path = strict_dir.parent / "threshold_sensitivity/leiden_multibasin_threshold_sensitivity.csv"
    frame = _read_csv(threshold_path)
    if frame.empty:
        return {
            "has_threshold_sensitivity": "no",
            "threshold_source_artifact": "",
            "threshold_coarse_count_min": "",
            "threshold_coarse_count_max": "",
            "threshold_exact_count_min": "",
            "threshold_exact_count_max": "",
        }
    case_rows = frame[frame["case"].astype(str) == case]
    if case_rows.empty:
        return {
            "has_threshold_sensitivity": "no",
            "threshold_source_artifact": "",
            "threshold_coarse_count_min": "",
            "threshold_coarse_count_max": "",
            "threshold_exact_count_min": "",
            "threshold_exact_count_max": "",
        }
    coarse = pd.to_numeric(case_rows["coarse_basin_count"], errors="coerce")
    exact = pd.to_numeric(case_rows["exact_basin_count"], errors="coerce")
    return {
        "has_threshold_sensitivity": "yes",
        "threshold_source_artifact": _rel(threshold_path),
        "threshold_coarse_count_min": int(coarse.min()) if coarse.notna().any() else "",
        "threshold_coarse_count_max": int(coarse.max()) if coarse.notna().any() else "",
        "threshold_exact_count_min": int(exact.min()) if exact.notna().any() else "",
        "threshold_exact_count_max": int(exact.max()) if exact.notna().any() else "",
    }


def _route_trace_sources() -> dict[str, list[Path]]:
    sources: dict[str, list[Path]] = {}
    if not COMBINED_ROOT.exists():
        return sources
    for path in sorted(COMBINED_ROOT.iterdir()):
        if not path.is_dir():
            continue
        name = path.name
        if not name.startswith("basin_transition_"):
            continue
        case_key = "field34_all_edges_cc_cosine_budget12" if "field34_cc" in name else "unknown"
        sources.setdefault(case_key, []).append(path)
    return sources


def _source_specs() -> list[dict[str, Any]]:
    return [
        {
            "source_label": "combined_crossfield_support",
            "signature_dir": COMBINED_SIGNATURE_DIR,
        },
        {
            "source_label": "strict_field30_support",
            "signature_dir": STRICT_FIELD30_ROOT / "signature_review",
        },
        {
            "source_label": "strict_field26_support",
            "signature_dir": STRICT_FIELD26_ROOT / "signature_review",
        },
    ]


def _case_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for spec in _source_specs():
        signature_dir = Path(spec["signature_dir"])
        basin_summary = _read_csv(signature_dir / "leiden_multibasin_basin_summary.csv")
        coarse_rows = _read_csv(signature_dir / "leiden_multibasin_coarse_basin_rows.csv")
        pairwise = _read_csv(signature_dir / "leiden_multibasin_pairwise_basin_matrix.csv")
        if basin_summary.empty:
            continue
        for _, summary_row in basin_summary.sort_values("case").iterrows():
            candidate_budget = _safe_int(summary_row.get("candidate_budget"))
            case = str(summary_row["case"])
            case_id = _case_slug(case, candidate_budget)
            records.append(
                {
                    "case_id": case_id,
                    "case": case,
                    "candidate_budget": candidate_budget,
                    "source_label": spec["source_label"],
                    "signature_dir": signature_dir,
                    "summary_row": summary_row,
                    "coarse_rows": coarse_rows[
                        (coarse_rows["case"].astype(str) == case)
                        & (
                            pd.to_numeric(
                                coarse_rows["candidate_budget"],
                                errors="coerce",
                            ).fillna(-1)
                            == candidate_budget
                        )
                    ]
                    if not coarse_rows.empty
                    else pd.DataFrame(),
                    "pairwise": pairwise[
                        (pairwise["case"].astype(str) == case)
                        & (
                            pd.to_numeric(
                                pairwise["candidate_budget"],
                                errors="coerce",
                            ).fillna(-1)
                            == candidate_budget
                        )
                    ]
                    if not pairwise.empty
                    else pd.DataFrame(),
                }
            )
    return records


def _artifact_counts(path: Path) -> dict[str, int]:
    counts = {"csv_files": 0, "json_files": 0, "md_files": 0, "csv_rows": 0}
    for csv_path in path.glob("*.csv"):
        counts["csv_files"] += 1
        frame = _read_csv(csv_path)
        counts["csv_rows"] += len(frame)
    counts["json_files"] = len(list(path.glob("*.json")))
    counts["md_files"] = len(list(path.glob("*.md")))
    return counts


def _route_inventory_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    wall_rows: list[dict[str, Any]] = []
    route_rows: list[dict[str, Any]] = []
    if not COMBINED_ROOT.exists():
        return pd.DataFrame(), pd.DataFrame()
    for path in sorted(COMBINED_ROOT.iterdir()):
        if not path.is_dir():
            continue
        name = path.name
        matched = None
        for pattern, route_family, reference in ROUTE_SOURCE_PATTERNS:
            if pattern in name:
                matched = (route_family, reference)
                break
        if matched is None:
            continue
        route_family, reference = matched
        counts = _artifact_counts(path)
        evidence_types: list[str] = []
        if any(token in name for token in ("branch_target_growth", "search", "post_gate")):
            evidence_types.extend(["objective_debt_or_wall", "path_progress"])
        if "boundary" in name or "attachment_margin" in name:
            evidence_types.append("support_incompatibility")
        if "closure_operator" in name or "label_internal" in name or "stage2" in name:
            evidence_types.append("negative_control_or_failed_path")
        if not evidence_types:
            evidence_types.append("route_trace")
        wall_rows.append(
            {
                "case_id": "field34_all_edges_cc_cosine"
                if "field34_cc" in name
                else "unknown",
                "basin_level": "support_local",
                "source_artifact_dir": _rel(path),
                "route_family": route_family,
                "wall_evidence_type_inventory": ";".join(sorted(set(evidence_types))),
                "wall_assignment_status": "unknown_pending_basin_index",
                "csv_files": counts["csv_files"],
                "csv_rows": counts["csv_rows"],
                "notes": "inventory_only_no_wall_claim",
            }
        )
        route_rows.append(
            {
                "case_id": "field34_all_edges_cc_cosine"
                if "field34_cc" in name
                else "unknown",
                "route_family": route_family,
                "source_artifact_dir": _rel(path),
                "source_reference": reference,
                "source_basin_candidate": "unassigned",
                "target_basin_candidate": "unassigned",
                "route_label": "unknown",
                "route_assignment_status": "inventory_only_pending_basin_index",
                "csv_files": counts["csv_files"],
                "csv_rows": counts["csv_rows"],
                "notes": "route labels blocked until basin candidates are fixed",
            }
        )
    return pd.DataFrame(wall_rows), pd.DataFrame(route_rows)


def _build_indexes() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    candidates_by_case = _candidate_rows_by_case()
    route_sources = _route_trace_sources()

    landscape_rows: list[dict[str, Any]] = []
    hygiene_rows: list[dict[str, Any]] = []
    cartography_rows: list[dict[str, Any]] = []

    for record in _case_records():
        case = str(record["case"])
        case_id = str(record["case_id"])
        candidate_budget = int(record["candidate_budget"])
        summary_row = record["summary_row"]
        coarse_case_rows = record["coarse_rows"]
        pairwise_rows = record["pairwise"]
        signature_dir = Path(record["signature_dir"])
        source_label = str(record["source_label"])
        field, method = _case_field_method(case)
        graph_kind = _classify_graph_kind(method)
        family = _method_family(method)
        source_root = _case_source_root(case, source_label)
        candidate_frame = candidates_by_case.get(
            _candidate_key(case, candidate_budget),
            pd.DataFrame(),
        )
        strict_dir = _path_for_strict_signature_dir(case, candidate_budget)
        pair_stats = _pairwise_stats(pairwise_rows, case)
        support_stats = _support_count_stats(candidate_frame)
        duplicate_stats = _duplicate_signature_stats(candidate_frame)
        threshold = _threshold_stats(case, candidate_budget)

        route_key = case_id
        route_paths = route_sources.get(route_key, [])
        has_route_trace = "yes" if route_paths else "no"
        route_source_dirs = ";".join(_rel(path) for path in route_paths)

        p5_rows = _safe_int(summary_row.get("p5_labeled_count"))
        endpoint_identity_count = _safe_int(summary_row.get("distinct_basin_count"))
        support_local_group_count = len(coarse_case_rows)
        zero_support_rows = int(support_stats["zero_support_rows"])
        duplicate_endpoint_rows = int(duplicate_stats["duplicate_endpoint_rows"])

        if zero_support_rows or duplicate_endpoint_rows:
            hygiene_status = "needs_filtering"
        elif support_stats["exact_support_capture"] == "yes" and pair_stats["pairwise_rows"]:
            hygiene_status = "usable_for_phase1"
        else:
            hygiene_status = "partial"

        if threshold["has_threshold_sensitivity"] == "yes":
            ambiguity_flag = "threshold_sensitive"
        elif zero_support_rows or duplicate_endpoint_rows:
            ambiguity_flag = "no_op_or_duplicate_rows_present"
        else:
            ambiguity_flag = "threshold_not_swept"

        if source_root.startswith("strict"):
            evidence_grade = "strong_signature_exact_support"
        elif hygiene_status == "usable_for_phase1":
            evidence_grade = "strong_signature"
        else:
            evidence_grade = "proxy_or_needs_filtering"

        landscape_rows.append(
            {
                "case_id": case_id,
                "case": case,
                "field": field,
                "method": method,
                "method_family": family,
                "graph_kind": graph_kind,
                "seed_family": f"seed{_safe_int(summary_row.get('seed'))}",
                "candidate_budget": candidate_budget,
                "endpoint_protocol": str(summary_row.get("candidate_eval_mode", "")),
                "endpoint_rows": p5_rows,
                "source_root_role": source_root,
                "source_candidate_rows": str(candidate_frame["source_artifact"].iloc[0])
                if not candidate_frame.empty and "source_artifact" in candidate_frame
                else "",
                "source_signature_review": _rel(signature_dir),
                "strict_signature_review": _rel(strict_dir) if strict_dir else "",
                "has_route_trace_source": has_route_trace,
                "route_trace_source_dirs": route_source_dirs,
                "rerun_command_pointer": "see source portfolio_batch_cases.csv command",
            }
        )

        hygiene_rows.append(
            {
                "case_id": case_id,
                "field": field,
                "method": method,
                "endpoint_rows": p5_rows,
                "endpoint_identity_count": endpoint_identity_count,
                "endpoint_identity_count_from_candidate_rows": duplicate_stats[
                    "endpoint_identity_count_from_rows"
                ],
                "duplicate_signature_groups": duplicate_stats["duplicate_signature_groups"],
                "duplicate_endpoint_rows": duplicate_endpoint_rows,
                "zero_support_rows": zero_support_rows,
                "support_count_min": support_stats["support_count_min"],
                "support_count_max": support_stats["support_count_max"],
                "exact_support_capture": support_stats["exact_support_capture"],
                "support_distance_source": pair_stats["support_distance_source"],
                "has_pairwise_endpoint_distance": pair_stats[
                    "has_pairwise_endpoint_distance"
                ],
                "pairwise_rows": pair_stats["pairwise_rows"],
                "has_threshold_sensitivity": threshold["has_threshold_sensitivity"],
                "threshold_coarse_count_min": threshold["threshold_coarse_count_min"],
                "threshold_coarse_count_max": threshold["threshold_coarse_count_max"],
                "has_route_trace_source": has_route_trace,
                "metric_hygiene_status": hygiene_status,
                "hygiene_notes": "filter no-op/duplicate endpoint rows before basin interpretation"
                if hygiene_status == "needs_filtering"
                else "",
                "source_artifact": str(candidate_frame["source_artifact"].iloc[0])
                if not candidate_frame.empty and "source_artifact" in candidate_frame
                else "",
            }
        )

        cartography_rows.append(
            {
                "case_id": case_id,
                "field": field,
                "method": method,
                "endpoint_protocol": str(summary_row.get("candidate_eval_mode", "")),
                "endpoint_identity_count": endpoint_identity_count,
                "global_observed_basin_status": "unresolved_sampled_proxy_only",
                "global_distance_metric": pair_stats["endpoint_distance_metric"],
                "global_distance_threshold": "",
                "global_observed_basin_count": "",
                "global_assignment_status": "unresolved",
                "support_local_basin_id_scope": "case_local",
                "support_distance_metric": pair_stats["support_distance_metric"],
                "support_distance_threshold": SUPPORT_TAU,
                "endpoint_distance_threshold": ENDPOINT_TAU,
                "support_local_basin_count": support_local_group_count,
                "support_assignment_status": "candidate_grouping_declared_threshold",
                "support_assignment_method": "coarse_support_distance_connected_components",
                "support_distance_min": pair_stats["support_distance_min"],
                "support_distance_mean": pair_stats["support_distance_mean"],
                "support_distance_max": pair_stats["support_distance_max"],
                "same_coarse_pairs": pair_stats["same_coarse_pairs"],
                "largest_support_local_basin_size": int(
                    pd.to_numeric(coarse_case_rows.get("candidate_count"), errors="coerce").max()
                )
                if not coarse_case_rows.empty
                else "",
                "top_basin_dominance": _fmt_float(
                    float(
                        pd.to_numeric(
                            coarse_case_rows.get("candidate_count"),
                            errors="coerce",
                        ).max()
                    )
                    / float(p5_rows)
                )
                if not coarse_case_rows.empty and p5_rows
                else "",
                "evidence_grade": evidence_grade,
                "ambiguity_flag": ambiguity_flag,
                "definition_notes": "support-local grouping only; global basin unresolved",
                "source_artifact": _rel(signature_dir / "leiden_multibasin_coarse_basin_rows.csv"),
            }
        )

    wall_rows, route_rows = _route_inventory_rows()

    summary = {
        "status": "phase1_basin_only_index",
        "date": "2026-05-28",
        "endpoint_tau": ENDPOINT_TAU,
        "support_tau": SUPPORT_TAU,
        "case_count": len(landscape_rows),
        "endpoint_rows": int(sum(row["endpoint_rows"] for row in landscape_rows)),
        "endpoint_identity_count_sum": int(
            sum(row["endpoint_identity_count"] for row in hygiene_rows)
        ),
        "support_local_group_count_sum": int(
            sum(row["support_local_basin_count"] for row in cartography_rows)
        ),
        "cases_needing_filtering": int(
            sum(row["metric_hygiene_status"] == "needs_filtering" for row in hygiene_rows)
        ),
        "cases_with_threshold_sensitivity": int(
            sum(row["has_threshold_sensitivity"] == "yes" for row in hygiene_rows)
        ),
        "cases_with_route_trace_source": int(
            sum(row["has_route_trace_source"] == "yes" for row in hygiene_rows)
        ),
        "wall_source_inventory_rows": int(len(wall_rows)),
        "route_source_inventory_rows": int(len(route_rows)),
        "claim_boundary": (
            "No quality, materiality, cost, ranking, or operator-success fields "
            "are used to define basin identities."
        ),
    }

    return (
        pd.DataFrame(landscape_rows),
        pd.DataFrame(hygiene_rows),
        pd.DataFrame(cartography_rows),
        wall_rows,
        route_rows,
        summary,
    )


def _quality_column_leaks(frames: dict[str, pd.DataFrame]) -> list[str]:
    leaks: list[str] = []
    for name, frame in frames.items():
        for column in frame.columns:
            lower = column.lower()
            if any(token in lower for token in QUALITY_LIKE_COLUMNS):
                leaks.append(f"{name}:{column}")
    return leaks


def _markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 20) -> list[str]:
    if frame.empty:
        return ["_No rows._"]
    display = frame[columns].head(max_rows)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in display.iterrows():
        values = [str(row.get(column, "")) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _write_report(
    path: Path,
    *,
    landscape: pd.DataFrame,
    hygiene: pd.DataFrame,
    cartography: pd.DataFrame,
    wall_rows: pd.DataFrame,
    route_rows: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    lines = [
        "# Leiden Basin Phase 1 Diagnostics",
        "",
        "Status: basin-only artifact consolidation",
        "Date: 2026-05-28",
        "",
        "This report indexes existing Track C artifacts without ranking basins by quality, cost, materiality, or operator success.",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "| --- | --- |",
    ]
    for key in (
        "case_count",
        "endpoint_rows",
        "endpoint_identity_count_sum",
        "support_local_group_count_sum",
        "cases_needing_filtering",
        "cases_with_threshold_sensitivity",
        "cases_with_route_trace_source",
        "wall_source_inventory_rows",
        "route_source_inventory_rows",
    ):
        lines.append(f"| {key} | {summary.get(key, '')} |")

    lines.extend(
        [
            "",
            "## Basin-Only Case Index",
            "",
            "Support-local grouping is reported at the declared diagnostic threshold. It is not a final basin count.",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            cartography,
            [
                "case_id",
                "endpoint_identity_count",
                "support_local_basin_count",
                "global_observed_basin_status",
                "evidence_grade",
                "ambiguity_flag",
            ],
            max_rows=30,
        )
    )

    lines.extend(["", "## Metric Hygiene Flags", ""])
    lines.extend(
        _markdown_table(
            hygiene,
            [
                "case_id",
                "zero_support_rows",
                "duplicate_endpoint_rows",
                "exact_support_capture",
                "support_distance_source",
                "has_threshold_sensitivity",
                "metric_hygiene_status",
            ],
            max_rows=30,
        )
    )

    needs_filter = hygiene[hygiene["metric_hygiene_status"] == "needs_filtering"]
    lines.extend(["", "## Readiness Judgment", ""])
    if needs_filter.empty:
        lines.append("- All indexed cases are directly usable for Phase 1 support-local grouping.")
    else:
        lines.append(
            "- Field34-like cases with zero-support or duplicate endpoints must be filtered before basin-count interpretation."
        )
    lines.extend(
        [
            "- Global observed basin assignment remains unresolved because the global distance rule is still only a sampled proxy.",
            "- Wall and route rows are inventory-only until accepted basin candidates are fixed.",
            "- G1 endpoint inventory is ready for clean field12, field26, and field30 cases; field34 needs hygiene filtering.",
            "- G2 basin definition remains open because same/distinct/ambiguous rules are not final.",
            "",
            "## Wall And Route Source Inventory",
            "",
        ]
    )
    lines.append(f"- Wall evidence source rows: {len(wall_rows)}")
    lines.append(f"- Route taxonomy source rows: {len(route_rows)}")
    lines.append(
        "- All wall assignments are `unknown_pending_basin_index`; no wall existence claim is made here."
    )
    lines.append("")
    lines.append("## Output Files")
    lines.append("")
    for filename in (
        LANDSCAPE_CASE_INDEX,
        METRIC_HYGIENE_AUDIT,
        BASIN_CARTOGRAPHY_CASE_INDEX,
        WALL_EVIDENCE_ROWS,
        ROUTE_TAXONOMY_ROWS,
        SUMMARY_JSON,
        CONFIG_JSON,
    ):
        lines.append(f"- `{filename}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    landscape, hygiene, cartography, wall_rows, route_rows, summary = _build_indexes()
    frames = {
        LANDSCAPE_CASE_INDEX: landscape,
        METRIC_HYGIENE_AUDIT: hygiene,
        BASIN_CARTOGRAPHY_CASE_INDEX: cartography,
        WALL_EVIDENCE_ROWS: wall_rows,
        ROUTE_TAXONOMY_ROWS: route_rows,
    }
    leaks = _quality_column_leaks(frames)
    if leaks:
        raise ValueError("quality-like columns leaked into Phase 1 outputs: " + ", ".join(leaks))

    _write_csv(landscape, output_dir / LANDSCAPE_CASE_INDEX)
    _write_csv(hygiene, output_dir / METRIC_HYGIENE_AUDIT)
    _write_csv(cartography, output_dir / BASIN_CARTOGRAPHY_CASE_INDEX)
    _write_csv(wall_rows, output_dir / WALL_EVIDENCE_ROWS)
    _write_csv(route_rows, output_dir / ROUTE_TAXONOMY_ROWS)

    config = {
        "script": _rel(Path(__file__)),
        "combined_signature_dir": _rel(COMBINED_SIGNATURE_DIR),
        "crossfield_root": _rel(CROSSFIELD_ROOT),
        "strict_field30_root": _rel(STRICT_FIELD30_ROOT),
        "strict_field26_root": _rel(STRICT_FIELD26_ROOT),
        "endpoint_tau": ENDPOINT_TAU,
        "support_tau": SUPPORT_TAU,
        "quality_fields_excluded": True,
    }
    (output_dir / CONFIG_JSON).write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_report(
        output_dir / REPORT_MD,
        landscape=landscape,
        hygiene=hygiene,
        cartography=cartography,
        wall_rows=wall_rows,
        route_rows=route_rows,
        summary=summary,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    summary = run(args.output_dir)
    print(json.dumps({"output_dir": _rel(args.output_dir), **summary}, indent=2))


if __name__ == "__main__":
    main()
