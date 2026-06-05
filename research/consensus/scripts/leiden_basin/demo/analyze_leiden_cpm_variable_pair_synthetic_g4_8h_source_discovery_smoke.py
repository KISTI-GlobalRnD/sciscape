#!/usr/bin/env python3
"""Run a bounded target-free source-discovery smoke after G4.8G.

G4.8G validates the construction-read rule: ready cells expose a full
8-source neutral/selected/robust signature set, nonrobust cells expose only
two-side source-nonneutral signatures, and target cells expose no separated
source. This G4.8H smoke asks the next narrower question: can those source
signatures be recovered from the ordinary endpoint pool using only endpoint and
source-local bridge-release features, without reading target endpoint outcomes?

The decision rule is frozen from the previous construction read:

1. discover bridge-release source candidates from pair-separated endpoints with
   pair-attached bridge nodes and a valid bridge-release initialization;
2. narrow ready-source candidates with the frozen source-neutral/direct-support
   selector rule;
3. compare the discovered sets with the G4.8G oracle signatures only after the
   target-free decision has been made.

This is a source-discovery smoke only. It does not retune the selector, run new
Leiden jobs, promote walls/pathways, evaluate quality/cost value, replay
NanoClustering, or claim a method.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from analyze_leiden_cpm_variable_pair_synthetic_g4_5_selector_suppression import (
    DIRECT_PAIR_SUPPORT_MIN,
    SOURCE_NEUTRAL_DELTA_ABS_MAX,
)
from analyze_leiden_cpm_variable_pair_synthetic_g4_8f_centerline_signature_audit import (
    _endpoint_features,
    _median_or_none,
)
from run_leiden_cpm_variable_pair_synthetic_demo import (
    BASE_RESULT_DIR,
    _json_safe,
    _write_csv,
)
from run_leiden_cpm_variable_pair_synthetic_g4_3_handle_generalization import (
    ENDPOINT_SUMMARY_CSV as G4_3_ENDPOINT_SUMMARY_CSV,
    HANDLE_POLICY_SUMMARY_CSV as G4_3_HANDLE_POLICY_SUMMARY_CSV,
)
from run_leiden_cpm_variable_pair_synthetic_g4_8g_fresh_context_signature_validation import (
    DEFAULT_OUTPUT_DIR as DEFAULT_G4_8G_DIR,
    G4_3_DIRNAME,
    PANEL_DESIGN_CSV as G4_8G_PANEL_DESIGN_CSV,
    SIGNATURE_CASE_SUMMARY_CSV as G4_8G_SIGNATURE_CASE_SUMMARY_CSV,
    SOURCE_SIGNATURES_CSV as G4_8G_SOURCE_SIGNATURES_CSV,
)


DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_variable_pair_synthetic_g4_8h_source_discovery_smoke_v1_20260603"
)

ENDPOINT_DISCOVERY_ROWS_CSV = (
    "variable_pair_synthetic_g4_8h_endpoint_discovery_rows.csv"
)
CASE_SUMMARY_CSV = "variable_pair_synthetic_g4_8h_case_summary.csv"
CONTEXT_SUMMARY_CSV = "variable_pair_synthetic_g4_8h_context_summary.csv"
ROLE_SUMMARY_CSV = "variable_pair_synthetic_g4_8h_role_summary.csv"
SUMMARY_JSON = "variable_pair_synthetic_g4_8h_summary.json"
CONFIG_JSON = "variable_pair_synthetic_g4_8h_config.json"
REPORT_MD = "variable_pair_synthetic_g4_8h_report.md"

HANDLE_POLICY = "bridge_context_release_without_pair_merge"
SOURCE_DISCOVERY_RULE_ID = (
    "pair_separated_bridge_attached_then_neutral_release_v1"
)
CLAIM_BOUNDARY = (
    "Variable-pair synthetic G4.8H source-discovery smoke only; reads "
    "materialized G4.8G endpoint and bridge-release initialization rows to "
    "recover source signatures using target-free endpoint/source-local "
    "features. No selector retuning, no new Leiden run, no wall or pathway "
    "promotion, no quality/cost value, no NanoClustering replay, and no "
    "algorithm-level claims."
)
ROUTE_EXECUTION_STATUS = "not_executed_g4_8h_read_only_source_discovery_smoke"
WALL_PROMOTION_STATUS = "not_promoted_source_discovery_smoke_only"
METHOD_STATUS = "source_discovery_smoke_not_method_claim"

DECISION_INPUT_COLUMNS = (
    "pair_coassigned",
    "pair_attached_bridge_count",
    "handle_eligible",
    "released_bridge_count",
    "initial_pair_coassigned",
    "initial_keeps_pair_relation",
    "initial_quality_delta_vs_source",
    "direct_weight",
)
EVALUATION_ONLY_COLUMNS = (
    "role_symbol",
    "expected_role_symbol",
    "known_coassigned_endpoint_rate",
    "handle_known_coassigned_hit_rate",
    "robust_bridge_release_source",
    "oracle_source_signature",
    "oracle_selected_source",
)


def _claim_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame.drop(
        columns=[
            "route_execution_status",
            "wall_promotion_status",
            "method_status",
            "claim_boundary",
        ],
        errors="ignore",
    ).copy()
    rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["method_status"] = METHOD_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


def _bool(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _float(value: Any, default: float = 0.0) -> float:
    if pd.isna(value):
        return default
    return float(value)


def _set_text(values: set[str]) -> str:
    return ";".join(sorted(values))


def _read_inputs(g4_8g_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        "panel_design": pd.read_csv(g4_8g_dir / G4_8G_PANEL_DESIGN_CSV),
        "case_summary": pd.read_csv(
            g4_8g_dir / G4_8G_SIGNATURE_CASE_SUMMARY_CSV
        ),
        "source_signatures": pd.read_csv(
            g4_8g_dir / G4_8G_SOURCE_SIGNATURES_CSV
        ),
        "endpoint_summary": pd.read_csv(
            g4_8g_dir / G4_3_DIRNAME / G4_3_ENDPOINT_SUMMARY_CSV
        ),
        "handle_policy_summary": pd.read_csv(
            g4_8g_dir / G4_3_DIRNAME / G4_3_HANDLE_POLICY_SUMMARY_CSV
        ),
    }


def _endpoint_discovery_rows(inputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    panel = inputs["panel_design"][
        [
            "case_id",
            "context_id",
            "context_index",
            "centerline_index",
            "expected_role_symbol",
            "direct_weight",
            "pair_bridge_weight",
            "bridge_host_weight",
            "host_clique_weight",
        ]
    ].copy()
    case_summary = inputs["case_summary"][
        [
            "case_id",
            "role_symbol",
            "cartography_status",
            "baseline_pair_coassigned_run_share",
        ]
    ].copy()
    endpoints = inputs["endpoint_summary"].copy()
    features = pd.DataFrame(
        [
            _endpoint_features(signature)
            for signature in endpoints["endpoint_signature"].astype(str)
        ]
    )
    endpoints = pd.concat([endpoints.reset_index(drop=True), features], axis=1)

    bridge_policy = inputs["handle_policy_summary"][
        inputs["handle_policy_summary"]["handle_policy"].eq(HANDLE_POLICY)
    ].copy()
    handle_cols = [
        "case_id",
        "source_endpoint_signature_id",
        "handle_eligible",
        "released_bridge_nodes",
        "released_bridge_count",
        "changed_nodes_vs_source",
        "source_pair_coassigned",
        "initial_pair_coassigned",
        "initial_keeps_pair_relation",
        "initial_quality",
        "initial_cluster_count",
        "initial_quality_delta_vs_source",
        "initial_coassoc_distance_vs_source",
        "known_coassigned_endpoint_rate",
        "handle_policy_class",
    ]
    rows = (
        endpoints.merge(panel, on="case_id", how="left")
        .merge(case_summary, on="case_id", how="left")
        .merge(
            bridge_policy[handle_cols],
            left_on=["case_id", "endpoint_signature_id"],
            right_on=["case_id", "source_endpoint_signature_id"],
            how="left",
        )
    )

    oracle_source = inputs["source_signatures"].copy()
    oracle_source["oracle_source_signature"] = True
    oracle_source["oracle_selected_source"] = oracle_source[
        "selected_source"
    ].fillna(False).astype(bool)
    oracle_source["oracle_robust_bridge_release_source"] = oracle_source[
        "robust_bridge_release_source"
    ].fillna(False).astype(bool)
    oracle_cols = [
        "case_id",
        "source_endpoint_signature_id",
        "oracle_source_signature",
        "oracle_selected_source",
        "oracle_robust_bridge_release_source",
        "handle_known_coassigned_hit_rate",
        "source_availability_rate",
        "selected_schedule_contribution_rate",
    ]
    rows = rows.merge(
        oracle_source[oracle_cols],
        left_on=["case_id", "endpoint_signature_id"],
        right_on=["case_id", "source_endpoint_signature_id"],
        how="left",
        suffixes=("", "_oracle"),
    )
    rows["oracle_source_signature"] = rows["oracle_source_signature"].fillna(False)
    rows["oracle_selected_source"] = rows["oracle_selected_source"].fillna(False)
    rows["oracle_robust_bridge_release_source"] = rows[
        "oracle_robust_bridge_release_source"
    ].fillna(False)

    rows["endpoint_bridge_candidate"] = [
        (not _bool(row["pair_coassigned"]))
        and int(row["pair_attached_bridge_count"]) > 0
        for row in rows.to_dict("records")
    ]
    rows["release_source_candidate"] = [
        _release_source_candidate(row) for row in rows.to_dict("records")
    ]
    rows["source_neutral_release"] = (
        rows["initial_quality_delta_vs_source"].astype(float).abs()
        <= SOURCE_NEUTRAL_DELTA_ABS_MAX
    ).fillna(False)
    rows["direct_pair_support_floor_passed"] = (
        rows["direct_weight"].astype(float) >= DIRECT_PAIR_SUPPORT_MIN
    )
    rows["ready_source_candidate"] = [
        _ready_source_candidate(row) for row in rows.to_dict("records")
    ]
    rows["source_discovery_decision_status"] = [
        _decision_status(row) for row in rows.to_dict("records")
    ]
    rows["source_discovery_rule_id"] = SOURCE_DISCOVERY_RULE_ID
    rows["decision_input_columns"] = ",".join(DECISION_INPUT_COLUMNS)
    rows["evaluation_only_columns"] = ",".join(EVALUATION_ONLY_COLUMNS)
    return _claim_columns(
        rows.sort_values(
            [
                "context_id",
                "centerline_index",
                "pair_coassigned",
                "bridge_signature_family",
                "endpoint_rank_within_case",
                "endpoint_signature_id",
            ],
            ascending=[True, True, True, True, True, True],
            kind="stable",
        )
    )


def _release_source_candidate(row: dict[str, Any]) -> bool:
    return bool(
        _bool(row["endpoint_bridge_candidate"])
        and _bool(row["handle_eligible"])
        and int(_float(row["released_bridge_count"])) > 0
        and not _bool(row["initial_pair_coassigned"])
        and _bool(row["initial_keeps_pair_relation"])
    )


def _ready_source_candidate(row: dict[str, Any]) -> bool:
    return bool(
        _bool(row["release_source_candidate"])
        and abs(_float(row["initial_quality_delta_vs_source"], default=1.0))
        <= SOURCE_NEUTRAL_DELTA_ABS_MAX
        and _float(row["direct_weight"]) >= DIRECT_PAIR_SUPPORT_MIN
    )


def _decision_status(row: dict[str, Any]) -> str:
    if _bool(row["ready_source_candidate"]):
        return "ready_source_discovered"
    if _bool(row["release_source_candidate"]):
        if abs(_float(row["initial_quality_delta_vs_source"], default=1.0)) > SOURCE_NEUTRAL_DELTA_ABS_MAX:
            return "release_source_suppressed_nonneutral"
        if _float(row["direct_weight"]) < DIRECT_PAIR_SUPPORT_MIN:
            return "release_source_suppressed_low_direct_support"
        return "release_source_suppressed_other"
    if _bool(row["endpoint_bridge_candidate"]):
        return "endpoint_bridge_candidate_no_valid_release"
    if _bool(row["pair_coassigned"]):
        return "coassigned_endpoint_not_source"
    return "separated_endpoint_without_pair_bridge"


def _case_summary(endpoint_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for case_id, group in endpoint_rows.groupby("case_id", sort=True):
        release_set = set(
            group[group["release_source_candidate"].astype(bool)][
                "endpoint_signature_id"
            ].astype(str)
        )
        ready_set = set(
            group[group["ready_source_candidate"].astype(bool)][
                "endpoint_signature_id"
            ].astype(str)
        )
        oracle_source_set = set(
            group[group["oracle_source_signature"].astype(bool)][
                "endpoint_signature_id"
            ].astype(str)
        )
        oracle_selected_set = set(
            group[group["oracle_selected_source"].astype(bool)][
                "endpoint_signature_id"
            ].astype(str)
        )
        source_false_positive = release_set - oracle_source_set
        source_false_negative = oracle_source_set - release_set
        ready_false_positive = ready_set - oracle_selected_set
        ready_false_negative = oracle_selected_set - ready_set
        coassigned = group[group["pair_coassigned"].astype(bool)]
        release = group[group["release_source_candidate"].astype(bool)]
        ready = group[group["ready_source_candidate"].astype(bool)]
        source_exact = release_set == oracle_source_set
        ready_exact = ready_set == oracle_selected_set
        rows.append(
            {
                "case_id": str(case_id),
                "context_id": str(group["context_id"].iloc[0]),
                "context_index": int(group["context_index"].iloc[0]),
                "centerline_index": int(group["centerline_index"].iloc[0]),
                "expected_role_symbol": str(group["expected_role_symbol"].iloc[0]),
                "role_symbol": str(group["role_symbol"].iloc[0]),
                "cartography_status": str(group["cartography_status"].iloc[0]),
                "endpoint_signature_count": int(len(group)),
                "coassigned_endpoint_count": int(len(coassigned)),
                "endpoint_bridge_candidate_count": int(
                    group["endpoint_bridge_candidate"].astype(bool).sum()
                ),
                "release_source_candidate_count": int(len(release)),
                "ready_source_candidate_count": int(len(ready)),
                "oracle_source_signature_count": int(len(oracle_source_set)),
                "oracle_selected_source_count": int(len(oracle_selected_set)),
                "source_set_exact_match": bool(source_exact),
                "ready_set_exact_match": bool(ready_exact),
                "source_false_positive_count": int(len(source_false_positive)),
                "source_false_negative_count": int(len(source_false_negative)),
                "ready_false_positive_count": int(len(ready_false_positive)),
                "ready_false_negative_count": int(len(ready_false_negative)),
                "source_false_positive_ids": _set_text(source_false_positive),
                "source_false_negative_ids": _set_text(source_false_negative),
                "ready_false_positive_ids": _set_text(ready_false_positive),
                "ready_false_negative_ids": _set_text(ready_false_negative),
                "release_source_candidate_ids": _set_text(release_set),
                "ready_source_candidate_ids": _set_text(ready_set),
                "oracle_source_signature_ids": _set_text(oracle_source_set),
                "oracle_selected_source_ids": _set_text(oracle_selected_set),
                "coassigned_run_share": float(
                    coassigned["endpoint_run_share_within_case"].sum()
                ),
                "release_source_run_share": float(
                    release["endpoint_run_share_within_case"].sum()
                ),
                "ready_source_run_share": float(
                    ready["endpoint_run_share_within_case"].sum()
                ),
                "target_free_noop_run_share": float(
                    max(
                        0.0,
                        1.0
                        - coassigned["endpoint_run_share_within_case"].sum()
                        - ready["endpoint_run_share_within_case"].sum(),
                    )
                ),
                "expected_restart_plus_handle_unit_per_restart": float(
                    1.0 + ready["endpoint_run_share_within_case"].sum()
                ),
                "ready_source_initial_delta_median": _median_or_none(
                    ready["initial_quality_delta_vs_source"]
                ),
                "oracle_ready_handle_hit_rate_median": _median_or_none(
                    ready["handle_known_coassigned_hit_rate"]
                ),
            }
        )
    frame = pd.DataFrame(rows)
    frame["source_discovery_expectation_status"] = [
        _case_status(row) for row in frame.to_dict("records")
    ]
    frame["source_discovery_expectation_passed"] = frame[
        "source_discovery_expectation_status"
    ].eq("source_discovery_expectation_passed")
    return _claim_columns(frame.sort_values(["context_id", "centerline_index"], kind="stable"))


def _case_status(row: dict[str, Any]) -> str:
    if not bool(row["source_set_exact_match"]):
        return "release_source_set_mismatch"
    if not bool(row["ready_set_exact_match"]):
        return "ready_source_set_mismatch"
    role = str(row["role_symbol"])
    release_count = int(row["release_source_candidate_count"])
    ready_count = int(row["ready_source_candidate_count"])
    if role == "R" and release_count == 8 and ready_count == 8:
        return "source_discovery_expectation_passed"
    if role == "N" and release_count == 4 and ready_count == 0:
        return "source_discovery_expectation_passed"
    if role == "T" and release_count == 0 and ready_count == 0:
        return "source_discovery_expectation_passed"
    return "role_count_pattern_mismatch"


def _group_summary(
    case_rows: pd.DataFrame,
    group_col: str,
    key_col_name: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, group in case_rows.groupby(group_col, sort=True):
        rows.append(
            {
                key_col_name: str(key),
                "case_count": int(len(group)),
                "source_discovery_expectation_pass_count": int(
                    group["source_discovery_expectation_passed"].astype(bool).sum()
                ),
                "source_set_exact_match_count": int(
                    group["source_set_exact_match"].astype(bool).sum()
                ),
                "ready_set_exact_match_count": int(
                    group["ready_set_exact_match"].astype(bool).sum()
                ),
                "release_source_candidate_count_sum": int(
                    group["release_source_candidate_count"].sum()
                ),
                "ready_source_candidate_count_sum": int(
                    group["ready_source_candidate_count"].sum()
                ),
                "target_free_noop_run_share_median": float(
                    group["target_free_noop_run_share"].median()
                ),
                "expected_restart_plus_handle_unit_per_restart_median": float(
                    group[
                        "expected_restart_plus_handle_unit_per_restart"
                    ].median()
                ),
                "role_sequence": "".join(
                    group.sort_values("centerline_index")["role_symbol"].astype(str)
                ),
                "status_counts": json.dumps(
                    group["source_discovery_expectation_status"]
                    .value_counts()
                    .to_dict(),
                    sort_keys=True,
                ),
            }
        )
    return _claim_columns(pd.DataFrame(rows))


def _summary(
    *,
    output_dir: Path,
    g4_8g_dir: Path,
    endpoint_rows: pd.DataFrame,
    case_rows: pd.DataFrame,
    context_summary: pd.DataFrame,
    role_summary: pd.DataFrame,
) -> dict[str, Any]:
    case_count = int(len(case_rows))
    pass_count = int(
        case_rows["source_discovery_expectation_passed"].astype(bool).sum()
    )
    source_match_count = int(case_rows["source_set_exact_match"].astype(bool).sum())
    ready_match_count = int(case_rows["ready_set_exact_match"].astype(bool).sum())
    if pass_count == case_count:
        smoke_status = "source_discovery_smoke_passed"
    elif source_match_count == case_count or ready_match_count == case_count:
        smoke_status = "source_discovery_partial_match"
    else:
        smoke_status = "source_discovery_smoke_failed"
    return {
        "schema": "variable_pair_synthetic_g4_8h_source_discovery_smoke_summary.v1",
        "status": ROUTE_EXECUTION_STATUS,
        "smoke_status": smoke_status,
        "output_dir": str(output_dir),
        "g4_8g_dir": str(g4_8g_dir),
        "source_discovery_rule_id": SOURCE_DISCOVERY_RULE_ID,
        "case_count": case_count,
        "endpoint_row_count": int(len(endpoint_rows)),
        "source_discovery_expectation_pass_count": pass_count,
        "source_set_exact_match_count": source_match_count,
        "ready_set_exact_match_count": ready_match_count,
        "source_discovery_status_counts": case_rows[
            "source_discovery_expectation_status"
        ].value_counts().to_dict(),
        "role_counts": case_rows["role_symbol"].value_counts().to_dict(),
        "release_source_candidate_count_by_role": case_rows.groupby("role_symbol")[
            "release_source_candidate_count"
        ].sum().astype(int).to_dict(),
        "ready_source_candidate_count_by_role": case_rows.groupby("role_symbol")[
            "ready_source_candidate_count"
        ].sum().astype(int).to_dict(),
        "target_free_noop_run_share_median_by_role": case_rows.groupby("role_symbol")[
            "target_free_noop_run_share"
        ].median().to_dict(),
        "expected_restart_plus_handle_unit_median_by_role": case_rows.groupby(
            "role_symbol"
        )["expected_restart_plus_handle_unit_per_restart"].median().to_dict(),
        "context_summary_row_count": int(len(context_summary)),
        "role_summary_row_count": int(len(role_summary)),
        "decision_input_columns": list(DECISION_INPUT_COLUMNS),
        "evaluation_only_columns": list(EVALUATION_ONLY_COLUMNS),
        "recommended_next_gate": _recommended_next_gate(smoke_status),
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _recommended_next_gate(smoke_status: str) -> str:
    if smoke_status == "source_discovery_smoke_passed":
        return (
            "Freeze the target-free source-discovery rule and run a fresh "
            "predeclared panel where discovered sources drive the G4.6 schedule "
            "without oracle source-signature reads."
        )
    if smoke_status == "source_discovery_partial_match":
        return (
            "Inspect source/ready set mismatches before adding any fresh "
            "schedule panel."
        )
    return (
        "Do not proceed to schedule replay; redesign the source-discovery rule "
        "or construction-read contract."
    )


def _write_report(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    case_rows: pd.DataFrame,
    context_summary: pd.DataFrame,
    role_summary: pd.DataFrame,
) -> None:
    lines = [
        "# Variable-Pair Synthetic G4.8H Source-Discovery Smoke",
        "",
        f"- status: `{summary['status']}`",
        f"- smoke_status: {summary['smoke_status']}",
        f"- case_count: {summary['case_count']}",
        f"- source_discovery_expectation_pass_count: {summary['source_discovery_expectation_pass_count']}",
        f"- source_set_exact_match_count: {summary['source_set_exact_match_count']}",
        f"- ready_set_exact_match_count: {summary['ready_set_exact_match_count']}",
        f"- source_discovery_status_counts: {summary['source_discovery_status_counts']}",
        f"- recommended_next_gate: {summary['recommended_next_gate']}",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Decision Contract",
        "",
        f"- rule_id: `{SOURCE_DISCOVERY_RULE_ID}`",
        f"- decision_input_columns: {', '.join(DECISION_INPUT_COLUMNS)}",
        f"- evaluation_only_columns: {', '.join(EVALUATION_ONLY_COLUMNS)}",
        "",
        "## Context Summary",
        "",
    ]
    lines.extend(
        _markdown_table(
            context_summary[
                [
                    "context_id",
                    "case_count",
                    "source_discovery_expectation_pass_count",
                    "release_source_candidate_count_sum",
                    "ready_source_candidate_count_sum",
                    "target_free_noop_run_share_median",
                    "expected_restart_plus_handle_unit_per_restart_median",
                    "role_sequence",
                    "status_counts",
                ]
            ]
        )
    )
    lines.extend(["", "## Role Summary", ""])
    lines.extend(
        _markdown_table(
            role_summary[
                [
                    "role_symbol",
                    "case_count",
                    "source_discovery_expectation_pass_count",
                    "release_source_candidate_count_sum",
                    "ready_source_candidate_count_sum",
                    "target_free_noop_run_share_median",
                    "expected_restart_plus_handle_unit_per_restart_median",
                    "status_counts",
                ]
            ]
        )
    )
    lines.extend(["", "## Case Summary", ""])
    lines.extend(
        _markdown_table(
            case_rows[
                [
                    "context_id",
                    "centerline_index",
                    "role_symbol",
                    "release_source_candidate_count",
                    "ready_source_candidate_count",
                    "oracle_source_signature_count",
                    "oracle_selected_source_count",
                    "source_set_exact_match",
                    "ready_set_exact_match",
                    "coassigned_run_share",
                    "release_source_run_share",
                    "ready_source_run_share",
                    "target_free_noop_run_share",
                    "source_discovery_expectation_status",
                ]
            ]
        )
    )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            (
                "G4.8H is a bounded smoke over materialized G4.8G endpoint "
                "and initialization rows. It recovers source signatures from "
                "target-free local features, then compares to oracle rows only "
                "for evaluation. It does not establish independent source "
                "discovery on new graphs, wall/pathway evidence, quality/cost "
                "value, NanoClustering replay, or a method claim."
            ),
            "",
        ]
    )
    (output_dir / REPORT_MD).write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> list[str]:
    cols = [str(col) for col in frame.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for row in frame.to_dict("records"):
        lines.append("| " + " | ".join(_format_cell(row[col]) for col in cols) + " |")
    return lines


def _format_cell(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    g4_8g_dir = Path(args.g4_8g_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = _read_inputs(g4_8g_dir)
    endpoint_rows = _endpoint_discovery_rows(inputs)
    case_rows = _case_summary(endpoint_rows)
    context_summary = _group_summary(case_rows, "context_id", "context_id")
    role_summary = _group_summary(case_rows, "role_symbol", "role_symbol")
    _write_csv(endpoint_rows, output_dir / ENDPOINT_DISCOVERY_ROWS_CSV)
    _write_csv(case_rows, output_dir / CASE_SUMMARY_CSV)
    _write_csv(context_summary, output_dir / CONTEXT_SUMMARY_CSV)
    _write_csv(role_summary, output_dir / ROLE_SUMMARY_CSV)
    summary = _summary(
        output_dir=output_dir,
        g4_8g_dir=g4_8g_dir,
        endpoint_rows=endpoint_rows,
        case_rows=case_rows,
        context_summary=context_summary,
        role_summary=role_summary,
    )
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    config = {
        "schema": "variable_pair_synthetic_g4_8h_source_discovery_smoke_config.v1",
        "output_dir": str(output_dir),
        "g4_8g_dir": str(g4_8g_dir),
        "source_discovery_rule_id": SOURCE_DISCOVERY_RULE_ID,
        "handle_policy": HANDLE_POLICY,
        "source_neutral_delta_abs_max": SOURCE_NEUTRAL_DELTA_ABS_MAX,
        "direct_pair_support_min": DIRECT_PAIR_SUPPORT_MIN,
        "decision_input_columns": list(DECISION_INPUT_COLUMNS),
        "evaluation_only_columns": list(EVALUATION_ONLY_COLUMNS),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(
        output_dir=output_dir,
        summary=summary,
        case_rows=case_rows,
        context_summary=context_summary,
        role_summary=role_summary,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g4-8g-dir", type=Path, default=DEFAULT_G4_8G_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    summary = analyze(parse_args())
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
