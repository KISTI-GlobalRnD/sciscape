#!/usr/bin/env python3
"""Audit centerline endpoint/source signatures after G4.8E.

G4.8E resolves the sparse diagonal into a centerline resonance lattice. This
read-only G4.8F audit does not run Leiden, retune selectors, or change the
frozen G4.3/G4.5/G4.6 rules. It compares the centerline ``R/T/N`` cells at the
endpoint-signature and source-signature level to explain why only the resonance
cells produce full bridge-release source-handle fire.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from run_leiden_cpm_variable_pair_synthetic_demo import (
    BASE_RESULT_DIR,
    _json_safe,
    _write_csv,
)
from run_leiden_cpm_variable_pair_synthetic_g4_3_handle_generalization import (
    ENDPOINT_SUMMARY_CSV as G4_3_ENDPOINT_SUMMARY_CSV,
    HANDLE_POLICY_SUMMARY_CSV as G4_3_HANDLE_POLICY_SUMMARY_CSV,
)
from run_leiden_cpm_variable_pair_synthetic_g4_8e_diagonal_ridge_refinement import (
    CASE_SUMMARY_CSV as G4_8E_CASE_SUMMARY_CSV,
    DEFAULT_OUTPUT_DIR as DEFAULT_G4_8E_DIR,
    G4_3_DIRNAME,
    G4_5_DIRNAME,
    G4_6_DIRNAME,
)
from analyze_leiden_cpm_variable_pair_synthetic_g4_5_selector_suppression import (
    SELECTOR_SOURCE_ROWS_CSV as G4_5_SELECTOR_SOURCE_ROWS_CSV,
)
from analyze_leiden_cpm_variable_pair_synthetic_g4_6_schedule_accounting import (
    SOURCE_AVAILABILITY_ROWS_CSV as G4_6_SOURCE_AVAILABILITY_ROWS_CSV,
)


DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_variable_pair_synthetic_g4_8f_centerline_signature_audit_v1_20260603"
)

CENTERLINE_CASE_SUMMARY_CSV = (
    "variable_pair_synthetic_g4_8f_centerline_case_summary.csv"
)
ENDPOINT_SIGNATURES_CSV = (
    "variable_pair_synthetic_g4_8f_centerline_endpoint_signatures.csv"
)
SOURCE_SIGNATURES_CSV = (
    "variable_pair_synthetic_g4_8f_centerline_source_signatures.csv"
)
ROLE_FAMILY_SUMMARY_CSV = (
    "variable_pair_synthetic_g4_8f_role_signature_family_summary.csv"
)
NEIGHBOR_CONTRAST_CSV = (
    "variable_pair_synthetic_g4_8f_centerline_neighbor_contrast_rows.csv"
)
SIGNATURE_PRESENCE_MATRIX_CSV = (
    "variable_pair_synthetic_g4_8f_signature_presence_matrix.csv"
)
SUMMARY_JSON = "variable_pair_synthetic_g4_8f_summary.json"
CONFIG_JSON = "variable_pair_synthetic_g4_8f_config.json"
REPORT_MD = "variable_pair_synthetic_g4_8f_report.md"

HANDLE_POLICY = "bridge_context_release_without_pair_merge"
CLAIM_BOUNDARY = (
    "Variable-pair synthetic G4.8F centerline signature audit only; reads the "
    "materialized G4.8E/G4.3/G4.5/G4.6 outputs to compare endpoint and source "
    "signatures. No new Leiden runs, no selector retuning, no source-discovery "
    "replacement, no wall or pathway promotion, no quality/cost value, and no "
    "algorithm-level claims."
)
ROUTE_EXECUTION_STATUS = "not_executed_g4_8f_read_only_signature_audit"
WALL_PROMOTION_STATUS = "not_promoted_signature_audit_only"
METHOD_STATUS = "centerline_signature_audit_not_method_claim"

LEFT_BRIDGES = ("lb0", "lb1")
RIGHT_BRIDGES = ("rb0", "rb1")


def _claim_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame.copy()
    rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["method_status"] = METHOD_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


def _load_signature_groups(endpoint_signature: str) -> list[list[str]]:
    return [
        sorted(str(node) for node in group)
        for group in json.loads(str(endpoint_signature))
    ]


def _group_with(groups: list[list[str]], node: str) -> list[str]:
    for group in groups:
        if node in group:
            return group
    return []


def _endpoint_features(endpoint_signature: str) -> dict[str, Any]:
    groups = _load_signature_groups(endpoint_signature)
    left_group = _group_with(groups, "L")
    right_group = _group_with(groups, "R")
    left_pair_bridges = sorted(node for node in left_group if node in LEFT_BRIDGES)
    right_pair_bridges = sorted(node for node in right_group if node in RIGHT_BRIDGES)
    pair_attached_bridge_count = len(left_pair_bridges) + len(right_pair_bridges)
    pair_coassigned = bool("R" in left_group and left_group)
    if pair_coassigned:
        bridge_signature_family = "coassigned_no_pair_bridge"
    elif pair_attached_bridge_count == 2:
        bridge_signature_family = "two_side_bridge_split"
    elif pair_attached_bridge_count == 1:
        bridge_signature_family = "single_side_bridge_split"
    else:
        bridge_signature_family = "pair_separated_without_pair_bridge"
    return {
        "left_pair_bridges": ";".join(left_pair_bridges) or "none",
        "right_pair_bridges": ";".join(right_pair_bridges) or "none",
        "pair_attached_bridge_count": int(pair_attached_bridge_count),
        "left_pair_bridge_count": int(len(left_pair_bridges)),
        "right_pair_bridge_count": int(len(right_pair_bridges)),
        "bridge_signature_family": bridge_signature_family,
    }


def _source_family(released_bridge_count: Any) -> str:
    if pd.isna(released_bridge_count):
        return "no_source"
    count = int(released_bridge_count)
    if count == 2:
        return "two_side_release_source"
    if count == 1:
        return "single_side_release_source"
    return "zero_release_source"


def _read_inputs(g4_8e_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        "case_summary": pd.read_csv(g4_8e_dir / G4_8E_CASE_SUMMARY_CSV),
        "endpoint_summary": pd.read_csv(
            g4_8e_dir / G4_3_DIRNAME / G4_3_ENDPOINT_SUMMARY_CSV
        ),
        "handle_policy_summary": pd.read_csv(
            g4_8e_dir / G4_3_DIRNAME / G4_3_HANDLE_POLICY_SUMMARY_CSV
        ),
        "selector_source_rows": pd.read_csv(
            g4_8e_dir / G4_5_DIRNAME / G4_5_SELECTOR_SOURCE_ROWS_CSV
        ),
        "source_availability_rows": pd.read_csv(
            g4_8e_dir / G4_6_DIRNAME / G4_6_SOURCE_AVAILABILITY_ROWS_CSV
        ),
    }


def _centerline_cases(case_summary: pd.DataFrame) -> pd.DataFrame:
    rows = case_summary[
        case_summary["bridge_host_offset"].fillna(0.0).astype(float).eq(0.0)
    ].copy()
    return rows.sort_values("pair_bridge_weight", kind="stable").reset_index(drop=True)


def _endpoint_signature_rows(
    *,
    centerline: pd.DataFrame,
    endpoint_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows = endpoint_summary[
        endpoint_summary["case_id"].isin(centerline["case_id"].astype(str))
    ].copy()
    rows = rows.merge(
        centerline[
            [
                "case_id",
                "pair_bridge_weight",
                "diagonal_bridge_host_weight",
                "bridge_host_weight",
                "role_symbol",
                "cartography_status",
            ]
        ],
        on="case_id",
        how="left",
    )
    features = pd.DataFrame(
        [_endpoint_features(sig) for sig in rows["endpoint_signature"].astype(str)]
    )
    rows = pd.concat([rows.reset_index(drop=True), features], axis=1)
    return _claim_columns(
        rows.sort_values(
            [
                "pair_bridge_weight",
                "pair_coassigned",
                "bridge_signature_family",
                "endpoint_rank_within_case",
            ],
            ascending=[True, True, True, True],
            kind="stable",
        )
    )


def _source_signature_rows(
    *,
    endpoint_rows: pd.DataFrame,
    handle_policy_summary: pd.DataFrame,
    selector_source_rows: pd.DataFrame,
    source_availability_rows: pd.DataFrame,
) -> pd.DataFrame:
    bridge_policy = handle_policy_summary[
        handle_policy_summary["handle_policy"].eq(HANDLE_POLICY)
    ].copy()
    cols = [
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
        "pair_coassigned_rate",
        "known_coassigned_endpoint_rate",
        "source_bounce_rate",
        "distinct_result_endpoint_count",
        "result_quality_delta_vs_source_median",
    ]
    sources = bridge_policy[cols].copy()
    selector_cols = [
        "case_id",
        "source_endpoint_signature_id",
        "handle_known_coassigned_hit_rate",
        "handle_pair_coassigned_hit_rate",
        "pair_only_pair_coassigned_hit_rate",
        "source_neutral_release",
        "direct_pair_support_floor_passed",
        "selector_selected",
        "selector_suppression_reason",
        "g4_5_selector_status",
    ]
    sources = sources.merge(
        selector_source_rows[selector_cols],
        on=["case_id", "source_endpoint_signature_id"],
        how="left",
    )
    availability_cols = [
        "case_id",
        "source_endpoint_signature_id",
        "source_observed_count",
        "source_availability_rate",
        "selected_schedule_contribution_rate",
    ]
    sources = sources.merge(
        source_availability_rows[availability_cols],
        on=["case_id", "source_endpoint_signature_id"],
        how="left",
    )
    endpoint_cols = [
        "case_id",
        "endpoint_signature_id",
        "endpoint_signature",
        "endpoint_run_count",
        "endpoint_run_share_within_case",
        "mechanism_read",
        "pair_bridge_weight",
        "diagonal_bridge_host_weight",
        "bridge_host_weight",
        "role_symbol",
        "cartography_status",
        "bridge_signature_family",
        "left_pair_bridges",
        "right_pair_bridges",
        "pair_attached_bridge_count",
    ]
    sources = sources.merge(
        endpoint_rows[endpoint_cols],
        left_on=["case_id", "source_endpoint_signature_id"],
        right_on=["case_id", "endpoint_signature_id"],
        how="left",
    )
    sources["source_signature_family"] = [
        _source_family(value) for value in sources["released_bridge_count"]
    ]
    sources["robust_bridge_release_source"] = (
        sources["known_coassigned_endpoint_rate"].fillna(0.0).astype(float).ge(1.0)
    )
    sources["selected_source"] = (
        sources["selector_selected"].fillna(False).astype(bool)
    )
    return _claim_columns(
        sources.sort_values(
            [
                "pair_bridge_weight",
                "source_signature_family",
                "source_endpoint_signature_id",
            ],
            kind="stable",
        )
    )


def _case_rows(
    *,
    centerline: pd.DataFrame,
    endpoint_rows: pd.DataFrame,
    source_rows: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for case in centerline.to_dict("records"):
        case_id = str(case["case_id"])
        endpoints = endpoint_rows[endpoint_rows["case_id"].astype(str).eq(case_id)]
        sources = source_rows[source_rows["case_id"].astype(str).eq(case_id)]
        coassigned = endpoints[endpoints["pair_coassigned"].astype(bool)]
        separated = endpoints[~endpoints["pair_coassigned"].astype(bool)]
        split = separated[
            separated["bridge_signature_family"].eq("two_side_bridge_split")
        ]
        single = separated[
            separated["bridge_signature_family"].eq("single_side_bridge_split")
        ]
        source_neutral = sources[
            sources["source_neutral_release"].fillna(False).astype(bool)
        ]
        selected = sources[sources["selected_source"].fillna(False).astype(bool)]
        robust = sources[
            sources["robust_bridge_release_source"].fillna(False).astype(bool)
        ]
        rows.append(
            {
                "case_id": case_id,
                "pair_bridge_weight": float(case["pair_bridge_weight"]),
                "diagonal_bridge_host_weight": float(
                    case["diagonal_bridge_host_weight"]
                ),
                "bridge_host_weight": float(case["bridge_host_weight"]),
                "role_symbol": str(case["role_symbol"]),
                "cartography_status": str(case["cartography_status"]),
                "baseline_pair_coassigned_run_share": float(
                    case["baseline_pair_coassigned_run_share"]
                ),
                "endpoint_signature_count": int(len(endpoints)),
                "coassigned_endpoint_count": int(len(coassigned)),
                "separated_endpoint_count": int(len(separated)),
                "two_side_split_endpoint_count": int(len(split)),
                "single_side_endpoint_count": int(len(single)),
                "coassigned_run_share": float(
                    coassigned["endpoint_run_share_within_case"].sum()
                )
                if not coassigned.empty
                else 0.0,
                "two_side_split_run_share": float(
                    split["endpoint_run_share_within_case"].sum()
                )
                if not split.empty
                else 0.0,
                "single_side_run_share": float(
                    single["endpoint_run_share_within_case"].sum()
                )
                if not single.empty
                else 0.0,
                "coassigned_quality_median": _median_or_none(
                    coassigned["quality_median"]
                ),
                "separated_quality_median": _median_or_none(
                    separated["quality_median"]
                ),
                "source_signature_count": int(len(sources)),
                "two_side_release_source_count": int(
                    sources["source_signature_family"]
                    .eq("two_side_release_source")
                    .sum()
                ),
                "single_side_release_source_count": int(
                    sources["source_signature_family"]
                    .eq("single_side_release_source")
                    .sum()
                ),
                "source_neutral_count": int(len(source_neutral)),
                "selected_source_count": int(len(selected)),
                "robust_bridge_release_source_count": int(len(robust)),
                "source_handle_fire": bool(case["source_handle_fire"]),
                "handle_known_hit_rate_min": _min_or_none(
                    sources["handle_known_coassigned_hit_rate"]
                ),
                "handle_known_hit_rate_median": _median_or_none(
                    sources["handle_known_coassigned_hit_rate"]
                ),
                "handle_known_hit_rate_max": _max_or_none(
                    sources["handle_known_coassigned_hit_rate"]
                ),
                "initial_quality_delta_min": _min_or_none(
                    sources["initial_quality_delta_vs_source"]
                ),
                "initial_quality_delta_median": _median_or_none(
                    sources["initial_quality_delta_vs_source"]
                ),
                "initial_quality_delta_max": _max_or_none(
                    sources["initial_quality_delta_vs_source"]
                ),
                "source_signature_ids": ";".join(
                    sorted(sources["source_endpoint_signature_id"].astype(str))
                ),
            }
        )
    frame = pd.DataFrame(rows)
    frame["quality_gap_coassigned_minus_separated"] = (
        frame["coassigned_quality_median"].astype(float)
        - frame["separated_quality_median"].astype(float)
    )
    frame.loc[
        frame["separated_quality_median"].isna(),
        "quality_gap_coassigned_minus_separated",
    ] = None
    return _claim_columns(frame)


def _median_or_none(series: pd.Series) -> float | None:
    values = series.dropna()
    if values.empty:
        return None
    return float(values.astype(float).median())


def _min_or_none(series: pd.Series) -> float | None:
    values = series.dropna()
    if values.empty:
        return None
    return float(values.astype(float).min())


def _max_or_none(series: pd.Series) -> float | None:
    values = series.dropna()
    if values.empty:
        return None
    return float(values.astype(float).max())


def _role_summary(case_rows: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "endpoint_signature_count",
        "coassigned_endpoint_count",
        "separated_endpoint_count",
        "two_side_split_endpoint_count",
        "single_side_endpoint_count",
        "source_signature_count",
        "two_side_release_source_count",
        "single_side_release_source_count",
        "source_neutral_count",
        "selected_source_count",
        "robust_bridge_release_source_count",
        "handle_known_hit_rate_median",
        "initial_quality_delta_median",
    ]
    rows: list[dict[str, Any]] = []
    for role, group in case_rows.groupby("role_symbol", sort=True):
        row: dict[str, Any] = {
            "role_symbol": str(role),
            "case_count": int(len(group)),
            "pair_bridge_values": ";".join(
                f"{value:.3f}" for value in group["pair_bridge_weight"].astype(float)
            ),
        }
        for metric in metrics:
            values = group[metric].dropna().astype(float)
            row[f"{metric}_min"] = float(values.min()) if not values.empty else None
            row[f"{metric}_median"] = (
                float(values.median()) if not values.empty else None
            )
            row[f"{metric}_max"] = float(values.max()) if not values.empty else None
        rows.append(row)
    return _claim_columns(pd.DataFrame(rows))


def _neighbor_rows(case_rows: pd.DataFrame) -> pd.DataFrame:
    ordered = case_rows.sort_values("pair_bridge_weight", kind="stable").reset_index(
        drop=True
    )
    rows: list[dict[str, Any]] = []
    diff_cols = [
        "baseline_pair_coassigned_run_share",
        "endpoint_signature_count",
        "separated_endpoint_count",
        "two_side_split_endpoint_count",
        "single_side_endpoint_count",
        "source_signature_count",
        "source_neutral_count",
        "selected_source_count",
        "robust_bridge_release_source_count",
        "handle_known_hit_rate_median",
        "initial_quality_delta_median",
    ]
    for index in range(len(ordered) - 1):
        left = ordered.iloc[index]
        right = ordered.iloc[index + 1]
        row = {
            "left_case_id": str(left["case_id"]),
            "right_case_id": str(right["case_id"]),
            "left_pair_bridge_weight": float(left["pair_bridge_weight"]),
            "right_pair_bridge_weight": float(right["pair_bridge_weight"]),
            "delta_pair_bridge_weight": float(
                right["pair_bridge_weight"] - left["pair_bridge_weight"]
            ),
            "left_role": str(left["role_symbol"]),
            "right_role": str(right["role_symbol"]),
            "role_transition": f"{left['role_symbol']}_to_{right['role_symbol']}",
        }
        for col in diff_cols:
            left_value = left[col]
            right_value = right[col]
            row[f"left_{col}"] = left_value
            row[f"right_{col}"] = right_value
            if pd.isna(left_value) or pd.isna(right_value):
                row[f"delta_{col}"] = None
            else:
                row[f"delta_{col}"] = float(right_value) - float(left_value)
        rows.append(row)
    return _claim_columns(pd.DataFrame(rows))


def _signature_presence(endpoint_rows: pd.DataFrame, source_rows: pd.DataFrame) -> pd.DataFrame:
    role_order = ["R", "T", "N"]
    endpoint_records = []
    for signature_id, group in endpoint_rows.groupby("endpoint_signature_id", sort=True):
        row = {
            "endpoint_signature_id": str(signature_id),
            "bridge_signature_family": str(group["bridge_signature_family"].iloc[0]),
            "mechanism_read": str(group["mechanism_read"].iloc[0]),
            "pair_coassigned": bool(group["pair_coassigned"].astype(bool).iloc[0]),
        }
        for role in role_order:
            role_group = group[group["role_symbol"].eq(role)]
            row[f"{role}_case_count"] = int(role_group["case_id"].nunique())
            row[f"{role}_run_share_sum"] = float(
                role_group["endpoint_run_share_within_case"].sum()
            )
        source_group = source_rows[
            source_rows["source_endpoint_signature_id"].astype(str).eq(str(signature_id))
        ]
        row["source_role_symbols"] = ";".join(
            sorted(source_group["role_symbol"].dropna().astype(str).unique())
        )
        row["source_selected_case_count"] = int(
            source_group[source_group["selected_source"].astype(bool)][
                "case_id"
            ].nunique()
        )
        row["source_robust_case_count"] = int(
            source_group[source_group["robust_bridge_release_source"].astype(bool)][
                "case_id"
            ].nunique()
        )
        endpoint_records.append(row)
    return _claim_columns(pd.DataFrame(endpoint_records))


def _summary(
    *,
    output_dir: Path,
    g4_8e_dir: Path,
    case_rows: pd.DataFrame,
    source_rows: pd.DataFrame,
    role_summary: pd.DataFrame,
    neighbor_rows: pd.DataFrame,
) -> dict[str, Any]:
    role_case_counts = case_rows["role_symbol"].value_counts().to_dict()
    source_family_by_role = (
        source_rows.groupby(["role_symbol", "source_signature_family"])
        .size()
        .to_dict()
    )
    neutral_by_role = (
        source_rows.groupby("role_symbol")["source_neutral_release"]
        .sum()
        .astype(int)
        .to_dict()
    )
    selected_by_role = (
        source_rows.groupby("role_symbol")["selected_source"]
        .sum()
        .astype(int)
        .to_dict()
    )
    robust_by_role = (
        source_rows.groupby("role_symbol")["robust_bridge_release_source"]
        .sum()
        .astype(int)
        .to_dict()
    )
    interpretation = _interpretation(case_rows)
    return {
        "schema": "variable_pair_synthetic_g4_8f_centerline_signature_audit_summary.v1",
        "status": ROUTE_EXECUTION_STATUS,
        "output_dir": str(output_dir),
        "g4_8e_dir": str(g4_8e_dir),
        "centerline_case_count": int(len(case_rows)),
        "role_case_counts": role_case_counts,
        "source_family_by_role": {
            f"{role}|{family}": int(count)
            for (role, family), count in source_family_by_role.items()
        },
        "source_neutral_count_by_role": {
            str(role): int(count) for role, count in neutral_by_role.items()
        },
        "selected_source_count_by_role": {
            str(role): int(count) for role, count in selected_by_role.items()
        },
        "robust_source_count_by_role": {
            str(role): int(count) for role, count in robust_by_role.items()
        },
        "neighbor_transition_counts": neighbor_rows[
            "role_transition"
        ].value_counts().to_dict(),
        "role_summary_row_count": int(len(role_summary)),
        "neighbor_row_count": int(len(neighbor_rows)),
        "signature_read_status": interpretation["status"],
        "primary_signature_read": interpretation["primary_read"],
        "recommended_next_gate": interpretation["recommended_next_gate"],
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _interpretation(case_rows: pd.DataFrame) -> dict[str, str]:
    r_rows = case_rows[case_rows["role_symbol"].eq("R")]
    n_rows = case_rows[case_rows["role_symbol"].eq("N")]
    t_rows = case_rows[case_rows["role_symbol"].eq("T")]
    r_ok = (
        not r_rows.empty
        and r_rows["single_side_release_source_count"].astype(int).min() > 0
        and r_rows["source_neutral_count"].astype(int).eq(
            r_rows["source_signature_count"].astype(int)
        ).all()
        and r_rows["robust_bridge_release_source_count"].astype(int).eq(
            r_rows["source_signature_count"].astype(int)
        ).all()
    )
    n_ok = (
        not n_rows.empty
        and n_rows["single_side_release_source_count"].astype(int).eq(0).all()
        and n_rows["source_neutral_count"].astype(int).eq(0).all()
        and n_rows["robust_bridge_release_source_count"].astype(int).eq(0).all()
    )
    t_ok = (
        not t_rows.empty
        and t_rows["source_signature_count"].astype(int).eq(0).all()
        and t_rows["baseline_pair_coassigned_run_share"].astype(float).eq(1.0).all()
    )
    if r_ok and n_ok and t_ok:
        status = "signature_split_explains_centerline_roles"
        primary = (
            "R cells expose all 8 source signatures, including single-side "
            "sources, with source-neutral release and robust hit rate 1.0; N "
            "cells expose only 4 two-side sources, all source-nonneutral and "
            "suppressed; T cells are target-saturated and expose no source."
        )
        next_gate = (
            "Freeze this signature split as the construction-read hypothesis "
            "and test it on a fresh predeclared context before source-discovery "
            "replacement."
        )
    else:
        status = "signature_split_incomplete"
        primary = (
            "The centerline roles do not reduce cleanly to the expected "
            "source-signature split; inspect endpoint/source tables before "
            "freezing a construction rule."
        )
        next_gate = (
            "Inspect unresolved signature rows and redesign the construction "
            "audit before source-discovery replacement."
        )
    return {
        "status": status,
        "primary_read": primary,
        "recommended_next_gate": next_gate,
    }


def _write_report(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    case_rows: pd.DataFrame,
    role_summary: pd.DataFrame,
    neighbor_rows: pd.DataFrame,
    signature_presence: pd.DataFrame,
) -> None:
    lines = [
        "# Variable-Pair Synthetic G4.8F Centerline Signature Audit",
        "",
        f"- status: `{summary['status']}`",
        f"- signature_read_status: {summary['signature_read_status']}",
        f"- centerline_case_count: {summary['centerline_case_count']}",
        f"- role_case_counts: {summary['role_case_counts']}",
        f"- source_family_by_role: {summary['source_family_by_role']}",
        f"- source_neutral_count_by_role: {summary['source_neutral_count_by_role']}",
        f"- selected_source_count_by_role: {summary['selected_source_count_by_role']}",
        f"- robust_source_count_by_role: {summary['robust_source_count_by_role']}",
        f"- primary_signature_read: {summary['primary_signature_read']}",
        f"- recommended_next_gate: {summary['recommended_next_gate']}",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Centerline Case Summary",
        "",
    ]
    display_case_cols = [
        "pair_bridge_weight",
        "bridge_host_weight",
        "role_symbol",
        "baseline_pair_coassigned_run_share",
        "endpoint_signature_count",
        "two_side_split_endpoint_count",
        "single_side_endpoint_count",
        "source_signature_count",
        "source_neutral_count",
        "selected_source_count",
        "robust_bridge_release_source_count",
        "handle_known_hit_rate_median",
        "initial_quality_delta_median",
    ]
    lines.extend(_markdown_table(case_rows[display_case_cols]))
    lines.extend(["", "## Role Summary", ""])
    role_cols = [
        "role_symbol",
        "case_count",
        "pair_bridge_values",
        "source_signature_count_median",
        "single_side_release_source_count_median",
        "source_neutral_count_median",
        "selected_source_count_median",
        "robust_bridge_release_source_count_median",
        "handle_known_hit_rate_median_median",
        "initial_quality_delta_median_median",
    ]
    lines.extend(_markdown_table(role_summary[role_cols]))
    lines.extend(["", "## Neighbor Contrasts", ""])
    neighbor_cols = [
        "left_pair_bridge_weight",
        "right_pair_bridge_weight",
        "role_transition",
        "delta_baseline_pair_coassigned_run_share",
        "delta_source_signature_count",
        "delta_single_side_endpoint_count",
        "delta_source_neutral_count",
        "delta_robust_bridge_release_source_count",
        "delta_initial_quality_delta_median",
    ]
    lines.extend(_markdown_table(neighbor_rows[neighbor_cols]))
    lines.extend(["", "## Signature Presence", ""])
    presence_cols = [
        "endpoint_signature_id",
        "bridge_signature_family",
        "R_case_count",
        "T_case_count",
        "N_case_count",
        "source_selected_case_count",
        "source_robust_case_count",
    ]
    lines.extend(_markdown_table(signature_presence[presence_cols]))
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            (
                "G4.8F is a read-only signature audit. It can justify a fresh "
                "construction-read hypothesis, but it does not replace source "
                "discovery, prove a wall/pathway, compare quality/cost, or make "
                "a method claim."
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
    g4_8e_dir = Path(args.g4_8e_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = _read_inputs(g4_8e_dir)
    centerline = _centerline_cases(inputs["case_summary"])
    endpoint_rows = _endpoint_signature_rows(
        centerline=centerline,
        endpoint_summary=inputs["endpoint_summary"],
    )
    source_rows = _source_signature_rows(
        endpoint_rows=endpoint_rows,
        handle_policy_summary=inputs["handle_policy_summary"],
        selector_source_rows=inputs["selector_source_rows"],
        source_availability_rows=inputs["source_availability_rows"],
    )
    case_rows = _case_rows(
        centerline=centerline,
        endpoint_rows=endpoint_rows,
        source_rows=source_rows,
    )
    role_summary = _role_summary(case_rows)
    neighbor_rows = _neighbor_rows(case_rows)
    signature_presence = _signature_presence(endpoint_rows, source_rows)
    _write_csv(case_rows, output_dir / CENTERLINE_CASE_SUMMARY_CSV)
    _write_csv(endpoint_rows, output_dir / ENDPOINT_SIGNATURES_CSV)
    _write_csv(source_rows, output_dir / SOURCE_SIGNATURES_CSV)
    _write_csv(role_summary, output_dir / ROLE_FAMILY_SUMMARY_CSV)
    _write_csv(neighbor_rows, output_dir / NEIGHBOR_CONTRAST_CSV)
    _write_csv(signature_presence, output_dir / SIGNATURE_PRESENCE_MATRIX_CSV)
    summary = _summary(
        output_dir=output_dir,
        g4_8e_dir=g4_8e_dir,
        case_rows=case_rows,
        source_rows=source_rows,
        role_summary=role_summary,
        neighbor_rows=neighbor_rows,
    )
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    config = {
        "schema": "variable_pair_synthetic_g4_8f_centerline_signature_audit_config.v1",
        "g4_8e_dir": str(g4_8e_dir),
        "output_dir": str(output_dir),
        "handle_policy": HANDLE_POLICY,
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
        role_summary=role_summary,
        neighbor_rows=neighbor_rows,
        signature_presence=signature_presence,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g4-8e-dir", type=Path, default=DEFAULT_G4_8E_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    summary = analyze(parse_args())
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
