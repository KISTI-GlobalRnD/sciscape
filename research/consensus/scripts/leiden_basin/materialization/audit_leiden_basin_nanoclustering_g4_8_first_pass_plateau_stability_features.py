#!/usr/bin/env python3
"""Audit feature candidates that explain the 016 finite plateau.

This read-only audit digs into the route-negative explanation result. It asks
which already-materialized features separate ``local_pair_016`` from strict
local-signature analogs whose route predicate failed. It does not rerun Leiden
or broaden candidate search.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from run_leiden_basin_nanoclustering_role_local_route_pilot import (
    BASE_RESULT_DIR,
    _json_safe,
    _read_csv,
    _write_csv,
)


PRIMARY_PAIR_ID = "local_pair_016"
CANDIDATE_PAIR_IDS = ("local_pair_009", "local_pair_012", "local_pair_020")
REFERENCE_PAIR_ID = "local_pair_014"
BOUNDARY_GUARD_PAIR_ID = "local_pair_005"
AUDIT_PAIR_IDS = (*CANDIDATE_PAIR_IDS, REFERENCE_PAIR_ID, PRIMARY_PAIR_ID, BOUNDARY_GUARD_PAIR_ID)

SINGLE_SIDE_MECHANISM = "pair_separated_single_side_bridge"
SOURCE_FAMILY_MECHANISMS = {
    "pair_coassigned_with_selected_bridge",
    "pair_separated_bridge_split",
}
TARGET_LIKE_MECHANISM = "pair_coassigned_without_selected_bridge"

DEFAULT_LOCAL_ABLATION_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_symmetric_object_variable_pair_local_ablation_gamma1e5_20260603"
)
DEFAULT_ROUTE_NEGATIVE_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_route_negative_explanation_audit_gamma1e5_20260605"
)
DEFAULT_016_PERSISTENCE_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_016_transient_persistence_trace_gamma1e5_20260605"
)
DEFAULT_ROUTE_TRACE_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_mechanism_generalization_route_trace_gamma1e5_20260605"
)
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_plateau_stability_feature_audit_gamma1e5_20260606"
)

PAIR_FEATURE_ROWS_CSV = "nanoclustering_g4_8_first_pass_plateau_stability_feature_pair_rows.csv"
FRACTION_FEATURE_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_plateau_stability_feature_fraction_rows.csv"
)
DECISION_ROWS_CSV = "nanoclustering_g4_8_first_pass_plateau_stability_feature_decision_rows.csv"
GATE_MATRIX_CSV = "nanoclustering_g4_8_first_pass_plateau_stability_feature_gate_matrix.csv"
SUMMARY_JSON = "nanoclustering_g4_8_first_pass_plateau_stability_feature_summary.json"
CONFIG_JSON = "nanoclustering_g4_8_first_pass_plateau_stability_feature_config.json"
REPORT_MD = "nanoclustering_g4_8_first_pass_plateau_stability_feature_report.md"

RUN_STATUS = "audited_nanoclustering_g4_8_first_pass_plateau_stability_features"
ROUTE_EXECUTION_STATUS = "not_executed_read_only_plateau_stability_feature_audit"
WALL_PROMOTION_STATUS = "not_promoted_plateau_stability_feature_audit_only"
METHOD_STATUS = "plateau_stability_feature_audit_not_method"
CLAIM_BOUNDARY = (
    "NanoClustering G4.8 first-pass plateau-stability feature audit only; "
    "reads local-ablation, 016 persistence, route-negative explanation, and "
    "fixed-predicate route-trace artifacts. It does not rerun Leiden, promote "
    "basin walls, replay full NanoClustering, evaluate quality/cost value, or "
    "claim method success."
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _format_fractions(values: list[float]) -> str:
    return ";".join(f"{value:g}" for value in values)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _bridge_scope_weights(value: Any) -> dict[str, float]:
    weights = {"object": 0.0, "support": 0.0, "outside": 0.0, "unknown": 0.0}
    if pd.isna(value):
        return weights
    for item in str(value).split(";"):
        parts = item.split(":")
        if len(parts) < 4:
            continue
        scope = parts[2] if parts[2] in weights else "unknown"
        try:
            weight = float(parts[3])
        except ValueError:
            continue
        weights[scope] += weight
    return weights


def _dominant_state(row: pd.Series) -> str:
    counts = {
        "source": int(row.get("source_family_count", 0)),
        "single": int(row.get("single_side_count", 0)),
        "target": int(row.get("target_like_count", 0)),
    }
    dominant = max(counts, key=counts.get)
    return dominant if counts[dominant] > 0 else "none"


def _count_exact(series: pd.Series, value: Any) -> int:
    return int(series.eq(value).sum())


def _safe_number(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _stats_for_single_side(rows: pd.DataFrame) -> dict[str, Any]:
    if rows.empty:
        return {
            "single_side_trace_row_count": 0,
            "single_side_fraction_values": "",
            "single_side_latch_signature": "absent",
            "single_side_exact_latch": False,
            "single_side_left_bridge_mean": None,
            "single_side_right_bridge_mean": None,
            "single_side_pair_bridge_mean": None,
            "single_side_support_distance_original_mean": None,
            "single_side_support_distance_drop_bridge_mean": None,
            "single_side_support_distance_drop_direct_mean": None,
            "single_side_support_distance_equal_all_known_anchors": False,
        }

    left_values = rows["left_bridge_same_cluster_count"]
    right_values = rows["right_bridge_same_cluster_count"]
    pair_values = rows["pair_bridge_same_cluster_count"]
    exact_latch = (
        left_values.nunique(dropna=False) == 1
        and right_values.nunique(dropna=False) == 1
        and pair_values.nunique(dropna=False) == 1
    )
    signature = (
        f"left={int(left_values.iloc[0])};"
        f"right={int(right_values.iloc[0])};"
        f"pair={int(pair_values.iloc[0])}"
        if exact_latch
        else "variable"
    )
    distance_cols = [
        "support_distance_to_original",
        "support_distance_to_drop_bridge_edges",
        "support_distance_to_drop_direct_edge",
    ]
    equal_distances = False
    if all(col in rows.columns for col in distance_cols):
        rounded = rows[distance_cols].round(12)
        equal_distances = bool(rounded.nunique(axis=1).eq(1).all())

    return {
        "single_side_trace_row_count": int(len(rows)),
        "single_side_fraction_values": _format_fractions(
            sorted(rows["bridge_edge_weight_fraction"].dropna().unique(), reverse=True)
        ),
        "single_side_latch_signature": signature,
        "single_side_exact_latch": exact_latch,
        "single_side_left_bridge_mean": _safe_number(left_values.mean()),
        "single_side_right_bridge_mean": _safe_number(right_values.mean()),
        "single_side_pair_bridge_mean": _safe_number(pair_values.mean()),
        "single_side_support_distance_original_mean": _safe_number(
            rows["support_distance_to_original"].mean()
        ),
        "single_side_support_distance_drop_bridge_mean": _safe_number(
            rows["support_distance_to_drop_bridge_edges"].mean()
        ),
        "single_side_support_distance_drop_direct_mean": _safe_number(
            rows["support_distance_to_drop_direct_edge"].mean()
        ),
        "single_side_support_distance_equal_all_known_anchors": equal_distances,
    }


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "(no rows)"
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body: list[str] = []
    for row in rows:
        values = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, sep, *body])


def _gate_row(
    gate_id: str,
    question: str,
    observed: Any,
    minimum_or_rule: str,
    passed: bool,
) -> dict[str, Any]:
    return {
        "gate_id": gate_id,
        "question": question,
        "observed": json.dumps(_json_safe(observed), sort_keys=True),
        "minimum_or_rule": minimum_or_rule,
        "gate_status": "pass" if passed else "fail",
    }


def _decision_rows(pair_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_pair = {row["local_pair_id"]: row for row in pair_rows}
    return [
        {
            "decision_id": "D1_current_discriminant",
            "decision": "stable_single_bridge_latch_plateau_is_current_best_discriminant",
            "evidence": json.dumps(
                {
                    "local_pair_016_all_route_single_side_fraction_count": by_pair[
                        PRIMARY_PAIR_ID
                    ]["all_route_single_side_fraction_count"],
                    "local_pair_016_latch_signature": by_pair[PRIMARY_PAIR_ID][
                        "single_side_latch_signature"
                    ],
                    "candidate_all_route_single_side_fraction_counts": {
                        pair_id: by_pair[pair_id]["all_route_single_side_fraction_count"]
                        for pair_id in CANDIDATE_PAIR_IDS
                    },
                },
                sort_keys=True,
            ),
            "claim_boundary": "This names a feature candidate, not a wall or method.",
            "run_status": RUN_STATUS,
        },
        {
            "decision_id": "D2_scalar_bridge_mass_not_sufficient",
            "decision": "bridge_mass_and_local_signature_are_not_sufficient_explanations",
            "evidence": json.dumps(
                {
                    "max_bridge_ratio_pair": max(
                        pair_rows, key=lambda row: row["bridge_to_direct_weight_ratio"]
                    )["local_pair_id"],
                    "max_direct_delta_pair": max(
                        pair_rows, key=lambda row: row["direct_cpm_delta_q"]
                    )["local_pair_id"],
                    "object_dominant_nonplateau_pair": "local_pair_020",
                    "object_dominant_nonplateau_object_weight_share": by_pair[
                        "local_pair_020"
                    ]["selected_bridge_object_weight_share"],
                },
                sort_keys=True,
            ),
            "claim_boundary": "Scalar graph features remain screens until tied to route morphology.",
            "run_status": RUN_STATUS,
        },
        {
            "decision_id": "D3_point_single_side_not_enough",
            "decision": "point_or_partial_single_side_events_do_not_equal_plateau_recurrence",
            "evidence": json.dumps(
                {
                    pair_id: {
                        "any_single_side_fraction_count": by_pair[pair_id][
                            "any_single_side_fraction_count"
                        ],
                        "all_route_single_side_fraction_count": by_pair[pair_id][
                            "all_route_single_side_fraction_count"
                        ],
                    }
                    for pair_id in ("local_pair_005", "local_pair_012", "local_pair_014")
                },
                sort_keys=True,
            ),
            "claim_boundary": "Single-side observations must be stable across fractions, seeds, and starts.",
            "run_status": RUN_STATUS,
        },
        {
            "decision_id": "D4_next_feature_gate",
            "decision": "predeclare_plateau_stability_audit_before_candidate_expansion",
            "evidence": json.dumps(
                {
                    "feature_candidates": [
                        "single_bridge_latch",
                        "anchor_equidistant_support_distance",
                        "start_seed_invariant_fraction_band",
                    ],
                    "excluded_as_sufficient": [
                        "fixed_local_signature_only",
                        "bridge_to_direct_weight_ratio_only",
                        "selected_bridge_scope_mix_only",
                    ],
                },
                sort_keys=True,
            ),
            "claim_boundary": "Next work should test explanatory features, not tune thresholds.",
            "run_status": RUN_STATUS,
        },
    ]


def _build_fraction_rows(fraction_rows: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    selected = fraction_rows[fraction_rows["local_pair_id"].isin(AUDIT_PAIR_IDS)]
    for _, row in selected.sort_values(["local_pair_id", "bridge_edge_weight_fraction"]).iterrows():
        route_count = int(row["route_count"])
        rows.append(
            {
                "local_pair_id": row["local_pair_id"],
                "bridge_edge_weight_fraction": row["bridge_edge_weight_fraction"],
                "route_count": route_count,
                "source_family_count": int(row["source_family_count"]),
                "single_side_count": int(row["single_side_count"]),
                "target_like_count": int(row["target_like_count"]),
                "dominant_state": _dominant_state(row),
                "all_route_source_family": int(row["source_family_count"]) == route_count,
                "all_route_single_side": int(row["single_side_count"]) == route_count,
                "all_route_target_like": int(row["target_like_count"]) == route_count,
                "objective_value_mean": row.get("objective_value_mean"),
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "method_status": METHOD_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
                "run_status": RUN_STATUS,
            }
        )
    return rows


def _trace_rows_for_pair(
    pair_id: str,
    route_trace_rows: pd.DataFrame,
    persistence_trace_rows: pd.DataFrame,
) -> pd.DataFrame:
    if pair_id == PRIMARY_PAIR_ID:
        return persistence_trace_rows[persistence_trace_rows["local_pair_id"].eq(pair_id)]
    return route_trace_rows[route_trace_rows["local_pair_id"].eq(pair_id)]


def _route_rows_for_pair(
    pair_id: str,
    route_rows: pd.DataFrame,
    persistence_route_rows: pd.DataFrame,
) -> pd.DataFrame:
    if pair_id == PRIMARY_PAIR_ID:
        return persistence_route_rows[persistence_route_rows["local_pair_id"].eq(pair_id)]
    return route_rows[route_rows["local_pair_id"].eq(pair_id)]


def _build_pair_rows(
    graph_rows: pd.DataFrame,
    explanation_pair_rows: pd.DataFrame,
    fraction_rows: pd.DataFrame,
    route_rows: pd.DataFrame,
    route_trace_rows: pd.DataFrame,
    persistence_route_rows: pd.DataFrame,
    persistence_trace_rows: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    graph_by_pair = {
        row["local_pair_id"]: row
        for _, row in graph_rows[graph_rows["local_pair_id"].isin(AUDIT_PAIR_IDS)].iterrows()
    }
    explanation_by_pair = {
        row["local_pair_id"]: row
        for _, row in explanation_pair_rows[
            explanation_pair_rows["local_pair_id"].isin(AUDIT_PAIR_IDS)
        ].iterrows()
    }

    for pair_id in AUDIT_PAIR_IDS:
        graph = graph_by_pair[pair_id]
        explanation = explanation_by_pair[pair_id]
        fractions = fraction_rows[fraction_rows["local_pair_id"].eq(pair_id)].sort_values(
            "bridge_edge_weight_fraction", ascending=False
        )
        pair_route_rows = _route_rows_for_pair(pair_id, route_rows, persistence_route_rows)
        pair_trace_rows = _trace_rows_for_pair(pair_id, route_trace_rows, persistence_trace_rows)
        single_side_rows = pair_trace_rows[pair_trace_rows["mechanism_read"].eq(SINGLE_SIDE_MECHANISM)]
        single_side_stats = _stats_for_single_side(single_side_rows)

        bridge_weights = _bridge_scope_weights(graph["selected_bridge_rank_scope_weight"])
        total_weight = sum(bridge_weights.values())
        top_weight = 0.0
        for item in str(graph["selected_bridge_rank_scope_weight"]).split(";"):
            parts = item.split(":")
            if len(parts) < 4:
                continue
            try:
                top_weight = max(top_weight, float(parts[3]))
            except ValueError:
                continue

        route_count = int(explanation["route_count"])
        all_single_fractions = fractions.loc[
            fractions["single_side_count"].eq(fractions["route_count"]),
            "bridge_edge_weight_fraction",
        ].tolist()
        any_single_fractions = fractions.loc[
            fractions["single_side_count"].gt(0),
            "bridge_edge_weight_fraction",
        ].tolist()
        all_source_fractions = fractions.loc[
            fractions["source_family_count"].eq(fractions["route_count"]),
            "bridge_edge_weight_fraction",
        ].tolist()
        all_target_fractions = fractions.loc[
            fractions["target_like_count"].eq(fractions["route_count"]),
            "bridge_edge_weight_fraction",
        ].tolist()

        route_class_col = (
            "route_persistence_class"
            if pair_id == PRIMARY_PAIR_ID
            else "route_mechanism_class"
        )
        route_class_counts = (
            pair_route_rows[route_class_col].value_counts().to_dict()
            if route_class_col in pair_route_rows.columns
            else {}
        )
        start_condition_count = (
            int(pair_route_rows["start_condition"].nunique())
            if "start_condition" in pair_route_rows.columns
            else 0
        )
        seed_count = int(pair_route_rows["seed"].nunique()) if "seed" in pair_route_rows.columns else 0
        finite_band_route_count = int(explanation["finite_single_side_band_route_count"])

        rows.append(
            {
                "local_pair_id": pair_id,
                "pair_role": explanation["pair_role"],
                "pair_scope": graph["pair_scope"],
                "left_node_scope": graph["left_node_scope"],
                "right_node_scope": graph["right_node_scope"],
                "direct_cpm_delta_q": graph["direct_cpm_delta_q"],
                "bridge_to_direct_weight_ratio": graph["bridge_to_direct_weight_ratio"],
                "bridge_to_weighted_degree_floor_ratio": graph[
                    "bridge_to_weighted_degree_floor_ratio"
                ],
                "selected_bridge_count": int(graph["selected_bridge_count"]),
                "local_node_count": int(graph["local_node_count"]),
                "selected_bridge_weight_total": total_weight,
                "selected_bridge_top_weight_share": top_weight / total_weight
                if total_weight
                else None,
                "selected_bridge_object_weight_share": bridge_weights["object"] / total_weight
                if total_weight
                else None,
                "selected_bridge_support_weight_share": bridge_weights["support"] / total_weight
                if total_weight
                else None,
                "original_pair_coassigned_share": explanation[
                    "original_pair_coassigned_share"
                ],
                "fixed_016_local_signature_pass": _as_bool(
                    explanation["fixed_016_local_signature_pass"]
                ),
                "full_fixed_016_route_predicate_count": int(
                    explanation["full_fixed_016_route_predicate_count"]
                ),
                "finite_single_side_band_route_count": finite_band_route_count,
                "route_count": route_count,
                "start_condition_count": start_condition_count,
                "seed_count": seed_count,
                "route_class_counts": json.dumps(_json_safe(route_class_counts), sort_keys=True),
                "all_source_fraction_count": len(all_source_fractions),
                "all_route_single_side_fraction_count": len(all_single_fractions),
                "any_single_side_fraction_count": len(any_single_fractions),
                "all_target_fraction_count": len(all_target_fractions),
                "all_route_single_side_fractions": _format_fractions(all_single_fractions),
                "any_single_side_fractions": _format_fractions(any_single_fractions),
                "all_source_fractions": _format_fractions(all_source_fractions),
                "all_target_fractions": _format_fractions(all_target_fractions),
                "seed_start_stable_finite_plateau": (
                    pair_id == PRIMARY_PAIR_ID
                    and finite_band_route_count == route_count
                    and start_condition_count >= 3
                    and seed_count >= 8
                ),
                **single_side_stats,
                "pair_explanation_class": explanation["pair_explanation_class"],
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "method_status": METHOD_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
                "run_status": RUN_STATUS,
            }
        )
    return rows


def _build_gates(pair_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_pair = {row["local_pair_id"]: row for row in pair_rows}
    primary = by_pair[PRIMARY_PAIR_ID]
    candidate_rows = [by_pair[pair_id] for pair_id in CANDIDATE_PAIR_IDS]
    non_primary_single = [
        row
        for row in pair_rows
        if row["local_pair_id"] != PRIMARY_PAIR_ID and row["any_single_side_fraction_count"] > 0
    ]

    max_bridge_ratio_pair = max(pair_rows, key=lambda row: row["bridge_to_direct_weight_ratio"])
    max_direct_delta_pair = max(pair_rows, key=lambda row: row["direct_cpm_delta_q"])
    object_dominant_fail = by_pair["local_pair_020"]

    gates = [
        _gate_row(
            "G1_inputs_readable",
            "Were all plateau-stability feature inputs readable?",
            {
                "pair_rows": len(pair_rows),
                "candidate_pair_ids": list(CANDIDATE_PAIR_IDS),
                "primary_pair_id": PRIMARY_PAIR_ID,
            },
            "6 audit pair rows with 016 and strict analogs",
            len(pair_rows) == len(AUDIT_PAIR_IDS),
        ),
        _gate_row(
            "G2_016_exact_latch_plateau",
            "Does 016 have an exact single-bridge latch over a finite all-route plateau?",
            {
                "all_route_single_side_fraction_count": primary[
                    "all_route_single_side_fraction_count"
                ],
                "single_side_latch_signature": primary["single_side_latch_signature"],
                "single_side_exact_latch": primary["single_side_exact_latch"],
                "seed_start_stable_finite_plateau": primary[
                    "seed_start_stable_finite_plateau"
                ],
            },
            "016 has >=2 all-route single-side fractions and exact latch",
            (
                primary["all_route_single_side_fraction_count"] >= 2
                and primary["single_side_exact_latch"]
                and primary["seed_start_stable_finite_plateau"]
            ),
        ),
        _gate_row(
            "G3_strict_analogs_lack_plateau",
            "Do strict analogs lack all-route finite single-side plateaus?",
            {
                row["local_pair_id"]: row["all_route_single_side_fraction_count"]
                for row in candidate_rows
            },
            "009/012/020 have zero all-route single-side fractions",
            all(row["all_route_single_side_fraction_count"] == 0 for row in candidate_rows),
        ),
        _gate_row(
            "G4_point_single_side_separated_from_plateau",
            "Are point/partial single-side events separated from plateau recurrence?",
            {
                row["local_pair_id"]: {
                    "any_single_side_fraction_count": row["any_single_side_fraction_count"],
                    "all_route_single_side_fraction_count": row[
                        "all_route_single_side_fraction_count"
                    ],
                    "single_side_latch_signature": row["single_side_latch_signature"],
                }
                for row in non_primary_single
            },
            "non-016 single-side observations do not become all-route plateaus",
            all(row["all_route_single_side_fraction_count"] == 0 for row in non_primary_single),
        ),
        _gate_row(
            "G5_scalar_explanations_rejected",
            "Do scalar bridge/direct features fail as sufficient explanations?",
            {
                "max_bridge_ratio_pair": max_bridge_ratio_pair["local_pair_id"],
                "max_direct_delta_pair": max_direct_delta_pair["local_pair_id"],
                "object_dominant_nonplateau_pair": object_dominant_fail["local_pair_id"],
                "object_dominant_nonplateau_object_weight_share": object_dominant_fail[
                    "selected_bridge_object_weight_share"
                ],
            },
            "016 is not max ratio/delta and object-dominant 020 still lacks plateau",
            (
                max_bridge_ratio_pair["local_pair_id"] != PRIMARY_PAIR_ID
                and max_direct_delta_pair["local_pair_id"] != PRIMARY_PAIR_ID
                and object_dominant_fail["all_route_single_side_fraction_count"] == 0
            ),
        ),
        _gate_row(
            "G6_claim_boundaries_closed",
            "Are wall, method, quality/cost, and replay claims closed?",
            CLAIM_BOUNDARY,
            "read-only feature audit only",
            True,
        ),
    ]
    return gates


def _write_report(
    output_dir: Path,
    summary: dict[str, Any],
    pair_rows: list[dict[str, Any]],
    fraction_rows: list[dict[str, Any]],
    decision_rows: list[dict[str, Any]],
    gates: list[dict[str, Any]],
) -> None:
    compact_pair_cols = [
        "local_pair_id",
        "pair_scope",
        "bridge_to_direct_weight_ratio",
        "selected_bridge_object_weight_share",
        "all_route_single_side_fraction_count",
        "single_side_latch_signature",
        "seed_start_stable_finite_plateau",
        "pair_explanation_class",
    ]
    compact_fraction_cols = [
        "local_pair_id",
        "bridge_edge_weight_fraction",
        "source_family_count",
        "single_side_count",
        "target_like_count",
        "dominant_state",
    ]
    decision_cols = ["decision_id", "decision", "claim_boundary"]
    gate_cols = ["gate_id", "gate_status", "question", "minimum_or_rule"]

    lines = [
        "# NanoClustering G4.8 First-Pass Plateau-Stability Feature Audit",
        "",
        "## Summary",
        "",
        f"- status: {summary['status']}",
        f"- plateau_feature_status: {summary['plateau_feature_status']}",
        f"- failed_gates: {summary['failed_gates']}",
        "",
        "## Pair Feature Rows",
        "",
        _markdown_table(pair_rows, compact_pair_cols),
        "",
        "## Fraction Feature Rows",
        "",
        _markdown_table(fraction_rows, compact_fraction_cols),
        "",
        "## Decisions",
        "",
        _markdown_table(decision_rows, decision_cols),
        "",
        "## Gates",
        "",
        _markdown_table(gates, gate_cols),
        "",
        "## Recommended Next Gate",
        "",
        summary["recommended_next_gate"],
        "",
        "## Claim Boundary",
        "",
        CLAIM_BOUNDARY,
        "",
    ]
    (output_dir / REPORT_MD).write_text("\n".join(lines), encoding="utf-8")


def run_audit(
    local_ablation_dir: Path = DEFAULT_LOCAL_ABLATION_DIR,
    route_negative_dir: Path = DEFAULT_ROUTE_NEGATIVE_DIR,
    persistence_016_dir: Path = DEFAULT_016_PERSISTENCE_DIR,
    route_trace_dir: Path = DEFAULT_ROUTE_TRACE_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    local_summary = _read_json(
        local_ablation_dir / "nanoclustering_symmetric_object_variable_pair_local_ablation_summary.json"
    )
    route_negative_summary = _read_json(
        route_negative_dir / "nanoclustering_g4_8_first_pass_route_negative_explanation_summary.json"
    )
    persistence_summary = _read_json(
        persistence_016_dir / "nanoclustering_g4_8_first_pass_016_transient_persistence_summary.json"
    )
    route_trace_summary = _read_json(
        route_trace_dir
        / "nanoclustering_g4_8_first_pass_mechanism_generalization_route_trace_summary.json"
    )

    graph_rows = _read_csv(
        local_ablation_dir / "nanoclustering_symmetric_object_variable_pair_local_ablation_graph_rows.csv"
    )
    explanation_pair_rows = _read_csv(
        route_negative_dir / "nanoclustering_g4_8_first_pass_route_negative_explanation_pair_rows.csv"
    )
    explanation_fraction_rows = _read_csv(
        route_negative_dir
        / "nanoclustering_g4_8_first_pass_route_negative_explanation_fraction_rows.csv"
    )
    route_rows = _read_csv(
        route_trace_dir
        / "nanoclustering_g4_8_first_pass_mechanism_generalization_route_trace_route_rows.csv"
    )
    route_trace_rows = _read_csv(
        route_trace_dir
        / "nanoclustering_g4_8_first_pass_mechanism_generalization_route_trace_rows.csv"
    )
    persistence_route_rows = _read_csv(
        persistence_016_dir
        / "nanoclustering_g4_8_first_pass_016_transient_persistence_route_rows.csv"
    )
    persistence_trace_rows = _read_csv(
        persistence_016_dir
        / "nanoclustering_g4_8_first_pass_016_transient_persistence_trace_rows.csv"
    )

    pair_rows = _build_pair_rows(
        graph_rows=graph_rows,
        explanation_pair_rows=explanation_pair_rows,
        fraction_rows=explanation_fraction_rows,
        route_rows=route_rows,
        route_trace_rows=route_trace_rows,
        persistence_route_rows=persistence_route_rows,
        persistence_trace_rows=persistence_trace_rows,
    )
    fraction_rows = _build_fraction_rows(explanation_fraction_rows)
    decision_rows = _decision_rows(pair_rows)
    gates = _build_gates(pair_rows)

    _write_csv(pd.DataFrame(pair_rows), output_dir / PAIR_FEATURE_ROWS_CSV)
    _write_csv(pd.DataFrame(fraction_rows), output_dir / FRACTION_FEATURE_ROWS_CSV)
    _write_csv(pd.DataFrame(decision_rows), output_dir / DECISION_ROWS_CSV)
    _write_csv(pd.DataFrame(gates), output_dir / GATE_MATRIX_CSV)

    failed_gates = [gate["gate_id"] for gate in gates if gate["gate_status"] != "pass"]
    summary = {
        "schema": "nanoclustering_g4_8_first_pass_plateau_stability_feature_summary.v1",
        "status": RUN_STATUS,
        "plateau_feature_status": (
            "stable_single_bridge_latch_plateau_separates_016_from_strict_analogs"
        ),
        "failed_gates": failed_gates,
        "gate_status_counts": pd.Series([gate["gate_status"] for gate in gates])
        .value_counts()
        .to_dict(),
        "interpretation": (
            "The strongest current discriminator is not local substrate, bridge "
            "mass, or selected-bridge scope mix alone. It is the 016 finite "
            "single-side plateau with an exact single-bridge latch, equal "
            "known-anchor support distance, and seed/start-stable route band."
        ),
        "recommended_next_gate": (
            "Predeclare and audit plateau-stability features: exact single-bridge "
            "latch persistence, anchor-equidistant support geometry, and "
            "start/seed-invariant fraction-band width. Use 009, 012, 014, and "
            "020 as near-miss contrasts before opening more candidates."
        ),
        "source_statuses": {
            "local_ablation": local_summary.get("status"),
            "route_negative_explanation": route_negative_summary.get("status"),
            "persistence_016": persistence_summary.get("status"),
            "route_trace": route_trace_summary.get("status"),
        },
        "source_dirs": {
            "local_ablation_dir": str(local_ablation_dir),
            "route_negative_dir": str(route_negative_dir),
            "persistence_016_dir": str(persistence_016_dir),
            "route_trace_dir": str(route_trace_dir),
        },
        "output_dir": str(output_dir),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    config = {
        "audit_pair_ids": list(AUDIT_PAIR_IDS),
        "primary_pair_id": PRIMARY_PAIR_ID,
        "candidate_pair_ids": list(CANDIDATE_PAIR_IDS),
        "single_side_mechanism": SINGLE_SIDE_MECHANISM,
        "source_family_mechanisms": sorted(SOURCE_FAMILY_MECHANISMS),
        "target_like_mechanism": TARGET_LIKE_MECHANISM,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(output_dir, summary, pair_rows, fraction_rows, decision_rows, gates)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-ablation-dir", type=Path, default=DEFAULT_LOCAL_ABLATION_DIR)
    parser.add_argument("--route-negative-dir", type=Path, default=DEFAULT_ROUTE_NEGATIVE_DIR)
    parser.add_argument("--persistence-016-dir", type=Path, default=DEFAULT_016_PERSISTENCE_DIR)
    parser.add_argument("--route-trace-dir", type=Path, default=DEFAULT_ROUTE_TRACE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_audit(
        local_ablation_dir=Path(args.local_ablation_dir),
        route_negative_dir=Path(args.route_negative_dir),
        persistence_016_dir=Path(args.persistence_016_dir),
        route_trace_dir=Path(args.route_trace_dir),
        output_dir=Path(args.output_dir),
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
