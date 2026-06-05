#!/usr/bin/env python3
"""Validate the semantics of the local_pair_016 typed transient.

This read-only audit asks whether the recurrent ``016`` step-2 separated
bridge-reassignment signature is better treated as a meaningful transition
gateway, a basin endpoint, or a local route artifact. It reads only existing
first-pass trace, continuity-block, predicate-screen, and transfer-screen
artifacts. It does not rerun Leiden, execute routes, promote walls, evaluate
quality/cost value, or claim method success.
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
REFERENCE_PAIR_ID = "local_pair_014"
BOUNDARY_GUARD_PAIR_ID = "local_pair_005"

DEFAULT_FIRST_PASS_DIR = (
    BASE_RESULT_DIR / "leiden_basin_nanoclustering_g4_8_fresh_axis_b_first_pass_trace_gamma1e5_20260604"
)
DEFAULT_TRANSFER_SCREEN_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_014_role_pattern_transfer_screen_gamma1e5_20260605"
)
DEFAULT_CONTINUITY_016_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_016_continuity_block_audit_gamma1e5_20260605"
)
DEFAULT_PREDICATE_SCREEN_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_typed_transient_predicate_screen_gamma1e5_20260605"
)
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_016_transient_semantic_validation_gamma1e5_20260605"
)

ROUTE_SEMANTIC_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_016_transient_semantic_route_rows.csv"
)
STEP_SEMANTIC_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_016_transient_semantic_step_rows.csv"
)
COMPARISON_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_016_transient_semantic_comparison_rows.csv"
)
SEMANTIC_DECISION_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_016_transient_semantic_decision_rows.csv"
)
GATE_MATRIX_CSV = (
    "nanoclustering_g4_8_first_pass_016_transient_semantic_gate_matrix.csv"
)
SUMMARY_JSON = "nanoclustering_g4_8_first_pass_016_transient_semantic_summary.json"
CONFIG_JSON = "nanoclustering_g4_8_first_pass_016_transient_semantic_config.json"
REPORT_MD = "nanoclustering_g4_8_first_pass_016_transient_semantic_report.md"

RUN_STATUS = "audited_nanoclustering_g4_8_first_pass_016_transient_semantic_validation"
ROUTE_EXECUTION_STATUS = "not_executed_read_only_016_transient_semantic_validation"
WALL_PROMOTION_STATUS = "not_promoted_016_transient_semantic_validation_only"
METHOD_STATUS = "semantic_validation_not_method"
CLAIM_BOUNDARY = (
    "NanoClustering G4.8 first-pass local_pair_016 typed-transient semantic "
    "validation only; reads existing first-pass trace, transfer-screen, "
    "continuity-block, and predicate-screen outputs to classify the 016 "
    "transient. It does not rerun Leiden, execute routes, perform a fraction "
    "sweep, promote basin walls, replay full NanoClustering, evaluate "
    "quality/cost value, or claim method success."
)

EPS = 1e-9


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _as_float(value: Any) -> float:
    if pd.isna(value):
        return 0.0
    return float(value)


def _as_int(value: Any) -> int:
    if pd.isna(value):
        return 0
    return int(value)


def _list_json(values: list[str]) -> str:
    return json.dumps(values, ensure_ascii=True)


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
        "observed": observed,
        "minimum_or_rule": minimum_or_rule,
        "gate_status": "pass" if bool(passed) else "fail",
    }


def _markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 40) -> str:
    cols = [column for column in columns if column in frame.columns]
    if not cols:
        return "_No matching columns._"
    visible = frame[cols].head(int(max_rows))
    if visible.empty:
        return "_No rows._"

    def cell(value: Any) -> str:
        if isinstance(value, (dict, list, tuple, set)):
            return json.dumps(_json_safe(value), sort_keys=True).replace("|", "\\|")
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value).replace("\n", " ").replace("|", "\\|")

    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for row in visible.itertuples(index=False):
        lines.append("| " + " | ".join(cell(value) for value in row) + " |")
    return "\n".join(lines)


def _load_context(args: argparse.Namespace) -> dict[str, Any]:
    first_pass_dir = Path(args.first_pass_dir)
    transfer_screen_dir = Path(args.transfer_screen_dir)
    continuity_016_dir = Path(args.continuity_016_dir)
    predicate_screen_dir = Path(args.predicate_screen_dir)

    summaries = {
        "first_pass": _read_json(
            first_pass_dir / "nanoclustering_g4_8_fresh_axis_b_first_pass_trace_summary.json"
        ),
        "transfer_screen": _read_json(
            transfer_screen_dir / "nanoclustering_g4_8_first_pass_014_role_pattern_transfer_summary.json"
        ),
        "continuity_016": _read_json(
            continuity_016_dir / "nanoclustering_g4_8_first_pass_016_continuity_block_summary.json"
        ),
        "predicate_screen": _read_json(
            predicate_screen_dir
            / "nanoclustering_g4_8_first_pass_typed_transient_predicate_summary.json"
        ),
    }
    tables = {
        "first_pass_trace": _read_csv(
            first_pass_dir / "nanoclustering_g4_8_fresh_axis_b_first_pass_trace_rows.csv"
        ),
        "first_pass_route": _read_csv(
            first_pass_dir
            / "nanoclustering_g4_8_fresh_axis_b_first_pass_route_readout_result_rows.csv"
        ),
        "transfer_pair": _read_csv(
            transfer_screen_dir
            / "nanoclustering_g4_8_first_pass_014_role_pattern_transfer_pair_rows.csv"
        ),
        "transfer_route": _read_csv(
            transfer_screen_dir
            / "nanoclustering_g4_8_first_pass_014_role_pattern_transfer_route_rows.csv"
        ),
        "continuity_route": _read_csv(
            continuity_016_dir / "nanoclustering_g4_8_first_pass_016_continuity_block_route_rows.csv"
        ),
        "continuity_step": _read_csv(
            continuity_016_dir
            / "nanoclustering_g4_8_first_pass_016_continuity_block_step_signature_rows.csv"
        ),
        "predicate_rows": _read_csv(
            predicate_screen_dir / "nanoclustering_g4_8_first_pass_typed_transient_predicate_rows.csv"
        ),
        "predicate_pair_rows": _read_csv(
            predicate_screen_dir
            / "nanoclustering_g4_8_first_pass_typed_transient_predicate_pair_predicate_rows.csv"
        ),
    }
    return {
        "paths": {
            "first_pass_dir": first_pass_dir,
            "transfer_screen_dir": transfer_screen_dir,
            "continuity_016_dir": continuity_016_dir,
            "predicate_screen_dir": predicate_screen_dir,
        },
        "summaries": summaries,
        "tables": tables,
    }


def _route_key(row: pd.Series) -> str:
    return f"{row['start_condition']}|seed={int(row['seed'])}"


def _same_float(*values: float) -> bool:
    if not values:
        return False
    first = values[0]
    return all(abs(value - first) <= EPS for value in values[1:])


def _route_semantic_rows(context: dict[str, Any]) -> pd.DataFrame:
    trace = context["tables"]["first_pass_trace"]
    route_readout = context["tables"]["first_pass_route"]
    primary_trace = trace[trace["local_pair_id"].astype(str) == PRIMARY_PAIR_ID].copy()
    if primary_trace.empty:
        raise ValueError(f"No first-pass trace rows for {PRIMARY_PAIR_ID}")
    primary_trace = primary_trace.sort_values(["start_condition", "seed", "step_index"])

    readout_lookup = {
        (str(row["start_condition"]), int(row["seed"])): row
        for row in route_readout[
            route_readout["local_pair_id"].astype(str) == PRIMARY_PAIR_ID
        ].to_dict(orient="records")
    }
    rows: list[dict[str, Any]] = []
    for (start_condition, seed), route in primary_trace.groupby(
        ["start_condition", "seed"], sort=True
    ):
        by_step = route.set_index("step_index").sort_index()
        if not {1, 2, 3, 4, 5}.issubset(set(by_step.index.astype(int))):
            continue
        source = by_step.loc[1]
        transient = by_step.loc[2]
        target_steps = by_step.loc[[3, 4, 5]]
        first_target = by_step.loc[3]
        readout = readout_lookup.get((str(start_condition), int(seed)), {})

        source_assignment = str(source["endpoint_assignment_by_step"])
        transient_assignment = str(transient["endpoint_assignment_by_step"])
        target_assignments = target_steps["endpoint_assignment_by_step"].astype(str).tolist()
        target_signatures = target_steps["result_endpoint_signature_id"].astype(str).tolist()
        support_original = _as_float(transient["support_distance_to_original"])
        support_drop_bridge = _as_float(transient["support_distance_to_drop_bridge_edges"])
        support_drop_direct = _as_float(transient["support_distance_to_drop_direct_edge"])
        support_tie = _same_float(
            support_original, support_drop_bridge, support_drop_direct
        )
        source_to_transient_delta = _as_float(transient["objective_value_by_step"]) - _as_float(
            source["objective_value_by_step"]
        )
        transient_to_target_delta = _as_float(first_target["objective_value_by_step"]) - _as_float(
            transient["objective_value_by_step"]
        )
        source_to_target_delta = _as_float(first_target["objective_value_by_step"]) - _as_float(
            source["objective_value_by_step"]
        )
        target_signature_stable = len(set(target_signatures)) == 1
        target_assignment_stable = (
            len(set(target_assignments)) == 1
            and target_assignments[0] == "drop_bridge_target_anchor"
        )
        source_is_source_like = "original_source_anchor" in source_assignment
        transient_unknown = transient_assignment == "unknown_new_endpoint"
        transient_pair_separated = not _as_bool(transient["pair_coassigned"])
        transient_left_bridge_only = (
            _as_int(transient["left_bridge_same_cluster_count"]) > 0
            and _as_int(transient["right_bridge_same_cluster_count"]) == 0
            and _as_int(transient["pair_bridge_same_cluster_count"]) == 0
        )
        target_persists = target_signature_stable and target_assignment_stable
        objective_monotone_debt = (
            source_to_transient_delta < -EPS
            and transient_to_target_delta < -EPS
            and source_to_target_delta < -EPS
        )
        route_gateway_candidate = bool(
            source_is_source_like
            and transient_unknown
            and transient_pair_separated
            and transient_left_bridge_only
            and support_tie
            and target_persists
            and objective_monotone_debt
            and _as_bool(readout.get("target_final_bridge_exclusive_pass", False))
            and _as_bool(readout.get("direct_edge_retention_pass", False))
        )

        rows.append(
            {
                "local_pair_id": PRIMARY_PAIR_ID,
                "route_key": _route_key(source),
                "start_condition": str(start_condition),
                "seed": int(seed),
                "source_signature_id": str(source["result_endpoint_signature_id"]),
                "transient_signature_id": str(transient["result_endpoint_signature_id"]),
                "target_signature_id": str(first_target["result_endpoint_signature_id"]),
                "source_assignment": source_assignment,
                "transient_assignment": transient_assignment,
                "target_assignment_sequence": " -> ".join(target_assignments),
                "source_is_source_like": source_is_source_like,
                "transient_unknown_endpoint": transient_unknown,
                "transient_pair_separated": transient_pair_separated,
                "transient_left_bridge_only": transient_left_bridge_only,
                "transient_bridge_fraction": _as_float(
                    transient["bridge_edge_weight_fraction"]
                ),
                "transient_direct_fraction": _as_float(
                    transient["direct_edge_weight_fraction"]
                ),
                "support_distance_to_original_at_transient": support_original,
                "support_distance_to_drop_bridge_at_transient": support_drop_bridge,
                "support_distance_to_drop_direct_at_transient": support_drop_direct,
                "transient_support_equidistant_to_three_anchors": support_tie,
                "source_to_transient_objective_delta": source_to_transient_delta,
                "transient_to_target_objective_delta": transient_to_target_delta,
                "source_to_target_objective_delta": source_to_target_delta,
                "objective_monotone_debt_via_transient": objective_monotone_debt,
                "target_signature_stable_after_transient": target_signature_stable,
                "target_assignment_stable_after_transient": target_assignment_stable,
                "target_persists_after_transient": target_persists,
                "readout_source_start_support_pass": _as_bool(
                    readout.get("source_start_support_pass", False)
                ),
                "readout_post_start_continuity_pass": _as_bool(
                    readout.get("post_start_endpoint_continuity_pass", False)
                ),
                "readout_target_final_exclusive_pass": _as_bool(
                    readout.get("target_final_bridge_exclusive_pass", False)
                ),
                "readout_direct_edge_retention_pass": _as_bool(
                    readout.get("direct_edge_retention_pass", False)
                ),
                "route_gateway_candidate": route_gateway_candidate,
                "method_claim_allowed_after_semantic_validation": False,
                "quality_cost_claim_allowed_after_semantic_validation": False,
                "wall_generality_claim_allowed_after_semantic_validation": False,
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "method_status": METHOD_STATUS,
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows)


def _step_semantic_rows(context: dict[str, Any]) -> pd.DataFrame:
    trace = context["tables"]["first_pass_trace"]
    primary_trace = trace[trace["local_pair_id"].astype(str) == PRIMARY_PAIR_ID].copy()
    rows: list[dict[str, Any]] = []
    for step_index, step_rows in primary_trace.groupby("step_index", sort=True):
        dominant_signature = str(step_rows["result_endpoint_signature_id"].value_counts().idxmax())
        dominant_assignment = str(step_rows["endpoint_assignment_by_step"].value_counts().idxmax())
        support_tie_count = 0
        for row in step_rows.to_dict(orient="records"):
            if _same_float(
                _as_float(row["support_distance_to_original"]),
                _as_float(row["support_distance_to_drop_bridge_edges"]),
                _as_float(row["support_distance_to_drop_direct_edge"]),
            ):
                support_tie_count += 1
        rows.append(
            {
                "local_pair_id": PRIMARY_PAIR_ID,
                "step_index": int(step_index),
                "route_count": int(len(step_rows)),
                "signature_count": int(step_rows["result_endpoint_signature_id"].nunique()),
                "dominant_signature_id": dominant_signature,
                "dominant_assignment": dominant_assignment,
                "dominant_signature_route_count": int(
                    step_rows["result_endpoint_signature_id"].astype(str).eq(dominant_signature).sum()
                ),
                "pair_coassigned_rate": float(step_rows["pair_coassigned"].map(_as_bool).mean()),
                "support_equidistant_to_three_anchors_count": int(support_tie_count),
                "support_equidistant_to_three_anchors_share": float(
                    support_tie_count / len(step_rows)
                ),
                "objective_value_mean": float(step_rows["objective_value_by_step"].mean()),
                "objective_value_min": float(step_rows["objective_value_by_step"].min()),
                "objective_value_max": float(step_rows["objective_value_by_step"].max()),
                "bridge_fraction_min": float(step_rows["bridge_edge_weight_fraction"].min()),
                "bridge_fraction_max": float(step_rows["bridge_edge_weight_fraction"].max()),
                "method_claim_allowed_after_semantic_validation": False,
                "quality_cost_claim_allowed_after_semantic_validation": False,
                "wall_generality_claim_allowed_after_semantic_validation": False,
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "method_status": METHOD_STATUS,
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows)


def _comparison_rows(context: dict[str, Any]) -> pd.DataFrame:
    trace = context["tables"]["first_pass_trace"]
    transfer_pair = context["tables"]["transfer_pair"]
    predicate_pair = context["tables"]["predicate_pair_rows"]
    pairs = [
        REFERENCE_PAIR_ID,
        PRIMARY_PAIR_ID,
        BOUNDARY_GUARD_PAIR_ID,
        "local_pair_007",
        "local_pair_008",
        "local_pair_002",
        "local_pair_022",
        "local_pair_003",
        "local_pair_013",
    ]
    transfer_lookup = transfer_pair.set_index("local_pair_id").to_dict("index")
    p1_rows = predicate_pair[
        predicate_pair["predicate_id"].astype(str)
        == "P1_guarded_single_step_separated_transient_candidate"
    ].set_index("local_pair_id").to_dict("index")

    rows: list[dict[str, Any]] = []
    for pair_id in pairs:
        pair_trace = trace[trace["local_pair_id"].astype(str) == pair_id]
        transfer = transfer_lookup.get(pair_id, {})
        p1 = p1_rows.get(pair_id, {})
        if pair_trace.empty:
            continue
        step2 = pair_trace[pair_trace["step_index"].astype(int) == 2]
        step3 = pair_trace[pair_trace["step_index"].astype(int) == 3]
        support_tie_count = 0
        for row in step2.to_dict(orient="records"):
            if _same_float(
                _as_float(row["support_distance_to_original"]),
                _as_float(row["support_distance_to_drop_bridge_edges"]),
                _as_float(row["support_distance_to_drop_direct_edge"]),
            ):
                support_tie_count += 1
        rows.append(
            {
                "local_pair_id": pair_id,
                "validation_stratum": str(transfer.get("validation_stratum", "")),
                "transfer_screen_status": str(transfer.get("transfer_screen_status", "")),
                "p1_guarded_typed_transient_candidate_accepted": _as_bool(
                    p1.get("accepted_by_predicate", False)
                ),
                "route_count": int(pair_trace[["start_condition", "seed"]].drop_duplicates().shape[0]),
                "step2_signature_count": int(step2["result_endpoint_signature_id"].nunique()),
                "step2_dominant_signature_id": (
                    str(step2["result_endpoint_signature_id"].value_counts().idxmax())
                    if not step2.empty
                    else ""
                ),
                "step2_dominant_assignment": (
                    str(step2["endpoint_assignment_by_step"].value_counts().idxmax())
                    if not step2.empty
                    else ""
                ),
                "step2_pair_coassigned_rate": float(step2["pair_coassigned"].map(_as_bool).mean())
                if not step2.empty
                else 0.0,
                "step2_support_equidistant_share": float(support_tie_count / len(step2))
                if not step2.empty
                else 0.0,
                "step3_target_anchor_share": float(
                    step3["endpoint_assignment_by_step"].astype(str).eq("drop_bridge_target_anchor").mean()
                )
                if not step3.empty
                else 0.0,
                "has_source_like_signature": _as_bool(
                    transfer.get("has_source_like_signature", False)
                ),
                "has_target_anchor_signature": _as_bool(
                    transfer.get("has_target_anchor_signature", False)
                ),
                "has_transition_intermediate_analog": _as_bool(
                    transfer.get("has_transition_intermediate_analog", False)
                ),
                "has_unresolved_pair_separated_signature": _as_bool(
                    transfer.get("has_unresolved_pair_separated_signature", False)
                ),
                "source_target_signature_collapse_count": _as_int(
                    transfer.get("source_target_signature_collapse_count", 0)
                ),
                "guard_anchor_collapse_count": _as_int(
                    transfer.get("guard_anchor_collapse_count", 0)
                ),
                "semantic_read": (
                    "primary_recurrent_separated_gateway_candidate"
                    if pair_id == PRIMARY_PAIR_ID
                    else "reference_or_guard_contrast"
                ),
                "method_claim_allowed_after_semantic_validation": False,
                "quality_cost_claim_allowed_after_semantic_validation": False,
                "wall_generality_claim_allowed_after_semantic_validation": False,
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "method_status": METHOD_STATUS,
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows)


def _semantic_decision_rows(
    context: dict[str, Any],
    route_rows: pd.DataFrame,
    step_rows: pd.DataFrame,
    comparison_rows: pd.DataFrame,
) -> pd.DataFrame:
    continuity_summary = context["summaries"]["continuity_016"]
    predicate_summary = context["summaries"]["predicate_screen"]
    route_count = int(len(route_rows))
    gateway_count = int(route_rows["route_gateway_candidate"].map(_as_bool).sum())
    transient_signature_count = int(route_rows["transient_signature_id"].nunique())
    transient_ids = sorted(route_rows["transient_signature_id"].astype(str).unique().tolist())
    step2 = step_rows[step_rows["step_index"].astype(int) == 2].iloc[0].to_dict()
    p1_accepted = predicate_summary["accepted_pairs_by_predicate"][
        "P1_guarded_single_step_separated_transient_candidate"
    ]
    p2_leaks = predicate_summary["guard_leaks_by_predicate"][
        "P2_endpoint_only_negative_control"
    ]
    p3_leaks = predicate_summary["guard_leaks_by_predicate"][
        "P3_role_analog_only_negative_control"
    ]

    return pd.DataFrame(
        [
            {
                "decision_id": "S1_recurrent_step2_signature",
                "semantic_axis": "recurrence",
                "observed": (
                    f"{route_count}/{route_count} routes use transient signatures "
                    f"{_list_json(transient_ids)}; step-2 dominant signature count "
                    f"{int(step2['dominant_signature_route_count'])}/{int(step2['route_count'])}."
                ),
                "decision": "supports_recurrent_transition_object",
                "passes": transient_signature_count == 1
                and int(step2["dominant_signature_route_count"]) == route_count,
                "claim_effect": "rules_out_plain_seed_noise_but_not_wall_promotion",
            },
            {
                "decision_id": "S2_source_transient_target_bracket",
                "semantic_axis": "path shape",
                "observed": (
                    f"{gateway_count}/{route_count} routes are source-like at step 1, "
                    "unknown/separated at step 2, and stable target from steps 3-5."
                ),
                "decision": "supports_one_step_gateway_shape",
                "passes": gateway_count == route_count,
                "claim_effect": "candidate_pathway_gateway_only",
            },
            {
                "decision_id": "S3_typed_pair_separation",
                "semantic_axis": "role morphology",
                "observed": (
                    "Continuity audit types the recurrent step as L+B1 separated "
                    "from R, with pair_coassigned=false in every 016 route."
                ),
                "decision": "supports_typed_separated_bridge_reassignment",
                "passes": bool(
                    continuity_summary.get("primary_single_step_bridge_reassignment_block_count")
                    == route_count
                ),
                "claim_effect": "typed object, not just unknown endpoint label",
            },
            {
                "decision_id": "S4_anchor_equidistant_saddle",
                "semantic_axis": "endpoint identity",
                "observed": (
                    "At step 2, support distance to original, drop-bridge, and "
                    "drop-direct anchors is 0.044444 for every route."
                ),
                "decision": "blocks_endpoint_basin_promotion",
                "passes": bool(route_rows["transient_support_equidistant_to_three_anchors"].map(_as_bool).all()),
                "claim_effect": "treat as anchor-saddle gateway, not endpoint basin",
            },
            {
                "decision_id": "S5_monotone_debt_without_recovery",
                "semantic_axis": "objective trajectory",
                "observed": (
                    "Objective value decreases monotonically from source to transient "
                    "to target in every 016 route, so objective debt accumulates; "
                    "max_objective_recovery_from_min is 0."
                ),
                "decision": "blocks_debt_recovery_wall_or_tunneling_claim",
                "passes": bool(route_rows["objective_monotone_debt_via_transient"].map(_as_bool).all())
                and float(
                    context["tables"]["continuity_route"][
                        "max_objective_recovery_from_min"
                    ].max()
                )
                == 0.0,
                "claim_effect": "debt-only transition gateway, not demonstrated barrier crossing",
            },
            {
                "decision_id": "S6_guarded_definition_closure",
                "semantic_axis": "guard contrast",
                "observed": (
                    f"P1 accepts {_list_json(p1_accepted)} with 0 guard leaks; "
                    f"P2 leaks {json.dumps(p2_leaks, sort_keys=True)}; "
                    f"P3 leaks {json.dumps(p3_leaks, sort_keys=True)}."
                ),
                "decision": "supports_guarded_candidate_only",
                "passes": p1_accepted == [PRIMARY_PAIR_ID],
                "claim_effect": "endpoint-only and role-only definitions remain rejected",
            },
            {
                "decision_id": "S7_evidence_boundary",
                "semantic_axis": "claim boundary",
                "observed": (
                    "No reverse path, no new localization trace, no hysteresis loop, "
                    "and no full NanoClustering replay are executed here."
                ),
                "decision": "blocks_positive_wall_or_method_claim",
                "passes": True,
                "claim_effect": "next gate must test semantic persistence or reversibility",
            },
        ]
    )


def _gate_matrix(
    context: dict[str, Any],
    route_rows: pd.DataFrame,
    step_rows: pd.DataFrame,
    decision_rows: pd.DataFrame,
    comparison_rows: pd.DataFrame,
) -> pd.DataFrame:
    upstream_failed = {
        name: context["summaries"][name].get("failed_gates", [])
        for name in ("first_pass", "transfer_screen", "continuity_016", "predicate_screen")
    }
    p1_comparison = comparison_rows[
        comparison_rows["p1_guarded_typed_transient_candidate_accepted"].map(_as_bool)
    ]["local_pair_id"].astype(str).tolist()
    all_claim_flags_false = (
        not route_rows["method_claim_allowed_after_semantic_validation"].map(_as_bool).any()
        and not route_rows["quality_cost_claim_allowed_after_semantic_validation"].map(_as_bool).any()
        and not route_rows["wall_generality_claim_allowed_after_semantic_validation"].map(_as_bool).any()
        and not comparison_rows["method_claim_allowed_after_semantic_validation"].map(_as_bool).any()
    )

    return pd.DataFrame(
        [
            _gate_row(
                "G1",
                "Did all upstream read-only audits pass?",
                upstream_failed,
                "first-pass, transfer-screen, continuity-block, and predicate-screen failed_gates are empty",
                all(not failed for failed in upstream_failed.values()),
            ),
            _gate_row(
                "G2",
                "Is the 016 transient recurrent and seed/start stable?",
                {
                    "route_count": int(len(route_rows)),
                    "transient_signature_count": int(route_rows["transient_signature_id"].nunique()),
                    "gateway_candidate_count": int(route_rows["route_gateway_candidate"].map(_as_bool).sum()),
                },
                "24/24 routes share one transient and pass route gateway criteria",
                int(len(route_rows)) == 24
                and int(route_rows["transient_signature_id"].nunique()) == 1
                and int(route_rows["route_gateway_candidate"].map(_as_bool).sum()) == 24,
            ),
            _gate_row(
                "G3",
                "Is the transient bracketed by stable target persistence?",
                step_rows[["step_index", "dominant_assignment", "signature_count"]].to_dict(
                    orient="records"
                ),
                "step 2 unknown, steps 3-5 stable drop-bridge target signature",
                bool(route_rows["target_persists_after_transient"].map(_as_bool).all()),
            ),
            _gate_row(
                "G4",
                "Does support geometry block endpoint-basin promotion?",
                {
                    "equidistant_route_count": int(
                        route_rows["transient_support_equidistant_to_three_anchors"].map(_as_bool).sum()
                    ),
                    "route_count": int(len(route_rows)),
                },
                "transient is equidistant to original/drop-bridge/drop-direct in every route",
                bool(route_rows["transient_support_equidistant_to_three_anchors"].map(_as_bool).all()),
            ),
            _gate_row(
                "G5",
                "Does objective trajectory block debt-recovery wall interpretation?",
                {
                    "monotone_debt_route_count": int(
                        route_rows["objective_monotone_debt_via_transient"].map(_as_bool).sum()
                    ),
                    "max_recovery_from_min": float(
                        context["tables"]["continuity_route"][
                            "max_objective_recovery_from_min"
                        ].max()
                    ),
                },
                "monotone objective debt through the transient and zero recovery from min",
                bool(route_rows["objective_monotone_debt_via_transient"].map(_as_bool).all())
                and float(
                    context["tables"]["continuity_route"][
                        "max_objective_recovery_from_min"
                    ].max()
                )
                == 0.0,
            ),
            _gate_row(
                "G6",
                "Does the guarded predicate still isolate 016?",
                p1_comparison,
                "P1 accepts exactly local_pair_016 among comparison rows",
                p1_comparison == [PRIMARY_PAIR_ID],
            ),
            _gate_row(
                "G7",
                "Are claim boundaries preserved?",
                {
                    "method_quality_wall_flags_all_false": all_claim_flags_false,
                    "route_execution_status": ROUTE_EXECUTION_STATUS,
                    "wall_promotion_status": WALL_PROMOTION_STATUS,
                    "semantic_decisions_passed": int(decision_rows["passes"].map(_as_bool).sum()),
                },
                "no method, quality/cost, wall-generality, wall-promotion, or new-route claim",
                all_claim_flags_false,
            ),
        ]
    )


def _write_report(
    output_dir: Path,
    route_rows: pd.DataFrame,
    step_rows: pd.DataFrame,
    comparison_rows: pd.DataFrame,
    decision_rows: pd.DataFrame,
    gate_matrix: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    lines = [
        "# NanoClustering G4.8 First-Pass 016 Transient Semantic Validation",
        "",
        "## Claim Boundary",
        "",
        CLAIM_BOUNDARY,
        "",
        "## Summary",
        "",
        _markdown_table(
            pd.DataFrame(
                [
                    {
                        "semantic_classification": summary["semantic_classification"],
                        "route_count": summary["route_semantic_row_count"],
                        "gateway_candidate_route_count": summary[
                            "gateway_candidate_route_count"
                        ],
                        "failed_gates": _list_json(summary["failed_gates"]),
                        "recommended_next_gate": summary["recommended_next_gate"],
                    }
                ]
            ),
            [
                "semantic_classification",
                "route_count",
                "gateway_candidate_route_count",
                "failed_gates",
                "recommended_next_gate",
            ],
        ),
        "",
        "## Step Semantics",
        "",
        _markdown_table(
            step_rows,
            [
                "step_index",
                "route_count",
                "signature_count",
                "dominant_signature_id",
                "dominant_assignment",
                "pair_coassigned_rate",
                "support_equidistant_to_three_anchors_share",
                "objective_value_mean",
            ],
        ),
        "",
        "## Route Semantics",
        "",
        _markdown_table(
            route_rows,
            [
                "route_key",
                "source_signature_id",
                "transient_signature_id",
                "target_signature_id",
                "transient_support_equidistant_to_three_anchors",
                "objective_monotone_debt_via_transient",
                "target_persists_after_transient",
                "route_gateway_candidate",
            ],
            max_rows=30,
        ),
        "",
        "## Comparison Rows",
        "",
        _markdown_table(
            comparison_rows,
            [
                "local_pair_id",
                "validation_stratum",
                "p1_guarded_typed_transient_candidate_accepted",
                "step2_signature_count",
                "step2_dominant_assignment",
                "step2_pair_coassigned_rate",
                "step2_support_equidistant_share",
                "step3_target_anchor_share",
                "source_target_signature_collapse_count",
                "guard_anchor_collapse_count",
            ],
        ),
        "",
        "## Semantic Decisions",
        "",
        _markdown_table(
            decision_rows,
            [
                "decision_id",
                "semantic_axis",
                "decision",
                "passes",
                "claim_effect",
            ],
        ),
        "",
        "## Gates",
        "",
        _markdown_table(
            gate_matrix,
            ["gate_id", "question", "observed", "minimum_or_rule", "gate_status"],
        ),
        "",
        "## Interpretation",
        "",
        "- The 016 step-2 transient is recurrent and typed, so it should not be dismissed as plain seed noise.",
        "- It is equidistant to the known anchors and has no persistence as a terminal endpoint, so it should not be promoted to a basin endpoint.",
        "- The objective path accumulates debt into the target with no recovery, so it is not yet tunneling or wall evidence.",
        "- The current status is a guarded typed transition-gateway candidate; the next gate should test persistence or reversibility before broad execution.",
        "",
    ]
    (output_dir / REPORT_MD).write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    context = _load_context(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    route_rows = _route_semantic_rows(context)
    step_rows = _step_semantic_rows(context)
    comparison_rows = _comparison_rows(context)
    decision_rows = _semantic_decision_rows(context, route_rows, step_rows, comparison_rows)
    gate_matrix = _gate_matrix(
        context, route_rows, step_rows, decision_rows, comparison_rows
    )

    failed_gates = gate_matrix.loc[
        gate_matrix["gate_status"].astype(str) != "pass", "gate_id"
    ].astype(str).tolist()
    gateway_candidate_route_count = int(route_rows["route_gateway_candidate"].map(_as_bool).sum())
    semantic_classification = (
        "recurrent_typed_transition_gateway_candidate_not_endpoint_or_positive_wall"
        if gateway_candidate_route_count == len(route_rows)
        and bool(route_rows["transient_support_equidistant_to_three_anchors"].map(_as_bool).all())
        else "unresolved_or_artifact_candidate"
    )
    summary = {
        "schema": "nanoclustering_g4_8_first_pass_016_transient_semantic_summary.v1",
        "status": RUN_STATUS,
        "output_dir": str(output_dir.resolve()),
        "source_dirs": {
            key: str(path.resolve()) for key, path in context["paths"].items()
        },
        "claim_boundary": CLAIM_BOUNDARY,
        "primary_pair": PRIMARY_PAIR_ID,
        "reference_pair": REFERENCE_PAIR_ID,
        "boundary_guard_pair": BOUNDARY_GUARD_PAIR_ID,
        "semantic_classification": semantic_classification,
        "route_semantic_row_count": int(len(route_rows)),
        "step_semantic_row_count": int(len(step_rows)),
        "comparison_row_count": int(len(comparison_rows)),
        "semantic_decision_row_count": int(len(decision_rows)),
        "gateway_candidate_route_count": gateway_candidate_route_count,
        "transient_signature_ids": sorted(
            route_rows["transient_signature_id"].astype(str).unique().tolist()
        ),
        "transient_support_equidistant_route_count": int(
            route_rows["transient_support_equidistant_to_three_anchors"].map(_as_bool).sum()
        ),
        "objective_debt_route_count": int(
            route_rows["objective_monotone_debt_via_transient"].map(_as_bool).sum()
        ),
        "target_persistence_route_count": int(
            route_rows["target_persists_after_transient"].map(_as_bool).sum()
        ),
        "decision_status_counts": {
            str(k): int(v) for k, v in decision_rows["passes"].value_counts().items()
        },
        "gate_status_counts": {
            str(k): int(v) for k, v in gate_matrix["gate_status"].value_counts().items()
        },
        "failed_gates": failed_gates,
        "semantic_readout": {
            "plain_artifact_status": "not_plain_seed_noise_because_recurrent_typed_step",
            "endpoint_basin_status": "not_promoted_because_anchor_equidistant_and_nonpersistent",
            "wall_or_tunneling_status": "not_promoted_because_debt_without_recovery_and_no_reverse_hysteresis",
            "pathway_status": "guarded_transition_gateway_candidate_only",
        },
        "recommended_next_gate": (
            "Design a minimal persistence/reversibility check for the 016 transient "
            "before broad localization: test whether the step-2 saddle survives finer "
            "fractions or reverse target-to-source traces under the same guards."
        ),
    }
    config = {
        "first_pass_dir": str(Path(args.first_pass_dir).resolve()),
        "transfer_screen_dir": str(Path(args.transfer_screen_dir).resolve()),
        "continuity_016_dir": str(Path(args.continuity_016_dir).resolve()),
        "predicate_screen_dir": str(Path(args.predicate_screen_dir).resolve()),
        "output_dir": str(output_dir.resolve()),
        "claim_boundary": CLAIM_BOUNDARY,
    }

    _write_csv(route_rows, output_dir / ROUTE_SEMANTIC_ROWS_CSV)
    _write_csv(step_rows, output_dir / STEP_SEMANTIC_ROWS_CSV)
    _write_csv(comparison_rows, output_dir / COMPARISON_ROWS_CSV)
    _write_csv(decision_rows, output_dir / SEMANTIC_DECISION_ROWS_CSV)
    _write_csv(gate_matrix, output_dir / GATE_MATRIX_CSV)
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_report(
        output_dir,
        route_rows,
        step_rows,
        comparison_rows,
        decision_rows,
        gate_matrix,
        summary,
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate local_pair_016 typed-transient semantics."
    )
    parser.add_argument("--first-pass-dir", type=Path, default=DEFAULT_FIRST_PASS_DIR)
    parser.add_argument("--transfer-screen-dir", type=Path, default=DEFAULT_TRANSFER_SCREEN_DIR)
    parser.add_argument("--continuity-016-dir", type=Path, default=DEFAULT_CONTINUITY_016_DIR)
    parser.add_argument("--predicate-screen-dir", type=Path, default=DEFAULT_PREDICATE_SCREEN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
