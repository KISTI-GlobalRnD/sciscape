#!/usr/bin/env python3
"""Design the G4.8 fresh Axis B first-pass readout contract.

The fresh Axis B panel freezes 36 first-pass route rows, but those rows are a
conditional ready-like screen rather than a stable-positive generality test.
This script fixes the evidence ladder, control traps, aggregation rules, and
outcome taxonomy before any route execution reads those rows.

It does not run Leiden, execute route/pathway traces, promote walls, evaluate
quality/cost value, replay full NanoClustering, or claim method success.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from run_leiden_basin_nanoclustering_role_local_route_pilot import (
    BASE_RESULT_DIR,
    _json_safe,
    _read_csv,
    _write_csv,
)


DEFAULT_PANEL_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_fresh_axis_b_panel_contract_gamma1e5_20260604"
)
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_fresh_axis_b_first_pass_readout_contract_gamma1e5_20260604"
)

PANEL_PAIR_ROWS_CSV = "nanoclustering_g4_8_fresh_axis_b_panel_pair_rows.csv"
PANEL_ROUTE_PLAN_ROWS_CSV = (
    "nanoclustering_g4_8_fresh_axis_b_panel_route_plan_rows.csv"
)
PANEL_BATCH_PLAN_ROWS_CSV = (
    "nanoclustering_g4_8_fresh_axis_b_panel_batch_plan_rows.csv"
)
PANEL_FIELD_CONTRACT_ROWS_CSV = (
    "nanoclustering_g4_8_fresh_axis_b_panel_field_contract_rows.csv"
)
PANEL_GATE_MATRIX_CSV = "nanoclustering_g4_8_fresh_axis_b_panel_gate_matrix.csv"
PANEL_SUMMARY_JSON = "nanoclustering_g4_8_fresh_axis_b_panel_summary.json"

CLAIM_LADDER_ROWS_CSV = (
    "nanoclustering_g4_8_fresh_axis_b_first_pass_claim_ladder_rows.csv"
)
READOUT_FIELD_ROWS_CSV = (
    "nanoclustering_g4_8_fresh_axis_b_first_pass_readout_field_rows.csv"
)
CONTROL_TRAP_ROWS_CSV = (
    "nanoclustering_g4_8_fresh_axis_b_first_pass_control_trap_rows.csv"
)
AGGREGATION_RULE_ROWS_CSV = (
    "nanoclustering_g4_8_fresh_axis_b_first_pass_aggregation_rule_rows.csv"
)
OUTCOME_TAXONOMY_ROWS_CSV = (
    "nanoclustering_g4_8_fresh_axis_b_first_pass_outcome_taxonomy_rows.csv"
)
ROUTE_READOUT_ROWS_CSV = (
    "nanoclustering_g4_8_fresh_axis_b_first_pass_route_readout_rows.csv"
)
PAIR_READOUT_ROWS_CSV = (
    "nanoclustering_g4_8_fresh_axis_b_first_pass_pair_readout_rows.csv"
)
READOUT_ORDER_ROWS_CSV = (
    "nanoclustering_g4_8_fresh_axis_b_first_pass_readout_order_rows.csv"
)
GATE_MATRIX_CSV = (
    "nanoclustering_g4_8_fresh_axis_b_first_pass_readout_gate_matrix.csv"
)
CONFIG_JSON = "nanoclustering_g4_8_fresh_axis_b_first_pass_readout_config.json"
SUMMARY_JSON = "nanoclustering_g4_8_fresh_axis_b_first_pass_readout_summary.json"
REPORT_MD = "nanoclustering_g4_8_fresh_axis_b_first_pass_readout_report.md"

RUN_STATUS = "designed_nanoclustering_g4_8_fresh_axis_b_first_pass_readout_contract"
CLAIM_BOUNDARY = (
    "NanoClustering G4.8 fresh Axis B first-pass readout contract design only; "
    "reads the fresh Axis B panel contract and fixes claim ladder, control "
    "traps, aggregation rules, and outcome taxonomy for the 36 first-pass rows. "
    "It does not run Leiden, execute route/pathway traces, promote walls, "
    "evaluate quality/cost value, replay full NanoClustering, or claim method "
    "success."
)

READY_ROLE = "fresh_core_ready_conditional_pair"
CONTROL_ROLE = "fresh_first_pass_control_pair"

CONTROL_TRAP_SPECS = {
    "local_pair_002": {
        "control_trap_id": "C_target_saturated_false_positive_trap",
        "control_trap_family": "target_saturated_no_handle",
        "trap_question": (
            "Does an already-target-saturated pair look like Axis B success "
            "without a distinct source-to-target transition?"
        ),
        "positive_leak_signal": (
            "target-final continuity appears without distinct source-start and "
            "post-start transition evidence"
        ),
        "expected_guard_outcome": "should_not_count_as_ready_like_positive",
    },
    "local_pair_008": {
        "control_trap_id": "C_latent_release_without_source_trap",
        "control_trap_family": "latent_release_without_original_coassigned_source",
        "trap_question": (
            "Does bridge release alone look positive when the original "
            "coassigned source handle is absent?"
        ),
        "positive_leak_signal": (
            "bridge-release endpoint movement without source-start support"
        ),
        "expected_guard_outcome": "should_block_release_only_false_positive",
    },
    "local_pair_013": {
        "control_trap_id": "C_hard_no_release_trap",
        "control_trap_family": "hard_no_release_control",
        "trap_question": (
            "Does the readout stay closed when no bridge-release coassignment is "
            "observed?"
        ),
        "positive_leak_signal": "ready-like continuity appears in a hard no-release control",
        "expected_guard_outcome": "should_remain_negative",
    },
    "local_pair_022": {
        "control_trap_id": "C_coupled_direct_bridge_failure_trap",
        "control_trap_family": "coupled_direct_bridge_failure",
        "trap_question": (
            "Does a coupled direct/bridge failure produce a false positive when "
            "bridge removal destroys the target?"
        ),
        "positive_leak_signal": (
            "source-to-target continuity appears despite coupled direct/bridge failure"
        ),
        "expected_guard_outcome": "should_block_coupled_context_false_positive",
    },
}


def _count_dict(series: pd.Series) -> dict[str, int]:
    if series.empty:
        return {}
    return {str(key): int(value) for key, value in series.value_counts(dropna=False).items()}


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _join_values(values: pd.Series) -> str:
    clean = [str(value) for value in values.dropna().astype(str).tolist()]
    return ";".join(sorted(set(clean)))


def _regime(value: Any, *, low: float, high: float) -> str:
    if pd.isna(value):
        return "unknown"
    number = float(value)
    if number < low:
        return "low"
    if number > high:
        return "high"
    return "medium"


def _markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 50) -> str:
    cols = [col for col in columns if col in frame.columns]
    if not cols:
        return "No columns."
    visible = frame[cols].head(int(max_rows))
    header = "| " + " | ".join(cols) + " |"
    separator = "| " + " | ".join("---" for _ in cols) + " |"
    rows: list[str] = []
    for row in visible.itertuples(index=False):
        values: list[str] = []
        for value in row:
            if isinstance(value, (dict, list, tuple, set)):
                values.append(json.dumps(_json_safe(value), sort_keys=True))
            elif pd.isna(value):
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value).replace("\n", " "))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def _claim_ladder_rows() -> pd.DataFrame:
    specs = [
        (
            0,
            "execution_and_field_completion",
            "All first-pass route rows execute and populate required split fields.",
            "execution completeness only",
            "not basin evidence",
        ),
        (
            1,
            "ready_like_target_final_continuity",
            "Ready-like rows show target-final continuity under allowed starts.",
            "conditional ready-like route evidence",
            "not ready/control separation by itself",
        ),
        (
            2,
            "ready_control_separation",
            "Ready-like rows separate from the four control traps.",
            "first-pass conditional Axis B screen",
            "not stable-positive generality",
        ),
        (
            3,
            "source_start_support_stability",
            "Source-start support remains stable enough to avoid interior-only claims.",
            "source-start-supported conditional screen",
            "not full seed/start robustness",
        ),
        (
            4,
            "seed_start_rotation_robustness",
            "Allowed-start evidence remains robust under seed/start rotation.",
            "rotation-robust conditional screen",
            "not branch or direct-dependent generality",
        ),
        (
            5,
            "basin_pathway_generality",
            "Multiple stable-positive and diagnostic sentinel rows support generality.",
            "future claim only",
            "closed for this first-pass contract",
        ),
    ]
    rows = pd.DataFrame(
        [
            {
                "claim_level": level,
                "claim_level_id": level_id,
                "claim_question": question,
                "claim_if_passed": claim,
                "claim_boundary": boundary,
                "available_in_first_pass": level <= 3,
                "allowed_as_first_pass_claim": level <= 2,
            }
            for level, level_id, question, claim, boundary in specs
        ]
    )
    rows["run_status"] = RUN_STATUS
    rows["contract_boundary"] = CLAIM_BOUNDARY
    return rows


def _readout_field_rows() -> pd.DataFrame:
    specs = [
        (
            "source_start_support_pass",
            "boolean",
            "route",
            "source_start",
            "Step-1 endpoint has source-start support.",
            True,
        ),
        (
            "post_start_endpoint_continuity_pass",
            "boolean",
            "route",
            "interior_endpoint",
            "Post-start endpoint signatures are known and not true-novel.",
            True,
        ),
        (
            "target_final_continuity_pass",
            "boolean",
            "route",
            "interior_endpoint",
            "Final route endpoint is the intended pair-level target.",
            True,
        ),
        (
            "direct_edge_retention_pass",
            "boolean",
            "route",
            "physical_direct_path",
            "Direct pair edge remains physically retained during the test.",
            True,
        ),
        (
            "same_seed_unknown_reclassified_pair_known",
            "boolean",
            "route",
            "endpoint_atlas",
            "Same-seed unknown endpoints are pair-level known where applicable.",
            False,
        ),
        (
            "objective_debt_recovery_status",
            "categorical",
            "route",
            "pathway_shape",
            "Objective debt/recovery shape; diagnostic only.",
            False,
        ),
        (
            "control_trap_positive_leak",
            "boolean",
            "route",
            "control_guard",
            "Control row leaks a ready-like positive signal.",
            True,
        ),
        (
            "route_outcome_class",
            "categorical",
            "route",
            "outcome_taxonomy",
            "Route-level result mapped to the predeclared taxonomy.",
            True,
        ),
        (
            "wall_claim_allowed",
            "boolean",
            "route",
            "claim_boundary",
            "Must remain false in first-pass readout.",
            True,
        ),
        (
            "method_quality_cost_claim_allowed",
            "boolean",
            "route",
            "claim_boundary",
            "Must remain false in first-pass readout.",
            True,
        ),
    ]
    rows = pd.DataFrame(
        [
            {
                "readout_field": field,
                "field_type": field_type,
                "aggregation_level": level,
                "field_family": family,
                "field_question": question,
                "required_for_first_pass_readout": required,
                "missing_value_action": (
                    "route_incomplete" if required else "retain_as_unknown_diagnostic"
                ),
            }
            for field, field_type, level, family, question, required in specs
        ]
    )
    rows["run_status"] = RUN_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


def _control_trap_rows(pair_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pair_id, spec in CONTROL_TRAP_SPECS.items():
        pair = pair_rows[pair_rows["local_pair_id"].astype(str).eq(pair_id)]
        if pair.empty:
            base: dict[str, Any] = {}
        else:
            base = pair.iloc[0].to_dict()
        rows.append(
            {
                "local_pair_id": pair_id,
                "branch": base.get("branch", ""),
                "validation_stratum": base.get("validation_stratum", ""),
                "limitation_axis_resolved": base.get("limitation_axis_resolved", ""),
                "control_trap_id": spec["control_trap_id"],
                "control_trap_family": spec["control_trap_family"],
                "trap_question": spec["trap_question"],
                "positive_leak_signal": spec["positive_leak_signal"],
                "expected_guard_outcome": spec["expected_guard_outcome"],
                "bridge_release_lift_proxy": base.get("bridge_release_lift_proxy"),
                "direct_dependency_proxy": base.get("direct_dependency_proxy"),
                "allowed_execution_unit_count": int(
                    base.get("allowed_execution_unit_count", 0) or 0
                ),
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows)


def _aggregation_rule_rows() -> pd.DataFrame:
    specs = [
        (
            "A1_route_level",
            "route",
            "Interpret each pair/start route independently first.",
            "No route can open a wall, method, quality/cost, or full-replay claim.",
        ),
        (
            "A2_start_level",
            "start",
            "Aggregate across seeds only after route-level required fields are present.",
            "Do not compare blocked starts; allowed starts only.",
        ),
        (
            "A3_pair_level",
            "pair",
            "Pair-level support requires consistent allowed-start behavior.",
            "local_pair_003 has one allowed start, so pair-level robustness is limited.",
        ),
        (
            "A4_control_trap_level",
            "control_trap",
            "Read the four control traps separately before ready-like positives.",
            "Do not average controls into a single negative score.",
        ),
        (
            "A5_panel_level",
            "panel",
            "Panel-level screen passes only if ready-like rows separate from controls.",
            "This is a conditional ready-like screen, not stable-positive generality.",
        ),
        (
            "A6_escalation_level",
            "post_first_pass",
            "Open boundary sentinel local_pair_020 only after first-pass inspection.",
            "Direct-dependent generality remains closed until sentinel evidence exists.",
        ),
    ]
    rows = pd.DataFrame(
        [
            {
                "aggregation_rule_id": rule_id,
                "aggregation_level": level,
                "rule": rule,
                "anti_overclaim_boundary": boundary,
            }
            for rule_id, level, rule, boundary in specs
        ]
    )
    rows["run_status"] = RUN_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


def _outcome_taxonomy_rows() -> pd.DataFrame:
    specs = [
        (
            "strong_conditional_axis_b_screen_pass",
            "ready-like source-start, post-start, target-final, and direct-edge fields pass; controls stay closed",
            "claim level 2, with source-start support evidence toward level 3",
            "still not stable-positive or basin/pathway generality",
        ),
        (
            "interior_only_pass_source_start_weak",
            "ready-like post-start and target-final pass but source-start support is weak",
            "interior endpoint continuity only",
            "cannot repair source-start caveats",
        ),
        (
            "false_positive_risk_control_leak",
            "one or more control traps leak ready-like positive signals",
            "screen not passed",
            "ready-like positives must be reinterpreted as non-specific",
        ),
        (
            "conditional_start_artifact",
            "signal exists only in sparse allowed-start conditions",
            "route-local or start-local evidence",
            "no pair-level robustness claim",
        ),
        (
            "direct_dependency_unresolved",
            "ready-like rows pass but only low/medium direct-dependency regimes are covered",
            "release-like conditional screen",
            "direct-dependent generality remains closed",
        ),
        (
            "branch_generalization_unresolved",
            "java-dominant first-pass rows pass but rust coverage remains thin",
            "branch-limited screen",
            "branch generality remains closed",
        ),
        (
            "null_ready_like_target_failure",
            "ready-like rows do not reach target-final continuity",
            "screen failed",
            "do not open boundary sentinel",
        ),
        (
            "execution_incomplete",
            "required readout fields are missing",
            "no evidence claim",
            "rerun or repair instrumentation before interpretation",
        ),
    ]
    rows = pd.DataFrame(
        [
            {
                "outcome_class": outcome,
                "trigger_condition": trigger,
                "allowed_interpretation": interpretation,
                "blocked_interpretation": blocked,
            }
            for outcome, trigger, interpretation, blocked in specs
        ]
    )
    rows["run_status"] = RUN_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


def _readout_order_rows() -> pd.DataFrame:
    specs = [
        (
            1,
            "control_trap_scan",
            "Inspect target-saturated, latent-source-absent, hard-no-release, and coupled-failure controls first.",
            "prevents ready-like false-positive overread",
        ),
        (
            2,
            "ready_like_target_final_scan",
            "Inspect target-final continuity in ready-like conditional rows.",
            "opens claim level 1 only",
        ),
        (
            3,
            "source_start_support_scan",
            "Inspect source-start support separately from post-start endpoint continuity.",
            "decides strong versus interior-only pass",
        ),
        (
            4,
            "seed_start_robustness_scan",
            "Inspect seed/start robustness only where multiple allowed starts exist.",
            "local_pair_003 remains route-local because it has one allowed start",
        ),
        (
            5,
            "escalation_decision",
            "Decide whether to open local_pair_020 and reserve controls.",
            "keeps direct-dependent and branch generality closed until warranted",
        ),
    ]
    rows = pd.DataFrame(
        [
            {
                "readout_order": order,
                "readout_step": step,
                "instruction": instruction,
                "why_first_pass_requires_this": reason,
            }
            for order, step, instruction, reason in specs
        ]
    )
    rows["run_status"] = RUN_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


def _build_first_pass_route_rows(
    route_plan_rows: pd.DataFrame,
    pair_rows: pd.DataFrame,
    control_trap_rows: pd.DataFrame,
) -> pd.DataFrame:
    first = route_plan_rows[
        route_plan_rows["new_route_execution_required"].map(_as_bool)
    ].copy()
    pair_keep = [
        "local_pair_id",
        "branch",
        "validation_stratum",
        "validation_family",
        "allowed_execution_unit_count",
        "allowed_start_conditions",
        "blocked_start_conditions",
        "bridge_release_lift_proxy",
        "direct_dependency_proxy",
        "original_distinct_endpoint_count",
        "discovery_original_source_endpoint_signature_proxy_count",
        "heldout_original_source_endpoint_signature_proxy_count",
    ]
    first = first.merge(
        pair_rows[[col for col in pair_keep if col in pair_rows.columns]],
        on="local_pair_id",
        how="left",
        validate="many_to_one",
        suffixes=("", "_pair"),
    )
    first = first.merge(
        control_trap_rows[
            [
                "local_pair_id",
                "control_trap_id",
                "control_trap_family",
                "positive_leak_signal",
                "expected_guard_outcome",
            ]
        ],
        on="local_pair_id",
        how="left",
        validate="many_to_one",
    )
    first["readout_plan_id"] = (
        first["route_plan_id"].astype(str) + "__first_pass_readout"
    )
    first["evidence_role"] = np.select(
        [
            first["panel_role"].astype(str).eq(READY_ROLE),
            first["panel_role"].astype(str).eq(CONTROL_ROLE),
        ],
        ["conditional_ready_like_test", "control_false_positive_guard"],
        default="unexpected_first_pass_role",
    )
    first["max_claim_level_if_success"] = np.select(
        [
            first["panel_role"].astype(str).eq(READY_ROLE),
            first["panel_role"].astype(str).eq(CONTROL_ROLE),
        ],
        [2, 0],
        default=0,
    )
    first["counts_as_fresh_positive_evidence"] = first["panel_role"].astype(str).eq(
        READY_ROLE
    )
    first["counts_as_control_guard"] = first["panel_role"].astype(str).eq(CONTROL_ROLE)
    first["pair_context_evaluable"] = first["allowed_execution_unit_count"].astype(int).ge(3)
    first["single_allowed_start_caveat"] = first["allowed_execution_unit_count"].astype(
        int
    ).eq(1)
    first["conditional_start_only_caveat"] = True
    first["stable_positive_generality_allowed"] = False
    first["direct_dependency_generality_allowed"] = False
    first["branch_generality_allowed"] = False
    first["wall_claim_allowed_after_readout"] = False
    first["method_claim_allowed_after_readout"] = False
    first["quality_cost_claim_allowed_after_readout"] = False
    first["bridge_release_regime"] = first["bridge_release_lift_proxy"].map(
        lambda value: _regime(value, low=0.3, high=0.7)
    )
    first["direct_dependency_regime"] = first["direct_dependency_proxy"].map(
        lambda value: _regime(value, low=0.3, high=0.7)
    )
    first["readout_success_requires"] = np.where(
        first["panel_role"].astype(str).eq(READY_ROLE),
        (
            "source_start_support_pass;post_start_endpoint_continuity_pass;"
            "target_final_continuity_pass;direct_edge_retention_pass;"
            "control_traps_closed_at_panel_level"
        ),
        "control_trap_positive_leak_false;wall_claim_allowed_false",
    )
    first["overclaim_guardrail"] = np.where(
        first["single_allowed_start_caveat"].astype(bool),
        "route-local only because this pair has one allowed start",
        "conditional allowed-start evidence only; blocked starts are not evidence",
    )
    first["run_status"] = RUN_STATUS
    first["claim_boundary"] = CLAIM_BOUNDARY
    sort_cols = ["panel_role", "local_pair_id", "start_condition"]
    return first.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)


def _build_pair_readout_rows(route_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pair_id, group in route_rows.groupby("local_pair_id", sort=True):
        first = group.iloc[0]
        route_count = int(len(group))
        pair_context_evaluable = bool(group["pair_context_evaluable"].map(_as_bool).all())
        rows.append(
            {
                "local_pair_id": str(pair_id),
                "branch": first.get("branch", ""),
                "panel_role": first["panel_role"],
                "evidence_role": first["evidence_role"],
                "validation_stratum": first.get("validation_stratum", ""),
                "limitation_axis_resolved": first.get("limitation_axis_resolved", ""),
                "route_readout_row_count": route_count,
                "start_conditions": _join_values(group["start_condition"]),
                "allowed_execution_unit_count": int(
                    first.get("allowed_execution_unit_count", route_count)
                ),
                "pair_context_evaluable": pair_context_evaluable,
                "single_allowed_start_caveat": bool(
                    group["single_allowed_start_caveat"].map(_as_bool).any()
                ),
                "bridge_release_lift_proxy": first.get("bridge_release_lift_proxy"),
                "direct_dependency_proxy": first.get("direct_dependency_proxy"),
                "bridge_release_regime": first["bridge_release_regime"],
                "direct_dependency_regime": first["direct_dependency_regime"],
                "control_trap_id": first.get("control_trap_id", ""),
                "max_claim_level_if_success": int(group["max_claim_level_if_success"].max()),
                "pair_readout_claim_ceiling": (
                    "conditional_axis_b_screen_only"
                    if str(first["panel_role"]) == READY_ROLE
                    else "control_false_positive_guard_only"
                ),
                "post_first_pass_escalation_candidate": str(pair_id)
                in {"local_pair_020"},
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["panel_role", "local_pair_id"], kind="mergesort"
    ).reset_index(drop=True)


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
        "run_status": RUN_STATUS,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _gate_matrix(
    *,
    panel_gates: pd.DataFrame,
    panel_field_rows: pd.DataFrame,
    claim_ladder_rows: pd.DataFrame,
    readout_field_rows: pd.DataFrame,
    control_trap_rows: pd.DataFrame,
    aggregation_rule_rows: pd.DataFrame,
    outcome_taxonomy_rows: pd.DataFrame,
    route_rows: pd.DataFrame,
    pair_rows: pd.DataFrame,
    readout_order_rows: pd.DataFrame,
) -> pd.DataFrame:
    ready_routes = route_rows[route_rows["panel_role"].astype(str).eq(READY_ROLE)]
    control_routes = route_rows[route_rows["panel_role"].astype(str).eq(CONTROL_ROLE)]
    ready_pairs = pair_rows[pair_rows["panel_role"].astype(str).eq(READY_ROLE)]
    control_pairs = pair_rows[pair_rows["panel_role"].astype(str).eq(CONTROL_ROLE)]
    return pd.DataFrame(
        [
            _gate_row(
                "G1_upstream_fresh_panel_gates_pass",
                "Did the upstream fresh Axis B panel gates pass?",
                _count_dict(panel_gates["gate_status"]),
                "all upstream panel gates pass",
                bool(panel_gates["gate_status"].astype(str).eq("pass").all()),
            ),
            _gate_row(
                "G2_first_pass_rows_isolated",
                "Are only the 36 first-pass fresh route rows included?",
                {
                    "route_rows": int(len(route_rows)),
                    "ready_routes": int(len(ready_routes)),
                    "control_routes": int(len(control_routes)),
                    "panel_roles": _count_dict(route_rows["panel_role"]),
                },
                "36 rows: 16 ready-like and 20 controls",
                int(len(route_rows)) == 36
                and int(len(ready_routes)) == 16
                and int(len(control_routes)) == 20,
            ),
            _gate_row(
                "G3_calibration_excluded_from_first_pass",
                "Are carryover calibration rows excluded from first-pass readout rows?",
                _count_dict(route_rows["panel_role"]),
                "zero carryover_axis_b_calibration_pair rows",
                not bool(
                    route_rows["panel_role"]
                    .astype(str)
                    .eq("carryover_axis_b_calibration_pair")
                    .any()
                ),
            ),
            _gate_row(
                "G4_claim_ladder_bounds_first_pass_claims",
                "Is the claim ladder explicit and does it close basin/pathway generality?",
                claim_ladder_rows[
                    ["claim_level", "claim_level_id", "allowed_as_first_pass_claim"]
                ].to_dict(orient="records"),
                "levels 0-5 materialized; level 5 not allowed as first-pass claim",
                set(claim_ladder_rows["claim_level"].astype(int)) == set(range(6))
                and not bool(
                    claim_ladder_rows[
                        claim_ladder_rows["claim_level"].astype(int).eq(5)
                    ]["allowed_as_first_pass_claim"].map(_as_bool).any()
                ),
            ),
            _gate_row(
                "G5_split_readout_fields_required",
                "Are source-start, post-start, target-final, and direct-edge fields required?",
                _count_dict(readout_field_rows["field_family"]),
                "required split fields are present",
                {
                    "source_start",
                    "interior_endpoint",
                    "physical_direct_path",
                    "control_guard",
                    "claim_boundary",
                }.issubset(set(readout_field_rows["field_family"].astype(str))),
            ),
            _gate_row(
                "G6_control_traps_cover_four_failure_modes",
                "Are all four first-pass control traps materialized?",
                sorted(control_trap_rows["control_trap_family"].astype(str).tolist()),
                "target saturation, latent source absence, hard no-release, and coupled failure",
                {
                    "target_saturated_no_handle",
                    "latent_release_without_original_coassigned_source",
                    "hard_no_release_control",
                    "coupled_direct_bridge_failure",
                }.issubset(set(control_trap_rows["control_trap_family"].astype(str))),
            ),
            _gate_row(
                "G7_aggregation_rules_prevent_average_overread",
                "Do aggregation rules force route/start/pair/control/panel separation?",
                _count_dict(aggregation_rule_rows["aggregation_level"]),
                "route, start, pair, control_trap, panel, and escalation rules exist",
                {
                    "route",
                    "start",
                    "pair",
                    "control_trap",
                    "panel",
                    "post_first_pass",
                }.issubset(set(aggregation_rule_rows["aggregation_level"].astype(str))),
            ),
            _gate_row(
                "G8_outcome_taxonomy_has_failure_and_limit_classes",
                "Does the taxonomy include false-positive, conditional, branch, direct-dependency, null, and incomplete outcomes?",
                sorted(outcome_taxonomy_rows["outcome_class"].astype(str).tolist()),
                "success, limited-success, false-positive, null, and incomplete outcomes exist",
                {
                    "false_positive_risk_control_leak",
                    "conditional_start_artifact",
                    "direct_dependency_unresolved",
                    "branch_generalization_unresolved",
                    "null_ready_like_target_failure",
                    "execution_incomplete",
                }.issubset(set(outcome_taxonomy_rows["outcome_class"].astype(str))),
            ),
            _gate_row(
                "G9_conditional_and_single_start_caveats_explicit",
                "Are conditional-only and one-start caveats explicit?",
                {
                    "conditional_start_only_rows": int(
                        route_rows["conditional_start_only_caveat"].map(_as_bool).sum()
                    ),
                    "single_allowed_start_rows": int(
                        route_rows["single_allowed_start_caveat"].map(_as_bool).sum()
                    ),
                    "single_start_pairs": pair_rows.loc[
                        pair_rows["single_allowed_start_caveat"].map(_as_bool),
                        "local_pair_id",
                    ].tolist(),
                },
                "all rows conditional-only; local_pair_003 is one-start caveat",
                int(route_rows["conditional_start_only_caveat"].map(_as_bool).sum())
                == int(len(route_rows))
                and pair_rows.loc[
                    pair_rows["single_allowed_start_caveat"].map(_as_bool),
                    "local_pair_id",
                ].tolist()
                == ["local_pair_003"],
            ),
            _gate_row(
                "G10_overclaim_flags_closed",
                "Are stable-positive, direct-dependent, branch, wall, method, and quality/cost claims closed?",
                {
                    "stable_positive_generality_allowed": int(
                        route_rows["stable_positive_generality_allowed"].map(_as_bool).sum()
                    ),
                    "direct_dependency_generality_allowed": int(
                        route_rows["direct_dependency_generality_allowed"].map(_as_bool).sum()
                    ),
                    "branch_generality_allowed": int(
                        route_rows["branch_generality_allowed"].map(_as_bool).sum()
                    ),
                    "wall_claim_allowed_after_readout": int(
                        route_rows["wall_claim_allowed_after_readout"].map(_as_bool).sum()
                    ),
                    "method_claim_allowed_after_readout": int(
                        route_rows["method_claim_allowed_after_readout"].map(_as_bool).sum()
                    ),
                    "quality_cost_claim_allowed_after_readout": int(
                        route_rows["quality_cost_claim_allowed_after_readout"].map(_as_bool).sum()
                    ),
                },
                "all overclaim flags are zero",
                not bool(
                    route_rows[
                        [
                            "stable_positive_generality_allowed",
                            "direct_dependency_generality_allowed",
                            "branch_generality_allowed",
                            "wall_claim_allowed_after_readout",
                            "method_claim_allowed_after_readout",
                            "quality_cost_claim_allowed_after_readout",
                        ]
                    ]
                    .apply(lambda column: column.map(_as_bool))
                    .any()
                    .any()
                ),
            ),
            _gate_row(
                "G11_readout_order_controls_first",
                "Does readout order inspect controls before ready-like positives?",
                readout_order_rows[
                    ["readout_order", "readout_step"]
                ].to_dict(orient="records"),
                "control_trap_scan is first",
                str(readout_order_rows.sort_values("readout_order").iloc[0]["readout_step"])
                == "control_trap_scan",
            ),
            _gate_row(
                "G12_no_new_leiden_execution",
                "Is this a readout contract rather than a new route execution?",
                RUN_STATUS,
                "design/materialization only",
                True,
            ),
        ]
    )


def _status(gates: pd.DataFrame) -> str:
    if not bool(gates["gate_status"].astype(str).eq("pass").all()):
        return "fresh_axis_b_first_pass_readout_contract_gate_failed"
    return "fresh_axis_b_first_pass_readout_contract_ready_controls_first_claim_limited"


def _summary(
    *,
    output_dir: Path,
    panel_dir: Path,
    panel_summary: dict[str, Any],
    claim_ladder_rows: pd.DataFrame,
    readout_field_rows: pd.DataFrame,
    control_trap_rows: pd.DataFrame,
    route_rows: pd.DataFrame,
    pair_rows: pd.DataFrame,
    gates: pd.DataFrame,
) -> dict[str, Any]:
    ready_routes = route_rows[route_rows["panel_role"].astype(str).eq(READY_ROLE)]
    control_routes = route_rows[route_rows["panel_role"].astype(str).eq(CONTROL_ROLE)]
    return {
        "schema": "nanoclustering_g4_8_fresh_axis_b_first_pass_readout_summary.v1",
        "status": _status(gates),
        "run_status": RUN_STATUS,
        "claim_boundary": CLAIM_BOUNDARY,
        "panel_dir": str(panel_dir),
        "output_dir": str(output_dir),
        "upstream_panel_status": panel_summary.get("status"),
        "claim_ladder_row_count": int(len(claim_ladder_rows)),
        "readout_field_row_count": int(len(readout_field_rows)),
        "control_trap_row_count": int(len(control_trap_rows)),
        "first_pass_route_row_count": int(len(route_rows)),
        "first_pass_pair_count": int(pair_rows["local_pair_id"].nunique()),
        "ready_like_route_row_count": int(len(ready_routes)),
        "control_route_row_count": int(len(control_routes)),
        "ready_like_pair_count": int(
            pair_rows["panel_role"].astype(str).eq(READY_ROLE).sum()
        ),
        "control_pair_count": int(
            pair_rows["panel_role"].astype(str).eq(CONTROL_ROLE).sum()
        ),
        "panel_role_counts": _count_dict(pair_rows["panel_role"]),
        "route_panel_role_counts": _count_dict(route_rows["panel_role"]),
        "route_branch_counts": _count_dict(route_rows["branch"]),
        "route_start_condition_counts": _count_dict(route_rows["start_condition"]),
        "direct_dependency_regime_counts": _count_dict(
            pair_rows["direct_dependency_regime"]
        ),
        "bridge_release_regime_counts": _count_dict(pair_rows["bridge_release_regime"]),
        "single_allowed_start_pairs": pair_rows.loc[
            pair_rows["single_allowed_start_caveat"].map(_as_bool),
            "local_pair_id",
        ].tolist(),
        "max_allowed_first_pass_claim_level": int(
            claim_ladder_rows.loc[
                claim_ladder_rows["allowed_as_first_pass_claim"].map(_as_bool),
                "claim_level",
            ].max()
        ),
        "gate_status_counts": _count_dict(gates["gate_status"]),
        "failed_gates": gates.loc[
            ~gates["gate_status"].astype(str).eq("pass"),
            "gate_id",
        ].tolist(),
        "interpretation": (
            "This readout contract makes the first-pass result a conditional "
            "ready-like Axis B screen. It cannot by itself claim stable-positive "
            "generality, branch generality, direct-dependent generality, wall "
            "evidence, method success, or quality/cost advantage. Controls must "
            "be inspected before ready-like positives."
        ),
        "recommended_next_gate": (
            "Execute or simulate only the 36 first-pass rows. Fill required "
            "source-start, post-start, target-final, direct-edge, control-trap, "
            "and outcome fields. After inspection, decide whether boundary "
            "sentinel local_pair_020 or reserve controls should open."
        ),
        "written_artifacts": [
            CLAIM_LADDER_ROWS_CSV,
            READOUT_FIELD_ROWS_CSV,
            CONTROL_TRAP_ROWS_CSV,
            AGGREGATION_RULE_ROWS_CSV,
            OUTCOME_TAXONOMY_ROWS_CSV,
            ROUTE_READOUT_ROWS_CSV,
            PAIR_READOUT_ROWS_CSV,
            READOUT_ORDER_ROWS_CSV,
            GATE_MATRIX_CSV,
            CONFIG_JSON,
            SUMMARY_JSON,
            REPORT_MD,
        ],
    }


def _write_report(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    claim_ladder_rows: pd.DataFrame,
    readout_field_rows: pd.DataFrame,
    control_trap_rows: pd.DataFrame,
    aggregation_rule_rows: pd.DataFrame,
    outcome_taxonomy_rows: pd.DataFrame,
    route_rows: pd.DataFrame,
    pair_rows: pd.DataFrame,
    readout_order_rows: pd.DataFrame,
    gates: pd.DataFrame,
) -> None:
    lines = [
        "# NanoClustering G4.8 Fresh Axis B First-Pass Readout Contract",
        "",
        f"- status: `{summary['status']}`",
        f"- first_pass_route_row_count: {summary['first_pass_route_row_count']}",
        f"- ready_like_route_row_count: {summary['ready_like_route_row_count']}",
        f"- control_route_row_count: {summary['control_route_row_count']}",
        f"- first_pass_pair_count: {summary['first_pass_pair_count']}",
        f"- route_branch_counts: {summary['route_branch_counts']}",
        f"- route_start_condition_counts: {summary['route_start_condition_counts']}",
        f"- direct_dependency_regime_counts: {summary['direct_dependency_regime_counts']}",
        f"- single_allowed_start_pairs: {summary['single_allowed_start_pairs']}",
        f"- max_allowed_first_pass_claim_level: {summary['max_allowed_first_pass_claim_level']}",
        f"- gate_status_counts: {summary['gate_status_counts']}",
        f"- failed_gates: {summary['failed_gates']}",
        f"- interpretation: {summary['interpretation']}",
        f"- recommended_next_gate: {summary['recommended_next_gate']}",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Claim Ladder",
        "",
        _markdown_table(
            claim_ladder_rows,
            [
                "claim_level",
                "claim_level_id",
                "claim_question",
                "allowed_as_first_pass_claim",
                "claim_boundary",
            ],
            max_rows=10,
        ),
        "",
        "## Required Readout Fields",
        "",
        _markdown_table(
            readout_field_rows,
            [
                "readout_field",
                "field_type",
                "field_family",
                "required_for_first_pass_readout",
                "field_question",
                "missing_value_action",
            ],
            max_rows=20,
        ),
        "",
        "## Control Traps",
        "",
        _markdown_table(
            control_trap_rows,
            [
                "local_pair_id",
                "branch",
                "control_trap_id",
                "control_trap_family",
                "trap_question",
                "expected_guard_outcome",
            ],
            max_rows=10,
        ),
        "",
        "## Pair Readout Rows",
        "",
        _markdown_table(
            pair_rows,
            [
                "local_pair_id",
                "branch",
                "panel_role",
                "route_readout_row_count",
                "start_conditions",
                "pair_context_evaluable",
                "single_allowed_start_caveat",
                "bridge_release_regime",
                "direct_dependency_regime",
                "pair_readout_claim_ceiling",
            ],
            max_rows=20,
        ),
        "",
        "## First-Pass Route Rows",
        "",
        _markdown_table(
            route_rows,
            [
                "readout_plan_id",
                "local_pair_id",
                "start_condition",
                "branch",
                "evidence_role",
                "max_claim_level_if_success",
                "single_allowed_start_caveat",
                "direct_dependency_regime",
                "readout_success_requires",
            ],
            max_rows=50,
        ),
        "",
        "## Aggregation Rules",
        "",
        _markdown_table(
            aggregation_rule_rows,
            [
                "aggregation_rule_id",
                "aggregation_level",
                "rule",
                "anti_overclaim_boundary",
            ],
            max_rows=20,
        ),
        "",
        "## Outcome Taxonomy",
        "",
        _markdown_table(
            outcome_taxonomy_rows,
            [
                "outcome_class",
                "trigger_condition",
                "allowed_interpretation",
                "blocked_interpretation",
            ],
            max_rows=20,
        ),
        "",
        "## Readout Order",
        "",
        _markdown_table(
            readout_order_rows,
            [
                "readout_order",
                "readout_step",
                "instruction",
                "why_first_pass_requires_this",
            ],
            max_rows=10,
        ),
        "",
        "## Gate Matrix",
        "",
        _markdown_table(
            gates,
            ["gate_id", "gate_status", "observed", "minimum_or_rule", "question"],
            max_rows=20,
        ),
        "",
        "## Boundary",
        "",
        (
            "This artifact is a readout contract. It opens no route execution and "
            "keeps wall, stable-positive generality, direct-dependent generality, "
            "branch generality, method, and quality/cost claims closed."
        ),
        "",
    ]
    (output_dir / REPORT_MD).write_text("\n".join(lines), encoding="utf-8")


def run(*, panel_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    panel_pair_rows = _read_csv(panel_dir / PANEL_PAIR_ROWS_CSV)
    panel_route_plan_rows = _read_csv(panel_dir / PANEL_ROUTE_PLAN_ROWS_CSV)
    panel_batch_rows = _read_csv(panel_dir / PANEL_BATCH_PLAN_ROWS_CSV)
    panel_field_rows = _read_csv(panel_dir / PANEL_FIELD_CONTRACT_ROWS_CSV)
    panel_gates = _read_csv(panel_dir / PANEL_GATE_MATRIX_CSV)
    panel_summary = json.loads((panel_dir / PANEL_SUMMARY_JSON).read_text(encoding="utf-8"))
    # Validate the upstream batch surface is present even though this script
    # derives rows from the route plan.
    if int(panel_batch_rows["new_route_execution_row_count"].sum()) != 36:
        raise ValueError("Fresh panel first-pass batch is not 36 route rows")

    claim_ladder_rows = _claim_ladder_rows()
    readout_field_rows = _readout_field_rows()
    control_trap_rows = _control_trap_rows(panel_pair_rows)
    aggregation_rule_rows = _aggregation_rule_rows()
    outcome_taxonomy_rows = _outcome_taxonomy_rows()
    readout_order_rows = _readout_order_rows()
    route_rows = _build_first_pass_route_rows(
        panel_route_plan_rows,
        panel_pair_rows,
        control_trap_rows,
    )
    pair_rows = _build_pair_readout_rows(route_rows)
    gates = _gate_matrix(
        panel_gates=panel_gates,
        panel_field_rows=panel_field_rows,
        claim_ladder_rows=claim_ladder_rows,
        readout_field_rows=readout_field_rows,
        control_trap_rows=control_trap_rows,
        aggregation_rule_rows=aggregation_rule_rows,
        outcome_taxonomy_rows=outcome_taxonomy_rows,
        route_rows=route_rows,
        pair_rows=pair_rows,
        readout_order_rows=readout_order_rows,
    )
    summary = _summary(
        output_dir=output_dir,
        panel_dir=panel_dir,
        panel_summary=panel_summary,
        claim_ladder_rows=claim_ladder_rows,
        readout_field_rows=readout_field_rows,
        control_trap_rows=control_trap_rows,
        route_rows=route_rows,
        pair_rows=pair_rows,
        gates=gates,
    )
    config = {
        "schema": "nanoclustering_g4_8_fresh_axis_b_first_pass_readout_config.v1",
        "panel_dir": str(panel_dir),
        "output_dir": str(output_dir),
        "ready_role": READY_ROLE,
        "control_role": CONTROL_ROLE,
        "run_status": RUN_STATUS,
        "claim_boundary": CLAIM_BOUNDARY,
    }

    _write_csv(claim_ladder_rows, output_dir / CLAIM_LADDER_ROWS_CSV)
    _write_csv(readout_field_rows, output_dir / READOUT_FIELD_ROWS_CSV)
    _write_csv(control_trap_rows, output_dir / CONTROL_TRAP_ROWS_CSV)
    _write_csv(aggregation_rule_rows, output_dir / AGGREGATION_RULE_ROWS_CSV)
    _write_csv(outcome_taxonomy_rows, output_dir / OUTCOME_TAXONOMY_ROWS_CSV)
    _write_csv(route_rows, output_dir / ROUTE_READOUT_ROWS_CSV)
    _write_csv(pair_rows, output_dir / PAIR_READOUT_ROWS_CSV)
    _write_csv(readout_order_rows, output_dir / READOUT_ORDER_ROWS_CSV)
    _write_csv(gates, output_dir / GATE_MATRIX_CSV)
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(
        output_dir=output_dir,
        summary=summary,
        claim_ladder_rows=claim_ladder_rows,
        readout_field_rows=readout_field_rows,
        control_trap_rows=control_trap_rows,
        aggregation_rule_rows=aggregation_rule_rows,
        outcome_taxonomy_rows=outcome_taxonomy_rows,
        route_rows=route_rows,
        pair_rows=pair_rows,
        readout_order_rows=readout_order_rows,
        gates=gates,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-dir", type=Path, default=DEFAULT_PANEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run(panel_dir=args.panel_dir, output_dir=args.output_dir)
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
