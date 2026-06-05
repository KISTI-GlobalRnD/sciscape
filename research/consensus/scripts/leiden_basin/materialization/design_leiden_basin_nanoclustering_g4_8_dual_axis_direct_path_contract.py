#!/usr/bin/env python3
"""Design the G4.8 dual-axis direct-path contract.

The first direct-path contract treated ``unknown_new_endpoint`` as a strict
same-seed anchor-consistency failure. The cross-seed endpoint atlas showed that
all such unknown labels are pair-level known signatures. This v2 contract keeps
the strict same-seed axis, but adds a separate pair-level endpoint-continuity
axis.

It does not run Leiden, broaden route execution, promote walls, evaluate
quality/cost value, replay full NanoClustering, or claim method success.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from audit_leiden_basin_nanoclustering_g4_8_cross_seed_endpoint_atlas import (
    CLAIM_BOUNDARY as ATLAS_CLAIM_BOUNDARY,
    CONTRACT_ROWS_CSV as ATLAS_CONTRACT_ROWS_CSV,
    DEFAULT_OUTPUT_DIR as DEFAULT_ATLAS_DIR,
    GATE_MATRIX_CSV as ATLAS_GATE_MATRIX_CSV,
    SIGNATURE_ROWS_CSV as ATLAS_SIGNATURE_ROWS_CSV,
)
from design_leiden_basin_nanoclustering_g4_8_direct_path_acceptance_contract import (
    CLAIM_BOUNDARY as DIRECT_CLAIM_BOUNDARY,
    CONTRACT_ROWS_CSV as DIRECT_CONTRACT_ROWS_CSV,
    DEFAULT_OUTPUT_DIR as DEFAULT_DIRECT_DIR,
    GATE_MATRIX_CSV as DIRECT_GATE_MATRIX_CSV,
    SEED_EVIDENCE_ROWS_CSV as DIRECT_SEED_ROWS_CSV,
)
from run_leiden_basin_nanoclustering_g4_8_scoped_pathway_probe_trace import (
    DEFAULT_OUTPUT_DIR as DEFAULT_TRACE_DIR,
    TRACE_ROWS_CSV,
)
from run_leiden_basin_nanoclustering_role_local_route_pilot import (
    BASE_RESULT_DIR,
    _json_safe,
    _read_csv,
    _write_csv,
)


DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_dual_axis_direct_path_contract_gamma1e5_20260604"
)

PRIMARY_ROUTE_FAMILY = "bridge_release_interpolation_probe"
RUN_STATUS = "designed_nanoclustering_g4_8_dual_axis_direct_path_contract"
CLAIM_BOUNDARY = (
    "NanoClustering G4.8 dual-axis direct-path contract design only; reads the "
    "existing direct-path contract, cross-seed endpoint atlas, and scoped route "
    "trace. It separates same-seed anchor consistency from pair-level endpoint "
    "continuity. It does not run Leiden, broaden route execution, promote "
    "walls, evaluate quality/cost value, replay full NanoClustering, or claim "
    "method or algorithm success."
)

RULE_ROWS_CSV = "nanoclustering_g4_8_dual_axis_direct_path_contract_rule_rows.csv"
SEED_AXIS_ROWS_CSV = "nanoclustering_g4_8_dual_axis_direct_path_contract_seed_axis_rows.csv"
CONTRACT_AXIS_ROWS_CSV = (
    "nanoclustering_g4_8_dual_axis_direct_path_contract_contract_axis_rows.csv"
)
PAIR_AXIS_ROWS_CSV = "nanoclustering_g4_8_dual_axis_direct_path_contract_pair_axis_rows.csv"
GATE_MATRIX_CSV = "nanoclustering_g4_8_dual_axis_direct_path_contract_gate_matrix.csv"
CONFIG_JSON = "nanoclustering_g4_8_dual_axis_direct_path_contract_config.json"
SUMMARY_JSON = "nanoclustering_g4_8_dual_axis_direct_path_contract_summary.json"
REPORT_MD = "nanoclustering_g4_8_dual_axis_direct_path_contract_report.md"

TRUE_NOVEL_ROLE = "pair_level_true_novel_signature"


AXIS_RULES = (
    {
        "rule_id": "A1_same_seed_anchor_consistency_axis",
        "axis": "A_same_seed_strict",
        "rule_question": "Does the route stay on same-seed known anchors?",
        "seed_level_requirement": (
            "D1-D6 from the v1 direct-path contract all pass, including zero "
            "same-seed unknown/support-incompatible steps"
        ),
        "contract_level_requirement": "all seeds pass the v1 strict same-seed route rules",
        "claim_effect": "strict direct-path acceptance remains closed on current evidence",
    },
    {
        "rule_id": "A2_same_seed_contract_aggregation",
        "axis": "A_same_seed_strict",
        "rule_question": "Does every seed in the start-conditioned contract pass Axis A?",
        "seed_level_requirement": "axis_a_same_seed_seed_pass == true",
        "contract_level_requirement": "axis_a_seed_pass_count == seed_count",
        "claim_effect": "same-seed strictness is reported but not merged with Axis B",
    },
    {
        "rule_id": "B1_pair_level_endpoint_known",
        "axis": "B_pair_level_endpoint_atlas",
        "rule_question": "Are all route endpoints pair-level known signatures?",
        "seed_level_requirement": "true_novel_endpoint_step_count == 0",
        "contract_level_requirement": "all seeds have zero true-novel endpoint steps",
        "claim_effect": "same-seed unknown labels are topology blockers only if pair-level novel",
    },
    {
        "rule_id": "B2_pair_level_source_start",
        "axis": "B_pair_level_endpoint_atlas",
        "rule_question": "Does the route start in a pair-level source signature?",
        "seed_level_requirement": "first endpoint atlas role contains original_source",
        "contract_level_requirement": "all seeds start from a pair-level source signature",
        "claim_effect": "keeps endpoint continuity tied to source-to-target routes",
    },
    {
        "rule_id": "B3_pair_level_target_reached",
        "axis": "B_pair_level_endpoint_atlas",
        "rule_question": "Does the route end in a pair-level drop-bridge target signature?",
        "seed_level_requirement": (
            "last endpoint atlas role contains drop_bridge_target and the expected "
            "same-seed final anchor is reached"
        ),
        "contract_level_requirement": "all seeds end at the pair-level target signature",
        "claim_effect": "keeps pair-level continuity from being a generic known-endpoint claim",
    },
    {
        "rule_id": "B4_physical_direct_edge_retained",
        "axis": "B_pair_level_endpoint_atlas",
        "rule_question": "Is the direct pair edge retained while endpoint continuity is tested?",
        "seed_level_requirement": "physical_direct_edge_retained_all_steps == true",
        "contract_level_requirement": "all seeds retain the direct pair edge",
        "claim_effect": "retains the physical pathway condition from v1",
    },
    {
        "rule_id": "B5_same_seed_flags_are_diagnostics",
        "axis": "B_pair_level_endpoint_atlas",
        "rule_question": "Are same-seed unknown/support flags kept as diagnostics rather than topology failure?",
        "seed_level_requirement": (
            "same_seed_unknown/support flags do not block Axis B when their "
            "endpoint signatures are pair-level known"
        ),
        "contract_level_requirement": (
            "same-seed strictness is reported separately and cannot block the "
            "pair-level continuity axis by itself"
        ),
        "claim_effect": "prevents over-pruning pair-level pathway evidence",
    },
    {
        "rule_id": "C1_objective_wall_separation",
        "axis": "C_claim_boundary",
        "rule_question": "Are objective recovery and wall claims kept separate?",
        "seed_level_requirement": "objective recovery is reported but not used for Axis A or B",
        "contract_level_requirement": "wall_contract_ready remains false for every contract",
        "claim_effect": "Axis B continuity cannot promote wall or quality claims",
    },
    {
        "rule_id": "C2_no_new_execution_or_method_claim",
        "axis": "C_claim_boundary",
        "rule_question": "Is this a contract redesign rather than a new run or method result?",
        "seed_level_requirement": "readout-only from existing artifacts",
        "contract_level_requirement": "no new Leiden execution, no full replay, no algorithm claim",
        "claim_effect": "limits this artifact to methodology clarification",
    },
)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _count_dict(series: pd.Series) -> dict[str, int]:
    if series.empty:
        return {}
    return {str(key): int(value) for key, value in series.value_counts(dropna=False).items()}


def _join_steps(steps: list[int]) -> str:
    return ";".join(str(int(step)) for step in sorted(set(steps)))


def _role_is_source(role: Any) -> bool:
    return "original_source" in str(role)


def _role_is_target(role: Any) -> bool:
    return "drop_bridge_target" in str(role)


def _role_is_true_novel(role: Any) -> bool:
    return str(role) == TRUE_NOVEL_ROLE or str(role).strip() == "" or pd.isna(role)


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


def _rule_rows() -> pd.DataFrame:
    rows = pd.DataFrame(list(AXIS_RULES))
    rows["run_status"] = RUN_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


def _atlas_enriched_trace(trace_rows: pd.DataFrame, signature_rows: pd.DataFrame) -> pd.DataFrame:
    primary = trace_rows[
        trace_rows["planned_route_family"].astype(str).eq(PRIMARY_ROUTE_FAMILY)
    ].copy()
    vocab = signature_rows[
        [
            "local_pair_id",
            "result_endpoint_signature_id",
            "cross_seed_known_assignments",
            "cross_seed_known_signature",
            "endpoint_atlas_role",
        ]
    ].copy()
    return primary.merge(
        vocab,
        on=["local_pair_id", "result_endpoint_signature_id"],
        how="left",
        validate="many_to_one",
    )


def _seed_axis_rows(direct_seed_rows: pd.DataFrame, enriched_trace: pd.DataFrame) -> pd.DataFrame:
    key_cols = [
        "route_contract_id",
        "validation_unit_id",
        "local_pair_id",
        "start_condition",
        "seed",
    ]
    direct_lookup = {
        tuple(row[col] for col in key_cols): row
        for row in direct_seed_rows.to_dict(orient="records")
    }

    rows: list[dict[str, Any]] = []
    for keys, group in enriched_trace.groupby(key_cols, sort=False):
        ordered = group.sort_values("step_index", kind="mergesort").reset_index(drop=True)
        first = ordered.iloc[0]
        last = ordered.iloc[-1]
        direct = direct_lookup.get(tuple(keys))
        if direct is None:
            raise KeyError(f"Missing direct seed evidence for {keys}")

        roles = ordered["endpoint_atlas_role"]
        role_list = [str(role) for role in roles.tolist()]
        true_novel_mask = roles.map(_role_is_true_novel).astype(bool)
        source_mask = roles.map(_role_is_source).astype(bool)
        target_mask = roles.map(_role_is_target).astype(bool)
        same_seed_unknown_mask = ordered["endpoint_assignment_by_step"].astype(str).eq(
            "unknown_new_endpoint"
        )
        support_mask = ordered["support_incompatibility_check"].map(_as_bool).astype(bool)
        pair_known_mask = ~true_novel_mask
        source_steps = ordered.loc[source_mask, "step_index"].astype(int).tolist()
        target_steps = ordered.loc[target_mask, "step_index"].astype(int).tolist()
        first_source_step = None if not source_steps else int(min(source_steps))
        first_target_role_step = None if not target_steps else int(min(target_steps))
        last_target_role_step = None if not target_steps else int(max(target_steps))

        axis_a_seed_pass = _as_bool(direct["seed_direct_path_candidate"])
        axis_b_no_true_novel = int(true_novel_mask.sum()) == 0
        axis_b_source_start = _role_is_source(first["endpoint_atlas_role"])
        axis_b_target_final = (
            _role_is_target(last["endpoint_atlas_role"])
            and _as_bool(direct["expected_final_anchor_reached"])
        )
        axis_b_source_to_target = (
            first_source_step is not None
            and first_target_role_step is not None
            and first_source_step <= first_target_role_step
            and last_target_role_step == int(last["step_index"])
        )
        axis_b_direct_edge_retained = _as_bool(direct["d2_direct_edge_retained_pass"])
        axis_b_primary_scope = _as_bool(direct["d1_primary_scope_pass"])
        axis_b_seed_pass = all(
            [
                axis_b_primary_scope,
                axis_b_direct_edge_retained,
                axis_b_no_true_novel,
                axis_b_source_start,
                axis_b_target_final,
                axis_b_source_to_target,
            ]
        )

        if axis_b_seed_pass and not axis_a_seed_pass:
            axis_b_seed_status = (
                "pair_level_continuity_pass_same_seed_strictness_fails"
            )
        elif axis_b_seed_pass and axis_a_seed_pass:
            axis_b_seed_status = "pair_level_continuity_and_same_seed_strictness_pass"
        elif not axis_b_no_true_novel:
            axis_b_seed_status = "pair_level_continuity_blocked_by_true_novel_endpoint"
        elif not axis_b_source_start or not axis_b_target_final:
            axis_b_seed_status = "pair_level_continuity_blocked_by_source_target_role"
        else:
            axis_b_seed_status = "pair_level_continuity_blocked_by_path_condition"

        rows.append(
            {
                "route_contract_id": str(keys[0]),
                "validation_unit_id": str(keys[1]),
                "local_pair_id": str(keys[2]),
                "start_condition": str(keys[3]),
                "seed": int(keys[4]),
                "planned_route_family": PRIMARY_ROUTE_FAMILY,
                "route_step_count": int(len(ordered)),
                "axis_a_same_seed_seed_pass": bool(axis_a_seed_pass),
                "axis_a_same_seed_block_reason": str(
                    direct.get("direct_path_seed_block_reason", "")
                ),
                "axis_b_pair_level_continuity_seed_pass": bool(axis_b_seed_pass),
                "axis_b_seed_status": axis_b_seed_status,
                "axis_b_primary_scope_pass": bool(axis_b_primary_scope),
                "axis_b_direct_edge_retained_pass": bool(axis_b_direct_edge_retained),
                "axis_b_no_true_novel_endpoint_pass": bool(axis_b_no_true_novel),
                "axis_b_pair_level_source_start_pass": bool(axis_b_source_start),
                "axis_b_pair_level_target_final_pass": bool(axis_b_target_final),
                "axis_b_source_to_target_transition_pass": bool(axis_b_source_to_target),
                "pair_level_known_step_count": int(pair_known_mask.sum()),
                "true_novel_endpoint_step_count": int(true_novel_mask.sum()),
                "same_seed_unknown_step_count": int(same_seed_unknown_mask.sum()),
                "same_seed_unknown_but_pair_known_step_count": int(
                    (same_seed_unknown_mask & pair_known_mask).sum()
                ),
                "same_seed_support_incompatible_step_count": int(support_mask.sum()),
                "same_seed_support_but_pair_known_step_count": int(
                    (support_mask & pair_known_mask).sum()
                ),
                "pair_level_source_step_indices": _join_steps(source_steps),
                "pair_level_target_step_indices": _join_steps(target_steps),
                "first_pair_level_source_step": first_source_step,
                "first_pair_level_target_step": first_target_role_step,
                "last_pair_level_target_step": last_target_role_step,
                "endpoint_atlas_role_sequence": ";".join(role_list),
                "endpoint_atlas_role_counts": json.dumps(_count_dict(roles), sort_keys=True),
                "same_seed_endpoint_assignment_sequence": ";".join(
                    ordered["endpoint_assignment_by_step"].astype(str).tolist()
                ),
                "d1_primary_scope_pass": bool(_as_bool(direct["d1_primary_scope_pass"])),
                "d2_direct_edge_retained_pass": bool(
                    _as_bool(direct["d2_direct_edge_retained_pass"])
                ),
                "d3_source_start_known_pass": bool(
                    _as_bool(direct["d3_source_start_known_pass"])
                ),
                "d4_target_reached_known_pass": bool(
                    _as_bool(direct["d4_target_reached_known_pass"])
                ),
                "d5_no_intermediate_unknown_pass": bool(
                    _as_bool(direct["d5_no_intermediate_unknown_pass"])
                ),
                "d6_no_support_incompatibility_pass": bool(
                    _as_bool(direct["d6_no_support_incompatibility_pass"])
                ),
                "unknown_step_indices": str(direct.get("unknown_step_indices", "")),
                "support_incompatibility_step_indices": str(
                    direct.get("support_incompatibility_step_indices", "")
                ),
                "max_objective_debt_from_start": float(
                    direct["max_objective_debt_from_start"]
                ),
                "max_objective_recovery_from_min": float(
                    direct["max_objective_recovery_from_min"]
                ),
                "final_objective_delta_from_start": float(
                    direct["final_objective_delta_from_start"]
                ),
                "objective_shape_class": str(direct.get("objective_shape_class", "")),
                "pathway_shape_class": str(direct.get("pathway_shape_class", "")),
                "wall_seed_ready": False,
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["local_pair_id", "start_condition", "seed"],
        kind="mergesort",
    ).reset_index(drop=True)


def _contract_axis_rows(
    direct_contract_rows: pd.DataFrame,
    atlas_contract_rows: pd.DataFrame,
    seed_axis_rows: pd.DataFrame,
) -> pd.DataFrame:
    groups = (
        seed_axis_rows.groupby(
            ["route_contract_id", "validation_unit_id", "local_pair_id", "start_condition"],
            sort=False,
        )
        .agg(
            seed_count=("seed", "nunique"),
            axis_a_seed_pass_count=("axis_a_same_seed_seed_pass", "sum"),
            axis_b_seed_pass_count=("axis_b_pair_level_continuity_seed_pass", "sum"),
            same_seed_unknown_seed_count=(
                "same_seed_unknown_step_count",
                lambda values: int((values.astype(int) > 0).sum()),
            ),
            same_seed_unknown_but_pair_known_seed_count=(
                "same_seed_unknown_but_pair_known_step_count",
                lambda values: int((values.astype(int) > 0).sum()),
            ),
            true_novel_endpoint_seed_count=(
                "true_novel_endpoint_step_count",
                lambda values: int((values.astype(int) > 0).sum()),
            ),
            same_seed_support_seed_count=(
                "same_seed_support_incompatible_step_count",
                lambda values: int((values.astype(int) > 0).sum()),
            ),
            objective_recovery_seed_count=(
                "max_objective_recovery_from_min",
                lambda values: int((values.astype(float) > 1e-9).sum()),
            ),
            objective_shape_class_counts=(
                "objective_shape_class",
                lambda values: json.dumps(_count_dict(values), sort_keys=True),
            ),
            pathway_shape_class_counts=(
                "pathway_shape_class",
                lambda values: json.dumps(_count_dict(values), sort_keys=True),
            ),
            axis_b_seed_status_counts=(
                "axis_b_seed_status",
                lambda values: json.dumps(_count_dict(values), sort_keys=True),
            ),
        )
        .reset_index()
    )
    direct_keep = [
        "route_contract_id",
        "seed_direct_path_candidate_count",
        "accepted_direct_path_contract",
        "direct_path_contract_status",
        "wall_contract_ready",
    ]
    groups = groups.merge(
        direct_contract_rows[direct_keep],
        on="route_contract_id",
        how="left",
        validate="one_to_one",
    )
    atlas_keep = [
        "route_contract_id",
        "same_seed_unknown_count",
        "cross_seed_known_unknown_count",
        "true_novel_unknown_count",
        "no_true_novel_unknown_endpoint",
        "endpoint_atlas_contract_status",
    ]
    groups = groups.merge(
        atlas_contract_rows[atlas_keep],
        on="route_contract_id",
        how="left",
        validate="one_to_one",
    )
    groups["axis_a_same_seed_contract_pass"] = groups["axis_a_seed_pass_count"].astype(
        int
    ).eq(groups["seed_count"].astype(int))
    groups["axis_b_pair_level_continuity_contract_pass"] = groups[
        "axis_b_seed_pass_count"
    ].astype(int).eq(groups["seed_count"].astype(int))
    groups["all_seed_objective_recovery_contract_pass"] = groups[
        "objective_recovery_seed_count"
    ].astype(int).eq(groups["seed_count"].astype(int))
    groups["wall_contract_ready_v2"] = False

    def status(row: pd.Series) -> str:
        if bool(row["axis_a_same_seed_contract_pass"]):
            return "strict_same_seed_direct_path_contract_passes_currently_unobserved"
        if bool(row["axis_b_pair_level_continuity_contract_pass"]):
            return "pair_level_continuity_pass_same_seed_strictness_closed_wall_closed"
        if int(row["true_novel_endpoint_seed_count"]) > 0:
            return "pair_level_continuity_blocked_by_true_novel_endpoint"
        return "pair_level_continuity_not_established"

    groups["dual_axis_contract_status"] = groups.apply(status, axis=1)
    groups["contract_claim_boundary_note"] = (
        "Axis B accepts pair-level endpoint continuity only. It does not override "
        "Axis A strict same-seed failure and cannot promote wall, quality, cost, "
        "full replay, or method claims."
    )
    groups["run_status"] = RUN_STATUS
    groups["claim_boundary"] = CLAIM_BOUNDARY
    return groups.sort_values(
        ["local_pair_id", "start_condition"],
        kind="mergesort",
    ).reset_index(drop=True)


def _pair_axis_rows(contract_axis_rows: pd.DataFrame) -> pd.DataFrame:
    rows = (
        contract_axis_rows.groupby("local_pair_id", sort=False)
        .agg(
            contract_count=("route_contract_id", "nunique"),
            seed_count=("seed_count", "sum"),
            axis_a_seed_pass_count=("axis_a_seed_pass_count", "sum"),
            axis_b_seed_pass_count=("axis_b_seed_pass_count", "sum"),
            axis_a_contract_pass_count=("axis_a_same_seed_contract_pass", "sum"),
            axis_b_contract_pass_count=(
                "axis_b_pair_level_continuity_contract_pass",
                "sum",
            ),
            same_seed_unknown_seed_count=("same_seed_unknown_seed_count", "sum"),
            same_seed_unknown_but_pair_known_seed_count=(
                "same_seed_unknown_but_pair_known_seed_count",
                "sum",
            ),
            true_novel_endpoint_seed_count=("true_novel_endpoint_seed_count", "sum"),
            objective_recovery_seed_count=("objective_recovery_seed_count", "sum"),
            all_seed_objective_recovery_contract_count=(
                "all_seed_objective_recovery_contract_pass",
                "sum",
            ),
            dual_axis_contract_status_counts=(
                "dual_axis_contract_status",
                lambda values: json.dumps(_count_dict(values), sort_keys=True),
            ),
        )
        .reset_index()
    )
    rows["pair_axis_status"] = rows.apply(
        lambda row: (
            "pair_level_continuity_open_same_seed_strictness_closed"
            if int(row["axis_b_contract_pass_count"]) == int(row["contract_count"])
            and int(row["axis_a_contract_pass_count"]) == 0
            else "pair_level_continuity_or_strictness_mixed"
        ),
        axis=1,
    )
    rows["wall_pair_ready"] = False
    rows["run_status"] = RUN_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows.sort_values("local_pair_id", kind="mergesort").reset_index(drop=True)


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
    direct_gates: pd.DataFrame,
    atlas_gates: pd.DataFrame,
    rule_rows: pd.DataFrame,
    seed_axis_rows: pd.DataFrame,
    contract_axis_rows: pd.DataFrame,
) -> pd.DataFrame:
    axis_a_seed_pass = int(seed_axis_rows["axis_a_same_seed_seed_pass"].astype(bool).sum())
    axis_b_seed_pass = int(
        seed_axis_rows["axis_b_pair_level_continuity_seed_pass"].astype(bool).sum()
    )
    axis_a_contract_pass = int(
        contract_axis_rows["axis_a_same_seed_contract_pass"].astype(bool).sum()
    )
    axis_b_contract_pass = int(
        contract_axis_rows["axis_b_pair_level_continuity_contract_pass"].astype(bool).sum()
    )
    true_novel_seed_count = int(
        (seed_axis_rows["true_novel_endpoint_step_count"].astype(int) > 0).sum()
    )
    unknown_pair_known_seed_count = int(
        (
            seed_axis_rows["same_seed_unknown_but_pair_known_step_count"].astype(int) > 0
        ).sum()
    )
    objective_recovery_seed_count = int(
        seed_axis_rows["max_objective_recovery_from_min"].astype(float).gt(1e-9).sum()
    )
    all_seed_recovery_contracts = int(
        contract_axis_rows["all_seed_objective_recovery_contract_pass"].astype(bool).sum()
    )
    return pd.DataFrame(
        [
            _gate_row(
                "G1_upstream_contract_and_atlas_gates_pass",
                "Did the upstream v1 direct-path contract and endpoint-atlas gates pass?",
                {
                    "direct_gate_status_counts": _count_dict(direct_gates["gate_status"]),
                    "atlas_gate_status_counts": _count_dict(atlas_gates["gate_status"]),
                },
                "all upstream direct-contract and endpoint-atlas gates pass",
                bool(direct_gates["gate_status"].astype(str).eq("pass").all())
                and bool(atlas_gates["gate_status"].astype(str).eq("pass").all()),
            ),
            _gate_row(
                "G2_dual_axis_rules_are_materialized",
                "Are same-seed strictness and pair-level continuity split into explicit axes?",
                f"rule_count={len(rule_rows)} axes={sorted(rule_rows['axis'].unique())}",
                "A, B, and claim-boundary C rules are materialized",
                len(rule_rows) == 9
                and set(rule_rows["axis"].astype(str))
                == {
                    "A_same_seed_strict",
                    "B_pair_level_endpoint_atlas",
                    "C_claim_boundary",
                },
            ),
            _gate_row(
                "G3_axis_a_strict_same_seed_result_preserved",
                "Does Axis A preserve the old strict same-seed closure result?",
                (
                    f"axis_a_seed_pass={axis_a_seed_pass}/80 "
                    f"axis_a_contract_pass={axis_a_contract_pass}/10"
                ),
                "53 seed passes and 0 contract passes under strict same-seed rules",
                axis_a_seed_pass == 53 and axis_a_contract_pass == 0,
            ),
            _gate_row(
                "G4_axis_b_pair_level_continuity_open",
                "Does Axis B show pair-level source-to-target continuity on current evidence?",
                (
                    f"axis_b_seed_pass={axis_b_seed_pass}/80 "
                    f"axis_b_contract_pass={axis_b_contract_pass}/10"
                ),
                "80 seed passes and 10 contract passes under pair-level endpoint continuity",
                axis_b_seed_pass == 80 and axis_b_contract_pass == 10,
            ),
            _gate_row(
                "G5_no_true_novel_pair_level_endpoints",
                "Are same-seed unknowns still free of true pair-level novel endpoints?",
                f"true_novel_endpoint_seed_count={true_novel_seed_count}",
                "zero seeds contain true-novel pair-level endpoints",
                true_novel_seed_count == 0,
            ),
            _gate_row(
                "G6_same_seed_flags_reinterpreted_not_discarded",
                "Are same-seed unknown/support flags retained as diagnostics?",
                (
                    f"same_seed_unknown_but_pair_known_seed_count="
                    f"{unknown_pair_known_seed_count}"
                ),
                "27 same-seed unknown seeds are pair-level known and reported separately",
                unknown_pair_known_seed_count == 27,
            ),
            _gate_row(
                "G7_objective_recovery_kept_out_of_pathway_axes",
                "Is objective recovery still separated from both pathway axes?",
                (
                    f"objective_recovery_seed_count={objective_recovery_seed_count} "
                    f"all_seed_recovery_contracts={all_seed_recovery_contracts}"
                ),
                "objective recovery is reported, not used for Axis A or Axis B acceptance",
                objective_recovery_seed_count == 8 and all_seed_recovery_contracts == 0,
            ),
            _gate_row(
                "G8_wall_claim_remains_closed",
                "Are wall claims still closed after Axis B is opened?",
                (
                    f"wall_ready_contracts="
                    f"{int(contract_axis_rows['wall_contract_ready_v2'].astype(bool).sum())}"
                ),
                "zero wall-ready contracts",
                not bool(contract_axis_rows["wall_contract_ready_v2"].astype(bool).any()),
            ),
            _gate_row(
                "G9_no_new_leiden_method_quality_or_full_replay_claim",
                "Are execution, method, quality/cost, and full-replay claims closed?",
                CLAIM_BOUNDARY,
                "claim boundary explicitly closed",
                True,
            ),
        ]
    )


def _summary(
    *,
    direct_dir: Path,
    atlas_dir: Path,
    trace_dir: Path,
    output_dir: Path,
    rule_rows: pd.DataFrame,
    seed_axis_rows: pd.DataFrame,
    contract_axis_rows: pd.DataFrame,
    pair_axis_rows: pd.DataFrame,
    gates: pd.DataFrame,
) -> dict[str, Any]:
    axis_a_seed_pass = int(seed_axis_rows["axis_a_same_seed_seed_pass"].astype(bool).sum())
    axis_b_seed_pass = int(
        seed_axis_rows["axis_b_pair_level_continuity_seed_pass"].astype(bool).sum()
    )
    axis_a_contract_pass = int(
        contract_axis_rows["axis_a_same_seed_contract_pass"].astype(bool).sum()
    )
    axis_b_contract_pass = int(
        contract_axis_rows["axis_b_pair_level_continuity_contract_pass"].astype(bool).sum()
    )
    same_seed_unknown_seed_count = int(
        (seed_axis_rows["same_seed_unknown_step_count"].astype(int) > 0).sum()
    )
    unknown_pair_known_seed_count = int(
        (
            seed_axis_rows["same_seed_unknown_but_pair_known_step_count"].astype(int) > 0
        ).sum()
    )
    true_novel_seed_count = int(
        (seed_axis_rows["true_novel_endpoint_step_count"].astype(int) > 0).sum()
    )
    objective_recovery_seed_count = int(
        seed_axis_rows["max_objective_recovery_from_min"].astype(float).gt(1e-9).sum()
    )
    wall_ready_contract_count = int(
        contract_axis_rows["wall_contract_ready_v2"].astype(bool).sum()
    )
    return {
        "schema": "nanoclustering_g4_8_dual_axis_direct_path_contract_summary.v1",
        "status": (
            "dual_axis_direct_path_contract_materialized_axis_a_closed_axis_b_open_wall_closed"
        ),
        "run_status": RUN_STATUS,
        "direct_contract_dir": str(direct_dir),
        "endpoint_atlas_dir": str(atlas_dir),
        "trace_dir": str(trace_dir),
        "output_dir": str(output_dir),
        "rule_count": int(len(rule_rows)),
        "seed_axis_row_count": int(len(seed_axis_rows)),
        "contract_axis_row_count": int(len(contract_axis_rows)),
        "pair_axis_row_count": int(len(pair_axis_rows)),
        "axis_a_same_seed_seed_pass_count": axis_a_seed_pass,
        "axis_b_pair_level_seed_pass_count": axis_b_seed_pass,
        "axis_a_same_seed_contract_pass_count": axis_a_contract_pass,
        "axis_b_pair_level_contract_pass_count": axis_b_contract_pass,
        "same_seed_unknown_seed_count": same_seed_unknown_seed_count,
        "same_seed_unknown_but_pair_known_seed_count": unknown_pair_known_seed_count,
        "true_novel_endpoint_seed_count": true_novel_seed_count,
        "objective_recovery_seed_count": objective_recovery_seed_count,
        "wall_ready_contract_count": wall_ready_contract_count,
        "dual_axis_contract_status_counts": _count_dict(
            contract_axis_rows["dual_axis_contract_status"]
        ),
        "pair_axis_status_counts": _count_dict(pair_axis_rows["pair_axis_status"]),
        "gate_status_counts": _count_dict(gates["gate_status"]),
        "failed_gates": gates.loc[
            ~gates["gate_status"].astype(str).eq("pass"),
            "gate_id",
        ].tolist(),
        "interpretation": (
            "The old strict same-seed direct-path closure is preserved as Axis A "
            "(53 of 80 seed passes, 0 of 10 contract passes). The cross-seed "
            "endpoint atlas opens a separate Axis B: all 80 seed-routes and all "
            "10 contracts preserve pair-level source-to-target endpoint continuity "
            "with no true-novel pair-level endpoint. This is a pathway-topology "
            "clarification, not wall or quality evidence."
        ),
        "recommended_next_gate": (
            "Predeclare a fresh validation panel or seed-anchor rotation for Axis B "
            "endpoint-atlas continuity. Keep Axis A, objective recovery, wall, cost, "
            "quality, and full-replay claims separate until independent gates pass."
        ),
        "direct_claim_boundary": DIRECT_CLAIM_BOUNDARY,
        "atlas_claim_boundary": ATLAS_CLAIM_BOUNDARY,
        "claim_boundary": CLAIM_BOUNDARY,
        "written_artifacts": [
            RULE_ROWS_CSV,
            SEED_AXIS_ROWS_CSV,
            CONTRACT_AXIS_ROWS_CSV,
            PAIR_AXIS_ROWS_CSV,
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
    rule_rows: pd.DataFrame,
    seed_axis_rows: pd.DataFrame,
    contract_axis_rows: pd.DataFrame,
    pair_axis_rows: pd.DataFrame,
    gates: pd.DataFrame,
) -> None:
    lines = [
        "# NanoClustering G4.8 Dual-Axis Direct-Path Contract",
        "",
        f"- status: `{summary['status']}`",
        f"- rule_count: {summary['rule_count']}",
        f"- seed_axis_row_count: {summary['seed_axis_row_count']}",
        f"- contract_axis_row_count: {summary['contract_axis_row_count']}",
        f"- pair_axis_row_count: {summary['pair_axis_row_count']}",
        (
            "- axis_a_same_seed_seed_pass_count: "
            f"{summary['axis_a_same_seed_seed_pass_count']}"
        ),
        (
            "- axis_b_pair_level_seed_pass_count: "
            f"{summary['axis_b_pair_level_seed_pass_count']}"
        ),
        (
            "- axis_a_same_seed_contract_pass_count: "
            f"{summary['axis_a_same_seed_contract_pass_count']}"
        ),
        (
            "- axis_b_pair_level_contract_pass_count: "
            f"{summary['axis_b_pair_level_contract_pass_count']}"
        ),
        f"- same_seed_unknown_seed_count: {summary['same_seed_unknown_seed_count']}",
        (
            "- same_seed_unknown_but_pair_known_seed_count: "
            f"{summary['same_seed_unknown_but_pair_known_seed_count']}"
        ),
        f"- true_novel_endpoint_seed_count: {summary['true_novel_endpoint_seed_count']}",
        f"- objective_recovery_seed_count: {summary['objective_recovery_seed_count']}",
        f"- wall_ready_contract_count: {summary['wall_ready_contract_count']}",
        f"- dual_axis_contract_status_counts: {summary['dual_axis_contract_status_counts']}",
        f"- gate_status_counts: {summary['gate_status_counts']}",
        f"- failed_gates: {summary['failed_gates']}",
        f"- interpretation: {summary['interpretation']}",
        f"- recommended_next_gate: {summary['recommended_next_gate']}",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Rules",
        "",
        _markdown_table(
            rule_rows,
            [
                "rule_id",
                "axis",
                "rule_question",
                "seed_level_requirement",
                "contract_level_requirement",
                "claim_effect",
            ],
            max_rows=20,
        ),
        "",
        "## Contract Axes",
        "",
        _markdown_table(
            contract_axis_rows,
            [
                "local_pair_id",
                "start_condition",
                "seed_count",
                "axis_a_seed_pass_count",
                "axis_b_seed_pass_count",
                "axis_a_same_seed_contract_pass",
                "axis_b_pair_level_continuity_contract_pass",
                "same_seed_unknown_seed_count",
                "same_seed_unknown_but_pair_known_seed_count",
                "true_novel_endpoint_seed_count",
                "objective_recovery_seed_count",
                "dual_axis_contract_status",
            ],
            max_rows=20,
        ),
        "",
        "## Pair Axes",
        "",
        _markdown_table(
            pair_axis_rows,
            [
                "local_pair_id",
                "contract_count",
                "seed_count",
                "axis_a_seed_pass_count",
                "axis_b_seed_pass_count",
                "axis_a_contract_pass_count",
                "axis_b_contract_pass_count",
                "same_seed_unknown_seed_count",
                "same_seed_unknown_but_pair_known_seed_count",
                "true_novel_endpoint_seed_count",
                "objective_recovery_seed_count",
                "pair_axis_status",
            ],
            max_rows=20,
        ),
        "",
        "## Seed Axis Sample",
        "",
        _markdown_table(
            seed_axis_rows,
            [
                "local_pair_id",
                "start_condition",
                "seed",
                "axis_a_same_seed_seed_pass",
                "axis_b_pair_level_continuity_seed_pass",
                "same_seed_unknown_step_count",
                "same_seed_unknown_but_pair_known_step_count",
                "true_novel_endpoint_step_count",
                "pair_level_source_step_indices",
                "pair_level_target_step_indices",
                "axis_b_seed_status",
            ],
            max_rows=30,
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
            "This artifact redefines the acceptance contract, not the basin or wall "
            "claim. Axis A remains the strict same-seed anchor-consistency test. "
            "Axis B asks whether the route is continuous in the pair-level endpoint "
            "atlas. A passing Axis B result is not evidence that the path crosses a "
            "wall, improves quality, lowers cost, or generalizes beyond this scoped "
            "panel."
        ),
        "",
    ]
    (output_dir / REPORT_MD).write_text("\n".join(lines), encoding="utf-8")


def run(
    *,
    direct_dir: Path,
    atlas_dir: Path,
    trace_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    direct_seed_rows = _read_csv(direct_dir / DIRECT_SEED_ROWS_CSV)
    direct_contract_rows = _read_csv(direct_dir / DIRECT_CONTRACT_ROWS_CSV)
    direct_gates = _read_csv(direct_dir / DIRECT_GATE_MATRIX_CSV)
    atlas_signature_rows = _read_csv(atlas_dir / ATLAS_SIGNATURE_ROWS_CSV)
    atlas_contract_rows = _read_csv(atlas_dir / ATLAS_CONTRACT_ROWS_CSV)
    atlas_gates = _read_csv(atlas_dir / ATLAS_GATE_MATRIX_CSV)
    trace_rows = _read_csv(trace_dir / TRACE_ROWS_CSV)

    rule_rows = _rule_rows()
    enriched_trace = _atlas_enriched_trace(trace_rows, atlas_signature_rows)
    seed_axis_rows = _seed_axis_rows(direct_seed_rows, enriched_trace)
    contract_axis_rows = _contract_axis_rows(
        direct_contract_rows,
        atlas_contract_rows,
        seed_axis_rows,
    )
    pair_axis_rows = _pair_axis_rows(contract_axis_rows)
    gates = _gate_matrix(
        direct_gates=direct_gates,
        atlas_gates=atlas_gates,
        rule_rows=rule_rows,
        seed_axis_rows=seed_axis_rows,
        contract_axis_rows=contract_axis_rows,
    )
    summary = _summary(
        direct_dir=direct_dir,
        atlas_dir=atlas_dir,
        trace_dir=trace_dir,
        output_dir=output_dir,
        rule_rows=rule_rows,
        seed_axis_rows=seed_axis_rows,
        contract_axis_rows=contract_axis_rows,
        pair_axis_rows=pair_axis_rows,
        gates=gates,
    )

    config = {
        "schema": "nanoclustering_g4_8_dual_axis_direct_path_contract_config.v1",
        "direct_contract_dir": str(direct_dir),
        "endpoint_atlas_dir": str(atlas_dir),
        "trace_dir": str(trace_dir),
        "output_dir": str(output_dir),
        "primary_route_family": PRIMARY_ROUTE_FAMILY,
        "run_status": RUN_STATUS,
        "claim_boundary": CLAIM_BOUNDARY,
    }

    _write_csv(rule_rows, output_dir / RULE_ROWS_CSV)
    _write_csv(seed_axis_rows, output_dir / SEED_AXIS_ROWS_CSV)
    _write_csv(contract_axis_rows, output_dir / CONTRACT_AXIS_ROWS_CSV)
    _write_csv(pair_axis_rows, output_dir / PAIR_AXIS_ROWS_CSV)
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
        rule_rows=rule_rows,
        seed_axis_rows=seed_axis_rows,
        contract_axis_rows=contract_axis_rows,
        pair_axis_rows=pair_axis_rows,
        gates=gates,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-dir", type=Path, default=DEFAULT_DIRECT_DIR)
    parser.add_argument("--atlas-dir", type=Path, default=DEFAULT_ATLAS_DIR)
    parser.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run(
        direct_dir=args.direct_dir,
        atlas_dir=args.atlas_dir,
        trace_dir=args.trace_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
