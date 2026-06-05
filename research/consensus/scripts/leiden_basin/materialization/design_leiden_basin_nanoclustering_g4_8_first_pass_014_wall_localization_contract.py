#!/usr/bin/env python3
"""Design the first-pass local_pair_014 wall-localization contract.

This consumes the accepted ``local_pair_014`` primitive wall-evidence audit and
the synthetic G4.9A parameter-localization map. It freezes the next real-data
contract: a fine bridge-fraction scan for ``local_pair_014`` with
``local_pair_005`` retained as the matched boundary guard.

This is a contract design only. It does not run Leiden, execute routes, refine
thresholds, promote walls, evaluate quality/cost value, replay full
NanoClustering, or claim method success.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from audit_leiden_basin_nanoclustering_g4_8_first_pass_014_wall_evidence import (
    BOUNDARY_GUARD_ROWS_CSV as WALL_BOUNDARY_GUARD_ROWS_CSV,
    DEFAULT_OUTPUT_DIR as DEFAULT_WALL_EVIDENCE_DIR,
    GATE_MATRIX_CSV as WALL_GATE_MATRIX_CSV,
    PAIR_WALL_ROWS_CSV,
    SEED_WALL_ROWS_CSV,
)
from run_leiden_basin_nanoclustering_role_local_route_pilot import (
    BASE_RESULT_DIR,
    _json_safe,
    _read_csv,
    _write_csv,
)


DEFAULT_SYNTHETIC_LOCALIZATION_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_variable_pair_synthetic_g4_9a_parameter_localization_v1_20260604"
)
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_014_wall_localization_contract_gamma1e5_20260604"
)

SYNTHETIC_CASE_SUMMARY_CSV = "variable_pair_synthetic_g4_9a_case_summary.csv"
SYNTHETIC_GATE_MATRIX_CSV = "variable_pair_synthetic_g4_9a_gate_matrix.csv"
SYNTHETIC_PLANE_MATRIX_CSV = "variable_pair_synthetic_g4_9a_plane_matrix.csv"

BOUNDARY_VOCAB_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_014_wall_localization_boundary_vocab_rows.csv"
)
PAIR_ROWS_CSV = "nanoclustering_g4_8_first_pass_014_wall_localization_pair_rows.csv"
ROUTE_PLAN_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_014_wall_localization_route_plan_rows.csv"
)
FRACTION_STEP_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_014_wall_localization_fraction_step_rows.csv"
)
READOUT_RULE_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_014_wall_localization_readout_rule_rows.csv"
)
GATE_MATRIX_CSV = "nanoclustering_g4_8_first_pass_014_wall_localization_gate_matrix.csv"
SUMMARY_JSON = "nanoclustering_g4_8_first_pass_014_wall_localization_summary.json"
CONFIG_JSON = "nanoclustering_g4_8_first_pass_014_wall_localization_config.json"
REPORT_MD = "nanoclustering_g4_8_first_pass_014_wall_localization_report.md"

POSITIVE_PAIR_ID = "local_pair_014"
BOUNDARY_PAIR_ID = "local_pair_005"
AUDIT_PAIR_IDS = (POSITIVE_PAIR_ID, BOUNDARY_PAIR_ID)

RUN_STATUS = "designed_nanoclustering_g4_8_first_pass_014_wall_localization_contract"
CLAIM_BOUNDARY = (
    "NanoClustering G4.8 first-pass local_pair_014 wall-localization contract "
    "design only; reads accepted 014 primitive wall evidence and the synthetic "
    "G4.9A boundary vocabulary, then predeclares fine bridge-fraction scans for "
    "014 with 005 as a boundary guard. It does not run Leiden, execute routes, "
    "retune thresholds, promote wall generality, evaluate quality/cost value, "
    "replay full NanoClustering, or claim method success."
)

DESCENT_FRACTIONS = (1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.625, 0.50, 0.375, 0.25, 0.125, 0.00)
ASCENT_FRACTIONS = tuple(reversed(DESCENT_FRACTIONS))

BOUNDARY_VOCAB: tuple[dict[str, str], ...] = (
    {
        "vocab_code": "W",
        "vocab_label": "full_primitive_wall_regime",
        "synthetic_source": "G4.9A",
        "real_data_readout_rule": (
            "all same-start same-seed descent/ascent units show source-like and "
            "exclusive-target object intervals with accepted objective debt/recovery"
        ),
        "claim_effect": "localizes an already accepted local primitive wall only",
    },
    {
        "vocab_code": "w",
        "vocab_label": "partial_or_fragile_wall_regime",
        "synthetic_source": "G4.9A",
        "real_data_readout_rule": (
            "some starts or seeds show the source/target/source object relation, "
            "but not all matched units"
        ),
        "claim_effect": "diagnostic partial evidence; not a positive wall unit",
    },
    {
        "vocab_code": "T",
        "vocab_label": "target_saturated_regime",
        "synthetic_source": "G4.9A",
        "real_data_readout_rule": (
            "source-like endpoint object is absent or immediately replaced by "
            "target-like object before bridge suppression can localize a wall"
        ),
        "claim_effect": "closed boundary; do not count as wall evidence",
    },
    {
        "vocab_code": "N",
        "vocab_label": "target_absent_or_source_locked_regime",
        "synthetic_source": "G4.9A",
        "real_data_readout_rule": (
            "exclusive target object never appears under direct-retained bridge "
            "fraction scans"
        ),
        "claim_effect": "closed boundary; do not count as wall evidence",
    },
    {
        "vocab_code": "P",
        "vocab_label": "nonrobust_or_mixed_boundary_regime",
        "synthetic_source": "G4.9A",
        "real_data_readout_rule": (
            "target appears only in a nonmonotone, mixed, ambiguous, unknown, or "
            "support-incompatible pattern"
        ),
        "claim_effect": "closed boundary unless a later audit separates a stable object",
    },
)

READOUT_RULES: tuple[dict[str, str], ...] = (
    {
        "rule_id": "L1_scope",
        "rule_group": "scope",
        "rule_question": "Is the localization contract scoped to 014 with 005 as guard?",
        "acceptance_requirement": "positive_pair_id == local_pair_014 and boundary_pair_id == local_pair_005",
        "claim_effect": "prevents broad generality language",
    },
    {
        "rule_id": "L2_fine_bridge_scan",
        "rule_group": "schedule",
        "rule_question": "Are descent and ascent bridge-fraction scans predeclared?",
        "acceptance_requirement": (
            "direct_edge_weight_fraction=1.0 throughout; descent uses "
            "1.00,0.95,0.90,0.85,0.80,0.75,0.625,0.50,0.375,0.25,0.125,0.00; "
            "ascent uses the reverse sequence"
        ),
        "claim_effect": "allows coarse wall-location intervals, not exact thresholds",
    },
    {
        "rule_id": "L3_boundary_vocabulary",
        "rule_group": "readout",
        "rule_question": "Does the readout preserve G4.9A boundary modes?",
        "acceptance_requirement": "classify every unit as W, w, T, N, P, or unresolved",
        "claim_effect": "prevents collapsing all nonready results into generic failure",
    },
    {
        "rule_id": "L4_transition_intervals",
        "rule_group": "readout",
        "rule_question": "Are transition intervals reported instead of tuned thresholds?",
        "acceptance_requirement": (
            "report first target fraction, last source-like fraction, first "
            "recovered source-like fraction, and fraction interval width by seed/start"
        ),
        "claim_effect": "localization only; no threshold policy",
    },
    {
        "rule_id": "L5_boundary_guard",
        "rule_group": "control",
        "rule_question": "Does 005 remain a matched non-positive control?",
        "acceptance_requirement": "005 scan units must not be counted as positive W evidence",
        "claim_effect": "false-positive guard",
    },
    {
        "rule_id": "L6_claim_boundary",
        "rule_group": "claim_boundary",
        "rule_question": "Are wall generality, method, and quality/cost claims closed?",
        "acceptance_requirement": "wall_generality_claim_allowed_after_contract == false",
        "claim_effect": "execution and audit required before any stronger claim",
    },
)

ROUTE_FAMILIES: tuple[dict[str, str], ...] = (
    {
        "planned_route_family": "first_pass_014_wall_localization_descent_scan",
        "route_family_role": "positive_descent_transition_scan",
        "planned_intervention_schedule": "direct retained; bridge fraction descends from 1.00 to 0.00",
        "expected_endpoint_pattern": "source-like interval then exclusive-target interval if wall is localizable",
        "counts_as_positive_if_accepted": "true",
    },
    {
        "planned_route_family": "first_pass_014_wall_localization_ascent_scan",
        "route_family_role": "positive_ascent_recovery_scan",
        "planned_intervention_schedule": "direct retained; bridge fraction ascends from 0.00 to 1.00",
        "expected_endpoint_pattern": "exclusive-target interval then source-like recovery interval if wall is localizable",
        "counts_as_positive_if_accepted": "true",
    },
    {
        "planned_route_family": "first_pass_005_boundary_wall_localization_descent_guard",
        "route_family_role": "boundary_descent_transition_guard",
        "planned_intervention_schedule": "direct retained; bridge fraction descends from 1.00 to 0.00",
        "expected_endpoint_pattern": "must remain non-positive under G4.9A boundary vocabulary",
        "counts_as_positive_if_accepted": "false",
    },
    {
        "planned_route_family": "first_pass_005_boundary_wall_localization_ascent_guard",
        "route_family_role": "boundary_ascent_recovery_guard",
        "planned_intervention_schedule": "direct retained; bridge fraction ascends from 0.00 to 1.00",
        "expected_endpoint_pattern": "must remain non-positive under G4.9A boundary vocabulary",
        "counts_as_positive_if_accepted": "false",
    },
)


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


def _boundary_vocab_rows(synthetic_case_summary: pd.DataFrame) -> pd.DataFrame:
    rows = pd.DataFrame(list(BOUNDARY_VOCAB))
    status_counts = _count_dict(synthetic_case_summary["case_result_status"])
    rows["synthetic_case_count_for_vocab"] = rows["vocab_label"].map(
        lambda label: int(status_counts.get(str(label), 0))
    )
    rows["run_status"] = RUN_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


def _readout_rule_rows() -> pd.DataFrame:
    rows = pd.DataFrame(list(READOUT_RULES))
    rows["run_status"] = RUN_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


def _pair_rows(
    pair_wall_rows: pd.DataFrame,
    wall_seed_rows: pd.DataFrame,
    boundary_guard_rows: pd.DataFrame,
) -> pd.DataFrame:
    wall_pair = pair_wall_rows.iloc[0].to_dict() if not pair_wall_rows.empty else {}
    positive = {
        "local_pair_id": POSITIVE_PAIR_ID,
        "contract_pair_role": "positive_wall_localization_candidate",
        "source_evidence_dir_role": "accepted_primitive_wall_pair",
        "current_wall_seed_count": int(wall_pair.get("wall_seed_count", len(wall_seed_rows))),
        "current_wall_ready_seed_count": int(
            wall_pair.get("wall_ready_seed_count", wall_seed_rows["wall_seed_ready"].map(_as_bool).sum())
        ),
        "current_evidence_status": str(
            wall_pair.get("primitive_wall_evidence_status", "accepted_local_primitive_wall")
        ),
        "boundary_guard_closed_seed_count": int(wall_pair.get("boundary_guard_closed_seed_count", 0)),
        "planned_route_family_count": 2,
        "planned_scan_role": "localize_014_transition_intervals",
        "wall_generality_claim_allowed_after_contract": False,
        "method_claim_allowed_after_contract": False,
        "quality_cost_claim_allowed_after_contract": False,
        "run_status": RUN_STATUS,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    boundary = {
        "local_pair_id": BOUNDARY_PAIR_ID,
        "contract_pair_role": "boundary_collapse_control",
        "source_evidence_dir_role": "retained_closed_boundary_guard",
        "current_wall_seed_count": 0,
        "current_wall_ready_seed_count": 0,
        "current_evidence_status": (
            "closed_boundary_guard"
            if bool(boundary_guard_rows["boundary_guard_closed"].map(_as_bool).all())
            else "boundary_guard_not_closed"
        ),
        "boundary_guard_closed_seed_count": int(
            boundary_guard_rows["boundary_guard_closed"].map(_as_bool).sum()
        ),
        "planned_route_family_count": 2,
        "planned_scan_role": "false_positive_boundary_guard",
        "wall_generality_claim_allowed_after_contract": False,
        "method_claim_allowed_after_contract": False,
        "quality_cost_claim_allowed_after_contract": False,
        "run_status": RUN_STATUS,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    return pd.DataFrame([positive, boundary])


def _start_seed_summary(
    wall_seed_rows: pd.DataFrame,
    boundary_guard_rows: pd.DataFrame,
    *,
    pair_id: str,
) -> pd.DataFrame:
    if pair_id == POSITIVE_PAIR_ID:
        source = wall_seed_rows.copy()
    else:
        source = boundary_guard_rows.copy()
    return (
        source.groupby(["local_pair_id", "branch", "start_condition"], sort=False)
        .agg(seed_count=("seed", "nunique"))
        .reset_index()
        .sort_values(["local_pair_id", "start_condition"], kind="mergesort")
        .reset_index(drop=True)
    )


def _route_plan_rows(
    wall_seed_rows: pd.DataFrame,
    boundary_guard_rows: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    positive_starts = _start_seed_summary(
        wall_seed_rows,
        boundary_guard_rows,
        pair_id=POSITIVE_PAIR_ID,
    )
    boundary_starts = _start_seed_summary(
        wall_seed_rows,
        boundary_guard_rows,
        pair_id=BOUNDARY_PAIR_ID,
    )
    family_by_pair = {
        POSITIVE_PAIR_ID: ROUTE_FAMILIES[:2],
        BOUNDARY_PAIR_ID: ROUTE_FAMILIES[2:],
    }
    starts_by_pair = {
        POSITIVE_PAIR_ID: positive_starts,
        BOUNDARY_PAIR_ID: boundary_starts,
    }
    role_by_pair = {
        POSITIVE_PAIR_ID: "positive_wall_localization_candidate",
        BOUNDARY_PAIR_ID: "boundary_collapse_control",
    }
    for pair_id in AUDIT_PAIR_IDS:
        for start in starts_by_pair[pair_id].itertuples(index=False):
            for order, family in enumerate(family_by_pair[pair_id], start=1):
                route_contract_id = (
                    f"{pair_id}__{start.start_condition}__{family['planned_route_family']}"
                )
                rows.append(
                    {
                        "route_contract_id": route_contract_id,
                        "local_pair_id": pair_id,
                        "branch": str(start.branch),
                        "start_condition": str(start.start_condition),
                        "contract_pair_role": role_by_pair[pair_id],
                        "route_family_order": int(order),
                        **family,
                        "current_seed_count": int(start.seed_count),
                        "direct_edge_weight_fraction": 1.0,
                        "bridge_fraction_sequence": (
                            ";".join(f"{value:.3f}" for value in DESCENT_FRACTIONS)
                            if "descent" in family["planned_route_family"]
                            else ";".join(f"{value:.3f}" for value in ASCENT_FRACTIONS)
                        ),
                        "planned_step_count": int(len(DESCENT_FRACTIONS)),
                        "new_runner_support_required": True,
                        "required_runner_change": (
                            "register localization scan schedules and read endpoint-object "
                            "states without a single expected final anchor"
                        ),
                        "wall_generality_claim_allowed_after_contract": False,
                        "method_claim_allowed_after_contract": False,
                        "quality_cost_claim_allowed_after_contract": False,
                        "run_status": RUN_STATUS,
                        "claim_boundary": CLAIM_BOUNDARY,
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["contract_pair_role", "local_pair_id", "start_condition", "route_family_order"],
        kind="mergesort",
    ).reset_index(drop=True)


def _fraction_step_rows(route_plan: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for route in route_plan.itertuples(index=False):
        fractions = (
            DESCENT_FRACTIONS
            if "descent" in str(route.planned_route_family)
            else ASCENT_FRACTIONS
        )
        for step_index, fraction in enumerate(fractions, start=1):
            rows.append(
                {
                    "route_contract_id": route.route_contract_id,
                    "local_pair_id": route.local_pair_id,
                    "branch": route.branch,
                    "start_condition": route.start_condition,
                    "planned_route_family": route.planned_route_family,
                    "route_family_role": route.route_family_role,
                    "step_index": int(step_index),
                    "step_label": f"bridge_fraction_{fraction:.3f}",
                    "direct_edge_weight_fraction": 1.0,
                    "bridge_edge_weight_fraction": float(fraction),
                    "expected_endpoint_object_family": (
                        "source_like_or_target_or_boundary_by_fraction"
                    ),
                    "expected_final_anchor_variant": "localization_scan_no_single_expected_anchor",
                    "transition_interval_readout_required": True,
                    "run_status": RUN_STATUS,
                    "claim_boundary": CLAIM_BOUNDARY,
                }
            )
    return pd.DataFrame(rows)


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


def _gate_matrix(
    *,
    wall_gates: pd.DataFrame,
    synthetic_gates: pd.DataFrame,
    boundary_vocab: pd.DataFrame,
    pair_rows: pd.DataFrame,
    route_plan: pd.DataFrame,
    fraction_steps: pd.DataFrame,
) -> pd.DataFrame:
    positive_routes = route_plan[
        route_plan["contract_pair_role"].astype(str).eq(
            "positive_wall_localization_candidate"
        )
    ]
    boundary_routes = route_plan[
        route_plan["contract_pair_role"].astype(str).eq("boundary_collapse_control")
    ]
    vocab_codes = set(boundary_vocab["vocab_code"].astype(str))
    rows = [
        _gate_row(
            "G1_upstream_wall_and_synthetic_gates_pass",
            "Did the 014 wall evidence and G4.9A localization gates pass?",
            {
                "wall_gates": _count_dict(wall_gates["gate_status"]),
                "synthetic_gates": _count_dict(synthetic_gates["gate_status"]),
            },
            "all upstream gates pass",
            bool(wall_gates["gate_status"].astype(str).eq("pass").all())
            and bool(synthetic_gates["gate_status"].astype(str).eq("pass").all()),
        ),
        _gate_row(
            "G2_scope_014_positive_005_boundary",
            "Is the contract scoped to 014 positive and 005 boundary only?",
            pair_rows[["local_pair_id", "contract_pair_role", "current_evidence_status"]].to_dict("records"),
            "exactly two pair rows with fixed roles",
            len(pair_rows) == 2
            and set(pair_rows["local_pair_id"].astype(str)) == set(AUDIT_PAIR_IDS),
        ),
        _gate_row(
            "G3_boundary_vocabulary_materialized",
            "Is the G4.9A W/w/T/N/P vocabulary materialized for real-data readout?",
            sorted(vocab_codes),
            "W, w, T, N, P present",
            vocab_codes == {"W", "w", "T", "N", "P"},
        ),
        _gate_row(
            "G4_route_families_predeclared",
            "Are descent/ascent scans predeclared for positive and boundary pairs?",
            {
                "positive_route_count": int(len(positive_routes)),
                "boundary_route_count": int(len(boundary_routes)),
                "families": sorted(route_plan["planned_route_family"].astype(str).unique().tolist()),
            },
            "8 positive rows and 8 boundary rows",
            len(positive_routes) == 8 and len(boundary_routes) == 8,
        ),
        _gate_row(
            "G5_fraction_steps_complete",
            "Are fine bridge-fraction steps fully materialized?",
            f"fraction_step_rows={len(fraction_steps)}",
            "16 route rows * 12 fraction steps = 192",
            len(fraction_steps) == 192
            and bool(fraction_steps["direct_edge_weight_fraction"].astype(float).eq(1.0).all()),
        ),
        _gate_row(
            "G6_runner_support_not_assumed",
            "Does the contract avoid assuming existing runner support?",
            _count_dict(route_plan["new_runner_support_required"]),
            "all rows require new localization runner support",
            bool(route_plan["new_runner_support_required"].map(_as_bool).all()),
        ),
        _gate_row(
            "G7_claims_closed",
            "Are wall generality, method, and quality/cost claims closed?",
            CLAIM_BOUNDARY,
            "all claim flags false",
            bool(pair_rows["wall_generality_claim_allowed_after_contract"].eq(False).all())
            and bool(route_plan["wall_generality_claim_allowed_after_contract"].eq(False).all()),
        ),
    ]
    gates = pd.DataFrame(rows)
    gates["run_status"] = RUN_STATUS
    gates["claim_boundary"] = CLAIM_BOUNDARY
    return gates


def _summary(
    *,
    wall_evidence_dir: Path,
    synthetic_localization_dir: Path,
    output_dir: Path,
    boundary_vocab: pd.DataFrame,
    pair_rows: pd.DataFrame,
    route_plan: pd.DataFrame,
    fraction_steps: pd.DataFrame,
    gates: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "schema": "nanoclustering_g4_8_first_pass_014_wall_localization_contract_summary.v1",
        "status": RUN_STATUS,
        "wall_evidence_dir": str(wall_evidence_dir),
        "synthetic_localization_dir": str(synthetic_localization_dir),
        "output_dir": str(output_dir),
        "boundary_vocab_count": int(len(boundary_vocab)),
        "pair_row_count": int(len(pair_rows)),
        "route_plan_row_count": int(len(route_plan)),
        "positive_route_plan_row_count": int(
            route_plan["contract_pair_role"].astype(str).eq(
                "positive_wall_localization_candidate"
            ).sum()
        ),
        "boundary_route_plan_row_count": int(
            route_plan["contract_pair_role"].astype(str).eq("boundary_collapse_control").sum()
        ),
        "fraction_step_row_count": int(len(fraction_steps)),
        "planned_route_family_counts": _count_dict(route_plan["planned_route_family"]),
        "planned_fraction_sequences": {
            "descent": [float(value) for value in DESCENT_FRACTIONS],
            "ascent": [float(value) for value in ASCENT_FRACTIONS],
        },
        "gate_status_counts": _count_dict(gates["gate_status"]),
        "failed_gates": gates.loc[
            ~gates["gate_status"].astype(str).eq("pass"), "gate_id"
        ].tolist(),
        "interpretation": (
            "The next real-data step is not another synthetic sweep. It is a "
            "fine bridge-fraction localization contract for the one accepted "
            "real-data primitive wall candidate, local_pair_014, while retaining "
            "local_pair_005 as a false-positive boundary guard and preserving "
            "the G4.9A W/w/T/N/P failure vocabulary."
        ),
        "recommended_next_gate": (
            "Implement a localization runner that consumes these 16 route rows "
            "and 192 fraction steps, then audit transition intervals by "
            "start/seed before any wall-location or generality language."
        ),
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 60) -> str:
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


def _write_report(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    boundary_vocab: pd.DataFrame,
    pair_rows: pd.DataFrame,
    route_plan: pd.DataFrame,
    fraction_steps: pd.DataFrame,
    gates: pd.DataFrame,
) -> None:
    lines = [
        "# NanoClustering G4.8 First-Pass 014 Wall-Localization Contract",
        "",
        f"- status: `{summary['status']}`",
        f"- route_plan_row_count: {summary['route_plan_row_count']}",
        f"- fraction_step_row_count: {summary['fraction_step_row_count']}",
        f"- boundary_vocab_count: {summary['boundary_vocab_count']}",
        f"- planned_route_family_counts: {summary['planned_route_family_counts']}",
        f"- gate_status_counts: {summary['gate_status_counts']}",
        f"- failed_gates: {summary['failed_gates']}",
        f"- interpretation: {summary['interpretation']}",
        f"- recommended_next_gate: {summary['recommended_next_gate']}",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Boundary Vocabulary",
        "",
        _markdown_table(
            boundary_vocab,
            [
                "vocab_code",
                "vocab_label",
                "synthetic_case_count_for_vocab",
                "real_data_readout_rule",
                "claim_effect",
            ],
            max_rows=10,
        ),
        "",
        "## Pair Rows",
        "",
        _markdown_table(
            pair_rows,
            [
                "local_pair_id",
                "contract_pair_role",
                "current_wall_seed_count",
                "current_wall_ready_seed_count",
                "boundary_guard_closed_seed_count",
                "current_evidence_status",
                "planned_scan_role",
            ],
            max_rows=10,
        ),
        "",
        "## Route Plan",
        "",
        _markdown_table(
            route_plan,
            [
                "route_contract_id",
                "local_pair_id",
                "start_condition",
                "contract_pair_role",
                "planned_route_family",
                "route_family_role",
                "bridge_fraction_sequence",
                "new_runner_support_required",
                "counts_as_positive_if_accepted",
            ],
            max_rows=40,
        ),
        "",
        "## Fraction Steps",
        "",
        _markdown_table(
            fraction_steps,
            [
                "route_contract_id",
                "step_index",
                "step_label",
                "direct_edge_weight_fraction",
                "bridge_edge_weight_fraction",
                "expected_final_anchor_variant",
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
            "This is only a design contract. It must not be interpreted as "
            "executed wall-location evidence, broader basin generality, method "
            "success, or quality/cost value."
        ),
        "",
    ]
    (output_dir / REPORT_MD).write_text("\n".join(lines), encoding="utf-8")


def design(args: argparse.Namespace) -> dict[str, Any]:
    wall_evidence_dir = Path(args.wall_evidence_dir)
    synthetic_localization_dir = Path(args.synthetic_localization_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wall_seed_rows = _read_csv(wall_evidence_dir / SEED_WALL_ROWS_CSV)
    boundary_guard_rows = _read_csv(wall_evidence_dir / WALL_BOUNDARY_GUARD_ROWS_CSV)
    pair_wall_rows = _read_csv(wall_evidence_dir / PAIR_WALL_ROWS_CSV)
    wall_gates = _read_csv(wall_evidence_dir / WALL_GATE_MATRIX_CSV)
    synthetic_case_summary = _read_csv(
        synthetic_localization_dir / SYNTHETIC_CASE_SUMMARY_CSV
    )
    synthetic_plane_matrix = _read_csv(
        synthetic_localization_dir / SYNTHETIC_PLANE_MATRIX_CSV
    )
    synthetic_gates = _read_csv(synthetic_localization_dir / SYNTHETIC_GATE_MATRIX_CSV)

    boundary_vocab = _boundary_vocab_rows(synthetic_case_summary)
    readout_rules = _readout_rule_rows()
    pair_rows = _pair_rows(pair_wall_rows, wall_seed_rows, boundary_guard_rows)
    route_plan = _route_plan_rows(wall_seed_rows, boundary_guard_rows)
    fraction_steps = _fraction_step_rows(route_plan)
    gates = _gate_matrix(
        wall_gates=wall_gates,
        synthetic_gates=synthetic_gates,
        boundary_vocab=boundary_vocab,
        pair_rows=pair_rows,
        route_plan=route_plan,
        fraction_steps=fraction_steps,
    )
    summary = _summary(
        wall_evidence_dir=wall_evidence_dir,
        synthetic_localization_dir=synthetic_localization_dir,
        output_dir=output_dir,
        boundary_vocab=boundary_vocab,
        pair_rows=pair_rows,
        route_plan=route_plan,
        fraction_steps=fraction_steps,
        gates=gates,
    )

    _write_csv(boundary_vocab, output_dir / BOUNDARY_VOCAB_ROWS_CSV)
    _write_csv(pair_rows, output_dir / PAIR_ROWS_CSV)
    _write_csv(route_plan, output_dir / ROUTE_PLAN_ROWS_CSV)
    _write_csv(fraction_steps, output_dir / FRACTION_STEP_ROWS_CSV)
    _write_csv(readout_rules, output_dir / READOUT_RULE_ROWS_CSV)
    _write_csv(gates, output_dir / GATE_MATRIX_CSV)
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    config = {
        "schema": "nanoclustering_g4_8_first_pass_014_wall_localization_contract_config.v1",
        "wall_evidence_dir": str(wall_evidence_dir),
        "synthetic_localization_dir": str(synthetic_localization_dir),
        "output_dir": str(output_dir),
        "positive_pair_id": POSITIVE_PAIR_ID,
        "boundary_pair_id": BOUNDARY_PAIR_ID,
        "descent_fractions": [float(value) for value in DESCENT_FRACTIONS],
        "ascent_fractions": [float(value) for value in ASCENT_FRACTIONS],
        "synthetic_boundary_codes": sorted(
            synthetic_plane_matrix["matrix_code"].astype(str).unique().tolist()
        ),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(
        output_dir=output_dir,
        summary=summary,
        boundary_vocab=boundary_vocab,
        pair_rows=pair_rows,
        route_plan=route_plan,
        fraction_steps=fraction_steps,
        gates=gates,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wall-evidence-dir", type=Path, default=DEFAULT_WALL_EVIDENCE_DIR)
    parser.add_argument(
        "--synthetic-localization-dir",
        type=Path,
        default=DEFAULT_SYNTHETIC_LOCALIZATION_DIR,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    summary = design(parse_args())
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
