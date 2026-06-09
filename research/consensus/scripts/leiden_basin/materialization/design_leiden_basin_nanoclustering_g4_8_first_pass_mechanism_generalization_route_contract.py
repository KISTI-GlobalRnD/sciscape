#!/usr/bin/env python3
"""Design the fixed-predicate mechanism-generalization route contract.

This contract follows the mechanism-generalization screen. It opens only the
strict nonboundary local-signature analogs that were predeclared by that screen
(``009``, ``012``, ``020``) and retains ``014``/``005`` as controls. Route
execution is restricted to source-family start rows and the same fine
bridge-fraction schedule used for ``016``.
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


DEFAULT_GENERALIZATION_SCREEN_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_mechanism_generalization_screen_gamma1e5_20260605"
)
DEFAULT_LOCAL_VALIDATION_DIR = (
    BASE_RESULT_DIR / "leiden_basin_nanoclustering_g4_8_local_validation_readout_gamma1e5_20260604"
)
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_mechanism_generalization_route_contract_gamma1e5_20260605"
)

CANDIDATE_PAIR_IDS = ("local_pair_009", "local_pair_012", "local_pair_020")
REFERENCE_PAIR_ID = "local_pair_014"
BOUNDARY_GUARD_PAIR_ID = "local_pair_005"
CONTROL_PAIR_IDS = (REFERENCE_PAIR_ID, BOUNDARY_GUARD_PAIR_ID)
ALL_EXECUTION_PAIR_IDS = (*CANDIDATE_PAIR_IDS, *CONTROL_PAIR_IDS)

PLANNED_ROUTE_FAMILY = (
    "first_pass_mechanism_generalization_fixed_bridge_fraction_scan"
)
FINE_BRIDGE_FRACTIONS = (
    1.0,
    0.875,
    0.8125,
    0.78125,
    0.75,
    0.71875,
    0.6875,
    0.625,
    0.5,
)
START_CONDITION_ORDER = (
    "bridges_to_left",
    "pair_together",
    "singleton",
)

ROUTE_PLAN_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_mechanism_generalization_route_contract_route_plan_rows.csv"
)
FRACTION_STEP_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_mechanism_generalization_route_contract_fraction_step_rows.csv"
)
READOUT_RULE_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_mechanism_generalization_route_contract_readout_rule_rows.csv"
)
GATE_MATRIX_CSV = (
    "nanoclustering_g4_8_first_pass_mechanism_generalization_route_contract_gate_matrix.csv"
)
SUMMARY_JSON = (
    "nanoclustering_g4_8_first_pass_mechanism_generalization_route_contract_summary.json"
)
CONFIG_JSON = (
    "nanoclustering_g4_8_first_pass_mechanism_generalization_route_contract_config.json"
)
REPORT_MD = (
    "nanoclustering_g4_8_first_pass_mechanism_generalization_route_contract_report.md"
)

RUN_STATUS = "designed_nanoclustering_g4_8_first_pass_mechanism_generalization_route_contract"
ROUTE_EXECUTION_STATUS = (
    "not_executed_contract_only_mechanism_generalization_route_trace"
)
WALL_PROMOTION_STATUS = "not_promoted_contract_only"
METHOD_STATUS = "mechanism_generalization_route_contract_not_method"
CLAIM_BOUNDARY = (
    "NanoClustering G4.8 first-pass mechanism-generalization route contract "
    "only; predeclares a narrow fixed-predicate bridge-fraction trace for "
    "strict local-signature analogs plus two controls. It does not execute "
    "Leiden, promote basin walls, replay full NanoClustering, evaluate "
    "quality/cost value, or claim method success."
)


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


def _load_context(
    *,
    generalization_screen_dir: Path,
    local_validation_dir: Path,
) -> dict[str, Any]:
    return {
        "paths": {
            "generalization_screen_dir": generalization_screen_dir,
            "local_validation_dir": local_validation_dir,
        },
        "summaries": {
            "generalization_screen": _read_json(
                generalization_screen_dir
                / "nanoclustering_g4_8_first_pass_mechanism_generalization_summary.json"
            ),
            "local_validation": _read_json(
                local_validation_dir
                / "nanoclustering_g4_8_local_validation_readout_summary.json"
            ),
        },
        "tables": {
            "generalization_next_gate": _read_csv(
                generalization_screen_dir
                / "nanoclustering_g4_8_first_pass_mechanism_generalization_next_gate_rows.csv"
            ),
            "generalization_gate_matrix": _read_csv(
                generalization_screen_dir
                / "nanoclustering_g4_8_first_pass_mechanism_generalization_gate_matrix.csv"
            ),
            "start_condition_rows": _read_csv(
                local_validation_dir
                / "nanoclustering_g4_8_local_validation_readout_start_condition_rows.csv"
            ),
        },
    }


def _pair_role(pair_id: str) -> str:
    if pair_id in CANDIDATE_PAIR_IDS:
        return "strict_nonboundary_local_signature_analog"
    if pair_id == REFERENCE_PAIR_ID:
        return "positive_reference_control"
    if pair_id == BOUNDARY_GUARD_PAIR_ID:
        return "boundary_guard_control"
    raise ValueError(f"unexpected pair id: {pair_id}")


def _source_start_rows(start_rows: pd.DataFrame) -> pd.DataFrame:
    scoped = start_rows[start_rows["local_pair_id"].astype(str).isin(ALL_EXECUTION_PAIR_IDS)].copy()
    scoped = scoped[scoped["start_condition"].astype(str).isin(START_CONDITION_ORDER)].copy()
    scoped = scoped[
        scoped["start_condition_macro_role"].astype(str).eq("R_candidate")
        & scoped["start_condition_expected_validation_pass"].map(_as_bool)
    ].copy()
    order = {start: index for index, start in enumerate(START_CONDITION_ORDER)}
    pair_order = {pair_id: index for index, pair_id in enumerate(ALL_EXECUTION_PAIR_IDS)}
    scoped["_pair_order"] = scoped["local_pair_id"].astype(str).map(pair_order)
    scoped["_start_order"] = scoped["start_condition"].astype(str).map(order)
    return scoped.sort_values(["_pair_order", "_start_order"], kind="mergesort").drop(
        columns=["_pair_order", "_start_order"]
    )


def _route_plan(context: dict[str, Any]) -> pd.DataFrame:
    start_rows = _source_start_rows(context["tables"]["start_condition_rows"])
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(start_rows.itertuples(index=False), start=1):
        pair_id = str(row.local_pair_id)
        start = str(row.start_condition)
        route_contract_id = f"{pair_id}__{start}__{PLANNED_ROUTE_FAMILY}"
        rows.append(
            {
                "route_contract_id": route_contract_id,
                "validation_unit_id": route_contract_id,
                "local_pair_id": pair_id,
                "contract_pair_role": _pair_role(pair_id),
                "start_condition": start,
                "planned_route_family": PLANNED_ROUTE_FAMILY,
                "route_family_order": int(index),
                "route_family_role": "fixed_016_bridge_fraction_predicate_probe",
                "seed_count": 8,
                "fraction_step_count": int(len(FINE_BRIDGE_FRACTIONS)),
                "source_start_macro_role": str(row.start_condition_macro_role),
                "source_start_condition": str(row.start_condition_source_condition),
                "source_start_expected_validation_pass": _as_bool(
                    row.start_condition_expected_validation_pass
                ),
                "expected_final_anchor_variant": "drop_bridge_edges",
                "wall_generality_claim_allowed_after_contract": False,
                "method_claim_allowed_after_contract": False,
                "quality_cost_claim_allowed_after_contract": False,
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "method_status": METHOD_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
                "run_status": RUN_STATUS,
            }
        )
    return pd.DataFrame(rows)


def _fraction_steps(route_plan: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for route in route_plan.itertuples(index=False):
        route_data = route._asdict()
        for index, fraction in enumerate(FINE_BRIDGE_FRACTIONS, start=1):
            if fraction == 1.0:
                step_role = "source_family_bracket"
            elif fraction > 0.75:
                step_role = "upper_transition_band_probe"
            elif fraction == 0.75:
                step_role = "center_transition_band_probe"
            elif fraction > 0.5:
                step_role = "lower_transition_band_probe"
            else:
                step_role = "target_like_bracket"
            rows.append(
                {
                    **route_data,
                    "step_index": int(index),
                    "step_label": f"bridge_fraction_{fraction:.5g}",
                    "direct_edge_weight_fraction": 1.0,
                    "bridge_edge_weight_fraction": float(fraction),
                    "expected_final_anchor_variant": "drop_bridge_edges",
                    "step_role": step_role,
                    "predeclared_readout": (
                        "source-family at high bridge fraction, pair-separated "
                        "single-side bridge finite band, target-like final "
                        "coassignment without selected bridges"
                    ),
                    "route_execution_status": ROUTE_EXECUTION_STATUS,
                    "wall_promotion_status": WALL_PROMOTION_STATUS,
                    "method_status": METHOD_STATUS,
                    "claim_boundary": CLAIM_BOUNDARY,
                    "run_status": RUN_STATUS,
                }
            )
    return pd.DataFrame(rows)


def _readout_rules() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "rule_id": "R1_exact_scope",
                "readout_axis": "execution_scope",
                "rule": "execute only 009/012/020 plus 014/005 controls, source-family starts only, eight seeds, and nine fixed bridge fractions",
                "claim_effect": "blocks threshold/policy/localization sweep expansion",
            },
            {
                "rule_id": "R2_source_family_start",
                "readout_axis": "source_surface",
                "rule": "the first fraction must be pair_coassigned_with_selected_bridge or pair_separated_bridge_split",
                "claim_effect": "requires a source-family high-bridge surface before route recurrence can pass",
            },
            {
                "rule_id": "R3_single_side_band",
                "readout_axis": "transition_band",
                "rule": "finite transition band requires pair_separated_single_side_bridge at two or more adjacent bridge fractions",
                "claim_effect": "tests the fixed 016 transient mechanism rather than endpoint-only motion",
            },
            {
                "rule_id": "R4_target_like_final",
                "readout_axis": "target_surface",
                "rule": "the final 0.5 bridge-fraction step must be pair_coassigned_without_selected_bridge",
                "claim_effect": "requires target-like bridge-release endpoint under the same predicate",
            },
            {
                "rule_id": "R5_claim_boundary",
                "readout_axis": "claims",
                "rule": "route recurrence is not wall, method, quality/cost, or full NanoClustering replay evidence",
                "claim_effect": "keeps all promotion claims closed",
            },
        ]
    )


def _gate_matrix(
    *,
    context: dict[str, Any],
    route_plan: pd.DataFrame,
    fraction_steps: pd.DataFrame,
) -> pd.DataFrame:
    screen_summary = context["summaries"]["generalization_screen"]
    screen_gates = context["tables"]["generalization_gate_matrix"]
    non_g4_gates = screen_gates[
        ~screen_gates["gate_id"].astype(str).eq("G4_route_level_generality_not_yet_established")
    ]
    next_gate_pairs = sorted(
        context["tables"]["generalization_next_gate"]["local_pair_id"].astype(str).tolist()
    )
    route_counts = route_plan["local_pair_id"].astype(str).value_counts().to_dict()
    return pd.DataFrame(
        [
            _gate_row(
                "G1_upstream_screen_ready",
                "Did the upstream screen leave only the expected route-generality gap?",
                {
                    "screen_failed_gates": screen_summary.get("failed_gates"),
                    "non_g4_gate_status_counts": non_g4_gates["gate_status"].value_counts().to_dict(),
                },
                "only G4 route-level generality is failed upstream",
                screen_summary.get("failed_gates") == [
                    "G4_route_level_generality_not_yet_established"
                ]
                and bool(non_g4_gates["gate_status"].astype(str).eq("pass").all()),
            ),
            _gate_row(
                "G2_candidate_queue_exact",
                "Are only the predeclared strict nonboundary analogs opened?",
                next_gate_pairs,
                "009, 012, and 020",
                next_gate_pairs == sorted(CANDIDATE_PAIR_IDS),
            ),
            _gate_row(
                "G3_controls_retained",
                "Are the positive reference and boundary guard included as controls?",
                route_counts,
                "014 and 005 have at least one route row each",
                all(route_counts.get(pair_id, 0) > 0 for pair_id in CONTROL_PAIR_IDS),
            ),
            _gate_row(
                "G4_source_family_start_scope",
                "Are all route rows restricted to source-family start rows?",
                route_plan[
                    [
                        "local_pair_id",
                        "start_condition",
                        "source_start_macro_role",
                        "source_start_expected_validation_pass",
                    ]
                ].to_dict("records"),
                "every route row is R_candidate and expected pass",
                bool(route_plan["source_start_macro_role"].astype(str).eq("R_candidate").all())
                and bool(route_plan["source_start_expected_validation_pass"].map(_as_bool).all()),
            ),
            _gate_row(
                "G5_fixed_016_fraction_schedule",
                "Does the contract reuse the fixed 016 fine bridge-fraction schedule?",
                {
                    "fractions": list(FINE_BRIDGE_FRACTIONS),
                    "direct_fraction_values": sorted(
                        fraction_steps["direct_edge_weight_fraction"].astype(float).unique().tolist()
                    ),
                    "route_plan_rows": int(len(route_plan)),
                    "fraction_step_rows": int(len(fraction_steps)),
                },
                "nine bridge fractions, direct fraction fixed at 1.0",
                sorted(fraction_steps["bridge_edge_weight_fraction"].astype(float).unique().tolist())
                == sorted(FINE_BRIDGE_FRACTIONS)
                and set(fraction_steps["direct_edge_weight_fraction"].astype(float)) == {1.0},
            ),
            _gate_row(
                "G6_claim_boundaries_closed",
                "Are wall, method, quality/cost, and full-replay claims closed?",
                CLAIM_BOUNDARY,
                "contract-only, all promotion flags false",
                not bool(route_plan["wall_generality_claim_allowed_after_contract"].map(_as_bool).any())
                and not bool(route_plan["method_claim_allowed_after_contract"].map(_as_bool).any())
                and not bool(route_plan["quality_cost_claim_allowed_after_contract"].map(_as_bool).any()),
            ),
        ]
    )


def _summary(
    *,
    generalization_screen_dir: Path,
    local_validation_dir: Path,
    output_dir: Path,
    route_plan: pd.DataFrame,
    fraction_steps: pd.DataFrame,
    readout_rules: pd.DataFrame,
    gates: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "schema": "nanoclustering_g4_8_first_pass_mechanism_generalization_route_contract_summary.v1",
        "status": RUN_STATUS,
        "generalization_screen_dir": str(generalization_screen_dir),
        "local_validation_dir": str(local_validation_dir),
        "output_dir": str(output_dir),
        "candidate_pair_ids": list(CANDIDATE_PAIR_IDS),
        "control_pair_ids": list(CONTROL_PAIR_IDS),
        "route_plan_row_count": int(len(route_plan)),
        "fraction_step_row_count": int(len(fraction_steps)),
        "readout_rule_count": int(len(readout_rules)),
        "route_rows_by_pair": {
            str(key): int(value)
            for key, value in route_plan["local_pair_id"].astype(str).value_counts().to_dict().items()
        },
        "fine_bridge_fractions": list(map(float, FINE_BRIDGE_FRACTIONS)),
        "gate_status_counts": gates["gate_status"].value_counts().to_dict(),
        "failed_gates": gates.loc[
            ~gates["gate_status"].astype(str).eq("pass"),
            "gate_id",
        ].tolist(),
        "interpretation": (
            "This contract opens the smallest route-level test implied by the "
            "mechanism-generalization screen: fixed 016 predicate, source-family "
            "starts only, strict analogs 009/012/020, and controls 014/005."
        ),
        "recommended_next_gate": (
            "Run this contract and audit whether any strict nonboundary analog "
            "has source-family start, finite single-side transition band, and "
            "target-like final state across all source-family seed routes."
        ),
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _write_report(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    route_plan: pd.DataFrame,
    fraction_steps: pd.DataFrame,
    readout_rules: pd.DataFrame,
    gates: pd.DataFrame,
) -> None:
    lines = [
        "# NanoClustering G4.8 First-Pass Mechanism Generalization Route Contract",
        "",
        "## Summary",
        "",
        f"- status: {summary['status']}",
        f"- route_plan_row_count: {summary['route_plan_row_count']}",
        f"- fraction_step_row_count: {summary['fraction_step_row_count']}",
        f"- route_rows_by_pair: {summary['route_rows_by_pair']}",
        f"- failed_gates: {summary['failed_gates']}",
        "",
        "## Route Plan",
        "",
        _markdown_table(
            route_plan,
            [
                "route_contract_id",
                "local_pair_id",
                "contract_pair_role",
                "start_condition",
                "source_start_macro_role",
                "source_start_expected_validation_pass",
                "seed_count",
                "fraction_step_count",
            ],
            max_rows=40,
        ),
        "",
        "## Fraction Steps",
        "",
        _markdown_table(
            fraction_steps,
            [
                "local_pair_id",
                "start_condition",
                "step_index",
                "step_label",
                "direct_edge_weight_fraction",
                "bridge_edge_weight_fraction",
                "step_role",
            ],
            max_rows=60,
        ),
        "",
        "## Readout Rules",
        "",
        _markdown_table(readout_rules, ["rule_id", "readout_axis", "rule", "claim_effect"]),
        "",
        "## Gates",
        "",
        _markdown_table(
            gates,
            ["gate_id", "question", "observed", "minimum_or_rule", "gate_status"],
            max_rows=20,
        ),
        "",
        "## Claim Boundary",
        "",
        CLAIM_BOUNDARY,
        "",
    ]
    (output_dir / REPORT_MD).write_text("\n".join(lines), encoding="utf-8")


def run_design(
    *,
    generalization_screen_dir: Path = DEFAULT_GENERALIZATION_SCREEN_DIR,
    local_validation_dir: Path = DEFAULT_LOCAL_VALIDATION_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    context = _load_context(
        generalization_screen_dir=generalization_screen_dir,
        local_validation_dir=local_validation_dir,
    )
    route_plan = _route_plan(context)
    fraction_steps = _fraction_steps(route_plan)
    readout_rules = _readout_rules()
    gates = _gate_matrix(
        context=context,
        route_plan=route_plan,
        fraction_steps=fraction_steps,
    )
    summary = _summary(
        generalization_screen_dir=generalization_screen_dir,
        local_validation_dir=local_validation_dir,
        output_dir=output_dir,
        route_plan=route_plan,
        fraction_steps=fraction_steps,
        readout_rules=readout_rules,
        gates=gates,
    )
    config = {
        "schema": "nanoclustering_g4_8_first_pass_mechanism_generalization_route_contract_config.v1",
        "generalization_screen_dir": str(generalization_screen_dir),
        "local_validation_dir": str(local_validation_dir),
        "output_dir": str(output_dir),
        "candidate_pair_ids": list(CANDIDATE_PAIR_IDS),
        "control_pair_ids": list(CONTROL_PAIR_IDS),
        "planned_route_family": PLANNED_ROUTE_FAMILY,
        "fine_bridge_fractions": list(map(float, FINE_BRIDGE_FRACTIONS)),
        "start_condition_order": list(START_CONDITION_ORDER),
        "claim_boundary": CLAIM_BOUNDARY,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(route_plan, output_dir / ROUTE_PLAN_ROWS_CSV)
    _write_csv(fraction_steps, output_dir / FRACTION_STEP_ROWS_CSV)
    _write_csv(readout_rules, output_dir / READOUT_RULE_ROWS_CSV)
    _write_csv(gates, output_dir / GATE_MATRIX_CSV)
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(
        output_dir=output_dir,
        summary=summary,
        route_plan=route_plan,
        fraction_steps=fraction_steps,
        readout_rules=readout_rules,
        gates=gates,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generalization-screen-dir",
        type=Path,
        default=DEFAULT_GENERALIZATION_SCREEN_DIR,
    )
    parser.add_argument("--local-validation-dir", type=Path, default=DEFAULT_LOCAL_VALIDATION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_design(
        generalization_screen_dir=Path(args.generalization_screen_dir),
        local_validation_dir=Path(args.local_validation_dir),
        output_dir=Path(args.output_dir),
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
