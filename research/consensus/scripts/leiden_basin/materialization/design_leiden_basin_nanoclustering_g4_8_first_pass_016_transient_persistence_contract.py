#!/usr/bin/env python3
"""Design the local_pair_016 transient persistence contract.

This contract follows the read-only semantic validation for ``local_pair_016``
and predeclares one narrow execution: a fine bridge-fraction scan around the
previous step-2 transient at bridge fraction 0.75. It is intentionally not a
wall-localization contract and not a method-success contract.
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
TRANSIENT_SIGNATURE_ID = "aeb59ab537e6"
TARGET_SIGNATURE_ID = "3c9b8a190753"
PLANNED_ROUTE_FAMILY = "first_pass_016_transient_fine_bridge_persistence_scan"

DEFAULT_SEMANTIC_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_016_transient_semantic_validation_gamma1e5_20260605"
)
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_016_transient_persistence_contract_gamma1e5_20260605"
)

ROUTE_PLAN_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_016_transient_persistence_contract_route_plan_rows.csv"
)
FRACTION_STEP_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_016_transient_persistence_contract_fraction_step_rows.csv"
)
READOUT_RULE_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_016_transient_persistence_contract_readout_rule_rows.csv"
)
GATE_MATRIX_CSV = (
    "nanoclustering_g4_8_first_pass_016_transient_persistence_contract_gate_matrix.csv"
)
SUMMARY_JSON = "nanoclustering_g4_8_first_pass_016_transient_persistence_contract_summary.json"
CONFIG_JSON = "nanoclustering_g4_8_first_pass_016_transient_persistence_contract_config.json"
REPORT_MD = "nanoclustering_g4_8_first_pass_016_transient_persistence_contract_report.md"

RUN_STATUS = "designed_nanoclustering_g4_8_first_pass_016_transient_persistence_contract"
ROUTE_EXECUTION_STATUS = "not_executed_contract_only_016_transient_persistence"
WALL_PROMOTION_STATUS = "not_promoted_contract_only"
METHOD_STATUS = "contract_design_not_method"
CLAIM_BOUNDARY = (
    "NanoClustering G4.8 first-pass local_pair_016 transient-persistence "
    "contract only; predeclares a narrow fine bridge-fraction scan around the "
    "previous recurrent step-2 transient. It does not execute Leiden, promote "
    "basin walls, run reverse hysteresis, replay full NanoClustering, evaluate "
    "quality/cost value, or claim method/algorithm success."
)

FINE_BRIDGE_FRACTIONS = (1.0, 0.875, 0.8125, 0.78125, 0.75, 0.71875, 0.6875, 0.625, 0.5)
START_CONDITION_ORDER = ("bridges_to_left", "pair_together", "singleton")


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


def _load_semantic_context(semantic_dir: Path) -> dict[str, Any]:
    return {
        "summary": _read_json(
            semantic_dir
            / "nanoclustering_g4_8_first_pass_016_transient_semantic_summary.json"
        ),
        "gates": _read_csv(
            semantic_dir
            / "nanoclustering_g4_8_first_pass_016_transient_semantic_gate_matrix.csv"
        ),
        "route_rows": _read_csv(
            semantic_dir
            / "nanoclustering_g4_8_first_pass_016_transient_semantic_route_rows.csv"
        ),
        "step_rows": _read_csv(
            semantic_dir
            / "nanoclustering_g4_8_first_pass_016_transient_semantic_step_rows.csv"
        ),
    }


def _route_plan(route_rows: pd.DataFrame) -> pd.DataFrame:
    primary = route_rows[route_rows["local_pair_id"].astype(str).eq(PRIMARY_PAIR_ID)]
    observed_starts = set(primary["start_condition"].astype(str))
    starts = [start for start in START_CONDITION_ORDER if start in observed_starts]
    rows: list[dict[str, Any]] = []
    for order, start in enumerate(starts, start=1):
        route_contract_id = f"{PRIMARY_PAIR_ID}__{start}__{PLANNED_ROUTE_FAMILY}"
        rows.append(
            {
                "route_contract_id": route_contract_id,
                "validation_unit_id": route_contract_id,
                "local_pair_id": PRIMARY_PAIR_ID,
                "contract_pair_role": "primary_typed_transient_gateway_candidate",
                "start_condition": start,
                "planned_route_family": PLANNED_ROUTE_FAMILY,
                "route_family_order": int(order),
                "route_family_role": "fine_bridge_fraction_persistence_scan",
                "seed_count": int(primary[primary["start_condition"].astype(str).eq(start)]["seed"].nunique()),
                "fraction_step_count": int(len(FINE_BRIDGE_FRACTIONS)),
                "predeclared_transient_signature_id": TRANSIENT_SIGNATURE_ID,
                "predeclared_target_signature_id": TARGET_SIGNATURE_ID,
                "expected_final_anchor_variant": "drop_bridge_edges",
                "reverse_hysteresis_executed_in_contract": False,
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
                step_role = "source_bracket"
            elif fraction == 0.75:
                step_role = "previous_coarse_transient_fraction"
            elif fraction > 0.75:
                step_role = "upper_side_persistence_probe"
            elif fraction > 0.5:
                step_role = "lower_side_persistence_probe"
            else:
                step_role = "target_bracket"
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
                        "classify endpoint as source, recurrent transient, target, "
                        "or other using signature and anchor support distances"
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
    rows = [
        {
            "rule_id": "R1_exact_scope",
            "readout_axis": "execution_scope",
            "rule": "execute only local_pair_016, three previously semantic-valid starts, eight seeds, and the nine predeclared bridge fractions",
            "claim_effect": "blocks broad sweep or control expansion in this gate",
        },
        {
            "rule_id": "R2_transient_recurrence",
            "readout_axis": "signature_recurrence",
            "rule": f"count routes where result_endpoint_signature_id == {TRANSIENT_SIGNATURE_ID}",
            "claim_effect": "tests whether the previous step-2 object survives rerun at fine fractions",
        },
        {
            "rule_id": "R3_persistence_band",
            "readout_axis": "fraction_persistence",
            "rule": "finite band requires the transient signature at two or more adjacent fine fractions in a seed route; one fraction only is point-saddle evidence",
            "claim_effect": "separates robust gateway candidate from knife-edge artifact",
        },
        {
            "rule_id": "R4_support_geometry",
            "readout_axis": "endpoint_boundary",
            "rule": "support ties to original/drop-bridge/drop-direct anchors block endpoint-basin promotion",
            "claim_effect": "keeps basin endpoint claims closed",
        },
        {
            "rule_id": "R5_objective_shape",
            "readout_axis": "wall_boundary",
            "rule": "objective debt without recovery blocks positive wall/tunneling interpretation",
            "claim_effect": "keeps wall language closed unless a later reverse/hysteresis gate changes the evidence",
        },
    ]
    return pd.DataFrame(rows)


def _gate_matrix(
    *,
    context: dict[str, Any],
    route_plan: pd.DataFrame,
    fraction_steps: pd.DataFrame,
) -> pd.DataFrame:
    semantic_summary = context["summary"]
    semantic_gates = context["gates"]
    primary_step = context["step_rows"]
    step2 = primary_step[primary_step["step_index"].astype(int).eq(2)]
    fractions = list(map(float, FINE_BRIDGE_FRACTIONS))
    upper = [fraction for fraction in fractions if fraction > 0.75 and fraction < 1.0]
    lower = [fraction for fraction in fractions if fraction < 0.75 and fraction > 0.5]
    return pd.DataFrame(
        [
            _gate_row(
                "G1_upstream_semantic_gates_pass",
                "Did the upstream semantic validation pass before execution design?",
                semantic_gates["gate_status"].value_counts().to_dict(),
                "all semantic gates pass",
                bool(semantic_gates["gate_status"].astype(str).eq("pass").all()),
            ),
            _gate_row(
                "G2_primary_pair_only",
                "Is the contract restricted to the typed-transient candidate only?",
                route_plan["local_pair_id"].value_counts().to_dict(),
                "only local_pair_016 route rows",
                set(route_plan["local_pair_id"].astype(str)) == {PRIMARY_PAIR_ID},
            ),
            _gate_row(
                "G3_semantic_starts_preserved",
                "Are only the three semantic-valid starts predeclared?",
                route_plan["start_condition"].tolist(),
                "bridges_to_left, pair_together, singleton",
                route_plan["start_condition"].tolist() == list(START_CONDITION_ORDER),
            ),
            _gate_row(
                "G4_fine_fraction_bracket",
                "Does the fraction schedule bracket 0.75 without perturbing direct edges?",
                {
                    "fractions": fractions,
                    "upper_side_count": len(upper),
                    "lower_side_count": len(lower),
                    "direct_fraction_values": sorted(
                        fraction_steps["direct_edge_weight_fraction"].unique().tolist()
                    ),
                },
                "at least two upper and two lower side probes, direct fraction fixed at 1.0",
                len(upper) >= 2
                and len(lower) >= 2
                and set(fraction_steps["direct_edge_weight_fraction"].astype(float)) == {1.0},
            ),
            _gate_row(
                "G5_previous_transient_anchor_recorded",
                "Is the previous recurrent step-2 transient signature explicitly recorded?",
                {
                    "semantic_classification": semantic_summary.get("semantic_classification"),
                    "step2_signature": step2["dominant_signature_id"].tolist(),
                },
                f"step-2 signature is {TRANSIENT_SIGNATURE_ID}",
                step2["dominant_signature_id"].astype(str).eq(TRANSIENT_SIGNATURE_ID).all(),
            ),
            _gate_row(
                "G6_claim_boundaries_closed",
                "Are method, quality/cost, wall, and reverse-hysteresis claims closed?",
                {
                    "reverse_hysteresis_executed": bool(
                        route_plan["reverse_hysteresis_executed_in_contract"].any()
                    ),
                    "wall_flags": route_plan[
                        "wall_generality_claim_allowed_after_contract"
                    ].astype(bool).unique().tolist(),
                    "method_flags": route_plan[
                        "method_claim_allowed_after_contract"
                    ].astype(bool).unique().tolist(),
                    "quality_flags": route_plan[
                        "quality_cost_claim_allowed_after_contract"
                    ].astype(bool).unique().tolist(),
                },
                "all claim flags false and reverse hysteresis not executed here",
                not bool(route_plan["reverse_hysteresis_executed_in_contract"].any())
                and not bool(route_plan["wall_generality_claim_allowed_after_contract"].map(_as_bool).any())
                and not bool(route_plan["method_claim_allowed_after_contract"].map(_as_bool).any())
                and not bool(route_plan["quality_cost_claim_allowed_after_contract"].map(_as_bool).any()),
            ),
        ]
    )


def _summary(
    *,
    semantic_dir: Path,
    output_dir: Path,
    route_plan: pd.DataFrame,
    fraction_steps: pd.DataFrame,
    readout_rules: pd.DataFrame,
    gates: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "schema": "nanoclustering_g4_8_first_pass_016_transient_persistence_contract_summary.v1",
        "status": RUN_STATUS,
        "semantic_dir": str(semantic_dir),
        "output_dir": str(output_dir),
        "primary_pair": PRIMARY_PAIR_ID,
        "planned_route_family": PLANNED_ROUTE_FAMILY,
        "route_plan_row_count": int(len(route_plan)),
        "fraction_step_row_count": int(len(fraction_steps)),
        "readout_rule_count": int(len(readout_rules)),
        "fine_bridge_fractions": list(map(float, FINE_BRIDGE_FRACTIONS)),
        "predeclared_transient_signature_id": TRANSIENT_SIGNATURE_ID,
        "predeclared_target_signature_id": TARGET_SIGNATURE_ID,
        "gate_status_counts": gates["gate_status"].value_counts().to_dict(),
        "failed_gates": gates.loc[
            ~gates["gate_status"].astype(str).eq("pass"),
            "gate_id",
        ].tolist(),
        "interpretation": (
            "This predeclares a minimal persistence test for the 016 typed "
            "transient. It is designed to decide whether the recurrent step-2 "
            "signature is a finite fraction band or a single point saddle."
        ),
        "recommended_next_gate": (
            "Run the contract trace. If the transient is a finite band, design "
            "a same-seed target-anchor reverse trace; if it is point-only, treat "
            "reverse hysteresis as lower priority and return to definition design."
        ),
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _write_report(
    *,
    path: Path,
    summary: dict[str, Any],
    route_plan: pd.DataFrame,
    fraction_steps: pd.DataFrame,
    readout_rules: pd.DataFrame,
    gates: pd.DataFrame,
) -> None:
    lines = [
        "# NanoClustering G4.8 First-Pass 016 Transient Persistence Contract",
        "",
        "## Summary",
        "",
        f"- status: {summary['status']}",
        f"- primary_pair: {summary['primary_pair']}",
        f"- route_plan_row_count: {summary['route_plan_row_count']}",
        f"- fraction_step_row_count: {summary['fraction_step_row_count']}",
        f"- failed_gates: {summary['failed_gates']}",
        "",
        "## Route Plan",
        "",
        _markdown_table(
            route_plan,
            [
                "route_contract_id",
                "local_pair_id",
                "start_condition",
                "planned_route_family",
                "seed_count",
                "fraction_step_count",
            ],
        ),
        "",
        "## Fraction Steps",
        "",
        _markdown_table(
            fraction_steps,
            [
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
        summary["claim_boundary"],
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic-dir", type=Path, default=DEFAULT_SEMANTIC_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    semantic_dir = Path(args.semantic_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    context = _load_semantic_context(semantic_dir)
    route_plan = _route_plan(context["route_rows"])
    fraction_steps = _fraction_steps(route_plan)
    readout_rules = _readout_rules()
    gates = _gate_matrix(
        context=context,
        route_plan=route_plan,
        fraction_steps=fraction_steps,
    )
    summary = _summary(
        semantic_dir=semantic_dir,
        output_dir=output_dir,
        route_plan=route_plan,
        fraction_steps=fraction_steps,
        readout_rules=readout_rules,
        gates=gates,
    )
    config = {
        "schema": "nanoclustering_g4_8_first_pass_016_transient_persistence_contract_config.v1",
        "semantic_dir": str(semantic_dir),
        "output_dir": str(output_dir),
        "primary_pair": PRIMARY_PAIR_ID,
        "planned_route_family": PLANNED_ROUTE_FAMILY,
        "fine_bridge_fractions": list(map(float, FINE_BRIDGE_FRACTIONS)),
        "claim_boundary": CLAIM_BOUNDARY,
    }

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
        path=output_dir / REPORT_MD,
        summary=summary,
        route_plan=route_plan,
        fraction_steps=fraction_steps,
        readout_rules=readout_rules,
        gates=gates,
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
