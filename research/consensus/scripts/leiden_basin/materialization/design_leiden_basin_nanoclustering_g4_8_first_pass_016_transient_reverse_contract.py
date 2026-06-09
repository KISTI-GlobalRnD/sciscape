#!/usr/bin/env python3
"""Design the local_pair_016 same-seed reverse trace contract.

This contract follows the 016 transient persistence trace. It predeclares a
target-anchor reverse scan: for each semantic-valid start condition and seed,
initialize from the same-seed ``drop_bridge_edges`` endpoint and restore bridge
edge weight from 0.5 back to 1.0.
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
PLANNED_ROUTE_FAMILY = "first_pass_016_transient_same_seed_target_anchor_reverse_scan"

DEFAULT_PERSISTENCE_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_016_transient_persistence_trace_gamma1e5_20260605"
)
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_016_transient_reverse_contract_gamma1e5_20260605"
)

ROUTE_PLAN_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_016_transient_reverse_contract_route_plan_rows.csv"
)
FRACTION_STEP_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_016_transient_reverse_contract_fraction_step_rows.csv"
)
READOUT_RULE_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_016_transient_reverse_contract_readout_rule_rows.csv"
)
GATE_MATRIX_CSV = (
    "nanoclustering_g4_8_first_pass_016_transient_reverse_contract_gate_matrix.csv"
)
SUMMARY_JSON = "nanoclustering_g4_8_first_pass_016_transient_reverse_contract_summary.json"
CONFIG_JSON = "nanoclustering_g4_8_first_pass_016_transient_reverse_contract_config.json"
REPORT_MD = "nanoclustering_g4_8_first_pass_016_transient_reverse_contract_report.md"

RUN_STATUS = "designed_nanoclustering_g4_8_first_pass_016_transient_reverse_contract"
ROUTE_EXECUTION_STATUS = "not_executed_contract_only_016_transient_reverse"
WALL_PROMOTION_STATUS = "not_promoted_contract_only"
METHOD_STATUS = "contract_design_not_method"
CLAIM_BOUNDARY = (
    "NanoClustering G4.8 first-pass local_pair_016 target-anchor reverse "
    "contract only; predeclares a same-seed drop-bridge-anchor initialization "
    "and ascending bridge-fraction scan. It does not execute Leiden, promote "
    "basin walls, replay full NanoClustering, evaluate quality/cost value, or "
    "claim method/algorithm success."
)

REVERSE_BRIDGE_FRACTIONS = (0.5, 0.625, 0.6875, 0.71875, 0.75, 0.78125, 0.8125, 0.875, 1.0)
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


def _route_plan(persistence_route_rows: pd.DataFrame) -> pd.DataFrame:
    primary = persistence_route_rows[
        persistence_route_rows["local_pair_id"].astype(str).eq(PRIMARY_PAIR_ID)
    ]
    observed_starts = set(primary["start_condition"].astype(str))
    starts = [start for start in START_CONDITION_ORDER if start in observed_starts]
    rows: list[dict[str, Any]] = []
    for order, start in enumerate(starts, start=1):
        route_contract_id = f"{PRIMARY_PAIR_ID}__{start}__{PLANNED_ROUTE_FAMILY}"
        start_rows = primary[primary["start_condition"].astype(str).eq(start)]
        rows.append(
            {
                "route_contract_id": route_contract_id,
                "validation_unit_id": route_contract_id,
                "local_pair_id": PRIMARY_PAIR_ID,
                "contract_pair_role": "primary_typed_transient_reverse_candidate",
                "start_condition": start,
                "planned_route_family": PLANNED_ROUTE_FAMILY,
                "route_family_order": int(order),
                "route_family_role": "same_seed_target_anchor_reverse_scan",
                "seed_count": int(start_rows["seed"].nunique()),
                "fraction_step_count": int(len(REVERSE_BRIDGE_FRACTIONS)),
                "initial_anchor_variant": "drop_bridge_edges",
                "expected_final_anchor_variant": "original",
                "predeclared_transient_signature_id": TRANSIENT_SIGNATURE_ID,
                "predeclared_target_signature_id": TARGET_SIGNATURE_ID,
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
        for index, fraction in enumerate(REVERSE_BRIDGE_FRACTIONS, start=1):
            if fraction == 0.5:
                step_role = "target_anchor_start_bracket"
            elif fraction < 0.75:
                step_role = "lower_side_reverse_probe"
            elif fraction == 0.75:
                step_role = "previous_coarse_transient_fraction"
            elif fraction < 1.0:
                step_role = "upper_side_reverse_probe"
            else:
                step_role = "source_anchor_final_bracket"
            rows.append(
                {
                    **route_data,
                    "step_index": int(index),
                    "step_label": f"reverse_bridge_fraction_{fraction:.5g}",
                    "direct_edge_weight_fraction": 1.0,
                    "bridge_edge_weight_fraction": float(fraction),
                    "initial_anchor_variant": "drop_bridge_edges",
                    "expected_final_anchor_variant": "original",
                    "step_role": step_role,
                    "predeclared_readout": (
                        "classify target-anchor initialization under restored "
                        "bridge fractions as target, transient, source, or other"
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
            "rule_id": "R1_same_seed_target_initialization",
            "readout_axis": "initial_condition",
            "rule": "use the same-seed drop_bridge_edges endpoint membership as the reverse initial membership",
            "claim_effect": "keeps reverse evidence tied to the previously observed target anchor",
        },
        {
            "rule_id": "R2_reverse_endpoint_sequence",
            "readout_axis": "reverse_path_shape",
            "rule": "record whether target initialization remains target, enters the transient band, or returns to source as bridge weight is restored",
            "claim_effect": "tests reversibility/hysteresis without promoting a wall",
        },
        {
            "rule_id": "R3_forward_band_contrast",
            "readout_axis": "pathway_asymmetry",
            "rule": "compare reverse classifications against the forward finite transient band from the persistence trace",
            "claim_effect": "supports pathway-mechanism design only if asymmetry is explicit",
        },
        {
            "rule_id": "R4_objective_shape",
            "readout_axis": "wall_boundary",
            "rule": "reverse objective debt/recovery is diagnostic only and cannot alone promote wall language",
            "claim_effect": "keeps wall/tunneling claims closed",
        },
    ]
    return pd.DataFrame(rows)


def _gate_matrix(
    *,
    persistence_summary: dict[str, Any],
    persistence_gates: pd.DataFrame,
    route_plan: pd.DataFrame,
    fraction_steps: pd.DataFrame,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            _gate_row(
                "G1_upstream_persistence_trace_passed",
                "Did the upstream 016 persistence trace pass before reverse design?",
                {
                    "semantic_persistence_class": persistence_summary.get(
                        "semantic_persistence_class"
                    ),
                    "persistence_gate_status_counts": persistence_gates[
                        "gate_status"
                    ].value_counts().to_dict(),
                },
                "persistence trace gates pass and finite band was observed",
                bool(persistence_gates["gate_status"].astype(str).eq("pass").all())
                and persistence_summary.get("semantic_persistence_class")
                == "persistent_finite_saddle_band_candidate_not_wall",
            ),
            _gate_row(
                "G2_primary_pair_only",
                "Is the reverse contract restricted to local_pair_016 only?",
                route_plan["local_pair_id"].value_counts().to_dict(),
                "only local_pair_016 route rows",
                set(route_plan["local_pair_id"].astype(str)) == {PRIMARY_PAIR_ID},
            ),
            _gate_row(
                "G3_same_start_seed_grid",
                "Does the reverse contract preserve the same starts and seed count?",
                {
                    "starts": route_plan["start_condition"].tolist(),
                    "seed_counts": route_plan["seed_count"].tolist(),
                },
                "three semantic-valid starts, eight seeds each",
                route_plan["start_condition"].tolist() == list(START_CONDITION_ORDER)
                and route_plan["seed_count"].astype(int).eq(8).all(),
            ),
            _gate_row(
                "G4_reverse_fraction_schedule",
                "Does the reverse schedule restore bridge fraction from target to source bracket?",
                fraction_steps[
                    [
                        "step_index",
                        "bridge_edge_weight_fraction",
                        "direct_edge_weight_fraction",
                    ]
                ].drop_duplicates().to_dict("records"),
                "0.5 -> 1.0 bridge fractions, direct fraction fixed at 1.0",
                tuple(
                    fraction_steps.drop_duplicates("step_index")
                    .sort_values("step_index", kind="mergesort")[
                        "bridge_edge_weight_fraction"
                    ]
                    .astype(float)
                )
                == REVERSE_BRIDGE_FRACTIONS
                and set(fraction_steps["direct_edge_weight_fraction"].astype(float)) == {1.0},
            ),
            _gate_row(
                "G5_claim_boundaries_closed",
                "Are method, quality/cost, and wall claims closed in the reverse contract?",
                {
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
                "all claim flags false",
                not bool(route_plan["wall_generality_claim_allowed_after_contract"].map(_as_bool).any())
                and not bool(route_plan["method_claim_allowed_after_contract"].map(_as_bool).any())
                and not bool(route_plan["quality_cost_claim_allowed_after_contract"].map(_as_bool).any()),
            ),
        ]
    )


def _summary(
    *,
    persistence_dir: Path,
    output_dir: Path,
    route_plan: pd.DataFrame,
    fraction_steps: pd.DataFrame,
    readout_rules: pd.DataFrame,
    gates: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "schema": "nanoclustering_g4_8_first_pass_016_transient_reverse_contract_summary.v1",
        "status": RUN_STATUS,
        "persistence_dir": str(persistence_dir),
        "output_dir": str(output_dir),
        "primary_pair": PRIMARY_PAIR_ID,
        "planned_route_family": PLANNED_ROUTE_FAMILY,
        "route_plan_row_count": int(len(route_plan)),
        "fraction_step_row_count": int(len(fraction_steps)),
        "readout_rule_count": int(len(readout_rules)),
        "reverse_bridge_fractions": list(map(float, REVERSE_BRIDGE_FRACTIONS)),
        "initial_anchor_variant": "drop_bridge_edges",
        "expected_final_anchor_variant": "original",
        "gate_status_counts": gates["gate_status"].value_counts().to_dict(),
        "failed_gates": gates.loc[
            ~gates["gate_status"].astype(str).eq("pass"),
            "gate_id",
        ].tolist(),
        "interpretation": (
            "This predeclares a reverse initialization check for the 016 finite "
            "transient band. It tests path asymmetry and hysteresis only."
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
        "# NanoClustering G4.8 First-Pass 016 Transient Reverse Contract",
        "",
        "## Summary",
        "",
        f"- status: {summary['status']}",
        f"- initial_anchor_variant: {summary['initial_anchor_variant']}",
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
                "initial_anchor_variant",
                "seed_count",
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
                "bridge_edge_weight_fraction",
                "initial_anchor_variant",
                "expected_final_anchor_variant",
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
    parser.add_argument("--persistence-dir", type=Path, default=DEFAULT_PERSISTENCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    persistence_dir = Path(args.persistence_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    persistence_summary = _read_json(
        persistence_dir / "nanoclustering_g4_8_first_pass_016_transient_persistence_summary.json"
    )
    persistence_gates = _read_csv(
        persistence_dir / "nanoclustering_g4_8_first_pass_016_transient_persistence_gate_matrix.csv"
    )
    persistence_route_rows = _read_csv(
        persistence_dir / "nanoclustering_g4_8_first_pass_016_transient_persistence_route_rows.csv"
    )
    route_plan = _route_plan(persistence_route_rows)
    fraction_steps = _fraction_steps(route_plan)
    readout_rules = _readout_rules()
    gates = _gate_matrix(
        persistence_summary=persistence_summary,
        persistence_gates=persistence_gates,
        route_plan=route_plan,
        fraction_steps=fraction_steps,
    )
    summary = _summary(
        persistence_dir=persistence_dir,
        output_dir=output_dir,
        route_plan=route_plan,
        fraction_steps=fraction_steps,
        readout_rules=readout_rules,
        gates=gates,
    )
    config = {
        "schema": "nanoclustering_g4_8_first_pass_016_transient_reverse_contract_config.v1",
        "persistence_dir": str(persistence_dir),
        "output_dir": str(output_dir),
        "primary_pair": PRIMARY_PAIR_ID,
        "planned_route_family": PLANNED_ROUTE_FAMILY,
        "reverse_bridge_fractions": list(map(float, REVERSE_BRIDGE_FRACTIONS)),
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
