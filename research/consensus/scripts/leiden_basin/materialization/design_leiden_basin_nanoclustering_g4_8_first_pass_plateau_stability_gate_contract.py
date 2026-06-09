#!/usr/bin/env python3
"""Design the plateau-stability feature gate contract.

This contract follows the read-only plateau-stability feature audit. It fixes
the predicates that must be satisfied before a future candidate can be called a
``016``-like finite plateau recurrence. It is intentionally not an execution
runner and not a wall/method contract.
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
NEAR_MISS_PAIR_IDS = ("local_pair_009", "local_pair_012", "local_pair_014", "local_pair_020")
BOUNDARY_GUARD_PAIR_ID = "local_pair_005"
ALL_CONTRACT_PAIR_IDS = (PRIMARY_PAIR_ID, *NEAR_MISS_PAIR_IDS, BOUNDARY_GUARD_PAIR_ID)

DEFAULT_PLATEAU_AUDIT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_plateau_stability_feature_audit_gamma1e5_20260606"
)
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_plateau_stability_gate_contract_gamma1e5_20260606"
)

FEATURE_PREDICATE_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_plateau_stability_gate_contract_feature_predicate_rows.csv"
)
CONTRAST_CASE_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_plateau_stability_gate_contract_contrast_case_rows.csv"
)
EVALUATION_RULE_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_plateau_stability_gate_contract_evaluation_rule_rows.csv"
)
GATE_MATRIX_CSV = (
    "nanoclustering_g4_8_first_pass_plateau_stability_gate_contract_gate_matrix.csv"
)
SUMMARY_JSON = "nanoclustering_g4_8_first_pass_plateau_stability_gate_contract_summary.json"
CONFIG_JSON = "nanoclustering_g4_8_first_pass_plateau_stability_gate_contract_config.json"
REPORT_MD = "nanoclustering_g4_8_first_pass_plateau_stability_gate_contract_report.md"

RUN_STATUS = "designed_nanoclustering_g4_8_first_pass_plateau_stability_gate_contract"
ROUTE_EXECUTION_STATUS = "not_executed_contract_only_plateau_stability_gate"
WALL_PROMOTION_STATUS = "not_promoted_contract_only"
METHOD_STATUS = "plateau_stability_gate_contract_not_method"
CLAIM_BOUNDARY = (
    "NanoClustering G4.8 first-pass plateau-stability gate contract only; "
    "predeclares feature predicates and near-miss controls for interpreting "
    "finite single-side plateau recurrence. It does not execute Leiden, "
    "promote basin walls, replay full NanoClustering, evaluate quality/cost "
    "value, or claim method success."
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
        "observed": json.dumps(_json_safe(observed), sort_keys=True),
        "minimum_or_rule": minimum_or_rule,
        "gate_status": "pass" if passed else "fail",
    }


def _markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 50) -> str:
    cols = [column for column in columns if column in frame.columns]
    if not cols:
        return "_No matching columns._"
    visible = frame[cols].head(max_rows)
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


def _load_context(plateau_audit_dir: Path) -> dict[str, Any]:
    return {
        "summary": _read_json(
            plateau_audit_dir
            / "nanoclustering_g4_8_first_pass_plateau_stability_feature_summary.json"
        ),
        "pair_rows": _read_csv(
            plateau_audit_dir
            / "nanoclustering_g4_8_first_pass_plateau_stability_feature_pair_rows.csv"
        ),
        "fraction_rows": _read_csv(
            plateau_audit_dir
            / "nanoclustering_g4_8_first_pass_plateau_stability_feature_fraction_rows.csv"
        ),
        "gate_matrix": _read_csv(
            plateau_audit_dir
            / "nanoclustering_g4_8_first_pass_plateau_stability_feature_gate_matrix.csv"
        ),
        "decision_rows": _read_csv(
            plateau_audit_dir
            / "nanoclustering_g4_8_first_pass_plateau_stability_feature_decision_rows.csv"
        ),
    }


def _feature_predicate_rows(primary_row: pd.Series) -> pd.DataFrame:
    rows = [
        {
            "predicate_id": "P1_fixed_local_signature_screen",
            "predicate_scope": "candidate_precondition",
            "predicate_definition": "fixed_016_local_signature_pass == true",
            "minimum_or_rule": "required before route-level plateau interpretation",
            "positive_reference_observed": bool(_as_bool(primary_row["fixed_016_local_signature_pass"])),
            "negative_or_guard_role": "screens out boundary/control nonanalogs but is not sufficient",
            "claim_boundary": CLAIM_BOUNDARY,
            "run_status": RUN_STATUS,
        },
        {
            "predicate_id": "P2_source_and_target_brackets",
            "predicate_scope": "route_ladder",
            "predicate_definition": (
                "at least one all-route source-family fraction above the band and "
                "at least one all-route target-like fraction below the band"
            ),
            "minimum_or_rule": "source bracket count >=1 and target bracket count >=1",
            "positive_reference_observed": (
                int(primary_row["all_source_fraction_count"]) >= 1
                and int(primary_row["all_target_fraction_count"]) >= 1
            ),
            "negative_or_guard_role": "blocks single-side observations without route ladder context",
            "claim_boundary": CLAIM_BOUNDARY,
            "run_status": RUN_STATUS,
        },
        {
            "predicate_id": "P3_finite_all_route_single_side_band",
            "predicate_scope": "route_ladder",
            "predicate_definition": (
                "two or more adjacent bridge fractions have all routes in "
                "pair-separated single-side state"
            ),
            "minimum_or_rule": "all_route_single_side_fraction_count >= 2",
            "positive_reference_observed": int(
                primary_row["all_route_single_side_fraction_count"]
            )
            >= 2,
            "negative_or_guard_role": "rejects point-only or seed-fragile single-side events",
            "claim_boundary": CLAIM_BOUNDARY,
            "run_status": RUN_STATUS,
        },
        {
            "predicate_id": "P4_exact_single_bridge_latch",
            "predicate_scope": "transition_state",
            "predicate_definition": (
                "every single-side row has the same latch signature, currently "
                "left=1;right=0;pair=0"
            ),
            "minimum_or_rule": "single_side_exact_latch == true and latch signature fixed",
            "positive_reference_observed": (
                bool(_as_bool(primary_row["single_side_exact_latch"]))
                and str(primary_row["single_side_latch_signature"]) == "left=1;right=0;pair=0"
            ),
            "negative_or_guard_role": "rejects variable or opposite-side latch events",
            "claim_boundary": CLAIM_BOUNDARY,
            "run_status": RUN_STATUS,
        },
        {
            "predicate_id": "P5_anchor_equidistant_support_geometry",
            "predicate_scope": "transition_state",
            "predicate_definition": (
                "single-side rows have equal support distance to original, "
                "drop-bridge, and drop-direct known anchors"
            ),
            "minimum_or_rule": "single_side_support_distance_equal_all_known_anchors == true",
            "positive_reference_observed": bool(
                _as_bool(primary_row["single_side_support_distance_equal_all_known_anchors"])
            ),
            "negative_or_guard_role": "separates a stable transition state from anchor-near endpoint states",
            "claim_boundary": CLAIM_BOUNDARY,
            "run_status": RUN_STATUS,
        },
        {
            "predicate_id": "P6_seed_start_stable_band",
            "predicate_scope": "reproducibility",
            "predicate_definition": (
                "finite band appears across all predeclared starts and seeds in the "
                "reference route surface"
            ),
            "minimum_or_rule": "seed_start_stable_finite_plateau == true",
            "positive_reference_observed": bool(
                _as_bool(primary_row["seed_start_stable_finite_plateau"])
            ),
            "negative_or_guard_role": "rejects seed-only or start-only route artifacts",
            "claim_boundary": CLAIM_BOUNDARY,
            "run_status": RUN_STATUS,
        },
    ]
    return pd.DataFrame(rows)


def _contrast_role(row: pd.Series) -> str:
    pair_id = str(row["local_pair_id"])
    if pair_id == PRIMARY_PAIR_ID:
        return "positive_reference_exact_plateau"
    if pair_id == "local_pair_009":
        return "abrupt_high_threshold_source_to_target_near_miss"
    if pair_id == "local_pair_012":
        return "point_or_seed_fragile_single_side_near_miss"
    if pair_id == "local_pair_014":
        return "same_latch_point_event_positive_reference_control"
    if pair_id == "local_pair_020":
        return "object_dominant_abrupt_low_threshold_near_miss"
    if pair_id == BOUNDARY_GUARD_PAIR_ID:
        return "boundary_guard_source_family_absent"
    return "other"


def _contrast_case_rows(pair_rows: pd.DataFrame) -> pd.DataFrame:
    scoped = pair_rows[pair_rows["local_pair_id"].astype(str).isin(ALL_CONTRACT_PAIR_IDS)].copy()
    rows: list[dict[str, Any]] = []
    for _, row in scoped.iterrows():
        rows.append(
            {
                "local_pair_id": row["local_pair_id"],
                "contract_contrast_role": _contrast_role(row),
                "pair_scope": row["pair_scope"],
                "fixed_016_local_signature_pass": _as_bool(row["fixed_016_local_signature_pass"]),
                "bridge_to_direct_weight_ratio": row["bridge_to_direct_weight_ratio"],
                "direct_cpm_delta_q": row["direct_cpm_delta_q"],
                "selected_bridge_object_weight_share": row[
                    "selected_bridge_object_weight_share"
                ],
                "all_source_fraction_count": int(row["all_source_fraction_count"]),
                "all_route_single_side_fraction_count": int(
                    row["all_route_single_side_fraction_count"]
                ),
                "any_single_side_fraction_count": int(row["any_single_side_fraction_count"]),
                "all_target_fraction_count": int(row["all_target_fraction_count"]),
                "single_side_latch_signature": row["single_side_latch_signature"],
                "single_side_exact_latch": _as_bool(row["single_side_exact_latch"]),
                "single_side_support_distance_equal_all_known_anchors": _as_bool(
                    row["single_side_support_distance_equal_all_known_anchors"]
                ),
                "seed_start_stable_finite_plateau": _as_bool(
                    row["seed_start_stable_finite_plateau"]
                ),
                "pair_explanation_class": row["pair_explanation_class"],
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "method_status": METHOD_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
                "run_status": RUN_STATUS,
            }
        )
    return pd.DataFrame(rows)


def _evaluation_rule_rows() -> pd.DataFrame:
    rows = [
        {
            "rule_id": "R1_positive_plateau_candidate_acceptance",
            "rule_type": "acceptance",
            "rule_definition": (
                "Accept a future candidate as 016-like plateau recurrence only if "
                "P1-P6 all pass under the predeclared route/fraction readout."
            ),
            "failure_interpretation": "If any predicate fails, classify as near miss or nonanalog.",
            "claim_allowed_after_rule": "route-level plateau recurrence only; no wall or method claim",
            "claim_boundary": CLAIM_BOUNDARY,
            "run_status": RUN_STATUS,
        },
        {
            "rule_id": "R2_point_single_side_blocker",
            "rule_type": "blocker",
            "rule_definition": (
                "A candidate with any single-side rows but zero all-route single-side "
                "fractions is point/partial evidence, not finite plateau evidence."
            ),
            "failure_interpretation": "Use local_pair_012, local_pair_014, and local_pair_005 as controls.",
            "claim_allowed_after_rule": "diagnostic near-miss classification",
            "claim_boundary": CLAIM_BOUNDARY,
            "run_status": RUN_STATUS,
        },
        {
            "rule_id": "R3_abrupt_switch_blocker",
            "rule_type": "blocker",
            "rule_definition": (
                "A candidate with source and target brackets but no single-side band "
                "is an abrupt source-to-target switch, not plateau recurrence."
            ),
            "failure_interpretation": "Use local_pair_009 and local_pair_020 as controls.",
            "claim_allowed_after_rule": "diagnostic near-miss classification",
            "claim_boundary": CLAIM_BOUNDARY,
            "run_status": RUN_STATUS,
        },
        {
            "rule_id": "R4_scalar_feature_blocker",
            "rule_type": "blocker",
            "rule_definition": (
                "Bridge/direct ratio, direct delta, and selected-bridge scope mix "
                "cannot independently promote a plateau claim."
            ),
            "failure_interpretation": "Scalar features are screens until tied to route morphology.",
            "claim_allowed_after_rule": "screening evidence only",
            "claim_boundary": CLAIM_BOUNDARY,
            "run_status": RUN_STATUS,
        },
        {
            "rule_id": "R5_near_miss_specificity_guard",
            "rule_type": "specificity_guard",
            "rule_definition": (
                "Near misses 009, 012, 014, and 020 must remain negative under the "
                "fixed feature predicates before widening the candidate panel."
            ),
            "failure_interpretation": "If guards leak, revise the predicate before expansion.",
            "claim_allowed_after_rule": "predicate specificity only",
            "claim_boundary": CLAIM_BOUNDARY,
            "run_status": RUN_STATUS,
        },
    ]
    return pd.DataFrame(rows)


def _build_gates(
    *,
    source_gate_matrix: pd.DataFrame,
    feature_rows: pd.DataFrame,
    contrast_rows: pd.DataFrame,
    evaluation_rows: pd.DataFrame,
) -> pd.DataFrame:
    source_failed = source_gate_matrix[source_gate_matrix["gate_status"].astype(str).ne("pass")]
    primary = contrast_rows[contrast_rows["local_pair_id"].astype(str).eq(PRIMARY_PAIR_ID)]
    near_misses = contrast_rows[
        contrast_rows["local_pair_id"].astype(str).isin(NEAR_MISS_PAIR_IDS)
    ]
    predicates_complete = set(feature_rows["predicate_id"]) == {
        "P1_fixed_local_signature_screen",
        "P2_source_and_target_brackets",
        "P3_finite_all_route_single_side_band",
        "P4_exact_single_bridge_latch",
        "P5_anchor_equidistant_support_geometry",
        "P6_seed_start_stable_band",
    }
    near_miss_roles = set(near_misses["contract_contrast_role"])
    required_near_miss_roles = {
        "abrupt_high_threshold_source_to_target_near_miss",
        "point_or_seed_fragile_single_side_near_miss",
        "same_latch_point_event_positive_reference_control",
        "object_dominant_abrupt_low_threshold_near_miss",
    }
    gates = [
        _gate_row(
            "G1_source_feature_audit_passed",
            "Did the source plateau-stability feature audit pass all gates?",
            {"failed_source_gates": source_failed["gate_id"].tolist()},
            "no failed source audit gates",
            source_failed.empty,
        ),
        _gate_row(
            "G2_positive_reference_predicates_declared",
            "Are all positive-reference predicates predeclared and observed on 016?",
            {
                "predicate_ids": feature_rows["predicate_id"].tolist(),
                "positive_reference_observed_all": bool(
                    feature_rows["positive_reference_observed"].map(_as_bool).all()
                ),
            },
            "P1-P6 exist and are true on 016",
            predicates_complete
            and bool(feature_rows["positive_reference_observed"].map(_as_bool).all())
            and not primary.empty,
        ),
        _gate_row(
            "G3_near_miss_specificity_set_declared",
            "Are the near-miss contrast roles broad enough for this feature gate?",
            {"near_miss_roles": sorted(near_miss_roles)},
            "abrupt, point/fragile, same-latch point, and object-dominant abrupt guards",
            required_near_miss_roles.issubset(near_miss_roles),
        ),
        _gate_row(
            "G4_no_near_miss_has_all_route_plateau",
            "Do all near misses stay negative for all-route single-side plateau?",
            near_misses[
                ["local_pair_id", "all_route_single_side_fraction_count"]
            ].to_dict(orient="records"),
            "all near misses have all_route_single_side_fraction_count == 0",
            bool(near_misses["all_route_single_side_fraction_count"].eq(0).all()),
        ),
        _gate_row(
            "G5_evaluation_rules_close_failed_directions",
            "Do evaluation rules block point-only, abrupt-switch, and scalar-only claims?",
            evaluation_rows[["rule_id", "rule_type"]].to_dict(orient="records"),
            "acceptance plus blocker/specificity rules are declared",
            {
                "acceptance",
                "blocker",
                "specificity_guard",
            }.issubset(set(evaluation_rows["rule_type"])),
        ),
        _gate_row(
            "G6_claim_boundaries_closed",
            "Are wall, method, quality/cost, and full-replay claims closed?",
            CLAIM_BOUNDARY,
            "contract-only feature gate",
            True,
        ),
    ]
    return pd.DataFrame(gates)


def _write_report(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    feature_rows: pd.DataFrame,
    contrast_rows: pd.DataFrame,
    evaluation_rows: pd.DataFrame,
    gates: pd.DataFrame,
) -> None:
    lines = [
        "# NanoClustering G4.8 First-Pass Plateau-Stability Gate Contract",
        "",
        "## Summary",
        "",
        f"- status: {summary['status']}",
        f"- contract_status: {summary['contract_status']}",
        f"- failed_gates: {summary['failed_gates']}",
        "",
        "## Feature Predicates",
        "",
        _markdown_table(
            feature_rows,
            [
                "predicate_id",
                "predicate_scope",
                "minimum_or_rule",
                "positive_reference_observed",
                "negative_or_guard_role",
            ],
        ),
        "",
        "## Contrast Cases",
        "",
        _markdown_table(
            contrast_rows,
            [
                "local_pair_id",
                "contract_contrast_role",
                "all_route_single_side_fraction_count",
                "single_side_latch_signature",
                "seed_start_stable_finite_plateau",
                "pair_explanation_class",
            ],
        ),
        "",
        "## Evaluation Rules",
        "",
        _markdown_table(
            evaluation_rows,
            ["rule_id", "rule_type", "rule_definition", "claim_allowed_after_rule"],
        ),
        "",
        "## Gates",
        "",
        _markdown_table(gates, ["gate_id", "gate_status", "question", "minimum_or_rule"]),
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


def run_design(
    plateau_audit_dir: Path = DEFAULT_PLATEAU_AUDIT_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    context = _load_context(plateau_audit_dir)
    source_summary = context["summary"]
    pair_rows = context["pair_rows"]
    source_gate_matrix = context["gate_matrix"]

    primary_rows = pair_rows[pair_rows["local_pair_id"].astype(str).eq(PRIMARY_PAIR_ID)]
    if primary_rows.empty:
        raise ValueError(f"missing primary pair row: {PRIMARY_PAIR_ID}")
    primary_row = primary_rows.iloc[0]

    feature_rows = _feature_predicate_rows(primary_row)
    contrast_rows = _contrast_case_rows(pair_rows)
    evaluation_rows = _evaluation_rule_rows()
    gates = _build_gates(
        source_gate_matrix=source_gate_matrix,
        feature_rows=feature_rows,
        contrast_rows=contrast_rows,
        evaluation_rows=evaluation_rows,
    )

    _write_csv(feature_rows, output_dir / FEATURE_PREDICATE_ROWS_CSV)
    _write_csv(contrast_rows, output_dir / CONTRAST_CASE_ROWS_CSV)
    _write_csv(evaluation_rows, output_dir / EVALUATION_RULE_ROWS_CSV)
    _write_csv(gates, output_dir / GATE_MATRIX_CSV)

    failed_gates = gates.loc[gates["gate_status"].astype(str).ne("pass"), "gate_id"].tolist()
    summary = {
        "schema": "nanoclustering_g4_8_first_pass_plateau_stability_gate_contract_summary.v1",
        "status": RUN_STATUS,
        "contract_status": "plateau_stability_feature_gate_predeclared",
        "failed_gates": failed_gates,
        "gate_status_counts": gates["gate_status"].value_counts().to_dict(),
        "interpretation": (
            "The next executable or readout step must evaluate exact latch "
            "persistence, anchor-equidistant support geometry, and seed/start "
            "stable band width before any candidate expansion. Point-only, "
            "abrupt-switch, and scalar-only evidence are explicitly blocked."
        ),
        "recommended_next_gate": (
            "Apply this contract as the fixed readout vocabulary for any future "
            "plateau-stability candidate: require P1-P6, keep 009/012/014/020 "
            "as near-miss guards, and do not promote wall or method claims."
        ),
        "source_statuses": {
            "plateau_audit": source_summary.get("status"),
        },
        "source_dirs": {
            "plateau_audit_dir": str(plateau_audit_dir),
        },
        "output_dir": str(output_dir),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    config = {
        "primary_pair_id": PRIMARY_PAIR_ID,
        "near_miss_pair_ids": list(NEAR_MISS_PAIR_IDS),
        "boundary_guard_pair_id": BOUNDARY_GUARD_PAIR_ID,
        "all_contract_pair_ids": list(ALL_CONTRACT_PAIR_IDS),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(
        output_dir=output_dir,
        summary=summary,
        feature_rows=feature_rows,
        contrast_rows=contrast_rows,
        evaluation_rows=evaluation_rows,
        gates=gates,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plateau-audit-dir", type=Path, default=DEFAULT_PLATEAU_AUDIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_design(
        plateau_audit_dir=Path(args.plateau_audit_dir),
        output_dir=Path(args.output_dir),
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
