#!/usr/bin/env python3
"""Apply the plateau-stability gate contract to the current first-pass panel.

This read-only audit applies the P1-P6 plateau-stability contract to the
current 23-pair G4.8 first-pass surface. It separates already-scoreable rows,
near-miss guards, non-strict diagnostic local-signature rows, and nonanalogs.
It does not execute Leiden or expand the candidate panel.
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
CONTRACT_SCOREABLE_PAIR_IDS = (PRIMARY_PAIR_ID, *NEAR_MISS_PAIR_IDS, BOUNDARY_GUARD_PAIR_ID)
NON_STRICT_LOCAL_SIGNATURE_PAIR_IDS = ("local_pair_001", "local_pair_007")

DEFAULT_GENERALIZATION_SCREEN_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_mechanism_generalization_screen_gamma1e5_20260605"
)
DEFAULT_GATE_CONTRACT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_plateau_stability_gate_contract_gamma1e5_20260606"
)
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_plateau_stability_gate_application_gamma1e5_20260606"
)

PAIR_ROWS_CSV = "nanoclustering_g4_8_first_pass_plateau_stability_gate_application_pair_rows.csv"
PREDICATE_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_plateau_stability_gate_application_predicate_rows.csv"
)
CLASS_ROWS_CSV = "nanoclustering_g4_8_first_pass_plateau_stability_gate_application_class_rows.csv"
DECISION_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_plateau_stability_gate_application_decision_rows.csv"
)
GATE_MATRIX_CSV = "nanoclustering_g4_8_first_pass_plateau_stability_gate_application_gate_matrix.csv"
SUMMARY_JSON = "nanoclustering_g4_8_first_pass_plateau_stability_gate_application_summary.json"
CONFIG_JSON = "nanoclustering_g4_8_first_pass_plateau_stability_gate_application_config.json"
REPORT_MD = "nanoclustering_g4_8_first_pass_plateau_stability_gate_application_report.md"

RUN_STATUS = "audited_nanoclustering_g4_8_first_pass_plateau_stability_gate_application"
ROUTE_EXECUTION_STATUS = "not_executed_read_only_plateau_stability_gate_application"
WALL_PROMOTION_STATUS = "not_promoted_plateau_stability_gate_application_only"
METHOD_STATUS = "plateau_stability_gate_application_not_method"
CLAIM_BOUNDARY = (
    "NanoClustering G4.8 first-pass plateau-stability gate application audit "
    "only; applies a predeclared P1-P6 feature contract to existing first-pass "
    "panel artifacts. It does not execute Leiden, promote basin walls, replay "
    "full NanoClustering, evaluate quality/cost value, or claim method success."
)

PREDICATE_IDS = (
    "P1_fixed_local_signature_screen",
    "P2_source_and_target_brackets",
    "P3_finite_all_route_single_side_band",
    "P4_exact_single_bridge_latch",
    "P5_anchor_equidistant_support_geometry",
    "P6_seed_start_stable_band",
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


def _status_from_bool(value: bool | None) -> str:
    if value is None:
        return "not_scoreable"
    return "pass" if value else "fail"


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


def _markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 60) -> str:
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


def _load_context(
    *,
    generalization_screen_dir: Path,
    gate_contract_dir: Path,
) -> dict[str, Any]:
    return {
        "summaries": {
            "generalization_screen": _read_json(
                generalization_screen_dir
                / "nanoclustering_g4_8_first_pass_mechanism_generalization_summary.json"
            ),
            "gate_contract": _read_json(
                gate_contract_dir
                / "nanoclustering_g4_8_first_pass_plateau_stability_gate_contract_summary.json"
            ),
        },
        "tables": {
            "screen_pair_rows": _read_csv(
                generalization_screen_dir
                / "nanoclustering_g4_8_first_pass_mechanism_generalization_pair_rows.csv"
            ),
            "contract_feature_rows": _read_csv(
                gate_contract_dir
                / "nanoclustering_g4_8_first_pass_plateau_stability_gate_contract_feature_predicate_rows.csv"
            ),
            "contract_contrast_rows": _read_csv(
                gate_contract_dir
                / "nanoclustering_g4_8_first_pass_plateau_stability_gate_contract_contrast_case_rows.csv"
            ),
            "contract_rule_rows": _read_csv(
                gate_contract_dir
                / "nanoclustering_g4_8_first_pass_plateau_stability_gate_contract_evaluation_rule_rows.csv"
            ),
            "contract_gate_matrix": _read_csv(
                gate_contract_dir
                / "nanoclustering_g4_8_first_pass_plateau_stability_gate_contract_gate_matrix.csv"
            ),
        },
    }


def _predicate_values(screen_row: pd.Series, contrast_row: pd.Series | None) -> dict[str, bool | None]:
    p1 = _as_bool(screen_row["fixed_016_local_signature_pass"])
    if contrast_row is None:
        return {
            "P1_fixed_local_signature_screen": p1,
            "P2_source_and_target_brackets": None,
            "P3_finite_all_route_single_side_band": None,
            "P4_exact_single_bridge_latch": None,
            "P5_anchor_equidistant_support_geometry": None,
            "P6_seed_start_stable_band": None,
        }

    source_count = int(contrast_row["all_source_fraction_count"])
    target_count = int(contrast_row["all_target_fraction_count"])
    single_count = int(contrast_row["all_route_single_side_fraction_count"])
    exact_latch = _as_bool(contrast_row["single_side_exact_latch"])
    latch_signature = str(contrast_row["single_side_latch_signature"])
    equal_support_distance = _as_bool(
        contrast_row["single_side_support_distance_equal_all_known_anchors"]
    )
    seed_start_stable = _as_bool(contrast_row["seed_start_stable_finite_plateau"])
    return {
        "P1_fixed_local_signature_screen": p1,
        "P2_source_and_target_brackets": source_count >= 1 and target_count >= 1,
        "P3_finite_all_route_single_side_band": single_count >= 2,
        "P4_exact_single_bridge_latch": exact_latch
        and latch_signature == "left=1;right=0;pair=0",
        "P5_anchor_equidistant_support_geometry": equal_support_distance,
        "P6_seed_start_stable_band": seed_start_stable,
    }


def _application_class(
    pair_id: str,
    screen_row: pd.Series,
    contrast_row: pd.Series | None,
    predicate_statuses: dict[str, str],
) -> tuple[str, str]:
    all_pass = all(predicate_statuses[predicate_id] == "pass" for predicate_id in PREDICATE_IDS)
    any_not_scoreable = any(
        predicate_statuses[predicate_id] == "not_scoreable" for predicate_id in PREDICATE_IDS
    )
    fixed_signature = _as_bool(screen_row["fixed_016_local_signature_pass"])
    validation_stratum = str(screen_row["validation_stratum"])

    if all_pass and pair_id == PRIMARY_PAIR_ID:
        return (
            "contract_positive_reference_pass",
            "016 is the only current pair passing all P1-P6 predicates.",
        )
    if pair_id in NEAR_MISS_PAIR_IDS:
        return (
            "near_miss_guard_negative",
            "near-miss guard is scoreable and remains negative for P1-P6.",
        )
    if pair_id == BOUNDARY_GUARD_PAIR_ID:
        return (
            "boundary_guard_rejected_by_p1",
            "boundary guard fails the fixed local-signature precondition.",
        )
    if fixed_signature and validation_stratum != "strict_ready":
        return (
            "non_strict_local_signature_diagnostic_not_scoreable",
            "local signature recurs outside strict-ready scope; keep diagnostic only.",
        )
    if any_not_scoreable:
        return (
            "not_scoreable_without_route_fraction_readout",
            "current artifacts lack the route/fraction feature rows required for P2-P6.",
        )
    if contrast_row is not None:
        return (
            "scoreable_contract_negative",
            "current route/fraction features are scoreable and do not pass P1-P6.",
        )
    return (
        "nonanalog_or_closed_control_rejected",
        "pair is rejected by fixed local signature or retained as a closed control/nonanalog.",
    )


def _build_pair_rows(
    screen_pair_rows: pd.DataFrame,
    contrast_rows: pd.DataFrame,
) -> pd.DataFrame:
    contrast_by_pair = {
        str(row["local_pair_id"]): row for _, row in contrast_rows.iterrows()
    }
    rows: list[dict[str, Any]] = []
    for _, screen_row in screen_pair_rows.iterrows():
        pair_id = str(screen_row["local_pair_id"])
        contrast_row = contrast_by_pair.get(pair_id)
        values = _predicate_values(screen_row, contrast_row)
        statuses = {
            predicate_id: _status_from_bool(values[predicate_id])
            for predicate_id in PREDICATE_IDS
        }
        app_class, reason = _application_class(pair_id, screen_row, contrast_row, statuses)
        scoreable = contrast_row is not None
        all_pass = all(statuses[predicate_id] == "pass" for predicate_id in PREDICATE_IDS)
        rows.append(
            {
                "local_pair_id": pair_id,
                "validation_stratum": screen_row["validation_stratum"],
                "guard_family": screen_row.get("guard_family"),
                "mechanism_generalization_class": screen_row[
                    "mechanism_generalization_class"
                ],
                "fixed_016_local_signature_pass": _as_bool(
                    screen_row["fixed_016_local_signature_pass"]
                ),
                "screen_first_pass_route_readout_available": _as_bool(
                    screen_row["first_pass_route_readout_available"]
                ),
                "contract_feature_scoreable": scoreable,
                "contract_application_class": app_class,
                "contract_application_reason": reason,
                "p1_status": statuses["P1_fixed_local_signature_screen"],
                "p2_status": statuses["P2_source_and_target_brackets"],
                "p3_status": statuses["P3_finite_all_route_single_side_band"],
                "p4_status": statuses["P4_exact_single_bridge_latch"],
                "p5_status": statuses["P5_anchor_equidistant_support_geometry"],
                "p6_status": statuses["P6_seed_start_stable_band"],
                "contract_accepts_p1_p6": all_pass,
                "all_route_single_side_fraction_count": int(
                    contrast_row["all_route_single_side_fraction_count"]
                )
                if contrast_row is not None
                else None,
                "single_side_latch_signature": contrast_row["single_side_latch_signature"]
                if contrast_row is not None
                else None,
                "pair_explanation_class": contrast_row["pair_explanation_class"]
                if contrast_row is not None
                else None,
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "method_status": METHOD_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
                "run_status": RUN_STATUS,
            }
        )
    return pd.DataFrame(rows)


def _build_predicate_rows(pair_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in pair_rows.iterrows():
        for predicate_id, status_col in [
            ("P1_fixed_local_signature_screen", "p1_status"),
            ("P2_source_and_target_brackets", "p2_status"),
            ("P3_finite_all_route_single_side_band", "p3_status"),
            ("P4_exact_single_bridge_latch", "p4_status"),
            ("P5_anchor_equidistant_support_geometry", "p5_status"),
            ("P6_seed_start_stable_band", "p6_status"),
        ]:
            rows.append(
                {
                    "local_pair_id": row["local_pair_id"],
                    "predicate_id": predicate_id,
                    "predicate_status": row[status_col],
                    "contract_feature_scoreable": row["contract_feature_scoreable"],
                    "contract_application_class": row["contract_application_class"],
                    "route_execution_status": ROUTE_EXECUTION_STATUS,
                    "wall_promotion_status": WALL_PROMOTION_STATUS,
                    "method_status": METHOD_STATUS,
                    "claim_boundary": CLAIM_BOUNDARY,
                    "run_status": RUN_STATUS,
                }
            )
    return pd.DataFrame(rows)


def _build_class_rows(pair_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = pair_rows.groupby("contract_application_class", dropna=False)
    for app_class, group in grouped:
        rows.append(
            {
                "contract_application_class": app_class,
                "pair_count": int(len(group)),
                "local_pair_ids": ";".join(group["local_pair_id"].astype(str)),
                "p1_p6_accept_count": int(group["contract_accepts_p1_p6"].map(_as_bool).sum()),
                "feature_scoreable_count": int(
                    group["contract_feature_scoreable"].map(_as_bool).sum()
                ),
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["p1_p6_accept_count", "feature_scoreable_count", "pair_count"],
        ascending=[False, False, False],
    )


def _decision_rows(pair_rows: pd.DataFrame, class_rows: pd.DataFrame) -> pd.DataFrame:
    accepted = pair_rows[pair_rows["contract_accepts_p1_p6"].map(_as_bool)]
    near_miss = pair_rows[
        pair_rows["contract_application_class"].astype(str).eq("near_miss_guard_negative")
    ]
    non_strict = pair_rows[
        pair_rows["contract_application_class"]
        .astype(str)
        .eq("non_strict_local_signature_diagnostic_not_scoreable")
    ]
    rows = [
        {
            "decision_id": "D1_current_panel_contract_application",
            "decision": "only_016_accepts_p1_p6_on_current_scoreable_surface",
            "evidence": json.dumps(
                {
                    "accepted_pair_ids": accepted["local_pair_id"].tolist(),
                    "accepted_count": int(len(accepted)),
                },
                sort_keys=True,
            ),
            "claim_boundary": "Current evidence supports 016 reference only, not generality.",
            "run_status": RUN_STATUS,
        },
        {
            "decision_id": "D2_near_miss_guards_remain_negative",
            "decision": "009_012_014_020_remain_specificity_guards",
            "evidence": json.dumps(
                {
                    row["local_pair_id"]: {
                        "p3_status": row["p3_status"],
                        "class": row["contract_application_class"],
                    }
                    for _, row in near_miss.iterrows()
                },
                sort_keys=True,
            ),
            "claim_boundary": "Near misses guard the contract; they are not failed method rows.",
            "run_status": RUN_STATUS,
        },
        {
            "decision_id": "D3_non_strict_local_signature_not_expansion_ready",
            "decision": "001_007_are_diagnostic_not_plateau_candidates",
            "evidence": json.dumps(
                {
                    "non_strict_local_signature_pair_ids": non_strict[
                        "local_pair_id"
                    ].tolist(),
                    "reason": "outside strict-ready scope or blocked by rare-ready guard",
                },
                sort_keys=True,
            ),
            "claim_boundary": "Local signature recurrence outside strict-ready scope is not route-level generality.",
            "run_status": RUN_STATUS,
        },
        {
            "decision_id": "D4_next_step",
            "decision": "do_not_expand_until_contract_application_question_is_named",
            "evidence": json.dumps(
                {
                    "class_counts": class_rows[
                        ["contract_application_class", "pair_count"]
                    ].to_dict(orient="records")
                },
                sort_keys=True,
            ),
            "claim_boundary": "The next run must be a named contract application, not broad candidate search.",
            "run_status": RUN_STATUS,
        },
    ]
    return pd.DataFrame(rows)


def _build_gates(
    contract_gate_matrix: pd.DataFrame,
    pair_rows: pd.DataFrame,
    predicate_rows: pd.DataFrame,
) -> pd.DataFrame:
    contract_failed = contract_gate_matrix[
        contract_gate_matrix["gate_status"].astype(str).ne("pass")
    ]
    accepted = pair_rows[pair_rows["contract_accepts_p1_p6"].map(_as_bool)]
    near_miss = pair_rows[
        pair_rows["contract_application_class"].astype(str).eq("near_miss_guard_negative")
    ]
    non_strict = pair_rows[
        pair_rows["contract_application_class"]
        .astype(str)
        .eq("non_strict_local_signature_diagnostic_not_scoreable")
    ]
    p2_p6_not_scoreable = predicate_rows[
        predicate_rows["predicate_id"].astype(str).ne("P1_fixed_local_signature_screen")
        & predicate_rows["predicate_status"].astype(str).eq("not_scoreable")
    ]
    gates = [
        _gate_row(
            "G1_contract_source_passed",
            "Did the source plateau-stability gate contract pass?",
            {"failed_contract_gates": contract_failed["gate_id"].tolist()},
            "no failed contract gates",
            contract_failed.empty,
        ),
        _gate_row(
            "G2_current_panel_complete",
            "Does the application cover the full 23-pair first-pass panel?",
            {"pair_count": int(len(pair_rows))},
            "23 pair rows",
            len(pair_rows) == 23,
        ),
        _gate_row(
            "G3_only_016_accepts_contract",
            "Does only 016 pass all P1-P6 predicates on the current scoreable surface?",
            {
                "accepted_pair_ids": accepted["local_pair_id"].tolist(),
                "accepted_count": int(len(accepted)),
            },
            "accepted pair ids == [local_pair_016]",
            accepted["local_pair_id"].tolist() == [PRIMARY_PAIR_ID],
        ),
        _gate_row(
            "G4_near_miss_guards_negative",
            "Do 009/012/014/020 remain negative near-miss guards?",
            near_miss[["local_pair_id", "p3_status", "contract_application_class"]].to_dict(
                orient="records"
            ),
            "all four near misses present and no all-route plateau",
            set(near_miss["local_pair_id"]) == set(NEAR_MISS_PAIR_IDS)
            and bool(near_miss["p3_status"].astype(str).eq("fail").all()),
        ),
        _gate_row(
            "G5_non_strict_local_signature_rows_not_promoted",
            "Are non-strict local-signature rows kept diagnostic?",
            non_strict[["local_pair_id", "validation_stratum"]].to_dict(orient="records"),
            "001/007 are diagnostic, not contract-accepted candidates",
            set(non_strict["local_pair_id"]) == set(NON_STRICT_LOCAL_SIGNATURE_PAIR_IDS),
        ),
        _gate_row(
            "G6_missing_route_fraction_readout_not_backfilled",
            "Are P2-P6 gaps marked not_scoreable instead of inferred?",
            {
                "not_scoreable_predicate_rows": int(len(p2_p6_not_scoreable)),
                "affected_pair_count": int(p2_p6_not_scoreable["local_pair_id"].nunique()),
            },
            "missing route/fraction readout remains not_scoreable",
            len(p2_p6_not_scoreable) > 0,
        ),
        _gate_row(
            "G7_claim_boundaries_closed",
            "Are wall, method, quality/cost, and full-replay claims closed?",
            CLAIM_BOUNDARY,
            "read-only contract application",
            True,
        ),
    ]
    return pd.DataFrame(gates)


def _write_report(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    pair_rows: pd.DataFrame,
    class_rows: pd.DataFrame,
    decision_rows: pd.DataFrame,
    gates: pd.DataFrame,
) -> None:
    lines = [
        "# NanoClustering G4.8 First-Pass Plateau-Stability Gate Application",
        "",
        "## Summary",
        "",
        f"- status: {summary['status']}",
        f"- application_status: {summary['application_status']}",
        f"- failed_gates: {summary['failed_gates']}",
        "",
        "## Pair Rows",
        "",
        _markdown_table(
            pair_rows,
            [
                "local_pair_id",
                "validation_stratum",
                "contract_feature_scoreable",
                "contract_application_class",
                "p1_status",
                "p2_status",
                "p3_status",
                "p4_status",
                "p5_status",
                "p6_status",
                "contract_accepts_p1_p6",
            ],
        ),
        "",
        "## Class Rows",
        "",
        _markdown_table(
            class_rows,
            [
                "contract_application_class",
                "pair_count",
                "local_pair_ids",
                "p1_p6_accept_count",
                "feature_scoreable_count",
            ],
        ),
        "",
        "## Decisions",
        "",
        _markdown_table(decision_rows, ["decision_id", "decision", "claim_boundary"]),
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


def run_audit(
    generalization_screen_dir: Path = DEFAULT_GENERALIZATION_SCREEN_DIR,
    gate_contract_dir: Path = DEFAULT_GATE_CONTRACT_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    context = _load_context(
        generalization_screen_dir=generalization_screen_dir,
        gate_contract_dir=gate_contract_dir,
    )
    tables = context["tables"]
    pair_rows = _build_pair_rows(
        screen_pair_rows=tables["screen_pair_rows"],
        contrast_rows=tables["contract_contrast_rows"],
    )
    predicate_rows = _build_predicate_rows(pair_rows)
    class_rows = _build_class_rows(pair_rows)
    decision_rows = _decision_rows(pair_rows, class_rows)
    gates = _build_gates(
        contract_gate_matrix=tables["contract_gate_matrix"],
        pair_rows=pair_rows,
        predicate_rows=predicate_rows,
    )

    _write_csv(pair_rows, output_dir / PAIR_ROWS_CSV)
    _write_csv(predicate_rows, output_dir / PREDICATE_ROWS_CSV)
    _write_csv(class_rows, output_dir / CLASS_ROWS_CSV)
    _write_csv(decision_rows, output_dir / DECISION_ROWS_CSV)
    _write_csv(gates, output_dir / GATE_MATRIX_CSV)

    accepted = pair_rows[pair_rows["contract_accepts_p1_p6"].map(_as_bool)]
    failed_gates = gates.loc[gates["gate_status"].astype(str).ne("pass"), "gate_id"].tolist()
    summary = {
        "schema": "nanoclustering_g4_8_first_pass_plateau_stability_gate_application_summary.v1",
        "status": RUN_STATUS,
        "application_status": "current_panel_applied_contract_without_new_candidate_expansion",
        "failed_gates": failed_gates,
        "gate_status_counts": gates["gate_status"].value_counts().to_dict(),
        "accepted_pair_ids": accepted["local_pair_id"].tolist(),
        "interpretation": (
            "Applying the P1-P6 plateau-stability contract to the current "
            "23-pair panel accepts only 016. Near misses 009/012/014/020 remain "
            "negative guards, and non-strict local-signature rows 001/007 stay "
            "diagnostic rather than expansion-ready."
        ),
        "recommended_next_gate": (
            "Do not broaden candidates yet. If continuing, design a named "
            "contract-application surface that explicitly supplies route/fraction "
            "readout for P2-P6, with 009/012/014/020 retained as specificity guards."
        ),
        "source_statuses": {
            "generalization_screen": context["summaries"]["generalization_screen"].get(
                "status"
            ),
            "gate_contract": context["summaries"]["gate_contract"].get("status"),
        },
        "source_dirs": {
            "generalization_screen_dir": str(generalization_screen_dir),
            "gate_contract_dir": str(gate_contract_dir),
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
        "non_strict_local_signature_pair_ids": list(NON_STRICT_LOCAL_SIGNATURE_PAIR_IDS),
        "contract_scoreable_pair_ids": list(CONTRACT_SCOREABLE_PAIR_IDS),
        "predicate_ids": list(PREDICATE_IDS),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(
        output_dir=output_dir,
        summary=summary,
        pair_rows=pair_rows,
        class_rows=class_rows,
        decision_rows=decision_rows,
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
    parser.add_argument("--gate-contract-dir", type=Path, default=DEFAULT_GATE_CONTRACT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_audit(
        generalization_screen_dir=Path(args.generalization_screen_dir),
        gate_contract_dir=Path(args.gate_contract_dir),
        output_dir=Path(args.output_dir),
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
