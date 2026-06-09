#!/usr/bin/env python3
"""Design the first-pass surface-rule gap-fill contract.

This contract follows the panel-readiness audit. It does not broaden the
current panel into a generality claim. Instead, it opens only the two
diagnostic-but-not-scoreable rows (``001`` and ``007``) for a narrow
route/fraction readout, while locking the six scoreable core rows as the fixed
reference/guard set.

This is a contract design only. It does not run Leiden, execute routes, promote
wall/pathway labels, evaluate quality/cost value, replay full NanoClustering,
or claim method success.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from audit_leiden_basin_nanoclustering_g4_8_first_pass_surface_rule_panel_readiness import (
    DEFAULT_OUTPUT_DIR as DEFAULT_PANEL_READINESS_DIR,
    GATE_MATRIX_CSV as PANEL_READINESS_GATE_MATRIX_CSV,
    PAIR_SURFACE_ROWS_CSV as PANEL_READINESS_PAIR_SURFACE_ROWS_CSV,
    SUMMARY_JSON as PANEL_READINESS_SUMMARY_JSON,
)
from run_leiden_basin_nanoclustering_role_local_route_pilot import (
    BASE_RESULT_DIR,
    _json_safe,
    _read_csv,
    _write_csv,
)


DEFAULT_LOCAL_VALIDATION_DIR = (
    BASE_RESULT_DIR / "leiden_basin_nanoclustering_g4_8_local_validation_readout_gamma1e5_20260604"
)
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_surface_rule_gap_fill_contract_gamma1e5_20260609"
)

LOCAL_VALIDATION_START_ROWS_CSV = (
    "nanoclustering_g4_8_local_validation_readout_start_condition_rows.csv"
)

PAIR_ROLE_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_surface_rule_gap_fill_contract_pair_role_rows.csv"
)
GAP_FILL_CANDIDATE_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_surface_rule_gap_fill_contract_candidate_rows.csv"
)
ROUTE_PLAN_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_surface_rule_gap_fill_contract_route_plan_rows.csv"
)
ACCEPTANCE_RULE_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_surface_rule_gap_fill_contract_acceptance_rule_rows.csv"
)
DECISION_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_surface_rule_gap_fill_contract_decision_rows.csv"
)
GATE_MATRIX_CSV = (
    "nanoclustering_g4_8_first_pass_surface_rule_gap_fill_contract_gate_matrix.csv"
)
SUMMARY_JSON = "nanoclustering_g4_8_first_pass_surface_rule_gap_fill_contract_summary.json"
CONFIG_JSON = "nanoclustering_g4_8_first_pass_surface_rule_gap_fill_contract_config.json"
REPORT_MD = "nanoclustering_g4_8_first_pass_surface_rule_gap_fill_contract_report.md"

RUN_STATUS = "designed_nanoclustering_g4_8_first_pass_surface_rule_gap_fill_contract"
ROUTE_EXECUTION_STATUS = "design_only_not_executed"
WALL_PROMOTION_STATUS = "not_promoted_gap_fill_contract_only"
METHOD_STATUS = "surface_rule_gap_fill_contract_not_method"
CLAIM_BOUNDARY = (
    "NanoClustering G4.8 first-pass surface-rule gap-fill contract design only; "
    "reads the panel-readiness audit and local validation start rows. It opens "
    "only diagnostic-not-scoreable local-signature rows 001 and 007 for a narrow "
    "route/fraction readout, locks 016/014/009/012/020/005 as fixed reference "
    "and guard rows, excludes screened gaps, and does not promote wall, pathway, "
    "panel-generality, quality/cost, full-replay, or method claims."
)

GAP_FILL_CANDIDATE_IDS = ("local_pair_001", "local_pair_007")
REFERENCE_PAIR_ID = "local_pair_016"
CROSS_SURFACE_GUARD_PAIR_ID = "local_pair_014"
STRICT_NEGATIVE_GUARD_IDS = ("local_pair_009", "local_pair_012", "local_pair_020")
BOUNDARY_GUARD_PAIR_ID = "local_pair_005"
FIXED_CORE_IDS = (
    BOUNDARY_GUARD_PAIR_ID,
    *STRICT_NEGATIVE_GUARD_IDS,
    CROSS_SURFACE_GUARD_PAIR_ID,
    REFERENCE_PAIR_ID,
)
EXPECTED_PANEL_COUNT = 23
EXPECTED_PANEL_READINESS_GATES = 9
ROUTE_FAMILY = "first_pass_surface_rule_gap_fill_bridge_fraction_scan"
ALLOWED_START_ROLES = ("R_candidate", "R_weak")
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

ACCEPTANCE_RULES: tuple[dict[str, str], ...] = (
    {
        "rule_id": "GF1",
        "rule_group": "scope",
        "rule_question": "Are only diagnostic-not-scoreable rows opened?",
        "acceptance_requirement": "execution_pair_ids == {local_pair_001, local_pair_007}",
        "claim_effect": "prevents screened gaps from becoming a route sweep",
    },
    {
        "rule_id": "GF2",
        "rule_group": "start_scope",
        "rule_question": "Are starts restricted to locally expected pass starts?",
        "acceptance_requirement": (
            "start_condition_expected_validation_pass == true and "
            "start_condition_macro_role in {R_candidate, R_weak}"
        ),
        "claim_effect": "prevents incompatible start rows from being interpreted as failures",
    },
    {
        "rule_id": "GF3",
        "rule_group": "positive_readout",
        "rule_question": "What would count as a positive diagnostic recurrence?",
        "acceptance_requirement": (
            "candidate route produces stable finite single-side transition-band morphology "
            "under the fixed readout vocabulary and does not violate the fixed guard set"
        ),
        "claim_effect": "opens diagnostic recurrence wording only, not panel generality",
    },
    {
        "rule_id": "GF4",
        "rule_group": "negative_readout",
        "rule_question": "What would count as an informative negative?",
        "acceptance_requirement": (
            "route/fraction readout is present but the route is abrupt, fragmented, "
            "point-only, boundary-like, or lacks object-wall evidence"
        ),
        "claim_effect": "may promote the row to a scoreable negative guard only",
    },
    {
        "rule_id": "GF5",
        "rule_group": "gap_readout",
        "rule_question": "What remains a gap after execution?",
        "acceptance_requirement": (
            "missing route/fraction readout, missing required state columns, or "
            "uninterpretable route state keeps the row not-scoreable"
        ),
        "claim_effect": "prevents forced classification",
    },
    {
        "rule_id": "GF6",
        "rule_group": "fixed_guards",
        "rule_question": "Are 016 and the five guards fixed rather than re-selected?",
        "acceptance_requirement": (
            "016 remains the single reference; 014/009/012/020/005 remain fixed "
            "specificity or boundary guards"
        ),
        "claim_effect": "keeps current positives and negatives from being retuned",
    },
    {
        "rule_id": "GF7",
        "rule_group": "claim_boundary",
        "rule_question": "Are wall, pathway, method, quality/cost, replay, and panel-generality claims closed?",
        "acceptance_requirement": (
            "contract and execution cannot promote these claims without separate "
            "predeclared gates"
        ),
        "claim_effect": "claim promotion blocked",
    },
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _json_dump(value: Any) -> str:
    return json.dumps(_json_safe(value), sort_keys=True, ensure_ascii=True)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _count_dict(series: pd.Series) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in series.astype(str).value_counts(dropna=False).sort_index().items()
    }


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
        "observed": _json_dump(observed),
        "minimum_or_rule": minimum_or_rule,
        "gate_status": "pass" if bool(passed) else "fail",
    }


def _markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 80) -> str:
    cols = [column for column in columns if column in frame.columns]
    if not cols:
        return "_No matching columns._"
    visible = frame[cols].head(max_rows)
    if visible.empty:
        return "_No rows._"

    def cell(value: Any) -> str:
        if isinstance(value, (dict, list, tuple, set)):
            return _json_dump(value).replace("|", "\\|")
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


def _pair_contract_role(row: pd.Series) -> tuple[str, str, str]:
    pair_id = str(row["local_pair_id"])
    if pair_id in GAP_FILL_CANDIDATE_IDS:
        return (
            "gap_fill_candidate",
            "execute_narrow_route_fraction_readout",
            "diagnostic-not-scoreable non-strict local-signature row",
        )
    if pair_id == REFERENCE_PAIR_ID:
        return (
            "fixed_reference",
            "lock_as_reference_do_not_reselect",
            "single diagnostic transition-band reference",
        )
    if pair_id == CROSS_SURFACE_GUARD_PAIR_ID:
        return (
            "fixed_cross_surface_guard",
            "lock_as_guard_do_not_promote",
            "different endpoint-object surface with current morphology mismatch",
        )
    if pair_id in STRICT_NEGATIVE_GUARD_IDS:
        return (
            "fixed_strict_negative_guard",
            "lock_as_guard_do_not_promote",
            "strict analog negative route morphology",
        )
    if pair_id == BOUNDARY_GUARD_PAIR_ID:
        return (
            "fixed_boundary_guard",
            "lock_as_boundary_guard_do_not_promote",
            "source-target collapse boundary guard",
        )
    return (
        "deferred_screened_gap",
        "do_not_execute_in_this_contract",
        "screened gap or nonanalog row lacks current route/fraction surface",
    )


def _pair_role_rows(panel_rows: pd.DataFrame) -> pd.DataFrame:
    rows = panel_rows.copy()
    roles = rows.apply(_pair_contract_role, axis=1).apply(pd.Series)
    roles.columns = ["next_contract_role", "next_contract_action", "next_contract_reason"]
    rows = pd.concat([rows, roles], axis=1)
    rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["method_status"] = METHOD_STATUS
    rows["run_status"] = RUN_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows[
        [
            "local_pair_id",
            "scoreability_status",
            "surface_rule_class",
            "generalization_status",
            "promotion_status",
            "readiness_gap",
            "readiness_decision",
            "next_contract_role",
            "next_contract_action",
            "next_contract_reason",
            "route_execution_status",
            "wall_promotion_status",
            "method_status",
            "run_status",
            "claim_boundary",
        ]
    ].sort_values("local_pair_id", kind="mergesort")


def _candidate_rows(pair_role_rows: pd.DataFrame, start_rows: pd.DataFrame) -> pd.DataFrame:
    candidate_rows = pair_role_rows[
        pair_role_rows["next_contract_role"].eq("gap_fill_candidate")
    ].copy()
    start_scoped = start_rows[
        start_rows["local_pair_id"].astype(str).isin(GAP_FILL_CANDIDATE_IDS)
    ].copy()
    start_scoped["start_allowed_for_gap_fill"] = (
        start_scoped["start_condition_expected_validation_pass"].map(_as_bool)
        & start_scoped["start_condition_macro_role"].astype(str).isin(ALLOWED_START_ROLES)
    )
    grouped = (
        start_scoped.groupby("local_pair_id", dropna=False)
        .agg(
            available_start_count=("start_condition", "size"),
            allowed_start_count=("start_allowed_for_gap_fill", "sum"),
            allowed_start_conditions=(
                "start_condition",
                lambda values: ";".join(
                    str(value)
                    for value, allowed in zip(
                        values,
                        start_scoped.loc[values.index, "start_allowed_for_gap_fill"],
                    )
                    if bool(allowed)
                ),
            ),
            allowed_start_roles=(
                "start_condition_macro_role",
                lambda values: ";".join(
                    str(value)
                    for value, allowed in zip(
                        values,
                        start_scoped.loc[values.index, "start_allowed_for_gap_fill"],
                    )
                    if bool(allowed)
                ),
            ),
        )
        .reset_index()
    )
    rows = candidate_rows.merge(grouped, on="local_pair_id", how="left", validate="one_to_one")
    rows["candidate_gate_status"] = rows["allowed_start_count"].map(
        lambda value: "ready" if int(value) > 0 else "blocked_no_allowed_start"
    )
    rows["execution_priority"] = rows["local_pair_id"].map(
        {"local_pair_007": 1, "local_pair_001": 2}
    )
    return rows.sort_values(["execution_priority", "local_pair_id"], kind="mergesort")


def _route_plan_rows(candidate_rows: pd.DataFrame, start_rows: pd.DataFrame) -> pd.DataFrame:
    allowed = start_rows[
        start_rows["local_pair_id"].astype(str).isin(GAP_FILL_CANDIDATE_IDS)
    ].copy()
    allowed = allowed[
        allowed["start_condition_expected_validation_pass"].map(_as_bool)
        & allowed["start_condition_macro_role"].astype(str).isin(ALLOWED_START_ROLES)
    ].copy()
    candidate_lookup = candidate_rows.set_index("local_pair_id").to_dict("index")
    route_rows: list[dict[str, Any]] = []
    for start in allowed.sort_values(["local_pair_id", "start_condition"], kind="mergesort").itertuples(
        index=False
    ):
        pair_id = str(start.local_pair_id)
        for fraction_index, bridge_fraction in enumerate(FINE_BRIDGE_FRACTIONS, start=1):
            route_rows.append(
                {
                    "route_contract_id": (
                        f"surface_gap_fill_{pair_id.replace('local_pair_', '')}_"
                        f"{str(start.start_condition)}_bf{fraction_index:02d}"
                    ),
                    "local_pair_id": pair_id,
                    "next_contract_role": str(candidate_lookup[pair_id]["next_contract_role"]),
                    "route_family": ROUTE_FAMILY,
                    "start_condition": str(start.start_condition),
                    "start_condition_macro_role": str(start.start_condition_macro_role),
                    "start_condition_expected_validation_pass": _as_bool(
                        start.start_condition_expected_validation_pass
                    ),
                    "bridge_fraction": float(bridge_fraction),
                    "fraction_order": fraction_index,
                    "positive_recurrence_claim_allowed_after_contract": False,
                    "panel_generality_claim_allowed_after_contract": False,
                    "wall_claim_allowed_after_contract": False,
                    "pathway_claim_allowed_after_contract": False,
                    "method_claim_allowed_after_contract": False,
                    "quality_cost_claim_allowed_after_contract": False,
                    "route_execution_status": ROUTE_EXECUTION_STATUS,
                    "wall_promotion_status": WALL_PROMOTION_STATUS,
                    "method_status": METHOD_STATUS,
                    "run_status": RUN_STATUS,
                    "claim_boundary": CLAIM_BOUNDARY,
                }
            )
    return pd.DataFrame(route_rows)


def _acceptance_rule_rows() -> pd.DataFrame:
    rows = pd.DataFrame(ACCEPTANCE_RULES)
    rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["method_status"] = METHOD_STATUS
    rows["run_status"] = RUN_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


def _decision_rows() -> pd.DataFrame:
    rows = [
        {
            "decision_id": "D1",
            "decision": "gap_fill_before_generality",
            "rationale": (
                "The panel-readiness result establishes accounting readiness, not "
                "panel-level generality. The next step should reduce the most local "
                "scoreability gap before broadening."
            ),
        },
        {
            "decision_id": "D2",
            "decision": "open_only_001_007",
            "rationale": (
                "Only 001 and 007 are diagnostic-not-scoreable non-strict "
                "local-signature rows; all screened rows remain excluded."
            ),
        },
        {
            "decision_id": "D3",
            "decision": "lock_six_row_core",
            "rationale": (
                "016 is the fixed reference; 014, 009, 012, 020, and 005 are fixed "
                "specificity or boundary guards, not re-selected candidates."
            ),
        },
        {
            "decision_id": "D4",
            "decision": "route_plan_is_readout_only",
            "rationale": (
                "The planned route rows can only reclassify 001/007 as diagnostic "
                "recurrence, scoreable negative, or still-not-scoreable gap."
            ),
        },
        {
            "decision_id": "D5",
            "decision": "no_claim_promotion",
            "rationale": (
                "Even a positive 001/007 readout would not by itself establish wall, "
                "pathway, method, quality/cost, full-replay, or panel-generality claims."
            ),
        },
    ]
    frame = pd.DataFrame(rows)
    frame["run_status"] = RUN_STATUS
    frame["claim_boundary"] = CLAIM_BOUNDARY
    return frame


def _gate_matrix(
    *,
    panel_summary: dict[str, Any],
    panel_gates: pd.DataFrame,
    pair_role_rows: pd.DataFrame,
    candidate_rows: pd.DataFrame,
    route_plan_rows: pd.DataFrame,
    acceptance_rule_rows: pd.DataFrame,
) -> pd.DataFrame:
    role_counts = _count_dict(pair_role_rows["next_contract_role"])
    candidate_ids = list(candidate_rows["local_pair_id"].astype(str))
    fixed_ids = list(
        pair_role_rows[
            pair_role_rows["next_contract_role"].astype(str).str.startswith("fixed_")
        ]["local_pair_id"].astype(str)
    )
    route_pair_ids = sorted(route_plan_rows["local_pair_id"].astype(str).unique())
    route_start_counts = (
        route_plan_rows.groupby("local_pair_id")["start_condition"].nunique().astype(int).to_dict()
        if not route_plan_rows.empty
        else {}
    )
    rows = [
        _gate_row(
            "G1_panel_readiness_ready",
            "Did the upstream panel-readiness audit pass?",
            {
                "panel_failed_gates": panel_summary.get("failed_gates"),
                "panel_gate_status_counts": _count_dict(panel_gates["gate_status"]),
                "pair_row_count": panel_summary.get("pair_row_count"),
            },
            "panel has 23 rows, 9 passing gates, and no failed readiness gates",
            bool(panel_summary.get("failed_gates") == [])
            and int(panel_summary.get("pair_row_count", 0)) == EXPECTED_PANEL_COUNT
            and int((panel_gates["gate_status"].astype(str) == "pass").sum())
            == EXPECTED_PANEL_READINESS_GATES,
        ),
        _gate_row(
            "G2_only_diagnostic_gaps_opened",
            "Are only 001 and 007 opened as gap-fill candidates?",
            {"candidate_ids": candidate_ids, "role_counts": role_counts},
            "candidate ids equal 001/007 and no screened gap is opened",
            set(candidate_ids) == set(GAP_FILL_CANDIDATE_IDS)
            and role_counts.get("gap_fill_candidate") == len(GAP_FILL_CANDIDATE_IDS)
            and role_counts.get("deferred_screened_gap") == 15,
        ),
        _gate_row(
            "G3_six_row_core_locked",
            "Is the six-row scoreable core locked as reference/guard evidence?",
            {"fixed_ids": fixed_ids, "expected_fixed_ids": list(FIXED_CORE_IDS)},
            "016/014/009/012/020/005 are fixed and not re-selected",
            set(fixed_ids) == set(FIXED_CORE_IDS),
        ),
        _gate_row(
            "G4_route_plan_scoped_to_candidates",
            "Is the route plan scoped only to 001/007 allowed starts and fixed fractions?",
            {
                "route_pair_ids": route_pair_ids,
                "route_row_count": int(len(route_plan_rows)),
                "route_start_counts": route_start_counts,
                "fraction_count": len(FINE_BRIDGE_FRACTIONS),
            },
            "route rows are only candidate allowed starts times the fixed fraction schedule",
            set(route_pair_ids) == set(GAP_FILL_CANDIDATE_IDS)
            and int(len(route_plan_rows))
            == int(candidate_rows["allowed_start_count"].sum()) * len(FINE_BRIDGE_FRACTIONS)
            and bool(route_plan_rows["bridge_fraction"].isin(FINE_BRIDGE_FRACTIONS).all()),
        ),
        _gate_row(
            "G5_acceptance_rules_written",
            "Are positive, negative, residual-gap, guard, and claim-boundary rules explicit?",
            {
                "rule_count": int(len(acceptance_rule_rows)),
                "rule_groups": sorted(acceptance_rule_rows["rule_group"].astype(str).unique()),
            },
            "at least seven rule rows cover scope, starts, positive, negative, gap, guards, and claims",
            len(acceptance_rule_rows) >= 7
            and {
                "scope",
                "start_scope",
                "positive_readout",
                "negative_readout",
                "gap_readout",
                "fixed_guards",
                "claim_boundary",
            }.issubset(set(acceptance_rule_rows["rule_group"].astype(str))),
        ),
        _gate_row(
            "G6_no_screened_gap_execution",
            "Are screened gaps excluded from this execution contract?",
            {
                "deferred_screened_gap_count": role_counts.get("deferred_screened_gap"),
                "route_pair_ids": route_pair_ids,
            },
            "the 15 screened rows remain deferred and have zero route rows",
            role_counts.get("deferred_screened_gap") == 15
            and not bool(
                set(
                    pair_role_rows[
                        pair_role_rows["next_contract_role"].eq("deferred_screened_gap")
                    ]["local_pair_id"].astype(str)
                )
                & set(route_pair_ids)
            ),
        ),
        _gate_row(
            "G7_no_claim_promotion",
            "Are generality, wall, pathway, method, quality, and replay claims closed?",
            {
                "panel_generality_flags": _count_dict(
                    route_plan_rows["panel_generality_claim_allowed_after_contract"]
                ),
                "wall_flags": _count_dict(route_plan_rows["wall_claim_allowed_after_contract"]),
                "method_flags": _count_dict(
                    route_plan_rows["method_claim_allowed_after_contract"]
                ),
            },
            "all route-plan promotion flags remain false",
            bool(route_plan_rows["panel_generality_claim_allowed_after_contract"].eq(False).all())
            and bool(route_plan_rows["wall_claim_allowed_after_contract"].eq(False).all())
            and bool(route_plan_rows["pathway_claim_allowed_after_contract"].eq(False).all())
            and bool(route_plan_rows["method_claim_allowed_after_contract"].eq(False).all())
            and bool(route_plan_rows["quality_cost_claim_allowed_after_contract"].eq(False).all()),
        ),
    ]
    return pd.DataFrame(rows)


def _summary(
    *,
    output_dir: Path,
    panel_readiness_dir: Path,
    local_validation_dir: Path,
    pair_role_rows: pd.DataFrame,
    candidate_rows: pd.DataFrame,
    route_plan_rows: pd.DataFrame,
    acceptance_rule_rows: pd.DataFrame,
    decision_rows: pd.DataFrame,
    gate_matrix: pd.DataFrame,
) -> dict[str, Any]:
    failed_gates = list(
        gate_matrix.loc[gate_matrix["gate_status"].ne("pass"), "gate_id"].astype(str)
    )
    return {
        "schema": "nanoclustering_g4_8_first_pass_surface_rule_gap_fill_contract_summary.v1",
        "status": RUN_STATUS,
        "output_dir": str(output_dir),
        "panel_readiness_dir": str(panel_readiness_dir),
        "local_validation_dir": str(local_validation_dir),
        "claim_boundary": CLAIM_BOUNDARY,
        "pair_role_row_count": int(len(pair_role_rows)),
        "candidate_pair_ids": list(candidate_rows["local_pair_id"].astype(str)),
        "fixed_core_pair_ids": list(FIXED_CORE_IDS),
        "deferred_screened_gap_count": int(
            pair_role_rows["next_contract_role"].eq("deferred_screened_gap").sum()
        ),
        "route_plan_row_count": int(len(route_plan_rows)),
        "allowed_start_count_by_candidate": {
            str(row.local_pair_id): int(row.allowed_start_count)
            for row in candidate_rows.itertuples(index=False)
        },
        "fraction_count": len(FINE_BRIDGE_FRACTIONS),
        "acceptance_rule_count": int(len(acceptance_rule_rows)),
        "decision_row_count": int(len(decision_rows)),
        "gate_status_counts": _count_dict(gate_matrix["gate_status"]),
        "failed_gates": failed_gates,
        "panel_generality_claim_allowed_after_contract": False,
        "wall_claim_allowed_after_contract": False,
        "pathway_claim_allowed_after_contract": False,
        "method_claim_allowed_after_contract": False,
        "quality_cost_claim_allowed_after_contract": False,
        "route_execution_opened": False,
        "interpretation": (
            "The next gate should resolve the closest scoreability gap, not broaden "
            "the panel. Only 001/007 are opened for a narrow readout; 016 remains "
            "the reference and 014/009/012/020/005 remain fixed guards."
        ),
        "recommended_next_gate": (
            "If executed, run exactly the 54 route-plan rows for 001/007 and audit "
            "them into diagnostic recurrence, scoreable negative, or residual gap. "
            "Do not execute the 15 screened gaps or promote panel generality."
        ),
    }


def _report(
    *,
    summary: dict[str, Any],
    candidate_rows: pd.DataFrame,
    pair_role_rows: pd.DataFrame,
    route_plan_rows: pd.DataFrame,
    acceptance_rule_rows: pd.DataFrame,
    decision_rows: pd.DataFrame,
    gate_matrix: pd.DataFrame,
) -> str:
    lines = [
        "# NanoClustering G4.8 First-Pass Surface Rule Gap-Fill Contract",
        "",
        f"- status: `{summary['status']}`",
        f"- candidate_pair_ids: {summary['candidate_pair_ids']}",
        f"- fixed_core_pair_ids: {summary['fixed_core_pair_ids']}",
        f"- deferred_screened_gap_count: {summary['deferred_screened_gap_count']}",
        f"- route_plan_row_count: {summary['route_plan_row_count']}",
        f"- gate_status_counts: {summary['gate_status_counts']}",
        f"- failed_gates: {summary['failed_gates']}",
        f"- interpretation: {summary['interpretation']}",
        f"- recommended_next_gate: {summary['recommended_next_gate']}",
        f"- claim_boundary: {summary['claim_boundary']}",
        "",
        "## Candidate Rows",
        "",
        _markdown_table(
            candidate_rows,
            [
                "local_pair_id",
                "surface_rule_class",
                "readiness_gap",
                "allowed_start_count",
                "allowed_start_conditions",
                "candidate_gate_status",
                "execution_priority",
            ],
        ),
        "",
        "## Pair Role Rows",
        "",
        _markdown_table(
            pair_role_rows,
            [
                "local_pair_id",
                "scoreability_status",
                "surface_rule_class",
                "next_contract_role",
                "next_contract_action",
                "next_contract_reason",
            ],
        ),
        "",
        "## Route Plan Rows",
        "",
        _markdown_table(
            route_plan_rows,
            [
                "route_contract_id",
                "local_pair_id",
                "start_condition",
                "start_condition_macro_role",
                "bridge_fraction",
            ],
        ),
        "",
        "## Acceptance Rules",
        "",
        _markdown_table(
            acceptance_rule_rows,
            [
                "rule_id",
                "rule_group",
                "rule_question",
                "acceptance_requirement",
                "claim_effect",
            ],
        ),
        "",
        "## Decisions",
        "",
        _markdown_table(decision_rows, ["decision_id", "decision", "rationale"]),
        "",
        "## Gate Matrix",
        "",
        _markdown_table(
            gate_matrix,
            ["gate_id", "gate_status", "observed", "minimum_or_rule", "question"],
        ),
        "",
        "## Boundary",
        "",
        "This is a design contract. It must not be interpreted as a route execution,",
        "a wall/pathway result, or a panel-generality result.",
        "",
    ]
    return "\n".join(lines)


def run(
    *,
    panel_readiness_dir: Path,
    local_validation_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    panel_summary = _read_json(panel_readiness_dir / PANEL_READINESS_SUMMARY_JSON)
    panel_gates = _read_csv(panel_readiness_dir / PANEL_READINESS_GATE_MATRIX_CSV)
    panel_rows = _read_csv(panel_readiness_dir / PANEL_READINESS_PAIR_SURFACE_ROWS_CSV)
    start_rows = _read_csv(local_validation_dir / LOCAL_VALIDATION_START_ROWS_CSV)

    pair_role_rows = _pair_role_rows(panel_rows)
    candidate_rows = _candidate_rows(pair_role_rows, start_rows)
    route_plan_rows = _route_plan_rows(candidate_rows, start_rows)
    acceptance_rule_rows = _acceptance_rule_rows()
    decision_rows = _decision_rows()
    gate_matrix = _gate_matrix(
        panel_summary=panel_summary,
        panel_gates=panel_gates,
        pair_role_rows=pair_role_rows,
        candidate_rows=candidate_rows,
        route_plan_rows=route_plan_rows,
        acceptance_rule_rows=acceptance_rule_rows,
    )
    summary = _summary(
        output_dir=output_dir,
        panel_readiness_dir=panel_readiness_dir,
        local_validation_dir=local_validation_dir,
        pair_role_rows=pair_role_rows,
        candidate_rows=candidate_rows,
        route_plan_rows=route_plan_rows,
        acceptance_rule_rows=acceptance_rule_rows,
        decision_rows=decision_rows,
        gate_matrix=gate_matrix,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(pair_role_rows, output_dir / PAIR_ROLE_ROWS_CSV)
    _write_csv(candidate_rows, output_dir / GAP_FILL_CANDIDATE_ROWS_CSV)
    _write_csv(route_plan_rows, output_dir / ROUTE_PLAN_ROWS_CSV)
    _write_csv(acceptance_rule_rows, output_dir / ACCEPTANCE_RULE_ROWS_CSV)
    _write_csv(decision_rows, output_dir / DECISION_ROWS_CSV)
    _write_csv(gate_matrix, output_dir / GATE_MATRIX_CSV)
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True), encoding="utf-8"
    )
    config = {
        "schema": "nanoclustering_g4_8_first_pass_surface_rule_gap_fill_contract_config.v1",
        "panel_readiness_dir": str(panel_readiness_dir),
        "local_validation_dir": str(local_validation_dir),
        "output_dir": str(output_dir),
        "gap_fill_candidate_ids": list(GAP_FILL_CANDIDATE_IDS),
        "fixed_core_ids": list(FIXED_CORE_IDS),
        "allowed_start_roles": list(ALLOWED_START_ROLES),
        "fine_bridge_fractions": list(FINE_BRIDGE_FRACTIONS),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / REPORT_MD).write_text(
        _report(
            summary=summary,
            candidate_rows=candidate_rows,
            pair_role_rows=pair_role_rows,
            route_plan_rows=route_plan_rows,
            acceptance_rule_rows=acceptance_rule_rows,
            decision_rows=decision_rows,
            gate_matrix=gate_matrix,
        ),
        encoding="utf-8",
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panel-readiness-dir",
        type=Path,
        default=DEFAULT_PANEL_READINESS_DIR,
        help="Input panel-readiness audit directory.",
    )
    parser.add_argument(
        "--local-validation-dir",
        type=Path,
        default=DEFAULT_LOCAL_VALIDATION_DIR,
        help="Input local validation readout directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output contract directory.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = run(
        panel_readiness_dir=args.panel_readiness_dir,
        local_validation_dir=args.local_validation_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
