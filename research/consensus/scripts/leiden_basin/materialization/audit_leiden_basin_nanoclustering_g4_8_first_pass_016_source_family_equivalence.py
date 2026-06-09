#!/usr/bin/env python3
"""Audit source-family equivalence for local_pair_016 reverse final states.

This read-only audit follows the reverse non-return stratification. It compares
candidate source-equivalence rules for the 24 reverse final states and decides
whether strict same-seed source-anchor matching is too narrow for interpreting
the ``016`` reverse trace.
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

DEFAULT_NONRETURN_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_016_reverse_nonreturn_stratification_audit_gamma1e5_20260605"
)
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_016_source_family_equivalence_audit_gamma1e5_20260605"
)

ROUTE_RULE_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_016_source_family_equivalence_route_rule_rows.csv"
)
RULE_ROWS_CSV = "nanoclustering_g4_8_first_pass_016_source_family_equivalence_rule_rows.csv"
DECISION_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_016_source_family_equivalence_decision_rows.csv"
)
GATE_MATRIX_CSV = (
    "nanoclustering_g4_8_first_pass_016_source_family_equivalence_gate_matrix.csv"
)
SUMMARY_JSON = "nanoclustering_g4_8_first_pass_016_source_family_equivalence_summary.json"
CONFIG_JSON = "nanoclustering_g4_8_first_pass_016_source_family_equivalence_config.json"
REPORT_MD = "nanoclustering_g4_8_first_pass_016_source_family_equivalence_report.md"

RUN_STATUS = "audited_nanoclustering_g4_8_first_pass_016_source_family_equivalence"
ROUTE_EXECUTION_STATUS = "not_executed_read_only_016_source_family_equivalence"
WALL_PROMOTION_STATUS = "not_promoted_source_family_equivalence_only"
METHOD_STATUS = "source_family_equivalence_not_method"
CLAIM_BOUNDARY = (
    "NanoClustering G4.8 first-pass local_pair_016 source-family equivalence "
    "audit only; reads the reverse non-return stratification artifact to compare "
    "source-equivalence rules. It does not rerun Leiden, execute new routes, "
    "promote basin walls, replay full NanoClustering, evaluate quality/cost "
    "value, or claim method success."
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


def _load_context(nonreturn_dir: Path) -> dict[str, Any]:
    return {
        "nonreturn_summary": _read_json(
            nonreturn_dir
            / "nanoclustering_g4_8_first_pass_016_reverse_nonreturn_summary.json"
        ),
        "nonreturn_gates": _read_csv(
            nonreturn_dir
            / "nanoclustering_g4_8_first_pass_016_reverse_nonreturn_gate_matrix.csv"
        ),
        "final_rows": _read_csv(
            nonreturn_dir
            / "nanoclustering_g4_8_first_pass_016_reverse_nonreturn_final_state_rows.csv"
        ),
    }


def _route_rule_rows(final_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in final_rows.sort_values(["start_condition", "seed"], kind="mergesort").itertuples(
        index=False
    ):
        final_signature = str(row.final_signature_id)
        is_target_or_transient = final_signature in {TARGET_SIGNATURE_ID, TRANSIENT_SIGNATURE_ID}
        strict_same_seed = bool(_as_bool(row.same_seed_source_return))
        same_start_source = bool(_as_bool(row.same_start_source_family_signature))
        global_source = bool(_as_bool(row.global_source_family_signature))
        guard_match = bool(_as_bool(row.same_seed_drop_direct_guard_match))
        guard_only = bool(guard_match and not strict_same_seed)
        support_nearest_original = float(row.final_support_distance_to_original) <= min(
            float(row.final_support_distance_to_drop_bridge_edges),
            float(row.final_support_distance_to_drop_direct_edge),
        ) + EPS
        same_start_excluding_guard_only = bool(same_start_source and not guard_only)
        if strict_same_seed:
            preferred_status = "source_equivalent_same_seed_anchor"
        elif same_start_source and guard_only:
            preferred_status = "source_family_equivalent_guard_overlap_caveat"
        elif same_start_source:
            preferred_status = "source_family_equivalent_anchor_mismatch"
        elif global_source:
            preferred_status = "global_source_family_only_caveat"
        else:
            preferred_status = "not_source_family_equivalent"
        rows.append(
            {
                "local_pair_id": PRIMARY_PAIR_ID,
                "route_key": str(row.route_key),
                "start_condition": str(row.start_condition),
                "seed": int(row.seed),
                "final_signature_id": final_signature,
                "final_assignment_by_step": str(row.final_assignment_by_step),
                "final_state_stratum": str(row.final_state_stratum),
                "final_mechanism_read": str(row.final_mechanism_read),
                "strict_same_seed_source_equivalent": strict_same_seed,
                "same_start_source_family_equivalent": same_start_source,
                "same_start_source_family_excluding_guard_only_equivalent": same_start_excluding_guard_only,
                "global_source_family_equivalent": global_source,
                "support_nearest_original_equivalent": bool(support_nearest_original),
                "same_seed_drop_direct_guard_match": guard_match,
                "guard_only_source_family_overlap": guard_only,
                "target_or_transient_final_signature": bool(is_target_or_transient),
                "preferred_source_equivalence_status": preferred_status,
                "final_support_distance_to_original": float(
                    row.final_support_distance_to_original
                ),
                "final_support_distance_to_drop_bridge_edges": float(
                    row.final_support_distance_to_drop_bridge_edges
                ),
                "final_support_distance_to_drop_direct_edge": float(
                    row.final_support_distance_to_drop_direct_edge
                ),
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "method_status": METHOD_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
                "run_status": RUN_STATUS,
            }
        )
    return pd.DataFrame(rows)


def _rule_rows(route_rows: pd.DataFrame) -> pd.DataFrame:
    rule_specs = [
        (
            "R0_strict_same_seed_source_anchor",
            "strict_same_seed_source_equivalent",
            "same-seed original anchor must match",
            "too_narrow_baseline",
        ),
        (
            "R1_same_start_source_family",
            "same_start_source_family_equivalent",
            "final signature must be in original signatures observed for the same start condition",
            "preferred_with_guard_caveat",
        ),
        (
            "R2_same_start_source_family_excluding_guard_only",
            "same_start_source_family_excluding_guard_only_equivalent",
            "same-start source family, but guard-only overlap remains non-equivalent",
            "conservative_guard_exclusion",
        ),
        (
            "R3_global_source_family",
            "global_source_family_equivalent",
            "final signature can match any original signature for the pair",
            "redundant_or_too_broad",
        ),
        (
            "R4_support_nearest_original",
            "support_nearest_original_equivalent",
            "same-seed support distance to original is no worse than target/guard anchors",
            "distance_only_negative_control",
        ),
    ]
    rows: list[dict[str, Any]] = []
    total = int(len(route_rows))
    target_transient_leaks = {
        rule_id: int(
            route_rows[
                route_rows[column].astype(bool)
                & route_rows["target_or_transient_final_signature"].astype(bool)
            ].shape[0]
        )
        for rule_id, column, _, _ in rule_specs
    }
    for rule_id, column, rule, interpretation in rule_specs:
        accepted = route_rows[route_rows[column].astype(bool)]
        guard_caveats = int(
            accepted["guard_only_source_family_overlap"].astype(bool).sum()
        )
        anchor_mismatch = int(
            accepted["preferred_source_equivalence_status"]
            .astype(str)
            .str.contains("anchor_mismatch")
            .sum()
        )
        if rule_id == "R1_same_start_source_family":
            status = (
                "preferred_operational_source_family_equivalence_with_guard_caveat"
                if len(accepted) == total and target_transient_leaks[rule_id] == 0
                else "not_preferred"
            )
        elif rule_id == "R0_strict_same_seed_source_anchor":
            status = "underaccepts_source_family_final_states"
        elif rule_id == "R2_same_start_source_family_excluding_guard_only":
            status = "conservative_but_rejects_named_guard_overlap"
        elif rule_id == "R3_global_source_family":
            status = "same_result_here_but_less_local_than_same_start"
        else:
            status = "underaccepts_because_support_nearest_is_not_source_family"
        rows.append(
            {
                "rule_id": rule_id,
                "rule": rule,
                "rule_column": column,
                "interpretation_role": interpretation,
                "accepted_route_count": int(len(accepted)),
                "rejected_route_count": int(total - len(accepted)),
                "target_or_transient_leak_count": target_transient_leaks[rule_id],
                "guard_overlap_caveat_count": guard_caveats,
                "source_family_anchor_mismatch_count": anchor_mismatch,
                "accepted_final_signatures": ";".join(
                    sorted(accepted["final_signature_id"].astype(str).unique())
                ),
                "rule_status": status,
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "method_status": METHOD_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
                "run_status": RUN_STATUS,
            }
        )
    return pd.DataFrame(rows)


def _decision_rows(route_rows: pd.DataFrame, rule_rows: pd.DataFrame) -> pd.DataFrame:
    preferred = rule_rows[rule_rows["rule_id"].astype(str).eq("R1_same_start_source_family")]
    strict = rule_rows[rule_rows["rule_id"].astype(str).eq("R0_strict_same_seed_source_anchor")]
    support = rule_rows[rule_rows["rule_id"].astype(str).eq("R4_support_nearest_original")]
    guard_caveat_count = int(
        route_rows["guard_only_source_family_overlap"].astype(bool).sum()
    )
    decisions = [
        {
            "decision_id": "D1_strict_same_seed_anchor_underaccepts",
            "axis": "source_equivalence_width",
            "observed": strict[
                ["accepted_route_count", "rejected_route_count"]
            ].to_dict("records"),
            "decision": "strict_same_seed_source_anchor_is_too_narrow",
            "passes": int(strict["accepted_route_count"].iloc[0]) == 15,
            "claim_effect": "same-seed anchor matching should remain a strict subcase, not the source-family definition",
        },
        {
            "decision_id": "D2_same_start_source_family_covers_final_states",
            "axis": "preferred_rule",
            "observed": preferred[
                [
                    "accepted_route_count",
                    "target_or_transient_leak_count",
                    "guard_overlap_caveat_count",
                    "source_family_anchor_mismatch_count",
                ]
            ].to_dict("records"),
            "decision": "same_start_source_family_is_preferred_operational_equivalence",
            "passes": int(preferred["accepted_route_count"].iloc[0]) == 24
            and int(preferred["target_or_transient_leak_count"].iloc[0]) == 0,
            "claim_effect": "reverse final states can be read as source-family return with caveats",
        },
        {
            "decision_id": "D3_guard_overlap_must_remain_named",
            "axis": "guard_caveat",
            "observed": f"{guard_caveat_count} guard-only overlap rows",
            "decision": "guard_overlap_is_source_family_caveat_not_clean_source",
            "passes": guard_caveat_count == 1,
            "claim_effect": "prevents drop-direct guard residual from becoming clean wall/pathway evidence",
        },
        {
            "decision_id": "D4_support_nearest_original_is_not_definition",
            "axis": "support_distance",
            "observed": support[
                ["accepted_route_count", "rejected_route_count"]
            ].to_dict("records"),
            "decision": "support_nearest_original_underaccepts_source_family_rows",
            "passes": int(support["accepted_route_count"].iloc[0]) < 24,
            "claim_effect": "support distance remains a caveat field, not the equivalence rule",
        },
        {
            "decision_id": "D5_claim_boundary",
            "axis": "claim_boundary",
            "observed": CLAIM_BOUNDARY,
            "decision": "read_only_definition_audit_only",
            "passes": True,
            "claim_effect": "keeps wall, method, full replay, and quality claims closed",
        },
    ]
    return pd.DataFrame(decisions)


def _gate_matrix(
    *,
    nonreturn_gates: pd.DataFrame,
    route_rows: pd.DataFrame,
    rule_rows: pd.DataFrame,
    decision_rows: pd.DataFrame,
) -> pd.DataFrame:
    preferred = rule_rows[rule_rows["rule_id"].astype(str).eq("R1_same_start_source_family")]
    strict = rule_rows[rule_rows["rule_id"].astype(str).eq("R0_strict_same_seed_source_anchor")]
    support = rule_rows[rule_rows["rule_id"].astype(str).eq("R4_support_nearest_original")]
    return pd.DataFrame(
        [
            _gate_row(
                "G1_upstream_nonreturn_audit_passed",
                "Did the upstream reverse non-return stratification pass?",
                nonreturn_gates["gate_status"].value_counts().to_dict(),
                "all upstream non-return gates pass",
                bool(nonreturn_gates["gate_status"].astype(str).eq("pass").all()),
            ),
            _gate_row(
                "G2_all_final_routes_scored",
                "Were all 24 reverse final states scored by every equivalence rule?",
                {
                    "route_rule_rows": len(route_rows),
                    "rule_rows": len(rule_rows),
                    "rule_ids": rule_rows["rule_id"].tolist(),
                },
                "24 route rows and 5 rule rows",
                len(route_rows) == 24 and len(rule_rows) == 5,
            ),
            _gate_row(
                "G3_strict_anchor_underacceptance_visible",
                "Does strict same-seed source-anchor matching underaccept source-family final states?",
                strict[["accepted_route_count", "rejected_route_count"]].to_dict(
                    "records"
                ),
                "strict rule accepts 15/24 and rejects 9/24",
                int(strict["accepted_route_count"].iloc[0]) == 15
                and int(strict["rejected_route_count"].iloc[0]) == 9,
            ),
            _gate_row(
                "G4_preferred_same_start_source_family_rule",
                "Does same-start source-family equivalence cover all finals without target/transient leaks?",
                preferred[
                    [
                        "accepted_route_count",
                        "target_or_transient_leak_count",
                        "guard_overlap_caveat_count",
                    ]
                ].to_dict("records"),
                "same-start source family accepts 24/24, leaks 0 target/transient finals, names guard caveat",
                int(preferred["accepted_route_count"].iloc[0]) == 24
                and int(preferred["target_or_transient_leak_count"].iloc[0]) == 0
                and int(preferred["guard_overlap_caveat_count"].iloc[0]) == 1,
            ),
            _gate_row(
                "G5_support_distance_not_promoted_to_definition",
                "Is support-nearest-original shown to be insufficient as the source equivalence rule?",
                support[["accepted_route_count", "rejected_route_count"]].to_dict(
                    "records"
                ),
                "support-nearest-original rejects source-family rows",
                int(support["accepted_route_count"].iloc[0]) < 24,
            ),
            _gate_row(
                "G6_claim_boundaries_closed",
                "Are wall, method, full replay, and quality/cost claims closed?",
                {
                    "decision_passes": int(decision_rows["passes"].map(_as_bool).sum()),
                    "claim_boundary": CLAIM_BOUNDARY,
                },
                "all decisions pass and claim boundary is read-only",
                bool(decision_rows["passes"].map(_as_bool).all()),
            ),
        ]
    )


def _summary(
    *,
    nonreturn_dir: Path,
    output_dir: Path,
    route_rows: pd.DataFrame,
    rule_rows: pd.DataFrame,
    decision_rows: pd.DataFrame,
    gates: pd.DataFrame,
) -> dict[str, Any]:
    preferred = rule_rows[rule_rows["rule_id"].astype(str).eq("R1_same_start_source_family")].iloc[0]
    strict = rule_rows[rule_rows["rule_id"].astype(str).eq("R0_strict_same_seed_source_anchor")].iloc[0]
    support = rule_rows[rule_rows["rule_id"].astype(str).eq("R4_support_nearest_original")].iloc[0]
    return {
        "schema": "nanoclustering_g4_8_first_pass_016_source_family_equivalence_summary.v1",
        "status": RUN_STATUS,
        "nonreturn_dir": str(nonreturn_dir),
        "output_dir": str(output_dir),
        "primary_pair": PRIMARY_PAIR_ID,
        "route_rule_row_count": int(len(route_rows)),
        "rule_row_count": int(len(rule_rows)),
        "decision_row_count": int(len(decision_rows)),
        "preferred_source_equivalence_rule": "same_start_source_family_with_guard_caveat",
        "preferred_rule_accepts": int(preferred["accepted_route_count"]),
        "preferred_rule_target_or_transient_leaks": int(
            preferred["target_or_transient_leak_count"]
        ),
        "preferred_rule_guard_caveats": int(preferred["guard_overlap_caveat_count"]),
        "strict_same_seed_accepts": int(strict["accepted_route_count"]),
        "support_nearest_original_accepts": int(support["accepted_route_count"]),
        "preferred_status_counts": route_rows[
            "preferred_source_equivalence_status"
        ].value_counts().to_dict(),
        "gate_status_counts": gates["gate_status"].value_counts().to_dict(),
        "failed_gates": gates.loc[
            ~gates["gate_status"].astype(str).eq("pass"),
            "gate_id",
        ].tolist(),
        "interpretation": (
            "For 016 reverse final states, strict same-seed source-anchor "
            "matching is too narrow: it accepts 15/24. Same-start source-family "
            "equivalence accepts all 24 final states without target/transient "
            "leaks, but one row remains a named drop-direct guard-overlap "
            "caveat. Support-nearest-original is also too narrow and should "
            "stay a diagnostic field."
        ),
        "recommended_next_gate": (
            "Use same-start source-family equivalence with a guard-overlap caveat "
            "as the 016 reverse final-state readout, then re-summarize forward/"
            "reverse pathway shape before any threshold localization."
        ),
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _write_report(
    *,
    path: Path,
    summary: dict[str, Any],
    route_rows: pd.DataFrame,
    rule_rows: pd.DataFrame,
    decision_rows: pd.DataFrame,
    gates: pd.DataFrame,
) -> None:
    lines = [
        "# NanoClustering G4.8 First-Pass 016 Source-Family Equivalence Audit",
        "",
        "## Summary",
        "",
        f"- status: {summary['status']}",
        f"- preferred_source_equivalence_rule: {summary['preferred_source_equivalence_rule']}",
        f"- preferred_rule_accepts: {summary['preferred_rule_accepts']}",
        f"- preferred_rule_target_or_transient_leaks: {summary['preferred_rule_target_or_transient_leaks']}",
        f"- preferred_rule_guard_caveats: {summary['preferred_rule_guard_caveats']}",
        f"- failed_gates: {summary['failed_gates']}",
        "",
        "## Rule Rows",
        "",
        _markdown_table(
            rule_rows,
            [
                "rule_id",
                "interpretation_role",
                "accepted_route_count",
                "rejected_route_count",
                "target_or_transient_leak_count",
                "guard_overlap_caveat_count",
                "source_family_anchor_mismatch_count",
                "accepted_final_signatures",
                "rule_status",
            ],
            max_rows=20,
        ),
        "",
        "## Route Rule Rows",
        "",
        _markdown_table(
            route_rows,
            [
                "route_key",
                "final_signature_id",
                "final_state_stratum",
                "strict_same_seed_source_equivalent",
                "same_start_source_family_equivalent",
                "guard_only_source_family_overlap",
                "support_nearest_original_equivalent",
                "preferred_source_equivalence_status",
            ],
            max_rows=30,
        ),
        "",
        "## Decisions",
        "",
        _markdown_table(
            decision_rows,
            ["decision_id", "axis", "observed", "decision", "passes", "claim_effect"],
            max_rows=20,
        ),
        "",
        "## Gates",
        "",
        _markdown_table(
            gates,
            ["gate_id", "question", "observed", "minimum_or_rule", "gate_status"],
            max_rows=20,
        ),
        "",
        "## Interpretation",
        "",
        summary["interpretation"],
        "",
        "## Recommended Next Gate",
        "",
        summary["recommended_next_gate"],
        "",
        "## Claim Boundary",
        "",
        summary["claim_boundary"],
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nonreturn-dir", type=Path, default=DEFAULT_NONRETURN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    nonreturn_dir = Path(args.nonreturn_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    context = _load_context(nonreturn_dir)
    route_rows = _route_rule_rows(context["final_rows"])
    rule_rows = _rule_rows(route_rows)
    decision_rows = _decision_rows(route_rows, rule_rows)
    gates = _gate_matrix(
        nonreturn_gates=context["nonreturn_gates"],
        route_rows=route_rows,
        rule_rows=rule_rows,
        decision_rows=decision_rows,
    )
    summary = _summary(
        nonreturn_dir=nonreturn_dir,
        output_dir=output_dir,
        route_rows=route_rows,
        rule_rows=rule_rows,
        decision_rows=decision_rows,
        gates=gates,
    )
    config = {
        "schema": "nanoclustering_g4_8_first_pass_016_source_family_equivalence_config.v1",
        "nonreturn_dir": str(nonreturn_dir),
        "output_dir": str(output_dir),
        "claim_boundary": CLAIM_BOUNDARY,
    }

    _write_csv(route_rows, output_dir / ROUTE_RULE_ROWS_CSV)
    _write_csv(rule_rows, output_dir / RULE_ROWS_CSV)
    _write_csv(decision_rows, output_dir / DECISION_ROWS_CSV)
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
        route_rows=route_rows,
        rule_rows=rule_rows,
        decision_rows=decision_rows,
        gates=gates,
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
