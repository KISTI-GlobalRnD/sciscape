#!/usr/bin/env python3
"""Materialize the precommitted non-field34 methodology-v0 panel.

This is M1 in the Leiden basin methodology-v0 design. It selects a panel from
existing basin-existence and calibration artifacts before any route execution.
It does not run probes, promote walls, inspect basin quality/cost, or make a
directed-search claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").exists()
)
BASE_RESULT_DIR = REPO_ROOT / "research/consensus/results/adaptive_refinement"
DEFAULT_EXISTENCE_DIR = BASE_RESULT_DIR / "leiden_basin_existence_assumption_audit_20260529"
DEFAULT_CALIBRATION_DIR = BASE_RESULT_DIR / "leiden_basin_definition_calibration_20260528"
DEFAULT_OUTPUT_DIR = BASE_RESULT_DIR / "leiden_basin_methodology_v0_20260529"

CASE_ROWS_CSV = "basin_existence_case_rows.csv"
PAIR_ROWS_CSV = "basin_existence_pair_rows.csv"
CALIBRATION_WALL_ROWS_CSV = "wall_candidate_pair_rows.csv"

PANEL_CSV = "precommitted_nonfield34_panel_v0.csv"
PANEL_DECISION_CSV = "precommitted_nonfield34_panel_decision_rows_v0.csv"
PAIR_CANDIDATES_CSV = "precommitted_nonfield34_pair_candidates_v0.csv"
SUMMARY_JSON = "precommitted_nonfield34_panel_v0_summary.json"
REPORT_MD = "precommitted_nonfield34_panel_v0_report.md"
CONFIG_JSON = "basin_methodology_v0_config.json"
SCHEMA_JSON = "wall_pathway_evidence_schema_v0.json"

CLAIM_BOUNDARY = (
    "Methodology-v0 panel materialization only; no route execution, "
    "wall-promotion change, basin-quality claim, cost claim, or "
    "directed-search claim."
)
QUALITY_COST_STATUS = "excluded_by_methodology_v0"
ROUTE_EXECUTION_STATUS = "not_executed_m1_panel_materialization_only"
WALL_PROMOTION_STATUS = "not_promoted_schema_only"
MISSING_WALL_EVIDENCE_STATUS = "not_evaluated_requires_m3_schema_review"

STRONG_ROLE = "strong_h1"
MODERATE_ROLE = "moderate_h1"
AMBIGUOUS_ROLE = "ambiguous_definition_control"

ROLE_QUOTAS = {
    STRONG_ROLE: 3,
    MODERATE_ROLE: 2,
    AMBIGUOUS_ROLE: 2,
}

MIN_PANEL_GATES = {
    "min_fields": 2,
    "min_method_families": 3,
    "min_strong_h1_cases": 2,
    "min_moderate_h1_cases": 1,
    "min_ambiguous_definition_controls": 1,
}

PAIR_STATUSES = {
    "strong_meaningful_distinct_basin_candidate_pair",
    "moderate_meaningful_distinct_basin_candidate_pair",
}


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"empty CSV: {path}") from exc


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _count(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in frame:
        return {}
    return {str(k): int(v) for k, v in frame[column].value_counts(dropna=False).to_dict().items()}


def _method_family(method: Any) -> str:
    text = "" if pd.isna(method) else str(method)
    for prefix in ("gcc_emb_full_knn30_", "all_edges_"):
        if text.startswith(prefix):
            return text[len(prefix) :]
    return text


def _safe_int(value: Any) -> int:
    if pd.isna(value):
        return 0
    return int(value)


def _role_for_case(row: pd.Series) -> str:
    field = str(row.get("field", ""))
    status = str(row.get("multi_basin_existence_status", ""))
    if field == "field34":
        return "exclude_field34_hygiene_limited"
    if status == "strong_candidate_multi_basin_existence_evidence":
        return STRONG_ROLE
    if status == "moderate_candidate_multi_basin_existence_evidence":
        return MODERATE_ROLE
    if status == "ambiguous_basin_relation_only":
        return AMBIGUOUS_ROLE
    if status == "weak_or_low_support_distinct_candidate_hold":
        return "exclude_weak_distinct_hold"
    return "exclude_no_current_h1_signal"


def _role_reason(row: pd.Series) -> str:
    role = str(row["panel_candidate_role"])
    if role == STRONG_ROLE:
        return (
            f"{_safe_int(row['strong_meaningful_pair_count'])} strong meaningful pairs; "
            f"support median {row['endpoint_support_median']}"
        )
    if role == MODERATE_ROLE:
        return (
            f"{_safe_int(row['moderate_meaningful_pair_count'])} moderate meaningful pairs; "
            f"support median {row['endpoint_support_median']}"
        )
    if role == AMBIGUOUS_ROLE:
        return (
            f"{_safe_int(row['accepted_endpoint_identity_count'])} endpoint identities but "
            f"{_safe_int(row['strong_meaningful_pair_count'])} strong and "
            f"{_safe_int(row['moderate_meaningful_pair_count'])} moderate accepted pairs"
        )
    if role == "exclude_field34_hygiene_limited":
        return "field34 is hygiene-limited under the current evidence audit"
    if role == "exclude_weak_distinct_hold":
        return "distinct relation exists only below the v0 support gate"
    return "not selected by methodology-v0 H1 panel rule"


def _prepare_case_decisions(case_rows: pd.DataFrame) -> pd.DataFrame:
    rows = case_rows.copy()
    rows["method_family"] = rows["method"].map(_method_family)
    rows["panel_candidate_role"] = rows.apply(_role_for_case, axis=1)
    rows["candidate_role_reason"] = rows.apply(_role_reason, axis=1)
    rows["selection_rank_score"] = (
        rows["strong_meaningful_pair_count"].fillna(0).astype(int) * 1_000_000
        + rows["moderate_meaningful_pair_count"].fillna(0).astype(int) * 10_000
        + rows["distinct_support_local"].fillna(0).astype(int) * 100
        + rows["endpoint_support_median"].fillna(0).astype(float)
    )
    rows["panel_decision"] = "not_selected"
    rows["panel_decision_reason"] = rows["candidate_role_reason"]
    rows["panel_role"] = ""
    rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["quality_cost_status"] = QUALITY_COST_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


def _select_strong(rows: pd.DataFrame) -> list[int]:
    candidates = rows[rows["panel_candidate_role"].eq(STRONG_ROLE)].copy()
    candidates = candidates.sort_values(
        [
            "strong_meaningful_pair_count",
            "distinct_support_local",
            "endpoint_support_median",
            "case_id",
        ],
        ascending=[False, False, False, True],
    )
    return list(candidates.head(ROLE_QUOTAS[STRONG_ROLE]).index)


def _select_moderate(rows: pd.DataFrame) -> list[int]:
    candidates = rows[rows["panel_candidate_role"].eq(MODERATE_ROLE)].copy()
    candidates = candidates.sort_values(
        [
            "moderate_meaningful_pair_count",
            "distinct_support_local",
            "endpoint_support_median",
            "case_id",
        ],
        ascending=[False, False, False, True],
    )
    return list(candidates.head(ROLE_QUOTAS[MODERATE_ROLE]).index)


def _select_ambiguous_controls(rows: pd.DataFrame) -> list[int]:
    candidates = rows[rows["panel_candidate_role"].eq(AMBIGUOUS_ROLE)].copy()
    candidates["citation_embedding_priority"] = candidates["method"].astype(str).str.contains(
        "citation_embedding"
    )
    candidates = candidates.sort_values(
        [
            "citation_embedding_priority",
            "endpoint_support_median",
            "accepted_endpoint_identity_count",
            "case_id",
        ],
        ascending=[False, False, False, True],
    )
    selected: list[int] = []
    used_fields: set[str] = set()
    for idx, row in candidates.iterrows():
        if str(row["field"]) in used_fields:
            continue
        selected.append(idx)
        used_fields.add(str(row["field"]))
        if len(selected) >= ROLE_QUOTAS[AMBIGUOUS_ROLE]:
            break
    if len(selected) < ROLE_QUOTAS[AMBIGUOUS_ROLE]:
        for idx in candidates.index:
            if idx in selected:
                continue
            selected.append(idx)
            if len(selected) >= ROLE_QUOTAS[AMBIGUOUS_ROLE]:
                break
    return selected


def _select_panel(case_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    decisions = _prepare_case_decisions(case_rows)
    selected_indices = (
        _select_strong(decisions)
        + _select_moderate(decisions)
        + _select_ambiguous_controls(decisions)
    )
    selected_set = set(selected_indices)
    decisions.loc[list(selected_set), "panel_decision"] = "selected_precommitted_panel_v0"
    decisions.loc[list(selected_set), "panel_role"] = decisions.loc[
        list(selected_set), "panel_candidate_role"
    ]
    decisions.loc[
        decisions["panel_decision"].eq("selected_precommitted_panel_v0"),
        "panel_decision_reason",
    ] = decisions.loc[
        decisions["panel_decision"].eq("selected_precommitted_panel_v0"),
        "candidate_role_reason",
    ].map(lambda reason: f"selected by methodology-v0 panel quota; {reason}")

    selected = decisions[decisions["panel_decision"].eq("selected_precommitted_panel_v0")].copy()
    role_order = {STRONG_ROLE: 0, MODERATE_ROLE: 1, AMBIGUOUS_ROLE: 2}
    selected["panel_role_order"] = selected["panel_role"].map(role_order)
    selected = selected.sort_values(
        [
            "panel_role_order",
            "field",
            "method_family",
            "case_id",
        ]
    ).drop(columns=["panel_role_order"])
    decisions = decisions.sort_values(
        ["panel_decision", "panel_candidate_role", "field", "method_family", "case_id"]
    )
    return selected.reset_index(drop=True), decisions.reset_index(drop=True)


def _pair_candidates(
    *,
    selected_panel: pd.DataFrame,
    pair_rows: pd.DataFrame,
    calibration_wall_rows: pd.DataFrame,
) -> pd.DataFrame:
    selected_case_roles = selected_panel.set_index("case_id")["panel_role"].to_dict()
    selected_cases = set(selected_case_roles)
    rows = pair_rows[
        pair_rows["case_id"].isin(selected_cases)
        & pair_rows["meaningful_basin_pair_status"].isin(PAIR_STATUSES)
        & pair_rows["field_hygiene_class"].eq("clean_non_field34")
    ].copy()
    if rows.empty:
        return rows

    join_cols = ["case_id", "left_endpoint_identity_id", "right_endpoint_identity_id"]
    add_cols = join_cols + [
        "endpoint_distance_min",
        "endpoint_distance_max",
        "calibrated_relation",
        "wall_evidence_allowed",
    ]
    rows = rows.merge(
        calibration_wall_rows[add_cols],
        on=join_cols,
        how="left",
    )
    rows["panel_role"] = rows["case_id"].map(selected_case_roles)
    rows["panel_pair_status"] = "accepted_distinct_basin_pair_for_m3_schema_review"
    rows["pathway_execution_status"] = ROUTE_EXECUTION_STATUS
    rows["wall_evidence_status"] = MISSING_WALL_EVIDENCE_STATUS
    rows["quality_cost_status"] = QUALITY_COST_STATUS
    rows["route_label_v0"] = "unknown_not_executed"
    rows["required_next_evidence"] = (
        "endpoint_assignment;direct_path_availability;objective_debt;"
        "debt_recovery;polish_reversion;support_incompatibility"
    )
    rows["claim_boundary"] = CLAIM_BOUNDARY
    role_order = {STRONG_ROLE: 0, MODERATE_ROLE: 1, AMBIGUOUS_ROLE: 2}
    rows["panel_role_order"] = rows["panel_role"].map(role_order)
    rows = rows.sort_values(
        [
            "panel_role_order",
            "case_id",
            "meaningful_basin_pair_status",
            "support_distance_max",
            "left_endpoint_identity_id",
            "right_endpoint_identity_id",
        ],
        ascending=[True, True, True, False, True, True],
    ).drop(columns=["panel_role_order"])
    cols = [
        "case_id",
        "field",
        "method",
        "candidate_budget",
        "panel_role",
        "left_endpoint_identity_id",
        "right_endpoint_identity_id",
        "endpoint_distance_min",
        "endpoint_distance_max",
        "support_distance_min",
        "support_distance_max",
        "left_support_node_count",
        "right_support_node_count",
        "min_endpoint_support",
        "support_substance_class",
        "calibrated_relation",
        "meaningful_basin_pair_status",
        "panel_pair_status",
        "wall_evidence_allowed",
        "pathway_execution_status",
        "wall_evidence_status",
        "route_label_v0",
        "required_next_evidence",
        "quality_cost_status",
        "claim_boundary",
    ]
    return rows[cols].reset_index(drop=True)


def _schema() -> dict[str, Any]:
    return {
        "schema_version": "wall_pathway_evidence_schema_v0",
        "scope": "schema only; no route execution in M1",
        "required_fields": [
            "source_basin_candidate",
            "target_basin_candidate",
            "endpoint_identity_evidence_grade",
            "support_local_relation_status",
            "route_family",
            "direct_path_availability",
            "objective_debt_evidence",
            "debt_recovery_evidence",
            "polish_reversion_evidence",
            "support_incompatibility_evidence",
            "final_endpoint_assignment_after_route",
            "route_label",
            "confidence",
        ],
        "primary_wall_signals": [
            "failed_direct_transition",
            "objective_debt_before_target_relation",
            "polish_reversion_from_target_like_state",
            "support_incompatibility",
        ],
        "consistency_checks": [
            "endpoint_assignment_measured_not_inferred_from_support_only",
            "not_field34_hygiene_limited",
            "not_same_control_behavior",
            "route_label_independent_of_final_quality",
        ],
        "route_labels": {
            "crosses": "source to target basin candidate with wall debt paid or recovered",
            "bounces": "target-like intermediate followed by return to source or ambiguity",
            "collapses": "leaves source but lands outside source and target candidates",
            "unknown": "missing endpoint assignment, relation ambiguity, or hygiene limit",
        },
        "forbidden_deciders": [
            "final_quality",
            "material_gain",
            "wall_time",
            "cost_adjusted_gain",
            "route_order_stability_alone",
        ],
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _panel_gates(selected: pd.DataFrame) -> dict[str, bool]:
    role_counts = selected["panel_role"].value_counts().to_dict()
    return {
        "at_least_2_fields": bool(
            int(selected["field"].nunique()) >= MIN_PANEL_GATES["min_fields"]
        ),
        "at_least_3_method_families": bool(
            int(selected["method_family"].nunique())
            >= MIN_PANEL_GATES["min_method_families"]
        ),
        "at_least_2_strong_h1_cases": bool(
            int(role_counts.get(STRONG_ROLE, 0)) >= MIN_PANEL_GATES["min_strong_h1_cases"]
        ),
        "at_least_1_moderate_h1_case": bool(
            int(role_counts.get(MODERATE_ROLE, 0)) >= MIN_PANEL_GATES["min_moderate_h1_cases"]
        ),
        "at_least_1_ambiguous_definition_control": bool(
            int(role_counts.get(AMBIGUOUS_ROLE, 0))
            >= MIN_PANEL_GATES["min_ambiguous_definition_controls"]
        ),
        "no_field34_rows": bool(not selected["field"].astype(str).eq("field34").any()),
        "quality_cost_excluded": bool(
            selected["quality_cost_status"].eq(QUALITY_COST_STATUS).all()
        ),
        "route_execution_not_run": bool(
            selected["route_execution_status"].eq(ROUTE_EXECUTION_STATUS).all()
        ),
    }


def _summary(
    *,
    selected_panel: pd.DataFrame,
    decisions: pd.DataFrame,
    pair_candidates: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    gates = _panel_gates(selected_panel)
    gate_pass = all(gates.values())
    return {
        "status": "precommitted_nonfield34_panel_v0_prepared",
        "date": "2026-05-29",
        "script": _rel(Path(__file__).resolve()),
        "output_dir": _rel(output_dir),
        "panel_case_count": int(len(selected_panel)),
        "panel_role_counts": _count(selected_panel, "panel_role"),
        "panel_field_count": int(selected_panel["field"].nunique()),
        "panel_fields": sorted(str(v) for v in selected_panel["field"].unique()),
        "panel_method_family_count": int(selected_panel["method_family"].nunique()),
        "panel_method_families": sorted(str(v) for v in selected_panel["method_family"].unique()),
        "panel_pair_candidate_count": int(len(pair_candidates)),
        "panel_pair_status_counts": _count(pair_candidates, "meaningful_basin_pair_status"),
        "all_case_decision_counts": _count(decisions, "panel_decision"),
        "all_candidate_role_counts": _count(decisions, "panel_candidate_role"),
        "panel_gates": gates,
        "panel_gate_status": "passed" if gate_pass else "failed",
        "decision": (
            "Methodology-v0 M1 panel passes the precommitted non-field34 shape gates."
            if gate_pass
            else "Methodology-v0 M1 panel does not yet pass the precommitted shape gates."
        ),
        "next_step": (
            "Proceed to M2 evidence enrichment without quality/cost joins, then run M3 "
            "wall/pathway schema review before any pathway probe."
            if gate_pass
            else "Revise the panel-selection rule before any pathway probe."
        ),
        "paths": {
            "panel": _rel(output_dir / PANEL_CSV),
            "panel_decisions": _rel(output_dir / PANEL_DECISION_CSV),
            "pair_candidates": _rel(output_dir / PAIR_CANDIDATES_CSV),
            "summary": _rel(output_dir / SUMMARY_JSON),
            "report": _rel(output_dir / REPORT_MD),
            "config": _rel(output_dir / CONFIG_JSON),
            "wall_pathway_schema": _rel(output_dir / SCHEMA_JSON),
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _write_report(
    path: Path,
    summary: dict[str, Any],
    selected_panel: pd.DataFrame,
    pair_candidates: pd.DataFrame,
) -> None:
    lines = [
        "# Precommitted Non-Field34 Panel v0",
        "",
        "Date: 2026-05-29",
        "",
        "## Scope",
        "",
        "This artifact materializes methodology-v0 M1 from existing evidence.",
        "It selects a non-field34 panel before route execution and keeps quality,",
        "cost, wall promotion, and directed-search claims out of scope.",
        "",
        "## Decision",
        "",
        str(summary["decision"]),
        "",
        "## Panel Shape",
        "",
        f"- cases: `{summary['panel_case_count']}`",
        f"- fields: `{summary['panel_field_count']}` ({', '.join(summary['panel_fields'])})",
        f"- method families: `{summary['panel_method_family_count']}` "
        f"({', '.join(summary['panel_method_families'])})",
        f"- pair candidates for M3 schema review: `{summary['panel_pair_candidate_count']}`",
        "",
        "## Gate Status",
        "",
        "| gate | passed |",
        "| --- | --- |",
    ]
    for gate, passed in summary["panel_gates"].items():
        lines.append(f"| {gate} | `{str(passed).lower()}` |")
    lines.extend(
        [
            "",
            "## Selected Cases",
            "",
            "| role | case_id | field | method | endpoint identities | strong pairs | moderate pairs | reason |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in selected_panel.itertuples(index=False):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.panel_role),
                    str(row.case_id),
                    str(row.field),
                    str(row.method),
                    str(row.accepted_endpoint_identity_count),
                    str(row.strong_meaningful_pair_count),
                    str(row.moderate_meaningful_pair_count),
                    str(row.panel_decision_reason),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Pair Candidate Counts",
            "",
            "| status | rows |",
            "| --- | ---: |",
        ]
    )
    for status, count in summary["panel_pair_status_counts"].items():
        lines.append(f"| {status} | {count} |")
    lines.extend(
        [
            "",
            "## Next Step",
            "",
            str(summary["next_step"]),
            "",
            "Claim boundary: " + CLAIM_BOUNDARY,
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(
    *,
    existence_dir: Path,
    calibration_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    case_rows = _read_csv(existence_dir / CASE_ROWS_CSV)
    pair_rows = _read_csv(existence_dir / PAIR_ROWS_CSV)
    calibration_wall_rows = _read_csv(calibration_dir / CALIBRATION_WALL_ROWS_CSV)

    selected_panel, decisions = _select_panel(case_rows)
    pair_candidates = _pair_candidates(
        selected_panel=selected_panel,
        pair_rows=pair_rows,
        calibration_wall_rows=calibration_wall_rows,
    )
    summary = _summary(
        selected_panel=selected_panel,
        decisions=decisions,
        pair_candidates=pair_candidates,
        output_dir=output_dir,
    )

    _write_csv(selected_panel, output_dir / PANEL_CSV)
    _write_csv(decisions, output_dir / PANEL_DECISION_CSV)
    _write_csv(pair_candidates, output_dir / PAIR_CANDIDATES_CSV)
    (output_dir / SCHEMA_JSON).write_text(
        json.dumps(_schema(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(
            {
                "existence_dir": _rel(existence_dir),
                "calibration_dir": _rel(calibration_dir),
                "output_dir": _rel(output_dir),
                "role_quotas": ROLE_QUOTAS,
                "minimum_panel_gates": MIN_PANEL_GATES,
                "pair_statuses": sorted(PAIR_STATUSES),
                "quality_cost_status": QUALITY_COST_STATUS,
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_report(output_dir / REPORT_MD, summary, selected_panel, pair_candidates)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existence-dir", type=Path, default=DEFAULT_EXISTENCE_DIR)
    parser.add_argument("--calibration-dir", type=Path, default=DEFAULT_CALIBRATION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = run(
        existence_dir=args.existence_dir,
        calibration_dir=args.calibration_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
