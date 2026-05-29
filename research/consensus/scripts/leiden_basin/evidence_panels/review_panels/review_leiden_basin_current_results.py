#!/usr/bin/env python3
"""Review current Leiden basin cartography results without adding new claims.

The review reconciles the current result chronology:

- relation taxonomy remains the relation-status source;
- combined route gate and Methodology v0 are the current route-status sources;
- margin validation review is the current margin-status source.

It emits a pair-state ledger, an evidence ledger, and a risk ledger for manual
review. It does not run routes or evaluate basin quality/cost.
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
SCRIPT_ROOT = REPO_ROOT / "research/consensus/scripts"
BASE_RESULT_DIR = REPO_ROOT / "research/consensus/results/adaptive_refinement"
DEFAULT_OUTPUT_DIR = BASE_RESULT_DIR / "leiden_basin_current_results_review_20260529"

PHASE1_INDEX_DIR = BASE_RESULT_DIR / "leiden_basin_phase1_index_20260528"
PHASE1_REVIEW_DIR = BASE_RESULT_DIR / "leiden_basin_phase1_review_20260528"
CALIBRATION_DIR = BASE_RESULT_DIR / "leiden_basin_definition_calibration_20260528"
ROUTE_JOIN_DIR = BASE_RESULT_DIR / "leiden_basin_route_wall_evidence_join_20260528"
DIRECT_AUDIT_DIR = BASE_RESULT_DIR / "direct_pair_route_audit_field34_cc_c0_c2_20260528"
PANEL_DIR = BASE_RESULT_DIR / "leiden_basin_wall_protocol_panel_20260528"
COMBINED_GATE_DIR = BASE_RESULT_DIR / "leiden_basin_route_gate_panel_combined_after_clean_distinct_20260528"
LATEST_COVERAGE_DIR = (
    BASE_RESULT_DIR / "leiden_basin_wall_panel_context_coverage_after_clean_distinct_route_gate_20260528"
)
MECHANISM_DIR = BASE_RESULT_DIR / "leiden_basin_clean_distinct_route_mechanism_review_20260528"
MARGIN_DIR = BASE_RESULT_DIR / "leiden_basin_polish_margin_gate_review_20260528"
METHODOLOGY_DIR = BASE_RESULT_DIR / "leiden_basin_methodology_v0_margin_validation_20260528"
MARGIN_VALIDATION_REVIEW_DIR = BASE_RESULT_DIR / "leiden_basin_margin_validation_panel_review_20260529"
RELATION_TAXONOMY_DIR = BASE_RESULT_DIR / "leiden_basin_relation_taxonomy_v01_20260528"

PAIR_STATE_CSV = "current_pair_state_ledger.csv"
EVIDENCE_LEDGER_CSV = "current_evidence_ledger.csv"
RISK_LEDGER_CSV = "current_risk_ledger.csv"
SUMMARY_JSON = "current_results_review_summary.json"
REPORT_MD = "current_results_review_report.md"
CONFIG_JSON = "current_results_review_config.json"


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


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


def _evidence_row(
    layer: str,
    artifact: Path,
    observation: str,
    interpretation: str,
    claim_boundary: str,
    count_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "layer": layer,
        "artifact": _rel(artifact),
        "observation": observation,
        "count_summary": json.dumps(count_summary or {}, sort_keys=True),
        "interpretation": interpretation,
        "claim_boundary": claim_boundary,
    }


def _risk_row(
    risk_id: str,
    severity: str,
    issue: str,
    evidence: str,
    consequence: str,
    next_check: str,
) -> dict[str, str]:
    return {
        "risk_id": risk_id,
        "severity": severity,
        "issue": issue,
        "evidence": evidence,
        "consequence": consequence,
        "next_check": next_check,
    }


def _current_review_status(row: pd.Series) -> str:
    state = str(row.get("methodology_v0_state", ""))
    margin_status = str(row.get("margin_validation_status", ""))
    next_action = str(row.get("next_action", ""))
    hygiene = str(row.get("field_hygiene_status", ""))
    relation = str(row.get("calibrated_relation", ""))

    if state == "partial_wall_gate_conservative":
        return "current_partial_wall_protocol_evidence"
    if margin_status == "validated_boundary_sensitive_hold":
        return "validated_boundary_sensitive_hold_no_promotion"
    if margin_status == "validated_support_loss_contrast":
        return "validated_support_loss_contrast_no_wall"
    if margin_status == "support_loss_contrast_mixed_hold":
        return "mixed_support_loss_contrast_no_wall"
    if state == "boundary_sensitive_margin_validation_candidate":
        return "margin_validation_candidate"
    if state == "support_loss_no_wall_contrast":
        return "support_loss_no_wall_contrast"
    if state == "relation_blocked_definition_evidence":
        return "relation_blocked_route_evidence"
    if state == "same_control_no_wall":
        return "same_control_no_wall"
    if "review_hygiene" in next_action or "hygiene_review_required" in hygiene:
        return "field34_hygiene_blocked"
    if relation == "ambiguous_support_local":
        return "ambiguous_relation_pending"
    if "control" in next_action:
        return "control_no_wall"
    return "not_currently_actionable"


def _pair_comment(row: pd.Series) -> str:
    status = str(row.get("current_review_status", ""))
    if status == "current_partial_wall_protocol_evidence":
        return "Retain as conservative partial-wall protocol evidence; no quality/cost claim."
    if status == "validated_boundary_sensitive_hold_no_promotion":
        return "Boundary-sensitive class survived held-out validation; do not promote wall claim."
    if status == "validated_support_loss_contrast_no_wall":
        return "Held-out validation repeated hard support-loss; keep as no-wall contrast."
    if status == "mixed_support_loss_contrast_no_wall":
        return "Hard support-loss remains only from prior schedules; keep no-wall but weaken contrast claim."
    if status == "margin_validation_candidate":
        return "Validate near-threshold W4 support loss before changing labels."
    if status == "support_loss_no_wall_contrast":
        return "Use as hard support-loss contrast in the margin validation panel."
    if status == "relation_blocked_route_evidence":
        return "Stable route evidence is blocked by ambiguous basin relation."
    if status == "field34_hygiene_blocked":
        return "Review field34 small/tiny support hygiene before route-gate execution."
    if status == "ambiguous_relation_pending":
        return "Resolve relation definition before wall promotion."
    if status == "same_control_no_wall":
        return "Same-control row; no wall promotion."
    if status == "control_no_wall":
        return "Control row; no wall promotion."
    return "No current route/wall action without a sharper mechanism question."


def _relation_blocker(row: pd.Series) -> str:
    relation = str(row.get("calibrated_relation", ""))
    if relation == "ambiguous_support_local":
        return "ambiguous_relation_blocks_wall_promotion"
    if relation in {"same_support_local", "same_endpoint_identity"}:
        return "control_relation_blocks_wall_promotion"
    if relation == "distinct_support_local":
        return "relation_not_blocking"
    return "relation_unknown"


def _hygiene_blocker(row: pd.Series) -> str:
    hygiene = str(row.get("field_hygiene_status", ""))
    if "hygiene_review_required" in hygiene:
        return "field34_hygiene_review_required"
    return "hygiene_not_blocking"


def _route_gate_group(row: pd.Series) -> str:
    gate = str(row.get("wall_claim_gate_status", ""))
    if gate == "passes_schedule_invariance_distinct_partial_wall_evidence":
        return "partial_wall_gate"
    if gate == "fails_schedule_invariance_no_supported_wall_claim":
        return "route_order_sensitive_no_wall"
    if gate == "stable_route_evidence_basin_relation_ambiguous_no_supported_wall_claim":
        return "stable_route_relation_blocked"
    if gate == "stable_control_no_wall_claim":
        return "stable_control_no_wall"
    if gate in {"", "nan", "None"}:
        return "not_run"
    return gate


def _pair_state() -> pd.DataFrame:
    coverage = _read_csv(LATEST_COVERAGE_DIR / "wall_panel_context_coverage_rows.csv")
    gate = _read_csv(COMBINED_GATE_DIR / "uniform_route_schedule_claim_panel_summary.csv")
    margin = _read_csv(MARGIN_DIR / "polish_margin_pair_gate_rows.csv")
    methodology = _read_csv(METHODOLOGY_DIR / "methodology_v0_route_gate_decision_rows.csv")
    margin_validation = _read_csv(MARGIN_VALIDATION_REVIEW_DIR / "margin_validation_pair_results.csv")
    taxonomy = _read_csv(RELATION_TAXONOMY_DIR / "basin_relation_taxonomy_rows.csv")

    coverage_cols = [
        "panel_pair_id",
        "field",
        "case_id",
        "panel_role",
        "calibrated_relation",
        "field_hygiene_status",
        "runner_context_status",
        "runner_preflight_status",
        "next_action",
    ]
    gate_cols = [
        "panel_pair_id",
        "support_distance_max",
        "route_order_sensitivity_status",
        "wall_claim_gate_status",
        "route_labels",
    ]
    margin_cols = [
        "panel_pair_id",
        "polish_margin_bands",
        "post_target_support_margin_max",
        "margin_gate_status",
    ]
    method_cols = [
        "panel_pair_id",
        "methodology_v0_state",
        "validation_role",
        "include_in_margin_validation_panel",
    ]
    taxonomy_cols = [
        "panel_pair_id",
        "relation_taxonomy_v0_1",
        "wall_promotion_status",
        "taxonomy_next_action",
    ]
    validation_cols = [
        "panel_pair_id",
        "validation_status",
        "heldout_margin_bands",
        "combined_margin_bands",
        "validation_note",
    ]
    state = coverage[coverage_cols].merge(gate[gate_cols], on="panel_pair_id", how="left")
    state = state.merge(margin[margin_cols], on="panel_pair_id", how="left")
    state = state.merge(methodology[method_cols], on="panel_pair_id", how="left")
    state = state.merge(
        margin_validation[validation_cols].rename(
            columns={
                "validation_status": "margin_validation_status",
                "validation_note": "margin_validation_note",
            }
        ),
        on="panel_pair_id",
        how="left",
    )
    state = state.merge(taxonomy[taxonomy_cols], on="panel_pair_id", how="left")
    state["relation_blocker_status"] = state.apply(_relation_blocker, axis=1)
    state["hygiene_blocker_status"] = state.apply(_hygiene_blocker, axis=1)
    state["route_gate_group"] = state.apply(_route_gate_group, axis=1)
    state["current_review_status"] = state.apply(_current_review_status, axis=1)
    state["review_comment"] = state.apply(_pair_comment, axis=1)
    return state.sort_values(
        ["field", "current_review_status", "panel_pair_id"], na_position="last"
    ).reset_index(drop=True)


def _evidence_ledger(pair_state: pd.DataFrame) -> pd.DataFrame:
    phase1 = _read_json(PHASE1_INDEX_DIR / "basin_cartography_summary.json")
    phase1_review = _read_json(PHASE1_REVIEW_DIR / "phase1_review_summary.json")
    calibration = _read_json(CALIBRATION_DIR / "basin_definition_calibration_summary.json")
    route_join = _read_json(ROUTE_JOIN_DIR / "wall_evidence_join_summary.json")
    direct_audit = _read_json(DIRECT_AUDIT_DIR / "direct_pair_route_audit_summary.json")
    panel = _read_json(PANEL_DIR / "wall_protocol_panel_summary.json")
    combined = _read_json(COMBINED_GATE_DIR / "uniform_route_schedule_claim_panel_summary.json")
    coverage = _read_json(LATEST_COVERAGE_DIR / "wall_panel_context_coverage_summary.json")
    mechanism = _read_json(MECHANISM_DIR / "clean_distinct_route_mechanism_review_summary.json")
    margin = _read_json(MARGIN_DIR / "polish_margin_gate_review_summary.json")
    methodology = _read_json(METHODOLOGY_DIR / "methodology_v0_margin_validation_summary.json")
    margin_validation = _read_json(
        MARGIN_VALIDATION_REVIEW_DIR / "margin_validation_panel_review_summary.json"
    )
    taxonomy = _read_json(RELATION_TAXONOMY_DIR / "basin_relation_taxonomy_summary.json")

    rows = [
        _evidence_row(
            "basin_inventory",
            PHASE1_INDEX_DIR,
            "Phase 1 basin-only index is internally consistent and excludes quality/cost columns.",
            "Good as a primitive basin inventory, not a final basin definition.",
            str(phase1.get("claim_boundary", "")),
            {
                "case_count": phase1["case_count"],
                "endpoint_rows": phase1["endpoint_rows"],
                "endpoint_identity_count_sum": phase1["endpoint_identity_count_sum"],
                "support_local_group_count_sum": phase1["support_local_group_count_sum"],
            },
        ),
        _evidence_row(
            "basin_definition",
            PHASE1_REVIEW_DIR,
            "Pair relations are dominated by ambiguous cases under the current support thresholds.",
            "The support-local rule is informative but not yet a final basin boundary.",
            "Definition review only.",
            {
                "total_pair_rows": phase1_review["total_pair_rows"],
                "same_pairs": phase1_review["same_pairs"],
                "ambiguous_pairs": phase1_review["ambiguous_pairs"],
                "distinct_pairs": phase1_review["distinct_pairs"],
            },
        ),
        _evidence_row(
            "basin_definition",
            CALIBRATION_DIR,
            "Calibration produces many distinct pairs but very few pairs with existing route traces.",
            "Wall evidence must be tested through a representative panel, not all distinct pairs.",
            str(calibration.get("claim_boundary", "")),
            {
                "candidate_pair_rows": calibration["candidate_pair_rows"],
                "distinct_identity_pair_rows": calibration["distinct_identity_pair_rows"],
                "route_join_candidate_pair_rows": calibration["route_join_candidate_pair_rows"],
            },
        ),
        _evidence_row(
            "wall_join",
            ROUTE_JOIN_DIR,
            "The narrow existing-artifact route-wall join has no supported wall claims.",
            "Existing route traces are insufficient for wall claims.",
            str(route_join.get("claim_boundary", "")),
            {
                "pair_count": route_join["pair_count"],
                "partial_pair_count": route_join["partial_pair_count"],
                "ambiguous_pair_count": route_join["ambiguous_pair_count"],
                "supported_pair_count": route_join["supported_pair_count"],
            },
        ),
        _evidence_row(
            "wall_join",
            DIRECT_AUDIT_DIR,
            "The c0-c2 direct audit finds self-endpoint routes only and no direct cross-route rows.",
            "c0-c2 remains a diagnostic control, not a wall-evidence anchor.",
            str(direct_audit.get("claim_boundary", "")),
            {
                "candidate_row_count": direct_audit["candidate_row_count"],
                "direct_cross_route_rows": direct_audit["direct_cross_route_rows"],
                "self_endpoint_route_rows": direct_audit["self_endpoint_route_rows"],
                "wall_claim_status": direct_audit["wall_claim_status"],
            },
        ),
        _evidence_row(
            "wall_protocol",
            PANEL_DIR,
            "The representative panel covers 23 pairs across distinct, ambiguous, control, and existing-route roles.",
            "This is the current testing surface.",
            str(panel.get("claim_boundary", "")),
            {
                "panel_pair_count": panel["panel_pair_count"],
                "panel_role_counts": panel["panel_role_counts"],
            },
        ),
        _evidence_row(
            "relation_taxonomy",
            RELATION_TAXONOMY_DIR,
            "Relation taxonomy v0.1 blocks boundary and ambiguous rows from wall promotion.",
            "Use it for relation status only; later route status is superseded by combined gates.",
            str(taxonomy.get("claim_boundary", "")),
            {
                "taxonomy_status_counts": taxonomy["taxonomy_status_counts"],
                "wall_promotion_status_counts": taxonomy["wall_promotion_status_counts"],
            },
        ),
        _evidence_row(
            "route_gate",
            COMBINED_GATE_DIR,
            "The latest 11-pair route-gate surface has 3 conservative partial-wall gates and 4 route-order-sensitive no-wall gates.",
            "Route-order stability is necessary but blocked by relation and margin gates where applicable.",
            str(combined.get("claim_boundary", "")),
            {
                "combined_gate_pair_count": combined["combined_gate_pair_count"],
                "route_order_sensitivity_status_counts": combined[
                    "route_order_sensitivity_status_counts"
                ],
                "wall_claim_gate_status_counts": combined["wall_claim_gate_status_counts"],
            },
        ),
        _evidence_row(
            "coverage",
            LATEST_COVERAGE_DIR,
            "The latest full-panel coverage has no immediately queued clean distinct not-run route-gate candidates.",
            "Open work is route-label interpretation freeze, relation refinement, and field34 hygiene.",
            str(coverage.get("claim_boundary", "")),
            {
                "panel_pair_count": coverage["panel_pair_count"],
                "runnable_not_run_distinct_pair_count": coverage[
                    "runnable_not_run_distinct_pair_count"
                ],
                "hygiene_review_distinct_pair_count": coverage[
                    "hygiene_review_distinct_pair_count"
                ],
                "ambiguous_refinement_pair_count": coverage["ambiguous_refinement_pair_count"],
                "next_action_counts": coverage["next_action_counts"],
            },
        ),
        _evidence_row(
            "route_mechanism",
            MECHANISM_DIR,
            "The clean distinct split is field26 stable target-polish versus field30 schedule-dependent post-polish support loss.",
            "The failure mode is W4 support assignment, not pre-polish target reach.",
            str(mechanism.get("claim_boundary", "")),
            {
                "pair_count": mechanism["pair_count"],
                "stable_partial_wall_pair_count": mechanism["stable_partial_wall_pair_count"],
                "schedule_sensitive_pair_count": mechanism["schedule_sensitive_pair_count"],
                "post_polish_support_assignment_loss_pair_count": mechanism[
                    "post_polish_support_assignment_loss_pair_count"
                ],
            },
        ),
        _evidence_row(
            "margin_gate",
            MARGIN_DIR,
            "W4 margin review separates near-threshold route holds from harder support-loss holds.",
            "Margin status is diagnostic until validated; it does not change wall gates.",
            str(margin.get("claim_boundary", "")),
            {
                "pair_count": margin["pair_count"],
                "margin_gate_status_counts": margin["margin_gate_status_counts"],
            },
        ),
        _evidence_row(
            "methodology_v0",
            METHODOLOGY_DIR,
            "Methodology v0 freezes 11 route-gate decisions and selects a 4-pair margin validation panel.",
            "This fixed the validation surface that was executed by the held-out margin review.",
            str(methodology.get("claim_boundary", "")),
            {
                "decision_pair_count": methodology["decision_pair_count"],
                "validation_panel_pair_count": methodology["validation_panel_pair_count"],
                "methodology_v0_state_counts": methodology["methodology_v0_state_counts"],
            },
        ),
        _evidence_row(
            "margin_validation",
            MARGIN_VALIDATION_REVIEW_DIR,
            "Held-out `target_label_desc` review validates both boundary-sensitive holds, validates field34 hard-loss contrast, and leaves field30 c6-c10 as a mixed contrast.",
            "Boundary-sensitive can become an uncertainty route class; support-loss contrast must stay split into strong versus mixed examples.",
            str(margin_validation.get("claim_boundary", "")),
            {
                "schedule_row_count": margin_validation["schedule_row_count"],
                "heldout_schedule_row_count": margin_validation["heldout_schedule_row_count"],
                "validation_status_counts": margin_validation["validation_status_counts"],
            },
        ),
    ]
    return pd.DataFrame(rows)


def _risk_ledger() -> pd.DataFrame:
    rows = [
        _risk_row(
            "R1",
            "high",
            "Support-local thresholds are still provisional.",
            "Phase 1 review has 509 ambiguous pairs out of 829; relation taxonomy has 5 boundary-review rows.",
            "A weak support threshold could create false basin boundaries.",
            "Keep relation-blocked and boundary-review rows out of wall promotion until the basin relation rule is fixed.",
        ),
        _risk_row(
            "R2",
            "high",
            "Partial-wall gates are protocol evidence, not full wall claims.",
            "Latest route gate has 3 partial-wall gates, all still bounded by schedule/relation/margin rules.",
            "Overstating them would turn diagnostic route evidence into an unsupported algorithm claim.",
            "Continue using `partial_wall_gate_conservative` wording.",
        ),
        _risk_row(
            "R3",
            "high",
            "Route-gate paths are constructed direct/support-closure traces, not a validated search operator.",
            "Uniform runner outputs W1-W6 diagnostics and explicitly forbids directed-search claims.",
            "Route success cannot yet imply a practical basin-tunneling algorithm.",
            "Do not run quality/cost or operator claims before wall/route gates are fixed.",
        ),
        _risk_row(
            "R4",
            "medium",
            "Field34 still has small/tiny-support hygiene blockers.",
            "Latest coverage sends 4 distinct field34 rows to hygiene review before route gates.",
            "Field34 evidence can inflate or destabilize basin counts.",
            "Review field34 hygiene separately before using those pairs as general evidence.",
        ),
        _risk_row(
            "R5",
            "medium",
            "Relation taxonomy route-status fields are chronologically stale for newly run clean-distinct pairs.",
            "Taxonomy v0.1 was prepared before the clean-distinct route gate; use latest combined gate for route status.",
            "Mixing taxonomy route status with current gate status can produce contradictory conclusions.",
            "Treat taxonomy as relation-status source and Methodology v0 as route-gate source.",
        ),
        _risk_row(
            "R6",
            "medium",
            "W4 margin validation is executed but still cannot promote wall claims.",
            "Boundary-sensitive holds survived held-out validation, but support-loss contrasts split into strong and mixed examples.",
            "A margin class can become a route-label uncertainty class, not a wall-promotion rule.",
            "Freeze the route-label interpretation before any broader route batch.",
        ),
        _risk_row(
            "R7",
            "medium",
            "The evidence is concentrated in a small current route-gate surface.",
            "Current Methodology v0 decision surface has 11 pairs and 3 partial-wall gates.",
            "General claims about large/dense graphs or broad basin structure remain unsupported.",
            "After blockers are resolved, expand by mechanism question, not by broad sweeping.",
        ),
        _risk_row(
            "R8",
            "low",
            "Objective-debt magnitudes are not normalized across cases.",
            "Schedule rows show objective debts ranging from single digits to thousands.",
            "Comparing wall difficulty across fields from raw debt alone would be misleading.",
            "Normalize or keep objective debt as within-pair route evidence only.",
        ),
    ]
    return pd.DataFrame(rows)


def _write_report(
    path: Path,
    summary: dict[str, Any],
    pair_state: pd.DataFrame,
    evidence: pd.DataFrame,
    risks: pd.DataFrame,
) -> None:
    lines = [
        "# Leiden Basin Current Results Review",
        "",
        "Status: current result surface audited",
        "Date: 2026-05-29",
        "",
        "This review reconciles current Track C basin, relation, route-gate, W4 polish-margin, Methodology v0, and held-out margin-validation artifacts. It does not rank basin quality or relax wall promotion.",
        "",
        "## Bottom Line",
        "",
        "- Current evidence supports a basin/wall cartography protocol, not a finished basin-tunneling algorithm.",
        "- The strongest positive result is 3 conservative distinct partial-wall route gates.",
        "- Held-out margin validation supports the boundary-sensitive uncertainty class, but not a wall-promotion rule.",
        "- The next method step is to freeze route-label interpretation before any broader route batch.",
        "",
        "## Pair State Counts",
        "",
        "| current_review_status | pairs |",
        "| --- | ---: |",
    ]
    for status, count in sorted(summary["current_review_status_counts"].items()):
        lines.append(f"| {status} | {count} |")
    lines.extend(
        [
            "",
            "## Blocker Counts",
            "",
            "| blocker_type | status | pairs |",
            "| --- | --- | ---: |",
        ]
    )
    for status, count in sorted(summary["relation_blocker_status_counts"].items()):
        lines.append(f"| relation | {status} | {count} |")
    for status, count in sorted(summary["hygiene_blocker_status_counts"].items()):
        lines.append(f"| hygiene | {status} | {count} |")
    for status, count in sorted(summary["route_gate_group_counts"].items()):
        lines.append(f"| route_gate | {status} | {count} |")
    lines.extend(
        [
            "",
            "## Evidence Ledger",
            "",
            "| layer | observation | interpretation |",
            "| --- | --- | --- |",
        ]
    )
    for _, row in evidence.iterrows():
        lines.append(
            f"| {row['layer']} | {row['observation']} | {row['interpretation']} |"
        )
    lines.extend(
        [
            "",
            "## Risk Ledger",
            "",
            "| risk | severity | issue | next_check |",
            "| --- | --- | --- | --- |",
        ]
    )
    for _, row in risks.iterrows():
        lines.append(
            f"| {row['risk_id']} | {row['severity']} | {row['issue']} | {row['next_check']} |"
        )
    lines.extend(
        [
            "",
            "## Current Pair Surface",
            "",
            "| pair_id | relation | hygiene | gate | methodology_v0 | margin_validation | current_status |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for _, row in pair_state.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["panel_pair_id"]),
                    str(row["calibrated_relation"]),
                    str(row["hygiene_blocker_status"]),
                    str(row.get("wall_claim_gate_status", "")),
                    str(row.get("methodology_v0_state", "")),
                    str(row.get("margin_validation_status", "")),
                    str(row["current_review_status"]),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_state = _pair_state()
    evidence = _evidence_ledger(pair_state)
    risks = _risk_ledger()

    _write_csv(pair_state, output_dir / PAIR_STATE_CSV)
    _write_csv(evidence, output_dir / EVIDENCE_LEDGER_CSV)
    _write_csv(risks, output_dir / RISK_LEDGER_CSV)

    summary = {
        "status": "current_results_review_prepared",
        "date": "2026-05-29",
        "script": _rel(Path(__file__)),
        "output_dir": _rel(output_dir),
        "pair_count": int(len(pair_state)),
        "current_review_status_counts": _count(pair_state, "current_review_status"),
        "relation_blocker_status_counts": _count(pair_state, "relation_blocker_status"),
        "hygiene_blocker_status_counts": _count(pair_state, "hygiene_blocker_status"),
        "route_gate_group_counts": _count(pair_state, "route_gate_group"),
        "evidence_ledger_rows": int(len(evidence)),
        "risk_ledger_rows": int(len(risks)),
        "decision": (
            "Freeze the route-label interpretation after held-out margin "
            "validation; do not broaden route execution or change wall promotion."
        ),
        "claim_boundary": (
            "Review artifact only; no basin-quality claim, cost claim, "
            "directed-search claim, or wall-promotion change."
        ),
        "paths": {
            "pair_state": _rel(output_dir / PAIR_STATE_CSV),
            "evidence_ledger": _rel(output_dir / EVIDENCE_LEDGER_CSV),
            "risk_ledger": _rel(output_dir / RISK_LEDGER_CSV),
            "summary": _rel(output_dir / SUMMARY_JSON),
            "report": _rel(output_dir / REPORT_MD),
        },
    }
    (output_dir / SUMMARY_JSON).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output_dir / CONFIG_JSON).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_report(output_dir / REPORT_MD, summary, pair_state, evidence, risks)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
