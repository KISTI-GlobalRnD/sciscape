#!/usr/bin/env python3
"""Bridge route-state morphology evidence to basin-state assignment blockers.

This read-only audit consumes the current first-pass route-state morphology
taxonomy and asks a narrower question: which parts are already route morphology,
and which parts are still missing before any basin identity, wall, or pathway
label can be promoted? It intentionally does not execute Leiden, rerun routes,
expand candidates, or select a demo.
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


DEFAULT_TAXONOMY_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_route_state_morphology_taxonomy_gamma1e5_20260606"
)
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_basin_state_route_morphology_bridge_gamma1e5_20260606"
)

TAXONOMY_PAIR_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_route_state_morphology_taxonomy_pair_rows.csv"
)
TAXONOMY_CLASS_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_route_state_morphology_taxonomy_class_rows.csv"
)
TAXONOMY_GATE_MATRIX_CSV = (
    "nanoclustering_g4_8_first_pass_route_state_morphology_taxonomy_gate_matrix.csv"
)
TAXONOMY_SUMMARY_JSON = (
    "nanoclustering_g4_8_first_pass_route_state_morphology_taxonomy_summary.json"
)

PAIR_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_basin_state_route_morphology_bridge_pair_rows.csv"
)
CLASS_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_basin_state_route_morphology_bridge_class_rows.csv"
)
REQUIREMENT_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_basin_state_route_morphology_bridge_requirement_rows.csv"
)
DECISION_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_basin_state_route_morphology_bridge_decision_rows.csv"
)
GATE_MATRIX_CSV = (
    "nanoclustering_g4_8_first_pass_basin_state_route_morphology_bridge_gate_matrix.csv"
)
SUMMARY_JSON = (
    "nanoclustering_g4_8_first_pass_basin_state_route_morphology_bridge_summary.json"
)
CONFIG_JSON = (
    "nanoclustering_g4_8_first_pass_basin_state_route_morphology_bridge_config.json"
)
REPORT_MD = (
    "nanoclustering_g4_8_first_pass_basin_state_route_morphology_bridge_report.md"
)

EXPECTED_ROUTE_SCOREABLE_PAIR_IDS = (
    "local_pair_016",
    "local_pair_014",
    "local_pair_009",
    "local_pair_012",
    "local_pair_020",
    "local_pair_005",
)

POSITIVE_CLASS = "stable_finite_single_side_plateau_reference"
NEGATIVE_CLASSES = {
    "abrupt_source_target_switch_negative",
    "fragmented_or_point_single_side_negative",
}
BOUNDARY_CLASS = "boundary_or_endpoint_surface_control"

RUN_STATUS = "audited_nanoclustering_g4_8_first_pass_basin_state_route_morphology_bridge"
ROUTE_EXECUTION_STATUS = "not_executed_read_only_basin_state_route_morphology_bridge"
WALL_PROMOTION_STATUS = "not_promoted_missing_accepted_basin_pair"
METHOD_STATUS = "basin_state_assignment_blocker_audit_not_method"
CLAIM_BOUNDARY = (
    "NanoClustering G4.8 first-pass basin-state x route-morphology bridge audit "
    "only; reads the route-state morphology taxonomy. It does not execute Leiden, "
    "promote basin identity, promote walls or pathway labels, replay full "
    "NanoClustering, evaluate quality/cost value, or claim method success."
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


def _as_int(value: Any, default: int = 0) -> int:
    if pd.isna(value):
        return default
    return int(float(value))


def _join_ids(values: pd.Series) -> str:
    return ";".join(str(value) for value in values)


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


def _markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 80) -> str:
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


def _bridge_class(morphology_class: str) -> str:
    if morphology_class == POSITIVE_CLASS:
        return "positive_morphology_basin_state_proxy_blocked"
    if morphology_class in NEGATIVE_CLASSES:
        return "negative_guard_basin_state_proxy_blocked"
    if morphology_class == BOUNDARY_CLASS:
        return "boundary_control_missing_source_basin"
    return "unsupported_route_morphology_bridge_class"


def _route_morphology_evidence_status(morphology_class: str) -> str:
    if morphology_class == POSITIVE_CLASS:
        return "positive_route_morphology_reference"
    if morphology_class in NEGATIVE_CLASSES:
        return "negative_route_morphology_guard"
    if morphology_class == BOUNDARY_CLASS:
        return "boundary_control_morphology"
    return "unsupported_route_morphology"


def _next_action(morphology_class: str, source_present: bool, target_present: bool) -> str:
    if morphology_class == POSITIVE_CLASS:
        return (
            "Attach source and target basin-state assignment evidence to the positive "
            "route-morphology reference before any demo, wall, or pathway label."
        )
    if morphology_class in NEGATIVE_CLASSES:
        return (
            "Keep as route-morphology guards; do not promote to route labels unless "
            "accepted source and target basin assignments plus wall evidence are added."
        )
    if not source_present:
        return (
            "Keep as a boundary control because source-family basin-state context is "
            "missing."
        )
    if not target_present:
        return "Keep as an endpoint control because target basin-state context is missing."
    return "Keep diagnostic until basin-state assignment rules are declared."


def _build_pair_rows(taxonomy_pair_rows: pd.DataFrame) -> pd.DataFrame:
    route_rows = taxonomy_pair_rows[
        taxonomy_pair_rows["current_route_state_readout_present"].map(_as_bool)
    ].copy()
    order = {pair_id: index for index, pair_id in enumerate(EXPECTED_ROUTE_SCOREABLE_PAIR_IDS)}
    route_rows = (
        route_rows.assign(
            _sort_key=lambda frame: frame["local_pair_id"].map(order).fillna(999)
        )
        .sort_values(["_sort_key", "local_pair_id"], kind="mergesort")
        .drop(columns=["_sort_key"])
    )

    rows: list[dict[str, Any]] = []
    for _, row in route_rows.iterrows():
        pair_id = str(row["local_pair_id"])
        morphology_class = str(row["route_state_morphology_class"])
        source_count = _as_int(row.get("source_family_start_count"))
        target_count = _as_int(row.get("final_target_like_count"))
        source_present = source_count > 0
        target_present = target_count > 0
        source_assignment_status = (
            "proxy_only_no_endpoint_identity" if source_present else "missing_source_basin"
        )
        target_assignment_status = (
            "proxy_only_no_endpoint_identity" if target_present else "missing_target_basin"
        )
        route_label_blocker = (
            "missing_source_basin_assignment_and_wall_evidence"
            if not source_present
            else (
                "missing_target_basin_assignment_and_wall_evidence"
                if not target_present
                else "missing_accepted_source_target_basin_assignment_and_wall_evidence"
            )
        )
        rows.append(
            {
                "local_pair_id": pair_id,
                "route_state_morphology_class": morphology_class,
                "route_state_morphology_role": row["route_state_morphology_role"],
                "basin_state_route_bridge_class": _bridge_class(morphology_class),
                "route_state_sequence": row["route_state_sequence"],
                "route_count": _as_int(row.get("route_count")),
                "source_family_start_count": source_count,
                "finite_single_side_band_count": _as_int(
                    row.get("finite_single_side_band_count")
                ),
                "final_target_like_count": target_count,
                "source_state_proxy": (
                    "source_family_proxy"
                    if source_present
                    else "source_absent_boundary_surface"
                ),
                "target_state_proxy": "target_like_proxy" if target_present else "target_absent",
                "source_basin_assignment_status": source_assignment_status,
                "target_basin_assignment_status": target_assignment_status,
                "source_target_relation_status": "proxy_relation_only",
                "basin_identity_claim_status": "blocked_no_endpoint_identity_evidence",
                "accepted_source_basin_assignment": False,
                "accepted_target_basin_assignment": False,
                "accepted_source_target_basin_pair": False,
                "route_morphology_evidence_status": _route_morphology_evidence_status(
                    morphology_class
                ),
                "wall_evidence_status": "unknown_not_tested_no_accepted_basin_pair",
                "route_label_v0": "unknown",
                "route_label_confidence": "not_supported",
                "route_label_blocker": route_label_blocker,
                "demo_readiness_status": "blocked_until_basin_state_assignment",
                "recommended_next_action": _next_action(
                    morphology_class, source_present, target_present
                ),
                "claim_status": "route_morphology_only_basin_state_blocked",
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
    grouped = pair_rows.groupby("basin_state_route_bridge_class", dropna=False)
    for bridge_class, group in grouped:
        rows.append(
            {
                "basin_state_route_bridge_class": bridge_class,
                "pair_count": int(len(group)),
                "local_pair_ids": _join_ids(group["local_pair_id"]),
                "route_morphology_classes": ";".join(
                    sorted(set(group["route_state_morphology_class"].astype(str)))
                ),
                "accepted_source_target_basin_pair_count": int(
                    group["accepted_source_target_basin_pair"].map(_as_bool).sum()
                ),
                "unknown_route_label_count": int(
                    group["route_label_v0"].astype(str).eq("unknown").sum()
                ),
                "wall_evidence_unknown_count": int(
                    group["wall_evidence_status"]
                    .astype(str)
                    .eq("unknown_not_tested_no_accepted_basin_pair")
                    .sum()
                ),
                "class_interpretation": _class_interpretation(str(bridge_class)),
                "claim_boundary": CLAIM_BOUNDARY,
                "run_status": RUN_STATUS,
            }
        )
    order = {
        "positive_morphology_basin_state_proxy_blocked": 0,
        "negative_guard_basin_state_proxy_blocked": 1,
        "boundary_control_missing_source_basin": 2,
        "unsupported_route_morphology_bridge_class": 3,
    }
    return (
        pd.DataFrame(rows)
        .assign(_sort_key=lambda frame: frame["basin_state_route_bridge_class"].map(order).fillna(99))
        .sort_values(["_sort_key", "basin_state_route_bridge_class"], kind="mergesort")
        .drop(columns=["_sort_key"])
    )


def _class_interpretation(bridge_class: str) -> str:
    if bridge_class == "positive_morphology_basin_state_proxy_blocked":
        return (
            "The positive 016-like route morphology exists, but source/target basin "
            "identity assignments are still proxy-only."
        )
    if bridge_class == "negative_guard_basin_state_proxy_blocked":
        return (
            "Negative route morphologies are useful guards, but they also lack "
            "accepted source/target basin assignments and wall evidence."
        )
    if bridge_class == "boundary_control_missing_source_basin":
        return (
            "Boundary/control morphology lacks source-family basin context and must "
            "stay outside pathway promotion."
        )
    return "Unsupported bridge class; keep diagnostic only."


def _build_requirement_rows(pair_rows: pd.DataFrame) -> pd.DataFrame:
    pair_count = int(len(pair_rows))
    source_proxy_count = int(
        pair_rows["source_basin_assignment_status"]
        .astype(str)
        .eq("proxy_only_no_endpoint_identity")
        .sum()
    )
    target_proxy_count = int(
        pair_rows["target_basin_assignment_status"]
        .astype(str)
        .eq("proxy_only_no_endpoint_identity")
        .sum()
    )
    accepted_pair_count = int(
        pair_rows["accepted_source_target_basin_pair"].map(_as_bool).sum()
    )
    unknown_wall_count = int(
        pair_rows["wall_evidence_status"]
        .astype(str)
        .eq("unknown_not_tested_no_accepted_basin_pair")
        .sum()
    )
    unknown_label_count = int(pair_rows["route_label_v0"].astype(str).eq("unknown").sum())
    rows = [
        {
            "requirement_id": "R1_accepted_source_basin_candidate",
            "requirement_question": "Does every scoreable route have an accepted source basin candidate?",
            "observed": f"{source_proxy_count} proxy-only sources among {pair_count} scoreable routes",
            "requirement_status": "not_satisfied",
            "blocker": "source evidence is proxy-only or missing; no endpoint identity/support-local acceptance exists",
        },
        {
            "requirement_id": "R2_accepted_target_basin_candidate",
            "requirement_question": "Does every scoreable route have an accepted target basin candidate?",
            "observed": f"{target_proxy_count} proxy-only targets among {pair_count} scoreable routes",
            "requirement_status": "not_satisfied",
            "blocker": "target evidence is proxy-only or missing; no endpoint identity/support-local acceptance exists",
        },
        {
            "requirement_id": "R3_accepted_source_target_basin_pair",
            "requirement_question": "Can source and target be treated as an accepted basin pair?",
            "observed": f"{accepted_pair_count} accepted source-target basin pairs",
            "requirement_status": "not_satisfied",
            "blocker": "accepted source and target basin assignments are both absent",
        },
        {
            "requirement_id": "R4_wall_evidence",
            "requirement_question": "Is wall evidence tested between accepted basin states?",
            "observed": f"{unknown_wall_count} rows have unknown wall evidence",
            "requirement_status": "not_satisfied",
            "blocker": "wall evidence cannot be tested until an accepted basin pair exists",
        },
        {
            "requirement_id": "R5_route_morphology",
            "requirement_question": "Is route-state morphology available for the current scoreable routes?",
            "observed": f"{pair_count} route-scoreable rows have current morphology classes",
            "requirement_status": "satisfied_for_morphology_only",
            "blocker": "does not by itself define basin identity or walls",
        },
        {
            "requirement_id": "R6_route_label_promotion",
            "requirement_question": "Can any route label move from unknown to accepted?",
            "observed": f"{unknown_label_count} unknown route labels among {pair_count} scoreable routes",
            "requirement_status": "blocked",
            "blocker": "route labels require accepted source/target basin assignment plus wall evidence",
        },
    ]
    for row in rows:
        row["claim_boundary"] = CLAIM_BOUNDARY
        row["run_status"] = RUN_STATUS
    return pd.DataFrame(rows)


def _build_decision_rows(pair_rows: pd.DataFrame, requirement_rows: pd.DataFrame) -> pd.DataFrame:
    scoreable_pair_ids = list(pair_rows["local_pair_id"].astype(str))
    decisions = [
        {
            "decision_id": "D1_taxonomy_is_not_basin_definition",
            "decision": (
                "The route-state morphology taxonomy is a route evidence layer, not "
                "a basin identity definition."
            ),
            "evidence": "Current scoreable rows have morphology classes but no accepted basin assignments.",
            "decision_status": "accepted_guardrail",
        },
        {
            "decision_id": "D2_no_accepted_source_target_assignment",
            "decision": (
                "No current route-scoreable pair has accepted source and target basin "
                "assignments."
            ),
            "evidence": f"route_scoreable_pair_ids={';'.join(scoreable_pair_ids)}",
            "decision_status": "accepted_blocker",
        },
        {
            "decision_id": "D3_route_labels_remain_unknown",
            "decision": "All route labels remain unknown and unsupported.",
            "evidence": "route_label_v0 is unknown and confidence is not_supported for every row.",
            "decision_status": "accepted_blocker",
        },
        {
            "decision_id": "D4_next_gate_basin_state_assignment_surface",
            "decision": (
                "The next work unit is a basin-state assignment surface before demo "
                "selection, route reruns, candidate expansion, or pathway promotion."
            ),
            "evidence": ";".join(requirement_rows["requirement_id"].astype(str)),
            "decision_status": "next_gate",
        },
    ]
    for row in decisions:
        row["claim_boundary"] = CLAIM_BOUNDARY
        row["run_status"] = RUN_STATUS
    return pd.DataFrame(decisions)


def _build_gate_matrix(
    *,
    taxonomy_pair_rows: pd.DataFrame,
    taxonomy_class_rows: pd.DataFrame,
    taxonomy_gate_rows: pd.DataFrame,
    taxonomy_summary: dict[str, Any],
    bridge_pair_rows: pd.DataFrame,
    requirement_rows: pd.DataFrame,
    recommended_next_gate: str,
) -> pd.DataFrame:
    scoreable_pair_ids = list(bridge_pair_rows["local_pair_id"].astype(str))
    positive_ids = list(
        bridge_pair_rows.loc[
            bridge_pair_rows["route_state_morphology_class"].astype(str).eq(POSITIVE_CLASS),
            "local_pair_id",
        ].astype(str)
    )
    negative_ids = list(
        bridge_pair_rows.loc[
            bridge_pair_rows["route_state_morphology_class"].astype(str).isin(NEGATIVE_CLASSES),
            "local_pair_id",
        ].astype(str)
    )
    boundary_ids = list(
        bridge_pair_rows.loc[
            bridge_pair_rows["route_state_morphology_class"].astype(str).eq(BOUNDARY_CLASS),
            "local_pair_id",
        ].astype(str)
    )
    accepted_pair_count = int(
        bridge_pair_rows["accepted_source_target_basin_pair"].map(_as_bool).sum()
    )
    unknown_label_count = int(
        bridge_pair_rows["route_label_v0"].astype(str).eq("unknown").sum()
    )
    requirement_statuses = dict(
        zip(
            requirement_rows["requirement_id"].astype(str),
            requirement_rows["requirement_status"].astype(str),
            strict=False,
        )
    )
    gates = [
        _gate_row(
            "G1_taxonomy_sources_readable",
            "Were taxonomy pair/class/gate/summary artifacts readable?",
            {
                "taxonomy_pair_rows": int(len(taxonomy_pair_rows)),
                "taxonomy_class_rows": int(len(taxonomy_class_rows)),
                "taxonomy_gate_rows": int(len(taxonomy_gate_rows)),
                "taxonomy_failed_gates": taxonomy_summary.get("failed_gates"),
            },
            "taxonomy has rows and no failed gates",
            bool(len(taxonomy_pair_rows))
            and bool(len(taxonomy_class_rows))
            and bool(len(taxonomy_gate_rows))
            and not taxonomy_summary.get("failed_gates"),
        ),
        _gate_row(
            "G2_route_scoreable_surface_preserved",
            "Does the bridge cover exactly the current six route-scoreable pairs?",
            scoreable_pair_ids,
            "exactly 016,014,009,012,020,005",
            tuple(scoreable_pair_ids) == EXPECTED_ROUTE_SCOREABLE_PAIR_IDS,
        ),
        _gate_row(
            "G3_route_morphology_families_preserved",
            "Are positive, negative, and boundary morphology families preserved?",
            {
                "positive_ids": positive_ids,
                "negative_ids": negative_ids,
                "boundary_ids": boundary_ids,
            },
            "016 positive, 009/012/014/020 negative guards, 005 boundary control",
            positive_ids == ["local_pair_016"]
            and set(negative_ids) == {"local_pair_009", "local_pair_012", "local_pair_014", "local_pair_020"}
            and boundary_ids == ["local_pair_005"],
        ),
        _gate_row(
            "G4_basin_state_assignment_not_promoted",
            "Are basin-state assignments kept proxy-only or missing rather than accepted?",
            {
                "accepted_source_target_basin_pair_count": accepted_pair_count,
                "source_statuses": sorted(
                    set(bridge_pair_rows["source_basin_assignment_status"].astype(str))
                ),
                "target_statuses": sorted(
                    set(bridge_pair_rows["target_basin_assignment_status"].astype(str))
                ),
            },
            "zero accepted source-target basin pairs",
            accepted_pair_count == 0,
        ),
        _gate_row(
            "G5_route_label_promotion_blocked",
            "Are all route labels blocked until accepted basin-pair and wall evidence exist?",
            {
                "unknown_route_label_count": unknown_label_count,
                "pair_count": int(len(bridge_pair_rows)),
                "route_label_confidences": sorted(
                    set(bridge_pair_rows["route_label_confidence"].astype(str))
                ),
            },
            "every route_label_v0 is unknown and confidence is not_supported",
            unknown_label_count == len(bridge_pair_rows)
            and set(bridge_pair_rows["route_label_confidence"].astype(str))
            == {"not_supported"},
        ),
        _gate_row(
            "G6_next_direction_is_basin_state_assignment",
            "Is the next direction a basin-state assignment surface rather than rerun/demo/promotion?",
            {
                "recommended_next_gate": recommended_next_gate,
                "requirement_statuses": requirement_statuses,
            },
            "R1/R2/R3/R4/R6 not satisfied or blocked; R5 morphology only",
            requirement_statuses.get("R5_route_morphology")
            == "satisfied_for_morphology_only"
            and requirement_statuses.get("R6_route_label_promotion") == "blocked"
            and "basin-state assignment surface" in recommended_next_gate,
        ),
        _gate_row(
            "G7_claim_boundaries_closed",
            "Are wall, method, quality/cost, and full-replay claims closed?",
            CLAIM_BOUNDARY,
            "read-only bridge audit with no promotions",
            all(
                bridge_pair_rows[column].astype(str).eq(expected).all()
                for column, expected in {
                    "route_execution_status": ROUTE_EXECUTION_STATUS,
                    "wall_promotion_status": WALL_PROMOTION_STATUS,
                    "method_status": METHOD_STATUS,
                }.items()
            ),
        ),
    ]
    return pd.DataFrame(gates)


def _write_report(
    *,
    output_dir: Path,
    pair_rows: pd.DataFrame,
    class_rows: pd.DataFrame,
    requirement_rows: pd.DataFrame,
    decision_rows: pd.DataFrame,
    gate_matrix: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    report = f"""# NanoClustering G4.8 First-Pass Basin-State x Route-Morphology Bridge Audit

## Status

- Bridge status: `{summary["bridge_status"]}`
- Route-scoreable pairs: `{", ".join(summary["route_scoreable_pair_ids"])}`
- Basin-state ready pairs: `{", ".join(summary["basin_state_ready_pair_ids"]) or "none"}`
- Accepted route-label pairs: `{", ".join(summary["accepted_route_label_pair_ids"]) or "none"}`
- Recommended next gate: {summary["recommended_next_gate"]}

## Bridge Pair Rows

{_markdown_table(
    pair_rows,
    [
        "local_pair_id",
        "basin_state_route_bridge_class",
        "route_state_morphology_class",
        "source_basin_assignment_status",
        "target_basin_assignment_status",
        "wall_evidence_status",
        "route_label_v0",
        "route_label_blocker",
    ],
)}

## Bridge Classes

{_markdown_table(
    class_rows,
    [
        "basin_state_route_bridge_class",
        "pair_count",
        "local_pair_ids",
        "accepted_source_target_basin_pair_count",
        "unknown_route_label_count",
        "class_interpretation",
    ],
)}

## Requirements

{_markdown_table(
    requirement_rows,
    [
        "requirement_id",
        "requirement_status",
        "observed",
        "blocker",
    ],
)}

## Decisions

{_markdown_table(
    decision_rows,
    [
        "decision_id",
        "decision_status",
        "decision",
        "evidence",
    ],
)}

## Gates

{_markdown_table(
    gate_matrix,
    [
        "gate_id",
        "gate_status",
        "question",
        "minimum_or_rule",
    ],
)}

## Claim Boundary

{CLAIM_BOUNDARY}
"""
    (output_dir / REPORT_MD).write_text(report, encoding="utf-8")


def run(*, taxonomy_dir: Path, output_dir: Path) -> dict[str, Any]:
    taxonomy_pair_rows = _read_csv(taxonomy_dir / TAXONOMY_PAIR_ROWS_CSV)
    taxonomy_class_rows = _read_csv(taxonomy_dir / TAXONOMY_CLASS_ROWS_CSV)
    taxonomy_gate_rows = _read_csv(taxonomy_dir / TAXONOMY_GATE_MATRIX_CSV)
    taxonomy_summary = _read_json(taxonomy_dir / TAXONOMY_SUMMARY_JSON)

    output_dir.mkdir(parents=True, exist_ok=True)
    pair_rows = _build_pair_rows(taxonomy_pair_rows)
    class_rows = _build_class_rows(pair_rows)
    requirement_rows = _build_requirement_rows(pair_rows)
    decision_rows = _build_decision_rows(pair_rows, requirement_rows)
    recommended_next_gate = (
        "Materialize a basin-state assignment surface for the six route-scoreable "
        "pairs by attaching endpoint identity/support-local basin evidence; keep "
        "route labels unknown until accepted source and target basin assignments "
        "plus wall evidence exist."
    )
    gate_matrix = _build_gate_matrix(
        taxonomy_pair_rows=taxonomy_pair_rows,
        taxonomy_class_rows=taxonomy_class_rows,
        taxonomy_gate_rows=taxonomy_gate_rows,
        taxonomy_summary=taxonomy_summary,
        bridge_pair_rows=pair_rows,
        requirement_rows=requirement_rows,
        recommended_next_gate=recommended_next_gate,
    )

    failed_gates = list(
        gate_matrix.loc[gate_matrix["gate_status"].astype(str).eq("fail"), "gate_id"].astype(str)
    )
    basin_state_ready_pair_ids = list(
        pair_rows.loc[
            pair_rows["accepted_source_target_basin_pair"].map(_as_bool), "local_pair_id"
        ].astype(str)
    )
    accepted_route_label_pair_ids = list(
        pair_rows.loc[pair_rows["route_label_v0"].astype(str).ne("unknown"), "local_pair_id"]
        .astype(str)
    )
    summary = {
        "bridge_status": "route_morphology_available_basin_state_assignment_blocked",
        "taxonomy_source_dir": str(taxonomy_dir),
        "output_dir": str(output_dir),
        "route_scoreable_pair_ids": list(pair_rows["local_pair_id"].astype(str)),
        "bridge_class_counts": [
            {
                "basin_state_route_bridge_class": row[
                    "basin_state_route_bridge_class"
                ],
                "pair_count": int(row["pair_count"]),
            }
            for _, row in class_rows.iterrows()
        ],
        "basin_state_ready_pair_ids": basin_state_ready_pair_ids,
        "accepted_route_label_pair_ids": accepted_route_label_pair_ids,
        "demo_ready_pair_ids": [],
        "failed_gates": failed_gates,
        "recommended_next_gate": recommended_next_gate,
        "claim_boundary": CLAIM_BOUNDARY,
        "run_status": RUN_STATUS,
    }
    config = {
        "taxonomy_dir": str(taxonomy_dir),
        "output_dir": str(output_dir),
        "expected_route_scoreable_pair_ids": list(EXPECTED_ROUTE_SCOREABLE_PAIR_IDS),
        "route_execution_status": ROUTE_EXECUTION_STATUS,
        "wall_promotion_status": WALL_PROMOTION_STATUS,
        "method_status": METHOD_STATUS,
        "claim_boundary": CLAIM_BOUNDARY,
        "run_status": RUN_STATUS,
    }

    _write_csv(pair_rows, output_dir / PAIR_ROWS_CSV)
    _write_csv(class_rows, output_dir / CLASS_ROWS_CSV)
    _write_csv(requirement_rows, output_dir / REQUIREMENT_ROWS_CSV)
    _write_csv(decision_rows, output_dir / DECISION_ROWS_CSV)
    _write_csv(gate_matrix, output_dir / GATE_MATRIX_CSV)
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
        pair_rows=pair_rows,
        class_rows=class_rows,
        requirement_rows=requirement_rows,
        decision_rows=decision_rows,
        gate_matrix=gate_matrix,
        summary=summary,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Bridge current route-state morphology taxonomy to basin-state "
            "assignment blockers."
        )
    )
    parser.add_argument(
        "--taxonomy-dir",
        type=Path,
        default=DEFAULT_TAXONOMY_DIR,
        help="Directory containing route-state morphology taxonomy artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for bridge-audit artifacts.",
    )
    args = parser.parse_args()
    summary = run(taxonomy_dir=args.taxonomy_dir, output_dir=args.output_dir)
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
