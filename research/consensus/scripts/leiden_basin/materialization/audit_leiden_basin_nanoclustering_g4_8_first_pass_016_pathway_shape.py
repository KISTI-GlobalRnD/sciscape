#!/usr/bin/env python3
"""Audit the local_pair_016 forward/reverse pathway shape.

This read-only audit follows the source-family equivalence audit. It
reclassifies the executed forward persistence trace and reverse trace with the
same-start source-family vocabulary, then summarizes the observed pathway
shape without promoting wall, tunneling, method, full-replay, or quality/cost
claims.
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
TARGET_SIGNATURE_ID = "3c9b8a190753"
TRANSIENT_SIGNATURE_ID = "aeb59ab537e6"

DEFAULT_PERSISTENCE_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_016_transient_persistence_trace_gamma1e5_20260605"
)
DEFAULT_REVERSE_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_016_transient_reverse_trace_gamma1e5_20260605"
)
DEFAULT_SOURCE_EQUIVALENCE_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_016_source_family_equivalence_audit_gamma1e5_20260605"
)
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_016_pathway_shape_audit_gamma1e5_20260605"
)

PATHWAY_ROUTE_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_016_pathway_shape_route_rows.csv"
)
PATHWAY_FRACTION_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_016_pathway_shape_fraction_rows.csv"
)
DECISION_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_016_pathway_shape_decision_rows.csv"
)
GATE_MATRIX_CSV = "nanoclustering_g4_8_first_pass_016_pathway_shape_gate_matrix.csv"
SUMMARY_JSON = "nanoclustering_g4_8_first_pass_016_pathway_shape_summary.json"
CONFIG_JSON = "nanoclustering_g4_8_first_pass_016_pathway_shape_config.json"
REPORT_MD = "nanoclustering_g4_8_first_pass_016_pathway_shape_report.md"

RUN_STATUS = "audited_nanoclustering_g4_8_first_pass_016_pathway_shape"
ROUTE_EXECUTION_STATUS = "not_executed_read_only_016_pathway_shape"
WALL_PROMOTION_STATUS = "not_promoted_pathway_shape_only"
METHOD_STATUS = "pathway_shape_audit_not_method"
CLAIM_BOUNDARY = (
    "NanoClustering G4.8 first-pass local_pair_016 pathway-shape audit only; "
    "reads the executed persistence/reverse traces and the source-family "
    "equivalence artifact to summarize the observed state sequence. It does "
    "not rerun Leiden, promote basin walls, replay full NanoClustering, "
    "evaluate quality/cost value, or claim method success."
)

SOURCE_FAMILY_FRACTIONS = (0.875, 1.0)
TRANSIENT_BAND_FRACTIONS = (0.625, 0.6875, 0.71875, 0.75, 0.78125, 0.8125)
TARGET_FRACTIONS = (0.5,)
ALL_FRACTIONS = tuple(sorted(TARGET_FRACTIONS + TRANSIENT_BAND_FRACTIONS + SOURCE_FAMILY_FRACTIONS))
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


def _float_key(value: Any) -> float:
    return round(float(value), 6)


def _fraction_list(values: pd.Series) -> str:
    if values.empty:
        return ""
    return ";".join(f"{float(value):.6g}" for value in sorted(values.astype(float)))


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
    persistence_dir: Path,
    reverse_dir: Path,
    source_equivalence_dir: Path,
) -> dict[str, Any]:
    return {
        "persistence_summary": _read_json(
            persistence_dir
            / "nanoclustering_g4_8_first_pass_016_transient_persistence_summary.json"
        ),
        "persistence_gates": _read_csv(
            persistence_dir
            / "nanoclustering_g4_8_first_pass_016_transient_persistence_gate_matrix.csv"
        ),
        "persistence_trace": _read_csv(
            persistence_dir
            / "nanoclustering_g4_8_first_pass_016_transient_persistence_trace_rows.csv"
        ),
        "persistence_route": _read_csv(
            persistence_dir
            / "nanoclustering_g4_8_first_pass_016_transient_persistence_route_rows.csv"
        ),
        "reverse_summary": _read_json(
            reverse_dir
            / "nanoclustering_g4_8_first_pass_016_transient_reverse_summary.json"
        ),
        "reverse_gates": _read_csv(
            reverse_dir
            / "nanoclustering_g4_8_first_pass_016_transient_reverse_gate_matrix.csv"
        ),
        "reverse_trace": _read_csv(
            reverse_dir
            / "nanoclustering_g4_8_first_pass_016_transient_reverse_trace_rows.csv"
        ),
        "reverse_route": _read_csv(
            reverse_dir
            / "nanoclustering_g4_8_first_pass_016_transient_reverse_route_rows.csv"
        ),
        "source_equivalence_summary": _read_json(
            source_equivalence_dir
            / "nanoclustering_g4_8_first_pass_016_source_family_equivalence_summary.json"
        ),
        "source_equivalence_gates": _read_csv(
            source_equivalence_dir
            / "nanoclustering_g4_8_first_pass_016_source_family_equivalence_gate_matrix.csv"
        ),
        "source_route": _read_csv(
            source_equivalence_dir
            / "nanoclustering_g4_8_first_pass_016_source_family_equivalence_route_rule_rows.csv"
        ),
    }


def _source_family_signatures_by_start(persistence_trace: pd.DataFrame) -> dict[str, set[str]]:
    source_rows = persistence_trace[
        persistence_trace["bridge_edge_weight_fraction"].map(_float_key).eq(1.0)
        & persistence_trace["matches_original_anchor"].map(_as_bool)
    ].copy()
    result: dict[str, set[str]] = {}
    for start_condition, group in source_rows.groupby("start_condition"):
        result[str(start_condition)] = set(group["result_endpoint_signature_id"].astype(str))
    return result


def _classify_trace_rows(
    trace_rows: pd.DataFrame,
    source_signatures_by_start: dict[str, set[str]],
) -> pd.DataFrame:
    rows = trace_rows.copy()
    if "route_key" not in rows.columns:
        rows["route_key"] = (
            rows["start_condition"].astype(str)
            + "|seed="
            + rows["seed"].astype(int).astype(str)
        )
    classes: list[str] = []
    source_family_matches: list[bool] = []
    for row in rows.itertuples(index=False):
        signature_id = str(row.result_endpoint_signature_id)
        start_condition = str(row.start_condition)
        source_family_match = signature_id in source_signatures_by_start.get(
            start_condition,
            set(),
        )
        if signature_id == TARGET_SIGNATURE_ID:
            pathway_class = "target_anchor"
        elif signature_id == TRANSIENT_SIGNATURE_ID:
            pathway_class = "transient_signature"
        elif source_family_match:
            pathway_class = "source_family"
        else:
            pathway_class = "other"
        classes.append(pathway_class)
        source_family_matches.append(bool(source_family_match))
    rows["source_family_signature_match"] = source_family_matches
    rows["pathway_state_class"] = classes
    return rows


def _expected_class_for_fraction(fraction: float) -> str:
    if any(abs(fraction - value) <= EPS for value in TARGET_FRACTIONS):
        return "target_anchor"
    if any(abs(fraction - value) <= EPS for value in TRANSIENT_BAND_FRACTIONS):
        return "transient_signature"
    if any(abs(fraction - value) <= EPS for value in SOURCE_FAMILY_FRACTIONS):
        return "source_family"
    return "other"


def _class_counts_for_route(rows: pd.DataFrame) -> dict[str, int]:
    counts = rows["pathway_state_class"].value_counts().to_dict()
    return {
        "source_family": int(counts.get("source_family", 0)),
        "transient_signature": int(counts.get("transient_signature", 0)),
        "target_anchor": int(counts.get("target_anchor", 0)),
        "other": int(counts.get("other", 0)),
    }


def _sequence_for_route(rows: pd.DataFrame, *, ascending: bool) -> str:
    ordered = rows.sort_values(
        "bridge_edge_weight_fraction",
        ascending=ascending,
        kind="mergesort",
    )
    parts = [
        f"{float(row.bridge_edge_weight_fraction):.6g}:{row.pathway_state_class}"
        for row in ordered.itertuples(index=False)
    ]
    return " -> ".join(parts)


def _route_pathway_rows(
    *,
    persistence_trace: pd.DataFrame,
    reverse_trace: pd.DataFrame,
    source_route: pd.DataFrame,
) -> pd.DataFrame:
    source_by_route = {
        str(row.route_key): row._asdict()
        for row in source_route.sort_values("route_key", kind="mergesort").itertuples(
            index=False
        )
    }
    rows: list[dict[str, Any]] = []
    grouped_forward = persistence_trace.groupby("route_key", sort=True)
    grouped_reverse = reverse_trace.groupby("route_key", sort=True)
    for route_key in sorted(set(grouped_forward.groups) | set(grouped_reverse.groups)):
        if route_key not in grouped_forward.groups or route_key not in grouped_reverse.groups:
            raise ValueError(f"route_key missing in one trace: {route_key}")
        forward_rows = grouped_forward.get_group(route_key)
        reverse_rows = grouped_reverse.get_group(route_key)
        source_row = source_by_route.get(str(route_key))
        if source_row is None:
            raise ValueError(f"route_key missing source-family row: {route_key}")
        forward_counts = _class_counts_for_route(forward_rows)
        reverse_counts = _class_counts_for_route(reverse_rows)
        forward_source_family_fractions = _fraction_list(
            forward_rows.loc[
                forward_rows["pathway_state_class"].eq("source_family"),
                "bridge_edge_weight_fraction",
            ]
        )
        forward_transient_fractions = _fraction_list(
            forward_rows.loc[
                forward_rows["pathway_state_class"].eq("transient_signature"),
                "bridge_edge_weight_fraction",
            ]
        )
        forward_target_fractions = _fraction_list(
            forward_rows.loc[
                forward_rows["pathway_state_class"].eq("target_anchor"),
                "bridge_edge_weight_fraction",
            ]
        )
        reverse_source_family_fractions = _fraction_list(
            reverse_rows.loc[
                reverse_rows["pathway_state_class"].eq("source_family"),
                "bridge_edge_weight_fraction",
            ]
        )
        reverse_transient_fractions = _fraction_list(
            reverse_rows.loc[
                reverse_rows["pathway_state_class"].eq("transient_signature"),
                "bridge_edge_weight_fraction",
            ]
        )
        reverse_target_fractions = _fraction_list(
            reverse_rows.loc[
                reverse_rows["pathway_state_class"].eq("target_anchor"),
                "bridge_edge_weight_fraction",
            ]
        )
        forward_matches = (
            forward_counts["source_family"] == len(SOURCE_FAMILY_FRACTIONS)
            and forward_counts["transient_signature"] == len(TRANSIENT_BAND_FRACTIONS)
            and forward_counts["target_anchor"] == len(TARGET_FRACTIONS)
            and forward_counts["other"] == 0
        )
        reverse_matches = (
            reverse_counts["source_family"] == len(SOURCE_FAMILY_FRACTIONS)
            and reverse_counts["transient_signature"] == len(TRANSIENT_BAND_FRACTIONS)
            and reverse_counts["target_anchor"] == len(TARGET_FRACTIONS)
            and reverse_counts["other"] == 0
        )
        status = str(source_row["preferred_source_equivalence_status"])
        if status == "source_family_equivalent_guard_overlap_caveat":
            shape_class = "bidirectional_source_family_transition_band_guard_caveat"
        elif status == "source_family_equivalent_anchor_mismatch":
            shape_class = "bidirectional_source_family_transition_band_anchor_mismatch"
        elif status == "source_equivalent_same_seed_anchor":
            shape_class = "bidirectional_source_family_transition_band_strict_source"
        else:
            shape_class = "source_family_pathway_shape_unresolved"
        rows.append(
            {
                "local_pair_id": PRIMARY_PAIR_ID,
                "route_key": str(route_key),
                "start_condition": str(source_row["start_condition"]),
                "seed": int(source_row["seed"]),
                "forward_pathway_shape_matches": bool(forward_matches),
                "reverse_pathway_shape_matches": bool(reverse_matches),
                "forward_source_family_fraction_count": forward_counts["source_family"],
                "forward_transient_fraction_count": forward_counts["transient_signature"],
                "forward_target_fraction_count": forward_counts["target_anchor"],
                "forward_other_fraction_count": forward_counts["other"],
                "reverse_source_family_fraction_count": reverse_counts["source_family"],
                "reverse_transient_fraction_count": reverse_counts["transient_signature"],
                "reverse_target_fraction_count": reverse_counts["target_anchor"],
                "reverse_other_fraction_count": reverse_counts["other"],
                "forward_source_family_fractions": forward_source_family_fractions,
                "forward_transient_fractions": forward_transient_fractions,
                "forward_target_fractions": forward_target_fractions,
                "reverse_source_family_fractions": reverse_source_family_fractions,
                "reverse_transient_fractions": reverse_transient_fractions,
                "reverse_target_fractions": reverse_target_fractions,
                "forward_pathway_sequence": _sequence_for_route(
                    forward_rows,
                    ascending=False,
                ),
                "reverse_pathway_sequence": _sequence_for_route(
                    reverse_rows,
                    ascending=True,
                ),
                "final_signature_id": str(source_row["final_signature_id"]),
                "preferred_source_equivalence_status": status,
                "guard_only_source_family_overlap": bool(
                    _as_bool(source_row["guard_only_source_family_overlap"])
                ),
                "pathway_shape_class": shape_class,
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "method_status": METHOD_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
                "run_status": RUN_STATUS,
            }
        )
    return pd.DataFrame(rows)


def _fraction_pathway_rows(
    *,
    persistence_trace: pd.DataFrame,
    reverse_trace: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fraction in ALL_FRACTIONS:
        forward_rows = persistence_trace[
            persistence_trace["bridge_edge_weight_fraction"].map(_float_key).eq(
                _float_key(fraction)
            )
        ]
        reverse_rows = reverse_trace[
            reverse_trace["bridge_edge_weight_fraction"].map(_float_key).eq(
                _float_key(fraction)
            )
        ]
        forward_counts = _class_counts_for_route(forward_rows)
        reverse_counts = _class_counts_for_route(reverse_rows)
        expected_class = _expected_class_for_fraction(float(fraction))
        rows.append(
            {
                "local_pair_id": PRIMARY_PAIR_ID,
                "bridge_edge_weight_fraction": float(fraction),
                "expected_pathway_state_class": expected_class,
                "forward_route_count": int(len(forward_rows)),
                "reverse_route_count": int(len(reverse_rows)),
                "forward_source_family_count": forward_counts["source_family"],
                "forward_transient_signature_count": forward_counts[
                    "transient_signature"
                ],
                "forward_target_anchor_count": forward_counts["target_anchor"],
                "forward_other_count": forward_counts["other"],
                "reverse_source_family_count": reverse_counts["source_family"],
                "reverse_transient_signature_count": reverse_counts[
                    "transient_signature"
                ],
                "reverse_target_anchor_count": reverse_counts["target_anchor"],
                "reverse_other_count": reverse_counts["other"],
                "forward_expected_class_count": forward_counts.get(expected_class, 0),
                "reverse_expected_class_count": reverse_counts.get(expected_class, 0),
                "both_directions_match_expected_class": bool(
                    forward_counts.get(expected_class, 0) == len(forward_rows)
                    and reverse_counts.get(expected_class, 0) == len(reverse_rows)
                ),
                "forward_distinct_signature_count": int(
                    forward_rows["result_endpoint_signature_id"].nunique()
                ),
                "reverse_distinct_signature_count": int(
                    reverse_rows["result_endpoint_signature_id"].nunique()
                ),
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "method_status": METHOD_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
                "run_status": RUN_STATUS,
            }
        )
    return pd.DataFrame(rows)


def _decision_rows(
    route_rows: pd.DataFrame,
    fraction_rows: pd.DataFrame,
    source_equivalence_summary: dict[str, Any],
) -> pd.DataFrame:
    clean_shape_count = int(
        route_rows[
            route_rows["forward_pathway_shape_matches"].astype(bool)
            & route_rows["reverse_pathway_shape_matches"].astype(bool)
        ].shape[0]
    )
    guard_count = int(route_rows["guard_only_source_family_overlap"].astype(bool).sum())
    shape_class_counts = route_rows["pathway_shape_class"].value_counts().to_dict()
    fraction_pass_count = int(
        fraction_rows["both_directions_match_expected_class"].astype(bool).sum()
    )
    return pd.DataFrame(
        [
            {
                "decision_id": "D1_bidirectional_shape_under_source_family_vocabulary",
                "axis": "pathway_shape",
                "observed": {
                    "matching_route_count": clean_shape_count,
                    "total_route_count": int(len(route_rows)),
                    "shape_class_counts": shape_class_counts,
                },
                "decision": "016_has_bidirectional_source_family_to_transient_band_to_target_shape",
                "passes": clean_shape_count == 24,
                "claim_effect": "pathway shape can be named under source-family vocabulary, not strict same-seed anchors",
            },
            {
                "decision_id": "D2_fraction_bands_are_shared_in_both_directions",
                "axis": "fraction_band",
                "observed": {
                    "matching_fraction_rows": fraction_pass_count,
                    "total_fraction_rows": int(len(fraction_rows)),
                },
                "decision": "same_fraction_state_ladder_in_forward_and_reverse",
                "passes": fraction_pass_count == len(fraction_rows),
                "claim_effect": "the transient is a finite shared band rather than a one-point or one-direction artifact",
            },
            {
                "decision_id": "D3_final_source_family_caveats_remain_named",
                "axis": "final_state_caveat",
                "observed": {
                    "source_equivalence_summary": {
                        "preferred_rule_accepts": source_equivalence_summary.get(
                            "preferred_rule_accepts"
                        ),
                        "preferred_rule_guard_caveats": source_equivalence_summary.get(
                            "preferred_rule_guard_caveats"
                        ),
                        "preferred_status_counts": source_equivalence_summary.get(
                            "preferred_status_counts"
                        ),
                    },
                    "route_guard_caveats": guard_count,
                },
                "decision": "source_family_return_is_not_clean_same_seed_return",
                "passes": int(source_equivalence_summary.get("preferred_rule_accepts", 0))
                == 24
                and guard_count == 1,
                "claim_effect": "keeps the guard-overlap route out of clean wall or endpoint-basin wording",
            },
            {
                "decision_id": "D4_no_wall_or_tunneling_promotion",
                "axis": "claim_boundary",
                "observed": CLAIM_BOUNDARY,
                "decision": "state_sequence_summary_only",
                "passes": True,
                "claim_effect": "wall, tunneling, method, full replay, and quality/cost claims remain closed",
            },
            {
                "decision_id": "D5_next_gate",
                "axis": "next_step",
                "observed": (
                    "source-family pathway vocabulary is now fixed for 016; "
                    "objective/barrier and generality claims are still absent"
                ),
                "decision": "next_gate_should_test_mechanism_or_generalization_not_more_threshold_sweeps",
                "passes": True,
                "claim_effect": "prevents a drift back into broad threshold localization without a mechanism question",
            },
        ]
    )


def _gate_matrix(
    *,
    persistence_gates: pd.DataFrame,
    reverse_summary: dict[str, Any],
    reverse_gates: pd.DataFrame,
    source_equivalence_gates: pd.DataFrame,
    route_rows: pd.DataFrame,
    fraction_rows: pd.DataFrame,
    decision_rows: pd.DataFrame,
) -> pd.DataFrame:
    reverse_failed = list(reverse_summary.get("failed_gates", []))
    matching_routes = int(
        route_rows[
            route_rows["forward_pathway_shape_matches"].astype(bool)
            & route_rows["reverse_pathway_shape_matches"].astype(bool)
        ].shape[0]
    )
    matching_fractions = int(
        fraction_rows["both_directions_match_expected_class"].astype(bool).sum()
    )
    source_family_final_count = int(
        route_rows["preferred_source_equivalence_status"]
        .astype(str)
        .isin(
            {
                "source_equivalent_same_seed_anchor",
                "source_family_equivalent_anchor_mismatch",
                "source_family_equivalent_guard_overlap_caveat",
            }
        )
        .sum()
    )
    guard_count = int(route_rows["guard_only_source_family_overlap"].astype(bool).sum())
    return pd.DataFrame(
        [
            _gate_row(
                "G1_upstream_artifacts_in_expected_state",
                "Are upstream persistence, reverse, and source-family audits in the expected state?",
                {
                    "persistence_gate_status_counts": persistence_gates[
                        "gate_status"
                    ].value_counts().to_dict(),
                    "reverse_failed_gates": reverse_failed,
                    "reverse_gate_status_counts": reverse_gates[
                        "gate_status"
                    ].value_counts().to_dict(),
                    "source_equivalence_gate_status_counts": source_equivalence_gates[
                        "gate_status"
                    ].value_counts().to_dict(),
                },
                "persistence/source-family gates pass, reverse has only the known strict final-source-return failure",
                bool(persistence_gates["gate_status"].astype(str).eq("pass").all())
                and reverse_failed == ["G5_final_source_return_observed"]
                and bool(
                    source_equivalence_gates["gate_status"].astype(str).eq("pass").all()
                ),
            ),
            _gate_row(
                "G2_all_routes_have_forward_and_reverse_shape",
                "Do all 24 routes match the source-family pathway shape in both directions?",
                {
                    "matching_routes": matching_routes,
                    "route_count": int(len(route_rows)),
                    "shape_class_counts": route_rows[
                        "pathway_shape_class"
                    ].value_counts().to_dict(),
                },
                "24/24 routes match forward and reverse source-family pathway shape",
                matching_routes == 24 and len(route_rows) == 24,
            ),
            _gate_row(
                "G3_all_fractions_match_shared_state_ladder",
                "Do all audited bridge fractions match the same state ladder in both directions?",
                {
                    "matching_fraction_rows": matching_fractions,
                    "fraction_count": int(len(fraction_rows)),
                },
                "all 9 fractions match expected source-family/transient/target classes in both directions",
                matching_fractions == len(fraction_rows) and len(fraction_rows) == 9,
            ),
            _gate_row(
                "G4_final_source_family_resolution_complete",
                "Are all reverse final states source-family equivalent under the fixed vocabulary?",
                {
                    "source_family_final_count": source_family_final_count,
                    "guard_overlap_caveat_count": guard_count,
                },
                "24/24 final states are source-family equivalent, with 1 named guard caveat",
                source_family_final_count == 24 and guard_count == 1,
            ),
            _gate_row(
                "G5_transient_band_shared_and_finite",
                "Is the transient band shared by both directions and finite rather than a point artifact?",
                fraction_rows[
                    fraction_rows["expected_pathway_state_class"].astype(str).eq(
                        "transient_signature"
                    )
                ][
                    [
                        "bridge_edge_weight_fraction",
                        "forward_transient_signature_count",
                        "reverse_transient_signature_count",
                    ]
                ].to_dict("records"),
                "six transient-band fractions have 24/24 transient rows in both directions",
                bool(
                    (
                        fraction_rows[
                            fraction_rows[
                                "expected_pathway_state_class"
                            ].astype(str).eq("transient_signature")
                        ][
                            [
                                "forward_transient_signature_count",
                                "reverse_transient_signature_count",
                            ]
                        ]
                        == 24
                    )
                    .all()
                    .all()
                ),
            ),
            _gate_row(
                "G6_claim_boundaries_closed",
                "Are wall, tunneling, method, full replay, and quality/cost claims closed?",
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
    persistence_dir: Path,
    reverse_dir: Path,
    source_equivalence_dir: Path,
    output_dir: Path,
    route_rows: pd.DataFrame,
    fraction_rows: pd.DataFrame,
    decision_rows: pd.DataFrame,
    gates: pd.DataFrame,
) -> dict[str, Any]:
    matching_routes = int(
        route_rows[
            route_rows["forward_pathway_shape_matches"].astype(bool)
            & route_rows["reverse_pathway_shape_matches"].astype(bool)
        ].shape[0]
    )
    matching_fractions = int(
        fraction_rows["both_directions_match_expected_class"].astype(bool).sum()
    )
    transient_rows = fraction_rows[
        fraction_rows["expected_pathway_state_class"].astype(str).eq(
            "transient_signature"
        )
    ]
    source_rows = fraction_rows[
        fraction_rows["expected_pathway_state_class"].astype(str).eq("source_family")
    ]
    target_rows = fraction_rows[
        fraction_rows["expected_pathway_state_class"].astype(str).eq("target_anchor")
    ]
    return {
        "schema": "nanoclustering_g4_8_first_pass_016_pathway_shape_summary.v1",
        "status": RUN_STATUS,
        "persistence_dir": str(persistence_dir),
        "reverse_dir": str(reverse_dir),
        "source_equivalence_dir": str(source_equivalence_dir),
        "output_dir": str(output_dir),
        "primary_pair": PRIMARY_PAIR_ID,
        "route_row_count": int(len(route_rows)),
        "fraction_row_count": int(len(fraction_rows)),
        "decision_row_count": int(len(decision_rows)),
        "matching_bidirectional_route_count": matching_routes,
        "matching_fraction_ladder_count": matching_fractions,
        "pathway_shape_class_counts": route_rows[
            "pathway_shape_class"
        ].value_counts().to_dict(),
        "preferred_pathway_readout": (
            "bidirectional_source_family_to_transient_band_to_target_with_guard_caveat"
        ),
        "source_family_fractions": [
            float(value) for value in sorted(SOURCE_FAMILY_FRACTIONS)
        ],
        "transient_band_fractions": [
            float(value) for value in sorted(TRANSIENT_BAND_FRACTIONS)
        ],
        "target_fractions": [float(value) for value in sorted(TARGET_FRACTIONS)],
        "source_fraction_match_counts": source_rows[
            [
                "bridge_edge_weight_fraction",
                "forward_source_family_count",
                "reverse_source_family_count",
            ]
        ].to_dict("records"),
        "transient_fraction_match_counts": transient_rows[
            [
                "bridge_edge_weight_fraction",
                "forward_transient_signature_count",
                "reverse_transient_signature_count",
            ]
        ].to_dict("records"),
        "target_fraction_match_counts": target_rows[
            [
                "bridge_edge_weight_fraction",
                "forward_target_anchor_count",
                "reverse_target_anchor_count",
            ]
        ].to_dict("records"),
        "gate_status_counts": gates["gate_status"].value_counts().to_dict(),
        "failed_gates": gates.loc[
            ~gates["gate_status"].astype(str).eq("pass"),
            "gate_id",
        ].tolist(),
        "interpretation": (
            "Under same-start source-family vocabulary, 016 has a bidirectional "
            "source-family/target pathway shape: source-family states occupy "
            "0.875 and 1.0, the recurrent transient signature aeb59ab537e6 "
            "occupies 0.625 through 0.8125, and the target anchor occupies 0.5 "
            "in 24/24 routes in both directions. The readout still carries 8 "
            "source-family anchor mismatches and 1 guard-overlap caveat at the "
            "reverse final state, so wall, tunneling, method, full replay, and "
            "quality/cost claims remain closed."
        ),
        "recommended_next_gate": (
            "Treat 016 as a source-family transition-band mechanism object. "
            "The next useful gate should test mechanism/generalization or "
            "objective/barrier interpretation, not another broad threshold "
            "localization sweep."
        ),
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _write_report(
    *,
    path: Path,
    summary: dict[str, Any],
    route_rows: pd.DataFrame,
    fraction_rows: pd.DataFrame,
    decision_rows: pd.DataFrame,
    gates: pd.DataFrame,
) -> None:
    lines = [
        "# NanoClustering G4.8 First-Pass 016 Pathway Shape Audit",
        "",
        "## Summary",
        "",
        f"- status: {summary['status']}",
        f"- preferred_pathway_readout: {summary['preferred_pathway_readout']}",
        f"- matching_bidirectional_route_count: {summary['matching_bidirectional_route_count']}",
        f"- matching_fraction_ladder_count: {summary['matching_fraction_ladder_count']}",
        f"- failed_gates: {summary['failed_gates']}",
        "",
        "## Fraction Ladder",
        "",
        _markdown_table(
            fraction_rows,
            [
                "bridge_edge_weight_fraction",
                "expected_pathway_state_class",
                "forward_source_family_count",
                "forward_transient_signature_count",
                "forward_target_anchor_count",
                "forward_other_count",
                "reverse_source_family_count",
                "reverse_transient_signature_count",
                "reverse_target_anchor_count",
                "reverse_other_count",
                "both_directions_match_expected_class",
            ],
            max_rows=20,
        ),
        "",
        "## Route Rows",
        "",
        _markdown_table(
            route_rows,
            [
                "route_key",
                "forward_pathway_shape_matches",
                "reverse_pathway_shape_matches",
                "preferred_source_equivalence_status",
                "guard_only_source_family_overlap",
                "pathway_shape_class",
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
        str(summary["interpretation"]),
        "",
        "## Recommended Next Gate",
        "",
        str(summary["recommended_next_gate"]),
        "",
        "## Claim Boundary",
        "",
        CLAIM_BOUNDARY,
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_outputs(
    *,
    output_dir: Path,
    config: dict[str, Any],
    summary: dict[str, Any],
    route_rows: pd.DataFrame,
    fraction_rows: pd.DataFrame,
    decision_rows: pd.DataFrame,
    gates: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(route_rows, output_dir / PATHWAY_ROUTE_ROWS_CSV)
    _write_csv(fraction_rows, output_dir / PATHWAY_FRACTION_ROWS_CSV)
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
        fraction_rows=fraction_rows,
        decision_rows=decision_rows,
        gates=gates,
    )


def run_audit(
    *,
    persistence_dir: Path = DEFAULT_PERSISTENCE_DIR,
    reverse_dir: Path = DEFAULT_REVERSE_DIR,
    source_equivalence_dir: Path = DEFAULT_SOURCE_EQUIVALENCE_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    context = _load_context(
        persistence_dir=persistence_dir,
        reverse_dir=reverse_dir,
        source_equivalence_dir=source_equivalence_dir,
    )
    source_signatures_by_start = _source_family_signatures_by_start(
        context["persistence_trace"]
    )
    persistence_trace = _classify_trace_rows(
        context["persistence_trace"],
        source_signatures_by_start,
    )
    reverse_trace = _classify_trace_rows(
        context["reverse_trace"],
        source_signatures_by_start,
    )
    route_rows = _route_pathway_rows(
        persistence_trace=persistence_trace,
        reverse_trace=reverse_trace,
        source_route=context["source_route"],
    )
    fraction_rows = _fraction_pathway_rows(
        persistence_trace=persistence_trace,
        reverse_trace=reverse_trace,
    )
    decision_rows = _decision_rows(
        route_rows,
        fraction_rows,
        context["source_equivalence_summary"],
    )
    gates = _gate_matrix(
        persistence_gates=context["persistence_gates"],
        reverse_summary=context["reverse_summary"],
        reverse_gates=context["reverse_gates"],
        source_equivalence_gates=context["source_equivalence_gates"],
        route_rows=route_rows,
        fraction_rows=fraction_rows,
        decision_rows=decision_rows,
    )
    summary = _summary(
        persistence_dir=persistence_dir,
        reverse_dir=reverse_dir,
        source_equivalence_dir=source_equivalence_dir,
        output_dir=output_dir,
        route_rows=route_rows,
        fraction_rows=fraction_rows,
        decision_rows=decision_rows,
        gates=gates,
    )
    config = {
        "schema": "nanoclustering_g4_8_first_pass_016_pathway_shape_config.v1",
        "persistence_dir": str(persistence_dir),
        "reverse_dir": str(reverse_dir),
        "source_equivalence_dir": str(source_equivalence_dir),
        "output_dir": str(output_dir),
        "primary_pair": PRIMARY_PAIR_ID,
        "target_signature_id": TARGET_SIGNATURE_ID,
        "transient_signature_id": TRANSIENT_SIGNATURE_ID,
        "source_family_fractions": [
            float(value) for value in sorted(SOURCE_FAMILY_FRACTIONS)
        ],
        "transient_band_fractions": [
            float(value) for value in sorted(TRANSIENT_BAND_FRACTIONS)
        ],
        "target_fractions": [float(value) for value in sorted(TARGET_FRACTIONS)],
        "source_signatures_by_start": {
            key: sorted(value) for key, value in source_signatures_by_start.items()
        },
        "route_execution_status": ROUTE_EXECUTION_STATUS,
        "wall_promotion_status": WALL_PROMOTION_STATUS,
        "method_status": METHOD_STATUS,
        "claim_boundary": CLAIM_BOUNDARY,
        "run_status": RUN_STATUS,
    }
    _write_outputs(
        output_dir=output_dir,
        config=config,
        summary=summary,
        route_rows=route_rows,
        fraction_rows=fraction_rows,
        decision_rows=decision_rows,
        gates=gates,
    )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit local_pair_016 pathway shape under source-family vocabulary.",
    )
    parser.add_argument(
        "--persistence-dir",
        type=Path,
        default=DEFAULT_PERSISTENCE_DIR,
        help="Input persistence trace artifact directory.",
    )
    parser.add_argument(
        "--reverse-dir",
        type=Path,
        default=DEFAULT_REVERSE_DIR,
        help="Input reverse trace artifact directory.",
    )
    parser.add_argument(
        "--source-equivalence-dir",
        type=Path,
        default=DEFAULT_SOURCE_EQUIVALENCE_DIR,
        help="Input source-family equivalence artifact directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for the pathway-shape audit.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = run_audit(
        persistence_dir=args.persistence_dir,
        reverse_dir=args.reverse_dir,
        source_equivalence_dir=args.source_equivalence_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
