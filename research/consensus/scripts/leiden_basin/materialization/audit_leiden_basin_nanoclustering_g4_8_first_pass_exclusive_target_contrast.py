#!/usr/bin/env python3
"""Audit first-pass exclusive-target contrast after the fresh Axis B trace.

This is a read-only audit over the executed first-pass trace. It compares clean
exclusive bridge-target routes against source/target collapse, guard-anchor
collapse, and intermediate unknown endpoints. It does not rerun Leiden, promote
walls, evaluate quality/cost value, replay full NanoClustering, or claim method
success.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from run_leiden_basin_nanoclustering_g4_8_fresh_axis_b_first_pass_trace import (
    DEFAULT_OUTPUT_DIR as DEFAULT_FIRST_PASS_TRACE_DIR,
    PAIR_READOUT_RESULT_ROWS_CSV,
    ROUTE_READOUT_RESULT_ROWS_CSV,
    TRACE_ROWS_CSV,
)
from run_leiden_basin_nanoclustering_role_local_route_pilot import (
    BASE_RESULT_DIR,
    _json_safe,
    _read_csv,
    _write_csv,
)


DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_exclusive_target_contrast_audit_gamma1e5_20260604"
)

ROUTE_CONTRAST_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_exclusive_target_route_contrast_rows.csv"
)
PAIR_CONTRAST_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_exclusive_target_pair_contrast_rows.csv"
)
SIGNATURE_CONTRAST_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_exclusive_target_signature_contrast_rows.csv"
)
GATE_MATRIX_CSV = (
    "nanoclustering_g4_8_first_pass_exclusive_target_contrast_gate_matrix.csv"
)
SUMMARY_JSON = (
    "nanoclustering_g4_8_first_pass_exclusive_target_contrast_summary.json"
)
CONFIG_JSON = "nanoclustering_g4_8_first_pass_exclusive_target_contrast_config.json"
REPORT_MD = "nanoclustering_g4_8_first_pass_exclusive_target_contrast_report.md"

RUN_STATUS = "audited_nanoclustering_g4_8_first_pass_exclusive_target_contrast"
ROUTE_EXECUTION_STATUS = "not_executed_read_only_first_pass_trace_audit"
WALL_PROMOTION_STATUS = "not_promoted_exclusive_target_contrast_audit_only"
METHOD_STATUS = "diagnostic_readout_audit_not_method"
CLAIM_BOUNDARY = (
    "NanoClustering G4.8 first-pass exclusive-target contrast audit only; reads "
    "executed route-local trace outputs and classifies endpoint exclusivity, "
    "source/target collapse, guard-anchor collapse, and intermediate unknown "
    "endpoints. It does not rerun Leiden, promote walls, evaluate quality/cost "
    "value, replay full NanoClustering, or claim method success."
)

READY_ROLE = "conditional_ready_like_test"
CONTROL_ROLE = "control_false_positive_guard"


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _contains_assignment(value: Any, label: str) -> bool:
    return label in str(value).split(";") or label in str(value)


def _route_contrast_rows(
    *,
    route_results: pd.DataFrame,
    trace_rows: pd.DataFrame,
) -> pd.DataFrame:
    first_rows = trace_rows[trace_rows["step_index"].astype(int).eq(1)].copy()
    final_rows = (
        trace_rows.sort_values("step_index", kind="mergesort")
        .groupby(["route_contract_id", "seed"], sort=False)
        .tail(1)
        .copy()
    )
    post_rows = trace_rows[trace_rows["step_index"].astype(int).gt(1)].copy()
    first_lookup = first_rows.set_index(["route_contract_id", "seed"]).to_dict("index")
    final_lookup = final_rows.set_index(["route_contract_id", "seed"]).to_dict("index")
    post_group = post_rows.groupby(["route_contract_id", "seed"], sort=False)

    rows: list[dict[str, Any]] = []
    for route in route_results.itertuples(index=False):
        route_contract_id = str(route.route_contract_id)
        seed = int(route.seed)
        first = first_lookup.get((route_contract_id, seed), {})
        final = final_lookup.get((route_contract_id, seed), {})
        try:
            post = post_group.get_group((route_contract_id, seed)).sort_values(
                "step_index", kind="mergesort"
            )
        except KeyError:
            post = pd.DataFrame()

        post_assignments = (
            post["endpoint_assignment_by_step"].astype(str).tolist()
            if not post.empty
            else []
        )
        post_signature_sequence = (
            " -> ".join(post["result_endpoint_signature_id"].astype(str).tolist())
            if not post.empty
            else ""
        )
        first_assignment = str(first.get("endpoint_assignment_by_step", ""))
        final_assignment = str(final.get("endpoint_assignment_by_step", ""))
        first_signature_id = str(first.get("result_endpoint_signature_id", ""))
        final_signature_id = str(final.get("result_endpoint_signature_id", ""))

        has_unknown_post_start = any(value == "unknown_new_endpoint" for value in post_assignments)
        has_ambiguous_post_start = any(
            value.startswith("ambiguous_anchor_match") for value in post_assignments
        )
        final_matches_bridge_target = _contains_assignment(
            final_assignment, "drop_bridge_target_anchor"
        )
        final_matches_original_source = _contains_assignment(
            final_assignment, "original_source_anchor"
        )
        final_matches_drop_both_guard = _contains_assignment(
            final_assignment, "drop_both_guard_anchor"
        )
        final_matches_drop_direct_guard = _contains_assignment(
            final_assignment, "drop_direct_guard_anchor"
        )
        final_exclusive_bridge_target = final_assignment == "drop_bridge_target_anchor"
        first_final_signature_same = first_signature_id == final_signature_id
        source_target_signature_collapse = (
            final_matches_bridge_target
            and final_matches_original_source
        ) or (
            final_matches_bridge_target
            and first_final_signature_same
        )
        guard_anchor_collapse = final_matches_bridge_target and (
            final_matches_drop_both_guard or final_matches_drop_direct_guard
        )

        if _as_bool(route.all_positive_requirements_pass):
            contrast_class = "exclusive_bridge_target_contrast_pass"
        elif source_target_signature_collapse:
            contrast_class = "source_target_signature_collapse"
        elif guard_anchor_collapse:
            contrast_class = "guard_anchor_collapse"
        elif has_unknown_post_start:
            contrast_class = "intermediate_unknown_endpoint"
        elif has_ambiguous_post_start:
            contrast_class = "intermediate_ambiguous_anchor"
        elif not _as_bool(route.source_start_support_pass):
            contrast_class = "source_start_support_failure"
        else:
            contrast_class = "other_first_pass_failure"

        rows.append(
            {
                "route_contract_id": route_contract_id,
                "local_pair_id": str(route.local_pair_id),
                "branch": str(route.branch),
                "start_condition": str(route.start_condition),
                "seed": seed,
                "evidence_role": str(route.evidence_role),
                "validation_stratum": str(route.validation_stratum),
                "source_start_support_pass": _as_bool(route.source_start_support_pass),
                "post_start_endpoint_continuity_pass": _as_bool(
                    route.post_start_endpoint_continuity_pass
                ),
                "target_final_continuity_pass": _as_bool(route.target_final_continuity_pass),
                "target_final_bridge_exclusive_pass": _as_bool(
                    route.target_final_bridge_exclusive_pass
                ),
                "direct_edge_retention_pass": _as_bool(route.direct_edge_retention_pass),
                "all_positive_requirements_pass": _as_bool(
                    route.all_positive_requirements_pass
                ),
                "has_unknown_post_start": has_unknown_post_start,
                "has_ambiguous_post_start": has_ambiguous_post_start,
                "final_matches_bridge_target": final_matches_bridge_target,
                "final_matches_original_source": final_matches_original_source,
                "final_matches_drop_both_guard": final_matches_drop_both_guard,
                "final_matches_drop_direct_guard": final_matches_drop_direct_guard,
                "final_exclusive_bridge_target": final_exclusive_bridge_target,
                "source_target_signature_collapse": source_target_signature_collapse,
                "guard_anchor_collapse": guard_anchor_collapse,
                "first_final_signature_same": first_final_signature_same,
                "first_endpoint_assignment": first_assignment,
                "final_endpoint_assignment": final_assignment,
                "first_signature_id": first_signature_id,
                "final_signature_id": final_signature_id,
                "post_endpoint_assignment_sequence": " -> ".join(post_assignments),
                "post_signature_sequence": post_signature_sequence,
                "route_outcome_class": str(route.route_outcome_class),
                "contrast_class": contrast_class,
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "method_status": METHOD_STATUS,
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows)


def _pair_contrast_rows(
    *,
    pair_results: pd.DataFrame,
    route_contrast: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pair in pair_results.sort_values("local_pair_id", kind="mergesort").itertuples(
        index=False
    ):
        local_pair_id = str(pair.local_pair_id)
        group = route_contrast[
            route_contrast["local_pair_id"].astype(str).eq(local_pair_id)
        ].copy()
        route_count = int(len(group))
        pass_count = int(group["all_positive_requirements_pass"].map(_as_bool).sum())
        source_target_collapse_count = int(
            group["source_target_signature_collapse"].map(_as_bool).sum()
        )
        guard_collapse_count = int(group["guard_anchor_collapse"].map(_as_bool).sum())
        unknown_count = int(group["has_unknown_post_start"].map(_as_bool).sum())
        ambiguous_count = int(group["has_ambiguous_post_start"].map(_as_bool).sum())
        exclusive_target_count = int(group["final_exclusive_bridge_target"].map(_as_bool).sum())
        clean_start_conditions = sorted(
            group.loc[
                group["all_positive_requirements_pass"].map(_as_bool),
                "start_condition",
            ]
            .astype(str)
            .unique()
            .tolist()
        )
        clean_seeds = sorted(
            map(
                int,
                group.loc[
                    group["all_positive_requirements_pass"].map(_as_bool),
                    "seed",
                ]
                .dropna()
                .unique()
                .tolist(),
            )
        )
        if str(pair.evidence_role) == READY_ROLE and route_count and pass_count == route_count:
            escalation_class = "clean_exclusive_target_candidate"
            escalation_allowed = True
        elif str(pair.evidence_role) == READY_ROLE and pass_count > 0:
            escalation_class = "partial_exclusive_target_candidate"
            escalation_allowed = True
        elif str(pair.evidence_role) == CONTROL_ROLE:
            escalation_class = "control_closed"
            escalation_allowed = False
        else:
            escalation_class = "not_escalation_candidate"
            escalation_allowed = False
        rows.append(
            {
                "local_pair_id": local_pair_id,
                "branch": str(pair.branch),
                "evidence_role": str(pair.evidence_role),
                "validation_stratum": str(pair.validation_stratum),
                "pair_first_pass_result": str(pair.pair_first_pass_result),
                "route_count": route_count,
                "exclusive_bridge_target_pass_count": pass_count,
                "exclusive_bridge_target_pass_share": float(pass_count / route_count)
                if route_count
                else 0.0,
                "final_exclusive_bridge_target_count": exclusive_target_count,
                "source_target_signature_collapse_count": source_target_collapse_count,
                "guard_anchor_collapse_count": guard_collapse_count,
                "intermediate_unknown_route_count": unknown_count,
                "intermediate_ambiguous_route_count": ambiguous_count,
                "distinct_first_signature_count": int(group["first_signature_id"].nunique())
                if route_count
                else 0,
                "distinct_final_signature_count": int(group["final_signature_id"].nunique())
                if route_count
                else 0,
                "clean_start_conditions": ";".join(clean_start_conditions),
                "clean_seed_count": len(clean_seeds),
                "clean_seeds": ";".join(str(seed) for seed in clean_seeds),
                "contrast_class_counts": group["contrast_class"].value_counts().to_dict()
                if route_count
                else {},
                "next_escalation_class": escalation_class,
                "exclusive_target_audit_escalation_allowed": escalation_allowed,
                "wall_claim_allowed_after_audit": False,
                "method_claim_allowed_after_audit": False,
                "quality_cost_claim_allowed_after_audit": False,
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "method_status": METHOD_STATUS,
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows)


def _signature_contrast_rows(route_contrast: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in route_contrast.groupby(
        ["local_pair_id", "evidence_role"], sort=False
    ):
        local_pair_id, evidence_role = keys
        for role, column in [
            ("first_source_like_signature", "first_signature_id"),
            ("final_signature", "final_signature_id"),
        ]:
            counts = group[column].astype(str).value_counts()
            for signature_id, count in counts.items():
                sig_group = group[group[column].astype(str).eq(str(signature_id))]
                rows.append(
                    {
                        "local_pair_id": str(local_pair_id),
                        "evidence_role": str(evidence_role),
                        "signature_role": role,
                        "signature_id": str(signature_id),
                        "route_count": int(count),
                        "start_conditions": ";".join(
                            sorted(sig_group["start_condition"].astype(str).unique().tolist())
                        ),
                        "seeds": ";".join(
                            str(int(seed))
                            for seed in sorted(sig_group["seed"].dropna().unique().tolist())
                        ),
                        "contrast_class_counts": sig_group[
                            "contrast_class"
                        ].value_counts().to_dict(),
                        "route_execution_status": ROUTE_EXECUTION_STATUS,
                        "wall_promotion_status": WALL_PROMOTION_STATUS,
                        "method_status": METHOD_STATUS,
                        "run_status": RUN_STATUS,
                        "claim_boundary": CLAIM_BOUNDARY,
                    }
                )
    return pd.DataFrame(rows)


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


def _gate_matrix(
    *,
    route_contrast: pd.DataFrame,
    pair_contrast: pd.DataFrame,
    signature_contrast: pd.DataFrame,
) -> pd.DataFrame:
    clean_ready = pair_contrast[
        pair_contrast["next_escalation_class"].astype(str).eq(
            "clean_exclusive_target_candidate"
        )
    ]
    partial_ready = pair_contrast[
        pair_contrast["next_escalation_class"].astype(str).eq(
            "partial_exclusive_target_candidate"
        )
    ]
    control_rows = pair_contrast[pair_contrast["evidence_role"].astype(str).eq(CONTROL_ROLE)]
    rows = [
        _gate_row(
            "G1_route_contrast_rows_complete",
            "Was every first-pass route result classified into an exclusivity contrast class?",
            f"route_contrast_rows={len(route_contrast)} missing_classes={int(route_contrast['contrast_class'].isna().sum())}",
            "288 route rows, no missing contrast classes",
            len(route_contrast) == 288 and not bool(route_contrast["contrast_class"].isna().any()),
        ),
        _gate_row(
            "G2_controls_remain_closed_after_exclusivity",
            "Do control pairs remain closed under exclusive-target contrast?",
            control_rows[["local_pair_id", "contrast_class_counts"]].to_dict("records"),
            "all controls have zero exclusive bridge-target pass count",
            bool(control_rows["exclusive_bridge_target_pass_count"].eq(0).all()),
        ),
        _gate_row(
            "G3_clean_and_partial_ready_scope_bounded",
            "Is escalation bounded to clean and partial ready-like candidates only?",
            {
                "clean": clean_ready["local_pair_id"].tolist(),
                "partial": partial_ready["local_pair_id"].tolist(),
            },
            "at least one clean candidate and no control escalation",
            not clean_ready.empty
            and bool(control_rows["exclusive_target_audit_escalation_allowed"].eq(False).all()),
        ),
        _gate_row(
            "G4_signature_contrast_materialized",
            "Were source-like and final signature contrast rows materialized?",
            f"signature_contrast_rows={len(signature_contrast)}",
            "nonempty signature contrast table",
            not signature_contrast.empty,
        ),
        _gate_row(
            "G5_wall_method_quality_claims_closed",
            "Are wall, method, and quality/cost claims still closed?",
            CLAIM_BOUNDARY,
            "all claim flags remain false",
            bool(pair_contrast["wall_claim_allowed_after_audit"].eq(False).all())
            and bool(pair_contrast["method_claim_allowed_after_audit"].eq(False).all())
            and bool(pair_contrast["quality_cost_claim_allowed_after_audit"].eq(False).all()),
        ),
    ]
    return pd.DataFrame(rows)


def _summary(
    *,
    first_pass_trace_dir: Path,
    output_dir: Path,
    route_contrast: pd.DataFrame,
    pair_contrast: pd.DataFrame,
    signature_contrast: pd.DataFrame,
    gates: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "schema": "nanoclustering_g4_8_first_pass_exclusive_target_contrast_summary.v1",
        "status": RUN_STATUS,
        "first_pass_trace_dir": str(first_pass_trace_dir),
        "output_dir": str(output_dir),
        "route_contrast_row_count": int(len(route_contrast)),
        "pair_contrast_row_count": int(len(pair_contrast)),
        "signature_contrast_row_count": int(len(signature_contrast)),
        "contrast_class_counts": route_contrast["contrast_class"].value_counts().to_dict(),
        "pair_escalation_class_counts": pair_contrast[
            "next_escalation_class"
        ].value_counts().to_dict(),
        "clean_candidates": pair_contrast.loc[
            pair_contrast["next_escalation_class"].astype(str).eq(
                "clean_exclusive_target_candidate"
            ),
            "local_pair_id",
        ].tolist(),
        "partial_candidates": pair_contrast.loc[
            pair_contrast["next_escalation_class"].astype(str).eq(
                "partial_exclusive_target_candidate"
            ),
            "local_pair_id",
        ].tolist(),
        "control_closed_pairs": pair_contrast.loc[
            pair_contrast["evidence_role"].astype(str).eq(CONTROL_ROLE),
            "local_pair_id",
        ].tolist(),
        "gate_status_counts": gates["gate_status"].value_counts().to_dict(),
        "failed_gates": gates.loc[
            ~gates["gate_status"].astype(str).eq("pass"), "gate_id"
        ].tolist(),
        "interpretation": (
            "Exclusive bridge-target contrast separates clean first-pass evidence "
            "from source/target signature collapse, guard-anchor collapse, and "
            "intermediate unknown endpoints. This bounds the next audit to the "
            "clean candidate and one partial boundary candidate."
        ),
        "recommended_next_gate": (
            "Build the next object-level audit on local_pair_014 as the clean "
            "candidate and local_pair_005 as a partial-collapse boundary case; "
            "do not include controls or post-start-failure pairs as positives."
        ),
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 40) -> str:
    cols = [col for col in columns if col in frame.columns]
    if not cols:
        return "No columns."
    visible = frame[cols].head(int(max_rows))
    header = "| " + " | ".join(cols) + " |"
    separator = "| " + " | ".join("---" for _ in cols) + " |"
    rows: list[str] = []
    for row in visible.itertuples(index=False):
        values: list[str] = []
        for value in row:
            if isinstance(value, (dict, list, tuple, set)):
                values.append(json.dumps(_json_safe(value), sort_keys=True))
            elif pd.isna(value):
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value).replace("\n", " "))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def _write_report(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    pair_contrast: pd.DataFrame,
    route_contrast: pd.DataFrame,
    gates: pd.DataFrame,
) -> None:
    lines = [
        "# NanoClustering G4.8 First-Pass Exclusive-Target Contrast Audit",
        "",
        f"- status: `{summary['status']}`",
        f"- route_contrast_row_count: {summary['route_contrast_row_count']}",
        f"- pair_contrast_row_count: {summary['pair_contrast_row_count']}",
        f"- signature_contrast_row_count: {summary['signature_contrast_row_count']}",
        f"- contrast_class_counts: {summary['contrast_class_counts']}",
        f"- pair_escalation_class_counts: {summary['pair_escalation_class_counts']}",
        f"- clean_candidates: {summary['clean_candidates']}",
        f"- partial_candidates: {summary['partial_candidates']}",
        f"- gate_status_counts: {summary['gate_status_counts']}",
        f"- failed_gates: {summary['failed_gates']}",
        f"- interpretation: {summary['interpretation']}",
        f"- recommended_next_gate: {summary['recommended_next_gate']}",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Pair Contrast",
        "",
        _markdown_table(
            pair_contrast.sort_values(
                ["evidence_role", "local_pair_id"], kind="mergesort"
            ),
            [
                "local_pair_id",
                "evidence_role",
                "validation_stratum",
                "route_count",
                "exclusive_bridge_target_pass_count",
                "exclusive_bridge_target_pass_share",
                "source_target_signature_collapse_count",
                "guard_anchor_collapse_count",
                "intermediate_unknown_route_count",
                "distinct_first_signature_count",
                "distinct_final_signature_count",
                "next_escalation_class",
            ],
            max_rows=20,
        ),
        "",
        "## Route Contrast",
        "",
        _markdown_table(
            route_contrast.sort_values(
                ["local_pair_id", "start_condition", "seed"], kind="mergesort"
            ),
            [
                "local_pair_id",
                "start_condition",
                "seed",
                "evidence_role",
                "contrast_class",
                "first_endpoint_assignment",
                "final_endpoint_assignment",
                "first_signature_id",
                "final_signature_id",
            ],
            max_rows=80,
        ),
        "",
        "## Gate Matrix",
        "",
        _markdown_table(
            gates,
            ["gate_id", "gate_status", "observed", "minimum_or_rule", "question"],
            max_rows=20,
        ),
        "",
        "## Boundary",
        "",
        (
            "This audit only interprets already executed route-local traces. It "
            "does not establish basin walls or method value. Its purpose is to "
            "bound the next object-level audit to the clean and partial cases."
        ),
        "",
    ]
    (output_dir / REPORT_MD).write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    first_pass_trace_dir = Path(args.first_pass_trace_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    route_results = _read_csv(first_pass_trace_dir / ROUTE_READOUT_RESULT_ROWS_CSV)
    pair_results = _read_csv(first_pass_trace_dir / PAIR_READOUT_RESULT_ROWS_CSV)
    trace_rows = _read_csv(first_pass_trace_dir / TRACE_ROWS_CSV)

    route_contrast = _route_contrast_rows(route_results=route_results, trace_rows=trace_rows)
    pair_contrast = _pair_contrast_rows(
        pair_results=pair_results,
        route_contrast=route_contrast,
    )
    signature_contrast = _signature_contrast_rows(route_contrast)
    gates = _gate_matrix(
        route_contrast=route_contrast,
        pair_contrast=pair_contrast,
        signature_contrast=signature_contrast,
    )
    summary = _summary(
        first_pass_trace_dir=first_pass_trace_dir,
        output_dir=output_dir,
        route_contrast=route_contrast,
        pair_contrast=pair_contrast,
        signature_contrast=signature_contrast,
        gates=gates,
    )

    _write_csv(route_contrast, output_dir / ROUTE_CONTRAST_ROWS_CSV)
    _write_csv(pair_contrast, output_dir / PAIR_CONTRAST_ROWS_CSV)
    _write_csv(signature_contrast, output_dir / SIGNATURE_CONTRAST_ROWS_CSV)
    _write_csv(gates, output_dir / GATE_MATRIX_CSV)
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    config = {
        "schema": "nanoclustering_g4_8_first_pass_exclusive_target_contrast_config.v1",
        "first_pass_trace_dir": str(first_pass_trace_dir),
        "output_dir": str(output_dir),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(
        output_dir=output_dir,
        summary=summary,
        pair_contrast=pair_contrast,
        route_contrast=route_contrast,
        gates=gates,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-pass-trace-dir", type=Path, default=DEFAULT_FIRST_PASS_TRACE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
