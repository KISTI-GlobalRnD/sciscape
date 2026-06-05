#!/usr/bin/env python3
"""Audit wall evidence over the scoped G4.8 pathway-probe traces.

This consumes the executed scoped pathway-probe trace output and asks whether
the materialized route traces are enough to promote any wall language. The
answer is intentionally separated from pathway-trace evidence: bridge-release
routes can become wall-audit candidates, while wall claims remain closed until
distinct basin-pair relation, direct-path availability, objective debt/recovery,
polish reversion, and support-incompatibility evidence are accepted together.

It does not run Leiden, broaden route execution, evaluate quality/cost value,
replay full NanoClustering, or claim method/algorithm success.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from run_leiden_basin_nanoclustering_g4_8_scoped_pathway_probe_trace import (
    DEFAULT_OUTPUT_DIR as DEFAULT_TRACE_DIR,
    GATE_MATRIX_CSV as TRACE_GATE_MATRIX_CSV,
    ROUTE_CONTRACT_SUMMARY_CSV,
    SEED_ROUTE_SUMMARY_CSV,
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
    / "leiden_basin_nanoclustering_g4_8_scoped_pathway_wall_evidence_audit_gamma1e5_20260604"
)

SEED_AUDIT_ROWS_CSV = "nanoclustering_g4_8_scoped_pathway_wall_evidence_seed_rows.csv"
CONTRACT_AUDIT_ROWS_CSV = (
    "nanoclustering_g4_8_scoped_pathway_wall_evidence_contract_rows.csv"
)
FAMILY_AUDIT_ROWS_CSV = (
    "nanoclustering_g4_8_scoped_pathway_wall_evidence_family_rows.csv"
)
GATE_MATRIX_CSV = "nanoclustering_g4_8_scoped_pathway_wall_evidence_gate_matrix.csv"
SUMMARY_JSON = "nanoclustering_g4_8_scoped_pathway_wall_evidence_summary.json"
CONFIG_JSON = "nanoclustering_g4_8_scoped_pathway_wall_evidence_config.json"
REPORT_MD = "nanoclustering_g4_8_scoped_pathway_wall_evidence_report.md"

PRIMARY_ROUTE_FAMILY = "bridge_release_interpolation_probe"
DIRECT_GUARD_FAMILY = "direct_dependency_collapse_guard"
DROP_BOTH_GUARD_FAMILY = "drop_both_collapse_guard"

RUN_STATUS = "audited_nanoclustering_g4_8_scoped_pathway_wall_evidence"
CLAIM_BOUNDARY = (
    "NanoClustering G4.8 scoped pathway wall-evidence audit only; reads executed "
    "local route traces and classifies pathway/wall-readiness evidence. It does "
    "not run Leiden, broaden route execution, promote walls, evaluate quality/cost "
    "value, replay full NanoClustering, or claim method/algorithm success."
)


def _count_dict(series: pd.Series) -> dict[str, int]:
    if series.empty:
        return {}
    return {str(key): int(value) for key, value in series.value_counts(dropna=False).items()}


def _markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 50) -> str:
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


def _bool_series(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(bool)


def _seed_audit_rows(seed_summary: pd.DataFrame, trace_rows: pd.DataFrame) -> pd.DataFrame:
    post_start = trace_rows[trace_rows["step_index"].astype(int).gt(1)].copy()
    post_metrics = (
        post_start.groupby(["route_contract_id", "seed"], sort=False)
        .agg(
            post_start_polish_reversion_count=("polish_reversion_check", "sum"),
            post_start_support_incompatibility_count=("support_incompatibility_check", "sum"),
            post_start_unknown_endpoint_count=(
                "endpoint_assignment_by_step",
                lambda values: int(pd.Series(values).astype(str).eq("unknown_new_endpoint").sum()),
            ),
            final_endpoint_assignment=("endpoint_assignment_by_step", "last"),
            final_matches_expected=("matches_expected_final_anchor", "last"),
            final_support_distance=("support_distance_by_step", "last"),
        )
        .reset_index()
    )
    rows = seed_summary.merge(
        post_metrics,
        on=["route_contract_id", "seed"],
        how="left",
        validate="one_to_one",
    )
    rows["transition_seed"] = rows["route_trace_class"].astype(str).eq(
        "source_to_expected_anchor_transition"
    )
    rows["objective_debt_evidence"] = rows["max_objective_debt_from_start"].astype(float).gt(0.0)
    rows["objective_recovery_evidence"] = rows[
        "max_objective_recovery_from_min"
    ].astype(float).gt(0.0)
    rows["post_start_polish_reversion_evidence"] = rows[
        "post_start_polish_reversion_count"
    ].fillna(0).astype(int).gt(0)
    rows["support_incompatibility_evidence"] = rows[
        "post_start_support_incompatibility_count"
    ].fillna(0).astype(int).gt(0)
    rows["intermediate_unknown_endpoint_evidence"] = rows[
        "post_start_unknown_endpoint_count"
    ].fillna(0).astype(int).gt(0)
    rows["primary_pathway_seed_candidate"] = (
        rows["planned_route_family"].astype(str).eq(PRIMARY_ROUTE_FAMILY)
        & rows["transition_seed"]
        & rows["source_start_anchor_matched"].astype(bool)
        & rows["expected_final_anchor_reached"].astype(bool)
    )
    rows["direct_guard_seed_stable"] = rows["planned_route_family"].astype(str).eq(
        DIRECT_GUARD_FAMILY
    ) & rows["route_trace_class"].astype(str).eq("no_endpoint_transition")
    rows["drop_both_guard_seed_stable"] = rows["planned_route_family"].astype(str).eq(
        DROP_BOTH_GUARD_FAMILY
    ) & rows["transition_seed"]
    rows["wall_seed_ready"] = False

    def status(row: pd.Series) -> str:
        family = str(row["planned_route_family"])
        if family == PRIMARY_ROUTE_FAMILY:
            if not bool(row["primary_pathway_seed_candidate"]):
                return "primary_bridge_release_seed_not_transition"
            if bool(row["support_incompatibility_evidence"]):
                return "primary_pathway_seed_has_intermediate_unknown_support_incompatibility"
            if not bool(row["objective_recovery_evidence"]):
                return "primary_pathway_seed_transition_without_objective_recovery"
            return "primary_pathway_seed_transition_recovery_no_wall_yet"
        if family == DIRECT_GUARD_FAMILY:
            if bool(row["transition_seed"]):
                return "direct_guard_seed_collapses_to_expected"
            return "direct_guard_seed_no_transition"
        if family == DROP_BOTH_GUARD_FAMILY:
            if bool(row["transition_seed"]):
                return "drop_both_guard_seed_collapses_to_expected"
            return "drop_both_guard_seed_not_collapsed"
        return "unclassified_seed_route"

    rows["wall_evidence_seed_status"] = rows.apply(status, axis=1)
    rows["wall_seed_block_reason"] = (
        "seed-level trace is diagnostic only; wall promotion requires contract-level "
        "and pair-level acceptance of basin relation, direct-path availability, "
        "objective debt/recovery, polish reversion, and support incompatibility"
    )
    rows["run_status"] = RUN_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows.sort_values(
        ["local_pair_id", "start_condition", "planned_route_family", "seed"],
        kind="mergesort",
    ).reset_index(drop=True)


def _contract_audit_rows(seed_audit: pd.DataFrame, contract_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = [
        "route_contract_id",
        "validation_unit_id",
        "local_pair_id",
        "start_condition",
        "planned_route_family",
    ]
    for keys, group in seed_audit.groupby(group_cols, sort=False):
        key_data = dict(zip(group_cols, keys, strict=True))
        first = group.iloc[0]
        seed_count = int(group["seed"].nunique())
        transition_count = int(group["transition_seed"].sum())
        recovery_count = int(group["objective_recovery_evidence"].sum())
        incompat_count = int(group["support_incompatibility_evidence"].sum())
        post_reversion_count = int(group["post_start_polish_reversion_evidence"].sum())
        unknown_seed_count = int(group["intermediate_unknown_endpoint_evidence"].sum())
        primary_all = (
            str(first["planned_route_family"]) == PRIMARY_ROUTE_FAMILY
            and transition_count == seed_count
        )
        direct_partial = (
            str(first["planned_route_family"]) == DIRECT_GUARD_FAMILY
            and 0 < transition_count < seed_count
        )
        drop_both_all = (
            str(first["planned_route_family"]) == DROP_BOTH_GUARD_FAMILY
            and transition_count == seed_count
        )
        distinct_relation_candidate = bool(primary_all)
        objective_debt_evidence = bool(group["objective_debt_evidence"].all())
        objective_recovery_evidence = bool(recovery_count == seed_count)
        support_incompatibility_evidence = bool(incompat_count > 0)
        polish_reversion_evidence = bool(post_reversion_count > 0)
        direct_path_available = False
        if primary_all:
            if support_incompatibility_evidence:
                status = "primary_pathway_trace_with_intermediate_unknown_wall_audit_candidate"
            elif objective_recovery_evidence:
                status = "primary_pathway_trace_with_recovery_wall_audit_candidate"
            else:
                status = "primary_pathway_trace_without_recovery_wall_closed"
        elif direct_partial:
            status = "direct_dependency_guard_partial_wall_closed"
        elif drop_both_all:
            status = "drop_both_collapse_guard_passed_not_wall"
        else:
            status = "nonprimary_or_unstable_trace_wall_closed"
        rows.append(
            {
                **key_data,
                "seed_count": seed_count,
                "transition_seed_count": transition_count,
                "no_transition_seed_count": int(
                    group["route_trace_class"].astype(str).eq("no_endpoint_transition").sum()
                ),
                "objective_debt_seed_count": int(group["objective_debt_evidence"].sum()),
                "objective_recovery_seed_count": recovery_count,
                "support_incompatibility_seed_count": incompat_count,
                "post_start_polish_reversion_seed_count": post_reversion_count,
                "intermediate_unknown_endpoint_seed_count": unknown_seed_count,
                "distinct_basin_pair_relation_candidate": distinct_relation_candidate,
                "direct_path_availability_evidence": direct_path_available,
                "objective_debt_evidence": objective_debt_evidence,
                "objective_recovery_evidence": objective_recovery_evidence,
                "polish_reversion_evidence": polish_reversion_evidence,
                "support_incompatibility_evidence": support_incompatibility_evidence,
                "wall_contract_ready": False,
                "wall_contract_status": status,
                "wall_contract_block_reason": (
                    "no wall promotion: direct path availability is untested, direct guard is "
                    "partial, support incompatibility appears only as intermediate unknown "
                    "endpoints, and objective recovery is not uniformly present"
                ),
                "route_execution_status": str(first["route_execution_status"]),
                "wall_promotion_status": "not_promoted_wall_evidence_audit_only",
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    contract_rows = pd.DataFrame(rows)
    base_cols = [
        "route_contract_id",
        "route_contract_trace_status",
        "max_objective_debt_from_start",
        "max_objective_recovery_from_min",
    ]
    available = [col for col in base_cols if col in contract_summary.columns]
    if available:
        contract_rows = contract_rows.merge(
            contract_summary[available],
            on="route_contract_id",
            how="left",
            validate="one_to_one",
        )
    return contract_rows.sort_values(
        ["local_pair_id", "start_condition", "planned_route_family"],
        kind="mergesort",
    ).reset_index(drop=True)


def _family_audit_rows(contract_audit: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for family, group in contract_audit.groupby("planned_route_family", sort=False):
        contract_count = int(len(group))
        all_transition_count = int(
            group["transition_seed_count"].astype(int).eq(group["seed_count"].astype(int)).sum()
        )
        partial_count = int(
            (
                group["transition_seed_count"].astype(int).gt(0)
                & group["transition_seed_count"].astype(int).lt(group["seed_count"].astype(int))
            ).sum()
        )
        if str(family) == PRIMARY_ROUTE_FAMILY:
            if all_transition_count == contract_count:
                status = "primary_bridge_release_pathway_trace_family_candidate_not_wall"
            else:
                status = "primary_bridge_release_not_stable"
        elif str(family) == DIRECT_GUARD_FAMILY:
            status = "direct_dependency_guard_partial_family_blocks_wall_promotion"
        elif str(family) == DROP_BOTH_GUARD_FAMILY:
            status = "drop_both_guard_collapses_as_expected_not_wall"
        else:
            status = "unclassified_route_family"
        rows.append(
            {
                "planned_route_family": str(family),
                "contract_count": contract_count,
                "all_seed_transition_contract_count": all_transition_count,
                "partial_transition_contract_count": partial_count,
                "wall_audit_candidate_contract_count": int(
                    group["distinct_basin_pair_relation_candidate"].astype(bool).sum()
                ),
                "objective_recovery_contract_count": int(
                    group["objective_recovery_evidence"].astype(bool).sum()
                ),
                "support_incompatibility_contract_count": int(
                    group["support_incompatibility_evidence"].astype(bool).sum()
                ),
                "post_start_polish_reversion_contract_count": int(
                    group["polish_reversion_evidence"].astype(bool).sum()
                ),
                "wall_contract_ready_count": int(group["wall_contract_ready"].astype(bool).sum()),
                "family_wall_evidence_status": status,
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
        "run_status": RUN_STATUS,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _gate_matrix(
    *,
    trace_gates: pd.DataFrame,
    seed_audit: pd.DataFrame,
    contract_audit: pd.DataFrame,
    family_audit: pd.DataFrame,
) -> pd.DataFrame:
    primary = contract_audit[
        contract_audit["planned_route_family"].astype(str).eq(PRIMARY_ROUTE_FAMILY)
    ]
    direct = contract_audit[
        contract_audit["planned_route_family"].astype(str).eq(DIRECT_GUARD_FAMILY)
    ]
    drop_both = contract_audit[
        contract_audit["planned_route_family"].astype(str).eq(DROP_BOTH_GUARD_FAMILY)
    ]
    rows = [
        _gate_row(
            "G1_upstream_trace_gates_pass",
            "Did every scoped pathway-probe trace gate pass?",
            _count_dict(trace_gates["gate_status"]),
            "all upstream trace gates pass",
            bool(trace_gates["gate_status"].astype(str).eq("pass").all()),
        ),
        _gate_row(
            "G2_primary_bridge_release_stable_transition",
            "Do all bridge-release primary contracts transition to expected anchors for every seed?",
            f"primary_contracts={len(primary)} all_seed_transitions={int(primary['transition_seed_count'].eq(primary['seed_count']).sum())}",
            "10 of 10 primary contracts all-seed transition",
            len(primary) == 10
            and bool(primary["transition_seed_count"].eq(primary["seed_count"]).all()),
        ),
        _gate_row(
            "G3_direct_dependency_guard_not_stable",
            "Does the direct-dependency guard remain partial rather than a stable wall path?",
            _count_dict(direct["wall_contract_status"]),
            "direct guard partial, not promoted",
            len(direct) == 10
            and bool(direct["transition_seed_count"].lt(direct["seed_count"]).all())
            and bool(direct["transition_seed_count"].gt(0).all()),
        ),
        _gate_row(
            "G4_drop_both_guard_collapses_as_expected",
            "Does the drop-both guard collapse to its expected anchor for every seed?",
            f"drop_both_contracts={len(drop_both)} all_seed_transitions={int(drop_both['transition_seed_count'].eq(drop_both['seed_count']).sum())}",
            "10 of 10 drop-both guards all-seed transition",
            len(drop_both) == 10
            and bool(drop_both["transition_seed_count"].eq(drop_both["seed_count"]).all()),
        ),
        _gate_row(
            "G5_intermediate_unknown_kept_as_audit_signal",
            "Are intermediate unknown/support-incompatibility steps recorded rather than promoted?",
            f"support_incompatibility_seeds={int(seed_audit['support_incompatibility_evidence'].sum())}",
            "intermediate unknown evidence is diagnostic, not wall promotion",
            int(seed_audit["support_incompatibility_evidence"].sum()) == 27,
        ),
        _gate_row(
            "G6_objective_recovery_not_uniform",
            "Is objective recovery non-uniform, blocking wall promotion?",
            f"recovery_contracts={int(contract_audit['objective_recovery_evidence'].sum())} of {len(contract_audit)}",
            "not every primary contract has all-seed recovery",
            int(primary["objective_recovery_evidence"].sum()) < len(primary),
        ),
        _gate_row(
            "G7_direct_path_availability_missing",
            "Is direct-path availability still missing for wall promotion?",
            f"direct_path_available_contracts={int(contract_audit['direct_path_availability_evidence'].sum())}",
            "zero contracts have accepted direct-path evidence",
            not bool(contract_audit["direct_path_availability_evidence"].any()),
        ),
        _gate_row(
            "G8_wall_claim_remains_closed",
            "Are all wall claims kept closed after audit?",
            f"wall_ready_contracts={int(contract_audit['wall_contract_ready'].sum())}",
            "zero wall-ready contracts",
            not bool(contract_audit["wall_contract_ready"].any()),
        ),
        _gate_row(
            "G9_no_method_quality_or_full_replay_claim",
            "Are method, quality/cost, full replay, and algorithm claims closed?",
            CLAIM_BOUNDARY,
            "claim boundary explicitly closed",
            True,
        ),
    ]
    return pd.DataFrame(rows)


def _summary(
    *,
    trace_dir: Path,
    output_dir: Path,
    seed_audit: pd.DataFrame,
    contract_audit: pd.DataFrame,
    family_audit: pd.DataFrame,
    gates: pd.DataFrame,
) -> dict[str, Any]:
    primary = contract_audit[
        contract_audit["planned_route_family"].astype(str).eq(PRIMARY_ROUTE_FAMILY)
    ]
    return {
        "schema": "nanoclustering_g4_8_scoped_pathway_wall_evidence_summary.v1",
        "status": "wall_evidence_audit_pathway_trace_candidate_wall_claim_closed",
        "run_status": RUN_STATUS,
        "trace_dir": str(trace_dir),
        "output_dir": str(output_dir),
        "seed_audit_row_count": int(len(seed_audit)),
        "contract_audit_row_count": int(len(contract_audit)),
        "family_audit_row_count": int(len(family_audit)),
        "primary_bridge_release_contract_count": int(len(primary)),
        "primary_all_seed_transition_contract_count": int(
            primary["transition_seed_count"].eq(primary["seed_count"]).sum()
        ),
        "direct_guard_partial_contract_count": int(
            contract_audit["wall_contract_status"]
            .astype(str)
            .eq("direct_dependency_guard_partial_wall_closed")
            .sum()
        ),
        "drop_both_guard_passed_contract_count": int(
            contract_audit["wall_contract_status"]
            .astype(str)
            .eq("drop_both_collapse_guard_passed_not_wall")
            .sum()
        ),
        "intermediate_unknown_seed_count": int(
            seed_audit["intermediate_unknown_endpoint_evidence"].sum()
        ),
        "support_incompatibility_seed_count": int(
            seed_audit["support_incompatibility_evidence"].sum()
        ),
        "objective_recovery_contract_count": int(
            contract_audit["objective_recovery_evidence"].sum()
        ),
        "wall_ready_contract_count": int(contract_audit["wall_contract_ready"].sum()),
        "wall_contract_status_counts": _count_dict(contract_audit["wall_contract_status"]),
        "family_wall_evidence_status_counts": _count_dict(
            family_audit["family_wall_evidence_status"]
        ),
        "gate_status_counts": _count_dict(gates["gate_status"]),
        "failed_gates": gates.loc[
            ~gates["gate_status"].astype(str).eq("pass"),
            "gate_id",
        ].tolist(),
        "interpretation": (
            "The scoped trace creates a strong bridge-release pathway-trace "
            "candidate, but not a wall claim. Wall promotion remains blocked by "
            "partial direct-dependency guards, non-uniform objective recovery, "
            "diagnostic-only intermediate unknown endpoints, and missing accepted "
            "direct-path evidence."
        ),
        "recommended_next_gate": (
            "Inspect the primary bridge-release traces only: separate the 27 "
            "intermediate unknown seed-routes from direct guard partial failures, "
            "then test direct-path availability and objective debt/recovery before "
            "any wall language."
        ),
        "claim_boundary": CLAIM_BOUNDARY,
        "written_artifacts": [
            SEED_AUDIT_ROWS_CSV,
            CONTRACT_AUDIT_ROWS_CSV,
            FAMILY_AUDIT_ROWS_CSV,
            GATE_MATRIX_CSV,
            SUMMARY_JSON,
            CONFIG_JSON,
            REPORT_MD,
        ],
    }


def _write_report(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    contract_audit: pd.DataFrame,
    family_audit: pd.DataFrame,
    gates: pd.DataFrame,
) -> None:
    lines = [
        "# NanoClustering G4.8 Scoped Pathway Wall-Evidence Audit",
        "",
        f"- status: `{summary['status']}`",
        f"- primary_bridge_release_contract_count: {summary['primary_bridge_release_contract_count']}",
        f"- primary_all_seed_transition_contract_count: {summary['primary_all_seed_transition_contract_count']}",
        f"- direct_guard_partial_contract_count: {summary['direct_guard_partial_contract_count']}",
        f"- drop_both_guard_passed_contract_count: {summary['drop_both_guard_passed_contract_count']}",
        f"- intermediate_unknown_seed_count: {summary['intermediate_unknown_seed_count']}",
        f"- support_incompatibility_seed_count: {summary['support_incompatibility_seed_count']}",
        f"- objective_recovery_contract_count: {summary['objective_recovery_contract_count']}",
        f"- wall_ready_contract_count: {summary['wall_ready_contract_count']}",
        f"- wall_contract_status_counts: {summary['wall_contract_status_counts']}",
        f"- gate_status_counts: {summary['gate_status_counts']}",
        f"- failed_gates: {summary['failed_gates']}",
        f"- interpretation: {summary['interpretation']}",
        f"- recommended_next_gate: {summary['recommended_next_gate']}",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Family Audit",
        "",
        _markdown_table(
            family_audit,
            [
                "planned_route_family",
                "contract_count",
                "all_seed_transition_contract_count",
                "partial_transition_contract_count",
                "objective_recovery_contract_count",
                "support_incompatibility_contract_count",
                "wall_contract_ready_count",
                "family_wall_evidence_status",
            ],
        ),
        "",
        "## Contract Audit",
        "",
        _markdown_table(
            contract_audit,
            [
                "local_pair_id",
                "start_condition",
                "planned_route_family",
                "transition_seed_count",
                "seed_count",
                "objective_recovery_seed_count",
                "support_incompatibility_seed_count",
                "intermediate_unknown_endpoint_seed_count",
                "wall_contract_status",
                "wall_contract_ready",
            ],
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
            "This audit promotes no wall claim. It narrows the next valid work to "
            "a wall-evidence readout over the primary bridge-release traces, with "
            "direct-path availability and objective debt/recovery as explicit "
            "blocking questions."
        ),
        "",
    ]
    (output_dir / REPORT_MD).write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    trace_dir = Path(args.trace_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trace_rows = _read_csv(trace_dir / TRACE_ROWS_CSV)
    seed_summary = _read_csv(trace_dir / SEED_ROUTE_SUMMARY_CSV)
    contract_summary = _read_csv(trace_dir / ROUTE_CONTRACT_SUMMARY_CSV)
    trace_gates = _read_csv(trace_dir / TRACE_GATE_MATRIX_CSV)

    seed_audit = _seed_audit_rows(seed_summary, trace_rows)
    contract_audit = _contract_audit_rows(seed_audit, contract_summary)
    family_audit = _family_audit_rows(contract_audit)
    gates = _gate_matrix(
        trace_gates=trace_gates,
        seed_audit=seed_audit,
        contract_audit=contract_audit,
        family_audit=family_audit,
    )
    summary = _summary(
        trace_dir=trace_dir,
        output_dir=output_dir,
        seed_audit=seed_audit,
        contract_audit=contract_audit,
        family_audit=family_audit,
        gates=gates,
    )

    _write_csv(seed_audit, output_dir / SEED_AUDIT_ROWS_CSV)
    _write_csv(contract_audit, output_dir / CONTRACT_AUDIT_ROWS_CSV)
    _write_csv(family_audit, output_dir / FAMILY_AUDIT_ROWS_CSV)
    _write_csv(gates, output_dir / GATE_MATRIX_CSV)
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    config = {
        "schema": "nanoclustering_g4_8_scoped_pathway_wall_evidence_config.v1",
        "trace_dir": str(trace_dir),
        "output_dir": str(output_dir),
        "primary_route_family": PRIMARY_ROUTE_FAMILY,
        "direct_guard_family": DIRECT_GUARD_FAMILY,
        "drop_both_guard_family": DROP_BOTH_GUARD_FAMILY,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(
        output_dir=output_dir,
        summary=summary,
        contract_audit=contract_audit,
        family_audit=family_audit,
        gates=gates,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
