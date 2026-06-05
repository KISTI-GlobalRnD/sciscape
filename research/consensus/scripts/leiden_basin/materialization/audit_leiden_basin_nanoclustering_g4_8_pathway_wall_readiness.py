#!/usr/bin/env python3
"""Audit pathway/wall readiness for the G4.8 stable primary surface.

This consumes the primary stable limitation readout and asks whether the narrow
ready signal is mature enough for the second-stage pathway/wall question. It
separates pathway-probe candidates from wall-claim evidence: ready rows may
enter a later predeclared pathway probe, but no row can support a wall claim
without route traces, objective debt/recovery, polish reversion, and measured
post-route endpoint assignment.

It does not run Leiden, execute route/pathway traces, promote walls, evaluate
wall-clock quality/cost value, replay full NanoClustering, or claim method or
algorithm success.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from run_leiden_basin_nanoclustering_role_local_route_pilot import (
    BASE_RESULT_DIR,
    _json_safe,
    _read_csv,
    _write_csv,
)


DEFAULT_LIMIT_READOUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_primary_stable_limit_readout_gamma1e5_20260604"
)
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_pathway_wall_readiness_audit_gamma1e5_20260604"
)

INPUT_UNIT_ROWS_CSV = "nanoclustering_g4_8_primary_stable_limit_readout_unit_rows.csv"
INPUT_PAIR_ROWS_CSV = "nanoclustering_g4_8_primary_stable_limit_readout_pair_rows.csv"
INPUT_GATE_MATRIX_CSV = "nanoclustering_g4_8_primary_stable_limit_readout_gate_matrix.csv"

AUDIT_UNIT_ROWS_CSV = "nanoclustering_g4_8_pathway_wall_readiness_audit_unit_rows.csv"
AUDIT_PAIR_ROWS_CSV = "nanoclustering_g4_8_pathway_wall_readiness_audit_pair_rows.csv"
AUDIT_STATUS_SUMMARY_CSV = (
    "nanoclustering_g4_8_pathway_wall_readiness_audit_status_summary.csv"
)
GATE_MATRIX_CSV = "nanoclustering_g4_8_pathway_wall_readiness_audit_gate_matrix.csv"
CONFIG_JSON = "nanoclustering_g4_8_pathway_wall_readiness_audit_config.json"
SUMMARY_JSON = "nanoclustering_g4_8_pathway_wall_readiness_audit_summary.json"
REPORT_MD = "nanoclustering_g4_8_pathway_wall_readiness_audit_report.md"

START_CONDITIONS = (
    "singleton",
    "pair_together",
    "bridges_to_left",
    "bridges_to_right",
    "all_local_together",
)

WALL_REQUIRED_MISSING_FIELDS = (
    "accepted_distinct_basin_pair_relation",
    "route_family",
    "direct_path_availability",
    "objective_debt_evidence",
    "debt_recovery_evidence",
    "polish_reversion_evidence",
    "support_incompatibility_evidence",
    "post_route_endpoint_assignment",
)

RUN_STATUS = "materialized_nanoclustering_g4_8_pathway_wall_readiness_audit"
CLAIM_BOUNDARY = (
    "NanoClustering G4.8 pathway/wall readiness audit only; reads the primary "
    "stable limitation readout and separates pathway-probe candidates from "
    "wall-claim evidence. It does not run Leiden, execute route/pathway traces, "
    "promote walls, evaluate wall-clock quality/cost value, replay full "
    "NanoClustering, or claim method or algorithm success."
)


def _count_dict(series: pd.Series) -> dict[str, int]:
    if series.empty:
        return {}
    return {str(key): int(value) for key, value in series.value_counts(dropna=False).items()}


def _prefix_stats(prefix: str, values: pd.Series | np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            f"{prefix}_min": None,
            f"{prefix}_median": None,
            f"{prefix}_max": None,
            f"{prefix}_mean": None,
        }
    return {
        f"{prefix}_min": float(array.min()),
        f"{prefix}_median": float(np.median(array)),
        f"{prefix}_max": float(array.max()),
        f"{prefix}_mean": float(array.mean()),
    }


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    cols = [col for col in columns if col in frame.columns]
    if not cols:
        return "No columns."
    header = "| " + " | ".join(cols) + " |"
    separator = "| " + " | ".join("---" for _ in cols) + " |"
    rows: list[str] = []
    for row in frame[cols].itertuples(index=False):
        values: list[str] = []
        for value in row:
            if pd.isna(value):
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value).replace("\n", " "))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def _bool_series(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(bool)


def _has_ready_tri_endpoint_contrast(row: pd.Series) -> bool:
    original = float(row["original_pair_coassigned_share"])
    drop_direct = float(row["drop_direct_pair_coassigned_share"])
    drop_bridge = float(row["drop_bridge_pair_coassigned_share"])
    drop_both = float(row["drop_direct_and_bridge_pair_coassigned_share"])
    return bool(
        0.0 < original < 0.95
        and drop_direct == 0.0
        and drop_bridge >= 0.95
        and drop_both == 0.0
        and float(row["bridge_release_lift_proxy"]) > 0.0
        and float(row["direct_dependency_proxy"]) > 0.0
        and float(row["original_source_endpoint_signature_proxy_count"]) > 0.0
        and float(row["original_coassigned_signature_count"]) > 0.0
    )


def _readiness_fields(row: pd.Series) -> dict[str, Any]:
    axis = str(row["limitation_axis"])
    if axis == "ready_partial_release":
        candidate = _has_ready_tri_endpoint_contrast(row)
        return {
            "pathway_readiness_status": (
                "pathway_probe_candidate_not_wall_claim"
                if candidate
                else "ready_row_failed_pathway_probe_precheck"
            ),
            "pathway_probe_candidate": bool(candidate),
            "pathway_probe_block_reason": "" if candidate else "ready row lacks tri-endpoint contrast",
            "wall_claim_ready": False,
            "wall_claim_block_reason": "missing optimizer-native route trace evidence",
            "readiness_interpretation": (
                "Source-like original partial coassignment, bridge-removal target "
                "coassignment, and direct-removal collapse define a candidate "
                "pathway-probe contrast, but not wall evidence."
            ),
        }
    if axis == "target_saturated_no_handle":
        return {
            "pathway_readiness_status": "blocked_target_saturated_false_positive_control",
            "pathway_probe_candidate": False,
            "pathway_probe_block_reason": "original endpoint is already target coassigned",
            "wall_claim_ready": False,
            "wall_claim_block_reason": "no distinct source handle and no route trace",
            "readiness_interpretation": (
                "This is a target-saturation control: release contrast is absent "
                "because original and bridge-removal endpoints are both target-like."
            ),
        }
    if axis == "latent_release_without_original_coassigned_source":
        return {
            "pathway_readiness_status": "blocked_latent_release_no_original_source",
            "pathway_probe_candidate": False,
            "pathway_probe_block_reason": (
                "bridge release exists but original coassigned source proxy is absent"
            ),
            "wall_claim_ready": False,
            "wall_claim_block_reason": "no accepted source basin and no route trace",
            "readiness_interpretation": (
                "This is useful release-control evidence, but not a pathway "
                "candidate because the original local endpoint has no coassigned "
                "source state."
            ),
        }
    if axis == "hard_no_release_control":
        return {
            "pathway_readiness_status": "blocked_hard_no_release_control",
            "pathway_probe_candidate": False,
            "pathway_probe_block_reason": "no bridge-release coassignment observed",
            "wall_claim_ready": False,
            "wall_claim_block_reason": "no release contrast and no route trace",
            "readiness_interpretation": "Hard negative control for pathway false positives.",
        }
    if axis == "coupled_direct_bridge_failure":
        return {
            "pathway_readiness_status": "blocked_coupled_direct_bridge_failure",
            "pathway_probe_candidate": False,
            "pathway_probe_block_reason": (
                "direct and bridge context are coupled; bridge removal destroys target"
            ),
            "wall_claim_ready": False,
            "wall_claim_block_reason": "support contrast is coupled and no route trace exists",
            "readiness_interpretation": (
                "Coupled-failure control: removing bridge support collapses the "
                "coassigned endpoint rather than opening a clean pathway contrast."
            ),
        }
    return {
        "pathway_readiness_status": "blocked_unclassified_limit",
        "pathway_probe_candidate": False,
        "pathway_probe_block_reason": "unclassified limitation axis",
        "wall_claim_ready": False,
        "wall_claim_block_reason": "unclassified limitation axis and no route trace",
        "readiness_interpretation": "Inspect before use.",
    }


def _audit_units(unit_rows: pd.DataFrame) -> pd.DataFrame:
    rows = unit_rows.copy()
    fields = rows.apply(_readiness_fields, axis=1)
    for key in [
        "pathway_readiness_status",
        "pathway_probe_candidate",
        "pathway_probe_block_reason",
        "wall_claim_ready",
        "wall_claim_block_reason",
        "readiness_interpretation",
    ]:
        rows[key] = [item[key] for item in fields]
    rows["ready_tri_endpoint_contrast_pass"] = rows.apply(
        _has_ready_tri_endpoint_contrast, axis=1
    )
    rows.loc[~rows["limitation_axis"].eq("ready_partial_release"), "ready_tri_endpoint_contrast_pass"] = False
    rows["accepted_distinct_basin_pair_relation_available"] = False
    rows["optimizer_native_route_trace_available"] = False
    rows["objective_debt_evidence_available"] = False
    rows["debt_recovery_evidence_available"] = False
    rows["polish_reversion_evidence_available"] = False
    rows["support_incompatibility_evidence_available"] = False
    rows["post_route_endpoint_assignment_available"] = False
    rows["missing_for_wall_claim"] = ";".join(WALL_REQUIRED_MISSING_FIELDS)
    rows["stage2_entry_role"] = np.where(
        rows["pathway_probe_candidate"],
        "stage2a_pathway_probe_entry_candidate",
        "stage2a_false_positive_or_limit_control",
    )
    rows["run_status"] = RUN_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows.sort_values(
        ["pathway_probe_candidate", "limitation_axis", "local_pair_id", "start_condition"],
        ascending=[False, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)


def _audit_pairs(unit_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metric_cols = [
        "original_pair_coassigned_share",
        "drop_direct_pair_coassigned_share",
        "drop_bridge_pair_coassigned_share",
        "drop_direct_and_bridge_pair_coassigned_share",
        "bridge_release_lift_proxy",
        "direct_dependency_proxy",
        "original_source_endpoint_signature_proxy_count",
        "original_coassigned_signature_count",
        "original_distinct_endpoint_count",
    ]
    for local_pair_id, group in unit_rows.groupby("local_pair_id", sort=False):
        first = group.iloc[0]
        candidate_count = int(group["pathway_probe_candidate"].sum())
        wall_ready_count = int(group["wall_claim_ready"].sum())
        row: dict[str, Any] = {
            "local_pair_id": str(local_pair_id),
            "branch": first.get("branch"),
            "left_node_id": first.get("left_node_id"),
            "right_node_id": first.get("right_node_id"),
            "limitation_axis": first.get("limitation_axis"),
            "validation_contract_class": first.get("validation_contract_class"),
            "pathway_readiness_status": (
                "pair_pathway_probe_candidate_not_wall_claim"
                if candidate_count == len(group)
                else "pair_control_or_blocked_for_pathway_probe"
            ),
            "pathway_probe_candidate_unit_count": candidate_count,
            "wall_claim_ready_unit_count": wall_ready_count,
            "unit_count": int(len(group)),
            "start_condition_count": int(group["start_condition"].nunique()),
            "start_conditions": ";".join(sorted(group["start_condition"].astype(str))),
            "pathway_probe_candidate_pair": bool(candidate_count == len(group)),
            "wall_claim_ready_pair": False,
            "missing_for_wall_claim": ";".join(WALL_REQUIRED_MISSING_FIELDS),
            "pathway_probe_block_reasons": ";".join(
                sorted(
                    {
                        str(value)
                        for value in group["pathway_probe_block_reason"]
                        if str(value)
                    }
                )
            ),
            "run_status": RUN_STATUS,
            "claim_boundary": CLAIM_BOUNDARY,
        }
        for col in metric_cols:
            if col in group.columns:
                row.update(_prefix_stats(col, group[col]))
        rows.append(row)
    return pd.DataFrame(rows)


def _status_summary(unit_rows: pd.DataFrame, pair_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for status, group in unit_rows.groupby("pathway_readiness_status", sort=True):
        pair_group = pair_rows[pair_rows["local_pair_id"].isin(group["local_pair_id"])]
        rows.append(
            {
                "pathway_readiness_status": str(status),
                "unit_count": int(len(group)),
                "pair_count": int(pair_group["local_pair_id"].nunique()),
                "pathway_probe_candidate_unit_count": int(
                    group["pathway_probe_candidate"].sum()
                ),
                "wall_claim_ready_unit_count": int(group["wall_claim_ready"].sum()),
                "limitation_axis_counts": json.dumps(
                    _count_dict(group["limitation_axis"]),
                    sort_keys=True,
                ),
                "start_condition_counts": json.dumps(
                    _count_dict(group["start_condition"]),
                    sort_keys=True,
                ),
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows)


def _gate_row(
    gate_id: str, question: str, passed: bool, observed: Any, minimum: Any
) -> dict[str, Any]:
    return {
        "gate_id": gate_id,
        "question": question,
        "gate_status": "pass" if bool(passed) else "fail",
        "observed": observed,
        "minimum_or_rule": minimum,
        "run_status": RUN_STATUS,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _build_gate_matrix(
    *,
    unit_rows: pd.DataFrame,
    pair_rows: pd.DataFrame,
    upstream_gates: pd.DataFrame,
) -> pd.DataFrame:
    candidates = unit_rows[unit_rows["pathway_probe_candidate"]]
    non_candidates = unit_rows[~unit_rows["pathway_probe_candidate"]]
    ready_units = unit_rows[unit_rows["limitation_axis"].eq("ready_partial_release")]
    target_units = unit_rows[unit_rows["limitation_axis"].eq("target_saturated_no_handle")]
    latent_units = unit_rows[
        unit_rows["limitation_axis"].eq("latent_release_without_original_coassigned_source")
    ]
    hard_units = unit_rows[unit_rows["limitation_axis"].eq("hard_no_release_control")]
    coupled_units = unit_rows[unit_rows["limitation_axis"].eq("coupled_direct_bridge_failure")]
    rows = [
        _gate_row(
            "G1_upstream_limit_readout_passes",
            "Did every upstream primary stable limitation readout gate pass?",
            bool(upstream_gates["gate_status"].astype(str).eq("pass").all()),
            _count_dict(upstream_gates["gate_status"]),
            "all upstream gates pass",
        ),
        _gate_row(
            "G2_ready_units_become_pathway_probe_candidates",
            "Do exactly the ready units become pathway-probe candidates?",
            int(len(candidates)) == 10
            and int(candidates["local_pair_id"].nunique()) == 2
            and bool(candidates["limitation_axis"].eq("ready_partial_release").all()),
            (
                f"candidate_units={len(candidates)} "
                f"candidate_pairs={candidates['local_pair_id'].nunique()}"
            ),
            "10 ready units across 2 ready pairs",
        ),
        _gate_row(
            "G3_ready_tri_endpoint_contrast_passes",
            "Do ready candidates have source/target/collapse contrast under local ablations?",
            bool(ready_units["ready_tri_endpoint_contrast_pass"].all()),
            json.dumps(_count_dict(ready_units["ready_tri_endpoint_contrast_pass"]), sort_keys=True),
            "original partial, drop-bridge target, drop-direct collapse",
        ),
        _gate_row(
            "G4_controls_do_not_become_pathway_candidates",
            "Do all non-ready limitation/control units remain outside pathway-probe candidates?",
            int(len(non_candidates)) == 65
            and not bool(non_candidates["limitation_axis"].eq("ready_partial_release").any()),
            f"non_candidate_units={len(non_candidates)}",
            "65 non-ready units excluded",
        ),
        _gate_row(
            "G5_target_saturation_blocked",
            "Are target-saturated controls blocked as already-target false positives?",
            int(len(target_units)) == 25
            and not bool(target_units["pathway_probe_candidate"].any())
            and bool(target_units["original_pair_coassigned_share"].astype(float).ge(0.95).all()),
            f"target_units={len(target_units)}",
            "25 target-saturated units blocked",
        ),
        _gate_row(
            "G6_latent_release_without_source_blocked",
            "Are latent-release controls blocked because original coassigned source is absent?",
            int(len(latent_units)) == 20
            and not bool(latent_units["pathway_probe_candidate"].any())
            and bool(latent_units["original_coassigned_signature_count"].astype(float).eq(0).all()),
            f"latent_units={len(latent_units)}",
            "20 latent-release controls blocked",
        ),
        _gate_row(
            "G7_hard_no_release_blocked",
            "Are hard no-release controls blocked?",
            int(len(hard_units)) == 10
            and not bool(hard_units["pathway_probe_candidate"].any())
            and bool(hard_units["drop_bridge_pair_coassigned_share"].astype(float).eq(0).all()),
            f"hard_units={len(hard_units)}",
            "10 hard no-release controls blocked",
        ),
        _gate_row(
            "G8_coupled_failure_blocked",
            "Are coupled direct/bridge failures blocked?",
            int(len(coupled_units)) == 10
            and not bool(coupled_units["pathway_probe_candidate"].any())
            and bool(coupled_units["bridge_release_lift_proxy"].astype(float).lt(0).all()),
            f"coupled_units={len(coupled_units)}",
            "10 coupled-failure controls blocked",
        ),
        _gate_row(
            "G9_wall_claim_gate_remains_closed",
            "Are all wall claims kept closed because route evidence is missing?",
            not bool(unit_rows["wall_claim_ready"].any())
            and not bool(pair_rows["wall_claim_ready_pair"].any()),
            "wall_claim_ready=false for all rows",
            "no route trace, debt/recovery, polish reversion, or post-route endpoint assignment",
        ),
        _gate_row(
            "G10_no_new_leiden_execution",
            "Is this a read-only audit over existing limitation rows rather than a new run?",
            True,
            RUN_STATUS,
            "read-only audit only",
        ),
        _gate_row(
            "G11_no_method_quality_or_replay_claim",
            "Are method, quality/cost, full replay, and algorithm claims closed?",
            True,
            CLAIM_BOUNDARY,
            "claim boundary explicitly closed",
        ),
    ]
    return pd.DataFrame(rows)


def _audit_status(gate_matrix: pd.DataFrame) -> str:
    if gate_matrix.empty or not bool(gate_matrix["gate_status"].astype(str).eq("pass").all()):
        return "pathway_wall_readiness_audit_gate_failed"
    return "pathway_probe_ready_for_scoped_candidates_wall_claim_closed"


def _build_summary(
    *,
    unit_rows: pd.DataFrame,
    pair_rows: pd.DataFrame,
    status_summary: pd.DataFrame,
    gate_matrix: pd.DataFrame,
    limit_readout_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    candidates = unit_rows[unit_rows["pathway_probe_candidate"]]
    return {
        "schema": "nanoclustering_g4_8_pathway_wall_readiness_audit_summary.v1",
        "status": _audit_status(gate_matrix),
        "run_status": RUN_STATUS,
        "claim_boundary": CLAIM_BOUNDARY,
        "limit_readout_dir": str(limit_readout_dir),
        "output_dir": str(output_dir),
        "unit_count": int(len(unit_rows)),
        "pair_count": int(len(pair_rows)),
        "pathway_probe_candidate_unit_count": int(len(candidates)),
        "pathway_probe_candidate_pair_count": int(candidates["local_pair_id"].nunique()),
        "wall_claim_ready_unit_count": int(unit_rows["wall_claim_ready"].sum()),
        "pathway_readiness_status_counts": _count_dict(unit_rows["pathway_readiness_status"]),
        "limitation_axis_counts": _count_dict(unit_rows["limitation_axis"]),
        "status_summary_rows": int(len(status_summary)),
        "gate_status_counts": _count_dict(gate_matrix["gate_status"]),
        "failed_gates": [
            str(row.gate_id)
            for row in gate_matrix.itertuples(index=False)
            if str(row.gate_status) != "pass"
        ],
        "wall_required_missing_fields": WALL_REQUIRED_MISSING_FIELDS,
        "interpretation": (
            "Stage 2A can open a scoped pathway-probe design for 2 ready pairs "
            "and 10 units. It cannot open wall claims: all rows still lack "
            "route traces, objective debt/recovery, polish reversion, support "
            "incompatibility, and post-route endpoint assignment."
        ),
        "recommended_next_gate": (
            "Design a predeclared tiny pathway-probe contract only for the two "
            "ready pairs. Keep target-saturated, latent-release, hard no-release, "
            "and coupled-failure rows as false-positive controls. Do not run a "
            "broad route batch or make quality/cost/method claims."
        ),
        "written_artifacts": [
            AUDIT_UNIT_ROWS_CSV,
            AUDIT_PAIR_ROWS_CSV,
            AUDIT_STATUS_SUMMARY_CSV,
            GATE_MATRIX_CSV,
            CONFIG_JSON,
            SUMMARY_JSON,
            REPORT_MD,
        ],
    }


def _write_report(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    pair_rows: pd.DataFrame,
    status_summary: pd.DataFrame,
    gate_matrix: pd.DataFrame,
) -> None:
    candidate_pairs = pair_rows[pair_rows["pathway_probe_candidate_pair"]]
    control_pairs = pair_rows[~pair_rows["pathway_probe_candidate_pair"]]
    lines = [
        "# NanoClustering G4.8 Pathway/Wall Readiness Audit",
        "",
        f"- status: `{summary['status']}`",
        f"- pair_count: {summary['pair_count']}",
        f"- unit_count: {summary['unit_count']}",
        f"- pathway_probe_candidate_pair_count: {summary['pathway_probe_candidate_pair_count']}",
        f"- pathway_probe_candidate_unit_count: {summary['pathway_probe_candidate_unit_count']}",
        f"- wall_claim_ready_unit_count: {summary['wall_claim_ready_unit_count']}",
        f"- pathway_readiness_status_counts: {summary['pathway_readiness_status_counts']}",
        f"- gate_status_counts: {summary['gate_status_counts']}",
        f"- failed_gates: {summary['failed_gates']}",
        f"- wall_required_missing_fields: {summary['wall_required_missing_fields']}",
        f"- interpretation: {summary['interpretation']}",
        f"- recommended_next_gate: {summary['recommended_next_gate']}",
        f"- claim_boundary: {summary['claim_boundary']}",
        "",
        "## Status Summary",
        "",
    ]
    lines.append(
        _markdown_table(
            status_summary,
            [
                "pathway_readiness_status",
                "pair_count",
                "unit_count",
                "pathway_probe_candidate_unit_count",
                "wall_claim_ready_unit_count",
                "limitation_axis_counts",
            ],
        )
    )
    lines.extend(["", "## Pathway-Probe Candidate Pairs", ""])
    lines.append(
        _markdown_table(
            candidate_pairs,
            [
                "local_pair_id",
                "pathway_readiness_status",
                "pathway_probe_candidate_unit_count",
                "unit_count",
                "original_pair_coassigned_share_median",
                "drop_bridge_pair_coassigned_share_median",
                "bridge_release_lift_proxy_median",
                "direct_dependency_proxy_median",
                "wall_claim_ready_pair",
            ],
        )
    )
    lines.extend(["", "## Control And Blocked Pairs", ""])
    lines.append(
        _markdown_table(
            control_pairs,
            [
                "local_pair_id",
                "limitation_axis",
                "pathway_probe_block_reasons",
                "pathway_probe_candidate_unit_count",
                "unit_count",
                "wall_claim_ready_pair",
            ],
        )
    )
    lines.extend(["", "## Gate Matrix", ""])
    lines.append(
        _markdown_table(
            gate_matrix,
            ["gate_id", "gate_status", "observed", "minimum_or_rule", "question"],
        )
    )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This audit opens only a scoped Stage 2A pathway-probe design gate for "
            "the two ready pairs. It does not promote wall evidence: all rows "
            "lack optimizer-native route traces, objective debt/recovery, polish "
            "reversion, support incompatibility, and measured post-route endpoint "
            "assignment. Control rows remain false-positive guards.",
            "",
        ]
    )
    (output_dir / REPORT_MD).write_text("\n".join(lines), encoding="utf-8")


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    limit_readout_dir = Path(args.limit_readout_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    unit_input = _read_csv(limit_readout_dir / INPUT_UNIT_ROWS_CSV)
    upstream_gates = _read_csv(limit_readout_dir / INPUT_GATE_MATRIX_CSV)

    unit_rows = _audit_units(unit_input)
    pair_rows = _audit_pairs(unit_rows)
    status_summary = _status_summary(unit_rows, pair_rows)
    gate_matrix = _build_gate_matrix(
        unit_rows=unit_rows,
        pair_rows=pair_rows,
        upstream_gates=upstream_gates,
    )
    summary = _build_summary(
        unit_rows=unit_rows,
        pair_rows=pair_rows,
        status_summary=status_summary,
        gate_matrix=gate_matrix,
        limit_readout_dir=limit_readout_dir,
        output_dir=output_dir,
    )

    _write_csv(unit_rows, output_dir / AUDIT_UNIT_ROWS_CSV)
    _write_csv(pair_rows, output_dir / AUDIT_PAIR_ROWS_CSV)
    _write_csv(status_summary, output_dir / AUDIT_STATUS_SUMMARY_CSV)
    _write_csv(gate_matrix, output_dir / GATE_MATRIX_CSV)
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(
            _json_safe(
                {
                    "limit_readout_dir": str(limit_readout_dir),
                    "output_dir": str(output_dir),
                    "start_conditions": START_CONDITIONS,
                    "wall_required_missing_fields": WALL_REQUIRED_MISSING_FIELDS,
                    "run_status": RUN_STATUS,
                    "claim_boundary": CLAIM_BOUNDARY,
                }
            ),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(
        output_dir=output_dir,
        summary=summary,
        pair_rows=pair_rows,
        status_summary=status_summary,
        gate_matrix=gate_matrix,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit-readout-dir", type=Path, default=DEFAULT_LIMIT_READOUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    summary = run_audit(parse_args())
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
