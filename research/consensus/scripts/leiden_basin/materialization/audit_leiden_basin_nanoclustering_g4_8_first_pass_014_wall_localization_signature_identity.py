#!/usr/bin/env python3
"""Audit signature identity in the first-pass 014 wall-localization trace.

This read-only audit checks a blind spot in the transition-band readout:
``endpoint_object_assignment_by_step`` is a row-local anchor match. The same
``result_endpoint_signature_id`` can therefore appear as unknown in one
start/seed context and as a known source-like or guard object in another. This
audit separates row-local unknowns from signature-level unresolved intermediate
objects.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from audit_leiden_basin_nanoclustering_g4_8_first_pass_014_wall_localization_transition_bands import (
    DEFAULT_OUTPUT_DIR as DEFAULT_TRANSITION_BAND_AUDIT_DIR,
    PAIR_BAND_ROWS_CSV as TRANSITION_PAIR_BAND_ROWS_CSV,
    SEED_BAND_ROWS_CSV as TRANSITION_SEED_BAND_ROWS_CSV,
)
from run_leiden_basin_nanoclustering_g4_8_first_pass_014_wall_localization_trace import (
    BOUNDARY_PAIR_ID,
    POSITIVE_PAIR_ID,
    SOURCE_OBJECTS,
    TARGET_OBJECT,
    TRACE_ROWS_CSV,
    DEFAULT_OUTPUT_DIR as DEFAULT_TRACE_DIR,
)
from run_leiden_basin_nanoclustering_role_local_route_pilot import (
    BASE_RESULT_DIR,
    _json_safe,
    _read_csv,
    _write_csv,
)


DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_014_wall_localization_signature_identity_audit_gamma1e5_20260605"
)

SIGNATURE_IDENTITY_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_014_wall_localization_signature_identity_rows.csv"
)
SEED_SIGNATURE_AUDIT_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_014_wall_localization_signature_identity_seed_rows.csv"
)
PAIR_SIGNATURE_AUDIT_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_014_wall_localization_signature_identity_pair_rows.csv"
)
GATE_MATRIX_CSV = (
    "nanoclustering_g4_8_first_pass_014_wall_localization_signature_identity_gate_matrix.csv"
)
SUMMARY_JSON = (
    "nanoclustering_g4_8_first_pass_014_wall_localization_signature_identity_summary.json"
)
CONFIG_JSON = (
    "nanoclustering_g4_8_first_pass_014_wall_localization_signature_identity_config.json"
)
REPORT_MD = (
    "nanoclustering_g4_8_first_pass_014_wall_localization_signature_identity_report.md"
)

RUN_STATUS = (
    "audited_nanoclustering_g4_8_first_pass_014_wall_localization_signature_identity"
)
CLAIM_BOUNDARY = (
    "NanoClustering G4.8 first-pass local_pair_014 signature-identity audit "
    "only; reads the executed localization trace and separates row-local "
    "unknown endpoint assignments from signature-level unresolved intermediate "
    "objects. It does not promote wall generality, evaluate quality/cost value, "
    "replay full NanoClustering, or claim method success."
)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _count_dict(series: pd.Series) -> dict[str, int]:
    if series.empty:
        return {}
    return {str(key): int(value) for key, value in series.value_counts(dropna=False).items()}


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = [str(column) for column in frame.columns]

    def cell(value: Any) -> str:
        if pd.isna(value):
            return ""
        return str(value).replace("|", "\\|")

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False):
        lines.append("| " + " | ".join(cell(value) for value in row) + " |")
    return "\n".join(lines)


def _object_family(endpoint_object: Any) -> str:
    value = str(endpoint_object)
    if value in SOURCE_OBJECTS:
        return "source_like"
    if value == TARGET_OBJECT:
        return "positive_target"
    if value == "boundary_target_endpoint_object_not_positive":
        return "boundary_target"
    if value in {
        "direct_drop_guard_endpoint_object",
        "drop_both_guard_endpoint_object",
    }:
        return "guard_intermediate"
    if "unknown" in value or "ambiguous" in value:
        return "row_local_unresolved"
    return "other_known"


def _is_row_local_unresolved(value: Any) -> bool:
    family = _object_family(value)
    return family == "row_local_unresolved"


def _signature_status(families: set[str], local_pair_id: str) -> str:
    known = families - {"row_local_unresolved"}
    has_unresolved = "row_local_unresolved" in families
    if has_unresolved and known:
        return "row_local_unresolved_but_signature_known_elsewhere"
    if has_unresolved and not known:
        if local_pair_id == POSITIVE_PAIR_ID:
            return "signature_level_unresolved_positive_intermediate"
        return "signature_level_unresolved_boundary_intermediate"
    if len(known) > 1:
        return "mixed_known_signature_roles"
    if "positive_target" in known:
        return "stable_positive_target_signature"
    if "boundary_target" in known:
        return "stable_boundary_target_signature"
    if "source_like" in known:
        return "stable_source_like_signature"
    if "guard_intermediate" in known:
        return "stable_guard_intermediate_signature"
    return "stable_other_known_signature"


def _signature_identity_rows(trace_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    trace = trace_rows.copy()
    trace["object_family"] = trace["endpoint_object_assignment_by_step"].map(_object_family)
    for (local_pair_id, signature_id), group in trace.groupby(
        ["local_pair_id", "result_endpoint_signature_id"],
        sort=True,
    ):
        families = set(group["object_family"].astype(str))
        endpoint_objects = sorted(set(group["endpoint_object_assignment_by_step"].astype(str)))
        known_families = sorted(families - {"row_local_unresolved"})
        status = _signature_status(families, str(local_pair_id))
        row_local_unresolved_count = int(
            group["endpoint_object_assignment_by_step"].map(_is_row_local_unresolved).sum()
        )
        rows.append(
            {
                "local_pair_id": str(local_pair_id),
                "result_endpoint_signature_id": str(signature_id),
                "signature_row_count": int(len(group)),
                "observed_endpoint_object_assignments": ";".join(endpoint_objects),
                "observed_object_families": ";".join(sorted(families)),
                "known_object_families": ";".join(known_families),
                "row_local_unresolved_row_count": row_local_unresolved_count,
                "signature_known_elsewhere": bool(row_local_unresolved_count and known_families),
                "signature_level_unresolved": bool(
                    row_local_unresolved_count and not known_families
                ),
                "signature_identity_status": status,
                "start_conditions": ";".join(sorted(set(group["start_condition"].astype(str)))),
                "seeds": ";".join(
                    str(value) for value in sorted(set(group["seed"].astype(int)))
                ),
                "route_family_roles": ";".join(
                    sorted(set(group["route_family_role"].astype(str)))
                ),
                "bridge_fractions": ";".join(
                    f"{value:.3g}"
                    for value in sorted(set(group["bridge_edge_weight_fraction"].astype(float)))
                ),
                "objective_value_mean": float(group["objective_value_by_step"].mean()),
                "objective_value_min": float(group["objective_value_by_step"].min()),
                "objective_value_max": float(group["objective_value_by_step"].max()),
                "pair_coassigned_rate": float(group["pair_coassigned"].mean()),
                "pair_bridge_same_cluster_mean": float(
                    group["pair_bridge_same_cluster_count"].mean()
                ),
                "left_bridge_same_cluster_mean": float(
                    group["left_bridge_same_cluster_count"].mean()
                ),
                "right_bridge_same_cluster_mean": float(
                    group["right_bridge_same_cluster_count"].mean()
                ),
                "support_distance_min_known_anchor_min": float(
                    group["support_distance_min_known_anchor"].min()
                ),
                "support_distance_min_known_anchor_max": float(
                    group["support_distance_min_known_anchor"].max()
                ),
                "method_claim_allowed_after_audit": False,
                "quality_cost_claim_allowed_after_audit": False,
                "wall_generality_claim_allowed_after_audit": False,
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows)


def _seed_signature_rows(
    trace_rows: pd.DataFrame,
    signature_rows: pd.DataFrame,
    transition_seed_rows: pd.DataFrame,
) -> pd.DataFrame:
    status_by_signature = signature_rows.set_index(
        ["local_pair_id", "result_endpoint_signature_id"]
    )["signature_identity_status"].to_dict()
    known_elsewhere_by_signature = signature_rows.set_index(
        ["local_pair_id", "result_endpoint_signature_id"]
    )["signature_known_elsewhere"].map(_as_bool).to_dict()
    signature_unresolved_by_signature = signature_rows.set_index(
        ["local_pair_id", "result_endpoint_signature_id"]
    )["signature_level_unresolved"].map(_as_bool).to_dict()

    positive = trace_rows[trace_rows["local_pair_id"].astype(str).eq(POSITIVE_PAIR_ID)].copy()
    transition_lookup = transition_seed_rows[
        transition_seed_rows["local_pair_id"].astype(str).eq(POSITIVE_PAIR_ID)
    ][
        [
            "local_pair_id",
            "branch",
            "start_condition",
            "seed",
            "seed_band_status",
            "descent_route_category_code",
            "ascent_route_category_code",
        ]
    ]
    rows: list[dict[str, Any]] = []
    key_cols = ["local_pair_id", "branch", "start_condition", "seed"]
    for key, group in positive.groupby(key_cols, sort=True):
        data = dict(zip(key_cols, key, strict=True))
        unresolved_rows = group[
            group["endpoint_object_assignment_by_step"].map(_is_row_local_unresolved)
        ]
        hidden_known = 0
        signature_unresolved = 0
        statuses: list[str] = []
        for row in unresolved_rows.itertuples(index=False):
            sig_key = (str(row.local_pair_id), str(row.result_endpoint_signature_id))
            if known_elsewhere_by_signature.get(sig_key, False):
                hidden_known += 1
            if signature_unresolved_by_signature.get(sig_key, False):
                signature_unresolved += 1
            statuses.append(str(status_by_signature.get(sig_key, "")))
        unresolved_signature_ids = sorted(
            set(unresolved_rows["result_endpoint_signature_id"].astype(str))
        )
        unresolved_status = (
            ";".join(sorted(set(statuses)))
            if statuses
            else "no_row_local_unresolved_steps"
        )
        rows.append(
            {
                **data,
                "row_local_unresolved_step_count": int(len(unresolved_rows)),
                "row_local_unresolved_signature_count": int(len(unresolved_signature_ids)),
                "signature_known_elsewhere_unresolved_step_count": int(hidden_known),
                "signature_level_unresolved_step_count": int(signature_unresolved),
                "signature_level_unresolved_signature_ids": ";".join(
                    sig
                    for sig in unresolved_signature_ids
                    if signature_unresolved_by_signature.get((str(data["local_pair_id"]), sig), False)
                ),
                "unresolved_signature_identity_statuses": unresolved_status,
                "method_claim_allowed_after_audit": False,
                "quality_cost_claim_allowed_after_audit": False,
                "wall_generality_claim_allowed_after_audit": False,
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    output = pd.DataFrame(rows)
    if output.empty:
        return output
    return output.merge(
        transition_lookup,
        on=key_cols,
        how="left",
        validate="one_to_one",
    )


def _pair_signature_rows(
    trace_rows: pd.DataFrame,
    signature_rows: pd.DataFrame,
    seed_rows: pd.DataFrame,
    transition_pair_rows: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    trace = trace_rows.copy()
    trace["row_local_unresolved"] = trace["endpoint_object_assignment_by_step"].map(
        _is_row_local_unresolved
    )
    signature_index = signature_rows.set_index(["local_pair_id", "result_endpoint_signature_id"])
    known_elsewhere = signature_index["signature_known_elsewhere"].map(_as_bool).to_dict()
    signature_level_unresolved = signature_index["signature_level_unresolved"].map(
        _as_bool
    ).to_dict()

    for pair_id, group in trace.groupby("local_pair_id", sort=True):
        unresolved_rows = group[group["row_local_unresolved"]]
        hidden_known_count = 0
        signature_unresolved_count = 0
        for row in unresolved_rows.itertuples(index=False):
            sig_key = (str(row.local_pair_id), str(row.result_endpoint_signature_id))
            if known_elsewhere.get(sig_key, False):
                hidden_known_count += 1
            if signature_level_unresolved.get(sig_key, False):
                signature_unresolved_count += 1
        pair_signature_rows = signature_rows[
            signature_rows["local_pair_id"].astype(str).eq(str(pair_id))
        ]
        pair_seed_rows = seed_rows[seed_rows["local_pair_id"].astype(str).eq(str(pair_id))]
        transition_pair = transition_pair_rows[
            transition_pair_rows["local_pair_id"].astype(str).eq(str(pair_id))
        ]
        pair_transition_status = (
            str(transition_pair.iloc[0]["pair_transition_band_status"])
            if not transition_pair.empty and "pair_transition_band_status" in transition_pair
            else ""
        )
        if str(pair_id) == POSITIVE_PAIR_ID:
            if (
                pair_seed_rows["signature_level_unresolved_step_count"].sum() > 0
                and pair_seed_rows["signature_known_elsewhere_unresolved_step_count"].sum() > 0
            ):
                pair_identity_status = "positive_has_hidden_known_and_true_unresolved_intermediates"
            elif pair_seed_rows["signature_level_unresolved_step_count"].sum() > 0:
                pair_identity_status = "positive_has_true_unresolved_intermediates"
            else:
                pair_identity_status = "positive_row_local_unknowns_signature_resolved"
        else:
            if signature_unresolved_count == 0 and hidden_known_count > 0:
                pair_identity_status = "boundary_row_local_unknowns_signature_resolved"
            elif signature_unresolved_count > 0:
                pair_identity_status = "boundary_has_signature_level_unresolved_rows"
            else:
                pair_identity_status = "boundary_no_row_local_unknown_rows"
        rows.append(
            {
                "local_pair_id": str(pair_id),
                "trace_row_count": int(len(group)),
                "signature_count": int(pair_signature_rows["result_endpoint_signature_id"].nunique()),
                "row_local_unresolved_row_count": int(len(unresolved_rows)),
                "signature_known_elsewhere_unresolved_row_count": int(hidden_known_count),
                "signature_level_unresolved_row_count": int(signature_unresolved_count),
                "signature_identity_status_counts": json.dumps(
                    _count_dict(pair_signature_rows["signature_identity_status"]),
                    ensure_ascii=True,
                    sort_keys=True,
                ),
                "seed_signature_status_counts": json.dumps(
                    _count_dict(pair_seed_rows["unresolved_signature_identity_statuses"]),
                    ensure_ascii=True,
                    sort_keys=True,
                )
                if not pair_seed_rows.empty
                else "{}",
                "pair_transition_band_status": pair_transition_status,
                "pair_signature_identity_status": pair_identity_status,
                "method_claim_allowed_after_audit": False,
                "quality_cost_claim_allowed_after_audit": False,
                "wall_generality_claim_allowed_after_audit": False,
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows)


def _gate_matrix(
    *,
    signature_rows: pd.DataFrame,
    seed_rows: pd.DataFrame,
    pair_rows: pd.DataFrame,
    transition_pair_rows: pd.DataFrame,
) -> pd.DataFrame:
    positive_pair = pair_rows[pair_rows["local_pair_id"].astype(str).eq(POSITIVE_PAIR_ID)]
    boundary_pair = pair_rows[pair_rows["local_pair_id"].astype(str).eq(BOUNDARY_PAIR_ID)]
    transition_positive = transition_pair_rows[
        transition_pair_rows["local_pair_id"].astype(str).eq(POSITIVE_PAIR_ID)
    ]
    positive_signature_ids = signature_rows[
        signature_rows["local_pair_id"].astype(str).eq(POSITIVE_PAIR_ID)
    ]["result_endpoint_signature_id"].nunique()
    positive_seed_count = int(len(seed_rows))
    positive_signature_level_unresolved = (
        int(positive_pair.iloc[0]["signature_level_unresolved_row_count"])
        if not positive_pair.empty
        else 0
    )
    boundary_signature_level_unresolved = (
        int(boundary_pair.iloc[0]["signature_level_unresolved_row_count"])
        if not boundary_pair.empty
        else 0
    )
    all_claims_closed = bool(
        not pair_rows["method_claim_allowed_after_audit"].map(_as_bool).any()
        and not pair_rows["quality_cost_claim_allowed_after_audit"].map(_as_bool).any()
        and not pair_rows["wall_generality_claim_allowed_after_audit"].map(_as_bool).any()
    )
    rows = [
        {
            "gate_id": "G1_signature_rows_materialized",
            "question": "Were signature identity rows materialized for both pairs?",
            "observed": f"signature_rows={len(signature_rows)} positive_signatures={positive_signature_ids}",
            "minimum_or_rule": "both pairs have signature rows and positive has >= 2 signatures",
            "gate_status": "pass"
            if len(signature_rows) > 0 and positive_signature_ids >= 2
            else "fail",
        },
        {
            "gate_id": "G2_positive_seed_identity_rows_materialized",
            "question": "Was every positive seed-start unit audited for signature identity?",
            "observed": f"positive_seed_rows={positive_seed_count}",
            "minimum_or_rule": "32 positive seed-start rows",
            "gate_status": "pass" if positive_seed_count == 32 else "fail",
        },
        {
            "gate_id": "G3_positive_has_true_unresolved_intermediates",
            "question": "Does 014 retain signature-level unresolved intermediate rows after identity audit?",
            "observed": f"signature_level_unresolved_rows={positive_signature_level_unresolved}",
            "minimum_or_rule": "> 0 unresolved rows after removing signature-known row-local unknowns",
            "gate_status": "pass" if positive_signature_level_unresolved > 0 else "fail",
        },
        {
            "gate_id": "G4_boundary_row_local_unknowns_resolved_by_signature",
            "question": "Are 005 row-local unknowns resolved at signature level?",
            "observed": f"boundary_signature_level_unresolved_rows={boundary_signature_level_unresolved}",
            "minimum_or_rule": "0 boundary signature-level unresolved rows",
            "gate_status": "pass" if boundary_signature_level_unresolved == 0 else "fail",
        },
        {
            "gate_id": "G5_transition_band_status_preserved",
            "question": "Does this audit preserve the transition-band result rather than rerunning it?",
            "observed": transition_positive[
                ["local_pair_id", "pair_transition_band_status"]
            ].to_dict(orient="records"),
            "minimum_or_rule": "transition-band pair status is present",
            "gate_status": "pass" if not transition_positive.empty else "fail",
        },
        {
            "gate_id": "G6_claims_closed",
            "question": "Are method, quality/cost, and wall-generality claims closed?",
            "observed": CLAIM_BOUNDARY,
            "minimum_or_rule": "all claim flags false",
            "gate_status": "pass" if all_claims_closed else "fail",
        },
        {
            "gate_id": "G7_read_only_audit",
            "question": "Was this a read-only audit over the executed trace?",
            "observed": RUN_STATUS,
            "minimum_or_rule": "audit status only; no Leiden rerun",
            "gate_status": "pass",
        },
    ]
    return pd.DataFrame(rows)


def _write_report(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    signature_rows: pd.DataFrame,
    seed_rows: pd.DataFrame,
    pair_rows: pd.DataFrame,
    gates: pd.DataFrame,
) -> None:
    report = [
        "# NanoClustering G4.8 First-Pass 014 Signature-Identity Audit",
        "",
        f"- status: `{RUN_STATUS}`",
        f"- signature_row_count: {summary['signature_row_count']}",
        f"- positive_signature_count: {summary['positive_signature_count']}",
        f"- positive_row_local_unresolved_row_count: {summary['positive_row_local_unresolved_row_count']}",
        f"- positive_signature_known_elsewhere_unresolved_row_count: {summary['positive_signature_known_elsewhere_unresolved_row_count']}",
        f"- positive_signature_level_unresolved_row_count: {summary['positive_signature_level_unresolved_row_count']}",
        f"- boundary_signature_level_unresolved_row_count: {summary['boundary_signature_level_unresolved_row_count']}",
        f"- gate_status_counts: {summary['gate_status_counts']}",
        f"- failed_gates: {summary['failed_gates']}",
        "- interpretation: Row-local endpoint assignments are not sufficient for "
        "object identity. The 014 trace has both hidden-known row-local unknowns "
        "and true signature-level unresolved intermediate objects; the 005 "
        "boundary row-local unknowns resolve at signature level.",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Pair Signature Identity",
        "",
        _markdown_table(pair_rows),
        "",
        "## Positive Seed Identity",
        "",
        _markdown_table(
            seed_rows[
                [
                    "start_condition",
                    "seed",
                    "seed_band_status",
                    "row_local_unresolved_step_count",
                    "signature_known_elsewhere_unresolved_step_count",
                    "signature_level_unresolved_step_count",
                    "signature_level_unresolved_signature_ids",
                    "descent_route_category_code",
                    "ascent_route_category_code",
                ]
            ]
        ),
        "",
        "## Signature Rows",
        "",
        _markdown_table(
            signature_rows[
                [
                    "local_pair_id",
                    "result_endpoint_signature_id",
                    "signature_row_count",
                    "observed_endpoint_object_assignments",
                    "signature_identity_status",
                    "bridge_fractions",
                    "objective_value_mean",
                    "pair_coassigned_rate",
                ]
            ]
        ),
        "",
        "## Gate Matrix",
        "",
        _markdown_table(gates),
        "",
        "## Boundary",
        "",
        "This audit does not reinterpret the trace as a method result. It exposes "
        "an object-identity blind spot that must be resolved before stronger "
        "wall-localization language.",
        "",
    ]
    (output_dir / REPORT_MD).write_text("\n".join(report), encoding="utf-8")


def run_audit(
    trace_dir: Path,
    transition_band_audit_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    trace_rows = _read_csv(trace_dir / TRACE_ROWS_CSV)
    transition_seed_rows = _read_csv(
        transition_band_audit_dir / TRANSITION_SEED_BAND_ROWS_CSV
    )
    transition_pair_rows = _read_csv(
        transition_band_audit_dir / TRANSITION_PAIR_BAND_ROWS_CSV
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    signature_rows = _signature_identity_rows(trace_rows)
    seed_rows = _seed_signature_rows(
        trace_rows=trace_rows,
        signature_rows=signature_rows,
        transition_seed_rows=transition_seed_rows,
    )
    pair_rows = _pair_signature_rows(
        trace_rows=trace_rows,
        signature_rows=signature_rows,
        seed_rows=seed_rows,
        transition_pair_rows=transition_pair_rows,
    )
    gates = _gate_matrix(
        signature_rows=signature_rows,
        seed_rows=seed_rows,
        pair_rows=pair_rows,
        transition_pair_rows=transition_pair_rows,
    )

    _write_csv(signature_rows, output_dir / SIGNATURE_IDENTITY_ROWS_CSV)
    _write_csv(seed_rows, output_dir / SEED_SIGNATURE_AUDIT_ROWS_CSV)
    _write_csv(pair_rows, output_dir / PAIR_SIGNATURE_AUDIT_ROWS_CSV)
    _write_csv(gates, output_dir / GATE_MATRIX_CSV)

    positive_pair = pair_rows[pair_rows["local_pair_id"].astype(str).eq(POSITIVE_PAIR_ID)].iloc[0]
    boundary_pair = pair_rows[pair_rows["local_pair_id"].astype(str).eq(BOUNDARY_PAIR_ID)].iloc[0]
    summary = {
        "run_status": RUN_STATUS,
        "trace_dir": str(trace_dir),
        "transition_band_audit_dir": str(transition_band_audit_dir),
        "output_dir": str(output_dir),
        "signature_row_count": int(len(signature_rows)),
        "positive_signature_count": int(positive_pair["signature_count"]),
        "positive_row_local_unresolved_row_count": int(
            positive_pair["row_local_unresolved_row_count"]
        ),
        "positive_signature_known_elsewhere_unresolved_row_count": int(
            positive_pair["signature_known_elsewhere_unresolved_row_count"]
        ),
        "positive_signature_level_unresolved_row_count": int(
            positive_pair["signature_level_unresolved_row_count"]
        ),
        "boundary_row_local_unresolved_row_count": int(
            boundary_pair["row_local_unresolved_row_count"]
        ),
        "boundary_signature_known_elsewhere_unresolved_row_count": int(
            boundary_pair["signature_known_elsewhere_unresolved_row_count"]
        ),
        "boundary_signature_level_unresolved_row_count": int(
            boundary_pair["signature_level_unresolved_row_count"]
        ),
        "positive_signature_identity_status_counts": _count_dict(
            signature_rows[
                signature_rows["local_pair_id"].astype(str).eq(POSITIVE_PAIR_ID)
            ]["signature_identity_status"]
        ),
        "boundary_signature_identity_status_counts": _count_dict(
            signature_rows[
                signature_rows["local_pair_id"].astype(str).eq(BOUNDARY_PAIR_ID)
            ]["signature_identity_status"]
        ),
        "positive_seed_signature_level_unresolved_step_count_distribution": _count_dict(
            seed_rows["signature_level_unresolved_step_count"]
        ),
        "gate_status_counts": _count_dict(gates["gate_status"]),
        "failed_gates": gates[gates["gate_status"].astype(str).ne("pass")][
            "gate_id"
        ].astype(str).tolist(),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    config = {
        "trace_dir": str(trace_dir),
        "transition_band_audit_dir": str(transition_band_audit_dir),
        "output_dir": str(output_dir),
        "read_only_trace_audit": True,
        "positive_pair_id": POSITIVE_PAIR_ID,
        "boundary_pair_id": BOUNDARY_PAIR_ID,
        "run_status": RUN_STATUS,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    _write_report(
        output_dir=output_dir,
        summary=summary,
        signature_rows=signature_rows,
        seed_rows=seed_rows,
        pair_rows=pair_rows,
        gates=gates,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE_DIR)
    parser.add_argument(
        "--transition-band-audit-dir",
        type=Path,
        default=DEFAULT_TRANSITION_BAND_AUDIT_DIR,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_audit(
        trace_dir=args.trace_dir,
        transition_band_audit_dir=args.transition_band_audit_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(_json_safe(summary), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
