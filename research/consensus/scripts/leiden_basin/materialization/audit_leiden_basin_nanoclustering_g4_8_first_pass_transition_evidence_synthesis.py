#!/usr/bin/env python3
"""Synthesize first-pass transition evidence for the basin/pathway definition.

This read-only synthesis gathers the current G4.8/G4.9A evidence chain around
``local_pair_014`` and the strict-ready ``local_pair_016`` diagnostic. It
separates pair-level facts, claim-level evidence, and definition-decision
tensions. It does not rerun Leiden, perform a fraction sweep, promote walls,
evaluate quality/cost value, or claim method success.
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


DEFAULT_FIRST_PASS_DIR = (
    BASE_RESULT_DIR / "leiden_basin_nanoclustering_g4_8_fresh_axis_b_first_pass_trace_gamma1e5_20260604"
)
DEFAULT_EXCLUSIVE_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_exclusive_target_contrast_audit_gamma1e5_20260604"
)
DEFAULT_WALL_TRACE_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_014_wall_localization_trace_gamma1e5_20260605"
)
DEFAULT_TRANSITION_BAND_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_014_wall_localization_transition_band_audit_gamma1e5_20260605"
)
DEFAULT_SIGNATURE_IDENTITY_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_014_wall_localization_signature_identity_audit_gamma1e5_20260605"
)
DEFAULT_ROLE_STABILITY_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_014_intermediate_role_stability_audit_gamma1e5_20260605"
)
DEFAULT_TRANSFER_SCREEN_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_014_role_pattern_transfer_screen_gamma1e5_20260605"
)
DEFAULT_CONTINUITY_016_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_016_continuity_block_audit_gamma1e5_20260605"
)
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_transition_evidence_synthesis_gamma1e5_20260605"
)

PAIR_EVIDENCE_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_transition_evidence_synthesis_pair_rows.csv"
)
CLAIM_EVIDENCE_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_transition_evidence_synthesis_claim_rows.csv"
)
DEFINITION_DECISION_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_transition_evidence_synthesis_definition_rows.csv"
)
GATE_MATRIX_CSV = (
    "nanoclustering_g4_8_first_pass_transition_evidence_synthesis_gate_matrix.csv"
)
SUMMARY_JSON = "nanoclustering_g4_8_first_pass_transition_evidence_synthesis_summary.json"
CONFIG_JSON = "nanoclustering_g4_8_first_pass_transition_evidence_synthesis_config.json"
REPORT_MD = "nanoclustering_g4_8_first_pass_transition_evidence_synthesis_report.md"

RUN_STATUS = "audited_nanoclustering_g4_8_first_pass_transition_evidence_synthesis"
ROUTE_EXECUTION_STATUS = "not_executed_read_only_transition_evidence_synthesis"
WALL_PROMOTION_STATUS = "not_promoted_transition_evidence_synthesis_only"
METHOD_STATUS = "definition_evidence_synthesis_not_method"
CLAIM_BOUNDARY = (
    "NanoClustering G4.8/G4.9A first-pass transition evidence synthesis only; "
    "reads existing first-pass, localization, transition-band, signature, "
    "role-stability, transfer-screen, and 016 continuity-block outputs to "
    "organize evidence for the basin/pathway definition. It does not rerun "
    "Leiden, perform a fraction sweep, promote basin walls, replay full "
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


def _count_dict(series: pd.Series) -> dict[str, int]:
    if series.empty:
        return {}
    return {str(key): int(value) for key, value in series.value_counts(dropna=False).items()}


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


def _load_context(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "first_pass_dir": Path(args.first_pass_dir),
        "exclusive_dir": Path(args.exclusive_dir),
        "wall_trace_dir": Path(args.wall_trace_dir),
        "transition_band_dir": Path(args.transition_band_dir),
        "signature_identity_dir": Path(args.signature_identity_dir),
        "role_stability_dir": Path(args.role_stability_dir),
        "transfer_screen_dir": Path(args.transfer_screen_dir),
        "continuity_016_dir": Path(args.continuity_016_dir),
    }
    summaries = {
        "first_pass": _read_json(
            paths["first_pass_dir"] / "nanoclustering_g4_8_fresh_axis_b_first_pass_trace_summary.json"
        ),
        "exclusive": _read_json(
            paths["exclusive_dir"]
            / "nanoclustering_g4_8_first_pass_exclusive_target_contrast_summary.json"
        ),
        "wall_trace": _read_json(
            paths["wall_trace_dir"]
            / "nanoclustering_g4_8_first_pass_014_wall_localization_trace_summary.json"
        ),
        "transition_band": _read_json(
            paths["transition_band_dir"]
            / "nanoclustering_g4_8_first_pass_014_wall_localization_transition_band_summary.json"
        ),
        "signature_identity": _read_json(
            paths["signature_identity_dir"]
            / "nanoclustering_g4_8_first_pass_014_wall_localization_signature_identity_summary.json"
        ),
        "role_stability": _read_json(
            paths["role_stability_dir"]
            / "nanoclustering_g4_8_first_pass_014_intermediate_role_stability_summary.json"
        ),
        "transfer_screen": _read_json(
            paths["transfer_screen_dir"]
            / "nanoclustering_g4_8_first_pass_014_role_pattern_transfer_summary.json"
        ),
        "continuity_016": _read_json(
            paths["continuity_016_dir"]
            / "nanoclustering_g4_8_first_pass_016_continuity_block_summary.json"
        ),
    }
    tables = {
        "first_pass_pair": _read_csv(
            paths["first_pass_dir"]
            / "nanoclustering_g4_8_fresh_axis_b_first_pass_pair_readout_result_rows.csv"
        ),
        "exclusive_pair": _read_csv(
            paths["exclusive_dir"]
            / "nanoclustering_g4_8_first_pass_exclusive_target_pair_contrast_rows.csv"
        ),
        "transition_pair": _read_csv(
            paths["transition_band_dir"]
            / "nanoclustering_g4_8_first_pass_014_wall_localization_transition_band_pair_rows.csv"
        ),
        "signature_identity_pair": _read_csv(
            paths["signature_identity_dir"]
            / "nanoclustering_g4_8_first_pass_014_wall_localization_signature_identity_pair_rows.csv"
        ),
        "role_pair": _read_csv(
            paths["role_stability_dir"]
            / "nanoclustering_g4_8_first_pass_014_intermediate_role_stability_pair_rows.csv"
        ),
        "transfer_pair": _read_csv(
            paths["transfer_screen_dir"]
            / "nanoclustering_g4_8_first_pass_014_role_pattern_transfer_pair_rows.csv"
        ),
        "continuity_pair": _read_csv(
            paths["continuity_016_dir"]
            / "nanoclustering_g4_8_first_pass_016_continuity_block_pair_comparison_rows.csv"
        ),
        "continuity_step": _read_csv(
            paths["continuity_016_dir"]
            / "nanoclustering_g4_8_first_pass_016_continuity_block_step_signature_rows.csv"
        ),
        "continuity_route": _read_csv(
            paths["continuity_016_dir"]
            / "nanoclustering_g4_8_first_pass_016_continuity_block_route_rows.csv"
        ),
    }
    gate_paths = {
        "first_pass": paths["first_pass_dir"]
        / "nanoclustering_g4_8_fresh_axis_b_first_pass_trace_gate_matrix.csv",
        "exclusive": paths["exclusive_dir"]
        / "nanoclustering_g4_8_first_pass_exclusive_target_contrast_gate_matrix.csv",
        "wall_trace": paths["wall_trace_dir"]
        / "nanoclustering_g4_8_first_pass_014_wall_localization_trace_gate_matrix.csv",
        "transition_band": paths["transition_band_dir"]
        / "nanoclustering_g4_8_first_pass_014_wall_localization_transition_band_gate_matrix.csv",
        "signature_identity": paths["signature_identity_dir"]
        / "nanoclustering_g4_8_first_pass_014_wall_localization_signature_identity_gate_matrix.csv",
        "role_stability": paths["role_stability_dir"]
        / "nanoclustering_g4_8_first_pass_014_intermediate_role_stability_gate_matrix.csv",
        "transfer_screen": paths["transfer_screen_dir"]
        / "nanoclustering_g4_8_first_pass_014_role_pattern_transfer_gate_matrix.csv",
        "continuity_016": paths["continuity_016_dir"]
        / "nanoclustering_g4_8_first_pass_016_continuity_block_gate_matrix.csv",
    }
    gates = {name: _read_csv(path) for name, path in gate_paths.items()}
    return {"paths": paths, "summaries": summaries, "tables": tables, "gates": gates}


def _transfer_row(tables: dict[str, pd.DataFrame], local_pair_id: str) -> dict[str, Any]:
    rows = tables["transfer_pair"][
        tables["transfer_pair"]["local_pair_id"].astype(str).eq(local_pair_id)
    ]
    return rows.iloc[0].to_dict() if not rows.empty else {}


def _first_pass_row(tables: dict[str, pd.DataFrame], local_pair_id: str) -> dict[str, Any]:
    rows = tables["first_pass_pair"][
        tables["first_pass_pair"]["local_pair_id"].astype(str).eq(local_pair_id)
    ]
    return rows.iloc[0].to_dict() if not rows.empty else {}


def _continuity_row(tables: dict[str, pd.DataFrame], local_pair_id: str) -> dict[str, Any]:
    rows = tables["continuity_pair"][
        tables["continuity_pair"]["local_pair_id"].astype(str).eq(local_pair_id)
    ]
    return rows.iloc[0].to_dict() if not rows.empty else {}


def _signature_identity_row(tables: dict[str, pd.DataFrame], local_pair_id: str) -> dict[str, Any]:
    rows = tables["signature_identity_pair"][
        tables["signature_identity_pair"]["local_pair_id"].astype(str).eq(local_pair_id)
    ]
    return rows.iloc[0].to_dict() if not rows.empty else {}


def _pair_evidence_rows(context: dict[str, Any]) -> pd.DataFrame:
    tables = context["tables"]
    pair_ids = [
        "local_pair_014",
        "local_pair_016",
        "local_pair_005",
        "local_pair_007",
        "local_pair_008",
        "local_pair_002",
        "local_pair_022",
        "local_pair_003",
        "local_pair_013",
    ]
    rows: list[dict[str, Any]] = []
    for local_pair_id in pair_ids:
        transfer = _transfer_row(tables, local_pair_id)
        first = _first_pass_row(tables, local_pair_id)
        continuity = _continuity_row(tables, local_pair_id)
        identity = _signature_identity_row(tables, local_pair_id)
        row = {
            "local_pair_id": local_pair_id,
            "evidence_role": str(first.get("evidence_role", transfer.get("evidence_role", ""))),
            "validation_stratum": str(transfer.get("validation_stratum", first.get("validation_stratum", ""))),
            "pair_first_pass_result": str(
                first.get("pair_first_pass_result", transfer.get("pair_first_pass_result", ""))
            ),
            "ready_like_seed_route_pass_count": int(
                first.get("ready_like_seed_route_pass_count", transfer.get("ready_like_seed_route_pass_count", 0))
            ),
            "route_readout_row_count": int(first.get("route_readout_row_count", 0)),
            "transfer_screen_status": str(transfer.get("transfer_screen_status", "")),
            "role_analog_feature_count": int(transfer.get("role_analog_feature_count", 0))
            if transfer
            else 0,
            "has_source_like_signature": _as_bool(transfer.get("has_source_like_signature", False)),
            "has_target_anchor_signature": _as_bool(transfer.get("has_target_anchor_signature", False)),
            "has_hidden_known_intermediate_signature": _as_bool(
                transfer.get("has_hidden_known_intermediate_signature", False)
            ),
            "has_unresolved_pair_coassigned_signature": _as_bool(
                transfer.get("has_unresolved_pair_coassigned_signature", False)
            ),
            "has_unresolved_pair_separated_signature": _as_bool(
                transfer.get("has_unresolved_pair_separated_signature", False)
            ),
            "new_positive_transfer_candidate": _as_bool(
                transfer.get("new_positive_transfer_candidate", False)
            ),
            "signature_level_unresolved_row_count": int(
                identity.get("signature_level_unresolved_row_count", 0)
            )
            if identity
            else 0,
            "signature_known_elsewhere_unresolved_row_count": int(
                identity.get("signature_known_elsewhere_unresolved_row_count", 0)
            )
            if identity
            else 0,
            "comparison_role": str(continuity.get("comparison_role", "")),
            "bridge_release_lift_proxy": float(continuity.get("bridge_release_lift_proxy", 0.0))
            if continuity
            else 0.0,
            "direct_dependency_proxy": float(continuity.get("direct_dependency_proxy", 0.0))
            if continuity
            else 0.0,
            "original_pair_coassigned_share": float(
                continuity.get("original_pair_coassigned_share", 0.0)
            )
            if continuity
            else 0.0,
            "drop_bridge_pair_coassigned_share": float(
                continuity.get("drop_bridge_pair_coassigned_share", 0.0)
            )
            if continuity
            else 0.0,
            "evidence_readout": "",
            "definition_relevance": "",
            "method_claim_allowed_after_synthesis": False,
            "quality_cost_claim_allowed_after_synthesis": False,
            "wall_generality_claim_allowed_after_synthesis": False,
            "route_execution_status": ROUTE_EXECUTION_STATUS,
            "wall_promotion_status": WALL_PROMOTION_STATUS,
            "method_status": METHOD_STATUS,
            "run_status": RUN_STATUS,
            "claim_boundary": CLAIM_BOUNDARY,
        }
        if local_pair_id == "local_pair_014":
            row["evidence_readout"] = (
                "reference bounded transition band: clean first-pass scaffold, "
                "32/32 bounded transition seeds, typed recurrent intermediate signatures"
            )
            row["definition_relevance"] = "supports typed pathway/basin-band definition, not point-wall definition"
        elif local_pair_id == "local_pair_016":
            row["evidence_readout"] = (
                "strict-ready role analog with source and final target support, "
                "but one recurrent step-2 bridge-reassignment transient on 24/24 routes"
            )
            row["definition_relevance"] = "central decision case for transient-as-pathway versus transient-as-blocker"
        elif local_pair_id == "local_pair_005":
            row["evidence_readout"] = (
                "partial boundary guard: 24/32 ready-like routes but source/target "
                "collapse and zero signature-level unresolved boundary rows"
            )
            row["definition_relevance"] = "guards against treating all target openings as pathway evidence"
        elif local_pair_id in {"local_pair_007", "local_pair_003"}:
            row["evidence_readout"] = "rare-ready continuity-blocked analog, secondary after 016"
            row["definition_relevance"] = "tests whether blocked analogs recur outside strict-ready context"
        elif local_pair_id in {"local_pair_008", "local_pair_002", "local_pair_022"}:
            row["evidence_readout"] = "closed control analog with role-like signatures but no positive transfer"
            row["definition_relevance"] = "negative-control guard for role-pattern overinterpretation"
        else:
            row["evidence_readout"] = "closed or low-signal control"
            row["definition_relevance"] = "background guard"
        rows.append(row)
    return pd.DataFrame(rows)


def _claim_rows(context: dict[str, Any]) -> pd.DataFrame:
    s = context["summaries"]
    rows = [
        {
            "claim_id": "C01_first_pass_scope",
            "claim_type": "scope_boundary",
            "claim": "The executed first-pass screen is route-local evidence, not wall or method evidence.",
            "evidence": (
                f"9 screened pairs; 288 route readouts; gates={s['first_pass']['gate_status_counts']}; "
                f"result_counts={s['first_pass']['pair_first_pass_result_counts']}"
            ),
            "supports": "bounded evidence accounting",
            "cautions": "does not justify wall generality or method value",
            "primary_artifact": str(context["paths"]["first_pass_dir"]),
            "definition_relevance": "sets the evidentiary level of all downstream claims",
            "claim_allowed": True,
            "overclaim_guard": CLAIM_BOUNDARY,
        },
        {
            "claim_id": "C02_only_014_clean_first_pass",
            "claim_type": "positive_reference",
            "claim": "local_pair_014 is the only clean first-pass scaffold in the current screen.",
            "evidence": (
                f"clean_candidates={s['exclusive']['clean_candidates']}; "
                f"partial_candidates={s['exclusive']['partial_candidates']}; "
                f"non014_new_positive_transfer_candidates={s['transfer_screen']['non014_new_positive_transfer_candidates']}"
            ),
            "supports": "014 as reference case",
            "cautions": "does not establish generality",
            "primary_artifact": str(context["paths"]["transfer_screen_dir"]),
            "definition_relevance": "reference case for basin/pathway definition",
            "claim_allowed": True,
            "overclaim_guard": "no non-014 positive promotion",
        },
        {
            "claim_id": "C03_014_is_transition_band_not_point_wall",
            "claim_type": "mechanism_readout",
            "claim": "014 evidence is a bounded transition band, not a single clean point wall.",
            "evidence": (
                "positive_seed_start_count=32; "
                f"strict={s['transition_band']['strict_interpretable_seed_count']}; "
                f"monotone_intermediate={s['transition_band']['monotone_intermediate_seed_count']}; "
                f"bounded_nonmonotone={s['transition_band']['nonmonotone_bounded_transition_seed_count']}; "
                f"boundary_positive_leak_closed={s['transition_band']['boundary_positive_leak_closed']}"
            ),
            "supports": "pathway/band vocabulary",
            "cautions": "weakens point-wall wording",
            "primary_artifact": str(context["paths"]["transition_band_dir"]),
            "definition_relevance": "basin wall should allow bounded transition intervals",
            "claim_allowed": True,
            "overclaim_guard": "not wall-location generality",
        },
        {
            "claim_id": "C04_row_local_unknown_not_identity",
            "claim_type": "identity_guard",
            "claim": "Row-local unknown endpoint assignments are not stable object identities.",
            "evidence": (
                f"014 row-local unresolved={s['signature_identity']['positive_row_local_unresolved_row_count']}; "
                f"known_elsewhere={s['signature_identity']['positive_signature_known_elsewhere_unresolved_row_count']}; "
                f"true_signature_level={s['signature_identity']['positive_signature_level_unresolved_row_count']}; "
                f"005 boundary true_signature_level={s['signature_identity']['boundary_signature_level_unresolved_row_count']}"
            ),
            "supports": "signature-level endpoint identity",
            "cautions": "do not use row-local unknown counts as basin evidence",
            "primary_artifact": str(context["paths"]["signature_identity_dir"]),
            "definition_relevance": "basin/pathway definitions need signature identity, not row labels alone",
            "claim_allowed": True,
            "overclaim_guard": "unknown is not automatically new basin",
        },
        {
            "claim_id": "C05_014_typed_intermediates_recur",
            "claim_type": "mechanism_readout",
            "claim": "014 intermediate signatures are typed and recurrent across seed routes.",
            "evidence": (
                f"typed_signature_count={s['role_stability']['typed_signature_count']}; "
                f"unresolved_signature_ids={s['role_stability']['signature_level_unresolved_signature_ids']}; "
                f"unresolved_seed_routes={s['role_stability']['seed_route_with_unresolved_intermediate_count']}/64; "
                f"hidden_known_seed_routes={s['role_stability']['seed_route_with_hidden_known_source_guard_count']}/64"
            ),
            "supports": "typed transition-band interpretation",
            "cautions": "still local to 014",
            "primary_artifact": str(context["paths"]["role_stability_dir"]),
            "definition_relevance": "intermediates should be role-typed before deciding pathway status",
            "claim_allowed": True,
            "overclaim_guard": "no method/generalization claim",
        },
        {
            "claim_id": "C06_transfer_screen_finds_no_second_positive",
            "claim_type": "generality_guard",
            "claim": "The 014 role pattern has analogs, but no non-014 positive transfer candidate.",
            "evidence": (
                f"screened_pair_count={s['transfer_screen']['screened_pair_count']}; "
                f"candidate_pairs={s['transfer_screen']['candidate_pairs']}; "
                f"non014_new_positive_transfer_candidates={s['transfer_screen']['non014_new_positive_transfer_candidates']}; "
                f"primary_diagnostic_pairs={s['transfer_screen']['primary_diagnostic_pairs']}"
            ),
            "supports": "016 as diagnostic, not positive",
            "cautions": "role similarity is insufficient for pathway/wall claim",
            "primary_artifact": str(context["paths"]["transfer_screen_dir"]),
            "definition_relevance": "separates role analog from accepted pathway evidence",
            "claim_allowed": True,
            "overclaim_guard": "do not localize every analog",
        },
        {
            "claim_id": "C07_016_block_is_single_typed_transient",
            "claim_type": "definition_tension",
            "claim": "016 fails current continuity because every route passes through one typed transient intermediate.",
            "evidence": (
                "24/24 routes source-start pass and target-exclusive final pass; "
                f"single_step_bridge_reassignment_block_count={s['continuity_016']['primary_single_step_bridge_reassignment_block_count']}; "
                f"signature_ids={s['continuity_016']['primary_unknown_signature_ids']}; "
                f"unknown_step_rows={s['continuity_016']['primary_unknown_step_rows']}"
            ),
            "supports": "definition decision is needed before more execution",
            "cautions": "does not promote 016 to positive wall evidence",
            "primary_artifact": str(context["paths"]["continuity_016_dir"]),
            "definition_relevance": "central transient-as-pathway versus transient-as-blocker case",
            "claim_allowed": True,
            "overclaim_guard": "016 remains diagnostic-only",
        },
        {
            "claim_id": "C08_005_boundary_blocks_overinclusive_definition",
            "claim_type": "negative_control",
            "claim": "005 prevents an overinclusive definition where any target opening counts as pathway evidence.",
            "evidence": (
                "partial first-pass ready-like 24/32; source/target collapse present; "
                f"boundary_signature_level_unresolved={s['signature_identity']['boundary_signature_level_unresolved_row_count']}; "
                f"boundary_positive_target_route_count={s['transition_band']['boundary_positive_target_route_count']}"
            ),
            "supports": "boundary guard for definition",
            "cautions": "target opening alone is not enough",
            "primary_artifact": str(context["paths"]["signature_identity_dir"]),
            "definition_relevance": "requires exclusivity plus typed identity, not target label alone",
            "claim_allowed": True,
            "overclaim_guard": "partial boundary not positive",
        },
    ]
    for row in rows:
        row.update(
            {
                "method_claim_allowed_after_synthesis": False,
                "quality_cost_claim_allowed_after_synthesis": False,
                "wall_generality_claim_allowed_after_synthesis": False,
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "method_status": METHOD_STATUS,
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows)


def _definition_rows(context: dict[str, Any]) -> pd.DataFrame:
    s = context["summaries"]
    rows = [
        {
            "definition_issue_id": "D01_endpoint_identity_level",
            "definition_issue": "What is the primitive endpoint object: row label, endpoint signature, or role-typed signature?",
            "evidence_for_inclusion": (
                "014 row-local unknowns split into known-elsewhere and true signature-level intermediates; "
                "014 role-stability types six signatures"
            ),
            "evidence_against_naive_rule": (
                "005 has 204 row-local unknown rows but 0 signature-level unresolved boundary rows"
            ),
            "provisional_rule": (
                "Use endpoint signature identity first, then role typing; row-local unknown is a diagnostic flag only."
            ),
            "remaining_risk": "signature identity is still local to these induced graphs",
            "next_gate": "keep signature-level rows in every future pathway audit",
        },
        {
            "definition_issue_id": "D02_wall_shape",
            "definition_issue": "Is a basin boundary a point wall or a transition band?",
            "evidence_for_inclusion": (
                "014 has 32/32 bounded transition-band seed starts, with only 1 strict wall interval"
            ),
            "evidence_against_naive_rule": "strict W-only rule would discard 31/32 positive bounded transitions",
            "provisional_rule": "Treat real-data wall evidence as bounded transition band unless a strict point wall is repeatedly observed.",
            "remaining_risk": "band width and localization still need a controlled definition",
            "next_gate": "define band acceptability by boundedness, endpoint exclusivity, and typed intermediate recurrence",
        },
        {
            "definition_issue_id": "D03_transient_intermediate_semantics",
            "definition_issue": "Should a typed transient intermediate count as pathway evidence or a continuity blocker?",
            "evidence_for_inclusion": (
                "016 has 24/24 source-start and target-final routes, with exactly one typed bridge-reassignment transient"
            ),
            "evidence_against_naive_rule": (
                "current continuity rule assigns 0 ready-like routes to 016 despite final target success"
            ),
            "provisional_rule": (
                "Do not promote 016 yet; define a separate typed-transient pathway class and test it against 005/008 guards."
            ),
            "remaining_risk": "including transients may admit control analogs unless guard tests are explicit",
            "next_gate": "test typed transient acceptance against 005 boundary and 008/002/022 closed controls",
        },
        {
            "definition_issue_id": "D04_generality_standard",
            "definition_issue": "What evidence is sufficient to claim multiple meaningful basins beyond 014?",
            "evidence_for_inclusion": (
                f"transfer screen finds role analogs in {s['transfer_screen']['candidate_pairs']}"
            ),
            "evidence_against_naive_rule": (
                f"non014_new_positive_transfer_candidates={s['transfer_screen']['non014_new_positive_transfer_candidates']}"
            ),
            "provisional_rule": (
                "Role analogs are diagnostic only. A second positive needs endpoint exclusivity, typed pathway semantics, and guard closure."
            ),
            "remaining_risk": "current sample has one clean reference and several analogs, not generality",
            "next_gate": "resolve D03 before searching for another positive",
        },
        {
            "definition_issue_id": "D05_boundary_guard_standard",
            "definition_issue": "Which negative evidence prevents overcalling pathways?",
            "evidence_for_inclusion": (
                "005 boundary, 008/002/022 closed controls, and rare-ready blocked analogs are all materialized"
            ),
            "evidence_against_naive_rule": (
                "role-like signatures appear in closed controls; target opening appears in 005 partial boundary"
            ),
            "provisional_rule": (
                "Every broadened pathway rule must preserve 005 as boundary guard and controls as non-positive."
            ),
            "remaining_risk": "guard criteria may be too strict if pathway semantics are broadened without re-auditing",
            "next_gate": "write acceptance/rejection predicates before executing additional traces",
        },
    ]
    for row in rows:
        row.update(
            {
                "method_claim_allowed_after_synthesis": False,
                "quality_cost_claim_allowed_after_synthesis": False,
                "wall_generality_claim_allowed_after_synthesis": False,
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "method_status": METHOD_STATUS,
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows)


def _gate_matrix(
    context: dict[str, Any],
    pair_rows: pd.DataFrame,
    claim_rows: pd.DataFrame,
    definition_rows: pd.DataFrame,
) -> pd.DataFrame:
    upstream_gate_counts: dict[str, int] = {}
    for gate_matrix in context["gates"].values():
        for status, count in _count_dict(gate_matrix["gate_status"]).items():
            upstream_gate_counts[status] = upstream_gate_counts.get(status, 0) + int(count)
    pair_lookup = pair_rows.set_index("local_pair_id").to_dict("index")
    all_claims_closed = bool(
        not pair_rows["method_claim_allowed_after_synthesis"].map(_as_bool).any()
        and not claim_rows["method_claim_allowed_after_synthesis"].map(_as_bool).any()
        and not definition_rows["method_claim_allowed_after_synthesis"].map(_as_bool).any()
        and not pair_rows["quality_cost_claim_allowed_after_synthesis"].map(_as_bool).any()
        and not claim_rows["quality_cost_claim_allowed_after_synthesis"].map(_as_bool).any()
        and not definition_rows["quality_cost_claim_allowed_after_synthesis"].map(_as_bool).any()
        and not pair_rows["wall_generality_claim_allowed_after_synthesis"].map(_as_bool).any()
        and not claim_rows["wall_generality_claim_allowed_after_synthesis"].map(_as_bool).any()
        and not definition_rows["wall_generality_claim_allowed_after_synthesis"].map(_as_bool).any()
    )
    rows = [
        _gate_row(
            "G1_upstream_gates_pass",
            "Did every upstream audit used by the synthesis pass?",
            json.dumps(upstream_gate_counts, ensure_ascii=True, sort_keys=True),
            "no upstream failed gates",
            upstream_gate_counts.get("fail", 0) == 0 and upstream_gate_counts.get("pass", 0) > 0,
        ),
        _gate_row(
            "G2_pair_evidence_covers_reference_diagnostic_and_guards",
            "Does pair evidence cover 014, 016, 005, analogs, and controls?",
            pair_rows[["local_pair_id", "evidence_readout"]].to_dict("records"),
            "at least 9 pair rows including 014/016/005",
            len(pair_rows) >= 9
            and {"local_pair_014", "local_pair_016", "local_pair_005"}.issubset(
                set(pair_rows["local_pair_id"].astype(str))
            ),
        ),
        _gate_row(
            "G3_014_reference_evidence_preserved",
            "Is 014 preserved as reference bounded transition-band evidence?",
            pair_lookup.get("local_pair_014", {}),
            "014 has 32 ready-like routes and reference readout",
            int(pair_lookup.get("local_pair_014", {}).get("ready_like_seed_route_pass_count", 0)) == 32,
        ),
        _gate_row(
            "G4_016_definition_tension_preserved",
            "Is 016 preserved as a definition tension rather than a positive promotion?",
            pair_lookup.get("local_pair_016", {}),
            "016 has 0 ready-like routes and no positive transfer candidate",
            int(pair_lookup.get("local_pair_016", {}).get("ready_like_seed_route_pass_count", 1)) == 0
            and not bool(pair_lookup.get("local_pair_016", {}).get("new_positive_transfer_candidate", True)),
        ),
        _gate_row(
            "G5_boundary_and_controls_preserved",
            "Are boundary and control guards preserved as non-positive evidence?",
            pair_rows[
                pair_rows["local_pair_id"].astype(str).isin(
                    ["local_pair_005", "local_pair_008", "local_pair_002", "local_pair_022"]
                )
            ][["local_pair_id", "evidence_readout", "new_positive_transfer_candidate"]].to_dict("records"),
            "005 and closed controls remain non-positive",
            not pair_rows[
                pair_rows["local_pair_id"].astype(str).isin(
                    ["local_pair_005", "local_pair_008", "local_pair_002", "local_pair_022"]
                )
            ]["new_positive_transfer_candidate"].map(_as_bool).any(),
        ),
        _gate_row(
            "G6_definition_decision_rows_materialized",
            "Were definition decision issues explicitly materialized?",
            definition_rows[["definition_issue_id", "definition_issue", "next_gate"]].to_dict("records"),
            "at least five definition rows",
            len(definition_rows) >= 5,
        ),
        _gate_row(
            "G7_claims_closed",
            "Are method, quality/cost, and wall-generality claims closed?",
            CLAIM_BOUNDARY,
            "all claim flags false",
            all_claims_closed,
        ),
    ]
    return pd.DataFrame(rows)


def _summary(
    *,
    output_dir: Path,
    context: dict[str, Any],
    pair_rows: pd.DataFrame,
    claim_rows: pd.DataFrame,
    definition_rows: pd.DataFrame,
    gates: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "schema": "nanoclustering_g4_8_first_pass_transition_evidence_synthesis_summary.v1",
        "status": RUN_STATUS,
        "output_dir": str(output_dir),
        "source_dirs": {name: str(path) for name, path in context["paths"].items()},
        "pair_evidence_row_count": int(len(pair_rows)),
        "claim_evidence_row_count": int(len(claim_rows)),
        "definition_decision_row_count": int(len(definition_rows)),
        "reference_pair": "local_pair_014",
        "primary_definition_tension_pair": "local_pair_016",
        "boundary_guard_pair": "local_pair_005",
        "non014_new_positive_transfer_candidates": context["summaries"]["transfer_screen"][
            "non014_new_positive_transfer_candidates"
        ],
        "core_evidence_summary": {
            "014_bounded_transition_seed_count": context["summaries"]["transition_band"][
                "bounded_transition_band_seed_count"
            ],
            "014_strict_wall_seed_count": context["summaries"]["transition_band"][
                "strict_interpretable_seed_count"
            ],
            "014_true_signature_level_unresolved_rows": context["summaries"]["signature_identity"][
                "positive_signature_level_unresolved_row_count"
            ],
            "014_unresolved_intermediate_seed_routes": context["summaries"]["role_stability"][
                "seed_route_with_unresolved_intermediate_count"
            ],
            "016_single_step_bridge_reassignment_blocks": context["summaries"]["continuity_016"][
                "primary_single_step_bridge_reassignment_block_count"
            ],
            "005_boundary_signature_level_unresolved_rows": context["summaries"]["signature_identity"][
                "boundary_signature_level_unresolved_row_count"
            ],
        },
        "definition_decision_focus": (
            "Resolve whether typed transient intermediates are pathway evidence "
            "or continuity blockers before new localization sweeps."
        ),
        "recommended_next_gate": (
            "Write explicit accept/reject predicates for typed transient "
            "intermediate pathways and test them against 014, 016, 005, and "
            "closed controls before running new traces."
        ),
        "gate_status_counts": _count_dict(gates["gate_status"]),
        "failed_gates": gates.loc[
            gates["gate_status"].astype(str).ne("pass"), "gate_id"
        ].astype(str).tolist(),
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _write_report(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    pair_rows: pd.DataFrame,
    claim_rows: pd.DataFrame,
    definition_rows: pd.DataFrame,
    gates: pd.DataFrame,
) -> None:
    lines = [
        "# NanoClustering G4.8 First-Pass Transition Evidence Synthesis",
        "",
        f"- status: `{summary['status']}`",
        f"- pair_evidence_row_count: {summary['pair_evidence_row_count']}",
        f"- claim_evidence_row_count: {summary['claim_evidence_row_count']}",
        f"- definition_decision_row_count: {summary['definition_decision_row_count']}",
        f"- reference_pair: `{summary['reference_pair']}`",
        f"- primary_definition_tension_pair: `{summary['primary_definition_tension_pair']}`",
        f"- boundary_guard_pair: `{summary['boundary_guard_pair']}`",
        f"- core_evidence_summary: {summary['core_evidence_summary']}",
        f"- definition_decision_focus: {summary['definition_decision_focus']}",
        f"- recommended_next_gate: {summary['recommended_next_gate']}",
        f"- gate_status_counts: {summary['gate_status_counts']}",
        f"- failed_gates: {summary['failed_gates']}",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Pair Evidence",
        "",
        _markdown_table(
            pair_rows,
            [
                "local_pair_id",
                "validation_stratum",
                "pair_first_pass_result",
                "ready_like_seed_route_pass_count",
                "transfer_screen_status",
                "role_analog_feature_count",
                "signature_level_unresolved_row_count",
                "bridge_release_lift_proxy",
                "direct_dependency_proxy",
                "evidence_readout",
                "definition_relevance",
            ],
            max_rows=20,
        ),
        "",
        "## Claim Evidence",
        "",
        _markdown_table(
            claim_rows,
            [
                "claim_id",
                "claim_type",
                "claim",
                "evidence",
                "supports",
                "cautions",
                "definition_relevance",
            ],
            max_rows=20,
        ),
        "",
        "## Definition Decisions",
        "",
        _markdown_table(
            definition_rows,
            [
                "definition_issue_id",
                "definition_issue",
                "evidence_for_inclusion",
                "evidence_against_naive_rule",
                "provisional_rule",
                "remaining_risk",
                "next_gate",
            ],
            max_rows=20,
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
            "This synthesis organizes evidence for a definition decision. It "
            "does not promote 016 to positive wall evidence, does not establish "
            "generality, and does not evaluate method or quality/cost value."
        ),
        "",
    ]
    (output_dir / REPORT_MD).write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    context = _load_context(args)
    pair_rows = _pair_evidence_rows(context)
    claim_rows = _claim_rows(context)
    definition_rows = _definition_rows(context)
    gates = _gate_matrix(context, pair_rows, claim_rows, definition_rows)
    summary = _summary(
        output_dir=output_dir,
        context=context,
        pair_rows=pair_rows,
        claim_rows=claim_rows,
        definition_rows=definition_rows,
        gates=gates,
    )
    _write_csv(pair_rows, output_dir / PAIR_EVIDENCE_ROWS_CSV)
    _write_csv(claim_rows, output_dir / CLAIM_EVIDENCE_ROWS_CSV)
    _write_csv(definition_rows, output_dir / DEFINITION_DECISION_ROWS_CSV)
    _write_csv(gates, output_dir / GATE_MATRIX_CSV)
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, ensure_ascii=True, sort_keys=True),
        encoding="utf-8",
    )
    config = {
        "schema": "nanoclustering_g4_8_first_pass_transition_evidence_synthesis_config.v1",
        "source_dirs": {name: str(path) for name, path in context["paths"].items()},
        "output_dir": str(output_dir),
        "read_only_synthesis": True,
        "run_status": RUN_STATUS,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, ensure_ascii=True, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(
        output_dir=output_dir,
        summary=summary,
        pair_rows=pair_rows,
        claim_rows=claim_rows,
        definition_rows=definition_rows,
        gates=gates,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-pass-dir", type=Path, default=DEFAULT_FIRST_PASS_DIR)
    parser.add_argument("--exclusive-dir", type=Path, default=DEFAULT_EXCLUSIVE_DIR)
    parser.add_argument("--wall-trace-dir", type=Path, default=DEFAULT_WALL_TRACE_DIR)
    parser.add_argument("--transition-band-dir", type=Path, default=DEFAULT_TRANSITION_BAND_DIR)
    parser.add_argument(
        "--signature-identity-dir",
        type=Path,
        default=DEFAULT_SIGNATURE_IDENTITY_DIR,
    )
    parser.add_argument("--role-stability-dir", type=Path, default=DEFAULT_ROLE_STABILITY_DIR)
    parser.add_argument("--transfer-screen-dir", type=Path, default=DEFAULT_TRANSFER_SCREEN_DIR)
    parser.add_argument("--continuity-016-dir", type=Path, default=DEFAULT_CONTINUITY_016_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(_json_safe(summary), indent=2, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
