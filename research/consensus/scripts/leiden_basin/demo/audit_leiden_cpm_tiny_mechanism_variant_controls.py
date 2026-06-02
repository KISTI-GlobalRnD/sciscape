#!/usr/bin/env python3
"""Audit Stress 4 P0-P4 controls before endpoint evaluation.

This P4.5 audit reads only the materialized P0-P4 mechanism-variant panel. It
does not run Leiden, read endpoint manifests, evaluate handles, or mutate the
phase-locked candidate registry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from run_leiden_cpm_tiny_demo_seed_sweep import _json_safe, _write_csv


REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").exists()
)
BASE_RESULT_DIR = REPO_ROOT / "research/consensus/results/adaptive_refinement"
DEFAULT_INPUT_DIR = BASE_RESULT_DIR / "leiden_basin_tiny_cpm_mechanism_variant_panel_v1_20260531"
DEFAULT_OUTPUT_DIR = BASE_RESULT_DIR / "leiden_basin_tiny_cpm_mechanism_variant_panel_p4_5_control_audit_v1_20260601"

GRAPH_MANIFEST_CSV = "tiny_cpm_variant_graph_manifest.csv"
MECHANISM_FEATURES_CSV = "tiny_cpm_variant_mechanism_features.csv"
ROLE_INVARIANCE_CSV = "tiny_cpm_variant_role_invariance.csv"
PHASE_LOCK_JSON = "tiny_cpm_variant_phase_lock.json"
BLIND_CANDIDATE_REGISTRY_CSV = "tiny_cpm_variant_blind_candidate_registry.csv"

CANDIDATE_CLASS_MATRIX_CSV = "tiny_cpm_variant_p4_5_candidate_class_matrix.csv"
CONTROL_STRENGTH_AUDIT_CSV = "tiny_cpm_variant_p4_5_control_strength_audit.csv"
GATE_MATRIX_CSV = "tiny_cpm_variant_p4_5_gate_matrix.csv"
SUMMARY_JSON = "tiny_cpm_variant_p4_5_summary.json"
REPORT_MD = "tiny_cpm_variant_p4_5_report.md"

CLAIM_BOUNDARY = (
    "Tiny CPM mechanism-variant P4.5 control-strength audit only; reads "
    "phase-locked P0-P4 graph, role, feature, candidate, invariance, and hash "
    "artifacts. No Leiden seed sweep, no endpoint evaluation, no route/pathway "
    "execution, no wall promotion, no quality/cost claim, no NanoClustering "
    "generality claim, and no algorithm-level claim."
)
ROUTE_EXECUTION_STATUS = "not_executed_p4_5_audit_only"
WALL_PROMOTION_STATUS = "not_promoted_no_route_trace"
METHOD_STATUS = "control_strength_audit_not_method_evaluation"

TARGET_HANDLE_TYPES = {
    "near_tie_bridge": {"blind_bridge_contact_initialization"},
    "absorption_triad": {"blind_small_module_boundary_core_initialization"},
    "balanced_split": {
        "blind_middle_contact_split_initialization",
        "blind_middle_module_boundary_core_initialization",
    },
    "diffuse_fragment": {
        "blind_weak_pair_tail_split_initialization",
        "blind_joint_weak_pair_tail_split_initialization",
    },
}
CONTROL_HANDLE_MARKERS = ("control", "separate")


def _rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _with_claim_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame.copy()
    rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["method_status"] = METHOD_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_inputs(input_dir: Path) -> dict[str, Any]:
    required = {
        "manifest": GRAPH_MANIFEST_CSV,
        "features": MECHANISM_FEATURES_CSV,
        "registry": BLIND_CANDIDATE_REGISTRY_CSV,
        "invariance": ROLE_INVARIANCE_CSV,
        "phase_lock": PHASE_LOCK_JSON,
    }
    missing = [name for name in required.values() if not (input_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"missing P0-P4 input artifacts in {_rel(input_dir)}: {missing}")
    return {
        "manifest": pd.read_csv(input_dir / GRAPH_MANIFEST_CSV),
        "features": pd.read_csv(input_dir / MECHANISM_FEATURES_CSV),
        "registry": pd.read_csv(input_dir / BLIND_CANDIDATE_REGISTRY_CSV),
        "invariance": pd.read_csv(input_dir / ROLE_INVARIANCE_CSV),
        "phase_lock": _load_json(input_dir / PHASE_LOCK_JSON),
    }


def _phase_lock_audit(input_dir: Path, phase_lock: dict[str, Any]) -> tuple[bool, list[str], dict[str, str]]:
    artifact_hashes = phase_lock.get("artifact_hashes", {})
    mismatches: list[str] = []
    observed: dict[str, str] = {}
    for filename, expected in sorted(artifact_hashes.items()):
        path = input_dir / filename
        if not path.exists():
            mismatches.append(f"{filename}:missing")
            continue
        digest = _sha256_file(path)
        observed[filename] = digest
        if digest != expected:
            mismatches.append(f"{filename}:hash_mismatch")
    return not mismatches, mismatches, observed


def _is_control_state(state: str) -> bool:
    return "control" in str(state)


def _target_handle_types(family: str) -> set[str]:
    return TARGET_HANDLE_TYPES.get(str(family), set())


def _is_control_handle(handle_type: str, target_mechanism_class: str) -> bool:
    text = f"{handle_type} {target_mechanism_class}".lower()
    return any(marker in text for marker in CONTROL_HANDLE_MARKERS)


def _candidate_matrix(registry: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in registry.groupby(
        ["variant_id", "mechanism_family", "mechanism_state", "handle_type", "target_mechanism_class"],
        sort=True,
    ):
        variant_id, family, state, handle_type, target_class = keys
        is_target_like = handle_type in _target_handle_types(str(family))
        rows.append(
            {
                "variant_id": variant_id,
                "mechanism_family": family,
                "mechanism_state": state,
                "handle_type": handle_type,
                "target_mechanism_class": target_class,
                "candidate_count": int(len(group)),
                "target_claim_allowed_count": int(group["target_claim_allowed"].astype(bool).sum()),
                "target_claim_disallowed_count": int((~group["target_claim_allowed"].astype(bool)).sum()),
                "target_like_rule_class": bool(is_target_like),
                "control_handle_class": bool(_is_control_handle(str(handle_type), str(target_class))),
                "candidate_class_signatures": json.dumps(
                    sorted(group["candidate_class_signature_id"].astype(str).unique().tolist()),
                    sort_keys=True,
                ),
            }
        )
    return _with_claim_columns(pd.DataFrame(rows).sort_values(["variant_id", "handle_type", "target_mechanism_class"]))


def _mechanism_removed_feature_pass(row: pd.Series) -> tuple[bool, str]:
    family = str(row["mechanism_family"])
    if family == "near_tie_bridge":
        passed = float(row["host_dominance"]) >= 0.8 and float(row["cut_gap"]) > 0.0
        return passed, "near-tie control expects dominant host contact and positive cut gap"
    if family == "absorption_triad":
        passed = float(row["boundary_core_concentration"]) <= 0.1
        return passed, "absorption control expects no compact boundary-core concentration"
    if family == "balanced_split":
        passed = float(row["host_dominance"]) >= 0.8 or float(row["cut_gap"]) >= 5.0
        return passed, "balanced control expects one host to dominate or a large contact gap"
    if family == "diffuse_fragment":
        weak_pair_count = int(row.get("weak_pair_count", 0))
        passed = weak_pair_count == 0 and float(row["weak_pair_concentration"]) <= 0.1
        return passed, "diffuse control expects no declared weak-pair concentration"
    return False, "unknown mechanism family"


def _control_audit(
    *,
    manifest: pd.DataFrame,
    features: pd.DataFrame,
    registry: pd.DataFrame,
    min_control_target_like_decoys: int,
) -> pd.DataFrame:
    feature_by_variant = features.set_index("variant_id")
    rows = []
    for manifest_row in manifest.sort_values("variant_id").itertuples(index=False):
        variant_id = str(manifest_row.variant_id)
        family = str(manifest_row.mechanism_family)
        state = str(manifest_row.mechanism_state)
        is_control = _is_control_state(state)
        variant_registry = registry[registry["variant_id"].astype(str).eq(variant_id)]
        target_types = _target_handle_types(family)
        target_like = variant_registry[variant_registry["handle_type"].astype(str).isin(target_types)]
        target_claims = variant_registry[variant_registry["target_claim_allowed"].astype(bool)]
        control_handles = variant_registry[
            variant_registry.apply(
                lambda row: _is_control_handle(str(row["handle_type"]), str(row["target_mechanism_class"])),
                axis=1,
            )
        ]
        feature_row = feature_by_variant.loc[variant_id]
        mechanism_removed_pass, mechanism_removed_rule = _mechanism_removed_feature_pass(feature_row)

        if is_control:
            false_positive_lock_pass = len(target_claims) == 0
            target_like_decoy_pass = len(target_like) >= min_control_target_like_decoys
            if not false_positive_lock_pass:
                status = "fail_false_positive_target_claim"
                recommendation = "do not run P5-P8 until target claims are disabled for this control"
            elif not mechanism_removed_pass:
                status = "fail_mechanism_removed_feature_not_clear"
                recommendation = "revise graph control before endpoint evaluation"
            elif not target_like_decoy_pass:
                status = "weak_control_missing_target_like_decoy"
                recommendation = "revise P0-P4 with matched target-like decoy candidates before P5-P8 promotion use"
            else:
                status = "pass_control_strength_preflight"
                recommendation = "control is strong enough for diagnostic P5-P8 use"
        else:
            false_positive_lock_pass = True
            target_like_decoy_pass = True
            if len(target_claims) > 0:
                status = "pass_preserved_candidate_coverage"
                recommendation = "eligible for baseline-reproduction check in P5"
            else:
                status = "blocked_preserved_missing_target_candidate"
                recommendation = "revise candidate rules before P5-P8"

        rows.append(
            {
                "variant_id": variant_id,
                "mechanism_family": family,
                "mechanism_state": state,
                "is_control_variant": is_control,
                "candidate_count": int(len(variant_registry)),
                "target_like_candidate_count": int(len(target_like)),
                "target_claim_candidate_count": int(len(target_claims)),
                "control_handle_candidate_count": int(len(control_handles)),
                "positive_candidate_class_count": int(target_claims["candidate_class_signature_id"].nunique())
                if len(target_claims)
                else 0,
                "mechanism_removed_feature_pass": bool(mechanism_removed_pass) if is_control else "",
                "mechanism_removed_feature_rule": mechanism_removed_rule if is_control else "",
                "control_false_positive_lock_pass": bool(false_positive_lock_pass),
                "control_target_like_decoy_pass": bool(target_like_decoy_pass) if is_control else "",
                "audit_status": status,
                "recommendation": recommendation,
                "local_contact_mass": float(feature_row["local_contact_mass"]),
                "cut_gap": float(feature_row["cut_gap"]),
                "host_dominance": float(feature_row["host_dominance"]),
                "boundary_core_concentration": float(feature_row["boundary_core_concentration"]),
                "weak_pair_concentration": float(feature_row["weak_pair_concentration"]),
                "weak_pair_count": int(feature_row.get("weak_pair_count", 0)),
                "decoy_role_count": int(feature_row.get("decoy_role_count", 0)),
                "decoy_contact_mass": float(feature_row.get("decoy_contact_mass", 0.0)),
                "decoy_touched_node_count": int(feature_row.get("decoy_touched_node_count", 0)),
                "decoy_match_status": str(feature_row.get("decoy_match_status", "")),
            }
        )
    return _with_claim_columns(pd.DataFrame(rows).sort_values("variant_id"))


def _gate_matrix(
    *,
    phase_lock_verified: bool,
    phase_lock_mismatches: list[str],
    registry: pd.DataFrame,
    invariance: pd.DataFrame,
    control_audit: pd.DataFrame,
    hard_control_decoy_gate: bool,
) -> pd.DataFrame:
    excluded_ok = True
    if "excluded_before_candidate_registry" in registry.columns:
        excluded_values = " ".join(registry["excluded_before_candidate_registry"].astype(str).tolist())
        excluded_ok = all(
            marker in excluded_values
            for marker in ["frozen_endpoint_manifest", "endpoint_replay_rows", "method_hit_rows", "seed_run_outcomes"]
        )
    preserved = control_audit[~control_audit["is_control_variant"].astype(bool)]
    controls = control_audit[control_audit["is_control_variant"].astype(bool)]
    preserved_coverage_pass = bool((preserved["target_claim_candidate_count"].astype(int) > 0).all())
    control_false_positive_pass = bool(controls["control_false_positive_lock_pass"].astype(bool).all())
    control_feature_pass = bool(controls["mechanism_removed_feature_pass"].astype(bool).all())
    control_decoy_pass = bool(controls["control_target_like_decoy_pass"].astype(bool).all())
    invariance_pass = bool(invariance["candidate_class_match"].astype(bool).all())
    rows = [
        {
            "gate_id": "P4_5_G1",
            "gate_name": "phase_lock_integrity",
            "gate_question": "Do current P0-P4 files still match the phase-lock hashes?",
            "status": "pass" if phase_lock_verified else "blocked_hash_mismatch",
            "evidence": "all locked artifact hashes match" if phase_lock_verified else json.dumps(phase_lock_mismatches),
            "decision": "P5 may read P0-P4 artifacts only if pass",
        },
        {
            "gate_id": "P4_5_G2",
            "gate_name": "no_endpoint_inputs_before_registry",
            "gate_question": "Does the registry preserve the declared no-endpoint-input boundary?",
            "status": "pass" if excluded_ok else "blocked_missing_exclusion_ledger",
            "evidence": "registry rows carry excluded endpoint-evaluation inputs",
            "decision": "candidate construction remains pre-endpoint only if pass",
        },
        {
            "gate_id": "P4_5_G3",
            "gate_name": "role_name_invariance",
            "gate_question": "Do opaque node ids and role-name permutations preserve candidate classes?",
            "status": "pass" if invariance_pass else "blocked_role_name_artifact",
            "evidence": f"candidate_class_match={invariance_pass}",
            "decision": "candidate rules may advance only if pass",
        },
        {
            "gate_id": "P4_5_G4",
            "gate_name": "preserved_candidate_coverage",
            "gate_question": "Does every preserved variant have at least one positive target candidate?",
            "status": "pass" if preserved_coverage_pass else "blocked_missing_preserved_target_candidate",
            "evidence": f"preserved_variant_count={len(preserved)}",
            "decision": "P5-P8 can only evaluate preserved targets that have candidate coverage",
        },
        {
            "gate_id": "P4_5_G5",
            "gate_name": "control_false_positive_lock",
            "gate_question": "Are mechanism-removed controls prevented from making positive target claims?",
            "status": "pass" if control_false_positive_pass else "blocked_false_positive_target_claim",
            "evidence": f"control_variant_count={len(controls)}",
            "decision": "controls cannot advance if target claims are already enabled",
        },
        {
            "gate_id": "P4_5_G6",
            "gate_name": "control_mechanism_removed_features",
            "gate_question": "Do controls actually remove the intended graph mechanism?",
            "status": "pass" if control_feature_pass else "blocked_control_not_mechanism_removed",
            "evidence": f"control_feature_pass={control_feature_pass}",
            "decision": "controls are not interpretable specificity tests unless pass",
        },
        {
            "gate_id": "P4_5_G7",
            "gate_name": "control_target_like_decoy_strength",
            "gate_question": "Do controls still contain target-like decoy candidates where specificity can be tested?",
            "status": "pass"
            if control_decoy_pass
            else ("blocked_control_decoy_too_weak" if hard_control_decoy_gate else "caveat_control_decoy_too_weak"),
            "evidence": json.dumps(
                controls[
                    [
                        "variant_id",
                        "target_like_candidate_count",
                        "target_claim_candidate_count",
                        "decoy_role_count",
                        "decoy_contact_mass",
                        "decoy_match_status",
                        "audit_status",
                    ]
                ].to_dict("records"),
                sort_keys=True,
            ),
            "decision": "must pass for promotion-oriented P5-P8; hard gate blocks when enabled",
        },
    ]
    return _with_claim_columns(pd.DataFrame(rows))


def _readiness(gates: pd.DataFrame) -> str:
    statuses = set(gates["status"].astype(str))
    if any(status.startswith("blocked") for status in statuses):
        return "blocked_fix_p0_p4_before_p5_p8"
    if any(status.startswith("caveat") for status in statuses):
        return "caveated_revise_controls_before_p5_p8_promotion"
    return "ready_for_p5_p8_diagnostic_execution"


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    rows = [[str(value) for value in row] for row in frame.to_numpy().tolist()]
    table = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        table.append("| " + " | ".join(row) + " |")
    return "\n".join(table)


def _write_report(
    *,
    output_dir: Path,
    input_dir: Path,
    summary: dict[str, Any],
    control_audit: pd.DataFrame,
    gates: pd.DataFrame,
) -> None:
    weak_controls = control_audit[
        control_audit["audit_status"].astype(str).eq("weak_control_missing_target_like_decoy")
    ][["variant_id", "mechanism_family", "target_like_candidate_count", "recommendation"]]
    lines = [
        "# Tiny CPM Mechanism Variant P4.5 Control Audit",
        "",
        "This audit reads phase-locked P0-P4 artifacts only. It does not run Leiden or endpoint evaluation.",
        "",
        f"- input_dir: `{_rel(input_dir)}`",
        f"- output_dir: `{_rel(output_dir)}`",
        f"- phase_lock_verified: `{summary['phase_lock_verified']}`",
        f"- p5_p8_readiness: `{summary['p5_p8_readiness']}`",
        f"- hard_control_decoy_gate: `{summary['hard_control_decoy_gate']}`",
        f"- control_variant_count: `{summary['control_variant_count']}`",
        f"- weak_control_count: `{summary['weak_control_count']}`",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Gate Matrix",
        "",
        _markdown_table(gates[["gate_id", "gate_name", "status", "decision"]]),
        "",
        "## Weak Controls",
        "",
        _markdown_table(weak_controls) if len(weak_controls) else "No weak controls found.",
        "",
        "## Interpretation",
        "",
        "- A pass here would allow P5-P8 diagnostic execution from the frozen P0-P4 registry.",
        "- A control-decoy caveat means P5-P8 can still be diagnostic, but not promotion evidence.",
        "- A blocked gate means revise P0-P4 before reading endpoint outcomes.",
        "",
    ]
    output_dir.joinpath(REPORT_MD).write_text("\n".join(lines), encoding="utf-8")


def run_audit(
    *,
    input_dir: Path,
    output_dir: Path,
    min_control_target_like_decoys: int,
    hard_control_decoy_gate: bool,
    force: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / SUMMARY_JSON
    if summary_path.exists() and not force:
        raise FileExistsError(f"{_rel(summary_path)} already exists. Use --force to regenerate the audit.")

    inputs = _load_inputs(input_dir)
    manifest = inputs["manifest"]
    features = inputs["features"]
    registry = inputs["registry"]
    invariance = inputs["invariance"]
    phase_lock = inputs["phase_lock"]

    phase_lock_verified, phase_lock_mismatches, observed_hashes = _phase_lock_audit(input_dir, phase_lock)
    candidate_matrix = _candidate_matrix(registry)
    control_audit = _control_audit(
        manifest=manifest,
        features=features,
        registry=registry,
        min_control_target_like_decoys=min_control_target_like_decoys,
    )
    gates = _gate_matrix(
        phase_lock_verified=phase_lock_verified,
        phase_lock_mismatches=phase_lock_mismatches,
        registry=registry,
        invariance=invariance,
        control_audit=control_audit,
        hard_control_decoy_gate=hard_control_decoy_gate,
    )
    readiness = _readiness(gates)
    controls = control_audit[control_audit["is_control_variant"].astype(bool)]
    weak_controls = controls[controls["audit_status"].astype(str).eq("weak_control_missing_target_like_decoy")]
    blocked_controls = controls[controls["audit_status"].astype(str).str.startswith("fail_")]
    summary = {
        "input_dir": _rel(input_dir),
        "output_dir": _rel(output_dir),
        "phase_lock_hash": phase_lock.get("phase_lock_hash"),
        "phase_lock_verified": phase_lock_verified,
        "phase_lock_mismatches": phase_lock_mismatches,
        "observed_locked_artifact_hashes": observed_hashes,
        "variant_count": int(len(manifest)),
        "candidate_count": int(len(registry)),
        "control_variant_count": int(len(controls)),
        "weak_control_count": int(len(weak_controls)),
        "blocked_control_count": int(len(blocked_controls)),
        "weak_controls": weak_controls["variant_id"].astype(str).tolist(),
        "blocked_controls": blocked_controls["variant_id"].astype(str).tolist(),
        "p5_p8_readiness": readiness,
        "min_control_target_like_decoys": min_control_target_like_decoys,
        "hard_control_decoy_gate": hard_control_decoy_gate,
        "claim_boundary": CLAIM_BOUNDARY,
        "written_artifacts": [
            CANDIDATE_CLASS_MATRIX_CSV,
            CONTROL_STRENGTH_AUDIT_CSV,
            GATE_MATRIX_CSV,
            SUMMARY_JSON,
            REPORT_MD,
        ],
    }

    _write_csv(candidate_matrix, output_dir / CANDIDATE_CLASS_MATRIX_CSV)
    _write_csv(control_audit, output_dir / CONTROL_STRENGTH_AUDIT_CSV)
    _write_csv(gates, output_dir / GATE_MATRIX_CSV)
    summary_path.write_text(json.dumps(_json_safe(summary), indent=2, sort_keys=True), encoding="utf-8")
    _write_report(
        output_dir=output_dir,
        input_dir=input_dir,
        summary=summary,
        control_audit=control_audit,
        gates=gates,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Stress 4 P0-P4 control strength before P5-P8 endpoint evaluation."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-control-target-like-decoys", type=int, default=1)
    parser.add_argument("--hard-control-decoy-gate", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    summary = run_audit(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        min_control_target_like_decoys=args.min_control_target_like_decoys,
        hard_control_decoy_gate=args.hard_control_decoy_gate,
        force=args.force,
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
