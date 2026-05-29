#!/usr/bin/env python3
"""Prepare uniform wall-probe subsets from the basin-pair panel.

This is the next gate after the representative wall-protocol panel. The initial
mode selects four pairs: one existing-route control, one non-field34 distinct
pair, one ambiguous boundary pair, and one same-zone control. The expanded
control mode keeps those four rows and adds a small field12/field30 control
panel for route-order gate replication.

No route is executed here, and no wall or basin-evaluation claim is made.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any
import sys

REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").exists()
)
SCRIPT_ROOT = REPO_ROOT / "research/consensus/scripts"
_SCRIPT_PATHS = [REPO_ROOT, SCRIPT_ROOT]
_SCRIPT_PATHS.extend(path for path in SCRIPT_ROOT.rglob("*") if path.is_dir())
for _script_path in reversed(_SCRIPT_PATHS):
    _script_path_str = str(_script_path)
    if _script_path_str not in sys.path:
        sys.path.insert(0, _script_path_str)


import pandas as pd

BASE_RESULT_DIR = REPO_ROOT / "research/consensus/results/adaptive_refinement"
COMBINED_DIR = (
    BASE_RESULT_DIR
    / "leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30"
)
DEFAULT_PANEL_DIR = BASE_RESULT_DIR / "leiden_basin_wall_protocol_panel_20260528"
DEFAULT_CALIBRATION_DIR = BASE_RESULT_DIR / "leiden_basin_definition_calibration_20260528"
DEFAULT_OUTPUT_DIR = BASE_RESULT_DIR / "leiden_basin_uniform_wall_probe_subset_20260528"

PANEL_CSV = "basin_pair_wall_protocol_panel.csv"
REQUIREMENTS_CSV = "wall_protocol_pair_requirements.csv"
ENDPOINT_ROWS_CSV = "endpoint_identity_rows.csv"

SUBSET_CSV = "uniform_wall_probe_subset.csv"
STATUS_MATRIX_CSV = "uniform_wall_probe_status_matrix.csv"
EXECUTION_MANIFEST_CSV = "uniform_wall_probe_execution_manifest.csv"
ARTIFACT_CONTRACT_CSV = "uniform_wall_probe_artifact_contract.csv"
SUMMARY_JSON = "uniform_wall_probe_subset_summary.json"
REPORT_MD = "uniform_wall_probe_subset_report.md"
CONFIG_JSON = "uniform_wall_probe_subset_config.json"

LEGACY_FIELD34_CC_LANDSCAPE = COMBINED_DIR / "basin_transition_landscape_field34_cc"
LEGACY_FIELD34_CC_BOUNDARY = COMBINED_DIR / "basin_transition_boundary_analysis_field34_cc"
LEGACY_FIELD34_CC_MINIMAL_PATHWAY = COMBINED_DIR / "basin_transition_minimal_pathway_field34_cc"
VANILLA_FIELD12_26 = COMBINED_DIR / "vanilla_reachability_sweep_field12_field26_material_core_exact"
VANILLA_FIELD30 = COMBINED_DIR / "vanilla_reachability_sweep_field30_material_core_exact"
VANILLA_FIELD34_CORE = COMBINED_DIR / "vanilla_reachability_sweep_field34_core_exact"
VANILLA_FIELD34_CC = COMBINED_DIR / "vanilla_reachability_sweep_field34_cc_n10_compatible_sketch"

def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)

def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()

def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)

def _safe_float(value: Any, default: float = math.nan) -> float:
    try:
        if pd.isna(value):
            return default
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default

def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default

def _select_first(frame: pd.DataFrame, reason: str, subset_role: str, order: int) -> pd.Series:
    if frame.empty:
        raise ValueError(f"No row selected for {subset_role}")
    row = frame.iloc[0].copy()
    row["subset_role"] = subset_role
    row["subset_order"] = order
    row["subset_selection_reason"] = reason
    return row

def _select_initial_rows(panel: pd.DataFrame) -> list[pd.Series]:
    rows: list[pd.Series] = []
    existing = panel[panel["panel_role"].eq("existing_route_diagnostic_control")].copy()
    existing["has_direct_audit"] = existing["direct_route_audit_status"].fillna("").astype(str).ne("")
    existing = existing.sort_values(
        ["has_direct_audit", "support_distance_max", "panel_pair_id"],
        ascending=[False, False, True],
    )
    rows.append(
        _select_first(
            existing,
            "existing route artifacts and direct audit are available, but only as a diagnostic control",
            "existing_route_control",
            1,
        )
    )

    distinct = panel[
        panel["panel_role"].eq("distinct_high_support_representative")
        & panel["field"].ne("field34")
    ].copy()
    distinct["support_max_num"] = pd.to_numeric(distinct["support_distance_max"], errors="coerce")
    distinct["vanilla_context_available"] = distinct.apply(
        lambda row: _case_has_vanilla_context(str(row["field"]), str(row["case_id"])),
        axis=1,
    )
    distinct = distinct.sort_values(
        ["vanilla_context_available", "support_max_num", "panel_pair_id"],
        ascending=[False, False, True],
    )
    rows.append(
        _select_first(
            distinct,
            "highest support-distance distinct representative outside field34",
            "non_field34_distinct_probe",
            2,
        )
    )

    ambiguous = panel[
        panel["panel_role"].eq("ambiguous_near_distinct_boundary")
        & panel["field"].ne("field34")
    ].copy()
    ambiguous["support_max_num"] = pd.to_numeric(ambiguous["support_distance_max"], errors="coerce")
    ambiguous["boundary_gap"] = (0.75 - ambiguous["support_max_num"]).abs()
    ambiguous["vanilla_context_available"] = ambiguous.apply(
        lambda row: _case_has_vanilla_context(str(row["field"]), str(row["case_id"])),
        axis=1,
    )
    ambiguous = ambiguous.sort_values(
        ["vanilla_context_available", "boundary_gap", "panel_pair_id"],
        ascending=[False, True, True],
    )
    rows.append(
        _select_first(
            ambiguous,
            "closest non-field34 ambiguous pair to the provisional distinct boundary",
            "ambiguous_boundary_probe",
            3,
        )
    )

    same = panel[panel["panel_role"].eq("same_or_identity_control")].copy()
    same["prefer_same_support"] = same["calibrated_relation"].eq("same_support_local")
    same = same.sort_values(["prefer_same_support", "panel_pair_id"], ascending=[False, True])
    rows.append(
        _select_first(
            same,
            "same-zone control to verify the protocol does not manufacture a wall",
            "same_zone_control",
            4,
        )
    )

    return rows

def _append_expanded_control_rows(panel: pd.DataFrame, rows: list[pd.Series]) -> None:
    selected_ids = {str(row["panel_pair_id"]) for row in rows}

    field12_distinct_method = panel[
        panel["panel_role"].eq("distinct_high_support_representative")
        & panel["field"].eq("field12")
        & ~panel["panel_pair_id"].astype(str).isin(selected_ids)
    ].copy()
    field12_distinct_method["support_max_num"] = pd.to_numeric(
        field12_distinct_method["support_distance_max"],
        errors="coerce",
    )
    field12_distinct_method["vanilla_context_available"] = field12_distinct_method.apply(
        lambda row: _case_has_vanilla_context(str(row["field"]), str(row["case_id"])),
        axis=1,
    )
    field12_distinct_method = field12_distinct_method.sort_values(
        ["vanilla_context_available", "support_max_num", "panel_pair_id"],
        ascending=[False, False, True],
    )
    rows.append(
        _select_first(
            field12_distinct_method,
            "second field12 distinct probe with vanilla context to test whether route-order sensitivity repeats across graph kinds",
            "field12_distinct_method_replicate_probe",
            5,
        )
    )

    field30_ambiguous_distinct = panel[
        panel["panel_role"].eq("ambiguous_near_distinct_boundary")
        & panel["field"].eq("field30")
        & ~panel["panel_pair_id"].astype(str).isin(selected_ids)
    ].copy()
    field30_ambiguous_distinct["support_max_num"] = pd.to_numeric(
        field30_ambiguous_distinct["support_distance_max"],
        errors="coerce",
    )
    field30_ambiguous_distinct["boundary_gap"] = (
        0.75 - field30_ambiguous_distinct["support_max_num"]
    ).abs()
    field30_ambiguous_distinct = field30_ambiguous_distinct.sort_values(
        ["boundary_gap", "panel_pair_id"],
        ascending=[True, True],
    )
    rows.append(
        _select_first(
            field30_ambiguous_distinct,
            "field30 ambiguous-near-distinct replicate near the provisional distinct threshold",
            "field30_ambiguous_distinct_replicate_probe",
            6,
        )
    )

    field30_ambiguous_same = panel[
        panel["panel_role"].eq("ambiguous_near_same_boundary")
        & panel["field"].eq("field30")
        & ~panel["panel_pair_id"].astype(str).isin(selected_ids)
    ].copy()
    field30_ambiguous_same["support_max_num"] = pd.to_numeric(
        field30_ambiguous_same["support_distance_max"],
        errors="coerce",
    )
    field30_ambiguous_same = field30_ambiguous_same.sort_values(
        ["support_max_num", "panel_pair_id"],
        ascending=[True, True],
    )
    rows.append(
        _select_first(
            field30_ambiguous_same,
            "field30 ambiguous-near-same control to check whether near-same pairs manufacture wall claims",
            "field30_ambiguous_same_control",
            7,
        )
    )

def _select_subset(panel: pd.DataFrame, selection_mode: str) -> pd.DataFrame:
    rows = _select_initial_rows(panel)
    if selection_mode == "expanded_controls":
        _append_expanded_control_rows(panel, rows)
    elif selection_mode != "initial":
        raise ValueError(f"unsupported selection_mode={selection_mode!r}")

    subset = pd.DataFrame([row.to_dict() for row in rows])
    subset["selection_mode"] = selection_mode
    public_cols = [
        "selection_mode",
        "subset_order",
        "subset_role",
        "subset_selection_reason",
        "panel_pair_id",
        "panel_role",
        "source_label",
        "case_id",
        "field",
        "method",
        "candidate_budget",
        "left_endpoint_identity_id",
        "right_endpoint_identity_id",
        "left_representative_candidate_index",
        "right_representative_candidate_index",
        "calibrated_relation",
        "endpoint_distance_min",
        "endpoint_distance_max",
        "support_distance_min",
        "support_distance_max",
        "has_existing_route_source",
        "existing_route_join_candidate",
        "existing_route_join_status",
        "existing_wall_claim_status",
        "direct_route_audit_status",
        "direct_cross_route_rows",
        "self_endpoint_route_rows",
    ]
    return subset[public_cols].sort_values("subset_order")

def _endpoint_source_lookup(endpoint_rows: pd.DataFrame) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for _, row in endpoint_rows.iterrows():
        lookup[str(row["endpoint_identity_id"])] = str(row.get("source_artifact", ""))
    return lookup

def _vanilla_dir_for_field(field: str, case_id: str) -> Path:
    if field in {"field12", "field26"}:
        return VANILLA_FIELD12_26
    if field == "field30":
        return VANILLA_FIELD30
    if field == "field34" and "all_edges_cc_cosine" in case_id:
        return VANILLA_FIELD34_CC
    if field == "field34":
        return VANILLA_FIELD34_CORE
    return Path("")

def _case_has_vanilla_context(field: str, case_id: str) -> bool:
    vanilla_dir = _vanilla_dir_for_field(field, case_id)
    if not vanilla_dir:
        return False
    return _csv_has_case(vanilla_dir / "vanilla_basin_rows.csv", case_id)

def _csv_has_case(path: Path, case_id: str) -> bool:
    if not path.exists():
        return False
    try:
        rows = pd.read_csv(path, usecols=["case"])
    except (pd.errors.EmptyDataError, ValueError):
        return False
    tail = str(case_id).removesuffix("_budget12").removesuffix("_budget15")
    return rows["case"].astype(str).str.endswith(tail).any()

def _execution_manifest(subset: pd.DataFrame, endpoint_rows: pd.DataFrame) -> pd.DataFrame:
    source_lookup = _endpoint_source_lookup(endpoint_rows)
    rows: list[dict[str, Any]] = []
    for _, row in subset.iterrows():
        field = str(row["field"])
        case_id = str(row["case_id"])
        left_id = str(row["left_endpoint_identity_id"])
        right_id = str(row["right_endpoint_identity_id"])
        vanilla_dir = _vanilla_dir_for_field(field, case_id)
        vanilla_rows_path = vanilla_dir / "vanilla_basin_rows.csv"
        vanilla_available = vanilla_rows_path.exists() and _csv_has_case(vanilla_rows_path, case_id)
        legacy_field34_cc_ready = (
            field == "field34"
            and "all_edges_cc_cosine" in case_id
            and (LEGACY_FIELD34_CC_LANDSCAPE / "basin_transition_landscape_hypotheses.csv").exists()
            and (LEGACY_FIELD34_CC_BOUNDARY / "basin_transition_boundary_group_rows.csv").exists()
            and (LEGACY_FIELD34_CC_MINIMAL_PATHWAY / "basin_transition_minimal_pathway_pairs.csv").exists()
        )
        uniform_ready = False
        if legacy_field34_cc_ready:
            readiness = "legacy_field34_cc_context_available_not_uniform_pair_route"
            next_action = "use as diagnostic control; implement uniform direct pair-route runner before accepting W1"
        elif vanilla_available:
            readiness = "vanilla_graph_context_available_boundary_and_uniform_runner_missing"
            next_action = "generate pair boundary context and run uniform direct pair-route runner"
        else:
            readiness = "missing_vanilla_graph_context"
            next_action = "locate graph context before route construction"
        rows.append(
            {
                "subset_order": int(row["subset_order"]),
                "subset_role": str(row["subset_role"]),
                "panel_pair_id": str(row["panel_pair_id"]),
                "case_id": case_id,
                "left_endpoint_identity_id": left_id,
                "right_endpoint_identity_id": right_id,
                "left_representative_candidate_index": int(row["left_representative_candidate_index"]),
                "right_representative_candidate_index": int(row["right_representative_candidate_index"]),
                "left_endpoint_source_artifact": source_lookup.get(left_id, ""),
                "right_endpoint_source_artifact": source_lookup.get(right_id, ""),
                "vanilla_context_dir": _rel(vanilla_dir) if vanilla_dir else "",
                "vanilla_context_status": "available" if vanilla_available else "missing",
                "legacy_field34_cc_pathway_status": "available" if legacy_field34_cc_ready else "not_applicable",
                "uniform_direct_pair_route_status": "available" if uniform_ready else "missing",
                "execution_readiness": readiness,
                "next_action": next_action,
                "claim_boundary": "Execution readiness only; no wall or basin-evaluation claim is made.",
            }
        )
    return pd.DataFrame(rows)

def _artifact_contract() -> pd.DataFrame:
    rows = [
        {
            "artifact_name": "uniform_direct_pair_route_rows.csv",
            "protocol_step_id": "W1",
            "required_columns": "panel_pair_id;route_id;step_index;source_endpoint_identity_id;target_endpoint_identity_id;state_membership_hash;edited_node_count;edited_node_ids",
            "acceptance_rule": "rows name both endpoint identities and form a single ordered route",
            "forbidden_use": "cannot compare basin value or define basin identity",
        },
        {
            "artifact_name": "uniform_objective_wall_rows.csv",
            "protocol_step_id": "W2",
            "required_columns": "panel_pair_id;route_id;step_index;objective_value;objective_debt_from_start;objective_recovery_from_min;wall_step_flag",
            "acceptance_rule": "objective fields are measured on the same W1 direct route",
            "forbidden_use": "cannot rank basins by final value",
        },
        {
            "artifact_name": "uniform_support_movement_rows.csv",
            "protocol_step_id": "W3",
            "required_columns": "panel_pair_id;route_id;step_index;support_distance_to_source;support_distance_to_target;endpoint_distance_to_source;endpoint_distance_to_target",
            "acceptance_rule": "support and endpoint distances are attached to the same W1 direct route",
            "forbidden_use": "cannot redefine endpoint identity",
        },
        {
            "artifact_name": "uniform_polish_reversion_rows.csv",
            "protocol_step_id": "W4",
            "required_columns": "panel_pair_id;route_id;pre_polish_state_id;post_polish_state_id;post_polish_endpoint_assignment;reversion_status",
            "acceptance_rule": "post-polish assignment uses the calibrated endpoint definition",
            "forbidden_use": "raw labels cannot be treated as basin identity",
        },
        {
            "artifact_name": "uniform_route_label_rows.csv",
            "protocol_step_id": "W5",
            "required_columns": "panel_pair_id;route_id;route_label;route_label_confidence;wall_assignment_status;support_assignment_status;evidence_notes",
            "acceptance_rule": "route label cites W1-W4 rows and is supported, partial, or unknown",
            "forbidden_use": "cannot make directed-search or basin-evaluation claims",
        },
        {
            "artifact_name": "uniform_route_schedule_claim_rows.csv",
            "protocol_step_id": "W6",
            "required_columns": "panel_pair_id;schedule_count;route_order_sensitivity_status;wall_claim_gate_status;route_labels;wall_assignment_statuses",
            "acceptance_rule": "pair-level gate blocks route-order-sensitive labels from promotion",
            "forbidden_use": "cannot rank basin value or bypass route controls",
        },
    ]
    return pd.DataFrame(rows)

def _status_matrix(subset: pd.DataFrame, requirements: pd.DataFrame) -> pd.DataFrame:
    selected_ids = set(subset["panel_pair_id"].astype(str))
    status = requirements[requirements["panel_pair_id"].astype(str).isin(selected_ids)].copy()
    order_lookup = subset.set_index("panel_pair_id")["subset_order"].to_dict()
    role_lookup = subset.set_index("panel_pair_id")["subset_role"].to_dict()
    status["subset_order"] = status["panel_pair_id"].map(order_lookup)
    status["subset_role"] = status["panel_pair_id"].map(role_lookup)
    cols = [
        "subset_order",
        "subset_role",
        "panel_pair_id",
        "protocol_step_id",
        "protocol_step",
        "requirement_status",
        "available_evidence_status",
        "missing_artifact",
        "claim_boundary",
    ]
    return status[cols].sort_values(["subset_order", "protocol_step_id"])

def _summary(
    subset: pd.DataFrame,
    status: pd.DataFrame,
    manifest: pd.DataFrame,
    selection_mode: str,
) -> dict[str, Any]:
    return {
        "status": "uniform_wall_probe_subset_prepared",
        "date": "2026-05-28",
        "selection_mode": selection_mode,
        "subset_pair_count": int(len(subset)),
        "subset_roles": subset["subset_role"].tolist(),
        "selected_pair_ids": subset["panel_pair_id"].tolist(),
        "requirement_status_counts": status["requirement_status"].value_counts().to_dict(),
        "execution_readiness_counts": manifest["execution_readiness"].value_counts().to_dict(),
        "uniform_direct_pair_ready_count": int(manifest["uniform_direct_pair_route_status"].eq("available").sum()),
        "decision": (
            "run the uniform direct pair-route runner and use route-schedule "
            "claim rows before any wall promotion"
        ),
            "claim_boundary": (
            "This subset is an execution manifest for W0-W6 protocol testing only; "
            "no wall, basin evaluation, or directed-search claim is made."
        ),
    }

def _write_report(
    path: Path,
    summary: dict[str, Any],
    subset: pd.DataFrame,
    status: pd.DataFrame,
    manifest: pd.DataFrame,
) -> None:
    selection_mode = str(summary.get("selection_mode", "initial"))
    lines = [
        "# Leiden Basin Uniform Wall-Probe Subset",
        "",
        f"Status: {selection_mode} subset prepared for uniform W0-W6 wall-protocol testing",
        "Date: 2026-05-28",
        "",
        "This artifact selects panel pairs and records what is needed before a fair wall comparison can be made. It does not run a route, compare basin value fields, or make a wall claim.",
        "",
        "## Selected Pairs",
        "",
        "| order | role | pair_id | relation | support_max | readiness |",
        "| ---: | --- | --- | --- | ---: | --- |",
    ]
    readiness = manifest.set_index("panel_pair_id")["execution_readiness"].to_dict()
    for _, row in subset.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["subset_order"]),
                    str(row["subset_role"]),
                    str(row["panel_pair_id"]),
                    str(row["calibrated_relation"]),
                    str(row["support_distance_max"]),
                    readiness.get(str(row["panel_pair_id"]), ""),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Requirement Status",
            "",
            "| status | rows |",
            "| --- | ---: |",
        ]
    )
    for status_name, count in sorted(summary["requirement_status_counts"].items()):
        lines.append(f"| {status_name} | {count} |")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- The subset is ready as an execution manifest, not as a wall result.",
            "- No selected pair currently has an accepted uniform direct pair-route trace.",
            "- Existing field34/cc artifacts remain useful only as diagnostic context because they are candidate-vs-vanilla or self-endpoint traces.",
            "- The next implementation task is a uniform direct pair-route runner that produces W1-W6 rows and route-schedule claim gates for these pairs.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def run(
    panel_dir: Path,
    calibration_dir: Path,
    output_dir: Path,
    selection_mode: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    panel = _read_csv(panel_dir / PANEL_CSV)
    requirements = _read_csv(panel_dir / REQUIREMENTS_CSV)
    endpoint_rows = _read_csv(calibration_dir / ENDPOINT_ROWS_CSV)
    if panel.empty:
        raise FileNotFoundError(panel_dir / PANEL_CSV)
    if requirements.empty:
        raise FileNotFoundError(panel_dir / REQUIREMENTS_CSV)
    if endpoint_rows.empty:
        raise FileNotFoundError(calibration_dir / ENDPOINT_ROWS_CSV)

    subset = _select_subset(panel, selection_mode)
    status = _status_matrix(subset, requirements)
    manifest = _execution_manifest(subset, endpoint_rows)
    contract = _artifact_contract()
    summary = _summary(subset, status, manifest, selection_mode)

    _write_csv(subset, output_dir / SUBSET_CSV)
    _write_csv(status, output_dir / STATUS_MATRIX_CSV)
    _write_csv(manifest, output_dir / EXECUTION_MANIFEST_CSV)
    _write_csv(contract, output_dir / ARTIFACT_CONTRACT_CSV)
    (output_dir / SUMMARY_JSON).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(
            {
                "script": _rel(Path(__file__)),
                "panel_dir": _rel(panel_dir),
                "calibration_dir": _rel(calibration_dir),
                "selection_mode": selection_mode,
                "scope": "execution manifest only; no route execution",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_report(output_dir / REPORT_MD, summary, subset, status, manifest)
    return {"output_dir": _rel(output_dir), **summary}

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-dir", type=Path, default=DEFAULT_PANEL_DIR)
    parser.add_argument("--calibration-dir", type=Path, default=DEFAULT_CALIBRATION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--selection-mode",
        choices=["initial", "expanded_controls"],
        default="initial",
    )
    args = parser.parse_args()
    print(json.dumps(
        run(args.panel_dir, args.calibration_dir, args.output_dir, args.selection_mode),
        indent=2,
    ))

if __name__ == "__main__":
    main()
