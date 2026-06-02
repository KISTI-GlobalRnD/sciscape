#!/usr/bin/env python3
"""Design next NanoClustering basin-universe candidates after local-mask failure.

The source/target pair-mask route probes collapsed under common release and
common-mask multistart. This script does not run clustering. It materializes
the next candidate movable universes that align the optimizer probe with the
frozen event-family success unit:

1. signature-level universe: dominant-host source handles plus all top1
   endpoint handles for the endpoint-family signature;
2. case-level universe: candidate and matched-control signature universes
   together;
3. symmetric-object resolver ledger: currently available object metadata that
   can later become a mask if endpoint membership paths are resolved.

It is a design artifact only: no route execution, no wall/pathway promotion, no
basin-quality or cost claim, no real-data method success, and no algorithm
claim.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from run_leiden_basin_nanoclustering_role_local_route_pilot import (
    BASE_RESULT_DIR,
    DEFAULT_BOUNDARY_PLAN_DIR,
    DEFAULT_READINESS_DIR,
    ENDPOINT_TARGET_ROWS_CSV,
    EXECUTION_PLAN_ROWS_CSV,
    GRAPH_INPUT_ROWS_CSV,
    _json_safe,
    _load_manifest,
    _mask_for_row,
    _mask_hash,
    _read_csv,
    _source_rows,
    _target_row,
    _union_masks,
    _write_csv,
)


DEFAULT_PANEL_DESIGN_DIR = (
    BASE_RESULT_DIR / "leiden_basin_nanoclustering_joint_weak_pair_local_panel_design_20260601"
)
DEFAULT_SYMMETRIC_OBJECT_DIR = (
    BASE_RESULT_DIR / "leiden_basin_nanoclustering_symmetric_endpoint_objects_20260531"
)
DEFAULT_SYMMETRIC_DECOMPOSITION_DIR = (
    BASE_RESULT_DIR / "leiden_basin_nanoclustering_symmetric_object_decomposition_v1_20260531"
)
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR / "leiden_basin_nanoclustering_basin_universe_redesign_20260601"
)

ROLE_ROWS_CSV = "nanoclustering_joint_weak_pair_local_panel_role_rows.csv"
SIGNATURE_ROWS_CSV = "nanoclustering_joint_weak_pair_local_panel_endpoint_signature_rows.csv"
CASE_ROWS_CSV = "nanoclustering_joint_weak_pair_local_panel_case_rows.csv"
SYMMETRIC_MAPPING_CSV = "nanoclustering_seed0_v2_2_mapping_to_symmetric_objects.csv"
SYMMETRIC_PROBE_CSV = "nanoclustering_object_to_mechanism_probe_candidates.csv"

UNIVERSE_SIGNATURE_ROWS_CSV = "nanoclustering_basin_universe_signature_rows.csv"
UNIVERSE_CASE_ROWS_CSV = "nanoclustering_basin_universe_case_rows.csv"
UNIVERSE_SYMMETRIC_ROWS_CSV = "nanoclustering_basin_universe_symmetric_resolver_rows.csv"
UNIVERSE_GATE_MATRIX_CSV = "nanoclustering_basin_universe_redesign_gate_matrix.csv"
UNIVERSE_CONFIG_JSON = "nanoclustering_basin_universe_redesign_config.json"
UNIVERSE_SUMMARY_JSON = "nanoclustering_basin_universe_redesign_summary.json"
UNIVERSE_REPORT_MD = "nanoclustering_basin_universe_redesign_report.md"

CLAIM_BOUNDARY = (
    "NanoClustering basin-universe redesign only; materializes next movable "
    "universe candidates after local-mask collapse. It does not run clustering, "
    "execute routes/pathways, promote walls, inspect basin quality/cost, claim "
    "real-data method success, or claim algorithm novelty."
)


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def _json_dump(value: Any, path: Path) -> None:
    path.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _parse_handles(value: Any) -> list[str]:
    if pd.isna(value):
        return []
    return [part for part in str(value).split(";") if part]


def _mask_from_handles(
    *,
    endpoint_targets: pd.DataFrame,
    handles: list[str],
    label_cache: dict[tuple[str, str], np.ndarray],
    n_nodes: int,
) -> tuple[np.ndarray, list[str], list[str]]:
    mask = np.zeros(n_nodes, dtype=np.bool_)
    present: list[str] = []
    missing: list[str] = []
    for handle in handles:
        rows = endpoint_targets[
            endpoint_targets["endpoint_handle_id"].astype(str).eq(str(handle))
            & endpoint_targets["membership_path_exists"].astype(bool)
            & endpoint_targets["cluster_label_present"].astype(bool)
        ]
        if rows.empty:
            missing.append(handle)
            continue
        row = rows.iloc[0]
        mask |= _mask_for_row(row, label_cache)
        present.append(handle)
    return mask, present, missing


def _doc_sum(mask: np.ndarray, weights: np.ndarray) -> float:
    return float(weights[mask].sum())


def _baseline_pair_stats(boundary_plan: pd.DataFrame) -> pd.DataFrame:
    rows = boundary_plan[
        boundary_plan["route_arm"].astype(str).isin(
            [
                "source_anchor_fixed_target_free",
                "target_anchor_fixed_source_free",
                "target_handle_seeded_fixed_outside",
                "source_state_fixed_outside_control",
            ]
        )
    ].copy()
    rows = rows.drop_duplicates(
        ["endpoint_signature_id", "target_handle_id", "method_seed"]
    )
    if rows.empty:
        return pd.DataFrame()
    grouped = rows.groupby("endpoint_signature_id", sort=False).agg(
        baseline_single_target_count=("target_handle_id", "nunique"),
        baseline_pair_free_node_count_median=("pair_free_node_count", "median"),
        baseline_pair_free_node_count_max=("pair_free_node_count", "max"),
        baseline_pair_free_doc_sum_median=("pair_free_doc_sum", "median"),
        baseline_pair_free_doc_sum_max=("pair_free_doc_sum", "max"),
    )
    return grouped.reset_index()


def _graph_weights_by_branch(
    graph_rows: pd.DataFrame,
) -> dict[str, tuple[int, np.ndarray, Path]]:
    out: dict[str, tuple[int, np.ndarray, Path]] = {}
    for _, row in graph_rows.iterrows():
        if not str(row.get("runtime_graph_status", "")).startswith("ready_"):
            continue
        branch = str(row["branch"])
        manifest_path = Path(str(row["runtime_node_manifest_path"]))
        _, weights = _load_manifest(manifest_path)
        out[branch] = (len(weights), weights, manifest_path)
    return out


def _signature_universe_rows(
    *,
    endpoint_targets: pd.DataFrame,
    signature_rows: pd.DataFrame,
    role_rows: pd.DataFrame,
    boundary_plan: pd.DataFrame,
    graph_weights: dict[str, tuple[int, np.ndarray, Path]],
) -> pd.DataFrame:
    baseline = _baseline_pair_stats(boundary_plan)
    label_cache: dict[tuple[str, str], np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    role_lookup = role_rows.set_index("role_id", drop=False)
    for sig in signature_rows.itertuples(index=False):
        branch = str(sig.primitive_id).split("_seed0_", maxsplit=1)[0]
        if branch not in graph_weights:
            continue
        n_nodes, weights, manifest_path = graph_weights[branch]
        source = _source_rows(endpoint_targets, str(sig.endpoint_signature_id))
        if source.empty:
            source_mask = np.zeros(n_nodes, dtype=np.bool_)
            source_handles: list[str] = []
        else:
            source_mask = _union_masks(source, label_cache, n_nodes)
            source_handles = sorted(source["endpoint_handle_id"].astype(str).unique())
        target_handles = _parse_handles(getattr(sig, "top1_endpoint_handle_ids"))
        target_mask, present_targets, missing_targets = _mask_from_handles(
            endpoint_targets=endpoint_targets,
            handles=target_handles,
            label_cache=label_cache,
            n_nodes=n_nodes,
        )
        universe_mask = np.logical_or(source_mask, target_mask)
        role_id = (
            f"{sig.panel_case_id}__{sig.role_side}"
            if f"{sig.panel_case_id}__{sig.role_side}" in role_lookup.index
            else None
        )
        role_match = role_rows[
            role_rows["panel_case_id"].astype(str).eq(str(sig.panel_case_id))
            & role_rows["role_side"].astype(str).eq(str(sig.role_side))
        ]
        role_id = str(role_match.iloc[0]["role_id"]) if not role_match.empty else ""
        rows.append(
            {
                "universe_id": f"{sig.endpoint_signature_id}__signature_top1_union",
                "universe_scope": "signature_top1_union",
                "panel_case_id": sig.panel_case_id,
                "panel_case_rank": int(sig.panel_case_rank),
                "analysis_tier": sig.analysis_tier,
                "strict_core_v0": bool(sig.strict_core_v0),
                "role_id": role_id,
                "role_side": sig.role_side,
                "primitive_id": sig.primitive_id,
                "branch": branch,
                "endpoint_signature_id": sig.endpoint_signature_id,
                "signature_target_complexity": sig.signature_target_complexity,
                "event_count": int(sig.event_count),
                "dominant_host_unique_count": int(sig.dominant_host_unique_count),
                "top1_endpoint_unique_count": int(sig.top1_endpoint_unique_count),
                "source_handle_count": len(source_handles),
                "source_handles": ";".join(source_handles),
                "source_node_count": int(source_mask.sum()),
                "source_doc_sum": _doc_sum(source_mask, weights),
                "target_handle_count": len(target_handles),
                "resolved_target_handle_count": len(present_targets),
                "missing_target_handle_count": len(missing_targets),
                "target_handles": ";".join(target_handles),
                "missing_target_handles": ";".join(missing_targets),
                "target_union_node_count": int(target_mask.sum()),
                "target_union_doc_sum": _doc_sum(target_mask, weights),
                "universe_node_count": int(universe_mask.sum()),
                "universe_doc_sum": _doc_sum(universe_mask, weights),
                "universe_node_share": float(universe_mask.sum() / n_nodes),
                "fixed_outside_node_count": int(n_nodes - universe_mask.sum()),
                "universe_mask_hash": _mask_hash(universe_mask),
                "runtime_node_manifest_path": str(manifest_path),
                "baseline_relation": (
                    "success_unit_aligned_signature_universe_replaces_single_top1_pair_mask"
                ),
                "route_execution_status": "not_executed_universe_design_only",
                "wall_promotion_status": "not_promoted_no_route_trace",
                "quality_cost_status": "excluded_universe_design_only",
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty and not baseline.empty:
        out = out.merge(baseline, on="endpoint_signature_id", how="left")
        out["node_expansion_vs_baseline_pair_median"] = (
            out["universe_node_count"] / out["baseline_pair_free_node_count_median"]
        )
        out["node_expansion_vs_baseline_pair_max"] = (
            out["universe_node_count"] / out["baseline_pair_free_node_count_max"]
        )
        out["baseline_pair_mask_status"] = "single_top1_pair_mask_failed_common_release"
    return out


def _case_universe_rows(signature_universes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if signature_universes.empty:
        return pd.DataFrame(rows)
    for case_id, group in signature_universes.groupby("panel_case_id", sort=True):
        candidate = group[group["role_side"].astype(str).eq("candidate")]
        control = group[group["role_side"].astype(str).eq("control")]
        rows.append(
            {
                "case_universe_id": f"{case_id}__candidate_control_signature_union",
                "universe_scope": "case_candidate_control_signature_union",
                "panel_case_id": case_id,
                "panel_case_rank": int(group["panel_case_rank"].iloc[0]),
                "analysis_tier": group["analysis_tier"].iloc[0],
                "strict_core_v0": bool(group["strict_core_v0"].iloc[0]),
                "branch": group["branch"].iloc[0],
                "signature_count": int(group["endpoint_signature_id"].nunique()),
                "candidate_signature_count": int(candidate["endpoint_signature_id"].nunique()),
                "control_signature_count": int(control["endpoint_signature_id"].nunique()),
                "signature_universe_node_count_sum_upper_bound": int(
                    group["universe_node_count"].sum()
                ),
                "signature_universe_doc_sum_sum_upper_bound": float(
                    group["universe_doc_sum"].sum()
                ),
                "max_signature_universe_node_count": int(
                    group["universe_node_count"].max()
                ),
                "median_signature_universe_node_count": float(
                    group["universe_node_count"].median()
                ),
                "total_signature_target_handle_count": int(group["target_handle_count"].sum()),
                "total_resolved_target_handle_count": int(
                    group["resolved_target_handle_count"].sum()
                ),
                "missing_target_handle_count": int(group["missing_target_handle_count"].sum()),
                "candidate_target_handle_count": int(candidate["target_handle_count"].sum())
                if not candidate.empty
                else 0,
                "control_target_handle_count": int(control["target_handle_count"].sum())
                if not control.empty
                else 0,
                "case_universe_status": (
                    "design_only_requires_materialized_union_mask_before_route"
                ),
                "route_execution_status": "not_executed_universe_design_only",
                "wall_promotion_status": "not_promoted_no_route_trace",
                "quality_cost_status": "excluded_universe_design_only",
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows)


def _symmetric_resolver_rows(
    *,
    role_rows: pd.DataFrame,
    symmetric_mapping: pd.DataFrame,
    symmetric_probe: pd.DataFrame,
) -> pd.DataFrame:
    if symmetric_mapping.empty:
        return pd.DataFrame()
    rows = role_rows[
        [
            "panel_case_id",
            "panel_case_rank",
            "analysis_tier",
            "strict_core_v0",
            "role_id",
            "role_side",
            "primitive_id",
            "source_family_id",
            "branch",
            "pre_endpoint_role",
            "split_vector_class_mode",
            "host_context_class_mode",
        ]
    ].merge(
        symmetric_mapping,
        on=["branch", "source_family_id"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_symmetric"),
    )
    if not symmetric_probe.empty and "symmetric_object_id" in rows.columns:
        probe_cols = [
            "branch",
            "symmetric_object_id",
            "decomposition_class",
            "mechanism_probe_label",
            "probe_priority",
            "candidate_status",
            "dominant_internal_relation_class",
            "dominant_internal_relation_edge_count",
        ]
        rows = rows.merge(
            symmetric_probe[probe_cols].drop_duplicates(["branch", "symmetric_object_id"]),
            on=["branch", "symmetric_object_id"],
            how="left",
            suffixes=("", "_probe"),
        )
    rows["symmetric_universe_resolver_status"] = np.where(
        rows["symmetric_object_id"].notna(),
        "symmetric_object_metadata_available_mask_not_materialized",
        "no_symmetric_object_mapping_for_role_source_family",
    )
    rows["route_execution_status"] = "not_executed_universe_design_only"
    rows["wall_promotion_status"] = "not_promoted_no_route_trace"
    rows["quality_cost_status"] = "excluded_universe_design_only"
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


def _gate_matrix(
    *,
    signature_rows: pd.DataFrame,
    case_rows: pd.DataFrame,
    symmetric_rows: pd.DataFrame,
) -> pd.DataFrame:
    gates = [
        {
            "gate_id": "U1_single_top1_pair_mask_retired",
            "status": "failed_previous_gate",
            "evidence": (
                "anchor-release and common-mask multistart collapsed under "
                "single top1 pair masks"
            ),
            "next_action": "do_not_extend_anchor-arm sweeps inside same local pair mask",
        },
        {
            "gate_id": "U2_signature_universe_materialized",
            "status": "passed_design_gate" if not signature_rows.empty else "blocked",
            "evidence": f"signature_universe_rows={len(signature_rows)}",
            "next_action": "use signature-level top1 union as first executable universe",
        },
        {
            "gate_id": "U3_case_union_materialized",
            "status": "passed_design_gate" if not case_rows.empty else "blocked",
            "evidence": f"case_universe_rows={len(case_rows)}",
            "next_action": "use candidate/control case union as broader sensitivity universe",
        },
        {
            "gate_id": "U4_symmetric_object_resolver_needed",
            "status": (
                "metadata_available_mask_resolver_needed"
                if not symmetric_rows.empty
                else "blocked_no_metadata"
            ),
            "evidence": f"symmetric_resolver_rows={len(symmetric_rows)}",
            "next_action": "resolve symmetric object components to membership masks before route",
        },
    ]
    return pd.DataFrame(gates)


def _summary(
    *,
    signature_rows: pd.DataFrame,
    case_rows: pd.DataFrame,
    symmetric_rows: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "schema": "nanoclustering_basin_universe_redesign_summary.v1",
        "status": "executed_basin_universe_redesign",
        "signature_universe_count": int(len(signature_rows)),
        "case_universe_count": int(len(case_rows)),
        "symmetric_resolver_row_count": int(len(symmetric_rows)),
        "strict_core_signature_universe_count": int(signature_rows["strict_core_v0"].sum())
        if not signature_rows.empty
        else 0,
        "signature_universe_node_count_median": float(
            signature_rows["universe_node_count"].median()
        )
        if not signature_rows.empty
        else None,
        "signature_universe_node_count_max": int(
            signature_rows["universe_node_count"].max()
        )
        if not signature_rows.empty
        else 0,
        "signature_target_handle_count_median": float(
            signature_rows["target_handle_count"].median()
        )
        if not signature_rows.empty
        else None,
        "node_expansion_vs_baseline_pair_median": float(
            signature_rows["node_expansion_vs_baseline_pair_median"].median()
        )
        if "node_expansion_vs_baseline_pair_median" in signature_rows.columns
        else None,
        "case_signature_node_sum_upper_bound_median": float(
            case_rows["signature_universe_node_count_sum_upper_bound"].median()
        )
        if not case_rows.empty
        else None,
        "symmetric_mapping_available_count": int(
            symmetric_rows["symmetric_object_id"].notna().sum()
        )
        if not symmetric_rows.empty
        else 0,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _write_report(output_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# NanoClustering Basin Universe Redesign",
        "",
        f"- status: `{summary['status']}`",
        f"- signature_universe_count: {summary['signature_universe_count']}",
        f"- case_universe_count: {summary['case_universe_count']}",
        f"- symmetric_resolver_row_count: {summary['symmetric_resolver_row_count']}",
        f"- strict_core_signature_universe_count: {summary['strict_core_signature_universe_count']}",
        f"- signature_universe_node_count_median: {summary['signature_universe_node_count_median']}",
        f"- signature_universe_node_count_max: {summary['signature_universe_node_count_max']}",
        f"- signature_target_handle_count_median: {summary['signature_target_handle_count_median']}",
        f"- node_expansion_vs_baseline_pair_median: {summary['node_expansion_vs_baseline_pair_median']}",
        f"- case_signature_node_sum_upper_bound_median: {summary['case_signature_node_sum_upper_bound_median']}",
        f"- symmetric_mapping_available_count: {summary['symmetric_mapping_available_count']}",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Read",
        "",
        "The failed local route universe used single top1 endpoint handles, while "
        "the frozen success unit is endpoint-family signature distance. The next "
        "executable universe should therefore start with signature-level target "
        "handle unions, not another source/target anchor-arm sweep.",
        "",
        "Case-level candidate/control signature unions are broader sensitivity "
        "universes. Symmetric-object universes are promising but need a resolver "
        "that turns object components into branch-specific membership masks before "
        "route execution.",
        "",
        "## Artifacts",
        "",
        f"- `{_rel(output_dir / UNIVERSE_SIGNATURE_ROWS_CSV)}`",
        f"- `{_rel(output_dir / UNIVERSE_CASE_ROWS_CSV)}`",
        f"- `{_rel(output_dir / UNIVERSE_SYMMETRIC_ROWS_CSV)}`",
        f"- `{_rel(output_dir / UNIVERSE_GATE_MATRIX_CSV)}`",
        f"- `{_rel(output_dir / UNIVERSE_SUMMARY_JSON)}`",
    ]
    (output_dir / UNIVERSE_REPORT_MD).write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness-dir", type=Path, default=DEFAULT_READINESS_DIR)
    parser.add_argument("--boundary-plan-dir", type=Path, default=DEFAULT_BOUNDARY_PLAN_DIR)
    parser.add_argument("--panel-design-dir", type=Path, default=DEFAULT_PANEL_DESIGN_DIR)
    parser.add_argument("--symmetric-object-dir", type=Path, default=DEFAULT_SYMMETRIC_OBJECT_DIR)
    parser.add_argument(
        "--symmetric-decomposition-dir",
        type=Path,
        default=DEFAULT_SYMMETRIC_DECOMPOSITION_DIR,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    endpoint_targets = _read_csv(Path(args.readiness_dir) / ENDPOINT_TARGET_ROWS_CSV)
    graph_rows = _read_csv(Path(args.readiness_dir) / GRAPH_INPUT_ROWS_CSV)
    boundary_plan = _read_csv(Path(args.boundary_plan_dir) / EXECUTION_PLAN_ROWS_CSV)
    role_rows = _read_csv(Path(args.panel_design_dir) / ROLE_ROWS_CSV)
    signature_rows = _read_csv(Path(args.panel_design_dir) / SIGNATURE_ROWS_CSV)
    case_design_rows = _read_csv(Path(args.panel_design_dir) / CASE_ROWS_CSV)
    symmetric_mapping_path = Path(args.symmetric_object_dir) / SYMMETRIC_MAPPING_CSV
    symmetric_mapping = (
        _read_csv(symmetric_mapping_path) if symmetric_mapping_path.exists() else pd.DataFrame()
    )
    symmetric_probe_path = Path(args.symmetric_decomposition_dir) / SYMMETRIC_PROBE_CSV
    symmetric_probe = (
        _read_csv(symmetric_probe_path) if symmetric_probe_path.exists() else pd.DataFrame()
    )

    graph_weights = _graph_weights_by_branch(graph_rows)
    signature_universes = _signature_universe_rows(
        endpoint_targets=endpoint_targets,
        signature_rows=signature_rows,
        role_rows=role_rows,
        boundary_plan=boundary_plan,
        graph_weights=graph_weights,
    )
    case_universes = _case_universe_rows(signature_universes)
    symmetric_rows = _symmetric_resolver_rows(
        role_rows=role_rows,
        symmetric_mapping=symmetric_mapping,
        symmetric_probe=symmetric_probe,
    )
    gates = _gate_matrix(
        signature_rows=signature_universes,
        case_rows=case_universes,
        symmetric_rows=symmetric_rows,
    )
    summary = _summary(
        signature_rows=signature_universes,
        case_rows=case_universes,
        symmetric_rows=symmetric_rows,
    )

    _write_csv(signature_universes, output_dir / UNIVERSE_SIGNATURE_ROWS_CSV)
    _write_csv(case_universes, output_dir / UNIVERSE_CASE_ROWS_CSV)
    _write_csv(symmetric_rows, output_dir / UNIVERSE_SYMMETRIC_ROWS_CSV)
    _write_csv(gates, output_dir / UNIVERSE_GATE_MATRIX_CSV)
    _json_dump(
        {
            "schema": "nanoclustering_basin_universe_redesign.v1",
            "readiness_dir": str(args.readiness_dir),
            "boundary_plan_dir": str(args.boundary_plan_dir),
            "panel_design_dir": str(args.panel_design_dir),
            "symmetric_object_dir": str(args.symmetric_object_dir),
            "symmetric_decomposition_dir": str(args.symmetric_decomposition_dir),
            "output_dir": str(output_dir),
            "claim_boundary": CLAIM_BOUNDARY,
        },
        output_dir / UNIVERSE_CONFIG_JSON,
    )
    _json_dump(summary, output_dir / UNIVERSE_SUMMARY_JSON)
    _write_report(output_dir, summary)
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
