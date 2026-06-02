#!/usr/bin/env python3
"""Materialize role-local boundary objects for NanoClustering replay.

This is the next contract step after the full-partition warm-start pilot.  It
does not run Leiden.  It converts frozen endpoint-family handles into explicit
role-local source/target masks and a fixed-outside route plan so that the next
runner can test local basin/pathway interventions instead of replaying an
entire seed partition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").exists()
)
BASE_RESULT_DIR = REPO_ROOT / "research/consensus/results/adaptive_refinement"
DEFAULT_READINESS_DIR = (
    BASE_RESULT_DIR / "leiden_basin_nanoclustering_endpoint_replay_readiness_20260601"
)
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_role_local_boundary_plan_20260601"
)

ENDPOINT_TARGET_ROWS_CSV = (
    "nanoclustering_endpoint_replay_readiness_endpoint_target_rows.csv"
)
ATTEMPT_PLAN_ROWS_CSV = "nanoclustering_endpoint_replay_readiness_attempt_plan_rows.csv"
GRAPH_INPUT_ROWS_CSV = "nanoclustering_endpoint_replay_readiness_graph_input_rows.csv"

OBJECT_ROWS_CSV = "nanoclustering_role_local_boundary_object_rows.csv"
TARGET_HANDLE_ROWS_CSV = "nanoclustering_role_local_boundary_target_handle_rows.csv"
CASE_CONTRAST_ROWS_CSV = "nanoclustering_role_local_boundary_case_contrast_rows.csv"
EXECUTION_PLAN_ROWS_CSV = "nanoclustering_role_local_boundary_execution_plan_rows.csv"
GATE_MATRIX_CSV = "nanoclustering_role_local_boundary_gate_matrix.csv"
SUMMARY_JSON = "nanoclustering_role_local_boundary_plan_summary.json"
CONFIG_JSON = "nanoclustering_role_local_boundary_plan_config.json"
REPORT_MD = "nanoclustering_role_local_boundary_plan_report.md"

CLAIM_BOUNDARY = (
    "NanoClustering role-local boundary plan only; materializes source/target "
    "node-mask objects and fixed-outside route contracts after the full-partition "
    "warm-start pilot failed to distinguish roles. It does not run clustering, "
    "execute routes/pathways, promote walls, inspect quality/cost, or claim "
    "real-data method success."
)
REPLAY_EXECUTION_STATUS = "not_executed_role_local_boundary_plan_only"
ROUTE_EXECUTION_STATUS = "not_executed_fixed_mask_plan_only"
WALL_PROMOTION_STATUS = "not_promoted_no_route_trace"
QUALITY_COST_STATUS = "excluded_boundary_plan_only"
DEFAULT_METHOD_SEEDS = tuple(range(10))
ROUTE_ARMS = (
    "source_state_fixed_outside_control",
    "target_handle_seeded_fixed_outside",
    "target_anchor_fixed_source_free",
    "source_anchor_fixed_target_free",
)


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _parse_seed_list(value: str | None) -> tuple[int, ...]:
    if value is None or not str(value).strip():
        return DEFAULT_METHOD_SEEDS
    return tuple(int(part.strip()) for part in str(value).split(",") if part.strip())


def _with_claim_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame.copy()
    rows["replay_execution_status"] = REPLAY_EXECUTION_STATUS
    rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["quality_cost_status"] = QUALITY_COST_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


def _mask_hash(mask: np.ndarray) -> str:
    packed = np.packbits(np.asarray(mask, dtype=np.bool_))
    return hashlib.blake2b(packed.tobytes(), digest_size=16).hexdigest()


def _jaccard(left: np.ndarray, right: np.ndarray) -> float:
    union = np.logical_or(left, right)
    denom = int(union.sum())
    if denom == 0:
        return 1.0
    return float(np.logical_and(left, right).sum() / denom)


def _mask_stats(mask: np.ndarray, weights: np.ndarray) -> tuple[int, float]:
    selected = np.asarray(mask, dtype=np.bool_)
    return int(selected.sum()), float(weights[selected].sum())


def _load_manifest(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.read_parquet(
        path,
        columns=["node_idx", "original_cluster_id", "doc_count"],
    ).sort_values("node_idx", kind="mergesort")
    expected = np.arange(len(frame), dtype=np.int64)
    if not np.array_equal(frame["node_idx"].to_numpy(dtype=np.int64), expected):
        raise ValueError(f"node_idx is not dense and sorted in {path}")
    return frame, frame["doc_count"].to_numpy(dtype=np.float64)


def _load_label_array(path: Path, label_col: str) -> np.ndarray:
    frame = pd.read_parquet(path, columns=["node_idx", label_col]).sort_values(
        "node_idx",
        kind="mergesort",
    )
    expected = np.arange(len(frame), dtype=np.int64)
    if not np.array_equal(frame["node_idx"].to_numpy(dtype=np.int64), expected):
        raise ValueError(f"node_idx is not dense and sorted in {path}")
    return frame[label_col].to_numpy(dtype=np.int64)


def _mask_for_row(row: pd.Series, cache: dict[tuple[str, str], np.ndarray]) -> np.ndarray:
    path = str(row["membership_path"])
    label_col = str(row["label_cols"]).split(";")[0] or "candidate_micro_id"
    key = (path, label_col)
    if key not in cache:
        cache[key] = _load_label_array(Path(path), label_col)
    return cache[key] == int(row["cluster_id"])


def _union_masks(rows: pd.DataFrame, cache: dict[tuple[str, str], np.ndarray], n_nodes: int) -> np.ndarray:
    out = np.zeros(n_nodes, dtype=np.bool_)
    for _, row in rows.iterrows():
        out |= _mask_for_row(row, cache)
    return out


def _selected_role_rows(attempt_plan: pd.DataFrame, analysis_tier: str) -> pd.DataFrame:
    return (
        attempt_plan[
            attempt_plan["analysis_tier"].astype(str).eq(analysis_tier)
            & attempt_plan["method_seed"].astype(int).eq(0)
            & attempt_plan["attempt_execution_status"].astype(str).eq("ready_to_execute")
        ]
        .drop_duplicates(["role_id"])
        .sort_values(["panel_case_rank", "role_side"], kind="mergesort")
        .reset_index(drop=True)
    )


def _role_status(*, source_count: int, target_count: int, free_share: float) -> str:
    if source_count == 0:
        return "blocked_missing_source_mask"
    if target_count == 0:
        return "blocked_missing_target_mask"
    if free_share >= 0.2:
        return "caveated_large_free_mask"
    return "ready_role_local_fixed_mask_contract"


def _contrast_status(source_jaccard: float, local_union_jaccard: float) -> str:
    if source_jaccard == 1.0 and local_union_jaccard == 1.0:
        return "blocked_identical_role_local_objects"
    if source_jaccard < 0.25 or local_union_jaccard < 0.5:
        return "distinct_role_local_objects"
    return "overlapping_but_distinguishable_role_local_objects"


def _materialize(
    *,
    readiness_dir: Path,
    output_dir: Path,
    analysis_tier: str,
    method_seeds: tuple[int, ...],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    endpoint_targets = _read_csv(readiness_dir / ENDPOINT_TARGET_ROWS_CSV)
    attempt_plan = _read_csv(readiness_dir / ATTEMPT_PLAN_ROWS_CSV)
    graph_rows = _read_csv(readiness_dir / GRAPH_INPUT_ROWS_CSV)
    role_rows = _selected_role_rows(attempt_plan, analysis_tier)

    manifest_by_branch: dict[str, tuple[pd.DataFrame, np.ndarray]] = {}
    for _, graph_row in graph_rows.iterrows():
        branch = str(graph_row["branch"])
        if not str(graph_row.get("runtime_graph_status", "")).startswith("ready_"):
            continue
        manifest_by_branch[branch] = _load_manifest(Path(str(graph_row["runtime_node_manifest_path"])))

    mask_cache: dict[tuple[str, str], np.ndarray] = {}
    role_masks: dict[str, dict[str, Any]] = {}
    object_rows: list[dict[str, Any]] = []
    target_rows_out: list[dict[str, Any]] = []
    execution_rows: list[dict[str, Any]] = []

    for _, role in role_rows.iterrows():
        signature_id = str(role["target_endpoint_signature_id"])
        branch = str(role["branch"])
        if branch not in manifest_by_branch:
            raise ValueError(f"missing ready graph manifest for branch={branch}")
        manifest, weights = manifest_by_branch[branch]
        n_nodes = int(len(manifest))
        signature_targets = endpoint_targets[
            endpoint_targets["endpoint_signature_id"].astype(str).eq(signature_id)
            & endpoint_targets["membership_path_exists"].astype(bool)
            & endpoint_targets["cluster_label_present"].astype(bool)
        ].copy()
        source_rows = signature_targets[
            signature_targets["target_handle_role"].astype(str).eq(
                "dominant_host_context_member"
            )
        ].sort_values(["seed", "endpoint_handle_id"], kind="mergesort")
        family_target_rows = signature_targets[
            signature_targets["target_handle_role"].astype(str).eq(
                "top1_endpoint_target_member"
            )
        ].sort_values(["seed", "endpoint_handle_id"], kind="mergesort")
        source_mask = _union_masks(source_rows, mask_cache, n_nodes)
        family_target_mask = _union_masks(family_target_rows, mask_cache, n_nodes)
        role_union_mask = np.logical_or(source_mask, family_target_mask)
        source_count, source_weight = _mask_stats(source_mask, weights)
        target_union_count, target_union_weight = _mask_stats(family_target_mask, weights)
        local_union_count, local_union_weight = _mask_stats(role_union_mask, weights)
        source_family_overlap = np.logical_and(source_mask, family_target_mask)
        source_family_overlap_count, source_family_overlap_weight = _mask_stats(
            source_family_overlap,
            weights,
        )
        free_share = local_union_count / n_nodes if n_nodes else 0.0
        role_id = str(role["role_id"])
        role_masks[role_id] = {
            "source_mask": source_mask,
            "family_target_mask": family_target_mask,
            "role_union_mask": role_union_mask,
            "weights": weights,
            "n_nodes": n_nodes,
            "object_row_base": role.to_dict(),
        }
        object_rows.append(
            {
                "role_local_object_id": f"{role_id}__role_local_boundary_object",
                "panel_case_id": role["panel_case_id"],
                "panel_case_rank": int(role["panel_case_rank"]),
                "analysis_tier": role["analysis_tier"],
                "strict_core_v0": bool(role["strict_core_v0"]),
                "role_id": role_id,
                "role_side": role["role_side"],
                "primitive_id": role["primitive_id"],
                "branch": branch,
                "endpoint_signature_id": signature_id,
                "source_handle_count": int(len(source_rows)),
                "target_handle_count": int(len(family_target_rows)),
                "n_nodes": n_nodes,
                "source_node_count": source_count,
                "source_doc_sum": source_weight,
                "target_union_node_count": target_union_count,
                "target_union_doc_sum": target_union_weight,
                "source_target_union_node_count": local_union_count,
                "source_target_union_doc_sum": local_union_weight,
                "source_target_union_node_share": free_share,
                "fixed_outside_node_count": n_nodes - local_union_count,
                "fixed_outside_node_share": 1.0 - free_share,
                "source_family_overlap_node_count": source_family_overlap_count,
                "source_family_overlap_doc_sum": source_family_overlap_weight,
                "source_family_overlap_source_doc_share": (
                    source_family_overlap_weight / source_weight if source_weight else 0.0
                ),
                "source_family_overlap_target_doc_share": (
                    source_family_overlap_weight / target_union_weight
                    if target_union_weight
                    else 0.0
                ),
                "source_mask_hash": _mask_hash(source_mask),
                "target_union_mask_hash": _mask_hash(family_target_mask),
                "role_union_mask_hash": _mask_hash(role_union_mask),
                "role_local_contract_status": _role_status(
                    source_count=source_count,
                    target_count=target_union_count,
                    free_share=free_share,
                ),
                "next_execution_object": (
                    "fixed_nodes_outside_source_plus_target_handle_mask"
                ),
            }
        )
        for _, target in family_target_rows.iterrows():
            target_mask = _mask_for_row(target, mask_cache)
            pair_mask = np.logical_or(source_mask, target_mask)
            overlap_mask = np.logical_and(source_mask, target_mask)
            target_count, target_weight = _mask_stats(target_mask, weights)
            pair_count, pair_weight = _mask_stats(pair_mask, weights)
            overlap_count, overlap_weight = _mask_stats(overlap_mask, weights)
            source_recall_weight = overlap_weight / source_weight if source_weight else 0.0
            target_precision_weight = overlap_weight / target_weight if target_weight else 0.0
            target_row = {
                "role_local_target_id": (
                    f"{role_id}__target_seed{int(target['seed']):03d}"
                    f"_{target['handle_kind']}{int(target['cluster_id'])}"
                ),
                "role_local_object_id": f"{role_id}__role_local_boundary_object",
                "panel_case_id": role["panel_case_id"],
                "panel_case_rank": int(role["panel_case_rank"]),
                "analysis_tier": role["analysis_tier"],
                "role_id": role_id,
                "role_side": role["role_side"],
                "primitive_id": role["primitive_id"],
                "branch": branch,
                "endpoint_signature_id": signature_id,
                "target_handle_id": target["endpoint_handle_id"],
                "target_run_id": target["run_id"],
                "target_seed": int(target["seed"]),
                "target_cluster_id": int(target["cluster_id"]),
                "source_node_count": source_count,
                "source_doc_sum": source_weight,
                "target_node_count": target_count,
                "target_doc_sum": target_weight,
                "source_target_overlap_node_count": overlap_count,
                "source_target_overlap_doc_sum": overlap_weight,
                "source_target_overlap_source_doc_share": source_recall_weight,
                "source_target_overlap_target_doc_share": target_precision_weight,
                "pair_free_node_count": pair_count,
                "pair_free_doc_sum": pair_weight,
                "pair_free_node_share": pair_count / n_nodes if n_nodes else 0.0,
                "pair_fixed_outside_node_count": n_nodes - pair_count,
                "pair_fixed_outside_node_share": 1.0 - (pair_count / n_nodes if n_nodes else 0.0),
                "target_mask_hash": _mask_hash(target_mask),
                "pair_mask_hash": _mask_hash(pair_mask),
                "route_contract_status": _role_status(
                    source_count=source_count,
                    target_count=target_count,
                    free_share=pair_count / n_nodes if n_nodes else 0.0,
                ),
            }
            target_rows_out.append(target_row)
            for method_seed in method_seeds:
                for route_arm in ROUTE_ARMS:
                    execution_rows.append(
                        {
                            **target_row,
                            "route_attempt_id": (
                                f"{target_row['role_local_target_id']}__"
                                f"{route_arm}__method_seed{int(method_seed):03d}"
                            ),
                            "method_seed": int(method_seed),
                            "route_arm": route_arm,
                            "initial_membership_base": (
                                "source_seed0_full_partition"
                            ),
                            "fixed_nodes_policy": "fix_all_nodes_outside_pair_mask",
                            "free_nodes_policy": (
                                "source_dominant_union_plus_single_top1_target_handle"
                            ),
                            "route_intervention_policy": (
                                "no_relabel_control"
                                if route_arm == "source_state_fixed_outside_control"
                                else "assign_target_handle_nodes_to_fresh_label_before_leiden"
                                if route_arm == "target_handle_seeded_fixed_outside"
                                else "fix_target_handle_nodes_to_fresh_label_and_free_source_only"
                                if route_arm == "target_anchor_fixed_source_free"
                                else "fix_source_nodes_to_seed0_labels_and_free_target_only"
                            ),
                            "success_readout": (
                                "terminal_distance_to_target_handle_and_role_family"
                            ),
                        }
                    )

    contrast_rows: list[dict[str, Any]] = []
    for case_id, group in role_rows.groupby("panel_case_id", sort=False):
        sides = {str(row["role_side"]): str(row["role_id"]) for _, row in group.iterrows()}
        if "candidate" not in sides or "control" not in sides:
            continue
        cand = role_masks[sides["candidate"]]
        ctrl = role_masks[sides["control"]]
        weights = cand["weights"]
        cand_union = cand["role_union_mask"]
        ctrl_union = ctrl["role_union_mask"]
        shared_union = np.logical_and(cand_union, ctrl_union)
        shared_count, shared_weight = _mask_stats(shared_union, weights)
        contrast_rows.append(
            {
                "panel_case_id": case_id,
                "panel_case_rank": int(group["panel_case_rank"].iloc[0]),
                "analysis_tier": group["analysis_tier"].iloc[0],
                "candidate_role_id": sides["candidate"],
                "control_role_id": sides["control"],
                "branch": group["branch"].iloc[0],
                "candidate_source_node_count": int(cand["source_mask"].sum()),
                "control_source_node_count": int(ctrl["source_mask"].sum()),
                "candidate_union_node_count": int(cand_union.sum()),
                "control_union_node_count": int(ctrl_union.sum()),
                "shared_union_node_count": shared_count,
                "shared_union_doc_sum": shared_weight,
                "source_mask_jaccard": _jaccard(cand["source_mask"], ctrl["source_mask"]),
                "target_union_mask_jaccard": _jaccard(
                    cand["family_target_mask"],
                    ctrl["family_target_mask"],
                ),
                "role_union_mask_jaccard": _jaccard(cand_union, ctrl_union),
                "role_local_contrast_status": _contrast_status(
                    _jaccard(cand["source_mask"], ctrl["source_mask"]),
                    _jaccard(cand_union, ctrl_union),
                ),
            }
        )

    object_frame = _with_claim_columns(pd.DataFrame(object_rows))
    target_frame = _with_claim_columns(pd.DataFrame(target_rows_out))
    contrast_frame = _with_claim_columns(pd.DataFrame(contrast_rows))
    execution_frame = _with_claim_columns(pd.DataFrame(execution_rows))

    gate_rows = [
        {
            "gate_id": "B1_readiness_loaded",
            "gate_question": "Did the readiness artifact provide executable strict-core roles?",
            "status": "pass" if not role_rows.empty else "fail_no_roles",
            "evidence": f"role_count={len(role_rows)}",
            "decision": "role-local planning can proceed only from frozen ready roles",
        },
        {
            "gate_id": "B2_source_masks_materialized",
            "gate_question": "Does every role have a nonempty source seed0 mask?",
            "status": (
                "pass"
                if not object_frame.empty and (object_frame["source_node_count"] > 0).all()
                else "fail_empty_source_mask"
            ),
            "evidence": (
                f"empty_source_roles={int((object_frame['source_node_count'] <= 0).sum())}"
                if not object_frame.empty
                else "empty_source_roles=unknown"
            ),
            "decision": "full partition replay is replaced by source-handle-local objects",
        },
        {
            "gate_id": "B3_target_masks_materialized",
            "gate_question": "Does every role have at least one target endpoint handle?",
            "status": (
                "pass"
                if not object_frame.empty and (object_frame["target_handle_count"] > 0).all()
                else "fail_empty_target_mask"
            ),
            "evidence": (
                f"target_handles={len(target_frame)}"
                if not target_frame.empty
                else "target_handles=0"
            ),
            "decision": "target handles are diagnostic members for route contracts",
        },
        {
            "gate_id": "B4_role_local_contrast",
            "gate_question": "Are candidate/control objects distinguishable at mask level?",
            "status": (
                "pass"
                if not contrast_frame.empty
                and not contrast_frame["role_local_contrast_status"]
                .astype(str)
                .eq("blocked_identical_role_local_objects")
                .any()
                else "fail_identical_role_local_objects"
            ),
            "evidence": (
                contrast_frame["role_local_contrast_status"]
                .value_counts(dropna=False)
                .to_dict()
                if not contrast_frame.empty
                else {}
            ),
            "decision": "candidate/control role distinction must exist before route execution",
        },
        {
            "gate_id": "B5_fixed_mask_route_contract",
            "gate_question": "Is a fixed-outside route plan materialized?",
            "status": "pass" if not execution_frame.empty else "fail_empty_route_plan",
            "evidence": f"route_plan_rows={len(execution_frame)}",
            "decision": "next runner should execute bounded route arms, not full-grid warm start",
        },
        {
            "gate_id": "B6_claim_boundary_closed",
            "gate_question": "Are route/wall and quality/cost claims still closed?",
            "status": "closed_by_design",
            "evidence": CLAIM_BOUNDARY,
            "decision": "plan-only artifact cannot promote method success",
        },
    ]
    gate_frame = _with_claim_columns(pd.DataFrame(gate_rows))

    _write_csv(object_frame, output_dir / OBJECT_ROWS_CSV)
    _write_csv(target_frame, output_dir / TARGET_HANDLE_ROWS_CSV)
    _write_csv(contrast_frame, output_dir / CASE_CONTRAST_ROWS_CSV)
    _write_csv(execution_frame, output_dir / EXECUTION_PLAN_ROWS_CSV)
    _write_csv(gate_frame, output_dir / GATE_MATRIX_CSV)

    summary = {
        "schema": "nanoclustering_role_local_boundary_plan_summary.v1",
        "status": "materialized_role_local_boundary_plan",
        "readiness_dir": str(readiness_dir),
        "output_dir": str(output_dir),
        "analysis_tier": analysis_tier,
        "role_count": int(len(object_frame)),
        "case_contrast_count": int(len(contrast_frame)),
        "target_handle_count": int(len(target_frame)),
        "route_plan_row_count": int(len(execution_frame)),
        "method_seed_count": int(len(method_seeds)),
        "route_arm_count": int(len(ROUTE_ARMS)),
        "median_pair_free_node_count": (
            float(target_frame["pair_free_node_count"].median())
            if not target_frame.empty
            else None
        ),
        "max_pair_free_node_share": (
            float(target_frame["pair_free_node_share"].max())
            if not target_frame.empty
            else None
        ),
        "min_pair_fixed_outside_node_share": (
            float(target_frame["pair_fixed_outside_node_share"].min())
            if not target_frame.empty
            else None
        ),
        "role_local_contract_status_counts": (
            object_frame["role_local_contract_status"].value_counts(dropna=False).to_dict()
            if not object_frame.empty
            else {}
        ),
        "role_local_contrast_status_counts": (
            contrast_frame["role_local_contrast_status"].value_counts(dropna=False).to_dict()
            if not contrast_frame.empty
            else {}
        ),
        "gate_status_counts": gate_frame["status"].value_counts(dropna=False).to_dict(),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    config = {
        "schema": "nanoclustering_role_local_boundary_plan.v1",
        "readiness_dir": str(readiness_dir),
        "output_dir": str(output_dir),
        "analysis_tier": analysis_tier,
        "method_seeds": list(method_seeds),
        "route_arms": list(ROUTE_ARMS),
        "source_mask_policy": "union_of_dominant_host_context_member_handles",
        "target_mask_policy": "single_top1_endpoint_target_handle_for_execution_rows",
        "role_union_policy": "source_dominant_union_plus_target_family_union",
        "fixed_nodes_policy": "fix_all_nodes_outside_pair_mask",
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(output_dir=output_dir, summary=summary)
    return summary


def _write_report(*, output_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# NanoClustering Role-Local Boundary Plan",
        "",
        f"- status: `{summary['status']}`",
        f"- role_count: {summary['role_count']}",
        f"- case_contrast_count: {summary['case_contrast_count']}",
        f"- target_handle_count: {summary['target_handle_count']}",
        f"- route_plan_row_count: {summary['route_plan_row_count']}",
        f"- median_pair_free_node_count: {summary['median_pair_free_node_count']}",
        f"- max_pair_free_node_share: {summary['max_pair_free_node_share']}",
        f"- min_pair_fixed_outside_node_share: {summary['min_pair_fixed_outside_node_share']}",
        f"- role_local_contrast_status_counts: `{summary['role_local_contrast_status_counts']}`",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Interpretation",
        "",
        (
            "This artifact replaces full-partition warm-start replay with explicit "
            "source/target masks. The next execution runner should use the "
            "per-target pair mask as free nodes and fix all other nodes to the "
            "source seed0 partition, then compare source-state control arms against "
            "target-handle seeded arms."
        ),
        "",
    ]
    (output_dir / REPORT_MD).write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness-dir", type=Path, default=DEFAULT_READINESS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--analysis-tier", default="strict_core_v0_primary")
    parser.add_argument("--method-seeds", default="0,1,2,3,4,5,6,7,8,9")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = _materialize(
        readiness_dir=Path(args.readiness_dir),
        output_dir=Path(args.output_dir),
        analysis_tier=str(args.analysis_tier),
        method_seeds=_parse_seed_list(args.method_seeds),
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
