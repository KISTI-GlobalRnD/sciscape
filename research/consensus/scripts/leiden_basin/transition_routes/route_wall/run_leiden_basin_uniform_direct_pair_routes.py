#!/usr/bin/env python3
"""Run uniform direct pair-route probes for the Track C wall subset.

This runner is diagnostic-only. It reconstructs the endpoint memberships named
by the uniform wall-probe subset and emits W1-W6 artifacts for the same direct
route plus the route-schedule claim gate. The route is not a search policy and
the objective trace is not used to define or rank basins.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").exists()
)
SCRIPT_ROOT = REPO_ROOT / "research/consensus/scripts"
sys.path.insert(0, str(SCRIPT_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from collect_leiden_vanilla_reachability_sweep import (  # noqa: E402
    _load_graph,
    compatible_sketch_nodes,
    encode_u32_sequence,
    hash_u32_sequence,
)
from run_leiden_basin_transition_operator_pilot import (  # noqa: E402
    CandidateMembership,
    RecreatedMembership,
    _recreate_candidate,
    _run_leiden,
    changed_support_nodes,
    endpoint_distance,
    support_distance,
)


BASE_RESULT_DIR = REPO_ROOT / "research/consensus/results/adaptive_refinement"
DEFAULT_SUBSET_DIR = BASE_RESULT_DIR / "leiden_basin_uniform_wall_probe_subset_20260528"
DEFAULT_OUTPUT_DIR = BASE_RESULT_DIR / "leiden_basin_uniform_wall_probe_runner_20260528"
DEFAULT_ENDPOINT_CACHE_DIR = BASE_RESULT_DIR / "leiden_basin_uniform_wall_probe_endpoint_cache_20260528"

EXECUTION_MANIFEST_CSV = "uniform_wall_probe_execution_manifest.csv"
SUBSET_CSV = "uniform_wall_probe_subset.csv"

DIRECT_ROUTE_CSV = "uniform_direct_pair_route_rows.csv"
OBJECTIVE_WALL_CSV = "uniform_objective_wall_rows.csv"
SUPPORT_MOVEMENT_CSV = "uniform_support_movement_rows.csv"
POLISH_REVERSION_CSV = "uniform_polish_reversion_rows.csv"
ROUTE_LABEL_CSV = "uniform_route_label_rows.csv"
ROUTE_CLAIM_CSV = "uniform_route_schedule_claim_rows.csv"
SUMMARY_JSON = "uniform_wall_probe_runner_summary.json"
REPORT_MD = "uniform_wall_probe_runner_report.md"
CONFIG_JSON = "uniform_wall_probe_runner_config.json"
PROGRESS_JSONL = "uniform_wall_probe_runner_progress.jsonl"
CACHE_ROWS_CSV = "uniform_endpoint_cache_rows.csv"

ENDPOINT_TAU = 0.02
SAME_SUPPORT_MAX = 0.5


@dataclass(frozen=True)
class EndpointContext:
    row: pd.Series
    candidate: CandidateMembership
    support_nodes: np.ndarray


@dataclass(frozen=True)
class PairContext:
    manifest_row: pd.Series
    subset_row: pd.Series | None
    candidate_rows: pd.DataFrame
    vanilla_row: pd.Series
    graph: Any
    node_weights: np.ndarray
    arrays: Any
    baseline: RecreatedMembership
    left: EndpointContext
    right: EndpointContext
    sketch_nodes: np.ndarray
    sketch_context: dict[str, Any]


@dataclass
class RunnerStats:
    baseline_cache_hits: int = 0
    baseline_cache_misses: int = 0
    endpoint_cache_hits: int = 0
    endpoint_cache_misses: int = 0


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _resolve(path_text: Any) -> Path:
    path = Path(str(path_text))
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fieldnames})


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _emit_progress(path: Path | None, event: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"timestamp_utc": _now_utc(), **event}
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def _safe_int(value: Any, default: int | None = None) -> int | None:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        if pd.isna(value):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = math.nan) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _case_tail(case_id: str) -> str:
    text = str(case_id)
    for suffix in ("_budget12", "_budget15"):
        if text.endswith(suffix):
            return text[: -len(suffix)]
    return text


def _case_rows(frame: pd.DataFrame, case_id: str) -> pd.DataFrame:
    if frame.empty or "case" not in frame:
        return pd.DataFrame()
    tail = _case_tail(case_id)
    return frame[frame["case"].astype(str).str.endswith(tail)].copy()


def _load_candidate_rows_for_manifest(row: pd.Series) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    seen_paths: set[Path] = set()
    for column in ("left_endpoint_source_artifact", "right_endpoint_source_artifact"):
        path_text = str(row.get(column, "")).strip()
        if not path_text:
            continue
        path = _resolve(path_text).resolve()
        if path in seen_paths:
            continue
        seen_paths.add(path)
        frame = _read_csv(path)
        if frame.empty:
            continue
        frame = frame.copy()
        frame["candidate_source_artifact"] = _rel(path)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    if "candidate_index" in out.columns:
        out["candidate_index"] = pd.to_numeric(out["candidate_index"], errors="coerce")
    return out.drop_duplicates(
        subset=["candidate_source_artifact", "case", "candidate_index"],
        keep="first",
    )


def _find_candidate_row(candidates: pd.DataFrame, case_id: str, candidate_index: int) -> pd.Series:
    case_rows = _case_rows(candidates, case_id)
    rows = case_rows[
        pd.to_numeric(case_rows.get("candidate_index"), errors="coerce")
        == int(candidate_index)
    ]
    if rows.empty:
        raise ValueError(f"missing candidate row for {case_id} index={candidate_index}")
    return rows.iloc[0]


def _select_vanilla_row(vanilla_dir: Path, case_id: str) -> pd.Series:
    rows = _case_rows(_read_csv(vanilla_dir / "vanilla_basin_rows.csv"), case_id)
    if rows.empty:
        raise ValueError(f"missing vanilla context for {case_id} in {vanilla_dir}")
    rows = rows.copy()
    rows["_seed_pref"] = (
        pd.to_numeric(rows.get("seed"), errors="coerce").fillna(-1).astype(int).eq(11)
    )
    rows["_n10_pref"] = rows.get("requested_n_iterations", "").astype(str).eq("10")
    rows["_randomness_abs"] = pd.to_numeric(
        rows.get("randomness"),
        errors="coerce",
    ).fillna(math.inf).abs()
    rows = rows.sort_values(
        ["_seed_pref", "_n10_pref", "_randomness_abs", "graph_dir"],
        ascending=[False, False, True, True],
    )
    return rows.iloc[0]


def _membership_hash(membership: np.ndarray) -> str:
    arr = np.asarray(membership, dtype=np.uint64)
    hasher = hashlib.blake2b(digest_size=16)
    hasher.update(np.asarray([arr.shape[0]], dtype=np.uint64).tobytes())
    hasher.update(arr.tobytes())
    return hasher.hexdigest()


def _cache_key(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.blake2b(encoded, digest_size=16).hexdigest()


def _cache_paths(cache_dir: Path, key: str) -> tuple[Path, Path]:
    return cache_dir / f"{key}.membership.npy", cache_dir / f"{key}.metadata.json"


def _save_membership_cache(
    *,
    cache_dir: Path,
    key: str,
    membership: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    membership_path, metadata_path = _cache_paths(cache_dir, key)
    np.save(membership_path, np.asarray(membership, dtype=np.uint64))
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_membership_cache(
    *,
    cache_dir: Path,
    key: str,
) -> tuple[np.ndarray, dict[str, Any]] | None:
    membership_path, metadata_path = _cache_paths(cache_dir, key)
    if not membership_path.exists() or not metadata_path.exists():
        return None
    membership = np.asarray(np.load(membership_path), dtype=np.uint64)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return membership, metadata


def _closed_route_scope(
    left_membership: np.ndarray,
    right_membership: np.ndarray,
    left_support: np.ndarray,
    right_support: np.ndarray,
) -> np.ndarray:
    left = np.asarray(left_membership, dtype=np.uint64)
    right = np.asarray(right_membership, dtype=np.uint64)
    left_labels = set(int(value) for value in left[np.asarray(left_support, dtype=np.int64)])
    right_labels = set(int(value) for value in right[np.asarray(right_support, dtype=np.int64)])
    if not left_labels and not right_labels:
        return np.asarray([], dtype=np.uint32)
    previous: tuple[int, int] | None = None
    nodes = np.asarray([], dtype=np.uint32)
    while previous != (len(left_labels), len(right_labels)):
        previous = (len(left_labels), len(right_labels))
        left_active = np.isin(left, np.fromiter(left_labels, dtype=np.uint64))
        right_active = np.isin(right, np.fromiter(right_labels, dtype=np.uint64))
        nodes = np.flatnonzero(left_active | right_active).astype(np.uint32)
        if nodes.size == 0:
            break
        left_labels.update(int(value) for value in left[nodes])
        right_labels.update(int(value) for value in right[nodes])
    return nodes


def _target_groups(
    right_membership: np.ndarray,
    route_scope_nodes: np.ndarray,
    route_schedule: str,
) -> list[tuple[int, np.ndarray]]:
    right = np.asarray(right_membership, dtype=np.uint64)
    scope = np.asarray(route_scope_nodes, dtype=np.int64)
    grouped: dict[int, list[int]] = {}
    for node in scope:
        label = int(right[int(node)])
        grouped.setdefault(label, []).append(int(node))
    if route_schedule == "target_size_desc":
        ordered = sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0]))
    elif route_schedule == "target_size_asc":
        ordered = sorted(grouped.items(), key=lambda item: (len(item[1]), item[0]))
    elif route_schedule == "target_label_asc":
        ordered = sorted(grouped.items(), key=lambda item: item[0])
    elif route_schedule == "target_label_desc":
        ordered = sorted(grouped.items(), key=lambda item: -item[0])
    else:
        raise ValueError(f"unsupported route_schedule={route_schedule!r}")
    return [(label, np.asarray(nodes, dtype=np.uint32)) for label, nodes in ordered]


def _state_rows(
    *,
    pair: PairContext,
    route_id: str,
    route_scope_nodes: np.ndarray,
    groups_per_step: int,
    max_route_steps: int,
    route_schedule: str,
    resolution: float,
    progress_path: Path | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], np.ndarray, str]:
    manifest = pair.manifest_row
    panel_pair_id = str(manifest["panel_pair_id"])
    left_membership = pair.left.candidate.recreated.membership
    right_membership = pair.right.candidate.recreated.membership
    current = np.asarray(left_membership, dtype=np.uint64).copy()
    groups = _target_groups(right_membership, route_scope_nodes, route_schedule)
    max_groups = max(0, int(max_route_steps) * int(groups_per_step))
    selected_groups = groups[:max_groups] if max_groups else []
    completion = (
        "complete_target_scope"
        if len(selected_groups) == len(groups)
        else "bounded_partial_target_scope"
    )
    fresh_start = int(current.max(initial=0)) + 1
    label_to_fresh = {
        label: fresh_start + idx
        for idx, (label, _nodes) in enumerate(groups)
    }

    direct_rows: list[dict[str, Any]] = []
    objective_rows: list[dict[str, Any]] = []
    movement_rows: list[dict[str, Any]] = []
    objective_values: list[float] = []

    left_support = pair.left.support_nodes
    right_support = pair.right.support_nodes

    def append_state(
        *,
        step_index: int,
        edited_nodes: np.ndarray,
        edited_labels: list[int],
        step_status: str,
    ) -> None:
        support = changed_support_nodes(pair.baseline.membership, current)
        source_distance, source_intersection, source_union = support_distance(support, left_support)
        target_distance, target_intersection, target_union = support_distance(support, right_support)
        objective_value = float(pair.graph.cpm_quality(current, resolution=float(resolution)))
        objective_values.append(objective_value)
        start_objective = objective_values[0]
        min_objective = min(objective_values)
        wall_step = bool(objective_value < start_objective and objective_value <= min_objective)
        direct_rows.append(
            {
                "panel_pair_id": panel_pair_id,
                "route_id": route_id,
                "step_index": int(step_index),
                "source_endpoint_identity_id": str(manifest["left_endpoint_identity_id"]),
                "target_endpoint_identity_id": str(manifest["right_endpoint_identity_id"]),
                "state_membership_hash": _membership_hash(current),
                "edited_node_count": int(edited_nodes.size),
                "edited_node_ids": encode_u32_sequence(edited_nodes),
                "edited_target_labels": ";".join(str(label) for label in edited_labels),
                "route_scope_node_count": int(route_scope_nodes.size),
                "target_group_count": int(len(groups)),
                "route_schedule": route_schedule,
                "route_completion_status": completion,
                "step_status": step_status,
            }
        )
        objective_rows.append(
            {
                "panel_pair_id": panel_pair_id,
                "route_id": route_id,
                "step_index": int(step_index),
                "objective_value": objective_value,
                "objective_debt_from_start": float(start_objective - objective_value),
                "objective_recovery_from_min": float(objective_value - min_objective),
                "objective_min_so_far": float(min_objective),
                "route_schedule": route_schedule,
                "wall_step_flag": wall_step,
            }
        )
        movement_rows.append(
            {
                "panel_pair_id": panel_pair_id,
                "route_id": route_id,
                "step_index": int(step_index),
                "support_node_count": int(support.size),
                "support_distance_to_source": source_distance,
                "support_intersection_with_source": source_intersection,
                "support_union_with_source": source_union,
                "support_distance_to_target": target_distance,
                "support_intersection_with_target": target_intersection,
                "support_union_with_target": target_union,
                "endpoint_distance_to_source": endpoint_distance(
                    current,
                    left_membership,
                    pair.sketch_nodes,
                ),
                "endpoint_distance_to_target": endpoint_distance(
                    current,
                    right_membership,
                    pair.sketch_nodes,
                ),
                "route_schedule": route_schedule,
                "sketch_node_hash": hash_u32_sequence(pair.sketch_nodes),
            }
        )
        _emit_progress(
            progress_path,
            {
                "event": "route_step",
                "panel_pair_id": panel_pair_id,
                "route_id": route_id,
                "route_schedule": route_schedule,
                "step_index": int(step_index),
                "edited_node_count": int(edited_nodes.size),
                "support_node_count": int(support.size),
                "support_distance_to_source": source_distance,
                "support_distance_to_target": target_distance,
                "endpoint_distance_to_source": endpoint_distance(
                    current,
                    left_membership,
                    pair.sketch_nodes,
                ),
                "endpoint_distance_to_target": endpoint_distance(
                    current,
                    right_membership,
                    pair.sketch_nodes,
                ),
                "objective_value": objective_value,
                "route_completion_status": completion,
            },
        )

    append_state(
        step_index=0,
        edited_nodes=np.asarray([], dtype=np.uint32),
        edited_labels=[],
        step_status="source_endpoint",
    )
    for chunk_start in range(0, len(selected_groups), int(groups_per_step)):
        chunk = selected_groups[chunk_start : chunk_start + int(groups_per_step)]
        edited_nodes = (
            np.concatenate([nodes for _label, nodes in chunk]).astype(np.uint32)
            if chunk
            else np.asarray([], dtype=np.uint32)
        )
        edited_labels = [label for label, _nodes in chunk]
        for label, nodes in chunk:
            current[np.asarray(nodes, dtype=np.int64)] = np.uint64(label_to_fresh[label])
        append_state(
            step_index=1 + chunk_start // int(groups_per_step),
            edited_nodes=np.unique(edited_nodes),
            edited_labels=edited_labels,
            step_status="target_scope_edit",
        )
    return direct_rows, objective_rows, movement_rows, current, completion


def _assign_endpoint(
    *,
    membership: np.ndarray,
    baseline: np.ndarray,
    left_membership: np.ndarray,
    right_membership: np.ndarray,
    left_support: np.ndarray,
    right_support: np.ndarray,
    sketch_nodes: np.ndarray,
) -> tuple[str, dict[str, Any]]:
    support = changed_support_nodes(baseline, membership)
    source_support_distance, _source_i, _source_u = support_distance(support, left_support)
    target_support_distance, _target_i, _target_u = support_distance(support, right_support)
    source_endpoint_distance = endpoint_distance(membership, left_membership, sketch_nodes)
    target_endpoint_distance = endpoint_distance(membership, right_membership, sketch_nodes)
    assignment = "other_or_ambiguous"
    if (
        math.isfinite(target_endpoint_distance)
        and target_endpoint_distance <= ENDPOINT_TAU
        and target_support_distance <= SAME_SUPPORT_MAX
    ):
        assignment = "target_endpoint"
    elif (
        math.isfinite(source_endpoint_distance)
        and source_endpoint_distance <= ENDPOINT_TAU
        and source_support_distance <= SAME_SUPPORT_MAX
    ):
        assignment = "source_endpoint"
    return assignment, {
        "post_polish_support_distance_to_source": source_support_distance,
        "post_polish_support_distance_to_target": target_support_distance,
        "post_polish_endpoint_distance_to_source": source_endpoint_distance,
        "post_polish_endpoint_distance_to_target": target_endpoint_distance,
        "post_polish_support_node_count": int(support.size),
    }


def _polish_row(
    *,
    pair: PairContext,
    route_id: str,
    pre_polish: np.ndarray,
    route_completion_status: str,
    route_schedule: str,
    polish_iterations: int,
    resolution: float,
    randomness: float,
    perturb_seed_offset: int,
    progress_path: Path | None,
) -> dict[str, Any]:
    manifest = pair.manifest_row
    panel_pair_id = str(manifest["panel_pair_id"])
    seed = int(pair.left.row.get("seed", 0)) + int(perturb_seed_offset) + 100_000
    post = _run_leiden(
        pair.graph,
        resolution=resolution,
        seed=seed,
        n_iterations=polish_iterations,
        randomness=randomness,
        initial_membership=np.asarray(pre_polish, dtype=np.uint64),
    )
    assignment, metrics = _assign_endpoint(
        membership=post.membership,
        baseline=pair.baseline.membership,
        left_membership=pair.left.candidate.recreated.membership,
        right_membership=pair.right.candidate.recreated.membership,
        left_support=pair.left.support_nodes,
        right_support=pair.right.support_nodes,
        sketch_nodes=pair.sketch_nodes,
    )
    if assignment == "source_endpoint":
        reversion_status = "reverted_to_source"
    elif assignment == "target_endpoint":
        reversion_status = "stays_at_target"
    else:
        reversion_status = "unassigned_after_polish"
    _emit_progress(
        progress_path,
        {
            "event": "polish_done",
            "panel_pair_id": panel_pair_id,
            "route_id": route_id,
            "route_schedule": route_schedule,
            "post_polish_endpoint_assignment": assignment,
            "reversion_status": reversion_status,
            **metrics,
        },
    )
    return {
        "panel_pair_id": panel_pair_id,
        "route_id": route_id,
        "pre_polish_state_id": _membership_hash(pre_polish),
        "post_polish_state_id": _membership_hash(post.membership),
        "post_polish_endpoint_assignment": assignment,
        "reversion_status": reversion_status,
        "polish_iterations": int(polish_iterations),
        "route_schedule": route_schedule,
        "route_completion_status": route_completion_status,
        **metrics,
    }


def _route_label_row(
    *,
    pair: PairContext,
    route_id: str,
    route_schedule: str,
    route_rows: list[dict[str, Any]],
    movement_rows: list[dict[str, Any]],
    objective_rows: list[dict[str, Any]],
    polish_row: dict[str, Any],
) -> dict[str, Any]:
    manifest = pair.manifest_row
    panel_pair_id = str(manifest["panel_pair_id"])
    subset_row = pair.subset_row
    subset_role = str(manifest.get("subset_role", ""))
    panel_role = ""
    calibrated_relation = ""
    if subset_row is not None:
        subset_role = str(subset_row.get("subset_role", subset_role))
        panel_role = str(subset_row.get("panel_role", ""))
        calibrated_relation = str(subset_row.get("calibrated_relation", ""))
    completion = str(route_rows[-1]["route_completion_status"]) if route_rows else "missing_route"
    last_movement = movement_rows[-1] if movement_rows else {}
    target_endpoint_distance = _safe_float(last_movement.get("endpoint_distance_to_target"))
    target_support_distance = _safe_float(last_movement.get("support_distance_to_target"))
    target_reached = (
        completion == "complete_target_scope"
        and math.isfinite(target_endpoint_distance)
        and target_endpoint_distance <= ENDPOINT_TAU
        and target_support_distance <= SAME_SUPPORT_MAX
    )
    post_assignment = str(polish_row.get("post_polish_endpoint_assignment", ""))
    if subset_role == "same_zone_control":
        route_label = "same_zone_control_trace"
        confidence = "partial"
        wall_status = "control_no_wall_claim"
        notes = "same-control pair traced to check that the protocol does not manufacture a wall"
    elif target_reached and post_assignment == "target_endpoint":
        route_label = "direct_route_reaches_target_and_polish_stays"
        confidence = "partial"
        wall_status = "wall_evidence_partial_objective_trace_present"
        notes = "W1-W4 are present for a complete direct scope route; replicate controls still required"
    elif target_reached and post_assignment == "source_endpoint":
        route_label = "direct_route_reaches_target_then_polish_reverts"
        confidence = "partial"
        wall_status = "candidate_wall_like_reversion"
        notes = "direct route reaches target by support-local criteria but free polish returns to source"
    elif completion != "complete_target_scope":
        route_label = "bounded_route_incomplete"
        confidence = "unknown"
        wall_status = "no_wall_claim"
        notes = "route was bounded before all target-scope groups were edited"
    else:
        route_label = "direct_route_unassigned"
        confidence = "unknown"
        wall_status = "no_wall_claim"
        notes = "route did not satisfy calibrated target assignment after W1-W4"
    return {
        "panel_pair_id": panel_pair_id,
        "route_id": route_id,
        "route_schedule": route_schedule,
        "subset_role": subset_role,
        "panel_role": panel_role,
        "calibrated_relation": calibrated_relation,
        "route_label": route_label,
        "route_label_confidence": confidence,
        "wall_assignment_status": wall_status,
        "support_assignment_status": post_assignment,
        "objective_row_count": int(len(objective_rows)),
        "direct_route_row_count": int(len(route_rows)),
        "target_endpoint_distance_final": target_endpoint_distance,
        "target_support_distance_final": target_support_distance,
        "evidence_notes": notes,
    }


def _unique_texts(rows: list[dict[str, Any]], column: str) -> list[str]:
    values = set()
    for row in rows:
        value = str(row.get(column, "")).strip()
        if value:
            values.add(value)
    return sorted(values)


def _numeric_values(rows: list[dict[str, Any]], column: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = _safe_float(row.get(column))
        if math.isfinite(value):
            values.append(value)
    return values


def _wall_step_count_by_schedule(
    objective_rows: list[dict[str, Any]],
) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    for row in objective_rows:
        pair_id = str(row.get("panel_pair_id", "")).strip()
        schedule = str(row.get("route_schedule", "")).strip()
        if not pair_id or not schedule:
            continue
        key = (pair_id, schedule)
        counts.setdefault(key, 0)
        if bool(row.get("wall_step_flag")):
            counts[key] += 1
    return counts


def _gate_status(
    *,
    schedule_replicated: bool,
    route_order_stable: bool,
    wall_statuses: list[str],
    calibrated_relations: list[str],
) -> tuple[str, str]:
    if not schedule_replicated:
        return (
            "not_replicated",
            "schedule_replicate_required_before_wall_claim",
        )
    if not route_order_stable:
        return (
            "route_order_sensitive",
            "fails_schedule_invariance_no_supported_wall_claim",
        )
    relation = calibrated_relations[0] if len(calibrated_relations) == 1 else ""
    if (
        wall_statuses == ["wall_evidence_partial_objective_trace_present"]
        and relation == "distinct_support_local"
    ):
        return (
            "route_order_stable",
            "passes_schedule_invariance_distinct_partial_wall_evidence",
        )
    if (
        wall_statuses == ["wall_evidence_partial_objective_trace_present"]
        and relation == "ambiguous_support_local"
    ):
        return (
            "route_order_stable",
            "stable_route_evidence_basin_relation_ambiguous_no_supported_wall_claim",
        )
    if wall_statuses == ["candidate_wall_like_reversion"]:
        return (
            "route_order_stable",
            "stable_reversion_evidence_relation_unresolved_no_supported_wall_claim",
        )
    if wall_statuses == ["control_no_wall_claim"]:
        return "route_order_stable", "stable_control_no_wall_claim"
    if wall_statuses == ["no_wall_claim"]:
        return "route_order_stable", "stable_no_wall_claim"
    return "route_order_stable", "stable_non_wall_diagnostic"


def _gate_notes(status: str) -> str:
    if status == "schedule_replicate_required_before_wall_claim":
        return "only one route schedule observed; replicate schedules are required before wall claims"
    if status == "fails_schedule_invariance_no_supported_wall_claim":
        return "route labels or wall assignments differ across schedules; keep as route-order-sensitive diagnostic evidence"
    if status == "passes_schedule_invariance_distinct_partial_wall_evidence":
        return "route label and wall assignment are stable across schedules for a distinct-support pair; still diagnostic until broader controls pass"
    if status == "stable_route_evidence_basin_relation_ambiguous_no_supported_wall_claim":
        return "route label is stable, but the basin relation is ambiguous; do not promote to supported wall evidence"
    if status == "stable_reversion_evidence_relation_unresolved_no_supported_wall_claim":
        return "polish-reversion evidence is stable, but relation or controls are unresolved; do not promote to supported wall evidence"
    if status == "stable_control_no_wall_claim":
        return "same-zone control did not produce a wall claim across schedules"
    if status == "stable_no_wall_claim":
        return "all observed route schedules agree on no wall claim"
    return "stable schedule-level diagnostic row; review before promotion"


def _build_route_schedule_claim_rows(
    label_rows: list[dict[str, Any]],
    objective_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_pair: dict[str, list[dict[str, Any]]] = {}
    for row in label_rows:
        pair_id = str(row.get("panel_pair_id", "")).strip()
        if not pair_id:
            continue
        by_pair.setdefault(pair_id, []).append(row)

    wall_counts = _wall_step_count_by_schedule(objective_rows)
    claim_rows: list[dict[str, Any]] = []
    for pair_id in sorted(by_pair):
        rows = by_pair[pair_id]
        schedules = _unique_texts(rows, "route_schedule")
        subset_roles = _unique_texts(rows, "subset_role")
        panel_roles = _unique_texts(rows, "panel_role")
        calibrated_relations = _unique_texts(rows, "calibrated_relation")
        route_labels = _unique_texts(rows, "route_label")
        wall_statuses = _unique_texts(rows, "wall_assignment_status")
        support_statuses = _unique_texts(rows, "support_assignment_status")
        stable_route_label = len(route_labels) == 1
        stable_wall_assignment = len(wall_statuses) == 1
        stable_support_assignment = len(support_statuses) == 1
        schedule_replicated = len(schedules) > 1
        route_order_stable = (
            schedule_replicated
            and stable_route_label
            and stable_wall_assignment
            and stable_support_assignment
        )
        sensitivity_status, gate_status = _gate_status(
            schedule_replicated=schedule_replicated,
            route_order_stable=route_order_stable,
            wall_statuses=wall_statuses,
            calibrated_relations=calibrated_relations,
        )
        counts = [wall_counts.get((pair_id, schedule), 0) for schedule in schedules]
        counts_by_schedule = [f"{schedule}:{wall_counts.get((pair_id, schedule), 0)}" for schedule in schedules]
        direct_counts = _numeric_values(rows, "direct_route_row_count")
        endpoint_distances = _numeric_values(rows, "target_endpoint_distance_final")
        support_distances = _numeric_values(rows, "target_support_distance_final")
        claim_rows.append(
            {
                "panel_pair_id": pair_id,
                "schedule_count": int(len(schedules)),
                "route_schedules": "|".join(schedules),
                "subset_roles": "|".join(subset_roles),
                "panel_roles": "|".join(panel_roles),
                "calibrated_relations": "|".join(calibrated_relations),
                "schedule_replicated": schedule_replicated,
                "stable_route_label": stable_route_label,
                "stable_wall_assignment": stable_wall_assignment,
                "stable_support_assignment": stable_support_assignment,
                "route_order_sensitivity_status": sensitivity_status,
                "wall_claim_gate_status": gate_status,
                "route_labels": "|".join(route_labels),
                "wall_assignment_statuses": "|".join(wall_statuses),
                "support_assignment_statuses": "|".join(support_statuses),
                "direct_route_row_count_min": int(min(direct_counts)) if direct_counts else "",
                "direct_route_row_count_max": int(max(direct_counts)) if direct_counts else "",
                "target_endpoint_distance_final_max": max(endpoint_distances) if endpoint_distances else "",
                "target_support_distance_final_max": max(support_distances) if support_distances else "",
                "objective_wall_step_count_min": int(min(counts)) if counts else "",
                "objective_wall_step_count_max": int(max(counts)) if counts else "",
                "objective_wall_steps_by_schedule": "|".join(counts_by_schedule),
                "evidence_notes": _gate_notes(gate_status),
            }
        )
    return claim_rows


def _build_pair_context(
    *,
    manifest_row: pd.Series,
    subset_by_pair: dict[str, pd.Series],
    graph_cache: dict[str, tuple[Any, np.ndarray, Any]],
    baseline_cache: dict[str, RecreatedMembership],
    candidate_cache: dict[tuple[str, int], CandidateMembership],
    endpoint_cache_dir: Path,
    reuse_endpoint_cache: bool,
    cache_rows: list[dict[str, Any]],
    stats: RunnerStats,
    progress_path: Path | None,
    baseline_iterations: int,
    polish_iterations: int,
    resolution: float,
    randomness: float,
    perturb_seed_offset: int,
) -> PairContext:
    case_id = str(manifest_row["case_id"])
    panel_pair_id = str(manifest_row["panel_pair_id"])
    candidates = _load_candidate_rows_for_manifest(manifest_row)
    if candidates.empty:
        raise ValueError(f"missing candidate rows for {panel_pair_id}")
    left_index = int(manifest_row["left_representative_candidate_index"])
    right_index = int(manifest_row["right_representative_candidate_index"])
    left_row = _find_candidate_row(candidates, case_id, left_index)
    right_row = _find_candidate_row(candidates, case_id, right_index)
    vanilla_dir = _resolve(manifest_row["vanilla_context_dir"])
    vanilla_row = _select_vanilla_row(vanilla_dir, case_id)
    graph_dir = Path(str(vanilla_row["graph_dir"]))
    graph_key = str(graph_dir)
    if graph_key not in graph_cache:
        graph_cache[graph_key] = _load_graph(graph_dir)
    graph, node_weights, arrays = graph_cache[graph_key]
    case_key = f"{graph_key}|{case_id}|seed={int(left_row.get('seed', 0))}"
    baseline_payload = {
        "kind": "baseline",
        "graph_dir": graph_key,
        "case_id": case_id,
        "seed": int(left_row.get("seed", 0)),
        "baseline_iterations": int(baseline_iterations),
        "resolution": float(resolution),
        "randomness": float(randomness),
    }
    baseline_disk_key = _cache_key(baseline_payload)
    if case_key not in baseline_cache:
        cached = (
            _load_membership_cache(cache_dir=endpoint_cache_dir, key=baseline_disk_key)
            if reuse_endpoint_cache
            else None
        )
        if cached is not None:
            membership, metadata = cached
            quality = _safe_float(metadata.get("objective_value"))
            if not math.isfinite(quality):
                quality = float(graph.cpm_quality(membership, resolution=float(resolution)))
            baseline_cache[case_key] = RecreatedMembership(
                membership=membership,
                quality=quality,
                elapsed_sec=0.0,
            )
            stats.baseline_cache_hits += 1
            cache_status = "hit"
        else:
            baseline_cache[case_key] = _run_leiden(
                graph,
                resolution=resolution,
                seed=int(left_row.get("seed", 0)),
                n_iterations=baseline_iterations,
                randomness=randomness,
            )
            _save_membership_cache(
                cache_dir=endpoint_cache_dir,
                key=baseline_disk_key,
                membership=baseline_cache[case_key].membership,
                metadata={
                    **baseline_payload,
                    "cache_key": baseline_disk_key,
                    "objective_value": baseline_cache[case_key].quality,
                    "membership_hash": _membership_hash(baseline_cache[case_key].membership),
                    "created_at_utc": _now_utc(),
                },
            )
            stats.baseline_cache_misses += 1
            cache_status = "miss"
        cache_rows.append(
            {
                "cache_kind": "baseline",
                "cache_status": cache_status,
                "cache_key": baseline_disk_key,
                "case_id": case_id,
                "candidate_index": "",
                "membership_path": _rel(_cache_paths(endpoint_cache_dir, baseline_disk_key)[0]),
                "metadata_path": _rel(_cache_paths(endpoint_cache_dir, baseline_disk_key)[1]),
            }
        )
        _emit_progress(
            progress_path,
            {
                "event": "baseline_ready",
                "panel_pair_id": panel_pair_id,
                "case_id": case_id,
                "cache_status": cache_status,
                "cache_key": baseline_disk_key,
            },
        )
    baseline = baseline_cache[case_key]

    def endpoint_context(candidate_index: int, row: pd.Series) -> EndpointContext:
        key = (case_key, candidate_index)
        if key not in candidate_cache:
            endpoint_payload = {
                "kind": "endpoint",
                "graph_dir": graph_key,
                "case_id": case_id,
                "candidate_index": int(candidate_index),
                "seed": int(row.get("seed", 0)),
                "source_cluster": int(row["source_cluster"]),
                "target_cluster": int(row["target_cluster"]),
                "baseline_cache_key": baseline_disk_key,
                "polish_iterations": int(polish_iterations),
                "resolution": float(resolution),
                "randomness": float(randomness),
                "perturb_seed_offset": int(perturb_seed_offset),
            }
            endpoint_disk_key = _cache_key(endpoint_payload)
            cached_endpoint = (
                _load_membership_cache(cache_dir=endpoint_cache_dir, key=endpoint_disk_key)
                if reuse_endpoint_cache
                else None
            )
            if cached_endpoint is not None:
                membership, metadata = cached_endpoint
                objective_value = _safe_float(metadata.get("objective_value"))
                if not math.isfinite(objective_value):
                    objective_value = float(graph.cpm_quality(membership, resolution=float(resolution)))
                support_nodes = changed_support_nodes(baseline.membership, membership)
                candidate_cache[key] = CandidateMembership(
                    recreated=RecreatedMembership(
                        membership=membership,
                        quality=objective_value,
                        elapsed_sec=0.0,
                    ),
                    row=row,
                    group_nodes=np.asarray([], dtype=np.uint32),
                    support_nodes=support_nodes,
                )
                stats.endpoint_cache_hits += 1
                cache_status = "hit"
            else:
                candidate_cache[key] = _recreate_candidate(
                    graph=graph,
                    arrays=arrays,
                    node_weights=node_weights,
                    baseline_membership=baseline.membership,
                    baseline_quality=baseline.quality,
                    row=row,
                    resolution=resolution,
                    randomness=randomness,
                    perturb_seed_offset=perturb_seed_offset,
                    polish_iterations=polish_iterations,
                )
                _save_membership_cache(
                    cache_dir=endpoint_cache_dir,
                    key=endpoint_disk_key,
                    membership=candidate_cache[key].recreated.membership,
                    metadata={
                        **endpoint_payload,
                        "cache_key": endpoint_disk_key,
                        "objective_value": candidate_cache[key].recreated.quality,
                        "membership_hash": _membership_hash(candidate_cache[key].recreated.membership),
                        "support_node_count": int(candidate_cache[key].support_nodes.size),
                        "created_at_utc": _now_utc(),
                    },
                )
                stats.endpoint_cache_misses += 1
                cache_status = "miss"
            cache_rows.append(
                {
                    "cache_kind": "endpoint",
                    "cache_status": cache_status,
                    "cache_key": endpoint_disk_key,
                    "case_id": case_id,
                    "candidate_index": int(candidate_index),
                    "membership_path": _rel(_cache_paths(endpoint_cache_dir, endpoint_disk_key)[0]),
                    "metadata_path": _rel(_cache_paths(endpoint_cache_dir, endpoint_disk_key)[1]),
                }
            )
            _emit_progress(
                progress_path,
                {
                    "event": "endpoint_ready",
                    "panel_pair_id": panel_pair_id,
                    "case_id": case_id,
                    "candidate_index": int(candidate_index),
                    "cache_status": cache_status,
                    "cache_key": endpoint_disk_key,
                },
            )
        candidate = candidate_cache[key]
        return EndpointContext(
            row=row,
            candidate=candidate,
            support_nodes=changed_support_nodes(baseline.membership, candidate.recreated.membership),
        )

    case_candidate_rows = _case_rows(candidates, case_id)
    sketch_nodes, sketch_context = compatible_sketch_nodes(
        arrays=arrays,
        baseline_membership=baseline.membership,
        node_weights=node_weights,
        candidate_rows=case_candidate_rows,
    )
    return PairContext(
        manifest_row=manifest_row,
        subset_row=subset_by_pair.get(panel_pair_id),
        candidate_rows=case_candidate_rows,
        vanilla_row=vanilla_row,
        graph=graph,
        node_weights=node_weights,
        arrays=arrays,
        baseline=baseline,
        left=endpoint_context(left_index, left_row),
        right=endpoint_context(right_index, right_row),
        sketch_nodes=sketch_nodes,
        sketch_context=sketch_context,
    )


def _run_pair(
    *,
    pair: PairContext,
    groups_per_step: int,
    max_route_steps: int,
    route_schedule: str,
    polish_iterations: int,
    resolution: float,
    randomness: float,
    perturb_seed_offset: int,
    progress_path: Path | None,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    manifest = pair.manifest_row
    panel_pair_id = str(manifest["panel_pair_id"])
    route_id = (
        f"{panel_pair_id}|support_closure_direct|"
        f"schedule={route_schedule}|groups_per_step={int(groups_per_step)}|"
        f"max_steps={int(max_route_steps)}"
    )
    route_scope_nodes = _closed_route_scope(
        pair.left.candidate.recreated.membership,
        pair.right.candidate.recreated.membership,
        pair.left.support_nodes,
        pair.right.support_nodes,
    )
    route_rows, objective_rows, movement_rows, final_state, completion = _state_rows(
        pair=pair,
        route_id=route_id,
        route_scope_nodes=route_scope_nodes,
        groups_per_step=groups_per_step,
        max_route_steps=max_route_steps,
        route_schedule=route_schedule,
        resolution=resolution,
        progress_path=progress_path,
    )
    polish = _polish_row(
        pair=pair,
        route_id=route_id,
        pre_polish=final_state,
        route_completion_status=completion,
        route_schedule=route_schedule,
        polish_iterations=polish_iterations,
        resolution=resolution,
        randomness=randomness,
        perturb_seed_offset=perturb_seed_offset,
        progress_path=progress_path,
    )
    label = _route_label_row(
        pair=pair,
        route_id=route_id,
        route_schedule=route_schedule,
        route_rows=route_rows,
        movement_rows=movement_rows,
        objective_rows=objective_rows,
        polish_row=polish,
    )
    return route_rows, objective_rows, movement_rows, polish, label


def _error_label(row: pd.Series, message: str) -> dict[str, Any]:
    panel_pair_id = str(row.get("panel_pair_id", ""))
    return {
        "panel_pair_id": panel_pair_id,
        "route_id": f"{panel_pair_id}|error",
        "route_label": "runner_error",
        "route_label_confidence": "unknown",
        "wall_assignment_status": "no_wall_claim",
        "support_assignment_status": "missing",
        "evidence_notes": message,
    }


def _parse_pair_ids(value: str) -> set[str]:
    if not value.strip():
        return set()
    return {part.strip() for part in value.split(",") if part.strip()}


def _parse_route_schedules(value: str) -> tuple[str, ...]:
    schedules = tuple(part.strip() for part in value.split(",") if part.strip())
    if not schedules:
        raise ValueError("--route-schedules must include at least one schedule")
    allowed = {
        "target_size_desc",
        "target_size_asc",
        "target_label_asc",
        "target_label_desc",
    }
    unsupported = sorted(set(schedules) - allowed)
    if unsupported:
        raise ValueError(f"unsupported route schedules: {unsupported}")
    return schedules


def _write_report(
    path: Path,
    summary: dict[str, Any],
    labels: list[dict[str, Any]],
    claim_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Leiden Basin Uniform Direct Pair-Route Runner",
        "",
        f"Status: {summary['status']}",
        f"Date: {summary['date']}",
        "",
        "This artifact executes the uniform W1-W6 wall-protocol subset. It traces basin-to-basin routes and wall evidence only; it does not define basins by objective value and does not rank basin value.",
        "",
        "## Pair Labels",
        "",
        "| pair_id | schedule | route_label | confidence | wall_status | support_status |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in labels:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("panel_pair_id", "")),
                    str(row.get("route_schedule", "")),
                    str(row.get("route_label", "")),
                    str(row.get("route_label_confidence", "")),
                    str(row.get("wall_assignment_status", "")),
                    str(row.get("support_assignment_status", "")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Route-Schedule Claim Gate",
            "",
            "| pair_id | schedule_count | sensitivity | gate_status | route_labels | wall_statuses |",
            "| --- | ---: | --- | --- | --- | --- |",
        ]
    )
    for row in claim_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("panel_pair_id", "")),
                    str(row.get("schedule_count", "")),
                    str(row.get("route_order_sensitivity_status", "")),
                    str(row.get("wall_claim_gate_status", "")),
                    str(row.get("route_labels", "")),
                    str(row.get("wall_assignment_statuses", "")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Guardrail",
            "",
            "- W1-W6 rows are wall-protocol evidence, not basin-definition evidence.",
            "- A partial route label is not a supported wall claim unless the pair passes predeclared route-schedule invariance and route controls.",
            "- Route-order-sensitive rows remain diagnostic evidence, not wall evidence.",
            "- The same-zone control must remain a control against manufacturing walls.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    *,
    subset_dir: Path,
    output_dir: Path,
    endpoint_cache_dir: Path,
    reuse_endpoint_cache: bool,
    pair_ids: set[str],
    route_schedules: tuple[str, ...],
    groups_per_step: int,
    max_route_steps: int,
    max_pairs: int | None,
    baseline_iterations: int,
    polish_iterations: int,
    resolution: float,
    randomness: float,
    perturb_seed_offset: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_JSONL
    if progress_path.exists():
        progress_path.unlink()
    manifest = _read_csv(subset_dir / EXECUTION_MANIFEST_CSV)
    subset = _read_csv(subset_dir / SUBSET_CSV)
    if manifest.empty:
        raise FileNotFoundError(subset_dir / EXECUTION_MANIFEST_CSV)
    subset_by_pair = {
        str(row["panel_pair_id"]): row
        for _, row in subset.iterrows()
    } if not subset.empty else {}

    graph_cache: dict[str, tuple[Any, np.ndarray, Any]] = {}
    baseline_cache: dict[str, RecreatedMembership] = {}
    candidate_cache: dict[tuple[str, int], CandidateMembership] = {}
    cache_rows: list[dict[str, Any]] = []
    stats = RunnerStats()

    direct_rows: list[dict[str, Any]] = []
    objective_rows: list[dict[str, Any]] = []
    movement_rows: list[dict[str, Any]] = []
    polish_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    ordered_manifest = manifest.sort_values("subset_order")
    if pair_ids:
        ordered_manifest = ordered_manifest[
            ordered_manifest["panel_pair_id"].astype(str).isin(pair_ids)
        ].copy()
    if max_pairs is not None:
        ordered_manifest = ordered_manifest.head(int(max_pairs))
    total_pairs = int(len(ordered_manifest))
    for pair_number, (_, row) in enumerate(ordered_manifest.iterrows(), start=1):
        panel_pair_id = str(row.get("panel_pair_id", ""))
        print(f"[{pair_number}/{total_pairs}] start {panel_pair_id}", flush=True)
        _emit_progress(
            progress_path,
            {
                "event": "pair_start",
                "pair_number": int(pair_number),
                "total_pairs": int(total_pairs),
                "panel_pair_id": panel_pair_id,
            },
        )
        try:
            pair = _build_pair_context(
                manifest_row=row,
                subset_by_pair=subset_by_pair,
                graph_cache=graph_cache,
                baseline_cache=baseline_cache,
                candidate_cache=candidate_cache,
                endpoint_cache_dir=endpoint_cache_dir,
                reuse_endpoint_cache=reuse_endpoint_cache,
                cache_rows=cache_rows,
                stats=stats,
                progress_path=progress_path,
                baseline_iterations=baseline_iterations,
                polish_iterations=polish_iterations,
                resolution=resolution,
                randomness=randomness,
                perturb_seed_offset=perturb_seed_offset,
            )
            pair_labels: list[str] = []
            for route_schedule in route_schedules:
                _emit_progress(
                    progress_path,
                    {
                        "event": "route_schedule_start",
                        "pair_number": int(pair_number),
                        "total_pairs": int(total_pairs),
                        "panel_pair_id": panel_pair_id,
                        "route_schedule": route_schedule,
                    },
                )
                route, objective, movement, polish, label = _run_pair(
                    pair=pair,
                    groups_per_step=groups_per_step,
                    max_route_steps=max_route_steps,
                    route_schedule=route_schedule,
                    polish_iterations=polish_iterations,
                    resolution=resolution,
                    randomness=randomness,
                    perturb_seed_offset=perturb_seed_offset,
                    progress_path=progress_path,
                )
                direct_rows.extend(route)
                objective_rows.extend(objective)
                movement_rows.extend(movement)
                polish_rows.append(polish)
                label_rows.append(label)
                pair_labels.append(
                    f"{route_schedule}:{label.get('route_label')}/"
                    f"{label.get('wall_assignment_status')}"
                )
                _emit_progress(
                    progress_path,
                    {
                        "event": "route_schedule_done",
                        "pair_number": int(pair_number),
                        "total_pairs": int(total_pairs),
                        "panel_pair_id": panel_pair_id,
                        "route_schedule": route_schedule,
                        "route_label": label.get("route_label"),
                        "wall_assignment_status": label.get("wall_assignment_status"),
                    },
                )
            print(
                f"[{pair_number}/{total_pairs}] done {panel_pair_id}: "
                + "; ".join(pair_labels),
                flush=True,
            )
            _emit_progress(
                progress_path,
                {
                    "event": "pair_done",
                    "pair_number": int(pair_number),
                    "total_pairs": int(total_pairs),
                    "panel_pair_id": panel_pair_id,
                    "route_schedule_count": int(len(route_schedules)),
                    "route_schedule_labels": pair_labels,
                },
            )
        except Exception as exc:  # noqa: BLE001 - preserve per-pair progress.
            message = f"{type(exc).__name__}: {exc}"
            errors.append({"panel_pair_id": str(row.get("panel_pair_id", "")), "error": message})
            label_rows.append(_error_label(row, message))
            print(f"[{pair_number}/{total_pairs}] error {panel_pair_id}: {message}", flush=True)
            _emit_progress(
                progress_path,
                {
                    "event": "pair_error",
                    "pair_number": int(pair_number),
                    "total_pairs": int(total_pairs),
                    "panel_pair_id": panel_pair_id,
                    "error": message,
                },
            )

    _write_csv(output_dir / DIRECT_ROUTE_CSV, direct_rows)
    _write_csv(output_dir / OBJECTIVE_WALL_CSV, objective_rows)
    _write_csv(output_dir / SUPPORT_MOVEMENT_CSV, movement_rows)
    _write_csv(output_dir / POLISH_REVERSION_CSV, polish_rows)
    _write_csv(output_dir / ROUTE_LABEL_CSV, label_rows)
    claim_rows = _build_route_schedule_claim_rows(label_rows, objective_rows)
    _write_csv(output_dir / ROUTE_CLAIM_CSV, claim_rows)
    _write_csv(output_dir / CACHE_ROWS_CSV, cache_rows)

    summary = {
        "schema": "leiden_basin_uniform_direct_pair_routes.v1",
        "status": "completed_with_errors" if errors else "completed",
        "date": "2026-05-28",
        "subset_dir": _rel(subset_dir),
        "output_dir": _rel(output_dir),
        "endpoint_cache_dir": _rel(endpoint_cache_dir),
        "pair_count": int(len(ordered_manifest)),
        "available_manifest_pair_count": int(len(manifest)),
        "selected_pair_ids": ordered_manifest["panel_pair_id"].astype(str).tolist(),
        "route_schedules": list(route_schedules),
        "label_rows": int(len(label_rows)),
        "route_schedule_claim_rows": int(len(claim_rows)),
        "direct_route_rows": int(len(direct_rows)),
        "objective_wall_rows": int(len(objective_rows)),
        "support_movement_rows": int(len(movement_rows)),
        "polish_reversion_rows": int(len(polish_rows)),
        "route_claim_rows_csv": _rel(output_dir / ROUTE_CLAIM_CSV),
        "schedule_stable_pair_count": sum(
            1 for row in claim_rows
            if row.get("route_order_sensitivity_status") == "route_order_stable"
        ),
        "schedule_sensitive_pair_count": sum(
            1 for row in claim_rows
            if row.get("route_order_sensitivity_status") == "route_order_sensitive"
        ),
        "schedule_stable_distinct_partial_wall_evidence_pair_count": sum(
            1 for row in claim_rows
            if row.get("wall_claim_gate_status")
            == "passes_schedule_invariance_distinct_partial_wall_evidence"
        ),
        "schedule_stable_ambiguous_route_evidence_pair_count": sum(
            1 for row in claim_rows
            if row.get("wall_claim_gate_status")
            == "stable_route_evidence_basin_relation_ambiguous_no_supported_wall_claim"
        ),
        "error_count": int(len(errors)),
        "errors": errors,
        "baseline_cache_hits": int(stats.baseline_cache_hits),
        "baseline_cache_misses": int(stats.baseline_cache_misses),
        "endpoint_cache_hits": int(stats.endpoint_cache_hits),
        "endpoint_cache_misses": int(stats.endpoint_cache_misses),
        "progress_jsonl": _rel(progress_path),
        "cache_rows_csv": _rel(output_dir / CACHE_ROWS_CSV),
        "groups_per_step": int(groups_per_step),
        "max_route_steps": int(max_route_steps),
        "max_pairs": None if max_pairs is None else int(max_pairs),
        "baseline_iterations": int(baseline_iterations),
        "polish_iterations": int(polish_iterations),
        "resolution": float(resolution),
        "randomness": float(randomness),
        "claim_boundary": (
            "Uniform W1-W6 diagnostic evidence only; no basin-evaluation, "
            "directed-search, or supported wall claim is made here."
        ),
    }
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(
            {
                "script": _rel(Path(__file__)),
                "subset_dir": _rel(subset_dir),
                "output_dir": _rel(output_dir),
                "endpoint_cache_dir": _rel(endpoint_cache_dir),
                "reuse_endpoint_cache": bool(reuse_endpoint_cache),
                "pair_ids": sorted(pair_ids),
                "route_schedules": list(route_schedules),
                "groups_per_step": int(groups_per_step),
                "max_route_steps": int(max_route_steps),
                "max_pairs": None if max_pairs is None else int(max_pairs),
                "baseline_iterations": int(baseline_iterations),
                "polish_iterations": int(polish_iterations),
                "resolution": float(resolution),
                "randomness": float(randomness),
                "perturb_seed_offset": int(perturb_seed_offset),
                "scope": "uniform W1-W6 wall protocol only",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_report(output_dir / REPORT_MD, summary, label_rows, claim_rows)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-dir", type=Path, default=DEFAULT_SUBSET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--endpoint-cache-dir", type=Path, default=DEFAULT_ENDPOINT_CACHE_DIR)
    parser.add_argument(
        "--no-reuse-endpoint-cache",
        action="store_true",
        help="Ignore existing endpoint cache files and rebuild memberships.",
    )
    parser.add_argument(
        "--pair-ids",
        default="",
        help="Optional comma-separated panel_pair_id values to run.",
    )
    parser.add_argument(
        "--route-schedules",
        default="target_size_desc",
        help=(
            "Comma-separated deterministic target-group schedules: "
            "target_size_desc,target_size_asc,target_label_asc,target_label_desc."
        ),
    )
    parser.add_argument("--groups-per-step", type=int, default=512)
    parser.add_argument("--max-route-steps", type=int, default=16)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--baseline-iterations", type=int, default=10)
    parser.add_argument("--polish-iterations", type=int, default=5)
    parser.add_argument("--resolution", type=float, default=0.01)
    parser.add_argument("--randomness", type=float, default=0.01)
    parser.add_argument("--perturb-seed-offset", type=int, default=5000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run(
        subset_dir=args.subset_dir,
        output_dir=args.output_dir,
        endpoint_cache_dir=args.endpoint_cache_dir,
        reuse_endpoint_cache=not args.no_reuse_endpoint_cache,
        pair_ids=_parse_pair_ids(args.pair_ids),
        route_schedules=_parse_route_schedules(args.route_schedules),
        groups_per_step=args.groups_per_step,
        max_route_steps=args.max_route_steps,
        max_pairs=args.max_pairs,
        baseline_iterations=args.baseline_iterations,
        polish_iterations=args.polish_iterations,
        resolution=args.resolution,
        randomness=args.randomness,
        perturb_seed_offset=args.perturb_seed_offset,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
