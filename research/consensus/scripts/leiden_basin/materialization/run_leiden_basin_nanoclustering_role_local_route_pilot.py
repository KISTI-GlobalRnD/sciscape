#!/usr/bin/env python3
"""Run a bounded role-local fixed-mask route pilot.

This is a raw-route smoke runner.  It uses the role-local boundary plan to
select a small source+target pair mask, fixes every node outside that mask to
the source seed0 partition, and compares a source-state control arm with a
target-handle seeded arm.  Postprocess is intentionally excluded because the
current Rust postprocess wrapper does not accept fixed-node masks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sciscape.clustering.leiden_rust import RustLeidenGraph, build_leiden_graph


REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").exists()
)
BASE_RESULT_DIR = REPO_ROOT / "research/consensus/results/adaptive_refinement"
DEFAULT_READINESS_DIR = (
    BASE_RESULT_DIR / "leiden_basin_nanoclustering_endpoint_replay_readiness_20260601"
)
DEFAULT_BOUNDARY_PLAN_DIR = (
    BASE_RESULT_DIR / "leiden_basin_nanoclustering_role_local_boundary_plan_20260601"
)
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_role_local_route_pilot_smoke_20260601"
)

ENDPOINT_TARGET_ROWS_CSV = (
    "nanoclustering_endpoint_replay_readiness_endpoint_target_rows.csv"
)
GRAPH_INPUT_ROWS_CSV = "nanoclustering_endpoint_replay_readiness_graph_input_rows.csv"
EXECUTION_PLAN_ROWS_CSV = "nanoclustering_role_local_boundary_execution_plan_rows.csv"

ROUTE_ATTEMPT_ROWS_CSV = "nanoclustering_role_local_route_pilot_attempt_rows.csv"
ROUTE_PAIR_SCORE_ROWS_CSV = "nanoclustering_role_local_route_pilot_pair_score_rows.csv"
ROUTE_SUMMARY_JSON = "nanoclustering_role_local_route_pilot_summary.json"
ROUTE_CONFIG_JSON = "nanoclustering_role_local_route_pilot_config.json"
ROUTE_REPORT_MD = "nanoclustering_role_local_route_pilot_report.md"

CLAIM_BOUNDARY = (
    "NanoClustering role-local raw-route pilot only; executes fixed-outside "
    "raw Leiden arms over tiny source+target masks. It excludes postprocess, "
    "does not promote endpoint replay, route/pathway walls, quality/cost, "
    "real-data method success, or algorithm claims."
)
REPLAY_EXECUTION_STATUS = "raw_route_pilot_not_endpoint_replay"
ROUTE_EXECUTION_STATUS = "executed_role_local_fixed_mask_raw_route_pilot"
WALL_PROMOTION_STATUS = "not_promoted_raw_route_smoke_only"
QUALITY_COST_STATUS = "excluded_raw_route_pilot_only"


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


def _parse_csv_list(value: str | None, cast: Any = str) -> tuple[Any, ...]:
    if value is None or not str(value).strip():
        return ()
    return tuple(cast(part.strip()) for part in str(value).split(",") if part.strip())


def _compact_membership(membership: np.ndarray) -> np.ndarray:
    labels, _ = pd.factorize(np.asarray(membership), sort=False)
    return np.asarray(labels, dtype=np.uint64)


def _array_hash(values: np.ndarray) -> str:
    arr = np.ascontiguousarray(values, dtype=np.uint64)
    return hashlib.blake2b(arr.tobytes(), digest_size=16).hexdigest()


def _mask_hash(mask: np.ndarray) -> str:
    return hashlib.blake2b(np.packbits(mask.astype(np.bool_)).tobytes(), digest_size=16).hexdigest()


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


def _source_rows(endpoint_targets: pd.DataFrame, signature_id: str) -> pd.DataFrame:
    rows = endpoint_targets[
        endpoint_targets["endpoint_signature_id"].astype(str).eq(signature_id)
        & endpoint_targets["target_handle_role"].astype(str).eq(
            "dominant_host_context_member"
        )
        & endpoint_targets["membership_path_exists"].astype(bool)
        & endpoint_targets["cluster_label_present"].astype(bool)
    ].copy()
    return rows.sort_values(["seed", "endpoint_handle_id"], kind="mergesort")


def _target_row(endpoint_targets: pd.DataFrame, handle_id: str) -> pd.Series:
    rows = endpoint_targets[
        endpoint_targets["endpoint_handle_id"].astype(str).eq(str(handle_id))
        & endpoint_targets["target_handle_role"].astype(str).eq(
            "top1_endpoint_target_member"
        )
    ].copy()
    if rows.empty:
        raise ValueError(f"missing target endpoint handle: {handle_id}")
    return rows.iloc[0]


def _load_graph(
    graph_row: pd.Series,
    manifest_cache: dict[str, tuple[pd.DataFrame, np.ndarray]],
) -> tuple[RustLeidenGraph, np.ndarray, float]:
    manifest_path = Path(str(graph_row["runtime_node_manifest_path"]))
    edge_path = Path(str(graph_row["runtime_int_edges_path"]))
    key = str(manifest_path)
    if key not in manifest_cache:
        manifest_cache[key] = _load_manifest(manifest_path)
    manifest, weights = manifest_cache[key]
    start = time.perf_counter()
    graph = build_leiden_graph(edge_path=edge_path, n_nodes=len(manifest), node_weights=weights)
    return graph, weights, time.perf_counter() - start


def _select_plan_rows(
    plan: pd.DataFrame,
    *,
    case_ranks: tuple[int, ...],
    method_seeds: tuple[int, ...],
    route_arms: tuple[str, ...],
    max_targets_per_role: int,
) -> pd.DataFrame:
    selected = plan.copy()
    if case_ranks:
        selected = selected[selected["panel_case_rank"].astype(int).isin(case_ranks)]
    if method_seeds:
        selected = selected[selected["method_seed"].astype(int).isin(method_seeds)]
    if route_arms:
        selected = selected[selected["route_arm"].astype(str).isin(route_arms)]
    selected = selected.sort_values(
        ["panel_case_rank", "role_side", "target_seed", "route_arm", "method_seed"],
        kind="mergesort",
    )
    if max_targets_per_role > 0:
        keys = ["role_id", "target_handle_id"]
        keep = (
            selected[keys + ["target_seed"]]
            .drop_duplicates(keys)
            .sort_values(["role_id", "target_seed"], kind="mergesort")
            .groupby("role_id", sort=False)
            .head(max_targets_per_role)[keys]
        )
        selected = selected.merge(keep, on=keys, how="inner")
    return selected.reset_index(drop=True)


def _score_target(
    terminal: np.ndarray,
    initial: np.ndarray,
    pair_mask: np.ndarray,
    target_mask: np.ndarray,
    source_mask: np.ndarray,
    weights: np.ndarray,
) -> dict[str, Any]:
    labels = np.asarray(terminal, dtype=np.int64)
    pair_labels = labels[pair_mask]
    target_labels = labels[target_mask]
    source_labels = labels[source_mask]
    pair_initial = np.asarray(initial, dtype=np.uint64)[pair_mask]
    pair_terminal = np.asarray(terminal, dtype=np.uint64)[pair_mask]

    def _best(mask_labels: np.ndarray, mask: np.ndarray) -> tuple[int, int, float, int, float, float]:
        if mask_labels.size == 0:
            return -1, 0, 0.0, 0, 0.0, 0.0
        counts = np.bincount(mask_labels)
        best_label = int(counts.argmax())
        best_count = int(counts[best_label])
        best_weight = float(weights[mask][mask_labels == best_label].sum())
        total_count = int(mask.sum())
        total_weight = float(weights[mask].sum())
        return best_label, best_count, best_weight, total_count, total_weight, (
            best_weight / total_weight if total_weight else 0.0
        )

    target_best_label, target_best_count, target_best_weight, target_count, target_weight, target_purity = _best(
        target_labels,
        target_mask,
    )
    source_best_label, source_best_count, source_best_weight, source_count, source_weight, source_purity = _best(
        source_labels,
        source_mask,
    )
    overlap = np.logical_and(source_mask, target_mask)
    overlap_weight = float(weights[overlap].sum())
    return {
        "pair_terminal_hash": _array_hash(_compact_membership(pair_terminal)),
        "pair_initial_hash": _array_hash(_compact_membership(pair_initial)),
        "pair_changed_vs_initial": _array_hash(_compact_membership(pair_terminal))
        != _array_hash(_compact_membership(pair_initial)),
        "pair_terminal_cluster_count": int(np.unique(pair_labels).size),
        "target_best_terminal_cluster_id": target_best_label,
        "target_best_terminal_cluster_node_count": target_best_count,
        "target_best_terminal_cluster_doc_sum": target_best_weight,
        "target_node_count": target_count,
        "target_doc_sum": target_weight,
        "target_best_cluster_doc_share": target_purity,
        "source_best_terminal_cluster_id": source_best_label,
        "source_best_terminal_cluster_node_count": source_best_count,
        "source_best_terminal_cluster_doc_sum": source_best_weight,
        "source_node_count": source_count,
        "source_doc_sum": source_weight,
        "source_best_cluster_doc_share": source_purity,
        "source_target_overlap_doc_sum": overlap_weight,
        "source_target_overlap_source_doc_share": (
            overlap_weight / source_weight if source_weight else 0.0
        ),
        "source_target_overlap_target_doc_share": (
            overlap_weight / target_weight if target_weight else 0.0
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    readiness_dir = Path(args.readiness_dir)
    boundary_plan_dir = Path(args.boundary_plan_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    endpoint_targets = _read_csv(readiness_dir / ENDPOINT_TARGET_ROWS_CSV)
    graph_rows = _read_csv(readiness_dir / GRAPH_INPUT_ROWS_CSV)
    execution_plan = _read_csv(boundary_plan_dir / EXECUTION_PLAN_ROWS_CSV)
    selected = _select_plan_rows(
        execution_plan,
        case_ranks=_parse_csv_list(args.case_ranks, int),
        method_seeds=_parse_csv_list(args.method_seeds, int),
        route_arms=_parse_csv_list(args.route_arms, str),
        max_targets_per_role=int(args.max_targets_per_role),
    )

    graph_by_branch = {
        str(row["branch"]): row
        for _, row in graph_rows.iterrows()
        if str(row.get("runtime_graph_status", "")).startswith("ready_")
    }
    manifest_cache: dict[str, tuple[pd.DataFrame, np.ndarray]] = {}
    label_cache: dict[tuple[str, str], np.ndarray] = {}
    graph_cache: dict[str, tuple[RustLeidenGraph, np.ndarray, float]] = {}

    attempt_rows: list[dict[str, Any]] = []
    pair_score_rows: list[dict[str, Any]] = []
    for _, route in selected.iterrows():
        branch = str(route["branch"])
        if branch not in graph_cache:
            graph, weights, load_seconds = _load_graph(graph_by_branch[branch], manifest_cache)
            graph_cache[branch] = graph, weights, load_seconds
        graph, weights, graph_load_seconds = graph_cache[branch]
        n_nodes = int(graph.n_nodes)

        source = _source_rows(endpoint_targets, str(route["endpoint_signature_id"]))
        if source.empty:
            raise ValueError(f"missing source handle for {route['endpoint_signature_id']}")
        source_full_path = Path(str(source.iloc[0]["membership_path"]))
        label_col = str(source.iloc[0]["label_cols"]).split(";")[0] or "candidate_micro_id"
        initial_labels = _compact_membership(_load_label_array(source_full_path, label_col))
        source_mask = _union_masks(source, label_cache, n_nodes)
        target = _target_row(endpoint_targets, str(route["target_handle_id"]))
        target_mask = _mask_for_row(target, label_cache)
        pair_mask = np.logical_or(source_mask, target_mask)
        route_arm = str(route["route_arm"])
        fixed_nodes = ~pair_mask

        initial = np.asarray(initial_labels, dtype=np.uint64).copy()
        if route_arm == "target_handle_seeded_fixed_outside":
            initial[target_mask] = np.uint64(initial.max() + 1)
        elif route_arm == "target_anchor_fixed_source_free":
            initial[target_mask] = np.uint64(initial.max() + 1)
            fixed_nodes = np.logical_or(~pair_mask, target_mask)
        elif route_arm == "source_anchor_fixed_target_free":
            fixed_nodes = np.logical_or(~pair_mask, source_mask)
        elif route_arm != "source_state_fixed_outside_control":
            raise ValueError(f"unsupported route arm: {route_arm}")

        actual_free_count = int((~fixed_nodes).sum())
        raw_route_executed = actual_free_count > 0
        if raw_route_executed:
            start = time.perf_counter()
            raw = graph.run_leiden(
                resolution=float(args.gamma),
                n_iterations=int(args.n_iterations),
                seed=int(route["method_seed"]),
                initial_membership=initial,
                fixed_nodes=fixed_nodes,
            )
            route_seconds = time.perf_counter() - start
            terminal = _compact_membership(raw.membership)
            raw_n_clusters = int(raw.n_clusters)
            raw_quality = float(raw.quality)
            row_route_status = ROUTE_EXECUTION_STATUS
        else:
            route_seconds = 0.0
            terminal = _compact_membership(initial)
            raw_n_clusters = int(np.unique(terminal).size)
            raw_quality = None
            row_route_status = "blocked_empty_free_mask_before_rust"
        score = _score_target(
            terminal=terminal,
            initial=initial,
            pair_mask=pair_mask,
            target_mask=target_mask,
            source_mask=source_mask,
            weights=weights,
        )
        attempt_row = {
            "route_attempt_id": route["route_attempt_id"],
            "panel_case_id": route["panel_case_id"],
            "panel_case_rank": int(route["panel_case_rank"]),
            "role_id": route["role_id"],
            "role_side": route["role_side"],
            "branch": branch,
            "endpoint_signature_id": route["endpoint_signature_id"],
            "target_handle_id": route["target_handle_id"],
            "method_seed": int(route["method_seed"]),
            "route_arm": route["route_arm"],
            "n_nodes": n_nodes,
            "n_edges": int(graph.n_edges),
            "pair_mask_hash": _mask_hash(pair_mask),
            "source_mask_hash": _mask_hash(source_mask),
            "target_mask_hash": _mask_hash(target_mask),
            "pair_node_count": int(pair_mask.sum()),
            "fixed_node_count": int(fixed_nodes.sum()),
            "free_node_count": actual_free_count,
            "free_node_share": float(actual_free_count / n_nodes),
            "raw_route_executed": raw_route_executed,
            "raw_n_clusters": raw_n_clusters,
            "raw_quality": raw_quality,
            "graph_load_seconds_cached_branch": float(graph_load_seconds),
            "route_seconds": float(route_seconds),
            **score,
            "replay_execution_status": REPLAY_EXECUTION_STATUS,
            "route_execution_status": row_route_status,
            "wall_promotion_status": WALL_PROMOTION_STATUS,
            "quality_cost_status": QUALITY_COST_STATUS,
            "claim_boundary": CLAIM_BOUNDARY,
        }
        attempt_rows.append(attempt_row)
        pair_score_rows.append(
            {
                key: attempt_row[key]
                for key in [
                    "route_attempt_id",
                    "panel_case_id",
                    "panel_case_rank",
                    "role_id",
                    "role_side",
                    "branch",
                    "target_handle_id",
                    "method_seed",
                    "route_arm",
                    "free_node_count",
                    "target_best_cluster_doc_share",
                    "source_best_cluster_doc_share",
                    "pair_terminal_cluster_count",
                    "pair_changed_vs_initial",
                ]
            }
        )
        _write_csv(pd.DataFrame(attempt_rows), output_dir / ROUTE_ATTEMPT_ROWS_CSV)
        _write_csv(pd.DataFrame(pair_score_rows), output_dir / ROUTE_PAIR_SCORE_ROWS_CSV)

    attempts = pd.DataFrame(attempt_rows)
    if attempts.empty:
        arm_distinction_rows = []
    else:
        arm_distinction_rows = []
        keys = ["role_id", "target_handle_id", "method_seed"]
        for key, group in attempts.groupby(keys, sort=False):
            hashes = group.set_index("route_arm")["pair_terminal_hash"].to_dict()
            source_hash = hashes.get("source_state_fixed_outside_control")
            target_hash = hashes.get("target_handle_seeded_fixed_outside")
            unique_hash_count = int(group["pair_terminal_hash"].nunique(dropna=True))
            arm_distinction_rows.append(
                {
                    "role_id": key[0],
                    "target_handle_id": key[1],
                    "method_seed": int(key[2]),
                    "route_arm_count": int(group["route_arm"].nunique()),
                    "terminal_hash_unique_count": unique_hash_count,
                    "route_arm_hashes": ";".join(
                        f"{arm}={hash_value}" for arm, hash_value in sorted(hashes.items())
                    ),
                    "source_control_hash": source_hash,
                    "target_seeded_hash": target_hash,
                    "route_arms_distinct": unique_hash_count > 1,
                }
            )
    arm_distinctions = pd.DataFrame(arm_distinction_rows)
    if not arm_distinctions.empty:
        arm_distinctions.to_csv(
            output_dir / "nanoclustering_role_local_route_pilot_arm_distinction_rows.csv",
            index=False,
        )

    summary = {
        "schema": "nanoclustering_role_local_route_pilot_summary.v1",
        "status": "executed_role_local_raw_route_pilot",
        "readiness_dir": str(readiness_dir),
        "boundary_plan_dir": str(boundary_plan_dir),
        "output_dir": str(output_dir),
        "selected_route_attempt_count": int(len(selected)),
        "executed_route_attempt_count": int(len(attempts)),
        "branch_count": int(attempts["branch"].nunique()) if not attempts.empty else 0,
        "role_count": int(attempts["role_id"].nunique()) if not attempts.empty else 0,
        "target_handle_count": int(attempts["target_handle_id"].nunique()) if not attempts.empty else 0,
        "arm_distinction_pair_count": int(len(arm_distinctions)),
        "route_arms_distinct_count": (
            int(arm_distinctions["route_arms_distinct"].sum())
            if not arm_distinctions.empty
            else 0
        ),
        "mean_free_node_count": (
            float(attempts["free_node_count"].mean()) if not attempts.empty else None
        ),
        "total_route_seconds": (
            float(attempts["route_seconds"].sum()) if not attempts.empty else 0.0
        ),
        "blocked_empty_free_mask_count": (
            int(
                attempts["route_execution_status"]
                .astype(str)
                .eq("blocked_empty_free_mask_before_rust")
                .sum()
            )
            if not attempts.empty
            else 0
        ),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / ROUTE_SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    config = {
        "schema": "nanoclustering_role_local_route_pilot.v1",
        "readiness_dir": str(readiness_dir),
        "boundary_plan_dir": str(boundary_plan_dir),
        "output_dir": str(output_dir),
        "case_ranks": list(_parse_csv_list(args.case_ranks, int)),
        "method_seeds": list(_parse_csv_list(args.method_seeds, int)),
        "route_arms": list(_parse_csv_list(args.route_arms, str)),
        "max_targets_per_role": int(args.max_targets_per_role),
        "gamma": float(args.gamma),
        "n_iterations": int(args.n_iterations),
        "postprocess": "excluded_raw_fixed_mask_smoke",
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / ROUTE_CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(output_dir=output_dir, summary=summary, attempts=attempts)
    return summary


def _write_report(*, output_dir: Path, summary: dict[str, Any], attempts: pd.DataFrame) -> None:
    lines = [
        "# NanoClustering Role-Local Raw Route Pilot",
        "",
        f"- status: `{summary['status']}`",
        f"- executed_route_attempt_count: {summary['executed_route_attempt_count']}",
        f"- role_count: {summary['role_count']}",
        f"- target_handle_count: {summary['target_handle_count']}",
        f"- route_arms_distinct_count: {summary['route_arms_distinct_count']}",
        f"- total_route_seconds: {summary['total_route_seconds']}",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Attempts",
    ]
    if attempts.empty:
        lines.append("- no attempts executed")
    else:
        for row in attempts.sort_values(["role_side", "route_arm"]).itertuples(index=False):
            data = row._asdict()
            lines.append(
                "- "
                f"{data['route_attempt_id']}: free={data['free_node_count']}, "
                f"pair_changed={data['pair_changed_vs_initial']}, "
                f"target_doc_share={data['target_best_cluster_doc_share']}, "
                f"source_doc_share={data['source_best_cluster_doc_share']}, "
                f"seconds={data['route_seconds']:.3f}"
            )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            (
                "This is raw fixed-mask evidence only. Because postprocess is not "
                "fixed-mask aware in the current wrapper, endpoint-level replay and "
                "wall/pathway claims remain closed."
            ),
            "",
        ]
    )
    (output_dir / ROUTE_REPORT_MD).write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness-dir", type=Path, default=DEFAULT_READINESS_DIR)
    parser.add_argument("--boundary-plan-dir", type=Path, default=DEFAULT_BOUNDARY_PLAN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--case-ranks", default="4")
    parser.add_argument("--method-seeds", default="0")
    parser.add_argument(
        "--route-arms",
        default="source_state_fixed_outside_control,target_handle_seeded_fixed_outside",
    )
    parser.add_argument("--max-targets-per-role", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.7)
    parser.add_argument("--n-iterations", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
