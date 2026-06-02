#!/usr/bin/env python3
"""Run a NanoClustering symmetric-object multistart pilot.

Earlier local pair, common-mask, signature-universe, and case-union probes did
not expose terminal multiplicity. This runner moves the feasible set to an
anchor-independent all-seed symmetric endpoint object. For each selected
role-level object universe, it fixes every node outside the object mask and
tests whether varied object-level initializations converge to distinct terminal
states.

It is a terminal-multiplicity diagnostic only. It does not promote wall/pathway
claims, inspect basin quality/cost, claim real-data method success, or claim
algorithm novelty.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sciscape.clustering.integer_remap import ensure_int_edge_sidecars

from materialize_leiden_basin_nanoclustering_symmetric_object_universe_plan import (
    DEFAULT_LANDSCAPE_DIR,
    DEFAULT_OUTPUT_DIR as DEFAULT_OBJECT_UNIVERSE_DIR,
    DEFAULT_SYMMETRIC_OBJECT_DIR,
    ENDPOINT_REGISTRY_CSV,
    OBJECT_COMPONENTS_CSV,
    SYMMETRIC_ROLE_ROWS_CSV,
    _component_mask,
    _pure_seed_membership_registry,
)
from run_leiden_basin_nanoclustering_anchor_release_pilot import _run_leiden_or_hold
from run_leiden_basin_nanoclustering_common_mask_multistart_pilot import (
    _pair_singleton_initial,
    _random_pair_initial,
)
from run_leiden_basin_nanoclustering_role_local_route_pilot import (
    BASE_RESULT_DIR,
    DEFAULT_READINESS_DIR,
    GRAPH_INPUT_ROWS_CSV,
    _array_hash,
    _compact_membership,
    _json_safe,
    _load_graph,
    _load_label_array,
    _mask_hash,
    _parse_csv_list,
    _read_csv,
    _write_csv,
)


DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_symmetric_object_multistart_strict_anchor_independent_p1_seed0_20260601"
)

OBJECT_ATTEMPT_ROWS_CSV = "nanoclustering_symmetric_object_multistart_attempt_rows.csv"
OBJECT_SUMMARY_ROWS_CSV = "nanoclustering_symmetric_object_multistart_object_rows.csv"
TERMINAL_PAIR_ROWS_CSV = "nanoclustering_symmetric_object_multistart_terminal_pair_rows.csv"
OBJECT_CONFIG_JSON = "nanoclustering_symmetric_object_multistart_config.json"
OBJECT_SUMMARY_JSON = "nanoclustering_symmetric_object_multistart_summary.json"
OBJECT_REPORT_MD = "nanoclustering_symmetric_object_multistart_report.md"
TERMINAL_MEMBERSHIP_DIRNAME = "terminal_memberships"

SELECTION_POLICIES = {
    "route_priority",
    "largest_object_node_count",
    "largest_component_sum_ratio",
    "largest_seed0_ratio",
}

CLAIM_BOUNDARY = (
    "NanoClustering symmetric-object/support-neighborhood multistart "
    "terminal-multiplicity pilot only; runs varied initializations under a "
    "fixed-outside all-seed symmetric object mask, optionally expanded by "
    "boundary-attachment support nodes. It does not promote wall/pathway claims, "
    "inspect basin quality/cost, claim real-data method success, or claim "
    "algorithm novelty."
)
RUN_STATUS = "executed_symmetric_object_multistart_pilot"

TERMINAL_PAIR_COLUMNS = [
    "object_role_universe_id",
    "left_start_id",
    "right_start_id",
    "left_start_policy",
    "right_start_policy",
    "left_start_index",
    "right_start_index",
    "left_object_terminal_hash",
    "right_object_terminal_hash",
    "left_universe_terminal_hash",
    "right_universe_terminal_hash",
    "same_object_terminal_hash",
    "same_universe_terminal_hash",
    "object_terminal_ari",
    "universe_terminal_ari",
    "left_quality",
    "right_quality",
    "quality_delta_abs",
    "left_terminal_membership_npz_path",
    "right_terminal_membership_npz_path",
    "pair_status",
    "run_status",
    "claim_boundary",
]


def _bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().eq("true")


def _series_startswith_any(series: pd.Series, prefixes: tuple[str, ...]) -> pd.Series:
    if not prefixes:
        return pd.Series(True, index=series.index)
    text = series.astype(str)
    out = pd.Series(False, index=series.index)
    for prefix in prefixes:
        out |= text.str.startswith(prefix)
    return out


def _select_object_rows(
    role_rows: pd.DataFrame,
    *,
    case_ranks: tuple[int, ...],
    role_sides: tuple[str, ...],
    analysis_tiers: tuple[str, ...],
    object_status_prefixes: tuple[str, ...],
    probe_priority_prefixes: tuple[str, ...],
    strict_core_only: bool,
    selection_policy: str,
    max_roles: int,
    dedupe_symmetric_objects: bool,
) -> pd.DataFrame:
    if selection_policy not in SELECTION_POLICIES:
        raise ValueError(
            f"unsupported selection policy: {selection_policy}; "
            f"expected one of {sorted(SELECTION_POLICIES)}"
        )
    rows = role_rows.copy()
    if case_ranks:
        rows = rows[rows["panel_case_rank"].astype(int).isin(case_ranks)]
    if role_sides:
        rows = rows[rows["role_side"].astype(str).isin(role_sides)]
    if analysis_tiers:
        rows = rows[rows["analysis_tier"].astype(str).isin(analysis_tiers)]
    if object_status_prefixes:
        rows = rows[
            _series_startswith_any(rows["object_resolution_status"], object_status_prefixes)
        ]
    if probe_priority_prefixes:
        rows = rows[_series_startswith_any(rows["probe_priority"], probe_priority_prefixes)]
    if strict_core_only:
        rows = rows[_bool_series(rows["strict_core_v0"])]

    if selection_policy == "largest_object_node_count":
        sort_cols = [
            "object_node_count",
            "symmetric_object_route_priority_rank",
            "panel_case_rank",
            "role_side",
        ]
        ascending = [False, True, True, True]
    elif selection_policy == "largest_component_sum_ratio":
        sort_cols = [
            "object_node_count_vs_component_sum_ratio",
            "object_node_count",
            "symmetric_object_route_priority_rank",
            "panel_case_rank",
        ]
        ascending = [False, False, True, True]
    elif selection_policy == "largest_seed0_ratio":
        sort_cols = [
            "object_node_count_vs_seed0_ratio",
            "object_node_count",
            "symmetric_object_route_priority_rank",
            "panel_case_rank",
        ]
        ascending = [False, False, True, True]
    else:
        sort_cols = [
            "symmetric_object_route_priority_rank",
            "panel_case_rank",
            "role_side",
        ]
        ascending = [True, True, True]
    rows = rows.sort_values(sort_cols, ascending=ascending, kind="mergesort")
    if dedupe_symmetric_objects:
        rows = rows.drop_duplicates(["branch", "symmetric_object_id"], keep="first")
    if max_roles > 0:
        rows = rows.head(max_roles)
    return rows.reset_index(drop=True)


def _comb2(value: int) -> float:
    if value < 2:
        return 0.0
    return float(value * (value - 1) / 2)


def _adjusted_rand_score(left: np.ndarray, right: np.ndarray) -> float:
    left_labels = pd.factorize(np.asarray(left), sort=False)[0]
    right_labels = pd.factorize(np.asarray(right), sort=False)[0]
    n = int(left_labels.size)
    if n < 2:
        return 1.0
    contingency: dict[tuple[int, int], int] = {}
    left_counts: dict[int, int] = {}
    right_counts: dict[int, int] = {}
    for a, b in zip(left_labels, right_labels, strict=True):
        ai = int(a)
        bi = int(b)
        contingency[(ai, bi)] = contingency.get((ai, bi), 0) + 1
        left_counts[ai] = left_counts.get(ai, 0) + 1
        right_counts[bi] = right_counts.get(bi, 0) + 1
    sum_comb = sum(_comb2(count) for count in contingency.values())
    sum_left = sum(_comb2(count) for count in left_counts.values())
    sum_right = sum(_comb2(count) for count in right_counts.values())
    total = _comb2(n)
    if total == 0:
        return 1.0
    expected = (sum_left * sum_right) / total
    max_index = 0.5 * (sum_left + sum_right)
    denom = max_index - expected
    if denom == 0:
        return 1.0 if sum_comb == max_index else 0.0
    return float((sum_comb - expected) / denom)


def _best_cluster_stats(
    *,
    terminal: np.ndarray,
    mask: np.ndarray,
    weights: np.ndarray,
) -> tuple[int, int, float, int, float, float]:
    labels = np.asarray(terminal, dtype=np.int64)[mask]
    if labels.size == 0:
        return -1, 0, 0.0, 0, 0.0, 0.0
    counts = np.bincount(labels)
    best_label = int(counts.argmax())
    best_count = int(counts[best_label])
    best_weight = float(weights[mask][labels == best_label].sum())
    total_count = int(mask.sum())
    total_weight = float(weights[mask].sum())
    return best_label, best_count, best_weight, total_count, total_weight, (
        best_weight / total_weight if total_weight else 0.0
    )


def _load_branch_edge_sidecars(
    graph_row: pd.Series,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    src_path, dst_path, weight_path = ensure_int_edge_sidecars(
        Path(str(graph_row["runtime_int_edges_path"]))
    )
    return (
        np.memmap(src_path, dtype=np.uint32, mode="r"),
        np.memmap(dst_path, dtype=np.uint32, mode="r"),
        np.memmap(weight_path, dtype=np.float64, mode="r"),
    )


def _support_neighborhood_mask(
    *,
    object_mask: np.ndarray,
    edge_src: np.ndarray,
    edge_dst: np.ndarray,
    edge_weight: np.ndarray,
    top_k: int,
    min_weight: float,
) -> dict[str, Any]:
    n_nodes = int(object_mask.size)
    if top_k <= 0:
        support_mask = np.zeros(n_nodes, dtype=np.bool_)
        return {
            "support_mask": support_mask,
            "support_neighborhood_status": "disabled_object_only",
            "support_neighborhood_top_k": int(top_k),
            "support_neighborhood_min_weight": float(min_weight),
            "support_node_count": 0,
            "support_edge_weight_sum": 0.0,
            "support_edge_weight_min": 0.0,
            "support_edge_weight_median": None,
            "support_edge_weight_max": 0.0,
            "support_node_ids": "",
        }

    src_in = object_mask[edge_src]
    dst_in = object_mask[edge_dst]
    boundary = np.logical_xor(src_in, dst_in)
    if not bool(boundary.any()):
        support_mask = np.zeros(n_nodes, dtype=np.bool_)
        return {
            "support_mask": support_mask,
            "support_neighborhood_status": "blocked_no_object_boundary_edges",
            "support_neighborhood_top_k": int(top_k),
            "support_neighborhood_min_weight": float(min_weight),
            "support_node_count": 0,
            "support_edge_weight_sum": 0.0,
            "support_edge_weight_min": 0.0,
            "support_edge_weight_median": None,
            "support_edge_weight_max": 0.0,
            "support_node_ids": "",
        }

    idx = np.flatnonzero(boundary)
    external_nodes = np.where(src_in[idx], edge_dst[idx], edge_src[idx]).astype(
        np.int64,
        copy=False,
    )
    scores = np.bincount(
        external_nodes,
        weights=np.asarray(edge_weight[idx], dtype=np.float64),
        minlength=n_nodes,
    )
    scores[object_mask] = 0.0
    if min_weight > 0:
        eligible = np.flatnonzero(scores >= float(min_weight))
    else:
        eligible = np.flatnonzero(scores > 0.0)
    if eligible.size == 0:
        support_mask = np.zeros(n_nodes, dtype=np.bool_)
        return {
            "support_mask": support_mask,
            "support_neighborhood_status": "blocked_no_positive_support_scores",
            "support_neighborhood_top_k": int(top_k),
            "support_neighborhood_min_weight": float(min_weight),
            "support_node_count": 0,
            "support_edge_weight_sum": 0.0,
            "support_edge_weight_min": 0.0,
            "support_edge_weight_median": None,
            "support_edge_weight_max": 0.0,
            "support_node_ids": "",
        }

    k = min(int(top_k), int(eligible.size))
    if k < int(eligible.size):
        local_order = np.argpartition(-scores[eligible], k - 1)[:k]
        chosen = eligible[local_order]
        chosen = chosen[np.argsort(-scores[chosen], kind="mergesort")]
    else:
        chosen = eligible[np.argsort(-scores[eligible], kind="mergesort")]
    support_mask = np.zeros(n_nodes, dtype=np.bool_)
    support_mask[chosen] = True
    chosen_scores = scores[chosen]
    return {
        "support_mask": support_mask,
        "support_neighborhood_status": "computed_boundary_weight_topk",
        "support_neighborhood_top_k": int(top_k),
        "support_neighborhood_min_weight": float(min_weight),
        "support_node_count": int(support_mask.sum()),
        "support_edge_weight_sum": float(chosen_scores.sum()),
        "support_edge_weight_min": float(chosen_scores.min()) if chosen_scores.size else 0.0,
        "support_edge_weight_median": float(np.median(chosen_scores))
        if chosen_scores.size
        else None,
        "support_edge_weight_max": float(chosen_scores.max()) if chosen_scores.size else 0.0,
        "support_node_ids": ";".join(str(int(node)) for node in chosen),
    }


def _load_object_masks(
    *,
    branch: str,
    symmetric_object_id: str,
    components: pd.DataFrame,
    membership_registry: dict[tuple[str, int], tuple[Path, str]],
    label_cache: dict[tuple[str, str], np.ndarray],
    n_nodes: int,
    weights: np.ndarray,
) -> dict[str, Any]:
    rows = components[
        components["branch"].astype(str).eq(branch)
        & components["symmetric_object_id"].astype(str).eq(str(symmetric_object_id))
    ].copy()
    rows = rows.sort_values(["seed", "cluster_id"], kind="mergesort")
    object_mask = np.zeros(n_nodes, dtype=np.bool_)
    seed0_mask = np.zeros(n_nodes, dtype=np.bool_)
    component_masks: list[dict[str, Any]] = []
    seed_masks: dict[int, np.ndarray] = {}
    status_counts: dict[str, int] = {}
    for _, component in rows.iterrows():
        mask, status = _component_mask(
            component=component,
            membership_registry=membership_registry,
            label_cache=label_cache,
            n_nodes=n_nodes,
        )
        status_counts[status] = status_counts.get(status, 0) + 1
        if status != "resolved":
            continue
        seed = int(component["seed"])
        object_mask |= mask
        if seed == 0:
            seed0_mask |= mask
        if seed not in seed_masks:
            seed_masks[seed] = np.zeros(n_nodes, dtype=np.bool_)
        seed_masks[seed] |= mask
        component_masks.append(
            {
                "seed": seed,
                "cluster_id": int(component["cluster_id"]),
                "endpoint_node_id": str(component["endpoint_node_id"]),
                "mask": mask,
            }
        )
    return {
        "object_mask": object_mask,
        "seed0_mask": seed0_mask,
        "seed_masks": seed_masks,
        "component_masks": component_masks,
        "component_rows": rows,
        "component_resolution_status_counts": ";".join(
            f"{key}:{status_counts[key]}" for key in sorted(status_counts)
        ),
        "object_doc_sum": float(weights[object_mask].sum()),
        "seed0_object_doc_sum": float(weights[seed0_mask].sum()),
    }


def _component_pattern_membership(
    *,
    initial_labels: np.ndarray,
    object_mask: np.ndarray,
    component_masks: list[dict[str, Any]],
) -> tuple[np.ndarray, int, int]:
    initial = np.asarray(initial_labels, dtype=np.uint64).copy()
    object_indices = np.flatnonzero(object_mask)
    signature_to_label: dict[tuple[int, ...], int] = {}
    next_label = int(initial.max()) + 1
    unassigned_count = 0
    for node in object_indices:
        signature = tuple(
            idx
            for idx, component in enumerate(component_masks)
            if bool(component["mask"][node])
        )
        if not signature:
            signature = (-1,)
            unassigned_count += 1
        if signature not in signature_to_label:
            signature_to_label[signature] = next_label
            next_label += 1
        initial[node] = np.uint64(signature_to_label[signature])
    return initial, len(signature_to_label), unassigned_count


def _seed0_object_seeded_initial(
    *,
    initial_labels: np.ndarray,
    seed0_mask: np.ndarray,
) -> np.ndarray:
    initial = np.asarray(initial_labels, dtype=np.uint64).copy()
    if int(seed0_mask.sum()):
        initial[seed0_mask] = np.uint64(int(initial.max()) + 1)
    return initial


def _object_start_specs(
    *,
    initial_labels: np.ndarray,
    object_mask: np.ndarray,
    seed0_mask: np.ndarray,
    component_pattern_membership: np.ndarray,
    method_seed: int,
    random_start_count: int,
    random_block_count: int,
) -> list[dict[str, Any]]:
    specs = [
        {
            "start_policy": "seed0_source_state",
            "start_index": 0,
            "leiden_seed": int(method_seed),
            "membership": np.asarray(initial_labels, dtype=np.uint64).copy(),
        },
        {
            "start_policy": "seed0_object_seeded",
            "start_index": 1,
            "leiden_seed": int(method_seed),
            "membership": _seed0_object_seeded_initial(
                initial_labels=initial_labels,
                seed0_mask=seed0_mask,
            ),
        },
        {
            "start_policy": "object_singleton",
            "start_index": 2,
            "leiden_seed": int(method_seed),
            "membership": _pair_singleton_initial(
                initial_labels=initial_labels,
                pair_mask=object_mask,
            ),
        },
        {
            "start_policy": "object_seed_component_blocks",
            "start_index": 3,
            "leiden_seed": int(method_seed),
            "membership": component_pattern_membership,
        },
    ]
    for idx in range(int(random_start_count)):
        start_seed = int(method_seed) * 1000 + idx + 1
        specs.append(
            {
                "start_policy": f"random_object_blocks_{idx:03d}",
                "start_index": len(specs),
                "leiden_seed": start_seed,
                "membership": _random_pair_initial(
                    initial_labels=initial_labels,
                    pair_mask=object_mask,
                    seed=start_seed,
                    block_count=int(random_block_count),
                ),
            }
        )
    return specs


def _score_object(
    *,
    terminal: np.ndarray,
    initial: np.ndarray,
    component_reference: np.ndarray,
    object_mask: np.ndarray,
    universe_mask: np.ndarray,
    support_mask: np.ndarray,
    seed0_mask: np.ndarray,
    seed_masks: dict[int, np.ndarray],
    weights: np.ndarray,
) -> dict[str, Any]:
    object_initial = np.asarray(initial, dtype=np.uint64)[object_mask]
    object_terminal = np.asarray(terminal, dtype=np.uint64)[object_mask]
    universe_initial = np.asarray(initial, dtype=np.uint64)[universe_mask]
    universe_terminal = np.asarray(terminal, dtype=np.uint64)[universe_mask]
    component_reference_object = np.asarray(component_reference, dtype=np.uint64)[object_mask]
    object_best = _best_cluster_stats(terminal=terminal, mask=object_mask, weights=weights)
    universe_best = _best_cluster_stats(
        terminal=terminal,
        mask=universe_mask,
        weights=weights,
    )
    support_best = _best_cluster_stats(
        terminal=terminal,
        mask=support_mask,
        weights=weights,
    )
    seed0_best = _best_cluster_stats(terminal=terminal, mask=seed0_mask, weights=weights)

    seed_shares: list[float] = []
    seed_node_counts: list[int] = []
    seed_doc_sums: list[float] = []
    for seed in sorted(seed_masks):
        mask = seed_masks[seed]
        _, _, _, count, weight, share = _best_cluster_stats(
            terminal=terminal,
            mask=mask,
            weights=weights,
        )
        seed_node_counts.append(count)
        seed_doc_sums.append(weight)
        seed_shares.append(share)

    terminal_hash = _array_hash(_compact_membership(object_terminal))
    initial_hash = _array_hash(_compact_membership(object_initial))
    universe_terminal_hash = _array_hash(_compact_membership(universe_terminal))
    universe_initial_hash = _array_hash(_compact_membership(universe_initial))
    return {
        "object_terminal_hash": terminal_hash,
        "object_initial_hash": initial_hash,
        "object_changed_vs_initial": terminal_hash != initial_hash,
        "object_mask_hash": _mask_hash(object_mask),
        "support_mask_hash": _mask_hash(support_mask),
        "universe_mask_hash": _mask_hash(universe_mask),
        "universe_terminal_hash": universe_terminal_hash,
        "universe_initial_hash": universe_initial_hash,
        "universe_changed_vs_initial": universe_terminal_hash != universe_initial_hash,
        "seed0_object_mask_hash": _mask_hash(seed0_mask),
        "object_terminal_cluster_count": int(np.unique(object_terminal).size)
        if object_terminal.size
        else 0,
        "object_best_terminal_cluster_id": object_best[0],
        "object_best_terminal_cluster_node_count": object_best[1],
        "object_best_terminal_cluster_doc_sum": object_best[2],
        "object_node_count": object_best[3],
        "object_doc_sum": object_best[4],
        "object_best_cluster_doc_share": object_best[5],
        "universe_terminal_cluster_count": int(np.unique(universe_terminal).size)
        if universe_terminal.size
        else 0,
        "universe_best_terminal_cluster_id": universe_best[0],
        "universe_best_terminal_cluster_node_count": universe_best[1],
        "universe_best_terminal_cluster_doc_sum": universe_best[2],
        "universe_node_count": universe_best[3],
        "universe_doc_sum": universe_best[4],
        "universe_best_cluster_doc_share": universe_best[5],
        "support_best_terminal_cluster_id": support_best[0],
        "support_best_terminal_cluster_node_count": support_best[1],
        "support_best_terminal_cluster_doc_sum": support_best[2],
        "support_node_count_score": support_best[3],
        "support_doc_sum": support_best[4],
        "support_best_cluster_doc_share": support_best[5],
        "seed0_best_terminal_cluster_id": seed0_best[0],
        "seed0_best_terminal_cluster_node_count": seed0_best[1],
        "seed0_best_terminal_cluster_doc_sum": seed0_best[2],
        "seed0_object_node_count": seed0_best[3],
        "seed0_object_doc_sum": seed0_best[4],
        "seed0_best_cluster_doc_share": seed0_best[5],
        "seed_component_count": len(seed_masks),
        "seed_component_node_count_min": int(min(seed_node_counts)) if seed_node_counts else 0,
        "seed_component_node_count_median": float(np.median(seed_node_counts))
        if seed_node_counts
        else None,
        "seed_component_node_count_max": int(max(seed_node_counts)) if seed_node_counts else 0,
        "seed_component_doc_sum_min": float(min(seed_doc_sums)) if seed_doc_sums else 0.0,
        "seed_component_doc_sum_median": float(np.median(seed_doc_sums))
        if seed_doc_sums
        else None,
        "seed_component_doc_sum_max": float(max(seed_doc_sums)) if seed_doc_sums else 0.0,
        "seed_component_best_cluster_doc_share_min": float(min(seed_shares))
        if seed_shares
        else 0.0,
        "seed_component_best_cluster_doc_share_median": float(np.median(seed_shares))
        if seed_shares
        else None,
        "seed_component_best_cluster_doc_share_max": float(max(seed_shares))
        if seed_shares
        else 0.0,
        "initial_terminal_object_ari": _adjusted_rand_score(
            object_initial,
            object_terminal,
        ),
        "initial_terminal_universe_ari": _adjusted_rand_score(
            universe_initial,
            universe_terminal,
        ),
        "component_reference_terminal_object_ari": _adjusted_rand_score(
            component_reference_object,
            object_terminal,
        ),
    }


def _safe_path_token(value: Any) -> str:
    token = "".join(
        char if char.isalnum() or char in {"-", "_", "."} else "_"
        for char in str(value)
    ).strip("._")
    return token or "value"


def _write_terminal_membership(
    *,
    output_dir: Path,
    object_role_id: str,
    start_index: int,
    start_policy: str,
    initial: np.ndarray,
    terminal: np.ndarray,
    object_mask: np.ndarray,
    universe_mask: np.ndarray,
    object_terminal_hash: str,
    universe_terminal_hash: str,
) -> str:
    membership_dir = output_dir / TERMINAL_MEMBERSHIP_DIRNAME
    membership_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        f"{_safe_path_token(object_role_id)}"
        f"__start{int(start_index):02d}_{_safe_path_token(start_policy)}"
        f"__obj{str(object_terminal_hash)[:8]}"
        f"__uni{str(universe_terminal_hash)[:8]}.npz"
    )
    path = membership_dir / filename
    object_node_ids = np.flatnonzero(object_mask).astype(np.uint32, copy=False)
    universe_node_ids = np.flatnonzero(universe_mask).astype(np.uint32, copy=False)
    np.savez_compressed(
        path,
        object_node_ids=object_node_ids,
        object_initial_labels=_compact_membership(np.asarray(initial)[object_mask]),
        object_terminal_labels=_compact_membership(np.asarray(terminal)[object_mask]),
        universe_node_ids=universe_node_ids,
        universe_initial_labels=_compact_membership(np.asarray(initial)[universe_mask]),
        universe_terminal_labels=_compact_membership(np.asarray(terminal)[universe_mask]),
    )
    return str(path.relative_to(output_dir))


def _load_terminal_membership(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _terminal_pair_rows(*, attempts: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    if attempts.empty or "terminal_membership_npz_path" not in attempts.columns:
        return pd.DataFrame(columns=TERMINAL_PAIR_COLUMNS)

    rows: list[dict[str, Any]] = []
    membership_cache: dict[str, dict[str, np.ndarray]] = {}
    for object_role_id, group in attempts.groupby("object_role_universe_id", sort=False):
        records = [
            record
            for record in group.sort_values("start_index", kind="mergesort").to_dict(
                "records"
            )
            if str(record.get("terminal_membership_npz_path", "")).strip()
            and str(record.get("terminal_membership_npz_path", "")).lower() != "nan"
        ]
        for left_index in range(len(records)):
            for right_index in range(left_index + 1, len(records)):
                left = records[left_index]
                right = records[right_index]
                left_rel = str(left["terminal_membership_npz_path"])
                right_rel = str(right["terminal_membership_npz_path"])
                left_path = output_dir / left_rel
                right_path = output_dir / right_rel
                if not left_path.exists() or not right_path.exists():
                    pair_status = "blocked_missing_terminal_membership_npz"
                    object_ari = None
                    universe_ari = None
                else:
                    if left_rel not in membership_cache:
                        membership_cache[left_rel] = _load_terminal_membership(left_path)
                    if right_rel not in membership_cache:
                        membership_cache[right_rel] = _load_terminal_membership(right_path)
                    left_membership = membership_cache[left_rel]
                    right_membership = membership_cache[right_rel]
                    object_aligned = np.array_equal(
                        left_membership["object_node_ids"],
                        right_membership["object_node_ids"],
                    )
                    universe_aligned = np.array_equal(
                        left_membership["universe_node_ids"],
                        right_membership["universe_node_ids"],
                    )
                    if not object_aligned or not universe_aligned:
                        pair_status = "blocked_misaligned_terminal_membership_nodes"
                        object_ari = None
                        universe_ari = None
                    else:
                        pair_status = "computed_terminal_membership_pair_ari"
                        object_ari = _adjusted_rand_score(
                            left_membership["object_terminal_labels"],
                            right_membership["object_terminal_labels"],
                        )
                        universe_ari = _adjusted_rand_score(
                            left_membership["universe_terminal_labels"],
                            right_membership["universe_terminal_labels"],
                        )
                left_quality = float(left["quality"])
                right_quality = float(right["quality"])
                rows.append(
                    {
                        "object_role_universe_id": object_role_id,
                        "left_start_id": left["start_id"],
                        "right_start_id": right["start_id"],
                        "left_start_policy": left["start_policy"],
                        "right_start_policy": right["start_policy"],
                        "left_start_index": int(left["start_index"]),
                        "right_start_index": int(right["start_index"]),
                        "left_object_terminal_hash": left["object_terminal_hash"],
                        "right_object_terminal_hash": right["object_terminal_hash"],
                        "left_universe_terminal_hash": left["universe_terminal_hash"],
                        "right_universe_terminal_hash": right["universe_terminal_hash"],
                        "same_object_terminal_hash": str(
                            left["object_terminal_hash"]
                        )
                        == str(right["object_terminal_hash"]),
                        "same_universe_terminal_hash": str(
                            left["universe_terminal_hash"]
                        )
                        == str(right["universe_terminal_hash"]),
                        "object_terminal_ari": object_ari,
                        "universe_terminal_ari": universe_ari,
                        "left_quality": left_quality,
                        "right_quality": right_quality,
                        "quality_delta_abs": abs(left_quality - right_quality),
                        "left_terminal_membership_npz_path": left_rel,
                        "right_terminal_membership_npz_path": right_rel,
                        "pair_status": pair_status,
                        "run_status": RUN_STATUS,
                        "claim_boundary": CLAIM_BOUNDARY,
                    }
                )
    return pd.DataFrame(rows, columns=TERMINAL_PAIR_COLUMNS)


def _object_summary_rows(attempts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if attempts.empty:
        return pd.DataFrame(rows)
    for object_role_id, group in attempts.groupby("object_role_universe_id", sort=False):
        terminal_hashes = sorted(set(group["object_terminal_hash"].astype(str)))
        universe_hashes = sorted(set(group["universe_terminal_hash"].astype(str)))
        rows.append(
            {
                "object_role_universe_id": object_role_id,
                "panel_case_id": group["panel_case_id"].iloc[0],
                "panel_case_rank": int(group["panel_case_rank"].iloc[0]),
                "analysis_tier": group["analysis_tier"].iloc[0],
                "strict_core_v0": bool(group["strict_core_v0"].iloc[0]),
                "role_id": group["role_id"].iloc[0],
                "role_side": group["role_side"].iloc[0],
                "branch": group["branch"].iloc[0],
                "symmetric_object_id": group["symmetric_object_id"].iloc[0],
                "probe_priority": group["probe_priority"].iloc[0],
                "symmetric_object_route_priority_rank": int(
                    group["symmetric_object_route_priority_rank"].iloc[0]
                ),
                "object_resolution_status": group["object_resolution_status"].iloc[0],
                "object_mask_hash": group["object_mask_hash"].iloc[0],
                "support_mask_hash": group["support_mask_hash"].iloc[0],
                "universe_mask_hash": group["universe_mask_hash"].iloc[0],
                "seed0_object_mask_hash": group["seed0_object_mask_hash"].iloc[0],
                "object_node_count": int(group["object_node_count"].iloc[0]),
                "support_node_count": int(group["support_node_count"].iloc[0]),
                "universe_node_count": int(group["universe_node_count"].iloc[0]),
                "seed0_object_node_count": int(group["seed0_object_node_count"].iloc[0]),
                "object_doc_sum": float(group["object_doc_sum"].iloc[0]),
                "support_doc_sum": float(group["support_doc_sum"].iloc[0]),
                "universe_doc_sum": float(group["universe_doc_sum"].iloc[0]),
                "seed0_object_doc_sum": float(group["seed0_object_doc_sum"].iloc[0]),
                "support_neighborhood_status": group["support_neighborhood_status"].iloc[0],
                "support_neighborhood_top_k": int(
                    group["support_neighborhood_top_k"].iloc[0]
                ),
                "support_edge_weight_sum": float(
                    group["support_edge_weight_sum"].iloc[0]
                ),
                "component_count": int(group["component_count"].iloc[0]),
                "seed_component_count": int(group["seed_component_count"].iloc[0]),
                "component_pattern_block_count": int(
                    group["component_pattern_block_count"].iloc[0]
                ),
                "start_attempt_count": int(len(group)),
                "executed_attempt_count": int(
                    group["execution_status"].astype(str).str.startswith("executed_").sum()
                ),
                "unique_terminal_object_hash_count": len(terminal_hashes),
                "terminal_multiplicity_detected": len(terminal_hashes) > 1,
                "terminal_object_hashes": ";".join(terminal_hashes),
                "unique_terminal_universe_hash_count": len(universe_hashes),
                "terminal_universe_multiplicity_detected": len(universe_hashes) > 1,
                "terminal_universe_hashes": ";".join(universe_hashes),
                "object_terminal_cluster_count_min": int(
                    group["object_terminal_cluster_count"].min()
                ),
                "object_terminal_cluster_count_median": float(
                    group["object_terminal_cluster_count"].median()
                ),
                "object_terminal_cluster_count_max": int(
                    group["object_terminal_cluster_count"].max()
                ),
                "terminal_singleton_all_starts": bool(
                    int(group["object_terminal_cluster_count"].min())
                    == int(group["object_node_count"].iloc[0])
                    and int(group["object_terminal_cluster_count"].max())
                    == int(group["object_node_count"].iloc[0])
                ),
                "universe_terminal_cluster_count_min": int(
                    group["universe_terminal_cluster_count"].min()
                ),
                "universe_terminal_cluster_count_median": float(
                    group["universe_terminal_cluster_count"].median()
                ),
                "universe_terminal_cluster_count_max": int(
                    group["universe_terminal_cluster_count"].max()
                ),
                "object_best_cluster_doc_share_min": float(
                    group["object_best_cluster_doc_share"].min()
                ),
                "object_best_cluster_doc_share_median": float(
                    group["object_best_cluster_doc_share"].median()
                ),
                "object_best_cluster_doc_share_max": float(
                    group["object_best_cluster_doc_share"].max()
                ),
                "universe_best_cluster_doc_share_min": float(
                    group["universe_best_cluster_doc_share"].min()
                ),
                "universe_best_cluster_doc_share_median": float(
                    group["universe_best_cluster_doc_share"].median()
                ),
                "universe_best_cluster_doc_share_max": float(
                    group["universe_best_cluster_doc_share"].max()
                ),
                "support_best_cluster_doc_share_min": float(
                    group["support_best_cluster_doc_share"].min()
                ),
                "support_best_cluster_doc_share_median": float(
                    group["support_best_cluster_doc_share"].median()
                ),
                "support_best_cluster_doc_share_max": float(
                    group["support_best_cluster_doc_share"].max()
                ),
                "seed0_best_cluster_doc_share_min": float(
                    group["seed0_best_cluster_doc_share"].min()
                ),
                "seed0_best_cluster_doc_share_median": float(
                    group["seed0_best_cluster_doc_share"].median()
                ),
                "seed0_best_cluster_doc_share_max": float(
                    group["seed0_best_cluster_doc_share"].max()
                ),
                "seed_component_best_cluster_doc_share_median_min": float(
                    group["seed_component_best_cluster_doc_share_median"].min()
                ),
                "seed_component_best_cluster_doc_share_median_median": float(
                    group["seed_component_best_cluster_doc_share_median"].median()
                ),
                "seed_component_best_cluster_doc_share_median_max": float(
                    group["seed_component_best_cluster_doc_share_median"].max()
                ),
                "component_reference_terminal_object_ari_min": float(
                    group["component_reference_terminal_object_ari"].min()
                ),
                "component_reference_terminal_object_ari_median": float(
                    group["component_reference_terminal_object_ari"].median()
                ),
                "component_reference_terminal_object_ari_max": float(
                    group["component_reference_terminal_object_ari"].max()
                ),
                "initial_terminal_object_ari_min": float(
                    group["initial_terminal_object_ari"].min()
                ),
                "initial_terminal_object_ari_median": float(
                    group["initial_terminal_object_ari"].median()
                ),
                "initial_terminal_object_ari_max": float(
                    group["initial_terminal_object_ari"].max()
                ),
                "initial_terminal_universe_ari_min": float(
                    group["initial_terminal_universe_ari"].min()
                ),
                "initial_terminal_universe_ari_median": float(
                    group["initial_terminal_universe_ari"].median()
                ),
                "initial_terminal_universe_ari_max": float(
                    group["initial_terminal_universe_ari"].max()
                ),
                "quality_min": float(group["quality"].min()),
                "quality_median": float(group["quality"].median()),
                "quality_max": float(group["quality"].max()),
                "seconds_sum": float(group["seconds"].sum()),
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    readiness_dir = Path(args.readiness_dir)
    landscape_dir = Path(args.landscape_dir)
    symmetric_object_dir = Path(args.symmetric_object_dir)
    object_universe_dir = Path(args.object_universe_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    graph_rows = _read_csv(readiness_dir / GRAPH_INPUT_ROWS_CSV)
    endpoint_registry = _read_csv(landscape_dir / ENDPOINT_REGISTRY_CSV)
    components = _read_csv(symmetric_object_dir / OBJECT_COMPONENTS_CSV)
    role_rows = _read_csv(object_universe_dir / SYMMETRIC_ROLE_ROWS_CSV)
    selected = _select_object_rows(
        role_rows,
        case_ranks=_parse_csv_list(args.case_ranks, int),
        role_sides=_parse_csv_list(args.role_sides, str),
        analysis_tiers=_parse_csv_list(args.analysis_tiers, str),
        object_status_prefixes=_parse_csv_list(args.object_status_prefixes, str),
        probe_priority_prefixes=_parse_csv_list(args.probe_priority_prefixes, str),
        strict_core_only=bool(args.strict_core_only),
        selection_policy=str(args.selection_policy),
        max_roles=int(args.max_roles),
        dedupe_symmetric_objects=bool(args.dedupe_symmetric_objects),
    )

    graph_by_branch = {
        str(row["branch"]): row
        for _, row in graph_rows.iterrows()
        if str(row.get("runtime_graph_status", "")).startswith("ready_")
    }
    membership_registry = _pure_seed_membership_registry(endpoint_registry)
    manifest_cache: dict[str, tuple[pd.DataFrame, np.ndarray]] = {}
    label_cache: dict[tuple[str, str], np.ndarray] = {}
    graph_cache: dict[str, tuple[Any, np.ndarray, float]] = {}
    edge_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    object_mask_cache: dict[tuple[str, str], dict[str, Any]] = {}
    attempt_rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    for row in selected.itertuples(index=False):
        branch = str(row.branch)
        object_id = str(row.symmetric_object_id)
        if branch not in graph_by_branch:
            raise ValueError(f"missing ready graph input for branch: {branch}")
        if branch not in graph_cache:
            graph, weights, load_seconds = _load_graph(
                graph_by_branch[branch],
                manifest_cache,
            )
            graph_cache[branch] = graph, weights, load_seconds
        graph, weights, graph_load_seconds = graph_cache[branch]
        n_nodes = int(graph.n_nodes)

        seed0_key = (branch, 0)
        if seed0_key not in membership_registry:
            raise ValueError(f"missing pure seed0 membership for branch: {branch}")
        seed0_path, label_col = membership_registry[seed0_key]
        initial_labels = _compact_membership(_load_label_array(seed0_path, label_col))

        object_key = (branch, object_id)
        if object_key not in object_mask_cache:
            object_mask_cache[object_key] = _load_object_masks(
                branch=branch,
                symmetric_object_id=object_id,
                components=components,
                membership_registry=membership_registry,
                label_cache=label_cache,
                n_nodes=n_nodes,
                weights=weights,
            )
        masks = object_mask_cache[object_key]
        object_mask = masks["object_mask"]
        seed0_mask = masks["seed0_mask"]
        seed_masks = masks["seed_masks"]
        component_masks = masks["component_masks"]
        support_stats: dict[str, Any]
        if int(args.support_neighborhood_top_k) > 0:
            if branch not in edge_cache:
                edge_cache[branch] = _load_branch_edge_sidecars(graph_by_branch[branch])
            edge_src, edge_dst, edge_weight = edge_cache[branch]
            support_stats = _support_neighborhood_mask(
                object_mask=object_mask,
                edge_src=edge_src,
                edge_dst=edge_dst,
                edge_weight=edge_weight,
                top_k=int(args.support_neighborhood_top_k),
                min_weight=float(args.support_neighborhood_min_weight),
            )
        else:
            support_stats = _support_neighborhood_mask(
                object_mask=object_mask,
                edge_src=np.asarray([], dtype=np.uint32),
                edge_dst=np.asarray([], dtype=np.uint32),
                edge_weight=np.asarray([], dtype=np.float64),
                top_k=0,
                min_weight=0.0,
            )
        support_mask = support_stats["support_mask"]
        universe_mask = np.logical_or(object_mask, support_mask)
        fixed_nodes = ~universe_mask

        component_initial, pattern_count, unassigned_count = _component_pattern_membership(
            initial_labels=initial_labels,
            object_mask=object_mask,
            component_masks=component_masks,
        )
        method_seed = int(args.method_seed)
        specs = _object_start_specs(
            initial_labels=initial_labels,
            object_mask=object_mask,
            seed0_mask=seed0_mask,
            component_pattern_membership=component_initial,
            method_seed=method_seed,
            random_start_count=int(args.random_start_count),
            random_block_count=int(args.random_block_count),
        )
        object_role_id = f"{row.role_id}__{object_id}"
        for spec in specs:
            terminal, meta = _run_leiden_or_hold(
                graph=graph,
                membership=spec["membership"],
                fixed_nodes=fixed_nodes,
                resolution=float(args.gamma),
                seed=int(spec["leiden_seed"]),
                n_iterations=int(args.n_iterations),
                blocked_status="blocked_empty_symmetric_object_free_mask",
                executed_status="executed_symmetric_object_multistart",
            )
            score = _score_object(
                terminal=terminal,
                initial=spec["membership"],
                component_reference=component_initial,
                object_mask=object_mask,
                universe_mask=universe_mask,
                support_mask=support_mask,
                seed0_mask=seed0_mask,
                seed_masks=seed_masks,
                weights=weights,
            )
            terminal_membership_npz_path = ""
            if bool(args.save_terminal_memberships):
                terminal_membership_npz_path = _write_terminal_membership(
                    output_dir=output_dir,
                    object_role_id=object_role_id,
                    start_index=int(spec["start_index"]),
                    start_policy=str(spec["start_policy"]),
                    initial=spec["membership"],
                    terminal=terminal,
                    object_mask=object_mask,
                    universe_mask=universe_mask,
                    object_terminal_hash=str(score["object_terminal_hash"]),
                    universe_terminal_hash=str(score["universe_terminal_hash"]),
                )
            attempt_rows.append(
                {
                    "object_role_universe_id": object_role_id,
                    "start_id": f"{object_role_id}__{spec['start_policy']}",
                    "panel_case_id": row.panel_case_id,
                    "panel_case_rank": int(row.panel_case_rank),
                    "analysis_tier": row.analysis_tier,
                    "strict_core_v0": bool(row.strict_core_v0),
                    "role_id": row.role_id,
                    "role_side": row.role_side,
                    "primitive_id": row.primitive_id,
                    "branch": branch,
                    "symmetric_object_id": object_id,
                    "probe_priority": row.probe_priority,
                    "symmetric_object_route_priority_rank": int(
                        row.symmetric_object_route_priority_rank
                    ),
                    "object_resolution_status": row.object_resolution_status,
                    "method_seed": method_seed,
                    "start_policy": spec["start_policy"],
                    "start_index": int(spec["start_index"]),
                    "leiden_seed": int(spec["leiden_seed"]),
                    "random_block_count": int(args.random_block_count),
                    "n_nodes": n_nodes,
                    "n_edges": int(graph.n_edges),
                    "fixed_outside_node_count": int(fixed_nodes.sum()),
                    "support_neighborhood_status": support_stats[
                        "support_neighborhood_status"
                    ],
                    "support_neighborhood_top_k": int(
                        support_stats["support_neighborhood_top_k"]
                    ),
                    "support_neighborhood_min_weight": float(
                        support_stats["support_neighborhood_min_weight"]
                    ),
                    "support_node_count": int(support_stats["support_node_count"]),
                    "support_edge_weight_sum": float(
                        support_stats["support_edge_weight_sum"]
                    ),
                    "support_edge_weight_min": float(
                        support_stats["support_edge_weight_min"]
                    ),
                    "support_edge_weight_median": support_stats[
                        "support_edge_weight_median"
                    ],
                    "support_edge_weight_max": float(
                        support_stats["support_edge_weight_max"]
                    ),
                    "support_node_ids": support_stats["support_node_ids"],
                    "universe_node_count_input": int(universe_mask.sum()),
                    "component_count": len(component_masks),
                    "component_resolution_status_counts": masks[
                        "component_resolution_status_counts"
                    ],
                    "component_pattern_block_count": int(pattern_count),
                    "component_pattern_unassigned_node_count": int(unassigned_count),
                    "seed_list": ";".join(str(seed) for seed in sorted(seed_masks)),
                    "graph_load_seconds_cached_branch": float(graph_load_seconds),
                    **meta,
                    **score,
                    "terminal_membership_npz_path": terminal_membership_npz_path,
                    "run_status": RUN_STATUS,
                    "claim_boundary": CLAIM_BOUNDARY,
                }
            )
        attempts = pd.DataFrame(attempt_rows)
        _write_csv(attempts, output_dir / OBJECT_ATTEMPT_ROWS_CSV)
        _write_csv(_object_summary_rows(attempts), output_dir / OBJECT_SUMMARY_ROWS_CSV)
        if bool(args.save_terminal_memberships):
            _write_csv(
                _terminal_pair_rows(attempts=attempts, output_dir=output_dir),
                output_dir / TERMINAL_PAIR_ROWS_CSV,
            )

    attempts = pd.DataFrame(attempt_rows)
    object_rows = _object_summary_rows(attempts)
    terminal_pair_rows = (
        _terminal_pair_rows(attempts=attempts, output_dir=output_dir)
        if bool(args.save_terminal_memberships)
        else pd.DataFrame(columns=TERMINAL_PAIR_COLUMNS)
    )
    _write_csv(attempts, output_dir / OBJECT_ATTEMPT_ROWS_CSV)
    _write_csv(object_rows, output_dir / OBJECT_SUMMARY_ROWS_CSV)
    if bool(args.save_terminal_memberships):
        _write_csv(terminal_pair_rows, output_dir / TERMINAL_PAIR_ROWS_CSV)
    summary = _build_summary(
        selected=selected,
        attempts=attempts,
        object_rows=object_rows,
        terminal_pair_rows=terminal_pair_rows,
        readiness_dir=readiness_dir,
        object_universe_dir=object_universe_dir,
        output_dir=output_dir,
        elapsed_seconds=time.perf_counter() - started,
    )
    (output_dir / OBJECT_SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    config = {
        "schema": "nanoclustering_symmetric_object_multistart_pilot.v1",
        "readiness_dir": str(readiness_dir),
        "landscape_dir": str(landscape_dir),
        "symmetric_object_dir": str(symmetric_object_dir),
        "object_universe_dir": str(object_universe_dir),
        "output_dir": str(output_dir),
        "case_ranks": list(_parse_csv_list(args.case_ranks, int)),
        "role_sides": list(_parse_csv_list(args.role_sides, str)),
        "analysis_tiers": list(_parse_csv_list(args.analysis_tiers, str)),
        "object_status_prefixes": list(
            _parse_csv_list(args.object_status_prefixes, str)
        ),
        "probe_priority_prefixes": list(
            _parse_csv_list(args.probe_priority_prefixes, str)
        ),
        "strict_core_only": bool(args.strict_core_only),
        "selection_policy": str(args.selection_policy),
        "max_roles": int(args.max_roles),
        "dedupe_symmetric_objects": bool(args.dedupe_symmetric_objects),
        "method_seed": int(args.method_seed),
        "random_start_count": int(args.random_start_count),
        "random_block_count": int(args.random_block_count),
        "support_neighborhood_top_k": int(args.support_neighborhood_top_k),
        "support_neighborhood_min_weight": float(args.support_neighborhood_min_weight),
        "gamma": float(args.gamma),
        "n_iterations": int(args.n_iterations),
        "save_terminal_memberships": bool(args.save_terminal_memberships),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / OBJECT_CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(output_dir=output_dir, summary=summary, object_rows=object_rows)
    return summary


def _build_summary(
    *,
    selected: pd.DataFrame,
    attempts: pd.DataFrame,
    object_rows: pd.DataFrame,
    terminal_pair_rows: pd.DataFrame,
    readiness_dir: Path,
    object_universe_dir: Path,
    output_dir: Path,
    elapsed_seconds: float,
) -> dict[str, Any]:
    if object_rows.empty:
        multiplicity_count = 0
        max_terminal_count = 0
        total_route_seconds = 0.0
        singleton_count = 0
        universe_multiplicity_count = 0
    else:
        multiplicity_count = int(object_rows["terminal_multiplicity_detected"].sum())
        max_terminal_count = int(object_rows["unique_terminal_object_hash_count"].max())
        total_route_seconds = float(object_rows["seconds_sum"].sum())
        singleton_count = int(object_rows["terminal_singleton_all_starts"].sum())
        universe_multiplicity_count = int(
            object_rows["terminal_universe_multiplicity_detected"].sum()
        )
    if terminal_pair_rows.empty:
        terminal_pair_row_count = 0
        same_object_hash_pair_count = 0
        same_universe_hash_pair_count = 0
        object_pair_ari_min = None
        object_pair_ari_median = None
        object_pair_ari_max = None
        universe_pair_ari_min = None
        universe_pair_ari_median = None
        universe_pair_ari_max = None
    else:
        terminal_pair_row_count = int(len(terminal_pair_rows))
        same_object_hash_pair_count = int(
            terminal_pair_rows["same_object_terminal_hash"].sum()
        )
        same_universe_hash_pair_count = int(
            terminal_pair_rows["same_universe_terminal_hash"].sum()
        )
        object_pair_ari = pd.to_numeric(
            terminal_pair_rows["object_terminal_ari"],
            errors="coerce",
        ).dropna()
        universe_pair_ari = pd.to_numeric(
            terminal_pair_rows["universe_terminal_ari"],
            errors="coerce",
        ).dropna()
        object_pair_ari_min = (
            float(object_pair_ari.min()) if not object_pair_ari.empty else None
        )
        object_pair_ari_median = (
            float(object_pair_ari.median()) if not object_pair_ari.empty else None
        )
        object_pair_ari_max = (
            float(object_pair_ari.max()) if not object_pair_ari.empty else None
        )
        universe_pair_ari_min = (
            float(universe_pair_ari.min()) if not universe_pair_ari.empty else None
        )
        universe_pair_ari_median = (
            float(universe_pair_ari.median()) if not universe_pair_ari.empty else None
        )
        universe_pair_ari_max = (
            float(universe_pair_ari.max()) if not universe_pair_ari.empty else None
        )
    return {
        "schema": "nanoclustering_symmetric_object_multistart_pilot_summary.v1",
        "status": RUN_STATUS if not object_rows.empty else "no_symmetric_object_roles",
        "readiness_dir": str(readiness_dir),
        "object_universe_dir": str(object_universe_dir),
        "output_dir": str(output_dir),
        "selected_role_count": int(len(selected)),
        "object_role_universe_count": int(len(object_rows)),
        "unique_symmetric_object_count": (
            int(object_rows["symmetric_object_id"].nunique())
            if not object_rows.empty
            else 0
        ),
        "start_attempt_count": int(len(attempts)),
        "terminal_multiplicity_object_role_count": multiplicity_count,
        "terminal_multiplicity_object_role_share": (
            float(multiplicity_count / len(object_rows)) if len(object_rows) else None
        ),
        "max_unique_terminal_object_hash_count": max_terminal_count,
        "terminal_singleton_object_role_count": singleton_count,
        "terminal_singleton_object_role_share": (
            float(singleton_count / len(object_rows)) if len(object_rows) else None
        ),
        "terminal_universe_multiplicity_object_role_count": universe_multiplicity_count,
        "terminal_universe_multiplicity_object_role_share": (
            float(universe_multiplicity_count / len(object_rows))
            if len(object_rows)
            else None
        ),
        "terminal_pair_row_count": terminal_pair_row_count,
        "terminal_pair_same_object_hash_pair_count": same_object_hash_pair_count,
        "terminal_pair_same_object_hash_pair_share": (
            float(same_object_hash_pair_count / terminal_pair_row_count)
            if terminal_pair_row_count
            else None
        ),
        "terminal_pair_same_universe_hash_pair_count": same_universe_hash_pair_count,
        "terminal_pair_same_universe_hash_pair_share": (
            float(same_universe_hash_pair_count / terminal_pair_row_count)
            if terminal_pair_row_count
            else None
        ),
        "terminal_pair_object_ari_min": object_pair_ari_min,
        "terminal_pair_object_ari_median": object_pair_ari_median,
        "terminal_pair_object_ari_max": object_pair_ari_max,
        "terminal_pair_universe_ari_min": universe_pair_ari_min,
        "terminal_pair_universe_ari_median": universe_pair_ari_median,
        "terminal_pair_universe_ari_max": universe_pair_ari_max,
        "unique_panel_case_count": (
            int(object_rows["panel_case_id"].nunique()) if not object_rows.empty else 0
        ),
        "branch_count": int(object_rows["branch"].nunique()) if not object_rows.empty else 0,
        "role_side_count": (
            int(object_rows["role_side"].nunique()) if not object_rows.empty else 0
        ),
        "object_node_count_median": (
            float(object_rows["object_node_count"].median()) if not object_rows.empty else None
        ),
        "support_node_count_median": (
            float(object_rows["support_node_count"].median())
            if not object_rows.empty
            else None
        ),
        "universe_node_count_median": (
            float(object_rows["universe_node_count"].median())
            if not object_rows.empty
            else None
        ),
        "object_node_count_max": (
            int(object_rows["object_node_count"].max()) if not object_rows.empty else 0
        ),
        "universe_node_count_max": (
            int(object_rows["universe_node_count"].max()) if not object_rows.empty else 0
        ),
        "object_terminal_cluster_count_median": (
            float(object_rows["object_terminal_cluster_count_median"].median())
            if not object_rows.empty
            else None
        ),
        "seed0_object_node_count_median": (
            float(object_rows["seed0_object_node_count"].median())
            if not object_rows.empty
            else None
        ),
        "object_best_cluster_doc_share_median": (
            float(object_rows["object_best_cluster_doc_share_median"].median())
            if not object_rows.empty
            else None
        ),
        "universe_best_cluster_doc_share_median": (
            float(object_rows["universe_best_cluster_doc_share_median"].median())
            if not object_rows.empty
            else None
        ),
        "support_best_cluster_doc_share_median": (
            float(object_rows["support_best_cluster_doc_share_median"].median())
            if not object_rows.empty
            else None
        ),
        "seed0_best_cluster_doc_share_median": (
            float(object_rows["seed0_best_cluster_doc_share_median"].median())
            if not object_rows.empty
            else None
        ),
        "component_reference_terminal_object_ari_median": (
            float(object_rows["component_reference_terminal_object_ari_median"].median())
            if not object_rows.empty
            else None
        ),
        "initial_terminal_object_ari_median": (
            float(object_rows["initial_terminal_object_ari_median"].median())
            if not object_rows.empty
            else None
        ),
        "initial_terminal_universe_ari_median": (
            float(object_rows["initial_terminal_universe_ari_median"].median())
            if not object_rows.empty
            else None
        ),
        "total_route_seconds": total_route_seconds,
        "elapsed_seconds": float(elapsed_seconds),
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _write_report(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    object_rows: pd.DataFrame,
) -> None:
    lines = [
        "# NanoClustering Symmetric-Object Multistart Pilot",
        "",
        f"- status: `{summary['status']}`",
        f"- object_role_universe_count: {summary['object_role_universe_count']}",
        f"- unique_symmetric_object_count: {summary['unique_symmetric_object_count']}",
        f"- start_attempt_count: {summary['start_attempt_count']}",
        f"- terminal_multiplicity_object_role_count: {summary['terminal_multiplicity_object_role_count']}",
        f"- terminal_multiplicity_object_role_share: {summary['terminal_multiplicity_object_role_share']}",
        f"- max_unique_terminal_object_hash_count: {summary['max_unique_terminal_object_hash_count']}",
        f"- terminal_singleton_object_role_count: {summary['terminal_singleton_object_role_count']}",
        f"- terminal_singleton_object_role_share: {summary['terminal_singleton_object_role_share']}",
        f"- terminal_universe_multiplicity_object_role_count: {summary['terminal_universe_multiplicity_object_role_count']}",
        f"- terminal_universe_multiplicity_object_role_share: {summary['terminal_universe_multiplicity_object_role_share']}",
        f"- terminal_pair_row_count: {summary['terminal_pair_row_count']}",
        f"- terminal_pair_same_object_hash_pair_share: {summary['terminal_pair_same_object_hash_pair_share']}",
        f"- terminal_pair_object_ari_median: {summary['terminal_pair_object_ari_median']}",
        f"- terminal_pair_universe_ari_median: {summary['terminal_pair_universe_ari_median']}",
        f"- object_node_count_median: {summary['object_node_count_median']}",
        f"- support_node_count_median: {summary['support_node_count_median']}",
        f"- universe_node_count_median: {summary['universe_node_count_median']}",
        f"- object_node_count_max: {summary['object_node_count_max']}",
        f"- universe_node_count_max: {summary['universe_node_count_max']}",
        f"- object_terminal_cluster_count_median: {summary['object_terminal_cluster_count_median']}",
        f"- component_reference_terminal_object_ari_median: {summary['component_reference_terminal_object_ari_median']}",
        f"- total_route_seconds: {summary['total_route_seconds']}",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Objects",
    ]
    if object_rows.empty:
        lines.append("- no symmetric-object roles")
    else:
        for row in object_rows.sort_values(
            ["terminal_multiplicity_detected", "object_node_count"],
            ascending=[False, False],
        ).itertuples(index=False):
            data = row._asdict()
            lines.append(
                "- "
                f"{data['object_role_universe_id']}: "
                f"unique_terminals={data['unique_terminal_object_hash_count']}, "
                f"multiplicity={data['terminal_multiplicity_detected']}, "
                f"object_nodes={data['object_node_count']}, "
                f"support_nodes={data['support_node_count']}, "
                f"universe_nodes={data['universe_node_count']}, "
                f"terminal_clusters={data['object_terminal_cluster_count_median']}, "
                f"singleton={data['terminal_singleton_all_starts']}, "
                f"seed_components={data['seed_component_count']}, "
                f"object_share_median={data['object_best_cluster_doc_share_median']}, "
                f"component_ari_median={data['component_reference_terminal_object_ari_median']}, "
                f"quality_range=[{data['quality_min']}, {data['quality_max']}]"
            )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            (
                "Distinct terminals under the same symmetric-object universe are "
                "terminal-multiplicity evidence only. They are not yet wall, pathway, "
                "quality, cost, method-success, or algorithm evidence."
            ),
            "",
        ]
    )
    (output_dir / OBJECT_REPORT_MD).write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness-dir", type=Path, default=DEFAULT_READINESS_DIR)
    parser.add_argument("--landscape-dir", type=Path, default=DEFAULT_LANDSCAPE_DIR)
    parser.add_argument(
        "--symmetric-object-dir",
        type=Path,
        default=DEFAULT_SYMMETRIC_OBJECT_DIR,
    )
    parser.add_argument(
        "--object-universe-dir",
        type=Path,
        default=DEFAULT_OBJECT_UNIVERSE_DIR,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--case-ranks", default="")
    parser.add_argument("--role-sides", default="")
    parser.add_argument("--analysis-tiers", default="strict_core_v0_primary")
    parser.add_argument("--object-status-prefixes", default="ready_anchor_independent")
    parser.add_argument("--probe-priority-prefixes", default="P1_")
    parser.add_argument(
        "--strict-core-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--selection-policy",
        choices=sorted(SELECTION_POLICIES),
        default="route_priority",
    )
    parser.add_argument("--max-roles", type=int, default=6)
    parser.add_argument(
        "--dedupe-symmetric-objects",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--method-seed", type=int, default=0)
    parser.add_argument("--random-start-count", type=int, default=4)
    parser.add_argument("--random-block-count", type=int, default=8)
    parser.add_argument("--support-neighborhood-top-k", type=int, default=0)
    parser.add_argument("--support-neighborhood-min-weight", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.7)
    parser.add_argument("--n-iterations", type=int, default=2)
    parser.add_argument(
        "--save-terminal-memberships",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
