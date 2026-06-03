#!/usr/bin/env python3
"""Analyze terminal membership differences from a symmetric-object multistart run.

This reads compact object/universe terminal membership slices emitted by
``run_leiden_basin_nanoclustering_symmetric_object_multistart_pilot.py
--save-terminal-memberships`` and decomposes terminal multiplicity into:

- start-policy terminal hash groups;
- pairwise co-assignment differences between starts;
- object/support nodes that sit at the center of those differences.

It is a terminal-structure diagnostic only. It does not promote wall/pathway,
quality/cost, real-data method-success, or algorithm claims.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from run_leiden_basin_nanoclustering_symmetric_object_multistart_pilot import (
    CLAIM_BOUNDARY as MULTISTART_CLAIM_BOUNDARY,
    OBJECT_ATTEMPT_ROWS_CSV,
    TERMINAL_PAIR_ROWS_CSV,
    _adjusted_rand_score,
)
from run_leiden_basin_nanoclustering_role_local_route_pilot import (
    BASE_RESULT_DIR,
    GRAPH_INPUT_ROWS_CSV,
    _json_safe,
    _read_csv,
    _write_csv,
)


DEFAULT_MULTISTART_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_symmetric_object_multistart_support_top100_gamma1e5_membership_p1_unique_20260602"
)
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_symmetric_object_terminal_membership_difference_review_gamma1e5_20260603"
)

OBJECT_ROWS_CSV = "nanoclustering_symmetric_object_terminal_difference_object_rows.csv"
PAIR_ROWS_CSV = "nanoclustering_symmetric_object_terminal_difference_pair_rows.csv"
NODE_ROWS_CSV = "nanoclustering_symmetric_object_terminal_difference_node_rows.csv"
NODE_PAIR_ROWS_CSV = (
    "nanoclustering_symmetric_object_terminal_difference_variable_node_pair_rows.csv"
)
GROUP_ROWS_CSV = "nanoclustering_symmetric_object_terminal_difference_group_rows.csv"
SUMMARY_JSON = "nanoclustering_symmetric_object_terminal_difference_summary.json"
REPORT_MD = "nanoclustering_symmetric_object_terminal_difference_report.md"
CONFIG_JSON = "nanoclustering_symmetric_object_terminal_difference_config.json"

RUN_STATUS = "executed_symmetric_object_terminal_membership_difference_review"
CLAIM_BOUNDARY = (
    "NanoClustering symmetric-object terminal membership difference review only; "
    "decomposes saved compact terminal slices into hash groups, co-assignment "
    "variation, and variable-node summaries. It does not promote wall/pathway, "
    "basin-quality, cost, real-data method-success, or algorithm claims."
)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _load_doc_weights(multistart_dir: Path, attempts: pd.DataFrame) -> dict[str, np.ndarray]:
    config_path = multistart_dir / "nanoclustering_symmetric_object_multistart_config.json"
    if not config_path.exists():
        return {}
    config = json.loads(config_path.read_text(encoding="utf-8"))
    readiness_dir = Path(str(config["readiness_dir"]))
    graph_rows = _read_csv(readiness_dir / GRAPH_INPUT_ROWS_CSV)
    out: dict[str, np.ndarray] = {}
    for branch in sorted(set(attempts["branch"].astype(str))):
        rows = graph_rows[graph_rows["branch"].astype(str).eq(branch)]
        if rows.empty:
            continue
        manifest_path = Path(str(rows.iloc[0]["runtime_node_manifest_path"]))
        frame = pd.read_parquet(
            manifest_path,
            columns=["node_idx", "original_cluster_id", "doc_count"],
        ).sort_values("node_idx", kind="mergesort")
        expected = np.arange(len(frame), dtype=np.int64)
        if not np.array_equal(frame["node_idx"].to_numpy(dtype=np.int64), expected):
            raise ValueError(f"node_idx is not dense and sorted in {manifest_path}")
        out[branch] = frame["doc_count"].to_numpy(dtype=np.float64)
    return out


def _same_cluster_cube(labels_by_start: np.ndarray) -> np.ndarray:
    return labels_by_start[:, :, None] == labels_by_start[:, None, :]


def _coassignment_stats(labels_by_start: np.ndarray) -> dict[str, Any]:
    n_starts, n_nodes = labels_by_start.shape
    if n_nodes < 2 or n_starts < 2:
        return {
            "coassignment_pair_count": 0,
            "variable_pair_count": 0,
            "variable_pair_share": None,
            "stable_same_pair_count": 0,
            "stable_apart_pair_count": 0,
            "variable_partner_count_by_node": np.zeros(n_nodes, dtype=np.int64),
        }
    counts = _same_cluster_cube(labels_by_start).sum(axis=0)
    tri = np.triu(np.ones((n_nodes, n_nodes), dtype=np.bool_), k=1)
    variable = (counts > 0) & (counts < n_starts) & tri
    stable_same = (counts == n_starts) & tri
    stable_apart = (counts == 0) & tri
    variable_full = ((counts > 0) & (counts < n_starts)).copy()
    np.fill_diagonal(variable_full, False)
    pair_count = int(tri.sum())
    variable_count = int(variable.sum())
    return {
        "coassignment_pair_count": pair_count,
        "variable_pair_count": variable_count,
        "variable_pair_share": float(variable_count / pair_count) if pair_count else None,
        "stable_same_pair_count": int(stable_same.sum()),
        "stable_apart_pair_count": int(stable_apart.sum()),
        "variable_partner_count_by_node": variable_full.sum(axis=1).astype(np.int64),
    }


def _node_cluster_stats(
    *,
    labels_by_start: np.ndarray,
    doc_weights: np.ndarray,
) -> dict[str, np.ndarray]:
    n_starts, n_nodes = labels_by_start.shape
    cluster_sizes = np.zeros((n_starts, n_nodes), dtype=np.int64)
    cluster_docs = np.zeros((n_starts, n_nodes), dtype=np.float64)
    for start_idx, labels in enumerate(labels_by_start):
        series = pd.Series(labels)
        counts = series.map(series.value_counts()).to_numpy(dtype=np.int64)
        doc_sum_by_label: dict[int, float] = {}
        for label in np.unique(labels):
            doc_sum_by_label[int(label)] = float(doc_weights[labels == label].sum())
        docs = np.asarray([doc_sum_by_label[int(label)] for label in labels], dtype=np.float64)
        cluster_sizes[start_idx] = counts
        cluster_docs[start_idx] = docs
    return {
        "cluster_size_min": cluster_sizes.min(axis=0),
        "cluster_size_median": np.median(cluster_sizes, axis=0),
        "cluster_size_max": cluster_sizes.max(axis=0),
        "cluster_doc_sum_min": cluster_docs.min(axis=0),
        "cluster_doc_sum_median": np.median(cluster_docs, axis=0),
        "cluster_doc_sum_max": cluster_docs.max(axis=0),
        "singleton_start_count": (cluster_sizes == 1).sum(axis=0),
    }


def _terminal_group_rows(group: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for terminal_hash, terminal_group in group.groupby("object_terminal_hash", sort=False):
        policies = sorted(terminal_group["start_policy"].astype(str))
        rows.append(
            {
                "object_role_universe_id": group["object_role_universe_id"].iloc[0],
                "object_terminal_hash": terminal_hash,
                "universe_terminal_hashes": ";".join(
                    sorted(set(terminal_group["universe_terminal_hash"].astype(str)))
                ),
                "start_policy_count": int(len(terminal_group)),
                "start_policies": ";".join(policies),
                "start_indices": ";".join(
                    str(int(value)) for value in sorted(terminal_group["start_index"])
                ),
                "quality_min": float(terminal_group["quality"].min()),
                "quality_median": float(terminal_group["quality"].median()),
                "quality_max": float(terminal_group["quality"].max()),
                "object_terminal_cluster_count_median": float(
                    terminal_group["object_terminal_cluster_count"].median()
                ),
                "universe_terminal_cluster_count_median": float(
                    terminal_group["universe_terminal_cluster_count"].median()
                ),
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return rows


def _policy_split_pattern(group_rows: list[dict[str, Any]]) -> str:
    policy_sets = {
        frozenset(str(row["start_policies"]).split(";")): int(row["start_policy_count"])
        for row in group_rows
    }
    if len(policy_sets) <= 1:
        return "closed_single_terminal_group"
    if frozenset({"object_seed_component_blocks"}) in policy_sets and len(policy_sets) == 2:
        return "component_pattern_alternate_terminal"
    if (
        frozenset({"seed0_source_state", "seed0_object_seeded"}) in policy_sets
        and frozenset({"object_singleton"}) in policy_sets
        and frozenset({"object_seed_component_blocks"}) in policy_sets
    ):
        return "source_seeded_singleton_component_three_way_split"
    if (
        frozenset({"seed0_object_seeded", "object_singleton"}) in policy_sets
        and frozenset({"seed0_source_state"}) in policy_sets
        and frozenset({"object_seed_component_blocks"}) in policy_sets
    ):
        return "seeded_singleton_source_component_three_way_split"
    return "mixed_start_policy_terminal_split"


def _mechanism_read(group_rows: list[dict[str, Any]], object_stats: dict[str, Any]) -> str:
    pattern = str(object_stats["policy_split_pattern"])
    variable_share = object_stats["object_variable_pair_share"]
    if object_stats["terminal_group_count"] <= 1:
        return "closed_control_no_terminal_multiplicity"
    if pattern == "component_pattern_alternate_terminal":
        return "component_pattern_initialization_selects_alternate_partial_coarsening"
    if pattern in {
        "source_seeded_singleton_component_three_way_split",
        "seeded_singleton_source_component_three_way_split",
    }:
        return "start_condition_selects_multiple_partial_coarsening_terminals"
    if variable_share is not None and float(variable_share) < 0.1:
        return "high_overlap_low_variable_pair_terminal_variants"
    return "mixed_terminal_variants_need_node_level_mechanism_review"


def _pair_difference_rows(
    *,
    object_role_id: str,
    records: list[dict[str, Any]],
    memberships: list[dict[str, np.ndarray]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for left_index in range(len(records)):
        for right_index in range(left_index + 1, len(records)):
            left = records[left_index]
            right = records[right_index]
            left_obj = memberships[left_index]["object_terminal_labels"]
            right_obj = memberships[right_index]["object_terminal_labels"]
            left_uni = memberships[left_index]["universe_terminal_labels"]
            right_uni = memberships[right_index]["universe_terminal_labels"]
            same_obj = left_obj[:, None] == left_obj[None, :]
            same_obj_right = right_obj[:, None] == right_obj[None, :]
            same_uni = left_uni[:, None] == left_uni[None, :]
            same_uni_right = right_uni[:, None] == right_uni[None, :]
            obj_tri = np.triu(np.ones(same_obj.shape, dtype=np.bool_), k=1)
            uni_tri = np.triu(np.ones(same_uni.shape, dtype=np.bool_), k=1)
            obj_changed = np.logical_xor(same_obj, same_obj_right) & obj_tri
            uni_changed = np.logical_xor(same_uni, same_uni_right) & uni_tri
            rows.append(
                {
                    "object_role_universe_id": object_role_id,
                    "left_start_policy": left["start_policy"],
                    "right_start_policy": right["start_policy"],
                    "left_start_index": int(left["start_index"]),
                    "right_start_index": int(right["start_index"]),
                    "same_object_terminal_hash": str(left["object_terminal_hash"])
                    == str(right["object_terminal_hash"]),
                    "same_universe_terminal_hash": str(left["universe_terminal_hash"])
                    == str(right["universe_terminal_hash"]),
                    "object_terminal_ari": _adjusted_rand_score(left_obj, right_obj),
                    "universe_terminal_ari": _adjusted_rand_score(left_uni, right_uni),
                    "object_changed_coassignment_pair_count": int(obj_changed.sum()),
                    "object_changed_coassignment_pair_share": float(
                        obj_changed.sum() / obj_tri.sum()
                    )
                    if int(obj_tri.sum())
                    else None,
                    "universe_changed_coassignment_pair_count": int(uni_changed.sum()),
                    "universe_changed_coassignment_pair_share": float(
                        uni_changed.sum() / uni_tri.sum()
                    )
                    if int(uni_tri.sum())
                    else None,
                    "quality_delta_abs": abs(float(left["quality"]) - float(right["quality"])),
                    "run_status": RUN_STATUS,
                    "claim_boundary": CLAIM_BOUNDARY,
                }
            )
    return rows


def _variable_node_pair_rows(
    *,
    object_role_id: str,
    records: list[dict[str, Any]],
    universe_node_ids: np.ndarray,
    universe_labels_by_start: np.ndarray,
    universe_doc_weights: np.ndarray,
    object_id_set: set[int],
) -> list[dict[str, Any]]:
    n_starts, n_nodes = universe_labels_by_start.shape
    if n_starts < 2 or n_nodes < 2:
        return []
    counts = _same_cluster_cube(universe_labels_by_start).sum(axis=0)
    variable = (counts > 0) & (counts < n_starts)
    rows: list[dict[str, Any]] = []
    for left_idx, right_idx in zip(*np.triu_indices(n_nodes, k=1), strict=True):
        if not bool(variable[left_idx, right_idx]):
            continue
        left_node = int(universe_node_ids[left_idx])
        right_node = int(universe_node_ids[right_idx])
        left_scope = "object" if left_node in object_id_set else "support"
        right_scope = "object" if right_node in object_id_set else "support"
        scopes = sorted([left_scope, right_scope])
        if scopes == ["object", "object"]:
            pair_scope = "object_object"
        elif scopes == ["support", "support"]:
            pair_scope = "support_support"
        else:
            pair_scope = "object_support"
        together_policies: list[str] = []
        apart_policies: list[str] = []
        for start_idx, record in enumerate(records):
            policy = str(record["start_policy"])
            if (
                universe_labels_by_start[start_idx, left_idx]
                == universe_labels_by_start[start_idx, right_idx]
            ):
                together_policies.append(policy)
            else:
                apart_policies.append(policy)
        rows.append(
            {
                "object_role_universe_id": object_role_id,
                "left_node_id": left_node,
                "right_node_id": right_node,
                "left_node_scope": left_scope,
                "right_node_scope": right_scope,
                "pair_scope": pair_scope,
                "left_doc_count": float(universe_doc_weights[left_idx]),
                "right_doc_count": float(universe_doc_weights[right_idx]),
                "pair_doc_count_sum": float(
                    universe_doc_weights[left_idx] + universe_doc_weights[right_idx]
                ),
                "same_terminal_start_count": int(counts[left_idx, right_idx]),
                "apart_terminal_start_count": int(n_starts - counts[left_idx, right_idx]),
                "together_start_policies": ";".join(together_policies),
                "apart_start_policies": ";".join(apart_policies),
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return rows


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    multistart_dir = Path(args.multistart_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    attempts = _read_csv(multistart_dir / OBJECT_ATTEMPT_ROWS_CSV)
    prior_pair_rows = _read_csv(multistart_dir / TERMINAL_PAIR_ROWS_CSV)
    weights_by_branch = _load_doc_weights(multistart_dir, attempts)

    object_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    node_rows: list[dict[str, Any]] = []
    node_pair_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []

    for object_role_id, group in attempts.groupby("object_role_universe_id", sort=False):
        group = group.sort_values("start_index", kind="mergesort")
        records = group.to_dict("records")
        memberships = [
            _load_npz(multistart_dir / str(record["terminal_membership_npz_path"]))
            for record in records
        ]
        object_node_ids = memberships[0]["object_node_ids"].astype(np.int64)
        universe_node_ids = memberships[0]["universe_node_ids"].astype(np.int64)
        for membership in memberships[1:]:
            if not np.array_equal(object_node_ids, membership["object_node_ids"]):
                raise ValueError(f"object node ids do not align for {object_role_id}")
            if not np.array_equal(universe_node_ids, membership["universe_node_ids"]):
                raise ValueError(f"universe node ids do not align for {object_role_id}")

        branch = str(group["branch"].iloc[0])
        full_weights = weights_by_branch.get(branch)
        if full_weights is None:
            object_doc_weights = np.ones(len(object_node_ids), dtype=np.float64)
            universe_doc_weights = np.ones(len(universe_node_ids), dtype=np.float64)
        else:
            object_doc_weights = full_weights[object_node_ids]
            universe_doc_weights = full_weights[universe_node_ids]

        object_labels_by_start = np.vstack(
            [membership["object_terminal_labels"] for membership in memberships]
        )
        universe_labels_by_start = np.vstack(
            [membership["universe_terminal_labels"] for membership in memberships]
        )
        object_coassignment = _coassignment_stats(object_labels_by_start)
        universe_coassignment = _coassignment_stats(universe_labels_by_start)
        object_node_cluster_stats = _node_cluster_stats(
            labels_by_start=object_labels_by_start,
            doc_weights=object_doc_weights,
        )
        universe_node_cluster_stats = _node_cluster_stats(
            labels_by_start=universe_labels_by_start,
            doc_weights=universe_doc_weights,
        )

        object_group_rows = _terminal_group_rows(group)
        group_rows.extend(object_group_rows)
        policy_pattern = _policy_split_pattern(object_group_rows)

        object_variable_partner_counts = object_coassignment[
            "variable_partner_count_by_node"
        ]
        universe_variable_partner_counts = universe_coassignment[
            "variable_partner_count_by_node"
        ]
        object_id_set = set(int(node) for node in object_node_ids)
        object_index_by_node = {int(node): idx for idx, node in enumerate(object_node_ids)}

        object_variable_nodes = object_variable_partner_counts > 0
        support_mask = np.asarray(
            [int(node) not in object_id_set for node in universe_node_ids],
            dtype=np.bool_,
        )
        universe_variable_nodes = universe_variable_partner_counts > 0
        support_variable_nodes = support_mask & universe_variable_nodes

        prior_pairs = prior_pair_rows[
            prior_pair_rows["object_role_universe_id"].astype(str).eq(str(object_role_id))
        ]
        object_stat = {
            "object_role_universe_id": object_role_id,
            "panel_case_id": group["panel_case_id"].iloc[0],
            "panel_case_rank": int(group["panel_case_rank"].iloc[0]),
            "branch": branch,
            "symmetric_object_id": group["symmetric_object_id"].iloc[0],
            "start_attempt_count": int(len(group)),
            "terminal_group_count": int(len(object_group_rows)),
            "terminal_group_policy_signature": " | ".join(
                sorted(str(row["start_policies"]) for row in object_group_rows)
            ),
            "policy_split_pattern": policy_pattern,
            "object_node_count": int(len(object_node_ids)),
            "support_node_count": int(len(universe_node_ids) - len(object_node_ids)),
            "universe_node_count": int(len(universe_node_ids)),
            "object_doc_sum": float(object_doc_weights.sum()),
            "support_doc_sum": float(universe_doc_weights[support_mask].sum()),
            "universe_doc_sum": float(universe_doc_weights.sum()),
            "object_pair_ari_min": float(prior_pairs["object_terminal_ari"].min()),
            "object_pair_ari_median": float(prior_pairs["object_terminal_ari"].median()),
            "universe_pair_ari_min": float(prior_pairs["universe_terminal_ari"].min()),
            "universe_pair_ari_median": float(
                prior_pairs["universe_terminal_ari"].median()
            ),
            "same_object_hash_pair_count": int(
                prior_pairs["same_object_terminal_hash"].sum()
            ),
            "object_coassignment_pair_count": object_coassignment[
                "coassignment_pair_count"
            ],
            "object_variable_pair_count": object_coassignment["variable_pair_count"],
            "object_variable_pair_share": object_coassignment["variable_pair_share"],
            "universe_coassignment_pair_count": universe_coassignment[
                "coassignment_pair_count"
            ],
            "universe_variable_pair_count": universe_coassignment["variable_pair_count"],
            "universe_variable_pair_share": universe_coassignment["variable_pair_share"],
            "object_variable_node_count": int(object_variable_nodes.sum()),
            "object_variable_node_share": float(object_variable_nodes.mean())
            if object_variable_nodes.size
            else None,
            "object_variable_node_doc_sum": float(
                object_doc_weights[object_variable_nodes].sum()
            ),
            "object_variable_node_doc_share": float(
                object_doc_weights[object_variable_nodes].sum() / object_doc_weights.sum()
            )
            if float(object_doc_weights.sum())
            else None,
            "support_variable_node_count": int(support_variable_nodes.sum()),
            "support_variable_node_share": float(
                support_variable_nodes.sum() / support_mask.sum()
            )
            if int(support_mask.sum())
            else None,
            "support_variable_node_doc_sum": float(
                universe_doc_weights[support_variable_nodes].sum()
            ),
            "support_variable_node_doc_share": float(
                universe_doc_weights[support_variable_nodes].sum()
                / universe_doc_weights[support_mask].sum()
            )
            if float(universe_doc_weights[support_mask].sum())
            else None,
            "object_cluster_size_median_of_medians": float(
                np.median(object_node_cluster_stats["cluster_size_median"])
            ),
            "object_cluster_size_max": int(
                object_node_cluster_stats["cluster_size_max"].max()
            ),
            "universe_cluster_size_median_of_medians": float(
                np.median(universe_node_cluster_stats["cluster_size_median"])
            ),
            "quality_min": float(group["quality"].min()),
            "quality_median": float(group["quality"].median()),
            "quality_max": float(group["quality"].max()),
            "quality_range": float(group["quality"].max() - group["quality"].min()),
            "run_status": RUN_STATUS,
            "claim_boundary": CLAIM_BOUNDARY,
        }
        object_stat["mechanism_read"] = _mechanism_read(object_group_rows, object_stat)
        object_rows.append(object_stat)

        pair_rows.extend(
            _pair_difference_rows(
                object_role_id=object_role_id,
                records=records,
                memberships=memberships,
            )
        )
        node_pair_rows.extend(
            _variable_node_pair_rows(
                object_role_id=object_role_id,
                records=records,
                universe_node_ids=universe_node_ids,
                universe_labels_by_start=universe_labels_by_start,
                universe_doc_weights=universe_doc_weights,
                object_id_set=object_id_set,
            )
        )

        for idx, node_id in enumerate(universe_node_ids):
            is_object = int(node_id) in object_id_set
            object_idx = object_index_by_node.get(int(node_id))
            object_variable_partner_count = (
                int(object_variable_partner_counts[object_idx]) if object_idx is not None else None
            )
            node_rows.append(
                {
                    "object_role_universe_id": object_role_id,
                    "branch": branch,
                    "node_id": int(node_id),
                    "node_scope": "object" if is_object else "support",
                    "doc_count": float(universe_doc_weights[idx]),
                    "object_variable_partner_count": object_variable_partner_count,
                    "universe_variable_partner_count": int(
                        universe_variable_partner_counts[idx]
                    ),
                    "universe_cluster_size_min": int(
                        universe_node_cluster_stats["cluster_size_min"][idx]
                    ),
                    "universe_cluster_size_median": float(
                        universe_node_cluster_stats["cluster_size_median"][idx]
                    ),
                    "universe_cluster_size_max": int(
                        universe_node_cluster_stats["cluster_size_max"][idx]
                    ),
                    "universe_cluster_doc_sum_min": float(
                        universe_node_cluster_stats["cluster_doc_sum_min"][idx]
                    ),
                    "universe_cluster_doc_sum_median": float(
                        universe_node_cluster_stats["cluster_doc_sum_median"][idx]
                    ),
                    "universe_cluster_doc_sum_max": float(
                        universe_node_cluster_stats["cluster_doc_sum_max"][idx]
                    ),
                    "singleton_start_count": int(
                        universe_node_cluster_stats["singleton_start_count"][idx]
                    ),
                    "is_top_variable_node": False,
                    "run_status": RUN_STATUS,
                    "claim_boundary": CLAIM_BOUNDARY,
                }
            )

    node_frame = pd.DataFrame(node_rows)
    if not node_frame.empty:
        node_frame["variable_rank"] = (
            node_frame.sort_values(
                [
                    "object_role_universe_id",
                    "universe_variable_partner_count",
                    "doc_count",
                ],
                ascending=[True, False, False],
                kind="mergesort",
            )
            .groupby("object_role_universe_id")
            .cumcount()
            + 1
        )
        node_frame["is_top_variable_node"] = (
            node_frame["variable_rank"] <= int(args.top_variable_nodes_per_object)
        ) & (node_frame["universe_variable_partner_count"] > 0)

    object_frame = pd.DataFrame(object_rows)
    pair_frame = pd.DataFrame(pair_rows)
    node_pair_frame = pd.DataFrame(node_pair_rows)
    group_frame = pd.DataFrame(group_rows)

    _write_csv(object_frame, output_dir / OBJECT_ROWS_CSV)
    _write_csv(pair_frame, output_dir / PAIR_ROWS_CSV)
    _write_csv(node_frame, output_dir / NODE_ROWS_CSV)
    _write_csv(node_pair_frame, output_dir / NODE_PAIR_ROWS_CSV)
    _write_csv(group_frame, output_dir / GROUP_ROWS_CSV)

    summary = _build_summary(
        multistart_dir=multistart_dir,
        output_dir=output_dir,
        object_frame=object_frame,
        pair_frame=pair_frame,
        node_frame=node_frame,
        node_pair_frame=node_pair_frame,
    )
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    config = {
        "schema": "nanoclustering_symmetric_object_terminal_difference_review.v1",
        "multistart_dir": str(multistart_dir),
        "output_dir": str(output_dir),
        "top_variable_nodes_per_object": int(args.top_variable_nodes_per_object),
        "source_claim_boundary": MULTISTART_CLAIM_BOUNDARY,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(
        output_dir=output_dir,
        summary=summary,
        object_frame=object_frame,
        node_frame=node_frame,
        node_pair_frame=node_pair_frame,
    )
    return summary


def _build_summary(
    *,
    multistart_dir: Path,
    output_dir: Path,
    object_frame: pd.DataFrame,
    pair_frame: pd.DataFrame,
    node_frame: pd.DataFrame,
    node_pair_frame: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "schema": "nanoclustering_symmetric_object_terminal_difference_review_summary.v1",
        "status": RUN_STATUS if not object_frame.empty else "no_objects",
        "multistart_dir": str(multistart_dir),
        "output_dir": str(output_dir),
        "object_count": int(len(object_frame)),
        "terminal_multiplicity_object_count": int(
            (object_frame["terminal_group_count"] > 1).sum()
        )
        if not object_frame.empty
        else 0,
        "pair_count": int(len(pair_frame)),
        "object_variable_pair_share_median": float(
            object_frame["object_variable_pair_share"].median()
        )
        if not object_frame.empty
        else None,
        "universe_variable_pair_share_median": float(
            object_frame["universe_variable_pair_share"].median()
        )
        if not object_frame.empty
        else None,
        "object_variable_node_share_median": float(
            object_frame["object_variable_node_share"].median()
        )
        if not object_frame.empty
        else None,
        "support_variable_node_share_median": float(
            object_frame["support_variable_node_share"].median()
        )
        if not object_frame.empty
        else None,
        "mechanism_reads": (
            object_frame["mechanism_read"].value_counts().to_dict()
            if not object_frame.empty
            else {}
        ),
        "top_variable_node_count": int(node_frame["is_top_variable_node"].sum())
        if not node_frame.empty
        else 0,
        "variable_node_pair_count": int(len(node_pair_frame)),
        "variable_node_pair_scope_counts": (
            node_pair_frame["pair_scope"].value_counts().to_dict()
            if not node_pair_frame.empty
            else {}
        ),
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _write_report(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    object_frame: pd.DataFrame,
    node_frame: pd.DataFrame,
    node_pair_frame: pd.DataFrame,
) -> None:
    lines = [
        "# NanoClustering Symmetric-Object Terminal Difference Review",
        "",
        f"- status: `{summary['status']}`",
        f"- object_count: {summary['object_count']}",
        f"- terminal_multiplicity_object_count: {summary['terminal_multiplicity_object_count']}",
        f"- object_variable_pair_share_median: {summary['object_variable_pair_share_median']}",
        f"- universe_variable_pair_share_median: {summary['universe_variable_pair_share_median']}",
        f"- object_variable_node_share_median: {summary['object_variable_node_share_median']}",
        f"- support_variable_node_share_median: {summary['support_variable_node_share_median']}",
        f"- variable_node_pair_count: {summary['variable_node_pair_count']}",
        f"- variable_node_pair_scope_counts: {summary['variable_node_pair_scope_counts']}",
        f"- mechanism_reads: {summary['mechanism_reads']}",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Objects",
    ]
    for row in object_frame.sort_values(
        ["object_variable_pair_share", "quality_range"],
        ascending=[False, False],
    ).itertuples(index=False):
        data = row._asdict()
        lines.append(
            "- "
            f"{data['object_role_universe_id']}: "
            f"groups={data['terminal_group_count']}, "
            f"pattern={data['policy_split_pattern']}, "
            f"read={data['mechanism_read']}, "
            f"object_variable_pair_share={data['object_variable_pair_share']}, "
            f"universe_variable_pair_share={data['universe_variable_pair_share']}, "
            f"object_variable_node_share={data['object_variable_node_share']}, "
            f"support_variable_node_share={data['support_variable_node_share']}, "
            f"quality_range={data['quality_range']}"
        )
        top_nodes = node_frame[
            node_frame["object_role_universe_id"].astype(str).eq(
                str(data["object_role_universe_id"])
            )
            & node_frame["is_top_variable_node"].astype(bool)
        ].sort_values(
            ["universe_variable_partner_count", "doc_count"],
            ascending=[False, False],
            kind="mergesort",
        )
        node_bits = [
            f"{int(node.node_id)}:{node.node_scope}:var{int(node.universe_variable_partner_count)}:doc{float(node.doc_count)}"
            for node in top_nodes.head(5).itertuples(index=False)
        ]
        if node_bits:
            lines.append(f"  - top_variable_nodes: {'; '.join(node_bits)}")
        if node_pair_frame.empty or "object_role_universe_id" not in node_pair_frame.columns:
            top_pairs = pd.DataFrame()
        else:
            top_pairs = node_pair_frame[
                node_pair_frame["object_role_universe_id"].astype(str).eq(
                    str(data["object_role_universe_id"])
                )
            ].sort_values(
                ["same_terminal_start_count", "pair_doc_count_sum"],
                ascending=[False, False],
                kind="mergesort",
            )
        pair_bits = [
            (
                f"{int(pair.left_node_id)}-{int(pair.right_node_id)}:"
                f"{pair.pair_scope}:same{int(pair.same_terminal_start_count)}:"
                f"together[{pair.together_start_policies}]"
            )
            for pair in top_pairs.head(5).itertuples(index=False)
        ]
        if pair_bits:
            lines.append(f"  - variable_node_pairs: {'; '.join(pair_bits)}")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            (
                "These rows describe terminal membership structure under saved "
                "multistart slices only. They do not establish optimizer walls, "
                "pathways, basin quality, cost advantage, method success, or "
                "algorithm novelty."
            ),
            "",
        ]
    )
    (output_dir / REPORT_MD).write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--multistart-dir", type=Path, default=DEFAULT_MULTISTART_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-variable-nodes-per-object", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    summary = analyze(parse_args())
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
