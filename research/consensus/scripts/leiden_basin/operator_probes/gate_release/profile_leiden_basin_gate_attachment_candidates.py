#!/usr/bin/env python3
"""Score target nodes that may attach after releasing a source-side gate.

This diagnostic is deliberately pre-operator: it rebuilds the source post-gate
state, scores target nodes from source-state graph features, and then overlays
the already observed gate-trace moved nodes as labels.  It does not run a new
Leiden polish per candidate.
"""

from __future__ import annotations

import argparse
import json
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


import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent

from analyze_leiden_basin_barrier_aware_pathways import (  # noqa: E402
    DEFAULT_OUTPUT_DIR as DEFAULT_PREFIX_DIR,
    PREFIX_ROWS_FILENAME as BARRIER_PREFIX_ROWS_FILENAME,
)
from evaluate_leiden_basin_polish_prefixes import select_prefix_rows  # noqa: E402
from evaluate_leiden_basin_target_elbow_polish import (  # noqa: E402
    _rank_and_filter_prefix_rows,
)
from probe_leiden_basin_post_gate_recovery_moves import (  # noqa: E402
    DEFAULT_CANDIDATE_DIRS,
    DEFAULT_POST_GATE_DIR,
    DEFAULT_PROFILE_BATCH_DIR,
    DEFAULT_VANILLA_DIR,
    POST_GATE_PATH_SUMMARY_FILENAME,
    _load_case_context,
    _markdown_table,
    _prefix_context,
    _replay_to_source_state,
    _select_source_path,
)
from probe_leiden_basin_post_gate_recovery_subsets import (  # noqa: E402
    DEFAULT_SOURCE_MOVE_DIR,
    SOURCE_MOVE_ROWS_FILENAME,
    _load_source_config,
    _select_source_move,
)
from profile_leiden_basin_post_gate_gate_trace import (  # noqa: E402
    DEFAULT_OUTPUT_DIR as DEFAULT_GATE_TRACE_DIR,
    DEFAULT_SOURCE_RECOVERY_POLICY,
    DEFAULT_SUFFICIENT_DIR,
    NODE_ROWS_FILENAME as GATE_TRACE_NODE_ROWS_FILENAME,
    _load_json,
    _selected_gate_nodes,
)
from sciscape.clustering.leiden_basin_search import (  # noqa: E402
    POST_GATE_VERDICT_NEAR_MISS,
    node_csv,
    unique_sorted_u32,
    weighted_pull_to_nodes,
)

DEFAULT_OUTPUT_DIR = DEFAULT_SOURCE_MOVE_DIR.parent / (
    "basin_transition_post_gate_gate_attachment_candidates_field34_cc_c0_p8_v0"
)

CANDIDATE_ROWS_FILENAME = "gate_attachment_candidate_rows.csv"
RANK_SUMMARY_ROWS_FILENAME = "gate_attachment_rank_summary_rows.csv"
SUMMARY_FILENAME = "gate_attachment_candidate_summary.json"
CONFIG_FILENAME = "gate_attachment_candidate_config.json"
REPORT_FILENAME = "gate_attachment_candidate_report.md"

def _parse_csv_tuple(value: str | None) -> tuple[int, ...]:
    if value is None:
        return ()
    return tuple(int(part.strip()) for part in str(value).split(",") if part.strip())

def _mode_int(values: pd.Series) -> int | None:
    if values.empty:
        return None
    counts = values.astype(int).value_counts()
    if counts.empty:
        return None
    return int(counts.index[0])

def _rank_desc(frame: pd.DataFrame, score_column: str, rank_column: str) -> None:
    ordered = frame.sort_values([score_column, "node"], ascending=[False, True])
    ranks = np.arange(1, len(ordered) + 1, dtype=np.int64)
    frame[rank_column] = pd.Series(ranks, index=ordered.index).sort_index()

def _incident_pull_to_same_label(
    *,
    nodes: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    weight: np.ndarray,
    labels: np.ndarray,
    node_count: int,
) -> np.ndarray:
    scores = np.zeros(int(node_count), dtype=np.float64)
    node_arr = unique_sorted_u32(nodes).astype(np.int64)
    if node_arr.size == 0:
        return scores
    node_mask = np.zeros(int(node_count), dtype=np.bool_)
    node_mask[node_arr] = True
    src_arr = np.asarray(src, dtype=np.int64)
    dst_arr = np.asarray(dst, dtype=np.int64)
    weights = np.asarray(weight, dtype=np.float64)
    label_arr = np.asarray(labels, dtype=np.uint64)
    src_hit = node_mask[src_arr] & (label_arr[src_arr] == label_arr[dst_arr])
    dst_hit = node_mask[dst_arr] & (label_arr[dst_arr] == label_arr[src_arr])
    np.add.at(scores, src_arr[src_hit], weights[src_hit])
    np.add.at(scores, dst_arr[dst_hit], weights[dst_hit])
    return scores

def _incident_edge_count_to_nodes(
    *,
    src: np.ndarray,
    dst: np.ndarray,
    target_nodes: np.ndarray,
    node_count: int,
) -> np.ndarray:
    counts = np.zeros(int(node_count), dtype=np.int64)
    targets = unique_sorted_u32(target_nodes).astype(np.int64)
    if targets.size == 0:
        return counts
    target_mask = np.zeros(int(node_count), dtype=np.bool_)
    target_mask[targets] = True
    src_arr = np.asarray(src, dtype=np.int64)
    dst_arr = np.asarray(dst, dtype=np.int64)
    src_hit = target_mask[src_arr]
    dst_hit = target_mask[dst_arr]
    np.add.at(counts, dst_arr[src_hit], 1)
    np.add.at(counts, src_arr[dst_hit], 1)
    counts[targets] = 0
    return counts

def _label_size_array(labels: np.ndarray) -> np.ndarray:
    label_arr = np.asarray(labels, dtype=np.int64)
    counts = pd.Series(label_arr).value_counts(sort=False).to_dict()
    return np.asarray([counts[int(label)] for label in label_arr], dtype=np.int64)

def _label_weight_array(labels: np.ndarray, node_weights: np.ndarray) -> np.ndarray:
    frame = pd.DataFrame(
        {
            "label": np.asarray(labels, dtype=np.int64),
            "weight": np.asarray(node_weights, dtype=np.float64),
        }
    )
    weights = frame.groupby("label", sort=False)["weight"].sum().to_dict()
    return np.asarray([weights[int(label)] for label in frame["label"]], dtype=np.float64)

def _load_trace_moved_nodes(gate_trace_dir: Path) -> np.ndarray:
    path = gate_trace_dir / GATE_TRACE_NODE_ROWS_FILENAME
    if not path.exists():
        return np.asarray([], dtype=np.uint32)
    rows = pd.read_csv(path)
    if rows.empty or "moved_source_to_gate" not in rows:
        return np.asarray([], dtype=np.uint32)
    moved = rows[rows["moved_source_to_gate"].astype(bool)]["node"].to_numpy(
        dtype=np.uint32
    )
    return unique_sorted_u32(moved)

def _candidate_rows(
    *,
    source_state: Any,
    gate_nodes: np.ndarray,
    full_context_nodes: np.ndarray,
    moved_trace_nodes: np.ndarray,
    case_ctx: dict[str, Any],
) -> pd.DataFrame:
    arrays = case_ctx["arrays"]
    src = np.asarray(arrays.src, dtype=np.uint32)
    dst = np.asarray(arrays.dst, dtype=np.uint32)
    weight = np.asarray(arrays.weight, dtype=np.float64)
    node_count = int(case_ctx["baseline"].membership.size)
    node_weights = np.asarray(arrays.node_weights, dtype=np.float64)
    source = np.asarray(source_state.membership, dtype=np.uint64)
    baseline = np.asarray(case_ctx["baseline"].membership, dtype=np.uint64)
    candidate = np.asarray(case_ctx["candidate"].recreated.membership, dtype=np.uint64)
    vanilla = np.asarray(case_ctx["vanilla"].membership, dtype=np.uint64)

    target_nodes = unique_sorted_u32(source_state.target_nodes)
    gate_nodes = unique_sorted_u32(gate_nodes)
    full_context_nodes = unique_sorted_u32(full_context_nodes)
    source_action_nodes = unique_sorted_u32(source_state.action_nodes)
    source_mutable_nodes = unique_sorted_u32(source_state.mutable_nodes)
    direct_nodes = unique_sorted_u32(source_state.direct_nodes)

    pull_to_gate = weighted_pull_to_nodes(
        src=src,
        dst=dst,
        weight=weight,
        target_nodes=gate_nodes,
        node_count=node_count,
    )
    pull_to_full_context = weighted_pull_to_nodes(
        src=src,
        dst=dst,
        weight=weight,
        target_nodes=full_context_nodes,
        node_count=node_count,
    )
    pull_to_action = weighted_pull_to_nodes(
        src=src,
        dst=dst,
        weight=weight,
        target_nodes=source_action_nodes,
        node_count=node_count,
    )
    pull_to_mutable = weighted_pull_to_nodes(
        src=src,
        dst=dst,
        weight=weight,
        target_nodes=source_mutable_nodes,
        node_count=node_count,
    )
    pull_to_direct = weighted_pull_to_nodes(
        src=src,
        dst=dst,
        weight=weight,
        target_nodes=direct_nodes,
        node_count=node_count,
    )
    pull_to_target_nodes = weighted_pull_to_nodes(
        src=src,
        dst=dst,
        weight=weight,
        target_nodes=target_nodes,
        node_count=node_count,
    )
    count_to_gate = _incident_edge_count_to_nodes(
        src=src,
        dst=dst,
        target_nodes=gate_nodes,
        node_count=node_count,
    )
    count_to_full_context = _incident_edge_count_to_nodes(
        src=src,
        dst=dst,
        target_nodes=full_context_nodes,
        node_count=node_count,
    )
    pull_to_source_label = _incident_pull_to_same_label(
        nodes=target_nodes,
        src=src,
        dst=dst,
        weight=weight,
        labels=source,
        node_count=node_count,
    )
    source_label_sizes = _label_size_array(source)
    source_label_weights = _label_weight_array(source, node_weights)

    gate_index = gate_nodes.astype(np.int64)
    gate_source_label = _mode_int(pd.Series(source[gate_index].astype(np.int64)))
    gate_baseline_label = _mode_int(pd.Series(baseline[gate_index].astype(np.int64)))
    gate_candidate_label = _mode_int(pd.Series(candidate[gate_index].astype(np.int64)))
    gate_vanilla_label = _mode_int(pd.Series(vanilla[gate_index].astype(np.int64)))

    gate_set = set(int(node) for node in gate_nodes)
    full_set = set(int(node) for node in full_context_nodes)
    action_set = set(int(node) for node in source_action_nodes)
    mutable_set = set(int(node) for node in source_mutable_nodes)
    direct_set = set(int(node) for node in direct_nodes)
    moved_set = set(int(node) for node in moved_trace_nodes)

    rows: list[dict[str, Any]] = []
    eps = 1e-12
    for node in target_nodes.astype(np.int64):
        node_i = int(node)
        gate_pull = float(pull_to_gate[node_i])
        source_pull = float(pull_to_source_label[node_i])
        full_pull = float(pull_to_full_context[node_i])
        action_pull = float(pull_to_action[node_i])
        mutable_pull = float(pull_to_mutable[node_i])
        direct_pull = float(pull_to_direct[node_i])
        target_pull = float(pull_to_target_nodes[node_i])
        gate_edge_count = int(count_to_gate[node_i])
        full_edge_count = int(count_to_full_context[node_i])
        rows.append(
            {
                "node": node_i,
                "observed_moved_source_to_gate": node_i in moved_set,
                "in_gate_context": node_i in gate_set,
                "in_full_context": node_i in full_set,
                "in_source_action": node_i in action_set,
                "in_source_mutable": node_i in mutable_set,
                "in_direct_nodes": node_i in direct_set,
                "baseline_label": int(baseline[node_i]),
                "candidate_label": int(candidate[node_i]),
                "vanilla_label": int(vanilla[node_i]),
                "source_label": int(source[node_i]),
                "same_source_label_as_gate": (
                    gate_source_label is not None and int(source[node_i]) == gate_source_label
                ),
                "same_baseline_label_as_gate": (
                    gate_baseline_label is not None
                    and int(baseline[node_i]) == gate_baseline_label
                ),
                "same_candidate_label_as_gate": (
                    gate_candidate_label is not None
                    and int(candidate[node_i]) == gate_candidate_label
                ),
                "same_vanilla_label_as_gate": (
                    gate_vanilla_label is not None
                    and int(vanilla[node_i]) == gate_vanilla_label
                ),
                "source_label_size": int(source_label_sizes[node_i]),
                "source_label_weight": float(source_label_weights[node_i]),
                "node_weight": float(node_weights[node_i]),
                "pull_to_gate_context": gate_pull,
                "edge_count_to_gate_context": gate_edge_count,
                "mean_edge_weight_to_gate_context": (
                    gate_pull / gate_edge_count if gate_edge_count else 0.0
                ),
                "pull_to_full_context": full_pull,
                "edge_count_to_full_context": full_edge_count,
                "pull_to_source_action": action_pull,
                "pull_to_source_mutable": mutable_pull,
                "pull_to_direct_nodes": direct_pull,
                "pull_to_target_nodes": target_pull,
                "pull_to_current_source_label": source_pull,
                "gate_pull_margin_vs_current_source": gate_pull - source_pull,
                "gate_pull_share_vs_current_source": gate_pull
                / (gate_pull + source_pull + eps),
                "gate_pull_margin_vs_source_action": gate_pull - action_pull,
                "gate_pull_share_vs_action": gate_pull / (gate_pull + action_pull + eps),
                "gate_pull_density_per_gate_node": gate_pull / max(int(gate_nodes.size), 1),
                "gate_pull_per_source_label_node": gate_pull
                / max(int(source_label_sizes[node_i]), 1),
            }
        )

    frame = pd.DataFrame(rows)
    for score_column, rank_column in [
        ("pull_to_gate_context", "rank_pull_to_gate_context"),
        ("gate_pull_margin_vs_current_source", "rank_gate_margin_vs_source"),
        ("gate_pull_share_vs_current_source", "rank_gate_share_vs_source"),
        ("gate_pull_margin_vs_source_action", "rank_gate_margin_vs_action"),
        ("gate_pull_per_source_label_node", "rank_gate_pull_per_source_label_node"),
        ("mean_edge_weight_to_gate_context", "rank_mean_edge_weight_to_gate"),
    ]:
        _rank_desc(frame, score_column, rank_column)
    rank_columns = [column for column in frame.columns if column.startswith("rank_")]
    frame["rank_mean_consensus"] = frame[rank_columns].mean(axis=1)
    frame["rank_best_consensus"] = frame[rank_columns].min(axis=1)
    return frame.sort_values(
        ["rank_mean_consensus", "rank_best_consensus", "node"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

def _rank_summary_rows(rows: pd.DataFrame, moved_nodes: np.ndarray) -> pd.DataFrame:
    moved_set = set(int(node) for node in moved_nodes)
    summaries: list[dict[str, Any]] = []
    for score_column in [
        "pull_to_gate_context",
        "gate_pull_margin_vs_current_source",
        "gate_pull_share_vs_current_source",
        "gate_pull_margin_vs_source_action",
        "gate_pull_per_source_label_node",
        "mean_edge_weight_to_gate_context",
    ]:
        rank_column = "rank_" + {
            "pull_to_gate_context": "pull_to_gate_context",
            "gate_pull_margin_vs_current_source": "gate_margin_vs_source",
            "gate_pull_share_vs_current_source": "gate_share_vs_source",
            "gate_pull_margin_vs_source_action": "gate_margin_vs_action",
            "gate_pull_per_source_label_node": "gate_pull_per_source_label_node",
            "mean_edge_weight_to_gate_context": "mean_edge_weight_to_gate",
        }[score_column]
        top = rows.sort_values([score_column, "node"], ascending=[False, True]).iloc[0]
        moved_subset = rows[rows["node"].astype(int).isin(moved_set)].copy()
        if moved_subset.empty:
            moved_best_rank = None
            moved_best_node = None
            moved_best_score = None
        else:
            moved_subset = moved_subset.sort_values([rank_column, "node"])
            best = moved_subset.iloc[0]
            moved_best_rank = int(best[rank_column])
            moved_best_node = int(best["node"])
            moved_best_score = float(best[score_column])
        summaries.append(
            {
                "score_column": score_column,
                "rank_column": rank_column,
                "top_node": int(top["node"]),
                "top_score": float(top[score_column]),
                "top_observed_moved": bool(top["node"] in moved_set),
                "moved_best_node": moved_best_node,
                "moved_best_score": moved_best_score,
                "moved_best_rank": moved_best_rank,
                "moved_best_top_fraction": (
                    float(moved_best_rank) / float(len(rows))
                    if moved_best_rank is not None and len(rows)
                    else None
                ),
            }
        )
    moved_rows = rows[rows["node"].astype(int).isin(moved_set)].copy()
    if not moved_rows.empty:
        consensus = rows.sort_values(["rank_mean_consensus", "node"]).reset_index(
            drop=True
        )
        consensus["consensus_order_rank"] = np.arange(
            1, len(consensus) + 1, dtype=np.int64
        )
        moved = consensus[consensus["node"].astype(int).isin(moved_set)].iloc[0]
        top = consensus.iloc[0]
        summaries.append(
            {
                "score_column": "rank_mean_consensus",
                "rank_column": "rank_mean_consensus",
                "top_node": int(top["node"]),
                "top_score": float(top["rank_mean_consensus"]),
                "top_observed_moved": bool(int(top["node"]) in moved_set),
                "moved_best_node": int(moved["node"]),
                "moved_best_score": float(moved["rank_mean_consensus"]),
                "moved_best_rank": int(moved["consensus_order_rank"]),
                "moved_best_top_fraction": float(moved["consensus_order_rank"])
                / float(len(rows)),
            }
        )
    return pd.DataFrame(summaries)

def _write_report(
    path: Path,
    *,
    summary: dict[str, Any],
    rank_rows: pd.DataFrame,
    candidate_rows: pd.DataFrame,
) -> None:
    top_rows = candidate_rows.sort_values(
        ["rank_mean_consensus", "rank_best_consensus", "node"]
    ).head(20)
    moved_rows = candidate_rows[
        candidate_rows["observed_moved_source_to_gate"].astype(bool)
    ].copy()
    moved_cols = [
        "node",
        "pull_to_gate_context",
        "edge_count_to_gate_context",
        "pull_to_current_source_label",
        "gate_pull_margin_vs_current_source",
        "gate_pull_share_vs_current_source",
        "pull_to_source_action",
        "same_vanilla_label_as_gate",
        "source_label_size",
        "rank_pull_to_gate_context",
        "rank_gate_margin_vs_source",
        "rank_gate_share_vs_source",
        "rank_mean_consensus",
    ]
    top_cols = [
        "node",
        "observed_moved_source_to_gate",
        "pull_to_gate_context",
        "edge_count_to_gate_context",
        "pull_to_current_source_label",
        "gate_pull_margin_vs_current_source",
        "gate_pull_share_vs_current_source",
        "pull_to_source_action",
        "same_vanilla_label_as_gate",
        "source_label_size",
        "rank_mean_consensus",
    ]
    lines = [
        "# Gate Attachment Candidate Scores",
        "",
        "This diagnostic scores target nodes from source-state graph features before",
        "running another recovery polish.  Observed moved nodes are labels from the",
        "existing gate trace, not training targets used by the score.",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "| --- | --- |",
    ]
    for key in [
        "output_dir",
        "pair_id",
        "prefix_rank",
        "target_node_count",
        "gate_node_count",
        "full_context_node_count",
        "observed_moved_node_count",
        "observed_moved_node_ids",
        "gate_dominant_source_label",
        "gate_dominant_vanilla_label",
        "best_consensus_node",
        "best_consensus_observed_moved",
        "moved_best_consensus_rank",
        "moved_best_pull_rank",
        "moved_best_gate_margin_rank",
    ]:
        lines.append(f"| {key} | {summary.get(key, '')} |")
    lines.extend(["", "## Moved Nodes", ""])
    if moved_rows.empty:
        lines.append("_No observed moved nodes were available from the gate trace._")
    else:
        lines.extend(_markdown_table(moved_rows[moved_cols], max_rows=20))
    lines.extend(["", "## Rank Summary", ""])
    lines.extend(_markdown_table(rank_rows, max_rows=20))
    lines.extend(["", "## Top Consensus Candidates", ""])
    lines.extend(_markdown_table(top_rows[top_cols], max_rows=20))
    lines.extend(
        [
            "",
            "## Reading",
            "",
            "- High gate pull means the target node already has source-state edge mass into the released gate context.",
            "- High margin versus current source label means the gate is a stronger local attachment than the node's current source label.",
            "- These are descriptive attachment scores; CPM penalties and Leiden refinement are not modeled here.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def run_profile(
    *,
    sufficient_dir: Path,
    gate_trace_dir: Path,
    source_move_dir: Path,
    post_gate_dir: Path,
    prefix_dir: Path,
    profile_batch_dir: Path,
    output_dir: Path,
    candidate_dirs: tuple[Path, ...],
    vanilla_dir: Path,
    pair_id: str,
    prefix_rank: int,
    source_verdict: str,
    source_recovery_policy: str,
    baseline_iterations: int,
    candidate_polish_iterations: int,
    local_polish_iterations: int,
    target_action_multiplier: float,
    max_target_action_nodes: int,
    cumulative_fraction: float,
    min_score_fraction: float,
    min_gap_fraction: float,
    min_guarded_pull_fraction: float,
    resolution: float,
    randomness: float,
    perturb_seed_offset: int,
    polish_seed_offset: int,
    min_support_shift_from_vanilla: float,
    min_material_q_gain: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    gate_nodes, sufficient_final = _selected_gate_nodes(sufficient_dir)
    source_moves = pd.read_csv(source_move_dir / SOURCE_MOVE_ROWS_FILENAME)
    source_move, _ = _select_source_move(
        source_moves,
        recovery_policy=source_recovery_policy,
    )
    full_context_nodes = unique_sorted_u32(
        _parse_csv_tuple(source_move.get("selected_node_ids"))
    )

    trace_moved_nodes = _load_trace_moved_nodes(gate_trace_dir)
    post_gate_paths = pd.read_csv(post_gate_dir / POST_GATE_PATH_SUMMARY_FILENAME)
    source_path = _select_source_path(
        post_gate_paths,
        pair_id=pair_id,
        prefix_rank=prefix_rank,
        verdict=source_verdict,
    )
    prefixes = select_prefix_rows(
        pd.read_csv(prefix_dir / BARRIER_PREFIX_ROWS_FILENAME),
        pair_ids=(pair_id,),
        top_prefixes_per_case=max(prefix_rank, 10),
    )
    prefixes = _rank_and_filter_prefix_rows(
        prefixes,
        selected_prefix_ranks=(prefix_rank,),
    )
    if prefixes.empty:
        raise ValueError(f"No prefix row selected for {pair_id} rank {prefix_rank}")
    prefix_row = prefixes.iloc[0]
    case_ctx = _load_case_context(
        prefix_row=prefix_row,
        profile_batch_dir=profile_batch_dir,
        candidate_dirs=candidate_dirs,
        vanilla_dir=vanilla_dir,
        baseline_iterations=baseline_iterations,
        candidate_polish_iterations=candidate_polish_iterations,
        resolution=resolution,
        randomness=randomness,
        perturb_seed_offset=perturb_seed_offset,
    )
    source_state, source_row, _, _ = _replay_to_source_state(
        prefix_row=prefix_row,
        source_path=source_path,
        case_ctx=case_ctx,
        target_action_multiplier=target_action_multiplier,
        max_target_action_nodes=max_target_action_nodes,
        cumulative_fraction=cumulative_fraction,
        min_score_fraction=min_score_fraction,
        min_gap_fraction=min_gap_fraction,
        min_guarded_pull_fraction=min_guarded_pull_fraction,
        local_polish_iterations=local_polish_iterations,
        resolution=resolution,
        randomness=randomness,
        polish_seed_offset=polish_seed_offset,
        min_support_shift_from_vanilla=min_support_shift_from_vanilla,
        min_material_q_gain=min_material_q_gain,
    )
    if full_context_nodes.size == 0:
        full_context_nodes = gate_nodes

    candidate_rows = _candidate_rows(
        source_state=source_state,
        gate_nodes=gate_nodes,
        full_context_nodes=full_context_nodes,
        moved_trace_nodes=trace_moved_nodes,
        case_ctx=case_ctx,
    )
    rank_rows = _rank_summary_rows(candidate_rows, trace_moved_nodes)
    moved_rank_rows = candidate_rows[
        candidate_rows["observed_moved_source_to_gate"].astype(bool)
    ].copy()
    consensus_order = candidate_rows.sort_values(
        ["rank_mean_consensus", "rank_best_consensus", "node"]
    ).reset_index(drop=True)
    best_consensus = consensus_order.iloc[0]

    arrays = case_ctx["arrays"]
    source = np.asarray(source_state.membership, dtype=np.uint64)
    vanilla = np.asarray(case_ctx["vanilla"].membership, dtype=np.uint64)
    gate_idx = gate_nodes.astype(np.int64)
    gate_source_label = _mode_int(pd.Series(source[gate_idx].astype(np.int64)))
    gate_vanilla_label = _mode_int(pd.Series(vanilla[gate_idx].astype(np.int64)))
    summary = {
        "schema": "leiden_basin_gate_attachment_candidates.v0",
        "output_dir": str(output_dir),
        "sufficient_dir": str(sufficient_dir),
        "gate_trace_dir": str(gate_trace_dir),
        "source_move_dir": str(source_move_dir),
        "pair_id": pair_id,
        "prefix_rank": int(prefix_rank),
        "source_recovery_policy": source_recovery_policy,
        "target_node_count": int(candidate_rows.shape[0]),
        "gate_node_count": int(gate_nodes.size),
        "full_context_node_count": int(full_context_nodes.size),
        "source_mutable_node_count": int(unique_sorted_u32(source_state.mutable_nodes).size),
        "source_action_node_count": int(unique_sorted_u32(source_state.action_nodes).size),
        "observed_moved_node_count": int(trace_moved_nodes.size),
        "observed_moved_node_ids": node_csv(trace_moved_nodes),
        "gate_dominant_source_label": gate_source_label,
        "gate_dominant_vanilla_label": gate_vanilla_label,
        "source_state_delta_q_vs_start": float(source_row["state_delta_q_vs_start"]),
        "source_state_support_distance_to_vanilla": float(
            source_row["state_support_distance_to_vanilla"]
        ),
        "best_consensus_node": int(best_consensus["node"]),
        "best_consensus_observed_moved": bool(
            best_consensus["observed_moved_source_to_gate"]
        ),
        "moved_best_consensus_rank": None,
        "moved_best_pull_rank": None,
        "moved_best_gate_margin_rank": None,
        "edge_count": int(np.asarray(arrays.weight).size),
    }
    if not moved_rank_rows.empty:
        moved_best = moved_rank_rows.sort_values(["rank_mean_consensus", "node"]).iloc[0]
        summary.update(
            {
                "moved_best_consensus_node": int(moved_best["node"]),
                "moved_best_consensus_rank": int(
                    consensus_order.index[
                        consensus_order["node"].astype(int).eq(int(moved_best["node"]))
                    ][0]
                    + 1
                ),
                "moved_best_consensus_score": float(moved_best["rank_mean_consensus"]),
                "moved_best_pull_rank": int(moved_best["rank_pull_to_gate_context"]),
                "moved_best_gate_margin_rank": int(
                    moved_best["rank_gate_margin_vs_source"]
                ),
                "moved_best_gate_share_rank": int(
                    moved_best["rank_gate_share_vs_source"]
                ),
                "moved_best_pull_to_gate_context": float(
                    moved_best["pull_to_gate_context"]
                ),
                "moved_best_gate_margin_vs_source": float(
                    moved_best["gate_pull_margin_vs_current_source"]
                ),
            }
        )

    config = {
        "sufficient_dir": str(sufficient_dir),
        "gate_trace_dir": str(gate_trace_dir),
        "source_move_dir": str(source_move_dir),
        "post_gate_dir": str(post_gate_dir),
        "prefix_dir": str(prefix_dir),
        "profile_batch_dir": str(profile_batch_dir),
        "candidate_dirs": [str(path) for path in candidate_dirs],
        "vanilla_dir": str(vanilla_dir),
        "pair_id": pair_id,
        "prefix_rank": int(prefix_rank),
        "source_verdict": source_verdict,
        "source_recovery_policy": source_recovery_policy,
        "baseline_iterations": int(baseline_iterations),
        "candidate_polish_iterations": int(candidate_polish_iterations),
        "local_polish_iterations": int(local_polish_iterations),
        "target_action_multiplier": float(target_action_multiplier),
        "max_target_action_nodes": int(max_target_action_nodes),
        "resolution": float(resolution),
        "randomness": float(randomness),
        "prefix_context": _prefix_context(prefix_row),
    }
    (output_dir / CONFIG_FILENAME).write_text(
        json.dumps(config, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / SUMMARY_FILENAME).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    candidate_rows.to_csv(output_dir / CANDIDATE_ROWS_FILENAME, index=False)
    rank_rows.to_csv(output_dir / RANK_SUMMARY_ROWS_FILENAME, index=False)
    _write_report(
        output_dir / REPORT_FILENAME,
        summary=summary,
        rank_rows=rank_rows,
        candidate_rows=candidate_rows,
    )
    return summary

def build_parser() -> argparse.ArgumentParser:
    sufficient_config = _load_json(
        DEFAULT_SUFFICIENT_DIR / "post_gate_sufficient_subset_config.json"
    )
    source_config = _load_source_config(DEFAULT_SOURCE_MOVE_DIR)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sufficient-dir", type=Path, default=DEFAULT_SUFFICIENT_DIR)
    parser.add_argument("--gate-trace-dir", type=Path, default=DEFAULT_GATE_TRACE_DIR)
    parser.add_argument("--source-move-dir", type=Path, default=DEFAULT_SOURCE_MOVE_DIR)
    parser.add_argument(
        "--post-gate-dir",
        type=Path,
        default=Path(sufficient_config.get("post_gate_dir", DEFAULT_POST_GATE_DIR)),
    )
    parser.add_argument(
        "--prefix-dir",
        type=Path,
        default=Path(sufficient_config.get("prefix_dir", DEFAULT_PREFIX_DIR)),
    )
    parser.add_argument(
        "--profile-batch-dir",
        type=Path,
        default=Path(
            sufficient_config.get("profile_batch_dir", DEFAULT_PROFILE_BATCH_DIR)
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--candidate-dir", type=Path, action="append", default=None)
    parser.add_argument(
        "--vanilla-dir",
        type=Path,
        default=Path(sufficient_config.get("vanilla_dir", DEFAULT_VANILLA_DIR)),
    )
    parser.add_argument(
        "--pair-id",
        default=str(sufficient_config.get("pair_id", "c0-s11-r0.001")),
    )
    parser.add_argument(
        "--prefix-rank",
        type=int,
        default=int(sufficient_config.get("prefix_rank", 8)),
    )
    parser.add_argument(
        "--source-verdict",
        default=str(
            sufficient_config.get("source_verdict", POST_GATE_VERDICT_NEAR_MISS)
        ),
    )
    parser.add_argument("--source-recovery-policy", default=DEFAULT_SOURCE_RECOVERY_POLICY)
    parser.add_argument(
        "--baseline-iterations",
        type=int,
        default=int(sufficient_config.get("baseline_iterations", 10)),
    )
    parser.add_argument(
        "--candidate-polish-iterations",
        type=int,
        default=int(sufficient_config.get("candidate_polish_iterations", 5)),
    )
    parser.add_argument(
        "--local-polish-iterations",
        type=int,
        default=int(sufficient_config.get("local_polish_iterations", 3)),
    )
    parser.add_argument(
        "--target-action-multiplier",
        type=float,
        default=float(sufficient_config.get("target_action_multiplier", 0.5)),
    )
    parser.add_argument(
        "--max-target-action-nodes",
        type=int,
        default=int(sufficient_config.get("max_target_action_nodes", 64)),
    )
    parser.add_argument("--cumulative-fraction", type=float, default=0.80)
    parser.add_argument("--min-score-fraction", type=float, default=0.05)
    parser.add_argument("--min-gap-fraction", type=float, default=0.25)
    parser.add_argument("--min-guarded-pull-fraction", type=float, default=0.50)
    parser.add_argument(
        "--resolution",
        type=float,
        default=float(sufficient_config.get("resolution", 0.01)),
    )
    parser.add_argument(
        "--randomness",
        type=float,
        default=float(sufficient_config.get("randomness", 0.01)),
    )
    parser.add_argument(
        "--perturb-seed-offset",
        type=int,
        default=int(source_config.get("perturb_seed_offset", 1000)),
    )
    parser.add_argument(
        "--polish-seed-offset",
        type=int,
        default=int(source_config.get("polish_seed_offset", 2000)),
    )
    parser.add_argument(
        "--min-support-shift-from-vanilla",
        type=float,
        default=float(source_config.get("min_support_shift_from_vanilla", 0.01)),
    )
    parser.add_argument(
        "--min-material-q-gain",
        type=float,
        default=float(source_config.get("min_material_q_gain", 0.01)),
    )
    return parser

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    candidate_dirs = (
        tuple(args.candidate_dir)
        if args.candidate_dir
        else tuple(Path(path) for path in DEFAULT_CANDIDATE_DIRS)
    )
    summary = run_profile(
        sufficient_dir=args.sufficient_dir,
        gate_trace_dir=args.gate_trace_dir,
        source_move_dir=args.source_move_dir,
        post_gate_dir=args.post_gate_dir,
        prefix_dir=args.prefix_dir,
        profile_batch_dir=args.profile_batch_dir,
        output_dir=args.output_dir,
        candidate_dirs=candidate_dirs,
        vanilla_dir=args.vanilla_dir,
        pair_id=args.pair_id,
        prefix_rank=args.prefix_rank,
        source_verdict=args.source_verdict,
        source_recovery_policy=args.source_recovery_policy,
        baseline_iterations=args.baseline_iterations,
        candidate_polish_iterations=args.candidate_polish_iterations,
        local_polish_iterations=args.local_polish_iterations,
        target_action_multiplier=args.target_action_multiplier,
        max_target_action_nodes=args.max_target_action_nodes,
        cumulative_fraction=args.cumulative_fraction,
        min_score_fraction=args.min_score_fraction,
        min_gap_fraction=args.min_gap_fraction,
        min_guarded_pull_fraction=args.min_guarded_pull_fraction,
        resolution=args.resolution,
        randomness=args.randomness,
        perturb_seed_offset=args.perturb_seed_offset,
        polish_seed_offset=args.polish_seed_offset,
        min_support_shift_from_vanilla=args.min_support_shift_from_vanilla,
        min_material_q_gain=args.min_material_q_gain,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))

if __name__ == "__main__":
    main()
