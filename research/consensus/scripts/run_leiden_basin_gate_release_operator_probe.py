#!/usr/bin/env python3
"""Probe gate-release recovery actions selected by attachment scores.

This is a diagnostic operator probe.  It starts from the p8 post-gate source
state, selects target nodes from precomputed gate-attachment scores, opens a
source-side gate context, and runs bounded polish.  The goal is to test whether
attachment-margin scoring can explain or improve the previously observed
single-node recovery without forcing a candidate transplant.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT))

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
from profile_leiden_basin_gate_attachment_candidates import (  # noqa: E402
    CANDIDATE_ROWS_FILENAME as ATTACHMENT_CANDIDATE_ROWS_FILENAME,
    DEFAULT_OUTPUT_DIR as DEFAULT_ATTACHMENT_SCORE_DIR,
)
from profile_leiden_basin_post_gate_gate_trace import (  # noqa: E402
    DEFAULT_SOURCE_RECOVERY_POLICY,
    DEFAULT_SUFFICIENT_DIR,
    _selected_gate_nodes,
)
from sciscape.clustering.leiden_basin_profile import changed_support_nodes  # noqa: E402
from sciscape.clustering.leiden_basin_search import (  # noqa: E402
    POST_GATE_VERDICT_NEAR_MISS,
    TransitionAction,
    edge_public_row,
    node_csv,
    unique_sorted_u32,
)
from search_leiden_basin_transitions import (  # noqa: E402
    _evaluate_state,
    _polished_child,
)


DEFAULT_OUTPUT_DIR = DEFAULT_SOURCE_MOVE_DIR.parent / (
    "basin_transition_gate_release_operator_probe_field34_cc_c0_p8_v0"
)

ACTION_GATE_ATTACHMENT_CONTEXT = "recovery_gate_attachment_context"

PROBE_ROWS_FILENAME = "gate_release_operator_probe_rows.csv"
EDGE_ROWS_FILENAME = "gate_release_operator_probe_edges.csv"
SEED_SUMMARY_ROWS_FILENAME = "gate_release_operator_seed_summary_rows.csv"
SUMMARY_FILENAME = "gate_release_operator_probe_summary.json"
CONFIG_FILENAME = "gate_release_operator_probe_config.json"
REPORT_FILENAME = "gate_release_operator_probe_report.md"

SELECTOR_SCORE_COLUMNS = {
    "margin": ("gate_pull_margin_vs_current_source", False),
    "share": ("gate_pull_share_vs_current_source", False),
    "raw_pull": ("pull_to_gate_context", False),
    "consensus": ("rank_mean_consensus", True),
    "per_source_label_node": ("gate_pull_per_source_label_node", False),
}


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in str(value).split(",") if part.strip())


def _parse_int_tuple(value: str, default: tuple[int, ...]) -> tuple[int, ...]:
    text = str(value).strip()
    if not text:
        return default
    if text.lower() in {"none", "null", "-"}:
        return ()
    parsed = _parse_csv_ints(text)
    return parsed or default


def _parse_str_tuple(value: str, default: tuple[str, ...]) -> tuple[str, ...]:
    text = str(value).strip()
    if not text:
        return default
    if text.lower() in {"none", "null", "-"}:
        return ()
    parsed = tuple(part.strip() for part in text.split(",") if part.strip())
    return parsed or default


def _parse_manual_node_sets(value: str) -> tuple[tuple[str, np.ndarray], ...]:
    if not str(value).strip():
        return ()
    out: list[tuple[str, np.ndarray]] = []
    for part in str(value).split(";"):
        if not part.strip():
            continue
        if ":" not in part:
            raise ValueError(
                "manual node sets must use name:node,node;name2:node syntax"
            )
        name, nodes = part.split(":", 1)
        label = name.strip()
        if not label:
            raise ValueError("manual node-set name cannot be empty")
        parsed = np.asarray(_parse_csv_ints(nodes), dtype=np.uint32)
        if parsed.size == 0:
            raise ValueError(f"manual node-set {label} has no nodes")
        out.append((label, unique_sorted_u32(parsed)))
    return tuple(out)


def _load_attachment_scores(path: Path) -> pd.DataFrame:
    rows = pd.read_csv(path / ATTACHMENT_CANDIDATE_ROWS_FILENAME)
    if rows.empty:
        raise ValueError(f"No attachment candidate rows in {path}")
    return rows


def _select_nodes(
    rows: pd.DataFrame,
    *,
    selector: str,
    selected_k: int,
) -> np.ndarray:
    if selector == "trace_moved":
        selected = rows[rows["observed_moved_source_to_gate"].astype(bool)].copy()
        if selected.empty:
            return np.asarray([], dtype=np.uint32)
        return unique_sorted_u32(selected.head(int(selected_k))["node"].to_numpy(np.uint32))
    if selector not in SELECTOR_SCORE_COLUMNS:
        raise ValueError(f"Unknown selector: {selector}")
    score_column, ascending = SELECTOR_SCORE_COLUMNS[selector]
    ordered = rows.sort_values([score_column, "node"], ascending=[ascending, True])
    ordered = ordered.head(int(selected_k))
    return unique_sorted_u32(ordered["node"].to_numpy(dtype=np.uint32))


def _selector_score_summary(rows: pd.DataFrame, nodes: np.ndarray) -> dict[str, Any]:
    selected = rows[rows["node"].astype(int).isin(set(int(node) for node in nodes))]
    if selected.empty:
        return {
            "selected_gate_pull_sum": 0.0,
            "selected_gate_margin_sum": 0.0,
            "selected_current_source_pull_sum": 0.0,
            "selected_min_source_label_size": 0,
            "selected_max_source_label_size": 0,
        }
    return {
        "selected_gate_pull_sum": float(selected["pull_to_gate_context"].sum()),
        "selected_gate_margin_sum": float(
            selected["gate_pull_margin_vs_current_source"].sum()
        ),
        "selected_current_source_pull_sum": float(
            selected["pull_to_current_source_label"].sum()
        ),
        "selected_min_source_label_size": int(selected["source_label_size"].min()),
        "selected_max_source_label_size": int(selected["source_label_size"].max()),
    }


def _make_action(
    *,
    gate_nodes: np.ndarray,
    selected_nodes: np.ndarray,
    selector: str,
    selected_k: int,
    mode: str,
) -> TransitionAction:
    selected = unique_sorted_u32(selected_nodes)
    gate = unique_sorted_u32(gate_nodes)
    if mode == "gate_only":
        context = gate
    elif mode == "target_only":
        context = selected
    elif mode == "gate_plus_target":
        context = unique_sorted_u32(np.concatenate([gate, selected]))
    else:
        raise ValueError(f"Unknown action mode: {mode}")
    return TransitionAction(
        action_type=ACTION_GATE_ATTACHMENT_CONTEXT,
        action_params=(
            f"selector={selector};selected_k={int(selected_k)};"
            f"mode={mode};gate_k={int(gate.size)};target_k={int(selected.size)}"
        ),
        context_nodes=context,
        action_nodes=None,
    )


def _seed_summary_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    out: list[dict[str, Any]] = []
    group_columns = ["selector", "selected_k", "action_mode"]
    for key, group in rows.groupby(group_columns, sort=True):
        selector, selected_k, action_mode = key
        q_gain = group["gate_release_delta_q_gain"].astype(float)
        support = group["state_support_distance_to_vanilla"].astype(float)
        progress = group["state_target_progress_from_vanilla"].astype(float)
        retained = group["gate_release_verdict"].astype(str).eq(
            "q_gain_support_retained"
        )
        out.append(
            {
                "selector": str(selector),
                "selected_k": int(selected_k),
                "action_mode": str(action_mode),
                "seed_count": int(group["gate_release_seed"].nunique()),
                "q_gain_mean": float(q_gain.mean()),
                "q_gain_min": float(q_gain.min()),
                "q_gain_max": float(q_gain.max()),
                "q_gain_std": float(q_gain.std(ddof=0)),
                "support_mean": float(support.mean()),
                "support_min": float(support.min()),
                "support_max": float(support.max()),
                "progress_mean": float(progress.mean()),
                "progress_min": float(progress.min()),
                "progress_max": float(progress.max()),
                "mutable_node_count_min": int(group["mutable_node_count"].min()),
                "mutable_node_count_max": int(group["mutable_node_count"].max()),
                "context_node_count_min": int(group["context_node_count"].min()),
                "context_node_count_max": int(group["context_node_count"].max()),
                "q_gain_support_retained_count": int(retained.sum()),
                "changed_trace_moved_count_sum": int(
                    group["changed_trace_moved_count"].astype(int).sum()
                ),
                "all_seed_q_gain_support_retained": bool(retained.all()),
            }
        )
    return pd.DataFrame(out).sort_values(
        [
            "all_seed_q_gain_support_retained",
            "q_gain_mean",
            "q_gain_min",
            "support_min",
            "mutable_node_count_max",
        ],
        ascending=[False, False, False, False, True],
    )


def _write_report(
    path: Path,
    *,
    summary: dict[str, Any],
    rows: pd.DataFrame,
    seed_rows: pd.DataFrame,
) -> None:
    best_cols = [
        "selector",
        "selected_k",
        "action_mode",
        "selected_node_ids",
        "selected_contains_trace_moved",
        "aligned_changed_node_count",
        "changed_trace_moved_count",
        "gate_release_delta_q_gain",
        "state_delta_q_vs_start",
        "state_support_distance_to_vanilla",
        "state_target_progress_from_vanilla",
        "mutable_node_count",
        "context_node_count",
        "elapsed_sec",
        "gate_release_verdict",
    ]
    lines = [
        "# Gate Release Operator Probe",
        "",
        "This diagnostic opens source-side gate context and selected target nodes as",
        "mutable context, then runs bounded polish without candidate transplant.",
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
        "source_delta_q_vs_start",
        "gate_node_count",
        "target_node_count",
        "trace_moved_node_ids",
        "row_count",
        "best_q_gain_selector",
        "best_q_gain_selected_k",
        "best_q_gain_action_mode",
        "best_q_gain",
        "best_q_gain_state_delta_q",
        "best_q_gain_support",
        "best_q_gain_progress",
        "best_trace_changed_selector",
        "best_trace_changed_selected_k",
        "best_trace_changed_action_mode",
    ]:
        lines.append(f"| {key} | {summary.get(key, '')} |")
    lines.extend(["", "## Best By Q Gain", ""])
    if rows.empty:
        lines.append("_No rows._")
    else:
        best_q = rows.sort_values(
            [
                "gate_release_delta_q_gain",
                "state_delta_q_vs_start",
                "state_support_distance_to_vanilla",
                "state_target_progress_from_vanilla",
                "mutable_node_count",
            ],
            ascending=[False, False, False, False, True],
        ).head(20)
        lines.extend(_markdown_table(best_q[best_cols], max_rows=20))
    lines.extend(["", "## Rows Changing Trace-Moved Nodes", ""])
    trace_changed = rows[rows["changed_trace_moved_count"].astype(int).gt(0)].copy()
    if trace_changed.empty:
        lines.append("_No probe row semantically changed the trace-moved target node._")
    else:
        trace_changed = trace_changed.sort_values(
            [
                "changed_trace_moved_count",
                "gate_release_delta_q_gain",
                "state_delta_q_vs_start",
            ],
            ascending=[False, False, False],
        )
        lines.extend(_markdown_table(trace_changed[best_cols], max_rows=20))
    lines.extend(["", "## Seed Summary", ""])
    if seed_rows.empty:
        lines.append("_No seed summary rows._")
    else:
        seed_cols = [
            "selector",
            "selected_k",
            "action_mode",
            "seed_count",
            "q_gain_mean",
            "q_gain_min",
            "q_gain_max",
            "q_gain_std",
            "support_min",
            "progress_min",
            "q_gain_support_retained_count",
            "changed_trace_moved_count_sum",
            "all_seed_q_gain_support_retained",
        ]
        display_cols = [column for column in seed_cols if column in seed_rows]
        lines.extend(_markdown_table(seed_rows[display_cols], max_rows=30))
    lines.extend(
        [
            "",
            "## Reading",
            "",
            "- `gate_only` is the narrowed 209-node gate context control.",
            "- `target_only` tests whether selected target nodes can move without opening the gate.",
            "- `gate_plus_target` tests whether attachment-margin target nodes add value beyond the gate.",
            "- This remains diagnostic until seed controls and cost-adjusted comparisons are run.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_probe(
    *,
    attachment_score_dir: Path,
    sufficient_dir: Path,
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
    selectors: tuple[str, ...],
    selected_ks: tuple[int, ...],
    action_modes: tuple[str, ...],
    manual_node_sets: tuple[tuple[str, np.ndarray], ...],
    include_gate_only_control: bool,
    baseline_iterations: int,
    candidate_polish_iterations: int,
    local_polish_iterations: int,
    recovery_polish_iterations: int,
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
    recovery_seed_offset: int,
    recovery_seeds: tuple[int, ...],
    min_support_shift_from_vanilla: float,
    min_material_q_gain: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    attachment_rows = _load_attachment_scores(attachment_score_dir)
    gate_nodes, _ = _selected_gate_nodes(sufficient_dir)
    source_moves = pd.read_csv(source_move_dir / SOURCE_MOVE_ROWS_FILENAME)
    _, source_recovery_index = _select_source_move(
        source_moves,
        recovery_policy=source_recovery_policy,
    )
    trace_moved_nodes = unique_sorted_u32(
        attachment_rows[attachment_rows["observed_moved_source_to_gate"].astype(bool)][
            "node"
        ].to_numpy(dtype=np.uint32)
    )

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
    default_recovery_seed = int(recovery_seed_offset) + int(source_recovery_index)
    effective_recovery_seeds = (
        tuple(int(seed) for seed in recovery_seeds)
        if recovery_seeds
        else (int(default_recovery_seed),)
    )
    base_context = {
        **case_ctx["public_context"],
        **_prefix_context(prefix_row),
        "path_policy": "gate_release_operator_probe",
        "selection_policy": "gate_attachment_score",
        "escalation_reason": "diagnostic_gate_release",
        "target_stage_index": int(source_row.get("target_stage_index", 0)),
        "recovery_policy": source_recovery_policy,
        "recovery_source_state_id": source_state.state_id,
    }

    rows: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    trial_index = 0

    def run_action(
        *,
        selector: str,
        selected_k: int,
        mode: str,
        selected_nodes: np.ndarray,
        recovery_seed: int,
    ) -> None:
        nonlocal trial_index
        trial_index += 1
        action = _make_action(
            gate_nodes=gate_nodes,
            selected_nodes=selected_nodes,
            selector=selector,
            selected_k=selected_k,
            mode=mode,
        )
        child = _polished_child(
            parent=source_state,
            action=action,
            graph=case_ctx["graph"],
            donor_membership=case_ctx["candidate"].recreated.membership,
            resolution=resolution,
            seed=recovery_seed,
            n_iterations=recovery_polish_iterations,
            randomness=randomness,
            child_index=trial_index,
        )
        changed = changed_support_nodes(source_state.membership, child.membership)
        changed_set = set(int(node) for node in changed)
        trace_set = set(int(node) for node in trace_moved_nodes)
        selected_set = set(int(node) for node in selected_nodes)
        score_summary = _selector_score_summary(attachment_rows, selected_nodes)
        row = _evaluate_state(
            state=child,
            baseline_membership=case_ctx["baseline"].membership,
            candidate_membership=case_ctx["candidate"].recreated.membership,
            vanilla_membership=case_ctx["vanilla"].membership,
            sketch_nodes=case_ctx["sketch_nodes"],
            start_quality=case_ctx["vanilla"].quality,
            candidate_quality=case_ctx["candidate"].recreated.quality,
            vanilla_quality=case_ctx["vanilla"].quality,
            vanilla_support_distance_to_candidate=case_ctx[
                "vanilla_support_distance_to_candidate"
            ],
            context={
                **base_context,
                "selector": selector,
                "selected_k": int(selected_k),
                "action_mode": mode,
                "selected_node_ids": node_csv(selected_nodes),
                "selected_contains_trace_moved": bool(trace_set & selected_set),
                "selected_trace_moved_count": int(len(trace_set & selected_set)),
                **score_summary,
            },
            parent_row=source_row,
            min_support_shift_from_vanilla=min_support_shift_from_vanilla,
            min_material_q_gain=min_material_q_gain,
        )
        delta_q_gain = float(row["state_delta_q_vs_start"]) - float(
            source_row["state_delta_q_vs_start"]
        )
        support_gain = float(row["state_support_distance_to_vanilla"]) - float(
            source_row["state_support_distance_to_vanilla"]
        )
        progress_gain = float(row["state_target_progress_from_vanilla"]) - float(
            source_row["state_target_progress_from_vanilla"]
        )
        q_recovered = delta_q_gain >= float(min_material_q_gain)
        support_retained = float(row["state_support_distance_to_vanilla"]) >= float(
            source_row["state_support_distance_to_vanilla"]
        )
        verdict = (
            "q_gain_support_retained"
            if q_recovered and support_retained
            else "q_gain_support_lost"
            if q_recovered
            else "support_deepened_quality_loss"
            if support_gain > 0.0 and delta_q_gain < 0.0
            else "plateau"
            if abs(delta_q_gain) < 1e-12 and abs(support_gain) < 1e-12
            else "quality_loss"
        )
        row.update(
            {
                "gate_release_trial_index": int(trial_index),
                "gate_release_seed": int(recovery_seed),
                "gate_node_count": int(gate_nodes.size),
                "trace_moved_node_ids": node_csv(trace_moved_nodes),
                "changed_node_count": int(changed.size),
                "changed_node_ids": node_csv(changed),
                "aligned_changed_node_count": int(changed.size),
                "aligned_changed_node_ids": node_csv(changed),
                "changed_trace_moved_count": int(len(changed_set & trace_set)),
                "changed_trace_moved_node_ids": node_csv(
                    np.asarray(sorted(changed_set & trace_set), dtype=np.uint32)
                ),
                "gate_release_delta_q_gain": delta_q_gain,
                "gate_release_support_gain": support_gain,
                "gate_release_target_progress_gain": progress_gain,
                "gate_release_q_recovered": bool(q_recovered),
                "gate_release_support_retained": bool(support_retained),
                "gate_release_verdict": verdict,
                "path_elapsed_sec": float(source_row.get("path_elapsed_sec", 0.0))
                + float(child.elapsed_sec),
            }
        )
        rows.append(row)
        edges.append(
            edge_public_row(
                parent_state_id=source_state.state_id,
                child_state_id=child.state_id,
                action=action,
                context={
                    **case_ctx["public_context"],
                    "path_policy": "gate_release_operator_probe",
                    "selector": selector,
                    "selected_k": int(selected_k),
                    "action_mode": mode,
                    "gate_release_seed": int(recovery_seed),
                },
            )
        )

    if include_gate_only_control:
        for seed in effective_recovery_seeds:
            run_action(
                selector="gate_only_control",
                selected_k=0,
                mode="gate_only",
                selected_nodes=np.asarray([], dtype=np.uint32),
                recovery_seed=int(seed),
            )
    for seed in effective_recovery_seeds:
        for selector in selectors:
            for selected_k in selected_ks:
                selected = _select_nodes(
                    attachment_rows,
                    selector=selector,
                    selected_k=int(selected_k),
                )
                if selected.size == 0:
                    continue
                for mode in action_modes:
                    run_action(
                        selector=selector,
                        selected_k=int(selected_k),
                        mode=mode,
                        selected_nodes=selected,
                        recovery_seed=int(seed),
                    )
        for manual_name, manual_nodes in manual_node_sets:
            for mode in action_modes:
                run_action(
                    selector=f"manual_{manual_name}",
                    selected_k=int(manual_nodes.size),
                    mode=mode,
                    selected_nodes=manual_nodes,
                    recovery_seed=int(seed),
                )

    row_frame = pd.DataFrame(rows)
    edge_frame = pd.DataFrame(edges)
    seed_summary = _seed_summary_rows(row_frame)
    if row_frame.empty:
        summary: dict[str, Any] = {
            "schema": "leiden_basin_gate_release_operator_probe.v0",
            "output_dir": str(output_dir),
            "row_count": 0,
            "seed_count": int(len(effective_recovery_seeds)),
            "recovery_seeds": [int(seed) for seed in effective_recovery_seeds],
        }
    else:
        best_q = row_frame.sort_values(
            [
                "gate_release_delta_q_gain",
                "state_delta_q_vs_start",
                "state_support_distance_to_vanilla",
                "state_target_progress_from_vanilla",
                "mutable_node_count",
            ],
            ascending=[False, False, False, False, True],
        ).iloc[0]
        trace_changed = row_frame[
            row_frame["changed_trace_moved_count"].astype(int).gt(0)
        ].copy()
        best_trace = None
        if not trace_changed.empty:
            best_trace = trace_changed.sort_values(
                [
                    "changed_trace_moved_count",
                    "gate_release_delta_q_gain",
                    "state_delta_q_vs_start",
                ],
                ascending=[False, False, False],
            ).iloc[0]
        summary = {
            "schema": "leiden_basin_gate_release_operator_probe.v0",
            "output_dir": str(output_dir),
            "attachment_score_dir": str(attachment_score_dir),
            "sufficient_dir": str(sufficient_dir),
            "source_move_dir": str(source_move_dir),
            "pair_id": pair_id,
            "prefix_rank": int(prefix_rank),
            "source_delta_q_vs_start": float(source_row["state_delta_q_vs_start"]),
            "source_support_distance_to_vanilla": float(
                source_row["state_support_distance_to_vanilla"]
            ),
            "source_target_progress_from_vanilla": float(
                source_row["state_target_progress_from_vanilla"]
            ),
            "gate_node_count": int(gate_nodes.size),
            "target_node_count": int(attachment_rows.shape[0]),
            "trace_moved_node_ids": node_csv(trace_moved_nodes),
            "row_count": int(row_frame.shape[0]),
            "seed_count": int(len(effective_recovery_seeds)),
            "recovery_seeds": [int(seed) for seed in effective_recovery_seeds],
            "best_q_gain_selector": str(best_q["selector"]),
            "best_q_gain_selected_k": int(best_q["selected_k"]),
            "best_q_gain_action_mode": str(best_q["action_mode"]),
            "best_q_gain_seed": int(best_q["gate_release_seed"]),
            "best_q_gain": float(best_q["gate_release_delta_q_gain"]),
            "best_q_gain_state_delta_q": float(best_q["state_delta_q_vs_start"]),
            "best_q_gain_support": float(best_q["state_support_distance_to_vanilla"]),
            "best_q_gain_progress": float(best_q["state_target_progress_from_vanilla"]),
            "best_q_gain_mutable_node_count": int(best_q["mutable_node_count"]),
            "best_q_gain_context_node_count": int(best_q["context_node_count"]),
            "best_trace_changed_selector": (
                None if best_trace is None else str(best_trace["selector"])
            ),
            "best_trace_changed_selected_k": (
                None if best_trace is None else int(best_trace["selected_k"])
            ),
            "best_trace_changed_action_mode": (
                None if best_trace is None else str(best_trace["action_mode"])
            ),
            "best_trace_changed_seed": (
                None if best_trace is None else int(best_trace["gate_release_seed"])
            ),
            "trace_changed_row_count": int(trace_changed.shape[0]),
        }
    config = {
        "attachment_score_dir": str(attachment_score_dir),
        "sufficient_dir": str(sufficient_dir),
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
        "selectors": list(selectors),
        "selected_ks": [int(value) for value in selected_ks],
        "action_modes": list(action_modes),
        "manual_node_sets": [
            {"name": name, "node_ids": node_csv(nodes)}
            for name, nodes in manual_node_sets
        ],
        "include_gate_only_control": bool(include_gate_only_control),
        "baseline_iterations": int(baseline_iterations),
        "candidate_polish_iterations": int(candidate_polish_iterations),
        "local_polish_iterations": int(local_polish_iterations),
        "recovery_polish_iterations": int(recovery_polish_iterations),
        "recovery_seed_offset": int(recovery_seed_offset),
        "default_recovery_seed": int(default_recovery_seed),
        "recovery_seeds": [int(seed) for seed in effective_recovery_seeds],
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
    row_frame.to_csv(output_dir / PROBE_ROWS_FILENAME, index=False)
    edge_frame.to_csv(output_dir / EDGE_ROWS_FILENAME, index=False)
    seed_summary.to_csv(output_dir / SEED_SUMMARY_ROWS_FILENAME, index=False)
    _write_report(
        output_dir / REPORT_FILENAME,
        summary=summary,
        rows=row_frame,
        seed_rows=seed_summary,
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    source_config = _load_source_config(DEFAULT_SOURCE_MOVE_DIR)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--attachment-score-dir", type=Path, default=DEFAULT_ATTACHMENT_SCORE_DIR
    )
    parser.add_argument("--sufficient-dir", type=Path, default=DEFAULT_SUFFICIENT_DIR)
    parser.add_argument("--source-move-dir", type=Path, default=DEFAULT_SOURCE_MOVE_DIR)
    parser.add_argument("--post-gate-dir", type=Path, default=DEFAULT_POST_GATE_DIR)
    parser.add_argument("--prefix-dir", type=Path, default=DEFAULT_PREFIX_DIR)
    parser.add_argument("--profile-batch-dir", type=Path, default=DEFAULT_PROFILE_BATCH_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--candidate-dir", type=Path, action="append", default=None)
    parser.add_argument("--vanilla-dir", type=Path, default=DEFAULT_VANILLA_DIR)
    parser.add_argument("--pair-id", default="c0-s11-r0.001")
    parser.add_argument("--prefix-rank", type=int, default=8)
    parser.add_argument("--source-verdict", default=POST_GATE_VERDICT_NEAR_MISS)
    parser.add_argument("--source-recovery-policy", default=DEFAULT_SOURCE_RECOVERY_POLICY)
    parser.add_argument(
        "--selectors",
        default="margin,raw_pull,consensus,trace_moved",
    )
    parser.add_argument("--selected-ks", default="1,2,4,8")
    parser.add_argument("--action-modes", default="target_only,gate_plus_target")
    parser.add_argument(
        "--manual-node-sets",
        default="",
        help="Optional semicolon list like n2890:2890;n7325:7325;n_pair:2890,7325",
    )
    parser.add_argument("--no-gate-only-control", action="store_true")
    parser.add_argument("--baseline-iterations", type=int, default=10)
    parser.add_argument("--candidate-polish-iterations", type=int, default=5)
    parser.add_argument(
        "--local-polish-iterations",
        type=int,
        default=int(source_config.get("local_polish_iterations", 3)),
    )
    parser.add_argument(
        "--recovery-polish-iterations",
        type=int,
        default=int(source_config.get("recovery_polish_iterations", 10)),
    )
    parser.add_argument("--target-action-multiplier", type=float, default=0.5)
    parser.add_argument("--max-target-action-nodes", type=int, default=64)
    parser.add_argument("--cumulative-fraction", type=float, default=0.80)
    parser.add_argument("--min-score-fraction", type=float, default=0.05)
    parser.add_argument("--min-gap-fraction", type=float, default=0.25)
    parser.add_argument("--min-guarded-pull-fraction", type=float, default=0.50)
    parser.add_argument("--resolution", type=float, default=0.01)
    parser.add_argument("--randomness", type=float, default=0.01)
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
        "--recovery-seed-offset",
        type=int,
        default=int(source_config.get("recovery_seed_offset", 21000)),
    )
    parser.add_argument(
        "--recovery-seeds",
        default="",
        help="Optional comma-separated actual recovery seeds; overrides recovery-seed-offset.",
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
    summary = run_probe(
        attachment_score_dir=args.attachment_score_dir,
        sufficient_dir=args.sufficient_dir,
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
        selectors=_parse_str_tuple(args.selectors, ("margin", "raw_pull", "consensus")),
        selected_ks=_parse_int_tuple(args.selected_ks, (1, 2, 4, 8, 16)),
        action_modes=_parse_str_tuple(args.action_modes, ("target_only", "gate_plus_target")),
        manual_node_sets=_parse_manual_node_sets(args.manual_node_sets),
        include_gate_only_control=not bool(args.no_gate_only_control),
        baseline_iterations=args.baseline_iterations,
        candidate_polish_iterations=args.candidate_polish_iterations,
        local_polish_iterations=args.local_polish_iterations,
        recovery_polish_iterations=args.recovery_polish_iterations,
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
        recovery_seed_offset=args.recovery_seed_offset,
        recovery_seeds=_parse_int_tuple(args.recovery_seeds, ()),
        min_support_shift_from_vanilla=args.min_support_shift_from_vanilla,
        min_material_q_gain=args.min_material_q_gain,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
