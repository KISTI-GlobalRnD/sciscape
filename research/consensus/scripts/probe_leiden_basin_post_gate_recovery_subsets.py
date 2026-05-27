#!/usr/bin/env python3
"""Probe partial post-gate recovery context subsets.

This diagnostic starts from the same post-gate near-miss state used by the
full-context recovery move probe, then replays smaller slices of the successful
full vanilla-closure context action.  The goal is to measure whether recovery
signal appears gradually, only after a large context release, or in a narrow
rank band.
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
    COMBINED_DIR,
    CONFIG_FILENAME as SOURCE_CONFIG_FILENAME,
    DEFAULT_CANDIDATE_DIRS,
    DEFAULT_POST_GATE_DIR,
    DEFAULT_PROFILE_BATCH_DIR,
    DEFAULT_VANILLA_DIR,
    MOVE_ROWS_FILENAME as SOURCE_MOVE_ROWS_FILENAME,
    POST_GATE_PATH_SUMMARY_FILENAME,
    _load_case_context,
    _markdown_table,
    _prefix_context,
    _replay_to_source_state,
    _select_source_path,
)
from sciscape.clustering.leiden_basin_profile import parse_node_ids  # noqa: E402
from sciscape.clustering.leiden_basin_search import (  # noqa: E402
    ACTION_RECOVERY_VANILLA_CONTEXT_TOPK,
    POST_GATE_RECOVERY_MOVE_Q_GAIN,
    POST_GATE_RECOVERY_MOVE_RECOVERED,
    POST_GATE_VERDICT_NEAR_MISS,
    TransitionAction,
    annotate_pathway_debt_area_rows,
    annotate_post_gate_recovery_step_rows,
    annotate_tunneling_evidence_rows,
    classify_post_gate_recovery_move_rows,
    compute_pathway_wall_rows,
    edge_public_row,
    node_csv,
    summarize_post_gate_recovery_paths,
    trace_tunneling_path_states,
    unique_sorted_u32,
    weighted_pull_to_nodes,
)
from search_leiden_basin_transitions import (  # noqa: E402
    _evaluate_state,
    _polished_child,
)


DEFAULT_SOURCE_MOVE_DIR = (
    COMBINED_DIR / "basin_transition_post_gate_recovery_moves_field34_cc_c0_p8_fullctx_v0"
)
DEFAULT_OUTPUT_DIR = (
    COMBINED_DIR / "basin_transition_post_gate_recovery_subsets_field34_cc_c0_p8_v0"
)

STATE_ROWS_FILENAME = "post_gate_recovery_subset_states.csv"
EDGE_ROWS_FILENAME = "post_gate_recovery_subset_edges.csv"
SUBSET_ROWS_FILENAME = "post_gate_recovery_subset_rows.csv"
PATH_ROWS_FILENAME = "post_gate_recovery_subset_path_rows.csv"
TRACE_ROWS_FILENAME = "post_gate_recovery_subset_trace_rows.csv"
STEP_ROWS_FILENAME = "post_gate_recovery_subset_step_rows.csv"
SUMMARY_ROWS_FILENAME = "post_gate_recovery_subset_path_summary_rows.csv"
SUMMARY_FILENAME = "post_gate_recovery_subset_summary.json"
CONFIG_FILENAME = "post_gate_recovery_subset_config.json"
REPORT_FILENAME = "post_gate_recovery_subset_report.md"

DEFAULT_SOURCE_RECOVERY_POLICY = "vanilla_closure_topk:context_only"
DEFAULT_SUBSET_SIZES = (16, 32, 64, 96, 128, 192, 256, 320, 384, 436)


def _parse_int_tuple(value: str, default: tuple[int, ...]) -> tuple[int, ...]:
    if not str(value).strip():
        return default
    out = tuple(int(part.strip()) for part in str(value).split(",") if part.strip())
    return out or default


def _load_source_config(source_move_dir: Path) -> dict[str, Any]:
    path = source_move_dir / SOURCE_CONFIG_FILENAME
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _select_source_move(
    move_rows: pd.DataFrame,
    *,
    recovery_policy: str,
) -> tuple[pd.Series, int]:
    rows = move_rows[move_rows["recovery_policy"].astype(str).eq(str(recovery_policy))]
    if rows.empty:
        raise ValueError(f"No recovery move row for policy={recovery_policy}")
    rows = rows.copy()
    rows["_source_order"] = rows.index.astype(int) + 1
    rows = rows.sort_values(
        [
            "post_gate_move_delta_q_gain",
            "state_delta_q_vs_start",
            "state_support_distance_to_vanilla",
            "state_target_progress_from_vanilla",
        ],
        ascending=[False, False, False, False],
    )
    row = rows.iloc[0]
    return row.drop(labels=["_source_order"]), int(row["_source_order"])


def _rank_selected_nodes(
    *,
    selected_nodes: np.ndarray,
    source_state: Any,
    arrays: Any,
    node_count: int,
) -> pd.DataFrame:
    action_nodes = unique_sorted_u32(source_state.action_nodes)
    direct_nodes = (
        action_nodes if action_nodes.size else unique_sorted_u32(source_state.direct_nodes)
    )
    pull = weighted_pull_to_nodes(
        src=np.asarray(arrays.src, dtype=np.uint32),
        dst=np.asarray(arrays.dst, dtype=np.uint32),
        weight=np.asarray(arrays.weight, dtype=np.float64),
        target_nodes=direct_nodes,
        node_count=node_count,
    )
    nodes = unique_sorted_u32(selected_nodes)
    frame = pd.DataFrame(
        {
            "node": nodes.astype(np.uint32),
            "pull_score": pull[nodes.astype(np.int64)],
        }
    )
    frame = frame.sort_values(
        ["pull_score", "node"],
        ascending=[False, True],
    ).reset_index(drop=True)
    frame["pull_rank"] = np.arange(1, len(frame) + 1, dtype=np.int64)
    return frame


def _subset_specs(
    *,
    ranked_nodes: pd.DataFrame,
    subset_sizes: tuple[int, ...],
    include_bands: bool,
) -> list[dict[str, Any]]:
    node_count = int(len(ranked_nodes))
    sizes = sorted({int(size) for size in subset_sizes if 0 < int(size) <= node_count})
    if node_count and (not sizes or sizes[-1] != node_count):
        sizes.append(node_count)
    specs: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()

    def add(policy: str, start_rank: int, end_rank: int) -> None:
        if end_rank < start_rank:
            return
        key = (policy, int(start_rank), int(end_rank))
        if key in seen:
            return
        seen.add(key)
        segment = ranked_nodes.iloc[int(start_rank) - 1 : int(end_rank)]
        nodes = np.asarray(segment["node"], dtype=np.uint32)
        if nodes.size == 0:
            return
        specs.append(
            {
                "subset_policy": policy,
                "subset_rank_start": int(start_rank),
                "subset_rank_end": int(end_rank),
                "selected_nodes": nodes,
                "selected_pull_sum": float(segment["pull_score"].sum()),
                "selected_pull_mean": float(segment["pull_score"].mean()),
                "selected_pull_min": float(segment["pull_score"].min()),
                "selected_pull_max": float(segment["pull_score"].max()),
            }
        )

    for size in sizes:
        add("pull_prefix", 1, int(size))
    if include_bands:
        previous = 0
        for size in sizes:
            add("pull_band", previous + 1, int(size))
            previous = int(size)
    return specs


def _write_report(
    path: Path,
    *,
    summary: dict[str, Any],
    source_move: pd.Series,
    subset_rows: pd.DataFrame,
    recovery_path_rows: pd.DataFrame,
) -> None:
    lines = [
        "# Post-Gate Recovery Subset Probe",
        "",
        "This diagnostic narrows the full vanilla-closure context release into",
        "ranked partial subsets.  It is diagnostic-only and should not be read as",
        "a default Dongdaemun operator.",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "| --- | --- |",
    ]
    for key in [
        "output_dir",
        "source_pair_id",
        "source_prefix_rank",
        "source_recovery_policy",
        "source_full_selected_node_count",
        "source_delta_q",
        "source_support",
        "subset_rows",
        "q_recovered_support_retained_rows",
        "q_gain_support_retained_rows",
        "best_delta_q_gain",
        "best_delta_q",
        "best_subset_policy",
        "best_subset_rank_end",
        "min_prefix_q_gain_size",
    ]:
        lines.append(f"| {key} | {summary.get(key, '')} |")
    lines.extend(["", "## Source Full Move", ""])
    source_cols = [
        "recovery_policy",
        "post_gate_move_verdict",
        "post_gate_move_delta_q_gain",
        "state_delta_q_vs_start",
        "state_support_distance_to_vanilla",
        "state_target_progress_from_vanilla",
        "recovery_selected_node_count",
        "mutable_node_count",
        "state_id",
    ]
    lines.extend(
        _markdown_table(
            pd.DataFrame([source_move])[
                [column for column in source_cols if column in source_move.index]
            ]
        )
    )
    lines.extend(["", "## Subset Rows", ""])
    subset_cols = [
        "subset_policy",
        "subset_rank_start",
        "subset_rank_end",
        "recovery_selected_node_count",
        "post_gate_move_verdict",
        "post_gate_move_delta_q_gain",
        "state_delta_q_vs_start",
        "post_gate_move_support_gain",
        "state_support_distance_to_vanilla",
        "post_gate_move_target_progress_gain",
        "state_target_progress_from_vanilla",
        "mutable_node_count",
        "selected_pull_sum",
        "elapsed_sec",
    ]
    display = subset_rows.sort_values(
        ["subset_policy", "subset_rank_end", "subset_rank_start"],
        ascending=[True, True, True],
    )
    lines.extend(
        _markdown_table(
            display[[column for column in subset_cols if column in display]],
            max_rows=120,
        )
    )
    lines.extend(["", "## Best Rows", ""])
    best = subset_rows.sort_values(
        [
            "post_gate_move_q_recovered",
            "post_gate_move_delta_q_gain",
            "state_support_distance_to_vanilla",
            "state_target_progress_from_vanilla",
            "recovery_selected_node_count",
        ],
        ascending=[False, False, False, False, True],
    ).head(10)
    lines.extend(
        _markdown_table(best[[column for column in subset_cols if column in best]], max_rows=10)
    )
    lines.extend(["", "## Recovery Path Rows", ""])
    path_cols = [
        "path_final_state_id",
        "path_q_wall",
        "path_q_debt_area_step",
        "path_final_delta_q_vs_start",
        "path_final_support_distance_to_vanilla",
        "path_final_target_progress_from_vanilla",
        "path_final_mutable_node_count",
        "tunnel_route_label",
    ]
    lines.extend(
        _markdown_table(
            recovery_path_rows[[column for column in path_cols if column in recovery_path_rows]],
            max_rows=120,
        )
    )
    lines.extend(
        [
            "",
            "## Reading",
            "",
            "- `pull_prefix` asks how much ranked context is needed before QF debt improves.",
            "- `pull_band` asks whether a narrow rank band carries a recovery signal by itself.",
            "- A positive QF gain with retained support means the opened search region is larger than the narrow gate path; it is not yet a recovered basin transition.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_probe(
    *,
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
    subset_sizes: tuple[int, ...],
    include_bands: bool,
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
    min_support_shift_from_vanilla: float,
    min_material_q_gain: float,
    support_gate: float,
    progress_margin: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    source_moves = pd.read_csv(source_move_dir / SOURCE_MOVE_ROWS_FILENAME)
    source_move, source_recovery_index = _select_source_move(
        source_moves,
        recovery_policy=source_recovery_policy,
    )
    selected_nodes = parse_node_ids(source_move["selected_node_ids"])
    if selected_nodes.size == 0:
        raise ValueError("Source recovery move does not contain selected_node_ids")

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
    source_state, source_row, replay_rows, replay_edges = _replay_to_source_state(
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
    expected_source_id = str(source_move.get("recovery_source_state_id", ""))
    if expected_source_id and expected_source_id != source_state.state_id:
        raise RuntimeError(
            "Replayed source state does not match source move artifact: "
            f"{source_state.state_id} != {expected_source_id}"
        )

    ranked_nodes = _rank_selected_nodes(
        selected_nodes=selected_nodes,
        source_state=source_state,
        arrays=case_ctx["arrays"],
        node_count=int(case_ctx["baseline"].membership.size),
    )
    specs = _subset_specs(
        ranked_nodes=ranked_nodes,
        subset_sizes=subset_sizes,
        include_bands=include_bands,
    )
    subset_seed = int(recovery_seed_offset) + int(source_recovery_index)
    subset_rows: list[dict[str, Any]] = []
    subset_edges: list[dict[str, Any]] = []
    for index, spec in enumerate(specs, start=1):
        selected = unique_sorted_u32(spec["selected_nodes"])
        action = TransitionAction(
            action_type=ACTION_RECOVERY_VANILLA_CONTEXT_TOPK,
            action_params=(
                f"source_recovery_policy={source_recovery_policy};"
                f"subset_policy={spec['subset_policy']};"
                f"subset_rank_start={int(spec['subset_rank_start'])};"
                f"subset_rank_end={int(spec['subset_rank_end'])};"
                f"selected_k={int(selected.size)}"
            ),
            context_nodes=selected,
            action_nodes=None,
        )
        child = _polished_child(
            parent=source_state,
            action=action,
            graph=case_ctx["graph"],
            donor_membership=case_ctx["candidate"].recreated.membership,
            resolution=resolution,
            seed=subset_seed,
            n_iterations=recovery_polish_iterations,
            randomness=randomness,
            child_index=index,
        )
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
                **case_ctx["public_context"],
                **_prefix_context(prefix_row),
                "path_policy": "post_gate_recovery_subset",
                "selection_policy": source_recovery_policy,
                "escalation_reason": "post_gate_subset_probe",
                "target_stage_index": int(source_row.get("target_stage_index", 0)),
                "selected_k": int(selected.size),
                "selected_node_ids": node_csv(selected),
                "recovery_policy": source_recovery_policy,
                "recovery_source_action_type": "vanilla_closure_topk",
                "recovery_move_kind": "context_only",
                "recovery_selected_node_count": int(selected.size),
                "recovery_context_node_count": int(selected.size),
                "recovery_action_node_count": 0,
                "recovery_source_state_id": source_state.state_id,
                "subset_policy": spec["subset_policy"],
                "subset_rank_start": int(spec["subset_rank_start"]),
                "subset_rank_end": int(spec["subset_rank_end"]),
                "source_full_selected_node_count": int(selected_nodes.size),
                "selected_pull_sum": float(spec["selected_pull_sum"]),
                "selected_pull_mean": float(spec["selected_pull_mean"]),
                "selected_pull_min": float(spec["selected_pull_min"]),
                "selected_pull_max": float(spec["selected_pull_max"]),
            },
            parent_row=source_row,
            min_support_shift_from_vanilla=min_support_shift_from_vanilla,
            min_material_q_gain=min_material_q_gain,
        )
        row["path_elapsed_sec"] = float(source_row.get("path_elapsed_sec", 0.0)) + float(
            child.elapsed_sec
        )
        subset_rows.append(row)
        subset_edges.append(
            edge_public_row(
                parent_state_id=source_state.state_id,
                child_state_id=child.state_id,
                action=action,
                context={
                    **case_ctx["public_context"],
                    "path_policy": "post_gate_recovery_subset",
                    "recovery_policy": source_recovery_policy,
                    "subset_policy": spec["subset_policy"],
                    "subset_rank_start": int(spec["subset_rank_start"]),
                    "subset_rank_end": int(spec["subset_rank_end"]),
                },
            )
        )

    subsets = pd.DataFrame(subset_rows)
    subsets = classify_post_gate_recovery_move_rows(
        subsets,
        target_delta_q=float(source_row["state_delta_q_vs_start"]),
        target_support=float(source_row["state_support_distance_to_vanilla"]),
        target_progress=float(source_row["state_target_progress_from_vanilla"]),
        support_gate=support_gate,
        progress_margin=progress_margin,
    )
    state_rows = pd.concat([replay_rows, subsets], ignore_index=True)
    edge_rows = pd.concat([replay_edges, pd.DataFrame(subset_edges)], ignore_index=True)
    path_rows = compute_pathway_wall_rows(
        state_rows,
        source_label="post_gate_recovery_subset_v0",
        support_gate=support_gate,
    )
    path_rows = annotate_pathway_debt_area_rows(
        path_rows,
        state_rows=state_rows,
        support_gate=support_gate,
    )
    path_rows = annotate_tunneling_evidence_rows(
        path_rows,
        support_gate=support_gate,
        progress_margin=progress_margin,
    )
    recovery_ids = set(subsets["state_id"].astype(str))
    recovery_path_rows = path_rows[
        path_rows["path_final_state_id"].astype(str).isin(recovery_ids)
    ].copy()
    recovery_trace_rows = trace_tunneling_path_states(
        recovery_path_rows,
        state_rows=state_rows,
        support_gate=support_gate,
        progress_margin=progress_margin,
    )
    recovery_step_rows = annotate_post_gate_recovery_step_rows(recovery_trace_rows)
    recovery_summary_rows = summarize_post_gate_recovery_paths(recovery_trace_rows)

    ranked_nodes.to_csv(output_dir / "post_gate_recovery_subset_ranked_nodes.csv", index=False)
    state_rows.to_csv(output_dir / STATE_ROWS_FILENAME, index=False)
    edge_rows.to_csv(output_dir / EDGE_ROWS_FILENAME, index=False)
    subsets.to_csv(output_dir / SUBSET_ROWS_FILENAME, index=False)
    recovery_path_rows.to_csv(output_dir / PATH_ROWS_FILENAME, index=False)
    recovery_trace_rows.to_csv(output_dir / TRACE_ROWS_FILENAME, index=False)
    recovery_step_rows.to_csv(output_dir / STEP_ROWS_FILENAME, index=False)
    recovery_summary_rows.to_csv(output_dir / SUMMARY_ROWS_FILENAME, index=False)

    config = {
        "source_move_dir": str(source_move_dir),
        "post_gate_dir": str(post_gate_dir),
        "prefix_dir": str(prefix_dir),
        "profile_batch_dir": str(profile_batch_dir),
        "output_dir": str(output_dir),
        "candidate_dirs": [str(path) for path in candidate_dirs],
        "vanilla_dir": str(vanilla_dir),
        "pair_id": pair_id,
        "prefix_rank": int(prefix_rank),
        "source_verdict": source_verdict,
        "source_recovery_policy": source_recovery_policy,
        "subset_sizes": list(subset_sizes),
        "include_bands": bool(include_bands),
        "baseline_iterations": int(baseline_iterations),
        "candidate_polish_iterations": int(candidate_polish_iterations),
        "local_polish_iterations": int(local_polish_iterations),
        "recovery_polish_iterations": int(recovery_polish_iterations),
        "target_action_multiplier": float(target_action_multiplier),
        "max_target_action_nodes": int(max_target_action_nodes),
        "resolution": float(resolution),
        "randomness": float(randomness),
        "support_gate": float(support_gate),
        "progress_margin": float(progress_margin),
        "subset_seed": int(subset_seed),
    }
    (output_dir / CONFIG_FILENAME).write_text(
        json.dumps(config, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    verdict_counts = subsets["post_gate_move_verdict"].astype(str).value_counts().to_dict()
    best = subsets.sort_values(
        [
            "post_gate_move_q_recovered",
            "post_gate_move_delta_q_gain",
            "state_support_distance_to_vanilla",
            "state_target_progress_from_vanilla",
            "recovery_selected_node_count",
        ],
        ascending=[False, False, False, False, True],
    ).iloc[0]
    prefix_q_gain = subsets[
        subsets["subset_policy"].astype(str).eq("pull_prefix")
        & subsets["post_gate_move_verdict"].astype(str).isin(
            [POST_GATE_RECOVERY_MOVE_Q_GAIN, POST_GATE_RECOVERY_MOVE_RECOVERED]
        )
    ].copy()
    min_prefix_q_gain_size = (
        int(prefix_q_gain["recovery_selected_node_count"].min())
        if not prefix_q_gain.empty
        else None
    )
    summary = {
        "schema": "leiden_basin_post_gate_recovery_subset_probe.v0",
        **config,
        "source_pair_id": pair_id,
        "source_prefix_rank": int(prefix_rank),
        "source_state_id": source_state.state_id,
        "source_recovery_policy": source_recovery_policy,
        "source_recovery_index": int(source_recovery_index),
        "source_full_selected_node_count": int(selected_nodes.size),
        "source_delta_q": float(source_row["state_delta_q_vs_start"]),
        "source_support": float(source_row["state_support_distance_to_vanilla"]),
        "source_target_progress": float(
            source_row["state_target_progress_from_vanilla"]
        ),
        "state_rows": int(len(state_rows)),
        "edge_rows": int(len(edge_rows)),
        "subset_rows": int(len(subsets)),
        "recovery_path_rows": int(len(recovery_path_rows)),
        "q_recovered_support_retained_rows": int(
            verdict_counts.get(POST_GATE_RECOVERY_MOVE_RECOVERED, 0)
        ),
        "q_gain_support_retained_rows": int(
            verdict_counts.get(POST_GATE_RECOVERY_MOVE_Q_GAIN, 0)
        ),
        "quality_regression_rows": int(verdict_counts.get("quality_regression", 0)),
        "verdict_counts": verdict_counts,
        "best_delta_q_gain": float(best["post_gate_move_delta_q_gain"]),
        "best_delta_q": float(best["state_delta_q_vs_start"]),
        "best_support": float(best["state_support_distance_to_vanilla"]),
        "best_progress": float(best["state_target_progress_from_vanilla"]),
        "best_subset_policy": str(best["subset_policy"]),
        "best_subset_rank_start": int(best["subset_rank_start"]),
        "best_subset_rank_end": int(best["subset_rank_end"]),
        "best_subset_node_count": int(best["recovery_selected_node_count"]),
        "min_prefix_q_gain_size": min_prefix_q_gain_size,
    }
    (output_dir / SUMMARY_FILENAME).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(
        output_dir / REPORT_FILENAME,
        summary=summary,
        source_move=source_move,
        subset_rows=subsets,
        recovery_path_rows=recovery_path_rows,
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    source_config = _load_source_config(DEFAULT_SOURCE_MOVE_DIR)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-move-dir", type=Path, default=DEFAULT_SOURCE_MOVE_DIR)
    parser.add_argument(
        "--post-gate-dir",
        type=Path,
        default=Path(source_config.get("post_gate_dir", DEFAULT_POST_GATE_DIR)),
    )
    parser.add_argument(
        "--prefix-dir",
        type=Path,
        default=Path(source_config.get("prefix_dir", DEFAULT_PREFIX_DIR)),
    )
    parser.add_argument(
        "--profile-batch-dir",
        type=Path,
        default=Path(source_config.get("profile_batch_dir", DEFAULT_PROFILE_BATCH_DIR)),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--candidate-dir", type=Path, action="append", default=None)
    parser.add_argument(
        "--vanilla-dir",
        type=Path,
        default=Path(source_config.get("vanilla_dir", DEFAULT_VANILLA_DIR)),
    )
    parser.add_argument("--pair-id", default=str(source_config.get("pair_id", "c0-s11-r0.001")))
    parser.add_argument("--prefix-rank", type=int, default=int(source_config.get("prefix_rank", 8)))
    parser.add_argument(
        "--source-verdict",
        default=str(source_config.get("source_verdict", POST_GATE_VERDICT_NEAR_MISS)),
    )
    parser.add_argument("--source-recovery-policy", default=DEFAULT_SOURCE_RECOVERY_POLICY)
    parser.add_argument(
        "--subset-sizes",
        default=",".join(str(value) for value in DEFAULT_SUBSET_SIZES),
    )
    parser.add_argument("--include-bands", action="store_true")
    parser.add_argument(
        "--baseline-iterations",
        type=int,
        default=int(source_config.get("baseline_iterations", 10)),
    )
    parser.add_argument(
        "--candidate-polish-iterations",
        type=int,
        default=int(source_config.get("candidate_polish_iterations", 5)),
    )
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
    parser.add_argument(
        "--target-action-multiplier",
        type=float,
        default=float(source_config.get("target_action_multiplier", 0.5)),
    )
    parser.add_argument(
        "--max-target-action-nodes",
        type=int,
        default=int(source_config.get("max_target_action_nodes", 64)),
    )
    parser.add_argument("--cumulative-fraction", type=float, default=0.80)
    parser.add_argument("--min-score-fraction", type=float, default=0.05)
    parser.add_argument("--min-gap-fraction", type=float, default=0.25)
    parser.add_argument("--min-guarded-pull-fraction", type=float, default=0.50)
    parser.add_argument(
        "--resolution",
        type=float,
        default=float(source_config.get("resolution", 0.01)),
    )
    parser.add_argument(
        "--randomness",
        type=float,
        default=float(source_config.get("randomness", 0.01)),
    )
    parser.add_argument("--perturb-seed-offset", type=int, default=5000)
    parser.add_argument("--polish-seed-offset", type=int, default=11000)
    parser.add_argument("--recovery-seed-offset", type=int, default=21000)
    parser.add_argument("--min-support-shift-from-vanilla", type=float, default=0.05)
    parser.add_argument("--min-material-q-gain", type=float, default=0.0)
    parser.add_argument(
        "--support-gate",
        type=float,
        default=float(source_config.get("support_gate", 0.05)),
    )
    parser.add_argument(
        "--progress-margin",
        type=float,
        default=float(source_config.get("progress_margin", 0.005)),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    candidate_dirs = (
        tuple(args.candidate_dir)
        if args.candidate_dir
        else tuple(
            Path(path)
            for path in _load_source_config(args.source_move_dir).get(
                "candidate_dirs",
                DEFAULT_CANDIDATE_DIRS,
            )
        )
    )
    subset_sizes = _parse_int_tuple(args.subset_sizes, DEFAULT_SUBSET_SIZES)
    summary = run_probe(
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
        subset_sizes=subset_sizes,
        include_bands=bool(args.include_bands),
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
        min_support_shift_from_vanilla=args.min_support_shift_from_vanilla,
        min_material_q_gain=args.min_material_q_gain,
        support_gate=args.support_gate,
        progress_margin=args.progress_margin,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
