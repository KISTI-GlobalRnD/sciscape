#!/usr/bin/env python3
"""Cross-prefix smoke probe for compact attachment-margin tunneling.

For each source recovery artifact, this script rebuilds the post-gate source
state, recomputes target-node attachment scores against that source's selected
context, and tests small target-only mutable sets.  It is a cross-source
diagnostic, not a default Dongdaemun policy.
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
    PREFIX_ROWS_FILENAME as BARRIER_PREFIX_ROWS_FILENAME,
)
from evaluate_leiden_basin_polish_prefixes import select_prefix_rows  # noqa: E402
from evaluate_leiden_basin_target_elbow_polish import (  # noqa: E402
    _rank_and_filter_prefix_rows,
)
from probe_leiden_basin_post_gate_recovery_moves import (  # noqa: E402
    POST_GATE_PATH_SUMMARY_FILENAME,
    _load_case_context,
    _markdown_table,
    _prefix_context,
    _replay_to_source_state,
    _select_source_path,
)
from probe_leiden_basin_post_gate_recovery_subsets import (  # noqa: E402
    SOURCE_MOVE_ROWS_FILENAME,
    _select_source_move,
)
from profile_leiden_basin_gate_attachment_candidates import _candidate_rows  # noqa: E402
from sciscape.clustering.leiden_basin_profile import parse_node_ids  # noqa: E402
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

COMBINED_DIR = REPO_ROOT / (
    "research/consensus/results/adaptive_refinement/"
    "leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30"
)
DEFAULT_SOURCE_MOVE_DIRS = (
    COMBINED_DIR / "basin_transition_post_gate_recovery_moves_field34_cc_c0_p6_wide_v0",
    COMBINED_DIR / "basin_transition_post_gate_recovery_moves_field34_cc_c0_p8_fullctx_v0",
    COMBINED_DIR / "basin_transition_post_gate_recovery_moves_field34_cc_c0_p10_wide_v0",
)
DEFAULT_OUTPUT_DIR = COMBINED_DIR / (
    "basin_transition_attachment_margin_cross_prefix_field34_cc_c0_p6_p8_p10_v1"
)

ACTION_ATTACHMENT_MARGIN_TARGET_ONLY = "recovery_attachment_margin_target_only"
ACTION_ATTACHMENT_MARGIN_CONTEXT_ONLY = "recovery_attachment_margin_context_only"

SCORE_ROWS_FILENAME = "attachment_margin_cross_prefix_score_rows.csv"
PROBE_ROWS_FILENAME = "attachment_margin_cross_prefix_probe_rows.csv"
EDGE_ROWS_FILENAME = "attachment_margin_cross_prefix_edges.csv"
SUMMARY_ROWS_FILENAME = "attachment_margin_cross_prefix_summary_rows.csv"
SUMMARY_FILENAME = "attachment_margin_cross_prefix_summary.json"
CONFIG_FILENAME = "attachment_margin_cross_prefix_config.json"
REPORT_FILENAME = "attachment_margin_cross_prefix_report.md"

DEFAULT_SOURCE_RECOVERY_POLICY = "vanilla_closure_topk:context_only"

def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))

def _parse_int_tuple(value: str, default: tuple[int, ...]) -> tuple[int, ...]:
    text = str(value).strip()
    if not text:
        return default
    if text.lower() in {"none", "null", "-"}:
        return ()
    parsed = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    return parsed or default

def _select_margin_nodes(score_rows: pd.DataFrame, *, selected_k: int) -> np.ndarray:
    ordered = score_rows.sort_values(
        ["gate_pull_margin_vs_current_source", "node"],
        ascending=[False, True],
    ).head(int(selected_k))
    return unique_sorted_u32(ordered["node"].to_numpy(dtype=np.uint32))

def _score_summary(score_rows: pd.DataFrame, selected_nodes: np.ndarray) -> dict[str, Any]:
    selected = score_rows[
        score_rows["node"].astype(int).isin(set(int(node) for node in selected_nodes))
    ]
    if selected.empty:
        return {
            "selected_gate_pull_sum": 0.0,
            "selected_gate_margin_sum": 0.0,
            "selected_current_source_pull_sum": 0.0,
        }
    return {
        "selected_gate_pull_sum": float(selected["pull_to_gate_context"].sum()),
        "selected_gate_margin_sum": float(
            selected["gate_pull_margin_vs_current_source"].sum()
        ),
        "selected_current_source_pull_sum": float(
            selected["pull_to_current_source_label"].sum()
        ),
    }

def _write_report(
    path: Path,
    *,
    summary_rows: pd.DataFrame,
    probe_rows: pd.DataFrame,
) -> None:
    cols = [
        "source_case",
        "prefix_rank",
        "source_recovery_index",
        "effective_recovery_seed",
        "source_context_node_count",
        "source_delta_q_vs_start",
        "best_selector",
        "best_selected_k",
        "best_selected_node_ids",
        "best_delta_q_gain",
        "best_state_delta_q_vs_start",
        "best_support",
        "best_progress",
        "best_mutable_node_count",
        "context_control_delta_q_gain",
        "context_control_mutable_node_count",
        "best_beats_context_control",
    ]
    row_cols = [
        "source_case",
        "prefix_rank",
        "source_recovery_index",
        "effective_recovery_seed",
        "action_mode",
        "selected_k",
        "selected_node_ids",
        "gate_release_delta_q_gain",
        "state_delta_q_vs_start",
        "state_support_distance_to_vanilla",
        "state_target_progress_from_vanilla",
        "mutable_node_count",
        "context_node_count",
        "gate_release_verdict",
    ]
    lines = [
        "# Attachment-Margin Cross-Prefix Probe",
        "",
        "This smoke probe recomputes attachment-margin target nodes for each source",
        "state and tests compact target-only mutable sets against the source",
        "context-only control.",
        "",
        "## Summary Rows",
        "",
    ]
    lines.extend(_markdown_table(summary_rows[[c for c in cols if c in summary_rows]], max_rows=30))
    lines.extend(["", "## Probe Rows", ""])
    lines.extend(_markdown_table(probe_rows[[c for c in row_cols if c in probe_rows]], max_rows=80))
    lines.extend(
        [
            "",
            "## Reading",
            "",
            "- `context_only_control` reproduces the source recovery move's selected context.",
            "- With the default `--recovery-seed 0`, each source reuses its original recovery seed (`21000 + source_recovery_index`).",
            "- `target_only_margin` is the compact tunneling candidate selected after recomputing margins for that source state.",
            "- A positive result must beat the context control on QF gain with lower mutable/context cost; otherwise the margin selector is source-specific.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def _probe_one_source(
    *,
    source_move_dir: Path,
    source_case: str,
    output_dir: Path,
    selected_ks: tuple[int, ...],
    source_recovery_policy: str,
    recovery_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config = _load_json(source_move_dir / "post_gate_recovery_move_config.json")
    if not config:
        raise ValueError(f"Missing source config in {source_move_dir}")

    pair_id = str(config.get("pair_id", "c0-s11-r0.001"))
    prefix_rank = int(config["prefix_rank"])
    post_gate_dir = Path(config["post_gate_dir"])
    prefix_dir = Path(config["prefix_dir"])
    profile_batch_dir = Path(config["profile_batch_dir"])
    vanilla_dir = Path(config["vanilla_dir"])
    candidate_dirs = tuple(Path(path) for path in config["candidate_dirs"])
    source_verdict = str(config.get("source_verdict", POST_GATE_VERDICT_NEAR_MISS))
    post_gate_config = _load_json(post_gate_dir / "post_gate_recovery_config.json")
    recorded_state_rows: pd.DataFrame | None = None
    recorded_state_dir = post_gate_config.get("state_dir")
    recorded_state_filename = post_gate_config.get("state_rows_filename")
    if recorded_state_dir and recorded_state_filename:
        recorded_state_path = Path(str(recorded_state_dir)) / str(recorded_state_filename)
        if recorded_state_path.exists():
            recorded_state_rows = pd.read_csv(recorded_state_path)

    source_moves = pd.read_csv(source_move_dir / SOURCE_MOVE_ROWS_FILENAME)
    source_move, source_recovery_index = _select_source_move(
        source_moves,
        recovery_policy=source_recovery_policy,
    )
    effective_recovery_seed = (
        int(recovery_seed)
        if int(recovery_seed) > 0
        else 21000 + int(source_recovery_index)
    )
    source_context_nodes = unique_sorted_u32(parse_node_ids(source_move["selected_node_ids"]))
    if source_context_nodes.size == 0:
        raise ValueError(f"Selected source context is empty for {source_move_dir}")

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
    source_polish_seed_offset = (
        int(config.get("polish_seed_offset", 11000))
        if str(source_path.get("path_policy", "")) == "branch_target_growth"
        else 2000
    )
    case_ctx = _load_case_context(
        prefix_row=prefix_row,
        profile_batch_dir=profile_batch_dir,
        candidate_dirs=candidate_dirs,
        vanilla_dir=vanilla_dir,
        baseline_iterations=int(config.get("baseline_iterations", 10)),
        candidate_polish_iterations=int(config.get("candidate_polish_iterations", 5)),
        resolution=float(config.get("resolution", 0.01)),
        randomness=float(config.get("randomness", 0.01)),
        perturb_seed_offset=1000,
    )
    source_state, source_row, _, _ = _replay_to_source_state(
        prefix_row=prefix_row,
        source_path=source_path,
        case_ctx=case_ctx,
        target_action_multiplier=float(config.get("target_action_multiplier", 0.5)),
        max_target_action_nodes=int(config.get("max_target_action_nodes", 64)),
        cumulative_fraction=0.80,
        min_score_fraction=0.05,
        min_gap_fraction=0.25,
        min_guarded_pull_fraction=0.50,
        local_polish_iterations=int(config.get("local_polish_iterations", 3)),
        resolution=float(config.get("resolution", 0.01)),
        randomness=float(config.get("randomness", 0.01)),
        polish_seed_offset=source_polish_seed_offset,
        min_support_shift_from_vanilla=0.01,
        min_material_q_gain=0.01,
        recorded_state_rows=recorded_state_rows,
    )

    score_rows = _candidate_rows(
        source_state=source_state,
        gate_nodes=source_context_nodes,
        full_context_nodes=source_context_nodes,
        moved_trace_nodes=np.asarray([], dtype=np.uint32),
        case_ctx=case_ctx,
    )
    score_rows.insert(0, "source_case", source_case)
    score_rows.insert(1, "prefix_rank", prefix_rank)
    score_rows.insert(2, "source_recovery_index", int(source_recovery_index))
    score_rows.insert(3, "effective_recovery_seed", int(effective_recovery_seed))

    rows: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    trial_index = 0
    base_context = {
        **case_ctx["public_context"],
        **_prefix_context(prefix_row),
        "path_policy": "attachment_margin_cross_prefix",
        "selection_policy": "attachment_margin_recomputed",
        "escalation_reason": "cross_prefix_smoke",
        "source_case": source_case,
        "source_move_dir": str(source_move_dir),
        "source_recovery_policy": source_recovery_policy,
        "source_recovery_index": int(source_recovery_index),
        "effective_recovery_seed": int(effective_recovery_seed),
        "source_context_node_count": int(source_context_nodes.size),
        "source_state_id": source_state.state_id,
        "target_stage_index": int(source_row.get("target_stage_index", 0)),
    }

    def run_action(
        *,
        action_mode: str,
        selected_k: int,
        selected_nodes: np.ndarray,
    ) -> None:
        nonlocal trial_index
        trial_index += 1
        selected = unique_sorted_u32(selected_nodes)
        if action_mode == "context_only_control":
            context_nodes = source_context_nodes
            action_type = ACTION_ATTACHMENT_MARGIN_CONTEXT_ONLY
        elif action_mode == "target_only_margin":
            context_nodes = selected
            action_type = ACTION_ATTACHMENT_MARGIN_TARGET_ONLY
        else:
            raise ValueError(f"Unknown action_mode: {action_mode}")
        action = TransitionAction(
            action_type=action_type,
            action_params=(
                f"source_case={source_case};mode={action_mode};"
                f"selected_k={int(selected_k)};context_k={int(context_nodes.size)}"
            ),
            context_nodes=context_nodes,
            action_nodes=None,
        )
        child = _polished_child(
            parent=source_state,
            action=action,
            graph=case_ctx["graph"],
            donor_membership=case_ctx["candidate"].recreated.membership,
            resolution=float(config.get("resolution", 0.01)),
            seed=int(effective_recovery_seed),
            n_iterations=int(config.get("recovery_polish_iterations", 6)),
            randomness=float(config.get("randomness", 0.01)),
            child_index=trial_index,
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
                **base_context,
                "action_mode": action_mode,
                "selected_k": int(selected_k),
                "selected_node_ids": node_csv(selected),
                "requested_recovery_seed": int(recovery_seed),
                "recovery_seed": int(effective_recovery_seed),
                **_score_summary(score_rows, selected),
            },
            parent_row=source_row,
            min_support_shift_from_vanilla=0.01,
            min_material_q_gain=0.01,
        )
        delta_q_gain = float(row["state_delta_q_vs_start"]) - float(
            source_row["state_delta_q_vs_start"]
        )
        support_gain = float(row["state_support_distance_to_vanilla"]) - float(
            source_row["state_support_distance_to_vanilla"]
        )
        q_recovered = delta_q_gain >= 0.01
        support_retained = float(row["state_support_distance_to_vanilla"]) >= float(
            source_row["state_support_distance_to_vanilla"]
        )
        verdict = (
            "q_gain_support_retained"
            if q_recovered and support_retained
            else "q_gain_support_lost"
            if q_recovered
            else "support_deepened_quality_loss"
            if support_gain > 0 and delta_q_gain < 0
            else "plateau"
            if abs(delta_q_gain) < 1e-12 and abs(support_gain) < 1e-12
            else "quality_loss"
        )
        row.update(
            {
                "source_delta_q_vs_start": float(source_row["state_delta_q_vs_start"]),
                "source_support_distance_to_vanilla": float(
                    source_row["state_support_distance_to_vanilla"]
                ),
                "source_target_progress_from_vanilla": float(
                    source_row["state_target_progress_from_vanilla"]
                ),
                "source_context_node_count": int(source_context_nodes.size),
                "attachment_margin_trial_index": int(trial_index),
                "gate_release_delta_q_gain": delta_q_gain,
                "gate_release_support_gain": support_gain,
                "gate_release_target_progress_gain": float(
                    row["state_target_progress_from_vanilla"]
                )
                - float(source_row["state_target_progress_from_vanilla"]),
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
                    "path_policy": "attachment_margin_cross_prefix",
                    "source_case": source_case,
                    "source_recovery_index": int(source_recovery_index),
                    "recovery_seed": int(effective_recovery_seed),
                    "action_mode": action_mode,
                    "selected_k": int(selected_k),
                },
            )
        )

    run_action(
        action_mode="context_only_control",
        selected_k=0,
        selected_nodes=np.asarray([], dtype=np.uint32),
    )
    for selected_k in selected_ks:
        selected = _select_margin_nodes(score_rows, selected_k=int(selected_k))
        if selected.size == 0:
            continue
        run_action(
            action_mode="target_only_margin",
            selected_k=int(selected_k),
            selected_nodes=selected,
        )

    probe_rows = pd.DataFrame(rows)
    edge_rows = pd.DataFrame(edges)
    context_control = probe_rows[
        probe_rows["action_mode"].astype(str).eq("context_only_control")
    ].iloc[0]
    target_rows = probe_rows[
        probe_rows["action_mode"].astype(str).eq("target_only_margin")
    ].copy()
    best_target = target_rows.sort_values(
        [
            "gate_release_delta_q_gain",
            "state_delta_q_vs_start",
            "state_support_distance_to_vanilla",
            "mutable_node_count",
        ],
        ascending=[False, False, False, True],
    ).iloc[0]
    summary = pd.DataFrame(
        [
            {
                "source_case": source_case,
                "prefix_rank": int(prefix_rank),
                "source_move_dir": str(source_move_dir),
                "source_recovery_index": int(source_recovery_index),
                "effective_recovery_seed": int(effective_recovery_seed),
                "source_context_node_count": int(source_context_nodes.size),
                "source_delta_q_vs_start": float(source_row["state_delta_q_vs_start"]),
                "source_support_distance_to_vanilla": float(
                    source_row["state_support_distance_to_vanilla"]
                ),
                "best_selector": "attachment_margin",
                "best_selected_k": int(best_target["selected_k"]),
                "best_selected_node_ids": str(best_target["selected_node_ids"]),
                "best_delta_q_gain": float(best_target["gate_release_delta_q_gain"]),
                "best_state_delta_q_vs_start": float(best_target["state_delta_q_vs_start"]),
                "best_support": float(best_target["state_support_distance_to_vanilla"]),
                "best_progress": float(best_target["state_target_progress_from_vanilla"]),
                "best_mutable_node_count": int(best_target["mutable_node_count"]),
                "best_context_node_count": int(best_target["context_node_count"]),
                "context_control_delta_q_gain": float(
                    context_control["gate_release_delta_q_gain"]
                ),
                "context_control_state_delta_q_vs_start": float(
                    context_control["state_delta_q_vs_start"]
                ),
                "context_control_support": float(
                    context_control["state_support_distance_to_vanilla"]
                ),
                "context_control_progress": float(
                    context_control["state_target_progress_from_vanilla"]
                ),
                "context_control_mutable_node_count": int(
                    context_control["mutable_node_count"]
                ),
                "context_control_context_node_count": int(
                    context_control["context_node_count"]
                ),
                "best_beats_context_control": bool(
                    float(best_target["gate_release_delta_q_gain"])
                    > float(context_control["gate_release_delta_q_gain"])
                    and int(best_target["mutable_node_count"])
                    < int(context_control["mutable_node_count"])
                ),
            }
        ]
    )
    return score_rows, probe_rows, edge_rows, summary

def run_probe(
    *,
    source_move_dirs: tuple[Path, ...],
    output_dir: Path,
    selected_ks: tuple[int, ...],
    source_recovery_policy: str,
    recovery_seed: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_scores: list[pd.DataFrame] = []
    all_rows: list[pd.DataFrame] = []
    all_edges: list[pd.DataFrame] = []
    all_summaries: list[pd.DataFrame] = []
    for directory in source_move_dirs:
        config = _load_json(directory / "post_gate_recovery_move_config.json")
        label = f"p{int(config.get('prefix_rank', 0))}_{directory.name.split('_')[-2]}"
        scores, rows, edges, summary = _probe_one_source(
            source_move_dir=directory,
            source_case=label,
            output_dir=output_dir,
            selected_ks=selected_ks,
            source_recovery_policy=source_recovery_policy,
            recovery_seed=recovery_seed,
        )
        all_scores.append(scores)
        all_rows.append(rows)
        all_edges.append(edges)
        all_summaries.append(summary)

    score_rows = pd.concat(all_scores, ignore_index=True) if all_scores else pd.DataFrame()
    probe_rows = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    edge_rows = pd.concat(all_edges, ignore_index=True) if all_edges else pd.DataFrame()
    summary_rows = (
        pd.concat(all_summaries, ignore_index=True) if all_summaries else pd.DataFrame()
    )
    score_rows.to_csv(output_dir / SCORE_ROWS_FILENAME, index=False)
    probe_rows.to_csv(output_dir / PROBE_ROWS_FILENAME, index=False)
    edge_rows.to_csv(output_dir / EDGE_ROWS_FILENAME, index=False)
    summary_rows.to_csv(output_dir / SUMMARY_ROWS_FILENAME, index=False)
    summary = {
        "schema": "leiden_basin_attachment_margin_cross_prefix.v0",
        "output_dir": str(output_dir),
        "source_count": int(len(source_move_dirs)),
        "selected_ks": [int(k) for k in selected_ks],
        "source_recovery_policy": source_recovery_policy,
        "requested_recovery_seed": int(recovery_seed),
        "source_cases": summary_rows["source_case"].astype(str).tolist()
        if not summary_rows.empty
        else [],
        "best_beats_context_control_count": int(
            summary_rows["best_beats_context_control"].astype(bool).sum()
        )
        if not summary_rows.empty
        else 0,
    }
    (output_dir / SUMMARY_FILENAME).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    config = {
        "source_move_dirs": [str(path) for path in source_move_dirs],
        "selected_ks": [int(k) for k in selected_ks],
        "source_recovery_policy": source_recovery_policy,
        "requested_recovery_seed": int(recovery_seed),
        "recovery_seed_policy": "0 means 21000 + source_recovery_index per source",
    }
    (output_dir / CONFIG_FILENAME).write_text(
        json.dumps(config, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(
        output_dir / REPORT_FILENAME,
        summary_rows=summary_rows,
        probe_rows=probe_rows,
    )
    return summary

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-move-dir",
        type=Path,
        action="append",
        default=None,
        help="Post-gate recovery move artifact directory. Repeat to override defaults.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--selected-ks", default="1,2,4")
    parser.add_argument("--source-recovery-policy", default=DEFAULT_SOURCE_RECOVERY_POLICY)
    parser.add_argument(
        "--recovery-seed",
        type=int,
        default=0,
        help=(
            "Recovery polish seed. Use 0 to reuse each source row's original "
            "seed, 21000 + source_recovery_index."
        ),
    )
    return parser

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    source_move_dirs = (
        tuple(args.source_move_dir)
        if args.source_move_dir
        else tuple(DEFAULT_SOURCE_MOVE_DIRS)
    )
    summary = run_probe(
        source_move_dirs=source_move_dirs,
        output_dir=args.output_dir,
        selected_ks=_parse_int_tuple(args.selected_ks, (1, 2, 4)),
        source_recovery_policy=args.source_recovery_policy,
        recovery_seed=args.recovery_seed,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))

if __name__ == "__main__":
    main()
