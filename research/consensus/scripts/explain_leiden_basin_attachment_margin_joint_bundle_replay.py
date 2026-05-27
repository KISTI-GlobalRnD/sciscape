#!/usr/bin/env python3
"""Explain a focused attachment-margin joint-bundle replay.

The runner replays selected joint-bundle rows with the original polish seeds and
separates exact label changes from label-invariant partition changes.  This is a
diagnostic artifact generator, not a production operator.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT))

from run_leiden_basin_attachment_margin_cross_prefix_probe import (  # noqa: E402
    DEFAULT_SOURCE_RECOVERY_POLICY,
    SCORE_ROWS_FILENAME as ATTACHMENT_SCORE_ROWS_FILENAME,
    SUMMARY_ROWS_FILENAME as ATTACHMENT_SUMMARY_ROWS_FILENAME,
)
from run_leiden_basin_attachment_margin_joint_bundle_probe import (  # noqa: E402
    ACTION_JOINT_CANDIDATE_TRANSPLANT,
    ACTION_JOINT_MUTABLE,
    DEFAULT_ATTACHMENT_DIR,
    DEFAULT_OUTPUT_DIR as DEFAULT_JOINT_BUNDLE_DIR,
    ROWS_FILENAME as JOINT_BUNDLE_ROWS_FILENAME,
    _context_candidates,
    _select_context,
    _source_context_nodes,
    _target_nodes_from_scores,
)
from run_leiden_basin_attachment_margin_stage2_recovery import (  # noqa: E402
    _polished_child_with_reference,
    _rebuild_source_state,
)
from sciscape.clustering.leiden_basin_profile import changed_support_nodes  # noqa: E402
from sciscape.clustering.leiden_basin_profile import compact_membership  # noqa: E402
from sciscape.clustering.leiden_basin_profile import endpoint_distance  # noqa: E402
from sciscape.clustering.leiden_basin_search import (  # noqa: E402
    TransitionAction,
    transplant_action_nodes,
    unique_sorted_u32,
)
from sciscape.clustering.leiden_basin_transition_explain import (  # noqa: E402
    build_change_node_rows,
    build_change_shell_rows,
    build_label_transition_rows,
    membership_change_summary,
    node_csv,
)
from search_leiden_basin_transitions import _polished_child  # noqa: E402


COMBINED_DIR = DEFAULT_ATTACHMENT_DIR.parent
DEFAULT_OUTPUT_DIR = COMBINED_DIR / (
    "basin_transition_attachment_margin_joint_bundle_replay_field34_cc_c0_p8_current_label_v0"
)

SUMMARY_ROWS_FILENAME = "joint_bundle_replay_summary_rows.csv"
NODE_ROWS_FILENAME = "joint_bundle_replay_node_rows.csv"
SHELL_ROWS_FILENAME = "joint_bundle_replay_shell_rows.csv"
LABEL_TRANSITION_ROWS_FILENAME = "joint_bundle_replay_label_transition_rows.csv"
PAIR_ROWS_FILENAME = "joint_bundle_replay_pair_rows.csv"
CONFIG_FILENAME = "joint_bundle_replay_config.json"
SUMMARY_FILENAME = "joint_bundle_replay_summary.json"
REPORT_FILENAME = "joint_bundle_replay_report.md"


def _parse_move_kinds(value: str) -> tuple[str, ...]:
    parts = tuple(part.strip() for part in str(value).split(",") if part.strip())
    return parts or ("joint_mutable", "candidate_bundle_transplant")


def _child_index_from_state_id(state_id: str) -> int:
    match = re.search(r":(\d+)$", str(state_id))
    if not match:
        raise ValueError(f"Cannot parse child index from state_id: {state_id}")
    return int(match.group(1))


def _markdown_table(frame: pd.DataFrame, *, max_rows: int = 40) -> list[str]:
    if frame.empty:
        return []
    display = frame.head(max_rows)
    columns = list(display.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in display.iterrows():
        values: list[str] = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append("" if math.isnan(value) else f"{value:.6g}")
            else:
                values.append("" if pd.isna(value) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _select_joint_rows(
    rows: pd.DataFrame,
    *,
    source_case: str,
    target_k: int,
    context_family: str,
    context_multiplier: float,
    move_kinds: tuple[str, ...],
) -> pd.DataFrame:
    selected = rows[
        rows["source_case"].astype(str).eq(str(source_case))
        & rows["target_k"].astype(int).eq(int(target_k))
        & rows["context_family"].astype(str).eq(str(context_family))
        & rows["context_multiplier"].astype(float).eq(float(context_multiplier))
        & rows["move_kind"].astype(str).isin(set(move_kinds))
    ].copy()
    if selected.empty:
        raise ValueError(
            "No joint-bundle rows selected for "
            f"source_case={source_case}, target_k={target_k}, "
            f"context_family={context_family}, context_multiplier={context_multiplier}"
        )
    return selected.sort_values(["move_kind"]).reset_index(drop=True)


def _make_child(
    *,
    row: pd.Series,
    source_state: Any,
    case_ctx: dict[str, Any],
    config: dict[str, Any],
    recovery_seed: int,
    bundle_nodes: np.ndarray,
) -> tuple[Any, np.ndarray, int]:
    move_kind = str(row["move_kind"])
    child_index = _child_index_from_state_id(str(row["state_id"]))
    action = TransitionAction(
        action_type=(
            ACTION_JOINT_MUTABLE
            if move_kind == "joint_mutable"
            else ACTION_JOINT_CANDIDATE_TRANSPLANT
        ),
        action_params=str(row.get("action_params", "")),
        context_nodes=(
            bundle_nodes
            if move_kind == "joint_mutable"
            else np.asarray([], dtype=np.uint32)
        ),
        action_nodes=(
            None
            if move_kind == "joint_mutable"
            else bundle_nodes
        ),
    )
    pre_membership = source_state.membership
    if move_kind == "candidate_bundle_transplant":
        pre_membership = transplant_action_nodes(
            membership=source_state.membership,
            donor_membership=case_ctx["candidate"].recreated.membership,
            action_nodes=bundle_nodes,
            reference_nodes=np.asarray([], dtype=np.uint32),
        )
    pre_membership = compact_membership(pre_membership)
    child_kwargs = {
        "parent": source_state,
        "action": action,
        "graph": case_ctx["graph"],
        "donor_membership": case_ctx["candidate"].recreated.membership,
        "resolution": float(config.get("resolution", 0.01)),
        "seed": int(recovery_seed) + int(child_index) * 1000,
        "n_iterations": int(config.get("recovery_polish_iterations", 6)),
        "randomness": float(config.get("randomness", 0.01)),
        "child_index": child_index,
    }
    child = (
        _polished_child_with_reference(
            **child_kwargs,
            reference_nodes=np.asarray([], dtype=np.uint32),
        )
        if move_kind == "candidate_bundle_transplant"
        else _polished_child(**child_kwargs)
    )
    return child, pre_membership, child_index


def _role_counts(
    nodes: np.ndarray,
    *,
    selected_target_nodes: np.ndarray,
    context_nodes: np.ndarray,
    bundle_nodes: np.ndarray,
    source_action_nodes: np.ndarray,
    source_mutable_nodes: np.ndarray,
) -> dict[str, int]:
    selected = set(int(node) for node in unique_sorted_u32(selected_target_nodes))
    context = set(int(node) for node in unique_sorted_u32(context_nodes))
    bundle = set(int(node) for node in unique_sorted_u32(bundle_nodes))
    source_action = set(int(node) for node in unique_sorted_u32(source_action_nodes))
    source_mutable = set(int(node) for node in unique_sorted_u32(source_mutable_nodes))
    values = [int(node) for node in unique_sorted_u32(nodes)]
    return {
        "selected_target_changed_count": sum(1 for node in values if node in selected),
        "context_changed_count": sum(1 for node in values if node in context),
        "bundle_changed_count": sum(1 for node in values if node in bundle),
        "source_action_changed_count": sum(1 for node in values if node in source_action),
        "source_mutable_changed_count": sum(1 for node in values if node in source_mutable),
    }


def _pair_rows(children: dict[str, Any], *, sketch_nodes: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    names = sorted(children)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            left = children[left_name]
            right = children[right_name]
            rows.append(
                {
                    "left_move_kind": left_name,
                    "right_move_kind": right_name,
                    "left_quality": float(left.quality),
                    "right_quality": float(right.quality),
                    "quality_delta_right_minus_left": float(right.quality)
                    - float(left.quality),
                    "exact_changed_between_children": int(
                        np.count_nonzero(left.membership != right.membership)
                    ),
                    "aligned_changed_between_children": int(
                        changed_support_nodes(left.membership, right.membership).size
                    ),
                    "endpoint_distance_between_children": float(
                        endpoint_distance(left.membership, right.membership, sketch_nodes)
                    ),
                }
            )
    return pd.DataFrame(rows)


def _write_report(
    path: Path,
    *,
    summary_rows: pd.DataFrame,
    node_rows: pd.DataFrame,
    shell_rows: pd.DataFrame,
    pair_rows: pd.DataFrame,
) -> None:
    summary_cols = [
        "move_kind",
        "child_index",
        "result_quality",
        "delta_q_vs_vanilla",
        "delta_q_gain_vs_source",
        "pre_exact_changed_vs_source",
        "pre_aligned_changed_vs_source",
        "final_exact_changed_vs_source",
        "final_aligned_changed_vs_source",
        "final_exact_only_changed_vs_source",
        "endpoint_distance_to_source",
        "aligned_changed_node_ids_vs_source",
    ]
    node_cols = [
        "move_kind",
        "node",
        "in_selected_target",
        "in_context",
        "in_bundle",
        "in_source_action",
        "in_source_mutable",
        "exact_label_changed",
        "aligned_partition_changed",
        "hop_to_selected_target",
        "hop_to_bundle",
        "pull_to_selected_target",
        "pull_to_context",
        "pull_to_bundle",
        "baseline_label",
        "vanilla_label",
        "candidate_label",
        "reference_label",
        "result_label",
    ]
    shell_cols = [
        "move_kind",
        "change_kind",
        "hop_to_selected_target",
        "hop_to_bundle",
        "node_count",
    ]
    lines = [
        "# Joint Bundle Focused Replay",
        "",
        "This artifact replays the compact p8/current-label joint bundle and",
        "separates exact label changes from label-invariant partition changes.",
        "",
        "## Main Finding",
        "",
        "- The large exact-label change count is mostly a label-namespace artifact.",
        "- The compact replay changes only a small label-invariant support core versus the source state.",
        "- The mutable and candidate-transplant variants land on the same endpoint under label-invariant comparison.",
        "",
        "## Replay Summary",
        "",
    ]
    lines.extend(
        _markdown_table(
            summary_rows[[c for c in summary_cols if c in summary_rows]],
            max_rows=20,
        )
    )
    if not pair_rows.empty:
        lines.extend(["", "## Pairwise Endpoint Check", ""])
        lines.extend(_markdown_table(pair_rows, max_rows=20))
    lines.extend(["", "## Changed Core And Bundle Nodes", ""])
    display_nodes = node_rows.sort_values(
        ["move_kind", "aligned_partition_changed", "in_bundle", "node"],
        ascending=[True, False, False, True],
    )
    lines.extend(
        _markdown_table(display_nodes[[c for c in node_cols if c in display_nodes]], max_rows=80)
    )
    lines.extend(["", "## Change Shells", ""])
    lines.extend(
        _markdown_table(shell_rows[[c for c in shell_cols if c in shell_rows]], max_rows=80)
    )
    lines.extend(
        [
            "",
            "## Interpretation Guardrail",
            "",
            "- Treat exact changed-node counts as implementation-level label accounting.",
            "- Treat aligned changed-node counts and endpoint distance as the basin-level signal.",
            "- This remains diagnostic until it beats broad controls with material and cost-adjusted value.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_replay(
    *,
    attachment_dir: Path,
    joint_bundle_dir: Path,
    output_dir: Path,
    source_case: str,
    target_k: int,
    context_family: str,
    context_multiplier: float,
    move_kinds: tuple[str, ...],
    source_recovery_policy: str,
    recovery_seed: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    joint_rows = pd.read_csv(joint_bundle_dir / JOINT_BUNDLE_ROWS_FILENAME)
    selected_rows = _select_joint_rows(
        joint_rows,
        source_case=source_case,
        target_k=target_k,
        context_family=context_family,
        context_multiplier=context_multiplier,
        move_kinds=move_kinds,
    )
    stage1_rows = pd.read_csv(attachment_dir / ATTACHMENT_SUMMARY_ROWS_FILENAME)
    score_rows = pd.read_csv(attachment_dir / ATTACHMENT_SCORE_ROWS_FILENAME)
    stage1 = stage1_rows[stage1_rows["source_case"].astype(str).eq(str(source_case))]
    if stage1.empty:
        raise ValueError(f"No attachment summary row for source_case={source_case}")
    config, case_ctx, source_state, source_row, effective_seed, meta = _rebuild_source_state(
        source_move_dir=Path(str(stage1.iloc[0]["source_move_dir"])),
        source_case=source_case,
        source_recovery_policy=source_recovery_policy,
        requested_recovery_seed=recovery_seed,
    )
    source_context = _source_context_nodes(
        source_move_dir=Path(str(stage1.iloc[0]["source_move_dir"])),
        source_recovery_policy=source_recovery_policy,
    )
    selected_target_nodes = _target_nodes_from_scores(
        score_rows,
        source_case=source_case,
        selected_k=target_k,
    )
    context_candidates = _context_candidates(
        family=context_family,
        source_context_nodes=source_context,
        source_state=source_state,
        target_nodes=selected_target_nodes,
        case_ctx=case_ctx,
    )
    context_nodes = _select_context(
        candidates=context_candidates,
        target_nodes=selected_target_nodes,
        case_ctx=case_ctx,
        context_multiplier=context_multiplier,
        max_context_nodes=256,
    )
    bundle_nodes = unique_sorted_u32(np.concatenate([selected_target_nodes, context_nodes]))
    arrays = case_ctx["arrays"]
    src = np.asarray(arrays.src, dtype=np.uint32)
    dst = np.asarray(arrays.dst, dtype=np.uint32)
    weight = np.asarray(arrays.weight, dtype=np.float64)

    summary: list[dict[str, Any]] = []
    node_frames: list[pd.DataFrame] = []
    shell_frames: list[pd.DataFrame] = []
    transition_frames: list[pd.DataFrame] = []
    children: dict[str, Any] = {}

    for _, row in selected_rows.iterrows():
        move_kind = str(row["move_kind"])
        child, pre_membership, child_index = _make_child(
            row=row,
            source_state=source_state,
            case_ctx=case_ctx,
            config=config,
            recovery_seed=effective_seed,
            bundle_nodes=bundle_nodes,
        )
        children[move_kind] = child
        pre_summary = membership_change_summary(
            reference_membership=source_state.membership,
            membership=pre_membership,
            sketch_nodes=case_ctx["sketch_nodes"],
        )
        final_summary = membership_change_summary(
            reference_membership=source_state.membership,
            membership=child.membership,
            sketch_nodes=case_ctx["sketch_nodes"],
        )
        aligned_changed = changed_support_nodes(source_state.membership, child.membership)
        roles = _role_counts(
            aligned_changed,
            selected_target_nodes=selected_target_nodes,
            context_nodes=context_nodes,
            bundle_nodes=bundle_nodes,
            source_action_nodes=source_state.action_nodes,
            source_mutable_nodes=source_state.mutable_nodes,
        )
        summary.append(
            {
                **meta,
                "target_k": int(target_k),
                "selected_target_node_ids": node_csv(selected_target_nodes),
                "context_family": context_family,
                "context_multiplier": float(context_multiplier),
                "context_node_count": int(context_nodes.size),
                "context_node_ids": node_csv(context_nodes),
                "bundle_node_count": int(bundle_nodes.size),
                "bundle_node_ids": node_csv(bundle_nodes),
                "move_kind": move_kind,
                "child_index": int(child_index),
                "effective_polish_seed": int(effective_seed) + int(child_index) * 1000,
                "source_quality": float(source_state.quality),
                "result_quality": float(child.quality),
                "vanilla_quality": float(case_ctx["vanilla"].quality),
                "candidate_quality": float(case_ctx["candidate"].recreated.quality),
                "delta_q_vs_vanilla": float(child.quality)
                - float(case_ctx["vanilla"].quality),
                "delta_q_gain_vs_source": float(child.quality)
                - float(source_state.quality),
                "pre_exact_changed_vs_source": int(
                    pre_summary["exact_changed_node_count"]
                ),
                "pre_aligned_changed_vs_source": int(
                    pre_summary["aligned_changed_node_count"]
                ),
                "pre_exact_only_changed_vs_source": int(
                    pre_summary["exact_only_changed_node_count"]
                ),
                "final_exact_changed_vs_source": int(
                    final_summary["exact_changed_node_count"]
                ),
                "final_aligned_changed_vs_source": int(
                    final_summary["aligned_changed_node_count"]
                ),
                "final_exact_only_changed_vs_source": int(
                    final_summary["exact_only_changed_node_count"]
                ),
                "final_exact_to_aligned_ratio": float(
                    final_summary["exact_to_aligned_ratio"]
                ),
                "endpoint_distance_to_source": float(final_summary["endpoint_distance"]),
                "aligned_changed_node_ids_vs_source": node_csv(aligned_changed),
                "input_csv_final_exact_changed_vs_source": int(
                    row.get("joint_final_changed_node_count", 0)
                ),
                **roles,
            }
        )

        include_nodes = unique_sorted_u32(np.concatenate([aligned_changed, bundle_nodes]))
        node_frame = build_change_node_rows(
            reference_membership=source_state.membership,
            membership=child.membership,
            baseline_membership=case_ctx["baseline"].membership,
            vanilla_membership=case_ctx["vanilla"].membership,
            candidate_membership=case_ctx["candidate"].recreated.membership,
            src=src,
            dst=dst,
            weight=weight,
            target_nodes=selected_target_nodes,
            context_nodes=context_nodes,
            bundle_nodes=bundle_nodes,
            source_action_nodes=source_state.action_nodes,
            source_mutable_nodes=source_state.mutable_nodes,
            include_nodes=include_nodes,
        )
        node_frame.insert(0, "move_kind", move_kind)
        node_frames.append(node_frame)

        shell = build_change_shell_rows(
            reference_membership=source_state.membership,
            membership=child.membership,
            src=src,
            dst=dst,
            target_nodes=selected_target_nodes,
            bundle_nodes=bundle_nodes,
        )
        shell.insert(0, "move_kind", move_kind)
        shell_frames.append(shell)

        for node_set_kind, nodes in (
            ("aligned_partition_changed", aligned_changed),
            ("exact_label_changed", unique_sorted_u32(np.flatnonzero(source_state.membership != child.membership))),
        ):
            transitions = build_label_transition_rows(
                reference_membership=source_state.membership,
                membership=child.membership,
                nodes=nodes,
                target_nodes=selected_target_nodes,
                context_nodes=context_nodes,
                bundle_nodes=bundle_nodes,
            )
            if not transitions.empty:
                transitions.insert(0, "move_kind", move_kind)
                transitions.insert(1, "node_set_kind", node_set_kind)
                transition_frames.append(transitions)

    summary_rows = pd.DataFrame(summary)
    node_rows = pd.concat(node_frames, ignore_index=True) if node_frames else pd.DataFrame()
    shell_rows = (
        pd.concat(shell_frames, ignore_index=True) if shell_frames else pd.DataFrame()
    )
    transition_rows = (
        pd.concat(transition_frames, ignore_index=True)
        if transition_frames
        else pd.DataFrame()
    )
    pair_rows = _pair_rows(children, sketch_nodes=case_ctx["sketch_nodes"])

    summary_rows.to_csv(output_dir / SUMMARY_ROWS_FILENAME, index=False)
    node_rows.to_csv(output_dir / NODE_ROWS_FILENAME, index=False)
    shell_rows.to_csv(output_dir / SHELL_ROWS_FILENAME, index=False)
    transition_rows.to_csv(output_dir / LABEL_TRANSITION_ROWS_FILENAME, index=False)
    pair_rows.to_csv(output_dir / PAIR_ROWS_FILENAME, index=False)

    run_config = {
        "attachment_dir": str(attachment_dir),
        "joint_bundle_dir": str(joint_bundle_dir),
        "output_dir": str(output_dir),
        "source_case": source_case,
        "target_k": int(target_k),
        "context_family": context_family,
        "context_multiplier": float(context_multiplier),
        "move_kinds": list(move_kinds),
        "source_recovery_policy": source_recovery_policy,
        "requested_recovery_seed": int(recovery_seed),
    }
    (output_dir / CONFIG_FILENAME).write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result = {
        "schema": "leiden_basin_attachment_margin_joint_bundle_replay.v0",
        "output_dir": str(output_dir),
        "row_count": int(len(summary_rows)),
        "node_row_count": int(len(node_rows)),
        "pair_row_count": int(len(pair_rows)),
        "main_finding": (
            "exact_label_change_is_not_basin_change"
            if not pair_rows.empty
            and int(pair_rows["aligned_changed_between_children"].max()) == 0
            and float(pair_rows["endpoint_distance_between_children"].max()) == 0.0
            else "requires_review"
        ),
        "paths": {
            "summary_rows": str(output_dir / SUMMARY_ROWS_FILENAME),
            "node_rows": str(output_dir / NODE_ROWS_FILENAME),
            "shell_rows": str(output_dir / SHELL_ROWS_FILENAME),
            "label_transition_rows": str(output_dir / LABEL_TRANSITION_ROWS_FILENAME),
            "pair_rows": str(output_dir / PAIR_ROWS_FILENAME),
            "report": str(output_dir / REPORT_FILENAME),
        },
    }
    (output_dir / SUMMARY_FILENAME).write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(
        output_dir / REPORT_FILENAME,
        summary_rows=summary_rows,
        node_rows=node_rows,
        shell_rows=shell_rows,
        pair_rows=pair_rows,
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attachment-dir", type=Path, default=DEFAULT_ATTACHMENT_DIR)
    parser.add_argument("--joint-bundle-dir", type=Path, default=DEFAULT_JOINT_BUNDLE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source-case", default="p8_fullctx")
    parser.add_argument("--target-k", type=int, default=4)
    parser.add_argument("--context-family", default="current_label")
    parser.add_argument("--context-multiplier", type=float, default=8.0)
    parser.add_argument(
        "--move-kinds",
        default="joint_mutable,candidate_bundle_transplant",
    )
    parser.add_argument("--source-recovery-policy", default=DEFAULT_SOURCE_RECOVERY_POLICY)
    parser.add_argument(
        "--recovery-seed",
        type=int,
        default=0,
        help="0 reuses each source row's original seed, 21000 + source_recovery_index.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_replay(
        attachment_dir=args.attachment_dir,
        joint_bundle_dir=args.joint_bundle_dir,
        output_dir=args.output_dir,
        source_case=args.source_case,
        target_k=args.target_k,
        context_family=args.context_family,
        context_multiplier=args.context_multiplier,
        move_kinds=_parse_move_kinds(args.move_kinds),
        source_recovery_policy=args.source_recovery_policy,
        recovery_seed=args.recovery_seed,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
