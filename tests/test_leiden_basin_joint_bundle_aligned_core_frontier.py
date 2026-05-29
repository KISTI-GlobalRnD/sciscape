"""Tests for joint-bundle aligned-core frontier profiling."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "research/consensus/scripts/leiden_basin/operator_probes/joint_bundle/profile_leiden_basin_joint_bundle_aligned_core_frontier.py"
)


def _load_module():
    if str(SCRIPT_PATH.parent) not in sys.path:
        sys.path.insert(0, str(SCRIPT_PATH.parent))
    spec = importlib.util.spec_from_file_location(
        "profile_leiden_basin_joint_bundle_aligned_core_frontier_for_test",
        SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _top_row(
    *,
    source_case: str,
    target_k: int,
    context_family: str,
    move_kind: str,
    quality_gain: float,
    aligned: int,
    rank: int,
) -> dict[str, object]:
    return {
        "artifact": "joint_bundle",
        "rank_kind": "quality_gain",
        "rank": rank,
        "source_case": source_case,
        "target_k": target_k,
        "context_family": context_family,
        "context_multiplier": 8.0,
        "move_kind": move_kind,
        "quality_gain": quality_gain,
        "final_aligned_changed": aligned,
        "final_exact_changed": aligned,
        "final_exact_only_changed": 0,
        "endpoint_distance": 0.0,
        "state_delta_q_vs_vanilla": quality_gain,
        "joint_verdict": "joint_beats_same_randomness_control",
    }


def test_select_replay_configs_dedupes_positive_joint_bundle_rows():
    module = _load_module()
    rows = pd.DataFrame(
        [
            _top_row(
                source_case="p8",
                target_k=4,
                context_family="current_label",
                move_kind="joint_mutable",
                quality_gain=0.3,
                aligned=6,
                rank=2,
            ),
            _top_row(
                source_case="p8",
                target_k=4,
                context_family="current_label",
                move_kind="joint_mutable",
                quality_gain=0.2,
                aligned=6,
                rank=3,
            ),
            _top_row(
                source_case="p10",
                target_k=8,
                context_family="candidate_label",
                move_kind="candidate_bundle_transplant",
                quality_gain=-1.0,
                aligned=35,
                rank=1,
            ),
        ]
    )

    configs = module.select_replay_configs(rows)

    assert len(configs) == 1
    assert configs.iloc[0]["config_rank"] == 1
    assert configs.iloc[0]["source_case"] == "p8"
    assert configs.iloc[0]["replay_slug"]


def test_aggregate_node_frontier_ranks_repeated_aligned_core_nodes():
    module = _load_module()
    rows = pd.DataFrame(
        [
            {
                "replay_slug": "a",
                "config_rank": 1,
                "source_case": "p8",
                "node": 10,
                "aligned_partition_changed": True,
                "exact_label_changed": True,
                "in_selected_target": True,
                "in_context": False,
                "in_bundle": True,
                "in_source_action": False,
                "in_source_mutable": False,
                "hop_to_selected_target": 0,
                "hop_to_bundle": 0,
                "pull_to_selected_target": 0.0,
                "pull_to_context": 1.0,
                "pull_to_bundle": 2.0,
                "baseline_label": 1,
                "vanilla_label": 2,
                "candidate_label": 3,
            },
            {
                "replay_slug": "b",
                "config_rank": 2,
                "source_case": "p8",
                "node": 10,
                "aligned_partition_changed": True,
                "exact_label_changed": True,
                "in_selected_target": True,
                "in_context": False,
                "in_bundle": True,
                "in_source_action": False,
                "in_source_mutable": False,
                "hop_to_selected_target": 0,
                "hop_to_bundle": 0,
                "pull_to_selected_target": 0.0,
                "pull_to_context": 2.0,
                "pull_to_bundle": 3.0,
                "baseline_label": 1,
                "vanilla_label": 2,
                "candidate_label": 3,
            },
            {
                "replay_slug": "a",
                "config_rank": 1,
                "source_case": "p8",
                "node": 20,
                "aligned_partition_changed": False,
                "exact_label_changed": False,
                "in_selected_target": False,
                "in_context": True,
                "in_bundle": True,
                "in_source_action": False,
                "in_source_mutable": False,
                "hop_to_selected_target": 1,
                "hop_to_bundle": 0,
                "pull_to_selected_target": 1.0,
                "pull_to_context": 0.0,
                "pull_to_bundle": 1.0,
                "baseline_label": 4,
                "vanilla_label": 5,
                "candidate_label": 6,
            },
        ]
    )

    frontier = module.aggregate_node_frontier(rows)

    assert frontier.iloc[0]["node"] == 10
    assert frontier.iloc[0]["aligned_change_count"] == 2
    assert frontier.iloc[0]["frontier_role"] == "target_core"
    assert frontier.iloc[0]["config_ranks"] == "1,2"
