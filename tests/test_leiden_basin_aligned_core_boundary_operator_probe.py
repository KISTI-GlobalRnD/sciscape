"""Tests for aligned-core boundary operator planning."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sciscape.clustering.leiden_basin_search import (
    LOCAL_SELECTOR_READINESS_ALREADY_RECOVERED,
    LOCAL_SELECTOR_READINESS_LABEL_COMPLETION,
    LOCAL_SELECTOR_READINESS_NO_LABEL_COMPETITION,
    LOCAL_SELECTOR_READINESS_READY,
    LOCAL_SELECTOR_READINESS_TOO_FEW_HANDLES,
    build_aligned_core_boundary_plan_rows,
    build_aligned_core_handle_subset_plan_rows,
    build_aligned_core_handle_selector_plan_rows,
    build_local_handle_selector_plan_rows,
    node_csv,
    score_aligned_core_handle_nodes,
    score_local_handle_candidates,
    select_aligned_core_boundary_nodes,
    summarize_local_selector_readiness_rows,
)


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "research/consensus/scripts/run_leiden_basin_aligned_core_boundary_operator_probe.py"
)
SUBSET_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "research/consensus/scripts/run_leiden_basin_aligned_core_handle_subset_probe.py"
)
STABILITY_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "research/consensus/scripts/run_leiden_basin_aligned_core_handle_stability_probe.py"
)
SELECTOR_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "research/consensus/scripts/run_leiden_basin_aligned_core_handle_selector_probe.py"
)
LOCAL_SELECTOR_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "research/consensus/scripts/run_leiden_basin_local_handle_selector_probe.py"
)
SOURCE_SCREEN_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "research/consensus/scripts/screen_leiden_basin_selector_sources.py"
)
SOURCE_SCREEN_BATCH_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "research/consensus/scripts/run_leiden_basin_selector_source_screen_batch.py"
)


def _load_script_module():
    if str(SCRIPT_PATH.parent) not in sys.path:
        sys.path.insert(0, str(SCRIPT_PATH.parent))
    spec = importlib.util.spec_from_file_location(
        "run_leiden_basin_aligned_core_boundary_operator_probe_for_test",
        SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_subset_script_module():
    if str(SUBSET_SCRIPT_PATH.parent) not in sys.path:
        sys.path.insert(0, str(SUBSET_SCRIPT_PATH.parent))
    spec = importlib.util.spec_from_file_location(
        "run_leiden_basin_aligned_core_handle_subset_probe_for_test",
        SUBSET_SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_stability_script_module():
    if str(STABILITY_SCRIPT_PATH.parent) not in sys.path:
        sys.path.insert(0, str(STABILITY_SCRIPT_PATH.parent))
    spec = importlib.util.spec_from_file_location(
        "run_leiden_basin_aligned_core_handle_stability_probe_for_test",
        STABILITY_SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_selector_script_module():
    if str(SELECTOR_SCRIPT_PATH.parent) not in sys.path:
        sys.path.insert(0, str(SELECTOR_SCRIPT_PATH.parent))
    spec = importlib.util.spec_from_file_location(
        "run_leiden_basin_aligned_core_handle_selector_probe_for_test",
        SELECTOR_SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_local_selector_script_module():
    if str(LOCAL_SELECTOR_SCRIPT_PATH.parent) not in sys.path:
        sys.path.insert(0, str(LOCAL_SELECTOR_SCRIPT_PATH.parent))
    spec = importlib.util.spec_from_file_location(
        "run_leiden_basin_local_handle_selector_probe_for_test",
        LOCAL_SELECTOR_SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_source_screen_script_module():
    if str(SOURCE_SCREEN_SCRIPT_PATH.parent) not in sys.path:
        sys.path.insert(0, str(SOURCE_SCREEN_SCRIPT_PATH.parent))
    spec = importlib.util.spec_from_file_location(
        "screen_leiden_basin_selector_sources_for_test",
        SOURCE_SCREEN_SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_source_screen_batch_script_module():
    if str(SOURCE_SCREEN_BATCH_SCRIPT_PATH.parent) not in sys.path:
        sys.path.insert(0, str(SOURCE_SCREEN_BATCH_SCRIPT_PATH.parent))
    spec = importlib.util.spec_from_file_location(
        "run_leiden_basin_selector_source_screen_batch_for_test",
        SOURCE_SCREEN_BATCH_SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_select_aligned_core_boundary_nodes_separates_target_and_boundary_roles():
    rows = pd.DataFrame(
        [
            {
                "node": 10,
                "frontier_role": "target_core",
                "aligned_change_count": 5,
                "context_count": 0,
                "source_mutable_count": 0,
                "max_pull_to_bundle": 0.0,
            },
            {
                "node": 20,
                "frontier_role": "source_mutable_core",
                "aligned_change_count": 5,
                "context_count": 0,
                "source_mutable_count": 5,
                "max_pull_to_bundle": 0.4,
            },
            {
                "node": 30,
                "frontier_role": "context_core",
                "aligned_change_count": 1,
                "context_count": 6,
                "source_mutable_count": 0,
                "max_pull_to_bundle": 2.0,
            },
        ]
    )

    selection = select_aligned_core_boundary_nodes(
        rows,
        min_target_change_count=5,
        min_boundary_change_count=5,
        max_context_core_nodes=1,
    )

    assert node_csv(selection.target_nodes) == "10"
    assert node_csv(selection.boundary_core_nodes) == "20"
    assert node_csv(selection.context_core_nodes) == "30"


def test_build_aligned_core_boundary_plan_rows_adds_boundary_and_candidate_context():
    plans = build_aligned_core_boundary_plan_rows(
        target_nodes=np.asarray([10, 11], dtype=np.uint32),
        boundary_core_nodes=np.asarray([20], dtype=np.uint32),
        context_core_nodes=np.asarray([30], dtype=np.uint32),
        candidate_context_by_cap={8: np.asarray([40, 41], dtype=np.uint32)},
    )

    assert list(plans["plan_kind"]) == [
        "target_core_only",
        "target_core_plus_boundary_core",
        "target_core_plus_boundary_context_core",
        "target_core_plus_candidate_context",
    ]
    boundary = plans[plans["plan_kind"].eq("target_core_plus_boundary_core")].iloc[0]
    assert boundary["context_node_ids"] == "20"
    assert boundary["included_boundary_core_node_ids"] == "20"
    candidate = plans[plans["plan_kind"].eq("target_core_plus_candidate_context")].iloc[0]
    assert candidate["candidate_context_cap"] == 8
    assert candidate["candidate_context_node_ids"] == "40,41"
    assert candidate["bundle_node_ids"] == "10,11,20,40,41"


def test_build_aligned_core_handle_subset_plan_rows_enumerates_bounded_subsets():
    plans = build_aligned_core_handle_subset_plan_rows(
        target_nodes=np.asarray([3, 1, 2], dtype=np.uint32),
        min_subset_size=2,
        max_subset_size=3,
    )

    assert len(plans) == 4
    assert list(plans["subset_size"]) == [2, 2, 2, 3]
    assert plans.iloc[0]["subset_node_ids"] == "1,2"
    assert plans.iloc[-1]["subset_node_ids"] == "1,2,3"


def test_handle_selector_scores_context_pull_and_mutable_penalty():
    rows = pd.DataFrame(
        [
            {
                "node": 10,
                "frontier_role": "target_core",
                "aligned_change_count": 6,
                "selected_target_count": 6,
                "source_action_count": 6,
                "source_mutable_count": 6,
                "max_pull_to_context": 10.0,
            },
            {
                "node": 20,
                "frontier_role": "target_core",
                "aligned_change_count": 5,
                "selected_target_count": 6,
                "source_action_count": 0,
                "source_mutable_count": 0,
                "max_pull_to_context": 8.0,
            },
            {
                "node": 30,
                "frontier_role": "target_core",
                "aligned_change_count": 5,
                "selected_target_count": 6,
                "source_action_count": 0,
                "source_mutable_count": 0,
                "max_pull_to_context": 7.0,
            },
        ]
    )

    context = score_aligned_core_handle_nodes(
        rows,
        selector_policy="context_pull",
        min_target_change_count=5,
    )
    penalized = score_aligned_core_handle_nodes(
        rows,
        selector_policy="mutable_penalized_context_pull",
        min_target_change_count=5,
    )

    assert list(context["node"]) == [10, 20, 30]
    assert list(penalized["node"]) == [20, 30, 10]
    assert not bool(context.iloc[0]["selector_uses_replay_features"])


def test_handle_selector_plan_rows_build_topk_prefixes():
    rows = pd.DataFrame(
        [
            {
                "node": 10,
                "frontier_role": "target_core",
                "aligned_change_count": 6,
                "max_pull_to_context": 1.0,
            },
            {
                "node": 20,
                "frontier_role": "target_core",
                "aligned_change_count": 5,
                "max_pull_to_context": 2.0,
            },
        ]
    )

    plans, scores = build_aligned_core_handle_selector_plan_rows(
        rows,
        selector_policies=("context_pull",),
        min_subset_size=1,
        max_subset_size=2,
    )

    assert list(scores["node"]) == [20, 10]
    assert list(plans["subset_node_ids"]) == ["20", "10,20"]
    assert list(plans["selector_ordered_node_ids"]) == ["20", "20,10"]


def test_local_handle_selector_scores_do_not_use_replay_frontier_features():
    rows = pd.DataFrame(
        [
            {
                "source_case": "p8",
                "node": 10,
                "candidate_label": 100,
                "pull_to_gate_context": 10.0,
                "gate_pull_margin_vs_current_source": 1.0,
                "in_source_action": "True",
                "in_source_mutable": "True",
                "rank_best_consensus": 2,
            },
            {
                "source_case": "p8",
                "node": 20,
                "candidate_label": 200,
                "pull_to_gate_context": 8.0,
                "gate_pull_margin_vs_current_source": 2.0,
                "in_source_action": "False",
                "in_source_mutable": "False",
                "rank_best_consensus": 1,
            },
        ]
    )

    margin = score_local_handle_candidates(
        rows,
        selector_policy="attachment_margin",
        source_case="p8",
    )
    non_source = score_local_handle_candidates(
        rows,
        selector_policy="non_source_gate_pull",
        source_case="p8",
    )
    plans, scores = build_local_handle_selector_plan_rows(
        rows,
        selector_policies=("gate_pull",),
        selected_ks=(1, 2),
        source_case="p8",
    )

    assert list(margin["node"]) == [20, 10]
    assert list(non_source["node"]) == [20, 10]
    assert not bool(scores.iloc[0]["selector_uses_replay_features"])
    assert list(plans["selector_ordered_node_ids"]) == ["10", "10,20"]
    assert list(plans["subset_node_ids"]) == ["10", "10,20"]


def test_local_candidate_label_coherent_selector_prefers_positive_margin_group():
    rows = pd.DataFrame(
        [
            {
                "source_case": "p6",
                "node": 10,
                "candidate_label": 1184,
                "pull_to_gate_context": 5.0,
                "gate_pull_margin_vs_current_source": -2.0,
                "in_source_mutable": False,
            },
            {
                "source_case": "p6",
                "node": 20,
                "candidate_label": 1184,
                "pull_to_gate_context": 4.0,
                "gate_pull_margin_vs_current_source": -1.0,
                "in_source_mutable": False,
            },
            {
                "source_case": "p6",
                "node": 30,
                "candidate_label": 1090,
                "pull_to_gate_context": 3.0,
                "gate_pull_margin_vs_current_source": 2.0,
                "in_source_mutable": False,
            },
            {
                "source_case": "p6",
                "node": 40,
                "candidate_label": 1090,
                "pull_to_gate_context": 2.0,
                "gate_pull_margin_vs_current_source": 1.0,
                "in_source_mutable": False,
            },
        ]
    )

    ranked = score_local_handle_candidates(
        rows,
        selector_policy="candidate_label_margin_coherent",
        source_case="p6",
    )

    assert list(ranked["node"]) == [30, 40]
    assert set(ranked["selector_candidate_label"]) == {"1090"}


def test_local_candidate_label_coherent_selector_uses_positive_rows_before_fillers():
    rows = pd.DataFrame(
        [
            {
                "source_case": "p5",
                "node": 10,
                "candidate_label": 957,
                "pull_to_gate_context": 0.0,
                "gate_pull_margin_vs_current_source": 0.0,
                "in_source_mutable": False,
            },
            {
                "source_case": "p5",
                "node": 20,
                "candidate_label": 1090,
                "pull_to_gate_context": 0.4,
                "gate_pull_margin_vs_current_source": 0.4,
                "in_source_mutable": False,
            },
            {
                "source_case": "p5",
                "node": 30,
                "candidate_label": 1090,
                "pull_to_gate_context": 0.0,
                "gate_pull_margin_vs_current_source": -1.0,
                "in_source_mutable": False,
            },
        ]
    )

    ranked = score_local_handle_candidates(
        rows,
        selector_policy="candidate_label_margin_coherent",
        source_case="p5",
    )

    assert list(ranked["node"])[:2] == [20, 30]
    assert set(ranked["selector_candidate_label"]) == {"1090"}


def test_local_selector_readiness_marks_competing_unrecovered_source_ready():
    scores = pd.DataFrame(
        [
            {
                "source_case": "p8",
                "node": 10,
                "candidate_label": 100,
                "gate_pull_margin_vs_current_source": 1.5,
                "pull_to_gate_context": 2.0,
                "in_source_action": False,
                "in_source_mutable": False,
            },
            {
                "source_case": "p8",
                "node": 20,
                "candidate_label": 200,
                "gate_pull_margin_vs_current_source": 1.0,
                "pull_to_gate_context": 1.0,
                "in_source_action": False,
                "in_source_mutable": False,
            },
            {
                "source_case": "p8",
                "node": 30,
                "candidate_label": 200,
                "gate_pull_margin_vs_current_source": 0.5,
                "pull_to_gate_context": 0.5,
                "in_source_action": True,
                "in_source_mutable": True,
            },
        ]
    )
    source = pd.DataFrame(
        [
            {
                "source_case": "p8",
                "source_delta_q_vs_start": -0.2,
                "source_support_distance_to_vanilla": 0.04,
                "source_target_progress_from_vanilla": 0.02,
            }
        ]
    )

    out = summarize_local_selector_readiness_rows(scores, source_summary_rows=source)

    assert out.iloc[0]["readiness_verdict"] == LOCAL_SELECTOR_READINESS_READY
    assert out.iloc[0]["positive_margin_non_source_count"] == 2
    assert out.iloc[0]["positive_margin_candidate_label_count"] == 2
    assert out.iloc[0]["top_candidate_label"] == "100"


def test_local_selector_readiness_separates_controls_and_sparse_cases():
    scores = pd.DataFrame(
        [
            {
                "source_case": "recovered",
                "node": 10,
                "candidate_label": 100,
                "gate_pull_margin_vs_current_source": 1.0,
                "pull_to_gate_context": 1.0,
                "in_source_action": False,
                "in_source_mutable": False,
            },
            {
                "source_case": "sparse",
                "node": 20,
                "candidate_label": 200,
                "gate_pull_margin_vs_current_source": 1.0,
                "pull_to_gate_context": 1.0,
                "in_source_action": False,
                "in_source_mutable": False,
            },
            {
                "source_case": "one_label",
                "node": 30,
                "candidate_label": 300,
                "gate_pull_margin_vs_current_source": 1.0,
                "pull_to_gate_context": 1.0,
                "in_source_action": False,
                "in_source_mutable": False,
            },
            {
                "source_case": "one_label",
                "node": 40,
                "candidate_label": 300,
                "gate_pull_margin_vs_current_source": 0.5,
                "pull_to_gate_context": 0.5,
                "in_source_action": False,
                "in_source_mutable": False,
            },
            {
                "source_case": "label_completion",
                "node": 50,
                "candidate_label": 400,
                "gate_pull_margin_vs_current_source": 1.0,
                "pull_to_gate_context": 1.0,
                "in_source_action": False,
                "in_source_mutable": False,
            },
            {
                "source_case": "label_completion",
                "node": 51,
                "candidate_label": 400,
                "gate_pull_margin_vs_current_source": 0.0,
                "pull_to_gate_context": 0.5,
                "in_source_action": False,
                "in_source_mutable": False,
            },
            {
                "source_case": "label_completion",
                "node": 52,
                "candidate_label": 400,
                "gate_pull_margin_vs_current_source": -0.2,
                "pull_to_gate_context": 0.5,
                "in_source_action": False,
                "in_source_mutable": False,
            },
            {
                "source_case": "label_completion",
                "node": 53,
                "candidate_label": 400,
                "gate_pull_margin_vs_current_source": -0.3,
                "pull_to_gate_context": 0.5,
                "in_source_action": False,
                "in_source_mutable": False,
            },
        ]
    )
    source = pd.DataFrame(
        [
            {
                "source_case": "recovered",
                "source_delta_q_vs_start": 0.2,
                "source_support_distance_to_vanilla": 0.06,
            },
            {
                "source_case": "sparse",
                "source_delta_q_vs_start": -0.2,
                "source_support_distance_to_vanilla": 0.04,
            },
            {
                "source_case": "one_label",
                "source_delta_q_vs_start": -0.2,
                "source_support_distance_to_vanilla": 0.04,
            },
            {
                "source_case": "label_completion",
                "source_delta_q_vs_start": -0.2,
                "source_support_distance_to_vanilla": 0.04,
            },
        ]
    )

    out = summarize_local_selector_readiness_rows(scores, source_summary_rows=source)
    verdicts = dict(zip(out["source_case"], out["readiness_verdict"]))

    assert verdicts["recovered"] == LOCAL_SELECTOR_READINESS_ALREADY_RECOVERED
    assert verdicts["sparse"] == LOCAL_SELECTOR_READINESS_TOO_FEW_HANDLES
    assert verdicts["one_label"] == LOCAL_SELECTOR_READINESS_NO_LABEL_COMPETITION
    assert verdicts["label_completion"] == LOCAL_SELECTOR_READINESS_LABEL_COMPLETION


def test_source_screen_context_nodes_use_path_action_union_and_last_action():
    module = _load_source_screen_script_module()
    recorded = pd.DataFrame(
        [
            {"state_id": "root", "selected_node_ids": ""},
            {"state_id": "a", "selected_node_ids": "3,1"},
            {"state_id": "b", "selected_node_ids": "5,3"},
        ]
    )

    union = module._context_nodes_from_recorded_path(
        recorded,
        context_mode="path_action_union",
    )
    last = module._context_nodes_from_recorded_path(
        recorded,
        context_mode="last_action",
    )

    assert union.tolist() == [1, 3, 5]
    assert last.tolist() == [3, 5]


def test_source_screen_source_rows_filter_and_limit_by_prefix():
    module = _load_source_screen_script_module()
    rows = pd.DataFrame(
        [
            {
                "path_prefix_rank": 6,
                "post_gate_verdict": "near_miss_recovery_trend",
                "post_gate_best_delta_q_gain_from_gate": 0.1,
                "post_gate_final_support": 0.05,
                "post_gate_final_target_progress": 0.01,
                "post_gate_step_count": 2,
            },
            {
                "path_prefix_rank": 6,
                "post_gate_verdict": "near_miss_recovery_trend",
                "post_gate_best_delta_q_gain_from_gate": 0.2,
                "post_gate_final_support": 0.05,
                "post_gate_final_target_progress": 0.01,
                "post_gate_step_count": 2,
            },
            {
                "path_prefix_rank": 7,
                "post_gate_verdict": "post_gate_plateau",
                "post_gate_best_delta_q_gain_from_gate": 0.0,
                "post_gate_final_support": 0.02,
                "post_gate_final_target_progress": 0.01,
                "post_gate_step_count": 3,
            },
            {
                "path_prefix_rank": 8,
                "post_gate_verdict": "no_gate",
                "post_gate_best_delta_q_gain_from_gate": 9.0,
                "post_gate_final_support": 1.0,
                "post_gate_final_target_progress": 1.0,
                "post_gate_step_count": 1,
            },
        ]
    )

    selected = module._select_source_rows(
        rows,
        source_verdicts=("near_miss_recovery_trend", "post_gate_plateau"),
        max_sources=10,
        max_sources_per_prefix=1,
    )

    assert selected["path_prefix_rank"].tolist() == [6, 7]
    assert selected.iloc[0]["post_gate_best_delta_q_gain_from_gate"] == 0.2
    assert selected["source_case"].tolist() == ["p6_s1", "p7_s2"]


def test_source_screen_batch_discovers_and_filters_non_c0_dirs(tmp_path):
    module = _load_source_screen_batch_script_module()
    root = tmp_path / "root"
    root.mkdir()
    c0 = root / "basin_transition_post_gate_recovery_field34_cc_c0_v0"
    c2 = root / "basin_transition_post_gate_recovery_field34_cc_c2_v0"
    other = root / "unrelated"
    for path, pair_id in ((c0, "c0-s11-r0"), (c2, "c2-s11-r0")):
        path.mkdir()
        (path / "post_gate_recovery_path_summary_rows.csv").write_text(
            f"pair_id,post_gate_verdict\n{pair_id},near_miss_recovery_trend\n",
            encoding="utf-8",
        )
    other.mkdir()

    discovered = module._discover_post_gate_dirs(
        root,
        pattern="basin_transition_post_gate_recovery_field34_cc_*",
    )
    selected = module._select_post_gate_dirs(
        discovered,
        include_pair_prefixes=(),
        exclude_pair_prefixes=("c0",),
        max_artifacts=0,
    )

    assert discovered == (c0, c2)
    assert selected == (c2,)


def test_source_screen_batch_child_output_dir_names_are_local(tmp_path):
    module = _load_source_screen_batch_script_module()
    post_gate_dir = (
        tmp_path / "basin_transition_post_gate_recovery_field34_cc_c2_branch_v0"
    )

    out = module._screen_output_dir(tmp_path / "batch", post_gate_dir)

    assert out == (
        tmp_path / "batch" / "selector_source_screen_field34_cc_c2_branch_v0"
    )


def test_source_screen_batch_load_child_rows_handles_empty_csv(tmp_path):
    module = _load_source_screen_batch_script_module()
    screen_dir = tmp_path / "screen"
    screen_dir.mkdir()
    (screen_dir / "empty.csv").write_text("", encoding="utf-8")

    rows = module._load_child_rows(
        screen_dir,
        filename="empty.csv",
        post_gate_artifact="artifact",
        post_gate_dir=tmp_path / "post_gate",
    )

    assert rows.empty


def test_target_only_deltas_are_computed_per_move_kind():
    module = _load_script_module()
    rows = pd.DataFrame(
        [
            {
                "plan_kind": "target_core_only",
                "move_kind": "joint_mutable",
                "state_quality": 10.0,
                "operator_final_aligned_changed_support_node_count": 2,
            },
            {
                "plan_kind": "target_core_plus_boundary_core",
                "move_kind": "joint_mutable",
                "state_quality": 10.5,
                "operator_final_aligned_changed_support_node_count": 3,
            },
            {
                "plan_kind": "target_core_only",
                "move_kind": "candidate_bundle_transplant",
                "state_quality": 9.0,
                "operator_final_aligned_changed_support_node_count": 1,
            },
        ]
    )

    out = module._apply_target_only_deltas(rows)

    boundary = out[out["plan_kind"].eq("target_core_plus_boundary_core")].iloc[0]
    assert boundary["quality_gain_vs_target_only_same_move_kind"] == 0.5
    assert boundary["aligned_gain_vs_target_only_same_move_kind"] == 1.0
    candidate_base = out[
        (out["plan_kind"].eq("target_core_only"))
        & (out["move_kind"].eq("candidate_bundle_transplant"))
    ].iloc[0]
    assert candidate_base["quality_gain_vs_target_only_same_move_kind"] == 0.0


def test_handle_subset_comparisons_mark_minimal_sufficient_rows():
    module = _load_subset_script_module()
    rows = pd.DataFrame(
        [
            {
                "subset_size": 1,
                "target_node_count": 2,
                "state_quality": 9.0,
                "operator_delta_q_gain_vs_source": 0.1,
                "operator_final_aligned_changed_support_node_ids": "10",
            },
            {
                "subset_size": 2,
                "target_node_count": 2,
                "state_quality": 10.0,
                "operator_delta_q_gain_vs_source": 0.5,
                "operator_final_aligned_changed_support_node_ids": "10,20,30",
            },
        ]
    )

    out = module._add_full_set_comparisons(
        rows,
        required_nodes=np.asarray([10, 20, 30], dtype=np.uint32),
        quality_tolerance=1e-9,
    )

    assert out.iloc[0]["required_aligned_core_hit_count"] == 1
    assert not bool(out.iloc[0]["recovers_required_aligned_core"])
    assert out.iloc[1]["handle_subset_verdict"] == "sufficient_full_core_quality_match"


def test_select_stability_subsets_keeps_sufficient_and_best_near_misses():
    module = _load_stability_script_module()
    rows = pd.DataFrame(
        [
            {
                "subset_node_ids": "1,2",
                "subset_size": 2,
                "handle_subset_verdict": "sufficient_full_core_quality_match",
                "operator_delta_q_gain_vs_source": 0.5,
                "required_aligned_core_hit_count": 3,
                "quality_gap_vs_full_handle_set": 0.0,
            },
            {
                "subset_node_ids": "1",
                "subset_size": 1,
                "handle_subset_verdict": "partial_core_quality_gain",
                "operator_delta_q_gain_vs_source": 0.4,
                "required_aligned_core_hit_count": 2,
                "quality_gap_vs_full_handle_set": -0.1,
            },
            {
                "subset_node_ids": "2",
                "subset_size": 1,
                "handle_subset_verdict": "partial_core_quality_gain",
                "operator_delta_q_gain_vs_source": 0.3,
                "required_aligned_core_hit_count": 1,
                "quality_gap_vs_full_handle_set": -0.2,
            },
        ]
    )

    selected = module.select_stability_subsets(rows, max_partial_rows=1)

    assert list(selected["subset_node_ids"]) == ["1,2", "1"]
    assert selected.iloc[0]["subset_role"] == "full_handle_set"
    assert selected.iloc[1]["subset_role"] == "near_miss_size_1"


def test_stability_verdicts_require_core_and_quality_match():
    module = _load_stability_script_module()
    rows = pd.DataFrame(
        [
            {
                "recovers_required_aligned_core": True,
                "quality_gap_vs_full_handle_set": 0.0,
                "operator_delta_q_gain_vs_source": 0.5,
            },
            {
                "recovers_required_aligned_core": True,
                "quality_gap_vs_full_handle_set": -0.2,
                "operator_delta_q_gain_vs_source": 0.3,
            },
            {
                "recovers_required_aligned_core": False,
                "quality_gap_vs_full_handle_set": -0.5,
                "operator_delta_q_gain_vs_source": 0.1,
            },
        ]
    )

    out = module._with_stability_verdicts(rows, quality_tolerance=1e-9)

    assert list(out["stability_verdict"]) == [
        "stable_sufficient",
        "stable_core_quality_lag",
        "partial_core",
    ]


def test_selector_outcomes_and_summary_mark_k4_minimal_sufficient():
    module = _load_selector_script_module()
    plans = pd.DataFrame(
        [
            {
                "selector_policy": "context_pull",
                "selector_feature_family": "local_graph_proxy",
                "selector_uses_replay_features": False,
                "subset_size": 3,
                "selector_ordered_node_ids": "2,3,4",
                "subset_node_ids": "2,3,4",
            },
            {
                "selector_policy": "context_pull",
                "selector_feature_family": "local_graph_proxy",
                "selector_uses_replay_features": False,
                "subset_size": 4,
                "selector_ordered_node_ids": "2,3,4,5",
                "subset_node_ids": "2,3,4,5",
            },
        ]
    )
    subset_rows = pd.DataFrame(
        [
            {
                "subset_node_ids": "2,3,4",
                "subset_size": 3,
                "handle_subset_verdict": "partial_core_quality_gain",
                "recovers_required_aligned_core": False,
                "required_aligned_core_hit_count": 5,
                "required_aligned_core_hit_fraction": 0.8,
                "operator_delta_q_gain_vs_source": 0.2,
                "quality_gap_vs_full_handle_set": -0.1,
                "quality_gain_per_bundle_node": 0.07,
                "state_delta_q_vs_vanilla": -0.1,
                "operator_final_aligned_changed_support_node_ids": "1,2,3,4,5",
            },
            {
                "subset_node_ids": "2,3,4,5",
                "subset_size": 4,
                "handle_subset_verdict": "sufficient_full_core_quality_match",
                "recovers_required_aligned_core": True,
                "required_aligned_core_hit_count": 6,
                "required_aligned_core_hit_fraction": 1.0,
                "operator_delta_q_gain_vs_source": 0.3,
                "quality_gap_vs_full_handle_set": 0.0,
                "quality_gain_per_bundle_node": 0.075,
                "state_delta_q_vs_vanilla": 0.03,
                "operator_final_aligned_changed_support_node_ids": "1,2,3,4,5,6",
            },
        ]
    )
    stability = pd.DataFrame(
        [
            {
                "subset_node_ids": "2,3,4,5",
                "evaluation_count": 9,
                "stable_sufficient_count": 9,
                "stable_sufficient_fraction": 1.0,
            }
        ]
    )

    rows = module._with_selector_outcomes(
        plans,
        subset_rows=subset_rows,
        stability_summary_rows=stability,
    )
    summary = module._summary_rows(rows)

    assert rows.iloc[1]["matches_minimal_sufficient_subset"]
    assert summary.iloc[0]["first_sufficient_k"] == 4
    assert summary.iloc[0]["k4_stable_sufficient_fraction"] == 1.0
    assert summary.iloc[0]["k4_matches_minimal_sufficient_subset"]


def test_local_selector_required_core_verdicts_are_evaluation_only():
    module = _load_local_selector_script_module()
    rows = pd.DataFrame(
        [
            {
                "operator_final_aligned_changed_support_node_ids": "1,2,3",
                "operator_delta_q_gain_vs_source": 0.2,
                "quality_minus_best_same_randomness_control": 0.01,
            },
            {
                "operator_final_aligned_changed_support_node_ids": "1,2",
                "operator_delta_q_gain_vs_source": 0.2,
                "quality_minus_best_same_randomness_control": -0.5,
            },
        ]
    )

    out = module._with_required_core_verdicts(
        rows,
        required_nodes=np.asarray([1, 2, 3], dtype=np.uint32),
        min_material_q_gain=0.01,
    )

    assert out.iloc[0]["evaluation_required_core_hit_fraction"] == 1.0
    assert out.iloc[0]["local_selector_verdict"] == (
        "local_required_core_same_randomness_win"
    )
    assert out.iloc[1]["local_selector_verdict"] == "local_partial_core_quality_gain"


def test_local_selector_empty_core_uses_quality_control_verdicts():
    module = _load_local_selector_script_module()
    rows = pd.DataFrame(
        [
            {
                "operator_final_aligned_changed_support_node_ids": "1,2,3",
                "operator_delta_q_gain_vs_source": 0.2,
                "quality_minus_best_same_randomness_control": 0.01,
            },
            {
                "operator_final_aligned_changed_support_node_ids": "1,2",
                "operator_delta_q_gain_vs_source": 0.2,
                "quality_minus_best_same_randomness_control": -0.5,
            },
            {
                "operator_final_aligned_changed_support_node_ids": "",
                "operator_delta_q_gain_vs_source": 0.0,
                "quality_minus_best_same_randomness_control": 0.01,
            },
        ]
    )

    out = module._with_required_core_verdicts(
        rows,
        required_nodes=np.asarray([], dtype=np.uint32),
        min_material_q_gain=0.01,
    )

    assert out["evaluation_required_core_count"].eq(0).all()
    assert not out["evaluation_recovers_required_core"].any()
    assert list(out["local_selector_verdict"]) == [
        "local_quality_same_randomness_win",
        "local_material_quality_gain",
        "local_no_material_gain",
    ]
