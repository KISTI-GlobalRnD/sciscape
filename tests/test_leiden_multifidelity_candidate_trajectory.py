"""Tests for Leiden multi-fidelity candidate trajectory replay summaries."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "research" / "consensus" / "scripts"
ANALYSIS_PATH = SCRIPT_DIR / "analyze_leiden_multifidelity_candidate_trajectory.py"


def _load_script(module_name: str):
    if str(ANALYSIS_PATH.parent) not in sys.path:
        sys.path.insert(0, str(ANALYSIS_PATH.parent))
    spec = importlib.util.spec_from_file_location(module_name, ANALYSIS_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_rank_summary_marks_late_winner_entering_top2_at_p2():
    module = _load_script("leiden_multifidelity_candidate_trajectory_rank")
    labels = pd.DataFrame(
        [
            _label(0, p1_rank=1, p5_rank=2, p1=2.0, p5=2.1, winner=False),
            _label(1, p1_rank=2, p5_rank=3, p1=1.0, p5=1.1, winner=False),
            _label(2, p1_rank=3, p5_rank=1, p1=0.5, p5=3.2, winner=True),
        ]
    )
    run_rows = pd.DataFrame(
        [
            _run(0, 1, 2.0),
            _run(1, 1, 1.0),
            _run(2, 1, 0.5),
            _run(0, 2, 2.0),
            _run(1, 2, 1.0),
            _run(2, 2, 1.5),
            _run(0, 5, 2.1),
            _run(1, 5, 1.1),
            _run(2, 5, 3.2),
        ]
    )

    rank_summary = module.build_rank_summary(run_rows, labels)
    transition = module.build_transition_summary(rank_summary).iloc[0]

    assert transition["full_p5_winner_candidate_index"] == 2
    assert transition["winner_rank_at_p1"] == 3
    assert transition["winner_rank_at_p2"] == 2
    assert transition["winner_rank_at_p5"] == 1
    assert transition["first_replay_iterations_top2"] == 2


def test_rank_summary_tie_breaks_by_candidate_index():
    module = _load_script("leiden_multifidelity_candidate_trajectory_tie")
    labels = pd.DataFrame(
        [
            _label(0, p1_rank=1, p5_rank=1, p1=1.0, p5=1.0, winner=True),
            _label(1, p1_rank=2, p5_rank=2, p1=1.0, p5=1.0, winner=False),
        ]
    )
    run_rows = pd.DataFrame([_run(1, 1, 1.0), _run(0, 1, 1.0)])

    rank_summary = module.build_rank_summary(run_rows, labels)

    assert list(rank_summary["candidate_index"]) == [0, 1]
    assert list(rank_summary["replay_rank"]) == [1, 2]


def test_local_move_gain_rows_track_depth_gains():
    module = _load_script("leiden_multifidelity_candidate_trajectory_local_move")
    phase = pd.DataFrame(
        [
            _phase("run", iteration=2, depth=0, quality=101.0),
            _phase("run", iteration=2, depth=1, quality=101.5),
            _phase("run", iteration=2, depth=2, quality=101.75),
        ]
    )
    run_rows = pd.DataFrame(
        [
            {
                "run_id": "run",
                "case": "case",
                "seed": 11,
                "candidate_index": 2,
                "replay_iterations": 2,
                "baseline_quality": 100.0,
            }
        ]
    )

    gains = module.build_local_move_gain_rows(phase, run_rows)

    assert list(gains["quality_delta_vs_baseline"]) == [1.0, 1.5, 1.75]
    assert pd.isna(gains.iloc[0]["quality_gain_since_previous_local_move"])
    assert gains.iloc[1]["quality_gain_since_previous_local_move"] == 0.5
    assert gains.iloc[2]["quality_gain_since_previous_local_move"] == 0.25


def test_local_move_margin_summary_is_deterministic():
    module = _load_script("leiden_multifidelity_candidate_trajectory_move_margin")
    run_rows = pd.DataFrame([_run(2, 2, 1.0)])
    run_id = "case|seed=11|candidate=2|p2"
    events = [
        _local_move_margin(run_id, node=9, rank=2, margin=0.20, moved=True, best=4.0, second=3.8),
        _local_move_margin(run_id, node=7, rank=1, margin=0.00, moved=False, best=2.0, second=2.0),
        _local_move_margin(run_id, node=8, rank=0, margin=0.05, moved=True, best=3.0, second=2.95),
    ]

    summary = module.build_local_move_margin_summary(events, run_rows)
    row = summary.iloc[0]

    assert int(row["event_count"]) == 3
    assert int(row["moved_count"]) == 2
    assert int(row["near_zero_margin_count"]) == 1
    assert row["margin_min"] == 0.0
    assert row["margin_p50"] == 0.05
    assert row["best_increment_max"] == 4.0
    assert row["second_increment_min"] == 2.0
    assert row["top_low_margin_node_ids"] == "7,8,9"
    assert row["top_moved_low_margin_node_ids"] == "8,9"


def test_local_merge_parent_summary_sorts_parent_lists_stably():
    module = _load_script("leiden_multifidelity_candidate_trajectory_merge_margin")
    run_rows = pd.DataFrame([_run(2, 2, 1.0)])
    run_id = "case|seed=11|candidate=2|p2"
    events = [
        _local_merge_margin(run_id, parent_id=20, decision=5, low=3, changed=0, min_margin=0.2, largest=0.4),
        _local_merge_margin(run_id, parent_id=10, decision=10, low=3, changed=1, min_margin=0.1, largest=0.6),
        _local_merge_margin(run_id, parent_id=5, decision=20, low=0, changed=0, min_margin=0.01, largest=0.9),
    ]

    summary = module.build_local_merge_parent_summary(events, run_rows)
    row = summary.iloc[0]

    assert int(row["parent_row_count"]) == 3
    assert row["decision_count"] == 35.0
    assert row["low_margin_count"] == 6.0
    assert row["changed_count"] == 1.0
    assert row["min_margin_min"] == 0.01
    assert row["largest_child_fraction_max"] == 0.9
    assert row["top_low_margin_parent_ids"] == "10,20"
    assert row["top_decision_parent_ids"] == "5,10,20"
    assert row["top_small_margin_parent_ids"] == "5,10,20"
    assert row["top_largest_child_fraction_parent_ids"] == "5,10,20"


def test_depth_attribution_classifies_local_move_signal():
    module = _load_script("leiden_multifidelity_candidate_trajectory_move_signal")
    gain = pd.DataFrame([_gain(2, 2, iteration=2, depth=1, gain=0.636)])
    local_move = pd.DataFrame(
        [
            _move_summary(2, 2, moved=3, zeros=2, p50=0.001),
            _move_summary(0, 2, moved=0, zeros=0, p50=0.010),
            _move_summary(1, 2, moved=0, zeros=0, p50=0.011),
        ]
    )

    attribution = module.classify_depth_attribution(
        local_move_margin_summary=local_move,
        local_merge_parent_summary=pd.DataFrame(),
        local_move_gain=gain,
    )

    assert attribution["classification"] == "local_move_margin_signal"
    assert attribution["instrumentation_gate_open"] is False


def test_depth_attribution_classifies_local_merge_parent_signal():
    module = _load_script("leiden_multifidelity_candidate_trajectory_merge_signal")
    gain = pd.DataFrame([_gain(2, 2, iteration=2, depth=1, gain=0.636)])
    local_move = pd.DataFrame(
        [
            _move_summary(2, 2, moved=0, zeros=0, p50=0.010),
            _move_summary(0, 2, moved=0, zeros=0, p50=0.010),
            _move_summary(1, 2, moved=0, zeros=0, p50=0.010),
        ]
    )
    local_merge = pd.DataFrame(
        [
            _merge_summary(2, 2, iteration=2, depth=1, low=4, top="2867,5121,1931,2678"),
            _merge_summary(2, 1, iteration=1, depth=1, low=2, top="1931"),
            _merge_summary(0, 2, iteration=2, depth=1, low=1, top="1931"),
            _merge_summary(1, 2, iteration=2, depth=1, low=1, top="1931"),
        ]
    )

    attribution = module.classify_depth_attribution(
        local_move_margin_summary=local_move,
        local_merge_parent_summary=local_merge,
        local_move_gain=gain,
    )

    assert attribution["classification"] == "local_merge_parent_signal"
    assert attribution["target_local_merge_top_low_margin_parent_ids"] == "2867,5121,1931,2678"
    assert attribution["instrumentation_gate_open"] is False


def test_empty_trace_writes_empty_margin_csvs_and_no_signal_report(tmp_path):
    module = _load_script("leiden_multifidelity_candidate_trajectory_empty_trace")
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text("", encoding="utf-8")
    run_rows = pd.DataFrame([_run(2, 2, 1.0)])
    gain = pd.DataFrame([_gain(2, 2, iteration=2, depth=1, gain=0.636)])

    _, _, attribution = module.write_trace_margin_outputs(
        trajectory_path=trace_path,
        run_rows=run_rows,
        local_move_gain=gain,
        local_move_margin_summary_path=tmp_path / "move.csv",
        local_merge_parent_summary_path=tmp_path / "merge.csv",
        depth_attribution_report_path=tmp_path / "report.md",
    )

    assert attribution["classification"] == "no_existing_trace_signal"
    assert pd.read_csv(tmp_path / "move.csv").empty
    assert pd.read_csv(tmp_path / "merge.csv").empty
    assert "Classification: `no_existing_trace_signal`" in (tmp_path / "report.md").read_text()


def test_target_parent_events_include_phase_context_and_filter_parents():
    module = _load_script("leiden_multifidelity_candidate_trajectory_parent_events")
    run_rows = pd.DataFrame([_run(2, 2, 1.0)])
    run_id = "case|seed=11|candidate=2|p2"
    phase = pd.DataFrame(
        [
            _phase_with_kind(run_id, iteration=2, depth=1, phase="after_local_move", quality=101.0),
            _phase_with_kind(run_id, iteration=2, depth=1, phase="after_refinement", quality=99.0),
            _phase_with_kind(
                run_id,
                iteration=2,
                depth=1,
                phase="after_aggregation_refined",
                quality=101.0,
            ),
        ]
    )
    gain = pd.DataFrame([_gain(2, 2, iteration=2, depth=1, gain=0.636)])
    events = [
        _local_merge_margin(run_id, parent_id=10, decision=4, low=1, changed=0, min_margin=0.0, largest=1.0),
        _local_merge_margin(run_id, parent_id=11, decision=4, low=1, changed=0, min_margin=0.0, largest=1.0),
    ]

    rows = module.build_target_parent_event_rows(
        events,
        run_rows=run_rows,
        phase_frame=phase,
        local_move_gain=gain,
        target_parent_ids=[10],
    )

    assert list(rows["parent_id"]) == [10]
    row = rows.iloc[0]
    assert row["context_role"] == "target"
    assert row["after_local_move_quality"] == 101.0
    assert row["after_refinement_quality"] == 99.0
    assert row["after_aggregation_phase"] == "after_aggregation_refined"
    assert row["quality_gain_since_previous_local_move"] == 0.636


def test_target_parent_contrast_orders_contexts_and_computes_deltas():
    module = _load_script("leiden_multifidelity_candidate_trajectory_parent_contrast")
    run_rows = pd.DataFrame(
        [
            _run(2, 2, 1.0),
            _run(0, 2, 0.7),
            _run(2, 3, 1.2),
        ]
    )
    gain = pd.DataFrame(
        [
            _gain(2, 2, iteration=2, depth=0, gain=0.1),
            _gain(2, 2, iteration=2, depth=1, gain=0.636),
            _gain(0, 2, iteration=2, depth=1, gain=0.2),
            _gain(2, 3, iteration=2, depth=1, gain=0.636),
        ]
    )
    events = pd.DataFrame(
        [
            _target_parent_event_row("target", 2, 2, 2, 1, 10, low=4, min_margin=0.0),
            _target_parent_event_row("target_pre_window", 2, 2, 2, 0, 10, low=1, min_margin=0.2),
            _target_parent_event_row("peer_candidate_0", 0, 2, 2, 1, 10, low=2, min_margin=0.1),
            _target_parent_event_row("candidate2_p3_same_window", 2, 3, 2, 1, 10, low=4, min_margin=0.0),
        ]
    )

    contrast = module.build_target_parent_contrast(
        events,
        run_rows=run_rows,
        local_move_gain=gain,
        target_parent_ids=[10],
    )

    assert list(contrast["context_role"]) == [
        "target",
        "target_pre_window",
        "peer_candidate_0",
        "candidate2_p3_same_window",
    ]
    peer = contrast[contrast["context_role"].eq("peer_candidate_0")].iloc[0]
    assert peer["target_low_margin_delta"] == 2.0
    assert peer["target_min_margin_delta"] == -0.1


def test_parent_causal_window_classifications_are_deterministic():
    module = _load_script("leiden_multifidelity_candidate_trajectory_parent_classify")
    gain = pd.DataFrame([_gain(2, 2, iteration=2, depth=1, gain=0.636)])

    post_contrast = pd.DataFrame(
        [
            _parent_contrast("target", low=4),
            _parent_contrast("target_pre_window", low=1),
            _parent_contrast("candidate2_p1_depth1", low=2),
            _parent_contrast("peer_candidate_0", low=1),
        ]
    )
    post_events = pd.DataFrame(
        [
            _target_parent_event_row("target_pre_window", 2, 2, 2, 0, 10, low=1, min_margin=0.1),
        ]
    )
    post = module.classify_parent_causal_window(
        target_parent_events=post_events,
        target_parent_contrast=post_contrast,
        local_move_gain=gain,
    )
    assert post["classification"] == "post_gain_parent_symptom"

    pre_contrast = pd.DataFrame(
        [
            _parent_contrast("target", low=4),
            _parent_contrast("target_pre_window", low=4),
        ]
    )
    pre_events = pd.DataFrame(
        [
            _target_parent_event_row("target_pre_window", 2, 2, 2, 0, 10, low=4, min_margin=0.0),
        ]
    )
    pre = module.classify_parent_causal_window(
        target_parent_events=pre_events,
        target_parent_contrast=pre_contrast,
        local_move_gain=gain,
    )
    assert pre["classification"] == "pre_gain_parent_setup"

    ambiguous = module.classify_parent_causal_window(
        target_parent_events=pd.DataFrame(),
        target_parent_contrast=pd.DataFrame([_parent_contrast("target", low=0)]),
        local_move_gain=gain,
    )
    assert ambiguous["classification"] == "ambiguous_parent_signal"
    assert ambiguous["instrumentation_gate_open"] is True


def test_one_hop_neighbor_extraction_is_deterministic():
    module = _load_script("leiden_multifidelity_candidate_trajectory_neighbors")

    neighbors = module._one_hop_neighbor_nodes(
        src=[3, 1, 2, 4, 5, 0],
        dst=[1, 0, 3, 2, 3, 6],
        target_nodes=[2, 1, 1],
    )

    assert neighbors == [0, 3, 4]


def test_hop_distance_sets_are_deterministic():
    module = _load_script("leiden_multifidelity_candidate_trajectory_hops")

    hops = module._hop_distance_sets(
        src=[0, 1, 2, 3, 5],
        dst=[1, 2, 3, 4, 2],
        target_nodes=[0, 0],
        max_hop=2,
    )

    assert {hop: sorted(nodes) for hop, nodes in hops.items()} == {
        0: [0],
        1: [1],
        2: [2],
    }


def test_local_move_focus_context_sets_and_restores_env(monkeypatch):
    module = _load_script("leiden_multifidelity_candidate_trajectory_focus_env")
    monkeypatch.setenv(module.LOCAL_MOVE_FOCUS_NODE_ENV, "old-target")
    monkeypatch.setenv(module.LOCAL_MOVE_NEIGHBOR_NODE_ENV, "old-neighbor")

    with module._local_move_focus_context(
        target_nodes=[3, 1, 3],
        neighbor_nodes=[9, 4],
    ):
        assert module.os.environ[module.LOCAL_MOVE_FOCUS_NODE_ENV] == "1,3"
        assert module.os.environ[module.LOCAL_MOVE_NEIGHBOR_NODE_ENV] == "4,9"

    assert module.os.environ[module.LOCAL_MOVE_FOCUS_NODE_ENV] == "old-target"
    assert module.os.environ[module.LOCAL_MOVE_NEIGHBOR_NODE_ENV] == "old-neighbor"


def test_local_move_focus_outputs_summarize_synthetic_trace(tmp_path):
    module = _load_script("leiden_multifidelity_candidate_trajectory_focus_outputs")
    trace_path = tmp_path / "trajectory.jsonl"
    run_id = "case|seed=11|candidate=2|p2"
    events = [
        _local_move_focus(run_id, node=7, role="target", moved=True, margin=0.01),
        _local_move_focus(run_id, node=8, role="neighbor", moved=True, margin=0.02),
        _local_move_focus(run_id, node=9, role="neighbor", moved=False, margin=0.03),
    ]
    trace_path.write_text(
        "\n".join(module.json.dumps(event) for event in events) + "\n",
        encoding="utf-8",
    )
    run_rows = pd.DataFrame([_run(2, 2, 1.0)])
    gain = pd.DataFrame([_gain(2, 2, iteration=2, depth=1, gain=0.636)])

    event_rows, summary, attribution = module.write_local_move_focus_outputs(
        trajectory_path=trace_path,
        run_rows=run_rows,
        local_move_gain=gain,
        local_move_focus_events_path=tmp_path / "focus_events.csv",
        local_move_focus_summary_path=tmp_path / "focus_summary.csv",
        local_move_movement_report_path=tmp_path / "movement.md",
    )

    assert len(event_rows) == 3
    row = summary.iloc[0]
    assert row["target_moved_count"] == 1
    assert row["neighbor_moved_count"] == 1
    assert row["target_moved_node_ids"] == "7"
    assert row["neighbor_moved_node_ids"] == "8"
    assert row["moved_node_ids"] == "7,8"
    assert row["moved_overlap_target_window_count"] == 2
    assert attribution["classification"] == "target_node_move_signal"
    assert "Classification: `target_node_move_signal`" in (tmp_path / "movement.md").read_text()
    assert not pd.read_csv(tmp_path / "focus_events.csv").empty
    assert not pd.read_csv(tmp_path / "focus_summary.csv").empty


def test_local_move_movement_classification_covers_signal_types():
    module = _load_script("leiden_multifidelity_candidate_trajectory_focus_classify")
    gain = pd.DataFrame([_gain(2, 2, iteration=2, depth=1, gain=0.636)])

    target = module.classify_local_move_movement_attribution(
        local_move_focus_summary=pd.DataFrame(
            [_focus_summary(2, 2, target_moved=1, neighbor_moved=0)]
        ),
        local_move_gain=gain,
    )
    assert target["classification"] == "target_node_move_signal"

    neighbor = module.classify_local_move_movement_attribution(
        local_move_focus_summary=pd.DataFrame(
            [
                _focus_summary(2, 2, target_moved=0, neighbor_moved=3),
                _focus_summary(2, 1, target_moved=0, neighbor_moved=1, iteration=1),
                _focus_summary(0, 2, target_moved=0, neighbor_moved=2),
            ]
        ),
        local_move_gain=gain,
    )
    assert neighbor["classification"] == "neighbor_node_move_signal"

    none = module.classify_local_move_movement_attribution(
        local_move_focus_summary=pd.DataFrame(
            [
                _focus_summary(2, 2, target_moved=0, neighbor_moved=1),
                _focus_summary(0, 2, target_moved=0, neighbor_moved=2),
            ]
        ),
        local_move_gain=gain,
    )
    assert none["classification"] == "no_focused_move_signal"
    assert none["instrumentation_gate_open"] is True


def test_perturbation_footprint_rows_summarize_synthetic_memberships(tmp_path):
    module = _load_script("leiden_multifidelity_candidate_trajectory_footprint")

    event_rows, summary = _footprint_rows(
        module,
        observed=[0, 4, 4, 4, 4, 4],
        contrast_kind="extra",
    )

    assert len(event_rows) == 4
    assert summary["changed_nodes_total"] == 3
    assert summary["changed_hop0_count"] == 1
    assert summary["changed_hop1_count"] == 1
    assert summary["changed_hop2_count"] == 1
    assert summary["changed_hop3plus_count"] == 0
    assert summary["fraction_changed_within_2hop"] == 1.0
    assert summary["classification"] == "local_2hop"

    report_path = tmp_path / "footprint.md"
    module.write_perturbation_footprint_report(
        report_path,
        footprint_summary=pd.DataFrame([summary]),
        extra_contrast=True,
    )
    assert "Classification Counts" in report_path.read_text(encoding="utf-8")


def test_perturbation_footprint_aligns_cluster_labels_before_diffing():
    module = _load_script("leiden_multifidelity_candidate_trajectory_footprint_align")

    _, summary = _footprint_rows(
        module,
        observed=[10, 11, 12, 13, 14, 14],
        contrast_kind="extra",
    )

    assert summary["changed_nodes_total"] == 0
    assert summary["classification"] == "unknown_insufficient_trace"


def test_perturbation_footprint_classification_covers_scopes():
    module = _load_script("leiden_multifidelity_candidate_trajectory_footprint_classify")

    _, local1 = _footprint_rows(module, observed=[0, 4, 4, 3, 4, 4])
    assert local1["classification"] == "local_1hop"

    _, local2 = _footprint_rows(module, observed=[0, 4, 4, 4, 4, 4])
    assert local2["classification"] == "local_2hop"

    parent = module.classify_perturbation_footprint(
        {
            "changed_nodes_total": 4,
            "fraction_changed_within_1hop": 0.25,
            "fraction_changed_within_2hop": 0.50,
            "fraction_changed_source_or_target_initial": 1.0,
            "contrast_kind": "extra",
        }
    )
    assert parent == "parent_local"

    _, diffuse = _footprint_rows(module, observed=[0, 1, 2, 3, 4, 0])
    assert diffuse["classification"] == "diffuse_global"

    _, unknown = _footprint_rows(
        module,
        observed=[0, 1, 2, 3, 4, 0],
        contrast_kind="baseline",
    )
    assert unknown["classification"] == "unknown_insufficient_trace"


def test_drilldown_only_requires_existing_trace_outputs(tmp_path):
    module = _load_script("leiden_multifidelity_candidate_trajectory_drilldown_missing")
    args = type(
        "Args",
        (),
        {
            "output_dir": tmp_path,
            "target_parent_ids": "10",
        },
    )()

    with pytest.raises(FileNotFoundError, match="--drilldown-only requires"):
        module.run_parent_drilldown_only(args)


def _label(
    candidate_index: int,
    *,
    p1_rank: int,
    p5_rank: int,
    p1: float,
    p5: float,
    winner: bool,
) -> dict[str, object]:
    return {
        "candidate_index": candidate_index,
        "p1_rank": p1_rank,
        "p5_rank": p5_rank,
        "p1_delta_q": p1,
        "p5_delta_q": p5,
        "is_full_p5_winner": winner,
        "group_kind": "best",
        "group_count": 2,
    }


def _run(candidate_index: int, replay_iterations: int, delta_q: float) -> dict[str, object]:
    return {
        "case": "case",
        "seed": 11,
        "candidate_index": candidate_index,
        "replay_iterations": replay_iterations,
        "delta_q": delta_q,
        "quality": 100.0 + delta_q,
        "elapsed_sec": 0.1,
        "run_id": f"case|seed=11|candidate={candidate_index}|p{replay_iterations}",
    }


def _phase(run_id: str, *, iteration: int, depth: int, quality: float) -> dict[str, object]:
    return {
        "schema": "dongdaemun_trajectory_trace.v1",
        "event": "phase_checkpoint",
        "run_id": run_id,
        "depth": depth,
        "iteration": iteration,
        "phase": "after_local_move",
        "membership_hash": f"h{depth}",
        "n_clusters": 10 + depth,
        "quality": quality,
    }


def _phase_with_kind(
    run_id: str,
    *,
    iteration: int,
    depth: int,
    phase: str,
    quality: float,
) -> dict[str, object]:
    row = _phase(run_id, iteration=iteration, depth=depth, quality=quality)
    row["phase"] = phase
    return row


def _local_move_margin(
    run_id: str,
    *,
    node: int,
    rank: int,
    margin: float,
    moved: bool,
    best: float,
    second: float,
) -> dict[str, object]:
    return {
        "event": "local_move_margin",
        "run_id": run_id,
        "iteration": 2,
        "depth": 1,
        "rank": rank,
        "node": node,
        "current_cluster": 1,
        "best_cluster": 2,
        "second_cluster": 3,
        "best_increment": best,
        "second_increment": second,
        "margin": margin,
        "moved": moved,
    }


def _local_move_focus(
    run_id: str,
    *,
    node: int,
    role: str,
    moved: bool,
    margin: float,
) -> dict[str, object]:
    return {
        "event": "local_move_focus_node",
        "run_id": run_id,
        "iteration": 2,
        "depth": 1,
        "node": node,
        "role": role,
        "current_cluster": 1,
        "best_cluster": 2 if moved else 1,
        "second_cluster": 3,
        "best_increment": 2.0,
        "second_increment": 2.0 - margin,
        "margin": margin,
        "moved": moved,
    }


def _local_merge_margin(
    run_id: str,
    *,
    parent_id: int,
    decision: int,
    low: int,
    changed: int,
    min_margin: float,
    largest: float,
) -> dict[str, object]:
    return {
        "event": "local_merge_margin_summary",
        "run_id": run_id,
        "iteration": 2,
        "depth": 1,
        "parent_id": parent_id,
        "parent_visit_index": 0,
        "source": "standard_refinement",
        "parent_size": decision,
        "parent_weight": float(decision),
        "decision_count": decision,
        "low_margin_decision_count": low,
        "changed_decision_count": changed,
        "min_margin": min_margin,
        "p10_margin": min_margin + 0.01,
        "p50_margin": min_margin + 0.02,
        "selected_child_count": 2,
        "largest_child_fraction": largest,
    }


def _target_parent_event_row(
    context_role: str,
    candidate_index: int,
    replay_iterations: int,
    iteration: int,
    depth: int,
    parent_id: int,
    *,
    low: int,
    min_margin: float,
) -> dict[str, object]:
    run_id = f"case|seed=11|candidate={candidate_index}|p{replay_iterations}"
    return {
        "context_role": context_role,
        "case": "case",
        "seed": 11,
        "candidate_index": candidate_index,
        "replay_iterations": replay_iterations,
        "run_id": run_id,
        "iteration": iteration,
        "depth": depth,
        "parent_id": parent_id,
        "parent_visit_index": 0,
        "source": "standard_refinement",
        "parent_size": 4,
        "parent_weight": 4.0,
        "decision_count": 4.0,
        "low_margin_decision_count": float(low),
        "changed_decision_count": 0.0,
        "min_margin": min_margin,
        "p10_margin": min_margin,
        "p50_margin": min_margin,
        "selected_child_count": 1,
        "largest_child_fraction": 1.0,
        "after_local_move_quality": 101.0,
        "after_local_move_membership_hash": "h-local",
        "after_refinement_quality": 99.0,
        "after_refinement_membership_hash": "h-refined",
        "after_aggregation_phase": "after_aggregation_refined",
        "after_aggregation_quality": 101.0,
        "after_aggregation_membership_hash": "h-agg",
        "quality_gain_since_previous_local_move": 0.636,
    }


def _parent_contrast(context_role: str, *, low: int) -> dict[str, object]:
    return {
        "context_role": context_role,
        "candidate_index": 2,
        "replay_iterations": 2,
        "iteration": 2,
        "depth": 1,
        "parent_id": 10,
        "low_margin_decision_count": float(low),
    }


def _footprint_rows(
    module,
    *,
    observed: list[int],
    contrast_kind: str = "extra",
) -> tuple[list[dict[str, object]], dict[str, object]]:
    baseline = np.asarray([0, 1, 2, 3, 4, 4], dtype=np.uint64)
    return module.build_perturbation_footprint_rows(
        case="case",
        seed=11,
        candidate_index=2,
        replay_iterations=2,
        run_id="case|seed=11|candidate=2|p2",
        contrast_kind=contrast_kind,
        contrast_run_id="contrast",
        source_cluster=1,
        target_cluster=2,
        target_nodes=[1],
        hop_sets={0: {1}, 1: {2}, 2: {3}},
        baseline_membership=baseline,
        reference_membership=baseline,
        perturb_membership=np.asarray(observed, dtype=np.uint64),
        baseline_quality=100.0,
        reference_quality=100.1,
        perturb_quality=100.2,
    )


def _gain(
    candidate_index: int,
    replay_iterations: int,
    *,
    iteration: int,
    depth: int,
    gain: float,
) -> dict[str, object]:
    return {
        "case": "case",
        "seed": 11,
        "candidate_index": candidate_index,
        "replay_iterations": replay_iterations,
        "run_id": f"case|seed=11|candidate={candidate_index}|p{replay_iterations}",
        "iteration": iteration,
        "depth": depth,
        "quality_gain_since_previous_local_move": gain,
    }


def _move_summary(
    candidate_index: int,
    replay_iterations: int,
    *,
    moved: int,
    zeros: int,
    p50: float,
) -> dict[str, object]:
    return {
        "case": "case",
        "seed": 11,
        "candidate_index": candidate_index,
        "replay_iterations": replay_iterations,
        "run_id": f"case|seed=11|candidate={candidate_index}|p{replay_iterations}",
        "iteration": 2,
        "depth": 1,
        "moved_count": moved,
        "near_zero_margin_count": zeros,
        "margin_p50": p50,
        "top_low_margin_node_ids": "",
    }


def _focus_summary(
    candidate_index: int,
    replay_iterations: int,
    *,
    target_moved: int,
    neighbor_moved: int,
    iteration: int = 2,
    depth: int = 1,
) -> dict[str, object]:
    return {
        "case": "case",
        "seed": 11,
        "candidate_index": candidate_index,
        "replay_iterations": replay_iterations,
        "run_id": f"case|seed=11|candidate={candidate_index}|p{replay_iterations}",
        "iteration": iteration,
        "depth": depth,
        "quality_gain_since_previous_local_move": 0.636,
        "target_event_count": target_moved,
        "target_moved_count": target_moved,
        "target_moved_node_ids": "7" if target_moved else "",
        "target_margin_min": 0.01,
        "target_margin_p50": 0.01,
        "neighbor_event_count": neighbor_moved,
        "neighbor_moved_count": neighbor_moved,
        "neighbor_moved_node_ids": "8" if neighbor_moved else "",
        "neighbor_margin_min": 0.02,
        "neighbor_margin_p50": 0.02,
        "moved_count": target_moved + neighbor_moved,
        "moved_node_ids": "7,8",
        "moved_margin_min": 0.01,
        "moved_margin_p50": 0.02,
        "best_increment_min": 2.0,
        "best_increment_p50": 2.0,
        "best_increment_max": 2.0,
        "second_increment_min": 1.9,
        "second_increment_p50": 1.9,
        "second_increment_max": 1.9,
        "moved_overlap_previous_window_count": 0,
        "moved_overlap_next_window_count": 0,
        "moved_overlap_target_window_count": target_moved + neighbor_moved,
    }


def _merge_summary(
    candidate_index: int,
    replay_iterations: int,
    *,
    iteration: int,
    depth: int,
    low: int,
    top: str,
) -> dict[str, object]:
    return {
        "case": "case",
        "seed": 11,
        "candidate_index": candidate_index,
        "replay_iterations": replay_iterations,
        "run_id": f"case|seed=11|candidate={candidate_index}|p{replay_iterations}",
        "iteration": iteration,
        "depth": depth,
        "parent_row_count": 3,
        "decision_count": 10.0,
        "low_margin_count": float(low),
        "changed_count": 0.0,
        "min_margin_min": 0.0,
        "largest_child_fraction_max": 1.0,
        "top_low_margin_parent_ids": top,
    }
