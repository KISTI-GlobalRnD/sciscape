"""Tests for Leiden hysteresis work-acceleration monitor helpers."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "research" / "consensus" / "scripts"
ANALYSIS_PATH = Path(__file__).resolve().parents[1] / "research/consensus/scripts/leiden_basin/hysteresis/analyze_leiden_hysteresis_work_acceleration.py"
MONITOR_PATH = Path(__file__).resolve().parents[1] / "research/consensus/scripts/leiden_basin/hysteresis/run_leiden_hysteresis_work_acceleration_monitor.py"


def _load_script(path: Path, module_name: str):
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_target_policy_reports_did_not_reach_for_branch_independent_target():
    module = _load_script(ANALYSIS_PATH, "leiden_work_acceleration_analysis_for_test")
    extra = pd.DataFrame(
        [
            _point("extra", 0, 0, 0.0),
            _point("extra", 1, 10, 50.0),
        ]
    )
    perturb = pd.DataFrame(
        [
            _point("perturb", 0, 0, -5.0),
            _point("perturb", 1, 5, 20.0),
        ]
    )

    row = module._score_target_policy(
        extra_points=extra,
        perturb_points=perturb,
        target_policy="baseline_plus_25ppm",
        target_ppm=25.0,
    )

    assert row["extra_tau_status"] == "reached"
    assert row["perturb_tau_status"] == "did_not_reach_target"
    assert pd.isna(row["k_work_saving_pct"])


def test_build_monitor_graph_probe_only_uses_raw_sidecar_paths(tmp_path, monkeypatch):
    module = _load_script(
        MONITOR_PATH, "leiden_work_acceleration_monitor_for_test_probe_graph"
    )
    node_weights_path = tmp_path / "node_weights.f64.bin"
    np.arange(1, 4, dtype=np.float64).tofile(node_weights_path)
    calls: list[dict[str, object]] = []

    def fake_build_leiden_graph(**kwargs):
        calls.append(kwargs)
        return "graph"

    def fail_load_graph_arrays(_graph_dir):
        raise AssertionError(
            "_load_graph_arrays should not run for probe-only graph loading"
        )

    monkeypatch.setattr(module, "build_leiden_graph", fake_build_leiden_graph)
    monkeypatch.setattr(module, "_load_graph_arrays", fail_load_graph_arrays)

    graph, node_weights, arrays = module._build_monitor_graph(tmp_path, probe_only=True)

    assert graph == "graph"
    assert arrays is None
    assert np.asarray(node_weights).tolist() == [1.0, 2.0, 3.0]
    assert calls == [
        {
            "edge_path": tmp_path / "int_edges.parquet",
            "n_nodes": 3,
            "node_weights_path": node_weights_path,
        }
    ]
    module._release_memmap_array(node_weights)


def test_release_memmap_array_closes_backing_mmap():
    module = _load_script(
        MONITOR_PATH, "leiden_work_acceleration_monitor_for_test_memmap_release"
    )
    closed = []

    class FakeMmap:
        def close(self):
            closed.append(True)

    class FakeArray:
        _mmap = FakeMmap()

    module._release_memmap_array(FakeArray())
    module._release_memmap_array(np.asarray([1.0], dtype=np.float64))

    assert closed == [True]


def test_build_monitor_graph_non_probe_keeps_reconstruct_arrays(monkeypatch):
    module = _load_script(
        MONITOR_PATH,
        "leiden_work_acceleration_monitor_for_test_non_probe_graph",
    )
    arrays = type(
        "Arrays",
        (),
        {
            "src": np.asarray([0], dtype=np.uint32),
            "dst": np.asarray([1], dtype=np.uint32),
            "weight": np.asarray([1.0], dtype=np.float64),
            "node_weights": np.asarray([1.0, 1.0], dtype=np.float64),
        },
    )()
    calls: list[dict[str, object]] = []

    def fake_load_graph_arrays(_graph_dir):
        return arrays

    def fake_build_leiden_graph(**kwargs):
        calls.append(kwargs)
        return "graph"

    monkeypatch.setattr(module, "_load_graph_arrays", fake_load_graph_arrays)
    monkeypatch.setattr(module, "build_leiden_graph", fake_build_leiden_graph)

    graph, node_weights, loaded_arrays = module._build_monitor_graph(
        Path("graph"),
        probe_only=False,
    )

    assert graph == "graph"
    assert loaded_arrays is arrays
    assert node_weights is arrays.node_weights
    assert len(calls) == 1
    assert calls[0]["edges_src"] is arrays.src
    assert calls[0]["edges_dst"] is arrays.dst
    assert calls[0]["edges_weight"] is arrays.weight
    assert calls[0]["n_nodes"] == 2
    assert calls[0]["node_weights"] is arrays.node_weights


def test_target_policy_marks_zero_inside_min_as_degenerate():
    module = _load_script(
        ANALYSIS_PATH, "leiden_work_acceleration_analysis_for_test_zero"
    )
    extra = pd.DataFrame([_point("extra", 0, 0, 0.0)])
    perturb = pd.DataFrame([_point("perturb", 0, 0, 0.0)])

    row = module._score_target_policy(
        extra_points=extra,
        perturb_points=perturb,
        target_policy="inside_min_10ppm",
        target_ppm=0.0,
    )

    assert row["extra_tau_status"] == "degenerate_zero_target"
    assert row["perturb_tau_status"] == "degenerate_zero_target"
    assert pd.isna(row["k_work_saving_pct"])


def test_scorecard_elapsed_cost_excludes_perturb_trace_replay():
    module = _load_script(MONITOR_PATH, "leiden_work_acceleration_monitor_for_test")
    points = pd.DataFrame(
        [
            _point(
                "case|seed=11|budget=3|extra",
                0,
                0,
                0.0,
                case="case",
                seed=11,
                budget=3,
                branch="extra",
            ),
            _point(
                "case|seed=11|budget=3|extra",
                1,
                10,
                30.0,
                case="case",
                seed=11,
                budget=3,
                branch="extra",
            ),
            _point(
                "case|seed=11|budget=3|perturb",
                0,
                0,
                -10.0,
                case="case",
                seed=11,
                budget=3,
                branch="perturb",
            ),
            _point(
                "case|seed=11|budget=3|perturb",
                1,
                5,
                30.0,
                case="case",
                seed=11,
                budget=3,
                branch="perturb",
            ),
        ]
    )
    run_rows = pd.DataFrame(
        [
            {
                "case": "case",
                "seed": 11,
                "candidate_budget": 3,
                "branch": "extra",
                "elapsed_sec": 10.0,
                "baseline_quality": 100.0,
                "quality": 100.003,
            },
            {
                "case": "case",
                "seed": 11,
                "candidate_budget": 3,
                "branch": "perturb",
                "elapsed_sec": 99.0,
                "baseline_quality": 100.0,
                "quality": 100.003,
                "candidate_cluster_selection_elapsed_sec": 1.0,
                "candidate_probe_elapsed_sec": 4.0,
                "group_count": 1,
            },
        ]
    )

    scorecard = module._scorecard(points, run_rows)
    row = scorecard[scorecard["target_policy"] == "extra_p5_final"].iloc[0]

    assert row["extra_tau_status"] == "reached"
    assert row["perturb_tau_status"] == "reached"
    assert row["k_work_saving_pct"] == 50.0
    assert row["operational_perturb_elapsed_sec"] == 5.0
    assert row["net_elapsed_saving_pct"] == 50.0
    assert row["analysis_trace_overhead_sec"] == 99.0
    assert row["group_size_class"] == "single_node"


def test_first_divergence_rows_include_budget_and_moved_nodes():
    module = _load_script(
        MONITOR_PATH, "leiden_work_acceleration_monitor_for_test_divergence"
    )
    phase_frame = pd.DataFrame(
        [
            _checkpoint(
                "case|seed=11|budget=1|extra", 1, "after_local_move", "h0", 10.0
            ),
            _checkpoint(
                "case|seed=11|budget=1|perturb", 1, "after_local_move", "h0", 10.0
            ),
            _checkpoint(
                "case|seed=11|budget=1|extra", 1, "after_refinement", "h1", 10.5
            ),
            _checkpoint(
                "case|seed=11|budget=1|perturb", 1, "after_refinement", "h2", 10.4
            ),
        ]
    )
    points = pd.DataFrame(
        [
            _point("case|seed=11|budget=1|extra", 1, 7, 5.0, moved_nodes=3),
            _point("case|seed=11|budget=1|perturb", 1, 4, 4.0, moved_nodes=8),
        ]
    )
    run_rows = pd.DataFrame(
        [
            {
                "case": "case",
                "seed": 11,
                "candidate_budget": 1,
                "branch": "extra",
                "run_id": "case|seed=11|budget=1|extra",
                "quality": 10.5,
            },
            {
                "case": "case",
                "seed": 11,
                "candidate_budget": 1,
                "branch": "perturb",
                "run_id": "case|seed=11|budget=1|perturb",
                "quality": 10.4,
                "group_count": 1,
            },
        ]
    )
    local_merge = pd.DataFrame(
        [
            {
                "run_id": "case|seed=11|budget=1|perturb",
                "iteration": 1,
                "parent_id": 42,
            }
        ]
    )

    rows = module._build_first_divergence_rows(
        phase_frame=phase_frame,
        points=points,
        run_rows=run_rows,
        local_merge_frame=local_merge,
    )
    row = rows.iloc[0]

    assert row["candidate_budget"] == 1
    assert row["group_size_class"] == "single_node"
    assert row["first_divergence_phase"] == "after_refinement"
    assert row["extra_moved_nodes_iter"] == 3
    assert row["perturb_moved_nodes_iter"] == 8
    assert row["local_merge_parent_ids_sample"] == "42"


def test_compact_local_merge_parent_summary_writes_one_row_per_branch(tmp_path):
    module = _load_script(
        MONITOR_PATH, "leiden_work_acceleration_monitor_for_test_compact"
    )
    trace_path = tmp_path / "trajectory.jsonl"
    trace_events = [
        _local_merge(
            "case|seed=11|budget=1|extra", 1, 42, decision=10, low=2, changed=1
        ),
        _local_merge(
            "case|seed=11|budget=1|extra", 1, 7, decision=20, low=0, changed=3
        ),
        _local_merge(
            "case|seed=11|budget=1|perturb", 1, 99, decision=5, low=4, changed=0
        ),
        _local_merge(
            "case|seed=11|budget=1|perturb", 2, 100, decision=500, low=500, changed=500
        ),
    ]
    trace_path.write_text(
        "\n".join(json.dumps(event) for event in trace_events), encoding="utf-8"
    )
    first_divergence = pd.DataFrame(
        [
            {
                "case": "case",
                "seed": 11,
                "candidate_budget": 1,
                "run_id_extra": "case|seed=11|budget=1|extra",
                "run_id_perturb": "case|seed=11|budget=1|perturb",
                "first_divergence_iteration": 1,
            }
        ]
    )

    frame = module._extract_compact_local_merge_parent_summary(
        trace_path,
        first_divergence,
        tmp_path / "compact.csv",
    )

    assert list(frame.columns) == module.LOCAL_MERGE_PARENT_SUMMARY_COLUMNS
    assert len(frame) == 2
    extra = frame[frame["branch"] == "extra"].iloc[0]
    perturb = frame[frame["branch"] == "perturb"].iloc[0]
    assert extra["n_parent_rows"] == 2
    assert extra["total_decision_count"] == 30
    assert extra["total_changed_decision_count"] == 4
    assert extra["top_decision_parent_ids"] == "7,42"
    assert extra["top_changed_parent_ids"] == "7,42"
    assert perturb["n_parent_rows"] == 1
    assert perturb["top_low_margin_parent_ids"] == "99"


def test_extract_phase_checkpoints_handles_missing_trace_file(tmp_path):
    module = _load_script(
        MONITOR_PATH, "leiden_work_acceleration_monitor_for_test_missing_trace"
    )

    frame = module._extract_phase_checkpoints(
        tmp_path / "missing_trajectory.jsonl",
        tmp_path / "phase.csv",
    )

    assert frame.empty
    assert (tmp_path / "phase.csv").exists()


def test_multifidelity_policy_rows_keep_cost_comparison_visible():
    module = _load_script(
        MONITOR_PATH, "leiden_work_acceleration_monitor_for_test_multifidelity_cost"
    )
    rows = [
        {"policy": "full_top3_p5", "total_elapsed_ms": 30.0},
        {"policy": "p1_top1_then_p5", "total_elapsed_ms": 12.0},
    ]

    by_name = module._policy_rows_by_name(rows)

    assert (
        by_name["p1_top1_then_p5"]["total_elapsed_ms"]
        < by_name["full_top3_p5"]["total_elapsed_ms"]
    )


def test_portfolio_policy_row_keeps_wall_and_cpu_cost_separate():
    module = _load_script(
        MONITOR_PATH, "leiden_work_acceleration_monitor_for_test_portfolio_policy"
    )
    probe = type(
        "Probe",
        (),
        {
            "accepted": True,
            "elapsed_ms": 55.0,
            "candidate_eval_parallel": True,
            "candidate_eval_wall_elapsed_ms": 50.0,
            "candidate_eval_cpu_sum_elapsed_ms": 120.0,
            "candidate_eval_parallel_speedup": 2.4,
            "candidate_eval_parallel_workers": 3,
            "candidate_rows": [
                {
                    "candidate_index": 0,
                    "post_polish_delta_q": 1.0,
                    "post_polish_quality": 101.0,
                    "elapsed_ms": 40.0,
                },
                {
                    "candidate_index": 1,
                    "post_polish_delta_q": 3.0,
                    "post_polish_quality": 103.0,
                    "elapsed_ms": 80.0,
                },
            ],
        },
    )()

    best = module._best_candidate_row(probe.candidate_rows)
    row = module._portfolio_policy_row(
        policy="parallel_full_p5_portfolio",
        probe=probe,
        best=best,
    )

    assert row["selected_candidate_index"] == 1
    assert row["p5_elapsed_ms"] == 120.0
    assert row["total_elapsed_ms"] == 55.0
    assert row["candidate_eval_wall_elapsed_ms"] == 50.0
    assert row["candidate_eval_parallel_speedup"] == 2.4
    assert row["candidate_eval_parallel_workers"] == 3


def test_probe_run_row_records_selected_candidate_without_trace_polish():
    module = _load_script(
        MONITOR_PATH, "leiden_work_acceleration_monitor_for_test_probe_only_row"
    )
    probe = type(
        "Probe",
        (),
        {
            "accepted": True,
            "candidate_eval_parallel": True,
            "candidate_eval_wall_elapsed_ms": 50.0,
            "candidate_eval_cpu_sum_elapsed_ms": 120.0,
            "candidate_eval_parallel_speedup": 2.4,
            "candidate_eval_parallel_workers": 3,
        },
    )()
    best = {
        "candidate_index": 2,
        "source_cluster": 10,
        "target_cluster": 20,
        "group_kind": "best",
        "group_count": 4,
        "group_weight": 4.0,
        "post_polish_delta_q": 3.0,
        "accepted_by_quality": True,
    }

    row = module._probe_run_row(
        case="case",
        seed=11,
        candidate_budget=3,
        baseline_quality=100.0,
        baseline_elapsed=1.0,
        candidate_selection_elapsed=0.5,
        probe_elapsed=0.055,
        probe=probe,
        candidate_eval_mode="parallel_full_p5_portfolio",
        selected_policy="parallel_full_p5_portfolio",
        selected_policy_row={"quality": 103.0, "available": True},
        candidate_clusters=[10],
        best=best,
    )

    assert row["branch"] == "probe"
    assert row["candidate_index"] == 2
    assert row["quality"] == 103.0
    assert row["candidate_probe_eval_wall_elapsed_sec"] == 0.05
    assert row["candidate_probe_cpu_sum_elapsed_sec"] == 0.12
    assert row["candidate_eval_mode"] == "parallel_full_p5_portfolio"


def test_scorecard_exposes_parallel_probe_cost_columns():
    module = _load_script(
        MONITOR_PATH, "leiden_work_acceleration_monitor_for_test_parallel_cost"
    )
    points = pd.DataFrame(
        [
            _point(
                "case|seed=11|budget=3|extra",
                0,
                0,
                0.0,
                case="case",
                seed=11,
                budget=3,
                branch="extra",
            ),
            _point(
                "case|seed=11|budget=3|extra",
                1,
                10,
                30.0,
                case="case",
                seed=11,
                budget=3,
                branch="extra",
            ),
            _point(
                "case|seed=11|budget=3|perturb",
                0,
                0,
                0.0,
                case="case",
                seed=11,
                budget=3,
                branch="perturb",
            ),
            _point(
                "case|seed=11|budget=3|perturb",
                1,
                5,
                30.0,
                case="case",
                seed=11,
                budget=3,
                branch="perturb",
            ),
        ]
    )
    run_rows = pd.DataFrame(
        [
            {
                "case": "case",
                "seed": 11,
                "candidate_budget": 3,
                "branch": "extra",
                "elapsed_sec": 10.0,
                "baseline_quality": 100.0,
                "quality": 100.003,
            },
            {
                "case": "case",
                "seed": 11,
                "candidate_budget": 3,
                "branch": "perturb",
                "elapsed_sec": 99.0,
                "baseline_quality": 100.0,
                "quality": 100.003,
                "candidate_cluster_selection_elapsed_sec": 1.0,
                "candidate_probe_elapsed_sec": 4.0,
                "candidate_probe_eval_wall_elapsed_sec": 3.5,
                "candidate_probe_cpu_sum_elapsed_sec": 9.0,
                "candidate_probe_parallel": True,
                "candidate_probe_parallel_speedup": 2.57,
                "candidate_probe_parallel_workers": 3,
                "process_hwm_mb": 1024.0,
                "group_count": 1,
            },
        ]
    )

    scorecard = module._scorecard(points, run_rows)
    row = scorecard[scorecard["target_policy"] == "extra_p5_final"].iloc[0]

    assert row["operational_perturb_elapsed_sec"] == 5.0
    assert row["candidate_probe_eval_wall_elapsed_sec"] == 3.5
    assert row["candidate_probe_cpu_sum_elapsed_sec"] == 9.0
    assert bool(row["candidate_probe_parallel"]) is True
    assert row["candidate_probe_parallel_speedup"] == 2.57
    assert row["process_hwm_mb"] >= 0.0


def test_multifidelity_candidate_and_policy_outputs_have_required_columns(tmp_path):
    module = _load_script(
        MONITOR_PATH, "leiden_work_acceleration_monitor_for_test_multifidelity_outputs"
    )
    base = {
        "case": "case",
        "seed": 11,
        "candidate_budget": 3,
        "candidate_eval_mode": "multifidelity_label",
        "selected_policy": "p1_top1_then_p5",
    }
    candidate_path = tmp_path / "candidate_level_rows.csv"
    policy_path = tmp_path / "policy_comparison_rows.csv"

    module._append_rows(
        candidate_path,
        base,
        [
            {
                "candidate_index": 0,
                "source_cluster": 1,
                "target_cluster": 2,
                "pre_delta_q": 0.1,
                "p1_delta_q": 0.2,
                "p5_delta_q": 0.3,
                "selected_by_p1_top1": True,
                "selected_by_full_p5": True,
            }
        ],
    )
    module._append_rows(
        policy_path,
        base,
        [
            {
                "policy": "p1_top1_then_p5",
                "selected_candidate_index": 0,
                "p1_evaluated": 3,
                "p5_evaluated": 1,
                "total_elapsed_ms": 12.0,
                "final_delta_q": 0.3,
                "available": True,
            }
        ],
    )

    candidate_columns = set(pd.read_csv(candidate_path).columns)
    policy_columns = set(pd.read_csv(policy_path).columns)
    assert {
        "candidate_eval_mode",
        "candidate_index",
        "pre_delta_q",
        "p1_delta_q",
        "p5_delta_q",
        "selected_by_p1_top1",
        "selected_by_full_p5",
    } <= candidate_columns
    assert {
        "policy",
        "selected_candidate_index",
        "p1_evaluated",
        "p5_evaluated",
        "total_elapsed_ms",
        "final_delta_q",
    } <= policy_columns


def test_multifidelity_policy_rows_include_p1_top3_diagnostic_row():
    module = _load_script(
        MONITOR_PATH, "leiden_work_acceleration_monitor_for_test_multifidelity_top3"
    )
    candidate_rows = [
        _multifidelity_candidate(0, p1=3.0, p5=1.0, p1_elapsed=0.0, p5_elapsed=10.0),
        _multifidelity_candidate(1, p1=2.0, p5=2.0, p1_elapsed=0.0, p5_elapsed=20.0),
        _multifidelity_candidate(2, p1=1.0, p5=3.0, p1_elapsed=0.0, p5_elapsed=30.0),
    ]
    rows = module._ensure_multifidelity_policy_rows(
        [
            {
                "policy": "full_top3_p5",
                "selected_candidate_index": 2,
                "available": True,
                "accepted": True,
                "matches_full_p5": True,
                "total_elapsed_ms": 60.0,
            }
        ],
        candidate_rows,
    )

    policies = {row["policy"]: row for row in rows}

    assert "p1_top3_then_p5" in policies
    assert (
        policies["p1_top1_then_p5"]["total_elapsed_ms"]
        <= policies["full_top3_p5"]["total_elapsed_ms"]
    )
    assert (
        policies["p1_top2_then_p5"]["total_elapsed_ms"]
        <= policies["full_top3_p5"]["total_elapsed_ms"]
    )
    assert (
        policies["p1_top3_then_p5"]["total_elapsed_ms"]
        <= policies["full_top3_p5"]["total_elapsed_ms"]
    )
    assert policies["p1_top3_then_p5"]["selected_candidate_index"] == 2
    assert policies["p1_top3_then_p5"]["matches_full_p5"] is True


def test_approx_polish_policy_rows_include_shadow_diagnostics():
    module = _load_script(
        MONITOR_PATH, "leiden_work_acceleration_monitor_for_test_approx_policy_rows"
    )
    candidate_rows = [
        {
            **_multifidelity_candidate(0, p1=1.0, p5=1.0, p5_elapsed=10.0),
            "localized_delta_q": 3.0,
            "localized_elapsed_ms": 1.0,
            "quotient_delta_q": 1.0,
            "quotient_elapsed_ms": 0.5,
            "ub_delta_q": 5.0,
            "ub_elapsed_ms": 0.25,
            "ub_covers_p5": True,
            "ub_violation": 0.0,
        },
        {
            **_multifidelity_candidate(1, p1=2.0, p5=3.0, p5_elapsed=20.0),
            "localized_delta_q": 2.0,
            "localized_elapsed_ms": 1.0,
            "quotient_delta_q": 3.0,
            "quotient_elapsed_ms": 0.5,
            "ub_delta_q": 4.0,
            "ub_elapsed_ms": 0.25,
            "ub_covers_p5": True,
            "ub_violation": 0.0,
        },
        {
            **_multifidelity_candidate(2, p1=0.5, p5=2.0, p5_elapsed=30.0),
            "localized_delta_q": 1.0,
            "localized_elapsed_ms": 1.0,
            "quotient_delta_q": 2.0,
            "quotient_elapsed_ms": 0.5,
            "ub_delta_q": 0.0,
            "ub_elapsed_ms": 0.25,
            "ub_covers_p5": False,
            "ub_violation": 2.0,
        },
    ]

    rows = module._ensure_approx_polish_policy_rows([], candidate_rows)
    policies = {row["policy"]: row for row in rows}

    assert policies["localized_top2_then_p5"]["selected_candidate_index"] == 1
    assert policies["localized_top2_then_p5"]["matches_full_p5"] is True
    assert policies["localized_top2_then_p5"]["p5_evaluated"] == 2
    assert policies["quotient_top1_then_p5"]["selected_candidate_index"] == 1
    assert policies["quotient_top1_then_p5"]["matches_full_p5"] is True
    assert policies["ub_shadow_skip_margin_0"]["matches_full_p5"] is True
    assert policies["ub_shadow_skip_margin_0"]["ub_skipped"] >= 1


def test_multifidelity_operational_missing_p5_labels_keep_policy_rows_available_false():
    module = _load_script(
        MONITOR_PATH,
        "leiden_work_acceleration_monitor_for_test_multifidelity_missing_labels",
    )
    candidate_rows = [
        _multifidelity_candidate(0, p1=3.0, p5=1.0),
        _multifidelity_candidate(1, p1=2.0, p5=2.0),
        _multifidelity_candidate(2, p1=1.0, p5=float("nan")),
    ]

    rows = module._ensure_multifidelity_policy_rows([], candidate_rows)
    policies = {row["policy"]: row for row in rows}

    assert policies["p1_top3_then_p5"]["available"] is False
    assert policies["p1_top3_then_p5"]["p5_evaluated"] == 2


def _point(
    run_id: str,
    iteration: int,
    k_work: int,
    ppm: float,
    *,
    case: str | None = None,
    seed: int | None = None,
    budget: int | None = None,
    branch: str | None = None,
    moved_nodes: int = 0,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "case": case or "case",
        "seed": seed if seed is not None else 11,
        "candidate_budget": budget if budget is not None else 1,
        "branch": branch or run_id.split("|")[-1],
        "t_i": iteration,
        "t_k_phase": iteration,
        "t_k_work": k_work,
        "t_label": f"({iteration},{iteration},{k_work})",
        "qf_delta_ppm": ppm,
        "moved_nodes": moved_nodes,
    }


def _checkpoint(
    run_id: str, iteration: int, phase: str, membership_hash: str, quality: float
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "iteration": iteration,
        "depth": 0,
        "phase": phase,
        "membership_hash": membership_hash,
        "n_clusters": 3,
        "quality": quality,
    }


def _local_merge(
    run_id: str,
    iteration: int,
    parent_id: int,
    *,
    decision: int,
    low: int,
    changed: int,
) -> dict[str, object]:
    return {
        "event": "local_merge_margin_summary",
        "run_id": run_id,
        "iteration": iteration,
        "depth": 0,
        "parent_id": parent_id,
        "parent_visit_index": 0,
        "source": "standard_refinement",
        "parent_size": decision,
        "parent_weight": float(decision),
        "decision_count": decision,
        "low_margin_decision_count": low,
        "changed_decision_count": changed,
        "min_margin": 0.01,
        "p10_margin": 0.02,
        "p50_margin": 0.03,
        "selected_child_count": 2,
        "largest_child_fraction": 0.5,
    }


def _multifidelity_candidate(
    candidate_index: int,
    *,
    p1: float,
    p5: float,
    p1_elapsed: float = 1.0,
    p5_elapsed: float = 5.0,
) -> dict[str, object]:
    return {
        "candidate_index": candidate_index,
        "p1_delta_q": p1,
        "p5_delta_q": p5,
        "p1_quality": 100.0 + p1,
        "p5_quality": 100.0 + p5,
        "p1_elapsed_ms": p1_elapsed,
        "p5_elapsed_ms": p5_elapsed,
    }
