"""Tests for the Slice 4 Dongdaemun refinement pilot runner."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "research"
    / "consensus"
    / "scripts"
    / "evaluate_dongdaemun_refinement_slice4.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "evaluate_dongdaemun_refinement_slice4_for_test",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_graph_sidecars(
    graph_dir: Path,
    *,
    src: np.ndarray,
    dst: np.ndarray,
    weight: np.ndarray,
    node_weights: np.ndarray,
) -> None:
    graph_dir.mkdir(parents=True, exist_ok=True)
    np.asarray(src, dtype=np.uint32).tofile(graph_dir / "src.u32.bin")
    np.asarray(dst, dtype=np.uint32).tofile(graph_dir / "dst.u32.bin")
    np.asarray(weight, dtype=np.float64).tofile(graph_dir / "weight.f64.bin")
    np.asarray(node_weights, dtype=np.float64).tofile(
        graph_dir / "node_weights.f64.bin"
    )


def _write_membership(path: Path, clusters: np.ndarray) -> None:
    table = pa.table(
        {
            "node_idx": np.arange(int(clusters.shape[0]), dtype=np.uint32),
            "cluster": np.asarray(clusters, dtype=np.uint64),
        }
    )
    pq.write_table(table, path)


def test_summary_path_resolver_reads_graph_membership_seed_and_target(tmp_path):
    module = _load_module()
    graph_dir = tmp_path / "graph"
    membership_path = tmp_path / "membership.parquet"
    summary_path = tmp_path / "prepare_summary.json"
    summary = {
        "sample": "toy_sample",
        "seed": 11,
        "resolution": 0.01,
        "target_max_doc_weight": 5.0,
        "n_nodes": 3,
        "paths": {
            "graph_dir": str(graph_dir),
            "membership": str(membership_path),
        },
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    cfg = module._resolve_input_from_summary(summary_path)

    assert cfg.sample == "toy_sample"
    assert cfg.graph_dir == graph_dir
    assert cfg.membership_path == membership_path
    assert cfg.node_weights_path == graph_dir / "node_weights.f64.bin"
    assert cfg.seed == 11
    assert cfg.resolution == 0.01
    assert cfg.target_max_doc_weight == 5.0
    assert cfg.n_nodes == 3


def test_synthetic_tiny_graph_writes_three_variant_rows(tmp_path):
    module = _load_module()
    pytest.importorskip("sciscape_leiden")
    if not module.RUST_DONGDAEMUN_REFINEMENT_AVAILABLE:
        pytest.skip("Rust Dongdaemun refinement binding required")

    graph_dir = tmp_path / "graph"
    membership_path = tmp_path / "membership.parquet"
    _write_graph_sidecars(
        graph_dir,
        src=np.asarray([0, 2], dtype=np.uint32),
        dst=np.asarray([1, 3], dtype=np.uint32),
        weight=np.asarray([10.0, 10.0], dtype=np.float64),
        node_weights=np.ones(4, dtype=np.float64),
    )
    _write_membership(membership_path, np.zeros(4, dtype=np.uint64))
    cfg = module.Slice4Input(
        sample="toy",
        graph_dir=graph_dir,
        membership_path=membership_path,
        node_weights_path=graph_dir / "node_weights.f64.bin",
        resolution=1.0,
        target_max_doc_weight=1.5,
        seed=11,
    )

    payload = module.run_pilot(
        cfg,
        output_dir=tmp_path / "out",
        run_config=module.Slice4RunConfig(
            n_iterations=1,
            randomness=0.0,
            max_extra_parents_per_iteration=2,
            max_extra_children_per_parent=8,
            max_singleton_weight_fraction=0.0,
            min_largest_child_fraction_improvement=0.0,
            gamma_multipliers=(100.0,),
        ),
    )

    rows = payload["rows"]
    assert [row["variant"] for row in rows] == [
        module.VARIANT_STANDARD,
        module.VARIANT_REPAIR_OFF,
        module.VARIANT_REPAIR_ON,
    ]
    assert all(set(module.CSV_FIELDS) == set(row) for row in rows)
    assert all(row["supported"] for row in rows)
    assert (tmp_path / "out" / "slice4_refinement_pilot.csv").exists()
    assert (tmp_path / "out" / "slice4_refinement_pilot.parquet").exists()
    assert (tmp_path / "out" / "slice4_refinement_pilot_summary.json").exists()
    assert (tmp_path / "out" / "slice4_refinement_pilot_report.md").exists()

    repair_off = rows[1]
    for field in module.BASELINE_REPAIR_AUDIT_FIELDS:
        assert repair_off[field] == 0


def test_repair_on_row_missing_audit_fields_falls_back_to_zero():
    module = _load_module()

    class SparseAudit:
        enabled = True
        selected_parent_count_total = np.int64(2)
        applied_parent_count_total = np.int64(1)
        iteration_depth = np.asarray([0], dtype=np.uint64)

    class SparseResult:
        membership = np.asarray([0, 0], dtype=np.uint64)
        quality = np.float64(1.25)
        n_clusters = np.int64(1)
        n_iterations_used = np.int64(1)
        audit = SparseAudit()

    row = module._flatten_result(
        sample="toy",
        variant=module.VARIANT_REPAIR_ON,
        elapsed_sec=0.1,
        result=SparseResult(),
        node_weights=np.ones(2, dtype=np.float64),
        target_max_doc_weight=10.0,
        standard_membership=np.asarray([0, 0], dtype=np.uint64),
        standard_quality=1.0,
    )

    assert row["selected_parent_count_total"] == 2
    assert row["applied_parent_count_total"] == 1
    assert row["quotient_candidates_total"] == 0
    assert row["baseline_repair_candidates_total"] == 0
    assert row["baseline_repair_improved_candidates_total"] == 0
    assert row["baseline_repair_selected_total"] == 0
    assert row["baseline_repair_merge_count_total"] == 0
    assert row["baseline_repair_delta_sum"] == 0.0
    assert row["candidate_quality_delta_sum"] == 0.0
    assert row["candidate_positive_quality_delta_total"] == 0
    assert row["candidate_selected_positive_quality_delta_total"] == 0
    assert row["candidate_rejected_by_quality_total"] == 0
    assert row["candidate_qpos_spos_total"] == 0
    assert row["candidate_qpos_sneg_total"] == 0
    assert row["candidate_qneg_spos_total"] == 0
    assert row["candidate_qneg_sneg_total"] == 0
    assert row["candidate_true_positive_total"] == 0
    assert row["candidate_false_positive_total"] == 0
    assert row["candidate_false_negative_total"] == 0
    assert row["candidate_true_negative_total"] == 0
    assert row["final_quality_guard_enabled"] == 0
    assert row["final_quality_guard_triggered"] == 0
    assert row["final_quality_guard_standard_quality"] == 0.0
    assert row["final_quality_guard_pre_guard_quality"] == 0.0
    assert row["final_quality_delta_vs_guard_standard"] == 0.0


def test_output_writers_json_safe_numpy_scalars_and_arrays(tmp_path):
    module = _load_module()
    graph_dir = tmp_path / "graph"
    input_cfg = module.Slice4Input(
        sample="toy",
        graph_dir=graph_dir,
        membership_path=tmp_path / "membership.parquet",
        node_weights_path=graph_dir / "node_weights.f64.bin",
        resolution=np.float64(0.01),
        target_max_doc_weight=np.float64(5.0),
        seed=np.int64(11),
    )
    row = module._unsupported_row(
        sample="toy",
        variant=module.VARIANT_REPAIR_ON,
        reason="missing binding",
    )
    row.update(
        {
            "supported": np.bool_(True),
            "unsupported_reason": "",
            "elapsed_sec": np.float64(0.25),
            "n_clusters": np.int64(2),
            "quality": np.float64(3.5),
            "quality_delta_vs_standard": np.float64(0.0),
            "max_doc_weight": np.float64(4.0),
            "max_doc_weight_ratio": np.float64(0.8),
            "n_above_max_doc_weight": np.int64(0),
            "top10_doc_weights": np.asarray([4.0, 2.0], dtype=np.float64),
            "membership_equal_to_standard": np.bool_(True),
            "membership_diff_nodes_vs_standard": np.int64(0),
            "membership_equal_repair_off_on": np.bool_(True),
        }
    )

    paths = module._write_outputs(
        output_dir=tmp_path / "out",
        input_cfg=input_cfg,
        run_config=module.Slice4RunConfig(),
        rows=[row],
        aggregate={"np_scalar": np.float64(1.5), "np_array": np.asarray([1, 2])},
    )

    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert summary["aggregate"]["np_scalar"] == 1.5
    assert summary["aggregate"]["np_array"] == [1, 2]
    assert summary["rows"][0]["top10_doc_weights"] == [4.0, 2.0]
    assert paths["csv"].exists()
    assert paths["parquet"].exists()
    assert paths["report"].exists()


def test_run_variant_forwards_adaptive_near_tie_options(tmp_path):
    module = _load_module()

    class FakeResult:
        membership = np.asarray([0, 1], dtype=np.uint64)
        quality = 2.0
        n_clusters = 2
        n_iterations_used = 1
        audit = None

    class FakeGraph:
        def __init__(self):
            self.kwargs = None

        def run_leiden_dongdaemun_refinement(self, **kwargs):
            self.kwargs = kwargs
            return FakeResult()

    graph = FakeGraph()
    input_cfg = module.Slice4Input(
        sample="toy",
        graph_dir=tmp_path / "graph",
        membership_path=None,
        node_weights_path=tmp_path / "node_weights.f64.bin",
        resolution=0.01,
        target_max_doc_weight=5.0,
        seed=11,
    )

    module._run_variant(
        graph=graph,
        input_cfg=input_cfg,
        run_config=module.Slice4RunConfig(
            adaptive_near_tie_probe_mode="qf_replace",
            adaptive_near_tie_margin_parent_weight=1e-4,
            adaptive_near_tie_randomness=0.05,
            adaptive_near_tie_max_decisions_per_parent=8,
        ),
        node_weights=np.ones(2, dtype=np.float64),
        variant=module.VARIANT_REPAIR_OFF,
        standard_membership=None,
        standard_quality=None,
    )

    assert graph.kwargs["adaptive_near_tie_probe_mode"] == "qf_replace"
    assert graph.kwargs["adaptive_near_tie_margin_parent_weight"] == pytest.approx(
        1e-4
    )
    assert graph.kwargs["adaptive_near_tie_randomness"] == pytest.approx(0.05)
    assert graph.kwargs["adaptive_near_tie_max_decisions_per_parent"] == 8


def test_run_variant_forwards_adaptive_local_shake_options(tmp_path):
    module = _load_module()

    class FakeResult:
        membership = np.asarray([0, 1], dtype=np.uint64)
        quality = 2.0
        n_clusters = 2
        n_iterations_used = 1
        audit = None

    class FakeGraph:
        def __init__(self):
            self.kwargs = None

        def run_leiden_dongdaemun_refinement(self, **kwargs):
            self.kwargs = kwargs
            return FakeResult()

    graph = FakeGraph()
    input_cfg = module.Slice4Input(
        sample="toy",
        graph_dir=tmp_path / "graph",
        membership_path=None,
        node_weights_path=tmp_path / "node_weights.f64.bin",
        resolution=0.01,
        target_max_doc_weight=5.0,
        seed=11,
    )

    module._run_variant(
        graph=graph,
        input_cfg=input_cfg,
        run_config=module.Slice4RunConfig(
            adaptive_local_shake_mode="pressure_guarded",
            adaptive_local_shake_arms=("resolution_up", "seed_local_refinement"),
            adaptive_local_shake_resolution_up_multipliers=(1.02,),
            adaptive_local_shake_seed_perturbations=1,
            adaptive_local_shake_final_guard_mode="runner_audit",
        ),
        node_weights=np.ones(2, dtype=np.float64),
        variant=module.VARIANT_REPAIR_OFF,
        standard_membership=None,
        standard_quality=None,
    )

    assert graph.kwargs["adaptive_local_shake_mode"] == "pressure_guarded"
    assert graph.kwargs["adaptive_local_shake_arms"] == (
        "resolution_up",
        "seed_local_refinement",
    )
    assert graph.kwargs["adaptive_local_shake_resolution_up_multipliers"] == (
        pytest.approx(1.02),
    )
    assert graph.kwargs["adaptive_local_shake_seed_perturbations"] == 1
    assert graph.kwargs["adaptive_local_shake_final_guard_mode"] == "runner_audit"
