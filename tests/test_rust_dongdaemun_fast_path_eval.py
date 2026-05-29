"""Tests for the Rust Dongdaemun fast-path validation script helpers."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "research"
    / "consensus"
    / "scripts"
    / "dongdaemun_hierarchy"
    / "refinement_runs"
    / "evaluate_rust_dongdaemun_fast_path.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "evaluate_rust_dongdaemun_fast_path_for_test",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_flatten_result_records_rust_audit_and_membership_diff():
    module = _load_module()
    result = module.LevelPostprocessResult(
        membership=np.asarray([0, 1, 1], dtype=np.uint64),
        accepted=True,
        status="committed",
        small_cluster_summary={},
        oversize_summary={
            "before": {"n_above_max_doc_weight": 1, "max_doc_weight": 3.0},
            "after": {"n_above_max_doc_weight": 0, "max_doc_weight": 2.0},
            "changed_nodes": 1,
            "split_repair_exact_delta_q": 1.25,
            "trim_exact_delta_q": 0.5,
            "final_exact_delta_q": 1.75,
            "target_max_satisfied": True,
            "iterations": [{"iteration": 1}],
            "rust_audit": {
                "status": "committed",
                "trim_moves_committed": 2,
            },
        },
        final_summary={"n_clusters": 2},
        backend="rust_dongdaemun",
    )

    row = module._flatten_result(
        sample="toy",
        backend="rust_dongdaemun",
        elapsed_sec=0.25,
        result=result,
        python_membership=np.asarray([0, 0, 1], dtype=np.uint64),
    )

    assert row["backend"] == "rust_dongdaemun"
    assert row["quality_delta"] == 1.75
    assert row["n_oversize_before"] == 1
    assert row["n_oversize_after"] == 0
    assert row["membership_equal_to_python"] is False
    assert row["membership_diff_nodes"] == 1
    assert row["rust_audit_status"] == "committed"
    assert row["rust_split_iterations"] == 1
    assert row["rust_trim_moves_committed"] == 2


def test_flatten_result_leaves_rust_only_fields_blank_for_python_backend():
    module = _load_module()
    result = module.LevelPostprocessResult(
        membership=np.asarray([0, 1], dtype=np.uint64),
        accepted=True,
        status="committed",
        small_cluster_summary={},
        oversize_summary={
            "before": {"n_above_max_doc_weight": 1, "max_doc_weight": 3.0},
            "after": {"n_above_max_doc_weight": 0, "max_doc_weight": 2.0},
            "iterations": [{"iteration": 1}],
            "rust_audit": {
                "status": "committed",
                "trim_moves_committed": 2,
            },
        },
        final_summary={"n_clusters": 2},
        backend="python",
    )

    row = module._flatten_result(
        sample="toy",
        backend="python",
        elapsed_sec=0.25,
        result=result,
    )

    assert row["rust_audit_status"] == ""
    assert row["rust_split_iterations"] == ""
    assert row["rust_trim_moves_committed"] == ""


def test_comparison_marks_unsupported_rust_row_incomplete():
    module = _load_module()
    python_row = {
        "backend": "python",
        "supported": True,
        "status": "committed",
        "accepted": True,
        "quality_delta": 1.0,
        "n_oversize_after": 1,
        "max_doc_weight_after": 10.0,
        "elapsed_sec": 2.0,
    }
    rust_row = module._unsupported_row(
        sample="toy",
        backend="rust_dongdaemun",
        reason="missing binding",
    )

    comparison = module._comparison_summary([python_row, rust_row])

    assert comparison["status"] == "incomplete"
    assert comparison["reason"] == "missing binding"


def test_prepare_summary_resolution_preserves_seed_and_defaults_node_weights(tmp_path):
    module = _load_module()
    graph_dir = tmp_path / "graph"
    membership_path = tmp_path / "membership.parquet"
    summary_path = tmp_path / "prepare_summary.json"
    summary = {
        "sample": "toy_sample",
        "seed": 11,
        "resolution": 0.01,
        "target_min_doc_weight": 2.0,
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
    assert cfg.seed == 11
    assert cfg.graph_dir == graph_dir
    assert cfg.membership_path == membership_path
    assert cfg.node_weights_path == graph_dir / "node_weights.f64.bin"
    assert cfg.n_nodes == 3
