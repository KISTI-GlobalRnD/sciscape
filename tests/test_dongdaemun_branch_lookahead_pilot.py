"""Smoke tests for the Dongdaemun branch-lookahead pilot runner."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from sciscape.clustering.leiden_rust import RUST_AVAILABLE


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "research"
    / "consensus"
    / "scripts"
    / "run_dongdaemun_branch_lookahead_pilot.py"
)


def _load_module():
    script_dir = SCRIPT_PATH.parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    spec = importlib.util.spec_from_file_location(
        "run_dongdaemun_branch_lookahead_pilot_for_test",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_tiny_summary(tmp_path: Path) -> Path:
    graph_dir = tmp_path / "graph"
    graph_dir.mkdir()
    np.asarray([0, 2, 0, 0, 1, 1], dtype=np.uint32).tofile(graph_dir / "src.u32.bin")
    np.asarray([1, 3, 2, 3, 2, 3], dtype=np.uint32).tofile(graph_dir / "dst.u32.bin")
    np.asarray([10.0, 10.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64).tofile(
        graph_dir / "weight.f64.bin"
    )
    np.ones(4, dtype=np.float64).tofile(graph_dir / "node_weights.f64.bin")
    summary_path = tmp_path / "prepare_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "edge_layer": "tiny",
                "n_nodes": 4,
                "paths": {
                    "graph_dir": str(graph_dir),
                    "membership": str(graph_dir / "unused_membership.parquet"),
                    "node_weights": str(graph_dir / "node_weights.f64.bin"),
                },
                "resolution": 0.000001,
                "sample": "tiny_sample",
                "source_sample": "tiny_source",
                "seed": 42,
                "target_max_doc_weight": 2.5,
            }
        ),
        encoding="utf-8",
    )
    return summary_path


def test_promoted_keys_for_policy_uses_iter5_quality_order():
    module = _load_module()
    rows = [
        {"seed": 1, "randomness": 0.0, "quality": 10.0, "max_doc_weight_ratio": 1.0, "n_above_max_doc_weight": 0, "elapsed_sec": 1.0, "supported": True},
        {"seed": 2, "randomness": 0.0, "quality": 30.0, "max_doc_weight_ratio": 1.0, "n_above_max_doc_weight": 0, "elapsed_sec": 1.0, "supported": True},
        {"seed": 3, "randomness": 0.0, "quality": 20.0, "max_doc_weight_ratio": 1.0, "n_above_max_doc_weight": 0, "elapsed_sec": 1.0, "supported": True},
        {"seed": 4, "randomness": 0.0, "quality": 5.0, "max_doc_weight_ratio": 1.0, "n_above_max_doc_weight": 0, "elapsed_sec": 1.0, "supported": True},
    ]

    assert module.promoted_keys_for_policy("iter5_screen_all_top3", rows) == [
        (2, 0.0),
        (3, 0.0),
        (1, 0.0),
    ]
    assert len(module.promoted_keys_for_policy("iter5_screen_all_top5", rows)) == 4


def test_margin_policy_polishes_top2_only_when_iter10_margin_is_small():
    module = _load_module()
    close_rows = [
        {"seed": 1, "randomness": 0.0, "quality": 200.0, "max_doc_weight_ratio": 1.0, "n_above_max_doc_weight": 0, "elapsed_sec": 1.0, "supported": True},
        {"seed": 2, "randomness": 0.0, "quality": 190.0, "max_doc_weight_ratio": 1.0, "n_above_max_doc_weight": 0, "elapsed_sec": 1.0, "supported": True},
    ]
    wide_rows = [
        {"seed": 1, "randomness": 0.0, "quality": 200.0, "max_doc_weight_ratio": 1.0, "n_above_max_doc_weight": 0, "elapsed_sec": 1.0, "supported": True},
        {"seed": 2, "randomness": 0.0, "quality": 100.0, "max_doc_weight_ratio": 1.0, "n_above_max_doc_weight": 0, "elapsed_sec": 1.0, "supported": True},
    ]

    assert module.convergence_polish_keys_for_policy(
        "margin_polish_top2",
        close_rows,
    ) == [(1, 0.0), (2, 0.0)]
    assert module.convergence_polish_keys_for_policy(
        "margin_polish_top2",
        wide_rows,
    ) == [(1, 0.0)]


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust backend required")
def test_branch_lookahead_pilot_tiny_graph_writes_staged_outputs(tmp_path):
    module = _load_module()
    summary_path = _write_tiny_summary(tmp_path)

    payload = module.run_branch_lookahead_pilot(
        summaries=(summary_path,),
        output_dir=tmp_path / "out",
        seeds=(11, 42),
        randomness_values=(0.0,),
        policy_name="iter5_screen_all_top3",
    )

    assert payload["n_rows"] == 5
    assert payload["n_stage_summaries"] == 1
    for path in payload["paths"].values():
        assert Path(path).exists()

    with Path(payload["paths"]["rows_csv"]).open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert {row["requested_n_iterations"] for row in rows} == {
        "5",
        "10",
        "convergence",
    }
    assert sum(row["requested_n_iterations"] == "5" for row in rows) == 2
    assert sum(row["requested_n_iterations"] == "10" for row in rows) == 2
    assert sum(row["requested_n_iterations"] == "convergence" for row in rows) == 1

    with Path(payload["paths"]["stage_summary"]).open(encoding="utf-8") as fh:
        stage_rows = list(csv.DictReader(fh))
    assert stage_rows[0]["policy_name"] == "iter5_screen_all_top3"
    assert stage_rows[0]["n_stage1_candidates"] == "2"
    assert stage_rows[0]["n_promoted_iter10"] == "2"
    assert stage_rows[0]["n_polished_convergence"] == "1"
    assert stage_rows[0]["selected_candidate_id"]
