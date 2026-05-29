"""Smoke tests for the standard Leiden random refinement profiler."""

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
    / "leiden_basin"
    / "basin_signatures"
    / "branch_growth"
    / "run_leiden_random_refinement_profile.py"
)


def _load_module():
    script_dir = SCRIPT_PATH.parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    spec = importlib.util.spec_from_file_location(
        "run_leiden_random_refinement_profile_for_test",
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


def test_n_iterations_values_parse_convergence_backend_value():
    module = _load_module()

    budgets = module._parse_n_iterations_values("1,convergence")

    assert [budget.requested for budget in budgets] == ["1", "convergence"]
    assert [budget.n_iterations for budget in budgets] == [1, 0]
    assert [budget.mode for budget in budgets] == ["fixed", "convergence"]


def test_row_key_includes_requested_n_iterations(tmp_path):
    module = _load_module()
    summary_path = tmp_path / "summary.json"

    fixed_key = module._row_key(
        summary_path=summary_path,
        seed=11,
        randomness=0.0,
        requested_n_iterations="1",
    )
    convergence_key = module._row_key(
        summary_path=summary_path,
        seed=11,
        randomness=0.0,
        requested_n_iterations="convergence",
    )

    assert fixed_key != convergence_key
    assert module._run_id(fixed_key) != module._run_id(convergence_key)


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust backend required")
def test_run_profile_tiny_graph_writes_expected_outputs(tmp_path):
    module = _load_module()
    summary_path = _write_tiny_summary(tmp_path)

    payload = module.run_profile(
        summaries=(summary_path,),
        output_dir=tmp_path / "out",
        seeds=(11,),
        randomness_values=(0.0,),
        n_iterations_values=module._parse_n_iterations_values("1,10,convergence"),
    )

    assert payload["n_rows"] == 3
    assert payload["n_expected_runs"] == 3
    assert payload["n_runs_with_start_and_final_trace"] == 3
    assert payload["n_quality_or_trace_mismatches"] == 0
    for path in payload["paths"].values():
        assert Path(path).exists()
    with Path(payload["paths"]["rows_csv"]).open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert {row["requested_n_iterations"] for row in rows} == {
        "1",
        "10",
        "convergence",
    }
    assert {row["n_iterations"] for row in rows} == {"0", "1", "10"}
    assert all(row["n_iterations_used"] for row in rows)
    assert all(row["trace_has_start"] == "True" for row in rows)
    assert all(row["trace_has_final"] == "True" for row in rows)
    assert all(row["quality_recompute_ok"] == "True" for row in rows)
    assert all(row["late_quality_gain_vs_iter1"] != "" for row in rows)
    assert all(row["quality_recovery_ratio_vs_best_10"] != "" for row in rows)
    assert all(row["quality_recovery_ratio_vs_best_convergence"] != "" for row in rows)
    assert rows[0]["sample"] == "tiny_sample"
    assert rows[0]["variant"] == "standard_leiden"

    with Path(payload["paths"]["rows_jsonl"]).open(encoding="utf-8") as fh:
        jsonl_rows = [json.loads(line) for line in fh if line.strip()]
    assert len(jsonl_rows) == 3
    assert all("requested_n_iterations" in row for row in jsonl_rows)
    assert all("n_iterations_used" in row for row in jsonl_rows)
    assert all("late_quality_gain_vs_iter1" in row for row in jsonl_rows)

    with Path(payload["paths"]["quality_trace_runs"]).open(encoding="utf-8") as fh:
        trace_runs = [json.loads(line) for line in fh if line.strip()]
    assert {run["requested_n_iterations"] for run in trace_runs} == {
        "1",
        "10",
        "convergence",
    }
    assert {run["n_iterations"] for run in trace_runs} == {0, 1, 10}
    assert all(run["n_iterations_used"] is not None for run in trace_runs)

    with Path(payload["paths"]["iteration_budget_by_run"]).open(encoding="utf-8") as fh:
        ladder_rows = list(csv.DictReader(fh))
    assert ladder_rows[0]["iter1_quality"] != ""
    assert ladder_rows[0]["iter10_quality"] != ""
    assert ladder_rows[0]["convergence_quality"] != ""

    with Path(payload["paths"]["iteration_budget_by_group"]).open(encoding="utf-8") as fh:
        group_rows = list(csv.DictReader(fh))
    assert {row["requested_n_iterations"] for row in group_rows} == {
        "1",
        "10",
        "convergence",
    }

    with Path(payload["paths"]["shortlist_policy_simulation"]).open(encoding="utf-8") as fh:
        policy_rows = list(csv.DictReader(fh))
    assert policy_rows
    assert policy_rows[0]["estimated_elapsed_sec"] != ""
