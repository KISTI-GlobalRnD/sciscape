"""Tests for online-first adaptive probe neighborhood analysis."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "research"
    / "consensus"
    / "scripts"
    / "analyze_dongdaemun_online_first_neighborhood.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "analyze_dongdaemun_online_first_neighborhood_for_test",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_run_matrix_computes_policy_and_oracle_deltas():
    module = _load_module()
    baseline = {
        "r1": {
            "run_id": "r1",
            "sample": "tiny",
            "variant": "off",
            "quality": "10.0",
            "n_clusters": "3",
            "n_above_max_doc_weight": "1",
        },
        "r2": {
            "run_id": "r2",
            "sample": "tiny",
            "variant": "off",
            "quality": "20.0",
            "n_clusters": "4",
            "n_above_max_doc_weight": "2",
        },
    }
    policy_rows = {
        "safe": {
            "r1": {"quality": "11.0", "n_clusters": "3", "n_above_max_doc_weight": "1"},
            "r2": {"quality": "20.0", "n_clusters": "4", "n_above_max_doc_weight": "2"},
        },
        "risky": {
            "r1": {"quality": "9.0", "n_clusters": "3", "n_above_max_doc_weight": "1"},
            "r2": {"quality": "25.0", "n_clusters": "4", "n_above_max_doc_weight": "2"},
        },
    }

    matrix = module.build_run_matrix(
        baseline_rows=baseline,
        policy_rows=policy_rows,
    )
    summary = module.summarize_policies(matrix, ("safe", "risky"))

    assert [row["oracle_best_policy"] for row in matrix] == ["safe", "risky"]
    risky = next(row for row in summary if row["policy"] == "risky")
    assert risky["quality_delta_sum"] == 4.0
    assert risky["quality_wins"] == 1
    assert risky["quality_losses"] == 1
    oracle = next(row for row in summary if row["policy"] == "oracle_best_known_neighborhood")
    assert oracle["quality_delta_sum"] == 6.0


def test_summarize_local_ambiguity_flags_same_commit_with_win_and_loss():
    module = _load_module()
    matrix = [
        {"run_id": "r1", "node_quality_delta": -1.0},
        {"run_id": "r2", "node_quality_delta": 3.0},
    ]
    commits = [
        {
            "policy": "node",
            "run_id": "r1",
            "parent_id": 7,
            "source": "node_order_control",
            "source_index": 0,
            "gain_parent_weight": 1.01,
            "candidate_n_clusters": 5,
            "largest_child_fraction": 0.5,
        },
        {
            "policy": "node",
            "run_id": "r2",
            "parent_id": 7,
            "source": "node_order_control",
            "source_index": 0,
            "gain_parent_weight": 1.01,
            "candidate_n_clusters": 5,
            "largest_child_fraction": 0.5,
        },
    ]

    rows = module.summarize_local_ambiguity(commits=commits, matrix=matrix)

    assert len(rows) == 1
    assert rows[0]["has_win_and_loss"] is True
    assert rows[0]["quality_delta_min"] == -1.0
    assert rows[0]["quality_delta_max"] == 3.0
