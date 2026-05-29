from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "research"
    / "consensus"
    / "scripts"
    / "leiden_basin"
    / "transition_routes"
    / "closure_context"
    / "analyze_leiden_basin_transition_closure_context.py"
)


def _load_module():
    script_dir = SCRIPT_PATH.parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    spec = importlib.util.spec_from_file_location(
        "analyze_leiden_basin_transition_closure_context_for_test",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_support_set_summary_counts_direct_and_missing_nodes():
    module = _load_module()
    baseline = np.asarray([0, 0, 1, 1, 2], dtype=np.uint64)
    candidate = np.asarray([0, 0, 1, 1, 2], dtype=np.uint64)
    vanilla = np.asarray([0, 3, 3, 1, 2], dtype=np.uint64)

    summary = module.support_set_summary(
        baseline_membership=baseline,
        candidate_membership=candidate,
        vanilla_membership=vanilla,
    )

    assert summary["candidate_support_size"] == 0
    assert summary["vanilla_support_size"] == 2
    assert summary["direct_support_edit_lower_bound"] == 2
    assert summary["missing_candidate_support_count"] == 0
    assert summary["support_symmetric_edit_lower_bound"] == 2
    assert summary["direct_nodes"].tolist() == [1, 2]


def test_closure_rows_measure_label_context_extra_nodes():
    module = _load_module()
    baseline = np.asarray([0, 0, 1, 1, 2], dtype=np.uint64)
    candidate_support = np.asarray([], dtype=np.uint32)
    vanilla_support = np.asarray([1, 2], dtype=np.uint32)
    direct_nodes = np.asarray([1, 2], dtype=np.uint32)

    rows, summary = module.closure_rows_for_mode(
        mode="baseline_label",
        membership=baseline,
        direct_nodes=direct_nodes,
        candidate_support=candidate_support,
        vanilla_support=vanilla_support,
        context={
            "case": "toy",
            "field": "field0",
            "method": "cc",
            "candidate_index": 0,
            "vanilla_seed": 11,
            "vanilla_randomness": 0.0,
            "vanilla_requested_n_iterations": 10,
        },
    )

    assert summary["closure_label_count"] == 2
    assert summary["closure_node_count"] == 4
    assert summary["closure_context_extra_count"] == 2
    assert summary["closure_outside_support_count"] == 2
    assert summary["closure_context_ratio"] == 1.0
    assert rows["direct_node_count"].sum() == 2
