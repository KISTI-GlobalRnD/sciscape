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
    / "analyze_leiden_basin_transition_boundaries.py"
)


def _load_module():
    script_dir = SCRIPT_PATH.parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    spec = importlib.util.spec_from_file_location(
        "analyze_leiden_basin_transition_boundaries_for_test",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_classify_support_nodes_separates_core_shared_and_extra():
    module = _load_module()

    classes = module.classify_support_nodes(
        np.asarray([1, 3], dtype=np.uint32),
        np.asarray([1, 2], dtype=np.uint32),
    )

    assert classes == {1: "shared", 2: "vanilla_extra", 3: "core"}


def test_boundary_anchor_nodes_returns_sorted_capped_one_hop_context():
    module = _load_module()
    src = np.asarray([0, 1, 2, 4], dtype=np.uint32)
    dst = np.asarray([1, 2, 3, 2], dtype=np.uint32)

    anchors, truncated = module.boundary_anchor_nodes(
        src,
        dst,
        np.asarray([1, 2], dtype=np.uint32),
        max_anchors=2,
    )

    assert anchors.tolist() == [0, 3]
    assert truncated == 1


def test_compute_boundary_node_rows_marks_bridge_and_collateral_extra_nodes():
    module = _load_module()
    src = np.asarray([1, 2, 4, 4, 2], dtype=np.uint32)
    dst = np.asarray([2, 3, 5, 1, 4], dtype=np.uint32)
    weight = np.asarray([5.0, 1.0, 4.0, 0.2, 0.1], dtype=np.float64)
    node_weights = np.ones(6, dtype=np.float64)
    baseline = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.uint64)
    candidate = np.asarray([0, 3, 1, 1, 2, 2], dtype=np.uint64)
    vanilla = np.asarray([0, 3, 3, 1, 7, 7], dtype=np.uint64)

    rows, summary = module.compute_boundary_node_rows(
        src=src,
        dst=dst,
        weight=weight,
        node_weights=node_weights,
        baseline_membership=baseline,
        candidate_membership=candidate,
        vanilla_membership=vanilla,
        candidate_support=np.asarray([1], dtype=np.uint32),
        vanilla_support=np.asarray([1, 2, 4], dtype=np.uint32),
        context={
            "case": "toy",
            "field": "field0",
            "method": "cc",
            "candidate_index": 0,
            "vanilla_seed": 11,
            "vanilla_randomness": 0.0,
            "vanilla_requested_n_iterations": 10,
            "source_cluster": 0,
            "target_cluster": 1,
        },
        max_boundary_anchors=8,
        role_margin=0.05,
    )

    by_node = rows.set_index("node")
    assert by_node.loc[1, "support_class"] == "shared"
    assert by_node.loc[2, "support_class"] == "vanilla_extra"
    assert by_node.loc[2, "boundary_role"] == "bridge_like"
    assert by_node.loc[4, "support_class"] == "vanilla_extra"
    assert by_node.loc[4, "boundary_role"] == "collateral_like"
    assert set(by_node[by_node["support_class"].eq("boundary_anchor")].index) == {3, 5}
    assert summary["candidate_support_size"] == 1
    assert summary["vanilla_support_size"] == 3


def test_aggregate_boundary_group_rows_is_deterministic():
    module = _load_module()
    src = np.asarray([1, 2, 4, 4, 2], dtype=np.uint32)
    dst = np.asarray([2, 3, 5, 1, 4], dtype=np.uint32)
    weight = np.asarray([5.0, 1.0, 4.0, 0.2, 0.1], dtype=np.float64)
    node_weights = np.ones(6, dtype=np.float64)
    baseline = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.uint64)
    candidate = np.asarray([0, 3, 1, 1, 2, 2], dtype=np.uint64)
    vanilla = np.asarray([0, 3, 3, 1, 7, 7], dtype=np.uint64)

    rows, _summary = module.compute_boundary_node_rows(
        src=src,
        dst=dst,
        weight=weight,
        node_weights=node_weights,
        baseline_membership=baseline,
        candidate_membership=candidate,
        vanilla_membership=vanilla,
        candidate_support=np.asarray([1], dtype=np.uint32),
        vanilla_support=np.asarray([1, 2, 4], dtype=np.uint32),
        context={
            "case": "toy",
            "field": "field0",
            "method": "cc",
            "candidate_index": 0,
            "vanilla_seed": 11,
            "vanilla_randomness": 0.0,
            "vanilla_requested_n_iterations": 10,
            "source_cluster": 0,
            "target_cluster": 1,
        },
        max_boundary_anchors=8,
        role_margin=0.05,
    )

    groups = module.aggregate_boundary_group_rows(rows)

    assert {"bridge_like", "collateral_like"} <= set(groups["boundary_role"])
    assert int(groups["node_count"].sum()) == len(rows)
