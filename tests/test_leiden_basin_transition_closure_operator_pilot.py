from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "research"
    / "consensus"
    / "scripts"
    / "leiden_basin"
    / "transition_routes"
    / "transition_operators"
    / "run_leiden_basin_transition_closure_operator_pilot.py"
)


def _load_module():
    script_dir = SCRIPT_PATH.parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    spec = importlib.util.spec_from_file_location(
        "run_leiden_basin_transition_closure_operator_pilot_for_test",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _frontier_row(label: int, *, selected: bool = True) -> dict:
    return {
        "case": "toy",
        "field": "field0",
        "method": "cc",
        "candidate_index": 0,
        "vanilla_seed": 11,
        "vanilla_randomness": 0.0,
        "vanilla_requested_n_iterations": 10,
        "closure_mode": "candidate_label",
        "closure_label": label,
        "closure_node_count": 5,
        "closure_context_extra_count": 3,
        "closure_outside_support_count": 3,
        "direct_node_weight_sum": 2.0,
        "frontier_score": 10.0 - label,
        "frontier_rank_in_pair": label,
        "frontier_selected": selected,
    }


def _node_row(node: int, candidate_label: int, support_class: str) -> dict:
    return {
        "case": "toy",
        "field": "field0",
        "method": "cc",
        "candidate_index": 0,
        "vanilla_seed": 11,
        "vanilla_randomness": 0.0,
        "vanilla_requested_n_iterations": 10,
        "node": node,
        "support_class": support_class,
        "baseline_label": candidate_label,
        "candidate_label": candidate_label,
        "vanilla_label": candidate_label + 100,
    }


def test_selected_frontier_rows_keeps_pair_top_ranked_selected_labels():
    module = _load_module()
    rows = pd.DataFrame(
        [
            _frontier_row(1),
            _frontier_row(2),
            _frontier_row(3, selected=False),
        ]
    )

    selected = module.selected_frontier_rows(
        rows,
        closure_mode="candidate_label",
        max_pairs=1,
        max_labels_per_pair=1,
    )

    assert selected["closure_label"].tolist() == [1]


def test_direct_nodes_for_frontier_row_uses_direct_vanilla_extra_nodes_only():
    module = _load_module()
    frontier = pd.Series(_frontier_row(7))
    nodes = pd.DataFrame(
        [
            _node_row(1, 7, "vanilla_extra"),
            _node_row(2, 7, "shared"),
            _node_row(3, 8, "vanilla_extra"),
        ]
    )

    direct = module.direct_nodes_for_frontier_row(
        node_rows=nodes,
        frontier_row=frontier,
    )

    assert direct.tolist() == [1]


def test_split_nodes_to_fresh_donor_labels_preserves_donor_coassignment():
    module = _load_module()
    membership = np.asarray([0, 0, 1, 1, 2], dtype=np.uint64)
    donor = np.asarray([5, 5, 6, 6, 5], dtype=np.uint64)

    split, mapping, next_label = module.split_nodes_to_fresh_donor_labels(
        membership,
        donor,
        np.asarray([0, 2, 3], dtype=np.uint32),
    )

    assert split[0] != membership[0]
    assert split[2] == split[3]
    assert split[0] != split[2]
    assert set(mapping) == {5, 6}
    assert next_label == int(membership.max()) + 3


def test_assign_nodes_to_nearest_existing_donor_label_uses_context_label():
    module = _load_module()
    membership = np.asarray([0, 9, 9, 2, 2], dtype=np.uint64)
    donor = np.asarray([5, 5, 5, 6, 6], dtype=np.uint64)

    assigned, next_label = module.assign_nodes_to_nearest_existing_donor_label(
        membership,
        donor,
        np.asarray([0], dtype=np.uint32),
        blocked_nodes=np.asarray([0], dtype=np.uint32),
    )

    assert assigned[0] == 9
    assert next_label == int(membership.max()) + 1


def test_evaluate_result_records_support_burden_reduction():
    module = _load_module()
    baseline = module.RecreatedMembership(
        membership=np.asarray([0, 0, 1, 1], dtype=np.uint64),
        quality=10.0,
        elapsed_sec=0.0,
    )
    candidate = module.CandidateMembership(
        recreated=module.RecreatedMembership(
            membership=np.asarray([0, 0, 1, 1], dtype=np.uint64),
            quality=10.0,
            elapsed_sec=0.0,
        ),
        row=pd.Series({}),
        group_nodes=np.asarray([], dtype=np.uint32),
        support_nodes=np.asarray([], dtype=np.uint32),
    )
    vanilla = module.RecreatedMembership(
        membership=np.asarray([0, 2, 2, 1], dtype=np.uint64),
        quality=12.0,
        elapsed_sec=0.0,
    )
    result = module.RecreatedMembership(
        membership=np.asarray([0, 0, 1, 1], dtype=np.uint64),
        quality=10.0,
        elapsed_sec=0.1,
    )

    row = module._evaluate_result(
        context={"case": "toy"},
        operator="raw",
        result=result,
        baseline=baseline,
        candidate=candidate,
        vanilla=vanilla,
        candidate_support=np.asarray([], dtype=np.uint32),
        vanilla_support=module.changed_support_nodes(
            baseline.membership,
            vanilla.membership,
        ),
        sketch_nodes=np.asarray([0, 1, 2, 3], dtype=np.uint32),
        released_stats=module._zero_release_stats(),
    )

    assert row["result_support_size"] == 0
    assert row["support_burden_reduction_vs_vanilla"] == 2
    assert row["delta_vs_vanilla"] == -2.0


def test_diagnostic_label_separates_same_basin_from_support_shift():
    module = _load_module()
    same_basin = pd.Series(
        {
            "operator": module.NEAREST_RAW_OPERATOR,
            "delta_vs_vanilla": 1.0,
            "delta_vs_control_extra": 0.5,
            "result_support_distance_to_vanilla": 0.02,
        }
    )
    support_shift = same_basin.copy()
    support_shift["result_support_distance_to_vanilla"] = 0.2

    assert module.diagnostic_label_for_row(same_basin) == "quality_win_same_basin"
    assert module.diagnostic_label_for_row(support_shift) == "quality_win_support_shift"
