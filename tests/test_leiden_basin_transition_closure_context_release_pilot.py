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
    / "run_leiden_basin_transition_closure_context_release_pilot.py"
)


def _load_module():
    script_dir = SCRIPT_PATH.parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    spec = importlib.util.spec_from_file_location(
        "run_leiden_basin_transition_closure_context_release_pilot_for_test",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _prefix_row(step: int, *, label: str = "quality_win_same_basin") -> dict:
    return {
        "case": "toy",
        "field": "field0",
        "method": "cc",
        "candidate_index": 0,
        "vanilla_seed": 11,
        "vanilla_randomness": 0.0,
        "vanilla_requested_n_iterations": 10,
        "closure_mode": "candidate_label",
        "operator": "closure_split_shrink_from_vanilla_candidate_nearest_raw",
        "diagnostic_label": label,
        "step_index": step,
        "released_closure_labels": ",".join(str(i) for i in range(1, step + 1)),
        "delta_vs_vanilla": 1.0,
        "delta_vs_control_extra": float(10 - step),
        "support_burden_reduction_vs_vanilla": step,
    }


def test_parse_label_prefix_ignores_empty_values():
    module = _load_module()

    assert module.parse_label_prefix("1,2,3") == (1, 2, 3)
    assert module.parse_label_prefix("") == ()


def test_selected_direct_prefix_rows_keeps_top_positive_prefixes():
    module = _load_module()
    rows = pd.DataFrame(
        [
            _prefix_row(1),
            _prefix_row(2),
            _prefix_row(3, label="quality_loss"),
        ]
    )

    selected = module.selected_direct_prefix_rows(
        rows,
        source_operators=("closure_split_shrink_from_vanilla_candidate_nearest_raw",),
        source_labels=("quality_win_same_basin",),
        max_pairs=1,
        max_prefixes_per_pair=1,
    )

    assert selected["step_index"].tolist() == [1]


def test_edge_pull_to_direct_nodes_scores_weighted_edges():
    module = _load_module()
    src = np.asarray([0, 1, 2, 3], dtype=np.uint32)
    dst = np.asarray([2, 2, 3, 4], dtype=np.uint32)
    weight = np.asarray([5.0, 1.0, 2.0, 7.0], dtype=np.float64)

    scored = module.edge_pull_to_direct_nodes(
        src=src,
        dst=dst,
        weight=weight,
        candidate_nodes=np.asarray([2, 3], dtype=np.uint32),
        direct_nodes=np.asarray([0, 1], dtype=np.uint32),
        node_count=5,
    ).set_index("node")

    assert scored.loc[2, "edge_pull_to_direct"] == 6.0
    assert scored.loc[3, "edge_pull_to_direct"] == 0.0


def test_bounded_context_nodes_for_label_prefers_outside_support_with_pull():
    module = _load_module()
    candidate_membership = np.asarray([7, 7, 7, 7, 8], dtype=np.uint64)
    src = np.asarray([0, 1, 2, 3], dtype=np.uint32)
    dst = np.asarray([2, 2, 3, 4], dtype=np.uint32)
    weight = np.asarray([1.0, 3.0, 2.0, 9.0], dtype=np.float64)

    selected, stats = module.bounded_context_nodes_for_label(
        candidate_membership=candidate_membership,
        label=7,
        direct_nodes=np.asarray([0, 1], dtype=np.uint32),
        support_union=np.asarray([0, 1], dtype=np.uint32),
        src=src,
        dst=dst,
        weight=weight,
        node_weights=np.ones(5, dtype=np.float64),
        max_context_nodes=1,
        context_pool="outside_support",
    )

    assert selected.tolist() == [2]
    assert stats["candidate_context_node_count"] == 2
    assert stats["selected_context_edge_pull_sum"] == 4.0


def test_diagnostic_label_marks_support_shift_only_after_moving_from_vanilla():
    module = _load_module()
    same_basin = pd.Series(
        {
            "operator": module.RAW_OPERATOR,
            "delta_vs_vanilla": 1.0,
            "delta_vs_control_extra": 1.0,
            "result_support_distance_to_vanilla": 0.05,
        }
    )
    shifted = same_basin.copy()
    shifted["result_support_distance_to_vanilla"] = 0.2

    assert module.diagnostic_label_for_context_row(same_basin) == "quality_win_same_basin"
    assert module.diagnostic_label_for_context_row(shifted) == "quality_win_support_shift"
