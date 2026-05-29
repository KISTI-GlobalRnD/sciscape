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
    / "closure_context"
    / "run_leiden_basin_transition_label_internal_repair_pilot.py"
)


def _load_module():
    script_dir = SCRIPT_PATH.parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    spec = importlib.util.spec_from_file_location(
        "run_leiden_basin_transition_label_internal_repair_pilot_for_test",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _frontier_row(
    label: int,
    *,
    case: str = "toy",
    candidate_index: int = 0,
    ratio: float = 25.0,
    nodes: int = 5,
    selected: bool = True,
    score: float | None = None,
) -> dict:
    return {
        "case": case,
        "field": "field0",
        "method": "cc",
        "candidate_index": candidate_index,
        "vanilla_seed": 11,
        "vanilla_randomness": 0.0,
        "vanilla_requested_n_iterations": 10,
        "closure_mode": "candidate_label",
        "closure_label": label,
        "closure_node_count": nodes,
        "closure_context_ratio": ratio,
        "frontier_score": float(score if score is not None else 10.0 - label),
        "frontier_selected": selected,
    }


def test_selected_repair_labels_filters_high_ratio_selected_rows():
    module = _load_module()
    rows = pd.DataFrame(
        [
            _frontier_row(1, ratio=30.0, nodes=10, score=4.0),
            _frontier_row(2, ratio=5.0, nodes=10, score=100.0),
            _frontier_row(3, ratio=40.0, nodes=500, score=90.0),
            _frontier_row(4, ratio=35.0, nodes=10, selected=False, score=80.0),
            _frontier_row(5, case="toy2", candidate_index=1, ratio=22.0),
        ]
    )

    selected = module.selected_repair_labels(
        rows,
        closure_mode="candidate_label",
        max_pairs=1,
        max_labels_per_pair=1,
        min_closure_context_ratio=20.0,
        max_closure_nodes=300,
    )

    assert selected["closure_label"].tolist() == [1]


def test_closure_nodes_for_label_returns_candidate_membership_nodes():
    module = _load_module()

    nodes = module.closure_nodes_for_label(
        np.asarray([7, 7, 8, 7], dtype=np.uint64),
        7,
    )

    assert nodes.dtype == np.uint32
    assert nodes.tolist() == [0, 1, 3]


def test_mutable_nodes_for_label_repair_adds_boundary_anchors():
    module = _load_module()
    src = np.asarray([0, 1, 2, 3], dtype=np.uint32)
    dst = np.asarray([1, 2, 3, 4], dtype=np.uint32)

    mutable, truncated = module.mutable_nodes_for_label_repair(
        src=src,
        dst=dst,
        closure_nodes=np.asarray([2], dtype=np.uint32),
        direct_nodes=np.asarray([2], dtype=np.uint32),
        max_boundary_anchors=2,
    )

    assert mutable.tolist() == [1, 2, 3]
    assert truncated == 0


def test_split_closure_by_donor_uses_fresh_vanilla_substructure():
    module = _load_module()
    membership = np.asarray([0, 0, 0, 1], dtype=np.uint64)
    donor = np.asarray([5, 6, 6, 7], dtype=np.uint64)

    repaired = module.split_closure_by_donor(
        membership=membership,
        donor_membership=donor,
        closure_nodes=np.asarray([0, 1, 2], dtype=np.uint32),
    )

    assert repaired[1] == repaired[2]
    assert repaired[0] != repaired[1]
    assert repaired[3] != repaired[0]
    assert repaired[3] != repaired[1]


def test_diagnostic_label_for_repair_row_requires_control_and_support_shift():
    module = _load_module()
    same_basin = pd.Series(
        {
            "operator": module.RAW_OPERATOR,
            "delta_vs_candidate": 1.0,
            "delta_vs_vanilla": 1.0,
            "delta_vs_control_extra": 1.0,
            "result_support_distance_to_candidate": 0.05,
        }
    )
    support_shift = same_basin.copy()
    support_shift["result_support_distance_to_candidate"] = 0.2
    dominated = same_basin.copy()
    dominated["delta_vs_control_extra"] = -0.1
    loss = same_basin.copy()
    loss["delta_vs_candidate"] = -0.1

    assert module.diagnostic_label_for_repair_row(same_basin) == "quality_win_same_basin"
    assert module.diagnostic_label_for_repair_row(support_shift) == "quality_win_support_shift"
    assert module.diagnostic_label_for_repair_row(dominated) == "seed_control_dominates"
    assert module.diagnostic_label_for_repair_row(loss) == "quality_loss"
