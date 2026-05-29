from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "research"
    / "consensus"
    / "scripts"
    / "leiden_basin"
    / "transition_routes"
    / "closure_context"
    / "rank_leiden_basin_transition_closure_frontier.py"
)


def _load_module():
    script_dir = SCRIPT_PATH.parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    spec = importlib.util.spec_from_file_location(
        "rank_leiden_basin_transition_closure_frontier_for_test",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _label_row(label: int, direct: int, extra: int, ratio: float) -> dict:
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
        "direct_node_count": direct,
        "closure_node_count": direct + extra,
        "closure_context_extra_count": extra,
        "closure_outside_support_count": extra,
        "closure_candidate_support_count": 0,
        "closure_vanilla_support_count": direct,
        "closure_context_ratio": ratio,
    }


def _node_row(node: int, label: int, role: str) -> dict:
    return {
        "case": "toy",
        "field": "field0",
        "method": "cc",
        "candidate_index": 0,
        "vanilla_seed": 11,
        "vanilla_randomness": 0.0,
        "vanilla_requested_n_iterations": 10,
        "node": node,
        "support_class": "vanilla_extra",
        "boundary_role": role,
        "baseline_label": label,
        "candidate_label": label,
        "vanilla_label": label + 100,
        "node_weight": 1.0,
        "incident_weight_total": 1.0,
        "bridge_score": 0.1,
        "collateral_score": 0.9,
        "necessity_score": 0.2,
        "core_pull": 0.0,
        "vanilla_extra_pull": 0.1,
        "baseline_pull": 0.9,
        "candidate_pull": 0.9,
        "vanilla_pull": 0.1,
        "boundary_role_margin": -0.8,
    }


def test_direct_role_features_joins_boundary_roles_by_closure_mode():
    module = _load_module()
    labels = pd.DataFrame([_label_row(7, direct=2, extra=50, ratio=25.0)])
    nodes = pd.DataFrame(
        [
            _node_row(1, 7, "collateral_like"),
            _node_row(2, 7, "ambiguous"),
            _node_row(3, 8, "bridge_like"),
        ]
    )

    rows = module.direct_role_features(label_rows=labels, node_rows=nodes)

    row = rows.iloc[0]
    assert int(row["direct_collateral_like_nodes"]) == 1
    assert int(row["direct_ambiguous_nodes"]) == 1
    assert int(row["direct_bridge_like_nodes"]) == 0
    assert float(row["direct_collateralish_fraction"]) == 1.0
    assert float(row["direct_bridge_fraction"]) == 0.0


def test_score_frontier_rows_selects_top_eligible_and_marks_rejections():
    module = _load_module()
    labels = pd.DataFrame(
        [
            _label_row(7, direct=2, extra=50, ratio=25.0),
            _label_row(8, direct=2, extra=5, ratio=2.5),
            _label_row(9, direct=2, extra=80, ratio=40.0),
        ]
    )
    nodes = pd.DataFrame(
        [
            _node_row(1, 7, "collateral_like"),
            _node_row(2, 7, "ambiguous"),
            _node_row(3, 8, "collateral_like"),
            _node_row(4, 8, "ambiguous"),
            _node_row(5, 9, "bridge_like"),
            _node_row(6, 9, "bridge_like"),
        ]
    )
    rows = module.direct_role_features(label_rows=labels, node_rows=nodes)

    scored = module.score_frontier_rows(
        rows,
        closure_modes=("candidate_label",),
        min_context_extra=20,
        top_labels_per_pair=1,
    )

    by_label = scored.set_index("closure_label")
    assert bool(by_label.loc[7, "frontier_selected"])
    assert by_label.loc[8, "frontier_reason"] == "reject_low_context"
    assert by_label.loc[9, "frontier_reason"] == "reject_bridge_heavy"
