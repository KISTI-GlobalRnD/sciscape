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
    / "analyze_leiden_basin_transition_minimal_pathway.py"
)


class FakeGraph:
    def cpm_quality(self, membership, *, resolution):
        del resolution
        membership = np.asarray(membership, dtype=np.uint64)
        return float(np.sum(membership != np.uint64(3)))


def _load_module():
    script_dir = SCRIPT_PATH.parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    spec = importlib.util.spec_from_file_location(
        "analyze_leiden_basin_transition_minimal_pathway_for_test",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_pathway_units_for_pair_selects_only_vanilla_extra_groups():
    module = _load_module()
    node_rows = pd.DataFrame(
        [
            {
                "case": "c",
                "field": "f",
                "method": "m",
                "candidate_index": 0,
                "vanilla_seed": 11,
                "vanilla_randomness": 0.0,
                "vanilla_requested_n_iterations": 10,
                "support_class": "vanilla_extra",
                "boundary_role": "collateral_like",
                "baseline_label": 1,
                "candidate_label": 1,
                "vanilla_label": 3,
                "node": 2,
            },
            {
                "case": "c",
                "field": "f",
                "method": "m",
                "candidate_index": 0,
                "vanilla_seed": 11,
                "vanilla_randomness": 0.0,
                "vanilla_requested_n_iterations": 10,
                "support_class": "shared",
                "boundary_role": "shared",
                "baseline_label": 0,
                "candidate_label": 4,
                "vanilla_label": 4,
                "node": 1,
            },
        ]
    )
    group_rows = pd.DataFrame(
        [
            {
                "case": "c",
                "field": "f",
                "method": "m",
                "candidate_index": 0,
                "vanilla_seed": 11,
                "vanilla_randomness": 0.0,
                "vanilla_requested_n_iterations": 10,
                "support_class": "vanilla_extra",
                "boundary_role": "collateral_like",
                "baseline_label": 1,
                "candidate_label": 1,
                "vanilla_label": 3,
                "node_count": 1,
                "node_weight_sum": 1.0,
                "bridge_score_mean": 0.0,
                "collateral_score_mean": 1.0,
                "necessity_score_mean": -1.0,
                "boundary_role_margin_mean": -1.0,
            },
            {
                "case": "c",
                "field": "f",
                "method": "m",
                "candidate_index": 0,
                "vanilla_seed": 11,
                "vanilla_randomness": 0.0,
                "vanilla_requested_n_iterations": 10,
                "support_class": "shared",
                "boundary_role": "shared",
                "baseline_label": 0,
                "candidate_label": 4,
                "vanilla_label": 4,
                "node_count": 1,
                "node_weight_sum": 1.0,
                "bridge_score_mean": 0.0,
                "collateral_score_mean": 0.0,
                "necessity_score_mean": 0.0,
                "boundary_role_margin_mean": 0.0,
            },
        ]
    )

    units = module.pathway_units_for_pair(
        node_rows=node_rows,
        group_rows=group_rows,
        pair_key={
            "case": "c",
            "candidate_index": 0,
            "vanilla_seed": 11,
            "vanilla_randomness": 0.0,
            "vanilla_requested_n_iterations": "10",
        },
    )

    assert units["unit_id"].tolist() == ["unit_00000"]
    assert units["node_ids"].tolist() == ["2"]


def test_order_units_least_direct_debt_prefers_larger_delta():
    module = _load_module()
    units = pd.DataFrame(
        [
            {"unit_id": "a", "direct_delta_q": -2.0, "unit_node_count": 1, "collateral_score_mean": 1.0},
            {"unit_id": "b", "direct_delta_q": 0.5, "unit_node_count": 1, "collateral_score_mean": 0.0},
        ]
    )

    ordered = module.order_units(units, policy="least_direct_debt")

    assert ordered["unit_id"].tolist() == ["b", "a"]


def test_compute_cumulative_pathway_records_quality_barrier():
    module = _load_module()
    baseline = np.asarray([0, 0, 1, 1], dtype=np.uint64)
    candidate = np.asarray([0, 0, 1, 1], dtype=np.uint64)
    vanilla = np.asarray([0, 0, 3, 3], dtype=np.uint64)
    units = pd.DataFrame(
        [
            {
                "unit_id": "u0",
                "node_ids": "2",
                "unit_node_count": 1,
                "boundary_role": "collateral_like",
                "direct_delta_q": 1.0,
                "collateral_score_mean": 1.0,
            },
            {
                "unit_id": "u1",
                "node_ids": "3",
                "unit_node_count": 1,
                "boundary_role": "collateral_like",
                "direct_delta_q": 1.0,
                "collateral_score_mean": 1.0,
            },
        ]
    )

    steps, summary = module.compute_cumulative_pathway(
        graph=FakeGraph(),
        units=units,
        action="baseline_forced",
        ordering_policy="least_direct_debt",
        baseline_membership=baseline,
        candidate_membership=candidate,
        vanilla_membership=vanilla,
        baseline_quality=4.0,
        candidate_quality=4.0,
        vanilla_quality=2.0,
        sketch_nodes=np.asarray([0, 1, 2, 3], dtype=np.uint32),
        resolution=0.01,
        context={
            "case": "c",
            "field": "f",
            "method": "m",
            "candidate_index": 0,
            "vanilla_seed": 11,
            "vanilla_randomness": 0.0,
            "vanilla_requested_n_iterations": 10,
        },
    )

    assert steps["quality"].tolist() == [3.0, 4.0]
    assert summary["node_edit_lower_bound"] == 2
    assert summary["residual_extra_support_after_path"] == 0
    assert summary["missing_candidate_support_after_path"] == 0
    assert summary["closure_node_edit_lower_bound"] == 2
    assert summary["quality_barrier"] == 0.0
    assert summary["final_support_distance_to_candidate"] == 0.0
