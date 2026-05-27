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
    / "calibrate_leiden_basin_transition_boundary_groups.py"
)


def _load_module():
    script_dir = SCRIPT_PATH.parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    spec = importlib.util.spec_from_file_location(
        "calibrate_leiden_basin_transition_boundary_groups_for_test",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_aligned_group_transplant_uses_best_partner_labels():
    module = _load_module()
    base = np.asarray([10, 10, 20, 20, 30], dtype=np.uint64)
    donor = np.asarray([1, 1, 2, 2, 2], dtype=np.uint64)

    transplanted, fallback_count = module.aligned_group_transplant(
        base,
        donor,
        np.asarray([2, 4], dtype=np.uint32),
    )

    assert fallback_count == 0
    assert transplanted.tolist() == [10, 10, 20, 20, 20]


def test_select_calibration_groups_limits_each_role_per_pair():
    module = _load_module()
    rows = pd.DataFrame(
        [
            {
                "case": "c",
                "candidate_index": 0,
                "vanilla_seed": 11,
                "vanilla_randomness": 0.0,
                "vanilla_requested_n_iterations": 10,
                "boundary_role": "collateral_like",
                "node_count": 1,
                "node_weight_sum": 1.0,
                "collateral_score_mean": 0.8,
                "bridge_score_mean": 0.0,
            },
            {
                "case": "c",
                "candidate_index": 0,
                "vanilla_seed": 11,
                "vanilla_randomness": 0.0,
                "vanilla_requested_n_iterations": 10,
                "boundary_role": "collateral_like",
                "node_count": 4,
                "node_weight_sum": 4.0,
                "collateral_score_mean": 0.7,
                "bridge_score_mean": 0.0,
            },
            {
                "case": "c",
                "candidate_index": 0,
                "vanilla_seed": 11,
                "vanilla_randomness": 0.0,
                "vanilla_requested_n_iterations": 10,
                "boundary_role": "bridge_like",
                "node_count": 2,
                "node_weight_sum": 2.0,
                "collateral_score_mean": 0.0,
                "bridge_score_mean": 0.9,
            },
        ]
    )

    selected = module.select_calibration_groups(
        rows,
        roles=("bridge_like", "collateral_like"),
        max_groups_per_role=1,
        max_groups_total=4,
        min_node_count=1,
    )

    assert selected["boundary_role"].tolist() == ["bridge_like", "collateral_like"]
    assert selected["calibration_group_id"].tolist() == ["group_0000", "group_0001"]
    assert selected.loc[1, "collateral_score_mean"] == 0.8


def test_action_initial_membership_reports_expected_base_and_donor():
    module = _load_module()
    baseline = np.asarray([0, 0, 1, 1], dtype=np.uint64)
    candidate = np.asarray([0, 2, 2, 1], dtype=np.uint64)
    vanilla = np.asarray([0, 3, 3, 1], dtype=np.uint64)

    initial, base_kind, donor_kind, fallback = module._action_initial_membership(
        action="vanilla_revert_candidate_aligned",
        group_nodes=np.asarray([1, 2], dtype=np.uint32),
        baseline_membership=baseline,
        candidate_membership=candidate,
        vanilla_membership=vanilla,
    )

    assert base_kind == "vanilla"
    assert donor_kind == "candidate"
    assert fallback == 0
    assert initial.tolist() == [0, 3, 3, 1]


def test_select_role_chunks_builds_node_id_backed_groups():
    module = _load_module()
    rows = pd.DataFrame(
        [
            {
                "case": "c",
                "field": "f",
                "method": "m",
                "candidate_index": 0,
                "vanilla_seed": 11,
                "vanilla_randomness": 0.0,
                "vanilla_requested_n_iterations": 10,
                "boundary_role": "collateral_like",
                "node": 5,
                "node_weight": 1.0,
                "incident_weight_total": 1.0,
                "bridge_score": 0.0,
                "collateral_score": 0.2,
                "necessity_score": -0.1,
                "boundary_role_margin": -0.2,
            },
            {
                "case": "c",
                "field": "f",
                "method": "m",
                "candidate_index": 0,
                "vanilla_seed": 11,
                "vanilla_randomness": 0.0,
                "vanilla_requested_n_iterations": 10,
                "boundary_role": "collateral_like",
                "node": 7,
                "node_weight": 1.0,
                "incident_weight_total": 3.0,
                "bridge_score": 0.0,
                "collateral_score": 0.8,
                "necessity_score": -0.4,
                "boundary_role_margin": -0.8,
            },
        ]
    )

    chunks = module.select_role_chunks(
        rows,
        roles=("collateral_like",),
        chunk_sizes=(2,),
        max_chunk_groups=1,
    )

    assert chunks.loc[0, "selection_kind"] == "role_chunk_2"
    assert chunks.loc[0, "node_ids"] == "7,5"
    assert chunks.loc[0, "node_count"] == 2
