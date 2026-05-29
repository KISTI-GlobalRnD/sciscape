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
    / "transition_diagnostics"
    / "analyze_leiden_basin_transition_landscape.py"
)


def _load_module():
    script_dir = SCRIPT_PATH.parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    spec = importlib.util.spec_from_file_location(
        "analyze_leiden_basin_transition_landscape_for_test",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_filter_frame_infers_field_method_from_candidate_case():
    module = _load_module()
    frame = pd.DataFrame(
        [
            {
                "case": (
                    "adaptive_refinement_leiden_hysteresis_exception_detector_graphs_"
                    "20260514_field34_all_edges_cc_cosine"
                ),
                "quality": 1.0,
            }
        ]
    )

    filtered = module._filter_frame(
        frame,
        field=34,
        method="cc_cosine",
        n_iterations=None,
    )

    assert len(filtered) == 1
    assert int(filtered.iloc[0]["field"]) == 34
    assert filtered.iloc[0]["method"] == "cc_cosine"


def test_support_overlap_reports_directional_subset():
    module = _load_module()
    candidate = pd.Series({module.CHANGED_SUPPORT_COLUMN: "1;2"})
    vanilla = pd.Series({module.CHANGED_SUPPORT_COLUMN: "1;2;3;4"})

    overlap = module._support_overlap(candidate, vanilla)

    assert overlap["left_support_size"] == 2
    assert overlap["right_support_size"] == 4
    assert overlap["support_intersection_size"] == 2
    assert overlap["left_overlap_ratio"] == 1.0
    assert overlap["right_overlap_ratio"] == 0.5
    assert bool(overlap["left_support_subset_of_right"]) is True
    assert bool(overlap["right_support_subset_of_left"]) is False


def test_transition_hypothesis_labels_candidate_core_inside_vanilla():
    module = _load_module()
    nodes = pd.DataFrame(
        [
            _node(
                node_id="candidate:case:0",
                node_kind="dongdaemun_candidate",
                delta=5.0,
                support="1;2",
            ),
            _node(
                node_id="vanilla:case:seed=11",
                node_kind="vanilla_seed_basin",
                delta=8.0,
                support="1;2;3;4",
            ),
        ]
    )

    edges = module.build_edge_rows(nodes)
    hypotheses = module.build_transition_hypotheses(edges)

    row = hypotheses.iloc[0]
    assert row["hypothesis"] == "candidate_local_core_inside_broader_vanilla"
    assert bool(row["candidate_support_subset_of_vanilla"]) is True
    assert row["candidate_overlap_ratio"] == 1.0
    assert row["vanilla_extra_support_size"] == 2
    assert row["endpoint_distance"] == 0.0


def _node(
    *,
    node_id: str,
    node_kind: str,
    delta: float,
    support: str,
) -> dict[str, object]:
    return {
        "node_id": node_id,
        "node_kind": node_kind,
        "case": "case",
        "field": 34,
        "method": "cc_cosine",
        "quality": 100.0 + delta,
        "quality_delta_vs_baseline": delta,
        "p5_basin_sketch_node_hash": "hash",
        "p5_basin_sketch_membership": "0;0;1;1",
        "p5_basin_changed_support_nodes": support,
    }
