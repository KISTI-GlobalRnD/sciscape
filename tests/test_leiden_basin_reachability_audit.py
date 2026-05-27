"""Tests for useful-basin seed/iteration reachability audit."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "research" / "consensus" / "scripts"
ANALYSIS_PATH = SCRIPT_DIR / "analyze_leiden_basin_reachability_audit.py"


def _load_script(module_name: str):
    if str(ANALYSIS_PATH.parent) not in sys.path:
        sys.path.insert(0, str(ANALYSIS_PATH.parent))
    spec = importlib.util.spec_from_file_location(module_name, ANALYSIS_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_target_basin_rows_classify_material_and_core_targets():
    module = _load_script("leiden_basin_reachability_targets_for_test")
    targets = module.build_target_basin_rows(
        _candidate_frame(),
        material_regret_q=10.0,
        near_best_delta_q=1.0,
        support_distinct_tau=0.5,
    )

    by_candidate = {
        int(row["candidate_index"]): row for _, row in targets.iterrows()
    }
    assert by_candidate[2]["target_class"] == "material_winner"
    assert by_candidate[1]["target_class"] == "core_alternative"
    assert by_candidate[2]["p5_basin_signature"] == "basin-2"


def test_reachability_rows_distinguish_seed_reachable_and_unreached():
    module = _load_script("leiden_basin_reachability_rows_for_test")
    targets = module.build_target_basin_rows(
        _candidate_frame(),
        material_regret_q=10.0,
        near_best_delta_q=1.0,
        support_distinct_tau=0.5,
    )
    vanilla = pd.DataFrame(
        [
            _vanilla(signature="basin-2", seed=42),
            _vanilla(signature="basin-2", seed=73),
            _vanilla(signature="other-basin", seed=101),
        ]
    )

    rows = module.build_reachability_rows(targets, vanilla)
    material = rows[rows["target_class"] == "material_winner"].iloc[0]
    core = rows[
        (rows["target_class"] == "core_alternative")
        & (rows["candidate_index"] == 1)
    ].iloc[0]

    assert material["reachability_label"] == "seed_reachable"
    assert material["match_count"] == 2
    assert material["match_type"] == "exact_signature"
    assert core["reachability_label"] == "not_reached_in_available_sweep"
    assert bool(core["same_case_vanilla_has_signature_evidence"]) is True


def test_reachability_rows_do_not_overclaim_without_vanilla_signature_evidence():
    module = _load_script("leiden_basin_reachability_unresolved_for_test")
    targets = module.build_target_basin_rows(
        _candidate_frame(),
        material_regret_q=10.0,
        near_best_delta_q=1.0,
        support_distinct_tau=0.5,
    )
    vanilla = pd.DataFrame(
        [
            {
                "case": "adaptive_refinement_field12_gcc_emb_full_knn30_bc_cosine",
                "seed": 42,
                "randomness": 0.001,
                "n_iterations": 10,
                "quality": 123.0,
            }
        ]
    )

    rows = module.build_reachability_rows(targets, vanilla)

    assert set(rows["reachability_label"]) == {
        "unresolved_no_vanilla_signature_evidence"
    }


def test_reachability_rows_preserve_endpoint_near_support_far_near_miss():
    module = _load_script("leiden_basin_reachability_sketch_near_miss_for_test")
    targets = module.build_target_basin_rows(
        _candidate_frame(),
        material_regret_q=10.0,
        near_best_delta_q=1.0,
        support_distinct_tau=0.5,
    )
    vanilla = pd.DataFrame(
        [
            {
                **_vanilla(signature="other-basin", seed=42),
                "p5_basin_sketch_membership": "0;0;1;1;2;2;3;3",
                "p5_basin_changed_support_nodes": "6;7",
            }
        ]
    )

    rows = module.build_reachability_rows(targets, vanilla)
    core = rows[
        (rows["target_class"] == "core_alternative")
        & (rows["candidate_index"] == 1)
    ].iloc[0]
    summary = module.build_reachability_summary(rows)

    assert core["reachability_label"] == "not_reached_in_available_sweep"
    assert core["same_case_sketch_comparable_count"] == 1
    assert core["best_endpoint_distance"] == 0.0
    assert core["best_support_distance"] == 1.0
    assert core["best_support_intersection_size"] == 0
    assert core["best_support_union_size"] == 4
    assert core["best_target_support_size"] == 2
    assert core["best_vanilla_support_size"] == 2
    assert core["best_support_union_fraction_of_sketch"] == 0.5
    assert bool(core["endpoint_near_support_far"]) is True
    assert int(summary["endpoint_near_support_far_count"].sum()) >= 1
    assert summary["endpoint_near_support_far_union_median"].max() == 4.0


def test_reachability_summary_and_report(tmp_path):
    module = _load_script("leiden_basin_reachability_report_for_test")
    targets = module.build_target_basin_rows(
        _candidate_frame(),
        material_regret_q=10.0,
        near_best_delta_q=1.0,
        support_distinct_tau=0.5,
    )
    vanilla = pd.DataFrame(
        [
            _vanilla(signature="basin-2", seed=42),
            _vanilla(signature="other-basin", seed=101),
        ]
    )
    rows = module.build_reachability_rows(targets, vanilla)
    summary = module.build_reachability_summary(rows)

    material = summary[summary["target_class"] == "material_winner"].iloc[0]
    assert material["rare_seed_reachable_count"] == 1

    module.write_report(tmp_path, targets, rows, summary, vanilla)
    report = (
        tmp_path / "dongdaemun_basin_reachability_audit_report.md"
    ).read_text(encoding="utf-8")
    assert "Dongdaemun Basin Reachability Audit" in report
    assert "Reachability Summary" in report
    assert "must not be interpreted as perturbation-only" in report


def test_report_surfaces_endpoint_near_support_far_examples(tmp_path):
    module = _load_script("leiden_basin_reachability_near_miss_report_for_test")
    targets = module.build_target_basin_rows(
        _candidate_frame(),
        material_regret_q=10.0,
        near_best_delta_q=1.0,
        support_distinct_tau=0.5,
    )
    vanilla = pd.DataFrame(
        [
            {
                **_vanilla(signature="other-basin", seed=42),
                "p5_basin_sketch_membership": "0;0;1;1;2;2;3;3",
                "p5_basin_changed_support_nodes": "6;7",
            }
        ]
    )
    rows = module.build_reachability_rows(targets, vanilla)
    summary = module.build_reachability_summary(rows)

    module.write_report(tmp_path, targets, rows, summary, vanilla)

    report = (
        tmp_path / "dongdaemun_basin_reachability_audit_report.md"
    ).read_text(encoding="utf-8")
    assert "Endpoint-Near Support-Far Examples" in report
    assert "best_support_union_size" in report


def _candidate_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _candidate(0, p1=50.0, p5=0.0, support_nodes="0;1"),
            _candidate(1, p1=40.0, p5=99.5, support_nodes="0;1"),
            _candidate(2, p1=30.0, p5=100.0, support_nodes="2;3"),
            _candidate(3, p1=20.0, p5=20.0, support_nodes="4;5"),
            _candidate(4, p1=10.0, p5=20.5, support_nodes="6;7"),
        ]
    )


def _candidate(
    candidate_index: int,
    *,
    p1: float,
    p5: float,
    support_nodes: str,
    rel: float = 100.0,
) -> dict[str, object]:
    return {
        "candidate_eval_mode": "multifidelity_label",
        "case": "adaptive_refinement_field12_gcc_emb_full_knn30_bc_cosine",
        "seed": 11,
        "candidate_budget": 5,
        "max_group_candidates": 5,
        "candidate_index": candidate_index,
        "p1_delta_q": p1,
        "p5_delta_q": p5,
        "p5_relative_delta_q_ppm": rel + candidate_index,
        "p5_basin_signature": f"basin-{candidate_index}",
        "p5_changed_fraction_vs_baseline": 0.001 * (candidate_index + 1),
        "p5_changed_nodes_vs_baseline": candidate_index + 1,
        "p5_basin_sketch_node_hash": "sample-hash",
        "p5_basin_sketch_baseline_membership": "0;0;1;1;2;2;3;3",
        "p5_basin_sketch_membership": "0;0;1;1;2;2;3;3",
        "p5_basin_changed_support_node_count": 2,
        "p5_basin_changed_support_sketch_sample_size": 2,
        "p5_basin_changed_support_nodes": support_nodes,
    }


def _vanilla(*, signature: str, seed: int) -> dict[str, object]:
    if signature == "basin-2":
        membership = "0;0;1;1;2;2;3;3"
        support_nodes = "2;3"
    else:
        membership = "0;1;0;1;2;3;2;3"
        support_nodes = "6;7"
    return {
        "case": "adaptive_refinement_field12_gcc_emb_full_knn30_bc_cosine",
        "field": 12,
        "method": "bc_cosine",
        "seed": seed,
        "randomness": 0.001,
        "requested_n_iterations": "10",
        "quality": 123.0 + seed,
        "p5_basin_signature": signature,
        "p5_basin_sketch_node_hash": "sample-hash",
        "p5_basin_sketch_membership": membership,
        "p5_basin_changed_support_nodes": support_nodes,
    }
