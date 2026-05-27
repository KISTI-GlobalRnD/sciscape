"""Tests for Dongdaemun contract tiering and oracle subset diagnostics."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "research" / "consensus" / "scripts"
ANALYSIS_PATH = SCRIPT_DIR / "analyze_leiden_contract_tiered_subset.py"


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


def test_tiered_subset_rows_split_hard_core_and_diagnostic_requirements():
    module = _load_script("leiden_contract_tiered_subset_for_test")
    candidates = _candidate_frame()

    rows = module.build_tiered_subset_rows(
        candidates,
        material_regret_q=10.0,
        near_best_delta_q=1.0,
        support_distinct_tau=0.5,
    )
    by_tier = {row["contract_tier"]: row for _, row in rows.iterrows()}

    hard = by_tier["hard"]
    assert hard["required_candidate_indices"] == "0;2"
    assert hard["oracle_required_candidate_count"] == 2
    assert hard["p1_prefix_k_required"] == 3
    assert hard["p1_prefix_overhead_count"] == 1
    assert hard["support_distinct_pair_required_count"] == 0
    assert bool(hard["p1_top1_covers_tier"]) is False
    assert bool(hard["p1_top3_covers_tier"]) is True

    core = by_tier["core"]
    assert core["required_candidate_indices"] == "0;1;2"
    assert core["oracle_required_candidate_count"] == 3
    assert core["p1_prefix_k_required"] == 3
    assert core["support_distinct_pair_required_count"] == 1
    assert bool(core["p1_top3_covers_tier"]) is True

    diagnostic = by_tier["diagnostic"]
    assert diagnostic["required_candidate_indices"] == "0;1;2;3;4"
    assert diagnostic["oracle_required_candidate_count"] == 5
    assert diagnostic["p1_prefix_k_required"] == 5
    assert diagnostic["support_distinct_pair_required_count"] == 2
    assert bool(diagnostic["p1_top3_covers_tier"]) is False
    assert bool(diagnostic["p1_top5_covers_tier"]) is True


def test_pair_tiers_mark_near_best_pairs_as_core_and_other_pairs_as_diagnostic():
    module = _load_script("leiden_contract_pair_tier_for_test")
    candidates = _candidate_frame()

    pairs = module.build_pair_tier_rows(
        candidates,
        material_regret_q=10.0,
        near_best_delta_q=1.0,
        support_distinct_tau=0.5,
    )

    assert len(pairs) == 2
    pair_by_key = {
        tuple(sorted((int(row["left_candidate_index"]), int(row["right_candidate_index"])))): row
        for _, row in pairs.iterrows()
    }
    assert pair_by_key[(1, 2)]["pair_obligation_tier"] == "core"
    assert pair_by_key[(1, 2)]["pair_tier_reason"] in {
        "touches_quality_first_best",
        "touches_near_qf_candidate",
    }
    assert pair_by_key[(3, 4)]["pair_obligation_tier"] == "diagnostic"
    assert pair_by_key[(3, 4)]["pair_tier_reason"] == "support_distinct_iso_q_inventory"


def test_tiered_subset_summary_and_report(tmp_path):
    module = _load_script("leiden_contract_tiered_report_for_test")
    candidates = _candidate_frame()
    tier_rows = module.build_tiered_subset_rows(
        candidates,
        material_regret_q=10.0,
        near_best_delta_q=1.0,
        support_distinct_tau=0.5,
    )
    pair_rows = module.build_pair_tier_rows(
        candidates,
        material_regret_q=10.0,
        near_best_delta_q=1.0,
        support_distinct_tau=0.5,
    )
    summary = module.build_tiered_subset_summary(tier_rows)

    hard = summary[summary["contract_tier"] == "hard"].iloc[0]
    diagnostic = summary[summary["contract_tier"] == "diagnostic"].iloc[0]
    assert hard["oracle_required_candidate_count_mean"] == 2.0
    assert hard["p1_top3_covers_tier_count"] == 1
    assert diagnostic["oracle_required_candidate_count_mean"] == 5.0
    assert diagnostic["p1_top3_covers_tier_count"] == 0
    assert diagnostic["p1_top5_covers_tier_count"] == 1

    module.write_report(tmp_path, tier_rows, pair_rows, summary)
    report = (tmp_path / "dongdaemun_contract_tiered_subset_report.md").read_text(
        encoding="utf-8"
    )
    assert "Dongdaemun Contract Tiered Oracle Subset" in report
    assert "hard/core/diagnostic obligations" in report
    assert "diagnostic: oracle mean candidates 5" in report


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
        "p5_elapsed_ms": 10.0 + candidate_index,
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
