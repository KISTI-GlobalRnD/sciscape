"""Tests for offline Leiden branch-lookahead analysis."""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "research"
    / "consensus"
    / "scripts"
    / "leiden_basin"
    / "basin_signatures"
    / "branch_growth"
    / "analyze_leiden_branch_lookahead.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "analyze_leiden_branch_lookahead_for_test",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_synthetic_rows(path: Path) -> None:
    budgets = ("1", "2", "3", "5", "10", "convergence")
    candidates = {
        1: {"randomness": 0.0, "qualities": [100, 98, 94, 88, 82, 80], "pressure": 1.6},
        2: {"randomness": 0.001, "qualities": [90, 96, 104, 116, 119, 120], "pressure": 1.0},
        3: {"randomness": 0.01, "qualities": [80, 88, 96, 106, 111, 110], "pressure": 0.9},
        4: {"randomness": 0.0, "qualities": [70, 82, 92, 102, 106, 105], "pressure": 0.95},
        5: {"randomness": 0.001, "qualities": [60, 72, 84, 92, 96, 95], "pressure": 1.1},
        6: {"randomness": 0.02, "qualities": [50, 86, 98, 118, 131, 130], "pressure": 0.7},
    }
    fieldnames = [
        "sample",
        "source_sample",
        "edge_layer",
        "variant",
        "seed",
        "randomness",
        "requested_n_iterations",
        "iteration_mode",
        "n_iterations",
        "n_iterations_used",
        "supported",
        "elapsed_sec",
        "quality",
        "n_clusters",
        "max_doc_weight_ratio",
        "n_above_max_doc_weight",
    ]
    rows = []
    for seed, candidate in candidates.items():
        for index, budget in enumerate(budgets):
            pressure = candidate["pressure"] - index * 0.01
            rows.append(
                {
                    "sample": "tiny_bc",
                    "source_sample": "tiny",
                    "edge_layer": "bc_cosine",
                    "variant": "standard_leiden",
                    "seed": seed,
                    "randomness": candidate["randomness"],
                    "requested_n_iterations": budget,
                    "iteration_mode": "convergence" if budget == "convergence" else "fixed",
                    "n_iterations": 0 if budget == "convergence" else int(budget),
                    "n_iterations_used": 12 if budget == "convergence" else int(budget),
                    "supported": True,
                    "elapsed_sec": (index + 1) * 0.1 + seed * 0.001,
                    "quality": candidate["qualities"][index],
                    "n_clusters": 10 + seed,
                    "max_doc_weight_ratio": pressure,
                    "n_above_max_doc_weight": 1 if pressure > 1.0 else 0,
                }
            )
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_rank_path_and_late_riser_faller_classification(tmp_path):
    module = _load_module()
    csv_path = tmp_path / "rows.csv"
    _write_synthetic_rows(csv_path)
    rows = module.load_profile_rows(csv_path)

    rank_rows = module.build_rank_flip_rows(rows)
    late_rows = module.build_late_riser_faller_rows(rows)

    late_riser = next(
        row
        for row in rank_rows
        if row["candidate_id"] == "seed=6|randomness=0.02"
    )
    late_faller = next(
        row
        for row in rank_rows
        if row["candidate_id"] == "seed=1|randomness=0"
    )
    assert late_riser["iter1_rank"] == 6
    assert late_riser["convergence_rank"] == 1
    assert late_faller["iter1_rank"] == 1
    assert late_faller["convergence_rank"] == 6

    assert any(
        row["candidate_id"] == "seed=6|randomness=0.02"
        and row["classification"] == "late_riser"
        and row["early_budget"] == "1"
        and row["final_budget"] == "convergence"
        for row in late_rows
    )
    assert any(
        row["candidate_id"] == "seed=1|randomness=0"
        and row["classification"] == "late_faller"
        and row["early_budget"] == "1"
        and row["final_budget"] == "convergence"
        for row in late_rows
    )


def test_survival_by_budget_calculates_final_top_hits(tmp_path):
    module = _load_module()
    csv_path = tmp_path / "rows.csv"
    _write_synthetic_rows(csv_path)
    rows = module.load_profile_rows(csv_path)

    survival_rows = module.build_branch_survival_rows(rows, top_k_values=(3, 5))
    top3_row = next(
        row
        for row in survival_rows
        if row["sample"] == "tiny_bc"
        and row["early_budget"] == "1"
        and row["top_k"] == 3
        and row["final_budget"] == "convergence"
    )
    top5_row = next(
        row
        for row in survival_rows
        if row["sample"] == "tiny_bc"
        and row["early_budget"] == "1"
        and row["top_k"] == 5
        and row["final_budget"] == "convergence"
    )

    assert top3_row["final_top1_in_early_topk"] is False
    assert top3_row["final_top3_hit_count"] == 2
    assert top3_row["final_top3_hit_rate"] == 2 / 3
    assert top5_row["final_top1_in_early_topk"] is False


def test_mixed_beam_retains_quality_pressure_and_diversity_without_duplicates(tmp_path):
    module = _load_module()
    csv_path = tmp_path / "rows.csv"
    _write_synthetic_rows(csv_path)
    rows = module.load_profile_rows(csv_path)
    stage1_rows = [row for row in rows if row["requested_n_iterations"] == "1"]

    keys, reasons = module.select_policy_candidates("mixed_beam_v1", stage1_rows)
    candidate_ids = [module._candidate_id_from_key(key) for key in keys]

    assert len(candidate_ids) == len(set(candidate_ids))
    assert "seed=1|randomness=0" in candidate_ids
    assert "seed=2|randomness=0.001" in candidate_ids
    assert "seed=3|randomness=0.01" in candidate_ids
    assert "seed=6|randomness=0.02" in candidate_ids
    assert "pressure_safe_top1" in reasons[(6, 0.02)]


def test_mixed_beam_v2_adds_seed_family_rescue_without_duplicates():
    module = _load_module()
    stage1_rows = [
        {
            "sample": "tiny_bc",
            "seed": seed,
            "randomness": randomness,
            "supported": True,
            "quality": quality,
            "max_doc_weight_ratio": pressure,
            "n_above_max_doc_weight": 0,
            "elapsed_sec": 1.0,
        }
        for seed, randomness, quality, pressure in [
            (10, 0.0, 100.0, 1.0),
            (11, 0.0, 99.0, 1.1),
            (12, 0.0, 98.0, 1.2),
            (13, 0.0, 97.0, 1.3),
            (14, 0.0, 96.0, 1.4),
            (10, 0.01, 95.0, 1.0),
            (15, 0.0, 94.0, 0.8),
            (16, 0.03, 93.0, 1.0),
        ]
    ]

    keys, reasons = module.select_policy_candidates("mixed_beam_v2", stage1_rows)
    candidate_ids = [module._candidate_id_from_key(key) for key in keys]

    assert len(candidate_ids) == len(set(candidate_ids))
    assert "seed=10|randomness=0.01" in candidate_ids
    assert "seed_family_rescue" in reasons[(10, 0.01)]


def test_iter5_screen_all_policies_recover_late_riser_and_track_cost(tmp_path):
    module = _load_module()
    csv_path = tmp_path / "rows.csv"
    _write_synthetic_rows(csv_path)
    rows = module.load_profile_rows(csv_path)

    policy_rows = module.simulate_branch_policies(rows)
    top3 = next(
        row
        for row in policy_rows
        if row["policy_name"] == "iter5_screen_all_top3"
    )
    top5 = next(
        row
        for row in policy_rows
        if row["policy_name"] == "iter5_screen_all_top5"
    )

    assert top3["stage1_budget"] == "5"
    assert top3["stage2_promote_k"] == 3
    assert top3["n_final_evaluated"] == 3
    assert top3["selected_candidate_id"] == "seed=6|randomness=0.02"
    assert top3["selected_budget"] == "convergence"
    assert top3["quality_gap_to_best_convergence"] == 0.0
    assert top5["n_final_evaluated"] == 5
    assert top5["estimated_elapsed_proxy_sec"] > top3["estimated_elapsed_proxy_sec"]


def test_margin_polish_top2_uses_threshold_and_polishes_two_candidates(tmp_path):
    module = _load_module()
    csv_path = tmp_path / "rows.csv"
    _write_synthetic_rows(csv_path)
    rows = module.load_profile_rows(csv_path)

    policy_rows = module.simulate_branch_policies(rows)
    margin_row = next(
        row for row in policy_rows if row["policy_name"] == "margin_polish_top2"
    )

    assert margin_row["convergence_polish_k"] == 2
    assert margin_row["n_convergence_polished"] == 2
    assert margin_row["polished_candidate_ids"] == [
        "seed=6|randomness=0.02",
        "seed=2|randomness=0.001",
    ]

    wide_gap_rows = [
        {"quality": 200.0, "max_doc_weight_ratio": 1.0, "n_above_max_doc_weight": 0, "elapsed_sec": 1.0, "seed": 1, "randomness": 0.0},
        {"quality": 100.0, "max_doc_weight_ratio": 1.0, "n_above_max_doc_weight": 0, "elapsed_sec": 1.0, "seed": 2, "randomness": 0.0},
    ]
    assert module._margin_convergence_polish_k(wide_gap_rows) == 1


def test_analyze_branch_lookahead_writes_expected_outputs(tmp_path):
    module = _load_module()
    csv_path = tmp_path / "rows.csv"
    output_dir = tmp_path / "out"
    _write_synthetic_rows(csv_path)

    payload = module.analyze_branch_lookahead(
        input_csv=csv_path,
        output_dir=output_dir,
        top_k_values=(1, 3, 5),
    )

    for path in payload["paths"].values():
        assert Path(path).exists()

    with Path(payload["paths"]["rank_flip_by_candidate"]).open(encoding="utf-8") as fh:
        rank_rows = list(csv.DictReader(fh))
    assert "iter1_rank" in rank_rows[0]
    assert "convergence_quality" in rank_rows[0]

    with Path(payload["paths"]["policy_simulation"]).open(encoding="utf-8") as fh:
        policy_rows = list(csv.DictReader(fh))
    assert policy_rows
    assert {
        "policy_name",
        "selected_seed",
        "quality_gap_to_best10",
        "convergence_polish_k",
        "n_convergence_polished",
        "selected_before_polish_candidate_id",
    }.issubset(policy_rows[0])

    report = Path(payload["paths"]["report"]).read_text(encoding="utf-8")
    assert "Leiden Branch Lookahead Analysis" in report
    assert "Best v2 Policies" in report
    assert "Greedy Failure Diagnosis" in report
    assert "iter5_screen_all_top3" in report
