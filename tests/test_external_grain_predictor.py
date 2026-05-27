import csv
import importlib.util
import sys
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "evaluate_external_grain_predictor.py"
)
SPEC = importlib.util.spec_from_file_location(
    "evaluate_external_grain_predictor",
    MODULE_PATH,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_external(path, rows):
    _write_csv(
        path,
        [
            "rank",
            "cluster",
            "block_count",
            "doc_weight",
            "incident_directed_edges",
            "assigned_fraction",
            "best_group_delta_q",
            "best_group_fraction",
            "best_group_action",
            "positive_group_count",
            "recommended_for_split_repair",
        ],
        rows,
    )


def _write_split(path, rows):
    _write_csv(
        path,
        [
            "rank",
            "cluster",
            "gamma_multiplier",
            "probe_resolution",
            "doc_weight",
            "n_parts",
            "core_part_count",
            "singleton_weight",
            "split_delta_q_base",
            "repair_delta_q",
            "net_delta_q",
            "escaped_source_units",
            "escaped_source_weight",
            "restored_source_cluster",
        ],
        rows,
    )


def test_join_picks_best_split_repair_row_per_cluster(tmp_path):
    external_path = tmp_path / "external.csv"
    split_path = tmp_path / "split.csv"
    _write_external(
        external_path,
        [
            {
                "rank": 1,
                "cluster": 10,
                "block_count": 4,
                "doc_weight": 1600,
                "incident_directed_edges": 100,
                "assigned_fraction": 1.0,
                "best_group_delta_q": 0.2,
                "best_group_fraction": 0.1,
                "best_group_action": 1,
                "positive_group_count": 1,
                "recommended_for_split_repair": "true",
            },
            {
                "rank": 2,
                "cluster": 20,
                "block_count": 2,
                "doc_weight": 800,
                "incident_directed_edges": 50,
                "assigned_fraction": 1.0,
                "best_group_delta_q": -0.1,
                "best_group_fraction": 0.0,
                "best_group_action": 0,
                "positive_group_count": 0,
                "recommended_for_split_repair": "false",
            },
        ],
    )
    _write_split(
        split_path,
        [
            {
                "rank": 1,
                "cluster": 10,
                "gamma_multiplier": 1.05,
                "probe_resolution": 0.00105,
                "doc_weight": 1600,
                "n_parts": 2,
                "core_part_count": 1,
                "singleton_weight": 100,
                "split_delta_q_base": -5,
                "repair_delta_q": 6,
                "net_delta_q": 0.5,
                "escaped_source_units": 1,
                "escaped_source_weight": 100,
                "restored_source_cluster": "false",
            },
            {
                "rank": 2,
                "cluster": 10,
                "gamma_multiplier": 1.20,
                "probe_resolution": 0.0012,
                "doc_weight": 1600,
                "n_parts": 4,
                "core_part_count": 3,
                "singleton_weight": 80,
                "split_delta_q_base": -4,
                "repair_delta_q": 8,
                "net_delta_q": 2.5,
                "escaped_source_units": 1,
                "escaped_source_weight": 90,
                "restored_source_cluster": "false",
            },
            {
                "rank": 3,
                "cluster": 20,
                "gamma_multiplier": 1.10,
                "probe_resolution": 0.0011,
                "doc_weight": 800,
                "n_parts": 1,
                "core_part_count": 1,
                "singleton_weight": 0,
                "split_delta_q_base": 0,
                "repair_delta_q": 0,
                "net_delta_q": -0.1,
                "escaped_source_units": 0,
                "escaped_source_weight": 0,
                "restored_source_cluster": "true",
            },
        ],
    )

    records, summary = MODULE.join_sample_records(
        MODULE.SampleSpec(
            name="synthetic",
            external_grain_csv=external_path,
            split_repair_csv=split_path,
        )
    )

    by_cluster = {record["cluster"]: record for record in records}
    assert summary["n_labeled_rows"] == 2
    assert by_cluster[10]["best_net_delta_q"] == 2.5
    assert by_cluster[10]["best_gamma_multiplier"] == 1.2
    assert by_cluster[10]["strong_success"] is True
    assert by_cluster[10]["practical_success"] is True
    assert by_cluster[20]["strong_success"] is False


def test_confusion_matrix_counts_and_costs():
    records = [
        {"best_group_delta_q": 1.0, "strong_success": True, "incident_directed_edges": 10},
        {"best_group_delta_q": 2.0, "strong_success": False, "incident_directed_edges": 20},
        {"best_group_delta_q": 0.0, "strong_success": True, "incident_directed_edges": 30},
        {"best_group_delta_q": -1.0, "strong_success": False, "incident_directed_edges": 40},
    ]
    rule = MODULE.Rule(
        "external_positive",
        lambda record: record["best_group_delta_q"] > 0,
    )

    metrics = MODULE.confusion_for_rule(records, rule, "strong_success")

    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["tn"] == 1
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5
    assert metrics["selected_count"] == 2
    assert metrics["estimated_full_repair_cost_units"] == 100
    assert metrics["selected_full_repair_cost_units"] == 30
    assert metrics["estimated_saved_full_repair_cost_units"] == 70


def test_missing_split_repair_labels_require_cost_only_sample(tmp_path):
    external_path = tmp_path / "external.csv"
    split_path = tmp_path / "split.csv"
    _write_external(
        external_path,
        [
            {
                "rank": 1,
                "cluster": 1,
                "block_count": 1,
                "doc_weight": 100,
                "incident_directed_edges": 10,
                "assigned_fraction": 1.0,
                "best_group_delta_q": 0.1,
                "best_group_fraction": 0.1,
                "best_group_action": 1,
                "positive_group_count": 1,
                "recommended_for_split_repair": "true",
            },
            {
                "rank": 2,
                "cluster": 2,
                "block_count": 1,
                "doc_weight": 200,
                "incident_directed_edges": 20,
                "assigned_fraction": 1.0,
                "best_group_delta_q": 0.0,
                "best_group_fraction": 0.0,
                "best_group_action": 0,
                "positive_group_count": 0,
                "recommended_for_split_repair": "false",
            },
        ],
    )
    _write_split(
        split_path,
        [
            {
                "rank": 1,
                "cluster": 1,
                "gamma_multiplier": 1.05,
                "probe_resolution": 0.00105,
                "doc_weight": 100,
                "n_parts": 2,
                "core_part_count": 1,
                "singleton_weight": 0,
                "split_delta_q_base": 0,
                "repair_delta_q": 0,
                "net_delta_q": 1.5,
                "escaped_source_units": 1,
                "escaped_source_weight": 50,
                "restored_source_cluster": "false",
            },
        ],
    )

    with pytest.raises(ValueError, match="without split-repair labels"):
        MODULE.join_sample_records(
            MODULE.SampleSpec(
                name="partial",
                external_grain_csv=external_path,
                split_repair_csv=split_path,
            )
        )

    records, summary = MODULE.join_sample_records(
        MODULE.SampleSpec(
            name="cost-only",
            external_grain_csv=external_path,
            cost_only=True,
        )
    )
    assert summary["n_missing_labels"] == 2
    assert all(record["strong_success"] is None for record in records)


def test_acceptance_and_coverage_summaries(tmp_path):
    external_path = tmp_path / "external.csv"
    split_path = tmp_path / "split.csv"
    external_path.write_text("cluster\n1\n", encoding="utf-8")
    split_path.write_text("cluster\n1\n", encoding="utf-8")

    records = [
        {
            "sample": "giant",
            "regime": "giant",
            "best_group_delta_q": 1.0,
            "best_group_fraction": 0.01,
            "assigned_fraction": 1.0,
            "doc_weight": 2000.0,
            "incident_directed_edges": 10,
            "strong_success": True,
            "practical_success": True,
        },
        {
            "sample": "band",
            "regime": "band",
            "best_group_delta_q": 0.0,
            "best_group_fraction": 0.0,
            "assigned_fraction": 1.0,
            "doc_weight": 800.0,
            "incident_directed_edges": 20,
            "strong_success": False,
            "practical_success": False,
        },
    ]
    rules = MODULE.build_rules()
    per_sample = []
    for sample_name, sample_records, regime, ratio in [
        ("giant", [records[0]], "giant", 12.0),
        ("band", [records[1]], "band", 15.0),
    ]:
        metrics, detail = MODULE.evaluate_records(
            sample_records,
            rules,
            {
                "sample": sample_name,
                "regime": regime,
                "gamma": None,
                "external_probe_elapsed_sec": 1.0,
                "split_repair_probe_elapsed_sec": ratio,
                "split_to_external_elapsed_ratio": ratio,
            },
        )
        assert metrics
        per_sample.append(detail)
    global_metrics, global_detail = MODULE.evaluate_records(
        records,
        rules,
        {"sample": "__global__", "regime": "", "gamma": None},
    )
    result = {
        "per_sample": per_sample,
        "global_weighted_summary": global_detail,
    }

    acceptance = MODULE.summarize_acceptance(result, records)
    assert acceptance["external_grain_predictor_useful"]["pass"] is True
    assert acceptance["cascade_ready_for_apply_mode_design"]["pass"] is True

    coverage = MODULE.summarize_coverage(
        [
            MODULE.CoverageSpec(
                name="matched",
                external_grain_csv=external_path,
                split_repair_csv=split_path,
            ),
            MODULE.CoverageSpec(name="missing", split_repair_csv=split_path),
        ]
    )
    assert coverage["counts"] == {"matched_labeled": 1, "missing_external_grain": 1}
