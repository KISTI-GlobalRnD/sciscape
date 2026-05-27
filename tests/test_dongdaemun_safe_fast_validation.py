"""Tests for the Dongdaemun safe-fast validation script helpers."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "research"
    / "consensus"
    / "scripts"
    / "run_dongdaemun_safe_fast_validation.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "run_dongdaemun_safe_fast_validation_for_test",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_prepare_pressure_summary_reads_prepare_fields(tmp_path):
    module = _load_module()
    summary_path = tmp_path / "prepare_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "post_max_cluster_size": 1040.0,
                "post_n_clusters_gt_target_max": 2,
            }
        ),
        encoding="utf-8",
    )

    pressure = module._prepare_pressure_summary(
        summary_path,
        target_max_doc_weight=1000.0,
        trigger_max_doc_weight_ratio=1.03,
        trigger_min_above_max_doc_weight=2,
    )

    assert pressure["prepare_max_doc_weight"] == 1040.0
    assert pressure["prepare_max_doc_weight_ratio"] == 1.04
    assert pressure["prepare_n_above_max_doc_weight"] == 2
    assert pressure["prepare_triggered"] is True


def test_prepare_pressure_summary_reads_oversize_fields(tmp_path):
    module = _load_module()
    summary_path = tmp_path / "postprocess_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "oversize_summary": {
                    "after": {
                        "max_doc_weight": 1010.0,
                        "n_above_max_doc_weight": 1,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    pressure = module._prepare_pressure_summary(
        summary_path,
        target_max_doc_weight=1000.0,
        trigger_max_doc_weight_ratio=1.03,
        trigger_min_above_max_doc_weight=2,
    )

    assert pressure["prepare_max_doc_weight"] == 1010.0
    assert pressure["prepare_max_doc_weight_ratio"] == 1.01
    assert pressure["prepare_n_above_max_doc_weight"] == 1
    assert pressure["prepare_triggered"] is False


def test_aggregate_rows_counts_direct_outcomes():
    module = _load_module()
    rows = [
        {
            "supported": True,
            "skipped_by_prepare": False,
            "triggered": True,
            "fallback_triggered": False,
            "selected_variant": "refine_repair_off",
            "fallback_reason": "",
            "quality_delta_vs_standard": 3.0,
            "quality_recompute_abs_delta": 0.0,
            "quality_recompute_ok": True,
            "elapsed_sec": 2.0,
        },
        {
            "supported": True,
            "skipped_by_prepare": False,
            "triggered": False,
            "fallback_triggered": True,
            "selected_variant": "standard",
            "fallback_reason": "trigger_not_met",
            "quality_delta_vs_standard": 0.0,
            "quality_recompute_abs_delta": 0.0,
            "quality_recompute_ok": True,
            "elapsed_sec": 1.0,
        },
        {
            "supported": True,
            "skipped_by_prepare": True,
            "quality_delta_vs_standard": None,
        },
    ]

    aggregate = module._aggregate_rows(rows)

    assert aggregate["n_rows"] == 3
    assert aggregate["n_direct_rows"] == 2
    assert aggregate["n_skipped_by_prepare"] == 1
    assert aggregate["n_triggered"] == 1
    assert aggregate["n_fallback"] == 1
    assert aggregate["n_improved_vs_standard"] == 1
    assert aggregate["best_quality_delta_vs_standard"] == 3.0
    assert aggregate["quality_recompute_all_ok"] is True
    assert aggregate["selected_variant_counts"] == {
        "refine_repair_off": 1,
        "standard": 1,
    }
