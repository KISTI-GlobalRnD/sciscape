"""Tests for the Slice 4 quality-first sweep helpers."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "research"
    / "consensus"
    / "scripts"
    / "run_dongdaemun_refinement_slice4_quality_sweep.py"
)


def _load_module():
    script_dir = SCRIPT_PATH.parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    spec = importlib.util.spec_from_file_location(
        "run_dongdaemun_refinement_slice4_quality_sweep_for_test",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_quality_summary_module():
    script_path = (
        SCRIPT_PATH.parent / "summarize_dongdaemun_quality_trace.py"
    )
    spec = importlib.util.spec_from_file_location(
        "summarize_dongdaemun_quality_trace_for_test",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_candidate_summary_module():
    script_path = SCRIPT_PATH.parent / "summarize_dongdaemun_candidate_trace.py"
    spec = importlib.util.spec_from_file_location(
        "summarize_dongdaemun_candidate_trace_for_test",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_sweep_configs_crosses_presets_and_perturbations():
    module = _load_module()

    configs = module._build_sweep_configs(
        gamma_presets=("mild", "aggressive"),
        seed_perturbations=(0, 2),
    )

    assert [config.config_id for config in configs] == [
        "mild_sp0",
        "mild_sp2",
        "aggressive_sp0",
        "aggressive_sp2",
    ]
    assert configs[0].gamma_multipliers == (1.02, 1.05)
    assert configs[-1].gamma_multipliers == (1.10, 1.25, 1.50, 2.00)


def test_build_sweep_configs_accepts_quality_guarded_structural_policy():
    module = _load_module()

    configs = module._build_sweep_configs(
        gamma_presets=("mild",),
        seed_perturbations=(0,),
        candidate_quality_policies=("quality_guarded_structural",),
        min_candidate_delta_q=-1.0e-6,
    )

    assert [config.config_id for config in configs] == [
        "mild_sp0_quality_guarded_structural"
    ]
    assert configs[0].candidate_quality_policy == "quality_guarded_structural"
    assert configs[0].min_candidate_delta_q == -1.0e-6


def test_build_sweep_configs_accepts_selective_policy():
    module = _load_module()

    configs = module._build_sweep_configs(
        gamma_presets=("current",),
        seed_perturbations=(0,),
        candidate_quality_policies=("selective",),
        use_final_quality_guard=True,
    )

    assert [config.config_id for config in configs] == [
        "current_sp0_selective_final_guard"
    ]
    assert configs[0].candidate_quality_policy == "selective"
    assert configs[0].use_final_quality_guard is True


def test_build_sweep_configs_accepts_pressure_aware_policy():
    module = _load_module()

    configs = module._build_sweep_configs(
        gamma_presets=("current",),
        seed_perturbations=(0,),
        candidate_quality_policies=("pressure_aware",),
        min_candidate_delta_q=-10.0,
    )

    assert [config.config_id for config in configs] == ["current_sp0_pressure_aware"]
    assert configs[0].candidate_quality_policy == "pressure_aware"
    assert configs[0].min_candidate_delta_q == -10.0


def test_build_sweep_configs_accepts_adaptive_plateau_bands():
    module = _load_module()

    configs = module._build_sweep_configs(
        gamma_presets=("current",),
        seed_perturbations=(0,),
        candidate_quality_policies=("adaptive_plateau",),
        min_candidate_delta_q=-5.0,
        adaptive_plateau_quality_bands=(0.0, 1.0, 5.0),
    )

    assert [config.config_id for config in configs] == [
        "current_sp0_adaptive_plateau",
        "current_sp0_adaptive_plateau_band1",
        "current_sp0_adaptive_plateau_band5",
    ]
    assert [config.adaptive_plateau_quality_band for config in configs] == [
        0.0,
        1.0,
        5.0,
    ]
    assert all(config.min_candidate_delta_q == -5.0 for config in configs)


def test_build_sweep_configs_appends_final_guard_suffix():
    module = _load_module()

    configs = module._build_sweep_configs(
        gamma_presets=("current",),
        seed_perturbations=(1,),
        use_final_quality_guard=True,
        min_final_quality_delta=0.0,
    )

    assert [config.config_id for config in configs] == ["current_sp1_final_guard"]
    assert configs[0].use_final_quality_guard is True
    assert configs[0].min_final_quality_delta == 0.0


def test_build_sweep_configs_appends_repair_augment_suffix():
    module = _load_module()

    configs = module._build_sweep_configs(
        gamma_presets=("current",),
        seed_perturbations=(2,),
        baseline_repair_policies=("augment",),
        use_final_quality_guard=True,
    )

    assert [config.config_id for config in configs] == [
        "current_sp2_repair_augment_final_guard"
    ]
    assert configs[0].baseline_repair_policy == "augment"


def test_build_sweep_configs_appends_repair_adaptive_suffix_and_ratio():
    module = _load_module()

    configs = module._build_sweep_configs(
        gamma_presets=("current",),
        seed_perturbations=(2,),
        baseline_repair_policies=("adaptive",),
        baseline_repair_replace_min_parent_ratio=1.07,
    )

    assert [config.config_id for config in configs] == [
        "current_sp2_repair_adaptive"
    ]
    assert configs[0].baseline_repair_policy == "adaptive"
    assert configs[0].baseline_repair_replace_min_parent_ratio == 1.07


def test_build_sweep_configs_appends_runtime_budget_suffixes():
    module = _load_module()

    configs = module._build_sweep_configs(
        gamma_presets=("mild",),
        seed_perturbations=(1,),
        max_extra_parents_per_iteration=4,
        max_extra_children_per_parent=32,
    )

    assert [config.config_id for config in configs] == ["mild_sp1_p4_c32"]
    assert configs[0].max_extra_parents_per_iteration == 4
    assert configs[0].max_extra_children_per_parent == 32


def test_build_sweep_configs_appends_parent_selection_suffix():
    module = _load_module()

    configs = module._build_sweep_configs(
        gamma_presets=("mild",),
        seed_perturbations=(0,),
        parent_selection_policies=("pressure_boundary",),
    )

    assert [config.config_id for config in configs] == [
        "mild_sp0_parent_pressure_boundary"
    ]
    assert configs[0].parent_selection_policy == "pressure_boundary"


def test_build_sweep_configs_appends_auto_fast_suffix_and_thresholds():
    module = _load_module()

    configs = module._build_sweep_configs(
        gamma_presets=("mild",),
        seed_perturbations=(0,),
        max_extra_parents_per_iteration=4,
        max_extra_children_per_parent=16,
        auto_fast_trigger_max_doc_weight_ratio=1.03,
        auto_fast_trigger_min_above_max_doc_weight=2,
        auto_fast_accept_max_doc_weight_ratio=1.01,
    )

    assert [config.config_id for config in configs] == ["mild_sp0_p4_c16_auto_fast"]
    assert configs[0].auto_fast_trigger_max_doc_weight_ratio == 1.03
    assert configs[0].auto_fast_trigger_min_above_max_doc_weight == 2
    assert configs[0].auto_fast_accept_max_doc_weight_ratio == 1.01


def test_build_sweep_configs_appends_auto_fast_quality_accept_suffix():
    module = _load_module()

    configs = module._build_sweep_configs(
        gamma_presets=("mild",),
        seed_perturbations=(0,),
        auto_fast_accept_min_quality_delta=0.0,
    )

    assert [config.config_id for config in configs] == [
        "mild_sp0_auto_fast_quality_accept"
    ]
    assert configs[0].auto_fast_accept_min_quality_delta == 0.0


def test_build_sweep_configs_appends_auto_fast_quality_accept_ratio_suffix():
    module = _load_module()

    configs = module._build_sweep_configs(
        gamma_presets=("mild",),
        seed_perturbations=(0,),
        auto_fast_accept_min_quality_delta_ratio=1.0e-5,
    )

    assert [config.config_id for config in configs] == [
        "mild_sp0_auto_fast_quality_accept_ratio"
    ]
    assert configs[0].auto_fast_accept_min_quality_delta_ratio == 1.0e-5


def test_auto_fast_trigger_uses_any_pressure_signal():
    module = _load_module()
    config = module.SweepConfig(
        config_id="mild_sp0_p4_c16_auto_fast",
        gamma_preset="mild",
        gamma_multipliers=(1.02, 1.05),
        seed_perturbations=0,
        auto_fast_trigger_max_doc_weight_ratio=1.03,
        auto_fast_trigger_min_above_max_doc_weight=2,
    )

    assert (
        module._auto_fast_should_run(
            {"max_doc_weight_ratio": 1.02, "n_above_max_doc_weight": 1},
            config,
        )
        is False
    )
    assert (
        module._auto_fast_should_run(
            {"max_doc_weight_ratio": 1.04, "n_above_max_doc_weight": 1},
            config,
        )
        is True
    )
    assert (
        module._auto_fast_should_run(
            {"max_doc_weight_ratio": 1.02, "n_above_max_doc_weight": 2},
            config,
        )
        is True
    )


def test_auto_fast_acceptance_falls_back_to_standard_on_max_weight_guard():
    module = _load_module()
    pilot = module.pilot
    config = module.SweepConfig(
        config_id="mild_sp0_p4_c16_auto_fast",
        gamma_preset="mild",
        gamma_multipliers=(1.02, 1.05),
        seed_perturbations=0,
        auto_fast_accept_max_doc_weight_ratio=1.01,
    )
    standard_row = {field: None for field in pilot.CSV_FIELDS}
    standard_row.update(
        {
            "sample": "field30",
            "variant": pilot.VARIANT_STANDARD,
            "supported": True,
            "unsupported_reason": "",
            "elapsed_sec": 10.0,
            "n_clusters": 3,
            "quality": 100.0,
            "quality_delta_vs_standard": 0.0,
            "max_doc_weight": 100.0,
            "max_doc_weight_ratio": 1.0,
            "n_above_max_doc_weight": 1,
        }
    )
    candidate_row = dict(standard_row)
    candidate_row.update(
        {
            "variant": pilot.VARIANT_REPAIR_OFF,
            "elapsed_sec": 12.0,
            "quality": 110.0,
            "quality_delta_vs_standard": 10.0,
            "quality_improved_vs_standard": True,
            "max_doc_weight": 102.0,
        }
    )
    standard_membership = np.asarray([0, 1, 1], dtype=np.uint64)
    candidate_membership = np.asarray([0, 0, 1], dtype=np.uint64)

    row, membership = module._apply_auto_fast_acceptance(
        row=candidate_row,
        membership=candidate_membership,
        standard_row=standard_row,
        standard_membership=standard_membership,
        sweep_config=config,
        variant=pilot.VARIANT_REPAIR_OFF,
        use_baseline_repair=False,
    )

    assert np.array_equal(membership, standard_membership)
    assert row["quality"] == 100.0
    assert row["quality_delta_vs_standard"] == 0.0
    assert row["auto_fast_triggered"] is True
    assert row["auto_fast_fallback_triggered"] is True
    assert row["auto_fast_fallback_reason"] == "max_doc_weight_guard"


def test_auto_fast_acceptance_falls_back_to_standard_on_quality_guard():
    module = _load_module()
    pilot = module.pilot
    config = module.SweepConfig(
        config_id="mild_sp0_auto_fast_quality_accept",
        gamma_preset="mild",
        gamma_multipliers=(1.02, 1.05),
        seed_perturbations=0,
        auto_fast_accept_min_quality_delta=0.0,
    )
    standard_row = {field: None for field in pilot.CSV_FIELDS}
    standard_row.update(
        {
            "sample": "field30",
            "variant": pilot.VARIANT_STANDARD,
            "supported": True,
            "unsupported_reason": "",
            "elapsed_sec": 10.0,
            "n_clusters": 3,
            "quality": 100.0,
            "quality_delta_vs_standard": 0.0,
            "max_doc_weight": 100.0,
            "max_doc_weight_ratio": 1.0,
            "n_above_max_doc_weight": 1,
        }
    )
    candidate_row = dict(standard_row)
    candidate_row.update(
        {
            "variant": pilot.VARIANT_REPAIR_OFF,
            "elapsed_sec": 12.0,
            "quality": 99.9,
            "quality_delta_vs_standard": -0.1,
            "quality_improved_vs_standard": False,
            "max_doc_weight": 99.0,
        }
    )
    standard_membership = np.asarray([0, 1, 1], dtype=np.uint64)
    candidate_membership = np.asarray([0, 0, 1], dtype=np.uint64)

    row, membership = module._apply_auto_fast_acceptance(
        row=candidate_row,
        membership=candidate_membership,
        standard_row=standard_row,
        standard_membership=standard_membership,
        sweep_config=config,
        variant=pilot.VARIANT_REPAIR_OFF,
        use_baseline_repair=False,
    )

    assert np.array_equal(membership, standard_membership)
    assert row["quality"] == 100.0
    assert row["quality_delta_vs_standard"] == 0.0
    assert row["auto_fast_triggered"] is True
    assert row["auto_fast_fallback_triggered"] is True
    assert row["auto_fast_fallback_reason"] == "quality_guard"


def test_auto_fast_acceptance_uses_normalized_quality_guard():
    module = _load_module()
    pilot = module.pilot
    config = module.SweepConfig(
        config_id="mild_sp0_auto_fast_quality_accept_ratio",
        gamma_preset="mild",
        gamma_multipliers=(1.02, 1.05),
        seed_perturbations=0,
        auto_fast_accept_min_quality_delta_ratio=0.01,
    )
    standard_row = {field: None for field in pilot.CSV_FIELDS}
    standard_row.update(
        {
            "sample": "field30",
            "variant": pilot.VARIANT_STANDARD,
            "supported": True,
            "unsupported_reason": "",
            "elapsed_sec": 10.0,
            "n_clusters": 3,
            "quality": 100.0,
            "quality_delta_vs_standard": 0.0,
            "max_doc_weight": 100.0,
            "max_doc_weight_ratio": 1.0,
            "n_above_max_doc_weight": 1,
        }
    )
    candidate_row = dict(standard_row)
    candidate_row.update(
        {
            "variant": pilot.VARIANT_REPAIR_OFF,
            "elapsed_sec": 12.0,
            "quality": 100.5,
            "quality_delta_vs_standard": 0.5,
            "quality_improved_vs_standard": True,
            "max_doc_weight": 99.0,
        }
    )
    standard_membership = np.asarray([0, 1, 1], dtype=np.uint64)
    candidate_membership = np.asarray([0, 0, 1], dtype=np.uint64)

    row, membership = module._apply_auto_fast_acceptance(
        row=candidate_row,
        membership=candidate_membership,
        standard_row=standard_row,
        standard_membership=standard_membership,
        sweep_config=config,
        variant=pilot.VARIANT_REPAIR_OFF,
        use_baseline_repair=False,
    )

    assert np.array_equal(membership, standard_membership)
    assert row["quality"] == 100.0
    assert row["auto_fast_fallback_reason"] == "quality_guard"


def test_recompute_quality_records_delta_and_ok_status():
    module = _load_module()

    class FakeGraph:
        def cpm_quality(self, membership, *, resolution):
            assert resolution == 0.01
            return float(np.asarray(membership).sum()) + 0.5

    row = module._recompute_quality(
        FakeGraph(),
        np.asarray([0, 1, 1], dtype=np.uint64),
        0.01,
        2.5,
    )

    assert row["quality_recomputed"] == 2.5
    assert row["quality_recompute_delta"] == 0.0
    assert row["quality_recompute_abs_delta"] == 0.0
    assert row["quality_recompute_ok"] is True


def test_aggregate_rows_tracks_best_quality_and_repair_gain():
    module = _load_module()
    pilot = module.pilot
    rows = [
        {
            "variant": pilot.VARIANT_STANDARD,
            "quality_recompute_ok": True,
            "quality_recompute_abs_delta": 0.0,
        },
        {
            "variant": pilot.VARIANT_REPAIR_OFF,
            "config_id": "mild_sp0",
            "gamma_preset": "mild",
            "seed_perturbations": 0,
            "parent_selection_policy": "pressure_boundary",
            "quality_delta_vs_standard": -1.0,
            "quality_improved_vs_standard": False,
            "baseline_repair_candidates_total": 0,
            "baseline_repair_improved_candidates_total": 0,
            "baseline_repair_selected_total": 0,
            "baseline_repair_merge_count_total": 0,
            "baseline_repair_delta_sum": 0.0,
            "quality_recompute_ok": True,
            "quality_recompute_abs_delta": 0.0,
        },
        {
            "variant": pilot.VARIANT_REPAIR_ON,
            "config_id": "mild_sp0",
            "gamma_preset": "mild",
            "seed_perturbations": 0,
            "parent_selection_policy": "pressure_boundary",
            "quality_delta_vs_standard": 0.25,
            "quality_delta_vs_repair_off": 1.25,
            "quality_improved_vs_standard": True,
            "quality_improved_vs_repair_off": True,
            "baseline_repair_candidates_total": 3,
            "quality_recompute_ok": True,
            "quality_recompute_abs_delta": 1.0e-12,
        },
    ]

    aggregate = module._aggregate_rows(rows)

    assert aggregate["n_refinement_improved_vs_standard"] == 1
    assert aggregate["n_repair_on_improved_vs_repair_off"] == 1
    assert aggregate["quality_recompute_all_ok"] is True
    assert aggregate["best_refinement_row"]["variant"] == pilot.VARIANT_REPAIR_ON
    assert aggregate["best_repair_gain_row"]["quality_delta_vs_repair_off"] == 1.25
    assert aggregate["by_config"][0]["n_improved_vs_standard"] == 1
    assert aggregate["by_config"][0]["parent_selection_policy"] == "pressure_boundary"


def test_aggregate_rows_preserves_zero_quality_delta_as_best_value():
    module = _load_module()
    pilot = module.pilot
    rows = [
        {
            "variant": pilot.VARIANT_STANDARD,
            "quality_recompute_ok": True,
            "quality_recompute_abs_delta": 0.0,
        },
        {
            "variant": pilot.VARIANT_REPAIR_OFF,
            "config_id": "guarded",
            "gamma_preset": "mild",
            "seed_perturbations": 0,
            "candidate_quality_policy": "quality_guarded_structural",
            "min_candidate_delta_q": 0.0,
            "quality_delta_vs_standard": 0.0,
            "quality_delta_vs_repair_off": 0.0,
            "quality_improved_vs_standard": False,
            "quality_improved_vs_repair_off": False,
            "baseline_repair_candidates_total": 0,
            "baseline_repair_improved_candidates_total": 0,
            "baseline_repair_selected_total": 0,
            "baseline_repair_merge_count_total": 0,
            "baseline_repair_delta_sum": 0.0,
            "quality_recompute_ok": True,
            "quality_recompute_abs_delta": 0.0,
        },
    ]

    aggregate = module._aggregate_rows(rows)

    assert aggregate["best_refinement_row"]["quality_delta_vs_standard"] == 0.0
    assert aggregate["by_config"][0]["best_quality_delta_vs_standard"] == 0.0


def test_checkpoint_rows_round_trip_deduplicates_latest(tmp_path):
    module = _load_module()
    pilot = module.pilot
    path = tmp_path / module.CHECKPOINT_ROWS_FILENAME
    first = {
        "summary_path": "field30/seed42/prepare_summary.json",
        "seed": 42,
        "config_id": "current_sp1_final_guard",
        "variant": pilot.VARIANT_REPAIR_ON,
        "use_baseline_repair": True,
        "quality": 1.0,
    }
    latest = dict(first, quality=2.0)

    module._append_checkpoint_rows(path, [first, latest])
    rows_by_key = module._load_checkpoint_rows(path)

    assert len(rows_by_key) == 1
    assert next(iter(rows_by_key.values()))["quality"] == 2.0


def test_ordered_rows_from_checkpoint_keeps_planned_order():
    module = _load_module()
    pilot = module.pilot
    summary_path = Path("field30/seed42/prepare_summary.json")
    config = module.SweepConfig(
        config_id="current_sp1_final_guard",
        gamma_preset="current",
        gamma_multipliers=(1.02,),
        seed_perturbations=1,
        use_final_quality_guard=True,
    )
    input_cfg = pilot.Slice4Input(
        sample="field30",
        graph_dir=Path("graph"),
        membership_path=Path("membership.parquet"),
        node_weights_path=Path("node_weights.f64.bin"),
        resolution=0.01,
        target_max_doc_weight=100.0,
        seed=42,
        summary_path=summary_path,
    )
    original_resolver = pilot._resolve_input_from_summary
    pilot._resolve_input_from_summary = lambda path: input_cfg
    try:
        rows = [
            {
                "summary_path": str(summary_path),
                "seed": 42,
                "config_id": config.config_id,
                "variant": pilot.VARIANT_REPAIR_ON,
                "use_baseline_repair": True,
            },
            {
                "summary_path": str(summary_path),
                "seed": 42,
                "config_id": "standard",
                "variant": pilot.VARIANT_STANDARD,
                "use_baseline_repair": None,
            },
            {
                "summary_path": str(summary_path),
                "seed": 42,
                "config_id": config.config_id,
                "variant": pilot.VARIANT_REPAIR_OFF,
                "use_baseline_repair": False,
            },
        ]
        rows_by_key = {module._row_key(row): row for row in rows}

        ordered = module._ordered_rows_from_checkpoint(
            summary_paths=[summary_path],
            sweep_configs=[config],
            rows_by_key=rows_by_key,
        )
    finally:
        pilot._resolve_input_from_summary = original_resolver

    assert [row["variant"] for row in ordered] == [
        pilot.VARIANT_STANDARD,
        pilot.VARIANT_REPAIR_OFF,
        pilot.VARIANT_REPAIR_ON,
    ]


def test_candidate_trace_run_metadata_is_joinable_to_sweep_rows():
    module = _load_module()
    pilot = module.pilot
    input_cfg = pilot.Slice4Input(
        sample="field30",
        graph_dir=Path("graph"),
        membership_path=Path("membership.parquet"),
        node_weights_path=Path("node_weights.f64.bin"),
        resolution=0.01,
        target_max_doc_weight=100.0,
        seed=42,
        summary_path=Path("field30/seed42/prepare_summary.json"),
    )
    config = module.SweepConfig(
        config_id="mild_sp1_p4_c16",
        gamma_preset="mild",
        gamma_multipliers=(1.02, 1.05),
        seed_perturbations=1,
        max_extra_parents_per_iteration=4,
        max_extra_children_per_parent=16,
    )
    row_key = module._input_row_key(
        input_cfg=input_cfg,
        config_id=config.config_id,
        variant=pilot.VARIANT_REPAIR_ON,
        use_baseline_repair=True,
    )
    run_id = module._candidate_trace_run_id(row_key)

    metadata = module._candidate_trace_run_metadata(
        input_cfg=input_cfg,
        sweep_config=config,
        variant=pilot.VARIANT_REPAIR_ON,
        use_baseline_repair=True,
        run_id=run_id,
    )
    quality_metadata = module._quality_trace_run_metadata(
        input_cfg=input_cfg,
        sweep_config=config,
        variant=pilot.VARIANT_REPAIR_ON,
        use_baseline_repair=True,
        run_id=run_id,
    )
    row = module._add_sweep_fields(
        row={
            "variant": pilot.VARIANT_REPAIR_ON,
            "candidate_trace_run_id": run_id,
            "quality_trace_run_id": run_id,
        },
        input_cfg=input_cfg,
        sweep_config=config,
        use_baseline_repair=True,
        quality_check={},
    )

    assert metadata["run_id"] == run_id
    assert metadata["row_key"] == row_key
    assert metadata["candidate_quality_policy"] == "structural"
    assert metadata["adaptive_plateau_quality_band"] == 0.0
    assert quality_metadata["schema"] == "dongdaemun_refinement_quality_trace_run.v1"
    assert quality_metadata["run_id"] == run_id
    assert quality_metadata["row_key"] == row_key
    assert row["candidate_trace_run_id"] == run_id
    assert row["quality_trace_run_id"] == run_id
    assert row["adaptive_plateau_quality_band"] == 0.0


def test_candidate_trace_context_sets_and_restores_run_id(monkeypatch):
    module = _load_module()
    monkeypatch.setenv(module.CANDIDATE_TRACE_RUN_ID_ENV, "outer")

    with module._candidate_trace_context("inner"):
        assert module.os.environ[module.CANDIDATE_TRACE_RUN_ID_ENV] == "inner"

    assert module.os.environ[module.CANDIDATE_TRACE_RUN_ID_ENV] == "outer"


def test_candidate_trace_path_context_sets_epoch_and_restores_env(tmp_path, monkeypatch):
    module = _load_module()
    trace_path = tmp_path / "candidate_trace.jsonl"
    trace_path.write_text("old\n", encoding="utf-8")
    monkeypatch.setenv(module.CANDIDATE_TRACE_PATH_ENV, "outer-path")
    monkeypatch.setenv(module.CANDIDATE_TRACE_EPOCH_ENV, "outer-epoch")

    with module._candidate_trace_path_context(
        trace_path,
        explicit=True,
        resume=False,
    ):
        assert module.os.environ[module.CANDIDATE_TRACE_PATH_ENV] == str(trace_path)
        assert module.os.environ[module.CANDIDATE_TRACE_EPOCH_ENV] != "outer-epoch"
        assert not trace_path.exists()

    assert module.os.environ[module.CANDIDATE_TRACE_PATH_ENV] == "outer-path"
    assert module.os.environ[module.CANDIDATE_TRACE_EPOCH_ENV] == "outer-epoch"


def test_quality_trace_context_sets_and_restores_run_id(monkeypatch):
    module = _load_module()
    monkeypatch.setenv(module.QUALITY_TRACE_RUN_ID_ENV, "outer")

    with module._quality_trace_context("inner"):
        assert module.os.environ[module.QUALITY_TRACE_RUN_ID_ENV] == "inner"

    assert module.os.environ[module.QUALITY_TRACE_RUN_ID_ENV] == "outer"


def test_quality_trace_path_context_sets_epoch_and_restores_env(tmp_path, monkeypatch):
    module = _load_module()
    trace_path = tmp_path / "quality_trace.jsonl"
    trace_path.write_text("old\n", encoding="utf-8")
    monkeypatch.setenv(module.QUALITY_TRACE_PATH_ENV, "outer-path")
    monkeypatch.setenv(module.QUALITY_TRACE_EPOCH_ENV, "outer-epoch")

    with module._quality_trace_path_context(
        trace_path,
        explicit=True,
        resume=False,
    ):
        assert module.os.environ[module.QUALITY_TRACE_PATH_ENV] == str(trace_path)
        assert module.os.environ[module.QUALITY_TRACE_EPOCH_ENV] != "outer-epoch"
        assert not trace_path.exists()

    assert module.os.environ[module.QUALITY_TRACE_PATH_ENV] == "outer-path"
    assert module.os.environ[module.QUALITY_TRACE_EPOCH_ENV] == "outer-epoch"


def test_candidate_trace_summarizer_records_adaptive_diagnostics(tmp_path):
    module = _load_candidate_summary_module()
    trace_path = tmp_path / "candidate_trace.jsonl"
    runs_path = tmp_path / "candidate_trace_runs.jsonl"
    run_id = "run-1"
    events = [
        {
            "event": "candidate_profile",
            "run_id": run_id,
            "depth": 0,
            "parent_id": 3,
            "candidate_id": 7,
            "source": "high_gamma",
            "quadrant": "qpos_spos",
            "decision": "selected_by_policy",
            "candidate_delta_q": 0.1,
            "largest_child_fraction_improvement": 0.4,
            "largest_child_fraction": 0.6,
            "standard_max_child_weight_ratio": 1.4,
            "candidate_max_child_weight_ratio": 1.0,
            "pressure_reduction": 0.4,
            "singleton_weight_fraction": 0.0,
            "quotient_score": 0.2,
            "adaptive_diagnostic_score": 1.0,
            "baseline_repair_delta_sum": 0.0,
            "valid": True,
            "quality_passes": True,
        },
        {
            "event": "candidate_decision",
            "run_id": run_id,
            "depth": 0,
            "parent_id": 3,
            "candidate_id": 7,
            "decision": "selected_applied",
        },
    ]
    trace_path.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    runs_path.write_text(
        json.dumps(
            {
                "schema": "dongdaemun_refinement_candidate_trace_run.v1",
                "run_id": run_id,
                "sample": "field30",
                "variant": "refine_repair_off",
                "use_baseline_repair": False,
                "config_id": "current_sp0_adaptive_plateau_band1",
                "gamma_preset": "current",
                "seed_perturbations": 0,
                "parent_selection_policy": "weight",
                "candidate_quality_policy": "adaptive_plateau",
                "adaptive_plateau_quality_band": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    payload = module.summarize_candidate_trace(
        trace_path=trace_path,
        runs_path=runs_path,
        output_dir=tmp_path / "summary",
    )

    assert payload["n_runs"] == 1
    assert payload["overall"]["adaptive_diagnostic_score_mean"] == 1.0
    assert payload["overall"]["selected_applied_adaptive_diagnostic_score_mean"] == 1.0
    with Path(payload["paths"]["by_run"]).open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0]["candidate_quality_policy"] == "adaptive_plateau"
    assert rows[0]["adaptive_plateau_quality_band"] == "1.0"
    assert rows[0]["selected_applied_profiles"] == "1"


def test_quality_trace_summarizer_writes_csv_and_png_outputs(tmp_path):
    module = _load_quality_summary_module()
    trace_path = tmp_path / "quality_trace.jsonl"
    runs_path = tmp_path / "quality_trace_runs.jsonl"
    run_id = "run-1"
    events = [
        {
            "event": "quality_checkpoint",
            "run_id": run_id,
            "checkpoint_index": 0,
            "phase": "start",
            "iteration": 0,
            "quality": 10.0,
            "quality_delta_vs_start": 0.0,
            "elapsed_ms_since_run_start": 0.0,
            "iteration_elapsed_ms": 0.0,
            "n_clusters": 2,
            "max_doc_weight": 7.0,
            "max_doc_weight_ratio": 1.4,
            "n_above_max_doc_weight": 1,
            "moved_nodes": 0,
            "selected_parent_count_total": 0,
            "applied_parent_count_total": 0,
        },
        {
            "event": "quality_checkpoint",
            "run_id": run_id,
            "checkpoint_index": 1,
            "phase": "after_iteration",
            "iteration": 1,
            "quality": 11.5,
            "quality_delta_vs_start": 1.5,
            "elapsed_ms_since_run_start": 100.0,
            "iteration_elapsed_ms": 100.0,
            "n_clusters": 3,
            "max_doc_weight": 5.0,
            "max_doc_weight_ratio": 1.0,
            "n_above_max_doc_weight": 0,
            "moved_nodes": 2,
            "selected_parent_count_total": 1,
            "applied_parent_count_total": 1,
        },
        {
            "event": "quality_checkpoint",
            "run_id": run_id,
            "checkpoint_index": 2,
            "phase": "final",
            "iteration": 1,
            "quality": 11.5,
            "quality_delta_vs_start": 1.5,
            "elapsed_ms_since_run_start": 200.0,
            "iteration_elapsed_ms": 0.0,
            "n_clusters": 3,
            "max_doc_weight": 5.0,
            "max_doc_weight_ratio": 1.0,
            "n_above_max_doc_weight": 0,
            "moved_nodes": 0,
            "selected_parent_count_total": 1,
            "applied_parent_count_total": 1,
        },
    ]
    trace_path.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    runs_path.write_text(
        json.dumps(
            {
                "schema": "dongdaemun_refinement_quality_trace_run.v1",
                "run_id": run_id,
                "row_key": "row-key",
                "sample": "field30",
                "variant": "refine_repair_off",
                "config_id": "mild_sp0",
                "candidate_quality_policy": "selective",
                "gamma_preset": "mild",
                "seed_perturbations": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    payload = module.summarize_quality_trace(
        trace_path=trace_path,
        runs_path=runs_path,
        output_dir=tmp_path / "summary",
    )

    assert payload["n_checkpoints"] == 3
    assert payload["n_runs"] == 1
    for path in payload["paths"].values():
        assert Path(path).exists()
    with Path(payload["paths"]["by_run"]).open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0]["final_quality_delta_vs_start"] == "1.5"
    assert rows[0]["elapsed_ms_final"] == "200.0"
    assert rows[0]["time_to_95pct_final_quality_gain_ms"] == "100.0"
    assert rows[0]["best_quality_delta_per_sec"] == "15.0"
    assert rows[0]["final_pressure_reduction_per_sec"] == "1.9999999999999996"
    assert rows[0]["candidate_quality_policy"] == "selective"

    with Path(payload["paths"]["quality_gain_per_sec_by_run"]).open(
        encoding="utf-8"
    ) as fh:
        gain_rows = list(csv.DictReader(fh))
    assert gain_rows[0]["final_quality_gain_per_sec"] == "7.5"


def test_quality_trace_summarizer_handles_old_traces_without_elapsed(tmp_path):
    module = _load_quality_summary_module()
    trace_path = tmp_path / "quality_trace.jsonl"
    run_id = "old-run"
    events = [
        {
            "event": "quality_checkpoint",
            "run_id": run_id,
            "checkpoint_index": 0,
            "phase": "start",
            "iteration": 0,
            "quality": 10.0,
            "quality_delta_vs_start": 0.0,
            "n_clusters": 2,
            "max_doc_weight": 7.0,
            "max_doc_weight_ratio": 1.4,
            "n_above_max_doc_weight": 1,
        },
        {
            "event": "quality_checkpoint",
            "run_id": run_id,
            "checkpoint_index": 1,
            "phase": "final",
            "iteration": 1,
            "quality": 11.0,
            "quality_delta_vs_start": 1.0,
            "n_clusters": 3,
            "max_doc_weight": 6.0,
            "max_doc_weight_ratio": 1.2,
            "n_above_max_doc_weight": 1,
        },
    ]
    trace_path.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )

    payload = module.summarize_quality_trace(
        trace_path=trace_path,
        runs_path=None,
        output_dir=tmp_path / "summary",
    )

    with Path(payload["paths"]["by_run"]).open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0]["elapsed_ms_final"] == ""
    assert rows[0]["time_to_95pct_final_quality_gain_ms"] == ""
    assert rows[0]["best_quality_delta_per_sec"] == ""
    assert rows[0]["final_pressure_reduction_per_sec"] == ""
