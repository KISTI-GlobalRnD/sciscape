"""Tests for cyclic lookahead guard dataset collection."""

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
    / "collect_cyclic_lookahead_guard_dataset.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "collect_cyclic_lookahead_guard_dataset_for_test",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_rows(path: Path) -> None:
    rows = [
        {
            "sample": "tiny",
            "variant": "local_qf_beam_cyclic_lookahead",
            "phase": "refinement_chunk",
            "step": 2,
            "chunk_index": 2,
            "quality": 100.0,
            "quality_delta_vs_start": 20.0,
            "max_doc_weight_ratio": 1.2,
            "n_above_max_doc_weight": 2,
            "n_clusters": 10,
            "n_singletons": 1,
            "selected_parent_count_total": 4,
            "applied_parent_count_total": 0,
            "same_gamma_candidates_total": 0,
            "high_gamma_candidates_total": 12,
            "candidate_quality_delta_sum": 5.0,
        },
        {
            "sample": "tiny",
            "variant": "local_qf_beam_cyclic_lookahead",
            "phase": "cyclic_postprocess",
            "step": 2,
            "chunk_index": 2,
            "quality": 120.0,
            "postprocess_status": "committed",
            "postprocess_reasons": '["no_applied_parents","accepted"]',
            "postprocess_accepted": True,
            "postprocess_quality_before": 100.0,
            "postprocess_quality_after": 101.0,
            "lookahead_guard_used": True,
            "lookahead_guard_accepted": True,
            "lookahead_iterations": 2,
            "lookahead_baseline_quality": 110.0,
            "lookahead_candidate_quality": 120.0,
            "lookahead_delta_q": 10.0,
            "lookahead_min_delta_q": 1.0,
            "lookahead_elapsed_sec": 0.5,
        },
        {
            "sample": "tiny",
            "variant": "local_qf_beam_cyclic_lookahead",
            "phase": "refinement_chunk",
            "step": 4,
            "chunk_index": 4,
            "quality": 130.0,
            "quality_delta_vs_start": 50.0,
            "max_doc_weight_ratio": 1.1,
            "n_above_max_doc_weight": 1,
            "n_clusters": 11,
            "n_singletons": 2,
            "selected_parent_count_total": 3,
            "applied_parent_count_total": 0,
            "same_gamma_candidates_total": 0,
            "high_gamma_candidates_total": 12,
            "candidate_quality_delta_sum": -5.0,
        },
        {
            "sample": "tiny",
            "variant": "local_qf_beam_cyclic_lookahead",
            "phase": "cyclic_postprocess",
            "step": 4,
            "chunk_index": 4,
            "quality": 130.0,
            "postprocess_status": "lookahead_guard_rejected",
            "postprocess_reasons": '["no_applied_parents","lookahead_rejected"]',
            "postprocess_accepted": False,
            "postprocess_quality_before": 130.0,
            "postprocess_quality_after": 131.0,
            "lookahead_guard_used": True,
            "lookahead_guard_accepted": False,
            "lookahead_iterations": 2,
            "lookahead_baseline_quality": 140.0,
            "lookahead_candidate_quality": 130.0,
            "lookahead_delta_q": -10.0,
            "lookahead_min_delta_q": 1.0,
            "lookahead_elapsed_sec": 0.4,
        },
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    fieldnames.append(key)
                    seen.add(key)
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_build_trigger_rows_extracts_lookahead_labels(tmp_path):
    module = _load_module()
    rows_path = tmp_path / "run" / "cyclic_postprocess_pilot_rows.csv"
    rows_path.parent.mkdir()
    _write_rows(rows_path)

    rows = module.build_trigger_rows([rows_path])

    assert len(rows) == 2
    assert rows[0]["lookahead_label"] is True
    assert rows[1]["lookahead_label"] is False
    assert rows[0]["immediate_delta_q"] == 1.0
    assert rows[1]["postprocess_status"] == "lookahead_guard_rejected"
    assert rows[0]["refinement_candidate_quality_delta_sum"] == 5.0


def test_collect_guard_dataset_writes_outputs(tmp_path):
    module = _load_module()
    rows_path = tmp_path / "run" / "cyclic_postprocess_pilot_rows.csv"
    output_dir = tmp_path / "guard"
    rows_path.parent.mkdir()
    _write_rows(rows_path)

    payload = module.collect_guard_dataset(
        row_paths=[rows_path],
        output_dir=output_dir,
    )

    assert payload["n_trigger_rows"] == 2
    assert payload["n_labeled_rows"] == 2
    assert payload["n_oracle_accept"] == 1
    assert payload["n_oracle_reject"] == 1
    for path in payload["paths"].values():
        assert Path(path).exists()
    report = Path(payload["paths"]["report"]).read_text(encoding="utf-8")
    assert "Rule Screen" in report
    with Path(payload["paths"]["rule_screen"]).open(encoding="utf-8") as fh:
        rule_rows = list(csv.DictReader(fh))
    assert any("immediate_delta_q" in row["rule"] for row in rule_rows)
