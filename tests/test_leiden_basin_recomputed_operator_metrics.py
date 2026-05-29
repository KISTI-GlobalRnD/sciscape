"""Tests for recomputed Leiden basin operator metric review."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "research/consensus/scripts/leiden_basin/basin_signatures/endpoint_flips/summarize_leiden_basin_recomputed_operator_metrics.py"
)


def _load_module():
    if str(SCRIPT_PATH.parent) not in sys.path:
        sys.path.insert(0, str(SCRIPT_PATH.parent))
    spec = importlib.util.spec_from_file_location(
        "summarize_leiden_basin_recomputed_operator_metrics_for_test",
        SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_summarize_artifact_keeps_exact_and_aligned_separate(tmp_path):
    module = _load_module()
    rows_path = tmp_path / "rows.csv"
    pd.DataFrame(
        [
            {
                "gain": 1.0,
                "verdict": "better",
                "exact": 100,
                "aligned": 3,
                "exact_only": 97,
                "endpoint": 0.1,
            },
            {
                "gain": 0.5,
                "verdict": "worse",
                "exact": 2,
                "aligned": 2,
                "exact_only": 0,
                "endpoint": 0.0,
            },
        ]
    ).to_csv(rows_path, index=False)
    spec = module.ArtifactSpec(
        artifact="toy",
        rows_path=rows_path,
        quality_gain_col="gain",
        verdict_col="verdict",
        final_exact_col="exact",
        final_aligned_col="aligned",
        final_exact_only_col="exact_only",
        endpoint_col="endpoint",
    )

    summary, top_rows = module.summarize_artifact(spec, top_k=1)

    assert summary["row_count"] == 2
    assert summary["max_quality_gain"] == 1.0
    assert summary["max_final_exact_changed"] == 100
    assert summary["max_final_aligned_changed"] == 3
    assert summary["max_final_exact_only_changed"] == 97
    assert summary["verdict_counts"] == "better:1;worse:1"
    assert len(top_rows) == 2


def test_run_review_writes_artifacts_with_patched_specs(tmp_path, monkeypatch):
    module = _load_module()
    rows_path = tmp_path / "rows.csv"
    pd.DataFrame(
        [
            {
                "gain": 1.0,
                "verdict": "better",
                "changed_node_count": 2,
                "aligned_changed_node_count": 2,
                "endpoint": 0.0,
            }
        ]
    ).to_csv(rows_path, index=False)
    monkeypatch.setattr(
        module,
        "DEFAULT_ARTIFACTS",
        (
            module.ArtifactSpec(
                artifact="gate",
                rows_path=rows_path,
                quality_gain_col="gain",
                verdict_col="verdict",
                final_exact_col="changed_node_count",
                final_aligned_col="aligned_changed_node_count",
                endpoint_col="endpoint",
            ),
        ),
    )

    payload = module.run_review(output_dir=tmp_path / "out", top_k=1)

    assert Path(payload["paths"]["summary"]).exists()
    assert Path(payload["paths"]["top_rows"]).exists()
    assert Path(payload["paths"]["report"]).exists()
    report = Path(payload["paths"]["report"]).read_text(encoding="utf-8")
    assert "Recomputed Operator Metric Review" in report
