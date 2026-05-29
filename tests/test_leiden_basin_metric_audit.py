from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "research/consensus/scripts/leiden_basin/evidence_panels/audits/audit_leiden_basin_evaluation_metrics.py"
)
SPEC = importlib.util.spec_from_file_location("basin_metric_audit", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_audit_flags_changed_node_count_without_label_invariant_metric(tmp_path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "rows.csv").write_text(
        "case,changed_node_count,state_quality\nx,10,1.0\n",
        encoding="utf-8",
    )

    rows = audit.audit_root(tmp_path)

    assert rows.iloc[0]["risk_label"] == "rerun_or_backfill_required"


def test_audit_allows_changed_count_when_aligned_metric_is_present(tmp_path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "rows.csv").write_text(
        "case,changed_node_count,aligned_changed_node_count,endpoint_distance\nx,10,2,0.1\n",
        encoding="utf-8",
    )

    rows = audit.audit_root(tmp_path)

    assert rows.iloc[0]["risk_label"] == "relabel_exact_columns_and_reinterpret"


def test_audit_treats_changed_support_aliases_as_label_invariant(tmp_path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "rows.csv").write_text(
        (
            "case,p5_changed_nodes_vs_baseline,"
            "p5_basin_changed_support_node_count,p5_basin_changed_support_nodes\n"
            "x,10,10,1;2\n"
        ),
        encoding="utf-8",
    )

    rows = audit.audit_root(tmp_path)

    assert rows.iloc[0]["risk_label"] == "label_invariant_metrics_present"
    assert bool(rows.iloc[0]["has_changed_support_metric"]) is True


def test_audit_skips_its_own_output_family(tmp_path):
    artifact = tmp_path / "basin_evaluation_metric_audit_v0"
    artifact.mkdir()
    (artifact / "basin_evaluation_metric_audit_rows.csv").write_text(
        "case,changed_node_count\nx,1\n",
        encoding="utf-8",
    )

    rows = audit.audit_root(tmp_path)

    assert rows.empty
