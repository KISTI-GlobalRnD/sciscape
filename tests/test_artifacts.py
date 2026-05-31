from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from sciscape.artifacts import (
    build_report_data_contract,
    validate_result_root,
    write_artifact_contract,
)
from sciscape.keyword_extraction.visualization import export_dashboard


def _write_valid_result_root(root: Path) -> Path:
    landscape = root / "landscape"
    report = landscape / "report"
    report.mkdir(parents=True)

    pd.DataFrame(
        {
            "uid": ["D0", "D1", "D2", "D3"],
            "title": [
                "Perovskite interface passivation",
                "Perovskite device stability",
                "Graph neural traffic forecasting",
                "Graph neural anomaly detection",
            ],
            "abstract": [
                "Interface passivation improves perovskite solar cell stability.",
                "Stable perovskite devices use passivation layers.",
                "Graph neural networks forecast traffic over sensor graphs.",
                "Graph neural networks detect anomalies in dynamic graphs.",
            ],
            "pubyear": [2021, 2022, 2021, 2022],
        }
    ).to_parquet(root / "abstracts.parquet", index=False)
    pd.DataFrame(
        {
            "uid1": ["D0", "D1", "D2"],
            "uid2": ["D1", "D2", "D3"],
            "rel_sum2": [2.0, 1.0, 2.0],
        }
    ).to_parquet(root / "edges.parquet", index=False)
    pd.DataFrame(
        {
            "uid": ["D0", "D1", "D2", "D3"],
            "cluster": [0, 0, 1, 1],
        }
    ).to_parquet(landscape / "membership.parquet", index=False)
    pd.DataFrame(
        {
            "cluster_id": [0, 0, 1, 1],
            "term": [
                "perovskite solar cells",
                "interface passivation",
                "graph neural networks",
                "traffic forecasting",
            ],
            "score": [0.9, 0.8, 0.95, 0.75],
            "frequency": [2, 1, 2, 1],
        }
    ).to_parquet(landscape / "keywords.parquet", index=False)

    report_data = {
        "0": {
            "label": "perovskite solar cells",
            "keywords": [{"term": "perovskite solar cells"}, {"term": "interface passivation"}],
            "network_edges": [{"source": "perovskite solar cells", "target": "interface passivation", "weight": 1}],
            "cooccurrence_table": [{"source": "perovskite solar cells", "target": "interface passivation", "count": 1}],
        },
        "1": {
            "label": "graph neural networks",
            "keywords": [{"term": "graph neural networks"}, {"term": "traffic forecasting"}],
            "network_edges": [{"source": "graph neural networks", "target": "traffic forecasting", "weight": 1}],
            "cooccurrence_table": [{"source": "graph neural networks", "target": "traffic forecasting", "count": 1}],
        },
        "_trend_scores": {"perovskite solar cells": {"2021": 1, "2022": 2}},
    }
    report_data["_sciscape"] = build_report_data_contract(report_data)
    (report / "data.json").write_text(json.dumps(report_data), encoding="utf-8")
    return root


def test_validate_result_root_infers_features_and_counts(tmp_path):
    root = _write_valid_result_root(tmp_path / "result")

    result = validate_result_root(root)
    payload = result.to_dict()

    assert payload["ok"] is True
    assert payload["result_state"] == "loaded"
    assert payload["features"]["overview"] is True
    assert payload["features"]["cluster_map"] is True
    assert payload["features"]["keyword"] is True
    assert payload["features"]["term_network"] is True
    assert payload["features"]["matrix"] is True
    assert payload["features"]["evidence"] is True
    assert payload["features"]["temporal"] is True
    assert payload["features"]["quality"] is True
    assert payload["features"]["export"] is True
    assert payload["counts"]["abstract_rows"] == 4
    assert payload["counts"]["membership_rows"] == 4
    assert payload["counts"]["keyword_rows"] == 4
    assert payload["counts"]["report_clusters"] == 2
    assert not [w for w in payload["warnings"] if w["severity"] == "error"]


def test_validate_result_root_blocks_advertised_missing_feature(tmp_path):
    report = tmp_path / "result" / "landscape" / "report"
    report.mkdir(parents=True)
    (report / "data.json").write_text(
        json.dumps(
            {
                "0": {"label": "empty", "keywords": []},
                "_sciscape": {
                    "features": {"term_network": True},
                    "schema_version": "sciscape_report_data_contract_v1",
                },
            }
        ),
        encoding="utf-8",
    )

    result = validate_result_root(report / "data.json")
    payload = result.to_dict()

    assert payload["ok"] is False
    assert payload["result_state"] == "blocked"
    assert any(w["code"] == "advertised_feature_missing" for w in payload["warnings"])


def test_validate_result_root_blocks_top_metadata_artifacts(tmp_path):
    root = _write_valid_result_root(tmp_path / "result")
    keyword_path = root / "landscape" / "keywords.parquet"
    keywords = pd.read_parquet(keyword_path)
    contaminated = pd.concat(
        [
            pd.DataFrame(
                {
                    "cluster_id": [0],
                    "term": ["class htmlview paragraph"],
                    "score": [999.0],
                    "frequency": [10],
                    "quality_flags": ["metadata_fragment"],
                    "representative_rank": [1],
                }
            ),
            keywords.assign(representative_rank=range(2, len(keywords) + 2)),
        ],
        ignore_index=True,
    )
    contaminated.to_parquet(keyword_path, index=False)

    data_path = root / "landscape" / "report" / "data.json"
    data = json.loads(data_path.read_text(encoding="utf-8"))
    data["0"]["keywords"].insert(
        0,
        {
            "term": "usepackage",
            "display_label": "usepackage",
            "quality_flags": "metadata_fragment",
        },
    )
    data_path.write_text(json.dumps(data), encoding="utf-8")

    result = validate_result_root(root)
    payload = result.to_dict()

    assert payload["ok"] is False
    assert payload["result_state"] == "blocked"
    assert payload["counts"]["keyword_top_artifact_rows"] == 1
    assert payload["counts"]["report_keyword_top_artifact_rows"] == 1
    codes = {warning["code"] for warning in payload["warnings"]}
    assert "top_keyword_artifact" in codes
    assert "top_report_keyword_artifact" in codes


def test_write_artifact_contract_uses_landscape_qa_dir(tmp_path):
    root = _write_valid_result_root(tmp_path / "result")

    result = write_artifact_contract(root)
    contract = root / "landscape" / "qa" / "artifact_contract.json"

    assert result.ok is True
    assert contract.exists()
    payload = json.loads(contract.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "sciscape_artifact_contract_v1"
    assert payload["features"]["term_network"] is True


def test_quality_gate_validates_artifact_root_and_writes_contract(tmp_path):
    root = _write_valid_result_root(tmp_path / "result")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/sciscape_quality_gate.py",
            "--artifact-root",
            str(root),
            "--write-artifact-contract",
            "--json",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(completed.stdout)

    gate = payload["gates"]["artifact_contract"]
    assert payload["status"] == "passed"
    assert gate["ok"] is True
    assert gate["features"]["term_network"] is True
    assert Path(gate["artifact_contract_path"]).exists()


def test_dashboard_export_embeds_report_data_contract(tmp_path):
    keywords = pd.DataFrame(
        {
            "cluster_id": [0, 0],
            "term": ["perovskite solar cells", "interface passivation"],
            "score": [0.9, 0.8],
            "frequency": [2, 1],
        }
    )
    dashboard = tmp_path / "dashboard.html"

    export_dashboard(keywords, output_path=str(dashboard))
    html = dashboard.read_text(encoding="utf-8")

    assert "sciscape_report_data_contract_v1" in html
    assert '"sciscape_version"' in html
    assert "SCISCAPE_CONTRACT" in html
    assert "Result contract" in html
    assert "TAB_FEATURES" in html
