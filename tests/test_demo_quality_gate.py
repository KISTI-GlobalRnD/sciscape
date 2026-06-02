from __future__ import annotations

import json
import subprocess
import sys

from examples.openalex_live_demo import PRESETS, load_demo_manifest


def test_demo_manifest_defines_canonical_public_presets():
    manifest = load_demo_manifest()

    assert set(PRESETS) == {"perovskite", "gnn"}
    assert manifest["default_output_root"] == "workspace/examples_output/openalex_live"
    assert PRESETS["perovskite"].slug == "perovskite_solar_cells_2020_2024"
    assert PRESETS["gnn"].slug == "graph_neural_networks_2020_2024"
    assert PRESETS["gnn"].filters["title_and_abstract.search"] == "graph neural networks"

    for preset in manifest["presets"].values():
        assert "result_manifest.json" in preset["expected_artifacts"]
        assert "landscape/report/data.json" in preset["expected_artifacts"]
        assert "landscape/keywords.parquet" in preset["expected_artifacts"]
        assert "landscape/edge_evidence_samples.json" in preset["expected_artifacts"]


def test_quality_gate_smoke_runs_without_external_data():
    result = subprocess.run(
        [sys.executable, "scripts/sciscape_quality_gate.py", "--smoke", "--json"],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)

    smoke = payload["gates"]["smoke"]
    assert smoke["status"] == "passed"
    assert smoke["keywords"] > 0
    assert smoke["term_network_nodes"] > 0
    assert smoke["term_network_edges"] > 0


def test_quality_gate_web_demo_smoke_runs_without_external_data():
    result = subprocess.run(
        [sys.executable, "scripts/sciscape_quality_gate.py", "--web-demo-smoke", "--json"],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)

    smoke = payload["gates"]["web_demo_smoke"]
    assert smoke["status"] == "passed"
    assert smoke["term_network_nodes"] > 0
    assert smoke["term_network_edges"] > 0
    assert smoke["edge_evidence_samples"] > 0
