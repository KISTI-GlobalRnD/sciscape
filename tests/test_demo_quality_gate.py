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


def test_quality_gate_p1_atlas_smoke_runs_full_pipeline_without_external_data():
    result = subprocess.run(
        [sys.executable, "scripts/sciscape_quality_gate.py", "--p1-atlas-smoke", "--json"],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)

    smoke = payload["gates"]["p1_atlas_smoke"]
    assert smoke["status"] == "passed"
    assert smoke["clusters"] >= 2
    assert smoke["keywords"] > 0
    assert smoke["term_network_nodes"] > 0
    assert smoke["term_network_edges"] > 0
    assert smoke["cooccurrence_rows"] > 0
    assert smoke["edge_evidence_samples"] > 0
    assert smoke["atlas_neighbor_rows"] > 0
    assert smoke["atlas_neighbor_aggregate_contract_rows"] == smoke["atlas_neighbor_rows"]
    assert smoke["atlas_neighbor_sampled_rows"] > 0
    assert smoke["atlas_neighbor_shared_term_rows"] >= 0
    assert smoke["atlas_neighbor_shared_term_contract_rows"] == smoke["atlas_neighbor_rows"]
    assert smoke["atlas_render_nodes"] >= 2
    assert smoke["atlas_render_labels"] >= 2
    assert smoke["atlas_render_coordinate_source"] in {"generated", "mixed", "node_coordinates"}
    assert smoke["feature_states"]["cluster_map"] == "stable"
    assert smoke["feature_states"]["keyword"] == "stable"
    assert smoke["feature_states"]["term_network"] == "stable"
    assert smoke["feature_states"]["cooccurrence"] == "stable"
    assert smoke["feature_states"]["evidence"] == "stable"


def test_quality_gate_atlas_render_perf_smoke_runs_without_browser():
    result = subprocess.run(
        [sys.executable, "scripts/sciscape_quality_gate.py", "--atlas-render-perf-smoke", "--json"],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)

    smoke = payload["gates"]["atlas_render_perf_smoke"]
    assert smoke["status"] == "passed"
    assert smoke["nodes"] == 100
    assert smoke["edges"] == 500
    assert smoke["labels"] == 100
    assert smoke["hierarchy_edges"] >= 90
    assert smoke["coordinate_source"] == "node_coordinates"
    assert smoke["payload_json_bytes"] < 2_000_000
    assert smoke["build_ms"] < 1000.0
    assert smoke["recommended_layers"]["nodes"] == "ScatterplotLayer"
    assert smoke["recommended_layers"]["edges"] == "LineLayer"
    assert smoke["recommended_layers"]["labels"] == "TextLayer"


def test_quality_gate_atlas_render_scale_smoke_runs_without_browser():
    result = subprocess.run(
        [sys.executable, "scripts/sciscape_quality_gate.py", "--atlas-render-scale-smoke", "--json"],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)

    smoke = payload["gates"]["atlas_render_scale_smoke"]
    assert smoke["status"] == "passed"
    assert smoke["nodes"] == 5000
    assert smoke["edges"] == 25000
    assert smoke["labels"] == 5000
    assert smoke["hierarchy_edges"] >= 4990
    assert smoke["coordinate_source"] == "node_coordinates"
    assert smoke["payload_json_bytes"] < 50_000_000
    assert smoke["build_ms"] < 5000.0


def test_quality_gate_exposes_optional_atlas_interaction_smoke():
    result = subprocess.run(
        [sys.executable, "scripts/sciscape_quality_gate.py", "--help"],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "--atlas-interaction-smoke" in result.stdout
    assert "--atlas-inspector-smoke" in result.stdout


def test_quality_gate_atlas_inspector_smoke_runs_or_skips_without_browser():
    result = subprocess.run(
        [sys.executable, "scripts/sciscape_quality_gate.py", "--atlas-inspector-smoke", "--json"],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)

    smoke = payload["gates"]["atlas_inspector_smoke"]
    assert smoke["status"] in {"passed", "skipped"}
    if smoke["status"] == "skipped":
        assert "browser" in smoke["reason"]
        return
    assert smoke["schema"] == "sciscape_inspector_evidence_view_v1"
    assert smoke["selected_uid"] in {"micro:1", "micro:2"}
    assert smoke["relation_state"] == "stable"
    assert smoke["works_state"] == "stable"
    assert smoke["qa_state"] == "stable"
    assert smoke["initial_review_state"] == "ready"
    assert smoke["selected_review_state"] == "review"
    assert smoke["review_queue_rows"] >= 1
    assert smoke["review_filter"] == "review"
    assert smoke["next_target_uid"] in {"micro:1", "micro:2"}
    assert smoke["review_packet_seen"] is True
    assert smoke["filtered_evidence_rows"] == 2
    assert smoke["sample_rows"] >= 1
    assert smoke["aggregate_fallback_seen"] is True
    assert smoke["non_background_pixel_ratio"] > 0.001
