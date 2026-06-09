"""Tests for the SciScape FastAPI web surface."""

from __future__ import annotations

import json
import uuid
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import sciscape.web.app as web_app
from sciscape.artifacts import (
    validate_export_manifest,
    write_evolution_synthetic_smoke_artifact,
    write_export_manifest,
    write_workspace_manifest,
)
from sciscape.web.jobstore import JobStore

app = web_app.app


@pytest.fixture(autouse=True)
def isolated_job_store(monkeypatch, tmp_path):
    store = JobStore(tmp_path / "sciscape_test_jobs.db")
    monkeypatch.setattr(web_app, "_jobs", store)
    yield


def _register_done_job(job_id: str, output_dir: Path) -> None:
    web_app._jobs.create(job_id, {"query": "test query"})
    job = web_app._jobs.get(job_id)
    assert job is not None
    job["status"] = "done"
    job["progress"] = []
    job["result"] = {"output_dir": str(output_dir)}
    web_app._jobs.persist(job_id)


def test_download_route_supports_nested_output_artifacts(tmp_path):
    job_id = f"testnested{uuid.uuid4().hex[:8]}"
    artifact = tmp_path / "landscape" / "report" / "index.html"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("<html>dashboard</html>", encoding="utf-8")
    _register_done_job(job_id, tmp_path)

    client = TestClient(app)
    response = client.get(f"/api/jobs/{job_id}/download/landscape/report/index.html")

    assert response.status_code == 200
    assert response.text == "<html>dashboard</html>"
    assert "index.html" in response.headers["content-disposition"]


def test_view_route_opens_nested_html_artifact_inline(tmp_path):
    job_id = f"testview{uuid.uuid4().hex[:8]}"
    artifact = tmp_path / "landscape" / "report" / "report.html"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("<html>report</html>", encoding="utf-8")
    _register_done_job(job_id, tmp_path)

    client = TestClient(app)
    response = client.get(f"/api/jobs/{job_id}/view/landscape/report/report.html")

    assert response.status_code == 200
    assert response.text == "<html>report</html>"
    assert "content-disposition" not in response.headers


def test_output_artifact_route_rejects_path_traversal(tmp_path):
    job_id = f"testtraversal{uuid.uuid4().hex[:8]}"
    _register_done_job(job_id, tmp_path)

    client = TestClient(app)
    response = client.get(f"/api/jobs/{job_id}/view/%2E%2E/secret.txt")

    assert response.status_code == 400
    assert response.json()["detail"] == "invalid artifact path"


def test_network_export_endpoint_writes_export_manifest(tmp_path):
    import polars as pl

    job_id = f"testexport{uuid.uuid4().hex[:8]}"
    output_dir = tmp_path / "result"
    landscape_dir = output_dir / "landscape"
    landscape_dir.mkdir(parents=True)
    edges_path = output_dir / "edges.parquet"
    abstracts_path = output_dir / "abstracts.parquet"
    pl.DataFrame({"uid1": ["D0", "D1"], "uid2": ["D1", "D2"], "rel_sum2": [1.0, 0.5]}).write_parquet(edges_path)
    pl.DataFrame({"uid": ["D0", "D1", "D2"], "cluster": [0, 1, 2]}).write_parquet(
        landscape_dir / "membership.parquet"
    )
    pl.DataFrame(
        {
            "uid": ["D0", "D1", "D2"],
            "title": ["Paper A", "Paper B", "Paper C"],
            "abstract": ["A", "B", "C"],
            "pubyear": [2021, 2022, 2023],
        }
    ).write_parquet(abstracts_path)

    web_app._jobs.create(job_id, {"query": "export test"})
    job = web_app._jobs.get(job_id)
    assert job is not None
    job["status"] = "done"
    job["progress"] = []
    job["result"] = {
        "output_dir": str(output_dir),
        "landscape_dir": str(landscape_dir),
        "edges_path": str(edges_path),
        "abstracts_path": str(abstracts_path),
    }
    web_app._jobs.persist(job_id)

    client = TestClient(app)
    response = client.get(
        f"/api/jobs/{job_id}/export/graphml",
        params={
            "atlas_level": "cluster",
            "atlas_node": "cluster:0",
            "atlas_query": "passivation",
            "atlas_lens": "evidence",
            "atlas_view": "map",
            "atlas_focus": "neighbors",
            "atlas_review": "review",
            "atlas_layers": "edges,labels",
            "atlas_edge_min": "0.25",
            "atlas_label_limit": "24",
            "atlas_neighbor": "cluster:1",
            "atlas_subset_mode": "neighbors",
            "atlas_subset_count": "2",
            "atlas_subset_uids": "cluster:0,cluster:1",
            "atlas_subset_truncated": "false",
            "atlas_pinned": "cluster:1",
        },
    )

    assert response.status_code == 200
    manifest_path = output_dir / "exports" / "network_graphml" / "export_manifest.json"
    assert manifest_path.exists()
    validation = validate_export_manifest(manifest_path).to_dict()
    assert validation["status"] == "passed"
    assert validation["export_kind"] == "graphml_graph"
    export_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert export_manifest["selection"]["view"] == {
        "mode": "web_network_export",
        "surface": "web_export_endpoint",
        "family": "graph",
    }
    assert export_manifest["selection"]["layer_state"] == {
        "network_format": "graphml",
        "membership_source": str(landscape_dir / "membership.parquet"),
        "atlas_lens": "evidence",
        "atlas_view": "map",
        "atlas_layers": ["edges", "labels"],
        "atlas_label_limit": 24,
        "subset_applied": True,
        "subset_membership_column": "cluster",
        "subset_output_node_count": 2,
        "subset_output_edge_count": 1,
    }
    assert export_manifest["selection"]["scope"] == "selected_subset"
    assert export_manifest["selection"]["cluster_level"] == "cluster"
    assert export_manifest["selection"]["filters"] == [
        {"field": "atlas_query", "op": "contains", "value": "passivation"},
        {"field": "atlas_review_state", "op": "eq", "value": "review"},
    ]
    assert export_manifest["selection"]["thresholds"] == {"atlas_edge_min": 0.25}
    assert export_manifest["selection"]["focus"] == {
        "cluster_uid": "cluster:0",
        "focus_mode": "neighbors",
        "neighbor_uid": "cluster:1",
    }
    assert export_manifest["selection"]["subset"] == {
        "mode": "neighbors",
        "count": 2,
        "uids": ["cluster:0", "cluster:1"],
        "truncated": False,
        "pinned_uids": ["cluster:1"],
        "applied": True,
        "membership_column": "cluster",
        "cluster_level": "cluster",
        "source_node_count": 3,
        "source_edge_count": 2,
        "output_node_count": 2,
        "output_edge_count": 1,
    }
    graphml = (output_dir / "network.graphml").read_text(encoding="utf-8")
    assert 'node id="D0"' in graphml
    assert 'node id="D1"' in graphml
    assert 'node id="D2"' not in graphml
    assert 'source="D0" target="D1"' in graphml
    assert 'source="D1" target="D2"' not in graphml
    transform_table = pd.read_parquet(output_dir / "exports" / "network_graphml" / "export_transforms.parquet")
    assert transform_table["transform_type"].tolist() == [
        "load_edge_table",
        "apply_selected_subset",
        "write_graphml",
    ]

    job_payload = client.get(f"/api/jobs/{job_id}").json()
    exports = job_payload["result"]["result_manifest"]["exports"]
    graphml_exports = [row for row in exports if row["export_id"] == "network_graphml"]
    assert len(graphml_exports) == 1
    assert graphml_exports[0]["path"] == "network.graphml"
    assert graphml_exports[0]["export_manifest_ref"] == "exports/network_graphml/export_manifest.json"
    assert graphml_exports[0]["selection_summary"]["view_mode"] == "web_network_export"
    assert graphml_exports[0]["selection_summary"]["scope"] == "selected_subset"
    assert graphml_exports[0]["selection_summary"]["cluster_level"] == "cluster"
    assert graphml_exports[0]["selection_summary"]["filter_count"] == 2
    assert graphml_exports[0]["selection_summary"]["threshold_keys"] == ["atlas_edge_min"]
    assert graphml_exports[0]["selection_summary"]["focus_keys"] == [
        "cluster_uid",
        "focus_mode",
        "neighbor_uid",
    ]
    assert graphml_exports[0]["selection_summary"]["subset_mode"] == "neighbors"
    assert graphml_exports[0]["selection_summary"]["subset_count"] == 2
    assert graphml_exports[0]["selection_summary"]["subset_keys"] == [
        "applied",
        "cluster_level",
        "count",
        "membership_column",
        "mode",
        "output_edge_count",
        "output_node_count",
        "pinned_uids",
        "source_edge_count",
        "source_node_count",
        "truncated",
        "uids",
    ]
    assert graphml_exports[0]["selection_summary"]["layer_state_keys"] == [
        "atlas_label_limit",
        "atlas_layers",
        "atlas_lens",
        "atlas_view",
        "membership_source",
        "network_format",
        "subset_applied",
        "subset_membership_column",
        "subset_output_edge_count",
        "subset_output_node_count",
    ]


def test_web_homepage_exposes_query_analysis_controls():
    client = TestClient(app)
    response = client.get("/")

    assert response.status_code == 200
    assert "Query OpenAlex" in response.text
    assert "Recommended Demos" in response.text
    assert "Local Data" in response.text
    assert "Artifact contract" in response.text
    assert 'id="atlas-shell"' in response.text
    assert "deck.gl@^9.0.0/dist.min.js" in response.text
    assert "applyResultFeatureAvailability" in response.text
    assert "renderAtlasShell" in response.text
    assert "renderAtlasLineage" in response.text
    assert "renderAtlasEvidenceProfile" in response.text
    assert "buildAtlasInspectorEvidenceModel" in response.text
    assert "renderAtlasInspectorModelSummary" in response.text
    assert "atlasReviewReadiness" in response.text
    assert "renderAtlasReviewChecklist" in response.text
    assert "atlasReviewPacket" in response.text
    assert "renderAtlasReviewPacket" in response.text
    assert "renderAtlasReviewQueue" in response.text
    assert "atlasReviewQueueRows" in response.text
    assert "atlasFilteredReviewRows" in response.text
    assert "selectAtlasReviewFilter" in response.text
    assert "openNextAtlasReviewTarget" in response.text
    assert "manifestExportCards" in response.text
    assert "manifestExportRow" in response.text
    assert "manifestExportSelectionMeta" in response.text
    assert "cooccurrenceExportConfig" in response.text
    assert "term_cooccurrence_table" in response.text
    assert "cooccurrenceExport" in response.text
    assert "term-export-link" in response.text
    assert "term-quality-bar" in response.text
    assert "result_manifest.exports" in response.text
    assert "selection_summary" in response.text
    assert "dl-meta" in response.text
    assert "Export manifest" in response.text
    assert "Next target" in response.text
    assert "Review checklist" in response.text
    assert "Review queue" in response.text
    assert "atlas_review" in response.text
    assert "Cluster reading" in response.text
    assert "renderAtlasMeaningLayer" in response.text
    assert "renderAtlasQALimitations" in response.text
    assert "aggregate only" in response.text
    assert "Raw pair samples unavailable" in response.text
    assert "atlasFeatureState" in response.text
    assert "sciscape_inspector_evidence_view_v1" in response.text
    assert "renderAtlasChildren" in response.text
    assert "atlasChildrenForNode" in response.text
    assert "renderAtlasSearchPanel" in response.text
    assert "atlasSearchRows" in response.text
    assert "atlas_query" in response.text
    assert "renderAtlasLensControls" in response.text
    assert "renderAtlasLensScalePanel" in response.text
    assert "atlasLensStats" in response.text
    assert "atlasMetricRange" in response.text
    assert "renderAtlasOrientation" in response.text
    assert "atlasOrientationStats" in response.text
    assert "atlasSourceLabel" in response.text
    assert "renderAtlasSessionRail" in response.text
    assert "toggleAtlasPin" in response.text
    assert "atlasSessionStorageKey" in response.text
    assert "renderAtlasViewControls" in response.text
    assert "selectAtlasViewMode" in response.text
    assert "renderAtlasDeckPanel" in response.text
    assert "renderAtlasDeckControls" in response.text
    assert "setAtlasDeckLayer" in response.text
    assert "setAtlasDeckMinEdgeWeight" in response.text
    assert "setAtlasDeckLabelLimit" in response.text
    assert "loadAtlasDeckRender" in response.text
    assert "/atlas-render" in response.text
    assert "new deck.ScatterplotLayer" in response.text
    assert "new deck.LineLayer" in response.text
    assert "new deck.TextLayer" in response.text
    assert "new deck.OrthographicView" in response.text
    assert "renderAtlasHierarchyView" in response.text
    assert "renderAtlasEvidenceNodeView" in response.text
    assert "atlas_view" in response.text
    assert "atlas_layers" in response.text
    assert "atlas_edge_min" in response.text
    assert "atlas_label_limit" in response.text
    assert "resultFeatureBlock" in response.text
    assert "renderAtlasModulePanel" in response.text
    assert "atlasModuleReadinessNote" in response.text
    assert "renderAtlasFocusControls" in response.text
    assert "selectAtlasFocusMode" in response.text
    assert "atlasFocusedNodes" in response.text
    assert "atlas_focus" in response.text
    assert "selectAtlasNeighborEvidence" in response.text
    assert "renderAtlasNeighborEvidence" in response.text
    assert "atlasNeighborSamples" in response.text
    assert "atlas_neighbor" in response.text

    network_response = client.get("/static/network.js")
    assert network_response.status_code == 200
    assert "term-label-state" in network_response.text
    assert "term-edge-state" in network_response.text
    assert "data-te-preset" in network_response.text
    assert "_setEdgeThreshold" in network_response.text
    assert "_edgePresetThreshold" in network_response.text
    assert "_edgeQuantile" in network_response.text
    assert "_applyEdgeFilter" in network_response.text
    assert "term-quality-metric" in network_response.text
    assert "renderAtlasNeighborSummary" in response.text
    assert 'data-tab="evolution"' in response.text
    assert 'id="evolution-content"' in response.text
    assert "loadEvolution" in response.text
    assert "renderEvolutionLens" in response.text
    assert "selectEvolutionEventFilter" in response.text
    assert "/api/jobs/${currentJobId}/evolution" in response.text
    assert "evolution-shell" in response.text
    assert "selectAtlasLens" in response.text
    assert "atlasEvidenceScore" in response.text
    assert "atlas_lens" in response.text
    assert "applyAtlasUrlState" in response.text
    assert "syncAtlasUrlState" in response.text
    assert "atlas_node" in response.text
    assert "renderAtlasRepresentativeWorks" in response.text
    assert "atlasNodeByUid" in response.text
    assert "preferredAtlasLevel" in response.text
    assert "atlas-search-hit" in response.text
    assert "atlas-lens-panel" in response.text
    assert "atlas-lens-scale-panel" in response.text
    assert "atlas-lens-cell" in response.text
    assert "atlas-orientation" in response.text
    assert "atlas-session-rail" in response.text
    assert "atlas-module-panel" in response.text
    assert "atlas-module-note" in response.text
    assert "atlas-view-panel" in response.text
    assert "atlas-relation-panel" in response.text
    assert "atlas-evidence-node-row" in response.text
    assert "atlas-focus-panel" in response.text
    assert "atlas-neighbor-evidence" in response.text
    assert "atlas-neighbor-summary" in response.text
    assert "atlas-pin-button" in response.text
    assert "atlas-evidence-row" in response.text
    assert "atlas-child-row" in response.text
    assert "atlas-neighbor-row" in response.text
    assert "atlas-work-row" in response.text
    assert 'id="q-search"' in response.text
    assert "submitQuery()" in response.text
    assert "fetch('/api/query'" in response.text
    assert "fetch('/api/demo-presets'" in response.text
    assert "fetch('/api/local-data" in response.text
    assert "new URLSearchParams(window.location.search).get('job')" in response.text
    assert 'id="file-input"' not in response.text


def test_atlas_deck_edge_handlers_prioritize_relation_rows():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    text = response.text

    tooltip = text[text.index("function atlasDeckTooltip") : text.index("function atlasDeckClick")]
    assert tooltip.index("object.source_uid && object.target_uid") < tooltip.index("object.cluster_uid || object.id")

    click = text[text.index("function atlasDeckClick") : text.index("function atlasDeckHover")]
    assert "const edgeUid = [object.source_uid, object.target_uid].find(uid => atlasNodeByUid(uid));" in click
    assert "object.cluster_uid || object.id || object.source_uid" not in click

    hover = text[text.index("function atlasDeckHover") : text.index("function renderAtlasDeckLegend")]
    assert hover.index("object.source_uid && object.target_uid") < hover.index("object.cluster_uid || object.id")


def test_query_endpoint_enqueues_openalex_analysis(monkeypatch, tmp_path):
    def fake_run_job(job_id, req):
        job = web_app._jobs[job_id]
        job["status"] = "done"
        job["progress"].append(f"fake pipeline: {req.query}")
        job["result"] = {
            "n_works": req.max_works,
            "n_edges": {"dc": 0, "bc": 0, "cc": 0},
            "output_dir": str(tmp_path),
            "abstracts_path": None,
            "edges_path": None,
            "landscape_dir": None,
        }
        web_app._jobs.persist(job_id)

    monkeypatch.setattr("sciscape.web.app._run_job", fake_run_job)

    client = TestClient(app)
    response = client.post(
        "/api/query",
        json={
            "query": "graph neural networks",
            "years": "2020-2024",
            "max_works": 12,
            "edge_types": "dc,bc,cc",
            "run_landscape": True,
        },
    )

    assert response.status_code == 200
    job_id = response.json()["job_id"]
    job_response = client.get(f"/api/jobs/{job_id}")

    assert job_response.status_code == 200
    payload = job_response.json()
    assert payload["status"] == "done"
    assert payload["progress"] == ["fake pipeline: graph neural networks"]
    assert payload["result"]["n_works"] == 12


def test_run_job_writes_live_status_and_manifest(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    original_write_result_manifest = web_app.write_result_manifest
    manifest_calls = []

    def counted_write_result_manifest(*args, **kwargs):
        manifest_calls.append((args, kwargs))
        return original_write_result_manifest(*args, **kwargs)

    def fake_openalex_pipeline(config):
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        abstracts_path = output_dir / "abstracts.parquet"
        pd.DataFrame(
            {
                "uid": ["W1"],
                "title": ["Graph neural network survey"],
                "abstract": ["Graph neural networks for citation analysis."],
                "pubyear": [2024],
            }
        ).to_parquet(abstracts_path, index=False)
        config.progress("fetched works")
        return SimpleNamespace(
            n_works=1,
            n_edges={"dc": 0, "bc": 0, "cc": 0},
            abstracts_path=abstracts_path,
            edges_path=None,
            landscape_dir=None,
        )

    monkeypatch.setattr(web_app, "write_result_manifest", counted_write_result_manifest)
    monkeypatch.setattr("sciscape.openalex.run_openalex_pipeline", fake_openalex_pipeline)

    req = web_app.QueryRequest(
        query="graph neural networks",
        years="2020-2024",
        max_works=1,
        run_landscape=False,
    )
    job_id = "jobstatus1"
    web_app._jobs[job_id] = {
        "status": "pending",
        "progress": [],
        "result": None,
        "request": req.model_dump(),
    }

    web_app._run_job(job_id, req)

    output_dir = tmp_path / "workspace" / "web_output" / job_id
    status_payload = json.loads((output_dir / "job_status.json").read_text(encoding="utf-8"))
    manifest_payload = json.loads((output_dir / "result_manifest.json").read_text(encoding="utf-8"))
    job = web_app._jobs[job_id]

    assert job["status"] == "done"
    assert job["progress"] == ["fetched works"]
    assert job["result"]["job_status_path"] == "workspace/web_output/jobstatus1/job_status.json"
    assert status_payload["schema_version"] == "sciscape_live_job_status_v1"
    assert status_payload["status"] == "done"
    assert status_payload["run_state"]["status"] == "complete"
    assert status_payload["started_at_utc"]
    assert status_payload["finished_at_utc"]
    assert status_payload["updated_at_utc"]
    assert status_payload["progress"] == ["fetched works"]
    assert manifest_payload["schema_version"] == "sciscape_result_manifest_v1"
    assert manifest_payload["source"]["query"] == "graph neural networks"
    assert manifest_payload["source"]["filters"] == {"publication_year": "2020-2024"}
    assert manifest_payload["run_state"]["status"] == "complete"
    assert manifest_payload["run_state"]["progress"]["unit"] == "messages"
    assert manifest_payload["artifacts"]["job_status"]["path"] == "job_status.json"
    assert len(manifest_calls) == 2

    reloaded = JobStore(Path(web_app._jobs._db_path))
    reloaded_job = reloaded.get(job_id)
    assert reloaded_job is not None
    assert reloaded_job["started_at_utc"] == status_payload["started_at_utc"]
    assert reloaded_job["finished_at_utc"] == status_payload["finished_at_utc"]
    assert reloaded_job["updated_at_utc"] == status_payload["updated_at_utc"]


def test_local_data_endpoint_lists_workspace_outputs(monkeypatch, tmp_path):
    output_dir = tmp_path / "workspace" / "examples_output" / "demo"
    report_dir = output_dir / "landscape" / "report"
    report_dir.mkdir(parents=True)
    (report_dir / "data.json").write_text("{}", encoding="utf-8")
    (output_dir / "landscape" / "keywords.parquet").write_bytes(b"keyword-data")
    (output_dir / "landscape" / "membership.parquet").write_bytes(b"membership-data")

    monkeypatch.setattr(
        "sciscape.web.app._LOCAL_DATA_ROOTS",
        [tmp_path / "workspace" / "examples_output"],
    )

    client = TestClient(app)
    response = client.get("/api/local-data")

    assert response.status_code == 200
    payload = response.json()
    assert payload["artifacts"]
    data_rows = [row for row in payload["artifacts"] if row["path"].endswith("data.json")]
    assert data_rows
    assert data_rows[0]["has_web_result"] is True
    assert data_rows[0]["has_data_json"] is True
    assert data_rows[0]["has_keywords"] is True
    assert data_rows[0]["has_membership"] is True


def test_local_data_endpoint_prefers_workspace_manifest_results(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    result_root = tmp_path / "registered_results" / "demo"
    landscape = result_root / "landscape"
    report_dir = landscape / "report"
    report_dir.mkdir(parents=True)
    (report_dir / "data.json").write_text(
        json.dumps(
            {
                "0": {
                    "label": "workspace result",
                    "keywords": [{"term": "workspace result"}],
                }
            }
        ),
        encoding="utf-8",
    )
    (result_root / "result_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "sciscape_result_manifest_v1",
                "result_id": "registered_demo",
                "title": "Registered Demo",
            }
        ),
        encoding="utf-8",
    )
    write_workspace_manifest(
        tmp_path,
        workspace_id="workspace_test",
        name="Workspace Test",
        results=[
            {
                "result_id": "registered_demo",
                "path": "registered_results/demo/result_manifest.json",
                "state": "validated",
                "title": "Registered Demo",
            }
        ],
        defaults={"result_id": "registered_demo"},
    )
    monkeypatch.setattr(
        "sciscape.web.app._LOCAL_DATA_ROOTS",
        [tmp_path / "workspace" / "web_output"],
    )

    client = TestClient(app)
    response = client.get("/api/local-data")

    assert response.status_code == 200
    payload = response.json()
    assert payload["workspace"]["state"] == "stable"
    assert payload["discovery_source"] == "workspace_manifest"
    assert len(payload["artifacts"]) == 1
    artifact = payload["artifacts"][0]
    assert artifact["source"] == "workspace"
    assert artifact["workspace_result_id"] == "registered_demo"
    assert artifact["result_title"] == "Registered Demo"
    assert artifact["path"] == "registered_results/demo/landscape/report/data.json"

    open_response = client.post("/api/local-data/open", json={"path": artifact["path"]})
    assert open_response.status_code == 200
    assert open_response.json()["result"]["output_dir"] == str(result_root)


def test_demo_presets_endpoint_finds_timestamped_demo_output(monkeypatch, tmp_path):
    slug = "demo_topic_2020_2024"
    manifest_path = tmp_path / "examples" / "demo_presets.json"
    manifest_path.parent.mkdir()
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "default_output_root": str(tmp_path / "workspace" / "examples_output" / "openalex_live"),
                "presets": {
                    "demo": {
                        "slug": slug,
                        "title": "Demo Topic",
                        "query": "demo topic",
                        "max_works": 1000,
                        "expected_artifacts": [
                            "abstracts.parquet",
                            "edges.parquet",
                            "landscape/membership.parquet",
                            "landscape/keywords.parquet",
                            "landscape/edge_evidence_samples.json",
                            "landscape/report/data.json",
                        ],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "workspace" / "examples_output" / "openalex_live_20260530_010203" / slug
    report_dir = output_dir / "landscape" / "report"
    report_dir.mkdir(parents=True)
    data_json = report_dir / "data.json"
    data_json.write_text("{}", encoding="utf-8")
    (output_dir / "abstracts.parquet").write_bytes(b"abstracts")
    (output_dir / "edges.parquet").write_bytes(b"edges")
    (output_dir / "landscape" / "membership.parquet").write_bytes(b"membership")
    (output_dir / "landscape" / "keywords.parquet").write_bytes(b"keywords")
    (output_dir / "landscape" / "edge_evidence_samples.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr("sciscape.web.app._DEMO_MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(
        "sciscape.web.app._LOCAL_DATA_ROOTS",
        [tmp_path / "workspace" / "examples_output"],
    )

    client = TestClient(app)
    response = client.get("/api/demo-presets")

    assert response.status_code == 200
    payload = response.json()
    demo = payload["demos"][0]
    assert demo["key"] == "demo"
    assert demo["status"] == "available"
    assert demo["can_open"] is True
    assert demo["primary_path"] == str(data_json)
    assert demo["missing_artifacts"] == []


def test_demo_presets_endpoint_reports_missing_output(monkeypatch, tmp_path):
    manifest_path = tmp_path / "examples" / "demo_presets.json"
    manifest_path.parent.mkdir()
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "default_output_root": str(tmp_path / "workspace" / "examples_output" / "openalex_live"),
                "presets": {
                    "demo": {
                        "slug": "missing_demo",
                        "title": "Missing Demo",
                        "expected_artifacts": ["landscape/report/data.json"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("sciscape.web.app._DEMO_MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(
        "sciscape.web.app._LOCAL_DATA_ROOTS",
        [tmp_path / "workspace" / "examples_output"],
    )

    client = TestClient(app)
    response = client.get("/api/demo-presets")

    assert response.status_code == 200
    demo = response.json()["demos"][0]
    assert demo["status"] == "missing"
    assert demo["can_open"] is False
    assert demo["primary_path"] is None
    assert demo["missing_artifacts"] == ["landscape/report/data.json"]


def test_open_local_data_registers_completed_job(monkeypatch, tmp_path):
    output_dir = tmp_path / "workspace" / "web_output" / "demo"
    landscape_dir = output_dir / "landscape"
    report_dir = landscape_dir / "report"
    report_dir.mkdir(parents=True)
    (report_dir / "data.json").write_text(
        json.dumps(
            {
                "0": {
                    "label": "perovskite",
                    "keywords": [{"term": "perovskite"}, {"term": "passivation"}],
                    "network_edges": [{"source": "perovskite", "target": "passivation", "weight": 1}],
                }
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "uid": ["D0", "D1"],
            "title": ["Perovskite passivation", "Stable perovskite device"],
            "abstract": ["Passivation improves stability.", "Perovskite devices are stable."],
            "pubyear": [2021, 2022],
        }
    ).to_parquet(output_dir / "abstracts.parquet", index=False)
    pd.DataFrame({"uid": ["D0", "D1"], "cluster": [0, 0]}).to_parquet(
        landscape_dir / "membership.parquet",
        index=False,
    )
    pd.DataFrame(
        {
            "cluster_id": [0, 0],
            "term": ["perovskite", "passivation"],
            "score": [0.9, 0.8],
        }
    ).to_parquet(landscape_dir / "keywords.parquet", index=False)
    pd.DataFrame({"uid1": ["D0"], "uid2": ["D1"], "rel_sum2": [1.0]}).to_parquet(
        output_dir / "edges.parquet",
        index=False,
    )
    vos_dir = output_dir / "vosviewer"
    vos_dir.mkdir()
    vos_map = vos_dir / "vosviewer_map.txt"
    vos_network = vos_dir / "vosviewer_network.txt"
    vos_map.write_text("id\tlabel\tcluster\nD0\tPerovskite passivation\t0\n", encoding="utf-8")
    vos_network.write_text("source\ttarget\tweight\nD0\tD1\t1.0\n", encoding="utf-8")
    write_export_manifest(
        output_dir,
        export_id="vosviewer_map_network",
        export_family="vosviewer",
        export_kind="vosviewer_map_network",
        primary_file=vos_map,
        source_artifacts=[
            {"artifact_ref": "edges", "artifact_role": "network", "path": output_dir / "edges.parquet"},
            {
                "artifact_ref": "membership",
                "artifact_role": "cluster_membership",
                "path": landscape_dir / "membership.parquet",
            },
        ],
        feature_refs=["cluster_map", "export"],
        files=[
            {
                "file_id": "map",
                "path": vos_map,
                "role": "map",
                "format": "vosviewer_map",
            },
            {
                "file_id": "network",
                "path": vos_network,
                "role": "network",
                "format": "vosviewer_network",
            },
        ],
    )
    (output_dir / "result_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "sciscape_result_manifest_v1",
                "result_id": "curated-web-result",
                "title": "Curated Web Result",
            }
        ),
        encoding="utf-8",
    )
    write_evolution_synthetic_smoke_artifact(output_dir)
    monkeypatch.setattr(
        "sciscape.web.app._LOCAL_DATA_ROOTS",
        [tmp_path / "workspace" / "web_output"],
    )

    client = TestClient(app)
    response = client.post("/api/local-data/open", json={"path": str(report_dir / "data.json")})

    assert response.status_code == 200
    payload = response.json()
    job_id = payload["job_id"]
    job_response = client.get(f"/api/jobs/{job_id}")
    job_payload = job_response.json()

    assert job_payload["status"] == "done"
    assert job_payload["result"]["output_dir"] == str(output_dir)
    assert job_payload["result"]["edges_path"] == str(output_dir / "edges.parquet")
    assert job_payload["result"]["landscape_dir"] == str(landscape_dir)
    assert job_payload["result"]["landscape_rel_path"] == "landscape"
    assert job_payload["result"]["result_state"] == "loaded"
    assert job_payload["result"]["features"]["term_network"] is True
    assert job_payload["result"]["features"]["keyword"] is True
    assert job_payload["result"]["feature_states"]["keyword"] == "stable"
    assert job_payload["result"]["result_manifest"]["schema_version"] == "sciscape_result_manifest_v1"
    assert job_payload["result"]["result_manifest"]["manifest_state"] == "present"
    assert job_payload["result"]["result_manifest"]["title"] == "Curated Web Result"
    assert job_payload["result"]["result_manifest"]["artifacts"]["keywords"]["path"] == "landscape/keywords.parquet"
    exports = job_payload["result"]["result_manifest"]["exports"]
    vos_exports = [row for row in exports if row["export_id"] == "vosviewer_map_network"]
    assert len(vos_exports) == 1
    assert vos_exports[0]["path"] == "vosviewer/vosviewer_map.txt"
    assert vos_exports[0]["export_manifest_ref"] == "exports/vosviewer_map_network/export_manifest.json"
    assert vos_exports[0]["selection_summary"]["view_mode"] == "vosviewer_map_network"
    assert [row["path"] for row in vos_exports[0]["files"]] == [
        "vosviewer/vosviewer_map.txt",
        "vosviewer/vosviewer_network.txt",
    ]
    cooc_exports = [row for row in exports if row["export_id"] == "term_cooccurrence_table"]
    assert len(cooc_exports) == 1
    assert cooc_exports[0]["path"] == "exports/term_cooccurrence_table/term_cooccurrence.tsv"
    assert cooc_exports[0]["export_manifest_ref"] == "exports/term_cooccurrence_table/export_manifest.json"
    assert cooc_exports[0]["selection_summary"]["scope"] == "cooccurrence_artifact"
    assert cooc_exports[0]["selection_summary"]["view_mode"] == "term_cooccurrence_table"
    assert cooc_exports[0]["selection_summary"]["layer_state_keys"] == [
        "map_file",
        "row_count",
        "source_table",
        "table_format",
    ]
    assert [row["path"] for row in cooc_exports[0]["files"]] == [
        "exports/term_cooccurrence_table/term_cooccurrence.tsv",
        "exports/term_cooccurrence_table/term_cooccurrence_map.json",
    ]
    term_vos_exports = [row for row in exports if row["export_id"] == "vosviewer_term_cooccurrence"]
    assert len(term_vos_exports) == 1
    assert term_vos_exports[0]["path"] == "vosviewer/vosviewer_term_map.txt"
    assert term_vos_exports[0]["export_manifest_ref"] == "exports/vosviewer_term_cooccurrence/export_manifest.json"
    assert term_vos_exports[0]["selection_summary"]["scope"] == "cooccurrence_artifact"
    assert term_vos_exports[0]["selection_summary"]["view_mode"] == "vosviewer_term_cooccurrence"
    assert term_vos_exports[0]["selection_summary"]["threshold_keys"] == ["min_link_strength"]
    assert [row["path"] for row in term_vos_exports[0]["files"]] == [
        "vosviewer/vosviewer_term_map.txt",
        "vosviewer/vosviewer_term_network.txt",
    ]
    assert job_payload["result"]["artifact_contract"]["ok"] is True
    assert job_payload["result"]["features"]["evolution"] is True
    assert job_payload["result"]["feature_states"]["evolution"] == "stable"
    assert job_payload["result"]["evolution_summary"]["status"] == "passed"
    assert job_payload["result"]["evolution_summary"]["event_counts"]["continuation"] == 3
    assert job_payload["result"]["atlas"]["node_count"] == 1
    assert job_payload["result"]["atlas"]["nodes"][0]["label"] == "perovskite"
    assert job_payload["result"]["atlas"]["nodes"][0]["doc_count"] == 2
    assert job_payload["result"]["atlas"]["nodes"][0]["doc_count_source"] == "membership:cluster"
    assert job_payload["result"]["atlas"]["nodes"][0]["representative_work_count"] == 2
    assert job_payload["result"]["atlas"]["nodes"][0]["representative_works"][0]["title"] == "Stable perovskite device"
    assert job_payload["result"]["atlas_render_summary"]["schema_version"] == "sciscape_atlas_render_payload_v1"
    assert job_payload["result"]["atlas_render_summary"]["engine_family"] == "deck.gl"
    assert job_payload["result"]["atlas_render_summary"]["node_count"] == 1
    assert job_payload["result"]["atlas_render_summary"]["available_layers"] == [
        "edges",
        "hierarchy",
        "labels",
        "nodes",
    ]
    assert job_payload["result"]["atlas_report_rel_path"] == "landscape/report/data.json"

    render_response = client.get(f"/api/jobs/{job_id}/atlas-render")
    assert render_response.status_code == 200
    render_payload = render_response.json()
    assert render_payload["schema_version"] == "sciscape_atlas_render_payload_v1"
    assert render_payload["view"]["type"] == "OrthographicView"
    assert render_payload["layers"]["nodes"]["recommended_deck_layer"] == "ScatterplotLayer"
    assert render_payload["layers"]["labels"]["rows"][0]["text"] == "perovskite"

    evolution_response = client.get(f"/api/jobs/{job_id}/evolution")
    assert evolution_response.status_code == 200
    evolution = evolution_response.json()
    assert evolution["available"] is True
    assert evolution["status"] == "passed"
    assert evolution["event_counts"]["split"] == 1
    assert evolution["event_counts"]["merge"] == 1
    assert len(evolution["time_slices"]) == 3
    assert len(evolution["events"]) == 8

    export_download = client.get(f"/api/jobs/{job_id}/download/vosviewer/vosviewer_map.txt")
    assert export_download.status_code == 200
    assert "Perovskite passivation" in export_download.text

    cooc_download = client.get(
        f"/api/jobs/{job_id}/download/exports/term_cooccurrence_table/term_cooccurrence.tsv"
    )
    assert cooc_download.status_code == 200
    assert "perovskite" in cooc_download.text
    assert "passivation" in cooc_download.text

    term_vos_download = client.get(f"/api/jobs/{job_id}/download/vosviewer/vosviewer_term_map.txt")
    assert term_vos_download.status_code == 200
    assert "perovskite" in term_vos_download.text
    assert "passivation" in term_vos_download.text

    bundle_download = client.get(f"/api/jobs/{job_id}/download/vosviewer-bundle.zip")
    assert bundle_download.status_code == 200
    bundle_path = output_dir / "exports" / "vosviewer_bundle" / "vosviewer_bundle.zip"
    assert bundle_path.exists()
    with zipfile.ZipFile(bundle_path) as archive:
        names = set(archive.namelist())
        assert {
            "vosviewer/vosviewer_map.txt",
            "vosviewer/vosviewer_network.txt",
            "vosviewer/vosviewer_term_map.txt",
            "vosviewer/vosviewer_term_network.txt",
            "exports/vosviewer_map_network/export_manifest.json",
            "exports/vosviewer_map_network/export_qa.json",
            "exports/vosviewer_term_cooccurrence/export_manifest.json",
            "exports/vosviewer_term_cooccurrence/export_qa.json",
            "vosviewer_bundle_inventory.json",
        }.issubset(names)

    refreshed = client.get(f"/api/jobs/{job_id}").json()
    refreshed_exports = refreshed["result"]["result_manifest"]["exports"]
    bundle_exports = [row for row in refreshed_exports if row["export_id"] == "vosviewer_bundle"]
    assert len(bundle_exports) == 1
    assert bundle_exports[0]["path"] == "exports/vosviewer_bundle/vosviewer_bundle.zip"
    assert bundle_exports[0]["selection_summary"]["view_mode"] == "download_bundle"
    assert bundle_exports[0]["selection_summary"]["filter_count"] == 1


def test_open_local_data_prefers_selected_landscape_variant(monkeypatch, tmp_path):
    output_dir = tmp_path / "workspace" / "examples_output" / "demo"
    default_landscape = output_dir / "landscape"
    selected_landscape = output_dir / "landscape_representative_latest"
    (default_landscape / "report").mkdir(parents=True)
    (selected_landscape / "report").mkdir(parents=True)
    (default_landscape / "report" / "data.json").write_text("{}", encoding="utf-8")
    selected_data = selected_landscape / "report" / "data.json"
    selected_data.write_text("{}", encoding="utf-8")
    (default_landscape / "keywords.parquet").write_bytes(b"default-keywords")
    (selected_landscape / "keywords.parquet").write_bytes(b"selected-keywords")

    monkeypatch.setattr(
        "sciscape.web.app._LOCAL_DATA_ROOTS",
        [tmp_path / "workspace" / "examples_output"],
    )

    client = TestClient(app)
    response = client.post("/api/local-data/open", json={"path": str(selected_data)})

    assert response.status_code == 200
    payload = response.json()
    assert payload["result"]["output_dir"] == str(output_dir)
    assert payload["result"]["landscape_dir"] == str(selected_landscape)
    assert payload["result"]["landscape_rel_path"] == "landscape_representative_latest"


def test_term_network_endpoint_uses_single_cluster_cooccurrence_by_default(tmp_path):
    import polars as pl

    job_id = f"testterm{uuid.uuid4().hex[:8]}"
    output_dir = tmp_path / "demo"
    landscape_dir = output_dir / "landscape"
    landscape_dir.mkdir(parents=True)
    pl.DataFrame(
        {
            "cluster_id": [0, 0, 1, 1],
            "display_label": ["alpha beta", "gamma delta", "alpha beta", "epsilon zeta"],
            "score": [0.9, 0.8, 0.7, 0.6],
        }
    ).write_parquet(landscape_dir / "keywords.parquet")

    web_app._jobs.create(job_id, {"query": "term network test"})
    job = web_app._jobs.get(job_id)
    assert job is not None
    job["status"] = "done"
    job["progress"] = []
    job["result"] = {"output_dir": str(output_dir), "landscape_dir": str(landscape_dir)}
    web_app._jobs.persist(job_id)

    client = TestClient(app)
    response = client.get(f"/api/jobs/{job_id}/term-network?top_k=2")

    assert response.status_code == 200
    payload = response.json()
    assert "error" not in payload
    assert len(payload["nodes"]) == 3
    assert len(payload["edges"]) == 2
    assert {edge["weight"] for edge in payload["edges"]} == {1}


def test_open_local_data_rejects_paths_outside_allowed_roots(monkeypatch, tmp_path):
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside" / "data.json"
    allowed.mkdir()
    outside.parent.mkdir()
    outside.write_text("{}", encoding="utf-8")
    monkeypatch.setattr("sciscape.web.app._LOCAL_DATA_ROOTS", [allowed])

    client = TestClient(app)
    response = client.post("/api/local-data/open", json={"path": str(outside)})

    assert response.status_code == 400
    assert response.json()["detail"] == "local data path is outside allowed roots"
