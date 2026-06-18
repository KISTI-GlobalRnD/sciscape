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
    write_cluster_review_packet_artifact,
    write_cooccurrence_artifacts,
    write_evolution_artifacts,
    write_evolution_synthetic_smoke_artifact,
    write_export_manifest,
    write_narrative_evidence_artifacts,
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


def _write_cluster_sharded_run_sidecars(output_dir: Path) -> Path:
    run_dir = output_dir / "landscape" / "keyword_cluster_sharded" / "full_run"
    candidate_dir = run_dir / "candidates"
    candidate_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "sciscape_keyword_cluster_shards_v1",
                "created_at_utc": "2026-06-02T01:00:00+00:00",
                "total_clusters": 3,
                "total_docs": 300,
                "shards": [
                    {"shard_id": 0, "cluster_ids": [0], "doc_count": 100},
                    {"shard_id": 1, "cluster_ids": [1], "doc_count": 100},
                    {"shard_id": 2, "cluster_ids": [2], "doc_count": 100},
                ],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "progress.json").write_text(
        json.dumps(
            {
                "updated_at_utc": "2026-06-02T01:05:00+00:00",
                "stage": "candidate_mining",
                "processed": 1,
                "total": 3,
                "percent": 33.3,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "preflight_summary.json").write_text(
        json.dumps(
            {
                "schema_version": "sciscape_keyword_cluster_sharded_preflight_v1",
                "status": "ok",
                "shard_count": 3,
                "abstract_path": str(output_dir / "abstracts.parquet"),
                "membership_path": str(output_dir / "landscape" / "membership.parquet"),
                "cluster_level": "cluster",
            }
        ),
        encoding="utf-8",
    )
    candidate_path = candidate_dir / "candidate_shard_0000.parquet"
    candidate_path.write_bytes(b"placeholder")
    (candidate_dir / "candidate_shard_0000.done.json").write_text(
        json.dumps(
            {
                "schema_version": "sciscape_keyword_candidate_shard_done_v1",
                "status": "complete",
                "shard_id": 0,
                "rows": 12,
                "source_rows": 100,
                "path": str(candidate_path),
            }
        ),
        encoding="utf-8",
    )
    (candidate_dir / "candidate_shard_0002.progress.json").write_text(
        json.dumps(
            {
                "schema_version": "sciscape_keyword_candidate_shard_progress_v1",
                "status": "failed",
                "shard_id": 2,
                "rows_processed": 14,
                "rows_total": 100,
                "output_path": str(candidate_dir / "candidate_shard_0002.parquet"),
            }
        ),
        encoding="utf-8",
    )
    return run_dir


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
    assert "renderAtlasNarrativeReview" in response.text
    assert "atlas-narrative-panel" in response.text
    assert "submitAtlasNarrativeReview" in response.text
    assert "/narrative/review" in response.text
    assert "renderAtlasReviewQueue" in response.text
    assert "atlasReviewQueueRows" in response.text
    assert "atlasFilteredReviewRows" in response.text
    assert "selectAtlasReviewFilter" in response.text
    assert "openNextAtlasReviewTarget" in response.text
    assert "manifestExportCards" in response.text
    assert "manifestExportRow" in response.text
    assert "manifestExportSelectionMeta" in response.text
    assert "manifestMatrixArtifactCards" in response.text
    assert "matrixArtifactLabel" in response.text
    assert "matrix_values" in response.text
    assert "Matrix sparse triplets" in response.text
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
    assert "Narrative review" in response.text
    assert "not_required" in response.text
    assert "currentAtlasNarrativeReviewStatus" in response.text
    assert "renderAtlasNarrativeReviewStatus" in response.text
    assert "review saved:" in response.text
    assert "review failed:" in response.text
    assert "latest review:" in response.text
    assert "atlas inspector review action:" in response.text
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
    assert "/atlas-render/summary" in response.text
    assert "/atlas-render/layers/" in response.text
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
    assert "renderEvolutionMap" in response.text
    assert "renderEvolutionStateDetail" in response.text
    assert "renderEvolutionMatchingDiagnostics" in response.text
    assert "Matching Diagnostics" in response.text
    assert "candidate transitions" in response.text
    assert "slice pair diagnostics" in response.text
    assert "renderEvolutionLoadControls" in response.text
    assert "selectEvolutionLoadProfile" in response.text
    assert "evolutionLoadProfileQuery" in response.text
    assert "evolution_profile" in response.text
    assert "loaded within limits" in response.text
    assert "evolutionCoverageLabels" in response.text
    assert "membership" in response.text
    assert "hidden states" in response.text
    assert "selectEvolutionState" in response.text
    assert "currentEvolutionSelectedStateId" in response.text
    assert "applyEvolutionUrlState" in response.text
    assert "syncEvolutionUrlState" in response.text
    assert "evolution_state" in response.text
    assert "evolution_event" in response.text
    assert "evolution_focus" in response.text
    assert "renderEvolutionFocusControls" in response.text
    assert "selectEvolutionFocusMode" in response.text
    assert "clearEvolutionSelection" in response.text
    assert "Clear selection" in response.text
    assert "focusedEvolutionMap" in response.text
    assert "evolution-map-edge.selected" in response.text
    assert 'role="button" aria-label="' in response.text
    assert '" onkeydown="' in response.text
    assert "evolutionActivateOnKey" in response.text
    assert "evolutionKeySelectHandler" in response.text
    assert "aria-label" in response.text
    assert "pointer-events: stroke" in response.text
    assert "evolution-event-row:focus" in response.text
    assert "evolution-state-row:focus" in response.text
    assert "Neighborhood" in response.text
    assert "evolution-map-panel" in response.text
    assert "evolution-detail-grid" in response.text
    assert "Select a state node" in response.text
    assert "selectEvolutionEventFilter" in response.text
    assert "/api/jobs/${currentJobId}/evolution" in response.text
    assert "evolution/time_slices.parquet" in response.text
    assert "evolution/cluster_states.parquet" in response.text
    assert "evolution/state_membership.parquet" in response.text
    assert "evolutionHasStateMembership" in response.text
    assert "state_membership_summary" in response.text
    assert "loaded links" in response.text
    assert "evolution/transitions.parquet" in response.text
    assert "evolution/lineages.parquet" in response.text
    assert "evolution/evolution_qa.json" in response.text
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
    assert "/api/jobs/' + encodeURIComponent(jobId) + '/retry" in response.text
    assert "Retry Query" in response.text
    assert "/api/jobs/' + encodeURIComponent(jobId) + '/' + endpoint" in response.text
    assert "resume-failed-shards" in response.text
    assert "jobRunStateUrl" in response.text
    assert "/run-state" in response.text
    assert "Resume In App" in response.text
    assert "Rerun Failed Shards" in response.text
    assert "sciscape_run_state_summary_v1" in response.text
    assert "renderRunStateSummary" in response.text
    assert "resumeFailedShardsCurrentJob" in response.text
    assert "run-state-chip" in response.text
    assert 'id="btn-cancel"' in response.text
    assert "/api/jobs/' + encodeURIComponent(jobId) + '/cancel" in response.text
    assert "Cancel Query" in response.text
    assert "fetch('/api/demo-presets'" in response.text
    assert "fetch('/api/local-data" in response.text
    assert "evolution_status" in response.text
    assert "evolutionCounts.events" in response.text
    assert "new URLSearchParams(window.location.search).get('job')" in response.text
    assert "currentShareUrl" in response.text
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


def test_safe_json_response_sanitizes_non_finite_values():
    payload = json.loads(web_app.SafeJSONResponse({"x": float("nan"), "items": [float("inf")]}).body)

    assert payload == {"x": None, "items": [None]}


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


def test_retry_endpoint_enqueues_previous_openalex_request(monkeypatch):
    def fake_run_job(job_id, req):
        job = web_app._jobs[job_id]
        job["status"] = "done"
        job["progress"].append(f"retried pipeline: {req.query}")
        job["result"] = {
            "n_works": req.max_works,
            "n_edges": {"dc": 0, "bc": 0, "cc": 0},
            "output_dir": "workspace/web_output/retry",
            "abstracts_path": None,
            "edges_path": None,
            "landscape_dir": None,
        }
        web_app._jobs.persist(job_id)

    monkeypatch.setattr("sciscape.web.app._run_job", fake_run_job)
    source_id = "failedjob"
    web_app._jobs.create(
        source_id,
        {
            "query": "graph neural networks",
            "years": "2020-2024",
            "max_works": 12,
            "edge_types": "dc,bc",
            "run_landscape": True,
        },
    )
    source_job = web_app._jobs[source_id]
    source_job["status"] = "error"
    source_job["progress"] = ["ERROR: previous failure"]
    source_job["result"] = {"error": "previous failure"}
    web_app._jobs.persist(source_id)

    client = TestClient(app)
    response = client.post(f"/api/jobs/{source_id}/retry")

    assert response.status_code == 200
    payload = response.json()
    retry_id = payload["job_id"]
    assert payload["retry_of"] == source_id
    assert payload["request"]["retry_of"] == source_id
    assert payload["request"]["source_type"] == "openalex_query"
    assert retry_id != source_id

    retry_payload = client.get(f"/api/jobs/{retry_id}").json()
    assert retry_payload["status"] == "done"
    assert retry_payload["progress"] == [
        f"Retry of job {source_id}",
        "retried pipeline: graph neural networks",
    ]
    assert retry_payload["result"]["n_works"] == 12


def test_retry_endpoint_rejects_local_result_jobs():
    source_id = "localresult"
    web_app._jobs.create(source_id, {"query": "Local output: workspace/examples_output/demo"})
    source_job = web_app._jobs[source_id]
    source_job["status"] = "done"
    source_job["progress"] = ["Loaded local SciScape output"]
    source_job["result"] = {"output_dir": "workspace/examples_output/demo"}
    web_app._jobs.persist(source_id)

    client = TestClient(app)
    response = client.post(f"/api/jobs/{source_id}/retry")

    assert response.status_code == 400
    assert response.json()["detail"] == "job does not contain a replayable OpenAlex query request"


def test_safe_resume_argv_accepts_cluster_sharded_keyword_resume():
    command = (
        "sciscape keywords abstracts.parquet landscape/membership.parquet "
        "--keyword-engine cluster_sharded "
        "--cluster-level cluster "
        "--cluster-sharded-output-dir landscape/keyword_cluster_sharded/full_run "
        "--progress-path landscape/keyword_cluster_sharded/full_run/progress.json "
        "-o landscape/keyword_cluster_sharded/full_run/keywords.parquet "
        "--scoring-shard-resume"
    )

    argv = web_app._safe_resume_argv(command)

    assert argv[:3] == ["keywords", "abstracts.parquet", "landscape/membership.parquet"]
    assert "--keyword-engine" in argv
    assert "cluster_sharded" in argv
    assert "--scoring-shard-resume" in argv

    shard_argv = web_app._safe_resume_argv(command + " --cluster-sharded-shard-ids 2,4")
    assert "--cluster-sharded-shard-ids" in shard_argv
    assert shard_argv[shard_argv.index("--cluster-sharded-shard-ids") + 1] == "2,4"

    scheduled_argv = web_app._safe_resume_argv(command + " --n-jobs 2 --parallel-backend threading")
    assert scheduled_argv[scheduled_argv.index("--n-jobs") + 1] == "2"
    assert scheduled_argv[scheduled_argv.index("--parallel-backend") + 1] == "threading"


def test_safe_resume_argv_rejects_non_keyword_commands():
    with pytest.raises(web_app.HTTPException) as exc:
        web_app._safe_resume_argv("sciscape query graphene --max-works 10")

    assert exc.value.status_code == 400
    assert exc.value.detail == "unsupported resume command"

    with pytest.raises(web_app.HTTPException) as bad_shards:
        web_app._safe_resume_argv(
            "sciscape keywords abstracts.parquet membership.parquet "
            "--keyword-engine cluster_sharded "
            "--cluster-sharded-output-dir out "
            "--progress-path out/progress.json "
            "-o out/keywords.parquet "
            "--scoring-shard-resume "
            "--cluster-sharded-shard-ids a,b"
        )
    assert bad_shards.value.status_code == 400
    assert bad_shards.value.detail == "invalid shard ID list"

    with pytest.raises(web_app.HTTPException) as bad_n_jobs:
        web_app._safe_resume_argv(
            "sciscape keywords abstracts.parquet membership.parquet "
            "--keyword-engine cluster_sharded "
            "--cluster-sharded-output-dir out "
            "--progress-path out/progress.json "
            "-o out/keywords.parquet "
            "--scoring-shard-resume "
            "--n-jobs 0"
        )
    assert bad_n_jobs.value.status_code == 400
    assert bad_n_jobs.value.detail == "n_jobs must be between 1 and 64"

    with pytest.raises(web_app.HTTPException) as bad_backend:
        web_app._safe_resume_argv(
            "sciscape keywords abstracts.parquet membership.parquet "
            "--keyword-engine cluster_sharded "
            "--cluster-sharded-output-dir out "
            "--progress-path out/progress.json "
            "-o out/keywords.parquet "
            "--scoring-shard-resume "
            "--parallel-backend forkbomb"
        )
    assert bad_backend.value.status_code == 400
    assert bad_backend.value.detail == "invalid parallel_backend"


def test_resume_endpoint_enqueues_validated_cli_resume_job(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    output_dir = tmp_path / "workspace" / "examples_output" / "resume-result"
    run_dir = output_dir / "landscape" / "keyword_cluster_sharded" / "full_run"
    output_dir.mkdir(parents=True)
    run_dir.mkdir(parents=True)
    command = (
        f"sciscape keywords {output_dir / 'abstracts.parquet'} "
        f"{output_dir / 'landscape' / 'membership.parquet'} "
        "--keyword-engine cluster_sharded "
        f"--cluster-sharded-output-dir {run_dir} "
        f"--progress-path {run_dir / 'progress.json'} "
        f"-o {run_dir / 'keywords.parquet'} "
        "--scoring-shard-resume"
    )
    source_id = "resumejob"
    web_app._jobs.create(source_id, {"query": "Local output: resume-result"})
    source_job = web_app._jobs[source_id]
    source_job["status"] = "error"
    source_job["progress"] = ["keyword extraction failed"]
    source_job["result"] = {
        "output_dir": str(output_dir),
        "run_state": {
            "status": "failed",
            "resume": {"supported": True, "command": command},
        },
    }
    web_app._jobs.persist(source_id)
    calls = []

    def fake_run_resume(argv, progress):
        calls.append(argv)
        progress("fake resume command ran")

    def fake_refresh(job_id, result, root, *, mode="local_result"):
        result["result_manifest"] = {"schema_version": "test_manifest"}
        result["run_state"] = {"status": "complete", "resume": {"supported": False, "command": None}}
        result["feature_states"] = {"keyword": "stable"}
        result["features"] = {"keyword": True}
        result["result_state"] = "loaded"
        job = web_app._jobs[job_id]
        job["result"] = result
        web_app._jobs.persist(job_id)

    monkeypatch.setattr(web_app, "_run_sciscape_resume_command", fake_run_resume)
    monkeypatch.setattr(web_app, "_refresh_job_result_manifest", fake_refresh)

    client = TestClient(app)
    response = client.post(f"/api/jobs/{source_id}/resume")

    assert response.status_code == 200
    payload = response.json()
    resume_id = payload["job_id"]
    assert payload["resume_of"] == source_id
    assert payload["argv"][0] == "keywords"
    assert calls == [payload["argv"]]

    resume_payload = client.get(f"/api/jobs/{resume_id}").json()
    assert resume_payload["status"] == "done"
    assert resume_payload["progress"] == [
        f"Resume of job {source_id}",
        "Resume job started.",
        "fake resume command ran",
        "Resume command complete.",
    ]
    assert resume_payload["result"]["resume_of"] == source_id
    assert resume_payload["result"]["result_manifest"]["schema_version"] == "test_manifest"
    assert resume_payload["result"]["run_state"]["status"] == "complete"
    status_path = tmp_path / "workspace" / "web_output" / resume_id / "job_status.json"
    status_payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert status_payload["schema_version"] == "sciscape_resume_job_status_v1"
    assert status_payload["request"]["source_type"] == "sciscape_cli_resume"
    assert status_payload["run_state"]["status"] == "complete"


def test_resume_failed_shards_endpoint_enqueues_selected_shard_resume(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    output_dir = tmp_path / "workspace" / "examples_output" / "resume-result"
    run_dir = output_dir / "landscape" / "keyword_cluster_sharded" / "full_run"
    output_dir.mkdir(parents=True)
    run_dir.mkdir(parents=True)
    command = (
        f"sciscape keywords {output_dir / 'abstracts.parquet'} "
        f"{output_dir / 'landscape' / 'membership.parquet'} "
        "--keyword-engine cluster_sharded "
        f"--cluster-sharded-output-dir {run_dir} "
        f"--progress-path {run_dir / 'progress.json'} "
        f"-o {run_dir / 'keywords.parquet'} "
        "--scoring-shard-resume"
    )
    source_id = "failedshards"
    web_app._jobs.create(source_id, {"query": "Local output: resume-result"})
    source_job = web_app._jobs[source_id]
    source_job["status"] = "error"
    source_job["progress"] = ["keyword extraction failed"]
    source_job["result"] = {
        "output_dir": str(output_dir),
        "run_state": {
            "status": "failed",
            "failure": {"reason": "2 shards failed", "failed_shards": [4, 2]},
            "resume": {"supported": True, "command": command},
        },
    }
    web_app._jobs.persist(source_id)
    calls = []

    def fake_run_resume(argv, progress):
        calls.append(argv)
        progress("fake failed-shard resume command ran")

    monkeypatch.setattr(web_app, "_run_sciscape_resume_command", fake_run_resume)

    client = TestClient(app)
    response = client.post(f"/api/jobs/{source_id}/resume-failed-shards")

    assert response.status_code == 200
    payload = response.json()
    assert payload["resume_of"] == source_id
    assert payload["shard_ids"] == [2, 4]
    assert "--cluster-sharded-shard-ids" in payload["argv"]
    assert payload["argv"][payload["argv"].index("--cluster-sharded-shard-ids") + 1] == "2,4"
    assert calls == [payload["argv"]]

    resume_payload = client.get(f"/api/jobs/{payload['job_id']}").json()
    assert resume_payload["status"] == "done"
    stored_request = web_app._jobs[payload["job_id"]]["request"]
    assert stored_request["resume_scope"] == "failed_shards"
    assert stored_request["shard_ids"] == [2, 4]
    assert resume_payload["progress"][0] == f"Resume failed shards [2, 4] of job {source_id}"


def test_resume_shards_endpoint_enqueues_user_selected_shard_resume(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    output_dir = tmp_path / "workspace" / "examples_output" / "resume-result"
    run_dir = output_dir / "landscape" / "keyword_cluster_sharded" / "full_run"
    output_dir.mkdir(parents=True)
    run_dir.mkdir(parents=True)
    command = (
        f"sciscape keywords {output_dir / 'abstracts.parquet'} "
        f"{output_dir / 'landscape' / 'membership.parquet'} "
        "--keyword-engine cluster_sharded "
        f"--cluster-sharded-output-dir {run_dir} "
        f"--progress-path {run_dir / 'progress.json'} "
        f"-o {run_dir / 'keywords.parquet'} "
        "--scoring-shard-resume"
    )
    source_id = "selectedshards"
    web_app._jobs.create(source_id, {"query": "Local output: resume-result"})
    source_job = web_app._jobs[source_id]
    source_job["status"] = "error"
    source_job["progress"] = ["keyword extraction stopped"]
    source_job["result"] = {
        "output_dir": str(output_dir),
        "run_state": {
            "status": "partial",
            "shards": {"total": 8, "complete": 5, "failed": 0, "running": 0},
            "resume": {"supported": True, "command": command},
        },
    }
    web_app._jobs.persist(source_id)
    calls = []

    def fake_run_resume(argv, progress):
        calls.append(argv)
        progress("fake selected-shard resume command ran")

    monkeypatch.setattr(web_app, "_run_sciscape_resume_command", fake_run_resume)

    client = TestClient(app)
    response = client.post(
        f"/api/jobs/{source_id}/resume-shards",
        json={"shard_ids": [5, 1, 5], "n_jobs": 2, "parallel_backend": "threading"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["resume_of"] == source_id
    assert payload["shard_ids"] == [1, 5]
    assert "--cluster-sharded-shard-ids" in payload["argv"]
    assert payload["argv"][payload["argv"].index("--cluster-sharded-shard-ids") + 1] == "1,5"
    assert payload["argv"][payload["argv"].index("--n-jobs") + 1] == "2"
    assert payload["argv"][payload["argv"].index("--parallel-backend") + 1] == "threading"
    assert payload["schedule_options"] == {"n_jobs": 2, "parallel_backend": "threading"}
    assert calls == [payload["argv"]]

    resume_payload = client.get(f"/api/jobs/{payload['job_id']}").json()
    assert resume_payload["status"] == "done"
    stored_request = web_app._jobs[payload["job_id"]]["request"]
    assert stored_request["resume_scope"] == "selected_shards"
    assert stored_request["shard_ids"] == [1, 5]
    assert stored_request["schedule_options"] == {"n_jobs": 2, "parallel_backend": "threading"}
    assert resume_payload["progress"][0] == f"Resume selected shards [1, 5] of job {source_id}"

    status_payload = json.loads(
        (
            tmp_path
            / "workspace"
            / "web_output"
            / payload["job_id"]
            / "job_status.json"
        ).read_text(encoding="utf-8")
    )
    assert status_payload["request"]["schedule_options"] == {"n_jobs": 2, "parallel_backend": "threading"}


def test_resume_shards_endpoint_rejects_out_of_range_shards(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    output_dir = tmp_path / "workspace" / "examples_output" / "resume-result"
    run_dir = output_dir / "landscape" / "keyword_cluster_sharded" / "full_run"
    output_dir.mkdir(parents=True)
    run_dir.mkdir(parents=True)
    command = (
        f"sciscape keywords {output_dir / 'abstracts.parquet'} "
        f"{output_dir / 'landscape' / 'membership.parquet'} "
        "--keyword-engine cluster_sharded "
        f"--cluster-sharded-output-dir {run_dir} "
        f"--progress-path {run_dir / 'progress.json'} "
        f"-o {run_dir / 'keywords.parquet'} "
        "--scoring-shard-resume"
    )
    source_id = "selectedshardsbad"
    web_app._jobs.create(source_id, {"query": "Local output: resume-result"})
    source_job = web_app._jobs[source_id]
    source_job["status"] = "error"
    source_job["result"] = {
        "output_dir": str(output_dir),
        "run_state": {
            "status": "partial",
            "shards": {"total": 3, "complete": 1, "failed": 0, "running": 0},
            "resume": {"supported": True, "command": command},
        },
    }
    web_app._jobs.persist(source_id)

    client = TestClient(app)
    response = client.post(f"/api/jobs/{source_id}/resume-shards", json={"shard_ids": [3]})

    assert response.status_code == 400
    assert response.json()["detail"] == "shard IDs outside available range 0-2: [3]"


def test_resume_shards_endpoint_rejects_invalid_schedule_options(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    output_dir = tmp_path / "workspace" / "examples_output" / "resume-result"
    run_dir = output_dir / "landscape" / "keyword_cluster_sharded" / "full_run"
    output_dir.mkdir(parents=True)
    run_dir.mkdir(parents=True)
    command = (
        f"sciscape keywords {output_dir / 'abstracts.parquet'} "
        f"{output_dir / 'landscape' / 'membership.parquet'} "
        "--keyword-engine cluster_sharded "
        f"--cluster-sharded-output-dir {run_dir} "
        f"--progress-path {run_dir / 'progress.json'} "
        f"-o {run_dir / 'keywords.parquet'} "
        "--scoring-shard-resume"
    )
    source_id = "selectedshardsbadschedule"
    web_app._jobs.create(source_id, {"query": "Local output: resume-result"})
    source_job = web_app._jobs[source_id]
    source_job["status"] = "error"
    source_job["result"] = {
        "output_dir": str(output_dir),
        "run_state": {
            "status": "partial",
            "shards": {"total": 3, "complete": 1, "failed": 0, "running": 0},
            "resume": {"supported": True, "command": command},
        },
    }
    web_app._jobs.persist(source_id)

    client = TestClient(app)
    response = client.post(
        f"/api/jobs/{source_id}/resume-shards",
        json={"shard_ids": [1], "n_jobs": 0},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "n_jobs must be between 1 and 64"


def test_resume_endpoint_rejects_unsupported_resume_command():
    source_id = "badresume"
    web_app._jobs.create(source_id, {"query": "Local output: bad"})
    source_job = web_app._jobs[source_id]
    source_job["status"] = "error"
    source_job["progress"] = ["failed"]
    source_job["result"] = {
        "output_dir": "workspace/examples_output/bad",
        "run_state": {
            "status": "failed",
            "resume": {
                "supported": True,
                "command": "sciscape query graphene --max-works 10",
            },
        },
    }
    web_app._jobs.persist(source_id)

    client = TestClient(app)
    response = client.post(f"/api/jobs/{source_id}/resume")

    assert response.status_code == 400
    assert response.json()["detail"] == "unsupported resume command"


def test_cancel_endpoint_records_cancel_request(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    source_id = "runningjob"
    request_payload = {
        "query": "graph neural networks",
        "years": "2020-2024",
        "max_works": 12,
        "edge_types": "dc,bc",
        "run_landscape": True,
    }
    web_app._jobs.create(source_id, request_payload)
    job = web_app._jobs[source_id]
    job["status"] = "running"
    job["started_at_utc"] = "2026-06-02T00:00:00+00:00"
    job["progress"] = ["Querying OpenAlex"]
    web_app._jobs.persist(source_id)

    client = TestClient(app)
    response = client.post(f"/api/jobs/{source_id}/cancel")

    assert response.status_code == 200
    payload = response.json()
    assert payload["cancel_requested"] is True
    assert payload["cancel_requested_at_utc"]
    updated = web_app._jobs[source_id]
    assert updated["status"] == "running"
    assert updated["cancel_requested_at_utc"] == payload["cancel_requested_at_utc"]
    assert updated["progress"][-1] == "Cancellation requested."

    output_dir = tmp_path / "workspace" / "web_output" / source_id
    status_payload = json.loads((output_dir / "job_status.json").read_text(encoding="utf-8"))
    assert status_payload["status"] == "running"
    assert status_payload["cancel_requested_at_utc"] == payload["cancel_requested_at_utc"]
    assert status_payload["run_state"]["cancel_requested_at_utc"] == payload["cancel_requested_at_utc"]

    list_payload = client.get("/api/jobs").json()
    row = next(item for item in list_payload if item["job_id"] == source_id)
    assert row["cancel_requested"] is True


def test_run_job_honors_cancel_request_at_progress_boundary(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    def fake_openalex_pipeline(config):
        config.progress("fetched first page")
        job = web_app._jobs[job_id]
        job["cancel_requested_at_utc"] = web_app._utc_now()
        config.progress("this should not be recorded")
        raise AssertionError("pipeline should stop at cancellation boundary")

    monkeypatch.setattr("sciscape.openalex.run_openalex_pipeline", fake_openalex_pipeline)

    req = web_app.QueryRequest(
        query="graph neural networks",
        years="2020-2024",
        max_works=12,
        run_landscape=False,
    )
    job_id = "canceljob"
    web_app._jobs.create(job_id, req.model_dump())

    web_app._run_job(job_id, req)

    job = web_app._jobs[job_id]
    output_dir = tmp_path / "workspace" / "web_output" / job_id
    status_payload = json.loads((output_dir / "job_status.json").read_text(encoding="utf-8"))
    manifest_payload = json.loads((output_dir / "result_manifest.json").read_text(encoding="utf-8"))

    assert job["status"] == "cancelled"
    assert job["result"]["cancelled"] is True
    assert job["result"]["run_state"]["status"] == "cancelled"
    assert job["progress"] == ["fetched first page", "Job cancelled."]
    assert status_payload["status"] == "cancelled"
    assert status_payload["run_state"]["status"] == "cancelled"
    assert status_payload["run_state"]["failure"] is None
    assert status_payload["run_state"]["cancellation"]["reason"] == "Cancellation requested"
    assert manifest_payload["run_state"]["status"] == "cancelled"


def test_run_job_honors_cancel_request_at_pipeline_checkpoint(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    def fake_openalex_pipeline(config):
        job = web_app._jobs[job_id]
        job["cancel_requested_at_utc"] = web_app._utc_now()
        config.checkpoint()
        raise AssertionError("pipeline should stop at cancellation checkpoint")

    monkeypatch.setattr("sciscape.openalex.run_openalex_pipeline", fake_openalex_pipeline)

    req = web_app.QueryRequest(
        query="graph neural networks",
        years="2020-2024",
        max_works=12,
        run_landscape=False,
    )
    job_id = "checkpointcanceljob"
    web_app._jobs.create(job_id, req.model_dump())

    web_app._run_job(job_id, req)

    job = web_app._jobs[job_id]
    output_dir = tmp_path / "workspace" / "web_output" / job_id
    status_payload = json.loads((output_dir / "job_status.json").read_text(encoding="utf-8"))

    assert job["status"] == "cancelled"
    assert job["result"]["cancelled"] is True
    assert job["progress"] == ["Job cancelled."]
    assert status_payload["run_state"]["status"] == "cancelled"
    assert status_payload["run_state"]["cancellation"]["reason"] == "Cancellation requested"


def test_run_job_passes_openalex_budget_controls(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    captured = {}

    def fake_openalex_pipeline(config):
        captured["api_attempt_budget"] = config.api_attempt_budget
        captured["retry_wait_budget_seconds"] = config.retry_wait_budget_seconds
        captured["interruptible_requests"] = config.interruptible_requests
        captured["request_poll_interval"] = config.request_poll_interval
        return SimpleNamespace(
            n_works=0,
            n_edges={},
            abstracts_path=None,
            edges_path=None,
            landscape_dir=None,
            api_telemetry=None,
        )

    monkeypatch.setattr("sciscape.openalex.run_openalex_pipeline", fake_openalex_pipeline)

    req = web_app.QueryRequest(
        query="graph neural networks",
        max_works=450,
        run_landscape=False,
        retry_wait_budget_seconds=12,
    )
    job_id = "budgetjob"
    web_app._jobs.create(job_id, req.model_dump())

    web_app._run_job(job_id, req)

    assert captured["api_attempt_budget"] == 16
    assert captured["retry_wait_budget_seconds"] == 12
    assert captured["interruptible_requests"] is True
    assert captured["request_poll_interval"] == 0.25


def test_run_job_uses_explicit_openalex_attempt_budget(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    captured = {}

    def fake_openalex_pipeline(config):
        captured["api_attempt_budget"] = config.api_attempt_budget
        return SimpleNamespace(
            n_works=0,
            n_edges={},
            abstracts_path=None,
            edges_path=None,
            landscape_dir=None,
            api_telemetry=None,
        )

    monkeypatch.setattr("sciscape.openalex.run_openalex_pipeline", fake_openalex_pipeline)

    req = web_app.QueryRequest(
        query="graph neural networks",
        max_works=450,
        run_landscape=False,
        api_attempt_budget=9,
    )
    job_id = "explicitbudgetjob"
    web_app._jobs.create(job_id, req.model_dump())

    web_app._run_job(job_id, req)

    assert captured["api_attempt_budget"] == 9


def test_run_job_preserves_openalex_budget_abort_telemetry_on_error(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    def fake_openalex_pipeline(config):
        config.api_telemetry(
            {
                "source": "openalex",
                "attempts_total": 1,
                "api_attempt_budget": 1,
                "quota_budget_exceeded": True,
                "quota_abort_reason": "OpenAlex API attempt budget exceeded: 1/1",
            }
        )
        raise RuntimeError("OpenAlex API attempt budget exceeded: 1/1")

    monkeypatch.setattr("sciscape.openalex.run_openalex_pipeline", fake_openalex_pipeline)

    req = web_app.QueryRequest(
        query="graph neural networks",
        max_works=450,
        run_landscape=False,
        api_attempt_budget=1,
    )
    job_id = "budgetabortjob"
    web_app._jobs.create(job_id, req.model_dump())

    web_app._run_job(job_id, req)

    output_dir = tmp_path / "workspace" / "web_output" / job_id
    status_payload = json.loads((output_dir / "job_status.json").read_text(encoding="utf-8"))
    manifest_payload = json.loads((output_dir / "result_manifest.json").read_text(encoding="utf-8"))
    job = web_app._jobs[job_id]

    assert job["status"] == "error"
    assert job["result"]["error"] == "OpenAlex API attempt budget exceeded: 1/1"
    assert status_payload["run_state"]["api_telemetry"]["quota_budget_exceeded"] is True
    assert "attempt budget exceeded" in status_payload["run_state"]["api_telemetry"]["quota_abort_reason"]
    assert manifest_payload["run_state"]["api_telemetry"]["api_attempt_budget"] == 1
    assert manifest_payload["source"]["api_telemetry"]["quota_budget_exceeded"] is True


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
        api_telemetry = {
            "source": "openalex",
            "attempts_total": 2,
            "successful_requests_total": 1,
            "retry_attempts_total": 1,
            "status_counts": {"429": 1, "200": 1},
        }
        pd.DataFrame(
            {
                "uid": ["W1"],
                "title": ["Graph neural network survey"],
                "abstract": ["Graph neural networks for citation analysis."],
                "pubyear": [2024],
            }
        ).to_parquet(abstracts_path, index=False)
        config.api_telemetry(api_telemetry)
        config.progress("fetched works")
        return SimpleNamespace(
            n_works=1,
            n_edges={"dc": 0, "bc": 0, "cc": 0},
            abstracts_path=abstracts_path,
            edges_path=None,
            landscape_dir=None,
            api_telemetry=api_telemetry,
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
    assert job["result"]["api_telemetry"]["attempts_total"] == 2
    assert job["result"]["run_state"]["status"] == "complete"
    assert job["result"]["run_state"]["api_telemetry"]["retry_attempts_total"] == 1
    assert status_payload["schema_version"] == "sciscape_live_job_status_v1"
    assert status_payload["status"] == "done"
    assert status_payload["api_telemetry"]["status_counts"] == {"429": 1, "200": 1}
    assert status_payload["run_state"]["status"] == "complete"
    assert status_payload["run_state"]["api_telemetry"]["attempts_total"] == 2
    assert status_payload["started_at_utc"]
    assert status_payload["finished_at_utc"]
    assert status_payload["updated_at_utc"]
    assert status_payload["progress"] == ["fetched works"]
    assert manifest_payload["schema_version"] == "sciscape_result_manifest_v1"
    assert manifest_payload["source"]["query"] == "graph neural networks"
    assert manifest_payload["source"]["filters"] == {"publication_year": "2020-2024"}
    assert manifest_payload["source"]["api_telemetry"]["attempts_total"] == 2
    assert manifest_payload["run_state"]["status"] == "complete"
    assert manifest_payload["run_state"]["api_telemetry"]["retry_attempts_total"] == 1
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
    write_evolution_synthetic_smoke_artifact(output_dir)

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
    assert data_rows[0]["has_evolution"] is True
    assert data_rows[0]["evolution_status"] == "passed"
    assert data_rows[0]["evolution_counts"]["events"] == 8
    assert data_rows[0]["evolution_counts"]["states"] == 15
    assert data_rows[0]["evolution_event_counts"]["split"] == 1
    evolution_rows = [row for row in payload["artifacts"] if row["path"].endswith("evolution_manifest.json")]
    assert evolution_rows
    assert evolution_rows[0]["role"] == "evolution"
    assert evolution_rows[0]["has_web_result"] is True
    assert evolution_rows[0]["has_evolution"] is True
    assert evolution_rows[0]["evolution_status"] == "passed"


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
    write_cooccurrence_artifacts(output_dir)
    write_cluster_review_packet_artifact(output_dir)
    write_narrative_evidence_artifacts(output_dir)
    _write_cluster_sharded_run_sidecars(output_dir)
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
    assert job_payload["result"]["features"]["matrix"] is True
    assert job_payload["result"]["feature_states"]["keyword"] == "stable"
    assert job_payload["result"]["feature_states"]["matrix"] == "stable"
    assert job_payload["result"]["result_manifest"]["schema_version"] == "sciscape_result_manifest_v1"
    assert job_payload["result"]["result_manifest"]["manifest_state"] == "present"
    assert job_payload["result"]["result_manifest"]["title"] == "Curated Web Result"
    assert job_payload["result"]["result_manifest"]["artifacts"]["keywords"]["path"] == "landscape/keywords.parquet"
    assert job_payload["result"]["run_state"]["status"] == "failed"
    assert job_payload["result"]["run_state"]["shards"] == {"total": 3, "complete": 1, "failed": 1, "running": 1}
    assert job_payload["result"]["run_state"]["resume"]["supported"] is True
    assert "--keyword-engine cluster_sharded" in job_payload["result"]["run_state"]["resume"]["command"]
    assert job_payload["result"]["run_state_summary"]["schema_version"] == "sciscape_run_state_summary_v1"
    assert job_payload["result"]["run_state_summary"]["shards_failed"] == 1
    assert job_payload["result"]["run_state_summary"]["failed_shards"] == [2]
    assert job_payload["result"]["run_state_summary"]["recoverable_output_kinds"] == {"candidate_shard": 1}
    assert job_payload["result"]["run_state_summary"]["checkpoint_kinds"] == {
        "cluster_sharded_manifest": 1,
        "preflight": 1,
        "progress": 1,
    }
    assert job_payload["result"]["run_state_summary"]["resume_state"] == "command_available"
    partial_response = client.get(
        f"/api/jobs/{job_id}/download/"
        "landscape/keyword_cluster_sharded/full_run/candidates/candidate_shard_0000.parquet"
    )
    assert partial_response.status_code == 200
    assert partial_response.content == b"placeholder"
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
    matrix_vos_exports = [
        row
        for row in exports
        if row["export_id"] == "matrix_term_cooccurrence_default_vosviewer_network"
    ]
    assert len(matrix_vos_exports) == 1
    assert (
        matrix_vos_exports[0]["path"]
        == "exports/matrix_term_cooccurrence_default_vosviewer_network/vosviewer_matrix_map.txt"
    )
    assert matrix_vos_exports[0]["export_manifest_ref"] == (
        "exports/matrix_term_cooccurrence_default_vosviewer_network/export_manifest.json"
    )
    assert matrix_vos_exports[0]["selection_summary"]["scope"] == "matrix_artifact"
    assert matrix_vos_exports[0]["selection_summary"]["view_mode"] == "matrix_vosviewer_network"
    assert matrix_vos_exports[0]["selection_summary"]["threshold_keys"] == ["min_link_strength"]
    assert [row["path"] for row in matrix_vos_exports[0]["files"]] == [
        "exports/matrix_term_cooccurrence_default_vosviewer_network/vosviewer_matrix_map.txt",
        "exports/matrix_term_cooccurrence_default_vosviewer_network/vosviewer_matrix_network.txt",
    ]
    assert job_payload["result"]["artifact_contract"]["ok"] is True
    assert job_payload["result"]["features"]["evolution"] is True
    assert job_payload["result"]["feature_states"]["evolution"] == "stable"
    assert job_payload["result"]["evolution_summary"]["status"] == "passed"
    assert job_payload["result"]["evolution_summary"]["event_counts"]["continuation"] == 3
    assert job_payload["result"]["narrative_summary"]["available"] is True
    assert job_payload["result"]["narrative_summary"]["feature_state"] == "beta"
    assert job_payload["result"]["narrative_summary"]["cluster_count"] == 1
    assert job_payload["result"]["atlas"]["node_count"] == 1
    assert job_payload["result"]["atlas"]["nodes"][0]["label"] == "perovskite"
    assert job_payload["result"]["atlas"]["nodes"][0]["doc_count"] == 2
    assert job_payload["result"]["atlas"]["nodes"][0]["doc_count_source"] == "membership:cluster"
    assert job_payload["result"]["atlas"]["nodes"][0]["representative_work_count"] == 2
    assert job_payload["result"]["atlas"]["nodes"][0]["representative_works"][0]["title"] == "Stable perovskite device"
    assert job_payload["result"]["atlas"]["nodes"][0]["narrative"]["state"] == "beta"
    assert job_payload["result"]["atlas"]["nodes"][0]["narrative"]["claims"][0]["evidence"]
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

    features_response = client.get(f"/api/jobs/{job_id}/features")
    assert features_response.status_code == 200
    features_payload = features_response.json()
    assert features_payload["schema_version"] == "sciscape_job_features_v1"
    assert features_payload["readiness"] == "ready"
    assert features_payload["api_profile"] == "job_result"
    assert features_payload["features"]["keyword"] is True
    assert features_payload["feature_states"]["keyword"] == "stable"
    assert features_payload["feature_states"]["narrative"] == "beta"
    assert features_payload["modules"]["keyword"]["ready"] is True
    assert features_payload["modules"]["narrative"]["ready"] is True
    assert "keywords" in features_payload["modules"]["keyword"]["artifact_refs"]
    assert "narrative" in features_payload["modules"]["narrative"]["artifact_refs"]
    assert features_payload["quality"]["validation_state"] == "passed_with_warnings"
    assert features_payload["artifacts"]["keywords"]["path"] == "landscape/keywords.parquet"
    assert features_payload["artifacts"]["narrative"]["path"] == "narrative/narrative_manifest.json"
    assert features_payload["run_state"]["status"] == "failed"
    assert features_payload["run_state"]["shards"]["failed"] == 1
    assert "--scoring-shard-resume" in features_payload["run_state"]["resume"]["command"]
    assert features_payload["run_state_summary"]["has_recoverable_state"] is True
    assert features_payload["run_state_summary"]["failed_shard_count"] == 1
    assert features_payload["run_state_summary"]["resume_command_kind"] == "cli_resume"
    shard_action = next(row for row in features_payload["operator_actions"] if row["action_id"] == "rerun_failed_shards")
    assert shard_action["enabled"] is True
    assert shard_action["label"] == "Rerun Failed Shards"
    assert shard_action["failed_shards"] == [2]
    resume_action = next(row for row in features_payload["operator_actions"] if row["action_id"] == "resume_cli")
    assert resume_action["enabled"] is True
    assert resume_action["label"] == "Resume In App"
    selected_action = next(row for row in features_payload["operator_actions"] if row["action_id"] == "rerun_selected_shards")
    assert selected_action["enabled"] is True
    assert selected_action["endpoint"] == f"/api/jobs/{job_id}/resume-shards"
    assert selected_action["requires_input"] is True
    assert selected_action["body_schema"]["n_jobs"] == "optional int 1-64"
    assert selected_action["body_schema"]["parallel_backend"] == (
        "optional enum auto|loky|threading|sequential"
    )

    run_state_response = client.get(f"/api/jobs/{job_id}/run-state")
    assert run_state_response.status_code == 200
    run_state_payload = run_state_response.json()
    assert run_state_payload["schema_version"] == "sciscape_job_run_state_v1"
    assert run_state_payload["recommended_action"] == "rerun_failed_shards"
    assert run_state_payload["run_state_summary"]["resume_state"] == "command_available"
    assert [row["action_id"] for row in run_state_payload["operator_actions"]] == [
        "rerun_failed_shards",
        "resume_cli",
        "rerun_selected_shards",
        "retry_query",
        "cancel_job",
    ]
    assert run_state_payload["operator_actions"][0]["label"] == "Rerun Failed Shards"
    assert run_state_payload["operator_actions"][0]["enabled"] is True
    assert run_state_payload["operator_actions"][1]["label"] == "Resume In App"
    assert run_state_payload["operator_actions"][1]["enabled"] is True
    assert run_state_payload["operator_actions"][2]["label"] == "Rerun Selected Shards"
    assert run_state_payload["operator_actions"][2]["enabled"] is True
    assert run_state_payload["operator_actions"][2]["valid_shard_id_min"] == 0
    assert run_state_payload["operator_actions"][2]["valid_shard_id_max"] == 2
    assert run_state_payload["operator_actions"][2]["body_schema"]["shard_ids"] == "list[int]"
    assert run_state_payload["operator_actions"][0]["schedule_options_schema"]["n_jobs"] == "optional int 1-64"
    assert run_state_payload["operator_actions"][3]["enabled"] is False
    assert run_state_payload["recoverable_artifacts"][0]["download_path"] == (
        "landscape/keyword_cluster_sharded/full_run/candidates/candidate_shard_0000.parquet"
    )
    assert {row["family"] for row in run_state_payload["recoverable_artifacts"]} == {"partial", "checkpoint"}

    readiness_response = client.get(f"/api/jobs/{job_id}/readiness")
    assert readiness_response.status_code == 200
    assert readiness_response.json()["schema_version"] == "sciscape_job_features_v1"

    render_response = client.get(f"/api/jobs/{job_id}/atlas-render")
    assert render_response.status_code == 200
    render_payload = render_response.json()
    assert render_payload["schema_version"] == "sciscape_atlas_render_payload_v1"
    assert render_payload["view"]["type"] == "OrthographicView"
    assert render_payload["layers"]["nodes"]["recommended_deck_layer"] == "ScatterplotLayer"
    assert render_payload["layers"]["labels"]["rows"][0]["text"] == "perovskite"

    render_summary_response = client.get(f"/api/jobs/{job_id}/atlas-render/summary")
    assert render_summary_response.status_code == 200
    render_summary = render_summary_response.json()
    assert render_summary["schema_version"] == "sciscape_atlas_render_summary_v1"
    assert render_summary["source_schema_version"] == "sciscape_atlas_render_payload_v1"
    assert render_summary["node_count"] == 1
    assert render_summary["layer_summaries"]["nodes"]["row_count"] == 1
    assert render_summary["layer_summaries"]["labels"]["recommended_deck_layer"] == "TextLayer"

    render_layer_response = client.get(f"/api/jobs/{job_id}/atlas-render/layers/nodes")
    assert render_layer_response.status_code == 200
    render_layer = render_layer_response.json()
    assert render_layer["schema_version"] == "sciscape_atlas_render_layer_response_v1"
    assert render_layer["layer_key"] == "nodes"
    assert render_layer["row_count"] == 1
    assert render_layer["layer"]["rows"][0]["cluster_uid"] == "cluster:0"

    missing_layer_response = client.get(f"/api/jobs/{job_id}/atlas-render/layers/missing")
    assert missing_layer_response.status_code == 200
    assert missing_layer_response.json()["available_layers"] == ["edges", "hierarchy", "labels", "nodes"]

    narrative_response = client.get(f"/api/jobs/{job_id}/narrative")
    assert narrative_response.status_code == 200
    narrative = narrative_response.json()
    assert narrative["schema_version"] == "sciscape_narrative_api_v1"
    assert narrative["available"] is True
    assert narrative["feature_state"] == "beta"
    assert narrative["clusters"][0]["cluster_uid"] == "cluster:0"
    assert narrative["clusters"][0]["claims"][0]["evidence"][0]["artifact_ref"]

    cluster_narrative_response = client.get(f"/api/jobs/{job_id}/clusters/cluster:0/narrative")
    assert cluster_narrative_response.status_code == 200
    cluster_narrative = cluster_narrative_response.json()
    assert cluster_narrative["target_found"] is True
    assert cluster_narrative["cluster"]["cluster_uid"] == "cluster:0"
    assert cluster_narrative["cluster"]["claim_count"] >= 3
    claim_id = cluster_narrative["cluster"]["claims"][0]["claim_id"]

    invalid_review_response = client.post(
        f"/api/jobs/{job_id}/clusters/cluster:0/narrative/review",
        json={
            "claim_id": claim_id,
            "decision_type": "maybe",
            "reviewer": "tester",
        },
    )
    assert invalid_review_response.status_code == 200
    assert invalid_review_response.json()["available"] is False
    assert invalid_review_response.json()["allowed_decision_types"] == [
        "accepted",
        "needs_revision",
        "not_required",
        "rejected",
    ]

    review_response = client.post(
        f"/api/jobs/{job_id}/clusters/cluster:0/narrative/review",
        json={
            "claim_id": claim_id,
            "decision_type": "accepted",
            "reviewer": "tester",
            "reason": "evidence refs are sufficient",
        },
    )
    assert review_response.status_code == 200
    review_payload = review_response.json()
    assert review_payload["available"] is True
    assert review_payload["review_decision"]["claim_id"] == claim_id
    assert review_payload["review_decision"]["decision_type"] == "accepted"
    assert review_payload["review_decision"]["reviewer"] == "tester"
    assert review_payload["review_validation"]["blocking_issues"] == []
    assert review_payload["cluster"]["claims"][0]["review_state"] == "accepted"
    review_decisions = pd.read_parquet(output_dir / "narrative" / "review_decisions.parquet")
    assert review_decisions["claim_id"].tolist() == [claim_id]
    assert review_decisions["decision_type"].tolist() == ["accepted"]
    assert review_decisions["reason"].tolist() == ["evidence refs are sufficient"]
    claims_after_review = pd.read_parquet(output_dir / "narrative" / "claims.parquet")
    assert claims_after_review.loc[claims_after_review["claim_id"] == claim_id, "review_state"].iloc[0] == "accepted"

    cluster_narrative_after_review_response = client.get(
        f"/api/jobs/{job_id}/clusters/cluster:0/narrative"
    )
    assert cluster_narrative_after_review_response.status_code == 200
    cluster_after_review = cluster_narrative_after_review_response.json()["cluster"]
    assert cluster_after_review["review_count"] == 1
    assert cluster_after_review["claims"][0]["review_state"] == "accepted"
    assert cluster_after_review["claims"][0]["review_count"] == 1
    latest_review = cluster_after_review["claims"][0]["latest_review"]
    assert latest_review["decision_type"] == "accepted"
    assert latest_review["reviewer"] == "tester"
    assert latest_review["reason"] == "evidence refs are sufficient"

    missing_narrative_response = client.get(f"/api/jobs/{job_id}/clusters/cluster:missing/narrative")
    assert missing_narrative_response.status_code == 200
    assert missing_narrative_response.json()["target_found"] is False

    evolution_response = client.get(f"/api/jobs/{job_id}/evolution")
    assert evolution_response.status_code == 200
    evolution = evolution_response.json()
    assert evolution["available"] is True
    assert evolution["status"] == "passed"
    assert evolution["event_counts"]["split"] == 1
    assert evolution["event_counts"]["merge"] == 1
    assert len(evolution["time_slices"]) == 3
    assert len(evolution["events"]) == 8
    assert evolution["evolution_map"]["schema_version"] == "sciscape_evolution_map_v1"
    assert evolution["evolution_map"]["layout"] == "lineage_time_grid"
    assert evolution["evolution_map"]["slice_count"] == 3
    assert evolution["evolution_map"]["node_count"] == 15
    assert evolution["evolution_map"]["edge_count"] == 9
    assert evolution["evolution_map"]["event_count"] == 8
    assert evolution["evolution_map"]["total_lineage_count"] == 12
    assert evolution["evolution_map"]["total_node_count"] == 15
    assert evolution["evolution_map"]["total_edge_count"] == 9
    assert evolution["evolution_map"]["hidden_node_count"] == 0
    assert evolution["evolution_map"]["hidden_edge_count"] == 0
    assert evolution["evolution_map"]["slices"][0]["x"] == 0.0
    assert evolution["evolution_map"]["slices"][-1]["x"] == 1.0
    assert any(edge["relation"] == "split_child" for edge in evolution["evolution_map"]["edges"])

    bounded_evolution_response = client.get(f"/api/jobs/{job_id}/evolution?map_node_limit=2")
    assert bounded_evolution_response.status_code == 200
    bounded_evolution = bounded_evolution_response.json()
    assert bounded_evolution["evolution_map"]["node_count"] == 2
    assert bounded_evolution["evolution_map"]["total_node_count"] == 15
    assert bounded_evolution["evolution_map"]["hidden_node_count"] == 13
    assert bounded_evolution["evolution_map"]["truncated"]["nodes"] is True

    evolution_qa_download = client.get(f"/api/jobs/{job_id}/download/evolution/evolution_qa.json")
    assert evolution_qa_download.status_code == 200
    assert evolution_qa_download.json()["status"] == "passed"

    evolution_states_download = client.get(f"/api/jobs/{job_id}/download/evolution/cluster_states.parquet")
    assert evolution_states_download.status_code == 200
    assert "cluster_states.parquet" in evolution_states_download.headers["content-disposition"]
    assert evolution_states_download.content

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

    matrix_vos_download = client.get(
        f"/api/jobs/{job_id}/download/"
        "exports/matrix_term_cooccurrence_default_vosviewer_network/vosviewer_matrix_map.txt"
    )
    assert matrix_vos_download.status_code == 200
    assert "perovskite" in matrix_vos_download.text
    assert "passivation" in matrix_vos_download.text

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
            "exports/matrix_term_cooccurrence_default_vosviewer_network/vosviewer_matrix_map.txt",
            "exports/matrix_term_cooccurrence_default_vosviewer_network/vosviewer_matrix_network.txt",
            "exports/vosviewer_map_network/export_manifest.json",
            "exports/vosviewer_map_network/export_qa.json",
            "exports/vosviewer_term_cooccurrence/export_manifest.json",
            "exports/vosviewer_term_cooccurrence/export_qa.json",
            "exports/matrix_term_cooccurrence_default_vosviewer_network/export_manifest.json",
            "exports/matrix_term_cooccurrence_default_vosviewer_network/export_qa.json",
            "vosviewer_bundle_inventory.json",
        }.issubset(names)

    refreshed = client.get(f"/api/jobs/{job_id}").json()
    refreshed_exports = refreshed["result"]["result_manifest"]["exports"]
    bundle_exports = [row for row in refreshed_exports if row["export_id"] == "vosviewer_bundle"]
    assert len(bundle_exports) == 1
    assert bundle_exports[0]["path"] == "exports/vosviewer_bundle/vosviewer_bundle.zip"
    assert bundle_exports[0]["selection_summary"]["view_mode"] == "download_bundle"
    assert bundle_exports[0]["selection_summary"]["filter_count"] == 1


def test_open_local_data_accepts_evolution_manifest_path(monkeypatch, tmp_path):
    output_dir = tmp_path / "workspace" / "web_output" / "evolution_demo"
    (output_dir / "landscape" / "report").mkdir(parents=True)
    (output_dir / "landscape" / "report" / "data.json").write_text("{}", encoding="utf-8")
    write_evolution_synthetic_smoke_artifact(output_dir)
    monkeypatch.setattr(
        "sciscape.web.app._LOCAL_DATA_ROOTS",
        [tmp_path / "workspace" / "web_output"],
    )

    client = TestClient(app)
    manifest_path = output_dir / "evolution" / "evolution_manifest.json"
    response = client.post("/api/local-data/open", json={"path": str(manifest_path)})

    assert response.status_code == 200
    payload = response.json()
    assert payload["result"]["output_dir"] == str(output_dir)
    assert payload["result"]["features"]["evolution"] is True
    assert payload["result"]["feature_states"]["evolution"] == "stable"
    assert payload["result"]["evolution_summary"]["status"] == "passed"


def test_evolution_api_exposes_state_membership_sidecar(monkeypatch, tmp_path):
    output_dir = tmp_path / "workspace" / "web_output" / "membership_evolution_demo"
    landscape_dir = output_dir / "landscape"
    report_dir = landscape_dir / "report"
    report_dir.mkdir(parents=True)
    (report_dir / "data.json").write_text("{}", encoding="utf-8")
    records = pd.DataFrame(
        {
            "uid": ["D0", "D1", "D2", "D3"],
            "title": ["Alpha 2021", "Alpha 2022", "Beta 2021", "Beta 2022"],
            "abstract": ["alpha topic", "alpha topic", "beta topic", "beta topic"],
            "pubyear": [2021, 2022, 2021, 2022],
        }
    )
    membership = pd.DataFrame({"uid": ["D0", "D1", "D2", "D3"], "cluster": [0, 0, 1, 1]})
    keywords = pd.DataFrame(
        {
            "cluster_id": [0, 1],
            "term": ["alpha topic", "beta topic"],
            "score": [0.9, 0.8],
        }
    )
    records.to_parquet(output_dir / "abstracts.parquet", index=False)
    membership.to_parquet(landscape_dir / "membership.parquet", index=False)
    keywords.to_parquet(landscape_dir / "keywords.parquet", index=False)
    write_evolution_artifacts(
        output_dir,
        evolution_id="membership_projection",
        records_df=records,
        membership_df=membership,
        keywords_df=keywords,
    )
    monkeypatch.setattr(
        "sciscape.web.app._LOCAL_DATA_ROOTS",
        [tmp_path / "workspace" / "web_output"],
    )

    client = TestClient(app)
    manifest_path = output_dir / "evolution" / "evolution_manifest.json"
    response = client.post("/api/local-data/open", json={"path": str(manifest_path)})

    assert response.status_code == 200
    payload = response.json()
    job_id = payload["job_id"]
    assert payload["result"]["evolution_summary"]["counts"]["state_membership_rows"] == 4

    evolution_response = client.get(f"/api/jobs/{job_id}/evolution?state_membership_limit=2")
    assert evolution_response.status_code == 200
    evolution = evolution_response.json()
    assert evolution["available"] is True
    assert evolution["counts"]["state_membership_rows"] == 4
    assert len(evolution["state_membership"]) == 2
    assert evolution["truncated"]["state_membership"] is True
    assert evolution["state_membership_summary"]["loaded_rows"] == 2
    assert evolution["state_membership_summary"]["truncated"] is True
    assert evolution["state_membership_summary"]["by_state"]
    matching_diagnostics = evolution["matching_method"]["diagnostics"]
    assert matching_diagnostics["source"] == "projected_membership_transitions"
    assert matching_diagnostics["candidate_transition_rows"] == len(evolution["transitions"])
    assert matching_diagnostics["retained_transition_rows"] == len(evolution["transitions"])
    assert matching_diagnostics["dropped_transition_rows"] == 0
    assert matching_diagnostics["slice_pair_count"] == 1
    assert matching_diagnostics["slice_pairs"][0]["candidate_count"] == len(evolution["transitions"])
    assert matching_diagnostics["slice_pairs"][0]["retained_count"] == len(evolution["transitions"])

    sidecar_download = client.get(f"/api/jobs/{job_id}/download/evolution/state_membership.parquet")
    assert sidecar_download.status_code == 200
    assert "state_membership.parquet" in sidecar_download.headers["content-disposition"]
    assert sidecar_download.content


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
