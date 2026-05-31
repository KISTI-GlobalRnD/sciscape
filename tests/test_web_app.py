"""Tests for the SciScape FastAPI web surface."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import sciscape.web.app as web_app
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


def test_web_homepage_exposes_query_analysis_controls():
    client = TestClient(app)
    response = client.get("/")

    assert response.status_code == 200
    assert "Query OpenAlex" in response.text
    assert "Recommended Demos" in response.text
    assert "Local Data" in response.text
    assert "Artifact contract" in response.text
    assert "applyResultFeatureAvailability" in response.text
    assert 'id="q-search"' in response.text
    assert "submitQuery()" in response.text
    assert "fetch('/api/query'" in response.text
    assert "fetch('/api/demo-presets'" in response.text
    assert "fetch('/api/local-data" in response.text
    assert 'id="file-input"' not in response.text


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
    assert job_payload["result"]["artifact_contract"]["ok"] is True


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
