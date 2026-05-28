"""Tests for the SciScape FastAPI web surface."""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi.testclient import TestClient

from sciscape.web.app import _jobs, app


def _register_done_job(job_id: str, output_dir: Path) -> None:
    _jobs.create(job_id, {"query": "test query"})
    job = _jobs.get(job_id)
    assert job is not None
    job["status"] = "done"
    job["progress"] = []
    job["result"] = {"output_dir": str(output_dir)}
    _jobs.persist(job_id)


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
    assert 'id="q-search"' in response.text
    assert "submitQuery()" in response.text
    assert "fetch('/api/query'" in response.text
    assert 'id="file-input"' not in response.text


def test_query_endpoint_enqueues_openalex_analysis(monkeypatch, tmp_path):
    def fake_run_job(job_id, req):
        job = _jobs[job_id]
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
        _jobs.persist(job_id)

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
