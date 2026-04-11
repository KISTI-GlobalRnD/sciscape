"""FastAPI application for SciScape web interface.

Run:
    uvicorn sciscape.web.app:app --reload
    # or: sciscape web
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.responses import StreamingResponse

log = logging.getLogger(__name__)

app = FastAPI(title="SciScape", version="0.1.0")

# Serve static files (frontend)
_STATIC_DIR = Path(__file__).parent / "static"
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

# In-memory job store
_jobs: Dict[str, Dict[str, Any]] = {}


# ── Models ──────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    years: str | None = None
    max_works: int = 5000
    email: str | None = None
    edge_types: str = "dc,bc"
    run_landscape: bool = True


class JobStatus(BaseModel):
    job_id: str
    status: str  # "pending", "running", "done", "error"
    progress: list[str]
    result: dict | None = None


# ── Routes ──────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main frontend page."""
    index_path = _STATIC_DIR / "index.html"
    if index_path.exists():
        return index_path.read_text(encoding="utf-8")
    return HTMLResponse("<h1>SciScape</h1><p>Static files not found. Run from repo root.</p>")


@app.post("/api/query")
async def submit_query(req: QueryRequest, bg: BackgroundTasks):
    """Submit an OpenAlex query job. Returns job_id for progress tracking."""
    job_id = str(uuid.uuid4())[:8]
    _jobs[job_id] = {
        "status": "pending",
        "progress": [],
        "result": None,
        "request": req.model_dump(),
    }
    bg.add_task(_run_job, job_id, req)
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    """Get job status and progress."""
    job = _jobs.get(job_id)
    if not job:
        return {"error": "job not found"}
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        result=job["result"],
    )


@app.get("/api/jobs/{job_id}/stream")
async def stream_progress(job_id: str):
    """SSE stream of job progress updates."""
    job = _jobs.get(job_id)
    if not job:
        return {"error": "job not found"}

    async def event_generator():
        last_idx = 0
        while True:
            job = _jobs.get(job_id, {})
            progress = job.get("progress", [])
            # Send new messages
            while last_idx < len(progress):
                data = json.dumps({"message": progress[last_idx]})
                yield f"data: {data}\n\n"
                last_idx += 1

            if job.get("status") in ("done", "error"):
                result = job.get("result")
                data = json.dumps({"status": job["status"], "result": result})
                yield f"data: {data}\n\n"
                break

            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/jobs/{job_id}/download/{filename}")
async def download_file(job_id: str, filename: str):
    """Download output files from a completed job."""
    job = _jobs.get(job_id)
    if not job or job["status"] != "done":
        return {"error": "job not done"}
    result = job.get("result", {})
    output_dir = result.get("output_dir")
    if not output_dir:
        return {"error": "no output directory"}
    file_path = Path(output_dir) / filename
    if not file_path.exists():
        return {"error": f"file not found: {filename}"}
    return FileResponse(file_path, filename=filename)


@app.get("/api/jobs")
async def list_jobs():
    """List all jobs."""
    return [
        {"job_id": jid, "status": j["status"], "query": j["request"].get("query", "")}
        for jid, j in _jobs.items()
    ]


# ── Background job runner ────────────────────────────────────

def _run_job(job_id: str, req: QueryRequest) -> None:
    """Execute the OpenAlex pipeline in background."""
    from sciscape.openalex import run_openalex_pipeline, OpenAlexPipelineConfig

    job = _jobs[job_id]
    job["status"] = "running"

    def progress_cb(msg: str) -> None:
        job["progress"].append(msg)

    try:
        output_dir = Path("sciscape_web_output") / job_id
        filters = {}
        if req.years:
            filters["publication_year"] = req.years

        config = OpenAlexPipelineConfig(
            query=req.query,
            filters=filters,
            max_works=req.max_works,
            email=req.email,
            edge_types=req.edge_types.split(","),
            output_dir=output_dir,
            run_landscape=req.run_landscape,
            progress=progress_cb,
        )
        result = run_openalex_pipeline(config)

        job["status"] = "done"
        job["result"] = {
            "n_works": result.n_works,
            "n_edges": result.n_edges,
            "output_dir": str(output_dir),
            "abstracts_path": str(result.abstracts_path) if result.abstracts_path else None,
            "edges_path": str(result.edges_path) if result.edges_path else None,
            "landscape_dir": str(result.landscape_dir) if result.landscape_dir else None,
        }
    except Exception as e:
        job["status"] = "error"
        job["progress"].append(f"ERROR: {e}")
        job["result"] = {"error": str(e)}
        log.exception("Job %s failed", job_id)
