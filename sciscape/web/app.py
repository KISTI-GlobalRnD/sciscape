"""FastAPI application for SciScape web interface.

Run:
    uvicorn sciscape.web.app:app --reload
    # or: sciscape web
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import shlex
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Sequence

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from sciscape.artifacts import (
    NARRATIVE_REVIEW_DECISIONS_SCHEMA_VERSION,
    build_atlas_payload_from_report_data,
    build_atlas_render_payload,
    infer_result_artifacts,
    load_result_manifest,
    validate_evolution_artifact,
    validate_narrative_artifact,
    validate_result_root,
    validate_workspace,
    write_narrative_publication_artifacts,
    write_result_manifest,
)

log = logging.getLogger(__name__)


class SafeJSONResponse(JSONResponse):
    """Emit strict JSON while sanitizing non-finite values on the slow path."""

    def render(self, content: Any) -> bytes:
        try:
            return json.dumps(
                content,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
        except ValueError:
            return json.dumps(
                _json_safe(content),
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")


app = FastAPI(title="SciScape", version="0.2.0", default_response_class=SafeJSONResponse)
app.add_middleware(GZipMiddleware, minimum_size=1024)

# Serve static files (frontend)
_STATIC_DIR = Path(__file__).parent / "static"
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

# Persistent job store (SQLite with in-memory fallback)
from .jobstore import get_store as _get_store
_jobs = _get_store()


# ── Models ──────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    years: str | None = None
    max_works: int = 5000
    email: str | None = None
    edge_types: str = "dc,bc,cc"
    run_landscape: bool = True
    combine_strategy: str = "consensus"
    combine_top_k: int | str = "auto"
    auto_gamma: bool = True
    auto_gamma_target: float = 3.0
    n_levels: int = 4
    api_attempt_budget: int | None = None
    retry_wait_budget_seconds: float | None = None


class JobStatus(BaseModel):
    job_id: str
    status: str  # "pending", "running", "done", "error", "cancelled"
    progress: list[str]
    result: dict | None = None
    cancel_requested_at_utc: str | None = None


class LocalDataOpenRequest(BaseModel):
    path: str


class NarrativeReviewRequest(BaseModel):
    claim_id: str
    decision_type: str
    reviewer: str | None = "web"
    reason: str | None = None


_RESUME_PARALLEL_BACKENDS = {"auto", "loky", "threading", "sequential"}
_MAX_RESUME_N_JOBS = 64


class ResumeRunRequest(BaseModel):
    n_jobs: int | None = None
    parallel_backend: str | None = None


class ShardResumeRequest(ResumeRunRequest):
    shard_ids: list[int]


class JobCancelled(RuntimeError):
    """Raised when a web job observes a cooperative cancellation request."""


_LOCAL_DATA_ROOTS = [
    Path("workspace/web_output"),
    Path("workspace/examples_output"),
    Path("workspace/output"),
    Path("viewer"),
]
_DEMO_MANIFEST_PATH = Path("examples/demo_presets.json")
_DEFAULT_DEMO_ARTIFACTS = [
    "result_manifest.json",
    "abstracts.parquet",
    "edges.parquet",
    "landscape/membership.parquet",
    "landscape/keywords.parquet",
    "landscape/edge_evidence_samples.json",
    "landscape/report/data.json",
]


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


def _job_retry_request(job: dict[str, Any]) -> QueryRequest:
    request_payload = job.get("request") if isinstance(job.get("request"), dict) else {}
    query = str(request_payload.get("query") or "")
    if not query or query.startswith("Local output:"):
        raise HTTPException(
            status_code=400,
            detail="job does not contain a replayable OpenAlex query request",
        )
    try:
        return QueryRequest(**request_payload)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="job does not contain a replayable OpenAlex query request",
        ) from exc


def _parse_resume_n_jobs(raw: Any) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="invalid n_jobs") from exc
    if value < 1 or value > _MAX_RESUME_N_JOBS:
        raise HTTPException(
            status_code=400,
            detail=f"n_jobs must be between 1 and {_MAX_RESUME_N_JOBS}",
        )
    return value


def _parse_resume_parallel_backend(raw: Any) -> str:
    backend = str(raw or "").strip().lower()
    if backend not in _RESUME_PARALLEL_BACKENDS:
        raise HTTPException(status_code=400, detail="invalid parallel_backend")
    return backend


def _resume_schedule_options(req: ResumeRunRequest | None) -> dict[str, Any]:
    if req is None:
        return {}
    options: dict[str, Any] = {}
    if req.n_jobs is not None:
        options["n_jobs"] = _parse_resume_n_jobs(req.n_jobs)
    if req.parallel_backend is not None:
        options["parallel_backend"] = _parse_resume_parallel_backend(req.parallel_backend)
    return options


def _safe_resume_argv(command: str) -> list[str]:
    """Parse and validate the narrow SciScape CLI resume surface."""

    try:
        parts = shlex.split(str(command or ""))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="invalid resume command") from exc
    if len(parts) < 4 or parts[0] != "sciscape" or parts[1] != "keywords":
        raise HTTPException(status_code=400, detail="unsupported resume command")

    argv = parts[1:]
    if len(argv) < 3 or any(str(value).startswith("-") for value in argv[1:3]):
        raise HTTPException(status_code=400, detail="unsupported resume command")

    value_flags = {
        "--keyword-engine",
        "--cluster-level",
        "--cluster-sharded-output-dir",
        "--cluster-sharded-shard-ids",
        "--progress-path",
        "--n-jobs",
        "--parallel-backend",
        "-o",
        "--output",
    }
    bool_flags = {"--scoring-shard-resume"}
    values: dict[str, str] = {}
    seen_bools: set[str] = set()
    i = 3
    while i < len(argv):
        token = argv[i]
        if token in bool_flags:
            seen_bools.add(token)
            i += 1
            continue
        if token in value_flags:
            if i + 1 >= len(argv) or argv[i + 1].startswith("-"):
                raise HTTPException(status_code=400, detail="unsupported resume command")
            values[token] = argv[i + 1]
            i += 2
            continue
        if token.startswith("--") and "=" in token:
            flag, value = token.split("=", 1)
            if flag not in value_flags or not value:
                raise HTTPException(status_code=400, detail="unsupported resume command")
            values[flag] = value
            i += 1
            continue
        raise HTTPException(status_code=400, detail="unsupported resume command")

    if values.get("--keyword-engine") != "cluster_sharded":
        raise HTTPException(status_code=400, detail="unsupported resume command")
    if "--cluster-sharded-output-dir" not in values:
        raise HTTPException(status_code=400, detail="unsupported resume command")
    if "--progress-path" not in values:
        raise HTTPException(status_code=400, detail="unsupported resume command")
    if "-o" not in values and "--output" not in values:
        raise HTTPException(status_code=400, detail="unsupported resume command")
    if "--scoring-shard-resume" not in seen_bools or "--no-scoring-shard-resume" in argv:
        raise HTTPException(status_code=400, detail="unsupported resume command")
    if "--cluster-sharded-shard-ids" in values:
        _parse_resume_shard_ids(values["--cluster-sharded-shard-ids"])
    if "--n-jobs" in values:
        _parse_resume_n_jobs(values["--n-jobs"])
    if "--parallel-backend" in values:
        _parse_resume_parallel_backend(values["--parallel-backend"])
    return argv


def _parse_resume_shard_ids(raw: Any) -> list[int]:
    try:
        ids = sorted({int(part.strip()) for part in str(raw or "").split(",") if part.strip()})
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="invalid shard ID list") from exc
    if not ids:
        raise HTTPException(status_code=400, detail="missing shard IDs")
    if any(value < 0 for value in ids):
        raise HTTPException(status_code=400, detail="invalid shard ID list")
    return ids


def _resume_argv_with_shard_ids(argv: Sequence[str], shard_ids: Sequence[int]) -> list[str]:
    shard_value = ",".join(str(int(value)) for value in sorted({int(item) for item in shard_ids}))
    if not shard_value:
        raise HTTPException(status_code=400, detail="missing shard IDs")
    out: list[str] = []
    i = 0
    replaced = False
    while i < len(argv):
        token = str(argv[i])
        if token == "--cluster-sharded-shard-ids":
            out.extend([token, shard_value])
            replaced = True
            i += 2
            continue
        if token.startswith("--cluster-sharded-shard-ids="):
            out.append(f"--cluster-sharded-shard-ids={shard_value}")
            replaced = True
            i += 1
            continue
        out.append(token)
        i += 1
    if not replaced:
        out.extend(["--cluster-sharded-shard-ids", shard_value])
    return _safe_resume_argv("sciscape " + shlex.join(out))


def _resume_argv_with_schedule_options(
    argv: Sequence[str],
    schedule_options: dict[str, Any] | None = None,
) -> list[str]:
    options = dict(schedule_options or {})
    if not options:
        return list(argv)

    replacement_flags: dict[str, str] = {}
    if "n_jobs" in options:
        replacement_flags["--n-jobs"] = str(_parse_resume_n_jobs(options["n_jobs"]))
    if "parallel_backend" in options:
        replacement_flags["--parallel-backend"] = _parse_resume_parallel_backend(options["parallel_backend"])
    if not replacement_flags:
        return list(argv)

    out: list[str] = []
    i = 0
    replaced: set[str] = set()
    while i < len(argv):
        token = str(argv[i])
        if token in replacement_flags:
            out.extend([token, replacement_flags[token]])
            replaced.add(token)
            i += 2
            continue
        matched_inline = False
        for flag, value in replacement_flags.items():
            if token.startswith(f"{flag}="):
                out.append(f"{flag}={value}")
                replaced.add(flag)
                matched_inline = True
                break
        if matched_inline:
            i += 1
            continue
        out.append(token)
        i += 1
    for flag, value in replacement_flags.items():
        if flag not in replaced:
            out.extend([flag, value])
    return _safe_resume_argv("sciscape " + shlex.join(out))


def _job_resume_request(
    job: dict[str, Any],
    schedule_options: dict[str, Any] | None = None,
) -> tuple[str, list[str]]:
    status = str(job.get("status") or "")
    if status in {"pending", "running"}:
        raise HTTPException(status_code=409, detail="job is still running")
    result = job.get("result") if isinstance(job.get("result"), dict) else {}
    run_state = (
        result.get("run_state")
        if isinstance(result.get("run_state"), dict)
        else _run_state_for_result(result)
    )
    resume = run_state.get("resume") if isinstance(run_state.get("resume"), dict) else {}
    if not resume.get("supported"):
        raise HTTPException(status_code=400, detail="job has no supported resume action")
    command = str(resume.get("command") or "").strip()
    if not command:
        raise HTTPException(status_code=400, detail="job has no resume command")
    argv = _resume_argv_with_schedule_options(_safe_resume_argv(command), schedule_options)
    return "sciscape " + shlex.join(argv), argv


def _job_failed_shard_resume_request(
    job: dict[str, Any],
    schedule_options: dict[str, Any] | None = None,
) -> tuple[str, list[str], list[int]]:
    command, argv = _job_resume_request(job, schedule_options)
    result = job.get("result") if isinstance(job.get("result"), dict) else {}
    run_state = (
        result.get("run_state")
        if isinstance(result.get("run_state"), dict)
        else _run_state_for_result(result)
    )
    failure = run_state.get("failure") if isinstance(run_state.get("failure"), dict) else {}
    shard_ids = _parse_resume_shard_ids(",".join(str(value) for value in failure.get("failed_shards") or []))
    shard_argv = _resume_argv_with_shard_ids(argv, shard_ids)
    shard_command = "sciscape " + shlex.join(shard_argv)
    return shard_command, shard_argv, shard_ids


def _job_selected_shard_resume_request(
    job: dict[str, Any],
    shard_ids: Sequence[int],
    schedule_options: dict[str, Any] | None = None,
) -> tuple[str, list[str], list[int]]:
    command, argv = _job_resume_request(job, schedule_options)
    result = job.get("result") if isinstance(job.get("result"), dict) else {}
    run_state = (
        result.get("run_state")
        if isinstance(result.get("run_state"), dict)
        else _run_state_for_result(result)
    )
    shards = run_state.get("shards") if isinstance(run_state.get("shards"), dict) else {}
    try:
        total = int(shards.get("total") or 0)
    except (TypeError, ValueError):
        total = 0
    if total <= 0:
        raise HTTPException(status_code=400, detail="job has no shard schedule metadata")

    requested = _parse_resume_shard_ids(",".join(str(value) for value in shard_ids))
    invalid = [value for value in requested if value >= total]
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=f"shard IDs outside available range 0-{total - 1}: {invalid}",
        )
    shard_argv = _resume_argv_with_shard_ids(argv, requested)
    shard_command = "sciscape " + shlex.join(shard_argv)
    return shard_command, shard_argv, requested


def _enqueue_retry_job(source_job_id: str, source_job: dict[str, Any], bg: BackgroundTasks) -> dict[str, Any]:
    status = str(source_job.get("status") or "")
    if status in {"pending", "running"}:
        raise HTTPException(status_code=409, detail="job is still running")

    req = _job_retry_request(source_job)
    job_id = str(uuid.uuid4())[:8]
    request_payload = req.model_dump()
    request_payload["source_type"] = "openalex_query"
    request_payload["retry_of"] = source_job_id
    _jobs.create(job_id, request_payload)
    retry_job = _jobs[job_id]
    retry_job["status"] = "pending"
    retry_job["progress"] = [f"Retry of job {source_job_id}"]
    retry_job["result"] = None
    _jobs.persist(job_id)
    bg.add_task(_run_job, job_id, req)
    return {"job_id": job_id, "retry_of": source_job_id, "request": request_payload}


def _enqueue_resume_job(
    source_job_id: str,
    source_job: dict[str, Any],
    bg: BackgroundTasks,
    *,
    failed_shards_only: bool = False,
    selected_shard_ids: Sequence[int] | None = None,
    schedule_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    schedule_options = dict(schedule_options or {})
    if failed_shards_only and selected_shard_ids is not None:
        raise HTTPException(status_code=400, detail="choose either failed shards or selected shards")
    if selected_shard_ids is not None:
        command, argv, shard_ids = _job_selected_shard_resume_request(
            source_job,
            selected_shard_ids,
            schedule_options,
        )
        resume_scope = "selected_shards"
        progress_message = f"Resume selected shards {shard_ids} of job {source_job_id}"
    elif failed_shards_only:
        command, argv, shard_ids = _job_failed_shard_resume_request(source_job, schedule_options)
        resume_scope = "failed_shards"
        progress_message = f"Resume failed shards {shard_ids} of job {source_job_id}"
    else:
        command, argv = _job_resume_request(source_job, schedule_options)
        shard_ids = []
        resume_scope = "all_resumable_shards"
        progress_message = f"Resume of job {source_job_id}"
    job_id = str(uuid.uuid4())[:8]
    request_payload = {
        "source_type": "sciscape_cli_resume",
        "resume_of": source_job_id,
        "resume_scope": resume_scope,
        "command": command,
        "argv": argv,
    }
    if shard_ids:
        request_payload["shard_ids"] = shard_ids
    if schedule_options:
        request_payload["schedule_options"] = schedule_options
    _jobs.create(job_id, request_payload)
    resume_job = _jobs[job_id]
    resume_job["status"] = "pending"
    resume_job["progress"] = [progress_message]
    resume_job["result"] = None
    _jobs.persist(job_id)
    bg.add_task(_run_resume_job, job_id, source_job_id, command, argv)
    payload = {"job_id": job_id, "resume_of": source_job_id, "command": command, "argv": argv}
    if shard_ids:
        payload["shard_ids"] = shard_ids
    if schedule_options:
        payload["schedule_options"] = schedule_options
    return payload


@app.post("/api/jobs/{job_id}/retry")
async def retry_job(job_id: str, bg: BackgroundTasks):
    """Re-run a previous OpenAlex query job as a new job."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return _enqueue_retry_job(job_id, job, bg)


@app.post("/api/jobs/{job_id}/resume")
async def resume_job(job_id: str, bg: BackgroundTasks, req: ResumeRunRequest | None = None):
    """Resume a supported SciScape CLI run as a new web job."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return _enqueue_resume_job(job_id, job, bg, schedule_options=_resume_schedule_options(req))


@app.post("/api/jobs/{job_id}/resume-failed-shards")
async def resume_failed_shards_job(job_id: str, bg: BackgroundTasks, req: ResumeRunRequest | None = None):
    """Resume only failed cluster-sharded keyword shards as a new web job."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return _enqueue_resume_job(
        job_id,
        job,
        bg,
        failed_shards_only=True,
        schedule_options=_resume_schedule_options(req),
    )


@app.post("/api/jobs/{job_id}/resume-shards")
async def resume_selected_shards_job(job_id: str, req: ShardResumeRequest, bg: BackgroundTasks):
    """Resume user-selected cluster-sharded keyword shards as a new web job."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return _enqueue_resume_job(
        job_id,
        job,
        bg,
        selected_shard_ids=req.shard_ids,
        schedule_options=_resume_schedule_options(req),
    )


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Request cooperative cancellation for a queued or running query job."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    status = str(job.get("status") or "")
    if status in {"done", "error", "cancelled"}:
        raise HTTPException(status_code=409, detail="job is already finished")
    if status not in {"pending", "running"}:
        raise HTTPException(status_code=409, detail="job cannot be cancelled")

    now = _utc_now()
    job["cancel_requested_at_utc"] = job.get("cancel_requested_at_utc") or now
    _append_progress_once(job, "Cancellation requested.")
    _jobs.persist(job_id)

    try:
        req = _job_retry_request(job)
        output_dir = Path("workspace/web_output") / job_id
        _write_live_job_status_artifacts(
            job_id=job_id,
            job=job,
            output_dir=output_dir,
            req=req,
            filters=_query_filters(req),
            status=status,
        )
        _jobs.persist(job_id)
    except HTTPException:
        pass
    return {
        "job_id": job_id,
        "status": job.get("status"),
        "cancel_requested": True,
        "cancel_requested_at_utc": job["cancel_requested_at_utc"],
    }


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
        cancel_requested_at_utc=job.get("cancel_requested_at_utc"),
    )


@app.get("/api/jobs/{job_id}/features")
async def get_job_features(job_id: str):
    """Get job-scoped capability, feature-state, and artifact readiness flags."""
    job = _jobs.get(job_id)
    if not job:
        return {"error": "job not found"}
    return _job_feature_payload(job_id, job)


@app.get("/api/jobs/{job_id}/run-state")
async def get_job_run_state(job_id: str):
    """Get run-state, recoverable artifacts, and available operator actions."""
    job = _jobs.get(job_id)
    if not job:
        return {"error": "job not found"}
    return _job_run_state_payload(job_id, job)


@app.get("/api/jobs/{job_id}/readiness")
async def get_job_readiness(job_id: str):
    """Alias for the job capability map, named for Atlas-style runtime checks."""
    return await get_job_features(job_id)


@app.get("/api/jobs/{job_id}/narrative")
async def get_job_narrative(job_id: str, limit: int = 80):
    """Get the evidence-backed narrative claim graph for a completed job."""
    job = _jobs.get(job_id)
    if not job or job["status"] != "done":
        return {"available": False, "error": "job not done"}
    result = job.get("result") if isinstance(job.get("result"), dict) else {}
    return _load_narrative_payload_for_result(result, claim_limit=limit)


@app.get("/api/jobs/{job_id}/clusters/{cluster_uid}/narrative")
async def get_cluster_narrative(job_id: str, cluster_uid: str, limit: int = 40):
    """Get claim/evidence rows for one selected Atlas cluster."""
    job = _jobs.get(job_id)
    if not job or job["status"] != "done":
        return {"available": False, "error": "job not done"}
    result = job.get("result") if isinstance(job.get("result"), dict) else {}
    return _load_narrative_payload_for_result(result, cluster_uid=cluster_uid, claim_limit=limit)


@app.post("/api/jobs/{job_id}/clusters/{cluster_uid}/narrative/review")
async def review_cluster_narrative_claim(job_id: str, cluster_uid: str, req: NarrativeReviewRequest):
    """Persist a review decision for one narrative claim and refresh the job view."""
    job = _jobs.get(job_id)
    if not job or job["status"] != "done":
        return {"available": False, "error": "job not done"}
    result = job.get("result") if isinstance(job.get("result"), dict) else {}
    written = _write_narrative_review_decision(result, cluster_uid=cluster_uid, request=req)
    if written.get("error"):
        return written
    publication = written.get("publication")
    root = _result_root_for_result(result)
    if root is not None:
        try:
            _refresh_job_result_manifest(job_id, result, root, mode="local_result")
        except Exception:
            pass
    _attach_report_atlas(result)
    _attach_narrative_summary(result)
    job["result"] = result
    _jobs.persist(job_id)
    payload = _load_narrative_payload_for_result(result, cluster_uid=cluster_uid, claim_limit=40)
    payload["review_decision"] = written.get("review_decision")
    payload["review_validation"] = written.get("validation")
    payload["publication"] = publication
    payload["result_manifest"] = result.get("result_manifest")
    return payload


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

            if job.get("status") in ("done", "error", "cancelled"):
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


def _resolve_job_output_file(job_id: str, filename: str) -> Path:
    """Resolve an output artifact path inside a terminal job directory."""
    job = _jobs.get(job_id)
    if not job or job["status"] not in {"done", "error", "cancelled"}:
        raise HTTPException(status_code=400, detail="job not in terminal state")
    result = job.get("result", {})
    output_dir = result.get("output_dir")
    if not output_dir:
        raise HTTPException(status_code=404, detail="no output directory")

    root = Path(output_dir).resolve()
    file_path = (root / filename).resolve()
    try:
        file_path.relative_to(root)
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid artifact path")
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"file not found: {filename}")
    return file_path


def _refresh_job_result_manifest(job_id: str, result: dict[str, Any], output_dir: Path, *, mode: str = "live_query") -> None:
    refreshed_manifest = load_result_manifest(output_dir, mode=mode)
    result["result_manifest"] = refreshed_manifest
    result["run_state"] = _run_state_for_result(result, refreshed_manifest)
    _attach_run_state_summary(result, refreshed_manifest)
    result["feature_states"] = {
        name: feature.get("state", "hidden")
        for name, feature in refreshed_manifest.get("features", {}).items()
        if isinstance(feature, dict)
    }
    features = dict(result.get("features") or {})
    for name, state in result["feature_states"].items():
        features[name] = state != "hidden"
    result["features"] = features
    job = _jobs.get(job_id)
    if job is not None:
        job["result"] = result
        _jobs.persist(job_id)


def _manifest_for_result(result: dict[str, Any]) -> dict[str, Any]:
    manifest = result.get("result_manifest")
    if isinstance(manifest, dict):
        return manifest
    output_dir = result.get("output_dir")
    if output_dir:
        try:
            loaded = load_result_manifest(output_dir)
            if isinstance(loaded, dict):
                return loaded
        except Exception:
            return {}
    return {}


def _run_state_for_result(
    result: dict[str, Any],
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest_payload = manifest if isinstance(manifest, dict) else _manifest_for_result(result)
    run_state = manifest_payload.get("run_state") if isinstance(manifest_payload, dict) else None
    if isinstance(run_state, dict):
        return run_state
    result_run_state = result.get("run_state") if isinstance(result.get("run_state"), dict) else None
    return dict(result_run_state or {})


def _count_run_state_kinds(rows: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        key = str(row.get("kind") or row.get("role") or "artifact")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _run_state_summary(run_state: dict[str, Any]) -> dict[str, Any]:
    """Return a compact operational summary for run-state UI/API surfaces."""

    def as_int(value: Any) -> int:
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    progress = run_state.get("progress") if isinstance(run_state.get("progress"), dict) else {}
    shards = run_state.get("shards") if isinstance(run_state.get("shards"), dict) else {}
    failure = run_state.get("failure") if isinstance(run_state.get("failure"), dict) else {}
    resume = run_state.get("resume") if isinstance(run_state.get("resume"), dict) else {}
    partial_outputs = run_state.get("partial_outputs") if isinstance(run_state.get("partial_outputs"), list) else []
    checkpoints = run_state.get("checkpoints") if isinstance(run_state.get("checkpoints"), list) else []

    current = progress.get("current")
    total = progress.get("total")
    percent = progress.get("percent")
    try:
        current_number = float(current)
        total_number = float(total)
        if percent is None and total_number > 0:
            percent = current_number / total_number * 100.0
    except (TypeError, ValueError):
        percent = percent if isinstance(percent, (int, float)) else None

    failed_shards = failure.get("failed_shards") if isinstance(failure.get("failed_shards"), list) else []
    output_kinds = _count_run_state_kinds(partial_outputs)
    checkpoint_kinds = _count_run_state_kinds(checkpoints)
    resume_supported = bool(resume.get("supported"))
    resume_command = str(resume.get("command") or "").strip()
    if resume_supported and resume_command:
        resume_state = "command_available"
    elif resume_supported:
        resume_state = "metadata_only"
    else:
        resume_state = "not_supported"

    return _json_safe(
        {
            "schema_version": "sciscape_run_state_summary_v1",
            "status": str(run_state.get("status") or "unknown"),
            "stage": progress.get("stage"),
            "progress_percent": percent,
            "shards_total": as_int(shards.get("total")),
            "shards_complete": as_int(shards.get("complete")),
            "shards_failed": as_int(shards.get("failed")),
            "shards_running": as_int(shards.get("running")),
            "failed_shards": failed_shards[:50],
            "failed_shard_count": len(failed_shards),
            "recoverable_output_count": len(partial_outputs),
            "checkpoint_count": len(checkpoints),
            "recoverable_output_kinds": output_kinds,
            "checkpoint_kinds": checkpoint_kinds,
            "resume_state": resume_state,
            "resume_supported": resume_supported,
            "resume_command_kind": resume.get("command_kind"),
            "resume_mode": resume.get("mode"),
            "resume_artifact_dir": resume.get("artifact_dir"),
            "failure_reason": failure.get("reason"),
            "has_recoverable_state": bool(partial_outputs or checkpoints or resume_supported),
        }
    )


def _attach_run_state_summary(result: dict[str, Any], manifest: dict[str, Any] | None = None) -> None:
    result["run_state_summary"] = _run_state_summary(_run_state_for_result(result, manifest))


def _run_state_recoverable_artifacts(run_state: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(row: Any, family: str) -> None:
        if not isinstance(row, dict) or not row.get("path"):
            return
        path = str(row.get("path") or "")
        kind = str(row.get("kind") or row.get("role") or "")
        if not path or path.endswith("/") or kind == "landscape" or path in seen:
            return
        seen.add(path)
        record = {
            key: value
            for key, value in row.items()
            if key in {"path", "kind", "role", "status", "shard_id", "rows", "source_rows", "size_bytes"}
        }
        record["family"] = family
        record["download_path"] = path
        rows.append(record)

    for row in run_state.get("partial_outputs") if isinstance(run_state.get("partial_outputs"), list) else []:
        add(row, "partial")
        if isinstance(row, dict) and row.get("flagged_path"):
            flagged = dict(row)
            flagged["path"] = row.get("flagged_path")
            flagged["kind"] = f"{row.get('kind') or 'output'}_flagged"
            add(flagged, "partial")
    for row in run_state.get("checkpoints") if isinstance(run_state.get("checkpoints"), list) else []:
        add(row, "checkpoint")
    return rows


def _can_retry_job(job: dict[str, Any]) -> tuple[bool, str | None]:
    try:
        _job_retry_request(job)
        return True, None
    except HTTPException as exc:
        return False, str(exc.detail)


def _can_resume_job(job: dict[str, Any]) -> tuple[bool, str | None]:
    try:
        _job_resume_request(job)
        return True, None
    except HTTPException as exc:
        return False, str(exc.detail)


def _run_state_operator_actions(
    *,
    job_id: str,
    job: dict[str, Any],
    run_state: dict[str, Any],
    summary: dict[str, Any],
) -> list[dict[str, Any]]:
    status = str(job.get("status") or "")
    run_status = str(run_state.get("status") or status)
    resume = run_state.get("resume") if isinstance(run_state.get("resume"), dict) else {}
    resume_enabled, resume_reason = _can_resume_job(job)
    failed_shards = summary.get("failed_shards") or []
    failed_shard_count = int(summary.get("failed_shard_count") or len(failed_shards) or 0)
    shards_total = int(summary.get("shards_total") or 0)
    rerun_failed_enabled = bool(resume_enabled and failed_shard_count > 0)
    rerun_selected_enabled = bool(resume_enabled and shards_total > 0)
    schedule_options_schema = {
        "n_jobs": f"optional int 1-{_MAX_RESUME_N_JOBS}",
        "parallel_backend": "optional enum auto|loky|threading|sequential",
    }
    actions = []
    actions.append(
        {
            "action_id": "rerun_failed_shards",
            "label": "Rerun Failed Shards",
            "method": "POST",
            "endpoint": f"/api/jobs/{job_id}/resume-failed-shards",
            "enabled": rerun_failed_enabled,
            "reason": None if rerun_failed_enabled else resume_reason or "no failed shards",
            "scope": "cluster_sharded_keyword_shards",
            "command_kind": resume.get("command_kind"),
            "mode": "selected_failed_shards",
            "failed_shards": failed_shards,
            "schedule_options_schema": schedule_options_schema,
        }
    )
    actions.append(
        {
            "action_id": "resume_cli",
            "label": "Resume In App",
            "method": "POST",
            "endpoint": f"/api/jobs/{job_id}/resume",
            "enabled": resume_enabled,
            "reason": None if resume_enabled else resume_reason,
            "scope": "cluster_sharded_keywords",
            "command_kind": resume.get("command_kind"),
            "mode": resume.get("mode"),
            "failed_shards": failed_shards,
            "schedule_options_schema": schedule_options_schema,
        }
    )
    actions.append(
        {
            "action_id": "rerun_selected_shards",
            "label": "Rerun Selected Shards",
            "method": "POST",
            "endpoint": f"/api/jobs/{job_id}/resume-shards",
            "enabled": rerun_selected_enabled,
            "reason": None if rerun_selected_enabled else resume_reason or "no shard schedule metadata",
            "scope": "cluster_sharded_keyword_shards",
            "command_kind": resume.get("command_kind"),
            "mode": "user_selected_shards",
            "requires_input": True,
            "body_schema": {"shard_ids": "list[int]", **schedule_options_schema},
            "shards_total": shards_total,
            "valid_shard_id_min": 0 if shards_total > 0 else None,
            "valid_shard_id_max": shards_total - 1 if shards_total > 0 else None,
            "failed_shards": failed_shards,
        }
    )

    retry_enabled, retry_reason = _can_retry_job(job)
    retry_enabled = retry_enabled and run_status in {"failed", "cancelled", "stopped_by_qc", "error"}
    actions.append(
        {
            "action_id": "retry_query",
            "label": "Retry Query",
            "method": "POST",
            "endpoint": f"/api/jobs/{job_id}/retry",
            "enabled": retry_enabled,
            "reason": None if retry_enabled else retry_reason or "run status is not retryable",
            "scope": "openalex_query",
        }
    )

    cancel_enabled = status in {"pending", "running"} and not bool(job.get("cancel_requested_at_utc"))
    actions.append(
        {
            "action_id": "cancel_job",
            "label": "Cancel Job",
            "method": "POST",
            "endpoint": f"/api/jobs/{job_id}/cancel",
            "enabled": cancel_enabled,
            "reason": None if cancel_enabled else "job is not cancellable",
            "scope": "running_job",
        }
    )
    return actions


def _job_run_state_payload(job_id: str, job: dict[str, Any]) -> dict[str, Any]:
    result = job.get("result") if isinstance(job.get("result"), dict) else {}
    manifest = _manifest_for_result(result)
    run_state = _run_state_for_result(result, manifest)
    summary = _run_state_summary(run_state)
    actions = _run_state_operator_actions(
        job_id=job_id,
        job=job,
        run_state=run_state,
        summary=summary,
    )
    recommended = next((row["action_id"] for row in actions if row.get("enabled")), None)
    return _json_safe(
        {
            "schema_version": "sciscape_job_run_state_v1",
            "job_id": job_id,
            "job_status": job.get("status"),
            "cancel_requested": bool(job.get("cancel_requested_at_utc")),
            "cancel_requested_at_utc": job.get("cancel_requested_at_utc"),
            "available": bool(run_state),
            "run_state": run_state,
            "run_state_summary": summary,
            "recoverable_artifacts": _run_state_recoverable_artifacts(run_state),
            "operator_actions": actions,
            "recommended_action": recommended,
        }
    )


def _job_feature_payload(job_id: str, job: dict[str, Any]) -> dict[str, Any]:
    result = job.get("result") if isinstance(job.get("result"), dict) else {}
    manifest = _manifest_for_result(result)
    run_state = _run_state_for_result(result, manifest)
    run_state_summary = _run_state_summary(run_state)
    feature_details = manifest.get("features") if isinstance(manifest.get("features"), dict) else {}
    result_feature_states = result.get("feature_states") if isinstance(result.get("feature_states"), dict) else {}
    feature_states: dict[str, str] = {
        str(name): str(details.get("state", "hidden"))
        for name, details in feature_details.items()
        if isinstance(details, dict)
    }
    for name, state in result_feature_states.items():
        feature_states.setdefault(str(name), str(state or "hidden"))

    bool_features: dict[str, bool] = {
        name: state not in {"hidden", "blocked", "false", "False"}
        for name, state in feature_states.items()
    }
    for name, value in (result.get("features") if isinstance(result.get("features"), dict) else {}).items():
        bool_features.setdefault(str(name), bool(value))

    artifact_contract = result.get("artifact_contract") if isinstance(result.get("artifact_contract"), dict) else {}
    quality = manifest.get("quality") if isinstance(manifest.get("quality"), dict) else {}
    artifacts = manifest.get("artifacts") if isinstance(manifest.get("artifacts"), dict) else {}
    artifact_summaries = {
        str(key): {
            field: record.get(field)
            for field in ("role", "path", "status", "schema_version", "rows", "size_bytes")
            if isinstance(record, dict) and field in record
        }
        for key, record in artifacts.items()
        if isinstance(record, dict)
    }
    warnings = (
        artifact_contract.get("warnings")
        if isinstance(artifact_contract.get("warnings"), list)
        else []
    )
    blocking_issues = [
        warning
        for warning in warnings
        if isinstance(warning, dict) and warning.get("severity") in {"error", "blocking"}
    ]
    modules: dict[str, dict[str, Any]] = {}
    for feature, state in sorted(feature_states.items()):
        details = feature_details.get(feature) if isinstance(feature_details.get(feature), dict) else {}
        feature_warnings = details.get("warnings") if isinstance(details.get("warnings"), list) else []
        modules[feature] = {
            "state": state,
            "ready": state in {"stable", "beta"},
            "reason": details.get("reason"),
            "artifact_refs": details.get("artifact_refs") if isinstance(details.get("artifact_refs"), list) else [],
            "warning_count": len(feature_warnings),
            "required_for_profile": feature in {"overview", "cluster_map", "keyword", "quality"},
        }

    result_state = str(result.get("result_state") or artifact_contract.get("result_state") or "unknown")
    if job.get("status") != "done":
        readiness = "not_ready"
    elif result_state == "blocked" or quality.get("validation_state") == "blocked" or blocking_issues:
        readiness = "blocked"
    elif any(state in {"stable", "beta"} for state in feature_states.values()):
        readiness = "ready"
    else:
        readiness = "partial"

    return _json_safe(
        {
            "schema_version": "sciscape_job_features_v1",
            "job_id": job_id,
            "job_status": job.get("status"),
            "readiness": readiness,
            "api_profile": "job_result",
            "result_state": result_state,
            "features": bool_features,
            "feature_states": feature_states,
            "feature_details": feature_details,
            "run_state": run_state,
            "run_state_summary": run_state_summary,
            "operator_actions": _run_state_operator_actions(
                job_id=job_id,
                job=job,
                run_state=run_state,
                summary=run_state_summary,
            ),
            "modules": modules,
            "hidden_features": sorted(
                feature for feature, state in feature_states.items() if state == "hidden"
            ),
            "quality": {
                "validation_state": quality.get("validation_state"),
                "warning_count": quality.get("warning_count", len(warnings)),
                "blocking_count": quality.get("blocking_count", len(blocking_issues)),
                "gate_paths": quality.get("gate_paths", []),
                "last_validated_at_utc": quality.get("last_validated_at_utc"),
            },
            "counts": artifact_contract.get("counts", {}),
            "artifacts": artifact_summaries,
            "artifact_count": len(artifact_summaries),
            "warning_count": len(warnings),
            "blocking_issue_count": len(blocking_issues),
            "versions": {
                "sciscape_version": manifest.get("sciscape_version"),
                "result_manifest_schema": manifest.get("schema_version"),
                "result_id": manifest.get("result_id"),
                "updated_at_utc": manifest.get("updated_at_utc"),
            },
        }
    )


def _result_root_for_result(result: dict[str, Any]) -> Path | None:
    output_dir = result.get("output_dir")
    if not output_dir:
        return None
    try:
        return Path(output_dir).expanduser().resolve()
    except Exception:
        return None


def _path_inside_result_root(root: Path, path: str | Path) -> Path | None:
    candidate = Path(path)
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        return None
    return resolved


def _narrative_manifest_path_for_result(result: dict[str, Any]) -> Path | None:
    root = _result_root_for_result(result)
    if root is None:
        return None
    manifest = _manifest_for_result(result)
    artifacts = manifest.get("artifacts") if isinstance(manifest.get("artifacts"), dict) else {}
    for record in artifacts.values():
        if not isinstance(record, dict):
            continue
        if record.get("role") != "narrative" or record.get("status") != "present":
            continue
        path = _path_inside_result_root(root, str(record.get("path") or ""))
        if path is not None and path.exists() and path.is_file():
            return path
    fallback = root / "narrative" / "narrative_manifest.json"
    return fallback if fallback.exists() and fallback.is_file() else None


def _narrative_output_path(narrative_dir: Path, manifest: dict[str, Any], key: str, default_name: str) -> Path:
    outputs = manifest.get("outputs") if isinstance(manifest.get("outputs"), dict) else {}
    raw = str(outputs.get(key) or default_name)
    path = Path(raw)
    return path if path.is_absolute() else narrative_dir / path


def _read_narrative_records(path: Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    import pandas as pd

    if not path.exists():
        return []
    df = pd.read_parquet(path)
    if "sort_order" in df.columns:
        df = df.sort_values("sort_order", kind="stable")
    if limit is not None and len(df) > limit:
        df = df.head(limit)
    return [_json_safe(row) for row in df.to_dict(orient="records")]


def _empty_narrative_review_frame() -> Any:
    import pandas as pd

    return pd.DataFrame(
        columns=[
            "schema_version",
            "narrative_id",
            "decision_id",
            "claim_id",
            "decision_type",
            "reviewer",
            "decided_at_utc",
            "reason",
            "target_id",
            "cluster_uid",
        ]
    )


def _narrative_target_matches(row: dict[str, Any], cluster_uid: str) -> bool:
    wanted = str(cluster_uid or "").strip()
    if not wanted:
        return True
    keys = {
        str(row.get("target_key") or ""),
        str(row.get("target_id") or ""),
        str(row.get("cluster_uid") or ""),
        str(row.get("cluster_id") or ""),
    }
    if wanted in keys:
        return True
    if ":" in wanted:
        suffix = wanted.split(":", 1)[1]
        return suffix in keys
    return False


_NARRATIVE_REVIEW_DECISIONS = {"accepted", "rejected", "needs_revision", "not_required"}


def _write_narrative_review_decision(
    result: dict[str, Any],
    *,
    cluster_uid: str,
    request: NarrativeReviewRequest,
) -> dict[str, Any]:
    import pandas as pd

    decision_type = str(request.decision_type or "").strip()
    if decision_type not in _NARRATIVE_REVIEW_DECISIONS:
        return {
            "available": False,
            "error": f"unsupported decision_type: {decision_type}",
            "allowed_decision_types": sorted(_NARRATIVE_REVIEW_DECISIONS),
        }
    claim_id = str(request.claim_id or "").strip()
    if not claim_id:
        return {"available": False, "error": "claim_id is required"}
    manifest_path = _narrative_manifest_path_for_result(result)
    if manifest_path is None:
        return {"available": False, "error": "no narrative claim graph artifact"}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"available": False, "error": f"could not read narrative manifest: {exc}"}
    narrative_dir = manifest_path.parent
    targets_path = _narrative_output_path(narrative_dir, manifest, "targets", "narrative_targets.parquet")
    claims_path = _narrative_output_path(narrative_dir, manifest, "claims", "claims.parquet")
    targets = _read_narrative_records(targets_path)
    target_ids = {
        str(target.get("target_id") or "")
        for target in targets
        if _narrative_target_matches(target, cluster_uid)
    }
    if not target_ids:
        return {"available": False, "error": f"narrative target not found: {cluster_uid}"}
    try:
        claims = pd.read_parquet(claims_path)
    except Exception as exc:
        return {"available": False, "error": f"could not read narrative claims: {exc}"}
    if "claim_id" not in claims.columns or "target_id" not in claims.columns:
        return {"available": False, "error": "narrative claims table is missing claim_id or target_id"}
    mask = (claims["claim_id"].map(str) == claim_id) & claims["target_id"].map(str).isin(target_ids)
    if not bool(mask.any()):
        return {"available": False, "error": f"claim not found for cluster: {claim_id}"}
    if "review_state" not in claims.columns:
        claims["review_state"] = "not_reviewed"
    claims.loc[mask, "review_state"] = decision_type
    claims.to_parquet(claims_path, index=False)

    outputs = manifest.get("outputs") if isinstance(manifest.get("outputs"), dict) else {}
    outputs["reviews"] = str(outputs.get("reviews") or "review_decisions.parquet")
    manifest["outputs"] = outputs
    manifest["review_state_advertised"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    reviews_path = _narrative_output_path(narrative_dir, manifest, "reviews", "review_decisions.parquet")
    try:
        reviews = pd.read_parquet(reviews_path) if reviews_path.exists() else _empty_narrative_review_frame()
    except Exception:
        reviews = _empty_narrative_review_frame()
    target_id = sorted(target_ids)[0]
    decided_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    review_row = {
        "schema_version": NARRATIVE_REVIEW_DECISIONS_SCHEMA_VERSION,
        "narrative_id": str(manifest.get("narrative_id") or ""),
        "decision_id": f"decision_{uuid.uuid4().hex[:12]}",
        "claim_id": claim_id,
        "decision_type": decision_type,
        "reviewer": str(request.reviewer or "web"),
        "decided_at_utc": decided_at,
        "reason": str(request.reason or ""),
        "target_id": target_id,
        "cluster_uid": str(cluster_uid),
    }
    reviews = pd.concat([reviews, pd.DataFrame([review_row])], ignore_index=True)
    reviews.to_parquet(reviews_path, index=False)
    publication = write_narrative_publication_artifacts(manifest_path)
    validation = validate_narrative_artifact(manifest_path).to_dict()
    return {
        "available": True,
        "review_decision": review_row,
        "validation": validation,
        "publication": publication,
        "reviews_path": str(reviews_path),
    }


def _cluster_narrative_view(
    target: dict[str, Any],
    *,
    sections: list[dict[str, Any]],
    claims: list[dict[str, Any]],
    refs_by_id: dict[str, dict[str, Any]],
    links_by_claim: dict[str, list[dict[str, Any]]],
    reviews_by_claim: dict[str, list[dict[str, Any]]],
    sources_by_id: dict[str, dict[str, Any]],
    claim_limit: int,
) -> dict[str, Any]:
    target_id = str(target.get("target_id") or "")
    target_claims = [
        claim
        for claim in claims
        if str(claim.get("target_id") or "") == target_id
    ][: max(1, int(claim_limit))]
    claim_rows: list[dict[str, Any]] = []
    aggregate_only_refs = 0
    review_count = 0
    reviewed_claim_count = 0
    for claim in target_claims:
        claim_id = str(claim.get("claim_id") or "")
        target_claim_key = f"{target_id}\x1f{claim_id}"
        claim_reviews = reviews_by_claim.get(target_claim_key) or reviews_by_claim.get(claim_id, [])
        latest_review = claim_reviews[-1] if claim_reviews else None
        review_count += len(claim_reviews)
        if claim_reviews:
            reviewed_claim_count += 1
        evidence_rows = []
        for link in links_by_claim.get(claim_id, []):
            ref_id = str(link.get("evidence_ref_id") or "")
            ref = dict(refs_by_id.get(ref_id, {}))
            source_id = str(ref.get("evidence_source_id") or "")
            source = sources_by_id.get(source_id, {})
            if ref.get("aggregate_only") is True or ref.get("locator_type") == "aggregate":
                aggregate_only_refs += 1
            evidence_rows.append(
                {
                    "evidence_ref_id": ref_id,
                    "evidence_role": link.get("evidence_role"),
                    "link_strength": link.get("link_strength"),
                    "required": bool(link.get("required", False)),
                    "evidence_type": ref.get("evidence_type"),
                    "evidence_label": ref.get("evidence_label"),
                    "locator_type": ref.get("locator_type"),
                    "locator": ref.get("locator"),
                    "aggregate_only": bool(ref.get("aggregate_only", False) or ref.get("locator_type") == "aggregate"),
                    "artifact_ref": source.get("artifact_ref"),
                    "artifact_path": source.get("artifact_path"),
                }
            )
        claim_rows.append(
            {
                "claim_id": claim_id,
                "section_id": claim.get("section_id"),
                "claim_type": claim.get("claim_type"),
                "claim_text": claim.get("claim_text"),
                "support_state": claim.get("support_state"),
                "confidence": claim.get("confidence"),
                "evidence_ref_count": claim.get("evidence_ref_count"),
                "text_origin": claim.get("text_origin"),
                "review_state": claim.get("review_state"),
                "review_count": len(claim_reviews),
                "latest_review": latest_review,
                "warning_flags": claim.get("warning_flags"),
                "evidence": evidence_rows,
            }
        )
    section_rows = [
        section
        for section in sections
        if str(section.get("target_id") or "") == target_id
    ]
    state = "stable"
    if any(str(claim.get("support_state") or "") in {"weak", "caveat"} for claim in target_claims):
        state = "beta"
    if any(str(claim.get("support_state") or "") in {"unsupported", "contradicted"} for claim in target_claims):
        state = "blocked"
    return {
        "cluster_uid": target.get("target_key") or target.get("cluster_uid") or target_id,
        "target_id": target_id,
        "target": target,
        "state": state,
        "claim_count": len(target_claims),
        "review_count": int(review_count),
        "reviewed_claim_count": int(reviewed_claim_count),
        "pending_review_claim_count": max(0, len(target_claims) - reviewed_claim_count),
        "aggregate_only_ref_count": int(aggregate_only_refs),
        "sections": section_rows,
        "claims": claim_rows,
    }


def _load_narrative_payload_for_result(
    result: dict[str, Any],
    *,
    cluster_uid: str | None = None,
    claim_limit: int = 80,
) -> dict[str, Any]:
    manifest_path = _narrative_manifest_path_for_result(result)
    if manifest_path is None:
        return {
            "schema_version": "sciscape_narrative_api_v1",
            "available": False,
            "reason": "no narrative claim graph artifact",
            "clusters": [],
        }
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "schema_version": "sciscape_narrative_api_v1",
            "available": False,
            "reason": f"could not read narrative manifest: {exc}",
            "clusters": [],
        }
    narrative_dir = manifest_path.parent
    validation = validate_narrative_artifact(manifest_path).to_dict()
    targets = _read_narrative_records(
        _narrative_output_path(narrative_dir, manifest, "targets", "narrative_targets.parquet")
    )
    claims = _read_narrative_records(
        _narrative_output_path(narrative_dir, manifest, "claims", "claims.parquet")
    )
    refs = _read_narrative_records(
        _narrative_output_path(narrative_dir, manifest, "evidence_refs", "evidence_refs.parquet")
    )
    links = _read_narrative_records(
        _narrative_output_path(narrative_dir, manifest, "claim_evidence_links", "claim_evidence_links.parquet")
    )
    sections = _read_narrative_records(
        _narrative_output_path(narrative_dir, manifest, "sections", "narrative_sections.parquet")
    )
    sources = _read_narrative_records(
        _narrative_output_path(narrative_dir, manifest, "evidence_sources", "evidence_sources.parquet")
    )
    reviews = _read_narrative_records(
        _narrative_output_path(narrative_dir, manifest, "reviews", "review_decisions.parquet")
    )
    refs_by_id = {str(ref.get("evidence_ref_id") or ""): ref for ref in refs}
    sources_by_id = {str(source.get("evidence_source_id") or ""): source for source in sources}
    links_by_claim: dict[str, list[dict[str, Any]]] = {}
    for link in links:
        links_by_claim.setdefault(str(link.get("claim_id") or ""), []).append(link)
    reviews_by_claim: dict[str, list[dict[str, Any]]] = {}
    for review in sorted(
        reviews,
        key=lambda row: (str(row.get("decided_at_utc") or ""), str(row.get("decision_id") or "")),
    ):
        claim_id = str(review.get("claim_id") or "")
        target_id = str(review.get("target_id") or "")
        if target_id:
            reviews_by_claim.setdefault(f"{target_id}\x1f{claim_id}", []).append(review)
        reviews_by_claim.setdefault(claim_id, []).append(review)
    matched_targets = [
        target
        for target in targets
        if cluster_uid is None or _narrative_target_matches(target, cluster_uid)
    ]
    cluster_views = [
        _cluster_narrative_view(
            target,
            sections=sections,
            claims=claims,
            refs_by_id=refs_by_id,
            links_by_claim=links_by_claim,
            reviews_by_claim=reviews_by_claim,
            sources_by_id=sources_by_id,
            claim_limit=claim_limit,
        )
        for target in matched_targets
    ]
    payload = {
        "schema_version": "sciscape_narrative_api_v1",
        "available": True,
        "target_found": bool(cluster_views) if cluster_uid is not None else None,
        "cluster_uid": cluster_uid,
        "narrative_id": manifest.get("narrative_id"),
        "title": manifest.get("title"),
        "status": validation.get("status"),
        "feature_state": validation.get("feature_state"),
        "manifest_path": str(manifest_path),
        "counts": validation.get("counts", {}),
        "claim_counts": validation.get("claim_counts", {}),
        "checks": validation.get("checks", {}),
        "warnings": validation.get("warnings", []),
        "blocking_issues": validation.get("blocking_issues", []),
        "clusters": cluster_views,
    }
    if cluster_uid is not None:
        payload["cluster"] = cluster_views[0] if cluster_views else None
    return _json_safe(payload)


def _attach_narrative_summary(result: dict[str, Any]) -> None:
    payload = _load_narrative_payload_for_result(result, claim_limit=6)
    if not payload.get("available"):
        return
    result["narrative_summary"] = {
        "available": True,
        "narrative_id": payload.get("narrative_id"),
        "status": payload.get("status"),
        "feature_state": payload.get("feature_state"),
        "counts": payload.get("counts", {}),
        "claim_counts": payload.get("claim_counts", {}),
        "cluster_count": len(payload.get("clusters", [])),
        "warning_count": len(payload.get("warnings", [])),
        "blocking_issue_count": len(payload.get("blocking_issues", [])),
    }
    atlas = result.get("atlas")
    if not isinstance(atlas, dict) or not isinstance(atlas.get("nodes"), list):
        return
    narratives = {
        str(cluster.get("cluster_uid") or ""): cluster
        for cluster in payload.get("clusters", [])
        if isinstance(cluster, dict)
    }
    for node in atlas["nodes"]:
        if not isinstance(node, dict):
            continue
        cluster_uid = str(node.get("cluster_uid") or "")
        if cluster_uid in narratives:
            node["narrative"] = narratives[cluster_uid]


@app.get("/api/jobs/{job_id}/download/vosviewer-bundle.zip")
async def download_vosviewer_bundle(job_id: str):
    """Build and download a zip bundle from manifest-backed VOSviewer exports."""
    job = _jobs.get(job_id)
    if not job or job["status"] != "done":
        raise HTTPException(status_code=400, detail="job not done")
    result = job.get("result", {})
    output_dir = result.get("output_dir")
    if not output_dir:
        raise HTTPException(status_code=404, detail="no output directory")

    from sciscape.export import export_vosviewer_bundle

    root = Path(output_dir).resolve()
    try:
        written = export_vosviewer_bundle(root)
        _refresh_job_result_manifest(job_id, result, root)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"could not build VOSviewer bundle: {exc}")
    bundle_path = Path(written["bundle_path"])
    return FileResponse(bundle_path, filename="vosviewer_bundle.zip", media_type="application/zip")


@app.get("/api/jobs/{job_id}/download/{filename:path}")
async def download_file(job_id: str, filename: str):
    """Download output files from a completed job."""
    file_path = _resolve_job_output_file(job_id, filename)
    return FileResponse(file_path, filename=file_path.name)


@app.get("/api/jobs/{job_id}/view/{filename:path}")
async def view_file(job_id: str, filename: str):
    """Open an output artifact inline, e.g. generated HTML reports."""
    file_path = _resolve_job_output_file(job_id, filename)
    return FileResponse(file_path)


@app.get("/api/jobs/{job_id}/atlas-render")
async def get_atlas_render(job_id: str):
    """Get renderer-oriented Atlas layer rows for deck.gl-style map engines."""
    return _atlas_render_payload_for_job(job_id)


@app.get("/api/jobs/{job_id}/atlas-render/summary")
async def get_atlas_render_summary(job_id: str):
    """Get Atlas render metadata and layer row counts without full layer rows."""
    payload = _atlas_render_payload_for_job(job_id)
    if payload.get("error"):
        return payload
    layers = payload.get("layers") if isinstance(payload.get("layers"), dict) else {}
    layer_summaries = {
        str(key): {
            "layer_id": layer.get("layer_id"),
            "recommended_deck_layer": layer.get("recommended_deck_layer"),
            "row_count": len(layer.get("rows") or []),
        }
        for key, layer in layers.items()
        if isinstance(layer, dict)
    }
    return _json_safe(
        {
            "schema_version": "sciscape_atlas_render_summary_v1",
            "source_schema_version": payload.get("schema_version"),
            "semantic_schema_version": payload.get("source_schema_version"),
            "job_id": job_id,
            "engine_family": payload.get("engine_family"),
            "view": payload.get("view"),
            "levels": payload.get("levels", []),
            "node_count": payload.get("node_count"),
            "edge_count": payload.get("edge_count"),
            "label_count": payload.get("label_count"),
            "hierarchy_edge_count": payload.get("hierarchy_edge_count"),
            "layer_summaries": layer_summaries,
            "warnings": payload.get("warnings", []),
        }
    )


@app.get("/api/jobs/{job_id}/atlas-render/layers/{layer_key}")
async def get_atlas_render_layer(job_id: str, layer_key: str):
    """Get one Atlas render layer row group for lazy map hydration."""
    payload = _atlas_render_payload_for_job(job_id)
    if payload.get("error"):
        return payload
    layers = payload.get("layers") if isinstance(payload.get("layers"), dict) else {}
    layer = layers.get(layer_key)
    if not isinstance(layer, dict):
        return {
            "error": f"atlas render layer not found: {layer_key}",
            "available_layers": sorted(str(key) for key in layers),
        }
    return _json_safe(
        {
            "schema_version": "sciscape_atlas_render_layer_response_v1",
            "source_schema_version": payload.get("schema_version"),
            "job_id": job_id,
            "layer_key": layer_key,
            "view": payload.get("view"),
            "levels": payload.get("levels", []),
            "layer": layer,
            "row_count": len(layer.get("rows") or []),
            "warnings": payload.get("warnings", []),
        }
    )


def _atlas_render_payload_for_job(job_id: str) -> dict[str, Any]:
    job = _jobs.get(job_id)
    if not job or job["status"] != "done":
        return {"error": "job not done"}
    result = job.get("result", {})
    payload = _atlas_render_payload_for_result(result)
    if payload is None:
        return {"error": "no atlas payload"}
    return _json_safe(payload)


@app.get("/api/jobs/{job_id}/network")
async def get_network(job_id: str):
    """Get cluster network data for D3 visualization."""
    from ._helpers import require_done_job, get_edges_path, find_membership, find_keywords
    try:
        job, result = require_done_job(_jobs, job_id)
    except Exception:
        return {"error": "job not done"}
    output_dir = result.get("output_dir")
    if not output_dir:
        return {"error": "no output directory"}

    from .network_data import build_network_json

    edges_path = result.get("edges_path")
    if not edges_path or not Path(edges_path).exists():
        return {"error": "no edges file"}

    # Check for membership and keywords
    out = Path(output_dir)
    membership_path = None
    keywords_path = None
    landscape_dir = result.get("landscape_dir")
    if landscape_dir:
        ld = Path(landscape_dir)
        for f in ld.glob("membership*.parquet"):
            membership_path = f
            break
        for f in ld.glob("keywords*.parquet"):
            keywords_path = f
            break

    # Check for per-layer edge files
    layer_paths = {}
    for layer in ("dc", "bc", "cc"):
        p = out / f"edges_{layer}.parquet"
        if p.exists():
            layer_paths[layer] = p

    try:
        data = build_network_json(
            Path(edges_path),
            membership_path=membership_path,
            keywords_path=keywords_path,
            edge_layer_paths=layer_paths if layer_paths else None,
        )
        return data
    except Exception as e:
        return {"error": str(e)}


def _local_data_roots() -> list[Path]:
    """Return local roots the web app may browse for existing outputs."""
    roots: list[Path] = []
    for root in _LOCAL_DATA_ROOTS:
        resolved = root if root.is_absolute() else Path.cwd() / root
        roots.append(resolved.resolve())
    workspace = _current_workspace_validation()
    if workspace is not None and workspace.status != "blocked":
        for root in workspace.defaults.get("output_roots") or []:
            path = Path(str(root))
            resolved = path if path.is_absolute() else Path.cwd() / path
            roots.append(resolved.resolve())
    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = root.as_posix()
        if key not in seen:
            unique.append(root)
            seen.add(key)
    return unique


def _workspace_root() -> Path:
    return Path.cwd().resolve()


def _current_workspace_validation():
    try:
        return validate_workspace(_workspace_root())
    except Exception as exc:
        log.warning("workspace validation failed: %s", exc)
        return None


def _workspace_summary() -> dict[str, Any]:
    validation = _current_workspace_validation()
    if validation is None:
        return {
            "state": "unknown",
            "status": "warning",
            "manifest_path": None,
            "warnings": [{"code": "workspace_validation_failed"}],
            "blocking_issues": [],
            "counts": {},
        }
    payload = validation.to_dict()
    return {
        "workspace_id": payload.get("workspace_id"),
        "state": payload.get("state"),
        "status": payload.get("status"),
        "manifest_path": payload.get("manifest_path"),
        "qa_path": payload.get("qa_path"),
        "counts": payload.get("counts", {}),
        "warnings": payload.get("warnings", []),
        "blocking_issues": payload.get("blocking_issues", []),
    }


def _workspace_result_ref_root(row: dict[str, Any]) -> Path | None:
    raw_path = row.get("path")
    if not raw_path:
        return None
    path = Path(str(raw_path))
    resolved = path if path.is_absolute() else _workspace_root() / path
    if resolved.name in {"result_manifest.json", "MANIFEST.json"}:
        return resolved.parent.resolve()
    try:
        return Path(infer_result_artifacts(resolved).result_root).resolve()
    except Exception:
        return resolved.resolve() if resolved.is_dir() else resolved.parent.resolve()


def _workspace_registered_result_roots() -> list[Path]:
    validation = _current_workspace_validation()
    if validation is None or validation.status == "blocked":
        return []
    roots: list[Path] = []
    for row in validation.objects.get("results", []):
        if row.get("path_state") not in {"present", "external"}:
            continue
        root = _workspace_result_ref_root(row)
        if root is not None:
            roots.append(root)
    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = root.as_posix()
        if key not in seen:
            unique.append(root)
            seen.add(key)
    return unique


def _workspace_primary_result_path(root: Path) -> Path:
    artifacts = infer_result_artifacts(root)
    if artifacts.report_data_path and artifacts.report_data_path.exists():
        return artifacts.report_data_path
    manifest = root / "result_manifest.json"
    if manifest.exists():
        return manifest
    legacy_manifest = root / "MANIFEST.json"
    if legacy_manifest.exists():
        return legacy_manifest
    return root


def _workspace_local_artifacts(limit: int = 80) -> list[dict[str, Any]]:
    validation = _current_workspace_validation()
    if validation is None or validation.status == "blocked":
        return []
    records: list[dict[str, Any]] = []
    for row in validation.objects.get("results", []):
        if row.get("path_state") not in {"present", "external"}:
            continue
        root = _workspace_result_ref_root(row)
        if root is None or not root.exists():
            continue
        try:
            path = _workspace_primary_result_path(root)
            record = _local_artifact_record(path)
        except Exception as exc:
            log.warning("could not summarize workspace result %s: %s", row.get("result_id"), exc)
            continue
        record["source"] = "workspace"
        record["workspace_result_id"] = row.get("result_id")
        record["workspace_state"] = row.get("state")
        record["result_title"] = row.get("title")
        record["validation_state"] = row.get("validation_state")
        records.append(record)
    records.sort(key=lambda item: item["modified"], reverse=True)
    return records[:limit]


def _safe_local_path(path: str | Path) -> Path:
    """Resolve a user-supplied local artifact path inside allowed roots."""
    raw = Path(path)
    candidate = raw if raw.is_absolute() else Path.cwd() / raw
    resolved = candidate.resolve()
    for root in _local_data_roots():
        if not root.exists():
            continue
        try:
            resolved.relative_to(root)
            return resolved
        except ValueError:
            continue
    for root in _workspace_registered_result_roots():
        try:
            resolved.relative_to(root)
            return resolved
        except ValueError:
            continue
    raise HTTPException(status_code=400, detail="local data path is outside allowed roots")


def _is_allowed_local_path(path: Path) -> bool:
    """Return True when a path resolves inside a configured local data root."""
    resolved = path.resolve()
    for root in _local_data_roots():
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            continue
    for root in _workspace_registered_result_roots():
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _path_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        if child.is_file() and not child.is_symlink():
            try:
                total += child.stat().st_size
            except FileNotFoundError:
                continue
    return total


def _display_local_path(path: Path) -> str:
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.as_posix()


def _looks_like_landscape_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if path.name != "landscape" and not path.name.startswith("landscape_"):
        return False
    if (path / "report").is_dir():
        return True
    if any(path.glob("keywords*.parquet")):
        return True
    if any(path.glob("membership*.parquet")):
        return True
    return False


def _landscape_score(path: Path) -> tuple[int, int, str]:
    score = 0
    if path.name == "landscape":
        score += 10
    if (path / "report" / "data.json").exists():
        score += 4
    if any(path.glob("keywords*.parquet")):
        score += 3
    if any(path.glob("membership*.parquet")):
        score += 2
    if (path / "report").is_dir():
        score += 1
    try:
        mtime = int(path.stat().st_mtime)
    except FileNotFoundError:
        mtime = 0
    return (score, mtime, path.name)


def _find_landscape_dirs(output_dir: Path) -> list[Path]:
    if not output_dir.exists() or not output_dir.is_dir():
        return []
    return [child for child in output_dir.iterdir() if _looks_like_landscape_dir(child)]


def _infer_landscape_dir(output_dir: Path, selected_path: Path | None = None) -> Path | None:
    """Infer the landscape artifact directory, preferring the selected variant."""
    if selected_path is not None:
        current = selected_path if selected_path.is_dir() else selected_path.parent
        for candidate in [current, *current.parents]:
            if candidate == output_dir.parent:
                break
            if _looks_like_landscape_dir(candidate):
                return candidate

    exact = output_dir / "landscape"
    if _looks_like_landscape_dir(exact):
        return exact

    landscapes = _find_landscape_dirs(output_dir)
    if not landscapes:
        return None
    return max(landscapes, key=_landscape_score)


def _infer_output_dir(path: Path) -> Path | None:
    """Infer the containing SciScape output directory for a local artifact."""
    current = path if path.is_dir() else path.parent
    candidates = [current, *current.parents]
    for candidate in candidates:
        if candidate.name == "evolution" and (candidate / "evolution_manifest.json").exists():
            return candidate.parent
        if _looks_like_landscape_dir(candidate):
            return candidate.parent
        if _find_landscape_dirs(candidate):
            return candidate
        if (candidate / "evolution" / "evolution_manifest.json").exists():
            return candidate
        if (candidate / "edges.parquet").exists() or (candidate / "abstracts.parquet").exists():
            return candidate
    return None


def _ensure_local_result_table_exports(output_dir: Path) -> None:
    """Create small manifest-backed table exports for local result browsing."""

    try:
        from sciscape.artifacts import write_cooccurrence_artifacts, write_matrix_from_term_cooccurrence
        from sciscape.export import (
            export_cooccurrence_table,
            export_matrix_artifact,
            export_vosviewer_term_cooccurrence,
        )

        written = write_cooccurrence_artifacts(output_dir)
    except Exception as exc:
        # Local result opening should not fail just because an optional export
        # sidecar cannot be generated from a partial result root.
        log.warning("Could not create local result co-occurrence artifacts for %s: %s", output_dir, exc)
        return
    if written is None:
        return
    for export_name, export_fn in [
        ("cooccurrence_table", export_cooccurrence_table),
        ("vosviewer_term_cooccurrence", export_vosviewer_term_cooccurrence),
    ]:
        try:
            export_fn(output_dir)
        except Exception as exc:
            log.warning("Could not create local result %s export for %s: %s", export_name, output_dir, exc)
    try:
        matrix_written = write_matrix_from_term_cooccurrence(output_dir)
    except Exception as exc:
        log.warning("Could not create local result term co-occurrence matrix for %s: %s", output_dir, exc)
        matrix_written = None
    if matrix_written is not None:
        try:
            export_matrix_artifact(output_dir, export_format="vosviewer-network")
        except Exception as exc:
            log.warning("Could not create local result matrix VOSviewer export for %s: %s", output_dir, exc)


def _infer_local_result(path: Path) -> dict[str, Any]:
    """Build a completed-job result dict from an existing local output path."""
    output_dir = _infer_output_dir(path)
    if output_dir is None:
        raise HTTPException(status_code=400, detail="could not infer SciScape output directory")

    landscape_dir = _infer_landscape_dir(output_dir, selected_path=path)
    result: dict[str, Any] = {
        "output_dir": str(output_dir),
        "abstracts_path": None,
        "edges_path": None,
        "landscape_dir": str(landscape_dir) if landscape_dir else None,
        "landscape_rel_path": (
            landscape_dir.relative_to(output_dir).as_posix() if landscape_dir else None
        ),
        "n_edges": {},
    }
    abstracts = output_dir / "abstracts.parquet"
    if abstracts.exists():
        result["abstracts_path"] = str(abstracts)
    edges = output_dir / "edges.parquet"
    if edges.exists():
        result["edges_path"] = str(edges)
    _ensure_local_result_table_exports(output_dir)
    contract = validate_result_root(path).to_dict()
    manifest = load_result_manifest(path)
    result["artifact_contract"] = contract
    result["result_manifest"] = manifest
    result["run_state"] = _run_state_for_result(result, manifest)
    _attach_run_state_summary(result, manifest)
    result["features"] = contract["features"]
    result["feature_states"] = {
        name: feature.get("state", "hidden")
        for name, feature in manifest.get("features", {}).items()
    }
    result["result_state"] = contract["result_state"]
    _attach_report_atlas(result)
    _attach_narrative_summary(result)
    _attach_evolution_summary(result)
    return result


def _report_data_path_for_result(result: dict[str, Any]) -> Path | None:
    landscape_dir = result.get("landscape_dir")
    if landscape_dir:
        path = Path(landscape_dir) / "report" / "data.json"
        if path.exists():
            return path
    output_dir = result.get("output_dir")
    if output_dir:
        path = Path(output_dir) / "data.json"
        if path.exists():
            return path
    return None


def _first_landscape_file(result: dict[str, Any], pattern: str) -> Path | None:
    landscape_dir = result.get("landscape_dir")
    if not landscape_dir:
        return None
    root = Path(landscape_dir)
    if not root.exists():
        return None
    for path in root.glob(pattern):
        if path.is_file():
            return path
    return None


def _clean_export_query_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _split_export_id_list(value: Any, *, limit: int = 100) -> list[str]:
    text = _clean_export_query_value(value)
    if not text:
        return []
    return [part.strip() for part in text.split(",") if part.strip()][:limit]


def _atlas_cluster_uid_parts(uid: str, default_level: str | None) -> tuple[str, str] | None:
    text = _clean_export_query_value(uid)
    if not text:
        return None
    if ":" in text:
        level, key = text.split(":", 1)
    else:
        level, key = default_level or "cluster", text
    level = _clean_export_query_value(level) or "cluster"
    key = _clean_export_query_value(key)
    if not key:
        return None
    return level, key


def _membership_cluster_column(membership: Any, level: str | None) -> str | None:
    columns = list(getattr(membership, "columns", []) or [])
    if not columns:
        return None
    level = _clean_export_query_value(level) or "cluster"
    candidates: list[str] = []
    if level == "cluster":
        candidates.extend(["cluster", "cluster_cluster"])
    else:
        candidates.extend([f"cluster_{level}", level])
    candidates.extend([column for column in columns if str(column).startswith("cluster_")])
    if "cluster" in columns:
        candidates.append("cluster")
    for column in candidates:
        if column in columns:
            return column
    return None


def _filter_network_export_to_selection(
    *,
    edges: Any,
    membership: Any,
    abstracts: Any | None,
    selection: dict[str, Any],
) -> tuple[Any, Any, Any | None, dict[str, Any] | None]:
    subset = selection.get("subset") if isinstance(selection.get("subset"), dict) else {}
    subset_uids = _split_export_id_list(",".join(subset.get("uids", [])) if isinstance(subset.get("uids"), list) else subset.get("uids"))
    if not subset_uids or not hasattr(membership, "columns") or "uid" not in getattr(membership, "columns", []):
        return edges, membership, abstracts, None

    default_level = selection.get("cluster_level")
    cluster_keys_by_level: dict[str, set[str]] = {}
    for uid in subset_uids:
        parts = _atlas_cluster_uid_parts(uid, str(default_level) if default_level else None)
        if parts is None:
            continue
        level, key = parts
        cluster_keys_by_level.setdefault(level, set()).add(key)
    if not cluster_keys_by_level:
        return edges, membership, abstracts, None

    selected_level = str(default_level) if default_level in cluster_keys_by_level else next(iter(cluster_keys_by_level))
    cluster_column = _membership_cluster_column(membership, selected_level)
    if cluster_column is None:
        return edges, membership, abstracts, None

    import polars as pl

    selected_cluster_keys = sorted(cluster_keys_by_level[selected_level])
    source_edge_count = int(edges.height)
    source_node_count = int(membership.height)
    membership_with_key = membership.with_columns(pl.col(cluster_column).cast(pl.Utf8).alias("_sciscape_export_cluster_key"))
    filtered_membership = (
        membership_with_key
        .filter(pl.col("_sciscape_export_cluster_key").is_in(selected_cluster_keys))
        .drop("_sciscape_export_cluster_key")
    )
    selected_paper_uids = filtered_membership["uid"].cast(pl.Utf8).to_list()
    filtered_edges = edges.filter(
        pl.col("uid1").cast(pl.Utf8).is_in(selected_paper_uids)
        & pl.col("uid2").cast(pl.Utf8).is_in(selected_paper_uids)
    )
    filtered_abstracts = abstracts
    if abstracts is not None and hasattr(abstracts, "columns") and "uid" in abstracts.columns:
        filtered_abstracts = abstracts.filter(pl.col("uid").cast(pl.Utf8).is_in(selected_paper_uids))

    subset["applied"] = True
    subset["membership_column"] = cluster_column
    subset["cluster_level"] = selected_level
    subset["source_node_count"] = source_node_count
    subset["source_edge_count"] = source_edge_count
    subset["output_node_count"] = int(filtered_membership.height)
    subset["output_edge_count"] = int(filtered_edges.height)
    selection["subset"] = subset
    selection["scope"] = "selected_subset"
    layer_state = selection.get("layer_state") if isinstance(selection.get("layer_state"), dict) else {}
    layer_state["subset_applied"] = True
    layer_state["subset_membership_column"] = cluster_column
    layer_state["subset_output_node_count"] = int(filtered_membership.height)
    layer_state["subset_output_edge_count"] = int(filtered_edges.height)
    selection["layer_state"] = layer_state
    transform = {
        "transform_type": "apply_selected_subset",
        "description": "Filter exported network to the selected Atlas cluster subset.",
        "parameters": {
            "cluster_level": selected_level,
            "membership_column": cluster_column,
            "selected_cluster_count": len(selected_cluster_keys),
            "source_node_count": source_node_count,
            "source_edge_count": source_edge_count,
            "output_node_count": int(filtered_membership.height),
            "output_edge_count": int(filtered_edges.height),
            "truncated": bool(subset.get("truncated", False)),
        },
    }
    return filtered_edges, filtered_membership, filtered_abstracts, transform


def _web_network_export_selection(
    *,
    fmt: str,
    membership_path: Path | None,
    atlas_level: str | None = None,
    atlas_node: str | None = None,
    atlas_query: str | None = None,
    atlas_lens: str | None = None,
    atlas_view: str | None = None,
    atlas_focus: str | None = None,
    atlas_review: str | None = None,
    atlas_layers: str | None = None,
    atlas_edge_min: float | None = None,
    atlas_label_limit: int | None = None,
    atlas_neighbor: str | None = None,
    atlas_subset_mode: str | None = None,
    atlas_subset_count: int | None = None,
    atlas_subset_uids: str | None = None,
    atlas_subset_truncated: bool | None = None,
    atlas_pinned: str | None = None,
) -> dict[str, Any]:
    filters: list[dict[str, Any]] = []
    query = _clean_export_query_value(atlas_query)
    review = _clean_export_query_value(atlas_review)
    if query:
        filters.append({"field": "atlas_query", "op": "contains", "value": query})
    if review and review != "all":
        filters.append({"field": "atlas_review_state", "op": "eq", "value": review})

    thresholds: dict[str, Any] = {}
    if atlas_edge_min is not None and atlas_edge_min > 0:
        thresholds["atlas_edge_min"] = round(float(atlas_edge_min), 3)

    layers = [
        layer.strip()
        for layer in _clean_export_query_value(atlas_layers).split(",")
        if layer.strip() and layer.strip() != "none"
    ]
    layer_state: dict[str, Any] = {
        "network_format": fmt,
        "membership_source": str(membership_path) if membership_path else "",
    }
    lens = _clean_export_query_value(atlas_lens)
    view = _clean_export_query_value(atlas_view)
    if lens:
        layer_state["atlas_lens"] = lens
    if view:
        layer_state["atlas_view"] = view
    if layers or atlas_layers == "none":
        layer_state["atlas_layers"] = layers
    if atlas_label_limit is not None:
        layer_state["atlas_label_limit"] = int(atlas_label_limit)

    focus: dict[str, Any] = {}
    level = _clean_export_query_value(atlas_level)
    node = _clean_export_query_value(atlas_node)
    focus_mode = _clean_export_query_value(atlas_focus)
    neighbor = _clean_export_query_value(atlas_neighbor)
    if node:
        focus["cluster_uid"] = node
    if focus_mode and focus_mode != "global":
        focus["focus_mode"] = focus_mode
    if neighbor:
        focus["neighbor_uid"] = neighbor

    subset_mode = _clean_export_query_value(atlas_subset_mode) or focus_mode
    subset_uids = _split_export_id_list(atlas_subset_uids)
    pinned_uids = _split_export_id_list(atlas_pinned, limit=20)
    subset: dict[str, Any] = {}
    if subset_mode or subset_uids or atlas_subset_count is not None or pinned_uids:
        subset["mode"] = subset_mode or "global"
        if atlas_subset_count is not None:
            subset["count"] = int(atlas_subset_count)
        if subset_uids:
            subset["uids"] = subset_uids
        if atlas_subset_truncated is not None:
            subset["truncated"] = bool(atlas_subset_truncated)
        if pinned_uids:
            subset["pinned_uids"] = pinned_uids

    return {
        "scope": "full_result",
        "view": {"mode": "web_network_export", "surface": "web_export_endpoint"},
        "cluster_level": level or None,
        "filters": filters,
        "thresholds": thresholds,
        "layer_state": layer_state,
        "focus": focus,
        "subset": subset,
    }


def _attach_report_atlas(result: dict[str, Any]) -> None:
    """Attach the report-level Atlas payload when a viewer data file is present."""
    data_path = _report_data_path_for_result(result)
    if data_path is None:
        return
    try:
        report_data = json.loads(data_path.read_text(encoding="utf-8"))
    except Exception:
        return
    if not isinstance(report_data, dict):
        return

    meta = report_data.get("_sciscape")
    embedded_atlas = meta.get("atlas") if isinstance(meta, dict) else None
    membership_path = _first_landscape_file(result, "membership*.parquet")
    edges_path = result.get("edges_path")
    abstracts_path = result.get("abstracts_path")
    edge_evidence_paths = infer_result_artifacts(data_path).edge_evidence_paths
    atlas = build_atlas_payload_from_report_data(
        report_data,
        membership_path=membership_path,
        edges_path=edges_path,
        abstracts_path=abstracts_path,
        edge_evidence_paths=edge_evidence_paths,
    )
    if not isinstance(atlas, dict) or not atlas.get("nodes"):
        atlas = embedded_atlas
    if not isinstance(atlas, dict) or not atlas.get("nodes"):
        return

    result["atlas"] = atlas
    render_payload = build_atlas_render_payload(atlas)
    result["atlas_render_summary"] = {
        "schema_version": render_payload["schema_version"],
        "engine_family": render_payload["engine_family"],
        "node_count": render_payload["node_count"],
        "edge_count": render_payload["edge_count"],
        "label_count": render_payload["label_count"],
        "hierarchy_edge_count": render_payload["hierarchy_edge_count"],
        "coordinate_source": render_payload["view"]["coordinate_source"],
        "available_layers": sorted(render_payload["layers"].keys()),
    }
    output_dir = result.get("output_dir")
    try:
        result["atlas_report_rel_path"] = (
            data_path.relative_to(Path(output_dir)).as_posix()
            if output_dir
            else str(data_path)
        )
    except ValueError:
        result["atlas_report_rel_path"] = str(data_path)


def _atlas_render_payload_for_result(result: dict[str, Any]) -> dict[str, Any] | None:
    atlas = result.get("atlas")
    if not isinstance(atlas, dict) or not atlas.get("nodes"):
        _attach_report_atlas(result)
        _attach_narrative_summary(result)
        atlas = result.get("atlas")
    if not isinstance(atlas, dict) or not atlas.get("nodes"):
        return None
    return build_atlas_render_payload(atlas)


def _json_safe(value: Any) -> Any:
    """Convert pandas/numpy-ish values to strict JSON-friendly values."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    try:
        import pandas as pd

        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            return str(value)
    return value


def _read_evolution_table(path: Path, *, limit: int) -> tuple[list[dict[str, Any]], bool]:
    import pandas as pd

    df = pd.read_parquet(path)
    truncated = len(df) > limit
    if truncated:
        df = df.head(limit)
    return [_json_safe(row) for row in df.to_dict(orient="records")], truncated


def _parse_evolution_refs(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = [part.strip() for part in text.replace(",", "|").split("|")]
        if isinstance(payload, list):
            return [str(item) for item in payload if str(item).strip()]
        return [str(payload)] if str(payload).strip() else []
    return [str(value)] if str(value).strip() else []


def _normalize_evolution_rows(payload: dict[str, Any]) -> None:
    for state in payload.get("cluster_states", []):
        terms = _parse_evolution_refs(state.get("top_terms"))
        state["top_terms"] = terms[:8]
    for transition in payload.get("transitions", []):
        transition["score"] = _json_safe(transition.get("score"))
    for lineage in payload.get("lineages", []):
        lineage["event_refs"] = _parse_evolution_refs(lineage.get("event_refs"))
    for event in payload.get("events", []):
        event["transition_refs"] = _parse_evolution_refs(event.get("transition_refs"))
        event["source_state_ids"] = _parse_evolution_refs(event.get("source_state_ids"))
        event["target_state_ids"] = _parse_evolution_refs(event.get("target_state_ids"))


def _summarize_evolution_state_membership(payload: dict[str, Any]) -> dict[str, Any]:
    rows = [row for row in payload.get("state_membership", []) if isinstance(row, dict)]
    by_state: dict[str, dict[str, Any]] = {}
    for row in rows:
        state_id = str(row.get("state_id") or "").strip()
        uid = str(row.get("uid") or "").strip()
        if not state_id:
            continue
        summary = by_state.setdefault(
            state_id,
            {
                "state_id": state_id,
                "loaded_count": 0,
                "uid_samples": [],
            },
        )
        summary["loaded_count"] = int(summary.get("loaded_count", 0)) + 1
        samples = summary.setdefault("uid_samples", [])
        if uid and len(samples) < 5:
            samples.append(uid)
    return {
        "loaded_rows": len(rows),
        "truncated": bool((payload.get("truncated") or {}).get("state_membership")),
        "by_state": by_state,
    }


def _evolution_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        number = float(value)
        if math.isfinite(number):
            return int(number)
    except Exception:
        pass
    return default


def _evolution_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        number = float(value)
        if math.isfinite(number):
            return float(number)
    except Exception:
        pass
    return default


def _build_evolution_map_payload(
    payload: dict[str, Any],
    *,
    node_limit: int = 240,
    edge_limit: int = 360,
    lineage_limit: int = 80,
) -> dict[str, Any]:
    """Derive a bounded lineage-by-time map from validated evolution rows."""

    slices = sorted(
        [row for row in payload.get("time_slices", []) if isinstance(row, dict)],
        key=lambda row: (_evolution_int(row.get("slice_index")), str(row.get("slice_id") or "")),
    )
    states = [row for row in payload.get("cluster_states", []) if isinstance(row, dict)]
    transitions = [row for row in payload.get("transitions", []) if isinstance(row, dict)]
    lineages = [row for row in payload.get("lineages", []) if isinstance(row, dict)]
    events = [row for row in payload.get("events", []) if isinstance(row, dict)]
    if not slices or not states:
        return {
            "schema_version": "sciscape_evolution_map_v1",
            "available": False,
            "reason": "time slices or cluster states are unavailable",
            "layout": "lineage_time_grid",
            "nodes": [],
            "edges": [],
            "events": [],
            "lineages": [],
            "slices": [],
        }

    node_limit = max(0, int(node_limit))
    edge_limit = max(0, int(edge_limit))
    lineage_limit = max(1, int(lineage_limit))
    slice_ids = [str(row.get("slice_id") or "") for row in slices]
    slice_count = len(slice_ids)
    slice_x = {
        slice_id: (0.5 if slice_count <= 1 else index / max(1, slice_count - 1))
        for index, slice_id in enumerate(slice_ids)
    }
    slice_index_by_id = {
        str(row.get("slice_id") or ""): _evolution_int(row.get("slice_index"))
        for row in slices
    }
    state_by_id = {
        str(row.get("state_id") or ""): row
        for row in states
        if str(row.get("state_id") or "").strip()
    }
    lineage_by_state: dict[str, str] = {}
    lineage_meta: dict[str, dict[str, Any]] = {}
    for row in lineages:
        state_id = str(row.get("state_id") or "").strip()
        lineage_id = str(row.get("lineage_id") or "").strip()
        if not lineage_id:
            continue
        if state_id and state_id not in lineage_by_state:
            lineage_by_state[state_id] = lineage_id
        meta = lineage_meta.setdefault(
            lineage_id,
            {
                "lineage_id": lineage_id,
                "label": str(row.get("lineage_label") or lineage_id),
                "state_count": 0,
                "max_doc_count": 0,
                "total_doc_count": 0,
                "min_slice_index": _evolution_int(row.get("slice_index"), 999999),
                "stability_scores": [],
                "roles": {},
                "state_ids": set(),
            },
        )
        if state_id:
            meta.setdefault("state_ids", set()).add(state_id)
        meta["state_count"] = len(meta.get("state_ids") or []) or int(meta.get("state_count", 0))
        meta["min_slice_index"] = min(int(meta.get("min_slice_index", 999999)), _evolution_int(row.get("slice_index"), 999999))
        score = _evolution_float(row.get("stability_score"), default=float("nan"))
        if math.isfinite(score):
            meta["stability_scores"].append(score)
        role = str(row.get("role") or "").strip()
        if role:
            roles = meta.setdefault("roles", {})
            roles[role] = int(roles.get(role, 0)) + 1

    for row in states:
        state_id = str(row.get("state_id") or "").strip()
        if not state_id:
            continue
        lineage_id = lineage_by_state.get(state_id)
        if not lineage_id:
            lineage_id = f"lineage:{row.get('cluster_key') or state_id}"
            lineage_by_state[state_id] = lineage_id
        doc_count = _evolution_int(row.get("doc_count"))
        slice_index = _evolution_int(row.get("slice_index"), slice_index_by_id.get(str(row.get("slice_id") or ""), 999999))
        meta = lineage_meta.setdefault(
            lineage_id,
            {
                "lineage_id": lineage_id,
                "label": str(row.get("cluster_label") or row.get("cluster_key") or lineage_id),
                "state_count": 0,
                "max_doc_count": 0,
                "total_doc_count": 0,
                "min_slice_index": slice_index,
                "stability_scores": [],
                "roles": {},
                "state_ids": set(),
            },
        )
        meta.setdefault("state_ids", set()).add(state_id)
        meta["state_count"] = len(meta.get("state_ids") or []) or int(meta.get("state_count", 0))
        meta["max_doc_count"] = max(int(meta.get("max_doc_count", 0)), doc_count)
        meta["total_doc_count"] = int(meta.get("total_doc_count", 0)) + doc_count
        meta["min_slice_index"] = min(int(meta.get("min_slice_index", 999999)), slice_index)

    ordered_lineages = sorted(
        lineage_meta.values(),
        key=lambda row: (
            int(row.get("min_slice_index", 999999)),
            -int(row.get("max_doc_count", 0)),
            str(row.get("label") or row.get("lineage_id") or ""),
        ),
    )
    total_lineage_count = len(ordered_lineages)
    visible_lineages = ordered_lineages[:lineage_limit]
    visible_lineage_ids = {str(row["lineage_id"]) for row in visible_lineages}
    row_index_by_lineage = {
        str(row["lineage_id"]): index
        for index, row in enumerate(visible_lineages)
    }
    visible_lineage_count = max(1, len(visible_lineages))

    event_types_by_state: dict[str, set[str]] = {}
    for event in events:
        event_type = str(event.get("event_type") or "").strip()
        if not event_type:
            continue
        anchors = [str(event.get("state_id") or "").strip()]
        anchors.extend(_parse_evolution_refs(event.get("source_state_ids")))
        anchors.extend(_parse_evolution_refs(event.get("target_state_ids")))
        for state_id in anchors:
            if state_id and state_id in state_by_id:
                event_types_by_state.setdefault(state_id, set()).add(event_type)

    node_source = []
    for row in states:
        state_id = str(row.get("state_id") or "").strip()
        lineage_id = lineage_by_state.get(state_id, "")
        if state_id and lineage_id in visible_lineage_ids:
            node_source.append(row)
    total_visible_states = len(node_source)
    node_source = sorted(
        node_source,
        key=lambda row: (
            row_index_by_lineage.get(lineage_by_state.get(str(row.get("state_id") or ""), ""), 999999),
            _evolution_int(row.get("slice_index"), slice_index_by_id.get(str(row.get("slice_id") or ""), 999999)),
            -_evolution_int(row.get("doc_count")),
            str(row.get("state_id") or ""),
        ),
    )[:node_limit]

    nodes: list[dict[str, Any]] = []
    for row in node_source:
        state_id = str(row.get("state_id") or "")
        lineage_id = lineage_by_state.get(state_id, "")
        row_index = row_index_by_lineage.get(lineage_id, 0)
        slice_id = str(row.get("slice_id") or "")
        y = 0.5 if visible_lineage_count <= 1 else row_index / max(1, visible_lineage_count - 1)
        event_types = sorted(event_types_by_state.get(state_id, set()))
        nodes.append(
            {
                "state_id": state_id,
                "slice_id": slice_id,
                "slice_index": _evolution_int(row.get("slice_index"), slice_index_by_id.get(slice_id, 0)),
                "lineage_id": lineage_id,
                "lineage_row": row_index,
                "x": round(float(slice_x.get(slice_id, 0.5)), 6),
                "y": round(float(y), 6),
                "cluster_key": row.get("cluster_key"),
                "cluster_label": row.get("cluster_label"),
                "cluster_uid": row.get("cluster_uid"),
                "doc_count": _evolution_int(row.get("doc_count")),
                "top_terms": row.get("top_terms") if isinstance(row.get("top_terms"), list) else _parse_evolution_refs(row.get("top_terms")),
                "event_types": event_types,
                "primary_event_type": next((event for event in ["split", "merge", "ambiguous", "emergence", "decline", "continuation"] if event in event_types), event_types[0] if event_types else ""),
            }
        )
    visible_state_ids = {node["state_id"] for node in nodes}
    node_coord = {
        node["state_id"]: (node["x"], node["y"])
        for node in nodes
    }

    edge_source = []
    for row in transitions:
        source_id = str(row.get("source_state_id") or "")
        target_id = str(row.get("target_state_id") or "")
        if source_id in visible_state_ids and target_id in visible_state_ids:
            edge_source.append(row)
    total_visible_edges = len(edge_source)
    edge_source = sorted(
        edge_source,
        key=lambda row: (
            _evolution_int(row.get("source_slice_id"), slice_index_by_id.get(str(row.get("source_slice_id") or ""), 0)),
            -_evolution_float(row.get("score")),
            str(row.get("transition_id") or ""),
        ),
    )[:edge_limit]
    edges = []
    for row in edge_source:
        source_id = str(row.get("source_state_id") or "")
        target_id = str(row.get("target_state_id") or "")
        source_x, source_y = node_coord[source_id]
        target_x, target_y = node_coord[target_id]
        edges.append(
            {
                "transition_id": row.get("transition_id"),
                "source_state_id": source_id,
                "target_state_id": target_id,
                "relation": row.get("relation"),
                "score": _evolution_float(row.get("score")),
                "support_count": _evolution_int(row.get("support_count")),
                "source_x": round(float(source_x), 6),
                "source_y": round(float(source_y), 6),
                "target_x": round(float(target_x), 6),
                "target_y": round(float(target_y), 6),
            }
        )

    map_events = []
    for event in events:
        anchor_ids = [str(event.get("state_id") or "").strip()]
        anchor_ids.extend(_parse_evolution_refs(event.get("target_state_ids")))
        anchor_ids.extend(_parse_evolution_refs(event.get("source_state_ids")))
        anchor_id = next((state_id for state_id in anchor_ids if state_id in node_coord), "")
        if not anchor_id:
            continue
        x, y = node_coord[anchor_id]
        map_events.append(
            {
                "event_id": event.get("event_id"),
                "event_type": event.get("event_type"),
                "state_id": anchor_id,
                "slice_id": event.get("slice_id"),
                "x": round(float(x), 6),
                "y": round(float(y), 6),
                "score": _evolution_float(event.get("score")),
                "support_count": _evolution_int(event.get("support_count")),
                "transition_refs": _parse_evolution_refs(event.get("transition_refs")),
            }
        )

    map_lineages = []
    for row in visible_lineages:
        lineage_id = str(row["lineage_id"])
        scores = row.get("stability_scores") or []
        stability = float(sum(scores) / len(scores)) if scores else None
        map_lineages.append(
            {
                "lineage_id": lineage_id,
                "label": row.get("label"),
                "row": row_index_by_lineage[lineage_id],
                "state_count": int(row.get("state_count", 0)),
                "max_doc_count": int(row.get("max_doc_count", 0)),
                "total_doc_count": int(row.get("total_doc_count", 0)),
                "stability_score": round(stability, 6) if stability is not None else None,
                "roles": row.get("roles", {}),
            }
        )

    return _json_safe(
        {
            "schema_version": "sciscape_evolution_map_v1",
            "available": True,
            "layout": "lineage_time_grid",
            "slices": [
                {
                    "slice_id": str(row.get("slice_id") or ""),
                    "slice_index": _evolution_int(row.get("slice_index")),
                    "slice_label": row.get("slice_label") or row.get("slice_id"),
                    "x": round(float(slice_x.get(str(row.get("slice_id") or ""), 0.5)), 6),
                    "doc_count": _evolution_int(row.get("doc_count")),
                    "active_cluster_count": _evolution_int(row.get("active_cluster_count")),
                }
                for row in slices
            ],
            "lineages": map_lineages,
            "nodes": nodes,
            "edges": edges,
            "events": map_events,
            "slice_count": len(slices),
            "lineage_count": len(map_lineages),
            "node_count": len(nodes),
            "edge_count": len(edges),
            "event_count": len(map_events),
            "total_lineage_count": total_lineage_count,
            "total_node_count": len(states),
            "total_edge_count": len(transitions),
            "eligible_node_count": total_visible_states,
            "eligible_edge_count": total_visible_edges,
            "hidden_lineage_count": max(total_lineage_count - len(map_lineages), 0),
            "hidden_node_count": max(len(states) - len(nodes), 0),
            "hidden_edge_count": max(len(transitions) - len(edges), 0),
            "truncated": {
                "lineages": total_lineage_count > len(map_lineages),
                "nodes": total_visible_states > len(nodes),
                "edges": total_visible_edges > len(edges),
            },
            "limits": {
                "lineages": lineage_limit,
                "nodes": node_limit,
                "edges": edge_limit,
            },
        }
    )


def _evolution_manifest_path_for_result(result: dict[str, Any]) -> Path | None:
    output_dir = result.get("output_dir")
    if not output_dir:
        return None
    try:
        artifacts = infer_result_artifacts(output_dir)
    except Exception:
        return None
    return artifacts.evolution_manifest_paths[0] if artifacts.evolution_manifest_paths else None


def _load_evolution_payload(
    result: dict[str, Any],
    *,
    state_limit: int = 120,
    transition_limit: int = 180,
    event_limit: int = 180,
    lineage_limit: int = 180,
    state_membership_limit: int = 2000,
    include_map: bool = True,
    map_node_limit: int = 240,
    map_edge_limit: int = 360,
    map_lineage_limit: int = 80,
) -> dict[str, Any]:
    manifest_path = _evolution_manifest_path_for_result(result)
    if manifest_path is None:
        return {"available": False, "reason": "no evolution artifact"}

    evolution_dir = manifest_path.parent
    validation = validate_evolution_artifact(manifest_path).to_dict()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "available": True,
            "status": "blocked",
            "reason": f"could not read evolution manifest: {exc}",
            "validation": validation,
        }
    outputs = manifest.get("outputs") if isinstance(manifest.get("outputs"), dict) else {}
    table_specs = {
        "time_slices": ("time_slices", 500),
        "cluster_states": ("cluster_states", state_limit),
        "transitions": ("transitions", transition_limit),
        "lineages": ("lineages", lineage_limit),
        "events": ("events", event_limit),
        "state_membership": ("state_membership", state_membership_limit),
    }
    tables: dict[str, list[dict[str, Any]]] = {}
    truncated: dict[str, bool] = {}
    errors: list[dict[str, Any]] = []
    for key, (output_key, limit) in table_specs.items():
        rel_path = outputs.get(output_key)
        if not rel_path:
            tables[key] = []
            truncated[key] = False
            continue
        path = evolution_dir / str(rel_path)
        try:
            rows, is_truncated = _read_evolution_table(path, limit=limit)
        except Exception as exc:
            rows, is_truncated = [], False
            errors.append({"code": "evolution_table_read_failed", "table": key, "message": str(exc)})
        tables[key] = rows
        truncated[key] = is_truncated
    payload = {
        "available": True,
        "schema_version": manifest.get("schema_version"),
        "evolution_id": manifest.get("evolution_id"),
        "title": manifest.get("title"),
        "status": validation.get("status"),
        "feature_state": (result.get("feature_states") or {}).get("evolution"),
        "manifest_path": str(manifest_path),
        "counts": validation.get("counts", {}),
        "event_counts": validation.get("event_counts", {}),
        "warnings": validation.get("warnings", []),
        "blocking_issues": validation.get("blocking_issues", []),
        "matching_method": manifest.get("matching_method", {}),
        "event_rules": manifest.get("event_rules", {}),
        "slice_method": manifest.get("slice_method", {}),
        "entity_scope": manifest.get("entity_scope", {}),
        "paths": validation.get("paths", {}),
        "truncated": truncated,
        "read_errors": errors,
        **tables,
    }
    _normalize_evolution_rows(payload)
    payload["state_membership_summary"] = _summarize_evolution_state_membership(payload)
    if include_map:
        payload["evolution_map"] = _build_evolution_map_payload(
            payload,
            node_limit=map_node_limit,
            edge_limit=map_edge_limit,
            lineage_limit=map_lineage_limit,
        )
    return payload


def _attach_evolution_summary(result: dict[str, Any]) -> None:
    payload = _load_evolution_payload(
        result,
        state_limit=0,
        transition_limit=0,
        event_limit=0,
        lineage_limit=0,
        state_membership_limit=0,
        include_map=False,
    )
    if not payload.get("available"):
        return
    result["evolution_summary"] = {
        "available": True,
        "evolution_id": payload.get("evolution_id"),
        "title": payload.get("title"),
        "status": payload.get("status"),
        "feature_state": payload.get("feature_state"),
        "counts": payload.get("counts", {}),
        "event_counts": payload.get("event_counts", {}),
        "warning_count": len(payload.get("warnings", [])),
        "blocking_issue_count": len(payload.get("blocking_issues", [])),
    }


def _artifact_role(path: Path) -> str:
    name = path.name
    if name == "data.json":
        return "viewer_data"
    if name == "evolution_manifest.json":
        return "evolution"
    if name == "keywords.parquet":
        return "keywords"
    if name == "membership.parquet":
        return "membership"
    if name == "report.html":
        return "report"
    if name == "index.html" and path.parent.name == "report":
        return "dashboard"
    if path.is_dir() and _find_landscape_dirs(path):
        return "output_dir"
    return "artifact"


def _local_evolution_summary(manifest_path: Path | None) -> dict[str, Any]:
    if manifest_path is None or not manifest_path.exists():
        return {}
    try:
        validation = validate_evolution_artifact(manifest_path).to_dict()
    except Exception as exc:
        return {
            "status": "blocked",
            "counts": {},
            "event_counts": {},
            "warning_count": 0,
            "blocking_issue_count": 1,
            "error": str(exc),
        }
    return {
        "status": validation.get("status"),
        "counts": validation.get("counts", {}),
        "event_counts": validation.get("event_counts", {}),
        "warning_count": len(validation.get("warnings", [])),
        "blocking_issue_count": len(validation.get("blocking_issues", [])),
    }


def _local_artifact_record(path: Path) -> dict[str, Any]:
    output_dir = _infer_output_dir(path)
    relative_path = _display_local_path(path)
    artifact_id = hashlib.sha1(relative_path.encode("utf-8")).hexdigest()[:12]
    landscape_dir = _infer_landscape_dir(output_dir, selected_path=path) if output_dir is not None else None
    data_json = landscape_dir / "report" / "data.json" if landscape_dir else None
    keywords = landscape_dir / "keywords.parquet" if landscape_dir else None
    membership = landscape_dir / "membership.parquet" if landscape_dir else None
    evolution_manifest = output_dir / "evolution" / "evolution_manifest.json" if output_dir else None
    evolution_summary = _local_evolution_summary(evolution_manifest)
    return {
        "id": artifact_id,
        "path": relative_path,
        "name": path.name,
        "role": _artifact_role(path),
        "size_bytes": _path_size(path),
        "modified": int(path.stat().st_mtime),
        "output_dir": _display_local_path(output_dir) if output_dir else None,
        "landscape_dir": _display_local_path(landscape_dir) if landscape_dir else None,
        "landscape_name": landscape_dir.name if landscape_dir else None,
        "has_web_result": output_dir is not None,
        "has_data_json": bool(data_json and data_json.exists()),
        "has_keywords": bool(keywords and keywords.exists()),
        "has_membership": bool(membership and membership.exists()),
        "has_evolution": bool(evolution_manifest and evolution_manifest.exists()),
        "evolution_status": evolution_summary.get("status"),
        "evolution_counts": evolution_summary.get("counts", {}),
        "evolution_event_counts": evolution_summary.get("event_counts", {}),
        "evolution_warning_count": evolution_summary.get("warning_count", 0),
        "evolution_blocking_issue_count": evolution_summary.get("blocking_issue_count", 0),
    }


def _discover_legacy_local_artifacts(limit: int = 80) -> list[dict[str, Any]]:
    """Find local SciScape outputs by scanning legacy output roots."""
    candidates: dict[str, Path] = {}
    globs = [
        "**/landscape/report/data.json",
        "**/landscape/report/index.html",
        "**/landscape/report/report.html",
        "**/landscape/keywords.parquet",
        "**/landscape/membership.parquet",
        "**/evolution/evolution_manifest.json",
        "**/data.json",
    ]
    for root in _local_data_roots():
        if not root.exists() or not root.is_dir():
            continue
        for pattern in globs:
            for path in root.glob(pattern):
                if path.is_file():
                    candidates[_display_local_path(path)] = path
        for child in root.iterdir():
            if child.is_dir() and (child / "landscape").is_dir():
                candidates[_display_local_path(child)] = child

    records = []
    for path in candidates.values():
        record = _local_artifact_record(path)
        record["source"] = "legacy_scan"
        records.append(record)
    records.sort(key=lambda item: item["modified"], reverse=True)
    return records[:limit]


def _discover_local_artifacts(limit: int = 80) -> tuple[list[dict[str, Any]], str]:
    """Find local SciScape outputs, preferring registered workspace results."""
    workspace_records = _workspace_local_artifacts(limit=limit)
    if workspace_records:
        return workspace_records, "workspace_manifest"
    return _discover_legacy_local_artifacts(limit=limit), "legacy_scan"


def _demo_manifest_file() -> Path:
    manifest = _DEMO_MANIFEST_PATH
    return manifest if manifest.is_absolute() else (Path.cwd() / manifest).resolve()


def _load_demo_manifest() -> dict[str, Any]:
    manifest_path = _demo_manifest_file()
    if not manifest_path.exists():
        return {"schema_version": 1, "presets": {}}
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"invalid demo manifest: {exc}") from exc
    if not isinstance(data, dict):
        raise HTTPException(status_code=500, detail="invalid demo manifest")
    return data


def _demo_expected_artifacts(preset: dict[str, Any]) -> list[str]:
    artifacts = preset.get("expected_artifacts", _DEFAULT_DEMO_ARTIFACTS)
    if not isinstance(artifacts, list):
        return list(_DEFAULT_DEMO_ARTIFACTS)
    return [str(item) for item in artifacts if isinstance(item, str) and item]


def _demo_missing_artifacts(output_dir: Path, preset: dict[str, Any]) -> list[str]:
    return [
        rel_path
        for rel_path in _demo_expected_artifacts(preset)
        if not (output_dir / rel_path).exists()
    ]


def _demo_candidate_mtime(output_dir: Path, preset: dict[str, Any]) -> int:
    mtimes: list[int] = []
    for rel_path in ["", *_demo_expected_artifacts(preset)]:
        path = output_dir / rel_path if rel_path else output_dir
        try:
            mtimes.append(int(path.stat().st_mtime))
        except FileNotFoundError:
            continue
    return max(mtimes, default=0)


def _demo_candidate_score(output_dir: Path, preset: dict[str, Any]) -> tuple[int, int, int, str]:
    missing = _demo_missing_artifacts(output_dir, preset)
    score = 0
    if not missing:
        score += 100
    if (output_dir / "landscape" / "report" / "data.json").exists():
        score += 30
    if output_dir.parent.name == "openalex_live":
        score += 20
    elif output_dir.parent.name.startswith("openalex_live_"):
        score += 15
    if _infer_landscape_dir(output_dir) is not None:
        score += 5
    return (score, -len(missing), _demo_candidate_mtime(output_dir, preset), output_dir.as_posix())


def _add_demo_candidate(candidates: dict[str, Path], path: Path) -> None:
    if not path.exists() or not path.is_dir():
        return
    if not _is_allowed_local_path(path):
        return
    try:
        candidates[str(path.resolve())] = path.resolve()
    except FileNotFoundError:
        return


def _find_demo_output_dir(slug: str, preset: dict[str, Any], manifest: dict[str, Any]) -> Path | None:
    """Find the best existing output directory for a curated demo slug."""
    candidates: dict[str, Path] = {}

    default_root = manifest.get("default_output_root")
    if isinstance(default_root, str) and default_root:
        root = Path(default_root)
        root = root if root.is_absolute() else Path.cwd() / root
        _add_demo_candidate(candidates, root / slug)

    for root in _local_data_roots():
        if not root.exists() or not root.is_dir():
            continue
        _add_demo_candidate(candidates, root / slug)
        _add_demo_candidate(candidates, root / "openalex_live" / slug)
        for child in root.iterdir():
            if child.is_dir() and child.name.startswith("openalex_live"):
                _add_demo_candidate(candidates, child / slug)
        for path in root.rglob(slug):
            _add_demo_candidate(candidates, path)

    if not candidates:
        return None
    return max(candidates.values(), key=lambda path: _demo_candidate_score(path, preset))


def _demo_run_command(key: str) -> str:
    return (
        "uv run --extra dev python examples/openalex_live_demo.py "
        f"--preset {key} --email you@example.org"
    )


def _demo_preset_record(key: str, preset: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    slug = str(preset.get("slug") or key)
    output_dir = _find_demo_output_dir(slug, preset, manifest)
    primary_path = output_dir / "landscape" / "report" / "data.json" if output_dir else None
    missing = _demo_missing_artifacts(output_dir, preset) if output_dir else _demo_expected_artifacts(preset)
    can_open = bool(primary_path and primary_path.exists())
    status = "available" if can_open and not missing else "partial" if can_open else "missing"
    return {
        "key": key,
        "title": str(preset.get("title") or key),
        "slug": slug,
        "query": str(preset.get("query") or ""),
        "filters": preset.get("filters") if isinstance(preset.get("filters"), dict) else {},
        "max_works": preset.get("max_works"),
        "status": status,
        "can_open": can_open,
        "output_dir": _display_local_path(output_dir) if output_dir else None,
        "primary_path": _display_local_path(primary_path) if primary_path and primary_path.exists() else None,
        "missing_artifacts": missing,
        "expected_artifacts": _demo_expected_artifacts(preset),
        "run_command": _demo_run_command(key),
    }


@app.get("/api/demo-presets")
async def list_demo_presets():
    """List curated demo presets and any matching local generated outputs."""
    manifest = _load_demo_manifest()
    presets = manifest.get("presets", {})
    if not isinstance(presets, dict):
        raise HTTPException(status_code=500, detail="invalid demo manifest presets")
    return {
        "manifest_path": _display_local_path(_demo_manifest_file()),
        "demos": [
            _demo_preset_record(str(key), preset, manifest)
            for key, preset in presets.items()
            if isinstance(preset, dict)
        ],
    }


@app.get("/api/local-data")
async def list_local_data(limit: int = 80):
    """List existing local SciScape output artifacts the web app can open."""
    artifacts, discovery_source = _discover_local_artifacts(limit=limit)
    return {
        "workspace": _workspace_summary(),
        "discovery_source": discovery_source,
        "roots": [_display_local_path(root) for root in _local_data_roots()],
        "artifacts": artifacts,
        "expected_files": [
            "workspace/web_output/<job>/landscape/report/data.json",
            "workspace/examples_output/<demo>/landscape/keywords.parquet",
            "workspace/examples_output/<demo>/landscape/membership.parquet",
            "workspace/examples_output/<demo>/evolution/evolution_manifest.json",
            "viewer/data.json",
        ],
    }


@app.post("/api/local-data/open")
async def open_local_data(req: LocalDataOpenRequest):
    """Register an existing local SciScape output as a completed web job."""
    path = _safe_local_path(req.path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="local data path not found")

    result = _infer_local_result(path)
    rel = _display_local_path(path)
    job_id = "local" + hashlib.sha1(rel.encode("utf-8")).hexdigest()[:8]
    _jobs.create(job_id, {"query": f"Local output: {rel}"})
    job = _jobs[job_id]
    job["status"] = "done"
    job["progress"] = [
        f"Loaded local SciScape output from {rel}",
        "No fetch or clustering was run for this local view.",
    ]
    job["result"] = result
    _jobs.persist(job_id)
    return {"job_id": job_id, "result": result}


@app.get("/api/jobs/{job_id}/labels")
async def get_labels(job_id: str, strategy: str = "tfidf_distinct", top_k: int = 3):
    """Auto-generate cluster labels from keywords (no LLM needed)."""
    job = _jobs.get(job_id)
    if not job or job["status"] != "done":
        return {"error": "job not done"}

    result = job.get("result", {})
    landscape_dir = result.get("landscape_dir")
    if not landscape_dir:
        return {"error": "no landscape output"}

    import polars as pl
    from sciscape.clustering.auto_label import auto_label_clusters

    # Find keywords file
    ld = Path(landscape_dir)
    kw_path = None
    for f in ld.glob("keywords*.parquet"):
        kw_path = f
        break
    if not kw_path:
        return {"error": "no keywords file found"}

    try:
        kw_df = pl.read_parquet(kw_path)
        labels = auto_label_clusters(kw_df, strategy=strategy, top_k=top_k)
        return {"labels": {str(k): v for k, v in labels.items()}}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/jobs/{job_id}/labels/llm")
async def generate_llm_labels(job_id: str, bg: BackgroundTasks):
    """Generate cluster names using LLM (requires Ollama or OpenAI API)."""
    job = _jobs.get(job_id)
    if not job or job["status"] != "done":
        return {"error": "job not done"}

    # Store LLM labeling status
    if "llm_labels" not in job:
        job["llm_labels"] = {"status": "running", "labels": {}}
    bg.add_task(_run_llm_labeling, job_id)
    return {"status": "started"}


def _run_llm_labeling(job_id: str) -> None:
    """Background task for LLM cluster naming."""
    job = _jobs.get(job_id)
    if not job:
        return

    try:
        import polars as pl
        from sciscape.clustering.cluster_naming import create_client, summarise_cluster
        from sciscape.clustering.core_documents import ClusterDocument

        result = job.get("result", {})
        output_dir = Path(result.get("output_dir", ""))
        landscape_dir = result.get("landscape_dir")

        # Load abstracts
        abs_path = result.get("abstracts_path")
        if not abs_path:
            job["llm_labels"] = {"status": "error", "error": "no abstracts"}
            return

        abs_df = pl.read_parquet(abs_path)

        # Load membership
        mem_path = None
        if landscape_dir:
            for f in Path(landscape_dir).glob("membership*.parquet"):
                mem_path = f
                break
        if not mem_path:
            job["llm_labels"] = {"status": "error", "error": "no membership"}
            return

        mem_df = pl.read_parquet(mem_path)
        cluster_cols = [c for c in mem_df.columns if c.startswith("cluster_")]
        if not cluster_cols:
            job["llm_labels"] = {"status": "error", "error": "no cluster columns found"}
            return
        cluster_col = cluster_cols[0]

        # Join
        joined = abs_df.join(mem_df.select("uid", cluster_col), on="uid", how="inner")

        # Group docs by cluster
        client = create_client()
        model = getattr(client, "_sciscape_model", "gpt-oss:20b")
        labels = {}

        cluster_ids = sorted(joined[cluster_col].unique().to_list())
        for cid in cluster_ids:
            docs_df = joined.filter(pl.col(cluster_col) == cid).head(8)
            docs = [
                ClusterDocument(
                    uid=row["uid"],
                    title=row.get("title", ""),
                    abstract=row.get("abstract", ""),
                )
                for row in docs_df.iter_rows(named=True)
            ]
            if not docs:
                continue
            try:
                summary = summarise_cluster(client, str(cid), docs, model=model)
                labels[str(cid)] = {
                    "name": summary.name,
                    "description": summary.description,
                    "keywords": summary.keywords,
                }
            except Exception as e:
                labels[str(cid)] = {"name": f"Cluster {cid}", "error": str(e)}

        job["llm_labels"] = {"status": "done", "labels": labels}
        _jobs.persist(job_id)

    except Exception as e:
        job["llm_labels"] = {"status": "error", "error": str(e)}
        _jobs.persist(job_id)


@app.get("/api/jobs/{job_id}/labels/llm/status")
async def llm_label_status(job_id: str):
    """Check LLM labeling progress."""
    job = _jobs.get(job_id)
    if not job:
        return {"error": "job not found"}
    return job.get("llm_labels", {"status": "not_started"})


@app.get("/api/jobs/{job_id}/temporal")
async def get_temporal(job_id: str):
    """Get per-year cluster network snapshots for temporal playback."""
    job = _jobs.get(job_id)
    if not job or job["status"] != "done":
        return {"error": "job not done"}
    result = job.get("result", {})
    from .network_data import build_temporal_snapshots

    edges_path = result.get("edges_path")
    landscape_dir = result.get("landscape_dir")
    if not edges_path or not landscape_dir:
        return {"error": "missing files"}

    mem_path = None
    for f in Path(landscape_dir).glob("membership*.parquet"):
        mem_path = f
        break
    if not mem_path:
        return {"error": "no membership"}

    try:
        return build_temporal_snapshots(
            Path(edges_path), mem_path,
            abstracts_path=result.get("abstracts_path"),
        )
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/jobs/{job_id}/bridge")
async def get_bridge(job_id: str, cluster_a: int = 0, cluster_b: int = 1):
    """Find bridging papers between two clusters."""
    job = _jobs.get(job_id)
    if not job or job["status"] != "done":
        return {"error": "job not done"}
    result = job.get("result", {})
    from .network_data import find_bridge_papers

    edges_path = result.get("edges_path")
    landscape_dir = result.get("landscape_dir")
    if not edges_path or not landscape_dir:
        return {"error": "missing files"}

    mem_path = None
    for f in Path(landscape_dir).glob("membership*.parquet"):
        mem_path = f
        break
    if not mem_path:
        return {"error": "no membership"}

    try:
        papers = find_bridge_papers(
            Path(edges_path), mem_path,
            abstracts_path=result.get("abstracts_path"),
            cluster_a=cluster_a, cluster_b=cluster_b,
        )
        return {"papers": papers}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/jobs/{job_id}/term-network")
async def get_term_network(job_id: str, top_k: int = 10, max_terms: int = 150, min_cooc: int = 1):
    """Get term co-occurrence network data for D3 visualization."""
    job = _jobs.get(job_id)
    if not job or job["status"] != "done":
        return {"error": "job not done"}

    result = job.get("result", {})
    landscape_dir = result.get("landscape_dir")
    if not landscape_dir:
        return {"error": "no landscape output"}

    from .network_data import build_term_network_json

    kw_path = None
    for f in Path(landscape_dir).glob("keywords*.parquet"):
        kw_path = f
        break
    if not kw_path:
        return {"error": "no keywords file"}

    try:
        return build_term_network_json(
            kw_path,
            top_k_per_cluster=max(1, top_k),
            max_terms=max(1, max_terms),
            min_cooc=max(1, min_cooc),
        )
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/jobs/{job_id}/cluster/{cluster_id}")
async def get_cluster_papers(job_id: str, cluster_id: int):
    """Get papers belonging to a specific cluster."""
    job = _jobs.get(job_id)
    if not job or job["status"] != "done":
        return {"error": "job not done"}

    result = job.get("result", {})
    abs_path = result.get("abstracts_path")
    landscape_dir = result.get("landscape_dir")
    if not abs_path or not landscape_dir:
        return {"papers": []}

    import polars as pl

    try:
        abs_df = pl.read_parquet(abs_path)
        # Find membership file
        mem_path = None
        for f in Path(landscape_dir).glob("membership*.parquet"):
            mem_path = f
            break
        if not mem_path:
            return {"papers": []}

        mem_df = pl.read_parquet(mem_path)
        cluster_cols = [c for c in mem_df.columns if c.startswith("cluster_")]
        if not cluster_cols:
            return {"error": "no cluster columns found in membership data"}
        cluster_col = cluster_cols[0]

        # Join and filter
        joined = abs_df.join(mem_df.select("uid", cluster_col), on="uid", how="inner")
        cluster_papers = joined.filter(pl.col(cluster_col) == cluster_id)

        papers = []
        for row in cluster_papers.head(50).iter_rows(named=True):
            papers.append({
                "uid": row.get("uid", ""),
                "title": row.get("title", ""),
                "year": row.get("pubyear"),
                "cited_by_count": row.get("cited_by_count", 0),
            })
        return {"papers": papers, "total": cluster_papers.height}
    except Exception as e:
        return {"papers": [], "error": str(e)}


class MergeRequest(BaseModel):
    merge_map: dict  # {source_label: target_label}
    level: str = "nano"


@app.get("/api/jobs/{job_id}/label-merges/{level}")
async def get_label_merges(job_id: str, level: str = "nano", min_sim: float = 0.5):
    """Get suggested label merge candidates for a hierarchy level."""
    job = _jobs.get(job_id)
    if not job or job["status"] != "done":
        return {"error": "job not done"}

    result = job.get("result", {})
    landscape_dir = result.get("landscape_dir")
    if not landscape_dir:
        return {"error": "no landscape"}

    import polars as pl
    from sciscape.clustering.label_pipeline import extract_cluster_labels, suggest_merges

    abs_path = result.get("abstracts_path")
    if not abs_path:
        return {"error": "no abstracts"}

    # Find membership
    ld = Path(landscape_dir)
    mem_path = None
    for f in ld.glob("membership*.parquet"):
        mem_path = f
        break
    if not mem_path:
        return {"error": "no membership"}

    try:
        abs_df = pl.read_parquet(abs_path)
        mem_df = pl.read_parquet(mem_path)
        labels = extract_cluster_labels(abs_df, mem_df, level=level, top_k=5)
        candidates = suggest_merges(labels, min_similarity=min_sim)
        return {"candidates": candidates, "labels": labels.to_dicts()}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/jobs/{job_id}/label-merges/apply")
async def apply_label_merges(job_id: str, req: MergeRequest):
    """Apply confirmed label merges."""
    job = _jobs.get(job_id)
    if not job or job["status"] != "done":
        return {"error": "job not done"}

    from sciscape.clustering.label_pipeline import apply_merges
    import polars as pl

    # Store merge map in job
    if "label_merges" not in job:
        job["label_merges"] = {}
    job["label_merges"][req.level] = req.merge_map
    _jobs.persist(job_id)
    return {"status": "applied", "level": req.level, "n_merges": len(req.merge_map)}


@app.get("/api/jobs/{job_id}/temporal-tracking")
async def get_temporal_tracking(job_id: str, window: int = 5, step: int = 1):
    """Get temporal cluster evolution data."""
    job = _jobs.get(job_id)
    if not job or job["status"] != "done":
        return {"error": "job not done"}
    result = job.get("result", {})

    from sciscape.visualization.temporal_tracking import (
        compute_temporal_snapshots, detect_emerging_clusters, temporal_to_plotly,
    )
    import polars as pl

    edges_path = result.get("edges_path")
    abs_path = result.get("abstracts_path")
    if not edges_path or not abs_path:
        return {"error": "missing data"}

    try:
        edges = pl.read_parquet(edges_path)
        abs_df = pl.read_parquet(abs_path)
        year_map = dict(zip(abs_df["uid"].to_list(),
                            abs_df["pubyear"].to_list() if "pubyear" in abs_df.columns else [None]*abs_df.height))

        snapshots = compute_temporal_snapshots(edges, year_map, window_years=window, step_years=step)
        emerging = detect_emerging_clusters(snapshots)
        figures = temporal_to_plotly(snapshots)

        return {"snapshots": snapshots[:50], "emerging": emerging[:10], "figures": figures}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/jobs/{job_id}/evolution")
async def get_evolution_artifact(
    job_id: str,
    state_limit: int = 120,
    transition_limit: int = 180,
    event_limit: int = 180,
    state_membership_limit: int = 2000,
    map_node_limit: int = 240,
    map_edge_limit: int = 360,
    map_lineage_limit: int = 80,
):
    """Get artifact-backed cluster evolution rows for the Evolution lens."""
    job = _jobs.get(job_id)
    if not job or job["status"] != "done":
        return {"error": "job not done"}
    result = job.get("result", {})
    try:
        payload = _load_evolution_payload(
            result,
            state_limit=max(0, min(int(state_limit), 500)),
            transition_limit=max(0, min(int(transition_limit), 1000)),
            event_limit=max(0, min(int(event_limit), 1000)),
            state_membership_limit=max(0, min(int(state_membership_limit), 10000)),
            map_node_limit=max(0, min(int(map_node_limit), 1000)),
            map_edge_limit=max(0, min(int(map_edge_limit), 2000)),
            map_lineage_limit=max(1, min(int(map_lineage_limit), 300)),
        )
    except Exception as exc:
        return {"error": str(exc)}
    if not payload.get("available"):
        return {"error": payload.get("reason") or "no evolution artifact"}
    return payload


@app.get("/api/jobs/{job_id}/treemap")
async def get_treemap(job_id: str, mode: str = "treemap"):
    """Get Plotly treemap/sunburst data for cluster hierarchy."""
    job = _jobs.get(job_id)
    if not job or job["status"] != "done":
        return {"error": "job not done"}
    result = job.get("result", {})
    landscape_dir = result.get("landscape_dir")
    if not landscape_dir:
        return {"error": "no landscape"}
    import polars as pl
    from sciscape.visualization.hierarchy_treemap import build_treemap_data, treemap_to_plotly

    mem_path = None
    for f in Path(landscape_dir).glob("*hierarchy*.parquet"):
        mem_path = f
        break
    if not mem_path:
        for f in Path(landscape_dir).glob("membership*.parquet"):
            mem_path = f
            break
    if not mem_path:
        return {"error": "no membership"}
    try:
        hier_df = pl.read_parquet(mem_path)
        data = build_treemap_data(hier_df)
        return treemap_to_plotly(data, mode=mode)
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/jobs/{job_id}/abbreviations")
async def get_abbreviations(job_id: str, min_count: int = 3):
    """Extract abbreviation dictionary from paper abstracts."""
    job = _jobs.get(job_id)
    if not job or job["status"] != "done":
        return {"error": "job not done"}
    abs_path = job.get("result", {}).get("abstracts_path")
    if not abs_path:
        return {"error": "no abstracts"}
    import polars as pl
    from sciscape.clustering.abbreviation_dict import extract_abbreviations
    try:
        abs_df = pl.read_parquet(abs_path)
        abbr = extract_abbreviations(abs_df, min_count=min_count)
        return {"abbreviations": abbr, "count": len(abbr)}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/jobs/{job_id}/consensus")
async def get_consensus(job_id: str):
    """Get consensus distribution stats for multi-layer edges."""
    job = _jobs.get(job_id)
    if not job or job["status"] != "done":
        return {"error": "job not done"}
    result = job.get("result", {})
    output_dir = result.get("output_dir")
    if not output_dir:
        return {"error": "no output"}

    from .network_data import build_network_json
    from sciscape.visualization.consensus import compute_consensus_stats
    import polars as pl

    # Load per-layer edge files
    out = Path(output_dir)
    layers = {}
    for layer in ("dc", "bc", "cc"):
        p = out / f"edges_{layer}.parquet"
        if p.exists():
            layers[layer] = pl.read_parquet(p)

    nonempty = {k: v for k, v in layers.items() if v.height > 0}
    if len(nonempty) < 2:
        return {"error": "need at least 2 non-empty layers for consensus analysis"}

    try:
        # adaptive top-k for consensus stats
        from sciscape.linkage.filters import compute_adaptive_k
        all_uids = pl.concat([
            pl.concat([df["uid1"], df["uid2"]])
            for df in nonempty.values()
        ]).unique()
        k = compute_adaptive_k(all_uids.len())
        stats = compute_consensus_stats(layers, top_k=k)
        return stats
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/jobs")
async def list_jobs():
    """List all jobs."""
    return [
        {
            "job_id": jid,
            "status": j["status"],
            "query": j["request"].get("query", ""),
            "cancel_requested": bool(j.get("cancel_requested_at_utc")),
            "cancel_requested_at_utc": j.get("cancel_requested_at_utc"),
        }
        for jid, j in _jobs.items()
    ]


# ── What-if gamma re-clustering ───────────────────────────────

@app.post("/api/jobs/{job_id}/what-if")
async def what_if_gamma(job_id: str, gamma: float, min_size: int = 10):
    """Re-cluster with a different gamma without full pipeline re-run."""
    job = _jobs.get(job_id)
    if not job or job["status"] != "done":
        return {"error": "job not ready"}
    result = job.get("result", {})
    edges_path = result.get("edges_path")
    if not edges_path or not Path(edges_path).exists():
        return {"error": "edges not found"}

    import polars as pl
    from sciscape.clustering.leiden_rust import (
        RUST_AVAILABLE,
        build_leiden_graph,
        postprocess_small_clusters_rust,
        run_leiden_rust,
    )
    from sciscape.clustering.integer_remap import integer_remap_memory
    import numpy as np

    if not RUST_AVAILABLE:
        return {"error": "Rust backend required"}

    edges = pl.read_parquet(edges_path)
    src, dst, w, n_nodes, uids = integer_remap_memory(edges)
    try:
        graph = build_leiden_graph(
            edges_src=src, edges_dst=dst, edges_weight=w, n_nodes=n_nodes,
        )
    except AttributeError:
        graph = None
    if graph is not None:
        r = graph.run_leiden(
            resolution=gamma, seed=42, n_iterations=10,
        )
        p = graph.postprocess_small_clusters(
            resolution=gamma, min_size=min_size, membership=r.membership,
            seed=42, gamma_decay=0.5, max_rounds=3,
            use_greedy=True, use_component_merge=True,
        )
    else:
        r = run_leiden_rust(edges_src=src, edges_dst=dst, edges_weight=w,
                            resolution=gamma, n_nodes=n_nodes, seed=42, n_iterations=10)
        p = postprocess_small_clusters_rust(
            resolution=gamma, min_size=min_size, membership=r.membership,
            edges_src=src, edges_dst=dst, edges_weight=w, n_nodes=n_nodes, seed=42,
            gamma_decay=0.5, max_rounds=3, use_greedy=True, use_component_merge=True,
        )
    mem = np.asarray(p.membership, dtype=np.int32)
    size_arr = np.bincount(mem)
    size_arr_nz = size_arr[size_arr > 0]
    n_cl = len(size_arr_nz)
    mx = int(size_arr_nz.max()) if n_cl > 0 else 0
    return {
        "gamma": gamma,
        "n_clusters": n_cl,
        "max_pct": round(100 * mx / max(n_nodes, 1), 1),
        "top10": sorted(size_arr_nz.tolist(), reverse=True)[:10],
        "n_nodes": n_nodes,
    }


@app.get("/api/jobs/{job_id}/quality")
async def get_quality_report(job_id: str):
    """Get quality metrics + stability for a completed job."""
    job = _jobs.get(job_id)
    if not job or job["status"] != "done":
        return {"error": "job not ready"}
    result = job.get("result", {})
    edges_path = result.get("edges_path")
    landscape_dir = result.get("landscape_dir")
    if not edges_path or not Path(edges_path).exists():
        return {"error": "edges not found"}

    import polars as pl
    import numpy as np
    from sciscape.evaluation.stability import compute_quality_report

    edges = pl.read_parquet(edges_path)
    mem_path = None
    if landscape_dir:
        ld = Path(landscape_dir)
        for f in ld.glob("**/membership*.parquet"):
            mem_path = f
            break
    if not mem_path or not mem_path.exists():
        return {"error": "membership not found"}

    mem_df = pl.read_parquet(mem_path)
    cluster_cols = [c for c in mem_df.columns if c.startswith("cluster_")]
    if not cluster_cols:
        return {"error": "no cluster columns"}
    membership = mem_df[cluster_cols[0]].to_numpy()
    qr = compute_quality_report(edges, membership, gamma=1.0)
    return {
        "n_nodes": qr.n_nodes, "n_edges": qr.n_edges,
        "n_clusters": qr.n_clusters, "max_pct": qr.max_cluster_pct,
        "singleton_pct": qr.singleton_pct, "top5": qr.top5_sizes,
        "consensus_edges": qr.consensus_edge_pct,
    }


@app.get("/api/jobs/{job_id}/export/{fmt}")
async def export_network(
    job_id: str,
    fmt: str,
    atlas_level: str | None = None,
    atlas_node: str | None = None,
    atlas_query: str | None = None,
    atlas_lens: str | None = None,
    atlas_view: str | None = None,
    atlas_focus: str | None = None,
    atlas_review: str | None = None,
    atlas_layers: str | None = None,
    atlas_edge_min: float | None = None,
    atlas_label_limit: int | None = None,
    atlas_neighbor: str | None = None,
    atlas_subset_mode: str | None = None,
    atlas_subset_count: int | None = None,
    atlas_subset_uids: str | None = None,
    atlas_subset_truncated: bool | None = None,
    atlas_pinned: str | None = None,
):
    """Export network as GEXF or GraphML."""
    if fmt not in ("gexf", "graphml"):
        return {"error": f"unsupported format: {fmt}"}
    job = _jobs.get(job_id)
    if not job or job["status"] != "done":
        return {"error": "job not ready"}
    result = job.get("result", {})
    edges_path = result.get("edges_path")
    landscape_dir = result.get("landscape_dir")
    if not edges_path or not Path(edges_path).exists():
        return {"error": "edges not found"}

    import polars as pl
    from sciscape.export import export_gexf, export_graphml

    edges = pl.read_parquet(edges_path)
    mem_path = None
    if landscape_dir:
        for f in Path(landscape_dir).glob("**/membership*.parquet"):
            mem_path = f
            break
    membership = pl.read_parquet(mem_path) if mem_path and mem_path.exists() else {}

    out_dir = Path(result.get("output_dir", "."))
    out_path = out_dir / f"network.{fmt}"

    abs_path = result.get("abstracts_path")
    abstracts = pl.read_parquet(abs_path) if abs_path and Path(abs_path).exists() else None
    source_paths = {
        "edges": edges_path,
        "membership": mem_path,
        "abstracts": abs_path,
    }
    export_selection = _web_network_export_selection(
        fmt=fmt,
        membership_path=mem_path,
        atlas_level=atlas_level,
        atlas_node=atlas_node,
        atlas_query=atlas_query,
        atlas_lens=atlas_lens,
        atlas_view=atlas_view,
        atlas_focus=atlas_focus,
        atlas_review=atlas_review,
        atlas_layers=atlas_layers,
        atlas_edge_min=atlas_edge_min,
        atlas_label_limit=atlas_label_limit,
        atlas_neighbor=atlas_neighbor,
        atlas_subset_mode=atlas_subset_mode,
        atlas_subset_count=atlas_subset_count,
        atlas_subset_uids=atlas_subset_uids,
        atlas_subset_truncated=atlas_subset_truncated,
        atlas_pinned=atlas_pinned,
    )
    edges, membership, abstracts, subset_transform = _filter_network_export_to_selection(
        edges=edges,
        membership=membership,
        abstracts=abstracts,
        selection=export_selection,
    )
    export_transforms = [subset_transform] if subset_transform is not None else None

    if fmt == "graphml":
        export_graphml(
            edges,
            membership,
            out_path,
            abstracts=abstracts,
            write_manifest=True,
            result_root=out_dir,
            source_paths=source_paths,
            selection=export_selection,
            transforms=export_transforms,
        )
    else:
        export_gexf(
            edges,
            membership,
            out_path,
            abstracts=abstracts,
            write_manifest=True,
            result_root=out_dir,
            source_paths=source_paths,
            selection=export_selection,
            transforms=export_transforms,
        )

    try:
        _refresh_job_result_manifest(job_id, result, out_dir)
    except Exception:
        # The file export itself should remain downloadable even if manifest
        # refresh fails on a partially populated result root.
        pass

    return FileResponse(str(out_path), filename=f"sciscape_network.{fmt}",
                        media_type="application/xml")


# ── Background job runner ────────────────────────────────────

def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _job_status_for_manifest(status: str) -> str:
    return {
        "pending": "queued",
        "running": "running",
        "done": "complete",
        "error": "failed",
        "cancelled": "cancelled",
    }.get(status, status)


def _job_cancel_requested(job: dict[str, Any]) -> bool:
    return bool(job.get("cancel_requested_at_utc"))


def _append_progress_once(job: dict[str, Any], message: str) -> None:
    progress = job.setdefault("progress", [])
    if not any(str(msg) == message for msg in progress):
        progress.append(message)


def _query_filters(req: QueryRequest) -> dict[str, Any]:
    filters: dict[str, Any] = {}
    if req.years:
        filters["publication_year"] = req.years
    return filters


def _default_openalex_api_attempt_budget(req: QueryRequest) -> int:
    page_budget = max(1, math.ceil(max(0, int(req.max_works)) / 200))
    retry_multiplier = 4
    return max(8, page_budget * retry_multiplier + 4)


def _rel_output_path(path: Path, output_dir: Path) -> str:
    try:
        return path.relative_to(output_dir).as_posix()
    except ValueError:
        return str(path)


def _live_job_partial_outputs(output_dir: Path, result: Any | None = None) -> list[dict[str, Any]]:
    candidates: list[tuple[str, Path | None]] = [
        ("abstracts", output_dir / "abstracts.parquet"),
        ("edges", output_dir / "edges.parquet"),
        ("job_status", output_dir / "job_status.json"),
        ("membership", output_dir / "landscape" / "membership.parquet"),
        ("keywords", output_dir / "landscape" / "keywords.parquet"),
        ("report_data", output_dir / "landscape" / "report" / "data.json"),
    ]
    if result is not None:
        candidates.extend(
            [
                ("abstracts", getattr(result, "abstracts_path", None)),
                ("edges", getattr(result, "edges_path", None)),
                ("landscape", getattr(result, "landscape_dir", None)),
            ]
        )
    outputs: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for kind, candidate in candidates:
        if candidate is None:
            continue
        path = Path(candidate)
        if not path.exists():
            continue
        rel_path = _rel_output_path(path, output_dir)
        key = (kind, rel_path)
        if key in seen:
            continue
        seen.add(key)
        outputs.append(
            {
                "kind": kind,
                "path": rel_path,
                "status": "present",
                "size_bytes": int(path.stat().st_size) if path.is_file() else None,
            }
        )
    return outputs


def _write_live_job_status_artifacts(
    *,
    job_id: str,
    job: dict[str, Any],
    output_dir: Path,
    req: QueryRequest,
    filters: dict[str, Any],
    status: str,
    result: Any | None = None,
    error: str | None = None,
    validation_path: Path | None = None,
    write_manifest: bool = True,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    now = _utc_now()
    job["updated_at_utc"] = now
    progress_messages = list(job.get("progress") or [])
    if status in {"done", "error", "cancelled"}:
        job.setdefault("finished_at_utc", now)
    partial_outputs = _live_job_partial_outputs(output_dir, result)
    api_telemetry = getattr(result, "api_telemetry", None) if result is not None else None
    if api_telemetry is None:
        api_telemetry = job.get("openalex_api_telemetry")
    run_state = {
        "status": _job_status_for_manifest(status),
        "started_at_utc": job.get("started_at_utc"),
        "finished_at_utc": job.get("finished_at_utc"),
        "heartbeat_at_utc": now,
        "progress": {
            "current": len(progress_messages),
            "total": len(progress_messages) if status in {"done", "error", "cancelled"} else None,
            "unit": "messages",
        },
        "shards": {"total": 0, "complete": 0, "failed": 0, "running": 0},
        "checkpoints": [{"path": "job_status.json", "kind": "job_status", "status": "present"}],
        "partial_outputs": partial_outputs,
        "failure": {"reason": error} if error and status != "cancelled" else None,
        "resume": {"supported": False, "command": None},
    }
    if api_telemetry:
        run_state["api_telemetry"] = api_telemetry
    if job.get("cancel_requested_at_utc"):
        run_state["cancel_requested_at_utc"] = job.get("cancel_requested_at_utc")
    if status == "cancelled":
        run_state["cancellation"] = {
            "reason": error or "Cancellation requested",
            "requested_at_utc": job.get("cancel_requested_at_utc"),
        }
    record_count = getattr(result, "n_works", None) if result is not None else None
    source_overrides = {
        "source_type": "openalex_query",
        "query": req.query,
        "filters": filters,
        "record_count": record_count,
        "api_telemetry": api_telemetry,
    }
    payload = {
        "schema_version": "sciscape_live_job_status_v1",
        "job_id": job_id,
        "status": status,
        "updated_at_utc": now,
        "started_at_utc": job.get("started_at_utc"),
        "finished_at_utc": job.get("finished_at_utc"),
        "cancel_requested_at_utc": job.get("cancel_requested_at_utc"),
        "request": req.model_dump(),
        "output_dir": str(output_dir),
        "progress": progress_messages,
        "progress_messages_count": len(progress_messages),
        "partial_outputs": partial_outputs,
        "api_telemetry": api_telemetry,
        "run_state": run_state,
        "error": error,
    }
    _write_json_atomic(output_dir / "job_status.json", payload)

    manifest_payload = None
    if write_manifest:
        try:
            manifest = write_result_manifest(
                validation_path or output_dir,
                mode="live_query",
                source_overrides=source_overrides,
                run_state_overrides=run_state,
            )
            manifest_payload = manifest.to_dict()
        except Exception as exc:  # pragma: no cover - defensive status metadata sidecar
            log.warning("Result manifest status update skipped for job %s: %s", job_id, exc)
    return payload, manifest_payload


def _mark_job_cancelled(
    *,
    job_id: str,
    job: dict[str, Any],
    output_dir: Path,
    req: QueryRequest,
    filters: dict[str, Any],
    reason: str = "Cancellation requested",
) -> None:
    job["status"] = "cancelled"
    job["finished_at_utc"] = _utc_now()
    _append_progress_once(job, "Job cancelled.")
    _, manifest = _write_live_job_status_artifacts(
        job_id=job_id,
        job=job,
        output_dir=output_dir,
        req=req,
        filters=filters,
        status="cancelled",
        error=reason,
    )
    job_result = {
        "cancelled": True,
        "output_dir": str(output_dir),
        "job_status_path": str(output_dir / "job_status.json"),
        "result_manifest": manifest,
        "run_state": (manifest or {}).get("run_state"),
        "result_state": "partial",
    }
    _attach_run_state_summary(job_result, manifest)
    job["result"] = job_result
    _jobs.persist(job_id)


def _write_resume_job_status_artifact(
    *,
    job_id: str,
    source_job_id: str,
    job: dict[str, Any],
    output_dir: Path,
    command: str,
    argv: list[str],
    status: str,
    error: str | None = None,
) -> dict[str, Any]:
    now = _utc_now()
    job["updated_at_utc"] = now
    if status in {"done", "error", "cancelled"}:
        job.setdefault("finished_at_utc", now)
    run_state = {
        "status": _job_status_for_manifest(status),
        "started_at_utc": job.get("started_at_utc"),
        "finished_at_utc": job.get("finished_at_utc"),
        "heartbeat_at_utc": now,
        "progress": {
            "current": len(job.get("progress") or []),
            "total": len(job.get("progress") or []) if status in {"done", "error", "cancelled"} else None,
            "unit": "messages",
        },
        "shards": {"total": 0, "complete": 0, "failed": 0, "running": 0},
        "checkpoints": [{"path": "job_status.json", "kind": "job_status", "status": "present"}],
        "partial_outputs": _live_job_partial_outputs(output_dir),
        "failure": {"reason": error} if error else None,
        "resume": {"supported": False, "command": None},
    }
    request_payload = {
        "source_type": "sciscape_cli_resume",
        "resume_of": source_job_id,
        "command": command,
        "argv": argv,
    }
    stored_request = job.get("request") if isinstance(job.get("request"), dict) else {}
    for key in ("resume_scope", "shard_ids", "schedule_options"):
        if key in stored_request:
            request_payload[key] = stored_request[key]
    payload = {
        "schema_version": "sciscape_resume_job_status_v1",
        "job_id": job_id,
        "status": status,
        "updated_at_utc": now,
        "started_at_utc": job.get("started_at_utc"),
        "finished_at_utc": job.get("finished_at_utc"),
        "source_job_id": source_job_id,
        "request": request_payload,
        "output_dir": str(output_dir),
        "progress": list(job.get("progress") or []),
        "progress_messages_count": len(job.get("progress") or []),
        "run_state": run_state,
        "error": error,
    }
    _write_json_atomic(output_dir / "job_status.json", payload)
    return payload


def _run_sciscape_resume_command(argv: list[str], progress: Any) -> None:
    """Run a validated SciScape resume argv without invoking a shell."""

    from sciscape.cli import main as sciscape_cli_main

    progress("Executing: sciscape " + shlex.join(argv))
    sciscape_cli_main(list(argv))


def _source_result_after_resume(source_job_id: str) -> dict[str, Any]:
    source_job = _jobs.get(source_job_id)
    if not source_job or not isinstance(source_job.get("result"), dict):
        return {}
    source_result = dict(source_job["result"])
    root = _result_root_for_result(source_result)
    if root is not None:
        try:
            _refresh_job_result_manifest(source_job_id, source_result, root, mode="local_result")
            refreshed_job = _jobs.get(source_job_id)
            if refreshed_job and isinstance(refreshed_job.get("result"), dict):
                source_result = dict(refreshed_job["result"])
        except Exception as exc:
            log.warning("Resume job could not refresh source result %s: %s", source_job_id, exc)
    try:
        _attach_report_atlas(source_result)
        _attach_narrative_summary(source_result)
        _attach_evolution_summary(source_result)
    except Exception as exc:
        log.warning("Resume job could not refresh source result summaries %s: %s", source_job_id, exc)
    source_job = _jobs.get(source_job_id)
    if source_job is not None:
        source_job["result"] = source_result
        _jobs.persist(source_job_id)
    return source_result


def _run_resume_job(job_id: str, source_job_id: str, command: str, argv: list[str]) -> None:
    """Execute a validated CLI resume command as a background web job."""

    job = _jobs[job_id]
    if not job.get("started_at_utc"):
        job["started_at_utc"] = _utc_now()
    output_dir = Path("workspace/web_output") / job_id
    job["status"] = "running"
    _append_progress_once(job, "Resume job started.")
    _write_resume_job_status_artifact(
        job_id=job_id,
        source_job_id=source_job_id,
        job=job,
        output_dir=output_dir,
        command=command,
        argv=argv,
        status="running",
    )
    _jobs.persist(job_id)

    def progress_cb(message: str) -> None:
        job["progress"].append(str(message))
        _write_resume_job_status_artifact(
            job_id=job_id,
            source_job_id=source_job_id,
            job=job,
            output_dir=output_dir,
            command=command,
            argv=argv,
            status=str(job.get("status") or "running"),
        )
        _jobs.persist(job_id)

    try:
        _run_sciscape_resume_command(argv, progress_cb)
        source_result = _source_result_after_resume(source_job_id)
        job["status"] = "done"
        job["finished_at_utc"] = _utc_now()
        _append_progress_once(job, "Resume command complete.")
        result = dict(_json_safe(source_result))
        if "output_dir" not in result:
            result["output_dir"] = str(output_dir)
        result.update(
            {
                "resume_of": source_job_id,
                "resume_command": command,
                "resume_argv": argv,
                "resume_job_status_path": str(output_dir / "job_status.json"),
            }
        )
        if isinstance(result.get("run_state"), dict):
            _attach_run_state_summary(result)
        job["result"] = result
        _write_resume_job_status_artifact(
            job_id=job_id,
            source_job_id=source_job_id,
            job=job,
            output_dir=output_dir,
            command=command,
            argv=argv,
            status="done",
        )
        _jobs.persist(job_id)
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        if code in {0, None}:  # pragma: no cover - CLI main normally returns
            job["status"] = "done"
            job["finished_at_utc"] = _utc_now()
            _append_progress_once(job, "Resume command complete.")
            job["result"] = {
                "resume_of": source_job_id,
                "resume_command": command,
                "resume_argv": argv,
                "output_dir": str(output_dir),
                "resume_job_status_path": str(output_dir / "job_status.json"),
            }
            _write_resume_job_status_artifact(
                job_id=job_id,
                source_job_id=source_job_id,
                job=job,
                output_dir=output_dir,
                command=command,
                argv=argv,
                status="done",
            )
            _jobs.persist(job_id)
            return
        error = f"resume command exited with status {exc.code}"
        job["status"] = "error"
        job["finished_at_utc"] = _utc_now()
        job["progress"].append(f"ERROR: {error}")
        job["result"] = {
            "error": error,
            "resume_of": source_job_id,
            "resume_command": command,
            "resume_argv": argv,
            "output_dir": str(output_dir),
            "resume_job_status_path": str(output_dir / "job_status.json"),
        }
        _write_resume_job_status_artifact(
            job_id=job_id,
            source_job_id=source_job_id,
            job=job,
            output_dir=output_dir,
            command=command,
            argv=argv,
            status="error",
            error=error,
        )
        _jobs.persist(job_id)
        log.error("Resume job %s failed: %s", job_id, error)
    except Exception as exc:
        job["status"] = "error"
        job["finished_at_utc"] = _utc_now()
        job["progress"].append(f"ERROR: {exc}")
        job["result"] = {
            "error": str(exc),
            "resume_of": source_job_id,
            "resume_command": command,
            "resume_argv": argv,
            "output_dir": str(output_dir),
            "resume_job_status_path": str(output_dir / "job_status.json"),
        }
        _write_resume_job_status_artifact(
            job_id=job_id,
            source_job_id=source_job_id,
            job=job,
            output_dir=output_dir,
            command=command,
            argv=argv,
            status="error",
            error=str(exc),
        )
        _jobs.persist(job_id)
        log.exception("Resume job %s failed", job_id)


def _run_job(job_id: str, req: QueryRequest) -> None:
    """Execute the OpenAlex pipeline in background."""
    from sciscape.openalex import run_openalex_pipeline, OpenAlexPipelineConfig

    job = _jobs[job_id]
    if not job.get("started_at_utc"):
        job["started_at_utc"] = _utc_now()
    output_dir = Path("workspace/web_output") / job_id
    filters = _query_filters(req)
    if _job_cancel_requested(job):
        _mark_job_cancelled(
            job_id=job_id,
            job=job,
            output_dir=output_dir,
            req=req,
            filters=filters,
        )
        return
    job["status"] = "running"
    _write_live_job_status_artifacts(
        job_id=job_id,
        job=job,
        output_dir=output_dir,
        req=req,
        filters=filters,
        status="running",
    )
    _jobs.persist(job_id)

    def progress_cb(msg: str) -> None:
        if _job_cancel_requested(job):
            raise JobCancelled("Cancellation requested")
        job["progress"].append(msg)
        _write_live_job_status_artifacts(
            job_id=job_id,
            job=job,
            output_dir=output_dir,
            req=req,
            filters=filters,
            status="running",
            write_manifest=False,
        )
        _jobs.persist(job_id)
        if _job_cancel_requested(job):
            raise JobCancelled("Cancellation requested")

    def cancel_checkpoint() -> None:
        if _job_cancel_requested(job):
            raise JobCancelled("Cancellation requested")

    def api_telemetry_cb(snapshot: dict[str, Any]) -> None:
        job["openalex_api_telemetry"] = snapshot
        _write_live_job_status_artifacts(
            job_id=job_id,
            job=job,
            output_dir=output_dir,
            req=req,
            filters=filters,
            status="running",
            write_manifest=False,
        )
        _jobs.persist(job_id)
        if _job_cancel_requested(job):
            raise JobCancelled("Cancellation requested")

    try:
        config = OpenAlexPipelineConfig(
            query=req.query,
            filters=filters,
            max_works=req.max_works,
            email=req.email,
            edge_types=req.edge_types.split(","),
            output_dir=output_dir,
            run_landscape=req.run_landscape,
            combine_strategy=req.combine_strategy,
            combine_top_k=req.combine_top_k,
            auto_gamma=req.auto_gamma,
            auto_gamma_target=req.auto_gamma_target,
            progress=progress_cb,
            checkpoint=cancel_checkpoint,
            api_telemetry=api_telemetry_cb,
            api_attempt_budget=(
                req.api_attempt_budget
                if req.api_attempt_budget is not None
                else _default_openalex_api_attempt_budget(req)
            ),
            retry_wait_budget_seconds=req.retry_wait_budget_seconds,
            interruptible_requests=True,
            request_poll_interval=0.25,
        )
        result = run_openalex_pipeline(config)
        if _job_cancel_requested(job):
            raise JobCancelled("Cancellation requested")

        job["status"] = "done"
        job_result = {
            "n_works": result.n_works,
            "n_edges": result.n_edges,
            "output_dir": str(output_dir),
            "abstracts_path": str(result.abstracts_path) if result.abstracts_path else None,
            "edges_path": str(result.edges_path) if result.edges_path else None,
            "landscape_dir": str(result.landscape_dir) if result.landscape_dir else None,
            "job_status_path": str(output_dir / "job_status.json"),
            "api_telemetry": getattr(result, "api_telemetry", None),
        }
        try:
            validation_path = result.landscape_dir or output_dir
            contract = validate_result_root(validation_path, mode="live_query").to_dict()
            _, manifest = _write_live_job_status_artifacts(
                job_id=job_id,
                job=job,
                output_dir=output_dir,
                req=req,
                filters=filters,
                status="done",
                result=result,
                validation_path=validation_path,
            )
            if manifest is None:
                manifest = load_result_manifest(
                    validation_path,
                    mode="live_query",
                    source_overrides={
                        "source_type": "openalex_query",
                        "query": req.query,
                        "filters": filters,
                        "record_count": result.n_works,
                    },
                )
            job_result["artifact_contract"] = contract
            job_result["result_manifest"] = manifest
            job_result["run_state"] = _run_state_for_result(job_result, manifest)
            job_result["features"] = contract["features"]
            job_result["feature_states"] = {
                name: feature.get("state", "hidden")
                for name, feature in manifest.get("features", {}).items()
            }
            job_result["result_state"] = contract["result_state"]
            _attach_run_state_summary(job_result, manifest)
        except Exception as exc:
            job_result["artifact_contract_error"] = str(exc)
        _attach_report_atlas(job_result)
        _attach_evolution_summary(job_result)
        job["result"] = job_result
        _jobs.persist(job_id)
    except JobCancelled as e:
        _mark_job_cancelled(
            job_id=job_id,
            job=job,
            output_dir=output_dir,
            req=req,
            filters=filters,
            reason=str(e) or "Cancellation requested",
        )
    except Exception as e:
        job["status"] = "error"
        job["progress"].append(f"ERROR: {e}")
        job["result"] = {"error": str(e)}
        _write_live_job_status_artifacts(
            job_id=job_id,
            job=job,
            output_dir=output_dir,
            req=req,
            filters=filters,
            status="error",
            error=str(e),
        )
        _jobs.persist(job_id)
        log.exception("Job %s failed", job_id)
