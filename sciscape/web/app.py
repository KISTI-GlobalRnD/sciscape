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


@app.get("/api/jobs/{job_id}/network")
async def get_network(job_id: str):
    """Get cluster network data for D3 visualization."""
    job = _jobs.get(job_id)
    if not job or job["status"] != "done":
        return {"error": "job not done"}
    result = job.get("result", {})
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
        cluster_col = [c for c in mem_df.columns if c.startswith("cluster_")][0]

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

    except Exception as e:
        job["llm_labels"] = {"status": "error", "error": str(e)}


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
async def get_term_network(job_id: str, top_k: int = 10, max_terms: int = 150):
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
            top_k_per_cluster=top_k,
            max_terms=max_terms,
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
        cluster_col = [c for c in mem_df.columns if c.startswith("cluster_")][0]

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
    return {"status": "applied", "level": req.level, "n_merges": len(req.merge_map)}


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

    if len(layers) < 2:
        return {"error": "need at least 2 layers for consensus analysis"}

    try:
        stats = compute_consensus_stats(layers, top_k=30)
        return stats
    except Exception as e:
        return {"error": str(e)}


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
