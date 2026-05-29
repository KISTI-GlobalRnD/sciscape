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
import uuid
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.responses import StreamingResponse

log = logging.getLogger(__name__)

app = FastAPI(title="SciScape", version="0.2.0")

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


class JobStatus(BaseModel):
    job_id: str
    status: str  # "pending", "running", "done", "error"
    progress: list[str]
    result: dict | None = None


class LocalDataOpenRequest(BaseModel):
    path: str


_LOCAL_DATA_ROOTS = [
    Path("workspace/web_output"),
    Path("workspace/examples_output"),
    Path("workspace/output"),
    Path("viewer"),
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


def _resolve_job_output_file(job_id: str, filename: str) -> Path:
    """Resolve an output artifact path inside a completed job directory."""
    job = _jobs.get(job_id)
    if not job or job["status"] != "done":
        raise HTTPException(status_code=400, detail="job not done")
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
    return roots


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
    raise HTTPException(status_code=400, detail="local data path is outside allowed roots")


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
        if _looks_like_landscape_dir(candidate):
            return candidate.parent
        if _find_landscape_dirs(candidate):
            return candidate
        if (candidate / "edges.parquet").exists() or (candidate / "abstracts.parquet").exists():
            return candidate
    return None


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
    return result


def _artifact_role(path: Path) -> str:
    name = path.name
    if name == "data.json":
        return "viewer_data"
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


def _local_artifact_record(path: Path) -> dict[str, Any]:
    output_dir = _infer_output_dir(path)
    relative_path = _display_local_path(path)
    artifact_id = hashlib.sha1(relative_path.encode("utf-8")).hexdigest()[:12]
    landscape_dir = _infer_landscape_dir(output_dir, selected_path=path) if output_dir is not None else None
    data_json = landscape_dir / "report" / "data.json" if landscape_dir else None
    keywords = landscape_dir / "keywords.parquet" if landscape_dir else None
    membership = landscape_dir / "membership.parquet" if landscape_dir else None
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
    }


def _discover_local_artifacts(limit: int = 80) -> list[dict[str, Any]]:
    """Find local SciScape outputs that can be opened from the web UI."""
    candidates: dict[str, Path] = {}
    globs = [
        "**/landscape/report/data.json",
        "**/landscape/report/index.html",
        "**/landscape/report/report.html",
        "**/landscape/keywords.parquet",
        "**/landscape/membership.parquet",
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

    records = [_local_artifact_record(path) for path in candidates.values()]
    records.sort(key=lambda item: item["modified"], reverse=True)
    return records[:limit]


@app.get("/api/local-data")
async def list_local_data(limit: int = 80):
    """List existing local SciScape output artifacts the web app can open."""
    return {
        "roots": [_display_local_path(root) for root in _local_data_roots()],
        "artifacts": _discover_local_artifacts(limit=limit),
        "expected_files": [
            "workspace/web_output/<job>/landscape/report/data.json",
            "workspace/examples_output/<demo>/landscape/keywords.parquet",
            "workspace/examples_output/<demo>/landscape/membership.parquet",
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
        {"job_id": jid, "status": j["status"], "query": j["request"].get("query", "")}
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
async def export_network(job_id: str, fmt: str):
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

    if fmt == "graphml":
        export_graphml(edges, membership, out_path, abstracts=abstracts)
    else:
        export_gexf(edges, membership, out_path, abstracts=abstracts)

    return FileResponse(str(out_path), filename=f"sciscape_network.{fmt}",
                        media_type="application/xml")


# ── Background job runner ────────────────────────────────────

def _run_job(job_id: str, req: QueryRequest) -> None:
    """Execute the OpenAlex pipeline in background."""
    from sciscape.openalex import run_openalex_pipeline, OpenAlexPipelineConfig

    job = _jobs[job_id]
    job["status"] = "running"

    def progress_cb(msg: str) -> None:
        job["progress"].append(msg)

    try:
        output_dir = Path("workspace/web_output") / job_id
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
            combine_strategy=req.combine_strategy,
            combine_top_k=req.combine_top_k,
            auto_gamma=req.auto_gamma,
            auto_gamma_target=req.auto_gamma_target,
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
        _jobs.persist(job_id)
    except Exception as e:
        job["status"] = "error"
        job["progress"].append(f"ERROR: {e}")
        job["result"] = {"error": str(e)}
        _jobs.persist(job_id)
        log.exception("Job %s failed", job_id)
