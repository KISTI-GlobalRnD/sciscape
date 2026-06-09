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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from sciscape.artifacts import (
    build_atlas_payload_from_report_data,
    build_atlas_render_payload,
    infer_result_artifacts,
    load_result_manifest,
    validate_evolution_artifact,
    validate_result_root,
    validate_workspace,
    write_result_manifest,
)

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


def _refresh_job_result_manifest(job_id: str, result: dict[str, Any], output_dir: Path, *, mode: str = "live_query") -> None:
    refreshed_manifest = load_result_manifest(output_dir, mode=mode)
    result["result_manifest"] = refreshed_manifest
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
        if _looks_like_landscape_dir(candidate):
            return candidate.parent
        if _find_landscape_dirs(candidate):
            return candidate
        if (candidate / "edges.parquet").exists() or (candidate / "abstracts.parquet").exists():
            return candidate
    return None


def _ensure_local_result_table_exports(output_dir: Path) -> None:
    """Create small manifest-backed table exports for local result browsing."""

    try:
        from sciscape.artifacts import write_cooccurrence_artifacts
        from sciscape.export import export_cooccurrence_table, export_vosviewer_term_cooccurrence

        written = write_cooccurrence_artifacts(output_dir)
        if written is not None:
            export_cooccurrence_table(output_dir)
            export_vosviewer_term_cooccurrence(output_dir)
    except Exception:
        # Local result opening should not fail just because an optional export
        # sidecar cannot be generated from a partial result root.
        return


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
    result["features"] = contract["features"]
    result["feature_states"] = {
        name: feature.get("state", "hidden")
        for name, feature in manifest.get("features", {}).items()
    }
    result["result_state"] = contract["result_state"]
    _attach_report_atlas(result)
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
        atlas = result.get("atlas")
    if not isinstance(atlas, dict) or not atlas.get("nodes"):
        return None
    return build_atlas_render_payload(atlas)


def _json_safe(value: Any) -> Any:
    """Convert pandas/numpy-ish values to strict JSON-friendly values."""
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
    return payload


def _attach_evolution_summary(result: dict[str, Any]) -> None:
    payload = _load_evolution_payload(
        result,
        state_limit=0,
        transition_limit=0,
        event_limit=0,
        lineage_limit=0,
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


def _discover_legacy_local_artifacts(limit: int = 80) -> list[dict[str, Any]]:
    """Find local SciScape outputs by scanning legacy output roots."""
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
    }.get(status, status)


def _query_filters(req: QueryRequest) -> dict[str, Any]:
    filters: dict[str, Any] = {}
    if req.years:
        filters["publication_year"] = req.years
    return filters


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
    if status in {"done", "error"}:
        job.setdefault("finished_at_utc", now)
    partial_outputs = _live_job_partial_outputs(output_dir, result)
    run_state = {
        "status": _job_status_for_manifest(status),
        "started_at_utc": job.get("started_at_utc"),
        "finished_at_utc": job.get("finished_at_utc"),
        "heartbeat_at_utc": now,
        "progress": {
            "current": len(progress_messages),
            "total": len(progress_messages) if status in {"done", "error"} else None,
            "unit": "messages",
        },
        "shards": {"total": 0, "complete": 0, "failed": 0, "running": 0},
        "checkpoints": [{"path": "job_status.json", "kind": "job_status", "status": "present"}],
        "partial_outputs": partial_outputs,
        "failure": {"reason": error} if error else None,
        "resume": {"supported": False, "command": None},
    }
    record_count = getattr(result, "n_works", None) if result is not None else None
    source_overrides = {
        "source_type": "openalex_query",
        "query": req.query,
        "filters": filters,
        "record_count": record_count,
    }
    payload = {
        "schema_version": "sciscape_live_job_status_v1",
        "job_id": job_id,
        "status": status,
        "updated_at_utc": now,
        "started_at_utc": job.get("started_at_utc"),
        "finished_at_utc": job.get("finished_at_utc"),
        "request": req.model_dump(),
        "output_dir": str(output_dir),
        "progress": progress_messages,
        "progress_messages_count": len(progress_messages),
        "partial_outputs": partial_outputs,
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


def _run_job(job_id: str, req: QueryRequest) -> None:
    """Execute the OpenAlex pipeline in background."""
    from sciscape.openalex import run_openalex_pipeline, OpenAlexPipelineConfig

    job = _jobs[job_id]
    job["status"] = "running"
    job["started_at_utc"] = _utc_now()
    output_dir = Path("workspace/web_output") / job_id
    filters = _query_filters(req)
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
        )
        result = run_openalex_pipeline(config)

        job["status"] = "done"
        job_result = {
            "n_works": result.n_works,
            "n_edges": result.n_edges,
            "output_dir": str(output_dir),
            "abstracts_path": str(result.abstracts_path) if result.abstracts_path else None,
            "edges_path": str(result.edges_path) if result.edges_path else None,
            "landscape_dir": str(result.landscape_dir) if result.landscape_dir else None,
            "job_status_path": str(output_dir / "job_status.json"),
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
            job_result["features"] = contract["features"]
            job_result["feature_states"] = {
                name: feature.get("state", "hidden")
                for name, feature in manifest.get("features", {}).items()
            }
            job_result["result_state"] = contract["result_state"]
        except Exception as exc:
            job_result["artifact_contract_error"] = str(exc)
        _attach_report_atlas(job_result)
        _attach_evolution_summary(job_result)
        job["result"] = job_result
        _jobs.persist(job_id)
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
