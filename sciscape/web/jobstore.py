"""Persistent job store backed by SQLite.

Stores job metadata and progress in a local SQLite database so that
jobs survive server restarts. Falls back to in-memory dict if SQLite
is unavailable.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

_DEFAULT_DB = Path("sciscape_jobs.db")


class JobStore:
    """Thread-safe job store with SQLite persistence.

    Supports dict-like access for backward compatibility:
        store[job_id] returns a mutable proxy dict.
    """

    def __init__(self, db_path: Path | str | None = None):
        self._db_path = str(db_path) if db_path else str(_DEFAULT_DB)
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        # In-memory cache for fast access (synced to SQLite on state transitions)
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._init_db()

    def _init_db(self) -> None:
        try:
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL DEFAULT 'pending',
                    request TEXT DEFAULT '{}',
                    result TEXT DEFAULT NULL,
                    progress TEXT DEFAULT '[]',
                    started_at_utc TEXT DEFAULT NULL,
                    finished_at_utc TEXT DEFAULT NULL,
                    updated_at_utc TEXT DEFAULT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self._ensure_timestamp_columns()
            self._conn.commit()
            log.info("JobStore: SQLite at %s", self._db_path)
        except Exception as e:
            log.warning("JobStore: SQLite failed (%s), using in-memory", e)
            self._conn = None
            self._memory: Dict[str, Dict[str, Any]] = {}

    def _use_sqlite(self) -> bool:
        return self._conn is not None

    def _ensure_timestamp_columns(self) -> None:
        if self._conn is None:
            return
        existing = {
            row[1]
            for row in self._conn.execute("PRAGMA table_info(jobs)").fetchall()
        }
        for column in ("started_at_utc", "finished_at_utc", "updated_at_utc"):
            if column not in existing:
                self._conn.execute(f"ALTER TABLE jobs ADD COLUMN {column} TEXT DEFAULT NULL")

    def create(self, job_id: str, request: dict) -> None:
        job = {"status": "pending", "progress": [], "result": None, "request": request}
        with self._lock:
            self._cache[job_id] = job
            if self._use_sqlite():
                self._conn.execute(
                    "INSERT OR REPLACE INTO jobs (job_id, status, request, progress) VALUES (?, 'pending', ?, '[]')",
                    (job_id, json.dumps(request)),
                )
                self._conn.commit()

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job by ID. Returns mutable cache dict (mutations auto-sync on persist)."""
        with self._lock:
            if job_id in self._cache:
                return self._cache[job_id]
            # Try loading from SQLite
            if self._use_sqlite():
                row = self._conn.execute(
                    (
                        "SELECT status, request, result, progress, "
                        "started_at_utc, finished_at_utc, updated_at_utc "
                        "FROM jobs WHERE job_id = ?"
                    ),
                    (job_id,),
                ).fetchone()
                if row is not None:
                    job = {
                        "status": row[0],
                        "request": json.loads(row[1] or "{}"),
                        "result": json.loads(row[2]) if row[2] else None,
                        "progress": json.loads(row[3] or "[]"),
                        "started_at_utc": row[4],
                        "finished_at_utc": row[5],
                        "updated_at_utc": row[6],
                    }
                    self._cache[job_id] = job
                    return job
            return None

    def update(self, job_id: str, **kwargs) -> None:
        with self._lock:
            if self._use_sqlite():
                sets = []
                vals = []
                for k, v in kwargs.items():
                    if k in ("status", "started_at_utc", "finished_at_utc", "updated_at_utc"):
                        sets.append(f"{k} = ?")
                        vals.append(v)
                    elif k in ("result", "request"):
                        sets.append(f"{k} = ?")
                        vals.append(json.dumps(v) if v is not None else None)
                    elif k == "progress":
                        sets.append("progress = ?")
                        vals.append(json.dumps(v))
                if sets:
                    vals.append(job_id)
                    self._conn.execute(
                        f"UPDATE jobs SET {', '.join(sets)} WHERE job_id = ?", vals,
                    )
                    self._conn.commit()
            else:
                job = self._memory.get(job_id)
                if job:
                    job.update(kwargs)

    def persist(self, job_id: str) -> None:
        """Sync cached job state to SQLite."""
        with self._lock:
            job = self._cache.get(job_id)
            if job and self._use_sqlite():
                self._conn.execute(
                    (
                        "UPDATE jobs SET status=?, result=?, progress=?, "
                        "started_at_utc=?, finished_at_utc=?, updated_at_utc=? "
                        "WHERE job_id=?"
                    ),
                    (job["status"],
                     json.dumps(job.get("result")) if job.get("result") else None,
                     json.dumps(job.get("progress", [])),
                     job.get("started_at_utc"),
                     job.get("finished_at_utc"),
                     job.get("updated_at_utc"),
                     job_id),
                )
                self._conn.commit()

    def append_progress(self, job_id: str, msg: str) -> None:
        with self._lock:
            if self._use_sqlite():
                row = self._conn.execute(
                    "SELECT progress FROM jobs WHERE job_id = ?", (job_id,),
                ).fetchone()
                if row:
                    progress = json.loads(row[0] or "[]")
                    progress.append(msg)
                    self._conn.execute(
                        "UPDATE jobs SET progress = ? WHERE job_id = ?",
                        (json.dumps(progress), job_id),
                    )
                    self._conn.commit()
            else:
                job = self._memory.get(job_id)
                if job:
                    job["progress"].append(msg)

    # ── Dict-like interface for backward compatibility ──

    def __contains__(self, job_id: str) -> bool:
        return self.get(job_id) is not None

    def __getitem__(self, job_id: str) -> Dict[str, Any]:
        job = self.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return job

    def __setitem__(self, job_id: str, value: Dict[str, Any]) -> None:
        if self.get(job_id) is None:
            self.create(job_id, value.get("request", {}))
        self.update(job_id, **{k: v for k, v in value.items() if k != "request"})

    def items(self):
        with self._lock:
            if self._use_sqlite():
                rows = self._conn.execute(
                    (
                        "SELECT job_id, status, request, result, progress, "
                        "started_at_utc, finished_at_utc, updated_at_utc FROM jobs"
                    )
                ).fetchall()
                return [
                    (r[0], {"status": r[1], "request": json.loads(r[2] or "{}"),
                            "result": json.loads(r[3]) if r[3] else None,
                            "progress": json.loads(r[4] or "[]"),
                            "started_at_utc": r[5],
                            "finished_at_utc": r[6],
                            "updated_at_utc": r[7]})
                    for r in rows
                ]
            else:
                return list(self._memory.items())

    def list_all(self) -> List[Dict[str, Any]]:
        with self._lock:
            if self._use_sqlite():
                rows = self._conn.execute(
                    (
                        "SELECT job_id, status, request, started_at_utc, "
                        "finished_at_utc, updated_at_utc FROM jobs ORDER BY created_at DESC"
                    )
                ).fetchall()
                return [
                    {
                        "job_id": r[0],
                        "status": r[1],
                        "query": json.loads(r[2] or "{}").get("query", ""),
                        "started_at_utc": r[3],
                        "finished_at_utc": r[4],
                        "updated_at_utc": r[5],
                    }
                    for r in rows
                ]
            else:
                return [
                    {"job_id": jid, "status": j["status"], "query": j["request"].get("query", "")}
                    for jid, j in self._memory.items()
                ]


# Module-level singleton
_store: Optional[JobStore] = None


def get_store(db_path: Path | str | None = None) -> JobStore:
    """Get or create the global JobStore singleton."""
    global _store
    if _store is None:
        _store = JobStore(db_path)
    return _store
