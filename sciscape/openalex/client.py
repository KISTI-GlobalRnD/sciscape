"""OpenAlex API client with pagination, rate limiting, and caching.

Polite pool: pass ``email`` for 10x higher rate limit.
Docs: https://docs.openalex.org
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

import requests

log = logging.getLogger(__name__)

API_BASE = "https://api.openalex.org"
FIELDS_WORKS = (
    "id,title,publication_year,type,language,"
    "abstract_inverted_index,authorships,concepts,topics,"
    "cited_by_count,referenced_works,referenced_works_count"
)
# Batch fetch by IDs: max ~50 per request (URL length limit)
BATCH_SIZE = 50
# Rate limit: polite pool = 10 req/s, default = 1 req/s
POLITE_DELAY = 0.1
DEFAULT_DELAY = 1.0


def _reconstruct_abstract(inverted_index: dict | None) -> str:
    """Reconstruct plain text from OpenAlex abstract_inverted_index."""
    if not inverted_index:
        return ""
    positions: list[tuple[int, str]] = []
    for word, idxs in inverted_index.items():
        for pos in idxs:
            positions.append((pos, word))
    positions.sort()
    return " ".join(w for _, w in positions)


@dataclass
class WorkRecord:
    """Parsed OpenAlex work record."""
    id: str
    title: str
    abstract: str
    year: int | None
    referenced_works: List[str]
    cited_by_count: int
    work_type: str
    language: str
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)


class OpenAlexClient:
    """Thin HTTP client for the OpenAlex API.

    Parameters
    ----------
    email : str, optional
        Email for polite pool (recommended).
    cache_dir : Path, optional
        Directory to cache API responses as JSON.
    progress : callable, optional
        Progress callback ``(message: str) -> None``.
    """

    def __init__(
        self,
        email: str | None = None,
        cache_dir: Path | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        self._session = requests.Session()
        self._email = email
        self._delay = POLITE_DELAY if email else DEFAULT_DELAY
        self._cache_dir = Path(cache_dir) if cache_dir else None
        self._progress = progress
        if email:
            self._session.params = {"mailto": email}  # type: ignore[assignment]

    def _log(self, msg: str) -> None:
        log.info(msg)
        if self._progress:
            self._progress(msg)

    def _get(self, url: str, params: dict | None = None) -> dict:
        """GET with rate limiting."""
        time.sleep(self._delay)
        resp = self._session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    # ── Search works ─────────────────────────────────────────

    def search_works(
        self,
        query: str,
        *,
        filters: dict[str, str] | None = None,
        max_results: int = 10000,
        per_page: int = 200,
    ) -> List[WorkRecord]:
        """Search OpenAlex works by query string.

        Parameters
        ----------
        query : str
            Free-text search (title + abstract).
        filters : dict, optional
            Additional API filters, e.g. ``{"publication_year": "2020-2024"}``.
        max_results : int
            Maximum number of works to fetch.
        per_page : int
            Results per API page (max 200).

        Returns
        -------
        list of WorkRecord
        """
        params: dict[str, Any] = {
            "search": query,
            "select": FIELDS_WORKS,
            "per_page": min(per_page, 200),
            "sort": "cited_by_count:desc",
        }
        if filters:
            filter_parts = [f"{k}:{v}" for k, v in filters.items()]
            params["filter"] = ",".join(filter_parts)

        works: List[WorkRecord] = []
        cursor = "*"
        page = 0

        while len(works) < max_results:
            params["cursor"] = cursor
            data = self._get(f"{API_BASE}/works", params)
            results = data.get("results", [])
            if not results:
                break

            for r in results:
                works.append(self._parse_work(r))
                if len(works) >= max_results:
                    break

            cursor = data.get("meta", {}).get("next_cursor")
            if not cursor:
                break

            page += 1
            total = data.get("meta", {}).get("count", "?")
            self._log(f"fetched {len(works)}/{total} works (page {page})")

        self._log(f"search complete: {len(works)} works for '{query}'")
        return works

    # ── Fetch by IDs ─────────────────────────────────────────

    def fetch_works_by_ids(
        self,
        work_ids: Sequence[str],
    ) -> List[WorkRecord]:
        """Fetch works by OpenAlex IDs in batches."""
        # Normalize IDs
        ids = [_normalize_id(wid) for wid in work_ids]
        works: List[WorkRecord] = []

        for i in range(0, len(ids), BATCH_SIZE):
            batch = ids[i:i + BATCH_SIZE]
            pipe_filter = "|".join(batch)
            params = {
                "filter": f"openalex:{pipe_filter}",
                "select": FIELDS_WORKS,
                "per_page": len(batch),
            }
            data = self._get(f"{API_BASE}/works", params)
            for r in data.get("results", []):
                works.append(self._parse_work(r))

            if (i // BATCH_SIZE) % 10 == 0 and i > 0:
                self._log(f"fetched {len(works)}/{len(ids)} works by ID")

        return works

    # ── Fetch referenced works (for citation edges) ──────────

    def fetch_reference_ids(
        self,
        works: Sequence[WorkRecord],
    ) -> Dict[str, List[str]]:
        """Extract citing → [cited] map from already-fetched works.

        No additional API calls needed — referenced_works is in the work record.
        """
        ref_map: Dict[str, List[str]] = {}
        for w in works:
            ref_map[w.id] = w.referenced_works
        return ref_map

    # ── Parse ─────────────────────────────────────────────────

    @staticmethod
    def _parse_work(raw: dict) -> WorkRecord:
        abstract = _reconstruct_abstract(raw.get("abstract_inverted_index"))
        refs = raw.get("referenced_works") or []
        # Normalize reference IDs
        refs = [_normalize_id(r) for r in refs]

        return WorkRecord(
            id=_normalize_id(raw.get("id", "")),
            title=raw.get("title") or "",
            abstract=abstract,
            year=raw.get("publication_year"),
            referenced_works=refs,
            cited_by_count=raw.get("cited_by_count") or 0,
            work_type=raw.get("type") or "",
            language=raw.get("language") or "",
            raw=raw,
        )


def _normalize_id(oa_id: str) -> str:
    """Normalize OpenAlex ID to short form (W1234567890)."""
    if oa_id.startswith("https://openalex.org/"):
        return oa_id.split("/")[-1]
    return oa_id


__all__ = ["OpenAlexClient", "WorkRecord"]
