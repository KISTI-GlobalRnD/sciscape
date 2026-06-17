"""OpenAlex API client with pagination, rate limiting, and caching.

Polite pool: pass ``email`` for 10x higher rate limit.
Docs: https://docs.openalex.org
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
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
REQUEST_TIMEOUT = 30.0
MAX_RETRIES = 3
BACKOFF_BASE = 1.0
BACKOFF_MAX = 30.0
RETRY_STATUS_CODES = {408, 429, 500, 502, 503, 504}


class OpenAlexQuotaBudgetExceeded(RuntimeError):
    """Raised when an OpenAlex query exceeds configured API budget limits."""


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
        checkpoint: Callable[[], None] | None = None,
        telemetry: Callable[[dict[str, Any]], None] | None = None,
        request_timeout: float = REQUEST_TIMEOUT,
        max_retries: int = MAX_RETRIES,
        backoff_base: float = BACKOFF_BASE,
        backoff_max: float = BACKOFF_MAX,
        api_attempt_budget: int | None = None,
        retry_wait_budget_seconds: float | None = None,
        interruptible_requests: bool = False,
        request_poll_interval: float = 0.25,
    ) -> None:
        self._session = requests.Session()
        self._email = email
        self._delay = POLITE_DELAY if email else DEFAULT_DELAY
        self._cache_dir = Path(cache_dir) if cache_dir else None
        self._progress = progress
        self._checkpoint = checkpoint
        self._telemetry_callback = telemetry
        self._request_timeout = max(0.1, float(request_timeout))
        self._max_retries = max(0, int(max_retries))
        self._backoff_base = max(0.0, float(backoff_base))
        self._backoff_max = max(0.0, float(backoff_max))
        self._api_attempt_budget = max(1, int(api_attempt_budget)) if api_attempt_budget is not None else None
        self._retry_wait_budget_seconds = (
            max(0.0, float(retry_wait_budget_seconds))
            if retry_wait_budget_seconds is not None
            else None
        )
        self._interruptible_requests = bool(interruptible_requests)
        self._request_poll_interval = max(0.05, float(request_poll_interval))
        self._telemetry: dict[str, Any] = {
            "source": "openalex",
            "polite_pool": bool(email),
            "configured_delay_seconds": self._delay,
            "request_timeout_seconds": self._request_timeout,
            "interruptible_requests": self._interruptible_requests,
            "request_poll_interval_seconds": self._request_poll_interval,
            "max_retries": self._max_retries,
            "backoff_base_seconds": self._backoff_base,
            "backoff_max_seconds": self._backoff_max,
            "api_attempt_budget": self._api_attempt_budget,
            "retry_wait_budget_seconds": self._retry_wait_budget_seconds,
            "quota_budget_exceeded": False,
            "quota_abort_reason": None,
            "attempts_total": 0,
            "successful_requests_total": 0,
            "failed_requests_total": 0,
            "retry_attempts_total": 0,
            "inflight_cancel_checks_total": 0,
            "inflight_interruptions_total": 0,
            "rate_limit_wait_seconds_total": 0.0,
            "retry_wait_seconds_total": 0.0,
            "status_counts": {},
            "exception_counts": {},
            "last_status_code": None,
            "last_exception": None,
            "last_retry_wait_seconds": None,
        }
        if email:
            self._session.params = {"mailto": email}  # type: ignore[assignment]

    def telemetry(self) -> dict[str, Any]:
        data = dict(self._telemetry)
        data["status_counts"] = dict(data.get("status_counts") or {})
        data["exception_counts"] = dict(data.get("exception_counts") or {})
        return data

    def _notify_telemetry(self) -> None:
        if self._telemetry_callback is not None:
            self._telemetry_callback(self.telemetry())

    def _increment_telemetry_counter(self, section: str, key: str | int) -> None:
        counters = self._telemetry.setdefault(section, {})
        text_key = str(key)
        counters[text_key] = int(counters.get(text_key, 0)) + 1

    def _abort_for_budget(self, reason: str) -> None:
        self._telemetry["quota_budget_exceeded"] = True
        self._telemetry["quota_abort_reason"] = reason
        self._notify_telemetry()
        raise OpenAlexQuotaBudgetExceeded(reason)

    def _check_attempt_budget(self) -> None:
        if self._api_attempt_budget is None:
            return
        attempts = int(self._telemetry.get("attempts_total") or 0)
        if attempts >= self._api_attempt_budget:
            self._abort_for_budget(
                f"OpenAlex API attempt budget exceeded: {attempts}/{self._api_attempt_budget}"
            )

    def _check_retry_wait_budget(self, next_delay: float) -> None:
        if self._retry_wait_budget_seconds is None:
            return
        used = float(self._telemetry.get("retry_wait_seconds_total") or 0.0)
        projected = used + max(0.0, next_delay)
        if projected > self._retry_wait_budget_seconds:
            self._abort_for_budget(
                "OpenAlex retry wait budget exceeded: "
                f"{projected:.1f}/{self._retry_wait_budget_seconds:.1f}s"
            )

    def _record_response(self, resp: requests.Response) -> None:
        status_code = int(resp.status_code)
        self._telemetry["last_status_code"] = status_code
        self._increment_telemetry_counter("status_counts", status_code)
        self._notify_telemetry()

    def _record_exception(self, exc: requests.RequestException) -> None:
        name = exc.__class__.__name__
        self._telemetry["last_exception"] = name
        self._increment_telemetry_counter("exception_counts", name)
        self._notify_telemetry()

    def _checkpoint_or_continue(self) -> None:
        if self._checkpoint is not None:
            self._checkpoint()

    def _log(self, msg: str) -> None:
        log.info(msg)
        if self._progress:
            self._progress(msg)

    def _sleep_with_checkpoint(self, delay: float, *, telemetry_bucket: str | None = None) -> None:
        remaining = max(0.0, delay)
        if telemetry_bucket == "rate_limit":
            self._telemetry["rate_limit_wait_seconds_total"] = round(
                float(self._telemetry.get("rate_limit_wait_seconds_total") or 0.0) + remaining,
                6,
            )
            self._notify_telemetry()
        elif telemetry_bucket == "retry":
            self._telemetry["retry_wait_seconds_total"] = round(
                float(self._telemetry.get("retry_wait_seconds_total") or 0.0) + remaining,
                6,
            )
            self._telemetry["last_retry_wait_seconds"] = remaining
            self._notify_telemetry()
        while remaining > 0:
            self._checkpoint_or_continue()
            step = min(remaining, 1.0)
            time.sleep(step)
            remaining -= step
        self._checkpoint_or_continue()

    @staticmethod
    def _retry_after_delay(value: str | None) -> float | None:
        if not value:
            return None
        try:
            return max(0.0, float(value))
        except ValueError:
            pass
        try:
            dt = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        now = parsedate_to_datetime(time.strftime("%a, %d %b %Y %H:%M:%S GMT", time.gmtime()))
        return max(0.0, (dt - now).total_seconds())

    def _retry_delay(self, attempt: int, resp: requests.Response | None = None) -> float:
        retry_after = self._retry_after_delay(resp.headers.get("Retry-After") if resp is not None else None)
        if retry_after is not None:
            return min(retry_after, self._backoff_max)
        if self._backoff_base <= 0:
            return 0.0
        return min(self._backoff_base * (2 ** max(0, attempt - 1)), self._backoff_max)

    @staticmethod
    def _should_retry_response(resp: requests.Response) -> bool:
        return int(resp.status_code) in RETRY_STATUS_CODES

    @staticmethod
    def _should_retry_exception(exc: requests.RequestException) -> bool:
        return isinstance(exc, (requests.Timeout, requests.ConnectionError))

    def _request_get(self, url: str, params: dict | None = None) -> requests.Response:
        if not self._interruptible_requests:
            return self._session.get(url, params=params, timeout=self._request_timeout)

        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sciscape-openalex-request")
        future = executor.submit(self._session.get, url, params=params, timeout=self._request_timeout)
        try:
            while True:
                try:
                    return future.result(timeout=self._request_poll_interval)
                except FutureTimeoutError:
                    self._telemetry["inflight_cancel_checks_total"] = int(
                        self._telemetry.get("inflight_cancel_checks_total") or 0
                    ) + 1
                    self._notify_telemetry()
                    try:
                        self._checkpoint_or_continue()
                    except Exception:
                        self._telemetry["inflight_interruptions_total"] = int(
                            self._telemetry.get("inflight_interruptions_total") or 0
                        ) + 1
                        self._notify_telemetry()
                        future.cancel()
                        self._session.close()
                        raise
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def _get(self, url: str, params: dict | None = None) -> dict:
        """GET with rate limiting and bounded retry for transient failures."""
        attempts = self._max_retries + 1
        last_exc: requests.RequestException | None = None
        for attempt in range(1, attempts + 1):
            self._checkpoint_or_continue()
            self._check_attempt_budget()
            self._sleep_with_checkpoint(self._delay, telemetry_bucket="rate_limit")
            try:
                self._telemetry["attempts_total"] = int(self._telemetry.get("attempts_total") or 0) + 1
                self._notify_telemetry()
                resp = self._request_get(url, params=params)
                self._checkpoint_or_continue()
            except requests.RequestException as exc:
                last_exc = exc
                self._record_exception(exc)
                if attempt >= attempts or not self._should_retry_exception(exc):
                    self._telemetry["failed_requests_total"] = int(
                        self._telemetry.get("failed_requests_total") or 0
                    ) + 1
                    self._notify_telemetry()
                    raise
                delay = self._retry_delay(attempt)
                self._check_retry_wait_budget(delay)
                self._telemetry["retry_attempts_total"] = int(
                    self._telemetry.get("retry_attempts_total") or 0
                ) + 1
                self._notify_telemetry()
                self._log(
                    f"OpenAlex request failed ({exc.__class__.__name__}); "
                    f"retrying in {delay:.1f}s ({attempt}/{self._max_retries})"
                )
                self._sleep_with_checkpoint(delay, telemetry_bucket="retry")
                continue

            self._record_response(resp)
            if not self._should_retry_response(resp) or attempt >= attempts:
                if resp.status_code >= 400:
                    self._telemetry["failed_requests_total"] = int(
                        self._telemetry.get("failed_requests_total") or 0
                    ) + 1
                    self._notify_telemetry()
                else:
                    self._telemetry["successful_requests_total"] = int(
                        self._telemetry.get("successful_requests_total") or 0
                    ) + 1
                    self._notify_telemetry()
                resp.raise_for_status()
                return resp.json()

            delay = self._retry_delay(attempt, resp)
            self._check_retry_wait_budget(delay)
            self._telemetry["retry_attempts_total"] = int(
                self._telemetry.get("retry_attempts_total") or 0
            ) + 1
            self._notify_telemetry()
            self._log(
                f"OpenAlex HTTP {resp.status_code}; retrying in {delay:.1f}s "
                f"({attempt}/{self._max_retries})"
            )
            self._sleep_with_checkpoint(delay, telemetry_bucket="retry")

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("OpenAlex retry loop exited without a response")

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
            self._checkpoint_or_continue()
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

        self._checkpoint_or_continue()
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
            self._checkpoint_or_continue()
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

        self._checkpoint_or_continue()
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


__all__ = ["OpenAlexClient", "OpenAlexQuotaBudgetExceeded", "WorkRecord"]
