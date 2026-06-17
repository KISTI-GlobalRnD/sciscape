import pytest
import requests

from sciscape.openalex.client import OpenAlexClient


class FakeResponse:
    def __init__(self, status_code, payload=None, headers=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)

    def json(self):
        return self._payload


def test_openalex_client_checkpoint_runs_before_http(monkeypatch):
    def checkpoint():
        raise RuntimeError("cancelled")

    client = OpenAlexClient(checkpoint=checkpoint)
    sleeps: list[float] = []

    def fake_sleep(delay):
        sleeps.append(delay)

    def fake_get(*args, **kwargs):
        raise AssertionError("HTTP request should not run after checkpoint cancellation")

    monkeypatch.setattr("time.sleep", fake_sleep)
    monkeypatch.setattr(client._session, "get", fake_get)

    with pytest.raises(RuntimeError, match="cancelled"):
        client._get("https://api.openalex.org/works")

    assert sleeps == []


def test_openalex_client_retries_429_with_retry_after(monkeypatch):
    progress: list[str] = []
    snapshots: list[dict] = []
    client = OpenAlexClient(
        progress=progress.append,
        telemetry=snapshots.append,
        max_retries=1,
        request_timeout=7,
    )
    responses = [
        FakeResponse(429, headers={"Retry-After": "2"}),
        FakeResponse(200, payload={"results": [{"id": "https://openalex.org/W1"}]}),
    ]
    calls: list[dict] = []
    sleeps: list[float] = []

    def fake_sleep(delay):
        sleeps.append(delay)

    def fake_get(url, params=None, timeout=None):
        calls.append({"url": url, "params": params, "timeout": timeout})
        return responses.pop(0)

    monkeypatch.setattr("time.sleep", fake_sleep)
    monkeypatch.setattr(client._session, "get", fake_get)

    payload = client._get("https://api.openalex.org/works", {"search": "graph"})

    assert payload["results"][0]["id"] == "https://openalex.org/W1"
    assert [call["timeout"] for call in calls] == [7, 7]
    assert sleeps == [1.0, 1.0, 1.0, 1.0]
    assert any("OpenAlex HTTP 429" in msg for msg in progress)
    telemetry = client.telemetry()
    assert telemetry["attempts_total"] == 2
    assert telemetry["successful_requests_total"] == 1
    assert telemetry["failed_requests_total"] == 0
    assert telemetry["retry_attempts_total"] == 1
    assert telemetry["rate_limit_wait_seconds_total"] == 2.0
    assert telemetry["retry_wait_seconds_total"] == 2.0
    assert telemetry["status_counts"] == {"429": 1, "200": 1}
    assert snapshots[-1]["status_counts"] == {"429": 1, "200": 1}


def test_openalex_client_retries_timeout_then_succeeds(monkeypatch):
    progress: list[str] = []
    client = OpenAlexClient(
        progress=progress.append,
        max_retries=1,
        backoff_base=0.5,
        request_timeout=3,
    )
    calls = 0
    sleeps: list[float] = []

    def fake_sleep(delay):
        sleeps.append(delay)

    def fake_get(url, params=None, timeout=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise requests.Timeout("slow")
        return FakeResponse(200, payload={"ok": True})

    monkeypatch.setattr("time.sleep", fake_sleep)
    monkeypatch.setattr(client._session, "get", fake_get)

    assert client._get("https://api.openalex.org/works") == {"ok": True}
    assert calls == 2
    assert sleeps == [1.0, 0.5, 1.0]
    assert any("Timeout" in msg for msg in progress)
    telemetry = client.telemetry()
    assert telemetry["attempts_total"] == 2
    assert telemetry["successful_requests_total"] == 1
    assert telemetry["retry_attempts_total"] == 1
    assert telemetry["rate_limit_wait_seconds_total"] == 2.0
    assert telemetry["retry_wait_seconds_total"] == 0.5
    assert telemetry["exception_counts"] == {"Timeout": 1}
    assert telemetry["status_counts"] == {"200": 1}


def test_openalex_client_checkpoint_can_cancel_retry_sleep(monkeypatch):
    checkpoint_calls = 0

    def checkpoint():
        nonlocal checkpoint_calls
        checkpoint_calls += 1
        if checkpoint_calls >= 4:
            raise RuntimeError("cancelled")

    client = OpenAlexClient(checkpoint=checkpoint, max_retries=1)

    def fake_sleep(delay):
        pass

    def fake_get(url, params=None, timeout=None):
        return FakeResponse(429, headers={"Retry-After": "5"})

    monkeypatch.setattr("time.sleep", fake_sleep)
    monkeypatch.setattr(client._session, "get", fake_get)

    with pytest.raises(RuntimeError, match="cancelled"):
        client._get("https://api.openalex.org/works")
