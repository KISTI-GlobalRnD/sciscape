import pytest

from sciscape.openalex.client import OpenAlexClient


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
