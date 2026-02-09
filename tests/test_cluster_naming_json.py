from __future__ import annotations

import clustering.cluster_naming as cluster_naming


class _DummyClient:
    pass


def test_detect_and_translate_parses_fenced_json(monkeypatch) -> None:
    def fake_call_chat_completion(*args, **kwargs) -> str:
        return "```json\n{\"lang\":\"en\",\"text\":\"\"}\n```"

    monkeypatch.setattr(cluster_naming, "_call_chat_completion", fake_call_chat_completion)
    result = cluster_naming.detect_and_translate(_DummyClient(), "Hello world", model="dummy-model")
    assert result["lang"] == "en"
    assert result["text"] == "Hello world"


def test_detect_and_translate_falls_back_on_invalid_json(monkeypatch) -> None:
    def fake_call_chat_completion(*args, **kwargs) -> str:
        return "NOT JSON"

    monkeypatch.setattr(cluster_naming, "_call_chat_completion", fake_call_chat_completion)
    result = cluster_naming.detect_and_translate(_DummyClient(), "Hello", model="dummy-model")
    assert result["lang"] == "Invalid JSON"
