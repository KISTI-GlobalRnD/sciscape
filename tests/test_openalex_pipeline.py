import pytest

import sciscape.artifacts as artifacts
import sciscape.openalex.pipeline as pipeline
from sciscape.openalex.client import WorkRecord


class JobCancelled(RuntimeError):
    pass


def test_openalex_pipeline_does_not_swallow_cancellation_in_manifest_guard(monkeypatch, tmp_path):
    class FakeClient:
        def __init__(self, **kwargs):
            self.checkpoint = kwargs.get("checkpoint")

        def search_works(self, query, *, filters=None, max_results=10000, per_page=200):
            return [
                WorkRecord(
                    id="W1",
                    title="Graph neural networks",
                    abstract="Graph neural networks for citation analysis.",
                    year=2024,
                    referenced_works=[],
                    cited_by_count=3,
                    work_type="article",
                    language="en",
                )
            ]

    def raise_cancel(*args, **kwargs):
        raise JobCancelled("stop")

    monkeypatch.setattr(pipeline, "OpenAlexClient", FakeClient)
    monkeypatch.setattr(artifacts, "write_result_manifest", raise_cancel)

    config = pipeline.OpenAlexPipelineConfig(
        query="graph neural networks",
        max_works=1,
        output_dir=tmp_path / "out",
        run_landscape=False,
    )

    with pytest.raises(JobCancelled, match="stop"):
        pipeline.run_openalex_pipeline(config)


def test_openalex_pipeline_passes_retry_config_to_client(monkeypatch, tmp_path):
    captured_kwargs = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

        def search_works(self, query, *, filters=None, max_results=10000, per_page=200):
            return []

    monkeypatch.setattr(pipeline, "OpenAlexClient", FakeClient)

    config = pipeline.OpenAlexPipelineConfig(
        query="graph neural networks",
        max_works=1,
        output_dir=tmp_path / "out",
        request_timeout=9,
        max_retries=5,
        backoff_base=0.25,
        backoff_max=4,
    )

    result = pipeline.run_openalex_pipeline(config)

    assert result.n_works == 0
    assert captured_kwargs["request_timeout"] == 9
    assert captured_kwargs["max_retries"] == 5
    assert captured_kwargs["backoff_base"] == 0.25
    assert captured_kwargs["backoff_max"] == 4
