import json
from pathlib import Path

import pandas as pd

from sciscape.keyword_extraction import KeywordExtractionConfig, run_keyword_pipeline
from sciscape.keyword_extraction.pipeline import KeywordExtractionPipeline


def test_keyword_pipeline_smoke(tmp_path):
    abstracts = pd.DataFrame(
        {
            "uid": ["D1", "D2", "D3", "D4"],
            "title": [
                "Advances in quantum sensors",
                "Resilient grids with machine learning",
                "Battery lifecycle modelling",
                "Solar materials innovation",
            ],
            "abstract": [
                "Quantum sensing enables precise magnetic field measurements.",
                "Machine learning improves grid resilience and forecasting accuracy.",
                "Lifecycle models for batteries utilise physics informed learning.",
                "New perovskite solar materials deliver higher efficiency cells.",
            ],
            "pubyear": [2018, 2019, 2020, 2021],
        }
    )
    membership = pd.DataFrame(
        {
            "uid": ["D1", "D2", "D3", "D4"],
            "cluster": [0, 1, 1, 0],
        }
    )

    abstract_path = Path(tmp_path) / "abstracts.parquet"
    membership_path = Path(tmp_path) / "membership.parquet"
    abstracts.to_parquet(abstract_path, index=False)
    membership.to_parquet(membership_path, index=False)

    cfg = KeywordExtractionConfig(
        abstract_path=abstract_path,
        membership_path=membership_path,
        cluster_level="cluster",
        include_title=True,
        title_weight=1.0,
        min_df_unigram=1,
        min_df_phrase=1,
        max_df_unigram=1.0,
        max_df_phrase=1.0,
        phrase_min_count_per_cluster=1,
        min_cluster_doc_coverage=1,
        min_cluster_doc_coverage_ratio=0.1,
        top_n_keywords=3,
        ngram_min=2,
        ngram_max=3,
        use_phrase_vectorizer=True,
        mmr_jaccard_lambda=0.3,
        mmr_pool_factor=3.0,
        w_llr=0.5,
        n_jobs=1,
    )

    keywords = run_keyword_pipeline(cfg)
    assert not keywords.empty
    required = {"cluster_id", "term", "score", "frequency", "pub_year_series"}
    assert required.issubset(set(keywords.columns))


def test_cluster_task_progress_file(tmp_path):
    abstracts = pd.DataFrame(
        {
            "uid": ["D1", "D2"],
            "title": ["Quantum sensors", "Solar cells"],
            "abstract": ["Quantum sensing text.", "Solar cell text."],
            "pubyear": [2020, 2021],
        }
    )
    membership = pd.DataFrame({"uid": ["D1", "D2"], "cluster": [0, 1]})

    abstract_path = Path(tmp_path) / "abstracts.parquet"
    membership_path = Path(tmp_path) / "membership.parquet"
    progress_path = Path(tmp_path) / "keyword_progress.json"
    abstracts.to_parquet(abstract_path, index=False)
    membership.to_parquet(membership_path, index=False)

    cfg = KeywordExtractionConfig(
        abstract_path=abstract_path,
        membership_path=membership_path,
        cluster_level="cluster",
        parallel_backend="sequential",
        progress_path=progress_path,
        progress_interval_clusters=1,
    )
    pipeline = KeywordExtractionPipeline(cfg)

    assert pipeline._run_cluster_tasks("unit_stage", 3, lambda idx: idx + 10) == [10, 11, 12]
    payload = json.loads(progress_path.read_text())
    assert payload["stage"] == "unit_stage"
    assert payload["processed"] == 3
    assert payload["total"] == 3
    assert payload["parallel_backend"] == "sequential"


def test_keyword_pipeline_scoring_shards_match_unsharded(tmp_path):
    abstracts = pd.DataFrame(
        {
            "uid": ["D1", "D2", "D3", "D4", "D5", "D6"],
            "title": [
                "Quantum sensor devices",
                "Quantum magnetic sensing",
                "Solar cell materials",
                "Perovskite solar devices",
                "Graph neural networks",
                "Traffic graph forecasting",
            ],
            "abstract": [
                "Quantum sensing improves magnetic field measurement.",
                "Magnetic quantum sensors support precision metrology.",
                "Solar cell materials improve photovoltaic efficiency.",
                "Perovskite solar cells improve device efficiency.",
                "Graph neural networks learn molecular representations.",
                "Traffic forecasting uses graph neural network models.",
            ],
            "pubyear": [2018, 2019, 2020, 2021, 2022, 2023],
        }
    )
    membership = pd.DataFrame(
        {"uid": ["D1", "D2", "D3", "D4", "D5", "D6"], "cluster": [0, 0, 1, 1, 2, 2]}
    )
    abstract_path = Path(tmp_path) / "abstracts.parquet"
    membership_path = Path(tmp_path) / "membership.parquet"
    abstracts.to_parquet(abstract_path, index=False)
    membership.to_parquet(membership_path, index=False)

    common = dict(
        abstract_path=abstract_path,
        membership_path=membership_path,
        cluster_level="cluster",
        include_title=True,
        title_weight=1.0,
        min_df_unigram=1,
        min_df_phrase=1,
        phrase_min_count_per_cluster=1,
        top_n_keywords=4,
        scoring_pool_factor=1.0,
        ngram_min=2,
        ngram_max=3,
        use_phrase_vectorizer=True,
        normalization_enabled=False,
        quality_diagnostics_enabled=False,
        abbreviation_dictionary_enabled=False,
        n_jobs=1,
        parallel_backend="sequential",
    )
    unsharded = run_keyword_pipeline(KeywordExtractionConfig(**common))
    shard_dir = Path(tmp_path) / "scoring_shards"
    progress_path = Path(tmp_path) / "progress.json"
    sharded = run_keyword_pipeline(
        KeywordExtractionConfig(
            **common,
            scoring_shard_dir=shard_dir,
            scoring_shard_size_clusters=1,
            progress_path=progress_path,
            progress_interval_clusters=1,
        )
    )

    cols = ["cluster_id", "term", "score", "frequency", "doc_coverage"]
    left = unsharded[cols].sort_values(cols[:2]).reset_index(drop=True)
    right = sharded[cols].sort_values(cols[:2]).reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right, check_exact=False, atol=1e-12)

    manifest = json.loads((shard_dir / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["shard_count"] == 3
    assert len(list(shard_dir.glob("*.done.json"))) == 3
    progress = json.loads(progress_path.read_text())
    assert progress["stage"] == "complete"


def test_sos_shim_imports():
    import sos

    from sos.keyword_extraction import KeywordExtractionConfig as ShimCfg
    from sos.keyword_extraction import run_keyword_pipeline as shim_run

    assert sos.__version__
    assert ShimCfg is KeywordExtractionConfig
    assert shim_run is run_keyword_pipeline
