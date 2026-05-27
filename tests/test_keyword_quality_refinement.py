"""Tests for domain-agnostic keyword quality refinement."""

import pandas as pd
import pytest

from sciscape.keyword_extraction import (
    KeywordExtractionConfig,
    annotate_keyword_quality,
    quality_flag_counts,
    run_keyword_pipeline,
)


def test_quality_annotation_demotes_global_terms_and_prefers_phrases():
    df = pd.DataFrame(
        {
            "cluster_id": [0, 0, 0, 0, 1, 1, 1, 2, 2, 2],
            "term": [
                "graph",
                "graph neural network",
                "traffic flow prediction",
                "abstract",
                "graph",
                "drug drug interaction",
                "ddi",
                "graph",
                "self assembled monolayer",
                "sam",
            ],
            "score": [10.0, 8.0, 7.5, 7.0, 10.0, 8.0, 6.0, 10.0, 8.0, 6.0],
            "frequency": [30, 10, 8, 20, 30, 10, 8, 30, 10, 8],
            "doc_coverage": [25, 8, 7, 18, 25, 8, 7, 25, 8, 7],
        }
    )

    result = annotate_keyword_quality(df, rerank=True)

    c0_terms = result[result["cluster_id"] == 0]["term"].tolist()
    assert c0_terms.index("traffic flow prediction") < c0_terms.index("graph")
    assert c0_terms.index("graph neural network") < c0_terms.index("graph")

    graph_rows = result[result["term"] == "graph"]
    assert graph_rows["quality_flags"].str.contains("too_global").all()
    assert graph_rows["quality_flags"].str.contains("phrase_preferred").any()

    abstract = result[result["term"] == "abstract"].iloc[0]
    assert "artifact_like" in abstract["quality_flags"]
    assert abstract["quality_score"] < abstract["score"]


def test_quality_annotation_expands_acronym_display_labels():
    df = pd.DataFrame(
        {
            "cluster_id": [0, 0, 1, 1, 2, 2],
            "term": [
                "drug drug interaction",
                "ddi",
                "self assembled monolayer",
                "sam",
                "single cell rna sequencing",
                "scrna",
            ],
            "score": [8.0, 6.0, 8.0, 6.0, 8.0, 6.0],
            "frequency": [10, 5, 10, 5, 10, 5],
        }
    )

    result = annotate_keyword_quality(df, rerank=True)
    labels = dict(zip(result["term"], result["display_label"]))

    assert labels["ddi"] == "drug drug interaction"
    assert labels["sam"] == "self assembled monolayer"
    assert labels["scrna"] == "single cell rna sequencing"
    assert "acronym_like" in result[result["term"] == "ddi"].iloc[0]["quality_flags"]


def test_quality_flag_counts_counts_pipe_delimited_flags():
    df = pd.DataFrame({"quality_flags": ["too_global|phrase_preferred", "phrase", ""]})
    assert quality_flag_counts(df) == {
        "too_global": 1,
        "phrase_preferred": 1,
        "phrase": 1,
    }


@pytest.fixture
def quality_pipeline_data(tmp_path):
    abstracts = pd.DataFrame(
        {
            "uid": [f"D{i}" for i in range(6)],
            "title": [
                "Graph neural network traffic prediction",
                "Traffic flow prediction with graph models",
                "Graph neural network drug interaction",
                "Drug drug interaction prediction",
                "Self assembled monolayer perovskite solar cell",
                "Perovskite solar cell transport layer",
            ],
            "abstract": [
                "Graph models support traffic flow prediction and road network forecasting.",
                "Traffic flow prediction uses graph neural network architectures.",
                "Graph neural network methods identify drug drug interaction signals.",
                "Drug drug interaction prediction estimates binding affinity.",
                "Self assembled monolayer improves perovskite solar cell interfaces.",
                "Perovskite solar cell electron transport layer controls efficiency.",
            ],
            "pubyear": [2020, 2021, 2020, 2021, 2020, 2021],
        }
    )
    membership = pd.DataFrame(
        {
            "uid": [f"D{i}" for i in range(6)],
            "cluster": [0, 0, 1, 1, 2, 2],
        }
    )
    abstract_path = tmp_path / "abstracts.parquet"
    membership_path = tmp_path / "membership.parquet"
    abstracts.to_parquet(abstract_path, index=False)
    membership.to_parquet(membership_path, index=False)
    return abstract_path, membership_path


def test_pipeline_emits_quality_columns_when_enabled(quality_pipeline_data):
    cfg = KeywordExtractionConfig(
        abstract_path=quality_pipeline_data[0],
        membership_path=quality_pipeline_data[1],
        cluster_level="cluster",
        include_title=True,
        title_weight=1.0,
        min_df_unigram=1,
        min_df_phrase=1,
        phrase_min_count_per_cluster=1,
        top_n_keywords=5,
        scoring_pool_factor=2.0,
        ngram_min=1,
        ngram_max=3,
        use_phrase_vectorizer=True,
        cross_cluster_penalty_enabled=True,
        quality_diagnostics_enabled=True,
        quality_rerank_enabled=True,
        n_jobs=1,
    )

    keywords = run_keyword_pipeline(cfg)

    assert not keywords.empty
    for column in ("raw_term", "normalized_term", "display_label", "quality_score", "quality_flags"):
        assert column in keywords.columns
    assert keywords["quality_score"].notna().all()
