import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from leiden_module.keyword_extraction import KeywordExtractionConfig, run_keyword_pipeline


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
