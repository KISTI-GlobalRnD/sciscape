#!/usr/bin/env python3
"""Pilot test: run full keyword extraction pipeline on KRISS sample data.

Uses Data/sample_abst.parquet (10,325 docs) and Data/sample_memb.parquet
from the sibling KRISS workspace.
"""

import logging
import sys
import time
from pathlib import Path

# Ensure repo root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sciscape.keyword_extraction import (
    KeywordExtractionConfig,
    run_keyword_pipeline,
    CORE_COLUMNS,
    TIER2_COLUMNS,
    TIER3_COLUMNS,
)
from sciscape.keyword_extraction.config import VocabMergeConfig
from sciscape.keyword_extraction.depth import DepthConfig
from sciscape.keyword_extraction.term_network import TermNetworkConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

KRISS_DIR = Path.home() / "Desktop/Workspace/1.4.2.KRISS"
ABS_PATH = KRISS_DIR / "Data" / "sample_abst.parquet"
MEM_PATH = KRISS_DIR / "Data" / "sample_memb.parquet"

assert ABS_PATH.exists(), f"Not found: {ABS_PATH}"
assert MEM_PATH.exists(), f"Not found: {MEM_PATH}"


def main():
    cfg = KeywordExtractionConfig(
        abstract_path=ABS_PATH,
        membership_path=MEM_PATH,
        cluster_level="cluster_meso",

        # Include titles
        include_title=True,
        title_weight=2.0,

        # Vectoriser tuning
        min_df_unigram=5,
        min_df_phrase=3,
        use_phrase_vectorizer=True,
        ngram_min=2,
        ngram_max=3,
        phrase_min_count_per_cluster=5,

        # Top-K
        top_n_unigrams=80,
        top_n_keywords=50,

        # Stage 2: vocab merge
        vocab_merge=VocabMergeConfig(
            enabled=True,
            plural_to_singular=True,
            hyphen_normalize=True,
            merge_frequency_ratio=0.01,
        ),

        # Stage 5: normalization + P2 plural merge
        normalization_enabled=True,
        norm_max_edit_distance=2,
        norm_min_frequency_ratio=0.01,
        norm_plural_merge_enabled=True,

        # Stage 6-7: cooccurrence + term network
        cooccurrence_enabled=True,
        cooccurrence_min_count=3,
        term_network=TermNetworkConfig(
            enabled=True,
            layers=["string", "token", "cooccurrence"],
            merge_threshold=0.5,
            min_token_overlap=0.5,
        ),

        # Stage 9: depth estimation
        depth=DepthConfig(
            enabled=True,
            n_levels=3,
        ),

        # P1 + P5: academic stopwords + artifact filter (enabled by default)
        academic_stopwords_enabled=True,
        artifact_filter_enabled=True,

        # P6: cross-cluster penalty
        cross_cluster_penalty_enabled=True,
        cross_cluster_penalty_min_count=2,

        # P4: short-term abbreviation expansion
        short_term_expansion_enabled=True,
        short_term_max_length=2,
        short_term_min_cooc_ratio=0.05,
        short_term_expansion_mode="annotate",

        # P3: auto-merge high-confidence pairs from term network
        auto_merge_enabled=True,
        auto_merge_min_similarity=0.85,

        # Execution
        n_jobs=4,
        verbose=True,
    )

    print(f"\n{'='*60}")
    print(f"  KRISS Pilot: {ABS_PATH.name} + {MEM_PATH.name}")
    print(f"  cluster_level = cluster_meso (5 clusters, 10,325 docs)")
    print(f"  Stages: vectorize → vocab_merge → aggregate → score")
    print(f"          → normalize → cooccurrence → term_network")
    print(f"          → depth → temporal")
    print(f"{'='*60}\n")

    t0 = time.perf_counter()
    result = run_keyword_pipeline(cfg)
    elapsed = time.perf_counter() - t0

    print(f"\n{'='*60}")
    print(f"  Pipeline finished in {elapsed:.1f}s")
    print(f"  Result shape: {result.shape}")
    print(f"{'='*60}\n")

    # Column presence check
    present = set(result.columns)
    print("Column check:")
    for tier_name, tier_cols in [("CORE", CORE_COLUMNS), ("TIER2", TIER2_COLUMNS), ("TIER3", TIER3_COLUMNS)]:
        for col in tier_cols:
            status = "OK" if col in present else "MISSING"
            print(f"  [{status}] {tier_name}: {col}")

    # Per-cluster summary
    print(f"\nPer-cluster keyword counts:")
    for cid, grp in result.groupby("cluster_id"):
        top5 = grp.nlargest(5, "score")["term"].tolist()
        print(f"  cluster {cid}: {len(grp)} keywords — top 5: {top5}")

    # Temporal check
    if "pub_year_series" in result.columns:
        sample_series = result["pub_year_series"].iloc[0]
        print(f"\nTemporal sample (first keyword): {sample_series}")

    # Depth check
    if "depth_score" in result.columns:
        depth_stats = result["depth_score"].describe()
        print(f"\nDepth score stats:\n{depth_stats}")

    # Save output (JSON-encode dict columns for pyarrow compatibility)
    import json
    out_path = Path(__file__).resolve().parent.parent / "pilot_output.parquet"
    save_df = result.copy()
    dict_cols = ("pub_year_series", "year_denominators", "ppm_series",
                 "loglift_series", "bayesian_log_odds_series")
    for col in dict_cols:
        if col in save_df.columns:
            save_df[col] = save_df[col].apply(
                lambda v: json.dumps(v) if isinstance(v, (dict, list)) else v
            )
    save_df.to_parquet(out_path, index=False)
    print(f"\nSaved to: {out_path}")

    return result


if __name__ == "__main__":
    main()
