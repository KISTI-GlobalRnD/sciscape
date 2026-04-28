"""Automatic cluster labeling from keywords (no LLM required).

Generates concise cluster names from top-scored keywords.
Falls back to "Cluster N" when no keywords are available.

Strategies:
  - "top_keywords": join top-k keywords (e.g. "machine learning, neural networks")
  - "longest_first": pick the most specific (longest) keyword as primary name
  - "tfidf_distinct": pick keywords that best distinguish this cluster from others
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np
import polars as pl

log = logging.getLogger(__name__)


def auto_label_clusters(
    keywords_df: pl.DataFrame,
    *,
    strategy: str = "top_keywords",
    top_k: int = 3,
    separator: str = ", ",
    cluster_col: str | None = None,
    keyword_col: str | None = None,
    score_col: str | None = None,
) -> Dict[int, str]:
    """Generate cluster labels from a keywords DataFrame.

    Parameters
    ----------
    keywords_df : pl.DataFrame
        Keywords table with cluster, keyword, and score columns.
    strategy : str
        Labeling strategy: "top_keywords", "longest_first", "tfidf_distinct".
    top_k : int
        Number of keywords to use per cluster.

    Returns
    -------
    dict
        Mapping from cluster ID to label string.
    """
    # Auto-detect column names
    cols = keywords_df.columns
    if cluster_col is None:
        cluster_col = next((c for c in cols if "cluster" in c.lower()), cols[0])
    if keyword_col is None:
        keyword_col = next(
            (c for c in cols if any(k in c.lower() for k in ("keyword", "term", "word"))),
            cols[1] if len(cols) > 1 else cols[0],
        )
    if score_col is None:
        score_col = next(
            (c for c in cols if any(k in c.lower() for k in ("score", "weight", "tfidf", "rank"))),
            cols[2] if len(cols) > 2 else None,
        )

    # Group keywords per cluster, sorted by score
    cluster_keywords: Dict[int, List[tuple[str, float]]] = defaultdict(list)
    for row in keywords_df.iter_rows(named=True):
        cid = int(row[cluster_col])
        kw = str(row[keyword_col])
        score = float(row[score_col]) if score_col and row.get(score_col) is not None else 0.0
        cluster_keywords[cid].append((kw, score))

    # Sort by score descending within each cluster
    for cid in cluster_keywords:
        cluster_keywords[cid].sort(key=lambda x: -x[1])

    labels: Dict[int, str] = {}

    if strategy == "top_keywords":
        for cid, kws in cluster_keywords.items():
            top = [kw for kw, _ in kws[:top_k]]
            labels[cid] = separator.join(top) if top else f"Cluster {cid}"

    elif strategy == "longest_first":
        for cid, kws in cluster_keywords.items():
            if not kws:
                labels[cid] = f"Cluster {cid}"
                continue
            # Primary: longest keyword (most specific)
            primary = max(kws[:top_k * 2], key=lambda x: len(x[0]))[0]
            # Secondary: top-scored keywords excluding primary
            secondary = [kw for kw, _ in kws if kw != primary][:top_k - 1]
            parts = [primary] + secondary
            labels[cid] = separator.join(parts)

    elif strategy == "tfidf_distinct":
        # Pick keywords that are most unique to each cluster
        # (appear in fewest other clusters)
        keyword_cluster_count: Counter = Counter()
        for cid, kws in cluster_keywords.items():
            seen = set()
            for kw, _ in kws[:top_k * 3]:
                if kw not in seen:
                    keyword_cluster_count[kw] += 1
                    seen.add(kw)

        n_clusters = len(cluster_keywords)
        for cid, kws in cluster_keywords.items():
            if not kws:
                labels[cid] = f"Cluster {cid}"
                continue
            # Score = original_score / log(1 + cluster_frequency)
            scored = []
            for kw, s in kws[:top_k * 3]:
                cf = keyword_cluster_count.get(kw, 1)
                distinctiveness = s / np.log1p(cf)
                scored.append((kw, distinctiveness))
            scored.sort(key=lambda x: -x[1])
            top = [kw for kw, _ in scored[:top_k]]
            labels[cid] = separator.join(top) if top else f"Cluster {cid}"

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    log.info("auto_label: %d clusters labeled (strategy=%s, top_k=%d)",
             len(labels), strategy, top_k)
    return labels


def labels_to_dataframe(labels: Dict[int, str]) -> pl.DataFrame:
    """Convert label dict to DataFrame."""
    return pl.DataFrame({
        "cluster": list(labels.keys()),
        "label": list(labels.values()),
    }).sort("cluster")


__all__ = ["auto_label_clusters", "labels_to_dataframe"]
