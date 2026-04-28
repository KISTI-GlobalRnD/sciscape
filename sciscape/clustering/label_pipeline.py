"""Hierarchical cluster labeling with cleansing.

Pipeline:
1. Keyword extraction per level (TF-IDF + scoring)
2. Label candidate generation (top-k keywords per cluster)
3. Label cleansing via string_grouper (TF-IDF cosine dedup)
4. Hierarchy consistency (nano labels should specialize micro labels)
5. Final label assignment
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import polars as pl

log = logging.getLogger(__name__)


def extract_cluster_labels(
    abstracts: pl.DataFrame,
    membership: pl.DataFrame | Dict[str, np.ndarray],
    *,
    level: str = "nano",
    top_k: int = 5,
    strategy: str = "tfidf_distinct",
    min_df: int = 3,
) -> pl.DataFrame:
    """Extract keyword-based labels for clusters at one level.

    Parameters
    ----------
    abstracts : pl.DataFrame
        uid, title, abstract columns.
    membership : pl.DataFrame or dict
        If DataFrame: uid + cluster_{level} columns.
        If dict: {level_name: np.ndarray}.
    level : str
        Which level to label.
    top_k : int
        Keywords per cluster for labeling.

    Returns
    -------
    pl.DataFrame
        cluster, label, keywords (list), n_papers.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Resolve membership
    if isinstance(membership, dict):
        mem_arr = membership[level]
        uids = None  # assume same order as abstracts
    else:
        cluster_col = f"cluster_{level}"
        if cluster_col not in membership.columns:
            raise ValueError(f"Column {cluster_col} not in membership")
        joined = abstracts.join(membership.select("uid", cluster_col), on="uid", how="inner")
        mem_arr = joined[cluster_col].to_numpy()
        abstracts = joined

    # Build per-cluster text corpus
    texts = abstracts["title"].to_list()
    if "abstract" in abstracts.columns:
        abs_texts = abstracts["abstract"].to_list()
        texts = [f"{t or ''} {a or ''}" for t, a in zip(texts, abs_texts)]

    cluster_ids = sorted(set(int(c) for c in mem_arr))
    cluster_texts: Dict[int, List[str]] = defaultdict(list)
    for i, cid in enumerate(mem_arr):
        cluster_texts[int(cid)].append(texts[i] if i < len(texts) else "")

    # TF-IDF per cluster (concatenate all docs)
    cluster_docs = {cid: " ".join(txts) for cid, txts in cluster_texts.items()}
    cids_ordered = sorted(cluster_docs.keys())
    docs = [cluster_docs[cid] for cid in cids_ordered]

    effective_min_df = max(1, min(min_df, len(docs) - 1)) if len(docs) > 1 else 1
    use_stop = "english" if len(docs) > 10 else None
    try:
        vectorizer = TfidfVectorizer(
            max_features=5000, min_df=effective_min_df,
            stop_words=use_stop, ngram_range=(1, 3),
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z-]{2,}\b",
        )
        tfidf = vectorizer.fit_transform(docs)
        feature_names = vectorizer.get_feature_names_out()
    except ValueError:
        # Fallback: no IDF, just count
        from sklearn.feature_extraction.text import CountVectorizer
        vectorizer = CountVectorizer(
            max_features=1000, min_df=1,
            stop_words=use_stop, ngram_range=(1, 3),
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z-]{2,}\b",
        )
        tfidf = vectorizer.fit_transform(docs)
        feature_names = vectorizer.get_feature_names_out()

    # Top-k keywords per cluster
    rows = []
    for idx, cid in enumerate(cids_ordered):
        scores = tfidf[idx].toarray().ravel()
        top_indices = np.argsort(-scores)[:top_k]
        keywords = [feature_names[i] for i in top_indices if scores[i] > 0]
        label = ", ".join(keywords[:3]) if keywords else f"Cluster {cid}"
        rows.append({
            "cluster": cid,
            "label": label,
            "keywords": keywords,
            "n_papers": len(cluster_texts[cid]),
        })

    return pl.DataFrame(rows)


def suggest_merges(
    labels_df: pl.DataFrame,
    *,
    min_similarity: float = 0.5,
) -> List[Dict[str, Any]]:
    """Suggest label merges based on string similarity.

    Returns a list of merge candidates, each:
    {
        "source": label_a,
        "target": label_b (representative),
        "similarity": cosine score,
        "source_cluster": cluster_id,
        "target_cluster": cluster_id,
        "source_papers": n,
        "target_papers": n,
    }

    The caller (UI or script) decides which to apply.
    """
    from string_grouper import match_strings
    import pandas as pd

    labels = labels_df["label"].to_list()
    clusters = labels_df["cluster"].to_list()
    n_papers = labels_df["n_papers"].to_list()

    if len(labels) < 2:
        return []

    series = pd.Series(labels)
    try:
        matches = match_strings(series, min_similarity=min_similarity)
    except Exception as e:
        log.warning("string_grouper failed: %s", e)
        return []

    candidates = []
    seen = set()
    for _, row in matches.iterrows():
        left, right = int(row["left_index"]), int(row["right_index"])
        if left == right:
            continue
        pair = (min(left, right), max(left, right))
        if pair in seen:
            continue
        seen.add(pair)
        sim = float(row["similarity"])

        # Source = smaller cluster, target = larger
        if n_papers[left] >= n_papers[right]:
            src, tgt = right, left
        else:
            src, tgt = left, right

        candidates.append({
            "source": labels[src],
            "target": labels[tgt],
            "similarity": round(sim, 3),
            "source_cluster": clusters[src],
            "target_cluster": clusters[tgt],
            "source_papers": n_papers[src],
            "target_papers": n_papers[tgt],
        })

    candidates.sort(key=lambda x: -x["similarity"])
    log.info("suggest_merges: %d candidates (threshold=%.2f)", len(candidates), min_similarity)
    return candidates


def apply_merges(
    labels_df: pl.DataFrame,
    merge_map: Dict[str, str],
) -> pl.DataFrame:
    """Apply confirmed label merges.

    Parameters
    ----------
    merge_map : dict
        {source_label: target_label} — source gets renamed to target.
    """
    labels = labels_df["label"].to_list()
    new_labels = [merge_map.get(l, l) for l in labels]
    n_changed = sum(1 for a, b in zip(labels, new_labels) if a != b)
    if n_changed:
        log.info("apply_merges: %d labels renamed", n_changed)
    return labels_df.with_columns(pl.Series("label", new_labels))


def cleanse_labels(
    labels_df: pl.DataFrame,
    *,
    min_similarity: float = 0.5,
) -> pl.DataFrame:
    """Deduplicate similar labels using string_grouper.

    Groups similar labels and picks the most frequent as representative.

    Parameters
    ----------
    labels_df : pl.DataFrame
        cluster, label, keywords, n_papers.
    min_similarity : float
        TF-IDF cosine threshold for grouping (default 0.5).

    Returns
    -------
    pl.DataFrame
        Same schema with deduplicated labels.
    """
    from string_grouper import match_strings
    import pandas as pd

    labels = labels_df["label"].to_list()
    if len(labels) < 2:
        return labels_df

    # Match similar labels
    series = pd.Series(labels)
    try:
        matches = match_strings(series, min_similarity=min_similarity)
    except Exception as e:
        log.warning("string_grouper failed: %s, skipping dedup", e)
        return labels_df

    # Build groups: find connected components of similar labels
    groups: Dict[int, int] = {}  # label_idx → group_id
    group_id = 0
    for _, row in matches.iterrows():
        left = row["left_index"]
        right = row["right_index"]
        if left == right:
            continue
        gl = groups.get(left)
        gr = groups.get(right)
        if gl is None and gr is None:
            groups[left] = group_id
            groups[right] = group_id
            group_id += 1
        elif gl is not None and gr is None:
            groups[right] = gl
        elif gl is None and gr is not None:
            groups[left] = gr
        elif gl != gr:
            # Merge: remap all of gr to gl
            for k, v in groups.items():
                if v == gr:
                    groups[k] = gl

    # For each group, pick representative (largest n_papers)
    group_members: Dict[int, List[int]] = defaultdict(list)
    for idx, gid in groups.items():
        group_members[gid].append(idx)

    n_papers = labels_df["n_papers"].to_list()
    new_labels = list(labels)
    n_deduped = 0

    for gid, members in group_members.items():
        if len(members) < 2:
            continue
        # Pick the label with most papers
        best = max(members, key=lambda i: n_papers[i])
        rep_label = labels[best]
        for m in members:
            if m != best and new_labels[m] != rep_label:
                new_labels[m] = rep_label
                n_deduped += 1

    if n_deduped > 0:
        log.info("Label cleansing: %d labels deduplicated into representatives", n_deduped)

    return labels_df.with_columns(pl.Series("label", new_labels))


def ensure_hierarchy_consistency(
    level_labels: Dict[str, pl.DataFrame],
    hierarchy_df: pl.DataFrame,
    level_order: Sequence[str] = ("nano", "micro", "meso", "macro"),
) -> Dict[str, pl.DataFrame]:
    """Ensure child labels specialize parent labels.

    If a nano cluster's label is identical to its micro parent,
    append a distinguishing keyword.

    Parameters
    ----------
    level_labels : dict
        {level_name: labels_df} with cluster, label columns.
    hierarchy_df : pl.DataFrame
        uid + cluster_{level} columns for all levels.

    Returns
    -------
    dict of updated labels_df per level.
    """
    present = [l for l in level_order if l in level_labels and f"cluster_{l}" in hierarchy_df.columns]
    if len(present) < 2:
        return level_labels

    result = dict(level_labels)

    for i in range(len(present) - 1):
        child_level = present[i]
        parent_level = present[i + 1]

        child_col = f"cluster_{child_level}"
        parent_col = f"cluster_{parent_level}"

        # Map child cluster → parent cluster (majority vote)
        mapping = (
            hierarchy_df.select(child_col, parent_col)
            .group_by(child_col)
            .agg(pl.col(parent_col).mode().first().alias("parent"))
        )

        child_labels = dict(zip(
            result[child_level]["cluster"].to_list(),
            result[child_level]["label"].to_list(),
        ))
        parent_labels = dict(zip(
            result[parent_level]["cluster"].to_list(),
            result[parent_level]["label"].to_list(),
        ))

        # Check for duplicate labels
        updated_labels = dict(child_labels)
        n_fixed = 0
        for row in mapping.iter_rows(named=True):
            child_cid = row[child_col]
            parent_cid = row["parent"]
            child_label = child_labels.get(child_cid, "")
            parent_label = parent_labels.get(parent_cid, "")

            if child_label and parent_label and child_label == parent_label:
                # Append first unique keyword from child
                child_kws = result[child_level].filter(
                    pl.col("cluster") == child_cid
                )["keywords"].to_list()
                if child_kws and child_kws[0]:
                    extra = [kw for kw in child_kws[0] if kw not in parent_label]
                    if extra:
                        updated_labels[child_cid] = f"{child_label} ({extra[0]})"
                        n_fixed += 1

        if n_fixed > 0:
            log.info("Hierarchy consistency: %d %s labels specialized", n_fixed, child_level)
            new_labels = [updated_labels.get(c, l) for c, l in zip(
                result[child_level]["cluster"].to_list(),
                result[child_level]["label"].to_list(),
            )]
            result[child_level] = result[child_level].with_columns(pl.Series("label", new_labels))

    return result


def label_hierarchy(
    abstracts: pl.DataFrame,
    hierarchy_df: pl.DataFrame,
    *,
    levels: Sequence[str] = ("nano", "micro", "meso", "macro"),
    top_k: int = 5,
    cleanse: bool = True,
    min_similarity: float = 0.5,
    progress: callable | None = None,
) -> Dict[str, pl.DataFrame]:
    """Full labeling pipeline for hierarchical clustering.

    Returns {level_name: DataFrame(cluster, label, keywords, n_papers)}.
    """
    def _log(msg):
        log.info(msg)
        if progress:
            progress(msg)

    present = [l for l in levels if f"cluster_{l}" in hierarchy_df.columns]
    _log(f"Labeling {len(present)} levels: {present}")

    # Extract abbreviation dictionary from abstracts
    from .abbreviation_dict import extract_abbreviations, expand_labels_with_abbreviations
    abbr_dict = extract_abbreviations(abstracts, min_count=3)
    _log(f"Abbreviation dictionary: {len(abbr_dict)} entries")

    level_labels = {}
    for level in present:
        _log(f"  {level}: extracting keywords...")
        labels = extract_cluster_labels(
            abstracts, hierarchy_df, level=level, top_k=top_k,
        )
        if cleanse:
            _log(f"  {level}: cleansing labels...")
            labels = cleanse_labels(labels, min_similarity=min_similarity)
        # Expand abbreviations in labels
        if abbr_dict:
            expanded = expand_labels_with_abbreviations(
                labels["label"].to_list(), abbr_dict, mode="append",
            )
            labels = labels.with_columns(pl.Series("label", expanded))

        level_labels[level] = labels
        _log(f"  {level}: {labels.height} clusters labeled")

    if len(present) >= 2:
        _log("Ensuring hierarchy consistency...")
        level_labels = ensure_hierarchy_consistency(level_labels, hierarchy_df, present)

    return level_labels


__all__ = [
    "extract_cluster_labels",
    "suggest_merges",
    "apply_merges",
    "cleanse_labels",
    "ensure_hierarchy_consistency",
    "label_hierarchy",
]
