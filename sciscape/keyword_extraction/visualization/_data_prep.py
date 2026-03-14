"""Data preparation helpers for visualization."""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


def _parse_json_col(series: pd.Series) -> pd.Series:
    """Parse a JSON-encoded string column into dicts."""
    def _parse(v):
        if isinstance(v, dict):
            return v
        if isinstance(v, str):
            try:
                return json.loads(v)
            except (json.JSONDecodeError, ValueError):
                return {}
        return {}
    return series.apply(_parse)


def _build_cluster_labels(df: pd.DataFrame, n: int = 3) -> Dict[int, str]:
    labels = {}
    for cid, grp in df.groupby("cluster_id"):
        top = grp.nlargest(n, "score")["term"].tolist()
        labels[int(cid)] = ", ".join(top[:n])
    return labels


def _compute_network_edges(
    terms: pd.DataFrame,
    min_weight: float = 0.1,
) -> List[Dict]:
    """Compute co-occurrence network edges from token overlap + subphrase."""
    term_list = terms["term"].tolist()
    scores = dict(zip(terms["term"], terms["score"]))
    edges = []
    seen: Set[Tuple[str, str]] = set()

    for i, t1 in enumerate(term_list):
        w1 = set(t1.split())
        for t2 in term_list[i + 1:]:
            w2 = set(t2.split())
            inter = w1 & w2
            union = w1 | w2
            if not inter:
                continue
            jaccard = len(inter) / len(union)
            containment = 0.0
            if w1 < w2 or w2 < w1:
                containment = 0.3
            weight = jaccard + containment
            if weight >= min_weight:
                key = tuple(sorted([t1, t2]))
                if key not in seen:
                    seen.add(key)
                    edges.append({
                        "source": t1,
                        "target": t2,
                        "weight": round(weight, 3),
                    })
    return edges


def prepare_cluster_data(
    df: pd.DataFrame,
    viz_data: Optional[Dict] = None,
    max_edges_per_cluster: int = 80,
) -> Dict:
    """Prepare all data for the dashboard as a JSON-serializable dict."""
    labels = _build_cluster_labels(df)
    clusters = {}

    cooc_edges_all = viz_data.get("cooc_edges", []) if viz_data else []
    subphrase_tree_all = viz_data.get("subphrase_tree", []) if viz_data else []
    vocab_merges = viz_data.get("vocab_merges", {}) if viz_data else {}
    norm_merges = viz_data.get("norm_merges", {}) if viz_data else {}

    subphrase_by_cluster: Dict[int, List[Dict]] = {}
    for entry in subphrase_tree_all:
        cid_sp = int(entry["cluster_id"])
        subphrase_by_cluster.setdefault(cid_sp, []).append(entry)

    _TEMPORAL_METRICS = ["pub_year_series", "ppm_series", "loglift_series"]

    for cid, grp in df.groupby("cluster_id"):
        cid = int(cid)
        grp_sorted = grp.sort_values("score", ascending=False)
        cluster_terms = set(grp_sorted["term"].tolist())

        keywords = []
        for _, r in grp_sorted.iterrows():
            kw = {
                "term": r["term"],
                "score": round(float(r["score"]), 6),
                "frequency": int(r["frequency"]),
                "doc_coverage": int(r.get("doc_coverage", r["frequency"])),
            }

            if "depth_level" in r.index and pd.notna(r["depth_level"]):
                kw["depth_level"] = int(r["depth_level"])
                kw["depth_score"] = round(float(r["depth_score"]), 4)

            for metric in _TEMPORAL_METRICS:
                if metric in r.index:
                    val = r[metric]
                    if isinstance(val, str):
                        try:
                            val = json.loads(val)
                        except (json.JSONDecodeError, ValueError):
                            val = {}
                    if isinstance(val, dict) and val:
                        kw[metric] = {str(k): v for k, v in val.items()}

            if "pub_year_series" in kw:
                kw["temporal"] = kw["pub_year_series"]

            if "cross_cluster_count" in r.index and pd.notna(r["cross_cluster_count"]):
                kw["cross_cluster_count"] = int(r["cross_cluster_count"])

            if "expanded_from" in r.index and pd.notna(r["expanded_from"]):
                ef = r["expanded_from"]
                if isinstance(ef, str) and ef.strip():
                    kw["expanded_from"] = ef

            if "source_terms" in r.index:
                st = r["source_terms"]
                if isinstance(st, np.ndarray):
                    st = st.tolist()
                elif isinstance(st, str):
                    try:
                        st = json.loads(st)
                    except (json.JSONDecodeError, ValueError):
                        st = []
                if isinstance(st, list) and len(st) > 1:
                    kw["source_terms"] = st

            keywords.append(kw)

        if cooc_edges_all:
            edges = [
                e for e in cooc_edges_all
                if e["source"] in cluster_terms and e["target"] in cluster_terms
            ]
            edges = edges[:max_edges_per_cluster]
        else:
            edges = _compute_network_edges(grp_sorted)

        subphrases = subphrase_by_cluster.get(cid, [])

        cluster_norm_merges = {
            t: srcs for t, srcs in norm_merges.items() if t in cluster_terms
        }

        clusters[cid] = {
            "label": labels[cid],
            "n_keywords": len(keywords),
            "keywords": keywords,
            "network_edges": edges,
            "subphrase_tree": subphrases,
            "norm_merges": cluster_norm_merges,
        }

    trend_scores = viz_data.get("trend_scores", {}) if viz_data else {}
    centrality = viz_data.get("centrality", {}) if viz_data else {}
    cross_cluster_terms = viz_data.get("cross_cluster_terms", []) if viz_data else []
    pipeline_config = viz_data.get("pipeline_config", {}) if viz_data else {}

    global_data = {
        "_vocab_merges": vocab_merges,
        "_norm_merges": norm_merges,
        "_trend_scores": trend_scores,
        "_centrality": centrality,
        "_cross_cluster_terms": cross_cluster_terms,
        "_pipeline_config": pipeline_config,
    }

    return {**clusters, **global_data}
