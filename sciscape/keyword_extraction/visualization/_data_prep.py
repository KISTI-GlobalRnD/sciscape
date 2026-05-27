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


def _keyword_label_col(df: pd.DataFrame) -> str:
    return "display_label" if "display_label" in df.columns else "term"


def _keyword_score_col(df: pd.DataFrame) -> str:
    return "quality_score" if "quality_score" in df.columns else "score"


def _build_cluster_labels(df: pd.DataFrame, n: int = 3) -> Dict[int, str]:
    labels = {}
    label_col = _keyword_label_col(df)
    score_col = _keyword_score_col(df)
    for cid, grp in df.groupby("cluster_id"):
        top = []
        for label in grp.nlargest(max(n * 2, n), score_col)[label_col].astype(str).tolist():
            if label not in top:
                top.append(label)
            if len(top) >= n:
                break
        labels[int(cid)] = ", ".join(top[:n])
    return labels


def _compute_network_edges(
    terms: pd.DataFrame,
    min_weight: float = 0.1,
) -> List[Dict]:
    """Compute co-occurrence network edges from token overlap + subphrase."""
    term_list = terms["term"].tolist()
    dict(zip(terms["term"], terms["score"]))
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
    label_col = _keyword_label_col(df)
    score_col = _keyword_score_col(df)

    for cid, grp in df.groupby("cluster_id"):
        cid = int(cid)
        grp_sorted = grp.sort_values(score_col, ascending=False)
        cluster_terms = set(grp_sorted["term"].tolist())
        raw_to_label = dict(zip(grp_sorted["term"].astype(str), grp_sorted[label_col].astype(str)))

        keywords = []
        for _, r in grp_sorted.iterrows():
            raw_term = str(r["term"])
            label = str(r[label_col])
            kw = {
                "term": label,
                "raw_term": raw_term,
                "display_label": label,
                "score": round(float(r[score_col]), 6),
                "frequency": int(r["frequency"]),
                "doc_coverage": int(r.get("doc_coverage", r["frequency"])),
            }
            if score_col != "score" and "score" in r.index:
                kw["raw_score"] = round(float(r["score"]), 6)
            if "quality_flags" in r.index and pd.notna(r["quality_flags"]):
                kw["quality_flags"] = str(r["quality_flags"])
            if "quality_multiplier" in r.index and pd.notna(r["quality_multiplier"]):
                kw["quality_multiplier"] = round(float(r["quality_multiplier"]), 6)

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
                {
                    **e,
                    "source": raw_to_label.get(str(e["source"]), str(e["source"])),
                    "target": raw_to_label.get(str(e["target"]), str(e["target"])),
                }
                for e in cooc_edges_all
                if e["source"] in cluster_terms and e["target"] in cluster_terms
            ]
            edges = edges[:max_edges_per_cluster]
        else:
            edge_terms = grp_sorted.copy()
            edge_terms["term"] = edge_terms[label_col].astype(str)
            edges = _compute_network_edges(edge_terms)

        subphrases = subphrase_by_cluster.get(cid, [])

        cluster_norm_merges = {}
        for t, srcs in norm_merges.items():
            if t not in cluster_terms:
                continue
            cluster_norm_merges[raw_to_label.get(str(t), str(t))] = [
                raw_to_label.get(str(src), str(src)) for src in srcs
            ]

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
    if label_col != "term":
        raw_to_label_all = dict(zip(df["term"].astype(str), df[label_col].astype(str)))
        trend_scores = dict(trend_scores)
        centrality = dict(centrality)
        for raw, label in raw_to_label_all.items():
            if raw in trend_scores and label not in trend_scores:
                trend_scores[label] = trend_scores[raw]
            if raw in centrality and label not in centrality:
                centrality[label] = centrality[raw]

    global_data = {
        "_vocab_merges": vocab_merges,
        "_norm_merges": norm_merges,
        "_trend_scores": trend_scores,
        "_centrality": centrality,
        "_cross_cluster_terms": cross_cluster_terms,
        "_pipeline_config": pipeline_config,
    }

    return {**clusters, **global_data}
