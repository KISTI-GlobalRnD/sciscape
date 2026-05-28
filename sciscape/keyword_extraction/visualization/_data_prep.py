"""Data preparation helpers for visualization."""

from __future__ import annotations

import json
import re
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
    if "representative_score" in df.columns:
        return "representative_score"
    return "quality_score" if "quality_score" in df.columns else "score"


_LABEL_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _label_tokens(label: object) -> set[str]:
    return set(_LABEL_TOKEN_RE.findall(str(label).lower()))


def _label_similarity(left: object, right: object) -> float:
    left_tokens = _label_tokens(left)
    right_tokens = _label_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    if left_tokens == right_tokens:
        return 1.0
    overlap = len(left_tokens & right_tokens)
    if overlap == 0:
        return 0.0
    jaccard = overlap / len(left_tokens | right_tokens)
    containment = overlap / min(len(left_tokens), len(right_tokens))
    return min(1.0, max(jaccard, containment * 0.85))


def _is_containment_duplicate(label: str, selected: List[str]) -> bool:
    label_tokens = _label_tokens(label)
    if not label_tokens:
        return False
    for other in selected:
        other_tokens = _label_tokens(other)
        if not other_tokens:
            continue
        if label_tokens < other_tokens or other_tokens < label_tokens:
            return True
    return False


def _select_diverse_labels(
    label_scores: List[Tuple[str, float]],
    *,
    n: int,
    relevance_weight: float = 0.72,
) -> List[str]:
    """Select readable cluster labels with light MMR-style de-duplication."""

    if n <= 0:
        return []

    deduped: list[tuple[str, float]] = []
    seen: set[str] = set()
    for label, score in label_scores:
        label = str(label).strip()
        if not label or label in seen:
            continue
        seen.add(label)
        deduped.append((label, float(score)))
    if len(deduped) <= n:
        return [label for label, _ in deduped[:n]]

    max_score = max((score for _, score in deduped), default=0.0)
    if max_score <= 0:
        max_score = 1.0

    selected: list[str] = []
    remaining = deduped.copy()
    while remaining and len(selected) < n:
        candidate_pool = [
            (index, label, score)
            for index, (label, score) in enumerate(remaining)
            if not _is_containment_duplicate(label, selected)
        ]
        if not candidate_pool:
            candidate_pool = [
                (index, label, score)
                for index, (label, score) in enumerate(remaining)
            ]
        best_index = 0
        best_value = float("-inf")
        for index, label, score in candidate_pool:
            relevance = score / max_score
            redundancy = max((_label_similarity(label, other) for other in selected), default=0.0)
            value = relevance_weight * relevance - (1.0 - relevance_weight) * redundancy
            if value > best_value:
                best_index = index
                best_value = value
        selected.append(remaining.pop(best_index)[0])

    return selected


def _build_cluster_labels(df: pd.DataFrame, n: int = 3) -> Dict[int, str]:
    labels = {}
    label_col = _keyword_label_col(df)
    score_col = _keyword_score_col(df)
    for cid, grp in df.groupby("cluster_id"):
        candidate_rows = grp.nlargest(max(n * 6, n), score_col)
        label_scores = [
            (str(row[label_col]), float(row[score_col]))
            for _, row in candidate_rows.iterrows()
        ]
        labels[int(cid)] = ", ".join(_select_diverse_labels(label_scores, n=n))
    return labels


def _group_keywords_by_scope(keywords: List[Dict], *, max_per_scope: int = 25) -> Dict[str, List[Dict]]:
    groups = {"cluster_specific": [], "shared": [], "common": []}
    for kw in keywords:
        scope = str(kw.get("keyword_scope") or "cluster_specific")
        if scope not in groups:
            scope = "cluster_specific"
        groups[scope].append(kw)
    return {scope: values[:max_per_scope] for scope, values in groups.items()}


def _aggregate_keyword_scope_terms(
    df: pd.DataFrame,
    *,
    label_col: str,
    score_col: str,
) -> Dict[str, List[Dict]]:
    groups = {"cluster_specific": [], "shared": [], "common": []}
    if df.empty or "cluster_id" not in df.columns or label_col not in df.columns:
        return groups

    n_clusters = max(1, int(df["cluster_id"].nunique()))
    for label, term_group in df.groupby(label_col, sort=False):
        clusters = sorted(int(cid) for cid in term_group["cluster_id"].dropna().unique().tolist())
        cluster_count = len(clusters)
        cluster_ratio = cluster_count / n_clusters
        scope_values = set()
        if "keyword_scope" in term_group.columns:
            scope_values = set(term_group["keyword_scope"].dropna().astype(str).tolist())
        if "common" in scope_values:
            scope = "common"
        elif "shared" in scope_values or cluster_count > 1:
            scope = "shared"
        else:
            scope = "cluster_specific"
        if "frequency" in term_group.columns:
            frequency = int(pd.to_numeric(term_group["frequency"], errors="coerce").fillna(0).sum())
        else:
            frequency = 0
        max_score = float(pd.to_numeric(term_group[score_col], errors="coerce").fillna(0).max())
        groups[scope].append(
            {
                "term": str(label),
                "cluster_count": int(cluster_count),
                "cluster_ratio": round(float(cluster_ratio), 6),
                "clusters": clusters,
                "score": round(max_score, 6),
                "frequency": frequency,
            }
        )

    for scope, values in groups.items():
        groups[scope] = sorted(
            values,
            key=lambda item: (
                -int(item["cluster_count"]),
                -float(item["score"]),
                -int(item["frequency"]),
                str(item["term"]),
            ),
        )
    return groups


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
        keywords_by_label: Dict[str, Dict] = {}
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
            if "quality_score" in r.index and pd.notna(r["quality_score"]):
                kw["quality_score"] = round(float(r["quality_score"]), 6)
            if "representative_score" in r.index and pd.notna(r["representative_score"]):
                kw["representative_score"] = round(float(r["representative_score"]), 6)
            if "representative_multiplier" in r.index and pd.notna(r["representative_multiplier"]):
                kw["representative_multiplier"] = round(float(r["representative_multiplier"]), 6)
            if "representative_rank" in r.index and pd.notna(r["representative_rank"]):
                kw["representative_rank"] = int(r["representative_rank"])
            if "representative_role" in r.index and pd.notna(r["representative_role"]):
                kw["representative_role"] = str(r["representative_role"])
            if "representative_flags" in r.index and pd.notna(r["representative_flags"]):
                representative_flags = str(r["representative_flags"])
                if representative_flags.strip():
                    kw["representative_flags"] = representative_flags
            if "quality_flags" in r.index and pd.notna(r["quality_flags"]):
                kw["quality_flags"] = str(r["quality_flags"])
            if "quality_multiplier" in r.index and pd.notna(r["quality_multiplier"]):
                kw["quality_multiplier"] = round(float(r["quality_multiplier"]), 6)
            if "network_role" in r.index and pd.notna(r["network_role"]):
                kw["network_role"] = str(r["network_role"])
            if "network_score" in r.index and pd.notna(r["network_score"]):
                kw["network_score"] = round(float(r["network_score"]), 6)
            if "network_flags" in r.index and pd.notna(r["network_flags"]):
                network_flags = str(r["network_flags"])
                if network_flags.strip():
                    kw["network_flags"] = network_flags
            if "keyword_scope" in r.index and pd.notna(r["keyword_scope"]):
                kw["keyword_scope"] = str(r["keyword_scope"])
            if "keyword_cluster_count" in r.index and pd.notna(r["keyword_cluster_count"]):
                kw["keyword_cluster_count"] = int(r["keyword_cluster_count"])
            if "keyword_cluster_ratio" in r.index and pd.notna(r["keyword_cluster_ratio"]):
                kw["keyword_cluster_ratio"] = round(float(r["keyword_cluster_ratio"]), 6)
            if "abbreviation_status" in r.index and pd.notna(r["abbreviation_status"]):
                kw["abbreviation_status"] = str(r["abbreviation_status"])
            if "abbreviation_target" in r.index and pd.notna(r["abbreviation_target"]):
                abbreviation_target = str(r["abbreviation_target"])
                if abbreviation_target.strip():
                    kw["abbreviation_target"] = abbreviation_target
            if "abbreviation_confidence" in r.index and pd.notna(r["abbreviation_confidence"]):
                kw["abbreviation_confidence"] = round(float(r["abbreviation_confidence"]), 6)
            if "abbreviation_source" in r.index and pd.notna(r["abbreviation_source"]):
                abbreviation_source = str(r["abbreviation_source"])
                if abbreviation_source.strip():
                    kw["abbreviation_source"] = abbreviation_source
            if "abbreviation_support_docs" in r.index and pd.notna(r["abbreviation_support_docs"]):
                kw["abbreviation_support_docs"] = int(r["abbreviation_support_docs"])
            if "abbreviation_cluster_support_docs" in r.index and pd.notna(r["abbreviation_cluster_support_docs"]):
                kw["abbreviation_cluster_support_docs"] = int(r["abbreviation_cluster_support_docs"])
            if "abbreviation_top_support_ratio" in r.index and pd.notna(r["abbreviation_top_support_ratio"]):
                kw["abbreviation_top_support_ratio"] = round(float(r["abbreviation_top_support_ratio"]), 6)
            if "abbreviation_ambiguity_type" in r.index and pd.notna(r["abbreviation_ambiguity_type"]):
                abbreviation_ambiguity_type = str(r["abbreviation_ambiguity_type"])
                if abbreviation_ambiguity_type.strip():
                    kw["abbreviation_ambiguity_type"] = abbreviation_ambiguity_type

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

            existing = keywords_by_label.get(label)
            if existing is not None:
                alias = {
                    "raw_term": raw_term,
                    "score": kw["score"],
                    "frequency": kw["frequency"],
                    "doc_coverage": kw["doc_coverage"],
                }
                for field in (
                    "abbreviation_status",
                    "abbreviation_target",
                    "abbreviation_confidence",
                    "abbreviation_source",
                    "abbreviation_ambiguity_type",
                    "quality_flags",
                    "representative_role",
                    "representative_flags",
                    "network_role",
                    "network_flags",
                ):
                    if field in kw:
                        alias[field] = kw[field]
                existing.setdefault("raw_aliases", []).append(alias)
                continue

            keywords_by_label[label] = kw
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
            "keyword_groups": _group_keywords_by_scope(keywords),
            "network_edges": edges,
            "subphrase_tree": subphrases,
            "norm_merges": cluster_norm_merges,
        }

    trend_scores = viz_data.get("trend_scores", {}) if viz_data else {}
    centrality = viz_data.get("centrality", {}) if viz_data else {}
    cross_cluster_terms = viz_data.get("cross_cluster_terms", []) if viz_data else []
    abbreviation_evidence = viz_data.get("abbreviation_evidence", []) if viz_data else []
    abbreviation_evidence_total = viz_data.get("abbreviation_evidence_total", len(abbreviation_evidence)) if viz_data else 0
    pipeline_config = viz_data.get("pipeline_config", {}) if viz_data else {}
    keyword_scope_terms = _aggregate_keyword_scope_terms(
        df,
        label_col=label_col,
        score_col=score_col,
    )
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
        "_abbreviation_evidence": abbreviation_evidence,
        "_abbreviation_evidence_total": abbreviation_evidence_total,
        "_common_keywords": keyword_scope_terms["common"],
        "_shared_keywords": keyword_scope_terms["shared"],
        "_pipeline_config": pipeline_config,
    }

    return {**clusters, **global_data}
