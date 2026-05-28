"""Diagnostics and before/after scoring for keyword extraction outputs.

This module is intentionally lightweight (no plotting deps). It produces a JSON-friendly
payload that downstream notebooks/HTML reports can visualise.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from collections import Counter
from dataclasses import dataclass, field
import json
import math
import random
from typing import Any, Optional

import pandas as pd


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _quantiles(series: pd.Series, qs: Sequence[float]) -> dict[str, float]:
    if series.empty:
        return {}
    out: dict[str, float] = {}
    # Drop NA while preserving numeric conversion.
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {}
    for q in qs:
        out[f"p{int(q * 100)}"] = float(values.quantile(q))
    return out


def _safe_year_series(value: object) -> dict[int, int]:
    if value is None or (isinstance(value, float) and math.isnan(value)):  # type: ignore[arg-type]
        return {}
    if isinstance(value, Mapping):
        out: dict[int, int] = {}
        for k, v in value.items():
            try:
                year = int(k)
            except (ValueError, TypeError):
                continue
            try:
                count = int(v)
            except (ValueError, TypeError):
                continue
            if count:
                out[year] = count
        return out
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {}
        return _safe_year_series(parsed)
    return {}


def _sample_cluster_ids(
    cluster_ids: Sequence[int],
    *,
    sample_clusters: int | None,
    seed: int,
) -> list[int]:
    unique = sorted(set(int(cid) for cid in cluster_ids))
    if sample_clusters is None or sample_clusters >= len(unique):
        return unique
    rng = random.Random(int(seed))
    return sorted(rng.sample(unique, k=max(1, int(sample_clusters))))


def _subphrase_redundancy_ratio(terms: Sequence[str]) -> tuple[int, int]:
    """Return (#redundant_terms, #unique_terms) based on substring containment."""
    cleaned: list[str] = []
    seen: set[str] = set()
    for term in terms:
        t = str(term).strip().lower()
        if not t or t in seen:
            continue
        seen.add(t)
        cleaned.append(t)
    if not cleaned:
        return 0, 0

    # Compare each term to longer terms only.
    ordered = sorted(cleaned, key=len, reverse=True)
    redundant = 0
    for idx, term in enumerate(ordered):
        for longer in ordered[:idx]:
            if term != longer and term in longer:
                redundant += 1
                break
    return redundant, len(ordered)


def _token_jaccard_redundancy_ratio(
    terms: Sequence[str],
    *,
    threshold: float,
) -> tuple[int, int]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for term in terms:
        t = str(term).strip().lower()
        if not t or t in seen:
            continue
        seen.add(t)
        cleaned.append(t)
    if not cleaned:
        return 0, 0

    tokens = [set(t.split()) for t in cleaned]
    n = len(tokens)
    redundant_flags = [False] * n

    for i in range(n):
        if redundant_flags[i]:
            continue
        a = tokens[i]
        if not a:
            continue
        for j in range(i + 1, n):
            b = tokens[j]
            if not b:
                continue
            inter = len(a & b)
            if inter == 0:
                continue
            union = len(a | b)
            sim = inter / union if union else 0.0
            if sim >= threshold:
                redundant_flags[i] = True
                redundant_flags[j] = True
    return sum(1 for f in redundant_flags if f), n


def _pipe_flag_set(value: object) -> set[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):  # type: ignore[arg-type]
        return set()
    return {flag.strip() for flag in str(value).split("|") if flag.strip()}


def _value_counts(series: pd.Series) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for raw in series.dropna().astype(str):
        value = raw.strip()
        if value:
            counts[value] += 1
    return dict(sorted(counts.items()))


def _ratio_from_mask(mask: Sequence[bool]) -> Optional[float]:
    if not mask:
        return None
    return float(sum(1 for value in mask if value) / len(mask))


def _family_compression_ratio(terms: Sequence[str], *, min_parent_tokens: int = 2) -> Optional[float]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for term in terms:
        label = str(term).strip().lower()
        if not label or label in seen:
            continue
        seen.add(label)
        cleaned.append(label)
    if not cleaned:
        return None

    token_sets = [set(term.split()) for term in cleaned]
    child_terms = 0
    for idx, tokens in enumerate(token_sets):
        if not tokens:
            continue
        for parent_idx, parent_tokens in enumerate(token_sets):
            if idx == parent_idx or len(parent_tokens) < min_parent_tokens:
                continue
            if len(parent_tokens) < len(tokens) and parent_tokens < tokens:
                child_terms += 1
                break
    return child_terms / len(cleaned)


def _representative_diversity_ratio(df: pd.DataFrame, *, top_n: int = 3) -> Optional[float]:
    if "cluster_id" not in df.columns:
        return None
    label_col = "display_label" if "display_label" in df.columns else "term"
    if label_col not in df.columns:
        return None
    if "representative_rank" in df.columns:
        sort_cols = ["cluster_id", "representative_rank"]
        ascending = [True, True]
    elif "representative_score" in df.columns:
        sort_cols = ["cluster_id", "representative_score"]
        ascending = [True, False]
    elif "score" in df.columns:
        sort_cols = ["cluster_id", "score"]
        ascending = [True, False]
    else:
        return None

    labels: list[str] = []
    for _, group in df.sort_values(sort_cols, ascending=ascending, kind="mergesort").groupby("cluster_id", sort=False):
        labels.extend(group[label_col].dropna().astype(str).head(top_n).tolist())
    if not labels:
        return None
    return float(len(set(labels)) / len(labels))


@dataclass(frozen=True)
class KeywordDiagnostics:
    n_rows: int
    n_clusters: int
    terms_per_cluster: dict[str, float]
    doc_coverage: dict[str, float]
    score: dict[str, float]
    years_per_term: dict[str, float]
    single_year_term_ratio: Optional[float]
    redundancy_subphrase_ratio: Optional[float]
    redundancy_token_jaccard_ratio: Optional[float]
    scope_counts: dict[str, int] = field(default_factory=dict)
    scope_ratios: dict[str, float] = field(default_factory=dict)
    abbreviation_status_counts: dict[str, int] = field(default_factory=dict)
    unresolved_short_form_ratio: Optional[float] = None
    review_flag_ratio: Optional[float] = None
    representative_role_counts: dict[str, int] = field(default_factory=dict)
    representative_diversity_ratio: Optional[float] = None
    family_compression_ratio: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_rows": int(self.n_rows),
            "n_clusters": int(self.n_clusters),
            "terms_per_cluster": dict(self.terms_per_cluster),
            "doc_coverage": dict(self.doc_coverage),
            "score": dict(self.score),
            "years_per_term": dict(self.years_per_term),
            "single_year_term_ratio": None if self.single_year_term_ratio is None else float(self.single_year_term_ratio),
            "redundancy_subphrase_ratio": None if self.redundancy_subphrase_ratio is None else float(self.redundancy_subphrase_ratio),
            "redundancy_token_jaccard_ratio": None if self.redundancy_token_jaccard_ratio is None else float(self.redundancy_token_jaccard_ratio),
            "scope_counts": dict(self.scope_counts),
            "scope_ratios": dict(self.scope_ratios),
            "abbreviation_status_counts": dict(self.abbreviation_status_counts),
            "unresolved_short_form_ratio": None if self.unresolved_short_form_ratio is None else float(self.unresolved_short_form_ratio),
            "review_flag_ratio": None if self.review_flag_ratio is None else float(self.review_flag_ratio),
            "representative_role_counts": dict(self.representative_role_counts),
            "representative_diversity_ratio": None if self.representative_diversity_ratio is None else float(self.representative_diversity_ratio),
            "family_compression_ratio": None if self.family_compression_ratio is None else float(self.family_compression_ratio),
        }


def keyword_diagnostics(
    df: pd.DataFrame,
    *,
    sample_clusters: int | None = 50,
    seed: int = 0,
    token_jaccard_threshold: float = 0.8,
) -> KeywordDiagnostics:
    """Compute lightweight diagnostics for a keyword dataframe.

    Expected columns (best-effort):
    - cluster_id (int)
    - term (str)
    - score (float)
    - doc_coverage (int)
    - pub_year_series (dict[int,int] or JSON string)
    """

    if df is None or df.empty:
        return KeywordDiagnostics(
            n_rows=0,
            n_clusters=0,
            terms_per_cluster={},
            doc_coverage={},
            score={},
            years_per_term={},
            single_year_term_ratio=None,
            redundancy_subphrase_ratio=None,
            redundancy_token_jaccard_ratio=None,
        )

    out_df = df.copy()
    n_rows = int(len(out_df))

    if "cluster_id" in out_df.columns:
        cluster_ids = pd.to_numeric(out_df["cluster_id"], errors="coerce").dropna().astype(int)
        n_clusters = int(cluster_ids.nunique())
    else:
        cluster_ids = pd.Series([], dtype=int)
        n_clusters = 0

    terms_per_cluster: dict[str, float] = {}
    if "cluster_id" in out_df.columns:
        counts = out_df.groupby("cluster_id").size()
        terms_per_cluster = _quantiles(counts, qs=(0.1, 0.5, 0.9))
        if counts.size:
            terms_per_cluster["mean"] = float(counts.mean())

    doc_coverage: dict[str, float] = {}
    if "doc_coverage" in out_df.columns:
        doc_coverage = _quantiles(out_df["doc_coverage"], qs=(0.1, 0.5, 0.9))
        values = pd.to_numeric(out_df["doc_coverage"], errors="coerce").dropna()
        if not values.empty:
            doc_coverage["mean"] = float(values.mean())

    score: dict[str, float] = {}
    if "score" in out_df.columns:
        score = _quantiles(out_df["score"], qs=(0.1, 0.5, 0.9))
        values = pd.to_numeric(out_df["score"], errors="coerce").dropna()
        if not values.empty:
            score["mean"] = float(values.mean())

    scope_counts: dict[str, int] = {}
    scope_ratios: dict[str, float] = {}
    if "keyword_scope" in out_df.columns:
        scope_counts = _value_counts(out_df["keyword_scope"])
        denom = max(1, sum(scope_counts.values()))
        scope_ratios = {scope: count / denom for scope, count in scope_counts.items()}

    abbreviation_status_counts: dict[str, int] = {}
    unresolved_short_form_ratio: Optional[float] = None
    if "abbreviation_status" in out_df.columns:
        abbreviation_status_counts = _value_counts(out_df["abbreviation_status"])
        unresolved_short_form_ratio = _ratio_from_mask([
            status in {"candidate_short_form", "unlinked_short_form"}
            for status in out_df["abbreviation_status"].fillna("").astype(str).tolist()
        ])

    review_flag_ratio: Optional[float] = None
    if "quality_flags" in out_df.columns:
        review_flags = {
            "ambiguous_expansion",
            "artifact_formula",
            "artifact_like",
            "candidate_short_form",
            "low_support_expansion",
            "unlinked_short_form",
        }
        review_flag_ratio = _ratio_from_mask([
            bool(_pipe_flag_set(value) & review_flags)
            for value in out_df["quality_flags"].tolist()
        ])

    representative_role_counts: dict[str, int] = {}
    if "representative_role" in out_df.columns:
        representative_role_counts = _value_counts(out_df["representative_role"])
    representative_diversity = _representative_diversity_ratio(out_df)

    # Year-series sparsity
    years_per_term: dict[str, float] = {}
    single_year_ratio: Optional[float] = None
    if "pub_year_series" in out_df.columns:
        ys = [_safe_year_series(v) for v in out_df["pub_year_series"].tolist()]
        years_nonzero = pd.Series([len(v) for v in ys], dtype=float)
        years_per_term = _quantiles(years_nonzero, qs=(0.1, 0.5, 0.9))
        if not years_nonzero.empty:
            years_per_term["mean"] = float(years_nonzero.mean())
            single_year_ratio = float((years_nonzero <= 1).mean())

    # Redundancy metrics (sample clusters to keep it cheap on large runs)
    redundancy_subphrase_ratio: Optional[float] = None
    redundancy_token_ratio: Optional[float] = None
    if "cluster_id" in out_df.columns and "term" in out_df.columns and n_clusters > 0:
        sampled = _sample_cluster_ids(cluster_ids.tolist(), sample_clusters=sample_clusters, seed=seed)
        subphrase_red_total = 0
        token_red_total = 0
        term_total = 0
        family_sum = 0.0
        family_count = 0

        for cid in sampled:
            group = out_df[out_df["cluster_id"].astype(int) == int(cid)]
            terms = group["term"].astype(str).tolist()
            sub_red, sub_total = _subphrase_redundancy_ratio(terms)
            tok_red, tok_total = _token_jaccard_redundancy_ratio(terms, threshold=float(token_jaccard_threshold))
            family_ratio = _family_compression_ratio(terms)

            # Use the token total as the denominator for both (same unique term count).
            subphrase_red_total += int(sub_red)
            token_red_total += int(tok_red)
            term_total += int(tok_total)
            if family_ratio is not None:
                family_sum += float(family_ratio)
                family_count += 1

        if term_total > 0:
            redundancy_subphrase_ratio = subphrase_red_total / term_total
            redundancy_token_ratio = token_red_total / term_total
        family_compression = family_sum / family_count if family_count else None
    else:
        family_compression = None

    return KeywordDiagnostics(
        n_rows=n_rows,
        n_clusters=n_clusters,
        terms_per_cluster=terms_per_cluster,
        doc_coverage=doc_coverage,
        score=score,
        years_per_term=years_per_term,
        single_year_term_ratio=single_year_ratio,
        redundancy_subphrase_ratio=redundancy_subphrase_ratio,
        redundancy_token_jaccard_ratio=redundancy_token_ratio,
        scope_counts=scope_counts,
        scope_ratios=scope_ratios,
        abbreviation_status_counts=abbreviation_status_counts,
        unresolved_short_form_ratio=unresolved_short_form_ratio,
        review_flag_ratio=review_flag_ratio,
        representative_role_counts=representative_role_counts,
        representative_diversity_ratio=representative_diversity,
        family_compression_ratio=family_compression,
    )


def score_before_after(
    before: pd.DataFrame,
    after: pd.DataFrame,
    *,
    sample_clusters: int | None = 50,
    seed: int = 0,
    token_jaccard_threshold: float = 0.8,
    weights: Optional[Mapping[str, float]] = None,
) -> dict[str, Any]:
    """Score a before/after change using keyword-level diagnostics.

    Returns a JSON-friendly dict:
    - total_score: 0..100 (baseline 50; positive means improvement)
    - components: per-metric contributions
    - before/after: full diagnostics payloads
    """

    w = {
        "redundancy": 25.0,
        "coverage": 15.0,
        "temporal_sparsity": 10.0,
        "review_load": 10.0,
        "representative_diversity": 10.0,
    }
    if weights:
        w.update({str(k): float(v) for k, v in weights.items()})

    diag_before = keyword_diagnostics(
        before,
        sample_clusters=sample_clusters,
        seed=seed,
        token_jaccard_threshold=token_jaccard_threshold,
    )
    diag_after = keyword_diagnostics(
        after,
        sample_clusters=sample_clusters,
        seed=seed,
        token_jaccard_threshold=token_jaccard_threshold,
    )

    # Pick the most stable redundancy metric available.
    def _pick_redundancy(d: KeywordDiagnostics) -> Optional[float]:
        if d.redundancy_token_jaccard_ratio is not None:
            return float(d.redundancy_token_jaccard_ratio)
        if d.redundancy_subphrase_ratio is not None:
            return float(d.redundancy_subphrase_ratio)
        return None

    before_red = _pick_redundancy(diag_before)
    after_red = _pick_redundancy(diag_after)
    before_cov = diag_before.doc_coverage.get("p50")
    after_cov = diag_after.doc_coverage.get("p50")
    before_single_year = diag_before.single_year_term_ratio
    after_single_year = diag_after.single_year_term_ratio
    before_review = diag_before.review_flag_ratio
    after_review = diag_after.review_flag_ratio
    before_diversity = diag_before.representative_diversity_ratio
    after_diversity = diag_after.representative_diversity_ratio

    components: dict[str, Any] = {}
    baseline = 50.0
    total = baseline

    if before_red is not None and after_red is not None:
        denom = before_red if before_red > 1e-12 else 1.0
        delta = (before_red - after_red) / denom
        contrib = w["redundancy"] * _clamp(delta, -1.0, 1.0)
        total += contrib
        components["redundancy"] = {
            "weight": w["redundancy"],
            "before": before_red,
            "after": after_red,
            "contribution": contrib,
        }
    else:
        components["redundancy"] = {"weight": w["redundancy"], "before": before_red, "after": after_red, "contribution": 0.0}

    if before_cov is not None and after_cov is not None:
        denom = before_cov if before_cov > 1e-12 else 1.0
        delta = (after_cov - before_cov) / denom
        contrib = w["coverage"] * _clamp(delta, -1.0, 1.0)
        total += contrib
        components["coverage"] = {
            "weight": w["coverage"],
            "before_p50_doc_coverage": before_cov,
            "after_p50_doc_coverage": after_cov,
            "contribution": contrib,
        }
    else:
        components["coverage"] = {"weight": w["coverage"], "before_p50_doc_coverage": before_cov, "after_p50_doc_coverage": after_cov, "contribution": 0.0}

    if before_single_year is not None and after_single_year is not None:
        denom = before_single_year if before_single_year > 1e-12 else 1.0
        delta = (before_single_year - after_single_year) / denom
        contrib = w["temporal_sparsity"] * _clamp(delta, -1.0, 1.0)
        total += contrib
        components["temporal_sparsity"] = {
            "weight": w["temporal_sparsity"],
            "before_single_year_ratio": before_single_year,
            "after_single_year_ratio": after_single_year,
            "contribution": contrib,
        }
    else:
        components["temporal_sparsity"] = {"weight": w["temporal_sparsity"], "before_single_year_ratio": before_single_year, "after_single_year_ratio": after_single_year, "contribution": 0.0}

    if before_review is not None and after_review is not None:
        denom = before_review if before_review > 1e-12 else 1.0
        delta = (before_review - after_review) / denom
        contrib = w["review_load"] * _clamp(delta, -1.0, 1.0)
        total += contrib
        components["review_load"] = {
            "weight": w["review_load"],
            "before_review_flag_ratio": before_review,
            "after_review_flag_ratio": after_review,
            "contribution": contrib,
        }
    else:
        components["review_load"] = {"weight": w["review_load"], "before_review_flag_ratio": before_review, "after_review_flag_ratio": after_review, "contribution": 0.0}

    if before_diversity is not None and after_diversity is not None:
        denom = before_diversity if before_diversity > 1e-12 else 1.0
        delta = (after_diversity - before_diversity) / denom
        contrib = w["representative_diversity"] * _clamp(delta, -1.0, 1.0)
        total += contrib
        components["representative_diversity"] = {
            "weight": w["representative_diversity"],
            "before_representative_diversity_ratio": before_diversity,
            "after_representative_diversity_ratio": after_diversity,
            "contribution": contrib,
        }
    else:
        components["representative_diversity"] = {
            "weight": w["representative_diversity"],
            "before_representative_diversity_ratio": before_diversity,
            "after_representative_diversity_ratio": after_diversity,
            "contribution": 0.0,
        }

    total_clamped = float(_clamp(total, 0.0, 100.0))

    return {
        "total_score": round(total_clamped, 2),
        "baseline": baseline,
        "components": components,
        "before": diag_before.to_dict(),
        "after": diag_after.to_dict(),
        "params": {
            "sample_clusters": sample_clusters,
            "seed": int(seed),
            "token_jaccard_threshold": float(token_jaccard_threshold),
        },
    }


__all__ = [
    "KeywordDiagnostics",
    "keyword_diagnostics",
    "score_before_after",
]
