"""Utilities and data loading for keyword extraction pipeline."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import math

import numpy as np
import pandas as pd
from scipy import sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

from .config import KeywordExtractionConfig
from .utils import _normalize_text_basic

try:  # optional dependency
    import polars as pl
except Exception:  # pragma: no cover
    pl = None

try:  # optional dependency for row-group streaming
    import pyarrow.parquet as pq
    _HAS_ARROW = True
except Exception:  # pragma: no cover
    pq = None
    _HAS_ARROW = False


def _effective_n_jobs(n_jobs: Optional[int]) -> int:
    if n_jobs in (None, 0):
        return 1
    try:
        n = int(n_jobs)
    except Exception:
        return 1
    if n == -1:
        import os
        return max(1, os.cpu_count() or 1)
    return max(1, n)


def _argpartition_topk(values: np.ndarray, indices: np.ndarray, k: int) -> List[Tuple[int, float]]:
    """Return top-k (col, value) sorted descending from sparse row arrays."""
    nnz = values.shape[0]
    if nnz == 0:
        return []
    k = min(k, nnz)
    part = np.argpartition(values, -k)[-k:]
    v = values[part]
    idx = indices[part]
    order = np.argsort(v)[::-1]
    return list(zip(idx[order].tolist(), v[order].tolist()))


def _group_sum_by_cluster(X: sp.csr_matrix, codes: np.ndarray, K: int) -> sp.csr_matrix:
    """Aggregate document rows to cluster rows by integer codes."""
    if X.nnz == 0:
        return sp.csr_matrix((K, X.shape[1]), dtype=X.dtype)
    coo = X.tocoo(copy=False)
    new_rows = codes[coo.row]
    res = sp.coo_matrix((coo.data, (new_rows, coo.col)), shape=(K, X.shape[1]))
    return res.tocsr()


def _mmr_jaccard_select(candidates: List[str], scores: Dict[str, float], lambda_: float, top_k: int) -> List[str]:
    """Basic MMR selection using Jaccard overlap between token sets."""
    if lambda_ <= 0.0:
        return candidates[:top_k]
    selected: List[str] = []
    cand = candidates[:]

    def token_set(term: str) -> set[str]:
        return set(term.split())

    tokenised = {term: token_set(term) for term in cand}

    while cand and len(selected) < top_k:
        best_term = None
        best_score = float("-inf")
        for term in cand:
            relevance = scores.get(term, 0.0)
            diversity = 0.0
            if selected:
                term_tokens = tokenised[term]
                diversity = max(
                    (len(term_tokens & tokenised[sel]) / max(1, len(term_tokens | tokenised[sel])))
                    for sel in selected
                )
            mmr = lambda_ * relevance - (1 - lambda_) * diversity
            if mmr > best_score:
                best_score = mmr
                best_term = term
        selected.append(best_term)  # type: ignore[arg-type]
        cand.remove(best_term)  # type: ignore[arg-type]
    return selected


def _llr_2x2(k11: int, k12: int, k21: int, k22: int) -> float:
    """Log-likelihood ratio for a 2x2 contingency table."""
    n = k11 + k12 + k21 + k22
    if n == 0:
        return 0.0

    def _safe_log(x: float) -> float:
        return math.log(x) if x > 0 else 0.0

    row1 = k11 + k12
    row2 = k21 + k22
    col1 = k11 + k21
    col2 = k12 + k22

    def expected(r: int, c: int) -> float:
        return (r * c) / n if n else 0.0

    e11 = expected(row1, col1)
    e12 = expected(row1, col2)
    e21 = expected(row2, col1)
    e22 = expected(row2, col2)

    g2 = 0.0
    for observed, exp in ((k11, e11), (k12, e12), (k21, e21), (k22, e22)):
        if observed > 0 and exp > 0:
            g2 += 2.0 * observed * _safe_log(observed / exp)
    return g2


def _suppress_subphrases(terms: List[str], max_keep: int) -> List[str]:
    """Drop terms that are subphrases of already selected longer terms."""
    kept: List[str] = []
    for term in terms:
        padded = f" {term} "
        if any(padded in f" {existing} " for existing in kept):
            continue
        kept.append(term)
        if len(kept) >= max_keep:
            break
    return kept


class _DataSource:
    """Streaming data loader joining abstracts with membership."""

    def __init__(self, config: KeywordExtractionConfig) -> None:
        self.cfg = config
        self._membership: Optional[pd.Series] = None
        self._clusters_sorted: Optional[np.ndarray] = None
        self._author_keywords: Optional[pd.Series] = None

    def membership_map(self) -> pd.Series:
        if self._membership is not None:
            return self._membership
        cfg = self.cfg
        df = pd.read_parquet(cfg.membership_path, columns=[cfg.uid_col, cfg.cluster_level])
        df = df.dropna(subset=[cfg.cluster_level])
        df[cfg.cluster_level] = df[cfg.cluster_level].astype(int)
        df = df.drop_duplicates(subset=[cfg.uid_col], keep="last")
        series = df.set_index(cfg.uid_col)[cfg.cluster_level]
        self._membership = series
        self._clusters_sorted = np.array(sorted(series.unique().tolist()), dtype=int)
        return series

    def cluster_ids_sorted(self) -> np.ndarray:
        if self._clusters_sorted is None:
            _ = self.membership_map()
        assert self._clusters_sorted is not None
        return self._clusters_sorted

    def cluster_indexer(self) -> Dict[int, int]:
        ids = self.cluster_ids_sorted()
        return {int(cid): idx for idx, cid in enumerate(ids)}

    def author_keyword_map(self) -> Optional[pd.Series]:
        cfg = self.cfg
        path = cfg.author_keyword_path
        if path is None:
            return None
        if self._author_keywords is not None:
            return self._author_keywords

        uid_col = cfg.author_keyword_uid_col
        term_col = cfg.author_keyword_term_col
        try:
            keywords = pd.read_parquet(path, columns=[uid_col, term_col])
        except Exception:
            keywords = pd.read_parquet(path)
            missing = {uid_col, term_col} - set(keywords.columns)
            if missing:
                raise KeyError(f"Author keyword parquet is missing columns: {missing}")
            keywords = keywords[[uid_col, term_col]]

        keywords = keywords.dropna(subset=[uid_col, term_col])
        if keywords.empty:
            self._author_keywords = pd.Series(dtype=str)
            return self._author_keywords

        membership = self.membership_map()
        uid_dtype = membership.index.dtype
        try:
            keywords[uid_col] = keywords[uid_col].astype(uid_dtype, copy=False)
        except Exception:
            # fallback to string if dtype conversion fails
            keywords[uid_col] = keywords[uid_col].astype(str)

        keywords[term_col] = keywords[term_col].astype(str).map(_normalize_text_basic)

        joiner = cfg.author_keyword_join

        def _collapse(vals: pd.Series) -> str:
            seen: List[str] = []
            for val in vals:
                term = str(val).strip()
                if not term:
                    continue
                term_clean = term.replace("'", " ").replace('"', " ")
                term_clean = _normalize_text_basic(term_clean)
                tokens = term_clean.split()
                if tokens and all(tok == "sup" for tok in tokens):
                    continue
                if not term_clean or term_clean in seen:
                    continue
                seen.append(term_clean)
            return joiner.join(seen)

        collapsed = keywords.groupby(uid_col, dropna=True)[term_col].apply(_collapse)
        collapsed = collapsed[collapsed.astype(bool)]
        self._author_keywords = collapsed
        return self._author_keywords

    def batch_iter(self) -> Iterator[pd.DataFrame]:
        cfg = self.cfg
        membership = self.membership_map()
        uid, abstract_col, title_col, year_col = cfg.uid_col, cfg.abstract_col, cfg.title_col, cfg.year_col

        if _HAS_ARROW and cfg.use_pyarrow_streaming:
            pf = pq.ParquetFile(str(cfg.abstract_path))
            columns = [uid, abstract_col, year_col]
            if cfg.include_title and title_col in pf.schema_arrow.names:
                columns.append(title_col)
            for rg in range(pf.num_row_groups):
                table = pf.read_row_group(rg, columns=columns)
                docs = table.to_pandas()
                docs["cluster_id"] = docs[uid].map(membership)
                docs = docs.dropna(subset=["cluster_id", abstract_col])
                if docs.empty:
                    continue
                texts = self._build_text_column(docs, abstract_col, title_col)
                yield pd.DataFrame(
                    {
                        "cluster_id": docs["cluster_id"].astype(int).to_numpy(),
                        "text": texts.to_numpy(),
                        "pubyear": pd.to_numeric(docs[year_col], errors="coerce").astype("Int64"),
                    }
                )
        else:
            docs = self._read_full_docs()
            docs["cluster_id"] = docs[uid].map(membership)
            docs = docs.dropna(subset=["cluster_id", abstract_col])
            if docs.empty:
                return
            texts = self._build_text_column(docs, abstract_col, title_col)
            yield pd.DataFrame(
                {
                    "cluster_id": docs["cluster_id"].astype(int).to_numpy(),
                    "text": texts.to_numpy(),
                    "pubyear": pd.to_numeric(docs[year_col], errors="coerce").astype("Int64"),
                }
            )

    def _read_full_docs(self) -> pd.DataFrame:
        cfg = self.cfg
        uid, abstract_col, title_col, year_col = cfg.uid_col, cfg.abstract_col, cfg.title_col, cfg.year_col
        columns = [uid, abstract_col, year_col]
        if cfg.include_title:
            columns.append(title_col)
        if cfg.use_polars and pl is not None:
            try:
                table = pl.scan_parquet(str(cfg.abstract_path)).select(columns).collect(streaming=False)
            except Exception:
                table = pl.scan_parquet(str(cfg.abstract_path)).select([uid, abstract_col, year_col]).collect(streaming=False)
            return table.to_pandas()
        return pd.read_parquet(cfg.abstract_path, columns=columns)

    def _build_text_column(self, df: pd.DataFrame, abstract_col: str, title_col: str) -> pd.Series:
        cfg = self.cfg
        abstracts = df[abstract_col].map(_normalize_text_basic)
        text_series: pd.Series
        if cfg.include_title and title_col in df.columns:
            titles = df[title_col].map(_normalize_text_basic).fillna("")
            rep = max(0, int(round(cfg.title_weight)))
            if rep > 1:
                boost = (titles + " ") * rep
            elif rep == 1:
                boost = titles + " "
            else:
                boost = ""
            text_series = (boost + abstracts).astype(str)
        else:
            text_series = abstracts.astype(str)

        author_map = self.author_keyword_map()
        if author_map is not None and not author_map.empty:
            uid_col = cfg.uid_col
            if uid_col in df.columns:
                keywords = df[uid_col].map(author_map).fillna("")
                keywords = keywords.astype(str).map(_normalize_text_basic)
                text_series = text_series.str.cat(keywords, sep=" ", na_rep="")
        return text_series.str.replace(r"\s+", " ", regex=True).str.strip()
