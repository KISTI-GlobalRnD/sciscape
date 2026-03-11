"""Keyword extraction pipeline for cluster-level summaries (document-level aggregation).

Core ideas:
* Vectorise documents (abstracts with optional title boost) using CountVectorizer.
* Aggregate document-term matrices per cluster via sparse group sums.
* Compute c‑TF‑IDF exactly on the aggregated matrix and rank terms (uni+bi).
* Recompute publication-year series only for selected terms in a second pass.
* Optionally stream Parquet row-groups using pyarrow.
"""

from __future__ import annotations

from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import json
import logging
import math
import os
import re
import hashlib
from datetime import datetime, timezone
from textwrap import dedent

import numpy as np
import pandas as pd
import joblib
from joblib import Parallel, delayed
from scipy import sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

from .config import KeywordExtractionConfig, KeywordRecord
from .llm_canonicalize import LLMCanonicalizeMixin

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

logger = logging.getLogger(__name__)


TokenCounter = Counter[str]
YearCounter = Counter[int]
TermYearCounter = MutableMapping[str, YearCounter]
ClusterTermYearCounter = MutableMapping[int, TermYearCounter]

_HTML_TAG_RE = re.compile(r"</?[^<>]+>")


def _normalize_text_basic(text: object) -> str:
    """Cheap normalisation shared across stages."""
    if not isinstance(text, str):
        return ""
    text = _HTML_TAG_RE.sub(" ", text)
    return " ".join(text.replace("\r", " ").replace("\n", " ").split())


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


class KeywordExtractionPipeline(LLMCanonicalizeMixin):
    """Coordinate the keyword extraction stages."""

    def __init__(self, config: KeywordExtractionConfig) -> None:
        self.config = config
        self.verbose = bool(config.verbose)
        if self.verbose:
            if not logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
                logger.addHandler(handler)
            if logger.level > logging.INFO:
                logger.setLevel(logging.INFO)

        stopwords_set = config.build_stopword_set()
        self.stopwords_set: set[str] = stopwords_set
        self.stopwords_list: Optional[List[str]] = sorted(stopwords_set) if stopwords_set else None
        self.n_jobs_effective: int = _effective_n_jobs(config.n_jobs)

        self._data = _DataSource(config)
        self.cluster_ids: np.ndarray = self._data.cluster_ids_sorted()
        self.cluster_index: Dict[int, int] = self._data.cluster_indexer()
        self.K: int = len(self.cluster_ids)

        self.vec_uni: Optional[CountVectorizer] = None
        self.vec_phrase: Optional[CountVectorizer] = None
        self.feature_names_uni: Optional[np.ndarray] = None
        self.feature_names_phrase: Optional[np.ndarray] = None
        self.feature_names_all: Optional[np.ndarray] = None

        self.C_uni: Optional[sp.csr_matrix] = None
        self.C_phrase: Optional[sp.csr_matrix] = None
        self.DF_uni: Optional[sp.csr_matrix] = None
        self.DF_phrase: Optional[sp.csr_matrix] = None
        self.DF_all: Optional[sp.csr_matrix] = None
        self.C_all: Optional[sp.csr_matrix] = None
        self.final_keywords: Optional[pd.DataFrame] = None
        self.cluster_doc_counts: Optional[np.ndarray] = None
        self.total_docs: int = 0
        self._alias_client = None
        self.cluster_year_token_denoms: Dict[int, Counter[int]] = defaultdict(Counter)
        self._alias_cache_dir: Optional[Path] = None
        if (
            self.config.apply_alias_map
            and (self.config.alias_strategy or "none").lower() != "none"
            and self.config.alias_cache_enabled
        ):
            if self.config.alias_cache_path is None:
                timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
                default_base = Path("artifacts") / "canonicalise" / timestamp
                self.config.alias_cache_path = default_base
            base = Path(self.config.alias_cache_path)
            base.mkdir(parents=True, exist_ok=True)
            self._alias_cache_dir = base
        self._builtin_alias_cache: Optional[Dict[str, str]] = None
        self._log("Initialised pipeline for %d clusters (n_jobs=%d)", self.K, self.n_jobs_effective)

    def _log(self, message: str, *args) -> None:
        if self.config.verbose:
            logger.info(message, *args)

    def _is_stopword_ngram(self, term: str) -> bool:
        if not term:
            return False
        stopwords = self.stopwords_set or ENGLISH_STOP_WORDS
        tokens = term.split()
        if not tokens:
            return False
        if self.config.lowercase:
            return all(token in stopwords for token in tokens)
        return all(token.lower() in stopwords for token in tokens)

    # ----- Stage 0: fit vectorisers on streamed text -----

    def _fit_vectorizers(self) -> None:
        cfg = self.config
        self._log("Stage 0: fitting vectorisers (include_title=%s, author_keywords=%s)",
                  cfg.include_title, cfg.author_keyword_path is not None)

        def stream_texts() -> Iterator[str]:
            for batch in self._data.batch_iter():
                for txt in batch["text"]:
                    yield txt

        vec_uni = CountVectorizer(
            lowercase=cfg.lowercase,
            stop_words=self.stopwords_list,
            token_pattern=cfg.token_pattern,
            strip_accents=cfg.strip_accents,
            min_df=cfg.min_df_unigram,
            max_df=cfg.max_df_unigram,
            max_features=cfg.max_features_unigram,
            ngram_range=(1, 1),
            dtype=np.int32,
        )
        vec_uni.fit(stream_texts())

        self.vec_uni = vec_uni
        self.feature_names_uni = vec_uni.get_feature_names_out()
        self._log("Stage 0: unigram vocabulary size = %d", len(self.feature_names_uni))

        if cfg.use_phrase_vectorizer and cfg.ngram_max >= cfg.ngram_min >= 2:
            vec_phrase = CountVectorizer(
                lowercase=cfg.lowercase,
                stop_words=self.stopwords_list,
                token_pattern=cfg.token_pattern,
                strip_accents=cfg.strip_accents,
                min_df=cfg.min_df_phrase,
                max_df=cfg.max_df_phrase,
                max_features=cfg.max_features_phrase,
                ngram_range=(cfg.ngram_min, cfg.ngram_max),
                dtype=np.int32,
            )
            vec_phrase.fit(stream_texts())
            self.vec_phrase = vec_phrase
            self.feature_names_phrase = vec_phrase.get_feature_names_out()
            self._log(
                "Stage 0: phrase vocabulary size = %d (ngram_range=%s)",
                len(self.feature_names_phrase),
                (cfg.ngram_min, cfg.ngram_max),
            )
        else:
            self.vec_phrase = None
            self.feature_names_phrase = np.array([], dtype=str)
            self._log("Stage 0: phrase vectoriser disabled or empty output")

    # ----- Stage 1: aggregate document counts to cluster counts -----

    def _aggregate_counts(self) -> None:
        self._log("Stage 1: aggregating counts per cluster...")
        assert self.vec_uni is not None
        vec_phrase = self.vec_phrase
        K = self.K
        V_uni = len(self.vec_uni.vocabulary_)
        C_uni = sp.csr_matrix((K, V_uni), dtype=np.int64)
        DF_uni = sp.csr_matrix((K, V_uni), dtype=np.int64)
        cluster_doc_counts = np.zeros(K, dtype=np.int64)
        total_docs = 0

        if vec_phrase is not None:
            V_phrase = len(vec_phrase.vocabulary_)
            C_phrase = sp.csr_matrix((K, V_phrase), dtype=np.int64)
            DF_phrase = sp.csr_matrix((K, V_phrase), dtype=np.int64)
        else:
            C_phrase = None
            DF_phrase = None

        for batch in self._data.batch_iter():
            texts = batch["text"].tolist()
            clusters = batch["cluster_id"].astype(int).to_numpy()
            codes = np.array([self.cluster_index[int(cid)] for cid in clusters], dtype=np.int32)

            X_uni = self.vec_uni.transform(texts)
            C_uni = C_uni + _group_sum_by_cluster(X_uni, codes, K)
            X_uni_bool = X_uni.copy()
            X_uni_bool.data[:] = 1
            DF_uni = DF_uni + _group_sum_by_cluster(X_uni_bool, codes, K)
            cluster_doc_counts += np.bincount(codes, minlength=K)
            total_docs += len(texts)
            if vec_phrase is not None:
                X_phrase = vec_phrase.transform(texts)
                C_phrase = C_phrase + _group_sum_by_cluster(X_phrase, codes, K)
                X_phrase_bool = X_phrase.copy()
                X_phrase_bool.data[:] = 1
                DF_phrase = DF_phrase + _group_sum_by_cluster(X_phrase_bool, codes, K)

        C_uni = C_uni.tocsr()
        DF_uni = DF_uni.tocsr()
        if C_phrase is not None:
            C_phrase = C_phrase.tocsr()
        if DF_phrase is not None:
            DF_phrase = DF_phrase.tocsr()

        if C_phrase is not None and C_phrase.shape[1] > 0 and self.config.phrase_min_count_per_cluster > 1:
            max_per_col_matrix = C_phrase.max(axis=0)
            if sp.issparse(max_per_col_matrix):
                max_per_col = max_per_col_matrix.toarray().ravel()
            elif hasattr(max_per_col_matrix, "A1"):
                max_per_col = max_per_col_matrix.A1
            else:
                max_per_col = np.asarray(max_per_col_matrix).ravel()
            thresh = int(self.config.phrase_min_count_per_cluster)
            keep = max_per_col >= thresh
            if keep.size == 0 or not keep.any():
                C_phrase = sp.csr_matrix((K, 0), dtype=np.int64)
                DF_phrase = sp.csr_matrix((K, 0), dtype=np.int64) if DF_phrase is not None else None
                self.feature_names_phrase = np.array([], dtype=str)
            elif not np.all(keep):
                C_phrase = C_phrase[:, keep]
                if DF_phrase is not None:
                    DF_phrase = DF_phrase[:, keep]
                self.feature_names_phrase = self.feature_names_phrase[keep]  # type: ignore

        self.C_uni = C_uni
        self.C_phrase = C_phrase
        self.DF_uni = DF_uni
        self.DF_phrase = DF_phrase
        self.cluster_doc_counts = cluster_doc_counts
        self.total_docs = total_docs
        self._log(
            "Stage 1: processed %d documents across %d clusters (total tokens: %d)",
            self.total_docs,
            self.K,
            int(C_uni.sum()) + int(C_phrase.sum()) if C_phrase is not None else int(C_uni.sum()),
        )

    # ----- Stage 2: compute c‑TF‑IDF and select top terms -----

    @staticmethod
    def _compute_c_tfidf(C: sp.csr_matrix) -> sp.csr_matrix:
        K, V = C.shape
        if K == 0 or V == 0:
            return sp.csr_matrix((K, V), dtype=np.float32)
        row_sums = np.asarray(C.sum(axis=1)).ravel().astype(np.float64)
        row_sums[row_sums == 0] = 1.0
        tf = C.multiply(1.0 / row_sums[:, None])
        df = np.asarray((C > 0).sum(axis=0)).ravel().astype(np.float64)
        idf = np.log((1.0 + K) / (1.0 + df)) + 1.0
        return tf.multiply(idf).tocsr()

    def _rank_topk(
        self,
        C_all: sp.csr_matrix,
        scores: sp.csr_matrix,
        feature_names: np.ndarray,
        DF_all: Optional[sp.csr_matrix],
        df_global: Optional[np.ndarray],
    ) -> pd.DataFrame:
        K, _ = scores.shape
        top_k = max(1, int(self.config.top_n_keywords))
        pool_size = max(top_k, int(np.ceil(self.config.mmr_pool_factor * top_k)))
        min_doc_cov = max(0, int(self.config.min_cluster_doc_coverage))
        cluster_doc_counts = getattr(self, "cluster_doc_counts", None)
        total_docs = getattr(self, "total_docs", 0)

        def extract_row(r: int) -> List[Tuple[int, str, float, int, int]]:
            row = scores.getrow(r)
            if row.nnz == 0:
                return []
            cluster_id = int(self.cluster_ids[r])
            cluster_docs = int(cluster_doc_counts[r]) if cluster_doc_counts is not None else None
            top_terms = _argpartition_topk(row.data, row.indices, pool_size)
            counts_row = C_all.getrow(r)
            freq_map = {j: int(v) for j, v in zip(counts_row.indices, counts_row.data)}
            doc_cov_map: Dict[int, int] = {}
            if DF_all is not None:
                df_row = DF_all.getrow(r)
                doc_cov_map = {j: int(v) for j, v in zip(df_row.indices, df_row.data)}

            terms: List[str] = []
            ctf_scores: List[float] = []
            freqs: List[int] = []
            doc_covs: List[int] = []
            llr_scores: List[float] = []

            for idx, raw_score in top_terms:
                freq = freq_map.get(idx, 0)
                if freq <= 0:
                    continue
                doc_cov = doc_cov_map.get(idx, 0)
                if doc_cov < min_doc_cov:
                    continue
                if (
                    cluster_docs
                    and self.config.min_cluster_doc_coverage_ratio > 0.0
                    and (doc_cov / max(1, cluster_docs)) < self.config.min_cluster_doc_coverage_ratio
                ):
                    continue
                term = feature_names[idx]
                if self._is_stopword_ngram(term):
                    continue
                terms.append(term)
                ctf_scores.append(float(raw_score))
                freqs.append(freq)
                doc_covs.append(doc_cov)

                if (
                    self.config.w_llr > 0.0
                    and df_global is not None
                    and cluster_docs is not None
                    and total_docs > 0
                ):
                    gdf = int(df_global[idx]) if idx < len(df_global) else 0
                    k11 = doc_cov
                    k12 = max(0, cluster_docs - k11)
                    k21 = max(0, gdf - k11)
                    k22 = max(0, total_docs - cluster_docs - k21)
                    llr_scores.append(_llr_2x2(k11, k12, k21, k22))
                else:
                    llr_scores.append(0.0)

            if not terms:
                return []

            ctf_arr = np.asarray(ctf_scores, dtype=float)

            if self.config.w_llr > 0.0 and any(llr_scores):
                llr_arr = np.asarray(llr_scores, dtype=float)

                def _z(arr: np.ndarray) -> np.ndarray:
                    if arr.size < 2:
                        return np.zeros_like(arr)
                    mean = float(arr.mean())
                    std = float(arr.std())
                    return np.zeros_like(arr) if std <= 1e-12 else (arr - mean) / std

                final_scores = self.config.w_ctfidf * _z(ctf_arr) + self.config.w_llr * _z(llr_arr)
            else:
                final_scores = self.config.w_ctfidf * ctf_arr

            scored_terms = sorted(
                [(term, float(score), freq) for term, score, freq in zip(terms, final_scores.tolist(), freqs)],
                key=lambda item: -item[1],
            )

            ordered_terms = _suppress_subphrases([term for term, _, _ in scored_terms], max_keep=pool_size)
            term_score = {term: score for term, score, _ in scored_terms}
            term_freq = {term: freq for term, _, freq in scored_terms}
            term_doc_cov_map = {term: cov for term, cov in zip(terms, doc_covs)}

            if self.config.mmr_jaccard_lambda > 0.0:
                selected_terms = _mmr_jaccard_select(
                    ordered_terms, term_score, self.config.mmr_jaccard_lambda, top_k
                )
            else:
                selected_terms = ordered_terms[:top_k]

            return [
                (
                    cluster_id,
                    term,
                    term_score[term],
                    term_freq[term],
                    int(term_doc_cov_map.get(term, term_freq[term])),
                )
                for term in selected_terms
            ]

        results = Parallel(n_jobs=self.n_jobs_effective)(
            delayed(extract_row)(r) for r in range(K)
        )
        flat = [item for sub in results for item in sub]
        return pd.DataFrame(flat, columns=["cluster_id", "term", "score", "frequency", "doc_coverage"])

    def _stage_scores_and_topk(self) -> pd.DataFrame:
        self._log("Stage 2: scoring terms (top_n=%d)", self.config.top_n_keywords)
        fn_uni = self.feature_names_uni if self.feature_names_uni is not None else np.array([], dtype=str)
        fn_phrase = self.feature_names_phrase if self.feature_names_phrase is not None else np.array([], dtype=str)
        C_all = sp.hstack([m for m in (self.C_uni, self.C_phrase) if m is not None], format="csr").astype(np.int64)
        feature_names = np.concatenate([fn_uni, fn_phrase], axis=0)
        df_components = [m for m in (self.DF_uni, self.DF_phrase) if m is not None]
        DF_all = (
            sp.hstack(df_components, format="csr").astype(np.int64)
            if df_components
            else None
        )
        assert C_all.shape[1] == feature_names.shape[0], "Feature-name length mismatch with C_all"
        if df_components:
            assert DF_all is not None and DF_all.shape[1] == C_all.shape[1], "DF_all mismatch with C_all"
        self.C_all = C_all
        self.feature_names_all = feature_names
        scores = self._compute_c_tfidf(C_all)
        self.DF_all = DF_all
        df_global = (
            np.asarray(DF_all.sum(axis=0)).ravel().astype(np.int64)
            if DF_all is not None
            else None
        )
        result = self._rank_topk(C_all, scores, feature_names, DF_all, df_global)
        self._log("Stage 2: produced %d keyword rows", len(result))
        return result

    # ----- Stage 2.5: canonicalisation via optional alias map -----

    def save_stage2_snapshot(self, directory: Path, raw_top_df: pd.DataFrame) -> None:
        """Persist vectorisers, matrices, and the Stage 2 raw dataframe for later reuse."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.vec_uni, directory / "vec_uni.joblib")
        if self.vec_phrase is not None:
            joblib.dump(self.vec_phrase, directory / "vec_phrase.joblib")

        if self.C_uni is not None:
            sp.save_npz(directory / "C_uni.npz", self.C_uni)
        if self.C_phrase is not None:
            sp.save_npz(directory / "C_phrase.npz", self.C_phrase)
        if self.C_all is not None:
            sp.save_npz(directory / "C_all.npz", self.C_all)

        if self.DF_uni is not None:
            sp.save_npz(directory / "DF_uni.npz", self.DF_uni)
        if self.DF_phrase is not None:
            sp.save_npz(directory / "DF_phrase.npz", self.DF_phrase)
        if self.DF_all is not None:
            sp.save_npz(directory / "DF_all.npz", self.DF_all)

        if self.cluster_ids is not None:
            np.save(directory / "cluster_ids.npy", self.cluster_ids)
        if self.cluster_doc_counts is not None:
            np.save(directory / "cluster_doc_counts.npy", self.cluster_doc_counts)

        if self.feature_names_all is not None:
            np.save(directory / "feature_names_all.npy", self.feature_names_all)
        if self.feature_names_uni is not None:
            np.save(directory / "feature_names_uni.npy", self.feature_names_uni)
        if self.feature_names_phrase is not None:
            np.save(directory / "feature_names_phrase.npy", self.feature_names_phrase)

        raw_top_df.to_parquet(directory / "stage2_raw.parquet", index=False)

    def load_stage2_snapshot(self, directory: Path) -> pd.DataFrame:
        """Restore Stage 0–2 artefacts from disk and return the saved Stage 2 dataframe."""
        directory = Path(directory)
        vec_uni_path = directory / "vec_uni.joblib"
        if vec_uni_path.exists():
            self.vec_uni = joblib.load(vec_uni_path)
            self.feature_names_uni = np.load(directory / "feature_names_uni.npy", allow_pickle=True)
        else:
            self.vec_uni = None
            self.feature_names_uni = None

        vec_phrase_path = directory / "vec_phrase.joblib"
        if vec_phrase_path.exists():
            self.vec_phrase = joblib.load(vec_phrase_path)
            feature_phrase_path = directory / "feature_names_phrase.npy"
            if feature_phrase_path.exists():
                self.feature_names_phrase = np.load(feature_phrase_path, allow_pickle=True)
            else:
                self.feature_names_phrase = None
        else:
            self.vec_phrase = None
            self.feature_names_phrase = None

        C_uni_path = directory / "C_uni.npz"
        self.C_uni = sp.load_npz(C_uni_path) if C_uni_path.exists() else None
        C_phrase_path = directory / "C_phrase.npz"
        self.C_phrase = sp.load_npz(C_phrase_path) if C_phrase_path.exists() else None
        C_all_path = directory / "C_all.npz"
        self.C_all = sp.load_npz(C_all_path) if C_all_path.exists() else None

        DF_uni_path = directory / "DF_uni.npz"
        self.DF_uni = sp.load_npz(DF_uni_path) if DF_uni_path.exists() else None
        DF_phrase_path = directory / "DF_phrase.npz"
        self.DF_phrase = sp.load_npz(DF_phrase_path) if DF_phrase_path.exists() else None
        DF_all_path = directory / "DF_all.npz"
        self.DF_all = sp.load_npz(DF_all_path) if DF_all_path.exists() else None

        cluster_ids_path = directory / "cluster_ids.npy"
        if cluster_ids_path.exists():
            self.cluster_ids = np.load(cluster_ids_path, allow_pickle=True)
            self.K = len(self.cluster_ids)
            self.cluster_index = {int(cid): idx for idx, cid in enumerate(self.cluster_ids)}
        cluster_doc_counts_path = directory / "cluster_doc_counts.npy"
        if cluster_doc_counts_path.exists():
            self.cluster_doc_counts = np.load(cluster_doc_counts_path, allow_pickle=True)

        feature_names_all_path = directory / "feature_names_all.npy"
        if feature_names_all_path.exists():
            self.feature_names_all = np.load(feature_names_all_path, allow_pickle=True)

        raw_path = directory / "stage2_raw.parquet"
        if not raw_path.exists():
            raise FileNotFoundError(f"Missing Stage 2 dataframe at {raw_path}")
        raw_top_df = pd.read_parquet(raw_path)
        return raw_top_df

    def run_from_stage2_snapshot(self, directory: Path) -> pd.DataFrame:
        """Restore Stage 2 artefacts and execute from Stage 2.5 onward."""
        raw_top_df = self.load_stage2_snapshot(directory)
        canonical = self._maybe_canonicalise(raw_top_df)
        term_year = self._compute_year_series(canonical)
        canonical = canonical.assign(
            pub_year_series=[
                dict(term_year.get(int(row.cluster_id), {}).get(str(row.term), {}))
                for row in canonical.itertuples(index=False)
            ]
        )
        final = self._build_time_series_metrics(canonical, term_year)
        final = self._filter_stopword_only_terms(final)
        self.final_keywords = final
        return final

    def _filter_stopword_only_terms(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        def _valid(term: object) -> bool:
            if term is None:
                return False
            term_str = str(term).strip()
            if not term_str:
                return False
            return not self._is_stopword_ngram(term_str)

        mask = df["term"].map(_valid)
        filtered = df[mask].reset_index(drop=True)
        dropped = len(df) - len(filtered)
        if dropped > 0:
            self._log("Final cleanup: dropped %d stopword-only terms", dropped)
        return filtered

    # ----- Stage 3: year series for selected terms -----

    def _compute_year_series(self, top_df: pd.DataFrame) -> ClusterTermYearCounter:
        if top_df.empty:
            return defaultdict(lambda: defaultdict(Counter))

        cfg = self.config
        self._log("Stage 3: computing year series for %d rows", len(top_df))
        token_to_targets_uni: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
        token_to_targets_phrase: Dict[str, List[Tuple[int, str]]] = defaultdict(list)

        for row in top_df.itertuples(index=False):
            cluster_id = int(row.cluster_id)
            canonical = str(row.term)
            sources = getattr(row, "source_terms", None)
            if sources is None or (isinstance(sources, float) and math.isnan(sources)):  # type: ignore[arg-type]
                sources_list: List[str] = [canonical]
            elif isinstance(sources, str):
                sources_list = [sources]
            else:
                sources_list = [str(token) for token in sources if str(token).strip()]
            if not sources_list:
                sources_list = [canonical]
            unique_sources: List[str] = []
            for token in sources_list:
                token_str = token.strip()
                if not token_str or token_str in unique_sources:
                    continue
                unique_sources.append(token_str)
                if " " in token_str:
                    token_to_targets_phrase[token_str].append((cluster_id, canonical))
                else:
                    token_to_targets_uni[token_str].append((cluster_id, canonical))

        uni_vocab = sorted(token_to_targets_uni.keys())
        phrase_vocab = sorted(token_to_targets_phrase.keys())

        vec_uni = CountVectorizer(
            lowercase=cfg.lowercase,
            stop_words=self.stopwords_list,
            token_pattern=cfg.token_pattern,
            strip_accents=cfg.strip_accents,
            vocabulary={t: i for i, t in enumerate(uni_vocab)} if uni_vocab else None,
            ngram_range=(1, 1),
            dtype=np.int32,
        ) if uni_vocab else None

        vec_phrase = CountVectorizer(
            lowercase=cfg.lowercase,
            stop_words=self.stopwords_list,
            token_pattern=cfg.token_pattern,
            strip_accents=cfg.strip_accents,
            vocabulary={t: i for i, t in enumerate(phrase_vocab)} if phrase_vocab else None,
            ngram_range=(cfg.ngram_min, cfg.ngram_max),
            dtype=np.int32,
        ) if phrase_vocab else None

        idx2term_uni = np.array(uni_vocab, dtype=str) if uni_vocab else None
        idx2term_phrase = np.array(phrase_vocab, dtype=str) if phrase_vocab else None

        term_year: ClusterTermYearCounter = defaultdict(lambda: defaultdict(Counter))
        denom_tokens: Dict[int, Counter[int]] = defaultdict(Counter)
        full_uni_vec = self.vec_uni

        for batch in self._data.batch_iter():
            clusters = batch["cluster_id"].astype(int).to_numpy()
            years = batch["pubyear"].to_numpy()
            texts = batch["text"].tolist()

            if full_uni_vec is not None:
                doc_totals = np.asarray(full_uni_vec.transform(texts).sum(axis=1)).ravel()
                for idx, total in enumerate(doc_totals):
                    if total == 0:
                        continue
                    year = years[idx]
                    if pd.isna(year):
                        continue
                    cid = int(clusters[idx])
                    denom_tokens[cid][int(year)] += int(total)

            if vec_uni is not None:
                X = vec_uni.transform(texts).tocoo()
                for r, c, v in zip(X.row, X.col, X.data):
                    year = years[r]
                    if pd.isna(year):
                        continue
                    cid = int(clusters[r])
                    token = idx2term_uni[c]  # type: ignore[index]
                    for target_cid, canonical in token_to_targets_uni.get(token, []):
                        if target_cid != cid:
                            continue
                        term_year[cid][canonical][int(year)] += int(v)

            if vec_phrase is not None:
                X = vec_phrase.transform(texts).tocoo()
                for r, c, v in zip(X.row, X.col, X.data):
                    year = years[r]
                    if pd.isna(year):
                        continue
                    cid = int(clusters[r])
                    token = idx2term_phrase[c]  # type: ignore[index]
                    for target_cid, canonical in token_to_targets_phrase.get(token, []):
                        if target_cid != cid:
                            continue
                        term_year[cid][canonical][int(year)] += int(v)

        self.cluster_year_token_denoms = denom_tokens
        self._log("Stage 3: year series computed for %d clusters", len(term_year))
        return term_year

    def _build_time_series_metrics(self, top_df: pd.DataFrame, term_year: ClusterTermYearCounter) -> pd.DataFrame:
        if top_df.empty:
            return top_df.assign(
                year_denominators=[],
                ppm_series=[],
                loglift_series=[],
                bayesian_log_odds_series=[],
            )

        global_term_counts: Dict[str, Counter[int]] = defaultdict(Counter)
        for term_year_map in term_year.values():
            for term, year_counter in term_year_map.items():
                for year, count in year_counter.items():
                    global_term_counts[str(term)][int(year)] += int(count)

        global_year_denoms: Dict[int, int] = defaultdict(int)
        for year_map in self.cluster_year_token_denoms.values():
            for year, denom in year_map.items():
                global_year_denoms[int(year)] += int(denom)

        alpha = 0.5
        prior = 0.5

        pub_year_list: List[Dict[int, int]] = []
        year_denom_list: List[Dict[int, int]] = []
        ppm_list: List[Dict[int, float]] = []
        loglift_list: List[Dict[int, float]] = []
        bayes_list: List[Dict[int, float]] = []

        for row_idx, row in enumerate(top_df.itertuples(index=False)):
            cid = int(row.cluster_id)
            term = str(row.term)
            year_counts_raw = getattr(row, "pub_year_series", None)
            if not year_counts_raw:
                year_counts_raw = term_year.get(cid, {}).get(term, {})
            year_items = sorted(
                ((int(k), int(v)) for k, v in dict(year_counts_raw).items() if v),
                key=lambda kv: kv[0],
            )
            year_counts_sorted: Dict[int, int] = OrderedDict(year_items)
            denom_map = self.cluster_year_token_denoms.get(cid, {})

            year_denoms = OrderedDict((year, int(denom_map.get(year, 0))) for year in year_counts_sorted)
            ppm_series: Dict[int, float] = {}
            loglift_series: Dict[int, float] = {}
            bayes_series: Dict[int, float] = {}

            global_counts_map = global_term_counts.get(term, {})

            for year, count in year_counts_sorted.items():
                year_int = int(year)
                denom_c = denom_map.get(year_int, 0)
                global_count = global_counts_map.get(year_int, 0)
                global_denom = global_year_denoms.get(year_int, 0)

                ppm_series[year_int] = 1e6 * count / denom_c if denom_c else float("nan")

                if denom_c > 0 and global_denom > 0:
                    p_cluster = (count + alpha) / (denom_c + alpha)
                    p_global = (global_count + alpha) / (global_denom + alpha)
                    loglift_series[year_int] = float(np.log(p_cluster) - np.log(p_global))

                    p_year = (global_count + alpha) / (global_denom + alpha)
                    theta_cluster = (count + prior * p_year) / (denom_c + prior)
                    theta_global = (global_count + prior * p_year) / (global_denom + prior)
                    theta_cluster = float(np.clip(theta_cluster, 1e-9, 1 - 1e-9))
                    theta_global = float(np.clip(theta_global, 1e-9, 1 - 1e-9))
                    bayes_series[year_int] = float(
                        (np.log(theta_cluster) - np.log(1 - theta_cluster))
                        - (np.log(theta_global) - np.log(1 - theta_global))
                    )
                else:
                    loglift_series[year_int] = float("nan")
                    bayes_series[year_int] = float("nan")

            pub_year_list.append(year_counts_sorted)
            year_denom_list.append(year_denoms)
            ppm_list.append(ppm_series)
            loglift_list.append(loglift_series)
            bayes_list.append(bayes_series)

        return top_df.assign(
            pub_year_series=pub_year_list,
            year_denominators=year_denom_list,
            ppm_series=ppm_list,
            loglift_series=loglift_list,
            bayesian_log_odds_series=bayes_list,
        )

    # ----- Public API -----

    def run(self) -> pd.DataFrame:
        self._log("Pipeline run started")
        self._fit_vectorizers()
        self._aggregate_counts()
        top_df = self._stage_scores_and_topk()
        top_df = self._maybe_canonicalise(top_df)
        term_year = self._compute_year_series(top_df)
        if not top_df.empty:
            top_df = top_df.assign(
                pub_year_series=[
                    dict(term_year.get(int(row.cluster_id), {}).get(str(row.term), {}))
                    for row in top_df.itertuples(index=False)
                ]
            )
        else:
            top_df = top_df.assign(pub_year_series=[])
        top_df = self._build_time_series_metrics(top_df, term_year)
        top_df = self._filter_stopword_only_terms(top_df)
        self.final_keywords = top_df
        self._log("Pipeline run complete: final rows = %d", len(top_df))
        return top_df

    def top_unigrams(self) -> pd.DataFrame:
        """Return per-cluster top unigrams for debugging."""
        if self.C_uni is None or self.feature_names_uni is None:
            raise RuntimeError("Run the pipeline before requesting top unigrams.")
        scores = self._compute_c_tfidf(self.C_uni)
        top_k = max(1, int(self.config.top_n_unigrams))
        names = self.feature_names_uni

        def extract(r: int) -> List[Tuple[int, str, float, int]]:
            row = scores.getrow(r)
            if row.nnz == 0:
                return []
            cid = int(self.cluster_ids[r])
            top_terms = _argpartition_topk(row.data, row.indices, top_k)
            counts_row = self.C_uni.getrow(r)
            freq_map = {j: int(v) for j, v in zip(counts_row.indices, counts_row.data)}
            return [
                (cid, names[idx], float(score), freq_map.get(idx, 0))
                for idx, score in top_terms
                if freq_map.get(idx, 0) > 0
            ]

        results = Parallel(n_jobs=self.n_jobs_effective)(
            delayed(extract)(r) for r in range(self.C_uni.shape[0])
        )
        flat = [item for sub in results for item in sub]
        return pd.DataFrame(flat, columns=["cluster_id", "term", "score", "frequency"])


def run_keyword_pipeline(config: KeywordExtractionConfig) -> pd.DataFrame:
    """Convenience entry point returning the keyword dataframe."""
    pipeline = KeywordExtractionPipeline(config)
    return pipeline.run()
