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


class KeywordExtractionPipeline:
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

    def _alias_cache_enabled(self) -> bool:
        return self._alias_cache_dir is not None

    def _alias_cache_file(self, cluster_id: int, batch_hash: str) -> Path:
        assert self._alias_cache_dir is not None
        subdir = self._alias_cache_dir / str(cluster_id)
        subdir.mkdir(parents=True, exist_ok=True)
        return subdir / f"{batch_hash}.json"

    def _alias_batch_hash(self, cluster_id: int, subset: pd.DataFrame) -> str:
        key_fields = self.config.alias_cache_key_fields or ("term", "frequency", "doc_coverage", "score")
        alias_strategy = (self.config.alias_strategy or "none").lower()
        if alias_strategy == "llm_candidates":
            cand_col = str(self.config.alias_candidate_column or "candidates")
            if cand_col and cand_col not in key_fields and cand_col in subset.columns:
                key_fields = tuple([*key_fields, cand_col])

        def _json_safe(value: object) -> object:
            """Make values stable+JSON-serialisable for cache keys (e.g., sets, numpy scalars)."""
            if isinstance(value, np.generic):
                return value.item()
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, Mapping):
                return {str(k): _json_safe(v) for k, v in value.items()}
            if isinstance(value, list):
                return [_json_safe(v) for v in value]
            if isinstance(value, tuple):
                return [_json_safe(v) for v in value]
            if isinstance(value, set):
                items = [_json_safe(v) for v in value]
                return sorted(items, key=lambda x: json.dumps(x, sort_keys=True, default=str))
            return value

        records: List[Dict[str, object]] = []
        for row in subset.itertuples(index=False):
            entry: Dict[str, object] = {"cluster_id": int(cluster_id)}
            for field in key_fields:
                entry[field] = _json_safe(getattr(row, field, None))
            records.append(entry)
        records_sorted = sorted(records, key=lambda x: json.dumps(x, sort_keys=True))
        payload = {
            "alias_model": self.config.alias_model,
            "alias_strategy": alias_strategy,
            "alias_stopword_strictness": self.config.alias_stopword_strictness,
            "alias_allow_translation": self.config.alias_allow_translation,
            "alias_candidate_column": self.config.alias_candidate_column,
            "alias_candidate_max": int(self.config.alias_candidate_max),
            "alias_candidate_enforce": bool(self.config.alias_candidate_enforce),
            "records": records_sorted,
        }
        payload_str = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(payload_str.encode("utf-8")).hexdigest()

    def _load_alias_cache(self, cluster_id: int, batch_hash: str) -> Optional[str]:
        if not self._alias_cache_enabled():
            return None
        cache_file = self._alias_cache_file(cluster_id, batch_hash)
        if not cache_file.exists():
            return None
        with cache_file.open("r", encoding="utf-8") as fh:
            cached = json.load(fh)
        return cached.get("raw_response")

    def _save_alias_cache(
        self,
        cluster_id: int,
        batch_hash: str,
        payload: Mapping[str, object],
        raw_response: str,
    ) -> None:
        if not self._alias_cache_enabled():
            return
        cache_file = self._alias_cache_file(cluster_id, batch_hash)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with cache_file.open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "cluster_id": int(cluster_id),
                    "alias_model": self.config.alias_model,
                    "alias_stopword_strictness": self.config.alias_stopword_strictness,
                    "alias_allow_translation": self.config.alias_allow_translation,
                    "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    "payload": payload,
                    "raw_response": raw_response,
                },
                fh,
                ensure_ascii=False,
            )

    def _mapping_path(self, cluster_id: int) -> Path:
        assert self._alias_cache_dir is not None
        base = self._alias_cache_dir / "mapping"
        base.mkdir(parents=True, exist_ok=True)
        return base / f"{int(cluster_id)}.json"

    def _load_alias_mapping(self, cluster_id: int) -> Dict[str, Dict[str, object]]:
        if not self._alias_cache_enabled():
            return {}
        path = self._mapping_path(cluster_id)
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        items = payload.get("items", [])
        mapping: Dict[str, Dict[str, object]] = {}
        for item in items:
            term = str(item.get("original") or item.get("term") or "").strip()
            if not term:
                continue
            mapping[term] = {
                "cluster_id": int(cluster_id),
                "original": term,
                "action": item.get("action", "keep"),
                "canonical": item.get("canonical", term),
                "notes": item.get("notes", ""),
                "reason": item.get("reason", ""),
            }
        return mapping

    def _save_alias_mapping(self, cluster_id: int, mapping: Dict[str, Dict[str, object]]) -> None:
        if not self._alias_cache_enabled():
            return
        items = []
        for term, entry in sorted(mapping.items(), key=lambda kv: kv[0]):
            items.append(
                {
                    "original": term,
                    "action": entry.get("action", "keep"),
                    "canonical": entry.get("canonical", term),
                    "notes": entry.get("notes", ""),
                    "reason": entry.get("reason", ""),
                }
            )
        payload = {
            "cluster_id": int(cluster_id),
            "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "items": items,
        }
        path = self._mapping_path(cluster_id)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

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

    def _maybe_canonicalise(self, top_df: pd.DataFrame) -> pd.DataFrame:
        if top_df.empty:
            return top_df
        cfg = self.config
        strategy = (cfg.alias_strategy or "none").lower()
        if not cfg.apply_alias_map or strategy == "none":
            self._log(
                "Stage 2.5: skipping canonicalisation (apply_alias_map=%s, strategy=%s)",
                cfg.apply_alias_map,
                strategy,
            )
            if "source_terms" not in top_df.columns:
                return top_df.assign(source_terms=top_df["term"].apply(lambda t: [str(t)]))
            return top_df

        self._log("Stage 2.5: canonicalising %d rows using strategy '%s'", len(top_df), strategy)
        alias_df = self._request_alias_actions(top_df)
        if alias_df.empty:
            logger.warning("Alias map requested but no instructions received; retaining original terms.")
            if "source_terms" not in top_df.columns:
                return top_df.assign(source_terms=top_df["term"].apply(lambda t: [str(t)]))
            return top_df

        manual_df = self._load_manual_aliases()
        if not manual_df.empty:
            alias_df = self._merge_alias_sources(alias_df, manual_df)

        result = self._apply_alias_instructions(top_df, alias_df)
        result = self._rescore_with_aliases(result)
        result = self._enforce_forbidden(result)
        self._log("Stage 2.5: canonicalisation complete -> %d rows", len(result))
        return result

    def _request_alias_actions(self, top_df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        strategy = (cfg.alias_strategy or "none").lower()
        columns = ["cluster_id", "original", "action", "canonical", "notes", "reason"]
        if strategy == "none":
            return pd.DataFrame(columns=columns)
        if strategy in {"cache_only", "load_only", "prev_top_df"}:
            if strategy == "load_only":
                self._rebuild_alias_mappings_from_cache(top_df)
            elif strategy == "prev_top_df":
                self._rebuild_alias_mappings_from_previous_topdf(top_df)

            records: List[Dict[str, object]] = []
            for cluster_id, group in top_df.groupby("cluster_id", sort=False):
                mapping = self._load_alias_mapping(int(cluster_id))
                for term in group["term"].astype(str):
                    entry = mapping.get(term)
                    if entry is None:
                        entry = {
                            "cluster_id": int(cluster_id),
                            "original": term,
                            "action": "keep",
                            "canonical": term,
                            "notes": "",
                            "reason": "default_keep",
                        }
                    records.append(entry)
            return pd.DataFrame.from_records(records, columns=columns) if records else pd.DataFrame(columns=columns)
        if strategy not in {"llm", "llm_candidates"}:
            raise ValueError(f"Unsupported alias_strategy '{cfg.alias_strategy}'.")
        if strategy == "llm_candidates":
            cand_col = str(cfg.alias_candidate_column or "candidates")
            if cand_col not in top_df.columns:
                raise KeyError(
                    f"alias_strategy='llm_candidates' requires a '{cand_col}' column in top_df "
                    f"(each row: list[str] or JSON-encoded list of candidate canonical terms)."
                )

        client = self._get_alias_client()
        max_terms = max(1, int(cfg.alias_max_terms_per_prompt))
        records: List[Dict[str, object]] = []
        self._log(
            "Stage 2.5: requesting alias actions for %d clusters (batch size %d)",
            top_df["cluster_id"].nunique(),
            max_terms,
        )

        for cluster_id, group in top_df.groupby("cluster_id", sort=False):
            ordered = group.sort_values("score", ascending=False)
            if ordered.empty:
                continue

            term_strings = ordered["term"].astype(str).tolist()
            mapping = self._load_alias_mapping(cluster_id)

            missing_terms = [term for term in term_strings if term not in mapping]
            mapping_changed = False

            if missing_terms:
                subset = ordered[ordered["term"].astype(str).isin(missing_terms)]
                subset = subset.sort_values("score", ascending=False)
                for start in range(0, len(subset), max_terms):
                    chunk = subset.iloc[start : start + max_terms]
                    batch_hash = self._alias_batch_hash(cluster_id, chunk)
                    cached_response = self._load_alias_cache(cluster_id, batch_hash)
                    if cached_response is not None:
                        raw_response = cached_response
                        payload = None
                    else:
                        if strategy == "llm_candidates":
                            messages, payload = self._build_alias_messages_candidates(cluster_id, chunk)
                        else:
                            messages, payload = self._build_alias_messages(cluster_id, chunk)
                        raw_response = None
                        last_error: Optional[Exception] = None
                        attempts = max(1, int(cfg.alias_retry))
                        for attempt in range(attempts):
                            try:
                                raw_response = self._invoke_alias_model(client, messages)
                                break
                            except Exception as exc:  # pragma: no cover - external dependency
                                last_error = exc
                                logger.warning(
                                    "Alias model request failed for cluster %s (attempt %s/%s): %s",
                                    cluster_id,
                                    attempt + 1,
                                    attempts,
                                    exc,
                                )
                        if raw_response is None:
                            logger.error(
                                "Unable to obtain alias response for cluster %s after %s attempts.",
                                cluster_id,
                                attempts,
                            )
                            continue
                        if self._alias_cache_enabled() and payload is not None:
                            self._save_alias_cache(cluster_id, batch_hash, payload, raw_response)

                    parsed = self._parse_alias_items(cluster_id, raw_response, chunk)
                    if not parsed:
                        continue
                    mapping_changed = True
                    for item in parsed:
                        term_key = str(item.get("original", "")).strip()
                        if not term_key:
                            continue
                        mapping[term_key] = {
                            "cluster_id": int(cluster_id),
                            "original": term_key,
                            "action": item.get("action", "keep"),
                            "canonical": item.get("canonical", term_key),
                            "notes": item.get("notes", ""),
                            "reason": item.get("reason", ""),
                        }

            cluster_records: List[Dict[str, object]] = []
            for term in term_strings:
                entry = mapping.get(term)
                if entry is None:
                    entry = {
                        "cluster_id": int(cluster_id),
                        "original": term,
                        "action": "keep",
                        "canonical": term,
                        "notes": "",
                        "reason": "default_keep",
                    }
                    mapping[term] = entry
                    mapping_changed = True
                cluster_records.append(entry)

            if mapping_changed:
                self._save_alias_mapping(cluster_id, mapping)

            records.extend(cluster_records)

        if not records:
            return pd.DataFrame(columns=columns)
        result = pd.DataFrame.from_records(records, columns=columns)
        self._log(
            "Stage 2.5: received alias instructions for %d terms",
            len(result),
        )
        return result

    def _load_manual_aliases(self) -> pd.DataFrame:
        columns = ["cluster_id", "original", "action", "canonical", "notes", "reason"]
        path = self.config.manual_alias_path
        if path is None:
            return pd.DataFrame(columns=columns)

        try:
            if not path.exists():
                logger.warning("Manual alias file not found at %s", path)
                return pd.DataFrame(columns=columns)
            suffix = path.suffix.lower()
            if suffix in (".csv", ".tsv"):
                sep = "," if suffix == ".csv" else "\t"
                manual_df = pd.read_csv(path, sep=sep)
            elif suffix in (".json", ".jsonl"):
                manual_df = pd.read_json(path, lines=suffix == ".jsonl")
            else:
                manual_df = pd.read_csv(path)
        except Exception as exc:
            logger.error("Failed to load manual alias file %s: %s", path, exc)
            return pd.DataFrame(columns=columns)

        manual_df = manual_df.rename(columns={c: c.lower() for c in manual_df.columns})
        missing = {"cluster_id", "original"} - set(manual_df.columns)
        if missing:
            logger.error(
                "Manual alias file %s missing required columns: %s",
                path,
                ", ".join(sorted(missing)),
            )
            return pd.DataFrame(columns=columns)

        manual_df["cluster_id"] = pd.to_numeric(manual_df["cluster_id"], errors="coerce").astype("Int64")
        manual_df = manual_df.dropna(subset=["cluster_id", "original"])
        if manual_df.empty:
            return pd.DataFrame(columns=columns)

        manual_df["cluster_id"] = manual_df["cluster_id"].astype(int)
        manual_df["original"] = manual_df["original"].astype(str).str.strip()
        manual_df = manual_df[manual_df["original"].astype(bool)]

        manual_df["action"] = (
            manual_df.get("action", "keep")
            .fillna("keep")
            .astype(str)
            .str.strip()
            .str.lower()
        )
        manual_df["canonical"] = manual_df.get("canonical", manual_df["original"])
        manual_df["canonical"] = manual_df["canonical"].fillna("").astype(str).str.strip()
        manual_df["canonical"] = manual_df.apply(
            lambda row: row["original"] if not row["canonical"] else row["canonical"],
            axis=1,
        )
        for col in ("notes", "reason"):
            manual_df[col] = manual_df.get(col, "").fillna("").astype(str)

        return manual_df[columns]

    @staticmethod
    def _merge_alias_sources(primary: pd.DataFrame, overrides: pd.DataFrame) -> pd.DataFrame:
        if overrides.empty:
            return primary
        columns = ["cluster_id", "original", "action", "canonical", "notes", "reason"]
        def _ensure(df: pd.DataFrame) -> pd.DataFrame:
            missing_cols = [col for col in columns if col not in df.columns]
            if missing_cols:
                for col in missing_cols:
                    df[col] = "" if col not in {"cluster_id"} else 0
            return df[columns]

        primary = _ensure(primary.copy()) if not primary.empty else pd.DataFrame(columns=columns)
        overrides = _ensure(overrides.copy())
        overrides["cluster_id"] = overrides["cluster_id"].astype(int)
        overrides["original"] = overrides["original"].astype(str).str.strip()

        primary["_manual_priority"] = 0
        overrides["_manual_priority"] = 1

        combined = pd.concat([primary, overrides], ignore_index=True)
        combined = combined.sort_values(
            ["cluster_id", "original", "_manual_priority"],
            ascending=[True, True, True],
            kind="mergesort",
        )
        combined = combined.drop_duplicates(subset=["cluster_id", "original"], keep="last")
        combined = combined.drop(columns="_manual_priority")
        combined["action"] = combined["action"].astype(str).str.strip().str.lower()
        combined["canonical"] = combined["canonical"].astype(str).str.strip()
        mask_empty_canonical = combined["canonical"] == ""
        if mask_empty_canonical.any():
            combined.loc[mask_empty_canonical, "canonical"] = combined.loc[mask_empty_canonical, "original"]
        for col in ("notes", "reason"):
            combined[col] = combined[col].fillna("").astype(str)
        return combined[columns]

    def _rebuild_alias_mappings_from_cache(self, top_df: pd.DataFrame) -> None:
        """Populate mapping/<cluster_id>.json purely from cached raw LLM responses."""
        if self._alias_cache_dir is None:
            return
        for cluster_id, group in top_df.groupby("cluster_id", sort=False):
            cluster_id = int(cluster_id)
            mapping: Dict[str, Dict[str, object]] = {}
            cache_dir = self._alias_cache_dir / str(cluster_id)
            if cache_dir.exists():
                for cache_file in sorted(cache_dir.glob("*.json")):
                    try:
                        data = json.loads(cache_file.read_text(encoding="utf-8"))
                        raw = data.get("raw_response", "")
                        payload = data.get("payload", {}) if isinstance(data, dict) else {}
                    except Exception:
                        continue
                    if not raw:
                        continue
                    # Reconstruct the subset that was sent to the model from the cached payload, so that
                    # `_parse_alias_items` can (a) default-fill only for that batch and (b) enforce candidate allowlists.
                    subset_records: List[Dict[str, object]] = []
                    if isinstance(payload, Mapping):
                        terms_payload = payload.get("terms")
                        if isinstance(terms_payload, list):
                            cand_col = str(self.config.alias_candidate_column or "candidates")
                            for term_obj in terms_payload:
                                if not isinstance(term_obj, Mapping):
                                    continue
                                term = str(term_obj.get("term", "")).strip()
                                if not term:
                                    continue
                                rec: Dict[str, object] = {"term": term}
                                for col in ("score", "frequency", "doc_coverage"):
                                    if col in term_obj:
                                        rec[col] = term_obj.get(col)
                                # Cached payload uses a stable key ("candidates") even if the input column name differs.
                                if "candidates" in term_obj:
                                    rec[cand_col] = term_obj.get("candidates")
                                elif cand_col in term_obj:
                                    rec[cand_col] = term_obj.get(cand_col)
                                subset_records.append(rec)
                    subset_df = pd.DataFrame.from_records(subset_records) if subset_records else group
                    try:
                        parsed_items = self._parse_alias_items(cluster_id, raw, subset_df)
                    except Exception:
                        continue
                    for item in parsed_items:
                        term = str(item.get("original") or "").strip()
                        if not term:
                            continue
                        mapping[term] = {
                            "cluster_id": cluster_id,
                            "original": term,
                            "action": item.get("action", "keep"),
                            "canonical": item.get("canonical", term),
                            "notes": item.get("notes", ""),
                            "reason": item.get("reason", ""),
                        }
            self._save_alias_mapping(cluster_id, mapping)

    def _rebuild_alias_mappings_from_previous_topdf(self, top_df: pd.DataFrame) -> None:
        """Reconstruct mapping/<cluster_id>.json from an existing canonical top_df artifact."""
        artifact_path = self.config.previous_top_df_path
        if artifact_path is None:
            return
        try:
            suffix = artifact_path.suffix.lower()
            if suffix in {".parquet", ".pq"}:
                prev_df = pd.read_parquet(artifact_path)
            elif suffix in {".json", ".jsonl"}:
                prev_df = pd.read_json(artifact_path, lines=suffix == ".jsonl")
            else:
                prev_df = pd.read_csv(artifact_path)
        except Exception as exc:
            logger.warning("Failed to load previous_top_df artifact %s: %s", artifact_path, exc)
            return

        required_cols = {"cluster_id", "term", "source_terms"}
        if not required_cols.issubset(prev_df.columns):
            logger.warning(
                "previous_top_df artifact %s missing required columns %s",
                artifact_path,
                ", ".join(sorted(required_cols)),
            )
            return

        def _source_list(value: object) -> List[str]:
            if isinstance(value, list):
                return [str(item).strip() for item in value if str(item).strip()]
            if isinstance(value, str):
                value = value.strip()
                if not value:
                    return []
                try:
                    decoded = json.loads(value)
                    if isinstance(decoded, list):
                        return [str(item).strip() for item in decoded if str(item).strip()]
                except Exception:
                    pass
                return [value]
            return []

        for cluster_id, group in prev_df.groupby("cluster_id", sort=False):
            try:
                cluster_id = int(cluster_id)
            except Exception:
                continue
            mapping = self._load_alias_mapping(cluster_id)
            for row in group.itertuples(index=False):
                canonical = self._clean_canonical_term(str(getattr(row, "term")))
                sources = _source_list(getattr(row, "source_terms"))
                if not sources:
                    sources = [canonical]
                for src in sources:
                    if not src:
                        continue
                    action = "keep" if self._clean_canonical_term(src) == canonical else "merge_into"
                    mapping[src] = {
                        "cluster_id": cluster_id,
                        "original": src,
                        "action": action,
                        "canonical": canonical,
                        "notes": "imported_from_previous_top_df",
                        "reason": "offline_rebuild",
                    }
            self._save_alias_mapping(cluster_id, mapping)

    def _build_alias_messages(self, cluster_id: int, subset: pd.DataFrame) -> Tuple[List[Dict[str, str]], Dict[str, object]]:
        cfg = self.config
        allow_translation = "yes" if cfg.alias_allow_translation else "no"
        stopwords = self.stopwords_set or ENGLISH_STOP_WORDS
        stopword_examples = sorted(list(stopwords))[:20]
        stopword_instruction = (
            " Pure stopword phrases (e.g., "
            + ", ".join(repr(w) for w in stopword_examples)
            + ") must be returned with action 'drop'."
        )

        system_prompt = dedent(
            """
            You are a scientific terminology normalisation assistant.
            For each keyword decide one action: keep, merge_into, translate, or drop.
            - keep: the term is already canonical.
            - merge_into: merge this term into another canonical form from the provided list.
            - translate: supply an English canonical form for non-English terms (allowed: {allow_translation}).
            - drop: remove junk terms (isolated stopwords, markup artefacts, etc.).{stopword_instruction}
            Respond with strict JSON only; no commentary.
            """
        ).format(allow_translation=allow_translation, stopword_instruction=stopword_instruction).strip()

        payload = {
            "cluster_id": int(cluster_id),
            "allow_translation": allow_translation,
            "stopword_strictness": self.config.alias_stopword_strictness,
            "stopwords_hint": stopword_examples,
            "terms": [
                {
                    "term": str(row.term),
                    "score": float(row.score),
                    "frequency": int(row.frequency),
                    "doc_coverage": int(getattr(row, "doc_coverage", 0)),
                }
                for row in subset.itertuples(index=False)
            ],
            "schema": {
                "fields": ["term", "action", "canonical", "notes", "reason"],
                "actions": ["keep", "merge_into", "translate", "drop"],
                "example": {
                    "term": "artificial intelligence",
                    "action": "keep",
                    "canonical": "artificial intelligence",
                    "notes": "",
                    "reason": "already canonical",
                },
            },
        }

        user_content = json.dumps(payload, ensure_ascii=False)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        return messages, payload

    def _build_alias_messages_candidates(
        self, cluster_id: int, subset: pd.DataFrame
    ) -> Tuple[List[Dict[str, str]], Dict[str, object]]:
        cfg = self.config
        cand_col = str(cfg.alias_candidate_column or "candidates")
        if cand_col not in subset.columns:
            raise KeyError(f"Missing candidate column '{cand_col}' for alias_strategy='llm_candidates'.")

        allow_translation = "yes" if cfg.alias_allow_translation else "no"
        stopwords = self.stopwords_set or ENGLISH_STOP_WORDS
        stopword_examples = sorted(list(stopwords))[:20]
        stopword_instruction = (
            " Pure stopword phrases (e.g., "
            + ", ".join(repr(w) for w in stopword_examples)
            + ") must be returned with action 'drop'."
        )

        def _candidate_objects(value: object) -> List[Dict[str, object]]:
            if value is None:
                return []
            if isinstance(value, np.ndarray):
                value = value.tolist()
            if isinstance(value, list):
                out: List[Dict[str, object]] = []
                for item in value:
                    if isinstance(item, Mapping):
                        term = str(item.get("term", "")).strip()
                        if not term:
                            continue
                        out.append({"term": term, **{k: v for k, v in item.items() if k != "term"}})
                    else:
                        term = str(item).strip()
                        if not term:
                            continue
                        out.append({"term": term})
                return out
            if isinstance(value, (tuple, set)):
                if isinstance(value, set):
                    iterable = sorted(value, key=lambda x: json.dumps(x, sort_keys=True, default=str))
                else:
                    iterable = value
                out: List[Dict[str, object]] = []
                for item in iterable:
                    if isinstance(item, Mapping):
                        term = str(item.get("term", "")).strip()
                        if not term:
                            continue
                        out.append({"term": term, **{k: v for k, v in item.items() if k != "term"}})
                    else:
                        term = str(item).strip()
                        if not term:
                            continue
                        out.append({"term": term})
                return out
            if isinstance(value, str):
                raw = value.strip()
                if not raw:
                    return []
                try:
                    decoded = json.loads(raw)
                    if isinstance(decoded, list):
                        return _candidate_objects(decoded)
                except Exception:
                    pass
                # Fallback: allow pipe/comma separated lists.
                sep = "|" if "|" in raw else ("," if "," in raw else None)
                if sep:
                    return [{"term": part.strip()} for part in raw.split(sep) if part.strip()]
                return [{"term": raw}]
            return []

        candidate_max = int(cfg.alias_candidate_max) if int(cfg.alias_candidate_max) > 0 else 0

        system_prompt = dedent(
            """
            You are a scientific terminology normalisation assistant.
            For each keyword decide one action: keep, merge_into, translate, or drop.

            - keep: the term is already a good canonical form.
            - merge_into: merge this term into ONE canonical form chosen from the provided candidates list for that term ONLY.
              Use merge_into ONLY when the candidate is clearly an equivalent spelling/variant/synonym of the same concept
              (e.g., plural/singular, hyphen/spacing, minor spelling variations, abbreviation expansion when unambiguous).
              Do NOT merge when the candidate looks like a broader term, a subtype, a related topic, or a contextual phrase
              (e.g., 'hiv' vs 'hiv 1', 'apoptosis' vs 'cell apoptosis').
              If candidates are provided with frequencies/df, prefer the candidate with higher df when merging.
              If the candidates list is empty or no candidate is appropriate, do not use merge_into; use keep instead.
            - translate: supply an English canonical form for non-English terms (allowed: {allow_translation}).
            - drop: remove junk terms (isolated stopwords, markup artefacts, etc.).{stopword_instruction}

            Respond with strict JSON only; no commentary.
            """
        ).format(allow_translation=allow_translation, stopword_instruction=stopword_instruction).strip()

        terms_payload: List[Dict[str, object]] = []
        for row in subset.itertuples(index=False):
            candidates = _candidate_objects(getattr(row, cand_col, None))
            if candidate_max and candidates:
                candidates = candidates[:candidate_max]
            terms_payload.append(
                {
                    "term": str(row.term),
                    "score": float(row.score),
                    "frequency": int(row.frequency),
                    "doc_coverage": int(getattr(row, "doc_coverage", 0)),
                    "candidates": candidates,
                }
            )

        payload = {
            "cluster_id": int(cluster_id),
            "allow_translation": allow_translation,
            "stopword_strictness": self.config.alias_stopword_strictness,
            "stopwords_hint": stopword_examples,
            "terms": terms_payload,
            "schema": {
                "fields": ["term", "action", "canonical", "notes", "reason"],
                "actions": ["keep", "merge_into", "translate", "drop"],
                "example": {
                    "term": "artificial intelligence",
                    "action": "keep",
                    "canonical": "artificial intelligence",
                    "notes": "",
                    "reason": "already canonical",
                },
            },
            "constraints": {
                "merge_into_requires_candidates": True,
                "candidate_column": cand_col,
                "candidate_max": candidate_max,
            },
        }

        user_content = json.dumps(payload, ensure_ascii=False)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        return messages, payload

    def _invoke_alias_model(self, client, messages: List[Dict[str, str]]) -> str:
        cfg = self.config
        response = client.chat.completions.create(
            model=cfg.alias_model,
            messages=messages,
            temperature=float(cfg.alias_temperature),
            timeout=float(cfg.alias_timeout),
        )
        return response.choices[0].message.content.strip()

    def _parse_alias_items(
        self,
        cluster_id: int,
        raw_response: str,
        subset: pd.DataFrame,
    ) -> List[Dict[str, object]]:
        cfg = self.config
        parsed = self._safe_json_loads(raw_response)
        if isinstance(parsed, dict):
            if isinstance(parsed.get("items"), list):
                items = parsed["items"]  # type: ignore[assignment]
            elif isinstance(parsed.get("terms"), list):
                items = parsed["terms"]  # type: ignore[assignment]
            else:
                items = []
        elif isinstance(parsed, list):
            items = parsed  # type: ignore[assignment]
        else:
            items = []

        instructions: List[Dict[str, object]] = []
        provided_terms = set()
        for entry in items:
            if not isinstance(entry, Mapping):
                continue
            term = str(entry.get("term", "")).strip()
            if not term:
                continue
            action = str(entry.get("action", "keep")).strip().lower()
            canonical = str(entry.get("canonical", "")).strip()
            notes = str(entry.get("notes", "")).strip()
            reason = str(entry.get("reason", "")).strip()
            instructions.append(
                {
                    "cluster_id": int(cluster_id),
                    "original": term,
                    "action": action or "keep",
                    "canonical": canonical or term,
                    "notes": notes,
                    "reason": reason,
                }
            )
            provided_terms.add(term)

        for term in subset["term"].tolist():
            term_str = str(term)
            if term_str not in provided_terms:
                instructions.append(
                    {
                        "cluster_id": int(cluster_id),
                        "original": term_str,
                        "action": "keep",
                        "canonical": term_str,
                        "notes": "",
                        "reason": "default_keep",
                    }
                )

        # Enforce candidate allowlist (and guardrails) for merge_into if enabled.
        cand_col = str(cfg.alias_candidate_column or "candidates")
        if (
            bool(cfg.alias_candidate_enforce)
            and isinstance(subset, pd.DataFrame)
            and not subset.empty
            and cand_col in subset.columns
        ):
            sep_re = re.compile(r"[-_/]+")
            boundary_re = re.compile(r"(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])")
            digit_re = re.compile(r"\d+")
            alpha_single_re = re.compile(r"^[a-z]$", flags=re.IGNORECASE)
            roman_re = re.compile(r"^(?=[ivxlcdm]+$)[ivxlcdm]+$", flags=re.IGNORECASE)
            greek_symbol_to_name = {
                "\u03b1": "alpha",
                "\u03b2": "beta",
                "\u03b3": "gamma",
                "\u03b4": "delta",
                "\u03b5": "epsilon",
                "\u03b6": "zeta",
                "\u03b7": "eta",
                "\u03b8": "theta",
                "\u03b9": "iota",
                "\u03ba": "kappa",
                "\u03bb": "lambda",
                "\u03bc": "mu",
                "\u03bd": "nu",
                "\u03be": "xi",
                "\u03bf": "omicron",
                "\u03c0": "pi",
                "\u03c1": "rho",
                "\u03c3": "sigma",
                "\u03c4": "tau",
                "\u03c5": "upsilon",
                "\u03c6": "phi",
                "\u03c7": "chi",
                "\u03c8": "psi",
                "\u03c9": "omega",
                "\u0391": "alpha",
                "\u0392": "beta",
                "\u0393": "gamma",
                "\u0394": "delta",
                "\u0395": "epsilon",
                "\u0396": "zeta",
                "\u0397": "eta",
                "\u0398": "theta",
                "\u0399": "iota",
                "\u039a": "kappa",
                "\u039b": "lambda",
                "\u039c": "mu",
                "\u039d": "nu",
                "\u039e": "xi",
                "\u039f": "omicron",
                "\u03a0": "pi",
                "\u03a1": "rho",
                "\u03a3": "sigma",
                "\u03a4": "tau",
                "\u03a5": "upsilon",
                "\u03a6": "phi",
                "\u03a7": "chi",
                "\u03a8": "psi",
                "\u03a9": "omega",
            }
            greek_names = set(greek_symbol_to_name.values())

            def _candidate_key(text: str) -> str:
                cleaned = _normalize_text_basic(text or "")
                if cfg.lowercase:
                    cleaned = cleaned.lower()
                cleaned = cleaned.strip()
                # Normalise micro symbols to ASCII (common in units).
                cleaned = cleaned.replace("µ", "u").replace("μ", "u")
                # Map Greek symbols to names so beta/β etc match.
                for sym, name in greek_symbol_to_name.items():
                    if sym in cleaned:
                        cleaned = cleaned.replace(sym, f" {name} ")
                cleaned = boundary_re.sub(" ", cleaned)
                cleaned = sep_re.sub(" ", cleaned)
                return " ".join(cleaned.split())

            def _candidate_list(value: object) -> List[str]:
                if value is None:
                    return []
                if isinstance(value, np.ndarray):
                    value = value.tolist()
                if isinstance(value, list):
                    out: List[str] = []
                    for item in value:
                        if isinstance(item, Mapping):
                            term = str(item.get("term", "")).strip()
                            if term:
                                out.append(term)
                        else:
                            term = str(item).strip()
                            if term:
                                out.append(term)
                    return out
                if isinstance(value, (tuple, set)):
                    iterable = (
                        sorted(value, key=lambda x: json.dumps(x, sort_keys=True, default=str))
                        if isinstance(value, set)
                        else value
                    )
                    out = []
                    for item in iterable:
                        if isinstance(item, Mapping):
                            term = str(item.get("term", "")).strip()
                            if term:
                                out.append(term)
                        else:
                            term = str(item).strip()
                            if term:
                                out.append(term)
                    return out
                if isinstance(value, str):
                    raw = value.strip()
                    if not raw:
                        return []
                    try:
                        decoded = json.loads(raw)
                        if isinstance(decoded, list):
                            return _candidate_list(decoded)
                    except Exception:
                        pass
                    sep = "|" if "|" in raw else ("," if "," in raw else None)
                    if sep:
                        return [part.strip() for part in raw.split(sep) if part.strip()]
                    return [raw]
                return []

            def _append_reason(existing: str, extra: str) -> str:
                existing = (existing or "").strip()
                extra = (extra or "").strip()
                if not existing:
                    return extra
                if not extra:
                    return existing
                if extra in existing:
                    return existing
                return f"{existing};{extra}"

            def _specifier_set(text: str) -> set[str]:
                """Extract specifier tokens that should not be dropped when merging.

                Examples: digits (e.g., IL-6), single-letter classes (IgG), Greek letters (alpha/beta), roman numerals (II).
                """
                key = _candidate_key(text or "")
                if not key:
                    return set()
                specs: set[str] = set()
                for tok in key.split():
                    if not tok:
                        continue
                    if digit_re.search(tok):
                        specs.update(digit_re.findall(tok))
                        continue
                    tok_lower = tok.lower()
                    if alpha_single_re.fullmatch(tok_lower):
                        specs.add(tok_lower)
                        continue
                    if tok_lower in greek_names:
                        specs.add(tok_lower)
                        continue
                    if len(tok_lower) >= 2 and roman_re.fullmatch(tok_lower):
                        specs.add(tok_lower)
                return specs

            candidate_max = int(cfg.alias_candidate_max) if int(cfg.alias_candidate_max) > 0 else 0
            term_key_to_terms: Dict[str, List[str]] = defaultdict(list)
            candidates_by_term: Dict[str, Dict[str, str]] = {}
            for row in subset.itertuples(index=False):
                term_raw = str(getattr(row, "term", "")).strip()
                if not term_raw:
                    continue
                term_key_to_terms[_candidate_key(term_raw)].append(term_raw)
                candidates = _candidate_list(getattr(row, cand_col, None))
                if candidate_max and candidates:
                    candidates = candidates[:candidate_max]
                cand_map: Dict[str, str] = {}
                for cand in candidates:
                    cand_key = _candidate_key(cand)
                    if cand_key and cand_key not in cand_map:
                        cand_map[cand_key] = cand
                candidates_by_term[term_raw] = cand_map

            for inst in instructions:
                action = str(inst.get("action", "keep")).strip().lower()
                if action != "merge_into":
                    continue
                original = str(inst.get("original", "")).strip()
                if not original:
                    continue
                canonical = str(inst.get("canonical", "")).strip()

                # Locate the original term in this subset (exact or normalized unique match).
                source_term: Optional[str] = original if original in candidates_by_term else None
                if source_term is None:
                    key = _candidate_key(original)
                    matches = term_key_to_terms.get(key) or []
                    if len(matches) == 1:
                        source_term = matches[0]
                if source_term is None:
                    continue

                cand_map = candidates_by_term.get(source_term) or {}
                if not cand_map:
                    inst["action"] = "keep"
                    inst["canonical"] = source_term
                    inst["reason"] = _append_reason(str(inst.get("reason", "")), "no_candidates")
                    continue

                if not canonical:
                    inst["action"] = "keep"
                    inst["canonical"] = source_term
                    inst["reason"] = _append_reason(str(inst.get("reason", "")), "empty_canonical")
                    continue

                if canonical.strip() == source_term.strip():
                    inst["action"] = "keep"
                    inst["canonical"] = source_term
                    inst["reason"] = _append_reason(str(inst.get("reason", "")), "merge_into_self")
                    continue

                cand_key = _candidate_key(canonical)
                if cand_key not in cand_map:
                    inst["action"] = "keep"
                    inst["canonical"] = source_term
                    inst["reason"] = _append_reason(str(inst.get("reason", "")), "canonical_not_in_candidates")
                    continue

                # Snap canonical to the exact candidate string.
                snapped = cand_map[cand_key]
                if _specifier_set(source_term) != _specifier_set(snapped):
                    inst["action"] = "keep"
                    inst["canonical"] = source_term
                    inst["reason"] = _append_reason(str(inst.get("reason", "")), "specifier_mismatch")
                    continue
                inst["canonical"] = snapped

        return instructions

    def _apply_alias_instructions(self, top_df: pd.DataFrame, alias_df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        alias_df = alias_df.copy()
        alias_df["action"] = alias_df["action"].str.strip().str.lower()
        alias_df["canonical"] = alias_df["canonical"].astype(str)

        records: List[Dict[str, object]] = []
        top_k = max(1, int(cfg.top_n_keywords))

        for cluster_id, group in top_df.groupby("cluster_id", sort=False):
            cluster_alias = alias_df[alias_df["cluster_id"] == cluster_id]
            instruction_map = {
                str(row.original): row
                for row in cluster_alias.itertuples(index=False)
            }
            buckets: Dict[str, Dict[str, object]] = {}

            for row in group.itertuples(index=False):
                original_term = str(row.term)
                inst = instruction_map.get(original_term)
                action = getattr(inst, "action", "keep") if inst else "keep"
                canonical_raw = getattr(inst, "canonical", original_term) if inst else original_term
                built_alias = self._builtin_alias(original_term)
                if built_alias:
                    canonical_raw = built_alias
                    if self._clean_canonical_term(built_alias) != self._clean_canonical_term(original_term):
                        action = "translate"
                cleaned = self._clean_canonical_term(canonical_raw)

                if not cleaned or action == "drop":
                    continue

                # Optional stopword strictness
                tokens = cleaned.split()
                stopwords = self.stopwords_set or ENGLISH_STOP_WORDS
                if (
                    cfg.alias_stopword_strictness == "drop_if_empty"
                    and tokens
                    and all(token in stopwords for token in tokens)
                ):
                    continue

                bucket = buckets.setdefault(
                    cleaned,
                    {
                        "cluster_id": int(cluster_id),
                        "canonical_raw": canonical_raw,
                        "canonical_cleaned": cleaned,
                        "items": [],
                        "notes": [],
                        "reasons": [],
                        "actions": [],
                    },
                )
                if canonical_raw and not bucket.get("canonical_raw"):
                    bucket["canonical_raw"] = canonical_raw
                bucket["items"].append(row)
                if inst:
                    note = getattr(inst, "notes", "")
                    reason = getattr(inst, "reason", "")
                    if note:
                        bucket["notes"].append(str(note))
                    if reason:
                        bucket["reasons"].append(str(reason))
                    bucket["actions"].append(action)
                else:
                    bucket["actions"].append("keep")

            aggregated_rows: List[Dict[str, object]] = []
            for canonical_cleaned, payload in buckets.items():
                items = payload["items"]
                if not items:
                    continue
                freq_array = np.array([int(item.frequency) for item in items], dtype=float)
                score_array = np.array([float(item.score) for item in items], dtype=float)
                doc_cov_array = np.array([int(getattr(item, "doc_coverage", 0)) for item in items], dtype=int)

                total_freq = int(freq_array.sum())
                combined_score = float(np.average(score_array, weights=freq_array)) if freq_array.sum() > 0 else float(score_array.mean())
                doc_coverage = int(doc_cov_array.max()) if doc_cov_array.size else 0
                source_terms = sorted({str(item.term) for item in items})

                canonical_raw = str(payload["canonical_raw"]).strip()
                display_term = str(payload.get("canonical_cleaned") or canonical_cleaned)

                notes_unique = sorted({note for note in payload["notes"] if note})
                reasons_unique = sorted({reason for reason in payload["reasons"] if reason})

                aggregated_rows.append(
                    {
                        "cluster_id": int(cluster_id),
                        "term": str(display_term),
                        "score": combined_score,
                        "frequency": total_freq,
                        "doc_coverage": doc_coverage,
                        "source_terms": source_terms,
                        "alias_actions": sorted(set(payload["actions"])),
                        "alias_notes": "; ".join(notes_unique) if notes_unique else "",
                        "alias_reason": "; ".join(reasons_unique) if reasons_unique else "",
                    }
                )

            if not aggregated_rows:
                for row in group.itertuples(index=False):
                    records.append(
                        {
                            "cluster_id": int(cluster_id),
                            "term": str(row.term),
                            "score": float(row.score),
                            "frequency": int(row.frequency),
                            "doc_coverage": int(getattr(row, "doc_coverage", 0)),
                            "source_terms": [str(row.term)],
                            "alias_actions": ["keep"],
                            "alias_notes": "",
                            "alias_reason": "fallback_keep",
                        }
                    )
                continue

            aggregated_rows.sort(key=lambda item: -float(item["score"]))
            for row in aggregated_rows[:top_k]:
                records.append(row)

        if not records:
            return top_df.assign(source_terms=top_df["term"].apply(lambda t: [str(t)]))
        result = pd.DataFrame.from_records(records)
        result = self._merge_equivalent_source_terms(result)
        self._log(
            "Stage 2.5: aggregated %d rows from %d input rows",
            len(result),
            len(top_df),
        )
        return result

    def _merge_equivalent_source_terms(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or "source_terms" not in df.columns:
            return df

        frame = df.copy()

        def _normalise_sources(value: object) -> Tuple[str, ...]:
            if value is None:
                return tuple()
            if isinstance(value, str):
                token = value.strip()
                return (token,) if token else tuple()
            if isinstance(value, (list, tuple, set)):
                tokens = {str(item).strip() for item in value if str(item).strip()}
                return tuple(sorted(tokens))
            return tuple()

        frame["_source_key"] = frame["source_terms"].apply(_normalise_sources)
        if frame["_source_key"].map(bool).sum() <= 1:
            return frame.drop(columns="_source_key")

        merged_rows: List[pd.Series] = []
        dict_sum_cols = {"pub_year_series", "year_denominators"}
        dict_passthrough_cols = {"ppm_series", "loglift_series", "bayesian_log_odds_series"}

        for (cluster_id, key), group in frame.groupby(["cluster_id", "_source_key"], sort=False):
            if not key or len(group) == 0:
                row = group.iloc[0].drop(labels="_source_key")
                merged_rows.append(row)
                continue
            if len(group) == 1:
                row = group.iloc[0].drop(labels="_source_key")
                merged_rows.append(row)
                continue

            group_local = group.copy()

            if "alias_actions" in group_local.columns:
                def _priority_value(actions: object) -> int:
                    if isinstance(actions, (list, tuple, set)):
                        action_set = {str(a).strip().lower() for a in actions if str(a).strip()}
                    elif isinstance(actions, str):
                        action_set = {actions.strip().lower()} if actions.strip() else set()
                    else:
                        action_set = set()
                    if "translate" in action_set:
                        return 2
                    if "merge_into" in action_set:
                        return 1
                    return 0

                group_local["_priority"] = group_local["alias_actions"].apply(_priority_value)
                group_sorted = group_local.sort_values(
                    by=["_priority", "score"],
                    ascending=[False, False],
                    kind="mergesort",
                )
            else:
                group_sorted = group_local.sort_values("score", ascending=False, kind="mergesort")

            base = group_sorted.iloc[0].copy()

            if "frequency" in group.columns:
                freq_series = group["frequency"].fillna(0)
                total_freq = float(freq_series.sum())
                base["frequency"] = int(total_freq)
            else:
                freq_series = None
                total_freq = 0.0

            if "score" in group.columns:
                if freq_series is not None and freq_series.sum() > 0:
                    base["score"] = float(np.average(group["score"].to_numpy(), weights=freq_series.to_numpy()))
                else:
                    base["score"] = float(group["score"].mean())

            if "doc_coverage" in group.columns:
                base["doc_coverage"] = int(group["doc_coverage"].max())

            base["source_terms"] = list(key)

            if "alias_actions" in group.columns:
                action_set: set[str] = set()
                for actions in group["alias_actions"]:
                    if isinstance(actions, (list, tuple, set)):
                        action_set.update(str(a).strip() for a in actions if str(a).strip())
                    elif isinstance(actions, str) and actions.strip():
                        action_set.add(actions.strip())
                base["alias_actions"] = sorted(action_set) if action_set else ["keep"]

            for col, sep in (("alias_notes", "; "), ("alias_reason", "; ")):
                if col in group.columns:
                    fragments: List[str] = []
                    for value in group[col]:
                        if isinstance(value, str) and value.strip():
                            parts = [part.strip() for part in value.split(";") if part.strip()]
                            fragments.extend(parts)
                    base[col] = sep.join(sorted(set(fragments))) if fragments else ""

            for col in dict_sum_cols & set(group.columns):
                merged: Dict[int, float] = {}
                for value in group[col]:
                    if isinstance(value, Mapping):
                        for k, v in value.items():
                            try:
                                key_int = int(k)
                            except Exception:
                                continue
                            try:
                                merged[key_int] = merged.get(key_int, 0.0) + float(v)
                            except Exception:
                                continue
                if merged:
                    merged_int = {int(k): int(v) if float(v).is_integer() else float(v) for k, v in merged.items()}
                    base[col] = dict(sorted(merged_int.items()))

            for col in dict_passthrough_cols & set(group.columns):
                base[col] = group_sorted.iloc[0][col]

            for extra_col in ("_source_key", "_priority"):
                if extra_col in base:
                    base = base.drop(labels=extra_col)
            merged_rows.append(base)

        result = pd.DataFrame(merged_rows).reindex(columns=df.columns)

        def _priority_value(actions: object) -> int:
            action_set: set[str]
            if isinstance(actions, (list, tuple, set)):
                action_set = {str(a).strip().lower() for a in actions if str(a).strip()}
            elif isinstance(actions, str):
                action_set = {actions.strip().lower()} if actions.strip() else set()
            else:
                action_set = set()
            if "translate" in action_set:
                return 3
            if "merge_into" in action_set:
                return 2
            if "drop" in action_set:
                return 1
            return 0

        result["_source_key"] = result["source_terms"].apply(_normalise_sources)
        if not result.empty:
            result = (
                result.assign(_priority=result["alias_actions"].apply(_priority_value),
                              _score=result.get("score", pd.Series(dtype=float)))
                .sort_values(
                    by=["cluster_id", "_source_key", "_priority", "_score"],
                    ascending=[True, True, False, False],
                    kind="mergesort",
                )
                .drop_duplicates(subset=["cluster_id", "_source_key"], keep="first")
                .drop(columns=["_priority", "_score"])
            )
        result = result.drop(columns="_source_key", errors="ignore")

        if not result.empty:
            def _has_non_keep(actions: object) -> bool:
                if isinstance(actions, (list, tuple, set)):
                    return any(str(a).strip().lower() != "keep" for a in actions)
                if isinstance(actions, str):
                    return str(actions).strip().lower() != "keep"
                return False

            result["_source_fset"] = result["source_terms"].apply(
                lambda terms: tuple(sorted(str(t).strip() for t in terms)) if isinstance(terms, (list, tuple, set)) else tuple()
            )
            drop_indices: List[int] = []
            for (cluster_id, key), group in result.groupby(["cluster_id", "_source_fset"], sort=False):
                if not key or len(group) <= 1:
                    continue
                preferred = group[group["alias_actions"].apply(_has_non_keep)]
                if preferred.empty:
                    keeper_idx = group.sort_values("score", ascending=False, kind="mergesort").index[0]
                else:
                    keeper_idx = preferred.sort_values("score", ascending=False, kind="mergesort").index[0]
                drop_indices.extend(idx for idx in group.index if idx != keeper_idx)
            if drop_indices:
                result = result.drop(index=drop_indices)
            result = result.drop(columns="_source_fset", errors="ignore")

        return result

    def _rescore_with_aliases(self, canonical_df: pd.DataFrame) -> pd.DataFrame:
        if canonical_df.empty:
            return canonical_df

        canonical_df = canonical_df.loc[:, ~pd.Index(canonical_df.columns).duplicated()]
        allowed_pairs: set[Tuple[int, str]] = set()
        for row in canonical_df.itertuples(index=False):
            try:
                cid = int(row.cluster_id)
            except Exception:
                continue
            term_key = str(row.term)
            if term_key:
                allowed_pairs.add((cid, term_key))

        def _combine_list(series: pd.Series) -> List[str]:
            values: List[str] = []
            for val in series.dropna():
                if isinstance(val, list):
                    values.extend(str(v).strip() for v in val if str(v).strip())
                else:
                    sval = str(val).strip()
                    if sval:
                        values.append(sval)
            return sorted(set(values)) if values else []

        def _combine_text(series: pd.Series) -> str:
            unique = sorted({str(val).strip() for val in series if isinstance(val, str) and val.strip()})
            return "; ".join(unique)

        metadata_aggs: Dict[str, object] = {}
        if "source_terms" in canonical_df.columns:
            metadata_aggs["source_terms"] = _combine_list
        if "alias_actions" in canonical_df.columns:
            metadata_aggs["alias_actions"] = _combine_list
        if "alias_notes" in canonical_df.columns:
            metadata_aggs["alias_notes"] = _combine_text
        if "alias_reason" in canonical_df.columns:
            metadata_aggs["alias_reason"] = _combine_text

        if metadata_aggs:
            metadata_group = (
                canonical_df.groupby(["cluster_id", "term"], sort=False)
                .agg(metadata_aggs)
                .reset_index()
            )
        else:
            metadata_group = canonical_df[["cluster_id", "term"]].drop_duplicates().copy()

        old_C = self.C_all
        feature_names = self.feature_names_all
        if old_C is None or feature_names is None:
            return canonical_df
        old_DF = self.DF_all

        feature_index = {str(term): idx for idx, term in enumerate(feature_names)}
        alias_targets: Dict[str, set[int]] = {}
        for row in canonical_df.itertuples(index=False):
            canonical = str(row.term)
            sources = getattr(row, "source_terms", None)
            if not sources:
                sources = [canonical]
            indices: List[int] = []
            for token in sources:
                idx = feature_index.get(str(token))
                if idx is not None:
                    indices.append(idx)
            if indices:
                alias_targets.setdefault(canonical, set()).update(indices)

        alias_targets = {canonical: sorted(idx_set) for canonical, idx_set in alias_targets.items()}

        missing_terms = set()
        if metadata_aggs:
            missing_terms = {
                str(row.term)
                for row in metadata_group.itertuples(index=False)
                if str(row.term) not in alias_targets
            }

        if not alias_targets:
            self._log("Stage 2.5: no canonical columns to rescore; keeping existing scores")
            return canonical_df

        merged_cols: List[sp.csr_matrix] = []
        merged_df_cols: List[sp.csr_matrix] = []
        merged_names: List[str] = []

        for canonical, indices in alias_targets.items():
            summed_counts = old_C[:, indices].sum(axis=1)
            merged_cols.append(sp.csr_matrix(summed_counts))
            if old_DF is not None:
                summed_docs = old_DF[:, indices].sum(axis=1)
                merged_df_cols.append(sp.csr_matrix(summed_docs))
            merged_names.append(canonical)

        C_canonical = sp.hstack(merged_cols, format="csr").astype(np.int64)
        DF_canonical = (
            sp.hstack(merged_df_cols, format="csr").astype(np.int64)
            if merged_df_cols else None
        )

        prev_n_jobs = self.n_jobs_effective
        self.n_jobs_effective = 1
        self._alias_client = None

        self.C_all = C_canonical
        self.DF_all = DF_canonical
        self.feature_names_all = np.array(merged_names, dtype=str)

        scores = self._compute_c_tfidf(C_canonical)
        df_global = (
            np.asarray(DF_canonical.sum(axis=0)).ravel().astype(np.int64)
            if DF_canonical is not None else None
        )
        reranked = self._rank_topk(
            C_canonical,
            scores,
            self.feature_names_all,
            DF_canonical,
            df_global,
        )
        if allowed_pairs:
            mask_allowed: List[bool] = []
            for cid, term in zip(reranked["cluster_id"], reranked["term"]):
                try:
                    key = (int(cid), str(term))
                except Exception:
                    mask_allowed.append(False)
                    continue
                mask_allowed.append(key in allowed_pairs)
            reranked = reranked.loc[mask_allowed].reset_index(drop=True)

        self.n_jobs_effective = prev_n_jobs

        metadata_cols = [col for col in metadata_group.columns if col not in {"cluster_id", "term"}]
        merged = reranked.merge(metadata_group, on=["cluster_id", "term"], how="left")

        if missing_terms:
            missing_rows = (
                canonical_df[canonical_df["term"].astype(str).isin(missing_terms)]
                .drop_duplicates(subset=["cluster_id", "term"], keep="first")
                .merge(metadata_group, on=["cluster_id", "term"], how="left")
            )
            if not missing_rows.empty:
                merged = pd.concat([merged, missing_rows], ignore_index=True, sort=False)

        def _collapse_suffix_columns(df: pd.DataFrame) -> pd.DataFrame:
            result = df
            for suffix in ("_x", "_y"):
                suffix_cols = [col for col in result.columns if col.endswith(suffix)]
                for col in suffix_cols:
                    base = col[:-2]
                    if base in result.columns:
                        result[base] = result[base].where(result[base].notna(), result[col])
                        result = result.drop(columns=col)
                    else:
                        result = result.rename(columns={col: base})
            return result

        merged = _collapse_suffix_columns(merged)

        for col in ("frequency", "doc_coverage"):
            if col in merged.columns:
                merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0).astype(int)

        if "source_terms" in merged.columns:
            merged["source_terms"] = merged.apply(
                lambda row: row.source_terms
                if isinstance(row.source_terms, list) and row.source_terms
                else [str(row.term)],
                axis=1,
            )
        if "alias_actions" in merged.columns:
            merged["alias_actions"] = merged.apply(
                lambda row: row.alias_actions
                if isinstance(row.alias_actions, list) and row.alias_actions
                else ["keep"],
                axis=1,
            )
        if "alias_notes" in merged.columns:
            merged["alias_notes"] = merged["alias_notes"].fillna("")
        if "alias_reason" in merged.columns:
            merged["alias_reason"] = merged["alias_reason"].fillna("")

        return merged

    def _clean_canonical_term(self, term: str) -> str:
        cleaned = _normalize_text_basic(term or "")
        if self.config.lowercase:
            cleaned = cleaned.lower()
        cleaned = cleaned.strip()
        # Normalise micro symbols to ASCII
        cleaned = cleaned.replace("µ", "u").replace("μ", "u")
        # Very light singularisation for a few measurement units
        unit_roots = ("becquerel", "sievert", "gray", "curie")
        unit_prefixes = ("", "kilo", "mega", "giga", "milli", "micro")
        for root in unit_roots:
            singular = None
            for prefix in unit_prefixes:
                plural = f"{prefix}{root}s"
                if cleaned == plural:
                    singular = f"{prefix}{root}"
                    break
            if singular:
                cleaned = singular
                break
        return cleaned

    def _builtin_alias(self, value: str) -> Optional[str]:
        if not self.config.builtin_aliases:
            return None
        if self._builtin_alias_cache is None:
            try:
                self._builtin_alias_cache = {
                    str(k).strip().lower(): str(v).strip()
                    for k, v in self.config.builtin_aliases.items()
                }
            except Exception:
                self._builtin_alias_cache = {}
        key = str(value).strip().lower()
        return self._builtin_alias_cache.get(key)

    def _enforce_forbidden(self, df: pd.DataFrame) -> pd.DataFrame:
        forbidden = {s.lower() for s in (self.config.forbid_abbreviations or ())}
        if not forbidden or df.empty:
            return df
        out = df.copy()
        terms_lower = out["term"].astype(str).str.lower()
        mask = terms_lower.isin(forbidden)
        if not mask.any():
            return out

        modified_indices: List[int] = []
        for idx in out.index[mask]:
            replacement = self._builtin_alias(out.at[idx, "term"])
            if not replacement:
                # no replacement available; drop the row
                continue
            new_term = self._clean_canonical_term(replacement)
            out.at[idx, "term"] = new_term
            # ensure alias_actions contains 'translate'
            actions = out.at[idx, "alias_actions"]
            if isinstance(actions, (list, tuple, set)):
                updated = {str(a).strip().lower() for a in actions if str(a).strip()}
            elif isinstance(actions, str) and actions.strip():
                updated = {actions.strip().lower()}
            else:
                updated = set()
            updated.add("translate")
            out.at[idx, "alias_actions"] = sorted(updated)
            modified_indices.append(idx)

        # remove any rows that still have forbidden canonical terms
        out = out[~out["term"].astype(str).str.lower().isin(forbidden)]

        # normalise duplicates after replacements
        if modified_indices:
            out = self._merge_equivalent_source_terms(out)

        return out.reset_index(drop=True)

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

    @staticmethod
    def _safe_json_loads(raw: str) -> Optional[object]:
        if raw is None:
            return None
        text = str(raw).strip()
        if not text:
            return None

        if text.startswith("```"):
            text = re.sub(r"^```[\w-]*\s*", "", text, count=1)
            text = re.sub(r"\s*```$", "", text, count=1).strip()

        candidates = [text]
        if "{" in text:
            candidates.append(text[text.find("{") :])
        if "[" in text:
            candidates.append(text[text.find("[") :])
        for candidate in candidates:
            try:
                return json.loads(candidate)
            except Exception:  # pragma: no cover
                continue
        return None

    def _get_alias_client(self):
        if self._alias_client is not None:
            return self._alias_client
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "openai package is required for alias_strategy in {'llm', 'llm_candidates'}. "
                "Install via `pip install openai`."
            ) from exc
        kwargs: Dict[str, object] = {}
        if self.config.alias_base_url:
            kwargs["base_url"] = self.config.alias_base_url
        api_key = self.config.alias_api_key or os.getenv("OPENAI_API_KEY", "ollama")
        kwargs["api_key"] = api_key
        client = OpenAI(**kwargs)
        self._alias_client = client
        self._log("Stage 2.5: alias client created (model=%s)", self.config.alias_model)
        return client

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
