"""Keyword extraction pipeline orchestrator.

Coordinates vectorization, aggregation, scoring, canonicalization, and
temporal metrics into a single run() call.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import joblib
from joblib import Parallel, delayed
from scipy import sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

from .config import KeywordExtractionConfig, KeywordRecord
from .extraction import (
    _DataSource,
    _argpartition_topk,
    _effective_n_jobs,
    _group_sum_by_cluster,
    _llr_2x2,
    _mmr_jaccard_select,
    _suppress_subphrases,
)
from .llm_canonicalize import LLMCanonicalizeMixin
from .temporal import TemporalMixin

logger = logging.getLogger(__name__)


class KeywordExtractionPipeline(LLMCanonicalizeMixin, TemporalMixin):
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

    # ----- Stage 2: compute c-TF-IDF and select top terms -----

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

    # ----- Checkpointing -----

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
        """Restore Stage 0-2 artefacts from disk and return the saved Stage 2 dataframe."""
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
