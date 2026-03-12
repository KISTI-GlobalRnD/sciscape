"""Keyword extraction pipeline orchestrator.

Coordinates vectorization, aggregation, scoring, canonicalization, and
temporal metrics into a single run() call.
"""

from __future__ import annotations

import math
import re
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

from .config import KeywordExtractionConfig
from .extraction import (
    _DataSource,
    _argpartition_topk,
    _effective_n_jobs,
    _group_sum_by_cluster,
    _llr_2x2,
    _mmr_jaccard_select,
    _suppress_subphrases,
)
from .cooccurrence import collect_cooccurrence
from .depth import estimate_depth
from .llm_canonicalize import LLMCanonicalizeMixin
from .normalization import normalize_keywords
from .temporal import TemporalMixin
from .term_network import TermNetwork, TermNetworkConfig
from .vocab_merge import apply_merge_map, build_merge_map

logger = logging.getLogger(__name__)

# Domain-agnostic academic boilerplate — terms that appear across most clusters
# in scientific papers regardless of topic.
ACADEMIC_STOPWORDS: frozenset = frozenset({
    "based", "using", "used", "results", "proposed", "propose",
    "method", "methods", "model", "approach", "study",
    "analysis", "system", "applied", "presented", "performance",
    "novel", "paper", "show", "shown", "experimental", "compared",
    "improved", "obtained", "investigated", "developed", "discussed",
    "considered", "problem", "solution", "technique", "framework",
    "scheme", "process", "different", "similar", "large", "high",
    "low", "new", "first", "two", "three",
})


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
        self._builtin_alias_cache: Optional[Dict[str, str]] = None
        self._init_alias_cache()

        # Quality filters (P1/P5/P6)
        self._academic_sw: frozenset = ACADEMIC_STOPWORDS
        if config.academic_stopwords_extra:
            self._academic_sw = self._academic_sw | frozenset(
                s.lower() for s in config.academic_stopwords_extra
            )
        self._artifact_res: List[re.Pattern] = []
        if config.artifact_filter_enabled:
            self._artifact_res = [
                re.compile(p, re.IGNORECASE) for p in config.artifact_filter_patterns
            ]

        # Optional stage artifacts (populated during run)
        self.cooc_matrix: Optional[sp.csr_matrix] = None
        self.merge_candidates: Optional[List[Dict]] = None

        self._log("Initialised pipeline for %d clusters (n_jobs=%d)", self.K, self.n_jobs_effective)

    def _init_alias_cache(self) -> None:
        """Set up alias cache directory if caching is enabled."""
        cfg = self.config
        if not (cfg.apply_alias_map
                and (cfg.alias_strategy or "none").lower() != "none"
                and cfg.alias_cache_enabled):
            return
        if cfg.alias_cache_path is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            cfg.alias_cache_path = Path("artifacts") / "canonicalise" / timestamp
        base = Path(cfg.alias_cache_path)
        base.mkdir(parents=True, exist_ok=True)
        self._alias_cache_dir = base

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

    def _is_artifact(self, term: str) -> bool:
        """P5: Check if term matches an artifact pattern (LaTeX noise, pure numbers, etc.)."""
        return any(pat.search(term) for pat in self._artifact_res)

    def _is_academic_stopword(self, term: str) -> bool:
        """P1: Check if a *single-token* term is academic boilerplate.

        Multi-word terms like "fault diagnosis" pass through even if they
        contain an academic stopword token.
        """
        return " " not in term and term.lower() in self._academic_sw

    # ----- Stage 1 (vectorization): fit vectorisers on streamed text -----

    def _fit_vectorizers(self) -> None:
        cfg = self.config
        self._log("Stage 1 (vectorization): fitting vectorisers (include_title=%s, author_keywords=%s)",
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
        self._log("Stage 1 (vectorization): unigram vocabulary size = %d", len(self.feature_names_uni))

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
                "Stage 1 (vectorization): phrase vocabulary size = %d (ngram_range=%s)",
                len(self.feature_names_phrase),
                (cfg.ngram_min, cfg.ngram_max),
            )
        else:
            self.vec_phrase = None
            self.feature_names_phrase = np.array([], dtype=str)
            self._log("Stage 1 (vectorization): phrase vectoriser disabled or empty output")

    # ----- Stage 3 (aggregation): aggregate document counts to cluster counts -----

    def _aggregate_counts(self) -> None:
        self._log("Stage 3 (aggregation): aggregating counts per cluster...")
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

        C_phrase, DF_phrase = self._filter_rare_phrases(C_phrase, DF_phrase, K)

        self.C_uni = C_uni
        self.C_phrase = C_phrase
        self.DF_uni = DF_uni
        self.DF_phrase = DF_phrase
        self.cluster_doc_counts = cluster_doc_counts
        self.total_docs = total_docs
        self._log(
            "Stage 3 (aggregation): processed %d documents across %d clusters (total tokens: %d)",
            self.total_docs,
            self.K,
            int(C_uni.sum()) + int(C_phrase.sum()) if C_phrase is not None else int(C_uni.sum()),
        )

    def _filter_rare_phrases(
        self,
        C_phrase: Optional[sp.csr_matrix],
        DF_phrase: Optional[sp.csr_matrix],
        K: int,
    ) -> Tuple[Optional[sp.csr_matrix], Optional[sp.csr_matrix]]:
        """Drop phrase columns below min count threshold."""
        if C_phrase is None or C_phrase.shape[1] == 0 or self.config.phrase_min_count_per_cluster <= 1:
            return C_phrase, DF_phrase
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
            self.feature_names_phrase = np.array([], dtype=str)
            return sp.csr_matrix((K, 0), dtype=np.int64), (
                sp.csr_matrix((K, 0), dtype=np.int64) if DF_phrase is not None else None
            )
        if not np.all(keep):
            C_phrase = C_phrase[:, keep]
            if DF_phrase is not None:
                DF_phrase = DF_phrase[:, keep]
            self.feature_names_phrase = self.feature_names_phrase[keep]  # type: ignore
        return C_phrase, DF_phrase

    # ----- Stage 2 (vocab_merge): optional vocabulary merge -----

    def _apply_vocab_merge(self) -> None:
        """Merge vocabulary columns for plural/hyphen variants on aggregated matrices."""
        vm_cfg = self.config.vocab_merge
        if vm_cfg is None or not vm_cfg.enabled:
            return
        if self.feature_names_uni is None or self.C_uni is None:
            return

        merge_map = build_merge_map(self.feature_names_uni, vm_cfg, C=self.C_uni)
        if not merge_map:
            self._log("Vocab merge: no mergeable pairs found")
            return

        orig_size = len(self.feature_names_uni)
        self._log("Vocab merge: merging %d pairs in unigram vocabulary", len(merge_map))

        # Apply same merge_map to both count and doc-frequency matrices
        self.C_uni, self.feature_names_uni = apply_merge_map(
            self.C_uni, self.feature_names_uni, merge_map
        )
        if self.DF_uni is not None:
            # Use a dummy names array since we already updated feature_names_uni above
            dummy_names = np.arange(self.DF_uni.shape[1])
            self.DF_uni, _ = apply_merge_map(self.DF_uni, dummy_names, merge_map)

        self._log("Vocab merge: unigram vocabulary %d -> %d terms", orig_size, len(self.feature_names_uni))

    # ----- Stage 4 (scoring): compute c-TF-IDF and select top terms -----

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

        # P6: pre-compute per-term cluster frequency for cross-cluster penalty
        cluster_freq: Optional[np.ndarray] = None
        if self.config.cross_cluster_penalty_enabled and DF_all is not None:
            cluster_freq = np.asarray((DF_all > 0).astype(bool).sum(axis=0)).ravel()

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
            term_vocab_indices: List[int] = []  # original vocab index for P6
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
                # P5: artifact filter
                if self._artifact_res and self._is_artifact(term):
                    continue
                # P1: academic stopword filter (unigrams only)
                if self.config.academic_stopwords_enabled and self._is_academic_stopword(term):
                    continue
                terms.append(term)
                term_vocab_indices.append(idx)
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

            # P6: cross-cluster penalty — downweight terms appearing in many clusters
            if cluster_freq is not None:
                min_cc = self.config.cross_cluster_penalty_min_count
                pen_fn = self.config.cross_cluster_penalty_fn
                for i, vidx in enumerate(term_vocab_indices):
                    cc = int(cluster_freq[vidx])
                    if cc >= min_cc:
                        if pen_fn == "log_inverse":
                            final_scores[i] /= (1 + math.log(cc))
                        else:  # "inverse"
                            final_scores[i] /= cc

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
        self._log("Stage 4 (scoring): scoring terms (top_n=%d)", self.config.top_n_keywords)
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
        self._log("Stage 4 (scoring): produced %d keyword rows", len(result))
        return result

    # ----- Checkpointing -----

    # Stage order for resume: stage name → numeric index
    _STAGE_ORDER: Dict[str, int] = {
        "scoring": 4,
        "normalization": 5,
        "cooccurrence": 6,
        "term_network": 7,
        "canonicalize": 8,
    }

    def save_checkpoint(self, directory: Path, top_df: pd.DataFrame, stage: str = "scoring") -> None:
        """Save pipeline state after a given stage for later resumption.

        Parameters
        ----------
        directory : Path
            Directory to save artifacts into (created if needed).
        top_df : pd.DataFrame
            Current keyword DataFrame at this stage.
        stage : str
            Stage name (scoring, normalization, cooccurrence, term_network, canonicalize).
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        # Always save: vectorizers, matrices, cluster info
        joblib.dump(self.vec_uni, directory / "vec_uni.joblib")
        if self.vec_phrase is not None:
            joblib.dump(self.vec_phrase, directory / "vec_phrase.joblib")

        for name, mat in [
            ("C_uni", self.C_uni), ("C_phrase", self.C_phrase), ("C_all", self.C_all),
            ("DF_uni", self.DF_uni), ("DF_phrase", self.DF_phrase), ("DF_all", self.DF_all),
        ]:
            if mat is not None:
                sp.save_npz(directory / f"{name}.npz", mat)

        for name, arr in [
            ("cluster_ids", self.cluster_ids),
            ("cluster_doc_counts", self.cluster_doc_counts),
            ("feature_names_all", self.feature_names_all),
            ("feature_names_uni", self.feature_names_uni),
            ("feature_names_phrase", self.feature_names_phrase),
        ]:
            if arr is not None:
                np.save(directory / f"{name}.npy", arr)

        # Co-occurrence matrix (available after stage 6+)
        if self.cooc_matrix is not None:
            sp.save_npz(directory / "cooc_matrix.npz", self.cooc_matrix)

        # Keyword DataFrame — convert dict/list columns to JSON for parquet compat
        import json as _json
        save_df = top_df.copy()
        _DICT_COLS = ("pub_year_series", "year_denominators", "ppm_series",
                      "loglift_series", "bayesian_log_odds_series")
        for col in _DICT_COLS:
            if col in save_df.columns:
                save_df[col] = save_df[col].apply(
                    lambda v: _json.dumps(v) if isinstance(v, (dict, list)) else v
                )
        if "source_terms" in save_df.columns:
            save_df["source_terms"] = save_df["source_terms"].apply(
                lambda v: _json.dumps(v) if isinstance(v, list) else v
            )
        if "candidates" in save_df.columns:
            save_df["candidates"] = save_df["candidates"].apply(
                lambda v: _json.dumps(v) if isinstance(v, list) else v
            )
        save_df.to_parquet(directory / "top_df.parquet", index=False)

        # Stage metadata
        import json
        meta = {"stage": stage, "n_terms": len(top_df)}
        (directory / "checkpoint_meta.json").write_text(json.dumps(meta))
        self._log("Checkpoint saved at stage '%s' -> %s", stage, directory)

    def load_checkpoint(self, directory: Path) -> Tuple[pd.DataFrame, str]:
        """Load pipeline state from a checkpoint directory.

        Returns
        -------
        tuple of (top_df, stage_name)
        """
        directory = Path(directory)

        # Vectorizers
        vec_uni_path = directory / "vec_uni.joblib"
        self.vec_uni = joblib.load(vec_uni_path) if vec_uni_path.exists() else None

        vec_phrase_path = directory / "vec_phrase.joblib"
        self.vec_phrase = joblib.load(vec_phrase_path) if vec_phrase_path.exists() else None

        # Sparse matrices
        for name in ("C_uni", "C_phrase", "C_all", "DF_uni", "DF_phrase", "DF_all"):
            path = directory / f"{name}.npz"
            setattr(self, name, sp.load_npz(path) if path.exists() else None)

        # Numpy arrays
        for name in ("cluster_ids", "cluster_doc_counts", "feature_names_all",
                      "feature_names_uni", "feature_names_phrase"):
            path = directory / f"{name}.npy"
            val = np.load(path, allow_pickle=True) if path.exists() else None
            setattr(self, name, val)

        if self.cluster_ids is not None:
            self.K = len(self.cluster_ids)
            self.cluster_index = {int(cid): idx for idx, cid in enumerate(self.cluster_ids)}

        # Co-occurrence matrix
        cooc_path = directory / "cooc_matrix.npz"
        self.cooc_matrix = sp.load_npz(cooc_path) if cooc_path.exists() else None

        # Load stage metadata
        import json
        meta_path = directory / "checkpoint_meta.json"
        stage = "scoring"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            stage = meta.get("stage", "scoring")

        # Load keyword DataFrame (try new name, fall back to legacy)
        df_path = directory / "top_df.parquet"
        if not df_path.exists():
            df_path = directory / "stage2_raw.parquet"
        if not df_path.exists():
            raise FileNotFoundError(f"No checkpoint DataFrame found in {directory}")
        top_df = pd.read_parquet(df_path)

        # Restore JSON-serialized dict/list columns
        _DICT_COLS = ("pub_year_series", "year_denominators", "ppm_series",
                      "loglift_series", "bayesian_log_odds_series")
        for col in _DICT_COLS:
            if col in top_df.columns:
                top_df[col] = top_df[col].apply(
                    lambda v: json.loads(v) if isinstance(v, str) else v
                )
        for col in ("source_terms", "candidates"):
            if col in top_df.columns:
                top_df[col] = top_df[col].apply(
                    lambda v: json.loads(v) if isinstance(v, str) else v
                )

        self._log("Checkpoint loaded from stage '%s' (%d terms)", stage, len(top_df))
        return top_df, stage

    def run_from_checkpoint(self, directory: Path) -> pd.DataFrame:
        """Load checkpoint and resume pipeline from the next stage.

        Runs all stages after the checkpointed stage through to completion.
        """
        top_df, stage = self.load_checkpoint(directory)
        stage_idx = self._STAGE_ORDER.get(stage, 4)

        # Validate essential attributes were restored
        if self.vec_uni is None:
            raise RuntimeError(
                f"Checkpoint at '{directory}' is missing vec_uni. "
                "Cannot resume pipeline without a fitted vectorizer."
            )
        if self.cluster_ids is None:
            raise RuntimeError(
                f"Checkpoint at '{directory}' is missing cluster_ids."
            )

        # Stage 5: normalization
        if stage_idx < 5:
            top_df = self._stage_normalization(top_df)

        selected_terms = top_df["term"].unique().tolist() if not top_df.empty else []

        # Stage 6: cooccurrence
        if stage_idx < 6:
            self._stage_cooccurrence(selected_terms)

        # Stage 7: term network
        if stage_idx < 7:
            self._stage_term_network(selected_terms, top_df)
            top_df = self._bridge_merge_candidates(top_df)

        # Stage 8: canonicalization
        if stage_idx < 8:
            top_df = self._maybe_canonicalise(top_df)

        # Stage 9-10: depth + temporal (always run)
        top_df = self._stage_depth(top_df, selected_terms)
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
        self._log("Pipeline resumed from '%s': final rows = %d", stage, len(top_df))
        return top_df

    # ----- Backward-compatible aliases -----

    def save_stage2_snapshot(self, directory: Path, raw_top_df: pd.DataFrame) -> None:
        """Backward-compatible alias for ``save_checkpoint(directory, top_df, 'scoring')``."""
        self.save_checkpoint(directory, raw_top_df, stage="scoring")

    def load_stage2_snapshot(self, directory: Path) -> pd.DataFrame:
        """Backward-compatible alias: load checkpoint and return the DataFrame."""
        top_df, _stage = self.load_checkpoint(directory)
        return top_df

    def run_from_stage2_snapshot(self, directory: Path) -> pd.DataFrame:
        """Backward-compatible alias for ``run_from_checkpoint``."""
        return self.run_from_checkpoint(directory)

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

    # ----- Optional stages (no-op when disabled) -----

    def _stage_normalization(self, top_df: pd.DataFrame) -> pd.DataFrame:
        """Stage 5: post-top-K keyword normalization."""
        if not self.config.normalization_enabled or top_df.empty:
            return top_df
        self._log("Stage 5: normalizing %d keywords", len(top_df))
        result = normalize_keywords(
            top_df,
            builtin_aliases=self.config.builtin_aliases,
            stopwords=self.stopwords_set,
            max_edit_distance=self.config.norm_max_edit_distance,
            min_frequency_ratio=self.config.norm_min_frequency_ratio,
            plural_merge_enabled=self.config.norm_plural_merge_enabled,
        )
        self._log("Stage 5: %d -> %d keywords after normalization", len(top_df), len(result))
        return result

    def _stage_cooccurrence(self, selected_terms: List[str]) -> None:
        """Stage 6: collect term co-occurrence matrix."""
        if not self.config.cooccurrence_enabled or not selected_terms:
            return
        self._log("Stage 6: collecting co-occurrence for %d terms", len(selected_terms))

        def text_batches():
            for batch in self._data.batch_iter():
                yield batch["text"].tolist()

        self.cooc_matrix = collect_cooccurrence(
            texts_iter=text_batches(),
            selected_terms=selected_terms,
            lowercase=self.config.lowercase,
            token_pattern=self.config.token_pattern,
            strip_accents=self.config.strip_accents,
            stopwords=self.stopwords_list,
            min_cooc_count=self.config.cooccurrence_min_count,
        )
        self._log("Stage 6: co-occurrence matrix %s, nnz=%d",
                   self.cooc_matrix.shape, self.cooc_matrix.nnz)

    def _stage_term_network(self, selected_terms: List[str], top_df: pd.DataFrame) -> None:
        """Stage 7: build similarity network and find merge groups."""
        tn_cfg = self.config.term_network
        if tn_cfg is None or not getattr(tn_cfg, "enabled", False) or not selected_terms:
            return
        self._log("Stage 7: building term network for %d terms", len(selected_terms))
        network = TermNetwork(tn_cfg)

        layers, weights = [], []
        if "string" in tn_cfg.layers:
            layers.append(network.build_layer_string(selected_terms))
            weights.append(tn_cfg.layer_weights.get("string", 1.0))
        if "token" in tn_cfg.layers:
            layers.append(network.build_layer_token(selected_terms))
            weights.append(tn_cfg.layer_weights.get("token", 0.8))
        if "cooccurrence" in tn_cfg.layers and self.cooc_matrix is not None:
            layers.append(network.build_layer_cooccurrence(self.cooc_matrix))
            weights.append(tn_cfg.layer_weights.get("cooccurrence", 0.6))

        if not layers:
            return

        combined = network.combine_layers(layers, weights)
        groups = network.find_merge_groups(combined, selected_terms)
        self.merge_candidates = network.generate_candidate_sets(groups, top_df)
        self._log("Stage 7: found %d merge groups", len(groups))

    def _bridge_merge_candidates(self, top_df: pd.DataFrame) -> pd.DataFrame:
        """Bridge Stage 7→8: inject merge candidates as per-term candidate lists.

        Converts group-based ``self.merge_candidates`` into a per-row
        ``candidates`` column expected by the ``llm_candidates`` alias strategy.
        """
        if not self.merge_candidates or top_df.empty:
            return top_df

        cand_col = self.config.alias_candidate_column  # default "candidates"

        # Build term → list of other terms in same group
        term_to_candidates: Dict[str, List[str]] = {}
        for entry in self.merge_candidates:
            group_terms = entry["terms"]
            for term in group_terms:
                others = [t for t in group_terms if t != term]
                if others:
                    # Merge with any existing candidates (term may appear in multiple groups)
                    existing = term_to_candidates.get(term, [])
                    for o in others:
                        if o not in existing:
                            existing.append(o)
                    term_to_candidates[term] = existing

        top_df = top_df.copy()
        top_df[cand_col] = top_df["term"].map(
            lambda t: term_to_candidates.get(t, [])
        )

        n_with_cands = (top_df[cand_col].apply(len) > 0).sum()
        self._log(
            "Stage 7→8: injected candidates for %d/%d terms from %d merge groups",
            n_with_cands, len(top_df), len(self.merge_candidates),
        )
        return top_df

    def _expand_short_terms(self, top_df: pd.DataFrame, selected_terms: List[str]) -> pd.DataFrame:
        """P4: Use cooccurrence to annotate/expand short abbreviation terms.

        For each short term (len <= short_term_max_length), find the most
        strongly cooccurring longer term that contains it as a substring
        or shares initials.  Annotates with ``expanded_from`` column.
        """
        cfg = self.config
        if not cfg.short_term_expansion_enabled or self.cooc_matrix is None or top_df.empty:
            return top_df

        max_len = cfg.short_term_max_length
        min_ratio = cfg.short_term_min_cooc_ratio
        mode = cfg.short_term_expansion_mode

        term_to_idx = {t: i for i, t in enumerate(selected_terms)}
        cooc = self.cooc_matrix

        top_df = top_df.copy()
        expansions: Dict[str, str] = {}

        for _, row in top_df.iterrows():
            term = row["term"]
            if len(term) > max_len or " " in term:
                continue
            tidx = term_to_idx.get(term)
            if tidx is None:
                continue

            # Get cooccurrence partners sorted by strength
            cooc_row = cooc.getrow(tidx)
            if cooc_row.nnz == 0:
                continue
            term_total = float(cooc_row.sum())
            if term_total == 0:
                continue

            best_expansion = None
            best_ratio = 0.0

            for partner_idx, count in zip(cooc_row.indices, cooc_row.data):
                partner = selected_terms[partner_idx]
                ratio = float(count) / term_total
                if ratio <= best_ratio or ratio < min_ratio:
                    continue
                # Check if partner contains the short term (substring match)
                if term.lower() in partner.lower() and len(partner) > len(term):
                    best_expansion = partner
                    best_ratio = ratio

            if best_expansion is not None:
                expansions[term] = best_expansion

        if expansions:
            if mode in ("annotate", "both"):
                top_df["expanded_from"] = top_df["term"].map(
                    lambda t: expansions.get(t, "")
                )
            if mode in ("replace", "both"):
                top_df["term"] = top_df["term"].map(
                    lambda t: expansions.get(t, t)
                )
            self._log("P4: expanded %d short terms via cooccurrence context", len(expansions))
        return top_df

    def _auto_merge_candidates(self, top_df: pd.DataFrame) -> pd.DataFrame:
        """P3: Auto-merge high-confidence term network groups without LLM.

        For each merge group where all terms are within the same cluster
        and the frequency-dominant term is clear, merge directly.
        """
        cfg = self.config
        if not cfg.auto_merge_enabled or not self.merge_candidates or top_df.empty:
            return top_df

        min_sim = cfg.auto_merge_min_similarity
        merge_actions: Dict[str, str] = {}  # term → canonical
        consumed_groups = set()

        term_freq = dict(zip(top_df["term"], top_df["frequency"]))

        for gi, entry in enumerate(self.merge_candidates):
            group_terms = entry["terms"]
            sims = entry.get("pair_similarities", {})

            # Only auto-merge if ALL pair similarities are high
            if sims:
                if any(s < min_sim for s in sims.values()):
                    continue
            elif len(group_terms) > 2:
                continue  # no similarity data, skip large groups

            # Pick canonical = highest frequency
            ranked = sorted(group_terms, key=lambda t: -term_freq.get(t, 0))
            canonical = ranked[0]
            for t in ranked[1:]:
                if t != canonical and t in term_freq:
                    merge_actions[t] = canonical
            consumed_groups.add(gi)

        if not merge_actions:
            return top_df

        # Apply merges: sum frequencies, keep max score
        top_df = top_df.copy()
        rows_to_drop = []
        for idx, row in top_df.iterrows():
            target = merge_actions.get(row["term"])
            if target is None:
                continue
            target_rows = top_df[top_df["term"] == target]
            if target_rows.empty:
                continue
            tidx = target_rows.index[0]
            top_df.at[tidx, "frequency"] = top_df.at[tidx, "frequency"] + row["frequency"]
            top_df.at[tidx, "score"] = max(top_df.at[tidx, "score"], row["score"])
            if "doc_coverage" in top_df.columns:
                top_df.at[tidx, "doc_coverage"] = max(
                    top_df.at[tidx, "doc_coverage"], row["doc_coverage"]
                )
            rows_to_drop.append(idx)

        top_df = top_df.drop(rows_to_drop).reset_index(drop=True)

        # Remove consumed groups from merge_candidates so Stage 8 doesn't re-process
        self.merge_candidates = [
            e for gi, e in enumerate(self.merge_candidates) if gi not in consumed_groups
        ]

        self._log("P3: auto-merged %d terms into %d canonical forms (%d groups consumed)",
                  len(merge_actions), len(set(merge_actions.values())), len(consumed_groups))
        return top_df

    def _stage_depth(self, top_df: pd.DataFrame, selected_terms: List[str]) -> pd.DataFrame:
        """Stage 9: estimate conceptual depth for each keyword."""
        depth_cfg = self.config.depth
        if depth_cfg is None or not getattr(depth_cfg, "enabled", False) or top_df.empty:
            return top_df
        self._log("Stage 9: estimating depth for %d keywords", len(top_df))
        return estimate_depth(
            top_df,
            cooc_matrix=self.cooc_matrix,
            selected_terms=selected_terms if self.cooc_matrix is not None else None,
            config=depth_cfg,
        )

    # ----- Public API -----

    def run(self) -> pd.DataFrame:
        self._log("Pipeline run started")

        # Pass 1: vectorization → aggregation → scoring
        self._fit_vectorizers()                          # Stage 1 (vectorization)
        self._aggregate_counts()                         # Stage 3 (aggregation)
        self._apply_vocab_merge()                        # Stage 2 (vocab_merge)
        top_df = self._stage_scores_and_topk()           # Stage 4 (scoring)

        # Pass 2: normalization → network → canonicalization
        top_df = self._stage_normalization(top_df)       # Stage 5 (+ P2 plural merge)
        selected_terms = top_df["term"].unique().tolist() if not top_df.empty else []
        self._stage_cooccurrence(selected_terms)         # Stage 6
        top_df = self._expand_short_terms(top_df, selected_terms)  # P4: abbreviation expansion
        self._stage_term_network(selected_terms, top_df) # Stage 7
        top_df = self._auto_merge_candidates(top_df)     # P3: auto-merge high-confidence
        top_df = self._bridge_merge_candidates(top_df)   # Stage 7→8
        top_df = self._maybe_canonicalise(top_df)        # Stage 8

        # Pass 3: depth → temporal
        top_df = self._stage_depth(top_df, selected_terms)  # Stage 9
        term_year = self._compute_year_series(top_df)       # Stage 10
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
