"""Keyword extraction pipeline orchestrator.

Coordinates vectorization, aggregation, scoring, canonicalization, and
temporal metrics into a single run() call.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import joblib
from joblib import Parallel, delayed
from scipy import sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

from .abbreviations import build_abbreviation_lookup, extract_parenthetical_abbreviations
from .config import KeywordExtractionConfig
from .extraction import (
    _DataSource,
    _argpartition_topk,
    _detect_boundary_fragments,
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
from .quality import annotate_keyword_quality
from .temporal import TemporalMixin
from .term_network import TermNetwork
from .utils import METADATA_ARTIFACT_FILTER_VERSION, _looks_like_metadata_artifact_term
from .vocab_cleansing import VocabSimGraph, run_vocab_cleansing
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
    "data", "time", "effect", "important", "report", "describe",
    "demonstrate", "significant", "observed", "determine", "evaluate",
    "increase", "decrease", "achieve", "present", "provide",
    "suggest", "indicate", "require", "include", "total",
    "various", "possible", "recent", "current", "particular",
    "general", "several", "certain", "good", "best", "order",
    "found", "given", "known", "made", "taken", "number",
    "information", "set", "case", "condition", "range",
    "accuracy", "proposed method",
    # Generic measurement / description terms (from pilot review)
    "real", "systems", "degrees", "angle", "level", "value",
    "type", "form", "part", "area", "point", "rate", "state",
    "small", "simple", "complex", "related", "specific",
    "potential", "major", "main", "key", "based method",
    # Generic unigrams identified from pilot output inspection
    "body", "dataset", "datasets", "error", "errors",
    "feature", "features", "field", "fields",
    "local", "module", "modules", "multi",
    "objects", "parameters", "points", "test", "tests",
    # Generic terms leaked through at higher top_n_keywords (200+)
    "application", "property", "sample", "structure", "work",
    "use", "research", "experiment", "design", "accurate",
    "calculated", "tested", "failure", "electromagnetic",
    "strategy", "reconstruction", "dense", "neural",
    "smart", "valid", "validity", "camera",
    "young", "source", "radial", "magnetically",
    # Participial / adjectival forms that are not domain-specific
    "thermodynamic", "vibrational", "characteristic",
    "material", "materials", "numerical",
    # Generic short words (not domain-specific in isolation)
    "non", "art", "end", "self", "near", "long", "aged", "mean",
    "hand", "turn", "life", "phas",  # "phas" = truncated "phase"
    # Verbs / adverbs / adjectives that are not domain-specific
    "finally", "effectively", "designed", "determined", "introduced",
    "established", "influence", "estimate", "reference", "function",
    "observation", "evolution", "speed", "position", "multiple",
    "dynamic", "measurement", "date", "online", "article", "articles",
    "download", "downloads", "view", "views", "counter", "description",
    "option", "options", "received",
})

# Publisher / copyright boilerplate tokens — if *any* of these words appear
# in a term, the term is almost certainly journal metadata that leaked into
# the abstract text.
_PUBLISHER_TOKENS: frozenset = frozenset({
    "elsevier", "springer", "wiley", "copyright", "reserved",
    "llc", "published", "publication", "publications", "publisher",
    "ltd", "inc", "gmbh", "journal",
})

# Parsing residue patterns that should be removed regardless of domain.
# These are *substring* tokens that indicate broken formula / notation parsing.
_ARTIFACT_PHRASE_TOKENS: frozenset = frozenset({
    "center dot",   # HTML middle-dot (·) converted to text
    "dot center",
    "circle dot",   # alternate rendering of ·
    "dot circle",
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
        self.abbreviation_evidence: Optional[pd.DataFrame] = None
        self._abbreviation_lookup: Optional[Dict[str, Any]] = None
        self._abbreviation_evidence_loaded: bool = False
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
        # Known short (≤2 char) terms that are valid domain abbreviations.
        # Anything ≤2 chars NOT in this set is filtered as an artifact.
        self._known_short_terms: frozenset = frozenset({
            # Physics / chemistry / units
            "2d", "3d", "uv", "ir", "ph", "dc", "ac", "rf",
            # Common scientific abbreviations
            "ai", "ml", "dl", "nn", "cv", "io", "os", "db",
            "ct", "mr",
        })

        # Optional stage artifacts (populated during run)
        self.cooc_matrix: Optional[sp.csr_matrix] = None
        self.cooc_terms: Optional[List[str]] = None  # term order for cooc_matrix indices
        self.merge_candidates: Optional[List[Dict]] = None
        self.vocab_sim_graph: Optional[VocabSimGraph] = None
        self._vocab_cleansing_done: bool = False  # True after Stage 3 runs

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
            cfg.alias_cache_path = Path("workspace/artifacts") / "canonicalise" / timestamp
        base = Path(cfg.alias_cache_path)
        base.mkdir(parents=True, exist_ok=True)
        self._alias_cache_dir = base

    def _log(self, message: str, *args) -> None:
        if self.config.verbose:
            logger.info(message, *args)

    def _parallel_backend(self) -> str:
        backend = self.config.parallel_backend
        if backend == "auto":
            if self.K >= int(self.config.parallel_large_cluster_threshold):
                return "threading"
            return "loky"
        return backend

    def _write_progress(self, stage: str, processed: int, total: int, **extra: Any) -> None:
        path = self.config.progress_path
        if path is None:
            return
        payload = {
            "updated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "stage": stage,
            "processed": int(processed),
            "total": int(total),
            "percent": round((100.0 * int(processed) / max(1, int(total))), 3),
            "parallel_backend": self._parallel_backend(),
            "n_jobs": int(self.n_jobs_effective),
        }
        payload.update(extra)
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(target)

    @staticmethod
    def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(target)

    def _run_cluster_tasks(self, stage: str, total: int, func) -> list[Any]:
        return self._run_cluster_range_tasks(stage, 0, total, func, total)

    def _run_cluster_range_tasks(
        self,
        stage: str,
        start: int,
        end: int,
        func,
        total: Optional[int] = None,
        **progress_extra: Any,
    ) -> list[Any]:
        backend = self._parallel_backend()
        interval = max(1, int(self.config.progress_interval_clusters))
        total_progress = int(total if total is not None else end - start)
        if backend == "sequential" or self.n_jobs_effective == 1:
            results: list[Any] = []
            self._write_progress(stage, start, total_progress, **progress_extra)
            for idx in range(start, end):
                results.append(func(idx))
                done = idx + 1
                if done == end or done % interval == 0:
                    self._write_progress(stage, done, total_progress, **progress_extra)
            return results

        prefer = "threads" if backend == "threading" else "processes"
        self._write_progress(stage, start, total_progress, **progress_extra)
        results = Parallel(n_jobs=self.n_jobs_effective, prefer=prefer)(
            delayed(func)(r) for r in range(start, end)
        )
        self._write_progress(stage, end, total_progress, **progress_extra)
        return results

    def _effective_scoring_shard_size(self, total: int) -> int:
        if self.config.scoring_shard_dir is None:
            return 0
        configured = int(self.config.scoring_shard_size_clusters)
        if configured > 0:
            return configured
        return min(256, max(1, int(total)))

    @staticmethod
    def _feature_names_digest(feature_names: np.ndarray) -> str:
        digest = hashlib.sha256()
        for term in feature_names:
            digest.update(str(term).encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()

    def _scoring_fingerprint(
        self,
        C_all: sp.csr_matrix,
        DF_all: Optional[sp.csr_matrix],
        feature_names: np.ndarray,
        top_k: int,
        pool_size: int,
    ) -> Dict[str, Any]:
        cluster_digest = hashlib.sha256(np.asarray(self.cluster_ids).tobytes()).hexdigest()
        payload: Dict[str, Any] = {
            "schema_version": "sciscape_scoring_shard_fingerprint_v2",
            "metadata_artifact_filter_version": int(METADATA_ARTIFACT_FILTER_VERSION),
            "clusters": int(C_all.shape[0]),
            "features": int(C_all.shape[1]),
            "top_k": int(top_k),
            "pool_size": int(pool_size),
            "matrix_nnz": int(C_all.nnz),
            "matrix_sum": int(C_all.sum()),
            "feature_names_sha256": self._feature_names_digest(feature_names),
            "cluster_ids_sha256": cluster_digest,
            "min_cluster_doc_coverage": int(self.config.min_cluster_doc_coverage),
            "min_cluster_doc_coverage_ratio": float(self.config.min_cluster_doc_coverage_ratio),
            "mmr_jaccard_lambda": float(self.config.mmr_jaccard_lambda),
            "mmr_pool_factor": float(self.config.mmr_pool_factor),
            "w_ctfidf": float(self.config.w_ctfidf),
            "w_llr": float(self.config.w_llr),
            "cross_cluster_penalty_enabled": bool(self.config.cross_cluster_penalty_enabled),
            "cross_cluster_penalty_min_count": int(self.config.cross_cluster_penalty_min_count),
            "cross_cluster_penalty_fn": str(self.config.cross_cluster_penalty_fn),
            "fragment_suppression_enabled": bool(self.config.fragment_suppression_enabled),
            "fragment_min_longer_ratio": float(self.config.fragment_min_longer_ratio),
            "artifact_filter_enabled": bool(self.config.artifact_filter_enabled),
            "artifact_filter_patterns": list(self.config.artifact_filter_patterns),
            "academic_stopwords_enabled": bool(self.config.academic_stopwords_enabled),
        }
        if DF_all is not None:
            payload.update(
                {
                    "df_nnz": int(DF_all.nnz),
                    "df_sum": int(DF_all.sum()),
                }
            )
        else:
            payload.update({"df_nnz": None, "df_sum": None})
        return payload

    @staticmethod
    def _scoring_shard_payload_matches(
        payload: Dict[str, Any],
        *,
        fingerprint: Dict[str, Any],
        shard_index: int,
        start: int,
        end: int,
        total: int,
    ) -> bool:
        return (
            payload.get("schema_version") == "sciscape_scoring_shard_done_v1"
            and int(payload.get("shard_index", -1)) == int(shard_index)
            and int(payload.get("row_start", -1)) == int(start)
            and int(payload.get("row_end", -1)) == int(end)
            and int(payload.get("total_clusters", -1)) == int(total)
            and payload.get("fingerprint") == fingerprint
            and payload.get("status") == "complete"
        )

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
        if any(pat.search(term) for pat in self._artifact_res):
            return True
        lower = term.lower()
        if _looks_like_metadata_artifact_term(lower):
            return True
        # Publisher / copyright boilerplate (any token match)
        tokens = set(lower.split())
        if tokens & _PUBLISHER_TOKENS:
            return True
        # Parsing residue — "center dot" patterns from HTML entity conversion
        if any(phrase in lower for phrase in _ARTIFACT_PHRASE_TOKENS):
            return True
        # Short meaningless terms: ≤2 chars that are not known chemical/physics symbols
        if len(lower) <= 2 and lower not in self._known_short_terms:
            return True
        # Numeric-heavy residue: e.g. "similar 10", "10 circle dot"
        if any(tok.isdigit() for tok in tokens):
            # Allow if *most* tokens are alphabetic (e.g. "3d" is OK as single token)
            alpha_tokens = [t for t in tokens if t.isalpha()]
            if len(alpha_tokens) < len(tokens) * 0.5:
                return True
        return False

    def _is_academic_stopword(self, term: str) -> bool:
        """P1: Check if term is academic boilerplate.

        Single-word: exact match against the stopword set.
        Multi-word: filtered only if *every* token is an academic stopword
        (e.g., "proposed method" → all tokens generic → filtered,
         "fault diagnosis" → "fault" is not generic → kept).
        """
        tokens = term.lower().split()
        return all(t in self._academic_sw for t in tokens)

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

    # ----- Stage 2 (aggregation): aggregate document counts to cluster counts -----

    def _aggregate_counts(self) -> None:
        self._log("Stage 2 (aggregation): aggregating counts per cluster...")
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

        # Build numpy lookup array for cluster_id → index (avoids per-row dict lookup)
        _ci = self.cluster_index
        _max_cid = max(_ci.keys()) if _ci else 0
        _cid_to_idx = np.full(_max_cid + 1, -1, dtype=np.int32)
        for _cid, _idx in _ci.items():
            _cid_to_idx[_cid] = _idx

        for batch in self._data.batch_iter():
            texts = batch["text"].tolist()
            clusters = batch["cluster_id"].astype(int).to_numpy()
            codes = _cid_to_idx[clusters]

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
            "Stage 2 (aggregation): processed %d documents across %d clusters (total tokens: %d)",
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

    # ----- Legacy vocab_merge (superseded by Stage 3 vocab_cleansing) -----

    def _apply_vocab_merge(self) -> None:
        """Legacy: merge vocabulary columns for plural/hyphen variants only.

        Superseded by ``_stage_vocab_cleansing()`` when ``vocab_merge.enabled=True``.
        Kept as fallback for configs that set ``vocab_merge=None`` or ``enabled=False``.
        """
        vm_cfg = self.config.vocab_merge
        if vm_cfg is None or not vm_cfg.enabled:
            return
        if self.feature_names_uni is None or self.C_uni is None:
            return

        merge_map = build_merge_map(self.feature_names_uni, vm_cfg, C=self.C_uni)
        if not merge_map:
            self._log("Vocab merge: no mergeable pairs found")
            return

        # Store human-readable merge dictionary for visualization
        self.vocab_merge_dict: Dict[str, str] = {
            str(self.feature_names_uni[src]): str(self.feature_names_uni[tgt])
            for src, tgt in merge_map.items()
        }

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

    # ----- Stage 3 (vocab cleansing): full-vocabulary normalization + merge -----

    def _stage_vocab_cleansing(self) -> None:
        """Stage 3: Full-vocabulary cleansing (notation, spelling, plural, edit-distance).

        Replaces the old vocab_merge stage with a comprehensive cleansing pass
        that operates on the entire vocabulary before scoring.  Also builds a
        similarity graph for later LLM candidate generation.
        """
        if self.feature_names_uni is None or self.C_uni is None:
            return

        vm_cfg = self.config.vocab_merge
        merge_freq_ratio = vm_cfg.merge_frequency_ratio if vm_cfg else 0.01

        orig_uni = len(self.feature_names_uni)
        orig_phrase = len(self.feature_names_phrase) if self.feature_names_phrase is not None else 0
        self._log("Stage 3 (vocab cleansing): starting on %d unigrams + %d phrases",
                  orig_uni, orig_phrase)

        fn_phrase = self.feature_names_phrase if self.feature_names_phrase is not None else np.array([], dtype=str)
        tn_cfg = self.config.term_network
        build_similarity_graph = bool(tn_cfg is not None and getattr(tn_cfg, "enabled", False))

        (
            self.feature_names_uni,
            fn_phrase_out,
            self.C_uni,
            self.C_phrase,
            self.DF_uni,
            self.DF_phrase,
            self.vocab_sim_graph,
            merge_dict,
        ) = run_vocab_cleansing(
            feature_names_uni=self.feature_names_uni,
            feature_names_phrase=fn_phrase,
            C_uni=self.C_uni,
            C_phrase=self.C_phrase,
            DF_uni=self.DF_uni,
            DF_phrase=self.DF_phrase,
            merge_frequency_ratio=merge_freq_ratio,
            edit_distance_max=1,
            edit_distance_ratio=0.01,
            sim_graph_max_dist=2,
            build_similarity_graph=build_similarity_graph,
            verbose_callback=self._log,
        )
        self.feature_names_phrase = fn_phrase_out

        # Store human-readable merge dictionary for visualization
        self.vocab_merge_dict: Dict[str, str] = merge_dict

        self._log(
            "Stage 3 (vocab cleansing): unigrams %d -> %d, phrases %d -> %d, "
            "sim_graph=%s, total merges=%d",
            orig_uni, len(self.feature_names_uni),
            orig_phrase, len(self.feature_names_phrase),
            repr(self.vocab_sim_graph),
            len(merge_dict),
        )
        self._vocab_cleansing_done = True

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
        top_k_override: int = 0,
    ) -> pd.DataFrame:
        K, _ = scores.shape
        top_k = max(1, top_k_override if top_k_override > 0 else int(self.config.top_n_keywords))
        pool_size = max(top_k, int(np.ceil(self.config.mmr_pool_factor * top_k)))
        min_doc_cov = max(0, int(self.config.min_cluster_doc_coverage))
        cluster_doc_counts = getattr(self, "cluster_doc_counts", None)
        total_docs = getattr(self, "total_docs", 0)

        # P6: pre-compute per-term cluster frequency for cross-cluster penalty
        cluster_freq: Optional[np.ndarray] = None
        if self.config.cross_cluster_penalty_enabled and DF_all is not None:
            cluster_freq = np.asarray((DF_all > 0).astype(bool).sum(axis=0)).ravel()

        # Pre-extract CSR internals for direct indptr access (avoids getrow() overhead)
        sc_indptr, sc_indices, sc_data = scores.indptr, scores.indices, scores.data
        ca_indptr, ca_indices, ca_data = C_all.indptr, C_all.indices, C_all.data
        df_indptr, df_indices, df_data = (
            (DF_all.indptr, DF_all.indices, DF_all.data) if DF_all is not None
            else (None, None, None)
        )

        def extract_row(r: int) -> List[Tuple[int, str, float, int, int]]:
            s0, s1 = sc_indptr[r], sc_indptr[r + 1]
            if s0 == s1:
                return []
            cluster_id = int(self.cluster_ids[r])
            cluster_docs = int(cluster_doc_counts[r]) if cluster_doc_counts is not None else None
            top_terms = _argpartition_topk(sc_data[s0:s1], sc_indices[s0:s1], pool_size)
            c0, c1 = ca_indptr[r], ca_indptr[r + 1]
            freq_map = {j: int(v) for j, v in zip(ca_indices[c0:c1], ca_data[c0:c1])}
            doc_cov_map: Dict[int, int] = {}
            if df_indptr is not None:
                d0, d1 = df_indptr[r], df_indptr[r + 1]
                doc_cov_map = {j: int(v) for j, v in zip(df_indices[d0:d1], df_data[d0:d1])}

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

            # Fragment suppression: remove truncated n-gram boundary artifacts
            if self.config.fragment_suppression_enabled:
                cluster_freq_vec = np.zeros(C_all.shape[1], dtype=ca_data.dtype)
                cluster_freq_vec[ca_indices[c0:c1]] = ca_data[c0:c1]
                fragments = _detect_boundary_fragments(
                    scored_terms,
                    feature_names,
                    cluster_freq_vec,
                    min_longer_ratio=self.config.fragment_min_longer_ratio,
                )
                if fragments:
                    scored_terms = [t for t in scored_terms if t[0] not in fragments]

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

        columns = ["cluster_id", "term", "score", "frequency", "doc_coverage"]
        shard_size = self._effective_scoring_shard_size(K)
        if shard_size > 0:
            assert self.config.scoring_shard_dir is not None
            shard_dir = Path(self.config.scoring_shard_dir)
            shard_dir.mkdir(parents=True, exist_ok=True)
            n_shards = int(math.ceil(K / shard_size))
            fingerprint = self._scoring_fingerprint(
                C_all, DF_all, feature_names, top_k=top_k, pool_size=pool_size
            )
            manifest_path = shard_dir / "manifest.json"
            manifest: Dict[str, Any] = {
                "schema_version": "sciscape_scoring_shard_manifest_v1",
                "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                "status": "running",
                "stage": "scoring_topk",
                "total_clusters": int(K),
                "shard_size_clusters": int(shard_size),
                "shard_count": int(n_shards),
                "resume": bool(self.config.scoring_shard_resume),
                "fingerprint": fingerprint,
                "completed_shards": [],
            }
            self._write_json_atomic(manifest_path, manifest)

            frames: List[pd.DataFrame] = []
            completed_shards: List[Dict[str, Any]] = []
            for shard_index, start in enumerate(range(0, K, shard_size)):
                end = min(K, start + shard_size)
                shard_path = shard_dir / f"scoring_topk_shard_{shard_index:04d}.parquet"
                done_path = shard_dir / f"scoring_topk_shard_{shard_index:04d}.done.json"
                shard_loaded = False

                if self.config.scoring_shard_resume and shard_path.exists() and done_path.exists():
                    try:
                        done_payload = json.loads(done_path.read_text(encoding="utf-8"))
                        if self._scoring_shard_payload_matches(
                            done_payload,
                            fingerprint=fingerprint,
                            shard_index=shard_index,
                            start=start,
                            end=end,
                            total=K,
                        ):
                            shard_df = pd.read_parquet(shard_path)
                            frames.append(shard_df)
                            shard_loaded = True
                            self._write_progress(
                                "scoring_topk",
                                end,
                                K,
                                shard_index=shard_index,
                                shard_count=n_shards,
                                row_start=start,
                                row_end=end,
                                shard_status="loaded",
                                shard_path=str(shard_path),
                            )
                    except Exception as exc:  # stale/corrupt shard: recompute below
                        self._log(
                            "Stage 4 (scoring): ignoring shard %04d due to %s",
                            shard_index,
                            exc,
                        )

                if not shard_loaded:
                    results = self._run_cluster_range_tasks(
                        "scoring_topk",
                        start,
                        end,
                        extract_row,
                        K,
                        shard_index=shard_index,
                        shard_count=n_shards,
                        row_start=start,
                        row_end=end,
                        shard_status="running",
                    )
                    flat = [item for sub in results for item in sub]
                    shard_df = pd.DataFrame(flat, columns=columns)
                    tmp_shard_path = shard_path.with_suffix(".tmp.parquet")
                    shard_df.to_parquet(tmp_shard_path, index=False)
                    tmp_shard_path.replace(shard_path)
                    done_payload = {
                        "schema_version": "sciscape_scoring_shard_done_v1",
                        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                        "status": "complete",
                        "stage": "scoring_topk",
                        "shard_index": int(shard_index),
                        "row_start": int(start),
                        "row_end": int(end),
                        "total_clusters": int(K),
                        "rows": int(len(shard_df)),
                        "fingerprint": fingerprint,
                        "shard_path": str(shard_path),
                    }
                    self._write_json_atomic(done_path, done_payload)
                    frames.append(shard_df)
                    self._write_progress(
                        "scoring_topk",
                        end,
                        K,
                        shard_index=shard_index,
                        shard_count=n_shards,
                        row_start=start,
                        row_end=end,
                        shard_status="complete",
                        shard_path=str(shard_path),
                    )

                completed_shards.append(
                    {
                        "shard_index": int(shard_index),
                        "row_start": int(start),
                        "row_end": int(end),
                        "path": str(shard_path),
                        "loaded": bool(shard_loaded),
                    }
                )
                manifest.update(
                    {
                        "updated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                        "completed_shards": completed_shards,
                    }
                )
                self._write_json_atomic(manifest_path, manifest)

            manifest.update(
                {
                    "updated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                    "status": "complete",
                    "output_rows": int(sum(len(frame) for frame in frames)),
                }
            )
            self._write_json_atomic(manifest_path, manifest)
            df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
        else:
            results = self._run_cluster_tasks("scoring_topk", K, extract_row)
            flat = [item for sub in results for item in sub]
            df = pd.DataFrame(flat, columns=columns)

        # Re-apply quality filters after parallel extraction.
        # Joblib workers may not correctly serialize closure-captured filter
        # state (academic stopwords, artifact patterns), so we re-filter here.
        if not df.empty:
            before = len(df)
            mask = df["term"].apply(
                lambda t: (
                    not (self.config.academic_stopwords_enabled and self._is_academic_stopword(t))
                    and not (self.config.artifact_filter_enabled and self._is_artifact(t))
                )
            )
            df = df[mask].reset_index(drop=True)
            dropped = before - len(df)
            if dropped > 0:
                self._log("Post-parallel filter: dropped %d terms (stopwords/artifacts)", dropped)
        return df

    def _stage_scores_and_topk(self, pool_override: int = 0) -> pd.DataFrame:
        """Stage 4: score and extract top keywords.

        Parameters
        ----------
        pool_override : int
            If > 0, extract this many keywords per cluster instead of
            ``top_n_keywords``.  Used to build a wider pre-merge pool.
        """
        effective_top_n = pool_override if pool_override > 0 else self.config.top_n_keywords
        self._log("Stage 4 (scoring): scoring terms (top_n=%d)", effective_top_n)
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
        result = self._rank_topk(C_all, scores, feature_names, DF_all, df_global,
                                 top_k_override=effective_top_n)
        self._log("Stage 4 (scoring): produced %d keyword rows", len(result))
        return result

    # ----- Checkpointing -----

    # Stage order for resume: stage name → numeric index
    # TO-BE: 1=vectorization, 2=aggregation, 3=vocab_cleansing,
    #         4=scoring, 5=cooccurrence, 6=term_network,
    #         7=canonicalize, 8=depth, 9=temporal
    _STAGE_ORDER: Dict[str, int] = {
        "vectorization": 1,
        "aggregation": 2,
        "vocab_cleansing": 3,
        "scoring": 4,
        "cooccurrence": 5,
        "term_network": 6,
        "canonicalize": 7,
        "depth": 8,
        "temporal": 9,
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

        # Post-scoring normalization (between Stage 4 and 5)
        if stage_idx < 5:
            top_df = self._stage_normalization(top_df)
            top_df = self._stage_quality_refinement(top_df, rerank=True)

        selected_terms = top_df["term"].unique().tolist() if not top_df.empty else []

        # Stage 5: cooccurrence
        if stage_idx < 5:
            self._stage_cooccurrence(selected_terms)

        # Stage 6: term network
        if stage_idx < 6:
            self._stage_term_network(selected_terms, top_df)
            top_df = self._bridge_merge_candidates(top_df)

        # Stage 7: canonicalization
        if stage_idx < 7:
            top_df = self._maybe_canonicalise(top_df)

        # Stage 8-9: depth + temporal (always run)
        top_df = self._stage_depth(top_df, selected_terms)
        term_year = self._compute_year_series(top_df)
        if not top_df.empty:
            top_df = top_df.assign(
                pub_year_series=top_df.apply(
                    lambda r: dict(term_year.get(int(r["cluster_id"]), {}).get(str(r["term"]), {})),
                    axis=1,
                )
            )
        else:
            top_df = top_df.assign(pub_year_series=[])
        top_df = self._build_time_series_metrics(top_df, term_year)
        top_df = self._filter_stopword_only_terms(top_df)
        top_df = self._stage_quality_refinement(top_df, rerank=False)

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
            if self._is_stopword_ngram(term_str):
                return False
            if self.config.artifact_filter_enabled and self._is_artifact(term_str):
                return False
            return True

        mask = df["term"].map(_valid)
        filtered = df[mask].reset_index(drop=True)
        dropped = len(df) - len(filtered)
        if dropped > 0:
            self._log("Final cleanup: dropped %d stopword/artifact terms", dropped)
        return filtered

    def _get_abbreviation_lookup(self) -> Optional[Dict[str, Any]]:
        """Build or return cached corpus-level abbreviation evidence."""
        cfg = self.config
        if not cfg.abbreviation_dictionary_enabled:
            return None
        if self._abbreviation_evidence_loaded:
            return self._abbreviation_lookup

        self._abbreviation_evidence_loaded = True
        evidence = extract_parenthetical_abbreviations(
            self._data.abbreviation_batch_iter(),
            uid_col=cfg.uid_col,
            cluster_col="cluster_id",
            title_col=cfg.title_col,
            abstract_col=cfg.abstract_col,
            max_long_form_words=cfg.abbreviation_max_long_form_words,
        )
        self.abbreviation_evidence = evidence
        self._abbreviation_lookup = build_abbreviation_lookup(
            evidence,
            min_support_docs=cfg.abbreviation_min_support_docs,
            min_cluster_support_docs=cfg.abbreviation_min_cluster_support_docs,
            min_top_support_ratio=cfg.abbreviation_min_top_support_ratio,
        )
        if not evidence.empty:
            usable = sum(1 for value in self._abbreviation_lookup.get("global", {}).values() if value.get("usable"))
            self._log(
                "Abbreviation dictionary: extracted %d evidence pairs (%d globally usable)",
                len(evidence),
                usable,
            )
        return self._abbreviation_lookup

    def _stage_quality_refinement(self, top_df: pd.DataFrame, *, rerank: bool) -> pd.DataFrame:
        """Annotate keyword quality and optionally rerank by quality score."""
        cfg = self.config
        if top_df.empty or not cfg.quality_diagnostics_enabled:
            return top_df

        abbreviation_lookup = self._get_abbreviation_lookup()
        result = annotate_keyword_quality(
            top_df,
            rerank=bool(rerank and cfg.quality_rerank_enabled),
            global_term_threshold=cfg.quality_global_term_threshold,
            global_term_penalty=cfg.quality_global_term_penalty,
            entropy_penalty=cfg.quality_cross_cluster_entropy_penalty,
            phrase_preference_weight=cfg.quality_phrase_preference_weight,
            artifact_demotion_weight=cfg.quality_artifact_demotion_weight,
            acronym_demotion_weight=cfg.quality_acronym_demotion_weight,
            formula_demotion_weight=cfg.quality_formula_demotion_weight,
            single_token_shadow_penalty=cfg.quality_single_token_shadow_penalty,
            cluster_specific_bonus=cfg.quality_cluster_specific_bonus,
            min_multiplier=cfg.quality_min_multiplier,
            acronym_max_length=cfg.quality_acronym_max_length,
            network_roles_enabled=cfg.quality_network_roles_enabled,
            abbreviation_lookup=abbreviation_lookup,
            family_representative_enabled=cfg.quality_family_representative_enabled,
            family_representative_weight=cfg.quality_family_representative_weight,
            family_representative_max_bonus=cfg.quality_family_representative_max_bonus,
        )
        if rerank and cfg.quality_rerank_enabled:
            self._log(
                "Quality refinement: reranked %d keyword candidates by quality_score",
                len(result),
            )
        return result

    # ----- Optional stages (no-op when disabled) -----

    def _stage_normalization(self, top_df: pd.DataFrame) -> pd.DataFrame:
        """Post-top-K keyword normalization (safety net after Stage 3+4).

        When Stage 3 (vocab cleansing) already ran, notation/spelling/plural
        normalization was applied at full-vocabulary level.  This stage then
        only handles: abbreviation expansion and edit-distance merge for
        any post-scoring anomalies.
        """
        if not self.config.normalization_enabled or top_df.empty:
            return top_df

        # When Stage 3 already cleaned notation/spelling/plural on full vocab,
        # skip redundant work here — only abbreviation expansion + edit-distance.
        if self._vocab_cleansing_done:
            plural_merge = False  # already done in Stage 3b
            max_edit_dist = 0     # already done in Stage 3c (unigrams); phrases via sim_graph
            self._log("Post-scoring normalization: abbreviation expansion only (%d keywords, Stage 3 did notation/plural)", len(top_df))
        else:
            plural_merge = self.config.norm_plural_merge_enabled
            max_edit_dist = self.config.norm_max_edit_distance
            self._log("Post-scoring normalization: full pass on %d keywords", len(top_df))

        result = normalize_keywords(
            top_df,
            builtin_aliases=self.config.builtin_aliases,
            stopwords=self.stopwords_set,
            max_edit_distance=max_edit_dist,
            min_frequency_ratio=self.config.norm_min_frequency_ratio,
            plural_merge_enabled=plural_merge,
        )
        self._log("Post-scoring normalization: %d -> %d keywords", len(top_df), len(result))

        # Re-apply quality filters: normalization can produce terms (e.g. via
        # plural→singular) that should have been filtered.
        if not result.empty and (self.config.academic_stopwords_enabled or self.config.artifact_filter_enabled):
            before = len(result)
            mask = result["term"].apply(
                lambda t: (
                    not (self.config.academic_stopwords_enabled and self._is_academic_stopword(t))
                    and not (self.config.artifact_filter_enabled and self._is_artifact(t))
                )
            )
            result = result[mask].reset_index(drop=True)
            dropped = before - len(result)
            if dropped > 0:
                self._log("Post-scoring filter: dropped %d terms (stopwords/artifacts after normalization)", dropped)

        return result

    def _stage_cooccurrence(self, selected_terms: List[str]) -> None:
        """Stage 5: collect term co-occurrence matrix."""
        if not self.config.cooccurrence_enabled or not selected_terms:
            return
        self._log("Stage 5 (cooccurrence): collecting co-occurrence for %d terms", len(selected_terms))

        def text_batches():
            for batch in self._data.batch_iter():
                yield batch["text"].tolist()

        self.cooc_terms = list(selected_terms)
        self.cooc_matrix = collect_cooccurrence(
            texts_iter=text_batches(),
            selected_terms=selected_terms,
            lowercase=self.config.lowercase,
            token_pattern=self.config.token_pattern,
            strip_accents=self.config.strip_accents,
            stopwords=self.stopwords_list,
            min_cooc_count=self.config.cooccurrence_min_count,
        )
        self._log("Stage 5 (cooccurrence): co-occurrence matrix %s, nnz=%d",
                   self.cooc_matrix.shape, self.cooc_matrix.nnz)

    def _stage_term_network(self, selected_terms: List[str], top_df: pd.DataFrame) -> None:
        """Stage 6: build similarity network and find merge groups."""
        tn_cfg = self.config.term_network
        if tn_cfg is None or not getattr(tn_cfg, "enabled", False) or not selected_terms:
            return
        self._log("Stage 6 (term network): building term network for %d terms", len(selected_terms))
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
        self.merge_candidates = network.generate_candidate_sets(
            groups, top_df, combined=combined, terms_list=selected_terms,
        )
        self._log("Stage 6 (term network): found %d merge groups", len(groups))

    def _bridge_merge_candidates(self, top_df: pd.DataFrame) -> pd.DataFrame:
        """Bridge Stage 6→7: inject merge candidates as per-term candidate lists.

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

        # Inject 1-hop neighbors from vocab similarity graph (Stage 3d)
        if getattr(self, "vocab_sim_graph", None) is not None:
            selected_set = set(top_df["term"].unique()) if not top_df.empty else set()
            for term in selected_set:
                nbrs = self.vocab_sim_graph.neighbor_terms(term)
                # Only include neighbors that are also in the selected terms
                relevant = [n for n in nbrs if n in selected_set and n != term]
                if relevant:
                    existing = term_to_candidates.get(term, [])
                    for n in relevant:
                        if n not in existing:
                            existing.append(n)
                    term_to_candidates[term] = existing

        top_df = top_df.copy()
        top_df[cand_col] = top_df["term"].map(
            lambda t: term_to_candidates.get(t, [])
        )

        n_with_cands = (top_df[cand_col].apply(len) > 0).sum()
        self._log(
            "Stage 6→7: injected candidates for %d/%d terms from %d merge groups",
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
        # Pre-extract CSR internals for direct indptr access
        co_indptr, co_indices, co_data = cooc.indptr, cooc.indices, cooc.data

        top_df = top_df.copy()
        expansions: Dict[str, str] = {}

        for row in top_df.itertuples(index=False):
            term = row.term
            if len(term) > max_len or " " in term:
                continue
            tidx = term_to_idx.get(term)
            if tidx is None:
                continue

            # Get cooccurrence partners via direct CSR access
            c0, c1 = co_indptr[tidx], co_indptr[tidx + 1]
            if c0 == c1:
                continue
            row_indices = co_indices[c0:c1]
            row_data = co_data[c0:c1]
            term_total = float(row_data.sum())
            if term_total == 0:
                continue

            best_substring = None   # partner that contains the short term
            best_sub_ratio = 0.0
            best_cooc = None       # highest cooccurrence partner (fallback)
            best_cooc_ratio = 0.0

            for partner_idx, count in zip(row_indices, row_data):
                partner = selected_terms[partner_idx]
                if len(partner) <= max_len:
                    continue  # skip other short terms
                ratio = float(count) / term_total
                if ratio < min_ratio:
                    continue
                # Prefer substring match (e.g., "mg" in "mgh2")
                if term.lower() in partner.lower():
                    if ratio > best_sub_ratio:
                        best_substring = partner
                        best_sub_ratio = ratio
                # Track best overall cooccurrence partner as fallback
                if ratio > best_cooc_ratio:
                    best_cooc = partner
                    best_cooc_ratio = ratio

            expansion = best_substring or best_cooc
            if expansion is not None:
                expansions[term] = expansion

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

        # Apply merges per cluster: sum frequencies, keep max score
        # Pre-build index: (term, cluster_id) → row index for O(1) lookup
        top_df = top_df.copy()
        tc_to_idx: Dict[Tuple[str, Any], int] = {}
        for row in top_df.itertuples():
            tc_to_idx.setdefault((row.term, row.cluster_id), row.Index)

        rows_to_drop = []
        for idx in list(top_df.index):
            row = top_df.loc[idx]
            target = merge_actions.get(row["term"])
            if target is None:
                continue
            cluster_id = row["cluster_id"]
            tidx = tc_to_idx.get((target, cluster_id))
            if tidx is None:
                continue
            top_df.at[tidx, "frequency"] = top_df.at[tidx, "frequency"] + row["frequency"]
            top_df.at[tidx, "score"] = max(top_df.at[tidx, "score"], row["score"])
            if "doc_coverage" in top_df.columns:
                top_df.at[tidx, "doc_coverage"] = max(
                    top_df.at[tidx, "doc_coverage"], row["doc_coverage"]
                )
            rows_to_drop.append(idx)

        top_df = top_df.drop(rows_to_drop).reset_index(drop=True)

        # Remove consumed groups from merge_candidates so Stage 7 (LLM) doesn't re-process
        self.merge_candidates = [
            e for gi, e in enumerate(self.merge_candidates) if gi not in consumed_groups
        ]

        self._log("P3: auto-merged %d terms into %d canonical forms (%d groups consumed)",
                  len(merge_actions), len(set(merge_actions.values())), len(consumed_groups))
        return top_df

    def _stage_depth(self, top_df: pd.DataFrame, selected_terms: List[str]) -> pd.DataFrame:
        """Stage 8: estimate conceptual depth for each keyword."""
        depth_cfg = self.config.depth
        if depth_cfg is None or not getattr(depth_cfg, "enabled", False) or top_df.empty:
            return top_df
        self._log("Stage 8 (depth): estimating depth for %d keywords", len(top_df))
        return estimate_depth(
            top_df,
            cooc_matrix=self.cooc_matrix,
            selected_terms=selected_terms if self.cooc_matrix is not None else None,
            config=depth_cfg,
        )

    # ----- Public API -----

    def run(self) -> pd.DataFrame:
        if self.config.keyword_engine == "cluster_sharded":
            from .cluster_sharded import run_cluster_sharded_keyword_pipeline

            self._log("Pipeline run started (cluster_sharded engine)")
            self._write_progress("pipeline_start", 0, 1, keyword_engine="cluster_sharded")

            def _progress(stage: str, processed: int, total: int) -> None:
                self._write_progress(stage, processed, total, keyword_engine="cluster_sharded")

            top_df = run_cluster_sharded_keyword_pipeline(self.config, progress_callback=_progress)
            self.final_keywords = top_df
            self._log("Pipeline run complete: final rows = %d", len(top_df))
            self._write_progress("complete", 1, 1, final_rows=int(len(top_df)), keyword_engine="cluster_sharded")
            return top_df

        self._log("Pipeline run started")
        self._write_progress("pipeline_start", 0, 1)

        # Pass 1: vectorization → aggregation → vocab cleansing → scoring (wide pool)
        self._write_progress("vectorization", 0, 1)
        self._fit_vectorizers()                          # Stage 1 (vectorization)
        self._write_progress("vectorization", 1, 1)
        self._write_progress("aggregation", 0, 1)
        self._aggregate_counts()                         # Stage 2 (aggregation)
        self._write_progress("aggregation", 1, 1)

        # Stage 3: vocab cleansing (replaces old vocab_merge)
        self._write_progress("vocab_cleansing", 0, 1)
        vm_cfg = self.config.vocab_merge
        if vm_cfg is not None and vm_cfg.enabled:
            self._stage_vocab_cleansing()
        else:
            self._apply_vocab_merge()  # fallback for backward compat
        self._write_progress("vocab_cleansing", 1, 1)
        pool_factor = max(1.0, self.config.scoring_pool_factor)
        pool_size = int(np.ceil(self.config.top_n_keywords * pool_factor))
        top_df = self._stage_scores_and_topk(pool_override=pool_size)  # Stage 4 (wide pool)

        # Pass 2: keyword refinement (post-scoring normalization → trim → network → canonicalize)
        self._write_progress("normalization", 0, 1)
        top_df = self._stage_normalization(top_df)       # post-scoring normalization (+ P2 plural)
        self._write_progress("normalization", 1, 1)
        self._write_progress("quality_rerank", 0, 1)
        top_df = self._stage_quality_refinement(top_df, rerank=True)
        self._write_progress("quality_rerank", 1, 1)

        # Trim back to top_n_keywords per cluster after merge
        final_k = self.config.top_n_keywords
        if not top_df.empty and pool_factor > 1.0:
            before_trim = len(top_df)
            sort_col = "quality_score" if (
                self.config.quality_rerank_enabled and "quality_score" in top_df.columns
            ) else "score"
            top_df = (
                top_df
                .sort_values(["cluster_id", sort_col, "score"], ascending=[True, False, False])
                .groupby("cluster_id", sort=False)
                .head(final_k)
                .reset_index(drop=True)
            )
            self._log("Pool trim: %d -> %d keywords (top_n=%d per cluster)",
                       before_trim, len(top_df), final_k)

        selected_terms = top_df["term"].unique().tolist() if not top_df.empty else []
        self._write_progress("cooccurrence", 0, 1)
        self._stage_cooccurrence(selected_terms)         # Stage 5
        self._write_progress("cooccurrence", 1, 1)
        top_df = self._expand_short_terms(top_df, selected_terms)  # P4: abbreviation expansion
        self._write_progress("term_network", 0, 1)
        self._stage_term_network(selected_terms, top_df) # Stage 6
        self._write_progress("term_network", 1, 1)
        top_df = self._auto_merge_candidates(top_df)     # P3: auto-merge high-confidence
        top_df = self._bridge_merge_candidates(top_df)   # Stage 6→7
        self._write_progress("canonicalize", 0, 1)
        top_df = self._maybe_canonicalise(top_df)        # Stage 7
        self._write_progress("canonicalize", 1, 1)

        # Pass 3: metadata (depth → temporal)
        self._write_progress("depth", 0, 1)
        top_df = self._stage_depth(top_df, selected_terms)  # Stage 8
        self._write_progress("depth", 1, 1)
        self._write_progress("temporal", 0, 1)
        term_year = self._compute_year_series(top_df)       # Stage 9
        if not top_df.empty:
            top_df = top_df.assign(
                pub_year_series=top_df.apply(
                    lambda r: dict(term_year.get(int(r["cluster_id"]), {}).get(str(r["term"]), {})),
                    axis=1,
                )
            )
        else:
            top_df = top_df.assign(pub_year_series=[])
        top_df = self._build_time_series_metrics(top_df, term_year)
        self._write_progress("temporal", 1, 1)
        top_df = self._filter_stopword_only_terms(top_df)
        self._write_progress("quality_final", 0, 1)
        top_df = self._stage_quality_refinement(top_df, rerank=False)
        self._write_progress("quality_final", 1, 1)

        self.final_keywords = top_df
        self._log("Pipeline run complete: final rows = %d", len(top_df))
        self._write_progress("complete", self.K, self.K, final_rows=int(len(top_df)))
        return top_df

    def top_unigrams(self) -> pd.DataFrame:
        """Return per-cluster top unigrams for debugging."""
        if self.C_uni is None or self.feature_names_uni is None:
            raise RuntimeError("Run the pipeline before requesting top unigrams.")
        scores = self._compute_c_tfidf(self.C_uni)
        top_k = max(1, int(self.config.top_n_unigrams))
        names = self.feature_names_uni

        # Pre-extract CSR internals for direct indptr access
        sc_ip, sc_ix, sc_dt = scores.indptr, scores.indices, scores.data
        cu_ip, cu_ix, cu_dt = self.C_uni.indptr, self.C_uni.indices, self.C_uni.data

        def extract(r: int) -> List[Tuple[int, str, float, int]]:
            s0, s1 = sc_ip[r], sc_ip[r + 1]
            if s0 == s1:
                return []
            cid = int(self.cluster_ids[r])
            top_terms = _argpartition_topk(sc_dt[s0:s1], sc_ix[s0:s1], top_k)
            c0, c1 = cu_ip[r], cu_ip[r + 1]
            freq_map = {j: int(v) for j, v in zip(cu_ix[c0:c1], cu_dt[c0:c1])}
            return [
                (cid, names[idx], float(score), freq_map.get(idx, 0))
                for idx, score in top_terms
                if freq_map.get(idx, 0) > 0
            ]

        results = self._run_cluster_tasks("top_unigrams", self.C_uni.shape[0], extract)
        flat = [item for sub in results for item in sub]
        return pd.DataFrame(flat, columns=["cluster_id", "term", "score", "frequency"])


    def get_visualization_data(self, max_edges_per_cluster: int = 80) -> Dict:
        """Extract supplementary data for the visualization dashboard.

        Returns a dict with:
        - ``cooc_edges``: list of {source, target, weight} from cooccurrence matrix
        - ``vocab_merges``: dict mapping source→target from Stage 3 (vocab cleansing)
        - ``norm_merges``: dict mapping canonical→[merged_from] from post-scoring normalization
        - ``subphrase_tree``: list of {cluster_id, parent, child} containment pairs

        Parameters
        ----------
        max_edges_per_cluster : int
            Maximum co-occurrence edges to keep per cluster (by weight),
            to avoid hairball networks.
        """
        data: Dict = {}
        cfg = self.config
        final_terms = set()
        if self.final_keywords is not None:
            final_terms = set(self.final_keywords["term"].unique())

        # Cooccurrence edge list — use cooc_terms (Stage 5 index order)
        if (
            self.cooc_matrix is not None
            and self.cooc_terms is not None
            and self.final_keywords is not None
        ):
            terms = self.cooc_terms  # correct index alignment
            cx = self.cooc_matrix.tocoo()
            edges = []
            for i, j, v in zip(cx.row, cx.col, cx.data):
                if i < j:
                    src, tgt = terms[i], terms[j]
                    # Only keep edges where both terms survived to final output
                    if src in final_terms and tgt in final_terms:
                        edges.append({
                            "source": src,
                            "target": tgt,
                            "weight": int(v),
                        })
            data["cooc_edges"] = sorted(edges, key=lambda e: -e["weight"])
        else:
            data["cooc_edges"] = []

        # Vocab merge dictionary (Stage 3)
        data["vocab_merges"] = getattr(self, "vocab_merge_dict", {})

        # Normalization merges (post-scoring) — extracted from the output DataFrame
        norm_merges = {}
        if self.final_keywords is not None and "norm_merged_from" in self.final_keywords.columns:
            has_merge = self.final_keywords["norm_merged_from"].apply(
                lambda x: isinstance(x, list) and len(x) > 0
            )
            for term, merged in zip(
                self.final_keywords.loc[has_merge, "term"],
                self.final_keywords.loc[has_merge, "norm_merged_from"],
            ):
                real_sources = [t for t in merged if t != term]
                if real_sources:
                    norm_merges[term] = real_sources
        data["norm_merges"] = norm_merges

        # Corpus abbreviation evidence — separate dictionary artifact for
        # downstream inspection and keyword display decisions.
        if self.abbreviation_evidence is not None and not self.abbreviation_evidence.empty:
            evidence_df = self.abbreviation_evidence.copy()
            evidence_df = evidence_df.sort_values(
                ["support_docs", "short_form", "candidate_rank"],
                ascending=[False, True, True],
                kind="mergesort",
            )
            data["abbreviation_evidence_total"] = int(len(evidence_df))
            report_evidence_df = evidence_df[
                pd.to_numeric(evidence_df["support_docs"], errors="coerce").fillna(0)
                >= int(cfg.abbreviation_min_support_docs)
            ].copy()

            def _jsonable_supports(value: object) -> dict[str, int]:
                if not isinstance(value, dict):
                    return {}
                result: dict[str, int] = {}
                for key, count in value.items():
                    try:
                        result[str(int(key))] = int(count)
                    except (TypeError, ValueError):
                        result[str(key)] = int(count)
                return result

            records = []
            for row in report_evidence_df.itertuples(index=False):
                records.append(
                    {
                        "short_form": str(row.short_form),
                        "long_form": str(row.long_form),
                        "support_docs": int(row.support_docs),
                        "support_occurrences": int(row.support_occurrences),
                        "cluster_supports": _jsonable_supports(row.cluster_supports),
                        "candidate_rank": int(row.candidate_rank),
                        "short_form_candidate_count": int(row.short_form_candidate_count),
                        "top_support_ratio": float(row.top_support_ratio),
                        "is_ambiguous": bool(row.is_ambiguous),
                        "ambiguity_type": str(getattr(row, "ambiguity_type", "none")),
                        "confidence": float(row.confidence),
                        "pattern_types": str(row.pattern_types),
                    }
                )
            data["abbreviation_evidence"] = records
        else:
            data["abbreviation_evidence_total"] = 0
            data["abbreviation_evidence"] = []

        # Subphrase tree: parent-child containment pairs per cluster
        subphrase_tree = []
        if self.final_keywords is not None:
            for cid, grp in self.final_keywords.groupby("cluster_id"):
                terms = grp["term"].tolist()
                for t1 in terms:
                    w1 = set(t1.split())
                    for t2 in terms:
                        if t1 == t2:
                            continue
                        w2 = set(t2.split())
                        # t2 is parent of t1 if t1's words contain all of t2's words
                        if w2 < w1:  # strict subset
                            subphrase_tree.append({
                                "cluster_id": int(cid),
                                "parent": t2,
                                "child": t1,
                            })
        data["subphrase_tree"] = subphrase_tree

        # --- Trend scores (emerging / declining) ---
        trend_scores: Dict[str, float] = {}
        if self.final_keywords is not None:
            fk = self.final_keywords
            has_ppm = "ppm_series" in fk.columns
            has_pub = "pub_year_series" in fk.columns
            if has_ppm or has_pub:
                deduped = fk.drop_duplicates(subset=["term"], keep="first")
                for row in deduped.itertuples(index=False):
                    series = (getattr(row, "ppm_series", None) if has_ppm else None) or \
                             (getattr(row, "pub_year_series", None) if has_pub else None)
                    if not series or not isinstance(series, dict):
                        continue
                    try:
                        yearly = {int(k): float(v) for k, v in series.items()}
                    except (ValueError, TypeError):
                        continue
                    if len(yearly) < 2:
                        continue
                    years = sorted(yearly)
                    mid = len(years) // 2
                    first_half = [yearly[y] for y in years[:mid]]
                    second_half = [yearly[y] for y in years[mid:]]
                    mean_first = np.mean(first_half) if first_half else 0.0
                    mean_second = np.mean(second_half) if second_half else 0.0
                    diff = mean_second - mean_first
                    denom = max(abs(mean_first), abs(mean_second), 1e-12)
                    trend_scores[row.term] = float(np.clip(diff / denom, -1.0, 1.0))
        data["trend_scores"] = trend_scores

        # --- Network centrality ---
        centrality: Dict[str, Dict[str, float]] = {}
        edges_list = data.get("cooc_edges", [])
        if edges_list:
            degree_count: Dict[str, int] = defaultdict(int)
            weighted_deg: Dict[str, float] = defaultdict(float)
            for e in edges_list:
                degree_count[e["source"]] += 1
                degree_count[e["target"]] += 1
                weighted_deg[e["source"]] += e["weight"]
                weighted_deg[e["target"]] += e["weight"]
            max_deg = max(degree_count.values()) if degree_count else 1
            max_wdeg = max(weighted_deg.values()) if weighted_deg else 1.0
            for term in degree_count:
                centrality[term] = {
                    "degree": degree_count[term] / max_deg,
                    "weighted_degree": weighted_deg[term] / max_wdeg,
                    "betweenness": 0.0,
                }
        data["centrality"] = centrality

        # --- Cross-cluster shared terms ---
        cross_cluster_terms: List[Dict] = []
        if self.final_keywords is not None:
            term_clusters: Dict[str, List[int]] = defaultdict(list)
            term_scores: Dict[str, Dict[int, float]] = defaultdict(dict)
            term_freqs: Dict[str, Dict[int, int]] = defaultdict(dict)
            _score_col = "score" in self.final_keywords.columns
            _freq_col = "frequency" in self.final_keywords.columns
            for row in self.final_keywords.itertuples(index=False):
                t = row.term
                cid = int(row.cluster_id)
                term_clusters[t].append(cid)
                term_scores[t][cid] = float(row.score) if _score_col else 0.0
                term_freqs[t][cid] = int(row.frequency) if _freq_col else 0
            for t, cids in term_clusters.items():
                unique_cids = sorted(set(cids))
                if len(unique_cids) >= 2:
                    cross_cluster_terms.append({
                        "term": t,
                        "clusters": unique_cids,
                        "scores": {c: term_scores[t].get(c, 0.0) for c in unique_cids},
                        "frequencies": {c: term_freqs[t].get(c, 0) for c in unique_cids},
                    })
        data["cross_cluster_terms"] = cross_cluster_terms

        # --- Pipeline config summary ---
        cfg = self.config
        data["pipeline_config"] = {
            "n_documents": int(self.total_docs) if hasattr(self, "total_docs") else 0,
            "n_clusters": self.K,
            "cluster_level": str(cfg.cluster_level),
            "top_n_keywords": cfg.top_n_keywords,
            "top_n_unigrams": cfg.top_n_unigrams,
            "min_df_unigram": cfg.min_df_unigram,
            "min_df_phrase": cfg.min_df_phrase,
            "ngram_range": f"{cfg.ngram_min}-{cfg.ngram_max}",
            "include_title": cfg.include_title,
            "stages_enabled": {
                "vocab_merge": cfg.vocab_merge is not None and cfg.vocab_merge.enabled,
                "normalization": cfg.normalization_enabled,
                "cooccurrence": cfg.cooccurrence_enabled,
                "term_network": cfg.term_network is not None and getattr(cfg.term_network, "enabled", False),
                "depth": cfg.depth is not None and getattr(cfg.depth, "enabled", False),
                "academic_stopwords": cfg.academic_stopwords_enabled,
                "artifact_filter": cfg.artifact_filter_enabled,
                "cross_cluster_penalty": cfg.cross_cluster_penalty_enabled,
                "fragment_suppression": cfg.fragment_suppression_enabled,
                "auto_merge": cfg.auto_merge_enabled,
                "abbreviation_dictionary": cfg.abbreviation_dictionary_enabled,
            },
            "filter_stats": {
                "vocab_merges_applied": len(getattr(self, "vocab_merge_dict", {})),
                "norm_merges_applied": len(data.get("norm_merges", {})),
                "abbreviation_pairs": int(data.get("abbreviation_evidence_total", 0)),
                "abbreviation_pairs_reported": len(data.get("abbreviation_evidence", [])),
                "final_keywords": len(final_terms),
            },
        }

        return data


def run_keyword_pipeline(config: KeywordExtractionConfig) -> pd.DataFrame:
    """Convenience entry point returning the keyword dataframe."""
    pipeline = KeywordExtractionPipeline(config)
    return pipeline.run()


__all__ = [
    "ACADEMIC_STOPWORDS",
    "KeywordExtractionPipeline",
    "run_keyword_pipeline",
]
