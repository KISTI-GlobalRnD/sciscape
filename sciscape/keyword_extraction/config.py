"""Configuration and data classes for the keyword extraction pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Mapping, Optional, Tuple, Union

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

if TYPE_CHECKING:
    from .depth import DepthConfig
    from .term_network import TermNetworkConfig


@dataclass
class KeywordExtractionConfig:
    """Configuration for the keyword extraction pipeline."""

    abstract_path: Path
    membership_path: Path
    cluster_level: Optional[str] = None  # None → auto-detect finest level

    # Vectoriser/tokenisation parameters
    lowercase: bool = True
    stopwords: Optional[Iterable[str]] = None
    use_default_stopwords: bool = True
    token_pattern: str = r"(?u)\b\w\w+\b"
    strip_accents: Optional[str] = None
    max_features_unigram: Optional[int] = None
    max_features_phrase: Optional[int] = None
    min_df_unigram: Union[int, float] = 5
    max_df_unigram: Union[int, float] = 1.0
    min_df_phrase: Union[int, float] = 5
    max_df_phrase: Union[int, float] = 1.0
    use_phrase_vectorizer: bool = True
    ngram_min: int = 2
    ngram_max: int = 2

    # Author keyword integration
    author_keyword_path: Optional[Path] = None
    author_keyword_uid_col: str = "uid"
    author_keyword_term_col: str = "keyword"
    author_keyword_join: str = " "

    # Canonicalisation / Stage 2.5
    apply_alias_map: bool = False
    alias_strategy: str = "none"  # Supported: "none", "llm", "llm_candidates", "cache_only", "load_only", "prev_top_df"
    alias_model: str = "gpt-oss:120b"
    alias_base_url: Optional[str] = None
    alias_api_key: Optional[str] = None
    alias_max_terms_per_prompt: int = 40
    alias_temperature: float = 0.0
    alias_retry: int = 2
    alias_timeout: float = 30.0
    alias_allow_translation: bool = True
    alias_language_detection: bool = True
    alias_stopword_strictness: str = "drop_if_empty"
    alias_cache_enabled: bool = True
    alias_cache_path: Optional[Path] = None
    alias_cache_key_fields: Tuple[str, ...] = ("term", "frequency", "doc_coverage", "score")
    previous_top_df_path: Optional[Path] = None
    builtin_aliases: Mapping[str, str] = field(
        default_factory=lambda: {
            "bq": "becquerel",
            "kbq": "kilobecquerel",
            "mbq": "megabecquerel",
            "gbq": "gigabecquerel",
            "sv": "sievert",
            "msv": "millisievert",
            "μsv": "microsievert",
            "usv": "microsievert",
            "gy": "gray",
            "mgy": "milligray",
            "μgy": "microgray",
            "ugy": "microgray",
            "ci": "curie",
            "pci": "picocurie",
        }
    )
    forbid_abbreviations: Tuple[str, ...] = (
        "bq",
        "kbq",
        "mbq",
        "gbq",
        "sv",
        "msv",
        "μsv",
        "usv",
        "gy",
        "mgy",
        "μgy",
        "ugy",
        "ci",
        "pci",
    )
    manual_alias_path: Optional[Path] = None

    # Text columns
    uid_col: str = "uid"
    abstract_col: str = "abstract"
    title_col: str = "title"
    year_col: str = "pubyear"

    # Title handling
    include_title: bool = False
    title_weight: float = 2.0

    # Output sizes
    top_n_unigrams: int = 100
    top_n_keywords: int = 100
    # Pre-merge pool multiplier: Stage 4 extracts top_n_keywords × scoring_pool_factor
    # keywords, then trims back to top_n_keywords after Stage 5 merging.
    # This ensures merge candidates beyond the final cut-off are considered.
    scoring_pool_factor: float = 1.5

    # Post aggregation filtering
    phrase_min_count_per_cluster: int = 10
    min_cluster_doc_coverage: int = 0
    min_cluster_doc_coverage_ratio: float = 0.0
    mmr_jaccard_lambda: float = 0.0
    mmr_pool_factor: float = 2.0
    w_ctfidf: float = 1.0
    w_llr: float = 0.0

    # Vocabulary merge (Stage 2)
    vocab_merge: Optional["VocabMergeConfig"] = None

    # Post-top-K normalization (Stage 5)
    normalization_enabled: bool = False
    norm_max_edit_distance: int = 2
    norm_min_frequency_ratio: float = 0.01

    # Co-occurrence & term network (Stages 6-7)
    cooccurrence_enabled: bool = False
    cooccurrence_min_count: int = 3
    term_network: Optional[TermNetworkConfig] = None

    # Temporal metrics (Stage 10)
    bayesian_alpha: float = 0.5   # Laplace smoothing for log-lift / Bayesian log-odds
    bayesian_prior: float = 0.5   # prior strength for Bayesian log-odds

    # Depth estimation (Stage 9)
    depth: Optional[DepthConfig] = None

    # Quality filters (P1 + P5 + P6)
    academic_stopwords_enabled: bool = True
    academic_stopwords_extra: Optional[Tuple[str, ...]] = None
    artifact_filter_enabled: bool = True
    artifact_filter_patterns: Tuple[str, ...] = (
        r"^center\s*dot$",     # LaTeX center-dot artifact
        r"^\d+$",              # pure numbers
        r"^[^\w]+$",           # pure punctuation/symbols
        r"^.$",                # single characters
    )
    cross_cluster_penalty_enabled: bool = False
    cross_cluster_penalty_min_count: int = 2
    cross_cluster_penalty_fn: str = "inverse"  # "inverse" or "log_inverse"

    # Domain-agnostic quality refinement.
    #
    # diagnostics_enabled only appends audit/display columns. rerank_enabled
    # additionally uses the quality score for final top-K selection.
    quality_diagnostics_enabled: bool = True
    quality_rerank_enabled: bool = False
    quality_global_term_threshold: float = 0.5
    quality_global_term_penalty: float = 0.45
    quality_cross_cluster_entropy_penalty: float = 0.35
    quality_phrase_preference_weight: float = 0.25
    quality_artifact_demotion_weight: float = 0.8
    quality_acronym_demotion_weight: float = 0.1
    quality_formula_demotion_weight: float = 0.25
    quality_single_token_shadow_penalty: float = 0.65
    quality_cluster_specific_bonus: float = 0.08
    quality_min_multiplier: float = 0.05
    quality_acronym_max_length: int = 6
    quality_network_roles_enabled: bool = True
    quality_family_representative_enabled: bool = True
    quality_family_representative_weight: float = 0.08
    quality_family_representative_max_bonus: float = 0.15
    abbreviation_dictionary_enabled: bool = True
    abbreviation_min_support_docs: int = 2
    abbreviation_min_cluster_support_docs: int = 2
    abbreviation_min_top_support_ratio: float = 0.75
    abbreviation_max_long_form_words: int = 12
    keyword_rule_artifact_enabled: bool = True
    keyword_rule_set_id: str = "keyword_cleaning_default_v1"
    keyword_rule_result_root: Optional[Path] = None

    # Fragment suppression — suppress truncated n-grams like "supermassive black"
    # when a longer form "supermassive black hole" exists with comparable frequency.
    fragment_suppression_enabled: bool = True
    fragment_min_longer_ratio: float = 0.5  # freq(longer) >= ratio * freq(shorter) → suppress

    # Plural merging in normalization (P2)
    norm_plural_merge_enabled: bool = True

    # Short-term abbreviation expansion (P4)
    short_term_expansion_enabled: bool = False
    short_term_max_length: int = 2
    short_term_min_cooc_ratio: float = 0.05
    short_term_expansion_mode: str = "annotate"  # "annotate" | "replace" | "both"

    # Auto-merge without LLM (P3)
    auto_merge_enabled: bool = False
    auto_merge_min_similarity: float = 0.85

    # Execution
    n_jobs: int = -1
    parallel_backend: str = "auto"  # "auto", "loky", "threading", or "sequential"
    parallel_large_cluster_threshold: int = 1000
    progress_path: Optional[Path] = None
    progress_interval_clusters: int = 100
    scoring_shard_dir: Optional[Path] = None
    scoring_shard_size_clusters: int = 0
    scoring_shard_resume: bool = True
    use_polars: bool = True
    use_pyarrow_streaming: bool = True
    verbose: bool = False

    # Candidate allowlist for Stage 2.5 (alias_strategy="llm_candidates")
    alias_candidate_column: str = "candidates"
    alias_candidate_max: int = 15
    alias_candidate_enforce: bool = True

    # Cluster-sharded keyword engine (opt-in V2).
    #
    # The legacy engine remains the default.  The cluster-sharded engine first
    # builds bounded per-cluster candidate pools, then computes corpus/global
    # term statistics and performs streaming final scoring without materialising
    # a full cluster x term matrix.
    keyword_engine: str = "legacy"  # "legacy" | "cluster_sharded"
    cluster_sharded_output_dir: Optional[Path] = None
    candidate_pool_floor: int = 256
    candidate_pool_target: int = 512
    candidate_pool_large: int = 1024
    candidate_pool_hard_max: int = 1536
    target_docs_per_shard: int = 500_000
    max_clusters_per_shard: int = 1024
    large_cluster_single_shard: bool = True
    cluster_sharded_shard_ids: Optional[Tuple[int, ...]] = None
    global_candidate_row_target: int = 50_000_000
    global_candidate_row_warning: int = 80_000_000
    global_candidate_row_hard_stop: int = 100_000_000
    global_unique_term_target: int = 5_000_000
    global_unique_term_warning: int = 8_000_000
    global_unique_term_hard_stop: int = 10_000_000
    candidate_mining_progress_interval_docs: int = 25_000
    candidate_mining_prune_interval_docs: int = 50_000
    candidate_mining_prune_multiplier: int = 8

    def build_stopword_set(self) -> set[str]:
        base = set(ENGLISH_STOP_WORDS) if self.use_default_stopwords else set()
        if self.stopwords:
            extras = set(self.stopwords)
            if self.lowercase:
                extras = {s.lower() for s in extras}
            else:
                extras = extras | {s.lower() for s in extras}
            base.update(extras)
        return base

    def __post_init__(self) -> None:
        if self.ngram_min < 1:
            raise ValueError(
                f"ngram_min must be >= 1, got {self.ngram_min}"
            )
        if self.ngram_min > self.ngram_max:
            raise ValueError(
                f"ngram_min ({self.ngram_min}) must be <= ngram_max ({self.ngram_max})"
            )
        if self.top_n_keywords < 1:
            raise ValueError(
                f"top_n_keywords must be >= 1, got {self.top_n_keywords}"
            )
        if self.top_n_unigrams < 1:
            raise ValueError(
                f"top_n_unigrams must be >= 1, got {self.top_n_unigrams}"
            )
        # min_df must be <= max_df when both are the same type
        # (int=absolute count, float=ratio — cross-type comparison is invalid)
        if (type(self.min_df_unigram) is type(self.max_df_unigram)
                and isinstance(self.min_df_unigram, (int, float))
                and self.min_df_unigram > self.max_df_unigram):
            raise ValueError(
                f"min_df_unigram ({self.min_df_unigram}) must be <= "
                f"max_df_unigram ({self.max_df_unigram})"
            )
        if (type(self.min_df_phrase) is type(self.max_df_phrase)
                and isinstance(self.min_df_phrase, (int, float))
                and self.min_df_phrase > self.max_df_phrase):
            raise ValueError(
                f"min_df_phrase ({self.min_df_phrase}) must be <= "
                f"max_df_phrase ({self.max_df_phrase})"
            )
        if not (0.0 <= self.mmr_jaccard_lambda <= 1.0):
            raise ValueError(
                f"mmr_jaccard_lambda must be in [0.0, 1.0], got {self.mmr_jaccard_lambda}"
            )
        if self.mmr_pool_factor < 1.0:
            raise ValueError(
                f"mmr_pool_factor must be >= 1.0, got {self.mmr_pool_factor}"
            )
        if self.norm_max_edit_distance < 0:
            raise ValueError(
                f"norm_max_edit_distance must be >= 0, got {self.norm_max_edit_distance}"
            )
        if not (0.0 <= self.norm_min_frequency_ratio <= 1.0):
            raise ValueError(
                f"norm_min_frequency_ratio must be in [0.0, 1.0], got {self.norm_min_frequency_ratio}"
            )
        if self.cross_cluster_penalty_min_count < 1:
            raise ValueError(
                f"cross_cluster_penalty_min_count must be >= 1, got {self.cross_cluster_penalty_min_count}"
            )
        if self.cluster_sharded_shard_ids is not None:
            shard_ids = tuple(sorted({int(value) for value in self.cluster_sharded_shard_ids}))
            if not shard_ids:
                raise ValueError("cluster_sharded_shard_ids must not be empty when provided")
            if any(value < 0 for value in shard_ids):
                raise ValueError("cluster_sharded_shard_ids must contain non-negative shard IDs")
            self.cluster_sharded_shard_ids = shard_ids
        if self.cross_cluster_penalty_fn not in ("inverse", "log_inverse"):
            raise ValueError(
                f"cross_cluster_penalty_fn must be 'inverse' or 'log_inverse', got {self.cross_cluster_penalty_fn!r}"
            )
        if not (0.0 <= self.quality_global_term_threshold <= 1.0):
            raise ValueError(
                f"quality_global_term_threshold must be in [0.0, 1.0], got {self.quality_global_term_threshold}"
            )
        for name in (
            "quality_global_term_penalty",
            "quality_cross_cluster_entropy_penalty",
            "quality_phrase_preference_weight",
            "quality_artifact_demotion_weight",
            "quality_acronym_demotion_weight",
            "quality_formula_demotion_weight",
            "quality_single_token_shadow_penalty",
            "quality_cluster_specific_bonus",
            "quality_family_representative_weight",
            "quality_family_representative_max_bonus",
        ):
            value = float(getattr(self, name))
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be in [0.0, 1.0], got {value}")
        if not (0.0 < self.quality_min_multiplier <= 1.0):
            raise ValueError(
                f"quality_min_multiplier must be in (0.0, 1.0], got {self.quality_min_multiplier}"
            )
        if self.quality_acronym_max_length < 2:
            raise ValueError(
                f"quality_acronym_max_length must be >= 2, got {self.quality_acronym_max_length}"
            )
        if not str(self.keyword_rule_set_id).strip():
            raise ValueError("keyword_rule_set_id must be non-empty")
        if self.short_term_expansion_mode not in ("annotate", "replace", "both"):
            raise ValueError(
                f"short_term_expansion_mode must be 'annotate', 'replace', or 'both', got {self.short_term_expansion_mode!r}"
            )
        if not (0.0 <= self.auto_merge_min_similarity <= 1.0):
            raise ValueError(
                f"auto_merge_min_similarity must be in [0.0, 1.0], got {self.auto_merge_min_similarity}"
            )
        if self.alias_candidate_max < 1:
            raise ValueError(
                f"alias_candidate_max must be >= 1, got {self.alias_candidate_max}"
            )
        if self.keyword_engine not in ("legacy", "cluster_sharded"):
            raise ValueError(
                "keyword_engine must be 'legacy' or 'cluster_sharded', "
                f"got {self.keyword_engine!r}"
            )
        for name in (
            "candidate_pool_floor",
            "candidate_pool_target",
            "candidate_pool_large",
            "candidate_pool_hard_max",
            "target_docs_per_shard",
            "max_clusters_per_shard",
            "global_candidate_row_target",
            "global_candidate_row_warning",
            "global_candidate_row_hard_stop",
            "global_unique_term_target",
            "global_unique_term_warning",
            "global_unique_term_hard_stop",
            "candidate_mining_progress_interval_docs",
            "candidate_mining_prune_interval_docs",
            "candidate_mining_prune_multiplier",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")
        if self.candidate_pool_floor > self.candidate_pool_hard_max:
            raise ValueError(
                "candidate_pool_floor must be <= candidate_pool_hard_max "
                f"({self.candidate_pool_floor} > {self.candidate_pool_hard_max})"
            )
        if self.global_candidate_row_target > self.global_candidate_row_hard_stop:
            raise ValueError(
                "global_candidate_row_target must be <= global_candidate_row_hard_stop"
            )
        if self.global_unique_term_target > self.global_unique_term_hard_stop:
            raise ValueError(
                "global_unique_term_target must be <= global_unique_term_hard_stop"
            )
        if self.parallel_backend not in ("auto", "loky", "threading", "sequential"):
            raise ValueError(
                "parallel_backend must be 'auto', 'loky', 'threading', or "
                f"'sequential', got {self.parallel_backend!r}"
            )
        if self.parallel_large_cluster_threshold < 1:
            raise ValueError(
                "parallel_large_cluster_threshold must be >= 1, got "
                f"{self.parallel_large_cluster_threshold}"
            )
        if self.progress_interval_clusters < 1:
            raise ValueError(
                f"progress_interval_clusters must be >= 1, got {self.progress_interval_clusters}"
            )
        if self.scoring_shard_size_clusters < 0:
            raise ValueError(
                "scoring_shard_size_clusters must be >= 0, got "
                f"{self.scoring_shard_size_clusters}"
            )
        if self.w_ctfidf + self.w_llr <= 0:
            raise ValueError(
                f"w_ctfidf + w_llr must be > 0 (got {self.w_ctfidf} + {self.w_llr} = {self.w_ctfidf + self.w_llr})"
            )


@dataclass
class VocabMergeConfig:
    """Configuration for Stage 2 vocabulary-level merging (post-vectorizer)."""

    enabled: bool = False
    plural_to_singular: bool = True
    hyphen_normalize: bool = True
    merge_frequency_ratio: float = 0.01  # skip if minor form > this ratio of major


@dataclass
class KeywordRecord:
    """Final keyword representation for export."""

    cluster_id: int
    term: str
    score: float
    frequency: int
    pub_year_series: Mapping[int, int]


# Output schema tiers — columns that may appear in run() output.
# CORE: always present after scoring (Stage 4+).
CORE_COLUMNS = ["cluster_id", "term", "score", "frequency"]
# TIER2: present when corresponding stages are enabled.
#   doc_coverage: Stage 4 (scoring, when DF_all computed)
#   source_terms: Stage 8 (canonicalization)
#   pub_year_series, ppm_series, loglift_series, bayesian_log_odds_series,
#     year_denominators: Stage 10 (temporal)
TIER2_COLUMNS = ["doc_coverage", "source_terms", "pub_year_series",
                 "ppm_series", "loglift_series", "bayesian_log_odds_series",
                 "year_denominators"]
# TIER3: present when advanced stages/filters are enabled.
#   depth_score, depth_level, cross_cluster_count: Stage 9 (depth)
#   candidates: Stage 7→8 bridge (term_network)
#   expanded_from: P4 (short_term_expansion, annotate/both mode)
#   alias_actions, alias_notes, alias_reason: Stage 8 (canonicalization)
TIER3_COLUMNS = ["depth_score", "depth_level", "cross_cluster_count",
                 "candidates", "expanded_from",
                 "alias_actions", "alias_notes", "alias_reason",
                 "raw_term", "normalized_term", "display_label",
                 "quality_score", "quality_multiplier", "quality_flags",
                 "quality_risk_family", "quality_flag_basis",
                 "quality_flag_confidence", "clean_view_action",
                 "quality_decision_trace",
                 "representative_score", "representative_multiplier",
                 "representative_rank", "representative_role", "representative_flags",
                 "keyword_label_tier",
                 "representative_family_child_count", "representative_family_member_count",
                 "representative_family_avg_child_coverage", "representative_family_multiplier",
                 "keyword_scope", "keyword_cluster_count", "keyword_cluster_ratio",
                 "abbreviation_status", "abbreviation_target", "abbreviation_confidence",
                 "abbreviation_source", "abbreviation_support_docs",
                 "abbreviation_cluster_support_docs", "abbreviation_top_support_ratio",
                 "abbreviation_ambiguity_type",
                 "network_role", "network_score", "network_flags"]


__all__ = [
    "KeywordExtractionConfig",
    "KeywordRecord",
    "VocabMergeConfig",
    "CORE_COLUMNS",
    "TIER2_COLUMNS",
    "TIER3_COLUMNS",
]
