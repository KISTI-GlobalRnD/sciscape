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
    cluster_level: str = "cluster_micro"

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
    cross_cluster_penalty_min_count: int = 3
    cross_cluster_penalty_fn: str = "inverse"  # "inverse" or "log_inverse"

    # Plural merging in normalization (P2)
    norm_plural_merge_enabled: bool = True

    # Short-term abbreviation expansion (P4)
    short_term_expansion_enabled: bool = False
    short_term_max_length: int = 2
    short_term_min_cooc_ratio: float = 0.3
    short_term_expansion_mode: str = "annotate"  # "annotate" | "replace" | "both"

    # Auto-merge without LLM (P3)
    auto_merge_enabled: bool = False
    auto_merge_min_similarity: float = 0.85

    # Execution
    n_jobs: int = -1
    use_polars: bool = True
    use_pyarrow_streaming: bool = True
    verbose: bool = False

    # Candidate allowlist for Stage 2.5 (alias_strategy="llm_candidates")
    alias_candidate_column: str = "candidates"
    alias_candidate_max: int = 15
    alias_candidate_enforce: bool = True

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


# Output schema tiers
CORE_COLUMNS = ["cluster_id", "term", "score", "frequency"]
TIER2_COLUMNS = ["doc_coverage", "source_terms", "pub_year_series",
                 "ppm_series", "loglift_series", "bayesian_log_odds_series",
                 "year_denominators"]
TIER3_COLUMNS = ["depth_score", "depth_level", "cross_cluster_count",
                 "candidates"]


__all__ = [
    "KeywordExtractionConfig",
    "KeywordRecord",
    "VocabMergeConfig",
    "CORE_COLUMNS",
    "TIER2_COLUMNS",
    "TIER3_COLUMNS",
]
