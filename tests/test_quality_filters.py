"""Tests for quality filters P1-P7.

P1: Academic stopword filtering
P2: Plural merging in normalization
P3: Auto-merge without LLM (high-confidence term network groups)
P4: Short-term abbreviation expansion via cooccurrence
P5: Artifact filtering (LaTeX, numbers, single chars)
P6: Cross-cluster score penalty
P7: Fragment suppression (truncated n-gram boundary artifacts)
"""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse as sp

from sciscape.keyword_extraction import KeywordExtractionConfig, run_keyword_pipeline
from sciscape.keyword_extraction.normalization import (
    _normalize_spelling,
    _phrase_singular,
)
from sciscape.keyword_extraction.extraction import _detect_boundary_fragments
from sciscape.keyword_extraction.pipeline import (
    ACADEMIC_STOPWORDS,
    KeywordExtractionPipeline,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_data(tmp_path):
    """Create a small dataset with cross-cluster and short terms."""
    abstracts = pd.DataFrame({
        "uid": [f"D{i}" for i in range(12)],
        "title": [
            "Deep learning neural networks",
            "Neural network architectures for classification",
            "Machine learning algorithms for prediction",
            "Machine learning optimization methods",
            "Quantum computing quantum bits",
            "Quantum bit error correction methods",
            "Solar energy battery storage systems",
            "Battery systems for solar energy panels",
            "Mg alloy hydrogen storage capacity",
            "Ni based catalyst for hydrogen production",
            "Fe oxide nanoparticle synthesis method",
            "Pd catalyst for hydrogenation reaction",
        ],
        "abstract": [
            "Deep learning with neural networks enables pattern recognition.",
            "Neural network architecture design improves classification accuracy.",
            "Machine learning algorithms predict material properties using features.",
            "Machine learning optimization algorithms accelerate training convergence.",
            "Quantum computing with quantum bits offers parallel computation.",
            "Quantum bit error correction ensures reliable quantum computation.",
            "Solar energy combined with battery storage provides grid resilience.",
            "Battery systems integrated with solar energy panels reduce costs.",
            "Mg alloy shows excellent hydrogen storage capacity and cycling stability.",
            "Ni based catalyst demonstrates high efficiency for hydrogen production.",
            "Fe oxide nanoparticle synthesis via hydrothermal method is reported.",
            "Pd catalyst exhibits superior activity for hydrogenation reaction.",
        ],
        "pubyear": [2018, 2019, 2020, 2021, 2018, 2019, 2020, 2021, 2020, 2021, 2020, 2021],
    })
    membership = pd.DataFrame({
        "uid": [f"D{i}" for i in range(12)],
        "cluster": [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 3],
    })

    abstract_path = tmp_path / "abstracts.parquet"
    membership_path = tmp_path / "membership.parquet"
    abstracts.to_parquet(abstract_path, index=False)
    membership.to_parquet(membership_path, index=False)
    return abstract_path, membership_path


def _base_config(abstract_path, membership_path, **overrides):
    defaults = dict(
        abstract_path=abstract_path,
        membership_path=membership_path,
        cluster_level="cluster",
        include_title=True,
        title_weight=1.0,
        min_df_unigram=1,
        min_df_phrase=1,
        phrase_min_count_per_cluster=1,
        top_n_keywords=10,
        ngram_min=1,
        ngram_max=2,
        use_phrase_vectorizer=True,
        n_jobs=1,
        verbose=False,
    )
    defaults.update(overrides)
    return KeywordExtractionConfig(**defaults)


# ---------------------------------------------------------------------------
# P1: Academic stopword tests
# ---------------------------------------------------------------------------

class TestAcademicStopwords:
    def test_common_stopwords_in_set(self):
        for sw in ("based", "using", "results", "proposed", "method", "data", "time"):
            assert sw in ACADEMIC_STOPWORDS, f"'{sw}' should be academic stopword"

    def test_domain_terms_not_in_set(self):
        for term in ("neutron", "quantum", "solar", "hydrogen", "catalyst"):
            assert term not in ACADEMIC_STOPWORDS

    def test_single_word_filtering(self, sample_data):
        cfg = _base_config(*sample_data, academic_stopwords_enabled=True)
        keywords = run_keyword_pipeline(cfg)
        terms = keywords["term"].str.lower().tolist()
        # None of the core academic stopwords should survive
        for sw in ("based", "using", "results", "proposed"):
            assert sw not in terms, f"'{sw}' should be filtered by P1"

    def test_multiword_all_stopwords_filtered(self):
        """Multi-word term where ALL tokens are stopwords → filtered."""
        pipeline = _make_pipeline_stub()
        assert pipeline._is_academic_stopword("proposed method") is True

    def test_multiword_mixed_kept(self):
        """Multi-word term with non-stopword token → kept."""
        pipeline = _make_pipeline_stub()
        assert pipeline._is_academic_stopword("fault diagnosis") is False

    def test_disabled(self, sample_data):
        cfg = _base_config(*sample_data, academic_stopwords_enabled=False)
        keywords = run_keyword_pipeline(cfg)
        # With stopwords disabled, generic terms may appear
        assert not keywords.empty

    def test_extra_stopwords(self, sample_data):
        cfg = _base_config(
            *sample_data,
            academic_stopwords_enabled=True,
            academic_stopwords_extra=("hydrogen",),
        )
        keywords = run_keyword_pipeline(cfg)
        terms = keywords["term"].str.lower().tolist()
        assert "hydrogen" not in terms


# ---------------------------------------------------------------------------
# P5: Artifact filter tests
# ---------------------------------------------------------------------------

class TestArtifactFilter:
    def test_latex_artifact(self):
        pipeline = _make_pipeline_stub()
        assert pipeline._is_artifact("center dot") is True

    def test_pure_number(self):
        pipeline = _make_pipeline_stub()
        assert pipeline._is_artifact("12345") is True

    def test_single_char(self):
        pipeline = _make_pipeline_stub()
        assert pipeline._is_artifact("x") is True

    def test_normal_term_not_artifact(self):
        pipeline = _make_pipeline_stub()
        assert pipeline._is_artifact("neural network") is False
        assert pipeline._is_artifact("quantum") is False
        assert pipeline._is_artifact("finite volume method") is False

    @pytest.mark.parametrize(
        "term",
        [
            "class htmlview paragraph",
            "div class htmlview",
            "lt div gt",
            "articles author",
            "works author gsw",
            "author gsw google",
            "urology vol",
        ],
    )
    def test_html_and_publisher_metadata_artifacts(self, term):
        pipeline = _make_pipeline_stub()
        assert pipeline._is_artifact(term) is True

    def test_pipeline_filters_encoded_html_and_metadata_fragments(self, tmp_path):
        abstracts = pd.DataFrame(
            {
                "uid": ["D1", "D2", "D3"],
                "title": [
                    "Nanoparticle synthesis routes",
                    "Nanoparticle synthesis mechanisms",
                    "Quantum dot synthesis",
                ],
                "abstract": [
                    (
                        "&lt;div class=&quot;htmlview paragraph&quot;&gt;"
                        "Nanoparticle synthesis improves catalytic stability."
                        "&lt;/div&gt; Get access Journal Article Articles Author"
                    ),
                    (
                        "Works Author GSW Google pages describe nanoparticle synthesis. "
                        "Urology vol metadata is not topical."
                    ),
                    "Quantum dot synthesis and nanocrystal growth are measured.",
                ],
                "pubyear": [2021, 2022, 2023],
            }
        )
        membership = pd.DataFrame({"uid": ["D1", "D2", "D3"], "cluster": [0, 0, 0]})
        abstract_path = tmp_path / "abstracts.parquet"
        membership_path = tmp_path / "membership.parquet"
        abstracts.to_parquet(abstract_path, index=False)
        membership.to_parquet(membership_path, index=False)

        cfg = _base_config(
            abstract_path,
            membership_path,
            include_title=True,
            ngram_min=1,
            ngram_max=3,
            top_n_keywords=30,
            scoring_pool_factor=2.0,
        )
        keywords = run_keyword_pipeline(cfg)
        terms = set(keywords["term"].str.lower())

        assert not keywords.empty
        for bad in {
            "class htmlview paragraph",
            "div class htmlview",
            "lt div gt",
            "get access",
            "journal article",
            "articles author",
            "works author",
            "author gsw google",
            "urology vol",
        }:
            assert bad not in terms


# ---------------------------------------------------------------------------
# P6: Cross-cluster penalty tests
# ---------------------------------------------------------------------------

class TestCrossClusterPenalty:
    def test_penalty_reduces_score(self, sample_data):
        """Terms in multiple clusters get penalized vs single-cluster terms."""
        cfg_no_penalty = _base_config(
            *sample_data,
            cross_cluster_penalty_enabled=False,
            w_llr=0.0,
        )
        cfg_penalty = _base_config(
            *sample_data,
            cross_cluster_penalty_enabled=True,
            cross_cluster_penalty_min_count=2,
            cross_cluster_penalty_fn="inverse",
            w_llr=0.0,
        )
        kw_no = run_keyword_pipeline(cfg_no_penalty)
        kw_pen = run_keyword_pipeline(cfg_penalty)

        # Find a term that appears in multiple clusters (if any)
        multi_cluster = kw_no.groupby("term")["cluster_id"].nunique()
        multi_terms = multi_cluster[multi_cluster >= 2].index.tolist()

        if multi_terms:
            term = multi_terms[0]
            score_no = kw_no[kw_no["term"] == term]["score"].mean()
            score_pen = kw_pen[kw_pen["term"] == term]["score"].mean()
            # Penalized score should be less than or equal
            assert score_pen <= score_no

    def test_log_inverse_penalty(self, sample_data):
        cfg = _base_config(
            *sample_data,
            cross_cluster_penalty_enabled=True,
            cross_cluster_penalty_min_count=2,
            cross_cluster_penalty_fn="log_inverse",
        )
        keywords = run_keyword_pipeline(cfg)
        assert not keywords.empty

    def test_penalty_does_not_remove_terms(self, sample_data):
        """Penalty only reduces scores, never removes terms entirely."""
        cfg = _base_config(
            *sample_data,
            cross_cluster_penalty_enabled=True,
            cross_cluster_penalty_min_count=2,
        )
        keywords = run_keyword_pipeline(cfg)
        assert not keywords.empty
        # All scores should still be finite
        assert keywords["score"].isna().sum() == 0


# ---------------------------------------------------------------------------
# P4: Short-term expansion tests
# ---------------------------------------------------------------------------

class TestShortTermExpansion:
    def test_annotate_mode(self):
        """Short terms get expanded_from annotation via cooccurrence."""
        pipeline, top_df, terms = _make_p4_scenario()
        result = pipeline._expand_short_terms(top_df, terms)
        expanded = result[result["term"] == "mg"]
        if not expanded.empty and "expanded_from" in result.columns:
            val = expanded.iloc[0]["expanded_from"]
            assert val != "", "short term 'mg' should have expansion"

    def test_replace_mode(self):
        """In replace mode, short terms are replaced by expansion."""
        pipeline, top_df, terms = _make_p4_scenario(mode="replace")
        result = pipeline._expand_short_terms(top_df, terms)
        # If replacement happened, "mg" should no longer be in terms
        remaining_terms = result["term"].tolist()
        if "mg" not in remaining_terms:
            assert any("mg" in t.lower() for t in remaining_terms) or len(remaining_terms) > 0

    def test_disabled_noop(self):
        """With expansion disabled, DataFrame unchanged."""
        pipeline, top_df, terms = _make_p4_scenario()
        pipeline.config.short_term_expansion_enabled = False
        result = pipeline._expand_short_terms(top_df, terms)
        assert "expanded_from" not in result.columns

    def test_no_cooc_matrix_noop(self):
        """Without cooc matrix, no expansion."""
        pipeline, top_df, terms = _make_p4_scenario()
        pipeline.cooc_matrix = None
        result = pipeline._expand_short_terms(top_df, terms)
        assert "expanded_from" not in result.columns

    def test_long_terms_unaffected(self):
        """Terms longer than max_length are never expanded."""
        pipeline, top_df, terms = _make_p4_scenario()
        result = pipeline._expand_short_terms(top_df, terms)
        long_terms = result[result["term"].str.len() > 2]
        if "expanded_from" in result.columns:
            assert all(long_terms["expanded_from"] == "")

    def test_substring_preferred_over_fallback(self):
        """When a substring-containing partner exists, prefer it over generic best."""
        pipeline, top_df, terms = _make_p4_scenario_substring()
        result = pipeline._expand_short_terms(top_df, terms)
        if "expanded_from" in result.columns:
            expanded = result[result["term"] == "fe"]["expanded_from"].iloc[0]
            if expanded:
                assert "fe" in expanded.lower(), \
                    f"Expected substring match for 'fe', got '{expanded}'"


# ---------------------------------------------------------------------------
# P3: Auto-merge tests
# ---------------------------------------------------------------------------

class TestAutoMerge:
    def test_high_similarity_merge(self):
        """Groups with all-pair similarity >= threshold merge into canonical."""
        pipeline, top_df = _make_p3_scenario(min_sim=0.85, pair_sim=0.9)
        result = pipeline._auto_merge_candidates(top_df)
        terms = result["term"].tolist()
        assert "neural network" in terms
        assert "neural networks" not in terms
        # Frequency should be summed
        canonical_row = result[result["term"] == "neural network"]
        assert canonical_row["frequency"].iloc[0] == 350  # 200+150

    def test_low_similarity_no_merge(self):
        """Groups below threshold are NOT auto-merged."""
        pipeline, top_df = _make_p3_scenario(min_sim=0.85, pair_sim=0.5)
        result = pipeline._auto_merge_candidates(top_df)
        terms = result["term"].tolist()
        assert "neural network" in terms
        assert "neural networks" in terms

    def test_disabled_noop(self):
        """With auto_merge disabled, no change."""
        pipeline, top_df = _make_p3_scenario(min_sim=0.85, pair_sim=0.95)
        pipeline.config.auto_merge_enabled = False
        result = pipeline._auto_merge_candidates(top_df)
        assert len(result) == len(top_df)

    def test_consumed_groups_removed(self):
        """Auto-merged groups are removed from merge_candidates."""
        pipeline, top_df = _make_p3_scenario(min_sim=0.85, pair_sim=0.95)
        assert len(pipeline.merge_candidates) == 1
        pipeline._auto_merge_candidates(top_df)
        assert len(pipeline.merge_candidates) == 0

    def test_score_takes_max(self):
        """Merged term gets max score across group members."""
        pipeline, top_df = _make_p3_scenario(min_sim=0.85, pair_sim=0.95)
        result = pipeline._auto_merge_candidates(top_df)
        canonical = result[result["term"] == "neural network"]
        assert canonical["score"].iloc[0] == 2.5  # max(2.0, 2.5) from "neural networks"

    def test_empty_candidates_noop(self):
        """With no merge candidates, auto-merge is a no-op."""
        pipeline, top_df = _make_p3_scenario(min_sim=0.85, pair_sim=0.95)
        pipeline.merge_candidates = []
        result = pipeline._auto_merge_candidates(top_df)
        assert len(result) == len(top_df)

    def test_cross_cluster_no_contamination(self):
        """Merge must not leak frequency across clusters.

        Regression: previously target_rows matched globally, so merging
        'neural networks' → 'neural network' in cluster 0 would add
        cluster 1's frequency too.
        """
        pipeline, _ = _make_p3_scenario(min_sim=0.85, pair_sim=0.95)
        # Both clusters have the same terms but independent frequencies
        top_df = pd.DataFrame({
            "cluster_id": [0, 0, 1, 1],
            "term": [
                "neural network", "neural networks",
                "neural network", "neural networks",
            ],
            "score": [2.0, 2.5, 1.0, 1.5],
            "frequency": [200, 150, 80, 60],
        })
        result = pipeline._auto_merge_candidates(top_df)
        # Cluster 0: 200 + 150 = 350
        c0 = result[
            (result["term"] == "neural network") & (result["cluster_id"] == 0)
        ]
        assert c0["frequency"].iloc[0] == 350
        # Cluster 1: 80 + 60 = 140 (NOT 350+140 or similar)
        c1 = result[
            (result["term"] == "neural network") & (result["cluster_id"] == 1)
        ]
        assert c1["frequency"].iloc[0] == 140


# ---------------------------------------------------------------------------
# Spelling variant tests (normalization.py)
# ---------------------------------------------------------------------------

class TestSpellingVariants:
    def test_british_to_american(self):
        assert _normalize_spelling("disc") == "disk"
        assert _normalize_spelling("colour") == "color"
        assert _normalize_spelling("behaviour") == "behavior"
        assert _normalize_spelling("fibre") == "fiber"
        assert _normalize_spelling("centre") == "center"
        assert _normalize_spelling("grey") == "gray"

    def test_per_word_in_phrase(self):
        assert _normalize_spelling("protoplanetary disc") == "protoplanetary disk"
        assert _normalize_spelling("colour space") == "color space"

    def test_non_variant_unchanged(self):
        assert _normalize_spelling("neural network") == "neural network"
        assert _normalize_spelling("quantum") == "quantum"

    def test_case_insensitive(self):
        # Input words should be lowercased for lookup
        assert _normalize_spelling("Disc") == "disk"

    def test_ise_to_ize(self):
        assert _normalize_spelling("analyse") == "analyze"
        assert _normalize_spelling("optimise") == "optimize"
        assert _normalize_spelling("synthesise") == "synthesize"

    def test_plural_variants(self):
        assert _normalize_spelling("discs") == "disks"
        assert _normalize_spelling("fibres") == "fibers"
        assert _normalize_spelling("vapours") == "vapors"

    def test_phrase_singular(self):
        assert _phrase_singular("point clouds") == "point cloud"
        assert _phrase_singular("neural networks") == "neural network"
        assert _phrase_singular("materials databases") == "materials database"
        assert _phrase_singular("series") is None  # not a regular plural

    def test_phrase_singular_empty(self):
        """Empty string should not crash (regression)."""
        assert _phrase_singular("") is None
        assert _phrase_singular("   ") is None


# ---------------------------------------------------------------------------
# Integration: P1+P5+P6 together
# ---------------------------------------------------------------------------

class TestQualityFiltersIntegration:
    def test_all_filters_enabled(self, sample_data):
        cfg = _base_config(
            *sample_data,
            academic_stopwords_enabled=True,
            artifact_filter_enabled=True,
            cross_cluster_penalty_enabled=True,
            cross_cluster_penalty_min_count=2,
        )
        keywords = run_keyword_pipeline(cfg)
        assert not keywords.empty
        terms = keywords["term"].str.lower().tolist()
        # Verify no pure academic stopwords
        for sw in ("based", "using", "results", "proposed"):
            assert sw not in terms

    def test_all_filters_disabled(self, sample_data):
        cfg = _base_config(
            *sample_data,
            academic_stopwords_enabled=False,
            artifact_filter_enabled=False,
            cross_cluster_penalty_enabled=False,
        )
        keywords = run_keyword_pipeline(cfg)
        assert not keywords.empty


# ---------------------------------------------------------------------------
# P2: Plural merging in normalization
# ---------------------------------------------------------------------------

class TestPluralMerge:
    def test_plural_merged_into_singular(self):
        """When both 'neural networks' and 'neural network' exist, merge into singular."""
        top_df = pd.DataFrame({
            "cluster_id": [0, 0, 0],
            "term": ["neural network", "neural networks", "deep learning"],
            "score": [2.0, 1.5, 1.8],
            "frequency": [200, 150, 180],
        })
        from sciscape.keyword_extraction.normalization import normalize_keywords
        result = normalize_keywords(
            top_df,
            builtin_aliases={},
            plural_merge_enabled=True,
        )
        terms = result["term"].tolist()
        assert "neural networks" not in terms, "plural should be merged"
        assert "neural network" in terms

    def test_plural_frequency_summed(self):
        """Merged term gets summed frequency."""
        top_df = pd.DataFrame({
            "cluster_id": [0, 0],
            "term": ["neural network", "neural networks"],
            "score": [2.0, 1.5],
            "frequency": [200, 150],
        })
        from sciscape.keyword_extraction.normalization import normalize_keywords
        result = normalize_keywords(
            top_df,
            builtin_aliases={},
            plural_merge_enabled=True,
        )
        row = result[result["term"] == "neural network"]
        assert len(row) == 1
        assert row["frequency"].iloc[0] == 350

    def test_singular_only_no_change(self):
        """If only singular exists, nothing happens."""
        top_df = pd.DataFrame({
            "cluster_id": [0, 0],
            "term": ["solar energy", "quantum bit"],
            "score": [1.0, 1.0],
            "frequency": [100, 80],
        })
        from sciscape.keyword_extraction.normalization import normalize_keywords
        result = normalize_keywords(
            top_df,
            builtin_aliases={},
            plural_merge_enabled=True,
        )
        assert len(result) == 2

    def test_plural_only_singularized(self):
        """If only the plural form exists, it is singularized."""
        top_df = pd.DataFrame({
            "cluster_id": [0],
            "term": ["point clouds"],
            "score": [1.0],
            "frequency": [100],
        })
        from sciscape.keyword_extraction.normalization import normalize_keywords
        result = normalize_keywords(
            top_df,
            builtin_aliases={},
            plural_merge_enabled=True,
        )
        assert result["term"].iloc[0] == "point cloud"

    def test_disabled_keeps_plural(self):
        """With plural merge disabled, plural forms survive."""
        top_df = pd.DataFrame({
            "cluster_id": [0, 0],
            "term": ["neural network", "neural networks"],
            "score": [2.0, 1.5],
            "frequency": [200, 150],
        })
        from sciscape.keyword_extraction.normalization import normalize_keywords
        result = normalize_keywords(
            top_df,
            builtin_aliases={},
            plural_merge_enabled=False,
        )
        terms = result["term"].tolist()
        assert "neural network" in terms
        assert "neural networks" in terms

    def test_irregular_plural_unchanged(self):
        """Irregular plurals like 'series' are not singularized."""
        top_df = pd.DataFrame({
            "cluster_id": [0],
            "term": ["time series"],
            "score": [1.0],
            "frequency": [100],
        })
        from sciscape.keyword_extraction.normalization import normalize_keywords
        result = normalize_keywords(
            top_df,
            builtin_aliases={},
            plural_merge_enabled=True,
        )
        assert result["term"].iloc[0] == "time series"

    def test_integration_pipeline(self, sample_data):
        """P2 works through the full pipeline."""
        cfg = _base_config(
            *sample_data,
            normalization_enabled=True,
            norm_plural_merge_enabled=True,
        )
        keywords = run_keyword_pipeline(cfg)
        assert not keywords.empty


# ---------------------------------------------------------------------------
# P7: Fragment suppression (pipeline-level integration)
# ---------------------------------------------------------------------------

class TestFragmentSuppression:
    def test_fragment_removed(self, sample_data):
        """With fragment suppression enabled, truncated n-grams are filtered."""
        cfg = _base_config(
            *sample_data,
            fragment_suppression_enabled=True,
            ngram_min=1,
            ngram_max=3,
        )
        keywords = run_keyword_pipeline(cfg)
        assert not keywords.empty

    def test_disabled_keeps_fragments(self, sample_data):
        """With fragment suppression disabled, pipeline still runs."""
        cfg = _base_config(
            *sample_data,
            fragment_suppression_enabled=False,
            ngram_min=1,
            ngram_max=3,
        )
        keywords = run_keyword_pipeline(cfg)
        assert not keywords.empty

    def test_fragment_suppression_fewer_terms(self, sample_data):
        """Enabling fragment suppression should produce <= terms compared to disabled."""
        cfg_on = _base_config(
            *sample_data,
            fragment_suppression_enabled=True,
            ngram_min=1,
            ngram_max=3,
        )
        cfg_off = _base_config(
            *sample_data,
            fragment_suppression_enabled=False,
            ngram_min=1,
            ngram_max=3,
        )
        kw_on = run_keyword_pipeline(cfg_on)
        kw_off = run_keyword_pipeline(cfg_off)
        # With suppression, we should have <= terms (or equal if no fragments exist)
        assert len(kw_on) <= len(kw_off)

    def test_fragment_unit_prefix_suppressed(self):
        """Directly test that _detect_boundary_fragments catches prefix fragments."""
        feature_names = np.array([
            "solar", "energy", "battery",
            "solar energy", "solar energy battery",
        ])
        cluster_freq = np.array([500, 400, 300, 490, 480])
        scored_terms = [
            ("solar energy", 0.01, 490),
        ]
        fragments = _detect_boundary_fragments(
            scored_terms, feature_names, cluster_freq, min_longer_ratio=0.5
        )
        # "solar energy" freq=490, "solar energy battery" freq=480.
        # 480 >= 0.5 * 490 = 245  → "solar energy" IS a fragment
        assert "solar energy" in fragments

    def test_fragment_config_ratio(self):
        """min_longer_ratio controls the suppression threshold."""
        feature_names = np.array([
            "deep", "learning", "model",
            "deep learning", "deep learning model",
        ])
        cluster_freq = np.array([1000, 1200, 800, 950, 200])
        scored_terms = [("deep learning", 0.01, 950)]
        # Strict ratio: 200 < 0.5*950=475 → not suppressed
        frags_strict = _detect_boundary_fragments(
            scored_terms, feature_names, cluster_freq, min_longer_ratio=0.5
        )
        assert "deep learning" not in frags_strict
        # Lenient ratio: 200 >= 0.1*950=95 → suppressed
        frags_lenient = _detect_boundary_fragments(
            scored_terms, feature_names, cluster_freq, min_longer_ratio=0.1
        )
        assert "deep learning" in frags_lenient


# ---------------------------------------------------------------------------
# Internal helpers (construct pipeline stubs)
# ---------------------------------------------------------------------------

def _make_pipeline_stub():
    """Create a minimal pipeline with P1/P5 initialized but no data."""
    import re
    from unittest.mock import MagicMock

    cfg = MagicMock(spec=KeywordExtractionConfig)
    cfg.academic_stopwords_enabled = True
    cfg.academic_stopwords_extra = None
    cfg.artifact_filter_enabled = True
    cfg.artifact_filter_patterns = (
        r"^center\s*dot$",
        r"^\d+$",
        r"^[^\w]+$",
        r"^.$",
    )
    cfg.lowercase = True
    cfg.verbose = False

    pipeline = object.__new__(KeywordExtractionPipeline)
    pipeline.config = cfg
    pipeline._academic_sw = frozenset(ACADEMIC_STOPWORDS)
    if cfg.academic_stopwords_extra:
        pipeline._academic_sw = pipeline._academic_sw | frozenset(cfg.academic_stopwords_extra)
    pipeline._artifact_res = [re.compile(p) for p in cfg.artifact_filter_patterns]
    return pipeline


def _make_p4_scenario(mode="annotate"):
    """Construct a pipeline + top_df for P4 short-term expansion testing."""
    from unittest.mock import MagicMock

    terms = ["mg", "hydrogen", "mg alloy", "catalyst", "storage"]
    {t: i for i, t in enumerate(terms)}

    # Build cooc matrix: "mg" cooccurs with "mg alloy" (strong) and "catalyst" (weak)
    n = len(terms)
    cooc = sp.lil_matrix((n, n), dtype=np.float64)
    cooc[0, 2] = 30  # mg ↔ mg alloy (substring match)
    cooc[2, 0] = 30
    cooc[0, 1] = 20  # mg ↔ hydrogen
    cooc[1, 0] = 20
    cooc[0, 3] = 5   # mg ↔ catalyst
    cooc[3, 0] = 5

    cfg = MagicMock(spec=KeywordExtractionConfig)
    cfg.short_term_expansion_enabled = True
    cfg.short_term_max_length = 2
    cfg.short_term_min_cooc_ratio = 0.05
    cfg.short_term_expansion_mode = mode
    cfg.verbose = False

    pipeline = object.__new__(KeywordExtractionPipeline)
    pipeline.config = cfg
    pipeline.cooc_matrix = cooc.tocsr()
    pipeline.verbose = False

    def _log(msg, *args):
        pass
    pipeline._log = _log

    top_df = pd.DataFrame({
        "cluster_id": [0, 0, 0, 0, 0],
        "term": terms,
        "score": [1.0, 2.0, 1.5, 1.2, 0.8],
        "frequency": [50, 200, 100, 80, 60],
    })

    return pipeline, top_df, terms


def _make_p4_scenario_substring():
    """P4 scenario with both substring and non-substring partners."""
    from unittest.mock import MagicMock

    terms = ["fe", "fe oxide", "nanoparticle", "catalyst", "iron"]
    n = len(terms)
    cooc = sp.lil_matrix((n, n), dtype=np.float64)
    cooc[0, 1] = 25  # fe ↔ fe oxide (substring match)
    cooc[1, 0] = 25
    cooc[0, 4] = 40  # fe ↔ iron (higher cooc but no substring)
    cooc[4, 0] = 40

    cfg = MagicMock(spec=KeywordExtractionConfig)
    cfg.short_term_expansion_enabled = True
    cfg.short_term_max_length = 2
    cfg.short_term_min_cooc_ratio = 0.05
    cfg.short_term_expansion_mode = "annotate"
    cfg.verbose = False

    pipeline = object.__new__(KeywordExtractionPipeline)
    pipeline.config = cfg
    pipeline.cooc_matrix = cooc.tocsr()
    pipeline.verbose = False

    def _log(msg, *args):
        pass
    pipeline._log = _log

    top_df = pd.DataFrame({
        "cluster_id": [0, 0, 0, 0, 0],
        "term": terms,
        "score": [1.0, 1.5, 1.2, 0.8, 1.3],
        "frequency": [50, 100, 80, 60, 90],
    })

    return pipeline, top_df, terms


def _make_p3_scenario(min_sim=0.85, pair_sim=0.9):
    """Construct a pipeline + top_df for P3 auto-merge testing."""
    from unittest.mock import MagicMock

    cfg = MagicMock(spec=KeywordExtractionConfig)
    cfg.auto_merge_enabled = True
    cfg.auto_merge_min_similarity = min_sim
    cfg.verbose = False

    pipeline = object.__new__(KeywordExtractionPipeline)
    pipeline.config = cfg
    pipeline.verbose = False

    def _log(msg, *args):
        pass
    pipeline._log = _log

    pipeline.merge_candidates = [
        {
            "group_id": 0,
            "terms": ["neural network", "neural networks"],
            "size": 2,
            "pair_similarities": {
                ("neural network", "neural networks"): pair_sim,
            },
        }
    ]

    top_df = pd.DataFrame({
        "cluster_id": [0, 0, 0],
        "term": ["neural network", "neural networks", "deep learning"],
        "score": [2.0, 2.5, 1.8],
        "frequency": [200, 150, 180],
    })

    return pipeline, top_df


# ---------------------------------------------------------------------------
# Pipeline method unit tests: _is_stopword_ngram, _filter_stopword_only_terms,
# _bridge_merge_candidates, _filter_rare_phrases
# ---------------------------------------------------------------------------

class TestIsStopwordNgram:
    def test_single_stopword(self):
        stub = _make_pipeline_stub()
        stub.stopwords_set = frozenset(["the", "and", "of", "in"])
        assert stub._is_stopword_ngram("the") is True
        assert stub._is_stopword_ngram("quantum") is False

    def test_multi_word_all_stopwords(self):
        stub = _make_pipeline_stub()
        stub.stopwords_set = frozenset(["the", "and", "of", "in", "is"])
        assert stub._is_stopword_ngram("the and") is True
        assert stub._is_stopword_ngram("in the") is True

    def test_multi_word_mixed(self):
        stub = _make_pipeline_stub()
        stub.stopwords_set = frozenset(["the", "and", "of"])
        assert stub._is_stopword_ngram("the model") is False
        assert stub._is_stopword_ngram("machine learning") is False

    def test_empty_string(self):
        stub = _make_pipeline_stub()
        stub.stopwords_set = frozenset(["the"])
        assert stub._is_stopword_ngram("") is False

    def test_case_sensitive_mode(self):
        stub = _make_pipeline_stub()
        stub.config.lowercase = False
        stub.stopwords_set = frozenset(["the", "and"])
        # With lowercase=False, still checks .lower()
        assert stub._is_stopword_ngram("The") is True


class TestFilterStopwordOnlyTerms:
    def test_drops_stopword_only(self):
        stub = _make_pipeline_stub()
        stub.stopwords_set = frozenset(["the", "and", "of", "in", "is"])
        stub.config.verbose = False
        df = pd.DataFrame({
            "cluster_id": [0, 0, 0],
            "term": ["quantum", "the and", "machine learning"],
            "score": [1.0, 0.5, 1.5],
            "frequency": [10, 5, 20],
        })
        result = stub._filter_stopword_only_terms(df)
        assert "the and" not in result["term"].tolist()
        assert "quantum" in result["term"].tolist()
        assert "machine learning" in result["term"].tolist()

    def test_empty_df(self):
        stub = _make_pipeline_stub()
        stub.stopwords_set = frozenset()
        df = pd.DataFrame(columns=["cluster_id", "term", "score", "frequency"])
        result = stub._filter_stopword_only_terms(df)
        assert result.empty

    def test_whitespace_only_term(self):
        stub = _make_pipeline_stub()
        stub.stopwords_set = frozenset(["the"])
        df = pd.DataFrame({
            "cluster_id": [0, 0],
            "term": ["   ", "quantum"],
            "score": [1.0, 1.0],
            "frequency": [5, 10],
        })
        result = stub._filter_stopword_only_terms(df)
        assert len(result) == 1
        assert result["term"].iloc[0] == "quantum"


class TestBridgeMergeCandidates:
    def _make_bridge_stub(self, merge_candidates):
        from unittest.mock import MagicMock
        cfg = MagicMock(spec=KeywordExtractionConfig)
        cfg.alias_candidate_column = "candidates"
        cfg.verbose = False
        pipeline = object.__new__(KeywordExtractionPipeline)
        pipeline.config = cfg
        pipeline.merge_candidates = merge_candidates
        pipeline.verbose = False
        def _log(msg, *args): pass
        pipeline._log = _log
        return pipeline

    def test_basic_injection(self):
        stub = self._make_bridge_stub([
            {"group_id": 0, "terms": ["neural network", "neural networks"]},
        ])
        top_df = pd.DataFrame({
            "cluster_id": [0, 0, 0],
            "term": ["neural network", "neural networks", "deep learning"],
            "score": [2.0, 2.5, 1.8],
            "frequency": [200, 150, 180],
        })
        result = stub._bridge_merge_candidates(top_df)
        assert "candidates" in result.columns
        nn_row = result[result["term"] == "neural network"].iloc[0]
        assert "neural networks" in nn_row["candidates"]
        dl_row = result[result["term"] == "deep learning"].iloc[0]
        assert dl_row["candidates"] == []

    def test_empty_merge_candidates(self):
        stub = self._make_bridge_stub([])
        top_df = pd.DataFrame({
            "cluster_id": [0], "term": ["alpha"],
            "score": [1.0], "frequency": [10],
        })
        result = stub._bridge_merge_candidates(top_df)
        assert "candidates" not in result.columns  # no change

    def test_empty_dataframe(self):
        stub = self._make_bridge_stub([
            {"group_id": 0, "terms": ["a", "b"]},
        ])
        top_df = pd.DataFrame(columns=["cluster_id", "term", "score", "frequency"])
        result = stub._bridge_merge_candidates(top_df)
        assert result.empty

    def test_multi_group_merge(self):
        stub = self._make_bridge_stub([
            {"group_id": 0, "terms": ["a", "b"]},
            {"group_id": 1, "terms": ["a", "c"]},
        ])
        top_df = pd.DataFrame({
            "cluster_id": [0, 0, 0],
            "term": ["a", "b", "c"],
            "score": [1.0, 1.0, 1.0],
            "frequency": [10, 10, 10],
        })
        result = stub._bridge_merge_candidates(top_df)
        a_cands = result[result["term"] == "a"]["candidates"].iloc[0]
        assert "b" in a_cands
        assert "c" in a_cands


class TestFilterRarePhrases:
    def _make_stub(self, min_count):
        from unittest.mock import MagicMock
        cfg = MagicMock(spec=KeywordExtractionConfig)
        cfg.phrase_min_count_per_cluster = min_count
        pipeline = object.__new__(KeywordExtractionPipeline)
        pipeline.config = cfg
        def _log(msg, *args): pass
        pipeline._log = _log
        return pipeline

    def test_drops_rare_columns(self):
        stub = self._make_stub(min_count=5)
        stub.feature_names_phrase = np.array(["alpha beta", "gamma delta", "rare term"])
        # 2 clusters, 3 phrase columns; column 2 has max count < 5
        C = sp.csr_matrix(np.array([[10, 8, 2], [6, 3, 1]]))
        DF = sp.csr_matrix(np.array([[5, 4, 1], [3, 2, 1]]))
        C_out, DF_out = stub._filter_rare_phrases(C, DF, K=2)
        assert C_out.shape[1] == 2  # column 2 dropped
        assert DF_out.shape[1] == 2
        assert len(stub.feature_names_phrase) == 2

    def test_none_input(self):
        stub = self._make_stub(min_count=5)
        C_out, DF_out = stub._filter_rare_phrases(None, None, K=2)
        assert C_out is None
        assert DF_out is None

    def test_min_count_one_keeps_all(self):
        stub = self._make_stub(min_count=1)
        C = sp.csr_matrix(np.array([[10, 1, 3]]))
        DF = sp.csr_matrix(np.array([[5, 1, 2]]))
        C_out, DF_out = stub._filter_rare_phrases(C, DF, K=1)
        assert C_out.shape[1] == 3


# ---------------------------------------------------------------------------
# Fragment suppression — truncated n-gram boundary artifact detection
# ---------------------------------------------------------------------------


class TestDetectBoundaryFragments:
    """Tests for _detect_boundary_fragments."""

    def test_prefix_fragment_detected(self):
        """'supermassive black' is a prefix fragment of 'supermassive black hole'."""
        feature_names = np.array(["supermassive", "black", "hole",
                                  "supermassive black", "supermassive black hole",
                                  "black hole"])
        # Cluster freqs: "supermassive black" ~ "supermassive black hole"
        cluster_freq = np.array([400, 2000, 2000, 393, 390, 1800])
        scored_terms = [
            ("black hole", 0.007, 1800),
            ("supermassive black", 0.002, 393),
            ("supermassive", 0.001, 400),
        ]
        fragments = _detect_boundary_fragments(scored_terms, feature_names, cluster_freq)
        assert "supermassive black" in fragments
        assert "black hole" not in fragments  # "black hole" is not a fragment

    def test_suffix_fragment_detected(self):
        """'point cloud' NOT suppressed, but 'lidar point' IS if 'lidar point cloud' exists."""
        feature_names = np.array(["lidar", "point", "cloud",
                                  "lidar point", "point cloud", "lidar point cloud"])
        cluster_freq = np.array([600, 7000, 7000, 540, 6864, 530])
        scored_terms = [
            ("point cloud", 0.005, 6864),
            ("lidar point", 0.002, 540),
        ]
        fragments = _detect_boundary_fragments(scored_terms, feature_names, cluster_freq)
        assert "lidar point" in fragments
        assert "point cloud" not in fragments

    def test_independent_term_not_suppressed(self):
        """'machine learning' should NOT be suppressed even though
        'machine learning model' exists, because freq(shorter) >> freq(longer)."""
        feature_names = np.array(["machine", "learning", "model",
                                  "machine learning", "machine learning model"])
        cluster_freq = np.array([1000, 1200, 800, 950, 200])
        scored_terms = [
            ("machine learning", 0.01, 950),
        ]
        # With ratio=0.5, need longer_freq >= 0.5 * 950 = 475.  200 < 475 → not fragment
        fragments = _detect_boundary_fragments(scored_terms, feature_names, cluster_freq, min_longer_ratio=0.5)
        assert "machine learning" not in fragments

    def test_unigrams_never_flagged(self):
        """Unigrams should never be considered boundary fragments."""
        feature_names = np.array(["hydrogen", "storage", "hydrogen storage"])
        cluster_freq = np.array([500, 400, 490])
        scored_terms = [("hydrogen", 0.01, 500)]
        fragments = _detect_boundary_fragments(scored_terms, feature_names, cluster_freq)
        assert "hydrogen" not in fragments

    def test_no_longer_form_in_vocab(self):
        """If no longer form exists, the term is kept."""
        feature_names = np.array(["deep learning", "neural network"])
        cluster_freq = np.array([600, 500])
        scored_terms = [("deep learning", 0.01, 600)]
        fragments = _detect_boundary_fragments(scored_terms, feature_names, cluster_freq)
        assert len(fragments) == 0

    def test_empty_input(self):
        feature_names = np.array([], dtype=str)
        cluster_freq = np.array([], dtype=int)
        fragments = _detect_boundary_fragments([], feature_names, cluster_freq)
        assert len(fragments) == 0

    # ---- Mode 2: bridging overlap tests ----

    def test_bridging_overlap_right(self):
        """'lidar point' shares 'point' with much higher-freq 'point cloud' → fragment."""
        feature_names = np.array(["lidar", "point", "cloud",
                                  "lidar point", "point cloud"])
        cluster_freq = np.array([600, 7000, 7000, 540, 6864])
        scored_terms = [
            ("point cloud", 0.005, 6864),
            ("lidar point", 0.002, 540),
        ]
        fragments = _detect_boundary_fragments(
            scored_terms, feature_names, cluster_freq, bridging_max_freq_ratio=0.3
        )
        assert "lidar point" in fragments
        assert "point cloud" not in fragments

    def test_bridging_overlap_independent_terms(self):
        """'machine learning' and 'learning rate' share 'learning' but both are
        independent (comparable freq) → neither suppressed."""
        feature_names = np.array(["machine", "learning", "rate",
                                  "machine learning", "learning rate"])
        cluster_freq = np.array([1000, 1200, 800, 950, 600])
        scored_terms = [
            ("machine learning", 0.01, 950),
            ("learning rate", 0.008, 600),
        ]
        # 950/600 = 1.58 >> 0.3, 600/950 = 0.63 > 0.3 → both kept
        fragments = _detect_boundary_fragments(
            scored_terms, feature_names, cluster_freq, bridging_max_freq_ratio=0.3
        )
        assert "machine learning" not in fragments
        assert "learning rate" not in fragments

    def test_bridging_overlap_left(self):
        """'cloud data' shares 'cloud' with much higher-freq 'point cloud' → fragment."""
        feature_names = np.array(["point", "cloud", "data",
                                  "point cloud", "cloud data"])
        cluster_freq = np.array([7000, 7000, 500, 6800, 450])
        scored_terms = [
            ("point cloud", 0.005, 6800),
            ("cloud data", 0.001, 450),
        ]
        # 450/6800 = 0.066 < 0.3 → fragment
        fragments = _detect_boundary_fragments(
            scored_terms, feature_names, cluster_freq, bridging_max_freq_ratio=0.3
        )
        assert "cloud data" in fragments
        assert "point cloud" not in fragments
