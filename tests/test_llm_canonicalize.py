"""Tests for LLM canonicalization mixin (Stage 8).

Tests cover the non-LLM logic: JSON parsing, term cleaning,
alias application, cache save/load, and candidate enforcement.
LLM invocation itself is not tested (external dependency).
"""

import json
from pathlib import Path

import pandas as pd

from sciscape.keyword_extraction.config import KeywordExtractionConfig
from sciscape.keyword_extraction.llm_canonicalize import LLMCanonicalizeMixin
from sciscape.keyword_extraction.pipeline import KeywordExtractionPipeline


# ---------------------------------------------------------------------------
# Helpers — create a minimal mixin instance without full pipeline init
# ---------------------------------------------------------------------------

def _make_mixin(tmp_path=None, **config_overrides):
    """Create a LLMCanonicalizeMixin stub with minimal state."""
    defaults = dict(
        abstract_path=Path("/tmp/a.parquet"),
        membership_path=Path("/tmp/m.parquet"),
        lowercase=True,
        builtin_aliases={
            "bq": "becquerel",
            "sv": "sievert",
            "gy": "gray",
        },
        forbid_abbreviations=("bq", "sv", "gy"),
        alias_stopword_strictness="drop_if_empty",
        alias_strategy="none",
        apply_alias_map=False,
        alias_candidate_column="candidates",
        alias_candidate_max=15,
        alias_candidate_enforce=True,
        alias_model="test-model",
        alias_allow_translation=True,
        alias_cache_enabled=True,
        alias_cache_key_fields=("term", "frequency", "doc_coverage", "score"),
        top_n_keywords=10,
    )
    defaults.update(config_overrides)
    cfg = KeywordExtractionConfig(**defaults)

    mixin = object.__new__(LLMCanonicalizeMixin)
    mixin.config = cfg
    mixin.verbose = False
    mixin._alias_cache_dir = tmp_path
    mixin._alias_client = None
    mixin._builtin_alias_cache = None
    mixin.stopwords_set = {"the", "a", "an", "is", "are", "of", "in", "to", "and"}

    def _log(msg, *args):
        pass
    mixin._log = _log

    return mixin


# ---------------------------------------------------------------------------
# _safe_json_loads
# ---------------------------------------------------------------------------

class TestSafeJsonLoads:
    def test_valid_json_list(self):
        result = LLMCanonicalizeMixin._safe_json_loads('[{"term": "a", "action": "keep"}]')
        assert isinstance(result, list)
        assert result[0]["term"] == "a"

    def test_valid_json_dict(self):
        result = LLMCanonicalizeMixin._safe_json_loads('{"items": []}')
        assert isinstance(result, dict)

    def test_markdown_fence(self):
        raw = '```json\n[{"term": "x"}]\n```'
        result = LLMCanonicalizeMixin._safe_json_loads(raw)
        assert isinstance(result, list)

    def test_json_with_preamble(self):
        raw = 'Here are the results: [{"term": "y"}]'
        result = LLMCanonicalizeMixin._safe_json_loads(raw)
        assert isinstance(result, list)

    def test_empty_string(self):
        assert LLMCanonicalizeMixin._safe_json_loads("") is None

    def test_none(self):
        assert LLMCanonicalizeMixin._safe_json_loads(None) is None

    def test_invalid_json(self):
        assert LLMCanonicalizeMixin._safe_json_loads("not json at all") is None

    def test_nested_braces(self):
        raw = '{"items": [{"term": "a", "action": "keep"}]}'
        result = LLMCanonicalizeMixin._safe_json_loads(raw)
        assert result["items"][0]["term"] == "a"


# ---------------------------------------------------------------------------
# _clean_canonical_term
# ---------------------------------------------------------------------------

class TestCleanCanonicalTerm:
    def test_basic_cleaning(self):
        mixin = _make_mixin()
        assert mixin._clean_canonical_term("  Neural Network  ") == "neural network"

    def test_html_removal(self):
        mixin = _make_mixin()
        assert mixin._clean_canonical_term("<b>bold</b> text") == "bold text"

    def test_micro_symbol(self):
        mixin = _make_mixin()
        assert mixin._clean_canonical_term("\u00b5sv") == "usv"
        assert mixin._clean_canonical_term("\u03bcsv") == "usv"

    def test_unit_singularization(self):
        mixin = _make_mixin()
        assert mixin._clean_canonical_term("becquerels") == "becquerel"
        assert mixin._clean_canonical_term("sieverts") == "sievert"
        assert mixin._clean_canonical_term("millisieverts") == "millisievert"

    def test_empty_input(self):
        mixin = _make_mixin()
        assert mixin._clean_canonical_term("") == ""
        assert mixin._clean_canonical_term(None) == ""

    def test_no_lowercase_mode(self):
        mixin = _make_mixin(lowercase=False)
        result = mixin._clean_canonical_term("Neural Network")
        assert result == "Neural Network"


# ---------------------------------------------------------------------------
# _builtin_alias
# ---------------------------------------------------------------------------

class TestBuiltinAlias:
    def test_known_alias(self):
        mixin = _make_mixin()
        assert mixin._builtin_alias("bq") == "becquerel"
        assert mixin._builtin_alias("BQ") == "becquerel"
        assert mixin._builtin_alias("  sv  ") == "sievert"

    def test_unknown_term(self):
        mixin = _make_mixin()
        assert mixin._builtin_alias("neutron") is None

    def test_empty_aliases(self):
        mixin = _make_mixin(builtin_aliases={})
        assert mixin._builtin_alias("bq") is None

    def test_cache_initialized_once(self):
        mixin = _make_mixin()
        _ = mixin._builtin_alias("bq")
        cache = mixin._builtin_alias_cache
        _ = mixin._builtin_alias("sv")
        assert mixin._builtin_alias_cache is cache  # same dict object


# ---------------------------------------------------------------------------
# _parse_alias_items
# ---------------------------------------------------------------------------

class TestParseAliasItems:
    def _subset_df(self):
        return pd.DataFrame({
            "cluster_id": [0, 0, 0],
            "term": ["neural network", "deep learning", "cnn"],
            "score": [2.0, 1.5, 1.0],
            "frequency": [200, 150, 50],
            "doc_coverage": [80, 60, 20],
        })

    def test_valid_json_list(self):
        mixin = _make_mixin()
        raw = json.dumps([
            {"term": "neural network", "action": "keep", "canonical": "neural network"},
            {"term": "deep learning", "action": "keep", "canonical": "deep learning"},
            {"term": "cnn", "action": "merge_into", "canonical": "convolutional neural network"},
        ])
        items = mixin._parse_alias_items(0, raw, self._subset_df())
        assert len(items) == 3
        cnn_item = [i for i in items if i["original"] == "cnn"][0]
        assert cnn_item["action"] == "merge_into"
        assert cnn_item["canonical"] == "convolutional neural network"

    def test_valid_json_dict_with_items(self):
        mixin = _make_mixin()
        raw = json.dumps({
            "items": [
                {"term": "neural network", "action": "keep", "canonical": "neural network"},
            ]
        })
        items = mixin._parse_alias_items(0, raw, self._subset_df())
        # 1 from JSON + 2 default_keep for missing terms
        assert len(items) == 3
        defaults = [i for i in items if i["reason"] == "default_keep"]
        assert len(defaults) == 2

    def test_missing_terms_get_default_keep(self):
        mixin = _make_mixin()
        raw = json.dumps([
            {"term": "neural network", "action": "keep", "canonical": "neural network"},
        ])
        items = mixin._parse_alias_items(0, raw, self._subset_df())
        assert len(items) == 3
        missing = [i for i in items if i["original"] == "deep learning"]
        assert missing[0]["action"] == "keep"
        assert missing[0]["reason"] == "default_keep"

    def test_invalid_json_returns_all_default(self):
        mixin = _make_mixin()
        items = mixin._parse_alias_items(0, "not json", self._subset_df())
        assert len(items) == 3
        assert all(i["action"] == "keep" for i in items)

    def test_drop_action_preserved(self):
        mixin = _make_mixin()
        raw = json.dumps([
            {"term": "cnn", "action": "drop", "canonical": "", "reason": "junk"},
        ])
        items = mixin._parse_alias_items(0, raw, self._subset_df())
        cnn = [i for i in items if i["original"] == "cnn"][0]
        assert cnn["action"] == "drop"


# ---------------------------------------------------------------------------
# _apply_alias_instructions
# ---------------------------------------------------------------------------

class TestApplyAliasInstructions:
    def _top_df(self):
        return pd.DataFrame({
            "cluster_id": [0, 0, 0],
            "term": ["neural network", "neural networks", "deep learning"],
            "score": [2.0, 1.5, 1.8],
            "frequency": [200, 150, 180],
            "doc_coverage": [80, 60, 70],
        })

    def _alias_df(self):
        return pd.DataFrame({
            "cluster_id": [0, 0, 0],
            "original": ["neural network", "neural networks", "deep learning"],
            "action": ["keep", "merge_into", "keep"],
            "canonical": ["neural network", "neural network", "deep learning"],
            "notes": ["", "plural merge", ""],
            "reason": ["", "duplicate", ""],
        })

    def test_merge_sums_frequency(self):
        mixin = _make_mixin()
        result = mixin._apply_alias_instructions(self._top_df(), self._alias_df())
        nn_row = result[result["term"] == "neural network"]
        assert len(nn_row) == 1
        assert nn_row.iloc[0]["frequency"] == 350  # 200+150

    def test_drop_removes_term(self):
        mixin = _make_mixin()
        alias_df = self._alias_df().copy()
        alias_df.loc[alias_df["original"] == "deep learning", "action"] = "drop"
        result = mixin._apply_alias_instructions(self._top_df(), alias_df)
        assert "deep learning" not in result["term"].tolist()

    def test_keep_preserves_term(self):
        mixin = _make_mixin()
        result = mixin._apply_alias_instructions(self._top_df(), self._alias_df())
        assert "deep learning" in result["term"].tolist()

    def test_source_terms_tracked(self):
        mixin = _make_mixin()
        result = mixin._apply_alias_instructions(self._top_df(), self._alias_df())
        nn = result[result["term"] == "neural network"].iloc[0]
        assert "neural network" in nn["source_terms"]
        assert "neural networks" in nn["source_terms"]

    def test_empty_top_df(self):
        mixin = _make_mixin()
        empty = pd.DataFrame(columns=["cluster_id", "term", "score", "frequency", "doc_coverage"])
        alias = pd.DataFrame(columns=["cluster_id", "original", "action", "canonical", "notes", "reason"])
        result = mixin._apply_alias_instructions(empty, alias)
        assert "source_terms" in result.columns

    def test_stopword_drop(self):
        """Pure stopword canonical forms should be dropped (when other terms remain)."""
        mixin = _make_mixin()
        top = pd.DataFrame({
            "cluster_id": [0, 0],
            "term": ["the", "quantum"],
            "score": [0.1, 2.0],
            "frequency": [10, 200],
            "doc_coverage": [5, 80],
        })
        alias = pd.DataFrame({
            "cluster_id": [0, 0],
            "original": ["the", "quantum"],
            "action": ["keep", "keep"],
            "canonical": ["the", "quantum"],
            "notes": ["", ""],
            "reason": ["", ""],
        })
        result = mixin._apply_alias_instructions(top, alias)
        assert "the" not in result["term"].tolist()
        assert "quantum" in result["term"].tolist()


# ---------------------------------------------------------------------------
# _enforce_forbidden
# ---------------------------------------------------------------------------

class TestEnforceForbidden:
    def test_replaces_forbidden_with_alias(self):
        mixin = _make_mixin()
        df = pd.DataFrame({
            "cluster_id": [0, 0],
            "term": ["bq", "neutron"],
            "score": [1.0, 2.0],
            "frequency": [50, 200],
            "doc_coverage": [10, 80],
            "source_terms": [["bq"], ["neutron"]],
            "alias_actions": [["keep"], ["keep"]],
        })
        result = mixin._enforce_forbidden(df)
        terms = result["term"].tolist()
        assert "bq" not in terms
        assert "becquerel" in terms
        assert "neutron" in terms

    def test_drops_if_no_replacement(self):
        mixin = _make_mixin(
            builtin_aliases={},
            forbid_abbreviations=("xyz",),
        )
        df = pd.DataFrame({
            "cluster_id": [0],
            "term": ["xyz"],
            "score": [1.0],
            "frequency": [50],
            "doc_coverage": [10],
            "source_terms": [["xyz"]],
            "alias_actions": [["keep"]],
        })
        result = mixin._enforce_forbidden(df)
        assert result.empty

    def test_empty_forbidden_noop(self):
        mixin = _make_mixin(forbid_abbreviations=())
        df = pd.DataFrame({
            "cluster_id": [0],
            "term": ["bq"],
            "score": [1.0],
            "frequency": [50],
            "doc_coverage": [10],
            "source_terms": [["bq"]],
            "alias_actions": [["keep"]],
        })
        result = mixin._enforce_forbidden(df)
        assert "bq" in result["term"].tolist()


# ---------------------------------------------------------------------------
# Cache save/load roundtrip
# ---------------------------------------------------------------------------

class TestAliasCacheRoundtrip:
    def test_save_and_load_mapping(self, tmp_path):
        mixin = _make_mixin(tmp_path=tmp_path)
        mapping = {
            "neural network": {
                "original": "neural network",
                "action": "keep",
                "canonical": "neural network",
                "notes": "",
                "reason": "",
            },
            "nn": {
                "original": "nn",
                "action": "merge_into",
                "canonical": "neural network",
                "notes": "abbreviation",
                "reason": "initials",
            },
        }
        mixin._save_alias_mapping(0, mapping)
        loaded = mixin._load_alias_mapping(0)
        assert "neural network" in loaded
        assert loaded["nn"]["action"] == "merge_into"
        assert loaded["nn"]["canonical"] == "neural network"

    def test_load_missing_mapping(self, tmp_path):
        mixin = _make_mixin(tmp_path=tmp_path)
        loaded = mixin._load_alias_mapping(999)
        assert loaded == {}

    def test_load_corrupted_mapping(self, tmp_path):
        mixin = _make_mixin(tmp_path=tmp_path)
        mapping_dir = tmp_path / "mapping"
        mapping_dir.mkdir(parents=True)
        (mapping_dir / "0.json").write_text("NOT JSON", encoding="utf-8")
        loaded = mixin._load_alias_mapping(0)
        assert loaded == {}

    def test_save_and_load_cache(self, tmp_path):
        mixin = _make_mixin(tmp_path=tmp_path)
        payload = {"terms": [{"term": "test"}]}
        raw = '[{"term": "test", "action": "keep"}]'
        batch_hash = "abc123"
        mixin._save_alias_cache(0, batch_hash, payload, raw)
        loaded = mixin._load_alias_cache(0, batch_hash)
        assert loaded == raw

    def test_load_missing_cache(self, tmp_path):
        mixin = _make_mixin(tmp_path=tmp_path)
        assert mixin._load_alias_cache(0, "nonexistent") is None

    def test_cache_disabled(self):
        mixin = _make_mixin(tmp_path=None)
        assert mixin._alias_cache_enabled() is False
        assert mixin._load_alias_cache(0, "hash") is None


# ---------------------------------------------------------------------------
# _build_alias_messages
# ---------------------------------------------------------------------------

class TestBuildAliasMessages:
    def test_message_structure(self):
        mixin = _make_mixin()
        subset = pd.DataFrame({
            "cluster_id": [0, 0],
            "term": ["neural network", "deep learning"],
            "score": [2.0, 1.5],
            "frequency": [200, 150],
            "doc_coverage": [80, 60],
        })
        messages, payload = mixin._build_alias_messages(0, subset)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "keep" in messages[0]["content"]
        assert "merge_into" in messages[0]["content"]
        assert payload["cluster_id"] == 0
        assert len(payload["terms"]) == 2

    def test_payload_contains_terms(self):
        mixin = _make_mixin()
        subset = pd.DataFrame({
            "cluster_id": [0],
            "term": ["quantum bit"],
            "score": [1.0],
            "frequency": [100],
            "doc_coverage": [40],
        })
        _, payload = mixin._build_alias_messages(0, subset)
        assert payload["terms"][0]["term"] == "quantum bit"
        assert payload["terms"][0]["frequency"] == 100


# ---------------------------------------------------------------------------
# _maybe_canonicalise (integration — disabled/noop paths)
# ---------------------------------------------------------------------------

class TestMaybeCanonicalize:
    def test_disabled_returns_same(self, tmp_path):
        """When apply_alias_map=False, returns input unchanged."""
        abstract_path, membership_path = _make_sample_data(tmp_path)
        cfg = KeywordExtractionConfig(
            abstract_path=abstract_path,
            membership_path=membership_path,
            cluster_level="cluster",
            apply_alias_map=False,
            alias_strategy="none",
            min_df_unigram=1,
            min_df_phrase=1,
            phrase_min_count_per_cluster=1,
            top_n_keywords=5,
            n_jobs=1,
        )
        pipeline = KeywordExtractionPipeline(cfg)
        top_df = pd.DataFrame({
            "cluster_id": [0, 0],
            "term": ["quantum", "neural"],
            "score": [2.0, 1.5],
            "frequency": [200, 150],
        })
        result = pipeline._maybe_canonicalise(top_df)
        assert len(result) == 2

    def test_empty_df(self, tmp_path):
        abstract_path, membership_path = _make_sample_data(tmp_path)
        cfg = KeywordExtractionConfig(
            abstract_path=abstract_path,
            membership_path=membership_path,
            cluster_level="cluster",
            apply_alias_map=True,
            alias_strategy="llm",
            min_df_unigram=1,
            min_df_phrase=1,
            phrase_min_count_per_cluster=1,
            top_n_keywords=5,
            n_jobs=1,
        )
        pipeline = KeywordExtractionPipeline(cfg)
        empty = pd.DataFrame(columns=["cluster_id", "term", "score", "frequency"])
        result = pipeline._maybe_canonicalise(empty)
        assert result.empty


# ---------------------------------------------------------------------------
# Test helper
# ---------------------------------------------------------------------------

def _make_sample_data(tmp_path):
    abstracts = pd.DataFrame({
        "uid": ["D0", "D1"],
        "title": ["Test A", "Test B"],
        "abstract": ["Abstract one.", "Abstract two."],
        "pubyear": [2020, 2021],
    })
    membership = pd.DataFrame({
        "uid": ["D0", "D1"],
        "cluster": [0, 0],
    })
    abstract_path = tmp_path / "abstracts.parquet"
    membership_path = tmp_path / "membership.parquet"
    abstracts.to_parquet(abstract_path, index=False)
    membership.to_parquet(membership_path, index=False)
    return abstract_path, membership_path
