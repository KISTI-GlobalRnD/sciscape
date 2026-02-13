from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from keyword_extraction.keyword_extraction import KeywordExtractionConfig, KeywordExtractionPipeline


def _make_pipeline(**overrides) -> KeywordExtractionPipeline:
    cfg = KeywordExtractionConfig(
        abstract_path=Path("/tmp/abstract.parquet"),
        membership_path=Path("/tmp/membership.parquet"),
        apply_alias_map=True,
        alias_strategy="llm_candidates",
        **overrides,
    )
    pipe = KeywordExtractionPipeline.__new__(KeywordExtractionPipeline)
    pipe.config = cfg
    return pipe


def _subset_row(term: str, candidates) -> dict:
    return {
        "term": term,
        "score": 1.0,
        "frequency": 10,
        "doc_coverage": 5,
        "candidates": candidates,
    }


def test_llm_candidates_snaps_to_exact_candidate() -> None:
    pipe = _make_pipeline(alias_candidate_column="candidates", alias_candidate_enforce=True)
    subset = pd.DataFrame([_subset_row("IL6", ["IL-6"])])
    raw = json.dumps({"items": [{"term": "IL6", "action": "merge_into", "canonical": "IL-6"}]})
    parsed = pipe._parse_alias_items(cluster_id=1, raw_response=raw, subset=subset)
    inst = next(item for item in parsed if item["original"] == "IL6")
    assert inst["action"] == "merge_into"
    assert inst["canonical"] == "IL-6"


def test_llm_candidates_greek_symbol_matches_name() -> None:
    pipe = _make_pipeline(alias_candidate_column="candidates", alias_candidate_enforce=True)
    subset = pd.DataFrame([_subset_row("beta amyloid", ["beta amyloid"])])
    raw = json.dumps(
        {"items": [{"term": "beta amyloid", "action": "merge_into", "canonical": "\u03b2-amyloid"}]}
    )
    parsed = pipe._parse_alias_items(cluster_id=1, raw_response=raw, subset=subset)
    inst = next(item for item in parsed if item["original"] == "beta amyloid")
    assert inst["action"] == "merge_into"
    assert inst["canonical"] == "beta amyloid"


def test_llm_candidates_rejects_canonical_not_in_candidates() -> None:
    pipe = _make_pipeline(alias_candidate_column="candidates", alias_candidate_enforce=True)
    subset = pd.DataFrame([_subset_row("hiv", ["hiv"])])
    raw = json.dumps({"items": [{"term": "hiv", "action": "merge_into", "canonical": "hiv 1"}]})
    parsed = pipe._parse_alias_items(cluster_id=1, raw_response=raw, subset=subset)
    inst = next(item for item in parsed if item["original"] == "hiv")
    assert inst["action"] == "keep"
    assert inst["canonical"] == "hiv"
    assert "canonical_not_in_candidates" in str(inst.get("reason", ""))


def test_llm_candidates_rejects_specifier_mismatch() -> None:
    pipe = _make_pipeline(alias_candidate_column="candidates", alias_candidate_enforce=True)
    subset = pd.DataFrame([_subset_row("hiv", ["hiv 1"])])
    raw = json.dumps({"items": [{"term": "hiv", "action": "merge_into", "canonical": "hiv 1"}]})
    parsed = pipe._parse_alias_items(cluster_id=1, raw_response=raw, subset=subset)
    inst = next(item for item in parsed if item["original"] == "hiv")
    assert inst["action"] == "keep"
    assert inst["canonical"] == "hiv"
    assert "specifier_mismatch" in str(inst.get("reason", ""))


def test_load_only_rebuild_uses_cached_candidates_even_with_custom_column(tmp_path: Path) -> None:
    pipe = _make_pipeline(alias_candidate_column="my_candidates", alias_candidate_enforce=True)
    pipe._alias_cache_dir = tmp_path

    cluster_dir = tmp_path / "1"
    cluster_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cluster_dir / "test.json"
    raw = json.dumps({"items": [{"term": "hiv", "action": "merge_into", "canonical": "hiv 1"}]})
    payload = {
        "terms": [
            {
                "term": "hiv",
                "score": 1.0,
                "frequency": 10,
                "doc_coverage": 5,
                "candidates": ["hiv 1"],
            }
        ]
    }
    cache_file.write_text(json.dumps({"raw_response": raw, "payload": payload}), encoding="utf-8")

    top_df = pd.DataFrame([{"cluster_id": 1, "term": "hiv", "score": 1.0, "frequency": 10, "doc_coverage": 5}])
    pipe._rebuild_alias_mappings_from_cache(top_df)

    mapping_path = tmp_path / "mapping" / "1.json"
    saved = json.loads(mapping_path.read_text(encoding="utf-8"))
    item = next(it for it in saved["items"] if it["original"] == "hiv")
    assert item["action"] == "keep"
    assert item["canonical"] == "hiv"
    assert "specifier_mismatch" in str(item.get("reason", ""))

