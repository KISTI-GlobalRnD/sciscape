"""Tests for domain-agnostic keyword quality refinement."""

import json

import pandas as pd
import pytest

from sciscape.keyword_extraction import (
    KeywordExtractionConfig,
    annotate_keyword_quality,
    build_abbreviation_lookup,
    extract_parenthetical_abbreviations,
    quality_flag_counts,
    run_keyword_pipeline,
)


def test_quality_annotation_demotes_global_terms_and_prefers_phrases():
    df = pd.DataFrame(
        {
            "cluster_id": [0, 0, 0, 0, 1, 1, 1, 2, 2, 2],
            "term": [
                "graph",
                "graph neural network",
                "traffic flow prediction",
                "abstract",
                "graph",
                "drug drug interaction",
                "ddi",
                "graph",
                "self assembled monolayer",
                "sam",
            ],
            "score": [10.0, 8.0, 7.5, 7.0, 10.0, 8.0, 6.0, 10.0, 8.0, 6.0],
            "frequency": [30, 10, 8, 20, 30, 10, 8, 30, 10, 8],
            "doc_coverage": [25, 8, 7, 18, 25, 8, 7, 25, 8, 7],
        }
    )

    result = annotate_keyword_quality(df, rerank=True)

    c0_terms = result[result["cluster_id"] == 0]["term"].tolist()
    assert c0_terms.index("traffic flow prediction") < c0_terms.index("graph")
    assert c0_terms.index("graph neural network") < c0_terms.index("graph")

    graph_rows = result[result["term"] == "graph"]
    assert graph_rows["quality_flags"].str.contains("too_global").all()
    assert graph_rows["quality_flags"].str.contains("phrase_preferred").any()

    abstract = result[result["term"] == "abstract"].iloc[0]
    assert "artifact_like" in abstract["quality_flags"]
    assert abstract["quality_score"] < abstract["score"]

    assert "network_role" in result.columns
    assert "quality_decision_trace" in result.columns
    assert result[result["term"] == "graph"].iloc[0]["network_role"] == "generic_bridge"
    assert graph_rows["keyword_scope"].eq("common").all()
    traffic = result[result["term"] == "traffic flow prediction"].iloc[0]
    assert traffic["keyword_scope"] == "cluster_specific"
    trace = json.loads(traffic["quality_decision_trace"])
    assert trace["term"] == "traffic flow prediction"
    assert trace["display_label"] == "traffic flow prediction"
    assert trace["quality_score"] == pytest.approx(traffic["quality_score"])
    assert trace["representative_role"] == traffic["representative_role"]
    assert any(step["name"] == "phrase_specificity" for step in trace["quality_adjustments"])


def test_quality_annotation_expands_acronym_display_labels():
    df = pd.DataFrame(
        {
            "cluster_id": [0, 0, 1, 1, 2, 2],
            "term": [
                "drug drug interaction",
                "ddi",
                "self assembled monolayer",
                "sam",
                "single cell rna sequencing",
                "scrna",
            ],
            "score": [8.0, 6.0, 8.0, 6.0, 8.0, 6.0],
            "frequency": [10, 5, 10, 5, 10, 5],
        }
    )

    result = annotate_keyword_quality(df, rerank=True)
    labels = dict(zip(result["term"], result["display_label"]))

    assert labels["ddi"] == "drug drug interaction"
    assert labels["sam"] == "self assembled monolayer"
    assert labels["scrna"] == "single cell rna sequencing"
    statuses = dict(zip(result["term"], result["abbreviation_status"]))
    targets = dict(zip(result["term"], result["abbreviation_target"]))
    assert statuses["ddi"] == "duplicate_expansion"
    assert targets["ddi"] == "drug drug interaction"
    assert "acronym_like" in result[result["term"] == "ddi"].iloc[0]["quality_flags"]


def test_quality_annotation_marks_corpus_expansion_duplicate_when_phrase_exists():
    evidence = pd.DataFrame(
        {
            "short_form": ["sam"],
            "long_form": ["self assembled monolayers"],
            "support_docs": [4],
            "support_occurrences": [4],
            "cluster_supports": [{0: 4}],
            "pattern_types": ["long_before_short_in_parens"],
            "raw_long_forms": ["self assembled monolayers"],
            "candidate_rank": [1],
            "short_form_candidate_count": [1],
            "top_support_docs": [4],
            "top_support_ratio": [1.0],
            "is_ambiguous": [False],
            "confidence": [0.95],
        }
    )
    lookup = build_abbreviation_lookup(evidence, min_support_docs=2, min_cluster_support_docs=2)
    df = pd.DataFrame(
        {
            "cluster_id": [0, 0],
            "term": ["self assembled monolayer", "sam"],
            "score": [8.0, 20.0],
            "frequency": [10, 5],
        }
    )

    result = annotate_keyword_quality(df, rerank=True, abbreviation_lookup=lookup)
    sam = result[result["term"] == "sam"].iloc[0]
    phrase = result[result["term"] == "self assembled monolayer"].iloc[0]

    assert sam["display_label"] == "self assembled monolayer"
    assert sam["abbreviation_status"] == "duplicate_expansion"
    assert "duplicate_expansion" in sam["quality_flags"]
    assert sam["quality_score"] < phrase["quality_score"]


def test_parenthetical_abbreviation_evidence_extracts_and_supports_pairs():
    docs = pd.DataFrame(
        {
            "uid": ["D1", "D2", "D3"],
            "cluster_id": [0, 0, 1],
            "title": [
                "Graph neural networks (GNN) for traffic",
                "Graph neural network (GNN) models",
                "ML (machine learning) baseline",
            ],
            "abstract": [
                "The graph neural networks (GNNs) improve forecasting.",
                "We compare graph neural network (GNN) variants.",
                "Machine learning (ML) methods are included.",
            ],
        }
    )

    evidence = extract_parenthetical_abbreviations(docs)
    lookup = build_abbreviation_lookup(evidence, min_support_docs=2)

    gnn = evidence[evidence["short_form"] == "gnn"].iloc[0]
    assert gnn["long_form"] == "graph neural networks"
    assert gnn["support_docs"] == 2
    assert gnn["cluster_supports"] == {0: 2}
    assert gnn["ambiguity_type"] == "none"
    assert lookup["cluster"][(0, "gnn")]["long_form"] == "graph neural networks"


def test_quality_annotation_uses_corpus_abbreviation_lookup():
    lookup = {
        "global": {
            "gnn": {
                "long_form": "graph neural network",
                "support_docs": 4,
                "cluster_support_docs": 0,
                "confidence": 0.92,
                "top_support_ratio": 1.0,
                "status": "corpus_expanded",
                "usable": True,
            }
        },
        "cluster": {},
    }
    df = pd.DataFrame(
        {
            "cluster_id": [0],
            "term": ["gnn"],
            "score": [3.0],
            "frequency": [10],
        }
    )

    result = annotate_keyword_quality(df, rerank=True, abbreviation_lookup=lookup)
    row = result.iloc[0]

    assert row["display_label"] == "graph neural network"
    assert row["abbreviation_status"] == "corpus_expanded"
    assert row["abbreviation_source"] == "corpus_parenthetical"
    assert row["abbreviation_support_docs"] == 4


def test_quality_annotation_keeps_ambiguous_abbreviation_as_raw_label():
    lookup = {
        "global": {
            "abc": {
                "long_form": "alpha beta count",
                "support_docs": 2,
                "cluster_support_docs": 0,
                "confidence": 0.6,
                "top_support_ratio": 0.5,
                "status": "ambiguous_expansion",
                "usable": False,
            }
        },
        "cluster": {},
    }
    df = pd.DataFrame(
        {
            "cluster_id": [0],
            "term": ["abc"],
            "score": [3.0],
            "frequency": [10],
        }
    )

    result = annotate_keyword_quality(df, rerank=True, abbreviation_lookup=lookup)
    row = result.iloc[0]

    assert row["display_label"] == "abc"
    assert row["abbreviation_status"] == "ambiguous_expansion"
    assert "ambiguous_expansion" in row["quality_flags"]


def test_quality_annotation_marks_low_support_abbreviation_without_expanding():
    lookup = {
        "global": {
            "abc": {
                "long_form": "alpha beta count",
                "support_docs": 1,
                "cluster_support_docs": 0,
                "confidence": 0.55,
                "top_support_ratio": 1.0,
                "status": "low_support_expansion",
                "usable": False,
                "ambiguity_type": "none",
            }
        },
        "cluster": {},
    }
    df = pd.DataFrame(
        {
            "cluster_id": [0],
            "term": ["abc"],
            "score": [3.0],
            "frequency": [10],
        }
    )

    result = annotate_keyword_quality(df, rerank=True, abbreviation_lookup=lookup)
    row = result.iloc[0]

    assert row["display_label"] == "abc"
    assert row["abbreviation_status"] == "low_support_expansion"
    assert row["abbreviation_target"] == "alpha beta count"
    assert row["abbreviation_ambiguity_type"] == "none"
    assert "low_support_expansion" in row["quality_flags"]


def test_quality_annotation_expands_acronyms_independent_of_row_order():
    df = pd.DataFrame(
        {
            "cluster_id": [0, 0, 1, 1],
            "term": [
                "ddi",
                "drug drug interaction",
                "sam",
                "self assembled monolayer",
            ],
            "score": [6.0, 8.0, 6.0, 8.0],
            "frequency": [5, 10, 5, 10],
        }
    )

    result = annotate_keyword_quality(df, rerank=True)
    labels = dict(zip(result["term"], result["display_label"]))

    assert labels["ddi"] == "drug drug interaction"
    assert labels["sam"] == "self assembled monolayer"


def test_network_roles_preserve_cluster_specific_anchor_unigrams():
    df = pd.DataFrame(
        {
            "cluster_id": [0, 0, 0, 0, 1],
            "term": [
                "catalyst",
                "catalyst stability",
                "catalyst surface area",
                "model",
                "model",
            ],
            "score": [10.0, 8.0, 7.0, 11.0, 11.0],
            "frequency": [50, 25, 20, 80, 80],
            "doc_coverage": [40, 20, 16, 70, 70],
        }
    )

    result = annotate_keyword_quality(df, rerank=True)
    catalyst = result[
        (result["cluster_id"] == 0) & (result["term"] == "catalyst")
    ].iloc[0]
    model = result[
        (result["cluster_id"] == 0) & (result["term"] == "model")
    ].iloc[0]

    assert catalyst["network_role"] == "anchor_unigram"
    assert "anchor_unigram" in catalyst["quality_flags"]
    assert catalyst["quality_multiplier"] > 0.7
    assert catalyst["quality_score"] > model["quality_score"]


def test_representative_score_prefers_supported_phrases_over_shadowed_unigrams():
    df = pd.DataFrame(
        {
            "cluster_id": [0, 0, 0, 0],
            "term": [
                "session",
                "session based recommendation",
                "item",
                "user item recommendation",
            ],
            "score": [10.0, 5.0, 8.0, 4.5],
            "frequency": [80, 22, 64, 20],
            "doc_coverage": [30, 10, 25, 9],
        }
    )

    result = annotate_keyword_quality(df, rerank=True)
    ranks = dict(zip(result["term"], result["representative_rank"]))
    roles = dict(zip(result["term"], result["representative_role"]))

    assert ranks["session based recommendation"] < ranks["session"]
    assert ranks["user item recommendation"] < ranks["item"]
    assert roles["session"] in {"linked_unigram", "shadowed_unigram"}
    assert roles["session based recommendation"] == "representative_phrase"


def test_representative_score_demotes_unresolved_short_forms_for_labels():
    df = pd.DataFrame(
        {
            "cluster_id": [0, 0],
            "term": ["eeg", "emotion recognition"],
            "score": [10.0, 7.0],
            "frequency": [50, 25],
            "doc_coverage": [8, 7],
        }
    )

    result = annotate_keyword_quality(df, rerank=True)
    eeg = result[result["term"] == "eeg"].iloc[0]
    phrase = result[result["term"] == "emotion recognition"].iloc[0]

    assert eeg["abbreviation_status"] in {"candidate_short_form", "unlinked_short_form"}
    assert eeg["representative_role"] == "review_short_form"
    assert phrase["representative_rank"] < eeg["representative_rank"]
    assert phrase["representative_score"] > eeg["representative_score"]


def test_representative_score_demotes_shared_unigram_labels():
    df = pd.DataFrame(
        {
            "cluster_id": [0, 0, 1, 2],
            "term": ["session", "sequential recommendation", "session", "traffic flow"],
            "score": [10.0, 5.0, 10.0, 4.0],
            "frequency": [40, 20, 40, 12],
            "doc_coverage": [4, 8, 4, 6],
        }
    )

    result = annotate_keyword_quality(df, rerank=True, global_term_threshold=0.8)
    session = result[
        (result["cluster_id"] == 0) & (result["term"] == "session")
    ].iloc[0]
    phrase = result[
        (result["cluster_id"] == 0) & (result["term"] == "sequential recommendation")
    ].iloc[0]

    assert session["representative_role"] == "shared_unigram"
    assert phrase["representative_score"] > session["representative_score"]


def test_representative_score_uses_family_support_for_parent_terms():
    df = pd.DataFrame(
        {
            "cluster_id": [0, 0, 0, 0],
            "term": [
                "traffic flow",
                "traffic flow prediction",
                "traffic flow speed",
                "time series",
            ],
            "score": [8.0, 7.0, 6.0, 5.0],
            "frequency": [30, 20, 18, 14],
            "doc_coverage": [15, 10, 9, 8],
        }
    )

    result = annotate_keyword_quality(df, rerank=True)
    traffic = result[result["term"] == "traffic flow"].iloc[0]
    prediction = result[result["term"] == "traffic flow prediction"].iloc[0]
    trace = json.loads(traffic["quality_decision_trace"])

    assert traffic["representative_family_child_count"] == 2
    assert traffic["representative_family_member_count"] == 3
    assert traffic["representative_family_avg_child_coverage"] == pytest.approx(9.5)
    assert traffic["representative_family_multiplier"] > 1.0
    assert traffic["representative_score"] == pytest.approx(
        traffic["quality_score"]
        * traffic["representative_multiplier"]
    )
    assert prediction["representative_family_child_count"] == 0
    assert prediction["representative_family_member_count"] == 1
    assert trace["representative_family_support"]["child_count"] == 2
    assert any(
        step["name"] == "representative_family_support"
        for step in trace["quality_adjustments"]
    )


def test_representative_score_uses_hidden_candidate_family_support():
    df = pd.DataFrame(
        {
            "cluster_id": [0, 0],
            "term": ["traffic flow", "time series"],
            "score": [8.0, 5.0],
            "frequency": [30, 14],
            "doc_coverage": [15, 8],
            "candidates": [
                ["traffic flow prediction", "urban traffic flow"],
                [],
            ],
        }
    )

    result = annotate_keyword_quality(df, rerank=True)
    traffic = result[result["term"] == "traffic flow"].iloc[0]
    trace = json.loads(traffic["quality_decision_trace"])

    assert traffic["representative_family_child_count"] == 2
    assert traffic["representative_family_member_count"] == 3
    assert traffic["representative_family_multiplier"] > 1.0
    assert {child["term"] for child in trace["representative_family_support"]["children"]} == {
        "traffic flow prediction",
        "urban traffic flow",
    }
    assert all(
        child["evidence_source"] == "candidate_terms"
        for child in trace["representative_family_support"]["children"]
    )


def test_quality_annotation_does_not_flag_common_words_as_acronyms():
    df = pd.DataFrame(
        {
            "cluster_id": [0, 0, 0, 0],
            "term": ["film", "tin", "wind", "ion"],
            "score": [3.0, 2.0, 1.0, 1.0],
            "frequency": [10, 8, 6, 5],
        }
    )

    result = annotate_keyword_quality(df, rerank=True)

    assert not result["quality_flags"].str.contains("acronym_like").any()
    assert result["abbreviation_status"].eq("not_abbreviation").all()


def test_quality_annotation_flags_vowel_tolerant_scientific_short_forms():
    df = pd.DataFrame(
        {
            "cluster_id": [0, 0, 0, 0, 0],
            "term": ["hsi", "dtis", "scrna", "seq", "zno"],
            "score": [3.0, 2.5, 2.0, 1.8, 1.5],
            "frequency": [10, 9, 8, 7, 6],
        }
    )

    result = annotate_keyword_quality(df, rerank=True)
    statuses = dict(zip(result["term"], result["abbreviation_status"]))

    assert statuses["hsi"] in {"candidate_short_form", "unlinked_short_form"}
    assert statuses["dtis"] in {"candidate_short_form", "unlinked_short_form"}
    assert statuses["scrna"] in {"candidate_short_form", "unlinked_short_form"}
    assert statuses["seq"] in {"candidate_short_form", "unlinked_short_form"}
    assert statuses["zno"] == "not_abbreviation"


def test_quality_annotation_marks_unexpanded_short_forms_as_review_candidates():
    df = pd.DataFrame(
        {
            "cluster_id": [0, 0],
            "term": ["eeg", "brain signal"],
            "score": [3.0, 2.0],
            "frequency": [10, 8],
        }
    )

    result = annotate_keyword_quality(df, rerank=True)
    eeg = result[result["term"] == "eeg"].iloc[0]

    assert eeg["abbreviation_status"] in {"candidate_short_form", "unlinked_short_form"}
    assert any(flag in eeg["quality_flags"] for flag in ("candidate_short_form", "unlinked_short_form"))


def test_quality_annotation_can_disable_network_roles():
    df = pd.DataFrame(
        {
            "cluster_id": [0, 0],
            "term": ["graph", "graph neural network"],
            "score": [3.0, 2.0],
            "frequency": [10, 5],
        }
    )

    result = annotate_keyword_quality(df, rerank=True, network_roles_enabled=False)

    assert "network_role" not in result.columns


def test_quality_annotation_keeps_dimension_notation_from_formula_flag():
    df = pd.DataFrame(
        {
            "cluster_id": [0, 0, 0, 0, 0, 0],
            "term": ["quasi 2d", "cspbi3", "zno", "abc123", "oer", "ray"],
            "score": [3.0, 2.0, 1.5, 1.0, 0.9, 0.8],
            "frequency": [10, 8, 7, 6, 5, 4],
        }
    )

    result = annotate_keyword_quality(df, rerank=True)
    quasi = result[result["term"] == "quasi 2d"].iloc[0]
    cspbi = result[result["term"] == "cspbi3"].iloc[0]
    zno = result[result["term"] == "zno"].iloc[0]
    artifact = result[result["term"] == "abc123"].iloc[0]
    oer = result[result["term"] == "oer"].iloc[0]
    ray = result[result["term"] == "ray"].iloc[0]

    assert "formula_like" not in quasi["quality_flags"]
    assert "formula_like" in cspbi["quality_flags"]
    assert "material_formula" in cspbi["quality_flags"]
    assert "material_formula" in zno["quality_flags"]
    assert "artifact_formula" in artifact["quality_flags"]
    assert "material_formula" not in oer["quality_flags"]
    assert "material_formula" not in ray["quality_flags"]
    assert zno["quality_multiplier"] > artifact["quality_multiplier"]


def test_quality_flag_counts_counts_pipe_delimited_flags():
    df = pd.DataFrame({"quality_flags": ["too_global|phrase_preferred", "phrase", ""]})
    assert quality_flag_counts(df) == {
        "too_global": 1,
        "phrase_preferred": 1,
        "phrase": 1,
    }


def test_dashboard_data_uses_display_labels_and_preserves_raw_terms():
    from sciscape.keyword_extraction.visualization._data_prep import prepare_cluster_data

    df = pd.DataFrame(
        {
            "cluster_id": [0, 0],
            "term": ["ddi", "drug drug interaction"],
            "display_label": ["drug drug interaction", "drug drug interaction"],
            "quality_score": [1.0, 2.0],
            "score": [5.0, 1.0],
            "quality_flags": ["acronym_like", "phrase"],
            "network_role": ["alias_acronym", "expansion_phrase"],
            "network_score": [0.5, 0.9],
            "network_flags": ["duplicate_label", "expansion_phrase"],
            "keyword_scope": ["cluster_specific", "cluster_specific"],
            "keyword_cluster_count": [1, 1],
            "keyword_cluster_ratio": [1.0, 1.0],
            "abbreviation_status": ["duplicate_expansion", "not_abbreviation"],
            "abbreviation_target": ["drug drug interaction", ""],
            "abbreviation_confidence": [1.0, 0.0],
            "frequency": [3, 4],
            "doc_coverage": [2, 3],
        }
    )

    data = prepare_cluster_data(df)
    cluster = data[0]

    assert cluster["label"] == "drug drug interaction"
    assert cluster["keywords"][0]["term"] == "drug drug interaction"
    assert cluster["keywords"][0]["raw_term"] == "drug drug interaction"
    assert cluster["keywords"][0]["score"] == pytest.approx(2.0)
    assert cluster["keywords"][0]["raw_score"] == pytest.approx(1.0)
    assert cluster["keywords"][0]["network_role"] == "expansion_phrase"
    assert cluster["keywords"][0]["raw_aliases"][0]["raw_term"] == "ddi"
    assert cluster["keywords"][0]["raw_aliases"][0]["abbreviation_status"] == "duplicate_expansion"
    assert cluster["keywords"][0]["raw_aliases"][0]["abbreviation_target"] == "drug drug interaction"
    assert cluster["keywords"][0]["member_count"] == 2
    assert cluster["keywords"][0]["child_count"] == 0
    assert cluster["keywords"][0]["raw_alias_count"] == 1
    assert cluster["keyword_families"][0]["member_count"] == 2
    assert cluster["keyword_families"][0]["raw_alias_count"] == 1
    assert cluster["keyword_groups"]["cluster_specific"][0]["term"] == "drug drug interaction"


def test_dashboard_data_groups_common_and_cluster_specific_keywords():
    from sciscape.keyword_extraction.visualization._data_prep import prepare_cluster_data

    df = pd.DataFrame(
        {
            "cluster_id": [0, 0, 1, 2],
            "term": ["graph", "traffic flow prediction", "graph", "graph"],
            "display_label": ["graph", "traffic flow prediction", "graph", "graph"],
            "quality_score": [1.0, 3.0, 1.2, 1.1],
            "score": [1.0, 3.0, 1.2, 1.1],
            "quality_flags": ["too_global", "phrase|cluster_specific", "too_global", "too_global"],
            "keyword_scope": ["common", "cluster_specific", "common", "common"],
            "keyword_cluster_count": [3, 1, 3, 3],
            "keyword_cluster_ratio": [1.0, 1 / 3, 1.0, 1.0],
            "abbreviation_status": ["not_abbreviation"] * 4,
            "abbreviation_target": [""] * 4,
            "abbreviation_confidence": [0.0] * 4,
            "frequency": [30, 8, 25, 20],
            "doc_coverage": [25, 7, 22, 18],
        }
    )

    data = prepare_cluster_data(df)

    assert data[0]["keyword_groups"]["cluster_specific"][0]["term"] == "traffic flow prediction"
    assert data[0]["keyword_groups"]["common"][0]["term"] == "graph"
    assert data["_common_keywords"][0]["term"] == "graph"
    assert data["_common_keywords"][0]["cluster_count"] == 3


def test_dashboard_data_uses_representative_score_for_cluster_labels():
    from sciscape.keyword_extraction.visualization._data_prep import prepare_cluster_data

    df = pd.DataFrame(
        {
            "cluster_id": [0, 0],
            "term": ["session", "session based recommendation"],
            "display_label": ["session", "session based recommendation"],
            "quality_score": [10.0, 5.0],
            "representative_score": [1.0, 7.0],
            "score": [10.0, 5.0],
            "quality_flags": ["phrase_preferred", "phrase"],
            "representative_role": ["shadowed_unigram", "representative_phrase"],
            "representative_flags": ["shadowed_unigram", "phrase_label"],
            "keyword_scope": ["cluster_specific", "cluster_specific"],
            "keyword_cluster_count": [1, 1],
            "keyword_cluster_ratio": [1.0, 1.0],
            "abbreviation_status": ["not_abbreviation", "not_abbreviation"],
            "abbreviation_target": ["", ""],
            "abbreviation_confidence": [0.0, 0.0],
            "frequency": [40, 12],
            "doc_coverage": [20, 8],
        }
    )

    data = prepare_cluster_data(df)

    assert data[0]["label"].startswith("session based recommendation")
    assert data[0]["keywords"][0]["term"] == "session based recommendation"
    assert data[0]["keywords"][0]["representative_role"] == "representative_phrase"
    assert data[0]["keywords"][0]["quality_score"] == pytest.approx(5.0)


def test_dashboard_cluster_labels_use_mmr_for_redundant_representative_terms():
    from sciscape.keyword_extraction.visualization._data_prep import prepare_cluster_data

    df = pd.DataFrame(
        {
            "cluster_id": [0, 0, 0, 0],
            "term": [
                "traffic flow",
                "traffic flow prediction",
                "traffic forecasting",
                "time series",
            ],
            "display_label": [
                "traffic flow",
                "traffic flow prediction",
                "traffic forecasting",
                "time series",
            ],
            "representative_score": [10.0, 8.0, 7.2, 6.8],
            "quality_score": [10.0, 8.0, 7.2, 6.8],
            "score": [10.0, 8.0, 7.2, 6.8],
            "quality_flags": ["phrase"] * 4,
            "keyword_scope": ["cluster_specific"] * 4,
            "keyword_cluster_count": [1] * 4,
            "keyword_cluster_ratio": [1.0] * 4,
            "abbreviation_status": ["not_abbreviation"] * 4,
            "abbreviation_target": [""] * 4,
            "abbreviation_confidence": [0.0] * 4,
            "frequency": [30, 24, 21, 20],
            "doc_coverage": [15, 12, 10, 10],
        }
    )

    data = prepare_cluster_data(df)
    label_terms = [term.strip() for term in data[0]["label"].split(",")]

    assert label_terms[0] == "traffic flow"
    assert "time series" in label_terms
    assert "traffic flow prediction" not in label_terms


def test_dashboard_data_builds_keyword_families_with_counts():
    from sciscape.keyword_extraction.visualization._data_prep import prepare_cluster_data

    df = pd.DataFrame(
        {
            "cluster_id": [0, 0, 0, 0],
            "term": [
                "traffic flow",
                "traffic flow prediction",
                "traffic flow speed",
                "time series",
            ],
            "display_label": [
                "traffic flow",
                "traffic flow prediction",
                "traffic flow speed",
                "time series",
            ],
            "representative_score": [10.0, 8.0, 7.0, 6.0],
            "quality_score": [10.0, 8.0, 7.0, 6.0],
            "score": [10.0, 8.0, 7.0, 6.0],
            "quality_flags": ["phrase"] * 4,
            "keyword_scope": ["cluster_specific"] * 4,
            "keyword_cluster_count": [1] * 4,
            "keyword_cluster_ratio": [1.0] * 4,
            "abbreviation_status": ["not_abbreviation"] * 4,
            "abbreviation_target": [""] * 4,
            "abbreviation_confidence": [0.0] * 4,
            "frequency": [30, 24, 18, 12],
            "doc_coverage": [15, 12, 9, 6],
        }
    )

    data = prepare_cluster_data(df)
    cluster = data[0]
    families = {family["term"]: family for family in cluster["keyword_families"]}
    traffic_family = families["traffic flow"]

    assert cluster["keyword_family_count"] == 2
    assert traffic_family["child_count"] == 2
    assert traffic_family["raw_alias_count"] == 0
    assert traffic_family["member_count"] == 3
    assert {child["term"] for child in traffic_family["children"]} == {
        "traffic flow prediction",
        "traffic flow speed",
    }
    assert all("member_count" in family for family in cluster["keyword_families"])
    assert all("child_count" in family for family in cluster["keyword_families"])
    assert all("raw_alias_count" in family for family in cluster["keyword_families"])
    assert cluster["keyword_groups"]["cluster_specific"][0]["member_count"] == 3


@pytest.fixture
def quality_pipeline_data(tmp_path):
    abstracts = pd.DataFrame(
        {
            "uid": [f"D{i}" for i in range(6)],
            "title": [
                "Graph neural network traffic prediction",
                "Traffic flow prediction with graph models",
                "Graph neural network drug interaction",
                "Drug drug interaction prediction",
                "Self assembled monolayer perovskite solar cell",
                "Perovskite solar cell transport layer",
            ],
            "abstract": [
                "Graph models support traffic flow prediction and road network forecasting.",
                "Traffic flow prediction uses graph neural network architectures.",
                "Graph neural network methods identify drug drug interaction signals.",
                "Drug drug interaction prediction estimates binding affinity.",
                "Self assembled monolayer improves perovskite solar cell interfaces.",
                "Perovskite solar cell electron transport layer controls efficiency.",
            ],
            "pubyear": [2020, 2021, 2020, 2021, 2020, 2021],
        }
    )
    membership = pd.DataFrame(
        {
            "uid": [f"D{i}" for i in range(6)],
            "cluster": [0, 0, 1, 1, 2, 2],
        }
    )
    abstract_path = tmp_path / "abstracts.parquet"
    membership_path = tmp_path / "membership.parquet"
    abstracts.to_parquet(abstract_path, index=False)
    membership.to_parquet(membership_path, index=False)
    return abstract_path, membership_path


def test_pipeline_emits_quality_columns_when_enabled(quality_pipeline_data):
    cfg = KeywordExtractionConfig(
        abstract_path=quality_pipeline_data[0],
        membership_path=quality_pipeline_data[1],
        cluster_level="cluster",
        include_title=True,
        title_weight=1.0,
        min_df_unigram=1,
        min_df_phrase=1,
        phrase_min_count_per_cluster=1,
        top_n_keywords=5,
        scoring_pool_factor=2.0,
        ngram_min=1,
        ngram_max=3,
        use_phrase_vectorizer=True,
        cross_cluster_penalty_enabled=True,
        quality_diagnostics_enabled=True,
        quality_rerank_enabled=True,
        n_jobs=1,
    )

    keywords = run_keyword_pipeline(cfg)

    assert not keywords.empty
    for column in (
        "raw_term",
        "normalized_term",
        "display_label",
        "quality_score",
        "quality_flags",
        "quality_decision_trace",
        "keyword_scope",
        "keyword_cluster_count",
        "keyword_cluster_ratio",
        "abbreviation_status",
        "abbreviation_target",
        "abbreviation_confidence",
        "abbreviation_source",
        "abbreviation_support_docs",
        "abbreviation_cluster_support_docs",
        "abbreviation_top_support_ratio",
        "abbreviation_ambiguity_type",
        "network_role",
        "network_score",
        "network_flags",
        "representative_score",
        "representative_multiplier",
        "representative_rank",
        "representative_role",
        "representative_flags",
        "representative_family_child_count",
        "representative_family_member_count",
        "representative_family_avg_child_coverage",
        "representative_family_multiplier",
    ):
        assert column in keywords.columns
    assert keywords["quality_score"].notna().all()


def test_pipeline_uses_parenthetical_abbreviation_dictionary(tmp_path):
    abstracts = pd.DataFrame(
        {
            "uid": [f"D{i}" for i in range(4)],
            "title": [
                "Graph neural networks (GNN) for traffic",
                "Graph neural network (GNN) forecasting",
                "Traffic prediction with GNN",
                "Graph neural networks in roads",
            ],
            "abstract": [
                "Graph neural networks (GNNs) model road sensors.",
                "A graph neural network (GNN) improves traffic forecasting.",
                "GNN methods use graph structure for traffic prediction.",
                "Graph neural networks learn spatial relations.",
            ],
            "pubyear": [2020, 2021, 2022, 2023],
        }
    )
    membership = pd.DataFrame(
        {
            "uid": [f"D{i}" for i in range(4)],
            "cluster": [0, 0, 0, 0],
        }
    )
    abstract_path = tmp_path / "abstracts.parquet"
    membership_path = tmp_path / "membership.parquet"
    abstracts.to_parquet(abstract_path, index=False)
    membership.to_parquet(membership_path, index=False)
    cfg = KeywordExtractionConfig(
        abstract_path=abstract_path,
        membership_path=membership_path,
        cluster_level="cluster",
        include_title=True,
        title_weight=1.0,
        min_df_unigram=1,
        min_df_phrase=1,
        phrase_min_count_per_cluster=1,
        top_n_keywords=20,
        scoring_pool_factor=1.0,
        ngram_min=1,
        ngram_max=3,
        use_phrase_vectorizer=True,
        quality_diagnostics_enabled=True,
        quality_rerank_enabled=True,
        academic_stopwords_enabled=False,
        artifact_filter_enabled=True,
        n_jobs=1,
    )

    keywords = run_keyword_pipeline(cfg)
    gnn = keywords[keywords["term"].eq("gnn")]

    assert not gnn.empty
    assert gnn.iloc[0]["display_label"] == "graph neural networks"
    assert gnn.iloc[0]["abbreviation_status"] in {"cluster_expanded", "corpus_expanded"}
    assert gnn.iloc[0]["abbreviation_support_docs"] >= 2


def test_legacy_abbreviation_dict_uses_parenthetical_evidence_and_case_insensitive_expansion():
    import polars as pl

    from sciscape.clustering.abbreviation_dict import (
        expand_labels_with_abbreviations,
        extract_abbreviations,
    )

    docs = pl.DataFrame(
        {
            "uid": ["D1", "D2", "D3"],
            "title": [
                "Graph neural networks (GNN) for traffic",
                "Graph neural network (GNN) forecasting",
                "Noise paper",
            ],
            "abstract": [
                "Graph neural networks (GNNs) model road sensors.",
                "A graph neural network (GNN) improves traffic forecasting.",
                "This record has no abbreviation evidence.",
            ],
        }
    )

    abbr = extract_abbreviations(docs, min_count=2)
    assert abbr["gnn"] == "graph neural networks"

    labels = expand_labels_with_abbreviations(["GNN traffic", "gnn forecasting"], abbr)
    assert labels == [
        "GNN (graph neural networks) traffic",
        "gnn (graph neural networks) forecasting",
    ]
