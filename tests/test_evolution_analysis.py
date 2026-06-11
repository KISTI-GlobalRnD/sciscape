"""Tests for in-memory cluster evolution analysis."""

from __future__ import annotations

import pandas as pd
import pytest

import sciscape
from sciscape.evolution import (
    build_evolution_state_table,
    build_evolution_transition_table,
    build_membership_projection_evolution,
    classify_evolution_events,
    label_evolution_transition_relations,
    rank_evolution_transitions,
)


def test_evolution_is_public_lazy_submodule():
    assert sciscape.evolution.build_evolution_state_table is build_evolution_state_table
    assert sciscape.evolution.build_evolution_transition_table is build_evolution_transition_table
    assert sciscape.evolution.build_membership_projection_evolution is build_membership_projection_evolution
    assert sciscape.evolution.label_evolution_transition_relations is label_evolution_transition_relations
    assert sciscape.evolution.rank_evolution_transitions is rank_evolution_transitions


def test_membership_projection_evolution_builds_in_memory_tables():
    records = pd.DataFrame(
        {
            "uid": ["D0", "D1", "D2", "D3", "D4", "D5"],
            "title": ["a", "b", "c", "d", "e", "f"],
            "abstract": [""] * 6,
            "pubyear": [2020, 2020, 2021, 2021, 2021, 2022],
        }
    )
    membership = pd.DataFrame(
        {
            "uid": ["D0", "D1", "D2", "D3", "D4", "D5"],
            "cluster": [0, 0, 0, 0, 1, 1],
        }
    )
    keywords = pd.DataFrame(
        {
            "cluster_id": [0, 0, 1],
            "term": ["perovskite", "passivation", "stability"],
        }
    )

    result = build_membership_projection_evolution(
        evolution_id="Yearly Evolution Demo",
        records_df=records,
        membership_df=membership,
        keywords_df=keywords,
    )

    assert result.evolution_id == "Yearly_Evolution_Demo"
    assert result.periodization["start_year"] == 2020
    assert result.periodization["end_year"] == 2022
    assert result.entity_scope["cluster_level"] == "cluster"
    assert result.matching_method["metric"] == "projected_cluster_identity"
    assert result.slices["active_cluster_count"].tolist() == [1, 2, 1]
    assert len(result.states) == 4
    assert len(result.transitions) == 2
    assert set(result.transitions["relation"]) == {"continuation"}
    assert set(result.events["event_type"]) == {"continuation", "emergence", "decline"}
    state_terms = dict(zip(result.states["cluster_key"], result.states["top_terms"]))
    assert '"perovskite"' in state_terms["cluster:0"]
    assert '"stability"' in state_terms["cluster:1"]
    assert [row["step"] for row in result.transforms] == [
        "parse_publication_years",
        "build_time_slices",
        "project_static_membership_to_slices",
        "score_adjacent_slice_transitions",
        "build_lineages",
        "assign_evolution_events",
    ]


def test_membership_projection_evolution_rejects_missing_cluster_column():
    records = pd.DataFrame({"uid": ["D0"], "pubyear": [2020]})
    membership = pd.DataFrame({"uid": ["D0"], "label": [0]})

    try:
        build_membership_projection_evolution(
            evolution_id="bad",
            records_df=records,
            membership_df=membership,
        )
    except ValueError as exc:
        assert "cluster or cluster_* column" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected missing cluster column to fail")


def test_membership_projection_evolution_rejects_unsupported_periodization():
    records = pd.DataFrame({"uid": ["D0", "D1"], "pubyear": [2020, 2021]})
    membership = pd.DataFrame({"uid": ["D0", "D1"], "cluster": [0, 0]})

    with pytest.raises(ValueError, match="window_years=1"):
        build_membership_projection_evolution(
            evolution_id="bad-period",
            records_df=records,
            membership_df=membership,
            periodization={"window_years": 2},
        )


def test_membership_projection_evolution_counts_unique_docs_after_join():
    records = pd.DataFrame(
        {
            "uid": ["D0", "D0", "D1", "D2"],
            "pubyear": [2020, 2020, 2020, 2021],
        }
    )
    membership = pd.DataFrame(
        {
            "uid": ["D0", "D0", "D1", "D2"],
            "cluster": [0, 0, 0, 0],
        }
    )

    result = build_membership_projection_evolution(
        evolution_id="dedupe",
        records_df=records,
        membership_df=membership,
    )

    states_by_slice = dict(zip(result.states["slice_id"], result.states["doc_count"]))
    assert states_by_slice == {"year:2020": 2, "year:2021": 1}
    assert result.slices["doc_count"].tolist() == [2, 1]


def test_membership_projection_evolution_enforces_min_transition_support():
    records = pd.DataFrame({"uid": ["D0", "D1"], "pubyear": [2020, 2021]})
    membership = pd.DataFrame({"uid": ["D0", "D1"], "cluster": [0, 0]})

    result = build_membership_projection_evolution(
        evolution_id="support",
        records_df=records,
        membership_df=membership,
        matching_method={"min_support_count": 2},
    )

    assert result.transitions.empty
    assert set(result.events["event_type"]) == {"decline", "emergence"}


def test_membership_projection_evolution_rejects_jaccard_doc_overlap_metric():
    records = pd.DataFrame({"uid": ["D0", "D1"], "pubyear": [2020, 2021]})
    membership = pd.DataFrame({"uid": ["D0", "D1"], "cluster": [0, 0]})

    with pytest.raises(ValueError, match="unsupported evolution matching metric"):
        build_membership_projection_evolution(
            evolution_id="bad-metric",
            records_df=records,
            membership_df=membership,
            matching_method={"metric": "jaccard_doc_overlap"},
        )


def test_build_evolution_state_table_normalizes_raw_state_evidence():
    slices = pd.DataFrame(
        [
            {"slice_id": "year:2020", "slice_index": 0},
            {"slice_id": "year:2021", "slice_index": 1},
        ]
    )
    evidence = pd.DataFrame(
        [
            {
                "slice_id": "year:2020",
                "cluster_id": "A",
                "level": "topic",
                "cluster_label": "Alpha materials",
                "doc_count": 2,
                "top_terms": ["alpha", "materials"],
                "representative_work_ids": ["D0", "D1"],
                "centroid_x": 0.25,
            },
            {
                "slice_id": "year:2021",
                "cluster_key": "topic:B",
                "label": "Beta methods",
                "doc_count": "3",
                "top_terms": '["beta"]',
            },
        ]
    )

    states = build_evolution_state_table(
        evolution_id="state demo",
        slices=slices,
        state_evidence=evidence,
        default_level="topic",
    )

    by_state = {row.state_id: row for row in states.itertuples(index=False)}
    assert set(by_state) == {"year:2020_topic:A", "year:2021_topic:B"}
    assert by_state["year:2020_topic:A"].schema_version == "sciscape_evolution_cluster_states_v1"
    assert by_state["year:2020_topic:A"].evolution_id == "state_demo"
    assert by_state["year:2020_topic:A"].cluster_key == "topic:A"
    assert by_state["year:2020_topic:A"].cluster_id == "A"
    assert by_state["year:2020_topic:A"].term_count == 2
    assert by_state["year:2020_topic:A"].top_terms == '["alpha", "materials"]'
    assert by_state["year:2020_topic:A"].representative_work_ids == '["D0", "D1"]'
    assert by_state["year:2020_topic:A"].centroid_x == 0.25
    assert by_state["year:2021_topic:B"].slice_index == 1
    assert by_state["year:2021_topic:B"].cluster_label == "Beta methods"
    assert by_state["year:2021_topic:B"].doc_count == 3


def test_build_evolution_state_table_rejects_unknown_slice_and_duplicate_state_key():
    slices = pd.DataFrame([{"slice_id": "year:2020", "slice_index": 0}])
    unknown = pd.DataFrame([{"slice_id": "year:2021", "cluster_id": "A", "doc_count": 1}])

    with pytest.raises(ValueError, match="unknown slice_id"):
        build_evolution_state_table(evolution_id="bad", slices=slices, state_evidence=unknown)

    duplicate = pd.DataFrame(
        [
            {"slice_id": "year:2020", "cluster_id": "A", "doc_count": 1},
            {"slice_id": "year:2020", "cluster_id": "A", "doc_count": 2},
        ]
    )
    with pytest.raises(ValueError, match="duplicate slice_id/cluster_key"):
        build_evolution_state_table(evolution_id="bad", slices=slices, state_evidence=duplicate)


def test_classify_evolution_events_detects_split_merge_and_ambiguous():
    slices = pd.DataFrame(
        [
            {"slice_id": "year:2020", "slice_index": 0},
            {"slice_id": "year:2021", "slice_index": 1},
        ]
    )
    states = pd.DataFrame(
        [
            {"state_id": "B20", "slice_id": "year:2020", "slice_index": 0, "doc_count": 6},
            {"state_id": "B21a", "slice_id": "year:2021", "slice_index": 1, "doc_count": 3},
            {"state_id": "B21b", "slice_id": "year:2021", "slice_index": 1, "doc_count": 3},
            {"state_id": "C20a", "slice_id": "year:2020", "slice_index": 0, "doc_count": 3},
            {"state_id": "C20b", "slice_id": "year:2020", "slice_index": 0, "doc_count": 3},
            {"state_id": "C21", "slice_id": "year:2021", "slice_index": 1, "doc_count": 6},
            {"state_id": "X20", "slice_id": "year:2020", "slice_index": 0, "doc_count": 6},
            {"state_id": "Y21", "slice_id": "year:2021", "slice_index": 1, "doc_count": 3},
            {"state_id": "Z21", "slice_id": "year:2021", "slice_index": 1, "doc_count": 3},
        ]
    )
    transitions = pd.DataFrame(
        [
            {
                "transition_id": "t_B20_B21a",
                "source_state_id": "B20",
                "target_state_id": "B21a",
                "source_slice_id": "year:2020",
                "target_slice_id": "year:2021",
                "score": 0.76,
                "support_count": 3,
                "relation": "split_child",
            },
            {
                "transition_id": "t_B20_B21b",
                "source_state_id": "B20",
                "target_state_id": "B21b",
                "source_slice_id": "year:2020",
                "target_slice_id": "year:2021",
                "score": 0.74,
                "support_count": 3,
                "relation": "split_child",
            },
            {
                "transition_id": "t_C20a_C21",
                "source_state_id": "C20a",
                "target_state_id": "C21",
                "source_slice_id": "year:2020",
                "target_slice_id": "year:2021",
                "score": 0.81,
                "support_count": 3,
                "relation": "merge_parent",
            },
            {
                "transition_id": "t_C20b_C21",
                "source_state_id": "C20b",
                "target_state_id": "C21",
                "source_slice_id": "year:2020",
                "target_slice_id": "year:2021",
                "score": 0.79,
                "support_count": 3,
                "relation": "merge_parent",
            },
            {
                "transition_id": "t_X20_Y21",
                "source_state_id": "X20",
                "target_state_id": "Y21",
                "source_slice_id": "year:2020",
                "target_slice_id": "year:2021",
                "score": 0.61,
                "support_count": 3,
                "relation": "ambiguous",
            },
            {
                "transition_id": "t_X20_Z21",
                "source_state_id": "X20",
                "target_state_id": "Z21",
                "source_slice_id": "year:2020",
                "target_slice_id": "year:2021",
                "score": 0.60,
                "support_count": 3,
                "relation": "ambiguous",
            },
        ]
    )
    lineages = pd.DataFrame(
        {
            "state_id": states["state_id"],
            "lineage_id": [f"lineage_{state_id}" for state_id in states["state_id"]],
        }
    )

    events = classify_evolution_events(
        evolution_id="classifier",
        slices=slices,
        states=states,
        transitions=transitions,
        lineages=lineages,
    )

    assert events["event_type"].value_counts().to_dict() == {"split": 1, "merge": 1, "ambiguous": 1}
    event_by_type = {row.event_type: row for row in events.itertuples(index=False)}
    assert event_by_type["split"].source_state_ids == '["B20"]'
    assert event_by_type["split"].target_state_ids == '["B21a", "B21b"]'
    assert event_by_type["merge"].source_state_ids == '["C20a", "C20b"]'
    assert event_by_type["merge"].target_state_ids == '["C21"]'
    assert event_by_type["ambiguous"].transition_refs == '["t_X20_Y21", "t_X20_Z21"]'


def test_classify_evolution_events_rejects_degenerate_split_rule():
    slices = pd.DataFrame([{"slice_id": "year:2020", "slice_index": 0}])
    states = pd.DataFrame([{"state_id": "A20", "slice_id": "year:2020", "slice_index": 0, "doc_count": 1}])
    lineages = pd.DataFrame([{"state_id": "A20", "lineage_id": "lineage_A"}])

    with pytest.raises(ValueError, match="split_min_children"):
        classify_evolution_events(
            evolution_id="bad-rule",
            slices=slices,
            states=states,
            transitions=pd.DataFrame(),
            lineages=lineages,
            event_rules={"split_min_children": 1},
        )


def test_rank_evolution_transitions_is_deterministic_by_source_and_target():
    transitions = pd.DataFrame(
        [
            {
                "transition_id": "t_A_C",
                "source_state_id": "A",
                "target_state_id": "C",
                "score": 0.7,
                "support_count": 5,
            },
            {
                "transition_id": "t_A_B",
                "source_state_id": "A",
                "target_state_id": "B",
                "score": 0.9,
                "support_count": 2,
            },
            {
                "transition_id": "t_D_B",
                "source_state_id": "D",
                "target_state_id": "B",
                "score": 0.9,
                "support_count": 3,
            },
            {
                "transition_id": "t_E_B",
                "source_state_id": "E",
                "target_state_id": "B",
                "score": 0.9,
                "support_count": 3,
            },
        ]
    )

    ranked = rank_evolution_transitions(transitions)
    by_id = {row.transition_id: row for row in ranked.itertuples(index=False)}

    assert by_id["t_A_B"].rank_from_source == 1
    assert by_id["t_A_C"].rank_from_source == 2
    assert by_id["t_D_B"].rank_to_target == 1
    assert by_id["t_E_B"].rank_to_target == 2
    assert by_id["t_A_B"].rank_to_target == 3


def test_label_evolution_transition_relations_marks_candidates_without_overwriting_explicit_labels():
    transitions = pd.DataFrame(
        [
            {
                "transition_id": "t_A_B",
                "source_state_id": "A",
                "target_state_id": "B",
                "score": 0.9,
                "support_count": 3,
                "relation": "candidate",
            },
            {
                "transition_id": "t_A_C",
                "source_state_id": "A",
                "target_state_id": "C",
                "score": 0.8,
                "support_count": 3,
                "relation": "candidate",
            },
            {
                "transition_id": "t_D_E",
                "source_state_id": "D",
                "target_state_id": "E",
                "score": 0.9,
                "support_count": 3,
                "relation": "candidate",
            },
            {
                "transition_id": "t_F_E",
                "source_state_id": "F",
                "target_state_id": "E",
                "score": 0.8,
                "support_count": 3,
                "relation": "candidate",
            },
            {
                "transition_id": "t_G_H",
                "source_state_id": "G",
                "target_state_id": "H",
                "score": 0.9,
                "support_count": 3,
                "relation": "",
            },
            {
                "transition_id": "t_X_Y",
                "source_state_id": "X",
                "target_state_id": "Y",
                "score": 0.6,
                "support_count": 3,
                "relation": "ambiguous",
            },
        ]
    )

    labeled = label_evolution_transition_relations(transitions)
    by_id = {row.transition_id: row for row in labeled.itertuples(index=False)}

    assert by_id["t_A_B"].relation == "split_child"
    assert by_id["t_A_C"].relation == "split_child"
    assert by_id["t_D_E"].relation == "merge_parent"
    assert by_id["t_F_E"].relation == "merge_parent"
    assert by_id["t_G_H"].relation == "continuation"
    assert by_id["t_X_Y"].relation == "ambiguous"


def test_build_evolution_transition_table_normalizes_raw_evidence():
    states = pd.DataFrame(
        [
            {"state_id": "A20", "slice_id": "year:2020", "slice_index": 0, "doc_count": 6},
            {"state_id": "B21a", "slice_id": "year:2021", "slice_index": 1, "doc_count": 3},
            {"state_id": "B21b", "slice_id": "year:2021", "slice_index": 1, "doc_count": 3},
            {"state_id": "C20a", "slice_id": "year:2020", "slice_index": 0, "doc_count": 3},
            {"state_id": "C20b", "slice_id": "year:2020", "slice_index": 0, "doc_count": 3},
            {"state_id": "C21", "slice_id": "year:2021", "slice_index": 1, "doc_count": 6},
            {"state_id": "D20", "slice_id": "year:2020", "slice_index": 0, "doc_count": 4},
            {"state_id": "D21", "slice_id": "year:2021", "slice_index": 1, "doc_count": 4},
            {"state_id": "E21", "slice_id": "year:2021", "slice_index": 1, "doc_count": 4},
        ]
    )
    evidence = pd.DataFrame(
        [
            {"source_state_id": "A20", "target_state_id": "B21a", "score": 0.76, "support_count": 3},
            {"source_state_id": "A20", "target_state_id": "B21b", "score": 0.74, "support_count": 3},
            {"source_state_id": "C20a", "target_state_id": "C21", "score": 0.81, "support_count": 3},
            {"source_state_id": "C20b", "target_state_id": "C21", "score": 0.79, "support_count": 3},
            {"source_state_id": "D20", "target_state_id": "D21", "score": 0.91, "support_count": 4, "evidence_ref": "manual:D"},
            {"source_state_id": "D20", "target_state_id": "E21", "score": 0.10, "support_count": 4},
        ]
    )

    transitions = build_evolution_transition_table(
        evolution_id="raw evidence",
        states=states,
        transition_evidence=evidence,
        metric="term_overlap",
        matching_method={"min_transition_score": 0.5, "min_support_count": 2},
    )

    assert len(transitions) == 5
    by_pair = {(row.source_state_id, row.target_state_id): row for row in transitions.itertuples(index=False)}
    assert by_pair[("A20", "B21a")].relation == "split_child"
    assert by_pair[("A20", "B21a")].source_doc_count == 6
    assert by_pair[("A20", "B21a")].target_doc_count == 3
    assert by_pair[("C20a", "C21")].relation == "merge_parent"
    assert by_pair[("D20", "D21")].relation == "continuation"
    assert by_pair[("D20", "D21")].metric == "term_overlap"
    assert by_pair[("D20", "D21")].evidence_ref == "manual:D"
    assert ("D20", "E21") not in by_pair


def test_build_evolution_transition_table_rejects_unknown_or_non_adjacent_states():
    states = pd.DataFrame(
        [
            {"state_id": "A20", "slice_id": "year:2020", "slice_index": 0, "doc_count": 1},
            {"state_id": "A22", "slice_id": "year:2022", "slice_index": 2, "doc_count": 1},
        ]
    )
    unknown = pd.DataFrame([{"source_state_id": "missing", "target_state_id": "A22", "score": 1.0, "support_count": 1}])

    with pytest.raises(ValueError, match="unknown source_state_id"):
        build_evolution_transition_table(
            evolution_id="bad",
            states=states,
            transition_evidence=unknown,
            metric="term_overlap",
        )

    non_adjacent = pd.DataFrame([{"source_state_id": "A20", "target_state_id": "A22", "score": 1.0, "support_count": 1}])
    with pytest.raises(ValueError, match="adjacent slices"):
        build_evolution_transition_table(
            evolution_id="bad",
            states=states,
            transition_evidence=non_adjacent,
            metric="term_overlap",
        )
