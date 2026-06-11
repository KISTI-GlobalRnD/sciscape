"""Tests for in-memory cluster evolution analysis."""

from __future__ import annotations

import pandas as pd
import pytest

import sciscape
from sciscape.evolution import build_membership_projection_evolution, classify_evolution_events


def test_evolution_is_public_lazy_submodule():
    assert sciscape.evolution.build_membership_projection_evolution is build_membership_projection_evolution


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
