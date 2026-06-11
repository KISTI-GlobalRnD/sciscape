"""Tests for in-memory cluster evolution analysis."""

from __future__ import annotations

import pandas as pd

import sciscape
from sciscape.evolution import build_membership_projection_evolution


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
