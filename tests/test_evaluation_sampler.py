"""Tests for disagreement-case sampling."""

import polars as pl

from sciscape.evaluation.sampler import (
    collect_rank_shift_cases,
    sample_disagreement_cases,
    sample_rank_shift_cases,
    sample_worst_case,
)


def _make_edges(rows):
    return pl.DataFrame(
        {
            "uid1": [row[0] for row in rows],
            "uid2": [row[1] for row in rows],
            "rel_sum2": [row[2] for row in rows],
        }
    )


class TestSampleDisagreementCases:

    def test_samples_meaningfully_different_groups(self):
        edges_a = _make_edges(
            [
                ("T", "A1", 3.0),
                ("T", "A2", 2.5),
                ("A1", "A2", 2.0),
                ("B1", "B2", 2.0),
                ("B2", "B3", 1.5),
            ]
        )
        edges_b = _make_edges(
            [
                ("T", "B1", 3.0),
                ("T", "B2", 2.5),
                ("B1", "B2", 2.0),
                ("A1", "A2", 1.5),
            ]
        )
        membership_a = {"T": 1, "A1": 1, "A2": 1, "B1": 2, "B2": 2, "B3": 2}
        membership_b = {"T": 3, "B1": 3, "B2": 3, "A1": 4, "A2": 4, "B3": 5}

        result = sample_disagreement_cases(
            edges_a,
            membership_a,
            edges_b,
            membership_b,
            n_targets=10,
            n_neighbors=2,
            min_cluster_size=2,
            boundary_quantile=0.0,
            max_group_jaccard=0.5,
            seed=7,
        )

        target_case = next(case for case in result.cases if case.target_uid == "T")
        assert target_case.group_a_uids == ["A1", "A2"]
        assert target_case.group_b_uids == ["B1", "B2"]
        assert target_case.jaccard == 0.0

    def test_falls_back_to_cluster_members_without_direct_same_cluster_edges(self):
        edges_a = _make_edges(
            [
                ("A1", "A2", 1.5),
                ("T", "X", 2.0),
            ]
        )
        edges_b = _make_edges(
            [
                ("T", "B1", 2.0),
                ("T", "B2", 1.8),
                ("B1", "B2", 1.2),
            ]
        )
        membership_a = {"T": 1, "A1": 1, "A2": 1, "X": 2}
        membership_b = {"T": 3, "B1": 3, "B2": 3, "A1": 4, "A2": 4, "X": 5}

        result = sample_disagreement_cases(
            edges_a,
            membership_a,
            edges_b,
            membership_b,
            n_targets=10,
            n_neighbors=2,
            min_cluster_size=2,
            boundary_quantile=0.0,
            max_group_jaccard=1.0,
            seed=11,
        )

        target_case = next(case for case in result.cases if case.target_uid == "T")
        assert target_case.group_a_uids == ["A1", "A2"]
        assert target_case.group_b_uids == ["B1", "B2"]

    def test_excludes_identical_groups(self):
        edges_a = _make_edges(
            [
                ("T", "A1", 3.0),
                ("T", "A2", 2.5),
                ("A1", "A2", 2.0),
            ]
        )
        edges_b = _make_edges(
            [
                ("T", "A1", 4.0),
                ("T", "A2", 1.0),
                ("A1", "A2", 2.2),
            ]
        )
        membership = {"T": 1, "A1": 1, "A2": 1}

        result = sample_disagreement_cases(
            edges_a,
            membership,
            edges_b,
            membership,
            n_targets=10,
            n_neighbors=2,
            min_cluster_size=2,
            boundary_quantile=0.0,
            max_group_jaccard=1.0,
            seed=5,
        )

        assert result.cases == []

    def test_prefers_metadata_covered_targets_and_groups(self):
        edges_a = _make_edges(
            [
                ("T", "A1", 3.0),
                ("T", "A2", 2.5),
                ("T", "A3", 2.0),
                ("A1", "A2", 1.5),
                ("A2", "A3", 1.2),
                ("X", "Y", 1.0),
            ]
        )
        edges_b = _make_edges(
            [
                ("T", "B1", 3.1),
                ("T", "B2", 2.7),
                ("T", "B3", 2.3),
                ("B1", "B2", 1.6),
                ("B2", "B3", 1.1),
                ("X", "Y", 1.0),
            ]
        )
        membership_a = {
            "T": 1, "A1": 1, "A2": 1, "A3": 1,
            "B1": 2, "B2": 2, "B3": 2,
            "X": 3, "Y": 3,
        }
        membership_b = {
            "T": 4, "B1": 4, "B2": 4, "B3": 4,
            "A1": 5, "A2": 5, "A3": 5,
            "X": 6, "Y": 6,
        }

        result = sample_disagreement_cases(
            edges_a,
            membership_a,
            edges_b,
            membership_b,
            n_targets=10,
            n_neighbors=3,
            min_cluster_size=2,
            boundary_quantile=0.0,
            max_group_jaccard=1.0,
            allowed_uids={"T", "A1", "A2", "B1", "B2"},
            seed=13,
        )

        target_case = next(case for case in result.cases if case.target_uid == "T")
        assert target_case.group_a_uids == ["A1", "A2"]
        assert target_case.group_b_uids == ["B1", "B2"]


class TestSampleRankShiftCases:

    def test_samples_targets_with_changed_top_neighbors(self):
        edges_a = _make_edges(
            [
                ("T", "A1", 4.0),
                ("T", "A2", 3.0),
                ("T", "C1", 1.0),
                ("A1", "A2", 1.0),
                ("B1", "B2", 1.0),
            ]
        )
        edges_b = _make_edges(
            [
                ("T", "B1", 4.5),
                ("T", "B2", 3.5),
                ("T", "C1", 1.0),
                ("B1", "B2", 1.0),
                ("A1", "A2", 1.0),
            ]
        )
        membership_a = {"T": 1, "A1": 1, "A2": 1, "B1": 2, "B2": 2, "C1": 3}
        membership_b = {"T": 4, "B1": 4, "B2": 4, "A1": 5, "A2": 5, "C1": 6}

        result = sample_rank_shift_cases(
            edges_a,
            membership_a,
            edges_b,
            membership_b,
            n_targets=10,
            n_neighbors=2,
            min_cluster_size=2,
            max_rank_jaccard=1.0,
            allowed_uids={"T", "A1", "A2", "B1", "B2"},
            seed=17,
        )

        target_case = next(case for case in result.cases if case.target_uid == "T")
        assert [row["uid"] for row in target_case.neighbors_a] == ["A1", "A2"]
        assert [row["uid"] for row in target_case.neighbors_b] == ["B1", "B2"]
        assert target_case.rank_jaccard == 0.0
        assert target_case.cluster_overlap_coeff == 0.3333
        assert target_case.cluster_changed is True

    def test_does_not_treat_relabelled_cluster_ids_as_changed(self):
        edges_a = _make_edges(
            [
                ("T", "A1", 4.0),
                ("T", "A2", 3.0),
                ("A1", "A2", 1.0),
            ]
        )
        edges_b = _make_edges(
            [
                ("T", "A1", 4.0),
                ("T", "A2", 3.0),
                ("A1", "A2", 1.0),
            ]
        )
        membership_a = {"T": 1, "A1": 1, "A2": 1}
        membership_b = {"T": 9, "A1": 9, "A2": 9}

        result = sample_rank_shift_cases(
            edges_a,
            membership_a,
            edges_b,
            membership_b,
            n_targets=10,
            n_neighbors=2,
            min_cluster_size=2,
            max_rank_jaccard=1.0,
            allowed_uids={"T", "A1", "A2"},
            seed=23,
        )

        assert result.cases == []


class TestSampleWorstCase:

    def test_reports_cross_edge_counts_not_weight_sums(self):
        edges = _make_edges(
            [
                ("T", "A1", 2.0),
                ("T", "A2", 1.5),
                ("T", "X1", 4.0),
                ("T", "X2", 7.0),
                ("A1", "A2", 1.0),
            ]
        )
        membership = {"T": 1, "A1": 1, "A2": 1, "X1": 2, "X2": 3}

        result = sample_worst_case(
            edges,
            membership,
            n_targets=3,
            n_easy=2,
            n_hard=1,
            min_cluster_size=2,
            boundary_quantile=0.0,
            seed=19,
        )

        target_case = next(case for case in result.cases if case.target_uid == "T")
        assert target_case.n_cross_edges == 2
        assert target_case.cross_cluster_ratio == 0.7586

    def test_tracks_rank_reordering_for_shared_neighbors(self):
        edges_a = _make_edges(
            [
                ("T", "A1", 5.0),
                ("T", "A2", 4.0),
                ("T", "A3", 3.0),
                ("A1", "A2", 1.0),
            ]
        )
        edges_b = _make_edges(
            [
                ("T", "A3", 5.0),
                ("T", "A2", 4.0),
                ("T", "A1", 3.0),
                ("A1", "A2", 1.0),
            ]
        )
        membership = {"T": 1, "A1": 1, "A2": 1, "A3": 1}

        result = sample_rank_shift_cases(
            edges_a,
            membership,
            edges_b,
            membership,
            n_targets=10,
            n_neighbors=3,
            min_cluster_size=2,
            max_rank_jaccard=1.0,
            allowed_uids={"T", "A1", "A2", "A3"},
            seed=19,
        )

        target_case = next(case for case in result.cases if case.target_uid == "T")
        assert target_case.overlap_size == 3
        assert target_case.max_abs_rank_shift == 2
        assert target_case.mean_abs_rank_shift > 0
        assert target_case.shared_neighbors[0]["uid"] in {"A1", "A3"}

    def test_collect_rank_shift_cases_prefers_case_bank_order_then_fallback(self):
        edges_a = _make_edges(
            [
                ("T1", "A1", 4.0),
                ("T1", "A2", 3.0),
                ("T2", "C1", 4.0),
                ("T2", "C2", 3.0),
                ("A1", "A2", 1.0),
                ("C1", "C2", 1.0),
                ("B1", "B2", 1.0),
                ("D1", "D2", 1.0),
            ]
        )
        edges_b = _make_edges(
            [
                ("T1", "B1", 4.5),
                ("T1", "B2", 3.5),
                ("T2", "D1", 4.5),
                ("T2", "D2", 3.5),
                ("A1", "A2", 1.0),
                ("B1", "B2", 1.0),
                ("C1", "C2", 1.0),
                ("D1", "D2", 1.0),
            ]
        )
        membership_a = {
            "T1": 1, "A1": 1, "A2": 1,
            "T2": 2, "C1": 2, "C2": 2,
            "B1": 3, "B2": 3,
            "D1": 4, "D2": 4,
        }
        membership_b = {
            "T1": 5, "B1": 5, "B2": 5,
            "T2": 6, "D1": 6, "D2": 6,
            "A1": 7, "A2": 7,
            "C1": 8, "C2": 8,
        }

        cases, n_eligible = collect_rank_shift_cases(
            edges_a,
            membership_a,
            edges_b,
            membership_b,
            n_neighbors=2,
            min_cluster_size=2,
            max_rank_jaccard=1.0,
            allowed_uids={"T1", "T2", "A1", "A2", "B1", "B2", "C1", "C2", "D1", "D2"},
            target_uids=["T2"],
        )

        assert n_eligible >= 2
        assert [case.target_uid for case in cases[:2]] == ["T2", "T1"]

    def test_sample_rank_shift_cases_falls_back_outside_case_bank_when_not_strict(self):
        edges_a = _make_edges(
            [
                ("T1", "A1", 4.0),
                ("T1", "A2", 3.0),
                ("T2", "C1", 4.0),
                ("T2", "C2", 3.0),
                ("A1", "A2", 1.0),
                ("C1", "C2", 1.0),
                ("B1", "B2", 1.0),
                ("D1", "D2", 1.0),
            ]
        )
        edges_b = _make_edges(
            [
                ("T1", "B1", 4.5),
                ("T1", "B2", 3.5),
                ("T2", "D1", 4.5),
                ("T2", "D2", 3.5),
                ("A1", "A2", 1.0),
                ("B1", "B2", 1.0),
                ("C1", "C2", 1.0),
                ("D1", "D2", 1.0),
            ]
        )
        membership_a = {
            "T1": 1, "A1": 1, "A2": 1,
            "T2": 2, "C1": 2, "C2": 2,
            "B1": 3, "B2": 3,
            "D1": 4, "D2": 4,
        }
        membership_b = {
            "T1": 5, "B1": 5, "B2": 5,
            "T2": 6, "D1": 6, "D2": 6,
            "A1": 7, "A2": 7,
            "C1": 8, "C2": 8,
        }

        loose = sample_rank_shift_cases(
            edges_a,
            membership_a,
            edges_b,
            membership_b,
            n_targets=2,
            n_neighbors=2,
            min_cluster_size=2,
            max_rank_jaccard=1.0,
            allowed_uids={"T1", "T2", "A1", "A2", "B1", "B2", "C1", "C2", "D1", "D2"},
            target_uids=["T2"],
            strict_target_uids=False,
            seed=23,
        )
        strict = sample_rank_shift_cases(
            edges_a,
            membership_a,
            edges_b,
            membership_b,
            n_targets=2,
            n_neighbors=2,
            min_cluster_size=2,
            max_rank_jaccard=1.0,
            allowed_uids={"T1", "T2", "A1", "A2", "B1", "B2", "C1", "C2", "D1", "D2"},
            target_uids=["T2"],
            strict_target_uids=True,
            seed=23,
        )

        assert [case.target_uid for case in loose.cases] == ["T2", "T1"]
        assert [case.target_uid for case in strict.cases] == ["T2"]
