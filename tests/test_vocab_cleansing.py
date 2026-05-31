"""Tests for Stage 3: full-vocabulary cleansing."""

import numpy as np
from scipy import sparse as sp

from sciscape.keyword_extraction.vocab_cleansing import (
    VocabSimGraph,
    _build_edit_distance_merge_map,
    _build_norm_rename_map,
    _build_plural_merge_map,
    _build_similarity_graph,
    _merge_from_rename,
    run_vocab_cleansing,
)


def _make_count_matrix(names, counts_per_cluster):
    """Helper: create a (K, V) sparse count matrix."""
    K = len(counts_per_cluster)
    V = len(names)
    data = np.array(counts_per_cluster, dtype=np.int64)
    return sp.csr_matrix(data.reshape(K, V))


class TestNormRenameMap:
    def test_greek_letters(self):
        names = np.array(["α-ray", "beta", "normal"])
        rename = _build_norm_rename_map(names)
        assert 0 in rename
        assert rename[0] == "alpha ray"
        assert 1 not in rename  # already latin
        assert 2 not in rename

    def test_spelling_variants(self):
        names = np.array(["colour", "center", "behaviour"])
        rename = _build_norm_rename_map(names)
        assert rename[0] == "color"
        assert 1 not in rename  # "center" is already AmE
        assert rename[2] == "behavior"

    def test_hyphen_normalization(self):
        names = np.array(["gamma-ray", "x-ray"])
        rename = _build_norm_rename_map(names)
        assert rename[0] == "gamma ray"
        assert rename[1] == "x ray"


class TestMergeFromRename:
    def test_two_forms_merge(self):
        names = np.array(["colour", "color", "other"])
        rename = {0: "color"}  # colour → color
        merge_map, final_rename = _merge_from_rename(names, rename)
        # "colour"(0) normalizes to "color", "color"(1) is already "color"
        # idx 0 should merge into idx 1
        assert 0 in merge_map
        assert merge_map[0] == 1
        assert 2 not in merge_map

    def test_no_overlap(self):
        names = np.array(["alpha", "beta"])
        rename = {}
        merge_map, final_rename = _merge_from_rename(names, rename)
        assert merge_map == {}


class TestPluralMerge:
    def test_unigram_plural(self):
        names = np.array(["network", "networks", "other"])
        C = _make_count_matrix(names, [[100, 50, 30]])
        merge = _build_plural_merge_map(names, {}, C=C)
        assert 1 in merge  # networks → network
        assert merge[1] == 0

    def test_phrase_plural(self):
        names = np.array(["neural network", "neural networks"])
        C = _make_count_matrix(names, [[100, 50]])
        merge = _build_plural_merge_map(names, {}, C=C)
        assert 1 in merge
        assert merge[1] == 0

    def test_skip_invariant(self):
        names = np.array(["series", "network"])
        C = _make_count_matrix(names, [[50, 100]])
        merge = _build_plural_merge_map(names, {}, C=C)
        assert merge == {}


class TestEditDistanceMerge:
    def test_typo_merged(self):
        """Minor typo with very low frequency should be merged."""
        names = np.array(["network", "netwerk", "other"])
        # netwerk appears only 1 time globally, network 1000
        C = _make_count_matrix(names, [[1000, 1, 500]])
        merge = _build_edit_distance_merge_map(names, {}, C, max_edit_distance=1, global_ratio_threshold=0.01)
        assert 1 in merge
        assert merge[1] == 0

    def test_cluster_dominance_blocks_merge(self):
        """If minor form leads in any cluster, don't merge."""
        names = np.array(["network", "netwerk"])
        # Cluster 0: network=1000, netwerk=1
        # Cluster 1: network=1, netwerk=5  ← minor leads here
        C = _make_count_matrix(names, [[1000, 1], [1, 5]])
        merge = _build_edit_distance_merge_map(names, {}, C, max_edit_distance=1, global_ratio_threshold=0.1)
        # netwerk leads in cluster 1, so merge should be blocked
        assert merge == {}

    def test_phrases_skipped(self):
        """Edit-distance merge is unigram-only."""
        names = np.array(["deep learning", "deep learling"])
        C = _make_count_matrix(names, [[1000, 1]])
        merge = _build_edit_distance_merge_map(names, {}, C, max_edit_distance=1)
        assert merge == {}

    def test_short_terms_skipped(self):
        """Terms <= 3 chars are too short for safe edit-distance merge."""
        names = np.array(["abc", "abd"])
        C = _make_count_matrix(names, [[1000, 1]])
        merge = _build_edit_distance_merge_map(names, {}, C, max_edit_distance=1)
        assert merge == {}


class TestSimilarityGraph:
    def test_builds_edges(self):
        names = np.array(["network", "netwerk", "other"])
        C = _make_count_matrix(names, [[100, 50, 200]])
        graph = _build_similarity_graph(names, {}, C, max_edit_distance=2)
        assert len(graph) >= 1
        nbrs = graph.neighbor_terms("network")
        assert "netwerk" in nbrs

    def test_phrase_edges(self):
        names = np.array(["deep learning", "deep learling"])
        C = _make_count_matrix(names, [[100, 50]])
        graph = _build_similarity_graph(names, {}, C, max_edit_distance=2)
        assert len(graph) >= 1

    def test_excludes_merged(self):
        names = np.array(["network", "netwerk", "other"])
        C = _make_count_matrix(names, [[100, 50, 200]])
        graph = _build_similarity_graph(names, {1: 0}, C, max_edit_distance=2)
        # idx 1 is merged away, should not appear as edges
        assert all("netwerk" not in (a, b) for a, b, d, m in graph.edges)


class TestRunVocabCleansing:
    def test_end_to_end(self):
        """Smoke test for the full orchestrator."""
        uni_names = np.array(["network", "networks", "colour", "color", "other"])
        phrase_names = np.array(["neural network", "neural networks"])
        C_uni = _make_count_matrix(uni_names, [[100, 50, 30, 80, 200]])
        C_phrase = _make_count_matrix(phrase_names, [[60, 40]])
        DF_uni = C_uni.copy()
        DF_phrase = C_phrase.copy()

        (
            fn_uni, fn_phrase, C_uni_out, C_phrase_out,
            DF_uni_out, DF_phrase_out, sim_graph, merge_log,
        ) = run_vocab_cleansing(
            uni_names, phrase_names, C_uni, C_phrase, DF_uni, DF_phrase,
        )

        # "colour" should merge into "color" (3a)
        assert "colour" not in fn_uni
        assert "color" in fn_uni
        # "networks" should merge into "network" (3b)
        assert "networks" not in fn_uni
        assert "network" in fn_uni
        # "neural networks" should merge into "neural network" (3b)
        assert "neural networks" not in fn_phrase
        assert "neural network" in fn_phrase
        # sim_graph should exist
        assert isinstance(sim_graph, VocabSimGraph)
        # merge_log should record merges
        assert len(merge_log) > 0

    def test_no_data(self):
        """Empty input should not crash."""
        uni = np.array([], dtype=str)
        C = sp.csr_matrix((1, 0), dtype=np.int64)
        fn_uni, fn_phrase, *_ = run_vocab_cleansing(
            uni, np.array([], dtype=str), C, None, None, None,
        )
        assert len(fn_uni) == 0

    def test_can_skip_similarity_graph(self):
        """Batch profiles can skip the expensive full-vocabulary graph."""
        uni_names = np.array(["network", "netwerk"])
        C_uni = _make_count_matrix(uni_names, [[100, 50]])

        (
            _fn_uni, _fn_phrase, _C_uni_out, _C_phrase_out,
            _DF_uni_out, _DF_phrase_out, sim_graph, _merge_log,
        ) = run_vocab_cleansing(
            uni_names,
            np.array([], dtype=str),
            C_uni,
            None,
            C_uni.copy(),
            None,
            build_similarity_graph=False,
        )

        assert isinstance(sim_graph, VocabSimGraph)
        assert len(sim_graph) == 0
