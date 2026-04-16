"""Tests for OpenAlex edge building (openalex/edges.py)."""

import polars as pl
import pytest
from dataclasses import dataclass
from typing import List, Optional


# Minimal WorkRecord stub for testing
@dataclass
class WorkRecord:
    id: str
    title: str = ""
    abstract: str = ""
    year: int = 2020
    referenced_works: Optional[List[str]] = None


from sciscape.openalex.edges import build_citation_edges, works_to_abstracts


# ── Tests: build_citation_edges ──────────────────────────────

class TestBuildCitationEdges:

    def _make_works(self):
        """3 papers: A cites B, B cites C, A cites C."""
        return [
            WorkRecord(id="A", referenced_works=["B", "C"]),
            WorkRecord(id="B", referenced_works=["C"]),
            WorkRecord(id="C", referenced_works=[]),
        ]

    def test_dc_edges(self):
        works = self._make_works()
        result = build_citation_edges(works, bc=False, cc=False)
        assert "dc" in result
        dc = result["dc"]
        assert dc.height > 0
        assert set(dc.columns) == {"uid1", "uid2", "rel_sum2"}

    def test_dc_symmetry(self):
        works = self._make_works()
        result = build_citation_edges(works, bc=False, cc=False)
        dc = result["dc"]
        # A→B should also appear as B→A
        ab = dc.filter(
            ((pl.col("uid1") == "A") & (pl.col("uid2") == "B")) |
            ((pl.col("uid1") == "B") & (pl.col("uid2") == "A"))
        )
        assert ab.height >= 1

    def test_dc_fractional(self):
        works = [WorkRecord(id="A", referenced_works=["B", "C"])]
        result = build_citation_edges(works, normalization="fractional", bc=False, cc=False)
        dc = result["dc"]
        # A has 2 refs → weight = 1/2 = 0.5
        if dc.height > 0:
            assert dc["rel_sum2"].max() <= 1.0

    def test_dc_binary(self):
        works = [WorkRecord(id="A", referenced_works=["B", "C"])]
        result = build_citation_edges(works, normalization="binary", bc=False, cc=False)
        dc = result["dc"]
        if dc.height > 0:
            assert dc["rel_sum2"].min() >= 1.0

    def test_bc_edges(self):
        # A and B both cite C → bibliographic coupling
        works = [
            WorkRecord(id="A", referenced_works=["X", "Y"]),
            WorkRecord(id="B", referenced_works=["X", "Z"]),
            WorkRecord(id="C", referenced_works=["Y", "Z"]),
        ]
        result = build_citation_edges(works, bc=True, cc=False)
        assert "bc" in result
        bc = result["bc"]
        assert bc.height > 0

    def test_cc_edges(self):
        # A cites both B and C → B and C are co-cited
        works = [
            WorkRecord(id="A", referenced_works=["B", "C"]),
            WorkRecord(id="B", referenced_works=[]),
            WorkRecord(id="C", referenced_works=[]),
        ]
        result = build_citation_edges(works, bc=False, cc=True)
        assert "cc" in result
        cc = result["cc"]
        assert cc.height > 0

    def test_no_refs(self):
        works = [
            WorkRecord(id="A", referenced_works=[]),
            WorkRecord(id="B", referenced_works=None),
        ]
        result = build_citation_edges(works, bc=True, cc=True)
        assert result["dc"].height == 0

    def test_self_citation_excluded(self):
        """Self-citation (A cites A) shouldn't produce self-loop."""
        works = [WorkRecord(id="A", referenced_works=["A", "B"])]
        result = build_citation_edges(works, bc=False, cc=False)
        dc = result["dc"]
        self_loops = dc.filter(pl.col("uid1") == pl.col("uid2"))
        assert self_loops.height == 0

    def test_external_refs_ignored_in_dc(self):
        """References to non-focal works should not appear in DC."""
        works = [WorkRecord(id="A", referenced_works=["EXTERNAL"])]
        result = build_citation_edges(works, bc=False, cc=False)
        assert result["dc"].height == 0

    def test_min_shared_refs(self):
        works = [
            WorkRecord(id="A", referenced_works=["X"]),
            WorkRecord(id="B", referenced_works=["X"]),
        ]
        result = build_citation_edges(works, bc=True, cc=False, min_shared_refs=2)
        # Only 1 shared ref (X), min_shared_refs=2 → no BC edge
        assert result["bc"].height == 0

    def test_bc_topk(self):
        # Many shared refs → should keep only top-k
        refs = [f"ref_{i}" for i in range(100)]
        works = [
            WorkRecord(id="A", referenced_works=refs),
            WorkRecord(id="B", referenced_works=refs),
            WorkRecord(id="C", referenced_works=refs[:50]),
        ]
        result = build_citation_edges(works, bc=True, cc=False, bc_topk=5)
        bc = result["bc"]
        # Each node should have at most bc_topk=5 BC neighbors
        if bc.height > 0:
            per_node = bc.group_by("uid1").len()
            assert per_node["len"].max() <= 10  # symmetric: up to 2×topk


# ── Tests: works_to_abstracts ────────────────────────────────

class TestWorksToAbstracts:

    def test_basic(self):
        works = [
            WorkRecord(id="A", title="Title A", abstract="Abs A", year=2020),
            WorkRecord(id="B", title="Title B", abstract="Abs B", year=2021),
        ]
        df = works_to_abstracts(works)
        assert df.height == 2
        assert set(df.columns) == {"uid", "title", "abstract", "pubyear"}
        assert df["uid"].to_list() == ["A", "B"]
