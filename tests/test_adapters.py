"""Tests for sciscape.adapters (WoS, Scopus, OpenAlex input converters)."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pandas as pd
import pytest

from sciscape.adapters.bibtex import read_bibtex
from sciscape.adapters.wos import read_wos
from sciscape.adapters.scopus import read_scopus
from sciscape.adapters.openalex import read_openalex, _reconstruct_abstract


# ---------------------------------------------------------------------------
# Fixtures — small in-memory export files
# ---------------------------------------------------------------------------

@pytest.fixture()
def wos_txt(tmp_path: Path) -> Path:
    """Create a minimal WoS tab-delimited file."""
    content = "UT\tTI\tAB\tPY\n"
    content += "WOS:001\tNeural Networks\tWe study deep learning models.\t2023\n"
    content += "WOS:002\tQuantum Dots\tSynthesis of quantum dots for LEDs.\t2024\n"
    content += "WOS:003\tNo Abstract\t\t2024\n"
    p = tmp_path / "savedrecs.txt"
    p.write_text(content, encoding="utf-8")
    return p


@pytest.fixture()
def scopus_csv(tmp_path: Path) -> Path:
    """Create a minimal Scopus CSV file."""
    content = textwrap.dedent("""\
        EID,Title,Abstract,Year
        2-s2.0-001,Deep Learning,"A study of neural nets.",2023
        2-s2.0-002,Polymer Science,"[No abstract available]",2022
        2-s2.0-003,Catalysis,"Efficient catalytic reactions in water.",2024
    """)
    p = tmp_path / "scopus.csv"
    p.write_text(content, encoding="utf-8")
    return p


@pytest.fixture()
def openalex_jsonl(tmp_path: Path) -> Path:
    """Create a minimal OpenAlex JSONL file."""
    records = [
        {
            "id": "W001",
            "title": "Graph Theory",
            "abstract_inverted_index": {"We": [0], "study": [1], "graph": [2], "theory": [3]},
            "publication_year": "2023",
        },
        {
            "id": "W002",
            "title": "Empty Work",
            "abstract_inverted_index": {},
            "publication_year": "2024",
        },
        {
            "id": "W003",
            "title": "Photonics",
            "abstract_inverted_index": {"Advances": [0], "in": [1], "photonics": [2]},
            "publication_year": "2022",
        },
    ]
    p = tmp_path / "works.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    return p


@pytest.fixture()
def openalex_csv(tmp_path: Path) -> Path:
    """Create a minimal OpenAlex CSV with plain abstract column."""
    content = textwrap.dedent("""\
        id,title,abstract,publication_year
        W001,Graph Theory,We study graph theory,2023
        W002,Empty Work,,2024
        W003,Photonics,Advances in photonics,2022
    """)
    p = tmp_path / "works.csv"
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# WoS adapter
# ---------------------------------------------------------------------------

class TestReadWos:
    def test_basic(self, wos_txt):
        df = read_wos(wos_txt)
        assert set(df.columns) == {"uid", "title", "abstract", "pubyear"}
        assert len(df) == 2  # row without abstract dropped

    def test_keep_no_abstract(self, wos_txt):
        df = read_wos(wos_txt, drop_no_abstract=False)
        assert len(df) == 3

    def test_uid_values(self, wos_txt):
        df = read_wos(wos_txt)
        assert "WOS:001" in df["uid"].values
        assert "WOS:002" in df["uid"].values

    def test_pubyear_type(self, wos_txt):
        df = read_wos(wos_txt)
        assert df["pubyear"].dtype.name == "Int64"

    def test_missing_column_raises(self, tmp_path):
        p = tmp_path / "bad.txt"
        p.write_text("A\tB\n1\t2\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Missing required columns"):
            read_wos(p)

    def test_dedup(self, tmp_path):
        content = "UT\tTI\tAB\tPY\nID1\tA\tText1\t2023\nID1\tB\tText2\t2024\n"
        p = tmp_path / "dup.txt"
        p.write_text(content, encoding="utf-8")
        df = read_wos(p)
        assert len(df) == 1


# ---------------------------------------------------------------------------
# Scopus adapter
# ---------------------------------------------------------------------------

class TestReadScopus:
    def test_basic(self, scopus_csv):
        df = read_scopus(scopus_csv)
        assert len(df) == 2  # "[No abstract available]" and no-abstract dropped

    def test_no_abstract_placeholder_removed(self, scopus_csv):
        df = read_scopus(scopus_csv, drop_no_abstract=False)
        row = df[df["uid"] == "2-s2.0-002"].iloc[0]
        assert row["abstract"] == ""

    def test_columns(self, scopus_csv):
        df = read_scopus(scopus_csv)
        assert set(df.columns) == {"uid", "title", "abstract", "pubyear"}


# ---------------------------------------------------------------------------
# OpenAlex adapter
# ---------------------------------------------------------------------------

class TestReconstructAbstract:
    def test_basic(self):
        idx = {"Hello": [0], "world": [1]}
        assert _reconstruct_abstract(idx) == "Hello world"

    def test_json_string(self):
        idx = json.dumps({"A": [0], "B": [1]})
        assert _reconstruct_abstract(idx) == "A B"

    def test_empty(self):
        assert _reconstruct_abstract({}) == ""
        assert _reconstruct_abstract(None) == ""
        assert _reconstruct_abstract("") == ""

    def test_gap_positions(self):
        idx = {"first": [0], "third": [2]}
        result = _reconstruct_abstract(idx)
        assert "first" in result
        assert "third" in result

    def test_multiple_positions(self):
        idx = {"the": [0, 3], "cat": [1], "sat": [2], "mat": [4]}
        result = _reconstruct_abstract(idx)
        assert result == "the cat sat the mat"


class TestReadOpenalex:
    def test_jsonl(self, openalex_jsonl):
        df = read_openalex(openalex_jsonl)
        assert len(df) == 2  # empty inverted index dropped

    def test_csv_plain_abstract(self, openalex_csv):
        df = read_openalex(openalex_csv, abstract_col="abstract")
        assert len(df) == 2  # empty abstract dropped
        assert "We study graph theory" in df["abstract"].values

    def test_columns(self, openalex_jsonl):
        df = read_openalex(openalex_jsonl)
        assert set(df.columns) == {"uid", "title", "abstract", "pubyear"}

    def test_abstract_reconstruction(self, openalex_jsonl):
        df = read_openalex(openalex_jsonl)
        row = df[df["uid"] == "W001"].iloc[0]
        assert row["abstract"] == "We study graph theory"

    def test_parquet(self, tmp_path):
        data = pd.DataFrame({
            "id": ["W1", "W2"],
            "title": ["T1", "T2"],
            "abstract_inverted_index": [
                json.dumps({"hello": [0], "world": [1]}),
                json.dumps({"foo": [0]}),
            ],
            "publication_year": [2023, 2024],
        })
        p = tmp_path / "works.parquet"
        data.to_parquet(p)
        df = read_openalex(p)
        assert len(df) == 2

    def test_unsupported_ext(self, tmp_path):
        p = tmp_path / "works.xlsx"
        p.write_text("x")
        with pytest.raises(ValueError, match="Unsupported file extension"):
            read_openalex(p)


# ---------------------------------------------------------------------------
# CLI convert subcommand
# ---------------------------------------------------------------------------

class TestConvertCLI:
    def test_wos_convert(self, wos_txt, tmp_path):
        from sciscape.cli import main
        out = tmp_path / "out.parquet"
        main(["convert", "wos", str(wos_txt), "-o", str(out)])
        df = pd.read_parquet(out)
        assert len(df) == 2

    def test_scopus_convert(self, scopus_csv, tmp_path):
        from sciscape.cli import main
        out = tmp_path / "out.parquet"
        main(["convert", "scopus", str(scopus_csv), "-o", str(out)])
        df = pd.read_parquet(out)
        assert len(df) == 2

    def test_openalex_convert(self, openalex_jsonl, tmp_path):
        from sciscape.cli import main
        out = tmp_path / "out.parquet"
        main(["convert", "openalex", str(openalex_jsonl), "-o", str(out)])
        df = pd.read_parquet(out)
        assert len(df) == 2

    def test_bibtex_convert(self, tmp_path):
        from sciscape.cli import main
        bib = tmp_path / "refs.bib"
        bib.write_text(textwrap.dedent("""\
            @article{smith2023,
              title = {Deep Learning},
              abstract = {A survey of deep learning methods.},
              year = {2023},
            }
        """))
        out = tmp_path / "out.parquet"
        main(["convert", "bibtex", str(bib), "-o", str(out)])
        df = pd.read_parquet(out)
        assert len(df) == 1


# ---------------------------------------------------------------------------
# BibTeX adapter
# ---------------------------------------------------------------------------

@pytest.fixture()
def bib_file(tmp_path: Path) -> Path:
    content = textwrap.dedent("""\
        @article{smith2023neural,
          author = {Smith, John and Doe, Jane},
          title = {Neural Network Architectures},
          abstract = {We propose a novel architecture for deep learning.},
          year = {2023},
          journal = {Nature ML},
        }

        @inproceedings{lee2024quantum,
          author = {Lee, Alice},
          title = {Quantum Computing Survey},
          abstract = {A comprehensive review of quantum algorithms.},
          year = {2024},
          booktitle = {ICQC},
        }

        @article{no_abstract,
          title = {Missing Abstract Paper},
          year = {2022},
        }
    """)
    p = tmp_path / "refs.bib"
    p.write_text(content, encoding="utf-8")
    return p


class TestReadBibtex:
    def test_basic(self, bib_file):
        df = read_bibtex(bib_file)
        assert set(df.columns) == {"uid", "title", "abstract", "pubyear"}
        assert len(df) == 2  # no_abstract dropped

    def test_keep_no_abstract(self, bib_file):
        df = read_bibtex(bib_file, drop_no_abstract=False)
        assert len(df) == 3

    def test_uid_is_cite_key(self, bib_file):
        df = read_bibtex(bib_file)
        assert "smith2023neural" in df["uid"].values
        assert "lee2024quantum" in df["uid"].values

    def test_pubyear_type(self, bib_file):
        df = read_bibtex(bib_file)
        assert df["pubyear"].dtype.name == "Int64"

    def test_nested_braces(self, tmp_path):
        content = textwrap.dedent("""\
            @article{test1,
              title = {A {B}rief {H}istory},
              abstract = {Text with {nested} braces inside.},
              year = {2020},
            }
        """)
        p = tmp_path / "nested.bib"
        p.write_text(content)
        df = read_bibtex(p)
        assert len(df) == 1
        assert "nested" in df.iloc[0]["abstract"]

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.bib"
        p.write_text("")
        df = read_bibtex(p)
        assert len(df) == 0
