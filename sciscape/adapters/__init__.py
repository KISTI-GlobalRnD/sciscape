"""Input adapters for converting external bibliometric data to SciScape format.

Supported sources:
    wos         Web of Science tab-delimited / CSV exports
    scopus      Scopus CSV exports
    openalex    OpenAlex JSON / CSV (API or snapshot)
    bibtex      BibTeX (.bib) files

Each adapter produces a standardised abstract parquet with columns:
    uid, title, abstract, pubyear
"""

from .bibtex import read_bibtex
from .openalex import read_openalex
from .scopus import read_scopus
from .wos import read_wos

__all__ = [
    "read_bibtex",
    "read_openalex",
    "read_scopus",
    "read_wos",
]
