"""Input adapters for converting external bibliometric data to SciScape format.

Supported sources:
    wos         Web of Science tab-delimited / CSV exports
    scopus      Scopus CSV exports
    openalex    OpenAlex JSON / CSV (API or snapshot)

Each adapter produces a standardised abstract parquet with columns:
    uid, title, abstract, pubyear
"""

from .wos import read_wos
from .scopus import read_scopus
from .openalex import read_openalex

__all__ = [
    "read_wos",
    "read_scopus",
    "read_openalex",
]
