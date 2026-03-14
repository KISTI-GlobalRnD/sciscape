"""BibTeX adapter.

Reads BibTeX (.bib) files and converts to the SciScape abstract parquet
schema. Handles standard BibTeX fields: author, title, abstract, year.

Requires no external BibTeX parser — uses a lightweight regex-based parser
sufficient for standard exports from Google Scholar, DBLP, IEEE, etc.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Union

import pandas as pd


def _parse_bib(text: str) -> list[dict[str, str]]:
    """Parse BibTeX entries into a list of field dicts."""
    entries: list[dict[str, str]] = []

    # Match entry blocks: @type{key, ... }
    entry_pattern = re.compile(
        r"@\w+\s*\{([^,]*),\s*(.*?)\n\}", re.DOTALL
    )

    for match in entry_pattern.finditer(text):
        cite_key = match.group(1).strip()
        body = match.group(2)

        fields: dict[str, str] = {"_key": cite_key}

        # Match field = {value} or field = "value" or field = number
        field_pattern = re.compile(
            r"(\w+)\s*=\s*(?:\{((?:[^{}]|\{[^{}]*\})*)\}|\"([^\"]*)\"|(\d+))",
            re.DOTALL,
        )
        for fm in field_pattern.finditer(body):
            name = fm.group(1).lower()
            value = fm.group(2) or fm.group(3) or fm.group(4) or ""
            # Clean up whitespace and line breaks
            value = re.sub(r"\s+", " ", value).strip()
            fields[name] = value

        entries.append(fields)

    return entries


def read_bibtex(
    path: Union[str, Path],
    *,
    uid_field: str = "_key",
    title_field: str = "title",
    abstract_field: str = "abstract",
    year_field: str = "year",
    encoding: str = "utf-8",
    drop_no_abstract: bool = True,
) -> pd.DataFrame:
    """Read a BibTeX file and return a SciScape-compatible DataFrame.

    Parameters
    ----------
    path : str or Path
        Path to the .bib file.
    uid_field : str
        BibTeX field to use as unique identifier (default: citation key).
    title_field, abstract_field, year_field : str
        BibTeX field names.
    encoding : str
        File encoding.
    drop_no_abstract : bool
        If True, drop entries without abstracts.

    Returns
    -------
    pd.DataFrame
        Columns: ``uid``, ``title``, ``abstract``, ``pubyear``.
    """
    path = Path(path)
    text = path.read_text(encoding=encoding)
    entries = _parse_bib(text)

    if not entries:
        return pd.DataFrame(columns=["uid", "title", "abstract", "pubyear"])

    rows = []
    for entry in entries:
        rows.append({
            "uid": entry.get(uid_field, "").strip(),
            "title": entry.get(title_field, "").strip(),
            "abstract": entry.get(abstract_field, "").strip(),
            "pubyear": entry.get(year_field, ""),
        })

    df = pd.DataFrame(rows)
    df["pubyear"] = pd.to_numeric(df["pubyear"], errors="coerce").astype("Int64")

    if drop_no_abstract:
        df = df[df["abstract"].str.len() > 0].copy()

    df = df[df["uid"].str.len() > 0].copy()
    df = df.drop_duplicates(subset=["uid"], keep="first").reset_index(drop=True)

    return df
