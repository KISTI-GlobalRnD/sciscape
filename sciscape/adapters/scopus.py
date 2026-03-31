"""Scopus adapter.

Reads Scopus CSV exports and converts to the SciScape abstract parquet schema.

Scopus CSV columns used:
    EID         — unique identifier (e.g. "2-s2.0-85012345678")
    Title       — article title
    Abstract    — abstract text
    Year        — publication year
    Author Keywords — semicolon-separated keywords (optional)
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd


def read_scopus(
    path: Union[str, Path],
    *,
    uid_col: str = "EID",
    title_col: str = "Title",
    abstract_col: str = "Abstract",
    year_col: str = "Year",
    encoding: str = "utf-8-sig",
    drop_no_abstract: bool = True,
) -> pd.DataFrame:
    """Read a Scopus CSV export and return a SciScape-compatible DataFrame.

    Parameters
    ----------
    path : str or Path
        Path to the Scopus CSV file.
    uid_col, title_col, abstract_col, year_col : str
        Column names in the Scopus CSV.
    encoding : str
        File encoding.
    drop_no_abstract : bool
        If True, drop rows without abstracts.

    Returns
    -------
    pd.DataFrame
        Columns: ``uid``, ``title``, ``abstract``, ``pubyear``.
    """
    path = Path(path)
    raw = pd.read_csv(path, encoding=encoding, dtype=str, on_bad_lines="skip")

    missing = [c for c in [uid_col, abstract_col, year_col] if c not in raw.columns]
    if missing:
        raise ValueError(
            f"Missing required columns {missing} in Scopus file. "
            f"Available: {list(raw.columns)[:20]}"
        )

    # Scopus uses "[No abstract available]" as placeholder
    abstracts = raw[abstract_col].fillna("").astype(str).str.strip()
    abstracts = abstracts.replace(r"^\[No abstract available\]$", "", regex=True)

    df = pd.DataFrame({
        "uid": raw[uid_col].astype(str).str.strip(),
        "title": raw[title_col].fillna("").astype(str).str.strip() if title_col in raw.columns else "",
        "abstract": abstracts,
        "pubyear": pd.to_numeric(raw[year_col], errors="coerce"),
    })

    if drop_no_abstract:
        df = df[df["abstract"].str.len() > 0].copy()

    df["pubyear"] = df["pubyear"].astype("Int64")
    df = df.drop_duplicates(subset=["uid"], keep="first").reset_index(drop=True)

    return df
