"""Web of Science adapter.

Reads WoS tab-delimited text exports (*.txt) or CSV exports and converts
them to the SciScape abstract parquet schema.

WoS field tags used:
    UT  — unique identifier (Accession Number)
    TI  — title
    AB  — abstract
    PY  — publication year
    AU  — authors (optional)
    DE  — author keywords (optional)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import pandas as pd


def read_wos(
    path: Union[str, Path],
    *,
    uid_col: str = "UT",
    title_col: str = "TI",
    abstract_col: str = "AB",
    year_col: str = "PY",
    sep: Optional[str] = None,
    encoding: str = "utf-8-sig",
    drop_no_abstract: bool = True,
) -> pd.DataFrame:
    """Read a Web of Science export and return a SciScape-compatible DataFrame.

    Parameters
    ----------
    path : str or Path
        Path to the WoS export file (.txt tab-delimited or .csv).
    uid_col, title_col, abstract_col, year_col : str
        Column names in the WoS file.
    sep : str, optional
        Delimiter. Auto-detected from extension if None (tab for .txt, comma
        for .csv).
    encoding : str
        File encoding. WoS exports often use UTF-8 with BOM.
    drop_no_abstract : bool
        If True, drop rows without abstracts.

    Returns
    -------
    pd.DataFrame
        Columns: ``uid``, ``title``, ``abstract``, ``pubyear``.
    """
    path = Path(path)
    if sep is None:
        sep = "\t" if path.suffix.lower() == ".txt" else ","

    raw = pd.read_csv(path, sep=sep, encoding=encoding, dtype=str, on_bad_lines="skip")

    # Validate required columns
    missing = [c for c in [uid_col, abstract_col, year_col] if c not in raw.columns]
    if missing:
        raise ValueError(
            f"Missing required columns {missing} in WoS file. "
            f"Available: {list(raw.columns)[:20]}"
        )

    df = pd.DataFrame({
        "uid": raw[uid_col].astype(str).str.strip(),
        "title": raw[title_col].fillna("").astype(str).str.strip() if title_col in raw.columns else "",
        "abstract": raw[abstract_col].fillna("").astype(str).str.strip(),
        "pubyear": pd.to_numeric(raw[year_col], errors="coerce"),
    })

    if drop_no_abstract:
        df = df[df["abstract"].str.len() > 0].copy()

    df["pubyear"] = df["pubyear"].astype("Int64")
    df = df.drop_duplicates(subset=["uid"], keep="first").reset_index(drop=True)

    return df
