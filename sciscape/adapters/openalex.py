"""OpenAlex adapter.

Reads OpenAlex data (JSON lines from API, snapshot CSV/parquet, or
flattened CSV) and converts to the SciScape abstract parquet schema.

OpenAlex fields used:
    id                      — unique identifier (e.g. "W2741809807")
    title                   — article title
    abstract_inverted_index — inverted index dict (JSON API format)
    abstract                — plain text abstract (flattened exports)
    publication_year        — publication year
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import pandas as pd


def _reconstruct_abstract(inverted_index: Union[str, dict, None]) -> str:
    """Reconstruct abstract text from OpenAlex inverted index format.

    The inverted index maps tokens to position lists:
    ``{"neural": [0, 5], "network": [1, 6], ...}``
    """
    if inverted_index is None or inverted_index == "":
        return ""

    if isinstance(inverted_index, str):
        try:
            inverted_index = json.loads(inverted_index)
        except (json.JSONDecodeError, TypeError):
            return ""

    if not isinstance(inverted_index, dict):
        return ""

    # Build position → token mapping
    position_map: dict[int, str] = {}
    for token, positions in inverted_index.items():
        if isinstance(positions, list):
            for pos in positions:
                position_map[int(pos)] = token

    if not position_map:
        return ""

    max_pos = max(position_map.keys())
    words = [position_map.get(i, "") for i in range(max_pos + 1)]
    return " ".join(w for w in words if w)


def read_openalex(
    path: Union[str, Path],
    *,
    uid_col: str = "id",
    title_col: str = "title",
    abstract_col: Optional[str] = None,
    abstract_inverted_col: str = "abstract_inverted_index",
    year_col: str = "publication_year",
    encoding: str = "utf-8",
    drop_no_abstract: bool = True,
) -> pd.DataFrame:
    """Read OpenAlex data and return a SciScape-compatible DataFrame.

    Supports three input formats:
    - ``.jsonl`` / ``.json``: JSON lines (one work per line)
    - ``.parquet``: Parquet file
    - ``.csv``: Flattened CSV

    Parameters
    ----------
    path : str or Path
        Path to the OpenAlex data file.
    uid_col : str
        Column/field name for unique identifier.
    title_col : str
        Column/field name for title.
    abstract_col : str, optional
        Column name for plain-text abstract. If present and non-empty,
        used directly instead of the inverted index.
    abstract_inverted_col : str
        Column name for the inverted index field.
    year_col : str
        Column name for publication year.
    encoding : str
        File encoding (for CSV/JSONL).
    drop_no_abstract : bool
        If True, drop rows without abstracts.

    Returns
    -------
    pd.DataFrame
        Columns: ``uid``, ``title``, ``abstract``, ``pubyear``.
    """
    path = Path(path)
    ext = path.suffix.lower()

    if ext == ".parquet":
        raw = pd.read_parquet(path)
    elif ext in (".jsonl", ".json", ".ndjson"):
        # Don't use dtype=str — inverted index must stay as dict
        raw = pd.read_json(path, lines=True, encoding=encoding)
    elif ext == ".csv":
        raw = pd.read_csv(path, encoding=encoding, dtype=str, on_bad_lines="skip")
    else:
        raise ValueError(f"Unsupported file extension: {ext}. Use .jsonl, .parquet, or .csv")

    missing = [c for c in [uid_col, year_col] if c not in raw.columns]
    if missing:
        raise ValueError(
            f"Missing required columns {missing} in OpenAlex file. "
            f"Available: {list(raw.columns)[:20]}"
        )

    # Determine abstract source
    has_plain = abstract_col and abstract_col in raw.columns
    has_inverted = abstract_inverted_col in raw.columns

    if has_plain:
        abstracts = raw[abstract_col].fillna("").astype(str).str.strip()
        # Fill missing plain abstracts from inverted index
        if has_inverted:
            mask = abstracts.str.len() == 0
            if mask.any():
                reconstructed = raw.loc[mask, abstract_inverted_col].apply(
                    _reconstruct_abstract
                )
                abstracts.loc[mask] = reconstructed
    elif has_inverted:
        abstracts = raw[abstract_inverted_col].apply(_reconstruct_abstract)
    else:
        raise ValueError(
            f"No abstract column found. Provide '{abstract_col or 'abstract'}' "
            f"or '{abstract_inverted_col}' column."
        )

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
