"""Input helpers for reading edge lists (zip, CSV, Parquet)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union
import zipfile

import polars as pl
from polars.exceptions import ComputeError


def load_edge_table(
    zip_path: Union[str, Path],
    inner_name: Optional[str] = None,
    *,
    separator: str = "\t",
    streaming: bool = True,
) -> pl.DataFrame:
    """Load an edge list from zip archive, CSV, or Parquet.

    Parameters
    ----------
    zip_path : str or Path
        Path to the edge file. Supported formats:
        - ``.zip``: zip archive containing a CSV/TSV (requires ``inner_name``)
        - ``.parquet``: Parquet file with uid1, uid2, rel_sum2 columns
        - ``.csv`` / ``.tsv`` / ``.txt``: Plain CSV/TSV file
    inner_name : str, optional
        Filename inside the zip archive. Required for .zip files.
    separator : str
        Column delimiter for CSV/TSV (default: tab).
    streaming : bool
        Whether to use Polars streaming mode.
    """
    path = Path(zip_path)
    ext = path.suffix.lower()

    if ext == ".parquet":
        return pl.read_parquet(path)

    if ext == ".zip":
        if not inner_name:
            raise ValueError("inner_name is required for .zip files")
        with zipfile.ZipFile(path) as archive, archive.open(inner_name) as handle:
            scan = pl.scan_csv(
                handle,
                has_header=True,
                separator=separator,
                dtypes={"uid1": pl.Utf8, "uid2": pl.Utf8, "rel_sum2": pl.Float64},
            )
            try:
                return scan.collect(streaming=streaming)
            except ComputeError as err:
                if streaming and "Streaming scanning of in-memory buffers" in str(err):
                    return scan.collect(streaming=False)
                raise

    # Plain CSV/TSV
    scan = pl.scan_csv(
        path,
        has_header=True,
        separator=separator,
        dtypes={"uid1": pl.Utf8, "uid2": pl.Utf8, "rel_sum2": pl.Float64},
    )
    try:
        return scan.collect(streaming=streaming)
    except ComputeError as err:
        if streaming and "Streaming scanning of in-memory buffers" in str(err):
            return scan.collect(streaming=False)
        raise


__all__ = ["load_edge_table"]
