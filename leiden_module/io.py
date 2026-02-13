"""Input helpers for reading KRISS pair link edge lists."""

from __future__ import annotations

from pathlib import Path
import zipfile

import polars as pl
from polars.exceptions import ComputeError


def load_edge_table(
    zip_path: Path,
    inner_name: str,
    *,
    separator: str = "\t",
    streaming: bool = True,
) -> pl.DataFrame:
    """Load the edge list stored inside a zip archive."""

    with zipfile.ZipFile(zip_path) as archive, archive.open(inner_name) as handle:
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


__all__ = ["load_edge_table"]
