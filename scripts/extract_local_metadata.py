#!/usr/bin/env python3
"""Extract title/abstract from LOCAL OpenAlex parquet export for target work_ids.

Uses the pre-exported `openalex_works_text` parquet dataset
(479M rows, 4 columns: work_id, title, abstract, has_abstract).

Usage:
    python scripts/extract_local_metadata.py --field 34
    python scripts/extract_local_metadata.py --fields 34 30 29 12
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import polars as pl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("extract")

DATA_ROOT = Path.home() / "Desktop/HDD/local_map_analysis_data/processed/outputs/data"
NODE_DIR = DATA_ROOT / "oa26_gcc_only_k30"
TEXT_DIR = (
    Path.home()
    / "Desktop/Disk/Raid/dumps/OpenAlex/direct_exports"
    / "openalex_works_text_20260311_173530/parquet"
)
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "openalex_metadata"


def extract_field(field_id: int, text_lf: pl.LazyFrame) -> None:
    """Extract title/abstract for one field's work_ids."""
    node_path = NODE_DIR / f"field_{field_id}_nodes_oa_meta_gcc_k30.parquet"
    if not node_path.exists():
        log.warning("No node file for field %d, skipping", field_id)
        return

    nodes = pl.read_parquet(node_path, columns=["work_id"])
    target_ids = nodes["work_id"].unique().sort().to_list()
    log.info("Field %d: %d unique work_ids", field_id, len(target_ids))

    t0 = time.time()
    text_df = (
        text_lf.filter(pl.col("work_id").is_in(pl.Series(target_ids)))
        .collect()
    )
    elapsed = time.time() - t0

    # Stats
    n_total = len(text_df)
    n_title = text_df.filter(
        pl.col("title").is_not_null() & (pl.col("title") != "")
    ).height
    n_abstract = text_df.filter(
        pl.col("abstract").is_not_null() & (pl.col("abstract") != "")
    ).height
    n_missing = len(target_ids) - n_total

    log.info(
        "  Found %d/%d (%.1f%%) in %.1fs",
        n_total, len(target_ids), 100 * n_total / len(target_ids), elapsed,
    )
    log.info("    With title:    %d (%.1f%%)", n_title, 100 * n_title / max(n_total, 1))
    log.info("    With abstract: %d (%.1f%%)", n_abstract, 100 * n_abstract / max(n_total, 1))
    if n_missing > 0:
        log.info("    Missing IDs:   %d", n_missing)

    # Save
    out_dir = OUT_DIR / f"field_{field_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "works_text.parquet"
    text_df.write_parquet(out_path)
    log.info("  Saved → %s", out_path)


def main():
    parser = argparse.ArgumentParser(
        description="Extract title/abstract from local OpenAlex export"
    )
    parser.add_argument(
        "--fields",
        type=int,
        nargs="+",
        default=[34, 30, 29, 12],
        help="Field IDs to extract",
    )
    args = parser.parse_args()

    log.info("Source: %s", TEXT_DIR)
    log.info("Fields: %s", args.fields)

    # Lazy scan the full text export (4793 parquet files)
    text_lf = pl.scan_parquet(TEXT_DIR / "*.parquet")

    for field_id in args.fields:
        extract_field(field_id, text_lf)

    log.info("\nDone!")


if __name__ == "__main__":
    main()
