#!/usr/bin/env python3
"""Extract embedding-aligned title/abstract rows from a large OpenAlex text dump.

This is a robust fallback for cases where a single lazy scan/join over the full
text export is too memory-hungry or unstable. It iterates over parquet parts and
joins each one against the target work_id set derived from:

1. the embedding mapping for a field, and
2. the GCC node mapping used by the consensus experiments.
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
log = logging.getLogger("extract_emb_aligned_text")

TEXT_DIR = (
    Path.home()
    / "Desktop/Disk/Raid/dumps/OpenAlex/direct_exports"
    / "openalex_works_text_20260311_173530/parquet"
)
EMB_DIR = (
    Path.home()
    / "Desktop/HDD/local_map_analysis_data/processed/outputs/gpu_embeddings"
    / "oa26_perfield_specter2_20260331"
)
GCC_DIR = Path(__file__).resolve().parent.parent / "data" / "linktype_edges_gcc"
OUT_DIR = Path("/tmp/aligned_openalex_text_round2")


def resolve_field_dir(field_id: int) -> Path:
    matches = sorted(EMB_DIR.glob(f"field_{field_id}_*"))
    if not matches:
        raise FileNotFoundError(f"No embedding directory found for field {field_id} under {EMB_DIR}")
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple embedding directories found for field {field_id}: "
            + ", ".join(path.name for path in matches)
        )
    return matches[0]


def load_target_ids(field_id: int) -> pl.DataFrame:
    field_dir = resolve_field_dir(field_id)
    mapping = pl.read_parquet(field_dir / "mapping.parquet", columns=["work_id"])
    gcc = pl.read_parquet(GCC_DIR / f"field_{field_id}" / "node_mapping.parquet", columns=["work_id"])
    target = mapping.join(gcc, on="work_id", how="inner").unique().sort("work_id")
    log.info("Field %d: target work_ids=%d", field_id, target.height)
    return target


def extract_field(
    field_id: int,
    *,
    text_dir: Path,
    out_root: Path,
) -> Path:
    target = load_target_ids(field_id)
    target_seen: set[str] = set()
    chunks: list[pl.DataFrame] = []
    parts = sorted(text_dir.glob("*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No parquet parts found under {text_dir}")

    t0 = time.time()
    for idx, part in enumerate(parts, start=1):
        df = pl.read_parquet(part, columns=["work_id", "title", "abstract"])
        matched = df.join(target, on="work_id", how="inner")
        if matched.height:
            chunks.append(matched)
            target_seen.update(matched["work_id"].to_list())
        if idx == 1 or idx % 250 == 0 or idx == len(parts):
            log.info(
                "  parts %d/%d, matched=%d/%d, elapsed=%.1fs",
                idx,
                len(parts),
                len(target_seen),
                target.height,
                time.time() - t0,
            )
        if len(target_seen) == target.height:
            log.info("  all target ids matched early at part %d/%d", idx, len(parts))
            break

    if chunks:
        text_df = pl.concat(chunks).unique(subset=["work_id"]).sort("work_id")
    else:
        text_df = pl.DataFrame(schema={"work_id": pl.String, "title": pl.String, "abstract": pl.String})

    out_dir = out_root / f"field_{field_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "works_text.parquet"
    text_df.write_parquet(out_path)

    log.info(
        "Field %d: saved %d/%d rows to %s in %.1fs",
        field_id,
        text_df.height,
        target.height,
        out_path,
        time.time() - t0,
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract embedding-aligned OpenAlex text rows")
    parser.add_argument("--fields", type=int, nargs="+", required=True)
    parser.add_argument("--text-dir", type=Path, default=TEXT_DIR)
    parser.add_argument("--out-root", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    log.info("Text source: %s", args.text_dir)
    log.info("Output root: %s", args.out_root)
    for field_id in args.fields:
        extract_field(field_id, text_dir=args.text_dir, out_root=args.out_root)


if __name__ == "__main__":
    main()
