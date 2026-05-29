#!/usr/bin/env python3
"""Build individual DC/BC/CC edge tables from oa26_gcc_only (NO k-core).

For each field, loads citation edges + GCC node set, then uses
sciscape.linkage.builders to compute DC/BC/CC with all normalizations.

Outputs go to workspace/data/linktype_edges_gcc/field_{id}/ with one parquet per
link-type variant (e.g., bc_assoc_strength.parquet).

Usage:
    .venv/bin/python scripts/build_linktype_edges_gcc.py --field-ids 12 15
    .venv/bin/python scripts/build_linktype_edges_gcc.py --field-ids 15 --bc-topk 300
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import polars as pl

from sciscape.linkage.builders import build_bc, build_cc, build_dc
from sciscape.linkage.config import LinkageConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("build_linktype_gcc")

# ── Paths ──────────────────────────────────────────────────────────
HDD = Path.home() / "Desktop/HDD/local_map_analysis_data/processed/outputs/data"
GCC_DIR = HDD / "oa26_gcc_only"
CIT_DIR = HDD / "oa26_citation_edges"
OUT_ROOT = Path("workspace/data/linktype_edges_gcc")


def load_gcc_node_ids(field_id: int) -> set[str]:
    """Load unique work_ids from oa26_gcc_only node file."""
    path = GCC_DIR / f"field_{field_id}_nodes_oa_meta_gcc.parquet"
    df = pl.read_parquet(path, columns=["work_id"])
    ids = set(df["work_id"].unique().to_list())
    log.info("Field %d: %d unique GCC nodes", field_id, len(ids))
    return ids


def load_citations(field_id: int) -> pl.DataFrame:
    """Load raw citation edges for a field."""
    path = CIT_DIR / f"oa26_citation_edges_field_{field_id}.parquet"
    df = pl.read_parquet(path, columns=[
        "citing_work_id", "cited_work_id", "cited_in_set",
    ])
    log.info("Field %d: %d raw citation edges", field_id, df.height)
    return df


def process_field(field_id: int, cfg: LinkageConfig) -> None:
    """Build and save DC/BC/CC edge tables for one field."""
    out_dir = OUT_ROOT / f"field_{field_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    node_ids = load_gcc_node_ids(field_id)
    citations = load_citations(field_id)

    # ── DC ──
    t0 = time.time()
    log.info("=== Field %d: Building DC ===", field_id)
    dc_result = build_dc(citations, node_ids, config=cfg)
    for name, df in dc_result.items():
        out = out_dir / f"{name}.parquet"
        df.write_parquet(out)
        log.info("  %s: %d edges → %s", name, df.height, out)
    log.info("DC done in %.1fs", time.time() - t0)

    # ── BC ──
    t0 = time.time()
    log.info("=== Field %d: Building BC ===", field_id)
    bc_result = build_bc(citations, node_ids, config=cfg)
    for name, df in bc_result.items():
        out = out_dir / f"{name}.parquet"
        df.write_parquet(out)
        log.info("  %s: %d edges → %s", name, df.height, out)
    log.info("BC done in %.1fs", time.time() - t0)

    # ── CC ──
    t0 = time.time()
    log.info("=== Field %d: Building CC ===", field_id)
    cc_result = build_cc(citations, node_ids, config=cfg)
    for name, df in cc_result.items():
        out = out_dir / f"{name}.parquet"
        df.write_parquet(out)
        log.info("  %s: %d edges → %s", name, df.height, out)
    log.info("CC done in %.1fs", time.time() - t0)

    # ── Save node mapping ──
    categories = pl.Series("work_id", sorted(node_ids))
    node_map = pl.DataFrame({
        "idx": list(range(len(categories))),
        "work_id": categories,
    })
    node_map.write_parquet(out_dir / "node_mapping.parquet")
    log.info("Node mapping: %d nodes → %s", len(node_ids), out_dir / "node_mapping.parquet")


def main() -> None:
    p = argparse.ArgumentParser(description="Build DC/BC/CC from oa26_gcc_only")
    p.add_argument("--field-ids", nargs="+", type=int, required=True,
                   help="Field IDs to process (e.g., 12 15)")
    p.add_argument("--bc-min-shared", type=int, default=3)
    p.add_argument("--cc-min-shared", type=int, default=2)
    p.add_argument("--bc-topk", type=int, default=500)
    p.add_argument("--cc-topk", type=int, default=500)
    args = p.parse_args()

    cfg = LinkageConfig(
        bc_min_shared=args.bc_min_shared,
        cc_min_shared=args.cc_min_shared,
        bc_topk=args.bc_topk,
        cc_topk=args.cc_topk,
    )
    log.info("Config: bc_min_shared=%d, cc_min_shared=%d, bc_topk=%s, cc_topk=%s",
             cfg.bc_min_shared, cfg.cc_min_shared, cfg.bc_topk, cfg.cc_topk)

    for fid in args.field_ids:
        t_field = time.time()
        log.info("=" * 60)
        log.info("Processing field %d", fid)
        log.info("=" * 60)
        process_field(fid, cfg)
        log.info("Field %d total: %.1fs", fid, time.time() - t_field)
        log.info("")


if __name__ == "__main__":
    main()
