#!/usr/bin/env python3
"""Diagnose link-type characteristics: coverage, overlap, complementarity.

Characterizes each link type's unique information contribution
to guide combination strategy.

Usage:
    python scripts/linktype_diagnostic.py --field 34
    python scripts/linktype_diagnostic.py --field all
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import polars as pl

from sciscape.linkage.diagnostics import (
    complementarity_analysis,
    complementarity_table,
    degree_comparison,
    edge_overlap,
    edge_stats,
    overlap_matrix,
    stats_table,
    weight_correlation,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("diag")

EDGE_DIR = Path(__file__).resolve().parent.parent / "data" / "linktype_edges"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "eval_results"
FIELDS = [12, 15, 18, 26, 29, 30, 34]

# Base link types to analyze (best variant per group from prior experiments)
BASE_TYPES = {
    "dc_frac": "dc_fractional",
    "bc_cos": "bc_cosine",
    "cc_cos": "cc_cosine",
    "emb_prx": "emb_prx_knn30",
}

# Column mapping: script edge files use (src, dst, weight)
SRC, DST, W = "src", "dst", "weight"


def _load(field_id: int, name: str) -> pl.DataFrame | None:
    """Load edge file, rename to standard columns."""
    path = EDGE_DIR / f"field_{field_id}" / f"{name}.parquet"
    if not path.exists():
        log.warning("  Missing: %s", path)
        return None
    df = pl.read_parquet(path)
    return df.rename({"src": "uid1", "dst": "uid2", "weight": "rel_sum2"})


def _load_nodes(field_id: int) -> set[str]:
    """Load focal node set."""
    path = EDGE_DIR / f"field_{field_id}" / "node_mapping.parquet"
    return set(pl.read_parquet(path)["work_id"].to_list())


def diagnose_field(field_id: int) -> dict:
    """Run full diagnostic for one field."""
    log.info("=" * 70)
    log.info("FIELD %d — Link Type Diagnostic", field_id)
    log.info("=" * 70)

    node_ids = _load_nodes(field_id)
    log.info("Focal nodes: %d", len(node_ids))

    # ── Load edge sets ────────────────────────────────────────────
    edge_sets = {}
    for label, filename in BASE_TYPES.items():
        df = _load(field_id, filename)
        if df is not None:
            edge_sets[label] = df
            log.info("  %s: %d edges", label, df.height)

    if len(edge_sets) < 2:
        log.warning("  Not enough link types for comparison")
        return {}

    # ── 1. Per-type statistics ────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"1. PER-TYPE STATISTICS (Field {field_id})")
    print(f"{'─'*70}")
    stats_list = [
        edge_stats(df, name=name, node_ids=node_ids)
        for name, df in edge_sets.items()
    ]
    st = stats_table(stats_list)
    print(st)

    # ── 2. Edge overlap (Jaccard) ─────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"2. EDGE OVERLAP — Jaccard Matrix")
    print(f"{'─'*70}")
    overlaps, jaccard_df = overlap_matrix(edge_sets)
    print(jaccard_df)

    print(f"\n  Pairwise detail:")
    for o in overlaps:
        print(f"  {o.name_a} ∩ {o.name_b}: "
              f"{o.n_intersection:,} shared / {o.n_union:,} union "
              f"(J={o.jaccard:.4f}) | "
              f"{o.name_a} exclusive: {o.a_exclusive_frac:.1%}, "
              f"{o.name_b} exclusive: {o.b_exclusive_frac:.1%}")

    # ── 3. Complementarity ────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"3. COMPLEMENTARITY — Unique Contributions")
    print(f"{'─'*70}")
    comp = complementarity_analysis(edge_sets, node_ids=node_ids)
    ct = complementarity_table(comp)
    print(ct)

    # ── 4. Weight correlation on shared edges ─────────────────────
    print(f"\n{'─'*70}")
    print(f"4. WEIGHT CORRELATION (shared edges)")
    print(f"{'─'*70}")
    names = list(edge_sets.keys())
    corr_rows = []
    for i, a in enumerate(names):
        for b in names[i+1:]:
            c = weight_correlation(edge_sets[a], edge_sets[b], a, b)
            corr_rows.append(c)
            print(f"  {a} vs {b}: n_shared={c['n_shared']:,}, "
                  f"pearson={c['pearson']}, spearman={c['spearman']}")

    # ── 5. Degree comparison ──────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"5. DEGREE COMPARISON — Hub vs Peripheral Behavior")
    print(f"{'─'*70}")
    deg = degree_comparison(edge_sets)

    # Correlation between degree vectors
    from scipy.stats import spearmanr
    deg_names = [c for c in deg.columns if c.startswith("deg_")]
    for i, a in enumerate(deg_names):
        for b in deg_names[i+1:]:
            va = deg[a].fill_null(0).to_numpy()
            vb = deg[b].fill_null(0).to_numpy()
            rho, _ = spearmanr(va, vb)
            print(f"  degree({a[4:]}) vs degree({b[4:]}): "
                  f"Spearman ρ = {rho:.4f}")

    # Nodes high in one type but low in another
    for name in names:
        col = f"deg_{name}"
        others = [f"deg_{n}" for n in names if n != name]
        if not others:
            continue
        # Top-10% in this type
        threshold = deg[col].quantile(0.9)
        if threshold == 0:
            continue
        high = deg.filter(pl.col(col) >= threshold)
        for oc in others:
            other_name = oc[4:]
            low_in_other = high.filter(pl.col(oc) <= deg[oc].quantile(0.1))
            if low_in_other.height > 0:
                print(f"  {low_in_other.height} nodes are top-10% in {name} "
                      f"but bottom-10% in {other_name}")

    # ── 6. Coverage gap analysis ──────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"6. COVERAGE GAP — Uncovered Nodes")
    print(f"{'─'*70}")
    all_covered = set()
    for name, df in edge_sets.items():
        covered = set(df["uid1"].to_list()) | set(df["uid2"].to_list())
        all_covered |= covered
        uncovered = node_ids - covered
        print(f"  {name}: {len(covered):,} covered, "
              f"{len(uncovered):,} uncovered ({len(uncovered)/len(node_ids):.1%})")

    total_uncovered = node_ids - all_covered
    print(f"  ALL COMBINED: {len(total_uncovered):,} still uncovered "
          f"({len(total_uncovered)/len(node_ids):.1%})")

    # ── Save results ──────────────────────────────────────────────
    out_dir = OUT_DIR / f"field_{field_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    st.write_parquet(out_dir / "linktype_stats.parquet")
    jaccard_df.write_parquet(out_dir / "linktype_jaccard.parquet")
    ct.write_parquet(out_dir / "linktype_complementarity.parquet")
    deg.write_parquet(out_dir / "linktype_degree_comparison.parquet")

    if corr_rows:
        pl.DataFrame(corr_rows).write_parquet(out_dir / "linktype_weight_corr.parquet")

    log.info("Results saved to %s", out_dir)

    return {
        "stats": st,
        "jaccard": jaccard_df,
        "complementarity": ct,
        "degree": deg,
    }


def main():
    parser = argparse.ArgumentParser(description="Link-type diagnostic")
    parser.add_argument("--field", default="34",
                        help="Field ID or 'all'")
    args = parser.parse_args()

    fields = FIELDS if args.field == "all" else [int(args.field)]
    for fid in fields:
        diagnose_field(fid)


if __name__ == "__main__":
    main()
