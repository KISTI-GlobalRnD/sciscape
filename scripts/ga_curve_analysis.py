#!/usr/bin/env python3
"""
Resolution-neutral Granularity–Accuracy (GA) curve analysis.

For each field × link_type, plots BM25 (accuracy) vs n_clusters (granularity).
Compares link types at *matched cluster counts* via log-linear interpolation,
then computes AUC over a common resolution range.

Follows Waltman et al. (2020) §3 — evaluate clustering quality as a
*function* of resolution, not at a single γ.
"""

import argparse
import polars as pl
import numpy as np
from scipy import interpolate
from pathlib import Path
import json

# ── Configuration ──────────────────────────────────────────────────────
FIELD_IDS = [12, 15, 18, 26, 29, 30, 34]
FIELD_NAMES = {
    12: "Environmental Sci.",
    15: "Earth Sciences",
    18: "Decision Sciences",
    26: "Mathematics",
    29: "Biochemistry",
    30: "Chemical Eng.",
    34: "Engineering",
}
KEY_TYPES = ["bc_assoc_strength", "cc_assoc_strength", "dc_fractional", "extdc_waltman"]
SHORT = {"bc_assoc_strength": "BC", "cc_assoc_strength": "CC",
         "dc_fractional": "DC", "extdc_waltman": "ExtDC"}


def load_ga_data(field_id: int, data_dir: str = "workspace/data/eval_results") -> dict:
    """Load (n_clusters, bm25) pairs for each key link type."""
    path = Path(data_dir) / f"field_{field_id}" / "text_quality_results.parquet"
    df = pl.read_parquet(path)
    result = {}
    for lt in KEY_TYPES:
        sub = df.filter(pl.col("link_type") == lt).sort("n_clusters")
        if sub.height == 0:
            continue
        ncl = sub["n_clusters"].to_numpy().astype(float)
        bm25 = sub["bm25_unweighted"].to_numpy().astype(float)
        # Remove any NaN/zero entries
        mask = np.isfinite(bm25) & (ncl > 0)
        result[lt] = (ncl[mask], bm25[mask])
    return result


def interpolate_bm25(ncl: np.ndarray, bm25: np.ndarray,
                     grid: np.ndarray) -> np.ndarray:
    """Log-linear interpolation of BM25 over cluster count grid."""
    log_ncl = np.log10(ncl)
    log_grid = np.log10(grid)
    # Clamp grid to data range
    valid = (log_grid >= log_ncl.min()) & (log_grid <= log_ncl.max())
    out = np.full_like(grid, np.nan, dtype=float)
    if valid.sum() > 0:
        f = interpolate.interp1d(log_ncl, bm25, kind="linear",
                                 bounds_error=False, fill_value=np.nan)
        out[valid] = f(log_grid[valid])
    return out


def compute_auc(grid: np.ndarray, values: np.ndarray) -> float:
    """AUC in log10(n_clusters) space over non-NaN region."""
    mask = np.isfinite(values)
    if mask.sum() < 2:
        return np.nan
    log_g = np.log10(grid[mask])
    v = values[mask]
    return float(np.trapz(v, log_g) / (log_g[-1] - log_g[0]))


def find_common_range(ga_data: dict) -> tuple[float, float]:
    """Find cluster count range covered by ALL link types."""
    lo = max(ncl.min() for ncl, _ in ga_data.values())
    hi = min(ncl.max() for ncl, _ in ga_data.values())
    return lo, hi


def analyze_field(field_id: int, grid_points: int = 200) -> dict:
    """Full GA analysis for one field."""
    ga_data = load_ga_data(field_id)
    if not ga_data:
        return {}

    lo, hi = find_common_range(ga_data)
    if lo >= hi:
        return {"field": field_id, "error": "no overlapping range"}

    grid = np.logspace(np.log10(lo), np.log10(hi), grid_points)

    results = {"field": field_id, "field_name": FIELD_NAMES.get(field_id, ""),
               "range": [float(lo), float(hi)]}
    auc_scores = {}
    interp_curves = {}

    for lt in KEY_TYPES:
        if lt not in ga_data:
            continue
        ncl, bm25 = ga_data[lt]
        interp = interpolate_bm25(ncl, bm25, grid)
        auc = compute_auc(grid, interp)
        auc_scores[lt] = auc
        interp_curves[lt] = interp

    results["auc"] = auc_scores

    # Pairwise comparison at matched cluster counts
    comparisons = {}
    for i, lt_a in enumerate(KEY_TYPES):
        for lt_b in KEY_TYPES[i+1:]:
            if lt_a not in interp_curves or lt_b not in interp_curves:
                continue
            va = interp_curves[lt_a]
            vb = interp_curves[lt_b]
            both_valid = np.isfinite(va) & np.isfinite(vb)
            n_valid = both_valid.sum()
            if n_valid == 0:
                continue
            a_wins = (va[both_valid] > vb[both_valid]).sum()
            b_wins = (vb[both_valid] > va[both_valid]).sum()
            comparisons[f"{SHORT[lt_a]}_vs_{SHORT[lt_b]}"] = {
                "n_points": int(n_valid),
                f"{SHORT[lt_a]}_wins": int(a_wins),
                f"{SHORT[lt_b]}_wins": int(b_wins),
            }
    results["pairwise"] = comparisons

    # Curve shape: find where each link type dominates
    # Split into low-res (< median) and high-res (> median)
    mid = len(grid) // 2
    for lt in KEY_TYPES:
        if lt not in interp_curves:
            continue
        v = interp_curves[lt]
        results.setdefault("auc_split", {})[lt] = {
            "low_res": compute_auc(grid[:mid], v[:mid]),
            "high_res": compute_auc(grid[mid:], v[mid:]),
        }

    return results


def print_results(all_results: list[dict]):
    """Pretty-print the GA analysis results."""
    print("=" * 80)
    print("RESOLUTION-NEUTRAL GA CURVE ANALYSIS (BM25 vs n_clusters)")
    print("Following Waltman et al. (2020) — compare curves, not single γ points")
    print("=" * 80)

    # Per-field AUC table
    print("\n── AUC Scores (higher = better, averaged over common cluster range) ──\n")
    header = f"{'Field':>25s}"
    for lt in KEY_TYPES:
        header += f"  {SHORT[lt]:>8s}"
    header += "   Range (clusters)"
    print(header)
    print("-" * len(header))

    auc_by_type = {lt: [] for lt in KEY_TYPES}
    for r in all_results:
        if "error" in r:
            continue
        fid = r["field"]
        lo, hi = r["range"]
        row = f"{FIELD_NAMES.get(fid, fid):>25s}"
        aucs = r.get("auc", {})

        # Find winner
        best_lt = max(aucs, key=aucs.get) if aucs else None

        for lt in KEY_TYPES:
            auc = aucs.get(lt, float("nan"))
            marker = " *" if lt == best_lt else "  "
            row += f"  {auc:7.1f}{marker}"
            if np.isfinite(auc):
                auc_by_type[lt].append(auc)
        row += f"   {int(lo):>5d} – {int(hi):>5d}"
        print(row)

    # Average AUC
    print("-" * len(header))
    row = f"{'AVERAGE':>25s}"
    avg_aucs = {}
    for lt in KEY_TYPES:
        vals = auc_by_type[lt]
        avg = np.mean(vals) if vals else float("nan")
        avg_aucs[lt] = avg
        row += f"  {avg:7.1f}  "
    print(row)

    best_avg = max(avg_aucs, key=avg_aucs.get) if avg_aucs else None
    print(f"\n  * = field winner; Overall AUC winner: {SHORT.get(best_avg, '?')}")

    # AUC split: low-res vs high-res
    print("\n── AUC Split: Low-Resolution vs High-Resolution ──\n")
    for label, key in [("Low-Res (coarse)", "low_res"), ("High-Res (fine)", "high_res")]:
        print(f"  {label}:")
        wins = {lt: 0 for lt in KEY_TYPES}
        for r in all_results:
            if "error" in r or "auc_split" not in r:
                continue
            best = None
            best_v = -1
            for lt in KEY_TYPES:
                v = r["auc_split"].get(lt, {}).get(key, float("nan"))
                if np.isfinite(v) and v > best_v:
                    best_v = v
                    best = lt
            if best:
                wins[best] += 1
        for lt in KEY_TYPES:
            print(f"    {SHORT[lt]:>8s}: {wins[lt]} field wins")
        print()

    # Pairwise head-to-head at matched cluster counts
    print("── Pairwise Head-to-Head (at matched cluster counts) ──\n")
    pairs = ["BC_vs_CC", "BC_vs_DC", "BC_vs_ExtDC", "CC_vs_DC", "CC_vs_ExtDC", "DC_vs_ExtDC"]
    for pair in pairs:
        a_label, b_label = pair.split("_vs_")
        total_a = 0
        total_b = 0
        for r in all_results:
            if "error" in r:
                continue
            p = r.get("pairwise", {}).get(pair, {})
            total_a += p.get(f"{a_label}_wins", 0)
            total_b += p.get(f"{b_label}_wins", 0)
        total = total_a + total_b
        if total > 0:
            pct_a = total_a / total * 100
            print(f"  {a_label:>5s} vs {b_label:<5s}: {a_label} wins {total_a}/{total} ({pct_a:.0f}%), "
                  f"{b_label} wins {total_b}/{total} ({100-pct_a:.0f}%)")
    print()

    # Raw GA data for visual inspection
    print("── Raw GA Points (n_clusters → BM25) ──\n")
    for r in all_results:
        if "error" in r:
            continue
        fid = r["field"]
        ga_data = load_ga_data(fid)
        print(f"  Field {fid} ({FIELD_NAMES.get(fid, '')}):")
        for lt in KEY_TYPES:
            if lt not in ga_data:
                continue
            ncl, bm25 = ga_data[lt]
            pts = "  ".join(f"({int(n):>5d}, {b:5.1f})" for n, b in zip(ncl, bm25))
            print(f"    {SHORT[lt]:>5s}: {pts}")
        print()


def main():
    parser = argparse.ArgumentParser(description="GA curve analysis")
    parser.add_argument("--fields", type=int, nargs="+", default=FIELD_IDS)
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    all_results = []
    for fid in args.fields:
        r = analyze_field(fid)
        all_results.append(r)

    if args.json:
        # Convert numpy types for JSON serialization
        print(json.dumps(all_results, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x)))
    else:
        print_results(all_results)


if __name__ == "__main__":
    main()
