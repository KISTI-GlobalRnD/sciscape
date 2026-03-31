#!/usr/bin/env python3
"""Compare hub instability across DC, BC, CC link types.

Question: Is hub instability specific to the combined network,
or does it appear in each individual link type (DC, BC, CC)?

Strategy: BFS subsample on DC (fast), then load BC/CC only for those nodes.
"""
from __future__ import annotations
import sys, time, logging
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("hub"); log.setLevel(logging.INFO)

DATA_BASE = Path.home() / "Desktop/HDD/archive_local_map_analysis/oa6_intermediate_20260314/multidomain_networks_balanced"
FIELD_ID = "22"


def load_networks(field_id="22", n_target=5000, seed=42):
    """Load DC, BC, CC for a field, BFS subsample on DC then filter BC/CC."""
    import polars as pl
    import igraph as ig

    # ── 1. DC: load field-specific, BFS subsample ──
    log.info("Loading DC for field %s...", field_id)
    dc = pl.read_parquet(str(DATA_BASE / "dc_work_edges_balanced.parquet"))
    dc = dc.filter(
        (pl.col("src_field_id") == field_id) &
        (pl.col("dst_field_id") == field_id)
    )
    log.info("  DC: %d edges", len(dc))

    all_uids = sorted(set(dc["src_work_id"].to_list() + dc["dst_work_id"].to_list()))
    uid2i = {u: i for i, u in enumerate(all_uids)}

    g_dc = ig.Graph(
        n=len(all_uids),
        edges=list(zip(
            [uid2i[u] for u in dc["src_work_id"].to_list()],
            [uid2i[u] for u in dc["dst_work_id"].to_list()]
        )),
        directed=False
    )
    g_dc.es["weight"] = [1.0] * g_dc.ecount()
    g_dc = g_dc.simplify(combine_edges="sum")
    gcc = g_dc.connected_components().giant().vs.indices
    g_dc = g_dc.subgraph(gcc)
    gcc_uids = [all_uids[i] for i in gcc]
    log.info("  DC GCC: %d nodes, %d edges", g_dc.vcount(), g_dc.ecount())

    # BFS subsample
    if n_target and g_dc.vcount() > n_target:
        import random; random.seed(seed)
        start = random.randint(0, g_dc.vcount() - 1)
        keep = g_dc.bfs(start)[0][:n_target]
        g_dc = g_dc.subgraph(keep)
        g_dc = g_dc.simplify(combine_edges="sum")
        gcc2 = g_dc.connected_components().giant().vs.indices
        g_dc = g_dc.subgraph(gcc2)
        sample_uids = [gcc_uids[keep[i]] for i in gcc2]
    else:
        sample_uids = gcc_uids

    n = len(sample_uids)
    uid2s = {u: i for i, u in enumerate(sample_uids)}
    log.info("  Subsample: %d nodes, DC edges: %d", n, g_dc.ecount())

    # ── 2. BC: filter only sample nodes (fast — small filter set) ──
    log.info("Loading BC for %d sample nodes...", n)
    sample_series = pl.Series("sn", sample_uids)
    bc = pl.read_parquet(str(DATA_BASE / "bc_work_edges_balanced.parquet"))
    bc = bc.filter(
        pl.col("work_id_1").is_in(sample_series) &
        pl.col("work_id_2").is_in(sample_series)
    )
    log.info("  BC for subsample: %d edges", len(bc))

    bc_edges, bc_weights = [], []
    for row in bc.rows():
        u, v, w = row[0], row[1], float(row[2])
        if u in uid2s and v in uid2s:
            bc_edges.append((uid2s[u], uid2s[v]))
            bc_weights.append(w)

    g_bc = ig.Graph(n=n, edges=bc_edges, directed=False)
    g_bc.es["weight"] = bc_weights if bc_weights else []
    g_bc = g_bc.simplify(combine_edges="sum")
    log.info("  BC graph: %d edges (of %d nodes)", g_bc.ecount(), n)

    # ── 3. CC: same approach ──
    log.info("Loading CC for %d sample nodes...", n)
    cc = pl.read_parquet(str(DATA_BASE / "cc_work_edges_balanced.parquet"))
    cc = cc.filter(
        pl.col("work_id_1").is_in(sample_series) &
        pl.col("work_id_2").is_in(sample_series)
    )
    log.info("  CC for subsample: %d edges", len(cc))

    cc_edges, cc_weights = [], []
    for row in cc.rows():
        u, v, w = row[0], row[1], float(row[2])
        if u in uid2s and v in uid2s:
            cc_edges.append((uid2s[u], uid2s[v]))
            cc_weights.append(w)

    g_cc = ig.Graph(n=n, edges=cc_edges, directed=False)
    g_cc.es["weight"] = cc_weights if cc_weights else []
    g_cc = g_cc.simplify(combine_edges="sum")
    log.info("  CC graph: %d edges (of %d nodes)", g_cc.ecount(), n)

    # ── 4. Combined ──
    edge_map = {}
    for g_part in [g_dc, g_bc, g_cc]:
        for e in g_part.es:
            key = (min(e.source, e.target), max(e.source, e.target))
            edge_map[key] = edge_map.get(key, 0) + e["weight"]

    comb_e = list(edge_map.keys())
    comb_w = [edge_map[k] for k in comb_e]
    g_comb = ig.Graph(n=n, edges=comb_e, directed=False)
    g_comb.es["weight"] = comb_w
    g_comb = g_comb.simplify(combine_edges="sum")
    log.info("  Combined: %d edges", g_comb.ecount())

    return {"DC": g_dc, "BC": g_bc, "CC": g_cc, "Combined": g_comb}


def leiden_ensemble(g, gamma, n_runs=50):
    import leidenalg
    return [
        leidenalg.find_partition(
            g, leidenalg.CPMVertexPartition,
            resolution_parameter=gamma, weights="weight", seed=s
        ).membership
        for s in range(n_runs)
    ]


def node_stability(memberships):
    n_nodes = len(memberships[0])
    n_runs = len(memberships)
    stab = np.zeros(n_nodes)
    for i in range(n_nodes):
        counts = Counter(m[i] for m in memberships)
        stab[i] = max(counts.values()) / n_runs
    return stab


def analyze(g, name, gamma, n_runs=50):
    """Return stability analysis dict."""
    n = g.vcount()
    degrees = np.array(g.degree())
    strengths = np.array(g.strength(weights="weight"))

    log.info("  %s: Leiden ×%d (γ=%.4f, %d nodes, %d edges)...",
             name, n_runs, gamma, n, g.ecount())
    t0 = time.perf_counter()
    mems = leiden_ensemble(g, gamma, n_runs)
    dt = time.perf_counter() - t0

    stab = node_stability(mems)
    nc = [len(set(m)) for m in mems]

    return {
        "name": name, "n": n, "m": g.ecount(), "gamma": gamma,
        "time": dt,
        "nc_mean": np.mean(nc), "nc_std": np.std(nc),
        "stab": stab, "deg": degrees, "str": strengths,
        "vu": stab < 0.5, "un": (stab >= 0.5) & (stab < 0.8),
        "st": (stab >= 0.8) & (stab < 1.0), "pf": stab == 1.0,
    }


def print_results(results, field_id="22"):
    print(f"\n{'='*100}")
    print(f"HUB STABILITY BY LINK TYPE (OpenAlex field {field_id})")
    print(f"{'='*100}")

    # ── Basic ──
    print(f"\n{'Net':<10} {'N':>5} {'E':>7} {'γ':>7} "
          f"{'#C':>6}±{'std':>4} {'t':>5}")
    print("-" * 55)
    for r in results:
        print(f"{r['name']:<10} {r['n']:>5} {r['m']:>7} {r['gamma']:>7.4f} "
              f"{r['nc_mean']:>6.0f}±{r['nc_std']:>3.0f} {r['time']:>4.1f}s")

    # ── Stability tiers ──
    print(f"\n{'Net':<10} {'VU(<0.5)':>10} {'Unstab':>10} {'Stable':>10} {'Perfect':>10} {'mean':>6}")
    print("-" * 62)
    for r in results:
        n = r['n']
        print(f"{r['name']:<10} "
              f"{r['vu'].sum():>5}({r['vu'].sum()/n*100:>4.1f}%) "
              f"{r['un'].sum():>5}({r['un'].sum()/n*100:>4.1f}%) "
              f"{r['st'].sum():>5}({r['st'].sum()/n*100:>4.1f}%) "
              f"{r['pf'].sum():>5}({r['pf'].sum()/n*100:>4.1f}%) "
              f"{r['stab'].mean():>6.3f}")

    # ── Degree profile ──
    print(f"\n{'Net':<10} {'VU_deg':>8} {'VU_str':>8} {'PF_deg':>8} {'PF_str':>8} "
          f"{'VU_s/d':>7} {'PF_s/d':>7}")
    print("-" * 60)
    for r in results:
        vu, pf, d, s = r['vu'], r['pf'], r['deg'], r['str']
        if vu.sum() > 0 and pf.sum() > 0:
            vud, vus = d[vu].mean(), s[vu].mean()
            pfd, pfs = d[pf].mean(), s[pf].mean()
            print(f"{r['name']:<10} {vud:>8.1f} {vus:>8.1f} {pfd:>8.1f} {pfs:>8.1f} "
                  f"{vus/vud if vud else 0:>7.2f} {pfs/pfd if pfd else 0:>7.2f}")
        elif vu.sum() > 0:
            vud, vus = d[vu].mean(), s[vu].mean()
            print(f"{r['name']:<10} {vud:>8.1f} {vus:>8.1f} {'—':>8} {'—':>8} "
                  f"{vus/vud if vud else 0:>7.2f} {'—':>7}")
        else:
            print(f"{r['name']:<10} (no VU nodes)")

    # ── Where do VU nodes sit in degree distribution? ──
    print(f"\n{'Net':<10} {'VU%top5':>8} {'VU%top10':>9} {'VU%top25':>9} {'VU%bot50':>9}")
    print("-" * 48)
    for r in results:
        vu, d = r['vu'], r['deg']
        if vu.sum() == 0:
            print(f"{r['name']:<10} —"); continue
        vud = d[vu]
        t5 = (vud >= np.percentile(d, 95)).sum() / vu.sum() * 100
        t10 = (vud >= np.percentile(d, 90)).sum() / vu.sum() * 100
        t25 = (vud >= np.percentile(d, 75)).sum() / vu.sum() * 100
        b50 = (vud <= np.median(d)).sum() / vu.sum() * 100
        print(f"{r['name']:<10} {t5:>7.1f}% {t10:>8.1f}% {t25:>8.1f}% {b50:>8.1f}%")

    # ── VU overlap between networks ──
    print(f"\n{'─'*80}")
    print("VERY UNSTABLE NODE OVERLAP")
    print(f"{'─'*80}")
    vu_sets = {r['name']: set(np.where(r['vu'])[0]) for r in results}
    names = [r['name'] for r in results]
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            n1, n2 = names[i], names[j]
            s1, s2 = vu_sets[n1], vu_sets[n2]
            if not s1 or not s2: continue
            inter = len(s1 & s2)
            union = len(s1 | s2)
            jacc = inter / union if union else 0
            print(f"  {n1:>10} ∩ {n2:<10}: "
                  f"|∩|={inter:>5} Jacc={jacc:.3f} "
                  f"{n1}→{n2}={inter/len(s1)*100:.0f}% "
                  f"{n2}→{n1}={inter/len(s2)*100:.0f}%")

    # ── Consistently unstable ──
    singles = [n for n in names if n != 'Combined']
    if len(singles) >= 2:
        all_vu = vu_sets[singles[0]]
        any_vu = vu_sets[singles[0]].copy()
        for s in singles[1:]:
            all_vu = all_vu & vu_sets[s]
            any_vu = any_vu | vu_sets[s]

        n_total = results[0]['n']
        print(f"\n  VU in ALL of {singles}: {len(all_vu)} ({len(all_vu)/n_total*100:.1f}%)")
        print(f"  VU in ANY of {singles}: {len(any_vu)} ({len(any_vu)/n_total*100:.1f}%)")
        if any_vu:
            print(f"  Consistency: {len(all_vu)/len(any_vu)*100:.1f}%")

        # Degree profile of consistently VU vs link-specific VU
        comb_deg = [r['deg'] for r in results if r['name'] == 'Combined']
        d = comb_deg[0] if comb_deg else results[0]['deg']
        only_one = any_vu - all_vu
        if all_vu:
            print(f"  Consistently VU deg: mean={d[list(all_vu)].mean():.1f}, "
                  f"median={np.median(d[list(all_vu)]):.0f}")
        if only_one:
            print(f"  Link-specific VU deg: mean={d[list(only_one)].mean():.1f}, "
                  f"median={np.median(d[list(only_one)]):.0f}")

    pass  # edge overlap printed in main()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--field", default=FIELD_ID)
    parser.add_argument("--n-nodes", type=int, default=5000)
    parser.add_argument("--n-runs", type=int, default=50)
    args = parser.parse_args()

    field_id = args.field

    graphs = load_networks(field_id=field_id, n_target=args.n_nodes)

    # Print edge overlap between link types
    import igraph as ig
    for name in ["DC", "BC", "CC"]:
        g = graphs[name]
        e_set = {(min(e.source, e.target), max(e.source, e.target)) for e in g.es}
        print(f"  {name}: {len(e_set)} unique edges")

    dc_edges = {(min(e.source, e.target), max(e.source, e.target)) for e in graphs["DC"].es}
    bc_edges = {(min(e.source, e.target), max(e.source, e.target)) for e in graphs["BC"].es}
    cc_edges = {(min(e.source, e.target), max(e.source, e.target)) for e in graphs["CC"].es}
    print(f"  DC∩BC: {len(dc_edges & bc_edges)}, DC∩CC: {len(dc_edges & cc_edges)}, BC∩CC: {len(bc_edges & cc_edges)}")
    print(f"  DC∩BC∩CC: {len(dc_edges & bc_edges & cc_edges)}")
    print(f"  Union: {len(dc_edges | bc_edges | cc_edges)}")

    results = []
    for name, g in graphs.items():
        if g.ecount() == 0:
            log.warning("  %s: no edges, skipping", name)
            continue

        # Use GCC
        gcc = g.connected_components().giant()
        g_gcc = g.subgraph(gcc.vs.indices)
        if g_gcc.vcount() < 50:
            log.warning("  %s GCC too small (%d), skipping", name, g_gcc.vcount())
            continue

        # γ selection: use density-based heuristic
        n, m = g_gcc.vcount(), g_gcc.ecount()
        mean_w = np.mean(g_gcc.es["weight"])
        density = 2 * m / (n * (n - 1)) if n > 1 else 0

        # Target: meaningful clusters (not all singletons, not one giant)
        # CPM: cluster if internal density > γ
        # Set γ = fraction of mean weighted density
        gamma = mean_w * density * 2
        gamma = max(gamma, 0.001)
        gamma = min(gamma, 0.5)

        r = analyze(g_gcc, name, gamma, n_runs=args.n_runs)
        results.append(r)

    print_results(results, field_id=field_id)


if __name__ == "__main__":
    main()
