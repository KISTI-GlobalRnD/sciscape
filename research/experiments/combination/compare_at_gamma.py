#!/usr/bin/env python3
"""Compare sum vs boosted at a specific γ via LLM blind review.

Picks disagreement nodes (different cluster in sum vs boosted),
samples neighbors from each, asks LLM which cluster is more cohesive.

Usage:
    python research/experiments/combination/compare_at_gamma.py --gamma 5e-5
    python research/experiments/combination/compare_at_gamma.py --auto  # pick from sweep_summary

Requires: sweep_summary.json + mem_*.json from gamma_sweep.py
"""

import argparse
import json
import time
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sciscape.evaluation.reviewer import review_comparison


OUT = Path("research/experiments/combination/results")


def load_env_key():
    """Load OpenAI key from nearest .env."""
    for p in [
        Path("/home/kimyoungjin06/Desktop/Workspace/1.4.2.KRISS/.env"),
        Path("/home/kimyoungjin06/Desktop/Workspace/.env"),
    ]:
        if p.exists():
            env = dict(
                line.split("=", 1)
                for line in p.read_text().splitlines()
                if "=" in line and not line.startswith("#")
            )
            if "OPENAI_API_KEY" in env:
                return env["OPENAI_API_KEY"]
    raise RuntimeError("No OPENAI_API_KEY found")


def pick_gamma(max_pct=3.0):
    """Pick lowest γ where both methods have max cluster < max_pct%."""
    summary = json.load(open(OUT / "sweep_summary.json"))
    gammas = sorted(set(r["gamma"] for r in summary))
    for g in gammas:
        s = next((r for r in summary if r["strategy"] == "sum" and r["gamma"] == g), None)
        b = next((r for r in summary if r["strategy"] == "boosted" and r["gamma"] == g), None)
        if s and b and s["max_pct"] < max_pct and b["max_pct"] < max_pct:
            return g
    # Fallback: pick γ with smallest max(sum_max%, boosted_max%)
    best_g, best_score = gammas[0], 100
    for g in gammas:
        s = next((r for r in summary if r["strategy"] == "sum" and r["gamma"] == g), None)
        b = next((r for r in summary if r["strategy"] == "boosted" and r["gamma"] == g), None)
        if s and b:
            score = max(s["max_pct"], b["max_pct"])
            if score < best_score:
                best_score = score
                best_g = g
    return best_g


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--auto", action="store_true", help="Auto-pick γ from sweep")
    parser.add_argument("--max-pct", type=float, default=3.0, help="Max cluster % threshold")
    parser.add_argument("--n-compare", type=int, default=10)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    args = parser.parse_args()

    if args.auto or args.gamma is None:
        gamma = pick_gamma(args.max_pct)
        print(f"Auto-picked γ={gamma:.0e} (max_pct < {args.max_pct}%)")
    else:
        gamma = args.gamma

    # Load memberships
    g_str = f"{gamma:.0e}"
    sum_path = OUT / f"mem_sum_g{g_str}.json"
    boost_path = OUT / f"mem_boosted_g{g_str}.json"

    if not sum_path.exists() or not boost_path.exists():
        print(f"ERROR: membership files not found for γ={g_str}")
        print(f"  Expected: {sum_path}")
        print(f"  Expected: {boost_path}")
        print("Run gamma_sweep.py first.")
        return

    mem_sum = json.load(open(sum_path))
    mem_boost = json.load(open(boost_path))

    # Load abstracts
    abs_df = pl.read_parquet("workspace/data/openalex_metadata/field_15/works_text.parquet").rename({"work_id": "uid"})
    uid_to_abs = {r["uid"]: r for r in abs_df.iter_rows(named=True)}

    # Stats
    common = set(mem_sum) & set(mem_boost) & set(uid_to_abs)
    diff_uids = [u for u in common if mem_sum[u] != mem_boost[u]]
    same = len(common) - len(diff_uids)

    sum_sizes = Counter(mem_sum.values())
    boost_sizes = Counter(mem_boost.values())
    sum_max = max(sum_sizes.values())
    boost_max = max(boost_sizes.values())
    n = len(mem_sum)

    print(f"\n=== γ={gamma:.0e} comparison ===")
    print(f"sum:     {len(sum_sizes)} cl, max={sum_max} ({100*sum_max/n:.1f}%)")
    print(f"boosted: {len(boost_sizes)} cl, max={boost_max} ({100*boost_max/n:.1f}%)")
    print(f"Common with abs: {len(common):,}")
    print(f"Same cluster: {same:,} ({100*same/len(common):.1f}%)")
    print(f"Different:    {len(diff_uids):,} ({100*len(diff_uids)/len(common):.1f}%)")

    if not diff_uids:
        print("No disagreements — methods produce identical results!")
        return

    # Sample disagreement nodes
    rng = np.random.RandomState(42)
    rng.shuffle(diff_uids)
    targets = diff_uids[:args.n_compare * 2]  # extra in case some fail

    # Setup LLM
    from openai import OpenAI
    client = OpenAI(api_key=load_env_key())

    comparisons = []
    for uid in targets:
        if len(comparisons) >= args.n_compare:
            break

        t = uid_to_abs.get(uid)
        if not t or not t.get("abstract"):
            continue

        cid_a = mem_sum[uid]
        cid_b = mem_boost[uid]
        sz_a = sum(1 for v in mem_sum.values() if v == cid_a)
        sz_b = sum(1 for v in mem_boost.values() if v == cid_b)

        # Sample 5 neighbors from each cluster
        pool_a = [u for u in common if mem_sum[u] == cid_a and u != uid]
        pool_b = [u for u in common if mem_boost[u] == cid_b and u != uid]

        rng2 = np.random.RandomState(hash(uid) % 2**31)
        sa = [uid_to_abs[u] for u in rng2.choice(pool_a, min(5, len(pool_a)), replace=False)] if pool_a else []
        sb = [uid_to_abs[u] for u in rng2.choice(pool_b, min(5, len(pool_b)), replace=False)] if pool_b else []

        if len(sa) < 2 or len(sb) < 2:
            continue

        title_short = t["title"][:70] if t.get("title") else "?"
        print(f"\n[{len(comparisons)+1}/{args.n_compare}] {title_short}...")
        print(f"  sum=C{cid_a}({sz_a}) vs boosted=C{cid_b}({sz_b})")

        try:
            comp = review_comparison(
                client, t, sa, sb,
                method_a="sum", method_b="boosted",
                model=args.model,
            )
            comparisons.append({
                "uid": uid,
                "title": t.get("title", ""),
                "winner": comp.winner,
                "score_sum": comp.score_a,
                "score_boosted": comp.score_b,
                "reasoning": comp.reasoning,
                "sum_cluster": cid_a,
                "boosted_cluster": cid_b,
                "sum_size": sz_a,
                "boosted_size": sz_b,
            })
            w = "SUM" if comp.winner == "A" else "BOOSTED"
            print(f"  → {w} wins (sum={comp.score_a}, boosted={comp.score_b})")
            print(f"    {comp.reasoning[:120]}")
        except Exception as e:
            print(f"  → Error: {e}")

        time.sleep(1)

    # Save results
    result_path = OUT / f"compare_g{g_str}.json"
    json.dump(comparisons, open(result_path, "w"), indent=2, ensure_ascii=False)
    print(f"\nSaved to {result_path}")

    # Summary
    if comparisons:
        wins = Counter(c["winner"] for c in comparisons)
        avg_sum = np.mean([c["score_sum"] for c in comparisons])
        avg_boost = np.mean([c["score_boosted"] for c in comparisons])

        print(f"\n{'='*60}")
        print(f"RESULT: γ={gamma:.0e}, {len(comparisons)} comparisons")
        print(f"  SUM wins:     {wins.get('A', 0)}")
        print(f"  BOOSTED wins: {wins.get('B', 0)}")
        print(f"  Avg score — sum: {avg_sum:.1f}, boosted: {avg_boost:.1f}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
