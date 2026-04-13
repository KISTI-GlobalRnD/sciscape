#!/usr/bin/env python3
"""Triple evaluation: belonging + group cohesion + outlier detection.

For each disagreement node between sum and boosted:
  1. Belonging: "Which group does this paper belong to?"
  2. Group cohesion: score each cluster independently (no target)
  3. Outlier detection: how many outliers in each cluster?

Usage:
    python experiments/combination/triple_eval.py --gamma 1e-4 --n 10
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sciscape.evaluation.reviewer import (
    review_belonging, review_group_cohesion, review_outliers,
)

OUT = Path("experiments/combination/results")


def load_env_key():
    for p in [
        Path("/home/kimyoungjin06/Desktop/Workspace/1.4.2.KRISS/.env"),
        Path("/home/kimyoungjin06/Desktop/Workspace/.env"),
    ]:
        if p.exists():
            env = dict(line.split("=", 1) for line in p.read_text().splitlines()
                       if "=" in line and not line.startswith("#"))
            if "OPENAI_API_KEY" in env:
                return env["OPENAI_API_KEY"]
    raise RuntimeError("No OPENAI_API_KEY found")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", type=float, default=1e-4)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--n-neighbors", type=int, default=8)
    args = parser.parse_args()

    g_str = f"{args.gamma:.0e}"

    mem_sum = json.load(open(OUT / f"mem_sum_g{g_str}.json"))
    mem_boost = json.load(open(OUT / f"mem_boosted_g{g_str}.json"))

    abs_df = pl.read_parquet("data/openalex_metadata/field_15/works_text.parquet").rename({"work_id": "uid"})
    uid_to_abs = {r["uid"]: r for r in abs_df.iter_rows(named=True)}

    from openai import OpenAI
    client = OpenAI(api_key=load_env_key())

    common = set(mem_sum) & set(mem_boost) & set(uid_to_abs)
    diff_uids = [u for u in common if mem_sum[u] != mem_boost[u]]

    print(f"γ={g_str}, common={len(common):,}, diff={len(diff_uids):,}")

    rng = np.random.RandomState(42)
    rng.shuffle(diff_uids)

    results = []
    done = 0

    for uid in diff_uids:
        if done >= args.n:
            break

        t = uid_to_abs.get(uid)
        if not t or not t.get("abstract"):
            continue

        cid_a = mem_sum[uid]
        cid_b = mem_boost[uid]
        sz_a = sum(1 for v in mem_sum.values() if v == cid_a)
        sz_b = sum(1 for v in mem_boost.values() if v == cid_b)

        pool_a = [u for u in common if mem_sum[u] == cid_a and u != uid]
        pool_b = [u for u in common if mem_boost[u] == cid_b and u != uid]
        if len(pool_a) < args.n_neighbors or len(pool_b) < args.n_neighbors:
            continue

        rng2 = np.random.RandomState(hash(uid) % 2**31)
        nbrs_a = [uid_to_abs[u] for u in rng2.choice(pool_a, args.n_neighbors, replace=False)]
        nbrs_b = [uid_to_abs[u] for u in rng2.choice(pool_b, args.n_neighbors, replace=False)]

        title_short = t["title"][:65] if t.get("title") else "?"
        done += 1
        print(f"\n[{done}/{args.n}] {title_short}...")
        print(f"  sum=C{cid_a}({sz_a}) vs boosted=C{cid_b}({sz_b})")

        case = {"uid": uid, "title": t.get("title", ""),
                "sum_cluster": cid_a, "sum_size": sz_a,
                "boosted_cluster": cid_b, "boosted_size": sz_b}

        # Eval 1: Belonging
        try:
            b = review_belonging(client, t, nbrs_a, nbrs_b,
                                 method_a="sum", method_b="boosted", model=args.model)
            w = "SUM" if b.belongs_to == "A" else "BOOSTED"
            case["belonging"] = {"winner": w, "confidence": b.confidence, "reasoning": b.reasoning}
            print(f"  Belonging: {w} (conf={b.confidence})")
        except Exception as e:
            case["belonging"] = {"error": str(e)}
            print(f"  Belonging: error {e}")
        time.sleep(0.5)

        # Eval 2: Group cohesion (independently)
        for label, nbrs, method in [("sum", nbrs_a, "sum"), ("boosted", nbrs_b, "boosted")]:
            try:
                g = review_group_cohesion(client, nbrs, method=method, model=args.model)
                case[f"cohesion_{label}"] = {
                    "score": g.cohesion_score, "theme": g.theme,
                    "n_outliers": g.n_outliers, "reasoning": g.reasoning,
                }
                print(f"  Cohesion {label}: {g.cohesion_score}/5, outliers={g.n_outliers}, theme='{g.theme[:50]}'")
            except Exception as e:
                case[f"cohesion_{label}"] = {"error": str(e)}
                print(f"  Cohesion {label}: error {e}")
            time.sleep(0.5)

        # Eval 3: Outlier detection
        for label, nbrs, method in [("sum", nbrs_a, "sum"), ("boosted", nbrs_b, "boosted")]:
            try:
                o = review_outliers(client, t, nbrs, method=method, model=args.model)
                case[f"outliers_{label}"] = {
                    "n_outliers": o.n_outliers, "theme": o.cluster_theme,
                    "reasoning": o.reasoning,
                }
                print(f"  Outliers {label}: {o.n_outliers}/{args.n_neighbors}")
            except Exception as e:
                case[f"outliers_{label}"] = {"error": str(e)}
                print(f"  Outliers {label}: error {e}")
            time.sleep(0.5)

        results.append(case)

    # Save
    result_path = OUT / f"triple_eval_g{g_str}.json"
    json.dump(results, open(result_path, "w"), indent=2, ensure_ascii=False)
    print(f"\nSaved to {result_path}")

    # Summary
    if results:
        print(f"\n{'='*60}")
        print(f"SUMMARY ({len(results)} cases, γ={g_str})")
        print(f"{'='*60}")

        # Belonging
        bwins = Counter(r.get("belonging", {}).get("winner", "?") for r in results)
        print(f"\nBelonging: SUM={bwins.get('SUM',0)}, BOOSTED={bwins.get('BOOSTED',0)}")

        # Cohesion
        cs = [r.get("cohesion_sum", {}).get("score", 0) for r in results if "cohesion_sum" in r]
        cb = [r.get("cohesion_boosted", {}).get("score", 0) for r in results if "cohesion_boosted" in r]
        if cs and cb:
            print(f"Cohesion avg: sum={np.mean(cs):.1f}, boosted={np.mean(cb):.1f}")

        # Outliers
        os_ = [r.get("outliers_sum", {}).get("n_outliers", 0) for r in results if "outliers_sum" in r]
        ob = [r.get("outliers_boosted", {}).get("n_outliers", 0) for r in results if "outliers_boosted" in r]
        if os_ and ob:
            print(f"Outliers avg: sum={np.mean(os_):.1f}, boosted={np.mean(ob):.1f}")


if __name__ == "__main__":
    main()
