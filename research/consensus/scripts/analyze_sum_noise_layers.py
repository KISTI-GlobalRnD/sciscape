"""Analyze which edge layers inject noisy neighbors into local rank-shift cases."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from _common import (
    abstracts_lookup,
    load_abstracts_table,
    load_layer_tables,
    save_json,
)

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sciscape.linkage.filters import filter_top_k


def _canonical_pair(uid1: str, uid2: str) -> tuple[str, str]:
    return (uid1, uid2) if uid1 < uid2 else (uid2, uid1)


def _layer_weights(filtered_layers: dict[str, Any]) -> dict[str, float]:
    counts = {name: df.height for name, df in filtered_layers.items() if df.height > 0}
    total_inv = sum(1.0 / count for count in counts.values())
    n_layers = len(counts)
    return {
        name: (1.0 / count) / total_inv * n_layers
        for name, count in counts.items()
        if count > 0
    }


def _build_pair_lookup(
    filtered_layers: dict[str, Any],
    pairset: set[tuple[str, str]],
    layer_weights: dict[str, float],
) -> dict[str, dict[tuple[str, str], float]]:
    lookup: dict[str, dict[tuple[str, str], float]] = {}
    for name, df in filtered_layers.items():
        current: dict[tuple[str, str], float] = {}
        scale = layer_weights[name]
        for row in df.iter_rows(named=True):
            key = _canonical_pair(row["uid1"], row["uid2"])
            if key in pairset:
                current[key] = float(row["rel_sum2"]) * scale
        lookup[name] = current
    return lookup


def _suspect_neighbors(case: dict[str, Any], *, noisy_side: str) -> list[dict[str, Any]]:
    if noisy_side == "a":
        suspects = [dict(row, kind="only_a") for row in case.get("neighbors_only_a", [])]
        suspects.extend(
            dict(row, kind="shared")
            for row in case.get("shared_neighbors", [])
            if int(row.get("delta", 0)) > 0
        )
        return suspects
    if noisy_side == "b":
        suspects = [dict(row, kind="only_b") for row in case.get("neighbors_only_b", [])]
        suspects.extend(
            dict(row, kind="shared")
            for row in case.get("shared_neighbors", [])
            if int(row.get("delta", 0)) < 0
        )
        return suspects
    raise ValueError(f"Unknown noisy_side={noisy_side!r}")


def _winner_matches(case: dict[str, Any], winner: str) -> bool:
    return case.get("comparison", {}).get("winner") == winner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review_json", type=Path, help="rank_shift_review JSON file")
    parser.add_argument(
        "--winner",
        type=str,
        default="B",
        choices=("A", "B"),
        help="Only analyze cases where this side wins (default: B)",
    )
    parser.add_argument(
        "--noisy-side",
        type=str,
        default="a",
        choices=("a", "b"),
        help="Which side's promoted neighbors to treat as candidate noise (default: a)",
    )
    parser.add_argument(
        "--top-examples",
        type=int,
        default=8,
        help="How many high-confidence example pairs to keep",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Optional explicit output path; defaults next to review JSON",
    )
    args = parser.parse_args()

    review = json.loads(args.review_json.read_text())
    edge_dir = Path(review["edge_dir"])
    abstract_path = Path(review["abstract_path"])
    top_k = int(review["top_k"])

    layers = load_layer_tables(edge_dir)
    filtered_layers = {
        name: filter_top_k(df, top_k, mode="symmetric")
        for name, df in layers.items()
        if df.height > 0
    }
    layer_weights = _layer_weights(filtered_layers)

    meta = abstracts_lookup(load_abstracts_table(abstract_path))

    pairset: set[tuple[str, str]] = set()
    suspect_rows: list[dict[str, Any]] = []
    selected_cases = [case for case in review["reviewed_cases"] if _winner_matches(case, args.winner)]
    for case in selected_cases:
        target_uid = case["target_uid"]
        for row in _suspect_neighbors(case, noisy_side=args.noisy_side):
            neighbor_uid = row["uid"]
            key = _canonical_pair(target_uid, neighbor_uid)
            pairset.add(key)
            suspect_rows.append(
                {
                    "target_uid": target_uid,
                    "neighbor_uid": neighbor_uid,
                    "kind": row["kind"],
                    "rank": row.get("rank"),
                    "rank_a": row.get("rank_a"),
                    "rank_b": row.get("rank_b"),
                    "delta": row.get("delta"),
                    "pair_key": key,
                }
            )

    pair_lookup = _build_pair_lookup(filtered_layers, pairset, layer_weights)

    dominant_counts: Counter[str] = Counter()
    single_layer_counts: Counter[str] = Counter()
    present_any_counts: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []

    for row in suspect_rows:
        contributions = {
            name: pair_lookup[name].get(row["pair_key"], 0.0)
            for name in pair_lookup
        }
        contributions = {name: value for name, value in contributions.items() if value > 0}
        if not contributions:
            continue
        dominant_layer = max(contributions, key=contributions.get)
        dominant_counts[dominant_layer] += 1
        present_any_counts.update(contributions.keys())
        if len(contributions) == 1:
            single_layer_counts[dominant_layer] += 1

        examples.append(
            {
                "target_uid": row["target_uid"],
                "target_title": (meta.get(row["target_uid"], {}) or {}).get("title"),
                "neighbor_uid": row["neighbor_uid"],
                "neighbor_title": (meta.get(row["neighbor_uid"], {}) or {}).get("title"),
                "kind": row["kind"],
                "delta": row["delta"],
                "rank": row["rank"],
                "rank_a": row["rank_a"],
                "rank_b": row["rank_b"],
                "dominant_layer": dominant_layer,
                "n_layers_present": len(contributions),
                "contributions": {
                    name: round(value, 6)
                    for name, value in sorted(contributions.items(), key=lambda item: -item[1])
                },
            }
        )

    examples.sort(
        key=lambda item: (
            item["n_layers_present"] != 1,
            -max(item["contributions"].values()) if item["contributions"] else 0.0,
        )
    )

    payload = {
        "review_json": str(args.review_json),
        "field": review["field"],
        "method_a": review["method_a"],
        "method_b": review["method_b"],
        "winner_filter": args.winner,
        "noisy_side": args.noisy_side,
        "top_k": top_k,
        "n_selected_cases": len(selected_cases),
        "n_suspect_rows": len(suspect_rows),
        "n_unique_pairs": len(pairset),
        "layer_weights": {name: round(weight, 6) for name, weight in layer_weights.items()},
        "dominant_layer_counts": dict(dominant_counts),
        "single_layer_counts": dict(single_layer_counts),
        "present_any_counts": dict(present_any_counts),
        "top_examples": examples[: args.top_examples],
    }

    output_path = args.output or args.review_json.with_name(
        args.review_json.stem.replace("_rank_shift_review", "_noise_layers") + ".json"
    )
    save_json(payload, output_path)
    print(f"Saved → {output_path}")


if __name__ == "__main__":
    main()
