"""Aggregate one or more taxonomy result JSON files into a combined summary."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from _common import save_json


def _representative_by_label(classified_cases: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    labels = sorted({case["primary_label"] for case in classified_cases})
    by_label: dict[str, list[dict[str, Any]]] = {}
    for label in labels:
        cases = [case for case in classified_cases if case["primary_label"] == label]
        cases.sort(
            key=lambda case: (
                -float(case.get("score_gap", 0.0)),
                -int(case.get("taxonomy_confidence", 0)),
                case.get("target_uid", ""),
            )
        )
        by_label[label] = [
            {
                "field": case["field"],
                "top_k": case["top_k"],
                "winner_method": case["winner_method"],
                "target_uid": case["target_uid"],
                "target_title": case["target_title"],
                "score_gap": case["score_gap"],
                "winner_advantage": case.get("winner_advantage", ""),
                "loser_failure_mode": case.get("loser_failure_mode", ""),
                "review_reasoning": case.get("review_reasoning", ""),
            }
            for case in cases[:2]
        ]
    return by_label


def _representative_by_winner(classified_cases: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    winner_methods = sorted({case["winner_method"] for case in classified_cases})
    return {
        winner_method: [
            {
                "field": case["field"],
                "top_k": case["top_k"],
                "primary_label": case["primary_label"],
                "target_uid": case["target_uid"],
                "target_title": case["target_title"],
                "score_gap": case["score_gap"],
                "winner_advantage": case.get("winner_advantage", ""),
                "loser_failure_mode": case.get("loser_failure_mode", ""),
                "review_reasoning": case.get("review_reasoning", ""),
            }
            for case in sorted(
                [row for row in classified_cases if row["winner_method"] == winner_method],
                key=lambda case: (
                    -float(case.get("score_gap", 0.0)),
                    -int(case.get("taxonomy_confidence", 0)),
                    case.get("target_uid", ""),
                ),
            )[:4]
        ]
        for winner_method in winner_methods
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("taxonomy_json", nargs="+", type=Path, help="One or more *_taxonomy.json files")
    parser.add_argument("-o", "--output", type=Path, default=Path("results/taxonomy"))
    parser.add_argument("--stem", type=str, default="taxonomy_combined")
    args = parser.parse_args()

    combined_rows: list[dict[str, Any]] = []
    per_file_summaries: list[dict[str, Any]] = []
    model_used: list[str] = []

    for path in args.taxonomy_json:
        payload = json.loads(path.read_text(encoding="utf-8"))
        combined_rows.extend(payload.get("classified_cases", []))
        per_file_summaries.append(
            {
                "taxonomy_json": str(path),
                "review_json": payload.get("review_json", ""),
                "field": payload.get("field", ""),
                "top_k": int(payload.get("top_k", 0)),
                "label_a": payload.get("label_a", ""),
                "label_b": payload.get("label_b", ""),
                **payload.get("summary", {}),
            }
        )
        raw_model = payload.get("model_used")
        if raw_model:
            model_used.append(str(raw_model))

    label_counts: Counter[str] = Counter(row["primary_label"] for row in combined_rows)
    winner_counts: Counter[str] = Counter(row["winner_method"] for row in combined_rows)
    label_by_winner: dict[str, Counter[str]] = defaultdict(Counter)
    winner_by_k: dict[int, Counter[str]] = defaultdict(Counter)
    for row in combined_rows:
        label_by_winner[row["winner_method"]][row["primary_label"]] += 1
        winner_by_k[int(row["top_k"])][row["winner_method"]] += 1

    combined_summary = {
        "model_used": sorted(set(model_used)),
        "n_taxonomy_files": len(args.taxonomy_json),
        "n_classified_cases": len(combined_rows),
        "per_file": per_file_summaries,
        "label_counts": dict(label_counts),
        "winner_counts": dict(winner_counts),
        "label_by_winner": {
            winner: dict(counter)
            for winner, counter in sorted(label_by_winner.items())
        },
        "winner_by_k": {
            str(k): dict(counter)
            for k, counter in sorted(winner_by_k.items())
        },
        "representative_examples_by_winner": _representative_by_winner(combined_rows),
        "representative_by_label": _representative_by_label(combined_rows),
    }

    out_json = args.output / f"{args.stem}.json"
    save_json(combined_summary, out_json)
    print(f"Saved → {out_json}")

    out_csv = args.output / f"{args.stem}.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["field", "top_k", "winner_method", "primary_label", "count"],
        )
        writer.writeheader()
        counts: dict[tuple[str, int, str, str], int] = Counter(
            (row["field"], int(row["top_k"]), row["winner_method"], row["primary_label"])
            for row in combined_rows
        )
        for (field, top_k, winner_method, primary_label), count in sorted(counts.items()):
            writer.writerow(
                {
                    "field": field,
                    "top_k": top_k,
                    "winner_method": winner_method,
                    "primary_label": primary_label,
                    "count": count,
                }
            )
    print(f"Saved → {out_csv}")


if __name__ == "__main__":
    main()
