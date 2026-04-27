"""Analyze reproducibility, difficulty, and order-bias signals across paired review runs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from _common import save_json

AMBIGUITY_TERMS = (
    "close",
    "similar",
    "both",
    "either",
    "ambig",
    "unclear",
    "slight",
    "slightly",
    "marginal",
    "minor",
    "although",
    "however",
)


def _score_gap(case: dict[str, Any]) -> int:
    comparison = case["comparison"]
    return abs(int(comparison["score_a"]) - int(comparison["score_b"]))


def _ambiguity_hits(*reasonings: str) -> int:
    text = " ".join(reasonings).lower()
    return sum(term in text for term in AMBIGUITY_TERMS)


def _mean(values: list[float | int]) -> float | None:
    if not values:
        return None
    return round(float(mean(values)), 4)


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "n": 0,
            "avg_gap_mean": None,
            "min_gap_mean": None,
            "max_gap_mean": None,
            "ambiguity_hits_mean": None,
            "share_both_runs_gap_le1": None,
            "share_any_run_tie": None,
        }
    return {
        "n": len(rows),
        "avg_gap_mean": _mean([row["avg_gap"] for row in rows]),
        "min_gap_mean": _mean([row["min_gap"] for row in rows]),
        "max_gap_mean": _mean([row["max_gap"] for row in rows]),
        "ambiguity_hits_mean": _mean([row["ambiguity_hits"] for row in rows]),
        "share_both_runs_gap_le1": round(
            sum(1 for row in rows if row["old_gap"] <= 1 and row["new_gap"] <= 1) / len(rows),
            4,
        ),
        "share_any_run_tie": round(
            sum(1 for row in rows if row["old_gap"] == 0 or row["new_gap"] == 0) / len(rows),
            4,
        ),
    }


def _load_review(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("reviewed_cases", [])


def _analyze_pair(label: str, old_path: Path, new_path: Path) -> dict[str, Any]:
    old_cases = _load_review(old_path)
    new_cases = _load_review(new_path)
    if len(old_cases) != len(new_cases):
        raise ValueError(f"Review pair length mismatch for {label}: {len(old_cases)} vs {len(new_cases)}")

    old_uids = [case["target_uid"] for case in old_cases]
    new_uids = [case["target_uid"] for case in new_cases]
    if old_uids != new_uids:
        raise ValueError(f"Review pair target order mismatch for {label}")

    rows: list[dict[str, Any]] = []
    order_bias_rows: list[dict[str, Any]] = []
    category_counts: Counter[str] = Counter()

    for old_case, new_case in zip(old_cases, new_cases):
        old_cmp = old_case["comparison"]
        new_cmp = new_case["comparison"]
        old_gap = _score_gap(old_case)
        new_gap = _score_gap(new_case)
        flipped = old_cmp["winner"] != new_cmp["winner"]
        same_presented_winner = old_cmp.get("presented_winner") == new_cmp.get("presented_winner")
        opposite_swap = bool(old_cmp.get("swapped")) != bool(new_cmp.get("swapped"))
        order_bias_suspect = flipped and same_presented_winner and opposite_swap
        row = {
            "target_uid": old_case["target_uid"],
            "old_winner": old_cmp["winner"],
            "new_winner": new_cmp["winner"],
            "old_gap": old_gap,
            "new_gap": new_gap,
            "avg_gap": round((old_gap + new_gap) / 2.0, 4),
            "min_gap": min(old_gap, new_gap),
            "max_gap": max(old_gap, new_gap),
            "old_swapped": bool(old_cmp.get("swapped")),
            "new_swapped": bool(new_cmp.get("swapped")),
            "old_presented_winner": old_cmp.get("presented_winner"),
            "new_presented_winner": new_cmp.get("presented_winner"),
            "same_presented_winner": same_presented_winner,
            "opposite_swap": opposite_swap,
            "flip": flipped,
            "order_bias_suspect": order_bias_suspect,
            "ambiguity_hits": _ambiguity_hits(old_cmp.get("reasoning", ""), new_cmp.get("reasoning", "")),
        }
        rows.append(row)
        if order_bias_suspect:
            order_bias_rows.append(row)
        if flipped:
            category_counts[
                f"same_presented={same_presented_winner}|opposite_swap={opposite_swap}|"
                f"presented={old_cmp.get('presented_winner')}->{new_cmp.get('presented_winner')}"
            ] += 1

    flips = [row for row in rows if row["flip"]]
    stable = [row for row in rows if not row["flip"]]
    return {
        "label": label,
        "old_review_json": str(old_path),
        "new_review_json": str(new_path),
        "n_cases": len(rows),
        "winner_agreement_rate": round(len(stable) / len(rows), 4) if rows else None,
        "flip_summary": {
            "n_flips": len(flips),
            "n_order_bias_suspect": len(order_bias_rows),
            "share_order_bias_suspect_among_flips": (
                round(len(order_bias_rows) / len(flips), 4) if flips else None
            ),
            "flip_categories": dict(category_counts),
        },
        "difficulty_summary": {
            "flips": _summarize_rows(flips),
            "stable": _summarize_rows(stable),
        },
        "order_bias_suspect_cases": order_bias_rows,
        "flipped_cases": flips,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair",
        action="append",
        required=True,
        help="label|old_review_json|new_review_json",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("research/consensus/results/taxonomy_corrected/review_reproducibility_audit.json"),
    )
    args = parser.parse_args()

    analyses = []
    for spec in args.pair:
        label, old_raw, new_raw = [part.strip() for part in spec.split("|", 2)]
        analyses.append(_analyze_pair(label, Path(old_raw), Path(new_raw)))

    save_json({"pairs": analyses}, args.output)
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
