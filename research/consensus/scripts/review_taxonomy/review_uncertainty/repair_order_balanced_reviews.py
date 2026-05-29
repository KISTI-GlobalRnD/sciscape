"""Repair order-balanced review outputs after comparison winner logic changes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import sys

REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").exists()
)
SCRIPT_ROOT = REPO_ROOT / "research/consensus/scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# Add project root to path for direct script execution.

from _common import save_json


def _mean(values: list[float | int]) -> float | None:
    if not values:
        return None
    return round(sum(float(value) for value in values) / len(values), 4)


def _resolve_comparison_winner(winner: str, *, score_a: float, score_b: float) -> str:
    if winner in {"A", "B", "TIE"}:
        return winner
    if score_a > score_b:
        return "A"
    if score_b > score_a:
        return "B"
    return "TIE"


def _winner_from_presented_labels(
    presented_winner: str,
    *,
    presented_method_a: str,
    presented_method_b: str,
    original_method_a: str,
    original_method_b: str,
) -> str:
    if presented_winner == "TIE":
        return "TIE"
    if presented_winner == "A":
        winner_method = presented_method_a
    elif presented_winner == "B":
        winner_method = presented_method_b
    else:
        return ""
    if winner_method == original_method_a:
        return "A"
    if winner_method == original_method_b:
        return "B"
    return ""


def _normalize_balanced_pass(
    pass_payload: dict[str, Any],
    *,
    original_method_a: str,
    original_method_b: str,
) -> dict[str, Any]:
    presented_method_a = str(pass_payload.get("presented_method_a", ""))
    presented_method_b = str(pass_payload.get("presented_method_b", ""))
    presented_score_a = float(pass_payload.get("presented_score_a", 0))
    presented_score_b = float(pass_payload.get("presented_score_b", 0))
    presented_winner = str(pass_payload.get("presented_winner", "")).upper()

    if presented_method_a == original_method_a and presented_method_b == original_method_b:
        score_a = presented_score_a
        score_b = presented_score_b
    elif presented_method_a == original_method_b and presented_method_b == original_method_a:
        score_a = presented_score_b
        score_b = presented_score_a
    else:
        score_a = float(pass_payload.get("score_a", 0))
        score_b = float(pass_payload.get("score_b", 0))

    winner = _winner_from_presented_labels(
        presented_winner,
        presented_method_a=presented_method_a,
        presented_method_b=presented_method_b,
        original_method_a=original_method_a,
        original_method_b=original_method_b,
    )
    winner = _resolve_comparison_winner(winner, score_a=score_a, score_b=score_b)

    normalized = dict(pass_payload)
    normalized["winner"] = winner
    normalized["score_a"] = score_a
    normalized["score_b"] = score_b
    normalized["method_a"] = original_method_a
    normalized["method_b"] = original_method_b
    return normalized


def _summarize_reviews(reviewed_cases: list[dict[str, Any]], *, method_a: str, method_b: str) -> dict[str, Any]:
    votes = [case["comparison"]["winner"] for case in reviewed_cases]
    method_a_wins = sum(1 for vote in votes if vote == "A")
    method_b_wins = sum(1 for vote in votes if vote == "B")
    valid = method_a_wins + method_b_wins
    n_cases = len(reviewed_cases)
    scores_a = [case["comparison"]["score_a"] for case in reviewed_cases]
    scores_b = [case["comparison"]["score_b"] for case in reviewed_cases]
    shift_scores = [case["shift_score"] for case in reviewed_cases]
    rank_jaccards = [case["rank_jaccard"] for case in reviewed_cases]
    mean_shifts = [case["mean_abs_rank_shift"] for case in reviewed_cases]
    cluster_overlaps = [
        case["cluster_overlap_coeff"]
        for case in reviewed_cases
        if case.get("cluster_overlap_coeff") is not None
    ]
    cluster_changed = [1 if case["cluster_changed"] else 0 for case in reviewed_cases]
    return {
        "n_reviewed_cases": n_cases,
        "comparison": {
            "method_a": method_a,
            "method_b": method_b,
            "method_a_wins": method_a_wins,
            "method_b_wins": method_b_wins,
            "ties_or_invalid": n_cases - valid,
            "method_a_win_rate": round(method_a_wins / n_cases, 4) if n_cases else None,
            "method_b_win_rate": round(method_b_wins / n_cases, 4) if n_cases else None,
            "method_a_win_rate_no_ties": round(method_a_wins / valid, 4) if valid else None,
            "method_b_win_rate_no_ties": round(method_b_wins / valid, 4) if valid else None,
            "score_a_mean": _mean(scores_a),
            "score_b_mean": _mean(scores_b),
            "score_gap_mean": _mean([a - b for a, b in zip(scores_a, scores_b)]),
        },
        "sample_stats": {
            "shift_score_mean": _mean(shift_scores),
            "rank_jaccard_mean": _mean(rank_jaccards),
            "mean_abs_rank_shift_mean": _mean(mean_shifts),
            "cluster_overlap_coeff_mean": _mean(cluster_overlaps),
            "cluster_changed_rate": _mean(cluster_changed),
        },
    }


def _repair_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
    repaired = json.loads(json.dumps(payload))
    method_a = str(repaired["label_a"])
    method_b = str(repaired["label_b"])
    changed_cases = 0

    for case in repaired.get("reviewed_cases", []):
        comparison = case.get("comparison", {})
        if comparison.get("order_balance_mode") != "dual_pass":
            continue
        passes = comparison.get("balanced_passes", [])
        if len(passes) != 2:
            continue

        normalized_passes = [
            _normalize_balanced_pass(
                pass_payload,
                original_method_a=method_a,
                original_method_b=method_b,
            )
            for pass_payload in passes
        ]

        stable = normalized_passes[0]["winner"] == normalized_passes[1]["winner"]
        repaired_winner = normalized_passes[0]["winner"] if stable else "TIE"
        repaired_reasoning = (
            normalized_passes[0].get("reasoning", "")
            if stable
            else (
                "Order-balanced passes disagreed across presentation order; "
                "the comparison is treated conservatively as a tie."
            )
        )
        repaired_score_a = _mean([row["score_a"] for row in normalized_passes])
        repaired_score_b = _mean([row["score_b"] for row in normalized_passes])

        if comparison.get("winner") != repaired_winner:
            changed_cases += 1

        comparison["balanced_passes"] = normalized_passes
        comparison["winner"] = repaired_winner
        comparison["reasoning"] = repaired_reasoning
        comparison["score_a"] = repaired_score_a
        comparison["score_b"] = repaired_score_b
        comparison["order_sensitive"] = not stable
        comparison["order_balance_mode"] = "dual_pass"

    repaired["summary"] = _summarize_reviews(
        repaired.get("reviewed_cases", []),
        method_a=method_a,
        method_b=method_b,
    )
    repaired["repair_metadata"] = {
        "repair": "order_balanced_winner_logic",
        "changed_cases": changed_cases,
    }
    return repaired, changed_cases


def _default_output_path(review_path: Path, *, output_dir: Path | None, suffix: str) -> Path:
    target_dir = output_dir or review_path.parent
    return target_dir / f"{review_path.stem}{suffix}.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair saved dual-pass review outputs after winner-logic changes")
    parser.add_argument("review_json", nargs="+", type=Path, help="One or more order-balanced *_rank_shift_review.json files")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for repaired files (default: alongside input)")
    parser.add_argument("--suffix", type=str, default="_winner_fix", help="Suffix inserted before .json in repaired outputs")
    args = parser.parse_args()

    for review_path in args.review_json:
        payload = json.loads(review_path.read_text(encoding="utf-8"))
        repaired, changed_cases = _repair_payload(payload)
        out_path = _default_output_path(review_path, output_dir=args.output_dir, suffix=args.suffix)
        save_json(repaired, out_path)
        before = payload.get("summary", {}).get("comparison", {})
        after = repaired.get("summary", {}).get("comparison", {})
        print(
            f"{review_path.name} -> {out_path.name}: "
            f"changed_cases={changed_cases}, "
            f"{before.get('method_a_wins')}:{before.get('method_b_wins')}:{before.get('ties_or_invalid')} -> "
            f"{after.get('method_a_wins')}:{after.get('method_b_wins')}:{after.get('ties_or_invalid')}"
        )


if __name__ == "__main__":
    main()
