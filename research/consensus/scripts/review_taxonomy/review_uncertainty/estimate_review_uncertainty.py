"""Estimate uncertainty for bank-based local review win rates."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
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


from _common import save_json


def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)
    phat = successes / total
    denom = 1.0 + (z * z) / total
    center = (phat + (z * z) / (2.0 * total)) / denom
    radius = (z / denom) * math.sqrt((phat * (1.0 - phat) / total) + ((z * z) / (4.0 * total * total)))
    return (max(0.0, center - radius), min(1.0, center + radius))


def _bootstrap_interval(values: list[int], *, seed: int, n_boot: int) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    rates: list[float] = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        rates.append(sum(sample) / n)
    rates.sort()
    lo_idx = max(0, int(0.025 * (n_boot - 1)))
    hi_idx = min(n_boot - 1, int(0.975 * (n_boot - 1)))
    return (rates[lo_idx], rates[hi_idx])


def _focal_method(review_payload: dict[str, Any], method_hint: str | None) -> str:
    if method_hint:
        return method_hint
    labels = [review_payload.get("label_a", ""), review_payload.get("label_b", "")]
    for label in labels:
        if "consensus" in label:
            return label
    raise ValueError(f"Could not infer focal method from review labels: {labels}")


def _case_indicator(case: dict[str, Any], *, focal_label: str, label_a: str, label_b: str) -> int | None:
    winner = case["comparison"]["winner"]
    if winner not in {"A", "B"}:
        return None
    winner_label = label_a if winner == "A" else label_b
    return 1 if winner_label == focal_label else 0


def _display_bucket(review_path: Path, payload: dict[str, Any]) -> str:
    label = review_path.stem.replace("_rank_shift_review", "")
    label = re.sub(r"_order_balanced_gemini_v\d+$", "", label)
    label = re.sub(r"_corrected$", "", label)
    if label:
        return label
    return str(payload.get("field", review_path.stem))


def _summarize_bucket(name: str, rows: list[dict[str, Any]], *, seed: int, n_boot: int) -> dict[str, Any]:
    values = [row["indicator"] for row in rows]
    total = len(values)
    successes = sum(values)
    failures = total - successes
    win_rate = successes / total if total else 0.0
    wilson_low, wilson_high = _wilson_interval(successes, total)
    boot_low, boot_high = _bootstrap_interval(values, seed=seed, n_boot=n_boot)
    return {
        "bucket": name,
        "n_cases": total,
        "focal_wins": successes,
        "baseline_wins": failures,
        "focal_win_rate": round(win_rate, 4),
        "wilson95": [round(wilson_low, 4), round(wilson_high, 4)],
        "bootstrap95": [round(boot_low, 4), round(boot_high, 4)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review_json", nargs="+", type=Path, help="One or more *_rank_shift_review.json files")
    parser.add_argument("--focal-method", type=str, default=None, help="Optional exact label to treat as the focal method")
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-o", "--output", type=Path, default=Path("research/consensus/results/taxonomy"))
    parser.add_argument("--stem", type=str, default="review_uncertainty")
    args = parser.parse_args()

    all_rows: list[dict[str, Any]] = []
    per_review: list[dict[str, Any]] = []

    for review_path in args.review_json:
        payload = json.loads(review_path.read_text(encoding="utf-8"))
        label_a = payload["label_a"]
        label_b = payload["label_b"]
        focal_label = _focal_method(payload, args.focal_method)
        baseline_label = label_b if label_a == focal_label else label_a

        rows: list[dict[str, Any]] = []
        for case in payload.get("reviewed_cases", []):
            indicator = _case_indicator(case, focal_label=focal_label, label_a=label_a, label_b=label_b)
            if indicator is None:
                continue
            row = {
                "review_json": str(review_path),
                "field": payload["field"],
                "top_k": int(payload["top_k"]),
                "focal_label": focal_label,
                "baseline_label": baseline_label,
                "indicator": indicator,
            }
            rows.append(row)
            all_rows.append(row)

        summary = _summarize_bucket(_display_bucket(review_path, payload), rows, seed=args.seed, n_boot=args.n_boot)
        summary.update(
            {
                "review_json": str(review_path),
                "top_k": int(payload["top_k"]),
                "focal_label": focal_label,
                "baseline_label": baseline_label,
            }
        )
        per_review.append(summary)

    by_k: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_field_prefix: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        by_k[int(row["top_k"])].append(row)
        field_prefix = row["field"].split("_k", 1)[0]
        by_field_prefix[field_prefix].append(row)

    overall = _summarize_bucket("overall", all_rows, seed=args.seed, n_boot=args.n_boot)
    summary = {
        "n_review_files": len(args.review_json),
        "n_total_cases": len(all_rows),
        "per_review": per_review,
        "overall": overall,
        "by_top_k": {
            str(k): _summarize_bucket(f"top_k={k}", rows, seed=args.seed, n_boot=args.n_boot)
            for k, rows in sorted(by_k.items())
        },
        "by_field_prefix": {
            field: _summarize_bucket(field, rows, seed=args.seed, n_boot=args.n_boot)
            for field, rows in sorted(by_field_prefix.items())
        },
    }

    out_path = args.output / f"{args.stem}.json"
    save_json(summary, out_path)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
