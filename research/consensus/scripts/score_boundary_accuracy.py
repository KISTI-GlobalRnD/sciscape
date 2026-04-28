"""Score method-vs-gold boundary accuracy from reviewed Protocol D outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import save_json
from sciscape.evaluation.boundary_accuracy import summarize_boundary_accuracy


def _score_file(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = summarize_boundary_accuracy(
        payload.get("reviewed_cases", []),
        method_a=payload.get("label_a", payload.get("method_a", "A")),
        method_b=payload.get("label_b", payload.get("method_b", "B")),
    )
    return {
        "source_json": str(path),
        "field": payload.get("field"),
        "protocol": payload.get("protocol"),
        "budget_mode": payload.get("budget_mode"),
        "effective_k": payload.get("effective_k"),
        "top_k": payload.get("top_k"),
        "label_a": payload.get("label_a"),
        "label_b": payload.get("label_b"),
        **summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reviews", nargs="+", type=Path, help="Boundary accuracy review JSON files")
    parser.add_argument("-o", "--output", type=Path, default=Path("results/boundary_accuracy_summary.json"))
    args = parser.parse_args()

    rows = [_score_file(path) for path in args.reviews]
    save_json({"rows": rows}, args.output)
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
