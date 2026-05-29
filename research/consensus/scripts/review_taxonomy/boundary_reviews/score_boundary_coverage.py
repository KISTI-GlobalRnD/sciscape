"""Aggregate Protocol D v2 boundary coverage review outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
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
from sciscape.evaluation.boundary_accuracy import summarize_boundary_coverage


def _merge_reviews(cases: list[dict], reviewed_cases: list[dict]) -> list[dict]:
    reviewed_by_uid = {case["target_uid"]: case for case in reviewed_cases if case.get("target_uid")}
    merged = []
    for case in cases:
        row = dict(case)
        reviewed = reviewed_by_uid.get(case.get("target_uid"), {})
        for key in ("gold", "unary_review_a", "unary_review_b", "review_error"):
            if key in reviewed:
                row[key] = reviewed[key]
        merged.append(row)
    return merged


def _review_complete(case: dict) -> bool:
    state = case.get("coverage_state")
    if state == "both_reviewable":
        return bool(case.get("gold") or case.get("review_error"))
    if state == "A_only_reviewable":
        return bool(case.get("unary_review_a") or case.get("review_error"))
    if state == "B_only_reviewable":
        return bool(case.get("unary_review_b") or case.get("review_error"))
    return True


def _summary_if_complete(cases: list[dict], *, method_a: str, method_b: str) -> dict | None:
    if not cases or not all(_review_complete(case) for case in cases):
        return None
    return summarize_boundary_coverage(cases, method_a=method_a, method_b=method_b)


def _score_file(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    method_a = payload.get("label_a", payload.get("method_a", "A"))
    method_b = payload.get("label_b", payload.get("method_b", "B"))
    reviewed_cases = payload.get("reviewed_cases", [])
    population = _merge_reviews(payload.get("population_cases", []), reviewed_cases)
    diagnostic = _merge_reviews(payload.get("diagnostic_cases", []), reviewed_cases)
    return {
        "source_json": str(path),
        "field": payload.get("field"),
        "protocol": payload.get("protocol"),
        "budget_mode": payload.get("budget_mode"),
        "effective_k": payload.get("effective_k"),
        "top_k": payload.get("top_k"),
        "label_a": method_a,
        "label_b": method_b,
        "n_target_universe": payload.get("n_target_universe"),
        "coverage_state_counts": payload.get("coverage_state_counts", {}),
        "review_progress": {
            "population_complete": sum(1 for case in population if _review_complete(case)),
            "population_total": len(population),
            "diagnostic_complete": sum(1 for case in diagnostic if _review_complete(case)),
            "diagnostic_total": len(diagnostic),
            "reviewed_unique_cases": len(reviewed_cases),
        },
        "population": _summary_if_complete(population, method_a=method_a, method_b=method_b),
        "diagnostic": _summary_if_complete(diagnostic, method_a=method_a, method_b=method_b),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reviews", nargs="+", type=Path, help="Protocol D v2 review JSON files")
    parser.add_argument("-o", "--output", type=Path, default=Path("results/boundary_coverage_v2_summary.json"))
    args = parser.parse_args()

    rows = [_score_file(path) for path in args.reviews]
    save_json({"rows": rows}, args.output)
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
