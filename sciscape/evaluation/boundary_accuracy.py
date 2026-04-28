"""Helpers for gold-label boundary accuracy evaluation."""

from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Any

VALID_BOUNDARY_DECISIONS = ("A_ONLY", "B_ONLY", "BOTH", "NEITHER", "UNCLEAR")
VALID_PLAUSIBILITY_DECISIONS = ("PLAUSIBLE", "NOT_PLAUSIBLE", "UNCLEAR")


def _mean(values: list[float | int]) -> float | None:
    if not values:
        return None
    return round(float(mean(values)), 4)


def parse_boolish(value: Any) -> bool | None:
    """Normalize common yes/no encodings to ``bool``."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    raw = str(value).strip().upper()
    if raw in {"YES", "Y", "TRUE", "T", "1"}:
        return True
    if raw in {"NO", "N", "FALSE", "F", "0"}:
        return False
    return None


def normalize_boundary_decision(
    decision: Any,
    *,
    belongs_with_a: Any = None,
    belongs_with_b: Any = None,
) -> str:
    """Return a stable boundary decision label."""
    raw = str(decision).strip().upper() if decision is not None else ""
    if raw in VALID_BOUNDARY_DECISIONS:
        return raw
    a = parse_boolish(belongs_with_a)
    b = parse_boolish(belongs_with_b)
    if a is True and b is False:
        return "A_ONLY"
    if a is False and b is True:
        return "B_ONLY"
    if a is True and b is True:
        return "BOTH"
    if a is False and b is False:
        return "NEITHER"
    return "UNCLEAR"


def score_boundary_decision(decision: str) -> dict[str, int | None]:
    """Score whether method A / B matches a gold boundary decision."""
    normalized = normalize_boundary_decision(decision)
    if normalized == "A_ONLY":
        return {"method_a_correct": 1, "method_b_correct": 0}
    if normalized == "B_ONLY":
        return {"method_a_correct": 0, "method_b_correct": 1}
    if normalized == "BOTH":
        return {"method_a_correct": 1, "method_b_correct": 1}
    if normalized == "NEITHER":
        return {"method_a_correct": 0, "method_b_correct": 0}
    return {"method_a_correct": None, "method_b_correct": None}


def normalize_plausibility_decision(decision: Any) -> str:
    """Return a stable unary plausibility label."""
    raw = str(decision).strip().upper() if decision is not None else ""
    if raw in VALID_PLAUSIBILITY_DECISIONS:
        return raw
    if raw in {"YES", "Y", "TRUE", "T", "1"}:
        return "PLAUSIBLE"
    if raw in {"NO", "N", "FALSE", "F", "0"}:
        return "NOT_PLAUSIBLE"
    return "UNCLEAR"


def score_plausibility_decision(decision: Any) -> int | None:
    """Score a unary plausibility decision as 1/0/unknown."""
    normalized = normalize_plausibility_decision(decision)
    if normalized == "PLAUSIBLE":
        return 1
    if normalized == "NOT_PLAUSIBLE":
        return 0
    return None


def _rate(numerator: int | float, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(float(numerator) / float(denominator), 4)


def summarize_boundary_accuracy(
    reviewed_cases: list[dict[str, Any]],
    *,
    method_a: str,
    method_b: str,
    gold_key: str = "gold",
) -> dict[str, Any]:
    """Summarize method-vs-gold accuracy on boundary adjudication cases."""
    decisions = [
        normalize_boundary_decision(
            case.get(gold_key, {}).get("decision"),
            belongs_with_a=case.get(gold_key, {}).get("belongs_with_a"),
            belongs_with_b=case.get(gold_key, {}).get("belongs_with_b"),
        )
        for case in reviewed_cases
    ]
    counts = Counter(decisions)
    scored = [score_boundary_decision(decision) for decision in decisions]
    scored_rows = [row for row in scored if row["method_a_correct"] is not None]
    a_scores = [int(row["method_a_correct"]) for row in scored_rows]
    b_scores = [int(row["method_b_correct"]) for row in scored_rows]
    a_better = sum(1 for row in scored_rows if row["method_a_correct"] > row["method_b_correct"])
    b_better = sum(1 for row in scored_rows if row["method_b_correct"] > row["method_a_correct"])
    both_correct = sum(1 for row in scored_rows if row["method_a_correct"] == row["method_b_correct"] == 1)
    both_wrong = sum(1 for row in scored_rows if row["method_a_correct"] == row["method_b_correct"] == 0)
    n_cases = len(reviewed_cases)
    n_scored = len(scored_rows)
    return {
        "n_reviewed_cases": n_cases,
        "n_scored_cases": n_scored,
        "n_unclear_cases": counts.get("UNCLEAR", 0),
        "decision_counts": {label: counts.get(label, 0) for label in VALID_BOUNDARY_DECISIONS},
        "accuracy": {
            "method_a": method_a,
            "method_b": method_b,
            "method_a_accuracy": round(sum(a_scores) / n_scored, 4) if n_scored else None,
            "method_b_accuracy": round(sum(b_scores) / n_scored, 4) if n_scored else None,
            "method_a_correct": sum(a_scores),
            "method_b_correct": sum(b_scores),
            "method_a_advantage_cases": a_better,
            "method_b_advantage_cases": b_better,
            "both_correct_cases": both_correct,
            "both_wrong_cases": both_wrong,
            "net_accuracy_gap": _mean([a - b for a, b in zip(a_scores, b_scores)]),
        },
    }


def summarize_boundary_accuracy_views(
    reviewed_cases: list[dict[str, Any]],
    *,
    method_a: str,
    method_b: str,
    gold_key: str = "gold",
) -> dict[str, Any]:
    """Summarize binary boundary accuracy under common exclusion views."""
    decisions = [
        normalize_boundary_decision(
            case.get(gold_key, {}).get("decision"),
            belongs_with_a=case.get(gold_key, {}).get("belongs_with_a"),
            belongs_with_b=case.get(gold_key, {}).get("belongs_with_b"),
        )
        for case in reviewed_cases
    ]
    full = summarize_boundary_accuracy(reviewed_cases, method_a=method_a, method_b=method_b, gold_key=gold_key)
    excluding_neither = [
        case for case, decision in zip(reviewed_cases, decisions)
        if decision != "NEITHER"
    ]
    excluding_neither_unclear = [
        case for case, decision in zip(reviewed_cases, decisions)
        if decision not in {"NEITHER", "UNCLEAR"}
    ]
    discriminative = [
        case for case, decision in zip(reviewed_cases, decisions)
        if decision in {"A_ONLY", "B_ONLY"}
    ]
    return {
        "full": full,
        "excluding_neither": summarize_boundary_accuracy(
            excluding_neither,
            method_a=method_a,
            method_b=method_b,
            gold_key=gold_key,
        ),
        "excluding_neither_unclear": summarize_boundary_accuracy(
            excluding_neither_unclear,
            method_a=method_a,
            method_b=method_b,
            gold_key=gold_key,
        ),
        "discriminative_A_ONLY_B_ONLY": summarize_boundary_accuracy(
            discriminative,
            method_a=method_a,
            method_b=method_b,
            gold_key=gold_key,
        ),
    }


def _case_binary_credit(case: dict[str, Any], gold_key: str) -> tuple[int | None, int | None, str]:
    decision = normalize_boundary_decision(
        case.get(gold_key, {}).get("decision"),
        belongs_with_a=case.get(gold_key, {}).get("belongs_with_a"),
        belongs_with_b=case.get(gold_key, {}).get("belongs_with_b"),
    )
    score = score_boundary_decision(decision)
    return score["method_a_correct"], score["method_b_correct"], decision


def _case_unary_credit(case: dict[str, Any], key: str) -> tuple[int | None, str]:
    review = case.get(key, {})
    decision = normalize_plausibility_decision(review.get("decision"))
    return score_plausibility_decision(decision), decision


def _mean_credit(values: list[int | None], *, unclear_as_zero: bool) -> float | None:
    if unclear_as_zero:
        return _rate(sum(0 if value is None else value for value in values), len(values))
    scored = [int(value) for value in values if value is not None]
    return _rate(sum(scored), len(scored))


def summarize_boundary_coverage(
    cases: list[dict[str, Any]],
    *,
    method_a: str,
    method_b: str,
    gold_key: str = "gold",
    unary_a_key: str = "unary_review_a",
    unary_b_key: str = "unary_review_b",
) -> dict[str, Any]:
    """Summarize coverage-aware Protocol D v2 reviewed cases."""
    counts = Counter(case.get("coverage_state", "unknown") for case in cases)
    n_cases = len(cases)
    binary_cases = [case for case in cases if case.get("coverage_state") == "both_reviewable" and case.get(gold_key)]
    a_credits: list[int | None] = []
    b_credits: list[int | None] = []
    binary_decisions: list[str] = []
    unary_counts = Counter()

    for case in cases:
        state = case.get("coverage_state")
        if state == "both_reviewable" and case.get(gold_key):
            a_credit, b_credit, decision = _case_binary_credit(case, gold_key)
            a_credits.append(a_credit)
            b_credits.append(b_credit)
            binary_decisions.append(decision)
            continue
        if state == "A_only_reviewable":
            a_credit, decision = _case_unary_credit(case, unary_a_key)
            a_credits.append(a_credit)
            b_credits.append(0)
            unary_counts[f"A_{decision}"] += 1
            continue
        if state == "B_only_reviewable":
            b_credit, decision = _case_unary_credit(case, unary_b_key)
            a_credits.append(0)
            b_credits.append(b_credit)
            unary_counts[f"B_{decision}"] += 1
            continue
        a_credits.append(0)
        b_credits.append(0)

    a_reviewable = counts.get("A_only_reviewable", 0) + counts.get("both_reviewable", 0)
    b_reviewable = counts.get("B_only_reviewable", 0) + counts.get("both_reviewable", 0)
    binary_counts = Counter(binary_decisions)
    return {
        "n_cases": n_cases,
        "coverage_state_counts": {
            "A_only_reviewable": counts.get("A_only_reviewable", 0),
            "B_only_reviewable": counts.get("B_only_reviewable", 0),
            "both_reviewable": counts.get("both_reviewable", 0),
            "neither_reviewable": counts.get("neither_reviewable", 0),
        },
        "coverage_rate": {
            method_a: _rate(a_reviewable, n_cases),
            method_b: _rate(b_reviewable, n_cases),
        },
        "reviewable_rate": {
            "any_reviewable": _rate(n_cases - counts.get("neither_reviewable", 0), n_cases),
            "both_reviewable": _rate(counts.get("both_reviewable", 0), n_cases),
        },
        "plausible_coverage_rate": {
            "including_unclear_as_zero": {
                method_a: _mean_credit(a_credits, unclear_as_zero=True),
                method_b: _mean_credit(b_credits, unclear_as_zero=True),
            },
            "excluding_unclear": {
                method_a: _mean_credit(a_credits, unclear_as_zero=False),
                method_b: _mean_credit(b_credits, unclear_as_zero=False),
            },
        },
        "overall_boundary_utility": {
            "including_unclear_as_zero": {
                method_a: _mean_credit(a_credits, unclear_as_zero=True),
                method_b: _mean_credit(b_credits, unclear_as_zero=True),
            },
            "excluding_unclear": {
                method_a: _mean_credit(a_credits, unclear_as_zero=False),
                method_b: _mean_credit(b_credits, unclear_as_zero=False),
            },
        },
        "conditional_boundary_accuracy": summarize_boundary_accuracy_views(
            binary_cases,
            method_a=method_a,
            method_b=method_b,
            gold_key=gold_key,
        ),
        "neither_rate": _rate(binary_counts.get("NEITHER", 0), len(binary_decisions)),
        "unclear_rate": _rate(binary_counts.get("UNCLEAR", 0), len(binary_decisions)),
        "unary_decision_counts": dict(unary_counts),
    }
