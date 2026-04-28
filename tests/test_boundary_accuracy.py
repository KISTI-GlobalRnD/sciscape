from sciscape.evaluation import (
    VALID_BOUNDARY_DECISIONS,
    VALID_PLAUSIBILITY_DECISIONS,
    normalize_boundary_decision,
    normalize_plausibility_decision,
    score_boundary_decision,
    score_plausibility_decision,
    summarize_boundary_accuracy,
    summarize_boundary_accuracy_views,
    summarize_boundary_coverage,
)


def test_boundary_decision_normalization_from_flags():
    assert normalize_boundary_decision("", belongs_with_a=True, belongs_with_b=False) == "A_ONLY"
    assert normalize_boundary_decision("", belongs_with_a=False, belongs_with_b=True) == "B_ONLY"
    assert normalize_boundary_decision("", belongs_with_a=True, belongs_with_b=True) == "BOTH"
    assert normalize_boundary_decision("", belongs_with_a=False, belongs_with_b=False) == "NEITHER"
    assert normalize_boundary_decision("", belongs_with_a=None, belongs_with_b=False) == "UNCLEAR"


def test_score_boundary_decision():
    assert score_boundary_decision("A_ONLY") == {"method_a_correct": 1, "method_b_correct": 0}
    assert score_boundary_decision("B_ONLY") == {"method_a_correct": 0, "method_b_correct": 1}
    assert score_boundary_decision("BOTH") == {"method_a_correct": 1, "method_b_correct": 1}
    assert score_boundary_decision("NEITHER") == {"method_a_correct": 0, "method_b_correct": 0}
    assert score_boundary_decision("UNCLEAR") == {"method_a_correct": None, "method_b_correct": None}


def test_plausibility_decision_normalization_and_scoring():
    assert normalize_plausibility_decision("yes") == "PLAUSIBLE"
    assert normalize_plausibility_decision("no") == "NOT_PLAUSIBLE"
    assert normalize_plausibility_decision("maybe") == "UNCLEAR"
    assert score_plausibility_decision("PLAUSIBLE") == 1
    assert score_plausibility_decision("NOT_PLAUSIBLE") == 0
    assert score_plausibility_decision("UNCLEAR") is None


def test_summarize_boundary_accuracy():
    reviewed_cases = [
        {"gold": {"decision": "A_ONLY"}},
        {"gold": {"decision": "B_ONLY"}},
        {"gold": {"decision": "BOTH"}},
        {"gold": {"decision": "NEITHER"}},
        {"gold": {"decision": "UNCLEAR"}},
    ]
    summary = summarize_boundary_accuracy(reviewed_cases, method_a="A", method_b="B")
    assert summary["n_reviewed_cases"] == 5
    assert summary["n_scored_cases"] == 4
    assert summary["n_unclear_cases"] == 1
    assert summary["decision_counts"]["A_ONLY"] == 1
    assert summary["decision_counts"]["BOTH"] == 1
    assert summary["accuracy"]["method_a_correct"] == 2
    assert summary["accuracy"]["method_b_correct"] == 2
    assert summary["accuracy"]["both_correct_cases"] == 1
    assert summary["accuracy"]["both_wrong_cases"] == 1


def test_summarize_boundary_accuracy_views():
    reviewed_cases = [
        {"gold": {"decision": "A_ONLY"}},
        {"gold": {"decision": "B_ONLY"}},
        {"gold": {"decision": "BOTH"}},
        {"gold": {"decision": "NEITHER"}},
        {"gold": {"decision": "UNCLEAR"}},
    ]
    views = summarize_boundary_accuracy_views(reviewed_cases, method_a="A", method_b="B")
    assert views["full"]["n_reviewed_cases"] == 5
    assert views["excluding_neither"]["n_reviewed_cases"] == 4
    assert views["excluding_neither_unclear"]["n_reviewed_cases"] == 3
    assert views["discriminative_A_ONLY_B_ONLY"]["n_reviewed_cases"] == 2


def test_summarize_boundary_coverage():
    cases = [
        {"coverage_state": "both_reviewable", "gold": {"decision": "A_ONLY"}},
        {"coverage_state": "both_reviewable", "gold": {"decision": "NEITHER"}},
        {"coverage_state": "A_only_reviewable", "unary_review_a": {"decision": "PLAUSIBLE"}},
        {"coverage_state": "B_only_reviewable", "unary_review_b": {"decision": "NOT_PLAUSIBLE"}},
        {"coverage_state": "neither_reviewable"},
    ]

    summary = summarize_boundary_coverage(cases, method_a="A", method_b="B")

    assert summary["coverage_state_counts"]["both_reviewable"] == 2
    assert summary["coverage_rate"]["A"] == 0.6
    assert summary["coverage_rate"]["B"] == 0.6
    assert summary["reviewable_rate"]["any_reviewable"] == 0.8
    assert summary["overall_boundary_utility"]["including_unclear_as_zero"]["A"] == 0.4
    assert summary["overall_boundary_utility"]["including_unclear_as_zero"]["B"] == 0.0
    assert summary["conditional_boundary_accuracy"]["full"]["decision_counts"]["NEITHER"] == 1
    assert summary["neither_rate"] == 0.5


def test_boundary_helpers_exported():
    assert "A_ONLY" in VALID_BOUNDARY_DECISIONS
    assert "PLAUSIBLE" in VALID_PLAUSIBILITY_DECISIONS
