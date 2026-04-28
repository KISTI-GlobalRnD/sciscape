from sciscape.evaluation import (
    BoundaryCoverageCase,
    SampleCase,
    TaxonomyResult,
    normalize_boundary_decision,
    classify_case_taxonomy,
    collect_boundary_coverage_cases,
    collect_rank_shift_cases,
    review_boundary_gold,
    review_boundary_plausibility,
    review_neighbor_rerank_order_balanced,
    summarize_boundary_accuracy,
    summarize_boundary_coverage,
)


def test_evaluation_namespace_exports_newer_helpers():
    assert SampleCase is not None
    assert BoundaryCoverageCase is not None
    assert collect_rank_shift_cases is not None
    assert collect_boundary_coverage_cases is not None
    assert review_neighbor_rerank_order_balanced is not None
    assert review_boundary_gold is not None
    assert review_boundary_plausibility is not None
    assert classify_case_taxonomy is not None
    assert TaxonomyResult is not None
    assert normalize_boundary_decision is not None
    assert summarize_boundary_accuracy is not None
    assert summarize_boundary_coverage is not None
