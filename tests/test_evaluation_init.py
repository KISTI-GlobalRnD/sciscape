from sciscape.evaluation import (
    SampleCase,
    TaxonomyResult,
    classify_case_taxonomy,
    collect_rank_shift_cases,
    review_neighbor_rerank_order_balanced,
)


def test_evaluation_namespace_exports_newer_helpers():
    assert SampleCase is not None
    assert collect_rank_shift_cases is not None
    assert review_neighbor_rerank_order_balanced is not None
    assert classify_case_taxonomy is not None
    assert TaxonomyResult is not None
