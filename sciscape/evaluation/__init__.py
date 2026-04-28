"""Cluster quality evaluation tools.

- Worst-case sampling: pick boundary/hard-case nodes for evaluation
- LLM reviewer: blind comparison of cluster cohesion
"""

from .sampler import (
    sample_worst_case,
    sample_disagreement_cases,
    collect_rank_shift_cases,
    sample_rank_shift_cases,
    collect_boundary_coverage_cases,
    sample_boundary_coverage_cases,
    SampleCase,
    SampleSet,
    DisagreementCase,
    DisagreementSampleSet,
    RankShiftCase,
    RankShiftSampleSet,
    BoundaryCoverageCase,
    BoundaryCoverageSampleSet,
)
from .reviewer import (
    review_cluster, review_comparison, review_belonging, review_boundary_gold,
    review_boundary_plausibility, review_group_cohesion, review_outliers, review_neighbor_rerank,
    review_neighbor_rerank_order_balanced, classify_case_taxonomy,
    ReviewResult, ComparisonResult, BelongingResult, BoundaryGoldResult, BoundaryPlausibilityResult,
    GroupCohesionResult, OutlierResult, TaxonomyResult,
)
from .boundary_accuracy import (
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

__all__ = [
    "sample_worst_case", "sample_disagreement_cases", "collect_rank_shift_cases", "sample_rank_shift_cases",
    "collect_boundary_coverage_cases", "sample_boundary_coverage_cases",
    "SampleCase", "SampleSet", "DisagreementCase", "DisagreementSampleSet", "RankShiftCase", "RankShiftSampleSet",
    "BoundaryCoverageCase", "BoundaryCoverageSampleSet",
    "review_cluster", "review_comparison", "review_belonging", "review_boundary_gold",
    "review_boundary_plausibility", "review_neighbor_rerank",
    "review_neighbor_rerank_order_balanced", "classify_case_taxonomy",
    "review_group_cohesion", "review_outliers",
    "ReviewResult", "ComparisonResult", "BelongingResult", "BoundaryGoldResult", "BoundaryPlausibilityResult",
    "GroupCohesionResult", "OutlierResult", "TaxonomyResult",
    "VALID_BOUNDARY_DECISIONS", "VALID_PLAUSIBILITY_DECISIONS",
    "normalize_boundary_decision", "normalize_plausibility_decision",
    "score_boundary_decision", "score_plausibility_decision",
    "summarize_boundary_accuracy", "summarize_boundary_accuracy_views", "summarize_boundary_coverage",
]
