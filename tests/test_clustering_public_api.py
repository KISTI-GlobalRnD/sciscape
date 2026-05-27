"""Tests for the stable clustering package API."""

import sciscape.clustering as clustering
import sciscape.clustering.hierarchy_oversize_postprocess as oversize_postprocess
import sciscape.clustering.hierarchy_postprocess as legacy_postprocess
import sciscape.clustering.leiden_rust as leiden_rust


def test_dongdaemun_helpers_are_not_public_clustering_api():
    """Dongdaemun helpers stay on explicit development modules."""

    hidden_names = {
        "RUST_DONGDAEMUN_AVAILABLE",
        "RUST_DONGDAEMUN_REFINEMENT_AVAILABLE",
        "RustDongdaemunAudit",
        "RustDongdaemunAutoFastResult",
        "RustDongdaemunRefinementAudit",
        "RustDongdaemunRefinementResult",
        "RustDongdaemunResult",
        "dongdaemun_refine_rust",
    }

    assert hidden_names.isdisjoint(set(clustering.__all__))
    assert hidden_names.isdisjoint(set(leiden_rust.__all__))
    for name in hidden_names:
        assert not hasattr(clustering, name)


def test_rust_leiden_helpers_remain_public_clustering_api():
    assert "RUST_AVAILABLE" in clustering.__all__
    assert "RustLeidenGraph" in clustering.__all__
    assert "RustLeidenResult" in clustering.__all__
    assert "run_leiden_rust" in clustering.__all__
    assert "postprocess_small_clusters_rust" in clustering.__all__


def test_hierarchy_oversize_postprocess_is_canonical_alias():
    assert (
        oversize_postprocess.HierarchyPostprocessConfig.__module__
        == "sciscape.clustering.hierarchy_oversize_postprocess"
    )
    assert (
        oversize_postprocess.HierarchyPostprocessConfig
        is legacy_postprocess.HierarchyPostprocessConfig
    )
    assert (
        oversize_postprocess.run_hierarchy_level_postprocess
        is legacy_postprocess.run_hierarchy_level_postprocess
    )
    assert "run_hierarchy_level_postprocess" in oversize_postprocess.__all__
