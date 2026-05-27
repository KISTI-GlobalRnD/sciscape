"""Tests for cyclic qf-guarded hierarchy postprocess orchestration."""

from __future__ import annotations

import numpy as np
import pytest

from sciscape.clustering.hierarchy_postprocess import (
    CyclicPostprocessConfig,
    CyclicPostprocessState,
    HierarchyPostprocessConfig,
    LevelPostprocessResult,
    RefinementCheckpoint,
    cyclic_postprocess_trigger_reasons,
    run_cyclic_postprocess_if_due,
)


def test_cyclic_trigger_respects_warmup_oversize_and_cooldown():
    config = CyclicPostprocessConfig(
        enabled=True,
        warmup_steps=2,
        interval_steps=2,
        plateau_window=0,
        cooldown_steps=2,
        require_oversize=True,
    )

    assert (
        cyclic_postprocess_trigger_reasons(
            [RefinementCheckpoint(step=1, quality=10.0, n_above_max_doc_weight=1)],
            config=config,
        )
        == []
    )
    assert (
        cyclic_postprocess_trigger_reasons(
            [RefinementCheckpoint(step=2, quality=10.0, n_above_max_doc_weight=0)],
            config=config,
        )
        == []
    )
    assert cyclic_postprocess_trigger_reasons(
        [RefinementCheckpoint(step=2, quality=10.0, n_above_max_doc_weight=1)],
        config=config,
    ) == ["interval"]
    assert (
        cyclic_postprocess_trigger_reasons(
            [RefinementCheckpoint(step=3, quality=11.0, n_above_max_doc_weight=1)],
            config=config,
            state=CyclicPostprocessState(n_calls=1, last_call_step=2),
        )
        == []
    )
    assert cyclic_postprocess_trigger_reasons(
        [RefinementCheckpoint(step=4, quality=12.0, n_above_max_doc_weight=1)],
        config=config,
        state=CyclicPostprocessState(n_calls=1, last_call_step=2),
    ) == ["interval"]


def test_cyclic_trigger_detects_quality_plateau_and_no_applied_parents():
    config = CyclicPostprocessConfig(
        enabled=True,
        warmup_steps=0,
        interval_steps=0,
        plateau_window=2,
        plateau_min_delta_q=0.2,
        require_oversize=False,
        trigger_on_no_applied=True,
        no_applied_window=2,
    )
    checkpoints = [
        RefinementCheckpoint(
            step=0,
            quality=100.0,
            n_above_max_doc_weight=0,
            applied_parent_count=1,
        ),
        RefinementCheckpoint(
            step=1,
            quality=100.05,
            n_above_max_doc_weight=0,
            applied_parent_count=0,
        ),
        RefinementCheckpoint(
            step=2,
            quality=100.10,
            n_above_max_doc_weight=0,
            applied_parent_count=0,
        ),
    ]

    assert cyclic_postprocess_trigger_reasons(
        checkpoints,
        config=config,
    ) == ["quality_plateau", "no_applied_parents"]


def test_cyclic_config_rejects_non_quality_first_postprocess():
    with pytest.raises(ValueError, match="quality_first"):
        CyclicPostprocessConfig(
            enabled=True,
            postprocess_config=HierarchyPostprocessConfig(
                enabled=True,
                oversize_policy="hard_cap",
            ),
        )
    with pytest.raises(ValueError, match="quality_floor_delta"):
        CyclicPostprocessConfig(
            enabled=True,
            postprocess_config=HierarchyPostprocessConfig(
                enabled=True,
                oversize_policy="quality_first",
                quality_floor_delta=-1.0,
            ),
        )


class _QualityMapGraph:
    def __init__(self, qualities: dict[tuple[int, ...], float]):
        self.qualities = qualities

    def cpm_quality(self, membership, *, resolution):
        key = tuple(int(x) for x in np.asarray(membership).tolist())
        return self.qualities[key]


def _result(membership: np.ndarray, *, accepted: bool, status: str) -> LevelPostprocessResult:
    return LevelPostprocessResult(
        membership=np.asarray(membership, dtype=np.uint64),
        accepted=accepted,
        status=status,
        small_cluster_summary={},
        oversize_summary={},
        final_summary={},
    )


def test_run_cyclic_postprocess_accepts_exact_q_improvement():
    current = np.asarray([0, 0, 1], dtype=np.uint64)
    improved = np.asarray([0, 1, 1], dtype=np.uint64)
    graph = _QualityMapGraph({(0, 0, 1): 10.0, (0, 1, 1): 11.0})

    def runner(*args, **kwargs):
        assert kwargs["config"].oversize_policy == "quality_first"
        return _result(improved, accepted=True, status="committed")

    decision = run_cyclic_postprocess_if_due(
        graph,
        raw_membership=current,
        current_membership=current,
        node_weights=np.ones(3, dtype=np.float64),
        resolution=0.1,
        min_doc_weight=1.0,
        target_max_doc_weight=2.0,
        checkpoints=[
            RefinementCheckpoint(step=1, quality=10.0, n_above_max_doc_weight=1)
        ],
        config=CyclicPostprocessConfig(
            enabled=True,
            warmup_steps=0,
            interval_steps=1,
            plateau_window=0,
            require_oversize=True,
            postprocess_config=HierarchyPostprocessConfig(
                enabled=True,
                oversize_policy="quality_first",
                quality_floor_delta=0.5,
            ),
        ),
        seed=42,
        postprocess_runner=runner,
    )

    assert decision.triggered is True
    assert decision.accepted is True
    assert decision.status == "committed"
    assert decision.reasons == ["interval", "accepted"]
    assert decision.quality_before == 10.0
    assert decision.quality_after == 11.0
    assert decision.state.n_calls == 1
    assert decision.state.last_call_step == 1
    np.testing.assert_array_equal(decision.membership, improved)


def test_run_cyclic_postprocess_rolls_back_quality_guard_failure():
    current = np.asarray([0, 0, 1], dtype=np.uint64)
    proposed = np.asarray([1, 1, 1], dtype=np.uint64)
    graph = _QualityMapGraph({(0, 0, 1): 10.0, (1, 1, 1): 11.0})

    def runner(*args, **kwargs):
        return _result(proposed, accepted=True, status="committed")

    decision = run_cyclic_postprocess_if_due(
        graph,
        raw_membership=current,
        current_membership=current,
        node_weights=np.ones(3, dtype=np.float64),
        resolution=0.1,
        min_doc_weight=1.0,
        target_max_doc_weight=2.0,
        checkpoints=[
            RefinementCheckpoint(step=1, quality=10.0, n_above_max_doc_weight=1)
        ],
        config=CyclicPostprocessConfig(
            enabled=True,
            warmup_steps=0,
            interval_steps=1,
            plateau_window=0,
            require_oversize=True,
            postprocess_config=HierarchyPostprocessConfig(
                enabled=True,
                oversize_policy="quality_first",
                quality_floor_delta=2.0,
            ),
        ),
        seed=42,
        postprocess_runner=runner,
    )

    assert decision.triggered is True
    assert decision.accepted is False
    assert decision.status == "quality_guard_rejected"
    assert decision.reasons == ["interval", "rejected"]
    assert decision.state.n_calls == 1
    np.testing.assert_array_equal(decision.membership, current)
