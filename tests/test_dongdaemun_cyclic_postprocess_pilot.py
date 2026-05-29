"""Tests for Dongdaemun cyclic postprocess pilot orchestration."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from sciscape.clustering.hierarchy_postprocess import LevelPostprocessResult


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "research"
    / "consensus"
    / "scripts"
    / "dongdaemun_hierarchy"
    / "prototype_runs"
    / "run_dongdaemun_cyclic_postprocess_pilot.py"
)


def _load_module():
    script_dir = SCRIPT_PATH.parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    spec = importlib.util.spec_from_file_location(
        "run_dongdaemun_cyclic_postprocess_pilot_for_test",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _Audit:
    selected_parent_count_total = 1
    same_gamma_candidates_total = 0
    high_gamma_candidates_total = 1
    candidate_quality_delta_sum = 1.0

    def __init__(self, applied_parent_count_total: int):
        self.applied_parent_count_total = applied_parent_count_total


class _Graph:
    def __init__(self):
        self.calls = []

    def cpm_quality(self, membership, *, resolution):
        key = tuple(int(x) for x in np.asarray(membership, dtype=np.uint64).tolist())
        if key == (0, 1, 1, 2):
            return 30.0
        if key == (0, 0, 1, 1):
            return 20.0
        return 25.0

    def run_leiden_dongdaemun_refinement(self, **kwargs):
        self.calls.append(kwargs)
        policy = kwargs["candidate_quality_policy"]
        initial = kwargs.get("initial_membership")
        membership = (
            np.asarray(initial, dtype=np.uint64).copy()
            if initial is not None
            else np.asarray([0, 0, 1, 1], dtype=np.uint64)
        )
        quality = self.cpm_quality(membership, resolution=kwargs["resolution"])
        if policy == "quality_first":
            quality += 2.0 + len(self.calls)
        else:
            quality += 0.5
        return SimpleNamespace(
            membership=membership,
            quality=quality,
            n_clusters=int(np.unique(membership).size),
            n_iterations_used=int(kwargs["n_iterations"]),
            audit=_Audit(applied_parent_count_total=1 if len(self.calls) == 1 else 0),
        )


class _LookaheadRejectGraph(_Graph):
    def run_leiden_dongdaemun_refinement(self, **kwargs):
        self.calls.append(kwargs)
        initial = kwargs.get("initial_membership")
        membership = (
            np.asarray(initial, dtype=np.uint64).copy()
            if initial is not None
            else np.asarray([0, 0, 1, 1], dtype=np.uint64)
        )
        n_iterations = int(kwargs["n_iterations"])
        key = tuple(int(x) for x in membership.tolist())
        quality = 40.0 if key == (0, 1, 1, 2) else 50.0
        return SimpleNamespace(
            membership=membership,
            quality=quality,
            n_clusters=int(np.unique(membership).size),
            n_iterations_used=n_iterations,
            audit=_Audit(applied_parent_count_total=0),
        )


def _input_cfg():
    return SimpleNamespace(
        sample="tiny",
        resolution=0.1,
        target_max_doc_weight=3.0,
        seed=10,
    )


def _postprocess_runner(*args, **kwargs):
    return LevelPostprocessResult(
        membership=np.asarray([0, 1, 1, 2], dtype=np.uint64),
        accepted=True,
        status="committed",
        small_cluster_summary={},
        oversize_summary={},
        final_summary={},
    )


def test_run_variant_uses_warm_start_and_accepts_cyclic_postprocess(tmp_path):
    module = _load_module()
    graph = _Graph()
    summary, rows = module.run_variant(
        graph,
        input_cfg=_input_cfg(),
        node_weights=np.asarray([2.0, 2.0, 2.0, 2.0]),
        variant=module.VARIANT_LOCAL_QF_CYCLIC_POST,
        config=module.CyclicPilotConfig(
            total_iterations=4,
            chunk_iterations=2,
            cyclic_warmup_steps=0,
            cyclic_interval_steps=2,
            cyclic_cooldown_steps=0,
            cyclic_max_calls=2,
        ),
        output_dir=tmp_path,
        postprocess_runner=_postprocess_runner,
    )

    assert len(graph.calls) == 2
    assert graph.calls[0]["initial_membership"] is None
    np.testing.assert_array_equal(
        graph.calls[1]["initial_membership"],
        np.asarray([0, 1, 1, 2], dtype=np.uint64),
    )
    assert summary["variant"] == module.VARIANT_LOCAL_QF_CYCLIC_POST
    assert summary["triggered_postprocess_calls"] == 2
    assert summary["accepted_postprocess_calls"] == 2
    assert summary["quality"] >= 30.0
    assert any(row["phase"] == "cyclic_postprocess" for row in rows)


def test_chunked_local_qf_control_uses_warm_start_without_postprocess(tmp_path):
    module = _load_module()
    graph = _Graph()
    summary, rows = module.run_variant(
        graph,
        input_cfg=_input_cfg(),
        node_weights=np.asarray([2.0, 2.0, 2.0, 2.0]),
        variant=module.VARIANT_LOCAL_QF_CHUNKED,
        config=module.CyclicPilotConfig(total_iterations=3, chunk_iterations=1),
        output_dir=tmp_path,
        postprocess_runner=_postprocess_runner,
    )

    assert len(graph.calls) == 3
    assert graph.calls[0]["initial_membership"] is None
    assert graph.calls[1]["initial_membership"] is not None
    assert graph.calls[2]["initial_membership"] is not None
    assert summary["variant"] == module.VARIANT_LOCAL_QF_CHUNKED
    assert summary["triggered_postprocess_calls"] == 0
    assert summary["accepted_postprocess_calls"] == 0
    assert summary["n_iterations_used"] == 3
    assert [row["phase"] for row in rows] == [
        "refinement_chunk",
        "refinement_chunk",
        "refinement_chunk",
    ]


def test_cyclic_lookahead_rejects_downstream_regret(tmp_path):
    module = _load_module()
    graph = _LookaheadRejectGraph()
    summary, rows = module.run_variant(
        graph,
        input_cfg=_input_cfg(),
        node_weights=np.asarray([2.0, 2.0, 2.0, 2.0]),
        variant=module.VARIANT_LOCAL_QF_CYCLIC_LOOKAHEAD,
        config=module.CyclicPilotConfig(
            total_iterations=3,
            chunk_iterations=1,
            cyclic_warmup_steps=0,
            cyclic_interval_steps=1,
            cyclic_cooldown_steps=0,
            cyclic_max_calls=1,
            cyclic_lookahead_iterations=2,
        ),
        output_dir=tmp_path,
        postprocess_runner=_postprocess_runner,
    )

    assert len(graph.calls) > 3
    assert summary["triggered_postprocess_calls"] == 1
    assert summary["accepted_postprocess_calls"] == 0
    assert summary["lookahead_guard_evaluations"] == 1
    assert summary["lookahead_guard_rejections"] == 1
    lookahead_rows = [row for row in rows if row["phase"] == "cyclic_postprocess"]
    assert lookahead_rows[0]["postprocess_status"] == "lookahead_guard_rejected"
    assert lookahead_rows[0]["lookahead_delta_q"] == -10.0
    refinement_rows = [row for row in rows if row["phase"] == "refinement_chunk"]
    assert refinement_rows[-1]["quality"] == 50.0


def test_run_pilot_on_graph_writes_summary_outputs(tmp_path):
    module = _load_module()
    payload = module.run_pilot_on_graph(
        _Graph(),
        input_cfg=_input_cfg(),
        node_weights=np.asarray([2.0, 2.0, 2.0, 2.0]),
        output_dir=tmp_path,
        config=module.CyclicPilotConfig(
            total_iterations=2,
            chunk_iterations=1,
            cyclic_warmup_steps=0,
            cyclic_interval_steps=1,
            cyclic_cooldown_steps=0,
            cyclic_max_calls=1,
        ),
        variants=(module.VARIANT_CURRENT_GREEDY, module.VARIANT_LOCAL_QF_CYCLIC_POST),
        postprocess_runner=_postprocess_runner,
    )

    assert payload["variants"] == [
        module.VARIANT_CURRENT_GREEDY,
        module.VARIANT_LOCAL_QF_CYCLIC_POST,
    ]
    for path in payload["paths"].values():
        assert Path(path).exists()
    cyclic = next(
        row
        for row in payload["summary_rows"]
        if row["variant"] == module.VARIANT_LOCAL_QF_CYCLIC_POST
    )
    assert cyclic["quality_delta_vs_current_greedy"] > 0
    assert "local_qf_beam_cyclic_postprocess" in Path(
        payload["paths"]["report"]
    ).read_text(encoding="utf-8")
