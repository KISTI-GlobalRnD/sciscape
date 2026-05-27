"""Run a cyclic Dongdaemun refinement + qf-guarded postprocess pilot.

The pilot compares full-run and chunked warm-start policies without changing
Rust internals.  ``quality_first`` is used as the current local-qf candidate
selection proxy: generated parent-local candidates are selected by CPM
``candidate_delta_q`` while pressure remains diagnostic.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import evaluate_dongdaemun_refinement_slice4 as slice4  # noqa: E402
from sciscape.clustering.hierarchy_postprocess import (  # noqa: E402
    CyclicPostprocessConfig,
    CyclicPostprocessDecision,
    CyclicPostprocessState,
    HierarchyPostprocessConfig,
    RefinementCheckpoint,
    membership_weight_summary,
    run_cyclic_postprocess_if_due,
    run_hierarchy_level_postprocess,
)
from sciscape.clustering.leiden_rust import (  # noqa: E402
    DEFAULT_DONGDAEMUN_GAMMA_MULTIPLIERS,
    RUST_DONGDAEMUN_REFINEMENT_AVAILABLE,
)


DEFAULT_OUTPUT_DIR = (
    Path("research/consensus/results/adaptive_refinement")
    / "dongdaemun_cyclic_postprocess_pilot_20260511"
)
SCHEMA_VERSION = 1

VARIANT_CURRENT_GREEDY = "current_greedy"
VARIANT_LOCAL_QF = "local_qf_beam"
VARIANT_LOCAL_QF_FINAL_POST = "local_qf_beam_final_postprocess"
VARIANT_LOCAL_QF_CHUNKED = "local_qf_beam_chunked"
VARIANT_LOCAL_QF_CYCLIC_POST = "local_qf_beam_cyclic_postprocess"
VARIANT_LOCAL_QF_CYCLIC_LOOKAHEAD = "local_qf_beam_cyclic_lookahead"
DEFAULT_VARIANTS = (
    VARIANT_CURRENT_GREEDY,
    VARIANT_LOCAL_QF,
    VARIANT_LOCAL_QF_CHUNKED,
    VARIANT_LOCAL_QF_FINAL_POST,
    VARIANT_LOCAL_QF_CYCLIC_POST,
)
AVAILABLE_VARIANTS = (*DEFAULT_VARIANTS, VARIANT_LOCAL_QF_CYCLIC_LOOKAHEAD)

ROW_FILENAME = "cyclic_postprocess_pilot_rows.csv"
SUMMARY_CSV_FILENAME = "cyclic_postprocess_pilot_summary.csv"
SUMMARY_JSON_FILENAME = "cyclic_postprocess_pilot_summary.json"
REPORT_FILENAME = "cyclic_postprocess_pilot_report.md"


@dataclass(frozen=True)
class CyclicPilotConfig:
    total_iterations: int = 10
    chunk_iterations: int = 2
    randomness: float = 0.01
    baseline_candidate_quality_policy: str = "structural"
    local_candidate_quality_policy: str = "quality_first"
    soft_min_ratio: float = 1.0
    max_extra_parents_per_iteration: int = 16
    max_extra_children_per_parent: int = 64
    parent_selection_policy: str = "weight"
    max_singleton_weight_fraction: float = 0.05
    min_largest_child_fraction_improvement: float = 0.05
    gamma_multipliers: tuple[float, ...] = DEFAULT_DONGDAEMUN_GAMMA_MULTIPLIERS
    seed_perturbations: int = 0
    use_quotient_diagnostic: bool = True
    use_baseline_repair: bool = False
    baseline_repair_policy: str = "replace"
    baseline_repair_replace_min_parent_ratio: float = 1.05
    baseline_repair_epsilon: float = 0.0
    min_candidate_delta_q: float = 0.0
    adaptive_plateau_quality_band: float = 0.0
    use_final_quality_guard: bool = False
    min_final_quality_delta: float = 0.0

    cyclic_warmup_steps: int = 2
    cyclic_interval_steps: int = 2
    cyclic_plateau_window: int = 2
    cyclic_plateau_min_delta_q: float = 0.0
    cyclic_cooldown_steps: int = 2
    cyclic_max_calls: int = 3
    cyclic_require_oversize: bool = True
    cyclic_trigger_on_no_applied: bool = True
    cyclic_no_applied_window: int = 2
    postprocess_apply_iterations: int = 4
    postprocess_use_rust_dongdaemun: bool = True
    cyclic_lookahead_iterations: int = 4
    cyclic_lookahead_min_delta_q: float = 1.0

    def __post_init__(self) -> None:
        if self.total_iterations < 1:
            raise ValueError("total_iterations must be >= 1")
        if self.chunk_iterations < 1:
            raise ValueError("chunk_iterations must be >= 1")
        if self.seed_perturbations < 0:
            raise ValueError("seed_perturbations must be >= 0")
        if self.cyclic_lookahead_iterations < 0:
            raise ValueError("cyclic_lookahead_iterations must be >= 0")
        if not np.isfinite(float(self.cyclic_lookahead_min_delta_q)):
            raise ValueError("cyclic_lookahead_min_delta_q must be finite")


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fieldnames})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _cluster_summary(
    membership: np.ndarray,
    node_weights: np.ndarray,
    *,
    target_max_doc_weight: float,
) -> dict[str, Any]:
    summary = membership_weight_summary(
        np.asarray(membership, dtype=np.uint64),
        np.asarray(node_weights, dtype=np.float64),
        max_weight=float(target_max_doc_weight),
    )
    max_doc_weight = float(summary["max_doc_weight"])
    target = float(target_max_doc_weight)
    return {
        "n_clusters": int(summary["n_clusters"]),
        "max_doc_weight": max_doc_weight,
        "max_doc_weight_ratio": max_doc_weight / target if target > 0.0 else None,
        "n_above_max_doc_weight": int(summary["n_above_max_doc_weight"]),
        "n_singletons": int(summary["n_singletons"]),
        "top10_doc_weights": summary.get("top10_doc_weights", []),
    }


def _postprocess_config(config: CyclicPilotConfig) -> HierarchyPostprocessConfig:
    return HierarchyPostprocessConfig(
        enabled=True,
        use_rust_dongdaemun=bool(config.postprocess_use_rust_dongdaemun),
        oversize_policy="quality_first",
        apply_iterations=int(config.postprocess_apply_iterations),
        quality_floor_delta=0.0,
        quality_first_trim_min_delta_q=0.0,
        write_artifacts=False,
    )


def _cyclic_config(config: CyclicPilotConfig) -> CyclicPostprocessConfig:
    return CyclicPostprocessConfig(
        enabled=True,
        warmup_steps=int(config.cyclic_warmup_steps),
        interval_steps=int(config.cyclic_interval_steps),
        plateau_window=int(config.cyclic_plateau_window),
        plateau_min_delta_q=float(config.cyclic_plateau_min_delta_q),
        cooldown_steps=int(config.cyclic_cooldown_steps),
        max_calls=int(config.cyclic_max_calls),
        require_oversize=bool(config.cyclic_require_oversize),
        trigger_on_no_applied=bool(config.cyclic_trigger_on_no_applied),
        no_applied_window=int(config.cyclic_no_applied_window),
        postprocess_config=_postprocess_config(config),
    )


def _run_refinement_chunk(
    graph: Any,
    *,
    input_cfg: Any,
    config: CyclicPilotConfig,
    candidate_quality_policy: str,
    initial_membership: np.ndarray | None,
    n_iterations: int,
    seed: int,
) -> Any:
    return graph.run_leiden_dongdaemun_refinement(
        target_max_weight=float(input_cfg.target_max_doc_weight),
        resolution=float(input_cfg.resolution),
        seed=int(seed),
        n_iterations=int(n_iterations),
        randomness=float(config.randomness),
        initial_membership=initial_membership,
        soft_min_ratio=float(config.soft_min_ratio),
        max_extra_parents_per_iteration=int(config.max_extra_parents_per_iteration),
        max_extra_children_per_parent=int(config.max_extra_children_per_parent),
        parent_selection_policy=str(config.parent_selection_policy),
        max_singleton_weight_fraction=float(config.max_singleton_weight_fraction),
        min_largest_child_fraction_improvement=float(
            config.min_largest_child_fraction_improvement
        ),
        gamma_multipliers=tuple(float(x) for x in config.gamma_multipliers),
        seed_perturbations=int(config.seed_perturbations),
        use_quotient_diagnostic=bool(config.use_quotient_diagnostic),
        use_baseline_repair=bool(config.use_baseline_repair),
        baseline_repair_policy=str(config.baseline_repair_policy),
        baseline_repair_replace_min_parent_ratio=float(
            config.baseline_repair_replace_min_parent_ratio
        ),
        baseline_repair_epsilon=float(config.baseline_repair_epsilon),
        candidate_quality_policy=str(candidate_quality_policy),
        min_candidate_delta_q=float(config.min_candidate_delta_q),
        adaptive_plateau_quality_band=float(config.adaptive_plateau_quality_band),
        use_final_quality_guard=bool(config.use_final_quality_guard),
        min_final_quality_delta=float(config.min_final_quality_delta),
    )


def _checkpoint_from_result(
    *,
    step: int,
    result: Any,
    membership: np.ndarray,
    node_weights: np.ndarray,
    target_max_doc_weight: float,
) -> RefinementCheckpoint:
    summary = _cluster_summary(
        membership,
        node_weights,
        target_max_doc_weight=target_max_doc_weight,
    )
    audit = getattr(result, "audit", None)
    return RefinementCheckpoint(
        step=int(step),
        quality=float(result.quality),
        n_above_max_doc_weight=int(summary["n_above_max_doc_weight"]),
        max_doc_weight_ratio=summary["max_doc_weight_ratio"],
        applied_parent_count=(
            None
            if audit is None
            else int(getattr(audit, "applied_parent_count_total", 0))
        ),
    )


def _phase_row(
    *,
    sample: str,
    variant: str,
    phase: str,
    step: int,
    chunk_index: int | None,
    elapsed_sec: float,
    quality: float,
    start_quality: float,
    membership: np.ndarray,
    node_weights: np.ndarray,
    target_max_doc_weight: float,
    audit: Any | None = None,
    postprocess_decision: CyclicPostprocessDecision | None = None,
    lookahead: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary = _cluster_summary(
        membership,
        node_weights,
        target_max_doc_weight=target_max_doc_weight,
    )
    row = {
        "sample": sample,
        "variant": variant,
        "phase": phase,
        "step": int(step),
        "chunk_index": chunk_index,
        "elapsed_sec": float(elapsed_sec),
        "quality": float(quality),
        "quality_delta_vs_start": float(quality) - float(start_quality),
        **summary,
    }
    if audit is not None:
        row.update(
            {
                "selected_parent_count_total": int(
                    getattr(audit, "selected_parent_count_total", 0)
                ),
                "applied_parent_count_total": int(
                    getattr(audit, "applied_parent_count_total", 0)
                ),
                "same_gamma_candidates_total": int(
                    getattr(audit, "same_gamma_candidates_total", 0)
                ),
                "high_gamma_candidates_total": int(
                    getattr(audit, "high_gamma_candidates_total", 0)
                ),
                "candidate_quality_delta_sum": float(
                    getattr(audit, "candidate_quality_delta_sum", 0.0)
                ),
            }
        )
    if postprocess_decision is not None:
        row.update(
            {
                "postprocess_triggered": bool(postprocess_decision.triggered),
                "postprocess_accepted": bool(postprocess_decision.accepted),
                "postprocess_status": postprocess_decision.status,
                "postprocess_reasons": postprocess_decision.reasons,
                "postprocess_quality_before": postprocess_decision.quality_before,
                "postprocess_quality_after": postprocess_decision.quality_after,
                "postprocess_calls": int(postprocess_decision.state.n_calls),
            }
        )
    if lookahead is not None:
        row.update(
            {
                "lookahead_guard_used": bool(lookahead.get("used", False)),
                "lookahead_guard_accepted": lookahead.get("accepted"),
                "lookahead_iterations": lookahead.get("iterations"),
                "lookahead_baseline_quality": lookahead.get("baseline_quality"),
                "lookahead_candidate_quality": lookahead.get("candidate_quality"),
                "lookahead_delta_q": lookahead.get("delta_q"),
                "lookahead_min_delta_q": lookahead.get("min_delta_q"),
                "lookahead_elapsed_sec": lookahead.get("elapsed_sec"),
            }
        )
    return row


def _apply_cyclic_lookahead_guard(
    graph: Any,
    *,
    input_cfg: Any,
    config: CyclicPilotConfig,
    current_membership: np.ndarray,
    decision: CyclicPostprocessDecision,
    completed_iterations: int,
    chunk_index: int,
    candidate_quality_policy: str,
) -> tuple[CyclicPostprocessDecision, dict[str, Any]]:
    remaining = int(config.total_iterations) - int(completed_iterations)
    n_lookahead = min(int(config.cyclic_lookahead_iterations), remaining)
    started = time.perf_counter()
    if n_lookahead <= 0:
        return decision, {
            "used": False,
            "accepted": bool(decision.accepted),
            "iterations": 0,
            "elapsed_sec": time.perf_counter() - started,
        }

    baseline_quality = _run_lookahead_path(
        graph,
        input_cfg=input_cfg,
        config=config,
        candidate_quality_policy=candidate_quality_policy,
        initial_membership=np.asarray(current_membership, dtype=np.uint64),
        n_iterations=n_lookahead,
        current_chunk_index=chunk_index,
    )
    candidate_quality = _run_lookahead_path(
        graph,
        input_cfg=input_cfg,
        config=config,
        candidate_quality_policy=candidate_quality_policy,
        initial_membership=np.asarray(decision.membership, dtype=np.uint64),
        n_iterations=n_lookahead,
        current_chunk_index=chunk_index,
    )
    delta_q = candidate_quality - baseline_quality
    accepted = delta_q >= float(config.cyclic_lookahead_min_delta_q) - 1e-9
    elapsed = time.perf_counter() - started
    lookahead = {
        "used": True,
        "accepted": bool(accepted),
        "iterations": int(n_lookahead),
        "baseline_quality": baseline_quality,
        "candidate_quality": candidate_quality,
        "delta_q": delta_q,
        "min_delta_q": float(config.cyclic_lookahead_min_delta_q),
        "elapsed_sec": elapsed,
    }
    if accepted:
        return decision, lookahead

    guarded = CyclicPostprocessDecision(
        membership=np.asarray(current_membership, dtype=np.uint64),
        triggered=True,
        accepted=False,
        status="lookahead_guard_rejected",
        reasons=[
            reason
            for reason in decision.reasons
            if reason not in ("accepted", "rejected")
        ]
        + ["lookahead_rejected"],
        state=decision.state,
        result=decision.result,
        quality_before=decision.quality_before,
        quality_after=decision.quality_after,
    )
    return guarded, lookahead


def _run_lookahead_path(
    graph: Any,
    *,
    input_cfg: Any,
    config: CyclicPilotConfig,
    candidate_quality_policy: str,
    initial_membership: np.ndarray,
    n_iterations: int,
    current_chunk_index: int,
) -> float:
    membership = np.asarray(initial_membership, dtype=np.uint64)
    completed = 0
    quality = float(
        graph.cpm_quality(membership=membership, resolution=float(input_cfg.resolution))
    )
    while completed < int(n_iterations):
        future_chunk_index = int(current_chunk_index) + 1 + completed // int(
            config.chunk_iterations
        )
        n_this = min(int(config.chunk_iterations), int(n_iterations) - completed)
        result = _run_refinement_chunk(
            graph,
            input_cfg=input_cfg,
            config=config,
            candidate_quality_policy=candidate_quality_policy,
            initial_membership=membership,
            n_iterations=n_this,
            seed=int(input_cfg.seed) + future_chunk_index - 1,
        )
        membership = np.asarray(result.membership, dtype=np.uint64)
        quality = float(result.quality)
        completed += n_this
    return quality


def run_variant(
    graph: Any,
    *,
    input_cfg: Any,
    node_weights: np.ndarray,
    variant: str,
    config: CyclicPilotConfig,
    output_dir: Path | None = None,
    postprocess_runner: Callable[..., Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if variant not in AVAILABLE_VARIANTS:
        raise ValueError(f"unknown variant: {variant}")

    candidate_quality_policy = (
        config.baseline_candidate_quality_policy
        if variant == VARIANT_CURRENT_GREEDY
        else config.local_candidate_quality_policy
    )
    use_chunks = variant in (
        VARIANT_LOCAL_QF_CHUNKED,
        VARIANT_LOCAL_QF_CYCLIC_POST,
        VARIANT_LOCAL_QF_CYCLIC_LOOKAHEAD,
    )
    use_cyclic_postprocess = variant in (
        VARIANT_LOCAL_QF_CYCLIC_POST,
        VARIANT_LOCAL_QF_CYCLIC_LOOKAHEAD,
    )
    use_lookahead_guard = variant == VARIANT_LOCAL_QF_CYCLIC_LOOKAHEAD
    n_iterations = (
        min(int(config.chunk_iterations), int(config.total_iterations))
        if use_chunks
        else int(config.total_iterations)
    )
    current_membership: np.ndarray | None = None
    checkpoints: list[RefinementCheckpoint] = []
    cyclic_state = CyclicPostprocessState()
    rows: list[dict[str, Any]] = []
    total_elapsed = 0.0
    total_postprocess_elapsed = 0.0
    total_lookahead_elapsed = 0.0
    start_quality: float | None = None
    final_quality = 0.0
    final_result: Any | None = None
    accepted_postprocess_calls = 0
    triggered_postprocess_calls = 0
    lookahead_guard_evaluations = 0
    lookahead_guard_rejections = 0
    completed_iterations = 0
    chunk_index = 0

    while completed_iterations < int(config.total_iterations):
        chunk_index += 1
        remaining = int(config.total_iterations) - completed_iterations
        n_this_chunk = min(n_iterations, remaining)
        chunk_seed = int(input_cfg.seed) + chunk_index - 1
        started = time.perf_counter()
        result = _run_refinement_chunk(
            graph,
            input_cfg=input_cfg,
            config=config,
            candidate_quality_policy=candidate_quality_policy,
            initial_membership=current_membership,
            n_iterations=n_this_chunk,
            seed=chunk_seed,
        )
        elapsed = time.perf_counter() - started
        total_elapsed += elapsed
        completed_iterations += n_this_chunk
        current_membership = np.asarray(result.membership, dtype=np.uint64)
        final_result = result
        final_quality = float(result.quality)
        if start_quality is None:
            start_quality = final_quality
        checkpoint = _checkpoint_from_result(
            step=completed_iterations,
            result=result,
            membership=current_membership,
            node_weights=node_weights,
            target_max_doc_weight=float(input_cfg.target_max_doc_weight),
        )
        checkpoints.append(checkpoint)
        rows.append(
            _phase_row(
                sample=str(input_cfg.sample),
                variant=variant,
                phase="refinement_chunk" if use_chunks else "refinement",
                step=completed_iterations,
                chunk_index=chunk_index if use_chunks else None,
                elapsed_sec=elapsed,
                quality=final_quality,
                start_quality=start_quality,
                membership=current_membership,
                node_weights=node_weights,
                target_max_doc_weight=float(input_cfg.target_max_doc_weight),
                audit=getattr(result, "audit", None),
            )
        )

        if not use_chunks:
            break
        if not use_cyclic_postprocess:
            continue

        postprocess_started = time.perf_counter()
        decision = run_cyclic_postprocess_if_due(
            graph,
            raw_membership=current_membership,
            current_membership=current_membership,
            node_weights=node_weights,
            resolution=float(input_cfg.resolution),
            min_doc_weight=1.0,
            target_max_doc_weight=float(input_cfg.target_max_doc_weight),
            checkpoints=checkpoints,
            config=_cyclic_config(config),
            seed=chunk_seed,
            state=cyclic_state,
            output_dir=None if output_dir is None else output_dir / variant / f"step_{completed_iterations:03d}",
            postprocess_runner=postprocess_runner,
        )
        postprocess_elapsed = time.perf_counter() - postprocess_started
        lookahead: dict[str, Any] | None = None
        if decision.triggered:
            triggered_postprocess_calls += 1
            total_postprocess_elapsed += postprocess_elapsed
            cyclic_state = decision.state
            if use_lookahead_guard and decision.accepted:
                decision, lookahead = _apply_cyclic_lookahead_guard(
                    graph,
                    input_cfg=input_cfg,
                    config=config,
                    current_membership=current_membership,
                    decision=decision,
                    completed_iterations=completed_iterations,
                    chunk_index=chunk_index,
                    candidate_quality_policy=candidate_quality_policy,
                )
                if lookahead is not None:
                    total_lookahead_elapsed += float(lookahead.get("elapsed_sec") or 0.0)
                    if lookahead.get("used"):
                        lookahead_guard_evaluations += 1
                        if not lookahead.get("accepted"):
                            lookahead_guard_rejections += 1
            if decision.accepted:
                accepted_postprocess_calls += 1
                current_membership = np.asarray(decision.membership, dtype=np.uint64)
                final_quality = float(decision.quality_after)
                accepted_summary = _cluster_summary(
                    current_membership,
                    node_weights,
                    target_max_doc_weight=float(input_cfg.target_max_doc_weight),
                )
                checkpoints[-1] = RefinementCheckpoint(
                    step=completed_iterations,
                    quality=final_quality,
                    n_above_max_doc_weight=int(
                        accepted_summary["n_above_max_doc_weight"]
                    ),
                    max_doc_weight_ratio=accepted_summary["max_doc_weight_ratio"],
                    applied_parent_count=checkpoints[-1].applied_parent_count,
                )
            rows.append(
                _phase_row(
                    sample=str(input_cfg.sample),
                    variant=variant,
                    phase="cyclic_postprocess",
                    step=completed_iterations,
                    chunk_index=chunk_index,
                    elapsed_sec=postprocess_elapsed,
                    quality=final_quality,
                    start_quality=start_quality,
                    membership=current_membership,
                    node_weights=node_weights,
                    target_max_doc_weight=float(input_cfg.target_max_doc_weight),
                    postprocess_decision=decision,
                    lookahead=lookahead,
                )
            )

    if variant == VARIANT_LOCAL_QF_FINAL_POST:
        assert current_membership is not None
        postprocess_started = time.perf_counter()
        decision = run_cyclic_postprocess_if_due(
            graph,
            raw_membership=current_membership,
            current_membership=current_membership,
            node_weights=node_weights,
            resolution=float(input_cfg.resolution),
            min_doc_weight=1.0,
            target_max_doc_weight=float(input_cfg.target_max_doc_weight),
            checkpoints=[
                RefinementCheckpoint(
                    step=int(config.total_iterations),
                    quality=final_quality,
                    n_above_max_doc_weight=_cluster_summary(
                        current_membership,
                        node_weights,
                        target_max_doc_weight=float(input_cfg.target_max_doc_weight),
                    )["n_above_max_doc_weight"],
                )
            ],
            config=CyclicPostprocessConfig(
                enabled=True,
                warmup_steps=0,
                interval_steps=1,
                plateau_window=0,
                cooldown_steps=0,
                max_calls=1,
                require_oversize=False,
                postprocess_config=_postprocess_config(config),
            ),
            seed=int(input_cfg.seed),
            output_dir=None if output_dir is None else output_dir / variant / "final_postprocess",
            postprocess_runner=postprocess_runner,
        )
        postprocess_elapsed = time.perf_counter() - postprocess_started
        if decision.triggered:
            triggered_postprocess_calls += 1
            total_postprocess_elapsed += postprocess_elapsed
            if decision.accepted:
                accepted_postprocess_calls += 1
                current_membership = np.asarray(decision.membership, dtype=np.uint64)
                final_quality = float(decision.quality_after)
            rows.append(
                _phase_row(
                    sample=str(input_cfg.sample),
                    variant=variant,
                    phase="final_postprocess",
                    step=int(config.total_iterations),
                    chunk_index=None,
                    elapsed_sec=postprocess_elapsed,
                    quality=final_quality,
                    start_quality=start_quality if start_quality is not None else final_quality,
                    membership=current_membership,
                    node_weights=node_weights,
                    target_max_doc_weight=float(input_cfg.target_max_doc_weight),
                    postprocess_decision=decision,
                )
            )

    assert current_membership is not None
    final_summary = _cluster_summary(
        current_membership,
        node_weights,
        target_max_doc_weight=float(input_cfg.target_max_doc_weight),
    )
    summary_row = {
        "sample": str(input_cfg.sample),
        "variant": variant,
        "supported": True,
        "total_iterations": int(config.total_iterations),
        "chunk_iterations": int(config.chunk_iterations) if use_chunks else "",
        "candidate_quality_policy": candidate_quality_policy,
        "elapsed_refinement_sec": total_elapsed,
        "elapsed_postprocess_sec": total_postprocess_elapsed,
        "elapsed_lookahead_sec": total_lookahead_elapsed,
        "elapsed_total_sec": total_elapsed
        + total_postprocess_elapsed
        + total_lookahead_elapsed,
        "quality": final_quality,
        "quality_delta_vs_start": (
            0.0 if start_quality is None else final_quality - float(start_quality)
        ),
        **final_summary,
        "triggered_postprocess_calls": triggered_postprocess_calls,
        "accepted_postprocess_calls": accepted_postprocess_calls,
        "lookahead_guard_evaluations": lookahead_guard_evaluations,
        "lookahead_guard_rejections": lookahead_guard_rejections,
        "n_phase_rows": len(rows),
        "n_iterations_used": (
            int(completed_iterations)
            if use_chunks
            else (
                None
                if final_result is None
                else int(
                    getattr(final_result, "n_iterations_used", config.total_iterations)
                )
            )
        ),
    }
    return summary_row, rows


def _set_quality_comparisons(summary_rows: list[dict[str, Any]]) -> None:
    by_variant = {str(row["variant"]): row for row in summary_rows}
    baseline = by_variant.get(VARIANT_CURRENT_GREEDY)
    local = by_variant.get(VARIANT_LOCAL_QF)
    chunked = by_variant.get(VARIANT_LOCAL_QF_CHUNKED)
    for row in summary_rows:
        quality = float(row.get("quality") or 0.0)
        row["quality_delta_vs_current_greedy"] = (
            None if baseline is None else quality - float(baseline.get("quality") or 0.0)
        )
        row["quality_delta_vs_local_qf_beam"] = (
            None if local is None else quality - float(local.get("quality") or 0.0)
        )
        row["quality_delta_vs_local_qf_beam_chunked"] = (
            None
            if chunked is None
            else quality - float(chunked.get("quality") or 0.0)
        )


def _build_report(summary_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Dongdaemun Cyclic Postprocess Pilot",
        "",
        "| variant | quality | dQ vs greedy | dQ vs local qf | dQ vs chunked | max ratio | post calls | accepted | lh eval | lh reject | elapsed sec |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| {variant} | {quality:.3f} | {dg} | {dl} | {dc} | {ratio} | {calls} | {accepted} | {lh_eval} | {lh_reject} | {elapsed:.3f} |".format(
                variant=row["variant"],
                quality=float(row["quality"]),
                dg=(
                    ""
                    if row.get("quality_delta_vs_current_greedy") is None
                    else f"{float(row['quality_delta_vs_current_greedy']):.3f}"
                ),
                dl=(
                    ""
                    if row.get("quality_delta_vs_local_qf_beam") is None
                    else f"{float(row['quality_delta_vs_local_qf_beam']):.3f}"
                ),
                dc=(
                    ""
                    if row.get("quality_delta_vs_local_qf_beam_chunked") is None
                    else f"{float(row['quality_delta_vs_local_qf_beam_chunked']):.3f}"
                ),
                ratio=(
                    ""
                    if row.get("max_doc_weight_ratio") is None
                    else f"{float(row['max_doc_weight_ratio']):.4f}"
                ),
                calls=row.get("triggered_postprocess_calls", 0),
                accepted=row.get("accepted_postprocess_calls", 0),
                lh_eval=row.get("lookahead_guard_evaluations", 0),
                lh_reject=row.get("lookahead_guard_rejections", 0),
                elapsed=float(row.get("elapsed_total_sec") or 0.0),
            )
        )
    lines.append("")
    return "\n".join(lines)


def run_pilot(
    input_cfg: Any,
    *,
    output_dir: Path,
    config: CyclicPilotConfig,
    variants: Iterable[str] = DEFAULT_VARIANTS,
    postprocess_runner: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    if not RUST_DONGDAEMUN_REFINEMENT_AVAILABLE:
        raise RuntimeError(
            "Rust Dongdaemun refinement is unavailable; rebuild the Rust extension."
        )
    n_nodes = slice4._infer_n_nodes(input_cfg)
    node_weights = slice4._load_node_weights(input_cfg.node_weights_path, n_nodes)
    graph = slice4._load_graph(input_cfg, node_weights)
    return run_pilot_on_graph(
        graph,
        input_cfg=input_cfg,
        node_weights=node_weights,
        output_dir=output_dir,
        config=config,
        variants=variants,
        postprocess_runner=postprocess_runner,
    )


def run_pilot_on_graph(
    graph: Any,
    *,
    input_cfg: Any,
    node_weights: np.ndarray,
    output_dir: Path,
    config: CyclicPilotConfig,
    variants: Iterable[str] = DEFAULT_VARIANTS,
    postprocess_runner: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
    for variant in variants:
        summary_row, rows = run_variant(
            graph,
            input_cfg=input_cfg,
            node_weights=node_weights,
            variant=str(variant),
            config=config,
            output_dir=output_dir,
            postprocess_runner=postprocess_runner,
        )
        summary_rows.append(summary_row)
        phase_rows.extend(rows)
    _set_quality_comparisons(summary_rows)

    row_path = output_dir / ROW_FILENAME
    summary_csv_path = output_dir / SUMMARY_CSV_FILENAME
    summary_json_path = output_dir / SUMMARY_JSON_FILENAME
    report_path = output_dir / REPORT_FILENAME
    _write_csv(row_path, phase_rows)
    _write_csv(summary_csv_path, summary_rows)
    report_path.write_text(_build_report(summary_rows), encoding="utf-8")
    payload = {
        "schema": "dongdaemun_cyclic_postprocess_pilot.v1",
        "schema_version": SCHEMA_VERSION,
        "input": {
            "sample": str(input_cfg.sample),
            "resolution": float(input_cfg.resolution),
            "target_max_doc_weight": float(input_cfg.target_max_doc_weight),
            "seed": int(input_cfg.seed),
        },
        "config": asdict(config),
        "variants": [str(row["variant"]) for row in summary_rows],
        "summary_rows": summary_rows,
        "paths": {
            "rows": str(row_path),
            "summary_csv": str(summary_csv_path),
            "summary_json": str(summary_json_path),
            "report": str(report_path),
        },
    }
    _write_json(summary_json_path, payload)
    return payload


def _parse_float_tuple(value: str | None) -> tuple[float, ...]:
    if value is None or not str(value).strip():
        return DEFAULT_DONGDAEMUN_GAMMA_MULTIPLIERS
    return tuple(float(item.strip()) for item in str(value).split(",") if item.strip())


def _parse_variants(value: str | None) -> tuple[str, ...]:
    if value is None or not value.strip():
        return DEFAULT_VARIANTS
    variants = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = [variant for variant in variants if variant not in AVAILABLE_VARIANTS]
    if unknown:
        raise ValueError(f"Unknown variants: {', '.join(unknown)}")
    return variants


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", "--prepare-summary", dest="summary", type=Path)
    parser.add_argument("--graph-dir", type=Path)
    parser.add_argument("--membership", type=Path)
    parser.add_argument("--node-weights", type=Path)
    parser.add_argument("--n-nodes", type=int)
    parser.add_argument("--sample")
    parser.add_argument("--resolution", type=float)
    parser.add_argument("--target-max-doc-weight", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--variants")
    parser.add_argument("--total-iterations", type=int, default=10)
    parser.add_argument("--chunk-iterations", type=int, default=2)
    parser.add_argument("--randomness", type=float, default=0.01)
    parser.add_argument(
        "--baseline-candidate-quality-policy",
        default="structural",
        choices=(
            "structural",
            "quality_guarded_structural",
            "quality_floor",
            "quality_first",
            "selective",
            "pressure_aware",
            "adaptive_plateau",
        ),
    )
    parser.add_argument("--seed-perturbations", type=int, default=0)
    parser.add_argument("--gamma-multipliers")
    parser.add_argument("--max-extra-parents-per-iteration", type=int, default=16)
    parser.add_argument("--max-extra-children-per-parent", type=int, default=64)
    parser.add_argument("--use-baseline-repair", action="store_true")
    parser.add_argument("--cyclic-warmup-steps", type=int, default=2)
    parser.add_argument("--cyclic-interval-steps", type=int, default=2)
    parser.add_argument("--cyclic-plateau-window", type=int, default=2)
    parser.add_argument("--cyclic-cooldown-steps", type=int, default=2)
    parser.add_argument("--cyclic-max-calls", type=int, default=3)
    parser.add_argument("--cyclic-lookahead-iterations", type=int, default=4)
    parser.add_argument("--cyclic-lookahead-min-delta-q", type=float, default=1.0)
    parser.add_argument("--postprocess-apply-iterations", type=int, default=4)
    parser.add_argument("--disable-rust-postprocess", action="store_true")
    return parser


def _input_from_args(args: argparse.Namespace) -> Any:
    if args.summary is not None:
        return slice4._resolve_input_from_summary(
            args.summary,
            sample=args.sample,
            resolution=args.resolution,
            target_max_doc_weight=args.target_max_doc_weight,
            seed=args.seed,
        )
    return slice4._resolve_explicit_input(args)


def _config_from_args(args: argparse.Namespace) -> CyclicPilotConfig:
    return CyclicPilotConfig(
        total_iterations=int(args.total_iterations),
        chunk_iterations=int(args.chunk_iterations),
        randomness=float(args.randomness),
        baseline_candidate_quality_policy=str(args.baseline_candidate_quality_policy),
        gamma_multipliers=_parse_float_tuple(args.gamma_multipliers),
        seed_perturbations=int(args.seed_perturbations),
        max_extra_parents_per_iteration=int(args.max_extra_parents_per_iteration),
        max_extra_children_per_parent=int(args.max_extra_children_per_parent),
        use_baseline_repair=bool(args.use_baseline_repair),
        cyclic_warmup_steps=int(args.cyclic_warmup_steps),
        cyclic_interval_steps=int(args.cyclic_interval_steps),
        cyclic_plateau_window=int(args.cyclic_plateau_window),
        cyclic_cooldown_steps=int(args.cyclic_cooldown_steps),
        cyclic_max_calls=int(args.cyclic_max_calls),
        cyclic_lookahead_iterations=int(args.cyclic_lookahead_iterations),
        cyclic_lookahead_min_delta_q=float(args.cyclic_lookahead_min_delta_q),
        postprocess_apply_iterations=int(args.postprocess_apply_iterations),
        postprocess_use_rust_dongdaemun=not bool(args.disable_rust_postprocess),
    )


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    payload = run_pilot(
        _input_from_args(args),
        output_dir=args.output_dir,
        config=_config_from_args(args),
        variants=_parse_variants(args.variants),
    )
    print(f"Saved cyclic postprocess pilot to {payload['paths']['summary_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
