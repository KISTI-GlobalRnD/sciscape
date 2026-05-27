"""Hierarchy-level oversize postprocess automation.

This module is intentionally opt-in. It packages the adaptive split-repair and
oversize boundary trim policies used by diagnostics so hierarchy construction
can repeat them level-by-level without changing default clustering behavior.

Dongdaemun-backed Rust shortcuts are development-only: they are explicit
``HierarchyPostprocessConfig`` opt-ins for local SciSci experiments, not part of
the stable clustering API or default hierarchy path.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np

from .adaptive_refinement import SplitRepairSelectionPolicy, rank_split_repair_candidates

OversizePolicy = Literal["quality_first", "hard_cap"]

DEFAULT_GAMMA_MULTIPLIERS = (1.02, 1.05, 1.10, 1.15, 1.20, 1.25)
DEFAULT_MIN_CORE_WEIGHT = 25.0
DEFAULT_RANDOMNESS = 0.01
DEFAULT_REPAIR_EPSILON = 0.0
DEFAULT_MAX_CANDIDATES = 1000

TRIM_FIELDS = [
    "move_index",
    "committed",
    "source",
    "target",
    "node",
    "node_weight",
    "delta_q",
    "source_weight_before",
    "source_weight_after",
    "target_weight_before",
    "target_weight_after",
]


@dataclass(frozen=True)
class HierarchyPostprocessConfig:
    """Internal opt-in knobs for hierarchy postprocess automation.

    ``use_rust_dongdaemun`` is a development-only fast path. The stable main
    path remains CPM Leiden plus the standard small-cluster postprocess.
    """

    enabled: bool = False
    use_rust_dongdaemun: bool = False
    oversize_policy: OversizePolicy = "quality_first"
    apply_iterations: int = 4
    selection_mode: str = "oversize_first"
    quality_floor_delta: float = 0.0
    quality_first_trim_min_delta_q: float = 0.0
    hard_cap_trim_min_delta_q: float = -1.0
    trim_max_moves_per_cluster: int = 100
    write_artifacts: bool = True

    def __post_init__(self) -> None:
        if self.oversize_policy not in {"quality_first", "hard_cap"}:
            raise ValueError(f"unknown oversize_policy: {self.oversize_policy!r}")
        if self.apply_iterations < 1:
            raise ValueError("apply_iterations must be >= 1")
        if self.selection_mode not in {"oversize_first", "utility_cost"}:
            raise ValueError(f"unknown selection_mode: {self.selection_mode!r}")
        if self.trim_max_moves_per_cluster < 0:
            raise ValueError("trim_max_moves_per_cluster must be >= 0")
        if (
            self.oversize_policy == "quality_first"
            and self.quality_first_trim_min_delta_q < 0.0
        ):
            raise ValueError("quality_first trim delta bound must be >= 0")


@dataclass
class LevelPostprocessResult:
    """Result for one hierarchy level postprocess pass."""

    membership: np.ndarray
    accepted: bool
    status: str
    small_cluster_summary: dict[str, Any]
    oversize_summary: dict[str, Any]
    final_summary: dict[str, Any]
    paths: dict[str, str] = field(default_factory=dict)
    backend: str = "python"


@dataclass(frozen=True)
class RefinementCheckpoint:
    """Lightweight state snapshot for cyclic refinement/postprocess policies."""

    step: int
    quality: float
    n_above_max_doc_weight: int = 0
    max_doc_weight_ratio: float | None = None
    applied_parent_count: int | None = None

    def __post_init__(self) -> None:
        if self.step < 0:
            raise ValueError("step must be >= 0")
        if not np.isfinite(float(self.quality)):
            raise ValueError("quality must be finite")
        if self.n_above_max_doc_weight < 0:
            raise ValueError("n_above_max_doc_weight must be >= 0")


@dataclass(frozen=True)
class CyclicPostprocessConfig:
    """Policy for interleaving qf-guarded postprocess during refinement."""

    enabled: bool = False
    warmup_steps: int = 1
    interval_steps: int = 0
    plateau_window: int = 2
    plateau_min_delta_q: float = 0.0
    cooldown_steps: int = 1
    max_calls: int = 3
    require_oversize: bool = True
    trigger_on_no_applied: bool = True
    no_applied_window: int = 2
    postprocess_config: HierarchyPostprocessConfig = field(
        default_factory=lambda: HierarchyPostprocessConfig(
            enabled=True,
            oversize_policy="quality_first",
            quality_floor_delta=0.0,
        )
    )

    def __post_init__(self) -> None:
        for name in (
            "warmup_steps",
            "interval_steps",
            "plateau_window",
            "cooldown_steps",
            "max_calls",
            "no_applied_window",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        if not np.isfinite(float(self.plateau_min_delta_q)):
            raise ValueError("plateau_min_delta_q must be finite")
        if self.postprocess_config.oversize_policy != "quality_first":
            raise ValueError("cyclic postprocess requires quality_first oversize_policy")
        if float(self.postprocess_config.quality_floor_delta) < 0.0:
            raise ValueError("cyclic postprocess quality_floor_delta must be >= 0")


@dataclass(frozen=True)
class CyclicPostprocessState:
    """Mutable-by-replacement state for cyclic postprocess scheduling."""

    n_calls: int = 0
    last_call_step: int | None = None


@dataclass
class CyclicPostprocessDecision:
    """Decision returned by ``run_cyclic_postprocess_if_due``."""

    membership: np.ndarray
    triggered: bool
    accepted: bool
    status: str
    reasons: list[str]
    state: CyclicPostprocessState
    result: LevelPostprocessResult | None = None
    quality_before: float | None = None
    quality_after: float | None = None


def postprocess_config_hash(config: HierarchyPostprocessConfig | None) -> str | None:
    """Return a stable short hash for cache validation."""

    if config is None:
        return None
    payload = {
        "schema": "hierarchy_postprocess.v1",
        "config": asdict(config),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def cyclic_postprocess_trigger_reasons(
    checkpoints: list[RefinementCheckpoint] | tuple[RefinementCheckpoint, ...],
    *,
    config: CyclicPostprocessConfig,
    state: CyclicPostprocessState | None = None,
) -> list[str]:
    """Return trigger reasons for qf-guarded cyclic postprocess.

    Empty output means no postprocess should be attempted at this checkpoint.
    """

    if not config.enabled:
        return []
    if not checkpoints:
        return []
    state = state or CyclicPostprocessState()
    latest = checkpoints[-1]
    if state.n_calls >= int(config.max_calls):
        return []
    if latest.step < int(config.warmup_steps):
        return []
    if (
        state.last_call_step is not None
        and latest.step - state.last_call_step < int(config.cooldown_steps)
    ):
        return []
    if config.require_oversize and latest.n_above_max_doc_weight <= 0:
        return []

    reasons: list[str] = []
    if config.interval_steps and latest.step % int(config.interval_steps) == 0:
        reasons.append("interval")

    if config.plateau_window and len(checkpoints) > int(config.plateau_window):
        previous = checkpoints[-int(config.plateau_window) - 1]
        quality_gain = float(latest.quality) - float(previous.quality)
        if quality_gain <= float(config.plateau_min_delta_q) + 1e-9:
            reasons.append("quality_plateau")

    if config.trigger_on_no_applied and config.no_applied_window:
        recent = checkpoints[-int(config.no_applied_window) :]
        if (
            len(recent) == int(config.no_applied_window)
            and all(
                checkpoint.applied_parent_count == 0
                for checkpoint in recent
                if checkpoint.applied_parent_count is not None
            )
            and all(checkpoint.applied_parent_count is not None for checkpoint in recent)
        ):
            reasons.append("no_applied_parents")

    return reasons


def _state_after_cyclic_call(
    state: CyclicPostprocessState | None,
    step: int,
) -> CyclicPostprocessState:
    state = state or CyclicPostprocessState()
    return CyclicPostprocessState(
        n_calls=int(state.n_calls) + 1,
        last_call_step=int(step),
    )


def run_cyclic_postprocess_if_due(
    graph: Any,
    *,
    raw_membership: np.ndarray,
    current_membership: np.ndarray,
    node_weights: np.ndarray,
    resolution: float,
    min_doc_weight: float,
    target_max_doc_weight: float,
    checkpoints: list[RefinementCheckpoint] | tuple[RefinementCheckpoint, ...],
    config: CyclicPostprocessConfig,
    seed: int,
    state: CyclicPostprocessState | None = None,
    output_dir: Path | None = None,
    postprocess_runner: Callable[..., LevelPostprocessResult] | None = None,
) -> CyclicPostprocessDecision:
    """Run qf-guarded postprocess when the cyclic trigger fires.

    This helper is intentionally orchestration-only: refinement code can call it
    after a checkpoint without changing the lower-level postprocess mechanics.
    """

    current_membership = np.asarray(current_membership, dtype=np.uint64)
    state = state or CyclicPostprocessState()
    reasons = cyclic_postprocess_trigger_reasons(
        checkpoints,
        config=config,
        state=state,
    )
    if not reasons:
        return CyclicPostprocessDecision(
            membership=current_membership,
            triggered=False,
            accepted=False,
            status="not_triggered",
            reasons=[],
            state=state,
        )

    latest = checkpoints[-1]
    state_after = _state_after_cyclic_call(state, latest.step)
    quality_before = float(
        graph.cpm_quality(membership=current_membership, resolution=float(resolution))
    )
    runner = postprocess_runner or run_hierarchy_level_postprocess
    result = runner(
        graph,
        raw_membership=np.asarray(raw_membership, dtype=np.uint64),
        small_membership=current_membership,
        node_weights=np.asarray(node_weights, dtype=np.float64),
        resolution=float(resolution),
        min_doc_weight=float(min_doc_weight),
        target_max_doc_weight=float(target_max_doc_weight),
        config=config.postprocess_config,
        seed=int(seed),
        output_dir=output_dir,
    )
    candidate_membership = np.asarray(result.membership, dtype=np.uint64)
    quality_after = float(
        graph.cpm_quality(membership=candidate_membership, resolution=float(resolution))
    )
    required = quality_before + float(config.postprocess_config.quality_floor_delta)
    quality_ok = quality_after >= required - 1e-9
    accepted = bool(result.accepted and quality_ok)
    if accepted:
        return CyclicPostprocessDecision(
            membership=candidate_membership,
            triggered=True,
            accepted=True,
            status=result.status,
            reasons=[*reasons, "accepted"],
            state=state_after,
            result=result,
            quality_before=quality_before,
            quality_after=quality_after,
        )

    status = "quality_guard_rejected" if not quality_ok else result.status
    return CyclicPostprocessDecision(
        membership=current_membership,
        triggered=True,
        accepted=False,
        status=status,
        reasons=[*reasons, "rejected"],
        state=state_after,
        result=result,
        quality_before=quality_before,
        quality_after=quality_after,
    )


def hierarchy_target_max_doc_weight(
    total_current_doc_weight: float,
    target_pct: float,
) -> float:
    """Compute the level-local max doc-weight target from hierarchy percentage."""

    return float(total_current_doc_weight) * float(target_pct) / 100.0


def trim_min_delta_q_for_policy(config: HierarchyPostprocessConfig) -> float:
    """Return the boundary-trim delta bound implied by the oversize policy."""

    if config.oversize_policy == "hard_cap":
        return float(config.hard_cap_trim_min_delta_q)
    return float(config.quality_first_trim_min_delta_q)


def _cluster_weight_arrays(
    membership: np.ndarray,
    node_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    membership_i64 = np.asarray(membership, dtype=np.int64)
    minlength = int(membership_i64.max()) + 1 if membership_i64.size else 0
    counts = np.bincount(membership_i64, minlength=minlength)
    weights = np.bincount(
        membership_i64,
        weights=np.asarray(node_weights, dtype=np.float64),
        minlength=minlength,
    )
    return counts, weights


def membership_weight_summary(
    membership: np.ndarray,
    node_weights: np.ndarray,
    *,
    min_weight: float = 0.0,
    max_weight: float = 0.0,
) -> dict[str, Any]:
    """Summarize cluster doc weights for small/oversize postprocess reporting."""

    counts, weights = _cluster_weight_arrays(membership, node_weights)
    active = counts > 0
    active_counts = counts[active]
    active_weights = weights[active]
    return {
        "n_clusters": int(active_weights.size),
        "max_doc_weight": float(active_weights.max()) if active_weights.size else 0.0,
        "n_above_max_doc_weight": (
            int((active_weights > max_weight).sum()) if max_weight > 0.0 else 0
        ),
        "n_singletons": int((active_counts == 1).sum()),
        "n_lt_min_doc_weight": (
            int((active_weights < min_weight).sum()) if min_weight > 0.0 else 0
        ),
        "n_lt_25_doc_weight": int((active_weights < 25.0).sum()),
        "n_lt_50_doc_weight": int((active_weights < 50.0).sum()),
        "top10_doc_weights": [float(x) for x in np.sort(active_weights)[::-1][:10]],
    }


def current_oversize_candidate_clusters(
    membership: np.ndarray,
    node_weights: np.ndarray,
    *,
    max_weight: float,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> np.ndarray:
    """Return current oversize cluster ids sorted by descending doc weight."""

    if max_weight <= 0.0:
        return np.asarray([], dtype=np.uint64)
    counts, weights = _cluster_weight_arrays(membership, node_weights)
    candidate_clusters = np.flatnonzero((counts > 0) & (weights > max_weight))
    order = np.lexsort((candidate_clusters, -weights[candidate_clusters]))
    if max_candidates > 0:
        order = order[:max_candidates]
    return np.asarray(candidate_clusters[order], dtype=np.uint64)


def _small_membership_view(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "n_clusters": int(stats["n_clusters"]),
        "n_singletons": int(stats["n_singletons"]),
        "n_lt_min_doc_weight": int(stats["n_lt_min_doc_weight"]),
    }


def _oversize_membership_view(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "n_clusters": int(stats["n_clusters"]),
        "n_above_max_doc_weight": int(stats["n_above_max_doc_weight"]),
        "max_doc_weight": float(stats["max_doc_weight"]),
        "top10_doc_weights": [float(x) for x in stats.get("top10_doc_weights", [])],
    }


def _membership_delta(
    after: dict[str, Any],
    before: dict[str, Any],
    keys: list[str],
) -> dict[str, Any]:
    return {key: after[key] - before[key] for key in keys}


def _quality_floor_prefix_move_count(
    delta_q: np.ndarray,
    *,
    quality_before: float,
    quality_floor: float,
) -> int:
    """Return the longest trim prefix whose final quality meets the floor."""

    deltas = np.asarray(delta_q, dtype=np.float64)
    if deltas.size == 0:
        return 0
    min_delta_q = float(quality_floor) - float(quality_before)
    cumulative = np.cumsum(deltas)
    valid = np.flatnonzero(cumulative >= min_delta_q - 1e-9)
    return int(valid[-1] + 1) if valid.size else 0


def _trim_prefix_membership(
    membership: np.ndarray,
    raw_trim: dict[str, np.ndarray],
    n_moves: int,
) -> np.ndarray:
    proposed = np.asarray(membership, dtype=np.uint64).copy()
    if n_moves <= 0:
        return proposed
    nodes = np.asarray(raw_trim["node"][:n_moves], dtype=np.int64)
    targets = np.asarray(raw_trim["target"][:n_moves], dtype=np.uint64)
    proposed[nodes] = targets
    return proposed


def _trim_source_move_counts(
    raw_trim: dict[str, np.ndarray],
    candidate_clusters: np.ndarray,
    n_moves: int,
) -> list[dict[str, Any]]:
    counts = {int(cluster): 0 for cluster in np.asarray(candidate_clusters).tolist()}
    if n_moves > 0:
        sources = np.asarray(raw_trim["source"][:n_moves], dtype=np.uint64)
        unique, move_counts = np.unique(sources, return_counts=True)
        for source, count in zip(unique, move_counts, strict=False):
            counts[int(source)] = int(count)
    return [
        {"cluster": cluster, "moves": moves}
        for cluster, moves in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def _oversize_residual_summary(
    membership: np.ndarray,
    node_weights: np.ndarray,
    *,
    max_weight: float,
    top_k: int = 10,
) -> dict[str, Any]:
    if max_weight <= 0.0:
        return {
            "n_above_max_doc_weight": 0,
            "max_doc_weight": 0.0,
            "excess_doc_weight_total": 0.0,
            "top_excess_clusters": [],
        }
    counts, weights = _cluster_weight_arrays(membership, node_weights)
    active = (counts > 0) & (weights > max_weight)
    clusters = np.flatnonzero(active)
    if clusters.size == 0:
        return {
            "n_above_max_doc_weight": 0,
            "max_doc_weight": float(weights[counts > 0].max()) if np.any(counts > 0) else 0.0,
            "excess_doc_weight_total": 0.0,
            "top_excess_clusters": [],
        }
    excess = weights[clusters] - max_weight
    order = np.lexsort((clusters, -excess))
    top_clusters = clusters[order[:top_k]]
    return {
        "n_above_max_doc_weight": int(clusters.size),
        "max_doc_weight": float(weights[clusters].max()),
        "excess_doc_weight_total": float(excess.sum()),
        "top_excess_clusters": [
            {
                "cluster": int(cluster),
                "doc_weight": float(weights[cluster]),
                "excess_doc_weight": float(weights[cluster] - max_weight),
            }
            for cluster in top_clusters
        ],
    }


def _trim_infeasibility_diagnostics(
    *,
    raw_trim: dict[str, np.ndarray],
    candidate_clusters: np.ndarray,
    committed_membership: np.ndarray,
    proposed_membership: np.ndarray,
    node_weights: np.ndarray,
    target_max_weight: float,
    trim_min_delta_q: float,
    max_moves_per_cluster: int,
    n_moves_committed: int,
    n_moves_proposed: int,
    quality_floor: float,
    quality_after_committed: float,
    quality_after_proposed: float,
) -> dict[str, Any]:
    committed_residual = _oversize_residual_summary(
        committed_membership,
        node_weights,
        max_weight=target_max_weight,
    )
    proposed_residual = _oversize_residual_summary(
        proposed_membership,
        node_weights,
        max_weight=target_max_weight,
    )
    target_max_satisfied = committed_residual["n_above_max_doc_weight"] == 0
    proposed_target_max_satisfied = proposed_residual["n_above_max_doc_weight"] == 0
    quality_floor_limited = bool(n_moves_committed < n_moves_proposed)
    source_moves_proposed = _trim_source_move_counts(
        raw_trim,
        candidate_clusters,
        n_moves_proposed,
    )
    move_budget_exhausted = (
        max_moves_per_cluster > 0
        and any(row["moves"] >= max_moves_per_cluster for row in source_moves_proposed)
        and not proposed_target_max_satisfied
    )

    inferred_blockers: list[str] = []
    if target_max_satisfied:
        inferred_blockers.append("target_satisfied")
    else:
        if n_moves_proposed == 0:
            inferred_blockers.append("no_candidate_boundary_moves")
        if quality_floor_limited:
            inferred_blockers.append("quality_floor")
        if move_budget_exhausted:
            inferred_blockers.append("move_budget")
        if (
            n_moves_proposed > 0
            and not proposed_target_max_satisfied
            and not move_budget_exhausted
        ):
            inferred_blockers.append("trim_delta_bound_or_receiver_cap")
        if not inferred_blockers:
            inferred_blockers.append("unclassified")

    return {
        "target_max_satisfied": bool(target_max_satisfied),
        "proposed_target_max_satisfied": bool(proposed_target_max_satisfied),
        "quality_floor_limited": quality_floor_limited,
        "quality_floor_margin_committed": float(quality_after_committed - quality_floor),
        "quality_floor_margin_proposed": float(quality_after_proposed - quality_floor),
        "move_budget_exhausted": bool(move_budget_exhausted),
        "trim_min_delta_q": float(trim_min_delta_q),
        "max_moves_per_cluster": int(max_moves_per_cluster),
        "n_moves_committed": int(n_moves_committed),
        "n_moves_proposed": int(n_moves_proposed),
        "source_move_counts_committed": _trim_source_move_counts(
            raw_trim,
            candidate_clusters,
            n_moves_committed,
        ),
        "source_move_counts_proposed": source_moves_proposed,
        "committed_oversize_residual": committed_residual,
        "proposed_oversize_residual": proposed_residual,
        "inferred_blockers": inferred_blockers,
    }


def _as_array_dict(result: Any) -> dict[str, np.ndarray]:
    if isinstance(result, dict):
        return {key: np.asarray(value) for key, value in result.items()}
    return {
        name: np.asarray(getattr(result, name))
        for name in getattr(result, "__dataclass_fields__", {})
    }


def _write_empty_trim_rows(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        csv.DictWriter(fh, fieldnames=TRIM_FIELDS).writeheader()


def _write_trim_move_rows(
    path: Path,
    raw_trim: dict[str, np.ndarray],
    *,
    n_moves_committed: int | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n_moves = int(raw_trim["node"].shape[0])
    if n_moves_committed is None:
        n_moves_committed = n_moves
    n_moves_committed = max(0, min(int(n_moves_committed), n_moves))
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=TRIM_FIELDS)
        writer.writeheader()
        for idx in range(n_moves):
            writer.writerow(
                {
                    "move_index": idx,
                    "committed": idx < n_moves_committed,
                    "source": int(raw_trim["source"][idx]),
                    "target": int(raw_trim["target"][idx]),
                    "node": int(raw_trim["node"][idx]),
                    "node_weight": float(raw_trim["node_weight"][idx]),
                    "delta_q": float(raw_trim["delta_q"][idx]),
                    "source_weight_before": float(raw_trim["source_weight_before"][idx]),
                    "source_weight_after": float(raw_trim["source_weight_after"][idx]),
                    "target_weight_before": float(raw_trim["target_weight_before"][idx]),
                    "target_weight_after": float(raw_trim["target_weight_after"][idx]),
                }
            )


def _write_current_membership(path: Path, membership: np.ndarray) -> None:
    import polars as pl

    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "node_idx": np.arange(membership.shape[0], dtype=np.uint64),
            "cluster": np.asarray(membership, dtype=np.uint64),
        }
    ).write_parquet(path)


_DONGDAEMUN_STATUS_BY_CODE = {
    0: "no_current_oversize_candidates",
    1: "committed",
    2: "no_selected_candidates",
    3: "no_progress",
    4: "split_quality_below_floor",
    5: "trim_quality_below_floor",
    6: "quality_below_floor",
    7: "hard_cap_not_satisfied",
}


def _rust_dongdaemun_audit_summary(audit: Any) -> dict[str, Any]:
    return {
        "accepted": bool(audit.accepted),
        "status": str(audit.status),
        "quality_before": float(audit.quality_before),
        "quality_after_candidate": float(audit.quality_after_candidate),
        "candidate_delta_q": float(audit.candidate_delta_q),
        "effective_delta_q": float(audit.effective_delta_q),
        "final_delta_q": float(audit.final_delta_q),
        "target_max_satisfied": bool(audit.target_max_satisfied),
        "n_oversize_before": int(audit.n_oversize_before),
        "n_oversize_after_candidate": int(audit.n_oversize_after_candidate),
        "max_weight_before": float(audit.max_weight_before),
        "max_weight_after_candidate": float(audit.max_weight_after_candidate),
        "trim_moves_committed": int(audit.trim_moves_committed),
        "trim_moves_proposed": int(audit.trim_moves_proposed),
    }


def _rust_dongdaemun_split_iterations(audit: Any) -> list[dict[str, Any]]:
    iterations = np.asarray(audit.split_iteration, dtype=np.uint64)
    candidate_clusters = np.asarray(audit.split_candidate_clusters, dtype=np.uint64)
    n_selected = np.asarray(audit.split_n_selected, dtype=np.uint64)
    n_applied = np.asarray(audit.split_n_applied, dtype=np.uint64)
    status_codes = np.asarray(audit.split_status_code, dtype=np.uint8)
    exact_delta_q = np.asarray(audit.split_exact_delta_q, dtype=np.float64)
    return [
        {
            "iteration": int(iterations[idx]),
            "candidate_clusters": int(candidate_clusters[idx]),
            "n_selected": int(n_selected[idx]),
            "n_applied": int(n_applied[idx]),
            "status": _DONGDAEMUN_STATUS_BY_CODE.get(
                int(status_codes[idx]),
                f"unknown_{int(status_codes[idx])}",
            ),
            "exact_delta_q": float(exact_delta_q[idx]),
        }
        for idx in range(int(iterations.shape[0]))
    ]


def _graph_has_rust_dongdaemun(graph: Any) -> bool:
    if not callable(getattr(graph, "dongdaemun_refine", None)):
        return False
    inner_graph = getattr(graph, "graph", None)
    return inner_graph is not None and callable(
        getattr(inner_graph, "dongdaemun_refine", None)
    )


def _run_rust_dongdaemun_postprocess(
    graph: Any,
    *,
    small_membership: np.ndarray,
    node_weights: np.ndarray,
    resolution: float,
    min_doc_weight: float,
    target_max_doc_weight: float,
    config: HierarchyPostprocessConfig,
    seed: int,
    small_after_stats: dict[str, Any],
    small_summary: dict[str, Any],
    paths: dict[str, str],
) -> LevelPostprocessResult:
    rust_result = graph.dongdaemun_refine(
        np.asarray(small_membership, dtype=np.uint64),
        resolution=float(resolution),
        target_max_weight=float(target_max_doc_weight),
        gamma_multipliers=DEFAULT_GAMMA_MULTIPLIERS,
        policy=str(config.oversize_policy),
        quality_floor_delta=float(config.quality_floor_delta),
        apply_iterations=int(config.apply_iterations),
        min_core_weight=DEFAULT_MIN_CORE_WEIGHT,
        randomness=DEFAULT_RANDOMNESS,
        repair_epsilon=DEFAULT_REPAIR_EPSILON,
        trim_min_delta_q_quality_first=float(config.quality_first_trim_min_delta_q),
        trim_min_delta_q_hard_cap=float(config.hard_cap_trim_min_delta_q),
        trim_max_moves_per_cluster=int(config.trim_max_moves_per_cluster),
        seed=int(seed),
        pair_seeded=True,
    )
    result_membership = np.asarray(rust_result.membership, dtype=np.uint64)
    final_stats = membership_weight_summary(
        result_membership,
        node_weights,
        min_weight=float(min_doc_weight),
        max_weight=float(target_max_doc_weight),
    )
    audit = rust_result.audit
    split_iterations = _rust_dongdaemun_split_iterations(audit)
    split_repair_exact_delta_q = float(
        np.asarray(audit.split_exact_delta_q, dtype=np.float64).sum()
    )
    changed_nodes = int(
        np.count_nonzero(
            np.asarray(small_membership, dtype=np.uint64) != result_membership
        )
    )
    target_max_satisfied = final_stats["n_above_max_doc_weight"] == 0
    oversize_summary = {
        "backend": "rust_dongdaemun",
        "before": _oversize_membership_view(small_after_stats),
        "after": _oversize_membership_view(final_stats),
        "delta": _membership_delta(
            _oversize_membership_view(final_stats),
            _oversize_membership_view(small_after_stats),
            ["n_clusters", "n_above_max_doc_weight", "max_doc_weight"],
        ),
        "changed_nodes": changed_nodes,
        "changed_nodes_step_sum": changed_nodes,
        "split_repair_exact_delta_q": split_repair_exact_delta_q,
        "trim_exact_delta_q": float(audit.effective_delta_q - split_repair_exact_delta_q),
        "final_exact_delta_q": float(audit.effective_delta_q),
        "predicted_delta_q_sum_total": 0.0,
        "target_max_satisfied": bool(target_max_satisfied),
        "stop_reason": str(audit.status),
        "iterations": split_iterations,
        "trim": {
            "status": "committed"
            if int(audit.trim_moves_committed) > 0
            else "no_trim_moves",
            "n_moves": int(audit.trim_moves_committed),
            "n_moves_proposed": int(audit.trim_moves_proposed),
            "n_moves_committed": int(audit.trim_moves_committed),
            "target_max_satisfied": bool(target_max_satisfied),
            "min_delta_q": trim_min_delta_q_for_policy(config),
        },
        "rust_audit": _rust_dongdaemun_audit_summary(audit),
    }
    return LevelPostprocessResult(
        membership=result_membership,
        accepted=bool(audit.accepted),
        status=str(audit.status),
        small_cluster_summary=small_summary,
        oversize_summary=oversize_summary,
        final_summary=final_stats,
        paths=paths,
        backend="rust_dongdaemun",
    )


def _apply_selected_candidates(
    graph: Any,
    membership: np.ndarray,
    candidate_clusters: np.ndarray,
    selection_rows: list[dict[str, Any]],
    *,
    resolution: float,
    seed: int,
    gamma_multipliers: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray | None]:
    selected_rows = [row for row in selection_rows if row["selected_for_apply"]]
    if not selected_rows:
        return {
            "status": "no_selected_candidates",
            "n_selected": 0,
            "n_applied": 0,
            "n_missing_candidates": 0,
            "exact_delta_q": 0.0,
            "predicted_delta_q_sum": 0.0,
            "changed_nodes": 0,
        }, None

    selected_clusters = np.asarray([row["cluster"] for row in selected_rows], dtype=np.uint64)
    selected_gamma_multipliers = np.asarray(
        [row["gamma_multiplier"] for row in selected_rows],
        dtype=np.float64,
    )
    quality_before = float(graph.cpm_quality(membership=membership, resolution=resolution))
    raw_apply = graph.apply_split_merge_repair_candidates(
        membership,
        candidate_clusters,
        selected_clusters,
        selected_gamma_multipliers,
        resolution=float(resolution),
        gamma_multipliers=gamma_multipliers,
        min_core_weight=DEFAULT_MIN_CORE_WEIGHT,
        randomness=DEFAULT_RANDOMNESS,
        repair_epsilon=DEFAULT_REPAIR_EPSILON,
        seed=int(seed),
    )
    apply_dict = _as_array_dict(raw_apply)
    proposed_membership = np.asarray(apply_dict["membership"], dtype=np.uint64)
    quality_after = float(
        graph.cpm_quality(membership=proposed_membership, resolution=resolution)
    )
    exact_delta_q = quality_after - quality_before
    n_selected = len(selected_rows)
    n_applied = int(apply_dict["cluster"].shape[0])
    missing_candidates = n_selected - n_applied
    status = "committed"
    if missing_candidates:
        status = "rolled_back_missing_candidates"
    elif exact_delta_q < 0.0:
        status = "rolled_back_quality_below_threshold"

    summary = {
        "status": status,
        "n_selected": int(n_selected),
        "n_applied": int(n_applied),
        "n_missing_candidates": int(missing_candidates),
        "quality_before": quality_before,
        "quality_after_proposed": quality_after,
        "exact_delta_q": float(exact_delta_q),
        "min_quality_delta": 0.0,
        "predicted_delta_q_sum": (
            float(apply_dict["predicted_net_delta_q"].sum()) if n_applied else 0.0
        ),
        "changed_nodes": int(apply_dict["changed_nodes"].sum()) if n_applied else 0,
        "moved_to_existing_cluster_nodes": (
            int(apply_dict["moved_to_existing_cluster_nodes"].sum()) if n_applied else 0
        ),
        "moved_to_new_cluster_nodes": (
            int(apply_dict["moved_to_new_cluster_nodes"].sum()) if n_applied else 0
        ),
        "new_retained_clusters": (
            int(apply_dict["new_retained_clusters"].sum()) if n_applied else 0
        ),
    }
    return summary, proposed_membership if status == "committed" else None


def _apply_oversize_boundary_trim(
    graph: Any,
    membership: np.ndarray,
    node_weights: np.ndarray,
    *,
    resolution: float,
    min_doc_weight: float,
    target_max_doc_weight: float,
    trim_min_delta_q: float,
    trim_max_moves_per_cluster: int,
    quality_floor: float,
    moves_path: Path | None,
) -> tuple[dict[str, Any], np.ndarray | None]:
    candidate_clusters = current_oversize_candidate_clusters(
        membership,
        node_weights,
        max_weight=float(target_max_doc_weight),
        max_candidates=DEFAULT_MAX_CANDIDATES,
    )
    before_stats = membership_weight_summary(
        membership,
        node_weights,
        min_weight=float(min_doc_weight),
        max_weight=float(target_max_doc_weight),
    )
    if candidate_clusters.size == 0:
        if moves_path is not None:
            _write_empty_trim_rows(moves_path)
        summary = {
            "status": "no_current_oversize_candidates",
            "candidate_clusters": 0,
            "n_moves": 0,
            "n_moves_proposed": 0,
            "n_moves_committed": 0,
            "target_max_satisfied": before_stats["n_above_max_doc_weight"] == 0,
            "min_delta_q": float(trim_min_delta_q),
            "before_membership": before_stats,
            "after_membership": before_stats,
        }
        return summary, None

    quality_before = float(graph.cpm_quality(membership=membership, resolution=resolution))
    raw_trim_result = graph.trim_oversize_boundary_moves(
        membership,
        candidate_clusters,
        resolution=float(resolution),
        target_max_weight=float(target_max_doc_weight),
        min_delta_q=float(trim_min_delta_q),
        max_moves_per_cluster=int(trim_max_moves_per_cluster),
    )
    raw_trim = _as_array_dict(raw_trim_result)
    proposed_membership = np.asarray(raw_trim["membership"], dtype=np.uint64)
    quality_after_proposed = float(
        graph.cpm_quality(membership=proposed_membership, resolution=resolution)
    )
    exact_delta_q_proposed = quality_after_proposed - quality_before
    n_moves_proposed = int(raw_trim["node"].shape[0])
    after_stats = before_stats
    status = "no_trim_moves"
    trim_commit_reason = "none"
    n_moves_committed = 0
    committed_membership = np.asarray(membership, dtype=np.uint64)
    quality_after_committed = quality_before

    if n_moves_proposed and quality_after_proposed < quality_floor:
        n_moves_committed = _quality_floor_prefix_move_count(
            raw_trim["delta_q"],
            quality_before=quality_before,
            quality_floor=quality_floor,
        )
        committed_membership = _trim_prefix_membership(
            membership,
            raw_trim,
            n_moves_committed,
        )
        quality_after_committed = float(
            graph.cpm_quality(membership=committed_membership, resolution=resolution)
        )
        while n_moves_committed > 0 and quality_after_committed < quality_floor:
            n_moves_committed -= 1
            committed_membership = _trim_prefix_membership(
                membership,
                raw_trim,
                n_moves_committed,
            )
            quality_after_committed = float(
                graph.cpm_quality(membership=committed_membership, resolution=resolution)
            )
        if n_moves_committed:
            status = "committed"
            trim_commit_reason = "quality_floor_prefix"
        else:
            status = "rolled_back_quality_below_threshold"
            trim_commit_reason = "quality_floor"
    elif n_moves_proposed:
        status = "committed"
        trim_commit_reason = "all_proposed"
        n_moves_committed = n_moves_proposed
        committed_membership = proposed_membership
        quality_after_committed = quality_after_proposed

    if moves_path is not None:
        _write_trim_move_rows(raw_trim=raw_trim, path=moves_path, n_moves_committed=n_moves_committed)

    exact_delta_q = quality_after_committed - quality_before
    if status == "committed":
        after_stats = membership_weight_summary(
            committed_membership,
            node_weights,
            min_weight=float(min_doc_weight),
            max_weight=float(target_max_doc_weight),
        )
    proposed_stats = membership_weight_summary(
        proposed_membership,
        node_weights,
        min_weight=float(min_doc_weight),
        max_weight=float(target_max_doc_weight),
    )
    trim_diagnostics = _trim_infeasibility_diagnostics(
        raw_trim=raw_trim,
        candidate_clusters=candidate_clusters,
        committed_membership=committed_membership,
        proposed_membership=proposed_membership,
        node_weights=node_weights,
        target_max_weight=float(target_max_doc_weight),
        trim_min_delta_q=float(trim_min_delta_q),
        max_moves_per_cluster=int(trim_max_moves_per_cluster),
        n_moves_committed=n_moves_committed,
        n_moves_proposed=n_moves_proposed,
        quality_floor=quality_floor,
        quality_after_committed=quality_after_committed,
        quality_after_proposed=quality_after_proposed,
    )
    summary = {
        "status": status,
        "candidate_clusters": int(candidate_clusters.size),
        "candidate_cluster_ids": [int(x) for x in candidate_clusters.tolist()],
        "n_moves": int(n_moves_committed),
        "n_moves_proposed": int(n_moves_proposed),
        "n_moves_committed": int(n_moves_committed),
        "quality_before": quality_before,
        "quality_after_proposed": quality_after_proposed,
        "quality_after_committed": quality_after_committed,
        "quality_floor": float(quality_floor),
        "target_max_satisfied": trim_diagnostics["target_max_satisfied"],
        "exact_delta_q": float(exact_delta_q),
        "exact_delta_q_proposed": float(exact_delta_q_proposed),
        "predicted_delta_q_sum": (
            float(raw_trim["delta_q"][:n_moves_committed].sum())
            if n_moves_committed
            else 0.0
        ),
        "predicted_delta_q_sum_proposed": (
            float(raw_trim["delta_q"].sum()) if n_moves_proposed else 0.0
        ),
        "quality_floor_limited": bool(n_moves_committed < n_moves_proposed),
        "trim_commit_reason": trim_commit_reason,
        "min_delta_q": float(trim_min_delta_q),
        "max_moves_per_cluster": int(trim_max_moves_per_cluster),
        "changed_nodes": int(np.count_nonzero(committed_membership != membership)),
        "before_membership": before_stats,
        "proposed_membership": proposed_stats,
        "after_membership": after_stats,
        "trim_diagnostics": trim_diagnostics,
    }
    return summary, committed_membership if status == "committed" else None


def run_hierarchy_level_postprocess(
    graph: Any,
    *,
    raw_membership: np.ndarray,
    small_membership: np.ndarray,
    node_weights: np.ndarray,
    resolution: float,
    min_doc_weight: float,
    target_max_doc_weight: float,
    config: HierarchyPostprocessConfig,
    seed: int,
    output_dir: Path | None = None,
) -> LevelPostprocessResult:
    """Run opt-in oversize postprocess for one hierarchy level.

    ``raw_membership`` is the Leiden result before the existing small-cluster
    repair. ``small_membership`` is the current baseline that will be used as
    the fallback if oversize postprocess is rejected.
    """

    if not config.enabled:
        small_after = membership_weight_summary(
            small_membership,
            node_weights,
            min_weight=float(min_doc_weight),
            max_weight=float(target_max_doc_weight),
        )
        return LevelPostprocessResult(
            membership=np.asarray(small_membership, dtype=np.uint64),
            accepted=True,
            status="disabled",
            small_cluster_summary={"after": _small_membership_view(small_after)},
            oversize_summary={},
            final_summary=small_after,
        )

    output_dir = Path(output_dir) if output_dir is not None and config.write_artifacts else None
    paths: dict[str, str] = {}
    summary_path = output_dir / "summary.json" if output_dir is not None else None
    moves_path = (
        output_dir / "oversize_boundary_trim_moves.csv"
        if output_dir is not None
        else None
    )
    if summary_path is not None:
        paths["summary"] = str(summary_path)
    if moves_path is not None:
        paths["oversize_boundary_trim_moves"] = str(moves_path)

    raw_membership = np.asarray(raw_membership, dtype=np.uint64)
    small_membership = np.asarray(small_membership, dtype=np.uint64)
    node_weights = np.asarray(node_weights, dtype=np.float64)

    small_before_stats = membership_weight_summary(
        raw_membership,
        node_weights,
        min_weight=float(min_doc_weight),
        max_weight=float(target_max_doc_weight),
    )
    small_after_stats = membership_weight_summary(
        small_membership,
        node_weights,
        min_weight=float(min_doc_weight),
        max_weight=float(target_max_doc_weight),
    )
    small_summary = {
        "before": _small_membership_view(small_before_stats),
        "after": _small_membership_view(small_after_stats),
        "delta": _membership_delta(
            _small_membership_view(small_after_stats),
            _small_membership_view(small_before_stats),
            ["n_clusters", "n_singletons", "n_lt_min_doc_weight"],
        ),
        "changed_nodes": int(np.count_nonzero(raw_membership != small_membership)),
    }

    if config.use_rust_dongdaemun and _graph_has_rust_dongdaemun(graph) and output_dir is None:
        result = _run_rust_dongdaemun_postprocess(
            graph,
            small_membership=small_membership,
            node_weights=node_weights,
            resolution=float(resolution),
            min_doc_weight=float(min_doc_weight),
            target_max_doc_weight=float(target_max_doc_weight),
            config=config,
            seed=seed,
            small_after_stats=small_after_stats,
            small_summary=small_summary,
            paths=paths,
        )
        _write_level_summary(
            summary_path,
            result,
            config=config,
            min_doc_weight=min_doc_weight,
            target_max_doc_weight=target_max_doc_weight,
            quality_delta=float(result.oversize_summary.get("final_exact_delta_q", 0.0)),
        )
        return result

    baseline_membership = small_membership.copy()
    current_membership = small_membership.copy()
    gamma_multipliers = np.asarray(DEFAULT_GAMMA_MULTIPLIERS, dtype=np.float64)
    quality_before = float(
        graph.cpm_quality(membership=baseline_membership, resolution=resolution)
    )
    quality_floor = quality_before + float(config.quality_floor_delta)
    iterations: list[dict[str, Any]] = []
    stop_reason = "max_iterations_reached"
    split_repair_exact_delta_q = 0.0
    predicted_delta_q_sum_total = 0.0
    changed_nodes_step_sum = 0

    initial_candidates = current_oversize_candidate_clusters(
        current_membership,
        node_weights,
        max_weight=float(target_max_doc_weight),
        max_candidates=DEFAULT_MAX_CANDIDATES,
    )
    if initial_candidates.size == 0:
        if moves_path is not None:
            _write_empty_trim_rows(moves_path)
        final_stats = small_after_stats
        oversize_summary = {
            "before": _oversize_membership_view(small_after_stats),
            "after": _oversize_membership_view(final_stats),
            "delta": _membership_delta(
                _oversize_membership_view(final_stats),
                _oversize_membership_view(small_after_stats),
                ["n_clusters", "n_above_max_doc_weight", "max_doc_weight"],
            ),
            "changed_nodes": 0,
            "split_repair_exact_delta_q": 0.0,
            "trim_exact_delta_q": 0.0,
            "final_exact_delta_q": 0.0,
            "target_max_satisfied": final_stats["n_above_max_doc_weight"] == 0,
            "stop_reason": "no_current_oversize_candidates",
        }
        result = LevelPostprocessResult(
            membership=baseline_membership,
            accepted=True,
            status="no_current_oversize_candidates",
            small_cluster_summary=small_summary,
            oversize_summary=oversize_summary,
            final_summary=final_stats,
            paths=paths,
        )
        _write_level_summary(
            summary_path,
            result,
            config=config,
            min_doc_weight=min_doc_weight,
            target_max_doc_weight=target_max_doc_weight,
            quality_delta=0.0,
        )
        return result

    for iteration in range(1, int(config.apply_iterations) + 1):
        before_stats = membership_weight_summary(
            current_membership,
            node_weights,
            min_weight=float(min_doc_weight),
            max_weight=float(target_max_doc_weight),
        )
        candidate_clusters = current_oversize_candidate_clusters(
            current_membership,
            node_weights,
            max_weight=float(target_max_doc_weight),
            max_candidates=DEFAULT_MAX_CANDIDATES,
        )
        if candidate_clusters.size == 0:
            stop_reason = "no_current_oversize_candidates"
            break

        probes = graph.split_merge_repair_probes(
            current_membership,
            candidate_clusters,
            resolution=float(resolution),
            gamma_multipliers=gamma_multipliers,
            min_core_weight=DEFAULT_MIN_CORE_WEIGHT,
            randomness=DEFAULT_RANDOMNESS,
            repair_epsilon=DEFAULT_REPAIR_EPSILON,
            seed=int(seed),
        )
        rows = rank_split_repair_candidates(
            probes,
            SplitRepairSelectionPolicy(
                name=str(config.selection_mode),
                mode=str(config.selection_mode),
                singleton_budget=DEFAULT_MIN_CORE_WEIGHT,
            ),
            min_weight=float(min_doc_weight),
            max_weight=float(target_max_doc_weight),
        )
        apply_summary, applied_membership = _apply_selected_candidates(
            graph,
            current_membership,
            candidate_clusters,
            rows,
            resolution=float(resolution),
            seed=int(seed),
            gamma_multipliers=gamma_multipliers,
        )
        after_stats = before_stats
        if apply_summary["status"] == "committed":
            assert applied_membership is not None
            current_membership = applied_membership
            after_stats = membership_weight_summary(
                current_membership,
                node_weights,
                min_weight=float(min_doc_weight),
                max_weight=float(target_max_doc_weight),
            )
            split_repair_exact_delta_q += float(apply_summary["exact_delta_q"])
            predicted_delta_q_sum_total += float(apply_summary["predicted_delta_q_sum"])
            changed_nodes_step_sum += int(apply_summary["changed_nodes"])

        iterations.append(
            {
                "iteration": iteration,
                "candidate_source": "current_oversize",
                "candidate_clusters": int(candidate_clusters.size),
                "candidate_cluster_ids": [int(x) for x in candidate_clusters.tolist()],
                "before_membership": before_stats,
                "after_membership": after_stats,
                "n_selected": int(apply_summary.get("n_selected", 0)),
                "n_applied": int(apply_summary.get("n_applied", 0)),
                "status": str(apply_summary.get("status", "")),
                "exact_delta_q": float(apply_summary.get("exact_delta_q", 0.0)),
                "predicted_delta_q_sum": float(
                    apply_summary.get("predicted_delta_q_sum", 0.0)
                ),
                "changed_nodes": int(apply_summary.get("changed_nodes", 0)),
            }
        )
        if apply_summary["status"] != "committed":
            stop_reason = str(apply_summary["status"])
            break
        if after_stats["n_above_max_doc_weight"] == 0:
            stop_reason = "target_max_satisfied"
            break

    trim_summary, trim_membership = _apply_oversize_boundary_trim(
        graph,
        current_membership,
        node_weights,
        resolution=float(resolution),
        min_doc_weight=float(min_doc_weight),
        target_max_doc_weight=float(target_max_doc_weight),
        trim_min_delta_q=trim_min_delta_q_for_policy(config),
        trim_max_moves_per_cluster=int(config.trim_max_moves_per_cluster),
        quality_floor=quality_floor,
        moves_path=moves_path,
    )
    trim_exact_delta_q = 0.0
    if trim_summary.get("status") == "committed":
        assert trim_membership is not None
        current_membership = trim_membership
        trim_exact_delta_q = float(trim_summary.get("exact_delta_q", 0.0))
        if trim_summary.get("target_max_satisfied"):
            stop_reason = "target_max_satisfied_after_trim"

    final_quality = float(graph.cpm_quality(membership=current_membership, resolution=resolution))
    final_exact_delta_q = final_quality - quality_before
    final_stats = membership_weight_summary(
        current_membership,
        node_weights,
        min_weight=float(min_doc_weight),
        max_weight=float(target_max_doc_weight),
    )
    target_max_satisfied = final_stats["n_above_max_doc_weight"] == 0
    quality_ok = final_exact_delta_q >= float(config.quality_floor_delta) - 1e-9
    if config.oversize_policy == "hard_cap":
        accepted = bool(quality_ok and target_max_satisfied)
    else:
        accepted = bool(quality_ok)

    changed_nodes_vs_initial = int(np.count_nonzero(baseline_membership != current_membership))
    if accepted:
        status = "committed" if changed_nodes_vs_initial else stop_reason
        result_membership = current_membership
    elif not quality_ok:
        status = "quality_below_floor"
        result_membership = baseline_membership
    else:
        status = "hard_cap_not_satisfied"
        result_membership = baseline_membership

    oversize_summary = {
        "before": _oversize_membership_view(small_after_stats),
        "after": _oversize_membership_view(final_stats),
        "delta": _membership_delta(
            _oversize_membership_view(final_stats),
            _oversize_membership_view(small_after_stats),
            ["n_clusters", "n_above_max_doc_weight", "max_doc_weight"],
        ),
        "changed_nodes": changed_nodes_vs_initial,
        "changed_nodes_step_sum": int(changed_nodes_step_sum),
        "split_repair_exact_delta_q": float(split_repair_exact_delta_q),
        "trim_exact_delta_q": float(trim_exact_delta_q),
        "final_exact_delta_q": float(final_exact_delta_q),
        "predicted_delta_q_sum_total": float(predicted_delta_q_sum_total),
        "target_max_satisfied": bool(target_max_satisfied),
        "stop_reason": stop_reason,
        "iterations": iterations,
        "trim": trim_summary,
    }

    if not accepted and changed_nodes_vs_initial and output_dir is not None:
        diagnostic_path = output_dir / "diagnostic_membership.parquet"
        _write_current_membership(diagnostic_path, current_membership)
        paths["diagnostic_membership"] = str(diagnostic_path)

    result = LevelPostprocessResult(
        membership=np.asarray(result_membership, dtype=np.uint64),
        accepted=accepted,
        status=status,
        small_cluster_summary=small_summary,
        oversize_summary=oversize_summary,
        final_summary=final_stats,
        paths=paths,
    )
    _write_level_summary(
        summary_path,
        result,
        config=config,
        min_doc_weight=min_doc_weight,
        target_max_doc_weight=target_max_doc_weight,
        quality_delta=final_exact_delta_q,
    )
    return result


def _write_level_summary(
    summary_path: Path | None,
    result: LevelPostprocessResult,
    *,
    config: HierarchyPostprocessConfig,
    min_doc_weight: float,
    target_max_doc_weight: float,
    quality_delta: float,
) -> None:
    if summary_path is None:
        return
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": result.status,
        "accepted": bool(result.accepted),
        "backend": result.backend,
        "oversize_policy": config.oversize_policy,
        "postprocess_config_hash": postprocess_config_hash(config),
        "target_min_doc_weight": float(min_doc_weight),
        "target_max_doc_weight": float(target_max_doc_weight),
        "target_max_satisfied": bool(
            result.oversize_summary.get("target_max_satisfied", False)
        ),
        "postprocess_quality_delta": float(quality_delta),
        "small_cluster_summary": result.small_cluster_summary,
        "oversize_summary": result.oversize_summary,
        "final_summary": result.final_summary,
        "paths": result.paths,
    }
    summary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


__all__ = [
    "CyclicPostprocessConfig",
    "CyclicPostprocessDecision",
    "CyclicPostprocessState",
    "HierarchyPostprocessConfig",
    "LevelPostprocessResult",
    "RefinementCheckpoint",
    "current_oversize_candidate_clusters",
    "cyclic_postprocess_trigger_reasons",
    "hierarchy_target_max_doc_weight",
    "membership_weight_summary",
    "postprocess_config_hash",
    "run_cyclic_postprocess_if_due",
    "run_hierarchy_level_postprocess",
    "trim_min_delta_q_for_policy",
]
