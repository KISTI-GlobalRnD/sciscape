"""Adaptive refinement diagnostics built on the Rust Leiden backend.

The functions here are observational. They summarize a baseline Leiden
membership and macro-merge dry-run candidates without changing the partition.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .leiden_rust import (
    RustBoundaryGroupProbes,
    RustBoundaryMoveProbes,
    RustClusterGraphStats,
    RustMultiCoreSplitProbes,
    RustSplitMergeRepairProbes,
)


@dataclass(frozen=True)
class MacroMergePolicy:
    """Dry-run policy for selecting non-conflicting macro-merge candidates."""

    name: str
    epsilon: float = 1e-4
    min_size_band_gain: float = 0.0
    max_leafness: float | None = None
    min_conductance: float | None = None
    require_any_below_min: bool = True
    require_merged_within_band: bool = False
    allow_singleton_endpoint: bool = True
    max_merged_weight: float | None = None
    max_merges: int | None = None
    q_debt_budget: float | None = None
    score_size_band_weight: float = 0.0
    score_conductance_weight: float = 0.0
    score_leafness_penalty: float = 0.0


@dataclass(frozen=True)
class MacroMergeSimulationResult:
    """Aggregate dry-run result for one macro-merge policy."""

    policy: MacroMergePolicy
    n_candidates_considered: int
    n_candidates_after_filters: int
    n_selected: int
    estimated_active_clusters_after: int
    sum_delta_q: float
    q_debt: float
    size_band_gain: float
    within_band_delta: int
    below_min_delta: int
    above_max_delta: int
    singleton_endpoint_pairs: int
    both_singleton_pairs: int
    selected_doc_weight_p50: float
    selected_doc_weight_p90: float
    selected_leafness_p90: float
    selected_conductance_p50: float

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["policy"] = asdict(self.policy)
        return out


@dataclass(frozen=True)
class BoundaryCandidatePolicy:
    """Policy for ranking ambiguous cluster-graph boundary candidates."""

    name: str
    min_block_count: int = 2
    min_doc_weight: float = 0.0
    max_doc_weight: float | None = None
    min_degree: int = 2
    min_conductance: float = 0.5
    max_leafness: float = 0.95
    min_neighbor_weight_ratio: float = 0.25
    exclude_singletons: bool = True
    top_k: int = 1000


@dataclass(frozen=True)
class SplitRepairSelectionPolicy:
    """Dry-run policy for ranking non-conflicting split-repair candidates."""

    name: str = "conservative"
    mode: str = "utility_cost"
    min_net_delta_q: float = 1e-6
    strong_net_delta_q: float = 1.0
    singleton_budget: float = 25.0
    min_retained_source_units: int = 2
    min_weak_structural_gain: float = 0.0
    max_selected: int | None = None
    utility_size_band_weight: float = 1e-3
    utility_target_cluster_count_weight: float = 1e-2
    utility_escaped_weight_weight: float = 1e-3
    utility_singleton_penalty_weight: float = 1e-3
    utility_restored_penalty_weight: float = 1.0
    cost_edge_scan_weight: float = 1.0
    cost_probe_edge_weight: float = 1.0
    cost_repair_quotient_weight: float = 1.0


def _percentiles(values: np.ndarray, percentiles: list[int]) -> dict[str, float]:
    if values.size == 0:
        return {f"p{p}": 0.0 for p in percentiles}
    return {f"p{p}": float(np.percentile(values, p)) for p in percentiles}


def _percentile(values: np.ndarray, percentile: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, percentile))


def summarize_cluster_graph_stats(
    stats: RustClusterGraphStats,
    *,
    min_weight: float = 0.0,
    max_weight: float = 0.0,
) -> dict[str, Any]:
    """Return compact aggregate diagnostics for a cluster-graph stats object."""

    active = stats.block_count > 0
    active_weight = stats.doc_weight[active]
    active_internal = stats.internal_weight[active]
    active_external = stats.external_weight[active]
    active_conductance = stats.conductance[active]
    active_leafness = stats.leafness[active]
    active_neighbor_ratio = stats.neighbor_weight_ratio[active]
    candidates = stats.candidate_source.shape[0]
    positive_delta = stats.candidate_delta_q > 0
    band_improving = stats.candidate_size_band_gain > 0

    summary: dict[str, Any] = {
        "n_clusters": int(stats.n_clusters),
        "n_active_clusters": int(active.sum()),
        "total_doc_weight": float(active_weight.sum()),
        "total_internal_weight": float(active_internal.sum()),
        "total_external_weight": float(active_external.sum()),
        "doc_weight": _percentiles(active_weight, [50, 90, 95, 99]),
        "conductance": _percentiles(active_conductance, [50, 90, 95, 99]),
        "leafness": _percentiles(active_leafness, [50, 90, 95, 99]),
        "neighbor_weight_ratio": _percentiles(active_neighbor_ratio, [50, 90, 95, 99]),
        "n_merge_candidates": int(candidates),
        "n_positive_delta_candidates": int(positive_delta.sum()),
        "n_band_improving_candidates": int(band_improving.sum()),
        "n_positive_and_band_improving_candidates": int((positive_delta & band_improving).sum()),
    }
    if min_weight > 0:
        summary["n_below_min_weight"] = int((active_weight < min_weight).sum())
    if max_weight > 0:
        summary["n_above_max_weight"] = int((active_weight > max_weight).sum())
    if min_weight > 0 and max_weight > 0:
        summary["n_within_weight_band"] = int(
            ((active_weight >= min_weight) & (active_weight <= max_weight)).sum()
        )
    return summary


def _band_codes(weights: np.ndarray, min_weight: float, max_weight: float) -> np.ndarray:
    codes = np.zeros(weights.shape[0], dtype=np.int8)
    if min_weight > 0:
        codes[weights < min_weight] = -1
    if max_weight > 0:
        codes[weights > max_weight] = 1
    return codes


def _count_band_delta(before_a: np.ndarray, before_b: np.ndarray, after: np.ndarray) -> tuple[int, int, int]:
    before_below = (before_a == -1).astype(np.int64) + (before_b == -1).astype(np.int64)
    before_within = (before_a == 0).astype(np.int64) + (before_b == 0).astype(np.int64)
    before_above = (before_a == 1).astype(np.int64) + (before_b == 1).astype(np.int64)
    after_below = (after == -1).astype(np.int64)
    after_within = (after == 0).astype(np.int64)
    after_above = (after == 1).astype(np.int64)
    below_delta = int((after_below - before_below).sum())
    within_delta = int((after_within - before_within).sum())
    above_delta = int((after_above - before_above).sum())
    return below_delta, within_delta, above_delta


def _boundary_candidate_mask(
    stats: RustClusterGraphStats,
    policy: BoundaryCandidatePolicy,
) -> np.ndarray:
    mask = stats.second_neighbor >= 0
    mask &= stats.degree >= int(policy.min_degree)
    mask &= stats.doc_weight >= float(policy.min_doc_weight)
    if policy.max_doc_weight is not None:
        mask &= stats.doc_weight <= float(policy.max_doc_weight)
    if policy.exclude_singletons:
        mask &= stats.block_count > 1
    else:
        mask &= stats.block_count >= int(policy.min_block_count)
    mask &= stats.block_count >= int(policy.min_block_count)
    mask &= stats.conductance >= float(policy.min_conductance)
    mask &= stats.leafness <= float(policy.max_leafness)
    mask &= stats.neighbor_weight_ratio >= float(policy.min_neighbor_weight_ratio)
    return mask


def score_boundary_candidates(
    stats: RustClusterGraphStats,
    policy: BoundaryCandidatePolicy,
) -> tuple[np.ndarray, np.ndarray]:
    """Return candidate cluster ids and scores for one boundary policy."""

    candidate_ids = np.flatnonzero(_boundary_candidate_mask(stats, policy))
    if candidate_ids.size == 0:
        return candidate_ids.astype(np.int64), np.empty(0, dtype=np.float64)
    score = (
        stats.neighbor_weight_ratio[candidate_ids]
        * stats.conductance[candidate_ids]
        * np.log1p(stats.doc_weight[candidate_ids])
        / np.maximum(stats.leafness[candidate_ids], 1e-9)
    )
    order = np.argsort(-score, kind="mergesort")
    top_n = min(int(policy.top_k), order.size)
    selected = candidate_ids[order[:top_n]].astype(np.int64, copy=False)
    return selected, score[order[:top_n]]


def exploratory_boundary_candidate_policies() -> list[BoundaryCandidatePolicy]:
    """Return a compact policy grid for boundary ambiguity diagnostics."""

    return [
        BoundaryCandidatePolicy(
            name="boundary_nonleaf",
            min_block_count=2,
            min_doc_weight=10.0,
            min_degree=2,
            min_conductance=0.8,
            max_leafness=0.8,
            min_neighbor_weight_ratio=0.25,
        ),
        BoundaryCandidatePolicy(
            name="boundary_high_ambiguity",
            min_block_count=2,
            min_doc_weight=10.0,
            min_degree=2,
            min_conductance=0.7,
            max_leafness=0.9,
            min_neighbor_weight_ratio=0.5,
        ),
        BoundaryCandidatePolicy(
            name="boundary_band_scale",
            min_block_count=2,
            min_doc_weight=50.0,
            max_doc_weight=1500.0,
            min_degree=2,
            min_conductance=0.7,
            max_leafness=0.9,
            min_neighbor_weight_ratio=0.25,
        ),
        BoundaryCandidatePolicy(
            name="boundary_strict",
            min_block_count=3,
            min_doc_weight=50.0,
            max_doc_weight=1500.0,
            min_degree=3,
            min_conductance=0.8,
            max_leafness=0.75,
            min_neighbor_weight_ratio=0.4,
        ),
    ]


def summarize_boundary_candidate_policies(
    stats: RustClusterGraphStats,
    policies: list[BoundaryCandidatePolicy] | None = None,
) -> list[dict[str, Any]]:
    """Summarize candidate counts and top-score distribution by policy."""

    if policies is None:
        policies = exploratory_boundary_candidate_policies()
    rows: list[dict[str, Any]] = []
    for policy in policies:
        mask = _boundary_candidate_mask(stats, policy)
        candidate_ids, scores = score_boundary_candidates(stats, policy)
        rows.append(
            {
                "policy": asdict(policy),
                "n_candidates_after_filters": int(mask.sum()),
                "n_exported": int(candidate_ids.size),
                "score_p50": _percentile(scores, 50),
                "score_p90": _percentile(scores, 90),
                "doc_weight_p50": _percentile(stats.doc_weight[candidate_ids], 50),
                "doc_weight_p90": _percentile(stats.doc_weight[candidate_ids], 90),
                "neighbor_weight_ratio_p50": _percentile(
                    stats.neighbor_weight_ratio[candidate_ids],
                    50,
                ),
                "conductance_p50": _percentile(stats.conductance[candidate_ids], 50),
                "leafness_p50": _percentile(stats.leafness[candidate_ids], 50),
            }
        )
    return rows


def simulate_macro_merge_policy(
    stats: RustClusterGraphStats,
    policy: MacroMergePolicy,
    *,
    min_weight: float = 0.0,
    max_weight: float = 0.0,
) -> MacroMergeSimulationResult:
    """Greedily simulate one non-conflicting macro-merge policy.

    The simulation uses the candidate list from ``cluster_graph_stats`` and
    does not mutate the membership. It is intended for comparing exploratory
    perturbation policies before adding rollback/polish execution.
    """

    n_candidates = stats.n_candidates
    active_clusters = int((stats.block_count > 0).sum())
    if n_candidates == 0:
        return MacroMergeSimulationResult(
            policy=policy,
            n_candidates_considered=0,
            n_candidates_after_filters=0,
            n_selected=0,
            estimated_active_clusters_after=active_clusters,
            sum_delta_q=0.0,
            q_debt=0.0,
            size_band_gain=0.0,
            within_band_delta=0,
            below_min_delta=0,
            above_max_delta=0,
            singleton_endpoint_pairs=0,
            both_singleton_pairs=0,
            selected_doc_weight_p50=0.0,
            selected_doc_weight_p90=0.0,
            selected_leafness_p90=0.0,
            selected_conductance_p50=0.0,
        )

    source = np.asarray(stats.candidate_source, dtype=np.int64)
    target = np.asarray(stats.candidate_target, dtype=np.int64)
    delta_q = np.asarray(stats.candidate_delta_q, dtype=np.float64)
    size_gain = np.asarray(stats.candidate_size_band_gain, dtype=np.float64)
    merged_weight = np.asarray(stats.candidate_merged_weight, dtype=np.float64)
    source_weight = stats.doc_weight[source]
    target_weight = stats.doc_weight[target]
    max_pair_leafness = np.maximum(stats.leafness[source], stats.leafness[target])
    max_pair_conductance = np.maximum(stats.conductance[source], stats.conductance[target])
    singleton_pair = (stats.block_count[source] == 1) | (stats.block_count[target] == 1)

    valid = np.isfinite(delta_q)
    valid &= delta_q >= -float(policy.epsilon)
    valid &= size_gain > float(policy.min_size_band_gain)
    if policy.max_leafness is not None:
        valid &= max_pair_leafness <= float(policy.max_leafness)
    if policy.min_conductance is not None:
        valid &= max_pair_conductance >= float(policy.min_conductance)
    if policy.require_any_below_min and min_weight > 0:
        valid &= (source_weight < min_weight) | (target_weight < min_weight)
    if policy.require_merged_within_band:
        valid &= merged_weight >= min_weight
        if max_weight > 0:
            valid &= merged_weight <= max_weight
    if policy.max_merged_weight is not None:
        valid &= merged_weight <= float(policy.max_merged_weight)
    elif max_weight > 0:
        valid &= merged_weight <= max_weight
    if not policy.allow_singleton_endpoint:
        valid &= ~singleton_pair

    valid_indices = np.flatnonzero(valid)
    n_after_filters = int(valid_indices.size)
    if n_after_filters == 0:
        return MacroMergeSimulationResult(
            policy=policy,
            n_candidates_considered=n_candidates,
            n_candidates_after_filters=0,
            n_selected=0,
            estimated_active_clusters_after=active_clusters,
            sum_delta_q=0.0,
            q_debt=0.0,
            size_band_gain=0.0,
            within_band_delta=0,
            below_min_delta=0,
            above_max_delta=0,
            singleton_endpoint_pairs=0,
            both_singleton_pairs=0,
            selected_doc_weight_p50=0.0,
            selected_doc_weight_p90=0.0,
            selected_leafness_p90=0.0,
            selected_conductance_p50=0.0,
        )

    score = delta_q[valid_indices].copy()
    normalizer = max(float(min_weight), 1.0)
    if policy.score_size_band_weight:
        score += float(policy.score_size_band_weight) * (size_gain[valid_indices] / normalizer)
    if policy.score_conductance_weight:
        score += float(policy.score_conductance_weight) * max_pair_conductance[valid_indices]
    if policy.score_leafness_penalty:
        score -= float(policy.score_leafness_penalty) * max_pair_leafness[valid_indices]

    order = np.lexsort((-delta_q[valid_indices], -score))
    ordered = valid_indices[order]
    used = np.zeros(stats.n_clusters, dtype=bool)
    selected: list[int] = []
    q_debt = 0.0
    for idx in ordered:
        s = int(source[idx])
        t = int(target[idx])
        if used[s] or used[t]:
            continue
        next_debt = q_debt + max(0.0, -float(delta_q[idx]))
        if policy.q_debt_budget is not None and next_debt > float(policy.q_debt_budget):
            continue
        selected.append(int(idx))
        used[s] = True
        used[t] = True
        q_debt = next_debt
        if policy.max_merges is not None and len(selected) >= int(policy.max_merges):
            break

    if not selected:
        return MacroMergeSimulationResult(
            policy=policy,
            n_candidates_considered=n_candidates,
            n_candidates_after_filters=n_after_filters,
            n_selected=0,
            estimated_active_clusters_after=active_clusters,
            sum_delta_q=0.0,
            q_debt=0.0,
            size_band_gain=0.0,
            within_band_delta=0,
            below_min_delta=0,
            above_max_delta=0,
            singleton_endpoint_pairs=0,
            both_singleton_pairs=0,
            selected_doc_weight_p50=0.0,
            selected_doc_weight_p90=0.0,
            selected_leafness_p90=0.0,
            selected_conductance_p50=0.0,
        )

    selected_idx = np.asarray(selected, dtype=np.int64)
    selected_source = source[selected_idx]
    selected_target = target[selected_idx]
    source_codes = _band_codes(stats.doc_weight[selected_source], min_weight, max_weight)
    target_codes = _band_codes(stats.doc_weight[selected_target], min_weight, max_weight)
    merged_codes = _band_codes(merged_weight[selected_idx], min_weight, max_weight)
    below_delta, within_delta, above_delta = _count_band_delta(
        source_codes,
        target_codes,
        merged_codes,
    )
    selected_weights = merged_weight[selected_idx]
    selected_leafness = np.maximum(
        stats.leafness[selected_source],
        stats.leafness[selected_target],
    )
    selected_conductance = np.maximum(
        stats.conductance[selected_source],
        stats.conductance[selected_target],
    )
    selected_singleton = (stats.block_count[selected_source] == 1) | (
        stats.block_count[selected_target] == 1
    )
    selected_both_singleton = (stats.block_count[selected_source] == 1) & (
        stats.block_count[selected_target] == 1
    )

    return MacroMergeSimulationResult(
        policy=policy,
        n_candidates_considered=n_candidates,
        n_candidates_after_filters=n_after_filters,
        n_selected=int(selected_idx.size),
        estimated_active_clusters_after=active_clusters - int(selected_idx.size),
        sum_delta_q=float(delta_q[selected_idx].sum()),
        q_debt=float(np.maximum(0.0, -delta_q[selected_idx]).sum()),
        size_band_gain=float(size_gain[selected_idx].sum()),
        within_band_delta=within_delta,
        below_min_delta=below_delta,
        above_max_delta=above_delta,
        singleton_endpoint_pairs=int(selected_singleton.sum()),
        both_singleton_pairs=int(selected_both_singleton.sum()),
        selected_doc_weight_p50=_percentile(selected_weights, 50),
        selected_doc_weight_p90=_percentile(selected_weights, 90),
        selected_leafness_p90=_percentile(selected_leafness, 90),
        selected_conductance_p50=_percentile(selected_conductance, 50),
    )


def exploratory_macro_merge_policies() -> list[MacroMergePolicy]:
    """Return a compact policy grid for first-pass adaptive refinement probes."""

    return [
        MacroMergePolicy(name="eps1e-4_band", epsilon=1e-4),
        MacroMergePolicy(name="eps1e-4_no_singletons", epsilon=1e-4, allow_singleton_endpoint=False),
        MacroMergePolicy(name="eps1e-4_leaf<=0.8", epsilon=1e-4, max_leafness=0.8),
        MacroMergePolicy(name="eps1e-4_boundary", epsilon=1e-4, min_conductance=0.8),
        MacroMergePolicy(
            name="eps1e-4_merged_within",
            epsilon=1e-4,
            require_merged_within_band=True,
        ),
        MacroMergePolicy(
            name="eps3e-4_leaf<=0.8",
            epsilon=3e-4,
            max_leafness=0.8,
            q_debt_budget=0.05,
        ),
        MacroMergePolicy(
            name="eps3e-4_boundary_leaf<=0.9",
            epsilon=3e-4,
            min_conductance=0.8,
            max_leafness=0.9,
            q_debt_budget=0.05,
        ),
        MacroMergePolicy(
            name="eps3e-4_merged_within",
            epsilon=3e-4,
            require_merged_within_band=True,
            q_debt_budget=0.05,
        ),
        MacroMergePolicy(
            name="eps1e-3_leaf<=0.8",
            epsilon=1e-3,
            max_leafness=0.8,
            q_debt_budget=0.5,
        ),
        MacroMergePolicy(
            name="eps1e-3_boundary_leaf<=0.9",
            epsilon=1e-3,
            min_conductance=0.8,
            max_leafness=0.9,
            q_debt_budget=0.5,
        ),
        MacroMergePolicy(
            name="eps1e-3_merged_within",
            epsilon=1e-3,
            require_merged_within_band=True,
            q_debt_budget=0.5,
        ),
        MacroMergePolicy(
            name="utility_eps1e-3",
            epsilon=1e-3,
            max_leafness=0.9,
            q_debt_budget=0.5,
            score_size_band_weight=1e-3,
            score_conductance_weight=1e-4,
            score_leafness_penalty=1e-4,
        ),
    ]


def run_macro_merge_policy_ensemble(
    stats: RustClusterGraphStats,
    policies: list[MacroMergePolicy] | None = None,
    *,
    min_weight: float = 0.0,
    max_weight: float = 0.0,
) -> list[MacroMergeSimulationResult]:
    """Compare multiple exploratory macro-merge policies on the same candidates."""

    if policies is None:
        policies = exploratory_macro_merge_policies()
    return [
        simulate_macro_merge_policy(
            stats,
            policy,
            min_weight=min_weight,
            max_weight=max_weight,
        )
        for policy in policies
    ]


def write_macro_merge_ensemble_report(
    stats: RustClusterGraphStats,
    output_dir: Path,
    policies: list[MacroMergePolicy] | None = None,
    *,
    min_weight: float = 0.0,
    max_weight: float = 0.0,
) -> dict[str, str]:
    """Write JSON/CSV comparison for exploratory macro-merge policy ensemble."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = run_macro_merge_policy_ensemble(
        stats,
        policies,
        min_weight=min_weight,
        max_weight=max_weight,
    )
    summary = [result.to_dict() for result in results]

    json_path = output_dir / "macro_merge_policy_ensemble.json"
    json_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    csv_path = output_dir / "macro_merge_policy_ensemble.csv"
    fieldnames = [
        "policy",
        "epsilon",
        "max_leafness",
        "min_conductance",
        "allow_singleton_endpoint",
        "q_debt_budget",
        "n_candidates_after_filters",
        "n_selected",
        "estimated_active_clusters_after",
        "sum_delta_q",
        "q_debt",
        "size_band_gain",
        "within_band_delta",
        "below_min_delta",
        "above_max_delta",
        "singleton_endpoint_pairs",
        "both_singleton_pairs",
        "selected_doc_weight_p50",
        "selected_doc_weight_p90",
        "selected_leafness_p90",
        "selected_conductance_p50",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            policy = result.policy
            writer.writerow(
                {
                    "policy": policy.name,
                    "epsilon": policy.epsilon,
                    "max_leafness": "" if policy.max_leafness is None else policy.max_leafness,
                    "min_conductance": ""
                    if policy.min_conductance is None
                    else policy.min_conductance,
                    "allow_singleton_endpoint": policy.allow_singleton_endpoint,
                    "q_debt_budget": ""
                    if policy.q_debt_budget is None
                    else policy.q_debt_budget,
                    "n_candidates_after_filters": result.n_candidates_after_filters,
                    "n_selected": result.n_selected,
                    "estimated_active_clusters_after": result.estimated_active_clusters_after,
                    "sum_delta_q": result.sum_delta_q,
                    "q_debt": result.q_debt,
                    "size_band_gain": result.size_band_gain,
                    "within_band_delta": result.within_band_delta,
                    "below_min_delta": result.below_min_delta,
                    "above_max_delta": result.above_max_delta,
                    "singleton_endpoint_pairs": result.singleton_endpoint_pairs,
                    "both_singleton_pairs": result.both_singleton_pairs,
                    "selected_doc_weight_p50": result.selected_doc_weight_p50,
                    "selected_doc_weight_p90": result.selected_doc_weight_p90,
                    "selected_leafness_p90": result.selected_leafness_p90,
                    "selected_conductance_p50": result.selected_conductance_p50,
                }
            )
    return {"json": str(json_path), "csv": str(csv_path)}


def write_boundary_candidate_report(
    stats: RustClusterGraphStats,
    output_dir: Path,
    policies: list[BoundaryCandidatePolicy] | None = None,
) -> dict[str, str]:
    """Write boundary ambiguity summaries and top candidate table."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if policies is None:
        policies = exploratory_boundary_candidate_policies()

    summary = summarize_boundary_candidate_policies(stats, policies)
    summary_path = output_dir / "boundary_candidate_policy_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    candidate_path = output_dir / "boundary_candidates.csv"
    fieldnames = [
        "policy",
        "rank",
        "cluster",
        "score",
        "doc_weight",
        "block_count",
        "internal_weight",
        "external_weight",
        "degree",
        "conductance",
        "leafness",
        "neighbor_weight_ratio",
        "top_neighbor",
        "top_neighbor_weight",
        "second_neighbor",
        "second_neighbor_weight",
        "band_distance",
    ]
    with candidate_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for policy in policies:
            candidate_ids, scores = score_boundary_candidates(stats, policy)
            for rank, (cluster, score) in enumerate(zip(candidate_ids, scores), start=1):
                c = int(cluster)
                writer.writerow(
                    {
                        "policy": policy.name,
                        "rank": rank,
                        "cluster": c,
                        "score": float(score),
                        "doc_weight": float(stats.doc_weight[c]),
                        "block_count": int(stats.block_count[c]),
                        "internal_weight": float(stats.internal_weight[c]),
                        "external_weight": float(stats.external_weight[c]),
                        "degree": int(stats.degree[c]),
                        "conductance": float(stats.conductance[c]),
                        "leafness": float(stats.leafness[c]),
                        "neighbor_weight_ratio": float(stats.neighbor_weight_ratio[c]),
                        "top_neighbor": int(stats.top_neighbor[c]),
                        "top_neighbor_weight": float(stats.top_neighbor_weight[c]),
                        "second_neighbor": int(stats.second_neighbor[c]),
                        "second_neighbor_weight": float(stats.second_neighbor_weight[c]),
                        "band_distance": float(stats.band_distance[c]),
                    }
                )

    return {"summary": str(summary_path), "candidates": str(candidate_path)}


def summarize_boundary_move_probes(probes: RustBoundaryMoveProbes) -> dict[str, Any]:
    """Return compact aggregate diagnostics for boundary move dry-runs."""

    positive = probes.positive_move_count > 0
    near_neutral = probes.near_neutral_move_count > 0
    return {
        "n_probes": int(probes.n_probes),
        "n_with_positive_moves": int(positive.sum()),
        "n_with_near_neutral_moves": int(near_neutral.sum()),
        "total_positive_move_count": int(probes.positive_move_count.sum()),
        "total_positive_move_weight": float(probes.positive_move_weight.sum()),
        "total_positive_delta_q": float(probes.positive_delta_q.sum()),
        "total_near_neutral_move_count": int(probes.near_neutral_move_count.sum()),
        "total_near_neutral_move_weight": float(probes.near_neutral_move_weight.sum()),
        "total_near_neutral_delta_q": float(probes.near_neutral_delta_q.sum()),
        "best_move_delta_q": _percentiles(probes.best_move_delta_q, [50, 90, 95, 99]),
        "positive_move_count": _percentiles(
            probes.positive_move_count.astype(np.float64),
            [50, 90, 95, 99],
        ),
        "positive_move_weight": _percentiles(probes.positive_move_weight, [50, 90, 95, 99]),
    }


def write_boundary_move_probe_report(
    probes: RustBoundaryMoveProbes,
    output_dir: Path,
    *,
    top_probes: int | None = None,
) -> dict[str, str]:
    """Write JSON/CSV diagnostics for boundary move dry-runs."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_boundary_move_probes(probes)
    summary_path = output_dir / "boundary_move_probe_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    order = np.argsort(-probes.best_move_delta_q, kind="mergesort")
    if top_probes is not None:
        order = order[: int(top_probes)]

    csv_path = output_dir / "boundary_move_probes.csv"
    fieldnames = [
        "rank",
        "cluster",
        "block_count",
        "doc_weight",
        "internal_weight",
        "external_weight",
        "conductance",
        "leafness",
        "top_neighbor",
        "top_neighbor_weight",
        "second_neighbor",
        "second_neighbor_weight",
        "neighbor_weight_ratio",
        "positive_move_count",
        "positive_move_weight",
        "positive_delta_q",
        "near_neutral_move_count",
        "near_neutral_move_weight",
        "near_neutral_delta_q",
        "best_move_delta_q",
        "best_move_node",
        "best_move_target",
        "top_move_count",
        "second_move_count",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for rank, idx in enumerate(order, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "cluster": int(probes.cluster[idx]),
                    "block_count": int(probes.block_count[idx]),
                    "doc_weight": float(probes.doc_weight[idx]),
                    "internal_weight": float(probes.internal_weight[idx]),
                    "external_weight": float(probes.external_weight[idx]),
                    "conductance": float(probes.conductance[idx]),
                    "leafness": float(probes.leafness[idx]),
                    "top_neighbor": int(probes.top_neighbor[idx]),
                    "top_neighbor_weight": float(probes.top_neighbor_weight[idx]),
                    "second_neighbor": int(probes.second_neighbor[idx]),
                    "second_neighbor_weight": float(probes.second_neighbor_weight[idx]),
                    "neighbor_weight_ratio": float(probes.neighbor_weight_ratio[idx]),
                    "positive_move_count": int(probes.positive_move_count[idx]),
                    "positive_move_weight": float(probes.positive_move_weight[idx]),
                    "positive_delta_q": float(probes.positive_delta_q[idx]),
                    "near_neutral_move_count": int(probes.near_neutral_move_count[idx]),
                    "near_neutral_move_weight": float(probes.near_neutral_move_weight[idx]),
                    "near_neutral_delta_q": float(probes.near_neutral_delta_q[idx]),
                    "best_move_delta_q": float(probes.best_move_delta_q[idx]),
                    "best_move_node": int(probes.best_move_node[idx]),
                    "best_move_target": int(probes.best_move_target[idx]),
                    "top_move_count": int(probes.top_move_count[idx]),
                    "second_move_count": int(probes.second_move_count[idx]),
                }
            )

    return {"summary": str(summary_path), "probes": str(csv_path)}


def summarize_boundary_group_probes(probes: RustBoundaryGroupProbes) -> dict[str, Any]:
    """Return compact aggregate diagnostics for grouped boundary probes."""

    positive = probes.best_delta_q > 0
    top_move = probes.best_action == 1
    second_move = probes.best_action == 2
    top_split = probes.best_action == 3
    second_split = probes.best_action == 4
    return {
        "n_probes": int(probes.n_probes),
        "n_positive_best": int(positive.sum()),
        "n_positive_top_group_move": int((probes.top_group_move_delta_q > 0).sum()),
        "n_positive_second_group_move": int((probes.second_group_move_delta_q > 0).sum()),
        "n_positive_top_group_split": int((probes.top_group_split_delta_q > 0).sum()),
        "n_positive_second_group_split": int((probes.second_group_split_delta_q > 0).sum()),
        "best_action_counts": {
            "none": int((probes.best_action == 0).sum()),
            "top_move": int(top_move.sum()),
            "second_move": int(second_move.sum()),
            "top_split": int(top_split.sum()),
            "second_split": int(second_split.sum()),
        },
        "best_delta_q": _percentiles(probes.best_delta_q, [50, 90, 95, 99]),
        "best_delta_q_max": float(probes.best_delta_q.max()) if probes.n_probes else 0.0,
        "top_group_weight": _percentiles(probes.top_group_weight, [50, 90, 95, 99]),
        "second_group_weight": _percentiles(probes.second_group_weight, [50, 90, 95, 99]),
    }


def write_boundary_group_probe_report(
    probes: RustBoundaryGroupProbes,
    output_dir: Path,
    *,
    top_probes: int | None = None,
) -> dict[str, str]:
    """Write JSON/CSV diagnostics for grouped boundary split/move probes."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_boundary_group_probes(probes)
    summary_path = output_dir / "boundary_group_probe_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    order = np.argsort(-probes.best_delta_q, kind="mergesort")
    if top_probes is not None:
        order = order[: int(top_probes)]

    csv_path = output_dir / "boundary_group_probes.csv"
    fieldnames = [
        "rank",
        "cluster",
        "block_count",
        "doc_weight",
        "top_neighbor",
        "second_neighbor",
        "top_group_count",
        "top_group_weight",
        "top_group_to_target_weight",
        "top_group_cut_weight",
        "top_group_move_delta_q",
        "top_group_split_delta_q",
        "top_group_is_full_cluster",
        "second_group_count",
        "second_group_weight",
        "second_group_to_target_weight",
        "second_group_cut_weight",
        "second_group_move_delta_q",
        "second_group_split_delta_q",
        "second_group_is_full_cluster",
        "best_delta_q",
        "best_action",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for rank, idx in enumerate(order, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "cluster": int(probes.cluster[idx]),
                    "block_count": int(probes.block_count[idx]),
                    "doc_weight": float(probes.doc_weight[idx]),
                    "top_neighbor": int(probes.top_neighbor[idx]),
                    "second_neighbor": int(probes.second_neighbor[idx]),
                    "top_group_count": int(probes.top_group_count[idx]),
                    "top_group_weight": float(probes.top_group_weight[idx]),
                    "top_group_to_target_weight": float(probes.top_group_to_target_weight[idx]),
                    "top_group_cut_weight": float(probes.top_group_cut_weight[idx]),
                    "top_group_move_delta_q": float(probes.top_group_move_delta_q[idx]),
                    "top_group_split_delta_q": float(probes.top_group_split_delta_q[idx]),
                    "top_group_is_full_cluster": bool(probes.top_group_is_full_cluster[idx]),
                    "second_group_count": int(probes.second_group_count[idx]),
                    "second_group_weight": float(probes.second_group_weight[idx]),
                    "second_group_to_target_weight": float(
                        probes.second_group_to_target_weight[idx]
                    ),
                    "second_group_cut_weight": float(probes.second_group_cut_weight[idx]),
                    "second_group_move_delta_q": float(probes.second_group_move_delta_q[idx]),
                    "second_group_split_delta_q": float(probes.second_group_split_delta_q[idx]),
                    "second_group_is_full_cluster": bool(
                        probes.second_group_is_full_cluster[idx]
                    ),
                    "best_delta_q": float(probes.best_delta_q[idx]),
                    "best_action": int(probes.best_action[idx]),
                }
            )

    return {"summary": str(summary_path), "probes": str(csv_path)}


def summarize_multi_core_split_probes(probes: RustMultiCoreSplitProbes) -> dict[str, Any]:
    """Return compact aggregate diagnostics for high-gamma split probes."""

    split = probes.n_parts > 1
    base_positive = probes.split_delta_q_base > 0
    probe_positive = probes.split_delta_q_probe > 0
    meaningful_core = probes.core_part_count >= 2
    return {
        "n_probes": int(probes.n_probes),
        "n_split": int(split.sum()),
        "n_base_positive": int(base_positive.sum()),
        "n_probe_positive": int(probe_positive.sum()),
        "n_hysteresis_only": int(probes.hysteresis_only.sum()),
        "n_meaningful_core_split": int((split & meaningful_core).sum()),
        "n_base_positive_meaningful_core_split": int(
            (base_positive & meaningful_core).sum()
        ),
        "n_parts": _percentiles(probes.n_parts.astype(np.float64), [50, 90, 95, 99]),
        "core_part_count": _percentiles(
            probes.core_part_count.astype(np.float64), [50, 90, 95, 99]
        ),
        "split_delta_q_base": _percentiles(probes.split_delta_q_base, [50, 90, 95, 99]),
        "split_delta_q_base_max": (
            float(probes.split_delta_q_base.max()) if probes.n_probes else 0.0
        ),
        "split_delta_q_probe": _percentiles(probes.split_delta_q_probe, [50, 90, 95, 99]),
        "split_delta_q_probe_max": (
            float(probes.split_delta_q_probe.max()) if probes.n_probes else 0.0
        ),
        "largest_part_fraction": _percentiles(
            probes.largest_part_fraction, [50, 90, 95, 99]
        ),
        "singleton_weight": _percentiles(probes.singleton_weight, [50, 90, 95, 99]),
    }


def write_multi_core_split_probe_report(
    probes: RustMultiCoreSplitProbes,
    output_dir: Path,
    *,
    top_probes: int | None = None,
) -> dict[str, str]:
    """Write JSON/CSV diagnostics for high-gamma induced split probes."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_multi_core_split_probes(probes)
    summary_path = output_dir / "multi_core_split_probe_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    order = np.lexsort(
        (
            probes.cluster,
            -probes.split_delta_q_probe,
            -probes.split_delta_q_base,
        )
    )
    if top_probes is not None:
        order = order[: int(top_probes)]

    csv_path = output_dir / "multi_core_split_probes.csv"
    fieldnames = [
        "rank",
        "cluster",
        "gamma_multiplier",
        "probe_resolution",
        "block_count",
        "doc_weight",
        "internal_weight",
        "induced_directed_edges",
        "n_parts",
        "non_singleton_parts",
        "singleton_parts",
        "singleton_weight",
        "core_part_count",
        "core_part_weight",
        "largest_part_weight",
        "second_part_weight",
        "largest_part_fraction",
        "cut_weight",
        "split_delta_q_base",
        "split_delta_q_probe",
        "hysteresis_only",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for rank, idx in enumerate(order, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "cluster": int(probes.cluster[idx]),
                    "gamma_multiplier": float(probes.gamma_multiplier[idx]),
                    "probe_resolution": float(probes.probe_resolution[idx]),
                    "block_count": int(probes.block_count[idx]),
                    "doc_weight": float(probes.doc_weight[idx]),
                    "internal_weight": float(probes.internal_weight[idx]),
                    "induced_directed_edges": int(probes.induced_directed_edges[idx]),
                    "n_parts": int(probes.n_parts[idx]),
                    "non_singleton_parts": int(probes.non_singleton_parts[idx]),
                    "singleton_parts": int(probes.singleton_parts[idx]),
                    "singleton_weight": float(probes.singleton_weight[idx]),
                    "core_part_count": int(probes.core_part_count[idx]),
                    "core_part_weight": float(probes.core_part_weight[idx]),
                    "largest_part_weight": float(probes.largest_part_weight[idx]),
                    "second_part_weight": float(probes.second_part_weight[idx]),
                    "largest_part_fraction": float(probes.largest_part_fraction[idx]),
                    "cut_weight": float(probes.cut_weight[idx]),
                    "split_delta_q_base": float(probes.split_delta_q_base[idx]),
                    "split_delta_q_probe": float(probes.split_delta_q_probe[idx]),
                    "hysteresis_only": bool(probes.hysteresis_only[idx]),
                }
            )

    return {"summary": str(summary_path), "probes": str(csv_path)}


def _probe_array(probes: RustSplitMergeRepairProbes | dict[str, np.ndarray], name: str) -> np.ndarray:
    if isinstance(probes, dict):
        return np.asarray(probes[name])
    return np.asarray(getattr(probes, name))


def _split_repair_probe_count_by_cluster(cluster: np.ndarray) -> dict[int, int]:
    unique, counts = np.unique(cluster.astype(np.int64, copy=False), return_counts=True)
    return {int(cluster_id): int(count) for cluster_id, count in zip(unique, counts)}


def rank_split_repair_candidates(
    probes: RustSplitMergeRepairProbes | dict[str, np.ndarray],
    policy: SplitRepairSelectionPolicy | None = None,
    *,
    min_weight: float = 0.0,
    max_weight: float = 0.0,
) -> list[dict[str, Any]]:
    """Rank split-repair probe rows and greedily select one row per source cluster.

    This is still a dry-run layer. It records the utility/cost terms and marks
    non-conflicting candidates that satisfy the conservative acceptance policy.
    """

    if policy is None:
        policy = SplitRepairSelectionPolicy()
    if policy.mode not in {"utility_cost", "oversize_first"}:
        raise ValueError(f"unknown split-repair selection mode: {policy.mode!r}")

    cluster = _probe_array(probes, "cluster").astype(np.int64, copy=False)
    n = int(cluster.shape[0])
    if n == 0:
        return []

    gamma_multiplier = _probe_array(probes, "gamma_multiplier").astype(np.float64, copy=False)
    probe_resolution = _probe_array(probes, "probe_resolution").astype(np.float64, copy=False)
    block_count = _probe_array(probes, "block_count").astype(np.uint64, copy=False)
    doc_weight = _probe_array(probes, "doc_weight").astype(np.float64, copy=False)
    induced_directed_edges = _probe_array(probes, "induced_directed_edges").astype(
        np.float64,
        copy=False,
    )
    n_parts = _probe_array(probes, "n_parts").astype(np.uint64, copy=False)
    core_part_count = _probe_array(probes, "core_part_count").astype(np.uint64, copy=False)
    singleton_weight = _probe_array(probes, "singleton_weight").astype(np.float64, copy=False)
    cut_weight = _probe_array(probes, "cut_weight").astype(np.float64, copy=False)
    split_delta_q_base = _probe_array(probes, "split_delta_q_base").astype(
        np.float64,
        copy=False,
    )
    split_delta_q_probe = _probe_array(probes, "split_delta_q_probe").astype(
        np.float64,
        copy=False,
    )
    repair_quotient_edges = _probe_array(probes, "repair_quotient_edges").astype(
        np.float64,
        copy=False,
    )
    repair_merge_count = _probe_array(probes, "repair_merge_count").astype(np.uint64, copy=False)
    repair_delta_q = _probe_array(probes, "repair_delta_q").astype(np.float64, copy=False)
    net_delta_q = _probe_array(probes, "net_delta_q").astype(np.float64, copy=False)
    final_source_units = _probe_array(probes, "final_source_units").astype(np.uint64, copy=False)
    retained_source_units = _probe_array(probes, "retained_source_units").astype(
        np.uint64,
        copy=False,
    )
    escaped_source_units = _probe_array(probes, "escaped_source_units").astype(
        np.uint64,
        copy=False,
    )
    escaped_source_weight = _probe_array(probes, "escaped_source_weight").astype(
        np.float64,
        copy=False,
    )
    final_small_source_units = _probe_array(probes, "final_small_source_units").astype(
        np.uint64,
        copy=False,
    )
    final_small_source_weight = _probe_array(probes, "final_small_source_weight").astype(
        np.float64,
        copy=False,
    )
    largest_source_unit_fraction = _probe_array(
        probes,
        "largest_source_unit_fraction",
    ).astype(np.float64, copy=False)
    restored_source_cluster = _probe_array(probes, "restored_source_cluster").astype(
        bool,
        copy=False,
    )

    probe_count_by_cluster = _split_repair_probe_count_by_cluster(cluster)
    largest_source_unit_weight = largest_source_unit_fraction * doc_weight
    if max_weight > 0:
        remaining_oversize_before = np.maximum(0.0, doc_weight - max_weight)
        remaining_oversize_after = np.maximum(0.0, largest_source_unit_weight - max_weight)
        size_band_improvement = np.maximum(
            0.0,
            remaining_oversize_before - remaining_oversize_after,
        )
    else:
        remaining_oversize_before = np.zeros(n, dtype=np.float64)
        remaining_oversize_after = np.zeros(n, dtype=np.float64)
        size_band_improvement = np.zeros(n, dtype=np.float64)
    oversize_reduction = size_band_improvement
    passes_size_objective_array = (
        (max_weight > 0)
        & (remaining_oversize_before > 0.0)
        & (oversize_reduction > 0.0)
        & (largest_source_unit_weight < doc_weight)
    )

    target_cluster_count_improvement = np.maximum(
        final_source_units.astype(np.float64) - 1.0,
        0.0,
    )
    escaped_if_net_positive = np.where(net_delta_q > 0.0, escaped_source_weight, 0.0)
    singleton_penalty = np.maximum(singleton_weight, final_small_source_weight)
    restored_penalty = np.where(restored_source_cluster, doc_weight, 0.0)
    structural_gain = escaped_source_weight + np.maximum(
        retained_source_units.astype(np.float64) - 1.0,
        0.0,
    )

    utility = (
        net_delta_q
        + policy.utility_size_band_weight * size_band_improvement
        + policy.utility_target_cluster_count_weight * target_cluster_count_improvement
        + policy.utility_escaped_weight_weight * escaped_if_net_positive
        - policy.utility_singleton_penalty_weight * singleton_penalty
        - policy.utility_restored_penalty_weight * restored_penalty
    )
    probe_count = np.asarray(
        [probe_count_by_cluster[int(cluster_id)] for cluster_id in cluster],
        dtype=np.float64,
    )
    cost = (
        policy.cost_edge_scan_weight * induced_directed_edges
        + policy.cost_probe_edge_weight * probe_count * induced_directed_edges
        + policy.cost_repair_quotient_weight * repair_quotient_edges
    )
    cost = np.maximum(cost, 1.0)
    priority = utility / cost

    rows: list[dict[str, Any]] = []
    for idx in range(n):
        passes_net = bool(net_delta_q[idx] > policy.min_net_delta_q)
        passes_restored = bool(not restored_source_cluster[idx])
        passes_structure = bool(
            escaped_source_weight[idx] > 0.0
            or retained_source_units[idx] >= policy.min_retained_source_units
        )
        passes_singleton_budget = bool(final_small_source_weight[idx] <= policy.singleton_budget)
        strong_candidate = bool(net_delta_q[idx] > policy.strong_net_delta_q)
        passes_weak_structure = bool(
            strong_candidate or structural_gain[idx] >= policy.min_weak_structural_gain
        )
        passes_size_objective = bool(passes_size_objective_array[idx])

        rejection_reason = ""
        if policy.mode == "oversize_first" and max_weight <= 0:
            rejection_reason = "target_max_weight_required"
        elif policy.mode == "oversize_first" and remaining_oversize_before[idx] <= 0.0:
            rejection_reason = "source_not_above_target_max"
        elif policy.mode == "oversize_first" and not passes_size_objective:
            rejection_reason = "no_oversize_reduction"
        elif policy.mode == "oversize_first" and retained_source_units[idx] == 0:
            rejection_reason = "no_retained_source_unit"
        elif not passes_net:
            rejection_reason = "net_delta_q_below_threshold"
        elif not passes_restored:
            rejection_reason = "restored_source_cluster"
        elif not passes_structure:
            rejection_reason = "no_escape_or_retained_split"
        elif not passes_singleton_budget:
            rejection_reason = "small_source_budget_exceeded"
        elif not passes_weak_structure:
            rejection_reason = "weak_structural_gain_below_threshold"

        rows.append(
            {
                "rank": 0,
                "selected_for_apply": False,
                "accepted_by_policy": rejection_reason == "",
                "rejection_reason": rejection_reason,
                "conflict_reason": "",
                "policy": policy.name,
                "selection_mode": policy.mode,
                "cluster": int(cluster[idx]),
                "gamma_multiplier": float(gamma_multiplier[idx]),
                "probe_resolution": float(probe_resolution[idx]),
                "block_count": int(block_count[idx]),
                "doc_weight": float(doc_weight[idx]),
                "induced_directed_edges": int(induced_directed_edges[idx]),
                "repair_quotient_edges": int(repair_quotient_edges[idx]),
                "probe_count_for_cluster": int(probe_count[idx]),
                "n_parts": int(n_parts[idx]),
                "core_part_count": int(core_part_count[idx]),
                "singleton_weight": float(singleton_weight[idx]),
                "cut_weight": float(cut_weight[idx]),
                "split_delta_q_base": float(split_delta_q_base[idx]),
                "split_delta_q_probe": float(split_delta_q_probe[idx]),
                "repair_merge_count": int(repair_merge_count[idx]),
                "repair_delta_q": float(repair_delta_q[idx]),
                "net_delta_q": float(net_delta_q[idx]),
                "final_source_units": int(final_source_units[idx]),
                "retained_source_units": int(retained_source_units[idx]),
                "escaped_source_units": int(escaped_source_units[idx]),
                "escaped_source_weight": float(escaped_source_weight[idx]),
                "final_small_source_units": int(final_small_source_units[idx]),
                "final_small_source_weight": float(final_small_source_weight[idx]),
                "largest_source_unit_fraction": float(largest_source_unit_fraction[idx]),
                "largest_source_unit_weight": float(largest_source_unit_weight[idx]),
                "restored_source_cluster": bool(restored_source_cluster[idx]),
                "strong_candidate": strong_candidate,
                "structural_gain": float(structural_gain[idx]),
                "size_band_improvement": float(size_band_improvement[idx]),
                "remaining_oversize_before": float(remaining_oversize_before[idx]),
                "remaining_oversize_after": float(remaining_oversize_after[idx]),
                "oversize_reduction": float(oversize_reduction[idx]),
                "passes_size_objective": passes_size_objective,
                "target_cluster_count_improvement": float(
                    target_cluster_count_improvement[idx]
                ),
                "escaped_source_weight_if_net_positive": float(escaped_if_net_positive[idx]),
                "singleton_or_leaf_penalty": float(singleton_penalty[idx]),
                "restored_source_cluster_penalty": float(restored_penalty[idx]),
                "utility": float(utility[idx]),
                "cost": float(cost[idx]),
                "priority": float(priority[idx]),
            }
        )

    if policy.mode == "oversize_first":
        rows.sort(
            key=lambda row: (
                not row["accepted_by_policy"],
                row["remaining_oversize_after"],
                -row["oversize_reduction"],
                row["final_small_source_weight"],
                -row["net_delta_q"],
                row["cost"],
                row["cluster"],
                row["gamma_multiplier"],
            )
        )
    else:
        rows.sort(
            key=lambda row: (
                not row["accepted_by_policy"],
                -row["priority"],
                -row["net_delta_q"],
                row["cluster"],
                row["gamma_multiplier"],
            )
        )

    used_clusters: set[int] = set()
    selected_count = 0
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
        if not row["accepted_by_policy"]:
            continue
        cluster_id = int(row["cluster"])
        if cluster_id in used_clusters:
            row["conflict_reason"] = "source_cluster_already_selected"
            continue
        if policy.max_selected is not None and selected_count >= int(policy.max_selected):
            row["conflict_reason"] = "selection_budget_exhausted"
            continue
        row["selected_for_apply"] = True
        used_clusters.add(cluster_id)
        selected_count += 1

    return rows


def summarize_split_repair_candidate_selection(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate a split-repair candidate selection table."""

    selected = [row for row in rows if row["selected_for_apply"]]
    accepted = [row for row in rows if row["accepted_by_policy"]]
    conflicted = [row for row in rows if row["conflict_reason"]]
    rejected = [row for row in rows if not row["accepted_by_policy"]]
    return {
        "n_candidates": len(rows),
        "n_accepted_by_policy": len(accepted),
        "n_selected_for_apply": len(selected),
        "n_conflicted": len(conflicted),
        "n_rejected": len(rejected),
        "selected_net_delta_q_sum": float(sum(row["net_delta_q"] for row in selected)),
        "selected_utility_sum": float(sum(row["utility"] for row in selected)),
        "selected_cost_sum": float(sum(row["cost"] for row in selected)),
        "selected_escaped_source_weight_sum": float(
            sum(row["escaped_source_weight"] for row in selected)
        ),
        "selected_final_small_source_weight_sum": float(
            sum(row["final_small_source_weight"] for row in selected)
        ),
        "selected_oversize_reduction_sum": float(
            sum(row["oversize_reduction"] for row in selected)
        ),
        "selected_remaining_oversize_after_sum": float(
            sum(row["remaining_oversize_after"] for row in selected)
        ),
        "selected_strong_candidate_count": int(
            sum(1 for row in selected if row["strong_candidate"])
        ),
        "rejection_reasons": {
            reason: sum(1 for row in rejected if row["rejection_reason"] == reason)
            for reason in sorted({row["rejection_reason"] for row in rejected})
            if reason
        },
        "conflict_reasons": {
            reason: sum(1 for row in conflicted if row["conflict_reason"] == reason)
            for reason in sorted({row["conflict_reason"] for row in conflicted})
            if reason
        },
    }


def write_split_repair_candidate_selection_report(
    probes: RustSplitMergeRepairProbes | dict[str, np.ndarray],
    output_dir: Path,
    policy: SplitRepairSelectionPolicy | None = None,
    *,
    min_weight: float = 0.0,
    max_weight: float = 0.0,
    rows: list[dict[str, Any]] | None = None,
) -> dict[str, str]:
    """Write a utility/cost-ranked split-repair candidate selection table."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if policy is None:
        policy = SplitRepairSelectionPolicy()
    if rows is None:
        rows = rank_split_repair_candidates(
            probes,
            policy,
            min_weight=min_weight,
            max_weight=max_weight,
        )
    summary = summarize_split_repair_candidate_selection(rows)
    summary["policy"] = asdict(policy)
    summary["min_weight"] = float(min_weight)
    summary["max_weight"] = float(max_weight)

    summary_path = output_dir / "split_repair_candidate_selection_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    csv_path = output_dir / "split_repair_candidate_selection.csv"
    fieldnames = [
        "rank",
        "selected_for_apply",
        "accepted_by_policy",
        "rejection_reason",
        "conflict_reason",
        "policy",
        "selection_mode",
        "cluster",
        "gamma_multiplier",
        "probe_resolution",
        "block_count",
        "doc_weight",
        "induced_directed_edges",
        "repair_quotient_edges",
        "probe_count_for_cluster",
        "n_parts",
        "core_part_count",
        "singleton_weight",
        "cut_weight",
        "split_delta_q_base",
        "split_delta_q_probe",
        "repair_merge_count",
        "repair_delta_q",
        "net_delta_q",
        "final_source_units",
        "retained_source_units",
        "escaped_source_units",
        "escaped_source_weight",
        "final_small_source_units",
        "final_small_source_weight",
        "largest_source_unit_fraction",
        "largest_source_unit_weight",
        "restored_source_cluster",
        "strong_candidate",
        "structural_gain",
        "size_band_improvement",
        "remaining_oversize_before",
        "remaining_oversize_after",
        "oversize_reduction",
        "passes_size_objective",
        "target_cluster_count_improvement",
        "escaped_source_weight_if_net_positive",
        "singleton_or_leaf_penalty",
        "restored_source_cluster_penalty",
        "utility",
        "cost",
        "priority",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return {"summary": str(summary_path), "candidates": str(csv_path)}


def summarize_split_merge_repair_probes(probes: RustSplitMergeRepairProbes) -> dict[str, Any]:
    """Return aggregate diagnostics for split-then-repair probes."""

    net_positive = probes.net_delta_q > 0
    net_positive_eps = probes.net_delta_q > 1e-6
    net_positive_one = probes.net_delta_q > 1.0
    escaped = probes.escaped_source_units > 0
    retained_split = probes.retained_source_units >= 2
    restored = probes.restored_source_cluster
    return {
        "n_probes": int(probes.n_probes),
        "n_net_positive": int(net_positive.sum()),
        "n_net_positive_gt_1e_minus_6": int(net_positive_eps.sum()),
        "n_net_positive_gt_1": int(net_positive_one.sum()),
        "n_net_positive_escaped_gt_1e_minus_6": int((net_positive_eps & escaped).sum()),
        "n_with_repair_merges": int((probes.repair_merge_count > 0).sum()),
        "n_with_escaped_source": int(escaped.sum()),
        "n_retained_split": int(retained_split.sum()),
        "n_restored_source_cluster": int(restored.sum()),
        "net_delta_q": _percentiles(probes.net_delta_q, [50, 90, 95, 99]),
        "net_delta_q_max": float(probes.net_delta_q.max()) if probes.n_probes else 0.0,
        "repair_delta_q": _percentiles(probes.repair_delta_q, [50, 90, 95, 99]),
        "escaped_source_weight": _percentiles(
            probes.escaped_source_weight, [50, 90, 95, 99]
        ),
        "largest_source_unit_fraction": _percentiles(
            probes.largest_source_unit_fraction, [50, 90, 95, 99]
        ),
        "final_source_units": _percentiles(
            probes.final_source_units.astype(np.float64), [50, 90, 95, 99]
        ),
        "induced_directed_edges": _percentiles(
            probes.induced_directed_edges.astype(np.float64), [50, 90, 95, 99]
        ),
        "repair_quotient_edges": _percentiles(
            probes.repair_quotient_edges.astype(np.float64), [50, 90, 95, 99]
        ),
    }


def write_split_merge_repair_probe_report(
    probes: RustSplitMergeRepairProbes,
    output_dir: Path,
    *,
    top_probes: int | None = None,
) -> dict[str, str]:
    """Write JSON/CSV diagnostics for split-then-repair probes."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_split_merge_repair_probes(probes)
    summary_path = output_dir / "split_merge_repair_probe_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    order = np.lexsort((probes.cluster, -probes.net_delta_q))
    if top_probes is not None:
        order = order[: int(top_probes)]

    csv_path = output_dir / "split_merge_repair_probes.csv"
    fieldnames = [
        "rank",
        "cluster",
        "gamma_multiplier",
        "probe_resolution",
        "block_count",
        "doc_weight",
        "induced_directed_edges",
        "n_parts",
        "core_part_count",
        "singleton_weight",
        "cut_weight",
        "split_delta_q_base",
        "split_delta_q_probe",
        "repair_quotient_edges",
        "repair_merge_count",
        "repair_delta_q",
        "net_delta_q",
        "final_source_units",
        "retained_source_units",
        "escaped_source_units",
        "escaped_source_weight",
        "final_small_source_units",
        "final_small_source_weight",
        "largest_source_unit_fraction",
        "restored_source_cluster",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for rank, idx in enumerate(order, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "cluster": int(probes.cluster[idx]),
                    "gamma_multiplier": float(probes.gamma_multiplier[idx]),
                    "probe_resolution": float(probes.probe_resolution[idx]),
                    "block_count": int(probes.block_count[idx]),
                    "doc_weight": float(probes.doc_weight[idx]),
                    "induced_directed_edges": int(probes.induced_directed_edges[idx]),
                    "n_parts": int(probes.n_parts[idx]),
                    "core_part_count": int(probes.core_part_count[idx]),
                    "singleton_weight": float(probes.singleton_weight[idx]),
                    "cut_weight": float(probes.cut_weight[idx]),
                    "split_delta_q_base": float(probes.split_delta_q_base[idx]),
                    "split_delta_q_probe": float(probes.split_delta_q_probe[idx]),
                    "repair_quotient_edges": int(probes.repair_quotient_edges[idx]),
                    "repair_merge_count": int(probes.repair_merge_count[idx]),
                    "repair_delta_q": float(probes.repair_delta_q[idx]),
                    "net_delta_q": float(probes.net_delta_q[idx]),
                    "final_source_units": int(probes.final_source_units[idx]),
                    "retained_source_units": int(probes.retained_source_units[idx]),
                    "escaped_source_units": int(probes.escaped_source_units[idx]),
                    "escaped_source_weight": float(probes.escaped_source_weight[idx]),
                    "final_small_source_units": int(probes.final_small_source_units[idx]),
                    "final_small_source_weight": float(
                        probes.final_small_source_weight[idx]
                    ),
                    "largest_source_unit_fraction": float(
                        probes.largest_source_unit_fraction[idx]
                    ),
                    "restored_source_cluster": bool(probes.restored_source_cluster[idx]),
                }
            )

    return {"summary": str(summary_path), "probes": str(csv_path)}


def write_adaptive_refinement_report(
    stats: RustClusterGraphStats,
    output_dir: Path,
    *,
    min_weight: float = 0.0,
    max_weight: float = 0.0,
    top_candidates: int = 1000,
    write_cluster_arrays: bool = True,
) -> dict[str, str]:
    """Write JSON/CSV/NPZ diagnostics for adaptive-refinement dry-runs."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = summarize_cluster_graph_stats(
        stats,
        min_weight=min_weight,
        max_weight=max_weight,
    )
    summary_path = output_dir / "cluster_graph_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    candidate_path = output_dir / "macro_merge_candidates.csv"
    n_candidates = min(int(top_candidates), stats.n_candidates)
    with candidate_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "rank",
                "source",
                "target",
                "source_doc_weight",
                "target_doc_weight",
                "source_block_count",
                "target_block_count",
                "source_internal_weight",
                "target_internal_weight",
                "source_external_weight",
                "target_external_weight",
                "source_degree",
                "target_degree",
                "source_conductance",
                "target_conductance",
                "source_leafness",
                "target_leafness",
                "edge_weight",
                "delta_q",
                "merged_weight",
                "size_band_gain",
            ],
        )
        writer.writeheader()
        for idx in range(n_candidates):
            source = int(stats.candidate_source[idx])
            target = int(stats.candidate_target[idx])
            writer.writerow(
                {
                    "rank": idx + 1,
                    "source": source,
                    "target": target,
                    "source_doc_weight": float(stats.doc_weight[source]),
                    "target_doc_weight": float(stats.doc_weight[target]),
                    "source_block_count": int(stats.block_count[source]),
                    "target_block_count": int(stats.block_count[target]),
                    "source_internal_weight": float(stats.internal_weight[source]),
                    "target_internal_weight": float(stats.internal_weight[target]),
                    "source_external_weight": float(stats.external_weight[source]),
                    "target_external_weight": float(stats.external_weight[target]),
                    "source_degree": int(stats.degree[source]),
                    "target_degree": int(stats.degree[target]),
                    "source_conductance": float(stats.conductance[source]),
                    "target_conductance": float(stats.conductance[target]),
                    "source_leafness": float(stats.leafness[source]),
                    "target_leafness": float(stats.leafness[target]),
                    "edge_weight": float(stats.candidate_edge_weight[idx]),
                    "delta_q": float(stats.candidate_delta_q[idx]),
                    "merged_weight": float(stats.candidate_merged_weight[idx]),
                    "size_band_gain": float(stats.candidate_size_band_gain[idx]),
                }
            )

    paths = {
        "summary": str(summary_path),
        "merge_candidates": str(candidate_path),
    }
    if write_cluster_arrays:
        arrays_path = output_dir / "cluster_graph_stats.npz"
        np.savez(
            arrays_path,
            block_count=stats.block_count,
            doc_weight=stats.doc_weight,
            internal_weight=stats.internal_weight,
            external_weight=stats.external_weight,
            degree=stats.degree,
            top_neighbor=stats.top_neighbor,
            top_neighbor_weight=stats.top_neighbor_weight,
            second_neighbor=stats.second_neighbor,
            second_neighbor_weight=stats.second_neighbor_weight,
            neighbor_weight_ratio=stats.neighbor_weight_ratio,
            conductance=stats.conductance,
            leafness=stats.leafness,
            band_distance=stats.band_distance,
            candidate_source=stats.candidate_source,
            candidate_target=stats.candidate_target,
            candidate_edge_weight=stats.candidate_edge_weight,
            candidate_delta_q=stats.candidate_delta_q,
            candidate_merged_weight=stats.candidate_merged_weight,
            candidate_size_band_gain=stats.candidate_size_band_gain,
        )
        paths["cluster_arrays"] = str(arrays_path)
    return paths


__all__ = [
    "BoundaryCandidatePolicy",
    "MacroMergePolicy",
    "MacroMergeSimulationResult",
    "SplitRepairSelectionPolicy",
    "exploratory_boundary_candidate_policies",
    "exploratory_macro_merge_policies",
    "rank_split_repair_candidates",
    "run_macro_merge_policy_ensemble",
    "score_boundary_candidates",
    "simulate_macro_merge_policy",
    "summarize_boundary_candidate_policies",
    "summarize_boundary_group_probes",
    "summarize_boundary_move_probes",
    "summarize_cluster_graph_stats",
    "summarize_multi_core_split_probes",
    "summarize_split_repair_candidate_selection",
    "summarize_split_merge_repair_probes",
    "write_adaptive_refinement_report",
    "write_boundary_candidate_report",
    "write_boundary_group_probe_report",
    "write_boundary_move_probe_report",
    "write_macro_merge_ensemble_report",
    "write_multi_core_split_probe_report",
    "write_split_repair_candidate_selection_report",
    "write_split_merge_repair_probe_report",
]
