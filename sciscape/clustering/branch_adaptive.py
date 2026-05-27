"""Branch-adaptive CPM split diagnostics.

The helpers in this module are intentionally small and deterministic.  They
support pilot scripts that generate local Leiden split candidates, then score
and select them using exact CPM accounting on the original graph convention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np


DEFAULT_STAGE1_ALPHAS = (1.05, 1.15, 1.25)
DEFAULT_STAGE2_ALPHAS = (1.02, 1.05, 1.10, 1.15, 1.20, 1.25)
DEFAULT_SOURCE_SEEDS = (11, 42, 73)
DEFAULT_LOCAL_SEEDS = (11, 42, 73)
DEFAULT_TAU_SPLIT_RATIOS = (0.0, 0.001, 0.005, 0.01, 0.05)


@dataclass(frozen=True)
class SplitAccounting:
    """CPM accounting for a proposed parent-to-children split."""

    delta_q_split_original: float
    e_between: float
    w_between: float
    gamma_star_split: float
    raw_split_gap: float
    normalized_split_gain: float


def cpm_merge_gain(
    edge_weight: float,
    weight_a: float,
    weight_b: float,
    gamma: float,
) -> float:
    """Return CPM quality change from merging two communities."""

    return float(edge_weight) - float(gamma) * float(weight_a) * float(weight_b)


def split_accounting_from_between(
    *,
    e_between: float,
    w_between: float,
    gamma: float,
) -> SplitAccounting:
    """Return critical-gamma diagnostics for a multiway split.

    ``e_between`` is the total original-graph edge weight between proposed
    children. ``w_between`` is ``sum_i<j W_i W_j`` over child document weights.
    """

    e_between = float(e_between)
    w_between = float(w_between)
    gamma = float(gamma)
    if w_between <= 0.0 or not math.isfinite(w_between):
        gamma_star = math.inf
        raw_gap = -math.inf
        normalized = -math.inf
        delta_q = -e_between
    else:
        gamma_star = e_between / w_between
        raw_gap = gamma - gamma_star
        normalized = raw_gap
        delta_q = gamma * w_between - e_between
    return SplitAccounting(
        delta_q_split_original=float(delta_q),
        e_between=e_between,
        w_between=w_between,
        gamma_star_split=float(gamma_star),
        raw_split_gap=float(raw_gap),
        normalized_split_gain=float(normalized),
    )


def split_accounting_from_delta_cut(
    *,
    delta_q_split_original: float,
    e_between: float,
    gamma: float,
) -> SplitAccounting:
    """Recover split accounting from Rust probe delta and cut weight."""

    gamma = float(gamma)
    delta_q = float(delta_q_split_original)
    e_between = float(e_between)
    if gamma <= 0.0 or not math.isfinite(gamma):
        w_between = math.nan
    else:
        w_between = (delta_q + e_between) / gamma
        if abs(w_between) < 1e-12:
            w_between = 0.0
    return split_accounting_from_between(
        e_between=e_between,
        w_between=w_between,
        gamma=gamma,
    )


def split_accounting_from_child_weights(
    child_weights: Sequence[float],
    *,
    e_between: float,
    gamma: float,
) -> SplitAccounting:
    """Compute split accounting directly from child document weights."""

    weights = np.asarray(child_weights, dtype=np.float64)
    total = float(weights.sum())
    sum_square = float(np.sum(weights * weights))
    w_between = (total * total - sum_square) / 2.0
    return split_accounting_from_between(
        e_between=float(e_between),
        w_between=w_between,
        gamma=float(gamma),
    )


def split_accounting_from_edges(
    src: Sequence[int],
    dst: Sequence[int],
    edge_weight: Sequence[float],
    *,
    nodes: Sequence[int],
    child_labels: Sequence[int],
    node_weights: Sequence[float] | None = None,
    gamma: float,
) -> SplitAccounting:
    """Compute split accounting from an edge list using original node ids.

    Edge rows are interpreted as undirected rows. If callers pass a symmetric
    directed edge list, they should halve weights or deduplicate first.
    """

    node_array = np.asarray(nodes, dtype=np.int64)
    labels = np.asarray(child_labels, dtype=np.int64)
    if node_array.shape[0] != labels.shape[0]:
        raise ValueError("nodes and child_labels must have the same length")
    if node_array.size == 0:
        return split_accounting_from_between(e_between=0.0, w_between=0.0, gamma=gamma)

    if node_weights is None:
        weights = np.ones(node_array.shape[0], dtype=np.float64)
    else:
        weights = np.asarray(node_weights, dtype=np.float64)
        if weights.shape[0] != node_array.shape[0]:
            raise ValueError("node_weights must match nodes length")

    local = {int(node): idx for idx, node in enumerate(node_array.tolist())}
    _, remapped_labels = np.unique(labels, return_inverse=True)
    child_weights = np.bincount(remapped_labels, weights=weights)
    e_between = 0.0
    for u, v, weight in zip(src, dst, edge_weight, strict=False):
        i = local.get(int(u))
        if i is None:
            continue
        j = local.get(int(v))
        if j is None:
            continue
        if int(remapped_labels[i]) != int(remapped_labels[j]):
            e_between += float(weight)
    return split_accounting_from_child_weights(
        child_weights,
        e_between=e_between,
        gamma=gamma,
    )


def child_size_diagnostics(
    child_weights: Sequence[float],
    *,
    min_doc_weight: float,
) -> dict[str, float | int]:
    """Return balance diagnostics for a candidate child-size distribution."""

    weights = np.asarray(child_weights, dtype=np.float64)
    weights = weights[np.isfinite(weights) & (weights > 0.0)]
    if weights.size == 0:
        return {
            "child_weight_entropy": 0.0,
            "n_children_below_min": 0,
            "largest_child_fraction": 0.0,
        }
    total = float(weights.sum())
    if weights.size == 1:
        entropy = 0.0
    else:
        p = weights / total
        entropy = -float(np.sum(p * np.log(p))) / math.log(float(weights.size))
    return {
        "child_weight_entropy": float(entropy),
        "n_children_below_min": int((weights < float(min_doc_weight)).sum()),
        "largest_child_fraction": float(weights.max() / total) if total > 0.0 else 0.0,
    }


def source_max_ratio_delta_if_applied(
    *,
    parent_doc_weight: float,
    largest_child_fraction: float,
    target_max_doc_weight: float,
) -> float:
    """Return after-before max/target ratio for splitting one parent.

    Negative values mean the candidate reduces source max pressure.
    """

    target = float(target_max_doc_weight)
    if target <= 0.0:
        return 0.0
    parent = float(parent_doc_weight)
    after = parent * float(largest_child_fraction)
    return float(after / target - parent / target)


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _row_passes_policy(
    row: dict[str, Any],
    *,
    tau_split_abs: float,
    epsilon_q: float,
) -> tuple[bool, str]:
    status = str(row.get("status", "ok"))
    if status != "ok":
        return False, status
    if _finite_float(row.get("k_children")) < 2:
        return False, "not_split"
    if _finite_float(row.get("w_between")) <= 0.0:
        return False, "zero_w_between"
    if _finite_float(row.get("delta_q_split_original")) < float(epsilon_q):
        return False, "quality_regression"
    if _finite_float(row.get("normalized_split_gain"), -math.inf) < float(tau_split_abs):
        return False, "below_tau_split"
    return True, ""


def _candidate_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    source_reduction = -_finite_float(row.get("source_max_ratio_delta_if_applied"))
    return (
        -_finite_float(row.get("normalized_split_gain"), -math.inf),
        -_finite_float(row.get("delta_q_split_original"), -math.inf),
        -source_reduction,
        -_finite_float(row.get("child_weight_entropy")),
        _finite_float(row.get("n_children_below_min"), math.inf),
        _finite_float(row.get("largest_child_fraction"), math.inf),
        str(row.get("field", "")),
        str(row.get("sample", "")),
        int(_finite_float(row.get("source_seed"))),
        int(_finite_float(row.get("parent_cluster"))),
        _finite_float(row.get("alpha")),
        int(_finite_float(row.get("local_seed"))),
    )


def rank_branch_split_candidates(
    rows: Iterable[dict[str, Any]],
    *,
    gamma: float,
    tau_split_ratio: float,
    epsilon_q: float = 0.0,
) -> list[dict[str, Any]]:
    """Tag and greedily select non-overlapping branch split candidates.

    The resolver is parent-exclusive: all candidates for the same source
    parent overlap, and the first policy-passing candidate in deterministic
    score order wins that parent.
    """

    tau_split_abs = float(tau_split_ratio) * float(gamma)
    prepared: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        out = dict(row)
        passes, reason = _row_passes_policy(
            out,
            tau_split_abs=tau_split_abs,
            epsilon_q=float(epsilon_q),
        )
        out["candidate_index"] = int(out.get("candidate_index", idx))
        out["tau_split_ratio"] = float(tau_split_ratio)
        out["tau_split_abs"] = tau_split_abs
        out["accepted_by_policy"] = bool(passes)
        out["rejection_reason"] = reason
        out["selected_for_apply"] = False
        out["selection_rank"] = None
        out["conflict_reason"] = ""
        prepared.append(out)

    selected_parents: set[tuple[str, int, int]] = set()
    rank = 0
    for out in sorted(prepared, key=_candidate_sort_key):
        if not out["accepted_by_policy"]:
            continue
        parent_key = (
            str(out.get("sample", "")),
            int(_finite_float(out.get("source_seed"))),
            int(_finite_float(out.get("parent_cluster"))),
        )
        if parent_key in selected_parents:
            out["conflict_reason"] = "parent_already_selected"
            continue
        selected_parents.add(parent_key)
        rank += 1
        out["selected_for_apply"] = True
        out["selection_rank"] = rank

    return sorted(
        prepared,
        key=lambda row: (
            row["selection_rank"] is None,
            row["selection_rank"] if row["selection_rank"] is not None else math.inf,
            row["candidate_index"],
        ),
    )


def mean_pairwise_ami(partitions: Sequence[Sequence[int]]) -> float:
    """Return mean pairwise adjusted mutual information for local partitions."""

    if len(partitions) < 2:
        return 1.0
    from sklearn.metrics import adjusted_mutual_info_score

    values: list[float] = []
    arrays = [np.asarray(partition, dtype=np.int64) for partition in partitions]
    for i in range(len(arrays)):
        for j in range(i + 1, len(arrays)):
            if arrays[i].shape[0] != arrays[j].shape[0]:
                continue
            values.append(float(adjusted_mutual_info_score(arrays[i], arrays[j])))
    return float(np.mean(values)) if values else 0.0


def best_match_child_jaccard(labels_a: Sequence[int], labels_b: Sequence[int]) -> float:
    """Return mean best child-overlap Jaccard from ``labels_a`` to ``labels_b``."""

    a = np.asarray(labels_a, dtype=np.int64)
    b = np.asarray(labels_b, dtype=np.int64)
    if a.shape[0] != b.shape[0]:
        raise ValueError("labels_a and labels_b must have the same length")
    if a.size == 0:
        return 1.0
    scores: list[float] = []
    for label in np.unique(a):
        mask_a = a == label
        best = 0.0
        for other in np.unique(b):
            mask_b = b == other
            union = int(np.count_nonzero(mask_a | mask_b))
            if union:
                best = max(best, float(np.count_nonzero(mask_a & mask_b) / union))
        scores.append(best)
    return float(np.mean(scores)) if scores else 0.0
