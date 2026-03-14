"""Resolution search utilities for Leiden clustering."""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from itertools import combinations
import math
import multiprocessing as mp
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import igraph as ig
import leidenalg as la

from .partitioning import partition_class
from .config import PostprocessConfig
from .postprocess import merge_small_clusters
from .runner import LeidenRunner


@dataclass
class ResolutionResult:
    name: str
    resolution: float
    partition: la.VertexPartition
    cluster_count: int
    quality: float


@dataclass(frozen=True)
class ResolutionScanEntry:
    resolution: float
    seed: int | None
    quality: float
    cluster_count: int
    membership: List[int]
    # Cluster count before postprocess (Leiden raw membership).
    # Useful to avoid selecting degenerate high-resolution solutions where every node
    # becomes its own community and postprocess does all the work.
    raw_cluster_count: int | None = None


@dataclass(frozen=True)
class ResolutionScanResult:
    entries: List[ResolutionScanEntry]
    stability: Dict[float, float] | None


_SCAN_GRAPH: ig.Graph | None = None
_SCAN_OBJECTIVE: str | None = None
_SCAN_ITERATIONS: int | None = None
_SCAN_POSTPROCESS: PostprocessConfig | None = None


def _scan_worker_init(
    graph: ig.Graph,
    objective: str,
    iterations: Optional[int],
    postprocess: PostprocessConfig | None,
) -> None:
    """Initialise shared state for resolution scan workers."""

    global _SCAN_GRAPH, _SCAN_OBJECTIVE, _SCAN_ITERATIONS, _SCAN_POSTPROCESS

    _SCAN_GRAPH = graph.copy()
    _SCAN_OBJECTIVE = objective
    _SCAN_ITERATIONS = iterations
    _SCAN_POSTPROCESS = postprocess


def _scan_worker(task: tuple[float, int | None]) -> ResolutionScanEntry:
    """Execute a single resolution/seed combination using worker globals."""

    if _SCAN_GRAPH is None or _SCAN_OBJECTIVE is None:
        raise RuntimeError("scan worker not initialised")

    gamma, seed = task

    runner = LeidenRunner(
        _SCAN_GRAPH,
        objective=_SCAN_OBJECTIVE,
        default_iterations=_SCAN_ITERATIONS,
        default_seed=seed,
    )
    result = runner.run(
        gamma,
        seed=seed,
        n_iterations=_SCAN_ITERATIONS,
    )
    raw_membership = result.membership
    raw_cluster_count = int(result.cluster_count)
    membership = raw_membership

    if _SCAN_POSTPROCESS is not None:
        node_weights = (
            _SCAN_GRAPH.vs["weight"] if "weight" in _SCAN_GRAPH.vs.attributes() else None
        )
        min_size, min_weight = _SCAN_POSTPROCESS.resolve_thresholds(
            has_node_weights=node_weights is not None
        )
        membership = merge_small_clusters(
            _SCAN_GRAPH,
            membership,
            min_size=min_size,
            min_weight=min_weight,
            node_weights=node_weights,
            max_passes=max(_SCAN_POSTPROCESS.max_passes, 1),
        ).membership

    return ResolutionScanEntry(
        resolution=gamma,
        seed=seed,
        quality=result.quality,
        cluster_count=len(set(membership)),
        membership=membership,
        raw_cluster_count=raw_cluster_count,
    )


def _evaluate_partition(
    graph: ig.Graph,
    partition_cls,
    weights,
    gamma: float,
    cache: Dict[float, ResolutionResult],
    name: str,
    n_iterations: Optional[int] = None,
    initial_membership: Optional[Sequence[int]] = None,
    seed: Optional[int] = None,
) -> ResolutionResult:
    if gamma in cache:
        return cache[gamma]

    if initial_membership is None and cache:
        nearest_gamma = min(cache, key=lambda g: abs(g - gamma))
        initial_membership = cache[nearest_gamma].partition.membership

    extra_kwargs = {}
    if n_iterations is not None:
        extra_kwargs["n_iterations"] = n_iterations
    if initial_membership is not None:
        extra_kwargs["initial_membership"] = initial_membership
    if seed is not None:
        extra_kwargs["seed"] = seed

    partition = la.find_partition(
        graph,
        partition_type=partition_cls,
        weights=weights,
        resolution_parameter=gamma,
        **extra_kwargs,
    )
    result = ResolutionResult(
        name=name,
        resolution=gamma,
        partition=partition,
        cluster_count=len(partition),
        quality=partition.quality(),
    )
    cache[gamma] = result
    return result


def _emit_result(
    progress: Optional[Callable[[str], None]],
    name: str,
    result: ResolutionResult,
    stage: str,
) -> None:
    if not progress:
        return
    progress(
        f"{name}: {stage} gamma={result.resolution:.6g} -> "
        f"{result.cluster_count} clusters (quality={result.quality:.6f})"
    )


def _distance_to_range(count: int, min_clusters: int, max_clusters: int) -> int:
    if count < min_clusters:
        return min_clusters - count
    if count > max_clusters:
        return count - max_clusters
    return 0


def _search_resolution(
    graph: ig.Graph,
    name: str,
    min_clusters: int,
    max_clusters: int,
    bounds: Tuple[float, float],
    max_iterations: int,
    objective: str,
    progress: Optional[Callable[[str], None]] = None,
    n_iterations: Optional[int] = None,
    cache: Optional[Dict[float, ResolutionResult]] = None,
    seed: Optional[int] = None,
) -> ResolutionResult:
    partition_cls = partition_class(objective)
    weights = graph.es["weight"] if "weight" in graph.es.attributes() else None

    lower_bound, upper_bound = bounds
    if lower_bound <= 0 or upper_bound <= 0:
        raise ValueError("resolution bounds must be positive")
    if lower_bound >= upper_bound:
        raise ValueError("resolution lower bound must be less than upper bound")

    if cache is None:
        cache = {}

    best_result: ResolutionResult | None = None
    best_distance = float("inf")

    def update_best(candidate: ResolutionResult) -> None:
        nonlocal best_result, best_distance
        distance = _distance_to_range(candidate.cluster_count, min_clusters, max_clusters)
        if distance < best_distance or (
            distance == best_distance and best_result is not None and candidate.resolution < best_result.resolution
        ):
            best_result = candidate
            best_distance = distance

    # Evaluate bounds and attempt to bracket the target range.
    lower_result = _evaluate_partition(
        graph,
        partition_cls,
        weights,
        lower_bound,
        cache,
        name,
        n_iterations=n_iterations,
        seed=seed,
    )
    _emit_result(progress, name, lower_result, "evaluated")
    update_best(lower_result)
    upper_result = _evaluate_partition(
        graph,
        partition_cls,
        weights,
        upper_bound,
        cache,
        name,
        n_iterations=n_iterations,
        seed=seed,
    )
    _emit_result(progress, name, upper_result, "evaluated")
    update_best(upper_result)

    # Expand bounds if needed.
    expansion_limit = max_iterations
    expand_lo = lower_bound
    count_lo = lower_result.cluster_count
    for _ in range(expansion_limit):
        if count_lo <= max_clusters or expand_lo < 1e-9:
            break
        expand_lo *= 0.5
        lower_result = _evaluate_partition(
            graph,
            partition_cls,
            weights,
            expand_lo,
            cache,
            name,
            n_iterations=n_iterations,
            seed=seed,
        )
        _emit_result(progress, name, lower_result, "expanded lower")
        update_best(lower_result)
        count_lo = lower_result.cluster_count
    lower_bound = expand_lo

    expand_hi = upper_bound
    count_hi = upper_result.cluster_count
    for _ in range(expansion_limit):
        if count_hi >= min_clusters or expand_hi > 1e9:
            break
        expand_hi *= 2.0
        upper_result = _evaluate_partition(
            graph,
            partition_cls,
            weights,
            expand_hi,
            cache,
            name,
            n_iterations=n_iterations,
            seed=seed,
        )
        _emit_result(progress, name, upper_result, "expanded upper")
        update_best(upper_result)
        count_hi = upper_result.cluster_count
    upper_bound = expand_hi

    lower_count = lower_result.cluster_count
    upper_count = upper_result.cluster_count

    if lower_count > max_clusters and upper_count > max_clusters:
        if best_result is None:
            raise RuntimeError("Failed to evaluate any Leiden partitions")
        return best_result
    if lower_count < min_clusters and upper_count < min_clusters:
        if best_result is None:
            raise RuntimeError("Failed to evaluate any Leiden partitions")
        return best_result

    lo_gamma = lower_result.resolution
    hi_gamma = upper_result.resolution

    for _ in range(max_iterations):
        mid_gamma = (lo_gamma + hi_gamma) / 2.0
        mid_result = _evaluate_partition(
            graph,
            partition_cls,
            weights,
            mid_gamma,
            cache,
            name,
            n_iterations=n_iterations,
            seed=seed,
        )
        _emit_result(progress, name, mid_result, "bisected")
        update_best(mid_result)

        if min_clusters <= mid_result.cluster_count <= max_clusters:
            return mid_result

        if mid_result.cluster_count < min_clusters:
            lo_gamma = mid_gamma
        else:
            hi_gamma = mid_gamma

    if best_result is None:
        raise RuntimeError("Failed to converge on a resolution; no candidates evaluated")
    return best_result


def resolve_resolution_schedule(
    graph: ig.Graph,
    constraints: Sequence[tuple[int, int]],
    objective: str,
    bounds: Tuple[float, float],
    max_iterations: int,
    progress: Optional[Callable[[str], None]] = None,
    n_iterations: Optional[int] = None,
    seed: Optional[int] = None,
) -> "OrderedDict[str, ResolutionResult]":
    """Determine resolution parameters that satisfy cluster count ranges."""

    schedule: "OrderedDict[str, ResolutionResult]" = OrderedDict()
    shared_cache: Dict[float, ResolutionResult] = {}

    for idx, constraint in enumerate(constraints, start=1):
        if len(constraint) != 2:
            raise ValueError("Each constraint must be a (min_clusters, max_clusters) tuple")
        min_clusters, max_clusters = constraint
        if min_clusters <= 0 or max_clusters <= 0:
            raise ValueError("Cluster bounds must be positive integers")
        if min_clusters > max_clusters:
            raise ValueError("min_clusters must be less than or equal to max_clusters")

        name = f"level-{idx}"
        result = _search_resolution(
            graph,
            name,
            min_clusters,
            max_clusters,
            bounds,
            max_iterations,
            objective,
            progress,
            n_iterations,
            cache=shared_cache,
            seed=seed,
        )
        if progress:
            progress(
                f"{name}: selected gamma={result.resolution:.6g} -> "
                f"{result.cluster_count} clusters"
            )
        schedule[name] = result

    return schedule


def _normalized_mutual_information(a: Sequence[int], b: Sequence[int]) -> float:
    if len(a) != len(b):
        raise ValueError("label sequences must have the same length")

    n = len(a)
    if n == 0:
        return 0.0

    counts_a: Dict[int, int] = defaultdict(int)
    counts_b: Dict[int, int] = defaultdict(int)
    joint: Dict[tuple[int, int], int] = defaultdict(int)

    for label_a, label_b in zip(a, b):
        counts_a[label_a] += 1
        counts_b[label_b] += 1
        joint[(label_a, label_b)] += 1

    def entropy(counts: Iterable[int]) -> float:
        total = float(n)
        value = 0.0
        for count in counts:
            if count == 0:
                continue
            p = count / total
            value -= p * math.log(p)
        return value

    entropy_a = entropy(counts_a.values())
    entropy_b = entropy(counts_b.values())
    if entropy_a == 0 and entropy_b == 0:
        return 1.0

    mutual_information = 0.0
    for (label_a, label_b), count in joint.items():
        p_ab = count / float(n)
        p_a = counts_a[label_a] / float(n)
        p_b = counts_b[label_b] / float(n)
        mutual_information += p_ab * math.log(p_ab / (p_a * p_b))

    denominator = (entropy_a + entropy_b) / 2.0
    if denominator == 0:
        return 0.0
    return max(0.0, min(1.0, mutual_information / denominator))


def scan_resolution_grid(
    graph: ig.Graph,
    resolutions: Sequence[float],
    seeds: Sequence[int | None] = (0,),
    *,
    objective: str = "modularity",
    n_iterations: int | None = None,
    postprocess: PostprocessConfig | None = None,
    stability_metric: str | None = None,
    parallel: bool = False,
    workers: int | None = None,
    start_method: str | None = None,
    progress: Optional[Callable[[str], None]] = None,
) -> ResolutionScanResult:
    """Evaluate a grid of resolution/seed pairs and report cluster statistics."""

    tasks = [(float(gamma), seed) for gamma in resolutions for seed in seeds]
    entries: List[ResolutionScanEntry] = []

    def log(msg: str) -> None:
        if progress:
            progress(msg)

    if parallel and tasks:
        ctx, resolved_start_method, used_fallback = _resolve_parallel_context(
            preferred=start_method or "fork"
        )
        if used_fallback:
            log(
                "parallel start_method fallback: "
                f"requested={start_method or 'fork'} -> using={resolved_start_method}"
            )

        with ctx.Pool(
            processes=workers,
            initializer=_scan_worker_init,
            initargs=(graph, objective, n_iterations, postprocess),
        ) as pool:
            for entry in pool.imap_unordered(_scan_worker, tasks):
                entries.append(entry)
                log(
                    f"gamma={entry.resolution:.6g} seed={entry.seed} -> "
                    f"{entry.cluster_count} clusters (quality={entry.quality:.5f})"
                )
    else:
        runner = LeidenRunner(
            graph,
            objective=objective,
            default_iterations=n_iterations,
        )
        for gamma, seed in tasks:
            result = runner.run(
                gamma,
                seed=seed,
                n_iterations=n_iterations,
            )
            raw_membership = result.membership
            raw_cluster_count = int(result.cluster_count)
            membership = raw_membership
            if postprocess is not None:
                node_weights = (
                    graph.vs["weight"] if "weight" in graph.vs.attributes() else None
                )
                min_size, min_weight = postprocess.resolve_thresholds(
                    has_node_weights=node_weights is not None
                )
                membership = merge_small_clusters(
                    graph,
                    membership,
                    min_size=min_size,
                    min_weight=min_weight,
                    node_weights=node_weights,
                    max_passes=max(postprocess.max_passes, 1),
                ).membership
            entries.append(
                ResolutionScanEntry(
                    resolution=gamma,
                    seed=seed,
                    quality=result.quality,
                    cluster_count=len(set(membership)),
                    membership=membership,
                    raw_cluster_count=raw_cluster_count,
                )
            )
            log(
                f"gamma={gamma:.6g} seed={seed} -> "
                f"{entries[-1].cluster_count} clusters (quality={result.quality:.5f})"
            )

    stability: Dict[float, float] | None = None
    if stability_metric:
        grouped: Dict[float, List[List[int]]] = defaultdict(list)
        for entry in entries:
            grouped[entry.resolution].append(entry.membership)

        stability = {}
        if stability_metric != "nmi":
            raise ValueError(f"Unsupported stability metric: {stability_metric}")

        for gamma, memberships in grouped.items():
            if len(memberships) < 2:
                stability[gamma] = 1.0
                continue
            scores = [
                _normalized_mutual_information(m1, m2)
                for m1, m2 in combinations(memberships, 2)
            ]
            stability[gamma] = sum(scores) / len(scores) if scores else 1.0

    return ResolutionScanResult(entries=entries, stability=stability)


def _resolve_parallel_context(preferred: str = "fork") -> tuple[mp.context.BaseContext, str, bool]:
    """Resolve multiprocessing context with platform-safe fallback."""

    available = tuple(mp.get_all_start_methods())
    requested = preferred

    try:
        return mp.get_context(requested), requested, False
    except ValueError:
        fallback = mp.get_start_method(allow_none=True)
        if fallback is None:
            fallback = available[0] if available else "spawn"
        return mp.get_context(fallback), fallback, True


__all__ = [
    "resolve_resolution_schedule",
    "ResolutionResult",
    "ResolutionScanEntry",
    "ResolutionScanResult",
    "scan_resolution_grid",
]
