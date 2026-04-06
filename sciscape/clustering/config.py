"""Configuration and data structures for Leiden clustering pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence, Tuple

import polars as pl


@dataclass
class LeidenConfig:
    """Configuration for running Leiden at multiple resolutions."""

    resolutions: Mapping[str, float] | None = None
    objective: str = "cpm"
    level_constraints: Sequence[tuple[int, int]] | None = None
    resolution_bounds: Tuple[float, float] = (1e-4, 1.0)
    max_iterations: int = 32
    progress: Callable[[str], None] | None = None
    leiden_iterations: int | None = None
    seed: int | None = None
    log_history: bool = False
    history_log_path: Path | None = None
    progress_log_path: Path | None = None
    log_dir: Path | None = None
    run_id: str | None = None
    postprocess: "PostprocessConfig | None" = None
    stability_metric: str | None = None
    stability_seeds: Sequence[int] | None = None

    # ── Large-scale Java backend ──────────────────────────────
    backend: str = "auto"                   # "auto" | "igraph" | "java"
    auto_backend_threshold: int = 5_000_000 # switch to java above this node count
    jar_path: Path | None = None            # path to networkanalysis JAR
    java_heap: str = "8g"                   # -Xmx for JVM


@dataclass
class ClusterTables:
    """Structured output containing membership and description tables."""

    membership: pl.DataFrame
    description: pl.DataFrame
    raw_membership: pl.DataFrame
    levels: Tuple[str, ...]
    resolutions: Mapping[str, float] | None = None
    qualities: Mapping[str, float] | None = None


@dataclass
class EnsembleConfig:
    """Configuration for running ensemble partitions across gamma values and seeds."""

    gamma_values: Sequence[float] | None = None
    gamma_min: float = 1e-4
    gamma_max: float = 0.5
    gamma_count: int = 10
    gamma_scale: str = "log"
    seeds: Sequence[int] = (0, 1, 2)
    n_iterations: int | None = None
    normalize_labels: bool = True
    parallel: bool = False
    workers: int | None = None
    start_method: str = "spawn"  # 'spawn' or 'fork'
    output_dir: Path | None = Path("ensemble_output")
    progress: Callable[[str], None] | None = None
    progress_log_path: Path | None = None
    log_dir: Path | None = None
    run_id: str | None = None
    retain_memberships: bool = False
    min_cluster_ratio: float = 0.001
    min_cluster_size: int | None = None

@dataclass
class PostprocessConfig:
    """Rules for merging tiny clusters after a Leiden run."""

    min_size: int | None = None
    min_weight: float | None = None
    min_docs: int | None = None
    strategy: str = "best_neighbor"
    max_passes: int = 1

    def resolve_thresholds(self, *, has_node_weights: bool) -> tuple[int | None, float | None]:
        """Resolve postprocess thresholds to merge_small_clusters arguments.

        `min_docs` is the canonical external threshold. For backward compatibility,
        explicit `min_size`/`min_weight` are still supported.

        Resolution policy:
        - min_docs only + no node weights: map to min_size.
        - min_docs only + node weights: map to min_weight.
        - min_docs + explicit legacy thresholds: values must match, otherwise error.
        """

        min_size = None if self.min_size is None else int(self.min_size)
        min_weight = None if self.min_weight is None else float(self.min_weight)

        if self.min_docs is None:
            return min_size, min_weight

        docs_value = float(self.min_docs)
        if not docs_value.is_integer():
            raise ValueError(f"postprocess min_docs must be an integer, got {self.min_docs!r}")
        min_docs = int(docs_value)
        if min_docs < 1:
            raise ValueError(f"postprocess min_docs must be >= 1, got {min_docs}")

        if min_size is not None and int(min_size) != min_docs:
            raise ValueError(
                "postprocess threshold conflict: "
                f"min_docs={min_docs} but min_size={min_size}"
            )
        if min_weight is not None and float(min_weight) != float(min_docs):
            raise ValueError(
                "postprocess threshold conflict: "
                f"min_docs={min_docs} but min_weight={min_weight}"
            )

        if min_size is None and min_weight is None:
            if has_node_weights:
                return None, float(min_docs)
            return int(min_docs), None
        return min_size, min_weight


@dataclass
class HierarchyLevelConfig:
    """Configuration for a single hierarchy level."""

    name: str
    resolution: float | None = None
    objective: str | None = None
    seeds: Sequence[int] = (0,)
    iterations: int | None = None
    postprocess: PostprocessConfig | None = None
    stop_if_singleton: bool = True


@dataclass
class HierarchyConfig:
    """Settings for constructing a Leiden hierarchy."""

    levels: Sequence[HierarchyLevelConfig]
    reuse_membership: bool = True
    contract_weights: str = "sum"
    contract_loops: bool = True


__all__ = [
    "LeidenConfig",
    "ClusterTables",
    "EnsembleConfig",
    "PostprocessConfig",
    "HierarchyLevelConfig",
    "HierarchyConfig",
]
