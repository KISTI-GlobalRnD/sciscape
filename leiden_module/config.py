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
    objective: str = "modularity"
    level_constraints: Sequence[tuple[int, int]] | None = None
    resolution_bounds: Tuple[float, float] = (1e-4, 50.0)
    max_iterations: int = 32
    progress: Callable[[str], None] | None = None
    leiden_iterations: int | None = None
    seed: int | None = None
    log_history: bool = False
    postprocess: "PostprocessConfig | None" = None
    stability_metric: str | None = None


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
    gamma_min: float = 1e-3
    gamma_max: float = 5.0
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
    retain_memberships: bool = False
    min_cluster_ratio: float = 0.001
    min_cluster_size: int | None = None

@dataclass
class PostprocessConfig:
    """Rules for merging tiny clusters after a Leiden run."""

    min_size: int | None = None
    min_weight: float | None = None
    strategy: str = "best_neighbor"
    max_passes: int = 1


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
