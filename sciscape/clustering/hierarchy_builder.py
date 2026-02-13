"""Utilities for constructing Leiden-based hierarchies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import igraph as ig

from .config import HierarchyConfig, HierarchyLevelConfig, PostprocessConfig
from .postprocess import PostprocessResult, merge_small_clusters
from .runner import LeidenRunResult, LeidenRunner


@dataclass(frozen=True)
class HierarchyLayer:
    """Metadata for a single level in the hierarchy."""

    name: str
    resolution: float
    seed: int | None
    quality: float
    cluster_count: int
    objective: str
    raw_result: LeidenRunResult
    postprocess: PostprocessResult | None


@dataclass(frozen=True)
class HierarchyBuildResult:
    """Final outcome of the hierarchy construction."""

    layers: List[HierarchyLayer]
    memberships_by_level: Dict[str, List[int]]


class HierarchyBuilder:
    """Construct nested Leiden partitions by contracting the graph per level."""

    def __init__(
        self,
        graph: ig.Graph,
        *,
        objective: str = "modularity",
        default_iterations: int | None = None,
        default_seed: int | None = None,
        default_postprocess: PostprocessConfig | None = None,
    ) -> None:
        self._base_runner = LeidenRunner(
            graph,
            objective=objective,
            default_iterations=default_iterations,
            default_seed=default_seed,
        )
        self._default_postprocess = default_postprocess

    def build(self, config: HierarchyConfig) -> HierarchyBuildResult:
        layers: List[HierarchyLayer] = []
        memberships_original: Dict[str, List[int]] = {}

        runner = self._base_runner
        prev_original_membership: List[int] | None = None
        prev_graph_membership: List[int] | None = None

        for level_cfg in config.levels:
            if level_cfg.resolution is None:
                raise ValueError(f"Hierarchy level '{level_cfg.name}' requires a resolution value")

            default_seed = runner.default_seed
            seeds = tuple(level_cfg.seeds) if level_cfg.seeds else ((default_seed,) if default_seed is not None else (None,))
            best_run: LeidenRunResult | None = None
            best_seed: int | None = None

            for seed in seeds:
                result = runner.run(
                    level_cfg.resolution,
                    objective=level_cfg.objective,
                    seed=seed,
                    n_iterations=level_cfg.iterations,
                    initial_membership=prev_graph_membership if config.reuse_membership else None,
                )
                if best_run is None or result.quality > best_run.quality:
                    best_run = result
                    best_seed = seed

            if best_run is None:
                raise RuntimeError(f"Failed to obtain Leiden result for hierarchy level '{level_cfg.name}'")

            post_cfg = level_cfg.postprocess or self._default_postprocess
            post_result: PostprocessResult | None = None
            graph_weights = runner.graph.vs["weight"] if "weight" in runner.graph.vs.attributes() else None
            if post_cfg is not None:
                post_result = merge_small_clusters(
                    runner.graph,
                    best_run.membership,
                    min_size=post_cfg.min_size,
                    min_weight=post_cfg.min_weight,
                    node_weights=graph_weights,
                    max_passes=max(post_cfg.max_passes, 1),
                )
                final_membership = post_result.membership
            else:
                final_membership = best_run.membership

            layer = HierarchyLayer(
                name=level_cfg.name,
                resolution=level_cfg.resolution,
                seed=best_seed,
                quality=best_run.quality,
                cluster_count=len(set(final_membership)),
                objective=level_cfg.objective or runner.objective,
                raw_result=best_run,
                postprocess=post_result,
            )
            layers.append(layer)

            if prev_original_membership is None:
                memberships_original[level_cfg.name] = list(final_membership)
            else:
                mapped = [final_membership[parent] for parent in prev_original_membership]
                memberships_original[level_cfg.name] = mapped

            prev_original_membership = memberships_original[level_cfg.name]
            prev_graph_membership = final_membership if config.reuse_membership else None

            if level_cfg.stop_if_singleton and layer.cluster_count <= 1:
                break

            if layer.cluster_count <= 1:
                continue

            contracted = runner.contract(
                final_membership,
                combine_weights=config.contract_weights,
                keep_loops=config.contract_loops,
            )
            runner = runner.clone_for_graph(contracted)
            prev_graph_membership = None

        return HierarchyBuildResult(layers=layers, memberships_by_level=memberships_original)


__all__ = ["HierarchyBuilder", "HierarchyLayer", "HierarchyBuildResult"]
