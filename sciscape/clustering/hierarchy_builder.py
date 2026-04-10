"""Utilities for constructing Leiden-based hierarchies."""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
from typing import Dict, List

import igraph as ig

from .config import HierarchyConfig, HierarchyLevelConfig, PostprocessConfig
from .postprocess import PostprocessResult, merge_small_clusters
from .runner import LeidenRunResult, LeidenRunner, RustLeidenRunner


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
    """Construct nested Leiden partitions by contracting the graph per level.

    Supports two usage patterns:

    **All-at-once** (existing API)::

        builder = HierarchyBuilder(graph)
        result = builder.build(config)

    **Level-by-level** (new — enables caching between levels)::

        builder = HierarchyBuilder(graph)
        layer0 = builder.build_level(level_cfg_0)   # nano
        # ... save / inspect / decide ...
        layer1 = builder.build_level(level_cfg_1)   # micro on contracted graph
        result = builder.result()
    """

    def __init__(
        self,
        graph: ig.Graph,
        *,
        objective: str = "cpm",
        default_iterations: int | None = None,
        default_seed: int | None = None,
        default_postprocess: PostprocessConfig | None = None,
        reuse_membership: bool = True,
        contract_weights: str = "sum",
        contract_loops: bool = True,
    ) -> None:
        self._base_runner = LeidenRunner(
            graph,
            objective=objective,
            default_iterations=default_iterations,
            default_seed=default_seed,
        )
        self._default_postprocess = default_postprocess
        self._reuse_membership = reuse_membership
        self._contract_weights = contract_weights
        self._contract_loops = contract_loops

        # Incremental state
        self._runner: LeidenRunner = self._base_runner
        self._layers: List[HierarchyLayer] = []
        self._memberships_original: Dict[str, List[int]] = {}
        self._prev_original_membership: List[int] | None = None
        self._prev_graph_membership: List[int] | None = None
        self._node_sizes: List[int] | None = None  # per-supernode original node counts
        self._stopped: bool = False

    # ------------------------------------------------------------------
    # Level-by-level API
    # ------------------------------------------------------------------
    def build_level(self, level_cfg: HierarchyLevelConfig) -> HierarchyLayer:
        """Run a single hierarchy level and contract the graph for the next.

        Returns the :class:`HierarchyLayer` for this level.  Call repeatedly
        for each level in order (finest → coarsest).  After the last level,
        call :meth:`result` to get the full :class:`HierarchyBuildResult`.
        """
        if self._stopped:
            raise RuntimeError(
                "Hierarchy building stopped (singleton reached). "
                "Call result() to retrieve current state."
            )

        if level_cfg.resolution is None:
            raise ValueError(
                f"Hierarchy level '{level_cfg.name}' requires a resolution value"
            )

        runner = self._runner

        # Run Leiden (possibly multiple seeds, keep best)
        default_seed = runner.default_seed
        seeds = (
            tuple(level_cfg.seeds) if level_cfg.seeds
            else ((default_seed,) if default_seed is not None else (None,))
        )
        best_run: LeidenRunResult | None = None
        best_seed: int | None = None

        for seed in seeds:
            result = runner.run(
                level_cfg.resolution,
                objective=level_cfg.objective,
                seed=seed,
                n_iterations=level_cfg.iterations,
                initial_membership=(
                    self._prev_graph_membership
                    if self._reuse_membership else None
                ),
                node_sizes=self._node_sizes,
            )
            if best_run is None or result.quality > best_run.quality:
                best_run = result
                best_seed = seed

        assert best_run is not None

        # Post-process (merge small clusters)
        post_cfg = level_cfg.postprocess or self._default_postprocess
        post_result: PostprocessResult | None = None

        if post_cfg is not None and isinstance(runner, RustLeidenRunner):
            # Rust path: use Rust postprocess with weighted thresholds
            from .leiden_rust import postprocess_small_clusters_rust
            import numpy as np
            has_nw = runner._node_weights is not None
            min_size, min_weight = post_cfg.resolve_thresholds(
                has_node_weights=has_nw
            )
            do_post = (
                (min_weight is not None and min_weight > 0)
                or (min_size is not None and min_size > 1)
            )
            if do_post:
                mem = np.asarray(best_run.membership, dtype=np.uint64)
                rust_post = postprocess_small_clusters_rust(
                    resolution=level_cfg.resolution,
                    min_size=int(min_size or 0),
                    min_weight=float(min_weight or 0.0),
                    membership=mem,
                    edges_src=runner._src,
                    edges_dst=runner._dst,
                    edges_weight=runner._weight,
                    node_weights=runner._node_weights,
                    n_nodes=runner.n_nodes,
                    seed=best_seed or 0,
                )
                final_membership = rust_post.membership.tolist()
            else:
                final_membership = best_run.membership
        elif post_cfg is not None:
            # igraph path
            graph_weights = (
                runner.graph.vs["weight"]
                if "weight" in runner.graph.vs.attributes() else None
            )
            min_size, min_weight = post_cfg.resolve_thresholds(
                has_node_weights=graph_weights is not None
            )
            post_result = merge_small_clusters(
                runner.graph,
                best_run.membership,
                min_size=min_size,
                min_weight=min_weight,
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
        self._layers.append(layer)

        # Map membership back to original node indices
        if self._prev_original_membership is None:
            self._memberships_original[level_cfg.name] = list(final_membership)
        else:
            mapped = [
                final_membership[parent]
                for parent in self._prev_original_membership
            ]
            self._memberships_original[level_cfg.name] = mapped

        self._prev_original_membership = self._memberships_original[level_cfg.name]
        self._prev_graph_membership = (
            final_membership if self._reuse_membership else None
        )

        # Contract graph for next level (or stop)
        if level_cfg.stop_if_singleton and layer.cluster_count <= 1:
            self._stopped = True
        elif layer.cluster_count > 1:
            if isinstance(runner, RustLeidenRunner):
                # Rust runner: contract returns a new runner directly
                self._runner = runner.contract(final_membership)
            else:
                contracted = runner.contract(
                    final_membership,
                    combine_weights=self._contract_weights,
                    keep_loops=self._contract_loops,
                )
                self._runner = runner.clone_for_graph(contracted)
            self._prev_graph_membership = None

            # Compute node_sizes for next level: each super-node
            # represents the total original nodes it contains.
            cluster_counts = Counter(final_membership)
            if self._node_sizes is not None:
                # Already contracted: accumulate original sizes
                agg: Dict[int, int] = {}
                for v, cid in enumerate(final_membership):
                    agg[cid] = agg.get(cid, 0) + self._node_sizes[v]
                self._node_sizes = [agg[cid] for cid in range(layer.cluster_count)]
            else:
                # First contraction: sizes = cluster membership counts
                self._node_sizes = [cluster_counts[cid] for cid in range(layer.cluster_count)]

        return layer

    @property
    def layers(self) -> List[HierarchyLayer]:
        """Layers built so far."""
        return list(self._layers)

    @property
    def memberships(self) -> Dict[str, List[int]]:
        """Memberships mapped to original node indices, built so far."""
        return dict(self._memberships_original)

    @property
    def stopped(self) -> bool:
        """Whether building stopped due to singleton."""
        return self._stopped

    def result(self) -> HierarchyBuildResult:
        """Return the accumulated result from all :meth:`build_level` calls."""
        return HierarchyBuildResult(
            layers=list(self._layers),
            memberships_by_level=dict(self._memberships_original),
        )

    # ------------------------------------------------------------------
    # All-at-once API (preserved for backward compatibility)
    # ------------------------------------------------------------------
    def build(self, config: HierarchyConfig) -> HierarchyBuildResult:
        """Run all hierarchy levels in one call.

        This is equivalent to calling :meth:`build_level` for each level
        in ``config.levels``, then :meth:`result`.
        """
        # Apply config-level settings
        self._reuse_membership = config.reuse_membership
        self._contract_weights = config.contract_weights
        self._contract_loops = config.contract_loops

        for level_cfg in config.levels:
            self.build_level(level_cfg)
            if self._stopped:
                break

        return self.result()


__all__ = ["HierarchyBuilder", "HierarchyLayer", "HierarchyBuildResult"]
