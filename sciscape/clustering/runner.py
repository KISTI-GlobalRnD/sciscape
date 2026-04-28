"""Reusable Leiden runner with graph contraction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import igraph as ig
import leidenalg as la

from .partitioning import partition_class


def _rust_iteration_count(n_iterations: int | None) -> int:
    if n_iterations is None:
        return 10
    if n_iterations < 0:
        return 0
    return int(n_iterations)


@dataclass(frozen=True)
class LeidenRunResult:
    """Outcome of a single Leiden optimisation."""

    resolution: float
    seed: int | None
    membership: Sequence[int] | np.ndarray
    partition: la.VertexPartition | None  # None for Rust backend
    quality: float
    n_clusters: int | None = None

    @property
    def cluster_count(self) -> int:
        if self.n_clusters is not None:
            return int(self.n_clusters)
        if self.partition is not None:
            return len(self.partition)
        if isinstance(self.membership, np.ndarray):
            return int(np.unique(self.membership).size)
        return len(set(self.membership))


class LeidenRunner:
    """Wrapper around `leidenalg` with optimiser reuse across runs."""

    def __init__(
        self,
        graph: ig.Graph,
        *,
        objective: str = "cpm",
        default_iterations: int | None = None,
        default_seed: int | None = None,
    ) -> None:
        self._graph = graph
        self._objective = objective
        self._weights = graph.es["weight"] if "weight" in graph.es.attributes() else None
        self._default_iterations = default_iterations
        self._default_seed = default_seed
        self._optimiser = la.Optimiser()

    @property
    def graph(self) -> ig.Graph:
        return self._graph

    @property
    def objective(self) -> str:
        return self._objective

    @property
    def default_iterations(self) -> int | None:
        return self._default_iterations

    @property
    def default_seed(self) -> int | None:
        return self._default_seed

    def run(
        self,
        resolution: float,
        *,
        objective: str | None = None,
        seed: int | None = None,
        n_iterations: int | None = None,
        initial_membership: Sequence[int] | None = None,
        node_sizes: Sequence[int] | None = None,
    ) -> LeidenRunResult:
        """Execute a Leiden optimisation and return the membership vector.

        Parameters
        ----------
        node_sizes : sequence of int, optional
            Per-vertex sizes, used by CPM on contracted graphs so the
            resolution term scales with the number of original nodes
            each supernode represents.
        """

        seed = self._default_seed if seed is None else seed
        n_iterations = self._default_iterations if n_iterations is None else n_iterations
        objective = self._objective if objective is None else objective

        partition_cls = partition_class(objective)
        kwargs: dict = dict(
            weights=self._weights,
            resolution_parameter=resolution,
        )
        if initial_membership is not None:
            kwargs["initial_membership"] = list(initial_membership)
        if node_sizes is not None:
            kwargs["node_sizes"] = list(node_sizes)
        partition = partition_cls(self._graph, **kwargs)

        if seed is not None:
            self._optimiser.set_rng_seed(int(seed))

        # leidenalg expects an int; treat None as "use library default" (-1).
        if n_iterations is None:
            n_iterations = -1
        self._optimiser.optimise_partition(partition, n_iterations=n_iterations)

        membership = list(partition.membership)

        return LeidenRunResult(
            resolution=resolution,
            seed=seed,
            membership=membership,
            partition=partition,
            quality=float(partition.quality()),
        )

    def contract(self, membership: Sequence[int], *, combine_weights: str = "sum", keep_loops: bool = True) -> ig.Graph:
        """Return a reduced graph where each community becomes a supernode."""

        contracted = self._graph.copy()
        mapping = list(membership)
        combine_attrs = {"uid": "first"}
        if "weight" in self._graph.es.attributes():
            combine_attrs["weight"] = "sum"
        contracted.contract_vertices(mapping, combine_attrs=combine_attrs)
        contracted.simplify(
            combine_edges={"weight": combine_weights},
            multiple=True,
            loops=keep_loops,
        )
        return contracted

    def clone_for_graph(self, graph: ig.Graph) -> "LeidenRunner":
        """Create a new runner sharing defaults but with a different graph."""

        return LeidenRunner(
            graph,
            objective=self._objective,
            default_iterations=self._default_iterations,
            default_seed=self._default_seed,
        )


class RustLeidenRunner:
    """Leiden runner using the Rust backend (same interface as LeidenRunner).

    Operates on numpy edge arrays instead of igraph objects.
    """

    def __init__(
        self,
        edges_src: np.ndarray | None,
        edges_dst: np.ndarray | None,
        edges_weight: np.ndarray | None,
        n_nodes: int,
        *,
        objective: str = "cpm",
        default_iterations: int | None = None,
        default_seed: int | None = None,
        node_weights: np.ndarray | None = None,
        edge_path: str | Path | None = None,
    ) -> None:
        from .leiden_rust import _check_available, build_leiden_graph
        _check_available()

        if (edges_src is None) != (edges_dst is None) or (edges_src is None) != (edges_weight is None):
            raise ValueError("edges_src, edges_dst, and edges_weight must be provided together")
        self._src = None if edges_src is None else np.ascontiguousarray(edges_src, dtype=np.uint32)
        self._dst = None if edges_dst is None else np.ascontiguousarray(edges_dst, dtype=np.uint32)
        self._weight = None if edges_weight is None else np.ascontiguousarray(edges_weight, dtype=np.float64)
        self._n_nodes = n_nodes
        self._objective = objective
        self._default_iterations = default_iterations
        self._default_seed = default_seed
        self._node_weights = None if node_weights is None else np.asarray(node_weights, dtype=np.float64)
        self._edge_path = None if edge_path is None else Path(edge_path)
        self._graph = build_leiden_graph(
            edge_path=self._edge_path,
            edges_src=self._src,
            edges_dst=self._dst,
            edges_weight=self._weight,
            n_nodes=self._n_nodes,
            node_weights=self._node_weights,
        )

    @classmethod
    def from_edge_path(
        cls,
        edge_path: str | Path,
        n_nodes: int,
        *,
        objective: str = "cpm",
        default_iterations: int | None = None,
        default_seed: int | None = None,
        node_weights: np.ndarray | None = None,
    ) -> "RustLeidenRunner":
        return cls(
            None,
            None,
            None,
            n_nodes,
            objective=objective,
            default_iterations=default_iterations,
            default_seed=default_seed,
            node_weights=node_weights,
            edge_path=edge_path,
        )

    @classmethod
    def from_graph(
        cls,
        graph,
        *,
        objective: str = "cpm",
        default_iterations: int | None = None,
        default_seed: int | None = None,
    ) -> "RustLeidenRunner":
        self = cls.__new__(cls)
        self._src = None
        self._dst = None
        self._weight = None
        self._n_nodes = graph.n_nodes
        self._objective = objective
        self._default_iterations = default_iterations
        self._default_seed = default_seed
        self._node_weights = graph.node_weights
        self._edge_path = None
        self._graph = graph
        return self

    @property
    def objective(self) -> str:
        return self._objective

    @property
    def default_iterations(self) -> int | None:
        return self._default_iterations

    @property
    def default_seed(self) -> int | None:
        return self._default_seed

    @property
    def n_nodes(self) -> int:
        return self._n_nodes

    @property
    def node_weights(self) -> np.ndarray | None:
        return self._node_weights

    @property
    def has_node_weights(self) -> bool:
        return self._node_weights is not None

    def run(
        self,
        resolution: float,
        *,
        objective: str | None = None,
        seed: int | None = None,
        n_iterations: int | None = None,
        initial_membership: Sequence[int] | None = None,
        node_sizes: Sequence[int] | None = None,
    ) -> LeidenRunResult:
        from .leiden_rust import run_leiden_rust

        seed = self._default_seed if seed is None else seed
        n_iterations = self._default_iterations if n_iterations is None else n_iterations
        rust_iterations = _rust_iteration_count(n_iterations)

        # node_sizes from caller (hierarchy contraction) → node_weights for Rust
        nw = self._node_weights
        if node_sizes is not None:
            nw = np.asarray(node_sizes, dtype=np.float64)

        init_mem = None
        if initial_membership is not None:
            init_mem = np.asarray(initial_membership, dtype=np.uint64)

        if (
            self._graph is not None
            and (
                nw is self._node_weights
                or (
                    nw is not None
                    and self._node_weights is not None
                    and np.array_equal(nw, self._node_weights)
                )
            )
        ):
            result = self._graph.run_leiden(
                resolution=resolution,
                seed=seed or 0,
                n_iterations=rust_iterations,
                initial_membership=init_mem,
            )
        else:
            if self._src is None or self._dst is None or self._weight is None:
                raise RuntimeError("cannot rebuild weighted Rust graph without materialized edge arrays")
            result = run_leiden_rust(
                edges_src=self._src,
                edges_dst=self._dst,
                edges_weight=self._weight,
                resolution=resolution,
                n_nodes=self._n_nodes,
                seed=seed or 0,
                n_iterations=rust_iterations,
                initial_membership=init_mem,
                node_weights=nw,
            )

        return LeidenRunResult(
            resolution=resolution,
            seed=seed,
            membership=result.membership,
            partition=None,
            quality=result.quality,
            n_clusters=result.n_clusters,
        )

    def postprocess(
        self,
        *,
        resolution: float,
        membership: Sequence[int] | np.ndarray,
        min_size: int = 0,
        min_weight: float = 0.0,
        seed: int | None = None,
        n_iterations: int | None = None,
        randomness: float = 0.01,
        max_rounds: int = 5,
        gamma_decay: float = 0.5,
        use_greedy: bool = True,
        greedy_anchor_only: bool = False,
        greedy_fallback_to_small: bool = False,
        greedy_max_weight: float = 0.0,
        use_component_merge: bool = True,
        component_max_weight: float = 0.0,
    ):
        from .leiden_rust import postprocess_small_clusters_rust

        seed = self._default_seed if seed is None else seed
        n_iterations = self._default_iterations if n_iterations is None else n_iterations
        rust_iterations = _rust_iteration_count(n_iterations)
        mem = np.asarray(membership, dtype=np.uint64)

        if self._graph is not None:
            return self._graph.postprocess_small_clusters(
                resolution=resolution,
                min_size=min_size,
                min_weight=min_weight,
                membership=mem,
                seed=seed or 0,
                n_iterations=rust_iterations,
                randomness=randomness,
                max_rounds=max_rounds,
                gamma_decay=gamma_decay,
                use_greedy=use_greedy,
                greedy_anchor_only=greedy_anchor_only,
                greedy_fallback_to_small=greedy_fallback_to_small,
                greedy_max_weight=greedy_max_weight,
                use_component_merge=use_component_merge,
                component_max_weight=component_max_weight,
            )

        if self._src is None or self._dst is None or self._weight is None:
            raise RuntimeError("cannot postprocess Rust runner without graph handle or edge arrays")
        return postprocess_small_clusters_rust(
            edges_src=self._src,
            edges_dst=self._dst,
            edges_weight=self._weight,
            n_nodes=self._n_nodes,
            node_weights=self._node_weights,
            resolution=resolution,
            min_size=min_size,
            min_weight=min_weight,
            membership=mem,
            seed=seed or 0,
            n_iterations=rust_iterations,
            randomness=randomness,
            max_rounds=max_rounds,
            gamma_decay=gamma_decay,
            use_greedy=use_greedy,
            greedy_anchor_only=greedy_anchor_only,
            greedy_fallback_to_small=greedy_fallback_to_small,
            greedy_max_weight=greedy_max_weight,
            use_component_merge=use_component_merge,
            component_max_weight=component_max_weight,
        )

    def search_resolution(
        self,
        *,
        min_clusters: int,
        max_clusters: int,
        bounds: tuple[float, float],
        max_iterations: int,
        seed: int | None = None,
        n_iterations: int | None = None,
        randomness: float = 0.01,
    ):
        """Search for a gamma on the Rust graph without returning probe memberships."""
        if self._graph is None:
            raise RuntimeError("cannot search Rust resolution without graph handle")
        seed = self._default_seed if seed is None else seed
        n_iterations = self._default_iterations if n_iterations is None else n_iterations
        return self._graph.search_resolution(
            min_clusters=min_clusters,
            max_clusters=max_clusters,
            lower_bound=float(bounds[0]),
            upper_bound=float(bounds[1]),
            max_iterations=max_iterations,
            n_iterations=_rust_iteration_count(n_iterations),
            randomness=randomness,
            seed=seed or 0,
        )

    def contract(
        self,
        membership: Sequence[int],
        *,
        combine_weights: str = "sum",
        keep_loops: bool = True,
    ) -> "RustLeidenRunner":
        """Contract the graph and return a new runner for the contracted graph."""
        from .pipeline import _contract_edges

        mem = np.asarray(membership, dtype=np.uint64)
        if self._graph is not None:
            return RustLeidenRunner.from_graph(
                self._graph.contract(mem, keep_self_loops=keep_loops),
                objective=self._objective,
                default_iterations=self._default_iterations,
                default_seed=self._default_seed,
            )
        if self._src is None or self._dst is None or self._weight is None:
            raise RuntimeError("cannot contract Rust runner without graph handle or edge arrays")
        new_src, new_dst, new_weight, new_n, new_node_sizes = _contract_edges(
            self._src, self._dst, self._weight, mem, self._node_weights,
        )
        return RustLeidenRunner(
            new_src, new_dst, new_weight, new_n,
            objective=self._objective,
            default_iterations=self._default_iterations,
            default_seed=self._default_seed,
            node_weights=new_node_sizes.astype(np.float64),
        )

    def clone_for_graph(self, contracted_runner: "RustLeidenRunner") -> "RustLeidenRunner":
        """Alias for compatibility — returns the already-contracted runner."""
        return contracted_runner


def ensure_membership_sequence(membership: Sequence[int] | Iterable[int]) -> List[int]:
    """Convert an arbitrary iterable of labels into a list (avoids igraph slicing)."""

    return list(membership if isinstance(membership, Sequence) else list(membership))


__all__ = [
    "LeidenRunner",
    "RustLeidenRunner",
    "LeidenRunResult",
    "ensure_membership_sequence",
]
