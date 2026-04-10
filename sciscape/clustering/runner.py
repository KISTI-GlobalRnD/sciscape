"""Reusable Leiden runner with graph contraction helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Sequence

import numpy as np
import igraph as ig
import leidenalg as la

from .partitioning import partition_class


@dataclass(frozen=True)
class LeidenRunResult:
    """Outcome of a single Leiden optimisation."""

    resolution: float
    seed: int | None
    membership: List[int]
    partition: la.VertexPartition | None  # None for Rust backend
    quality: float

    @property
    def cluster_count(self) -> int:
        if self.partition is not None:
            return len(self.partition)
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
        edges_src: np.ndarray,
        edges_dst: np.ndarray,
        edges_weight: np.ndarray,
        n_nodes: int,
        *,
        objective: str = "cpm",
        default_iterations: int | None = None,
        default_seed: int | None = None,
        node_weights: np.ndarray | None = None,
    ) -> None:
        from .leiden_rust import _check_available
        _check_available()

        self._src = np.ascontiguousarray(edges_src, dtype=np.uint32)
        self._dst = np.ascontiguousarray(edges_dst, dtype=np.uint32)
        self._weight = np.ascontiguousarray(edges_weight, dtype=np.float64)
        self._n_nodes = n_nodes
        self._objective = objective
        self._default_iterations = default_iterations
        self._default_seed = default_seed
        self._node_weights = node_weights

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

        # node_sizes from caller (hierarchy contraction) → node_weights for Rust
        nw = self._node_weights
        if node_sizes is not None:
            nw = np.asarray(node_sizes, dtype=np.float64)

        init_mem = None
        if initial_membership is not None:
            init_mem = np.asarray(initial_membership, dtype=np.uint64)

        result = run_leiden_rust(
            edges_src=self._src,
            edges_dst=self._dst,
            edges_weight=self._weight,
            resolution=resolution,
            n_nodes=self._n_nodes,
            seed=seed or 0,
            n_iterations=n_iterations or 10,
            initial_membership=init_mem,
            node_weights=nw,
        )

        return LeidenRunResult(
            resolution=resolution,
            seed=seed,
            membership=result.membership.tolist(),
            partition=None,
            quality=result.quality,
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
