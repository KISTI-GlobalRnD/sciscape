"""Reusable Leiden runner with graph contraction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import igraph as ig
import leidenalg as la

from .partitioning import partition_class


@dataclass(frozen=True)
class LeidenRunResult:
    """Outcome of a single Leiden optimisation."""

    resolution: float
    seed: int | None
    membership: List[int]
    partition: la.VertexPartition
    quality: float

    @property
    def cluster_count(self) -> int:
        return len(self.partition)


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


def ensure_membership_sequence(membership: Sequence[int] | Iterable[int]) -> List[int]:
    """Convert an arbitrary iterable of labels into a list (avoids igraph slicing)."""

    return list(membership if isinstance(membership, Sequence) else list(membership))


__all__ = ["LeidenRunner", "LeidenRunResult", "ensure_membership_sequence"]
