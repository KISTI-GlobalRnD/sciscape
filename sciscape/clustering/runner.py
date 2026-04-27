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
        edges_src: np.ndarray | None,
        edges_dst: np.ndarray | None,
        edges_weight: np.ndarray | None,
        n_nodes: int,
        *,
        objective: str = "cpm",
        default_iterations: int | None = None,
        default_seed: int | None = None,
        node_weights: np.ndarray | None = None,
        edge_path: str | None = None,
    ) -> None:
        from .leiden_rust import _check_available, load_graph_rust
        _check_available()

        self._src = None if edges_src is None else np.ascontiguousarray(edges_src, dtype=np.uint32)
        self._dst = None if edges_dst is None else np.ascontiguousarray(edges_dst, dtype=np.uint32)
        self._weight = None if edges_weight is None else np.ascontiguousarray(edges_weight, dtype=np.float64)
        self._n_nodes = n_nodes
        self._objective = objective
        self._default_iterations = default_iterations
        self._default_seed = default_seed
        self._node_weights = node_weights
        self._has_node_weights = node_weights is not None
        self._edge_path = edge_path
        self._handle = load_graph_rust(
            n_nodes=n_nodes,
            edge_path=edge_path,
            edges_src=self._src,
            edges_dst=self._dst,
            edges_weight=self._weight,
            node_weights=node_weights,
        )

    @classmethod
    def from_handle(
        cls,
        handle,
        *,
        objective: str = "cpm",
        default_iterations: int | None = None,
        default_seed: int | None = None,
        node_weights: np.ndarray | None = None,
        has_node_weights: bool | None = None,
    ) -> "RustLeidenRunner":
        self = cls.__new__(cls)
        self._src = None
        self._dst = None
        self._weight = None
        self._n_nodes = handle.n_nodes
        self._objective = objective
        self._default_iterations = default_iterations
        self._default_seed = default_seed
        self._node_weights = node_weights
        self._has_node_weights = (node_weights is not None) if has_node_weights is None else has_node_weights
        self._edge_path = None
        self._handle = handle
        return self

    @classmethod
    def from_edge_path(
        cls,
        edge_path: str,
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
        return self._has_node_weights

    def run(
        self,
        resolution: float,
        *,
        objective: str | None = None,
        seed: int | None = None,
        n_iterations: int | None = None,
        initial_membership: Sequence[int] | None = None,
        initial_membership_path: str | None = None,
        node_sizes: Sequence[int] | None = None,
    ) -> LeidenRunResult:
        result = self.run_array(
            resolution,
            objective=objective,
            seed=seed,
            n_iterations=n_iterations,
            initial_membership=initial_membership,
            initial_membership_path=initial_membership_path,
            node_sizes=node_sizes,
        )
        return LeidenRunResult(
            resolution=resolution,
            seed=seed,
            membership=result.membership.tolist(),
            partition=None,
            quality=result.quality,
        )

    def run_array(
        self,
        resolution: float,
        *,
        objective: str | None = None,
        seed: int | None = None,
        n_iterations: int | None = None,
        initial_membership: Sequence[int] | None = None,
        initial_membership_path: str | None = None,
        node_sizes: Sequence[int] | None = None,
    ):
        """Execute Rust Leiden and keep membership as a numpy array."""
        from .leiden_rust import _load_membership_path, run_leiden_rust, run_leiden_rust_handle

        seed = self._default_seed if seed is None else seed
        n_iterations = self._default_iterations if n_iterations is None else n_iterations
        # Rust uses 0 for "until convergence" (leidenalg uses -1)
        if n_iterations is not None and n_iterations < 0:
            n_iterations = 0

        init_mem = None
        init_mem_path = None if initial_membership_path is None else str(initial_membership_path)
        if initial_membership is not None:
            init_mem = np.asarray(initial_membership, dtype=np.uint64)
            init_mem_path = None

        if node_sizes is not None:
            requested = np.asarray(node_sizes, dtype=np.float64)
            existing = None if self._node_weights is None else np.asarray(self._node_weights, dtype=np.float64)
            if existing is None:
                if self._has_node_weights:
                    raise RuntimeError(
                        "RustLeidenRunner handle already owns contracted node weights; "
                        "omit node_sizes for handle-backed runs"
                    )
            if existing is None or not np.array_equal(requested, existing):
                if self._src is None or self._dst is None or self._weight is None:
                    raise RuntimeError(
                        "RustLeidenRunner loaded from edge_path cannot rebuild weighted graph "
                        "without materialized edge arrays"
                    )
                if init_mem is None and init_mem_path is not None:
                    init_mem = _load_membership_path(init_mem_path)
                    init_mem_path = None
                result = run_leiden_rust(
                    edges_src=self._src,
                    edges_dst=self._dst,
                    edges_weight=self._weight,
                    resolution=resolution,
                    n_nodes=self._n_nodes,
                    seed=seed or 0,
                    n_iterations=n_iterations or 10,
                    initial_membership=init_mem,
                    node_weights=requested,
                )
                return result

        return run_leiden_rust_handle(
            self._handle,
            resolution=resolution,
            seed=seed or 0,
            n_iterations=n_iterations or 10,
            n_starts=1,
            initial_membership=init_mem,
            initial_membership_path=init_mem_path,
        )

    def postprocess(
        self,
        *,
        resolution: float,
        membership: Sequence[int] | None = None,
        membership_path: str | None = None,
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
        track_changed_rounds: bool = False,
    ):
        from .leiden_rust import postprocess_small_clusters_rust_handle

        seed = self._default_seed if seed is None else seed
        n_iterations = self._default_iterations if n_iterations is None else n_iterations
        mem = None if membership is None else np.asarray(membership, dtype=np.uint64)
        return postprocess_small_clusters_rust_handle(
            handle=self._handle,
            resolution=resolution,
            min_size=min_size,
            min_weight=min_weight,
            membership=mem,
            membership_path=membership_path,
            seed=seed or 0,
            n_iterations=n_iterations or 10,
            randomness=randomness,
            max_rounds=max_rounds,
            gamma_decay=gamma_decay,
            use_greedy=use_greedy,
            greedy_anchor_only=greedy_anchor_only,
            greedy_fallback_to_small=greedy_fallback_to_small,
            greedy_max_weight=greedy_max_weight,
            use_component_merge=use_component_merge,
            component_max_weight=component_max_weight,
            track_changed_rounds=track_changed_rounds,
        )

    def contract(
        self,
        membership: Sequence[int],
        *,
        combine_weights: str = "sum",
        keep_loops: bool = True,
    ) -> "RustLeidenRunner":
        """Contract the graph and return a new runner for the contracted graph."""
        from .leiden_rust import contract_graph_rust_handle

        mem = np.asarray(membership, dtype=np.uint64)
        new_handle, new_node_weights = contract_graph_rust_handle(self._handle, mem)
        return RustLeidenRunner.from_handle(
            new_handle,
            objective=self._objective,
            default_iterations=self._default_iterations,
            default_seed=self._default_seed,
            node_weights=new_node_weights,
            has_node_weights=True,
        )

    def summarize_membership(
        self,
        membership: Sequence[int] | None = None,
        *,
        membership_path: str | None = None,
    ) -> tuple[int, int, float, float]:
        from .leiden_rust import summarize_membership_rust_handle

        mem = None if membership is None else np.asarray(membership, dtype=np.uint64)
        return summarize_membership_rust_handle(
            self._handle,
            membership=mem,
            membership_path=membership_path,
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
