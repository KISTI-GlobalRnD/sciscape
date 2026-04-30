"""Rust backend wrapper for Leiden clustering via sciscape-leiden.

Drop-in replacement for leiden_java.py functions using the Rust
native module. Much faster than Java (no JVM startup, no file I/O)
and no JDK dependency.

Requires::

    pip install sciscape-leiden

Or build from source::

    cd sciscape-leiden && maturin develop --release
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Sequence

import numpy as np
import polars as pl

from .integer_remap import ensure_int_edge_sidecars

log = logging.getLogger(__name__)

try:
    import sciscape_leiden as _rust
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False


def _check_available():
    if not RUST_AVAILABLE:
        raise ImportError(
            "sciscape-leiden Rust module not installed. "
            "Install with: pip install sciscape-leiden "
            "Or build: cd sciscape-leiden && maturin develop --release"
        )


@dataclass(frozen=True)
class RustLeidenResult:
    """Result of a Rust Leiden clustering run."""
    membership: np.ndarray
    quality: float
    n_clusters: int


@dataclass(frozen=True)
class RustPostprocessResult:
    """Result of Rust postprocessing with round-by-round monitoring."""
    membership: np.ndarray
    n_clusters: int
    changed_at_round: np.ndarray  # per-node: which round changed it (-1 = unchanged)
    rounds: list  # list of dicts with per-round info


@dataclass(frozen=True)
class RustResolutionSearchResult:
    """Result of a Rust-side resolution search."""
    resolution: float
    cluster_count: int
    quality: float
    eval_count: int
    membership: np.ndarray


@dataclass(frozen=True)
class RustClusterGraphStats:
    """Cluster-graph diagnostics for adaptive refinement dry-runs."""

    block_count: np.ndarray
    doc_weight: np.ndarray
    internal_weight: np.ndarray
    external_weight: np.ndarray
    degree: np.ndarray
    top_neighbor: np.ndarray
    top_neighbor_weight: np.ndarray
    second_neighbor: np.ndarray
    second_neighbor_weight: np.ndarray
    neighbor_weight_ratio: np.ndarray
    conductance: np.ndarray
    leafness: np.ndarray
    band_distance: np.ndarray
    candidate_source: np.ndarray
    candidate_target: np.ndarray
    candidate_edge_weight: np.ndarray
    candidate_delta_q: np.ndarray
    candidate_merged_weight: np.ndarray
    candidate_size_band_gain: np.ndarray

    @property
    def n_clusters(self) -> int:
        return int(self.block_count.shape[0])

    @property
    def n_candidates(self) -> int:
        return int(self.candidate_source.shape[0])


@dataclass(frozen=True)
class RustBoundaryMoveProbes:
    """Per-cluster dry-run probes for boundary block moves."""

    cluster: np.ndarray
    block_count: np.ndarray
    doc_weight: np.ndarray
    internal_weight: np.ndarray
    external_weight: np.ndarray
    conductance: np.ndarray
    leafness: np.ndarray
    top_neighbor: np.ndarray
    top_neighbor_weight: np.ndarray
    second_neighbor: np.ndarray
    second_neighbor_weight: np.ndarray
    neighbor_weight_ratio: np.ndarray
    positive_move_count: np.ndarray
    positive_move_weight: np.ndarray
    positive_delta_q: np.ndarray
    near_neutral_move_count: np.ndarray
    near_neutral_move_weight: np.ndarray
    near_neutral_delta_q: np.ndarray
    best_move_delta_q: np.ndarray
    best_move_node: np.ndarray
    best_move_target: np.ndarray
    top_move_count: np.ndarray
    second_move_count: np.ndarray

    @property
    def n_probes(self) -> int:
        return int(self.cluster.shape[0])


@dataclass(frozen=True)
class RustOversizeBoundaryTrimResult:
    """Applied boundary-node moves that trim clusters over a max doc weight."""

    membership: np.ndarray
    source: np.ndarray
    target: np.ndarray
    node: np.ndarray
    node_weight: np.ndarray
    delta_q: np.ndarray
    source_weight_before: np.ndarray
    source_weight_after: np.ndarray
    target_weight_before: np.ndarray
    target_weight_after: np.ndarray

    @property
    def n_moves(self) -> int:
        return int(self.node.shape[0])


@dataclass(frozen=True)
class RustBoundaryGroupProbes:
    """Per-cluster dry-run probes for grouped boundary split/move proposals."""

    cluster: np.ndarray
    block_count: np.ndarray
    doc_weight: np.ndarray
    top_neighbor: np.ndarray
    second_neighbor: np.ndarray
    top_group_count: np.ndarray
    top_group_weight: np.ndarray
    top_group_to_target_weight: np.ndarray
    top_group_cut_weight: np.ndarray
    top_group_move_delta_q: np.ndarray
    top_group_split_delta_q: np.ndarray
    top_group_is_full_cluster: np.ndarray
    second_group_count: np.ndarray
    second_group_weight: np.ndarray
    second_group_to_target_weight: np.ndarray
    second_group_cut_weight: np.ndarray
    second_group_move_delta_q: np.ndarray
    second_group_split_delta_q: np.ndarray
    second_group_is_full_cluster: np.ndarray
    best_delta_q: np.ndarray
    best_action: np.ndarray

    @property
    def n_probes(self) -> int:
        return int(self.cluster.shape[0])


@dataclass(frozen=True)
class RustExternalGrainProbes:
    """Cheap source-grain diagnostics based on strongest external attachment."""

    cluster: np.ndarray
    block_count: np.ndarray
    doc_weight: np.ndarray
    incident_directed_edges: np.ndarray
    source_directed_edges: np.ndarray
    external_directed_edges: np.ndarray
    n_external_groups: np.ndarray
    assigned_count: np.ndarray
    assigned_weight: np.ndarray
    assigned_fraction: np.ndarray
    largest_group_target: np.ndarray
    largest_group_count: np.ndarray
    largest_group_weight: np.ndarray
    largest_group_fraction: np.ndarray
    largest_group_to_target_weight: np.ndarray
    largest_group_cut_weight: np.ndarray
    largest_group_move_delta_q: np.ndarray
    largest_group_split_delta_q: np.ndarray
    second_group_target: np.ndarray
    second_group_weight: np.ndarray
    second_group_fraction: np.ndarray
    best_group_target: np.ndarray
    best_group_count: np.ndarray
    best_group_weight: np.ndarray
    best_group_fraction: np.ndarray
    best_group_to_target_weight: np.ndarray
    best_group_cut_weight: np.ndarray
    best_group_move_delta_q: np.ndarray
    best_group_split_delta_q: np.ndarray
    best_group_delta_q: np.ndarray
    best_group_action: np.ndarray
    positive_group_count: np.ndarray
    positive_group_weight: np.ndarray
    near_neutral_group_count: np.ndarray
    near_neutral_group_weight: np.ndarray
    recommended_for_split_repair: np.ndarray
    priority: np.ndarray

    @property
    def n_probes(self) -> int:
        return int(self.cluster.shape[0])


@dataclass(frozen=True)
class RustMultiCoreSplitProbes:
    """Per-cluster high-gamma induced split probes for multi-core diagnostics."""

    cluster: np.ndarray
    gamma_multiplier: np.ndarray
    probe_resolution: np.ndarray
    block_count: np.ndarray
    doc_weight: np.ndarray
    internal_weight: np.ndarray
    induced_directed_edges: np.ndarray
    n_parts: np.ndarray
    non_singleton_parts: np.ndarray
    singleton_parts: np.ndarray
    singleton_weight: np.ndarray
    core_part_count: np.ndarray
    core_part_weight: np.ndarray
    largest_part_weight: np.ndarray
    second_part_weight: np.ndarray
    largest_part_fraction: np.ndarray
    cut_weight: np.ndarray
    split_delta_q_base: np.ndarray
    split_delta_q_probe: np.ndarray
    hysteresis_only: np.ndarray

    @property
    def n_probes(self) -> int:
        return int(self.cluster.shape[0])


@dataclass(frozen=True)
class RustSplitMergeRepairProbes:
    """Dry-run forced split followed by baseline-gamma local merge repair."""

    cluster: np.ndarray
    gamma_multiplier: np.ndarray
    probe_resolution: np.ndarray
    block_count: np.ndarray
    doc_weight: np.ndarray
    induced_directed_edges: np.ndarray
    n_parts: np.ndarray
    core_part_count: np.ndarray
    singleton_weight: np.ndarray
    cut_weight: np.ndarray
    split_delta_q_base: np.ndarray
    split_delta_q_probe: np.ndarray
    repair_quotient_edges: np.ndarray
    repair_merge_count: np.ndarray
    repair_delta_q: np.ndarray
    net_delta_q: np.ndarray
    final_source_units: np.ndarray
    retained_source_units: np.ndarray
    escaped_source_units: np.ndarray
    escaped_source_weight: np.ndarray
    final_small_source_units: np.ndarray
    final_small_source_weight: np.ndarray
    largest_source_unit_fraction: np.ndarray
    restored_source_cluster: np.ndarray

    @property
    def n_probes(self) -> int:
        return int(self.cluster.shape[0])


@dataclass(frozen=True)
class RustSplitRepairApplyResult:
    """Proposed membership after applying selected split-repair candidates."""

    membership: np.ndarray
    selected_index: np.ndarray
    cluster: np.ndarray
    gamma_multiplier: np.ndarray
    probe_resolution: np.ndarray
    block_count: np.ndarray
    doc_weight: np.ndarray
    n_parts: np.ndarray
    split_delta_q_base: np.ndarray
    repair_delta_q: np.ndarray
    predicted_net_delta_q: np.ndarray
    repair_merge_count: np.ndarray
    final_source_units: np.ndarray
    retained_source_units: np.ndarray
    escaped_source_units: np.ndarray
    escaped_source_weight: np.ndarray
    final_small_source_units: np.ndarray
    final_small_source_weight: np.ndarray
    largest_source_unit_fraction: np.ndarray
    changed_nodes: np.ndarray
    moved_to_existing_cluster_nodes: np.ndarray
    moved_to_new_cluster_nodes: np.ndarray
    new_retained_clusters: np.ndarray

    @property
    def n_applied(self) -> int:
        return int(self.cluster.shape[0])


@dataclass(frozen=True)
class RustLeidenGraph:
    """Reusable Rust CSR graph for repeated Leiden/postprocess calls."""

    graph: object
    n_nodes: int
    n_edges: int
    node_weights: np.ndarray | None = None

    def run_leiden(
        self,
        *,
        resolution: float,
        seed: int = 0,
        n_iterations: int = 10,
        n_starts: int = 1,
        randomness: float = 0.01,
        randomness_schedule: Sequence[float] | None = None,
        initial_membership: np.ndarray | None = None,
        fixed_nodes: np.ndarray | None = None,
    ) -> RustLeidenResult:
        schedule = (
            None
            if randomness_schedule is None
            else [float(x) for x in randomness_schedule]
        )
        membership, quality, n_clusters = self.graph.run_leiden(
            resolution=resolution,
            n_iterations=n_iterations,
            n_starts=n_starts,
            randomness=randomness,
            randomness_schedule=schedule,
            seed=seed,
            initial_membership=initial_membership,
            fixed_nodes=fixed_nodes,
        )
        return RustLeidenResult(
            membership=membership,
            quality=quality,
            n_clusters=n_clusters,
        )

    def search_resolution(
        self,
        *,
        min_clusters: int,
        max_clusters: int,
        lower_bound: float,
        upper_bound: float,
        max_iterations: int = 32,
        n_iterations: int = 10,
        randomness: float = 0.01,
        seed: int = 0,
    ) -> RustResolutionSearchResult:
        search = getattr(self.graph, "search_resolution", None)
        if search is None:
            raise AttributeError("installed sciscape_leiden module does not expose Graph.search_resolution")
        resolution, cluster_count, quality, eval_count, membership = search(
            min_clusters=min_clusters,
            max_clusters=max_clusters,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            max_iterations=max_iterations,
            n_iterations=n_iterations,
            randomness=randomness,
            seed=seed,
        )
        return RustResolutionSearchResult(
            resolution=float(resolution),
            cluster_count=int(cluster_count),
            quality=float(quality),
            eval_count=int(eval_count),
            membership=np.ascontiguousarray(membership, dtype=np.uint64),
        )

    def cpm_quality(
        self,
        membership: np.ndarray,
        *,
        resolution: float,
    ) -> float:
        quality = getattr(self.graph, "cpm_quality", None)
        if quality is None:
            raise AttributeError("installed sciscape_leiden module does not expose Graph.cpm_quality")
        membership = np.ascontiguousarray(membership, dtype=np.uint64)
        return float(quality(membership=membership, resolution=resolution))

    def cluster_graph_stats(
        self,
        membership: np.ndarray,
        *,
        resolution: float,
        min_weight: float = 0.0,
        max_weight: float = 0.0,
        top_k: int = 1000,
    ) -> RustClusterGraphStats:
        """Build cluster-level diagnostics and macro-merge dry-run candidates.

        This is an observational helper for SciSci adaptive refinement. It
        contracts the graph by ``membership`` once, reports per-cluster edge and
        size statistics, and returns the top ``top_k`` inter-cluster merge
        candidates ranked by predicted CPM delta.
        """
        stats = getattr(self.graph, "cluster_graph_stats", None)
        if stats is None:
            raise AttributeError(
                "installed sciscape_leiden module does not expose Graph.cluster_graph_stats"
            )
        membership = np.ascontiguousarray(membership, dtype=np.uint64)
        raw = stats(
            membership=membership,
            resolution=float(resolution),
            min_weight=float(min_weight),
            max_weight=float(max_weight),
            top_k=int(top_k),
        )
        return RustClusterGraphStats(
            block_count=np.asarray(raw["block_count"], dtype=np.uint64),
            doc_weight=np.asarray(raw["doc_weight"], dtype=np.float64),
            internal_weight=np.asarray(raw["internal_weight"], dtype=np.float64),
            external_weight=np.asarray(raw["external_weight"], dtype=np.float64),
            degree=np.asarray(raw["degree"], dtype=np.uint64),
            top_neighbor=np.asarray(raw["top_neighbor"], dtype=np.int64),
            top_neighbor_weight=np.asarray(raw["top_neighbor_weight"], dtype=np.float64),
            second_neighbor=np.asarray(
                raw.get("second_neighbor", np.full_like(raw["top_neighbor"], -1)),
                dtype=np.int64,
            ),
            second_neighbor_weight=np.asarray(
                raw.get("second_neighbor_weight", np.zeros_like(raw["top_neighbor_weight"])),
                dtype=np.float64,
            ),
            neighbor_weight_ratio=np.asarray(
                raw.get("neighbor_weight_ratio", np.zeros_like(raw["top_neighbor_weight"])),
                dtype=np.float64,
            ),
            conductance=np.asarray(raw["conductance"], dtype=np.float64),
            leafness=np.asarray(raw["leafness"], dtype=np.float64),
            band_distance=np.asarray(raw["band_distance"], dtype=np.float64),
            candidate_source=np.asarray(raw["candidate_source"], dtype=np.uint64),
            candidate_target=np.asarray(raw["candidate_target"], dtype=np.uint64),
            candidate_edge_weight=np.asarray(raw["candidate_edge_weight"], dtype=np.float64),
            candidate_delta_q=np.asarray(raw["candidate_delta_q"], dtype=np.float64),
            candidate_merged_weight=np.asarray(raw["candidate_merged_weight"], dtype=np.float64),
            candidate_size_band_gain=np.asarray(raw["candidate_size_band_gain"], dtype=np.float64),
        )

    def boundary_move_probes(
        self,
        membership: np.ndarray,
        candidate_clusters: np.ndarray,
        *,
        resolution: float,
        epsilon: float = 0.0,
    ) -> RustBoundaryMoveProbes:
        """Probe block-level moves from boundary clusters to top/second neighbors.

        This is a dry-run diagnostic. It does not mutate ``membership``.
        """
        probes = getattr(self.graph, "boundary_move_probes", None)
        if probes is None:
            raise AttributeError(
                "installed sciscape_leiden module does not expose Graph.boundary_move_probes"
            )
        membership = np.ascontiguousarray(membership, dtype=np.uint64)
        candidate_clusters = np.ascontiguousarray(candidate_clusters, dtype=np.uint64)
        raw = probes(
            membership=membership,
            candidate_clusters=candidate_clusters,
            resolution=float(resolution),
            epsilon=float(epsilon),
        )
        return RustBoundaryMoveProbes(
            cluster=np.asarray(raw["cluster"], dtype=np.uint64),
            block_count=np.asarray(raw["block_count"], dtype=np.uint64),
            doc_weight=np.asarray(raw["doc_weight"], dtype=np.float64),
            internal_weight=np.asarray(raw["internal_weight"], dtype=np.float64),
            external_weight=np.asarray(raw["external_weight"], dtype=np.float64),
            conductance=np.asarray(raw["conductance"], dtype=np.float64),
            leafness=np.asarray(raw["leafness"], dtype=np.float64),
            top_neighbor=np.asarray(raw["top_neighbor"], dtype=np.int64),
            top_neighbor_weight=np.asarray(raw["top_neighbor_weight"], dtype=np.float64),
            second_neighbor=np.asarray(raw["second_neighbor"], dtype=np.int64),
            second_neighbor_weight=np.asarray(raw["second_neighbor_weight"], dtype=np.float64),
            neighbor_weight_ratio=np.asarray(raw["neighbor_weight_ratio"], dtype=np.float64),
            positive_move_count=np.asarray(raw["positive_move_count"], dtype=np.uint64),
            positive_move_weight=np.asarray(raw["positive_move_weight"], dtype=np.float64),
            positive_delta_q=np.asarray(raw["positive_delta_q"], dtype=np.float64),
            near_neutral_move_count=np.asarray(raw["near_neutral_move_count"], dtype=np.uint64),
            near_neutral_move_weight=np.asarray(raw["near_neutral_move_weight"], dtype=np.float64),
            near_neutral_delta_q=np.asarray(raw["near_neutral_delta_q"], dtype=np.float64),
            best_move_delta_q=np.asarray(raw["best_move_delta_q"], dtype=np.float64),
            best_move_node=np.asarray(raw["best_move_node"], dtype=np.uint64),
            best_move_target=np.asarray(raw["best_move_target"], dtype=np.int64),
            top_move_count=np.asarray(raw["top_move_count"], dtype=np.uint64),
            second_move_count=np.asarray(raw["second_move_count"], dtype=np.uint64),
        )

    def trim_oversize_boundary_moves(
        self,
        membership: np.ndarray,
        candidate_clusters: np.ndarray,
        *,
        resolution: float,
        target_max_weight: float,
        min_delta_q: float = 0.0,
        max_moves_per_cluster: int = 0,
    ) -> RustOversizeBoundaryTrimResult:
        """Move boundary nodes out of oversize clusters under a hard target cap."""
        trim = getattr(self.graph, "trim_oversize_boundary_moves", None)
        if trim is None:
            raise AttributeError(
                "installed sciscape_leiden module does not expose "
                "Graph.trim_oversize_boundary_moves"
            )
        membership = np.ascontiguousarray(membership, dtype=np.uint64)
        candidate_clusters = np.ascontiguousarray(candidate_clusters, dtype=np.uint64)
        raw = trim(
            membership=membership,
            candidate_clusters=candidate_clusters,
            resolution=float(resolution),
            target_max_weight=float(target_max_weight),
            min_delta_q=float(min_delta_q),
            max_moves_per_cluster=int(max_moves_per_cluster),
        )
        return RustOversizeBoundaryTrimResult(
            membership=np.asarray(raw["membership"], dtype=np.uint64),
            source=np.asarray(raw["source"], dtype=np.uint64),
            target=np.asarray(raw["target"], dtype=np.uint64),
            node=np.asarray(raw["node"], dtype=np.uint64),
            node_weight=np.asarray(raw["node_weight"], dtype=np.float64),
            delta_q=np.asarray(raw["delta_q"], dtype=np.float64),
            source_weight_before=np.asarray(
                raw["source_weight_before"], dtype=np.float64
            ),
            source_weight_after=np.asarray(raw["source_weight_after"], dtype=np.float64),
            target_weight_before=np.asarray(
                raw["target_weight_before"], dtype=np.float64
            ),
            target_weight_after=np.asarray(raw["target_weight_after"], dtype=np.float64),
        )

    def boundary_group_probes(
        self,
        membership: np.ndarray,
        candidate_clusters: np.ndarray,
        *,
        resolution: float,
    ) -> RustBoundaryGroupProbes:
        """Probe grouped split/move proposals for boundary clusters.

        This is a dry-run diagnostic. It does not mutate ``membership``.
        """
        probes = getattr(self.graph, "boundary_group_probes", None)
        if probes is None:
            raise AttributeError(
                "installed sciscape_leiden module does not expose Graph.boundary_group_probes"
            )
        membership = np.ascontiguousarray(membership, dtype=np.uint64)
        candidate_clusters = np.ascontiguousarray(candidate_clusters, dtype=np.uint64)
        raw = probes(
            membership=membership,
            candidate_clusters=candidate_clusters,
            resolution=float(resolution),
        )
        return RustBoundaryGroupProbes(
            cluster=np.asarray(raw["cluster"], dtype=np.uint64),
            block_count=np.asarray(raw["block_count"], dtype=np.uint64),
            doc_weight=np.asarray(raw["doc_weight"], dtype=np.float64),
            top_neighbor=np.asarray(raw["top_neighbor"], dtype=np.int64),
            second_neighbor=np.asarray(raw["second_neighbor"], dtype=np.int64),
            top_group_count=np.asarray(raw["top_group_count"], dtype=np.uint64),
            top_group_weight=np.asarray(raw["top_group_weight"], dtype=np.float64),
            top_group_to_target_weight=np.asarray(
                raw["top_group_to_target_weight"], dtype=np.float64
            ),
            top_group_cut_weight=np.asarray(raw["top_group_cut_weight"], dtype=np.float64),
            top_group_move_delta_q=np.asarray(
                raw["top_group_move_delta_q"], dtype=np.float64
            ),
            top_group_split_delta_q=np.asarray(
                raw["top_group_split_delta_q"], dtype=np.float64
            ),
            top_group_is_full_cluster=np.asarray(raw["top_group_is_full_cluster"], dtype=bool),
            second_group_count=np.asarray(raw["second_group_count"], dtype=np.uint64),
            second_group_weight=np.asarray(raw["second_group_weight"], dtype=np.float64),
            second_group_to_target_weight=np.asarray(
                raw["second_group_to_target_weight"], dtype=np.float64
            ),
            second_group_cut_weight=np.asarray(
                raw["second_group_cut_weight"], dtype=np.float64
            ),
            second_group_move_delta_q=np.asarray(
                raw["second_group_move_delta_q"], dtype=np.float64
            ),
            second_group_split_delta_q=np.asarray(
                raw["second_group_split_delta_q"], dtype=np.float64
            ),
            second_group_is_full_cluster=np.asarray(
                raw["second_group_is_full_cluster"], dtype=bool
            ),
            best_delta_q=np.asarray(raw["best_delta_q"], dtype=np.float64),
            best_action=np.asarray(raw["best_action"], dtype=np.uint8),
        )

    def external_grain_probes(
        self,
        membership: np.ndarray,
        candidate_clusters: np.ndarray,
        *,
        resolution: float,
        epsilon: float = 0.0,
        min_doc_weight: float = 0.0,
        max_incident_directed_edges: int = 0,
        min_best_delta_q: float = 0.0,
        min_assigned_fraction: float = 0.0,
        min_best_group_fraction: float = 0.0,
    ) -> RustExternalGrainProbes:
        """Probe cheap external-attachment grains before full split-repair.

        Each source node is assigned to the external neighbor cluster to which
        it has the strongest edge weight. Nodes sharing that destination form a
        candidate grain. The method evaluates direct split/move deltas for the
        resulting grains without high-gamma local reclustering or repair.
        """
        probes = getattr(self.graph, "external_grain_probes", None)
        if probes is None:
            raise AttributeError(
                "installed sciscape_leiden module does not expose "
                "Graph.external_grain_probes"
            )
        membership = np.ascontiguousarray(membership, dtype=np.uint64)
        candidate_clusters = np.ascontiguousarray(candidate_clusters, dtype=np.uint64)
        raw = probes(
            membership=membership,
            candidate_clusters=candidate_clusters,
            resolution=float(resolution),
            epsilon=float(epsilon),
            min_doc_weight=float(min_doc_weight),
            max_incident_directed_edges=int(max_incident_directed_edges),
            min_best_delta_q=float(min_best_delta_q),
            min_assigned_fraction=float(min_assigned_fraction),
            min_best_group_fraction=float(min_best_group_fraction),
        )
        return RustExternalGrainProbes(
            cluster=np.asarray(raw["cluster"], dtype=np.uint64),
            block_count=np.asarray(raw["block_count"], dtype=np.uint64),
            doc_weight=np.asarray(raw["doc_weight"], dtype=np.float64),
            incident_directed_edges=np.asarray(
                raw["incident_directed_edges"], dtype=np.uint64
            ),
            source_directed_edges=np.asarray(
                raw["source_directed_edges"], dtype=np.uint64
            ),
            external_directed_edges=np.asarray(
                raw["external_directed_edges"], dtype=np.uint64
            ),
            n_external_groups=np.asarray(raw["n_external_groups"], dtype=np.uint64),
            assigned_count=np.asarray(raw["assigned_count"], dtype=np.uint64),
            assigned_weight=np.asarray(raw["assigned_weight"], dtype=np.float64),
            assigned_fraction=np.asarray(raw["assigned_fraction"], dtype=np.float64),
            largest_group_target=np.asarray(raw["largest_group_target"], dtype=np.int64),
            largest_group_count=np.asarray(raw["largest_group_count"], dtype=np.uint64),
            largest_group_weight=np.asarray(
                raw["largest_group_weight"], dtype=np.float64
            ),
            largest_group_fraction=np.asarray(
                raw["largest_group_fraction"], dtype=np.float64
            ),
            largest_group_to_target_weight=np.asarray(
                raw["largest_group_to_target_weight"], dtype=np.float64
            ),
            largest_group_cut_weight=np.asarray(
                raw["largest_group_cut_weight"], dtype=np.float64
            ),
            largest_group_move_delta_q=np.asarray(
                raw["largest_group_move_delta_q"], dtype=np.float64
            ),
            largest_group_split_delta_q=np.asarray(
                raw["largest_group_split_delta_q"], dtype=np.float64
            ),
            second_group_target=np.asarray(raw["second_group_target"], dtype=np.int64),
            second_group_weight=np.asarray(
                raw["second_group_weight"], dtype=np.float64
            ),
            second_group_fraction=np.asarray(
                raw["second_group_fraction"], dtype=np.float64
            ),
            best_group_target=np.asarray(raw["best_group_target"], dtype=np.int64),
            best_group_count=np.asarray(raw["best_group_count"], dtype=np.uint64),
            best_group_weight=np.asarray(raw["best_group_weight"], dtype=np.float64),
            best_group_fraction=np.asarray(
                raw["best_group_fraction"], dtype=np.float64
            ),
            best_group_to_target_weight=np.asarray(
                raw["best_group_to_target_weight"], dtype=np.float64
            ),
            best_group_cut_weight=np.asarray(
                raw["best_group_cut_weight"], dtype=np.float64
            ),
            best_group_move_delta_q=np.asarray(
                raw["best_group_move_delta_q"], dtype=np.float64
            ),
            best_group_split_delta_q=np.asarray(
                raw["best_group_split_delta_q"], dtype=np.float64
            ),
            best_group_delta_q=np.asarray(raw["best_group_delta_q"], dtype=np.float64),
            best_group_action=np.asarray(raw["best_group_action"], dtype=np.uint8),
            positive_group_count=np.asarray(
                raw["positive_group_count"], dtype=np.uint64
            ),
            positive_group_weight=np.asarray(
                raw["positive_group_weight"], dtype=np.float64
            ),
            near_neutral_group_count=np.asarray(
                raw["near_neutral_group_count"], dtype=np.uint64
            ),
            near_neutral_group_weight=np.asarray(
                raw["near_neutral_group_weight"], dtype=np.float64
            ),
            recommended_for_split_repair=np.asarray(
                raw["recommended_for_split_repair"], dtype=bool
            ),
            priority=np.asarray(raw["priority"], dtype=np.float64),
        )

    def multi_core_split_probes(
        self,
        membership: np.ndarray,
        candidate_clusters: np.ndarray,
        *,
        resolution: float,
        gamma_multipliers: Sequence[float],
        min_core_weight: float = 25.0,
        randomness: float = 0.01,
        seed: int = 0,
    ) -> RustMultiCoreSplitProbes:
        """Probe high-gamma induced splits inside candidate clusters.

        The resulting partitions are evaluated both at the baseline resolution
        and at the probing resolution. This keeps hysteresis-only splits visible
        without mutating ``membership``.
        """
        probes = getattr(self.graph, "multi_core_split_probes", None)
        if probes is None:
            raise AttributeError(
                "installed sciscape_leiden module does not expose "
                "Graph.multi_core_split_probes"
            )
        membership = np.ascontiguousarray(membership, dtype=np.uint64)
        candidate_clusters = np.ascontiguousarray(candidate_clusters, dtype=np.uint64)
        gamma_multipliers_array = np.ascontiguousarray(gamma_multipliers, dtype=np.float64)
        raw = probes(
            membership=membership,
            candidate_clusters=candidate_clusters,
            resolution=float(resolution),
            gamma_multipliers=gamma_multipliers_array,
            min_core_weight=float(min_core_weight),
            randomness=float(randomness),
            seed=int(seed),
        )
        return RustMultiCoreSplitProbes(
            cluster=np.asarray(raw["cluster"], dtype=np.uint64),
            gamma_multiplier=np.asarray(raw["gamma_multiplier"], dtype=np.float64),
            probe_resolution=np.asarray(raw["probe_resolution"], dtype=np.float64),
            block_count=np.asarray(raw["block_count"], dtype=np.uint64),
            doc_weight=np.asarray(raw["doc_weight"], dtype=np.float64),
            internal_weight=np.asarray(raw["internal_weight"], dtype=np.float64),
            induced_directed_edges=np.asarray(raw["induced_directed_edges"], dtype=np.uint64),
            n_parts=np.asarray(raw["n_parts"], dtype=np.uint64),
            non_singleton_parts=np.asarray(raw["non_singleton_parts"], dtype=np.uint64),
            singleton_parts=np.asarray(raw["singleton_parts"], dtype=np.uint64),
            singleton_weight=np.asarray(raw["singleton_weight"], dtype=np.float64),
            core_part_count=np.asarray(raw["core_part_count"], dtype=np.uint64),
            core_part_weight=np.asarray(raw["core_part_weight"], dtype=np.float64),
            largest_part_weight=np.asarray(raw["largest_part_weight"], dtype=np.float64),
            second_part_weight=np.asarray(raw["second_part_weight"], dtype=np.float64),
            largest_part_fraction=np.asarray(raw["largest_part_fraction"], dtype=np.float64),
            cut_weight=np.asarray(raw["cut_weight"], dtype=np.float64),
            split_delta_q_base=np.asarray(raw["split_delta_q_base"], dtype=np.float64),
            split_delta_q_probe=np.asarray(raw["split_delta_q_probe"], dtype=np.float64),
            hysteresis_only=np.asarray(raw["hysteresis_only"], dtype=bool),
        )

    def split_merge_repair_probes(
        self,
        membership: np.ndarray,
        candidate_clusters: np.ndarray,
        *,
        resolution: float,
        gamma_multipliers: Sequence[float],
        min_core_weight: float = 25.0,
        randomness: float = 0.01,
        repair_epsilon: float = 0.0,
        seed: int = 0,
        pair_seeded: bool = False,
    ) -> RustSplitMergeRepairProbes:
        """Probe forced high-gamma splits followed by baseline-gamma repair."""
        probes = getattr(self.graph, "split_merge_repair_probes", None)
        if probes is None:
            raise AttributeError(
                "installed sciscape_leiden module does not expose "
                "Graph.split_merge_repair_probes"
            )
        membership = np.ascontiguousarray(membership, dtype=np.uint64)
        candidate_clusters = np.ascontiguousarray(candidate_clusters, dtype=np.uint64)
        gamma_multipliers_array = np.ascontiguousarray(gamma_multipliers, dtype=np.float64)
        probe_kwargs = {
            "membership": membership,
            "candidate_clusters": candidate_clusters,
            "resolution": float(resolution),
            "gamma_multipliers": gamma_multipliers_array,
            "min_core_weight": float(min_core_weight),
            "randomness": float(randomness),
            "repair_epsilon": float(repair_epsilon),
            "seed": int(seed),
        }
        if pair_seeded:
            probe_kwargs["pair_seeded"] = True
        raw = probes(**probe_kwargs)
        return RustSplitMergeRepairProbes(
            cluster=np.asarray(raw["cluster"], dtype=np.uint64),
            gamma_multiplier=np.asarray(raw["gamma_multiplier"], dtype=np.float64),
            probe_resolution=np.asarray(raw["probe_resolution"], dtype=np.float64),
            block_count=np.asarray(raw["block_count"], dtype=np.uint64),
            doc_weight=np.asarray(raw["doc_weight"], dtype=np.float64),
            induced_directed_edges=np.asarray(
                raw["induced_directed_edges"], dtype=np.uint64
            ),
            n_parts=np.asarray(raw["n_parts"], dtype=np.uint64),
            core_part_count=np.asarray(raw["core_part_count"], dtype=np.uint64),
            singleton_weight=np.asarray(raw["singleton_weight"], dtype=np.float64),
            cut_weight=np.asarray(raw["cut_weight"], dtype=np.float64),
            split_delta_q_base=np.asarray(raw["split_delta_q_base"], dtype=np.float64),
            split_delta_q_probe=np.asarray(raw["split_delta_q_probe"], dtype=np.float64),
            repair_quotient_edges=np.asarray(
                raw["repair_quotient_edges"], dtype=np.uint64
            ),
            repair_merge_count=np.asarray(raw["repair_merge_count"], dtype=np.uint64),
            repair_delta_q=np.asarray(raw["repair_delta_q"], dtype=np.float64),
            net_delta_q=np.asarray(raw["net_delta_q"], dtype=np.float64),
            final_source_units=np.asarray(raw["final_source_units"], dtype=np.uint64),
            retained_source_units=np.asarray(raw["retained_source_units"], dtype=np.uint64),
            escaped_source_units=np.asarray(raw["escaped_source_units"], dtype=np.uint64),
            escaped_source_weight=np.asarray(raw["escaped_source_weight"], dtype=np.float64),
            final_small_source_units=np.asarray(raw["final_small_source_units"], dtype=np.uint64),
            final_small_source_weight=np.asarray(
                raw["final_small_source_weight"], dtype=np.float64
            ),
            largest_source_unit_fraction=np.asarray(
                raw["largest_source_unit_fraction"], dtype=np.float64
            ),
            restored_source_cluster=np.asarray(raw["restored_source_cluster"], dtype=bool),
        )

    def apply_split_merge_repair_candidates(
        self,
        membership: np.ndarray,
        candidate_clusters: np.ndarray,
        selected_clusters: np.ndarray,
        selected_gamma_multipliers: Sequence[float],
        *,
        resolution: float,
        gamma_multipliers: Sequence[float],
        min_core_weight: float = 25.0,
        randomness: float = 0.01,
        repair_epsilon: float = 0.0,
        seed: int = 0,
        pair_seeded: bool = False,
    ) -> RustSplitRepairApplyResult:
        """Apply selected split-repair candidates to a proposed membership.

        This method is deterministic with respect to ``candidate_clusters`` and
        ``gamma_multipliers``: it replays the probe loop in the same order and
        only materializes rows listed in ``selected_clusters`` /
        ``selected_gamma_multipliers``.
        """
        apply = getattr(self.graph, "apply_split_merge_repair_candidates", None)
        if apply is None:
            raise AttributeError(
                "installed sciscape_leiden module does not expose "
                "Graph.apply_split_merge_repair_candidates"
            )
        membership = np.ascontiguousarray(membership, dtype=np.uint64)
        candidate_clusters = np.ascontiguousarray(candidate_clusters, dtype=np.uint64)
        selected_clusters = np.ascontiguousarray(selected_clusters, dtype=np.uint64)
        selected_gamma_multipliers_array = np.ascontiguousarray(
            selected_gamma_multipliers,
            dtype=np.float64,
        )
        gamma_multipliers_array = np.ascontiguousarray(gamma_multipliers, dtype=np.float64)
        apply_kwargs = {
            "membership": membership,
            "candidate_clusters": candidate_clusters,
            "selected_clusters": selected_clusters,
            "selected_gamma_multipliers": selected_gamma_multipliers_array,
            "resolution": float(resolution),
            "gamma_multipliers": gamma_multipliers_array,
            "min_core_weight": float(min_core_weight),
            "randomness": float(randomness),
            "repair_epsilon": float(repair_epsilon),
            "seed": int(seed),
        }
        if pair_seeded:
            apply_kwargs["pair_seeded"] = True
        raw = apply(**apply_kwargs)
        return RustSplitRepairApplyResult(
            membership=np.asarray(raw["membership"], dtype=np.uint64),
            selected_index=np.asarray(raw["selected_index"], dtype=np.uint64),
            cluster=np.asarray(raw["cluster"], dtype=np.uint64),
            gamma_multiplier=np.asarray(raw["gamma_multiplier"], dtype=np.float64),
            probe_resolution=np.asarray(raw["probe_resolution"], dtype=np.float64),
            block_count=np.asarray(raw["block_count"], dtype=np.uint64),
            doc_weight=np.asarray(raw["doc_weight"], dtype=np.float64),
            n_parts=np.asarray(raw["n_parts"], dtype=np.uint64),
            split_delta_q_base=np.asarray(raw["split_delta_q_base"], dtype=np.float64),
            repair_delta_q=np.asarray(raw["repair_delta_q"], dtype=np.float64),
            predicted_net_delta_q=np.asarray(raw["predicted_net_delta_q"], dtype=np.float64),
            repair_merge_count=np.asarray(raw["repair_merge_count"], dtype=np.uint64),
            final_source_units=np.asarray(raw["final_source_units"], dtype=np.uint64),
            retained_source_units=np.asarray(raw["retained_source_units"], dtype=np.uint64),
            escaped_source_units=np.asarray(raw["escaped_source_units"], dtype=np.uint64),
            escaped_source_weight=np.asarray(raw["escaped_source_weight"], dtype=np.float64),
            final_small_source_units=np.asarray(
                raw["final_small_source_units"], dtype=np.uint64
            ),
            final_small_source_weight=np.asarray(
                raw["final_small_source_weight"], dtype=np.float64
            ),
            largest_source_unit_fraction=np.asarray(
                raw["largest_source_unit_fraction"], dtype=np.float64
            ),
            changed_nodes=np.asarray(raw["changed_nodes"], dtype=np.uint64),
            moved_to_existing_cluster_nodes=np.asarray(
                raw["moved_to_existing_cluster_nodes"], dtype=np.uint64
            ),
            moved_to_new_cluster_nodes=np.asarray(
                raw["moved_to_new_cluster_nodes"], dtype=np.uint64
            ),
            new_retained_clusters=np.asarray(raw["new_retained_clusters"], dtype=np.uint64),
        )

    def postprocess_small_clusters(
        self,
        *,
        resolution: float,
        min_size: int = 0,
        min_weight: float = 0.0,
        membership: np.ndarray,
        seed: int = 0,
        n_iterations: int = 10,
        randomness: float = 0.01,
        max_rounds: int = 5,
        gamma_decay: float = 0.5,
        use_greedy: bool = True,
        greedy_anchor_only: bool = False,
        greedy_fallback_to_small: bool = False,
        greedy_max_weight: float = 0.0,
        use_component_merge: bool = True,
        component_max_weight: float = 0.0,
    ) -> RustPostprocessResult:
        membership = np.ascontiguousarray(membership, dtype=np.uint64)
        result_mem, n_clusters, changed_at, rounds = self.graph.run_postprocess(
            membership=membership,
            resolution=resolution,
            min_size=min_size,
            n_iterations=n_iterations,
            randomness=randomness,
            seed=seed,
            min_weight=min_weight,
            max_rounds=int(max_rounds),
            gamma_decay=float(gamma_decay),
            use_greedy=bool(use_greedy),
            greedy_anchor_only=bool(greedy_anchor_only),
            greedy_fallback_to_small=bool(greedy_fallback_to_small),
            greedy_max_weight=float(greedy_max_weight),
            use_component_merge=bool(use_component_merge),
            component_max_weight=float(component_max_weight),
        )
        return RustPostprocessResult(
            membership=result_mem,
            n_clusters=n_clusters,
            changed_at_round=changed_at,
            rounds=rounds,
        )

    def contract(
        self,
        membership: np.ndarray,
        *,
        keep_self_loops: bool = True,
    ) -> "RustLeidenGraph":
        membership = np.ascontiguousarray(membership, dtype=np.uint64)
        native, node_weights = self.graph.contract(
            membership=membership,
            keep_self_loops=keep_self_loops,
        )
        return RustLeidenGraph(
            graph=native,
            n_nodes=int(native.n_nodes),
            n_edges=int(native.n_edges),
            node_weights=np.asarray(node_weights, dtype=np.float64),
        )


def _load_or_coerce_edges(
    edge_path: Path | None = None,
    *,
    n_nodes: int | None = None,
    edges_src: np.ndarray | None = None,
    edges_dst: np.ndarray | None = None,
    edges_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    if edges_src is None:
        if edge_path is None:
            raise ValueError("Provide either edge_path or edges_src/dst/weight")
        src_path, dst_path, weight_path = ensure_int_edge_sidecars(Path(edge_path))
        edges_src = np.memmap(src_path, dtype=np.uint32, mode="r")
        edges_dst = np.memmap(dst_path, dtype=np.uint32, mode="r")
        edges_weight = np.memmap(weight_path, dtype=np.float64, mode="r")
        if n_nodes is None:
            n_nodes = int(max(int(edges_src.max()), int(edges_dst.max()))) + 1

    if edges_dst is None or edges_weight is None:
        raise ValueError("Provide edges_src, edges_dst, and edges_weight together")

    edges_src = np.ascontiguousarray(edges_src, dtype=np.uint32)
    edges_dst = np.ascontiguousarray(edges_dst, dtype=np.uint32)
    edges_weight = np.ascontiguousarray(edges_weight, dtype=np.float64)

    if n_nodes is None:
        n_nodes = int(max(edges_src.max(), edges_dst.max())) + 1

    return edges_src, edges_dst, edges_weight, n_nodes


def build_leiden_graph(
    edge_path: Path | None = None,
    *,
    n_nodes: int | None = None,
    edges_src: np.ndarray | None = None,
    edges_dst: np.ndarray | None = None,
    edges_weight: np.ndarray | None = None,
    node_weights: np.ndarray | None = None,
) -> RustLeidenGraph:
    """Build a reusable Rust CSR graph from edge arrays or an int-edge parquet."""
    _check_available()
    graph_cls = getattr(_rust, "Graph", None)
    if graph_cls is None:
        raise AttributeError("installed sciscape_leiden module does not expose Graph")

    if edge_path is not None and edges_src is None:
        raw_loader = getattr(_rust, "load_graph_raw_files", None)
        if raw_loader is not None:
            if n_nodes is None:
                raise ValueError("n_nodes is required when loading a graph from raw sidecars")
            src_path, dst_path, weight_path = ensure_int_edge_sidecars(Path(edge_path))
            node_weights_path = None
            try:
                if node_weights is not None:
                    with NamedTemporaryFile(
                        prefix="node_weights.",
                        suffix=".f64.bin",
                        dir=Path(edge_path).parent,
                        delete=False,
                    ) as fh:
                        np.ascontiguousarray(node_weights, dtype=np.float64).tofile(fh)
                        node_weights_path = fh.name
                graph = raw_loader(
                    n_nodes=n_nodes,
                    src_path=str(src_path),
                    dst_path=str(dst_path),
                    weights_path=str(weight_path),
                    node_weights_path=node_weights_path,
                )
            finally:
                if node_weights_path is not None:
                    Path(node_weights_path).unlink(missing_ok=True)
            return RustLeidenGraph(
                graph=graph,
                n_nodes=int(graph.n_nodes),
                n_edges=int(graph.n_edges),
                node_weights=None if node_weights is None else np.asarray(node_weights, dtype=np.float64),
            )

    edges_src, edges_dst, edges_weight, n_nodes = _load_or_coerce_edges(
        edge_path,
        n_nodes=n_nodes,
        edges_src=edges_src,
        edges_dst=edges_dst,
        edges_weight=edges_weight,
    )
    nw = None
    if node_weights is not None:
        nw = np.ascontiguousarray(node_weights, dtype=np.float64)

    graph = graph_cls(
        n_nodes=n_nodes,
        src=edges_src,
        dst=edges_dst,
        weights=edges_weight,
        node_weights=nw,
    )
    return RustLeidenGraph(
        graph=graph,
        n_nodes=int(graph.n_nodes),
        n_edges=int(graph.n_edges),
        node_weights=nw,
    )


def remap_parquet_to_leiden_graph(
    edge_path: Path,
    output_dir: Path,
    *,
    uid1_col: str = "uid1",
    uid2_col: str = "uid2",
    weight_col: str = "rel_sum2",
) -> tuple[object, RustLeidenGraph] | None:
    """Remap a string-UID parquet edge table and build a Rust graph directly.

    This avoids writing raw sidecars and then reading them back into CSR.  It is
    intended for the Rust pipeline's initial graph build; Java/compat paths keep
    using the file-backed remap outputs.
    """
    _check_available()
    remap_graph = getattr(_rust, "rust_integer_remap_parquet_graph", None)
    if remap_graph is None:
        return None

    from .integer_remap import RemapResult

    n_nodes, n_edges, manifest_path, int_edges_path, native = remap_graph(
        str(edge_path),
        str(output_dir),
        uid1_col,
        uid2_col,
        weight_col,
    )
    remap = RemapResult(
        n_nodes=int(n_nodes),
        n_edges=int(n_edges),
        node_manifest_path=Path(manifest_path),
        int_edges_path=Path(int_edges_path),
    )
    graph = RustLeidenGraph(
        graph=native,
        n_nodes=int(native.n_nodes),
        n_edges=int(native.n_edges),
        node_weights=None,
    )
    return remap, graph


def project_membership_rust(
    membership: np.ndarray,
    previous_membership: np.ndarray,
) -> np.ndarray:
    """Project contracted-graph membership back to original nodes in Rust."""
    _check_available()
    mem = np.ascontiguousarray(membership, dtype=np.uint64)
    prev = np.asarray(previous_membership)

    project_u32 = getattr(_rust, "rust_project_membership_u32", None)
    project_u64 = getattr(_rust, "rust_project_membership_u64", None)
    if project_u32 is None or project_u64 is None:
        return mem[np.ascontiguousarray(prev, dtype=np.uint64)]

    if prev.dtype == np.uint32:
        return project_u32(mem, np.ascontiguousarray(prev, dtype=np.uint32))
    if prev.dtype == np.uint64:
        return project_u64(mem, np.ascontiguousarray(prev, dtype=np.uint64))
    if np.issubdtype(prev.dtype, np.unsignedinteger) and prev.dtype.itemsize <= 4:
        return project_u32(mem, np.ascontiguousarray(prev, dtype=np.uint32))
    return project_u64(mem, np.ascontiguousarray(prev, dtype=np.uint64))


def run_leiden_rust(
    edge_path: Path | None = None,
    *,
    resolution: float,
    n_nodes: int | None = None,
    edges_src: np.ndarray | None = None,
    edges_dst: np.ndarray | None = None,
    edges_weight: np.ndarray | None = None,
    seed: int = 0,
    n_iterations: int = 10,
    n_starts: int = 1,
    randomness: float = 0.01,
    randomness_schedule: Sequence[float] | None = None,
    initial_membership: np.ndarray | None = None,
    fixed_nodes: np.ndarray | None = None,
    node_weights: np.ndarray | None = None,
) -> RustLeidenResult:
    """Run Leiden clustering via the Rust backend.

    Accepts either file path (parquet with src/dst/weight columns)
    or pre-loaded numpy arrays.

    Parameters
    ----------
    edge_path : Path, optional
        Path to int_edges.parquet (columns: src, dst, weight).
    resolution : float
        CPM resolution parameter.
    n_nodes : int, optional
        Total number of nodes. Required if using edge_path.
    edges_src, edges_dst, edges_weight : numpy arrays, optional
        Pre-loaded edge arrays. Alternative to edge_path.
    seed, n_iterations, n_starts, randomness
        Leiden parameters.
    randomness_schedule : sequence of float, optional
        Per-iteration refinement randomness. If provided, the last value is reused
        for iterations beyond the schedule length.
    initial_membership : numpy array, optional
        Initial cluster assignment (uint64).
    fixed_nodes : numpy array, optional
        Boolean mask of nodes that cannot change cluster.

    Returns
    -------
    RustLeidenResult
    """
    _check_available()

    schedule = (
        None
        if randomness_schedule is None
        else [float(x) for x in randomness_schedule]
    )

    nw = None
    if node_weights is not None:
        nw = np.ascontiguousarray(node_weights, dtype=np.float64)

    if hasattr(_rust, "Graph"):
        graph = build_leiden_graph(
            edge_path,
            n_nodes=n_nodes,
            edges_src=edges_src,
            edges_dst=edges_dst,
            edges_weight=edges_weight,
            node_weights=nw,
        )
        result = graph.run_leiden(
            resolution=resolution,
            seed=seed,
            n_iterations=n_iterations,
            n_starts=n_starts,
            randomness=randomness,
            randomness_schedule=schedule,
            initial_membership=initial_membership,
            fixed_nodes=fixed_nodes,
        )
        log.info(
            "leiden_rust: %d nodes → %d clusters (γ=%.6g, Q=%.4f)",
            graph.n_nodes, result.n_clusters, resolution, result.quality,
        )
        return result

    edges_src, edges_dst, edges_weight, n_nodes = _load_or_coerce_edges(
        edge_path,
        n_nodes=n_nodes,
        edges_src=edges_src,
        edges_dst=edges_dst,
        edges_weight=edges_weight,
    )

    membership, quality, n_clusters = _rust.run_leiden(
        n_nodes=n_nodes,
        src=edges_src,
        dst=edges_dst,
        weights=edges_weight,
        resolution=resolution,
        n_iterations=n_iterations,
        n_starts=n_starts,
        randomness=randomness,
        randomness_schedule=schedule,
        seed=seed,
        initial_membership=initial_membership,
        fixed_nodes=fixed_nodes,
        node_weights=nw,
    )

    log.info(
        "leiden_rust: %d nodes → %d clusters (γ=%.6g, Q=%.4f)",
        n_nodes, n_clusters, resolution, quality,
    )

    return RustLeidenResult(
        membership=membership,
        quality=quality,
        n_clusters=n_clusters,
    )


def postprocess_small_clusters_rust(
    *,
    resolution: float,
    min_size: int = 0,
    min_weight: float = 0.0,
    membership: np.ndarray,
    n_nodes: int | None = None,
    edge_path: Path | None = None,
    edges_src: np.ndarray | None = None,
    edges_dst: np.ndarray | None = None,
    edges_weight: np.ndarray | None = None,
    node_weights: np.ndarray | None = None,
    seed: int = 0,
    n_iterations: int = 10,
    randomness: float = 0.01,
    max_rounds: int = 5,
    gamma_decay: float = 0.5,
    use_greedy: bool = True,
    greedy_anchor_only: bool = False,
    greedy_fallback_to_small: bool = False,
    greedy_max_weight: float = 0.0,
    use_component_merge: bool = True,
    component_max_weight: float = 0.0,
) -> RustPostprocessResult:
    """Reassign small clusters using constrained Leiden (Rust backend).

    Threshold semantics:
    - If ``node_weights`` is provided and ``min_weight > 0``, clusters are
      considered "small" when their total node_weight < min_weight (doc_count).
    - Otherwise, raw node count < min_size is used.
    """
    _check_available()

    membership = np.ascontiguousarray(membership, dtype=np.uint64)

    nw = None
    if node_weights is not None:
        nw = np.ascontiguousarray(node_weights, dtype=np.float64)

    if hasattr(_rust, "Graph"):
        graph = build_leiden_graph(
            edge_path,
            n_nodes=n_nodes or len(membership),
            edges_src=edges_src,
            edges_dst=edges_dst,
            edges_weight=edges_weight,
            node_weights=nw,
        )
        post = graph.postprocess_small_clusters(
            resolution=resolution,
            min_size=min_size,
            min_weight=min_weight,
            membership=membership,
            seed=seed,
            n_iterations=n_iterations,
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
        result_mem = post.membership
        n_clusters = post.n_clusters
        changed_at = post.changed_at_round
        rounds = post.rounds
    else:
        edges_src, edges_dst, edges_weight, n_nodes = _load_or_coerce_edges(
            edge_path,
            n_nodes=n_nodes or len(membership),
            edges_src=edges_src,
            edges_dst=edges_dst,
            edges_weight=edges_weight,
        )
        result_mem, n_clusters, changed_at, rounds = _rust.run_postprocess(
            n_nodes=n_nodes,
            src=edges_src,
            dst=edges_dst,
            weights=edges_weight,
            membership=membership,
            resolution=resolution,
            min_size=min_size,
            n_iterations=n_iterations,
            randomness=randomness,
            seed=seed,
            node_weights=nw,
            min_weight=min_weight,
            max_rounds=int(max_rounds),
            gamma_decay=float(gamma_decay),
            use_greedy=bool(use_greedy),
            greedy_anchor_only=bool(greedy_anchor_only),
            greedy_fallback_to_small=bool(greedy_fallback_to_small),
            greedy_max_weight=float(greedy_max_weight),
            use_component_merge=bool(use_component_merge),
            component_max_weight=float(component_max_weight),
        )

    changed = int(np.sum(changed_at >= 0))
    threshold_str = (
        f"min_weight={min_weight}" if min_weight > 0
        else f"min_size={min_size}"
    )
    for r in rounds:
        log.info(
            "postprocess round %d: γ=%.4e, method=%s, small: %d→%d, "
            "merged: %d, total: %d, max_size: %d, max_weight: %.1f",
            r["round"], r["gamma"], r["method"],
            r["n_small_before"], r["n_small_after"],
            r["n_merged"], r["n_total_clusters"], r["max_cluster_size"],
            r["max_cluster_weight"],
        )
    log.info(
        "postprocess_rust: %d nodes changed, %d clusters (%s, %d rounds)",
        changed, n_clusters, threshold_str, len(rounds),
    )

    return RustPostprocessResult(
        membership=result_mem,
        n_clusters=n_clusters,
        changed_at_round=changed_at,
        rounds=rounds,
    )


__all__ = [
    "RUST_AVAILABLE",
    "RustLeidenGraph",
    "RustClusterGraphStats",
    "RustExternalGrainProbes",
    "RustOversizeBoundaryTrimResult",
    "RustLeidenResult",
    "RustPostprocessResult",
    "RustResolutionSearchResult",
    "RustSplitRepairApplyResult",
    "build_leiden_graph",
    "remap_parquet_to_leiden_graph",
    "project_membership_rust",
    "run_leiden_rust",
    "postprocess_small_clusters_rust",
]
