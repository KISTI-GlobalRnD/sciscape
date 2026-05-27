"""Rust backend wrapper for Leiden clustering via sciscape-leiden.

Drop-in replacement for leiden_java.py functions using the Rust
native module. Much faster than Java (no JVM startup, no file I/O)
and no JDK dependency.

Stable public use should center on CPM Leiden graph construction, Leiden runs,
membership projection, and small-cluster postprocess. Dongdaemun helpers in this
module are development-only SciSci research surfaces and are intentionally not
re-exported from ``sciscape.clustering``.

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
from typing import Any, Sequence

import numpy as np
import polars as pl

from .integer_remap import ensure_int_edge_sidecars

log = logging.getLogger(__name__)

DEFAULT_DONGDAEMUN_GAMMA_MULTIPLIERS = (1.02, 1.05, 1.10, 1.15, 1.20, 1.25)

try:
    import sciscape_leiden as _rust

    RUST_AVAILABLE = True
    _rust_graph_cls = getattr(_rust, "Graph", None)
    RUST_DONGDAEMUN_AVAILABLE = bool(
        _rust_graph_cls is not None
        and callable(getattr(_rust_graph_cls, "dongdaemun_refine", None))
    )
    RUST_DONGDAEMUN_REFINEMENT_AVAILABLE = bool(
        _rust_graph_cls is not None
        and callable(getattr(_rust_graph_cls, "run_leiden_dongdaemun_refinement", None))
    )
except ImportError:
    RUST_AVAILABLE = False
    RUST_DONGDAEMUN_AVAILABLE = False
    RUST_DONGDAEMUN_REFINEMENT_AVAILABLE = False


def _check_available():
    if not RUST_AVAILABLE:
        raise ImportError(
            "sciscape-leiden Rust module not installed. "
            "Install with: pip install sciscape-leiden "
            "Or build: cd sciscape-leiden && maturin develop --release"
        )


def _check_dongdaemun_available():
    _check_available()
    if not RUST_DONGDAEMUN_AVAILABLE:
        raise AttributeError(
            "installed sciscape_leiden module does not expose "
            "Graph.dongdaemun_refine. Rebuild the local extension with: "
            "uv run --extra dev maturin develop --manifest-path rust/Cargo.toml"
        )


def _check_dongdaemun_refinement_available():
    _check_available()
    if not RUST_DONGDAEMUN_REFINEMENT_AVAILABLE:
        raise AttributeError(
            "installed sciscape_leiden module does not expose "
            "Graph.run_leiden_dongdaemun_refinement. Rebuild the local "
            "extension with: uv run --extra dev maturin develop "
            "--manifest-path rust/Cargo.toml"
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
class RustNonMonotoneGroupEscapeResult:
    """Opt-in non-monotone external-grain move followed by normal Leiden polish."""

    membership: np.ndarray
    quality: float
    accepted: bool
    candidate_rows: list[dict[str, Any]]
    baseline_quality: float
    best_delta_q: float
    elapsed_ms: float
    candidate_eval_parallel: bool
    candidate_eval_wall_elapsed_ms: float
    candidate_eval_cpu_sum_elapsed_ms: float
    candidate_eval_parallel_speedup: float
    candidate_eval_parallel_workers: int


@dataclass(frozen=True)
class RustNonMonotoneGroupEscapeMultifidelityResult:
    """Multi-fidelity labels for ranked external-grain escape candidates."""

    membership: np.ndarray
    quality: float
    accepted: bool
    selected_policy: str
    selected_candidate_index: int
    candidate_rows: list[dict[str, Any]]
    policy_rows: list[dict[str, Any]]
    baseline_quality: float
    best_delta_q: float
    elapsed_ms: float


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
class RustDongdaemunAudit:
    """Audit fields returned by Rust Dongdaemun upper-tail refinement."""

    accepted: bool
    status: str
    quality_before: float
    quality_after_candidate: float
    candidate_delta_q: float
    effective_delta_q: float
    final_delta_q: float
    target_max_satisfied: bool
    n_oversize_before: int
    n_oversize_after_candidate: int
    max_weight_before: float
    max_weight_after_candidate: float
    trim_moves_committed: int
    trim_moves_proposed: int
    split_iteration: np.ndarray
    split_candidate_clusters: np.ndarray
    split_n_selected: np.ndarray
    split_n_applied: np.ndarray
    split_status_code: np.ndarray
    split_exact_delta_q: np.ndarray


@dataclass(frozen=True)
class RustDongdaemunResult:
    """Effective and diagnostic memberships from Rust Dongdaemun refinement."""

    membership: np.ndarray
    n_clusters: int
    diagnostic_membership: np.ndarray | None
    diagnostic_n_clusters: int
    audit: RustDongdaemunAudit


@dataclass(frozen=True)
class RustDongdaemunRefinementAudit:
    """Audit fields returned by integrated Dongdaemun-Leiden refinement."""

    enabled: bool
    selected_parent_count_total: int
    applied_parent_count_total: int
    rejected_candidate_count_total: int
    added_refined_clusters_total: int
    same_gamma_candidates_total: int
    high_gamma_candidates_total: int
    same_gamma_applied_total: int
    high_gamma_applied_total: int
    quotient_candidates_total: int
    quotient_positive_candidates_total: int
    quotient_selected_total: int
    quotient_score_sum: float
    baseline_repair_candidates_total: int
    baseline_repair_improved_candidates_total: int
    baseline_repair_selected_total: int
    baseline_repair_merge_count_total: int
    baseline_repair_delta_sum: float
    candidate_quality_delta_sum: float
    candidate_positive_quality_delta_total: int
    candidate_selected_positive_quality_delta_total: int
    candidate_rejected_by_quality_total: int
    same_gamma_quality_delta_sum: float
    high_gamma_quality_delta_sum: float
    same_gamma_positive_quality_delta_total: int
    high_gamma_positive_quality_delta_total: int
    same_gamma_selected_positive_quality_delta_total: int
    high_gamma_selected_positive_quality_delta_total: int
    same_gamma_rejected_by_quality_total: int
    high_gamma_rejected_by_quality_total: int
    candidate_valid_total: int
    candidate_invalid_total: int
    candidate_rejected_by_policy_total: int
    same_gamma_valid_total: int
    high_gamma_valid_total: int
    same_gamma_invalid_total: int
    high_gamma_invalid_total: int
    same_gamma_rejected_by_policy_total: int
    high_gamma_rejected_by_policy_total: int
    candidate_qpos_spos_total: int
    candidate_qpos_sneg_total: int
    candidate_qneg_spos_total: int
    candidate_qneg_sneg_total: int
    same_gamma_qpos_spos_total: int
    same_gamma_qpos_sneg_total: int
    same_gamma_qneg_spos_total: int
    same_gamma_qneg_sneg_total: int
    high_gamma_qpos_spos_total: int
    high_gamma_qpos_sneg_total: int
    high_gamma_qneg_spos_total: int
    high_gamma_qneg_sneg_total: int
    candidate_true_positive_total: int
    candidate_false_positive_total: int
    candidate_false_negative_total: int
    candidate_true_negative_total: int
    adaptive_local_shake_triggers_total: int
    adaptive_local_shake_candidates_total: int
    adaptive_local_shake_commits_total: int
    adaptive_local_shake_qf_gain_sum: float
    final_quality_guard_enabled: bool
    final_quality_guard_triggered: bool
    final_quality_guard_standard_quality: float
    final_quality_guard_pre_guard_quality: float
    final_quality_delta_vs_guard_standard: float
    max_parent_weight_seen: float
    iteration_depth: np.ndarray
    iteration_selected_parents: np.ndarray
    iteration_applied_parents: np.ndarray
    iteration_same_gamma_candidates: np.ndarray
    iteration_high_gamma_candidates: np.ndarray
    iteration_same_gamma_applied: np.ndarray
    iteration_high_gamma_applied: np.ndarray
    iteration_quotient_candidates: np.ndarray
    iteration_quotient_positive_candidates: np.ndarray
    iteration_quotient_selected: np.ndarray
    iteration_quotient_score_sum: np.ndarray
    iteration_baseline_repair_candidates: np.ndarray
    iteration_baseline_repair_improved_candidates: np.ndarray
    iteration_baseline_repair_selected: np.ndarray
    iteration_baseline_repair_merge_count: np.ndarray
    iteration_baseline_repair_delta_sum: np.ndarray
    iteration_candidate_quality_delta_sum: np.ndarray
    iteration_candidate_positive_quality_delta: np.ndarray
    iteration_candidate_selected_positive_quality_delta: np.ndarray
    iteration_candidate_rejected_by_quality: np.ndarray
    iteration_same_gamma_quality_delta_sum: np.ndarray
    iteration_high_gamma_quality_delta_sum: np.ndarray
    iteration_same_gamma_positive_quality_delta: np.ndarray
    iteration_high_gamma_positive_quality_delta: np.ndarray
    iteration_same_gamma_selected_positive_quality_delta: np.ndarray
    iteration_high_gamma_selected_positive_quality_delta: np.ndarray
    iteration_same_gamma_rejected_by_quality: np.ndarray
    iteration_high_gamma_rejected_by_quality: np.ndarray
    iteration_candidate_valid: np.ndarray
    iteration_candidate_invalid: np.ndarray
    iteration_candidate_rejected_by_policy: np.ndarray
    iteration_same_gamma_valid: np.ndarray
    iteration_high_gamma_valid: np.ndarray
    iteration_same_gamma_invalid: np.ndarray
    iteration_high_gamma_invalid: np.ndarray
    iteration_same_gamma_rejected_by_policy: np.ndarray
    iteration_high_gamma_rejected_by_policy: np.ndarray
    iteration_candidate_qpos_spos: np.ndarray
    iteration_candidate_qpos_sneg: np.ndarray
    iteration_candidate_qneg_spos: np.ndarray
    iteration_candidate_qneg_sneg: np.ndarray
    iteration_same_gamma_qpos_spos: np.ndarray
    iteration_same_gamma_qpos_sneg: np.ndarray
    iteration_same_gamma_qneg_spos: np.ndarray
    iteration_same_gamma_qneg_sneg: np.ndarray
    iteration_high_gamma_qpos_spos: np.ndarray
    iteration_high_gamma_qpos_sneg: np.ndarray
    iteration_high_gamma_qneg_spos: np.ndarray
    iteration_high_gamma_qneg_sneg: np.ndarray
    iteration_candidate_true_positive: np.ndarray
    iteration_candidate_false_positive: np.ndarray
    iteration_candidate_false_negative: np.ndarray
    iteration_candidate_true_negative: np.ndarray
    iteration_adaptive_local_shake_triggers: np.ndarray
    iteration_adaptive_local_shake_candidates: np.ndarray
    iteration_adaptive_local_shake_commits: np.ndarray
    iteration_adaptive_local_shake_qf_gain_sum: np.ndarray
    iteration_standard_refined_clusters: np.ndarray
    iteration_final_refined_clusters: np.ndarray


@dataclass(frozen=True)
class RustDongdaemunRefinementResult:
    """Result of Rust Leiden with opt-in Dongdaemun refinement allocation."""

    membership: np.ndarray
    quality: float
    n_clusters: int
    n_iterations_used: int
    audit: RustDongdaemunRefinementAudit


@dataclass(frozen=True)
class RustDongdaemunAutoFastResult:
    """Result of pressure-triggered Dongdaemun refinement selection."""

    membership: np.ndarray
    quality: float
    n_clusters: int
    selected_variant: str
    triggered: bool
    fallback_triggered: bool
    fallback_reason: str
    severe_tier_triggered: bool
    repair_escalated: bool
    repair_escalation_accepted: bool
    max_extra_parents_per_iteration: int
    max_extra_children_per_parent: int
    standard_max_doc_weight: float
    standard_max_doc_weight_ratio: float
    standard_n_above_max_doc_weight: int
    selected_max_doc_weight: float
    selected_max_doc_weight_ratio: float
    selected_n_above_max_doc_weight: int
    standard: RustLeidenResult
    repair_off: RustDongdaemunRefinementResult | None
    repair_on: RustDongdaemunRefinementResult | None


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
        membership_dtype: Any = np.uint64,
    ) -> RustLeidenResult:
        schedule = (
            None
            if randomness_schedule is None
            else [float(x) for x in randomness_schedule]
        )
        dtype = np.dtype(membership_dtype)
        run_leiden = self.graph.run_leiden
        if dtype == np.dtype(np.uint32):
            run_leiden_u32 = getattr(self.graph, "run_leiden_u32", None)
            if run_leiden_u32 is None:
                raise AttributeError(
                    "installed sciscape_leiden module does not expose Graph.run_leiden_u32"
                )
            run_leiden = run_leiden_u32
            if initial_membership is not None:
                initial_membership = np.ascontiguousarray(
                    initial_membership,
                    dtype=np.uint32,
                )
        membership, quality, n_clusters = run_leiden(
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

    def run_leiden_dongdaemun_refinement(
        self,
        *,
        target_max_weight: float,
        resolution: float,
        seed: int = 0,
        n_iterations: int = 10,
        randomness: float = 0.01,
        randomness_schedule: Sequence[float] | None = None,
        initial_membership: np.ndarray | None = None,
        fixed_nodes: np.ndarray | None = None,
        soft_min_ratio: float = 1.0,
        max_extra_parents_per_iteration: int = 16,
        max_extra_children_per_parent: int = 64,
        parent_selection_policy: str = "weight",
        max_singleton_weight_fraction: float = 0.05,
        min_largest_child_fraction_improvement: float = 0.05,
        gamma_multipliers: Sequence[float] = DEFAULT_DONGDAEMUN_GAMMA_MULTIPLIERS,
        seed_perturbations: int = 0,
        use_quotient_diagnostic: bool = False,
        use_baseline_repair: bool = False,
        baseline_repair_policy: str = "replace",
        baseline_repair_replace_min_parent_ratio: float = 1.05,
        baseline_repair_epsilon: float = 0.0,
        candidate_quality_policy: str = "structural",
        min_candidate_delta_q: float = 0.0,
        adaptive_plateau_quality_band: float = 0.0,
        use_final_quality_guard: bool = False,
        min_final_quality_delta: float = 0.0,
        adaptive_probe_mode: str = "off",
        adaptive_probe_perturbations: int = 0,
        adaptive_probe_targets: Sequence[str] | None = None,
        adaptive_probe_tolerance_parent_weight: float = 1e-6,
        adaptive_probe_include_node_order_control: bool = False,
        adaptive_probe_commit_min_gain_parent_weight: float = 0.0,
        adaptive_probe_max_commits_total: int = 0,
        adaptive_probe_max_commits_per_depth: int = 0,
        adaptive_probe_commit_sources: Sequence[str] | None = None,
        adaptive_probe_commit_strategy: str = "online_first",
        adaptive_near_tie_probe_mode: str = "off",
        adaptive_near_tie_margin_parent_weight: float = 0.0,
        adaptive_near_tie_randomness: float = 0.0,
        adaptive_near_tie_max_decisions_per_parent: int = 0,
        adaptive_local_shake_mode: str = "off",
        adaptive_local_shake_arms: Sequence[str] = (),
        adaptive_local_shake_max_arms_per_parent: int = 0,
        adaptive_local_shake_max_candidates_per_parent: int = 0,
        adaptive_local_shake_min_gain_parent_weight: float = 0.0,
        adaptive_local_shake_shape_eps: float = 1e-12,
        adaptive_local_shake_arm_priority: Sequence[str] = (),
        adaptive_local_shake_near_tie_min_count: int = 1,
        adaptive_local_shake_resolution_down_multipliers: Sequence[float] = (),
        adaptive_local_shake_resolution_up_multipliers: Sequence[float] = (),
        adaptive_local_shake_resolution_up_min_parent_ratio: float = 1.0,
        adaptive_local_shake_resolution_down_max_parent_ratio: float = 1.0,
        adaptive_local_shake_large_child_fraction: float = 0.95,
        adaptive_local_shake_singleton_fraction: float = 0.05,
        adaptive_local_shake_seed_perturbations: int = 0,
        adaptive_local_shake_seed_margin_count: int = 2,
        adaptive_local_shake_near_tie_margin_parent_weight: float = 0.0,
        adaptive_local_shake_near_tie_randomness: float = 0.0,
        adaptive_local_shake_final_guard_mode: str = "none",
    ) -> RustDongdaemunRefinementResult:
        """Run Rust Leiden with opt-in parent-internal Dongdaemun refinement."""
        run = getattr(self.graph, "run_leiden_dongdaemun_refinement", None)
        if run is None:
            _check_dongdaemun_refinement_available()
            raise AttributeError(
                "installed sciscape_leiden graph instance does not expose "
                "Graph.run_leiden_dongdaemun_refinement"
            )
        schedule = (
            None
            if randomness_schedule is None
            else [float(x) for x in randomness_schedule]
        )
        initial = (
            None
            if initial_membership is None
            else np.ascontiguousarray(initial_membership, dtype=np.uint64)
        )
        fixed = (
            None
            if fixed_nodes is None
            else np.ascontiguousarray(fixed_nodes, dtype=np.bool_)
        )
        run_kwargs = {
            "target_max_weight": float(target_max_weight),
            "resolution": float(resolution),
            "n_iterations": int(n_iterations),
            "randomness": float(randomness),
            "randomness_schedule": schedule,
            "seed": int(seed),
            "initial_membership": initial,
            "fixed_nodes": fixed,
            "soft_min_ratio": float(soft_min_ratio),
            "max_extra_parents_per_iteration": int(max_extra_parents_per_iteration),
            "max_extra_children_per_parent": int(max_extra_children_per_parent),
            "max_singleton_weight_fraction": float(max_singleton_weight_fraction),
            "min_largest_child_fraction_improvement": float(
                min_largest_child_fraction_improvement
            ),
            "gamma_multipliers": [float(x) for x in gamma_multipliers],
        }
        if parent_selection_policy != "weight":
            run_kwargs["parent_selection_policy"] = str(parent_selection_policy)
        if seed_perturbations:
            run_kwargs["seed_perturbations"] = int(seed_perturbations)
        if use_quotient_diagnostic:
            run_kwargs["use_quotient_diagnostic"] = True
        if use_baseline_repair:
            run_kwargs["use_baseline_repair"] = True
        if baseline_repair_policy != "replace":
            run_kwargs["baseline_repair_policy"] = str(baseline_repair_policy)
        if baseline_repair_replace_min_parent_ratio != 1.05:
            run_kwargs["baseline_repair_replace_min_parent_ratio"] = float(
                baseline_repair_replace_min_parent_ratio
            )
        if baseline_repair_epsilon:
            run_kwargs["baseline_repair_epsilon"] = float(baseline_repair_epsilon)
        if candidate_quality_policy != "structural":
            run_kwargs["candidate_quality_policy"] = str(candidate_quality_policy)
        if min_candidate_delta_q:
            run_kwargs["min_candidate_delta_q"] = float(min_candidate_delta_q)
        if adaptive_plateau_quality_band:
            run_kwargs["adaptive_plateau_quality_band"] = float(
                adaptive_plateau_quality_band
            )
        if use_final_quality_guard:
            run_kwargs["use_final_quality_guard"] = True
        if min_final_quality_delta:
            run_kwargs["min_final_quality_delta"] = float(min_final_quality_delta)
        if adaptive_probe_mode != "off":
            run_kwargs["adaptive_probe_mode"] = str(adaptive_probe_mode)
        if adaptive_probe_perturbations:
            run_kwargs["adaptive_probe_perturbations"] = int(
                adaptive_probe_perturbations
            )
        if adaptive_probe_targets:
            run_kwargs["adaptive_probe_targets"] = [
                str(x) for x in adaptive_probe_targets
            ]
        if adaptive_probe_tolerance_parent_weight != 1e-6:
            run_kwargs["adaptive_probe_tolerance_parent_weight"] = float(
                adaptive_probe_tolerance_parent_weight
            )
        if adaptive_probe_include_node_order_control:
            run_kwargs["adaptive_probe_include_node_order_control"] = True
        if adaptive_probe_commit_min_gain_parent_weight:
            run_kwargs["adaptive_probe_commit_min_gain_parent_weight"] = float(
                adaptive_probe_commit_min_gain_parent_weight
            )
        if adaptive_probe_max_commits_total:
            run_kwargs["adaptive_probe_max_commits_total"] = int(
                adaptive_probe_max_commits_total
            )
        if adaptive_probe_max_commits_per_depth:
            run_kwargs["adaptive_probe_max_commits_per_depth"] = int(
                adaptive_probe_max_commits_per_depth
            )
        if adaptive_probe_commit_sources:
            run_kwargs["adaptive_probe_commit_sources"] = [
                str(x) for x in adaptive_probe_commit_sources
            ]
        if adaptive_probe_commit_strategy != "online_first":
            run_kwargs["adaptive_probe_commit_strategy"] = str(
                adaptive_probe_commit_strategy
            )
        if adaptive_near_tie_probe_mode not in {
            "off",
            "trace_only",
            "candidate",
            "qf_replace",
            "near_tie_qf_replace",
        }:
            raise ValueError(
                "adaptive_near_tie_probe_mode must be 'off', 'trace_only', "
                "'candidate', 'qf_replace', or 'near_tie_qf_replace'"
            )
        if adaptive_near_tie_probe_mode != "off":
            run_kwargs["adaptive_near_tie_probe_mode"] = str(
                adaptive_near_tie_probe_mode
            )
        if adaptive_near_tie_margin_parent_weight:
            run_kwargs["adaptive_near_tie_margin_parent_weight"] = float(
                adaptive_near_tie_margin_parent_weight
            )
        if adaptive_near_tie_randomness:
            run_kwargs["adaptive_near_tie_randomness"] = float(
                adaptive_near_tie_randomness
            )
        if adaptive_near_tie_max_decisions_per_parent:
            run_kwargs["adaptive_near_tie_max_decisions_per_parent"] = int(
                adaptive_near_tie_max_decisions_per_parent
            )
        valid_local_shake_modes = {
            "off",
            "trace_only",
            "qf_replace",
            "pressure_guarded",
        }
        if adaptive_local_shake_mode not in valid_local_shake_modes:
            raise ValueError(
                "adaptive_local_shake_mode must be 'off', 'trace_only', "
                "'qf_replace', or 'pressure_guarded'"
            )
        valid_local_shake_arms = {
            "near_tie_refinement",
            "resolution_up",
            "resolution_down",
            "seed_local_refinement",
        }
        local_shake_arms = tuple(str(x) for x in adaptive_local_shake_arms)
        bad_arms = [
            arm for arm in local_shake_arms if arm not in valid_local_shake_arms
        ]
        if bad_arms:
            raise ValueError(
                "adaptive_local_shake_arms contains unsupported arm(s): "
                + ", ".join(bad_arms)
            )
        local_shake_arm_priority = tuple(
            str(x) for x in adaptive_local_shake_arm_priority
        )
        bad_priority = [
            arm for arm in local_shake_arm_priority if arm not in valid_local_shake_arms
        ]
        if bad_priority:
            raise ValueError(
                "adaptive_local_shake_arm_priority contains unsupported arm(s): "
                + ", ".join(bad_priority)
            )
        if adaptive_local_shake_mode != "off" and not local_shake_arms:
            raise ValueError(
                "adaptive_local_shake_arms must not be empty when "
                "adaptive_local_shake_mode is not 'off'"
            )
        if adaptive_local_shake_final_guard_mode not in {"none", "runner_audit"}:
            if adaptive_local_shake_final_guard_mode == "quality_guard":
                raise ValueError(
                    "adaptive_local_shake_final_guard_mode='quality_guard' is not "
                    "implemented in v1; use 'none' or 'runner_audit'"
                )
            raise ValueError(
                "adaptive_local_shake_final_guard_mode must be 'none' or 'runner_audit'"
            )
        if adaptive_local_shake_mode != "off":
            run_kwargs["adaptive_local_shake_mode"] = str(adaptive_local_shake_mode)
        if local_shake_arms:
            run_kwargs["adaptive_local_shake_arms"] = list(local_shake_arms)
        if adaptive_local_shake_max_arms_per_parent:
            run_kwargs["adaptive_local_shake_max_arms_per_parent"] = int(
                adaptive_local_shake_max_arms_per_parent
            )
        if adaptive_local_shake_max_candidates_per_parent:
            run_kwargs["adaptive_local_shake_max_candidates_per_parent"] = int(
                adaptive_local_shake_max_candidates_per_parent
            )
        if adaptive_local_shake_min_gain_parent_weight:
            run_kwargs["adaptive_local_shake_min_gain_parent_weight"] = float(
                adaptive_local_shake_min_gain_parent_weight
            )
        if adaptive_local_shake_shape_eps != 1e-12:
            run_kwargs["adaptive_local_shake_shape_eps"] = float(
                adaptive_local_shake_shape_eps
            )
        if local_shake_arm_priority:
            run_kwargs["adaptive_local_shake_arm_priority"] = list(
                local_shake_arm_priority
            )
        if adaptive_local_shake_near_tie_min_count != 1:
            run_kwargs["adaptive_local_shake_near_tie_min_count"] = int(
                adaptive_local_shake_near_tie_min_count
            )
        if adaptive_local_shake_resolution_down_multipliers:
            run_kwargs["adaptive_local_shake_resolution_down_multipliers"] = [
                float(x) for x in adaptive_local_shake_resolution_down_multipliers
            ]
        if adaptive_local_shake_resolution_up_multipliers:
            run_kwargs["adaptive_local_shake_resolution_up_multipliers"] = [
                float(x) for x in adaptive_local_shake_resolution_up_multipliers
            ]
        if adaptive_local_shake_resolution_up_min_parent_ratio != 1.0:
            run_kwargs["adaptive_local_shake_resolution_up_min_parent_ratio"] = float(
                adaptive_local_shake_resolution_up_min_parent_ratio
            )
        if adaptive_local_shake_resolution_down_max_parent_ratio != 1.0:
            run_kwargs["adaptive_local_shake_resolution_down_max_parent_ratio"] = float(
                adaptive_local_shake_resolution_down_max_parent_ratio
            )
        if adaptive_local_shake_large_child_fraction != 0.95:
            run_kwargs["adaptive_local_shake_large_child_fraction"] = float(
                adaptive_local_shake_large_child_fraction
            )
        if adaptive_local_shake_singleton_fraction != 0.05:
            run_kwargs["adaptive_local_shake_singleton_fraction"] = float(
                adaptive_local_shake_singleton_fraction
            )
        if adaptive_local_shake_seed_perturbations:
            run_kwargs["adaptive_local_shake_seed_perturbations"] = int(
                adaptive_local_shake_seed_perturbations
            )
        if adaptive_local_shake_seed_margin_count != 2:
            run_kwargs["adaptive_local_shake_seed_margin_count"] = int(
                adaptive_local_shake_seed_margin_count
            )
        if adaptive_local_shake_near_tie_margin_parent_weight:
            run_kwargs["adaptive_local_shake_near_tie_margin_parent_weight"] = float(
                adaptive_local_shake_near_tie_margin_parent_weight
            )
        if adaptive_local_shake_near_tie_randomness:
            run_kwargs["adaptive_local_shake_near_tie_randomness"] = float(
                adaptive_local_shake_near_tie_randomness
            )
        if adaptive_local_shake_final_guard_mode != "none":
            run_kwargs["adaptive_local_shake_final_guard_mode"] = str(
                adaptive_local_shake_final_guard_mode
            )
        raw = run(**run_kwargs)
        iteration_depth = np.asarray(raw["iteration_depth"], dtype=np.uint64)
        zero_iteration = np.zeros_like(iteration_depth)
        zero_iteration_float = np.zeros(iteration_depth.shape, dtype=np.float64)
        audit = RustDongdaemunRefinementAudit(
            enabled=bool(raw["audit_enabled"]),
            selected_parent_count_total=int(raw["selected_parent_count_total"]),
            applied_parent_count_total=int(raw["applied_parent_count_total"]),
            rejected_candidate_count_total=int(raw["rejected_candidate_count_total"]),
            added_refined_clusters_total=int(raw["added_refined_clusters_total"]),
            same_gamma_candidates_total=int(raw.get("same_gamma_candidates_total", 0)),
            high_gamma_candidates_total=int(raw.get("high_gamma_candidates_total", 0)),
            same_gamma_applied_total=int(raw.get("same_gamma_applied_total", 0)),
            high_gamma_applied_total=int(raw.get("high_gamma_applied_total", 0)),
            quotient_candidates_total=int(raw.get("quotient_candidates_total", 0)),
            quotient_positive_candidates_total=int(
                raw.get("quotient_positive_candidates_total", 0)
            ),
            quotient_selected_total=int(raw.get("quotient_selected_total", 0)),
            quotient_score_sum=float(raw.get("quotient_score_sum", 0.0)),
            baseline_repair_candidates_total=int(
                raw.get("baseline_repair_candidates_total", 0)
            ),
            baseline_repair_improved_candidates_total=int(
                raw.get("baseline_repair_improved_candidates_total", 0)
            ),
            baseline_repair_selected_total=int(
                raw.get("baseline_repair_selected_total", 0)
            ),
            baseline_repair_merge_count_total=int(
                raw.get("baseline_repair_merge_count_total", 0)
            ),
            baseline_repair_delta_sum=float(raw.get("baseline_repair_delta_sum", 0.0)),
            candidate_quality_delta_sum=float(
                raw.get("candidate_quality_delta_sum", 0.0)
            ),
            candidate_positive_quality_delta_total=int(
                raw.get("candidate_positive_quality_delta_total", 0)
            ),
            candidate_selected_positive_quality_delta_total=int(
                raw.get("candidate_selected_positive_quality_delta_total", 0)
            ),
            candidate_rejected_by_quality_total=int(
                raw.get("candidate_rejected_by_quality_total", 0)
            ),
            same_gamma_quality_delta_sum=float(
                raw.get("same_gamma_quality_delta_sum", 0.0)
            ),
            high_gamma_quality_delta_sum=float(
                raw.get("high_gamma_quality_delta_sum", 0.0)
            ),
            same_gamma_positive_quality_delta_total=int(
                raw.get("same_gamma_positive_quality_delta_total", 0)
            ),
            high_gamma_positive_quality_delta_total=int(
                raw.get("high_gamma_positive_quality_delta_total", 0)
            ),
            same_gamma_selected_positive_quality_delta_total=int(
                raw.get("same_gamma_selected_positive_quality_delta_total", 0)
            ),
            high_gamma_selected_positive_quality_delta_total=int(
                raw.get("high_gamma_selected_positive_quality_delta_total", 0)
            ),
            same_gamma_rejected_by_quality_total=int(
                raw.get("same_gamma_rejected_by_quality_total", 0)
            ),
            high_gamma_rejected_by_quality_total=int(
                raw.get("high_gamma_rejected_by_quality_total", 0)
            ),
            candidate_valid_total=int(raw.get("candidate_valid_total", 0)),
            candidate_invalid_total=int(raw.get("candidate_invalid_total", 0)),
            candidate_rejected_by_policy_total=int(
                raw.get("candidate_rejected_by_policy_total", 0)
            ),
            same_gamma_valid_total=int(raw.get("same_gamma_valid_total", 0)),
            high_gamma_valid_total=int(raw.get("high_gamma_valid_total", 0)),
            same_gamma_invalid_total=int(raw.get("same_gamma_invalid_total", 0)),
            high_gamma_invalid_total=int(raw.get("high_gamma_invalid_total", 0)),
            same_gamma_rejected_by_policy_total=int(
                raw.get("same_gamma_rejected_by_policy_total", 0)
            ),
            high_gamma_rejected_by_policy_total=int(
                raw.get("high_gamma_rejected_by_policy_total", 0)
            ),
            candidate_qpos_spos_total=int(raw.get("candidate_qpos_spos_total", 0)),
            candidate_qpos_sneg_total=int(raw.get("candidate_qpos_sneg_total", 0)),
            candidate_qneg_spos_total=int(raw.get("candidate_qneg_spos_total", 0)),
            candidate_qneg_sneg_total=int(raw.get("candidate_qneg_sneg_total", 0)),
            same_gamma_qpos_spos_total=int(raw.get("same_gamma_qpos_spos_total", 0)),
            same_gamma_qpos_sneg_total=int(raw.get("same_gamma_qpos_sneg_total", 0)),
            same_gamma_qneg_spos_total=int(raw.get("same_gamma_qneg_spos_total", 0)),
            same_gamma_qneg_sneg_total=int(raw.get("same_gamma_qneg_sneg_total", 0)),
            high_gamma_qpos_spos_total=int(raw.get("high_gamma_qpos_spos_total", 0)),
            high_gamma_qpos_sneg_total=int(raw.get("high_gamma_qpos_sneg_total", 0)),
            high_gamma_qneg_spos_total=int(raw.get("high_gamma_qneg_spos_total", 0)),
            high_gamma_qneg_sneg_total=int(raw.get("high_gamma_qneg_sneg_total", 0)),
            candidate_true_positive_total=int(
                raw.get("candidate_true_positive_total", 0)
            ),
            candidate_false_positive_total=int(
                raw.get("candidate_false_positive_total", 0)
            ),
            candidate_false_negative_total=int(
                raw.get("candidate_false_negative_total", 0)
            ),
            candidate_true_negative_total=int(
                raw.get("candidate_true_negative_total", 0)
            ),
            adaptive_local_shake_triggers_total=int(
                raw.get("adaptive_local_shake_triggers_total", 0)
            ),
            adaptive_local_shake_candidates_total=int(
                raw.get("adaptive_local_shake_candidates_total", 0)
            ),
            adaptive_local_shake_commits_total=int(
                raw.get("adaptive_local_shake_commits_total", 0)
            ),
            adaptive_local_shake_qf_gain_sum=float(
                raw.get("adaptive_local_shake_qf_gain_sum", 0.0)
            ),
            final_quality_guard_enabled=bool(
                raw.get("final_quality_guard_enabled", False)
            ),
            final_quality_guard_triggered=bool(
                raw.get("final_quality_guard_triggered", False)
            ),
            final_quality_guard_standard_quality=float(
                raw.get("final_quality_guard_standard_quality", 0.0)
            ),
            final_quality_guard_pre_guard_quality=float(
                raw.get("final_quality_guard_pre_guard_quality", 0.0)
            ),
            final_quality_delta_vs_guard_standard=float(
                raw.get("final_quality_delta_vs_guard_standard", 0.0)
            ),
            max_parent_weight_seen=float(raw["max_parent_weight_seen"]),
            iteration_depth=iteration_depth,
            iteration_selected_parents=np.asarray(
                raw["iteration_selected_parents"], dtype=np.uint64
            ),
            iteration_applied_parents=np.asarray(
                raw["iteration_applied_parents"], dtype=np.uint64
            ),
            iteration_same_gamma_candidates=np.asarray(
                raw.get("iteration_same_gamma_candidates", zero_iteration),
                dtype=np.uint64,
            ),
            iteration_high_gamma_candidates=np.asarray(
                raw.get("iteration_high_gamma_candidates", zero_iteration),
                dtype=np.uint64,
            ),
            iteration_same_gamma_applied=np.asarray(
                raw.get("iteration_same_gamma_applied", zero_iteration), dtype=np.uint64
            ),
            iteration_high_gamma_applied=np.asarray(
                raw.get("iteration_high_gamma_applied", zero_iteration), dtype=np.uint64
            ),
            iteration_quotient_candidates=np.asarray(
                raw.get("iteration_quotient_candidates", zero_iteration),
                dtype=np.uint64,
            ),
            iteration_quotient_positive_candidates=np.asarray(
                raw.get("iteration_quotient_positive_candidates", zero_iteration),
                dtype=np.uint64,
            ),
            iteration_quotient_selected=np.asarray(
                raw.get("iteration_quotient_selected", zero_iteration), dtype=np.uint64
            ),
            iteration_quotient_score_sum=np.asarray(
                raw.get("iteration_quotient_score_sum", zero_iteration_float),
                dtype=np.float64,
            ),
            iteration_baseline_repair_candidates=np.asarray(
                raw.get("iteration_baseline_repair_candidates", zero_iteration),
                dtype=np.uint64,
            ),
            iteration_baseline_repair_improved_candidates=np.asarray(
                raw.get(
                    "iteration_baseline_repair_improved_candidates", zero_iteration
                ),
                dtype=np.uint64,
            ),
            iteration_baseline_repair_selected=np.asarray(
                raw.get("iteration_baseline_repair_selected", zero_iteration),
                dtype=np.uint64,
            ),
            iteration_baseline_repair_merge_count=np.asarray(
                raw.get("iteration_baseline_repair_merge_count", zero_iteration),
                dtype=np.uint64,
            ),
            iteration_baseline_repair_delta_sum=np.asarray(
                raw.get("iteration_baseline_repair_delta_sum", zero_iteration_float),
                dtype=np.float64,
            ),
            iteration_candidate_quality_delta_sum=np.asarray(
                raw.get("iteration_candidate_quality_delta_sum", zero_iteration_float),
                dtype=np.float64,
            ),
            iteration_candidate_positive_quality_delta=np.asarray(
                raw.get("iteration_candidate_positive_quality_delta", zero_iteration),
                dtype=np.uint64,
            ),
            iteration_candidate_selected_positive_quality_delta=np.asarray(
                raw.get(
                    "iteration_candidate_selected_positive_quality_delta",
                    zero_iteration,
                ),
                dtype=np.uint64,
            ),
            iteration_candidate_rejected_by_quality=np.asarray(
                raw.get("iteration_candidate_rejected_by_quality", zero_iteration),
                dtype=np.uint64,
            ),
            iteration_same_gamma_quality_delta_sum=np.asarray(
                raw.get("iteration_same_gamma_quality_delta_sum", zero_iteration_float),
                dtype=np.float64,
            ),
            iteration_high_gamma_quality_delta_sum=np.asarray(
                raw.get("iteration_high_gamma_quality_delta_sum", zero_iteration_float),
                dtype=np.float64,
            ),
            iteration_same_gamma_positive_quality_delta=np.asarray(
                raw.get("iteration_same_gamma_positive_quality_delta", zero_iteration),
                dtype=np.uint64,
            ),
            iteration_high_gamma_positive_quality_delta=np.asarray(
                raw.get("iteration_high_gamma_positive_quality_delta", zero_iteration),
                dtype=np.uint64,
            ),
            iteration_same_gamma_selected_positive_quality_delta=np.asarray(
                raw.get(
                    "iteration_same_gamma_selected_positive_quality_delta",
                    zero_iteration,
                ),
                dtype=np.uint64,
            ),
            iteration_high_gamma_selected_positive_quality_delta=np.asarray(
                raw.get(
                    "iteration_high_gamma_selected_positive_quality_delta",
                    zero_iteration,
                ),
                dtype=np.uint64,
            ),
            iteration_same_gamma_rejected_by_quality=np.asarray(
                raw.get("iteration_same_gamma_rejected_by_quality", zero_iteration),
                dtype=np.uint64,
            ),
            iteration_high_gamma_rejected_by_quality=np.asarray(
                raw.get("iteration_high_gamma_rejected_by_quality", zero_iteration),
                dtype=np.uint64,
            ),
            iteration_candidate_valid=np.asarray(
                raw.get("iteration_candidate_valid", zero_iteration),
                dtype=np.uint64,
            ),
            iteration_candidate_invalid=np.asarray(
                raw.get("iteration_candidate_invalid", zero_iteration),
                dtype=np.uint64,
            ),
            iteration_candidate_rejected_by_policy=np.asarray(
                raw.get("iteration_candidate_rejected_by_policy", zero_iteration),
                dtype=np.uint64,
            ),
            iteration_same_gamma_valid=np.asarray(
                raw.get("iteration_same_gamma_valid", zero_iteration),
                dtype=np.uint64,
            ),
            iteration_high_gamma_valid=np.asarray(
                raw.get("iteration_high_gamma_valid", zero_iteration),
                dtype=np.uint64,
            ),
            iteration_same_gamma_invalid=np.asarray(
                raw.get("iteration_same_gamma_invalid", zero_iteration),
                dtype=np.uint64,
            ),
            iteration_high_gamma_invalid=np.asarray(
                raw.get("iteration_high_gamma_invalid", zero_iteration),
                dtype=np.uint64,
            ),
            iteration_same_gamma_rejected_by_policy=np.asarray(
                raw.get("iteration_same_gamma_rejected_by_policy", zero_iteration),
                dtype=np.uint64,
            ),
            iteration_high_gamma_rejected_by_policy=np.asarray(
                raw.get("iteration_high_gamma_rejected_by_policy", zero_iteration),
                dtype=np.uint64,
            ),
            iteration_candidate_qpos_spos=np.asarray(
                raw.get("iteration_candidate_qpos_spos", zero_iteration),
                dtype=np.uint64,
            ),
            iteration_candidate_qpos_sneg=np.asarray(
                raw.get("iteration_candidate_qpos_sneg", zero_iteration),
                dtype=np.uint64,
            ),
            iteration_candidate_qneg_spos=np.asarray(
                raw.get("iteration_candidate_qneg_spos", zero_iteration),
                dtype=np.uint64,
            ),
            iteration_candidate_qneg_sneg=np.asarray(
                raw.get("iteration_candidate_qneg_sneg", zero_iteration),
                dtype=np.uint64,
            ),
            iteration_same_gamma_qpos_spos=np.asarray(
                raw.get("iteration_same_gamma_qpos_spos", zero_iteration),
                dtype=np.uint64,
            ),
            iteration_same_gamma_qpos_sneg=np.asarray(
                raw.get("iteration_same_gamma_qpos_sneg", zero_iteration),
                dtype=np.uint64,
            ),
            iteration_same_gamma_qneg_spos=np.asarray(
                raw.get("iteration_same_gamma_qneg_spos", zero_iteration),
                dtype=np.uint64,
            ),
            iteration_same_gamma_qneg_sneg=np.asarray(
                raw.get("iteration_same_gamma_qneg_sneg", zero_iteration),
                dtype=np.uint64,
            ),
            iteration_high_gamma_qpos_spos=np.asarray(
                raw.get("iteration_high_gamma_qpos_spos", zero_iteration),
                dtype=np.uint64,
            ),
            iteration_high_gamma_qpos_sneg=np.asarray(
                raw.get("iteration_high_gamma_qpos_sneg", zero_iteration),
                dtype=np.uint64,
            ),
            iteration_high_gamma_qneg_spos=np.asarray(
                raw.get("iteration_high_gamma_qneg_spos", zero_iteration),
                dtype=np.uint64,
            ),
            iteration_high_gamma_qneg_sneg=np.asarray(
                raw.get("iteration_high_gamma_qneg_sneg", zero_iteration),
                dtype=np.uint64,
            ),
            iteration_candidate_true_positive=np.asarray(
                raw.get("iteration_candidate_true_positive", zero_iteration),
                dtype=np.uint64,
            ),
            iteration_candidate_false_positive=np.asarray(
                raw.get("iteration_candidate_false_positive", zero_iteration),
                dtype=np.uint64,
            ),
            iteration_candidate_false_negative=np.asarray(
                raw.get("iteration_candidate_false_negative", zero_iteration),
                dtype=np.uint64,
            ),
            iteration_candidate_true_negative=np.asarray(
                raw.get("iteration_candidate_true_negative", zero_iteration),
                dtype=np.uint64,
            ),
            iteration_adaptive_local_shake_triggers=np.asarray(
                raw.get("iteration_adaptive_local_shake_triggers", zero_iteration),
                dtype=np.uint64,
            ),
            iteration_adaptive_local_shake_candidates=np.asarray(
                raw.get("iteration_adaptive_local_shake_candidates", zero_iteration),
                dtype=np.uint64,
            ),
            iteration_adaptive_local_shake_commits=np.asarray(
                raw.get("iteration_adaptive_local_shake_commits", zero_iteration),
                dtype=np.uint64,
            ),
            iteration_adaptive_local_shake_qf_gain_sum=np.asarray(
                raw.get(
                    "iteration_adaptive_local_shake_qf_gain_sum",
                    zero_iteration_float,
                ),
                dtype=np.float64,
            ),
            iteration_standard_refined_clusters=np.asarray(
                raw["iteration_standard_refined_clusters"], dtype=np.uint64
            ),
            iteration_final_refined_clusters=np.asarray(
                raw["iteration_final_refined_clusters"], dtype=np.uint64
            ),
        )
        return RustDongdaemunRefinementResult(
            membership=np.asarray(raw["membership"], dtype=np.uint64),
            quality=float(raw["quality"]),
            n_clusters=int(raw["n_clusters"]),
            n_iterations_used=int(raw["n_iterations_used"]),
            audit=audit,
        )

    def _membership_weight_pressure(
        self,
        membership: np.ndarray,
        *,
        target_max_weight: float,
    ) -> tuple[float, float, int]:
        labels = np.asarray(membership, dtype=np.uint64)
        if labels.size == 0:
            return 0.0, 0.0, 0
        weights = (
            np.ones(labels.shape[0], dtype=np.float64)
            if self.node_weights is None
            else np.asarray(self.node_weights, dtype=np.float64)
        )
        if weights.shape[0] != labels.shape[0]:
            raise ValueError(
                "node_weights length must match membership length for auto-fast pressure"
            )
        cluster_weights = np.bincount(labels, weights=weights)
        max_weight = float(cluster_weights.max(initial=0.0))
        target = float(target_max_weight)
        ratio = 0.0 if target <= 0.0 else max_weight / target
        n_above = int(np.count_nonzero(cluster_weights > target))
        return max_weight, ratio, n_above

    @staticmethod
    def _pressure_triggered(
        *,
        max_doc_weight_ratio: float,
        n_above_max_doc_weight: int,
        trigger_max_doc_weight_ratio: float | None,
        trigger_min_above_max_doc_weight: int | None,
    ) -> bool:
        checks: list[bool] = []
        if trigger_max_doc_weight_ratio is not None:
            checks.append(
                float(max_doc_weight_ratio) > float(trigger_max_doc_weight_ratio)
            )
        if trigger_min_above_max_doc_weight is not None:
            checks.append(
                int(n_above_max_doc_weight) >= int(trigger_min_above_max_doc_weight)
            )
        return any(checks) if checks else True

    def run_leiden_dongdaemun_auto_fast_refinement(
        self,
        *,
        target_max_weight: float,
        resolution: float,
        seed: int = 0,
        n_iterations: int = 10,
        randomness: float = 0.01,
        randomness_schedule: Sequence[float] | None = None,
        initial_membership: np.ndarray | None = None,
        fixed_nodes: np.ndarray | None = None,
        trigger_max_doc_weight_ratio: float | None = 1.03,
        trigger_min_above_max_doc_weight: int | None = 2,
        accept_max_doc_weight_ratio: float = 1.01,
        accept_min_quality_delta: float | None = None,
        accept_min_quality_delta_ratio: float | None = None,
        max_extra_parents_per_iteration: int = 4,
        max_extra_children_per_parent: int = 16,
        parent_selection_policy: str = "weight",
        severe_trigger_max_doc_weight_ratio: float | None = None,
        severe_trigger_min_above_max_doc_weight: int | None = None,
        severe_max_extra_parents_per_iteration: int | None = None,
        severe_max_extra_children_per_parent: int | None = None,
        soft_min_ratio: float = 1.0,
        max_singleton_weight_fraction: float = 0.05,
        min_largest_child_fraction_improvement: float = 0.05,
        gamma_multipliers: Sequence[float] = (1.02, 1.05),
        seed_perturbations: int = 0,
        use_quotient_diagnostic: bool = False,
        candidate_quality_policy: str = "structural",
        min_candidate_delta_q: float = 0.0,
        adaptive_plateau_quality_band: float = 0.0,
        allow_repair_escalation: bool = False,
        repair_escalation_trigger_max_doc_weight_ratio: float | None = None,
        repair_escalation_trigger_min_above_max_doc_weight: int | None = None,
        repair_escalation_accept_max_doc_weight_ratio: float | None = 1.01,
        repair_escalation_min_quality_delta: float = 0.0,
        baseline_repair_policy: str = "adaptive",
        baseline_repair_replace_min_parent_ratio: float = 1.05,
        baseline_repair_epsilon: float = 0.0,
    ) -> RustDongdaemunAutoFastResult:
        """Run pressure-triggered Dongdaemun refinement with cheap fallback.

        This is an opt-in convenience wrapper around standard Rust Leiden plus
        integrated Dongdaemun refinement. It first runs standard Leiden,
        measures max-weight pressure, and only runs the bounded repair-off
        refinement path when pressure is high enough. The selected refinement is
        accepted only if its max cluster weight stays within a structural soft
        guard relative to the standard result.
        """
        standard = self.run_leiden(
            resolution=resolution,
            seed=seed,
            n_iterations=n_iterations,
            randomness=randomness,
            randomness_schedule=randomness_schedule,
            initial_membership=initial_membership,
            fixed_nodes=fixed_nodes,
        )
        (
            standard_max_weight,
            standard_max_ratio,
            standard_n_above,
        ) = self._membership_weight_pressure(
            standard.membership,
            target_max_weight=target_max_weight,
        )
        triggered = self._pressure_triggered(
            max_doc_weight_ratio=standard_max_ratio,
            n_above_max_doc_weight=standard_n_above,
            trigger_max_doc_weight_ratio=trigger_max_doc_weight_ratio,
            trigger_min_above_max_doc_weight=trigger_min_above_max_doc_weight,
        )
        if not triggered:
            return RustDongdaemunAutoFastResult(
                membership=standard.membership,
                quality=standard.quality,
                n_clusters=standard.n_clusters,
                selected_variant="standard",
                triggered=False,
                fallback_triggered=True,
                fallback_reason="trigger_not_met",
                severe_tier_triggered=False,
                repair_escalated=False,
                repair_escalation_accepted=False,
                max_extra_parents_per_iteration=int(max_extra_parents_per_iteration),
                max_extra_children_per_parent=int(max_extra_children_per_parent),
                standard_max_doc_weight=standard_max_weight,
                standard_max_doc_weight_ratio=standard_max_ratio,
                standard_n_above_max_doc_weight=standard_n_above,
                selected_max_doc_weight=standard_max_weight,
                selected_max_doc_weight_ratio=standard_max_ratio,
                selected_n_above_max_doc_weight=standard_n_above,
                standard=standard,
                repair_off=None,
                repair_on=None,
            )

        severe_tier = (
            self._pressure_triggered(
                max_doc_weight_ratio=standard_max_ratio,
                n_above_max_doc_weight=standard_n_above,
                trigger_max_doc_weight_ratio=severe_trigger_max_doc_weight_ratio,
                trigger_min_above_max_doc_weight=severe_trigger_min_above_max_doc_weight,
            )
            if severe_trigger_max_doc_weight_ratio is not None
            or severe_trigger_min_above_max_doc_weight is not None
            else False
        )
        use_parents = (
            int(severe_max_extra_parents_per_iteration)
            if severe_tier and severe_max_extra_parents_per_iteration is not None
            else int(max_extra_parents_per_iteration)
        )
        use_children = (
            int(severe_max_extra_children_per_parent)
            if severe_tier and severe_max_extra_children_per_parent is not None
            else int(max_extra_children_per_parent)
        )
        repair_off = self.run_leiden_dongdaemun_refinement(
            target_max_weight=target_max_weight,
            resolution=resolution,
            seed=seed,
            n_iterations=n_iterations,
            randomness=randomness,
            randomness_schedule=randomness_schedule,
            initial_membership=initial_membership,
            fixed_nodes=fixed_nodes,
            soft_min_ratio=soft_min_ratio,
            max_extra_parents_per_iteration=use_parents,
            max_extra_children_per_parent=use_children,
            parent_selection_policy=parent_selection_policy,
            max_singleton_weight_fraction=max_singleton_weight_fraction,
            min_largest_child_fraction_improvement=(
                min_largest_child_fraction_improvement
            ),
            gamma_multipliers=gamma_multipliers,
            seed_perturbations=seed_perturbations,
            use_quotient_diagnostic=use_quotient_diagnostic,
            use_baseline_repair=False,
            candidate_quality_policy=candidate_quality_policy,
            min_candidate_delta_q=min_candidate_delta_q,
            adaptive_plateau_quality_band=adaptive_plateau_quality_band,
        )
        (
            selected_max_weight,
            selected_max_ratio,
            selected_n_above,
        ) = self._membership_weight_pressure(
            repair_off.membership,
            target_max_weight=target_max_weight,
        )
        accept_limit = standard_max_weight * float(accept_max_doc_weight_ratio)
        if selected_max_weight > accept_limit:
            return RustDongdaemunAutoFastResult(
                membership=standard.membership,
                quality=standard.quality,
                n_clusters=standard.n_clusters,
                selected_variant="standard",
                triggered=True,
                fallback_triggered=True,
                fallback_reason="max_doc_weight_guard",
                severe_tier_triggered=severe_tier,
                repair_escalated=False,
                repair_escalation_accepted=False,
                max_extra_parents_per_iteration=use_parents,
                max_extra_children_per_parent=use_children,
                standard_max_doc_weight=standard_max_weight,
                standard_max_doc_weight_ratio=standard_max_ratio,
                standard_n_above_max_doc_weight=standard_n_above,
                selected_max_doc_weight=standard_max_weight,
                selected_max_doc_weight_ratio=standard_max_ratio,
                selected_n_above_max_doc_weight=standard_n_above,
                standard=standard,
                repair_off=repair_off,
                repair_on=None,
            )
        quality_accept_delta = 0.0
        if accept_min_quality_delta is not None:
            quality_accept_delta += float(accept_min_quality_delta)
        if accept_min_quality_delta_ratio is not None:
            quality_accept_delta += abs(float(standard.quality)) * float(
                accept_min_quality_delta_ratio
            )
        if (
            accept_min_quality_delta is not None
            or accept_min_quality_delta_ratio is not None
        ) and float(repair_off.quality) < float(
            standard.quality
        ) + quality_accept_delta:
            return RustDongdaemunAutoFastResult(
                membership=standard.membership,
                quality=standard.quality,
                n_clusters=standard.n_clusters,
                selected_variant="standard",
                triggered=True,
                fallback_triggered=True,
                fallback_reason="quality_guard",
                severe_tier_triggered=severe_tier,
                repair_escalated=False,
                repair_escalation_accepted=False,
                max_extra_parents_per_iteration=use_parents,
                max_extra_children_per_parent=use_children,
                standard_max_doc_weight=standard_max_weight,
                standard_max_doc_weight_ratio=standard_max_ratio,
                standard_n_above_max_doc_weight=standard_n_above,
                selected_max_doc_weight=standard_max_weight,
                selected_max_doc_weight_ratio=standard_max_ratio,
                selected_n_above_max_doc_weight=standard_n_above,
                standard=standard,
                repair_off=repair_off,
                repair_on=None,
            )

        selected_variant = "refine_repair_off"
        selected_membership = repair_off.membership
        selected_quality = repair_off.quality
        selected_n_clusters = repair_off.n_clusters
        repair_on: RustDongdaemunRefinementResult | None = None
        repair_escalated = False
        repair_escalation_accepted = False
        escalation_pressure = (
            self._pressure_triggered(
                max_doc_weight_ratio=selected_max_ratio,
                n_above_max_doc_weight=selected_n_above,
                trigger_max_doc_weight_ratio=(
                    repair_escalation_trigger_max_doc_weight_ratio
                ),
                trigger_min_above_max_doc_weight=(
                    repair_escalation_trigger_min_above_max_doc_weight
                ),
            )
            if repair_escalation_trigger_max_doc_weight_ratio is not None
            or repair_escalation_trigger_min_above_max_doc_weight is not None
            else True
        )
        if allow_repair_escalation and escalation_pressure:
            repair_escalated = True
            repair_on = self.run_leiden_dongdaemun_refinement(
                target_max_weight=target_max_weight,
                resolution=resolution,
                seed=seed,
                n_iterations=n_iterations,
                randomness=randomness,
                randomness_schedule=randomness_schedule,
                initial_membership=initial_membership,
                fixed_nodes=fixed_nodes,
                soft_min_ratio=soft_min_ratio,
                max_extra_parents_per_iteration=use_parents,
                max_extra_children_per_parent=use_children,
                parent_selection_policy=parent_selection_policy,
                max_singleton_weight_fraction=max_singleton_weight_fraction,
                min_largest_child_fraction_improvement=(
                    min_largest_child_fraction_improvement
                ),
                gamma_multipliers=gamma_multipliers,
                seed_perturbations=seed_perturbations,
                use_quotient_diagnostic=use_quotient_diagnostic,
                use_baseline_repair=True,
                baseline_repair_policy=baseline_repair_policy,
                baseline_repair_replace_min_parent_ratio=(
                    baseline_repair_replace_min_parent_ratio
                ),
                baseline_repair_epsilon=baseline_repair_epsilon,
                candidate_quality_policy=candidate_quality_policy,
                min_candidate_delta_q=min_candidate_delta_q,
                adaptive_plateau_quality_band=adaptive_plateau_quality_band,
            )
            repair_max_weight, repair_max_ratio, repair_n_above = (
                self._membership_weight_pressure(
                    repair_on.membership,
                    target_max_weight=target_max_weight,
                )
            )
            repair_accept_limit = (
                float("inf")
                if repair_escalation_accept_max_doc_weight_ratio is None
                else standard_max_weight
                * float(repair_escalation_accept_max_doc_weight_ratio)
            )
            if (
                repair_on.quality
                >= selected_quality + float(repair_escalation_min_quality_delta)
                and repair_max_weight <= repair_accept_limit
            ):
                repair_escalation_accepted = True
                selected_variant = "refine_repair_on"
                selected_membership = repair_on.membership
                selected_quality = repair_on.quality
                selected_n_clusters = repair_on.n_clusters
                selected_max_weight = repair_max_weight
                selected_max_ratio = repair_max_ratio
                selected_n_above = repair_n_above

        return RustDongdaemunAutoFastResult(
            membership=selected_membership,
            quality=float(selected_quality),
            n_clusters=int(selected_n_clusters),
            selected_variant=selected_variant,
            triggered=True,
            fallback_triggered=False,
            fallback_reason="",
            severe_tier_triggered=severe_tier,
            repair_escalated=repair_escalated,
            repair_escalation_accepted=repair_escalation_accepted,
            max_extra_parents_per_iteration=use_parents,
            max_extra_children_per_parent=use_children,
            standard_max_doc_weight=standard_max_weight,
            standard_max_doc_weight_ratio=standard_max_ratio,
            standard_n_above_max_doc_weight=standard_n_above,
            selected_max_doc_weight=selected_max_weight,
            selected_max_doc_weight_ratio=selected_max_ratio,
            selected_n_above_max_doc_weight=selected_n_above,
            standard=standard,
            repair_off=repair_off,
            repair_on=repair_on,
        )

    def run_leiden_dongdaemun_safe_fast_refinement(
        self,
        *,
        target_max_weight: float,
        resolution: float,
        seed: int = 0,
        n_iterations: int = 10,
        randomness: float = 0.01,
        randomness_schedule: Sequence[float] | None = None,
        initial_membership: np.ndarray | None = None,
        fixed_nodes: np.ndarray | None = None,
        trigger_max_doc_weight_ratio: float | None = 1.03,
        trigger_min_above_max_doc_weight: int | None = 2,
        accept_max_doc_weight_ratio: float = 1.01,
        accept_min_quality_delta: float | None = 0.0,
        accept_min_quality_delta_ratio: float | None = None,
    ) -> RustDongdaemunAutoFastResult:
        """Run the conservative Dongdaemun auto-fast preset.

        This opt-in preset mirrors the current safe experimental choice: only
        run the bounded repair-off path under measurable max-weight pressure,
        reject quality regressions, use the quotient-guided candidate tie-breaker,
        avoid severe/escalation tiers, and keep the low-cost mild high-gamma
        candidate set.
        """
        return self.run_leiden_dongdaemun_auto_fast_refinement(
            target_max_weight=target_max_weight,
            resolution=resolution,
            seed=seed,
            n_iterations=n_iterations,
            randomness=randomness,
            randomness_schedule=randomness_schedule,
            initial_membership=initial_membership,
            fixed_nodes=fixed_nodes,
            trigger_max_doc_weight_ratio=trigger_max_doc_weight_ratio,
            trigger_min_above_max_doc_weight=trigger_min_above_max_doc_weight,
            accept_max_doc_weight_ratio=accept_max_doc_weight_ratio,
            accept_min_quality_delta=accept_min_quality_delta,
            accept_min_quality_delta_ratio=accept_min_quality_delta_ratio,
            max_extra_parents_per_iteration=4,
            max_extra_children_per_parent=16,
            parent_selection_policy="weight",
            gamma_multipliers=(1.02, 1.05),
            use_quotient_diagnostic=True,
            candidate_quality_policy="structural",
            allow_repair_escalation=False,
            baseline_repair_policy="adaptive",
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
            raise AttributeError(
                "installed sciscape_leiden module does not expose Graph.search_resolution"
            )
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
            raise AttributeError(
                "installed sciscape_leiden module does not expose Graph.cpm_quality"
            )
        membership = np.ascontiguousarray(membership, dtype=np.uint64)
        return float(quality(membership=membership, resolution=resolution))

    def non_monotone_group_escape_probe(
        self,
        membership: np.ndarray,
        candidate_clusters: np.ndarray,
        *,
        resolution: float,
        max_candidates: int = 5,
        polish_iterations: int = 5,
        randomness: float = 0.01,
        seed: int = 0,
        min_doc_weight: float = 0.0,
        min_assigned_fraction: float = 0.0,
        min_best_group_fraction: float = 0.0,
        quality_eps: float = 0.0,
        parallel_candidates: bool = False,
        return_membership: bool = True,
    ) -> RustNonMonotoneGroupEscapeResult:
        """Try ranked external-grain group moves and accept only non-loss polish."""
        membership_dtype = np.asarray(membership).dtype
        probe_name = (
            "non_monotone_group_escape_probe_u32"
            if membership_dtype == np.dtype(np.uint32)
            else "non_monotone_group_escape_probe"
        )
        probe = getattr(self.graph, probe_name, None)
        if probe is None and probe_name.endswith("_u32"):
            probe_name = "non_monotone_group_escape_probe"
            probe = getattr(self.graph, probe_name, None)
        if probe is None:
            raise AttributeError(
                "installed sciscape_leiden module does not expose "
                "Graph.non_monotone_group_escape_probe"
            )
        membership = np.ascontiguousarray(
            membership,
            dtype=np.uint32 if probe_name.endswith("_u32") else np.uint64,
        )
        candidate_clusters = np.ascontiguousarray(candidate_clusters, dtype=np.uint64)
        probe_kwargs = {
            "membership": membership,
            "candidate_clusters": candidate_clusters,
            "resolution": float(resolution),
            "max_candidates": int(max_candidates),
            "polish_iterations": int(polish_iterations),
            "randomness": float(randomness),
            "seed": int(seed),
            "min_doc_weight": float(min_doc_weight),
            "min_assigned_fraction": float(min_assigned_fraction),
            "min_best_group_fraction": float(min_best_group_fraction),
            "quality_eps": float(quality_eps),
        }
        if parallel_candidates:
            probe_kwargs["parallel_candidates"] = True
        if not return_membership:
            probe_kwargs["return_membership"] = False
        raw = probe(**probe_kwargs)
        return RustNonMonotoneGroupEscapeResult(
            membership=np.asarray(raw["membership"], dtype=np.uint64),
            quality=float(raw["quality"]),
            accepted=bool(raw["accepted"]),
            candidate_rows=list(raw["candidate_rows"]),
            baseline_quality=float(raw["baseline_quality"]),
            best_delta_q=float(raw["best_delta_q"]),
            elapsed_ms=float(raw["elapsed_ms"]),
            candidate_eval_parallel=bool(raw.get("candidate_eval_parallel", False)),
            candidate_eval_wall_elapsed_ms=float(
                raw.get("candidate_eval_wall_elapsed_ms", raw["elapsed_ms"])
            ),
            candidate_eval_cpu_sum_elapsed_ms=float(
                raw.get(
                    "candidate_eval_cpu_sum_elapsed_ms",
                    sum(
                        float(row.get("elapsed_ms", 0.0))
                        for row in raw["candidate_rows"]
                    ),
                )
            ),
            candidate_eval_parallel_speedup=float(
                raw.get("candidate_eval_parallel_speedup", float("nan"))
            ),
            candidate_eval_parallel_workers=int(
                raw.get("candidate_eval_parallel_workers", 1)
            ),
        )

    def non_monotone_group_escape_multifidelity_probe(
        self,
        membership: np.ndarray,
        candidate_clusters: np.ndarray,
        *,
        resolution: float,
        max_candidates: int = 3,
        prescreen_iterations: int = 1,
        final_iterations: int = 5,
        finalists: int = 1,
        label_full_p5: bool = False,
        randomness: float = 0.01,
        seed: int = 0,
        min_doc_weight: float = 0.0,
        min_assigned_fraction: float = 0.0,
        min_best_group_fraction: float = 0.0,
        quality_eps: float = 0.0,
        return_membership: bool = True,
        approx_polish_labels: bool = False,
        basin_signatures: bool = False,
    ) -> RustNonMonotoneGroupEscapeMultifidelityResult:
        """Evaluate external-grain candidates with p1 prescreen and optional p5 labels."""
        membership_dtype = np.asarray(membership).dtype
        probe_name = (
            "non_monotone_group_escape_multifidelity_probe_u32"
            if membership_dtype == np.dtype(np.uint32)
            else "non_monotone_group_escape_multifidelity_probe"
        )
        probe = getattr(self.graph, probe_name, None)
        if probe is None and probe_name.endswith("_u32"):
            probe_name = "non_monotone_group_escape_multifidelity_probe"
            probe = getattr(self.graph, probe_name, None)
        if probe is None:
            raise AttributeError(
                "installed sciscape_leiden module does not expose "
                "Graph.non_monotone_group_escape_multifidelity_probe"
            )
        membership = np.ascontiguousarray(
            membership,
            dtype=np.uint32 if probe_name.endswith("_u32") else np.uint64,
        )
        candidate_clusters = np.ascontiguousarray(candidate_clusters, dtype=np.uint64)
        probe_kwargs = {
            "membership": membership,
            "candidate_clusters": candidate_clusters,
            "resolution": float(resolution),
            "max_candidates": int(max_candidates),
            "prescreen_iterations": int(prescreen_iterations),
            "final_iterations": int(final_iterations),
            "finalists": int(finalists),
            "label_full_p5": bool(label_full_p5),
            "randomness": float(randomness),
            "seed": int(seed),
            "min_doc_weight": float(min_doc_weight),
            "min_assigned_fraction": float(min_assigned_fraction),
            "min_best_group_fraction": float(min_best_group_fraction),
            "quality_eps": float(quality_eps),
        }
        if not return_membership:
            probe_kwargs["return_membership"] = False
        if approx_polish_labels:
            probe_kwargs["approx_polish_labels"] = True
        if basin_signatures:
            probe_kwargs["basin_signatures"] = True
        raw = probe(**probe_kwargs)
        return RustNonMonotoneGroupEscapeMultifidelityResult(
            membership=np.asarray(raw["membership"], dtype=np.uint64),
            quality=float(raw["quality"]),
            accepted=bool(raw["accepted"]),
            selected_policy=str(raw["selected_policy"]),
            selected_candidate_index=int(raw["selected_candidate_index"]),
            candidate_rows=list(raw["candidate_rows"]),
            policy_rows=list(raw["policy_rows"]),
            baseline_quality=float(raw["baseline_quality"]),
            best_delta_q=float(raw["best_delta_q"]),
            elapsed_ms=float(raw["elapsed_ms"]),
        )

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
            top_neighbor_weight=np.asarray(
                raw["top_neighbor_weight"], dtype=np.float64
            ),
            second_neighbor=np.asarray(
                raw.get("second_neighbor", np.full_like(raw["top_neighbor"], -1)),
                dtype=np.int64,
            ),
            second_neighbor_weight=np.asarray(
                raw.get(
                    "second_neighbor_weight", np.zeros_like(raw["top_neighbor_weight"])
                ),
                dtype=np.float64,
            ),
            neighbor_weight_ratio=np.asarray(
                raw.get(
                    "neighbor_weight_ratio", np.zeros_like(raw["top_neighbor_weight"])
                ),
                dtype=np.float64,
            ),
            conductance=np.asarray(raw["conductance"], dtype=np.float64),
            leafness=np.asarray(raw["leafness"], dtype=np.float64),
            band_distance=np.asarray(raw["band_distance"], dtype=np.float64),
            candidate_source=np.asarray(raw["candidate_source"], dtype=np.uint64),
            candidate_target=np.asarray(raw["candidate_target"], dtype=np.uint64),
            candidate_edge_weight=np.asarray(
                raw["candidate_edge_weight"], dtype=np.float64
            ),
            candidate_delta_q=np.asarray(raw["candidate_delta_q"], dtype=np.float64),
            candidate_merged_weight=np.asarray(
                raw["candidate_merged_weight"], dtype=np.float64
            ),
            candidate_size_band_gain=np.asarray(
                raw["candidate_size_band_gain"], dtype=np.float64
            ),
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
            top_neighbor_weight=np.asarray(
                raw["top_neighbor_weight"], dtype=np.float64
            ),
            second_neighbor=np.asarray(raw["second_neighbor"], dtype=np.int64),
            second_neighbor_weight=np.asarray(
                raw["second_neighbor_weight"], dtype=np.float64
            ),
            neighbor_weight_ratio=np.asarray(
                raw["neighbor_weight_ratio"], dtype=np.float64
            ),
            positive_move_count=np.asarray(raw["positive_move_count"], dtype=np.uint64),
            positive_move_weight=np.asarray(
                raw["positive_move_weight"], dtype=np.float64
            ),
            positive_delta_q=np.asarray(raw["positive_delta_q"], dtype=np.float64),
            near_neutral_move_count=np.asarray(
                raw["near_neutral_move_count"], dtype=np.uint64
            ),
            near_neutral_move_weight=np.asarray(
                raw["near_neutral_move_weight"], dtype=np.float64
            ),
            near_neutral_delta_q=np.asarray(
                raw["near_neutral_delta_q"], dtype=np.float64
            ),
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
            source_weight_after=np.asarray(
                raw["source_weight_after"], dtype=np.float64
            ),
            target_weight_before=np.asarray(
                raw["target_weight_before"], dtype=np.float64
            ),
            target_weight_after=np.asarray(
                raw["target_weight_after"], dtype=np.float64
            ),
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
            top_group_cut_weight=np.asarray(
                raw["top_group_cut_weight"], dtype=np.float64
            ),
            top_group_move_delta_q=np.asarray(
                raw["top_group_move_delta_q"], dtype=np.float64
            ),
            top_group_split_delta_q=np.asarray(
                raw["top_group_split_delta_q"], dtype=np.float64
            ),
            top_group_is_full_cluster=np.asarray(
                raw["top_group_is_full_cluster"], dtype=bool
            ),
            second_group_count=np.asarray(raw["second_group_count"], dtype=np.uint64),
            second_group_weight=np.asarray(
                raw["second_group_weight"], dtype=np.float64
            ),
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
            largest_group_target=np.asarray(
                raw["largest_group_target"], dtype=np.int64
            ),
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

    def external_grain_priority_clusters(
        self,
        membership: np.ndarray,
        candidate_clusters: np.ndarray,
        *,
        resolution: float,
        count: int,
        epsilon: float = 0.0,
        min_doc_weight: float = 0.0,
        max_incident_directed_edges: int = 0,
        min_best_delta_q: float = 0.0,
        min_assigned_fraction: float = 0.0,
        min_best_group_fraction: float = 0.0,
    ) -> list[int]:
        """Return external-grain priority clusters without materializing probe rows."""
        membership_dtype = np.asarray(membership).dtype
        if membership_dtype == np.dtype(np.uint32):
            priority = getattr(self.graph, "external_grain_priority_clusters_u32", None)
            if priority is not None:
                membership = np.ascontiguousarray(membership, dtype=np.uint32)
                candidate_clusters = np.ascontiguousarray(
                    candidate_clusters,
                    dtype=np.uint64,
                )
                raw = priority(
                    membership=membership,
                    candidate_clusters=candidate_clusters,
                    resolution=float(resolution),
                    count=int(count),
                    epsilon=float(epsilon),
                    min_doc_weight=float(min_doc_weight),
                    max_incident_directed_edges=int(max_incident_directed_edges),
                    min_best_delta_q=float(min_best_delta_q),
                    min_assigned_fraction=float(min_assigned_fraction),
                    min_best_group_fraction=float(min_best_group_fraction),
                )
                return [int(cluster) for cluster in np.asarray(raw, dtype=np.uint64)]

        probes = self.external_grain_probes(
            membership,
            candidate_clusters,
            resolution=resolution,
            epsilon=epsilon,
            min_doc_weight=min_doc_weight,
            max_incident_directed_edges=max_incident_directed_edges,
            min_best_delta_q=min_best_delta_q,
            min_assigned_fraction=min_assigned_fraction,
            min_best_group_fraction=min_best_group_fraction,
        )
        order = sorted(
            range(probes.n_probes),
            key=lambda idx: (
                bool(probes.recommended_for_split_repair[idx]),
                float(probes.priority[idx]),
                float(probes.best_group_to_target_weight[idx]),
                float(probes.best_group_weight[idx]),
            ),
            reverse=True,
        )
        selected: list[int] = []
        for idx in order:
            if len(selected) >= count:
                break
            if int(probes.best_group_target[idx]) < 0:
                continue
            selected.append(int(probes.cluster[idx]))
        return selected

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
        gamma_multipliers_array = np.ascontiguousarray(
            gamma_multipliers, dtype=np.float64
        )
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
            induced_directed_edges=np.asarray(
                raw["induced_directed_edges"], dtype=np.uint64
            ),
            n_parts=np.asarray(raw["n_parts"], dtype=np.uint64),
            non_singleton_parts=np.asarray(raw["non_singleton_parts"], dtype=np.uint64),
            singleton_parts=np.asarray(raw["singleton_parts"], dtype=np.uint64),
            singleton_weight=np.asarray(raw["singleton_weight"], dtype=np.float64),
            core_part_count=np.asarray(raw["core_part_count"], dtype=np.uint64),
            core_part_weight=np.asarray(raw["core_part_weight"], dtype=np.float64),
            largest_part_weight=np.asarray(
                raw["largest_part_weight"], dtype=np.float64
            ),
            second_part_weight=np.asarray(raw["second_part_weight"], dtype=np.float64),
            largest_part_fraction=np.asarray(
                raw["largest_part_fraction"], dtype=np.float64
            ),
            cut_weight=np.asarray(raw["cut_weight"], dtype=np.float64),
            split_delta_q_base=np.asarray(raw["split_delta_q_base"], dtype=np.float64),
            split_delta_q_probe=np.asarray(
                raw["split_delta_q_probe"], dtype=np.float64
            ),
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
        gamma_multipliers_array = np.ascontiguousarray(
            gamma_multipliers, dtype=np.float64
        )
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
            split_delta_q_probe=np.asarray(
                raw["split_delta_q_probe"], dtype=np.float64
            ),
            repair_quotient_edges=np.asarray(
                raw["repair_quotient_edges"], dtype=np.uint64
            ),
            repair_merge_count=np.asarray(raw["repair_merge_count"], dtype=np.uint64),
            repair_delta_q=np.asarray(raw["repair_delta_q"], dtype=np.float64),
            net_delta_q=np.asarray(raw["net_delta_q"], dtype=np.float64),
            final_source_units=np.asarray(raw["final_source_units"], dtype=np.uint64),
            retained_source_units=np.asarray(
                raw["retained_source_units"], dtype=np.uint64
            ),
            escaped_source_units=np.asarray(
                raw["escaped_source_units"], dtype=np.uint64
            ),
            escaped_source_weight=np.asarray(
                raw["escaped_source_weight"], dtype=np.float64
            ),
            final_small_source_units=np.asarray(
                raw["final_small_source_units"], dtype=np.uint64
            ),
            final_small_source_weight=np.asarray(
                raw["final_small_source_weight"], dtype=np.float64
            ),
            largest_source_unit_fraction=np.asarray(
                raw["largest_source_unit_fraction"], dtype=np.float64
            ),
            restored_source_cluster=np.asarray(
                raw["restored_source_cluster"], dtype=bool
            ),
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
        gamma_multipliers_array = np.ascontiguousarray(
            gamma_multipliers, dtype=np.float64
        )
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
            predicted_net_delta_q=np.asarray(
                raw["predicted_net_delta_q"], dtype=np.float64
            ),
            repair_merge_count=np.asarray(raw["repair_merge_count"], dtype=np.uint64),
            final_source_units=np.asarray(raw["final_source_units"], dtype=np.uint64),
            retained_source_units=np.asarray(
                raw["retained_source_units"], dtype=np.uint64
            ),
            escaped_source_units=np.asarray(
                raw["escaped_source_units"], dtype=np.uint64
            ),
            escaped_source_weight=np.asarray(
                raw["escaped_source_weight"], dtype=np.float64
            ),
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
            new_retained_clusters=np.asarray(
                raw["new_retained_clusters"], dtype=np.uint64
            ),
        )

    def dongdaemun_refine(
        self,
        membership: np.ndarray,
        *,
        resolution: float,
        target_max_weight: float,
        gamma_multipliers: Sequence[float],
        policy: str = "quality_first",
        quality_floor_delta: float = 0.0,
        apply_iterations: int = 4,
        min_core_weight: float = 25.0,
        randomness: float = 0.01,
        repair_epsilon: float = 0.0,
        trim_min_delta_q_quality_first: float = 0.0,
        trim_min_delta_q_hard_cap: float = -1.0,
        trim_max_moves_per_cluster: int = 100,
        seed: int = 42,
        pair_seeded: bool = True,
    ) -> RustDongdaemunResult:
        """Run Rust Dongdaemun upper-tail refinement with exact CPM audit."""
        refine = getattr(self.graph, "dongdaemun_refine", None)
        if refine is None:
            _check_dongdaemun_available()
            raise AttributeError(
                "installed sciscape_leiden graph instance does not expose "
                "Graph.dongdaemun_refine"
            )
        membership = np.ascontiguousarray(membership, dtype=np.uint64)
        gamma_multipliers_array = np.ascontiguousarray(
            gamma_multipliers, dtype=np.float64
        )
        raw = refine(
            membership=membership,
            resolution=float(resolution),
            target_max_weight=float(target_max_weight),
            gamma_multipliers=gamma_multipliers_array,
            policy=str(policy),
            quality_floor_delta=float(quality_floor_delta),
            apply_iterations=int(apply_iterations),
            min_core_weight=float(min_core_weight),
            randomness=float(randomness),
            repair_epsilon=float(repair_epsilon),
            trim_min_delta_q_quality_first=float(trim_min_delta_q_quality_first),
            trim_min_delta_q_hard_cap=float(trim_min_delta_q_hard_cap),
            trim_max_moves_per_cluster=int(trim_max_moves_per_cluster),
            seed=int(seed),
            pair_seeded=bool(pair_seeded),
        )
        diagnostic_membership = (
            np.asarray(raw["diagnostic_membership"], dtype=np.uint64)
            if bool(raw["diagnostic_present"])
            else None
        )
        audit = RustDongdaemunAudit(
            accepted=bool(raw["accepted"]),
            status=str(raw["status"]),
            quality_before=float(raw["quality_before"]),
            quality_after_candidate=float(raw["quality_after_candidate"]),
            candidate_delta_q=float(raw["candidate_delta_q"]),
            effective_delta_q=float(raw["effective_delta_q"]),
            final_delta_q=float(raw["final_delta_q"]),
            target_max_satisfied=bool(raw["target_max_satisfied"]),
            n_oversize_before=int(raw["n_oversize_before"]),
            n_oversize_after_candidate=int(raw["n_oversize_after_candidate"]),
            max_weight_before=float(raw["max_weight_before"]),
            max_weight_after_candidate=float(raw["max_weight_after_candidate"]),
            trim_moves_committed=int(raw["trim_moves_committed"]),
            trim_moves_proposed=int(raw["trim_moves_proposed"]),
            split_iteration=np.asarray(raw["split_iteration"], dtype=np.uint64),
            split_candidate_clusters=np.asarray(
                raw["split_candidate_clusters"], dtype=np.uint64
            ),
            split_n_selected=np.asarray(raw["split_n_selected"], dtype=np.uint64),
            split_n_applied=np.asarray(raw["split_n_applied"], dtype=np.uint64),
            split_status_code=np.asarray(raw["split_status_code"], dtype=np.uint8),
            split_exact_delta_q=np.asarray(
                raw["split_exact_delta_q"], dtype=np.float64
            ),
        )
        return RustDongdaemunResult(
            membership=np.asarray(raw["membership"], dtype=np.uint64),
            n_clusters=int(raw["n_clusters"]),
            diagnostic_membership=diagnostic_membership,
            diagnostic_n_clusters=int(raw["diagnostic_n_clusters"]),
            audit=audit,
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
    node_weights_path: Path | None = None,
) -> RustLeidenGraph:
    """Build a reusable Rust CSR graph from edge arrays or an int-edge parquet."""
    _check_available()
    graph_cls = getattr(_rust, "Graph", None)
    if graph_cls is None:
        raise AttributeError("installed sciscape_leiden module does not expose Graph")
    if node_weights is not None and node_weights_path is not None:
        raise ValueError("Provide either node_weights or node_weights_path, not both")

    if edge_path is not None and edges_src is None:
        raw_loader = getattr(_rust, "load_graph_raw_files", None)
        if raw_loader is not None:
            if n_nodes is None:
                raise ValueError(
                    "n_nodes is required when loading a graph from raw sidecars"
                )
            src_path, dst_path, weight_path = ensure_int_edge_sidecars(Path(edge_path))
            raw_node_weights_path = None
            temp_node_weights_path = None
            try:
                if node_weights_path is not None:
                    raw_node_weights_path = str(Path(node_weights_path))
                elif node_weights is not None:
                    with NamedTemporaryFile(
                        prefix="node_weights.",
                        suffix=".f64.bin",
                        dir=Path(edge_path).parent,
                        delete=False,
                    ) as fh:
                        np.ascontiguousarray(node_weights, dtype=np.float64).tofile(fh)
                        temp_node_weights_path = fh.name
                        raw_node_weights_path = temp_node_weights_path
                graph = raw_loader(
                    n_nodes=n_nodes,
                    src_path=str(src_path),
                    dst_path=str(dst_path),
                    weights_path=str(weight_path),
                    node_weights_path=raw_node_weights_path,
                )
            finally:
                if temp_node_weights_path is not None:
                    Path(temp_node_weights_path).unlink(missing_ok=True)
            returned_node_weights = None
            if node_weights is not None:
                returned_node_weights = np.asarray(node_weights, dtype=np.float64)
            elif node_weights_path is not None:
                returned_node_weights = np.memmap(
                    node_weights_path,
                    dtype=np.float64,
                    mode="r",
                )
            return RustLeidenGraph(
                graph=graph,
                n_nodes=int(graph.n_nodes),
                n_edges=int(graph.n_edges),
                node_weights=returned_node_weights,
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
    elif node_weights_path is not None:
        nw = np.ascontiguousarray(
            np.memmap(node_weights_path, dtype=np.float64, mode="r"),
            dtype=np.float64,
        )

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
        None if randomness_schedule is None else [float(x) for x in randomness_schedule]
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
            graph.n_nodes,
            result.n_clusters,
            resolution,
            result.quality,
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
        n_nodes,
        n_clusters,
        resolution,
        quality,
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
        f"min_weight={min_weight}" if min_weight > 0 else f"min_size={min_size}"
    )
    for r in rounds:
        log.info(
            "postprocess round %d: γ=%.4e, method=%s, small: %d→%d, "
            "merged: %d, total: %d, max_size: %d, max_weight: %.1f",
            r["round"],
            r["gamma"],
            r["method"],
            r["n_small_before"],
            r["n_small_after"],
            r["n_merged"],
            r["n_total_clusters"],
            r["max_cluster_size"],
            r["max_cluster_weight"],
        )
    log.info(
        "postprocess_rust: %d nodes changed, %d clusters (%s, %d rounds)",
        changed,
        n_clusters,
        threshold_str,
        len(rounds),
    )

    return RustPostprocessResult(
        membership=result_mem,
        n_clusters=n_clusters,
        changed_at_round=changed_at,
        rounds=rounds,
    )


def dongdaemun_refine_rust(
    *,
    membership: np.ndarray,
    resolution: float,
    target_max_weight: float,
    n_nodes: int | None = None,
    edge_path: Path | None = None,
    edges_src: np.ndarray | None = None,
    edges_dst: np.ndarray | None = None,
    edges_weight: np.ndarray | None = None,
    node_weights: np.ndarray | None = None,
    gamma_multipliers: Sequence[float] = DEFAULT_DONGDAEMUN_GAMMA_MULTIPLIERS,
    policy: str = "quality_first",
    quality_floor_delta: float = 0.0,
    apply_iterations: int = 4,
    min_core_weight: float = 25.0,
    randomness: float = 0.01,
    repair_epsilon: float = 0.0,
    trim_min_delta_q_quality_first: float = 0.0,
    trim_min_delta_q_hard_cap: float = -1.0,
    trim_max_moves_per_cluster: int = 100,
    seed: int = 42,
    pair_seeded: bool = True,
) -> RustDongdaemunResult:
    """Run Rust Dongdaemun refinement from edge arrays or an int-edge parquet path."""
    _check_dongdaemun_available()
    membership = np.ascontiguousarray(membership, dtype=np.uint64)
    nw = (
        None
        if node_weights is None
        else np.ascontiguousarray(node_weights, dtype=np.float64)
    )
    graph = build_leiden_graph(
        edge_path,
        n_nodes=n_nodes or int(membership.shape[0]),
        edges_src=edges_src,
        edges_dst=edges_dst,
        edges_weight=edges_weight,
        node_weights=nw,
    )
    return graph.dongdaemun_refine(
        membership,
        resolution=float(resolution),
        target_max_weight=float(target_max_weight),
        gamma_multipliers=gamma_multipliers,
        policy=str(policy),
        quality_floor_delta=float(quality_floor_delta),
        apply_iterations=int(apply_iterations),
        min_core_weight=float(min_core_weight),
        randomness=float(randomness),
        repair_epsilon=float(repair_epsilon),
        trim_min_delta_q_quality_first=float(trim_min_delta_q_quality_first),
        trim_min_delta_q_hard_cap=float(trim_min_delta_q_hard_cap),
        trim_max_moves_per_cluster=int(trim_max_moves_per_cluster),
        seed=int(seed),
        pair_seeded=bool(pair_seeded),
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
