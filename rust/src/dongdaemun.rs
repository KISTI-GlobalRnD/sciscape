//! Dongdaemun upper-tail refinement contract helpers.
//!
//! This module wires deterministic split-repair iterations and boundary-trim
//! proposals. All accepted output still passes through exact CPM audit and
//! policy-specific fallback.

use crate::adaptive::{
    apply_cached_split_merge_repair_candidates, split_merge_repair_cached_candidates,
    trim_oversize_boundary_moves, OversizeBoundaryMove, SplitMergeRepairCachedCandidate,
    SplitMergeRepairProbe,
};
use crate::clustering::{ClusterId, Clustering};
use crate::graph::Graph;
use crate::quality::{QualityFunction, CPM};
use crate::workspace::Workspace;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DongdaemunPolicy {
    QualityFirst,
    HardCap,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DongdaemunStatus {
    NoCurrentOversizeCandidates,
    Committed,
    NoSelectedCandidates,
    NoProgress,
    SplitQualityBelowFloor,
    TrimQualityBelowFloor,
    QualityBelowFloor,
    HardCapNotSatisfied,
}

#[derive(Clone, Debug)]
pub struct DongdaemunConfig {
    pub policy: DongdaemunPolicy,
    pub resolution: f64,
    /// Maximum allowed cluster weight for oversize detection.
    ///
    /// A cluster is oversized only when `weight > target_max_weight`, so cap
    /// satisfaction means `max_weight <= target_max_weight`.
    pub target_max_weight: f64,
    /// Minimum exact CPM delta required against the original baseline `P_min`.
    ///
    /// Split and trim proposals may be generated from intermediate states, but
    /// the floor is always checked as `Q(candidate) >= Q(P_min) + delta`.
    /// This is a cumulative floor, not a per-iteration monotonicity rule, so a
    /// later iteration may have negative local `exact_delta_q` if the full
    /// candidate remains above the original baseline floor.
    /// Positive values require improvement, `0.0` means non-regression, and
    /// negative values allow bounded quality loss.
    pub quality_floor_delta: f64,
    /// Maximum split-repair iterations before the single trim pass.
    pub apply_iterations: usize,
    pub gamma_multipliers: Vec<f64>,
    pub min_core_weight: f64,
    pub randomness: f64,
    pub repair_epsilon: f64,
    pub trim_min_delta_q_quality_first: f64,
    pub trim_min_delta_q_hard_cap: f64,
    pub trim_max_moves_per_cluster: usize,
    pub seed: u64,
    pub pair_seeded: bool,
}

impl DongdaemunConfig {
    pub fn default_for_quality_first(resolution: f64, target_max_weight: f64) -> Self {
        Self::default_with_policy(
            DongdaemunPolicy::QualityFirst,
            resolution,
            target_max_weight,
        )
    }

    pub fn default_for_hard_cap(resolution: f64, target_max_weight: f64) -> Self {
        Self::default_with_policy(DongdaemunPolicy::HardCap, resolution, target_max_weight)
    }

    pub fn validate(&self) -> Result<(), String> {
        validate_finite("resolution", self.resolution)?;
        validate_finite("target_max_weight", self.target_max_weight)?;
        validate_finite("quality_floor_delta", self.quality_floor_delta)?;
        validate_finite("min_core_weight", self.min_core_weight)?;
        validate_finite("randomness", self.randomness)?;
        validate_finite("repair_epsilon", self.repair_epsilon)?;
        validate_finite(
            "trim_min_delta_q_quality_first",
            self.trim_min_delta_q_quality_first,
        )?;
        validate_finite("trim_min_delta_q_hard_cap", self.trim_min_delta_q_hard_cap)?;

        if self.target_max_weight <= 0.0 {
            return Err("target_max_weight must be > 0".to_string());
        }
        if self.apply_iterations == 0 {
            return Err("apply_iterations must be >= 1".to_string());
        }
        for (idx, &multiplier) in self.gamma_multipliers.iter().enumerate() {
            if !multiplier.is_finite() {
                return Err(format!("gamma_multipliers[{idx}] must be finite"));
            }
            if multiplier <= 0.0 {
                return Err(format!("gamma_multipliers[{idx}] must be > 0"));
            }
        }

        Ok(())
    }

    fn default_with_policy(
        policy: DongdaemunPolicy,
        resolution: f64,
        target_max_weight: f64,
    ) -> Self {
        Self {
            policy,
            resolution,
            target_max_weight,
            quality_floor_delta: 0.0,
            apply_iterations: 4,
            gamma_multipliers: vec![1.02, 1.05, 1.10, 1.15, 1.20, 1.25],
            min_core_weight: 25.0,
            randomness: 0.01,
            repair_epsilon: 0.0,
            trim_min_delta_q_quality_first: 0.0,
            trim_min_delta_q_hard_cap: -1.0,
            trim_max_moves_per_cluster: 100,
            seed: 42,
            pair_seeded: true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct DongdaemunSplitIteration {
    pub iteration: usize,
    pub candidate_clusters: Vec<ClusterId>,
    pub n_selected: usize,
    pub n_applied: usize,
    pub status: DongdaemunStatus,
    /// Exact CPM delta from this iteration's input clustering to its candidate.
    pub exact_delta_q: f64,
}

#[derive(Clone, Debug)]
pub struct DongdaemunAudit {
    pub accepted: bool,
    pub status: DongdaemunStatus,
    pub quality_before: f64,
    /// Exact CPM quality of the audited candidate state.
    ///
    /// If no candidate was selected, this reports the returned baseline state
    /// as the candidate/effective state.
    pub quality_after_candidate: f64,
    /// Exact CPM delta from baseline to the audited candidate state.
    ///
    /// For rejected candidates, this reports the diagnostic candidate delta;
    /// the effective output remains at baseline.
    pub candidate_delta_q: f64,
    /// Exact CPM delta from baseline to the effective output.
    ///
    /// Rejected candidates always report `0.0` here because the effective
    /// clustering falls back to the baseline.
    pub effective_delta_q: f64,
    /// Deprecated compatibility alias for `candidate_delta_q`.
    ///
    /// Prefer `candidate_delta_q` for diagnostic candidate quality and
    /// `effective_delta_q` for the quality delta of the returned clustering.
    pub final_delta_q: f64,
    /// Whether the audited candidate state satisfies `max_weight <= target`.
    ///
    /// If no candidate was selected, this is evaluated on the returned baseline
    /// fallback. Rejected candidate diagnostics still report the rejected
    /// candidate's cap state.
    pub target_max_satisfied: bool,
    pub n_oversize_before: usize,
    pub n_oversize_after_candidate: usize,
    pub max_weight_before: f64,
    pub max_weight_after_candidate: f64,
    pub split_iterations: Vec<DongdaemunSplitIteration>,
    pub trim_moves_committed: usize,
    pub trim_moves_proposed: usize,
}

#[derive(Clone, Debug)]
pub struct DongdaemunResult {
    /// Effective clustering to pass forward.
    ///
    /// When `audit.accepted == false`, this is the baseline fallback. Inspect
    /// `audit.status` and `diagnostic_clustering` to distinguish fallback
    /// reasons and rejected candidate diagnostics.
    pub clustering: Clustering,
    pub diagnostic_clustering: Option<Clustering>,
    pub audit: DongdaemunAudit,
}

#[derive(Clone, Debug)]
struct ClusterWeightSummary {
    weights: Vec<f64>,
    target_max_weight: f64,
    n_oversize: usize,
    max_weight: f64,
    target_max_satisfied: bool,
}

#[derive(Clone, Debug)]
struct AcceptanceDecision {
    effective_clustering: Clustering,
    diagnostic_clustering: Option<Clustering>,
    audit: DongdaemunAudit,
}

#[derive(Clone, Debug, PartialEq)]
struct SelectedSplitRepairCandidate {
    cluster: u64,
    gamma_multiplier: f64,
}

#[derive(Clone, Copy, Debug)]
struct SplitRepairSelectionCandidate<'a> {
    probe: &'a SplitMergeRepairProbe,
    receiver_remaining_oversize_after: f64,
    receiver_oversize_increase: f64,
}

#[derive(Clone, Debug)]
struct SplitRepairOutcome {
    clustering: Clustering,
    diagnostic_clustering: Option<Clustering>,
    iteration: DongdaemunSplitIteration,
    accepted: bool,
}

#[derive(Clone, Debug)]
struct SplitRepairLoopOutcome {
    clustering: Clustering,
    diagnostic_clustering: Option<Clustering>,
    iterations: Vec<DongdaemunSplitIteration>,
}

pub fn dongdaemun_refine(
    graph: &Graph,
    baseline: &Clustering,
    config: &DongdaemunConfig,
    ws: &mut Workspace,
) -> DongdaemunResult {
    config.validate().expect("invalid DongdaemunConfig");
    assert_eq!(baseline.n_nodes, graph.n_nodes);

    let quality_before = CPM::new(config.resolution).quality(graph, baseline);
    let before_summary = cluster_weight_summary(graph, baseline, config.target_max_weight, ws);
    let current_oversize = current_oversize_clusters(&before_summary);

    if current_oversize.is_empty() {
        return DongdaemunResult {
            clustering: baseline.clone(),
            diagnostic_clustering: None,
            audit: DongdaemunAudit {
                accepted: true,
                status: DongdaemunStatus::NoCurrentOversizeCandidates,
                quality_before,
                quality_after_candidate: quality_before,
                candidate_delta_q: 0.0,
                effective_delta_q: 0.0,
                final_delta_q: 0.0,
                target_max_satisfied: true,
                n_oversize_before: 0,
                n_oversize_after_candidate: 0,
                max_weight_before: before_summary.max_weight,
                max_weight_after_candidate: before_summary.max_weight,
                split_iterations: Vec::new(),
                trim_moves_committed: 0,
                trim_moves_proposed: 0,
            },
        };
    }

    let split_loop = run_split_repair_iterations(
        graph,
        baseline,
        config,
        quality_before + config.quality_floor_delta,
        ws,
    );
    let current = split_loop.clustering;
    let split_iterations = split_loop.iterations;
    let split_diagnostic = split_loop.diagnostic_clustering;

    let current_summary = cluster_weight_summary(graph, &current, config.target_max_weight, ws);
    let current_oversize = current_oversize_clusters(&current_summary);
    if current_oversize.is_empty() {
        let mut decision = accept_candidate(graph, baseline, &current, config, ws);
        decision.audit.split_iterations = split_iterations;
        return DongdaemunResult {
            clustering: decision.effective_clustering,
            diagnostic_clustering: decision.diagnostic_clustering,
            audit: decision.audit,
        };
    }

    let candidate_clusters = current_oversize
        .iter()
        .copied()
        .map(u64::from)
        .collect::<Vec<_>>();
    let min_delta_q = trim_min_delta_q_for_policy(config);
    let (proposed_membership, moves) = trim_oversize_boundary_moves(
        graph,
        &current,
        &candidate_clusters,
        config.resolution,
        config.target_max_weight,
        min_delta_q,
        config.trim_max_moves_per_cluster,
        ws,
    );

    if moves.is_empty() {
        if current.clusters != baseline.clusters {
            let mut decision = accept_candidate(graph, baseline, &current, config, ws);
            decision.audit.split_iterations = split_iterations;
            return DongdaemunResult {
                clustering: decision.effective_clustering,
                diagnostic_clustering: decision.diagnostic_clustering,
                audit: decision.audit,
            };
        }
        if let Some(diagnostic) = split_diagnostic {
            let status = split_iterations
                .last()
                .map_or(DongdaemunStatus::NoProgress, |iteration| iteration.status);
            return rejected_candidate_result(
                graph,
                baseline,
                &diagnostic,
                config,
                ws,
                status,
                split_iterations,
                0,
                0,
            );
        }
        let status = split_iterations
            .last()
            .map_or(DongdaemunStatus::NoSelectedCandidates, |iteration| {
                iteration.status
            });
        return DongdaemunResult {
            clustering: baseline.clone(),
            diagnostic_clustering: None,
            audit: DongdaemunAudit {
                accepted: false,
                status,
                quality_before,
                quality_after_candidate: quality_before,
                candidate_delta_q: 0.0,
                effective_delta_q: 0.0,
                final_delta_q: 0.0,
                target_max_satisfied: before_summary.target_max_satisfied,
                n_oversize_before: current_oversize.len(),
                n_oversize_after_candidate: current_oversize.len(),
                max_weight_before: current_summary.max_weight,
                max_weight_after_candidate: current_summary.max_weight,
                split_iterations,
                trim_moves_committed: 0,
                trim_moves_proposed: 0,
            },
        };
    }

    let mut proposed =
        Clustering::from_u64_assignments(&proposed_membership).expect("trim produced valid ids");
    proposed.remove_empty_clusters();
    let cpm = CPM::new(config.resolution);
    let quality_after_proposed = cpm.quality(graph, &proposed);
    let quality_floor = quality_before + config.quality_floor_delta;
    let (candidate, candidate_move_count) = if quality_after_proposed >= quality_floor {
        (proposed, moves.len())
    } else {
        // Trim moves are relative to `current` after split-repair, while the
        // quality floor remains absolute against the original baseline P_min.
        let prefix_count = quality_floor_prefix_move_count(
            graph,
            &current,
            &moves,
            config.resolution,
            quality_floor,
        );
        if prefix_count == 0 {
            if current.clusters != baseline.clusters {
                let mut decision = accept_candidate(graph, baseline, &current, config, ws);
                decision.audit.split_iterations = split_iterations;
                decision.audit.trim_moves_proposed = moves.len();
                decision.audit.trim_moves_committed = 0;
                return DongdaemunResult {
                    clustering: decision.effective_clustering,
                    diagnostic_clustering: decision.diagnostic_clustering,
                    audit: decision.audit,
                };
            }
            return trim_quality_floor_fallback(
                graph,
                baseline,
                &proposed,
                config,
                ws,
                quality_before,
                quality_after_proposed,
                &before_summary,
                moves.len(),
                split_iterations,
            );
        }
        (
            apply_trim_move_prefix(&current, &moves, prefix_count),
            prefix_count,
        )
    };

    let mut decision = accept_candidate(graph, baseline, &candidate, config, ws);
    decision.audit.split_iterations = split_iterations;
    decision.audit.trim_moves_proposed = moves.len();
    decision.audit.trim_moves_committed = candidate_move_count;

    DongdaemunResult {
        clustering: decision.effective_clustering,
        diagnostic_clustering: decision.diagnostic_clustering,
        audit: decision.audit,
    }
}

fn validate_finite(name: &str, value: f64) -> Result<(), String> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(format!("{name} must be finite"))
    }
}

fn trim_min_delta_q_for_policy(config: &DongdaemunConfig) -> f64 {
    match config.policy {
        DongdaemunPolicy::QualityFirst => config.trim_min_delta_q_quality_first,
        DongdaemunPolicy::HardCap => config.trim_min_delta_q_hard_cap,
    }
}

fn trim_quality_floor_fallback(
    graph: &Graph,
    baseline: &Clustering,
    proposed: &Clustering,
    config: &DongdaemunConfig,
    ws: &mut Workspace,
    quality_before: f64,
    quality_after_proposed: f64,
    before_summary: &ClusterWeightSummary,
    trim_moves_proposed: usize,
    split_iterations: Vec<DongdaemunSplitIteration>,
) -> DongdaemunResult {
    let proposed_summary = cluster_weight_summary(graph, proposed, config.target_max_weight, ws);

    DongdaemunResult {
        clustering: baseline.clone(),
        diagnostic_clustering: if baseline.clusters == proposed.clusters {
            None
        } else {
            Some(proposed.clone())
        },
        audit: DongdaemunAudit {
            accepted: false,
            status: DongdaemunStatus::TrimQualityBelowFloor,
            quality_before,
            quality_after_candidate: quality_after_proposed,
            candidate_delta_q: quality_after_proposed - quality_before,
            effective_delta_q: 0.0,
            final_delta_q: quality_after_proposed - quality_before,
            target_max_satisfied: proposed_summary.target_max_satisfied,
            n_oversize_before: before_summary.n_oversize,
            n_oversize_after_candidate: proposed_summary.n_oversize,
            max_weight_before: before_summary.max_weight,
            max_weight_after_candidate: proposed_summary.max_weight,
            split_iterations,
            trim_moves_committed: 0,
            trim_moves_proposed,
        },
    }
}

fn run_split_repair_iterations(
    graph: &Graph,
    baseline: &Clustering,
    config: &DongdaemunConfig,
    quality_floor: f64,
    ws: &mut Workspace,
) -> SplitRepairLoopOutcome {
    let mut current = baseline.clone();
    let mut diagnostic_clustering = None;
    let mut iterations = Vec::new();

    for iteration in 1..=config.apply_iterations {
        let current_summary = cluster_weight_summary(graph, &current, config.target_max_weight, ws);
        let candidate_clusters = current_oversize_clusters(&current_summary);
        if candidate_clusters.is_empty() {
            iterations.push(DongdaemunSplitIteration {
                iteration,
                candidate_clusters,
                n_selected: 0,
                n_applied: 0,
                status: DongdaemunStatus::NoCurrentOversizeCandidates,
                exact_delta_q: 0.0,
            });
            break;
        }

        let split = split_repair_once(
            graph,
            &current,
            &candidate_clusters,
            iteration,
            config,
            quality_floor,
            ws,
        );
        let accepted = split.accepted;
        let diagnostic = split.diagnostic_clustering;
        iterations.push(split.iteration);

        if accepted {
            current = split.clustering;
            diagnostic_clustering = None;
        } else {
            diagnostic_clustering = diagnostic;
            break;
        }
    }

    SplitRepairLoopOutcome {
        clustering: current,
        diagnostic_clustering,
        iterations,
    }
}

fn split_repair_once(
    graph: &Graph,
    iteration_baseline: &Clustering,
    candidate_clusters: &[ClusterId],
    iteration: usize,
    config: &DongdaemunConfig,
    quality_floor: f64,
    ws: &mut Workspace,
) -> SplitRepairOutcome {
    let candidate_cluster_ids = candidate_clusters
        .iter()
        .copied()
        .map(u64::from)
        .collect::<Vec<_>>();
    let cached_candidates = split_merge_repair_cached_candidates(
        graph,
        iteration_baseline,
        &candidate_cluster_ids,
        config.resolution,
        &config.gamma_multipliers,
        config.min_core_weight,
        config.randomness,
        config.repair_epsilon,
        config.seed,
        config.pair_seeded,
        ws,
    );
    let selected = select_cached_split_repair_candidates(
        &cached_candidates,
        config.policy,
        config.target_max_weight,
    );
    let mut iteration = DongdaemunSplitIteration {
        iteration,
        candidate_clusters: candidate_clusters.to_vec(),
        n_selected: selected.len(),
        n_applied: 0,
        status: DongdaemunStatus::NoSelectedCandidates,
        exact_delta_q: 0.0,
    };

    if selected.is_empty() {
        return SplitRepairOutcome {
            clustering: iteration_baseline.clone(),
            diagnostic_clustering: None,
            iteration,
            accepted: false,
        };
    }

    let selected_clusters = selected
        .iter()
        .map(|candidate| candidate.cluster)
        .collect::<Vec<_>>();
    let selected_gamma_multipliers = selected
        .iter()
        .map(|candidate| candidate.gamma_multiplier)
        .collect::<Vec<_>>();
    let (membership, applied) = apply_cached_split_merge_repair_candidates(
        graph,
        iteration_baseline,
        &cached_candidates,
        &selected_clusters,
        &selected_gamma_multipliers,
        config.resolution,
        config.min_core_weight,
        config.repair_epsilon,
        ws,
    );
    iteration.n_applied = applied.len();

    let mut proposed = Clustering::from_u64_assignments(&membership)
        .expect("split-repair produced valid cluster ids");
    proposed.remove_empty_clusters();
    let membership_changed = proposed.clusters != iteration_baseline.clusters;
    if applied.is_empty() || !membership_changed {
        iteration.status = DongdaemunStatus::NoProgress;
        return SplitRepairOutcome {
            clustering: iteration_baseline.clone(),
            diagnostic_clustering: None,
            iteration,
            accepted: false,
        };
    }

    let cpm = CPM::new(config.resolution);
    let quality_before = cpm.quality(graph, iteration_baseline);
    let quality_after = cpm.quality(graph, &proposed);
    let exact_delta_q = quality_after - quality_before;
    iteration.exact_delta_q = exact_delta_q;
    if quality_after < quality_floor {
        iteration.status = DongdaemunStatus::SplitQualityBelowFloor;
        return SplitRepairOutcome {
            clustering: iteration_baseline.clone(),
            diagnostic_clustering: Some(proposed),
            iteration,
            accepted: false,
        };
    }

    iteration.status = DongdaemunStatus::Committed;
    SplitRepairOutcome {
        clustering: proposed,
        diagnostic_clustering: None,
        iteration,
        accepted: true,
    }
}

#[cfg(test)]
fn select_split_repair_candidates(
    probes: &[SplitMergeRepairProbe],
    policy: DongdaemunPolicy,
    target_max_weight: f64,
) -> Vec<SelectedSplitRepairCandidate> {
    select_split_repair_candidates_from_iter(probes.iter(), policy, target_max_weight)
}

fn select_cached_split_repair_candidates(
    cached_candidates: &[SplitMergeRepairCachedCandidate],
    policy: DongdaemunPolicy,
    target_max_weight: f64,
) -> Vec<SelectedSplitRepairCandidate> {
    select_split_repair_selection_candidates(
        cached_candidates
            .iter()
            .map(|candidate| SplitRepairSelectionCandidate {
                probe: &candidate.probe,
                receiver_remaining_oversize_after: candidate
                    .receiver_remaining_oversize_after(target_max_weight),
                receiver_oversize_increase: candidate.receiver_oversize_increase(target_max_weight),
            }),
        policy,
        target_max_weight,
    )
}

#[cfg(test)]
fn select_split_repair_candidates_from_iter<'a>(
    probes: impl IntoIterator<Item = &'a SplitMergeRepairProbe>,
    policy: DongdaemunPolicy,
    target_max_weight: f64,
) -> Vec<SelectedSplitRepairCandidate> {
    select_split_repair_selection_candidates(
        probes
            .into_iter()
            .map(|probe| SplitRepairSelectionCandidate {
                probe,
                receiver_remaining_oversize_after: 0.0,
                receiver_oversize_increase: 0.0,
            }),
        policy,
        target_max_weight,
    )
}

fn select_split_repair_selection_candidates<'a>(
    candidates: impl IntoIterator<Item = SplitRepairSelectionCandidate<'a>>,
    policy: DongdaemunPolicy,
    target_max_weight: f64,
) -> Vec<SelectedSplitRepairCandidate> {
    let mut candidates = candidates
        .into_iter()
        .filter(|candidate| split_repair_probe_is_eligible(candidate.probe))
        .collect::<Vec<_>>();

    candidates.sort_by(|left, right| {
        let base = match policy {
            DongdaemunPolicy::QualityFirst => compare_split_quality_first(left, right),
            DongdaemunPolicy::HardCap => compare_split_hard_cap(left, right, target_max_weight),
        };
        // Stable ties prefer lower cluster id, then lower gamma multiplier.
        // Lower gamma is the conservative choice when probe evidence is equal.
        base.then_with(|| left.probe.cluster.cmp(&right.probe.cluster))
            .then_with(|| {
                left.probe
                    .gamma_multiplier
                    .total_cmp(&right.probe.gamma_multiplier)
            })
    });

    let mut selected = Vec::new();
    for candidate in candidates {
        if selected
            .iter()
            .any(|selected: &SelectedSplitRepairCandidate| {
                selected.cluster == candidate.probe.cluster
            })
        {
            continue;
        }
        selected.push(SelectedSplitRepairCandidate {
            cluster: candidate.probe.cluster,
            gamma_multiplier: candidate.probe.gamma_multiplier,
        });
    }
    selected
}

fn split_repair_probe_is_eligible(probe: &SplitMergeRepairProbe) -> bool {
    probe.net_delta_q >= 0.0
        && !probe.restored_source_cluster
        && (probe.escaped_source_units > 0 || probe.retained_source_units > 1)
}

fn compare_split_quality_first(
    left: &SplitRepairSelectionCandidate<'_>,
    right: &SplitRepairSelectionCandidate<'_>,
) -> std::cmp::Ordering {
    left.probe
        .largest_source_unit_fraction
        .total_cmp(&right.probe.largest_source_unit_fraction)
        .then_with(|| {
            left.receiver_oversize_increase
                .total_cmp(&right.receiver_oversize_increase)
        })
        .then_with(|| {
            right
                .probe
                .escaped_source_weight
                .total_cmp(&left.probe.escaped_source_weight)
        })
        .then_with(|| right.probe.net_delta_q.total_cmp(&left.probe.net_delta_q))
}

fn compare_split_hard_cap(
    left: &SplitRepairSelectionCandidate<'_>,
    right: &SplitRepairSelectionCandidate<'_>,
    target_max_weight: f64,
) -> std::cmp::Ordering {
    let left_cap_satisfied =
        split_candidate_remaining_oversize_after(left, target_max_weight) == 0.0;
    let right_cap_satisfied =
        split_candidate_remaining_oversize_after(right, target_max_weight) == 0.0;
    match (left_cap_satisfied, right_cap_satisfied) {
        (true, false) => std::cmp::Ordering::Less,
        (false, true) => std::cmp::Ordering::Greater,
        _ => split_candidate_oversize_reduction(right, target_max_weight)
            .total_cmp(&split_candidate_oversize_reduction(left, target_max_weight))
            .then_with(|| {
                split_candidate_remaining_oversize_after(left, target_max_weight).total_cmp(
                    &split_candidate_remaining_oversize_after(right, target_max_weight),
                )
            })
            .then_with(|| {
                left.receiver_oversize_increase
                    .total_cmp(&right.receiver_oversize_increase)
            })
            .then_with(|| compare_split_quality_first(left, right)),
    }
}

fn split_candidate_oversize_reduction(
    candidate: &SplitRepairSelectionCandidate<'_>,
    target_max_weight: f64,
) -> f64 {
    let before = (candidate.probe.doc_weight - target_max_weight).max(0.0);
    let source_after = split_probe_remaining_oversize_after(candidate.probe, target_max_weight);
    let after = source_after + candidate.receiver_oversize_increase.max(0.0);
    (before - after).max(0.0)
}

fn split_candidate_remaining_oversize_after(
    candidate: &SplitRepairSelectionCandidate<'_>,
    target_max_weight: f64,
) -> f64 {
    split_probe_remaining_oversize_after(candidate.probe, target_max_weight)
        + candidate.receiver_remaining_oversize_after
}

fn split_probe_remaining_oversize_after(
    probe: &SplitMergeRepairProbe,
    target_max_weight: f64,
) -> f64 {
    let largest_source_unit_weight = probe.largest_source_unit_fraction * probe.doc_weight;
    (largest_source_unit_weight - target_max_weight).max(0.0)
}

fn rejected_candidate_result(
    graph: &Graph,
    baseline: &Clustering,
    candidate: &Clustering,
    config: &DongdaemunConfig,
    ws: &mut Workspace,
    status: DongdaemunStatus,
    split_iterations: Vec<DongdaemunSplitIteration>,
    trim_moves_proposed: usize,
    trim_moves_committed: usize,
) -> DongdaemunResult {
    let mut decision = accept_candidate(graph, baseline, candidate, config, ws);
    decision.audit.accepted = false;
    decision.audit.status = status;
    decision.audit.effective_delta_q = 0.0;
    decision.audit.split_iterations = split_iterations;
    decision.audit.trim_moves_proposed = trim_moves_proposed;
    decision.audit.trim_moves_committed = trim_moves_committed;
    DongdaemunResult {
        clustering: baseline.clone(),
        diagnostic_clustering: Some(candidate.clone()),
        audit: decision.audit,
    }
}

fn cluster_weight_summary(
    graph: &Graph,
    clustering: &Clustering,
    target_max_weight: f64,
    ws: &mut Workspace,
) -> ClusterWeightSummary {
    assert_eq!(clustering.n_nodes, graph.n_nodes);
    assert_eq!(graph.node_weights.len(), graph.n_nodes);

    clustering.fill_cluster_groups_and_weights(&graph.node_weights, ws);
    let weights = ws.cw[..clustering.n_clusters].to_vec();
    let mut n_oversize = 0usize;
    let mut max_weight = 0.0f64;

    for &weight in &weights {
        if weight > target_max_weight {
            n_oversize += 1;
        }
        if weight > max_weight {
            max_weight = weight;
        }
    }

    ClusterWeightSummary {
        weights,
        target_max_weight,
        n_oversize,
        max_weight,
        target_max_satisfied: n_oversize == 0,
    }
}

fn current_oversize_clusters(summary: &ClusterWeightSummary) -> Vec<ClusterId> {
    let mut oversize = summary
        .weights
        .iter()
        .enumerate()
        .filter_map(|(cluster, &weight)| {
            if weight > summary.target_max_weight {
                Some((Clustering::index_to_cluster(cluster), weight))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();

    oversize.sort_by(|(left_id, left_weight), (right_id, right_weight)| {
        right_weight
            .total_cmp(left_weight)
            .then_with(|| left_id.cmp(right_id))
    });

    oversize.into_iter().map(|(cluster, _)| cluster).collect()
}

fn apply_trim_move_prefix(
    baseline: &Clustering,
    moves: &[OversizeBoundaryMove],
    count: usize,
) -> Clustering {
    assert!(count <= moves.len());

    let mut out = baseline.clone();
    for mv in moves.iter().take(count) {
        let node = usize::try_from(mv.node).expect("trim node id exceeds usize::MAX");
        let target = u32::try_from(mv.target).expect("trim target id exceeds u32::MAX");
        assert!(node < out.n_nodes);
        out.clusters[node] = target;
        out.n_clusters = out.n_clusters.max(target as usize + 1);
    }
    out.remove_empty_clusters();
    out
}

fn quality_floor_prefix_move_count(
    graph: &Graph,
    baseline: &Clustering,
    moves: &[OversizeBoundaryMove],
    resolution: f64,
    quality_floor: f64,
) -> usize {
    let cpm = CPM::new(resolution);
    let quality_before = cpm.quality(graph, baseline);
    let mut predicted_quality = quality_before;
    let mut candidate_count = 0usize;

    for (idx, mv) in moves.iter().enumerate() {
        predicted_quality += mv.delta_q;
        if predicted_quality >= quality_floor {
            candidate_count = idx + 1;
        }
    }

    while candidate_count > 0 {
        let candidate = apply_trim_move_prefix(baseline, moves, candidate_count);
        if cpm.quality(graph, &candidate) >= quality_floor {
            return candidate_count;
        }
        candidate_count -= 1;
    }

    0
}

#[allow(dead_code)]
fn cpm_delta(graph: &Graph, before: &Clustering, after: &Clustering, resolution: f64) -> f64 {
    assert_eq!(before.n_nodes, graph.n_nodes);
    assert_eq!(after.n_nodes, graph.n_nodes);

    let cpm = CPM::new(resolution);
    cpm.quality(graph, after) - cpm.quality(graph, before)
}

fn accept_candidate(
    graph: &Graph,
    baseline: &Clustering,
    candidate: &Clustering,
    config: &DongdaemunConfig,
    ws: &mut Workspace,
) -> AcceptanceDecision {
    assert_eq!(baseline.n_nodes, graph.n_nodes);
    assert_eq!(candidate.n_nodes, graph.n_nodes);

    let cpm = CPM::new(config.resolution);
    let quality_before = cpm.quality(graph, baseline);
    let quality_after_candidate = cpm.quality(graph, candidate);
    let delta_q = quality_after_candidate - quality_before;

    let before_summary = cluster_weight_summary(graph, baseline, config.target_max_weight, ws);
    let candidate_summary = cluster_weight_summary(graph, candidate, config.target_max_weight, ws);
    let quality_ok = delta_q >= config.quality_floor_delta;
    let accepted = match config.policy {
        DongdaemunPolicy::QualityFirst => quality_ok,
        DongdaemunPolicy::HardCap => quality_ok && candidate_summary.target_max_satisfied,
    };

    let status = if accepted {
        DongdaemunStatus::Committed
    } else if !quality_ok {
        DongdaemunStatus::QualityBelowFloor
    } else {
        DongdaemunStatus::HardCapNotSatisfied
    };

    let membership_changed = baseline.clusters != candidate.clusters;
    let diagnostic_clustering = if accepted || !membership_changed {
        None
    } else {
        Some(candidate.clone())
    };
    let effective_clustering = if accepted {
        candidate.clone()
    } else {
        baseline.clone()
    };

    AcceptanceDecision {
        effective_clustering,
        diagnostic_clustering,
        audit: DongdaemunAudit {
            accepted,
            status,
            quality_before,
            quality_after_candidate,
            candidate_delta_q: delta_q,
            effective_delta_q: if accepted { delta_q } else { 0.0 },
            final_delta_q: delta_q,
            target_max_satisfied: candidate_summary.target_max_satisfied,
            n_oversize_before: before_summary.n_oversize,
            n_oversize_after_candidate: candidate_summary.n_oversize,
            max_weight_before: before_summary.max_weight,
            max_weight_after_candidate: candidate_summary.max_weight,
            split_iterations: Vec::new(),
            trim_moves_committed: 0,
            trim_moves_proposed: 0,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_graph_with_weights(node_weights: &[f64]) -> Graph {
        Graph::from_edge_list_weighted(node_weights.len(), &[], &[], &[], node_weights)
    }

    fn paired_graph(weight: f64) -> Graph {
        Graph::from_edge_list(4, &[0, 2], &[1, 3], &[weight, weight])
    }

    fn escaped_fragment_graph() -> Graph {
        Graph::from_edge_list(3, &[0, 1], &[1, 2], &[0.1, 10.0])
    }

    fn escaped_fragment_config(apply_iterations: usize) -> DongdaemunConfig {
        let mut config = DongdaemunConfig::default_for_quality_first(0.1, 1.5);
        config.apply_iterations = apply_iterations;
        config.gamma_multipliers = vec![10.0];
        config.min_core_weight = 1.0;
        config.randomness = 0.0;
        config.repair_epsilon = 0.0;
        config.pair_seeded = false;
        config
    }

    fn trim_move(node: u64, target: u64, delta_q: f64) -> OversizeBoundaryMove {
        OversizeBoundaryMove {
            source: 0,
            target,
            node,
            node_weight: 1.0,
            delta_q,
            source_weight_before: 0.0,
            source_weight_after: 0.0,
            target_weight_before: 0.0,
            target_weight_after: 0.0,
        }
    }

    fn split_probe(
        cluster: u64,
        gamma_multiplier: f64,
        net_delta_q: f64,
        doc_weight: f64,
        largest_source_unit_fraction: f64,
        escaped_source_weight: f64,
    ) -> SplitMergeRepairProbe {
        SplitMergeRepairProbe {
            cluster,
            gamma_multiplier,
            probe_resolution: gamma_multiplier,
            block_count: doc_weight as u64,
            doc_weight,
            induced_directed_edges: 0,
            n_parts: 2,
            core_part_count: 2,
            singleton_weight: 0.0,
            cut_weight: 0.0,
            split_delta_q_base: net_delta_q,
            split_delta_q_probe: net_delta_q,
            repair_quotient_edges: 0,
            repair_merge_count: 0,
            repair_delta_q: 0.0,
            net_delta_q,
            final_source_units: 2,
            retained_source_units: 1,
            escaped_source_units: if escaped_source_weight > 0.0 { 1 } else { 0 },
            escaped_source_weight,
            final_small_source_units: 0,
            final_small_source_weight: 0.0,
            largest_source_unit_fraction,
            restored_source_cluster: false,
        }
    }

    #[test]
    fn split_selector_filters_negative_delta_and_restored_source() {
        let mut restored = split_probe(2, 1.05, 10.0, 100.0, 0.5, 10.0);
        restored.restored_source_cluster = true;
        let probes = vec![
            split_probe(1, 1.05, -0.1, 100.0, 0.5, 10.0),
            restored,
            split_probe(4, 1.05, 0.0, 100.0, 1.0, 0.0),
            split_probe(3, 1.05, 2.0, 100.0, 0.5, 10.0),
        ];

        let selected =
            select_split_repair_candidates(&probes, DongdaemunPolicy::QualityFirst, 50.0);

        assert_eq!(
            selected,
            vec![SelectedSplitRepairCandidate {
                cluster: 3,
                gamma_multiplier: 1.05,
            }]
        );
    }

    #[test]
    fn split_selector_picks_one_deterministic_gamma_per_source_cluster() {
        let probes = vec![
            split_probe(1, 1.05, 2.0, 100.0, 0.6, 5.0),
            split_probe(1, 1.10, 3.0, 100.0, 0.4, 4.0),
            split_probe(2, 1.05, 1.0, 100.0, 0.7, 3.0),
        ];

        let selected =
            select_split_repair_candidates(&probes, DongdaemunPolicy::QualityFirst, 50.0);

        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0].cluster, 1);
        assert_eq!(selected[0].gamma_multiplier, 1.10);
        assert_eq!(selected[1].cluster, 2);
    }

    #[test]
    fn split_selector_tie_breaks_by_cluster_then_lower_gamma() {
        let probes = vec![
            split_probe(2, 1.10, 1.0, 100.0, 0.5, 10.0),
            split_probe(1, 1.10, 1.0, 100.0, 0.5, 10.0),
            split_probe(1, 1.05, 1.0, 100.0, 0.5, 10.0),
            split_probe(2, 1.05, 1.0, 100.0, 0.5, 10.0),
        ];

        let selected =
            select_split_repair_candidates(&probes, DongdaemunPolicy::QualityFirst, 50.0);

        assert_eq!(
            selected,
            vec![
                SelectedSplitRepairCandidate {
                    cluster: 1,
                    gamma_multiplier: 1.05,
                },
                SelectedSplitRepairCandidate {
                    cluster: 2,
                    gamma_multiplier: 1.05,
                },
            ]
        );
    }

    #[test]
    fn hard_cap_split_selector_prioritizes_oversize_reduction() {
        let probes = vec![
            split_probe(1, 1.05, 10.0, 300.0, 0.8, 20.0),
            split_probe(2, 1.05, 2.0, 300.0, 0.5, 10.0),
        ];

        let selected = select_split_repair_candidates(&probes, DongdaemunPolicy::HardCap, 200.0);

        assert_eq!(selected[0].cluster, 2);
    }

    #[test]
    fn hard_cap_split_selector_penalizes_receiver_oversize() {
        let probes = [
            split_probe(1, 1.05, 10.0, 300.0, 0.5, 20.0),
            split_probe(2, 1.05, 2.0, 300.0, 0.5, 10.0),
        ];

        let selected = select_split_repair_selection_candidates(
            [
                SplitRepairSelectionCandidate {
                    probe: &probes[0],
                    receiver_remaining_oversize_after: 25.0,
                    receiver_oversize_increase: 25.0,
                },
                SplitRepairSelectionCandidate {
                    probe: &probes[1],
                    receiver_remaining_oversize_after: 0.0,
                    receiver_oversize_increase: 0.0,
                },
            ],
            DongdaemunPolicy::HardCap,
            200.0,
        );

        assert_eq!(selected[0].cluster, 2);
    }

    #[test]
    fn split_loop_stops_at_apply_iterations() {
        let graph = escaped_fragment_graph();
        let baseline = Clustering::from_assignments(vec![0, 0, 1]);
        let config = escaped_fragment_config(1);
        let quality_floor = CPM::new(config.resolution).quality(&graph, &baseline);
        let mut ws = Workspace::new(graph.n_nodes);

        let outcome =
            run_split_repair_iterations(&graph, &baseline, &config, quality_floor, &mut ws);

        assert_eq!(outcome.clustering.clusters, vec![0, 1, 1]);
        assert!(outcome.diagnostic_clustering.is_none());
        assert_eq!(outcome.iterations.len(), 1);
        assert_eq!(outcome.iterations[0].iteration, 1);
        assert_eq!(outcome.iterations[0].status, DongdaemunStatus::Committed);
    }

    #[test]
    fn split_loop_stops_after_terminal_non_committed_status() {
        let graph = escaped_fragment_graph();
        let baseline = Clustering::from_assignments(vec![0, 0, 1]);
        let config = escaped_fragment_config(4);
        let quality_floor = CPM::new(config.resolution).quality(&graph, &baseline);
        let mut ws = Workspace::new(graph.n_nodes);

        let outcome =
            run_split_repair_iterations(&graph, &baseline, &config, quality_floor, &mut ws);

        assert_eq!(outcome.clustering.clusters, vec![0, 1, 1]);
        assert_eq!(outcome.iterations.len(), 2);
        assert_eq!(outcome.iterations[0].status, DongdaemunStatus::Committed);
        assert_eq!(
            outcome.iterations[1].status,
            DongdaemunStatus::NoSelectedCandidates
        );
    }

    #[test]
    fn split_loop_records_terminal_no_current_oversize_status() {
        let graph = empty_graph_with_weights(&[1.0, 1.0]);
        let baseline = Clustering::from_assignments(vec![0, 1]);
        let config = DongdaemunConfig::default_for_quality_first(0.1, 1.0);
        let quality_floor = CPM::new(config.resolution).quality(&graph, &baseline);
        let mut ws = Workspace::new(graph.n_nodes);

        let outcome =
            run_split_repair_iterations(&graph, &baseline, &config, quality_floor, &mut ws);

        assert_eq!(outcome.clustering.clusters, baseline.clusters);
        assert_eq!(outcome.iterations.len(), 1);
        assert_eq!(
            outcome.iterations[0].status,
            DongdaemunStatus::NoCurrentOversizeCandidates
        );
        assert!(outcome.iterations[0].candidate_clusters.is_empty());
    }

    #[test]
    fn split_iteration_exact_delta_is_iteration_local() {
        let graph = escaped_fragment_graph();
        let baseline = Clustering::from_assignments(vec![0, 0, 1]);
        let config = escaped_fragment_config(1);
        let quality_floor = CPM::new(config.resolution).quality(&graph, &baseline);
        let mut ws = Workspace::new(graph.n_nodes);

        let outcome =
            run_split_repair_iterations(&graph, &baseline, &config, quality_floor, &mut ws);
        let expected_delta = cpm_delta(&graph, &baseline, &outcome.clustering, config.resolution);

        assert_eq!(outcome.iterations.len(), 1);
        assert!((outcome.iterations[0].exact_delta_q - expected_delta).abs() < 1e-12);
    }

    #[test]
    fn config_validation_rejects_invalid_core_fields() {
        let mut config = DongdaemunConfig::default_for_quality_first(0.5, 10.0);
        assert!(config.validate().is_ok());

        config.resolution = f64::NAN;
        assert!(config.validate().unwrap_err().contains("resolution"));

        config = DongdaemunConfig::default_for_quality_first(0.5, 0.0);
        assert!(config.validate().unwrap_err().contains("target_max_weight"));

        config = DongdaemunConfig::default_for_quality_first(0.5, 10.0);
        config.gamma_multipliers[0] = f64::INFINITY;
        assert!(config
            .validate()
            .unwrap_err()
            .contains("gamma_multipliers[0]"));

        config = DongdaemunConfig::default_for_quality_first(0.5, 10.0);
        config.gamma_multipliers[0] = 0.0;
        assert!(config
            .validate()
            .unwrap_err()
            .contains("gamma_multipliers[0]"));

        config = DongdaemunConfig::default_for_quality_first(0.5, 10.0);
        config.apply_iterations = 0;
        assert!(config.validate().unwrap_err().contains("apply_iterations"));
    }

    #[test]
    fn trim_prefix_membership_applies_exactly_first_moves() {
        let baseline = Clustering::from_assignments(vec![0, 0, 1, 2]);
        let moves = vec![trim_move(0, 1, 1.0), trim_move(1, 1, 0.5)];

        let one = apply_trim_move_prefix(&baseline, &moves, 1);
        let two = apply_trim_move_prefix(&baseline, &moves, 2);

        assert_eq!(one.clusters, vec![1, 0, 1, 2]);
        assert_eq!(one.n_clusters, 3);
        assert_eq!(two.clusters, vec![0, 0, 0, 1]);
        assert_eq!(two.n_clusters, 2);
    }

    #[test]
    fn quality_floor_prefix_chooses_largest_exact_valid_prefix() {
        let graph = empty_graph_with_weights(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        let baseline = Clustering::from_assignments(vec![0, 0, 0, 0, 1, 2]);
        let moves = vec![trim_move(0, 1, 1.0), trim_move(1, 2, 0.5)];
        let cpm = CPM::new(0.5);
        let quality_floor = cpm.quality(&graph, &baseline) + 1.25;

        let count = quality_floor_prefix_move_count(&graph, &baseline, &moves, 0.5, quality_floor);

        assert_eq!(count, 2);
    }

    #[test]
    fn quality_floor_prefix_rolls_back_to_zero_when_none_pass() {
        let graph = empty_graph_with_weights(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        let baseline = Clustering::from_assignments(vec![0, 0, 0, 0, 1, 2]);
        let moves = vec![trim_move(0, 1, 0.1), trim_move(1, 2, 0.1)];
        let cpm = CPM::new(0.5);
        let quality_floor = cpm.quality(&graph, &baseline) + 2.0;

        let count = quality_floor_prefix_move_count(&graph, &baseline, &moves, 0.5, quality_floor);

        assert_eq!(count, 0);
    }

    #[test]
    fn oversize_clusters_sorted_by_weight_then_stable_id() {
        let graph = empty_graph_with_weights(&[2.0, 4.0, 3.0, 3.0, 6.0]);
        let clustering = Clustering::from_assignments(vec![0, 1, 2, 2, 3]);
        let mut ws = Workspace::new(graph.n_nodes);

        let summary = cluster_weight_summary(&graph, &clustering, 3.0, &mut ws);
        let oversize = current_oversize_clusters(&summary);

        assert_eq!(oversize, vec![2, 3, 1]);
    }

    #[test]
    fn quality_first_accepts_non_regressing_candidate_with_residual_oversize() {
        let graph = paired_graph(1.0);
        let baseline = Clustering::from_assignments(vec![0, 0, 0, 0]);
        let candidate = Clustering::from_assignments(vec![0, 0, 1, 1]);
        let config = DongdaemunConfig::default_for_quality_first(0.5, 1.0);
        let mut ws = Workspace::new(graph.n_nodes);

        let decision = accept_candidate(&graph, &baseline, &candidate, &config, &mut ws);

        assert!(decision.audit.accepted);
        assert_eq!(decision.audit.status, DongdaemunStatus::Committed);
        assert!(decision.audit.final_delta_q > 0.0);
        assert!(!decision.audit.target_max_satisfied);
        assert_eq!(decision.effective_clustering.clusters, candidate.clusters);
        assert!(decision.diagnostic_clustering.is_none());
    }

    #[test]
    fn quality_first_rejects_negative_exact_delta_q() {
        let graph = paired_graph(1.0);
        let baseline = Clustering::from_assignments(vec![0, 0, 1, 1]);
        let candidate = Clustering::from_assignments(vec![0, 0, 0, 0]);
        let config = DongdaemunConfig::default_for_quality_first(2.0, 10.0);
        let mut ws = Workspace::new(graph.n_nodes);

        let decision = accept_candidate(&graph, &baseline, &candidate, &config, &mut ws);

        assert!(!decision.audit.accepted);
        assert_eq!(decision.audit.status, DongdaemunStatus::QualityBelowFloor);
        assert!(decision.audit.final_delta_q < 0.0);
        assert_eq!(decision.effective_clustering.clusters, baseline.clusters);
    }

    #[test]
    fn hard_cap_rejects_non_cap_satisfying_candidate_even_when_delta_nonnegative() {
        let graph = paired_graph(1.0);
        let baseline = Clustering::from_assignments(vec![0, 0, 0, 0]);
        let candidate = Clustering::from_assignments(vec![0, 0, 1, 1]);
        let config = DongdaemunConfig::default_for_hard_cap(0.5, 1.0);
        let mut ws = Workspace::new(graph.n_nodes);

        let decision = accept_candidate(&graph, &baseline, &candidate, &config, &mut ws);

        assert!(!decision.audit.accepted);
        assert_eq!(decision.audit.status, DongdaemunStatus::HardCapNotSatisfied);
        assert!(decision.audit.final_delta_q > 0.0);
        assert_eq!(decision.effective_clustering.clusters, baseline.clusters);
    }

    #[test]
    fn hard_cap_accepts_candidate_only_when_quality_and_cap_pass() {
        let graph = empty_graph_with_weights(&[1.0, 1.0, 1.0, 1.0]);
        let baseline = Clustering::from_assignments(vec![0, 0, 0, 0]);
        let candidate = Clustering::singleton(4);
        let config = DongdaemunConfig::default_for_hard_cap(0.5, 1.0);
        let mut ws = Workspace::new(graph.n_nodes);

        let decision = accept_candidate(&graph, &baseline, &candidate, &config, &mut ws);

        assert!(decision.audit.accepted);
        assert_eq!(decision.audit.status, DongdaemunStatus::Committed);
        assert!(decision.audit.final_delta_q > 0.0);
        assert!(decision.audit.target_max_satisfied);
        assert_eq!(decision.effective_clustering.clusters, candidate.clusters);
    }

    #[test]
    fn changed_rejected_candidate_is_preserved_as_diagnostic() {
        let graph = paired_graph(1.0);
        let baseline = Clustering::from_assignments(vec![0, 0, 1, 1]);
        let candidate = Clustering::from_assignments(vec![0, 0, 0, 0]);
        let config = DongdaemunConfig::default_for_quality_first(2.0, 10.0);
        let mut ws = Workspace::new(graph.n_nodes);

        let decision = accept_candidate(&graph, &baseline, &candidate, &config, &mut ws);

        assert_eq!(decision.effective_clustering.clusters, baseline.clusters);
        assert_eq!(
            decision
                .diagnostic_clustering
                .as_ref()
                .map(|c| c.clusters.as_slice()),
            Some(candidate.clusters.as_slice())
        );
    }

    #[test]
    fn cpm_delta_matches_exact_quality_difference() {
        let graph = paired_graph(1.0);
        let before = Clustering::from_assignments(vec![0, 0, 1, 1]);
        let after = Clustering::from_assignments(vec![0, 0, 0, 0]);

        let delta = cpm_delta(&graph, &before, &after, 2.0);

        assert_eq!(delta, -8.0);
    }
}
