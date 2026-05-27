//! Leiden algorithm: move → refine → aggregate → recurse.
//!
//! Port of CWTS LeidenAlgorithm.java.

use crate::clustering::Clustering;
use crate::contraction::{create_reduced_network, create_reduced_network_from_starts};
use crate::fast_local_move;
use crate::graph::Graph;
use crate::local_merge;
use crate::trace;
use crate::workspace::Workspace;
use rand::Rng;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::OnceLock;
use std::time::Instant;

const DEFAULT_STREAMING_REFINEMENT_MIN_DIRECTED_EDGES: usize = 1_000_000;
const DEFAULT_CONVERGENCE_PATIENCE: usize = 3;
const DEFAULT_CONVERGENCE_MIN_REL_CLUSTER_DELTA: f64 = 1e-4;
const DEFAULT_RECURSION_GUARD_MIN_DIRECTED_EDGES: usize = 100_000_000;
const DEFAULT_RECURSION_MIN_REL_NODE_DELTA: f64 = 1e-4;

fn streaming_refinement_min_directed_edges() -> usize {
    static MIN_EDGES: OnceLock<usize> = OnceLock::new();
    *MIN_EDGES.get_or_init(|| {
        std::env::var("SCISCAPE_STREAMING_REFINEMENT_MIN_DIRECTED_EDGES")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(DEFAULT_STREAMING_REFINEMENT_MIN_DIRECTED_EDGES)
    })
}

fn convergence_patience() -> usize {
    static PATIENCE: OnceLock<usize> = OnceLock::new();
    *PATIENCE.get_or_init(|| {
        std::env::var("SCISCAPE_LEIDEN_CONVERGENCE_PATIENCE")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(DEFAULT_CONVERGENCE_PATIENCE)
    })
}

fn convergence_min_rel_cluster_delta() -> f64 {
    static MIN_DELTA: OnceLock<f64> = OnceLock::new();
    *MIN_DELTA.get_or_init(|| {
        std::env::var("SCISCAPE_LEIDEN_CONVERGENCE_MIN_REL_CLUSTER_DELTA")
            .ok()
            .and_then(|value| value.parse::<f64>().ok())
            .filter(|value| value.is_finite() && *value >= 0.0)
            .unwrap_or(DEFAULT_CONVERGENCE_MIN_REL_CLUSTER_DELTA)
    })
}

fn recursion_guard_min_directed_edges() -> usize {
    static MIN_EDGES: OnceLock<usize> = OnceLock::new();
    *MIN_EDGES.get_or_init(|| {
        std::env::var("SCISCAPE_LEIDEN_RECURSION_GUARD_MIN_DIRECTED_EDGES")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(DEFAULT_RECURSION_GUARD_MIN_DIRECTED_EDGES)
    })
}

fn recursion_min_rel_node_delta() -> f64 {
    static MIN_DELTA: OnceLock<f64> = OnceLock::new();
    *MIN_DELTA.get_or_init(|| {
        std::env::var("SCISCAPE_LEIDEN_RECURSION_MIN_REL_NODE_DELTA")
            .ok()
            .and_then(|value| value.parse::<f64>().ok())
            .filter(|value| value.is_finite() && *value >= 0.0)
            .unwrap_or(DEFAULT_RECURSION_MIN_REL_NODE_DELTA)
    })
}

fn should_trace_graph(graph: &Graph) -> bool {
    trace::should_trace_edges(graph.n_edges)
}

fn should_trace_graph_detail(_graph: &Graph) -> bool {
    trace::verbose()
}

macro_rules! trace_graph {
    ($graph:expr, $($arg:tt)*) => {{
        if should_trace_graph_detail($graph) {
            trace::emit(format_args!($($arg)*));
        }
    }};
}

macro_rules! trace_graph_summary {
    ($graph:expr, $($arg:tt)*) => {{
        if should_trace_graph($graph) {
            trace::emit(format_args!($($arg)*));
        }
    }};
}

/// Configuration for the Leiden algorithm.
#[derive(Clone, Debug)]
pub struct LeidenConfig {
    pub resolution: f64,
    pub n_iterations: usize, // 0 = until convergence
    pub randomness: f64,
    pub randomness_schedule: Vec<f64>,
    pub seed: u64,
}

impl Default for LeidenConfig {
    fn default() -> Self {
        LeidenConfig {
            resolution: 1.0,
            n_iterations: 10,
            randomness: 0.01,
            randomness_schedule: Vec::new(),
            seed: 0,
        }
    }
}

impl LeidenConfig {
    #[inline]
    fn randomness_for_iteration(&self, iteration: usize) -> f64 {
        if self.randomness_schedule.is_empty() {
            self.randomness
        } else {
            self.randomness_schedule[iteration.min(self.randomness_schedule.len() - 1)]
        }
    }
}

/// Result of a Leiden run.
#[derive(Clone, Debug)]
pub struct LeidenResult {
    pub clustering: Clustering,
    pub quality: f64,
    pub n_iterations_used: usize,
}

/// Opt-in experimental Dongdaemun refinement configuration for Leiden.
///
/// This is intentionally separate from `LeidenConfig` so the standard Leiden
/// API and CWTS-compatible defaults remain unchanged.
#[derive(Clone, Debug)]
pub struct DongdaemunRefinementConfig {
    pub target_max_weight: f64,
    pub soft_min_ratio: f64,
    pub max_extra_parents_per_iteration: usize,
    pub max_extra_children_per_parent: usize,
    pub parent_selection_policy: ParentSelectionPolicy,
    pub max_singleton_weight_fraction: f64,
    pub min_largest_child_fraction_improvement: f64,
    pub gamma_multipliers: Vec<f64>,
    pub seed_perturbations: usize,
    pub use_quotient_diagnostic: bool,
    pub use_baseline_repair: bool,
    pub baseline_repair_policy: BaselineRepairPolicy,
    pub baseline_repair_replace_min_parent_ratio: f64,
    pub baseline_repair_epsilon: f64,
    pub candidate_quality_policy: CandidateQualityPolicy,
    pub min_candidate_delta_q: f64,
    pub adaptive_plateau_quality_band: f64,
    pub use_final_quality_guard: bool,
    pub min_final_quality_delta: f64,
    pub adaptive_probe_mode: AdaptiveProbeMode,
    pub adaptive_probe_perturbations: usize,
    pub adaptive_probe_targets: Vec<AdaptiveProbeTarget>,
    pub adaptive_probe_tolerance_parent_weight: f64,
    pub adaptive_probe_include_node_order_control: bool,
    pub adaptive_probe_commit_min_gain_parent_weight: f64,
    pub adaptive_probe_max_commits_total: usize,
    pub adaptive_probe_max_commits_per_depth: usize,
    pub adaptive_probe_commit_sources: Vec<String>,
    pub adaptive_probe_commit_strategy: AdaptiveProbeCommitStrategy,
    pub adaptive_near_tie_probe_mode: AdaptiveNearTieProbeMode,
    pub adaptive_near_tie_margin_parent_weight: f64,
    pub adaptive_near_tie_randomness: f64,
    pub adaptive_near_tie_max_decisions_per_parent: usize,
    pub adaptive_local_shake_mode: AdaptiveLocalShakeMode,
    pub adaptive_local_shake_arms: Vec<AdaptiveLocalShakeArm>,
    pub adaptive_local_shake_max_arms_per_parent: usize,
    pub adaptive_local_shake_max_candidates_per_parent: usize,
    pub adaptive_local_shake_min_gain_parent_weight: f64,
    pub adaptive_local_shake_shape_eps: f64,
    pub adaptive_local_shake_arm_priority: Vec<AdaptiveLocalShakeArm>,
    pub adaptive_local_shake_near_tie_min_count: usize,
    pub adaptive_local_shake_resolution_down_multipliers: Vec<f64>,
    pub adaptive_local_shake_resolution_up_multipliers: Vec<f64>,
    pub adaptive_local_shake_resolution_up_min_parent_ratio: f64,
    pub adaptive_local_shake_resolution_down_max_parent_ratio: f64,
    pub adaptive_local_shake_large_child_fraction: f64,
    pub adaptive_local_shake_singleton_fraction: f64,
    pub adaptive_local_shake_seed_perturbations: usize,
    pub adaptive_local_shake_seed_margin_count: usize,
    pub adaptive_local_shake_near_tie_margin_parent_weight: f64,
    pub adaptive_local_shake_near_tie_randomness: f64,
    pub adaptive_local_shake_final_guard_mode: AdaptiveLocalShakeFinalGuardMode,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CandidateQualityPolicy {
    Structural,
    QualityGuardedStructural,
    QualityFloor,
    QualityFirst,
    Selective,
    PressureAware,
    AdaptivePlateau,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BaselineRepairPolicy {
    Replace,
    Augment,
    Adaptive,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ParentSelectionPolicy {
    Weight,
    PressureBoundary,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdaptiveProbeMode {
    Off,
    TraceOnly,
    ApplyIfWin,
    ConservativeApply,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdaptiveProbeCommitStrategy {
    OnlineFirst,
    BestQf,
    RiskAdjusted,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdaptiveNearTieProbeMode {
    Off,
    TraceOnly,
    Candidate,
    QfReplace,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdaptiveLocalShakeMode {
    Off,
    TraceOnly,
    QfReplace,
    PressureGuarded,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum AdaptiveLocalShakeArm {
    NearTieRefinement,
    ResolutionUp,
    ResolutionDown,
    SeedLocalRefinement,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdaptiveLocalShakeFinalGuardMode {
    None,
    RunnerAudit,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AdaptiveProbeTarget {
    pub depth: usize,
    pub parent_id: usize,
    pub parent_visit_index: usize,
}

impl Default for DongdaemunRefinementConfig {
    fn default() -> Self {
        Self {
            target_max_weight: 0.0,
            soft_min_ratio: 1.0,
            max_extra_parents_per_iteration: 0,
            max_extra_children_per_parent: 64,
            parent_selection_policy: ParentSelectionPolicy::Weight,
            max_singleton_weight_fraction: 0.05,
            min_largest_child_fraction_improvement: 0.05,
            gamma_multipliers: vec![1.02, 1.05, 1.10, 1.15, 1.20, 1.25],
            seed_perturbations: 0,
            use_quotient_diagnostic: false,
            use_baseline_repair: false,
            baseline_repair_policy: BaselineRepairPolicy::Replace,
            baseline_repair_replace_min_parent_ratio: 1.05,
            baseline_repair_epsilon: 0.0,
            candidate_quality_policy: CandidateQualityPolicy::Structural,
            min_candidate_delta_q: 0.0,
            adaptive_plateau_quality_band: 0.0,
            use_final_quality_guard: false,
            min_final_quality_delta: 0.0,
            adaptive_probe_mode: AdaptiveProbeMode::Off,
            adaptive_probe_perturbations: 0,
            adaptive_probe_targets: Vec::new(),
            adaptive_probe_tolerance_parent_weight: 1e-6,
            adaptive_probe_include_node_order_control: false,
            adaptive_probe_commit_min_gain_parent_weight: 0.0,
            adaptive_probe_max_commits_total: 0,
            adaptive_probe_max_commits_per_depth: 0,
            adaptive_probe_commit_sources: Vec::new(),
            adaptive_probe_commit_strategy: AdaptiveProbeCommitStrategy::OnlineFirst,
            adaptive_near_tie_probe_mode: AdaptiveNearTieProbeMode::Off,
            adaptive_near_tie_margin_parent_weight: 0.0,
            adaptive_near_tie_randomness: 0.0,
            adaptive_near_tie_max_decisions_per_parent: 0,
            adaptive_local_shake_mode: AdaptiveLocalShakeMode::Off,
            adaptive_local_shake_arms: Vec::new(),
            adaptive_local_shake_max_arms_per_parent: 0,
            adaptive_local_shake_max_candidates_per_parent: 0,
            adaptive_local_shake_min_gain_parent_weight: 0.0,
            adaptive_local_shake_shape_eps: 1e-12,
            adaptive_local_shake_arm_priority: Vec::new(),
            adaptive_local_shake_near_tie_min_count: 1,
            adaptive_local_shake_resolution_down_multipliers: Vec::new(),
            adaptive_local_shake_resolution_up_multipliers: Vec::new(),
            adaptive_local_shake_resolution_up_min_parent_ratio: 1.0,
            adaptive_local_shake_resolution_down_max_parent_ratio: 1.0,
            adaptive_local_shake_large_child_fraction: 0.95,
            adaptive_local_shake_singleton_fraction: 0.05,
            adaptive_local_shake_seed_perturbations: 0,
            adaptive_local_shake_seed_margin_count: 2,
            adaptive_local_shake_near_tie_margin_parent_weight: 0.0,
            adaptive_local_shake_near_tie_randomness: 0.0,
            adaptive_local_shake_final_guard_mode: AdaptiveLocalShakeFinalGuardMode::None,
        }
    }
}

impl DongdaemunRefinementConfig {
    pub fn validate(&self) -> Result<(), String> {
        if !self.target_max_weight.is_finite() || self.target_max_weight <= 0.0 {
            return Err("target_max_weight must be finite and positive".to_string());
        }
        if !self.soft_min_ratio.is_finite() || self.soft_min_ratio < 0.0 {
            return Err("soft_min_ratio must be finite and non-negative".to_string());
        }
        if self.max_extra_parents_per_iteration == 0 {
            return Err("max_extra_parents_per_iteration must be positive".to_string());
        }
        if self.max_extra_children_per_parent < 2 {
            return Err("max_extra_children_per_parent must be at least 2".to_string());
        }
        if !self.max_singleton_weight_fraction.is_finite()
            || !(0.0..=1.0).contains(&self.max_singleton_weight_fraction)
        {
            return Err("max_singleton_weight_fraction must be in [0, 1]".to_string());
        }
        if !self.min_largest_child_fraction_improvement.is_finite()
            || self.min_largest_child_fraction_improvement < 0.0
        {
            return Err(
                "min_largest_child_fraction_improvement must be finite and non-negative"
                    .to_string(),
            );
        }
        if self.gamma_multipliers.is_empty() && self.seed_perturbations == 0 {
            return Err(
                "gamma_multipliers must not be empty unless seed_perturbations is positive"
                    .to_string(),
            );
        }
        for multiplier in &self.gamma_multipliers {
            if !multiplier.is_finite() || *multiplier <= 0.0 {
                return Err("gamma_multipliers must be finite and positive".to_string());
            }
        }
        if !self.baseline_repair_epsilon.is_finite() || self.baseline_repair_epsilon < 0.0 {
            return Err("baseline_repair_epsilon must be finite and non-negative".to_string());
        }
        if !self.baseline_repair_replace_min_parent_ratio.is_finite()
            || self.baseline_repair_replace_min_parent_ratio < 0.0
        {
            return Err(
                "baseline_repair_replace_min_parent_ratio must be finite and non-negative"
                    .to_string(),
            );
        }
        if !self.min_candidate_delta_q.is_finite() {
            return Err("min_candidate_delta_q must be finite".to_string());
        }
        if !self.adaptive_plateau_quality_band.is_finite()
            || self.adaptive_plateau_quality_band < 0.0
        {
            return Err(
                "adaptive_plateau_quality_band must be finite and non-negative".to_string(),
            );
        }
        if !self.min_final_quality_delta.is_finite() {
            return Err("min_final_quality_delta must be finite".to_string());
        }
        if !self.adaptive_probe_tolerance_parent_weight.is_finite()
            || self.adaptive_probe_tolerance_parent_weight < 0.0
        {
            return Err(
                "adaptive_probe_tolerance_parent_weight must be finite and non-negative"
                    .to_string(),
            );
        }
        if !self
            .adaptive_probe_commit_min_gain_parent_weight
            .is_finite()
            || self.adaptive_probe_commit_min_gain_parent_weight < 0.0
        {
            return Err(
                "adaptive_probe_commit_min_gain_parent_weight must be finite and non-negative"
                    .to_string(),
            );
        }
        for source in &self.adaptive_probe_commit_sources {
            if !adaptive_probe_source_is_valid(source) {
                return Err(format!(
                    "adaptive_probe_commit_sources must contain only same_gamma_probe, node_order_control, or near_tie_refinement_probe, got {source:?}"
                ));
            }
        }
        if !self.adaptive_near_tie_margin_parent_weight.is_finite()
            || self.adaptive_near_tie_margin_parent_weight < 0.0
        {
            return Err(
                "adaptive_near_tie_margin_parent_weight must be finite and non-negative"
                    .to_string(),
            );
        }
        if !self.adaptive_near_tie_randomness.is_finite()
            || !(0.0..=1.0).contains(&self.adaptive_near_tie_randomness)
        {
            return Err("adaptive_near_tie_randomness must be in [0, 1]".to_string());
        }
        if self.adaptive_local_shake_mode != AdaptiveLocalShakeMode::Off
            && self.adaptive_local_shake_arms.is_empty()
        {
            return Err(
                "adaptive_local_shake_arms must not be empty when adaptive_local_shake_mode is not off"
                    .to_string(),
            );
        }
        if !self.adaptive_local_shake_min_gain_parent_weight.is_finite()
            || self.adaptive_local_shake_min_gain_parent_weight < 0.0
        {
            return Err(
                "adaptive_local_shake_min_gain_parent_weight must be finite and non-negative"
                    .to_string(),
            );
        }
        if !self.adaptive_local_shake_shape_eps.is_finite()
            || self.adaptive_local_shake_shape_eps < 0.0
        {
            return Err(
                "adaptive_local_shake_shape_eps must be finite and non-negative".to_string(),
            );
        }
        for multiplier in &self.adaptive_local_shake_resolution_up_multipliers {
            if !multiplier.is_finite() || *multiplier <= 1.0 {
                return Err(
                    "adaptive_local_shake_resolution_up_multipliers must contain finite values greater than 1.0"
                        .to_string(),
                );
            }
        }
        for multiplier in &self.adaptive_local_shake_resolution_down_multipliers {
            if !multiplier.is_finite() || *multiplier <= 0.0 || *multiplier >= 1.0 {
                return Err(
                    "adaptive_local_shake_resolution_down_multipliers must contain finite values in (0, 1)"
                        .to_string(),
                );
            }
        }
        if !self
            .adaptive_local_shake_resolution_up_min_parent_ratio
            .is_finite()
            || self.adaptive_local_shake_resolution_up_min_parent_ratio < 0.0
        {
            return Err(
                "adaptive_local_shake_resolution_up_min_parent_ratio must be finite and non-negative"
                    .to_string(),
            );
        }
        if !self
            .adaptive_local_shake_resolution_down_max_parent_ratio
            .is_finite()
            || self.adaptive_local_shake_resolution_down_max_parent_ratio < 0.0
        {
            return Err(
                "adaptive_local_shake_resolution_down_max_parent_ratio must be finite and non-negative"
                    .to_string(),
            );
        }
        if !self.adaptive_local_shake_large_child_fraction.is_finite()
            || !(0.0..=1.0).contains(&self.adaptive_local_shake_large_child_fraction)
        {
            return Err("adaptive_local_shake_large_child_fraction must be in [0, 1]".to_string());
        }
        if !self.adaptive_local_shake_singleton_fraction.is_finite()
            || !(0.0..=1.0).contains(&self.adaptive_local_shake_singleton_fraction)
        {
            return Err("adaptive_local_shake_singleton_fraction must be in [0, 1]".to_string());
        }
        if !self
            .adaptive_local_shake_near_tie_margin_parent_weight
            .is_finite()
            || self.adaptive_local_shake_near_tie_margin_parent_weight < 0.0
        {
            return Err(
                "adaptive_local_shake_near_tie_margin_parent_weight must be finite and non-negative"
                    .to_string(),
            );
        }
        if !self.adaptive_local_shake_near_tie_randomness.is_finite()
            || !(0.0..=1.0).contains(&self.adaptive_local_shake_near_tie_randomness)
        {
            return Err("adaptive_local_shake_near_tie_randomness must be in [0, 1]".to_string());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default)]
struct AdaptiveProbeCommitState {
    total: usize,
    by_depth: HashMap<usize, usize>,
}

thread_local! {
    static ADAPTIVE_PROBE_VISITS: RefCell<HashMap<(usize, usize), usize>> =
        RefCell::new(HashMap::new());
    static ADAPTIVE_PROBE_COMMITS: RefCell<AdaptiveProbeCommitState> =
        RefCell::new(AdaptiveProbeCommitState::default());
}

fn reset_adaptive_probe_state() {
    ADAPTIVE_PROBE_VISITS.with(|visits| visits.borrow_mut().clear());
    ADAPTIVE_PROBE_COMMITS
        .with(|commits| *commits.borrow_mut() = AdaptiveProbeCommitState::default());
}

fn next_adaptive_probe_visit(depth: usize, parent_id: usize) -> usize {
    ADAPTIVE_PROBE_VISITS.with(|visits| {
        let mut visits = visits.borrow_mut();
        let entry = visits.entry((depth, parent_id)).or_insert(0);
        *entry += 1;
        *entry
    })
}

fn adaptive_probe_enabled(config: &DongdaemunRefinementConfig) -> bool {
    config.adaptive_probe_mode != AdaptiveProbeMode::Off && config.adaptive_probe_perturbations > 0
}

fn adaptive_probe_should_probe(
    config: &DongdaemunRefinementConfig,
    depth: usize,
    parent_id: usize,
    parent_visit_index: usize,
) -> bool {
    if !adaptive_probe_enabled(config) {
        return false;
    }
    config.adaptive_probe_targets.is_empty()
        || config.adaptive_probe_targets.iter().any(|target| {
            target.depth == depth
                && target.parent_id == parent_id
                && target.parent_visit_index == parent_visit_index
        })
}

fn adaptive_probe_source_is_valid(source: &str) -> bool {
    matches!(
        source,
        "same_gamma_probe" | "node_order_control" | "near_tie_refinement_probe"
    )
}

fn adaptive_probe_source_is_allowed(config: &DongdaemunRefinementConfig, source: &str) -> bool {
    config.adaptive_probe_commit_sources.is_empty()
        || config
            .adaptive_probe_commit_sources
            .iter()
            .any(|allowed| allowed == source)
}

fn adaptive_probe_source_label(source: &str) -> Option<&'static str> {
    match source {
        "same_gamma_probe" => Some("same_gamma_probe"),
        "node_order_control" => Some("node_order_control"),
        "near_tie_refinement_probe" => Some("near_tie_refinement_probe"),
        _ => None,
    }
}

fn adaptive_probe_commit_strategy_score(
    strategy: AdaptiveProbeCommitStrategy,
    gain_vs_baseline: f64,
    commit_gain_parent_weight: f64,
    largest_fraction: f64,
    singleton_weight_fraction: f64,
    standard_largest_fraction: f64,
) -> f64 {
    match strategy {
        AdaptiveProbeCommitStrategy::OnlineFirst => 0.0,
        AdaptiveProbeCommitStrategy::BestQf => gain_vs_baseline,
        AdaptiveProbeCommitStrategy::RiskAdjusted => {
            let largest_improvement = (standard_largest_fraction - largest_fraction).max(0.0);
            commit_gain_parent_weight + largest_improvement - singleton_weight_fraction
        }
    }
}

fn adaptive_probe_commit_strategy_trace(strategy: AdaptiveProbeCommitStrategy) -> &'static str {
    match strategy {
        AdaptiveProbeCommitStrategy::OnlineFirst => "online_first",
        AdaptiveProbeCommitStrategy::BestQf => "best_qf",
        AdaptiveProbeCommitStrategy::RiskAdjusted => "risk_adjusted",
    }
}

fn adaptive_near_tie_probe_mode_trace(mode: AdaptiveNearTieProbeMode) -> &'static str {
    match mode {
        AdaptiveNearTieProbeMode::Off => "off",
        AdaptiveNearTieProbeMode::TraceOnly => "trace_only",
        AdaptiveNearTieProbeMode::Candidate => "candidate",
        AdaptiveNearTieProbeMode::QfReplace => "qf_replace",
    }
}

fn adaptive_near_tie_probe_enabled(config: &DongdaemunRefinementConfig) -> bool {
    config.adaptive_near_tie_probe_mode != AdaptiveNearTieProbeMode::Off
        && config.adaptive_near_tie_margin_parent_weight > 0.0
}

fn adaptive_local_shake_enabled(config: &DongdaemunRefinementConfig) -> bool {
    config.adaptive_local_shake_mode != AdaptiveLocalShakeMode::Off
        && !config.adaptive_local_shake_arms.is_empty()
}

fn adaptive_local_shake_mode_trace(mode: AdaptiveLocalShakeMode) -> &'static str {
    match mode {
        AdaptiveLocalShakeMode::Off => "off",
        AdaptiveLocalShakeMode::TraceOnly => "trace_only",
        AdaptiveLocalShakeMode::QfReplace => "qf_replace",
        AdaptiveLocalShakeMode::PressureGuarded => "pressure_guarded",
    }
}

fn adaptive_local_shake_final_guard_mode_trace(
    mode: AdaptiveLocalShakeFinalGuardMode,
) -> &'static str {
    match mode {
        AdaptiveLocalShakeFinalGuardMode::None => "none",
        AdaptiveLocalShakeFinalGuardMode::RunnerAudit => "runner_audit",
    }
}

fn adaptive_local_shake_arm_trace(arm: AdaptiveLocalShakeArm) -> &'static str {
    match arm {
        AdaptiveLocalShakeArm::NearTieRefinement => "near_tie_refinement",
        AdaptiveLocalShakeArm::ResolutionUp => "resolution_up",
        AdaptiveLocalShakeArm::ResolutionDown => "resolution_down",
        AdaptiveLocalShakeArm::SeedLocalRefinement => "seed_local_refinement",
    }
}

fn local_merge_low_margin_threshold(
    config: Option<&DongdaemunRefinementConfig>,
    parent_weight: f64,
) -> f64 {
    let Some(config) = config else {
        return 0.0;
    };
    let mut margin_parent_weight = 0.0_f64;
    if adaptive_near_tie_probe_enabled(config) {
        margin_parent_weight =
            margin_parent_weight.max(config.adaptive_near_tie_margin_parent_weight);
    }
    if adaptive_local_shake_enabled(config)
        && config
            .adaptive_local_shake_arms
            .contains(&AdaptiveLocalShakeArm::NearTieRefinement)
    {
        margin_parent_weight =
            margin_parent_weight.max(config.adaptive_local_shake_near_tie_margin_parent_weight);
    }
    margin_parent_weight * parent_weight.max(0.0)
}

fn adaptive_probe_commit_counts(depth: usize) -> (usize, usize) {
    ADAPTIVE_PROBE_COMMITS.with(|commits| {
        let commits = commits.borrow();
        (commits.total, *commits.by_depth.get(&depth).unwrap_or(&0))
    })
}

fn record_adaptive_probe_commit(depth: usize) {
    ADAPTIVE_PROBE_COMMITS.with(|commits| {
        let mut commits = commits.borrow_mut();
        commits.total += 1;
        *commits.by_depth.entry(depth).or_insert(0) += 1;
    });
}

#[derive(Clone, Debug, Default)]
pub struct DongdaemunRefinementIterationAudit {
    pub depth: usize,
    pub selected_parents: usize,
    pub applied_parents: usize,
    pub same_gamma_candidates: usize,
    pub high_gamma_candidates: usize,
    pub same_gamma_applied: usize,
    pub high_gamma_applied: usize,
    pub quotient_candidates: usize,
    pub quotient_positive_candidates: usize,
    pub quotient_selected: usize,
    pub quotient_score_sum: f64,
    pub baseline_repair_candidates: usize,
    pub baseline_repair_improved_candidates: usize,
    pub baseline_repair_selected: usize,
    pub baseline_repair_merge_count: usize,
    pub baseline_repair_delta_sum: f64,
    pub candidate_quality_delta_sum: f64,
    pub candidate_positive_quality_delta: usize,
    pub candidate_selected_positive_quality_delta: usize,
    pub candidate_rejected_by_quality: usize,
    pub same_gamma_quality_delta_sum: f64,
    pub high_gamma_quality_delta_sum: f64,
    pub same_gamma_positive_quality_delta: usize,
    pub high_gamma_positive_quality_delta: usize,
    pub same_gamma_selected_positive_quality_delta: usize,
    pub high_gamma_selected_positive_quality_delta: usize,
    pub same_gamma_rejected_by_quality: usize,
    pub high_gamma_rejected_by_quality: usize,
    pub candidate_valid: usize,
    pub candidate_invalid: usize,
    pub candidate_rejected_by_policy: usize,
    pub same_gamma_valid: usize,
    pub high_gamma_valid: usize,
    pub same_gamma_invalid: usize,
    pub high_gamma_invalid: usize,
    pub same_gamma_rejected_by_policy: usize,
    pub high_gamma_rejected_by_policy: usize,
    pub candidate_qpos_spos: usize,
    pub candidate_qpos_sneg: usize,
    pub candidate_qneg_spos: usize,
    pub candidate_qneg_sneg: usize,
    pub same_gamma_qpos_spos: usize,
    pub same_gamma_qpos_sneg: usize,
    pub same_gamma_qneg_spos: usize,
    pub same_gamma_qneg_sneg: usize,
    pub high_gamma_qpos_spos: usize,
    pub high_gamma_qpos_sneg: usize,
    pub high_gamma_qneg_spos: usize,
    pub high_gamma_qneg_sneg: usize,
    pub candidate_true_positive: usize,
    pub candidate_false_positive: usize,
    pub candidate_false_negative: usize,
    pub candidate_true_negative: usize,
    pub adaptive_local_shake_triggers: usize,
    pub adaptive_local_shake_candidates: usize,
    pub adaptive_local_shake_commits: usize,
    pub adaptive_local_shake_qf_gain_sum: f64,
    pub standard_refined_clusters: usize,
    pub final_refined_clusters: usize,
}

#[derive(Clone, Debug, Default)]
pub struct DongdaemunRefinementAudit {
    pub enabled: bool,
    pub selected_parent_count_total: usize,
    pub applied_parent_count_total: usize,
    pub rejected_candidate_count_total: usize,
    pub added_refined_clusters_total: usize,
    pub same_gamma_candidates_total: usize,
    pub high_gamma_candidates_total: usize,
    pub same_gamma_applied_total: usize,
    pub high_gamma_applied_total: usize,
    pub quotient_candidates_total: usize,
    pub quotient_positive_candidates_total: usize,
    pub quotient_selected_total: usize,
    pub quotient_score_sum: f64,
    pub baseline_repair_candidates_total: usize,
    pub baseline_repair_improved_candidates_total: usize,
    pub baseline_repair_selected_total: usize,
    pub baseline_repair_merge_count_total: usize,
    pub baseline_repair_delta_sum: f64,
    pub candidate_quality_delta_sum: f64,
    pub candidate_positive_quality_delta_total: usize,
    pub candidate_selected_positive_quality_delta_total: usize,
    pub candidate_rejected_by_quality_total: usize,
    pub same_gamma_quality_delta_sum: f64,
    pub high_gamma_quality_delta_sum: f64,
    pub same_gamma_positive_quality_delta_total: usize,
    pub high_gamma_positive_quality_delta_total: usize,
    pub same_gamma_selected_positive_quality_delta_total: usize,
    pub high_gamma_selected_positive_quality_delta_total: usize,
    pub same_gamma_rejected_by_quality_total: usize,
    pub high_gamma_rejected_by_quality_total: usize,
    pub candidate_valid_total: usize,
    pub candidate_invalid_total: usize,
    pub candidate_rejected_by_policy_total: usize,
    pub same_gamma_valid_total: usize,
    pub high_gamma_valid_total: usize,
    pub same_gamma_invalid_total: usize,
    pub high_gamma_invalid_total: usize,
    pub same_gamma_rejected_by_policy_total: usize,
    pub high_gamma_rejected_by_policy_total: usize,
    pub candidate_qpos_spos_total: usize,
    pub candidate_qpos_sneg_total: usize,
    pub candidate_qneg_spos_total: usize,
    pub candidate_qneg_sneg_total: usize,
    pub same_gamma_qpos_spos_total: usize,
    pub same_gamma_qpos_sneg_total: usize,
    pub same_gamma_qneg_spos_total: usize,
    pub same_gamma_qneg_sneg_total: usize,
    pub high_gamma_qpos_spos_total: usize,
    pub high_gamma_qpos_sneg_total: usize,
    pub high_gamma_qneg_spos_total: usize,
    pub high_gamma_qneg_sneg_total: usize,
    pub candidate_true_positive_total: usize,
    pub candidate_false_positive_total: usize,
    pub candidate_false_negative_total: usize,
    pub candidate_true_negative_total: usize,
    pub adaptive_local_shake_triggers_total: usize,
    pub adaptive_local_shake_candidates_total: usize,
    pub adaptive_local_shake_commits_total: usize,
    pub adaptive_local_shake_qf_gain_sum: f64,
    pub final_quality_guard_enabled: bool,
    pub final_quality_guard_triggered: bool,
    pub final_quality_guard_standard_quality: f64,
    pub final_quality_guard_pre_guard_quality: f64,
    pub final_quality_delta_vs_guard_standard: f64,
    pub max_parent_weight_seen: f64,
    pub iterations: Vec<DongdaemunRefinementIterationAudit>,
}

/// Result of the opt-in Dongdaemun-refinement Leiden path.
#[derive(Clone, Debug)]
pub struct DongdaemunRefinementLeidenResult {
    pub clustering: Clustering,
    pub quality: f64,
    pub n_iterations_used: usize,
    pub audit: DongdaemunRefinementAudit,
}

#[derive(Clone, Copy, Debug, Default)]
struct IterationStats {
    improved: bool,
    moved_nodes: usize,
}

#[derive(Clone, Copy, Debug, Default)]
struct RefinementDongdaemunStats {
    selected_parents: usize,
    applied_parents: usize,
    rejected_candidates: usize,
    added_refined_clusters: usize,
    same_gamma_candidates: usize,
    high_gamma_candidates: usize,
    same_gamma_applied: usize,
    high_gamma_applied: usize,
    quotient_candidates: usize,
    quotient_positive_candidates: usize,
    quotient_selected: usize,
    quotient_score_sum: f64,
    baseline_repair_candidates: usize,
    baseline_repair_improved_candidates: usize,
    baseline_repair_selected: usize,
    baseline_repair_merge_count: usize,
    baseline_repair_delta_sum: f64,
    candidate_quality_delta_sum: f64,
    candidate_positive_quality_delta: usize,
    candidate_selected_positive_quality_delta: usize,
    candidate_rejected_by_quality: usize,
    same_gamma_quality_delta_sum: f64,
    high_gamma_quality_delta_sum: f64,
    same_gamma_positive_quality_delta: usize,
    high_gamma_positive_quality_delta: usize,
    same_gamma_selected_positive_quality_delta: usize,
    high_gamma_selected_positive_quality_delta: usize,
    same_gamma_rejected_by_quality: usize,
    high_gamma_rejected_by_quality: usize,
    candidate_valid: usize,
    candidate_invalid: usize,
    candidate_rejected_by_policy: usize,
    same_gamma_valid: usize,
    high_gamma_valid: usize,
    same_gamma_invalid: usize,
    high_gamma_invalid: usize,
    same_gamma_rejected_by_policy: usize,
    high_gamma_rejected_by_policy: usize,
    candidate_qpos_spos: usize,
    candidate_qpos_sneg: usize,
    candidate_qneg_spos: usize,
    candidate_qneg_sneg: usize,
    same_gamma_qpos_spos: usize,
    same_gamma_qpos_sneg: usize,
    same_gamma_qneg_spos: usize,
    same_gamma_qneg_sneg: usize,
    high_gamma_qpos_spos: usize,
    high_gamma_qpos_sneg: usize,
    high_gamma_qneg_spos: usize,
    high_gamma_qneg_sneg: usize,
    candidate_true_positive: usize,
    candidate_false_positive: usize,
    candidate_false_negative: usize,
    candidate_true_negative: usize,
    adaptive_local_shake_triggers: usize,
    adaptive_local_shake_candidates: usize,
    adaptive_local_shake_commits: usize,
    adaptive_local_shake_qf_gain_sum: f64,
    max_parent_weight_seen: f64,
    standard_refined_clusters: usize,
}

struct RefinementResult {
    clustering: Clustering,
    parent_clusters: Vec<u32>,
    cluster_starts: Vec<u32>,
    fixed: Option<Vec<bool>>,
    dongdaemun_stats: RefinementDongdaemunStats,
}

enum IterationStep {
    Done {
        stats: IterationStats,
    },
    NonRefined {
        local_stats: IterationStats,
        reduced: Graph,
        reduced_clustering: Clustering,
        parent_nodes: usize,
        reduced_nodes: usize,
        trace_detail: bool,
    },
    Refined {
        local_stats: IterationStats,
        reduced: Graph,
        reduced_clustering: Clustering,
        refinement_clustering: Clustering,
        parent_nodes: usize,
        reduced_nodes: usize,
        trace_detail: bool,
    },
}

fn empty_refinement(n_nodes: usize) -> Clustering {
    Clustering {
        n_nodes,
        n_clusters: 0,
        clusters: vec![0; n_nodes],
        fixed: None,
    }
}

fn counts_to_starts(mut counts: Vec<u32>) -> Vec<u32> {
    let mut running = 0u32;
    for count in counts.iter_mut() {
        let next = running + *count;
        *count = running;
        running = next;
    }
    counts.push(running);
    counts
}

fn trace_contraction_progress(
    graph: &Graph,
    reduced: &Graph,
    depth: usize,
    mode: &str,
    elapsed_ms: f64,
) {
    let node_delta = graph.n_nodes.saturating_sub(reduced.n_nodes);
    let edge_delta = graph.n_edges.saturating_sub(reduced.n_edges);
    let rel_node_delta = node_delta as f64 / graph.n_nodes.max(1) as f64;
    let rel_edge_delta = edge_delta as f64 / graph.n_edges.max(1) as f64;
    trace_graph_summary!(
        graph,
        "phase=leiden_contraction depth={} mode={} input_nodes={} input_directed_edges={} reduced_nodes={} reduced_directed_edges={} node_delta={} rel_node_delta={:.8e} edge_delta={} rel_edge_delta={:.8e} elapsed_ms={:.1}",
        depth,
        mode,
        graph.n_nodes,
        graph.n_edges,
        reduced.n_nodes,
        reduced.n_edges,
        node_delta,
        rel_node_delta,
        edge_delta,
        rel_edge_delta,
        elapsed_ms,
    );
}

fn recursion_guard_triggers(
    input_nodes: usize,
    input_edges: usize,
    reduced_nodes: usize,
    min_directed_edges: usize,
    min_rel_node_delta: f64,
) -> bool {
    if input_edges < min_directed_edges || min_rel_node_delta <= 0.0 || input_nodes == 0 {
        return false;
    }
    let node_delta = input_nodes.saturating_sub(reduced_nodes);
    let rel_node_delta = node_delta as f64 / input_nodes as f64;
    rel_node_delta <= min_rel_node_delta
}

fn should_stop_recursion_after_contraction(graph: &Graph, reduced: &Graph) -> bool {
    recursion_guard_triggers(
        graph.n_nodes,
        graph.n_edges,
        reduced.n_nodes,
        recursion_guard_min_directed_edges(),
        recursion_min_rel_node_delta(),
    )
}

fn trace_recursion_stop(graph: &Graph, reduced: &Graph, depth: usize, mode: &str) {
    let node_delta = graph.n_nodes.saturating_sub(reduced.n_nodes);
    let rel_node_delta = node_delta as f64 / graph.n_nodes.max(1) as f64;
    trace_graph_summary!(
        graph,
        "phase=leiden_recursion_stop depth={} mode={} reason=small_contraction input_nodes={} input_directed_edges={} reduced_nodes={} reduced_directed_edges={} node_delta={} rel_node_delta={:.8e} threshold={:.8e}",
        depth,
        mode,
        graph.n_nodes,
        graph.n_edges,
        reduced.n_nodes,
        reduced.n_edges,
        node_delta,
        rel_node_delta,
        recursion_min_rel_node_delta(),
    );
}

fn membership_hash(clustering: &Clustering) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for &cluster in &clustering.clusters {
        hash ^= cluster as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash ^= clustering.n_clusters as u64;
    hash.wrapping_mul(0x0000_0100_0000_01b3)
}

fn trajectory_trace_run_id_json() -> String {
    trace::json_string_option(trace::ddm_trajectory_trace_run_id())
}

fn emit_trajectory_phase_checkpoint(
    graph: &Graph,
    clustering: &Clustering,
    resolution: f64,
    depth: usize,
    iteration: usize,
    phase: &str,
) {
    if !trace::ddm_trajectory_trace_enabled() {
        return;
    }
    let quality = crate::quality::CPM::new(resolution).quality(graph, clustering);
    trace::emit_ddm_trajectory_trace(format_args!(
        "{{\"schema\":\"dongdaemun_trajectory_trace.v1\",\"event\":\"phase_checkpoint\",\"run_id\":{},\"depth\":{},\"iteration\":{},\"phase\":\"{}\",\"membership_hash\":\"{:016x}\",\"n_clusters\":{},\"quality\":{}}}",
        trajectory_trace_run_id_json(),
        depth,
        iteration,
        phase,
        membership_hash(clustering),
        clustering.n_clusters,
        trace::json_f64(quality),
    ));
}

/// Run the Leiden algorithm on a graph.
///
/// If `initial` is None, starts from singleton clustering.
/// Returns the final clustering and quality.
pub fn leiden(
    graph: &Graph,
    config: &LeidenConfig,
    initial: Option<Clustering>,
    rng: &mut impl Rng,
) -> LeidenResult {
    let mut ws = Workspace::new(graph.n_nodes);
    leiden_with_workspace(graph, config, initial, rng, &mut ws)
}

pub(crate) fn leiden_with_workspace(
    graph: &Graph,
    config: &LeidenConfig,
    initial: Option<Clustering>,
    rng: &mut impl Rng,
    ws: &mut Workspace,
) -> LeidenResult {
    let trace_run = should_trace_graph(graph);
    let run_start = Instant::now();
    if trace_run {
        trace::emit(format_args!(
            "phase=leiden_start nodes={} directed_edges={} resolution={:.8} n_iterations={} randomness={:.6} seed={}{}",
            graph.n_nodes,
            graph.n_edges,
            config.resolution,
            config.n_iterations,
            config.randomness,
            config.seed,
            trace::memory_fields(),
        ));
    }

    let mut clustering = initial.unwrap_or_else(|| Clustering::singleton(graph.n_nodes));
    let mut quality_trace = LeidenQualityTraceState::start(graph, &clustering, config);

    let mut n_used = 0;
    if config.n_iterations > 0 {
        for _ in 0..config.n_iterations {
            let iter_start = Instant::now();
            let iter_randomness = config.randomness_for_iteration(n_used);
            let stats = improve_one_iteration(
                graph,
                &mut clustering,
                config,
                iter_randomness,
                rng,
                ws,
                n_used + 1,
            );
            n_used += 1;
            let iteration_elapsed_ms = iter_start.elapsed().as_secs_f64() * 1000.0;
            trace_graph_summary!(
                graph,
                "phase=leiden_iteration iter={} randomness={:.6} improved={} moved_nodes={} clusters={} elapsed_ms={:.1}",
                n_used,
                iter_randomness,
                stats.improved,
                stats.moved_nodes,
                clustering.n_clusters,
                iteration_elapsed_ms,
            );
            if let Some(trace) = quality_trace.as_mut() {
                let checkpoint_quality =
                    crate::quality::CPM::new(config.resolution).quality(graph, &clustering);
                trace.emit_checkpoint(
                    graph,
                    &clustering,
                    "after_iteration",
                    n_used,
                    checkpoint_quality,
                    iteration_elapsed_ms,
                    stats.moved_nodes,
                );
            }
            if !stats.improved {
                break;
            }
        }
    } else {
        let min_rel_cluster_delta = convergence_min_rel_cluster_delta();
        let patience = convergence_patience();
        let mut previous_n_clusters = clustering.n_clusters;
        let mut stagnant_iterations = 0usize;
        loop {
            let iter_start = Instant::now();
            let iter_randomness = config.randomness_for_iteration(n_used);
            let stats = improve_one_iteration(
                graph,
                &mut clustering,
                config,
                iter_randomness,
                rng,
                ws,
                n_used + 1,
            );
            n_used += 1;
            let iteration_elapsed_ms = iter_start.elapsed().as_secs_f64() * 1000.0;
            let cluster_delta = previous_n_clusters.abs_diff(clustering.n_clusters);
            let rel_cluster_delta =
                cluster_delta as f64 / previous_n_clusters.max(clustering.n_clusters).max(1) as f64;
            trace_graph_summary!(
                graph,
                "phase=leiden_iteration iter={} randomness={:.6} improved={} moved_nodes={} clusters={} cluster_delta={} rel_cluster_delta={:.8e} stagnant_iterations={} elapsed_ms={:.1}",
                n_used,
                iter_randomness,
                stats.improved,
                stats.moved_nodes,
                clustering.n_clusters,
                cluster_delta,
                rel_cluster_delta,
                stagnant_iterations,
                iteration_elapsed_ms,
            );
            if let Some(trace) = quality_trace.as_mut() {
                let checkpoint_quality =
                    crate::quality::CPM::new(config.resolution).quality(graph, &clustering);
                trace.emit_checkpoint(
                    graph,
                    &clustering,
                    "after_iteration",
                    n_used,
                    checkpoint_quality,
                    iteration_elapsed_ms,
                    stats.moved_nodes,
                );
            }
            if !stats.improved {
                break;
            }
            if patience > 0 && rel_cluster_delta <= min_rel_cluster_delta {
                stagnant_iterations += 1;
                if stagnant_iterations >= patience {
                    trace_graph_summary!(
                        graph,
                        "phase=leiden_convergence_stop iter={} reason=cluster_delta rel_cluster_delta={:.8e} threshold={:.8e} patience={}",
                        n_used,
                        rel_cluster_delta,
                        min_rel_cluster_delta,
                        patience,
                    );
                    break;
                }
            } else {
                stagnant_iterations = 0;
            }
            previous_n_clusters = clustering.n_clusters;
        }
    }

    let quality = crate::quality::CPM::new(config.resolution).quality(graph, &clustering);
    if let Some(trace) = quality_trace.as_mut() {
        trace.emit_checkpoint(graph, &clustering, "final", n_used, quality, 0.0, 0);
    }

    if trace_run {
        trace::emit(format_args!(
            "phase=leiden_done nodes={} directed_edges={} clusters={} quality={:.6} iterations_used={} elapsed_ms={:.1}{}",
            graph.n_nodes,
            graph.n_edges,
            clustering.n_clusters,
            quality,
            n_used,
            run_start.elapsed().as_secs_f64() * 1000.0,
            trace::memory_fields(),
        ));
    }

    LeidenResult {
        clustering,
        quality,
        n_iterations_used: n_used,
    }
}

/// Run Leiden with opt-in Dongdaemun parent-internal extra refinement.
pub fn leiden_with_dongdaemun_refinement(
    graph: &Graph,
    config: &LeidenConfig,
    dongdaemun: &DongdaemunRefinementConfig,
    initial: Option<Clustering>,
    rng: &mut impl Rng,
) -> DongdaemunRefinementLeidenResult {
    let mut ws = Workspace::new(graph.n_nodes);
    leiden_with_dongdaemun_refinement_workspace(graph, config, dongdaemun, initial, rng, &mut ws)
}

pub(crate) fn leiden_with_dongdaemun_refinement_workspace(
    graph: &Graph,
    config: &LeidenConfig,
    dongdaemun: &DongdaemunRefinementConfig,
    initial: Option<Clustering>,
    rng: &mut impl Rng,
    ws: &mut Workspace,
) -> DongdaemunRefinementLeidenResult {
    dongdaemun
        .validate()
        .expect("invalid Dongdaemun refinement config");
    reset_adaptive_probe_state();
    let trace_run = should_trace_graph(graph);
    let run_start = Instant::now();
    if trace_run {
        trace::emit(format_args!(
            "phase=leiden_dongdaemun_refinement_start nodes={} directed_edges={} resolution={:.8} n_iterations={} randomness={:.6} seed={} target_max_weight={:.6} max_extra_parents_per_iteration={}{}",
            graph.n_nodes,
            graph.n_edges,
            config.resolution,
            config.n_iterations,
            config.randomness,
            config.seed,
            dongdaemun.target_max_weight,
            dongdaemun.max_extra_parents_per_iteration,
            trace::memory_fields(),
        ));
    }

    let guard_initial = dongdaemun.use_final_quality_guard.then(|| initial.clone());
    let mut clustering = initial.unwrap_or_else(|| Clustering::singleton(graph.n_nodes));
    let mut audit = DongdaemunRefinementAudit {
        enabled: true,
        final_quality_guard_enabled: dongdaemun.use_final_quality_guard,
        ..DongdaemunRefinementAudit::default()
    };
    let mut quality_trace =
        DongdaemunQualityTraceState::start(graph, &clustering, config, dongdaemun, &audit);

    let mut n_used = 0;
    if config.n_iterations > 0 {
        for _ in 0..config.n_iterations {
            let iter_start = Instant::now();
            let iter_randomness = config.randomness_for_iteration(n_used);
            let stats = improve_one_iteration_dongdaemun(
                graph,
                &mut clustering,
                config,
                dongdaemun,
                &mut audit,
                iter_randomness,
                rng,
                ws,
                n_used + 1,
            );
            n_used += 1;
            let iteration_elapsed_ms = iter_start.elapsed().as_secs_f64() * 1000.0;
            trace_graph_summary!(
                graph,
                "phase=leiden_dongdaemun_refinement_iteration iter={} randomness={:.6} improved={} moved_nodes={} clusters={} elapsed_ms={:.1}",
                n_used,
                iter_randomness,
                stats.improved,
                stats.moved_nodes,
                clustering.n_clusters,
                iteration_elapsed_ms,
            );
            if let Some(trace) = quality_trace.as_mut() {
                let checkpoint_quality =
                    crate::quality::CPM::new(config.resolution).quality(graph, &clustering);
                trace.emit_checkpoint(
                    graph,
                    &clustering,
                    dongdaemun,
                    &audit,
                    "after_iteration",
                    n_used,
                    checkpoint_quality,
                    iteration_elapsed_ms,
                    stats.moved_nodes,
                );
            }
            if !stats.improved {
                break;
            }
        }
    } else {
        let min_rel_cluster_delta = convergence_min_rel_cluster_delta();
        let patience = convergence_patience();
        let mut previous_n_clusters = clustering.n_clusters;
        let mut stagnant_iterations = 0usize;
        loop {
            let iter_start = Instant::now();
            let iter_randomness = config.randomness_for_iteration(n_used);
            let stats = improve_one_iteration_dongdaemun(
                graph,
                &mut clustering,
                config,
                dongdaemun,
                &mut audit,
                iter_randomness,
                rng,
                ws,
                n_used + 1,
            );
            n_used += 1;
            let iteration_elapsed_ms = iter_start.elapsed().as_secs_f64() * 1000.0;
            let cluster_delta = previous_n_clusters.abs_diff(clustering.n_clusters);
            let rel_cluster_delta =
                cluster_delta as f64 / previous_n_clusters.max(clustering.n_clusters).max(1) as f64;
            trace_graph_summary!(
                graph,
                "phase=leiden_dongdaemun_refinement_iteration iter={} randomness={:.6} improved={} moved_nodes={} clusters={} cluster_delta={} rel_cluster_delta={:.8e} stagnant_iterations={} elapsed_ms={:.1}",
                n_used,
                iter_randomness,
                stats.improved,
                stats.moved_nodes,
                clustering.n_clusters,
                cluster_delta,
                rel_cluster_delta,
                stagnant_iterations,
                iteration_elapsed_ms,
            );
            if let Some(trace) = quality_trace.as_mut() {
                let checkpoint_quality =
                    crate::quality::CPM::new(config.resolution).quality(graph, &clustering);
                trace.emit_checkpoint(
                    graph,
                    &clustering,
                    dongdaemun,
                    &audit,
                    "after_iteration",
                    n_used,
                    checkpoint_quality,
                    iteration_elapsed_ms,
                    stats.moved_nodes,
                );
            }
            if !stats.improved {
                break;
            }
            if patience > 0 && rel_cluster_delta <= min_rel_cluster_delta {
                stagnant_iterations += 1;
                if stagnant_iterations >= patience {
                    trace_graph_summary!(
                        graph,
                        "phase=leiden_dongdaemun_refinement_convergence_stop iter={} reason=cluster_delta rel_cluster_delta={:.8e} threshold={:.8e} patience={}",
                        n_used,
                        rel_cluster_delta,
                        min_rel_cluster_delta,
                        patience,
                    );
                    break;
                }
            } else {
                stagnant_iterations = 0;
            }
            previous_n_clusters = clustering.n_clusters;
        }
    }

    let mut quality = crate::quality::CPM::new(config.resolution).quality(graph, &clustering);
    audit.final_quality_guard_pre_guard_quality = quality;
    if dongdaemun.use_final_quality_guard {
        if let Some(trace) = quality_trace.as_mut() {
            trace.emit_checkpoint(
                graph,
                &clustering,
                dongdaemun,
                &audit,
                "pre_final_guard",
                n_used,
                quality,
                0.0,
                0,
            );
        }
    }

    if dongdaemun.use_final_quality_guard {
        let mut standard_rng = rand::rngs::StdRng::seed_from_u64(config.seed);
        let mut standard_ws = Workspace::new(graph.n_nodes);
        let standard = leiden_with_workspace(
            graph,
            config,
            guard_initial.flatten(),
            &mut standard_rng,
            &mut standard_ws,
        );
        audit.final_quality_guard_standard_quality = standard.quality;
        audit.final_quality_delta_vs_guard_standard = quality - standard.quality;
        if audit.final_quality_delta_vs_guard_standard < dongdaemun.min_final_quality_delta {
            audit.final_quality_guard_triggered = true;
            clustering = standard.clustering;
            quality = standard.quality;
            n_used = standard.n_iterations_used;
        }
    }
    if let Some(trace) = quality_trace.as_mut() {
        trace.emit_checkpoint(
            graph,
            &clustering,
            dongdaemun,
            &audit,
            "final",
            n_used,
            quality,
            0.0,
            0,
        );
    }

    if trace_run {
        trace::emit(format_args!(
            "phase=leiden_dongdaemun_refinement_done nodes={} directed_edges={} clusters={} quality={:.6} iterations_used={} selected_parents={} applied_parents={} added_refined_clusters={} final_quality_guard_triggered={} elapsed_ms={:.1}{}",
            graph.n_nodes,
            graph.n_edges,
            clustering.n_clusters,
            quality,
            n_used,
            audit.selected_parent_count_total,
            audit.applied_parent_count_total,
            audit.added_refined_clusters_total,
            audit.final_quality_guard_triggered,
            run_start.elapsed().as_secs_f64() * 1000.0,
            trace::memory_fields(),
        ));
    }

    DongdaemunRefinementLeidenResult {
        clustering,
        quality,
        n_iterations_used: n_used,
        audit,
    }
}

/// Run Leiden with multiple random starts, return best quality.
/// Uses rayon for parallel execution when n_starts > 1.
pub fn leiden_multi_start(
    graph: &Graph,
    config: &LeidenConfig,
    n_starts: usize,
    initial: Option<&Clustering>,
) -> LeidenResult {
    use rayon::prelude::*;

    if n_starts <= 1 {
        let mut rng = rand::rngs::StdRng::seed_from_u64(config.seed);
        let init = initial.cloned();
        return leiden(graph, config, init, &mut rng);
    }

    let results: Vec<LeidenResult> = (0..n_starts)
        .into_par_iter()
        .map(|start| {
            let mut rng = rand::rngs::StdRng::seed_from_u64(config.seed + start as u64);
            let init = initial.cloned();
            leiden(graph, config, init, &mut rng)
        })
        .collect();

    results
        .into_iter()
        .max_by(|a, b| a.quality.total_cmp(&b.quality))
        .unwrap()
}

/// One iteration of Leiden: move → refine → aggregate → recurse.
fn improve_one_iteration(
    graph: &Graph,
    clustering: &mut Clustering,
    config: &LeidenConfig,
    randomness: f64,
    rng: &mut impl Rng,
    ws: &mut Workspace,
    iteration: usize,
) -> IterationStats {
    let mut no_audit: Option<&mut DongdaemunRefinementAudit> = None;
    let step = prepare_iteration_step(
        graph,
        clustering,
        config,
        None,
        no_audit.as_deref_mut(),
        randomness,
        rng,
        ws,
        0,
        iteration,
    );
    finish_iteration_step(
        step, clustering, config, None, no_audit, randomness, rng, ws, 0, iteration,
    )
}

fn improve_one_iteration_dongdaemun(
    graph: &Graph,
    clustering: &mut Clustering,
    config: &LeidenConfig,
    dongdaemun: &DongdaemunRefinementConfig,
    audit: &mut DongdaemunRefinementAudit,
    randomness: f64,
    rng: &mut impl Rng,
    ws: &mut Workspace,
    iteration: usize,
) -> IterationStats {
    let mut audit = Some(audit);
    let step = prepare_iteration_step(
        graph,
        clustering,
        config,
        Some(dongdaemun),
        audit.as_deref_mut(),
        randomness,
        rng,
        ws,
        0,
        iteration,
    );
    finish_iteration_step(
        step,
        clustering,
        config,
        Some(dongdaemun),
        audit,
        randomness,
        rng,
        ws,
        0,
        iteration,
    )
}

/// Recursive Leiden iteration for reduced graphs owned by the current frame.
///
/// The borrowed root graph must stay alive for quality computation and API
/// ownership, but reduced graphs do not need to survive while their own reduced
/// child is processed. Passing them by value lets us explicitly drop the parent
/// reduced graph before descending further. This matches the liveness behavior
/// Java can get from GC/JIT and prevents near-identity contractions from
/// accumulating one full CSR per recursion level.
fn improve_one_iteration_owned(
    graph: Graph,
    clustering: &mut Clustering,
    config: &LeidenConfig,
    dongdaemun: Option<&DongdaemunRefinementConfig>,
    mut audit: Option<&mut DongdaemunRefinementAudit>,
    randomness: f64,
    rng: &mut impl Rng,
    ws: &mut Workspace,
    depth: usize,
    iteration: usize,
) -> IterationStats {
    let step = prepare_iteration_step(
        &graph,
        clustering,
        config,
        dongdaemun,
        audit.as_deref_mut(),
        randomness,
        rng,
        ws,
        depth,
        iteration,
    );
    drop(graph);
    finish_iteration_step(
        step, clustering, config, dongdaemun, audit, randomness, rng, ws, depth, iteration,
    )
}

fn prepare_iteration_step(
    graph: &Graph,
    clustering: &mut Clustering,
    config: &LeidenConfig,
    dongdaemun: Option<&DongdaemunRefinementConfig>,
    mut audit: Option<&mut DongdaemunRefinementAudit>,
    randomness: f64,
    rng: &mut impl Rng,
    ws: &mut Workspace,
    depth: usize,
    iteration: usize,
) -> IterationStep {
    let trace_detail = should_trace_graph_detail(graph);
    let parent_nodes = graph.n_nodes;

    // Phase 1: Local moving
    let t_move = Instant::now();
    let local_move = fast_local_move::improve_clustering_with_trace(
        graph,
        clustering,
        config.resolution,
        rng,
        ws,
        Some(fast_local_move::LocalMoveTraceContext { depth, iteration }),
    );
    let local_stats = IterationStats {
        improved: local_move.improved,
        moved_nodes: local_move.moved_nodes,
    };
    trace_graph!(
        graph,
        "phase=local_move depth={} nodes={} directed_edges={} clusters={} updated={} moved_nodes={} elapsed_ms={:.1}",
        depth,
        graph.n_nodes,
        graph.n_edges,
        clustering.n_clusters,
        local_stats.improved,
        local_stats.moved_nodes,
        t_move.elapsed().as_secs_f64() * 1000.0,
    );
    emit_trajectory_phase_checkpoint(
        graph,
        clustering,
        config.resolution,
        depth,
        iteration,
        "after_local_move",
    );

    // If every node is its own cluster, nothing to do
    if clustering.n_clusters >= graph.n_nodes {
        return IterationStep::Done { stats: local_stats };
    }

    // Phase 2: Refinement
    let use_streaming_refinement = graph.n_edges >= streaming_refinement_min_directed_edges();
    let t_nodes = Instant::now();
    let refinement = if use_streaming_refinement {
        let parent_weights = if dongdaemun.is_some() {
            clustering.fill_cluster_groups_and_weights(&graph.node_weights, ws);
            Some(ws.cw[..clustering.n_clusters].to_vec())
        } else {
            clustering.fill_cluster_groups(ws);
            None
        };
        trace_graph!(
            graph,
            "phase=nodes_per_cluster depth={} mode=flat clusters={} elapsed_ms={:.1}",
            depth,
            clustering.n_clusters,
            t_nodes.elapsed().as_secs_f64() * 1000.0,
        );
        let t_refine = Instant::now();
        let refinement = {
            let starts = &ws.npc_starts[..clustering.n_clusters + 1];
            let nodes = &ws.npc_nodes[..graph.n_nodes];
            let local_index = &mut ws.local_index[..graph.n_nodes];
            refine_streaming_flat(
                graph,
                clustering,
                clustering.n_clusters,
                starts,
                nodes,
                local_index,
                config,
                dongdaemun,
                parent_weights.as_deref(),
                randomness,
                depth,
                iteration,
                rng,
            )
        };
        trace_graph!(
            graph,
            "phase=refinement depth={} mode=streaming refined_clusters={} elapsed_ms={:.1}",
            depth,
            refinement.clustering.n_clusters,
            t_refine.elapsed().as_secs_f64() * 1000.0,
        );
        refinement
    } else {
        let nodes_per_cluster = clustering.nodes_per_cluster();
        let parent_weights = dongdaemun.map(|_| clustering.cluster_weights(&graph.node_weights));
        trace_graph!(
            graph,
            "phase=nodes_per_cluster depth={} mode=vec clusters={} elapsed_ms={:.1}",
            depth,
            nodes_per_cluster.len(),
            t_nodes.elapsed().as_secs_f64() * 1000.0,
        );
        let t_refine = Instant::now();
        let refinement = refine_eager(
            graph,
            clustering,
            &nodes_per_cluster,
            config,
            dongdaemun,
            parent_weights.as_deref(),
            randomness,
            depth,
            iteration,
            rng,
        );
        trace_graph!(
            graph,
            "phase=refinement depth={} mode=eager refined_clusters={} elapsed_ms={:.1}",
            depth,
            refinement.clustering.n_clusters,
            t_refine.elapsed().as_secs_f64() * 1000.0,
        );
        refinement
    };
    record_dongdaemun_refinement_stats(
        depth,
        refinement.dongdaemun_stats,
        refinement.clustering.n_clusters,
        audit.as_deref_mut(),
    );
    emit_trajectory_phase_checkpoint(
        graph,
        &refinement.clustering,
        config.resolution,
        depth,
        iteration,
        "after_refinement",
    );

    if refinement.clustering.n_clusters >= graph.n_nodes {
        // Refinement produced singletons — aggregate on non-refined clustering
        let t_contract = Instant::now();
        let reduced = create_reduced_network(graph, clustering, true, ws);
        let contract_elapsed_ms = t_contract.elapsed().as_secs_f64() * 1000.0;
        trace_contraction_progress(graph, &reduced, depth, "non_refined", contract_elapsed_ms);
        trace_graph!(
            graph,
            "phase=contract_non_refined depth={} reduced_nodes={} reduced_directed_edges={} elapsed_ms={:.1}",
            depth,
            reduced.n_nodes,
            reduced.n_edges,
            contract_elapsed_ms,
        );
        if should_stop_recursion_after_contraction(graph, &reduced) {
            trace_recursion_stop(graph, &reduced, depth, "non_refined");
            return IterationStep::Done { stats: local_stats };
        }
        let mut reduced_clustering = Clustering::singleton(reduced.n_nodes);

        // Propagate fixed status to reduced graph
        if clustering.fixed.is_some() {
            let mut rf = vec![false; clustering.n_clusters];
            for i in 0..graph.n_nodes {
                if clustering.is_fixed(i) {
                    rf[clustering.clusters[i] as usize] = true;
                }
            }
            reduced_clustering.fixed = Some(rf);
        }
        emit_trajectory_phase_checkpoint(
            &reduced,
            &reduced_clustering,
            config.resolution,
            depth,
            iteration,
            "after_aggregation_non_refined",
        );

        let reduced_nodes = reduced.n_nodes;
        return IterationStep::NonRefined {
            local_stats,
            reduced,
            reduced_clustering,
            parent_nodes,
            reduced_nodes,
            trace_detail,
        };
    }

    // Phase 3: Aggregate based on refined clustering
    let t_contract = Instant::now();
    let reduced = create_reduced_network_from_starts(
        graph,
        &refinement.clustering,
        &refinement.cluster_starts,
        true,
        ws,
    );
    let contract_elapsed_ms = t_contract.elapsed().as_secs_f64() * 1000.0;
    trace_contraction_progress(graph, &reduced, depth, "refined", contract_elapsed_ms);
    trace_graph!(
        graph,
        "phase=contract_refined depth={} reduced_nodes={} reduced_directed_edges={} elapsed_ms={:.1}",
        depth,
        reduced.n_nodes,
        reduced.n_edges,
        contract_elapsed_ms,
    );
    if should_stop_recursion_after_contraction(graph, &reduced) {
        trace_recursion_stop(graph, &reduced, depth, "refined");
        return IterationStep::Done { stats: local_stats };
    }

    // Initial clustering for aggregate network: map non-refined clusters
    // to the move-phase cluster assignments (before refinement).
    // Each refined sub-cluster inherits the move-phase cluster ID of its parent.
    let reduced_n_clusters = refinement
        .parent_clusters
        .iter()
        .copied()
        .max()
        .map_or(0, |max_cid| max_cid as usize + 1);

    let reduced_clustering = Clustering {
        n_nodes: refinement.clustering.n_clusters,
        n_clusters: reduced_n_clusters,
        clusters: refinement.parent_clusters,
        fixed: refinement.fixed,
    };
    emit_trajectory_phase_checkpoint(
        &reduced,
        &reduced_clustering,
        config.resolution,
        depth,
        iteration,
        "after_aggregation_refined",
    );

    let reduced_nodes = reduced.n_nodes;
    IterationStep::Refined {
        local_stats,
        reduced,
        reduced_clustering,
        refinement_clustering: refinement.clustering,
        parent_nodes,
        reduced_nodes,
        trace_detail,
    }
}

fn select_extra_refinement_parents(
    parent_weights: &[f64],
    config: &DongdaemunRefinementConfig,
    boundary_pressure: Option<&[f64]>,
) -> (Vec<bool>, usize, f64) {
    let max_parent_weight_seen = parent_weights.iter().copied().fold(0.0_f64, f64::max);
    let threshold = config.target_max_weight * config.soft_min_ratio;
    let mut candidates = parent_weights
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, weight)| *weight >= threshold)
        .map(|(cid, weight)| {
            let pressure = if config.target_max_weight > 0.0 {
                weight / config.target_max_weight
            } else {
                weight
            };
            let score = match config.parent_selection_policy {
                ParentSelectionPolicy::Weight => weight,
                ParentSelectionPolicy::PressureBoundary => {
                    pressure
                        + boundary_pressure
                            .and_then(|scores| scores.get(cid))
                            .copied()
                            .unwrap_or(0.0)
                }
            };
            (cid, weight, score)
        })
        .collect::<Vec<_>>();
    candidates.sort_by(
        |(left_id, left_weight, left_score), (right_id, right_weight, right_score)| {
            right_score
                .total_cmp(left_score)
                .then_with(|| right_weight.total_cmp(left_weight))
                .then_with(|| left_id.cmp(right_id))
        },
    );

    let mut selected = vec![false; parent_weights.len()];
    let mut selected_count = 0usize;
    for (cid, _, _) in candidates
        .into_iter()
        .take(config.max_extra_parents_per_iteration)
    {
        selected[cid] = true;
        selected_count += 1;
    }
    (selected, selected_count, max_parent_weight_seen)
}

fn parent_boundary_pressure_for_nodes(
    graph: &Graph,
    clustering: &Clustering,
    parent_id: usize,
    parent_weight: f64,
    nodes: &[usize],
) -> f64 {
    if parent_weight <= 0.0 {
        return 0.0;
    }
    let mut external_weight = 0.0;
    for &node in nodes {
        let start = graph.first_neighbor_index[node] as usize;
        let end = graph.first_neighbor_index[node + 1] as usize;
        for edge in start..end {
            let neighbor = graph.neighbors[edge] as usize;
            if clustering.clusters[neighbor] as usize != parent_id {
                external_weight += graph.edge_weights[edge];
            }
        }
    }
    external_weight / parent_weight.max(1.0)
}

fn parent_boundary_pressure_eager(
    graph: &Graph,
    clustering: &Clustering,
    nodes_per_cluster: &[Vec<usize>],
    parent_weights: &[f64],
    config: &DongdaemunRefinementConfig,
) -> Option<Vec<f64>> {
    if config.parent_selection_policy != ParentSelectionPolicy::PressureBoundary {
        return None;
    }
    let threshold = config.target_max_weight * config.soft_min_ratio;
    let mut scores = vec![0.0; parent_weights.len()];
    for (cid, nodes) in nodes_per_cluster.iter().enumerate() {
        if parent_weights.get(cid).copied().unwrap_or(0.0) < threshold {
            continue;
        }
        scores[cid] =
            parent_boundary_pressure_for_nodes(graph, clustering, cid, parent_weights[cid], nodes);
    }
    Some(scores)
}

fn parent_boundary_pressure_streaming(
    graph: &Graph,
    clustering: &Clustering,
    npc_starts: &[u32],
    npc_nodes: &[u32],
    parent_weights: &[f64],
    config: &DongdaemunRefinementConfig,
) -> Option<Vec<f64>> {
    if config.parent_selection_policy != ParentSelectionPolicy::PressureBoundary {
        return None;
    }
    let threshold = config.target_max_weight * config.soft_min_ratio;
    let mut scores = vec![0.0; parent_weights.len()];
    for cid in 0..parent_weights.len() {
        if parent_weights[cid] < threshold {
            continue;
        }
        let start = npc_starts[cid] as usize;
        let end = npc_starts[cid + 1] as usize;
        let mut external_weight = 0.0;
        for &node_u32 in &npc_nodes[start..end] {
            let node = node_u32 as usize;
            let edge_start = graph.first_neighbor_index[node] as usize;
            let edge_end = graph.first_neighbor_index[node + 1] as usize;
            for edge in edge_start..edge_end {
                let neighbor = graph.neighbors[edge] as usize;
                if clustering.clusters[neighbor] as usize != cid {
                    external_weight += graph.edge_weights[edge];
                }
            }
        }
        scores[cid] = external_weight / parent_weights[cid].max(1.0);
    }
    Some(scores)
}

fn parent_partition_summary<F>(
    local_len: usize,
    n_clusters: usize,
    assignments: &[u32],
    parent_weight: f64,
    mut node_weight_at: F,
) -> (f64, f64)
where
    F: FnMut(usize) -> f64,
{
    if n_clusters == 0 || parent_weight <= 0.0 {
        return (1.0, 0.0);
    }
    let mut weights = vec![0.0; n_clusters];
    let mut counts = vec![0u32; n_clusters];
    for (local, &cluster) in assignments.iter().take(local_len).enumerate() {
        let cluster = cluster as usize;
        if cluster < n_clusters {
            weights[cluster] += node_weight_at(local);
            counts[cluster] += 1;
        }
    }
    let largest_fraction = weights.iter().copied().fold(0.0_f64, f64::max) / parent_weight;
    let singleton_weight: f64 = weights
        .iter()
        .zip(counts.iter())
        .filter_map(|(weight, count)| (*count == 1).then_some(*weight))
        .sum();
    (largest_fraction, singleton_weight / parent_weight)
}

fn parent_candidate_is_valid(
    n_clusters: usize,
    largest_fraction: f64,
    singleton_weight_fraction: f64,
    standard_largest_fraction: f64,
    config: &DongdaemunRefinementConfig,
) -> bool {
    let improves_structure = if matches!(
        config.candidate_quality_policy,
        CandidateQualityPolicy::Selective
            | CandidateQualityPolicy::PressureAware
            | CandidateQualityPolicy::AdaptivePlateau
    ) {
        largest_fraction < standard_largest_fraction
    } else {
        largest_fraction + config.min_largest_child_fraction_improvement
            <= standard_largest_fraction
    };
    n_clusters >= 2
        && n_clusters <= config.max_extra_children_per_parent
        && singleton_weight_fraction <= config.max_singleton_weight_fraction
        && improves_structure
}

fn parent_partition_quality_subgraph(
    graph: &Graph,
    assignments: &[u32],
    n_clusters: usize,
    resolution: f64,
) -> f64 {
    if n_clusters == 0 {
        return 0.0;
    }
    let local_len = assignments.len().min(graph.n_nodes);
    let mut internal_weight = vec![0.0f64; n_clusters];
    let mut cluster_weight = vec![0.0f64; n_clusters];
    let mut self_loop_weight = vec![0.0f64; n_clusters];

    for local in 0..local_len {
        let cid = assignments[local] as usize;
        if cid >= n_clusters {
            continue;
        }
        cluster_weight[cid] += graph.node_weights[local];
        self_loop_weight[cid] += graph.self_loop_weights[local];
        for (neighbor, weight) in graph.neighbors_of(local) {
            let neighbor = neighbor as usize;
            if neighbor < local_len && assignments[neighbor] as usize == cid {
                internal_weight[cid] += weight;
            }
        }
    }

    partition_quality_from_accumulators(
        &internal_weight,
        &cluster_weight,
        &self_loop_weight,
        resolution,
    )
}

fn parent_partition_quality_induced_u32(
    graph: &Graph,
    nodes: &[u32],
    local_index: &mut [u32],
    assignments: &[u32],
    n_clusters: usize,
    resolution: f64,
) -> f64 {
    assert_eq!(local_index.len(), graph.n_nodes);
    if n_clusters == 0 {
        return 0.0;
    }
    let local_len = assignments.len().min(nodes.len());
    for (local, &node) in nodes.iter().take(local_len).enumerate() {
        local_index[node as usize] = local as u32;
    }

    let mut internal_weight = vec![0.0f64; n_clusters];
    let mut cluster_weight = vec![0.0f64; n_clusters];
    let mut self_loop_weight = vec![0.0f64; n_clusters];

    for (local, &node_u32) in nodes.iter().take(local_len).enumerate() {
        let cid = assignments[local] as usize;
        if cid >= n_clusters {
            continue;
        }
        let node = node_u32 as usize;
        cluster_weight[cid] += graph.node_weights[node];
        self_loop_weight[cid] += graph.self_loop_weights[node];
        for (neighbor, weight) in graph.neighbors_of(node) {
            let neighbor_local = local_index[neighbor as usize];
            if neighbor_local != u32::MAX && assignments[neighbor_local as usize] as usize == cid {
                internal_weight[cid] += weight;
            }
        }
    }

    for &node in nodes.iter().take(local_len) {
        local_index[node as usize] = u32::MAX;
    }

    partition_quality_from_accumulators(
        &internal_weight,
        &cluster_weight,
        &self_loop_weight,
        resolution,
    )
}

fn partition_quality_from_accumulators(
    internal_weight: &[f64],
    cluster_weight: &[f64],
    self_loop_weight: &[f64],
    resolution: f64,
) -> f64 {
    let mut quality = 0.0;
    for cid in 0..cluster_weight.len() {
        let edge_mass = internal_weight[cid] / 2.0 + self_loop_weight[cid];
        let node_mass = cluster_weight[cid];
        quality += edge_mass - resolution * node_mass * (node_mass - 1.0) / 2.0;
    }
    quality
}

fn candidate_is_better(
    best_largest_fraction: f64,
    best_singleton_weight_fraction: f64,
    best_n_clusters: usize,
    largest_fraction: f64,
    singleton_weight_fraction: f64,
    n_clusters: usize,
) -> bool {
    largest_fraction
        .total_cmp(&best_largest_fraction)
        .then_with(|| singleton_weight_fraction.total_cmp(&best_singleton_weight_fraction))
        .then_with(|| n_clusters.cmp(&best_n_clusters))
        .is_lt()
}

fn candidate_is_better_with_quotient(
    choice: &ParentRefinementChoice,
    largest_fraction: f64,
    singleton_weight_fraction: f64,
    n_clusters: usize,
    quotient_score: f64,
    use_quotient_diagnostic: bool,
) -> bool {
    if use_quotient_diagnostic && (quotient_score > 0.0 || choice.quotient_score > 0.0) {
        match quotient_score.total_cmp(&choice.quotient_score) {
            std::cmp::Ordering::Greater => return true,
            std::cmp::Ordering::Less => return false,
            std::cmp::Ordering::Equal => {}
        }
    }
    candidate_is_better(
        choice.largest_fraction,
        choice.singleton_weight_fraction,
        choice.n_clusters,
        largest_fraction,
        singleton_weight_fraction,
        n_clusters,
    )
}

fn finite_or_zero(value: f64) -> f64 {
    if value.is_finite() {
        value
    } else {
        0.0
    }
}

fn adaptive_diagnostic_score(
    parent_weight: f64,
    target_max_weight: f64,
    standard_largest_fraction: f64,
    largest_fraction: f64,
    singleton_weight_fraction: f64,
    quotient_score: f64,
) -> f64 {
    let standard_max_child_weight_ratio = candidate_max_child_weight_ratio(
        parent_weight,
        standard_largest_fraction,
        target_max_weight,
    );
    let candidate_max_child_weight_ratio =
        candidate_max_child_weight_ratio(parent_weight, largest_fraction, target_max_weight);
    let pressure_reduction = oversize_pressure_excess(standard_max_child_weight_ratio)
        - oversize_pressure_excess(candidate_max_child_weight_ratio);
    finite_or_zero(standard_largest_fraction - largest_fraction)
        + finite_or_zero(pressure_reduction)
        + finite_or_zero(quotient_score)
        - finite_or_zero(singleton_weight_fraction)
}

fn adaptive_plateau_compared(
    choice: &ParentRefinementChoice,
    candidate_delta_q: f64,
    config: &DongdaemunRefinementConfig,
) -> bool {
    config.candidate_quality_policy == CandidateQualityPolicy::AdaptivePlateau
        && choice.source.is_some()
        && (candidate_delta_q - choice.candidate_delta_q).abs()
            <= config.adaptive_plateau_quality_band
}

fn candidate_is_better_adaptive_plateau(
    choice: &ParentRefinementChoice,
    largest_fraction: f64,
    singleton_weight_fraction: f64,
    n_clusters: usize,
    quotient_score: f64,
    candidate_delta_q: f64,
    standard_largest_fraction: f64,
    parent_weight: f64,
    config: &DongdaemunRefinementConfig,
) -> bool {
    if choice.source.is_none() {
        return true;
    }

    let quality_delta = candidate_delta_q - choice.candidate_delta_q;
    if quality_delta.abs() > config.adaptive_plateau_quality_band {
        return quality_delta > 0.0;
    }

    let diagnostic_score = adaptive_diagnostic_score(
        parent_weight,
        config.target_max_weight,
        standard_largest_fraction,
        largest_fraction,
        singleton_weight_fraction,
        quotient_score,
    );
    let best_diagnostic_score = adaptive_diagnostic_score(
        parent_weight,
        config.target_max_weight,
        standard_largest_fraction,
        choice.largest_fraction,
        choice.singleton_weight_fraction,
        choice.quotient_score,
    );
    match diagnostic_score.total_cmp(&best_diagnostic_score) {
        std::cmp::Ordering::Greater => return true,
        std::cmp::Ordering::Less => return false,
        std::cmp::Ordering::Equal => {}
    }
    match candidate_delta_q.total_cmp(&choice.candidate_delta_q) {
        std::cmp::Ordering::Greater => return true,
        std::cmp::Ordering::Less => return false,
        std::cmp::Ordering::Equal => {}
    }
    match largest_fraction.total_cmp(&choice.largest_fraction) {
        std::cmp::Ordering::Less => return true,
        std::cmp::Ordering::Greater => return false,
        std::cmp::Ordering::Equal => {}
    }
    match singleton_weight_fraction.total_cmp(&choice.singleton_weight_fraction) {
        std::cmp::Ordering::Less => return true,
        std::cmp::Ordering::Greater => return false,
        std::cmp::Ordering::Equal => {}
    }
    n_clusters < choice.n_clusters
}

fn candidate_is_better_by_policy(
    choice: &ParentRefinementChoice,
    largest_fraction: f64,
    singleton_weight_fraction: f64,
    n_clusters: usize,
    quotient_score: f64,
    candidate_delta_q: f64,
    standard_largest_fraction: f64,
    parent_weight: f64,
    config: &DongdaemunRefinementConfig,
) -> bool {
    if config.candidate_quality_policy == CandidateQualityPolicy::Selective {
        match candidate_delta_q.total_cmp(&choice.candidate_delta_q) {
            std::cmp::Ordering::Greater => return true,
            std::cmp::Ordering::Less => return false,
            std::cmp::Ordering::Equal => {}
        }
        return candidate_is_better(
            choice.largest_fraction,
            choice.singleton_weight_fraction,
            choice.n_clusters,
            largest_fraction,
            singleton_weight_fraction,
            n_clusters,
        );
    }
    if config.candidate_quality_policy == CandidateQualityPolicy::PressureAware {
        return candidate_is_better_pressure_aware(
            choice,
            largest_fraction,
            singleton_weight_fraction,
            n_clusters,
            candidate_delta_q,
        );
    }
    if config.candidate_quality_policy == CandidateQualityPolicy::AdaptivePlateau {
        return candidate_is_better_adaptive_plateau(
            choice,
            largest_fraction,
            singleton_weight_fraction,
            n_clusters,
            quotient_score,
            candidate_delta_q,
            standard_largest_fraction,
            parent_weight,
            config,
        );
    }
    if config.candidate_quality_policy == CandidateQualityPolicy::QualityFirst {
        match candidate_delta_q.total_cmp(&choice.candidate_delta_q) {
            std::cmp::Ordering::Greater => return true,
            std::cmp::Ordering::Less => return false,
            std::cmp::Ordering::Equal => {}
        }
    }
    candidate_is_better_with_quotient(
        choice,
        largest_fraction,
        singleton_weight_fraction,
        n_clusters,
        quotient_score,
        config.use_quotient_diagnostic,
    )
}

fn candidate_is_better_pressure_aware(
    choice: &ParentRefinementChoice,
    largest_fraction: f64,
    singleton_weight_fraction: f64,
    n_clusters: usize,
    candidate_delta_q: f64,
) -> bool {
    match largest_fraction.total_cmp(&choice.largest_fraction) {
        std::cmp::Ordering::Less => return true,
        std::cmp::Ordering::Greater => return false,
        std::cmp::Ordering::Equal => {}
    }
    match singleton_weight_fraction.total_cmp(&choice.singleton_weight_fraction) {
        std::cmp::Ordering::Less => return true,
        std::cmp::Ordering::Greater => return false,
        std::cmp::Ordering::Equal => {}
    }
    match candidate_delta_q.total_cmp(&choice.candidate_delta_q) {
        std::cmp::Ordering::Greater => return true,
        std::cmp::Ordering::Less => return false,
        std::cmp::Ordering::Equal => {}
    }
    n_clusters < choice.n_clusters
}

fn candidate_quality_passes(candidate_delta_q: f64, config: &DongdaemunRefinementConfig) -> bool {
    match config.candidate_quality_policy {
        CandidateQualityPolicy::Structural => true,
        CandidateQualityPolicy::QualityGuardedStructural
        | CandidateQualityPolicy::QualityFloor
        | CandidateQualityPolicy::QualityFirst
        | CandidateQualityPolicy::PressureAware
        | CandidateQualityPolicy::AdaptivePlateau => {
            candidate_delta_q >= config.min_candidate_delta_q
        }
        CandidateQualityPolicy::Selective => candidate_delta_q > config.min_candidate_delta_q,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RefinementCandidateSource {
    SameGammaSeed,
    HighGamma,
    NearTieRefinementProbe,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RefinementCandidateQuadrant {
    QPosSPos,
    QPosSNeg,
    QNegSPos,
    QNegSNeg,
}

impl RefinementCandidateSource {
    fn as_trace_str(self) -> &'static str {
        match self {
            RefinementCandidateSource::SameGammaSeed => "same_gamma_seed",
            RefinementCandidateSource::HighGamma => "high_gamma",
            RefinementCandidateSource::NearTieRefinementProbe => "near_tie_refinement_probe",
        }
    }
}

impl RefinementCandidateQuadrant {
    fn as_trace_str(self) -> &'static str {
        match self {
            RefinementCandidateQuadrant::QPosSPos => "qpos_spos",
            RefinementCandidateQuadrant::QPosSNeg => "qpos_sneg",
            RefinementCandidateQuadrant::QNegSPos => "qneg_spos",
            RefinementCandidateQuadrant::QNegSNeg => "qneg_sneg",
        }
    }
}

fn trace_json_f64(value: f64) -> String {
    if value.is_finite() {
        value.to_string()
    } else {
        "null".to_string()
    }
}

fn trace_json_string(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            ch if ch.is_control() => out.push_str(&format!("\\u{:04x}", ch as u32)),
            ch => out.push(ch),
        }
    }
    out.push('"');
    out
}

fn trace_json_string_option(value: Option<String>) -> String {
    value
        .as_deref()
        .map(trace_json_string)
        .unwrap_or_else(|| "null".to_string())
}

fn candidate_trace_run_id_json() -> String {
    trace_json_string_option(trace::ddm_candidate_trace_run_id())
}

fn leiden_quality_trace_run_id_json() -> String {
    trace_json_string_option(trace::leiden_quality_trace_run_id())
}

fn quality_trace_run_id_json() -> String {
    trace_json_string_option(trace::ddm_quality_trace_run_id())
}

struct LeidenQualityTraceState {
    checkpoint_index: usize,
    start_quality: f64,
    run_start: Instant,
    target_max_weight: f64,
}

impl LeidenQualityTraceState {
    fn start(graph: &Graph, clustering: &Clustering, config: &LeidenConfig) -> Option<Self> {
        if !trace::leiden_quality_trace_enabled() {
            return None;
        }
        let start_quality = crate::quality::CPM::new(config.resolution).quality(graph, clustering);
        let mut state = Self {
            checkpoint_index: 0,
            start_quality,
            run_start: Instant::now(),
            target_max_weight: trace::leiden_quality_trace_target_max_weight().unwrap_or(0.0),
        };
        state.emit_checkpoint(graph, clustering, "start", 0, start_quality, 0.0, 0);
        Some(state)
    }

    fn emit_checkpoint(
        &mut self,
        graph: &Graph,
        clustering: &Clustering,
        phase: &str,
        iteration: usize,
        quality: f64,
        iteration_elapsed_ms: f64,
        moved_nodes: usize,
    ) {
        let (max_doc_weight, max_doc_weight_ratio, n_above_max_doc_weight) =
            quality_trace_pressure_metrics(graph, clustering, self.target_max_weight);
        trace::emit_leiden_quality_trace(format_args!(
            "{{\"schema\":\"leiden_quality_checkpoint.v1\",\"event\":\"quality_checkpoint\",\"run_id\":{},\"checkpoint_index\":{},\"phase\":\"{}\",\"iteration\":{},\"quality\":{},\"quality_delta_vs_start\":{},\"elapsed_ms_since_run_start\":{},\"iteration_elapsed_ms\":{},\"n_clusters\":{},\"max_doc_weight\":{},\"max_doc_weight_ratio\":{},\"n_above_max_doc_weight\":{},\"moved_nodes\":{}}}",
            leiden_quality_trace_run_id_json(),
            self.checkpoint_index,
            phase,
            iteration,
            trace_json_f64(quality),
            trace_json_f64(quality - self.start_quality),
            trace_json_f64(self.run_start.elapsed().as_secs_f64() * 1000.0),
            trace_json_f64(iteration_elapsed_ms),
            clustering.n_clusters,
            trace_json_f64(max_doc_weight),
            trace_json_f64(max_doc_weight_ratio),
            n_above_max_doc_weight,
            moved_nodes,
        ));
        self.checkpoint_index += 1;
    }
}

struct DongdaemunQualityTraceState {
    checkpoint_index: usize,
    start_quality: f64,
    run_start: Instant,
}

impl DongdaemunQualityTraceState {
    fn start(
        graph: &Graph,
        clustering: &Clustering,
        config: &LeidenConfig,
        dongdaemun: &DongdaemunRefinementConfig,
        audit: &DongdaemunRefinementAudit,
    ) -> Option<Self> {
        if !trace::ddm_quality_trace_enabled() {
            return None;
        }
        let start_quality = crate::quality::CPM::new(config.resolution).quality(graph, clustering);
        let mut state = Self {
            checkpoint_index: 0,
            start_quality,
            run_start: Instant::now(),
        };
        state.emit_checkpoint(
            graph,
            clustering,
            dongdaemun,
            audit,
            "start",
            0,
            start_quality,
            0.0,
            0,
        );
        Some(state)
    }

    fn emit_checkpoint(
        &mut self,
        graph: &Graph,
        clustering: &Clustering,
        dongdaemun: &DongdaemunRefinementConfig,
        audit: &DongdaemunRefinementAudit,
        phase: &str,
        iteration: usize,
        quality: f64,
        iteration_elapsed_ms: f64,
        moved_nodes: usize,
    ) {
        let (max_doc_weight, max_doc_weight_ratio, n_above_max_doc_weight) =
            quality_trace_pressure_metrics(graph, clustering, dongdaemun.target_max_weight);
        trace::emit_ddm_quality_trace(format_args!(
            "{{\"schema\":\"dongdaemun_refinement_quality_checkpoint.v1\",\"event\":\"quality_checkpoint\",\"run_id\":{},\"checkpoint_index\":{},\"phase\":\"{}\",\"iteration\":{},\"quality\":{},\"quality_delta_vs_start\":{},\"elapsed_ms_since_run_start\":{},\"iteration_elapsed_ms\":{},\"n_clusters\":{},\"max_doc_weight\":{},\"max_doc_weight_ratio\":{},\"n_above_max_doc_weight\":{},\"moved_nodes\":{},\"selected_parent_count_total\":{},\"applied_parent_count_total\":{}}}",
            quality_trace_run_id_json(),
            self.checkpoint_index,
            phase,
            iteration,
            trace_json_f64(quality),
            trace_json_f64(quality - self.start_quality),
            trace_json_f64(self.run_start.elapsed().as_secs_f64() * 1000.0),
            trace_json_f64(iteration_elapsed_ms),
            clustering.n_clusters,
            trace_json_f64(max_doc_weight),
            trace_json_f64(max_doc_weight_ratio),
            n_above_max_doc_weight,
            moved_nodes,
            audit.selected_parent_count_total,
            audit.applied_parent_count_total,
        ));
        self.checkpoint_index += 1;
    }
}

fn quality_trace_pressure_metrics(
    graph: &Graph,
    clustering: &Clustering,
    target_max_weight: f64,
) -> (f64, f64, usize) {
    let weights = clustering.cluster_weights(&graph.node_weights);
    let max_doc_weight = weights.iter().copied().fold(0.0_f64, f64::max);
    let max_doc_weight_ratio = if target_max_weight > 0.0 {
        max_doc_weight / target_max_weight
    } else {
        f64::INFINITY
    };
    let n_above_max_doc_weight = if target_max_weight > 0.0 {
        weights
            .iter()
            .filter(|&&weight| weight > target_max_weight)
            .count()
    } else {
        0
    };
    (max_doc_weight, max_doc_weight_ratio, n_above_max_doc_weight)
}

#[derive(Clone, Copy, Debug)]
struct CandidateTraceContext {
    depth: usize,
    parent_id: usize,
    parent_visit_index: usize,
    parent_size: usize,
    parent_weight: f64,
    standard_n_clusters: usize,
    source_index: usize,
    gamma_multiplier: f64,
    repaired: bool,
}

#[derive(Clone, Copy, Debug)]
struct LocalShakeArmSpec {
    arm: AdaptiveLocalShakeArm,
    arm_index: usize,
    priority_rank: usize,
    multiplier: f64,
    seed_index: usize,
}

#[derive(Debug)]
struct LocalShakeCandidate {
    spec: LocalShakeArmSpec,
    candidate_index: usize,
    assignments: Vec<u32>,
    counts: Vec<u32>,
    n_clusters: usize,
    largest_fraction: f64,
    singleton_weight_fraction: f64,
    candidate_delta_q: f64,
    current_candidate_delta_q: f64,
    gain_vs_current: f64,
    current_max_child_weight_ratio: f64,
    candidate_max_child_weight_ratio: f64,
    pressure_guard_score: f64,
    assignment_hash: u64,
    changed_node_count: usize,
    distinct: bool,
    valid: bool,
    quality_passes: bool,
    near_tie_summary: Option<local_merge::LocalMergeMarginSummary>,
}

impl Default for CandidateTraceContext {
    fn default() -> Self {
        Self {
            depth: 0,
            parent_id: 0,
            parent_visit_index: 0,
            parent_size: 0,
            parent_weight: 0.0,
            standard_n_clusters: 0,
            source_index: 0,
            gamma_multiplier: 1.0,
            repaired: false,
        }
    }
}

fn local_shake_arm_is_configured(
    config: &DongdaemunRefinementConfig,
    arm: AdaptiveLocalShakeArm,
) -> bool {
    config.adaptive_local_shake_arms.contains(&arm)
}

fn local_shake_ordered_arms(config: &DongdaemunRefinementConfig) -> Vec<AdaptiveLocalShakeArm> {
    let mut arms = Vec::new();
    for &arm in config
        .adaptive_local_shake_arm_priority
        .iter()
        .chain(config.adaptive_local_shake_arms.iter())
    {
        if local_shake_arm_is_configured(config, arm) && !arms.contains(&arm) {
            arms.push(arm);
        }
    }
    arms
}

fn local_shake_parent_weight_ratio(config: &DongdaemunRefinementConfig, parent_weight: f64) -> f64 {
    if config.target_max_weight > 0.0 {
        parent_weight / config.target_max_weight
    } else {
        f64::INFINITY
    }
}

fn select_local_shake_arms(
    config: &DongdaemunRefinementConfig,
    parent_weight: f64,
    standard_largest_fraction: f64,
    standard_singleton_weight_fraction: f64,
    margin_summary: Option<&local_merge::LocalMergeMarginSummary>,
) -> Vec<LocalShakeArmSpec> {
    if !adaptive_local_shake_enabled(config) {
        return Vec::new();
    }
    let parent_weight_ratio = local_shake_parent_weight_ratio(config, parent_weight);
    let low_margin_count = margin_summary
        .map(|summary| summary.low_margin_decision_count)
        .unwrap_or(0);
    let ordered_arms = local_shake_ordered_arms(config);
    let mut specs = Vec::new();
    let mut selected_arm_count = 0usize;
    for (priority_rank, arm) in ordered_arms.into_iter().enumerate() {
        if config.adaptive_local_shake_max_arms_per_parent > 0
            && selected_arm_count >= config.adaptive_local_shake_max_arms_per_parent
        {
            break;
        }
        let candidate_count_before = specs.len();
        match arm {
            AdaptiveLocalShakeArm::NearTieRefinement => {
                if low_margin_count >= config.adaptive_local_shake_near_tie_min_count {
                    specs.push(LocalShakeArmSpec {
                        arm,
                        arm_index: selected_arm_count,
                        priority_rank,
                        multiplier: 1.0,
                        seed_index: 0,
                    });
                }
            }
            AdaptiveLocalShakeArm::ResolutionUp => {
                if parent_weight_ratio >= config.adaptive_local_shake_resolution_up_min_parent_ratio
                    && (standard_largest_fraction
                        >= config.adaptive_local_shake_large_child_fraction
                        || parent_weight_ratio > 1.0)
                {
                    for &multiplier in &config.adaptive_local_shake_resolution_up_multipliers {
                        specs.push(LocalShakeArmSpec {
                            arm,
                            arm_index: selected_arm_count,
                            priority_rank,
                            multiplier,
                            seed_index: 0,
                        });
                    }
                }
            }
            AdaptiveLocalShakeArm::ResolutionDown => {
                if parent_weight_ratio
                    <= config.adaptive_local_shake_resolution_down_max_parent_ratio
                    || standard_singleton_weight_fraction
                        >= config.adaptive_local_shake_singleton_fraction
                {
                    for &multiplier in &config.adaptive_local_shake_resolution_down_multipliers {
                        specs.push(LocalShakeArmSpec {
                            arm,
                            arm_index: selected_arm_count,
                            priority_rank,
                            multiplier,
                            seed_index: 0,
                        });
                    }
                }
            }
            AdaptiveLocalShakeArm::SeedLocalRefinement => {
                if low_margin_count >= config.adaptive_local_shake_seed_margin_count {
                    for seed_index in 0..config.adaptive_local_shake_seed_perturbations {
                        specs.push(LocalShakeArmSpec {
                            arm,
                            arm_index: selected_arm_count,
                            priority_rank,
                            multiplier: 1.0,
                            seed_index,
                        });
                    }
                }
            }
        }
        if specs.len() > candidate_count_before {
            selected_arm_count += 1;
        }
        if config.adaptive_local_shake_max_candidates_per_parent > 0
            && specs.len() >= config.adaptive_local_shake_max_candidates_per_parent
        {
            specs.truncate(config.adaptive_local_shake_max_candidates_per_parent);
            break;
        }
    }
    specs
}

fn local_shake_selected_arms_json(specs: &[LocalShakeArmSpec]) -> String {
    let mut arms = Vec::new();
    for spec in specs {
        let arm = adaptive_local_shake_arm_trace(spec.arm);
        if !arms.contains(&arm) {
            arms.push(arm);
        }
    }
    let values = arms
        .into_iter()
        .map(trace_json_string)
        .collect::<Vec<_>>()
        .join(",");
    format!("[{values}]")
}

fn local_shake_assignment_hash(assignments: &[u32], n_clusters: usize) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for value in (n_clusters as u64)
        .to_le_bytes()
        .into_iter()
        .chain(assignments.iter().flat_map(|value| value.to_le_bytes()))
    {
        hash ^= value as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn local_shake_changed_node_count(left: &[u32], right: &[u32]) -> usize {
    left.iter()
        .zip(right.iter())
        .filter(|(a, b)| a != b)
        .count()
        + left.len().abs_diff(right.len())
}

fn local_shake_candidate_distinct(
    choice: &ParentRefinementChoice,
    assignments: &[u32],
    n_clusters: usize,
    largest_fraction: f64,
    singleton_weight_fraction: f64,
    eps: f64,
) -> bool {
    n_clusters != choice.n_clusters
        || assignments.len() != choice.assignments.len()
        || assignments
            .iter()
            .zip(choice.assignments.iter())
            .any(|(left, right)| left != right)
        || (largest_fraction - choice.largest_fraction).abs() > eps
        || (singleton_weight_fraction - choice.singleton_weight_fraction).abs() > eps
}

#[allow(clippy::too_many_arguments)]
fn build_local_shake_candidate(
    spec: LocalShakeArmSpec,
    candidate_index: usize,
    choice: &ParentRefinementChoice,
    assignments: Vec<u32>,
    counts: Vec<u32>,
    n_clusters: usize,
    largest_fraction: f64,
    singleton_weight_fraction: f64,
    candidate_delta_q: f64,
    standard_largest_fraction: f64,
    parent_weight: f64,
    config: &DongdaemunRefinementConfig,
    near_tie_summary: Option<local_merge::LocalMergeMarginSummary>,
) -> LocalShakeCandidate {
    let current_candidate_delta_q = choice.candidate_delta_q;
    let gain_vs_current = candidate_delta_q - current_candidate_delta_q;
    let current_max_child_weight_ratio = candidate_max_child_weight_ratio(
        parent_weight,
        choice.largest_fraction,
        config.target_max_weight,
    );
    let candidate_max_child_weight_ratio =
        candidate_max_child_weight_ratio(parent_weight, largest_fraction, config.target_max_weight);
    let pressure_guard_score = current_max_child_weight_ratio - candidate_max_child_weight_ratio;
    let assignment_hash = local_shake_assignment_hash(&assignments, n_clusters);
    let changed_node_count = local_shake_changed_node_count(&assignments, &choice.assignments);
    let distinct = local_shake_candidate_distinct(
        choice,
        &assignments,
        n_clusters,
        largest_fraction,
        singleton_weight_fraction,
        config.adaptive_local_shake_shape_eps,
    );
    let valid = parent_candidate_is_valid(
        n_clusters,
        largest_fraction,
        singleton_weight_fraction,
        standard_largest_fraction,
        config,
    );
    let quality_passes = candidate_quality_passes(candidate_delta_q, config);
    LocalShakeCandidate {
        spec,
        candidate_index,
        assignments,
        counts,
        n_clusters,
        largest_fraction,
        singleton_weight_fraction,
        candidate_delta_q,
        current_candidate_delta_q,
        gain_vs_current,
        current_max_child_weight_ratio,
        candidate_max_child_weight_ratio,
        pressure_guard_score,
        assignment_hash,
        changed_node_count,
        distinct,
        valid,
        quality_passes,
        near_tie_summary,
    }
}

fn local_shake_commit_block_reason(
    candidate: &LocalShakeCandidate,
    config: &DongdaemunRefinementConfig,
    parent_weight: f64,
) -> &'static str {
    match config.adaptive_local_shake_mode {
        AdaptiveLocalShakeMode::Off => "off",
        AdaptiveLocalShakeMode::TraceOnly => "trace_only",
        AdaptiveLocalShakeMode::QfReplace | AdaptiveLocalShakeMode::PressureGuarded => {
            if !candidate.distinct {
                "not_distinct"
            } else if candidate.spec.arm == AdaptiveLocalShakeArm::NearTieRefinement
                && candidate
                    .near_tie_summary
                    .as_ref()
                    .map(|summary| summary.changed_decision_count == 0)
                    .unwrap_or(true)
            {
                "near_tie_unchanged_decision"
            } else if !candidate.valid {
                "invalid_candidate"
            } else if !candidate.quality_passes {
                "quality_rejected"
            } else if candidate.gain_vs_current
                <= (config.adaptive_local_shake_min_gain_parent_weight * parent_weight.max(1.0))
                    .max(config.adaptive_local_shake_shape_eps)
            {
                "not_qf_gain"
            } else if config.adaptive_local_shake_mode == AdaptiveLocalShakeMode::PressureGuarded
                && candidate.candidate_max_child_weight_ratio
                    > candidate.current_max_child_weight_ratio
                        + config.adaptive_local_shake_shape_eps
            {
                "pressure_guard"
            } else {
                "eligible"
            }
        }
    }
}

fn local_shake_commit_eligible(
    candidate: &LocalShakeCandidate,
    config: &DongdaemunRefinementConfig,
    parent_weight: f64,
) -> bool {
    if !matches!(
        config.adaptive_local_shake_mode,
        AdaptiveLocalShakeMode::QfReplace | AdaptiveLocalShakeMode::PressureGuarded
    ) {
        return false;
    }
    if !candidate.distinct || !candidate.valid || !candidate.quality_passes {
        return false;
    }
    if candidate.spec.arm == AdaptiveLocalShakeArm::NearTieRefinement
        && candidate
            .near_tie_summary
            .as_ref()
            .map(|summary| summary.changed_decision_count == 0)
            .unwrap_or(true)
    {
        return false;
    }
    let tolerance = (config.adaptive_local_shake_min_gain_parent_weight * parent_weight.max(1.0))
        .max(config.adaptive_local_shake_shape_eps);
    if candidate.gain_vs_current <= tolerance {
        return false;
    }
    if config.adaptive_local_shake_mode == AdaptiveLocalShakeMode::PressureGuarded
        && candidate.candidate_max_child_weight_ratio
            > candidate.current_max_child_weight_ratio + config.adaptive_local_shake_shape_eps
    {
        return false;
    }
    true
}

fn local_shake_candidate_is_better(
    current: &LocalShakeCandidate,
    candidate: &LocalShakeCandidate,
    config: &DongdaemunRefinementConfig,
    parent_weight: f64,
) -> bool {
    let current_eligible = local_shake_commit_eligible(current, config, parent_weight);
    let candidate_eligible = local_shake_commit_eligible(candidate, config, parent_weight);
    match candidate_eligible.cmp(&current_eligible) {
        std::cmp::Ordering::Greater => return true,
        std::cmp::Ordering::Less => return false,
        std::cmp::Ordering::Equal => {}
    }
    match candidate
        .gain_vs_current
        .total_cmp(&current.gain_vs_current)
    {
        std::cmp::Ordering::Greater => return true,
        std::cmp::Ordering::Less => return false,
        std::cmp::Ordering::Equal => {}
    }
    match candidate
        .pressure_guard_score
        .total_cmp(&current.pressure_guard_score)
    {
        std::cmp::Ordering::Greater => return true,
        std::cmp::Ordering::Less => return false,
        std::cmp::Ordering::Equal => {}
    }
    match current
        .spec
        .priority_rank
        .cmp(&candidate.spec.priority_rank)
    {
        std::cmp::Ordering::Greater => return true,
        std::cmp::Ordering::Less => return false,
        std::cmp::Ordering::Equal => {}
    }
    match current.spec.arm_index.cmp(&candidate.spec.arm_index) {
        std::cmp::Ordering::Greater => return true,
        std::cmp::Ordering::Less => return false,
        std::cmp::Ordering::Equal => {}
    }
    match current
        .spec
        .multiplier
        .total_cmp(&candidate.spec.multiplier)
    {
        std::cmp::Ordering::Greater => return true,
        std::cmp::Ordering::Less => return false,
        std::cmp::Ordering::Equal => {}
    }
    match current.spec.seed_index.cmp(&candidate.spec.seed_index) {
        std::cmp::Ordering::Greater => return true,
        std::cmp::Ordering::Less => return false,
        std::cmp::Ordering::Equal => {}
    }
    candidate.assignment_hash < current.assignment_hash
}

fn emit_local_shake_trigger_trace(
    trace_context: CandidateTraceContext,
    iteration: usize,
    standard_largest_child_fraction: f64,
    standard_singleton_weight_fraction: f64,
    selected_specs: &[LocalShakeArmSpec],
    summary: Option<&local_merge::LocalMergeMarginSummary>,
    config: &DongdaemunRefinementConfig,
) {
    if !trace::ddm_candidate_trace_enabled() {
        return;
    }
    let parent_weight_ratio = local_shake_parent_weight_ratio(config, trace_context.parent_weight);
    let low_margin_count = summary
        .map(|summary| summary.low_margin_decision_count)
        .unwrap_or(0);
    let min_margin = summary
        .map(|summary| summary.min_margin)
        .unwrap_or(f64::NAN);
    let p10_margin = summary
        .map(|summary| summary.p10_margin)
        .unwrap_or(f64::NAN);
    let selected_arms = local_shake_selected_arms_json(selected_specs);
    trace::emit_ddm_candidate_trace(format_args!(
        "{{\"event\":\"adaptive_local_shake_trigger\",\"run_id\":{},\"depth\":{},\"iteration\":{},\"parent_id\":{},\"parent_visit_index\":{},\"mode\":\"{}\",\"final_guard_mode\":\"{}\",\"parent_size\":{},\"parent_weight\":{},\"parent_weight_ratio\":{},\"standard_n_children\":{},\"standard_largest_child_fraction\":{},\"standard_singleton_weight_fraction\":{},\"local_merge_low_margin_count\":{},\"local_merge_min_margin\":{},\"local_merge_p10_margin\":{},\"optional_fields_present\":{},\"selected_arms\":{},\"trigger_reason\":\"{}\"}}",
        candidate_trace_run_id_json(),
        trace_context.depth,
        iteration,
        trace_context.parent_id,
        trace_context.parent_visit_index,
        adaptive_local_shake_mode_trace(config.adaptive_local_shake_mode),
        adaptive_local_shake_final_guard_mode_trace(config.adaptive_local_shake_final_guard_mode),
        trace_context.parent_size,
        trace_json_f64(trace_context.parent_weight),
        trace_json_f64(parent_weight_ratio),
        trace_context.standard_n_clusters,
        trace_json_f64(standard_largest_child_fraction),
        trace_json_f64(standard_singleton_weight_fraction),
        low_margin_count,
        trace_json_f64(min_margin),
        trace_json_f64(p10_margin),
        summary.is_some(),
        selected_arms,
        if selected_specs.is_empty() { "no_arms_selected" } else { "arms_selected" },
    ));
}

fn emit_local_shake_candidate_trace(
    trace_context: CandidateTraceContext,
    candidate: &LocalShakeCandidate,
    commit_eligible: bool,
    commit_block_reason: &str,
    config: &DongdaemunRefinementConfig,
) {
    if !trace::ddm_candidate_trace_enabled() {
        return;
    }
    let (low_margin_count, changed_count, min_margin, p10_margin, p50_margin) = candidate
        .near_tie_summary
        .as_ref()
        .map(|summary| {
            (
                summary.low_margin_decision_count,
                summary.changed_decision_count,
                summary.min_margin,
                summary.p10_margin,
                summary.p50_margin,
            )
        })
        .unwrap_or((0, 0, f64::NAN, f64::NAN, f64::NAN));
    trace::emit_ddm_candidate_trace(format_args!(
        "{{\"event\":\"adaptive_local_shake_candidate\",\"run_id\":{},\"depth\":{},\"parent_id\":{},\"parent_visit_index\":{},\"candidate_index\":{},\"arm\":\"{}\",\"arm_index\":{},\"arm_priority\":{},\"mode\":\"{}\",\"multiplier\":{},\"seed_index\":{},\"parent_size\":{},\"parent_weight\":{},\"standard_n_clusters\":{},\"candidate_n_clusters\":{},\"largest_child_fraction\":{},\"singleton_weight_fraction\":{},\"candidate_delta_q\":{},\"current_candidate_delta_q\":{},\"gain_vs_current\":{},\"current_max_child_weight_ratio\":{},\"candidate_max_child_weight_ratio\":{},\"pressure_guard_score\":{},\"assignment_hash\":{},\"changed_node_count\":{},\"distinct\":{},\"valid\":{},\"quality_passes\":{},\"commit_eligible\":{},\"commit_block_reason\":\"{}\",\"near_tie_low_margin_decision_count\":{},\"near_tie_changed_decision_count\":{},\"near_tie_min_margin\":{},\"near_tie_p10_margin\":{},\"near_tie_p50_margin\":{}}}",
        candidate_trace_run_id_json(),
        trace_context.depth,
        trace_context.parent_id,
        trace_context.parent_visit_index,
        candidate.candidate_index,
        adaptive_local_shake_arm_trace(candidate.spec.arm),
        candidate.spec.arm_index,
        candidate.spec.priority_rank,
        adaptive_local_shake_mode_trace(config.adaptive_local_shake_mode),
        trace_json_f64(candidate.spec.multiplier),
        candidate.spec.seed_index,
        trace_context.parent_size,
        trace_json_f64(trace_context.parent_weight),
        trace_context.standard_n_clusters,
        candidate.n_clusters,
        trace_json_f64(candidate.largest_fraction),
        trace_json_f64(candidate.singleton_weight_fraction),
        trace_json_f64(candidate.candidate_delta_q),
        trace_json_f64(candidate.current_candidate_delta_q),
        trace_json_f64(candidate.gain_vs_current),
        trace_json_f64(candidate.current_max_child_weight_ratio),
        trace_json_f64(candidate.candidate_max_child_weight_ratio),
        trace_json_f64(candidate.pressure_guard_score),
        candidate.assignment_hash,
        candidate.changed_node_count,
        candidate.distinct,
        candidate.valid,
        candidate.quality_passes,
        commit_eligible,
        commit_block_reason,
        low_margin_count,
        changed_count,
        trace_json_f64(min_margin),
        trace_json_f64(p10_margin),
        trace_json_f64(p50_margin),
    ));
}

fn emit_local_shake_decision_trace(
    trace_context: CandidateTraceContext,
    selected: Option<&LocalShakeCandidate>,
    committed: bool,
    commit_block_reason: &str,
    config: &DongdaemunRefinementConfig,
) {
    if !trace::ddm_candidate_trace_enabled() {
        return;
    }
    let selected_candidate_index = selected
        .map(|candidate| candidate.candidate_index.to_string())
        .unwrap_or_else(|| "null".to_string());
    let selected_arm = selected
        .map(|candidate| trace_json_string(adaptive_local_shake_arm_trace(candidate.spec.arm)))
        .unwrap_or_else(|| "null".to_string());
    let selected_gain = selected
        .map(|candidate| trace_json_f64(candidate.gain_vs_current))
        .unwrap_or_else(|| "null".to_string());
    trace::emit_ddm_candidate_trace(format_args!(
        "{{\"event\":\"adaptive_local_shake_decision\",\"run_id\":{},\"depth\":{},\"parent_id\":{},\"parent_visit_index\":{},\"mode\":\"{}\",\"selected_candidate_index\":{},\"selected_arm\":{},\"committed\":{},\"commit_block_reason\":\"{}\",\"selected_gain_vs_current\":{}}}",
        candidate_trace_run_id_json(),
        trace_context.depth,
        trace_context.parent_id,
        trace_context.parent_visit_index,
        adaptive_local_shake_mode_trace(config.adaptive_local_shake_mode),
        selected_candidate_index,
        selected_arm,
        committed,
        commit_block_reason,
        selected_gain,
    ));
}

fn reduce_local_shake_candidate(
    best: &mut Option<LocalShakeCandidate>,
    candidate: LocalShakeCandidate,
    config: &DongdaemunRefinementConfig,
    parent_weight: f64,
) {
    if best
        .as_ref()
        .map(|current| local_shake_candidate_is_better(current, &candidate, config, parent_weight))
        .unwrap_or(true)
    {
        *best = Some(candidate);
    }
}

fn commit_local_shake_candidate(
    choice: &mut ParentRefinementChoice,
    candidate: LocalShakeCandidate,
    stats: &mut RefinementDongdaemunStats,
) {
    stats.adaptive_local_shake_commits += 1;
    stats.adaptive_local_shake_qf_gain_sum += candidate.gain_vs_current;
    choice.assignments = candidate.assignments;
    choice.counts = candidate.counts;
    choice.n_clusters = candidate.n_clusters;
    choice.largest_fraction = candidate.largest_fraction;
    choice.singleton_weight_fraction = candidate.singleton_weight_fraction;
    choice.quotient_score = 0.0;
    choice.candidate_delta_q = candidate.candidate_delta_q;
    choice.source = Some(RefinementCandidateSource::NearTieRefinementProbe);
    choice.trace_candidate_id = None;
    choice.baseline_repair_merge_count = 0;
    choice.baseline_repair_delta_sum = 0.0;
    choice.adaptive_probe_committed = false;
    choice.adaptive_probe_score = 0.0;
    choice.adaptive_probe_source_label = Some("adaptive_local_shake");
}

fn local_shake_seed_salt(arm: AdaptiveLocalShakeArm) -> u64 {
    match arm {
        AdaptiveLocalShakeArm::NearTieRefinement => 0x4c53_4e45_4152_5449,
        AdaptiveLocalShakeArm::ResolutionUp => 0x4c53_5245_5355_5050,
        AdaptiveLocalShakeArm::ResolutionDown => 0x4c53_5245_5344_4e4e,
        AdaptiveLocalShakeArm::SeedLocalRefinement => 0x4c53_5345_4544_4c52,
    }
}

fn local_shake_near_tie_max_decisions(config: &DongdaemunRefinementConfig) -> usize {
    if config.adaptive_near_tie_max_decisions_per_parent > 0 {
        config.adaptive_near_tie_max_decisions_per_parent
    } else {
        config.max_extra_children_per_parent
    }
}

#[allow(clippy::too_many_arguments)]
fn maybe_apply_adaptive_local_shake_eager(
    choice: &mut ParentRefinementChoice,
    subgraph: &Graph,
    nodes: &[usize],
    config: &LeidenConfig,
    ddm_config: &DongdaemunRefinementConfig,
    standard_quality: f64,
    standard_largest_fraction: f64,
    standard_singleton_weight_fraction: f64,
    standard_margin_summary: Option<&local_merge::LocalMergeMarginSummary>,
    parent_weight: f64,
    trace_context: CandidateTraceContext,
    iteration: usize,
    randomness: f64,
    merge_ws: &mut local_merge::LocalMergeWorkspace,
    stats: &mut RefinementDongdaemunStats,
) {
    if !adaptive_local_shake_enabled(ddm_config) {
        return;
    }
    let specs = select_local_shake_arms(
        ddm_config,
        parent_weight,
        standard_largest_fraction,
        standard_singleton_weight_fraction,
        standard_margin_summary,
    );
    emit_local_shake_trigger_trace(
        trace_context,
        iteration,
        standard_largest_fraction,
        standard_singleton_weight_fraction,
        &specs,
        standard_margin_summary,
        ddm_config,
    );
    if specs.is_empty() {
        emit_local_shake_decision_trace(trace_context, None, false, "no_candidates", ddm_config);
        return;
    }
    stats.adaptive_local_shake_triggers += 1;
    let mut best: Option<LocalShakeCandidate> = None;
    for (candidate_index, spec) in specs.into_iter().enumerate() {
        let mut candidate_counts = Vec::new();
        let derived_seed = derive_adaptive_probe_seed(
            config.seed,
            trace_context.depth,
            trace_context.parent_id,
            trace_context.parent_visit_index,
            local_shake_seed_salt(spec.arm),
            candidate_index,
        );
        let mut candidate_rng = rand::rngs::StdRng::seed_from_u64(derived_seed);
        let (candidate_n_clusters, near_tie_summary) = match spec.arm {
            AdaptiveLocalShakeArm::NearTieRefinement => {
                let near_tie = local_merge::NearTieProbeConfig {
                    parent_weight,
                    margin_parent_weight: ddm_config
                        .adaptive_local_shake_near_tie_margin_parent_weight,
                    randomness: ddm_config.adaptive_local_shake_near_tie_randomness,
                    max_decisions_per_parent: local_shake_near_tie_max_decisions(ddm_config),
                };
                let (n_clusters, summary) =
                    local_merge::find_clustering_with_workspace_assignments_and_append_sizes_traced(
                        subgraph,
                        config.resolution,
                        randomness,
                        &mut candidate_rng,
                        merge_ws,
                        &mut candidate_counts,
                        ddm_config.adaptive_local_shake_near_tie_margin_parent_weight
                            * parent_weight.max(0.0),
                        Some(near_tie),
                    );
                (n_clusters, Some(summary))
            }
            AdaptiveLocalShakeArm::ResolutionUp | AdaptiveLocalShakeArm::ResolutionDown => {
                let n_clusters =
                    local_merge::find_clustering_with_workspace_assignments_and_append_sizes(
                        subgraph,
                        config.resolution * spec.multiplier,
                        randomness,
                        &mut candidate_rng,
                        merge_ws,
                        &mut candidate_counts,
                    );
                (n_clusters, None)
            }
            AdaptiveLocalShakeArm::SeedLocalRefinement => {
                let n_clusters =
                    local_merge::find_clustering_with_workspace_assignments_and_append_sizes(
                        subgraph,
                        config.resolution,
                        randomness,
                        &mut candidate_rng,
                        merge_ws,
                        &mut candidate_counts,
                    );
                (n_clusters, None)
            }
        };
        let candidate_assignments = merge_ws.assignments()[..nodes.len()].to_vec();
        let candidate_delta_q = parent_partition_quality_subgraph(
            subgraph,
            &candidate_assignments,
            candidate_n_clusters,
            config.resolution,
        ) - standard_quality;
        let (largest_fraction, singleton_weight_fraction) = parent_partition_summary(
            nodes.len(),
            candidate_n_clusters,
            &candidate_assignments,
            parent_weight,
            |local| subgraph.node_weights[local],
        );
        let candidate = build_local_shake_candidate(
            spec,
            candidate_index,
            choice,
            candidate_assignments,
            candidate_counts,
            candidate_n_clusters,
            largest_fraction,
            singleton_weight_fraction,
            candidate_delta_q,
            standard_largest_fraction,
            parent_weight,
            ddm_config,
            near_tie_summary,
        );
        stats.adaptive_local_shake_candidates += 1;
        let commit_eligible = local_shake_commit_eligible(&candidate, ddm_config, parent_weight);
        let commit_block_reason = if commit_eligible {
            "eligible"
        } else {
            local_shake_commit_block_reason(&candidate, ddm_config, parent_weight)
        };
        emit_local_shake_candidate_trace(
            trace_context,
            &candidate,
            commit_eligible,
            commit_block_reason,
            ddm_config,
        );
        reduce_local_shake_candidate(&mut best, candidate, ddm_config, parent_weight);
    }
    let committed = best
        .as_ref()
        .map(|candidate| local_shake_commit_eligible(candidate, ddm_config, parent_weight))
        .unwrap_or(false);
    let decision_reason = if committed {
        if choice.source.is_some() {
            "replaced_candidate"
        } else {
            "committed"
        }
    } else {
        best.as_ref()
            .map(|candidate| local_shake_commit_block_reason(candidate, ddm_config, parent_weight))
            .unwrap_or("no_candidates")
    };
    emit_local_shake_decision_trace(
        trace_context,
        best.as_ref(),
        committed,
        decision_reason,
        ddm_config,
    );
    if committed {
        if let Some(candidate) = best {
            commit_local_shake_candidate(choice, candidate, stats);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn maybe_apply_adaptive_local_shake_streaming(
    choice: &mut ParentRefinementChoice,
    graph: &Graph,
    nodes: &[u32],
    local_index: &mut [u32],
    config: &LeidenConfig,
    ddm_config: &DongdaemunRefinementConfig,
    standard_quality: f64,
    standard_largest_fraction: f64,
    standard_singleton_weight_fraction: f64,
    standard_margin_summary: Option<&local_merge::LocalMergeMarginSummary>,
    parent_weight: f64,
    trace_context: CandidateTraceContext,
    iteration: usize,
    randomness: f64,
    merge_ws: &mut local_merge::LocalMergeWorkspace,
    stats: &mut RefinementDongdaemunStats,
) {
    if !adaptive_local_shake_enabled(ddm_config) {
        return;
    }
    let specs = select_local_shake_arms(
        ddm_config,
        parent_weight,
        standard_largest_fraction,
        standard_singleton_weight_fraction,
        standard_margin_summary,
    );
    emit_local_shake_trigger_trace(
        trace_context,
        iteration,
        standard_largest_fraction,
        standard_singleton_weight_fraction,
        &specs,
        standard_margin_summary,
        ddm_config,
    );
    if specs.is_empty() {
        emit_local_shake_decision_trace(trace_context, None, false, "no_candidates", ddm_config);
        return;
    }
    stats.adaptive_local_shake_triggers += 1;
    let mut best: Option<LocalShakeCandidate> = None;
    for (candidate_index, spec) in specs.into_iter().enumerate() {
        let mut candidate_counts = Vec::new();
        let derived_seed = derive_adaptive_probe_seed(
            config.seed,
            trace_context.depth,
            trace_context.parent_id,
            trace_context.parent_visit_index,
            local_shake_seed_salt(spec.arm),
            candidate_index,
        );
        let mut candidate_rng = rand::rngs::StdRng::seed_from_u64(derived_seed);
        let (candidate_n_clusters, near_tie_summary) = match spec.arm {
            AdaptiveLocalShakeArm::NearTieRefinement => {
                let near_tie = local_merge::NearTieProbeConfig {
                    parent_weight,
                    margin_parent_weight: ddm_config
                        .adaptive_local_shake_near_tie_margin_parent_weight,
                    randomness: ddm_config.adaptive_local_shake_near_tie_randomness,
                    max_decisions_per_parent: local_shake_near_tie_max_decisions(ddm_config),
                };
                let (n_clusters, summary) =
                    local_merge::find_clustering_induced_u32_with_workspace_assignments_and_append_sizes_traced(
                        graph,
                        nodes,
                        local_index,
                        config.resolution,
                        randomness,
                        &mut candidate_rng,
                        merge_ws,
                        &mut candidate_counts,
                        ddm_config.adaptive_local_shake_near_tie_margin_parent_weight
                            * parent_weight.max(0.0),
                        Some(near_tie),
                    );
                (n_clusters, Some(summary))
            }
            AdaptiveLocalShakeArm::ResolutionUp | AdaptiveLocalShakeArm::ResolutionDown => {
                let n_clusters =
                    local_merge::find_clustering_induced_u32_with_workspace_assignments_and_append_sizes(
                        graph,
                        nodes,
                        local_index,
                        config.resolution * spec.multiplier,
                        randomness,
                        &mut candidate_rng,
                        merge_ws,
                        &mut candidate_counts,
                    );
                (n_clusters, None)
            }
            AdaptiveLocalShakeArm::SeedLocalRefinement => {
                let n_clusters =
                    local_merge::find_clustering_induced_u32_with_workspace_assignments_and_append_sizes(
                        graph,
                        nodes,
                        local_index,
                        config.resolution,
                        randomness,
                        &mut candidate_rng,
                        merge_ws,
                        &mut candidate_counts,
                    );
                (n_clusters, None)
            }
        };
        let candidate_assignments = merge_ws.assignments()[..nodes.len()].to_vec();
        let candidate_delta_q = parent_partition_quality_induced_u32(
            graph,
            nodes,
            local_index,
            &candidate_assignments,
            candidate_n_clusters,
            config.resolution,
        ) - standard_quality;
        let (largest_fraction, singleton_weight_fraction) = parent_partition_summary(
            nodes.len(),
            candidate_n_clusters,
            &candidate_assignments,
            parent_weight,
            |local| graph.node_weights[nodes[local] as usize],
        );
        let candidate = build_local_shake_candidate(
            spec,
            candidate_index,
            choice,
            candidate_assignments,
            candidate_counts,
            candidate_n_clusters,
            largest_fraction,
            singleton_weight_fraction,
            candidate_delta_q,
            standard_largest_fraction,
            parent_weight,
            ddm_config,
            near_tie_summary,
        );
        stats.adaptive_local_shake_candidates += 1;
        let commit_eligible = local_shake_commit_eligible(&candidate, ddm_config, parent_weight);
        let commit_block_reason = if commit_eligible {
            "eligible"
        } else {
            local_shake_commit_block_reason(&candidate, ddm_config, parent_weight)
        };
        emit_local_shake_candidate_trace(
            trace_context,
            &candidate,
            commit_eligible,
            commit_block_reason,
            ddm_config,
        );
        reduce_local_shake_candidate(&mut best, candidate, ddm_config, parent_weight);
    }
    let committed = best
        .as_ref()
        .map(|candidate| local_shake_commit_eligible(candidate, ddm_config, parent_weight))
        .unwrap_or(false);
    let decision_reason = if committed {
        if choice.source.is_some() {
            "replaced_candidate"
        } else {
            "committed"
        }
    } else {
        best.as_ref()
            .map(|candidate| local_shake_commit_block_reason(candidate, ddm_config, parent_weight))
            .unwrap_or("no_candidates")
    };
    emit_local_shake_decision_trace(
        trace_context,
        best.as_ref(),
        committed,
        decision_reason,
        ddm_config,
    );
    if committed {
        if let Some(candidate) = best {
            commit_local_shake_candidate(choice, candidate, stats);
        }
    }
}

fn refinement_candidate_quadrant(
    candidate_delta_q: f64,
    largest_fraction: f64,
    standard_largest_fraction: f64,
) -> RefinementCandidateQuadrant {
    match (
        candidate_delta_q > 0.0,
        largest_fraction < standard_largest_fraction,
    ) {
        (true, true) => RefinementCandidateQuadrant::QPosSPos,
        (true, false) => RefinementCandidateQuadrant::QPosSNeg,
        (false, true) => RefinementCandidateQuadrant::QNegSPos,
        (false, false) => RefinementCandidateQuadrant::QNegSNeg,
    }
}

impl RefinementDongdaemunStats {
    fn next_candidate_trace_id(&self) -> usize {
        self.same_gamma_candidates + self.high_gamma_candidates
    }

    fn record_candidate(&mut self, source: RefinementCandidateSource) {
        match source {
            RefinementCandidateSource::SameGammaSeed => self.same_gamma_candidates += 1,
            RefinementCandidateSource::HighGamma => self.high_gamma_candidates += 1,
            RefinementCandidateSource::NearTieRefinementProbe => {}
        }
    }

    fn record_candidate_quality(
        &mut self,
        source: RefinementCandidateSource,
        candidate_delta_q: f64,
    ) {
        self.candidate_quality_delta_sum += candidate_delta_q;
        if candidate_delta_q > 0.0 {
            self.candidate_positive_quality_delta += 1;
        }
        match source {
            RefinementCandidateSource::SameGammaSeed => {
                self.same_gamma_quality_delta_sum += candidate_delta_q;
                if candidate_delta_q > 0.0 {
                    self.same_gamma_positive_quality_delta += 1;
                }
            }
            RefinementCandidateSource::HighGamma => {
                self.high_gamma_quality_delta_sum += candidate_delta_q;
                if candidate_delta_q > 0.0 {
                    self.high_gamma_positive_quality_delta += 1;
                }
            }
            RefinementCandidateSource::NearTieRefinementProbe => {}
        }
    }

    fn record_candidate_quadrant(
        &mut self,
        source: RefinementCandidateSource,
        quadrant: RefinementCandidateQuadrant,
    ) {
        match quadrant {
            RefinementCandidateQuadrant::QPosSPos => self.candidate_qpos_spos += 1,
            RefinementCandidateQuadrant::QPosSNeg => self.candidate_qpos_sneg += 1,
            RefinementCandidateQuadrant::QNegSPos => self.candidate_qneg_spos += 1,
            RefinementCandidateQuadrant::QNegSNeg => self.candidate_qneg_sneg += 1,
        }
        match (source, quadrant) {
            (RefinementCandidateSource::SameGammaSeed, RefinementCandidateQuadrant::QPosSPos) => {
                self.same_gamma_qpos_spos += 1;
            }
            (RefinementCandidateSource::SameGammaSeed, RefinementCandidateQuadrant::QPosSNeg) => {
                self.same_gamma_qpos_sneg += 1;
            }
            (RefinementCandidateSource::SameGammaSeed, RefinementCandidateQuadrant::QNegSPos) => {
                self.same_gamma_qneg_spos += 1;
            }
            (RefinementCandidateSource::SameGammaSeed, RefinementCandidateQuadrant::QNegSNeg) => {
                self.same_gamma_qneg_sneg += 1;
            }
            (RefinementCandidateSource::HighGamma, RefinementCandidateQuadrant::QPosSPos) => {
                self.high_gamma_qpos_spos += 1;
            }
            (RefinementCandidateSource::HighGamma, RefinementCandidateQuadrant::QPosSNeg) => {
                self.high_gamma_qpos_sneg += 1;
            }
            (RefinementCandidateSource::HighGamma, RefinementCandidateQuadrant::QNegSPos) => {
                self.high_gamma_qneg_spos += 1;
            }
            (RefinementCandidateSource::HighGamma, RefinementCandidateQuadrant::QNegSNeg) => {
                self.high_gamma_qneg_sneg += 1;
            }
            (RefinementCandidateSource::NearTieRefinementProbe, _) => {}
        }
    }

    fn record_candidate_validity(&mut self, source: RefinementCandidateSource, is_valid: bool) {
        if is_valid {
            self.candidate_valid += 1;
        } else {
            self.candidate_invalid += 1;
        }
        match (source, is_valid) {
            (RefinementCandidateSource::SameGammaSeed, true) => self.same_gamma_valid += 1,
            (RefinementCandidateSource::SameGammaSeed, false) => self.same_gamma_invalid += 1,
            (RefinementCandidateSource::HighGamma, true) => self.high_gamma_valid += 1,
            (RefinementCandidateSource::HighGamma, false) => self.high_gamma_invalid += 1,
            (RefinementCandidateSource::NearTieRefinementProbe, _) => {}
        }
    }

    fn record_candidate_rejected_decision(&mut self, quadrant: RefinementCandidateQuadrant) {
        match quadrant {
            RefinementCandidateQuadrant::QPosSPos => self.candidate_false_negative += 1,
            RefinementCandidateQuadrant::QPosSNeg
            | RefinementCandidateQuadrant::QNegSPos
            | RefinementCandidateQuadrant::QNegSNeg => self.candidate_true_negative += 1,
        }
    }

    fn record_policy_rejected(
        &mut self,
        source: RefinementCandidateSource,
        quadrant: RefinementCandidateQuadrant,
    ) {
        self.candidate_rejected_by_policy += 1;
        match source {
            RefinementCandidateSource::SameGammaSeed => {
                self.same_gamma_rejected_by_policy += 1;
            }
            RefinementCandidateSource::HighGamma => {
                self.high_gamma_rejected_by_policy += 1;
            }
            RefinementCandidateSource::NearTieRefinementProbe => {}
        }
        self.record_candidate_rejected_decision(quadrant);
    }

    fn record_applied(&mut self, source: RefinementCandidateSource) {
        match source {
            RefinementCandidateSource::SameGammaSeed => self.same_gamma_applied += 1,
            RefinementCandidateSource::HighGamma => self.high_gamma_applied += 1,
            RefinementCandidateSource::NearTieRefinementProbe => {}
        }
    }

    fn record_selected_candidate_quality(
        &mut self,
        source: RefinementCandidateSource,
        candidate_delta_q: f64,
    ) {
        if candidate_delta_q <= 0.0 {
            return;
        }
        self.candidate_selected_positive_quality_delta += 1;
        match source {
            RefinementCandidateSource::SameGammaSeed => {
                self.same_gamma_selected_positive_quality_delta += 1;
            }
            RefinementCandidateSource::HighGamma => {
                self.high_gamma_selected_positive_quality_delta += 1;
            }
            RefinementCandidateSource::NearTieRefinementProbe => {}
        }
    }

    fn record_quality_rejected(&mut self, source: RefinementCandidateSource) {
        self.candidate_rejected_by_quality += 1;
        match source {
            RefinementCandidateSource::SameGammaSeed => {
                self.same_gamma_rejected_by_quality += 1;
            }
            RefinementCandidateSource::HighGamma => {
                self.high_gamma_rejected_by_quality += 1;
            }
            RefinementCandidateSource::NearTieRefinementProbe => {}
        }
    }

    fn record_selected_candidate_decision(&mut self, quadrant: RefinementCandidateQuadrant) {
        match quadrant {
            RefinementCandidateQuadrant::QPosSPos => self.candidate_true_positive += 1,
            RefinementCandidateQuadrant::QPosSNeg
            | RefinementCandidateQuadrant::QNegSPos
            | RefinementCandidateQuadrant::QNegSNeg => self.candidate_false_positive += 1,
        }
    }

    fn record_quotient_candidate(&mut self, score: f64) {
        self.quotient_candidates += 1;
        if score > 0.0 {
            self.quotient_positive_candidates += 1;
        }
    }

    fn record_quotient_selected(&mut self, score: f64) {
        if score > 0.0 {
            self.quotient_selected += 1;
            self.quotient_score_sum += score;
        }
    }

    fn record_baseline_repair_candidate(&mut self, repair: &BaselineRepairResult) {
        self.baseline_repair_candidates += 1;
        if repair.changed {
            self.baseline_repair_improved_candidates += 1;
        }
        self.baseline_repair_merge_count += repair.merge_count;
        self.baseline_repair_delta_sum += repair.delta_sum;
    }

    fn record_baseline_repair_selected(&mut self, repair_merge_count: usize) {
        if repair_merge_count > 0 {
            self.baseline_repair_selected += 1;
        }
    }
}

#[derive(Debug)]
struct ParentRefinementChoice {
    assignments: Vec<u32>,
    counts: Vec<u32>,
    n_clusters: usize,
    largest_fraction: f64,
    singleton_weight_fraction: f64,
    quotient_score: f64,
    candidate_delta_q: f64,
    source: Option<RefinementCandidateSource>,
    trace_candidate_id: Option<usize>,
    baseline_repair_merge_count: usize,
    baseline_repair_delta_sum: f64,
    adaptive_probe_baseline_delta_q: Option<f64>,
    adaptive_probe_committed: bool,
    adaptive_probe_score: f64,
    adaptive_probe_source_label: Option<&'static str>,
}

#[derive(Debug)]
struct BaselineRepairResult {
    assignments: Vec<u32>,
    counts: Vec<u32>,
    n_clusters: usize,
    merge_count: usize,
    delta_sum: f64,
    changed: bool,
}

fn parent_candidate_quotient_score<F>(
    graph: &Graph,
    clustering: &Clustering,
    parent_id: usize,
    local_len: usize,
    candidate_assignments: &[u32],
    candidate_n_clusters: usize,
    parent_weight: f64,
    parent_weights: &[f64],
    resolution: f64,
    node_at: F,
) -> f64
where
    F: Fn(usize) -> usize,
{
    if candidate_n_clusters == 0 || parent_weight <= 0.0 {
        return 0.0;
    }

    let mut child_weights = vec![0.0; candidate_n_clusters];
    let mut child_external_edges: Vec<HashMap<usize, f64>> =
        (0..candidate_n_clusters).map(|_| HashMap::new()).collect();

    for local in 0..local_len {
        let child = candidate_assignments[local] as usize;
        if child >= candidate_n_clusters {
            continue;
        }
        let node = node_at(local);
        child_weights[child] += graph.node_weights[node];
        for (neighbor, edge_weight) in graph.neighbors_of(node) {
            let neighbor_parent = clustering.clusters[neighbor as usize] as usize;
            if neighbor_parent == parent_id {
                continue;
            }
            *child_external_edges[child]
                .entry(neighbor_parent)
                .or_insert(0.0) += edge_weight;
        }
    }

    let mut positive_delta_sum = 0.0;
    let mut positive_attached_child_weight = 0.0;
    for child in 0..candidate_n_clusters {
        let child_weight = child_weights[child];
        if child_weight <= 0.0 {
            continue;
        }
        let mut best_delta = 0.0;
        for (&neighbor_parent, &edge_weight) in &child_external_edges[child] {
            let neighbor_weight = parent_weights.get(neighbor_parent).copied().unwrap_or(0.0);
            let delta = edge_weight - resolution * child_weight * neighbor_weight;
            if delta > best_delta {
                best_delta = delta;
            }
        }
        if best_delta > 0.0 {
            positive_delta_sum += best_delta;
            positive_attached_child_weight += child_weight;
        }
    }

    positive_delta_sum / parent_weight.max(1.0) + positive_attached_child_weight / parent_weight
}

fn add_parent_internal_repair_edge(
    adj: &mut [HashMap<usize, f64>],
    left: usize,
    right: usize,
    weight: f64,
) {
    if left == right {
        return;
    }
    *adj[left].entry(right).or_insert(0.0) += weight;
    *adj[right].entry(left).or_insert(0.0) += weight;
}

fn find_repair_root(parent: &mut [usize], node: usize) -> usize {
    let mut root = node;
    while parent[root] != root {
        root = parent[root];
    }

    let mut current = node;
    while parent[current] != current {
        let next = parent[current];
        parent[current] = root;
        current = next;
    }
    root
}

fn repair_parent_internal_candidate<F>(
    graph: &Graph,
    local_len: usize,
    assignments: &[u32],
    n_clusters: usize,
    resolution: f64,
    epsilon: f64,
    mut node_at: F,
) -> BaselineRepairResult
where
    F: FnMut(usize) -> usize,
{
    if local_len == 0 || n_clusters <= 1 {
        let mut counts = vec![0u32; n_clusters];
        for &cluster in assignments.iter().take(local_len) {
            let cluster = cluster as usize;
            if cluster < n_clusters {
                counts[cluster] += 1;
            }
        }
        return BaselineRepairResult {
            assignments: assignments.iter().take(local_len).copied().collect(),
            counts,
            n_clusters,
            merge_count: 0,
            delta_sum: 0.0,
            changed: false,
        };
    }

    let mut local_nodes = Vec::with_capacity(local_len);
    let mut local_lookup = HashMap::with_capacity(local_len);
    let mut child_weights = vec![0.0; n_clusters];
    for local in 0..local_len {
        let node = node_at(local);
        local_lookup.insert(node, local);
        local_nodes.push(node);
        let child = assignments[local] as usize;
        if child < n_clusters {
            child_weights[child] += graph.node_weights[node];
        }
    }

    let mut adj: Vec<HashMap<usize, f64>> = (0..n_clusters).map(|_| HashMap::new()).collect();
    for (local, &node) in local_nodes.iter().enumerate() {
        let child = assignments[local] as usize;
        if child >= n_clusters {
            continue;
        }
        for (neighbor, edge_weight) in graph.neighbors_of(node) {
            let neighbor = neighbor as usize;
            let Some(&neighbor_local) = local_lookup.get(&neighbor) else {
                continue;
            };
            if local >= neighbor_local {
                continue;
            }
            let neighbor_child = assignments[neighbor_local] as usize;
            if neighbor_child < n_clusters {
                add_parent_internal_repair_edge(&mut adj, child, neighbor_child, edge_weight);
            }
        }
    }

    let mut active = vec![true; n_clusters];
    let mut parent: Vec<usize> = (0..n_clusters).collect();
    let mut merge_count = 0usize;
    let mut delta_sum = 0.0;

    loop {
        let mut best_delta = f64::NEG_INFINITY;
        let mut best_pair: Option<(usize, usize)> = None;
        for u in 0..n_clusters {
            if !active[u] {
                continue;
            }
            for (&v, &edge_weight) in &adj[u] {
                if u >= v || !active[v] {
                    continue;
                }
                let delta = edge_weight - resolution * child_weights[u] * child_weights[v];
                let replace = match best_pair {
                    None => true,
                    Some((best_u, best_v)) => delta
                        .total_cmp(&best_delta)
                        .then_with(|| best_u.cmp(&u))
                        .then_with(|| best_v.cmp(&v))
                        .is_gt(),
                };
                if replace {
                    best_delta = delta;
                    best_pair = Some((u, v));
                }
            }
        }

        let Some((u, v)) = best_pair else {
            break;
        };
        let should_merge = if epsilon > 0.0 {
            best_delta >= -epsilon
        } else {
            best_delta > 0.0
        };
        if !should_merge {
            break;
        }

        let (keep, remove) = match child_weights[u].total_cmp(&child_weights[v]) {
            std::cmp::Ordering::Greater => (u, v),
            std::cmp::Ordering::Less => (v, u),
            std::cmp::Ordering::Equal => {
                if u <= v {
                    (u, v)
                } else {
                    (v, u)
                }
            }
        };

        merge_count += 1;
        delta_sum += best_delta;
        child_weights[keep] += child_weights[remove];
        child_weights[remove] = 0.0;
        parent[remove] = keep;
        active[remove] = false;

        adj[keep].remove(&remove);
        let removed_neighbors: Vec<(usize, f64)> = adj[remove].drain().collect();
        for (neighbor, weight) in removed_neighbors {
            if neighbor == keep || !active[neighbor] {
                continue;
            }
            adj[neighbor].remove(&remove);
            add_parent_internal_repair_edge(&mut adj, keep, neighbor, weight);
        }
    }

    if merge_count == 0 {
        let mut counts = vec![0u32; n_clusters];
        for &cluster in assignments.iter().take(local_len) {
            let cluster = cluster as usize;
            if cluster < n_clusters {
                counts[cluster] += 1;
            }
        }
        return BaselineRepairResult {
            assignments: assignments.iter().take(local_len).copied().collect(),
            counts,
            n_clusters,
            merge_count,
            delta_sum,
            changed: false,
        };
    }

    let mut repaired_roots = Vec::with_capacity(local_len);
    let mut root_counts = vec![0u32; n_clusters];
    for &cluster in assignments.iter().take(local_len) {
        let root = find_repair_root(&mut parent, cluster as usize);
        repaired_roots.push(root);
        root_counts[root] += 1;
    }

    let mut remap = vec![u32::MAX; n_clusters];
    let mut counts = Vec::new();
    for (root, &count) in root_counts.iter().enumerate() {
        if count > 0 {
            remap[root] = counts.len() as u32;
            counts.push(count);
        }
    }
    let assignments = repaired_roots
        .into_iter()
        .map(|root| remap[root])
        .collect::<Vec<_>>();
    let n_clusters = counts.len();

    BaselineRepairResult {
        assignments,
        counts,
        n_clusters,
        merge_count,
        delta_sum,
        changed: true,
    }
}

fn mix_seed(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

fn derive_same_gamma_seed(
    base_seed: u64,
    depth: usize,
    parent_id: usize,
    perturbation_index: usize,
) -> u64 {
    let mut seed = mix_seed(base_seed ^ 0xD0A1_DA3E_5EED_0001);
    seed = mix_seed(seed ^ depth as u64);
    seed = mix_seed(seed ^ (parent_id as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    mix_seed(seed ^ ((perturbation_index as u64) + 1))
}

fn derive_high_gamma_seed(
    base_seed: u64,
    depth: usize,
    parent_id: usize,
    multiplier_index: usize,
) -> u64 {
    let mut seed = mix_seed(base_seed ^ 0xD0A1_DA3E_5EED_0002);
    seed = mix_seed(seed ^ depth as u64);
    seed = mix_seed(seed ^ (parent_id as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9));
    mix_seed(seed ^ ((multiplier_index as u64) + 1))
}

fn derive_adaptive_probe_seed(
    base_seed: u64,
    depth: usize,
    parent_id: usize,
    parent_visit_index: usize,
    source_salt: u64,
    perturbation_index: usize,
) -> u64 {
    let mut seed = mix_seed(base_seed ^ 0xD0A1_DA3E_5EED_0003);
    seed = mix_seed(seed ^ depth as u64);
    seed = mix_seed(seed ^ (parent_id as u64).wrapping_mul(0xD6E8_FD50_9A21_7C15));
    seed = mix_seed(seed ^ (parent_visit_index as u64).wrapping_mul(0xA24B_AED4_963E_E407));
    seed = mix_seed(seed ^ source_salt);
    mix_seed(seed ^ ((perturbation_index as u64) + 1))
}

fn effective_baseline_repair_policy(
    config: &DongdaemunRefinementConfig,
    parent_weight: f64,
) -> BaselineRepairPolicy {
    match config.baseline_repair_policy {
        BaselineRepairPolicy::Adaptive => {
            if parent_weight
                >= config.target_max_weight * config.baseline_repair_replace_min_parent_ratio
            {
                BaselineRepairPolicy::Replace
            } else {
                BaselineRepairPolicy::Augment
            }
        }
        policy => policy,
    }
}

fn candidate_max_child_weight_ratio(
    parent_weight: f64,
    largest_fraction: f64,
    target_max_weight: f64,
) -> f64 {
    if parent_weight > 0.0 && target_max_weight > 0.0 {
        parent_weight * largest_fraction / target_max_weight
    } else {
        f64::NAN
    }
}

fn oversize_pressure_excess(weight_ratio: f64) -> f64 {
    if weight_ratio.is_finite() {
        (weight_ratio - 1.0).max(0.0)
    } else {
        f64::NAN
    }
}

fn emit_candidate_profile_trace(
    trace_context: CandidateTraceContext,
    candidate_trace_id: usize,
    source: RefinementCandidateSource,
    quadrant: RefinementCandidateQuadrant,
    candidate_n_clusters: usize,
    largest_fraction: f64,
    singleton_weight_fraction: f64,
    quotient_score: Option<f64>,
    baseline_repair_merge_count: usize,
    baseline_repair_delta_sum: f64,
    candidate_delta_q: f64,
    standard_largest_fraction: f64,
    is_valid: bool,
    quality_passes: bool,
    decision: &str,
    target_max_weight: f64,
    adaptive_quality_band: f64,
    adaptive_plateau_compared: bool,
) {
    if !trace::ddm_candidate_trace_enabled() {
        return;
    }
    let standard_max_child_weight_ratio = candidate_max_child_weight_ratio(
        trace_context.parent_weight,
        standard_largest_fraction,
        target_max_weight,
    );
    let candidate_max_child_weight_ratio = candidate_max_child_weight_ratio(
        trace_context.parent_weight,
        largest_fraction,
        target_max_weight,
    );
    let pressure_reduction = oversize_pressure_excess(standard_max_child_weight_ratio)
        - oversize_pressure_excess(candidate_max_child_weight_ratio);
    let quotient_score = quotient_score.unwrap_or(0.0);
    let adaptive_diagnostic_score = adaptive_diagnostic_score(
        trace_context.parent_weight,
        target_max_weight,
        standard_largest_fraction,
        largest_fraction,
        singleton_weight_fraction,
        quotient_score,
    );
    trace::emit_ddm_candidate_trace(format_args!(
        "{{\"event\":\"candidate_profile\",\"run_id\":{},\"depth\":{},\"parent_id\":{},\"parent_visit_index\":{},\"candidate_id\":{},\"source\":\"{}\",\"source_index\":{},\"gamma_multiplier\":{},\"repaired\":{},\"parent_size\":{},\"parent_weight\":{},\"standard_n_clusters\":{},\"candidate_n_clusters\":{},\"standard_largest_child_fraction\":{},\"largest_child_fraction\":{},\"largest_child_fraction_improvement\":{},\"standard_max_child_weight_ratio\":{},\"candidate_max_child_weight_ratio\":{},\"pressure_reduction\":{},\"singleton_weight_fraction\":{},\"candidate_delta_q\":{},\"adaptive_diagnostic_score\":{},\"adaptive_quality_band\":{},\"adaptive_plateau_compared\":{},\"quadrant\":\"{}\",\"valid\":{},\"quality_passes\":{},\"decision\":\"{}\",\"quotient_score\":{},\"baseline_repair_merge_count\":{},\"baseline_repair_delta_sum\":{}}}",
        candidate_trace_run_id_json(),
        trace_context.depth,
        trace_context.parent_id,
        trace_context.parent_visit_index,
        candidate_trace_id,
        source.as_trace_str(),
        trace_context.source_index,
        trace_json_f64(trace_context.gamma_multiplier),
        trace_context.repaired,
        trace_context.parent_size,
        trace_json_f64(trace_context.parent_weight),
        trace_context.standard_n_clusters,
        candidate_n_clusters,
        trace_json_f64(standard_largest_fraction),
        trace_json_f64(largest_fraction),
        trace_json_f64(standard_largest_fraction - largest_fraction),
        trace_json_f64(standard_max_child_weight_ratio),
        trace_json_f64(candidate_max_child_weight_ratio),
        trace_json_f64(pressure_reduction),
        trace_json_f64(singleton_weight_fraction),
        trace_json_f64(candidate_delta_q),
        trace_json_f64(adaptive_diagnostic_score),
        trace_json_f64(adaptive_quality_band),
        adaptive_plateau_compared,
        quadrant.as_trace_str(),
        is_valid,
        quality_passes,
        decision,
        trace_json_f64(quotient_score),
        baseline_repair_merge_count,
        trace_json_f64(baseline_repair_delta_sum),
    ));
}

fn emit_candidate_decision_trace(
    trace_context: CandidateTraceContext,
    candidate_trace_id: Option<usize>,
    decision: &str,
) {
    if !trace::ddm_candidate_trace_enabled() {
        return;
    }
    let Some(candidate_trace_id) = candidate_trace_id else {
        return;
    };
    trace::emit_ddm_candidate_trace(format_args!(
        "{{\"event\":\"candidate_decision\",\"run_id\":{},\"depth\":{},\"parent_id\":{},\"parent_visit_index\":{},\"candidate_id\":{},\"decision\":\"{}\"}}",
        candidate_trace_run_id_json(),
        trace_context.depth,
        trace_context.parent_id,
        trace_context.parent_visit_index,
        candidate_trace_id,
        decision
    ));
}

#[allow(clippy::too_many_arguments)]
fn emit_adaptive_probe_trace(
    trace_context: CandidateTraceContext,
    probe_source: &str,
    source_index: usize,
    candidate_n_clusters: usize,
    largest_fraction: f64,
    singleton_weight_fraction: f64,
    candidate_delta_q: f64,
    baseline_candidate_delta_q: f64,
    tolerance_delta_q: f64,
    standard_largest_fraction: f64,
    is_valid: bool,
    quality_passes: bool,
    local_win: bool,
    mode: AdaptiveProbeMode,
    commit_eligible: bool,
    committed: bool,
    commit_block_reason: &str,
    commit_gain_parent_weight: f64,
    commit_count_total_before: usize,
    commit_count_depth_before: usize,
    commit_strategy: AdaptiveProbeCommitStrategy,
    commit_strategy_score: f64,
) {
    if !trace::ddm_candidate_trace_enabled() {
        return;
    }
    trace::emit_ddm_candidate_trace(format_args!(
        "{{\"event\":\"adaptive_probe_candidate\",\"run_id\":{},\"depth\":{},\"parent_id\":{},\"parent_visit_index\":{},\"source\":\"{}\",\"source_index\":{},\"mode\":\"{}\",\"parent_size\":{},\"parent_weight\":{},\"standard_n_clusters\":{},\"candidate_n_clusters\":{},\"standard_largest_child_fraction\":{},\"largest_child_fraction\":{},\"singleton_weight_fraction\":{},\"candidate_delta_q\":{},\"baseline_candidate_delta_q\":{},\"tolerance_delta_q\":{},\"gain_vs_baseline\":{},\"valid\":{},\"quality_passes\":{},\"local_win\":{},\"commit_eligible\":{},\"committed\":{},\"commit_block_reason\":\"{}\",\"commit_gain_parent_weight\":{},\"commit_count_total_before\":{},\"commit_count_depth_before\":{},\"commit_strategy\":\"{}\",\"commit_strategy_score\":{}}}",
        candidate_trace_run_id_json(),
        trace_context.depth,
        trace_context.parent_id,
        trace_context.parent_visit_index,
        probe_source,
        source_index,
        match mode {
            AdaptiveProbeMode::Off => "off",
            AdaptiveProbeMode::TraceOnly => "trace_only",
            AdaptiveProbeMode::ApplyIfWin => "apply_if_win",
            AdaptiveProbeMode::ConservativeApply => "conservative_apply",
        },
        trace_context.parent_size,
        trace_json_f64(trace_context.parent_weight),
        trace_context.standard_n_clusters,
        candidate_n_clusters,
        trace_json_f64(standard_largest_fraction),
        trace_json_f64(largest_fraction),
        trace_json_f64(singleton_weight_fraction),
        trace_json_f64(candidate_delta_q),
        trace_json_f64(baseline_candidate_delta_q),
        trace_json_f64(tolerance_delta_q),
        trace_json_f64(candidate_delta_q - baseline_candidate_delta_q),
        is_valid,
        quality_passes,
        local_win,
        commit_eligible,
        committed,
        commit_block_reason,
        trace_json_f64(commit_gain_parent_weight),
        commit_count_total_before,
        commit_count_depth_before,
        adaptive_probe_commit_strategy_trace(commit_strategy),
        trace_json_f64(commit_strategy_score),
    ));
}

#[allow(clippy::too_many_arguments)]
fn emit_local_merge_margin_summary_trace(
    trace_context: CandidateTraceContext,
    iteration: usize,
    source: &str,
    selected_child_count: usize,
    largest_child_fraction: f64,
    summary: &local_merge::LocalMergeMarginSummary,
) {
    if !trace::ddm_trajectory_trace_enabled() {
        return;
    }
    trace::emit_ddm_trajectory_trace(format_args!(
        "{{\"schema\":\"dongdaemun_trajectory_trace.v1\",\"event\":\"local_merge_margin_summary\",\"run_id\":{},\"depth\":{},\"iteration\":{},\"parent_id\":{},\"parent_visit_index\":{},\"source\":\"{}\",\"parent_size\":{},\"parent_weight\":{},\"decision_count\":{},\"low_margin_decision_count\":{},\"changed_decision_count\":{},\"min_margin\":{},\"p10_margin\":{},\"p50_margin\":{},\"selected_child_count\":{},\"largest_child_fraction\":{}}}",
        trajectory_trace_run_id_json(),
        trace_context.depth,
        iteration,
        trace_context.parent_id,
        trace_context.parent_visit_index,
        source,
        trace_context.parent_size,
        trace::json_f64(trace_context.parent_weight),
        summary.decision_count,
        summary.low_margin_decision_count,
        summary.changed_decision_count,
        trace::json_f64(summary.min_margin),
        trace::json_f64(summary.p10_margin),
        trace::json_f64(summary.p50_margin),
        selected_child_count,
        trace::json_f64(largest_child_fraction),
    ));
}

#[allow(clippy::too_many_arguments)]
fn emit_near_tie_probe_trace(
    trace_context: CandidateTraceContext,
    candidate_n_clusters: usize,
    largest_fraction: f64,
    singleton_weight_fraction: f64,
    candidate_delta_q: f64,
    baseline_candidate_delta_q: f64,
    standard_largest_fraction: f64,
    is_valid: bool,
    quality_passes: bool,
    local_win: bool,
    mode: AdaptiveNearTieProbeMode,
    commit_eligible: bool,
    committed: bool,
    commit_block_reason: &str,
    summary: &local_merge::LocalMergeMarginSummary,
) {
    if !trace::ddm_candidate_trace_enabled() {
        return;
    }
    trace::emit_ddm_candidate_trace(format_args!(
        "{{\"event\":\"adaptive_probe_candidate\",\"run_id\":{},\"depth\":{},\"parent_id\":{},\"parent_visit_index\":{},\"source\":\"near_tie_refinement_probe\",\"source_index\":0,\"mode\":\"{}\",\"parent_size\":{},\"parent_weight\":{},\"standard_n_clusters\":{},\"candidate_n_clusters\":{},\"standard_largest_child_fraction\":{},\"largest_child_fraction\":{},\"singleton_weight_fraction\":{},\"candidate_delta_q\":{},\"baseline_candidate_delta_q\":{},\"tolerance_delta_q\":0.0,\"gain_vs_baseline\":{},\"valid\":{},\"quality_passes\":{},\"local_win\":{},\"commit_eligible\":{},\"committed\":{},\"commit_block_reason\":\"{}\",\"commit_gain_parent_weight\":{},\"commit_count_total_before\":0,\"commit_count_depth_before\":0,\"commit_strategy\":\"online_first\",\"commit_strategy_score\":0.0,\"near_tie_low_margin_decision_count\":{},\"near_tie_changed_decision_count\":{},\"near_tie_min_margin\":{},\"near_tie_p10_margin\":{},\"near_tie_p50_margin\":{}}}",
        candidate_trace_run_id_json(),
        trace_context.depth,
        trace_context.parent_id,
        trace_context.parent_visit_index,
        adaptive_near_tie_probe_mode_trace(mode),
        trace_context.parent_size,
        trace_json_f64(trace_context.parent_weight),
        trace_context.standard_n_clusters,
        candidate_n_clusters,
        trace_json_f64(standard_largest_fraction),
        trace_json_f64(largest_fraction),
        trace_json_f64(singleton_weight_fraction),
        trace_json_f64(candidate_delta_q),
        trace_json_f64(baseline_candidate_delta_q),
        trace_json_f64(candidate_delta_q - baseline_candidate_delta_q),
        is_valid,
        quality_passes,
        local_win,
        commit_eligible,
        committed,
        commit_block_reason,
        trace_json_f64((candidate_delta_q - baseline_candidate_delta_q) / trace_context.parent_weight.max(1.0)),
        summary.low_margin_decision_count,
        summary.changed_decision_count,
        trace_json_f64(summary.min_margin),
        trace_json_f64(summary.p10_margin),
        trace_json_f64(summary.p50_margin),
    ));
}

#[allow(clippy::too_many_arguments)]
fn maybe_apply_near_tie_probe_candidate(
    choice: &mut ParentRefinementChoice,
    candidate_assignments: Vec<u32>,
    candidate_counts: Vec<u32>,
    candidate_n_clusters: usize,
    largest_fraction: f64,
    singleton_weight_fraction: f64,
    candidate_delta_q: f64,
    standard_largest_fraction: f64,
    config: &DongdaemunRefinementConfig,
    trace_context: CandidateTraceContext,
    summary: &local_merge::LocalMergeMarginSummary,
) {
    let is_valid = parent_candidate_is_valid(
        candidate_n_clusters,
        largest_fraction,
        singleton_weight_fraction,
        standard_largest_fraction,
        config,
    );
    let quality_passes = candidate_quality_passes(candidate_delta_q, config);
    let baseline_candidate_delta_q = choice.candidate_delta_q;
    let local_win = is_valid && quality_passes && candidate_delta_q > baseline_candidate_delta_q;
    let changed = summary.changed_decision_count > 0;
    let mut commit_eligible = false;
    let mut committed = false;
    let commit_block_reason = match config.adaptive_near_tie_probe_mode {
        AdaptiveNearTieProbeMode::Off => "off",
        AdaptiveNearTieProbeMode::TraceOnly => "trace_only",
        AdaptiveNearTieProbeMode::Candidate | AdaptiveNearTieProbeMode::QfReplace => {
            if !changed {
                "unchanged_candidate"
            } else if !is_valid {
                "invalid_candidate"
            } else if !quality_passes {
                "quality_rejected"
            } else if !local_win {
                "not_local_win"
            } else {
                commit_eligible = true;
                committed = true;
                if choice.source.is_some() {
                    "replaced_candidate"
                } else {
                    "committed"
                }
            }
        }
    };
    emit_near_tie_probe_trace(
        trace_context,
        candidate_n_clusters,
        largest_fraction,
        singleton_weight_fraction,
        candidate_delta_q,
        baseline_candidate_delta_q,
        standard_largest_fraction,
        is_valid,
        quality_passes,
        local_win,
        config.adaptive_near_tie_probe_mode,
        commit_eligible,
        committed,
        commit_block_reason,
        summary,
    );
    if committed {
        choice.assignments = candidate_assignments;
        choice.counts = candidate_counts;
        choice.n_clusters = candidate_n_clusters;
        choice.largest_fraction = largest_fraction;
        choice.singleton_weight_fraction = singleton_weight_fraction;
        choice.quotient_score = 0.0;
        choice.candidate_delta_q = candidate_delta_q;
        choice.source = Some(RefinementCandidateSource::NearTieRefinementProbe);
        choice.trace_candidate_id = None;
        choice.baseline_repair_merge_count = 0;
        choice.baseline_repair_delta_sum = 0.0;
        choice.adaptive_probe_committed = true;
        choice.adaptive_probe_score = 0.0;
        choice.adaptive_probe_source_label = Some("near_tie_refinement_probe");
    }
}

#[allow(clippy::too_many_arguments)]
fn maybe_apply_adaptive_probe_candidate(
    choice: &mut ParentRefinementChoice,
    candidate_assignments: Vec<u32>,
    candidate_counts: Vec<u32>,
    candidate_n_clusters: usize,
    largest_fraction: f64,
    singleton_weight_fraction: f64,
    candidate_delta_q: f64,
    standard_largest_fraction: f64,
    config: &DongdaemunRefinementConfig,
    trace_context: CandidateTraceContext,
    probe_source: &str,
    source_index: usize,
) {
    let is_valid = parent_candidate_is_valid(
        candidate_n_clusters,
        largest_fraction,
        singleton_weight_fraction,
        standard_largest_fraction,
        config,
    );
    let quality_passes = candidate_quality_passes(candidate_delta_q, config);
    let uses_staged_probe_baseline = config.adaptive_probe_mode
        == AdaptiveProbeMode::ConservativeApply
        && config.adaptive_probe_commit_strategy != AdaptiveProbeCommitStrategy::OnlineFirst;
    let baseline_candidate_delta_q = if uses_staged_probe_baseline {
        match choice.adaptive_probe_baseline_delta_q {
            Some(value) => value,
            None => {
                choice.adaptive_probe_baseline_delta_q = Some(choice.candidate_delta_q);
                choice.candidate_delta_q
            }
        }
    } else {
        choice.candidate_delta_q
    };
    let tolerance_delta_q =
        config.adaptive_probe_tolerance_parent_weight * trace_context.parent_weight.max(1.0);
    let local_win = is_valid
        && quality_passes
        && candidate_delta_q > baseline_candidate_delta_q + tolerance_delta_q;
    let gain_vs_baseline = candidate_delta_q - baseline_candidate_delta_q;
    let commit_gain_parent_weight = gain_vs_baseline / trace_context.parent_weight.max(1.0);
    let commit_strategy_score = adaptive_probe_commit_strategy_score(
        config.adaptive_probe_commit_strategy,
        gain_vs_baseline,
        commit_gain_parent_weight,
        largest_fraction,
        singleton_weight_fraction,
        standard_largest_fraction,
    );
    let (commit_count_total_before, commit_count_depth_before) =
        adaptive_probe_commit_counts(trace_context.depth);
    let mut commit_eligible = false;
    let mut committed = false;
    let mut commit_block_reason = "not_local_win";
    let mut replacing_existing_commit = false;
    match config.adaptive_probe_mode {
        AdaptiveProbeMode::Off => {}
        AdaptiveProbeMode::TraceOnly => {
            commit_block_reason = "trace_only";
        }
        AdaptiveProbeMode::ApplyIfWin => {
            if local_win {
                commit_eligible = true;
                committed = true;
                commit_block_reason = "committed";
            }
        }
        AdaptiveProbeMode::ConservativeApply => {
            let allow_replacement = config.adaptive_probe_commit_strategy
                != AdaptiveProbeCommitStrategy::OnlineFirst
                && choice.adaptive_probe_committed;
            if !local_win {
                commit_block_reason = "not_local_win";
            } else if !adaptive_probe_source_is_allowed(config, probe_source) {
                commit_block_reason = "source_filtered";
            } else if commit_gain_parent_weight
                < config.adaptive_probe_commit_min_gain_parent_weight
            {
                commit_block_reason = "below_margin";
            } else if !allow_replacement
                && config.adaptive_probe_max_commits_total > 0
                && commit_count_total_before >= config.adaptive_probe_max_commits_total
            {
                commit_block_reason = "total_budget_exhausted";
            } else if !allow_replacement
                && config.adaptive_probe_max_commits_per_depth > 0
                && commit_count_depth_before >= config.adaptive_probe_max_commits_per_depth
            {
                commit_block_reason = "depth_budget_exhausted";
            } else if allow_replacement && commit_strategy_score <= choice.adaptive_probe_score {
                commit_block_reason = "lower_strategy_score";
            } else {
                commit_eligible = true;
                committed = true;
                replacing_existing_commit = allow_replacement;
                commit_block_reason = if replacing_existing_commit {
                    "replaced_commit"
                } else {
                    "committed"
                };
            }
        }
    }
    emit_adaptive_probe_trace(
        trace_context,
        probe_source,
        source_index,
        candidate_n_clusters,
        largest_fraction,
        singleton_weight_fraction,
        candidate_delta_q,
        baseline_candidate_delta_q,
        tolerance_delta_q,
        standard_largest_fraction,
        is_valid,
        quality_passes,
        local_win,
        config.adaptive_probe_mode,
        commit_eligible,
        committed,
        commit_block_reason,
        commit_gain_parent_weight,
        commit_count_total_before,
        commit_count_depth_before,
        config.adaptive_probe_commit_strategy,
        commit_strategy_score,
    );
    if committed {
        if config.adaptive_probe_mode == AdaptiveProbeMode::ConservativeApply
            && !replacing_existing_commit
        {
            record_adaptive_probe_commit(trace_context.depth);
        }
        choice.assignments = candidate_assignments;
        choice.counts = candidate_counts;
        choice.n_clusters = candidate_n_clusters;
        choice.largest_fraction = largest_fraction;
        choice.singleton_weight_fraction = singleton_weight_fraction;
        choice.quotient_score = 0.0;
        choice.candidate_delta_q = candidate_delta_q;
        choice.source = Some(RefinementCandidateSource::SameGammaSeed);
        choice.trace_candidate_id = None;
        choice.baseline_repair_merge_count = 0;
        choice.baseline_repair_delta_sum = 0.0;
        choice.adaptive_probe_committed =
            config.adaptive_probe_mode == AdaptiveProbeMode::ConservativeApply;
        choice.adaptive_probe_score = commit_strategy_score;
        choice.adaptive_probe_source_label = adaptive_probe_source_label(probe_source);
    }
}

#[cfg(test)]
fn consider_parent_candidate(
    choice: &mut ParentRefinementChoice,
    candidate_assignments: Vec<u32>,
    candidate_counts: Vec<u32>,
    candidate_n_clusters: usize,
    largest_fraction: f64,
    singleton_weight_fraction: f64,
    quotient_score: Option<f64>,
    baseline_repair_merge_count: usize,
    baseline_repair_delta_sum: f64,
    candidate_delta_q: f64,
    standard_largest_fraction: f64,
    source: RefinementCandidateSource,
    config: &DongdaemunRefinementConfig,
    stats: &mut RefinementDongdaemunStats,
) {
    consider_parent_candidate_with_trace(
        choice,
        candidate_assignments,
        candidate_counts,
        candidate_n_clusters,
        largest_fraction,
        singleton_weight_fraction,
        quotient_score,
        baseline_repair_merge_count,
        baseline_repair_delta_sum,
        candidate_delta_q,
        standard_largest_fraction,
        source,
        config,
        stats,
        CandidateTraceContext::default(),
    );
}

fn consider_parent_candidate_with_trace(
    choice: &mut ParentRefinementChoice,
    candidate_assignments: Vec<u32>,
    candidate_counts: Vec<u32>,
    candidate_n_clusters: usize,
    largest_fraction: f64,
    singleton_weight_fraction: f64,
    quotient_score: Option<f64>,
    baseline_repair_merge_count: usize,
    baseline_repair_delta_sum: f64,
    candidate_delta_q: f64,
    standard_largest_fraction: f64,
    source: RefinementCandidateSource,
    config: &DongdaemunRefinementConfig,
    stats: &mut RefinementDongdaemunStats,
    trace_context: CandidateTraceContext,
) {
    let candidate_trace_id = stats.next_candidate_trace_id();
    stats.record_candidate(source);
    stats.record_candidate_quality(source, candidate_delta_q);
    let quadrant = refinement_candidate_quadrant(
        candidate_delta_q,
        largest_fraction,
        standard_largest_fraction,
    );
    stats.record_candidate_quadrant(source, quadrant);
    let is_valid = parent_candidate_is_valid(
        candidate_n_clusters,
        largest_fraction,
        singleton_weight_fraction,
        standard_largest_fraction,
        config,
    );
    stats.record_candidate_validity(source, is_valid);
    if !is_valid {
        emit_candidate_profile_trace(
            trace_context,
            candidate_trace_id,
            source,
            quadrant,
            candidate_n_clusters,
            largest_fraction,
            singleton_weight_fraction,
            quotient_score,
            baseline_repair_merge_count,
            baseline_repair_delta_sum,
            candidate_delta_q,
            standard_largest_fraction,
            is_valid,
            false,
            "invalid",
            config.target_max_weight,
            config.adaptive_plateau_quality_band,
            false,
        );
        stats.rejected_candidates += 1;
        return;
    }
    if !candidate_quality_passes(candidate_delta_q, config) {
        emit_candidate_profile_trace(
            trace_context,
            candidate_trace_id,
            source,
            quadrant,
            candidate_n_clusters,
            largest_fraction,
            singleton_weight_fraction,
            quotient_score,
            baseline_repair_merge_count,
            baseline_repair_delta_sum,
            candidate_delta_q,
            standard_largest_fraction,
            is_valid,
            false,
            "quality_rejected",
            config.target_max_weight,
            config.adaptive_plateau_quality_band,
            false,
        );
        stats.record_quality_rejected(source);
        stats.record_candidate_rejected_decision(quadrant);
        stats.rejected_candidates += 1;
        return;
    }
    let quotient_score = if config.use_quotient_diagnostic {
        let score = quotient_score.unwrap_or(0.0);
        stats.record_quotient_candidate(score);
        score
    } else {
        0.0
    };
    let plateau_compared = adaptive_plateau_compared(choice, candidate_delta_q, config);
    if is_valid
        && candidate_is_better_by_policy(
            choice,
            largest_fraction,
            singleton_weight_fraction,
            candidate_n_clusters,
            quotient_score,
            candidate_delta_q,
            standard_largest_fraction,
            trace_context.parent_weight,
            config,
        )
    {
        if let Some(previous_source) = choice.source {
            let previous_quadrant = refinement_candidate_quadrant(
                choice.candidate_delta_q,
                choice.largest_fraction,
                standard_largest_fraction,
            );
            stats.record_policy_rejected(previous_source, previous_quadrant);
            emit_candidate_decision_trace(
                trace_context,
                choice.trace_candidate_id,
                "superseded_by_policy",
            );
        }
        emit_candidate_profile_trace(
            trace_context,
            candidate_trace_id,
            source,
            quadrant,
            candidate_n_clusters,
            largest_fraction,
            singleton_weight_fraction,
            Some(quotient_score),
            baseline_repair_merge_count,
            baseline_repair_delta_sum,
            candidate_delta_q,
            standard_largest_fraction,
            is_valid,
            true,
            "selected_by_policy",
            config.target_max_weight,
            config.adaptive_plateau_quality_band,
            plateau_compared,
        );
        choice.assignments = candidate_assignments;
        choice.counts = candidate_counts;
        choice.n_clusters = candidate_n_clusters;
        choice.largest_fraction = largest_fraction;
        choice.singleton_weight_fraction = singleton_weight_fraction;
        choice.quotient_score = quotient_score;
        choice.candidate_delta_q = candidate_delta_q;
        choice.source = Some(source);
        choice.trace_candidate_id = Some(candidate_trace_id);
        choice.baseline_repair_merge_count = baseline_repair_merge_count;
        choice.baseline_repair_delta_sum = baseline_repair_delta_sum;
    } else {
        emit_candidate_profile_trace(
            trace_context,
            candidate_trace_id,
            source,
            quadrant,
            candidate_n_clusters,
            largest_fraction,
            singleton_weight_fraction,
            Some(quotient_score),
            baseline_repair_merge_count,
            baseline_repair_delta_sum,
            candidate_delta_q,
            standard_largest_fraction,
            is_valid,
            true,
            "policy_rejected",
            config.target_max_weight,
            config.adaptive_plateau_quality_band,
            plateau_compared,
        );
        stats.record_policy_rejected(source, quadrant);
        stats.rejected_candidates += 1;
    }
}

fn record_dongdaemun_refinement_stats(
    depth: usize,
    stats: RefinementDongdaemunStats,
    final_refined_clusters: usize,
    audit: Option<&mut DongdaemunRefinementAudit>,
) {
    if let Some(audit) = audit {
        audit.selected_parent_count_total += stats.selected_parents;
        audit.applied_parent_count_total += stats.applied_parents;
        audit.rejected_candidate_count_total += stats.rejected_candidates;
        audit.added_refined_clusters_total += stats.added_refined_clusters;
        audit.same_gamma_candidates_total += stats.same_gamma_candidates;
        audit.high_gamma_candidates_total += stats.high_gamma_candidates;
        audit.same_gamma_applied_total += stats.same_gamma_applied;
        audit.high_gamma_applied_total += stats.high_gamma_applied;
        audit.quotient_candidates_total += stats.quotient_candidates;
        audit.quotient_positive_candidates_total += stats.quotient_positive_candidates;
        audit.quotient_selected_total += stats.quotient_selected;
        audit.quotient_score_sum += stats.quotient_score_sum;
        audit.baseline_repair_candidates_total += stats.baseline_repair_candidates;
        audit.baseline_repair_improved_candidates_total +=
            stats.baseline_repair_improved_candidates;
        audit.baseline_repair_selected_total += stats.baseline_repair_selected;
        audit.baseline_repair_merge_count_total += stats.baseline_repair_merge_count;
        audit.baseline_repair_delta_sum += stats.baseline_repair_delta_sum;
        audit.candidate_quality_delta_sum += stats.candidate_quality_delta_sum;
        audit.candidate_positive_quality_delta_total += stats.candidate_positive_quality_delta;
        audit.candidate_selected_positive_quality_delta_total +=
            stats.candidate_selected_positive_quality_delta;
        audit.candidate_rejected_by_quality_total += stats.candidate_rejected_by_quality;
        audit.same_gamma_quality_delta_sum += stats.same_gamma_quality_delta_sum;
        audit.high_gamma_quality_delta_sum += stats.high_gamma_quality_delta_sum;
        audit.same_gamma_positive_quality_delta_total += stats.same_gamma_positive_quality_delta;
        audit.high_gamma_positive_quality_delta_total += stats.high_gamma_positive_quality_delta;
        audit.same_gamma_selected_positive_quality_delta_total +=
            stats.same_gamma_selected_positive_quality_delta;
        audit.high_gamma_selected_positive_quality_delta_total +=
            stats.high_gamma_selected_positive_quality_delta;
        audit.same_gamma_rejected_by_quality_total += stats.same_gamma_rejected_by_quality;
        audit.high_gamma_rejected_by_quality_total += stats.high_gamma_rejected_by_quality;
        audit.candidate_valid_total += stats.candidate_valid;
        audit.candidate_invalid_total += stats.candidate_invalid;
        audit.candidate_rejected_by_policy_total += stats.candidate_rejected_by_policy;
        audit.same_gamma_valid_total += stats.same_gamma_valid;
        audit.high_gamma_valid_total += stats.high_gamma_valid;
        audit.same_gamma_invalid_total += stats.same_gamma_invalid;
        audit.high_gamma_invalid_total += stats.high_gamma_invalid;
        audit.same_gamma_rejected_by_policy_total += stats.same_gamma_rejected_by_policy;
        audit.high_gamma_rejected_by_policy_total += stats.high_gamma_rejected_by_policy;
        audit.candidate_qpos_spos_total += stats.candidate_qpos_spos;
        audit.candidate_qpos_sneg_total += stats.candidate_qpos_sneg;
        audit.candidate_qneg_spos_total += stats.candidate_qneg_spos;
        audit.candidate_qneg_sneg_total += stats.candidate_qneg_sneg;
        audit.same_gamma_qpos_spos_total += stats.same_gamma_qpos_spos;
        audit.same_gamma_qpos_sneg_total += stats.same_gamma_qpos_sneg;
        audit.same_gamma_qneg_spos_total += stats.same_gamma_qneg_spos;
        audit.same_gamma_qneg_sneg_total += stats.same_gamma_qneg_sneg;
        audit.high_gamma_qpos_spos_total += stats.high_gamma_qpos_spos;
        audit.high_gamma_qpos_sneg_total += stats.high_gamma_qpos_sneg;
        audit.high_gamma_qneg_spos_total += stats.high_gamma_qneg_spos;
        audit.high_gamma_qneg_sneg_total += stats.high_gamma_qneg_sneg;
        audit.candidate_true_positive_total += stats.candidate_true_positive;
        audit.candidate_false_positive_total += stats.candidate_false_positive;
        audit.candidate_false_negative_total += stats.candidate_false_negative;
        audit.candidate_true_negative_total += stats.candidate_true_negative;
        audit.adaptive_local_shake_triggers_total += stats.adaptive_local_shake_triggers;
        audit.adaptive_local_shake_candidates_total += stats.adaptive_local_shake_candidates;
        audit.adaptive_local_shake_commits_total += stats.adaptive_local_shake_commits;
        audit.adaptive_local_shake_qf_gain_sum += stats.adaptive_local_shake_qf_gain_sum;
        audit.max_parent_weight_seen = audit
            .max_parent_weight_seen
            .max(stats.max_parent_weight_seen);
        audit.iterations.push(DongdaemunRefinementIterationAudit {
            depth,
            selected_parents: stats.selected_parents,
            applied_parents: stats.applied_parents,
            same_gamma_candidates: stats.same_gamma_candidates,
            high_gamma_candidates: stats.high_gamma_candidates,
            same_gamma_applied: stats.same_gamma_applied,
            high_gamma_applied: stats.high_gamma_applied,
            quotient_candidates: stats.quotient_candidates,
            quotient_positive_candidates: stats.quotient_positive_candidates,
            quotient_selected: stats.quotient_selected,
            quotient_score_sum: stats.quotient_score_sum,
            baseline_repair_candidates: stats.baseline_repair_candidates,
            baseline_repair_improved_candidates: stats.baseline_repair_improved_candidates,
            baseline_repair_selected: stats.baseline_repair_selected,
            baseline_repair_merge_count: stats.baseline_repair_merge_count,
            baseline_repair_delta_sum: stats.baseline_repair_delta_sum,
            candidate_quality_delta_sum: stats.candidate_quality_delta_sum,
            candidate_positive_quality_delta: stats.candidate_positive_quality_delta,
            candidate_selected_positive_quality_delta: stats
                .candidate_selected_positive_quality_delta,
            candidate_rejected_by_quality: stats.candidate_rejected_by_quality,
            same_gamma_quality_delta_sum: stats.same_gamma_quality_delta_sum,
            high_gamma_quality_delta_sum: stats.high_gamma_quality_delta_sum,
            same_gamma_positive_quality_delta: stats.same_gamma_positive_quality_delta,
            high_gamma_positive_quality_delta: stats.high_gamma_positive_quality_delta,
            same_gamma_selected_positive_quality_delta: stats
                .same_gamma_selected_positive_quality_delta,
            high_gamma_selected_positive_quality_delta: stats
                .high_gamma_selected_positive_quality_delta,
            same_gamma_rejected_by_quality: stats.same_gamma_rejected_by_quality,
            high_gamma_rejected_by_quality: stats.high_gamma_rejected_by_quality,
            candidate_valid: stats.candidate_valid,
            candidate_invalid: stats.candidate_invalid,
            candidate_rejected_by_policy: stats.candidate_rejected_by_policy,
            same_gamma_valid: stats.same_gamma_valid,
            high_gamma_valid: stats.high_gamma_valid,
            same_gamma_invalid: stats.same_gamma_invalid,
            high_gamma_invalid: stats.high_gamma_invalid,
            same_gamma_rejected_by_policy: stats.same_gamma_rejected_by_policy,
            high_gamma_rejected_by_policy: stats.high_gamma_rejected_by_policy,
            candidate_qpos_spos: stats.candidate_qpos_spos,
            candidate_qpos_sneg: stats.candidate_qpos_sneg,
            candidate_qneg_spos: stats.candidate_qneg_spos,
            candidate_qneg_sneg: stats.candidate_qneg_sneg,
            same_gamma_qpos_spos: stats.same_gamma_qpos_spos,
            same_gamma_qpos_sneg: stats.same_gamma_qpos_sneg,
            same_gamma_qneg_spos: stats.same_gamma_qneg_spos,
            same_gamma_qneg_sneg: stats.same_gamma_qneg_sneg,
            high_gamma_qpos_spos: stats.high_gamma_qpos_spos,
            high_gamma_qpos_sneg: stats.high_gamma_qpos_sneg,
            high_gamma_qneg_spos: stats.high_gamma_qneg_spos,
            high_gamma_qneg_sneg: stats.high_gamma_qneg_sneg,
            candidate_true_positive: stats.candidate_true_positive,
            candidate_false_positive: stats.candidate_false_positive,
            candidate_false_negative: stats.candidate_false_negative,
            candidate_true_negative: stats.candidate_true_negative,
            adaptive_local_shake_triggers: stats.adaptive_local_shake_triggers,
            adaptive_local_shake_candidates: stats.adaptive_local_shake_candidates,
            adaptive_local_shake_commits: stats.adaptive_local_shake_commits,
            adaptive_local_shake_qf_gain_sum: stats.adaptive_local_shake_qf_gain_sum,
            standard_refined_clusters: stats.standard_refined_clusters,
            final_refined_clusters,
        });
    }
}

fn finish_iteration_step(
    step: IterationStep,
    clustering: &mut Clustering,
    config: &LeidenConfig,
    dongdaemun: Option<&DongdaemunRefinementConfig>,
    mut audit: Option<&mut DongdaemunRefinementAudit>,
    randomness: f64,
    rng: &mut impl Rng,
    ws: &mut Workspace,
    depth: usize,
    iteration: usize,
) -> IterationStats {
    match step {
        IterationStep::Done { stats } => stats,
        IterationStep::NonRefined {
            local_stats,
            reduced,
            mut reduced_clustering,
            parent_nodes,
            reduced_nodes,
            trace_detail,
        } => {
            let t_recurse = Instant::now();
            let recursive_stats = improve_one_iteration_owned(
                reduced,
                &mut reduced_clustering,
                config,
                dongdaemun,
                audit.as_deref_mut(),
                randomness,
                rng,
                ws,
                depth + 1,
                iteration,
            );
            if trace_detail {
                trace::emit(format_args!(
                    "phase=recurse_non_refined depth={} reduced_nodes={} improved={} moved_nodes={} elapsed_ms={:.1}",
                    depth,
                    reduced_nodes,
                    recursive_stats.improved,
                    recursive_stats.moved_nodes,
                    t_recurse.elapsed().as_secs_f64() * 1000.0,
                ));
            }
            if recursive_stats.improved {
                let t_merge = Instant::now();
                clustering.merge_clusters(&reduced_clustering);
                if trace_detail {
                    trace::emit(format_args!(
                        "phase=merge_non_refined depth={} nodes={} elapsed_ms={:.1}",
                        depth,
                        parent_nodes,
                        t_merge.elapsed().as_secs_f64() * 1000.0,
                    ));
                }
            }
            IterationStats {
                improved: local_stats.improved | recursive_stats.improved,
                moved_nodes: local_stats
                    .moved_nodes
                    .saturating_add(recursive_stats.moved_nodes),
            }
        }
        IterationStep::Refined {
            local_stats,
            reduced,
            mut reduced_clustering,
            refinement_clustering,
            parent_nodes,
            reduced_nodes,
            trace_detail,
        } => {
            let t_recurse = Instant::now();
            let recursive_stats = improve_one_iteration_owned(
                reduced,
                &mut reduced_clustering,
                config,
                dongdaemun,
                audit.as_deref_mut(),
                randomness,
                rng,
                ws,
                depth + 1,
                iteration,
            );
            if trace_detail {
                trace::emit(format_args!(
                    "phase=recurse_refined depth={} reduced_nodes={} improved={} moved_nodes={} elapsed_ms={:.1}",
                    depth,
                    reduced_nodes,
                    recursive_stats.improved,
                    recursive_stats.moved_nodes,
                    t_recurse.elapsed().as_secs_f64() * 1000.0,
                ));
            }

            // Merge back only if the recursive reduced graph actually changed.
            // When recursion reports no improvement, `reduced_clustering` is
            // equivalent to the initial projection and merge-back would only
            // restore the move-phase clustering after a full O(n) scan.
            if recursive_stats.improved {
                let t_merge = Instant::now();
                clustering.clusters = refinement_clustering.clusters;
                clustering.n_clusters = refinement_clustering.n_clusters;
                clustering.merge_clusters(&reduced_clustering);
                if trace_detail {
                    trace::emit(format_args!(
                        "phase=merge_refined depth={} nodes={} elapsed_ms={:.1}",
                        depth,
                        parent_nodes,
                        t_merge.elapsed().as_secs_f64() * 1000.0,
                    ));
                }
            }

            IterationStats {
                improved: local_stats.improved | recursive_stats.improved,
                moved_nodes: local_stats
                    .moved_nodes
                    .saturating_add(recursive_stats.moved_nodes),
            }
        }
    }
}

fn refine_eager(
    graph: &Graph,
    clustering: &Clustering,
    nodes_per_cluster: &[Vec<usize>],
    config: &LeidenConfig,
    dongdaemun: Option<&DongdaemunRefinementConfig>,
    parent_weights: Option<&[f64]>,
    randomness: f64,
    depth: usize,
    iteration: usize,
    rng: &mut impl Rng,
) -> RefinementResult {
    // Extract all cluster subnetworks in one O(n+m) pass. Fast for small and
    // medium graphs, but memory-heavy at large scale.
    let subnetworks = graph.create_subnetworks(nodes_per_cluster);
    let mut refinement = empty_refinement(graph.n_nodes);
    let mut parent_clusters = Vec::with_capacity(nodes_per_cluster.len());
    let mut cluster_counts = Vec::with_capacity(nodes_per_cluster.len());
    let mut reduced_fixed = clustering
        .fixed
        .as_ref()
        .map(|_| Vec::with_capacity(nodes_per_cluster.len()));
    let mut merge_ws = local_merge::LocalMergeWorkspace::new(0);
    let fixed = clustering.fixed.as_deref();
    let mut ddm_stats = RefinementDongdaemunStats::default();
    let selected = if let (Some(config), Some(parent_weights)) = (dongdaemun, parent_weights) {
        let boundary_pressure = parent_boundary_pressure_eager(
            graph,
            clustering,
            nodes_per_cluster,
            parent_weights,
            config,
        );
        let (selected, selected_count, max_parent_weight_seen) =
            select_extra_refinement_parents(parent_weights, config, boundary_pressure.as_deref());
        ddm_stats.selected_parents = selected_count;
        ddm_stats.max_parent_weight_seen = max_parent_weight_seen;
        selected
    } else {
        Vec::new()
    };

    for (cid, (subgraph, nodes)) in subnetworks.iter().enumerate() {
        if nodes.is_empty() {
            continue;
        }

        if let Some(fixed) = fixed {
            if nodes.iter().any(|&n| fixed[n]) {
                for &node in nodes {
                    refinement.clusters[node] = refinement.n_clusters as u32;
                }
                parent_clusters.push(cid as u32);
                cluster_counts.push(nodes.len() as u32);
                if let Some(rf) = reduced_fixed.as_mut() {
                    rf.push(true);
                }
                refinement.n_clusters += 1;
                ddm_stats.standard_refined_clusters += 1;
                continue;
            }
        }

        {
            let parent_weight = parent_weights
                .and_then(|weights| weights.get(cid).copied())
                .unwrap_or_else(|| nodes.iter().map(|&node| graph.node_weights[node]).sum());
            let parent_visit_index = if selected.get(cid).copied().unwrap_or(false) {
                next_adaptive_probe_visit(depth, cid)
            } else {
                0
            };
            let mut standard_counts = Vec::new();
            let trace_merge_margins = trace::ddm_trajectory_trace_enabled()
                || dongdaemun.is_some_and(adaptive_local_shake_enabled);
            let (standard_n_clusters, standard_margin_summary) = if trace_merge_margins {
                let (n_clusters, summary) =
                    local_merge::find_clustering_with_workspace_assignments_and_append_sizes_traced(
                        subgraph,
                        config.resolution,
                        randomness,
                        rng,
                        &mut merge_ws,
                        &mut standard_counts,
                        local_merge_low_margin_threshold(dongdaemun, parent_weight),
                        None,
                    );
                (n_clusters, Some(summary))
            } else {
                let n_clusters =
                    local_merge::find_clustering_with_workspace_assignments_and_append_sizes(
                        subgraph,
                        config.resolution,
                        randomness,
                        rng,
                        &mut merge_ws,
                        &mut standard_counts,
                    );
                (n_clusters, None)
            };
            let standard_assignments = merge_ws.assignments()[..nodes.len()].to_vec();
            let (standard_largest_fraction, standard_singleton_weight_fraction) =
                parent_partition_summary(
                    nodes.len(),
                    standard_n_clusters,
                    &standard_assignments,
                    parent_weight,
                    |local| graph.node_weights[nodes[local]],
                );
            let standard_quality = parent_partition_quality_subgraph(
                subgraph,
                &standard_assignments,
                standard_n_clusters,
                config.resolution,
            );
            if let Some(summary) = &standard_margin_summary {
                emit_local_merge_margin_summary_trace(
                    CandidateTraceContext {
                        depth,
                        parent_id: cid,
                        parent_visit_index,
                        parent_size: nodes.len(),
                        parent_weight,
                        standard_n_clusters,
                        source_index: 0,
                        gamma_multiplier: 1.0,
                        repaired: false,
                    },
                    iteration,
                    "standard_refinement",
                    standard_n_clusters,
                    standard_largest_fraction,
                    summary,
                );
            }

            let mut choice = ParentRefinementChoice {
                assignments: standard_assignments,
                counts: standard_counts,
                n_clusters: standard_n_clusters,
                largest_fraction: standard_largest_fraction,
                singleton_weight_fraction: standard_singleton_weight_fraction,
                quotient_score: 0.0,
                candidate_delta_q: 0.0,
                source: None,
                trace_candidate_id: None,
                baseline_repair_merge_count: 0,
                baseline_repair_delta_sum: 0.0,
                adaptive_probe_baseline_delta_q: None,
                adaptive_probe_committed: false,
                adaptive_probe_score: f64::NEG_INFINITY,
                adaptive_probe_source_label: None,
            };

            if selected.get(cid).copied().unwrap_or(false) {
                if let Some(ddm_config) = dongdaemun {
                    for perturbation_index in 0..ddm_config.seed_perturbations {
                        let mut candidate_counts = Vec::new();
                        let derived_seed =
                            derive_same_gamma_seed(config.seed, depth, cid, perturbation_index);
                        let mut candidate_rng = rand::rngs::StdRng::seed_from_u64(derived_seed);
                        let candidate_n_clusters = local_merge::find_clustering_with_workspace_assignments_and_append_sizes(
                            subgraph,
                            config.resolution,
                            randomness,
                            &mut candidate_rng,
                            &mut merge_ws,
                            &mut candidate_counts,
                        );
                        let candidate_assignments = merge_ws.assignments()[..nodes.len()].to_vec();
                        let candidate_delta_q = parent_partition_quality_subgraph(
                            subgraph,
                            &candidate_assignments,
                            candidate_n_clusters,
                            config.resolution,
                        ) - standard_quality;
                        let (largest_fraction, singleton_weight_fraction) =
                            parent_partition_summary(
                                nodes.len(),
                                candidate_n_clusters,
                                &candidate_assignments,
                                parent_weight,
                                |local| graph.node_weights[nodes[local]],
                            );
                        let quotient_score = if ddm_config.use_quotient_diagnostic
                            && nodes.len() > 1
                            && parent_candidate_is_valid(
                                candidate_n_clusters,
                                largest_fraction,
                                singleton_weight_fraction,
                                standard_largest_fraction,
                                ddm_config,
                            ) {
                            parent_weights.map(|parent_weights| {
                                parent_candidate_quotient_score(
                                    graph,
                                    clustering,
                                    cid,
                                    nodes.len(),
                                    &candidate_assignments,
                                    candidate_n_clusters,
                                    parent_weight,
                                    parent_weights,
                                    config.resolution,
                                    |local| nodes[local],
                                )
                            })
                        } else {
                            None
                        };
                        consider_parent_candidate_with_trace(
                            &mut choice,
                            candidate_assignments,
                            candidate_counts,
                            candidate_n_clusters,
                            largest_fraction,
                            singleton_weight_fraction,
                            quotient_score,
                            0,
                            0.0,
                            candidate_delta_q,
                            standard_largest_fraction,
                            RefinementCandidateSource::SameGammaSeed,
                            ddm_config,
                            &mut ddm_stats,
                            CandidateTraceContext {
                                depth,
                                parent_id: cid,
                                parent_visit_index,
                                parent_size: nodes.len(),
                                parent_weight,
                                standard_n_clusters,
                                source_index: perturbation_index,
                                gamma_multiplier: 1.0,
                                repaired: false,
                            },
                        );
                    }
                    for (multiplier_index, multiplier) in
                        ddm_config.gamma_multipliers.iter().enumerate()
                    {
                        let mut candidate_counts = Vec::new();
                        let derived_seed =
                            derive_high_gamma_seed(config.seed, depth, cid, multiplier_index);
                        let mut candidate_rng = rand::rngs::StdRng::seed_from_u64(derived_seed);
                        let candidate_n_clusters = local_merge::find_clustering_with_workspace_assignments_and_append_sizes(
                            subgraph,
                            config.resolution * *multiplier,
                            randomness,
                            &mut candidate_rng,
                            &mut merge_ws,
                            &mut candidate_counts,
                        );
                        let mut candidate_assignments =
                            merge_ws.assignments()[..nodes.len()].to_vec();
                        let mut candidate_counts = candidate_counts;
                        let mut candidate_n_clusters = candidate_n_clusters;
                        let mut baseline_repair_merge_count = 0usize;
                        let mut baseline_repair_delta_sum = 0.0;
                        let mut consider_repaired_candidate = true;
                        let repair_policy =
                            effective_baseline_repair_policy(ddm_config, parent_weight);
                        if ddm_config.use_baseline_repair
                            && nodes.len() > 1
                            && candidate_n_clusters > 1
                        {
                            if repair_policy == BaselineRepairPolicy::Augment {
                                let (largest_fraction, singleton_weight_fraction) =
                                    parent_partition_summary(
                                        nodes.len(),
                                        candidate_n_clusters,
                                        &candidate_assignments,
                                        parent_weight,
                                        |local| graph.node_weights[nodes[local]],
                                    );
                                let candidate_delta_q = parent_partition_quality_subgraph(
                                    subgraph,
                                    &candidate_assignments,
                                    candidate_n_clusters,
                                    config.resolution,
                                ) - standard_quality;
                                let quotient_score = if ddm_config.use_quotient_diagnostic
                                    && nodes.len() > 1
                                    && parent_candidate_is_valid(
                                        candidate_n_clusters,
                                        largest_fraction,
                                        singleton_weight_fraction,
                                        standard_largest_fraction,
                                        ddm_config,
                                    ) {
                                    parent_weights.map(|parent_weights| {
                                        parent_candidate_quotient_score(
                                            graph,
                                            clustering,
                                            cid,
                                            nodes.len(),
                                            &candidate_assignments,
                                            candidate_n_clusters,
                                            parent_weight,
                                            parent_weights,
                                            config.resolution,
                                            |local| nodes[local],
                                        )
                                    })
                                } else {
                                    None
                                };
                                consider_parent_candidate_with_trace(
                                    &mut choice,
                                    candidate_assignments.clone(),
                                    candidate_counts.clone(),
                                    candidate_n_clusters,
                                    largest_fraction,
                                    singleton_weight_fraction,
                                    quotient_score,
                                    0,
                                    0.0,
                                    candidate_delta_q,
                                    standard_largest_fraction,
                                    RefinementCandidateSource::HighGamma,
                                    ddm_config,
                                    &mut ddm_stats,
                                    CandidateTraceContext {
                                        depth,
                                        parent_id: cid,
                                        parent_visit_index,
                                        parent_size: nodes.len(),
                                        parent_weight,
                                        standard_n_clusters,
                                        source_index: multiplier_index,
                                        gamma_multiplier: *multiplier,
                                        repaired: false,
                                    },
                                );
                            }
                            let repair = repair_parent_internal_candidate(
                                graph,
                                nodes.len(),
                                &candidate_assignments,
                                candidate_n_clusters,
                                config.resolution,
                                ddm_config.baseline_repair_epsilon,
                                |local| nodes[local],
                            );
                            ddm_stats.record_baseline_repair_candidate(&repair);
                            if repair_policy == BaselineRepairPolicy::Augment && !repair.changed {
                                consider_repaired_candidate = false;
                            }
                            baseline_repair_merge_count = repair.merge_count;
                            baseline_repair_delta_sum = repair.delta_sum;
                            candidate_assignments = repair.assignments;
                            candidate_counts = repair.counts;
                            candidate_n_clusters = repair.n_clusters;
                        }
                        if consider_repaired_candidate {
                            let (largest_fraction, singleton_weight_fraction) =
                                parent_partition_summary(
                                    nodes.len(),
                                    candidate_n_clusters,
                                    &candidate_assignments,
                                    parent_weight,
                                    |local| graph.node_weights[nodes[local]],
                                );
                            let candidate_delta_q = parent_partition_quality_subgraph(
                                subgraph,
                                &candidate_assignments,
                                candidate_n_clusters,
                                config.resolution,
                            ) - standard_quality;
                            let quotient_score = if ddm_config.use_quotient_diagnostic
                                && nodes.len() > 1
                                && parent_candidate_is_valid(
                                    candidate_n_clusters,
                                    largest_fraction,
                                    singleton_weight_fraction,
                                    standard_largest_fraction,
                                    ddm_config,
                                ) {
                                parent_weights.map(|parent_weights| {
                                    parent_candidate_quotient_score(
                                        graph,
                                        clustering,
                                        cid,
                                        nodes.len(),
                                        &candidate_assignments,
                                        candidate_n_clusters,
                                        parent_weight,
                                        parent_weights,
                                        config.resolution,
                                        |local| nodes[local],
                                    )
                                })
                            } else {
                                None
                            };
                            consider_parent_candidate_with_trace(
                                &mut choice,
                                candidate_assignments,
                                candidate_counts,
                                candidate_n_clusters,
                                largest_fraction,
                                singleton_weight_fraction,
                                quotient_score,
                                baseline_repair_merge_count,
                                baseline_repair_delta_sum,
                                candidate_delta_q,
                                standard_largest_fraction,
                                RefinementCandidateSource::HighGamma,
                                ddm_config,
                                &mut ddm_stats,
                                CandidateTraceContext {
                                    depth,
                                    parent_id: cid,
                                    parent_visit_index,
                                    parent_size: nodes.len(),
                                    parent_weight,
                                    standard_n_clusters,
                                    source_index: multiplier_index,
                                    gamma_multiplier: *multiplier,
                                    repaired: baseline_repair_merge_count > 0,
                                },
                            );
                        }
                    }
                    if adaptive_probe_should_probe(ddm_config, depth, cid, parent_visit_index) {
                        for perturbation_index in 0..ddm_config.adaptive_probe_perturbations {
                            let mut candidate_counts = Vec::new();
                            let derived_seed = derive_adaptive_probe_seed(
                                config.seed,
                                depth,
                                cid,
                                parent_visit_index,
                                0x5155_414d_455f_4741,
                                perturbation_index,
                            );
                            let mut candidate_rng = rand::rngs::StdRng::seed_from_u64(derived_seed);
                            let candidate_n_clusters = local_merge::find_clustering_with_workspace_assignments_and_append_sizes(
                                subgraph,
                                config.resolution,
                                randomness,
                                &mut candidate_rng,
                                &mut merge_ws,
                                &mut candidate_counts,
                            );
                            let candidate_assignments =
                                merge_ws.assignments()[..nodes.len()].to_vec();
                            let candidate_delta_q = parent_partition_quality_subgraph(
                                subgraph,
                                &candidate_assignments,
                                candidate_n_clusters,
                                config.resolution,
                            ) - standard_quality;
                            let (largest_fraction, singleton_weight_fraction) =
                                parent_partition_summary(
                                    nodes.len(),
                                    candidate_n_clusters,
                                    &candidate_assignments,
                                    parent_weight,
                                    |local| graph.node_weights[nodes[local]],
                                );
                            maybe_apply_adaptive_probe_candidate(
                                &mut choice,
                                candidate_assignments,
                                candidate_counts,
                                candidate_n_clusters,
                                largest_fraction,
                                singleton_weight_fraction,
                                candidate_delta_q,
                                standard_largest_fraction,
                                ddm_config,
                                CandidateTraceContext {
                                    depth,
                                    parent_id: cid,
                                    parent_visit_index,
                                    parent_size: nodes.len(),
                                    parent_weight,
                                    standard_n_clusters,
                                    source_index: perturbation_index,
                                    gamma_multiplier: 1.0,
                                    repaired: false,
                                },
                                "same_gamma_probe",
                                perturbation_index,
                            );
                        }
                        if ddm_config.adaptive_probe_include_node_order_control {
                            for perturbation_index in 0..ddm_config.adaptive_probe_perturbations {
                                let mut candidate_counts = Vec::new();
                                let derived_seed = derive_adaptive_probe_seed(
                                    config.seed,
                                    depth,
                                    cid,
                                    parent_visit_index,
                                    0x4e4f_4445_4f52_4452,
                                    perturbation_index,
                                );
                                let mut candidate_rng =
                                    rand::rngs::StdRng::seed_from_u64(derived_seed);
                                let candidate_n_clusters = local_merge::find_clustering_with_workspace_assignments_and_append_sizes(
                                    subgraph,
                                    config.resolution,
                                    randomness,
                                    &mut candidate_rng,
                                    &mut merge_ws,
                                    &mut candidate_counts,
                                );
                                let candidate_assignments =
                                    merge_ws.assignments()[..nodes.len()].to_vec();
                                let candidate_delta_q = parent_partition_quality_subgraph(
                                    subgraph,
                                    &candidate_assignments,
                                    candidate_n_clusters,
                                    config.resolution,
                                ) - standard_quality;
                                let (largest_fraction, singleton_weight_fraction) =
                                    parent_partition_summary(
                                        nodes.len(),
                                        candidate_n_clusters,
                                        &candidate_assignments,
                                        parent_weight,
                                        |local| graph.node_weights[nodes[local]],
                                    );
                                maybe_apply_adaptive_probe_candidate(
                                    &mut choice,
                                    candidate_assignments,
                                    candidate_counts,
                                    candidate_n_clusters,
                                    largest_fraction,
                                    singleton_weight_fraction,
                                    candidate_delta_q,
                                    standard_largest_fraction,
                                    ddm_config,
                                    CandidateTraceContext {
                                        depth,
                                        parent_id: cid,
                                        parent_visit_index,
                                        parent_size: nodes.len(),
                                        parent_weight,
                                        standard_n_clusters,
                                        source_index: perturbation_index,
                                        gamma_multiplier: 1.0,
                                        repaired: false,
                                    },
                                    "node_order_control",
                                    perturbation_index,
                                );
                            }
                        }
                    }
                    if adaptive_near_tie_probe_enabled(ddm_config) {
                        let mut candidate_counts = Vec::new();
                        let derived_seed = derive_adaptive_probe_seed(
                            config.seed,
                            depth,
                            cid,
                            parent_visit_index,
                            0x4e45_4152_5449_4550,
                            0,
                        );
                        let mut candidate_rng = rand::rngs::StdRng::seed_from_u64(derived_seed);
                        let near_tie = local_merge::NearTieProbeConfig {
                            parent_weight,
                            margin_parent_weight: ddm_config.adaptive_near_tie_margin_parent_weight,
                            randomness: ddm_config.adaptive_near_tie_randomness,
                            max_decisions_per_parent: ddm_config
                                .adaptive_near_tie_max_decisions_per_parent,
                        };
                        let (candidate_n_clusters, summary) =
                            local_merge::find_clustering_with_workspace_assignments_and_append_sizes_traced(
                                subgraph,
                                config.resolution,
                                randomness,
                                &mut candidate_rng,
                                &mut merge_ws,
                                &mut candidate_counts,
                                ddm_config.adaptive_near_tie_margin_parent_weight
                                    * parent_weight.max(0.0),
                                Some(near_tie),
                            );
                        let candidate_assignments = merge_ws.assignments()[..nodes.len()].to_vec();
                        let candidate_delta_q = parent_partition_quality_subgraph(
                            subgraph,
                            &candidate_assignments,
                            candidate_n_clusters,
                            config.resolution,
                        ) - standard_quality;
                        let (largest_fraction, singleton_weight_fraction) =
                            parent_partition_summary(
                                nodes.len(),
                                candidate_n_clusters,
                                &candidate_assignments,
                                parent_weight,
                                |local| graph.node_weights[nodes[local]],
                            );
                        let trace_context = CandidateTraceContext {
                            depth,
                            parent_id: cid,
                            parent_visit_index,
                            parent_size: nodes.len(),
                            parent_weight,
                            standard_n_clusters,
                            source_index: 0,
                            gamma_multiplier: 1.0,
                            repaired: false,
                        };
                        emit_local_merge_margin_summary_trace(
                            trace_context,
                            iteration,
                            "near_tie_refinement_probe",
                            candidate_n_clusters,
                            largest_fraction,
                            &summary,
                        );
                        maybe_apply_near_tie_probe_candidate(
                            &mut choice,
                            candidate_assignments,
                            candidate_counts,
                            candidate_n_clusters,
                            largest_fraction,
                            singleton_weight_fraction,
                            candidate_delta_q,
                            standard_largest_fraction,
                            ddm_config,
                            trace_context,
                            &summary,
                        );
                    }
                    maybe_apply_adaptive_local_shake_eager(
                        &mut choice,
                        subgraph,
                        nodes,
                        config,
                        ddm_config,
                        standard_quality,
                        standard_largest_fraction,
                        standard_singleton_weight_fraction,
                        standard_margin_summary.as_ref(),
                        parent_weight,
                        CandidateTraceContext {
                            depth,
                            parent_id: cid,
                            parent_visit_index,
                            parent_size: nodes.len(),
                            parent_weight,
                            standard_n_clusters,
                            source_index: 0,
                            gamma_multiplier: 1.0,
                            repaired: false,
                        },
                        iteration,
                        randomness,
                        &mut merge_ws,
                        &mut ddm_stats,
                    );
                }
            }

            ddm_stats.standard_refined_clusters += standard_n_clusters;
            if let Some(source) = choice.source {
                emit_candidate_decision_trace(
                    CandidateTraceContext {
                        depth,
                        parent_id: cid,
                        parent_visit_index: ADAPTIVE_PROBE_VISITS.with(|visits| {
                            visits.borrow().get(&(depth, cid)).copied().unwrap_or(0)
                        }),
                        ..CandidateTraceContext::default()
                    },
                    choice.trace_candidate_id,
                    "selected_applied",
                );
                ddm_stats.applied_parents += 1;
                ddm_stats.record_applied(source);
                ddm_stats.record_baseline_repair_selected(choice.baseline_repair_merge_count);
                ddm_stats.record_selected_candidate_quality(source, choice.candidate_delta_q);
                ddm_stats.record_selected_candidate_decision(refinement_candidate_quadrant(
                    choice.candidate_delta_q,
                    choice.largest_fraction,
                    standard_largest_fraction,
                ));
                if dongdaemun.is_some_and(|config| config.use_quotient_diagnostic) {
                    ddm_stats.record_quotient_selected(choice.quotient_score);
                }
                ddm_stats.added_refined_clusters +=
                    choice.n_clusters.saturating_sub(standard_n_clusters);
            }

            for (local_idx, &node) in nodes.iter().enumerate() {
                refinement.clusters[node] =
                    refinement.n_clusters as u32 + choice.assignments[local_idx];
            }
            cluster_counts.extend(choice.counts);
            parent_clusters.resize(parent_clusters.len() + choice.n_clusters, cid as u32);
            if let Some(rf) = reduced_fixed.as_mut() {
                rf.resize(rf.len() + choice.n_clusters, false);
            }
            refinement.n_clusters += choice.n_clusters;
        }
    }

    debug_assert_eq!(parent_clusters.len(), refinement.n_clusters);
    if let Some(rf) = &reduced_fixed {
        debug_assert_eq!(rf.len(), refinement.n_clusters);
    }

    let cluster_starts = counts_to_starts(cluster_counts);
    debug_assert_eq!(cluster_starts.last().copied(), Some(graph.n_nodes as u32));

    RefinementResult {
        clustering: refinement,
        parent_clusters,
        cluster_starts,
        fixed: reduced_fixed,
        dongdaemun_stats: ddm_stats,
    }
}

fn refine_streaming_flat(
    graph: &Graph,
    clustering: &Clustering,
    n_clusters: usize,
    npc_starts: &[u32],
    npc_nodes: &[u32],
    local_index: &mut [u32],
    config: &LeidenConfig,
    dongdaemun: Option<&DongdaemunRefinementConfig>,
    parent_weights: Option<&[f64]>,
    randomness: f64,
    depth: usize,
    iteration: usize,
    rng: &mut impl Rng,
) -> RefinementResult {
    let mut refinement = empty_refinement(graph.n_nodes);
    let mut parent_clusters = Vec::with_capacity(n_clusters);
    let mut cluster_counts = Vec::with_capacity(n_clusters);
    let mut reduced_fixed = clustering
        .fixed
        .as_ref()
        .map(|_| Vec::with_capacity(n_clusters));
    let mut merge_ws = local_merge::LocalMergeWorkspace::new(0);
    let fixed = clustering.fixed.as_deref();
    let mut ddm_stats = RefinementDongdaemunStats::default();
    let selected = if let (Some(config), Some(parent_weights)) = (dongdaemun, parent_weights) {
        let boundary_pressure = parent_boundary_pressure_streaming(
            graph,
            clustering,
            npc_starts,
            npc_nodes,
            parent_weights,
            config,
        );
        let (selected, selected_count, max_parent_weight_seen) =
            select_extra_refinement_parents(parent_weights, config, boundary_pressure.as_deref());
        ddm_stats.selected_parents = selected_count;
        ddm_stats.max_parent_weight_seen = max_parent_weight_seen;
        selected
    } else {
        Vec::new()
    };

    for c in 0..n_clusters {
        let cs = npc_starts[c] as usize;
        let ce = npc_starts[c + 1] as usize;
        let nodes = &npc_nodes[cs..ce];
        if nodes.is_empty() {
            continue;
        }
        if nodes.len() == 1 {
            refinement.clusters[nodes[0] as usize] = refinement.n_clusters as u32;
            parent_clusters.push(c as u32);
            cluster_counts.push(1);
            if let Some(rf) = reduced_fixed.as_mut() {
                rf.push(fixed.is_some_and(|fixed| fixed[nodes[0] as usize]));
            }
            refinement.n_clusters += 1;
            ddm_stats.standard_refined_clusters += 1;
            continue;
        }

        if let Some(fixed) = fixed {
            if nodes.iter().any(|&n| fixed[n as usize]) {
                for &node in nodes {
                    refinement.clusters[node as usize] = refinement.n_clusters as u32;
                }
                parent_clusters.push(c as u32);
                cluster_counts.push(nodes.len() as u32);
                if let Some(rf) = reduced_fixed.as_mut() {
                    rf.push(true);
                }
                refinement.n_clusters += 1;
                ddm_stats.standard_refined_clusters += 1;
                continue;
            }
        }

        let parent_weight = parent_weights
            .and_then(|weights| weights.get(c).copied())
            .unwrap_or_else(|| {
                nodes
                    .iter()
                    .map(|&node| graph.node_weights[node as usize])
                    .sum()
            });
        let parent_visit_index = if selected.get(c).copied().unwrap_or(false) {
            next_adaptive_probe_visit(depth, c)
        } else {
            0
        };
        let mut standard_counts = Vec::new();
        let trace_merge_margins = trace::ddm_trajectory_trace_enabled()
            || dongdaemun.is_some_and(adaptive_local_shake_enabled);
        let (standard_n_clusters, standard_margin_summary) = if trace_merge_margins {
            let (n_clusters, summary) =
                local_merge::find_clustering_induced_u32_with_workspace_assignments_and_append_sizes_traced(
                    graph,
                    nodes,
                    local_index,
                    config.resolution,
                    randomness,
                    rng,
                    &mut merge_ws,
                    &mut standard_counts,
                    local_merge_low_margin_threshold(dongdaemun, parent_weight),
                    None,
                );
            (n_clusters, Some(summary))
        } else {
            let n_clusters =
                local_merge::find_clustering_induced_u32_with_workspace_assignments_and_append_sizes(
                    graph,
                    nodes,
                    local_index,
                    config.resolution,
                    randomness,
                    rng,
                    &mut merge_ws,
                    &mut standard_counts,
                );
            (n_clusters, None)
        };
        let standard_assignments = merge_ws.assignments()[..nodes.len()].to_vec();
        let (standard_largest_fraction, standard_singleton_weight_fraction) =
            parent_partition_summary(
                nodes.len(),
                standard_n_clusters,
                &standard_assignments,
                parent_weight,
                |local| graph.node_weights[nodes[local] as usize],
            );
        let standard_quality = parent_partition_quality_induced_u32(
            graph,
            nodes,
            local_index,
            &standard_assignments,
            standard_n_clusters,
            config.resolution,
        );
        if let Some(summary) = &standard_margin_summary {
            emit_local_merge_margin_summary_trace(
                CandidateTraceContext {
                    depth,
                    parent_id: c,
                    parent_visit_index,
                    parent_size: nodes.len(),
                    parent_weight,
                    standard_n_clusters,
                    source_index: 0,
                    gamma_multiplier: 1.0,
                    repaired: false,
                },
                iteration,
                "standard_refinement",
                standard_n_clusters,
                standard_largest_fraction,
                summary,
            );
        }

        let mut choice = ParentRefinementChoice {
            assignments: standard_assignments,
            counts: standard_counts,
            n_clusters: standard_n_clusters,
            largest_fraction: standard_largest_fraction,
            singleton_weight_fraction: standard_singleton_weight_fraction,
            quotient_score: 0.0,
            candidate_delta_q: 0.0,
            source: None,
            trace_candidate_id: None,
            baseline_repair_merge_count: 0,
            baseline_repair_delta_sum: 0.0,
            adaptive_probe_baseline_delta_q: None,
            adaptive_probe_committed: false,
            adaptive_probe_score: f64::NEG_INFINITY,
            adaptive_probe_source_label: None,
        };

        if selected.get(c).copied().unwrap_or(false) {
            if let Some(ddm_config) = dongdaemun {
                for perturbation_index in 0..ddm_config.seed_perturbations {
                    let mut candidate_counts = Vec::new();
                    let derived_seed =
                        derive_same_gamma_seed(config.seed, depth, c, perturbation_index);
                    let mut candidate_rng = rand::rngs::StdRng::seed_from_u64(derived_seed);
                    let candidate_n_clusters = local_merge::find_clustering_induced_u32_with_workspace_assignments_and_append_sizes(
                        graph,
                        nodes,
                        local_index,
                        config.resolution,
                        randomness,
                        &mut candidate_rng,
                        &mut merge_ws,
                        &mut candidate_counts,
                    );
                    let candidate_assignments = merge_ws.assignments()[..nodes.len()].to_vec();
                    let candidate_delta_q = parent_partition_quality_induced_u32(
                        graph,
                        nodes,
                        local_index,
                        &candidate_assignments,
                        candidate_n_clusters,
                        config.resolution,
                    ) - standard_quality;
                    let (largest_fraction, singleton_weight_fraction) = parent_partition_summary(
                        nodes.len(),
                        candidate_n_clusters,
                        &candidate_assignments,
                        parent_weight,
                        |local| graph.node_weights[nodes[local] as usize],
                    );
                    let quotient_score = if ddm_config.use_quotient_diagnostic
                        && nodes.len() > 1
                        && parent_candidate_is_valid(
                            candidate_n_clusters,
                            largest_fraction,
                            singleton_weight_fraction,
                            standard_largest_fraction,
                            ddm_config,
                        ) {
                        parent_weights.map(|parent_weights| {
                            parent_candidate_quotient_score(
                                graph,
                                clustering,
                                c,
                                nodes.len(),
                                &candidate_assignments,
                                candidate_n_clusters,
                                parent_weight,
                                parent_weights,
                                config.resolution,
                                |local| nodes[local] as usize,
                            )
                        })
                    } else {
                        None
                    };
                    consider_parent_candidate_with_trace(
                        &mut choice,
                        candidate_assignments,
                        candidate_counts,
                        candidate_n_clusters,
                        largest_fraction,
                        singleton_weight_fraction,
                        quotient_score,
                        0,
                        0.0,
                        candidate_delta_q,
                        standard_largest_fraction,
                        RefinementCandidateSource::SameGammaSeed,
                        ddm_config,
                        &mut ddm_stats,
                        CandidateTraceContext {
                            depth,
                            parent_id: c,
                            parent_visit_index,
                            parent_size: nodes.len(),
                            parent_weight,
                            standard_n_clusters,
                            source_index: perturbation_index,
                            gamma_multiplier: 1.0,
                            repaired: false,
                        },
                    );
                }
                for (multiplier_index, multiplier) in
                    ddm_config.gamma_multipliers.iter().enumerate()
                {
                    let mut candidate_counts = Vec::new();
                    let derived_seed =
                        derive_high_gamma_seed(config.seed, depth, c, multiplier_index);
                    let mut candidate_rng = rand::rngs::StdRng::seed_from_u64(derived_seed);
                    let candidate_n_clusters = local_merge::find_clustering_induced_u32_with_workspace_assignments_and_append_sizes(
                        graph,
                        nodes,
                        local_index,
                        config.resolution * *multiplier,
                        randomness,
                        &mut candidate_rng,
                        &mut merge_ws,
                        &mut candidate_counts,
                    );
                    let mut candidate_assignments = merge_ws.assignments()[..nodes.len()].to_vec();
                    let mut candidate_counts = candidate_counts;
                    let mut candidate_n_clusters = candidate_n_clusters;
                    let mut baseline_repair_merge_count = 0usize;
                    let mut baseline_repair_delta_sum = 0.0;
                    let mut consider_repaired_candidate = true;
                    let repair_policy = effective_baseline_repair_policy(ddm_config, parent_weight);
                    if ddm_config.use_baseline_repair && nodes.len() > 1 && candidate_n_clusters > 1
                    {
                        if repair_policy == BaselineRepairPolicy::Augment {
                            let (largest_fraction, singleton_weight_fraction) =
                                parent_partition_summary(
                                    nodes.len(),
                                    candidate_n_clusters,
                                    &candidate_assignments,
                                    parent_weight,
                                    |local| graph.node_weights[nodes[local] as usize],
                                );
                            let candidate_delta_q = parent_partition_quality_induced_u32(
                                graph,
                                nodes,
                                local_index,
                                &candidate_assignments,
                                candidate_n_clusters,
                                config.resolution,
                            ) - standard_quality;
                            let quotient_score = if ddm_config.use_quotient_diagnostic
                                && nodes.len() > 1
                                && parent_candidate_is_valid(
                                    candidate_n_clusters,
                                    largest_fraction,
                                    singleton_weight_fraction,
                                    standard_largest_fraction,
                                    ddm_config,
                                ) {
                                parent_weights.map(|parent_weights| {
                                    parent_candidate_quotient_score(
                                        graph,
                                        clustering,
                                        c,
                                        nodes.len(),
                                        &candidate_assignments,
                                        candidate_n_clusters,
                                        parent_weight,
                                        parent_weights,
                                        config.resolution,
                                        |local| nodes[local] as usize,
                                    )
                                })
                            } else {
                                None
                            };
                            consider_parent_candidate_with_trace(
                                &mut choice,
                                candidate_assignments.clone(),
                                candidate_counts.clone(),
                                candidate_n_clusters,
                                largest_fraction,
                                singleton_weight_fraction,
                                quotient_score,
                                0,
                                0.0,
                                candidate_delta_q,
                                standard_largest_fraction,
                                RefinementCandidateSource::HighGamma,
                                ddm_config,
                                &mut ddm_stats,
                                CandidateTraceContext {
                                    depth,
                                    parent_id: c,
                                    parent_visit_index,
                                    parent_size: nodes.len(),
                                    parent_weight,
                                    standard_n_clusters,
                                    source_index: multiplier_index,
                                    gamma_multiplier: *multiplier,
                                    repaired: false,
                                },
                            );
                        }
                        let repair = repair_parent_internal_candidate(
                            graph,
                            nodes.len(),
                            &candidate_assignments,
                            candidate_n_clusters,
                            config.resolution,
                            ddm_config.baseline_repair_epsilon,
                            |local| nodes[local] as usize,
                        );
                        ddm_stats.record_baseline_repair_candidate(&repair);
                        if repair_policy == BaselineRepairPolicy::Augment && !repair.changed {
                            consider_repaired_candidate = false;
                        }
                        baseline_repair_merge_count = repair.merge_count;
                        baseline_repair_delta_sum = repair.delta_sum;
                        candidate_assignments = repair.assignments;
                        candidate_counts = repair.counts;
                        candidate_n_clusters = repair.n_clusters;
                    }
                    if consider_repaired_candidate {
                        let (largest_fraction, singleton_weight_fraction) =
                            parent_partition_summary(
                                nodes.len(),
                                candidate_n_clusters,
                                &candidate_assignments,
                                parent_weight,
                                |local| graph.node_weights[nodes[local] as usize],
                            );
                        let candidate_delta_q = parent_partition_quality_induced_u32(
                            graph,
                            nodes,
                            local_index,
                            &candidate_assignments,
                            candidate_n_clusters,
                            config.resolution,
                        ) - standard_quality;
                        let quotient_score = if ddm_config.use_quotient_diagnostic
                            && nodes.len() > 1
                            && parent_candidate_is_valid(
                                candidate_n_clusters,
                                largest_fraction,
                                singleton_weight_fraction,
                                standard_largest_fraction,
                                ddm_config,
                            ) {
                            parent_weights.map(|parent_weights| {
                                parent_candidate_quotient_score(
                                    graph,
                                    clustering,
                                    c,
                                    nodes.len(),
                                    &candidate_assignments,
                                    candidate_n_clusters,
                                    parent_weight,
                                    parent_weights,
                                    config.resolution,
                                    |local| nodes[local] as usize,
                                )
                            })
                        } else {
                            None
                        };
                        consider_parent_candidate_with_trace(
                            &mut choice,
                            candidate_assignments,
                            candidate_counts,
                            candidate_n_clusters,
                            largest_fraction,
                            singleton_weight_fraction,
                            quotient_score,
                            baseline_repair_merge_count,
                            baseline_repair_delta_sum,
                            candidate_delta_q,
                            standard_largest_fraction,
                            RefinementCandidateSource::HighGamma,
                            ddm_config,
                            &mut ddm_stats,
                            CandidateTraceContext {
                                depth,
                                parent_id: c,
                                parent_visit_index,
                                parent_size: nodes.len(),
                                parent_weight,
                                standard_n_clusters,
                                source_index: multiplier_index,
                                gamma_multiplier: *multiplier,
                                repaired: baseline_repair_merge_count > 0,
                            },
                        );
                    }
                }
                if adaptive_probe_should_probe(ddm_config, depth, c, parent_visit_index) {
                    for perturbation_index in 0..ddm_config.adaptive_probe_perturbations {
                        let mut candidate_counts = Vec::new();
                        let derived_seed = derive_adaptive_probe_seed(
                            config.seed,
                            depth,
                            c,
                            parent_visit_index,
                            0x5155_414d_455f_4741,
                            perturbation_index,
                        );
                        let mut candidate_rng = rand::rngs::StdRng::seed_from_u64(derived_seed);
                        let candidate_n_clusters = local_merge::find_clustering_induced_u32_with_workspace_assignments_and_append_sizes(
                            graph,
                            nodes,
                            local_index,
                            config.resolution,
                            randomness,
                            &mut candidate_rng,
                            &mut merge_ws,
                            &mut candidate_counts,
                        );
                        let candidate_assignments = merge_ws.assignments()[..nodes.len()].to_vec();
                        let candidate_delta_q = parent_partition_quality_induced_u32(
                            graph,
                            nodes,
                            local_index,
                            &candidate_assignments,
                            candidate_n_clusters,
                            config.resolution,
                        ) - standard_quality;
                        let (largest_fraction, singleton_weight_fraction) =
                            parent_partition_summary(
                                nodes.len(),
                                candidate_n_clusters,
                                &candidate_assignments,
                                parent_weight,
                                |local| graph.node_weights[nodes[local] as usize],
                            );
                        maybe_apply_adaptive_probe_candidate(
                            &mut choice,
                            candidate_assignments,
                            candidate_counts,
                            candidate_n_clusters,
                            largest_fraction,
                            singleton_weight_fraction,
                            candidate_delta_q,
                            standard_largest_fraction,
                            ddm_config,
                            CandidateTraceContext {
                                depth,
                                parent_id: c,
                                parent_visit_index,
                                parent_size: nodes.len(),
                                parent_weight,
                                standard_n_clusters,
                                source_index: perturbation_index,
                                gamma_multiplier: 1.0,
                                repaired: false,
                            },
                            "same_gamma_probe",
                            perturbation_index,
                        );
                    }
                    if ddm_config.adaptive_probe_include_node_order_control {
                        for perturbation_index in 0..ddm_config.adaptive_probe_perturbations {
                            let mut candidate_counts = Vec::new();
                            let derived_seed = derive_adaptive_probe_seed(
                                config.seed,
                                depth,
                                c,
                                parent_visit_index,
                                0x4e4f_4445_4f52_4452,
                                perturbation_index,
                            );
                            let mut candidate_rng = rand::rngs::StdRng::seed_from_u64(derived_seed);
                            let candidate_n_clusters = local_merge::find_clustering_induced_u32_with_workspace_assignments_and_append_sizes(
                                graph,
                                nodes,
                                local_index,
                                config.resolution,
                                randomness,
                                &mut candidate_rng,
                                &mut merge_ws,
                                &mut candidate_counts,
                            );
                            let candidate_assignments =
                                merge_ws.assignments()[..nodes.len()].to_vec();
                            let candidate_delta_q = parent_partition_quality_induced_u32(
                                graph,
                                nodes,
                                local_index,
                                &candidate_assignments,
                                candidate_n_clusters,
                                config.resolution,
                            ) - standard_quality;
                            let (largest_fraction, singleton_weight_fraction) =
                                parent_partition_summary(
                                    nodes.len(),
                                    candidate_n_clusters,
                                    &candidate_assignments,
                                    parent_weight,
                                    |local| graph.node_weights[nodes[local] as usize],
                                );
                            maybe_apply_adaptive_probe_candidate(
                                &mut choice,
                                candidate_assignments,
                                candidate_counts,
                                candidate_n_clusters,
                                largest_fraction,
                                singleton_weight_fraction,
                                candidate_delta_q,
                                standard_largest_fraction,
                                ddm_config,
                                CandidateTraceContext {
                                    depth,
                                    parent_id: c,
                                    parent_visit_index,
                                    parent_size: nodes.len(),
                                    parent_weight,
                                    standard_n_clusters,
                                    source_index: perturbation_index,
                                    gamma_multiplier: 1.0,
                                    repaired: false,
                                },
                                "node_order_control",
                                perturbation_index,
                            );
                        }
                    }
                }
                if adaptive_near_tie_probe_enabled(ddm_config) {
                    let mut candidate_counts = Vec::new();
                    let derived_seed = derive_adaptive_probe_seed(
                        config.seed,
                        depth,
                        c,
                        parent_visit_index,
                        0x4e45_4152_5449_4550,
                        0,
                    );
                    let mut candidate_rng = rand::rngs::StdRng::seed_from_u64(derived_seed);
                    let near_tie = local_merge::NearTieProbeConfig {
                        parent_weight,
                        margin_parent_weight: ddm_config.adaptive_near_tie_margin_parent_weight,
                        randomness: ddm_config.adaptive_near_tie_randomness,
                        max_decisions_per_parent: ddm_config
                            .adaptive_near_tie_max_decisions_per_parent,
                    };
                    let (candidate_n_clusters, summary) =
                        local_merge::find_clustering_induced_u32_with_workspace_assignments_and_append_sizes_traced(
                            graph,
                            nodes,
                            local_index,
                            config.resolution,
                            randomness,
                            &mut candidate_rng,
                            &mut merge_ws,
                            &mut candidate_counts,
                            ddm_config.adaptive_near_tie_margin_parent_weight
                                * parent_weight.max(0.0),
                            Some(near_tie),
                        );
                    let candidate_assignments = merge_ws.assignments()[..nodes.len()].to_vec();
                    let candidate_delta_q = parent_partition_quality_induced_u32(
                        graph,
                        nodes,
                        local_index,
                        &candidate_assignments,
                        candidate_n_clusters,
                        config.resolution,
                    ) - standard_quality;
                    let (largest_fraction, singleton_weight_fraction) = parent_partition_summary(
                        nodes.len(),
                        candidate_n_clusters,
                        &candidate_assignments,
                        parent_weight,
                        |local| graph.node_weights[nodes[local] as usize],
                    );
                    let trace_context = CandidateTraceContext {
                        depth,
                        parent_id: c,
                        parent_visit_index,
                        parent_size: nodes.len(),
                        parent_weight,
                        standard_n_clusters,
                        source_index: 0,
                        gamma_multiplier: 1.0,
                        repaired: false,
                    };
                    emit_local_merge_margin_summary_trace(
                        trace_context,
                        iteration,
                        "near_tie_refinement_probe",
                        candidate_n_clusters,
                        largest_fraction,
                        &summary,
                    );
                    maybe_apply_near_tie_probe_candidate(
                        &mut choice,
                        candidate_assignments,
                        candidate_counts,
                        candidate_n_clusters,
                        largest_fraction,
                        singleton_weight_fraction,
                        candidate_delta_q,
                        standard_largest_fraction,
                        ddm_config,
                        trace_context,
                        &summary,
                    );
                }
                maybe_apply_adaptive_local_shake_streaming(
                    &mut choice,
                    graph,
                    nodes,
                    local_index,
                    config,
                    ddm_config,
                    standard_quality,
                    standard_largest_fraction,
                    standard_singleton_weight_fraction,
                    standard_margin_summary.as_ref(),
                    parent_weight,
                    CandidateTraceContext {
                        depth,
                        parent_id: c,
                        parent_visit_index,
                        parent_size: nodes.len(),
                        parent_weight,
                        standard_n_clusters,
                        source_index: 0,
                        gamma_multiplier: 1.0,
                        repaired: false,
                    },
                    iteration,
                    randomness,
                    &mut merge_ws,
                    &mut ddm_stats,
                );
            }
        }

        ddm_stats.standard_refined_clusters += standard_n_clusters;
        if let Some(source) = choice.source {
            emit_candidate_decision_trace(
                CandidateTraceContext {
                    depth,
                    parent_id: c,
                    parent_visit_index: ADAPTIVE_PROBE_VISITS
                        .with(|visits| visits.borrow().get(&(depth, c)).copied().unwrap_or(0)),
                    ..CandidateTraceContext::default()
                },
                choice.trace_candidate_id,
                "selected_applied",
            );
            ddm_stats.applied_parents += 1;
            ddm_stats.record_applied(source);
            ddm_stats.record_baseline_repair_selected(choice.baseline_repair_merge_count);
            ddm_stats.record_selected_candidate_quality(source, choice.candidate_delta_q);
            ddm_stats.record_selected_candidate_decision(refinement_candidate_quadrant(
                choice.candidate_delta_q,
                choice.largest_fraction,
                standard_largest_fraction,
            ));
            if dongdaemun.is_some_and(|config| config.use_quotient_diagnostic) {
                ddm_stats.record_quotient_selected(choice.quotient_score);
            }
            ddm_stats.added_refined_clusters +=
                choice.n_clusters.saturating_sub(standard_n_clusters);
        }

        for (local_idx, &node) in nodes.iter().enumerate() {
            refinement.clusters[node as usize] =
                refinement.n_clusters as u32 + choice.assignments[local_idx];
        }
        cluster_counts.extend(choice.counts);
        parent_clusters.resize(parent_clusters.len() + choice.n_clusters, c as u32);
        if let Some(rf) = reduced_fixed.as_mut() {
            rf.resize(rf.len() + choice.n_clusters, false);
        }
        refinement.n_clusters += choice.n_clusters;
    }

    debug_assert_eq!(parent_clusters.len(), refinement.n_clusters);
    if let Some(rf) = &reduced_fixed {
        debug_assert_eq!(rf.len(), refinement.n_clusters);
    }

    let cluster_starts = counts_to_starts(cluster_counts);
    debug_assert_eq!(cluster_starts.last().copied(), Some(graph.n_nodes as u32));

    RefinementResult {
        clustering: refinement,
        parent_clusters,
        cluster_starts,
        fixed: reduced_fixed,
        dongdaemun_stats: ddm_stats,
    }
}

use crate::quality::QualityFunction;
use rand::SeedableRng;

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};

    #[test]
    fn test_leiden_two_cliques() {
        // Two triangles with weak bridge
        let g = Graph::from_edge_list(
            6,
            &[0, 1, 2, 3, 4, 5, 2],
            &[1, 2, 0, 4, 5, 3, 3],
            &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.01],
        );
        let config = LeidenConfig {
            resolution: 0.5,
            n_iterations: 10,
            randomness: 0.01,
            randomness_schedule: Vec::new(),
            seed: 42,
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let result = leiden(&g, &config, None, &mut rng);

        // Should find 2 communities
        assert!(result.clustering.n_clusters <= 3);
        assert!(result.quality > 0.0);
        // Same clique = same cluster
        assert_eq!(result.clustering.clusters[0], result.clustering.clusters[1]);
        assert_eq!(result.clustering.clusters[3], result.clustering.clusters[4]);
    }

    #[test]
    fn test_dongdaemun_quality_trace_pressure_metrics_use_target_weight() {
        let g = Graph::from_edge_list_weighted(4, &[0], &[1], &[1.0], &[3.0, 3.0, 4.0, 1.0]);
        let clustering = Clustering::from_assignments(vec![0, 0, 1, 1]);

        let (max_doc_weight, max_doc_weight_ratio, n_above_max_doc_weight) =
            quality_trace_pressure_metrics(&g, &clustering, 5.0);

        assert!((max_doc_weight - 6.0).abs() < 1e-12);
        assert!((max_doc_weight_ratio - 1.2).abs() < 1e-12);
        assert_eq!(n_above_max_doc_weight, 1);
    }

    #[test]
    fn test_leiden_with_fixed() {
        let g = Graph::from_edge_list(
            6,
            &[0, 1, 2, 3, 4, 5, 2],
            &[1, 2, 0, 4, 5, 3, 3],
            &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.01],
        );

        // Fix nodes 0,1,2 in cluster 0; let 3,4,5 be free
        let mut init = Clustering::from_assignments(vec![0, 0, 0, 1, 2, 3]);
        init.set_fixed(vec![true, true, true, false, false, false]);

        let config = LeidenConfig {
            resolution: 0.5,
            n_iterations: 10,
            randomness: 0.01,
            randomness_schedule: Vec::new(),
            seed: 42,
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let result = leiden(&g, &config, Some(init), &mut rng);

        // Fixed nodes should stay in cluster 0
        assert_eq!(result.clustering.clusters[0], result.clustering.clusters[1]);
        assert_eq!(result.clustering.clusters[1], result.clustering.clusters[2]);
        // Free nodes 3,4,5 should form their own cluster
        assert_eq!(result.clustering.clusters[3], result.clustering.clusters[4]);
    }

    #[test]
    fn test_flat_nodes_per_cluster_layout() {
        let clustering = Clustering::from_assignments(vec![1, 0, 1, 0, 2]);
        let mut ws = Workspace::new(clustering.n_nodes);

        clustering.fill_cluster_groups(&mut ws);

        assert_eq!(&ws.npc_starts[..=clustering.n_clusters], &[0, 2, 4, 5]);
        assert_eq!(&ws.npc_nodes[..clustering.n_nodes], &[1, 3, 0, 2, 4]);
    }

    #[test]
    fn test_empty_refinement_is_zeroed_not_singleton() {
        let refinement = empty_refinement(4);
        assert_eq!(refinement.n_nodes, 4);
        assert_eq!(refinement.n_clusters, 0);
        assert_eq!(refinement.clusters, vec![0, 0, 0, 0]);
        assert!(refinement.fixed.is_none());
    }

    #[test]
    fn test_recursion_guard_only_triggers_for_large_near_identity_contractions() {
        assert!(!recursion_guard_triggers(
            1_000_000,
            99_999_999,
            999_999,
            100_000_000,
            1e-4
        ));
        assert!(!recursion_guard_triggers(
            1_000_000,
            100_000_000,
            999_800,
            100_000_000,
            1e-4
        ));
        assert!(recursion_guard_triggers(
            1_000_000,
            100_000_000,
            999_950,
            100_000_000,
            1e-4
        ));
        assert!(!recursion_guard_triggers(
            1_000_000,
            100_000_000,
            999_950,
            100_000_000,
            0.0
        ));
    }

    #[test]
    fn test_dongdaemun_refinement_no_eligible_parent_matches_standard_leiden() {
        let g = Graph::from_edge_list(
            4,
            &[0, 2, 0, 0, 1, 1],
            &[1, 3, 2, 3, 2, 3],
            &[10.0, 10.0, 1.0, 1.0, 1.0, 1.0],
        );
        let config = LeidenConfig {
            resolution: 0.001,
            n_iterations: 2,
            randomness: 0.0,
            randomness_schedule: Vec::new(),
            seed: 7,
        };
        let ddm = DongdaemunRefinementConfig {
            target_max_weight: 100.0,
            max_extra_parents_per_iteration: 1,
            gamma_multipliers: vec![1000.0],
            ..DongdaemunRefinementConfig::default()
        };

        let mut standard_rng = rand::rngs::StdRng::seed_from_u64(7);
        let standard = leiden(&g, &config, None, &mut standard_rng);
        let mut ddm_rng = rand::rngs::StdRng::seed_from_u64(7);
        let refined = leiden_with_dongdaemun_refinement(&g, &config, &ddm, None, &mut ddm_rng);

        assert_eq!(standard.clustering.clusters, refined.clustering.clusters);
        assert_eq!(
            standard.clustering.n_clusters,
            refined.clustering.n_clusters
        );
        assert!((standard.quality - refined.quality).abs() < 1e-12);
        assert_eq!(refined.audit.selected_parent_count_total, 0);
        assert_eq!(refined.audit.applied_parent_count_total, 0);
    }

    #[test]
    fn test_dongdaemun_refinement_applies_parent_internal_high_gamma_split() {
        let g = Graph::from_edge_list(4, &[0, 2, 1], &[1, 3, 2], &[10.0, 10.0, 0.01]);
        let clustering = Clustering::from_assignments(vec![0, 0, 0, 0]);
        let nodes_per_cluster = clustering.nodes_per_cluster();
        let config = LeidenConfig {
            resolution: 0.000001,
            n_iterations: 1,
            randomness: 0.0,
            randomness_schedule: Vec::new(),
            seed: 11,
        };
        let ddm = DongdaemunRefinementConfig {
            target_max_weight: 2.5,
            max_extra_parents_per_iteration: 1,
            max_singleton_weight_fraction: 1.0,
            gamma_multipliers: vec![20_000_000.0],
            ..DongdaemunRefinementConfig::default()
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(11);

        let refinement = refine_eager(
            &g,
            &clustering,
            &nodes_per_cluster,
            &config,
            Some(&ddm),
            Some(&[4.0]),
            config.randomness,
            0,
            0,
            &mut rng,
        );

        assert_eq!(refinement.dongdaemun_stats.selected_parents, 1);
        assert_eq!(refinement.dongdaemun_stats.applied_parents, 1);
        assert_eq!(refinement.dongdaemun_stats.same_gamma_candidates, 0);
        assert_eq!(refinement.dongdaemun_stats.high_gamma_candidates, 1);
        assert_eq!(refinement.dongdaemun_stats.same_gamma_applied, 0);
        assert_eq!(refinement.dongdaemun_stats.high_gamma_applied, 1);
        assert!(refinement.clustering.n_clusters >= 2);
        assert_eq!(
            refinement.parent_clusters.len(),
            refinement.clustering.n_clusters
        );
        assert!(refinement.parent_clusters.iter().all(|&parent| parent == 0));
    }

    #[test]
    fn test_dongdaemun_refinement_high_gamma_reject_does_not_consume_main_rng_eager() {
        let g = Graph::from_edge_list(4, &[0, 2, 1], &[1, 3, 2], &[10.0, 10.0, 0.01]);
        let clustering = Clustering::from_assignments(vec![0, 0, 0, 0]);
        let nodes_per_cluster = clustering.nodes_per_cluster();
        let config = LeidenConfig {
            resolution: 0.000001,
            n_iterations: 1,
            randomness: 0.0,
            randomness_schedule: Vec::new(),
            seed: 11,
        };
        let ddm = DongdaemunRefinementConfig {
            target_max_weight: 2.5,
            max_extra_parents_per_iteration: 1,
            max_singleton_weight_fraction: 1.0,
            gamma_multipliers: vec![20_000_000.0],
            candidate_quality_policy: CandidateQualityPolicy::QualityFloor,
            min_candidate_delta_q: 1.0e9,
            ..DongdaemunRefinementConfig::default()
        };

        let mut standard_rng = rand::rngs::StdRng::seed_from_u64(11);
        let standard = refine_eager(
            &g,
            &clustering,
            &nodes_per_cluster,
            &config,
            None,
            None,
            config.randomness,
            0,
            0,
            &mut standard_rng,
        );
        let standard_next: u64 = standard_rng.gen();

        let mut ddm_rng = rand::rngs::StdRng::seed_from_u64(11);
        let refined = refine_eager(
            &g,
            &clustering,
            &nodes_per_cluster,
            &config,
            Some(&ddm),
            Some(&[4.0]),
            config.randomness,
            0,
            0,
            &mut ddm_rng,
        );
        let refined_next: u64 = ddm_rng.gen();

        assert_eq!(standard.clustering.clusters, refined.clustering.clusters);
        assert_eq!(
            standard.clustering.n_clusters,
            refined.clustering.n_clusters
        );
        assert_eq!(standard_next, refined_next);
        assert_eq!(refined.dongdaemun_stats.high_gamma_candidates, 1);
        assert_eq!(refined.dongdaemun_stats.candidate_rejected_by_quality, 1);
        assert_eq!(refined.dongdaemun_stats.applied_parents, 0);
    }

    #[test]
    fn test_dongdaemun_refinement_final_quality_guard_falls_back_to_standard() {
        let g = Graph::from_edge_list(4, &[0, 2, 1], &[1, 3, 2], &[10.0, 10.0, 0.01]);
        let initial = Clustering::from_assignments(vec![0, 0, 0, 0]);
        let config = LeidenConfig {
            resolution: 0.000001,
            n_iterations: 1,
            randomness: 0.0,
            randomness_schedule: Vec::new(),
            seed: 11,
        };
        let ddm = DongdaemunRefinementConfig {
            target_max_weight: 2.5,
            max_extra_parents_per_iteration: 1,
            max_singleton_weight_fraction: 1.0,
            gamma_multipliers: vec![20_000_000.0],
            use_final_quality_guard: true,
            min_final_quality_delta: 1.0e9,
            ..DongdaemunRefinementConfig::default()
        };

        let mut standard_rng = rand::rngs::StdRng::seed_from_u64(11);
        let standard = leiden(&g, &config, Some(initial.clone()), &mut standard_rng);
        let mut ddm_rng = rand::rngs::StdRng::seed_from_u64(11);
        let guarded =
            leiden_with_dongdaemun_refinement(&g, &config, &ddm, Some(initial), &mut ddm_rng);

        assert!(guarded.audit.final_quality_guard_enabled);
        assert!(guarded.audit.final_quality_guard_triggered);
        assert_eq!(guarded.clustering.clusters, standard.clustering.clusters);
        assert_eq!(
            guarded.clustering.n_clusters,
            standard.clustering.n_clusters
        );
        assert_eq!(guarded.quality, standard.quality);
        assert!(guarded.audit.applied_parent_count_total >= 1);
        assert!(guarded.audit.final_quality_delta_vs_guard_standard < ddm.min_final_quality_delta);
    }

    #[test]
    fn test_dongdaemun_refinement_high_gamma_reject_does_not_consume_main_rng_streaming() {
        let g = Graph::from_edge_list(4, &[0, 2, 1], &[1, 3, 2], &[10.0, 10.0, 0.01]);
        let clustering = Clustering::from_assignments(vec![0, 0, 0, 0]);
        let parent_weights = clustering.cluster_weights(&g.node_weights);
        let config = LeidenConfig {
            resolution: 0.000001,
            n_iterations: 1,
            randomness: 0.0,
            randomness_schedule: Vec::new(),
            seed: 11,
        };
        let ddm = DongdaemunRefinementConfig {
            target_max_weight: 2.5,
            max_extra_parents_per_iteration: 1,
            max_singleton_weight_fraction: 1.0,
            gamma_multipliers: vec![20_000_000.0],
            candidate_quality_policy: CandidateQualityPolicy::QualityFloor,
            min_candidate_delta_q: 1.0e9,
            ..DongdaemunRefinementConfig::default()
        };

        let mut standard_ws = Workspace::new(g.n_nodes);
        clustering.fill_cluster_groups_and_weights(&g.node_weights, &mut standard_ws);
        let starts = standard_ws.npc_starts[..clustering.n_clusters + 1].to_vec();
        let flat_nodes = standard_ws.npc_nodes[..g.n_nodes].to_vec();
        let mut standard_local_index = vec![u32::MAX; g.n_nodes];
        let mut standard_rng = rand::rngs::StdRng::seed_from_u64(11);
        let standard = refine_streaming_flat(
            &g,
            &clustering,
            clustering.n_clusters,
            &starts,
            &flat_nodes,
            &mut standard_local_index,
            &config,
            None,
            None,
            config.randomness,
            0,
            0,
            &mut standard_rng,
        );
        let standard_next: u64 = standard_rng.gen();

        let mut refined_local_index = vec![u32::MAX; g.n_nodes];
        let mut ddm_rng = rand::rngs::StdRng::seed_from_u64(11);
        let refined = refine_streaming_flat(
            &g,
            &clustering,
            clustering.n_clusters,
            &starts,
            &flat_nodes,
            &mut refined_local_index,
            &config,
            Some(&ddm),
            Some(&parent_weights),
            config.randomness,
            0,
            0,
            &mut ddm_rng,
        );
        let refined_next: u64 = ddm_rng.gen();

        assert_eq!(standard.clustering.clusters, refined.clustering.clusters);
        assert_eq!(
            standard.clustering.n_clusters,
            refined.clustering.n_clusters
        );
        assert_eq!(standard_next, refined_next);
        assert_eq!(refined.dongdaemun_stats.high_gamma_candidates, 1);
        assert_eq!(refined.dongdaemun_stats.candidate_rejected_by_quality, 1);
        assert_eq!(refined.dongdaemun_stats.applied_parents, 0);
    }

    #[test]
    fn test_near_tie_refinement_probe_keeps_children_inside_parent() {
        let g = Graph::from_edge_list(4, &[0, 2, 1], &[1, 3, 2], &[10.0, 10.0, 0.01]);
        let clustering = Clustering::from_assignments(vec![0, 0, 0, 0]);
        let nodes_per_cluster = clustering.nodes_per_cluster();
        let config = LeidenConfig {
            resolution: 0.000001,
            n_iterations: 1,
            randomness: 0.0,
            randomness_schedule: Vec::new(),
            seed: 11,
        };
        let ddm = DongdaemunRefinementConfig {
            target_max_weight: 2.5,
            max_extra_parents_per_iteration: 1,
            max_singleton_weight_fraction: 1.0,
            gamma_multipliers: vec![],
            adaptive_near_tie_probe_mode: AdaptiveNearTieProbeMode::TraceOnly,
            adaptive_near_tie_margin_parent_weight: 1.0,
            adaptive_near_tie_randomness: 1.0,
            adaptive_near_tie_max_decisions_per_parent: 8,
            ..DongdaemunRefinementConfig::default()
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(11);

        let refinement = refine_eager(
            &g,
            &clustering,
            &nodes_per_cluster,
            &config,
            Some(&ddm),
            Some(&[4.0]),
            config.randomness,
            0,
            0,
            &mut rng,
        );

        assert_eq!(refinement.dongdaemun_stats.selected_parents, 1);
        assert!(refinement.parent_clusters.iter().all(|&parent| parent == 0));
        assert_eq!(
            refinement.parent_clusters.len(),
            refinement.clustering.n_clusters
        );
    }

    #[test]
    fn test_dongdaemun_refinement_same_gamma_seed_candidates_without_high_gamma() {
        let g = Graph::from_edge_list(4, &[0, 2, 1], &[1, 3, 2], &[10.0, 10.0, 0.01]);
        let clustering = Clustering::from_assignments(vec![0, 0, 0, 0]);
        let nodes_per_cluster = clustering.nodes_per_cluster();
        let config = LeidenConfig {
            resolution: 0.000001,
            n_iterations: 1,
            randomness: 0.01,
            randomness_schedule: Vec::new(),
            seed: 11,
        };
        let ddm = DongdaemunRefinementConfig {
            target_max_weight: 2.5,
            max_extra_parents_per_iteration: 1,
            max_singleton_weight_fraction: 1.0,
            gamma_multipliers: Vec::new(),
            seed_perturbations: 3,
            use_baseline_repair: true,
            ..DongdaemunRefinementConfig::default()
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(11);

        let refinement = refine_eager(
            &g,
            &clustering,
            &nodes_per_cluster,
            &config,
            Some(&ddm),
            Some(&[4.0]),
            config.randomness,
            0,
            0,
            &mut rng,
        );

        assert_eq!(refinement.dongdaemun_stats.selected_parents, 1);
        assert_eq!(refinement.dongdaemun_stats.same_gamma_candidates, 3);
        assert_eq!(refinement.dongdaemun_stats.high_gamma_candidates, 0);
        assert_eq!(refinement.dongdaemun_stats.high_gamma_applied, 0);
        assert_eq!(refinement.dongdaemun_stats.baseline_repair_candidates, 0);
        assert_eq!(
            refinement.parent_clusters.len(),
            refinement.clustering.n_clusters
        );
        assert!(refinement.parent_clusters.iter().all(|&parent| parent == 0));
    }

    fn record_test_final_candidate_decision(
        stats: &mut RefinementDongdaemunStats,
        choice: &ParentRefinementChoice,
        standard_largest_fraction: f64,
    ) {
        if choice.source.is_some() {
            stats.record_selected_candidate_decision(refinement_candidate_quadrant(
                choice.candidate_delta_q,
                choice.largest_fraction,
                standard_largest_fraction,
            ));
        }
    }

    fn test_parent_refinement_choice() -> ParentRefinementChoice {
        ParentRefinementChoice {
            assignments: vec![0, 0, 0, 0],
            counts: vec![4],
            n_clusters: 1,
            largest_fraction: 1.0,
            singleton_weight_fraction: 1.0,
            quotient_score: 0.0,
            candidate_delta_q: 0.0,
            source: None,
            trace_candidate_id: None,
            baseline_repair_merge_count: 0,
            baseline_repair_delta_sum: 0.0,
            adaptive_probe_baseline_delta_q: None,
            adaptive_probe_committed: false,
            adaptive_probe_score: f64::NEG_INFINITY,
            adaptive_probe_source_label: None,
        }
    }

    fn near_tie_summary(changed_decision_count: usize) -> local_merge::LocalMergeMarginSummary {
        let mut summary = local_merge::LocalMergeMarginSummary::default();
        summary.decision_count = 4;
        summary.low_margin_decision_count = changed_decision_count;
        summary.changed_decision_count = changed_decision_count;
        summary
    }

    fn local_shake_test_config(mode: AdaptiveLocalShakeMode) -> DongdaemunRefinementConfig {
        DongdaemunRefinementConfig {
            target_max_weight: 5.0,
            max_extra_parents_per_iteration: 1,
            max_extra_children_per_parent: 8,
            max_singleton_weight_fraction: 1.0,
            min_largest_child_fraction_improvement: 0.0,
            adaptive_local_shake_mode: mode,
            adaptive_local_shake_arms: vec![
                AdaptiveLocalShakeArm::NearTieRefinement,
                AdaptiveLocalShakeArm::ResolutionUp,
                AdaptiveLocalShakeArm::ResolutionDown,
                AdaptiveLocalShakeArm::SeedLocalRefinement,
            ],
            adaptive_local_shake_resolution_up_multipliers: vec![1.02],
            adaptive_local_shake_resolution_down_multipliers: vec![0.98],
            adaptive_local_shake_seed_perturbations: 1,
            adaptive_local_shake_near_tie_margin_parent_weight: 1e-4,
            adaptive_local_shake_near_tie_randomness: 0.05,
            ..DongdaemunRefinementConfig::default()
        }
    }

    fn local_shake_choice() -> ParentRefinementChoice {
        let mut choice = test_parent_refinement_choice();
        choice.assignments = vec![0, 0, 1, 1];
        choice.counts = vec![2, 2];
        choice.n_clusters = 2;
        choice.largest_fraction = 0.5;
        choice.singleton_weight_fraction = 0.0;
        choice.candidate_delta_q = 0.0;
        choice
    }

    #[test]
    fn test_local_shake_distinctness_detects_assignment_and_shape_changes() {
        let choice = local_shake_choice();
        assert!(!local_shake_candidate_distinct(
            &choice,
            &[0, 0, 1, 1],
            2,
            0.5,
            0.0,
            1e-12,
        ));
        assert!(local_shake_candidate_distinct(
            &choice,
            &[0, 1, 1, 0],
            2,
            0.5,
            0.0,
            1e-12,
        ));
        assert!(local_shake_candidate_distinct(
            &choice,
            &[0, 0, 1, 1],
            2,
            0.6,
            0.0,
            1e-12,
        ));
    }

    #[test]
    fn test_local_shake_arm_selector_is_deterministic_and_budgeted() {
        let mut config = local_shake_test_config(AdaptiveLocalShakeMode::TraceOnly);
        config.adaptive_local_shake_arm_priority = vec![
            AdaptiveLocalShakeArm::ResolutionUp,
            AdaptiveLocalShakeArm::NearTieRefinement,
            AdaptiveLocalShakeArm::SeedLocalRefinement,
        ];
        config.adaptive_local_shake_max_arms_per_parent = 2;
        config.adaptive_local_shake_seed_margin_count = 1;
        let summary = near_tie_summary(2);

        let specs = select_local_shake_arms(&config, 10.0, 0.96, 0.0, Some(&summary));

        let arms = specs.iter().map(|spec| spec.arm).collect::<Vec<_>>();
        assert_eq!(
            arms,
            vec![
                AdaptiveLocalShakeArm::ResolutionUp,
                AdaptiveLocalShakeArm::NearTieRefinement,
            ]
        );
        assert_eq!(specs[0].multiplier, 1.02);
        assert_eq!(specs[1].multiplier, 1.0);
    }

    #[test]
    fn test_local_shake_qf_replace_requires_distinct_valid_gain() {
        let config = local_shake_test_config(AdaptiveLocalShakeMode::QfReplace);
        let choice = local_shake_choice();
        let candidate = build_local_shake_candidate(
            LocalShakeArmSpec {
                arm: AdaptiveLocalShakeArm::ResolutionUp,
                arm_index: 0,
                priority_rank: 0,
                multiplier: 1.02,
                seed_index: 0,
            },
            0,
            &choice,
            vec![0, 1, 1, 0],
            vec![2, 2],
            2,
            0.4,
            0.0,
            0.1,
            0.6,
            10.0,
            &config,
            None,
        );

        assert!(local_shake_commit_eligible(&candidate, &config, 10.0));

        let same_assignment = build_local_shake_candidate(
            LocalShakeArmSpec {
                arm: AdaptiveLocalShakeArm::ResolutionUp,
                arm_index: 0,
                priority_rank: 0,
                multiplier: 1.02,
                seed_index: 0,
            },
            1,
            &choice,
            vec![0, 0, 1, 1],
            vec![2, 2],
            2,
            0.5,
            0.0,
            0.1,
            0.6,
            10.0,
            &config,
            None,
        );
        assert!(!local_shake_commit_eligible(
            &same_assignment,
            &config,
            10.0
        ));
    }

    #[test]
    fn test_local_shake_pressure_guard_blocks_pressure_regression() {
        let config = local_shake_test_config(AdaptiveLocalShakeMode::PressureGuarded);
        let choice = local_shake_choice();
        let candidate = build_local_shake_candidate(
            LocalShakeArmSpec {
                arm: AdaptiveLocalShakeArm::ResolutionDown,
                arm_index: 0,
                priority_rank: 0,
                multiplier: 0.98,
                seed_index: 0,
            },
            0,
            &choice,
            vec![0, 0, 0, 1],
            vec![3, 1],
            2,
            0.7,
            0.25,
            0.2,
            0.8,
            10.0,
            &config,
            None,
        );

        assert!(!local_shake_commit_eligible(&candidate, &config, 10.0));
        assert_eq!(
            local_shake_commit_block_reason(&candidate, &config, 10.0),
            "pressure_guard"
        );
    }

    fn test_candidate_trace_context() -> CandidateTraceContext {
        CandidateTraceContext {
            depth: 0,
            parent_id: 1,
            parent_visit_index: 1,
            parent_size: 4,
            parent_weight: 10.0,
            standard_n_clusters: 1,
            source_index: 0,
            gamma_multiplier: 1.0,
            repaired: false,
        }
    }

    #[test]
    fn test_near_tie_probe_replaces_existing_candidate_when_changed_and_guarded() {
        let ddm = DongdaemunRefinementConfig {
            target_max_weight: 2.0,
            max_extra_children_per_parent: 8,
            max_singleton_weight_fraction: 1.0,
            min_largest_child_fraction_improvement: 0.0,
            adaptive_near_tie_probe_mode: AdaptiveNearTieProbeMode::Candidate,
            ..DongdaemunRefinementConfig::default()
        };
        let mut choice = test_parent_refinement_choice();
        choice.source = Some(RefinementCandidateSource::SameGammaSeed);
        choice.n_clusters = 2;
        choice.largest_fraction = 0.6;
        choice.singleton_weight_fraction = 0.2;
        choice.candidate_delta_q = -2.0;

        maybe_apply_near_tie_probe_candidate(
            &mut choice,
            vec![0, 0, 1, 1],
            vec![2, 2],
            2,
            0.5,
            0.2,
            -1.0,
            0.8,
            &ddm,
            test_candidate_trace_context(),
            &near_tie_summary(1),
        );

        assert_eq!(
            choice.source,
            Some(RefinementCandidateSource::NearTieRefinementProbe)
        );
        assert_eq!(choice.candidate_delta_q, -1.0);
        assert!(choice.adaptive_probe_committed);
    }

    #[test]
    fn test_near_tie_probe_requires_changed_decision_for_commit() {
        let ddm = DongdaemunRefinementConfig {
            target_max_weight: 2.0,
            max_extra_children_per_parent: 8,
            max_singleton_weight_fraction: 1.0,
            min_largest_child_fraction_improvement: 0.0,
            adaptive_near_tie_probe_mode: AdaptiveNearTieProbeMode::Candidate,
            ..DongdaemunRefinementConfig::default()
        };
        let mut choice = test_parent_refinement_choice();

        maybe_apply_near_tie_probe_candidate(
            &mut choice,
            vec![0, 0, 1, 1],
            vec![2, 2],
            2,
            0.5,
            0.0,
            1.0,
            0.8,
            &ddm,
            test_candidate_trace_context(),
            &near_tie_summary(0),
        );

        assert_eq!(choice.source, None);
        assert_eq!(choice.candidate_delta_q, 0.0);
        assert!(!choice.adaptive_probe_committed);
    }

    #[test]
    fn test_near_tie_probe_allows_valid_quality_replacement() {
        let ddm = DongdaemunRefinementConfig {
            target_max_weight: 2.0,
            max_extra_children_per_parent: 8,
            max_singleton_weight_fraction: 1.0,
            min_largest_child_fraction_improvement: 0.0,
            adaptive_near_tie_probe_mode: AdaptiveNearTieProbeMode::Candidate,
            ..DongdaemunRefinementConfig::default()
        };
        let mut choice = test_parent_refinement_choice();
        choice.source = Some(RefinementCandidateSource::SameGammaSeed);
        choice.n_clusters = 2;
        choice.largest_fraction = 0.5;
        choice.singleton_weight_fraction = 0.0;
        choice.candidate_delta_q = -2.0;

        maybe_apply_near_tie_probe_candidate(
            &mut choice,
            vec![0, 0, 0, 1],
            vec![3, 1],
            2,
            0.7,
            0.0,
            -1.0,
            0.8,
            &ddm,
            test_candidate_trace_context(),
            &near_tie_summary(1),
        );

        assert_eq!(
            choice.source,
            Some(RefinementCandidateSource::NearTieRefinementProbe)
        );
        assert_eq!(choice.candidate_delta_q, -1.0);
        assert!(choice.adaptive_probe_committed);
    }

    #[test]
    fn test_adaptive_probe_conservative_apply_respects_margin_and_budget() {
        reset_adaptive_probe_state();
        let ddm = DongdaemunRefinementConfig {
            target_max_weight: 2.0,
            max_extra_parents_per_iteration: 1,
            max_extra_children_per_parent: 8,
            max_singleton_weight_fraction: 1.0,
            min_largest_child_fraction_improvement: 0.0,
            candidate_quality_policy: CandidateQualityPolicy::QualityFirst,
            adaptive_probe_mode: AdaptiveProbeMode::ConservativeApply,
            adaptive_probe_commit_min_gain_parent_weight: 0.5,
            adaptive_probe_max_commits_total: 1,
            adaptive_probe_max_commits_per_depth: 1,
            ..DongdaemunRefinementConfig::default()
        };
        let trace_context = CandidateTraceContext {
            depth: 2,
            parent_id: 7,
            parent_visit_index: 1,
            parent_size: 4,
            parent_weight: 10.0,
            standard_n_clusters: 1,
            source_index: 0,
            gamma_multiplier: 1.0,
            repaired: false,
        };
        let mut choice = test_parent_refinement_choice();

        maybe_apply_adaptive_probe_candidate(
            &mut choice,
            vec![0, 0, 1, 1],
            vec![2, 2],
            2,
            0.5,
            0.0,
            4.0,
            1.0,
            &ddm,
            trace_context,
            "same_gamma_probe",
            0,
        );
        assert_eq!(choice.n_clusters, 1);
        assert_eq!(adaptive_probe_commit_counts(2), (0, 0));

        maybe_apply_adaptive_probe_candidate(
            &mut choice,
            vec![0, 0, 1, 1],
            vec![2, 2],
            2,
            0.5,
            0.0,
            6.0,
            1.0,
            &ddm,
            trace_context,
            "same_gamma_probe",
            1,
        );
        assert_eq!(choice.n_clusters, 2);
        assert_eq!(choice.candidate_delta_q, 6.0);
        assert_eq!(adaptive_probe_commit_counts(2), (1, 1));

        maybe_apply_adaptive_probe_candidate(
            &mut choice,
            vec![0, 1, 2, 2],
            vec![1, 1, 2],
            3,
            0.5,
            0.0,
            20.0,
            1.0,
            &ddm,
            trace_context,
            "same_gamma_probe",
            2,
        );
        assert_eq!(choice.n_clusters, 2);
        assert_eq!(choice.candidate_delta_q, 6.0);
        assert_eq!(adaptive_probe_commit_counts(2), (1, 1));
    }

    #[test]
    fn test_adaptive_probe_conservative_apply_filters_sources() {
        reset_adaptive_probe_state();
        let ddm = DongdaemunRefinementConfig {
            target_max_weight: 2.0,
            max_extra_parents_per_iteration: 1,
            max_extra_children_per_parent: 8,
            max_singleton_weight_fraction: 1.0,
            min_largest_child_fraction_improvement: 0.0,
            candidate_quality_policy: CandidateQualityPolicy::QualityFirst,
            adaptive_probe_mode: AdaptiveProbeMode::ConservativeApply,
            adaptive_probe_commit_sources: vec!["node_order_control".to_string()],
            ..DongdaemunRefinementConfig::default()
        };
        let trace_context = CandidateTraceContext {
            depth: 1,
            parent_id: 3,
            parent_visit_index: 1,
            parent_size: 4,
            parent_weight: 10.0,
            standard_n_clusters: 1,
            source_index: 0,
            gamma_multiplier: 1.0,
            repaired: false,
        };
        let mut choice = test_parent_refinement_choice();

        maybe_apply_adaptive_probe_candidate(
            &mut choice,
            vec![0, 0, 1, 1],
            vec![2, 2],
            2,
            0.5,
            0.0,
            10.0,
            1.0,
            &ddm,
            trace_context,
            "same_gamma_probe",
            0,
        );
        assert_eq!(choice.n_clusters, 1);
        assert_eq!(adaptive_probe_commit_counts(1), (0, 0));

        maybe_apply_adaptive_probe_candidate(
            &mut choice,
            vec![0, 0, 1, 1],
            vec![2, 2],
            2,
            0.5,
            0.0,
            10.0,
            1.0,
            &ddm,
            trace_context,
            "node_order_control",
            1,
        );
        assert_eq!(choice.n_clusters, 2);
        assert_eq!(adaptive_probe_commit_counts(1), (1, 1));
    }

    #[test]
    fn test_adaptive_probe_best_qf_replaces_staged_commit_without_extra_budget() {
        reset_adaptive_probe_state();
        let ddm = DongdaemunRefinementConfig {
            target_max_weight: 2.0,
            max_extra_parents_per_iteration: 1,
            max_extra_children_per_parent: 8,
            max_singleton_weight_fraction: 1.0,
            min_largest_child_fraction_improvement: 0.0,
            candidate_quality_policy: CandidateQualityPolicy::QualityFirst,
            adaptive_probe_mode: AdaptiveProbeMode::ConservativeApply,
            adaptive_probe_commit_strategy: AdaptiveProbeCommitStrategy::BestQf,
            adaptive_probe_max_commits_total: 1,
            adaptive_probe_max_commits_per_depth: 1,
            ..DongdaemunRefinementConfig::default()
        };
        let trace_context = CandidateTraceContext {
            depth: 1,
            parent_id: 4,
            parent_visit_index: 1,
            parent_size: 4,
            parent_weight: 10.0,
            standard_n_clusters: 1,
            source_index: 0,
            gamma_multiplier: 1.0,
            repaired: false,
        };
        let mut choice = test_parent_refinement_choice();

        maybe_apply_adaptive_probe_candidate(
            &mut choice,
            vec![0, 0, 1, 1],
            vec![2, 2],
            2,
            0.5,
            0.0,
            6.0,
            1.0,
            &ddm,
            trace_context,
            "same_gamma_probe",
            0,
        );
        assert_eq!(choice.n_clusters, 2);
        assert_eq!(choice.candidate_delta_q, 6.0);
        assert_eq!(adaptive_probe_commit_counts(1), (1, 1));

        maybe_apply_adaptive_probe_candidate(
            &mut choice,
            vec![0, 1, 2, 2],
            vec![1, 1, 2],
            3,
            0.5,
            0.0,
            20.0,
            1.0,
            &ddm,
            trace_context,
            "same_gamma_probe",
            1,
        );
        assert_eq!(choice.n_clusters, 3);
        assert_eq!(choice.candidate_delta_q, 20.0);
        assert_eq!(adaptive_probe_commit_counts(1), (1, 1));
    }

    #[test]
    fn test_adaptive_probe_commit_source_validation_rejects_unknown_source() {
        let ddm = DongdaemunRefinementConfig {
            target_max_weight: 2.0,
            max_extra_parents_per_iteration: 1,
            adaptive_probe_commit_sources: vec!["bad_source".to_string()],
            ..DongdaemunRefinementConfig::default()
        };
        let err = ddm.validate().unwrap_err();
        assert!(err.contains("adaptive_probe_commit_sources"));
    }

    #[test]
    fn test_dongdaemun_refinement_candidate_source_selection_uses_structural_order() {
        let ddm = DongdaemunRefinementConfig {
            target_max_weight: 2.0,
            max_extra_parents_per_iteration: 1,
            max_extra_children_per_parent: 8,
            max_singleton_weight_fraction: 1.0,
            min_largest_child_fraction_improvement: 0.0,
            ..DongdaemunRefinementConfig::default()
        };
        let mut stats = RefinementDongdaemunStats::default();
        let mut choice = ParentRefinementChoice {
            assignments: vec![0, 0, 0, 0],
            counts: vec![4],
            n_clusters: 1,
            largest_fraction: 1.0,
            singleton_weight_fraction: 1.0,
            quotient_score: 0.0,
            candidate_delta_q: 0.0,
            source: None,
            trace_candidate_id: None,
            baseline_repair_merge_count: 0,
            baseline_repair_delta_sum: 0.0,
            adaptive_probe_baseline_delta_q: None,
            adaptive_probe_committed: false,
            adaptive_probe_score: f64::NEG_INFINITY,
            adaptive_probe_source_label: None,
        };

        consider_parent_candidate(
            &mut choice,
            vec![0, 0, 1, 1],
            vec![2, 2],
            2,
            0.5,
            0.0,
            None,
            0,
            0.0,
            0.0,
            1.0,
            RefinementCandidateSource::SameGammaSeed,
            &ddm,
            &mut stats,
        );
        consider_parent_candidate(
            &mut choice,
            vec![0, 1, 1, 2],
            vec![1, 2, 1],
            3,
            0.4,
            0.5,
            None,
            0,
            0.0,
            0.0,
            1.0,
            RefinementCandidateSource::HighGamma,
            &ddm,
            &mut stats,
        );

        assert_eq!(stats.same_gamma_candidates, 1);
        assert_eq!(stats.high_gamma_candidates, 1);
        assert_eq!(choice.source, Some(RefinementCandidateSource::HighGamma));
        assert_eq!(choice.n_clusters, 3);
        assert_eq!(choice.largest_fraction, 0.4);
    }

    #[test]
    fn test_dongdaemun_refinement_quality_diagnostics_count_positive_and_negative_candidates() {
        let ddm = DongdaemunRefinementConfig {
            target_max_weight: 2.0,
            max_extra_parents_per_iteration: 1,
            max_singleton_weight_fraction: 1.0,
            min_largest_child_fraction_improvement: 0.0,
            ..DongdaemunRefinementConfig::default()
        };
        let mut stats = RefinementDongdaemunStats::default();
        let mut choice = ParentRefinementChoice {
            assignments: vec![0, 0, 0, 0],
            counts: vec![4],
            n_clusters: 1,
            largest_fraction: 1.0,
            singleton_weight_fraction: 1.0,
            quotient_score: 0.0,
            candidate_delta_q: 0.0,
            source: None,
            trace_candidate_id: None,
            baseline_repair_merge_count: 0,
            baseline_repair_delta_sum: 0.0,
            adaptive_probe_baseline_delta_q: None,
            adaptive_probe_committed: false,
            adaptive_probe_score: f64::NEG_INFINITY,
            adaptive_probe_source_label: None,
        };

        consider_parent_candidate(
            &mut choice,
            vec![0, 0, 1, 1],
            vec![2, 2],
            2,
            0.5,
            0.0,
            None,
            0,
            0.0,
            -0.25,
            1.0,
            RefinementCandidateSource::SameGammaSeed,
            &ddm,
            &mut stats,
        );
        consider_parent_candidate(
            &mut choice,
            vec![0, 1, 1, 2],
            vec![1, 2, 1],
            3,
            0.4,
            0.5,
            None,
            0,
            0.0,
            0.75,
            1.0,
            RefinementCandidateSource::HighGamma,
            &ddm,
            &mut stats,
        );

        assert_eq!(stats.candidate_quality_delta_sum, 0.5);
        assert_eq!(stats.candidate_positive_quality_delta, 1);
        assert_eq!(stats.same_gamma_quality_delta_sum, -0.25);
        assert_eq!(stats.high_gamma_quality_delta_sum, 0.75);
        assert_eq!(stats.high_gamma_positive_quality_delta, 1);
    }

    #[test]
    fn test_dongdaemun_refinement_qpos_spos_candidate_profiles_as_true_positive() {
        let ddm = DongdaemunRefinementConfig {
            target_max_weight: 2.0,
            max_extra_parents_per_iteration: 1,
            max_singleton_weight_fraction: 1.0,
            min_largest_child_fraction_improvement: 0.0,
            ..DongdaemunRefinementConfig::default()
        };
        let mut stats = RefinementDongdaemunStats::default();
        let mut choice = ParentRefinementChoice {
            assignments: vec![0, 0, 0, 0],
            counts: vec![4],
            n_clusters: 1,
            largest_fraction: 1.0,
            singleton_weight_fraction: 1.0,
            quotient_score: 0.0,
            candidate_delta_q: 0.0,
            source: None,
            trace_candidate_id: None,
            baseline_repair_merge_count: 0,
            baseline_repair_delta_sum: 0.0,
            adaptive_probe_baseline_delta_q: None,
            adaptive_probe_committed: false,
            adaptive_probe_score: f64::NEG_INFINITY,
            adaptive_probe_source_label: None,
        };

        consider_parent_candidate(
            &mut choice,
            vec![0, 0, 1, 1],
            vec![2, 2],
            2,
            0.5,
            0.0,
            None,
            0,
            0.0,
            0.25,
            1.0,
            RefinementCandidateSource::SameGammaSeed,
            &ddm,
            &mut stats,
        );
        record_test_final_candidate_decision(&mut stats, &choice, 1.0);

        assert_eq!(stats.candidate_qpos_spos, 1);
        assert_eq!(stats.same_gamma_qpos_spos, 1);
        assert_eq!(stats.candidate_valid, 1);
        assert_eq!(stats.candidate_true_positive, 1);
        assert_eq!(stats.candidate_false_positive, 0);
        assert_eq!(stats.candidate_false_negative, 0);
    }

    #[test]
    fn test_dongdaemun_refinement_qpos_sneg_candidate_profiles_as_false_positive() {
        let ddm = DongdaemunRefinementConfig {
            target_max_weight: 2.0,
            max_extra_parents_per_iteration: 1,
            max_singleton_weight_fraction: 1.0,
            min_largest_child_fraction_improvement: 0.0,
            ..DongdaemunRefinementConfig::default()
        };
        let mut stats = RefinementDongdaemunStats::default();
        let mut choice = ParentRefinementChoice {
            assignments: vec![0, 0, 0, 0],
            counts: vec![4],
            n_clusters: 1,
            largest_fraction: 1.0,
            singleton_weight_fraction: 1.0,
            quotient_score: 0.0,
            candidate_delta_q: 0.0,
            source: None,
            trace_candidate_id: None,
            baseline_repair_merge_count: 0,
            baseline_repair_delta_sum: 0.0,
            adaptive_probe_baseline_delta_q: None,
            adaptive_probe_committed: false,
            adaptive_probe_score: f64::NEG_INFINITY,
            adaptive_probe_source_label: None,
        };

        consider_parent_candidate(
            &mut choice,
            vec![0, 0, 0, 0],
            vec![4, 0],
            2,
            1.0,
            0.0,
            None,
            0,
            0.0,
            0.25,
            1.0,
            RefinementCandidateSource::HighGamma,
            &ddm,
            &mut stats,
        );
        record_test_final_candidate_decision(&mut stats, &choice, 1.0);

        assert_eq!(stats.candidate_qpos_sneg, 1);
        assert_eq!(stats.high_gamma_qpos_sneg, 1);
        assert_eq!(stats.candidate_valid, 1);
        assert_eq!(stats.candidate_true_positive, 0);
        assert_eq!(stats.candidate_false_positive, 1);
    }

    #[test]
    fn test_dongdaemun_refinement_qneg_spos_candidate_profiles_as_false_positive() {
        let ddm = DongdaemunRefinementConfig {
            target_max_weight: 2.0,
            max_extra_parents_per_iteration: 1,
            max_singleton_weight_fraction: 1.0,
            min_largest_child_fraction_improvement: 0.0,
            ..DongdaemunRefinementConfig::default()
        };
        let mut stats = RefinementDongdaemunStats::default();
        let mut choice = ParentRefinementChoice {
            assignments: vec![0, 0, 0, 0],
            counts: vec![4],
            n_clusters: 1,
            largest_fraction: 1.0,
            singleton_weight_fraction: 1.0,
            quotient_score: 0.0,
            candidate_delta_q: 0.0,
            source: None,
            trace_candidate_id: None,
            baseline_repair_merge_count: 0,
            baseline_repair_delta_sum: 0.0,
            adaptive_probe_baseline_delta_q: None,
            adaptive_probe_committed: false,
            adaptive_probe_score: f64::NEG_INFINITY,
            adaptive_probe_source_label: None,
        };

        consider_parent_candidate(
            &mut choice,
            vec![0, 0, 1, 1],
            vec![2, 2],
            2,
            0.5,
            0.0,
            None,
            0,
            0.0,
            -0.25,
            1.0,
            RefinementCandidateSource::SameGammaSeed,
            &ddm,
            &mut stats,
        );
        record_test_final_candidate_decision(&mut stats, &choice, 1.0);

        assert_eq!(stats.candidate_qneg_spos, 1);
        assert_eq!(stats.same_gamma_qneg_spos, 1);
        assert_eq!(stats.candidate_valid, 1);
        assert_eq!(stats.candidate_true_positive, 0);
        assert_eq!(stats.candidate_false_positive, 1);
    }

    #[test]
    fn test_dongdaemun_refinement_quality_floor_rejects_negative_delta_candidate() {
        let ddm = DongdaemunRefinementConfig {
            target_max_weight: 2.0,
            max_extra_parents_per_iteration: 1,
            max_singleton_weight_fraction: 1.0,
            min_largest_child_fraction_improvement: 0.0,
            candidate_quality_policy: CandidateQualityPolicy::QualityFloor,
            min_candidate_delta_q: 0.0,
            ..DongdaemunRefinementConfig::default()
        };
        let mut stats = RefinementDongdaemunStats::default();
        let mut choice = ParentRefinementChoice {
            assignments: vec![0, 0, 0, 0],
            counts: vec![4],
            n_clusters: 1,
            largest_fraction: 1.0,
            singleton_weight_fraction: 1.0,
            quotient_score: 0.0,
            candidate_delta_q: 0.0,
            source: None,
            trace_candidate_id: None,
            baseline_repair_merge_count: 0,
            baseline_repair_delta_sum: 0.0,
            adaptive_probe_baseline_delta_q: None,
            adaptive_probe_committed: false,
            adaptive_probe_score: f64::NEG_INFINITY,
            adaptive_probe_source_label: None,
        };

        consider_parent_candidate(
            &mut choice,
            vec![0, 0, 1, 1],
            vec![2, 2],
            2,
            0.5,
            0.0,
            None,
            0,
            0.0,
            -0.1,
            1.0,
            RefinementCandidateSource::SameGammaSeed,
            &ddm,
            &mut stats,
        );
        consider_parent_candidate(
            &mut choice,
            vec![0, 1, 1, 2],
            vec![1, 2, 1],
            3,
            0.4,
            0.5,
            None,
            0,
            0.0,
            0.2,
            1.0,
            RefinementCandidateSource::HighGamma,
            &ddm,
            &mut stats,
        );
        record_test_final_candidate_decision(&mut stats, &choice, 1.0);

        assert_eq!(stats.candidate_qneg_spos, 1);
        assert_eq!(stats.candidate_qpos_spos, 1);
        assert_eq!(stats.candidate_rejected_by_quality, 1);
        assert_eq!(stats.same_gamma_rejected_by_quality, 1);
        assert_eq!(stats.candidate_true_positive, 1);
        assert_eq!(stats.candidate_true_negative, 1);
        assert_eq!(choice.source, Some(RefinementCandidateSource::HighGamma));
        assert_eq!(choice.candidate_delta_q, 0.2);
    }

    #[test]
    fn test_dongdaemun_refinement_quality_guarded_structural_keeps_structural_order_after_floor() {
        let ddm = DongdaemunRefinementConfig {
            target_max_weight: 2.0,
            max_extra_parents_per_iteration: 1,
            max_singleton_weight_fraction: 1.0,
            min_largest_child_fraction_improvement: 0.0,
            candidate_quality_policy: CandidateQualityPolicy::QualityGuardedStructural,
            min_candidate_delta_q: 0.0,
            ..DongdaemunRefinementConfig::default()
        };
        let mut stats = RefinementDongdaemunStats::default();
        let mut choice = ParentRefinementChoice {
            assignments: vec![0, 0, 0, 0],
            counts: vec![4],
            n_clusters: 1,
            largest_fraction: 1.0,
            singleton_weight_fraction: 1.0,
            quotient_score: 0.0,
            candidate_delta_q: 0.0,
            source: None,
            trace_candidate_id: None,
            baseline_repair_merge_count: 0,
            baseline_repair_delta_sum: 0.0,
            adaptive_probe_baseline_delta_q: None,
            adaptive_probe_committed: false,
            adaptive_probe_score: f64::NEG_INFINITY,
            adaptive_probe_source_label: None,
        };

        consider_parent_candidate(
            &mut choice,
            vec![0, 1, 2, 2],
            vec![1, 1, 2],
            3,
            0.5,
            0.5,
            None,
            0,
            0.0,
            -0.1,
            1.0,
            RefinementCandidateSource::SameGammaSeed,
            &ddm,
            &mut stats,
        );
        consider_parent_candidate(
            &mut choice,
            vec![0, 0, 1, 1],
            vec![2, 2],
            2,
            0.6,
            0.0,
            None,
            0,
            0.0,
            0.0,
            1.0,
            RefinementCandidateSource::HighGamma,
            &ddm,
            &mut stats,
        );

        assert_eq!(stats.candidate_rejected_by_quality, 1);
        assert_eq!(stats.same_gamma_rejected_by_quality, 1);
        assert_eq!(choice.source, Some(RefinementCandidateSource::HighGamma));
        assert_eq!(choice.candidate_delta_q, 0.0);
        assert_eq!(choice.largest_fraction, 0.6);
    }

    #[test]
    fn test_dongdaemun_refinement_quality_first_prefers_objective_best_candidate() {
        let ddm = DongdaemunRefinementConfig {
            target_max_weight: 2.0,
            max_extra_parents_per_iteration: 1,
            max_singleton_weight_fraction: 1.0,
            min_largest_child_fraction_improvement: 0.0,
            candidate_quality_policy: CandidateQualityPolicy::QualityFirst,
            min_candidate_delta_q: 0.0,
            ..DongdaemunRefinementConfig::default()
        };
        let mut stats = RefinementDongdaemunStats::default();
        let mut choice = ParentRefinementChoice {
            assignments: vec![0, 0, 0, 0],
            counts: vec![4],
            n_clusters: 1,
            largest_fraction: 1.0,
            singleton_weight_fraction: 1.0,
            quotient_score: 0.0,
            candidate_delta_q: 0.0,
            source: None,
            trace_candidate_id: None,
            baseline_repair_merge_count: 0,
            baseline_repair_delta_sum: 0.0,
            adaptive_probe_baseline_delta_q: None,
            adaptive_probe_committed: false,
            adaptive_probe_score: f64::NEG_INFINITY,
            adaptive_probe_source_label: None,
        };

        consider_parent_candidate(
            &mut choice,
            vec![0, 1, 2, 2],
            vec![1, 1, 2],
            3,
            0.5,
            0.5,
            None,
            0,
            0.0,
            0.1,
            1.0,
            RefinementCandidateSource::SameGammaSeed,
            &ddm,
            &mut stats,
        );
        consider_parent_candidate(
            &mut choice,
            vec![0, 0, 0, 0],
            vec![4, 0],
            2,
            1.0,
            0.0,
            None,
            0,
            0.0,
            0.2,
            1.0,
            RefinementCandidateSource::HighGamma,
            &ddm,
            &mut stats,
        );
        record_test_final_candidate_decision(&mut stats, &choice, 1.0);

        assert_eq!(choice.source, Some(RefinementCandidateSource::HighGamma));
        assert_eq!(choice.candidate_delta_q, 0.2);
        assert_eq!(choice.largest_fraction, 1.0);
        assert_eq!(stats.candidate_qpos_spos, 1);
        assert_eq!(stats.candidate_qpos_sneg, 1);
        assert_eq!(stats.candidate_rejected_by_policy, 1);
        assert_eq!(stats.candidate_true_positive, 0);
        assert_eq!(stats.candidate_false_positive, 1);
        assert_eq!(stats.candidate_false_negative, 1);
    }

    #[test]
    fn test_dongdaemun_refinement_selective_accepts_raw_structural_improvement() {
        let ddm = DongdaemunRefinementConfig {
            target_max_weight: 2.0,
            max_extra_parents_per_iteration: 1,
            max_singleton_weight_fraction: 1.0,
            min_largest_child_fraction_improvement: 0.05,
            candidate_quality_policy: CandidateQualityPolicy::Selective,
            min_candidate_delta_q: 0.0,
            ..DongdaemunRefinementConfig::default()
        };
        let mut stats = RefinementDongdaemunStats::default();
        let mut choice = test_parent_refinement_choice();

        consider_parent_candidate(
            &mut choice,
            vec![0, 0, 0, 1],
            vec![3, 1],
            2,
            0.98,
            0.0,
            None,
            0,
            0.0,
            0.25,
            1.0,
            RefinementCandidateSource::SameGammaSeed,
            &ddm,
            &mut stats,
        );
        record_test_final_candidate_decision(&mut stats, &choice, 1.0);

        assert_eq!(
            choice.source,
            Some(RefinementCandidateSource::SameGammaSeed)
        );
        assert_eq!(choice.candidate_delta_q, 0.25);
        assert_eq!(choice.largest_fraction, 0.98);
        assert_eq!(stats.candidate_valid, 1);
        assert_eq!(stats.candidate_qpos_spos, 1);
        assert_eq!(stats.candidate_true_positive, 1);
    }

    #[test]
    fn test_dongdaemun_refinement_pressure_aware_uses_quality_loss_floor_and_pressure_order() {
        let ddm = DongdaemunRefinementConfig {
            target_max_weight: 2.0,
            max_extra_parents_per_iteration: 1,
            max_singleton_weight_fraction: 1.0,
            min_largest_child_fraction_improvement: 0.05,
            candidate_quality_policy: CandidateQualityPolicy::PressureAware,
            min_candidate_delta_q: -0.2,
            ..DongdaemunRefinementConfig::default()
        };
        let mut stats = RefinementDongdaemunStats::default();
        let mut choice = test_parent_refinement_choice();

        consider_parent_candidate(
            &mut choice,
            vec![0, 0, 0, 1],
            vec![3, 1],
            2,
            0.98,
            0.0,
            None,
            0,
            0.0,
            -0.1,
            1.0,
            RefinementCandidateSource::SameGammaSeed,
            &ddm,
            &mut stats,
        );
        consider_parent_candidate(
            &mut choice,
            vec![0, 0, 1, 1],
            vec![2, 2],
            2,
            0.5,
            0.0,
            None,
            0,
            0.0,
            -0.15,
            1.0,
            RefinementCandidateSource::HighGamma,
            &ddm,
            &mut stats,
        );
        consider_parent_candidate(
            &mut choice,
            vec![0, 1, 2, 3],
            vec![1, 1, 1, 1],
            4,
            0.25,
            1.0,
            None,
            0,
            0.0,
            -0.25,
            1.0,
            RefinementCandidateSource::HighGamma,
            &ddm,
            &mut stats,
        );

        assert_eq!(stats.candidate_valid, 3);
        assert_eq!(stats.candidate_rejected_by_quality, 1);
        assert_eq!(choice.source, Some(RefinementCandidateSource::HighGamma));
        assert_eq!(choice.largest_fraction, 0.5);
        assert_eq!(choice.candidate_delta_q, -0.15);
    }

    #[test]
    fn test_dongdaemun_refinement_adaptive_plateau_zero_band_prioritizes_quality() {
        let ddm = DongdaemunRefinementConfig {
            target_max_weight: 2.0,
            max_extra_parents_per_iteration: 1,
            max_singleton_weight_fraction: 1.0,
            min_largest_child_fraction_improvement: 0.05,
            candidate_quality_policy: CandidateQualityPolicy::AdaptivePlateau,
            min_candidate_delta_q: -1.0,
            adaptive_plateau_quality_band: 0.0,
            ..DongdaemunRefinementConfig::default()
        };
        let mut stats = RefinementDongdaemunStats::default();
        let mut choice = test_parent_refinement_choice();

        consider_parent_candidate(
            &mut choice,
            vec![0, 0, 1, 1],
            vec![2, 2],
            2,
            0.4,
            0.0,
            None,
            0,
            0.0,
            0.1,
            1.0,
            RefinementCandidateSource::SameGammaSeed,
            &ddm,
            &mut stats,
        );
        consider_parent_candidate(
            &mut choice,
            vec![0, 0, 0, 1],
            vec![3, 1],
            2,
            0.9,
            0.9,
            None,
            0,
            0.0,
            0.2,
            1.0,
            RefinementCandidateSource::HighGamma,
            &ddm,
            &mut stats,
        );

        assert_eq!(choice.source, Some(RefinementCandidateSource::HighGamma));
        assert_eq!(choice.candidate_delta_q, 0.2);
        assert_eq!(choice.largest_fraction, 0.9);
    }

    #[test]
    fn test_dongdaemun_refinement_adaptive_plateau_uses_diagnostics_within_band() {
        let ddm = DongdaemunRefinementConfig {
            target_max_weight: 2.0,
            max_extra_parents_per_iteration: 1,
            max_singleton_weight_fraction: 1.0,
            min_largest_child_fraction_improvement: 0.05,
            candidate_quality_policy: CandidateQualityPolicy::AdaptivePlateau,
            min_candidate_delta_q: -1.0,
            adaptive_plateau_quality_band: 1.0,
            ..DongdaemunRefinementConfig::default()
        };
        let mut stats = RefinementDongdaemunStats::default();
        let mut choice = test_parent_refinement_choice();

        consider_parent_candidate(
            &mut choice,
            vec![0, 0, 0, 1],
            vec![3, 1],
            2,
            0.9,
            0.9,
            None,
            0,
            0.0,
            0.5,
            1.0,
            RefinementCandidateSource::SameGammaSeed,
            &ddm,
            &mut stats,
        );
        consider_parent_candidate(
            &mut choice,
            vec![0, 0, 1, 1],
            vec![2, 2],
            2,
            0.4,
            0.0,
            None,
            0,
            0.0,
            0.0,
            1.0,
            RefinementCandidateSource::HighGamma,
            &ddm,
            &mut stats,
        );

        assert_eq!(choice.source, Some(RefinementCandidateSource::HighGamma));
        assert_eq!(choice.candidate_delta_q, 0.0);
        assert_eq!(choice.largest_fraction, 0.4);
        assert_eq!(stats.candidate_rejected_by_policy, 1);
    }

    #[test]
    fn test_dongdaemun_refinement_adaptive_plateau_enforces_quality_floor() {
        let ddm = DongdaemunRefinementConfig {
            target_max_weight: 2.0,
            max_extra_parents_per_iteration: 1,
            max_singleton_weight_fraction: 1.0,
            min_largest_child_fraction_improvement: 0.05,
            candidate_quality_policy: CandidateQualityPolicy::AdaptivePlateau,
            min_candidate_delta_q: -0.1,
            adaptive_plateau_quality_band: 10.0,
            ..DongdaemunRefinementConfig::default()
        };
        let mut stats = RefinementDongdaemunStats::default();
        let mut choice = test_parent_refinement_choice();

        consider_parent_candidate(
            &mut choice,
            vec![0, 0, 1, 1],
            vec![2, 2],
            2,
            0.25,
            0.0,
            None,
            0,
            0.0,
            -0.2,
            1.0,
            RefinementCandidateSource::SameGammaSeed,
            &ddm,
            &mut stats,
        );

        assert_eq!(choice.source, None);
        assert_eq!(stats.candidate_valid, 1);
        assert_eq!(stats.candidate_rejected_by_quality, 1);
    }

    #[test]
    fn test_dongdaemun_refinement_adaptive_plateau_penalizes_singletons() {
        let ddm = DongdaemunRefinementConfig {
            target_max_weight: 2.0,
            max_extra_parents_per_iteration: 1,
            max_singleton_weight_fraction: 1.0,
            min_largest_child_fraction_improvement: 0.05,
            candidate_quality_policy: CandidateQualityPolicy::AdaptivePlateau,
            min_candidate_delta_q: -1.0,
            adaptive_plateau_quality_band: 1.0,
            ..DongdaemunRefinementConfig::default()
        };
        let mut stats = RefinementDongdaemunStats::default();
        let mut choice = test_parent_refinement_choice();
        let trace_context = CandidateTraceContext {
            parent_weight: 4.0,
            ..CandidateTraceContext::default()
        };

        consider_parent_candidate_with_trace(
            &mut choice,
            vec![0, 1, 2, 3],
            vec![1, 1, 1, 1],
            4,
            0.25,
            1.0,
            None,
            0,
            0.0,
            0.0,
            1.0,
            RefinementCandidateSource::SameGammaSeed,
            &ddm,
            &mut stats,
            trace_context,
        );
        consider_parent_candidate_with_trace(
            &mut choice,
            vec![0, 0, 1, 1],
            vec![2, 2],
            2,
            0.6,
            0.0,
            None,
            0,
            0.0,
            0.0,
            1.0,
            RefinementCandidateSource::HighGamma,
            &ddm,
            &mut stats,
            trace_context,
        );

        assert_eq!(choice.source, Some(RefinementCandidateSource::HighGamma));
        assert_eq!(choice.largest_fraction, 0.6);
        assert_eq!(choice.singleton_weight_fraction, 0.0);
    }

    #[test]
    fn test_dongdaemun_refinement_selective_rejects_positive_quality_without_structure() {
        let ddm = DongdaemunRefinementConfig {
            target_max_weight: 2.0,
            max_extra_parents_per_iteration: 1,
            max_singleton_weight_fraction: 1.0,
            min_largest_child_fraction_improvement: 0.0,
            candidate_quality_policy: CandidateQualityPolicy::Selective,
            min_candidate_delta_q: 0.0,
            ..DongdaemunRefinementConfig::default()
        };
        let mut stats = RefinementDongdaemunStats::default();
        let mut choice = test_parent_refinement_choice();

        consider_parent_candidate(
            &mut choice,
            vec![0, 0, 0, 0],
            vec![4, 0],
            2,
            1.0,
            0.0,
            None,
            0,
            0.0,
            0.25,
            1.0,
            RefinementCandidateSource::HighGamma,
            &ddm,
            &mut stats,
        );

        assert_eq!(choice.source, None);
        assert_eq!(stats.candidate_qpos_sneg, 1);
        assert_eq!(stats.candidate_valid, 0);
        assert_eq!(stats.candidate_invalid, 1);
    }

    #[test]
    fn test_dongdaemun_refinement_selective_rejects_negative_quality_with_structure() {
        let ddm = DongdaemunRefinementConfig {
            target_max_weight: 2.0,
            max_extra_parents_per_iteration: 1,
            max_singleton_weight_fraction: 1.0,
            min_largest_child_fraction_improvement: 0.0,
            candidate_quality_policy: CandidateQualityPolicy::Selective,
            min_candidate_delta_q: 0.0,
            ..DongdaemunRefinementConfig::default()
        };
        let mut stats = RefinementDongdaemunStats::default();
        let mut choice = test_parent_refinement_choice();

        consider_parent_candidate(
            &mut choice,
            vec![0, 0, 1, 1],
            vec![2, 2],
            2,
            0.5,
            0.0,
            None,
            0,
            0.0,
            -0.25,
            1.0,
            RefinementCandidateSource::SameGammaSeed,
            &ddm,
            &mut stats,
        );

        assert_eq!(choice.source, None);
        assert_eq!(stats.candidate_qneg_spos, 1);
        assert_eq!(stats.candidate_valid, 1);
        assert_eq!(stats.candidate_rejected_by_quality, 1);
        assert_eq!(stats.candidate_true_negative, 1);
    }

    #[test]
    fn test_dongdaemun_refinement_selective_rejects_zero_quality_with_structure() {
        let ddm = DongdaemunRefinementConfig {
            target_max_weight: 2.0,
            max_extra_parents_per_iteration: 1,
            max_singleton_weight_fraction: 1.0,
            min_largest_child_fraction_improvement: 0.0,
            candidate_quality_policy: CandidateQualityPolicy::Selective,
            min_candidate_delta_q: 0.0,
            ..DongdaemunRefinementConfig::default()
        };
        let mut stats = RefinementDongdaemunStats::default();
        let mut choice = test_parent_refinement_choice();

        consider_parent_candidate(
            &mut choice,
            vec![0, 0, 1, 1],
            vec![2, 2],
            2,
            0.5,
            0.0,
            None,
            0,
            0.0,
            0.0,
            1.0,
            RefinementCandidateSource::SameGammaSeed,
            &ddm,
            &mut stats,
        );

        assert_eq!(choice.source, None);
        assert_eq!(stats.candidate_qneg_spos, 1);
        assert_eq!(stats.candidate_valid, 1);
        assert_eq!(stats.candidate_rejected_by_quality, 1);
        assert_eq!(stats.candidate_true_negative, 1);
    }

    #[test]
    fn test_dongdaemun_refinement_selective_prefers_quality_then_structure() {
        let ddm = DongdaemunRefinementConfig {
            target_max_weight: 2.0,
            max_extra_parents_per_iteration: 1,
            max_singleton_weight_fraction: 1.0,
            min_largest_child_fraction_improvement: 0.0,
            candidate_quality_policy: CandidateQualityPolicy::Selective,
            min_candidate_delta_q: 0.0,
            ..DongdaemunRefinementConfig::default()
        };
        let mut stats = RefinementDongdaemunStats::default();
        let mut choice = test_parent_refinement_choice();

        consider_parent_candidate(
            &mut choice,
            vec![0, 0, 1, 1],
            vec![2, 2],
            2,
            0.4,
            0.0,
            None,
            0,
            0.0,
            0.1,
            1.0,
            RefinementCandidateSource::SameGammaSeed,
            &ddm,
            &mut stats,
        );
        consider_parent_candidate(
            &mut choice,
            vec![0, 0, 0, 1],
            vec![3, 1],
            2,
            0.8,
            0.0,
            None,
            0,
            0.0,
            0.2,
            1.0,
            RefinementCandidateSource::HighGamma,
            &ddm,
            &mut stats,
        );
        consider_parent_candidate(
            &mut choice,
            vec![0, 1, 1, 1],
            vec![1, 3],
            2,
            0.5,
            0.0,
            None,
            0,
            0.0,
            0.2,
            1.0,
            RefinementCandidateSource::SameGammaSeed,
            &ddm,
            &mut stats,
        );
        record_test_final_candidate_decision(&mut stats, &choice, 1.0);

        assert_eq!(
            choice.source,
            Some(RefinementCandidateSource::SameGammaSeed)
        );
        assert_eq!(choice.candidate_delta_q, 0.2);
        assert_eq!(choice.largest_fraction, 0.5);
        assert_eq!(stats.candidate_qpos_spos, 3);
        assert_eq!(stats.candidate_rejected_by_policy, 2);
        assert_eq!(stats.candidate_true_positive, 1);
        assert_eq!(stats.candidate_false_negative, 2);
    }

    #[test]
    fn test_dongdaemun_refinement_selective_ignores_quotient_tie_break() {
        let ddm = DongdaemunRefinementConfig {
            target_max_weight: 2.0,
            max_extra_parents_per_iteration: 1,
            max_singleton_weight_fraction: 1.0,
            min_largest_child_fraction_improvement: 0.0,
            candidate_quality_policy: CandidateQualityPolicy::Selective,
            min_candidate_delta_q: 0.0,
            use_quotient_diagnostic: true,
            ..DongdaemunRefinementConfig::default()
        };
        let mut stats = RefinementDongdaemunStats::default();
        let mut choice = test_parent_refinement_choice();

        consider_parent_candidate(
            &mut choice,
            vec![0, 0, 1, 1],
            vec![2, 2],
            2,
            0.4,
            0.0,
            Some(0.0),
            0,
            0.0,
            0.2,
            1.0,
            RefinementCandidateSource::SameGammaSeed,
            &ddm,
            &mut stats,
        );
        consider_parent_candidate(
            &mut choice,
            vec![0, 0, 0, 1],
            vec![3, 1],
            2,
            0.8,
            0.0,
            Some(1.0),
            0,
            0.0,
            0.2,
            1.0,
            RefinementCandidateSource::HighGamma,
            &ddm,
            &mut stats,
        );

        assert_eq!(
            choice.source,
            Some(RefinementCandidateSource::SameGammaSeed)
        );
        assert_eq!(choice.largest_fraction, 0.4);
        assert_eq!(choice.quotient_score, 0.0);
        assert_eq!(stats.quotient_candidates, 2);
        assert_eq!(stats.quotient_positive_candidates, 1);
        assert_eq!(stats.candidate_rejected_by_policy, 1);
    }

    #[test]
    fn test_dongdaemun_refinement_quotient_ranking_prefers_positive_candidate() {
        let g = Graph::from_edge_list(5, &[0], &[4], &[5.0]);
        let clustering = Clustering::from_assignments(vec![0, 0, 0, 0, 1]);
        let parent_weights = vec![4.0, 1.0];
        let nodes = [0usize, 1, 2, 3];
        let ddm = DongdaemunRefinementConfig {
            target_max_weight: 2.0,
            max_extra_parents_per_iteration: 1,
            max_extra_children_per_parent: 8,
            max_singleton_weight_fraction: 1.0,
            min_largest_child_fraction_improvement: 0.0,
            use_quotient_diagnostic: true,
            ..DongdaemunRefinementConfig::default()
        };
        let mut stats = RefinementDongdaemunStats::default();
        let mut choice = ParentRefinementChoice {
            assignments: vec![0, 0, 0, 0],
            counts: vec![4],
            n_clusters: 1,
            largest_fraction: 1.0,
            singleton_weight_fraction: 1.0,
            quotient_score: 0.0,
            candidate_delta_q: 0.0,
            source: None,
            trace_candidate_id: None,
            baseline_repair_merge_count: 0,
            baseline_repair_delta_sum: 0.0,
            adaptive_probe_baseline_delta_q: None,
            adaptive_probe_committed: false,
            adaptive_probe_score: f64::NEG_INFINITY,
            adaptive_probe_source_label: None,
        };

        let same_assignments = vec![0, 0, 1, 1];
        let same_score = parent_candidate_quotient_score(
            &g,
            &clustering,
            0,
            nodes.len(),
            &same_assignments,
            2,
            4.0,
            &parent_weights,
            4.0,
            |local| nodes[local],
        );
        let high_assignments = vec![0, 1, 1, 1];
        let high_score = parent_candidate_quotient_score(
            &g,
            &clustering,
            0,
            nodes.len(),
            &high_assignments,
            2,
            4.0,
            &parent_weights,
            4.0,
            |local| nodes[local],
        );

        consider_parent_candidate(
            &mut choice,
            same_assignments,
            vec![2, 2],
            2,
            0.5,
            0.0,
            Some(same_score),
            0,
            0.0,
            0.0,
            1.0,
            RefinementCandidateSource::SameGammaSeed,
            &ddm,
            &mut stats,
        );
        consider_parent_candidate(
            &mut choice,
            high_assignments,
            vec![1, 3],
            2,
            0.75,
            0.25,
            Some(high_score),
            0,
            0.0,
            0.0,
            1.0,
            RefinementCandidateSource::HighGamma,
            &ddm,
            &mut stats,
        );

        assert_eq!(same_score, 0.0);
        assert!(high_score > 0.0);
        assert_eq!(stats.quotient_candidates, 2);
        assert_eq!(stats.quotient_positive_candidates, 1);
        assert_eq!(choice.source, Some(RefinementCandidateSource::HighGamma));
        assert_eq!(choice.largest_fraction, 0.75);
        assert_eq!(choice.quotient_score, high_score);
    }

    #[test]
    fn test_dongdaemun_refinement_zero_quotient_falls_back_to_structural_order() {
        let g = Graph::from_edge_list(5, &[0], &[4], &[5.0]);
        let clustering = Clustering::from_assignments(vec![0, 0, 0, 0, 1]);
        let parent_weights = vec![4.0, 1.0];
        let nodes = [0usize, 1, 2, 3];
        let ddm = DongdaemunRefinementConfig {
            target_max_weight: 2.0,
            max_extra_parents_per_iteration: 1,
            max_extra_children_per_parent: 8,
            max_singleton_weight_fraction: 1.0,
            min_largest_child_fraction_improvement: 0.0,
            use_quotient_diagnostic: true,
            ..DongdaemunRefinementConfig::default()
        };
        let mut stats = RefinementDongdaemunStats::default();
        let mut choice = ParentRefinementChoice {
            assignments: vec![0, 0, 0, 0],
            counts: vec![4],
            n_clusters: 1,
            largest_fraction: 1.0,
            singleton_weight_fraction: 1.0,
            quotient_score: 0.0,
            candidate_delta_q: 0.0,
            source: None,
            trace_candidate_id: None,
            baseline_repair_merge_count: 0,
            baseline_repair_delta_sum: 0.0,
            adaptive_probe_baseline_delta_q: None,
            adaptive_probe_committed: false,
            adaptive_probe_score: f64::NEG_INFINITY,
            adaptive_probe_source_label: None,
        };

        let same_assignments = vec![0, 0, 1, 1];
        let same_score = parent_candidate_quotient_score(
            &g,
            &clustering,
            0,
            nodes.len(),
            &same_assignments,
            2,
            4.0,
            &parent_weights,
            10.0,
            |local| nodes[local],
        );
        let high_assignments = vec![0, 1, 1, 1];
        let high_score = parent_candidate_quotient_score(
            &g,
            &clustering,
            0,
            nodes.len(),
            &high_assignments,
            2,
            4.0,
            &parent_weights,
            10.0,
            |local| nodes[local],
        );

        consider_parent_candidate(
            &mut choice,
            same_assignments,
            vec![2, 2],
            2,
            0.5,
            0.0,
            Some(same_score),
            0,
            0.0,
            0.0,
            1.0,
            RefinementCandidateSource::SameGammaSeed,
            &ddm,
            &mut stats,
        );
        consider_parent_candidate(
            &mut choice,
            high_assignments,
            vec![1, 3],
            2,
            0.75,
            0.25,
            Some(high_score),
            0,
            0.0,
            0.0,
            1.0,
            RefinementCandidateSource::HighGamma,
            &ddm,
            &mut stats,
        );

        assert_eq!(same_score, 0.0);
        assert_eq!(high_score, 0.0);
        assert_eq!(stats.quotient_candidates, 2);
        assert_eq!(stats.quotient_positive_candidates, 0);
        assert_eq!(
            choice.source,
            Some(RefinementCandidateSource::SameGammaSeed)
        );
        assert_eq!(choice.largest_fraction, 0.5);
    }

    fn assert_refinement_children_stay_inside_parent(
        parent: &Clustering,
        refinement: &RefinementResult,
    ) {
        for node in 0..parent.clusters.len() {
            let child = refinement.clustering.clusters[node] as usize;
            assert_eq!(
                refinement.parent_clusters[child], parent.clusters[node],
                "node {node} escaped its move-phase parent"
            );
        }
    }

    #[test]
    fn test_dongdaemun_refinement_quotient_audit_counts_eager_and_streaming() {
        let g = Graph::from_edge_list(5, &[0, 2, 1, 0], &[1, 3, 2, 4], &[10.0, 10.0, 0.01, 5.0]);
        let clustering = Clustering::from_assignments(vec![0, 0, 0, 0, 1]);
        let parent_weights = clustering.cluster_weights(&g.node_weights);
        let config = LeidenConfig {
            resolution: 0.000001,
            n_iterations: 1,
            randomness: 0.0,
            randomness_schedule: Vec::new(),
            seed: 11,
        };
        let ddm = DongdaemunRefinementConfig {
            target_max_weight: 2.5,
            max_extra_parents_per_iteration: 1,
            max_singleton_weight_fraction: 1.0,
            gamma_multipliers: vec![20_000_000.0],
            use_quotient_diagnostic: true,
            ..DongdaemunRefinementConfig::default()
        };

        let nodes_per_cluster = clustering.nodes_per_cluster();
        let mut eager_rng = rand::rngs::StdRng::seed_from_u64(11);
        let eager = refine_eager(
            &g,
            &clustering,
            &nodes_per_cluster,
            &config,
            Some(&ddm),
            Some(&parent_weights),
            config.randomness,
            0,
            0,
            &mut eager_rng,
        );
        assert_eq!(eager.dongdaemun_stats.selected_parents, 1);
        assert_eq!(eager.dongdaemun_stats.high_gamma_candidates, 1);
        assert_eq!(eager.dongdaemun_stats.quotient_candidates, 1);
        assert_eq!(eager.dongdaemun_stats.quotient_positive_candidates, 1);
        assert_eq!(eager.dongdaemun_stats.quotient_selected, 1);
        assert!(eager.dongdaemun_stats.quotient_score_sum > 0.0);
        assert_refinement_children_stay_inside_parent(&clustering, &eager);

        let mut ws = Workspace::new(g.n_nodes);
        clustering.fill_cluster_groups_and_weights(&g.node_weights, &mut ws);
        let starts = ws.npc_starts[..clustering.n_clusters + 1].to_vec();
        let flat_nodes = ws.npc_nodes[..g.n_nodes].to_vec();
        let mut local_index = vec![u32::MAX; g.n_nodes];
        let mut streaming_rng = rand::rngs::StdRng::seed_from_u64(11);
        let streaming = refine_streaming_flat(
            &g,
            &clustering,
            clustering.n_clusters,
            &starts,
            &flat_nodes,
            &mut local_index,
            &config,
            Some(&ddm),
            Some(&parent_weights),
            config.randomness,
            0,
            0,
            &mut streaming_rng,
        );
        assert_eq!(streaming.dongdaemun_stats.selected_parents, 1);
        assert_eq!(streaming.dongdaemun_stats.high_gamma_candidates, 1);
        assert_eq!(streaming.dongdaemun_stats.quotient_candidates, 1);
        assert_eq!(streaming.dongdaemun_stats.quotient_positive_candidates, 1);
        assert_eq!(streaming.dongdaemun_stats.quotient_selected, 1);
        assert!(streaming.dongdaemun_stats.quotient_score_sum > 0.0);
        assert_refinement_children_stay_inside_parent(&clustering, &streaming);
    }

    #[test]
    fn test_dongdaemun_refinement_baseline_repair_merges_positive_child_pairs_only() {
        let g = Graph::from_edge_list(3, &[0, 1], &[1, 2], &[1.1, 1.0]);
        let positive =
            repair_parent_internal_candidate(&g, 3, &[0, 1, 2], 3, 1.0, 0.0, |local| local);

        assert_eq!(positive.merge_count, 1);
        assert_eq!(positive.n_clusters, 2);
        assert_eq!(positive.counts.iter().sum::<u32>(), 3);
        assert!((positive.delta_sum - 0.1).abs() < 1e-12);

        let zero = repair_parent_internal_candidate(&g, 2, &[0, 1], 2, 1.0, 0.0, |local| local + 1);
        assert_eq!(zero.merge_count, 0);
        assert_eq!(zero.n_clusters, 2);
        assert_eq!(zero.assignments, vec![0, 1]);
    }

    #[test]
    fn test_dongdaemun_refinement_baseline_repair_epsilon_allows_near_neutral_merge() {
        let g = Graph::from_edge_list(2, &[0], &[1], &[0.95]);
        let repaired = repair_parent_internal_candidate(&g, 2, &[0, 1], 2, 1.0, 0.1, |local| local);

        assert_eq!(repaired.merge_count, 1);
        assert_eq!(repaired.n_clusters, 1);
        assert!((repaired.delta_sum + 0.05).abs() < 1e-12);
    }

    #[test]
    fn test_dongdaemun_refinement_baseline_repair_rescues_high_gamma_candidate_eager_and_streaming()
    {
        let g = Graph::from_edge_list(5, &[0, 2, 0], &[1, 3, 4], &[10.0, 10.0, 5.0]);
        let clustering = Clustering::from_assignments(vec![0, 0, 0, 0, 1]);
        let parent_weights = clustering.cluster_weights(&g.node_weights);
        let config = LeidenConfig {
            resolution: 1.0,
            n_iterations: 1,
            randomness: 0.0,
            randomness_schedule: Vec::new(),
            seed: 17,
        };
        let ddm_off = DongdaemunRefinementConfig {
            target_max_weight: 2.5,
            max_extra_parents_per_iteration: 1,
            max_extra_children_per_parent: 8,
            max_singleton_weight_fraction: 0.0,
            min_largest_child_fraction_improvement: 0.0,
            gamma_multipliers: vec![100.0],
            use_quotient_diagnostic: true,
            ..DongdaemunRefinementConfig::default()
        };
        let ddm_on = DongdaemunRefinementConfig {
            use_baseline_repair: true,
            ..ddm_off.clone()
        };

        let nodes_per_cluster = clustering.nodes_per_cluster();
        let mut off_rng = rand::rngs::StdRng::seed_from_u64(17);
        let off = refine_eager(
            &g,
            &clustering,
            &nodes_per_cluster,
            &config,
            Some(&ddm_off),
            Some(&parent_weights),
            config.randomness,
            0,
            0,
            &mut off_rng,
        );
        assert_eq!(off.dongdaemun_stats.high_gamma_candidates, 1);
        assert_eq!(off.dongdaemun_stats.applied_parents, 0);
        assert_eq!(off.dongdaemun_stats.baseline_repair_candidates, 0);

        let mut eager_rng = rand::rngs::StdRng::seed_from_u64(17);
        let eager = refine_eager(
            &g,
            &clustering,
            &nodes_per_cluster,
            &config,
            Some(&ddm_on),
            Some(&parent_weights),
            config.randomness,
            0,
            0,
            &mut eager_rng,
        );
        assert_eq!(eager.dongdaemun_stats.high_gamma_candidates, 1);
        assert_eq!(eager.dongdaemun_stats.applied_parents, 1);
        assert_eq!(eager.dongdaemun_stats.high_gamma_applied, 1);
        assert_eq!(eager.dongdaemun_stats.baseline_repair_candidates, 1);
        assert_eq!(
            eager.dongdaemun_stats.baseline_repair_improved_candidates,
            1
        );
        assert_eq!(eager.dongdaemun_stats.baseline_repair_selected, 1);
        assert_eq!(eager.dongdaemun_stats.baseline_repair_merge_count, 2);
        assert!(eager.dongdaemun_stats.baseline_repair_delta_sum > 17.0);
        assert_eq!(eager.dongdaemun_stats.quotient_candidates, 1);
        assert_eq!(eager.dongdaemun_stats.quotient_positive_candidates, 1);
        assert_eq!(eager.dongdaemun_stats.quotient_selected, 1);
        assert_refinement_children_stay_inside_parent(&clustering, &eager);

        let mut ws = Workspace::new(g.n_nodes);
        clustering.fill_cluster_groups_and_weights(&g.node_weights, &mut ws);
        let starts = ws.npc_starts[..clustering.n_clusters + 1].to_vec();
        let flat_nodes = ws.npc_nodes[..g.n_nodes].to_vec();
        let mut local_index = vec![u32::MAX; g.n_nodes];
        let mut streaming_rng = rand::rngs::StdRng::seed_from_u64(17);
        let streaming = refine_streaming_flat(
            &g,
            &clustering,
            clustering.n_clusters,
            &starts,
            &flat_nodes,
            &mut local_index,
            &config,
            Some(&ddm_on),
            Some(&parent_weights),
            config.randomness,
            0,
            0,
            &mut streaming_rng,
        );
        assert_eq!(streaming.dongdaemun_stats.high_gamma_candidates, 1);
        assert_eq!(streaming.dongdaemun_stats.applied_parents, 1);
        assert_eq!(streaming.dongdaemun_stats.baseline_repair_candidates, 1);
        assert_eq!(streaming.dongdaemun_stats.baseline_repair_merge_count, 2);
        assert_eq!(streaming.dongdaemun_stats.quotient_positive_candidates, 1);
        assert_refinement_children_stay_inside_parent(&clustering, &streaming);

        let ddm_aug = DongdaemunRefinementConfig {
            baseline_repair_policy: BaselineRepairPolicy::Augment,
            ..ddm_on
        };
        let mut augment_rng = rand::rngs::StdRng::seed_from_u64(17);
        let augment = refine_eager(
            &g,
            &clustering,
            &nodes_per_cluster,
            &config,
            Some(&ddm_aug),
            Some(&parent_weights),
            config.randomness,
            0,
            0,
            &mut augment_rng,
        );
        assert_eq!(augment.dongdaemun_stats.high_gamma_candidates, 2);
        assert_eq!(augment.dongdaemun_stats.applied_parents, 1);
        assert_eq!(augment.dongdaemun_stats.baseline_repair_candidates, 1);
        assert_eq!(augment.dongdaemun_stats.baseline_repair_selected, 1);
        assert_refinement_children_stay_inside_parent(&clustering, &augment);
    }

    #[test]
    fn test_dongdaemun_refinement_candidate_screen_rejects_bad_shapes() {
        let config = DongdaemunRefinementConfig {
            target_max_weight: 2.0,
            max_extra_parents_per_iteration: 1,
            max_extra_children_per_parent: 3,
            max_singleton_weight_fraction: 0.25,
            min_largest_child_fraction_improvement: 0.05,
            ..DongdaemunRefinementConfig::default()
        };

        assert!(!parent_candidate_is_valid(1, 0.5, 0.0, 1.0, &config));
        assert!(!parent_candidate_is_valid(4, 0.5, 0.0, 1.0, &config));
        assert!(!parent_candidate_is_valid(2, 0.5, 0.5, 1.0, &config));
        assert!(!parent_candidate_is_valid(2, 0.98, 0.0, 1.0, &config));
        assert!(parent_candidate_is_valid(2, 0.5, 0.0, 1.0, &config));
    }

    #[test]
    fn test_dongdaemun_refinement_adaptive_repair_policy_switches_by_parent_ratio() {
        let config = DongdaemunRefinementConfig {
            target_max_weight: 100.0,
            baseline_repair_policy: BaselineRepairPolicy::Adaptive,
            baseline_repair_replace_min_parent_ratio: 1.05,
            ..DongdaemunRefinementConfig::default()
        };

        assert_eq!(
            effective_baseline_repair_policy(&config, 104.0),
            BaselineRepairPolicy::Augment
        );
        assert_eq!(
            effective_baseline_repair_policy(&config, 105.0),
            BaselineRepairPolicy::Replace
        );
    }

    #[test]
    fn test_dongdaemun_refinement_pressure_boundary_parent_selection_reorders_queue() {
        let parent_weights = vec![10.0, 9.0];
        let boundary_pressure = vec![0.0, 10.0];
        let weight_config = DongdaemunRefinementConfig {
            target_max_weight: 1.0,
            max_extra_parents_per_iteration: 1,
            parent_selection_policy: ParentSelectionPolicy::Weight,
            ..DongdaemunRefinementConfig::default()
        };
        let boundary_config = DongdaemunRefinementConfig {
            parent_selection_policy: ParentSelectionPolicy::PressureBoundary,
            ..weight_config.clone()
        };

        let (weight_selected, _, _) =
            select_extra_refinement_parents(&parent_weights, &weight_config, None);
        let (boundary_selected, _, _) = select_extra_refinement_parents(
            &parent_weights,
            &boundary_config,
            Some(&boundary_pressure),
        );

        assert_eq!(weight_selected, vec![true, false]);
        assert_eq!(boundary_selected, vec![false, true]);
    }
}
