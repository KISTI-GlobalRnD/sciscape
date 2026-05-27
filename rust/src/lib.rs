//! sciscape-leiden: High-performance CPM Leiden clustering with fixed-node support.
//!
//! # Features
//! - CPM (Constant Potts Model) quality function with node_sizes
//! - Fixed-node constraint: freeze large clusters during postprocess
//! - Graph contraction for hierarchical clustering
//! - Randomized refinement (LocalMergingAlgorithm)

pub mod adaptive;
pub mod clustering;
pub mod contraction;
pub mod dongdaemun;
pub mod fast_local_move;
pub mod graph;
pub mod graph_utils;
pub mod io;
pub mod leiden;
pub mod local_merge;
pub mod postprocess;
#[cfg(feature = "python")]
pub mod python;
pub mod quality;
mod random_utils;
pub mod remap;
mod trace;
pub mod workspace;

// Re-export main types
pub use clustering::Clustering;
pub use graph::Graph;
pub use leiden::{
    leiden, leiden_multi_start, leiden_with_dongdaemun_refinement, AdaptiveLocalShakeArm,
    AdaptiveLocalShakeFinalGuardMode, AdaptiveLocalShakeMode, AdaptiveNearTieProbeMode,
    AdaptiveProbeCommitStrategy, AdaptiveProbeMode, AdaptiveProbeTarget, BaselineRepairPolicy,
    CandidateQualityPolicy, DongdaemunRefinementAudit, DongdaemunRefinementConfig,
    DongdaemunRefinementIterationAudit, DongdaemunRefinementLeidenResult, LeidenConfig,
    LeidenResult, ParentSelectionPolicy,
};
pub use postprocess::{postprocess_small_clusters, PostprocessResult, PostprocessRound};
pub use quality::{QualityFunction, CPM};
