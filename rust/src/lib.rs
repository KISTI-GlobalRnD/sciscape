//! sciscape-leiden: High-performance CPM Leiden clustering with fixed-node support.
//!
//! # Features
//! - CPM (Constant Potts Model) quality function with node_sizes
//! - Fixed-node constraint: freeze large clusters during postprocess
//! - Graph contraction for hierarchical clustering
//! - Randomized refinement (LocalMergingAlgorithm)

pub mod graph;
pub mod clustering;
pub mod quality;
pub mod contraction;
pub mod fast_local_move;
pub mod local_merge;
pub mod leiden;
pub mod postprocess;
pub mod io;
pub mod workspace;
#[cfg(feature = "python")]
pub mod python;

// Re-export main types
pub use graph::Graph;
pub use clustering::Clustering;
pub use quality::{CPM, QualityFunction};
pub use leiden::{LeidenConfig, LeidenResult, leiden, leiden_multi_start};
pub use postprocess::{postprocess_small_clusters, PostprocessResult, PostprocessRound};
