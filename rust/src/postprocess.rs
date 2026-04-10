//! Constrained postprocessing on the CLUSTER GRAPH with cascading γ.
//!
//! 1. Build cluster graph (contraction) with node_sizes
//! 2. Fix large clusters, free small clusters
//! 3. Iteratively lower γ until all clusters ≥ threshold
//! 4. Map results back to original nodes
//!
//! Supports both raw node count and weighted (doc_count) thresholds.
//! When graph.node_weights are non-uniform (contracted graphs), the
//! "size" of a cluster is the sum of node_weights, not the raw count.

use crate::clustering::Clustering;
use crate::contraction::create_reduced_network;
use crate::graph::Graph;
use crate::leiden::{leiden, LeidenConfig};
use crate::workspace::Workspace;
use rand::Rng;
use rand::SeedableRng;

/// Info about one postprocess round.
#[derive(Clone, Debug)]
pub struct PostprocessRound {
    pub round: usize,
    pub gamma: f64,
    pub method: String,            // "leiden" or "greedy"
    pub n_small_before: usize,     // small clusters before this round
    pub n_small_after: usize,      // small clusters after
    pub n_merged: usize,           // clusters that merged in this round
    pub n_new_clusters: usize,     // new clusters formed from small+small merges
    pub n_total_clusters: usize,   // total clusters after this round
    pub max_cluster_size: usize,   // largest cluster after this round (raw count)
    pub max_cluster_weight: f64,   // largest cluster weight after this round
}

/// Result of postprocessing.
#[derive(Clone, Debug)]
pub struct PostprocessResult {
    pub clustering: Clustering,
    pub rounds: Vec<PostprocessRound>,
    /// Per-node: which round changed this node's cluster (-1 = unchanged).
    pub changed_at_round: Vec<i32>,
}

/// Compute cluster weights (sum of node_weights per cluster).
fn cluster_weights(clustering: &Clustering, node_weights: &[f64]) -> Vec<f64> {
    clustering.cluster_weights(node_weights)
}

/// Check if a cluster is "small" using the weighted threshold.
/// If min_weight > 0, compare against weight sum; else use raw count.
fn is_small(weight: f64, raw_size: usize, min_weight: f64, min_size: usize) -> bool {
    if min_weight > 0.0 {
        weight > 0.0 && weight < min_weight
    } else {
        raw_size > 0 && raw_size < min_size
    }
}

/// Reassign small clusters using cascading γ on the cluster graph.
///
/// Threshold semantics:
/// - `min_size`: raw node count threshold (used when node_weights are all 1.0)
/// - `min_weight`: weighted threshold (sum of node_weights, used for contracted graphs)
/// - If `min_weight > 0`, it takes precedence over `min_size`.
pub fn postprocess_small_clusters(
    graph: &Graph,
    clustering: &Clustering,
    config: &LeidenConfig,
    min_size: usize,
    min_weight: f64,
    rng: &mut impl Rng,
) -> PostprocessResult {
    let mut current = clustering.clone();
    let mut gamma = config.resolution;
    let max_rounds = 5;
    let gamma_decay = 0.1;
    let mut rounds = Vec::new();
    let mut changed_at = vec![-1i32; graph.n_nodes];
    let nw = &graph.node_weights;

    for round in 0..max_rounds {
        let sizes = current.cluster_sizes();
        let weights = cluster_weights(&current, nw);
        let n_clusters_before = current.n_clusters;
        let n_small_before = (0..current.n_clusters)
            .filter(|&c| is_small(weights[c], sizes[c], min_weight, min_size))
            .count();

        if n_small_before == 0 {
            break;
        }

        // Build cluster graph
        let mut ws = Workspace::new(graph.n_nodes.max(current.n_clusters));
        let cluster_graph = create_reduced_network(graph, &current, false, &mut ws);

        // Fix large clusters, free small ones
        let n_cls = current.n_clusters;
        let mut cluster_init = Clustering::singleton(n_cls);
        let mut fixed = vec![false; n_cls];
        for cid in 0..n_cls {
            if !is_small(weights[cid], sizes[cid], min_weight, min_size) {
                fixed[cid] = true;
            }
        }
        cluster_init.set_fixed(fixed);

        // Run Leiden on cluster graph
        let cluster_config = LeidenConfig {
            resolution: gamma,
            n_iterations: config.n_iterations,
            randomness: config.randomness,
            seed: config.seed + round as u64,
        };
        let mut round_rng = rand::rngs::StdRng::seed_from_u64(config.seed + round as u64);
        let result = leiden(&cluster_graph, &cluster_config, Some(cluster_init), &mut round_rng);

        // Map back and track changes
        let cluster_map = &result.clustering.clusters;
        let mut new_clusters = vec![0usize; graph.n_nodes];
        for node in 0..graph.n_nodes {
            let new_cid = cluster_map[current.clusters[node]];
            new_clusters[node] = new_cid;
            if new_cid != current.clusters[node] && changed_at[node] == -1 {
                changed_at[node] = round as i32;
            }
        }
        current = Clustering::from_assignments(new_clusters);
        current.remove_empty_clusters();

        let new_sizes = current.cluster_sizes();
        let new_weights = cluster_weights(&current, nw);
        let n_small_after = (0..current.n_clusters)
            .filter(|&c| is_small(new_weights[c], new_sizes[c], min_weight, min_size))
            .count();
        let max_size = new_sizes.iter().copied().max().unwrap_or(0);
        let max_weight = new_weights.iter().copied().fold(0.0f64, f64::max);
        let n_merged = n_clusters_before - current.n_clusters;

        rounds.push(PostprocessRound {
            round,
            gamma,
            method: "leiden".to_string(),
            n_small_before,
            n_small_after,
            n_merged,
            n_new_clusters: 0,
            n_total_clusters: current.n_clusters,
            max_cluster_size: max_size,
            max_cluster_weight: max_weight,
        });

        if n_small_after == 0 {
            break;
        }

        gamma *= gamma_decay;
    }

    // Greedy fallback
    let sizes = current.cluster_sizes();
    let weights = cluster_weights(&current, nw);
    let n_small_before = (0..current.n_clusters)
        .filter(|&c| is_small(weights[c], sizes[c], min_weight, min_size))
        .count();
    if n_small_before > 0 {
        let n_before = current.n_clusters;
        let prev_clusters = current.clusters.clone();
        greedy_merge_remaining(graph, &mut current, min_size, min_weight);

        let greedy_round = rounds.len() as i32;
        for node in 0..graph.n_nodes {
            if current.clusters[node] != prev_clusters[node] && changed_at[node] == -1 {
                changed_at[node] = greedy_round;
            }
        }

        let new_sizes = current.cluster_sizes();
        let new_weights = cluster_weights(&current, nw);
        let n_small_after = (0..current.n_clusters)
            .filter(|&c| is_small(new_weights[c], new_sizes[c], min_weight, min_size))
            .count();
        let max_size = new_sizes.iter().copied().max().unwrap_or(0);
        let max_weight = new_weights.iter().copied().fold(0.0f64, f64::max);

        rounds.push(PostprocessRound {
            round: rounds.len(),
            gamma: 0.0,
            method: "greedy".to_string(),
            n_small_before,
            n_small_after,
            n_merged: n_before - current.n_clusters,
            n_new_clusters: 0,
            n_total_clusters: current.n_clusters,
            max_cluster_size: max_size,
            max_cluster_weight: max_weight,
        });
    }

    PostprocessResult {
        clustering: current,
        rounds,
        changed_at_round: changed_at,
    }
}

/// Greedy fallback: merge remaining small clusters via cluster graph.
fn greedy_merge_remaining(
    graph: &Graph,
    clustering: &mut Clustering,
    min_size: usize,
    min_weight: f64,
) {
    let sizes = clustering.cluster_sizes();
    let weights = cluster_weights(clustering, &graph.node_weights);
    let n_cls = clustering.n_clusters;

    let mut ws = Workspace::new(graph.n_nodes.max(n_cls));
    let cg = create_reduced_network(graph, clustering, false, &mut ws);

    let mut merge_target = vec![usize::MAX; n_cls];
    for cid in 0..n_cls {
        if !is_small(weights[cid], sizes[cid], min_weight, min_size) {
            continue;
        }
        let start = cg.first_neighbor_index[cid] as usize;
        let end = cg.first_neighbor_index[cid + 1] as usize;
        let mut best_cid = cid;
        let mut best_w = 0.0;
        for k in start..end {
            if cg.edge_weights[k] > best_w {
                best_w = cg.edge_weights[k];
                best_cid = cg.neighbors[k] as usize;
            }
        }
        if best_cid != cid {
            merge_target[cid] = best_cid;
        }
    }

    for node in 0..graph.n_nodes {
        let cid = clustering.clusters[node];
        if merge_target[cid] != usize::MAX {
            clustering.clusters[node] = merge_target[cid];
        }
    }
    clustering.remove_empty_clusters();
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_postprocess_with_monitoring() {
        let g = Graph::from_edge_list(
            10,
            &[0,0,0,0,1,1,1,2,2,3, 5, 7, 4, 6, 8],
            &[1,2,3,4,2,3,4,3,4,4, 6, 8, 5, 7, 9],
            &[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0, 1.0, 1.0, 0.1, 0.3, 0.5],
        );
        let init = Clustering::from_assignments(vec![0,0,0,0,0, 1,1, 2,2, 3]);

        let config = LeidenConfig {
            resolution: 0.1,
            n_iterations: 10,
            randomness: 0.01,
            seed: 42,
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        // min_weight=0.0 → use min_size=4
        let result = postprocess_small_clusters(&g, &init, &config, 4, 0.0, &mut rng);

        let sizes = result.clustering.cluster_sizes();
        let remaining_small = sizes.iter().filter(|&&s| s > 0 && s < 4).count();
        assert!(remaining_small < 3, "postprocess should reduce small clusters");
        assert!(!result.rounds.is_empty(), "should have at least 1 round");
    }

    #[test]
    fn test_postprocess_weighted() {
        // 6 nodes: weights [10, 10, 10, 1, 1, 1]
        // clusters: [0,0,0, 1,1,1] → cluster 0 weight=30, cluster 1 weight=3
        let mut g = Graph::from_edge_list(
            6, &[0,1,2,3,4,5,2], &[1,2,0,4,5,3,3], &[1.0,1.0,1.0,1.0,1.0,1.0,0.5],
        );
        g.node_weights = vec![10.0, 10.0, 10.0, 1.0, 1.0, 1.0];
        let init = Clustering::from_assignments(vec![0,0,0,1,1,1]);

        let config = LeidenConfig::default();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        // min_weight=5.0 → cluster 1 (weight=3) is small, cluster 0 (weight=30) is large
        let result = postprocess_small_clusters(&g, &init, &config, 0, 5.0, &mut rng);

        // cluster 1 should merge into cluster 0
        assert_eq!(result.clustering.n_clusters, 1);
    }

    #[test]
    fn test_postprocess_no_small() {
        let g = Graph::from_edge_list(
            6, &[0,1,2,3,4,5], &[1,2,0,4,5,3], &[1.0;6],
        );
        let init = Clustering::from_assignments(vec![0,0,0,1,1,1]);
        let config = LeidenConfig::default();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let result = postprocess_small_clusters(&g, &init, &config, 2, 0.0, &mut rng);
        assert_eq!(result.clustering.n_clusters, 2);
        assert!(result.rounds.is_empty());
    }
}
