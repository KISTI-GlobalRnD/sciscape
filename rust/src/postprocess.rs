//! Constrained postprocessing on the CLUSTER GRAPH with cascading γ.
//!
//! 1. Build cluster graph (contraction) with node_sizes
//! 2. Fix large clusters, free small clusters
//! 3. Iteratively lower γ until all clusters ≥ min_size
//! 4. Map results back to original nodes
//!
//! Returns detailed per-round monitoring info.

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
    pub max_cluster_size: usize,   // largest cluster after this round
}

/// Result of postprocessing.
#[derive(Clone, Debug)]
pub struct PostprocessResult {
    pub clustering: Clustering,
    pub rounds: Vec<PostprocessRound>,
    /// Per-node: which round changed this node's cluster (-1 = unchanged).
    pub changed_at_round: Vec<i32>,
}

/// Reassign small clusters using cascading γ on the cluster graph.
pub fn postprocess_small_clusters(
    graph: &Graph,
    clustering: &Clustering,
    config: &LeidenConfig,
    min_size: usize,
    rng: &mut impl Rng,
) -> PostprocessResult {
    let mut current = clustering.clone();
    let mut gamma = config.resolution;
    let max_rounds = 5;
    let gamma_decay = 0.1;  // aggressive: γ drops 10x per round
    let mut rounds = Vec::new();
    // Track which round each node first changed (-1 = never changed)
    let mut changed_at = vec![-1i32; graph.n_nodes];

    for round in 0..max_rounds {
        let sizes = current.cluster_sizes();
        let n_clusters_before = current.n_clusters;
        let n_small_before = sizes.iter().filter(|&&s| s > 0 && s < min_size).count();

        if n_small_before == 0 {
            break;
        }

        // Build cluster graph
        let mut ws = Workspace::new(graph.n_nodes.max(current.n_clusters));
        let cluster_graph = create_reduced_network(graph, &current, false, &mut ws);

        // Fix large clusters
        let n_cls = current.n_clusters;
        let mut cluster_init = Clustering::singleton(n_cls);
        let mut fixed = vec![false; n_cls];
        for cid in 0..n_cls {
            if sizes[cid] >= min_size {
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

        // After remove_empty_clusters, IDs may have shifted — update changed_at
        // for nodes that changed in THIS round (re-check against new IDs)
        // Note: changed_at tracks the round, not the cluster ID, so renumbering is fine.

        let new_sizes = current.cluster_sizes();
        let n_small_after = new_sizes.iter().filter(|&&s| s > 0 && s < min_size).count();
        let max_size = new_sizes.iter().copied().max().unwrap_or(0);
        let n_merged = n_clusters_before - current.n_clusters;

        let n_new = 0; // simplified

        rounds.push(PostprocessRound {
            round,
            gamma,
            method: "leiden".to_string(),
            n_small_before,
            n_small_after,
            n_merged,
            n_new_clusters: n_new,
            n_total_clusters: current.n_clusters,
            max_cluster_size: max_size,
        });

        if n_small_after == 0 {
            break;
        }

        gamma *= gamma_decay;
    }

    // Greedy fallback
    let sizes = current.cluster_sizes();
    let n_small_before = sizes.iter().filter(|&&s| s > 0 && s < min_size).count();
    if n_small_before > 0 {
        let n_before = current.n_clusters;
        let prev_clusters = current.clusters.clone();
        greedy_merge_remaining(graph, &mut current, min_size);

        let greedy_round = rounds.len() as i32;
        for node in 0..graph.n_nodes {
            if current.clusters[node] != prev_clusters[node] && changed_at[node] == -1 {
                changed_at[node] = greedy_round;
            }
        }

        let new_sizes = current.cluster_sizes();
        let n_small_after = new_sizes.iter().filter(|&&s| s > 0 && s < min_size).count();
        let max_size = new_sizes.iter().copied().max().unwrap_or(0);

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
        });
    }

    PostprocessResult {
        clustering: current,
        rounds,
        changed_at_round: changed_at,
    }
}

/// Greedy fallback: merge remaining small clusters via cluster graph.
/// O(n + m) — builds cluster graph once, then greedy assignment.
fn greedy_merge_remaining(
    graph: &Graph,
    clustering: &mut Clustering,
    min_size: usize,
) {
    let sizes = clustering.cluster_sizes();
    let n_cls = clustering.n_clusters;

    // Build cluster graph for inter-cluster weights
    let mut ws = Workspace::new(graph.n_nodes.max(n_cls));
    let cg = create_reduced_network(graph, clustering, false, &mut ws);

    // For each small cluster, find strongest neighbor in cluster graph
    let mut merge_target = vec![usize::MAX; n_cls];
    for cid in 0..n_cls {
        if sizes[cid] == 0 || sizes[cid] >= min_size {
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

    // Apply merges
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
        let result = postprocess_small_clusters(&g, &init, &config, 4, &mut rng);

        // Verify postprocess ran and produced monitoring info
        let sizes = result.clustering.cluster_sizes();
        let remaining_small = sizes.iter().filter(|&&s| s > 0 && s < 4).count();
        let initial_small = 3; // we started with 3 small clusters
        println!("initial small: {}, remaining: {}", initial_small, remaining_small);
        // Should have reduced the number of small clusters
        assert!(remaining_small < initial_small, "postprocess should reduce small clusters");
        assert!(!result.rounds.is_empty(), "should have at least 1 round");

        // Check monitoring info exists
        assert!(!result.rounds.is_empty());
        for r in &result.rounds {
            println!("Round {}: γ={:.4}, method={}, small: {} → {}, merged: {}, total: {}, max: {}",
                     r.round, r.gamma, r.method, r.n_small_before, r.n_small_after,
                     r.n_merged, r.n_total_clusters, r.max_cluster_size);
        }
    }

    #[test]
    fn test_postprocess_no_small() {
        let g = Graph::from_edge_list(
            6, &[0,1,2,3,4,5], &[1,2,0,4,5,3], &[1.0;6],
        );
        let init = Clustering::from_assignments(vec![0,0,0,1,1,1]);
        let config = LeidenConfig::default();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let result = postprocess_small_clusters(&g, &init, &config, 2, &mut rng);
        assert_eq!(result.clustering.n_clusters, 2);
        assert!(result.rounds.is_empty()); // no processing needed
    }
}
