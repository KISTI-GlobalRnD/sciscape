//! Leiden algorithm: move → refine → aggregate → recurse.
//!
//! Port of CWTS LeidenAlgorithm.java.

use crate::clustering::Clustering;
use crate::contraction::create_reduced_network;
use crate::fast_local_move;
use crate::graph::Graph;
use crate::local_merge;
use crate::workspace::Workspace;
use rand::Rng;

/// Configuration for the Leiden algorithm.
#[derive(Clone, Debug)]
pub struct LeidenConfig {
    pub resolution: f64,
    pub n_iterations: usize, // 0 = until convergence
    pub randomness: f64,
    pub seed: u64,
}

impl Default for LeidenConfig {
    fn default() -> Self {
        LeidenConfig {
            resolution: 1.0,
            n_iterations: 10,
            randomness: 0.01,
            seed: 0,
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
    let mut clustering = initial.unwrap_or_else(|| Clustering::singleton(graph.n_nodes));
    let mut ws = Workspace::new(graph.n_nodes);

    let mut n_used = 0;
    if config.n_iterations > 0 {
        for _ in 0..config.n_iterations {
            let improved = improve_one_iteration(graph, &mut clustering, config, rng, &mut ws);
            n_used += 1;
            if !improved {
                break;
            }
        }
    } else {
        loop {
            let improved = improve_one_iteration(graph, &mut clustering, config, rng, &mut ws);
            n_used += 1;
            if !improved {
                break;
            }
        }
    }

    let quality = crate::quality::CPM::new(config.resolution).quality(graph, &clustering);

    LeidenResult {
        clustering,
        quality,
        n_iterations_used: n_used,
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
    rng: &mut impl Rng,
    ws: &mut Workspace,
) -> bool {
    // Phase 1: Local moving
    let update = fast_local_move::improve_clustering(graph, clustering, config.resolution, rng, ws);

    // If every node is its own cluster, nothing to do
    if clustering.n_clusters >= graph.n_nodes {
        return update;
    }

    // Phase 2: Refinement
    // Build and refine one cluster subgraph at a time to avoid
    // materializing every cluster subnetwork simultaneously.
    let nodes_per_cluster = clustering.nodes_per_cluster();

    let mut refinement = Clustering::singleton(graph.n_nodes);
    refinement.n_clusters = 0;

    for nodes in nodes_per_cluster.iter() {
        if nodes.is_empty() {
            continue;
        }

        // Check if cluster has any fixed node → skip refinement
        let has_fixed = nodes.iter().any(|&n| clustering.is_fixed(n));

        if has_fixed {
            // Keep cluster intact: all nodes get the same refinement cluster
            for &node in nodes.iter() {
                refinement.clusters[node] = refinement.n_clusters as u32;
            }
            refinement.n_clusters += 1;
        } else {
            let (subgraph, _) = graph.subgraph(nodes);
            // Find sub-communities within this cluster's subnetwork
            let sub_clustering = local_merge::find_clustering(
                &subgraph,
                config.resolution,
                config.randomness,
                rng,
                ws,
            );

            for (local_idx, &node) in nodes.iter().enumerate() {
                refinement.clusters[node] =
                    (refinement.n_clusters + sub_clustering.clusters[local_idx] as usize) as u32;
            }
            refinement.n_clusters += sub_clustering.n_clusters;
        }
    }

    if refinement.n_clusters >= graph.n_nodes {
        // Refinement produced singletons — aggregate on non-refined clustering
        let reduced = create_reduced_network(graph, clustering, true, ws);
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

        let improved = improve_one_iteration(&reduced, &mut reduced_clustering, config, rng, ws);
        clustering.merge_clusters(&reduced_clustering);
        return update | improved;
    }

    // Phase 3: Aggregate based on refined clustering
    let reduced = create_reduced_network(graph, &refinement, true, ws);

    // Initial clustering for aggregate network: map non-refined clusters
    // to the move-phase cluster assignments (before refinement).
    // Each refined sub-cluster inherits the move-phase cluster ID of its parent.
    let mut reduced_clusters = vec![0u32; refinement.n_clusters];
    for i in 0..graph.n_nodes {
        reduced_clusters[refinement.clusters[i] as usize] = clustering.clusters[i];
    }
    let max_cid = reduced_clusters.iter().copied().max().unwrap_or(0) as usize;
    // Propagate fixed status: if any original node in a refined sub-cluster
    // is fixed, the super-node in the reduced graph must also be fixed.
    let reduced_fixed = if clustering.fixed.is_some() {
        let mut rf = vec![false; refinement.n_clusters];
        for i in 0..graph.n_nodes {
            if clustering.is_fixed(i) {
                rf[refinement.clusters[i] as usize] = true;
            }
        }
        Some(rf)
    } else {
        None
    };

    let mut reduced_clustering = Clustering {
        n_nodes: refinement.n_clusters,
        n_clusters: max_cid + 1,
        clusters: reduced_clusters,
        fixed: reduced_fixed,
    };

    // Set the non-refined clustering to the refined one for proper merge-back
    clustering.clusters = refinement.clusters;
    clustering.n_clusters = refinement.n_clusters;

    // Recurse on reduced network
    let improved = improve_one_iteration(&reduced, &mut reduced_clustering, config, rng, ws);

    // Merge back
    clustering.merge_clusters(&reduced_clustering);

    update | improved
}

use crate::quality::QualityFunction;
use rand::SeedableRng;

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

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
}
