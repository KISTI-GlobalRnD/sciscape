//! CPM quality function for community detection.
//!
//! Q_CPM = Σ_c [ e_c - γ × s_c × (s_c - 1) / 2 ]
//!
//! where e_c = sum of edge weights within cluster c,
//!       s_c = sum of node weights in cluster c,
//!       γ   = resolution parameter.

use crate::graph::Graph;
use crate::clustering::Clustering;

/// Quality function trait for extensibility.
pub trait QualityFunction {
    /// Total quality of the clustering.
    fn quality(&self, graph: &Graph, clustering: &Clustering) -> f64;

    /// Quality increment for moving `node` from its current cluster
    /// to `target_cluster`.
    ///
    /// `edge_weight_to_target` = sum of edge weights from node to
    /// nodes in target_cluster.
    /// `cluster_weight` = sum of node_weights in target_cluster
    /// (excluding the node being moved).
    fn quality_increment(
        &self,
        node_weight: f64,
        edge_weight_to_target: f64,
        cluster_weight: f64,
    ) -> f64;
}

/// Constant Potts Model (CPM) quality function.
#[derive(Clone, Debug)]
pub struct CPM {
    pub resolution: f64,
}

impl CPM {
    pub fn new(resolution: f64) -> Self {
        CPM { resolution }
    }
}

impl QualityFunction for CPM {
    fn quality(&self, graph: &Graph, clustering: &Clustering) -> f64 {
        // Compute sum of internal edge weights per cluster
        let mut internal_weight = vec![0.0f64; clustering.n_clusters];
        let mut cluster_weight = vec![0.0f64; clustering.n_clusters];

        for node in 0..graph.n_nodes {
            let cid = clustering.clusters[node];
            cluster_weight[cid] += graph.node_weights[node];

            for (nbr, w) in graph.neighbors_of(node) {
                if clustering.clusters[nbr as usize] == cid {
                    internal_weight[cid] += w;
                }
            }
        }

        // Accumulate self-loop weights per cluster (O(n) total)
        let mut self_loop_per_cluster = vec![0.0f64; clustering.n_clusters];
        for node in 0..graph.n_nodes {
            self_loop_per_cluster[clustering.clusters[node]] += graph.self_loop_weights[node];
        }

        // internal_weight counted each edge twice (both directions in CSR)
        let mut quality = 0.0;
        for cid in 0..clustering.n_clusters {
            let e_c = internal_weight[cid] / 2.0 + self_loop_per_cluster[cid];
            let s_c = cluster_weight[cid];
            quality += e_c - self.resolution * s_c * (s_c - 1.0) / 2.0;
        }
        quality
    }

    #[inline]
    fn quality_increment(
        &self,
        node_weight: f64,
        edge_weight_to_target: f64,
        cluster_weight: f64,
    ) -> f64 {
        edge_weight_to_target - node_weight * cluster_weight * self.resolution
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Graph;
    use crate::clustering::Clustering;

    #[test]
    fn test_cpm_quality_triangle() {
        // Triangle: all in one cluster
        let g = Graph::from_edge_list(3, &[0, 1, 2], &[1, 2, 0], &[1.0, 1.0, 1.0]);
        let c = Clustering::from_assignments(vec![0, 0, 0]);
        let cpm = CPM::new(0.5);
        let q = cpm.quality(&g, &c);
        // e_c = 3.0, s_c = 3.0, Q = 3.0 - 0.5 * 3 * 2 / 2 = 3.0 - 1.5 = 1.5
        assert!((q - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_cpm_quality_singletons() {
        // Triangle: all singletons
        let g = Graph::from_edge_list(3, &[0, 1, 2], &[1, 2, 0], &[1.0, 1.0, 1.0]);
        let c = Clustering::singleton(3);
        let cpm = CPM::new(0.5);
        let q = cpm.quality(&g, &c);
        // No internal edges, no penalty → Q = 0
        assert!((q - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_quality_increment() {
        let cpm = CPM::new(0.5);
        // Moving node (weight=1) to cluster (weight=2) with edge_weight=1.5
        let inc = cpm.quality_increment(1.0, 1.5, 2.0);
        // 1.5 - 1.0 * 2.0 * 0.5 = 1.5 - 1.0 = 0.5
        assert!((inc - 0.5).abs() < 1e-10);
    }
}
