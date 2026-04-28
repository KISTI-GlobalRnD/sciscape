//! Cluster assignment with optional fixed-node constraints.

use std::collections::HashMap;

pub type ClusterId = u32;

/// Cluster assignment for n nodes.
#[derive(Clone, Debug)]
pub struct Clustering {
    pub n_nodes: usize,
    pub n_clusters: usize,
    pub clusters: Vec<ClusterId>,
    /// If Some, nodes where fixed[i] == true cannot change cluster.
    pub fixed: Option<Vec<bool>>,
}

impl Clustering {
    #[inline]
    pub fn cluster_to_index(cluster: ClusterId) -> usize {
        cluster as usize
    }

    #[inline]
    pub fn index_to_cluster(index: usize) -> ClusterId {
        u32::try_from(index).expect("cluster id exceeds u32::MAX")
    }

    /// Singleton clustering: each node in its own cluster.
    pub fn singleton(n_nodes: usize) -> Self {
        assert!(n_nodes <= u32::MAX as usize, "n_nodes exceeds u32::MAX");
        Clustering {
            n_nodes,
            n_clusters: n_nodes,
            clusters: (0..n_nodes as u32).collect(),
            fixed: None,
        }
    }

    /// From explicit cluster assignments.
    pub fn from_assignments(clusters: Vec<ClusterId>) -> Self {
        let n_nodes = clusters.len();
        let n_clusters = clusters.iter().copied().max().map_or(0, |m| m as usize + 1);
        Clustering {
            n_nodes,
            n_clusters,
            clusters,
            fixed: None,
        }
    }

    /// From usize assignments at API/file boundaries.
    pub fn from_usize_assignments(clusters: Vec<usize>) -> Self {
        let clusters = clusters
            .into_iter()
            .map(Self::index_to_cluster)
            .collect::<Vec<_>>();
        Self::from_assignments(clusters)
    }

    /// From u64 assignments at Python/API boundaries.
    pub fn from_u64_assignments(clusters: &[u64]) -> Result<Self, String> {
        let mut out = Vec::with_capacity(clusters.len());
        for &cid in clusters {
            let cid =
                u32::try_from(cid).map_err(|_| format!("cluster id {} exceeds u32::MAX", cid))?;
            out.push(cid);
        }
        Ok(Self::from_assignments(out))
    }

    /// Whether a node's cluster is fixed.
    #[inline]
    pub fn is_fixed(&self, node: usize) -> bool {
        self.fixed.as_ref().map_or(false, |f| f[node])
    }

    /// Set fixed nodes mask.
    pub fn set_fixed(&mut self, fixed: Vec<bool>) {
        assert_eq!(fixed.len(), self.n_nodes);
        self.fixed = Some(fixed);
    }

    /// Nodes grouped by cluster. Returns Vec<Vec<usize>> indexed by cluster id.
    pub fn nodes_per_cluster(&self) -> Vec<Vec<usize>> {
        let mut result = vec![Vec::new(); self.n_clusters];
        for (node, &cid) in self.clusters.iter().enumerate() {
            let cid = Self::cluster_to_index(cid);
            if cid < self.n_clusters {
                result[cid].push(node);
            }
        }
        result
    }

    /// Cluster sizes (raw node count).
    pub fn cluster_sizes(&self) -> Vec<usize> {
        let mut sizes = vec![0usize; self.n_clusters];
        for &cid in &self.clusters {
            let cid = Self::cluster_to_index(cid);
            if cid < self.n_clusters {
                sizes[cid] += 1;
            }
        }
        sizes
    }

    /// Cluster weights (sum of node_weights per cluster).
    /// For contracted graphs, this gives the total doc_count per cluster.
    pub fn cluster_weights(&self, node_weights: &[f64]) -> Vec<f64> {
        let mut weights = vec![0.0f64; self.n_clusters];
        for (node, &cid) in self.clusters.iter().enumerate() {
            let cid = Self::cluster_to_index(cid);
            if cid < self.n_clusters {
                weights[cid] += node_weights[node];
            }
        }
        weights
    }

    /// Remove empty clusters and renumber to 0..k-1.
    pub fn remove_empty_clusters(&mut self) {
        let mut used = vec![false; self.n_clusters];
        for &cid in &self.clusters {
            let cid = Self::cluster_to_index(cid);
            used[cid] = true;
        }
        let mut remap = vec![0u32; self.n_clusters];
        let mut new_id = 0u32;
        for (old_id, &is_used) in used.iter().enumerate() {
            if is_used {
                remap[old_id] = new_id;
                new_id += 1;
            }
        }
        for c in &mut self.clusters {
            *c = remap[Self::cluster_to_index(*c)];
        }
        self.n_clusters = new_id as usize;
    }

    /// Merge clusters based on another clustering on the contracted graph.
    ///
    /// `other` is a clustering of the contracted graph (n = self.n_clusters).
    /// After merge, each node gets the cluster assignment from `other`.
    pub fn merge_clusters(&mut self, other: &Clustering) {
        for c in &mut self.clusters {
            *c = other.clusters[Self::cluster_to_index(*c)];
        }
        self.n_clusters = other.n_clusters;
    }

    /// Count nodes per cluster as HashMap (useful for sparse clusters).
    pub fn cluster_counts(&self) -> HashMap<usize, usize> {
        let mut counts = HashMap::new();
        for &cid in &self.clusters {
            *counts.entry(Self::cluster_to_index(cid)).or_insert(0) += 1;
        }
        counts
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_singleton() {
        let c = Clustering::singleton(5);
        assert_eq!(c.n_clusters, 5);
        assert_eq!(c.clusters, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_fixed() {
        let mut c = Clustering::from_assignments(vec![0, 0, 1, 1]);
        assert!(!c.is_fixed(0));
        c.set_fixed(vec![true, true, false, false]);
        assert!(c.is_fixed(0));
        assert!(!c.is_fixed(2));
    }

    #[test]
    fn test_remove_empty() {
        let mut c = Clustering::from_assignments(vec![0, 0, 3, 3]);
        c.n_clusters = 4; // clusters 1,2 are empty
        c.remove_empty_clusters();
        assert_eq!(c.n_clusters, 2);
        assert_eq!(c.clusters, vec![0, 0, 1, 1]);
    }

    #[test]
    fn test_merge() {
        // Original: 4 nodes, 3 clusters: [0,1,1,2]
        let mut c = Clustering::from_assignments(vec![0, 1, 1, 2]);
        // Contracted graph: 3 nodes → 2 clusters: [0,0,1]
        let other = Clustering::from_assignments(vec![0, 0, 1]);
        c.merge_clusters(&other);
        assert_eq!(c.clusters, vec![0, 0, 0, 1]);
        assert_eq!(c.n_clusters, 2);
    }
}
