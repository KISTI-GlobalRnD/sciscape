//! Cluster assignment with optional fixed-node constraints.

use std::collections::HashMap;

/// Cluster assignment for n nodes.
#[derive(Clone, Debug)]
pub struct Clustering {
    pub n_nodes: usize,
    pub n_clusters: usize,
    pub clusters: Vec<usize>,
    /// If Some, nodes where fixed[i] == true cannot change cluster.
    pub fixed: Option<Vec<bool>>,
}

impl Clustering {
    /// Singleton clustering: each node in its own cluster.
    pub fn singleton(n_nodes: usize) -> Self {
        Clustering {
            n_nodes,
            n_clusters: n_nodes,
            clusters: (0..n_nodes).collect(),
            fixed: None,
        }
    }

    /// From explicit cluster assignments.
    pub fn from_assignments(clusters: Vec<usize>) -> Self {
        let n_nodes = clusters.len();
        let n_clusters = clusters.iter().copied().max().map_or(0, |m| m + 1);
        Clustering {
            n_nodes,
            n_clusters,
            clusters,
            fixed: None,
        }
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
            if cid < self.n_clusters {
                result[cid].push(node);
            }
        }
        result
    }

    /// Cluster sizes.
    pub fn cluster_sizes(&self) -> Vec<usize> {
        let mut sizes = vec![0usize; self.n_clusters];
        for &cid in &self.clusters {
            if cid < self.n_clusters {
                sizes[cid] += 1;
            }
        }
        sizes
    }

    /// Remove empty clusters and renumber to 0..k-1.
    pub fn remove_empty_clusters(&mut self) {
        let mut used = vec![false; self.n_clusters];
        for &cid in &self.clusters {
            used[cid] = true;
        }
        let mut remap = vec![0usize; self.n_clusters];
        let mut new_id = 0;
        for (old_id, &is_used) in used.iter().enumerate() {
            if is_used {
                remap[old_id] = new_id;
                new_id += 1;
            }
        }
        for c in &mut self.clusters {
            *c = remap[*c];
        }
        self.n_clusters = new_id;
    }

    /// Merge clusters based on another clustering on the contracted graph.
    ///
    /// `other` is a clustering of the contracted graph (n = self.n_clusters).
    /// After merge, each node gets the cluster assignment from `other`.
    pub fn merge_clusters(&mut self, other: &Clustering) {
        for c in &mut self.clusters {
            *c = other.clusters[*c];
        }
        self.n_clusters = other.n_clusters;
    }

    /// Count nodes per cluster as HashMap (useful for sparse clusters).
    pub fn cluster_counts(&self) -> HashMap<usize, usize> {
        let mut counts = HashMap::new();
        for &cid in &self.clusters {
            *counts.entry(cid).or_insert(0) += 1;
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
