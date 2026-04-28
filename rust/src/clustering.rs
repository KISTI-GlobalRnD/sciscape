//! Cluster assignment with optional fixed-node constraints.

use crate::workspace::Workspace;
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

    /// Fill flat cluster groups in `ws`.
    ///
    /// After this call:
    /// - `ws.npc[..n_clusters]` contains raw cluster sizes
    /// - `ws.npc_starts[..=n_clusters]` contains prefix offsets
    /// - `ws.npc_nodes[..n_nodes]` contains nodes grouped by cluster
    pub(crate) fn fill_cluster_groups(&self, ws: &mut Workspace) {
        let n_nodes = self.n_nodes;
        let n_clusters = self.n_clusters;
        ws.ensure_capacity(n_nodes.max(n_clusters));

        let counts = &mut ws.npc[..n_clusters];
        counts.fill(0);
        let counts_ptr = counts.as_mut_ptr();
        for &cid in &self.clusters {
            let cid = cid as usize;
            debug_assert!(cid < n_clusters);
            unsafe {
                *counts_ptr.add(cid) += 1;
            }
        }

        self.fill_group_offsets_and_nodes(ws);
    }

    /// Fill flat cluster groups and cluster weights in `ws`.
    ///
    /// In addition to `fill_cluster_groups`, `ws.cw[..n_clusters]` contains the
    /// sum of `node_weights` for each cluster.
    pub(crate) fn fill_cluster_groups_and_weights(&self, node_weights: &[f64], ws: &mut Workspace) {
        debug_assert_eq!(node_weights.len(), self.n_nodes);
        let n_nodes = self.n_nodes;
        let n_clusters = self.n_clusters;
        ws.ensure_capacity(n_nodes.max(n_clusters));

        let counts = &mut ws.npc[..n_clusters];
        let weights = &mut ws.cw[..n_clusters];
        counts.fill(0);
        weights.fill(0.0);
        let counts_ptr = counts.as_mut_ptr();
        let weights_ptr = weights.as_mut_ptr();
        for (node, &cid) in self.clusters.iter().enumerate() {
            let cid = cid as usize;
            debug_assert!(cid < n_clusters);
            unsafe {
                *counts_ptr.add(cid) += 1;
                *weights_ptr.add(cid) += *node_weights.get_unchecked(node);
            }
        }

        self.fill_group_offsets_and_nodes(ws);
    }

    fn fill_group_offsets_and_nodes(&self, ws: &mut Workspace) {
        let n_nodes = self.n_nodes;
        let n_clusters = self.n_clusters;

        {
            let counts = &ws.npc[..n_clusters];
            let starts = &mut ws.npc_starts[..n_clusters + 1];
            starts[0] = 0;
            for c in 0..n_clusters {
                starts[c + 1] = starts[c] + counts[c];
            }
        }

        {
            let starts = &ws.npc_starts[..n_clusters];
            let offsets = &mut ws.npc_off[..n_clusters];
            offsets.copy_from_slice(starts);
        }

        let nodes = &mut ws.npc_nodes[..n_nodes];
        let offsets = &mut ws.npc_off[..n_clusters];
        let nodes_ptr = nodes.as_mut_ptr();
        let offsets_ptr = offsets.as_mut_ptr();
        for (node, &cid) in self.clusters.iter().enumerate() {
            let cid = cid as usize;
            debug_assert!(cid < n_clusters);
            unsafe {
                let off = offsets_ptr.add(cid);
                let pos = *off as usize;
                *nodes_ptr.add(pos) = node as u32;
                *off += 1;
            }
        }
    }

    /// Compact cluster IDs using precomputed counts for the active cluster range.
    pub(crate) fn compact_from_counts(&mut self, counts: &mut [u32]) {
        debug_assert!(counts.len() >= self.n_clusters);
        let mut new_id = 0u32;
        let mut needs_remap = false;
        for (old_id, count_or_remap) in counts.iter_mut().take(self.n_clusters).enumerate() {
            if *count_or_remap > 0 {
                if old_id != new_id as usize {
                    needs_remap = true;
                }
                *count_or_remap = new_id;
                new_id += 1;
            }
        }

        let new_n_clusters = new_id as usize;
        if needs_remap {
            for cid in &mut self.clusters {
                *cid = counts[*cid as usize];
            }
        }
        self.n_clusters = new_n_clusters;
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
    fn test_fill_cluster_groups() {
        let clustering = Clustering::from_assignments(vec![1, 0, 1, 0, 2]);
        let mut ws = Workspace::new(clustering.n_nodes);

        clustering.fill_cluster_groups(&mut ws);

        assert_eq!(&ws.npc[..clustering.n_clusters], &[2, 2, 1]);
        assert_eq!(&ws.npc_starts[..=clustering.n_clusters], &[0, 2, 4, 5]);
        assert_eq!(&ws.npc_nodes[..clustering.n_nodes], &[1, 3, 0, 2, 4]);
    }

    #[test]
    fn test_fill_cluster_groups_and_weights() {
        let clustering = Clustering::from_assignments(vec![1, 0, 1, 0, 2]);
        let mut ws = Workspace::new(clustering.n_nodes);

        clustering.fill_cluster_groups_and_weights(&[1.0, 2.0, 3.0, 4.0, 5.0], &mut ws);

        assert_eq!(&ws.npc[..clustering.n_clusters], &[2, 2, 1]);
        assert_eq!(&ws.cw[..clustering.n_clusters], &[6.0, 4.0, 5.0]);
        assert_eq!(&ws.npc_nodes[..clustering.n_nodes], &[1, 3, 0, 2, 4]);
    }

    #[test]
    fn test_compact_from_counts() {
        let mut clustering = Clustering::from_assignments(vec![0, 0, 3, 3]);
        clustering.n_clusters = 4;
        let mut counts = vec![2, 0, 0, 2];

        clustering.compact_from_counts(&mut counts);

        assert_eq!(clustering.n_clusters, 2);
        assert_eq!(clustering.clusters, vec![0, 0, 1, 1]);
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
