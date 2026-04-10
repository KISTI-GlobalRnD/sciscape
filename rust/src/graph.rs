//! CSR (Compressed Sparse Row) graph for Leiden clustering.
//!
//! Undirected weighted graph stored as symmetric CSR. Each edge (u,v,w)
//! is stored in both directions. Supports `node_weights` for contracted
//! graphs where each super-node represents multiple original nodes.

/// Undirected weighted graph in CSR format.
#[derive(Clone, Debug)]
pub struct Graph {
    pub n_nodes: usize,
    pub n_edges: usize, // number of directed entries (= 2 × undirected edges)
    /// CSR row pointers: length n_nodes + 1
    pub first_neighbor_index: Vec<u64>,
    /// CSR column indices: length n_edges
    pub neighbors: Vec<u32>,
    /// Edge weights: length n_edges
    pub edge_weights: Vec<f64>,
    /// Per-node weight (= original node count for contracted graphs).
    /// For non-contracted graphs, all values are 1.0.
    pub node_weights: Vec<f64>,
    /// Per-node self-loop weight (internal edge weight for contracted graphs).
    /// For non-contracted graphs, all values are 0.0.
    pub self_loop_weights: Vec<f64>,
}

impl Graph {
    /// Build from edge list (src, dst, weight). Automatically symmetrizes.
    ///
    /// `n_nodes` must be >= max node index + 1.
    pub fn from_edge_list(
        n_nodes: usize,
        src: &[u32],
        dst: &[u32],
        weights: &[f64],
    ) -> Self {
        assert_eq!(src.len(), dst.len());
        assert_eq!(src.len(), weights.len());

        // Validate node indices
        for i in 0..src.len() {
            assert!(
                (src[i] as usize) < n_nodes,
                "src[{}] = {} >= n_nodes = {}", i, src[i], n_nodes
            );
            assert!(
                (dst[i] as usize) < n_nodes,
                "dst[{}] = {} >= n_nodes = {}", i, dst[i], n_nodes
            );
        }

        // Count degree per node (both directions)
        let mut degree = vec![0u64; n_nodes];
        for (&s, &d) in src.iter().zip(dst.iter()) {
            degree[s as usize] += 1;
            degree[d as usize] += 1;
        }

        // Build CSR row pointers
        let mut first_neighbor_index = vec![0u64; n_nodes + 1];
        for i in 0..n_nodes {
            first_neighbor_index[i + 1] = first_neighbor_index[i] + degree[i];
        }
        let n_edges = first_neighbor_index[n_nodes] as usize;

        // Fill CSR arrays
        let mut neighbors = vec![0u32; n_edges];
        let mut edge_weights = vec![0.0f64; n_edges];
        let mut offset = first_neighbor_index[..n_nodes].to_vec();

        for i in 0..src.len() {
            let s = src[i] as usize;
            let d = dst[i] as usize;
            let w = weights[i];

            let pos_s = offset[s] as usize;
            neighbors[pos_s] = dst[i];
            edge_weights[pos_s] = w;
            offset[s] += 1;

            let pos_d = offset[d] as usize;
            neighbors[pos_d] = src[i];
            edge_weights[pos_d] = w;
            offset[d] += 1;
        }

        let node_weights = vec![1.0; n_nodes];
        let self_loop_weights = vec![0.0; n_nodes];

        Graph {
            n_nodes,
            n_edges,
            first_neighbor_index,
            neighbors,
            edge_weights,
            node_weights,
            self_loop_weights,
        }
    }

    /// Remove duplicate edges and self-loops, summing weights.
    /// Also sorts neighbors per row for cache-friendly access.
    pub fn simplify(&mut self) {
        let mut new_neighbors = Vec::with_capacity(self.n_edges);
        let mut new_weights = Vec::with_capacity(self.n_edges);
        let mut new_first = vec![0u64; self.n_nodes + 1];

        for node in 0..self.n_nodes {
            let start = self.first_neighbor_index[node] as usize;
            let end = self.first_neighbor_index[node + 1] as usize;

            // Collect (neighbor, weight) pairs, excluding self-loops
            let mut pairs: Vec<(u32, f64)> = Vec::with_capacity(end - start);
            for k in start..end {
                if self.neighbors[k] != node as u32 {
                    pairs.push((self.neighbors[k], self.edge_weights[k]));
                }
            }

            // Sort by neighbor
            pairs.sort_by_key(|&(nbr, _)| nbr);

            // Merge duplicates (sum weights)
            let _merged_start = new_neighbors.len();
            for &(nbr, w) in &pairs {
                if new_neighbors.last() == Some(&nbr) {
                    let last_idx = new_weights.len() - 1;
                    new_weights[last_idx] += w;
                } else {
                    new_neighbors.push(nbr);
                    new_weights.push(w);
                }
            }

            new_first[node + 1] = new_neighbors.len() as u64;
        }

        self.n_edges = new_neighbors.len();
        self.first_neighbor_index = new_first;
        self.neighbors = new_neighbors;
        self.edge_weights = new_weights;
    }

    /// Neighbors of a node as (neighbor_index, edge_weight) iterator.
    #[inline]
    pub fn neighbors_of(&self, node: usize) -> NeighborIter<'_> {
        let start = self.first_neighbor_index[node] as usize;
        let end = self.first_neighbor_index[node + 1] as usize;
        NeighborIter {
            neighbors: &self.neighbors[start..end],
            weights: &self.edge_weights[start..end],
            pos: 0,
        }
    }

    /// Degree of a node.
    #[inline]
    pub fn degree(&self, node: usize) -> usize {
        let start = self.first_neighbor_index[node] as usize;
        let end = self.first_neighbor_index[node + 1] as usize;
        end - start
    }

    /// Total edge weight (sum of all directed edge weights / 2).
    pub fn total_edge_weight(&self) -> f64 {
        self.edge_weights.iter().sum::<f64>() / 2.0
    }

    /// Total edge weight of self-loops.
    pub fn total_self_loop_weight(&self) -> f64 {
        let mut total = 0.0;
        for node in 0..self.n_nodes {
            for (nbr, w) in self.neighbors_of(node) {
                if nbr == node as u32 {
                    total += w;
                }
            }
        }
        total / 2.0 // each self-loop counted twice in CSR
    }

    /// Extract subgraph induced by the given node set.
    /// Returns (subgraph, old_to_new_map).
    ///
    /// NOTE: For large graphs, prefer `create_subnetworks()` which
    /// extracts ALL cluster subgraphs in a single O(n+m) pass.
    pub fn subgraph(&self, nodes: &[usize]) -> (Graph, Vec<usize>) {
        use std::collections::HashMap;

        let n_sub = nodes.len();
        let mut old_to_new: HashMap<usize, usize> = HashMap::with_capacity(n_sub);
        for (new_idx, &old_idx) in nodes.iter().enumerate() {
            old_to_new.insert(old_idx, new_idx);
        }

        let mut sub_src = Vec::new();
        let mut sub_dst = Vec::new();
        let mut sub_w = Vec::new();

        for &old_node in nodes {
            let &new_node = old_to_new.get(&old_node).unwrap();
            for (nbr, w) in self.neighbors_of(old_node) {
                if let Some(&new_nbr) = old_to_new.get(&(nbr as usize)) {
                    if (nbr as usize) > old_node {
                        sub_src.push(new_node as u32);
                        sub_dst.push(new_nbr as u32);
                        sub_w.push(w);
                    }
                }
            }
        }

        let mut sub_graph = Graph::from_edge_list(n_sub, &sub_src, &sub_dst, &sub_w);
        sub_graph.node_weights = nodes.iter().map(|&n| self.node_weights[n]).collect();
        sub_graph.self_loop_weights = nodes.iter().map(|&n| self.self_loop_weights[n]).collect();

        // Build full old_to_new Vec for compatibility (sparse → only cluster nodes valid)
        let mut map_vec = vec![usize::MAX; self.n_nodes];
        for (&old, &new) in &old_to_new {
            map_vec[old] = new;
        }

        (sub_graph, map_vec)
    }

    /// Extract ALL cluster subgraphs in a single O(n+m) pass.
    ///
    /// Returns Vec<(Graph, Vec<usize>)> indexed by cluster ID,
    /// where Vec<usize> is the list of original node IDs in that cluster.
    /// Much more efficient than calling `subgraph()` per cluster.
    pub fn create_subnetworks(
        &self,
        nodes_per_cluster: &[Vec<usize>],
    ) -> Vec<(Graph, Vec<usize>)> {
        let n_clusters = nodes_per_cluster.len();

        // Build node → (cluster, intra-cluster-index) mapping
        let mut node_cluster = vec![0usize; self.n_nodes];
        let mut node_local_idx = vec![0u32; self.n_nodes];
        for (cid, nodes) in nodes_per_cluster.iter().enumerate() {
            for (local_idx, &node) in nodes.iter().enumerate() {
                node_cluster[node] = cid;
                node_local_idx[node] = local_idx as u32;
            }
        }

        // Collect edges per cluster in a single pass
        let mut cluster_src: Vec<Vec<u32>> = vec![Vec::new(); n_clusters];
        let mut cluster_dst: Vec<Vec<u32>> = vec![Vec::new(); n_clusters];
        let mut cluster_w: Vec<Vec<f64>> = vec![Vec::new(); n_clusters];

        for node in 0..self.n_nodes {
            let c = node_cluster[node];
            let local_node = node_local_idx[node];
            for (nbr, w) in self.neighbors_of(node) {
                let nbr = nbr as usize;
                if node_cluster[nbr] == c && nbr > node {
                    // Same cluster, add once (will be symmetrized)
                    cluster_src[c].push(local_node);
                    cluster_dst[c].push(node_local_idx[nbr]);
                    cluster_w[c].push(w);
                }
            }
        }

        // Build subgraphs (parallel across clusters)
        use rayon::prelude::*;

        let node_weights = &self.node_weights;
        let self_loop_weights = &self.self_loop_weights;

        let result: Vec<(Graph, Vec<usize>)> = (0..n_clusters)
            .into_par_iter()
            .map(|cid| {
                let nodes = &nodes_per_cluster[cid];
                let n_sub = nodes.len();
                let mut g = Graph::from_edge_list(
                    n_sub,
                    &cluster_src[cid],
                    &cluster_dst[cid],
                    &cluster_w[cid],
                );
                g.node_weights = nodes.iter().map(|&n| node_weights[n]).collect();
                g.self_loop_weights = nodes.iter().map(|&n| self_loop_weights[n]).collect();
                (g, nodes.clone())
            })
            .collect();

        result
    }
}

/// Iterator over (neighbor_index, edge_weight) pairs.
pub struct NeighborIter<'a> {
    neighbors: &'a [u32],
    weights: &'a [f64],
    pos: usize,
}

impl<'a> Iterator for NeighborIter<'a> {
    type Item = (u32, f64);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos < self.neighbors.len() {
            let result = (self.neighbors[self.pos], self.weights[self.pos]);
            self.pos += 1;
            Some(result)
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.neighbors.len() - self.pos;
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for NeighborIter<'a> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triangle() {
        // Triangle: 0-1, 1-2, 2-0
        let g = Graph::from_edge_list(
            3,
            &[0, 1, 2],
            &[1, 2, 0],
            &[1.0, 1.0, 1.0],
        );
        assert_eq!(g.n_nodes, 3);
        assert_eq!(g.n_edges, 6); // 3 undirected = 6 directed
        assert_eq!(g.degree(0), 2);
        assert_eq!(g.degree(1), 2);
        assert_eq!(g.degree(2), 2);
        assert!((g.total_edge_weight() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_subgraph() {
        // 4 nodes: 0-1, 1-2, 2-3
        let g = Graph::from_edge_list(
            4,
            &[0, 1, 2],
            &[1, 2, 3],
            &[1.0, 2.0, 3.0],
        );
        let (sub, map) = g.subgraph(&[1, 2]);
        assert_eq!(sub.n_nodes, 2);
        assert_eq!(sub.n_edges, 2); // 1 undirected = 2 directed
        assert_eq!(map[1], 0); // node 1 → local 0
        assert_eq!(map[2], 1); // node 2 → local 1
    }

    #[test]
    fn test_create_subnetworks() {
        // Two triangles: {0,1,2} and {3,4,5} connected by 2-3
        let g = Graph::from_edge_list(
            6,
            &[0, 1, 2, 3, 4, 5, 2],
            &[1, 2, 0, 4, 5, 3, 3],
            &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5],
        );
        let nodes_per_cluster = vec![vec![0, 1, 2], vec![3, 4, 5]];
        let subs = g.create_subnetworks(&nodes_per_cluster);
        assert_eq!(subs.len(), 2);
        // Cluster 0: triangle {0,1,2} → 3 internal edges
        assert_eq!(subs[0].0.n_nodes, 3);
        assert!((subs[0].0.total_edge_weight() - 3.0).abs() < 1e-10);
        // Cluster 1: triangle {3,4,5} → 3 internal edges
        assert_eq!(subs[1].0.n_nodes, 3);
        assert!((subs[1].0.total_edge_weight() - 3.0).abs() < 1e-10);
        // Cross-cluster edge 2-3 should NOT be in either subgraph
    }
}
