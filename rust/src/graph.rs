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

/// Reusable owned buffers for repeated induced-subgraph extraction.
pub struct SubgraphWorkspace {
    degree: Vec<u64>,
    first_neighbor_index: Vec<u64>,
    neighbors: Vec<u32>,
    edge_weights: Vec<f64>,
    node_weights: Vec<f64>,
    self_loop_weights: Vec<f64>,
}

impl SubgraphWorkspace {
    pub fn new() -> Self {
        Self {
            degree: Vec::new(),
            first_neighbor_index: Vec::new(),
            neighbors: Vec::new(),
            edge_weights: Vec::new(),
            node_weights: Vec::new(),
            self_loop_weights: Vec::new(),
        }
    }

    pub fn recycle(&mut self, graph: Graph) {
        self.first_neighbor_index = graph.first_neighbor_index;
        self.neighbors = graph.neighbors;
        self.edge_weights = graph.edge_weights;
        self.node_weights = graph.node_weights;
        self.self_loop_weights = graph.self_loop_weights;
    }
}

impl Graph {
    /// Build from edge list (src, dst, weight). Automatically symmetrizes.
    ///
    /// `n_nodes` must be >= max node index + 1.
    pub fn from_edge_list(n_nodes: usize, src: &[u32], dst: &[u32], weights: &[f64]) -> Self {
        assert_eq!(src.len(), dst.len());
        assert_eq!(src.len(), weights.len());

        // Validate node indices
        for i in 0..src.len() {
            assert!(
                (src[i] as usize) < n_nodes,
                "src[{}] = {} >= n_nodes = {}",
                i,
                src[i],
                n_nodes
            );
            assert!(
                (dst[i] as usize) < n_nodes,
                "dst[{}] = {} >= n_nodes = {}",
                i,
                dst[i],
                n_nodes
            );
        }

        Self::from_edge_list_trusted(n_nodes, src, dst, weights)
    }

    /// Build from edge list without validating endpoint bounds.
    ///
    /// Use only when upstream remapping already guarantees
    /// `src[i], dst[i] < n_nodes`. This saves one full edge pass for Python
    /// callers that construct many large graphs from validated int edges.
    pub fn from_edge_list_trusted(
        n_nodes: usize,
        src: &[u32],
        dst: &[u32],
        weights: &[f64],
    ) -> Self {
        assert_eq!(src.len(), dst.len());
        assert_eq!(src.len(), weights.len());

        // Count stored CSR degree per node. Self-loops are stored separately
        // in `self_loop_weights` because they are constant node-internal mass,
        // not neighbor entries.
        let mut degree = vec![0u64; n_nodes];
        for (&s, &d) in src.iter().zip(dst.iter()) {
            if s != d {
                degree[s as usize] += 1;
                degree[d as usize] += 1;
            }
        }

        Self::from_edge_list_with_degrees_trusted(n_nodes, src, dst, weights, degree)
    }

    /// Build from edge list using precomputed symmetric degrees.
    ///
    /// `degree[i]` must count both directions exactly as
    /// `from_edge_list_trusted` would. This lets callers that already scan
    /// edges for remapping avoid a second full degree pass during CSR build.
    pub fn from_edge_list_with_degrees_trusted(
        n_nodes: usize,
        src: &[u32],
        dst: &[u32],
        weights: &[f64],
        mut degree: Vec<u64>,
    ) -> Self {
        assert_eq!(src.len(), dst.len());
        assert_eq!(src.len(), weights.len());
        assert_eq!(degree.len(), n_nodes);

        // Build CSR row pointers
        let mut first_neighbor_index = vec![0u64; n_nodes + 1];
        let mut running = 0u64;
        for i in 0..n_nodes {
            first_neighbor_index[i] = running;
            let d = degree[i];
            degree[i] = running;
            running += d;
        }
        first_neighbor_index[n_nodes] = running;
        let n_edges = running as usize;

        // Fill CSR arrays
        let mut neighbors = vec![0u32; n_edges];
        let mut edge_weights = vec![0.0f64; n_edges];
        let node_weights = vec![1.0; n_nodes];
        let mut self_loop_weights = vec![0.0; n_nodes];
        let mut offset = degree;

        for i in 0..src.len() {
            let s = src[i] as usize;
            let d = dst[i] as usize;
            let w = weights[i];
            if s == d {
                self_loop_weights[s] += w;
                continue;
            }

            let pos_s = offset[s] as usize;
            neighbors[pos_s] = dst[i];
            edge_weights[pos_s] = w;
            offset[s] += 1;

            let pos_d = offset[d] as usize;
            neighbors[pos_d] = src[i];
            edge_weights[pos_d] = w;
            offset[d] += 1;
        }

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

    /// Build from edge list with explicit node weights (for contracted graphs).
    ///
    /// `node_weights[i]` = doc_count (or original node count) for super-node i.
    pub fn from_edge_list_weighted(
        n_nodes: usize,
        src: &[u32],
        dst: &[u32],
        weights: &[f64],
        node_weights: &[f64],
    ) -> Self {
        assert_eq!(node_weights.len(), n_nodes);
        let mut g = Self::from_edge_list(n_nodes, src, dst, weights);
        g.node_weights = node_weights.to_vec();
        g
    }

    /// Build from edge list with explicit node weights, skipping endpoint
    /// validation. See `from_edge_list_trusted`.
    pub fn from_edge_list_weighted_trusted(
        n_nodes: usize,
        src: &[u32],
        dst: &[u32],
        weights: &[f64],
        node_weights: &[f64],
    ) -> Self {
        assert_eq!(node_weights.len(), n_nodes);
        let mut g = Self::from_edge_list_trusted(n_nodes, src, dst, weights);
        g.node_weights = node_weights.to_vec();
        g
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
        let mut total = self.self_loop_weights.iter().sum::<f64>();
        let mut legacy_csr_total = 0.0;
        for node in 0..self.n_nodes {
            for (nbr, w) in self.neighbors_of(node) {
                if nbr == node as u32 {
                    legacy_csr_total += w;
                }
            }
        }
        total += legacy_csr_total / 2.0;
        total
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

        let mut sub_graph = Graph::from_edge_list_trusted(n_sub, &sub_src, &sub_dst, &sub_w);
        sub_graph.node_weights = nodes.iter().map(|&n| self.node_weights[n]).collect();
        sub_graph.self_loop_weights = nodes.iter().map(|&n| self.self_loop_weights[n]).collect();

        // Build full old_to_new Vec for compatibility (sparse → only cluster nodes valid)
        let mut map_vec = vec![usize::MAX; self.n_nodes];
        for (&old, &new) in &old_to_new {
            map_vec[old] = new;
        }

        (sub_graph, map_vec)
    }

    /// Extract a subgraph using a reusable node → local-index marker array.
    ///
    /// `local_index` must have length `self.n_nodes` and be initialized with
    /// `u32::MAX`. This method marks only the requested nodes, emits the
    /// induced edges, and unmarks the nodes before returning. It avoids both
    /// per-cluster hash maps and the all-cluster temporary edge vectors used by
    /// `create_subnetworks()`.
    pub fn subgraph_with_marker(&self, nodes: &[usize], local_index: &mut [u32]) -> Graph {
        assert_eq!(local_index.len(), self.n_nodes);

        for (new_idx, &old_idx) in nodes.iter().enumerate() {
            local_index[old_idx] = new_idx as u32;
        }

        let mut sub_src = Vec::new();
        let mut sub_dst = Vec::new();
        let mut sub_w = Vec::new();

        for (local_node, &old_node) in nodes.iter().enumerate() {
            for (nbr, w) in self.neighbors_of(old_node) {
                let nbr = nbr as usize;
                let local_nbr = local_index[nbr];
                if local_nbr != u32::MAX && nbr > old_node {
                    sub_src.push(local_node as u32);
                    sub_dst.push(local_nbr);
                    sub_w.push(w);
                }
            }
        }

        let mut sub_graph = Graph::from_edge_list_trusted(nodes.len(), &sub_src, &sub_dst, &sub_w);
        sub_graph.node_weights = nodes.iter().map(|&n| self.node_weights[n]).collect();
        sub_graph.self_loop_weights = nodes.iter().map(|&n| self.self_loop_weights[n]).collect();

        for &old_idx in nodes {
            local_index[old_idx] = u32::MAX;
        }

        sub_graph
    }

    /// Extract a subgraph from a flat u32 node slice.
    ///
    /// This is the large-graph path used by Leiden refinement. It avoids
    /// converting flat workspace storage back into per-cluster `Vec<usize>`.
    pub fn subgraph_with_marker_u32(&self, nodes: &[u32], local_index: &mut [u32]) -> Graph {
        assert_eq!(local_index.len(), self.n_nodes);

        for (new_idx, &old_idx) in nodes.iter().enumerate() {
            local_index[old_idx as usize] = new_idx as u32;
        }

        let n_sub = nodes.len();
        let mut degree = vec![0u64; n_sub];

        for (local_node, &old_node_u32) in nodes.iter().enumerate() {
            let old_node = old_node_u32 as usize;
            let start = self.first_neighbor_index[old_node] as usize;
            let end = self.first_neighbor_index[old_node + 1] as usize;
            for k in start..end {
                let nbr_usize = self.neighbors[k] as usize;
                let local_nbr = local_index[nbr_usize];
                if local_nbr != u32::MAX && nbr_usize > old_node {
                    degree[local_node] += 1;
                    degree[local_nbr as usize] += 1;
                }
            }
        }

        let mut first_neighbor_index = vec![0u64; n_sub + 1];
        let mut running = 0u64;
        for i in 0..n_sub {
            first_neighbor_index[i] = running;
            let d = degree[i];
            degree[i] = running;
            running += d;
        }
        first_neighbor_index[n_sub] = running;

        let n_edges = running as usize;
        let mut neighbors = vec![0u32; n_edges];
        let mut edge_weights = vec![0.0f64; n_edges];
        let mut offset = degree;

        for (local_node, &old_node_u32) in nodes.iter().enumerate() {
            let old_node = old_node_u32 as usize;
            let start = self.first_neighbor_index[old_node] as usize;
            let end = self.first_neighbor_index[old_node + 1] as usize;
            for k in start..end {
                let nbr_usize = self.neighbors[k] as usize;
                let local_nbr = local_index[nbr_usize];
                if local_nbr != u32::MAX && nbr_usize > old_node {
                    let local_nbr_usize = local_nbr as usize;
                    let w = self.edge_weights[k];

                    let pos_s = offset[local_node] as usize;
                    neighbors[pos_s] = local_nbr;
                    edge_weights[pos_s] = w;
                    offset[local_node] += 1;

                    let pos_d = offset[local_nbr_usize] as usize;
                    neighbors[pos_d] = local_node as u32;
                    edge_weights[pos_d] = w;
                    offset[local_nbr_usize] += 1;
                }
            }
        }

        let node_weights = nodes
            .iter()
            .map(|&n| self.node_weights[n as usize])
            .collect();
        let self_loop_weights = nodes
            .iter()
            .map(|&n| self.self_loop_weights[n as usize])
            .collect();

        for &old_idx in nodes {
            local_index[old_idx as usize] = u32::MAX;
        }

        Graph {
            n_nodes: n_sub,
            n_edges,
            first_neighbor_index,
            neighbors,
            edge_weights,
            node_weights,
            self_loop_weights,
        }
    }

    /// Extract a subgraph from a flat u32 node slice using caller-owned buffers.
    ///
    /// The returned `Graph` owns the buffers until callers recycle it back into
    /// the workspace after use.
    pub fn subgraph_with_marker_u32_reuse(
        &self,
        nodes: &[u32],
        local_index: &mut [u32],
        ws: &mut SubgraphWorkspace,
    ) -> Graph {
        assert_eq!(local_index.len(), self.n_nodes);

        for (new_idx, &old_idx) in nodes.iter().enumerate() {
            local_index[old_idx as usize] = new_idx as u32;
        }

        let n_sub = nodes.len();
        ws.degree.clear();
        ws.degree.resize(n_sub, 0);

        for (local_node, &old_node_u32) in nodes.iter().enumerate() {
            let old_node = old_node_u32 as usize;
            let start = self.first_neighbor_index[old_node] as usize;
            let end = self.first_neighbor_index[old_node + 1] as usize;
            for k in start..end {
                let nbr_usize = self.neighbors[k] as usize;
                let local_nbr = local_index[nbr_usize];
                if local_nbr != u32::MAX && nbr_usize > old_node {
                    ws.degree[local_node] += 1;
                    ws.degree[local_nbr as usize] += 1;
                }
            }
        }

        ws.first_neighbor_index.clear();
        ws.first_neighbor_index.reserve(n_sub + 1);
        let mut running = 0u64;
        for i in 0..n_sub {
            ws.first_neighbor_index.push(running);
            let d = ws.degree[i];
            ws.degree[i] = running;
            running += d;
        }
        ws.first_neighbor_index.push(running);

        let n_edges = running as usize;
        let mut neighbors = std::mem::take(&mut ws.neighbors);
        neighbors.clear();
        neighbors.resize(n_edges, 0);
        let mut edge_weights = std::mem::take(&mut ws.edge_weights);
        edge_weights.clear();
        edge_weights.resize(n_edges, 0.0);
        let offset = &mut ws.degree;

        for (local_node, &old_node_u32) in nodes.iter().enumerate() {
            let old_node = old_node_u32 as usize;
            let start = self.first_neighbor_index[old_node] as usize;
            let end = self.first_neighbor_index[old_node + 1] as usize;
            for k in start..end {
                let nbr_usize = self.neighbors[k] as usize;
                let local_nbr = local_index[nbr_usize];
                if local_nbr != u32::MAX && nbr_usize > old_node {
                    let local_nbr_usize = local_nbr as usize;
                    let w = self.edge_weights[k];

                    let pos_s = offset[local_node] as usize;
                    neighbors[pos_s] = local_nbr;
                    edge_weights[pos_s] = w;
                    offset[local_node] += 1;

                    let pos_d = offset[local_nbr_usize] as usize;
                    neighbors[pos_d] = local_node as u32;
                    edge_weights[pos_d] = w;
                    offset[local_nbr_usize] += 1;
                }
            }
        }

        let mut node_weights = std::mem::take(&mut ws.node_weights);
        node_weights.clear();
        node_weights.reserve(n_sub);
        let mut self_loop_weights = std::mem::take(&mut ws.self_loop_weights);
        self_loop_weights.clear();
        self_loop_weights.reserve(n_sub);
        for &node in nodes {
            let node = node as usize;
            node_weights.push(self.node_weights[node]);
            self_loop_weights.push(self.self_loop_weights[node]);
        }

        for &old_idx in nodes {
            local_index[old_idx as usize] = u32::MAX;
        }

        Graph {
            n_nodes: n_sub,
            n_edges,
            first_neighbor_index: std::mem::take(&mut ws.first_neighbor_index),
            neighbors,
            edge_weights,
            node_weights,
            self_loop_weights,
        }
    }

    /// Extract ALL cluster subgraphs in a single O(n+m) pass.
    ///
    /// Returns Vec<(Graph, Vec<usize>)> indexed by cluster ID,
    /// where Vec<usize> is the list of original node IDs in that cluster.
    /// Much more efficient than calling `subgraph()` per cluster.
    pub fn create_subnetworks(&self, nodes_per_cluster: &[Vec<usize>]) -> Vec<(Graph, Vec<usize>)> {
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
                let mut g = Graph::from_edge_list_trusted(
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
        let g = Graph::from_edge_list(3, &[0, 1, 2], &[1, 2, 0], &[1.0, 1.0, 1.0]);
        assert_eq!(g.n_nodes, 3);
        assert_eq!(g.n_edges, 6); // 3 undirected = 6 directed
        assert_eq!(g.degree(0), 2);
        assert_eq!(g.degree(1), 2);
        assert_eq!(g.degree(2), 2);
        assert!((g.total_edge_weight() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_self_loop_edges_are_stored_as_self_loop_weights() {
        let g = Graph::from_edge_list(2, &[0, 0], &[0, 1], &[3.0, 2.0]);

        assert_eq!(g.n_edges, 2);
        assert_eq!(g.degree(0), 1);
        assert_eq!(g.degree(1), 1);
        assert_eq!(g.self_loop_weights, vec![3.0, 0.0]);
        assert!((g.total_edge_weight() - 2.0).abs() < 1e-10);
        assert!((g.total_self_loop_weight() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_subgraph() {
        // 4 nodes: 0-1, 1-2, 2-3
        let g = Graph::from_edge_list(4, &[0, 1, 2], &[1, 2, 3], &[1.0, 2.0, 3.0]);
        let (sub, map) = g.subgraph(&[1, 2]);
        assert_eq!(sub.n_nodes, 2);
        assert_eq!(sub.n_edges, 2); // 1 undirected = 2 directed
        assert_eq!(map[1], 0); // node 1 → local 0
        assert_eq!(map[2], 1); // node 2 → local 1
    }

    #[test]
    fn test_subgraph_with_marker_u32() {
        let g = Graph::from_edge_list(4, &[0, 1, 2], &[1, 2, 3], &[1.0, 2.0, 3.0]);
        let mut marker = vec![u32::MAX; g.n_nodes];
        let sub = g.subgraph_with_marker_u32(&[1, 2], &mut marker);

        assert_eq!(sub.n_nodes, 2);
        assert_eq!(sub.n_edges, 2);
        assert!((sub.total_edge_weight() - 2.0).abs() < 1e-10);
        assert!(marker.iter().all(|&idx| idx == u32::MAX));
    }

    #[test]
    fn test_subgraph_with_marker_u32_reuses_workspace() {
        let g = Graph::from_edge_list(5, &[0, 1, 2, 3], &[1, 2, 3, 4], &[1.0, 2.0, 3.0, 4.0]);
        let mut marker = vec![u32::MAX; g.n_nodes];
        let mut ws = SubgraphWorkspace::new();

        let first = g.subgraph_with_marker_u32_reuse(&[1, 2, 3], &mut marker, &mut ws);
        assert_eq!(first.n_nodes, 3);
        assert_eq!(first.n_edges, 4);
        assert!((first.total_edge_weight() - 5.0).abs() < 1e-10);
        assert!(marker.iter().all(|&idx| idx == u32::MAX));
        ws.recycle(first);

        let second = g.subgraph_with_marker_u32_reuse(&[0, 1], &mut marker, &mut ws);
        assert_eq!(second.n_nodes, 2);
        assert_eq!(second.n_edges, 2);
        assert!((second.total_edge_weight() - 1.0).abs() < 1e-10);
        assert!(marker.iter().all(|&idx| idx == u32::MAX));
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
