//! High-performance graph utility functions.
//!
//! - `filter_top_k`: per-node top-k edge filtering (symmetric/mutual)
//! - `find_gcc`: giant connected component via Union-Find
//! - `contract_edges`: graph contraction via cluster membership

use std::collections::HashMap;

/// Per-node top-k edge filtering.
///
/// Keeps edges where at least one endpoint (symmetric) or both (mutual)
/// rank the edge in their top-k by weight.
///
/// Returns indices of edges that survive the filter.
pub fn filter_top_k(
    src: &[u32],
    dst: &[u32],
    weight: &[f64],
    k: usize,
    mutual: bool,
) -> Vec<usize> {
    let n_edges = src.len();
    if n_edges == 0 || k == 0 {
        return vec![];
    }

    // Build per-node neighbor list with (weight, edge_index, neighbor)
    let n_nodes =
        (src.iter().chain(dst.iter()).copied().max().unwrap_or(0) as usize).saturating_add(1);

    // Collect edges per node (bidirectional)
    let mut node_edges: Vec<Vec<(f64, usize)>> = vec![Vec::new(); n_nodes];
    for i in 0..n_edges {
        let s = src[i] as usize;
        let d = dst[i] as usize;
        let w = weight[i];
        node_edges[s].push((w, i));
        node_edges[d].push((w, i));
    }

    // For each node, find top-k by weight (partial sort)
    // Mark edges: in_top_k_of[edge_idx] = bitmask (bit0=src side, bit1=dst side)
    let mut in_top_k: Vec<u8> = vec![0; n_edges];

    for (node, edges) in node_edges.iter_mut().enumerate() {
        if edges.is_empty() {
            continue;
        }
        // Sort descending by weight, take top-k
        edges.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let limit = k.min(edges.len());
        for &(_, edge_idx) in &edges[..limit] {
            // Determine which side this node is
            if src[edge_idx] as usize == node {
                in_top_k[edge_idx] |= 1; // src side
            } else {
                in_top_k[edge_idx] |= 2; // dst side
            }
        }
    }

    // Filter: symmetric (either side) or mutual (both sides)
    (0..n_edges)
        .filter(|&i| {
            if mutual {
                in_top_k[i] == 3
            } else {
                in_top_k[i] > 0
            }
        })
        .collect()
}

/// Union-Find (Disjoint Set Union) for connected components.
struct UnionFind {
    parent: Vec<u32>,
    rank: Vec<u8>,
    size: Vec<u32>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n as u32).collect(),
            rank: vec![0; n],
            size: vec![1; n],
        }
    }

    fn find(&mut self, mut x: u32) -> u32 {
        while self.parent[x as usize] != x {
            self.parent[x as usize] = self.parent[self.parent[x as usize] as usize];
            x = self.parent[x as usize];
        }
        x
    }

    fn union(&mut self, a: u32, b: u32) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
        let (big, small) = if self.rank[ra as usize] >= self.rank[rb as usize] {
            (ra, rb)
        } else {
            (rb, ra)
        };
        self.parent[small as usize] = big;
        self.size[big as usize] += self.size[small as usize];
        if self.rank[big as usize] == self.rank[small as usize] {
            self.rank[big as usize] += 1;
        }
    }
}

/// Find the giant connected component (GCC).
///
/// Returns a boolean mask: true for nodes in the GCC.
pub fn find_gcc(src: &[u32], dst: &[u32], n_nodes: usize) -> Vec<bool> {
    let mut uf = UnionFind::new(n_nodes);
    for i in 0..src.len() {
        uf.union(src[i], dst[i]);
    }

    // Find largest component
    let mut comp_size: HashMap<u32, u32> = HashMap::new();
    for v in 0..n_nodes as u32 {
        let root = uf.find(v);
        *comp_size.entry(root).or_insert(0) += 1;
    }

    let gcc_root = comp_size
        .iter()
        .max_by_key(|(_, &size)| size)
        .map(|(&root, _)| root)
        .unwrap_or(0);

    let mut mask = vec![false; n_nodes];
    for v in 0..n_nodes as u32 {
        mask[v as usize] = uf.find(v) == gcc_root;
    }
    mask
}

/// Filter edges to keep only those within the GCC.
///
/// Returns indices of edges where both endpoints are in the GCC.
pub fn filter_gcc_edges(src: &[u32], dst: &[u32], n_nodes: usize) -> Vec<usize> {
    let mask = find_gcc(src, dst, n_nodes);
    (0..src.len())
        .filter(|&i| mask[src[i] as usize] && mask[dst[i] as usize])
        .collect()
}

/// Contract edges by cluster membership.
///
/// Merges all nodes in the same cluster into a super-node.
/// Returns (new_src, new_dst, new_weight, n_clusters, node_sizes).
///
/// `prev_node_sizes`: optional per-node weights (for hierarchical contraction).
/// If None, each node has weight 1.
pub fn contract_edges(
    src: &[u32],
    dst: &[u32],
    weight: &[f64],
    membership: &[u64],
    prev_node_sizes: Option<&[i64]>,
) -> (Vec<u32>, Vec<u32>, Vec<f64>, usize, Vec<i64>) {
    let n_edges = src.len();
    let n_clusters = (membership.iter().copied().max().unwrap_or(0) as usize).saturating_add(1);

    // Accumulate edge weights between cluster pairs
    // Use (min_cluster, max_cluster) as key for symmetry
    let mut edge_map: HashMap<(u32, u32), f64> = HashMap::with_capacity(n_edges / 2);
    let mem_len = membership.len();

    for i in 0..n_edges {
        let si = src[i] as usize;
        let di = dst[i] as usize;
        if si >= mem_len || di >= mem_len {
            continue; // skip out-of-bounds nodes
        }
        let cs = membership[si] as u32;
        let cd = membership[di] as u32;
        if cs == cd {
            continue; // intra-cluster edge
        }
        let (lo, hi) = if cs < cd { (cs, cd) } else { (cd, cs) };
        *edge_map.entry((lo, hi)).or_insert(0.0) += weight[i];
    }

    let n_out = edge_map.len();
    let mut out_src = Vec::with_capacity(n_out);
    let mut out_dst = Vec::with_capacity(n_out);
    let mut out_w = Vec::with_capacity(n_out);

    for (&(s, d), &w) in &edge_map {
        out_src.push(s);
        out_dst.push(d);
        out_w.push(w);
    }

    // Node sizes
    let mut node_sizes = vec![0i64; n_clusters];
    match prev_node_sizes {
        Some(prev) => {
            for (v, &m) in membership.iter().enumerate() {
                let cidx = m as usize;
                if v < prev.len() && cidx < n_clusters {
                    node_sizes[cidx] += prev[v];
                }
            }
        }
        None => {
            for &m in membership {
                node_sizes[m as usize] += 1;
            }
        }
    }

    (out_src, out_dst, out_w, n_clusters, node_sizes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_top_k_basic() {
        // Star graph: node 0 connected to 1,2,3,4 with varying weights
        let src = vec![0, 0, 0, 0, 1, 2];
        let dst = vec![1, 2, 3, 4, 2, 3];
        let weight = vec![1.0, 2.0, 3.0, 4.0, 0.5, 0.5];

        // top-2: each node keeps 2 strongest
        let kept = filter_top_k(&src, &dst, &weight, 2, false);
        assert!(kept.len() >= 2);
        assert!(kept.len() <= 6);
    }

    #[test]
    fn test_find_gcc() {
        // Two components: {0,1,2} and {3,4}
        let src = vec![0, 1, 3];
        let dst = vec![1, 2, 4];
        let mask = find_gcc(&src, &dst, 5);
        // GCC is {0,1,2}
        assert!(mask[0] && mask[1] && mask[2]);
        assert!(!mask[3] && !mask[4]);
    }

    #[test]
    fn test_contract_edges() {
        // 4 nodes, 2 clusters: {0,1} in cluster 0, {2,3} in cluster 1
        let src = vec![0, 0, 1, 2];
        let dst = vec![1, 2, 3, 3];
        let weight = vec![1.0, 2.0, 3.0, 1.0];
        let membership = vec![0u64, 0, 1, 1];

        let (out_src, _out_dst, out_w, n_cl, sizes) =
            contract_edges(&src, &dst, &weight, &membership, None);

        assert_eq!(n_cl, 2);
        assert_eq!(out_src.len(), 1); // one inter-cluster edge
        assert_eq!(out_w[0], 5.0); // 2.0 + 3.0
        assert_eq!(sizes, vec![2, 2]);
    }
}
