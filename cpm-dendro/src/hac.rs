/// Core HAC algorithm: lazy max-heap sparse average-linkage.
///
/// Builds a CPM-critical dendrogram by greedily merging the cluster pair
/// with highest inter-cluster density ρ(A,B) = e_AB / (|A|·|B|).
///
/// This is mathematically identical to average-linkage HAC on the graph's
/// weighted adjacency matrix (zeros for non-edges).
///
/// # Algorithm
///
/// 1. Initialize: each node is a singleton cluster.
///    For each edge (u,v,w): push (ρ=w, u, v) onto max-heap.
///
/// 2. Repeat n-1 times:
///    a. Pop (ρ, a, b) from heap.
///       - If a or b is dead (already merged): skip.
///       - If ρ is stale (doesn't match current density): skip.
///    b. Create new cluster c = merge(a, b), record in dendrogram.
///    c. For each neighbor d of a or b:
///       Compute e_cd and push new (ρ_cd, c, d) onto heap.
///    d. Mark a, b as dead.
///
/// # Complexity
///
/// - O(m · log n) amortized with lazy deletion for reducible linkages.
/// - Memory: O(n + m) for neighbor dicts, O(m · log n) worst-case heap.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use crate::dendrogram::DendrogramData;
use crate::graph::SparseGraph;

/// An entry in the max-heap representing a candidate merge.
#[derive(Debug, Clone)]
struct MergeCandidate {
    /// CPM density = e_ab / (size_a * size_b)
    density: f64,
    /// Cluster IDs (smaller first for deterministic tie-breaking)
    cluster_a: u32,
    cluster_b: u32,
}

impl PartialEq for MergeCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.density == other.density
            && self.cluster_a == other.cluster_a
            && self.cluster_b == other.cluster_b
    }
}

impl Eq for MergeCandidate {}

impl PartialOrd for MergeCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MergeCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Primary: higher density first
        // Secondary: smaller cluster IDs first (deterministic tie-breaking)
        self.density
            .partial_cmp(&other.density)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.cluster_a.cmp(&self.cluster_a))
            .then_with(|| other.cluster_b.cmp(&self.cluster_b))
    }
}

/// Build a CPM-critical dendrogram from a sparse weighted graph.
///
/// Returns a `DendrogramData` with n-1 merge records, compatible with
/// scipy's linkage matrix format.
///
/// # Panics
///
/// Panics if the graph has 0 nodes.
pub fn build_dendrogram(graph: &SparseGraph) -> DendrogramData {
    let n = graph.n;
    assert!(n > 0, "Graph must have at least 1 node");

    if n == 1 {
        return DendrogramData::new(1);
    }

    // --- Initialize cluster state ---

    // alive[i] = true if cluster i still exists
    let mut alive = vec![true; 2 * n]; // allocate space for internal nodes too

    // Cluster sizes
    let mut size = vec![0u64; 2 * n];
    for i in 0..n {
        size[i] = 1;
    }

    // Neighbor map: cluster → { neighbor_cluster → total_edge_weight }
    // We use "merge smaller into larger" to keep amortized cost low.
    let mut neighbors: Vec<HashMap<u32, f64>> = Vec::with_capacity(2 * n);
    for _ in 0..2 * n {
        neighbors.push(HashMap::new());
    }

    // Build initial neighbor maps from graph edges
    for v in 0..n {
        for he in graph.neighbors(v) {
            let u = he.target as usize;
            if v < u {
                // Process each undirected edge once
                *neighbors[v].entry(u as u32).or_insert(0.0) += he.weight;
                *neighbors[u].entry(v as u32).or_insert(0.0) += he.weight;
            }
        }
    }

    // Build initial heap
    let mut heap = BinaryHeap::new();
    for v in 0..n {
        for (&nbr, &edge_w) in &neighbors[v] {
            let nbr_idx = nbr as usize;
            if v < nbr_idx {
                let density = edge_w / (size[v] as f64 * size[nbr_idx] as f64);
                heap.push(MergeCandidate {
                    density,
                    cluster_a: v as u32,
                    cluster_b: nbr,
                });
            }
        }
    }

    // --- Agglomerative merging ---

    let mut dendro = DendrogramData::new(n);
    let mut next_id = n as u32; // next internal node ID

    let mut merges_done = 0usize;
    let target_merges = n - 1;

    while merges_done < target_merges {
        // Pop candidates until we find a valid one
        let (a, b, density) = loop {
            if let Some(cand) = heap.pop() {
                let a = cand.cluster_a as usize;
                let b = cand.cluster_b as usize;

                // Skip dead clusters
                if !alive[a] || !alive[b] {
                    continue;
                }

                // Verify density is current (not stale)
                let current_edge_w = neighbors[a].get(&(b as u32)).copied().unwrap_or(0.0);
                let current_density = current_edge_w / (size[a] as f64 * size[b] as f64);

                // Allow small floating-point tolerance
                if (current_density - cand.density).abs() < 1e-15 * current_density.max(1e-300) + 1e-300 {
                    break (a, b, current_density);
                }
                // Stale entry — skip
                continue;
            } else {
                // Heap exhausted but merges remain → disconnected components
                // Find two alive clusters and merge with density 0
                let remaining: Vec<usize> = (0..2 * n)
                    .filter(|&i| alive[i])
                    .collect();
                if remaining.len() >= 2 {
                    break (remaining[0], remaining[1], 0.0);
                } else {
                    // Should not happen if n > 1
                    panic!("Cannot complete dendrogram: fewer than 2 alive clusters");
                }
            }
        };

        // Create new internal node
        let c = next_id as usize;
        let new_node_id = dendro.record_merge(
            a as u32,
            b as u32,
            density,
            (size[a] + size[b]) as u32,
        );
        assert_eq!(new_node_id, next_id);
        next_id += 1;

        // Update cluster state
        alive[a] = false;
        alive[b] = false;
        alive[c] = true;
        size[c] = size[a] + size[b];

        // Merge neighbor maps: iterate over both a's and b's neighbors
        // Collect new neighbors for c
        let mut new_nbrs: HashMap<u32, f64> = HashMap::new();

        for (&nbr, &w) in &neighbors[a] {
            let nbr_idx = nbr as usize;
            if nbr_idx != b {
                *new_nbrs.entry(nbr).or_insert(0.0) += w;
            }
        }
        for (&nbr, &w) in &neighbors[b] {
            let nbr_idx = nbr as usize;
            if nbr_idx != a {
                *new_nbrs.entry(nbr).or_insert(0.0) += w;
            }
        }

        // Update neighbor maps of neighbors and push new heap entries
        for (&nbr, &edge_w) in &new_nbrs {
            let nbr_idx = nbr as usize;
            if !alive[nbr_idx] {
                continue;
            }

            // Remove old entries from neighbor's map
            neighbors[nbr_idx].remove(&(a as u32));
            neighbors[nbr_idx].remove(&(b as u32));

            // Add new entry
            neighbors[nbr_idx].insert(c as u32, edge_w);

            // Push new candidate to heap
            let new_density = edge_w / (size[c] as f64 * size[nbr_idx] as f64);
            let (ca, cb) = if c < nbr_idx {
                (c as u32, nbr as u32)
            } else {
                (nbr, c as u32)
            };
            heap.push(MergeCandidate {
                density: new_density,
                cluster_a: ca,
                cluster_b: cb,
            });
        }

        // Store c's neighbor map
        neighbors[c] = new_nbrs;

        // Clear a and b's neighbor maps to free memory
        neighbors[a].clear();
        neighbors[b].clear();

        merges_done += 1;
    }

    dendro
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::SparseGraph;

    #[test]
    fn test_two_nodes() {
        let g = SparseGraph::from_edges(2, &[0], &[1], &[1.0]);
        let d = build_dendrogram(&g);
        assert_eq!(d.merges.len(), 1);
        assert_eq!(d.merges[0].height, 1.0); // ρ = 1/(1·1) = 1.0
        assert_eq!(d.merges[0].size, 2);
    }

    #[test]
    fn test_triangle() {
        // Triangle: 0-1 (w=3), 1-2 (w=1), 0-2 (w=2)
        let g = SparseGraph::from_edges(
            3,
            &[0, 1, 0],
            &[1, 2, 2],
            &[3.0, 1.0, 2.0],
        );
        let d = build_dendrogram(&g);
        assert_eq!(d.merges.len(), 2);

        // First merge: 0+1 (ρ=3.0), not 0+2 (ρ=2.0) or 1+2 (ρ=1.0)
        let m0 = &d.merges[0];
        assert_eq!(m0.height, 3.0);
        assert!(
            (m0.left == 0 && m0.right == 1) || (m0.left == 1 && m0.right == 0),
            "First merge should be nodes 0 and 1"
        );

        // Second merge: {0,1}+{2}
        let m1 = &d.merges[1];
        assert_eq!(m1.size, 3);
        // ρ = (e_02 + e_12) / (2 * 1) = (2+1)/2 = 1.5
        assert!((m1.height - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_merge_heights_monotonic() {
        // 5-node path: 0-1-2-3-4
        let g = SparseGraph::from_edges(
            5,
            &[0, 1, 2, 3],
            &[1, 2, 3, 4],
            &[1.0, 1.0, 1.0, 1.0],
        );
        let d = build_dendrogram(&g);
        assert_eq!(d.merges.len(), 4);

        // Merge heights should be non-increasing (reducibility guarantee)
        for i in 1..d.merges.len() {
            assert!(
                d.merges[i].height <= d.merges[i - 1].height + 1e-10,
                "Merge height increased: {} at step {} > {} at step {}",
                d.merges[i].height,
                i,
                d.merges[i - 1].height,
                i - 1
            );
        }
    }

    #[test]
    fn test_disconnected_components() {
        // Two disconnected edges: 0-1 and 2-3
        let g = SparseGraph::from_edges(
            4,
            &[0, 2],
            &[1, 3],
            &[1.0, 1.0],
        );
        let d = build_dendrogram(&g);
        assert_eq!(d.merges.len(), 3);

        // First two merges at ρ=1.0, last merge at ρ=0.0
        assert_eq!(d.merges[0].height, 1.0);
        assert_eq!(d.merges[1].height, 1.0);
        assert_eq!(d.merges[2].height, 0.0);
    }

    #[test]
    fn test_deterministic() {
        let g = SparseGraph::from_edges(
            4,
            &[0, 1, 2, 0],
            &[1, 2, 3, 3],
            &[1.0, 2.0, 3.0, 1.5],
        );
        let d1 = build_dendrogram(&g);
        let d2 = build_dendrogram(&g);

        // Same graph → same dendrogram
        assert_eq!(d1.merges.len(), d2.merges.len());
        for (m1, m2) in d1.merges.iter().zip(d2.merges.iter()) {
            assert_eq!(m1.left, m2.left);
            assert_eq!(m1.right, m2.right);
            assert!((m1.height - m2.height).abs() < 1e-15);
            assert_eq!(m1.size, m2.size);
        }
    }

    #[test]
    fn test_weighted_edges() {
        // 0-1 (w=10), 0-2 (w=1)
        let g = SparseGraph::from_edges(
            3,
            &[0, 0],
            &[1, 2],
            &[10.0, 1.0],
        );
        let d = build_dendrogram(&g);
        // First merge: 0+1 (ρ=10), then {0,1}+2 (ρ=1/2=0.5)
        assert_eq!(d.merges[0].height, 10.0);
        assert!((d.merges[1].height - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_four_node_counterexample() {
        // The known CPM ≠ greedy counterexample
        // Edges: (0,1), (0,2), (1,2), (1,3) — all weight 1
        let g = SparseGraph::from_edges(
            4,
            &[0, 0, 1, 1],
            &[1, 2, 2, 3],
            &[1.0, 1.0, 1.0, 1.0],
        );
        let d = build_dendrogram(&g);
        assert_eq!(d.merges.len(), 3);

        // All singleton pairs with edges have ρ=1.0
        // Tie-breaking: smallest IDs first → merge 0+1 first
        // Verify dendrogram is valid (monotonic heights)
        for i in 1..d.merges.len() {
            assert!(d.merges[i].height <= d.merges[i - 1].height + 1e-10);
        }
    }
}
