/// Core HAC algorithm: lazy max-heap sparse average-linkage.
///
/// Supports multiple merge strategies via the [`Strategy`] enum:
///
/// - **`Cpm`** (default): merge by inter-cluster density ρ(A,B) = e_AB / (|A|·|B|).
///   Heights represent the CPM resolution threshold at which the merge is beneficial.
///
/// - **`InternalDensity`**: merge by merged internal density
///   ρ(A∪B) = (e_A + e_B + e_AB) / C(|A|+|B|, 2).
///   Heights represent the internal density of the merged cluster.
///   This is an ablation candidate — heights do NOT correspond to CPM thresholds.
///
/// Merge heights are stored as **similarity** (non-increasing). This is NOT the
/// same as a scipy distance linkage (which is non-decreasing). Use
/// `to_scipy_distance_flat()` for scipy-compatible output.
///
/// **Note**: This is a greedy approximation. The set of tree-cut partitions
/// does not necessarily contain the global CPM optimum for every γ.
///
/// # Algorithm
///
/// 1. Initialize: each node is a singleton cluster.
///    For each edge (u,v,w): push (score, u, v) onto max-heap.
///
/// 2. Repeat n-1 times:
///    a. Pop (score, a, b) from heap.
///       - If a or b is dead (already merged): skip.
///       - Recompute current score; use that as merge height.
///    b. Create new cluster c = merge(a, b), record in dendrogram.
///    c. For each neighbor d of a or b:
///       Compute score(c, d) and push onto heap.
///    d. Mark a, b as dead.
///
/// # Complexity
///
/// - Current implementation: worst-case Θ(n² log n) time and Θ(n²) heap
///   memory, even for sparse graphs (e.g., star graph).
/// - Better amortized bounds likely require a different data structure /
///   proof; union-by-size alone is not sufficient to guarantee O(m log n)
///   because heap candidate generation cost remains.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, BTreeSet, HashMap};

use crate::dendrogram::DendrogramData;
use crate::graph::SparseGraph;

/// Merge strategy for the HAC algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strategy {
    /// Inter-cluster density: ρ(A,B) = e_AB / (|A|·|B|).
    /// Height = CPM resolution threshold for this merge.
    Cpm,
    /// Merged internal density: ρ(A∪B) = (e_A + e_B + e_AB) / C(|A|+|B|, 2).
    /// Height = internal density of the merged cluster (NOT a CPM threshold).
    InternalDensity,
}

/// An entry in the max-heap representing a candidate merge.
#[derive(Debug, Clone)]
struct MergeCandidate {
    /// Merge score (interpretation depends on Strategy)
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
        // Primary: higher density first (using total_cmp for NaN safety)
        // Secondary: smaller cluster IDs first (deterministic tie-breaking)
        self.density
            .total_cmp(&other.density)
            .then_with(|| other.cluster_a.cmp(&self.cluster_a))
            .then_with(|| other.cluster_b.cmp(&self.cluster_b))
    }
}

/// Build a greedy dendrogram from a sparse weighted graph.
///
/// Returns a `DendrogramData` with n-1 merge records.  Heights are
/// similarity (non-increasing); interpretation depends on `strategy`:
/// - `Cpm`: height = inter-cluster density ρ(A,B)
/// - `InternalDensity`: height = merged internal density ρ(A∪B)
///
/// The input should be the giant connected component (GCC).  If the graph
/// is disconnected, remaining components are merged at height 0.
///
/// # Panics
///
/// Panics if the graph has 0 nodes.
pub fn build_dendrogram(
    graph: &SparseGraph,
    strategy: Strategy,
    node_sizes: Option<&[u64]>,
) -> DendrogramData {
    let n = graph.n;
    assert!(n > 0, "Graph must have at least 1 node");

    if let Some(ns) = node_sizes {
        assert_eq!(
            ns.len(),
            n,
            "node_sizes length ({}) must equal n ({})",
            ns.len(),
            n
        );
    }

    if n == 1 {
        return DendrogramData::new(1);
    }

    // --- Initialize cluster state ---

    // alive[i] = true if cluster i still exists (O(1) lookup)
    // alive_set: ordered set for O(1) disconnected-component fallback
    // Only leaf nodes 0..n-1 start alive; internal nodes n..2n-2 become alive on creation.
    let mut alive = vec![false; 2 * n];
    let mut alive_set: BTreeSet<usize> = BTreeSet::new();
    for i in 0..n {
        alive[i] = true;
        alive_set.insert(i);
    }

    // Cluster sizes — use node_sizes if provided (e.g., contracted graph
    // where each leaf represents a supernode with many original nodes).
    let mut size = vec![0u64; 2 * n];
    for i in 0..n {
        size[i] = node_sizes.map_or(1, |ns| ns[i]);
    }

    // Internal edge weight sum per cluster (only used for InternalDensity strategy).
    // For singletons: 0.0 (no internal edges in a single node).
    let mut internal_edges = vec![0.0f64; 2 * n];

    // Neighbor map: cluster → { neighbor_cluster → total_edge_weight }
    // TODO: implement union-by-size ("smaller into larger") for better amortized cost.
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

    // Compute merge score for a candidate pair (a, b) with inter-cluster edge weight e_ab.
    let compute_score = |_a: usize, _b: usize, e_ab: f64, int_a: f64, int_b: f64, sa: u64, sb: u64| -> f64 {
        match strategy {
            Strategy::Cpm => e_ab / (sa as f64 * sb as f64),
            Strategy::InternalDensity => {
                let total = sa + sb;
                let denom = (total * (total - 1)) as f64 / 2.0; // C(|A|+|B|, 2)
                if denom > 0.0 {
                    (int_a + int_b + e_ab) / denom
                } else {
                    0.0 // merging two singletons with no edge
                }
            }
        }
    };

    // Build initial heap
    let mut heap = BinaryHeap::new();
    for v in 0..n {
        for (&nbr, &edge_w) in &neighbors[v] {
            let nbr_idx = nbr as usize;
            if v < nbr_idx {
                let score = compute_score(
                    v, nbr_idx, edge_w,
                    internal_edges[v], internal_edges[nbr_idx],
                    size[v], size[nbr_idx],
                );
                heap.push(MergeCandidate {
                    density: score,
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

                // Both clusters alive → recompute current score.
                // (If both are alive, the neighbor map reflects the true edge weight.
                //  Stale heap entries are caught by the alive check above.)
                let current_edge_w = neighbors[a].get(&(b as u32)).copied().unwrap_or(0.0);
                if current_edge_w <= 0.0 {
                    // No edge between these clusters (shouldn't happen if both
                    // alive, but guard against it)
                    continue;
                }
                let current_score = compute_score(
                    a, b, current_edge_w,
                    internal_edges[a], internal_edges[b],
                    size[a], size[b],
                );
                break (a, b, current_score);
            } else {
                // Heap exhausted but merges remain → disconnected components
                // Use alive_set for O(1) access to first two alive clusters
                if alive_set.len() >= 2 {
                    let mut iter = alive_set.iter();
                    let first = *iter.next().unwrap();
                    let second = *iter.next().unwrap();
                    break (first, second, 0.0);
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
        alive_set.remove(&a);
        alive_set.remove(&b);
        alive_set.insert(c);
        size[c] = size[a] + size[b];

        // Internal edges of merged cluster: e_A + e_B + e_AB
        let e_ab = neighbors[a].get(&(b as u32)).copied().unwrap_or(0.0);
        internal_edges[c] = internal_edges[a] + internal_edges[b] + e_ab;

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
            let new_density = compute_score(
                c, nbr_idx, edge_w,
                internal_edges[c], internal_edges[nbr_idx],
                size[c], size[nbr_idx],
            );
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

    // Invariant: exactly one cluster should remain alive (the root).
    debug_assert_eq!(
        alive.iter().filter(|&&x| x).count(),
        1,
        "Expected exactly 1 alive cluster after {} merges, found {}",
        target_merges,
        alive.iter().filter(|&&x| x).count()
    );

    dendro
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::SparseGraph;

    #[test]
    fn test_two_nodes() {
        let g = SparseGraph::from_edges(2, &[0], &[1], &[1.0]);
        let d = build_dendrogram(&g, Strategy::Cpm, None);
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
        let d = build_dendrogram(&g, Strategy::Cpm, None);
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
        let d = build_dendrogram(&g, Strategy::Cpm, None);
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
        let d = build_dendrogram(&g, Strategy::Cpm, None);
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
        let d1 = build_dendrogram(&g, Strategy::Cpm, None);
        let d2 = build_dendrogram(&g, Strategy::Cpm, None);

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
        let d = build_dendrogram(&g, Strategy::Cpm, None);
        // First merge: 0+1 (ρ=10), then {0,1}+2 (ρ=1/2=0.5)
        assert_eq!(d.merges[0].height, 10.0);
        assert!((d.merges[1].height - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_greedy_vs_cpm_optimum_divergence() {
        // Demonstrates that the greedy tree does NOT contain all CPM optima.
        //
        // Graph: path 0-1, 1-2, 0-3 (all weight 1)
        //   Greedy merges at ρ=1: tie-break → 0+1 first, then {0,1}+2 at ρ=0.5,
        //   then {0,1,2}+3 at ρ=1/3.
        //
        // At γ=0.3, CPM optimum is {0,3},{1,2} with score 1.4,
        // but this partition is NOT achievable by any cut of the greedy tree
        // (since 0 and 3 are never in the same subtree without 1 or 2).
        let g = SparseGraph::from_edges(
            4,
            &[0, 1, 0],
            &[1, 2, 3],
            &[1.0, 1.0, 1.0],
        );
        let d = build_dendrogram(&g, Strategy::Cpm, None);
        assert_eq!(d.merges.len(), 3);

        // Verify tree is valid (monotonic heights)
        for i in 1..d.merges.len() {
            assert!(
                d.merges[i].height <= d.merges[i - 1].height + 1e-10,
                "Height increased at step {}", i
            );
        }

        // Assert exact merge sequence (deterministic tie-break):
        //   merge 0: 0+1 → node 4,  ρ=1.0,  size=2
        //   merge 1: 4+2 → node 5,  ρ=0.5,  size=3
        //   merge 2: 5+3 → node 6,  ρ=1/3,  size=4
        let m0 = &d.merges[0];
        assert!((m0.height - 1.0).abs() < 1e-10);
        assert_eq!(m0.size, 2);
        assert!(
            (m0.left == 0 && m0.right == 1) || (m0.left == 1 && m0.right == 0),
            "Expected merge of 0 and 1, got {} and {}", m0.left, m0.right
        );

        let m1 = &d.merges[1];
        assert!((m1.height - 0.5).abs() < 1e-10);
        assert_eq!(m1.size, 3);

        let m2 = &d.merges[2];
        assert!((m2.height - 1.0 / 3.0).abs() < 1e-10);
        assert_eq!(m2.size, 4);

        // Verify partition {0,3},{1,2} is NOT a tree cut.
        // Enumerate all tree cuts: at each internal node, either "keep" or "split".
        // Tree structure: 4={0,1}, 5={0,1,2}, 6={0,1,2,3}
        // Possible 2-cluster leaf-partitions from tree cuts:
        //   cut at node 6 → children 5,3 → [{0,1,2}, {3}]
        //   cut at node 5 → children 4,2; but need 4's parent 5 split
        //     if cut 5 into {4,2} and 6 keeps 3: [{0,1}, {2}, {3}] — 3 clusters, not 2
        // The only 2-cluster cuts are: [{0,1,2},{3}] or [{0,1},{2,3}]
        // Neither equals [{0,3},{1,2}].
        // Enumerate properly: collect leaves under each node
        fn leaves_under(node: u32, merges: &[crate::dendrogram::MergeRecord], n: usize) -> Vec<u32> {
            if (node as usize) < n {
                return vec![node];
            }
            let row = (node as usize) - n;
            let mut l = leaves_under(merges[row].left, merges, n);
            l.extend(leaves_under(merges[row].right, merges, n));
            l.sort();
            l
        }

        // Enumerate all 2-cluster cuts of the 3-node internal tree
        let n = 4usize;
        let mut two_cluster_cuts: Vec<(Vec<u32>, Vec<u32>)> = Vec::new();

        // A 2-cluster cut picks one internal node and splits it
        for row in 0..d.merges.len() {
            let left_leaves = leaves_under(d.merges[row].left, &d.merges, n);
            let right_leaves = leaves_under(d.merges[row].right, &d.merges, n);
            // This gives a 2-cluster partition only if the remaining tree
            // above this node is kept as-is. For simplicity, only the root
            // split gives a true 2-cluster partition.
            if row == d.merges.len() - 1 {
                two_cluster_cuts.push((left_leaves, right_leaves));
            }
        }

        // Target partition to check: {0,3} and {1,2}
        let target_a: Vec<u32> = vec![0, 3];
        let target_b: Vec<u32> = vec![1, 2];

        for (a, b) in &two_cluster_cuts {
            let matches = (a == &target_a && b == &target_b)
                || (a == &target_b && b == &target_a);
            assert!(
                !matches,
                "Partition {{0,3}},{{1,2}} should NOT be achievable, but found it as a tree cut"
            );
        }
    }

    #[test]
    fn test_regression_brute_force_agreement() {
        // Regression test: verify that the lazy-heap HAC produces the same
        // merge sequence as a brute-force recomputation on a fixed graph.

        // 5-node graph with varied weights
        let g = SparseGraph::from_edges(
            5,
            &[0, 0, 1, 1, 2, 3],
            &[1, 2, 2, 3, 4, 4],
            &[3.0, 1.0, 2.0, 1.5, 4.0, 0.5],
        );
        let d = build_dendrogram(&g, Strategy::Cpm, None);

        // Brute-force: recompute the same algorithm naively
        let d_brute = brute_force_greedy_hac(&g);

        assert_eq!(d.merges.len(), d_brute.len());
        for (i, (m, b)) in d.merges.iter().zip(d_brute.iter()).enumerate() {
            assert!(
                (m.height - b.2).abs() < 1e-10,
                "Step {}: heap height={}, brute height={}", i, m.height, b.2
            );
            assert_eq!(m.size, b.3, "Step {}: heap size={}, brute size={}", i, m.size, b.3);
        }
    }

    #[test]
    fn test_isolated_vertices() {
        // 4 vertices, only one edge: 0-1
        // Vertices 2 and 3 are isolated
        let g = SparseGraph::from_edges(4, &[0], &[1], &[2.0]);
        let d = build_dendrogram(&g, Strategy::Cpm, None);
        assert_eq!(d.merges.len(), 3);

        // First merge: 0+1 at ρ=2.0
        assert!((d.merges[0].height - 2.0).abs() < 1e-10);
        // Remaining merges at ρ=0.0 (disconnected)
        assert_eq!(d.merges[1].height, 0.0);
        assert_eq!(d.merges[2].height, 0.0);
        // Final size = 4
        assert_eq!(d.merges[2].size, 4);
    }

    #[test]
    fn test_single_node() {
        let g = SparseGraph::from_edges(1, &[], &[], &[]);
        let d = build_dendrogram(&g, Strategy::Cpm, None);
        assert_eq!(d.merges.len(), 0);
    }

    /// Brute-force greedy HAC for testing: O(n^3) naive implementation.
    /// Returns Vec<(left, right, density, size)>.
    fn brute_force_greedy_hac(graph: &SparseGraph) -> Vec<(u32, u32, f64, u32)> {
        use std::collections::{HashMap, HashSet};

        let n = graph.n;
        let mut alive: HashSet<u32> = (0..n as u32).collect();
        let mut cluster_size: HashMap<u32, u64> = (0..n as u32).map(|i| (i, 1u64)).collect();
        let mut edge_weight: HashMap<(u32, u32), f64> = HashMap::new();

        // Build initial edge weights
        for v in 0..n {
            for he in graph.neighbors(v) {
                let u = he.target;
                if (v as u32) < u {
                    let key = (v as u32, u);
                    *edge_weight.entry(key).or_insert(0.0) += he.weight;
                }
            }
        }

        let mut merges = Vec::new();
        let mut next_id = n as u32;

        for _ in 0..n - 1 {
            // Find best pair
            let mut best: Option<(u32, u32, f64)> = None;

            for (&(a, b), &w) in &edge_weight {
                if !alive.contains(&a) || !alive.contains(&b) {
                    continue;
                }
                let sa = *cluster_size.get(&a).unwrap() as f64;
                let sb = *cluster_size.get(&b).unwrap() as f64;
                let density = w / (sa * sb);

                if let Some((_, _, best_d)) = best {
                    if density > best_d + 1e-15
                        || (density > best_d - 1e-15 && (a, b) < (best.unwrap().0, best.unwrap().1))
                    {
                        best = Some((a, b, density));
                    }
                } else {
                    best = Some((a, b, density));
                }
            }

            let (a, b, density) = if let Some(found) = best {
                found
            } else {
                // Disconnected: find two alive clusters
                let remaining: Vec<u32> = alive.iter().copied().collect();
                (remaining[0].min(remaining[1]), remaining[0].max(remaining[1]), 0.0)
            };

            let c = next_id;
            let new_size = cluster_size[&a] + cluster_size[&b];
            merges.push((a, b, density, new_size as u32));

            // Update
            alive.remove(&a);
            alive.remove(&b);
            alive.insert(c);
            cluster_size.insert(c, new_size);

            // Compute new edges for c
            let mut new_edges: HashMap<u32, f64> = HashMap::new();
            for (&(x, y), &w) in &edge_weight {
                if !alive.contains(&x) && x != c || !alive.contains(&y) && y != c {
                    // Only process edges involving a or b with alive neighbors
                }
                if (x == a || x == b) && alive.contains(&y) {
                    *new_edges.entry(y).or_insert(0.0) += w;
                }
                if (y == a || y == b) && alive.contains(&x) {
                    *new_edges.entry(x).or_insert(0.0) += w;
                }
            }

            // Remove old edges for a, b
            edge_weight.retain(|&(x, y), _| {
                !(x == a || x == b || y == a || y == b)
            });

            // Add new edges
            for (&nbr, &w) in &new_edges {
                let key = if c < nbr { (c, nbr) } else { (nbr, c) };
                edge_weight.insert(key, w);
            }

            next_id += 1;
        }

        merges
    }

    #[test]
    fn test_internal_density_triangle() {
        // Triangle: 0-1 (w=3), 1-2 (w=1), 0-2 (w=2)
        let g = SparseGraph::from_edges(
            3,
            &[0, 1, 0],
            &[1, 2, 2],
            &[3.0, 1.0, 2.0],
        );
        let d = build_dendrogram(&g, Strategy::InternalDensity, None);
        assert_eq!(d.merges.len(), 2);

        // InternalDensity: score = (e_A + e_B + e_AB) / C(|A|+|B|, 2)
        // For singletons (e_A=0, e_B=0): score = e_AB / C(2,2) = e_AB / 1 = e_AB
        // So first merge is still 0+1 (score=3.0)
        let m0 = &d.merges[0];
        assert!((m0.height - 3.0).abs() < 1e-10);

        // Second merge: {0,1}+{2}
        // e_A=3.0 (internal edge of {0,1}), e_B=0, e_AB=1+2=3
        // score = (3+0+3) / C(3,2) = 6/3 = 2.0
        let m1 = &d.merges[1];
        assert!((m1.height - 2.0).abs() < 1e-10, "got {}", m1.height);
        assert_eq!(m1.size, 3);
    }

    #[test]
    fn test_internal_density_vs_cpm_different_order() {
        // Graph where InternalDensity and Cpm produce different merge orders.
        // 4 nodes: 0-1 (w=2), 2-3 (w=1), 1-2 (w=0.5)
        //
        // CPM scores: ρ(0,1)=2, ρ(2,3)=1, ρ(1,2)=0.5
        //   → merge order: 0+1, 2+3, {0,1}+{2,3}
        //
        // InternalDensity scores initially same (singletons have no internal edges).
        //   → First merge: 0+1 (score=2), second: 2+3 (score=1)
        //   → Third merge: {0,1} (int=2) + {2,3} (int=1), e_AB=0.5
        //     CPM: ρ = 0.5 / (2*2) = 0.125
        //     IntDens: (2+1+0.5)/C(4,2) = 3.5/6 ≈ 0.583
        let g = SparseGraph::from_edges(
            4,
            &[0, 2, 1],
            &[1, 3, 2],
            &[2.0, 1.0, 0.5],
        );

        let d_cpm = build_dendrogram(&g, Strategy::Cpm, None);
        let d_int = build_dendrogram(&g, Strategy::InternalDensity, None);

        // Both should have 3 merges
        assert_eq!(d_cpm.merges.len(), 3);
        assert_eq!(d_int.merges.len(), 3);

        // Final merge heights should differ
        let cpm_final = d_cpm.merges[2].height;   // 0.125
        let int_final = d_int.merges[2].height;    // 0.583...
        assert!((cpm_final - 0.125).abs() < 1e-10, "cpm={}", cpm_final);
        assert!((int_final - 3.5 / 6.0).abs() < 1e-10, "int={}", int_final);
    }

    #[test]
    fn test_internal_density_monotonic() {
        // 5-node path: heights should be non-increasing
        let g = SparseGraph::from_edges(
            5,
            &[0, 1, 2, 3],
            &[1, 2, 3, 4],
            &[1.0, 1.0, 1.0, 1.0],
        );
        let d = build_dendrogram(&g, Strategy::InternalDensity, None);
        assert_eq!(d.merges.len(), 4);

        for i in 1..d.merges.len() {
            assert!(
                d.merges[i].height <= d.merges[i - 1].height + 1e-10,
                "Height increased: {} at step {} > {} at step {}",
                d.merges[i].height, i, d.merges[i - 1].height, i - 1
            );
        }
    }

    #[test]
    fn test_node_sizes_contracted_graph() {
        // Simulate a contracted graph: 3 supernodes with sizes [100, 200, 50]
        // Edge 0-1 weight=500, edge 1-2 weight=300
        //
        // Without node_sizes: ρ(0,1) = 500/(1·1) = 500
        // With node_sizes:    ρ(0,1) = 500/(100·200) = 0.025
        let g = SparseGraph::from_edges(
            3,
            &[0, 1],
            &[1, 2],
            &[500.0, 300.0],
        );
        let sizes = vec![100u64, 200, 50];

        let d_no_sizes = build_dendrogram(&g, Strategy::Cpm, None);
        let d_with_sizes = build_dendrogram(&g, Strategy::Cpm, Some(&sizes));

        // Without sizes: first merge at ρ=500
        assert!((d_no_sizes.merges[0].height - 500.0).abs() < 1e-10);
        assert_eq!(d_no_sizes.merges[0].size, 2);

        // With sizes: ρ(0,1) = 500/(100·200) = 0.025
        //             ρ(1,2) = 300/(200·50)  = 0.03
        // So 1+2 merges first (higher density), then {1,2}+0
        assert!((d_with_sizes.merges[0].height - 0.03).abs() < 1e-10);
        assert_eq!(d_with_sizes.merges[0].size, 250); // 200+50

        // Second merge: {1,2}+{0}: ρ = 500/(250·100) = 0.02
        assert!((d_with_sizes.merges[1].height - 0.02).abs() < 1e-10);
        assert_eq!(d_with_sizes.merges[1].size, 350); // 250+100
    }

    #[test]
    fn test_node_sizes_preserves_monotonicity() {
        // 4-node contracted graph with varied sizes
        let g = SparseGraph::from_edges(
            4,
            &[0, 0, 1, 2],
            &[1, 2, 3, 3],
            &[100.0, 50.0, 80.0, 200.0],
        );
        let sizes = vec![500u64, 300, 1000, 200];

        let d = build_dendrogram(&g, Strategy::Cpm, Some(&sizes));
        assert_eq!(d.merges.len(), 3);

        // Heights must be non-increasing
        for i in 1..d.merges.len() {
            assert!(
                d.merges[i].height <= d.merges[i - 1].height + 1e-10,
                "Height increased at step {}: {} > {}",
                i, d.merges[i].height, d.merges[i - 1].height
            );
        }

        // Final size = sum of all node sizes
        assert_eq!(d.merges[2].size as u64, 500 + 300 + 1000 + 200);
    }
}
