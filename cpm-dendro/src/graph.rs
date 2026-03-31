/// Compressed Sparse Row (CSR) graph representation for efficient neighbor iteration.
///
/// Stores an undirected, simple weighted graph. Each undirected edge (u, v, w)
/// is stored in both directions: u→v and v→u.
///
/// **Duplicate edges** are coalesced at construction time: if (u, v) appears
/// multiple times, their weights are summed into a single edge.
/// **Self-loops** are silently skipped.
/// **Zero-weight edges** are treated as absent (dropped during coalescing).
///
/// The caller should ensure the graph is the giant connected component (GCC)
/// — disconnected components lead to density-0 merges in the dendrogram
/// that have no community semantics.

/// A single directed half-edge.
#[derive(Debug, Clone, Copy)]
pub struct HalfEdge {
    pub target: u32,
    pub weight: f64,
}

/// CSR sparse graph (simple, no duplicate edges).
#[derive(Debug)]
pub struct SparseGraph {
    pub n: usize,
    /// `offsets[i]..offsets[i+1]` indexes into `edges` for vertex i's neighbors.
    pub offsets: Vec<usize>,
    /// Neighbor list, sorted by target within each vertex's range.
    /// Each target appears at most once per vertex (coalesced).
    pub edges: Vec<HalfEdge>,
}

impl SparseGraph {
    /// Build a CSR graph from edge arrays.
    ///
    /// `sources` and `targets` are parallel arrays of vertex IDs.
    /// `weights` contains the edge weight for each pair.
    /// Edges are stored in both directions (undirected).
    /// Duplicates are coalesced (weights summed), self-loops skipped,
    /// and zero-weight edges dropped.
    pub fn from_edges(
        n: usize,
        sources: &[u32],
        targets: &[u32],
        weights: &[f64],
    ) -> Self {
        let m = sources.len();
        assert_eq!(m, targets.len());
        assert_eq!(m, weights.len());

        // Count degree for each vertex (both directions)
        let mut degree = vec![0usize; n];
        for i in 0..m {
            let s = sources[i] as usize;
            let t = targets[i] as usize;
            if s != t {
                degree[s] += 1;
                degree[t] += 1;
            }
        }

        // Build offsets (upper bound — may shrink after coalescing)
        let mut offsets = vec![0usize; n + 1];
        for i in 0..n {
            offsets[i + 1] = offsets[i] + degree[i];
        }
        let total = offsets[n];

        // Fill edges (may contain duplicates)
        let mut edges = vec![
            HalfEdge {
                target: 0,
                weight: 0.0,
            };
            total
        ];
        let mut pos = offsets.clone();
        for i in 0..m {
            let s = sources[i] as usize;
            let t = targets[i] as usize;
            if s == t {
                continue;
            }
            let w = weights[i];
            edges[pos[s]] = HalfEdge {
                target: t as u32,
                weight: w,
            };
            pos[s] += 1;
            edges[pos[t]] = HalfEdge {
                target: s as u32,
                weight: w,
            };
            pos[t] += 1;
        }

        // Sort each row by target, then coalesce duplicates (sum weights)
        let mut new_offsets = Vec::with_capacity(n + 1);
        let mut new_edges = Vec::new();
        new_offsets.push(0);

        for i in 0..n {
            let start = offsets[i];
            let end = offsets[i + 1];
            // Sort by target
            edges[start..end].sort_by_key(|e| e.target);

            // Coalesce: merge consecutive entries with the same target
            let mut j = start;
            while j < end {
                let target = edges[j].target;
                let mut w = 0.0;
                while j < end && edges[j].target == target {
                    w += edges[j].weight;
                    j += 1;
                }
                // Drop zero-weight edges
                if w > 0.0 {
                    new_edges.push(HalfEdge { target, weight: w });
                }
            }
            new_offsets.push(new_edges.len());
        }

        SparseGraph {
            n,
            offsets: new_offsets,
            edges: new_edges,
        }
    }

    /// Get neighbors of vertex `v`.
    #[inline]
    pub fn neighbors(&self, v: usize) -> &[HalfEdge] {
        &self.edges[self.offsets[v]..self.offsets[v + 1]]
    }

    /// Number of edges (undirected, counted once).
    #[allow(dead_code)]
    pub fn num_edges(&self) -> usize {
        self.edges.len() / 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triangle() {
        let g = SparseGraph::from_edges(
            3,
            &[0, 1, 0],
            &[1, 2, 2],
            &[1.0, 1.0, 1.0],
        );
        assert_eq!(g.n, 3);
        assert_eq!(g.num_edges(), 3);
        assert_eq!(g.neighbors(0).len(), 2);
        assert_eq!(g.neighbors(1).len(), 2);
        assert_eq!(g.neighbors(2).len(), 2);
    }

    #[test]
    fn test_self_loop_skipped() {
        let g = SparseGraph::from_edges(
            2,
            &[0, 0],
            &[1, 0],
            &[1.0, 5.0],
        );
        assert_eq!(g.num_edges(), 1);
        assert_eq!(g.neighbors(0).len(), 1);
    }

    #[test]
    fn test_duplicate_edges_coalesced() {
        // Edge (0,1) appears twice with weights 3.0 and 2.0 → coalesced to 5.0
        let g = SparseGraph::from_edges(
            2,
            &[0, 0],
            &[1, 1],
            &[3.0, 2.0],
        );
        assert_eq!(g.num_edges(), 1);
        assert_eq!(g.neighbors(0).len(), 1);
        assert!((g.neighbors(0)[0].weight - 5.0).abs() < 1e-10);
        assert!((g.neighbors(1)[0].weight - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_zero_weight_dropped() {
        let g = SparseGraph::from_edges(
            3,
            &[0, 1],
            &[1, 2],
            &[0.0, 1.0],
        );
        // Edge 0-1 (w=0) should be dropped
        assert_eq!(g.neighbors(0).len(), 0);
        assert_eq!(g.neighbors(2).len(), 1);
    }
}
