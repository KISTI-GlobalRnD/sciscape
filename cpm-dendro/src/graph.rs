/// Compressed Sparse Row (CSR) graph representation for efficient neighbor iteration.
///
/// Stores an undirected weighted graph. Each undirected edge (u, v, w) is stored
/// in both directions: u→v and v→u.

/// A single directed half-edge.
#[derive(Debug, Clone, Copy)]
pub struct HalfEdge {
    pub target: u32,
    pub weight: f64,
}

/// CSR sparse graph.
#[derive(Debug)]
pub struct SparseGraph {
    pub n: usize,
    /// `offsets[i]..offsets[i+1]` indexes into `edges` for vertex i's neighbors.
    pub offsets: Vec<usize>,
    /// Neighbor list, sorted by target within each vertex's range.
    pub edges: Vec<HalfEdge>,
}

impl SparseGraph {
    /// Build a CSR graph from edge arrays.
    ///
    /// `sources` and `targets` are parallel arrays of vertex IDs.
    /// `weights` contains the edge weight for each pair.
    /// Edges are stored in both directions (undirected).
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
            // Skip self-loops
        }

        // Build offsets
        let mut offsets = vec![0usize; n + 1];
        for i in 0..n {
            offsets[i + 1] = offsets[i] + degree[i];
        }
        let total = offsets[n];

        // Fill edges
        let mut edges = vec![
            HalfEdge {
                target: 0,
                weight: 0.0,
            };
            total
        ];
        let mut pos = offsets.clone(); // current insertion position per vertex
        for i in 0..m {
            let s = sources[i] as usize;
            let t = targets[i] as usize;
            if s == t {
                continue; // skip self-loops
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

        // Sort neighbors by target for consistent ordering
        for i in 0..n {
            let start = offsets[i];
            let end = offsets[i + 1];
            edges[start..end].sort_by_key(|e| e.target);
        }

        SparseGraph { n, offsets, edges }
    }

    /// Get neighbors of vertex `v`.
    #[inline]
    pub fn neighbors(&self, v: usize) -> &[HalfEdge] {
        &self.edges[self.offsets[v]..self.offsets[v + 1]]
    }

    /// Number of edges (undirected, counted once).
    pub fn num_edges(&self) -> usize {
        self.edges.len() / 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triangle() {
        // Triangle: 0-1, 1-2, 0-2, all weight 1.0
        let g = SparseGraph::from_edges(
            3,
            &[0, 1, 0],
            &[1, 2, 2],
            &[1.0, 1.0, 1.0],
        );
        assert_eq!(g.n, 3);
        assert_eq!(g.num_edges(), 3);
        // Each vertex has 2 neighbors
        assert_eq!(g.neighbors(0).len(), 2);
        assert_eq!(g.neighbors(1).len(), 2);
        assert_eq!(g.neighbors(2).len(), 2);
    }

    #[test]
    fn test_self_loop_skipped() {
        let g = SparseGraph::from_edges(
            2,
            &[0, 0],
            &[1, 0], // second edge is self-loop
            &[1.0, 5.0],
        );
        assert_eq!(g.num_edges(), 1);
        assert_eq!(g.neighbors(0).len(), 1);
    }
}
