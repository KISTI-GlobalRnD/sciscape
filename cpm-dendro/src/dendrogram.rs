/// Dendrogram data structure and scipy-compatible linkage matrix conversion.
///
/// The linkage matrix format matches `scipy.cluster.hierarchy.linkage`:
///   Each row: [left_id, right_id, merge_height, subtree_size]
///   - Leaf nodes: 0..n-1
///   - Internal nodes: n..2n-2 (row i corresponds to node n+i)
///   - Rows are ordered by merge sequence

/// A single merge event in the dendrogram.
#[derive(Debug, Clone, Copy)]
pub struct MergeRecord {
    /// Left child (leaf index 0..n-1 or internal node index n..2n-2)
    pub left: u32,
    /// Right child
    pub right: u32,
    /// Merge height = CPM critical resolution γ* = e_AB / (|A| · |B|)
    pub height: f64,
    /// Number of leaves in the merged subtree
    pub size: u32,
}

/// Complete dendrogram: sequence of n-1 merges.
#[derive(Debug)]
pub struct DendrogramData {
    pub n_leaves: usize,
    pub merges: Vec<MergeRecord>,
}

impl DendrogramData {
    /// Create a new empty dendrogram for `n` leaves.
    pub fn new(n_leaves: usize) -> Self {
        DendrogramData {
            n_leaves,
            merges: Vec::with_capacity(n_leaves.saturating_sub(1)),
        }
    }

    /// Record a merge and return the new internal node ID.
    pub fn record_merge(&mut self, left: u32, right: u32, height: f64, size: u32) -> u32 {
        let node_id = (self.n_leaves + self.merges.len()) as u32;
        self.merges.push(MergeRecord {
            left,
            right,
            height,
            size,
        });
        node_id
    }

    /// Convert to a flat array suitable for numpy: shape (n-1, 4).
    /// Returns a Vec<f64> in row-major order.
    pub fn to_linkage_flat(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.merges.len() * 4);
        for m in &self.merges {
            out.push(m.left as f64);
            out.push(m.right as f64);
            out.push(m.height);
            out.push(m.size as f64);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_dendrogram() {
        let mut d = DendrogramData::new(3);
        let n4 = d.record_merge(0, 1, 0.5, 2);
        assert_eq!(n4, 3);
        let n5 = d.record_merge(n4, 2, 0.2, 3);
        assert_eq!(n5, 4);

        let flat = d.to_linkage_flat();
        assert_eq!(flat.len(), 8); // 2 merges × 4 columns
        // First merge: (0, 1, 0.5, 2)
        assert_eq!(flat[0], 0.0);
        assert_eq!(flat[1], 1.0);
        assert_eq!(flat[2], 0.5);
        assert_eq!(flat[3], 2.0);
    }
}
