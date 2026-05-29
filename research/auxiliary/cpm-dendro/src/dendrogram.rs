/// Dendrogram data structure and linkage matrix conversion.
///
/// Internal format: rows ordered by merge sequence, heights are **similarity**
/// (CPM density, non-increasing). This differs from scipy's linkage format
/// where heights are **distances** (non-decreasing).
///
/// Layout per row: [left_id, right_id, merge_height, subtree_size]
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
    /// Merge height = pairwise merge density ρ(A,B) = e_AB / (|A| · |B|)
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

    /// Convert to flat similarity linkage: shape (n-1, 4), row-major.
    ///
    /// Heights are **density similarity** (non-increasing).
    /// NOT compatible with scipy's dendrogram/is_monotonic which expect distances.
    pub fn to_similarity_linkage_flat(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.merges.len() * 4);
        for m in &self.merges {
            out.push(m.left as f64);
            out.push(m.right as f64);
            out.push(m.height);
            out.push(m.size as f64);
        }
        out
    }

    /// Convert to scipy-compatible distance linkage: shape (n-1, 4), row-major.
    ///
    /// Heights are transformed to **distance** (non-decreasing) via:
    ///   distance = max_similarity - similarity
    ///
    /// This output is compatible with `scipy.cluster.hierarchy.dendrogram`,
    /// `is_valid_linkage`, and `is_monotonic`.
    ///
    /// **Note**: The transform `distance = max_similarity - similarity` is
    /// order-preserving but not the same scale as the raw density threshold.
    /// To cut at a raw CPM density γ, use scipy distance `t = max_h - γ`.
    pub fn to_scipy_distance_flat(&self) -> Vec<f64> {
        // Verify non-increasing invariant (our algorithm guarantees this)
        debug_assert!(
            self.merges.windows(2).all(|w| w[1].height <= w[0].height + 1e-12),
            "Merge heights must be non-increasing for similarity-to-distance conversion"
        );

        let max_h = self.merges.first().map_or(0.0, |m| m.height);
        let mut out = Vec::with_capacity(self.merges.len() * 4);
        for m in &self.merges {
            out.push(m.left as f64);
            out.push(m.right as f64);
            out.push(max_h - m.height);
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

        let flat = d.to_similarity_linkage_flat();
        assert_eq!(flat.len(), 8); // 2 merges × 4 columns
        // First merge: (0, 1, 0.5, 2)
        assert_eq!(flat[0], 0.0);
        assert_eq!(flat[1], 1.0);
        assert_eq!(flat[2], 0.5);
        assert_eq!(flat[3], 2.0);
    }

    #[test]
    fn test_scipy_distance_conversion() {
        let mut d = DendrogramData::new(3);
        d.record_merge(0, 1, 3.0, 2);  // highest density first
        d.record_merge(3, 2, 1.0, 3);  // lower density

        let dist = d.to_scipy_distance_flat();
        // max_h = 3.0
        // Row 0: distance = 3.0 - 3.0 = 0.0
        // Row 1: distance = 3.0 - 1.0 = 2.0
        assert_eq!(dist[2], 0.0);  // first merge distance
        assert_eq!(dist[6], 2.0);  // second merge distance
        // Non-decreasing: 0.0 <= 2.0 ✓
    }
}
