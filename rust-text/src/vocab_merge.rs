//! Edit-distance merge map builder — replaces todense() bottleneck.
//!
//! Operates on CSC sparse column data directly, avoiding dense conversion.

use crate::edit_distance::edit_distance_threshold;
use ahash::{AHashMap, AHashSet};
use rayon::prelude::*;

/// Sparse column representation (CSC-like): indices and values for one column.
struct SparseCol<'a> {
    indices: &'a [u32],
    values: &'a [f64],
}

impl<'a> SparseCol<'a> {
    /// Check if `minor` leads anywhere (minor_col[k] > major_col[k] for any k).
    /// Operates on sparse data directly — no dense conversion.
    fn minor_leads_anywhere(minor: &SparseCol, major: &SparseCol) -> bool {
        // Build a quick lookup for major values
        let mut major_map: AHashMap<u32, f64> = AHashMap::with_capacity(major.indices.len());
        for (&idx, &val) in major.indices.iter().zip(major.values.iter()) {
            major_map.insert(idx, val);
        }
        // Check if minor exceeds major in any row
        for (&idx, &val) in minor.indices.iter().zip(minor.values.iter()) {
            let major_val = major_map.get(&idx).copied().unwrap_or(0.0);
            if val > major_val {
                return true;
            }
        }
        false
    }
}

/// Build edit-distance merge map from CSC sparse matrix data.
///
/// Parameters:
/// - `feature_names`: vocabulary terms
/// - `existing_merge_keys`, `existing_merge_vals`: already-merged index pairs
/// - `csc_indptr`: CSC column pointer array (length = n_cols + 1)
/// - `csc_indices`: CSC row index array
/// - `csc_data`: CSC data array (f64 for generality)
/// - `col_sums`: precomputed column sums (length = n_cols)
/// - `max_edit_distance`: threshold
/// - `global_ratio_threshold`: frequency ratio below which merge is considered
///
/// Returns: Vec<(source_idx, target_idx)> merge pairs
pub fn build_edit_distance_merge_map(
    feature_names: &[String],
    existing_merge_keys: &[u32],
    existing_merge_vals: &[u32],
    csc_indptr: &[u64],
    csc_indices: &[u32],
    csc_data: &[f64],
    col_sums: &[f64],
    max_edit_distance: usize,
    global_ratio_threshold: f64,
) -> Vec<(u32, u32)> {
    let involved: AHashSet<u32> = existing_merge_keys.iter()
        .chain(existing_merge_vals.iter())
        .copied()
        .collect();

    // Filter to unigrams not already merged, len > 3
    let unigram_indices: Vec<usize> = feature_names.iter().enumerate()
        .filter(|(i, name)| {
            !name.contains(' ')
                && !involved.contains(&(*i as u32))
                && name.len() > 3
        })
        .map(|(i, _)| i)
        .collect();

    if unigram_indices.len() < 2 {
        return Vec::new();
    }

    // Block by prefix
    let prefix_len = 3;
    let mut blocks: AHashMap<String, Vec<usize>> = AHashMap::new();
    for &idx in &unigram_indices {
        let name = feature_names[idx].to_lowercase();
        let key: String = name.chars().take(prefix_len).collect();
        blocks.entry(key).or_default().push(idx);
    }

    // Process blocks (sequential — merge_map needs consistent ordering)
    let mut merge_map: AHashMap<usize, usize> = AHashMap::new();

    for block_indices in blocks.values() {
        if block_indices.len() < 2 {
            continue;
        }
        // Sort by frequency descending
        let mut sorted = block_indices.clone();
        sorted.sort_by(|&a, &b| col_sums[b].total_cmp(&col_sums[a]));

        for bi in 0..sorted.len() {
            let idx_i = sorted[bi];
            if merge_map.contains_key(&idx_i) {
                continue;
            }
            let name_i = feature_names[idx_i].to_lowercase();

            for bj in (bi + 1)..sorted.len() {
                let idx_j = sorted[bj];
                if merge_map.contains_key(&idx_j) {
                    continue;
                }
                let name_j = feature_names[idx_j].to_lowercase();

                // Edit distance check
                if edit_distance_threshold(&name_i, &name_j, max_edit_distance).is_none() {
                    continue;
                }

                // Frequency ratio check
                let freq_i = col_sums[idx_i];
                let freq_j = col_sums[idx_j];
                let major_freq = freq_i.max(freq_j);
                let minor_freq = freq_i.min(freq_j);
                if major_freq == 0.0 {
                    continue;
                }
                let ratio = minor_freq / major_freq;
                if ratio >= global_ratio_threshold {
                    continue;
                }

                // Cluster dominance check — SPARSE, no todense()
                let (major_idx, minor_idx) = if freq_i >= freq_j {
                    (idx_i, idx_j)
                } else {
                    (idx_j, idx_i)
                };

                let minor_start = csc_indptr[minor_idx] as usize;
                let minor_end = csc_indptr[minor_idx + 1] as usize;
                let minor_col = SparseCol {
                    indices: &csc_indices[minor_start..minor_end],
                    values: &csc_data[minor_start..minor_end],
                };

                let major_start = csc_indptr[major_idx] as usize;
                let major_end = csc_indptr[major_idx + 1] as usize;
                let major_col = SparseCol {
                    indices: &csc_indices[major_start..major_end],
                    values: &csc_data[major_start..major_end],
                };

                if SparseCol::minor_leads_anywhere(&minor_col, &major_col) {
                    continue;
                }

                merge_map.insert(minor_idx, major_idx);
            }
        }
    }

    merge_map.into_iter()
        .map(|(k, v)| (k as u32, v as u32))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_merge() {
        let names = vec![
            "clustering".to_string(),  // 0 — major
            "clustring".to_string(),   // 1 — typo (edit dist 1, same prefix "clu")
            "network".to_string(),     // 2
        ];
        // CSC: 2 clusters × 3 terms
        // cluster 0: clustering=100, clustring=1, network=50
        // cluster 1: clustering=80,  clustring=0, network=40
        let indptr = vec![0u64, 2, 3, 5];
        let indices = vec![0u32, 1, 0, 0, 1];
        let data = vec![100.0, 80.0, 1.0, 50.0, 40.0];
        let col_sums = vec![180.0, 1.0, 90.0];

        let result = build_edit_distance_merge_map(
            &names, &[], &[], &indptr, &indices, &data, &col_sums,
            1, 0.01,
        );
        // clustring (1) should merge into clustering (0)
        assert_eq!(result.len(), 1);
        assert!(result.contains(&(1, 0)));
    }
}
