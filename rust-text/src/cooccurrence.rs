//! Co-occurrence matrix accumulation using parallel COO construction.
//!
//! Replaces Python's nested loop + lil_matrix with flat COO arrays
//! aggregated via sort + merge.

use ahash::AHashMap;
use rayon::prelude::*;

/// Result: COO entries for symmetric co-occurrence matrix.
pub struct CoocResult {
    pub rows: Vec<u32>,
    pub cols: Vec<u32>,
    pub vals: Vec<i64>,
    pub n: usize,
}

/// Accumulate co-occurrence counts from per-document term index lists.
///
/// `doc_term_indices`: for each document, a sorted list of term indices
///     that appear in that document. Shape: Vec<Vec<u32>>.
/// `n_terms`: total number of unique terms (matrix dimension).
/// `min_count`: minimum co-occurrence count to keep.
///
/// Returns COO entries for a symmetric (n_terms × n_terms) matrix.
pub fn collect_cooccurrence(
    doc_term_indices: &[Vec<u32>],
    n_terms: usize,
    min_count: i64,
) -> CoocResult {
    // Parallel: each chunk of documents produces local COO entries
    let chunk_size = (doc_term_indices.len() / rayon::current_num_threads().max(1)).max(256);

    let local_maps: Vec<AHashMap<(u32, u32), i64>> = doc_term_indices
        .par_chunks(chunk_size)
        .map(|chunk| {
            let mut counts: AHashMap<(u32, u32), i64> = AHashMap::new();
            for indices in chunk {
                let n = indices.len();
                if n < 2 { continue; }
                for i in 0..n {
                    for j in (i + 1)..n {
                        let (ti, tj) = if indices[i] <= indices[j] {
                            (indices[i], indices[j])
                        } else {
                            (indices[j], indices[i])
                        };
                        *counts.entry((ti, tj)).or_insert(0) += 1;
                    }
                }
            }
            counts
        })
        .collect();

    // Merge local maps
    let mut merged: AHashMap<(u32, u32), i64> = AHashMap::new();
    for local in local_maps {
        for ((ti, tj), count) in local {
            *merged.entry((ti, tj)).or_insert(0) += count;
        }
    }

    // Build symmetric COO
    let mut rows = Vec::with_capacity(merged.len() * 2);
    let mut cols = Vec::with_capacity(merged.len() * 2);
    let mut vals = Vec::with_capacity(merged.len() * 2);

    for ((ti, tj), count) in merged {
        if count < min_count {
            continue;
        }
        rows.push(ti);
        cols.push(tj);
        vals.push(count);
        if ti != tj {
            rows.push(tj);
            cols.push(ti);
            vals.push(count);
        }
    }

    CoocResult { rows, cols, vals, n: n_terms }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_cooccurrence() {
        // 3 terms, 2 documents
        // doc0: terms [0, 1, 2] → pairs (0,1), (0,2), (1,2)
        // doc1: terms [0, 1]    → pairs (0,1)
        let docs = vec![
            vec![0u32, 1, 2],
            vec![0, 1],
        ];
        let result = collect_cooccurrence(&docs, 3, 1);
        // (0,1) count = 2, (0,2) count = 1, (1,2) count = 1
        assert!(!result.rows.is_empty());

        // Check symmetric: each pair appears twice (i,j) and (j,i)
        let total_entries = result.rows.len();
        assert_eq!(total_entries % 2, 0);
    }

    #[test]
    fn test_min_count_filter() {
        let docs = vec![
            vec![0u32, 1, 2],
            vec![0, 1],
        ];
        let result = collect_cooccurrence(&docs, 3, 2);
        // Only (0,1) has count >= 2
        // So we get 2 entries: (0,1) and (1,0)
        assert_eq!(result.rows.len(), 2);
    }
}
