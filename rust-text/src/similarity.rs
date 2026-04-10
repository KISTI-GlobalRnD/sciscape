//! Pairwise similarity layers: string-level and token-level.
//!
//! Produces COO sparse matrix entries (rows, cols, vals) for scipy consumption.

use crate::edit_distance::edit_distance_threshold;
use ahash::{AHashMap, AHashSet};
use rayon::prelude::*;

/// Character n-grams of a string.
fn char_ngrams(s: &str, n: usize) -> AHashSet<String> {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() < n {
        let mut set = AHashSet::new();
        set.insert(s.to_string());
        return set;
    }
    let mut set = AHashSet::with_capacity(chars.len() - n + 1);
    for i in 0..=(chars.len() - n) {
        set.insert(chars[i..i + n].iter().collect());
    }
    set
}

/// Jaccard similarity between two sets.
fn jaccard(a: &AHashSet<String>, b: &AHashSet<String>) -> f32 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }
    let inter = a.intersection(b).count();
    let union = a.len() + b.len() - inter;
    if union == 0 { 0.0 } else { inter as f32 / union as f32 }
}

/// Build blocks for pairwise comparison.
///
/// `strategy`: "prefix" or "token".
/// Also creates an "_abbrev" block pairing short terms with multi-word terms.
fn build_blocks(terms: &[String], prefix_len: usize, strategy: &str) -> AHashMap<String, Vec<usize>> {
    let mut blocks: AHashMap<String, Vec<usize>> = AHashMap::new();
    let mut abbrev_candidates: Vec<usize> = Vec::new();
    let mut multi_word: Vec<usize> = Vec::new();

    for (i, t) in terms.iter().enumerate() {
        let lower = t.to_lowercase();
        let nospace: String = lower.chars().filter(|c| !c.is_whitespace()).collect();

        if strategy == "token" {
            let tokens: Vec<&str> = lower.split_whitespace().collect();
            if tokens.is_empty() {
                blocks.entry(String::new()).or_default().push(i);
            } else {
                for token in &tokens {
                    blocks.entry(token.to_string()).or_default().push(i);
                }
            }
            if nospace.len() <= 5 {
                abbrev_candidates.push(i);
            } else if tokens.len() > 1 {
                multi_word.push(i);
            }
        } else {
            // prefix blocking
            let key: String = lower.chars().take(prefix_len).collect();
            blocks.entry(key).or_default().push(i);
            if nospace.len() <= 5 {
                abbrev_candidates.push(i);
            } else if lower.contains(' ') {
                multi_word.push(i);
            }
        }
    }

    // Abbreviation block: pair short terms with multi-word terms
    if !abbrev_candidates.is_empty() && !multi_word.is_empty() {
        let mut abbrev_block = abbrev_candidates;
        abbrev_block.extend(multi_word);
        blocks.insert("_abbrev".to_string(), abbrev_block);
    }

    blocks
}

/// Result of similarity computation: COO sparse matrix entries.
pub struct CooEntries {
    pub rows: Vec<u32>,
    pub cols: Vec<u32>,
    pub vals: Vec<f32>,
    pub n: usize,
}

/// Build string-level similarity layer (edit distance + char n-gram).
///
/// Returns COO entries for a symmetric sparse matrix.
pub fn build_layer_string(
    terms: &[String],
    char_ngram_n: usize,
    max_edit_distance: usize,
    min_sim: f32,
    max_block_size: usize,
    prefix_len: usize,
    blocking_strategy: &str,
) -> CooEntries {
    let n = terms.len();
    let lower_terms: Vec<String> = terms.iter().map(|t| t.to_lowercase()).collect();

    // Pre-compute char n-grams
    let ngrams: Vec<AHashSet<String>> = lower_terms
        .par_iter()
        .map(|t| char_ngrams(t, char_ngram_n))
        .collect();

    let blocks = build_blocks(terms, prefix_len, blocking_strategy);

    // Process blocks in parallel
    let block_results: Vec<(Vec<u32>, Vec<u32>, Vec<f32>)> = blocks
        .par_iter()
        .filter_map(|(_, indices)| {
            if indices.len() < 2 || indices.len() > max_block_size {
                return None;
            }
            let mut rows = Vec::new();
            let mut cols = Vec::new();
            let mut vals = Vec::new();

            for ii in 0..indices.len() {
                let i = indices[ii];
                for jj in (ii + 1)..indices.len() {
                    let j = indices[jj];

                    // Edit distance similarity
                    let ed_sim = if let Some(dist) = edit_distance_threshold(
                        &lower_terms[i], &lower_terms[j], max_edit_distance,
                    ) {
                        let max_len = lower_terms[i].len().max(lower_terms[j].len());
                        if max_len == 0 { 0.0 } else { 1.0 - (dist as f32 / max_len as f32) }
                    } else {
                        0.0
                    };

                    // Char n-gram Jaccard
                    let ng_sim = jaccard(&ngrams[i], &ngrams[j]);

                    let sim = ed_sim.max(ng_sim);
                    if sim >= min_sim {
                        rows.push(i as u32);
                        rows.push(j as u32);
                        cols.push(j as u32);
                        cols.push(i as u32);
                        vals.push(sim);
                        vals.push(sim);
                    }
                }
            }
            if rows.is_empty() { None } else { Some((rows, cols, vals)) }
        })
        .collect();

    // Merge results
    let total: usize = block_results.iter().map(|(r, _, _)| r.len()).sum();
    let mut all_rows = Vec::with_capacity(total);
    let mut all_cols = Vec::with_capacity(total);
    let mut all_vals = Vec::with_capacity(total);
    for (r, c, v) in block_results {
        all_rows.extend(r);
        all_cols.extend(c);
        all_vals.extend(v);
    }

    CooEntries { rows: all_rows, cols: all_cols, vals: all_vals, n }
}

/// Build token-level similarity layer (Jaccard, containment, abbreviation).
pub fn build_layer_token(
    terms: &[String],
    min_sim: f32,
    max_block_size: usize,
    prefix_len: usize,
    blocking_strategy: &str,
) -> CooEntries {
    let n = terms.len();
    let lower_terms: Vec<String> = terms.iter().map(|t| t.to_lowercase()).collect();

    // Pre-compute token sets
    let token_sets: Vec<AHashSet<String>> = lower_terms
        .iter()
        .map(|t| t.split_whitespace().map(String::from).collect())
        .collect();

    let blocks = build_blocks(terms, prefix_len, blocking_strategy);

    let block_results: Vec<(Vec<u32>, Vec<u32>, Vec<f32>)> = blocks
        .par_iter()
        .filter_map(|(_, indices)| {
            if indices.len() < 2 || indices.len() > max_block_size {
                return None;
            }
            let mut rows = Vec::new();
            let mut cols = Vec::new();
            let mut vals = Vec::new();

            for ii in 0..indices.len() {
                let i = indices[ii];
                for jj in (ii + 1)..indices.len() {
                    let j = indices[jj];
                    let ts_i = &token_sets[i];
                    let ts_j = &token_sets[j];
                    if ts_i.is_empty() || ts_j.is_empty() {
                        continue;
                    }

                    // Jaccard
                    let inter = ts_i.intersection(ts_j).count();
                    let union = ts_i.len() + ts_j.len() - inter;
                    let overlap = if union == 0 { 0.0 } else { inter as f32 / union as f32 };

                    // Containment
                    let containment = if ts_i.is_subset(ts_j) || ts_j.is_subset(ts_i) {
                        inter as f32 / ts_i.len().min(ts_j.len()) as f32
                    } else {
                        0.0
                    };

                    // Abbreviation check
                    let mut abbrev_sim: f32 = 0.0;
                    let ti = &lower_terms[i];
                    let tj = &lower_terms[j];
                    let ti_nospace: String = ti.chars().filter(|c| !c.is_whitespace()).collect();
                    let tj_nospace: String = tj.chars().filter(|c| !c.is_whitespace()).collect();
                    let ti_words: Vec<&str> = ti.split_whitespace().collect();
                    let tj_words: Vec<&str> = tj.split_whitespace().collect();

                    if ti_nospace.len() <= 5 && tj_words.len() > 1 {
                        let initials: String = tj_words.iter()
                            .filter_map(|w| w.chars().next())
                            .collect();
                        if ti_nospace == initials {
                            abbrev_sim = 0.9;
                        }
                    }
                    if tj_nospace.len() <= 5 && ti_words.len() > 1 {
                        let initials: String = ti_words.iter()
                            .filter_map(|w| w.chars().next())
                            .collect();
                        if tj_nospace == initials {
                            abbrev_sim = 0.9;
                        }
                    }

                    let sim = overlap.max(containment).max(abbrev_sim);
                    if sim >= min_sim {
                        rows.push(i as u32);
                        rows.push(j as u32);
                        cols.push(j as u32);
                        cols.push(i as u32);
                        vals.push(sim);
                        vals.push(sim);
                    }
                }
            }
            if rows.is_empty() { None } else { Some((rows, cols, vals)) }
        })
        .collect();

    let total: usize = block_results.iter().map(|(r, _, _)| r.len()).sum();
    let mut all_rows = Vec::with_capacity(total);
    let mut all_cols = Vec::with_capacity(total);
    let mut all_vals = Vec::with_capacity(total);
    for (r, c, v) in block_results {
        all_rows.extend(r);
        all_cols.extend(c);
        all_vals.extend(v);
    }

    CooEntries { rows: all_rows, cols: all_cols, vals: all_vals, n }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_char_ngrams() {
        let ng = char_ngrams("hello", 3);
        assert!(ng.contains("hel"));
        assert!(ng.contains("ell"));
        assert!(ng.contains("llo"));
        assert_eq!(ng.len(), 3);
    }

    #[test]
    fn test_build_layer_string() {
        let terms: Vec<String> = vec![
            "neural".into(), "neurol".into(), "network".into(),
        ];
        let result = build_layer_string(&terms, 3, 2, 0.5, 1000, 3, "prefix");
        // neural vs neurol: edit distance = 1, sim = 1 - 1/6 ≈ 0.833
        assert!(!result.rows.is_empty());
    }

    #[test]
    fn test_build_layer_token() {
        let terms: Vec<String> = vec![
            "machine learning".into(),
            "machine vision".into(),
            "quantum computing".into(),
        ];
        // Same prefix "mac" → same block. Jaccard({"machine","learning"}, {"machine","vision"}) = 1/3
        let result = build_layer_token(&terms, 0.3, 1000, 3, "prefix");
        assert!(!result.rows.is_empty());
    }
}
