//! PyO3 bindings for sciscape-text.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::{build_layer_string, build_layer_token, collect_cooccurrence, build_edit_distance_merge_map};

/// Build string-level similarity layer (edit distance + char n-gram).
/// Returns (rows, cols, vals, n) for scipy.sparse.coo_matrix construction.
#[pyfunction]
#[pyo3(signature = (
    terms,
    char_ngram_n = 3,
    max_edit_distance = 2,
    min_sim = 0.5,
    max_block_size = 5000,
    prefix_len = 3,
    blocking_strategy = "prefix",
))]
fn rust_build_layer_string<'py>(
    py: Python<'py>,
    terms: Vec<String>,
    char_ngram_n: usize,
    max_edit_distance: usize,
    min_sim: f32,
    max_block_size: usize,
    prefix_len: usize,
    blocking_strategy: &str,
) -> PyResult<(Py<PyArray1<u32>>, Py<PyArray1<u32>>, Py<PyArray1<f32>>, usize)> {
    let result = build_layer_string(
        &terms, char_ngram_n, max_edit_distance, min_sim, max_block_size, prefix_len, blocking_strategy,
    );
    Ok((
        PyArray1::from_vec(py, result.rows).into(),
        PyArray1::from_vec(py, result.cols).into(),
        PyArray1::from_vec(py, result.vals).into(),
        result.n,
    ))
}

/// Build token-level similarity layer (Jaccard, containment, abbreviation).
/// Returns (rows, cols, vals, n).
#[pyfunction]
#[pyo3(signature = (
    terms,
    min_sim = 0.3,
    max_block_size = 5000,
    prefix_len = 3,
    blocking_strategy = "prefix",
))]
fn rust_build_layer_token<'py>(
    py: Python<'py>,
    terms: Vec<String>,
    min_sim: f32,
    max_block_size: usize,
    prefix_len: usize,
    blocking_strategy: &str,
) -> PyResult<(Py<PyArray1<u32>>, Py<PyArray1<u32>>, Py<PyArray1<f32>>, usize)> {
    let result = build_layer_token(&terms, min_sim, max_block_size, prefix_len, blocking_strategy);
    Ok((
        PyArray1::from_vec(py, result.rows).into(),
        PyArray1::from_vec(py, result.cols).into(),
        PyArray1::from_vec(py, result.vals).into(),
        result.n,
    ))
}

/// Accumulate co-occurrence matrix from per-document term index lists.
/// Returns (rows, cols, vals, n) for scipy.sparse.coo_matrix.
#[pyfunction]
#[pyo3(signature = (doc_term_indices, n_terms, min_count = 1))]
fn rust_collect_cooccurrence<'py>(
    py: Python<'py>,
    doc_term_indices: Vec<Vec<u32>>,
    n_terms: usize,
    min_count: i64,
) -> PyResult<(Py<PyArray1<u32>>, Py<PyArray1<u32>>, Py<PyArray1<i64>>, usize)> {
    let result = collect_cooccurrence(&doc_term_indices, n_terms, min_count);
    Ok((
        PyArray1::from_vec(py, result.rows).into(),
        PyArray1::from_vec(py, result.cols).into(),
        PyArray1::from_vec(py, result.vals).into(),
        result.n,
    ))
}

/// Build edit-distance merge map using sparse column data (no todense).
/// Returns list of (source_idx, target_idx) pairs.
#[pyfunction]
#[pyo3(signature = (
    feature_names,
    existing_merge_keys,
    existing_merge_vals,
    csc_indptr,
    csc_indices,
    csc_data,
    col_sums,
    max_edit_distance = 1,
    global_ratio_threshold = 0.01,
))]
fn rust_build_edit_distance_merge_map(
    feature_names: Vec<String>,
    existing_merge_keys: PyReadonlyArray1<u32>,
    existing_merge_vals: PyReadonlyArray1<u32>,
    csc_indptr: PyReadonlyArray1<u64>,
    csc_indices: PyReadonlyArray1<u32>,
    csc_data: PyReadonlyArray1<f64>,
    col_sums: PyReadonlyArray1<f64>,
    max_edit_distance: usize,
    global_ratio_threshold: f64,
) -> PyResult<Vec<(u32, u32)>> {
    let result = build_edit_distance_merge_map(
        &feature_names,
        existing_merge_keys.as_slice()?,
        existing_merge_vals.as_slice()?,
        csc_indptr.as_slice()?,
        csc_indices.as_slice()?,
        csc_data.as_slice()?,
        col_sums.as_slice()?,
        max_edit_distance,
        global_ratio_threshold,
    );
    Ok(result)
}

/// Levenshtein edit distance between two strings.
#[pyfunction]
fn rust_edit_distance(a: &str, b: &str) -> usize {
    crate::edit_distance::edit_distance(a, b)
}

#[pymodule]
fn sciscape_text(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rust_build_layer_string, m)?)?;
    m.add_function(wrap_pyfunction!(rust_build_layer_token, m)?)?;
    m.add_function(wrap_pyfunction!(rust_collect_cooccurrence, m)?)?;
    m.add_function(wrap_pyfunction!(rust_build_edit_distance_merge_map, m)?)?;
    m.add_function(wrap_pyfunction!(rust_edit_distance, m)?)?;
    Ok(())
}
