//! CPM-critical dendrogram construction via sparse average-linkage HAC.
//!
//! This crate provides a high-performance Rust implementation of hierarchical
//! agglomerative clustering using CPM (Constant Potts Model) density as the
//! merge criterion. Since CPM density ρ(A,B) = e_AB / (|A|·|B|) is
//! mathematically identical to average-linkage on the weighted adjacency,
//! this produces valid ultrametric dendrograms with guaranteed monotonic
//! merge heights.
//!
//! Exposed to Python via PyO3.

mod dendrogram;
mod graph;
mod hac;
mod triadic;

use numpy::pyo3::Python;
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;

/// Build a CPM-critical dendrogram from edge arrays.
///
/// Parameters
/// ----------
/// n : int
///     Number of vertices.
/// sources : numpy.ndarray[uint32]
///     Source vertex IDs for each edge.
/// targets : numpy.ndarray[uint32]
///     Target vertex IDs for each edge.
/// weights : numpy.ndarray[float64]
///     Edge weights.
/// triadic : bool, default False
///     If True, apply triadic closure reweighting before HAC.
///
/// Returns
/// -------
/// numpy.ndarray, shape (n-1, 4)
///     Scipy-compatible linkage matrix: [left, right, height, size].
#[pyfunction]
#[pyo3(signature = (n, sources, targets, weights, triadic = false))]
fn build_cpm_dendrogram<'py>(
    py: Python<'py>,
    n: usize,
    sources: &Bound<'py, PyArray1<u32>>,
    targets: &Bound<'py, PyArray1<u32>>,
    weights: &Bound<'py, PyArray1<f64>>,
    triadic: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    // Convert numpy arrays to Rust slices
    let src = unsafe { sources.as_slice()? };
    let tgt = unsafe { targets.as_slice()? };
    let wgt = unsafe { weights.as_slice()? };

    // Build sparse graph
    let mut g = graph::SparseGraph::from_edges(n, src, tgt, wgt);

    // Optional triadic closure reweighting
    if triadic {
        g = triadic::reweight_triadic(&g);
    }

    // Build dendrogram
    let dendro = hac::build_dendrogram(&g);

    // Convert to numpy array
    let flat = dendro.to_linkage_flat();
    let n_merges = dendro.merges.len();

    // Create (n-1, 4) numpy array
    // Build as Vec<Vec<f64>> for from_vec2
    let mut rows: Vec<Vec<f64>> = Vec::with_capacity(n_merges);
    for i in 0..n_merges {
        rows.push(vec![flat[i * 4], flat[i * 4 + 1], flat[i * 4 + 2], flat[i * 4 + 3]]);
    }
    let result = PyArray2::from_vec2(py, &rows)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{:?}", e)))?;

    Ok(result)
}

/// Python module definition.
#[pymodule]
fn cpm_dendro(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(build_cpm_dendrogram, m)?)?;
    Ok(())
}
