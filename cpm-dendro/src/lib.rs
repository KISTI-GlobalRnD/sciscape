//! Greedy CPM-density dendrogram construction via sparse average-linkage HAC.
//!
//! This crate provides a Rust implementation of hierarchical agglomerative
//! clustering using CPM (Constant Potts Model) density as the merge criterion.
//! ρ(A,B) = e_AB / (|A|·|B|) is mathematically identical to average-linkage
//! on the weighted adjacency matrix (zeros for non-edges).
//!
//! **Important**: This is a greedy approximation. The resulting tree does not
//! necessarily contain the global CPM optimum partition for every γ.
//!
//! ## Modes
//!
//! - `"cpm"` (default): raw CPM density ρ(A,B) = e_AB / (|A|·|B|)
//! - `"triadic_cpm"`: triadic closure preprocessing, then CPM density.
//!   Heights reflect triadic-boosted density, not raw graph density.
//!
//! Exposed to Python via PyO3.

mod dendrogram;
mod graph;
mod hac;
mod triadic;

use numpy::pyo3::Python;
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;

/// Build a greedy CPM-density dendrogram from edge arrays.
///
/// Parameters
/// ----------
/// n : int
///     Number of vertices.
/// sources : numpy.ndarray[uint32]
///     Source vertex IDs for each edge. All values must be < n.
/// targets : numpy.ndarray[uint32]
///     Target vertex IDs for each edge. All values must be < n.
/// weights : numpy.ndarray[float64]
///     Edge weights. Must be finite and non-negative.
///     Duplicate edges are coalesced (weights summed).
///     Zero-weight edges are treated as absent.
/// node_sizes : numpy.ndarray[uint64] or None, default None
///     Initial sizes for each leaf node.  When building a dendrogram on a
///     **contracted graph** (supernodes), pass the original node counts so
///     that CPM density ρ(A,B) = e_AB / (|A|·|B|) uses the true sizes.
///     If None, every leaf has size 1 (standard HAC).
/// mode : str, default "cpm"
///     Scoring mode:
///     - ``"cpm"``: inter-cluster CPM density ρ(A,B) = e_AB / (|A|·|B|)
///     - ``"triadic_cpm"``: triadic closure reweighting + CPM density
///     - ``"internal_density"``: merged internal density ρ(A∪B) = (e_A+e_B+e_AB)/C(|A|+|B|,2)
///     - ``"triadic_internal_density"``: triadic reweighting + internal density
/// as_distance : bool, default False
///     If True, return scipy-compatible distance linkage (non-decreasing heights).
///     If False, return similarity linkage (non-increasing heights).
///
/// Returns
/// -------
/// numpy.ndarray, shape (n-1, 4)
///     Linkage matrix: [left, right, height, size].
///
/// Raises
/// ------
/// ValueError
///     If n=0, arrays have mismatched lengths, vertex IDs >= n,
///     weights contain NaN/Inf/negative values, or mode is unknown.
#[pyfunction]
#[pyo3(signature = (n, sources, targets, weights, node_sizes = None, mode = "cpm", as_distance = false))]
fn build_cpm_dendrogram<'py>(
    py: Python<'py>,
    n: usize,
    sources: &Bound<'py, PyArray1<u32>>,
    targets: &Bound<'py, PyArray1<u32>>,
    weights: &Bound<'py, PyArray1<f64>>,
    node_sizes: Option<&Bound<'py, PyArray1<u64>>>,
    mode: &str,
    as_distance: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    // --- Input validation ---
    if n == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "n must be >= 1",
        ));
    }

    let (use_triadic, strategy) = match mode {
        "cpm" => (false, hac::Strategy::Cpm),
        "triadic_cpm" => (true, hac::Strategy::Cpm),
        "internal_density" => (false, hac::Strategy::InternalDensity),
        "triadic_internal_density" => (true, hac::Strategy::InternalDensity),
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown mode '{}'. Supported: 'cpm', 'triadic_cpm', 'internal_density', 'triadic_internal_density'",
                mode
            )));
        }
    };

    // SAFETY: We hold the GIL and these arrays are not shared with other threads.
    let src = unsafe { sources.as_slice()? };
    let tgt = unsafe { targets.as_slice()? };
    let wgt = unsafe { weights.as_slice()? };

    if src.len() != tgt.len() || src.len() != wgt.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Array length mismatch: sources={}, targets={}, weights={}",
            src.len(),
            tgt.len(),
            wgt.len()
        )));
    }

    // Validate vertex IDs and weights
    for (i, (&s, &t)) in src.iter().zip(tgt.iter()).enumerate() {
        if s as usize >= n || t as usize >= n {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Edge {} has vertex ID >= n={}: source={}, target={}",
                i, n, s, t
            )));
        }
    }
    for (i, &w) in wgt.iter().enumerate() {
        if !w.is_finite() || w < 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Edge {} has invalid weight={}: must be finite and non-negative",
                i, w
            )));
        }
    }

    // Validate and extract node_sizes
    let ns_vec: Option<Vec<u64>> = if let Some(ns_arr) = node_sizes {
        let ns = unsafe { ns_arr.as_slice()? };
        if ns.len() != n {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "node_sizes length ({}) must equal n ({})",
                ns.len(),
                n
            )));
        }
        for (i, &s) in ns.iter().enumerate() {
            if s == 0 {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "node_sizes[{}] = 0: all sizes must be >= 1",
                    i
                )));
            }
        }
        Some(ns.to_vec())
    } else {
        None
    };

    // Build sparse graph
    let mut g = graph::SparseGraph::from_edges(n, src, tgt, wgt);

    // Apply mode-specific preprocessing
    if use_triadic {
        g = triadic::reweight_triadic(&g);
    }

    // Build dendrogram
    let dendro = hac::build_dendrogram(&g, strategy, ns_vec.as_deref());

    // Convert to numpy array
    let flat = if as_distance {
        dendro.to_scipy_distance_flat()
    } else {
        dendro.to_similarity_linkage_flat()
    };
    let n_merges = dendro.merges.len();

    // Create (n-1, 4) numpy array directly from flat Vec (single allocation)
    let flat_array = PyArray1::from_vec(py, flat);
    let result = flat_array
        .reshape([n_merges, 4])
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{:?}", e)))?;

    Ok(result)
}

/// Python module definition.
#[pymodule]
fn cpm_dendro(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(build_cpm_dendrogram, m)?)?;
    Ok(())
}
