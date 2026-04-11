//! PyO3 bindings for sciscape-leiden.
//!
//! Exposes Leiden clustering to Python via numpy arrays.

#[cfg(feature = "python")]
use numpy::{PyArray1, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::conversion::IntoPyObject;

#[cfg(feature = "python")]
use crate::{Graph, Clustering, LeidenConfig, leiden, leiden_multi_start, postprocess_small_clusters};
#[cfg(feature = "python")]
use crate::quality::{CPM, QualityFunction};
#[cfg(feature = "python")]
use rand::SeedableRng;

/// Run CPM Leiden clustering on an edge list.
///
/// Args:
///     n_nodes: Total number of nodes.
///     src: Source node indices (u32 numpy array).
///     dst: Destination node indices (u32 numpy array).
///     weights: Edge weights (f64 numpy array).
///     resolution: CPM resolution parameter (gamma).
///     n_iterations: Number of Leiden iterations (0 = until convergence).
///     n_starts: Number of random starts (best quality kept).
///     randomness: Randomness parameter for refinement.
///     seed: Random seed.
///     initial_membership: Optional initial cluster assignment.
///     fixed_nodes: Optional boolean mask of fixed nodes.
///     node_weights: Optional per-node weights (doc_count for contracted graphs).
///         If None, all nodes have weight 1.0.
///
/// Returns:
///     Tuple of (membership: numpy array, quality: float, n_clusters: int).
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (
    n_nodes,
    src,
    dst,
    weights,
    resolution = 1.0,
    n_iterations = 10,
    n_starts = 1,
    randomness = 0.01,
    seed = 0,
    initial_membership = None,
    fixed_nodes = None,
    node_weights = None,
))]
fn run_leiden<'py>(
    py: Python<'py>,
    n_nodes: usize,
    src: PyReadonlyArray1<u32>,
    dst: PyReadonlyArray1<u32>,
    weights: PyReadonlyArray1<f64>,
    resolution: f64,
    n_iterations: usize,
    n_starts: usize,
    randomness: f64,
    seed: u64,
    initial_membership: Option<PyReadonlyArray1<u64>>,
    fixed_nodes: Option<PyReadonlyArray1<bool>>,
    node_weights: Option<PyReadonlyArray1<f64>>,
) -> PyResult<(Py<PyArray1<u64>>, f64, usize)> {
    let src_slice = src.as_slice()?;
    let dst_slice = dst.as_slice()?;
    let w_slice = weights.as_slice()?;

    let graph = if let Some(nw) = node_weights {
        Graph::from_edge_list_weighted(n_nodes, src_slice, dst_slice, w_slice, nw.as_slice()?)
    } else {
        Graph::from_edge_list(n_nodes, src_slice, dst_slice, w_slice)
    };

    let config = LeidenConfig {
        resolution,
        n_iterations,
        randomness,
        seed,
    };

    let initial = if let Some(mem) = initial_membership {
        let mem_slice = mem.as_slice()?;
        let mut clustering = Clustering::from_assignments(
            mem_slice.iter().map(|&x| x as usize).collect()
        );
        if let Some(fixed) = fixed_nodes {
            let fixed_slice = fixed.as_slice()?;
            clustering.set_fixed(fixed_slice.to_vec());
        }
        Some(clustering)
    } else {
        None
    };

    let result = if n_starts > 1 {
        leiden_multi_start(&graph, &config, n_starts, initial.as_ref())
    } else {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        leiden(&graph, &config, initial, &mut rng)
    };

    let membership: Vec<u64> = result.clustering.clusters.iter().map(|&c| c as u64).collect();
    let n_clusters = result.clustering.n_clusters;
    let quality = result.quality;

    let py_arr = PyArray1::from_vec(py, membership).into();
    Ok((py_arr, quality, n_clusters))
}

/// Reassign small clusters using constrained Leiden.
///
/// Args:
///     node_weights: Optional per-node weights. If provided, min_weight is used
///         instead of min_size for threshold comparison.
///     min_weight: Weighted threshold (sum of node_weights). Only used when
///         node_weights is provided. Default 0.0 means use min_size instead.
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (
    n_nodes,
    src,
    dst,
    weights,
    membership,
    resolution,
    min_size,
    n_iterations = 10,
    randomness = 0.01,
    seed = 0,
    node_weights = None,
    min_weight = 0.0,
    max_rounds = 5,
    gamma_decay = 0.1,
    use_greedy = true,
    greedy_anchor_only = false,
    greedy_fallback_to_small = false,
    greedy_max_weight = 0.0,
    use_component_merge = true,
    component_max_weight = 0.0,
))]
fn run_postprocess<'py>(
    py: Python<'py>,
    n_nodes: usize,
    src: PyReadonlyArray1<u32>,
    dst: PyReadonlyArray1<u32>,
    weights: PyReadonlyArray1<f64>,
    membership: PyReadonlyArray1<u64>,
    resolution: f64,
    min_size: usize,
    n_iterations: usize,
    randomness: f64,
    seed: u64,
    node_weights: Option<PyReadonlyArray1<f64>>,
    min_weight: f64,
    max_rounds: usize,
    gamma_decay: f64,
    use_greedy: bool,
    greedy_anchor_only: bool,
    greedy_fallback_to_small: bool,
    greedy_max_weight: f64,
    use_component_merge: bool,
    component_max_weight: f64,
) -> PyResult<(Py<PyArray1<u64>>, usize, Py<PyArray1<i32>>, Vec<std::collections::HashMap<String, pyo3::PyObject>>)> {
    let graph = if let Some(nw) = node_weights {
        Graph::from_edge_list_weighted(
            n_nodes,
            src.as_slice()?,
            dst.as_slice()?,
            weights.as_slice()?,
            nw.as_slice()?,
        )
    } else {
        Graph::from_edge_list(
            n_nodes,
            src.as_slice()?,
            dst.as_slice()?,
            weights.as_slice()?,
        )
    };

    let clustering = Clustering::from_assignments(
        membership.as_slice()?.iter().map(|&x| x as usize).collect()
    );

    let config = LeidenConfig {
        resolution,
        n_iterations,
        randomness,
        seed,
    };

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let pp_result = postprocess_small_clusters(
        &graph,
        &clustering,
        &config,
        min_size,
        min_weight,
        max_rounds,
        gamma_decay,
        use_greedy,
        greedy_anchor_only,
        greedy_fallback_to_small,
        greedy_max_weight,
        use_component_merge,
        component_max_weight,
        &mut rng,
    );

    let mem_out: Vec<u64> = pp_result.clustering.clusters.iter().map(|&c| c as u64).collect();
    let n_clusters = pp_result.clustering.n_clusters;
    let changed_at: Py<PyArray1<i32>> = PyArray1::from_vec(py, pp_result.changed_at_round).into();

    // Build rounds info as list of dicts
    let rounds_info: Vec<std::collections::HashMap<String, pyo3::PyObject>> = pp_result.rounds.iter().map(|r| {
        Python::with_gil(|py| {
            let mut d = std::collections::HashMap::new();
            d.insert("round".to_string(), r.round.into_pyobject(py).unwrap().into_any().unbind());
            d.insert("gamma".to_string(), r.gamma.into_pyobject(py).unwrap().into_any().unbind());
            d.insert("method".to_string(), r.method.clone().into_pyobject(py).unwrap().into_any().unbind());
            d.insert("n_small_before".to_string(), r.n_small_before.into_pyobject(py).unwrap().into_any().unbind());
            d.insert("n_small_after".to_string(), r.n_small_after.into_pyobject(py).unwrap().into_any().unbind());
            d.insert("n_merged".to_string(), r.n_merged.into_pyobject(py).unwrap().into_any().unbind());
            d.insert("n_total_clusters".to_string(), r.n_total_clusters.into_pyobject(py).unwrap().into_any().unbind());
            d.insert("max_cluster_size".to_string(), r.max_cluster_size.into_pyobject(py).unwrap().into_any().unbind());
            d.insert("max_cluster_weight".to_string(), r.max_cluster_weight.into_pyobject(py).unwrap().into_any().unbind());
            d
        })
    }).collect();

    let py_arr = PyArray1::from_vec(py, mem_out).into();
    Ok((py_arr, n_clusters, changed_at, rounds_info))
}

/// Compute CPM quality of a clustering.
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (n_nodes, src, dst, weights, membership, resolution, node_weights = None))]
fn cpm_quality(
    n_nodes: usize,
    src: PyReadonlyArray1<u32>,
    dst: PyReadonlyArray1<u32>,
    weights: PyReadonlyArray1<f64>,
    membership: PyReadonlyArray1<u64>,
    resolution: f64,
    node_weights: Option<PyReadonlyArray1<f64>>,
) -> PyResult<f64> {
    let graph = if let Some(nw) = node_weights {
        Graph::from_edge_list_weighted(
            n_nodes, src.as_slice()?, dst.as_slice()?, weights.as_slice()?, nw.as_slice()?,
        )
    } else {
        Graph::from_edge_list(
            n_nodes, src.as_slice()?, dst.as_slice()?, weights.as_slice()?,
        )
    };
    let clustering = Clustering::from_assignments(
        membership.as_slice()?.iter().map(|&x| x as usize).collect()
    );
    let cpm = CPM::new(resolution);
    Ok(cpm.quality(&graph, &clustering))
}

/// Python module definition.
#[cfg(feature = "python")]
#[pymodule]
fn sciscape_leiden(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_leiden, m)?)?;
    m.add_function(wrap_pyfunction!(run_postprocess, m)?)?;
    m.add_function(wrap_pyfunction!(cpm_quality, m)?)?;
    Ok(())
}
