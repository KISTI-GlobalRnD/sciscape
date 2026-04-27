//! PyO3 bindings for sciscape-leiden.
//!
//! Exposes Leiden clustering to Python via numpy arrays.

#[cfg(feature = "python")]
use numpy::{PyArray1, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::conversion::IntoPyObject;
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use std::fs;

#[cfg(feature = "python")]
use crate::quality::{QualityFunction, CPM};
#[cfg(feature = "python")]
use crate::{
    contraction::create_reduced_network, leiden, leiden_multi_start, postprocess_small_clusters,
    workspace::Workspace, Clustering, Graph, LeidenConfig,
};
#[cfg(feature = "python")]
use rand::SeedableRng;

#[cfg(feature = "python")]
#[pyclass(module = "sciscape_leiden")]
struct PyGraphHandle {
    graph: Graph,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyGraphHandle {
    #[getter]
    fn n_nodes(&self) -> usize {
        self.graph.n_nodes
    }

    #[getter]
    fn n_edges(&self) -> usize {
        self.graph.n_edges
    }
}

#[cfg(feature = "python")]
fn build_graph(
    n_nodes: usize,
    src: &[u32],
    dst: &[u32],
    weights: &[f64],
    node_weights: Option<&[f64]>,
) -> Graph {
    if let Some(nw) = node_weights {
        Graph::from_edge_list_weighted(n_nodes, src, dst, weights, nw)
    } else {
        Graph::from_edge_list(n_nodes, src, dst, weights)
    }
}

#[cfg(feature = "python")]
fn read_u32_bin(path: &str) -> PyResult<Vec<u32>> {
    let bytes = fs::read(path)?;
    if bytes.len() % 4 != 0 {
        return Err(PyValueError::new_err(format!(
            "u32 binary file length not divisible by 4: {path}"
        )));
    }
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        out.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

#[cfg(feature = "python")]
fn read_f64_bin(path: &str) -> PyResult<Vec<f64>> {
    let bytes = fs::read(path)?;
    if bytes.len() % 8 != 0 {
        return Err(PyValueError::new_err(format!(
            "f64 binary file length not divisible by 8: {path}"
        )));
    }
    let mut out = Vec::with_capacity(bytes.len() / 8);
    for chunk in bytes.chunks_exact(8) {
        out.push(f64::from_le_bytes([
            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
        ]));
    }
    Ok(out)
}

#[cfg(feature = "python")]
fn validate_len(name: &str, got: usize, expected: usize) -> PyResult<()> {
    if got != expected {
        return Err(PyValueError::new_err(format!(
            "{name} length mismatch: expected {expected} got {got}"
        )));
    }
    Ok(())
}

#[cfg(feature = "python")]
fn build_initial_clustering(
    n_nodes: usize,
    initial_membership: Option<PyReadonlyArray1<u64>>,
    fixed_nodes: Option<PyReadonlyArray1<bool>>,
) -> PyResult<Option<Clustering>> {
    let fixed = if let Some(fixed) = fixed_nodes {
        let fixed_slice = fixed.as_slice()?;
        validate_len("fixed_nodes", fixed_slice.len(), n_nodes)?;
        Some(fixed_slice.to_vec())
    } else {
        None
    };

    let mut initial = if let Some(mem) = initial_membership {
        let mem_slice = mem.as_slice()?;
        validate_len("initial_membership", mem_slice.len(), n_nodes)?;
        Some(Clustering::from_assignments(
            mem_slice.iter().map(|&x| x as u32).collect(),
        ))
    } else if fixed.is_some() {
        Some(Clustering::singleton(n_nodes))
    } else {
        None
    };

    if let Some(clustering) = initial.as_mut() {
        if let Some(fixed) = fixed {
            clustering.set_fixed(fixed);
        }
    }

    Ok(initial)
}

#[cfg(feature = "python")]
fn run_leiden_on_graph(
    graph: &Graph,
    config: &LeidenConfig,
    n_starts: usize,
    initial: Option<Clustering>,
) -> crate::leiden::LeidenResult {
    if n_starts > 1 {
        leiden_multi_start(graph, config, n_starts, initial.as_ref())
    } else {
        let mut rng = rand::rngs::StdRng::seed_from_u64(config.seed);
        leiden(graph, config, initial, &mut rng)
    }
}

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
    let node_weights_slice = node_weights.as_ref().map(|nw| nw.as_slice()).transpose()?;
    let graph = build_graph(n_nodes, src_slice, dst_slice, w_slice, node_weights_slice);

    let config = LeidenConfig {
        resolution,
        n_iterations,
        randomness,
        seed,
    };

    let initial = build_initial_clustering(n_nodes, initial_membership, fixed_nodes)?;
    let result = run_leiden_on_graph(&graph, &config, n_starts, initial);

    let membership: Vec<u64> = result
        .clustering
        .clusters
        .iter()
        .map(|&c| c as u64)
        .collect();
    let n_clusters = result.clustering.n_clusters;
    let quality = result.quality;

    let py_arr = PyArray1::from_vec(py, membership).into();
    Ok((py_arr, quality, n_clusters))
}

/// Build an opaque graph handle for repeated Leiden/postprocess runs.
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (
    n_nodes,
    src,
    dst,
    weights,
    node_weights = None,
))]
fn load_graph(
    n_nodes: usize,
    src: PyReadonlyArray1<u32>,
    dst: PyReadonlyArray1<u32>,
    weights: PyReadonlyArray1<f64>,
    node_weights: Option<PyReadonlyArray1<f64>>,
) -> PyResult<PyGraphHandle> {
    let node_weights_slice = node_weights.as_ref().map(|nw| nw.as_slice()).transpose()?;
    let graph = build_graph(
        n_nodes,
        src.as_slice()?,
        dst.as_slice()?,
        weights.as_slice()?,
        node_weights_slice,
    );
    Ok(PyGraphHandle { graph })
}

/// Build an opaque graph handle from raw binary edge-array files.
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (
    n_nodes,
    src_path,
    dst_path,
    weights_path,
    node_weights_path = None,
))]
fn load_graph_raw_files(
    n_nodes: usize,
    src_path: &str,
    dst_path: &str,
    weights_path: &str,
    node_weights_path: Option<&str>,
) -> PyResult<PyGraphHandle> {
    let src = read_u32_bin(src_path)?;
    let dst = read_u32_bin(dst_path)?;
    let weights = read_f64_bin(weights_path)?;
    if src.len() != dst.len() || src.len() != weights.len() {
        return Err(PyValueError::new_err(format!(
            "raw edge file lengths differ: src={} dst={} weights={}",
            src.len(),
            dst.len(),
            weights.len()
        )));
    }
    let node_weights = if let Some(path) = node_weights_path {
        let nw = read_f64_bin(path)?;
        if nw.len() != n_nodes {
            return Err(PyValueError::new_err(format!(
                "node_weights length mismatch: expected {} got {}",
                n_nodes,
                nw.len()
            )));
        }
        Some(nw)
    } else {
        None
    };
    let graph = build_graph(n_nodes, &src, &dst, &weights, node_weights.as_deref());
    Ok(PyGraphHandle { graph })
}

/// Contract a graph handle by cluster membership and return a new graph handle.
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (graph, membership, materialize_node_weights = true))]
fn contract_graph_handle<'py>(
    py: Python<'py>,
    graph: PyRef<'_, PyGraphHandle>,
    membership: PyReadonlyArray1<u64>,
    materialize_node_weights: bool,
) -> PyResult<(PyGraphHandle, Option<Py<PyArray1<f64>>>)> {
    let membership_slice = membership.as_slice()?;
    validate_len("membership", membership_slice.len(), graph.graph.n_nodes)?;
    let clustering =
        Clustering::from_assignments(membership_slice.iter().map(|&x| x as u32).collect());
    let mut ws = Workspace::new(graph.graph.n_nodes.max(clustering.n_clusters));
    let reduced = create_reduced_network(&graph.graph, &clustering, false, &mut ws);
    let node_weights = if materialize_node_weights {
        Some(
            PyArray1::from_vec(
                py,
                reduced
                    .node_weights
                    .clone()
                    .unwrap_or_else(|| vec![1.0; reduced.n_nodes]),
            )
            .into(),
        )
    } else {
        None
    };
    Ok((PyGraphHandle { graph: reduced }, node_weights))
}

/// Summarize a cluster membership against a preloaded graph handle.
#[cfg(feature = "python")]
#[pyfunction]
fn summarize_membership_handle(
    graph: PyRef<'_, PyGraphHandle>,
    membership: PyReadonlyArray1<u64>,
) -> PyResult<(usize, usize, f64, f64)> {
    let membership_slice = membership.as_slice()?;
    validate_len("membership", membership_slice.len(), graph.graph.n_nodes)?;
    let clustering =
        Clustering::from_assignments(membership_slice.iter().map(|&x| x as u32).collect());
    let mut sizes = vec![0u32; clustering.n_clusters];
    let mut weights = vec![0.0f64; clustering.n_clusters];
    let mut total_weight = 0.0f64;

    for node in 0..graph.graph.n_nodes {
        let cid = clustering.clusters[node] as usize;
        sizes[cid] += 1;
        let w = graph.graph.node_weight(node);
        weights[cid] += w;
        total_weight += w;
    }

    let mut active = 0usize;
    let mut max_size = 0usize;
    let mut max_weight = 0.0f64;
    for cid in 0..clustering.n_clusters {
        if sizes[cid] > 0 {
            active += 1;
            max_size = max_size.max(sizes[cid] as usize);
            max_weight = max_weight.max(weights[cid]);
        }
    }

    Ok((active, max_size, max_weight, total_weight))
}

/// Run CPM Leiden clustering on a preloaded graph handle.
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (
    graph,
    resolution = 1.0,
    n_iterations = 10,
    n_starts = 1,
    randomness = 0.01,
    seed = 0,
    initial_membership = None,
    fixed_nodes = None,
))]
fn run_leiden_handle<'py>(
    py: Python<'py>,
    graph: PyRef<'_, PyGraphHandle>,
    resolution: f64,
    n_iterations: usize,
    n_starts: usize,
    randomness: f64,
    seed: u64,
    initial_membership: Option<PyReadonlyArray1<u64>>,
    fixed_nodes: Option<PyReadonlyArray1<bool>>,
) -> PyResult<(Py<PyArray1<u64>>, f64, usize)> {
    let config = LeidenConfig {
        resolution,
        n_iterations,
        randomness,
        seed,
    };
    let initial = build_initial_clustering(graph.graph.n_nodes, initial_membership, fixed_nodes)?;
    let result = run_leiden_on_graph(&graph.graph, &config, n_starts, initial);

    let membership: Vec<u64> = result
        .clustering
        .clusters
        .iter()
        .map(|&c| c as u64)
        .collect();
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
    gamma_decay = 0.5,
    use_greedy = true,
    greedy_anchor_only = false,
    greedy_fallback_to_small = false,
    greedy_max_weight = 0.0,
    use_component_merge = true,
    component_max_weight = 0.0,
    track_changed_rounds = true,
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
    track_changed_rounds: bool,
) -> PyResult<(
    Py<PyArray1<u64>>,
    usize,
    Py<PyArray1<i32>>,
    Vec<std::collections::HashMap<String, pyo3::PyObject>>,
)> {
    let node_weights_slice = node_weights.as_ref().map(|nw| nw.as_slice()).transpose()?;
    let graph = build_graph(
        n_nodes,
        src.as_slice()?,
        dst.as_slice()?,
        weights.as_slice()?,
        node_weights_slice,
    );

    let membership_slice = membership.as_slice()?;
    validate_len("membership", membership_slice.len(), n_nodes)?;
    let clustering =
        Clustering::from_assignments(membership_slice.iter().map(|&x| x as u32).collect());

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
        track_changed_rounds,
        &mut rng,
    );

    let mem_out: Vec<u64> = pp_result
        .clustering
        .clusters
        .iter()
        .map(|&c| c as u64)
        .collect();
    let n_clusters = pp_result.clustering.n_clusters;
    let changed_at: Py<PyArray1<i32>> =
        PyArray1::from_vec(py, pp_result.changed_at_round.unwrap_or_default()).into();

    // Build rounds info as list of dicts
    let rounds_info: Vec<std::collections::HashMap<String, pyo3::PyObject>> = pp_result
        .rounds
        .iter()
        .map(|r| {
            let mut d = std::collections::HashMap::new();
            d.insert(
                "round".to_string(),
                r.round.into_pyobject(py).unwrap().into_any().unbind(),
            );
            d.insert(
                "gamma".to_string(),
                r.gamma.into_pyobject(py).unwrap().into_any().unbind(),
            );
            d.insert(
                "method".to_string(),
                r.method
                    .clone()
                    .into_pyobject(py)
                    .unwrap()
                    .into_any()
                    .unbind(),
            );
            d.insert(
                "n_small_before".to_string(),
                r.n_small_before
                    .into_pyobject(py)
                    .unwrap()
                    .into_any()
                    .unbind(),
            );
            d.insert(
                "n_small_after".to_string(),
                r.n_small_after
                    .into_pyobject(py)
                    .unwrap()
                    .into_any()
                    .unbind(),
            );
            d.insert(
                "n_merged".to_string(),
                r.n_merged.into_pyobject(py).unwrap().into_any().unbind(),
            );
            d.insert(
                "n_total_clusters".to_string(),
                r.n_total_clusters
                    .into_pyobject(py)
                    .unwrap()
                    .into_any()
                    .unbind(),
            );
            d.insert(
                "max_cluster_size".to_string(),
                r.max_cluster_size
                    .into_pyobject(py)
                    .unwrap()
                    .into_any()
                    .unbind(),
            );
            d.insert(
                "max_cluster_weight".to_string(),
                r.max_cluster_weight
                    .into_pyobject(py)
                    .unwrap()
                    .into_any()
                    .unbind(),
            );
            d
        })
        .collect();

    let py_arr = PyArray1::from_vec(py, mem_out).into();
    Ok((py_arr, n_clusters, changed_at, rounds_info))
}

/// Reassign small clusters using constrained Leiden on a preloaded graph handle.
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (
    graph,
    membership,
    resolution,
    min_size,
    n_iterations = 10,
    randomness = 0.01,
    seed = 0,
    min_weight = 0.0,
    max_rounds = 5,
    gamma_decay = 0.5,
    use_greedy = true,
    greedy_anchor_only = false,
    greedy_fallback_to_small = false,
    greedy_max_weight = 0.0,
    use_component_merge = true,
    component_max_weight = 0.0,
    track_changed_rounds = true,
))]
fn run_postprocess_handle<'py>(
    py: Python<'py>,
    graph: PyRef<'_, PyGraphHandle>,
    membership: PyReadonlyArray1<u64>,
    resolution: f64,
    min_size: usize,
    n_iterations: usize,
    randomness: f64,
    seed: u64,
    min_weight: f64,
    max_rounds: usize,
    gamma_decay: f64,
    use_greedy: bool,
    greedy_anchor_only: bool,
    greedy_fallback_to_small: bool,
    greedy_max_weight: f64,
    use_component_merge: bool,
    component_max_weight: f64,
    track_changed_rounds: bool,
) -> PyResult<(
    Py<PyArray1<u64>>,
    usize,
    Py<PyArray1<i32>>,
    Vec<std::collections::HashMap<String, pyo3::PyObject>>,
)> {
    let membership_slice = membership.as_slice()?;
    validate_len("membership", membership_slice.len(), graph.graph.n_nodes)?;
    let clustering =
        Clustering::from_assignments(membership_slice.iter().map(|&x| x as u32).collect());

    let config = LeidenConfig {
        resolution,
        n_iterations,
        randomness,
        seed,
    };

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let pp_result = postprocess_small_clusters(
        &graph.graph,
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
        track_changed_rounds,
        &mut rng,
    );

    let mem_out: Vec<u64> = pp_result
        .clustering
        .clusters
        .iter()
        .map(|&c| c as u64)
        .collect();
    let n_clusters = pp_result.clustering.n_clusters;
    let changed_at: Py<PyArray1<i32>> =
        PyArray1::from_vec(py, pp_result.changed_at_round.unwrap_or_default()).into();

    let rounds_info: Vec<std::collections::HashMap<String, pyo3::PyObject>> = pp_result
        .rounds
        .iter()
        .map(|r| {
            let mut d = std::collections::HashMap::new();
            d.insert(
                "round".to_string(),
                r.round.into_pyobject(py).unwrap().into_any().unbind(),
            );
            d.insert(
                "gamma".to_string(),
                r.gamma.into_pyobject(py).unwrap().into_any().unbind(),
            );
            d.insert(
                "method".to_string(),
                r.method
                    .clone()
                    .into_pyobject(py)
                    .unwrap()
                    .into_any()
                    .unbind(),
            );
            d.insert(
                "n_small_before".to_string(),
                r.n_small_before
                    .into_pyobject(py)
                    .unwrap()
                    .into_any()
                    .unbind(),
            );
            d.insert(
                "n_small_after".to_string(),
                r.n_small_after
                    .into_pyobject(py)
                    .unwrap()
                    .into_any()
                    .unbind(),
            );
            d.insert(
                "n_merged".to_string(),
                r.n_merged.into_pyobject(py).unwrap().into_any().unbind(),
            );
            d.insert(
                "n_total_clusters".to_string(),
                r.n_total_clusters
                    .into_pyobject(py)
                    .unwrap()
                    .into_any()
                    .unbind(),
            );
            d.insert(
                "max_cluster_size".to_string(),
                r.max_cluster_size
                    .into_pyobject(py)
                    .unwrap()
                    .into_any()
                    .unbind(),
            );
            d.insert(
                "max_cluster_weight".to_string(),
                r.max_cluster_weight
                    .into_pyobject(py)
                    .unwrap()
                    .into_any()
                    .unbind(),
            );
            d
        })
        .collect();

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
    let clustering =
        Clustering::from_assignments(membership.as_slice()?.iter().map(|&x| x as u32).collect());
    let cpm = CPM::new(resolution);
    Ok(cpm.quality(&graph, &clustering))
}

// ── Graph utility bindings ──────────────────────────────────────

/// Per-node top-k edge filter (returns kept edge indices).
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (src, dst, weight, k, mutual = false))]
fn rust_filter_top_k<'py>(
    py: Python<'py>,
    src: PyReadonlyArray1<u32>,
    dst: PyReadonlyArray1<u32>,
    weight: PyReadonlyArray1<f64>,
    k: usize,
    mutual: bool,
) -> PyResult<Py<PyArray1<u32>>> {
    let kept = crate::graph_utils::filter_top_k(
        src.as_slice()?,
        dst.as_slice()?,
        weight.as_slice()?,
        k,
        mutual,
    );
    let indices: Vec<u32> = kept.into_iter().map(|i| i as u32).collect();
    Ok(PyArray1::from_vec(py, indices).into())
}

/// Find GCC membership mask (returns bool array).
#[cfg(feature = "python")]
#[pyfunction]
fn rust_find_gcc<'py>(
    py: Python<'py>,
    src: PyReadonlyArray1<u32>,
    dst: PyReadonlyArray1<u32>,
    n_nodes: usize,
) -> PyResult<Py<PyArray1<bool>>> {
    let mask = crate::graph_utils::find_gcc(src.as_slice()?, dst.as_slice()?, n_nodes);
    Ok(PyArray1::from_vec(py, mask).into())
}

/// Contract graph by cluster membership.
///
/// Returns (src, dst, weight, n_clusters, node_sizes).
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (src, dst, weight, membership, prev_node_sizes = None))]
fn rust_contract_edges<'py>(
    py: Python<'py>,
    src: PyReadonlyArray1<u32>,
    dst: PyReadonlyArray1<u32>,
    weight: PyReadonlyArray1<f64>,
    membership: PyReadonlyArray1<u64>,
    prev_node_sizes: Option<PyReadonlyArray1<i64>>,
) -> PyResult<(
    Py<PyArray1<u32>>,
    Py<PyArray1<u32>>,
    Py<PyArray1<f64>>,
    usize,
    Py<PyArray1<i64>>,
)> {
    let prev = prev_node_sizes.as_ref().map(|p| p.as_slice()).transpose()?;
    let (out_src, out_dst, out_w, n_cl, sizes) = crate::graph_utils::contract_edges(
        src.as_slice()?,
        dst.as_slice()?,
        weight.as_slice()?,
        membership.as_slice()?,
        prev,
    );
    Ok((
        PyArray1::from_vec(py, out_src).into(),
        PyArray1::from_vec(py, out_dst).into(),
        PyArray1::from_vec(py, out_w).into(),
        n_cl,
        PyArray1::from_vec(py, sizes).into(),
    ))
}

/// Python module definition.
#[cfg(feature = "python")]
#[pymodule]
fn sciscape_leiden(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGraphHandle>()?;
    m.add_function(wrap_pyfunction!(load_graph, m)?)?;
    m.add_function(wrap_pyfunction!(load_graph_raw_files, m)?)?;
    m.add_function(wrap_pyfunction!(contract_graph_handle, m)?)?;
    m.add_function(wrap_pyfunction!(summarize_membership_handle, m)?)?;
    m.add_function(wrap_pyfunction!(run_leiden, m)?)?;
    m.add_function(wrap_pyfunction!(run_leiden_handle, m)?)?;
    m.add_function(wrap_pyfunction!(run_postprocess, m)?)?;
    m.add_function(wrap_pyfunction!(run_postprocess_handle, m)?)?;
    m.add_function(wrap_pyfunction!(cpm_quality, m)?)?;
    m.add_function(wrap_pyfunction!(rust_filter_top_k, m)?)?;
    m.add_function(wrap_pyfunction!(rust_find_gcc, m)?)?;
    m.add_function(wrap_pyfunction!(rust_contract_edges, m)?)?;
    Ok(())
}
