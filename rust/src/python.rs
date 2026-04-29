//! PyO3 bindings for sciscape-leiden.
//!
//! Exposes Leiden clustering to Python via numpy arrays.

#[cfg(feature = "python")]
use numpy::{PyArray1, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::conversion::IntoPyObject;
#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
use crate::adaptive::cluster_graph_stats as compute_cluster_graph_stats;
#[cfg(feature = "python")]
use crate::contraction::create_reduced_network;
#[cfg(feature = "python")]
use crate::quality::{QualityFunction, CPM};
#[cfg(feature = "python")]
use crate::trace;
#[cfg(feature = "python")]
use crate::workspace::Workspace;
#[cfg(feature = "python")]
use crate::{
    leiden, leiden_multi_start, postprocess_small_clusters, Clustering, Graph, LeidenConfig,
    PostprocessRound,
};
#[cfg(feature = "python")]
use rand::SeedableRng;
#[cfg(feature = "python")]
use std::fs::File;
#[cfg(feature = "python")]
use std::io::Read;
#[cfg(feature = "python")]
use std::mem;
#[cfg(feature = "python")]
use std::time::Instant;

#[cfg(feature = "python")]
fn trace_python(message: &str) {
    trace::emit(format_args!("{}", message));
}

#[cfg(feature = "python")]
const DEFAULT_RAW_EDGE_CHUNK_SIZE: usize = 4_194_304;

#[cfg(feature = "python")]
fn raw_edge_chunk_size() -> usize {
    std::env::var("SCISCAPE_RAW_EDGE_CHUNK_SIZE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(DEFAULT_RAW_EDGE_CHUNK_SIZE)
}

#[cfg(feature = "python")]
fn file_elem_len(path: &str, elem_size: usize) -> PyResult<usize> {
    let byte_len = File::open(path)
        .and_then(|f| f.metadata())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("stat {}: {}", path, e)))?
        .len() as usize;
    if elem_size == 0 || byte_len % elem_size != 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "binary file length is not divisible by element size: {}",
            path
        )));
    }
    Ok(byte_len / elem_size)
}

#[cfg(feature = "python")]
fn read_pod_file<T: Copy>(path: &str) -> PyResult<Vec<T>> {
    let mut file = File::open(path).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("open {}: {}", path, e))
    })?;
    let byte_len = file
        .metadata()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("stat {}: {}", path, e)))?
        .len() as usize;
    let elem_size = mem::size_of::<T>();
    if elem_size == 0 || byte_len % elem_size != 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "binary file length is not divisible by element size: {}",
            path
        )));
    }

    let n = byte_len / elem_size;
    let mut out: Vec<T> = Vec::with_capacity(n);
    // u32 and f64 are plain data types with no invalid bit patterns for our
    // use here; reading directly into the vector avoids a second Vec<u8>.
    unsafe {
        let bytes = std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, byte_len);
        file.read_exact(bytes).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("read {}: {}", path, e))
        })?;
        out.set_len(n);
    }
    Ok(out)
}

#[cfg(feature = "python")]
fn read_file_chunk(file: &mut File, path: &str, buf: &mut [u8]) -> PyResult<()> {
    file.read_exact(buf)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("read {}: {}", path, e)))
}

#[cfg(feature = "python")]
#[inline]
fn read_u32_at(buf: &[u8], idx: usize) -> u32 {
    let start = idx * mem::size_of::<u32>();
    u32::from_ne_bytes(
        buf[start..start + mem::size_of::<u32>()]
            .try_into()
            .unwrap(),
    )
}

#[cfg(feature = "python")]
#[inline]
fn read_f64_at(buf: &[u8], idx: usize) -> f64 {
    let start = idx * mem::size_of::<f64>();
    f64::from_ne_bytes(
        buf[start..start + mem::size_of::<f64>()]
            .try_into()
            .unwrap(),
    )
}

#[cfg(feature = "python")]
fn open_raw_file(path: &str) -> PyResult<File> {
    File::open(path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("open {}: {}", path, e)))
}

#[cfg(feature = "python")]
fn build_graph_from_raw_files_streaming(
    n_nodes: usize,
    src_path: &str,
    dst_path: &str,
    weights_path: &str,
    node_weights_path: Option<&str>,
) -> PyResult<Graph> {
    let n_src = file_elem_len(src_path, mem::size_of::<u32>())?;
    let n_dst = file_elem_len(dst_path, mem::size_of::<u32>())?;
    let n_weights = file_elem_len(weights_path, mem::size_of::<f64>())?;
    if n_src != n_dst || n_src != n_weights {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "raw edge sidecars have different lengths: src={}, dst={}, weights={}",
            n_src, n_dst, n_weights,
        )));
    }

    let n_input_edges = n_src;
    let chunk_size = raw_edge_chunk_size();
    let mut degree = vec![0u64; n_nodes];
    let mut src_file = open_raw_file(src_path)?;
    let mut dst_file = open_raw_file(dst_path)?;
    let mut src_buf = vec![0u8; chunk_size * mem::size_of::<u32>()];
    let mut dst_buf = vec![0u8; chunk_size * mem::size_of::<u32>()];

    let mut remaining = n_input_edges;
    while remaining > 0 {
        let n_chunk = remaining.min(chunk_size);
        let src_bytes = n_chunk * mem::size_of::<u32>();
        let dst_bytes = n_chunk * mem::size_of::<u32>();
        read_file_chunk(&mut src_file, src_path, &mut src_buf[..src_bytes])?;
        read_file_chunk(&mut dst_file, dst_path, &mut dst_buf[..dst_bytes])?;

        for i in 0..n_chunk {
            let s = read_u32_at(&src_buf, i) as usize;
            let d = read_u32_at(&dst_buf, i) as usize;
            if s >= n_nodes || d >= n_nodes {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "raw edge endpoint out of bounds: src={}, dst={}, n_nodes={}",
                    s, d, n_nodes,
                )));
            }
            degree[s] += 1;
            degree[d] += 1;
        }
        remaining -= n_chunk;
    }

    let mut first_neighbor_index = vec![0u64; n_nodes + 1];
    let mut running = 0u64;
    for node in 0..n_nodes {
        first_neighbor_index[node] = running;
        let d = degree[node];
        degree[node] = running;
        running += d;
    }
    first_neighbor_index[n_nodes] = running;

    let n_edges = running as usize;
    let mut neighbors = vec![0u32; n_edges];
    let mut edge_weights = vec![0.0f64; n_edges];
    let mut offset = degree;

    let mut src_file = open_raw_file(src_path)?;
    let mut dst_file = open_raw_file(dst_path)?;
    let mut weights_file = open_raw_file(weights_path)?;
    let mut weight_buf = vec![0u8; chunk_size * mem::size_of::<f64>()];

    let mut remaining = n_input_edges;
    while remaining > 0 {
        let n_chunk = remaining.min(chunk_size);
        let src_bytes = n_chunk * mem::size_of::<u32>();
        let dst_bytes = n_chunk * mem::size_of::<u32>();
        let weight_bytes = n_chunk * mem::size_of::<f64>();
        read_file_chunk(&mut src_file, src_path, &mut src_buf[..src_bytes])?;
        read_file_chunk(&mut dst_file, dst_path, &mut dst_buf[..dst_bytes])?;
        read_file_chunk(
            &mut weights_file,
            weights_path,
            &mut weight_buf[..weight_bytes],
        )?;

        for i in 0..n_chunk {
            let s_u32 = read_u32_at(&src_buf, i);
            let d_u32 = read_u32_at(&dst_buf, i);
            let s = s_u32 as usize;
            let d = d_u32 as usize;
            let w = read_f64_at(&weight_buf, i);

            let pos_s = offset[s] as usize;
            neighbors[pos_s] = d_u32;
            edge_weights[pos_s] = w;
            offset[s] += 1;

            let pos_d = offset[d] as usize;
            neighbors[pos_d] = s_u32;
            edge_weights[pos_d] = w;
            offset[d] += 1;
        }
        remaining -= n_chunk;
    }

    let node_weights = if let Some(path) = node_weights_path {
        let weights: Vec<f64> = read_pod_file(path)?;
        if weights.len() != n_nodes {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "node_weights length {} does not match n_nodes {}",
                weights.len(),
                n_nodes,
            )));
        }
        weights
    } else {
        vec![1.0; n_nodes]
    };
    let self_loop_weights = vec![0.0; n_nodes];

    Ok(Graph {
        n_nodes,
        n_edges,
        first_neighbor_index,
        neighbors,
        edge_weights,
        node_weights,
        self_loop_weights,
    })
}

#[cfg(feature = "python")]
type PostprocessPyResult = (
    Py<PyArray1<u64>>,
    usize,
    Py<PyArray1<i32>>,
    Vec<std::collections::HashMap<String, pyo3::PyObject>>,
);

#[cfg(feature = "python")]
#[derive(Clone, Copy)]
struct ResolutionSearchStats {
    resolution: f64,
    cluster_count: usize,
    quality: f64,
}

#[cfg(feature = "python")]
#[derive(Clone)]
struct ResolutionSearchEval {
    stats: ResolutionSearchStats,
    membership: Vec<u32>,
}

#[cfg(feature = "python")]
fn cluster_distance(cluster_count: usize, min_clusters: usize, max_clusters: usize) -> usize {
    if cluster_count < min_clusters {
        min_clusters - cluster_count
    } else if cluster_count > max_clusters {
        cluster_count - max_clusters
    } else {
        0
    }
}

#[cfg(feature = "python")]
fn update_search_best(
    best: &mut Option<ResolutionSearchEval>,
    candidate: &ResolutionSearchEval,
    min_clusters: usize,
    max_clusters: usize,
) {
    let candidate_distance =
        cluster_distance(candidate.stats.cluster_count, min_clusters, max_clusters);
    let should_update = match best {
        None => true,
        Some(current) => {
            let current_distance =
                cluster_distance(current.stats.cluster_count, min_clusters, max_clusters);
            candidate_distance < current_distance
                || (candidate_distance == current_distance
                    && candidate.stats.resolution < current.stats.resolution)
        }
    };
    if should_update {
        *best = Some(candidate.clone());
    }
}

#[cfg(feature = "python")]
fn evaluate_resolution_probe(
    graph: &Graph,
    gamma: f64,
    n_iterations: usize,
    randomness: f64,
    seed: u64,
    initial_membership: Option<&[u32]>,
) -> ResolutionSearchEval {
    let config = LeidenConfig {
        resolution: gamma,
        n_iterations,
        randomness,
        randomness_schedule: Vec::new(),
        seed,
    };
    let initial =
        initial_membership.map(|membership| Clustering::from_assignments(membership.to_vec()));
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let result = leiden(graph, &config, initial, &mut rng);
    ResolutionSearchEval {
        stats: ResolutionSearchStats {
            resolution: gamma,
            cluster_count: result.clustering.n_clusters,
            quality: result.quality,
        },
        membership: result.clustering.clusters,
    }
}

#[cfg(feature = "python")]
fn search_resolution_on_graph(
    graph: &Graph,
    min_clusters: usize,
    max_clusters: usize,
    lower_bound: f64,
    upper_bound: f64,
    max_iterations: usize,
    n_iterations: usize,
    randomness: f64,
    seed: u64,
) -> Result<(ResolutionSearchEval, usize), String> {
    if min_clusters == 0 || max_clusters == 0 {
        return Err("cluster bounds must be positive".to_string());
    }
    if min_clusters > max_clusters {
        return Err("min_clusters must be less than or equal to max_clusters".to_string());
    }
    if lower_bound <= 0.0 || upper_bound <= 0.0 {
        return Err("resolution bounds must be positive".to_string());
    }
    if lower_bound >= upper_bound {
        return Err("resolution lower bound must be less than upper bound".to_string());
    }

    let mut best: Option<ResolutionSearchEval> = None;
    let mut eval_count = 0usize;
    let mut warm_membership: Option<Vec<u32>> = None;

    let mut lower_eval = evaluate_resolution_probe(
        graph,
        lower_bound,
        n_iterations,
        randomness,
        seed,
        warm_membership.as_deref(),
    );
    eval_count += 1;
    update_search_best(&mut best, &lower_eval, min_clusters, max_clusters);
    warm_membership = Some(lower_eval.membership);

    let mut upper_eval = evaluate_resolution_probe(
        graph,
        upper_bound,
        n_iterations,
        randomness,
        seed,
        warm_membership.as_deref(),
    );
    eval_count += 1;
    update_search_best(&mut best, &upper_eval, min_clusters, max_clusters);
    warm_membership = Some(upper_eval.membership);

    let mut expand_lo = lower_bound;
    let mut count_lo = lower_eval.stats.cluster_count;
    for _ in 0..max_iterations {
        if count_lo <= max_clusters || expand_lo < 1e-9 {
            break;
        }
        expand_lo *= 0.5;
        lower_eval = evaluate_resolution_probe(
            graph,
            expand_lo,
            n_iterations,
            randomness,
            seed,
            warm_membership.as_deref(),
        );
        eval_count += 1;
        update_search_best(&mut best, &lower_eval, min_clusters, max_clusters);
        count_lo = lower_eval.stats.cluster_count;
        warm_membership = Some(lower_eval.membership);
    }

    let mut expand_hi = upper_bound;
    let mut count_hi = upper_eval.stats.cluster_count;
    for _ in 0..max_iterations {
        if count_hi >= min_clusters || expand_hi > 1e9 {
            break;
        }
        expand_hi *= 2.0;
        upper_eval = evaluate_resolution_probe(
            graph,
            expand_hi,
            n_iterations,
            randomness,
            seed,
            warm_membership.as_deref(),
        );
        eval_count += 1;
        update_search_best(&mut best, &upper_eval, min_clusters, max_clusters);
        count_hi = upper_eval.stats.cluster_count;
        warm_membership = Some(upper_eval.membership);
    }

    let lower_count = lower_eval.stats.cluster_count;
    let upper_count = upper_eval.stats.cluster_count;
    if (lower_count > max_clusters && upper_count > max_clusters)
        || (lower_count < min_clusters && upper_count < min_clusters)
    {
        return best
            .map(|eval| (eval, eval_count))
            .ok_or_else(|| "failed to evaluate any Leiden partitions".to_string());
    }

    let mut lo_gamma = lower_eval.stats.resolution;
    let mut hi_gamma = upper_eval.stats.resolution;
    for _ in 0..max_iterations {
        let mid_gamma = (lo_gamma + hi_gamma) / 2.0;
        let mid_eval = evaluate_resolution_probe(
            graph,
            mid_gamma,
            n_iterations,
            randomness,
            seed,
            warm_membership.as_deref(),
        );
        eval_count += 1;
        update_search_best(&mut best, &mid_eval, min_clusters, max_clusters);
        let mid_count = mid_eval.stats.cluster_count;

        if min_clusters <= mid_count && mid_count <= max_clusters {
            return Ok((mid_eval, eval_count));
        }

        warm_membership = Some(mid_eval.membership);

        if mid_count < min_clusters {
            lo_gamma = mid_gamma;
        } else {
            hi_gamma = mid_gamma;
        }
    }

    best.map(|eval| (eval, eval_count))
        .ok_or_else(|| "failed to converge on a resolution; no candidates evaluated".to_string())
}

#[cfg(feature = "python")]
fn rounds_to_py(
    py: Python<'_>,
    rounds: &[PostprocessRound],
) -> Vec<std::collections::HashMap<String, pyo3::PyObject>> {
    rounds
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
        .collect()
}

/// Reusable CSR graph for repeated Leiden/postprocess calls.
///
/// Building CSR dominates repeated gamma probes and postprocessing on large
/// graphs. This object lets Python construct the graph once and reuse it.
#[cfg(feature = "python")]
#[pyclass(name = "Graph")]
struct PyGraph {
    graph: Graph,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyGraph {
    #[new]
    #[pyo3(signature = (n_nodes, src, dst, weights, node_weights = None))]
    fn new(
        n_nodes: usize,
        src: PyReadonlyArray1<u32>,
        dst: PyReadonlyArray1<u32>,
        weights: PyReadonlyArray1<f64>,
        node_weights: Option<PyReadonlyArray1<f64>>,
    ) -> PyResult<Self> {
        let t0 = Instant::now();
        let n_input_edges = src.len()?;
        let graph = if let Some(nw) = node_weights {
            Graph::from_edge_list_weighted_trusted(
                n_nodes,
                src.as_slice()?,
                dst.as_slice()?,
                weights.as_slice()?,
                nw.as_slice()?,
            )
        } else {
            Graph::from_edge_list_trusted(
                n_nodes,
                src.as_slice()?,
                dst.as_slice()?,
                weights.as_slice()?,
            )
        };
        trace_python(&format!(
            "phase=graph_build nodes={} undirected_edges={} directed_edges={} elapsed_ms={:.1}{}",
            graph.n_nodes,
            n_input_edges,
            graph.n_edges,
            t0.elapsed().as_secs_f64() * 1000.0,
            trace::memory_fields(),
        ));
        Ok(Self { graph })
    }

    #[getter]
    fn n_nodes(&self) -> usize {
        self.graph.n_nodes
    }

    #[getter]
    fn n_edges(&self) -> usize {
        self.graph.n_edges
    }

    #[pyo3(signature = (
        resolution = 1.0,
        n_iterations = 10,
        n_starts = 1,
        randomness = 0.01,
        randomness_schedule = None,
        seed = 0,
        initial_membership = None,
        fixed_nodes = None,
    ))]
    fn run_leiden<'py>(
        &self,
        py: Python<'py>,
        resolution: f64,
        n_iterations: usize,
        n_starts: usize,
        randomness: f64,
        randomness_schedule: Option<Vec<f64>>,
        seed: u64,
        initial_membership: Option<PyReadonlyArray1<u64>>,
        fixed_nodes: Option<PyReadonlyArray1<bool>>,
    ) -> PyResult<(Py<PyArray1<u64>>, f64, usize)> {
        let config = LeidenConfig {
            resolution,
            n_iterations,
            randomness,
            randomness_schedule: randomness_schedule.unwrap_or_default(),
            seed,
        };

        let initial = if let Some(mem) = initial_membership {
            let mem_slice = mem.as_slice()?;
            let mut clustering = Clustering::from_u64_assignments(mem_slice)
                .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
            if let Some(fixed) = fixed_nodes {
                clustering.set_fixed(fixed.as_slice()?.to_vec());
            }
            Some(clustering)
        } else {
            None
        };

        let graph = &self.graph;
        let result = py.allow_threads(|| {
            if n_starts > 1 {
                leiden_multi_start(graph, &config, n_starts, initial.as_ref())
            } else {
                let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
                leiden(graph, &config, initial, &mut rng)
            }
        });

        let membership: Vec<u64> = result
            .clustering
            .clusters
            .iter()
            .map(|&c| c as u64)
            .collect();
        let n_clusters = result.clustering.n_clusters;
        let quality = result.quality;

        Ok((
            PyArray1::from_vec(py, membership).into(),
            quality,
            n_clusters,
        ))
    }

    #[pyo3(signature = (membership, resolution))]
    fn cpm_quality(&self, membership: PyReadonlyArray1<u64>, resolution: f64) -> PyResult<f64> {
        let clustering = Clustering::from_u64_assignments(membership.as_slice()?)
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        if clustering.clusters.len() != self.graph.n_nodes {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "membership length {} does not match graph node count {}",
                clustering.clusters.len(),
                self.graph.n_nodes,
            )));
        }
        let cpm = CPM::new(resolution);
        Ok(cpm.quality(&self.graph, &clustering))
    }

    #[pyo3(signature = (
        membership,
        resolution,
        min_weight = 0.0,
        max_weight = 0.0,
        top_k = 1000,
    ))]
    fn cluster_graph_stats(
        &self,
        py: Python<'_>,
        membership: PyReadonlyArray1<u64>,
        resolution: f64,
        min_weight: f64,
        max_weight: f64,
        top_k: usize,
    ) -> PyResult<std::collections::HashMap<String, pyo3::PyObject>> {
        let clustering = Clustering::from_u64_assignments(membership.as_slice()?)
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        if clustering.clusters.len() != self.graph.n_nodes {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "membership length {} does not match graph node count {}",
                clustering.clusters.len(),
                self.graph.n_nodes,
            )));
        }

        let stats = py.allow_threads(|| {
            let mut ws = Workspace::new(self.graph.n_nodes.max(clustering.n_clusters));
            compute_cluster_graph_stats(
                &self.graph,
                &clustering,
                resolution,
                min_weight,
                max_weight,
                top_k,
                &mut ws,
            )
        });

        let n_clusters = stats.block_count.len();
        let candidates = stats.merge_candidates;
        let n_candidates = candidates.len();
        let mut candidate_source = Vec::with_capacity(n_candidates);
        let mut candidate_target = Vec::with_capacity(n_candidates);
        let mut candidate_edge_weight = Vec::with_capacity(n_candidates);
        let mut candidate_delta_q = Vec::with_capacity(n_candidates);
        let mut candidate_merged_weight = Vec::with_capacity(n_candidates);
        let mut candidate_size_band_gain = Vec::with_capacity(n_candidates);
        for candidate in candidates {
            candidate_source.push(candidate.source);
            candidate_target.push(candidate.target);
            candidate_edge_weight.push(candidate.edge_weight);
            candidate_delta_q.push(candidate.delta_q);
            candidate_merged_weight.push(candidate.merged_weight);
            candidate_size_band_gain.push(candidate.size_band_gain);
        }

        let mut out = std::collections::HashMap::new();
        out.insert(
            "n_clusters".to_string(),
            n_clusters.into_pyobject(py).unwrap().into_any().unbind(),
        );
        out.insert(
            "block_count".to_string(),
            PyArray1::from_vec(py, stats.block_count)
                .into_any()
                .unbind(),
        );
        out.insert(
            "doc_weight".to_string(),
            PyArray1::from_vec(py, stats.doc_weight).into_any().unbind(),
        );
        out.insert(
            "internal_weight".to_string(),
            PyArray1::from_vec(py, stats.internal_weight)
                .into_any()
                .unbind(),
        );
        out.insert(
            "external_weight".to_string(),
            PyArray1::from_vec(py, stats.external_weight)
                .into_any()
                .unbind(),
        );
        out.insert(
            "degree".to_string(),
            PyArray1::from_vec(py, stats.degree).into_any().unbind(),
        );
        out.insert(
            "top_neighbor".to_string(),
            PyArray1::from_vec(py, stats.top_neighbor)
                .into_any()
                .unbind(),
        );
        out.insert(
            "top_neighbor_weight".to_string(),
            PyArray1::from_vec(py, stats.top_neighbor_weight)
                .into_any()
                .unbind(),
        );
        out.insert(
            "conductance".to_string(),
            PyArray1::from_vec(py, stats.conductance)
                .into_any()
                .unbind(),
        );
        out.insert(
            "leafness".to_string(),
            PyArray1::from_vec(py, stats.leafness).into_any().unbind(),
        );
        out.insert(
            "band_distance".to_string(),
            PyArray1::from_vec(py, stats.band_distance)
                .into_any()
                .unbind(),
        );
        out.insert(
            "candidate_source".to_string(),
            PyArray1::from_vec(py, candidate_source).into_any().unbind(),
        );
        out.insert(
            "candidate_target".to_string(),
            PyArray1::from_vec(py, candidate_target).into_any().unbind(),
        );
        out.insert(
            "candidate_edge_weight".to_string(),
            PyArray1::from_vec(py, candidate_edge_weight)
                .into_any()
                .unbind(),
        );
        out.insert(
            "candidate_delta_q".to_string(),
            PyArray1::from_vec(py, candidate_delta_q)
                .into_any()
                .unbind(),
        );
        out.insert(
            "candidate_merged_weight".to_string(),
            PyArray1::from_vec(py, candidate_merged_weight)
                .into_any()
                .unbind(),
        );
        out.insert(
            "candidate_size_band_gain".to_string(),
            PyArray1::from_vec(py, candidate_size_band_gain)
                .into_any()
                .unbind(),
        );
        Ok(out)
    }

    #[pyo3(signature = (
        min_clusters,
        max_clusters,
        lower_bound,
        upper_bound,
        max_iterations = 32,
        n_iterations = 10,
        randomness = 0.01,
        seed = 0,
    ))]
    fn search_resolution(
        &self,
        py: Python<'_>,
        min_clusters: usize,
        max_clusters: usize,
        lower_bound: f64,
        upper_bound: f64,
        max_iterations: usize,
        n_iterations: usize,
        randomness: f64,
        seed: u64,
    ) -> PyResult<(f64, usize, f64, usize, Py<PyArray1<u64>>)> {
        let (search_eval, eval_count) = py
            .allow_threads(|| {
                search_resolution_on_graph(
                    &self.graph,
                    min_clusters,
                    max_clusters,
                    lower_bound,
                    upper_bound,
                    max_iterations,
                    n_iterations,
                    randomness,
                    seed,
                )
            })
            .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;
        let stats = search_eval.stats;
        let membership: Vec<u64> = search_eval
            .membership
            .into_iter()
            .map(|cluster| cluster as u64)
            .collect();
        Ok((
            stats.resolution,
            stats.cluster_count,
            stats.quality,
            eval_count,
            PyArray1::from_vec(py, membership).into(),
        ))
    }

    #[pyo3(signature = (
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
    ))]
    fn run_postprocess<'py>(
        &self,
        py: Python<'py>,
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
    ) -> PyResult<PostprocessPyResult> {
        let clustering = Clustering::from_u64_assignments(membership.as_slice()?)
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

        let config = LeidenConfig {
            resolution,
            n_iterations,
            randomness,
            randomness_schedule: Vec::new(),
            seed,
        };

        let graph = &self.graph;
        let pp_result = py.allow_threads(|| {
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            postprocess_small_clusters(
                graph,
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
            )
        });

        let mem_out: Vec<u64> = pp_result
            .clustering
            .clusters
            .iter()
            .map(|&c| c as u64)
            .collect();
        let n_clusters = pp_result.clustering.n_clusters;
        let changed_at: Py<PyArray1<i32>> =
            PyArray1::from_vec(py, pp_result.changed_at_round).into();
        let rounds_info = rounds_to_py(py, &pp_result.rounds);

        Ok((
            PyArray1::from_vec(py, mem_out).into(),
            n_clusters,
            changed_at,
            rounds_info,
        ))
    }

    #[pyo3(signature = (membership, keep_self_loops = true))]
    fn contract<'py>(
        &self,
        py: Python<'py>,
        membership: PyReadonlyArray1<u64>,
        keep_self_loops: bool,
    ) -> PyResult<(Py<PyGraph>, Py<PyArray1<f64>>)> {
        let clustering = Clustering::from_u64_assignments(membership.as_slice()?)
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        if clustering.clusters.len() != self.graph.n_nodes {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "membership length {} does not match graph node count {}",
                clustering.clusters.len(),
                self.graph.n_nodes,
            )));
        }

        let graph = &self.graph;
        let reduced = py.allow_threads(|| {
            let mut ws = Workspace::new(graph.n_nodes.max(clustering.n_clusters));
            create_reduced_network(graph, &clustering, keep_self_loops, &mut ws)
        });
        let node_weights = reduced.node_weights.clone();
        let py_graph = Py::new(py, PyGraph { graph: reduced })?;
        Ok((py_graph, PyArray1::from_vec(py, node_weights).into()))
    }
}

/// Load a reusable graph directly from raw binary sidecar files.
///
/// Files must be native-endian arrays produced by numpy `tofile`:
/// - src/dst: u32
/// - weights: f64
/// - node_weights: optional f64
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (n_nodes, src_path, dst_path, weights_path, node_weights_path = None))]
fn load_graph_raw_files(
    n_nodes: usize,
    src_path: &str,
    dst_path: &str,
    weights_path: &str,
    node_weights_path: Option<&str>,
) -> PyResult<PyGraph> {
    let t0 = Instant::now();
    let graph = build_graph_from_raw_files_streaming(
        n_nodes,
        src_path,
        dst_path,
        weights_path,
        node_weights_path,
    )?;

    trace_python(&format!(
        "phase=graph_build_raw_files_streaming nodes={} undirected_edges={} directed_edges={} chunk_size={} elapsed_ms={:.1}{}",
        graph.n_nodes,
        graph.n_edges / 2,
        graph.n_edges,
        raw_edge_chunk_size(),
        t0.elapsed().as_secs_f64() * 1000.0,
        trace::memory_fields(),
    ));
    Ok(PyGraph { graph })
}

/// Remap a parquet edge table with string UIDs to integer edge files.
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (
    edge_path,
    output_dir,
    uid1_col = "uid1",
    uid2_col = "uid2",
    weight_col = "rel_sum2",
))]
fn rust_integer_remap_parquet(
    py: Python<'_>,
    edge_path: &str,
    output_dir: &str,
    uid1_col: &str,
    uid2_col: &str,
    weight_col: &str,
) -> PyResult<(usize, usize, String, String)> {
    let result = py
        .allow_threads(|| {
            crate::remap::integer_remap_parquet(
                std::path::Path::new(edge_path),
                std::path::Path::new(output_dir),
                uid1_col,
                uid2_col,
                weight_col,
            )
        })
        .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;
    Ok((
        result.n_nodes,
        result.n_edges,
        result.node_manifest_path.to_string_lossy().to_string(),
        result.int_edges_path.to_string_lossy().to_string(),
    ))
}

/// Remap a parquet edge table and write only manifest + raw sidecars.
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (
    edge_path,
    output_dir,
    uid1_col = "uid1",
    uid2_col = "uid2",
    weight_col = "rel_sum2",
))]
fn rust_integer_remap_parquet_sidecars(
    py: Python<'_>,
    edge_path: &str,
    output_dir: &str,
    uid1_col: &str,
    uid2_col: &str,
    weight_col: &str,
) -> PyResult<(usize, usize, String, String, String, String, String)> {
    let result = py
        .allow_threads(|| {
            crate::remap::integer_remap_parquet_with_options(
                std::path::Path::new(edge_path),
                std::path::Path::new(output_dir),
                uid1_col,
                uid2_col,
                weight_col,
                false,
            )
        })
        .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;
    Ok((
        result.n_nodes,
        result.n_edges,
        result.node_manifest_path.to_string_lossy().to_string(),
        result.int_edges_path.to_string_lossy().to_string(),
        result.src_path.to_string_lossy().to_string(),
        result.dst_path.to_string_lossy().to_string(),
        result.weight_path.to_string_lossy().to_string(),
    ))
}

/// Remap a parquet edge table and return a reusable Rust graph directly.
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (
    edge_path,
    output_dir,
    uid1_col = "uid1",
    uid2_col = "uid2",
    weight_col = "rel_sum2",
))]
fn rust_integer_remap_parquet_graph(
    py: Python<'_>,
    edge_path: &str,
    output_dir: &str,
    uid1_col: &str,
    uid2_col: &str,
    weight_col: &str,
) -> PyResult<(usize, usize, String, String, Py<PyGraph>)> {
    let t0 = Instant::now();
    let result = py
        .allow_threads(|| {
            crate::remap::integer_remap_parquet_to_graph(
                std::path::Path::new(edge_path),
                std::path::Path::new(output_dir),
                uid1_col,
                uid2_col,
                weight_col,
            )
        })
        .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;
    trace_python(&format!(
        "phase=remap_graph_build nodes={} undirected_edges={} directed_edges={} elapsed_ms={:.1}{}",
        result.graph.n_nodes,
        result.n_edges,
        result.graph.n_edges,
        t0.elapsed().as_secs_f64() * 1000.0,
        trace::memory_fields(),
    ));
    let graph = Py::new(
        py,
        PyGraph {
            graph: result.graph,
        },
    )?;
    Ok((
        result.n_nodes,
        result.n_edges,
        result.node_manifest_path.to_string_lossy().to_string(),
        result.int_edges_path.to_string_lossy().to_string(),
        graph,
    ))
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
    randomness_schedule = None,
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
    randomness_schedule: Option<Vec<f64>>,
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
        randomness_schedule: randomness_schedule.unwrap_or_default(),
        seed,
    };

    let initial = if let Some(mem) = initial_membership {
        let mem_slice = mem.as_slice()?;
        let mut clustering = Clustering::from_u64_assignments(mem_slice)
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
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
) -> PyResult<PostprocessPyResult> {
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

    let clustering = Clustering::from_u64_assignments(membership.as_slice()?)
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

    let config = LeidenConfig {
        resolution,
        n_iterations,
        randomness,
        randomness_schedule: Vec::new(),
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

    let mem_out: Vec<u64> = pp_result
        .clustering
        .clusters
        .iter()
        .map(|&c| c as u64)
        .collect();
    let n_clusters = pp_result.clustering.n_clusters;
    let changed_at: Py<PyArray1<i32>> = PyArray1::from_vec(py, pp_result.changed_at_round).into();

    let rounds_info = rounds_to_py(py, &pp_result.rounds);

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
    let clustering = Clustering::from_u64_assignments(membership.as_slice()?)
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
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

#[cfg(feature = "python")]
enum ProjectedMembership {
    U32(Vec<u32>),
    U64(Vec<u64>),
}

#[cfg(feature = "python")]
#[inline]
fn membership_fits_u32(membership: &[u64]) -> bool {
    membership.iter().all(|&cluster| cluster <= u32::MAX as u64)
}

#[cfg(feature = "python")]
fn project_membership_from_u32(
    membership: &[u64],
    previous: &[u32],
) -> Result<ProjectedMembership, String> {
    if membership_fits_u32(membership) {
        let mut out = Vec::with_capacity(previous.len());
        for &idx in previous {
            let cluster = *membership.get(idx as usize).ok_or_else(|| {
                format!(
                    "previous membership index {} out of bounds for {} clusters",
                    idx,
                    membership.len(),
                )
            })?;
            out.push(cluster as u32);
        }
        Ok(ProjectedMembership::U32(out))
    } else {
        let mut out = Vec::with_capacity(previous.len());
        for &idx in previous {
            let cluster = *membership.get(idx as usize).ok_or_else(|| {
                format!(
                    "previous membership index {} out of bounds for {} clusters",
                    idx,
                    membership.len(),
                )
            })?;
            out.push(cluster);
        }
        Ok(ProjectedMembership::U64(out))
    }
}

#[cfg(feature = "python")]
fn project_membership_from_u64(
    membership: &[u64],
    previous: &[u64],
) -> Result<ProjectedMembership, String> {
    if membership_fits_u32(membership) {
        let mut out = Vec::with_capacity(previous.len());
        for &idx in previous {
            let idx_usize = usize::try_from(idx)
                .map_err(|_| format!("previous membership index {} does not fit in usize", idx))?;
            let cluster = *membership.get(idx_usize).ok_or_else(|| {
                format!(
                    "previous membership index {} out of bounds for {} clusters",
                    idx,
                    membership.len(),
                )
            })?;
            out.push(cluster as u32);
        }
        Ok(ProjectedMembership::U32(out))
    } else {
        let mut out = Vec::with_capacity(previous.len());
        for &idx in previous {
            let idx_usize = usize::try_from(idx)
                .map_err(|_| format!("previous membership index {} does not fit in usize", idx))?;
            let cluster = *membership.get(idx_usize).ok_or_else(|| {
                format!(
                    "previous membership index {} out of bounds for {} clusters",
                    idx,
                    membership.len(),
                )
            })?;
            out.push(cluster);
        }
        Ok(ProjectedMembership::U64(out))
    }
}

#[cfg(feature = "python")]
fn projected_to_py<'py>(py: Python<'py>, projected: ProjectedMembership) -> Py<PyAny> {
    match projected {
        ProjectedMembership::U32(values) => PyArray1::from_vec(py, values).into_any().unbind(),
        ProjectedMembership::U64(values) => PyArray1::from_vec(py, values).into_any().unbind(),
    }
}

/// Project a contracted-graph membership back to original nodes.
///
/// `membership` maps current graph nodes to clusters. `previous` maps original
/// nodes to current graph nodes. The output is compacted to u32 when possible.
#[cfg(feature = "python")]
#[pyfunction]
fn rust_project_membership_u32<'py>(
    py: Python<'py>,
    membership: PyReadonlyArray1<u64>,
    previous: PyReadonlyArray1<u32>,
) -> PyResult<Py<PyAny>> {
    let membership = membership.as_slice()?;
    let previous = previous.as_slice()?;
    let projected = py
        .allow_threads(|| project_membership_from_u32(membership, previous))
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    Ok(projected_to_py(py, projected))
}

/// Project a contracted-graph membership back to original nodes.
///
/// `membership` maps current graph nodes to clusters. `previous` maps original
/// nodes to current graph nodes. The output is compacted to u32 when possible.
#[cfg(feature = "python")]
#[pyfunction]
fn rust_project_membership_u64<'py>(
    py: Python<'py>,
    membership: PyReadonlyArray1<u64>,
    previous: PyReadonlyArray1<u64>,
) -> PyResult<Py<PyAny>> {
    let membership = membership.as_slice()?;
    let previous = previous.as_slice()?;
    let projected = py
        .allow_threads(|| project_membership_from_u64(membership, previous))
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    Ok(projected_to_py(py, projected))
}

/// Compute sorted dense hierarchy indices for u32 cluster columns.
#[cfg(feature = "python")]
#[pyfunction]
fn rust_hierarchy_indices_u32<'py>(
    py: Python<'py>,
    memberships: Vec<PyReadonlyArray1<u32>>,
) -> PyResult<Vec<Py<PyArray1<u32>>>> {
    if memberships.is_empty() {
        return Ok(Vec::new());
    }

    let mut slices = Vec::with_capacity(memberships.len());
    for membership in &memberships {
        slices.push(membership.as_slice()?);
    }

    let n_rows = slices[0].len();
    for slice in &slices[1..] {
        if slice.len() != n_rows {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "all membership arrays must have the same length",
            ));
        }
    }

    let mut outputs: Vec<Py<PyArray1<u32>>> = Vec::with_capacity(slices.len());
    for level_idx in 0..slices.len() {
        let current = slices[level_idx];
        let mut out = Vec::with_capacity(n_rows);

        if level_idx == 0 {
            let mut values = std::collections::BTreeSet::new();
            for &value in current {
                values.insert(value);
            }
            let ranks: std::collections::HashMap<u32, u32> = values
                .into_iter()
                .enumerate()
                .map(|(idx, value)| (value, (idx + 1) as u32))
                .collect();
            for &value in current {
                out.push(*ranks.get(&value).unwrap());
            }
        } else {
            let mut grouped: std::collections::HashMap<Vec<u32>, std::collections::BTreeSet<u32>> =
                std::collections::HashMap::new();
            for row in 0..n_rows {
                let mut key = Vec::with_capacity(level_idx);
                for previous in slices.iter().take(level_idx) {
                    key.push(previous[row]);
                }
                grouped.entry(key).or_default().insert(current[row]);
            }

            let mut rank_maps: std::collections::HashMap<
                Vec<u32>,
                std::collections::HashMap<u32, u32>,
            > = std::collections::HashMap::with_capacity(grouped.len());
            for (key, values) in grouped {
                let ranks = values
                    .into_iter()
                    .enumerate()
                    .map(|(idx, value)| (value, (idx + 1) as u32))
                    .collect();
                rank_maps.insert(key, ranks);
            }

            for row in 0..n_rows {
                let mut key = Vec::with_capacity(level_idx);
                for previous in slices.iter().take(level_idx) {
                    key.push(previous[row]);
                }
                let rank = rank_maps
                    .get(&key)
                    .and_then(|ranks| ranks.get(&current[row]))
                    .ok_or_else(|| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                            "failed to compute hierarchy rank",
                        )
                    })?;
                out.push(*rank);
            }
        }

        outputs.push(PyArray1::from_vec(py, out).into());
    }

    Ok(outputs)
}

/// Python module definition.
#[cfg(feature = "python")]
#[pymodule]
fn sciscape_leiden(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGraph>()?;
    m.add_function(wrap_pyfunction!(load_graph_raw_files, m)?)?;
    m.add_function(wrap_pyfunction!(rust_integer_remap_parquet, m)?)?;
    m.add_function(wrap_pyfunction!(rust_integer_remap_parquet_sidecars, m)?)?;
    m.add_function(wrap_pyfunction!(rust_integer_remap_parquet_graph, m)?)?;
    m.add_function(wrap_pyfunction!(run_leiden, m)?)?;
    m.add_function(wrap_pyfunction!(run_postprocess, m)?)?;
    m.add_function(wrap_pyfunction!(cpm_quality, m)?)?;
    m.add_function(wrap_pyfunction!(rust_filter_top_k, m)?)?;
    m.add_function(wrap_pyfunction!(rust_find_gcc, m)?)?;
    m.add_function(wrap_pyfunction!(rust_contract_edges, m)?)?;
    m.add_function(wrap_pyfunction!(rust_project_membership_u32, m)?)?;
    m.add_function(wrap_pyfunction!(rust_project_membership_u64, m)?)?;
    m.add_function(wrap_pyfunction!(rust_hierarchy_indices_u32, m)?)?;
    Ok(())
}
