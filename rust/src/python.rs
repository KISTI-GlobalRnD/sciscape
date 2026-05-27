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
use std::fmt::Write as _;

#[cfg(feature = "python")]
use crate::adaptive::{
    apply_split_merge_repair_candidates as compute_apply_split_merge_repair_candidates,
    boundary_group_probes as compute_boundary_group_probes,
    boundary_move_probes as compute_boundary_move_probes,
    cluster_graph_stats as compute_cluster_graph_stats,
    external_grain_group_candidates as compute_external_grain_group_candidates,
    external_grain_probes as compute_external_grain_probes,
    multi_core_split_probes as compute_multi_core_split_probes,
    split_merge_repair_probes as compute_split_merge_repair_probes,
    trim_oversize_boundary_moves as compute_trim_oversize_boundary_moves,
};
#[cfg(feature = "python")]
use crate::adaptive::{select_external_grain_probes, ExternalGrainSelectionPolicy};
#[cfg(feature = "python")]
use crate::contraction::create_reduced_network;
#[cfg(feature = "python")]
use crate::dongdaemun::{
    dongdaemun_refine as compute_dongdaemun_refine, DongdaemunConfig, DongdaemunPolicy,
    DongdaemunStatus,
};
#[cfg(feature = "python")]
use crate::fast_local_move::improve_clustering;
#[cfg(feature = "python")]
use crate::quality::{QualityFunction, CPM};
#[cfg(feature = "python")]
use crate::trace;
#[cfg(feature = "python")]
use crate::workspace::Workspace;
#[cfg(feature = "python")]
use crate::{
    leiden, leiden_multi_start, leiden_with_dongdaemun_refinement, postprocess_small_clusters,
    AdaptiveLocalShakeArm, AdaptiveLocalShakeFinalGuardMode, AdaptiveLocalShakeMode,
    AdaptiveNearTieProbeMode, AdaptiveProbeCommitStrategy, AdaptiveProbeMode, AdaptiveProbeTarget,
    BaselineRepairPolicy, CandidateQualityPolicy, Clustering, DongdaemunRefinementConfig, Graph,
    LeidenConfig, LeidenResult, ParentSelectionPolicy, PostprocessRound,
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
            if s != d {
                degree[s] += 1;
                degree[d] += 1;
            }
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
    let mut self_loop_weights = vec![0.0; n_nodes];
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
            if s == d {
                self_loop_weights[s] += w;
                continue;
            }

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
#[derive(Clone, Debug)]
struct NonMonotoneGroupEscapeCandidateRow {
    candidate_index: u64,
    source_cluster: u64,
    target_cluster: u64,
    group_kind: &'static str,
    block_count: u64,
    doc_weight: f64,
    incident_directed_edges: u64,
    assigned_fraction: f64,
    group_count: u64,
    group_weight: f64,
    group_fraction: f64,
    group_to_target_weight: f64,
    group_cut_weight: f64,
    group_move_delta_q: f64,
    group_split_delta_q: f64,
    group_delta_q: f64,
    best_group_delta_q: f64,
    best_group_action: u8,
    recommended_for_split_repair: bool,
    priority: f64,
    pre_polish_quality: f64,
    pre_polish_delta_q: f64,
    post_polish_quality: f64,
    post_polish_delta_q: f64,
    accepted_by_quality: bool,
    elapsed_ms: f64,
}

#[cfg(feature = "python")]
#[derive(Clone, Debug)]
struct NonMonotoneApproxPolishLabels {
    localized_quality: f64,
    localized_delta_q: f64,
    localized_elapsed_ms: f64,
    localized_active_nodes: u64,
    localized_active_clusters: u64,
    localized_rank: u64,
    quotient_quality: f64,
    quotient_delta_q: f64,
    quotient_elapsed_ms: f64,
    quotient_supernodes: u64,
    quotient_rank: u64,
    ub_delta_q: f64,
    ub_elapsed_ms: f64,
    ub_covers_p5: bool,
    ub_violation: f64,
    ub_rank: u64,
}

#[cfg(feature = "python")]
impl NonMonotoneApproxPolishLabels {
    fn empty() -> Self {
        Self {
            localized_quality: f64::NAN,
            localized_delta_q: f64::NAN,
            localized_elapsed_ms: f64::NAN,
            localized_active_nodes: 0,
            localized_active_clusters: 0,
            localized_rank: 0,
            quotient_quality: f64::NAN,
            quotient_delta_q: f64::NAN,
            quotient_elapsed_ms: f64::NAN,
            quotient_supernodes: 0,
            quotient_rank: 0,
            ub_delta_q: f64::NAN,
            ub_elapsed_ms: f64::NAN,
            ub_covers_p5: false,
            ub_violation: f64::NAN,
            ub_rank: 0,
        }
    }
}

#[cfg(feature = "python")]
#[derive(Clone, Debug)]
struct NonMonotoneBasinSignatureStats {
    signature: String,
    cluster_count: u64,
    changed_nodes_vs_baseline: u64,
    baseline_fragmentation_nodes: u64,
    baseline_mixing_nodes: u64,
    changed_fraction_vs_baseline: f64,
    relative_delta_q_ppm: f64,
    sketch_sample_size: u64,
    sketch_node_hash: String,
    sketch_baseline_membership: String,
    sketch_membership: String,
    changed_support_node_count: u64,
    changed_support_sketch_sample_size: u64,
    changed_support_node_hash: String,
    changed_support_nodes: String,
}

#[cfg(feature = "python")]
impl NonMonotoneBasinSignatureStats {
    fn empty() -> Self {
        Self {
            signature: String::new(),
            cluster_count: 0,
            changed_nodes_vs_baseline: 0,
            baseline_fragmentation_nodes: 0,
            baseline_mixing_nodes: 0,
            changed_fraction_vs_baseline: f64::NAN,
            relative_delta_q_ppm: f64::NAN,
            sketch_sample_size: 0,
            sketch_node_hash: String::new(),
            sketch_baseline_membership: String::new(),
            sketch_membership: String::new(),
            changed_support_node_count: 0,
            changed_support_sketch_sample_size: 0,
            changed_support_node_hash: String::new(),
            changed_support_nodes: String::new(),
        }
    }
}

#[cfg(feature = "python")]
#[derive(Clone, Debug)]
struct NonMonotoneGroupEscapeComputation {
    membership: Vec<u64>,
    quality: f64,
    accepted: bool,
    baseline_quality: f64,
    best_delta_q: f64,
    elapsed_ms: f64,
    candidate_eval_parallel: bool,
    candidate_eval_wall_elapsed_ms: f64,
    candidate_eval_cpu_sum_elapsed_ms: f64,
    candidate_eval_parallel_speedup: f64,
    candidate_eval_parallel_workers: u64,
    candidate_rows: Vec<NonMonotoneGroupEscapeCandidateRow>,
}

#[cfg(feature = "python")]
#[derive(Clone, Debug)]
struct NonMonotoneGroupEscapePolishResult {
    pre_polish_quality: f64,
    post_polish_quality: f64,
    membership: Option<Vec<u32>>,
    elapsed_ms: f64,
}

#[cfg(feature = "python")]
#[derive(Debug)]
struct NonMonotoneCandidateEvalAccumulator {
    rows: Vec<NonMonotoneGroupEscapeCandidateRow>,
    best_quality: f64,
    best_candidate_index: usize,
    best_membership: Option<Vec<u32>>,
    accepted: bool,
}

#[cfg(feature = "python")]
impl NonMonotoneCandidateEvalAccumulator {
    fn new() -> Self {
        Self {
            rows: Vec::new(),
            best_quality: f64::NEG_INFINITY,
            best_candidate_index: usize::MAX,
            best_membership: None,
            accepted: false,
        }
    }

    fn push(
        &mut self,
        idx: usize,
        candidate: &crate::adaptive::ExternalGrainGroupCandidate,
        polished: NonMonotoneGroupEscapePolishResult,
        baseline_quality: f64,
        quality_eps: f64,
        return_membership: bool,
    ) {
        let pre_polish_quality = polished.pre_polish_quality;
        let post_polish_quality = polished.post_polish_quality;
        let post_polish_delta_q = post_polish_quality - baseline_quality;
        let accepted_by_quality = post_polish_quality >= baseline_quality + quality_eps;
        if post_polish_quality > self.best_quality
            || (post_polish_quality == self.best_quality && idx < self.best_candidate_index)
        {
            self.best_quality = post_polish_quality;
            self.best_candidate_index = idx;
            self.best_membership = if return_membership {
                polished.membership
            } else {
                None
            };
        }
        if accepted_by_quality {
            self.accepted = true;
        }
        self.rows.push(NonMonotoneGroupEscapeCandidateRow {
            candidate_index: idx as u64,
            source_cluster: candidate.source_cluster,
            target_cluster: candidate.target_cluster,
            group_kind: candidate.group_kind.as_str(),
            block_count: candidate.block_count,
            doc_weight: candidate.doc_weight,
            incident_directed_edges: candidate.incident_directed_edges,
            assigned_fraction: candidate.assigned_fraction,
            group_count: candidate.group_count,
            group_weight: candidate.group_weight,
            group_fraction: candidate.group_fraction,
            group_to_target_weight: candidate.group_to_target_weight,
            group_cut_weight: candidate.group_cut_weight,
            group_move_delta_q: candidate.group_move_delta_q,
            group_split_delta_q: candidate.group_split_delta_q,
            group_delta_q: candidate.group_delta_q,
            best_group_delta_q: candidate.best_group_delta_q,
            best_group_action: candidate.best_group_action,
            recommended_for_split_repair: candidate.recommended_for_split_repair,
            priority: candidate.priority,
            pre_polish_quality,
            pre_polish_delta_q: pre_polish_quality - baseline_quality,
            post_polish_quality,
            post_polish_delta_q,
            accepted_by_quality,
            elapsed_ms: polished.elapsed_ms,
        });
    }

    fn merge(mut self, mut other: Self) -> Self {
        if other.best_quality > self.best_quality
            || (other.best_quality == self.best_quality
                && other.best_candidate_index < self.best_candidate_index)
        {
            self.best_quality = other.best_quality;
            self.best_candidate_index = other.best_candidate_index;
            self.best_membership = other.best_membership.take();
        }
        self.accepted |= other.accepted;
        self.rows.append(&mut other.rows);
        self
    }
}

#[cfg(feature = "python")]
#[derive(Clone, Debug)]
struct NonMonotoneGroupEscapeMultifidelityCandidateRow {
    candidate_index: u64,
    source_cluster: u64,
    target_cluster: u64,
    group_kind: &'static str,
    block_count: u64,
    doc_weight: f64,
    incident_directed_edges: u64,
    assigned_fraction: f64,
    group_count: u64,
    group_weight: f64,
    group_fraction: f64,
    group_to_target_weight: f64,
    group_cut_weight: f64,
    group_move_delta_q: f64,
    group_split_delta_q: f64,
    group_delta_q: f64,
    best_group_delta_q: f64,
    best_group_action: u8,
    recommended_for_split_repair: bool,
    priority: f64,
    pre_polish_quality: f64,
    pre_delta_q: f64,
    p1_quality: f64,
    p1_delta_q: f64,
    p1_elapsed_ms: f64,
    p5_quality: f64,
    p5_delta_q: f64,
    p5_elapsed_ms: f64,
    selected_by_p1_top1: bool,
    selected_by_p1_top2: bool,
    selected_by_full_p5: bool,
    approx: NonMonotoneApproxPolishLabels,
    basin: NonMonotoneBasinSignatureStats,
}

#[cfg(feature = "python")]
#[derive(Clone, Debug)]
struct NonMonotoneGroupEscapePolicyRow {
    policy: String,
    selected_candidate_index: i64,
    candidate_count: u64,
    p1_evaluated: u64,
    p5_evaluated: u64,
    p1_elapsed_ms: f64,
    p5_elapsed_ms: f64,
    total_elapsed_ms: f64,
    final_delta_q: f64,
    quality: f64,
    accepted: bool,
    available: bool,
    matches_full_p5: bool,
}

#[cfg(feature = "python")]
#[derive(Clone, Debug)]
struct NonMonotoneGroupEscapeMultifidelityComputation {
    membership: Vec<u64>,
    quality: f64,
    accepted: bool,
    selected_policy: String,
    selected_candidate_index: i64,
    baseline_quality: f64,
    best_delta_q: f64,
    elapsed_ms: f64,
    candidate_rows: Vec<NonMonotoneGroupEscapeMultifidelityCandidateRow>,
    policy_rows: Vec<NonMonotoneGroupEscapePolicyRow>,
}

#[cfg(feature = "python")]
fn external_group_kind_order(kind: crate::adaptive::ExternalGrainGroupKind) -> u8 {
    match kind {
        crate::adaptive::ExternalGrainGroupKind::Best => 0,
        crate::adaptive::ExternalGrainGroupKind::Largest => 1,
        crate::adaptive::ExternalGrainGroupKind::Second => 2,
    }
}

#[cfg(feature = "python")]
fn ranked_external_grain_group_candidates(
    graph: &Graph,
    baseline_clustering: &Clustering,
    candidate_clusters: &[u64],
    resolution: f64,
    max_candidates: usize,
    min_doc_weight: f64,
    min_assigned_fraction: f64,
    min_best_group_fraction: f64,
) -> Vec<crate::adaptive::ExternalGrainGroupCandidate> {
    let mut ws = Workspace::new(graph.n_nodes.max(baseline_clustering.n_clusters));
    let mut candidates = compute_external_grain_group_candidates(
        graph,
        baseline_clustering,
        candidate_clusters,
        resolution,
        0.0,
        ExternalGrainSelectionPolicy {
            min_doc_weight,
            max_incident_directed_edges: 0,
            min_best_delta_q: f64::NEG_INFINITY,
            min_assigned_fraction,
            min_best_group_fraction,
        },
        &mut ws,
    );
    candidates.sort_by(|left, right| {
        right
            .recommended_for_split_repair
            .cmp(&left.recommended_for_split_repair)
            .then_with(|| right.priority.total_cmp(&left.priority))
            .then_with(|| {
                right
                    .group_to_target_weight
                    .total_cmp(&left.group_to_target_weight)
            })
            .then_with(|| right.group_weight.total_cmp(&left.group_weight))
            .then_with(|| left.source_cluster.cmp(&right.source_cluster))
            .then_with(|| left.target_cluster.cmp(&right.target_cluster))
            .then_with(|| {
                external_group_kind_order(left.group_kind)
                    .cmp(&external_group_kind_order(right.group_kind))
            })
    });
    let mut seen_pairs = std::collections::HashSet::new();
    candidates.retain(|candidate| {
        seen_pairs.insert((candidate.source_cluster, candidate.target_cluster))
    });
    candidates.truncate(max_candidates);
    candidates
}

#[cfg(feature = "python")]
fn polish_external_grain_candidate(
    graph: &Graph,
    baseline_clustering: &Clustering,
    candidate: &crate::adaptive::ExternalGrainGroupCandidate,
    cpm: &CPM,
    baseline_quality: f64,
    resolution: f64,
    polish_iterations: usize,
    randomness: f64,
    seed: u64,
    return_membership: bool,
) -> Option<NonMonotoneGroupEscapePolishResult> {
    let candidate_start = Instant::now();
    let Ok(target_cluster) = u32::try_from(candidate.target_cluster) else {
        return None;
    };
    let mut perturbed_membership = baseline_clustering.clusters.clone();
    for &node_u32 in &candidate.nodes {
        let node = node_u32 as usize;
        if node < perturbed_membership.len() {
            perturbed_membership[node] = target_cluster;
        }
    }
    let mut perturbed = Clustering::from_assignments(perturbed_membership);
    perturbed.remove_empty_clusters();
    let pre_polish_quality = cpm.quality(graph, &perturbed);
    let polished = if polish_iterations == 0 {
        LeidenResult {
            clustering: perturbed,
            quality: pre_polish_quality,
            n_iterations_used: 0,
        }
    } else {
        let polish_config = LeidenConfig {
            resolution,
            n_iterations: polish_iterations,
            randomness,
            randomness_schedule: Vec::new(),
            seed,
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        leiden(graph, &polish_config, Some(perturbed), &mut rng)
    };
    let _ = baseline_quality;
    let post_polish_quality = polished.quality;
    let membership = if return_membership {
        Some(polished.clustering.clusters)
    } else {
        None
    };
    Some(NonMonotoneGroupEscapePolishResult {
        pre_polish_quality,
        post_polish_quality,
        membership,
        elapsed_ms: candidate_start.elapsed().as_secs_f64() * 1000.0,
    })
}

#[cfg(feature = "python")]
fn external_grain_active_region(
    graph: &Graph,
    baseline_clustering: &Clustering,
    candidate: &crate::adaptive::ExternalGrainGroupCandidate,
) -> (Vec<usize>, std::collections::HashSet<u32>) {
    let mut active_clusters = std::collections::HashSet::new();
    if let Ok(source) = u32::try_from(candidate.source_cluster) {
        active_clusters.insert(source);
    }
    if let Ok(target) = u32::try_from(candidate.target_cluster) {
        active_clusters.insert(target);
    }
    for &node_u32 in &candidate.nodes {
        let node = node_u32 as usize;
        if node >= graph.n_nodes {
            continue;
        }
        active_clusters.insert(baseline_clustering.clusters[node]);
        for (nbr, _) in graph.neighbors_of(node) {
            active_clusters.insert(baseline_clustering.clusters[nbr as usize]);
        }
    }
    let active_nodes = baseline_clustering
        .clusters
        .iter()
        .enumerate()
        .filter_map(|(node, &cluster)| active_clusters.contains(&cluster).then_some(node))
        .collect::<Vec<_>>();
    (active_nodes, active_clusters)
}

#[cfg(feature = "python")]
fn perturbed_external_grain_membership(
    baseline_clustering: &Clustering,
    candidate: &crate::adaptive::ExternalGrainGroupCandidate,
) -> Option<Vec<u32>> {
    let target_cluster = u32::try_from(candidate.target_cluster).ok()?;
    let mut membership = baseline_clustering.clusters.clone();
    for &node_u32 in &candidate.nodes {
        let node = node_u32 as usize;
        if node < membership.len() {
            membership[node] = target_cluster;
        }
    }
    Some(membership)
}

#[cfg(feature = "python")]
fn compact_local_cluster_id(
    cluster: u32,
    map: &mut std::collections::HashMap<u32, u32>,
    next_cluster: &mut u32,
) -> u32 {
    *map.entry(cluster).or_insert_with(|| {
        let assigned = *next_cluster;
        *next_cluster += 1;
        assigned
    })
}

#[cfg(feature = "python")]
#[allow(clippy::too_many_arguments)]
fn localized_constrained_polish_label(
    graph: &Graph,
    baseline_clustering: &Clustering,
    candidate: &crate::adaptive::ExternalGrainGroupCandidate,
    cpm: &CPM,
    baseline_quality: f64,
    resolution: f64,
    polish_iterations: usize,
    seed: u64,
) -> (f64, f64, f64, u64, u64) {
    let start = Instant::now();
    let Some(perturbed_membership) =
        perturbed_external_grain_membership(baseline_clustering, candidate)
    else {
        return (
            f64::NAN,
            f64::NAN,
            start.elapsed().as_secs_f64() * 1000.0,
            0,
            0,
        );
    };
    let (active_nodes, active_clusters) =
        external_grain_active_region(graph, baseline_clustering, candidate);
    if active_nodes.is_empty() {
        return (
            f64::NAN,
            f64::NAN,
            start.elapsed().as_secs_f64() * 1000.0,
            0,
            0,
        );
    }

    let mut active_node_to_local =
        std::collections::HashMap::<usize, usize>::with_capacity(active_nodes.len());
    let mut output_cluster_to_local_cluster = std::collections::HashMap::<u32, u32>::new();
    let mut next_local_cluster = 0u32;
    let mut local_node_weights = Vec::with_capacity(active_nodes.len());
    let mut local_initial_clusters = Vec::with_capacity(active_nodes.len());
    let mut local_output_clusters = Vec::with_capacity(active_nodes.len());
    let mut local_fixed = Vec::with_capacity(active_nodes.len());
    for (local_node, &global_node) in active_nodes.iter().enumerate() {
        active_node_to_local.insert(global_node, local_node);
        let output_cluster = perturbed_membership[global_node];
        local_node_weights.push(graph.node_weights[global_node]);
        local_initial_clusters.push(compact_local_cluster_id(
            output_cluster,
            &mut output_cluster_to_local_cluster,
            &mut next_local_cluster,
        ));
        local_output_clusters.push(output_cluster);
        local_fixed.push(false);
    }

    let mut anchor_cluster_to_local = std::collections::HashMap::<u32, usize>::new();
    let mut local_src = Vec::<u32>::new();
    let mut local_dst = Vec::<u32>::new();
    let mut local_weight = Vec::<f64>::new();

    for &global_node in &active_nodes {
        let local_node = active_node_to_local[&global_node];
        let self_loop = graph.self_loop_weights[global_node];
        if self_loop != 0.0 {
            local_src.push(local_node as u32);
            local_dst.push(local_node as u32);
            local_weight.push(self_loop);
        }
        for (nbr, weight) in graph.neighbors_of(global_node) {
            let nbr_node = nbr as usize;
            if let Some(&local_nbr) = active_node_to_local.get(&nbr_node) {
                if local_node < local_nbr {
                    local_src.push(local_node as u32);
                    local_dst.push(local_nbr as u32);
                    local_weight.push(weight);
                }
                continue;
            }
            let anchor_cluster = perturbed_membership[nbr_node];
            let anchor_local = *anchor_cluster_to_local
                .entry(anchor_cluster)
                .or_insert_with(|| {
                    let local_anchor = local_node_weights.len();
                    local_node_weights.push(0.0);
                    local_initial_clusters.push(compact_local_cluster_id(
                        anchor_cluster,
                        &mut output_cluster_to_local_cluster,
                        &mut next_local_cluster,
                    ));
                    local_output_clusters.push(anchor_cluster);
                    local_fixed.push(true);
                    local_anchor
                });
            local_src.push(local_node as u32);
            local_dst.push(anchor_local as u32);
            local_weight.push(weight);
        }
    }

    if !anchor_cluster_to_local.is_empty() {
        for (node, &cluster) in perturbed_membership.iter().enumerate() {
            if active_node_to_local.contains_key(&node) {
                continue;
            }
            if let Some(&local_anchor) = anchor_cluster_to_local.get(&cluster) {
                local_node_weights[local_anchor] += graph.node_weights[node];
            }
        }
    }

    let mut local_graph = Graph::from_edge_list_weighted(
        local_node_weights.len(),
        &local_src,
        &local_dst,
        &local_weight,
        &local_node_weights,
    );
    local_graph.simplify();
    let mut local_clustering = Clustering::from_assignments(local_initial_clusters);
    local_clustering.set_fixed(local_fixed.clone());
    if polish_iterations > 0 && local_graph.n_nodes > 1 {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut ws = Workspace::new(local_graph.n_nodes.max(local_clustering.n_clusters));
        for _ in 0..polish_iterations {
            let stats = improve_clustering(
                &local_graph,
                &mut local_clustering,
                resolution,
                &mut rng,
                &mut ws,
            );
            if !stats.improved {
                break;
            }
        }
    }

    let mut representative_by_cluster = vec![u32::MAX; local_clustering.n_clusters];
    for (local_node, &is_fixed) in local_fixed.iter().enumerate() {
        if !is_fixed {
            continue;
        }
        let cluster = local_clustering.clusters[local_node] as usize;
        if cluster < representative_by_cluster.len() {
            representative_by_cluster[cluster] =
                representative_by_cluster[cluster].min(local_output_clusters[local_node]);
        }
    }
    for (local_node, &is_fixed) in local_fixed.iter().enumerate() {
        if is_fixed {
            continue;
        }
        let cluster = local_clustering.clusters[local_node] as usize;
        if cluster < representative_by_cluster.len()
            && representative_by_cluster[cluster] == u32::MAX
        {
            representative_by_cluster[cluster] =
                representative_by_cluster[cluster].min(local_output_clusters[local_node]);
        }
    }

    let mut projected = baseline_clustering.clusters.clone();
    for (local_node, &global_node) in active_nodes.iter().enumerate() {
        let cluster = local_clustering.clusters[local_node] as usize;
        let output_cluster = representative_by_cluster
            .get(cluster)
            .copied()
            .filter(|&cluster| cluster != u32::MAX)
            .unwrap_or(local_output_clusters[local_node]);
        projected[global_node] = output_cluster;
    }
    let mut projected_clustering = Clustering::from_assignments(projected);
    projected_clustering.remove_empty_clusters();
    let quality = cpm.quality(graph, &projected_clustering);
    (
        quality,
        quality - baseline_quality,
        start.elapsed().as_secs_f64() * 1000.0,
        active_nodes.len() as u64,
        active_clusters.len() as u64,
    )
}

#[cfg(feature = "python")]
fn quotient_polish_label(
    graph: &Graph,
    baseline_clustering: &Clustering,
    candidate: &crate::adaptive::ExternalGrainGroupCandidate,
    cpm: &CPM,
    baseline_quality: f64,
    resolution: f64,
    final_iterations: usize,
    seed: u64,
) -> (f64, f64, f64, u64) {
    let start = Instant::now();
    let Some(target_cluster) = u32::try_from(candidate.target_cluster).ok() else {
        return (
            f64::NAN,
            f64::NAN,
            start.elapsed().as_secs_f64() * 1000.0,
            0,
        );
    };
    let Some(source_cluster) = u32::try_from(candidate.source_cluster).ok() else {
        return (
            f64::NAN,
            f64::NAN,
            start.elapsed().as_secs_f64() * 1000.0,
            0,
        );
    };
    let (_, active_clusters) = external_grain_active_region(graph, baseline_clustering, candidate);
    if active_clusters.is_empty() {
        return (
            f64::NAN,
            f64::NAN,
            start.elapsed().as_secs_f64() * 1000.0,
            0,
        );
    }

    let candidate_nodes = candidate
        .nodes
        .iter()
        .map(|&node| node as usize)
        .filter(|&node| node < graph.n_nodes)
        .collect::<std::collections::HashSet<_>>();
    if candidate_nodes.is_empty() {
        return (
            f64::NAN,
            f64::NAN,
            start.elapsed().as_secs_f64() * 1000.0,
            0,
        );
    }

    let mut cluster_nodes: std::collections::HashMap<u32, Vec<usize>> =
        std::collections::HashMap::new();
    for &cluster in &active_clusters {
        cluster_nodes.insert(cluster, Vec::new());
    }
    for (node, &cluster) in baseline_clustering.clusters.iter().enumerate() {
        if let Some(nodes) = cluster_nodes.get_mut(&cluster) {
            nodes.push(node);
        }
    }

    let mut supernodes: Vec<(u32, Vec<usize>)> = Vec::new();
    supernodes.push((
        target_cluster,
        candidate_nodes.iter().copied().collect::<Vec<_>>(),
    ));
    if let Some(source_nodes) = cluster_nodes.get(&source_cluster) {
        let remainder = source_nodes
            .iter()
            .copied()
            .filter(|node| !candidate_nodes.contains(node))
            .collect::<Vec<_>>();
        if !remainder.is_empty() {
            supernodes.push((source_cluster, remainder));
        }
    }
    if target_cluster != source_cluster {
        if let Some(target_nodes) = cluster_nodes.get(&target_cluster) {
            if !target_nodes.is_empty() {
                supernodes.push((target_cluster, target_nodes.clone()));
            }
        }
    }
    let mut neighbor_clusters = active_clusters
        .iter()
        .copied()
        .filter(|&cluster| cluster != source_cluster && cluster != target_cluster)
        .collect::<Vec<_>>();
    neighbor_clusters.sort_unstable();
    for cluster in neighbor_clusters {
        if let Some(nodes) = cluster_nodes.get(&cluster) {
            if !nodes.is_empty() {
                supernodes.push((cluster, nodes.clone()));
            }
        }
    }
    if supernodes.is_empty() {
        return (
            f64::NAN,
            f64::NAN,
            start.elapsed().as_secs_f64() * 1000.0,
            0,
        );
    }

    let mut active_node_to_super = std::collections::HashMap::new();
    let mut supernode_weights = Vec::with_capacity(supernodes.len());
    for (super_idx, (_, nodes)) in supernodes.iter().enumerate() {
        let mut weight = 0.0;
        for &node in nodes {
            active_node_to_super.insert(node, super_idx);
            weight += graph.node_weights[node];
        }
        supernode_weights.push(weight);
    }

    let mut q_src = Vec::new();
    let mut q_dst = Vec::new();
    let mut q_weight = Vec::new();
    for (super_idx, (_, nodes)) in supernodes.iter().enumerate() {
        for &node in nodes {
            let self_loop = graph.self_loop_weights[node];
            if self_loop != 0.0 {
                q_src.push(super_idx as u32);
                q_dst.push(super_idx as u32);
                q_weight.push(self_loop);
            }
            for (nbr, weight) in graph.neighbors_of(node) {
                let nbr_node = nbr as usize;
                if node > nbr_node {
                    continue;
                }
                let Some(&nbr_super) = active_node_to_super.get(&nbr_node) else {
                    continue;
                };
                q_src.push(super_idx as u32);
                q_dst.push(nbr_super as u32);
                q_weight.push(weight);
            }
        }
    }
    let quotient = Graph::from_edge_list_weighted(
        supernodes.len(),
        &q_src,
        &q_dst,
        &q_weight,
        &supernode_weights,
    );

    let mut compact_ids = std::collections::HashMap::<u32, u32>::new();
    let mut next_id = 0u32;
    let mut quotient_assignments = Vec::with_capacity(supernodes.len());
    for (output_cluster, _) in &supernodes {
        let cluster = *compact_ids.entry(*output_cluster).or_insert_with(|| {
            let assigned = next_id;
            next_id += 1;
            assigned
        });
        quotient_assignments.push(cluster);
    }
    quotient_assignments[0] = *compact_ids.entry(target_cluster).or_insert(0);
    let mut quotient_clustering = Clustering::from_assignments(quotient_assignments);
    let quotient_iterations = final_iterations.min(2);
    if quotient_iterations > 0 && quotient.n_nodes > 1 {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut ws = Workspace::new(quotient.n_nodes.max(quotient_clustering.n_clusters));
        for _ in 0..quotient_iterations {
            let stats = improve_clustering(
                &quotient,
                &mut quotient_clustering,
                resolution,
                &mut rng,
                &mut ws,
            );
            if !stats.improved {
                break;
            }
        }
    }

    let mut representative_by_cluster = vec![u32::MAX; quotient_clustering.n_clusters];
    for (super_idx, (output_cluster, _)) in supernodes.iter().enumerate() {
        let q_cluster = quotient_clustering.clusters[super_idx] as usize;
        if q_cluster >= representative_by_cluster.len() {
            continue;
        }
        representative_by_cluster[q_cluster] =
            representative_by_cluster[q_cluster].min(*output_cluster);
    }

    let mut projected = baseline_clustering.clusters.clone();
    for (super_idx, (_, nodes)) in supernodes.iter().enumerate() {
        let q_cluster = quotient_clustering.clusters[super_idx] as usize;
        let output_cluster = representative_by_cluster
            .get(q_cluster)
            .copied()
            .filter(|&cluster| cluster != u32::MAX)
            .unwrap_or(supernodes[super_idx].0);
        for &node in nodes {
            projected[node] = output_cluster;
        }
    }
    let mut projected_clustering = Clustering::from_assignments(projected);
    projected_clustering.remove_empty_clusters();
    let quality = cpm.quality(graph, &projected_clustering);
    (
        quality,
        quality - baseline_quality,
        start.elapsed().as_secs_f64() * 1000.0,
        supernodes.len() as u64,
    )
}

#[cfg(feature = "python")]
fn optimistic_upper_bound_label(
    graph: &Graph,
    baseline_clustering: &Clustering,
    candidate: &crate::adaptive::ExternalGrainGroupCandidate,
    baseline_quality: f64,
    pre_polish_quality: f64,
    resolution: f64,
) -> (f64, f64) {
    let start = Instant::now();
    let Some(perturbed_membership) =
        perturbed_external_grain_membership(baseline_clustering, candidate)
    else {
        return (f64::NAN, start.elapsed().as_secs_f64() * 1000.0);
    };
    let (active_nodes, _) = external_grain_active_region(graph, baseline_clustering, candidate);
    let clustering = Clustering::from_assignments(perturbed_membership);
    let cluster_weights = clustering.cluster_weights(&graph.node_weights);
    let pre_delta = pre_polish_quality - baseline_quality;
    let mut positive_gain_sum = 0.0;
    for node in active_nodes {
        let node_weight = graph.node_weights[node];
        let current_cluster = clustering.clusters[node] as usize;
        if current_cluster >= cluster_weights.len() {
            continue;
        }
        let mut edge_weight_by_cluster = std::collections::HashMap::<usize, f64>::new();
        for (nbr, weight) in graph.neighbors_of(node) {
            let cluster = clustering.clusters[nbr as usize] as usize;
            *edge_weight_by_cluster.entry(cluster).or_insert(0.0) += weight;
        }
        let current_cluster_weight = cluster_weights[current_cluster] - node_weight;
        let current_edge_weight = edge_weight_by_cluster
            .get(&current_cluster)
            .copied()
            .unwrap_or(0.0);
        let current_inc = current_edge_weight - node_weight * current_cluster_weight * resolution;
        let mut best_inc = current_inc.max(0.0);
        for (&cluster, &edge_weight) in &edge_weight_by_cluster {
            if cluster >= cluster_weights.len() {
                continue;
            }
            let target_weight = if cluster == current_cluster {
                cluster_weights[cluster] - node_weight
            } else {
                cluster_weights[cluster]
            };
            let inc = edge_weight - node_weight * target_weight * resolution;
            if inc > best_inc {
                best_inc = inc;
            }
        }
        if best_inc > current_inc {
            positive_gain_sum += best_inc - current_inc;
        }
    }
    (
        pre_delta + positive_gain_sum,
        start.elapsed().as_secs_f64() * 1000.0,
    )
}

#[cfg(feature = "python")]
#[allow(clippy::too_many_arguments)]
fn compute_approx_polish_labels(
    graph: &Graph,
    baseline_clustering: &Clustering,
    candidate: &crate::adaptive::ExternalGrainGroupCandidate,
    cpm: &CPM,
    baseline_quality: f64,
    pre_polish_quality: f64,
    resolution: f64,
    final_iterations: usize,
    seed: u64,
) -> NonMonotoneApproxPolishLabels {
    let (localized_quality, localized_delta_q, localized_elapsed_ms, active_nodes, active_clusters) =
        localized_constrained_polish_label(
            graph,
            baseline_clustering,
            candidate,
            cpm,
            baseline_quality,
            resolution,
            final_iterations,
            seed,
        );
    let (quotient_quality, quotient_delta_q, quotient_elapsed_ms, quotient_supernodes) =
        quotient_polish_label(
            graph,
            baseline_clustering,
            candidate,
            cpm,
            baseline_quality,
            resolution,
            final_iterations,
            seed,
        );
    let (ub_delta_q, ub_elapsed_ms) = optimistic_upper_bound_label(
        graph,
        baseline_clustering,
        candidate,
        baseline_quality,
        pre_polish_quality,
        resolution,
    );
    NonMonotoneApproxPolishLabels {
        localized_quality,
        localized_delta_q,
        localized_elapsed_ms,
        localized_active_nodes: active_nodes,
        localized_active_clusters: active_clusters,
        localized_rank: 0,
        quotient_quality,
        quotient_delta_q,
        quotient_elapsed_ms,
        quotient_supernodes,
        quotient_rank: 0,
        ub_delta_q,
        ub_elapsed_ms,
        ub_covers_p5: false,
        ub_violation: f64::NAN,
        ub_rank: 0,
    }
}

#[cfg(feature = "python")]
fn rank_multifidelity_rows_by(
    rows: &[NonMonotoneGroupEscapeMultifidelityCandidateRow],
    metric: fn(&NonMonotoneGroupEscapeMultifidelityCandidateRow) -> f64,
) -> Vec<usize> {
    let mut order = (0..rows.len()).collect::<Vec<_>>();
    order.sort_by(|&left, &right| {
        let left_value = metric(&rows[left]);
        let right_value = metric(&rows[right]);
        match (left_value.is_finite(), right_value.is_finite()) {
            (true, true) => right_value
                .total_cmp(&left_value)
                .then_with(|| rows[left].candidate_index.cmp(&rows[right].candidate_index)),
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            (false, false) => rows[left].candidate_index.cmp(&rows[right].candidate_index),
        }
    });
    order
}

#[cfg(feature = "python")]
fn assign_approx_polish_ranks(rows: &mut [NonMonotoneGroupEscapeMultifidelityCandidateRow]) {
    for row in rows.iter_mut() {
        row.approx.localized_rank = 0;
        row.approx.quotient_rank = 0;
        row.approx.ub_rank = 0;
    }
    for (rank, idx) in rank_multifidelity_rows_by(rows, |row| row.approx.localized_delta_q)
        .into_iter()
        .enumerate()
    {
        if rows[idx].approx.localized_delta_q.is_finite() {
            rows[idx].approx.localized_rank = rank as u64 + 1;
        }
    }
    for (rank, idx) in rank_multifidelity_rows_by(rows, |row| row.approx.quotient_delta_q)
        .into_iter()
        .enumerate()
    {
        if rows[idx].approx.quotient_delta_q.is_finite() {
            rows[idx].approx.quotient_rank = rank as u64 + 1;
        }
    }
    for (rank, idx) in rank_multifidelity_rows_by(rows, |row| row.approx.ub_delta_q)
        .into_iter()
        .enumerate()
    {
        if rows[idx].approx.ub_delta_q.is_finite() {
            rows[idx].approx.ub_rank = rank as u64 + 1;
        }
    }
}

#[cfg(feature = "python")]
fn clusters_to_u64(clusters: &[u32]) -> Vec<u64> {
    clusters.iter().map(|&cluster| cluster as u64).collect()
}

#[cfg(feature = "python")]
fn mix_fnv1a_u64(hash: &mut u64, value: u64) {
    const FNV_PRIME: u64 = 1_099_511_628_211;
    for byte in value.to_le_bytes() {
        *hash ^= u64::from(byte);
        *hash = hash.wrapping_mul(FNV_PRIME);
    }
}

#[cfg(feature = "python")]
fn canonical_partition_signature(clusters: &[u32]) -> (String, u64) {
    const FNV_OFFSET: u64 = 14_695_981_039_346_656_037;
    let mut canonical_ids: std::collections::HashMap<u32, u64> = std::collections::HashMap::new();
    let mut next_id = 0_u64;
    let mut hash = FNV_OFFSET;
    mix_fnv1a_u64(&mut hash, clusters.len() as u64);
    for &cluster in clusters {
        let canonical_id = match canonical_ids.get(&cluster) {
            Some(&id) => id,
            None => {
                let id = next_id;
                canonical_ids.insert(cluster, id);
                next_id += 1;
                id
            }
        };
        mix_fnv1a_u64(&mut hash, canonical_id);
    }
    (format!("{hash:016x}"), next_id)
}

#[cfg(feature = "python")]
const BASIN_SIGNATURE_SKETCH_SAMPLE_SIZE: usize = 1024;

#[cfg(feature = "python")]
const BASIN_CHANGED_SUPPORT_SKETCH_SAMPLE_SIZE: usize = 8192;

#[cfg(feature = "python")]
fn stable_node_sample_key(node: u32) -> u64 {
    let mut x = u64::from(node).wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

#[cfg(feature = "python")]
fn hash_u32_sequence(values: &[u32]) -> String {
    const FNV_OFFSET: u64 = 14_695_981_039_346_656_037;
    let mut hash = FNV_OFFSET;
    mix_fnv1a_u64(&mut hash, values.len() as u64);
    for &value in values {
        mix_fnv1a_u64(&mut hash, u64::from(value));
    }
    format!("{hash:016x}")
}

#[cfg(feature = "python")]
fn encode_membership_sketch(labels: &[u32], nodes: &[u32]) -> String {
    let mut out = String::new();
    for (idx, &node_u32) in nodes.iter().enumerate() {
        if idx > 0 {
            out.push(';');
        }
        let node = node_u32 as usize;
        let value = labels.get(node).copied().unwrap_or(u32::MAX);
        let _ = write!(&mut out, "{value}");
    }
    out
}

#[cfg(feature = "python")]
fn encode_u32_sketch(values: &[u32]) -> String {
    let mut out = String::new();
    for (idx, &value) in values.iter().enumerate() {
        if idx > 0 {
            out.push(';');
        }
        let _ = write!(&mut out, "{value}");
    }
    out
}

#[cfg(feature = "python")]
fn stable_sample_nodes(mut nodes: Vec<u32>, max_nodes: usize) -> Vec<u32> {
    nodes.sort_unstable();
    nodes.dedup();
    if nodes.len() > max_nodes {
        nodes.sort_unstable_by_key(|&node| (stable_node_sample_key(node), node));
        nodes.truncate(max_nodes);
        nodes.sort_unstable();
    }
    nodes
}

#[cfg(feature = "python")]
fn basin_signature_sketch_nodes(
    graph: &Graph,
    baseline_clustering: &Clustering,
    candidates: &[crate::adaptive::ExternalGrainGroupCandidate],
    max_nodes: usize,
) -> Vec<u32> {
    if max_nodes == 0 || graph.n_nodes == 0 {
        return Vec::new();
    }
    let mut active_clusters = std::collections::HashSet::new();
    for candidate in candidates {
        if let Ok(source) = u32::try_from(candidate.source_cluster) {
            active_clusters.insert(source);
        }
        if let Ok(target) = u32::try_from(candidate.target_cluster) {
            active_clusters.insert(target);
        }
        for &node_u32 in &candidate.nodes {
            let node = node_u32 as usize;
            if node >= graph.n_nodes || node >= baseline_clustering.clusters.len() {
                continue;
            }
            active_clusters.insert(baseline_clustering.clusters[node]);
            for (nbr, _) in graph.neighbors_of(node) {
                let nbr_idx = nbr as usize;
                if nbr_idx < baseline_clustering.clusters.len() {
                    active_clusters.insert(baseline_clustering.clusters[nbr_idx]);
                }
            }
        }
    }
    let nodes = if active_clusters.is_empty() {
        (0..graph.n_nodes)
            .filter_map(|node| u32::try_from(node).ok())
            .collect::<Vec<_>>()
    } else {
        baseline_clustering
            .clusters
            .iter()
            .enumerate()
            .filter_map(|(node, &cluster)| {
                active_clusters
                    .contains(&cluster)
                    .then(|| u32::try_from(node).ok())
                    .flatten()
            })
            .collect::<Vec<_>>()
    };
    stable_sample_nodes(nodes, max_nodes)
}

#[cfg(feature = "python")]
fn compute_basin_signature_stats(
    baseline: &[u32],
    membership: &[u32],
    p5_delta_q: f64,
    baseline_quality: f64,
    sketch_nodes: &[u32],
) -> NonMonotoneBasinSignatureStats {
    if baseline.len() != membership.len() {
        return NonMonotoneBasinSignatureStats::empty();
    }
    let (signature, cluster_count) = canonical_partition_signature(membership);
    let mut pair_counts: std::collections::HashMap<(u32, u32), u64> =
        std::collections::HashMap::new();
    let mut baseline_best: std::collections::HashMap<u32, u64> = std::collections::HashMap::new();
    let mut membership_best: std::collections::HashMap<u32, u64> = std::collections::HashMap::new();
    let mut baseline_best_membership: std::collections::HashMap<u32, (u32, u64)> =
        std::collections::HashMap::new();
    let mut membership_best_baseline: std::collections::HashMap<u32, (u32, u64)> =
        std::collections::HashMap::new();
    for (&baseline_cluster, &membership_cluster) in baseline.iter().zip(membership.iter()) {
        *pair_counts
            .entry((baseline_cluster, membership_cluster))
            .or_insert(0) += 1;
    }
    for ((baseline_cluster, membership_cluster), count) in pair_counts {
        let baseline_entry = baseline_best.entry(baseline_cluster).or_insert(0);
        *baseline_entry = (*baseline_entry).max(count);
        let membership_entry = membership_best.entry(membership_cluster).or_insert(0);
        *membership_entry = (*membership_entry).max(count);
        let baseline_target = baseline_best_membership
            .entry(baseline_cluster)
            .or_insert((membership_cluster, 0));
        if count > baseline_target.1
            || (count == baseline_target.1 && membership_cluster < baseline_target.0)
        {
            *baseline_target = (membership_cluster, count);
        }
        let membership_source = membership_best_baseline
            .entry(membership_cluster)
            .or_insert((baseline_cluster, 0));
        if count > membership_source.1
            || (count == membership_source.1 && baseline_cluster < membership_source.0)
        {
            *membership_source = (baseline_cluster, count);
        }
    }
    let n_nodes = baseline.len() as u64;
    let baseline_aligned = baseline_best.values().copied().sum::<u64>();
    let membership_aligned = membership_best.values().copied().sum::<u64>();
    let baseline_fragmentation_nodes = n_nodes.saturating_sub(baseline_aligned);
    let baseline_mixing_nodes = n_nodes.saturating_sub(membership_aligned);
    let changed_nodes_vs_baseline = baseline_fragmentation_nodes.max(baseline_mixing_nodes);
    let changed_fraction_vs_baseline = if n_nodes == 0 {
        f64::NAN
    } else {
        changed_nodes_vs_baseline as f64 / n_nodes as f64
    };
    let relative_delta_q_ppm = if p5_delta_q.is_finite() && baseline_quality.abs() > 0.0 {
        p5_delta_q / baseline_quality.abs() * 1_000_000.0
    } else {
        f64::NAN
    };
    let sketch_sample_size = sketch_nodes.len() as u64;
    let sketch_node_hash = hash_u32_sequence(sketch_nodes);
    let sketch_baseline_membership = encode_membership_sketch(baseline, sketch_nodes);
    let sketch_membership = encode_membership_sketch(membership, sketch_nodes);
    let changed_support_nodes = baseline
        .iter()
        .zip(membership.iter())
        .enumerate()
        .filter_map(|(node, (&baseline_cluster, &membership_cluster))| {
            let baseline_aligned = baseline_best_membership
                .get(&baseline_cluster)
                .is_some_and(|&(best_membership, _)| best_membership == membership_cluster);
            let membership_aligned = membership_best_baseline
                .get(&membership_cluster)
                .is_some_and(|&(best_baseline, _)| best_baseline == baseline_cluster);
            (!(baseline_aligned && membership_aligned))
                .then(|| u32::try_from(node).ok())
                .flatten()
        })
        .collect::<Vec<_>>();
    let changed_support_node_count = changed_support_nodes.len() as u64;
    let changed_support_sketch = stable_sample_nodes(
        changed_support_nodes,
        BASIN_CHANGED_SUPPORT_SKETCH_SAMPLE_SIZE,
    );
    let changed_support_sketch_sample_size = changed_support_sketch.len() as u64;
    let changed_support_node_hash = hash_u32_sequence(&changed_support_sketch);
    let changed_support_nodes = encode_u32_sketch(&changed_support_sketch);
    NonMonotoneBasinSignatureStats {
        signature,
        cluster_count,
        changed_nodes_vs_baseline,
        baseline_fragmentation_nodes,
        baseline_mixing_nodes,
        changed_fraction_vs_baseline,
        relative_delta_q_ppm,
        sketch_sample_size,
        sketch_node_hash,
        sketch_baseline_membership,
        sketch_membership,
        changed_support_node_count,
        changed_support_sketch_sample_size,
        changed_support_node_hash,
        changed_support_nodes,
    }
}

#[cfg(feature = "python")]
fn ensure_membership_len(clustering: &Clustering, n_nodes: usize) -> PyResult<()> {
    if clustering.clusters.len() != n_nodes {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "membership length {} does not match graph node count {}",
            clustering.clusters.len(),
            n_nodes,
        )));
    }
    Ok(())
}

#[cfg(feature = "python")]
#[allow(clippy::too_many_arguments)]
fn compute_external_grain_priority_clusters(
    graph: &Graph,
    clustering: &Clustering,
    candidate_clusters: &[u64],
    resolution: f64,
    epsilon: f64,
    count: usize,
    selection_policy: ExternalGrainSelectionPolicy,
) -> Vec<u64> {
    if count == 0 {
        return Vec::new();
    }
    let mut ws = Workspace::new(graph.n_nodes.max(clustering.n_clusters));
    let probes = compute_external_grain_probes(
        graph,
        clustering,
        candidate_clusters,
        resolution,
        epsilon,
        &mut ws,
    );
    let selection = select_external_grain_probes(&probes, selection_policy);
    let mut order = (0..probes.len()).collect::<Vec<_>>();
    order.sort_by(|&left, &right| {
        selection[right]
            .recommended_for_split_repair
            .cmp(&selection[left].recommended_for_split_repair)
            .then_with(|| {
                selection[right]
                    .priority
                    .total_cmp(&selection[left].priority)
            })
            .then_with(|| {
                probes[right]
                    .best_group_to_target_weight
                    .total_cmp(&probes[left].best_group_to_target_weight)
            })
            .then_with(|| {
                probes[right]
                    .best_group_weight
                    .total_cmp(&probes[left].best_group_weight)
            })
    });

    let mut out = Vec::with_capacity(count.min(order.len()));
    for idx in order {
        if out.len() >= count {
            break;
        }
        if probes[idx].best_group_target < 0 {
            continue;
        }
        out.push(probes[idx].cluster);
    }
    out
}

#[cfg(feature = "python")]
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
fn compute_non_monotone_group_escape_probe(
    graph: &Graph,
    baseline_clustering: &Clustering,
    candidate_clusters: &[u64],
    resolution: f64,
    max_candidates: usize,
    polish_iterations: usize,
    randomness: f64,
    seed: u64,
    min_doc_weight: f64,
    min_assigned_fraction: f64,
    min_best_group_fraction: f64,
    quality_eps: f64,
    parallel_candidates: bool,
) -> Result<NonMonotoneGroupEscapeComputation, String> {
    compute_non_monotone_group_escape_probe_impl(
        graph,
        baseline_clustering,
        candidate_clusters,
        resolution,
        max_candidates,
        polish_iterations,
        randomness,
        seed,
        min_doc_weight,
        min_assigned_fraction,
        min_best_group_fraction,
        quality_eps,
        parallel_candidates,
        true,
    )
}

#[cfg(feature = "python")]
#[allow(clippy::too_many_arguments)]
fn compute_non_monotone_group_escape_probe_impl(
    graph: &Graph,
    baseline_clustering: &Clustering,
    candidate_clusters: &[u64],
    resolution: f64,
    max_candidates: usize,
    polish_iterations: usize,
    randomness: f64,
    seed: u64,
    min_doc_weight: f64,
    min_assigned_fraction: f64,
    min_best_group_fraction: f64,
    quality_eps: f64,
    parallel_candidates: bool,
    return_membership: bool,
) -> Result<NonMonotoneGroupEscapeComputation, String> {
    if !resolution.is_finite() || resolution <= 0.0 {
        return Err("resolution must be finite and positive".to_string());
    }
    if !randomness.is_finite() || randomness < 0.0 {
        return Err("randomness must be finite and non-negative".to_string());
    }
    if !min_doc_weight.is_finite() || min_doc_weight < 0.0 {
        return Err("min_doc_weight must be finite and non-negative".to_string());
    }
    if !min_assigned_fraction.is_finite() || min_assigned_fraction < 0.0 {
        return Err("min_assigned_fraction must be finite and non-negative".to_string());
    }
    if !min_best_group_fraction.is_finite() || min_best_group_fraction < 0.0 {
        return Err("min_best_group_fraction must be finite and non-negative".to_string());
    }
    if !quality_eps.is_finite() {
        return Err("quality_eps must be finite".to_string());
    }
    if baseline_clustering.clusters.len() != graph.n_nodes {
        return Err(format!(
            "membership length {} does not match graph node count {}",
            baseline_clustering.clusters.len(),
            graph.n_nodes,
        ));
    }

    let total_start = Instant::now();
    let cpm = CPM::new(resolution);
    let baseline_quality = cpm.quality(graph, baseline_clustering);

    if max_candidates == 0 {
        return Ok(NonMonotoneGroupEscapeComputation {
            membership: if return_membership {
                clusters_to_u64(&baseline_clustering.clusters)
            } else {
                Vec::new()
            },
            quality: baseline_quality,
            accepted: false,
            baseline_quality,
            best_delta_q: 0.0,
            elapsed_ms: total_start.elapsed().as_secs_f64() * 1000.0,
            candidate_eval_parallel: false,
            candidate_eval_wall_elapsed_ms: 0.0,
            candidate_eval_cpu_sum_elapsed_ms: 0.0,
            candidate_eval_parallel_speedup: f64::NAN,
            candidate_eval_parallel_workers: 0,
            candidate_rows: Vec::new(),
        });
    }

    let candidates = ranked_external_grain_group_candidates(
        graph,
        baseline_clustering,
        candidate_clusters,
        resolution,
        max_candidates,
        min_doc_weight,
        min_assigned_fraction,
        min_best_group_fraction,
    );

    let mut rows = Vec::with_capacity(candidates.len());
    let candidate_eval_parallel = parallel_candidates && candidates.len() > 1;
    let candidate_eval_workers = if candidates.is_empty() {
        0
    } else if candidate_eval_parallel {
        rayon::current_num_threads().min(candidates.len()) as u64
    } else {
        1
    };
    let candidate_eval_start = Instant::now();
    let mut eval = if candidate_eval_parallel {
        use rayon::prelude::*;
        candidates
            .par_iter()
            .enumerate()
            .fold(
                NonMonotoneCandidateEvalAccumulator::new,
                |mut acc, (idx, candidate)| {
                    let polish_seed = seed.wrapping_add(idx as u64);
                    if let Some(polished) = polish_external_grain_candidate(
                        graph,
                        baseline_clustering,
                        candidate,
                        &cpm,
                        baseline_quality,
                        resolution,
                        polish_iterations,
                        randomness,
                        polish_seed,
                        return_membership,
                    ) {
                        acc.push(
                            idx,
                            candidate,
                            polished,
                            baseline_quality,
                            quality_eps,
                            return_membership,
                        );
                    }
                    acc
                },
            )
            .reduce(NonMonotoneCandidateEvalAccumulator::new, |left, right| {
                left.merge(right)
            })
    } else {
        let mut acc = NonMonotoneCandidateEvalAccumulator::new();
        for (idx, candidate) in candidates.iter().enumerate() {
            let polish_seed = seed.wrapping_add(idx as u64);
            if let Some(polished) = polish_external_grain_candidate(
                graph,
                baseline_clustering,
                candidate,
                &cpm,
                baseline_quality,
                resolution,
                polish_iterations,
                randomness,
                polish_seed,
                return_membership,
            ) {
                acc.push(
                    idx,
                    candidate,
                    polished,
                    baseline_quality,
                    quality_eps,
                    return_membership,
                );
            }
        }
        acc
    };
    let candidate_eval_wall_elapsed_ms = candidate_eval_start.elapsed().as_secs_f64() * 1000.0;

    rows.append(&mut eval.rows);
    rows.sort_by_key(|row| row.candidate_index);

    let accepted = eval.accepted;
    let best_quality = if accepted {
        eval.best_quality
    } else {
        baseline_quality
    };
    let best_membership = if accepted {
        if return_membership {
            eval.best_membership
                .take()
                .map(|membership| clusters_to_u64(&membership))
                .unwrap_or_else(|| clusters_to_u64(&baseline_clustering.clusters))
        } else {
            Vec::new()
        }
    } else if return_membership {
        clusters_to_u64(&baseline_clustering.clusters)
    } else {
        Vec::new()
    };
    let best_delta_q = if rows.is_empty() {
        0.0
    } else {
        rows.iter()
            .map(|row| row.post_polish_delta_q)
            .fold(f64::NEG_INFINITY, f64::max)
    };
    let candidate_eval_cpu_sum_elapsed_ms = rows.iter().map(|row| row.elapsed_ms).sum::<f64>();
    let candidate_eval_parallel_speedup = if candidate_eval_wall_elapsed_ms > 0.0 {
        candidate_eval_cpu_sum_elapsed_ms / candidate_eval_wall_elapsed_ms
    } else {
        f64::NAN
    };

    Ok(NonMonotoneGroupEscapeComputation {
        membership: best_membership,
        quality: best_quality,
        accepted,
        baseline_quality,
        best_delta_q,
        elapsed_ms: total_start.elapsed().as_secs_f64() * 1000.0,
        candidate_eval_parallel,
        candidate_eval_wall_elapsed_ms,
        candidate_eval_cpu_sum_elapsed_ms,
        candidate_eval_parallel_speedup,
        candidate_eval_parallel_workers: candidate_eval_workers,
        candidate_rows: rows,
    })
}

#[cfg(feature = "python")]
fn validate_non_monotone_multifidelity_inputs(
    graph: &Graph,
    baseline_clustering: &Clustering,
    resolution: f64,
    randomness: f64,
    min_doc_weight: f64,
    min_assigned_fraction: f64,
    min_best_group_fraction: f64,
    quality_eps: f64,
) -> Result<(), String> {
    if !resolution.is_finite() || resolution <= 0.0 {
        return Err("resolution must be finite and positive".to_string());
    }
    if !randomness.is_finite() || randomness < 0.0 {
        return Err("randomness must be finite and non-negative".to_string());
    }
    if !min_doc_weight.is_finite() || min_doc_weight < 0.0 {
        return Err("min_doc_weight must be finite and non-negative".to_string());
    }
    if !min_assigned_fraction.is_finite() || min_assigned_fraction < 0.0 {
        return Err("min_assigned_fraction must be finite and non-negative".to_string());
    }
    if !min_best_group_fraction.is_finite() || min_best_group_fraction < 0.0 {
        return Err("min_best_group_fraction must be finite and non-negative".to_string());
    }
    if !quality_eps.is_finite() {
        return Err("quality_eps must be finite".to_string());
    }
    if baseline_clustering.clusters.len() != graph.n_nodes {
        return Err(format!(
            "membership length {} does not match graph node count {}",
            baseline_clustering.clusters.len(),
            graph.n_nodes,
        ));
    }
    Ok(())
}

#[cfg(feature = "python")]
fn best_p5_index(
    rows: &[NonMonotoneGroupEscapeMultifidelityCandidateRow],
    indices: &[usize],
) -> Option<usize> {
    indices
        .iter()
        .copied()
        .filter(|&idx| idx < rows.len() && rows[idx].p5_delta_q.is_finite())
        .max_by(|&left, &right| {
            rows[left]
                .p5_delta_q
                .total_cmp(&rows[right].p5_delta_q)
                .then_with(|| rows[right].candidate_index.cmp(&rows[left].candidate_index))
        })
}

#[cfg(feature = "python")]
fn p5_candidate_is_better(
    rows: &[NonMonotoneGroupEscapeMultifidelityCandidateRow],
    left: usize,
    right: usize,
) -> bool {
    rows[left]
        .p5_delta_q
        .total_cmp(&rows[right].p5_delta_q)
        .then_with(|| rows[right].candidate_index.cmp(&rows[left].candidate_index))
        == std::cmp::Ordering::Greater
}

#[cfg(feature = "python")]
#[allow(clippy::too_many_arguments)]
fn build_policy_row(
    policy: String,
    rows: &[NonMonotoneGroupEscapeMultifidelityCandidateRow],
    candidate_indices: &[usize],
    p1_evaluated: u64,
    require_all_p5: bool,
    baseline_quality: f64,
    quality_eps: f64,
    full_p5_winner: Option<usize>,
) -> NonMonotoneGroupEscapePolicyRow {
    let candidate_count = rows.len() as u64;
    let p5_indices = candidate_indices
        .iter()
        .copied()
        .filter(|&idx| idx < rows.len() && rows[idx].p5_delta_q.is_finite())
        .collect::<Vec<_>>();
    let available = !candidate_indices.is_empty()
        && !p5_indices.is_empty()
        && (!require_all_p5 || p5_indices.len() == candidate_indices.len());
    let p1_elapsed_ms = if p1_evaluated > 0 {
        rows.iter().map(|row| row.p1_elapsed_ms).sum()
    } else {
        0.0
    };
    let p5_elapsed_ms = p5_indices
        .iter()
        .map(|&idx| rows[idx].p5_elapsed_ms)
        .sum::<f64>();
    let selected = if available {
        best_p5_index(rows, candidate_indices)
    } else {
        None
    };
    let (selected_candidate_index, final_delta_q, quality, accepted, matches_full_p5) =
        if let Some(idx) = selected {
            let row = &rows[idx];
            let quality = row.p5_quality;
            (
                row.candidate_index as i64,
                row.p5_delta_q,
                quality,
                quality >= baseline_quality + quality_eps,
                full_p5_winner == Some(idx),
            )
        } else {
            (-1, f64::NAN, f64::NAN, false, false)
        };
    NonMonotoneGroupEscapePolicyRow {
        policy,
        selected_candidate_index,
        candidate_count,
        p1_evaluated,
        p5_evaluated: p5_indices.len() as u64,
        p1_elapsed_ms,
        p5_elapsed_ms,
        total_elapsed_ms: p1_elapsed_ms + p5_elapsed_ms,
        final_delta_q,
        quality,
        accepted,
        available,
        matches_full_p5,
    }
}

#[cfg(feature = "python")]
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
fn compute_non_monotone_group_escape_multifidelity_probe(
    graph: &Graph,
    baseline_clustering: &Clustering,
    candidate_clusters: &[u64],
    resolution: f64,
    max_candidates: usize,
    prescreen_iterations: usize,
    final_iterations: usize,
    finalists: usize,
    label_full_p5: bool,
    randomness: f64,
    seed: u64,
    min_doc_weight: f64,
    min_assigned_fraction: f64,
    min_best_group_fraction: f64,
    quality_eps: f64,
    approx_polish_labels: bool,
    basin_signatures: bool,
) -> Result<NonMonotoneGroupEscapeMultifidelityComputation, String> {
    compute_non_monotone_group_escape_multifidelity_probe_impl(
        graph,
        baseline_clustering,
        candidate_clusters,
        resolution,
        max_candidates,
        prescreen_iterations,
        final_iterations,
        finalists,
        label_full_p5,
        randomness,
        seed,
        min_doc_weight,
        min_assigned_fraction,
        min_best_group_fraction,
        quality_eps,
        true,
        approx_polish_labels,
        basin_signatures,
    )
}

#[cfg(feature = "python")]
#[allow(clippy::too_many_arguments)]
fn compute_non_monotone_group_escape_multifidelity_probe_impl(
    graph: &Graph,
    baseline_clustering: &Clustering,
    candidate_clusters: &[u64],
    resolution: f64,
    max_candidates: usize,
    prescreen_iterations: usize,
    final_iterations: usize,
    finalists: usize,
    label_full_p5: bool,
    randomness: f64,
    seed: u64,
    min_doc_weight: f64,
    min_assigned_fraction: f64,
    min_best_group_fraction: f64,
    quality_eps: f64,
    return_membership: bool,
    approx_polish_labels: bool,
    basin_signatures: bool,
) -> Result<NonMonotoneGroupEscapeMultifidelityComputation, String> {
    validate_non_monotone_multifidelity_inputs(
        graph,
        baseline_clustering,
        resolution,
        randomness,
        min_doc_weight,
        min_assigned_fraction,
        min_best_group_fraction,
        quality_eps,
    )?;

    let total_start = Instant::now();
    let cpm = CPM::new(resolution);
    let baseline_quality = cpm.quality(graph, baseline_clustering);

    if max_candidates == 0 {
        return Ok(NonMonotoneGroupEscapeMultifidelityComputation {
            membership: if return_membership {
                clusters_to_u64(&baseline_clustering.clusters)
            } else {
                Vec::new()
            },
            quality: baseline_quality,
            accepted: false,
            selected_policy: "none".to_string(),
            selected_candidate_index: -1,
            baseline_quality,
            best_delta_q: 0.0,
            elapsed_ms: total_start.elapsed().as_secs_f64() * 1000.0,
            candidate_rows: Vec::new(),
            policy_rows: Vec::new(),
        });
    }

    let candidates = ranked_external_grain_group_candidates(
        graph,
        baseline_clustering,
        candidate_clusters,
        resolution,
        max_candidates,
        min_doc_weight,
        min_assigned_fraction,
        min_best_group_fraction,
    );

    let mut rows = Vec::with_capacity(candidates.len());

    for (idx, candidate) in candidates.iter().enumerate() {
        let prescreen_seed = seed.wrapping_add(idx as u64);
        let Some(p1) = polish_external_grain_candidate(
            graph,
            baseline_clustering,
            candidate,
            &cpm,
            baseline_quality,
            resolution,
            prescreen_iterations,
            randomness,
            prescreen_seed,
            false,
        ) else {
            continue;
        };
        rows.push(NonMonotoneGroupEscapeMultifidelityCandidateRow {
            candidate_index: idx as u64,
            source_cluster: candidate.source_cluster,
            target_cluster: candidate.target_cluster,
            group_kind: candidate.group_kind.as_str(),
            block_count: candidate.block_count,
            doc_weight: candidate.doc_weight,
            incident_directed_edges: candidate.incident_directed_edges,
            assigned_fraction: candidate.assigned_fraction,
            group_count: candidate.group_count,
            group_weight: candidate.group_weight,
            group_fraction: candidate.group_fraction,
            group_to_target_weight: candidate.group_to_target_weight,
            group_cut_weight: candidate.group_cut_weight,
            group_move_delta_q: candidate.group_move_delta_q,
            group_split_delta_q: candidate.group_split_delta_q,
            group_delta_q: candidate.group_delta_q,
            best_group_delta_q: candidate.best_group_delta_q,
            best_group_action: candidate.best_group_action,
            recommended_for_split_repair: candidate.recommended_for_split_repair,
            priority: candidate.priority,
            pre_polish_quality: p1.pre_polish_quality,
            pre_delta_q: p1.pre_polish_quality - baseline_quality,
            p1_quality: p1.post_polish_quality,
            p1_delta_q: p1.post_polish_quality - baseline_quality,
            p1_elapsed_ms: p1.elapsed_ms,
            p5_quality: f64::NAN,
            p5_delta_q: f64::NAN,
            p5_elapsed_ms: f64::NAN,
            selected_by_p1_top1: false,
            selected_by_p1_top2: false,
            selected_by_full_p5: false,
            approx: NonMonotoneApproxPolishLabels::empty(),
            basin: NonMonotoneBasinSignatureStats::empty(),
        });
    }

    if approx_polish_labels {
        for row in &mut rows {
            let candidate_index = row.candidate_index as usize;
            let Some(candidate) = candidates.get(candidate_index) else {
                continue;
            };
            let label_seed = seed
                .wrapping_add(10_000)
                .wrapping_add(candidate_index as u64);
            row.approx = compute_approx_polish_labels(
                graph,
                baseline_clustering,
                candidate,
                &cpm,
                baseline_quality,
                row.pre_polish_quality,
                resolution,
                final_iterations,
                label_seed,
            );
        }
        assign_approx_polish_ranks(&mut rows);
    }

    let mut p1_order = (0..rows.len()).collect::<Vec<_>>();
    p1_order.sort_by(|&left, &right| {
        rows[right]
            .p1_delta_q
            .total_cmp(&rows[left].p1_delta_q)
            .then_with(|| rows[left].candidate_index.cmp(&rows[right].candidate_index))
    });
    for (rank, &idx) in p1_order.iter().enumerate() {
        if rank == 0 {
            rows[idx].selected_by_p1_top1 = true;
        }
        if rank < 2 {
            rows[idx].selected_by_p1_top2 = true;
        }
    }

    let final_count = finalists.min(rows.len());
    let all_indices = (0..rows.len()).collect::<Vec<_>>();
    let top1_indices = if rows.is_empty() { Vec::new() } else { vec![0] };
    let full_top_indices = all_indices.clone();
    let p1_top1_indices = p1_order.iter().copied().take(1).collect::<Vec<_>>();
    let p1_top2_indices = p1_order.iter().copied().take(2).collect::<Vec<_>>();
    let p1_selected_indices = p1_order
        .iter()
        .copied()
        .take(final_count)
        .collect::<Vec<_>>();
    let mut p5_eval_indices = if label_full_p5 {
        (0..rows.len()).collect::<Vec<_>>()
    } else {
        p1_order
            .iter()
            .copied()
            .take(final_count)
            .collect::<Vec<_>>()
    };
    p5_eval_indices.sort_unstable();
    p5_eval_indices.dedup();
    let basin_sketch_nodes = if basin_signatures {
        basin_signature_sketch_nodes(
            graph,
            baseline_clustering,
            &candidates,
            BASIN_SIGNATURE_SKETCH_SAMPLE_SIZE,
        )
    } else {
        Vec::new()
    };
    let mut selected_membership: Option<(usize, Vec<u32>)> = None;
    for idx in p5_eval_indices {
        let candidate_index = rows[idx].candidate_index as usize;
        let final_seed = seed.wrapping_add(candidate_index as u64);
        let Some(p5) = polish_external_grain_candidate(
            graph,
            baseline_clustering,
            &candidates[candidate_index],
            &cpm,
            baseline_quality,
            resolution,
            final_iterations,
            randomness,
            final_seed,
            return_membership || basin_signatures,
        ) else {
            continue;
        };
        rows[idx].p5_quality = p5.post_polish_quality;
        rows[idx].p5_delta_q = p5.post_polish_quality - baseline_quality;
        rows[idx].p5_elapsed_ms = p5.elapsed_ms;
        if basin_signatures {
            if let Some(membership) = p5.membership.as_ref() {
                rows[idx].basin = compute_basin_signature_stats(
                    &baseline_clustering.clusters,
                    membership,
                    rows[idx].p5_delta_q,
                    baseline_quality,
                    &basin_sketch_nodes,
                );
            }
        }
        if approx_polish_labels && rows[idx].approx.ub_delta_q.is_finite() {
            rows[idx].approx.ub_violation =
                (rows[idx].p5_delta_q - rows[idx].approx.ub_delta_q).max(0.0);
            rows[idx].approx.ub_covers_p5 = rows[idx].approx.ub_violation <= 0.0;
        }
        if return_membership && p1_selected_indices.contains(&idx) {
            let replace = selected_membership
                .as_ref()
                .map(|(current_idx, _)| p5_candidate_is_better(&rows, idx, *current_idx))
                .unwrap_or(true);
            if replace {
                selected_membership = p5.membership.map(|membership| (idx, membership));
            }
        }
    }

    let full_p5_winner = best_p5_index(&rows, &all_indices);
    if label_full_p5 {
        if let Some(idx) = full_p5_winner {
            rows[idx].selected_by_full_p5 = true;
        }
    }

    let mut policy_rows = vec![
        build_policy_row(
            "top1_p5".to_string(),
            &rows,
            &top1_indices,
            0,
            true,
            baseline_quality,
            quality_eps,
            full_p5_winner,
        ),
        build_policy_row(
            "full_top3_p5".to_string(),
            &rows,
            &full_top_indices,
            0,
            true,
            baseline_quality,
            quality_eps,
            full_p5_winner,
        ),
        build_policy_row(
            "p1_top1_then_p5".to_string(),
            &rows,
            &p1_top1_indices,
            rows.len() as u64,
            true,
            baseline_quality,
            quality_eps,
            full_p5_winner,
        ),
        build_policy_row(
            "p1_top2_then_p5".to_string(),
            &rows,
            &p1_top2_indices,
            rows.len() as u64,
            true,
            baseline_quality,
            quality_eps,
            full_p5_winner,
        ),
    ];
    let selected_policy = if final_count == 0 {
        "none".to_string()
    } else if final_count == 1 {
        "p1_top1_then_p5".to_string()
    } else if final_count == 2 {
        "p1_top2_then_p5".to_string()
    } else {
        format!("p1_top{}_then_p5", final_count)
    };
    if final_count > 2 {
        policy_rows.push(build_policy_row(
            selected_policy.clone(),
            &rows,
            &p1_selected_indices,
            rows.len() as u64,
            true,
            baseline_quality,
            quality_eps,
            full_p5_winner,
        ));
    }

    let selected_policy_row = policy_rows
        .iter()
        .find(|row| row.policy == selected_policy)
        .cloned();
    let selected_candidate_index = selected_policy_row
        .as_ref()
        .map(|row| row.selected_candidate_index)
        .unwrap_or(-1);
    let accepted = selected_policy_row
        .as_ref()
        .map(|row| row.accepted)
        .unwrap_or(false);
    let mut quality = baseline_quality;
    let mut membership = if return_membership {
        clusters_to_u64(&baseline_clustering.clusters)
    } else {
        Vec::new()
    };
    if accepted {
        if let Some(row) = selected_policy_row.as_ref() {
            quality = row.quality;
            if return_membership {
                if let Some((idx, selected_membership)) = selected_membership {
                    if rows
                        .get(idx)
                        .map(|candidate_row| {
                            candidate_row.candidate_index as i64 == row.selected_candidate_index
                        })
                        .unwrap_or(false)
                    {
                        membership = clusters_to_u64(&selected_membership);
                    }
                }
            }
        }
    }
    let p5_delta_values = rows
        .iter()
        .filter(|row| row.p5_delta_q.is_finite())
        .map(|row| row.p5_delta_q)
        .collect::<Vec<_>>();
    let best_delta_q = if p5_delta_values.is_empty() {
        0.0
    } else {
        p5_delta_values
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
    };

    Ok(NonMonotoneGroupEscapeMultifidelityComputation {
        membership,
        quality,
        accepted,
        selected_policy,
        selected_candidate_index,
        baseline_quality,
        best_delta_q,
        elapsed_ms: total_start.elapsed().as_secs_f64() * 1000.0,
        candidate_rows: rows,
        policy_rows,
    })
}

#[cfg(feature = "python")]
fn non_monotone_candidate_rows_to_py(
    py: Python<'_>,
    rows: &[NonMonotoneGroupEscapeCandidateRow],
) -> Vec<std::collections::HashMap<String, pyo3::PyObject>> {
    rows.iter()
        .map(|row| {
            let mut d = std::collections::HashMap::new();
            d.insert(
                "candidate_index".to_string(),
                row.candidate_index
                    .into_pyobject(py)
                    .unwrap()
                    .into_any()
                    .unbind(),
            );
            d.insert(
                "source_cluster".to_string(),
                row.source_cluster
                    .into_pyobject(py)
                    .unwrap()
                    .into_any()
                    .unbind(),
            );
            d.insert(
                "target_cluster".to_string(),
                row.target_cluster
                    .into_pyobject(py)
                    .unwrap()
                    .into_any()
                    .unbind(),
            );
            d.insert(
                "group_kind".to_string(),
                row.group_kind
                    .into_pyobject(py)
                    .unwrap()
                    .into_any()
                    .unbind(),
            );
            d.insert(
                "block_count".to_string(),
                row.block_count
                    .into_pyobject(py)
                    .unwrap()
                    .into_any()
                    .unbind(),
            );
            d.insert(
                "doc_weight".to_string(),
                row.doc_weight
                    .into_pyobject(py)
                    .unwrap()
                    .into_any()
                    .unbind(),
            );
            d.insert(
                "incident_directed_edges".to_string(),
                row.incident_directed_edges
                    .into_pyobject(py)
                    .unwrap()
                    .into_any()
                    .unbind(),
            );
            d.insert(
                "assigned_fraction".to_string(),
                row.assigned_fraction
                    .into_pyobject(py)
                    .unwrap()
                    .into_any()
                    .unbind(),
            );
            d.insert(
                "group_count".to_string(),
                row.group_count
                    .into_pyobject(py)
                    .unwrap()
                    .into_any()
                    .unbind(),
            );
            d.insert(
                "group_weight".to_string(),
                row.group_weight
                    .into_pyobject(py)
                    .unwrap()
                    .into_any()
                    .unbind(),
            );
            d.insert(
                "group_fraction".to_string(),
                row.group_fraction
                    .into_pyobject(py)
                    .unwrap()
                    .into_any()
                    .unbind(),
            );
            d.insert(
                "group_to_target_weight".to_string(),
                row.group_to_target_weight
                    .into_pyobject(py)
                    .unwrap()
                    .into_any()
                    .unbind(),
            );
            d.insert(
                "group_cut_weight".to_string(),
                row.group_cut_weight
                    .into_pyobject(py)
                    .unwrap()
                    .into_any()
                    .unbind(),
            );
            d.insert(
                "group_move_delta_q".to_string(),
                row.group_move_delta_q
                    .into_pyobject(py)
                    .unwrap()
                    .into_any()
                    .unbind(),
            );
            d.insert(
                "group_split_delta_q".to_string(),
                row.group_split_delta_q
                    .into_pyobject(py)
                    .unwrap()
                    .into_any()
                    .unbind(),
            );
            d.insert(
                "group_delta_q".to_string(),
                row.group_delta_q
                    .into_pyobject(py)
                    .unwrap()
                    .into_any()
                    .unbind(),
            );
            d.insert(
                "best_group_delta_q".to_string(),
                row.best_group_delta_q
                    .into_pyobject(py)
                    .unwrap()
                    .into_any()
                    .unbind(),
            );
            d.insert(
                "best_group_action".to_string(),
                row.best_group_action
                    .into_pyobject(py)
                    .unwrap()
                    .into_any()
                    .unbind(),
            );
            d.insert(
                "recommended_for_split_repair".to_string(),
                pyo3::types::PyBool::new(py, row.recommended_for_split_repair)
                    .to_owned()
                    .into_any()
                    .unbind(),
            );
            d.insert(
                "priority".to_string(),
                row.priority.into_pyobject(py).unwrap().into_any().unbind(),
            );
            d.insert(
                "pre_polish_quality".to_string(),
                row.pre_polish_quality
                    .into_pyobject(py)
                    .unwrap()
                    .into_any()
                    .unbind(),
            );
            d.insert(
                "pre_polish_delta_q".to_string(),
                row.pre_polish_delta_q
                    .into_pyobject(py)
                    .unwrap()
                    .into_any()
                    .unbind(),
            );
            d.insert(
                "post_polish_quality".to_string(),
                row.post_polish_quality
                    .into_pyobject(py)
                    .unwrap()
                    .into_any()
                    .unbind(),
            );
            d.insert(
                "post_polish_delta_q".to_string(),
                row.post_polish_delta_q
                    .into_pyobject(py)
                    .unwrap()
                    .into_any()
                    .unbind(),
            );
            d.insert(
                "accepted_by_quality".to_string(),
                pyo3::types::PyBool::new(py, row.accepted_by_quality)
                    .to_owned()
                    .into_any()
                    .unbind(),
            );
            d.insert(
                "elapsed_ms".to_string(),
                row.elapsed_ms
                    .into_pyobject(py)
                    .unwrap()
                    .into_any()
                    .unbind(),
            );
            d
        })
        .collect()
}

#[cfg(feature = "python")]
fn non_monotone_multifidelity_candidate_rows_to_py(
    py: Python<'_>,
    rows: &[NonMonotoneGroupEscapeMultifidelityCandidateRow],
) -> Vec<std::collections::HashMap<String, pyo3::PyObject>> {
    rows.iter()
        .map(|row| {
            macro_rules! insert {
                ($dict:ident, $key:literal, $value:expr) => {
                    $dict.insert(
                        $key.to_string(),
                        ($value)
                            .into_pyobject(py)
                            .unwrap()
                            .clone()
                            .into_any()
                            .unbind(),
                    );
                };
            }
            let mut d = std::collections::HashMap::new();
            insert!(d, "candidate_index", row.candidate_index);
            insert!(d, "source_cluster", row.source_cluster);
            insert!(d, "target_cluster", row.target_cluster);
            insert!(d, "group_kind", row.group_kind);
            insert!(d, "block_count", row.block_count);
            insert!(d, "doc_weight", row.doc_weight);
            insert!(d, "incident_directed_edges", row.incident_directed_edges);
            insert!(d, "assigned_fraction", row.assigned_fraction);
            insert!(d, "group_count", row.group_count);
            insert!(d, "group_weight", row.group_weight);
            insert!(d, "group_fraction", row.group_fraction);
            insert!(d, "group_to_target_weight", row.group_to_target_weight);
            insert!(d, "group_cut_weight", row.group_cut_weight);
            insert!(d, "group_move_delta_q", row.group_move_delta_q);
            insert!(d, "group_split_delta_q", row.group_split_delta_q);
            insert!(d, "group_delta_q", row.group_delta_q);
            insert!(d, "best_group_delta_q", row.best_group_delta_q);
            insert!(d, "best_group_action", row.best_group_action);
            d.insert(
                "recommended_for_split_repair".to_string(),
                pyo3::types::PyBool::new(py, row.recommended_for_split_repair)
                    .to_owned()
                    .into_any()
                    .unbind(),
            );
            insert!(d, "priority", row.priority);
            insert!(d, "pre_polish_quality", row.pre_polish_quality);
            insert!(d, "pre_delta_q", row.pre_delta_q);
            insert!(d, "p1_quality", row.p1_quality);
            insert!(d, "p1_delta_q", row.p1_delta_q);
            insert!(d, "p1_elapsed_ms", row.p1_elapsed_ms);
            insert!(d, "p5_quality", row.p5_quality);
            insert!(d, "p5_delta_q", row.p5_delta_q);
            insert!(d, "p5_elapsed_ms", row.p5_elapsed_ms);
            insert!(d, "p5_basin_signature", row.basin.signature.as_str());
            insert!(d, "p5_basin_cluster_count", row.basin.cluster_count);
            insert!(
                d,
                "p5_changed_nodes_vs_baseline",
                row.basin.changed_nodes_vs_baseline
            );
            insert!(
                d,
                "p5_baseline_fragmentation_nodes",
                row.basin.baseline_fragmentation_nodes
            );
            insert!(
                d,
                "p5_baseline_mixing_nodes",
                row.basin.baseline_mixing_nodes
            );
            insert!(
                d,
                "p5_changed_fraction_vs_baseline",
                row.basin.changed_fraction_vs_baseline
            );
            insert!(d, "p5_relative_delta_q_ppm", row.basin.relative_delta_q_ppm);
            insert!(
                d,
                "p5_basin_sketch_sample_size",
                row.basin.sketch_sample_size
            );
            insert!(
                d,
                "p5_basin_sketch_node_hash",
                row.basin.sketch_node_hash.as_str()
            );
            insert!(
                d,
                "p5_basin_sketch_baseline_membership",
                row.basin.sketch_baseline_membership.as_str()
            );
            insert!(
                d,
                "p5_basin_sketch_membership",
                row.basin.sketch_membership.as_str()
            );
            insert!(
                d,
                "p5_basin_changed_support_node_count",
                row.basin.changed_support_node_count
            );
            insert!(
                d,
                "p5_basin_changed_support_sketch_sample_size",
                row.basin.changed_support_sketch_sample_size
            );
            insert!(
                d,
                "p5_basin_changed_support_node_hash",
                row.basin.changed_support_node_hash.as_str()
            );
            insert!(
                d,
                "p5_basin_changed_support_nodes",
                row.basin.changed_support_nodes.as_str()
            );
            insert!(d, "localized_quality", row.approx.localized_quality);
            insert!(d, "localized_delta_q", row.approx.localized_delta_q);
            insert!(d, "localized_elapsed_ms", row.approx.localized_elapsed_ms);
            insert!(
                d,
                "localized_active_nodes",
                row.approx.localized_active_nodes
            );
            insert!(
                d,
                "localized_active_clusters",
                row.approx.localized_active_clusters
            );
            insert!(d, "localized_rank", row.approx.localized_rank);
            insert!(d, "quotient_quality", row.approx.quotient_quality);
            insert!(d, "quotient_delta_q", row.approx.quotient_delta_q);
            insert!(d, "quotient_elapsed_ms", row.approx.quotient_elapsed_ms);
            insert!(d, "quotient_supernodes", row.approx.quotient_supernodes);
            insert!(d, "quotient_rank", row.approx.quotient_rank);
            insert!(d, "ub_delta_q", row.approx.ub_delta_q);
            insert!(d, "ub_elapsed_ms", row.approx.ub_elapsed_ms);
            d.insert(
                "ub_covers_p5".to_string(),
                pyo3::types::PyBool::new(py, row.approx.ub_covers_p5)
                    .to_owned()
                    .into_any()
                    .unbind(),
            );
            insert!(d, "ub_violation", row.approx.ub_violation);
            insert!(d, "ub_rank", row.approx.ub_rank);
            d.insert(
                "selected_by_p1_top1".to_string(),
                pyo3::types::PyBool::new(py, row.selected_by_p1_top1)
                    .to_owned()
                    .into_any()
                    .unbind(),
            );
            d.insert(
                "selected_by_p1_top2".to_string(),
                pyo3::types::PyBool::new(py, row.selected_by_p1_top2)
                    .to_owned()
                    .into_any()
                    .unbind(),
            );
            d.insert(
                "selected_by_full_p5".to_string(),
                pyo3::types::PyBool::new(py, row.selected_by_full_p5)
                    .to_owned()
                    .into_any()
                    .unbind(),
            );
            d
        })
        .collect()
}

#[cfg(feature = "python")]
fn non_monotone_multifidelity_policy_rows_to_py(
    py: Python<'_>,
    rows: &[NonMonotoneGroupEscapePolicyRow],
) -> Vec<std::collections::HashMap<String, pyo3::PyObject>> {
    rows.iter()
        .map(|row| {
            macro_rules! insert {
                ($dict:ident, $key:literal, $value:expr) => {
                    $dict.insert(
                        $key.to_string(),
                        ($value)
                            .into_pyobject(py)
                            .unwrap()
                            .clone()
                            .into_any()
                            .unbind(),
                    );
                };
            }
            let mut d = std::collections::HashMap::new();
            insert!(d, "policy", row.policy.clone());
            insert!(d, "selected_candidate_index", row.selected_candidate_index);
            insert!(d, "candidate_count", row.candidate_count);
            insert!(d, "p1_evaluated", row.p1_evaluated);
            insert!(d, "p5_evaluated", row.p5_evaluated);
            insert!(d, "p1_elapsed_ms", row.p1_elapsed_ms);
            insert!(d, "p5_elapsed_ms", row.p5_elapsed_ms);
            insert!(d, "total_elapsed_ms", row.total_elapsed_ms);
            insert!(d, "final_delta_q", row.final_delta_q);
            insert!(d, "quality", row.quality);
            d.insert(
                "accepted".to_string(),
                pyo3::types::PyBool::new(py, row.accepted)
                    .to_owned()
                    .into_any()
                    .unbind(),
            );
            d.insert(
                "available".to_string(),
                pyo3::types::PyBool::new(py, row.available)
                    .to_owned()
                    .into_any()
                    .unbind(),
            );
            d.insert(
                "matches_full_p5".to_string(),
                pyo3::types::PyBool::new(py, row.matches_full_p5)
                    .to_owned()
                    .into_any()
                    .unbind(),
            );
            d
        })
        .collect()
}

#[cfg(feature = "python")]
fn non_monotone_group_escape_result_to_py(
    py: Python<'_>,
    computed: NonMonotoneGroupEscapeComputation,
) -> std::collections::HashMap<String, pyo3::PyObject> {
    let mut out = std::collections::HashMap::new();
    out.insert(
        "membership".to_string(),
        PyArray1::from_vec(py, computed.membership)
            .into_any()
            .unbind(),
    );
    out.insert(
        "quality".to_string(),
        computed
            .quality
            .into_pyobject(py)
            .unwrap()
            .into_any()
            .unbind(),
    );
    out.insert(
        "accepted".to_string(),
        pyo3::types::PyBool::new(py, computed.accepted)
            .to_owned()
            .into_any()
            .unbind(),
    );
    out.insert(
        "candidate_rows".to_string(),
        non_monotone_candidate_rows_to_py(py, &computed.candidate_rows)
            .into_pyobject(py)
            .unwrap()
            .into_any()
            .unbind(),
    );
    out.insert(
        "baseline_quality".to_string(),
        computed
            .baseline_quality
            .into_pyobject(py)
            .unwrap()
            .into_any()
            .unbind(),
    );
    out.insert(
        "best_delta_q".to_string(),
        computed
            .best_delta_q
            .into_pyobject(py)
            .unwrap()
            .into_any()
            .unbind(),
    );
    out.insert(
        "elapsed_ms".to_string(),
        computed
            .elapsed_ms
            .into_pyobject(py)
            .unwrap()
            .into_any()
            .unbind(),
    );
    out.insert(
        "candidate_eval_parallel".to_string(),
        pyo3::types::PyBool::new(py, computed.candidate_eval_parallel)
            .to_owned()
            .into_any()
            .unbind(),
    );
    out.insert(
        "candidate_eval_wall_elapsed_ms".to_string(),
        computed
            .candidate_eval_wall_elapsed_ms
            .into_pyobject(py)
            .unwrap()
            .into_any()
            .unbind(),
    );
    out.insert(
        "candidate_eval_cpu_sum_elapsed_ms".to_string(),
        computed
            .candidate_eval_cpu_sum_elapsed_ms
            .into_pyobject(py)
            .unwrap()
            .into_any()
            .unbind(),
    );
    out.insert(
        "candidate_eval_parallel_speedup".to_string(),
        computed
            .candidate_eval_parallel_speedup
            .into_pyobject(py)
            .unwrap()
            .into_any()
            .unbind(),
    );
    out.insert(
        "candidate_eval_parallel_workers".to_string(),
        computed
            .candidate_eval_parallel_workers
            .into_pyobject(py)
            .unwrap()
            .into_any()
            .unbind(),
    );
    out
}

#[cfg(feature = "python")]
fn non_monotone_multifidelity_result_to_py(
    py: Python<'_>,
    computed: NonMonotoneGroupEscapeMultifidelityComputation,
) -> std::collections::HashMap<String, pyo3::PyObject> {
    let mut out = std::collections::HashMap::new();
    out.insert(
        "membership".to_string(),
        PyArray1::from_vec(py, computed.membership)
            .into_any()
            .unbind(),
    );
    out.insert(
        "quality".to_string(),
        computed
            .quality
            .into_pyobject(py)
            .unwrap()
            .into_any()
            .unbind(),
    );
    out.insert(
        "accepted".to_string(),
        pyo3::types::PyBool::new(py, computed.accepted)
            .to_owned()
            .into_any()
            .unbind(),
    );
    out.insert(
        "selected_policy".to_string(),
        computed
            .selected_policy
            .into_pyobject(py)
            .unwrap()
            .into_any()
            .unbind(),
    );
    out.insert(
        "selected_candidate_index".to_string(),
        computed
            .selected_candidate_index
            .into_pyobject(py)
            .unwrap()
            .into_any()
            .unbind(),
    );
    out.insert(
        "candidate_rows".to_string(),
        non_monotone_multifidelity_candidate_rows_to_py(py, &computed.candidate_rows)
            .into_pyobject(py)
            .unwrap()
            .into_any()
            .unbind(),
    );
    out.insert(
        "policy_rows".to_string(),
        non_monotone_multifidelity_policy_rows_to_py(py, &computed.policy_rows)
            .into_pyobject(py)
            .unwrap()
            .into_any()
            .unbind(),
    );
    out.insert(
        "baseline_quality".to_string(),
        computed
            .baseline_quality
            .into_pyobject(py)
            .unwrap()
            .into_any()
            .unbind(),
    );
    out.insert(
        "best_delta_q".to_string(),
        computed
            .best_delta_q
            .into_pyobject(py)
            .unwrap()
            .into_any()
            .unbind(),
    );
    out.insert(
        "elapsed_ms".to_string(),
        computed
            .elapsed_ms
            .into_pyobject(py)
            .unwrap()
            .into_any()
            .unbind(),
    );
    out
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

#[cfg(feature = "python")]
fn dongdaemun_status_str(status: DongdaemunStatus) -> &'static str {
    match status {
        DongdaemunStatus::NoCurrentOversizeCandidates => "no_current_oversize_candidates",
        DongdaemunStatus::Committed => "committed",
        DongdaemunStatus::NoSelectedCandidates => "no_selected_candidates",
        DongdaemunStatus::NoProgress => "no_progress",
        DongdaemunStatus::SplitQualityBelowFloor => "split_quality_below_floor",
        DongdaemunStatus::TrimQualityBelowFloor => "trim_quality_below_floor",
        DongdaemunStatus::QualityBelowFloor => "quality_below_floor",
        DongdaemunStatus::HardCapNotSatisfied => "hard_cap_not_satisfied",
    }
}

#[cfg(feature = "python")]
fn dongdaemun_status_code(status: DongdaemunStatus) -> u8 {
    match status {
        DongdaemunStatus::NoCurrentOversizeCandidates => 0,
        DongdaemunStatus::Committed => 1,
        DongdaemunStatus::NoSelectedCandidates => 2,
        DongdaemunStatus::NoProgress => 3,
        DongdaemunStatus::SplitQualityBelowFloor => 4,
        DongdaemunStatus::TrimQualityBelowFloor => 5,
        DongdaemunStatus::QualityBelowFloor => 6,
        DongdaemunStatus::HardCapNotSatisfied => 7,
    }
}

#[cfg(feature = "python")]
fn parse_dongdaemun_policy(policy: &str) -> PyResult<DongdaemunPolicy> {
    match policy {
        "quality_first" => Ok(DongdaemunPolicy::QualityFirst),
        "hard_cap" => Ok(DongdaemunPolicy::HardCap),
        other => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "unknown Dongdaemun policy: {other:?}"
        ))),
    }
}

#[cfg(feature = "python")]
fn parse_candidate_quality_policy(policy: &str) -> PyResult<CandidateQualityPolicy> {
    match policy {
        "structural" => Ok(CandidateQualityPolicy::Structural),
        "quality_guarded_structural" => Ok(CandidateQualityPolicy::QualityGuardedStructural),
        "quality_floor" => Ok(CandidateQualityPolicy::QualityFloor),
        "quality_first" => Ok(CandidateQualityPolicy::QualityFirst),
        "selective" => Ok(CandidateQualityPolicy::Selective),
        "pressure_aware" => Ok(CandidateQualityPolicy::PressureAware),
        "adaptive_plateau" => Ok(CandidateQualityPolicy::AdaptivePlateau),
        other => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "unknown candidate_quality_policy: {other:?}"
        ))),
    }
}

#[cfg(feature = "python")]
fn parse_baseline_repair_policy(policy: &str) -> PyResult<BaselineRepairPolicy> {
    match policy {
        "replace" => Ok(BaselineRepairPolicy::Replace),
        "augment" => Ok(BaselineRepairPolicy::Augment),
        "adaptive" => Ok(BaselineRepairPolicy::Adaptive),
        other => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "unknown baseline_repair_policy: {other:?}"
        ))),
    }
}

#[cfg(feature = "python")]
fn parse_parent_selection_policy(policy: &str) -> PyResult<ParentSelectionPolicy> {
    match policy {
        "weight" => Ok(ParentSelectionPolicy::Weight),
        "pressure_boundary" => Ok(ParentSelectionPolicy::PressureBoundary),
        other => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "unknown parent_selection_policy: {other:?}"
        ))),
    }
}

#[cfg(feature = "python")]
fn parse_adaptive_probe_mode(mode: &str) -> PyResult<AdaptiveProbeMode> {
    match mode {
        "off" => Ok(AdaptiveProbeMode::Off),
        "trace_only" => Ok(AdaptiveProbeMode::TraceOnly),
        "apply_if_win" => Ok(AdaptiveProbeMode::ApplyIfWin),
        "conservative_apply" => Ok(AdaptiveProbeMode::ConservativeApply),
        other => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "unknown adaptive_probe_mode: {other:?}"
        ))),
    }
}

#[cfg(feature = "python")]
fn parse_adaptive_probe_commit_strategy(strategy: &str) -> PyResult<AdaptiveProbeCommitStrategy> {
    match strategy {
        "online_first" => Ok(AdaptiveProbeCommitStrategy::OnlineFirst),
        "best_qf" => Ok(AdaptiveProbeCommitStrategy::BestQf),
        "risk_adjusted" => Ok(AdaptiveProbeCommitStrategy::RiskAdjusted),
        other => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "unknown adaptive_probe_commit_strategy: {other:?}"
        ))),
    }
}

#[cfg(feature = "python")]
fn parse_adaptive_near_tie_probe_mode(mode: &str) -> PyResult<AdaptiveNearTieProbeMode> {
    match mode {
        "off" => Ok(AdaptiveNearTieProbeMode::Off),
        "trace_only" => Ok(AdaptiveNearTieProbeMode::TraceOnly),
        "candidate" => Ok(AdaptiveNearTieProbeMode::Candidate),
        "qf_replace" | "near_tie_qf_replace" => Ok(AdaptiveNearTieProbeMode::QfReplace),
        other => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "unknown adaptive_near_tie_probe_mode: {other:?}"
        ))),
    }
}

#[cfg(feature = "python")]
fn parse_adaptive_local_shake_mode(mode: &str) -> PyResult<AdaptiveLocalShakeMode> {
    match mode {
        "off" => Ok(AdaptiveLocalShakeMode::Off),
        "trace_only" => Ok(AdaptiveLocalShakeMode::TraceOnly),
        "qf_replace" => Ok(AdaptiveLocalShakeMode::QfReplace),
        "pressure_guarded" => Ok(AdaptiveLocalShakeMode::PressureGuarded),
        other => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "unknown adaptive_local_shake_mode: {other:?}"
        ))),
    }
}

#[cfg(feature = "python")]
fn parse_adaptive_local_shake_arm(arm: &str) -> PyResult<AdaptiveLocalShakeArm> {
    match arm {
        "near_tie_refinement" => Ok(AdaptiveLocalShakeArm::NearTieRefinement),
        "resolution_up" => Ok(AdaptiveLocalShakeArm::ResolutionUp),
        "resolution_down" => Ok(AdaptiveLocalShakeArm::ResolutionDown),
        "seed_local_refinement" => Ok(AdaptiveLocalShakeArm::SeedLocalRefinement),
        other => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "unknown adaptive_local_shake_arm: {other:?}"
        ))),
    }
}

#[cfg(feature = "python")]
fn parse_adaptive_local_shake_arms(
    arms: Option<Vec<String>>,
) -> PyResult<Vec<AdaptiveLocalShakeArm>> {
    arms.unwrap_or_default()
        .iter()
        .map(|arm| parse_adaptive_local_shake_arm(arm))
        .collect()
}

#[cfg(feature = "python")]
fn parse_adaptive_local_shake_final_guard_mode(
    mode: &str,
) -> PyResult<AdaptiveLocalShakeFinalGuardMode> {
    match mode {
        "none" => Ok(AdaptiveLocalShakeFinalGuardMode::None),
        "runner_audit" => Ok(AdaptiveLocalShakeFinalGuardMode::RunnerAudit),
        "quality_guard" => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "adaptive_local_shake_final_guard_mode='quality_guard' is not implemented in v1; use 'none' or 'runner_audit'",
        )),
        other => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "unknown adaptive_local_shake_final_guard_mode: {other:?}"
        ))),
    }
}

#[cfg(feature = "python")]
fn parse_adaptive_probe_target(value: &str) -> PyResult<AdaptiveProbeTarget> {
    let parts = value.split(':').map(str::trim).collect::<Vec<_>>();
    if parts.len() != 3 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "adaptive_probe_targets must use depth:parent_id:parent_visit_index, got {value:?}"
        )));
    }
    let parse_part = |idx: usize, name: &str| -> PyResult<usize> {
        parts[idx].parse::<usize>().map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "invalid {name} in adaptive_probe_target {value:?}"
            ))
        })
    };
    Ok(AdaptiveProbeTarget {
        depth: parse_part(0, "depth")?,
        parent_id: parse_part(1, "parent_id")?,
        parent_visit_index: parse_part(2, "parent_visit_index")?,
    })
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
    fn run_leiden_u32<'py>(
        &self,
        py: Python<'py>,
        resolution: f64,
        n_iterations: usize,
        n_starts: usize,
        randomness: f64,
        randomness_schedule: Option<Vec<f64>>,
        seed: u64,
        initial_membership: Option<PyReadonlyArray1<u32>>,
        fixed_nodes: Option<PyReadonlyArray1<bool>>,
    ) -> PyResult<(Py<PyArray1<u32>>, f64, usize)> {
        let config = LeidenConfig {
            resolution,
            n_iterations,
            randomness,
            randomness_schedule: randomness_schedule.unwrap_or_default(),
            seed,
        };

        let initial = if let Some(mem) = initial_membership {
            let mut clustering = Clustering::from_assignments(mem.as_slice()?.to_vec());
            ensure_membership_len(&clustering, self.graph.n_nodes)?;
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

        let membership = result.clustering.clusters;
        let n_clusters = result.clustering.n_clusters;
        let quality = result.quality;

        Ok((
            PyArray1::from_vec(py, membership).into(),
            quality,
            n_clusters,
        ))
    }

    #[pyo3(signature = (
        target_max_weight,
        resolution = 1.0,
        n_iterations = 10,
        randomness = 0.01,
        randomness_schedule = None,
        seed = 0,
        initial_membership = None,
        fixed_nodes = None,
        soft_min_ratio = 1.0,
        max_extra_parents_per_iteration = 16,
        max_extra_children_per_parent = 64,
        parent_selection_policy = "weight",
        max_singleton_weight_fraction = 0.05,
        min_largest_child_fraction_improvement = 0.05,
        gamma_multipliers = None,
        seed_perturbations = 0,
        use_quotient_diagnostic = false,
        use_baseline_repair = false,
        baseline_repair_policy = "replace",
        baseline_repair_replace_min_parent_ratio = 1.05,
        baseline_repair_epsilon = 0.0,
        candidate_quality_policy = "structural",
        min_candidate_delta_q = 0.0,
        adaptive_plateau_quality_band = 0.0,
        use_final_quality_guard = false,
        min_final_quality_delta = 0.0,
        adaptive_probe_mode = "off",
        adaptive_probe_perturbations = 0,
        adaptive_probe_targets = None,
        adaptive_probe_tolerance_parent_weight = 1e-6,
        adaptive_probe_include_node_order_control = false,
        adaptive_probe_commit_min_gain_parent_weight = 0.0,
        adaptive_probe_max_commits_total = 0,
        adaptive_probe_max_commits_per_depth = 0,
        adaptive_probe_commit_sources = None,
        adaptive_probe_commit_strategy = "online_first",
        adaptive_near_tie_probe_mode = "off",
        adaptive_near_tie_margin_parent_weight = 0.0,
        adaptive_near_tie_randomness = 0.0,
        adaptive_near_tie_max_decisions_per_parent = 0,
        adaptive_local_shake_mode = "off",
        adaptive_local_shake_arms = None,
        adaptive_local_shake_max_arms_per_parent = 0,
        adaptive_local_shake_max_candidates_per_parent = 0,
        adaptive_local_shake_min_gain_parent_weight = 0.0,
        adaptive_local_shake_shape_eps = 1e-12,
        adaptive_local_shake_arm_priority = None,
        adaptive_local_shake_near_tie_min_count = 1,
        adaptive_local_shake_resolution_down_multipliers = None,
        adaptive_local_shake_resolution_up_multipliers = None,
        adaptive_local_shake_resolution_up_min_parent_ratio = 1.0,
        adaptive_local_shake_resolution_down_max_parent_ratio = 1.0,
        adaptive_local_shake_large_child_fraction = 0.95,
        adaptive_local_shake_singleton_fraction = 0.05,
        adaptive_local_shake_seed_perturbations = 0,
        adaptive_local_shake_seed_margin_count = 2,
        adaptive_local_shake_near_tie_margin_parent_weight = 0.0,
        adaptive_local_shake_near_tie_randomness = 0.0,
        adaptive_local_shake_final_guard_mode = "none",
    ))]
    fn run_leiden_dongdaemun_refinement<'py>(
        &self,
        py: Python<'py>,
        target_max_weight: f64,
        resolution: f64,
        n_iterations: usize,
        randomness: f64,
        randomness_schedule: Option<Vec<f64>>,
        seed: u64,
        initial_membership: Option<PyReadonlyArray1<u64>>,
        fixed_nodes: Option<PyReadonlyArray1<bool>>,
        soft_min_ratio: f64,
        max_extra_parents_per_iteration: usize,
        max_extra_children_per_parent: usize,
        parent_selection_policy: &str,
        max_singleton_weight_fraction: f64,
        min_largest_child_fraction_improvement: f64,
        gamma_multipliers: Option<Vec<f64>>,
        seed_perturbations: usize,
        use_quotient_diagnostic: bool,
        use_baseline_repair: bool,
        baseline_repair_policy: &str,
        baseline_repair_replace_min_parent_ratio: f64,
        baseline_repair_epsilon: f64,
        candidate_quality_policy: &str,
        min_candidate_delta_q: f64,
        adaptive_plateau_quality_band: f64,
        use_final_quality_guard: bool,
        min_final_quality_delta: f64,
        adaptive_probe_mode: &str,
        adaptive_probe_perturbations: usize,
        adaptive_probe_targets: Option<Vec<String>>,
        adaptive_probe_tolerance_parent_weight: f64,
        adaptive_probe_include_node_order_control: bool,
        adaptive_probe_commit_min_gain_parent_weight: f64,
        adaptive_probe_max_commits_total: usize,
        adaptive_probe_max_commits_per_depth: usize,
        adaptive_probe_commit_sources: Option<Vec<String>>,
        adaptive_probe_commit_strategy: &str,
        adaptive_near_tie_probe_mode: &str,
        adaptive_near_tie_margin_parent_weight: f64,
        adaptive_near_tie_randomness: f64,
        adaptive_near_tie_max_decisions_per_parent: usize,
        adaptive_local_shake_mode: &str,
        adaptive_local_shake_arms: Option<Vec<String>>,
        adaptive_local_shake_max_arms_per_parent: usize,
        adaptive_local_shake_max_candidates_per_parent: usize,
        adaptive_local_shake_min_gain_parent_weight: f64,
        adaptive_local_shake_shape_eps: f64,
        adaptive_local_shake_arm_priority: Option<Vec<String>>,
        adaptive_local_shake_near_tie_min_count: usize,
        adaptive_local_shake_resolution_down_multipliers: Option<Vec<f64>>,
        adaptive_local_shake_resolution_up_multipliers: Option<Vec<f64>>,
        adaptive_local_shake_resolution_up_min_parent_ratio: f64,
        adaptive_local_shake_resolution_down_max_parent_ratio: f64,
        adaptive_local_shake_large_child_fraction: f64,
        adaptive_local_shake_singleton_fraction: f64,
        adaptive_local_shake_seed_perturbations: usize,
        adaptive_local_shake_seed_margin_count: usize,
        adaptive_local_shake_near_tie_margin_parent_weight: f64,
        adaptive_local_shake_near_tie_randomness: f64,
        adaptive_local_shake_final_guard_mode: &str,
    ) -> PyResult<std::collections::HashMap<String, pyo3::PyObject>> {
        let config = LeidenConfig {
            resolution,
            n_iterations,
            randomness,
            randomness_schedule: randomness_schedule.unwrap_or_default(),
            seed,
        };
        let ddm_config = DongdaemunRefinementConfig {
            target_max_weight,
            soft_min_ratio,
            max_extra_parents_per_iteration,
            max_extra_children_per_parent,
            parent_selection_policy: parse_parent_selection_policy(parent_selection_policy)?,
            max_singleton_weight_fraction,
            min_largest_child_fraction_improvement,
            gamma_multipliers: gamma_multipliers
                .unwrap_or_else(|| DongdaemunRefinementConfig::default().gamma_multipliers),
            seed_perturbations,
            use_quotient_diagnostic,
            use_baseline_repair,
            baseline_repair_policy: parse_baseline_repair_policy(baseline_repair_policy)?,
            baseline_repair_replace_min_parent_ratio,
            baseline_repair_epsilon,
            candidate_quality_policy: parse_candidate_quality_policy(candidate_quality_policy)?,
            min_candidate_delta_q,
            adaptive_plateau_quality_band,
            use_final_quality_guard,
            min_final_quality_delta,
            adaptive_probe_mode: parse_adaptive_probe_mode(adaptive_probe_mode)?,
            adaptive_probe_perturbations,
            adaptive_probe_targets: adaptive_probe_targets
                .unwrap_or_default()
                .iter()
                .map(|value| parse_adaptive_probe_target(value))
                .collect::<PyResult<Vec<_>>>()?,
            adaptive_probe_tolerance_parent_weight,
            adaptive_probe_include_node_order_control,
            adaptive_probe_commit_min_gain_parent_weight,
            adaptive_probe_max_commits_total,
            adaptive_probe_max_commits_per_depth,
            adaptive_probe_commit_sources: adaptive_probe_commit_sources.unwrap_or_default(),
            adaptive_probe_commit_strategy: parse_adaptive_probe_commit_strategy(
                adaptive_probe_commit_strategy,
            )?,
            adaptive_near_tie_probe_mode: parse_adaptive_near_tie_probe_mode(
                adaptive_near_tie_probe_mode,
            )?,
            adaptive_near_tie_margin_parent_weight,
            adaptive_near_tie_randomness,
            adaptive_near_tie_max_decisions_per_parent,
            adaptive_local_shake_mode: parse_adaptive_local_shake_mode(adaptive_local_shake_mode)?,
            adaptive_local_shake_arms: parse_adaptive_local_shake_arms(adaptive_local_shake_arms)?,
            adaptive_local_shake_max_arms_per_parent,
            adaptive_local_shake_max_candidates_per_parent,
            adaptive_local_shake_min_gain_parent_weight,
            adaptive_local_shake_shape_eps,
            adaptive_local_shake_arm_priority: parse_adaptive_local_shake_arms(
                adaptive_local_shake_arm_priority,
            )?,
            adaptive_local_shake_near_tie_min_count,
            adaptive_local_shake_resolution_down_multipliers:
                adaptive_local_shake_resolution_down_multipliers.unwrap_or_default(),
            adaptive_local_shake_resolution_up_multipliers:
                adaptive_local_shake_resolution_up_multipliers.unwrap_or_default(),
            adaptive_local_shake_resolution_up_min_parent_ratio,
            adaptive_local_shake_resolution_down_max_parent_ratio,
            adaptive_local_shake_large_child_fraction,
            adaptive_local_shake_singleton_fraction,
            adaptive_local_shake_seed_perturbations,
            adaptive_local_shake_seed_margin_count,
            adaptive_local_shake_near_tie_margin_parent_weight,
            adaptive_local_shake_near_tie_randomness,
            adaptive_local_shake_final_guard_mode: parse_adaptive_local_shake_final_guard_mode(
                adaptive_local_shake_final_guard_mode,
            )?,
        };
        ddm_config
            .validate()
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

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
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            leiden_with_dongdaemun_refinement(graph, &config, &ddm_config, initial, &mut rng)
        });

        let membership: Vec<u64> = result
            .clustering
            .clusters
            .iter()
            .map(|&c| c as u64)
            .collect();
        let audit = result.audit;
        let mut iteration_depth = Vec::with_capacity(audit.iterations.len());
        let mut iteration_selected_parents = Vec::with_capacity(audit.iterations.len());
        let mut iteration_applied_parents = Vec::with_capacity(audit.iterations.len());
        let mut iteration_same_gamma_candidates = Vec::with_capacity(audit.iterations.len());
        let mut iteration_high_gamma_candidates = Vec::with_capacity(audit.iterations.len());
        let mut iteration_same_gamma_applied = Vec::with_capacity(audit.iterations.len());
        let mut iteration_high_gamma_applied = Vec::with_capacity(audit.iterations.len());
        let mut iteration_quotient_candidates = Vec::with_capacity(audit.iterations.len());
        let mut iteration_quotient_positive_candidates = Vec::with_capacity(audit.iterations.len());
        let mut iteration_quotient_selected = Vec::with_capacity(audit.iterations.len());
        let mut iteration_quotient_score_sum = Vec::with_capacity(audit.iterations.len());
        let mut iteration_baseline_repair_candidates = Vec::with_capacity(audit.iterations.len());
        let mut iteration_baseline_repair_improved_candidates =
            Vec::with_capacity(audit.iterations.len());
        let mut iteration_baseline_repair_selected = Vec::with_capacity(audit.iterations.len());
        let mut iteration_baseline_repair_merge_count = Vec::with_capacity(audit.iterations.len());
        let mut iteration_baseline_repair_delta_sum = Vec::with_capacity(audit.iterations.len());
        let mut iteration_candidate_quality_delta_sum = Vec::with_capacity(audit.iterations.len());
        let mut iteration_candidate_positive_quality_delta =
            Vec::with_capacity(audit.iterations.len());
        let mut iteration_candidate_selected_positive_quality_delta =
            Vec::with_capacity(audit.iterations.len());
        let mut iteration_candidate_rejected_by_quality =
            Vec::with_capacity(audit.iterations.len());
        let mut iteration_same_gamma_quality_delta_sum = Vec::with_capacity(audit.iterations.len());
        let mut iteration_high_gamma_quality_delta_sum = Vec::with_capacity(audit.iterations.len());
        let mut iteration_same_gamma_positive_quality_delta =
            Vec::with_capacity(audit.iterations.len());
        let mut iteration_high_gamma_positive_quality_delta =
            Vec::with_capacity(audit.iterations.len());
        let mut iteration_same_gamma_selected_positive_quality_delta =
            Vec::with_capacity(audit.iterations.len());
        let mut iteration_high_gamma_selected_positive_quality_delta =
            Vec::with_capacity(audit.iterations.len());
        let mut iteration_same_gamma_rejected_by_quality =
            Vec::with_capacity(audit.iterations.len());
        let mut iteration_high_gamma_rejected_by_quality =
            Vec::with_capacity(audit.iterations.len());
        let mut iteration_candidate_valid = Vec::with_capacity(audit.iterations.len());
        let mut iteration_candidate_invalid = Vec::with_capacity(audit.iterations.len());
        let mut iteration_candidate_rejected_by_policy = Vec::with_capacity(audit.iterations.len());
        let mut iteration_same_gamma_valid = Vec::with_capacity(audit.iterations.len());
        let mut iteration_high_gamma_valid = Vec::with_capacity(audit.iterations.len());
        let mut iteration_same_gamma_invalid = Vec::with_capacity(audit.iterations.len());
        let mut iteration_high_gamma_invalid = Vec::with_capacity(audit.iterations.len());
        let mut iteration_same_gamma_rejected_by_policy =
            Vec::with_capacity(audit.iterations.len());
        let mut iteration_high_gamma_rejected_by_policy =
            Vec::with_capacity(audit.iterations.len());
        let mut iteration_candidate_qpos_spos = Vec::with_capacity(audit.iterations.len());
        let mut iteration_candidate_qpos_sneg = Vec::with_capacity(audit.iterations.len());
        let mut iteration_candidate_qneg_spos = Vec::with_capacity(audit.iterations.len());
        let mut iteration_candidate_qneg_sneg = Vec::with_capacity(audit.iterations.len());
        let mut iteration_same_gamma_qpos_spos = Vec::with_capacity(audit.iterations.len());
        let mut iteration_same_gamma_qpos_sneg = Vec::with_capacity(audit.iterations.len());
        let mut iteration_same_gamma_qneg_spos = Vec::with_capacity(audit.iterations.len());
        let mut iteration_same_gamma_qneg_sneg = Vec::with_capacity(audit.iterations.len());
        let mut iteration_high_gamma_qpos_spos = Vec::with_capacity(audit.iterations.len());
        let mut iteration_high_gamma_qpos_sneg = Vec::with_capacity(audit.iterations.len());
        let mut iteration_high_gamma_qneg_spos = Vec::with_capacity(audit.iterations.len());
        let mut iteration_high_gamma_qneg_sneg = Vec::with_capacity(audit.iterations.len());
        let mut iteration_candidate_true_positive = Vec::with_capacity(audit.iterations.len());
        let mut iteration_candidate_false_positive = Vec::with_capacity(audit.iterations.len());
        let mut iteration_candidate_false_negative = Vec::with_capacity(audit.iterations.len());
        let mut iteration_candidate_true_negative = Vec::with_capacity(audit.iterations.len());
        let mut iteration_adaptive_local_shake_triggers =
            Vec::with_capacity(audit.iterations.len());
        let mut iteration_adaptive_local_shake_candidates =
            Vec::with_capacity(audit.iterations.len());
        let mut iteration_adaptive_local_shake_commits = Vec::with_capacity(audit.iterations.len());
        let mut iteration_adaptive_local_shake_qf_gain_sum =
            Vec::with_capacity(audit.iterations.len());
        let mut iteration_standard_refined_clusters = Vec::with_capacity(audit.iterations.len());
        let mut iteration_final_refined_clusters = Vec::with_capacity(audit.iterations.len());
        for row in &audit.iterations {
            iteration_depth.push(row.depth as u64);
            iteration_selected_parents.push(row.selected_parents as u64);
            iteration_applied_parents.push(row.applied_parents as u64);
            iteration_same_gamma_candidates.push(row.same_gamma_candidates as u64);
            iteration_high_gamma_candidates.push(row.high_gamma_candidates as u64);
            iteration_same_gamma_applied.push(row.same_gamma_applied as u64);
            iteration_high_gamma_applied.push(row.high_gamma_applied as u64);
            iteration_quotient_candidates.push(row.quotient_candidates as u64);
            iteration_quotient_positive_candidates.push(row.quotient_positive_candidates as u64);
            iteration_quotient_selected.push(row.quotient_selected as u64);
            iteration_quotient_score_sum.push(row.quotient_score_sum);
            iteration_baseline_repair_candidates.push(row.baseline_repair_candidates as u64);
            iteration_baseline_repair_improved_candidates
                .push(row.baseline_repair_improved_candidates as u64);
            iteration_baseline_repair_selected.push(row.baseline_repair_selected as u64);
            iteration_baseline_repair_merge_count.push(row.baseline_repair_merge_count as u64);
            iteration_baseline_repair_delta_sum.push(row.baseline_repair_delta_sum);
            iteration_candidate_quality_delta_sum.push(row.candidate_quality_delta_sum);
            iteration_candidate_positive_quality_delta
                .push(row.candidate_positive_quality_delta as u64);
            iteration_candidate_selected_positive_quality_delta
                .push(row.candidate_selected_positive_quality_delta as u64);
            iteration_candidate_rejected_by_quality.push(row.candidate_rejected_by_quality as u64);
            iteration_same_gamma_quality_delta_sum.push(row.same_gamma_quality_delta_sum);
            iteration_high_gamma_quality_delta_sum.push(row.high_gamma_quality_delta_sum);
            iteration_same_gamma_positive_quality_delta
                .push(row.same_gamma_positive_quality_delta as u64);
            iteration_high_gamma_positive_quality_delta
                .push(row.high_gamma_positive_quality_delta as u64);
            iteration_same_gamma_selected_positive_quality_delta
                .push(row.same_gamma_selected_positive_quality_delta as u64);
            iteration_high_gamma_selected_positive_quality_delta
                .push(row.high_gamma_selected_positive_quality_delta as u64);
            iteration_same_gamma_rejected_by_quality
                .push(row.same_gamma_rejected_by_quality as u64);
            iteration_high_gamma_rejected_by_quality
                .push(row.high_gamma_rejected_by_quality as u64);
            iteration_candidate_valid.push(row.candidate_valid as u64);
            iteration_candidate_invalid.push(row.candidate_invalid as u64);
            iteration_candidate_rejected_by_policy.push(row.candidate_rejected_by_policy as u64);
            iteration_same_gamma_valid.push(row.same_gamma_valid as u64);
            iteration_high_gamma_valid.push(row.high_gamma_valid as u64);
            iteration_same_gamma_invalid.push(row.same_gamma_invalid as u64);
            iteration_high_gamma_invalid.push(row.high_gamma_invalid as u64);
            iteration_same_gamma_rejected_by_policy.push(row.same_gamma_rejected_by_policy as u64);
            iteration_high_gamma_rejected_by_policy.push(row.high_gamma_rejected_by_policy as u64);
            iteration_candidate_qpos_spos.push(row.candidate_qpos_spos as u64);
            iteration_candidate_qpos_sneg.push(row.candidate_qpos_sneg as u64);
            iteration_candidate_qneg_spos.push(row.candidate_qneg_spos as u64);
            iteration_candidate_qneg_sneg.push(row.candidate_qneg_sneg as u64);
            iteration_same_gamma_qpos_spos.push(row.same_gamma_qpos_spos as u64);
            iteration_same_gamma_qpos_sneg.push(row.same_gamma_qpos_sneg as u64);
            iteration_same_gamma_qneg_spos.push(row.same_gamma_qneg_spos as u64);
            iteration_same_gamma_qneg_sneg.push(row.same_gamma_qneg_sneg as u64);
            iteration_high_gamma_qpos_spos.push(row.high_gamma_qpos_spos as u64);
            iteration_high_gamma_qpos_sneg.push(row.high_gamma_qpos_sneg as u64);
            iteration_high_gamma_qneg_spos.push(row.high_gamma_qneg_spos as u64);
            iteration_high_gamma_qneg_sneg.push(row.high_gamma_qneg_sneg as u64);
            iteration_candidate_true_positive.push(row.candidate_true_positive as u64);
            iteration_candidate_false_positive.push(row.candidate_false_positive as u64);
            iteration_candidate_false_negative.push(row.candidate_false_negative as u64);
            iteration_candidate_true_negative.push(row.candidate_true_negative as u64);
            iteration_adaptive_local_shake_triggers.push(row.adaptive_local_shake_triggers as u64);
            iteration_adaptive_local_shake_candidates
                .push(row.adaptive_local_shake_candidates as u64);
            iteration_adaptive_local_shake_commits.push(row.adaptive_local_shake_commits as u64);
            iteration_adaptive_local_shake_qf_gain_sum.push(row.adaptive_local_shake_qf_gain_sum);
            iteration_standard_refined_clusters.push(row.standard_refined_clusters as u64);
            iteration_final_refined_clusters.push(row.final_refined_clusters as u64);
        }

        let mut out = std::collections::HashMap::new();
        out.insert(
            "membership".to_string(),
            PyArray1::from_vec(py, membership).into_any().unbind(),
        );
        out.insert(
            "quality".to_string(),
            result
                .quality
                .into_pyobject(py)
                .unwrap()
                .into_any()
                .unbind(),
        );
        out.insert(
            "n_clusters".to_string(),
            (result.clustering.n_clusters as u64)
                .into_pyobject(py)
                .unwrap()
                .into_any()
                .unbind(),
        );
        out.insert(
            "n_iterations_used".to_string(),
            (result.n_iterations_used as u64)
                .into_pyobject(py)
                .unwrap()
                .into_any()
                .unbind(),
        );
        out.insert(
            "audit_enabled".to_string(),
            (audit.enabled as u8)
                .into_pyobject(py)
                .unwrap()
                .into_any()
                .unbind(),
        );
        out.insert(
            "selected_parent_count_total".to_string(),
            (audit.selected_parent_count_total as u64)
                .into_pyobject(py)
                .unwrap()
                .into_any()
                .unbind(),
        );
        out.insert(
            "applied_parent_count_total".to_string(),
            (audit.applied_parent_count_total as u64)
                .into_pyobject(py)
                .unwrap()
                .into_any()
                .unbind(),
        );
        out.insert(
            "rejected_candidate_count_total".to_string(),
            (audit.rejected_candidate_count_total as u64)
                .into_pyobject(py)
                .unwrap()
                .into_any()
                .unbind(),
        );
        out.insert(
            "added_refined_clusters_total".to_string(),
            (audit.added_refined_clusters_total as u64)
                .into_pyobject(py)
                .unwrap()
                .into_any()
                .unbind(),
        );
        out.insert(
            "same_gamma_candidates_total".to_string(),
            (audit.same_gamma_candidates_total as u64)
                .into_pyobject(py)
                .unwrap()
                .into_any()
                .unbind(),
        );
        out.insert(
            "high_gamma_candidates_total".to_string(),
            (audit.high_gamma_candidates_total as u64)
                .into_pyobject(py)
                .unwrap()
                .into_any()
                .unbind(),
        );
        out.insert(
            "same_gamma_applied_total".to_string(),
            (audit.same_gamma_applied_total as u64)
                .into_pyobject(py)
                .unwrap()
                .into_any()
                .unbind(),
        );
        out.insert(
            "high_gamma_applied_total".to_string(),
            (audit.high_gamma_applied_total as u64)
                .into_pyobject(py)
                .unwrap()
                .into_any()
                .unbind(),
        );
        out.insert(
            "quotient_candidates_total".to_string(),
            (audit.quotient_candidates_total as u64)
                .into_pyobject(py)
                .unwrap()
                .into_any()
                .unbind(),
        );
        out.insert(
            "quotient_positive_candidates_total".to_string(),
            (audit.quotient_positive_candidates_total as u64)
                .into_pyobject(py)
                .unwrap()
                .into_any()
                .unbind(),
        );
        out.insert(
            "quotient_selected_total".to_string(),
            (audit.quotient_selected_total as u64)
                .into_pyobject(py)
                .unwrap()
                .into_any()
                .unbind(),
        );
        out.insert(
            "quotient_score_sum".to_string(),
            audit
                .quotient_score_sum
                .into_pyobject(py)
                .unwrap()
                .into_any()
                .unbind(),
        );
        out.insert(
            "baseline_repair_candidates_total".to_string(),
            (audit.baseline_repair_candidates_total as u64)
                .into_pyobject(py)
                .unwrap()
                .into_any()
                .unbind(),
        );
        out.insert(
            "baseline_repair_improved_candidates_total".to_string(),
            (audit.baseline_repair_improved_candidates_total as u64)
                .into_pyobject(py)
                .unwrap()
                .into_any()
                .unbind(),
        );
        out.insert(
            "baseline_repair_selected_total".to_string(),
            (audit.baseline_repair_selected_total as u64)
                .into_pyobject(py)
                .unwrap()
                .into_any()
                .unbind(),
        );
        out.insert(
            "baseline_repair_merge_count_total".to_string(),
            (audit.baseline_repair_merge_count_total as u64)
                .into_pyobject(py)
                .unwrap()
                .into_any()
                .unbind(),
        );
        out.insert(
            "baseline_repair_delta_sum".to_string(),
            audit
                .baseline_repair_delta_sum
                .into_pyobject(py)
                .unwrap()
                .into_any()
                .unbind(),
        );
        out.insert(
            "candidate_quality_delta_sum".to_string(),
            audit
                .candidate_quality_delta_sum
                .into_pyobject(py)
                .unwrap()
                .into_any()
                .unbind(),
        );
        out.insert(
            "candidate_positive_quality_delta_total".to_string(),
            (audit.candidate_positive_quality_delta_total as u64)
                .into_pyobject(py)
                .unwrap()
                .into_any()
                .unbind(),
        );
        out.insert(
            "candidate_selected_positive_quality_delta_total".to_string(),
            (audit.candidate_selected_positive_quality_delta_total as u64)
                .into_pyobject(py)
                .unwrap()
                .into_any()
                .unbind(),
        );
        out.insert(
            "candidate_rejected_by_quality_total".to_string(),
            (audit.candidate_rejected_by_quality_total as u64)
                .into_pyobject(py)
                .unwrap()
                .into_any()
                .unbind(),
        );
        out.insert(
            "same_gamma_quality_delta_sum".to_string(),
            audit
                .same_gamma_quality_delta_sum
                .into_pyobject(py)
                .unwrap()
                .into_any()
                .unbind(),
        );
        out.insert(
            "high_gamma_quality_delta_sum".to_string(),
            audit
                .high_gamma_quality_delta_sum
                .into_pyobject(py)
                .unwrap()
                .into_any()
                .unbind(),
        );
        out.insert(
            "same_gamma_positive_quality_delta_total".to_string(),
            (audit.same_gamma_positive_quality_delta_total as u64)
                .into_pyobject(py)
                .unwrap()
                .into_any()
                .unbind(),
        );
        out.insert(
            "high_gamma_positive_quality_delta_total".to_string(),
            (audit.high_gamma_positive_quality_delta_total as u64)
                .into_pyobject(py)
                .unwrap()
                .into_any()
                .unbind(),
        );
        out.insert(
            "same_gamma_selected_positive_quality_delta_total".to_string(),
            (audit.same_gamma_selected_positive_quality_delta_total as u64)
                .into_pyobject(py)
                .unwrap()
                .into_any()
                .unbind(),
        );
        out.insert(
            "high_gamma_selected_positive_quality_delta_total".to_string(),
            (audit.high_gamma_selected_positive_quality_delta_total as u64)
                .into_pyobject(py)
                .unwrap()
                .into_any()
                .unbind(),
        );
        out.insert(
            "same_gamma_rejected_by_quality_total".to_string(),
            (audit.same_gamma_rejected_by_quality_total as u64)
                .into_pyobject(py)
                .unwrap()
                .into_any()
                .unbind(),
        );
        out.insert(
            "high_gamma_rejected_by_quality_total".to_string(),
            (audit.high_gamma_rejected_by_quality_total as u64)
                .into_pyobject(py)
                .unwrap()
                .into_any()
                .unbind(),
        );
        macro_rules! insert_u64_field {
            ($name:literal, $value:expr) => {
                out.insert(
                    $name.to_string(),
                    (($value) as u64)
                        .into_pyobject(py)
                        .unwrap()
                        .into_any()
                        .unbind(),
                );
            };
        }
        insert_u64_field!("candidate_valid_total", audit.candidate_valid_total);
        insert_u64_field!("candidate_invalid_total", audit.candidate_invalid_total);
        insert_u64_field!(
            "candidate_rejected_by_policy_total",
            audit.candidate_rejected_by_policy_total
        );
        insert_u64_field!("same_gamma_valid_total", audit.same_gamma_valid_total);
        insert_u64_field!("high_gamma_valid_total", audit.high_gamma_valid_total);
        insert_u64_field!("same_gamma_invalid_total", audit.same_gamma_invalid_total);
        insert_u64_field!("high_gamma_invalid_total", audit.high_gamma_invalid_total);
        insert_u64_field!(
            "same_gamma_rejected_by_policy_total",
            audit.same_gamma_rejected_by_policy_total
        );
        insert_u64_field!(
            "high_gamma_rejected_by_policy_total",
            audit.high_gamma_rejected_by_policy_total
        );
        insert_u64_field!("candidate_qpos_spos_total", audit.candidate_qpos_spos_total);
        insert_u64_field!("candidate_qpos_sneg_total", audit.candidate_qpos_sneg_total);
        insert_u64_field!("candidate_qneg_spos_total", audit.candidate_qneg_spos_total);
        insert_u64_field!("candidate_qneg_sneg_total", audit.candidate_qneg_sneg_total);
        insert_u64_field!(
            "same_gamma_qpos_spos_total",
            audit.same_gamma_qpos_spos_total
        );
        insert_u64_field!(
            "same_gamma_qpos_sneg_total",
            audit.same_gamma_qpos_sneg_total
        );
        insert_u64_field!(
            "same_gamma_qneg_spos_total",
            audit.same_gamma_qneg_spos_total
        );
        insert_u64_field!(
            "same_gamma_qneg_sneg_total",
            audit.same_gamma_qneg_sneg_total
        );
        insert_u64_field!(
            "high_gamma_qpos_spos_total",
            audit.high_gamma_qpos_spos_total
        );
        insert_u64_field!(
            "high_gamma_qpos_sneg_total",
            audit.high_gamma_qpos_sneg_total
        );
        insert_u64_field!(
            "high_gamma_qneg_spos_total",
            audit.high_gamma_qneg_spos_total
        );
        insert_u64_field!(
            "high_gamma_qneg_sneg_total",
            audit.high_gamma_qneg_sneg_total
        );
        insert_u64_field!(
            "candidate_true_positive_total",
            audit.candidate_true_positive_total
        );
        insert_u64_field!(
            "candidate_false_positive_total",
            audit.candidate_false_positive_total
        );
        insert_u64_field!(
            "candidate_false_negative_total",
            audit.candidate_false_negative_total
        );
        insert_u64_field!(
            "candidate_true_negative_total",
            audit.candidate_true_negative_total
        );
        insert_u64_field!(
            "adaptive_local_shake_triggers_total",
            audit.adaptive_local_shake_triggers_total
        );
        insert_u64_field!(
            "adaptive_local_shake_candidates_total",
            audit.adaptive_local_shake_candidates_total
        );
        insert_u64_field!(
            "adaptive_local_shake_commits_total",
            audit.adaptive_local_shake_commits_total
        );
        out.insert(
            "adaptive_local_shake_qf_gain_sum".to_string(),
            audit
                .adaptive_local_shake_qf_gain_sum
                .into_pyobject(py)
                .unwrap()
                .into_any()
                .unbind(),
        );
        out.insert(
            "final_quality_guard_enabled".to_string(),
            (audit.final_quality_guard_enabled as u8)
                .into_pyobject(py)
                .unwrap()
                .into_any()
                .unbind(),
        );
        out.insert(
            "final_quality_guard_triggered".to_string(),
            (audit.final_quality_guard_triggered as u8)
                .into_pyobject(py)
                .unwrap()
                .into_any()
                .unbind(),
        );
        out.insert(
            "final_quality_guard_standard_quality".to_string(),
            audit
                .final_quality_guard_standard_quality
                .into_pyobject(py)
                .unwrap()
                .into_any()
                .unbind(),
        );
        out.insert(
            "final_quality_guard_pre_guard_quality".to_string(),
            audit
                .final_quality_guard_pre_guard_quality
                .into_pyobject(py)
                .unwrap()
                .into_any()
                .unbind(),
        );
        out.insert(
            "final_quality_delta_vs_guard_standard".to_string(),
            audit
                .final_quality_delta_vs_guard_standard
                .into_pyobject(py)
                .unwrap()
                .into_any()
                .unbind(),
        );
        out.insert(
            "max_parent_weight_seen".to_string(),
            audit
                .max_parent_weight_seen
                .into_pyobject(py)
                .unwrap()
                .into_any()
                .unbind(),
        );
        out.insert(
            "iteration_depth".to_string(),
            PyArray1::from_vec(py, iteration_depth).into_any().unbind(),
        );
        out.insert(
            "iteration_selected_parents".to_string(),
            PyArray1::from_vec(py, iteration_selected_parents)
                .into_any()
                .unbind(),
        );
        out.insert(
            "iteration_applied_parents".to_string(),
            PyArray1::from_vec(py, iteration_applied_parents)
                .into_any()
                .unbind(),
        );
        out.insert(
            "iteration_same_gamma_candidates".to_string(),
            PyArray1::from_vec(py, iteration_same_gamma_candidates)
                .into_any()
                .unbind(),
        );
        out.insert(
            "iteration_high_gamma_candidates".to_string(),
            PyArray1::from_vec(py, iteration_high_gamma_candidates)
                .into_any()
                .unbind(),
        );
        out.insert(
            "iteration_same_gamma_applied".to_string(),
            PyArray1::from_vec(py, iteration_same_gamma_applied)
                .into_any()
                .unbind(),
        );
        out.insert(
            "iteration_high_gamma_applied".to_string(),
            PyArray1::from_vec(py, iteration_high_gamma_applied)
                .into_any()
                .unbind(),
        );
        out.insert(
            "iteration_quotient_candidates".to_string(),
            PyArray1::from_vec(py, iteration_quotient_candidates)
                .into_any()
                .unbind(),
        );
        out.insert(
            "iteration_quotient_positive_candidates".to_string(),
            PyArray1::from_vec(py, iteration_quotient_positive_candidates)
                .into_any()
                .unbind(),
        );
        out.insert(
            "iteration_quotient_selected".to_string(),
            PyArray1::from_vec(py, iteration_quotient_selected)
                .into_any()
                .unbind(),
        );
        out.insert(
            "iteration_quotient_score_sum".to_string(),
            PyArray1::from_vec(py, iteration_quotient_score_sum)
                .into_any()
                .unbind(),
        );
        out.insert(
            "iteration_baseline_repair_candidates".to_string(),
            PyArray1::from_vec(py, iteration_baseline_repair_candidates)
                .into_any()
                .unbind(),
        );
        out.insert(
            "iteration_baseline_repair_improved_candidates".to_string(),
            PyArray1::from_vec(py, iteration_baseline_repair_improved_candidates)
                .into_any()
                .unbind(),
        );
        out.insert(
            "iteration_baseline_repair_selected".to_string(),
            PyArray1::from_vec(py, iteration_baseline_repair_selected)
                .into_any()
                .unbind(),
        );
        out.insert(
            "iteration_baseline_repair_merge_count".to_string(),
            PyArray1::from_vec(py, iteration_baseline_repair_merge_count)
                .into_any()
                .unbind(),
        );
        out.insert(
            "iteration_baseline_repair_delta_sum".to_string(),
            PyArray1::from_vec(py, iteration_baseline_repair_delta_sum)
                .into_any()
                .unbind(),
        );
        out.insert(
            "iteration_candidate_quality_delta_sum".to_string(),
            PyArray1::from_vec(py, iteration_candidate_quality_delta_sum)
                .into_any()
                .unbind(),
        );
        out.insert(
            "iteration_candidate_positive_quality_delta".to_string(),
            PyArray1::from_vec(py, iteration_candidate_positive_quality_delta)
                .into_any()
                .unbind(),
        );
        out.insert(
            "iteration_candidate_selected_positive_quality_delta".to_string(),
            PyArray1::from_vec(py, iteration_candidate_selected_positive_quality_delta)
                .into_any()
                .unbind(),
        );
        out.insert(
            "iteration_candidate_rejected_by_quality".to_string(),
            PyArray1::from_vec(py, iteration_candidate_rejected_by_quality)
                .into_any()
                .unbind(),
        );
        out.insert(
            "iteration_same_gamma_quality_delta_sum".to_string(),
            PyArray1::from_vec(py, iteration_same_gamma_quality_delta_sum)
                .into_any()
                .unbind(),
        );
        out.insert(
            "iteration_high_gamma_quality_delta_sum".to_string(),
            PyArray1::from_vec(py, iteration_high_gamma_quality_delta_sum)
                .into_any()
                .unbind(),
        );
        out.insert(
            "iteration_same_gamma_positive_quality_delta".to_string(),
            PyArray1::from_vec(py, iteration_same_gamma_positive_quality_delta)
                .into_any()
                .unbind(),
        );
        out.insert(
            "iteration_high_gamma_positive_quality_delta".to_string(),
            PyArray1::from_vec(py, iteration_high_gamma_positive_quality_delta)
                .into_any()
                .unbind(),
        );
        out.insert(
            "iteration_same_gamma_selected_positive_quality_delta".to_string(),
            PyArray1::from_vec(py, iteration_same_gamma_selected_positive_quality_delta)
                .into_any()
                .unbind(),
        );
        out.insert(
            "iteration_high_gamma_selected_positive_quality_delta".to_string(),
            PyArray1::from_vec(py, iteration_high_gamma_selected_positive_quality_delta)
                .into_any()
                .unbind(),
        );
        out.insert(
            "iteration_same_gamma_rejected_by_quality".to_string(),
            PyArray1::from_vec(py, iteration_same_gamma_rejected_by_quality)
                .into_any()
                .unbind(),
        );
        out.insert(
            "iteration_high_gamma_rejected_by_quality".to_string(),
            PyArray1::from_vec(py, iteration_high_gamma_rejected_by_quality)
                .into_any()
                .unbind(),
        );
        macro_rules! insert_vec_u64_field {
            ($name:literal, $value:expr) => {
                out.insert(
                    $name.to_string(),
                    PyArray1::from_vec(py, $value).into_any().unbind(),
                );
            };
        }
        insert_vec_u64_field!("iteration_candidate_valid", iteration_candidate_valid);
        insert_vec_u64_field!("iteration_candidate_invalid", iteration_candidate_invalid);
        insert_vec_u64_field!(
            "iteration_candidate_rejected_by_policy",
            iteration_candidate_rejected_by_policy
        );
        insert_vec_u64_field!("iteration_same_gamma_valid", iteration_same_gamma_valid);
        insert_vec_u64_field!("iteration_high_gamma_valid", iteration_high_gamma_valid);
        insert_vec_u64_field!("iteration_same_gamma_invalid", iteration_same_gamma_invalid);
        insert_vec_u64_field!("iteration_high_gamma_invalid", iteration_high_gamma_invalid);
        insert_vec_u64_field!(
            "iteration_same_gamma_rejected_by_policy",
            iteration_same_gamma_rejected_by_policy
        );
        insert_vec_u64_field!(
            "iteration_high_gamma_rejected_by_policy",
            iteration_high_gamma_rejected_by_policy
        );
        insert_vec_u64_field!(
            "iteration_candidate_qpos_spos",
            iteration_candidate_qpos_spos
        );
        insert_vec_u64_field!(
            "iteration_candidate_qpos_sneg",
            iteration_candidate_qpos_sneg
        );
        insert_vec_u64_field!(
            "iteration_candidate_qneg_spos",
            iteration_candidate_qneg_spos
        );
        insert_vec_u64_field!(
            "iteration_candidate_qneg_sneg",
            iteration_candidate_qneg_sneg
        );
        insert_vec_u64_field!(
            "iteration_same_gamma_qpos_spos",
            iteration_same_gamma_qpos_spos
        );
        insert_vec_u64_field!(
            "iteration_same_gamma_qpos_sneg",
            iteration_same_gamma_qpos_sneg
        );
        insert_vec_u64_field!(
            "iteration_same_gamma_qneg_spos",
            iteration_same_gamma_qneg_spos
        );
        insert_vec_u64_field!(
            "iteration_same_gamma_qneg_sneg",
            iteration_same_gamma_qneg_sneg
        );
        insert_vec_u64_field!(
            "iteration_high_gamma_qpos_spos",
            iteration_high_gamma_qpos_spos
        );
        insert_vec_u64_field!(
            "iteration_high_gamma_qpos_sneg",
            iteration_high_gamma_qpos_sneg
        );
        insert_vec_u64_field!(
            "iteration_high_gamma_qneg_spos",
            iteration_high_gamma_qneg_spos
        );
        insert_vec_u64_field!(
            "iteration_high_gamma_qneg_sneg",
            iteration_high_gamma_qneg_sneg
        );
        insert_vec_u64_field!(
            "iteration_candidate_true_positive",
            iteration_candidate_true_positive
        );
        insert_vec_u64_field!(
            "iteration_candidate_false_positive",
            iteration_candidate_false_positive
        );
        insert_vec_u64_field!(
            "iteration_candidate_false_negative",
            iteration_candidate_false_negative
        );
        insert_vec_u64_field!(
            "iteration_candidate_true_negative",
            iteration_candidate_true_negative
        );
        insert_vec_u64_field!(
            "iteration_adaptive_local_shake_triggers",
            iteration_adaptive_local_shake_triggers
        );
        insert_vec_u64_field!(
            "iteration_adaptive_local_shake_candidates",
            iteration_adaptive_local_shake_candidates
        );
        insert_vec_u64_field!(
            "iteration_adaptive_local_shake_commits",
            iteration_adaptive_local_shake_commits
        );
        out.insert(
            "iteration_adaptive_local_shake_qf_gain_sum".to_string(),
            PyArray1::from_vec(py, iteration_adaptive_local_shake_qf_gain_sum)
                .into_any()
                .unbind(),
        );
        out.insert(
            "iteration_standard_refined_clusters".to_string(),
            PyArray1::from_vec(py, iteration_standard_refined_clusters)
                .into_any()
                .unbind(),
        );
        out.insert(
            "iteration_final_refined_clusters".to_string(),
            PyArray1::from_vec(py, iteration_final_refined_clusters)
                .into_any()
                .unbind(),
        );
        Ok(out)
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
        candidate_clusters,
        resolution,
        max_candidates = 5,
        polish_iterations = 5,
        randomness = 0.01,
        seed = 0,
        min_doc_weight = 0.0,
        min_assigned_fraction = 0.0,
        min_best_group_fraction = 0.0,
        quality_eps = 0.0,
        parallel_candidates = false,
        return_membership = true,
    ))]
    fn non_monotone_group_escape_probe(
        &self,
        py: Python<'_>,
        membership: PyReadonlyArray1<u64>,
        candidate_clusters: PyReadonlyArray1<u64>,
        resolution: f64,
        max_candidates: usize,
        polish_iterations: usize,
        randomness: f64,
        seed: u64,
        min_doc_weight: f64,
        min_assigned_fraction: f64,
        min_best_group_fraction: f64,
        quality_eps: f64,
        parallel_candidates: bool,
        return_membership: bool,
    ) -> PyResult<std::collections::HashMap<String, pyo3::PyObject>> {
        let baseline_clustering = Clustering::from_u64_assignments(membership.as_slice()?)
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        let candidate_clusters = candidate_clusters.as_slice()?.to_vec();

        let computed = py
            .allow_threads(|| {
                compute_non_monotone_group_escape_probe_impl(
                    &self.graph,
                    &baseline_clustering,
                    &candidate_clusters,
                    resolution,
                    max_candidates,
                    polish_iterations,
                    randomness,
                    seed,
                    min_doc_weight,
                    min_assigned_fraction,
                    min_best_group_fraction,
                    quality_eps,
                    parallel_candidates,
                    return_membership,
                )
            })
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

        Ok(non_monotone_group_escape_result_to_py(py, computed))
    }

    #[pyo3(signature = (
        membership,
        candidate_clusters,
        resolution,
        max_candidates = 5,
        polish_iterations = 5,
        randomness = 0.01,
        seed = 0,
        min_doc_weight = 0.0,
        min_assigned_fraction = 0.0,
        min_best_group_fraction = 0.0,
        quality_eps = 0.0,
        parallel_candidates = false,
        return_membership = true,
    ))]
    fn non_monotone_group_escape_probe_u32(
        &self,
        py: Python<'_>,
        membership: PyReadonlyArray1<u32>,
        candidate_clusters: PyReadonlyArray1<u64>,
        resolution: f64,
        max_candidates: usize,
        polish_iterations: usize,
        randomness: f64,
        seed: u64,
        min_doc_weight: f64,
        min_assigned_fraction: f64,
        min_best_group_fraction: f64,
        quality_eps: f64,
        parallel_candidates: bool,
        return_membership: bool,
    ) -> PyResult<std::collections::HashMap<String, pyo3::PyObject>> {
        let baseline_clustering = Clustering::from_assignments(membership.as_slice()?.to_vec());
        ensure_membership_len(&baseline_clustering, self.graph.n_nodes)?;
        let candidate_clusters = candidate_clusters.as_slice()?.to_vec();

        let computed = py
            .allow_threads(|| {
                compute_non_monotone_group_escape_probe_impl(
                    &self.graph,
                    &baseline_clustering,
                    &candidate_clusters,
                    resolution,
                    max_candidates,
                    polish_iterations,
                    randomness,
                    seed,
                    min_doc_weight,
                    min_assigned_fraction,
                    min_best_group_fraction,
                    quality_eps,
                    parallel_candidates,
                    return_membership,
                )
            })
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

        Ok(non_monotone_group_escape_result_to_py(py, computed))
    }

    #[pyo3(signature = (
        membership,
        candidate_clusters,
        resolution,
        max_candidates = 3,
        prescreen_iterations = 1,
        final_iterations = 5,
        finalists = 1,
        label_full_p5 = false,
        randomness = 0.01,
        seed = 0,
        min_doc_weight = 0.0,
        min_assigned_fraction = 0.0,
        min_best_group_fraction = 0.0,
        quality_eps = 0.0,
        return_membership = true,
        approx_polish_labels = false,
        basin_signatures = false,
    ))]
    fn non_monotone_group_escape_multifidelity_probe(
        &self,
        py: Python<'_>,
        membership: PyReadonlyArray1<u64>,
        candidate_clusters: PyReadonlyArray1<u64>,
        resolution: f64,
        max_candidates: usize,
        prescreen_iterations: usize,
        final_iterations: usize,
        finalists: usize,
        label_full_p5: bool,
        randomness: f64,
        seed: u64,
        min_doc_weight: f64,
        min_assigned_fraction: f64,
        min_best_group_fraction: f64,
        quality_eps: f64,
        return_membership: bool,
        approx_polish_labels: bool,
        basin_signatures: bool,
    ) -> PyResult<std::collections::HashMap<String, pyo3::PyObject>> {
        let baseline_clustering = Clustering::from_u64_assignments(membership.as_slice()?)
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        let candidate_clusters = candidate_clusters.as_slice()?.to_vec();

        let computed = py
            .allow_threads(|| {
                compute_non_monotone_group_escape_multifidelity_probe_impl(
                    &self.graph,
                    &baseline_clustering,
                    &candidate_clusters,
                    resolution,
                    max_candidates,
                    prescreen_iterations,
                    final_iterations,
                    finalists,
                    label_full_p5,
                    randomness,
                    seed,
                    min_doc_weight,
                    min_assigned_fraction,
                    min_best_group_fraction,
                    quality_eps,
                    return_membership,
                    approx_polish_labels,
                    basin_signatures,
                )
            })
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

        Ok(non_monotone_multifidelity_result_to_py(py, computed))
    }

    #[pyo3(signature = (
        membership,
        candidate_clusters,
        resolution,
        max_candidates = 3,
        prescreen_iterations = 1,
        final_iterations = 5,
        finalists = 1,
        label_full_p5 = false,
        randomness = 0.01,
        seed = 0,
        min_doc_weight = 0.0,
        min_assigned_fraction = 0.0,
        min_best_group_fraction = 0.0,
        quality_eps = 0.0,
        return_membership = true,
        approx_polish_labels = false,
        basin_signatures = false,
    ))]
    fn non_monotone_group_escape_multifidelity_probe_u32(
        &self,
        py: Python<'_>,
        membership: PyReadonlyArray1<u32>,
        candidate_clusters: PyReadonlyArray1<u64>,
        resolution: f64,
        max_candidates: usize,
        prescreen_iterations: usize,
        final_iterations: usize,
        finalists: usize,
        label_full_p5: bool,
        randomness: f64,
        seed: u64,
        min_doc_weight: f64,
        min_assigned_fraction: f64,
        min_best_group_fraction: f64,
        quality_eps: f64,
        return_membership: bool,
        approx_polish_labels: bool,
        basin_signatures: bool,
    ) -> PyResult<std::collections::HashMap<String, pyo3::PyObject>> {
        let baseline_clustering = Clustering::from_assignments(membership.as_slice()?.to_vec());
        ensure_membership_len(&baseline_clustering, self.graph.n_nodes)?;
        let candidate_clusters = candidate_clusters.as_slice()?.to_vec();

        let computed = py
            .allow_threads(|| {
                compute_non_monotone_group_escape_multifidelity_probe_impl(
                    &self.graph,
                    &baseline_clustering,
                    &candidate_clusters,
                    resolution,
                    max_candidates,
                    prescreen_iterations,
                    final_iterations,
                    finalists,
                    label_full_p5,
                    randomness,
                    seed,
                    min_doc_weight,
                    min_assigned_fraction,
                    min_best_group_fraction,
                    quality_eps,
                    return_membership,
                    approx_polish_labels,
                    basin_signatures,
                )
            })
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

        Ok(non_monotone_multifidelity_result_to_py(py, computed))
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
            "second_neighbor".to_string(),
            PyArray1::from_vec(py, stats.second_neighbor)
                .into_any()
                .unbind(),
        );
        out.insert(
            "second_neighbor_weight".to_string(),
            PyArray1::from_vec(py, stats.second_neighbor_weight)
                .into_any()
                .unbind(),
        );
        out.insert(
            "neighbor_weight_ratio".to_string(),
            PyArray1::from_vec(py, stats.neighbor_weight_ratio)
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
        membership,
        candidate_clusters,
        resolution,
        target_max_weight,
        min_delta_q = 0.0,
        max_moves_per_cluster = 0,
    ))]
    fn trim_oversize_boundary_moves(
        &self,
        py: Python<'_>,
        membership: PyReadonlyArray1<u64>,
        candidate_clusters: PyReadonlyArray1<u64>,
        resolution: f64,
        target_max_weight: f64,
        min_delta_q: f64,
        max_moves_per_cluster: usize,
    ) -> PyResult<std::collections::HashMap<String, pyo3::PyObject>> {
        if !target_max_weight.is_finite() || target_max_weight <= 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "target_max_weight must be finite and > 0",
            ));
        }
        if !min_delta_q.is_finite() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "min_delta_q must be finite",
            ));
        }
        let clustering = Clustering::from_u64_assignments(membership.as_slice()?)
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        if clustering.clusters.len() != self.graph.n_nodes {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "membership length {} does not match graph node count {}",
                clustering.clusters.len(),
                self.graph.n_nodes,
            )));
        }
        let candidate_clusters = candidate_clusters.as_slice()?.to_vec();

        let (proposed_membership, moves) = py.allow_threads(|| {
            let mut ws = Workspace::new(self.graph.n_nodes.max(clustering.n_clusters));
            compute_trim_oversize_boundary_moves(
                &self.graph,
                &clustering,
                &candidate_clusters,
                resolution,
                target_max_weight,
                min_delta_q,
                max_moves_per_cluster,
                &mut ws,
            )
        });

        let n = moves.len();
        let mut source = Vec::with_capacity(n);
        let mut target = Vec::with_capacity(n);
        let mut node = Vec::with_capacity(n);
        let mut node_weight = Vec::with_capacity(n);
        let mut delta_q = Vec::with_capacity(n);
        let mut source_weight_before = Vec::with_capacity(n);
        let mut source_weight_after = Vec::with_capacity(n);
        let mut target_weight_before = Vec::with_capacity(n);
        let mut target_weight_after = Vec::with_capacity(n);
        for mv in moves {
            source.push(mv.source);
            target.push(mv.target);
            node.push(mv.node);
            node_weight.push(mv.node_weight);
            delta_q.push(mv.delta_q);
            source_weight_before.push(mv.source_weight_before);
            source_weight_after.push(mv.source_weight_after);
            target_weight_before.push(mv.target_weight_before);
            target_weight_after.push(mv.target_weight_after);
        }

        let mut out = std::collections::HashMap::new();
        out.insert(
            "membership".to_string(),
            PyArray1::from_vec(py, proposed_membership)
                .into_any()
                .unbind(),
        );
        out.insert(
            "source".to_string(),
            PyArray1::from_vec(py, source).into_any().unbind(),
        );
        out.insert(
            "target".to_string(),
            PyArray1::from_vec(py, target).into_any().unbind(),
        );
        out.insert(
            "node".to_string(),
            PyArray1::from_vec(py, node).into_any().unbind(),
        );
        out.insert(
            "node_weight".to_string(),
            PyArray1::from_vec(py, node_weight).into_any().unbind(),
        );
        out.insert(
            "delta_q".to_string(),
            PyArray1::from_vec(py, delta_q).into_any().unbind(),
        );
        out.insert(
            "source_weight_before".to_string(),
            PyArray1::from_vec(py, source_weight_before)
                .into_any()
                .unbind(),
        );
        out.insert(
            "source_weight_after".to_string(),
            PyArray1::from_vec(py, source_weight_after)
                .into_any()
                .unbind(),
        );
        out.insert(
            "target_weight_before".to_string(),
            PyArray1::from_vec(py, target_weight_before)
                .into_any()
                .unbind(),
        );
        out.insert(
            "target_weight_after".to_string(),
            PyArray1::from_vec(py, target_weight_after)
                .into_any()
                .unbind(),
        );
        Ok(out)
    }

    #[pyo3(signature = (
        membership,
        candidate_clusters,
        resolution,
        epsilon = 0.0,
    ))]
    fn boundary_move_probes(
        &self,
        py: Python<'_>,
        membership: PyReadonlyArray1<u64>,
        candidate_clusters: PyReadonlyArray1<u64>,
        resolution: f64,
        epsilon: f64,
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
        let candidate_clusters = candidate_clusters.as_slice()?.to_vec();

        let probes = py.allow_threads(|| {
            let mut ws = Workspace::new(self.graph.n_nodes.max(clustering.n_clusters));
            compute_boundary_move_probes(
                &self.graph,
                &clustering,
                &candidate_clusters,
                resolution,
                epsilon,
                &mut ws,
            )
        });

        let n = probes.len();
        let mut cluster = Vec::with_capacity(n);
        let mut block_count = Vec::with_capacity(n);
        let mut doc_weight = Vec::with_capacity(n);
        let mut internal_weight = Vec::with_capacity(n);
        let mut external_weight = Vec::with_capacity(n);
        let mut conductance = Vec::with_capacity(n);
        let mut leafness = Vec::with_capacity(n);
        let mut top_neighbor = Vec::with_capacity(n);
        let mut top_neighbor_weight = Vec::with_capacity(n);
        let mut second_neighbor = Vec::with_capacity(n);
        let mut second_neighbor_weight = Vec::with_capacity(n);
        let mut neighbor_weight_ratio = Vec::with_capacity(n);
        let mut positive_move_count = Vec::with_capacity(n);
        let mut positive_move_weight = Vec::with_capacity(n);
        let mut positive_delta_q = Vec::with_capacity(n);
        let mut near_neutral_move_count = Vec::with_capacity(n);
        let mut near_neutral_move_weight = Vec::with_capacity(n);
        let mut near_neutral_delta_q = Vec::with_capacity(n);
        let mut best_move_delta_q = Vec::with_capacity(n);
        let mut best_move_node = Vec::with_capacity(n);
        let mut best_move_target = Vec::with_capacity(n);
        let mut top_move_count = Vec::with_capacity(n);
        let mut second_move_count = Vec::with_capacity(n);

        for probe in probes {
            cluster.push(probe.cluster);
            block_count.push(probe.block_count);
            doc_weight.push(probe.doc_weight);
            internal_weight.push(probe.internal_weight);
            external_weight.push(probe.external_weight);
            conductance.push(probe.conductance);
            leafness.push(probe.leafness);
            top_neighbor.push(probe.top_neighbor);
            top_neighbor_weight.push(probe.top_neighbor_weight);
            second_neighbor.push(probe.second_neighbor);
            second_neighbor_weight.push(probe.second_neighbor_weight);
            neighbor_weight_ratio.push(probe.neighbor_weight_ratio);
            positive_move_count.push(probe.positive_move_count);
            positive_move_weight.push(probe.positive_move_weight);
            positive_delta_q.push(probe.positive_delta_q);
            near_neutral_move_count.push(probe.near_neutral_move_count);
            near_neutral_move_weight.push(probe.near_neutral_move_weight);
            near_neutral_delta_q.push(probe.near_neutral_delta_q);
            best_move_delta_q.push(probe.best_move_delta_q);
            best_move_node.push(probe.best_move_node);
            best_move_target.push(probe.best_move_target);
            top_move_count.push(probe.top_move_count);
            second_move_count.push(probe.second_move_count);
        }

        let mut out = std::collections::HashMap::new();
        out.insert(
            "cluster".to_string(),
            PyArray1::from_vec(py, cluster).into_any().unbind(),
        );
        out.insert(
            "block_count".to_string(),
            PyArray1::from_vec(py, block_count).into_any().unbind(),
        );
        out.insert(
            "doc_weight".to_string(),
            PyArray1::from_vec(py, doc_weight).into_any().unbind(),
        );
        out.insert(
            "internal_weight".to_string(),
            PyArray1::from_vec(py, internal_weight).into_any().unbind(),
        );
        out.insert(
            "external_weight".to_string(),
            PyArray1::from_vec(py, external_weight).into_any().unbind(),
        );
        out.insert(
            "conductance".to_string(),
            PyArray1::from_vec(py, conductance).into_any().unbind(),
        );
        out.insert(
            "leafness".to_string(),
            PyArray1::from_vec(py, leafness).into_any().unbind(),
        );
        out.insert(
            "top_neighbor".to_string(),
            PyArray1::from_vec(py, top_neighbor).into_any().unbind(),
        );
        out.insert(
            "top_neighbor_weight".to_string(),
            PyArray1::from_vec(py, top_neighbor_weight)
                .into_any()
                .unbind(),
        );
        out.insert(
            "second_neighbor".to_string(),
            PyArray1::from_vec(py, second_neighbor).into_any().unbind(),
        );
        out.insert(
            "second_neighbor_weight".to_string(),
            PyArray1::from_vec(py, second_neighbor_weight)
                .into_any()
                .unbind(),
        );
        out.insert(
            "neighbor_weight_ratio".to_string(),
            PyArray1::from_vec(py, neighbor_weight_ratio)
                .into_any()
                .unbind(),
        );
        out.insert(
            "positive_move_count".to_string(),
            PyArray1::from_vec(py, positive_move_count)
                .into_any()
                .unbind(),
        );
        out.insert(
            "positive_move_weight".to_string(),
            PyArray1::from_vec(py, positive_move_weight)
                .into_any()
                .unbind(),
        );
        out.insert(
            "positive_delta_q".to_string(),
            PyArray1::from_vec(py, positive_delta_q).into_any().unbind(),
        );
        out.insert(
            "near_neutral_move_count".to_string(),
            PyArray1::from_vec(py, near_neutral_move_count)
                .into_any()
                .unbind(),
        );
        out.insert(
            "near_neutral_move_weight".to_string(),
            PyArray1::from_vec(py, near_neutral_move_weight)
                .into_any()
                .unbind(),
        );
        out.insert(
            "near_neutral_delta_q".to_string(),
            PyArray1::from_vec(py, near_neutral_delta_q)
                .into_any()
                .unbind(),
        );
        out.insert(
            "best_move_delta_q".to_string(),
            PyArray1::from_vec(py, best_move_delta_q)
                .into_any()
                .unbind(),
        );
        out.insert(
            "best_move_node".to_string(),
            PyArray1::from_vec(py, best_move_node).into_any().unbind(),
        );
        out.insert(
            "best_move_target".to_string(),
            PyArray1::from_vec(py, best_move_target).into_any().unbind(),
        );
        out.insert(
            "top_move_count".to_string(),
            PyArray1::from_vec(py, top_move_count).into_any().unbind(),
        );
        out.insert(
            "second_move_count".to_string(),
            PyArray1::from_vec(py, second_move_count)
                .into_any()
                .unbind(),
        );
        Ok(out)
    }

    #[pyo3(signature = (
        membership,
        candidate_clusters,
        resolution,
    ))]
    fn boundary_group_probes(
        &self,
        py: Python<'_>,
        membership: PyReadonlyArray1<u64>,
        candidate_clusters: PyReadonlyArray1<u64>,
        resolution: f64,
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
        let candidate_clusters = candidate_clusters.as_slice()?.to_vec();

        let probes = py.allow_threads(|| {
            let mut ws = Workspace::new(self.graph.n_nodes.max(clustering.n_clusters));
            compute_boundary_group_probes(
                &self.graph,
                &clustering,
                &candidate_clusters,
                resolution,
                &mut ws,
            )
        });

        let n = probes.len();
        let mut cluster = Vec::with_capacity(n);
        let mut block_count = Vec::with_capacity(n);
        let mut doc_weight = Vec::with_capacity(n);
        let mut top_neighbor = Vec::with_capacity(n);
        let mut second_neighbor = Vec::with_capacity(n);
        let mut top_group_count = Vec::with_capacity(n);
        let mut top_group_weight = Vec::with_capacity(n);
        let mut top_group_to_target_weight = Vec::with_capacity(n);
        let mut top_group_cut_weight = Vec::with_capacity(n);
        let mut top_group_move_delta_q = Vec::with_capacity(n);
        let mut top_group_split_delta_q = Vec::with_capacity(n);
        let mut top_group_is_full_cluster = Vec::with_capacity(n);
        let mut second_group_count = Vec::with_capacity(n);
        let mut second_group_weight = Vec::with_capacity(n);
        let mut second_group_to_target_weight = Vec::with_capacity(n);
        let mut second_group_cut_weight = Vec::with_capacity(n);
        let mut second_group_move_delta_q = Vec::with_capacity(n);
        let mut second_group_split_delta_q = Vec::with_capacity(n);
        let mut second_group_is_full_cluster = Vec::with_capacity(n);
        let mut best_delta_q = Vec::with_capacity(n);
        let mut best_action = Vec::with_capacity(n);

        for probe in probes {
            cluster.push(probe.cluster);
            block_count.push(probe.block_count);
            doc_weight.push(probe.doc_weight);
            top_neighbor.push(probe.top_neighbor);
            second_neighbor.push(probe.second_neighbor);
            top_group_count.push(probe.top_group_count);
            top_group_weight.push(probe.top_group_weight);
            top_group_to_target_weight.push(probe.top_group_to_target_weight);
            top_group_cut_weight.push(probe.top_group_cut_weight);
            top_group_move_delta_q.push(probe.top_group_move_delta_q);
            top_group_split_delta_q.push(probe.top_group_split_delta_q);
            top_group_is_full_cluster.push(probe.top_group_is_full_cluster);
            second_group_count.push(probe.second_group_count);
            second_group_weight.push(probe.second_group_weight);
            second_group_to_target_weight.push(probe.second_group_to_target_weight);
            second_group_cut_weight.push(probe.second_group_cut_weight);
            second_group_move_delta_q.push(probe.second_group_move_delta_q);
            second_group_split_delta_q.push(probe.second_group_split_delta_q);
            second_group_is_full_cluster.push(probe.second_group_is_full_cluster);
            best_delta_q.push(probe.best_delta_q);
            best_action.push(probe.best_action);
        }

        let mut out = std::collections::HashMap::new();
        out.insert(
            "cluster".to_string(),
            PyArray1::from_vec(py, cluster).into_any().unbind(),
        );
        out.insert(
            "block_count".to_string(),
            PyArray1::from_vec(py, block_count).into_any().unbind(),
        );
        out.insert(
            "doc_weight".to_string(),
            PyArray1::from_vec(py, doc_weight).into_any().unbind(),
        );
        out.insert(
            "top_neighbor".to_string(),
            PyArray1::from_vec(py, top_neighbor).into_any().unbind(),
        );
        out.insert(
            "second_neighbor".to_string(),
            PyArray1::from_vec(py, second_neighbor).into_any().unbind(),
        );
        out.insert(
            "top_group_count".to_string(),
            PyArray1::from_vec(py, top_group_count).into_any().unbind(),
        );
        out.insert(
            "top_group_weight".to_string(),
            PyArray1::from_vec(py, top_group_weight).into_any().unbind(),
        );
        out.insert(
            "top_group_to_target_weight".to_string(),
            PyArray1::from_vec(py, top_group_to_target_weight)
                .into_any()
                .unbind(),
        );
        out.insert(
            "top_group_cut_weight".to_string(),
            PyArray1::from_vec(py, top_group_cut_weight)
                .into_any()
                .unbind(),
        );
        out.insert(
            "top_group_move_delta_q".to_string(),
            PyArray1::from_vec(py, top_group_move_delta_q)
                .into_any()
                .unbind(),
        );
        out.insert(
            "top_group_split_delta_q".to_string(),
            PyArray1::from_vec(py, top_group_split_delta_q)
                .into_any()
                .unbind(),
        );
        out.insert(
            "top_group_is_full_cluster".to_string(),
            PyArray1::from_vec(py, top_group_is_full_cluster)
                .into_any()
                .unbind(),
        );
        out.insert(
            "second_group_count".to_string(),
            PyArray1::from_vec(py, second_group_count)
                .into_any()
                .unbind(),
        );
        out.insert(
            "second_group_weight".to_string(),
            PyArray1::from_vec(py, second_group_weight)
                .into_any()
                .unbind(),
        );
        out.insert(
            "second_group_to_target_weight".to_string(),
            PyArray1::from_vec(py, second_group_to_target_weight)
                .into_any()
                .unbind(),
        );
        out.insert(
            "second_group_cut_weight".to_string(),
            PyArray1::from_vec(py, second_group_cut_weight)
                .into_any()
                .unbind(),
        );
        out.insert(
            "second_group_move_delta_q".to_string(),
            PyArray1::from_vec(py, second_group_move_delta_q)
                .into_any()
                .unbind(),
        );
        out.insert(
            "second_group_split_delta_q".to_string(),
            PyArray1::from_vec(py, second_group_split_delta_q)
                .into_any()
                .unbind(),
        );
        out.insert(
            "second_group_is_full_cluster".to_string(),
            PyArray1::from_vec(py, second_group_is_full_cluster)
                .into_any()
                .unbind(),
        );
        out.insert(
            "best_delta_q".to_string(),
            PyArray1::from_vec(py, best_delta_q).into_any().unbind(),
        );
        out.insert(
            "best_action".to_string(),
            PyArray1::from_vec(py, best_action).into_any().unbind(),
        );
        Ok(out)
    }

    #[pyo3(signature = (
        membership,
        candidate_clusters,
        resolution,
        count,
        epsilon = 0.0,
        min_doc_weight = 0.0,
        max_incident_directed_edges = 0,
        min_best_delta_q = 0.0,
        min_assigned_fraction = 0.0,
        min_best_group_fraction = 0.0,
    ))]
    fn external_grain_priority_clusters_u32<'py>(
        &self,
        py: Python<'py>,
        membership: PyReadonlyArray1<u32>,
        candidate_clusters: PyReadonlyArray1<u64>,
        resolution: f64,
        count: usize,
        epsilon: f64,
        min_doc_weight: f64,
        max_incident_directed_edges: u64,
        min_best_delta_q: f64,
        min_assigned_fraction: f64,
        min_best_group_fraction: f64,
    ) -> PyResult<Py<PyArray1<u64>>> {
        let clustering = Clustering::from_assignments(membership.as_slice()?.to_vec());
        ensure_membership_len(&clustering, self.graph.n_nodes)?;
        let candidate_clusters = candidate_clusters.as_slice()?.to_vec();
        let selected = py.allow_threads(|| {
            compute_external_grain_priority_clusters(
                &self.graph,
                &clustering,
                &candidate_clusters,
                resolution,
                epsilon,
                count,
                ExternalGrainSelectionPolicy {
                    min_doc_weight,
                    max_incident_directed_edges,
                    min_best_delta_q,
                    min_assigned_fraction,
                    min_best_group_fraction,
                },
            )
        });
        Ok(PyArray1::from_vec(py, selected).into())
    }

    #[pyo3(signature = (
        membership,
        candidate_clusters,
        resolution,
        epsilon = 0.0,
        min_doc_weight = 0.0,
        max_incident_directed_edges = 0,
        min_best_delta_q = 0.0,
        min_assigned_fraction = 0.0,
        min_best_group_fraction = 0.0,
    ))]
    fn external_grain_probes(
        &self,
        py: Python<'_>,
        membership: PyReadonlyArray1<u64>,
        candidate_clusters: PyReadonlyArray1<u64>,
        resolution: f64,
        epsilon: f64,
        min_doc_weight: f64,
        max_incident_directed_edges: u64,
        min_best_delta_q: f64,
        min_assigned_fraction: f64,
        min_best_group_fraction: f64,
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
        let candidate_clusters = candidate_clusters.as_slice()?.to_vec();

        let probes = py.allow_threads(|| {
            let mut ws = Workspace::new(self.graph.n_nodes.max(clustering.n_clusters));
            compute_external_grain_probes(
                &self.graph,
                &clustering,
                &candidate_clusters,
                resolution,
                epsilon,
                &mut ws,
            )
        });

        let selection = select_external_grain_probes(
            &probes,
            ExternalGrainSelectionPolicy {
                min_doc_weight,
                max_incident_directed_edges,
                min_best_delta_q,
                min_assigned_fraction,
                min_best_group_fraction,
            },
        );
        let n = probes.len();
        let mut cluster = Vec::with_capacity(n);
        let mut block_count = Vec::with_capacity(n);
        let mut doc_weight = Vec::with_capacity(n);
        let mut incident_directed_edges = Vec::with_capacity(n);
        let mut source_directed_edges = Vec::with_capacity(n);
        let mut external_directed_edges = Vec::with_capacity(n);
        let mut n_external_groups = Vec::with_capacity(n);
        let mut assigned_count = Vec::with_capacity(n);
        let mut assigned_weight = Vec::with_capacity(n);
        let mut assigned_fraction = Vec::with_capacity(n);
        let mut largest_group_target = Vec::with_capacity(n);
        let mut largest_group_count = Vec::with_capacity(n);
        let mut largest_group_weight = Vec::with_capacity(n);
        let mut largest_group_fraction = Vec::with_capacity(n);
        let mut largest_group_to_target_weight = Vec::with_capacity(n);
        let mut largest_group_cut_weight = Vec::with_capacity(n);
        let mut largest_group_move_delta_q = Vec::with_capacity(n);
        let mut largest_group_split_delta_q = Vec::with_capacity(n);
        let mut second_group_target = Vec::with_capacity(n);
        let mut second_group_weight = Vec::with_capacity(n);
        let mut second_group_fraction = Vec::with_capacity(n);
        let mut best_group_target = Vec::with_capacity(n);
        let mut best_group_count = Vec::with_capacity(n);
        let mut best_group_weight = Vec::with_capacity(n);
        let mut best_group_fraction = Vec::with_capacity(n);
        let mut best_group_to_target_weight = Vec::with_capacity(n);
        let mut best_group_cut_weight = Vec::with_capacity(n);
        let mut best_group_move_delta_q = Vec::with_capacity(n);
        let mut best_group_split_delta_q = Vec::with_capacity(n);
        let mut best_group_delta_q = Vec::with_capacity(n);
        let mut best_group_action = Vec::with_capacity(n);
        let mut positive_group_count = Vec::with_capacity(n);
        let mut positive_group_weight = Vec::with_capacity(n);
        let mut near_neutral_group_count = Vec::with_capacity(n);
        let mut near_neutral_group_weight = Vec::with_capacity(n);
        let mut recommended_for_split_repair = Vec::with_capacity(n);
        let mut priority = Vec::with_capacity(n);

        for (probe, selected) in probes.into_iter().zip(selection) {
            cluster.push(probe.cluster);
            block_count.push(probe.block_count);
            doc_weight.push(probe.doc_weight);
            incident_directed_edges.push(probe.incident_directed_edges);
            source_directed_edges.push(probe.source_directed_edges);
            external_directed_edges.push(probe.external_directed_edges);
            n_external_groups.push(probe.n_external_groups);
            assigned_count.push(probe.assigned_count);
            assigned_weight.push(probe.assigned_weight);
            assigned_fraction.push(probe.assigned_fraction);
            largest_group_target.push(probe.largest_group_target);
            largest_group_count.push(probe.largest_group_count);
            largest_group_weight.push(probe.largest_group_weight);
            largest_group_fraction.push(probe.largest_group_fraction);
            largest_group_to_target_weight.push(probe.largest_group_to_target_weight);
            largest_group_cut_weight.push(probe.largest_group_cut_weight);
            largest_group_move_delta_q.push(probe.largest_group_move_delta_q);
            largest_group_split_delta_q.push(probe.largest_group_split_delta_q);
            second_group_target.push(probe.second_group_target);
            second_group_weight.push(probe.second_group_weight);
            second_group_fraction.push(probe.second_group_fraction);
            best_group_target.push(probe.best_group_target);
            best_group_count.push(probe.best_group_count);
            best_group_weight.push(probe.best_group_weight);
            best_group_fraction.push(probe.best_group_fraction);
            best_group_to_target_weight.push(probe.best_group_to_target_weight);
            best_group_cut_weight.push(probe.best_group_cut_weight);
            best_group_move_delta_q.push(probe.best_group_move_delta_q);
            best_group_split_delta_q.push(probe.best_group_split_delta_q);
            best_group_delta_q.push(probe.best_group_delta_q);
            best_group_action.push(probe.best_group_action);
            positive_group_count.push(probe.positive_group_count);
            positive_group_weight.push(probe.positive_group_weight);
            near_neutral_group_count.push(probe.near_neutral_group_count);
            near_neutral_group_weight.push(probe.near_neutral_group_weight);
            recommended_for_split_repair.push(selected.recommended_for_split_repair);
            priority.push(selected.priority);
        }

        let mut out = std::collections::HashMap::new();
        out.insert(
            "cluster".to_string(),
            PyArray1::from_vec(py, cluster).into_any().unbind(),
        );
        out.insert(
            "block_count".to_string(),
            PyArray1::from_vec(py, block_count).into_any().unbind(),
        );
        out.insert(
            "doc_weight".to_string(),
            PyArray1::from_vec(py, doc_weight).into_any().unbind(),
        );
        out.insert(
            "incident_directed_edges".to_string(),
            PyArray1::from_vec(py, incident_directed_edges)
                .into_any()
                .unbind(),
        );
        out.insert(
            "source_directed_edges".to_string(),
            PyArray1::from_vec(py, source_directed_edges)
                .into_any()
                .unbind(),
        );
        out.insert(
            "external_directed_edges".to_string(),
            PyArray1::from_vec(py, external_directed_edges)
                .into_any()
                .unbind(),
        );
        out.insert(
            "n_external_groups".to_string(),
            PyArray1::from_vec(py, n_external_groups)
                .into_any()
                .unbind(),
        );
        out.insert(
            "assigned_count".to_string(),
            PyArray1::from_vec(py, assigned_count).into_any().unbind(),
        );
        out.insert(
            "assigned_weight".to_string(),
            PyArray1::from_vec(py, assigned_weight).into_any().unbind(),
        );
        out.insert(
            "assigned_fraction".to_string(),
            PyArray1::from_vec(py, assigned_fraction)
                .into_any()
                .unbind(),
        );
        out.insert(
            "largest_group_target".to_string(),
            PyArray1::from_vec(py, largest_group_target)
                .into_any()
                .unbind(),
        );
        out.insert(
            "largest_group_count".to_string(),
            PyArray1::from_vec(py, largest_group_count)
                .into_any()
                .unbind(),
        );
        out.insert(
            "largest_group_weight".to_string(),
            PyArray1::from_vec(py, largest_group_weight)
                .into_any()
                .unbind(),
        );
        out.insert(
            "largest_group_fraction".to_string(),
            PyArray1::from_vec(py, largest_group_fraction)
                .into_any()
                .unbind(),
        );
        out.insert(
            "largest_group_to_target_weight".to_string(),
            PyArray1::from_vec(py, largest_group_to_target_weight)
                .into_any()
                .unbind(),
        );
        out.insert(
            "largest_group_cut_weight".to_string(),
            PyArray1::from_vec(py, largest_group_cut_weight)
                .into_any()
                .unbind(),
        );
        out.insert(
            "largest_group_move_delta_q".to_string(),
            PyArray1::from_vec(py, largest_group_move_delta_q)
                .into_any()
                .unbind(),
        );
        out.insert(
            "largest_group_split_delta_q".to_string(),
            PyArray1::from_vec(py, largest_group_split_delta_q)
                .into_any()
                .unbind(),
        );
        out.insert(
            "second_group_target".to_string(),
            PyArray1::from_vec(py, second_group_target)
                .into_any()
                .unbind(),
        );
        out.insert(
            "second_group_weight".to_string(),
            PyArray1::from_vec(py, second_group_weight)
                .into_any()
                .unbind(),
        );
        out.insert(
            "second_group_fraction".to_string(),
            PyArray1::from_vec(py, second_group_fraction)
                .into_any()
                .unbind(),
        );
        out.insert(
            "best_group_target".to_string(),
            PyArray1::from_vec(py, best_group_target)
                .into_any()
                .unbind(),
        );
        out.insert(
            "best_group_count".to_string(),
            PyArray1::from_vec(py, best_group_count).into_any().unbind(),
        );
        out.insert(
            "best_group_weight".to_string(),
            PyArray1::from_vec(py, best_group_weight)
                .into_any()
                .unbind(),
        );
        out.insert(
            "best_group_fraction".to_string(),
            PyArray1::from_vec(py, best_group_fraction)
                .into_any()
                .unbind(),
        );
        out.insert(
            "best_group_to_target_weight".to_string(),
            PyArray1::from_vec(py, best_group_to_target_weight)
                .into_any()
                .unbind(),
        );
        out.insert(
            "best_group_cut_weight".to_string(),
            PyArray1::from_vec(py, best_group_cut_weight)
                .into_any()
                .unbind(),
        );
        out.insert(
            "best_group_move_delta_q".to_string(),
            PyArray1::from_vec(py, best_group_move_delta_q)
                .into_any()
                .unbind(),
        );
        out.insert(
            "best_group_split_delta_q".to_string(),
            PyArray1::from_vec(py, best_group_split_delta_q)
                .into_any()
                .unbind(),
        );
        out.insert(
            "best_group_delta_q".to_string(),
            PyArray1::from_vec(py, best_group_delta_q)
                .into_any()
                .unbind(),
        );
        out.insert(
            "best_group_action".to_string(),
            PyArray1::from_vec(py, best_group_action)
                .into_any()
                .unbind(),
        );
        out.insert(
            "positive_group_count".to_string(),
            PyArray1::from_vec(py, positive_group_count)
                .into_any()
                .unbind(),
        );
        out.insert(
            "positive_group_weight".to_string(),
            PyArray1::from_vec(py, positive_group_weight)
                .into_any()
                .unbind(),
        );
        out.insert(
            "near_neutral_group_count".to_string(),
            PyArray1::from_vec(py, near_neutral_group_count)
                .into_any()
                .unbind(),
        );
        out.insert(
            "near_neutral_group_weight".to_string(),
            PyArray1::from_vec(py, near_neutral_group_weight)
                .into_any()
                .unbind(),
        );
        out.insert(
            "recommended_for_split_repair".to_string(),
            PyArray1::from_vec(py, recommended_for_split_repair)
                .into_any()
                .unbind(),
        );
        out.insert(
            "priority".to_string(),
            PyArray1::from_vec(py, priority).into_any().unbind(),
        );
        Ok(out)
    }

    #[pyo3(signature = (
        membership,
        candidate_clusters,
        resolution,
        gamma_multipliers,
        min_core_weight = 25.0,
        randomness = 0.01,
        seed = 0,
    ))]
    fn multi_core_split_probes(
        &self,
        py: Python<'_>,
        membership: PyReadonlyArray1<u64>,
        candidate_clusters: PyReadonlyArray1<u64>,
        resolution: f64,
        gamma_multipliers: PyReadonlyArray1<f64>,
        min_core_weight: f64,
        randomness: f64,
        seed: u64,
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
        let candidate_clusters = candidate_clusters.as_slice()?.to_vec();
        let gamma_multipliers = gamma_multipliers.as_slice()?.to_vec();

        let probes = py.allow_threads(|| {
            let mut ws = Workspace::new(self.graph.n_nodes.max(clustering.n_clusters));
            compute_multi_core_split_probes(
                &self.graph,
                &clustering,
                &candidate_clusters,
                resolution,
                &gamma_multipliers,
                min_core_weight,
                randomness,
                seed,
                &mut ws,
            )
        });

        let n = probes.len();
        let mut cluster = Vec::with_capacity(n);
        let mut gamma_multiplier = Vec::with_capacity(n);
        let mut probe_resolution = Vec::with_capacity(n);
        let mut block_count = Vec::with_capacity(n);
        let mut doc_weight = Vec::with_capacity(n);
        let mut internal_weight = Vec::with_capacity(n);
        let mut induced_directed_edges = Vec::with_capacity(n);
        let mut n_parts = Vec::with_capacity(n);
        let mut non_singleton_parts = Vec::with_capacity(n);
        let mut singleton_parts = Vec::with_capacity(n);
        let mut singleton_weight = Vec::with_capacity(n);
        let mut core_part_count = Vec::with_capacity(n);
        let mut core_part_weight = Vec::with_capacity(n);
        let mut largest_part_weight = Vec::with_capacity(n);
        let mut second_part_weight = Vec::with_capacity(n);
        let mut largest_part_fraction = Vec::with_capacity(n);
        let mut cut_weight = Vec::with_capacity(n);
        let mut split_delta_q_base = Vec::with_capacity(n);
        let mut split_delta_q_probe = Vec::with_capacity(n);
        let mut hysteresis_only = Vec::with_capacity(n);

        for probe in probes {
            cluster.push(probe.cluster);
            gamma_multiplier.push(probe.gamma_multiplier);
            probe_resolution.push(probe.probe_resolution);
            block_count.push(probe.block_count);
            doc_weight.push(probe.doc_weight);
            internal_weight.push(probe.internal_weight);
            induced_directed_edges.push(probe.induced_directed_edges);
            n_parts.push(probe.n_parts);
            non_singleton_parts.push(probe.non_singleton_parts);
            singleton_parts.push(probe.singleton_parts);
            singleton_weight.push(probe.singleton_weight);
            core_part_count.push(probe.core_part_count);
            core_part_weight.push(probe.core_part_weight);
            largest_part_weight.push(probe.largest_part_weight);
            second_part_weight.push(probe.second_part_weight);
            largest_part_fraction.push(probe.largest_part_fraction);
            cut_weight.push(probe.cut_weight);
            split_delta_q_base.push(probe.split_delta_q_base);
            split_delta_q_probe.push(probe.split_delta_q_probe);
            hysteresis_only.push(probe.hysteresis_only);
        }

        let mut out = std::collections::HashMap::new();
        out.insert(
            "cluster".to_string(),
            PyArray1::from_vec(py, cluster).into_any().unbind(),
        );
        out.insert(
            "gamma_multiplier".to_string(),
            PyArray1::from_vec(py, gamma_multiplier).into_any().unbind(),
        );
        out.insert(
            "probe_resolution".to_string(),
            PyArray1::from_vec(py, probe_resolution).into_any().unbind(),
        );
        out.insert(
            "block_count".to_string(),
            PyArray1::from_vec(py, block_count).into_any().unbind(),
        );
        out.insert(
            "doc_weight".to_string(),
            PyArray1::from_vec(py, doc_weight).into_any().unbind(),
        );
        out.insert(
            "internal_weight".to_string(),
            PyArray1::from_vec(py, internal_weight).into_any().unbind(),
        );
        out.insert(
            "induced_directed_edges".to_string(),
            PyArray1::from_vec(py, induced_directed_edges)
                .into_any()
                .unbind(),
        );
        out.insert(
            "n_parts".to_string(),
            PyArray1::from_vec(py, n_parts).into_any().unbind(),
        );
        out.insert(
            "non_singleton_parts".to_string(),
            PyArray1::from_vec(py, non_singleton_parts)
                .into_any()
                .unbind(),
        );
        out.insert(
            "singleton_parts".to_string(),
            PyArray1::from_vec(py, singleton_parts).into_any().unbind(),
        );
        out.insert(
            "singleton_weight".to_string(),
            PyArray1::from_vec(py, singleton_weight).into_any().unbind(),
        );
        out.insert(
            "core_part_count".to_string(),
            PyArray1::from_vec(py, core_part_count).into_any().unbind(),
        );
        out.insert(
            "core_part_weight".to_string(),
            PyArray1::from_vec(py, core_part_weight).into_any().unbind(),
        );
        out.insert(
            "largest_part_weight".to_string(),
            PyArray1::from_vec(py, largest_part_weight)
                .into_any()
                .unbind(),
        );
        out.insert(
            "second_part_weight".to_string(),
            PyArray1::from_vec(py, second_part_weight)
                .into_any()
                .unbind(),
        );
        out.insert(
            "largest_part_fraction".to_string(),
            PyArray1::from_vec(py, largest_part_fraction)
                .into_any()
                .unbind(),
        );
        out.insert(
            "cut_weight".to_string(),
            PyArray1::from_vec(py, cut_weight).into_any().unbind(),
        );
        out.insert(
            "split_delta_q_base".to_string(),
            PyArray1::from_vec(py, split_delta_q_base)
                .into_any()
                .unbind(),
        );
        out.insert(
            "split_delta_q_probe".to_string(),
            PyArray1::from_vec(py, split_delta_q_probe)
                .into_any()
                .unbind(),
        );
        out.insert(
            "hysteresis_only".to_string(),
            PyArray1::from_vec(py, hysteresis_only).into_any().unbind(),
        );
        Ok(out)
    }

    #[pyo3(signature = (
        membership,
        candidate_clusters,
        resolution,
        gamma_multipliers,
        min_core_weight = 25.0,
        randomness = 0.01,
        repair_epsilon = 0.0,
        seed = 0,
        pair_seeded = false,
    ))]
    fn split_merge_repair_probes(
        &self,
        py: Python<'_>,
        membership: PyReadonlyArray1<u64>,
        candidate_clusters: PyReadonlyArray1<u64>,
        resolution: f64,
        gamma_multipliers: PyReadonlyArray1<f64>,
        min_core_weight: f64,
        randomness: f64,
        repair_epsilon: f64,
        seed: u64,
        pair_seeded: bool,
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
        let candidate_clusters = candidate_clusters.as_slice()?.to_vec();
        let gamma_multipliers = gamma_multipliers.as_slice()?.to_vec();

        let probes = py.allow_threads(|| {
            let mut ws = Workspace::new(self.graph.n_nodes.max(clustering.n_clusters));
            compute_split_merge_repair_probes(
                &self.graph,
                &clustering,
                &candidate_clusters,
                resolution,
                &gamma_multipliers,
                min_core_weight,
                randomness,
                repair_epsilon,
                seed,
                pair_seeded,
                &mut ws,
            )
        });

        let n = probes.len();
        let mut cluster = Vec::with_capacity(n);
        let mut gamma_multiplier = Vec::with_capacity(n);
        let mut probe_resolution = Vec::with_capacity(n);
        let mut block_count = Vec::with_capacity(n);
        let mut doc_weight = Vec::with_capacity(n);
        let mut induced_directed_edges = Vec::with_capacity(n);
        let mut n_parts = Vec::with_capacity(n);
        let mut core_part_count = Vec::with_capacity(n);
        let mut singleton_weight = Vec::with_capacity(n);
        let mut cut_weight = Vec::with_capacity(n);
        let mut split_delta_q_base = Vec::with_capacity(n);
        let mut split_delta_q_probe = Vec::with_capacity(n);
        let mut repair_quotient_edges = Vec::with_capacity(n);
        let mut repair_merge_count = Vec::with_capacity(n);
        let mut repair_delta_q = Vec::with_capacity(n);
        let mut net_delta_q = Vec::with_capacity(n);
        let mut final_source_units = Vec::with_capacity(n);
        let mut retained_source_units = Vec::with_capacity(n);
        let mut escaped_source_units = Vec::with_capacity(n);
        let mut escaped_source_weight = Vec::with_capacity(n);
        let mut final_small_source_units = Vec::with_capacity(n);
        let mut final_small_source_weight = Vec::with_capacity(n);
        let mut largest_source_unit_fraction = Vec::with_capacity(n);
        let mut restored_source_cluster = Vec::with_capacity(n);

        for probe in probes {
            cluster.push(probe.cluster);
            gamma_multiplier.push(probe.gamma_multiplier);
            probe_resolution.push(probe.probe_resolution);
            block_count.push(probe.block_count);
            doc_weight.push(probe.doc_weight);
            induced_directed_edges.push(probe.induced_directed_edges);
            n_parts.push(probe.n_parts);
            core_part_count.push(probe.core_part_count);
            singleton_weight.push(probe.singleton_weight);
            cut_weight.push(probe.cut_weight);
            split_delta_q_base.push(probe.split_delta_q_base);
            split_delta_q_probe.push(probe.split_delta_q_probe);
            repair_quotient_edges.push(probe.repair_quotient_edges);
            repair_merge_count.push(probe.repair_merge_count);
            repair_delta_q.push(probe.repair_delta_q);
            net_delta_q.push(probe.net_delta_q);
            final_source_units.push(probe.final_source_units);
            retained_source_units.push(probe.retained_source_units);
            escaped_source_units.push(probe.escaped_source_units);
            escaped_source_weight.push(probe.escaped_source_weight);
            final_small_source_units.push(probe.final_small_source_units);
            final_small_source_weight.push(probe.final_small_source_weight);
            largest_source_unit_fraction.push(probe.largest_source_unit_fraction);
            restored_source_cluster.push(probe.restored_source_cluster);
        }

        let mut out = std::collections::HashMap::new();
        out.insert(
            "cluster".to_string(),
            PyArray1::from_vec(py, cluster).into_any().unbind(),
        );
        out.insert(
            "gamma_multiplier".to_string(),
            PyArray1::from_vec(py, gamma_multiplier).into_any().unbind(),
        );
        out.insert(
            "probe_resolution".to_string(),
            PyArray1::from_vec(py, probe_resolution).into_any().unbind(),
        );
        out.insert(
            "block_count".to_string(),
            PyArray1::from_vec(py, block_count).into_any().unbind(),
        );
        out.insert(
            "doc_weight".to_string(),
            PyArray1::from_vec(py, doc_weight).into_any().unbind(),
        );
        out.insert(
            "induced_directed_edges".to_string(),
            PyArray1::from_vec(py, induced_directed_edges)
                .into_any()
                .unbind(),
        );
        out.insert(
            "n_parts".to_string(),
            PyArray1::from_vec(py, n_parts).into_any().unbind(),
        );
        out.insert(
            "core_part_count".to_string(),
            PyArray1::from_vec(py, core_part_count).into_any().unbind(),
        );
        out.insert(
            "singleton_weight".to_string(),
            PyArray1::from_vec(py, singleton_weight).into_any().unbind(),
        );
        out.insert(
            "cut_weight".to_string(),
            PyArray1::from_vec(py, cut_weight).into_any().unbind(),
        );
        out.insert(
            "split_delta_q_base".to_string(),
            PyArray1::from_vec(py, split_delta_q_base)
                .into_any()
                .unbind(),
        );
        out.insert(
            "split_delta_q_probe".to_string(),
            PyArray1::from_vec(py, split_delta_q_probe)
                .into_any()
                .unbind(),
        );
        out.insert(
            "repair_quotient_edges".to_string(),
            PyArray1::from_vec(py, repair_quotient_edges)
                .into_any()
                .unbind(),
        );
        out.insert(
            "repair_merge_count".to_string(),
            PyArray1::from_vec(py, repair_merge_count)
                .into_any()
                .unbind(),
        );
        out.insert(
            "repair_delta_q".to_string(),
            PyArray1::from_vec(py, repair_delta_q).into_any().unbind(),
        );
        out.insert(
            "net_delta_q".to_string(),
            PyArray1::from_vec(py, net_delta_q).into_any().unbind(),
        );
        out.insert(
            "final_source_units".to_string(),
            PyArray1::from_vec(py, final_source_units)
                .into_any()
                .unbind(),
        );
        out.insert(
            "retained_source_units".to_string(),
            PyArray1::from_vec(py, retained_source_units)
                .into_any()
                .unbind(),
        );
        out.insert(
            "escaped_source_units".to_string(),
            PyArray1::from_vec(py, escaped_source_units)
                .into_any()
                .unbind(),
        );
        out.insert(
            "escaped_source_weight".to_string(),
            PyArray1::from_vec(py, escaped_source_weight)
                .into_any()
                .unbind(),
        );
        out.insert(
            "final_small_source_units".to_string(),
            PyArray1::from_vec(py, final_small_source_units)
                .into_any()
                .unbind(),
        );
        out.insert(
            "final_small_source_weight".to_string(),
            PyArray1::from_vec(py, final_small_source_weight)
                .into_any()
                .unbind(),
        );
        out.insert(
            "largest_source_unit_fraction".to_string(),
            PyArray1::from_vec(py, largest_source_unit_fraction)
                .into_any()
                .unbind(),
        );
        out.insert(
            "restored_source_cluster".to_string(),
            PyArray1::from_vec(py, restored_source_cluster)
                .into_any()
                .unbind(),
        );
        Ok(out)
    }

    #[pyo3(signature = (
        membership,
        candidate_clusters,
        selected_clusters,
        selected_gamma_multipliers,
        resolution,
        gamma_multipliers,
        min_core_weight = 25.0,
        randomness = 0.01,
        repair_epsilon = 0.0,
        seed = 0,
        pair_seeded = false,
    ))]
    fn apply_split_merge_repair_candidates(
        &self,
        py: Python<'_>,
        membership: PyReadonlyArray1<u64>,
        candidate_clusters: PyReadonlyArray1<u64>,
        selected_clusters: PyReadonlyArray1<u64>,
        selected_gamma_multipliers: PyReadonlyArray1<f64>,
        resolution: f64,
        gamma_multipliers: PyReadonlyArray1<f64>,
        min_core_weight: f64,
        randomness: f64,
        repair_epsilon: f64,
        seed: u64,
        pair_seeded: bool,
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
        let candidate_clusters = candidate_clusters.as_slice()?.to_vec();
        let selected_clusters = selected_clusters.as_slice()?.to_vec();
        let selected_gamma_multipliers = selected_gamma_multipliers.as_slice()?.to_vec();
        let gamma_multipliers = gamma_multipliers.as_slice()?.to_vec();

        if selected_clusters.len() != selected_gamma_multipliers.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "selected_clusters length {} does not match selected_gamma_multipliers length {}",
                selected_clusters.len(),
                selected_gamma_multipliers.len(),
            )));
        }

        let (membership, applied) = py.allow_threads(|| {
            let mut ws = Workspace::new(self.graph.n_nodes.max(clustering.n_clusters));
            compute_apply_split_merge_repair_candidates(
                &self.graph,
                &clustering,
                &candidate_clusters,
                &selected_clusters,
                &selected_gamma_multipliers,
                resolution,
                &gamma_multipliers,
                min_core_weight,
                randomness,
                repair_epsilon,
                seed,
                pair_seeded,
                &mut ws,
            )
        });

        let n = applied.len();
        let mut selected_index = Vec::with_capacity(n);
        let mut cluster = Vec::with_capacity(n);
        let mut gamma_multiplier = Vec::with_capacity(n);
        let mut probe_resolution = Vec::with_capacity(n);
        let mut block_count = Vec::with_capacity(n);
        let mut doc_weight = Vec::with_capacity(n);
        let mut n_parts = Vec::with_capacity(n);
        let mut split_delta_q_base = Vec::with_capacity(n);
        let mut repair_delta_q = Vec::with_capacity(n);
        let mut predicted_net_delta_q = Vec::with_capacity(n);
        let mut repair_merge_count = Vec::with_capacity(n);
        let mut final_source_units = Vec::with_capacity(n);
        let mut retained_source_units = Vec::with_capacity(n);
        let mut escaped_source_units = Vec::with_capacity(n);
        let mut escaped_source_weight = Vec::with_capacity(n);
        let mut final_small_source_units = Vec::with_capacity(n);
        let mut final_small_source_weight = Vec::with_capacity(n);
        let mut largest_source_unit_fraction = Vec::with_capacity(n);
        let mut changed_nodes = Vec::with_capacity(n);
        let mut moved_to_existing_cluster_nodes = Vec::with_capacity(n);
        let mut moved_to_new_cluster_nodes = Vec::with_capacity(n);
        let mut new_retained_clusters = Vec::with_capacity(n);

        for row in applied {
            selected_index.push(row.selected_index);
            cluster.push(row.cluster);
            gamma_multiplier.push(row.gamma_multiplier);
            probe_resolution.push(row.probe_resolution);
            block_count.push(row.block_count);
            doc_weight.push(row.doc_weight);
            n_parts.push(row.n_parts);
            split_delta_q_base.push(row.split_delta_q_base);
            repair_delta_q.push(row.repair_delta_q);
            predicted_net_delta_q.push(row.predicted_net_delta_q);
            repair_merge_count.push(row.repair_merge_count);
            final_source_units.push(row.final_source_units);
            retained_source_units.push(row.retained_source_units);
            escaped_source_units.push(row.escaped_source_units);
            escaped_source_weight.push(row.escaped_source_weight);
            final_small_source_units.push(row.final_small_source_units);
            final_small_source_weight.push(row.final_small_source_weight);
            largest_source_unit_fraction.push(row.largest_source_unit_fraction);
            changed_nodes.push(row.changed_nodes);
            moved_to_existing_cluster_nodes.push(row.moved_to_existing_cluster_nodes);
            moved_to_new_cluster_nodes.push(row.moved_to_new_cluster_nodes);
            new_retained_clusters.push(row.new_retained_clusters);
        }

        let mut out = std::collections::HashMap::new();
        out.insert(
            "membership".to_string(),
            PyArray1::from_vec(py, membership).into_any().unbind(),
        );
        out.insert(
            "selected_index".to_string(),
            PyArray1::from_vec(py, selected_index).into_any().unbind(),
        );
        out.insert(
            "cluster".to_string(),
            PyArray1::from_vec(py, cluster).into_any().unbind(),
        );
        out.insert(
            "gamma_multiplier".to_string(),
            PyArray1::from_vec(py, gamma_multiplier).into_any().unbind(),
        );
        out.insert(
            "probe_resolution".to_string(),
            PyArray1::from_vec(py, probe_resolution).into_any().unbind(),
        );
        out.insert(
            "block_count".to_string(),
            PyArray1::from_vec(py, block_count).into_any().unbind(),
        );
        out.insert(
            "doc_weight".to_string(),
            PyArray1::from_vec(py, doc_weight).into_any().unbind(),
        );
        out.insert(
            "n_parts".to_string(),
            PyArray1::from_vec(py, n_parts).into_any().unbind(),
        );
        out.insert(
            "split_delta_q_base".to_string(),
            PyArray1::from_vec(py, split_delta_q_base)
                .into_any()
                .unbind(),
        );
        out.insert(
            "repair_delta_q".to_string(),
            PyArray1::from_vec(py, repair_delta_q).into_any().unbind(),
        );
        out.insert(
            "predicted_net_delta_q".to_string(),
            PyArray1::from_vec(py, predicted_net_delta_q)
                .into_any()
                .unbind(),
        );
        out.insert(
            "repair_merge_count".to_string(),
            PyArray1::from_vec(py, repair_merge_count)
                .into_any()
                .unbind(),
        );
        out.insert(
            "final_source_units".to_string(),
            PyArray1::from_vec(py, final_source_units)
                .into_any()
                .unbind(),
        );
        out.insert(
            "retained_source_units".to_string(),
            PyArray1::from_vec(py, retained_source_units)
                .into_any()
                .unbind(),
        );
        out.insert(
            "escaped_source_units".to_string(),
            PyArray1::from_vec(py, escaped_source_units)
                .into_any()
                .unbind(),
        );
        out.insert(
            "escaped_source_weight".to_string(),
            PyArray1::from_vec(py, escaped_source_weight)
                .into_any()
                .unbind(),
        );
        out.insert(
            "final_small_source_units".to_string(),
            PyArray1::from_vec(py, final_small_source_units)
                .into_any()
                .unbind(),
        );
        out.insert(
            "final_small_source_weight".to_string(),
            PyArray1::from_vec(py, final_small_source_weight)
                .into_any()
                .unbind(),
        );
        out.insert(
            "largest_source_unit_fraction".to_string(),
            PyArray1::from_vec(py, largest_source_unit_fraction)
                .into_any()
                .unbind(),
        );
        out.insert(
            "changed_nodes".to_string(),
            PyArray1::from_vec(py, changed_nodes).into_any().unbind(),
        );
        out.insert(
            "moved_to_existing_cluster_nodes".to_string(),
            PyArray1::from_vec(py, moved_to_existing_cluster_nodes)
                .into_any()
                .unbind(),
        );
        out.insert(
            "moved_to_new_cluster_nodes".to_string(),
            PyArray1::from_vec(py, moved_to_new_cluster_nodes)
                .into_any()
                .unbind(),
        );
        out.insert(
            "new_retained_clusters".to_string(),
            PyArray1::from_vec(py, new_retained_clusters)
                .into_any()
                .unbind(),
        );
        Ok(out)
    }

    #[pyo3(signature = (
        membership,
        resolution,
        target_max_weight,
        gamma_multipliers,
        policy = "quality_first",
        quality_floor_delta = 0.0,
        apply_iterations = 4,
        min_core_weight = 25.0,
        randomness = 0.01,
        repair_epsilon = 0.0,
        trim_min_delta_q_quality_first = 0.0,
        trim_min_delta_q_hard_cap = -1.0,
        trim_max_moves_per_cluster = 100,
        seed = 42,
        pair_seeded = true,
    ))]
    fn dongdaemun_refine(
        &self,
        py: Python<'_>,
        membership: PyReadonlyArray1<u64>,
        resolution: f64,
        target_max_weight: f64,
        gamma_multipliers: PyReadonlyArray1<f64>,
        policy: &str,
        quality_floor_delta: f64,
        apply_iterations: usize,
        min_core_weight: f64,
        randomness: f64,
        repair_epsilon: f64,
        trim_min_delta_q_quality_first: f64,
        trim_min_delta_q_hard_cap: f64,
        trim_max_moves_per_cluster: usize,
        seed: u64,
        pair_seeded: bool,
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

        let policy = parse_dongdaemun_policy(policy)?;
        let mut config = match policy {
            DongdaemunPolicy::QualityFirst => {
                DongdaemunConfig::default_for_quality_first(resolution, target_max_weight)
            }
            DongdaemunPolicy::HardCap => {
                DongdaemunConfig::default_for_hard_cap(resolution, target_max_weight)
            }
        };
        config.quality_floor_delta = quality_floor_delta;
        config.apply_iterations = apply_iterations;
        config.gamma_multipliers = gamma_multipliers.as_slice()?.to_vec();
        config.min_core_weight = min_core_weight;
        config.randomness = randomness;
        config.repair_epsilon = repair_epsilon;
        config.trim_min_delta_q_quality_first = trim_min_delta_q_quality_first;
        config.trim_min_delta_q_hard_cap = trim_min_delta_q_hard_cap;
        config.trim_max_moves_per_cluster = trim_max_moves_per_cluster;
        config.seed = seed;
        config.pair_seeded = pair_seeded;
        config
            .validate()
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

        let result = py.allow_threads(|| {
            let mut ws = Workspace::new(self.graph.n_nodes.max(clustering.n_clusters));
            compute_dongdaemun_refine(&self.graph, &clustering, &config, &mut ws)
        });

        let membership: Vec<u64> = result
            .clustering
            .clusters
            .iter()
            .map(|&cluster| u64::from(cluster))
            .collect();
        let (diagnostic_present, diagnostic_n_clusters, diagnostic_membership) =
            if let Some(diagnostic) = result.diagnostic_clustering.as_ref() {
                (
                    true,
                    diagnostic.n_clusters as u64,
                    diagnostic
                        .clusters
                        .iter()
                        .map(|&cluster| u64::from(cluster))
                        .collect::<Vec<_>>(),
                )
            } else {
                (false, 0u64, Vec::new())
            };

        let mut split_iteration = Vec::with_capacity(result.audit.split_iterations.len());
        let mut split_candidate_clusters = Vec::with_capacity(result.audit.split_iterations.len());
        let mut split_n_selected = Vec::with_capacity(result.audit.split_iterations.len());
        let mut split_n_applied = Vec::with_capacity(result.audit.split_iterations.len());
        let mut split_status_code = Vec::with_capacity(result.audit.split_iterations.len());
        let mut split_exact_delta_q = Vec::with_capacity(result.audit.split_iterations.len());
        for row in &result.audit.split_iterations {
            split_iteration.push(row.iteration as u64);
            split_candidate_clusters.push(row.candidate_clusters.len() as u64);
            split_n_selected.push(row.n_selected as u64);
            split_n_applied.push(row.n_applied as u64);
            split_status_code.push(dongdaemun_status_code(row.status));
            split_exact_delta_q.push(row.exact_delta_q);
        }

        let audit = result.audit;
        let mut out = std::collections::HashMap::new();
        out.insert(
            "membership".to_string(),
            PyArray1::from_vec(py, membership).into_any().unbind(),
        );
        out.insert(
            "n_clusters".to_string(),
            (result.clustering.n_clusters as u64)
                .into_pyobject(py)
                .unwrap()
                .into_any()
                .unbind(),
        );
        out.insert(
            "diagnostic_present".to_string(),
            (diagnostic_present as u8)
                .into_pyobject(py)
                .unwrap()
                .into_any()
                .unbind(),
        );
        out.insert(
            "diagnostic_membership".to_string(),
            PyArray1::from_vec(py, diagnostic_membership)
                .into_any()
                .unbind(),
        );
        out.insert(
            "diagnostic_n_clusters".to_string(),
            diagnostic_n_clusters
                .into_pyobject(py)
                .unwrap()
                .into_any()
                .unbind(),
        );
        out.insert(
            "accepted".to_string(),
            (audit.accepted as u8)
                .into_pyobject(py)
                .unwrap()
                .into_any()
                .unbind(),
        );
        out.insert(
            "status".to_string(),
            dongdaemun_status_str(audit.status)
                .to_string()
                .into_pyobject(py)
                .unwrap()
                .into_any()
                .unbind(),
        );
        out.insert(
            "quality_before".to_string(),
            audit
                .quality_before
                .into_pyobject(py)
                .unwrap()
                .into_any()
                .unbind(),
        );
        out.insert(
            "quality_after_candidate".to_string(),
            audit
                .quality_after_candidate
                .into_pyobject(py)
                .unwrap()
                .into_any()
                .unbind(),
        );
        out.insert(
            "candidate_delta_q".to_string(),
            audit
                .candidate_delta_q
                .into_pyobject(py)
                .unwrap()
                .into_any()
                .unbind(),
        );
        out.insert(
            "effective_delta_q".to_string(),
            audit
                .effective_delta_q
                .into_pyobject(py)
                .unwrap()
                .into_any()
                .unbind(),
        );
        out.insert(
            "final_delta_q".to_string(),
            audit
                .final_delta_q
                .into_pyobject(py)
                .unwrap()
                .into_any()
                .unbind(),
        );
        out.insert(
            "target_max_satisfied".to_string(),
            (audit.target_max_satisfied as u8)
                .into_pyobject(py)
                .unwrap()
                .into_any()
                .unbind(),
        );
        out.insert(
            "n_oversize_before".to_string(),
            (audit.n_oversize_before as u64)
                .into_pyobject(py)
                .unwrap()
                .into_any()
                .unbind(),
        );
        out.insert(
            "n_oversize_after_candidate".to_string(),
            (audit.n_oversize_after_candidate as u64)
                .into_pyobject(py)
                .unwrap()
                .into_any()
                .unbind(),
        );
        out.insert(
            "max_weight_before".to_string(),
            audit
                .max_weight_before
                .into_pyobject(py)
                .unwrap()
                .into_any()
                .unbind(),
        );
        out.insert(
            "max_weight_after_candidate".to_string(),
            audit
                .max_weight_after_candidate
                .into_pyobject(py)
                .unwrap()
                .into_any()
                .unbind(),
        );
        out.insert(
            "trim_moves_committed".to_string(),
            (audit.trim_moves_committed as u64)
                .into_pyobject(py)
                .unwrap()
                .into_any()
                .unbind(),
        );
        out.insert(
            "trim_moves_proposed".to_string(),
            (audit.trim_moves_proposed as u64)
                .into_pyobject(py)
                .unwrap()
                .into_any()
                .unbind(),
        );
        out.insert(
            "split_iteration".to_string(),
            PyArray1::from_vec(py, split_iteration).into_any().unbind(),
        );
        out.insert(
            "split_candidate_clusters".to_string(),
            PyArray1::from_vec(py, split_candidate_clusters)
                .into_any()
                .unbind(),
        );
        out.insert(
            "split_n_selected".to_string(),
            PyArray1::from_vec(py, split_n_selected).into_any().unbind(),
        );
        out.insert(
            "split_n_applied".to_string(),
            PyArray1::from_vec(py, split_n_applied).into_any().unbind(),
        );
        out.insert(
            "split_status_code".to_string(),
            PyArray1::from_vec(py, split_status_code)
                .into_any()
                .unbind(),
        );
        out.insert(
            "split_exact_delta_q".to_string(),
            PyArray1::from_vec(py, split_exact_delta_q)
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

#[cfg(all(test, feature = "python"))]
mod non_monotone_tests {
    use super::*;

    fn run_probe(
        graph: &Graph,
        clustering: &Clustering,
        quality_eps: f64,
    ) -> NonMonotoneGroupEscapeComputation {
        compute_non_monotone_group_escape_probe(
            graph,
            clustering,
            &[0],
            0.1,
            3,
            0,
            0.0,
            11,
            0.0,
            0.0,
            0.0,
            quality_eps,
            false,
        )
        .unwrap()
    }

    #[test]
    fn non_monotone_group_escape_max_candidates_zero_is_noop() {
        let graph = Graph::from_edge_list(3, &[0, 1], &[1, 2], &[0.1, 10.0]);
        let clustering = Clustering::from_assignments(vec![0, 0, 1]);

        let result = compute_non_monotone_group_escape_probe(
            &graph,
            &clustering,
            &[0],
            0.1,
            0,
            0,
            0.0,
            11,
            0.0,
            0.0,
            0.0,
            0.0,
            false,
        )
        .unwrap();

        assert!(!result.accepted, "{result:?}");
        assert_eq!(result.membership, vec![0, 0, 1]);
        assert_eq!(result.best_delta_q, 0.0);
        assert!(result.candidate_rows.is_empty());
    }

    #[test]
    fn non_monotone_group_escape_accepts_better_polished_candidate() {
        let graph = Graph::from_edge_list(3, &[0, 1], &[1, 2], &[0.1, 10.0]);
        let clustering = Clustering::from_assignments(vec![0, 0, 1]);

        let result = run_probe(&graph, &clustering, 0.0);

        assert!(result.accepted);
        assert_eq!(result.membership, vec![0, 1, 1]);
        assert!(result.quality > result.baseline_quality);
        assert!(result.best_delta_q > 0.0);
        assert_eq!(result.candidate_rows.len(), 1);
        assert!(result.candidate_rows[0].accepted_by_quality);
        assert!(result.candidate_rows[0].pre_polish_delta_q > 0.0);
    }

    #[test]
    fn non_monotone_group_escape_rejects_worse_polished_candidate() {
        let graph = Graph::from_edge_list(3, &[0, 1], &[1, 2], &[10.0, 0.1]);
        let clustering = Clustering::from_assignments(vec![0, 0, 1]);

        let result = run_probe(&graph, &clustering, 0.0);

        assert!(!result.accepted, "{result:?}");
        assert_eq!(result.membership, vec![0, 0, 1]);
        assert_eq!(result.quality, result.baseline_quality);
        assert!(result.best_delta_q < 0.0);
        assert_eq!(result.candidate_rows.len(), 1);
        assert!(!result.candidate_rows[0].accepted_by_quality);
        assert!(result.candidate_rows[0].post_polish_delta_q < 0.0);
    }

    #[test]
    fn non_monotone_group_escape_ranking_is_deterministic() {
        let graph = Graph::from_edge_list(5, &[0, 1, 1, 3], &[1, 2, 3, 4], &[0.1, 7.0, 6.0, 0.2]);
        let clustering = Clustering::from_assignments(vec![0, 0, 1, 2, 2]);

        let first = compute_non_monotone_group_escape_probe(
            &graph,
            &clustering,
            &[0, 2],
            0.1,
            5,
            0,
            0.0,
            99,
            0.0,
            0.0,
            0.0,
            0.0,
            false,
        )
        .unwrap();
        let second = compute_non_monotone_group_escape_probe(
            &graph,
            &clustering,
            &[0, 2],
            0.1,
            5,
            0,
            0.0,
            99,
            0.0,
            0.0,
            0.0,
            0.0,
            false,
        )
        .unwrap();

        let first_keys = first
            .candidate_rows
            .iter()
            .map(|row| (row.source_cluster, row.target_cluster, row.group_kind))
            .collect::<Vec<_>>();
        let second_keys = second
            .candidate_rows
            .iter()
            .map(|row| (row.source_cluster, row.target_cluster, row.group_kind))
            .collect::<Vec<_>>();
        assert_eq!(first_keys, second_keys);
        assert_eq!(first.membership, second.membership);
        assert_eq!(first.quality, second.quality);
    }

    #[test]
    fn non_monotone_group_escape_parallel_candidates_match_serial() {
        let graph = Graph::from_edge_list(5, &[0, 1, 1, 3], &[1, 2, 3, 4], &[0.1, 7.0, 6.0, 0.2]);
        let clustering = Clustering::from_assignments(vec![0, 0, 1, 2, 2]);

        let serial = compute_non_monotone_group_escape_probe(
            &graph,
            &clustering,
            &[0, 2],
            0.1,
            5,
            1,
            0.0,
            99,
            0.0,
            0.0,
            0.0,
            0.0,
            false,
        )
        .unwrap();
        let parallel = compute_non_monotone_group_escape_probe(
            &graph,
            &clustering,
            &[0, 2],
            0.1,
            5,
            1,
            0.0,
            99,
            0.0,
            0.0,
            0.0,
            0.0,
            true,
        )
        .unwrap();

        let serial_keys = serial
            .candidate_rows
            .iter()
            .map(|row| {
                (
                    row.candidate_index,
                    row.source_cluster,
                    row.target_cluster,
                    row.group_kind,
                )
            })
            .collect::<Vec<_>>();
        let parallel_keys = parallel
            .candidate_rows
            .iter()
            .map(|row| {
                (
                    row.candidate_index,
                    row.source_cluster,
                    row.target_cluster,
                    row.group_kind,
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(serial_keys, parallel_keys);
        assert_eq!(serial.membership, parallel.membership);
        assert_eq!(serial.quality, parallel.quality);
        assert_eq!(serial.best_delta_q, parallel.best_delta_q);
        assert!(parallel.candidate_eval_cpu_sum_elapsed_ms >= 0.0);
        assert!(parallel.candidate_eval_wall_elapsed_ms >= 0.0);
        if parallel.candidate_rows.len() > 1 {
            assert!(parallel.candidate_eval_parallel);
            assert!(parallel.candidate_eval_parallel_workers >= 1);
        }
    }

    #[test]
    fn non_monotone_group_escape_can_omit_return_membership() {
        let graph = Graph::from_edge_list(5, &[0, 1, 1, 3], &[1, 2, 3, 4], &[0.1, 7.0, 6.0, 0.2]);
        let clustering = Clustering::from_assignments(vec![0, 0, 1, 2, 2]);

        let with_membership = compute_non_monotone_group_escape_probe_impl(
            &graph,
            &clustering,
            &[0, 2],
            0.1,
            5,
            1,
            0.0,
            99,
            0.0,
            0.0,
            0.0,
            0.0,
            true,
            true,
        )
        .unwrap();
        let without_membership = compute_non_monotone_group_escape_probe_impl(
            &graph,
            &clustering,
            &[0, 2],
            0.1,
            5,
            1,
            0.0,
            99,
            0.0,
            0.0,
            0.0,
            0.0,
            true,
            false,
        )
        .unwrap();

        assert!(without_membership.membership.is_empty());
        assert_eq!(with_membership.quality, without_membership.quality);
        assert_eq!(with_membership.accepted, without_membership.accepted);
        assert_eq!(
            with_membership.candidate_rows.len(),
            without_membership.candidate_rows.len()
        );
        assert_eq!(
            with_membership.best_delta_q,
            without_membership.best_delta_q
        );
    }

    #[test]
    fn non_monotone_multifidelity_ranking_matches_full_probe() {
        let graph = Graph::from_edge_list(5, &[0, 1, 1, 3], &[1, 2, 3, 4], &[0.1, 7.0, 6.0, 0.2]);
        let clustering = Clustering::from_assignments(vec![0, 0, 1, 2, 2]);

        let full = compute_non_monotone_group_escape_probe(
            &graph,
            &clustering,
            &[0, 2],
            0.1,
            5,
            0,
            0.0,
            99,
            0.0,
            0.0,
            0.0,
            0.0,
            false,
        )
        .unwrap();
        let multifidelity = compute_non_monotone_group_escape_multifidelity_probe(
            &graph,
            &clustering,
            &[0, 2],
            0.1,
            5,
            0,
            0,
            1,
            true,
            0.0,
            99,
            0.0,
            0.0,
            0.0,
            0.0,
            false,
            false,
        )
        .unwrap();

        let full_keys = full
            .candidate_rows
            .iter()
            .map(|row| (row.source_cluster, row.target_cluster, row.group_kind))
            .collect::<Vec<_>>();
        let multifidelity_keys = multifidelity
            .candidate_rows
            .iter()
            .map(|row| (row.source_cluster, row.target_cluster, row.group_kind))
            .collect::<Vec<_>>();
        assert_eq!(full_keys, multifidelity_keys);
    }

    #[test]
    fn non_monotone_multifidelity_records_selection_flags_and_policies() {
        let graph = Graph::from_edge_list(3, &[0, 1], &[1, 2], &[0.1, 10.0]);
        let clustering = Clustering::from_assignments(vec![0, 0, 1]);

        let result = compute_non_monotone_group_escape_multifidelity_probe(
            &graph,
            &clustering,
            &[0],
            0.1,
            3,
            0,
            0,
            1,
            true,
            0.0,
            11,
            0.0,
            0.0,
            0.0,
            0.0,
            false,
            false,
        )
        .unwrap();

        assert_eq!(result.candidate_rows.len(), 1);
        let row = &result.candidate_rows[0];
        assert!(row.selected_by_p1_top1);
        assert!(row.selected_by_p1_top2);
        assert!(row.selected_by_full_p5);
        assert!(row.p1_elapsed_ms >= 0.0);
        assert!(row.p5_elapsed_ms >= 0.0);
        assert_eq!(result.selected_policy, "p1_top1_then_p5");
        assert_eq!(result.selected_candidate_index, 0);
        assert!(result.accepted);
        assert!(result
            .policy_rows
            .iter()
            .any(|row| row.policy == "full_top3_p5" && row.available));
        assert!(result
            .policy_rows
            .iter()
            .any(|row| row.policy == "p1_top1_then_p5" && row.available));
    }

    #[test]
    fn non_monotone_multifidelity_max_candidates_zero_is_noop() {
        let graph = Graph::from_edge_list(3, &[0, 1], &[1, 2], &[0.1, 10.0]);
        let clustering = Clustering::from_assignments(vec![0, 0, 1]);

        let result = compute_non_monotone_group_escape_multifidelity_probe(
            &graph,
            &clustering,
            &[0],
            0.1,
            0,
            1,
            5,
            1,
            true,
            0.0,
            11,
            0.0,
            0.0,
            0.0,
            0.0,
            false,
            false,
        )
        .unwrap();

        assert!(!result.accepted);
        assert_eq!(result.membership, vec![0, 0, 1]);
        assert_eq!(result.best_delta_q, 0.0);
        assert!(result.candidate_rows.is_empty());
        assert!(result.policy_rows.is_empty());
    }

    #[test]
    fn non_monotone_multifidelity_records_basin_signature_without_return_membership() {
        let graph = Graph::from_edge_list(3, &[0, 1], &[1, 2], &[0.1, 10.0]);
        let clustering = Clustering::from_assignments(vec![0, 0, 1]);

        let result = compute_non_monotone_group_escape_multifidelity_probe_impl(
            &graph,
            &clustering,
            &[0],
            0.05,
            3,
            0,
            0,
            1,
            true,
            0.0,
            11,
            0.0,
            0.0,
            0.0,
            0.0,
            false,
            false,
            true,
        )
        .unwrap();

        assert!(result.membership.is_empty());
        assert_eq!(result.candidate_rows.len(), 1);
        let basin = &result.candidate_rows[0].basin;
        assert!(!basin.signature.is_empty());
        assert_eq!(basin.cluster_count, 2);
        assert_eq!(basin.changed_nodes_vs_baseline, 1);
        assert_eq!(basin.baseline_fragmentation_nodes, 1);
        assert_eq!(basin.baseline_mixing_nodes, 1);
        assert!(basin.changed_fraction_vs_baseline > 0.0);
        assert!(basin.relative_delta_q_ppm.is_finite());
        assert_eq!(basin.sketch_sample_size, 3);
        assert!(!basin.sketch_node_hash.is_empty());
        assert_eq!(basin.sketch_baseline_membership, "0;0;1");
        assert_eq!(basin.sketch_membership, "0;1;1");
        assert_eq!(basin.changed_support_node_count, 2);
        assert_eq!(basin.changed_support_sketch_sample_size, 2);
        assert!(!basin.changed_support_node_hash.is_empty());
        assert_eq!(basin.changed_support_nodes, "1;2");
    }

    #[test]
    fn non_monotone_multifidelity_can_omit_return_membership() {
        let graph = Graph::from_edge_list(3, &[0, 1], &[1, 2], &[0.1, 10.0]);
        let clustering = Clustering::from_assignments(vec![0, 0, 1]);

        let with_membership = compute_non_monotone_group_escape_multifidelity_probe_impl(
            &graph,
            &clustering,
            &[0],
            0.1,
            3,
            0,
            0,
            1,
            true,
            0.0,
            11,
            0.0,
            0.0,
            0.0,
            0.0,
            true,
            false,
            false,
        )
        .unwrap();
        let without_membership = compute_non_monotone_group_escape_multifidelity_probe_impl(
            &graph,
            &clustering,
            &[0],
            0.1,
            3,
            0,
            0,
            1,
            true,
            0.0,
            11,
            0.0,
            0.0,
            0.0,
            0.0,
            false,
            false,
            false,
        )
        .unwrap();

        assert!(without_membership.membership.is_empty());
        assert_eq!(with_membership.quality, without_membership.quality);
        assert_eq!(with_membership.accepted, without_membership.accepted);
        assert_eq!(
            with_membership.selected_policy,
            without_membership.selected_policy
        );
        assert_eq!(
            with_membership.selected_candidate_index,
            without_membership.selected_candidate_index
        );
        assert_eq!(
            with_membership.candidate_rows.len(),
            without_membership.candidate_rows.len()
        );
        assert_eq!(
            with_membership.policy_rows.len(),
            without_membership.policy_rows.len()
        );
    }
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
