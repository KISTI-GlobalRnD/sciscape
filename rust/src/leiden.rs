//! Leiden algorithm: move → refine → aggregate → recurse.
//!
//! Port of CWTS LeidenAlgorithm.java.

use crate::clustering::Clustering;
use crate::contraction::{create_reduced_network, create_reduced_network_from_starts};
use crate::fast_local_move;
use crate::graph::Graph;
use crate::local_merge;
use crate::trace;
use crate::workspace::Workspace;
use rand::Rng;
use std::sync::OnceLock;
use std::time::Instant;

const DEFAULT_STREAMING_REFINEMENT_MIN_DIRECTED_EDGES: usize = 1_000_000;
const DEFAULT_CONVERGENCE_PATIENCE: usize = 3;
const DEFAULT_CONVERGENCE_MIN_REL_CLUSTER_DELTA: f64 = 1e-4;

fn streaming_refinement_min_directed_edges() -> usize {
    static MIN_EDGES: OnceLock<usize> = OnceLock::new();
    *MIN_EDGES.get_or_init(|| {
        std::env::var("SCISCAPE_STREAMING_REFINEMENT_MIN_DIRECTED_EDGES")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(DEFAULT_STREAMING_REFINEMENT_MIN_DIRECTED_EDGES)
    })
}

fn convergence_patience() -> usize {
    static PATIENCE: OnceLock<usize> = OnceLock::new();
    *PATIENCE.get_or_init(|| {
        std::env::var("SCISCAPE_LEIDEN_CONVERGENCE_PATIENCE")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(DEFAULT_CONVERGENCE_PATIENCE)
    })
}

fn convergence_min_rel_cluster_delta() -> f64 {
    static MIN_DELTA: OnceLock<f64> = OnceLock::new();
    *MIN_DELTA.get_or_init(|| {
        std::env::var("SCISCAPE_LEIDEN_CONVERGENCE_MIN_REL_CLUSTER_DELTA")
            .ok()
            .and_then(|value| value.parse::<f64>().ok())
            .filter(|value| value.is_finite() && *value >= 0.0)
            .unwrap_or(DEFAULT_CONVERGENCE_MIN_REL_CLUSTER_DELTA)
    })
}

fn should_trace_graph(graph: &Graph) -> bool {
    trace::should_trace_edges(graph.n_edges)
}

fn should_trace_graph_detail(_graph: &Graph) -> bool {
    trace::verbose()
}

macro_rules! trace_graph {
    ($graph:expr, $($arg:tt)*) => {{
        if should_trace_graph_detail($graph) {
            trace::emit(format_args!($($arg)*));
        }
    }};
}

macro_rules! trace_graph_summary {
    ($graph:expr, $($arg:tt)*) => {{
        if should_trace_graph($graph) {
            trace::emit(format_args!($($arg)*));
        }
    }};
}

/// Configuration for the Leiden algorithm.
#[derive(Clone, Debug)]
pub struct LeidenConfig {
    pub resolution: f64,
    pub n_iterations: usize, // 0 = until convergence
    pub randomness: f64,
    pub randomness_schedule: Vec<f64>,
    pub seed: u64,
}

impl Default for LeidenConfig {
    fn default() -> Self {
        LeidenConfig {
            resolution: 1.0,
            n_iterations: 10,
            randomness: 0.01,
            randomness_schedule: Vec::new(),
            seed: 0,
        }
    }
}

impl LeidenConfig {
    #[inline]
    fn randomness_for_iteration(&self, iteration: usize) -> f64 {
        if self.randomness_schedule.is_empty() {
            self.randomness
        } else {
            self.randomness_schedule[iteration.min(self.randomness_schedule.len() - 1)]
        }
    }
}

/// Result of a Leiden run.
#[derive(Clone, Debug)]
pub struct LeidenResult {
    pub clustering: Clustering,
    pub quality: f64,
    pub n_iterations_used: usize,
}

#[derive(Clone, Copy, Debug, Default)]
struct IterationStats {
    improved: bool,
    moved_nodes: usize,
}

struct RefinementResult {
    clustering: Clustering,
    parent_clusters: Vec<u32>,
    cluster_starts: Vec<u32>,
    fixed: Option<Vec<bool>>,
}

enum IterationStep {
    Done {
        stats: IterationStats,
    },
    NonRefined {
        local_stats: IterationStats,
        reduced: Graph,
        reduced_clustering: Clustering,
        parent_nodes: usize,
        reduced_nodes: usize,
        trace_detail: bool,
    },
    Refined {
        local_stats: IterationStats,
        reduced: Graph,
        reduced_clustering: Clustering,
        refinement_clustering: Clustering,
        parent_nodes: usize,
        reduced_nodes: usize,
        trace_detail: bool,
    },
}

fn empty_refinement(n_nodes: usize) -> Clustering {
    Clustering {
        n_nodes,
        n_clusters: 0,
        clusters: vec![0; n_nodes],
        fixed: None,
    }
}

fn counts_to_starts(mut counts: Vec<u32>) -> Vec<u32> {
    let mut running = 0u32;
    for count in counts.iter_mut() {
        let next = running + *count;
        *count = running;
        running = next;
    }
    counts.push(running);
    counts
}

fn trace_contraction_progress(
    graph: &Graph,
    reduced: &Graph,
    depth: usize,
    mode: &str,
    elapsed_ms: f64,
) {
    let node_delta = graph.n_nodes.saturating_sub(reduced.n_nodes);
    let edge_delta = graph.n_edges.saturating_sub(reduced.n_edges);
    let rel_node_delta = node_delta as f64 / graph.n_nodes.max(1) as f64;
    let rel_edge_delta = edge_delta as f64 / graph.n_edges.max(1) as f64;
    trace_graph_summary!(
        graph,
        "phase=leiden_contraction depth={} mode={} input_nodes={} input_directed_edges={} reduced_nodes={} reduced_directed_edges={} node_delta={} rel_node_delta={:.8e} edge_delta={} rel_edge_delta={:.8e} elapsed_ms={:.1}",
        depth,
        mode,
        graph.n_nodes,
        graph.n_edges,
        reduced.n_nodes,
        reduced.n_edges,
        node_delta,
        rel_node_delta,
        edge_delta,
        rel_edge_delta,
        elapsed_ms,
    );
}

/// Run the Leiden algorithm on a graph.
///
/// If `initial` is None, starts from singleton clustering.
/// Returns the final clustering and quality.
pub fn leiden(
    graph: &Graph,
    config: &LeidenConfig,
    initial: Option<Clustering>,
    rng: &mut impl Rng,
) -> LeidenResult {
    let mut ws = Workspace::new(graph.n_nodes);
    leiden_with_workspace(graph, config, initial, rng, &mut ws)
}

pub(crate) fn leiden_with_workspace(
    graph: &Graph,
    config: &LeidenConfig,
    initial: Option<Clustering>,
    rng: &mut impl Rng,
    ws: &mut Workspace,
) -> LeidenResult {
    let trace_run = should_trace_graph(graph);
    let run_start = Instant::now();
    if trace_run {
        trace::emit(format_args!(
            "phase=leiden_start nodes={} directed_edges={} resolution={:.8} n_iterations={} randomness={:.6} seed={}{}",
            graph.n_nodes,
            graph.n_edges,
            config.resolution,
            config.n_iterations,
            config.randomness,
            config.seed,
            trace::memory_fields(),
        ));
    }

    let mut clustering = initial.unwrap_or_else(|| Clustering::singleton(graph.n_nodes));

    let mut n_used = 0;
    if config.n_iterations > 0 {
        for _ in 0..config.n_iterations {
            let iter_start = Instant::now();
            let iter_randomness = config.randomness_for_iteration(n_used);
            let stats =
                improve_one_iteration(graph, &mut clustering, config, iter_randomness, rng, ws);
            n_used += 1;
            trace_graph_summary!(
                graph,
                "phase=leiden_iteration iter={} randomness={:.6} improved={} moved_nodes={} clusters={} elapsed_ms={:.1}",
                n_used,
                iter_randomness,
                stats.improved,
                stats.moved_nodes,
                clustering.n_clusters,
                iter_start.elapsed().as_secs_f64() * 1000.0,
            );
            if !stats.improved {
                break;
            }
        }
    } else {
        let min_rel_cluster_delta = convergence_min_rel_cluster_delta();
        let patience = convergence_patience();
        let mut previous_n_clusters = clustering.n_clusters;
        let mut stagnant_iterations = 0usize;
        loop {
            let iter_start = Instant::now();
            let iter_randomness = config.randomness_for_iteration(n_used);
            let stats =
                improve_one_iteration(graph, &mut clustering, config, iter_randomness, rng, ws);
            n_used += 1;
            let cluster_delta = previous_n_clusters.abs_diff(clustering.n_clusters);
            let rel_cluster_delta =
                cluster_delta as f64 / previous_n_clusters.max(clustering.n_clusters).max(1) as f64;
            trace_graph_summary!(
                graph,
                "phase=leiden_iteration iter={} randomness={:.6} improved={} moved_nodes={} clusters={} cluster_delta={} rel_cluster_delta={:.8e} stagnant_iterations={} elapsed_ms={:.1}",
                n_used,
                iter_randomness,
                stats.improved,
                stats.moved_nodes,
                clustering.n_clusters,
                cluster_delta,
                rel_cluster_delta,
                stagnant_iterations,
                iter_start.elapsed().as_secs_f64() * 1000.0,
            );
            if !stats.improved {
                break;
            }
            if patience > 0 && rel_cluster_delta <= min_rel_cluster_delta {
                stagnant_iterations += 1;
                if stagnant_iterations >= patience {
                    trace_graph_summary!(
                        graph,
                        "phase=leiden_convergence_stop iter={} reason=cluster_delta rel_cluster_delta={:.8e} threshold={:.8e} patience={}",
                        n_used,
                        rel_cluster_delta,
                        min_rel_cluster_delta,
                        patience,
                    );
                    break;
                }
            } else {
                stagnant_iterations = 0;
            }
            previous_n_clusters = clustering.n_clusters;
        }
    }

    let quality = crate::quality::CPM::new(config.resolution).quality(graph, &clustering);

    if trace_run {
        trace::emit(format_args!(
            "phase=leiden_done nodes={} directed_edges={} clusters={} quality={:.6} iterations_used={} elapsed_ms={:.1}{}",
            graph.n_nodes,
            graph.n_edges,
            clustering.n_clusters,
            quality,
            n_used,
            run_start.elapsed().as_secs_f64() * 1000.0,
            trace::memory_fields(),
        ));
    }

    LeidenResult {
        clustering,
        quality,
        n_iterations_used: n_used,
    }
}

/// Run Leiden with multiple random starts, return best quality.
/// Uses rayon for parallel execution when n_starts > 1.
pub fn leiden_multi_start(
    graph: &Graph,
    config: &LeidenConfig,
    n_starts: usize,
    initial: Option<&Clustering>,
) -> LeidenResult {
    use rayon::prelude::*;

    if n_starts <= 1 {
        let mut rng = rand::rngs::StdRng::seed_from_u64(config.seed);
        let init = initial.cloned();
        return leiden(graph, config, init, &mut rng);
    }

    let results: Vec<LeidenResult> = (0..n_starts)
        .into_par_iter()
        .map(|start| {
            let mut rng = rand::rngs::StdRng::seed_from_u64(config.seed + start as u64);
            let init = initial.cloned();
            leiden(graph, config, init, &mut rng)
        })
        .collect();

    results
        .into_iter()
        .max_by(|a, b| a.quality.total_cmp(&b.quality))
        .unwrap()
}

/// One iteration of Leiden: move → refine → aggregate → recurse.
fn improve_one_iteration(
    graph: &Graph,
    clustering: &mut Clustering,
    config: &LeidenConfig,
    randomness: f64,
    rng: &mut impl Rng,
    ws: &mut Workspace,
) -> IterationStats {
    let step = prepare_iteration_step(graph, clustering, config, randomness, rng, ws, 0);
    finish_iteration_step(step, clustering, config, randomness, rng, ws, 0)
}

/// Recursive Leiden iteration for reduced graphs owned by the current frame.
///
/// The borrowed root graph must stay alive for quality computation and API
/// ownership, but reduced graphs do not need to survive while their own reduced
/// child is processed. Passing them by value lets us explicitly drop the parent
/// reduced graph before descending further. This matches the liveness behavior
/// Java can get from GC/JIT and prevents near-identity contractions from
/// accumulating one full CSR per recursion level.
fn improve_one_iteration_owned(
    graph: Graph,
    clustering: &mut Clustering,
    config: &LeidenConfig,
    randomness: f64,
    rng: &mut impl Rng,
    ws: &mut Workspace,
    depth: usize,
) -> IterationStats {
    let step = prepare_iteration_step(&graph, clustering, config, randomness, rng, ws, depth);
    drop(graph);
    finish_iteration_step(step, clustering, config, randomness, rng, ws, depth)
}

fn prepare_iteration_step(
    graph: &Graph,
    clustering: &mut Clustering,
    config: &LeidenConfig,
    randomness: f64,
    rng: &mut impl Rng,
    ws: &mut Workspace,
    depth: usize,
) -> IterationStep {
    let trace_detail = should_trace_graph_detail(graph);
    let parent_nodes = graph.n_nodes;

    // Phase 1: Local moving
    let t_move = Instant::now();
    let local_move =
        fast_local_move::improve_clustering(graph, clustering, config.resolution, rng, ws);
    let local_stats = IterationStats {
        improved: local_move.improved,
        moved_nodes: local_move.moved_nodes,
    };
    trace_graph!(
        graph,
        "phase=local_move depth={} nodes={} directed_edges={} clusters={} updated={} moved_nodes={} elapsed_ms={:.1}",
        depth,
        graph.n_nodes,
        graph.n_edges,
        clustering.n_clusters,
        local_stats.improved,
        local_stats.moved_nodes,
        t_move.elapsed().as_secs_f64() * 1000.0,
    );

    // If every node is its own cluster, nothing to do
    if clustering.n_clusters >= graph.n_nodes {
        return IterationStep::Done { stats: local_stats };
    }

    // Phase 2: Refinement
    let use_streaming_refinement = graph.n_edges >= streaming_refinement_min_directed_edges();
    let t_nodes = Instant::now();
    let refinement = if use_streaming_refinement {
        clustering.fill_cluster_groups(ws);
        trace_graph!(
            graph,
            "phase=nodes_per_cluster depth={} mode=flat clusters={} elapsed_ms={:.1}",
            depth,
            clustering.n_clusters,
            t_nodes.elapsed().as_secs_f64() * 1000.0,
        );
        let t_refine = Instant::now();
        let refinement = {
            let starts = &ws.npc_starts[..clustering.n_clusters + 1];
            let nodes = &ws.npc_nodes[..graph.n_nodes];
            let local_index = &mut ws.local_index[..graph.n_nodes];
            refine_streaming_flat(
                graph,
                clustering,
                clustering.n_clusters,
                starts,
                nodes,
                local_index,
                config,
                randomness,
                rng,
            )
        };
        trace_graph!(
            graph,
            "phase=refinement depth={} mode=streaming refined_clusters={} elapsed_ms={:.1}",
            depth,
            refinement.clustering.n_clusters,
            t_refine.elapsed().as_secs_f64() * 1000.0,
        );
        refinement
    } else {
        let nodes_per_cluster = clustering.nodes_per_cluster();
        trace_graph!(
            graph,
            "phase=nodes_per_cluster depth={} mode=vec clusters={} elapsed_ms={:.1}",
            depth,
            nodes_per_cluster.len(),
            t_nodes.elapsed().as_secs_f64() * 1000.0,
        );
        let t_refine = Instant::now();
        let refinement = refine_eager(
            graph,
            clustering,
            &nodes_per_cluster,
            config,
            randomness,
            rng,
        );
        trace_graph!(
            graph,
            "phase=refinement depth={} mode=eager refined_clusters={} elapsed_ms={:.1}",
            depth,
            refinement.clustering.n_clusters,
            t_refine.elapsed().as_secs_f64() * 1000.0,
        );
        refinement
    };

    if refinement.clustering.n_clusters >= graph.n_nodes {
        // Refinement produced singletons — aggregate on non-refined clustering
        let t_contract = Instant::now();
        let reduced = create_reduced_network(graph, clustering, true, ws);
        let contract_elapsed_ms = t_contract.elapsed().as_secs_f64() * 1000.0;
        trace_contraction_progress(graph, &reduced, depth, "non_refined", contract_elapsed_ms);
        trace_graph!(
            graph,
            "phase=contract_non_refined depth={} reduced_nodes={} reduced_directed_edges={} elapsed_ms={:.1}",
            depth,
            reduced.n_nodes,
            reduced.n_edges,
            contract_elapsed_ms,
        );
        let mut reduced_clustering = Clustering::singleton(reduced.n_nodes);

        // Propagate fixed status to reduced graph
        if clustering.fixed.is_some() {
            let mut rf = vec![false; clustering.n_clusters];
            for i in 0..graph.n_nodes {
                if clustering.is_fixed(i) {
                    rf[clustering.clusters[i] as usize] = true;
                }
            }
            reduced_clustering.fixed = Some(rf);
        }

        let reduced_nodes = reduced.n_nodes;
        return IterationStep::NonRefined {
            local_stats,
            reduced,
            reduced_clustering,
            parent_nodes,
            reduced_nodes,
            trace_detail,
        };
    }

    // Phase 3: Aggregate based on refined clustering
    let t_contract = Instant::now();
    let reduced = create_reduced_network_from_starts(
        graph,
        &refinement.clustering,
        &refinement.cluster_starts,
        true,
        ws,
    );
    let contract_elapsed_ms = t_contract.elapsed().as_secs_f64() * 1000.0;
    trace_contraction_progress(graph, &reduced, depth, "refined", contract_elapsed_ms);
    trace_graph!(
        graph,
        "phase=contract_refined depth={} reduced_nodes={} reduced_directed_edges={} elapsed_ms={:.1}",
        depth,
        reduced.n_nodes,
        reduced.n_edges,
        contract_elapsed_ms,
    );

    // Initial clustering for aggregate network: map non-refined clusters
    // to the move-phase cluster assignments (before refinement).
    // Each refined sub-cluster inherits the move-phase cluster ID of its parent.
    let reduced_n_clusters = refinement
        .parent_clusters
        .iter()
        .copied()
        .max()
        .map_or(0, |max_cid| max_cid as usize + 1);

    let reduced_clustering = Clustering {
        n_nodes: refinement.clustering.n_clusters,
        n_clusters: reduced_n_clusters,
        clusters: refinement.parent_clusters,
        fixed: refinement.fixed,
    };

    let reduced_nodes = reduced.n_nodes;
    IterationStep::Refined {
        local_stats,
        reduced,
        reduced_clustering,
        refinement_clustering: refinement.clustering,
        parent_nodes,
        reduced_nodes,
        trace_detail,
    }
}

fn finish_iteration_step(
    step: IterationStep,
    clustering: &mut Clustering,
    config: &LeidenConfig,
    randomness: f64,
    rng: &mut impl Rng,
    ws: &mut Workspace,
    depth: usize,
) -> IterationStats {
    match step {
        IterationStep::Done { stats } => stats,
        IterationStep::NonRefined {
            local_stats,
            reduced,
            mut reduced_clustering,
            parent_nodes,
            reduced_nodes,
            trace_detail,
        } => {
            let t_recurse = Instant::now();
            let recursive_stats = improve_one_iteration_owned(
                reduced,
                &mut reduced_clustering,
                config,
                randomness,
                rng,
                ws,
                depth + 1,
            );
            if trace_detail {
                trace::emit(format_args!(
                    "phase=recurse_non_refined depth={} reduced_nodes={} improved={} moved_nodes={} elapsed_ms={:.1}",
                    depth,
                    reduced_nodes,
                    recursive_stats.improved,
                    recursive_stats.moved_nodes,
                    t_recurse.elapsed().as_secs_f64() * 1000.0,
                ));
            }
            if recursive_stats.improved {
                let t_merge = Instant::now();
                clustering.merge_clusters(&reduced_clustering);
                if trace_detail {
                    trace::emit(format_args!(
                        "phase=merge_non_refined depth={} nodes={} elapsed_ms={:.1}",
                        depth,
                        parent_nodes,
                        t_merge.elapsed().as_secs_f64() * 1000.0,
                    ));
                }
            }
            IterationStats {
                improved: local_stats.improved | recursive_stats.improved,
                moved_nodes: local_stats
                    .moved_nodes
                    .saturating_add(recursive_stats.moved_nodes),
            }
        }
        IterationStep::Refined {
            local_stats,
            reduced,
            mut reduced_clustering,
            refinement_clustering,
            parent_nodes,
            reduced_nodes,
            trace_detail,
        } => {
            let t_recurse = Instant::now();
            let recursive_stats = improve_one_iteration_owned(
                reduced,
                &mut reduced_clustering,
                config,
                randomness,
                rng,
                ws,
                depth + 1,
            );
            if trace_detail {
                trace::emit(format_args!(
                    "phase=recurse_refined depth={} reduced_nodes={} improved={} moved_nodes={} elapsed_ms={:.1}",
                    depth,
                    reduced_nodes,
                    recursive_stats.improved,
                    recursive_stats.moved_nodes,
                    t_recurse.elapsed().as_secs_f64() * 1000.0,
                ));
            }

            // Merge back only if the recursive reduced graph actually changed.
            // When recursion reports no improvement, `reduced_clustering` is
            // equivalent to the initial projection and merge-back would only
            // restore the move-phase clustering after a full O(n) scan.
            if recursive_stats.improved {
                let t_merge = Instant::now();
                clustering.clusters = refinement_clustering.clusters;
                clustering.n_clusters = refinement_clustering.n_clusters;
                clustering.merge_clusters(&reduced_clustering);
                if trace_detail {
                    trace::emit(format_args!(
                        "phase=merge_refined depth={} nodes={} elapsed_ms={:.1}",
                        depth,
                        parent_nodes,
                        t_merge.elapsed().as_secs_f64() * 1000.0,
                    ));
                }
            }

            IterationStats {
                improved: local_stats.improved | recursive_stats.improved,
                moved_nodes: local_stats
                    .moved_nodes
                    .saturating_add(recursive_stats.moved_nodes),
            }
        }
    }
}

fn refine_eager(
    graph: &Graph,
    clustering: &Clustering,
    nodes_per_cluster: &[Vec<usize>],
    config: &LeidenConfig,
    randomness: f64,
    rng: &mut impl Rng,
) -> RefinementResult {
    // Extract all cluster subnetworks in one O(n+m) pass. Fast for small and
    // medium graphs, but memory-heavy at large scale.
    let subnetworks = graph.create_subnetworks(nodes_per_cluster);
    let mut refinement = empty_refinement(graph.n_nodes);
    let mut parent_clusters = Vec::with_capacity(nodes_per_cluster.len());
    let mut cluster_counts = Vec::with_capacity(nodes_per_cluster.len());
    let mut reduced_fixed = clustering
        .fixed
        .as_ref()
        .map(|_| Vec::with_capacity(nodes_per_cluster.len()));
    let mut merge_ws = local_merge::LocalMergeWorkspace::new(0);
    let fixed = clustering.fixed.as_deref();

    for (cid, (subgraph, nodes)) in subnetworks.iter().enumerate() {
        if nodes.is_empty() {
            continue;
        }

        if let Some(fixed) = fixed {
            if nodes.iter().any(|&n| fixed[n]) {
                for &node in nodes {
                    refinement.clusters[node] = refinement.n_clusters as u32;
                }
                parent_clusters.push(cid as u32);
                cluster_counts.push(nodes.len() as u32);
                if let Some(rf) = reduced_fixed.as_mut() {
                    rf.push(true);
                }
                refinement.n_clusters += 1;
                continue;
            }
        }

        {
            let sub_n_clusters =
                local_merge::find_clustering_with_workspace_assignments_and_append_sizes(
                    subgraph,
                    config.resolution,
                    randomness,
                    rng,
                    &mut merge_ws,
                    &mut cluster_counts,
                );
            let sub_clusters = merge_ws.assignments();

            for (local_idx, &node) in nodes.iter().enumerate() {
                refinement.clusters[node] = refinement.n_clusters as u32 + sub_clusters[local_idx];
            }
            parent_clusters.resize(parent_clusters.len() + sub_n_clusters, cid as u32);
            if let Some(rf) = reduced_fixed.as_mut() {
                rf.resize(rf.len() + sub_n_clusters, false);
            }
            refinement.n_clusters += sub_n_clusters;
        }
    }

    debug_assert_eq!(parent_clusters.len(), refinement.n_clusters);
    if let Some(rf) = &reduced_fixed {
        debug_assert_eq!(rf.len(), refinement.n_clusters);
    }

    let cluster_starts = counts_to_starts(cluster_counts);
    debug_assert_eq!(cluster_starts.last().copied(), Some(graph.n_nodes as u32));

    RefinementResult {
        clustering: refinement,
        parent_clusters,
        cluster_starts,
        fixed: reduced_fixed,
    }
}

fn refine_streaming_flat(
    graph: &Graph,
    clustering: &Clustering,
    n_clusters: usize,
    npc_starts: &[u32],
    npc_nodes: &[u32],
    local_index: &mut [u32],
    config: &LeidenConfig,
    randomness: f64,
    rng: &mut impl Rng,
) -> RefinementResult {
    let mut refinement = empty_refinement(graph.n_nodes);
    let mut parent_clusters = Vec::with_capacity(n_clusters);
    let mut cluster_counts = Vec::with_capacity(n_clusters);
    let mut reduced_fixed = clustering
        .fixed
        .as_ref()
        .map(|_| Vec::with_capacity(n_clusters));
    let mut merge_ws = local_merge::LocalMergeWorkspace::new(0);
    let fixed = clustering.fixed.as_deref();

    for c in 0..n_clusters {
        let cs = npc_starts[c] as usize;
        let ce = npc_starts[c + 1] as usize;
        let nodes = &npc_nodes[cs..ce];
        if nodes.is_empty() {
            continue;
        }
        if nodes.len() == 1 {
            refinement.clusters[nodes[0] as usize] = refinement.n_clusters as u32;
            parent_clusters.push(c as u32);
            cluster_counts.push(1);
            if let Some(rf) = reduced_fixed.as_mut() {
                rf.push(fixed.is_some_and(|fixed| fixed[nodes[0] as usize]));
            }
            refinement.n_clusters += 1;
            continue;
        }

        if let Some(fixed) = fixed {
            if nodes.iter().any(|&n| fixed[n as usize]) {
                for &node in nodes {
                    refinement.clusters[node as usize] = refinement.n_clusters as u32;
                }
                parent_clusters.push(c as u32);
                cluster_counts.push(nodes.len() as u32);
                if let Some(rf) = reduced_fixed.as_mut() {
                    rf.push(true);
                }
                refinement.n_clusters += 1;
                continue;
            }
        }

        let sub_n_clusters =
            local_merge::find_clustering_induced_u32_with_workspace_assignments_and_append_sizes(
                graph,
                nodes,
                local_index,
                config.resolution,
                randomness,
                rng,
                &mut merge_ws,
                &mut cluster_counts,
            );
        let sub_clusters = merge_ws.assignments();

        for (local_idx, &node) in nodes.iter().enumerate() {
            refinement.clusters[node as usize] =
                refinement.n_clusters as u32 + sub_clusters[local_idx];
        }
        parent_clusters.resize(parent_clusters.len() + sub_n_clusters, c as u32);
        if let Some(rf) = reduced_fixed.as_mut() {
            rf.resize(rf.len() + sub_n_clusters, false);
        }
        refinement.n_clusters += sub_n_clusters;
    }

    debug_assert_eq!(parent_clusters.len(), refinement.n_clusters);
    if let Some(rf) = &reduced_fixed {
        debug_assert_eq!(rf.len(), refinement.n_clusters);
    }

    let cluster_starts = counts_to_starts(cluster_counts);
    debug_assert_eq!(cluster_starts.last().copied(), Some(graph.n_nodes as u32));

    RefinementResult {
        clustering: refinement,
        parent_clusters,
        cluster_starts,
        fixed: reduced_fixed,
    }
}

use crate::quality::QualityFunction;
use rand::SeedableRng;

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_leiden_two_cliques() {
        // Two triangles with weak bridge
        let g = Graph::from_edge_list(
            6,
            &[0, 1, 2, 3, 4, 5, 2],
            &[1, 2, 0, 4, 5, 3, 3],
            &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.01],
        );
        let config = LeidenConfig {
            resolution: 0.5,
            n_iterations: 10,
            randomness: 0.01,
            randomness_schedule: Vec::new(),
            seed: 42,
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let result = leiden(&g, &config, None, &mut rng);

        // Should find 2 communities
        assert!(result.clustering.n_clusters <= 3);
        assert!(result.quality > 0.0);
        // Same clique = same cluster
        assert_eq!(result.clustering.clusters[0], result.clustering.clusters[1]);
        assert_eq!(result.clustering.clusters[3], result.clustering.clusters[4]);
    }

    #[test]
    fn test_leiden_with_fixed() {
        let g = Graph::from_edge_list(
            6,
            &[0, 1, 2, 3, 4, 5, 2],
            &[1, 2, 0, 4, 5, 3, 3],
            &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.01],
        );

        // Fix nodes 0,1,2 in cluster 0; let 3,4,5 be free
        let mut init = Clustering::from_assignments(vec![0, 0, 0, 1, 2, 3]);
        init.set_fixed(vec![true, true, true, false, false, false]);

        let config = LeidenConfig {
            resolution: 0.5,
            n_iterations: 10,
            randomness: 0.01,
            randomness_schedule: Vec::new(),
            seed: 42,
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let result = leiden(&g, &config, Some(init), &mut rng);

        // Fixed nodes should stay in cluster 0
        assert_eq!(result.clustering.clusters[0], result.clustering.clusters[1]);
        assert_eq!(result.clustering.clusters[1], result.clustering.clusters[2]);
        // Free nodes 3,4,5 should form their own cluster
        assert_eq!(result.clustering.clusters[3], result.clustering.clusters[4]);
    }

    #[test]
    fn test_flat_nodes_per_cluster_layout() {
        let clustering = Clustering::from_assignments(vec![1, 0, 1, 0, 2]);
        let mut ws = Workspace::new(clustering.n_nodes);

        clustering.fill_cluster_groups(&mut ws);

        assert_eq!(&ws.npc_starts[..=clustering.n_clusters], &[0, 2, 4, 5]);
        assert_eq!(&ws.npc_nodes[..clustering.n_nodes], &[1, 3, 0, 2, 4]);
    }

    #[test]
    fn test_empty_refinement_is_zeroed_not_singleton() {
        let refinement = empty_refinement(4);
        assert_eq!(refinement.n_nodes, 4);
        assert_eq!(refinement.n_clusters, 0);
        assert_eq!(refinement.clusters, vec![0, 0, 0, 0]);
        assert!(refinement.fixed.is_none());
    }
}
