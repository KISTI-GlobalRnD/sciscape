//! Leiden algorithm: move → refine → aggregate → recurse.
//!
//! Port of CWTS LeidenAlgorithm.java.

use crate::clustering::Clustering;
use crate::contraction::{create_reduced_network, create_reduced_network_from_starts};
use crate::fast_local_move;
use crate::graph::Graph;
use crate::local_merge;
use crate::workspace::Workspace;
use rand::Rng;
use std::sync::OnceLock;
use std::time::Instant;

const DEFAULT_STREAMING_REFINEMENT_MIN_DIRECTED_EDGES: usize = 1_000_000;
const TRACE_MIN_DIRECTED_EDGES: usize = 1_000_000;

fn trace_setting() -> &'static str {
    static TRACE: OnceLock<String> = OnceLock::new();
    TRACE
        .get_or_init(|| std::env::var("SCISCAPE_LEIDEN_TRACE").unwrap_or_default())
        .as_str()
}

fn trace_enabled() -> bool {
    !matches!(trace_setting(), "" | "0" | "false" | "False" | "FALSE")
}

fn trace_verbose() -> bool {
    matches!(trace_setting(), "verbose" | "1" | "true" | "True" | "TRUE")
}

fn streaming_refinement_min_directed_edges() -> usize {
    static MIN_EDGES: OnceLock<usize> = OnceLock::new();
    *MIN_EDGES.get_or_init(|| {
        std::env::var("SCISCAPE_STREAMING_REFINEMENT_MIN_DIRECTED_EDGES")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(DEFAULT_STREAMING_REFINEMENT_MIN_DIRECTED_EDGES)
    })
}

fn should_trace_graph(graph: &Graph) -> bool {
    trace_enabled() && (trace_verbose() || graph.n_edges >= TRACE_MIN_DIRECTED_EDGES)
}

macro_rules! trace_graph {
    ($graph:expr, $($arg:tt)*) => {{
        if should_trace_graph($graph) {
            eprintln!("[sciscape_leiden] {}", format_args!($($arg)*));
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

struct RefinementResult {
    clustering: Clustering,
    parent_clusters: Vec<u32>,
    cluster_starts: Vec<u32>,
    fixed: Option<Vec<bool>>,
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
    let mut clustering = initial.unwrap_or_else(|| Clustering::singleton(graph.n_nodes));
    let mut ws = Workspace::new(graph.n_nodes);

    let mut n_used = 0;
    if config.n_iterations > 0 {
        for _ in 0..config.n_iterations {
            let iter_start = Instant::now();
            let iter_randomness = config.randomness_for_iteration(n_used);
            let improved = improve_one_iteration(
                graph,
                &mut clustering,
                config,
                iter_randomness,
                rng,
                &mut ws,
            );
            n_used += 1;
            trace_graph!(
                graph,
                "phase=leiden_iteration iter={} randomness={:.6} improved={} clusters={} elapsed_ms={:.1}",
                n_used,
                iter_randomness,
                improved,
                clustering.n_clusters,
                iter_start.elapsed().as_secs_f64() * 1000.0,
            );
            if !improved {
                break;
            }
        }
    } else {
        loop {
            let iter_start = Instant::now();
            let iter_randomness = config.randomness_for_iteration(n_used);
            let improved = improve_one_iteration(
                graph,
                &mut clustering,
                config,
                iter_randomness,
                rng,
                &mut ws,
            );
            n_used += 1;
            trace_graph!(
                graph,
                "phase=leiden_iteration iter={} randomness={:.6} improved={} clusters={} elapsed_ms={:.1}",
                n_used,
                iter_randomness,
                improved,
                clustering.n_clusters,
                iter_start.elapsed().as_secs_f64() * 1000.0,
            );
            if !improved {
                break;
            }
        }
    }

    let quality = crate::quality::CPM::new(config.resolution).quality(graph, &clustering);

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
) -> bool {
    // Phase 1: Local moving
    let t_move = Instant::now();
    let update = fast_local_move::improve_clustering(graph, clustering, config.resolution, rng, ws);
    trace_graph!(
        graph,
        "phase=local_move nodes={} directed_edges={} clusters={} updated={} elapsed_ms={:.1}",
        graph.n_nodes,
        graph.n_edges,
        clustering.n_clusters,
        update,
        t_move.elapsed().as_secs_f64() * 1000.0,
    );

    // If every node is its own cluster, nothing to do
    if clustering.n_clusters >= graph.n_nodes {
        return update;
    }

    // Phase 2: Refinement
    let use_streaming_refinement = graph.n_edges >= streaming_refinement_min_directed_edges();
    let t_nodes = Instant::now();
    let refinement = if use_streaming_refinement {
        fill_nodes_per_cluster_flat(clustering, ws);
        trace_graph!(
            graph,
            "phase=nodes_per_cluster mode=flat clusters={} elapsed_ms={:.1}",
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
            "phase=refinement mode=streaming refined_clusters={} elapsed_ms={:.1}",
            refinement.clustering.n_clusters,
            t_refine.elapsed().as_secs_f64() * 1000.0,
        );
        refinement
    } else {
        let nodes_per_cluster = clustering.nodes_per_cluster();
        trace_graph!(
            graph,
            "phase=nodes_per_cluster mode=vec clusters={} elapsed_ms={:.1}",
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
            "phase=refinement mode=eager refined_clusters={} elapsed_ms={:.1}",
            refinement.clustering.n_clusters,
            t_refine.elapsed().as_secs_f64() * 1000.0,
        );
        refinement
    };

    if refinement.clustering.n_clusters >= graph.n_nodes {
        // Refinement produced singletons — aggregate on non-refined clustering
        let t_contract = Instant::now();
        let reduced = create_reduced_network(graph, clustering, true, ws);
        trace_graph!(
            graph,
            "phase=contract_non_refined reduced_nodes={} reduced_directed_edges={} elapsed_ms={:.1}",
            reduced.n_nodes,
            reduced.n_edges,
            t_contract.elapsed().as_secs_f64() * 1000.0,
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

        let t_recurse = Instant::now();
        let improved = improve_one_iteration(
            &reduced,
            &mut reduced_clustering,
            config,
            randomness,
            rng,
            ws,
        );
        trace_graph!(
            graph,
            "phase=recurse_non_refined reduced_nodes={} improved={} elapsed_ms={:.1}",
            reduced.n_nodes,
            improved,
            t_recurse.elapsed().as_secs_f64() * 1000.0,
        );
        if improved {
            let t_merge = Instant::now();
            clustering.merge_clusters(&reduced_clustering);
            trace_graph!(
                graph,
                "phase=merge_non_refined nodes={} elapsed_ms={:.1}",
                graph.n_nodes,
                t_merge.elapsed().as_secs_f64() * 1000.0,
            );
        }
        return update | improved;
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
    trace_graph!(
        graph,
        "phase=contract_refined reduced_nodes={} reduced_directed_edges={} elapsed_ms={:.1}",
        reduced.n_nodes,
        reduced.n_edges,
        t_contract.elapsed().as_secs_f64() * 1000.0,
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

    let mut reduced_clustering = Clustering {
        n_nodes: refinement.clustering.n_clusters,
        n_clusters: reduced_n_clusters,
        clusters: refinement.parent_clusters,
        fixed: refinement.fixed,
    };

    // Recurse on reduced network
    let t_recurse = Instant::now();
    let improved = improve_one_iteration(
        &reduced,
        &mut reduced_clustering,
        config,
        randomness,
        rng,
        ws,
    );
    trace_graph!(
        graph,
        "phase=recurse_refined reduced_nodes={} improved={} elapsed_ms={:.1}",
        reduced.n_nodes,
        improved,
        t_recurse.elapsed().as_secs_f64() * 1000.0,
    );

    // Merge back only if the recursive reduced graph actually changed. When
    // recursion reports no improvement, `reduced_clustering` is equivalent to
    // the initial projection and merge-back would only restore the move-phase
    // clustering after a full O(n) scan.
    if improved {
        let t_merge = Instant::now();
        clustering.clusters = refinement.clustering.clusters;
        clustering.n_clusters = refinement.clustering.n_clusters;
        clustering.merge_clusters(&reduced_clustering);
        trace_graph!(
            graph,
            "phase=merge_refined nodes={} elapsed_ms={:.1}",
            graph.n_nodes,
            t_merge.elapsed().as_secs_f64() * 1000.0,
        );
    }

    update | improved
}

fn fill_nodes_per_cluster_flat(clustering: &Clustering, ws: &mut Workspace) {
    let n_nodes = clustering.n_nodes;
    let n_clusters = clustering.n_clusters;
    ws.ensure_capacity(n_nodes.max(n_clusters));

    let counts = &mut ws.npc[..n_clusters];
    counts.fill(0);
    for &cid in &clustering.clusters {
        let cid = cid as usize;
        debug_assert!(cid < n_clusters);
        counts[cid] += 1;
    }

    {
        let starts = &mut ws.npc_starts[..n_clusters + 1];
        starts[0] = 0;
        for c in 0..n_clusters {
            starts[c + 1] = starts[c] + counts[c];
        }
    }

    {
        let starts = &ws.npc_starts[..n_clusters];
        let offsets = &mut ws.npc_off[..n_clusters];
        offsets.copy_from_slice(starts);
    }

    let nodes = &mut ws.npc_nodes[..n_nodes];
    let offsets = &mut ws.npc_off[..n_clusters];
    for (node, &cid) in clustering.clusters.iter().enumerate() {
        let cid = cid as usize;
        debug_assert!(cid < n_clusters);
        let pos = offsets[cid] as usize;
        nodes[pos] = node as u32;
        offsets[cid] += 1;
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

        fill_nodes_per_cluster_flat(&clustering, &mut ws);

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
