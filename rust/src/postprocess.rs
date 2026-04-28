//! Constrained postprocessing on the CLUSTER GRAPH with cascading γ.
//!
//! 1. Build cluster graph (contraction) with node_sizes
//! 2. Fix large clusters, free small clusters
//! 3. Iteratively lower γ until all clusters ≥ threshold
//! 4. Map results back to original nodes
//!
//! Supports both raw node count and weighted (doc_count) thresholds.
//! When graph.node_weights are non-uniform (contracted graphs), the
//! "size" of a cluster is the sum of node_weights, not the raw count.

use crate::clustering::Clustering;
use crate::contraction::{create_reduced_network, create_reduced_network_from_workspace_groups};
use crate::graph::Graph;
use crate::leiden::{leiden_with_workspace, LeidenConfig};
use crate::trace;
use crate::workspace::Workspace;
use rand::Rng;
use rand::SeedableRng;
use std::time::Instant;

/// Info about one postprocess round.
#[derive(Clone, Debug)]
pub struct PostprocessRound {
    pub round: usize,
    pub gamma: f64,
    pub method: String,          // "leiden" or "greedy"
    pub n_small_before: usize,   // small clusters before this round
    pub n_small_after: usize,    // small clusters after
    pub n_merged: usize,         // clusters that merged in this round
    pub n_new_clusters: usize,   // new clusters formed from small+small merges
    pub n_total_clusters: usize, // total clusters after this round
    pub max_cluster_size: usize, // largest cluster after this round (raw count)
    pub max_cluster_weight: f64, // largest cluster weight after this round
}

/// Result of postprocessing.
#[derive(Clone, Debug)]
pub struct PostprocessResult {
    pub clustering: Clustering,
    pub rounds: Vec<PostprocessRound>,
    /// Per-node: which round changed this node's cluster (-1 = unchanged).
    pub changed_at_round: Vec<i32>,
}

struct ClusterStats {
    sizes: Vec<usize>,
    weights: Vec<f64>,
    n_small: usize,
    max_size: usize,
    max_weight: f64,
}

impl ClusterStats {
    fn from_counts_and_weights(
        counts: &[u32],
        weights: &[f64],
        min_weight: f64,
        min_size: usize,
    ) -> Self {
        debug_assert_eq!(counts.len(), weights.len());
        let mut sizes = Vec::with_capacity(counts.len());
        let mut out_weights = Vec::with_capacity(weights.len());
        let mut n_small = 0usize;
        let mut max_size = 0usize;
        let mut max_weight = 0.0f64;

        for (&count, &weight) in counts.iter().zip(weights.iter()) {
            let size = count as usize;
            if is_small(weight, size, min_weight, min_size) {
                n_small += 1;
            }
            max_size = max_size.max(size);
            max_weight = max_weight.max(weight);
            sizes.push(size);
            out_weights.push(weight);
        }

        ClusterStats {
            sizes,
            weights: out_weights,
            n_small,
            max_size,
            max_weight,
        }
    }

    fn from_sizes_and_weights(
        sizes: Vec<usize>,
        weights: &[f64],
        min_weight: f64,
        min_size: usize,
    ) -> Self {
        debug_assert_eq!(sizes.len(), weights.len());
        let mut out_weights = Vec::with_capacity(weights.len());
        let mut n_small = 0usize;
        let mut max_size = 0usize;
        let mut max_weight = 0.0f64;

        for (&size, &weight) in sizes.iter().zip(weights.iter()) {
            if is_small(weight, size, min_weight, min_size) {
                n_small += 1;
            }
            max_size = max_size.max(size);
            max_weight = max_weight.max(weight);
            out_weights.push(weight);
        }

        ClusterStats {
            sizes,
            weights: out_weights,
            n_small,
            max_size,
            max_weight,
        }
    }
}

fn prepare_cluster_groups_and_stats(
    graph: &Graph,
    clustering: &Clustering,
    min_weight: f64,
    min_size: usize,
    ws: &mut Workspace,
) -> ClusterStats {
    clustering.fill_cluster_groups_and_weights(&graph.node_weights, ws);
    ClusterStats::from_counts_and_weights(
        &ws.npc[..clustering.n_clusters],
        &ws.cw[..clustering.n_clusters],
        min_weight,
        min_size,
    )
}

fn aggregate_cluster_sizes(sizes: &[usize], cluster_map: &[u32], n_clusters: usize) -> Vec<usize> {
    debug_assert!(cluster_map.len() >= sizes.len());
    let mut next_sizes = vec![0usize; n_clusters];
    for (cid, &size) in sizes.iter().enumerate() {
        let next_cid = cluster_map[cid] as usize;
        debug_assert!(next_cid < n_clusters);
        next_sizes[next_cid] += size;
    }
    next_sizes
}

fn cluster_map_has_changes(
    cluster_map: &[u32],
    n_clusters_before: usize,
    n_clusters_after: usize,
) -> bool {
    if n_clusters_before != n_clusters_after {
        return true;
    }
    cluster_map
        .iter()
        .take(n_clusters_before)
        .enumerate()
        .any(|(cid, &next)| next as usize != cid)
}

fn apply_cluster_map_to_projection(
    projection: &mut Clustering,
    cluster_map: &[u32],
    n_clusters: usize,
    cluster_changed_at: &mut [i32],
    round: i32,
) {
    debug_assert_eq!(cluster_changed_at.len(), projection.n_nodes);
    debug_assert!(cluster_map.len() >= projection.n_clusters);

    for cid in 0..projection.n_nodes {
        let old_cid = projection.clusters[cid];
        let new_cid = cluster_map[old_cid as usize];
        if new_cid != old_cid && cluster_changed_at[cid] == -1 {
            cluster_changed_at[cid] = round;
        }
        projection.clusters[cid] = new_cid;
    }
    projection.n_clusters = n_clusters;
}

fn project_back_to_nodes(
    base: &Clustering,
    projection: &Clustering,
    cluster_changed_at: &[i32],
    ws: &mut Workspace,
) -> (Clustering, Vec<i32>) {
    debug_assert!(projection.clusters.len() >= base.n_clusters);
    debug_assert!(cluster_changed_at.len() >= base.n_clusters);

    ws.ensure_capacity(base.n_nodes.max(projection.n_clusters));
    let counts = &mut ws.npc[..projection.n_clusters];
    counts.fill(0);

    let mut clusters = Vec::with_capacity(base.n_nodes);
    let mut changed_at = vec![-1i32; base.n_nodes];
    for node in 0..base.n_nodes {
        let initial_cid = base.clusters[node] as usize;
        let final_cid = projection.clusters[initial_cid];
        clusters.push(final_cid);
        counts[final_cid as usize] += 1;

        let changed_round = cluster_changed_at[initial_cid];
        if changed_round != -1 {
            changed_at[node] = changed_round;
        }
    }

    let mut clustering = Clustering {
        n_nodes: base.n_nodes,
        n_clusters: projection.n_clusters,
        clusters,
        fixed: base.fixed.clone(),
    };
    clustering.compact_from_counts(counts);
    (clustering, changed_at)
}

/// Check if a cluster is "small" using the weighted threshold.
/// If min_weight > 0, compare against weight sum; else use raw count.
fn is_small(weight: f64, raw_size: usize, min_weight: f64, min_size: usize) -> bool {
    if min_weight > 0.0 {
        weight > 0.0 && weight < min_weight
    } else {
        raw_size > 0 && raw_size < min_size
    }
}

/// Reassign small clusters using cascading γ on the cluster graph.
///
/// Threshold semantics:
/// - `min_size`: raw node count threshold (used when node_weights are all 1.0)
/// - `min_weight`: weighted threshold (sum of node_weights, used for contracted graphs)
/// - If `min_weight > 0`, it takes precedence over `min_size`.
pub fn postprocess_small_clusters(
    graph: &Graph,
    clustering: &Clustering,
    config: &LeidenConfig,
    min_size: usize,
    min_weight: f64,
    max_rounds: usize,
    gamma_decay: f64,
    use_greedy: bool,
    greedy_anchor_only: bool,
    greedy_fallback_to_small: bool,
    greedy_max_weight: f64,
    use_component_merge: bool,
    component_max_weight: f64,
    _rng: &mut impl Rng,
) -> PostprocessResult {
    let trace_run = trace::should_trace_edges(graph.n_edges);
    let total_start = Instant::now();
    let mut gamma = config.resolution;
    let mut rounds = Vec::new();
    let mut ws = Workspace::new(graph.n_nodes.max(clustering.n_clusters));

    // Initial contraction is the only postprocess pass that needs the original
    // graph. Subsequent rounds operate on the cluster graph and maintain a
    // projection from initial clusters to current clusters.
    let base = clustering.clone();
    let initial_stats_start = Instant::now();
    let mut stats = prepare_cluster_groups_and_stats(graph, &base, min_weight, min_size, &mut ws);
    let initial_stats_ms = initial_stats_start.elapsed().as_secs_f64() * 1000.0;
    if trace_run {
        trace::emit(format_args!(
            "phase=postprocess_start nodes={} directed_edges={} initial_clusters={} small_clusters={} min_size={} min_weight={:.3} max_rounds={} gamma_decay={:.4} stats_ms={:.1}{}",
            graph.n_nodes,
            graph.n_edges,
            base.n_clusters,
            stats.n_small,
            min_size,
            min_weight,
            max_rounds,
            gamma_decay,
            initial_stats_ms,
            trace::memory_fields(),
        ));
    }

    let initial_contract_start = Instant::now();
    let mut current_graph =
        create_reduced_network_from_workspace_groups(graph, &base, false, &mut ws);
    if trace_run {
        trace::emit(format_args!(
            "phase=postprocess_initial_contract cluster_nodes={} cluster_directed_edges={} elapsed_ms={:.1}{}",
            current_graph.n_nodes,
            current_graph.n_edges,
            initial_contract_start.elapsed().as_secs_f64() * 1000.0,
            trace::memory_fields(),
        ));
    }
    let mut projection = Clustering::singleton(base.n_clusters);
    let mut cluster_changed_at = vec![-1i32; base.n_clusters];
    let mut changed_any = false;

    for round in 0..max_rounds {
        let round_start = Instant::now();
        let n_clusters_before = current_graph.n_nodes;
        let n_edges_before = current_graph.n_edges;
        let n_small_before = stats.n_small;

        if n_small_before == 0 {
            break;
        }

        // Fix large clusters, free small ones
        let n_cls = current_graph.n_nodes;
        let mut cluster_init = Clustering::singleton(n_cls);
        let mut fixed = vec![false; n_cls];
        for cid in 0..n_cls {
            if !is_small(stats.weights[cid], stats.sizes[cid], min_weight, min_size) {
                fixed[cid] = true;
            }
        }
        cluster_init.set_fixed(fixed);

        // Run Leiden on cluster graph
        let cluster_config = LeidenConfig {
            resolution: gamma,
            n_iterations: config.n_iterations,
            randomness: config.randomness,
            randomness_schedule: config.randomness_schedule.clone(),
            seed: config.seed + round as u64,
        };
        let mut round_rng = rand::rngs::StdRng::seed_from_u64(config.seed + round as u64);
        let leiden_start = Instant::now();
        let result = leiden_with_workspace(
            &current_graph,
            &cluster_config,
            Some(cluster_init),
            &mut round_rng,
            &mut ws,
        );
        let leiden_ms = leiden_start.elapsed().as_secs_f64() * 1000.0;

        let round_changed = cluster_map_has_changes(
            &result.clustering.clusters,
            n_clusters_before,
            result.clustering.n_clusters,
        );

        let mut contract_ms = 0.0f64;
        let stats_ms: f64;
        let new_stats = if round_changed {
            let new_sizes = aggregate_cluster_sizes(
                &stats.sizes,
                &result.clustering.clusters,
                result.clustering.n_clusters,
            );
            apply_cluster_map_to_projection(
                &mut projection,
                &result.clustering.clusters,
                result.clustering.n_clusters,
                &mut cluster_changed_at,
                round as i32,
            );
            let contract_start = Instant::now();
            current_graph =
                create_reduced_network(&current_graph, &result.clustering, false, &mut ws);
            contract_ms = contract_start.elapsed().as_secs_f64() * 1000.0;
            changed_any = true;
            let stats_start = Instant::now();
            let next_stats = ClusterStats::from_sizes_and_weights(
                new_sizes,
                &current_graph.node_weights,
                min_weight,
                min_size,
            );
            stats_ms = stats_start.elapsed().as_secs_f64() * 1000.0;
            next_stats
        } else {
            let stats_start = Instant::now();
            let next_stats = ClusterStats::from_sizes_and_weights(
                stats.sizes.clone(),
                &current_graph.node_weights,
                min_weight,
                min_size,
            );
            stats_ms = stats_start.elapsed().as_secs_f64() * 1000.0;
            next_stats
        };

        let n_small_after = new_stats.n_small;
        let n_merged = n_clusters_before.saturating_sub(current_graph.n_nodes);

        rounds.push(PostprocessRound {
            round,
            gamma,
            method: "leiden".to_string(),
            n_small_before,
            n_small_after,
            n_merged,
            n_new_clusters: 0,
            n_total_clusters: current_graph.n_nodes,
            max_cluster_size: new_stats.max_size,
            max_cluster_weight: new_stats.max_weight,
        });

        if trace_run {
            trace::emit(format_args!(
                "phase=postprocess_round round={} gamma={:.8} changed={} cluster_nodes_before={} cluster_edges_before={} cluster_nodes_after={} cluster_edges_after={} small_before={} small_after={} merged={} leiden_ms={:.1} contract_ms={:.1} stats_ms={:.1} elapsed_ms={:.1}{}",
                round,
                gamma,
                round_changed,
                n_clusters_before,
                n_edges_before,
                current_graph.n_nodes,
                current_graph.n_edges,
                n_small_before,
                n_small_after,
                n_merged,
                leiden_ms,
                contract_ms,
                stats_ms,
                round_start.elapsed().as_secs_f64() * 1000.0,
                trace::memory_fields(),
            ));
        }

        stats = new_stats;
        if n_small_after == 0 {
            break;
        }

        gamma *= gamma_decay;
    }

    // Greedy fallback
    let n_small_before = stats.n_small;
    if use_greedy && n_small_before > 0 {
        let greedy_start = Instant::now();
        let n_before = current_graph.n_nodes;
        let edges_before = current_graph.n_edges;
        let greedy_round = rounds.len() as i32;
        let mut level_clustering = Clustering::singleton(current_graph.n_nodes);
        let mut level_changed_at = vec![-1i32; current_graph.n_nodes];
        let changed = greedy_merge_remaining(
            &current_graph,
            &mut level_clustering,
            min_size,
            min_weight,
            greedy_anchor_only,
            greedy_fallback_to_small,
            greedy_max_weight,
            &stats,
            &mut level_changed_at,
            greedy_round,
            &mut ws,
        );

        let mut contract_ms = 0.0f64;
        let mut stats_ms = 0.0f64;
        let mut n_small_after = n_small_before;
        if changed {
            let new_sizes = aggregate_cluster_sizes(
                &stats.sizes,
                &level_clustering.clusters,
                level_clustering.n_clusters,
            );
            apply_cluster_map_to_projection(
                &mut projection,
                &level_clustering.clusters,
                level_clustering.n_clusters,
                &mut cluster_changed_at,
                greedy_round,
            );
            let contract_start = Instant::now();
            current_graph =
                create_reduced_network(&current_graph, &level_clustering, false, &mut ws);
            contract_ms = contract_start.elapsed().as_secs_f64() * 1000.0;
            changed_any = true;
            let stats_start = Instant::now();
            let new_stats = ClusterStats::from_sizes_and_weights(
                new_sizes,
                &current_graph.node_weights,
                min_weight,
                min_size,
            );
            stats_ms = stats_start.elapsed().as_secs_f64() * 1000.0;
            n_small_after = new_stats.n_small;

            rounds.push(PostprocessRound {
                round: rounds.len(),
                gamma: 0.0,
                method: "greedy".to_string(),
                n_small_before,
                n_small_after,
                n_merged: n_before.saturating_sub(current_graph.n_nodes),
                n_new_clusters: 0,
                n_total_clusters: current_graph.n_nodes,
                max_cluster_size: new_stats.max_size,
                max_cluster_weight: new_stats.max_weight,
            });
            stats = new_stats;
        }
        if trace_run {
            trace::emit(format_args!(
                "phase=postprocess_greedy changed={} cluster_nodes_before={} cluster_edges_before={} cluster_nodes_after={} cluster_edges_after={} small_before={} small_after={} merged={} contract_ms={:.1} stats_ms={:.1} elapsed_ms={:.1}{}",
                changed,
                n_before,
                edges_before,
                current_graph.n_nodes,
                current_graph.n_edges,
                n_small_before,
                n_small_after,
                n_before.saturating_sub(current_graph.n_nodes),
                contract_ms,
                stats_ms,
                greedy_start.elapsed().as_secs_f64() * 1000.0,
                trace::memory_fields(),
            ));
        }
    }

    // Component-level Dijkstra assignment (multi-hop)
    if use_component_merge {
        let n_small_before = stats.n_small;
        if n_small_before > 0 {
            let component_start = Instant::now();
            let n_before = current_graph.n_nodes;
            let edges_before = current_graph.n_edges;
            let comp_round = rounds.len() as i32;
            let mut level_clustering = Clustering::singleton(current_graph.n_nodes);
            let mut level_changed_at = vec![-1i32; current_graph.n_nodes];
            let changed = component_merge_remaining(
                &current_graph,
                &mut level_clustering,
                min_size,
                min_weight,
                component_max_weight,
                &stats,
                &mut level_changed_at,
                comp_round,
                &mut ws,
            );

            let mut contract_ms = 0.0f64;
            let mut stats_ms = 0.0f64;
            let mut n_small_after = n_small_before;
            if changed {
                let new_sizes = aggregate_cluster_sizes(
                    &stats.sizes,
                    &level_clustering.clusters,
                    level_clustering.n_clusters,
                );
                apply_cluster_map_to_projection(
                    &mut projection,
                    &level_clustering.clusters,
                    level_clustering.n_clusters,
                    &mut cluster_changed_at,
                    comp_round,
                );
                let contract_start = Instant::now();
                current_graph =
                    create_reduced_network(&current_graph, &level_clustering, false, &mut ws);
                contract_ms = contract_start.elapsed().as_secs_f64() * 1000.0;
                changed_any = true;
                let stats_start = Instant::now();
                let new_stats = ClusterStats::from_sizes_and_weights(
                    new_sizes,
                    &current_graph.node_weights,
                    min_weight,
                    min_size,
                );
                stats_ms = stats_start.elapsed().as_secs_f64() * 1000.0;
                n_small_after = new_stats.n_small;

                rounds.push(PostprocessRound {
                    round: rounds.len(),
                    gamma: 0.0,
                    method: "component_dijkstra".to_string(),
                    n_small_before,
                    n_small_after,
                    n_merged: n_before.saturating_sub(current_graph.n_nodes),
                    n_new_clusters: 0,
                    n_total_clusters: current_graph.n_nodes,
                    max_cluster_size: new_stats.max_size,
                    max_cluster_weight: new_stats.max_weight,
                });
            }
            if trace_run {
                trace::emit(format_args!(
                    "phase=postprocess_component changed={} cluster_nodes_before={} cluster_edges_before={} cluster_nodes_after={} cluster_edges_after={} small_before={} small_after={} merged={} contract_ms={:.1} stats_ms={:.1} elapsed_ms={:.1}{}",
                    changed,
                    n_before,
                    edges_before,
                    current_graph.n_nodes,
                    current_graph.n_edges,
                    n_small_before,
                    n_small_after,
                    n_before.saturating_sub(current_graph.n_nodes),
                    contract_ms,
                    stats_ms,
                    component_start.elapsed().as_secs_f64() * 1000.0,
                    trace::memory_fields(),
                ));
            }
        }
    }

    if !changed_any {
        if trace_run {
            trace::emit(format_args!(
                "phase=postprocess_done final_clusters={} rounds={} changed_nodes=0 elapsed_ms={:.1}{}",
                clustering.n_clusters,
                rounds.len(),
                total_start.elapsed().as_secs_f64() * 1000.0,
                trace::memory_fields(),
            ));
        }
        return PostprocessResult {
            clustering: clustering.clone(),
            rounds,
            changed_at_round: vec![-1i32; graph.n_nodes],
        };
    }

    let project_start = Instant::now();
    let (final_clustering, changed_at) =
        project_back_to_nodes(&base, &projection, &cluster_changed_at, &mut ws);
    let project_ms = project_start.elapsed().as_secs_f64() * 1000.0;
    let changed_nodes = changed_at.iter().filter(|&&round| round >= 0).count();
    if trace_run {
        trace::emit(format_args!(
            "phase=postprocess_project_back nodes={} changed_nodes={} elapsed_ms={:.1}{}",
            graph.n_nodes,
            changed_nodes,
            project_ms,
            trace::memory_fields(),
        ));
        trace::emit(format_args!(
            "phase=postprocess_done final_clusters={} rounds={} changed_nodes={} elapsed_ms={:.1}{}",
            final_clustering.n_clusters,
            rounds.len(),
            changed_nodes,
            total_start.elapsed().as_secs_f64() * 1000.0,
            trace::memory_fields(),
        ));
    }
    PostprocessResult {
        clustering: final_clustering,
        rounds,
        changed_at_round: changed_at,
    }
}

fn apply_sparse_merge_targets_with_changed_at(
    clustering: &mut Clustering,
    changed_at: &mut [i32],
    round: i32,
    ws: &mut Workspace,
    n_clusters: usize,
    touched_targets: &[u32],
) -> bool {
    if touched_targets.is_empty() {
        return false;
    }
    debug_assert_eq!(changed_at.len(), clustering.n_nodes);
    debug_assert!(ws.temp_seen.len() >= n_clusters);
    debug_assert!(ws.npc.len() >= n_clusters);

    let Workspace { npc, temp_seen, .. } = ws;
    let counts = &mut npc[..n_clusters];
    let merge_target = &mut temp_seen[..n_clusters];
    counts.fill(0);

    let mut any_changed = false;
    for node in 0..clustering.n_nodes {
        let old_cid = clustering.clusters[node];
        let target = merge_target[old_cid as usize];
        let new_cid = if target == u32::MAX { old_cid } else { target };
        if new_cid != old_cid {
            any_changed = true;
            if changed_at[node] == -1 {
                changed_at[node] = round;
            }
        }
        clustering.clusters[node] = new_cid;
        counts[new_cid as usize] += 1;
    }

    for &cid in touched_targets {
        merge_target[cid as usize] = u32::MAX;
    }

    clustering.n_clusters = n_clusters;
    clustering.compact_from_counts(counts);
    any_changed
}

/// Greedy fallback: merge remaining small clusters via cluster graph.
fn greedy_merge_remaining(
    cluster_graph: &Graph,
    clustering: &mut Clustering,
    min_size: usize,
    min_weight: f64,
    anchor_only: bool,
    fallback_to_small: bool,
    max_weight: f64,
    stats: &ClusterStats,
    changed_at: &mut [i32],
    round: i32,
    ws: &mut Workspace,
) -> bool {
    let sizes = &stats.sizes;
    let weights = &stats.weights;
    let n_cls = clustering.n_clusters;
    let anchor_mask: Vec<bool> = (0..n_cls)
        .map(|cid| !is_small(weights[cid], sizes[cid], min_weight, min_size))
        .collect();

    let cg = cluster_graph;
    let mut touched_targets = std::mem::take(&mut ws.temp_used);
    touched_targets.clear();
    let mut projected_weights = weights.clone();
    let mut small_clusters: Vec<usize> = (0..n_cls)
        .filter(|&cid| is_small(weights[cid], sizes[cid], min_weight, min_size))
        .collect();
    if max_weight > 0.0 {
        small_clusters.sort_by(|&a, &b| {
            weights[b]
                .partial_cmp(&weights[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    for cid in small_clusters {
        let start = cg.first_neighbor_index[cid] as usize;
        let end = cg.first_neighbor_index[cid + 1] as usize;
        let mut best_fit_cid = usize::MAX;
        let mut best_fit_w = -1.0f64;
        let mut best_overflow_cid = usize::MAX;
        let mut best_overflow_w = -1.0f64;
        let mut best_overflow_amount = f64::INFINITY;
        let mut best_any_fit_cid = usize::MAX;
        let mut best_any_fit_w = -1.0f64;
        let mut best_any_overflow_cid = usize::MAX;
        let mut best_any_overflow_w = -1.0f64;
        let mut best_any_overflow_amount = f64::INFINITY;
        for k in start..end {
            let nbr = cg.neighbors[k] as usize;
            if nbr == cid {
                continue;
            }
            let edge_w = cg.edge_weights[k];
            let merged_weight = projected_weights[nbr] + weights[cid];
            let nbr_is_anchor = anchor_mask[nbr];
            let allow_anchor = !anchor_only || nbr_is_anchor;
            if max_weight > 0.0 {
                if merged_weight <= max_weight {
                    if allow_anchor && edge_w > best_fit_w {
                        best_fit_w = edge_w;
                        best_fit_cid = nbr;
                    }
                    if edge_w > best_any_fit_w {
                        best_any_fit_w = edge_w;
                        best_any_fit_cid = nbr;
                    }
                } else {
                    let overflow = merged_weight - max_weight;
                    if allow_anchor
                        && (overflow < best_overflow_amount
                            || (overflow == best_overflow_amount && edge_w > best_overflow_w))
                    {
                        best_overflow_amount = overflow;
                        best_overflow_w = edge_w;
                        best_overflow_cid = nbr;
                    }
                    if overflow < best_any_overflow_amount
                        || (overflow == best_any_overflow_amount && edge_w > best_any_overflow_w)
                    {
                        best_any_overflow_amount = overflow;
                        best_any_overflow_w = edge_w;
                        best_any_overflow_cid = nbr;
                    }
                }
            } else {
                if allow_anchor && edge_w > best_fit_w {
                    best_fit_w = edge_w;
                    best_fit_cid = nbr;
                }
                if edge_w > best_any_fit_w {
                    best_any_fit_w = edge_w;
                    best_any_fit_cid = nbr;
                }
            }
        }

        let target = if best_fit_cid != usize::MAX {
            best_fit_cid
        } else if best_overflow_cid != usize::MAX {
            best_overflow_cid
        } else if fallback_to_small {
            if best_any_fit_cid != usize::MAX {
                best_any_fit_cid
            } else {
                best_any_overflow_cid
            }
        } else {
            usize::MAX
        };
        if target != usize::MAX {
            ws.temp_seen[cid] = target as u32;
            touched_targets.push(cid as u32);
            projected_weights[target] += weights[cid];
        }
    }

    let changed = apply_sparse_merge_targets_with_changed_at(
        clustering,
        changed_at,
        round,
        ws,
        n_cls,
        &touched_targets,
    );
    touched_targets.clear();
    ws.temp_used = touched_targets;
    changed
}

/// Component-level assignment: multi-hop Dijkstra from anchors.
///
/// 1. Build cluster graph
/// 2. Find connected components of small clusters
/// 3. Components touching anchors → Dijkstra nearest-anchor assignment
/// 4. Components with no anchor → pick forced anchor (largest weight), then Dijkstra
/// 5. Respect max_weight soft cap (least-overflow fallback)
fn component_merge_remaining(
    cluster_graph: &Graph,
    clustering: &mut Clustering,
    min_size: usize,
    min_weight: f64,
    max_weight_cap: f64,
    stats: &ClusterStats,
    changed_at: &mut [i32],
    round: i32,
    ws: &mut Workspace,
) -> bool {
    let sizes = &stats.sizes;
    let weights = &stats.weights;
    let n_cls = clustering.n_clusters;

    let cg = cluster_graph;
    let first_nbr = cg.first_neighbor_index.as_slice();
    let neighbors = cg.neighbors.as_slice();
    let edge_weights = cg.edge_weights.as_slice();

    // Classify anchors vs small
    let is_anchor: Vec<bool> = (0..n_cls)
        .map(|c| !is_small(weights[c], sizes[c], min_weight, min_size))
        .collect();

    // Find connected components of ALL clusters (including anchors), then
    // process each component immediately. This avoids duplicating the cluster
    // graph CSR into Vec<Vec<...>> adjacency storage.
    let mut visited = vec![false; n_cls];
    let mut component = Vec::new();
    let mut stack = Vec::new();
    let mut comp_anchors: Vec<usize> = Vec::new();
    let mut comp_smalls: Vec<usize> = Vec::new();
    let mut dist = vec![f64::INFINITY; n_cls];
    let mut nearest_anchor = vec![usize::MAX; n_cls];
    let mut touched = Vec::new();
    let mut heap = std::collections::BinaryHeap::new();
    let mut touched_targets = std::mem::take(&mut ws.temp_used);
    touched_targets.clear();
    let mut projected_weights = weights.clone();

    for start in 0..n_cls {
        if visited[start] || sizes[start] == 0 {
            continue;
        }

        component.clear();
        stack.clear();
        stack.push(start);
        while let Some(node) = stack.pop() {
            if visited[node] {
                continue;
            }
            visited[node] = true;
            component.push(node);

            let row_start = first_nbr[node] as usize;
            let row_end = first_nbr[node + 1] as usize;
            for k in row_start..row_end {
                let nbr = neighbors[k] as usize;
                if nbr != node && !visited[nbr] && sizes[nbr] > 0 {
                    stack.push(nbr);
                }
            }
        }

        comp_anchors.clear();
        comp_smalls.clear();
        for &cid in &component {
            if is_anchor[cid] {
                comp_anchors.push(cid);
            } else {
                comp_smalls.push(cid);
            }
        }
        if comp_smalls.is_empty() {
            continue;
        }

        // If no anchor in component, pick forced anchor (largest weight)
        if comp_anchors.is_empty() {
            let forced = *component
                .iter()
                .max_by(|&&a, &&b| weights[a].total_cmp(&weights[b]))
                .unwrap();
            comp_anchors.push(forced);
            comp_smalls.retain(|&c| c != forced);
            if comp_smalls.is_empty() {
                continue;
            }
        }

        if comp_anchors.len() == 1 {
            let anchor = comp_anchors[0];
            for &small_cid in &comp_smalls {
                if small_cid == anchor {
                    continue;
                }
                ws.temp_seen[small_cid] = anchor as u32;
                touched_targets.push(small_cid as u32);
                projected_weights[anchor] += weights[small_cid];
            }
            continue;
        }

        // Multi-source Dijkstra from all anchors
        // dist[cid] and nearest_anchor[cid] are reset only for touched nodes.
        touched.clear();
        heap.clear();
        for &anchor in &comp_anchors {
            dist[anchor] = 0.0;
            nearest_anchor[anchor] = anchor;
            touched.push(anchor);
            // BinaryHeap is max-heap, negate distance for min-heap
            heap.push(std::cmp::Reverse((OrderedFloat(0.0), anchor, anchor)));
        }

        while let Some(std::cmp::Reverse((OrderedFloat(d), node, anchor))) = heap.pop() {
            if d > dist[node] {
                continue;
            }
            let row_start = first_nbr[node] as usize;
            let row_end = first_nbr[node + 1] as usize;
            for k in row_start..row_end {
                let nbr = neighbors[k] as usize;
                if nbr == node || sizes[nbr] == 0 {
                    continue;
                }
                let edge_w = edge_weights[k];
                let cost = 1.0 / edge_w.max(1e-12);
                let new_dist = d + cost;
                if new_dist < dist[nbr] {
                    if dist[nbr].is_infinite() {
                        touched.push(nbr);
                    }
                    dist[nbr] = new_dist;
                    nearest_anchor[nbr] = anchor;
                    heap.push(std::cmp::Reverse((OrderedFloat(new_dist), nbr, anchor)));
                }
            }
        }

        // Assign small clusters to nearest anchor, respecting cap
        // Sort smalls by weight descending (assign larger ones first for better cap control)
        if max_weight_cap > 0.0 {
            comp_smalls.sort_by(|&a, &b| weights[b].total_cmp(&weights[a]));
        }

        for &small_cid in &comp_smalls {
            let anchor = nearest_anchor[small_cid];
            let target = if anchor != usize::MAX && anchor != small_cid {
                anchor
            } else {
                usize::MAX
            };

            if target == usize::MAX {
                continue;
            }

            // Cap check
            if max_weight_cap > 0.0 {
                let merged = projected_weights[target] + weights[small_cid];
                if merged > max_weight_cap {
                    // Find least-overflow anchor from all anchors in component
                    let mut best_anchor = target;
                    let mut best_overflow = merged - max_weight_cap;
                    for &a in &comp_anchors {
                        let m = projected_weights[a] + weights[small_cid];
                        let overflow = if m > max_weight_cap {
                            m - max_weight_cap
                        } else {
                            0.0
                        };
                        if overflow < best_overflow {
                            best_overflow = overflow;
                            best_anchor = a;
                        }
                    }
                    ws.temp_seen[small_cid] = best_anchor as u32;
                    touched_targets.push(small_cid as u32);
                    projected_weights[best_anchor] += weights[small_cid];
                    continue;
                }
            }
            ws.temp_seen[small_cid] = target as u32;
            touched_targets.push(small_cid as u32);
            projected_weights[target] += weights[small_cid];
        }

        for &node in &touched {
            dist[node] = f64::INFINITY;
            nearest_anchor[node] = usize::MAX;
        }
    }

    let changed = apply_sparse_merge_targets_with_changed_at(
        clustering,
        changed_at,
        round,
        ws,
        n_cls,
        &touched_targets,
    );
    touched_targets.clear();
    ws.temp_used = touched_targets;
    changed
}

/// Ordered float wrapper for BinaryHeap (NaN-safe).
#[derive(Clone, Copy, PartialEq)]
struct OrderedFloat(f64);
impl Eq for OrderedFloat {}
impl PartialOrd for OrderedFloat {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for OrderedFloat {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_postprocess_with_monitoring() {
        let g = Graph::from_edge_list(
            10,
            &[0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 5, 7, 4, 6, 8],
            &[1, 2, 3, 4, 2, 3, 4, 3, 4, 4, 6, 8, 5, 7, 9],
            &[
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1, 0.3, 0.5,
            ],
        );
        let init = Clustering::from_assignments(vec![0, 0, 0, 0, 0, 1, 1, 2, 2, 3]);

        let config = LeidenConfig {
            resolution: 0.1,
            n_iterations: 10,
            randomness: 0.01,
            randomness_schedule: Vec::new(),
            seed: 42,
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        // min_weight=0.0 → use min_size=4
        let result = postprocess_small_clusters(
            &g, &init, &config, 4, 0.0, 5, 0.1, true, false, false, 0.0, true, 0.0, &mut rng,
        );

        let sizes = result.clustering.cluster_sizes();
        let remaining_small = sizes.iter().filter(|&&s| s > 0 && s < 4).count();
        assert!(
            remaining_small < 3,
            "postprocess should reduce small clusters"
        );
        assert!(!result.rounds.is_empty(), "should have at least 1 round");
    }

    #[test]
    fn test_postprocess_weighted() {
        // 6 nodes: weights [10, 10, 10, 1, 1, 1]
        // clusters: [0,0,0, 1,1,1] → cluster 0 weight=30, cluster 1 weight=3
        let mut g = Graph::from_edge_list(
            6,
            &[0, 1, 2, 3, 4, 5, 2],
            &[1, 2, 0, 4, 5, 3, 3],
            &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5],
        );
        g.node_weights = vec![10.0, 10.0, 10.0, 1.0, 1.0, 1.0];
        let init = Clustering::from_assignments(vec![0, 0, 0, 1, 1, 1]);

        let config = LeidenConfig::default();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        // min_weight=5.0 → cluster 1 (weight=3) is small, cluster 0 (weight=30) is large
        let result = postprocess_small_clusters(
            &g, &init, &config, 0, 5.0, 5, 0.1, true, false, false, 0.0, true, 0.0, &mut rng,
        );

        // cluster 1 should merge into cluster 0
        assert_eq!(result.clustering.n_clusters, 1);
    }

    #[test]
    fn test_postprocess_no_small() {
        let g = Graph::from_edge_list(6, &[0, 1, 2, 3, 4, 5], &[1, 2, 0, 4, 5, 3], &[1.0; 6]);
        let init = Clustering::from_assignments(vec![0, 0, 0, 1, 1, 1]);
        let config = LeidenConfig::default();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let result = postprocess_small_clusters(
            &g, &init, &config, 2, 0.0, 5, 0.1, true, false, false, 0.0, true, 0.0, &mut rng,
        );
        assert_eq!(result.clustering.n_clusters, 2);
        assert!(result.rounds.is_empty());
    }
}
