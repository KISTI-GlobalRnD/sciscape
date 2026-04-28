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
use crate::contraction::create_reduced_network;
use crate::graph::Graph;
use crate::leiden::{leiden, LeidenConfig};
use crate::workspace::Workspace;
use rand::Rng;
use rand::SeedableRng;

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

fn cluster_stats(
    clustering: &Clustering,
    node_weights: &[f64],
    min_weight: f64,
    min_size: usize,
) -> ClusterStats {
    let mut sizes = vec![0usize; clustering.n_clusters];
    let mut weights = vec![0.0f64; clustering.n_clusters];

    for (node, &cid) in clustering.clusters.iter().enumerate() {
        let cid = cid as usize;
        if cid < clustering.n_clusters {
            sizes[cid] += 1;
            weights[cid] += node_weights[node];
        }
    }

    let mut n_small = 0usize;
    let mut max_size = 0usize;
    let mut max_weight = 0.0f64;
    for cid in 0..clustering.n_clusters {
        let size = sizes[cid];
        let weight = weights[cid];
        if is_small(weight, size, min_weight, min_size) {
            n_small += 1;
        }
        max_size = max_size.max(size);
        max_weight = max_weight.max(weight);
    }

    ClusterStats {
        sizes,
        weights,
        n_small,
        max_size,
        max_weight,
    }
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
    let mut current = clustering.clone();
    let mut gamma = config.resolution;
    let mut rounds = Vec::new();
    let mut changed_at = vec![-1i32; graph.n_nodes];
    let nw = &graph.node_weights;
    let mut ws = Workspace::new(graph.n_nodes.max(clustering.n_clusters));

    for round in 0..max_rounds {
        let stats = cluster_stats(&current, nw, min_weight, min_size);
        let n_clusters_before = current.n_clusters;
        let n_small_before = stats.n_small;

        if n_small_before == 0 {
            break;
        }

        // Build cluster graph
        let cluster_graph = create_reduced_network(graph, &current, false, &mut ws);

        // Fix large clusters, free small ones
        let n_cls = current.n_clusters;
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
        let result = leiden(
            &cluster_graph,
            &cluster_config,
            Some(cluster_init),
            &mut round_rng,
        );

        // Map back and track changes
        let cluster_map = &result.clustering.clusters;
        for node in 0..graph.n_nodes {
            let old_cid = current.clusters[node];
            let new_cid = cluster_map[old_cid as usize];
            if new_cid != old_cid && changed_at[node] == -1 {
                changed_at[node] = round as i32;
            }
            current.clusters[node] = new_cid;
        }
        current.n_clusters = result.clustering.n_clusters;
        current.remove_empty_clusters();

        let new_stats = cluster_stats(&current, nw, min_weight, min_size);
        let n_small_after = new_stats.n_small;
        let n_merged = n_clusters_before - current.n_clusters;

        rounds.push(PostprocessRound {
            round,
            gamma,
            method: "leiden".to_string(),
            n_small_before,
            n_small_after,
            n_merged,
            n_new_clusters: 0,
            n_total_clusters: current.n_clusters,
            max_cluster_size: new_stats.max_size,
            max_cluster_weight: new_stats.max_weight,
        });

        if n_small_after == 0 {
            break;
        }

        gamma *= gamma_decay;
    }

    // Greedy fallback
    let mut stats = cluster_stats(&current, nw, min_weight, min_size);
    let n_small_before = stats.n_small;
    if use_greedy && n_small_before > 0 {
        let n_before = current.n_clusters;
        let greedy_round = rounds.len() as i32;
        greedy_merge_remaining(
            graph,
            &mut current,
            min_size,
            min_weight,
            greedy_anchor_only,
            greedy_fallback_to_small,
            greedy_max_weight,
            &stats,
            &mut changed_at,
            greedy_round,
            &mut ws,
        );

        let new_stats = cluster_stats(&current, nw, min_weight, min_size);
        let n_small_after = new_stats.n_small;

        rounds.push(PostprocessRound {
            round: rounds.len(),
            gamma: 0.0,
            method: "greedy".to_string(),
            n_small_before,
            n_small_after,
            n_merged: n_before - current.n_clusters,
            n_new_clusters: 0,
            n_total_clusters: current.n_clusters,
            max_cluster_size: new_stats.max_size,
            max_cluster_weight: new_stats.max_weight,
        });
        stats = new_stats;
    }

    // Component-level Dijkstra assignment (multi-hop)
    if use_component_merge {
        let n_small_before = stats.n_small;
        if n_small_before > 0 {
            let n_before = current.n_clusters;
            let comp_round = rounds.len() as i32;
            component_merge_remaining(
                graph,
                &mut current,
                min_size,
                min_weight,
                component_max_weight,
                &stats,
                &mut changed_at,
                comp_round,
                &mut ws,
            );

            let new_stats = cluster_stats(&current, nw, min_weight, min_size);
            let n_small_after = new_stats.n_small;

            rounds.push(PostprocessRound {
                round: rounds.len(),
                gamma: 0.0,
                method: "component_dijkstra".to_string(),
                n_small_before,
                n_small_after,
                n_merged: n_before - current.n_clusters,
                n_new_clusters: 0,
                n_total_clusters: current.n_clusters,
                max_cluster_size: new_stats.max_size,
                max_cluster_weight: new_stats.max_weight,
            });
        }
    }

    PostprocessResult {
        clustering: current,
        rounds,
        changed_at_round: changed_at,
    }
}

fn apply_merge_targets(
    clustering: &mut Clustering,
    merge_target: &[usize],
    changed_at: &mut [i32],
    round: i32,
) {
    debug_assert_eq!(changed_at.len(), clustering.n_nodes);
    for node in 0..clustering.n_nodes {
        let cid = clustering.clusters[node] as usize;
        let target = merge_target[cid];
        if target != usize::MAX {
            let target = target as u32;
            if target != clustering.clusters[node] && changed_at[node] == -1 {
                changed_at[node] = round;
            }
            clustering.clusters[node] = target;
        }
    }
    clustering.remove_empty_clusters();
}

/// Greedy fallback: merge remaining small clusters via cluster graph.
fn greedy_merge_remaining(
    graph: &Graph,
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
) {
    let sizes = &stats.sizes;
    let weights = &stats.weights;
    let n_cls = clustering.n_clusters;
    let anchor_mask: Vec<bool> = (0..n_cls)
        .map(|cid| !is_small(weights[cid], sizes[cid], min_weight, min_size))
        .collect();

    let cg = create_reduced_network(graph, clustering, false, ws);

    let mut merge_target = vec![usize::MAX; n_cls];
    let mut projected_weights = weights.clone();
    let mut small_clusters: Vec<usize> = (0..n_cls)
        .filter(|&cid| is_small(weights[cid], sizes[cid], min_weight, min_size))
        .collect();
    small_clusters.sort_by(|&a, &b| {
        weights[b]
            .partial_cmp(&weights[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

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
            merge_target[cid] = target;
            projected_weights[target] += weights[cid];
        }
    }

    apply_merge_targets(clustering, &merge_target, changed_at, round);
}

/// Component-level assignment: multi-hop Dijkstra from anchors.
///
/// 1. Build cluster graph
/// 2. Find connected components of small clusters
/// 3. Components touching anchors → Dijkstra nearest-anchor assignment
/// 4. Components with no anchor → pick forced anchor (largest weight), then Dijkstra
/// 5. Respect max_weight soft cap (least-overflow fallback)
fn component_merge_remaining(
    graph: &Graph,
    clustering: &mut Clustering,
    min_size: usize,
    min_weight: f64,
    max_weight_cap: f64,
    stats: &ClusterStats,
    changed_at: &mut [i32],
    round: i32,
    ws: &mut Workspace,
) {
    let sizes = &stats.sizes;
    let weights = &stats.weights;
    let n_cls = clustering.n_clusters;

    let cg = create_reduced_network(graph, clustering, false, ws);
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
    let mut merge_target = vec![usize::MAX; n_cls];
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
        comp_smalls.sort_by(|&a, &b| weights[b].total_cmp(&weights[a]));

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
                    merge_target[small_cid] = best_anchor;
                    projected_weights[best_anchor] += weights[small_cid];
                    continue;
                }
            }
            merge_target[small_cid] = target;
            projected_weights[target] += weights[small_cid];
        }

        for &node in &touched {
            dist[node] = f64::INFINITY;
            nearest_anchor[node] = usize::MAX;
        }
    }

    apply_merge_targets(clustering, &merge_target, changed_at, round);
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
