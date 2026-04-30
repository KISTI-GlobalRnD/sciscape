//! Experimental cluster-graph diagnostics for SciSci adaptive refinement.
//!
//! These helpers do not change Leiden behavior. They build a contracted
//! cluster graph from an existing membership and report the cheap statistics
//! needed to decide whether macro merge/split probes are worth running.

use crate::clustering::Clustering;
use crate::contraction::create_reduced_network;
use crate::graph::Graph;
use crate::local_merge::{self, LocalMergeWorkspace};
use crate::workspace::Workspace;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

#[derive(Clone, Debug)]
pub struct MergeCandidate {
    pub source: u64,
    pub target: u64,
    pub edge_weight: f64,
    pub delta_q: f64,
    pub merged_weight: f64,
    pub size_band_gain: f64,
}

#[derive(Clone, Debug)]
pub struct ClusterGraphStats {
    pub block_count: Vec<u64>,
    pub doc_weight: Vec<f64>,
    pub internal_weight: Vec<f64>,
    pub external_weight: Vec<f64>,
    pub degree: Vec<u64>,
    pub top_neighbor: Vec<i64>,
    pub top_neighbor_weight: Vec<f64>,
    pub second_neighbor: Vec<i64>,
    pub second_neighbor_weight: Vec<f64>,
    pub neighbor_weight_ratio: Vec<f64>,
    pub conductance: Vec<f64>,
    pub leafness: Vec<f64>,
    pub band_distance: Vec<f64>,
    pub merge_candidates: Vec<MergeCandidate>,
}

#[derive(Clone, Debug)]
pub struct BoundaryMoveProbe {
    pub cluster: u64,
    pub block_count: u64,
    pub doc_weight: f64,
    pub internal_weight: f64,
    pub external_weight: f64,
    pub conductance: f64,
    pub leafness: f64,
    pub top_neighbor: i64,
    pub top_neighbor_weight: f64,
    pub second_neighbor: i64,
    pub second_neighbor_weight: f64,
    pub neighbor_weight_ratio: f64,
    pub positive_move_count: u64,
    pub positive_move_weight: f64,
    pub positive_delta_q: f64,
    pub near_neutral_move_count: u64,
    pub near_neutral_move_weight: f64,
    pub near_neutral_delta_q: f64,
    pub best_move_delta_q: f64,
    pub best_move_node: u64,
    pub best_move_target: i64,
    pub top_move_count: u64,
    pub second_move_count: u64,
}

#[derive(Clone, Debug)]
pub struct OversizeBoundaryMove {
    pub source: u64,
    pub target: u64,
    pub node: u64,
    pub node_weight: f64,
    pub delta_q: f64,
    pub source_weight_before: f64,
    pub source_weight_after: f64,
    pub target_weight_before: f64,
    pub target_weight_after: f64,
}

#[derive(Clone, Debug)]
pub struct BoundaryGroupProbe {
    pub cluster: u64,
    pub block_count: u64,
    pub doc_weight: f64,
    pub top_neighbor: i64,
    pub second_neighbor: i64,
    pub top_group_count: u64,
    pub top_group_weight: f64,
    pub top_group_to_target_weight: f64,
    pub top_group_cut_weight: f64,
    pub top_group_move_delta_q: f64,
    pub top_group_split_delta_q: f64,
    pub top_group_is_full_cluster: bool,
    pub second_group_count: u64,
    pub second_group_weight: f64,
    pub second_group_to_target_weight: f64,
    pub second_group_cut_weight: f64,
    pub second_group_move_delta_q: f64,
    pub second_group_split_delta_q: f64,
    pub second_group_is_full_cluster: bool,
    pub best_delta_q: f64,
    pub best_action: u8,
}

#[derive(Clone, Debug)]
pub struct ExternalGrainProbe {
    pub cluster: u64,
    pub block_count: u64,
    pub doc_weight: f64,
    pub incident_directed_edges: u64,
    pub source_directed_edges: u64,
    pub external_directed_edges: u64,
    pub n_external_groups: u64,
    pub assigned_count: u64,
    pub assigned_weight: f64,
    pub assigned_fraction: f64,
    pub largest_group_target: i64,
    pub largest_group_count: u64,
    pub largest_group_weight: f64,
    pub largest_group_fraction: f64,
    pub largest_group_to_target_weight: f64,
    pub largest_group_cut_weight: f64,
    pub largest_group_move_delta_q: f64,
    pub largest_group_split_delta_q: f64,
    pub second_group_target: i64,
    pub second_group_weight: f64,
    pub second_group_fraction: f64,
    pub best_group_target: i64,
    pub best_group_count: u64,
    pub best_group_weight: f64,
    pub best_group_fraction: f64,
    pub best_group_to_target_weight: f64,
    pub best_group_cut_weight: f64,
    pub best_group_move_delta_q: f64,
    pub best_group_split_delta_q: f64,
    pub best_group_delta_q: f64,
    pub best_group_action: u8,
    pub positive_group_count: u64,
    pub positive_group_weight: f64,
    pub near_neutral_group_count: u64,
    pub near_neutral_group_weight: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct ExternalGrainSelectionPolicy {
    pub min_doc_weight: f64,
    pub max_incident_directed_edges: u64,
    pub min_best_delta_q: f64,
    pub min_assigned_fraction: f64,
    pub min_best_group_fraction: f64,
}

impl Default for ExternalGrainSelectionPolicy {
    fn default() -> Self {
        Self {
            min_doc_weight: 0.0,
            max_incident_directed_edges: 0,
            min_best_delta_q: 0.0,
            min_assigned_fraction: 0.0,
            min_best_group_fraction: 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ExternalGrainSelection {
    pub recommended_for_split_repair: bool,
    pub priority: f64,
}

#[derive(Clone, Debug)]
pub struct MultiCoreSplitProbe {
    pub cluster: u64,
    pub gamma_multiplier: f64,
    pub probe_resolution: f64,
    pub block_count: u64,
    pub doc_weight: f64,
    pub internal_weight: f64,
    pub induced_directed_edges: u64,
    pub n_parts: u64,
    pub non_singleton_parts: u64,
    pub singleton_parts: u64,
    pub singleton_weight: f64,
    pub core_part_count: u64,
    pub core_part_weight: f64,
    pub largest_part_weight: f64,
    pub second_part_weight: f64,
    pub largest_part_fraction: f64,
    pub cut_weight: f64,
    pub split_delta_q_base: f64,
    pub split_delta_q_probe: f64,
    pub hysteresis_only: bool,
}

#[derive(Clone, Debug)]
pub struct SplitMergeRepairProbe {
    pub cluster: u64,
    pub gamma_multiplier: f64,
    pub probe_resolution: f64,
    pub block_count: u64,
    pub doc_weight: f64,
    pub induced_directed_edges: u64,
    pub n_parts: u64,
    pub core_part_count: u64,
    pub singleton_weight: f64,
    pub cut_weight: f64,
    pub split_delta_q_base: f64,
    pub split_delta_q_probe: f64,
    pub repair_quotient_edges: u64,
    pub repair_merge_count: u64,
    pub repair_delta_q: f64,
    pub net_delta_q: f64,
    pub final_source_units: u64,
    pub retained_source_units: u64,
    pub escaped_source_units: u64,
    pub escaped_source_weight: f64,
    pub final_small_source_units: u64,
    pub final_small_source_weight: f64,
    pub largest_source_unit_fraction: f64,
    pub restored_source_cluster: bool,
}

#[derive(Clone, Debug)]
pub struct SplitRepairAppliedCandidate {
    pub selected_index: u64,
    pub cluster: u64,
    pub gamma_multiplier: f64,
    pub probe_resolution: f64,
    pub block_count: u64,
    pub doc_weight: f64,
    pub n_parts: u64,
    pub split_delta_q_base: f64,
    pub repair_delta_q: f64,
    pub predicted_net_delta_q: f64,
    pub repair_merge_count: u64,
    pub final_source_units: u64,
    pub retained_source_units: u64,
    pub escaped_source_units: u64,
    pub escaped_source_weight: f64,
    pub final_small_source_units: u64,
    pub final_small_source_weight: f64,
    pub largest_source_unit_fraction: f64,
    pub changed_nodes: u64,
    pub moved_to_existing_cluster_nodes: u64,
    pub moved_to_new_cluster_nodes: u64,
    pub new_retained_clusters: u64,
}

#[derive(Clone, Copy, Debug)]
struct GroupEval {
    count: u64,
    weight: f64,
    to_target_weight: f64,
    cut_weight: f64,
    move_delta_q: f64,
    split_delta_q: f64,
    is_full_cluster: bool,
}

#[derive(Clone, Debug)]
struct HeapCandidate(MergeCandidate);

impl PartialEq for HeapCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.0.delta_q.total_cmp(&other.0.delta_q) == Ordering::Equal
            && self.0.source == other.0.source
            && self.0.target == other.0.target
    }
}

impl Eq for HeapCandidate {}

impl PartialOrd for HeapCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse by score so BinaryHeap::peek returns the current worst
        // retained candidate. This keeps memory bounded at top_k.
        other
            .0
            .delta_q
            .total_cmp(&self.0.delta_q)
            .then_with(|| other.0.source.cmp(&self.0.source))
            .then_with(|| other.0.target.cmp(&self.0.target))
    }
}

#[inline]
fn distance_to_band(weight: f64, min_weight: f64, max_weight: f64) -> f64 {
    if min_weight > 0.0 && weight < min_weight {
        min_weight - weight
    } else if max_weight > 0.0 && weight > max_weight {
        weight - max_weight
    } else {
        0.0
    }
}

fn push_top_candidate(
    heap: &mut BinaryHeap<HeapCandidate>,
    top_k: usize,
    candidate: MergeCandidate,
) {
    if top_k == 0 || !candidate.delta_q.is_finite() {
        return;
    }
    let wrapped = HeapCandidate(candidate);
    if heap.len() < top_k {
        heap.push(wrapped);
        return;
    }
    if let Some(worst) = heap.peek() {
        if wrapped.0.delta_q > worst.0.delta_q {
            heap.pop();
            heap.push(wrapped);
        }
    }
}

#[inline]
fn update_top_two(
    nbr_cluster: usize,
    weight: f64,
    top_neighbor: &mut i64,
    top_weight: &mut f64,
    second_neighbor: &mut i64,
    second_weight: &mut f64,
) {
    if weight > *top_weight {
        *second_weight = *top_weight;
        *second_neighbor = *top_neighbor;
        *top_weight = weight;
        *top_neighbor = nbr_cluster as i64;
    } else if weight > *second_weight {
        *second_weight = weight;
        *second_neighbor = nbr_cluster as i64;
    }
}

fn empty_group_eval() -> GroupEval {
    GroupEval {
        count: 0,
        weight: 0.0,
        to_target_weight: 0.0,
        cut_weight: 0.0,
        move_delta_q: f64::NEG_INFINITY,
        split_delta_q: f64::NEG_INFINITY,
        is_full_cluster: false,
    }
}

fn evaluate_boundary_group(
    graph: &Graph,
    clustering: &Clustering,
    nodes: &[usize],
    selected: &[u8],
    source_cluster: usize,
    target_cluster: Option<usize>,
    source_weight: f64,
    resolution: f64,
    local_index: &[u32],
    cluster_weights: &[f64],
) -> GroupEval {
    let mut count = 0u64;
    let mut group_weight = 0.0;
    let mut to_target_weight = 0.0;
    let mut cut_weight = 0.0;
    for (local, &node) in nodes.iter().enumerate() {
        if selected[local] == 0 {
            continue;
        }
        count += 1;
        group_weight += graph.node_weights[node];
        let nbr_start = graph.first_neighbor_index[node] as usize;
        let nbr_end = graph.first_neighbor_index[node + 1] as usize;
        for edge_idx in nbr_start..nbr_end {
            let nbr = graph.neighbors[edge_idx] as usize;
            let nbr_cluster = clustering.clusters[nbr] as usize;
            let weight = graph.edge_weights[edge_idx];
            if Some(nbr_cluster) == target_cluster {
                to_target_weight += weight;
            } else if nbr_cluster == source_cluster {
                let local_nbr = local_index[nbr];
                if local_nbr == u32::MAX || selected[local_nbr as usize] == 0 {
                    cut_weight += weight;
                }
            }
        }
    }

    if count == 0 {
        return empty_group_eval();
    }

    let split_delta_q = -cut_weight + resolution * group_weight * (source_weight - group_weight);
    let move_delta_q = if let Some(target) = target_cluster {
        to_target_weight
            - cut_weight
            - resolution * group_weight * (cluster_weights[target] - source_weight + group_weight)
    } else {
        f64::NEG_INFINITY
    };
    GroupEval {
        count,
        weight: group_weight,
        to_target_weight,
        cut_weight,
        move_delta_q,
        split_delta_q,
        is_full_cluster: count as usize == nodes.len(),
    }
}

pub fn cluster_graph_stats(
    graph: &Graph,
    clustering: &Clustering,
    resolution: f64,
    min_weight: f64,
    max_weight: f64,
    top_k: usize,
    ws: &mut Workspace,
) -> ClusterGraphStats {
    assert_eq!(clustering.n_nodes, graph.n_nodes);

    let n_clusters = clustering.n_clusters;
    let mut block_count = vec![0u64; n_clusters];
    for &cluster in &clustering.clusters {
        block_count[cluster as usize] += 1;
    }

    let reduced = create_reduced_network(graph, clustering, true, ws);
    let doc_weight = reduced.node_weights.clone();
    let internal_weight = reduced.self_loop_weights.clone();
    let mut external_weight = vec![0.0f64; n_clusters];
    let mut degree = vec![0u64; n_clusters];
    let mut top_neighbor = vec![-1i64; n_clusters];
    let mut top_neighbor_weight = vec![0.0f64; n_clusters];
    let mut second_neighbor = vec![-1i64; n_clusters];
    let mut second_neighbor_weight = vec![0.0f64; n_clusters];
    let mut band_distance = vec![0.0f64; n_clusters];
    let mut candidate_heap: BinaryHeap<HeapCandidate> = BinaryHeap::with_capacity(top_k.min(4096));

    for c in 0..n_clusters {
        band_distance[c] = distance_to_band(doc_weight[c], min_weight, max_weight);
    }

    for c in 0..n_clusters {
        let weight = doc_weight[c];
        let start = reduced.first_neighbor_index[c] as usize;
        let end = reduced.first_neighbor_index[c + 1] as usize;
        degree[c] = (end - start) as u64;

        for edge_idx in start..end {
            let nbr = reduced.neighbors[edge_idx] as usize;
            let edge_weight = reduced.edge_weights[edge_idx];
            external_weight[c] += edge_weight;
            update_top_two(
                nbr,
                edge_weight,
                &mut top_neighbor[c],
                &mut top_neighbor_weight[c],
                &mut second_neighbor[c],
                &mut second_neighbor_weight[c],
            );

            if c < nbr {
                let merged_weight = weight + doc_weight[nbr];
                let before_distance = band_distance[c] + band_distance[nbr];
                let after_distance = distance_to_band(merged_weight, min_weight, max_weight);
                let candidate = MergeCandidate {
                    source: c as u64,
                    target: nbr as u64,
                    edge_weight,
                    delta_q: edge_weight - resolution * weight * doc_weight[nbr],
                    merged_weight,
                    size_band_gain: before_distance - after_distance,
                };
                push_top_candidate(&mut candidate_heap, top_k, candidate);
            }
        }
    }

    let mut conductance = vec![0.0f64; n_clusters];
    let mut leafness = vec![0.0f64; n_clusters];
    let mut neighbor_weight_ratio = vec![0.0f64; n_clusters];
    for c in 0..n_clusters {
        let external = external_weight[c];
        let volume = 2.0 * internal_weight[c] + external;
        if volume > 0.0 {
            conductance[c] = external / volume;
        }
        if external > 0.0 {
            leafness[c] = top_neighbor_weight[c] / external;
        }
        if top_neighbor_weight[c] > 0.0 {
            neighbor_weight_ratio[c] = second_neighbor_weight[c] / top_neighbor_weight[c];
        }
    }

    let mut merge_candidates: Vec<MergeCandidate> = candidate_heap
        .into_iter()
        .map(|candidate| candidate.0)
        .collect();
    merge_candidates.sort_by(|a, b| {
        b.delta_q
            .total_cmp(&a.delta_q)
            .then_with(|| a.source.cmp(&b.source))
            .then_with(|| a.target.cmp(&b.target))
    });

    ClusterGraphStats {
        block_count,
        doc_weight,
        internal_weight,
        external_weight,
        degree,
        top_neighbor,
        top_neighbor_weight,
        second_neighbor,
        second_neighbor_weight,
        neighbor_weight_ratio,
        conductance,
        leafness,
        band_distance,
        merge_candidates,
    }
}

pub fn boundary_move_probes(
    graph: &Graph,
    clustering: &Clustering,
    candidate_clusters: &[u64],
    resolution: f64,
    epsilon: f64,
    ws: &mut Workspace,
) -> Vec<BoundaryMoveProbe> {
    assert_eq!(clustering.n_nodes, graph.n_nodes);

    let n_clusters = clustering.n_clusters;
    clustering.fill_cluster_groups_and_weights(&graph.node_weights, ws);
    ws.temp_used.clear();
    let mut out = Vec::with_capacity(candidate_clusters.len());
    let clusters = clustering.clusters.as_slice();

    for &cluster_u64 in candidate_clusters {
        let Ok(cluster_u32) = u32::try_from(cluster_u64) else {
            continue;
        };
        let c = cluster_u32 as usize;
        if c >= n_clusters {
            continue;
        }

        let start = ws.npc_starts[c] as usize;
        let end = ws.npc_starts[c + 1] as usize;
        if start == end {
            continue;
        }
        let doc_weight = ws.cw[c];

        let mut self_loop_weight = 0.0;
        let mut internal_directed_weight = 0.0;
        let mut external_weight = 0.0;
        let mut top_neighbor = -1i64;
        let mut top_neighbor_weight = 0.0;
        let mut second_neighbor = -1i64;
        let mut second_neighbor_weight = 0.0;

        for pos in start..end {
            let node = ws.npc_nodes[pos] as usize;
            self_loop_weight += graph.self_loop_weights[node];
            let nbr_start = graph.first_neighbor_index[node] as usize;
            let nbr_end = graph.first_neighbor_index[node + 1] as usize;
            for edge_idx in nbr_start..nbr_end {
                let nbr = graph.neighbors[edge_idx] as usize;
                let nbr_cluster = clusters[nbr] as usize;
                let weight = graph.edge_weights[edge_idx];
                if nbr_cluster == c {
                    internal_directed_weight += weight;
                    continue;
                }

                external_weight += weight;
                if ws.temp_seen[nbr_cluster] != cluster_u32 {
                    ws.temp_seen[nbr_cluster] = cluster_u32;
                    ws.temp_w[nbr_cluster] = 0.0;
                    ws.temp_used.push(nbr_cluster as u32);
                }
                ws.temp_w[nbr_cluster] += weight;
            }
        }

        for &nbr_cluster_u32 in &ws.temp_used {
            let nbr_cluster = nbr_cluster_u32 as usize;
            update_top_two(
                nbr_cluster,
                ws.temp_w[nbr_cluster],
                &mut top_neighbor,
                &mut top_neighbor_weight,
                &mut second_neighbor,
                &mut second_neighbor_weight,
            );
        }

        let top_cluster = (top_neighbor >= 0).then_some(top_neighbor as usize);
        let second_cluster = (second_neighbor >= 0).then_some(second_neighbor as usize);
        let mut positive_move_count = 0u64;
        let mut positive_move_weight = 0.0;
        let mut positive_delta_q = 0.0;
        let mut near_neutral_move_count = 0u64;
        let mut near_neutral_move_weight = 0.0;
        let mut near_neutral_delta_q = 0.0;
        let mut best_move_delta_q = f64::NEG_INFINITY;
        let mut best_move_node = u64::MAX;
        let mut best_move_target = -1i64;
        let mut top_move_count = 0u64;
        let mut second_move_count = 0u64;

        for pos in start..end {
            let node = ws.npc_nodes[pos] as usize;
            let node_weight = graph.node_weights[node];
            let mut weight_to_current = 0.0;
            let mut weight_to_top = 0.0;
            let mut weight_to_second = 0.0;
            let nbr_start = graph.first_neighbor_index[node] as usize;
            let nbr_end = graph.first_neighbor_index[node + 1] as usize;
            for edge_idx in nbr_start..nbr_end {
                let nbr = graph.neighbors[edge_idx] as usize;
                let nbr_cluster = clusters[nbr] as usize;
                let weight = graph.edge_weights[edge_idx];
                if nbr_cluster == c {
                    weight_to_current += weight;
                } else if Some(nbr_cluster) == top_cluster {
                    weight_to_top += weight;
                } else if Some(nbr_cluster) == second_cluster {
                    weight_to_second += weight;
                }
            }

            let current_increment =
                weight_to_current - node_weight * (doc_weight - node_weight) * resolution;
            let mut best_delta = f64::NEG_INFINITY;
            let mut best_target = -1i64;
            if let Some(target) = top_cluster {
                let target_increment = weight_to_top - node_weight * ws.cw[target] * resolution;
                best_delta = target_increment - current_increment;
                best_target = target as i64;
            }
            if let Some(target) = second_cluster {
                let target_increment = weight_to_second - node_weight * ws.cw[target] * resolution;
                let delta = target_increment - current_increment;
                if delta > best_delta {
                    best_delta = delta;
                    best_target = target as i64;
                }
            }

            if best_delta > best_move_delta_q {
                best_move_delta_q = best_delta;
                best_move_node = node as u64;
                best_move_target = best_target;
            }
            if best_delta > 0.0 {
                positive_move_count += 1;
                positive_move_weight += node_weight;
                positive_delta_q += best_delta;
                if best_target == top_neighbor {
                    top_move_count += 1;
                } else if best_target == second_neighbor {
                    second_move_count += 1;
                }
            }
            if best_delta >= -epsilon {
                near_neutral_move_count += 1;
                near_neutral_move_weight += node_weight;
                near_neutral_delta_q += best_delta;
            }
        }

        if !best_move_delta_q.is_finite() {
            best_move_delta_q = 0.0;
        }

        let internal_weight = internal_directed_weight / 2.0 + self_loop_weight;
        let volume = 2.0 * internal_weight + external_weight;
        let conductance = if volume > 0.0 {
            external_weight / volume
        } else {
            0.0
        };
        let leafness = if external_weight > 0.0 {
            top_neighbor_weight / external_weight
        } else {
            0.0
        };
        let neighbor_weight_ratio = if top_neighbor_weight > 0.0 {
            second_neighbor_weight / top_neighbor_weight
        } else {
            0.0
        };

        out.push(BoundaryMoveProbe {
            cluster: cluster_u64,
            block_count: (end - start) as u64,
            doc_weight,
            internal_weight,
            external_weight,
            conductance,
            leafness,
            top_neighbor,
            top_neighbor_weight,
            second_neighbor,
            second_neighbor_weight,
            neighbor_weight_ratio,
            positive_move_count,
            positive_move_weight,
            positive_delta_q,
            near_neutral_move_count,
            near_neutral_move_weight,
            near_neutral_delta_q,
            best_move_delta_q,
            best_move_node,
            best_move_target,
            top_move_count,
            second_move_count,
        });

        for nbr_cluster_u32 in ws.temp_used.drain(..) {
            let nbr_cluster = nbr_cluster_u32 as usize;
            ws.temp_seen[nbr_cluster] = u32::MAX;
            ws.temp_w[nbr_cluster] = 0.0;
        }
    }

    out
}

pub fn trim_oversize_boundary_moves(
    graph: &Graph,
    clustering: &Clustering,
    candidate_clusters: &[u64],
    resolution: f64,
    target_max_weight: f64,
    min_delta_q: f64,
    max_moves_per_cluster: usize,
    ws: &mut Workspace,
) -> (Vec<u64>, Vec<OversizeBoundaryMove>) {
    assert_eq!(clustering.n_nodes, graph.n_nodes);

    if target_max_weight <= 0.0 {
        return (
            clustering.clusters.iter().copied().map(u64::from).collect(),
            Vec::new(),
        );
    }

    let n_clusters = clustering.n_clusters;
    clustering.fill_cluster_groups_and_weights(&graph.node_weights, ws);
    ws.temp_seen[..n_clusters].fill(u32::MAX);
    ws.temp_w[..n_clusters].fill(0.0);
    ws.temp_used.clear();

    let mut proposed = clustering.clusters.clone();
    let mut cluster_weights = ws.cw[..n_clusters].to_vec();
    let mut moves = Vec::new();
    let move_budget = if max_moves_per_cluster == 0 {
        usize::MAX
    } else {
        max_moves_per_cluster
    };

    for &cluster_u64 in candidate_clusters {
        let Ok(cluster_u32) = u32::try_from(cluster_u64) else {
            continue;
        };
        let c = cluster_u32 as usize;
        if c >= n_clusters {
            continue;
        }

        let start = ws.npc_starts[c] as usize;
        let end = ws.npc_starts[c + 1] as usize;
        let mut moves_for_cluster = 0usize;
        while cluster_weights[c] > target_max_weight && moves_for_cluster < move_budget {
            let mut best_node = usize::MAX;
            let mut best_target = usize::MAX;
            let mut best_node_weight = 0.0;
            let mut best_delta_q = f64::NEG_INFINITY;

            for pos in start..end {
                let node = ws.npc_nodes[pos] as usize;
                if proposed[node] as usize != c {
                    continue;
                }

                let node_weight = graph.node_weights[node];
                let mut weight_to_current = 0.0;
                let nbr_start = graph.first_neighbor_index[node] as usize;
                let nbr_end = graph.first_neighbor_index[node + 1] as usize;
                for edge_idx in nbr_start..nbr_end {
                    let nbr = graph.neighbors[edge_idx] as usize;
                    let nbr_cluster = proposed[nbr] as usize;
                    let weight = graph.edge_weights[edge_idx];
                    if nbr_cluster == c {
                        weight_to_current += weight;
                        continue;
                    }
                    if nbr_cluster >= n_clusters {
                        continue;
                    }
                    if ws.temp_seen[nbr_cluster] != 0 {
                        ws.temp_seen[nbr_cluster] = 0;
                        ws.temp_w[nbr_cluster] = 0.0;
                        ws.temp_used.push(nbr_cluster as u32);
                    }
                    ws.temp_w[nbr_cluster] += weight;
                }

                let current_increment = weight_to_current
                    - node_weight * (cluster_weights[c] - node_weight) * resolution;
                let mut node_best_target = usize::MAX;
                let mut node_best_delta_q = f64::NEG_INFINITY;
                for &target_u32 in ws.temp_used.iter() {
                    let target = target_u32 as usize;
                    if cluster_weights[target] + node_weight > target_max_weight {
                        continue;
                    }
                    let target_increment =
                        ws.temp_w[target] - node_weight * cluster_weights[target] * resolution;
                    let delta_q = target_increment - current_increment;
                    if delta_q > node_best_delta_q
                        || (delta_q == node_best_delta_q && target < node_best_target)
                    {
                        node_best_target = target;
                        node_best_delta_q = delta_q;
                    }
                }

                for target_u32 in ws.temp_used.drain(..) {
                    let target = target_u32 as usize;
                    ws.temp_seen[target] = u32::MAX;
                    ws.temp_w[target] = 0.0;
                }

                if node_best_target == usize::MAX || node_best_delta_q < min_delta_q {
                    continue;
                }
                if node_best_delta_q > best_delta_q
                    || (node_best_delta_q == best_delta_q
                        && (node_weight > best_node_weight
                            || (node_weight == best_node_weight && node < best_node)))
                {
                    best_node = node;
                    best_target = node_best_target;
                    best_node_weight = node_weight;
                    best_delta_q = node_best_delta_q;
                }
            }

            if best_node == usize::MAX {
                break;
            }

            let source_weight_before = cluster_weights[c];
            let target_weight_before = cluster_weights[best_target];
            let source_weight_after = source_weight_before - best_node_weight;
            let target_weight_after = target_weight_before + best_node_weight;
            proposed[best_node] = best_target as u32;
            cluster_weights[c] = source_weight_after;
            cluster_weights[best_target] = target_weight_after;
            moves.push(OversizeBoundaryMove {
                source: cluster_u64,
                target: best_target as u64,
                node: best_node as u64,
                node_weight: best_node_weight,
                delta_q: best_delta_q,
                source_weight_before,
                source_weight_after,
                target_weight_before,
                target_weight_after,
            });
            moves_for_cluster += 1;
        }
    }

    (proposed.into_iter().map(u64::from).collect(), moves)
}

pub fn boundary_group_probes(
    graph: &Graph,
    clustering: &Clustering,
    candidate_clusters: &[u64],
    resolution: f64,
    ws: &mut Workspace,
) -> Vec<BoundaryGroupProbe> {
    assert_eq!(clustering.n_nodes, graph.n_nodes);

    let n_clusters = clustering.n_clusters;
    clustering.fill_cluster_groups_and_weights(&graph.node_weights, ws);
    ws.temp_used.clear();
    let mut out = Vec::with_capacity(candidate_clusters.len());
    let clusters = clustering.clusters.as_slice();

    for &cluster_u64 in candidate_clusters {
        let Ok(cluster_u32) = u32::try_from(cluster_u64) else {
            continue;
        };
        let c = cluster_u32 as usize;
        if c >= n_clusters {
            continue;
        }

        let start = ws.npc_starts[c] as usize;
        let end = ws.npc_starts[c + 1] as usize;
        if start == end {
            continue;
        }
        let doc_weight = ws.cw[c];
        let nodes: Vec<usize> = ws.npc_nodes[start..end]
            .iter()
            .map(|&node| node as usize)
            .collect();
        for (local, &node) in nodes.iter().enumerate() {
            ws.local_index[node] = local as u32;
        }

        let mut top_neighbor = -1i64;
        let mut top_neighbor_weight = 0.0;
        let mut second_neighbor = -1i64;
        let mut second_neighbor_weight = 0.0;

        for &node in &nodes {
            let nbr_start = graph.first_neighbor_index[node] as usize;
            let nbr_end = graph.first_neighbor_index[node + 1] as usize;
            for edge_idx in nbr_start..nbr_end {
                let nbr = graph.neighbors[edge_idx] as usize;
                let nbr_cluster = clusters[nbr] as usize;
                if nbr_cluster == c {
                    continue;
                }
                if ws.temp_seen[nbr_cluster] != cluster_u32 {
                    ws.temp_seen[nbr_cluster] = cluster_u32;
                    ws.temp_w[nbr_cluster] = 0.0;
                    ws.temp_used.push(nbr_cluster as u32);
                }
                ws.temp_w[nbr_cluster] += graph.edge_weights[edge_idx];
            }
        }

        for &nbr_cluster_u32 in &ws.temp_used {
            let nbr_cluster = nbr_cluster_u32 as usize;
            update_top_two(
                nbr_cluster,
                ws.temp_w[nbr_cluster],
                &mut top_neighbor,
                &mut top_neighbor_weight,
                &mut second_neighbor,
                &mut second_neighbor_weight,
            );
        }
        let top_cluster = (top_neighbor >= 0).then_some(top_neighbor as usize);
        let second_cluster = (second_neighbor >= 0).then_some(second_neighbor as usize);

        let mut top_selected = vec![0u8; nodes.len()];
        let mut second_selected = vec![0u8; nodes.len()];
        for (local, &node) in nodes.iter().enumerate() {
            let mut weight_to_top = 0.0;
            let mut weight_to_second = 0.0;
            let nbr_start = graph.first_neighbor_index[node] as usize;
            let nbr_end = graph.first_neighbor_index[node + 1] as usize;
            for edge_idx in nbr_start..nbr_end {
                let nbr = graph.neighbors[edge_idx] as usize;
                let nbr_cluster = clusters[nbr] as usize;
                let weight = graph.edge_weights[edge_idx];
                if Some(nbr_cluster) == top_cluster {
                    weight_to_top += weight;
                } else if Some(nbr_cluster) == second_cluster {
                    weight_to_second += weight;
                }
            }
            if weight_to_top > 0.0 && weight_to_top >= weight_to_second {
                top_selected[local] = 1;
            } else if weight_to_second > 0.0 {
                second_selected[local] = 1;
            }
        }

        let top_eval = evaluate_boundary_group(
            graph,
            clustering,
            &nodes,
            &top_selected,
            c,
            top_cluster,
            doc_weight,
            resolution,
            &ws.local_index,
            &ws.cw,
        );
        let second_eval = evaluate_boundary_group(
            graph,
            clustering,
            &nodes,
            &second_selected,
            c,
            second_cluster,
            doc_weight,
            resolution,
            &ws.local_index,
            &ws.cw,
        );

        let mut best_delta_q = top_eval.move_delta_q;
        let mut best_action = 1u8;
        if second_eval.move_delta_q > best_delta_q {
            best_delta_q = second_eval.move_delta_q;
            best_action = 2;
        }
        if top_eval.split_delta_q > best_delta_q {
            best_delta_q = top_eval.split_delta_q;
            best_action = 3;
        }
        if second_eval.split_delta_q > best_delta_q {
            best_delta_q = second_eval.split_delta_q;
            best_action = 4;
        }
        if !best_delta_q.is_finite() {
            best_delta_q = 0.0;
            best_action = 0;
        } else if best_delta_q <= 0.0 {
            best_action = 0;
        }

        out.push(BoundaryGroupProbe {
            cluster: cluster_u64,
            block_count: nodes.len() as u64,
            doc_weight,
            top_neighbor,
            second_neighbor,
            top_group_count: top_eval.count,
            top_group_weight: top_eval.weight,
            top_group_to_target_weight: top_eval.to_target_weight,
            top_group_cut_weight: top_eval.cut_weight,
            top_group_move_delta_q: top_eval.move_delta_q,
            top_group_split_delta_q: top_eval.split_delta_q,
            top_group_is_full_cluster: top_eval.is_full_cluster,
            second_group_count: second_eval.count,
            second_group_weight: second_eval.weight,
            second_group_to_target_weight: second_eval.to_target_weight,
            second_group_cut_weight: second_eval.cut_weight,
            second_group_move_delta_q: second_eval.move_delta_q,
            second_group_split_delta_q: second_eval.split_delta_q,
            second_group_is_full_cluster: second_eval.is_full_cluster,
            best_delta_q,
            best_action,
        });

        for &node in &nodes {
            ws.local_index[node] = u32::MAX;
        }
        for nbr_cluster_u32 in ws.temp_used.drain(..) {
            let nbr_cluster = nbr_cluster_u32 as usize;
            ws.temp_seen[nbr_cluster] = u32::MAX;
            ws.temp_w[nbr_cluster] = 0.0;
        }
    }

    out
}

pub fn external_grain_probes(
    graph: &Graph,
    clustering: &Clustering,
    candidate_clusters: &[u64],
    resolution: f64,
    epsilon: f64,
    ws: &mut Workspace,
) -> Vec<ExternalGrainProbe> {
    assert_eq!(clustering.n_nodes, graph.n_nodes);

    let n_clusters = clustering.n_clusters;
    clustering.fill_cluster_groups_and_weights(&graph.node_weights, ws);
    let clusters = clustering.clusters.as_slice();
    let mut out = Vec::with_capacity(candidate_clusters.len());

    for &cluster_u64 in candidate_clusters {
        let Ok(cluster_u32) = u32::try_from(cluster_u64) else {
            continue;
        };
        let c = cluster_u32 as usize;
        if c >= n_clusters {
            continue;
        }

        let start = ws.npc_starts[c] as usize;
        let end = ws.npc_starts[c + 1] as usize;
        if start == end {
            continue;
        }
        let nodes = &ws.npc_nodes[start..end];
        let doc_weight = ws.cw[c];

        for (local, &node_u32) in nodes.iter().enumerate() {
            ws.local_index[node_u32 as usize] = local as u32;
        }

        let mut target_to_group: HashMap<usize, usize> = HashMap::new();
        let mut group_target: Vec<usize> = Vec::new();
        let mut group_count: Vec<u64> = Vec::new();
        let mut group_weight: Vec<f64> = Vec::new();
        let mut group_to_target_weight: Vec<f64> = Vec::new();
        let mut group_cut_weight: Vec<f64> = Vec::new();
        let mut node_group = vec![usize::MAX; nodes.len()];
        let mut incident_directed_edges = 0u64;
        let mut source_directed_edges = 0u64;
        let mut external_directed_edges = 0u64;

        for (local, &node_u32) in nodes.iter().enumerate() {
            let node = node_u32 as usize;
            let nbr_start = graph.first_neighbor_index[node] as usize;
            let nbr_end = graph.first_neighbor_index[node + 1] as usize;
            incident_directed_edges += (nbr_end - nbr_start) as u64;
            for edge_idx in nbr_start..nbr_end {
                let nbr = graph.neighbors[edge_idx] as usize;
                let nbr_cluster = clusters[nbr] as usize;
                if nbr_cluster == c {
                    source_directed_edges += 1;
                    continue;
                }
                external_directed_edges += 1;
                if ws.temp_seen[nbr_cluster] == u32::MAX {
                    ws.temp_seen[nbr_cluster] = 0;
                    ws.temp_w[nbr_cluster] = 0.0;
                    ws.temp_used.push(nbr_cluster as u32);
                }
                ws.temp_w[nbr_cluster] += graph.edge_weights[edge_idx];
            }

            let mut best_target = usize::MAX;
            let mut best_weight = 0.0;
            for &nbr_cluster_u32 in &ws.temp_used {
                let nbr_cluster = nbr_cluster_u32 as usize;
                let weight = ws.temp_w[nbr_cluster];
                if weight > best_weight {
                    best_weight = weight;
                    best_target = nbr_cluster;
                }
            }
            for nbr_cluster_u32 in ws.temp_used.drain(..) {
                let nbr_cluster = nbr_cluster_u32 as usize;
                ws.temp_seen[nbr_cluster] = u32::MAX;
                ws.temp_w[nbr_cluster] = 0.0;
            }

            if best_target == usize::MAX || best_weight <= 0.0 {
                continue;
            }
            let group = if let Some(&group) = target_to_group.get(&best_target) {
                group
            } else {
                let group = group_target.len();
                target_to_group.insert(best_target, group);
                group_target.push(best_target);
                group_count.push(0);
                group_weight.push(0.0);
                group_to_target_weight.push(0.0);
                group_cut_weight.push(0.0);
                group
            };
            node_group[local] = group;
            group_count[group] += 1;
            group_weight[group] += graph.node_weights[node];
        }

        for (local, &node_u32) in nodes.iter().enumerate() {
            let group = node_group[local];
            if group == usize::MAX {
                continue;
            }
            let target = group_target[group];
            let node = node_u32 as usize;
            let nbr_start = graph.first_neighbor_index[node] as usize;
            let nbr_end = graph.first_neighbor_index[node + 1] as usize;
            for edge_idx in nbr_start..nbr_end {
                let nbr = graph.neighbors[edge_idx] as usize;
                let nbr_cluster = clusters[nbr] as usize;
                let weight = graph.edge_weights[edge_idx];
                if nbr_cluster == c {
                    let local_nbr = ws.local_index[nbr] as usize;
                    if local_nbr >= node_group.len() || node_group[local_nbr] != group {
                        group_cut_weight[group] += weight;
                    }
                } else if nbr_cluster == target {
                    group_to_target_weight[group] += weight;
                }
            }
        }

        for &node_u32 in nodes {
            ws.local_index[node_u32 as usize] = u32::MAX;
        }

        let mut assigned_count = 0u64;
        let mut assigned_weight = 0.0;
        let mut largest_group = usize::MAX;
        let mut second_group = usize::MAX;
        let mut best_group = usize::MAX;
        let mut best_group_delta_q = f64::NEG_INFINITY;
        let mut best_group_action = 0u8;
        let mut positive_group_count = 0u64;
        let mut positive_group_weight = 0.0;
        let mut near_neutral_group_count = 0u64;
        let mut near_neutral_group_weight = 0.0;
        let mut group_move_delta_q = vec![f64::NEG_INFINITY; group_target.len()];
        let mut group_split_delta_q = vec![f64::NEG_INFINITY; group_target.len()];

        for group in 0..group_target.len() {
            assigned_count += group_count[group];
            assigned_weight += group_weight[group];
            if largest_group == usize::MAX || group_weight[group] > group_weight[largest_group] {
                second_group = largest_group;
                largest_group = group;
            } else if second_group == usize::MAX || group_weight[group] > group_weight[second_group]
            {
                second_group = group;
            }

            let target = group_target[group];
            let split_delta_q = -group_cut_weight[group]
                + resolution * group_weight[group] * (doc_weight - group_weight[group]);
            let move_delta_q = group_to_target_weight[group]
                - group_cut_weight[group]
                - resolution
                    * group_weight[group]
                    * (ws.cw[target] - doc_weight + group_weight[group]);
            group_move_delta_q[group] = move_delta_q;
            group_split_delta_q[group] = split_delta_q;
            let (delta_q, action) = if move_delta_q >= split_delta_q {
                (move_delta_q, 1u8)
            } else {
                (split_delta_q, 2u8)
            };
            if delta_q > best_group_delta_q {
                best_group_delta_q = delta_q;
                best_group = group;
                best_group_action = if delta_q > 0.0 { action } else { 0 };
            }
            if delta_q > 0.0 {
                positive_group_count += 1;
                positive_group_weight += group_weight[group];
            }
            if delta_q >= -epsilon {
                near_neutral_group_count += 1;
                near_neutral_group_weight += group_weight[group];
            }
        }

        if !best_group_delta_q.is_finite() {
            best_group_delta_q = 0.0;
            best_group_action = 0;
        }

        let assigned_fraction = if doc_weight > 0.0 {
            assigned_weight / doc_weight
        } else {
            0.0
        };
        let group_fraction = |group: usize| -> f64 {
            if group == usize::MAX || doc_weight <= 0.0 {
                0.0
            } else {
                group_weight[group] / doc_weight
            }
        };
        let group_target_i64 = |group: usize| -> i64 {
            if group == usize::MAX {
                -1
            } else {
                group_target[group] as i64
            }
        };
        let group_count_or_zero = |group: usize| -> u64 {
            if group == usize::MAX {
                0
            } else {
                group_count[group]
            }
        };
        let group_value = |values: &[f64], group: usize| -> f64 {
            if group == usize::MAX {
                0.0
            } else {
                values[group]
            }
        };

        out.push(ExternalGrainProbe {
            cluster: cluster_u64,
            block_count: nodes.len() as u64,
            doc_weight,
            incident_directed_edges,
            source_directed_edges,
            external_directed_edges,
            n_external_groups: group_target.len() as u64,
            assigned_count,
            assigned_weight,
            assigned_fraction,
            largest_group_target: group_target_i64(largest_group),
            largest_group_count: group_count_or_zero(largest_group),
            largest_group_weight: group_value(&group_weight, largest_group),
            largest_group_fraction: group_fraction(largest_group),
            largest_group_to_target_weight: group_value(&group_to_target_weight, largest_group),
            largest_group_cut_weight: group_value(&group_cut_weight, largest_group),
            largest_group_move_delta_q: group_value(&group_move_delta_q, largest_group),
            largest_group_split_delta_q: group_value(&group_split_delta_q, largest_group),
            second_group_target: group_target_i64(second_group),
            second_group_weight: group_value(&group_weight, second_group),
            second_group_fraction: group_fraction(second_group),
            best_group_target: group_target_i64(best_group),
            best_group_count: group_count_or_zero(best_group),
            best_group_weight: group_value(&group_weight, best_group),
            best_group_fraction: group_fraction(best_group),
            best_group_to_target_weight: group_value(&group_to_target_weight, best_group),
            best_group_cut_weight: group_value(&group_cut_weight, best_group),
            best_group_move_delta_q: group_value(&group_move_delta_q, best_group),
            best_group_split_delta_q: group_value(&group_split_delta_q, best_group),
            best_group_delta_q,
            best_group_action,
            positive_group_count,
            positive_group_weight,
            near_neutral_group_count,
            near_neutral_group_weight,
        });
    }

    out
}

#[inline]
pub fn external_grain_probe_priority(probe: &ExternalGrainProbe) -> f64 {
    probe.best_group_delta_q / (probe.incident_directed_edges.max(1) as f64)
}

pub fn select_external_grain_probes(
    probes: &[ExternalGrainProbe],
    policy: ExternalGrainSelectionPolicy,
) -> Vec<ExternalGrainSelection> {
    probes
        .iter()
        .map(|probe| {
            let within_budget = policy.max_incident_directed_edges == 0
                || probe.incident_directed_edges <= policy.max_incident_directed_edges;
            ExternalGrainSelection {
                recommended_for_split_repair: probe.doc_weight >= policy.min_doc_weight
                    && within_budget
                    && probe.best_group_delta_q >= policy.min_best_delta_q
                    && probe.assigned_fraction >= policy.min_assigned_fraction
                    && probe.best_group_fraction >= policy.min_best_group_fraction,
                priority: external_grain_probe_priority(probe),
            }
        })
        .collect()
}

struct SplitEval {
    n_parts: u64,
    non_singleton_parts: u64,
    singleton_parts: u64,
    singleton_weight: f64,
    core_part_count: u64,
    core_part_weight: f64,
    largest_part_weight: f64,
    second_part_weight: f64,
    largest_part_fraction: f64,
    induced_directed_edges: u64,
    cut_weight: f64,
    split_delta_q_base: f64,
    split_delta_q_probe: f64,
}

struct RepairEval {
    repair_quotient_edges: u64,
    repair_merge_count: u64,
    repair_delta_q: f64,
    net_delta_q: f64,
    final_source_units: u64,
    retained_source_units: u64,
    escaped_source_units: u64,
    escaped_source_weight: f64,
    final_small_source_units: u64,
    final_small_source_weight: f64,
    largest_source_unit_fraction: f64,
    restored_source_cluster: bool,
}

struct RepairPlan {
    eval: RepairEval,
    source_part_target_cluster: Vec<u32>,
    new_retained_clusters: u64,
}

fn add_local_edge(adj: &mut [HashMap<usize, f64>], u: usize, v: usize, weight: f64) {
    if u == v || weight == 0.0 {
        return;
    }
    *adj[u].entry(v).or_insert(0.0) += weight;
    *adj[v].entry(u).or_insert(0.0) += weight;
}

fn evaluate_split_partition(
    graph: &Graph,
    nodes: &[u32],
    assignments: &[u32],
    n_parts: usize,
    base_resolution: f64,
    probe_resolution: f64,
    min_core_weight: f64,
    local_index: &mut [u32],
) -> SplitEval {
    let mut part_counts = vec![0u64; n_parts];
    let mut part_weights = vec![0.0f64; n_parts];
    for (local, &node) in nodes.iter().enumerate() {
        local_index[node as usize] = local as u32;
        let part = assignments[local] as usize;
        part_counts[part] += 1;
        part_weights[part] += graph.node_weights[node as usize];
    }

    let mut directed_cut_weight = 0.0;
    let mut induced_directed_edges = 0u64;
    for (local, &node) in nodes.iter().enumerate() {
        let part = assignments[local] as usize;
        let start = graph.first_neighbor_index[node as usize] as usize;
        let end = graph.first_neighbor_index[node as usize + 1] as usize;
        for edge_idx in start..end {
            let nbr = graph.neighbors[edge_idx] as usize;
            let local_nbr = local_index[nbr];
            if local_nbr != u32::MAX {
                induced_directed_edges += 1;
                if assignments[local_nbr as usize] as usize != part {
                    directed_cut_weight += graph.edge_weights[edge_idx];
                }
            }
        }
    }

    for &node in nodes {
        local_index[node as usize] = u32::MAX;
    }

    let doc_weight: f64 = part_weights.iter().sum();
    let sum_square_weight: f64 = part_weights.iter().map(|weight| weight * weight).sum();
    let pair_weight = (doc_weight * doc_weight - sum_square_weight) / 2.0;
    let cut_weight = directed_cut_weight / 2.0;
    let split_delta_q_base = -cut_weight + base_resolution * pair_weight;
    let split_delta_q_probe = -cut_weight + probe_resolution * pair_weight;

    let mut singleton_parts = 0u64;
    let mut singleton_weight = 0.0;
    let mut non_singleton_parts = 0u64;
    let mut core_part_count = 0u64;
    let mut core_part_weight = 0.0;
    let mut largest_part_weight = 0.0;
    let mut second_part_weight = 0.0;
    for (&count, &weight) in part_counts.iter().zip(part_weights.iter()) {
        if count == 1 {
            singleton_parts += 1;
            singleton_weight += weight;
        } else if count > 1 {
            non_singleton_parts += 1;
        }
        if weight >= min_core_weight {
            core_part_count += 1;
            core_part_weight += weight;
        }
        if weight > largest_part_weight {
            second_part_weight = largest_part_weight;
            largest_part_weight = weight;
        } else if weight > second_part_weight {
            second_part_weight = weight;
        }
    }
    let largest_part_fraction = if doc_weight > 0.0 {
        largest_part_weight / doc_weight
    } else {
        0.0
    };

    SplitEval {
        n_parts: n_parts as u64,
        non_singleton_parts,
        singleton_parts,
        singleton_weight,
        core_part_count,
        core_part_weight,
        largest_part_weight,
        second_part_weight,
        largest_part_fraction,
        induced_directed_edges,
        cut_weight,
        split_delta_q_base,
        split_delta_q_probe,
    }
}

fn repair_root(parent: &[usize], mut unit: usize) -> usize {
    while parent[unit] != unit {
        unit = parent[unit];
    }
    unit
}

#[allow(clippy::too_many_arguments)]
fn repair_split_partition(
    graph: &Graph,
    clustering: &Clustering,
    nodes: &[u32],
    assignments: &[u32],
    n_parts: usize,
    source_cluster: usize,
    source_doc_weight: f64,
    split_delta_q_base: f64,
    resolution: f64,
    min_core_weight: f64,
    repair_epsilon: f64,
    ws: &mut Workspace,
) -> RepairEval {
    let mut unit_weight = vec![0.0f64; n_parts];
    let mut source_weight = vec![0.0f64; n_parts];
    let mut source_part_count = vec![1u64; n_parts];
    let mut has_external = vec![false; n_parts];
    let mut adj: Vec<HashMap<usize, f64>> = (0..n_parts).map(|_| HashMap::new()).collect();

    for (local, &node) in nodes.iter().enumerate() {
        ws.local_index[node as usize] = local as u32;
        let part = assignments[local] as usize;
        let weight = graph.node_weights[node as usize];
        unit_weight[part] += weight;
        source_weight[part] += weight;
    }

    ws.temp_used.clear();
    for (local, &node_u32) in nodes.iter().enumerate() {
        let node = node_u32 as usize;
        let part = assignments[local] as usize;
        let nbr_start = graph.first_neighbor_index[node] as usize;
        let nbr_end = graph.first_neighbor_index[node + 1] as usize;
        for edge_idx in nbr_start..nbr_end {
            let nbr = graph.neighbors[edge_idx] as usize;
            let nbr_cluster = clustering.clusters[nbr] as usize;
            let weight = graph.edge_weights[edge_idx];
            if nbr_cluster == source_cluster {
                let local_nbr = ws.local_index[nbr];
                if local_nbr != u32::MAX && local < local_nbr as usize {
                    let nbr_part = assignments[local_nbr as usize] as usize;
                    add_local_edge(&mut adj, part, nbr_part, weight);
                }
                continue;
            }

            let unit = if ws.temp_seen[nbr_cluster] == u32::MAX {
                let unit = unit_weight.len();
                ws.temp_seen[nbr_cluster] = unit as u32;
                ws.temp_used.push(nbr_cluster as u32);
                unit_weight.push(ws.cw[nbr_cluster]);
                source_weight.push(0.0);
                source_part_count.push(0);
                has_external.push(true);
                adj.push(HashMap::new());
                unit
            } else {
                ws.temp_seen[nbr_cluster] as usize
            };
            add_local_edge(&mut adj, part, unit, weight);
        }
    }

    for &node in nodes {
        ws.local_index[node as usize] = u32::MAX;
    }
    for nbr_cluster_u32 in ws.temp_used.drain(..) {
        ws.temp_seen[nbr_cluster_u32 as usize] = u32::MAX;
    }

    let repair_quotient_edges = adj
        .iter()
        .enumerate()
        .map(|(u, neighbors)| neighbors.keys().filter(|&&v| u < v).count() as u64)
        .sum();
    let mut active = vec![true; unit_weight.len()];
    let mut repair_merge_count = 0u64;
    let mut repair_delta_q = 0.0;

    loop {
        let mut best_delta_q = f64::NEG_INFINITY;
        let mut best_pair = None;
        for u in 0..adj.len() {
            if !active[u] {
                continue;
            }
            for (&v, &edge_weight) in &adj[u] {
                if u >= v || !active[v] {
                    continue;
                }
                if source_weight[u] == 0.0 && source_weight[v] == 0.0 {
                    continue;
                }
                if has_external[u] && has_external[v] {
                    continue;
                }
                let delta_q = edge_weight - resolution * unit_weight[u] * unit_weight[v];
                if delta_q > best_delta_q {
                    best_delta_q = delta_q;
                    best_pair = Some((u, v));
                }
            }
        }

        let should_merge = if repair_epsilon > 0.0 {
            best_delta_q >= -repair_epsilon
        } else {
            best_delta_q > 0.0
        };
        if !should_merge {
            break;
        }
        let Some((u, v)) = best_pair else {
            break;
        };

        let (keep, remove) = if has_external[u] && !has_external[v] {
            (u, v)
        } else if has_external[v] && !has_external[u] {
            (v, u)
        } else if unit_weight[u] >= unit_weight[v] {
            (u, v)
        } else {
            (v, u)
        };

        repair_merge_count += 1;
        repair_delta_q += best_delta_q;
        unit_weight[keep] += unit_weight[remove];
        source_weight[keep] += source_weight[remove];
        source_part_count[keep] += source_part_count[remove];
        has_external[keep] |= has_external[remove];
        active[remove] = false;

        adj[keep].remove(&remove);
        let removed_neighbors: Vec<(usize, f64)> = adj[remove].drain().collect();
        for (nbr, weight) in removed_neighbors {
            if nbr == keep || !active[nbr] {
                continue;
            }
            adj[nbr].remove(&remove);
            add_local_edge(&mut adj, keep, nbr, weight);
        }
    }

    let mut final_source_units = 0u64;
    let mut retained_source_units = 0u64;
    let mut escaped_source_units = 0u64;
    let mut escaped_source_weight = 0.0;
    let mut final_small_source_units = 0u64;
    let mut final_small_source_weight = 0.0;
    let mut largest_source_unit_weight = 0.0;
    for u in 0..active.len() {
        if !active[u] || source_weight[u] == 0.0 {
            continue;
        }
        final_source_units += 1;
        if has_external[u] {
            escaped_source_units += 1;
            escaped_source_weight += source_weight[u];
        } else {
            retained_source_units += 1;
        }
        if source_weight[u] < min_core_weight {
            final_small_source_units += 1;
            final_small_source_weight += source_weight[u];
        }
        if source_weight[u] > largest_source_unit_weight {
            largest_source_unit_weight = source_weight[u];
        }
    }

    RepairEval {
        repair_quotient_edges,
        repair_merge_count,
        repair_delta_q,
        net_delta_q: split_delta_q_base + repair_delta_q,
        final_source_units,
        retained_source_units,
        escaped_source_units,
        escaped_source_weight,
        final_small_source_units,
        final_small_source_weight,
        largest_source_unit_fraction: if source_doc_weight > 0.0 {
            largest_source_unit_weight / source_doc_weight
        } else {
            0.0
        },
        restored_source_cluster: final_source_units == 1 && escaped_source_units == 0,
    }
}

#[allow(clippy::too_many_arguments)]
fn repair_split_partition_plan(
    graph: &Graph,
    clustering: &Clustering,
    nodes: &[u32],
    assignments: &[u32],
    n_parts: usize,
    source_cluster: usize,
    source_doc_weight: f64,
    split_delta_q_base: f64,
    resolution: f64,
    min_core_weight: f64,
    repair_epsilon: f64,
    next_cluster: &mut u32,
    ws: &mut Workspace,
) -> RepairPlan {
    let mut unit_weight = vec![0.0f64; n_parts];
    let mut source_weight = vec![0.0f64; n_parts];
    let mut source_part_count = vec![1u64; n_parts];
    let mut has_external = vec![false; n_parts];
    let mut external_cluster = vec![None; n_parts];
    let mut parent: Vec<usize> = (0..n_parts).collect();
    let mut adj: Vec<HashMap<usize, f64>> = (0..n_parts).map(|_| HashMap::new()).collect();

    for (local, &node) in nodes.iter().enumerate() {
        ws.local_index[node as usize] = local as u32;
        let part = assignments[local] as usize;
        let weight = graph.node_weights[node as usize];
        unit_weight[part] += weight;
        source_weight[part] += weight;
    }

    ws.temp_used.clear();
    for (local, &node_u32) in nodes.iter().enumerate() {
        let node = node_u32 as usize;
        let part = assignments[local] as usize;
        let nbr_start = graph.first_neighbor_index[node] as usize;
        let nbr_end = graph.first_neighbor_index[node + 1] as usize;
        for edge_idx in nbr_start..nbr_end {
            let nbr = graph.neighbors[edge_idx] as usize;
            let nbr_cluster = clustering.clusters[nbr] as usize;
            let weight = graph.edge_weights[edge_idx];
            if nbr_cluster == source_cluster {
                let local_nbr = ws.local_index[nbr];
                if local_nbr != u32::MAX && local < local_nbr as usize {
                    let nbr_part = assignments[local_nbr as usize] as usize;
                    add_local_edge(&mut adj, part, nbr_part, weight);
                }
                continue;
            }

            let unit = if ws.temp_seen[nbr_cluster] == u32::MAX {
                let unit = unit_weight.len();
                ws.temp_seen[nbr_cluster] = unit as u32;
                ws.temp_used.push(nbr_cluster as u32);
                unit_weight.push(ws.cw[nbr_cluster]);
                source_weight.push(0.0);
                source_part_count.push(0);
                has_external.push(true);
                external_cluster.push(Some(nbr_cluster as u32));
                parent.push(unit);
                adj.push(HashMap::new());
                unit
            } else {
                ws.temp_seen[nbr_cluster] as usize
            };
            add_local_edge(&mut adj, part, unit, weight);
        }
    }

    for &node in nodes {
        ws.local_index[node as usize] = u32::MAX;
    }
    for nbr_cluster_u32 in ws.temp_used.drain(..) {
        ws.temp_seen[nbr_cluster_u32 as usize] = u32::MAX;
    }

    let repair_quotient_edges = adj
        .iter()
        .enumerate()
        .map(|(u, neighbors)| neighbors.keys().filter(|&&v| u < v).count() as u64)
        .sum();
    let mut active = vec![true; unit_weight.len()];
    let mut repair_merge_count = 0u64;
    let mut repair_delta_q = 0.0;

    loop {
        let mut best_delta_q = f64::NEG_INFINITY;
        let mut best_pair = None;
        for u in 0..adj.len() {
            if !active[u] {
                continue;
            }
            for (&v, &edge_weight) in &adj[u] {
                if u >= v || !active[v] {
                    continue;
                }
                if source_weight[u] == 0.0 && source_weight[v] == 0.0 {
                    continue;
                }
                if has_external[u] && has_external[v] {
                    continue;
                }
                let delta_q = edge_weight - resolution * unit_weight[u] * unit_weight[v];
                if delta_q > best_delta_q {
                    best_delta_q = delta_q;
                    best_pair = Some((u, v));
                }
            }
        }

        let should_merge = if repair_epsilon > 0.0 {
            best_delta_q >= -repair_epsilon
        } else {
            best_delta_q > 0.0
        };
        if !should_merge {
            break;
        }
        let Some((u, v)) = best_pair else {
            break;
        };

        let (keep, remove) = if has_external[u] && !has_external[v] {
            (u, v)
        } else if has_external[v] && !has_external[u] {
            (v, u)
        } else if unit_weight[u] >= unit_weight[v] {
            (u, v)
        } else {
            (v, u)
        };

        repair_merge_count += 1;
        repair_delta_q += best_delta_q;
        unit_weight[keep] += unit_weight[remove];
        source_weight[keep] += source_weight[remove];
        source_part_count[keep] += source_part_count[remove];
        has_external[keep] |= has_external[remove];
        if external_cluster[keep].is_none() {
            external_cluster[keep] = external_cluster[remove];
        }
        parent[remove] = keep;
        active[remove] = false;

        adj[keep].remove(&remove);
        let removed_neighbors: Vec<(usize, f64)> = adj[remove].drain().collect();
        for (nbr, weight) in removed_neighbors {
            if nbr == keep || !active[nbr] {
                continue;
            }
            adj[nbr].remove(&remove);
            add_local_edge(&mut adj, keep, nbr, weight);
        }
    }

    let mut final_source_units = 0u64;
    let mut retained_source_units = 0u64;
    let mut escaped_source_units = 0u64;
    let mut escaped_source_weight = 0.0;
    let mut final_small_source_units = 0u64;
    let mut final_small_source_weight = 0.0;
    let mut largest_source_unit_weight = 0.0;
    let mut retained_keep_source_unit = None;
    let mut retained_keep_source_weight = f64::NEG_INFINITY;
    for u in 0..active.len() {
        if !active[u] || source_weight[u] == 0.0 {
            continue;
        }
        final_source_units += 1;
        if has_external[u] {
            escaped_source_units += 1;
            escaped_source_weight += source_weight[u];
        } else {
            retained_source_units += 1;
            if source_weight[u] > retained_keep_source_weight {
                retained_keep_source_weight = source_weight[u];
                retained_keep_source_unit = Some(u);
            }
        }
        if source_weight[u] < min_core_weight {
            final_small_source_units += 1;
            final_small_source_weight += source_weight[u];
        }
        if source_weight[u] > largest_source_unit_weight {
            largest_source_unit_weight = source_weight[u];
        }
    }

    let mut retained_targets: HashMap<usize, u32> = HashMap::new();
    let mut new_retained_clusters = 0u64;
    for u in 0..active.len() {
        if !active[u] || source_weight[u] == 0.0 || has_external[u] {
            continue;
        }
        let target = if Some(u) == retained_keep_source_unit {
            source_cluster as u32
        } else {
            let target = *next_cluster;
            *next_cluster = next_cluster
                .checked_add(1)
                .expect("cluster id exceeds u32::MAX");
            new_retained_clusters += 1;
            target
        };
        retained_targets.insert(u, target);
    }

    let mut source_part_target_cluster = vec![source_cluster as u32; n_parts];
    for (part, target) in source_part_target_cluster.iter_mut().enumerate() {
        let unit = repair_root(&parent, part);
        *target = if let Some(target_cluster) = external_cluster[unit] {
            target_cluster
        } else {
            retained_targets
                .get(&unit)
                .copied()
                .unwrap_or(source_cluster as u32)
        };
    }

    RepairPlan {
        eval: RepairEval {
            repair_quotient_edges,
            repair_merge_count,
            repair_delta_q,
            net_delta_q: split_delta_q_base + repair_delta_q,
            final_source_units,
            retained_source_units,
            escaped_source_units,
            escaped_source_weight,
            final_small_source_units,
            final_small_source_weight,
            largest_source_unit_fraction: if source_doc_weight > 0.0 {
                largest_source_unit_weight / source_doc_weight
            } else {
                0.0
            },
            restored_source_cluster: final_source_units == 1 && escaped_source_units == 0,
        },
        source_part_target_cluster,
        new_retained_clusters,
    }
}

pub fn multi_core_split_probes(
    graph: &Graph,
    clustering: &Clustering,
    candidate_clusters: &[u64],
    resolution: f64,
    gamma_multipliers: &[f64],
    min_core_weight: f64,
    randomness: f64,
    seed: u64,
    ws: &mut Workspace,
) -> Vec<MultiCoreSplitProbe> {
    assert_eq!(clustering.n_nodes, graph.n_nodes);

    let n_clusters = clustering.n_clusters;
    clustering.fill_cluster_groups_and_weights(&graph.node_weights, ws);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut merge_ws = LocalMergeWorkspace::new(0);
    let mut cluster_sizes = Vec::new();
    let mut out = Vec::with_capacity(candidate_clusters.len() * gamma_multipliers.len().max(1));

    for &cluster_u64 in candidate_clusters {
        let Ok(cluster_u32) = u32::try_from(cluster_u64) else {
            continue;
        };
        let c = cluster_u32 as usize;
        if c >= n_clusters {
            continue;
        }

        let start = ws.npc_starts[c] as usize;
        let end = ws.npc_starts[c + 1] as usize;
        let nodes = &ws.npc_nodes[start..end];
        if nodes.is_empty() {
            continue;
        }

        let doc_weight = ws.cw[c];
        let mut self_loop_weight = 0.0;
        let mut internal_directed_weight = 0.0;
        let mut induced_directed_edges = 0u64;
        for &node_u32 in nodes {
            let node = node_u32 as usize;
            self_loop_weight += graph.self_loop_weights[node];
            let nbr_start = graph.first_neighbor_index[node] as usize;
            let nbr_end = graph.first_neighbor_index[node + 1] as usize;
            for edge_idx in nbr_start..nbr_end {
                let nbr = graph.neighbors[edge_idx] as usize;
                if clustering.clusters[nbr] as usize == c {
                    internal_directed_weight += graph.edge_weights[edge_idx];
                    induced_directed_edges += 1;
                }
            }
        }
        let internal_weight = internal_directed_weight / 2.0 + self_loop_weight;

        for &multiplier in gamma_multipliers {
            if !multiplier.is_finite() || multiplier <= 0.0 {
                continue;
            }
            let probe_resolution = resolution * multiplier;
            cluster_sizes.clear();
            let n_parts = local_merge::find_clustering_induced_u32_with_workspace_assignments_and_append_sizes(
                graph,
                nodes,
                &mut ws.local_index,
                probe_resolution,
                randomness,
                &mut rng,
                &mut merge_ws,
                &mut cluster_sizes,
            );
            let assignments = &merge_ws.assignments()[..nodes.len()];
            let eval = evaluate_split_partition(
                graph,
                nodes,
                assignments,
                n_parts,
                resolution,
                probe_resolution,
                min_core_weight,
                &mut ws.local_index,
            );
            out.push(MultiCoreSplitProbe {
                cluster: cluster_u64,
                gamma_multiplier: multiplier,
                probe_resolution,
                block_count: nodes.len() as u64,
                doc_weight,
                internal_weight,
                induced_directed_edges,
                n_parts: eval.n_parts,
                non_singleton_parts: eval.non_singleton_parts,
                singleton_parts: eval.singleton_parts,
                singleton_weight: eval.singleton_weight,
                core_part_count: eval.core_part_count,
                core_part_weight: eval.core_part_weight,
                largest_part_weight: eval.largest_part_weight,
                second_part_weight: eval.second_part_weight,
                largest_part_fraction: eval.largest_part_fraction,
                cut_weight: eval.cut_weight,
                split_delta_q_base: eval.split_delta_q_base,
                split_delta_q_probe: eval.split_delta_q_probe,
                hysteresis_only: eval.split_delta_q_base <= 0.0 && eval.split_delta_q_probe > 0.0,
            });
        }
    }

    out
}

#[allow(clippy::too_many_arguments)]
pub fn split_merge_repair_probes(
    graph: &Graph,
    clustering: &Clustering,
    candidate_clusters: &[u64],
    resolution: f64,
    gamma_multipliers: &[f64],
    min_core_weight: f64,
    randomness: f64,
    repair_epsilon: f64,
    seed: u64,
    pair_seeded: bool,
    ws: &mut Workspace,
) -> Vec<SplitMergeRepairProbe> {
    assert_eq!(clustering.n_nodes, graph.n_nodes);

    let n_clusters = clustering.n_clusters;
    clustering.fill_cluster_groups_and_weights(&graph.node_weights, ws);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut merge_ws = LocalMergeWorkspace::new(0);
    let mut cluster_sizes = Vec::new();
    let mut out = Vec::with_capacity(candidate_clusters.len() * gamma_multipliers.len().max(1));

    for &cluster_u64 in candidate_clusters {
        let Ok(cluster_u32) = u32::try_from(cluster_u64) else {
            continue;
        };
        let c = cluster_u32 as usize;
        if c >= n_clusters {
            continue;
        }

        let start = ws.npc_starts[c] as usize;
        let end = ws.npc_starts[c + 1] as usize;
        let nodes: Vec<u32> = ws.npc_nodes[start..end].to_vec();
        if nodes.is_empty() {
            continue;
        }

        let doc_weight = ws.cw[c];
        for &multiplier in gamma_multipliers {
            if !multiplier.is_finite() || multiplier <= 0.0 {
                continue;
            }
            let probe_resolution = resolution * multiplier;
            cluster_sizes.clear();
            let n_parts = if pair_seeded {
                let mut pair_rng =
                    StdRng::seed_from_u64(split_repair_pair_seed(seed, cluster_u64, multiplier));
                local_merge::find_clustering_induced_u32_with_workspace_assignments_and_append_sizes(
                    graph,
                    &nodes,
                    &mut ws.local_index,
                    probe_resolution,
                    randomness,
                    &mut pair_rng,
                    &mut merge_ws,
                    &mut cluster_sizes,
                )
            } else {
                local_merge::find_clustering_induced_u32_with_workspace_assignments_and_append_sizes(
                    graph,
                    &nodes,
                    &mut ws.local_index,
                    probe_resolution,
                    randomness,
                    &mut rng,
                    &mut merge_ws,
                    &mut cluster_sizes,
                )
            };
            let assignments = &merge_ws.assignments()[..nodes.len()];
            let split_eval = evaluate_split_partition(
                graph,
                &nodes,
                assignments,
                n_parts,
                resolution,
                probe_resolution,
                min_core_weight,
                &mut ws.local_index,
            );
            let repair_eval = repair_split_partition(
                graph,
                clustering,
                &nodes,
                assignments,
                n_parts,
                c,
                doc_weight,
                split_eval.split_delta_q_base,
                resolution,
                min_core_weight,
                repair_epsilon,
                ws,
            );
            out.push(SplitMergeRepairProbe {
                cluster: cluster_u64,
                gamma_multiplier: multiplier,
                probe_resolution,
                block_count: nodes.len() as u64,
                doc_weight,
                induced_directed_edges: split_eval.induced_directed_edges,
                n_parts: split_eval.n_parts,
                core_part_count: split_eval.core_part_count,
                singleton_weight: split_eval.singleton_weight,
                cut_weight: split_eval.cut_weight,
                split_delta_q_base: split_eval.split_delta_q_base,
                split_delta_q_probe: split_eval.split_delta_q_probe,
                repair_quotient_edges: repair_eval.repair_quotient_edges,
                repair_merge_count: repair_eval.repair_merge_count,
                repair_delta_q: repair_eval.repair_delta_q,
                net_delta_q: repair_eval.net_delta_q,
                final_source_units: repair_eval.final_source_units,
                retained_source_units: repair_eval.retained_source_units,
                escaped_source_units: repair_eval.escaped_source_units,
                escaped_source_weight: repair_eval.escaped_source_weight,
                final_small_source_units: repair_eval.final_small_source_units,
                final_small_source_weight: repair_eval.final_small_source_weight,
                largest_source_unit_fraction: repair_eval.largest_source_unit_fraction,
                restored_source_cluster: repair_eval.restored_source_cluster,
            });
        }
    }

    out
}

fn same_gamma_multiplier(left: f64, right: f64) -> bool {
    (left - right).abs() <= 1e-12 * left.abs().max(right.abs()).max(1.0)
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}

fn split_repair_pair_seed(seed: u64, cluster: u64, gamma_multiplier: f64) -> u64 {
    splitmix64(
        seed ^ splitmix64(cluster) ^ splitmix64(gamma_multiplier.to_bits()) ^ 0x6a09e667f3bcc909,
    )
}

fn selected_split_repair_pair_index(
    cluster: u64,
    gamma_multiplier: f64,
    selected_clusters: &[u64],
    selected_gamma_multipliers: &[f64],
    applied_selected: &[bool],
) -> Option<usize> {
    selected_clusters
        .iter()
        .zip(selected_gamma_multipliers.iter())
        .zip(applied_selected.iter())
        .enumerate()
        .find_map(|(idx, ((&selected_cluster, &selected_gamma), &applied))| {
            if !applied
                && selected_cluster == cluster
                && same_gamma_multiplier(selected_gamma, gamma_multiplier)
            {
                Some(idx)
            } else {
                None
            }
        })
}

#[allow(clippy::too_many_arguments)]
pub fn apply_split_merge_repair_candidates(
    graph: &Graph,
    clustering: &Clustering,
    candidate_clusters: &[u64],
    selected_clusters: &[u64],
    selected_gamma_multipliers: &[f64],
    resolution: f64,
    gamma_multipliers: &[f64],
    min_core_weight: f64,
    randomness: f64,
    repair_epsilon: f64,
    seed: u64,
    pair_seeded: bool,
    ws: &mut Workspace,
) -> (Vec<u64>, Vec<SplitRepairAppliedCandidate>) {
    assert_eq!(clustering.n_nodes, graph.n_nodes);
    assert_eq!(selected_clusters.len(), selected_gamma_multipliers.len());

    let n_clusters = clustering.n_clusters;
    clustering.fill_cluster_groups_and_weights(&graph.node_weights, ws);
    let mut next_cluster = u32::try_from(n_clusters).expect("cluster id exceeds u32::MAX");
    let mut proposed_membership = clustering.clusters.clone();
    let mut applied_selected = vec![false; selected_clusters.len()];
    let mut rng = StdRng::seed_from_u64(seed);
    let mut merge_ws = LocalMergeWorkspace::new(0);
    let mut cluster_sizes = Vec::new();
    let mut applied = Vec::new();
    let selected_total = selected_clusters.len();
    let mut applied_count = 0usize;

    for &cluster_u64 in candidate_clusters {
        if applied_count == selected_total {
            break;
        }
        let Ok(cluster_u32) = u32::try_from(cluster_u64) else {
            continue;
        };
        let c = cluster_u32 as usize;
        if c >= n_clusters {
            continue;
        }
        if pair_seeded
            && !selected_clusters
                .iter()
                .zip(applied_selected.iter())
                .any(|(&selected_cluster, &applied)| !applied && selected_cluster == cluster_u64)
        {
            continue;
        }

        let start = ws.npc_starts[c] as usize;
        let end = ws.npc_starts[c + 1] as usize;
        let nodes: Vec<u32> = ws.npc_nodes[start..end].to_vec();
        if nodes.is_empty() {
            continue;
        }

        let doc_weight = ws.cw[c];
        for &multiplier in gamma_multipliers {
            if applied_count == selected_total {
                break;
            }
            if !multiplier.is_finite() || multiplier <= 0.0 {
                continue;
            }
            let selected_index_before_clustering = if pair_seeded {
                selected_split_repair_pair_index(
                    cluster_u64,
                    multiplier,
                    selected_clusters,
                    selected_gamma_multipliers,
                    &applied_selected,
                )
            } else {
                None
            };
            if pair_seeded && selected_index_before_clustering.is_none() {
                continue;
            }
            let probe_resolution = resolution * multiplier;
            cluster_sizes.clear();
            let n_parts = if pair_seeded {
                let mut pair_rng =
                    StdRng::seed_from_u64(split_repair_pair_seed(seed, cluster_u64, multiplier));
                local_merge::find_clustering_induced_u32_with_workspace_assignments_and_append_sizes(
                    graph,
                    &nodes,
                    &mut ws.local_index,
                    probe_resolution,
                    randomness,
                    &mut pair_rng,
                    &mut merge_ws,
                    &mut cluster_sizes,
                )
            } else {
                local_merge::find_clustering_induced_u32_with_workspace_assignments_and_append_sizes(
                    graph,
                    &nodes,
                    &mut ws.local_index,
                    probe_resolution,
                    randomness,
                    &mut rng,
                    &mut merge_ws,
                    &mut cluster_sizes,
                )
            };
            let assignments = &merge_ws.assignments()[..nodes.len()];
            let selected_index = if let Some(selected_index) = selected_index_before_clustering {
                selected_index
            } else {
                let Some(selected_index) = selected_split_repair_pair_index(
                    cluster_u64,
                    multiplier,
                    selected_clusters,
                    selected_gamma_multipliers,
                    &applied_selected,
                ) else {
                    continue;
                };
                selected_index
            };

            let split_eval = evaluate_split_partition(
                graph,
                &nodes,
                assignments,
                n_parts,
                resolution,
                probe_resolution,
                min_core_weight,
                &mut ws.local_index,
            );
            let repair_plan = repair_split_partition_plan(
                graph,
                clustering,
                &nodes,
                assignments,
                n_parts,
                c,
                doc_weight,
                split_eval.split_delta_q_base,
                resolution,
                min_core_weight,
                repair_epsilon,
                &mut next_cluster,
                ws,
            );

            let mut changed_nodes = 0u64;
            let mut moved_to_existing_cluster_nodes = 0u64;
            let mut moved_to_new_cluster_nodes = 0u64;
            for (local, &node_u32) in nodes.iter().enumerate() {
                let node = node_u32 as usize;
                let part = assignments[local] as usize;
                let target = repair_plan.source_part_target_cluster[part];
                if target != clustering.clusters[node] {
                    changed_nodes += 1;
                    if target as usize >= n_clusters {
                        moved_to_new_cluster_nodes += 1;
                    } else {
                        moved_to_existing_cluster_nodes += 1;
                    }
                }
                proposed_membership[node] = target;
            }

            applied_selected[selected_index] = true;
            applied_count += 1;
            applied.push(SplitRepairAppliedCandidate {
                selected_index: selected_index as u64,
                cluster: cluster_u64,
                gamma_multiplier: multiplier,
                probe_resolution,
                block_count: nodes.len() as u64,
                doc_weight,
                n_parts: split_eval.n_parts,
                split_delta_q_base: split_eval.split_delta_q_base,
                repair_delta_q: repair_plan.eval.repair_delta_q,
                predicted_net_delta_q: repair_plan.eval.net_delta_q,
                repair_merge_count: repair_plan.eval.repair_merge_count,
                final_source_units: repair_plan.eval.final_source_units,
                retained_source_units: repair_plan.eval.retained_source_units,
                escaped_source_units: repair_plan.eval.escaped_source_units,
                escaped_source_weight: repair_plan.eval.escaped_source_weight,
                final_small_source_units: repair_plan.eval.final_small_source_units,
                final_small_source_weight: repair_plan.eval.final_small_source_weight,
                largest_source_unit_fraction: repair_plan.eval.largest_source_unit_fraction,
                changed_nodes,
                moved_to_existing_cluster_nodes,
                moved_to_new_cluster_nodes,
                new_retained_clusters: repair_plan.new_retained_clusters,
            });
        }
    }

    (
        proposed_membership.into_iter().map(u64::from).collect(),
        applied,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quality::{QualityFunction, CPM};

    fn external_probe_for_selection() -> ExternalGrainProbe {
        ExternalGrainProbe {
            cluster: 10,
            block_count: 5,
            doc_weight: 10.0,
            incident_directed_edges: 4,
            source_directed_edges: 1,
            external_directed_edges: 3,
            n_external_groups: 2,
            assigned_count: 4,
            assigned_weight: 8.0,
            assigned_fraction: 0.8,
            largest_group_target: 1,
            largest_group_count: 2,
            largest_group_weight: 4.0,
            largest_group_fraction: 0.4,
            largest_group_to_target_weight: 3.0,
            largest_group_cut_weight: 1.0,
            largest_group_move_delta_q: 2.0,
            largest_group_split_delta_q: 1.0,
            second_group_target: 2,
            second_group_weight: 3.0,
            second_group_fraction: 0.3,
            best_group_target: 1,
            best_group_count: 2,
            best_group_weight: 4.0,
            best_group_fraction: 0.4,
            best_group_to_target_weight: 3.0,
            best_group_cut_weight: 1.0,
            best_group_move_delta_q: 2.0,
            best_group_split_delta_q: 1.0,
            best_group_delta_q: 2.0,
            best_group_action: 1,
            positive_group_count: 1,
            positive_group_weight: 4.0,
            near_neutral_group_count: 1,
            near_neutral_group_weight: 4.0,
        }
    }

    #[test]
    fn cluster_graph_stats_reports_macro_merge_delta() {
        let graph = Graph::from_edge_list(4, &[0, 0, 1, 2], &[1, 2, 3, 3], &[2.0, 0.5, 0.5, 3.0]);
        let clustering = Clustering::from_assignments(vec![0, 0, 1, 1]);
        let mut ws = Workspace::new(4);

        let stats = cluster_graph_stats(&graph, &clustering, 0.1, 3.0, 10.0, 4, &mut ws);

        assert_eq!(stats.block_count, vec![2, 2]);
        assert_eq!(stats.doc_weight, vec![2.0, 2.0]);
        assert_eq!(stats.internal_weight, vec![2.0, 3.0]);
        assert_eq!(stats.external_weight, vec![1.0, 1.0]);
        assert_eq!(stats.degree, vec![1, 1]);
        assert_eq!(stats.top_neighbor, vec![1, 0]);
        assert_eq!(stats.second_neighbor, vec![-1, -1]);
        assert_eq!(stats.neighbor_weight_ratio, vec![0.0, 0.0]);
        assert_eq!(stats.merge_candidates.len(), 1);
        assert_eq!(stats.merge_candidates[0].source, 0);
        assert_eq!(stats.merge_candidates[0].target, 1);
        assert!((stats.merge_candidates[0].delta_q - 0.6).abs() < 1e-12);
        assert!((stats.merge_candidates[0].size_band_gain - 2.0).abs() < 1e-12);
    }

    #[test]
    fn cluster_graph_stats_reports_second_neighbor() {
        let graph = Graph::from_edge_list(3, &[0, 0, 1], &[1, 2, 2], &[4.0, 2.0, 1.0]);
        let clustering = Clustering::from_assignments(vec![0, 1, 2]);
        let mut ws = Workspace::new(3);

        let stats = cluster_graph_stats(&graph, &clustering, 0.1, 0.0, 0.0, 4, &mut ws);

        assert_eq!(stats.top_neighbor[0], 1);
        assert_eq!(stats.second_neighbor[0], 2);
        assert!((stats.top_neighbor_weight[0] - 4.0).abs() < 1e-12);
        assert!((stats.second_neighbor_weight[0] - 2.0).abs() < 1e-12);
        assert!((stats.neighbor_weight_ratio[0] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn boundary_move_probes_reports_positive_node_moves() {
        let graph = Graph::from_edge_list(4, &[0, 1, 0, 1], &[1, 2, 2, 3], &[0.1, 5.0, 1.0, 4.0]);
        let clustering = Clustering::from_assignments(vec![0, 0, 1, 2]);
        let mut ws = Workspace::new(4);

        let probes = boundary_move_probes(&graph, &clustering, &[0], 0.1, 0.0, &mut ws);

        assert_eq!(probes.len(), 1);
        let probe = &probes[0];
        assert_eq!(probe.cluster, 0);
        assert_eq!(probe.top_neighbor, 1);
        assert_eq!(probe.second_neighbor, 2);
        assert_eq!(probe.positive_move_count, 2);
        assert_eq!(probe.positive_move_weight, 2.0);
        assert_eq!(probe.best_move_node, 1);
        assert_eq!(probe.best_move_target, 1);
        assert!(probe.best_move_delta_q > 4.0);
    }

    #[test]
    fn trim_oversize_boundary_moves_moves_best_positive_boundary_node() {
        let graph = Graph::from_edge_list(4, &[0, 1, 2], &[1, 2, 3], &[3.0, 0.1, 4.0]);
        let clustering = Clustering::from_assignments(vec![0, 0, 0, 1]);
        let mut ws = Workspace::new(4);

        let (membership, moves) =
            trim_oversize_boundary_moves(&graph, &clustering, &[0], 0.1, 2.0, 0.0, 0, &mut ws);

        assert_eq!(membership, vec![0, 0, 1, 1]);
        assert_eq!(moves.len(), 1);
        let mv = &moves[0];
        assert_eq!(mv.source, 0);
        assert_eq!(mv.target, 1);
        assert_eq!(mv.node, 2);
        assert_eq!(mv.node_weight, 1.0);
        assert!((mv.delta_q - 4.0).abs() < 1e-12);
        assert_eq!(mv.source_weight_before, 3.0);
        assert_eq!(mv.source_weight_after, 2.0);
        assert_eq!(mv.target_weight_before, 1.0);
        assert_eq!(mv.target_weight_after, 2.0);

        let cpm = CPM::new(0.1);
        let proposed = Clustering::from_u64_assignments(&membership).unwrap();
        assert!(
            cpm.quality(&graph, &proposed) - cpm.quality(&graph, &clustering) >= mv.delta_q - 1e-12
        );
    }

    #[test]
    fn boundary_group_probes_can_find_positive_group_split() {
        let graph = Graph::from_edge_list(4, &[0, 0, 1, 1], &[1, 2, 2, 3], &[0.1, 10.0, 0.1, 9.0]);
        let clustering = Clustering::from_assignments(vec![0, 0, 1, 2]);
        let mut ws = Workspace::new(4);

        let probes = boundary_group_probes(&graph, &clustering, &[0], 0.1, &mut ws);

        assert_eq!(probes.len(), 1);
        let probe = &probes[0];
        assert_eq!(probe.top_neighbor, 1);
        assert_eq!(probe.second_neighbor, 2);
        assert_eq!(probe.top_group_count, 1);
        assert_eq!(probe.second_group_count, 1);
        assert!(probe.top_group_move_delta_q > 9.0);
        assert!(probe.second_group_move_delta_q > 8.0);
        assert!(probe.best_delta_q > 9.0);
    }

    #[test]
    fn external_grain_probes_group_by_strongest_external_target() {
        let graph = Graph::from_edge_list(4, &[0, 0, 1], &[1, 2, 3], &[0.1, 10.0, 9.0]);
        let clustering = Clustering::from_assignments(vec![0, 0, 1, 2]);
        let mut ws = Workspace::new(4);

        let probes = external_grain_probes(&graph, &clustering, &[0], 0.1, 0.0, &mut ws);

        assert_eq!(probes.len(), 1);
        let probe = &probes[0];
        assert_eq!(probe.cluster, 0);
        assert_eq!(probe.block_count, 2);
        assert_eq!(probe.n_external_groups, 2);
        assert_eq!(probe.assigned_count, 2);
        assert_eq!(probe.assigned_weight, 2.0);
        assert_eq!(probe.positive_group_count, 2);
        assert_eq!(probe.largest_group_count, 1);
        assert!(probe.best_group_delta_q > 8.0);
        assert_eq!(probe.best_group_action, 1);
    }

    #[test]
    fn external_grain_selection_default_matches_cli_thresholds() {
        let positive = external_probe_for_selection();
        let mut zero = external_probe_for_selection();
        zero.best_group_delta_q = 0.0;
        let mut negative = external_probe_for_selection();
        negative.best_group_delta_q = -0.1;

        let probes = vec![positive, zero, negative];
        let selected =
            select_external_grain_probes(&probes, ExternalGrainSelectionPolicy::default());

        assert!(selected[0].recommended_for_split_repair);
        assert!(selected[1].recommended_for_split_repair);
        assert!(!selected[2].recommended_for_split_repair);
    }

    #[test]
    fn external_grain_selection_applies_each_threshold() {
        let probe = external_probe_for_selection();
        assert!(select_external_grain_probes(
            &[probe.clone()],
            ExternalGrainSelectionPolicy::default(),
        )[0]
            .recommended_for_split_repair);

        let rejecting_policies = [
            ExternalGrainSelectionPolicy {
                min_doc_weight: 10.1,
                ..ExternalGrainSelectionPolicy::default()
            },
            ExternalGrainSelectionPolicy {
                max_incident_directed_edges: 3,
                ..ExternalGrainSelectionPolicy::default()
            },
            ExternalGrainSelectionPolicy {
                min_best_delta_q: 2.1,
                ..ExternalGrainSelectionPolicy::default()
            },
            ExternalGrainSelectionPolicy {
                min_assigned_fraction: 0.81,
                ..ExternalGrainSelectionPolicy::default()
            },
            ExternalGrainSelectionPolicy {
                min_best_group_fraction: 0.41,
                ..ExternalGrainSelectionPolicy::default()
            },
        ];

        for policy in rejecting_policies {
            assert!(
                !select_external_grain_probes(&[probe.clone()], policy)[0]
                    .recommended_for_split_repair
            );
        }
    }

    #[test]
    fn external_grain_selection_priority_handles_zero_incident_edges() {
        let mut probe = external_probe_for_selection();
        probe.incident_directed_edges = 0;
        probe.best_group_delta_q = 2.5;

        let selected =
            select_external_grain_probes(&[probe], ExternalGrainSelectionPolicy::default());

        assert!(selected[0].priority.is_finite());
        assert_eq!(selected[0].priority, 2.5);
    }

    #[test]
    fn multi_core_split_probes_reports_many_core_split() {
        let graph = Graph::from_edge_list(4, &[0, 2, 1], &[1, 3, 2], &[10.0, 10.0, 0.1]);
        let clustering = Clustering::from_assignments(vec![0, 0, 0, 0]);
        let mut ws = Workspace::new(4);

        let probes = multi_core_split_probes(
            &graph,
            &clustering,
            &[0],
            0.1,
            &[10.0],
            2.0,
            0.0,
            42,
            &mut ws,
        );

        assert_eq!(probes.len(), 1);
        let probe = &probes[0];
        assert_eq!(probe.cluster, 0);
        assert_eq!(probe.n_parts, 2);
        assert_eq!(probe.non_singleton_parts, 2);
        assert_eq!(probe.singleton_parts, 0);
        assert_eq!(probe.core_part_count, 2);
        assert!((probe.cut_weight - 0.1).abs() < 1e-12);
        assert!(probe.split_delta_q_base > 0.0);
        assert!(probe.split_delta_q_probe > probe.split_delta_q_base);
    }

    #[test]
    fn split_merge_repair_probes_can_escape_to_external_cluster() {
        let graph = Graph::from_edge_list(3, &[0, 1], &[1, 2], &[0.1, 10.0]);
        let clustering = Clustering::from_assignments(vec![0, 0, 1]);
        let mut ws = Workspace::new(3);

        let probes = split_merge_repair_probes(
            &graph,
            &clustering,
            &[0],
            0.1,
            &[10.0],
            1.0,
            0.0,
            0.0,
            42,
            false,
            &mut ws,
        );

        assert_eq!(probes.len(), 1);
        let probe = &probes[0];
        assert_eq!(probe.n_parts, 2);
        assert_eq!(probe.repair_merge_count, 1);
        assert!(probe.net_delta_q > 9.0);
        assert_eq!(probe.final_source_units, 2);
        assert_eq!(probe.escaped_source_units, 1);
        assert_eq!(probe.escaped_source_weight, 1.0);
        assert!(!probe.restored_source_cluster);
    }

    #[test]
    fn apply_split_merge_repair_candidates_moves_escaped_fragment() {
        let graph = Graph::from_edge_list(3, &[0, 1], &[1, 2], &[0.1, 10.0]);
        let clustering = Clustering::from_assignments(vec![0, 0, 1]);
        let mut ws = Workspace::new(3);

        let (membership, applied) = apply_split_merge_repair_candidates(
            &graph,
            &clustering,
            &[0],
            &[0],
            &[10.0],
            0.1,
            &[10.0],
            1.0,
            0.0,
            0.0,
            42,
            false,
            &mut ws,
        );

        assert_eq!(membership, vec![0, 1, 1]);
        assert_eq!(applied.len(), 1);
        assert_eq!(applied[0].changed_nodes, 1);
        assert_eq!(applied[0].moved_to_existing_cluster_nodes, 1);
        assert_eq!(applied[0].moved_to_new_cluster_nodes, 0);
        assert!(applied[0].predicted_net_delta_q > 9.0);
    }

    #[test]
    fn pair_seeded_apply_matches_probe_for_selected_late_gamma() {
        let graph = Graph::from_edge_list(
            5,
            &[0, 1, 1, 2, 3],
            &[1, 2, 3, 3, 4],
            &[0.1, 10.0, 0.5, 0.2, 8.0],
        );
        let clustering = Clustering::from_assignments(vec![0, 0, 0, 1, 1]);
        let gamma_multipliers = [1.05, 10.0];
        let mut probe_ws = Workspace::new(5);

        let probes = split_merge_repair_probes(
            &graph,
            &clustering,
            &[0],
            0.1,
            &gamma_multipliers,
            1.0,
            0.01,
            0.0,
            42,
            true,
            &mut probe_ws,
        );
        let selected = probes
            .iter()
            .find(|probe| same_gamma_multiplier(probe.gamma_multiplier, 10.0))
            .expect("selected gamma probe should exist");
        let mut apply_ws = Workspace::new(5);

        let (_membership, applied) = apply_split_merge_repair_candidates(
            &graph,
            &clustering,
            &[0],
            &[0],
            &[10.0],
            0.1,
            &gamma_multipliers,
            1.0,
            0.01,
            0.0,
            42,
            true,
            &mut apply_ws,
        );

        assert_eq!(applied.len(), 1);
        assert_eq!(applied[0].n_parts, selected.n_parts);
        assert!((applied[0].split_delta_q_base - selected.split_delta_q_base).abs() < 1e-12);
        assert!((applied[0].repair_delta_q - selected.repair_delta_q).abs() < 1e-12);
        assert!((applied[0].predicted_net_delta_q - selected.net_delta_q).abs() < 1e-12);
    }
}
