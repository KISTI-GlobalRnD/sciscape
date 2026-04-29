//! Experimental cluster-graph diagnostics for SciSci adaptive refinement.
//!
//! These helpers do not change Leiden behavior. They build a contracted
//! cluster graph from an existing membership and report the cheap statistics
//! needed to decide whether macro merge/split probes are worth running.

use crate::clustering::Clustering;
use crate::contraction::create_reduced_network;
use crate::graph::Graph;
use crate::workspace::Workspace;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

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
            if edge_weight > top_neighbor_weight[c] {
                second_neighbor_weight[c] = top_neighbor_weight[c];
                second_neighbor[c] = top_neighbor[c];
                top_neighbor_weight[c] = edge_weight;
                top_neighbor[c] = nbr as i64;
            } else if edge_weight > second_neighbor_weight[c] {
                second_neighbor_weight[c] = edge_weight;
                second_neighbor[c] = nbr as i64;
            }

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

#[cfg(test)]
mod tests {
    use super::*;

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
}
