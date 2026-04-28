//! Graph contraction (aggregation) for hierarchical Leiden.
//!
//! Single O(n+m) traversal with scatter-gather. Uses Workspace arrays.

use crate::clustering::Clustering;
use crate::graph::Graph;
use crate::trace;
use crate::workspace::Workspace;
use rayon::prelude::*;
use std::sync::OnceLock;

// Parallel contraction is kept as an experimental opt-in path. On current
// synthetic profiles the optimized sequential contraction wins at 3M directed
// edges because contraction is memory-bandwidth bound and the parallel path
// needs thread-local scatter buffers. Set
// SCISCAPE_PARALLEL_CONTRACTION_MIN_DIRECTED_EDGES explicitly to enable it.
const DEFAULT_PARALLEL_CONTRACTION_MIN_DIRECTED_EDGES: usize = usize::MAX;

fn leiden_threads() -> usize {
    static THREADS: OnceLock<usize> = OnceLock::new();
    *THREADS.get_or_init(|| {
        std::env::var("SCISCAPE_LEIDEN_THREADS")
            .ok()
            .or_else(|| std::env::var("RAYON_NUM_THREADS").ok())
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|&value| value > 0)
            .unwrap_or(1)
    })
}

fn parallel_contraction_min_directed_edges() -> usize {
    static MIN_EDGES: OnceLock<usize> = OnceLock::new();
    *MIN_EDGES.get_or_init(|| {
        std::env::var("SCISCAPE_PARALLEL_CONTRACTION_MIN_DIRECTED_EDGES")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(DEFAULT_PARALLEL_CONTRACTION_MIN_DIRECTED_EDGES)
    })
}

fn should_trace_contraction(input_edges: usize) -> bool {
    trace::should_trace_edges(input_edges)
}

fn reduced_edge_capacity_bound(input_edges: usize, n_clusters: usize) -> usize {
    let complete_directed_cluster_graph = n_clusters.saturating_mul(n_clusters.saturating_sub(1));
    input_edges.min(complete_directed_cluster_graph)
}

fn cluster_ranges(n_clusters: usize, n_threads: usize) -> Vec<(usize, usize)> {
    let n_threads = n_threads.min(n_clusters).max(1);
    let chunk_size = n_clusters.div_ceil(n_threads);
    (0..n_clusters)
        .step_by(chunk_size)
        .map(|start| (start, (start + chunk_size).min(n_clusters)))
        .collect()
}

/// Create a contracted (reduced) graph from a clustering.
pub fn create_reduced_network(
    graph: &Graph,
    clustering: &Clustering,
    keep_self_loops: bool,
    ws: &mut Workspace,
) -> Graph {
    let threads = leiden_threads();
    if threads > 1
        && graph.n_edges >= parallel_contraction_min_directed_edges()
        && clustering.n_clusters > 1
    {
        return create_reduced_network_parallel(graph, clustering, keep_self_loops, ws, threads);
    }

    create_reduced_network_sequential(graph, clustering, keep_self_loops, ws)
}

/// Create a contracted graph from node groups already stored in `ws`.
///
/// Expects `ws.npc_starts[..=n_clusters]` and `ws.npc_nodes[..graph.n_nodes]`
/// to describe nodes grouped by `clustering.clusters`. This skips the two
/// full-node grouping passes in `create_reduced_network`.
pub fn create_reduced_network_from_workspace_groups(
    graph: &Graph,
    clustering: &Clustering,
    keep_self_loops: bool,
    ws: &mut Workspace,
) -> Graph {
    create_reduced_network_grouped_from_workspace(graph, clustering, keep_self_loops, ws)
}

/// Create a contracted graph when row starts are known but grouped nodes are not.
///
/// This fills `ws.npc_nodes` in one full-node pass using the supplied starts,
/// then runs the grouped contraction core.
pub fn create_reduced_network_from_starts(
    graph: &Graph,
    clustering: &Clustering,
    group_starts: &[u32],
    keep_self_loops: bool,
    ws: &mut Workspace,
) -> Graph {
    let n_clusters = clustering.n_clusters;
    let n_nodes = graph.n_nodes;
    debug_assert_eq!(group_starts.len(), n_clusters + 1);

    ws.ensure_capacity(n_nodes.max(n_clusters));

    let offsets = &mut ws.npc_off[..n_clusters];
    offsets.copy_from_slice(&group_starts[..n_clusters]);

    let nodes = &mut ws.npc_nodes[..n_nodes];
    for node in 0..n_nodes {
        let c = clustering.clusters[node] as usize;
        let pos = offsets[c] as usize;
        nodes[pos] = node as u32;
        offsets[c] += 1;
    }

    create_reduced_network_grouped_from_external_starts(
        graph,
        clustering,
        group_starts,
        keep_self_loops,
        ws,
    )
}

fn create_reduced_network_sequential(
    graph: &Graph,
    clustering: &Clustering,
    keep_self_loops: bool,
    ws: &mut Workspace,
) -> Graph {
    clustering.fill_cluster_groups(ws);
    create_reduced_network_grouped_from_workspace(graph, clustering, keep_self_loops, ws)
}

fn create_reduced_network_grouped_from_workspace(
    graph: &Graph,
    clustering: &Clustering,
    keep_self_loops: bool,
    ws: &mut Workspace,
) -> Graph {
    let n_clusters = clustering.n_clusters;
    let n_nodes = graph.n_nodes;
    ws.ensure_capacity(n_nodes.max(n_clusters));

    let Workspace {
        npc_starts,
        npc_nodes,
        temp_w,
        temp_seen,
        temp_used,
        ..
    } = ws;

    create_reduced_network_grouped_core(
        graph,
        clustering,
        keep_self_loops,
        &npc_starts[..n_clusters + 1],
        &npc_nodes[..n_nodes],
        &mut temp_w[..n_clusters],
        &mut temp_seen[..n_clusters],
        temp_used,
    )
}

fn create_reduced_network_grouped_from_external_starts(
    graph: &Graph,
    clustering: &Clustering,
    group_starts: &[u32],
    keep_self_loops: bool,
    ws: &mut Workspace,
) -> Graph {
    let n_clusters = clustering.n_clusters;
    let n_nodes = graph.n_nodes;
    debug_assert_eq!(group_starts.len(), n_clusters + 1);
    ws.ensure_capacity(n_nodes.max(n_clusters));

    let Workspace {
        npc_nodes,
        temp_w,
        temp_seen,
        temp_used,
        ..
    } = ws;

    create_reduced_network_grouped_core(
        graph,
        clustering,
        keep_self_loops,
        group_starts,
        &npc_nodes[..n_nodes],
        &mut temp_w[..n_clusters],
        &mut temp_seen[..n_clusters],
        temp_used,
    )
}

fn create_reduced_network_grouped_core(
    graph: &Graph,
    clustering: &Clustering,
    keep_self_loops: bool,
    npc_starts: &[u32],
    npc_nodes: &[u32],
    temp_w: &mut [f64],
    temp_seen: &mut [u32],
    temp_used: &mut Vec<u32>,
) -> Graph {
    let n_clusters = clustering.n_clusters;
    let n_nodes = graph.n_nodes;
    debug_assert_eq!(npc_starts.len(), n_clusters + 1);
    debug_assert_eq!(npc_nodes.len(), n_nodes);
    debug_assert!(temp_w.len() >= n_clusters);
    debug_assert!(temp_seen.len() >= n_clusters);

    let clusters = clustering.clusters.as_slice();
    let first_nbr_ptr = graph.first_neighbor_index.as_ptr();
    let nbr_ptr = graph.neighbors.as_ptr();
    let ew_ptr = graph.edge_weights.as_ptr();

    let mut self_loop_weights = Vec::with_capacity(n_clusters);
    let mut cluster_weights = Vec::with_capacity(n_clusters);

    // Scatter arrays from workspace
    let temp_w_ptr = temp_w.as_mut_ptr();
    let temp_seen_ptr = temp_seen.as_mut_ptr();
    temp_used.clear();

    // Build CSR directly. This scans edges once: accumulate weights for one
    // source cluster, append its unique neighbor clusters, then reset markers.
    let mut first_neighbor_index = Vec::with_capacity(n_clusters + 1);
    first_neighbor_index.push(0);
    // The reduced graph cannot exceed either the input edge count or the
    // complete directed graph over reduced clusters. The second bound matters
    // in late recursion levels, where reserving the original edge count can
    // over-allocate by a large factor.
    let edge_capacity = reduced_edge_capacity_bound(graph.n_edges, n_clusters);
    let mut neighbors: Vec<u32> = Vec::with_capacity(edge_capacity);
    let mut edge_weights: Vec<f64> = Vec::with_capacity(edge_capacity);

    for c in 0..n_clusters {
        let cs = npc_starts[c] as usize;
        let ce = npc_starts[c + 1] as usize;
        let mut cluster_weight = 0.0f64;
        let mut self_loop_weight = 0.0f64;

        for idx in cs..ce {
            let node = unsafe { *npc_nodes.get_unchecked(idx) } as usize;
            cluster_weight += unsafe { *graph.node_weights.get_unchecked(node) };
            if keep_self_loops {
                self_loop_weight += unsafe { *graph.self_loop_weights.get_unchecked(node) };
            }

            let ns = unsafe { *first_nbr_ptr.add(node) } as usize;
            let ne = unsafe { *first_nbr_ptr.add(node + 1) } as usize;
            for k in ns..ne {
                let c_dst = unsafe { *clusters.get_unchecked(*nbr_ptr.add(k) as usize) } as usize;
                if c_dst != c {
                    if unsafe { *temp_seen_ptr.add(c_dst) } == u32::MAX {
                        temp_used.push(c_dst as u32);
                        unsafe {
                            *temp_seen_ptr.add(c_dst) = 1;
                        }
                    }
                    unsafe {
                        *temp_w_ptr.add(c_dst) += *ew_ptr.add(k);
                    }
                } else if keep_self_loops {
                    let nbr_node = unsafe { *nbr_ptr.add(k) } as usize;
                    if nbr_node > node {
                        self_loop_weight += unsafe { *ew_ptr.add(k) };
                    }
                }
            }
        }

        for &u in temp_used.iter() {
            let u = u as usize;
            neighbors.push(u as u32);
            edge_weights.push(unsafe { *temp_w_ptr.add(u) });
            unsafe {
                *temp_w_ptr.add(u) = 0.0;
                *temp_seen_ptr.add(u) = u32::MAX;
            }
        }
        temp_used.clear();
        first_neighbor_index.push(neighbors.len() as u64);
        cluster_weights.push(cluster_weight);
        self_loop_weights.push(self_loop_weight);
    }

    let n_edges = neighbors.len();
    if should_trace_contraction(graph.n_edges) {
        let edge_ratio = if graph.n_edges == 0 {
            0.0
        } else {
            n_edges as f64 / graph.n_edges as f64
        };
        let reserve_ratio = if graph.n_edges == 0 {
            0.0
        } else {
            edge_capacity as f64 / graph.n_edges as f64
        };
        trace::emit(format_args!(
            "phase=contraction_csr input_nodes={} input_directed_edges={} reduced_nodes={} reduced_directed_edges={} reserved_directed_edges={} edge_ratio={:.6} reserve_ratio={:.6}{}",
            graph.n_nodes,
            graph.n_edges,
            n_clusters,
            n_edges,
            edge_capacity,
            edge_ratio,
            reserve_ratio,
            trace::memory_fields(),
        ));
    }
    debug_assert_eq!(cluster_weights.len(), n_clusters);
    debug_assert_eq!(self_loop_weights.len(), n_clusters);

    Graph {
        n_nodes: n_clusters,
        n_edges,
        first_neighbor_index,
        neighbors,
        edge_weights,
        node_weights: cluster_weights,
        self_loop_weights,
    }
}

fn create_reduced_network_parallel(
    graph: &Graph,
    clustering: &Clustering,
    keep_self_loops: bool,
    ws: &mut Workspace,
    threads: usize,
) -> Graph {
    let n_clusters = clustering.n_clusters;
    let n_nodes = graph.n_nodes;
    let clusters = clustering.clusters.as_slice();

    let first_nbr = graph.first_neighbor_index.as_slice();
    let nbrs = graph.neighbors.as_slice();
    let ews = graph.edge_weights.as_slice();

    ws.ensure_capacity(n_nodes.max(n_clusters));

    let mut self_loop_weights = vec![0.0f64; n_clusters];
    let mut cluster_weights = vec![0.0f64; n_clusters];
    let npc_count = &mut ws.npc[..n_clusters];
    npc_count.fill(0);

    for node in 0..n_nodes {
        let c = unsafe { *clusters.get_unchecked(node) } as usize;
        npc_count[c] += 1;
        cluster_weights[c] += unsafe { *graph.node_weights.get_unchecked(node) };
        if keep_self_loops {
            unsafe {
                self_loop_weights[c] += *graph.self_loop_weights.get_unchecked(node);
            }
        }
    }

    let npc_starts = &mut ws.npc_starts[..n_clusters + 1];
    npc_starts[0] = 0;
    for c in 0..n_clusters {
        npc_starts[c + 1] = npc_starts[c] + npc_count[c];
    }

    let npc_nodes = &mut ws.npc_nodes[..n_nodes];
    let npc_off = &mut ws.npc_off[..n_clusters];
    npc_off.copy_from_slice(&npc_starts[..n_clusters]);
    for node in 0..n_nodes {
        let c = unsafe { *clusters.get_unchecked(node) } as usize;
        let pos = npc_off[c] as usize;
        npc_nodes[pos] = node as u32;
        npc_off[c] += 1;
    }

    let ranges = cluster_ranges(n_clusters, threads);
    let estimated_chunk_edges = graph.n_edges / ranges.len().max(1) + 1;

    struct ReducedChunk {
        start: usize,
        row_degrees: Vec<u64>,
        neighbors: Vec<u32>,
        edge_weights: Vec<f64>,
        self_loop_delta: Vec<f64>,
    }

    let chunks: Vec<ReducedChunk> = ranges
        .par_iter()
        .map(|&(c_start, c_end)| {
            let mut temp_w = vec![0.0f64; n_clusters];
            let mut temp_seen = vec![u32::MAX; n_clusters];
            let mut used: Vec<u32> = Vec::with_capacity(256);
            let mut row_degrees = Vec::with_capacity(c_end - c_start);
            let mut chunk_neighbors = Vec::with_capacity(estimated_chunk_edges);
            let mut chunk_edge_weights = Vec::with_capacity(estimated_chunk_edges);
            let mut self_loop_delta = vec![0.0f64; c_end - c_start];

            for c in c_start..c_end {
                let cs = npc_starts[c] as usize;
                let ce = npc_starts[c + 1] as usize;

                for idx in cs..ce {
                    let node = unsafe { *npc_nodes.get_unchecked(idx) } as usize;
                    let ns = unsafe { *first_nbr.get_unchecked(node) } as usize;
                    let ne = unsafe { *first_nbr.get_unchecked(node + 1) } as usize;
                    for k in ns..ne {
                        let c_dst =
                            unsafe { *clusters.get_unchecked(*nbrs.get_unchecked(k) as usize) }
                                as usize;
                        if c_dst != c {
                            if temp_seen[c_dst] == u32::MAX {
                                temp_seen[c_dst] = 1;
                                used.push(c_dst as u32);
                            }
                            temp_w[c_dst] += unsafe { *ews.get_unchecked(k) };
                        } else if keep_self_loops {
                            let nbr_node = unsafe { *nbrs.get_unchecked(k) } as usize;
                            if nbr_node > node {
                                self_loop_delta[c - c_start] += unsafe { *ews.get_unchecked(k) };
                            }
                        }
                    }
                }

                row_degrees.push(used.len() as u64);
                for &u in &used {
                    let u = u as usize;
                    chunk_neighbors.push(u as u32);
                    chunk_edge_weights.push(temp_w[u]);
                    temp_w[u] = 0.0;
                    temp_seen[u] = u32::MAX;
                }
                used.clear();
            }

            ReducedChunk {
                start: c_start,
                row_degrees,
                neighbors: chunk_neighbors,
                edge_weights: chunk_edge_weights,
                self_loop_delta,
            }
        })
        .collect();

    let mut first_neighbor_index = vec![0u64; n_clusters + 1];
    let mut running = 0u64;
    for chunk in &chunks {
        for (local, &degree) in chunk.row_degrees.iter().enumerate() {
            first_neighbor_index[chunk.start + local] = running;
            running += degree;
        }
    }
    first_neighbor_index[n_clusters] = running;

    let n_edges = running as usize;
    let mut neighbors = Vec::with_capacity(n_edges);
    let mut edge_weights = Vec::with_capacity(n_edges);
    for mut chunk in chunks {
        for (local, delta) in chunk.self_loop_delta.into_iter().enumerate() {
            self_loop_weights[chunk.start + local] += delta;
        }
        neighbors.append(&mut chunk.neighbors);
        edge_weights.append(&mut chunk.edge_weights);
    }
    debug_assert_eq!(neighbors.len(), n_edges);
    debug_assert_eq!(edge_weights.len(), n_edges);

    Graph {
        n_nodes: n_clusters,
        n_edges,
        first_neighbor_index,
        neighbors,
        edge_weights,
        node_weights: cluster_weights,
        self_loop_weights,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reduced_edge_capacity_bound_uses_cluster_complete_graph_bound() {
        assert_eq!(reduced_edge_capacity_bound(1_000, 1), 0);
        assert_eq!(reduced_edge_capacity_bound(1_000, 3), 6);
        assert_eq!(reduced_edge_capacity_bound(5, 3), 5);
    }

    #[test]
    fn test_contraction_self_loops() {
        let g = Graph::from_edge_list(3, &[0, 1, 2], &[1, 2, 0], &[1.0, 1.0, 1.0]);
        let c = Clustering::from_assignments(vec![0, 0, 0]);
        let mut ws = Workspace::new(3);
        let reduced = create_reduced_network(&g, &c, true, &mut ws);
        assert_eq!(reduced.n_nodes, 1);
        assert!((reduced.self_loop_weights[0] - 3.0).abs() < 1e-10);
        assert_eq!(reduced.n_edges, 0);
    }

    #[test]
    fn test_contraction_two_clusters() {
        let g = Graph::from_edge_list(4, &[0, 1, 2, 0], &[1, 2, 3, 3], &[1.0, 2.0, 3.0, 0.5]);
        let c = Clustering::from_assignments(vec![0, 0, 1, 1]);
        let mut ws = Workspace::new(4);
        let reduced = create_reduced_network(&g, &c, false, &mut ws);
        assert_eq!(reduced.n_nodes, 2);
        assert!((reduced.total_edge_weight() - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_contraction_reuses_marker_workspace() {
        let g = Graph::from_edge_list(5, &[0, 1, 2, 3], &[1, 2, 3, 4], &[1.0, 2.0, 3.0, 4.0]);
        let mut ws = Workspace::new(5);

        let first = create_reduced_network(
            &g,
            &Clustering::from_assignments(vec![0, 0, 1, 1, 2]),
            false,
            &mut ws,
        );
        let second = create_reduced_network(
            &g,
            &Clustering::from_assignments(vec![0, 1, 1, 2, 2]),
            false,
            &mut ws,
        );

        assert_eq!(first.n_edges, 4);
        assert_eq!(second.n_edges, 4);
        assert!((first.total_edge_weight() - 6.0).abs() < 1e-10);
        assert!((second.total_edge_weight() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_contraction_from_known_groups_matches_default() {
        let g = Graph::from_edge_list(
            6,
            &[0, 1, 2, 3, 4, 0, 2],
            &[1, 2, 3, 4, 5, 5, 4],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 0.5, 1.5],
        );
        let c = Clustering::from_assignments(vec![0, 0, 1, 1, 2, 2]);

        let mut default_ws = Workspace::new(6);
        let default = create_reduced_network(&g, &c, true, &mut default_ws);

        let starts = [0u32, 2, 4, 6];
        let mut starts_ws = Workspace::new(6);
        let from_starts = create_reduced_network_from_starts(&g, &c, &starts, true, &mut starts_ws);

        let mut grouped_ws = Workspace::new(6);
        grouped_ws.npc_starts[..4].copy_from_slice(&starts);
        grouped_ws.npc_nodes[..6].copy_from_slice(&[0, 1, 2, 3, 4, 5]);
        let from_groups =
            create_reduced_network_from_workspace_groups(&g, &c, true, &mut grouped_ws);

        assert_eq!(
            from_starts.first_neighbor_index,
            default.first_neighbor_index
        );
        assert_eq!(from_starts.neighbors, default.neighbors);
        assert_eq!(from_starts.edge_weights, default.edge_weights);
        assert_eq!(from_starts.node_weights, default.node_weights);
        assert_eq!(from_starts.self_loop_weights, default.self_loop_weights);

        assert_eq!(
            from_groups.first_neighbor_index,
            default.first_neighbor_index
        );
        assert_eq!(from_groups.neighbors, default.neighbors);
        assert_eq!(from_groups.edge_weights, default.edge_weights);
        assert_eq!(from_groups.node_weights, default.node_weights);
        assert_eq!(from_groups.self_loop_weights, default.self_loop_weights);
    }

    #[test]
    fn test_parallel_contraction_matches_sequential() {
        let g = Graph::from_edge_list(
            8,
            &[0, 1, 2, 3, 4, 5, 6, 0, 2, 4],
            &[1, 2, 3, 4, 5, 6, 7, 7, 6, 1],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 0.5, 1.5, 2.5],
        );
        let c = Clustering::from_assignments(vec![0, 0, 1, 1, 2, 2, 3, 3]);
        let mut seq_ws = Workspace::new(8);
        let mut par_ws = Workspace::new(8);

        let seq = create_reduced_network_sequential(&g, &c, true, &mut seq_ws);
        let par = create_reduced_network_parallel(&g, &c, true, &mut par_ws, 2);

        assert_eq!(par.n_nodes, seq.n_nodes);
        assert_eq!(par.n_edges, seq.n_edges);
        assert_eq!(par.first_neighbor_index, seq.first_neighbor_index);
        assert_eq!(par.neighbors, seq.neighbors);
        assert_eq!(par.edge_weights, seq.edge_weights);
        assert_eq!(par.node_weights, seq.node_weights);
        assert_eq!(par.self_loop_weights, seq.self_loop_weights);
    }
}
