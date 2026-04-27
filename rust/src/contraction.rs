//! Graph contraction (aggregation) for hierarchical Leiden.
//!
//! Single O(n+m) traversal with scatter-gather. Uses Workspace arrays.

use crate::clustering::Clustering;
use crate::graph::Graph;
use crate::workspace::Workspace;

/// Create a contracted (reduced) graph from a clustering.
pub fn create_reduced_network(
    graph: &Graph,
    clustering: &Clustering,
    keep_self_loops: bool,
    ws: &mut Workspace,
) -> Graph {
    let n_clusters = clustering.n_clusters;
    let n_nodes = graph.n_nodes;
    let clusters = clustering.clusters.as_slice();

    let first_nbr_ptr = graph.first_neighbor_index.as_ptr();
    let nbr_ptr = graph.neighbors.as_ptr();
    let ew_ptr = graph.edge_weights.as_ptr();

    ws.ensure_capacity(n_nodes.max(n_clusters));

    let node_weights = graph.node_weights.as_deref();

    // Build nodes_per_cluster + carry self-loop weights in ONE pass.
    // Only materialize per-cluster self-loop storage when the caller needs it.
    let mut self_loop_weights = keep_self_loops.then(|| vec![0.0f64; n_clusters]);
    let npc_count = &mut ws.npc[..n_clusters];
    npc_count.fill(0);

    for node in 0..n_nodes {
        let c = unsafe { *clusters.get_unchecked(node) as usize };
        npc_count[c] += 1;
        if let (Some(dst_self_loops), Some(src_self_loops)) =
            (self_loop_weights.as_mut(), graph.self_loop_weights.as_ref())
        {
            unsafe {
                dst_self_loops[c] += *src_self_loops.get_unchecked(node);
            }
        }
    }

    // Prefix sum for npc_starts
    let npc_starts = &mut ws.npc_starts[..n_clusters + 1];
    npc_starts[0] = 0;
    for c in 0..n_clusters {
        npc_starts[c + 1] = npc_starts[c] + npc_count[c];
    }

    // Fill npc_nodes
    let npc_nodes = &mut ws.npc_nodes[..n_nodes];
    let npc_off = &mut ws.npc_off[..n_clusters];
    npc_off.copy_from_slice(&npc_starts[..n_clusters]);
    for node in 0..n_nodes {
        let c = unsafe { *clusters.get_unchecked(node) as usize };
        let pos = npc_off[c] as usize;
        npc_nodes[pos] = node as u32;
        npc_off[c] += 1;
    }

    // Scatter arrays from workspace
    let temp_w = &mut ws.temp_w[..n_clusters];
    temp_w.fill(0.0);
    let temp_w_ptr = temp_w.as_mut_ptr();
    ws.temp_used.clear();

    // Pass 1: Count unique neighbor clusters + self-loops directly into the
    // future CSR row-pointer buffer. This avoids keeping a separate degree
    // vector alive during contraction.
    let mut first_neighbor_index = vec![0u64; n_clusters + 1];

    for c in 0..n_clusters {
        let cs = npc_starts[c] as usize;
        let ce = npc_starts[c + 1] as usize;

        for idx in cs..ce {
            let node = unsafe { *npc_nodes.get_unchecked(idx) } as usize;
            let ns = unsafe { *first_nbr_ptr.add(node) } as usize;
            let ne = unsafe { *first_nbr_ptr.add(node + 1) } as usize;
            for k in ns..ne {
                let c_dst = unsafe { *clusters.get_unchecked(*nbr_ptr.add(k) as usize) as usize };
                if c_dst != c {
                    if unsafe { *temp_w_ptr.add(c_dst) } == 0.0 {
                        ws.temp_used.push(c_dst as u32);
                        unsafe {
                            *temp_w_ptr.add(c_dst) = 1.0;
                        }
                    }
                } else if keep_self_loops {
                    let nbr_node = unsafe { *nbr_ptr.add(k) } as usize;
                    if nbr_node > node {
                        if let Some(dst_self_loops) = self_loop_weights.as_mut() {
                            dst_self_loops[c] += unsafe { *ew_ptr.add(k) };
                        }
                    }
                }
            }
        }

        first_neighbor_index[c + 1] = ws.temp_used.len() as u64;
        for &u in &ws.temp_used {
            unsafe {
                *temp_w_ptr.add(u as usize) = 0.0;
            }
        }
        ws.temp_used.clear();
    }

    // Build CSR row pointers in place.
    for c in 0..n_clusters {
        first_neighbor_index[c + 1] += first_neighbor_index[c];
    }
    let n_edges = first_neighbor_index[n_clusters] as usize;

    let mut neighbors = vec![0u32; n_edges];
    let mut edge_weights = vec![0.0f64; n_edges];

    for c in 0..n_clusters {
        let cs = npc_starts[c] as usize;
        let ce = npc_starts[c + 1] as usize;

        for idx in cs..ce {
            let node = unsafe { *npc_nodes.get_unchecked(idx) } as usize;
            let ns = unsafe { *first_nbr_ptr.add(node) } as usize;
            let ne = unsafe { *first_nbr_ptr.add(node + 1) } as usize;
            for k in ns..ne {
                let c_dst = unsafe { *clusters.get_unchecked(*nbr_ptr.add(k) as usize) as usize };
                if c_dst != c {
                    if unsafe { *temp_w_ptr.add(c_dst) } == 0.0 {
                        ws.temp_used.push(c_dst as u32);
                    }
                    unsafe {
                        *temp_w_ptr.add(c_dst) += *ew_ptr.add(k);
                    }
                }
            }
        }

        // Use the row end-pointer as a descending cursor so we do not need a
        // second write-position buffer. Row order is irrelevant.
        for &u in ws.temp_used.iter().rev() {
            let u = u as usize;
            first_neighbor_index[c + 1] -= 1;
            let pos = first_neighbor_index[c + 1] as usize;
            neighbors[pos] = u as u32;
            edge_weights[pos] = unsafe { *temp_w_ptr.add(u) };
            unsafe {
                *temp_w_ptr.add(u) = 0.0;
            }
        }
        ws.temp_used.clear();
    }

    // Restore CSR row pointers after using first_neighbor_index[c + 1] as the
    // temporary descending write cursor for row c.
    for c in 0..n_clusters {
        first_neighbor_index[c] = first_neighbor_index[c + 1];
    }
    first_neighbor_index[n_clusters] = n_edges as u64;

    // Node weights
    let mut cluster_weights = vec![0.0f64; n_clusters];
    for node in 0..n_nodes {
        let cid = unsafe { *clusters.get_unchecked(node) as usize };
        let node_w = node_weights.map_or(1.0, |weights| unsafe { *weights.get_unchecked(node) });
        cluster_weights[cid] += node_w;
    }

    Graph {
        n_nodes: n_clusters,
        n_edges,
        first_neighbor_index,
        neighbors,
        edge_weights,
        node_weights: Some(cluster_weights),
        self_loop_weights,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contraction_self_loops() {
        let g = Graph::from_edge_list(3, &[0, 1, 2], &[1, 2, 0], &[1.0, 1.0, 1.0]);
        let c = Clustering::from_assignments(vec![0, 0, 0]);
        let mut ws = Workspace::new(3);
        let reduced = create_reduced_network(&g, &c, true, &mut ws);
        assert_eq!(reduced.n_nodes, 1);
        let self_loops = reduced.self_loop_weights.as_ref().unwrap();
        assert!((self_loops[0] - 3.0).abs() < 1e-10);
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
}
