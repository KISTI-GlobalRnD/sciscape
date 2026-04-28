//! Local merging algorithm for the refinement phase of Leiden.
//!
//! Port of CWTS LocalMergingAlgorithm.java.
//! Hot-path optimized with unsafe unchecked indexing.

use crate::clustering::Clustering;
use crate::graph::Graph;
use crate::random_utils::{fill_identity_u32, permute_cwts_style};
use rand::Rng;

pub struct LocalMergeWorkspace {
    cw: Vec<f64>,
    npc: Vec<u32>,
    order: Vec<u32>,
    ewpc: Vec<f64>,
    external: Vec<f64>,
    non_singleton: Vec<u8>,
    nc_buf: Vec<u32>,
    clusters: Vec<u32>,
}

impl LocalMergeWorkspace {
    pub fn new(n: usize) -> Self {
        Self {
            cw: Vec::with_capacity(n),
            npc: Vec::with_capacity(n),
            order: Vec::with_capacity(n),
            ewpc: Vec::with_capacity(n),
            external: Vec::with_capacity(n),
            non_singleton: Vec::with_capacity(n),
            nc_buf: Vec::with_capacity(64),
            clusters: Vec::with_capacity(n),
        }
    }

    pub(crate) fn assignments(&self) -> &[u32] {
        &self.clusters
    }

    fn prepare_assignments(&mut self, n: usize) {
        fill_identity_u32(&mut self.clusters, n);
    }

    fn prepare_order(&mut self, n: usize) {
        fill_identity_u32(&mut self.order, n);
    }

    fn prepare(&mut self, graph: &Graph) {
        let n = graph.n_nodes;

        self.cw.clear();
        self.cw.extend_from_slice(&graph.node_weights);

        self.npc.clear();
        self.npc.resize(n, 1);

        self.prepare_order(n);

        // Kept zeroed by clearing every touched cluster after each node.
        self.ewpc.resize(n, 0.0);

        self.external.clear();
        self.external.resize(n, 0.0);
        for node in 0..n {
            let start = graph.first_neighbor_index[node] as usize;
            let end = graph.first_neighbor_index[node + 1] as usize;
            let mut total = 0.0;
            for k in start..end {
                if graph.neighbors[k] as usize != node {
                    total += graph.edge_weights[k];
                }
            }
            self.external[node] = total;
        }

        self.non_singleton.clear();
        self.non_singleton.resize(n, 0);

        self.nc_buf.clear();
        self.prepare_assignments(n);
    }

    fn prepare_induced(&mut self, graph: &Graph, nodes: &[u32]) {
        let n = nodes.len();

        self.cw.clear();
        self.cw.reserve(n);
        for &node in nodes {
            self.cw.push(graph.node_weights[node as usize]);
        }

        self.npc.clear();
        self.npc.resize(n, 1);

        self.prepare_order(n);

        // Kept zeroed by clearing every touched cluster after each node.
        self.ewpc.resize(n, 0.0);

        self.external.clear();
        self.external.resize(n, 0.0);

        self.non_singleton.clear();
        self.non_singleton.resize(n, 0);

        self.nc_buf.clear();
        self.prepare_assignments(n);
    }
}

#[inline]
fn java_fast_exp(exponent: f64) -> f64 {
    if exponent < -256.0 {
        return 0.0;
    }

    let mut value = 1.0 + exponent / 256.0;
    for _ in 0..8 {
        value *= value;
    }
    value
}

#[inline]
fn transformed_increment(inc: f64, randomness: f64) -> f64 {
    if randomness == 0.0 {
        if inc == 0.0 {
            1.0
        } else {
            f64::INFINITY
        }
    } else {
        java_fast_exp(inc / randomness)
    }
}

/// Find clustering of a (sub)network via local merging.
pub fn find_clustering(
    graph: &Graph,
    resolution: f64,
    randomness: f64,
    rng: &mut impl Rng,
) -> Clustering {
    let mut ws = LocalMergeWorkspace::new(graph.n_nodes);
    find_clustering_with_workspace(graph, resolution, randomness, rng, &mut ws)
}

/// Find clustering using caller-owned buffers.
pub fn find_clustering_with_workspace(
    graph: &Graph,
    resolution: f64,
    randomness: f64,
    rng: &mut impl Rng,
    ws: &mut LocalMergeWorkspace,
) -> Clustering {
    let mut cluster_sizes = Vec::new();
    find_clustering_with_workspace_and_append_sizes(
        graph,
        resolution,
        randomness,
        rng,
        ws,
        &mut cluster_sizes,
    )
}

/// Find clustering and append final cluster sizes into caller-owned storage.
pub(crate) fn find_clustering_with_workspace_and_append_sizes(
    graph: &Graph,
    resolution: f64,
    randomness: f64,
    rng: &mut impl Rng,
    ws: &mut LocalMergeWorkspace,
    cluster_sizes: &mut Vec<u32>,
) -> Clustering {
    let n_clusters = find_clustering_with_workspace_assignments_and_append_sizes(
        graph,
        resolution,
        randomness,
        rng,
        ws,
        cluster_sizes,
    );
    Clustering {
        n_nodes: graph.n_nodes,
        n_clusters,
        clusters: ws.assignments()[..graph.n_nodes].to_vec(),
        fixed: None,
    }
}

/// Find clustering into caller-owned workspace assignments and append final cluster sizes.
pub(crate) fn find_clustering_with_workspace_assignments_and_append_sizes(
    graph: &Graph,
    resolution: f64,
    randomness: f64,
    rng: &mut impl Rng,
    ws: &mut LocalMergeWorkspace,
    cluster_sizes: &mut Vec<u32>,
) -> usize {
    let n = graph.n_nodes;

    if n <= 1 {
        ws.clusters.clear();
        if n == 1 {
            ws.clusters.push(0);
            cluster_sizes.push(1);
        }
        return n;
    }

    let first_nbr = graph.first_neighbor_index.as_ptr();
    let nbr_arr = graph.neighbors.as_ptr();
    let ew_arr = graph.edge_weights.as_ptr();
    let nw_arr = graph.node_weights.as_slice();

    ws.prepare(graph);
    let clusters = ws.clusters.as_mut_ptr();
    let cw = &mut ws.cw[..n];
    let npc = &mut ws.npc[..n];
    let external = &mut ws.external[..n];
    let non_singleton = &mut ws.non_singleton[..n];
    let total_node_weight: f64 = nw_arr.iter().sum();

    // Random permutation
    let order = &mut ws.order[..n];
    permute_cwts_style(order, rng);

    let ewpc = &mut ws.ewpc[..n];
    let ewpc_ptr = ewpc.as_mut_ptr();
    let cw_ptr = cw.as_mut_ptr();
    let nc_buf = &mut ws.nc_buf;

    for &j in order.iter() {
        let j = j as usize;
        if non_singleton[j] != 0 || external[j] < cw[j] * (total_node_weight - cw[j]) * resolution {
            continue;
        }

        unsafe {
            *cw_ptr.add(j) = 0.0;
        }
        npc[j] = 0;
        external[j] = 0.0;

        // Scan neighbors
        nc_buf.clear();
        nc_buf.push(j as u32);
        let ns = unsafe { *first_nbr.add(j) } as usize;
        let ne = unsafe { *first_nbr.add(j + 1) } as usize;

        unsafe {
            for k in ns..ne {
                let nc = *clusters.add(*nbr_arr.add(k) as usize) as usize;
                if *ewpc_ptr.add(nc) == 0.0 {
                    nc_buf.push(nc as u32);
                }
                *ewpc_ptr.add(nc) += *ew_arr.add(k);
            }
        }

        let node_w = unsafe { *nw_arr.get_unchecked(j) };
        let mut best = j;
        let mut max_inc = 0.0;
        let mut total = 0.0;

        for idx in 0..nc_buf.len() {
            let nc = unsafe { *nc_buf.get_unchecked(idx) } as usize;
            if external[nc]
                >= unsafe { *cw_ptr.add(nc) }
                    * (total_node_weight - unsafe { *cw_ptr.add(nc) })
                    * resolution
            {
                let inc = unsafe { *ewpc_ptr.add(nc) - node_w * *cw_ptr.add(nc) * resolution };
                if inc > max_inc {
                    best = nc;
                    max_inc = inc;
                }
                if inc >= 0.0 {
                    total += transformed_increment(inc, randomness);
                }
            }
        }

        if total.is_finite() {
            let mut r = rng.gen::<f64>() * total;
            for idx in 0..nc_buf.len() {
                let nc = unsafe { *nc_buf.get_unchecked(idx) } as usize;
                if external[nc]
                    >= unsafe { *cw_ptr.add(nc) }
                        * (total_node_weight - unsafe { *cw_ptr.add(nc) })
                        * resolution
                {
                    let inc = unsafe { *ewpc_ptr.add(nc) - node_w * *cw_ptr.add(nc) * resolution };
                    if inc >= 0.0 {
                        let weight = transformed_increment(inc, randomness);
                        r -= weight;
                        if r <= 0.0 {
                            best = nc;
                            break;
                        }
                    }
                }
            }
        }

        // Reset
        unsafe {
            *ewpc_ptr.add(j) = 0.0;
        }
        for idx in 0..nc_buf.len() {
            unsafe {
                *ewpc_ptr.add(*nc_buf.get_unchecked(idx) as usize) = 0.0;
            }
        }

        // Assign
        unsafe {
            *clusters.add(j) = best as u32;
            *cw_ptr.add(best) += node_w;
        }
        npc[best] += 1;

        for k in ns..ne {
            let nbr = unsafe { *nbr_arr.add(k) } as usize;
            let w = unsafe { *ew_arr.add(k) };
            if unsafe { *clusters.add(nbr) } as usize == best {
                external[best] -= w;
            } else {
                external[best] += w;
            }
        }

        if best != j {
            non_singleton[best] = 1;
        }
    }

    compact_assignments_from_counts(&mut ws.clusters[..n], n, npc, cluster_sizes)
}

/// Find clustering for an induced subgraph without materializing its CSR.
///
/// `nodes` contains original graph node IDs. `local_index` must be initialized
/// with `u32::MAX`; this function marks `nodes`, filters original CSR neighbors
/// through the marker, and unmarks before returning.
pub fn find_clustering_induced_u32_with_workspace(
    graph: &Graph,
    nodes: &[u32],
    local_index: &mut [u32],
    resolution: f64,
    randomness: f64,
    rng: &mut impl Rng,
    ws: &mut LocalMergeWorkspace,
) -> Clustering {
    let mut cluster_sizes = Vec::new();
    find_clustering_induced_u32_with_workspace_and_append_sizes(
        graph,
        nodes,
        local_index,
        resolution,
        randomness,
        rng,
        ws,
        &mut cluster_sizes,
    )
}

/// Find induced clustering and append compacted cluster sizes into caller-owned storage.
pub(crate) fn find_clustering_induced_u32_with_workspace_and_append_sizes(
    graph: &Graph,
    nodes: &[u32],
    local_index: &mut [u32],
    resolution: f64,
    randomness: f64,
    rng: &mut impl Rng,
    ws: &mut LocalMergeWorkspace,
    cluster_sizes: &mut Vec<u32>,
) -> Clustering {
    let n_clusters = find_clustering_induced_u32_with_workspace_assignments_and_append_sizes(
        graph,
        nodes,
        local_index,
        resolution,
        randomness,
        rng,
        ws,
        cluster_sizes,
    );
    Clustering {
        n_nodes: nodes.len(),
        n_clusters,
        clusters: ws.assignments()[..nodes.len()].to_vec(),
        fixed: None,
    }
}

/// Find induced clustering into caller-owned workspace assignments and append compacted cluster sizes.
pub(crate) fn find_clustering_induced_u32_with_workspace_assignments_and_append_sizes(
    graph: &Graph,
    nodes: &[u32],
    local_index: &mut [u32],
    resolution: f64,
    randomness: f64,
    rng: &mut impl Rng,
    ws: &mut LocalMergeWorkspace,
    cluster_sizes: &mut Vec<u32>,
) -> usize {
    assert_eq!(local_index.len(), graph.n_nodes);

    let n = nodes.len();

    if n <= 1 {
        ws.clusters.clear();
        if n == 1 {
            ws.clusters.push(0);
            cluster_sizes.push(1);
        }
        return n;
    }

    for (local, &node) in nodes.iter().enumerate() {
        local_index[node as usize] = local as u32;
    }

    let first_nbr = graph.first_neighbor_index.as_ptr();
    let nbr_arr = graph.neighbors.as_ptr();
    let ew_arr = graph.edge_weights.as_ptr();
    let nw_arr = graph.node_weights.as_slice();
    let local_index_ptr = local_index.as_ptr();

    ws.prepare_induced(graph, nodes);
    let clusters = ws.clusters.as_mut_ptr();
    let cw = &mut ws.cw[..n];
    let npc = &mut ws.npc[..n];
    let external = &mut ws.external[..n];
    let non_singleton = &mut ws.non_singleton[..n];
    let total_node_weight: f64 = nodes
        .iter()
        .map(|&node| graph.node_weights[node as usize])
        .sum();

    for (local, &node) in nodes.iter().enumerate() {
        let old = node as usize;
        let ns = graph.first_neighbor_index[old] as usize;
        let ne = graph.first_neighbor_index[old + 1] as usize;
        let mut total = 0.0;
        for k in ns..ne {
            let local_nbr = local_index[graph.neighbors[k] as usize];
            if local_nbr != u32::MAX && local_nbr as usize != local {
                total += graph.edge_weights[k];
            }
        }
        external[local] = total;
    }

    let order = &mut ws.order[..n];
    permute_cwts_style(order, rng);

    let ewpc = &mut ws.ewpc[..n];
    let ewpc_ptr = ewpc.as_mut_ptr();
    let cw_ptr = cw.as_mut_ptr();
    let nc_buf = &mut ws.nc_buf;

    for &j in order.iter() {
        let j = j as usize;
        let old_j = unsafe { *nodes.get_unchecked(j) } as usize;
        if non_singleton[j] != 0 || external[j] < cw[j] * (total_node_weight - cw[j]) * resolution {
            continue;
        }

        unsafe {
            *cw_ptr.add(j) = 0.0;
        }
        npc[j] = 0;
        external[j] = 0.0;

        nc_buf.clear();
        nc_buf.push(j as u32);
        let ns = unsafe { *first_nbr.add(old_j) } as usize;
        let ne = unsafe { *first_nbr.add(old_j + 1) } as usize;

        unsafe {
            for k in ns..ne {
                let local_nbr = *local_index_ptr.add(*nbr_arr.add(k) as usize);
                if local_nbr == u32::MAX {
                    continue;
                }

                let nc = *clusters.add(local_nbr as usize) as usize;
                if *ewpc_ptr.add(nc) == 0.0 {
                    nc_buf.push(nc as u32);
                }
                *ewpc_ptr.add(nc) += *ew_arr.add(k);
            }
        }

        let node_w = unsafe { *nw_arr.get_unchecked(old_j) };
        let mut best = j;
        let mut max_inc = 0.0;
        let mut total = 0.0;

        for idx in 0..nc_buf.len() {
            let nc = unsafe { *nc_buf.get_unchecked(idx) } as usize;
            if external[nc]
                >= unsafe { *cw_ptr.add(nc) }
                    * (total_node_weight - unsafe { *cw_ptr.add(nc) })
                    * resolution
            {
                let inc = unsafe { *ewpc_ptr.add(nc) - node_w * *cw_ptr.add(nc) * resolution };
                if inc > max_inc {
                    best = nc;
                    max_inc = inc;
                }
                if inc >= 0.0 {
                    total += transformed_increment(inc, randomness);
                }
            }
        }

        if total.is_finite() {
            let mut r = rng.gen::<f64>() * total;
            for idx in 0..nc_buf.len() {
                let nc = unsafe { *nc_buf.get_unchecked(idx) } as usize;
                if external[nc]
                    >= unsafe { *cw_ptr.add(nc) }
                        * (total_node_weight - unsafe { *cw_ptr.add(nc) })
                        * resolution
                {
                    let inc = unsafe { *ewpc_ptr.add(nc) - node_w * *cw_ptr.add(nc) * resolution };
                    if inc >= 0.0 {
                        let weight = transformed_increment(inc, randomness);
                        r -= weight;
                        if r <= 0.0 {
                            best = nc;
                            break;
                        }
                    }
                }
            }
        }

        unsafe {
            *ewpc_ptr.add(j) = 0.0;
        }
        for idx in 0..nc_buf.len() {
            unsafe {
                *ewpc_ptr.add(*nc_buf.get_unchecked(idx) as usize) = 0.0;
            }
        }

        unsafe {
            *clusters.add(j) = best as u32;
            *cw_ptr.add(best) += node_w;
        }
        npc[best] += 1;

        for k in ns..ne {
            let local_nbr = unsafe { *local_index_ptr.add(*nbr_arr.add(k) as usize) };
            if local_nbr == u32::MAX {
                continue;
            }
            let w = unsafe { *ew_arr.add(k) };
            if unsafe { *clusters.add(local_nbr as usize) } as usize == best {
                external[best] -= w;
            } else {
                external[best] += w;
            }
        }

        if best != j {
            non_singleton[best] = 1;
        }
    }

    for &node in nodes {
        local_index[node as usize] = u32::MAX;
    }

    compact_assignments_from_counts(&mut ws.clusters[..n], n, npc, cluster_sizes)
}

fn compact_assignments_from_counts(
    assignments: &mut [u32],
    n_clusters: usize,
    counts: &mut [u32],
    cluster_sizes: &mut Vec<u32>,
) -> usize {
    let mut new_id = 0u32;
    for count_or_remap in counts.iter_mut().take(n_clusters) {
        let count = *count_or_remap;
        if count > 0 {
            cluster_sizes.push(count);
            *count_or_remap = new_id;
            new_id += 1;
        }
    }

    for cid in assignments {
        *cid = counts[*cid as usize];
    }
    new_id as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_local_merge_triangle() {
        let g = Graph::from_edge_list(3, &[0, 1, 2], &[1, 2, 0], &[1.0, 1.0, 1.0]);
        let mut rng = StdRng::seed_from_u64(42);
        let c = find_clustering(&g, 0.3, 0.01, &mut rng);
        assert_eq!(c.n_clusters, 1);
        assert_eq!(c.clusters, vec![0, 0, 0]);
    }

    #[test]
    fn test_local_merge_appends_cluster_sizes() {
        let g = Graph::from_edge_list(
            5,
            &[0, 1, 2, 3, 0],
            &[1, 2, 3, 4, 4],
            &[1.0, 2.0, 3.0, 4.0, 0.5],
        );
        let mut rng = StdRng::seed_from_u64(11);
        let mut ws = LocalMergeWorkspace::new(0);
        let mut sizes = vec![99];

        let c = find_clustering_with_workspace_and_append_sizes(
            &g, 0.3, 0.0, &mut rng, &mut ws, &mut sizes,
        );

        let mut expected = vec![0u32; c.n_clusters];
        for &cid in &c.clusters {
            expected[cid as usize] += 1;
        }
        assert_eq!(&sizes[1..], expected.as_slice());
    }

    #[test]
    fn test_local_merge_workspace_assignments_match_wrapper() {
        let g = Graph::from_edge_list(
            6,
            &[0, 1, 2, 3, 4, 0, 2],
            &[1, 2, 3, 4, 5, 5, 4],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 0.5, 1.5],
        );

        let mut wrapper_rng = StdRng::seed_from_u64(17);
        let mut wrapper_ws = LocalMergeWorkspace::new(0);
        let wrapped =
            find_clustering_with_workspace(&g, 0.2, 0.0, &mut wrapper_rng, &mut wrapper_ws);

        let mut direct_rng = StdRng::seed_from_u64(17);
        let mut direct_ws = LocalMergeWorkspace::new(0);
        let mut sizes = Vec::new();
        let n_clusters = find_clustering_with_workspace_assignments_and_append_sizes(
            &g,
            0.2,
            0.0,
            &mut direct_rng,
            &mut direct_ws,
            &mut sizes,
        );

        assert_eq!(n_clusters, wrapped.n_clusters);
        assert_eq!(direct_ws.assignments(), wrapped.clusters.as_slice());
        assert_eq!(sizes.iter().sum::<u32>(), g.n_nodes as u32);
    }

    #[test]
    fn test_local_merge_pair_fast_path_merges_and_sizes() {
        let g = Graph::from_edge_list(2, &[0], &[1], &[10.0]);
        let mut rng = StdRng::seed_from_u64(3);
        let mut ws = LocalMergeWorkspace::new(0);
        let mut sizes = Vec::new();

        let c = find_clustering_with_workspace_and_append_sizes(
            &g, 0.1, 0.0, &mut rng, &mut ws, &mut sizes,
        );

        assert_eq!(c.n_clusters, 1);
        assert_eq!(c.clusters, vec![0, 0]);
        assert_eq!(sizes, vec![2]);
    }

    #[test]
    fn test_induced_pair_fast_path_ignores_external_edges() {
        let g = Graph::from_edge_list(3, &[0, 0, 1], &[1, 2, 2], &[0.01, 10.0, 10.0]);
        let nodes = [0u32, 1];
        let mut marker = vec![u32::MAX; g.n_nodes];
        let mut rng = StdRng::seed_from_u64(5);
        let mut ws = LocalMergeWorkspace::new(0);
        let mut sizes = Vec::new();

        let c = find_clustering_induced_u32_with_workspace_and_append_sizes(
            &g,
            &nodes,
            &mut marker,
            1.0,
            0.0,
            &mut rng,
            &mut ws,
            &mut sizes,
        );

        assert_eq!(c.n_clusters, 2);
        assert_eq!(c.clusters, vec![0, 1]);
        assert_eq!(sizes, vec![1, 1]);
        assert!(marker.iter().all(|&idx| idx == u32::MAX));
    }

    #[test]
    fn test_induced_local_merge_ignores_external_edges() {
        let g = Graph::from_edge_list(
            5,
            &[0, 1, 2, 1, 3],
            &[1, 2, 3, 3, 4],
            &[10.0, 1.0, 1.0, 1.0, 10.0],
        );
        let nodes = [1u32, 2, 3];
        let mut marker = vec![u32::MAX; g.n_nodes];
        let mut sub_rng = StdRng::seed_from_u64(7);
        let subgraph = g.subgraph_with_marker_u32(&nodes, &mut marker);
        let materialized = find_clustering(&subgraph, 0.1, 0.0, &mut sub_rng);

        let mut induced_rng = StdRng::seed_from_u64(7);
        let mut ws = LocalMergeWorkspace::new(0);
        let induced = find_clustering_induced_u32_with_workspace(
            &g,
            &nodes,
            &mut marker,
            0.1,
            0.0,
            &mut induced_rng,
            &mut ws,
        );

        assert_eq!(materialized.n_clusters, 1);
        assert_eq!(induced.n_clusters, materialized.n_clusters);
        assert_eq!(induced.clusters, materialized.clusters);
        assert!(marker.iter().all(|&idx| idx == u32::MAX));
    }
}
