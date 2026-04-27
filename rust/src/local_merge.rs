//! Local merging algorithm for the refinement phase of Leiden.
//!
//! Port of CWTS LocalMergingAlgorithm.java.
//! Hot-path optimized with unsafe unchecked indexing.

use crate::clustering::Clustering;
use crate::graph::Graph;
use crate::workspace::Workspace;
use rand::Rng;

/// Find clustering of a (sub)network via local merging.
pub fn find_clustering(
    graph: &Graph,
    resolution: f64,
    randomness: f64,
    rng: &mut impl Rng,
    ws: &mut Workspace,
) -> Clustering {
    let n = graph.n_nodes;
    let mut clustering = Clustering::singleton(n);

    if n <= 1 {
        return clustering;
    }

    let first_nbr = graph.first_neighbor_index.as_ptr();
    let nbr_arr = graph.neighbors.as_ptr();
    let ew_arr = graph.edge_weights.as_ptr();
    let nw_arr = graph.node_weights.as_deref();
    let slw_arr = graph.self_loop_weights.as_deref();
    let clusters = clustering.clusters.as_mut_ptr();

    ws.ensure_capacity(n);
    let cw = &mut ws.cw[..n];
    if let Some(nw_arr) = nw_arr {
        cw.copy_from_slice(nw_arr);
    } else {
        cw.fill(1.0);
    }
    let npc = &mut ws.npc[..n];
    npc.fill(1);

    // Random permutation
    let mut order = std::mem::take(&mut ws.order_u32);
    order.clear();
    order.extend(0..n as u32);
    for i in (1..n).rev() {
        let j = rng.gen_range(0..=i);
        order.swap(i, j);
    }

    let ewpc = &mut ws.ewpc[..n];
    ewpc.fill(0.0);
    let ewpc_ptr = ewpc.as_mut_ptr();
    let cw_ptr = cw.as_mut_ptr();
    let mut nc_buf = std::mem::take(&mut ws.nc_buf);
    nc_buf.clear();

    for &j in &order {
        let j = j as usize;
        let cur = unsafe { *clusters.add(j) as usize };

        unsafe {
            *cw_ptr.add(cur) -= nw_arr.map_or(1.0, |nw| *nw.get_unchecked(j));
        }
        npc[cur] -= 1;

        // Scan neighbors
        nc_buf.clear();
        let ns = unsafe { *first_nbr.add(j) } as usize;
        let ne = unsafe { *first_nbr.add(j + 1) } as usize;

        unsafe {
            for k in ns..ne {
                let nc = *clusters.add(*nbr_arr.add(k) as usize) as usize;
                if nc != cur && *ewpc_ptr.add(nc) == 0.0 {
                    nc_buf.push(nc as u32);
                }
                *ewpc_ptr.add(nc) += *ew_arr.add(k);
            }
            if let Some(slw_arr) = slw_arr {
                *ewpc_ptr.add(cur) += *slw_arr.get_unchecked(j);
            }
        }

        let node_w = nw_arr.map_or(1.0, |nw| unsafe { *nw.get_unchecked(j) });
        let cur_inc = unsafe { *ewpc_ptr.add(cur) - node_w * *cw_ptr.add(cur) * resolution };

        let mut best = cur;
        let mut candidates: Vec<(usize, f64)> = Vec::new();

        for idx in 0..nc_buf.len() {
            let nc = unsafe { *nc_buf.get_unchecked(idx) } as usize;
            let inc = unsafe { *ewpc_ptr.add(nc) - node_w * *cw_ptr.add(nc) * resolution };
            if inc > cur_inc {
                candidates.push((nc, inc - cur_inc));
            }
        }

        if !candidates.is_empty() {
            if randomness == 0.0 {
                best = candidates
                    .iter()
                    .max_by(|a, b| a.1.total_cmp(&b.1))
                    .unwrap()
                    .0;
            } else {
                let max_val = candidates
                    .iter()
                    .map(|c| c.1)
                    .fold(f64::NEG_INFINITY, f64::max);
                let total: f64 = candidates
                    .iter()
                    .map(|c| ((c.1 - max_val) / randomness).exp())
                    .sum();
                let mut r = rng.gen::<f64>() * total;
                for &(nc, inc) in &candidates {
                    r -= ((inc - max_val) / randomness).exp();
                    if r <= 0.0 {
                        best = nc;
                        break;
                    }
                }
                if best == cur && !candidates.is_empty() {
                    best = candidates.last().unwrap().0;
                }
            }
        }

        // Reset
        unsafe {
            *ewpc_ptr.add(cur) = 0.0;
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
    }

    ws.order_u32 = order;
    ws.nc_buf = nc_buf;
    clustering.remove_empty_clusters();
    clustering
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
        let mut ws = Workspace::new(3);
        let c = find_clustering(&g, 0.3, 0.01, &mut rng, &mut ws);
        assert_eq!(c.n_clusters, 1);
    }
}
