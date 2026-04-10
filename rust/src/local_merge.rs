//! Local merging algorithm for the refinement phase of Leiden.
//!
//! Port of CWTS LocalMergingAlgorithm.java.
//! Hot-path optimized with unsafe unchecked indexing.

use crate::graph::Graph;
use crate::clustering::Clustering;
use rand::Rng;

/// Find clustering of a (sub)network via local merging.
pub fn find_clustering(
    graph: &Graph,
    resolution: f64,
    randomness: f64,
    rng: &mut impl Rng,
) -> Clustering {
    let n = graph.n_nodes;
    let mut clustering = Clustering::singleton(n);

    if n <= 1 {
        return clustering;
    }

    let first_nbr = graph.first_neighbor_index.as_ptr();
    let nbr_arr = graph.neighbors.as_ptr();
    let ew_arr = graph.edge_weights.as_ptr();
    let nw_arr = graph.node_weights.as_slice();
    let slw_arr = graph.self_loop_weights.as_slice();
    let clusters = clustering.clusters.as_mut_ptr();

    let mut cw: Vec<f64> = graph.node_weights.clone();
    let mut npc = vec![1u32; n];

    // Random permutation
    let mut order: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = rng.gen_range(0..=i);
        order.swap(i, j);
    }

    let mut ewpc = vec![0.0f64; n];
    let ewpc_ptr = ewpc.as_mut_ptr();
    let cw_ptr = cw.as_mut_ptr();
    let mut nc_buf: Vec<u32> = Vec::with_capacity(64);

    for &j in &order {
        let cur = unsafe { *clusters.add(j) };

        unsafe {
            *cw_ptr.add(cur) -= *nw_arr.get_unchecked(j);
        }
        npc[cur] -= 1;

        // Scan neighbors
        nc_buf.clear();
        let ns = unsafe { *first_nbr.add(j) } as usize;
        let ne = unsafe { *first_nbr.add(j + 1) } as usize;

        unsafe {
            for k in ns..ne {
                let nc = *clusters.add(*nbr_arr.add(k) as usize);
                if nc != cur && *ewpc_ptr.add(nc) == 0.0 {
                    nc_buf.push(nc as u32);
                }
                *ewpc_ptr.add(nc) += *ew_arr.add(k);
            }
            *ewpc_ptr.add(cur) += *slw_arr.get_unchecked(j);
        }

        let node_w = unsafe { *nw_arr.get_unchecked(j) };
        let cur_inc = unsafe {
            *ewpc_ptr.add(cur) - node_w * *cw_ptr.add(cur) * resolution
        };

        let mut best = cur;
        let mut candidates: Vec<(usize, f64)> = Vec::new();

        for idx in 0..nc_buf.len() {
            let nc = unsafe { *nc_buf.get_unchecked(idx) } as usize;
            let inc = unsafe {
                *ewpc_ptr.add(nc) - node_w * *cw_ptr.add(nc) * resolution
            };
            if inc > cur_inc {
                candidates.push((nc, inc - cur_inc));
            }
        }

        if !candidates.is_empty() {
            if randomness == 0.0 {
                best = candidates.iter().max_by(|a, b| a.1.total_cmp(&b.1)).unwrap().0;
            } else {
                let max_val = candidates.iter().map(|c| c.1).fold(f64::NEG_INFINITY, f64::max);
                let total: f64 = candidates.iter().map(|c| ((c.1 - max_val) / randomness).exp()).sum();
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
        unsafe { *ewpc_ptr.add(cur) = 0.0; }
        for idx in 0..nc_buf.len() {
            unsafe { *ewpc_ptr.add(*nc_buf.get_unchecked(idx) as usize) = 0.0; }
        }

        // Assign
        unsafe {
            *clusters.add(j) = best;
            *cw_ptr.add(best) += node_w;
        }
        npc[best] += 1;
    }

    clustering.remove_empty_clusters();
    clustering
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn test_local_merge_triangle() {
        let g = Graph::from_edge_list(3, &[0, 1, 2], &[1, 2, 0], &[1.0, 1.0, 1.0]);
        let mut rng = StdRng::seed_from_u64(42);
        let c = find_clustering(&g, 0.3, 0.01, &mut rng);
        assert_eq!(c.n_clusters, 1);
    }
}
