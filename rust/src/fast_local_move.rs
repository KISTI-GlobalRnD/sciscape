//! Fast local moving algorithm for the move phase of Leiden.
//!
//! Port of CWTS FastLocalMovingAlgorithm.java.
//! Hot-path: unsafe unchecked indexing where bounds are guaranteed.

use crate::graph::Graph;
use crate::clustering::Clustering;
use crate::workspace::Workspace;
use rand::seq::SliceRandom;
use rand::Rng;

/// Run one iteration of fast local moving.
/// `ws` is a reusable workspace (avoids allocation per call).
pub fn improve_clustering(
    graph: &Graph,
    clustering: &mut Clustering,
    resolution: f64,
    rng: &mut impl Rng,
    ws: &mut Workspace,
) -> bool {
    let n = graph.n_nodes;
    if n <= 1 {
        return false;
    }

    let mut update = false;

    let first_nbr = graph.first_neighbor_index.as_ptr();
    let nbr_arr = graph.neighbors.as_ptr();
    let ew_arr = graph.edge_weights.as_ptr();
    let nw_arr = graph.node_weights.as_slice();
    let slw_arr = graph.self_loop_weights.as_slice();
    let clusters = clustering.clusters.as_mut_ptr();
    let has_fixed = clustering.fixed.is_some();
    let fixed_arr: &[bool] = clustering.fixed.as_deref().unwrap_or(&[]);

    // Reuse workspace arrays (already zeroed by caller or reset)
    ws.ensure_capacity(n);
    let cw = &mut ws.cw[..n];
    let npc = &mut ws.npc[..n];
    cw.fill(0.0);
    npc.fill(0);
    for i in 0..n {
        unsafe {
            let cid = *clusters.add(i);
            *cw.get_unchecked_mut(cid) += nw_arr.get_unchecked(i);
            *npc.get_unchecked_mut(cid) += 1;
        }
    }

    let mut unused: Vec<u32> = Vec::with_capacity(n.min(1024));
    for i in (0..n).rev() {
        if npc[i] == 0 {
            unused.push(i as u32);
        }
    }

    let mut order: Vec<u32> = (0..n as u32).collect();
    order.shuffle(rng);

    let stable = &mut ws.stable[..n];
    stable.fill(false);
    let mut n_unstable: usize = n;

    if has_fixed {
        for i in 0..n {
            if fixed_arr[i] {
                stable[i] = true;
                n_unstable -= 1;
            }
        }
    }

    let ewpc = &mut ws.ewpc[..n];
    ewpc.fill(0.0);
    // Take nc_buf out of workspace to avoid borrow issues
    let mut nc_buf = std::mem::take(&mut ws.nc_buf);
    nc_buf.clear();

    let cw_ptr = cw.as_mut_ptr();
    let ewpc_ptr = ewpc.as_mut_ptr();

    let mut i: usize = 0;
    let mut consecutive_skips: usize = 0;
    while n_unstable > 0 {
        let j = unsafe { *order.get_unchecked(i) } as usize;

        if unsafe { *stable.get_unchecked(j) } {
            consecutive_skips += 1;
            if consecutive_skips >= n {
                // Unstable nodes lost from circular buffer (n_unstable overflowed n).
                // Recovery: scan all nodes to find the missing unstable ones.
                let mut recovered = 0;
                for node in 0..n {
                    if !stable[node] {
                        let slot = (i + 1 + recovered) % n;
                        order[slot] = node as u32;
                        recovered += 1;
                    }
                }
                consecutive_skips = 0;
                if recovered == 0 {
                    // n_unstable was stale — correct and exit
                    break;
                }
            }
            i += 1;
            if i >= n { i = 0; }
            continue;
        }
        consecutive_skips = 0;

        let cur_cl = unsafe { *clusters.add(j) };
        let node_w = unsafe { *nw_arr.get_unchecked(j) };

        unsafe {
            *cw_ptr.add(cur_cl) -= node_w;
            *npc.get_unchecked_mut(cur_cl) -= 1;
        }
        if npc[cur_cl] == 0 {
            unused.push(cur_cl as u32);
        }

        nc_buf.clear();
        if let Some(&empty) = unused.last() {
            nc_buf.push(empty);
        }

        let nbr_start = unsafe { *first_nbr.add(j) } as usize;
        let nbr_end = unsafe { *first_nbr.add(j + 1) } as usize;

        // Hot inner loop — unchecked for maximum throughput
        unsafe {
            for k in nbr_start..nbr_end {
                let nbr_node = *nbr_arr.add(k) as usize;
                let nbr_cl = *clusters.add(nbr_node);
                if *ewpc_ptr.add(nbr_cl) == 0.0 {
                    nc_buf.push(nbr_cl as u32);
                }
                *ewpc_ptr.add(nbr_cl) += *ew_arr.add(k);
            }
            *ewpc_ptr.add(cur_cl) += *slw_arr.get_unchecked(j);
        }

        let mut best_cl = cur_cl;
        let mut max_inc = unsafe {
            *ewpc_ptr.add(cur_cl) - node_w * *cw_ptr.add(cur_cl) * resolution
        };

        for idx in 0..nc_buf.len() {
            let nc = unsafe { *nc_buf.get_unchecked(idx) } as usize;
            let inc = unsafe {
                *ewpc_ptr.add(nc) - node_w * *cw_ptr.add(nc) * resolution
            };
            if inc > max_inc {
                best_cl = nc;
                max_inc = inc;
            }
            unsafe { *ewpc_ptr.add(nc) = 0.0; }
        }
        unsafe { *ewpc_ptr.add(cur_cl) = 0.0; }

        unsafe {
            *cw_ptr.add(best_cl) += node_w;
            *npc.get_unchecked_mut(best_cl) += 1;
        }
        if best_cl as u32 == unused.last().copied().unwrap_or(u32::MAX) {
            unused.pop();
        }

        stable[j] = true;
        n_unstable -= 1;

        if best_cl != cur_cl {
            unsafe { *clusters.add(j) = best_cl; }
            if best_cl >= clustering.n_clusters {
                clustering.n_clusters = best_cl + 1;
            }

            unsafe {
                for k in nbr_start..nbr_end {
                    let nbr = *nbr_arr.add(k) as usize;
                    if *stable.get_unchecked(nbr) && *clusters.add(nbr) != best_cl {
                        if has_fixed && *fixed_arr.get_unchecked(nbr) {
                            continue;
                        }
                        *stable.get_unchecked_mut(nbr) = false;
                        n_unstable += 1;
                        let slot = (i + n_unstable) % n;
                        *order.get_unchecked_mut(slot) = nbr as u32;
                    }
                }
            }

            update = true;
        }

        i += 1;
        if i >= n { i = 0; }
    }

    if update {
        clustering.remove_empty_clusters();
    }

    // Return nc_buf to workspace
    ws.nc_buf = nc_buf;

    update
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn test_two_cliques() {
        let g = Graph::from_edge_list(
            6,
            &[0, 1, 2, 3, 4, 5, 2],
            &[1, 2, 0, 4, 5, 3, 3],
            &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.01],
        );
        let mut c = Clustering::singleton(6);
        let mut rng = StdRng::seed_from_u64(42);
        let mut ws = Workspace::new(6);
        let improved = improve_clustering(&g, &mut c, 0.5, &mut rng, &mut ws);
        assert!(improved);
        assert!(c.n_clusters <= 3);
        assert_eq!(c.clusters[0], c.clusters[1]);
        assert_eq!(c.clusters[1], c.clusters[2]);
        assert_eq!(c.clusters[3], c.clusters[4]);
        assert_eq!(c.clusters[4], c.clusters[5]);
    }

    #[test]
    fn test_fixed_nodes() {
        let g = Graph::from_edge_list(
            3,
            &[0, 1, 2],
            &[1, 2, 0],
            &[1.0, 1.0, 1.0],
        );
        let mut c = Clustering::singleton(3);
        c.set_fixed(vec![true, false, false]);
        let mut rng = StdRng::seed_from_u64(42);
        let mut ws = Workspace::new(3);
        improve_clustering(&g, &mut c, 0.1, &mut rng, &mut ws);
        assert_ne!(c.clusters[0], c.clusters[1]);
    }
}
