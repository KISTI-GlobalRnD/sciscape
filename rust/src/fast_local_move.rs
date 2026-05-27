//! Fast local moving algorithm for the move phase of Leiden.
//!
//! Port of CWTS FastLocalMovingAlgorithm.java.
//! Hot-path: unsafe unchecked indexing where bounds are guaranteed.

use crate::clustering::Clustering;
use crate::graph::Graph;
use crate::random_utils::{fill_identity_u32, permute_cwts_style};
use crate::trace;
use crate::workspace::Workspace;
use rand::Rng;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LocalMoveStats {
    pub improved: bool,
    pub moved_nodes: usize,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct LocalMoveTraceContext {
    pub depth: usize,
    pub iteration: usize,
}

#[derive(Clone, Copy, Debug)]
struct LocalMoveMarginEvent {
    node: usize,
    current_cluster: usize,
    best_cluster: usize,
    second_cluster: usize,
    best_increment: f64,
    second_increment: f64,
    margin: f64,
    moved: bool,
}

#[derive(Clone, Copy, Debug)]
struct LocalMoveFocusEvent {
    node: usize,
    role: &'static str,
    current_cluster: usize,
    best_cluster: usize,
    second_cluster: Option<usize>,
    best_increment: f64,
    second_increment: f64,
    margin: f64,
    moved: bool,
}

/// Run one iteration of fast local moving.
/// `ws` is a reusable workspace (avoids allocation per call).
pub fn improve_clustering(
    graph: &Graph,
    clustering: &mut Clustering,
    resolution: f64,
    rng: &mut impl Rng,
    ws: &mut Workspace,
) -> LocalMoveStats {
    improve_clustering_with_trace(graph, clustering, resolution, rng, ws, None)
}

pub(crate) fn improve_clustering_with_trace(
    graph: &Graph,
    clustering: &mut Clustering,
    resolution: f64,
    rng: &mut impl Rng,
    ws: &mut Workspace,
    trace_context: Option<LocalMoveTraceContext>,
) -> LocalMoveStats {
    let n = graph.n_nodes;
    if n <= 1 {
        return LocalMoveStats::default();
    }

    let mut moved_nodes = 0usize;
    let trajectory_trace_enabled = trace_context.is_some() && trace::ddm_trajectory_trace_enabled();
    let collect_margins = trajectory_trace_enabled;
    let focus_nodes = if trajectory_trace_enabled {
        trace::ddm_local_move_focus_nodes()
    } else {
        None
    };
    let mut margin_events: Vec<LocalMoveMarginEvent> = Vec::new();

    let first_nbr = graph.first_neighbor_index.as_ptr();
    let nbr_arr = graph.neighbors.as_ptr();
    let ew_arr = graph.edge_weights.as_ptr();
    let nw_arr = graph.node_weights.as_slice();
    let clusters = clustering.clusters.as_mut_ptr();
    let has_fixed = clustering.fixed.is_some();
    let fixed_arr: &[bool] = clustering.fixed.as_deref().unwrap_or(&[]);

    // Reuse workspace arrays (already zeroed by caller or reset)
    ws.ensure_capacity(n);
    let stable_epoch = ws.next_stable_epoch(n);
    let cw = &mut ws.cw[..n];
    let npc = &mut ws.npc[..n];
    let initial_n_clusters = clustering.n_clusters.min(n);
    cw[..initial_n_clusters].fill(0.0);
    npc[..initial_n_clusters].fill(0);
    for i in 0..n {
        unsafe {
            let cid = *clusters.add(i) as usize;
            *cw.get_unchecked_mut(cid) += nw_arr.get_unchecked(i);
            *npc.get_unchecked_mut(cid) += 1;
        }
    }

    let mut unused = std::mem::take(&mut ws.unused);
    unused.clear();
    for i in (0..initial_n_clusters).rev() {
        if npc[i] == 0 {
            unused.push(i as u32);
        }
    }
    // Cluster ids above the current cluster count are all empty. Keep that
    // range lazy instead of pushing up to O(n) ids into `unused` each call.
    let mut next_tail_empty = initial_n_clusters;

    let mut order = std::mem::take(&mut ws.order);
    fill_identity_u32(&mut order, n);
    permute_cwts_style(&mut order, rng);

    let stable = &mut ws.stable[..n];
    let mut n_unstable: usize = n;

    if has_fixed {
        for i in 0..n {
            if fixed_arr[i] {
                stable[i] = stable_epoch;
                n_unstable -= 1;
            }
        }
    }

    // `ewpc` is kept zeroed by clearing every touched cluster after each node.
    // Avoid a full f64 memset on every local-move call.
    let ewpc = &mut ws.ewpc[..n];
    // Take nc_buf out of workspace to avoid borrow issues
    let mut nc_buf = std::mem::take(&mut ws.nc_buf);
    nc_buf.clear();

    let cw_ptr = cw.as_mut_ptr();
    let ewpc_ptr = ewpc.as_mut_ptr();

    let mut i: usize = 0;
    let mut consecutive_skips: usize = 0;
    while n_unstable > 0 {
        let j = unsafe { *order.get_unchecked(i) } as usize;

        if unsafe { *stable.get_unchecked(j) == stable_epoch } {
            consecutive_skips += 1;
            if consecutive_skips >= n {
                // Unstable nodes lost from circular buffer (n_unstable overflowed n).
                // Recovery: scan all nodes to find the missing unstable ones.
                let mut recovered = 0;
                for node in 0..n {
                    if stable[node] != stable_epoch {
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
            if i >= n {
                i = 0;
            }
            continue;
        }
        consecutive_skips = 0;

        let cur_cl = unsafe { *clusters.add(j) } as usize;
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
        } else if next_tail_empty < n {
            // Cluster IDs above the current cluster count are treated lazily.
            // Zero the next tail candidate immediately before it can influence
            // the quality calculation; this avoids clearing cw/npc for all
            // graph.n_nodes on every local-move pass.
            unsafe {
                *cw_ptr.add(next_tail_empty) = 0.0;
                *npc.get_unchecked_mut(next_tail_empty) = 0;
            }
            nc_buf.push(next_tail_empty as u32);
        }

        let nbr_start = unsafe { *first_nbr.add(j) } as usize;
        let nbr_end = unsafe { *first_nbr.add(j + 1) } as usize;

        // Hot inner loop — unchecked for maximum throughput
        unsafe {
            for k in nbr_start..nbr_end {
                let nbr_node = *nbr_arr.add(k) as usize;
                let nbr_cl = *clusters.add(nbr_node) as usize;
                if *ewpc_ptr.add(nbr_cl) == 0.0 {
                    nc_buf.push(nbr_cl as u32);
                }
                *ewpc_ptr.add(nbr_cl) += *ew_arr.add(k);
            }
        }

        let mut best_cl = cur_cl;
        let mut max_inc =
            unsafe { *ewpc_ptr.add(cur_cl) - node_w * *cw_ptr.add(cur_cl) * resolution };
        let mut second_best: Option<(usize, f64)> = None;

        for idx in 0..nc_buf.len() {
            let nc = unsafe { *nc_buf.get_unchecked(idx) } as usize;
            let inc = unsafe { *ewpc_ptr.add(nc) - node_w * *cw_ptr.add(nc) * resolution };
            if inc > max_inc {
                second_best = Some((best_cl, max_inc));
                best_cl = nc;
                max_inc = inc;
            } else if nc != best_cl {
                match second_best {
                    Some((_, second_inc)) if inc <= second_inc => {}
                    _ => second_best = Some((nc, inc)),
                }
            }
            unsafe {
                *ewpc_ptr.add(nc) = 0.0;
            }
        }
        unsafe {
            *ewpc_ptr.add(cur_cl) = 0.0;
        }
        if collect_margins {
            if let Some((second_cluster, second_increment)) = second_best {
                push_local_move_margin_event(
                    &mut margin_events,
                    LocalMoveMarginEvent {
                        node: j,
                        current_cluster: cur_cl,
                        best_cluster: best_cl,
                        second_cluster,
                        best_increment: max_inc,
                        second_increment,
                        margin: max_inc - second_increment,
                        moved: best_cl != cur_cl,
                    },
                );
            }
        }
        if let (Some(context), Some(focus_nodes)) = (trace_context, focus_nodes.as_ref()) {
            if let Some(role) = focus_nodes.role_for(j) {
                let second_cluster = second_best.map(|(cluster, _)| cluster);
                let second_increment = second_best
                    .map(|(_, increment)| increment)
                    .unwrap_or(f64::NAN);
                emit_local_move_focus_node(
                    context,
                    LocalMoveFocusEvent {
                        node: j,
                        role,
                        current_cluster: cur_cl,
                        best_cluster: best_cl,
                        second_cluster,
                        best_increment: max_inc,
                        second_increment,
                        margin: max_inc - second_increment,
                        moved: best_cl != cur_cl,
                    },
                );
            }
        }

        unsafe {
            *cw_ptr.add(best_cl) += node_w;
            *npc.get_unchecked_mut(best_cl) += 1;
        }
        if best_cl as u32 == unused.last().copied().unwrap_or(u32::MAX) {
            unused.pop();
        } else if best_cl == next_tail_empty {
            next_tail_empty += 1;
        }

        stable[j] = stable_epoch;
        n_unstable -= 1;

        if best_cl != cur_cl {
            unsafe {
                *clusters.add(j) = best_cl as u32;
            }
            if best_cl >= clustering.n_clusters {
                clustering.n_clusters = best_cl + 1;
            }

            if has_fixed {
                unsafe {
                    for k in nbr_start..nbr_end {
                        let nbr = *nbr_arr.add(k) as usize;
                        if *stable.get_unchecked(nbr) == stable_epoch
                            && *clusters.add(nbr) as usize != best_cl
                            && !*fixed_arr.get_unchecked(nbr)
                        {
                            *stable.get_unchecked_mut(nbr) = 0;
                            n_unstable += 1;
                            let slot = (i + n_unstable) % n;
                            *order.get_unchecked_mut(slot) = nbr as u32;
                        }
                    }
                }
            } else {
                unsafe {
                    for k in nbr_start..nbr_end {
                        let nbr = *nbr_arr.add(k) as usize;
                        if *stable.get_unchecked(nbr) == stable_epoch
                            && *clusters.add(nbr) as usize != best_cl
                        {
                            *stable.get_unchecked_mut(nbr) = 0;
                            n_unstable += 1;
                            let slot = (i + n_unstable) % n;
                            *order.get_unchecked_mut(slot) = nbr as u32;
                        }
                    }
                }
            }

            moved_nodes += 1;
        }

        i += 1;
        if i >= n {
            i = 0;
        }
    }

    if moved_nodes > 0 {
        clustering.compact_from_counts(npc);
    }

    // Return nc_buf to workspace
    ws.nc_buf = nc_buf;
    ws.order = order;
    ws.unused = unused;

    if let Some(context) = trace_context {
        if collect_margins {
            emit_local_move_margin_events(context, &margin_events);
        }
    }

    LocalMoveStats {
        improved: moved_nodes > 0,
        moved_nodes,
    }
}

fn push_local_move_margin_event(
    events: &mut Vec<LocalMoveMarginEvent>,
    event: LocalMoveMarginEvent,
) {
    const TOP_K: usize = 16;
    if !event.margin.is_finite() {
        return;
    }
    events.push(event);
    events.sort_by(|left, right| {
        left.margin
            .total_cmp(&right.margin)
            .then_with(|| left.node.cmp(&right.node))
    });
    if events.len() > TOP_K {
        events.pop();
    }
}

fn emit_local_move_margin_events(context: LocalMoveTraceContext, events: &[LocalMoveMarginEvent]) {
    for (rank, event) in events.iter().enumerate() {
        trace::emit_ddm_trajectory_trace(format_args!(
            "{{\"schema\":\"dongdaemun_trajectory_trace.v1\",\"event\":\"local_move_margin\",\"run_id\":{},\"depth\":{},\"iteration\":{},\"rank\":{},\"node\":{},\"current_cluster\":{},\"best_cluster\":{},\"second_cluster\":{},\"best_increment\":{},\"second_increment\":{},\"margin\":{},\"moved\":{}}}",
            trace::json_string_option(trace::ddm_trajectory_trace_run_id()),
            context.depth,
            context.iteration,
            rank,
            event.node,
            event.current_cluster,
            event.best_cluster,
            event.second_cluster,
            trace::json_f64(event.best_increment),
            trace::json_f64(event.second_increment),
            trace::json_f64(event.margin),
            event.moved,
        ));
    }
}

fn emit_local_move_focus_node(context: LocalMoveTraceContext, event: LocalMoveFocusEvent) {
    trace::emit_ddm_trajectory_trace(format_args!(
        "{{\"schema\":\"dongdaemun_trajectory_trace.v1\",\"event\":\"local_move_focus_node\",\"run_id\":{},\"depth\":{},\"iteration\":{},\"node\":{},\"role\":{},\"current_cluster\":{},\"best_cluster\":{},\"second_cluster\":{},\"best_increment\":{},\"second_increment\":{},\"margin\":{},\"moved\":{}}}",
        trace::json_string_option(trace::ddm_trajectory_trace_run_id()),
        context.depth,
        context.iteration,
        event.node,
        trace::json_string(event.role),
        event.current_cluster,
        event.best_cluster,
        trace::json_usize_option(event.second_cluster),
        trace::json_f64(event.best_increment),
        trace::json_f64(event.second_increment),
        trace::json_f64(event.margin),
        event.moved,
    ));
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

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
        let stats = improve_clustering(&g, &mut c, 0.5, &mut rng, &mut ws);
        assert!(stats.improved);
        assert!(stats.moved_nodes > 0);
        assert!(c.n_clusters <= 3);
        assert_eq!(c.clusters[0], c.clusters[1]);
        assert_eq!(c.clusters[1], c.clusters[2]);
        assert_eq!(c.clusters[3], c.clusters[4]);
        assert_eq!(c.clusters[4], c.clusters[5]);
    }

    #[test]
    fn test_fixed_nodes() {
        let g = Graph::from_edge_list(3, &[0, 1, 2], &[1, 2, 0], &[1.0, 1.0, 1.0]);
        let mut c = Clustering::singleton(3);
        c.set_fixed(vec![true, false, false]);
        let mut rng = StdRng::seed_from_u64(42);
        let mut ws = Workspace::new(3);
        improve_clustering(&g, &mut c, 0.1, &mut rng, &mut ws);
        assert_eq!(c.clusters[0], 0);
    }

    #[test]
    fn test_lazy_tail_empty_clusters_can_split() {
        let g = Graph::from_edge_list(4, &[], &[], &[]);
        let mut c = Clustering::from_assignments(vec![0, 0, 0, 0]);
        let mut rng = StdRng::seed_from_u64(42);
        let mut ws = Workspace::new(4);

        let stats = improve_clustering(&g, &mut c, 1.0, &mut rng, &mut ws);

        assert!(stats.improved);
        assert!(stats.moved_nodes > 0);
        assert!(c.n_clusters > 1);
    }

    #[test]
    fn test_self_loops_do_not_block_moves() {
        let g = Graph {
            n_nodes: 2,
            n_edges: 2,
            first_neighbor_index: vec![0, 1, 2],
            neighbors: vec![1, 0],
            edge_weights: vec![10.0, 10.0],
            node_weights: vec![1.0, 1.0],
            self_loop_weights: vec![1_000.0, 1_000.0],
        };
        let mut c = Clustering::singleton(2);
        let mut rng = StdRng::seed_from_u64(42);
        let mut ws = Workspace::new(2);

        let stats = improve_clustering(&g, &mut c, 0.1, &mut rng, &mut ws);

        assert!(stats.improved);
        assert!(stats.moved_nodes > 0);
        assert_eq!(c.n_clusters, 1);
    }

    #[test]
    fn test_stable_epoch_wrap_preserves_behavior() {
        let g = Graph::from_edge_list(3, &[0, 1, 2], &[1, 2, 0], &[1.0, 1.0, 1.0]);
        let mut rng = StdRng::seed_from_u64(42);
        let mut ws = Workspace::new(3);
        ws.stable_epoch = u8::MAX;

        let mut c = Clustering::singleton(3);
        c.set_fixed(vec![true, false, false]);

        improve_clustering(&g, &mut c, 0.1, &mut rng, &mut ws);

        assert_eq!(c.clusters[0], 0);
        assert_eq!(ws.stable_epoch, 1);
    }
}
