/// Triadic closure edge reweighting.
///
/// Reweights edges by: w'(i,j) = w(i,j) * (1 + |common_neighbors(i,j)|)
///
/// Edges embedded in triangles (triadic closure) get boosted,
/// while isolated cross-community edges remain at original weight.
/// This mitigates singleton noise in the HAC by preferring structurally
/// supported connections.

use crate::graph::{HalfEdge, SparseGraph};

/// Count common neighbors for each edge and return a new graph with
/// reweighted edges.
///
/// Uses sorted neighbor intersection for O(deg) per edge.
/// Total: O(Σ_e min(deg(u), deg(v))) ≈ O(m·√m) for sparse graphs.
pub fn reweight_triadic(graph: &SparseGraph) -> SparseGraph {
    let n = graph.n;
    let mut new_edges = graph.edges.clone();
    let offsets = &graph.offsets;

    // For each vertex, iterate over its neighbors
    for u in 0..n {
        let u_start = offsets[u];
        let u_end = offsets[u + 1];

        for idx in u_start..u_end {
            let v = graph.edges[idx].target as usize;
            if u >= v {
                continue; // process each undirected edge once
            }

            // Count common neighbors using sorted intersection
            let cn = sorted_intersection_count(
                &graph.edges[offsets[u]..offsets[u + 1]],
                &graph.edges[offsets[v]..offsets[v + 1]],
            );

            let multiplier = 1.0 + cn as f64;

            // Update both directions
            new_edges[idx].weight *= multiplier;

            // Find reverse edge v→u
            let v_start = offsets[v];
            let v_end = offsets[v + 1];
            for ridx in v_start..v_end {
                if graph.edges[ridx].target == u as u32 {
                    new_edges[ridx].weight *= multiplier;
                    break;
                }
            }
        }
    }

    SparseGraph {
        n,
        offsets: offsets.clone(),
        edges: new_edges,
    }
}

/// Count the number of common targets in two sorted neighbor lists.
fn sorted_intersection_count(a: &[HalfEdge], b: &[HalfEdge]) -> usize {
    let mut count = 0;
    let mut i = 0;
    let mut j = 0;
    while i < a.len() && j < b.len() {
        match a[i].target.cmp(&b[j].target) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                count += 1;
                i += 1;
                j += 1;
            }
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::SparseGraph;

    #[test]
    fn test_triangle_boost() {
        // Triangle: 0-1, 1-2, 0-2 (all weight 1.0)
        // Each edge has 1 common neighbor → multiplier = 2.0
        let g = SparseGraph::from_edges(3, &[0, 1, 0], &[1, 2, 2], &[1.0, 1.0, 1.0]);
        let g2 = reweight_triadic(&g);

        // All edges should have weight 2.0
        for he in &g2.edges {
            assert!(
                (he.weight - 2.0).abs() < 1e-10,
                "Expected 2.0, got {}",
                he.weight
            );
        }
    }

    #[test]
    fn test_no_triangles() {
        // Path: 0-1-2 (no triangles)
        let g = SparseGraph::from_edges(3, &[0, 1], &[1, 2], &[1.0, 1.0]);
        let g2 = reweight_triadic(&g);

        // No common neighbors → weights unchanged
        for he in &g2.edges {
            assert!(
                (he.weight - 1.0).abs() < 1e-10,
                "Expected 1.0, got {}",
                he.weight
            );
        }
    }

    #[test]
    fn test_mixed() {
        // 0-1-2-0 (triangle) + 2-3 (pendant, no triangle)
        let g = SparseGraph::from_edges(
            4,
            &[0, 1, 0, 2],
            &[1, 2, 2, 3],
            &[1.0, 1.0, 1.0, 1.0],
        );
        let g2 = reweight_triadic(&g);

        // Edge 2-3: no common neighbors → weight 1.0
        // Find edge 2→3
        for he in g2.neighbors(2) {
            if he.target == 3 {
                assert!((he.weight - 1.0).abs() < 1e-10);
            }
        }

        // Edge 0-1: common neighbor = 2 → weight 2.0
        for he in g2.neighbors(0) {
            if he.target == 1 {
                assert!((he.weight - 2.0).abs() < 1e-10);
            }
        }
    }
}
