//! Integration tests — verify Leiden end-to-end on non-trivial graphs.

use rand::SeedableRng;
use sciscape_leiden::*;

/// Build a "barbell" graph: two cliques connected by a bridge.
fn barbell(clique_size: usize, bridge_weight: f64) -> Graph {
    let n = clique_size * 2;
    let mut src = Vec::new();
    let mut dst = Vec::new();
    let mut w = Vec::new();

    // Clique A: nodes 0..clique_size
    for i in 0..clique_size {
        for j in (i + 1)..clique_size {
            src.push(i as u32);
            dst.push(j as u32);
            w.push(1.0);
        }
    }

    // Clique B: nodes clique_size..2*clique_size
    for i in clique_size..n {
        for j in (i + 1)..n {
            src.push(i as u32);
            dst.push(j as u32);
            w.push(1.0);
        }
    }

    // Bridge
    src.push((clique_size - 1) as u32);
    dst.push(clique_size as u32);
    w.push(bridge_weight);

    Graph::from_edge_list(n, &src, &dst, &w)
}

#[test]
fn test_barbell_finds_two_clusters() {
    let g = barbell(10, 0.01);
    let config = LeidenConfig {
        resolution: 0.3,
        n_iterations: 10,
        randomness: 0.01,
        seed: 42,
    };
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let result = leiden(&g, &config, None, &mut rng);

    assert_eq!(
        result.clustering.n_clusters, 2,
        "barbell should split into 2 clusters"
    );
    // Clique A: nodes 0..10 in same cluster
    let c0 = result.clustering.clusters[0];
    for i in 0..10 {
        assert_eq!(
            result.clustering.clusters[i], c0,
            "clique A node {} wrong",
            i
        );
    }
    // Clique B: nodes 10..20 in same cluster
    let c1 = result.clustering.clusters[10];
    for i in 10..20 {
        assert_eq!(
            result.clustering.clusters[i], c1,
            "clique B node {} wrong",
            i
        );
    }
    assert_ne!(c0, c1, "two cliques should be in different clusters");
}

#[test]
fn test_quality_is_consistent() {
    let g = barbell(8, 0.1);
    let config = LeidenConfig {
        resolution: 0.5,
        n_iterations: 10,
        randomness: 0.01,
        seed: 0,
    };
    let mut rng = rand::rngs::StdRng::seed_from_u64(0);
    let result = leiden(&g, &config, None, &mut rng);

    // Re-compute quality independently
    let cpm = CPM::new(config.resolution);
    let recomputed = cpm.quality(&g, &result.clustering);

    assert!(
        (result.quality - recomputed).abs() < 1e-10,
        "quality mismatch: {} vs {}",
        result.quality,
        recomputed,
    );
}

#[test]
fn test_quality_positive() {
    let g = barbell(10, 0.01);
    let config = LeidenConfig {
        resolution: 0.3,
        n_iterations: 10,
        randomness: 0.01,
        seed: 42,
    };
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let result = leiden(&g, &config, None, &mut rng);

    assert!(
        result.quality > 0.0,
        "quality should be positive, got {}",
        result.quality
    );
}

#[test]
fn test_quality_better_than_singleton() {
    let g = barbell(10, 0.01);
    let config = LeidenConfig {
        resolution: 0.3,
        n_iterations: 10,
        randomness: 0.01,
        seed: 42,
    };
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let result = leiden(&g, &config, None, &mut rng);

    let cpm = CPM::new(config.resolution);
    let singleton_q = cpm.quality(&g, &Clustering::singleton(g.n_nodes));
    assert!(
        result.quality > singleton_q,
        "Leiden quality {} should beat singletons {}",
        result.quality,
        singleton_q,
    );
}

#[test]
fn test_disconnected_graph() {
    // Two disconnected triangles
    let g = Graph::from_edge_list(
        6,
        &[0, 1, 2, 3, 4, 5],
        &[1, 2, 0, 4, 5, 3],
        &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    );
    let config = LeidenConfig {
        resolution: 0.3,
        n_iterations: 10,
        randomness: 0.01,
        seed: 42,
    };
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let result = leiden(&g, &config, None, &mut rng);

    // Nodes in different components should be in different clusters
    assert_ne!(
        result.clustering.clusters[0], result.clustering.clusters[3],
        "disconnected components should be in different clusters"
    );
}

#[test]
fn test_multi_start_finds_best() {
    let g = barbell(10, 0.01);
    let config = LeidenConfig {
        resolution: 0.3,
        n_iterations: 10,
        randomness: 0.01,
        seed: 0,
    };

    let result_1 = leiden_multi_start(&g, &config, 1, None);
    let result_5 = leiden_multi_start(&g, &config, 5, None);

    assert!(
        result_5.quality >= result_1.quality,
        "multi-start should find equal or better quality: {} vs {}",
        result_5.quality,
        result_1.quality,
    );
}

#[test]
fn test_fixed_nodes_preserved_in_postprocess() {
    let g = barbell(10, 0.5); // stronger bridge
    let config = LeidenConfig {
        resolution: 0.1, // low resolution → wants to merge everything
        n_iterations: 10,
        randomness: 0.01,
        seed: 42,
    };

    // Initial: two clusters
    let mut init =
        Clustering::from_assignments((0..20).map(|i| if i < 10 { 0 } else { 1 }).collect());
    // Fix all nodes in cluster 0
    let mut fixed = vec![false; 20];
    for i in 0..10 {
        fixed[i] = true;
    }
    init.set_fixed(fixed);

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let result = leiden(&g, &config, Some(init), &mut rng);

    // Fixed nodes 0..10 should all stay in the same cluster
    let c0 = result.clustering.clusters[0];
    for i in 0..10 {
        assert_eq!(
            result.clustering.clusters[i], c0,
            "fixed node {} should stay in same cluster",
            i
        );
    }
}

#[test]
fn test_postprocess_small_clusters() {
    // 4-clique (big) + 2 isolated nodes connected weakly
    let mut src = Vec::new();
    let mut dst = Vec::new();
    let mut w = Vec::new();

    // Clique {0,1,2,3}
    for i in 0..4u32 {
        for j in (i + 1)..4 {
            src.push(i);
            dst.push(j);
            w.push(1.0);
        }
    }
    // Node 4 weakly connected to node 3
    src.push(3);
    dst.push(4);
    w.push(0.1);
    // Node 5 weakly connected to node 4
    src.push(4);
    dst.push(5);
    w.push(0.1);

    let g = Graph::from_edge_list(6, &src, &dst, &w);

    // Initial clustering: {0,1,2,3}=cluster0, {4}=cluster1, {5}=cluster2
    let init = Clustering::from_assignments(vec![0, 0, 0, 0, 1, 2]);

    let config = LeidenConfig {
        resolution: 0.1,
        n_iterations: 10,
        randomness: 0.01,
        seed: 42,
    };

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let pp = postprocess_small_clusters(
        &g, &init, &config, 3, 0.0, 5, 0.1, true, false, false, 0.0, true, 0.0, true, &mut rng,
    );

    // Big cluster {0,1,2,3} should remain intact
    let c0 = pp.clustering.clusters[0];
    for i in 0..4 {
        assert_eq!(
            pp.clustering.clusters[i], c0,
            "big cluster node {} changed",
            i
        );
    }
}

#[test]
fn test_contraction_preserves_total_weight() {
    // Contract a graph and verify total edge weight + self-loop weight is preserved
    let g = barbell(5, 0.5);
    let config = LeidenConfig {
        resolution: 0.5,
        n_iterations: 10,
        randomness: 0.01,
        seed: 42,
    };
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let result = leiden(&g, &config, None, &mut rng);

    let mut ws = sciscape_leiden::workspace::Workspace::new(g.n_nodes);
    let reduced =
        sciscape_leiden::contraction::create_reduced_network(&g, &result.clustering, true, &mut ws);

    let original_total = g.total_edge_weight();
    let reduced_total = reduced.total_edge_weight() + reduced.self_loop_weights.iter().sum::<f64>();

    assert!(
        (original_total - reduced_total).abs() < 1e-10,
        "contraction should preserve total weight: {} vs {}",
        original_total,
        reduced_total,
    );
}
