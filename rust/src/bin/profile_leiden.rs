//! Profiling binary: generates a random k-NN graph and runs Leiden.
use rand::SeedableRng;
use sciscape_leiden::*;
use std::time::Instant;

fn make_knn_graph(n: usize, k: usize) -> Graph {
    use rand::Rng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let mut src = Vec::new();
    let mut dst = Vec::new();
    let mut w = Vec::new();

    for i in 0..n {
        for _ in 0..k {
            let j = rng.gen_range(0..n);
            if j > i {
                src.push(i as u32);
                dst.push(j as u32);
                w.push(rng.gen::<f64>());
            }
        }
    }

    Graph::from_edge_list(n, &src, &dst, &w)
}

fn main() {
    let n = std::env::args()
        .nth(1)
        .unwrap_or("50000".into())
        .parse::<usize>()
        .unwrap();
    eprintln!("Building graph: {} nodes", n);
    let t0 = Instant::now();
    let graph = make_knn_graph(n, 30);
    eprintln!(
        "Graph: {} nodes, {} edges ({:.1}s)",
        graph.n_nodes,
        graph.n_edges,
        t0.elapsed().as_secs_f64()
    );

    let config = LeidenConfig {
        resolution: 0.001,
        n_iterations: 10,
        randomness: 0.01,
        randomness_schedule: Vec::new(),
        seed: 42,
    };

    eprintln!("Running Leiden...");
    let t1 = Instant::now();
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let result = leiden(&graph, &config, None, &mut rng);
    let elapsed = t1.elapsed().as_secs_f64();

    eprintln!(
        "Done: {} clusters, Q={:.2}, {:.2}s",
        result.clustering.n_clusters, result.quality, elapsed
    );
}
