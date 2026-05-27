use sciscape_leiden::dongdaemun::{dongdaemun_refine, DongdaemunConfig, DongdaemunStatus};
use sciscape_leiden::workspace::Workspace;
use sciscape_leiden::{Clustering, Graph};

fn empty_weighted_graph(node_weights: &[f64]) -> Graph {
    Graph::from_edge_list_weighted(node_weights.len(), &[], &[], &[], node_weights)
}

fn assert_effective_equals_baseline(
    result: &sciscape_leiden::dongdaemun::DongdaemunResult,
    baseline: &Clustering,
) {
    assert_eq!(result.clustering.clusters, baseline.clusters);
    assert_eq!(result.clustering.n_clusters, baseline.n_clusters);
}

#[test]
fn dongdaemun_no_oversize_returns_baseline() {
    let graph = empty_weighted_graph(&[2.0, 3.0, 4.0]);
    let baseline = Clustering::from_assignments(vec![0, 1, 2]);
    let config = DongdaemunConfig::default_for_quality_first(0.5, 4.0);
    let mut ws = Workspace::new(graph.n_nodes);

    let result = dongdaemun_refine(&graph, &baseline, &config, &mut ws);

    assert_effective_equals_baseline(&result, &baseline);
    assert!(result.diagnostic_clustering.is_none());
    assert!(result.audit.accepted);
    assert_eq!(
        result.audit.status,
        DongdaemunStatus::NoCurrentOversizeCandidates
    );
    assert_eq!(result.audit.final_delta_q, 0.0);
    assert_eq!(result.audit.n_oversize_before, 0);
    assert_eq!(result.audit.n_oversize_after_candidate, 0);
    assert_eq!(result.audit.max_weight_before, 4.0);
    assert_eq!(result.audit.max_weight_after_candidate, 4.0);
    assert!(result.audit.target_max_satisfied);
}

#[test]
fn dongdaemun_oversize_with_no_trim_moves_returns_baseline_fallback() {
    let graph = empty_weighted_graph(&[1.0, 1.0, 1.0, 1.0]);
    let baseline = Clustering::from_assignments(vec![0, 0, 0, 0]);
    let mut config = DongdaemunConfig::default_for_quality_first(0.5, 2.0);
    config.gamma_multipliers = Vec::new();
    let mut ws = Workspace::new(graph.n_nodes);

    let result = dongdaemun_refine(&graph, &baseline, &config, &mut ws);

    assert_effective_equals_baseline(&result, &baseline);
    assert!(result.diagnostic_clustering.is_none());
    assert!(!result.audit.accepted);
    assert_eq!(result.audit.status, DongdaemunStatus::NoSelectedCandidates);
    assert_eq!(result.audit.final_delta_q, 0.0);
    assert_eq!(result.audit.n_oversize_before, 1);
    assert_eq!(result.audit.n_oversize_after_candidate, 1);
    assert_eq!(result.audit.max_weight_before, 4.0);
    assert_eq!(result.audit.max_weight_after_candidate, 4.0);
    assert!(!result.audit.target_max_satisfied);
}

#[test]
fn dongdaemun_weighted_graph_sets_public_oversize_audit_fields() {
    let graph = empty_weighted_graph(&[2.5, 7.0, 3.0, 4.0]);
    let baseline = Clustering::from_assignments(vec![0, 1, 2, 2]);
    let mut config = DongdaemunConfig::default_for_hard_cap(0.2, 6.0);
    config.gamma_multipliers = Vec::new();
    let mut ws = Workspace::new(graph.n_nodes);

    let result = dongdaemun_refine(&graph, &baseline, &config, &mut ws);

    assert_effective_equals_baseline(&result, &baseline);
    assert_eq!(result.audit.status, DongdaemunStatus::NoSelectedCandidates);
    assert_eq!(result.audit.n_oversize_before, 2);
    assert_eq!(result.audit.n_oversize_after_candidate, 2);
    assert_eq!(result.audit.max_weight_before, 7.0);
    assert_eq!(result.audit.max_weight_after_candidate, 7.0);
    assert!(!result.audit.target_max_satisfied);
}

#[test]
fn dongdaemun_quality_first_commits_positive_boundary_trim() {
    let graph = Graph::from_edge_list(4, &[0, 1, 2], &[1, 2, 3], &[3.0, 0.1, 4.0]);
    let baseline = Clustering::from_assignments(vec![0, 0, 0, 1]);
    let mut config = DongdaemunConfig::default_for_quality_first(0.1, 2.0);
    config.gamma_multipliers = Vec::new();
    let mut ws = Workspace::new(graph.n_nodes);

    let result = dongdaemun_refine(&graph, &baseline, &config, &mut ws);

    assert_eq!(result.clustering.clusters, vec![0, 0, 1, 1]);
    assert!(result.diagnostic_clustering.is_none());
    assert!(result.audit.accepted);
    assert_eq!(result.audit.status, DongdaemunStatus::Committed);
    assert_eq!(result.audit.trim_moves_proposed, 1);
    assert_eq!(result.audit.trim_moves_committed, 1);
    assert_eq!(result.audit.candidate_delta_q, result.audit.final_delta_q);
    assert_eq!(result.audit.effective_delta_q, result.audit.final_delta_q);
    assert!(result.audit.final_delta_q > 0.0);
    assert!(result.audit.target_max_satisfied);
    assert_eq!(result.audit.n_oversize_before, 1);
    assert_eq!(result.audit.n_oversize_after_candidate, 0);
}

#[test]
fn dongdaemun_split_repair_commits_escaped_fragment() {
    let graph = Graph::from_edge_list(3, &[0, 1], &[1, 2], &[0.1, 10.0]);
    let baseline = Clustering::from_assignments(vec![0, 0, 1]);
    let mut config = DongdaemunConfig::default_for_quality_first(0.1, 1.5);
    config.apply_iterations = 1;
    config.gamma_multipliers = vec![10.0];
    config.min_core_weight = 1.0;
    config.randomness = 0.0;
    config.repair_epsilon = 0.0;
    config.pair_seeded = false;
    let mut ws = Workspace::new(graph.n_nodes);

    let result = dongdaemun_refine(&graph, &baseline, &config, &mut ws);

    assert_eq!(result.clustering.clusters, vec![0, 1, 1]);
    assert_eq!(result.clustering.n_clusters, 2);
    assert!(result.diagnostic_clustering.is_none());
    assert!(result.audit.accepted);
    assert_eq!(result.audit.status, DongdaemunStatus::Committed);
    assert_eq!(result.audit.split_iterations.len(), 1);
    assert_eq!(result.audit.split_iterations[0].n_selected, 1);
    assert_eq!(result.audit.split_iterations[0].n_applied, 1);
    assert_eq!(
        result.audit.split_iterations[0].status,
        DongdaemunStatus::Committed
    );
    assert!(result.audit.split_iterations[0].exact_delta_q > 0.0);
    assert_eq!(result.audit.trim_moves_proposed, 0);
    assert_eq!(result.audit.trim_moves_committed, 0);
    assert!(result.audit.candidate_delta_q > 0.0);
    assert_eq!(
        result.audit.effective_delta_q,
        result.audit.candidate_delta_q
    );
}

#[test]
fn dongdaemun_split_loop_records_terminal_second_iteration() {
    let graph = Graph::from_edge_list(3, &[0, 1], &[1, 2], &[0.1, 10.0]);
    let baseline = Clustering::from_assignments(vec![0, 0, 1]);
    let mut config = DongdaemunConfig::default_for_quality_first(0.1, 1.5);
    config.apply_iterations = 4;
    config.gamma_multipliers = vec![10.0];
    config.min_core_weight = 1.0;
    config.randomness = 0.0;
    config.repair_epsilon = 0.0;
    config.pair_seeded = false;
    let mut ws = Workspace::new(graph.n_nodes);

    let result = dongdaemun_refine(&graph, &baseline, &config, &mut ws);

    assert_eq!(result.clustering.clusters, vec![0, 1, 1]);
    assert!(result.audit.accepted);
    assert_eq!(result.audit.status, DongdaemunStatus::Committed);
    assert_eq!(result.audit.split_iterations.len(), 2);
    assert_eq!(
        result.audit.split_iterations[0].status,
        DongdaemunStatus::Committed
    );
    assert_eq!(
        result.audit.split_iterations[1].status,
        DongdaemunStatus::NoSelectedCandidates
    );
    assert_eq!(result.audit.split_iterations[1].n_selected, 0);
}

#[test]
fn dongdaemun_pair_seeded_split_is_deterministic() {
    let graph = Graph::from_edge_list(
        5,
        &[0, 1, 1, 2, 3],
        &[1, 2, 3, 3, 4],
        &[0.1, 10.0, 0.5, 0.2, 8.0],
    );
    let baseline = Clustering::from_assignments(vec![0, 0, 0, 1, 1]);
    let mut config = DongdaemunConfig::default_for_quality_first(0.1, 2.5);
    config.gamma_multipliers = vec![1.05, 10.0];
    config.min_core_weight = 1.0;
    config.randomness = 0.01;
    config.repair_epsilon = 0.0;
    config.seed = 42;
    config.pair_seeded = true;
    let mut left_ws = Workspace::new(graph.n_nodes);
    let mut right_ws = Workspace::new(graph.n_nodes);

    let left = dongdaemun_refine(&graph, &baseline, &config, &mut left_ws);
    let right = dongdaemun_refine(&graph, &baseline, &config, &mut right_ws);

    assert_eq!(left.clustering.clusters, right.clustering.clusters);
    assert_eq!(left.audit.status, right.audit.status);
    assert_eq!(
        left.audit.split_iterations.len(),
        right.audit.split_iterations.len()
    );
    assert_eq!(left.audit.split_iterations.len(), 2);
    for (left_iteration, right_iteration) in left
        .audit
        .split_iterations
        .iter()
        .zip(right.audit.split_iterations.iter())
    {
        assert_eq!(left_iteration.status, right_iteration.status);
        assert!((left_iteration.exact_delta_q - right_iteration.exact_delta_q).abs() < 1e-12);
    }
}

#[test]
fn dongdaemun_split_below_quality_floor_falls_back_with_diagnostic() {
    let graph = Graph::from_edge_list(3, &[0, 1], &[1, 2], &[0.1, 10.0]);
    let baseline = Clustering::from_assignments(vec![0, 0, 1]);
    let mut config = DongdaemunConfig::default_for_quality_first(0.1, 1.5);
    config.gamma_multipliers = vec![10.0];
    config.min_core_weight = 1.0;
    config.randomness = 0.0;
    config.repair_epsilon = 0.0;
    config.pair_seeded = false;
    config.quality_floor_delta = 100.0;
    config.trim_min_delta_q_quality_first = 100.0;
    let mut ws = Workspace::new(graph.n_nodes);

    let result = dongdaemun_refine(&graph, &baseline, &config, &mut ws);

    assert_effective_equals_baseline(&result, &baseline);
    assert_eq!(
        result
            .diagnostic_clustering
            .as_ref()
            .map(|c| c.clusters.as_slice()),
        Some([0, 1, 1].as_slice())
    );
    assert!(!result.audit.accepted);
    assert_eq!(
        result.audit.status,
        DongdaemunStatus::SplitQualityBelowFloor
    );
    assert_eq!(result.audit.split_iterations.len(), 1);
    assert_eq!(
        result.audit.split_iterations[0].status,
        DongdaemunStatus::SplitQualityBelowFloor
    );
    assert_eq!(result.audit.effective_delta_q, 0.0);
    assert!(result.audit.candidate_delta_q > 0.0);
    assert!(result.audit.candidate_delta_q < config.quality_floor_delta);
}

#[test]
fn dongdaemun_hard_cap_rejects_split_when_cap_remains_unsatisfied() {
    let graph = Graph::from_edge_list(3, &[0, 1], &[1, 2], &[0.1, 10.0]);
    let baseline = Clustering::from_assignments(vec![0, 0, 1]);
    let mut config = DongdaemunConfig::default_for_hard_cap(0.1, 1.5);
    config.apply_iterations = 1;
    config.gamma_multipliers = vec![10.0];
    config.min_core_weight = 1.0;
    config.randomness = 0.0;
    config.repair_epsilon = 0.0;
    config.pair_seeded = false;
    let mut ws = Workspace::new(graph.n_nodes);

    let result = dongdaemun_refine(&graph, &baseline, &config, &mut ws);

    assert_effective_equals_baseline(&result, &baseline);
    assert_eq!(
        result
            .diagnostic_clustering
            .as_ref()
            .map(|c| c.clusters.as_slice()),
        Some([0, 1, 1].as_slice())
    );
    assert!(!result.audit.accepted);
    assert_eq!(result.audit.status, DongdaemunStatus::HardCapNotSatisfied);
    assert_eq!(result.audit.split_iterations.len(), 1);
    assert_eq!(
        result.audit.split_iterations[0].status,
        DongdaemunStatus::Committed
    );
    assert_eq!(result.audit.trim_moves_proposed, 0);
    assert_eq!(result.audit.trim_moves_committed, 0);
    assert!(result.audit.candidate_delta_q > 0.0);
    assert_eq!(result.audit.effective_delta_q, 0.0);
    assert!(!result.audit.target_max_satisfied);
}

#[test]
fn dongdaemun_hard_cap_falls_back_when_trim_leaves_cap_unsatisfied() {
    let graph = Graph::from_edge_list(5, &[0, 3], &[3, 4], &[0.5, 0.6]);
    let baseline = Clustering::from_assignments(vec![0, 0, 0, 0, 1]);
    let mut config = DongdaemunConfig::default_for_hard_cap(0.1, 2.0);
    config.gamma_multipliers = Vec::new();
    config.trim_max_moves_per_cluster = 1;
    let mut ws = Workspace::new(graph.n_nodes);

    let result = dongdaemun_refine(&graph, &baseline, &config, &mut ws);

    assert_effective_equals_baseline(&result, &baseline);
    assert_eq!(
        result
            .diagnostic_clustering
            .as_ref()
            .map(|c| c.clusters.as_slice()),
        Some([0, 0, 0, 1, 1].as_slice())
    );
    assert!(!result.audit.accepted);
    assert_eq!(result.audit.status, DongdaemunStatus::HardCapNotSatisfied);
    assert_eq!(result.audit.trim_moves_proposed, 1);
    assert_eq!(result.audit.trim_moves_committed, 1);
    assert!(result.audit.candidate_delta_q > 0.0);
    assert_eq!(result.audit.effective_delta_q, 0.0);
    assert_eq!(result.audit.final_delta_q, result.audit.candidate_delta_q);
    assert!(!result.audit.target_max_satisfied);
}

#[test]
fn dongdaemun_hard_cap_rolls_back_trim_below_quality_floor() {
    let graph = Graph::from_edge_list(5, &[0, 3], &[3, 4], &[0.5, 0.1]);
    let baseline = Clustering::from_assignments(vec![0, 0, 0, 0, 1]);
    let mut config = DongdaemunConfig::default_for_hard_cap(0.1, 2.0);
    config.gamma_multipliers = Vec::new();
    config.trim_max_moves_per_cluster = 1;
    let mut ws = Workspace::new(graph.n_nodes);

    let result = dongdaemun_refine(&graph, &baseline, &config, &mut ws);

    assert_effective_equals_baseline(&result, &baseline);
    assert_eq!(
        result
            .diagnostic_clustering
            .as_ref()
            .map(|c| c.clusters.as_slice()),
        Some([0, 0, 0, 1, 1].as_slice())
    );
    assert!(!result.audit.accepted);
    assert_eq!(result.audit.status, DongdaemunStatus::TrimQualityBelowFloor);
    assert_eq!(result.audit.trim_moves_proposed, 1);
    assert_eq!(result.audit.trim_moves_committed, 0);
    assert!(result.audit.candidate_delta_q < 0.0);
    assert_eq!(result.audit.effective_delta_q, 0.0);
    assert_eq!(result.audit.final_delta_q, result.audit.candidate_delta_q);
    assert!(!result.audit.target_max_satisfied);
}
