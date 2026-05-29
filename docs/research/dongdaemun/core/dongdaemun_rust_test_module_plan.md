# Dongdaemun Rust Test Module Plan

Status: implementation design before coding.

Target crate:

`rust/` (`sciscape-leiden`)

Do not use `cpm-dendro` for the first Dongdaemun implementation. That crate is
for CPM-density dendrogram construction. Dongdaemun belongs next to the Leiden
backend because it needs `Graph`, `Clustering`, `CPM`, `Workspace`, and the
existing adaptive split/trim primitives.

## Goal

Introduce a small Rust-native Dongdaemun module with tests that lock down the
algorithmic contract before exposing anything to Python.

The first implementation should not try to reproduce the full Python
`hierarchy_postprocess.py` artifact writer. It should implement the in-memory
core:

```text
P_min + G + gamma + T_max + policy -> P_eff + audit
```

where:

- `P_min` is already lower-tail repaired.
- `P_eff` is the effective membership passed forward.
- rejected memberships are diagnostic only.
- acceptance uses exact original-graph CPM accounting.

## Proposed Files

Add:

- `rust/src/dongdaemun.rs`
- `rust/tests/dongdaemun.rs`

Modify:

- `rust/src/lib.rs` to include `pub mod dongdaemun;`

Do not modify Python bindings in the first slice. Add PyO3 exposure only after
the Rust API and tests settle.

## Public API Sketch

```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DongdaemunPolicy {
    QualityFirst,
    HardCap,
}

#[derive(Clone, Debug)]
pub struct DongdaemunConfig {
    pub policy: DongdaemunPolicy,
    pub resolution: f64,
    pub target_max_weight: f64,
    pub quality_floor_delta: f64,
    pub apply_iterations: usize,
    pub gamma_multipliers: Vec<f64>,
    pub min_core_weight: f64,
    pub randomness: f64,
    pub repair_epsilon: f64,
    pub trim_min_delta_q_quality_first: f64,
    pub trim_min_delta_q_hard_cap: f64,
    pub trim_max_moves_per_cluster: usize,
    pub seed: u64,
    pub pair_seeded: bool,
}

#[derive(Clone, Debug)]
pub struct DongdaemunAudit {
    pub accepted: bool,
    pub status: DongdaemunStatus,
    pub quality_before: f64,
    pub quality_after_candidate: f64,
    pub final_delta_q: f64,
    pub target_max_satisfied: bool,
    pub n_oversize_before: usize,
    pub n_oversize_after_candidate: usize,
    pub max_weight_before: f64,
    pub max_weight_after_candidate: f64,
    pub split_iterations: Vec<DongdaemunSplitIteration>,
    pub trim_moves_committed: usize,
    pub trim_moves_proposed: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DongdaemunStatus {
    NoCurrentOversizeCandidates,
    Committed,
    CommittedBestCapState,
    NoSelectedCandidates,
    NoProgress,
    SplitQualityBelowFloor,
    TrimQualityBelowFloor,
    QualityBelowFloor,
    HardCapNotSatisfied,
}

#[derive(Clone, Debug)]
pub struct DongdaemunResult {
    pub clustering: Clustering,
    pub diagnostic_clustering: Option<Clustering>,
    pub audit: DongdaemunAudit,
}

pub fn dongdaemun_refine(
    graph: &Graph,
    baseline: &Clustering,
    config: &DongdaemunConfig,
    ws: &mut Workspace,
) -> DongdaemunResult;
```

Keep the first API deliberately narrow. It should accept the baseline
membership, not raw Leiden membership, because lower-tail repair is separate.

## Internal Helpers To Implement First

These helpers make the tests deterministic and keep the main function readable.

```rust
fn cluster_weight_summary(
    graph: &Graph,
    clustering: &Clustering,
    target_max_weight: f64,
    ws: &mut Workspace,
) -> ClusterWeightSummary;

fn current_oversize_clusters(
    graph: &Graph,
    clustering: &Clustering,
    target_max_weight: f64,
    ws: &mut Workspace,
) -> Vec<u64>;

fn cpm_delta(
    graph: &Graph,
    before: &Clustering,
    after: &Clustering,
    resolution: f64,
) -> f64;

fn accept_candidate(
    graph: &Graph,
    baseline: &Clustering,
    candidate: &Clustering,
    config: &DongdaemunConfig,
    ws: &mut Workspace,
) -> AcceptanceDecision;
```

`accept_candidate` is the contract-critical helper. It should be tested before
the stochastic candidate generators are wired in.

Full-core helpers to add before Slice 4:

```rust
fn candidate_conflicts(left: &DongdaemunCandidate, right: &DongdaemunCandidate) -> bool;

fn rank_candidates(
    candidates: &[DongdaemunCandidate],
    policy: DongdaemunPolicy,
) -> Vec<usize>;

fn apply_candidates_sequentially(
    graph: &Graph,
    current: &Clustering,
    ranked: &[DongdaemunCandidate],
    config: &DongdaemunConfig,
    ws: &mut Workspace,
) -> SequentialApplyResult;
```

The first full implementation should apply candidates sequentially with
rollback, not as an all-or-nothing batch. Batch application can be added later
only if the whole batch is re-audited and has a deterministic fallback path.

## Implementation Slices

### Slice 1: Contract Helpers

Implement:

- `DongdaemunPolicy`
- `DongdaemunConfig`
- `DongdaemunStatus`
- `DongdaemunAudit`
- `DongdaemunResult`
- weight summary helper
- oversize detection helper
- exact CPM delta helper
- acceptance helper

No split-repair, no trim yet.

Tests:

- detects oversize clusters sorted by descending weight and stable cluster id.
- `quality_first` accepts exact `Delta Q >= 0` even if cap remains violated.
- `quality_first` rejects exact `Delta Q < 0` and returns baseline.
- `hard_cap` rejects non-cap-satisfying candidate even when `Delta Q >= 0`.
- `hard_cap` can select a previously audited cap-satisfying state when a later
  diagnostic candidate fails.
- no-oversize returns baseline with `NoCurrentOversizeCandidates`.

### Slice 2: Boundary Trim Wrapper

Use existing primitive:

```rust
adaptive::trim_oversize_boundary_moves(...)
```

Implement quality-floor prefix behavior in Rust-native Dongdaemun if needed, or
reuse the same semantics as Python:

- `quality_first`: `min_delta_q = 0`.
- `hard_cap`: `min_delta_q = -1` by default.
- final effective output still must satisfy the policy audit.

Tests:

- quality-first trim commits only non-negative boundary moves.
- hard-cap may propose negative trim moves but falls back if final cap fails.
- diagnostic clustering is present when candidate changed but was rejected.

### Slice 3: Split-Repair Wrapper

Use existing primitives:

```rust
adaptive::split_merge_repair_probes(...)
adaptive::apply_split_merge_repair_candidates(...)
```

Selection should start conservative:

- filter candidates with `net_delta_q >= 0`,
- for `quality_first`, apply the quality gate first, then prefer upper-tail
  improvement among non-regressing candidates,
- for `hard_cap`, prefer cap satisfaction or cap-violation reduction among
  non-regressing candidates,
- prefer smaller `largest_source_unit_fraction`,
- prefer larger `escaped_source_weight`,
- prefer larger `net_delta_q`,
- stable tie-break by `cluster`, then `gamma_multiplier`.

This selection does not need to match Python exactly in the first slice, but it
must obey the final exact CPM audit.

Tests:

- split-repair can improve a toy oversized cluster with positive exact delta.
- selected candidate application is deterministic with `pair_seeded = true`.
- if split-repair produces negative exact delta, output falls back to baseline.
- a zero-delta no-op candidate is rejected and cannot create a repeated loop.
- conflicting candidates are skipped when `affected_nodes` or `touched_clusters`
  overlap.

### Slice 4: Full Dongdaemun Core

Wire:

```text
oversize detect
-> repeated split-repair
-> boundary trim
-> sequential exact audit with rollback
-> final exact audit
-> effective or fallback output
```

Tests:

- full quality-first path commits on the existing escape-fragment toy graph.
- full hard-cap path accepts only when final max weight is below target.
- full hard-cap returns the best audited cap-satisfying state when one exists.
- full hard-cap fallback preserves baseline as effective membership when no
  cap-satisfying audited state exists.
- audit fields match the committed/diagnostic membership.
- `apply_iterations` terminates repeated positive changes.
- no-progress detection terminates when candidates do not change membership.

### Slice 5: Python Binding Later

Only after Rust tests pass:

- expose `Graph.dongdaemun_refine(...)` or a standalone binding,
- return arrays plus audit dict,
- add Python tests that compare Rust audit results to existing Python
  hierarchy postprocess helpers on toy graphs.

## Test Fixture Design

Use tiny deterministic graphs. Avoid large validation artifacts and avoid
randomness-sensitive assertions.

### Fixture A: No Oversize

```text
nodes: 0--1, 2--3
membership: [0, 0, 1, 1]
node_weights: [1, 1, 1, 1]
T_max = 3
```

Expected:

- no oversize,
- output equals baseline,
- accepted = true,
- `Delta Q = 0`.

### Fixture B: Positive Boundary Trim

Existing adaptive test pattern:

```text
edges:
0--1 weight 3.0
1--2 weight 0.1
2--3 weight 4.0
membership: [0, 0, 0, 1]
gamma = 0.1
T_max = 2
```

Expected:

- node `2` moves from cluster `0` to cluster `1`,
- candidate membership `[0, 0, 1, 1]`,
- move delta is positive,
- quality-first accepts.

### Fixture C: Quality-First Allows Residual Oversize

Use a graph where a positive move reduces max weight but does not satisfy the
cap. Set `T_max` low enough that residual oversize remains.

Expected:

- accepted under `quality_first`,
- `target_max_satisfied = false`,
- final `Delta Q >= 0`.

This locks down the "quality-first is not hard-cap" distinction.

### Fixture D: Hard-Cap Fallback

Reuse Fixture C with `policy = HardCap`.

Expected:

- candidate may have non-negative `Delta Q`,
- cap remains violated,
- effective output equals baseline,
- diagnostic output contains candidate.

### Fixture E: Negative Quality Candidate

Use `accept_candidate` directly with a hand-built candidate membership that
splits a strong clique into weak pieces.

Expected:

- exact `Delta Q < 0`,
- rejected for both policies,
- effective output equals baseline.

## Invariants Every Test Should Check

For any accepted result:

```rust
Q(result.clustering) >= Q(baseline) + quality_floor_delta - 1e-9
```

For any rejected result:

```rust
result.clustering.clusters == baseline.clusters
result.diagnostic_clustering.is_some() only if a changed candidate existed
```

For hard-cap accepted result:

```rust
max_cluster_weight(result.clustering) <= target_max_weight
```

For quality-first accepted result:

```rust
cap satisfaction is reported, not required
```

For full-core sequential application:

```rust
each committed candidate has exact step Delta Q >= -1e-9
no committed candidate conflicts with an earlier committed candidate
```

## Test Module Layout

Use integration tests for public contract:

`rust/tests/dongdaemun.rs`

Suggested structure:

```rust
use sciscape_leiden::dongdaemun::*;
use sciscape_leiden::{Clustering, Graph, QualityFunction, CPM};
use sciscape_leiden::workspace::Workspace;

fn cpm_quality(graph: &Graph, clustering: &Clustering, resolution: f64) -> f64 { ... }
fn weighted_graph(...) -> Graph { ... }
fn assert_effective_equals_baseline(...) { ... }

#[test]
fn no_oversize_returns_baseline() { ... }

#[test]
fn quality_first_accepts_non_regressing_candidate_with_residual_oversize() { ... }

#[test]
fn hard_cap_falls_back_when_cap_not_satisfied() { ... }

#[test]
fn full_quality_first_commits_positive_boundary_trim() { ... }
```

Use unit tests inside `rust/src/dongdaemun.rs` only for private helpers that
should not become public API.

## Verification Commands

Run the new tests only:

```bash
cargo test --manifest-path rust/Cargo.toml dongdaemun
```

Run related adaptive primitive tests:

```bash
cargo test --manifest-path rust/Cargo.toml adaptive::tests::trim_oversize
cargo test --manifest-path rust/Cargo.toml adaptive::tests::split_merge
```

Run full Rust crate tests:

```bash
cargo test --manifest-path rust/Cargo.toml
```

Run Python tests only after bindings or Python-facing behavior changes:

```bash
uv run --extra dev maturin develop --manifest-path rust/Cargo.toml
uv run --extra dev python -m pytest -q tests/test_adaptive_refinement.py tests/test_leiden_rust.py
```

## Open Design Decisions

## Issue Disposition For Rust Slices

| Issue | Rust decision | Slice |
| --- | --- | --- |
| Termination | Implemented `apply_iterations`, no-progress detection, no-op rejection, and rejected candidate signatures in the Rust core. | Slice 4 done |
| Conflict resolution | Start with conservative `affected_nodes` or `touched_clusters` overlap. | Slice 3/Slice 4 |
| Batch apply | Do not batch in the first implementation. Use sequential exact audit with rollback. | Slice 3/Slice 4 |
| Hard-cap partial commit | Deferred. Track best audited cap-satisfying state and expose `CommittedBestCapState` only after a failing fixture or loop semantic need appears. | Later slice |
| Quality-first ranking | Implemented Rust receiver oversize-aware ranking. Python ranking parity is deferred. | Slice 3 done / later parity |
| Probe schedule | Use fixed `gamma_multipliers` and one seed. Add hierarchical/multi-seed probes later. | Slice 3 done / later |
| Boundary polish versus local move | Test boundary polish as targeted post-repair behavior, not as a general Leiden rerun. | Slice 2 |
| Config clarity | Keep config fields only if each maps to a concrete generator or audit rule. | Slice 1 |
| `pair_seeded` | Use pair-seeded replay in split tests that need deterministic probe/application parity. | Slice 3 |
| Lower-tail interaction ablation | Do not implement as Rust core logic. Keep `P_min` as input and add experiments later. | Later |
| Python parity | Do not chase Python ranking parity in the first Rust module. | Later |

## Open Design Decisions

1. Selection parity with Python:
   The first Rust selector can be simpler than Python's report-oriented
   `rank_split_repair_candidates`, as long as the exact audit contract is
   preserved. Later we can port the exact ranking if evidence reproduction needs
   byte-for-byte parity.

2. Candidate diagnostics:
   The first Rust result should return compact audit fields. Full CSV-style
   diagnostics should stay in Python until there is a clear need.

3. Node weights:
   Tests should include at least one weighted graph via
   `Graph::from_edge_list_weighted` because hierarchy contraction uses weighted
   supernodes.

4. Randomness:
   Tests involving split-repair should use `pair_seeded = true` and assert only
   stable structural outcomes. Helper/acceptance tests should avoid randomness
   entirely.

5. Conflict strictness:
   Start with conservative conflicts: candidates conflict if affected nodes
   overlap or source/target cluster sets overlap. A later implementation can
   regenerate candidates after each commit and relax the cluster-set rule.

6. Probe budget:
   The first Rust implementation should run one probe seed and the configured
   gamma multiplier list. Multi-seed probing belongs in a later robustness
   layer.

## Historical First Coding Task

Completed:

- [x] Create `rust/src/dongdaemun.rs`.
- [x] Export it from `rust/src/lib.rs`.
- [x] Add `rust/tests/dongdaemun.rs`.
- [x] Implement helper-level and integration contract tests.
- [x] Add PyO3/Python helper coverage.
- [x] Add hierarchy opt-in fast-path coverage.

Deferred later-slice work: `CommittedBestCapState`, Python ranking parity,
lower-tail ablation, multi-seed/hierarchical schedules, and fused-kernel
profiling.
