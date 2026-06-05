# Leiden Basin Generalization, Demo, And Method Design

Status: design draft
Date: 2026-05-31
Scope: Track C guiding-premise validation design

This document preserves the guiding premise:

> Graph clustering optimizers can get trapped in distinct partition basins, and
> compact interventions may move the optimizer between them more efficiently
> than broad restart.

The current NanoClustering evidence is useful discovery evidence, but it is not
the final validation design. One seed anchor, one data family, or one large
case cannot carry the premise. The next design must test generality,
mechanism, small-demo reproducibility under Leiden + CPM, and method-level
improvement in that order.

## Design Principle

Do not treat the current 83/223 primitive ladder as the main result. Treat it
as a phenomenon-mining surface.

The final study should answer four linked questions:

1. **Generality:** do basin-like endpoint alternatives recur across seeds,
   anchors, graph scales, and cases?
2. **Mechanism:** why do those alternatives separate? Which local graph
   structures create near-tie, absorption, balanced split, or fragmentation
   behavior?
3. **Baseline reproducibility:** can the same phenomenon be reproduced in a
   small graph with ordinary Leiden + CPM and no custom method?
4. **Method improvement:** can our method discover, diagnose, or navigate those
   alternatives better than Leiden + CPM restarts?

The fourth question is only meaningful if the first three pass.

## Evidence Ladder

| gate | question | minimum pass condition | blocked claim if it fails |
| --- | --- | --- | --- |
| G1 generality | Does the phenomenon survive seed and case variation? | repeated alternatives across reference-seed anchors, branches, and at least one additional graph family | seed-invariant basin taxonomy |
| G2 mechanism | Can we name the structural cause of separation? | alternatives align with measured near-tie cut, bridge mass, absorption, balanced split, or fragmentation axes | mechanism explanation |
| G3 demo | Can a small graph reproduce it under Leiden + CPM? | a small graph has two or more recurring Leiden + CPM endpoints with interpretable basin separation | general methodological result |
| G4 wall/pathway | Is there optimizer-native route structure? | route traces show crossing, bounce, collapse, or unknown under predeclared evidence fields | basin wall/pathway claim |
| G5 method | Does our method improve over baseline? | better basin discovery, cheaper basin navigation, or clearer wall diagnosis than restart baseline | algorithmic contribution |
| G6 downstream value | Is the navigated basin useful? | quality/cost is evaluated only after G1-G5 pass | quality or cost claim |

## Phase A: Generality Audit

Purpose:

- test whether the NanoClustering signal is a general phenomenon rather than a
  seed0 coordinate artifact.

Required artifacts:

- `seed_anchor_rotation_audit`;
- `symmetric_endpoint_object_audit`;
- branch-level and case-level replication tables.

Protocol:

1. Rotate the reference seed across all 10 Java and 10 Rust endpoints.
2. For each anchor, rebuild anchor-local primitive candidates using the same
   event vocabulary: fragmentation, absorption, balanced split, host context,
   and boundary pattern.
3. Measure recovery of the seed0 T1/T2/T3/T4/T5/T6 families under each anchor.
4. Build symmetric all-seed endpoint objects by overlap or co-association
   components so that objects no longer depend on one reference run.
5. Compare anchor-local and symmetric objects.

Pass:

- T1-like structure is recovered under multiple non-seed0 anchors;
- symmetric endpoint objects explain the same alternatives without relying on
  seed0 cluster IDs.

Fail:

- the signal remains only seed0-local. This does not reject the guiding
  premise, but it rejects this measurement strategy as a basin-taxonomy basis.

## Phase B: Mechanism Extraction

Purpose:

- turn endpoint differences into named structural mechanisms.

Candidate mechanism axes:

- near-tie CPM cut: two assignments have close CPM objective under local moves;
- bridge-mass ambiguity: a small boundary or bridge set supports either side;
- absorption: a reference cluster remains mostly intact but is absorbed into a
  host dominated by another cluster;
- balanced split: a source cluster divides into multiple substantial pieces
  with no single dominant survivor;
- diffuse fragmentation: a source cluster spreads across many comparison
  clusters;
- resolution interaction: small CPM gamma changes alter whether a bridge is
  cut or absorbed.

Required measures:

- local weighted degree;
- cut ratio;
- inter-candidate edge mass;
- top-neighbor concentration;
- CPM delta under local reassignment;
- segment-size entropy or effective segment count;
- host dominance and source-host preservation.

Pass:

- each recurrent basin-candidate family maps to one or more mechanism axes;
- matched controls are matched not only by size, but also by local topology.

Fail:

- endpoint alternatives remain descriptive labels with no graph-mechanism
  explanation.

## Phase C: Small Demo Case

Purpose:

- prove that the phenomenon is not a NanoClustering artifact.

The small demo must use ordinary Leiden + CPM as the baseline. It should be
small enough to inspect manually, rerun quickly, and visualize.

Minimum graph families:

| family | intended mechanism | expected Leiden + CPM behavior |
| --- | --- | --- |
| near-tie bridge cliques | near-tie CPM cut and bridge ambiguity | different seeds assign bridge nodes to different sides |
| absorption triad | external-host absorption | a small module can remain separate or be absorbed by a larger host |
| balanced split module | source-host balanced split | a middle module splits into two substantial pieces |
| diffuse fragment star | diffuse fragmentation | a weakly coherent module fragments across several hosts |

Each graph family should define:

- node and edge list generator;
- CPM gamma;
- seed count;
- expected endpoint signatures;
- mechanism label;
- wall/pathway probe eligibility;
- visualization snapshot.

Pass:

- at least one tiny graph produces two or more recurring Leiden + CPM endpoints
  under different seeds;
- the endpoints correspond to a named mechanism axis;
- the same graph is simple enough to explain without relying on large-data
  intuition.

Fail:

- the large-data phenomenon cannot be reproduced in small CPM graphs. In that
  case, the current approach may be describing data-scale endpoint instability,
  not a general basin-tunneling mechanism.

## Phase D: Baseline And Method Comparison

Baseline:

- Leiden + CPM with repeated random seeds;
- optionally broader restart budget as the strong baseline;
- no custom intervention.

Our method should not be evaluated first by final quality. It should be
evaluated by whether it improves basin discovery or navigation.

Possible method roles:

1. **Basin discovery:** find distinct endpoint families with fewer restarts.
2. **Wall diagnosis:** identify why a pair is separated before route execution.
3. **Navigation:** apply compact structural handles to move from one basin
   candidate toward another.
4. **Cost control:** achieve comparable basin coverage with less restart cost.

Required comparison metrics:

- number of distinct basin candidates found per run budget;
- endpoint-family recall against exhaustive small-demo restarts;
- route-label coverage: crosses, bounces, collapses, unknown;
- intervention size;
- wall time and run count;
- CPM quality only after basin/wall gates pass.

Pass:

- the method improves discovery, diagnosis, or navigation under fixed budget
  on small demo graphs and at least one larger empirical panel.

Fail:

- the method is only a different threshold or candidate-selection policy. This
  would repeat the failed Track C pattern and should not be promoted.

## Phase E: Integration Back To NanoClustering

The NanoClustering artifacts should be used as:

- phenomenon mining;
- stress testing;
- scale and density diagnostics;
- source of mechanism candidates.

They should not be used alone as:

- the proof of seed-invariant basin taxonomy;
- the proof of optimizer-native walls;
- the proof of method improvement;
- the semantic interpretation of basin quality.

Integration rule:

1. Mine candidate mechanisms from NanoClustering.
2. Reproduce the mechanism on a small demo graph.
3. Confirm the mechanism under Leiden + CPM baseline.
4. Apply the proposed method.
5. Return to NanoClustering only after the mechanism and method are validated
   in controlled form.

### Return-To-NanoClustering Constraint

The 2026-06-02 symmetric-object critical-gamma bracket is useful as a
constraint on this integration rule, not as a shortcut around it. The
support-top100 P1 unique membership bracket fixes the same 4 objects and 16
deterministic starts across `gamma=1e-5`, `3e-5`, and `1e-4`. At `1e-5`, all 4
objects show terminal multiplicity and start-pair object terminal ARI median is
0.8996455967664517. At `3e-5`, all 4 objects close to one terminal with
start-pair ARI 1.0, despite top100/doc-weight local critical-gamma maxima being
about `3.7e-5` to `4.6e-5`. At `1e-4`, all 4 objects remain closed and
singleton-collapse.

This means a local objective-positive merge screen is not enough to define a
basin. The empirical NanoClustering return path needs membership-level terminal
differences, an object-specific phase-boundary diagnostic, and mechanism labels
for what changes across terminals. It should not proceed directly to
quality/cost, wall/pathway, or algorithm claims.

The first membership-difference review narrows the mechanism target. At
`gamma=1e-5`, the 4 P1 objects have only 114 variable universe node-pairs
across all saved starts: 80 object-object, 16 object-support, and 18
support-support. Median object-level variable co-assignment pair share is
0.007416764891781547, and the `3e-5`/`1e-4` controls have zero variable
node-pairs. The controlled demo should therefore not try to reproduce a broad
partition rearrangement first. The closer target is a weak-pair or small-bridge
partial-coarsening phase transition where a compact node set switches
co-assignment under ordinary Leiden + CPM.

The graph-local variable-pair review makes this target more concrete. All 114
observed variable node-pairs have direct graph edges and shared neighbors; 79
are direct-positive under doc-weighted CPM at `gamma=1e-5`, but only 4 remain
direct-positive at `3e-5` and 0 at `1e-4`. A controlled demo should therefore
try to reproduce a compact weak-pair switch whose critical gamma sits near the
observed bracket, with shared-neighbor bridge mass as an alternate or coupled
mechanism. The demo should keep quality/cost and wall claims closed until this
mechanism is reproduced under ordinary Leiden + CPM.

The counterfactual panel gate freezes the first demo targets rather than
opening another broad search. It classifies the 114 graph-scored pairs into 75
direct phase-boundary pairs, 35 negative-direct bridge-mediated pairs, and 4
persistent direct-positive controls, then selects a 23-pair panel covering all
4 objects and all three pair scopes. The next controlled test should first use
this panel to ask whether direct-edge removal, bridge-mass removal, or a small
synthetic Leiden+CPM construction reproduces the observed partial-coarsening
switch. Passing this gate would support mechanism reproduction only; it would
still not establish basin walls, pathways, quality/cost improvement, or an
algorithm claim.

The first local ablation gate partially reproduces the mechanism but also
changes the mechanism wording. On the frozen 23-pair panel, the local induced
graphs use recoverable top common-neighbor bridge nodes and run ordinary
Leiden+CPM over 4 graph variants, 5 start conditions, and 8 seeds each. The
result has 17 of 23 pairs with local diagnostic support: 12 direct-edge
sensitive, 2 direct-and-bridge sensitive, and 3 seed/start sensitive. Six pairs
do not reproduce original local co-assignment. Across variants, direct-edge
removal drives pair co-assignment to zero, while pair-to-bridge removal often
collapses the pair together. The next controlled demo should therefore target
direct-contact phase sensitivity under bridge-context competition, not a
generic "bridge mass merges the pair" story. Full-graph replay, wall/pathway
evidence, quality/cost comparison, and method claims remain closed.

The synthetic-demo design gate fixes the next runner surface. The 23 local
ablation pairs separate into 6 design families: 7 stable direct-contact
competition cases, 5 partial direct-contact competition cases, 2 coupled
negative-direct bridge-contact cases, 3 rare start-sensitive direct-contact
cases, 4 overcompeting bridge-context controls, and 2 nonlocal negative-direct
context controls. The first synthetic runner should implement only this family
surface. It should not search new mechanisms. Its positive families should
start with direct pair contact plus symmetric bridge competitors and host
context; direct-edge removal must separate the pair, while bridge-context
removal should collapse or simplify the pair endpoint. The coupled
negative-direct family should remain separate because both direct and bridge
removal break co-assignment. The overcompeting and nonlocal families are
negative controls, not failures to patch away.

The first variable-pair synthetic CPM runner executes that fixed surface under
ordinary Leiden+CPM. It materializes
`../../../research/consensus/results/adaptive_refinement/leiden_basin_variable_pair_synthetic_demo_v1_20260603/`
from 6 small graph families, 4 graph variants, 5 start conditions, and 16
seeds. All 6 predeclared signatures reproduce: stable direct contact,
partial/mixed direct contact, coupled direct-plus-bridge contact, rare
start-sensitive contact, overcompeting bridge context, and nonlocal negative
context. This passes the controlled G3 mechanism-reproduction gate for the
variable-pair surface only. It does not establish full-graph basin walls,
pathways, method improvement, quality/cost value, or an algorithm claim.

The endpoint replay gate then checks whether those terminal signatures are
stable enough to become G4 route targets. It materializes
`../../../research/consensus/results/adaptive_refinement/leiden_basin_variable_pair_synthetic_endpoint_replay_v1_20260603/`.
All 21 original-variant endpoint signatures replay to themselves across 16
ordinary-Leiden+CPM seeds. Route candidates are therefore restricted to the
families where stable endpoints differ in L/R co-assignment:
`partial_direct_contact_competition` and
`rare_start_sensitive_direct_contact`. The artifact lists 24 candidate
relations, but it does not execute route traces or promote walls.

The first compact route-trace gate executes those 24 candidate relations under
ordinary Leiden+CPM and materializes
`../../../research/consensus/results/adaptive_refinement/leiden_basin_variable_pair_synthetic_route_trace_v1_20260603/`.
It tests source replay, target replay, pair-only, bridge-side-only, and
pair-plus-bridge-side initial memberships. The result is asymmetric:
`coassigned_to_separated` crosses robustly under `pair_plus_bridge_side` for all
12 candidates, while `separated_to_coassigned` has no robust compact crossing
policy. The next interpretation should therefore be endpoint-relation
directionality, not a bidirectional wall claim.

The stricter G4.1 audit corrects that first read. It materializes
`../../../research/consensus/results/adaptive_refinement/leiden_basin_variable_pair_synthetic_route_trace_g4_1_audit_v1_20260603/`
and a matching `n_iterations=10` check. The audit flags target-identical
initializations and source no-ops, adds bridge-context release policies for the
reverse direction, records intervention size, and includes 80 same-pair-state
controls. Once target reconstruction is excluded, strict crossing remains only
for `separated_to_coassigned`: 12 of 24 relation-change candidates have a
strict nonidentical crossing policy, while all 80 same-state controls have none.
The current G4 interpretation is therefore not bidirectional tunneling and not
the original `coassigned_to_separated` target-reconstruction result. It is a
synthetic trace diagnostic that bridge-context release can move separated
endpoints into coassigned targets under ordinary Leiden+CPM polish.

The G4.2 necessity audit then decomposes those strict G4.1 crossings instead of
adding more trace-policy sweeps. It materializes
`../../../research/consensus/results/adaptive_refinement/leiden_basin_variable_pair_synthetic_route_trace_g4_2_necessity_v1_20260603/`
and an `n_iterations=10` G4.1-input check. The audit records 12 strict-crossing
focus rows, 36 source/initial/target stage rows, 48 bridge-transition rows, and
sibling-policy context. It separates two component patterns:

- `partial_direct_contact_competition`: 4 crossings use
  `pair_plus_left_context_release`. The initial state already coassigns the
  pair and releases one left bridge into target context, but both initial and
  target CPM quality are lower than the separated source. This is a crossing
  diagnostic, not a better-basin or value claim.
- `rare_start_sensitive_direct_contact`: 8 crossings use
  `bridge_context_release_only`. The initial state keeps `L` and `R`
  separated, releases 1-2 bridge nodes into target host context, and then
  ordinary Leiden+CPM polish coassigns the pair. Initial quality is effectively
  tied with source, while target quality is higher by about `0.16`.

The stronger synthetic mechanism cue is therefore context release that can
precede pair coassignment. Direct pair merging remains a sibling policy to
audit, not the current mechanism center.

The G4.3 frozen-handle generalization probe then tests that cue without target
endpoint reconstruction. It materializes
`../../../research/consensus/results/adaptive_refinement/leiden_basin_variable_pair_synthetic_g4_3_handle_generalization_v1_20260603/`
and an `n_iterations=10` check. The predeclared handle releases bridge nodes
from pair-node clusters into same-side host context while keeping `L` and `R`
separated. On a fixed 9-case panel, all 4 positive holdouts pass: each has 8
eligible separated source endpoints, and all 8 robustly polish to a known
coassigned endpoint under the frozen handle. Pair-only merge has 0 robust rows.
All 5 matched/negative controls also pass their expected negative gate:
bridge-release has 0 robust rows. Three matched controls still show partial
pair coassignment at `0.3125`, so the evidence is not that the handle is
universal. The narrow read is that bridge-context release can robustly trigger
pair coassignment only inside a context-threshold regime with sufficient direct
support and appropriate pair-bridge balance.

The G4.4 fixed-panel restart comparison keeps that handle frozen and compares
it with ordinary Leiden+CPM restart discovery on the same G4.3 panel. It
materializes
`../../../research/consensus/results/adaptive_refinement/leiden_basin_variable_pair_synthetic_g4_4_restart_comparison_v1_20260603/`
and an `n_iterations=10` G4.3-input check. All 4 positive holdouts pass the
source-conditioned comparison: from each eligible separated source endpoint,
the frozen handle hits the known coassigned endpoint with probability `1.0`,
while case-level restart discovers the known coassigned endpoint with
probability `0.2375`. The median expected-run ratio is therefore `4.210526` in
favor of the handle. Pair-only remains non-robust. All 5 controls remain
non-robust, but 12 matched-control source rows show a partial-above-restart
caveat: `0.3125` for the handle versus `0.2375` for restart. This is
source-conditioned navigation evidence, not a full method comparison. The
missing pieces are source availability, graph/source-local handle selection,
control suppression, schedule overhead, and full-graph replay.

The G4.5 selector/suppression gate then freezes
`neutral_release_with_direct_support_v1` for that same handle and materializes
`../../../research/consensus/results/adaptive_refinement/leiden_basin_variable_pair_synthetic_g4_5_selector_suppression_v1_20260603/`
plus an `n_iterations=10` input check. The selector is target-free and
source-local: it requires handle eligibility, at least one released bridge,
pair-separating initialization, unchanged pair relation, source-neutral CPM
initialization delta (`abs(delta)<=1e-6`), and direct pair support at least
`1.0`. It selects all 32 positive source-conditioned wins and suppresses all 24
control source rows from G4.4, including all 12 matched-control
partial-above-restart caveats. This closes the selector/suppression diagnostic
for the fixed synthetic panel, but it is still not a method comparison because
source availability, selector overhead, schedule accounting, wall identification,
and full-graph replay remain unmeasured.

The G4.6 schedule-accounting gate then keeps both pieces frozen and materializes
`../../../research/consensus/results/adaptive_refinement/leiden_basin_variable_pair_synthetic_g4_6_schedule_accounting_v1_20260603/`
plus an `n_iterations=10` input check. One schedule cycle is: run ordinary
Leiden+CPM once; accept if the endpoint is already the known coassigned target;
otherwise apply the frozen handle once only if the endpoint passes the G4.5
selector; otherwise no-op. This includes observed source availability and a
restart-plus-handle unit accounting. The fixed panel passes: all 4 positives have
baseline target probability `0.2375`, selected-source probability `0.7625`,
schedule hit probability `1.0`, and restart-plus-handle unit ratio `2.388951`
over restart. All 5 controls add no probability over baseline. This is
schedule-accounting evidence for the frozen synthetic demo, not a full method
claim, because wall-clock timing, independent source discovery, independent
panels, wall identification, and full-graph replay remain unmeasured.

The G4.7 independent schedule-stress gate keeps the same frozen pieces and
materializes
`../../../research/consensus/results/adaptive_refinement/leiden_basin_variable_pair_synthetic_g4_7_independent_schedule_stress_v1_20260603/`
plus an `n_iterations=10` check. It replays the full G4.3 -> G4.4 -> G4.5 ->
G4.6 chain on a shifted 9-case panel. The result is a boundary failure, not a
selector-tuning prompt. All 5 controls remain suppressed with no added leak.
All 4 stress positives fail: 3 are already coassigned under all ordinary
restarts, so the schedule has no separated source opportunity, and 1 has a mixed
endpoint surface but only pair-only, not bridge-release, is robust. The frozen
schedule therefore requires a specific opportunity regime: coexistence of
separated and coassigned endpoints, bridge-release eligible separated sources,
and source-neutral release. The current evidence supports this regime-specific
diagnostic only.

The G4.8A opportunity-regime design artifact then freezes the next measurement
surface without running new Leiden jobs. It materializes
`../../../research/consensus/results/adaptive_refinement/leiden_basin_variable_pair_synthetic_g4_8_opportunity_regime_design_v1_20260603/`
by reading the G4.3 success panel, G4.6 schedule accounting, and G4.7 stress
panel. It classifies all 18 existing cases into 6 regimes: 4 ready
bridge-release opportunities, 6 suppressed coexistence controls, 3
target-saturated no-source boundaries, 1 pair-only boundary, and 4 no-target
boundaries. The metrics are now fixed for the next gate: endpoint coexistence,
bridge-release eligibility, source-neutral/selected release, pair-only
ambiguity, target saturation, target absence, and control leak. The next
runnable gate should be a fresh predeclared regime-cell panel, not a selector
sweep.

The G4.8B predeclared regime-cell panel is then executed at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_variable_pair_synthetic_g4_8b_regime_cell_panel_v1_20260603/`
with an `n_iterations=10` check. It keeps the G4.3 handle, G4.5 selector, and
G4.6 schedule frozen and runs 10 fresh cases across 5 predeclared regime cells.
It fails usefully. The suppressed-control, target-saturation, and no-target
boundary cells reproduce their expected roles and have no added source-handle
leak. But the 2 intended ready bridge-release cells and 2 intended pair-only
boundary cells are all observed as target-saturated no-source boundaries:
ordinary restarts already coassign the pair at rate 1.0, so no separated source
endpoint exists for the frozen schedule. This means the current unresolved
problem is not source discovery. It is first the construction or recognition of
the narrow endpoint-coexistence opportunity surface.

G4.8C then materializes that opportunity-construction cartography at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_variable_pair_synthetic_g4_8c_opportunity_cartography_v1_20260603/`
with an `n_iterations=10` check. It runs 30 predeclared anchor, one-axis, and
G4.8B-collapse-decomposition cases. The result is stable: 18 cases preserve
ready opportunity with source-handle fire, 7 collapse into target saturation,
and 5 keep endpoint coexistence but lose robust bridge release. The direct
support and host-clique axes are not the active construction boundary in the
tested range. The active boundary is the balance between pair-bridge and
bridge-host support: one side saturates the target, while the other side keeps
coexistence but makes bridge release nonrobust.

G4.8D then executes the 2D balance cartography at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_variable_pair_synthetic_g4_8d_balance_cartography_v1_20260603/`
with an `n_iterations=10` check. It runs a predeclared 56-cell
pair-bridge/bridge-host grid while keeping direct and host-clique support fixed.
The result is stable and more precise than the G4.8C one-axis read: ready
opportunity appears as a sparse diagonal ridge, not a broad connected band and
not a single anchor-only knife edge. The ready cells are `(1.32,1.44)`,
`(1.35,1.45)`, and `(1.38,1.46)`. Cells below the ridge are nonrobust
coexistence; cells above it saturate the target.

G4.8E then refines that diagonal at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_variable_pair_synthetic_g4_8e_diagonal_ridge_refinement_v1_20260603/`
with an `n_iterations=10` check. It runs a predeclared 65-cell strip around
`bridge_host = 1.44 + (pair_bridge - 1.32) / 3`, with `pair_bridge` in
`0.005` steps and offsets `-0.004,-0.002,0,+0.002,+0.004`. The ridge does not
become continuous. It resolves into a centerline resonance lattice: 5 ready
cells, all at offset `0` and spaced by `0.015` in pair-bridge weight, plus 30
target-saturated cells and 30 nonrobust-coexistence cells. The centerline
pattern repeats `R/T/N`. The ready centerline cells have 8 separated source
endpoints, 8 robust bridge-release sources, and schedule hit rate `1.0`; the
centerline nonrobust cells have 4 eligible sources but no robust bridge-release
source; the centerline target cells are already saturated at baseline.

G4.8F then audits the centerline endpoint/source signatures at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_variable_pair_synthetic_g4_8f_centerline_signature_audit_v1_20260603/`
with an `n_iterations=10` G4.8E-input check. It is read-only and explains the
centerline roles by signature availability and source neutrality. `R` cells
have 1 coassigned endpoint plus 8 separated source endpoints: 4 two-side
bridge-split sources and 4 single-side bridge sources. All 8 are
source-neutral, selected by the frozen G4.5 selector, and robust under the
G4.3 bridge-release handle. `N` cells have the coassigned endpoint plus only
the 4 two-side bridge-split sources; the release has
`initial_quality_delta_vs_source=-0.004`, so the frozen selector suppresses
them and the handle remains partial (`handle_known_hit_rate=0.3125`). `T`
cells are target-saturated and expose no separated source. The demo surface now
has a construction-read hypothesis, but this is still not source discovery,
wall/pathway evidence, quality/cost evidence, or a method claim.

G4.8G then tests that frozen construction-read hypothesis on fresh synthetic
contexts at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_variable_pair_synthetic_g4_8g_fresh_context_signature_validation_v1_20260603/`
with an `n_iterations=10` check. It changes the direct/host support context
away from the original G4.8D/G4.8E center while keeping the 13-cell centerline
and expected `R/T/N` role pattern fixed. Across 4 fresh contexts, all 52 cases
match the expected role and all 52 pass the role-specific source-signature
expectation. Each context repeats `RTNRTNRTNRTNR`; `R` keeps the full 8-source
neutral/selected/robust signature set, `N` keeps only the 4 two-side
source-nonneutral signatures, and `T` exposes no separated source. This is a
fresh-context validation of the construction-read rule, not independent source
discovery, wall/pathway evidence, quality/cost evidence, NanoClustering replay,
or a method claim.

G4.8H then runs the bounded source-discovery smoke at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_variable_pair_synthetic_g4_8h_source_discovery_smoke_v1_20260603/`
with an `n_iterations=10` G4.8G-input check. It reads materialized G4.8G
endpoint and bridge-release initialization rows and applies the target-free
`pair_separated_bridge_attached_then_neutral_release_v1` rule. The decision
uses only endpoint/source-local inputs: pair coassignment, pair-attached bridge
count, handle eligibility, released bridge count, initial pair relation,
initial quality delta, and direct support. Role labels, known target hit rates,
robustness, and oracle signature flags are evaluation-only. The smoke passes:
all 52 cases pass in both inputs, source-set exact match is 52/52, and
ready-source-set exact match is 52/52. `R` cells recover all 8 release sources
and all 8 ready sources; `N` cells recover the 4 release sources but suppress
all ready sources because the release is source-nonneutral; `T` cells expose no
source. This freezes a bounded source-discovery rule for the next fresh
schedule panel only. It is still not independent source discovery on new
graphs, wall/pathway evidence, quality/cost evidence, NanoClustering replay, or
a method claim.

G4.8I then executes that fresh schedule panel at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_variable_pair_synthetic_g4_8i_discovered_source_schedule_panel_v1_20260604/`
with an `n_iterations=10` check. It uses four new edge-mid direct/host contexts:
`(1.08,1.23)`, `(1.08,1.27)`, `(1.06,1.25)`, and `(1.10,1.25)`. It keeps the
13-cell centerline, the G4.3 bridge-release handle, and the G4.8H target-free
source-discovery rule fixed. Schedule decisions are made from discovered
sources, not oracle source-signature reads. The result passes in both runs:
52/52 cases match the expected role and 52/52 pass the schedule gate. Every
context repeats `RTNRTNRTNRTNR`. In `R` cells, the discovered-source schedule
finds 8 release sources and 8 ready sources, raises the hit rate from baseline
`0.2375` to `1.0`, and has restart-plus-handle unit ratio `2.388951`. In `N`
cells it finds 4 release sources but 0 ready sources, so the schedule adds no
handle and stays at baseline `0.2375`. In `T` cells it finds no source and
baseline already saturates the target. This is fresh synthetic
schedule-accounting evidence only, not wall/pathway evidence, wall-clock
quality/cost value, NanoClustering replay, independent real-data source
discovery, or a method claim.

G4.8J then tests the first off-center failure-mode expansion at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_variable_pair_synthetic_g4_8j_off_center_failure_mode_panel_v1_20260604/`
with an `n_iterations=10` check. It keeps the G4.8I edge-mid direct/host
contexts and shifts the bridge-host support by `-0.002` and `+0.002` from the
centerline. The negative-offset expectation is nonrobust coexistence (`N`):
release sources can be discovered, but ready sources should be zero and the
schedule should not add the handle. The positive-offset expectation is target
saturation (`T`): no source should be discovered and baseline should already
coassign the pair. Both runs pass. All 52 negative-offset cases recover 4
release sources and 0 ready sources, with schedule hit `0.2375`. All 52
positive-offset cases recover 0 source candidates and stay target-saturated at
hit `1.0`. This freezes the off-center failure contract as a synthetic
diagnostic only. It is not wall/pathway evidence, wall-clock quality/cost
value, NanoClustering replay, independent real-data source discovery, or a
method claim.

The first NanoClustering G4.8 source-condition analog screen is materialized at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_g4_8_source_condition_analog_screen_gamma1e5_20260604/`.
It reads the frozen symmetric-object variable-pair graph-mechanism,
local-ablation, and synthetic-demo-design artifacts and classifies 23 local
pairs using the predeclared local-ablation proxy. The screen passes as
`real_data_analog_surface_has_ready_and_controls`: 6 strict partial
release-ready analogs, 3 rare-start release-ready analogs, 6 target-saturated
direct-contact no-handle analogs, 2 coupled direct-bridge context failure
controls, and 6 `N_like` controls. The result says the G4.8 source-condition
distinction has a local real-data analog surface. It does not supply exact
G4.8F endpoint/source signature sets, independent source discovery,
wall/pathway evidence, wall-clock quality/cost value, full NanoClustering
replay, or a method claim.

The first NanoClustering G4.8 frozen local analog validation panel is
materialized at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_g4_8_local_analog_validation_panel_gamma1e5_20260604/`.
It consumes the analog screen and freezes all 23 local pairs without
within-stratum cherry-picking. The panel keeps 6 strict-ready pairs, 3
rare-ready pairs, 6 target-saturated no-handle pairs, 4 latent-release nonready
controls, 2 no-release controls, and 2 coupled direct-bridge failure controls;
all 10 design gates pass. This is the next local validation surface, not the
validation execution itself. It does not run Leiden, materialize exact G4.8F
source signatures, execute route/pathway traces, promote walls, evaluate
wall-clock quality/cost, replay full NanoClustering, or claim method success.

The first NanoClustering G4.8 local validation readout is materialized at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_g4_8_local_validation_readout_gamma1e5_20260604/`.
It reuses the existing local-ablation seed runs rather than running Leiden
again. The readout splits local seeds into discovery seeds `0-3` and held-out
seeds `4-7`, then materializes endpoint-derived source-signature proxies and
held-out stratum checks. The result is
`local_validation_readout_materialized_with_heldout_fragility`: 20 of 23 pairs
preserve their expected held-out stratum and 3 do not. The fragile rows expose
the missing contract detail: one rare-ready row becomes latent-release/no-source
under held-out seeds, one strict-ready row becomes target-saturated, and one
target-saturated row sits on the strict-ready threshold boundary. This is not a
failure of the analog-surface premise, but it blocks any direct replay or
method-language promotion until seed/start-stratified validation contracts are
written.

The NanoClustering G4.8 seed/start-stratified validation contract is
materialized at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_g4_8_seed_start_validation_contract_gamma1e5_20260604/`.
It converts the readout into explicit execution lanes. The result is
`seed_start_validation_contract_ready_with_boundary_lanes`: 15 stable-lane
rows, 5 conditional-lane rows, and 3 boundary-lane rows. Stable lane contains
2 strict-ready rows plus target-saturated, latent-release, no-release, and
coupled-failure controls. Conditional lane contains 3 strict-ready rows and
2 rare-ready rows that can be used only under their listed allowed start
conditions. Boundary lane isolates the rare-ready-to-latent-release,
strict-ready-to-target-saturation, and target-saturated-to-threshold-ready
cases as diagnostic controls. This contract closes the immediate ambiguity from
the readout, but it is still not route/pathway evidence, wall-clock
quality/cost, full NanoClustering replay, or method evidence.

The NanoClustering G4.8 local validation execution contract is materialized at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_g4_8_local_validation_execution_contract_gamma1e5_20260604/`.
It consumes the seed/start contract and freezes the next validation unit
surface. The result is
`local_validation_execution_contract_ready_stable_primary`: primary validation
is limited to the 15 stable-lane pairs across all five start conditions
(`75` primary units), while conditional rows are preserved only as `16`
secondary allowed-start units and boundary rows only as `10` diagnostic
allowed-start units. All 11 execution-contract gates pass. This moves the
current boundary from pair classification to execution-unit selection, but it
still does not run Leiden or support route/pathway, wall-clock quality/cost,
full NanoClustering replay, or method evidence.

The NanoClustering G4.8 primary stable limitation readout is materialized at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_g4_8_primary_stable_limit_readout_gamma1e5_20260604/`.
It consumes the execution contract and inspects only the 75 stable primary
units. The result is
`primary_stable_limit_readout_ready_scoped_ready_signal`: ready evidence is
real but narrow, with 2 ready pairs and 10 ready partial-release units. The
remaining 65 units are existing-Leiden limitation/control evidence: 25
target-saturated no-handle units, 20 latent-release-without-original-
coassigned-source units, 10 hard no-release units, and 10 coupled direct/bridge
failure units. All 10 readout gates pass. This clarifies what Leiden+CPM is
doing under the fixed local surface, but it remains limitation cartography
rather than route/pathway, wall-clock quality/cost, full NanoClustering replay,
or method evidence.

The NanoClustering G4.8 pathway/wall readiness audit is materialized at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_g4_8_pathway_wall_readiness_audit_gamma1e5_20260604/`.
It consumes the primary stable limitation readout and separates a scoped
pathway-probe entry gate from wall-claim evidence. The result is
`pathway_probe_ready_for_scoped_candidates_wall_claim_closed`: the two ready
pairs and 10 ready units pass a local tri-endpoint contrast precheck, so they
can feed a predeclared Stage 2A pathway-probe design. The other 65
limitation/control units remain false-positive controls. Wall claims stay
closed because accepted distinct basin-pair relation evidence, route-family
evidence, direct-path availability, objective debt/recovery, polish reversion,
support incompatibility, and measured post-route endpoint assignment are still
missing. All 11 readiness gates pass.

The NanoClustering G4.8 scoped pathway-probe contract is materialized at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_g4_8_scoped_pathway_probe_contract_gamma1e5_20260604/`.
It consumes the readiness audit and freezes the tiny Stage 2A execution surface
without running routes. The result is
`scoped_pathway_probe_contract_ready_wall_claim_closed`: only `local_pair_009`
and `local_pair_012` are route-probe candidates. Their 10 start-conditioned
units produce 30 predeclared route-plan rows across bridge-release
interpolation, direct-dependency collapse guard, and drop-both collapse guard
families. The other 65 limitation/control units stay as false-positive guards,
not route-execution rows. Required future fields are route trace rows,
objective value/debt/recovery, endpoint assignment, support distance, polish
reversion, and support incompatibility. All 9 contract gates pass. This is a
pathway-probe contract only, not route/pathway execution, wall promotion,
quality/cost, full NanoClustering replay, or method evidence.

The NanoClustering G4.8 scoped pathway-probe trace is materialized at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_g4_8_scoped_pathway_probe_trace_gamma1e5_20260604/`.
It consumes the scoped contract and executes only those 30 route-plan rows on
local induced graphs. The execution expands the plan into 130 predeclared
fraction-step configurations and 1,040 seed-step trace rows. The result is
`executed_nanoclustering_g4_8_scoped_pathway_probe_trace`: all 10
bridge-release interpolation contracts and all 10 drop-both collapse guards
reach their expected final anchors for every seed, while the 10
direct-dependency collapse guards are partial source-to-expected transitions.
Bridge-release traces expose intermediate unknown endpoints in 10 contracts and
27 seed-route summaries, but all final steps reconcile with expected anchors.
All 9 trace gates pass. This materializes route trace, objective debt/recovery,
endpoint assignment, support distance, polish reversion, and
support-incompatibility fields. It still does not promote a wall, evaluate
quality/cost value, replay full NanoClustering, or claim method evidence.

The NanoClustering G4.8 scoped pathway wall-evidence audit is materialized at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_g4_8_scoped_pathway_wall_evidence_audit_gamma1e5_20260604/`.
It consumes the executed trace and classifies wall readiness without opening new
execution. The result is
`wall_evidence_audit_pathway_trace_candidate_wall_claim_closed`: all 10 primary
bridge-release contracts are all-seed source-to-expected transitions, so they
are pathway-trace wall-audit candidates. However, 0 contracts are wall-ready:
the direct-dependency guard is partial in all 10 contracts, the drop-both guard
collapses as expected in all 10 contracts, intermediate unknown/support-
incompatibility evidence appears in 27 seed routes, objective recovery is not
uniform under the contract-level criterion, and accepted direct-path evidence
is still missing. All 9 audit gates pass. This audit narrows the next valid
question to the primary bridge-release traces; it does not promote a wall,
evaluate quality/cost, replay full NanoClustering, or claim method evidence.

The NanoClustering G4.8 primary bridge-release pathway-shape audit is
materialized at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_g4_8_primary_bridge_release_pathway_shape_gamma1e5_20260604/`.
It consumes only the primary bridge-release rows from the executed trace and
asks whether the missing direct-path evidence can be clarified without route
broadening. The result is
`primary_bridge_release_pathway_shape_audit_direct_candidates_seed_level_wall_closed`:
all 80 primary seed-routes reach the expected drop-bridge target and retain the
direct pair edge throughout, and 53 seed-routes remain on known anchors while
transitioning. This is only seed-level direct-path candidate evidence: 0 of 10
contracts have all-seed known-anchor direct-path acceptance because every
contract contains at least one intermediate unknown/support-incompatible
seed-route. Objective debt appears in all 80 seed-routes, objective recovery
appears in only 8 seed-routes, and 0 contracts have all-seed recovery.
`local_pair_009` and `local_pair_012` should now be treated as two different
pathway-shape regimes rather than one wall-positive class: `local_pair_009` is
step-3 debt-without-recovery, while `local_pair_012` is mostly step-2 with
partial recovery. The next valid test is therefore a predeclared direct-path
acceptance contract, not pair broadening, wall promotion, quality/cost
evaluation, full NanoClustering replay, or method evidence.

The NanoClustering G4.8 direct-path acceptance contract is materialized at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_g4_8_direct_path_acceptance_contract_gamma1e5_20260604/`.
It consumes the primary bridge-release pathway-shape audit and fixes the
D1-D9 acceptance rules before any new execution: primary scope, physical
direct-edge retention, source-start known anchor, expected target reached, no
intermediate unknown endpoint, no support-incompatibility flag, all-seed
contract acceptance, objective recovery kept separate, and wall claims closed.
The result is
`direct_path_acceptance_contract_materialized_current_evidence_contract_level_closed`:
the current trace contains 53 seed-level direct-path candidates, but 0 of 10
contracts pass strict all-seed D1-D7 acceptance because every contract contains
at least one intermediate unknown/support-incompatible seed-route. Objective
recovery is reported separately and cannot promote direct-path or wall
language. All 9 contract gates pass. The next executable test, if opened, must
evaluate only this D1-D9 contract over the two separated regimes.

The NanoClustering G4.8 cross-seed endpoint-atlas audit is materialized at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_g4_8_cross_seed_endpoint_atlas_gamma1e5_20260604/`.
It consumes the same primary bridge-release trace and reclassifies
same-seed `unknown_new_endpoint` labels against pair-level endpoint signatures.
The result is
`cross_seed_endpoint_atlas_reclassifies_same_seed_unknowns_no_true_novel_wall_closed`:
all 27 same-seed unknown rows map to signatures already known elsewhere in the
same local pair, 0 are true pair-level novel endpoints, and all 10 primary
contracts have no true-novel unknown endpoint. This is a material correction to
the direct-path interpretation. D5 in the direct-path contract is a strict
same-seed anchor-consistency guard; it is not a true-novel-endpoint test and
should not be used as basin-topology evidence by itself. `local_pair_009`
shows a clean pair-level atlas path: step 2 collapses to one source signature
for all seeds, then step 3 reaches the drop-bridge target for all seeds.
`local_pair_012` is mixed at step 2, but every same-seed unknown signature is
still known elsewhere as source or target. The next direct-path design must
split same-seed anchor consistency from pair-level endpoint-atlas continuity.

The NanoClustering G4.8 dual-axis direct-path contract is materialized at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_g4_8_dual_axis_direct_path_contract_gamma1e5_20260604/`.
It splits the v1 readout into Axis A and Axis B. Axis A preserves strict
same-seed anchor consistency and stays closed at the contract level: 53 of 80
seed-routes pass, but 0 of 10 contracts pass. Axis B tests pair-level
endpoint-atlas continuity and is open on this scoped evidence: 80 of 80
seed-routes and 10 of 10 contracts pass with 0 true-novel pair-level endpoints.
This is a topology-contract correction only. Objective recovery remains a
separate wall-readiness field (8 recovery seed-routes, 0 all-seed recovery
contracts), and wall readiness remains 0 of 10. The next valid test is a
predeclared fresh panel or seed-anchor rotation for Axis B continuity.

The NanoClustering G4.8 Axis B seed-anchor rotation audit is materialized at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_g4_8_axis_b_seed_anchor_rotation_audit_gamma1e5_20260604/`.
It validates the strongest part of the Axis B reinterpretation but exposes a
route-level caveat. Across full-pair, leave-start-out, leave-seed-out, and
leave-seed-and-start-out endpoint vocabularies, all 27 same-seed unknown rows
remain pair-level known and 0 become true-novel endpoints. Baseline and
leave-start-out route continuity pass 80 of 80 routes and 10 of 10 contracts,
but leave-seed-out and leave-seed-and-start-out pass only 78 of 80 routes and 8
of 10 contracts because two `local_pair_009` seed-0 source-start signatures
lack off-seed known support. This is a source-start singleton caveat, not an
unknown-endpoint, target, wall, quality/cost, or method result. The next bounded
test should record source-start support separately from interior endpoint
continuity.

The NanoClustering G4.8 Axis B source-start support contract is materialized at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_g4_8_axis_b_source_start_support_contract_gamma1e5_20260604/`.
It confirms the intended split. Source-start support passes in the full-pair
and leave-start-out modes (80 of 80 routes, 10 of 10 contracts), but
leave-seed-out and leave-seed-and-start-out preserve two `local_pair_009`
seed-0 singleton caveats (78 of 80 routes, 8 of 10 contracts). Post-start
interior continuity passes in every mode: 80 of 80 routes and 10 of 10
contracts, with 0 post-start true-novel endpoint routes. Therefore the next
fresh Axis B panel should be designed around separate fields for source-start
support, post-start endpoint continuity, target-final continuity, and direct
edge retention. Interior endpoint evidence must not be used to repair
source-start singleton caveats.

The fresh Axis B panel contract is now materialized at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_g4_8_fresh_axis_b_panel_contract_gamma1e5_20260604/`.
It keeps `local_pair_009` and `local_pair_012` as calibration only, and moves
fresh evidence to not-yet-routed ready-like rows. The first-pass fresh slice is
36 route rows: 16 conditional ready-like rows from `local_pair_003`,
`local_pair_005`, `local_pair_007`, `local_pair_014`, and `local_pair_016`,
plus 20 control rows from `local_pair_002`, `local_pair_008`,
`local_pair_013`, and `local_pair_022`. The contract also records a critical
limit: there are 0 fresh stable positive pairs beyond calibration in the
current surface. All route-plan rows require source-start support, post-start
endpoint continuity, target-final continuity, and direct-edge retention as
separate measurements, and all wall/method/quality-cost claims remain closed.

The first-pass readout contract is materialized at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_g4_8_fresh_axis_b_first_pass_readout_contract_gamma1e5_20260604/`.
It makes the first-pass result a conditional ready-like Axis B screen rather
than a stable-positive generality test. The claim ladder allows only levels
0-2 in the first pass: execution/field completion, ready-like target-final
continuity, and ready/control separation. Source-start support stability,
seed/start rotation robustness, basin/pathway generality, branch generality,
and direct-dependent generality remain closed. Controls must be inspected before
ready-like positives, and `local_pair_003` is flagged as route-local because it
has only one allowed start.

The first-pass trace is now materialized at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_g4_8_fresh_axis_b_first_pass_trace_gamma1e5_20260604/`.
It executes the 36 rows as bridge-release interpolation traces and records
1,440 trace rows over 8 seeds. The key correction from the trace is that
target-final continuity must be exclusive: a final endpoint that is also a
guard anchor such as `drop_both` is not a clean bridge-release target. With this
readout, all controls close and all execution/readout gates pass. `local_pair_014`
is the clean first-pass ready-like candidate, `local_pair_005` is only partial,
and `local_pair_003`, `local_pair_007`, and `local_pair_016` fail post-start
endpoint continuity. This is still a route-local screen, not a wall claim.

The exclusive-target contrast audit is materialized at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_g4_8_first_pass_exclusive_target_contrast_audit_gamma1e5_20260604/`.
It reads the executed first-pass traces without rerunning Leiden and separates
288 route rows into 56 exclusive bridge-target passes, 8 source/target signature
collapses, 80 guard-anchor collapses, and 144 intermediate unknown endpoints.
All 5 audit gates pass. This fixes the next object-level audit scope:
`local_pair_014` is the clean candidate, `local_pair_005` is the partial-collapse
boundary case, and controls plus post-start-failure pairs stay out of the
positive set.

The first-pass symmetric endpoint-object audit is materialized at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_g4_8_first_pass_symmetric_endpoint_objects_audit_gamma1e5_20260604/`.
It is bounded to `local_pair_014` and `local_pair_005`, and materializes 9
endpoint-object rows plus 64 object relation rows. `local_pair_014` has one
exclusive final target object and 32/32 clean source-to-target object relations.
`local_pair_005` has 24 clean relations but 8 source/target object collapses,
and both of its final signatures are mixed boundary objects. All 6 gates pass.
Therefore only `local_pair_014` remains a positive object-level candidate;
`local_pair_005` should be retained as the boundary negative/partial control.

The first-pass wall/pathway-readiness audit is materialized at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_g4_8_first_pass_wall_pathway_readiness_audit_gamma1e5_20260604/`.
It reads the executed first-pass trace plus object audit rows and does not rerun
Leiden. `local_pair_014` is the only pathway-probe candidate: 32/32 routes
retain the direct edge, follow the predeclared bridge-release schedule, reach
the exclusive target at step 2, and avoid post-start unknown/ambiguous/support
incompatibility. The shape is
`clean_known_anchor_step2_with_objective_debt_without_recovery`, which means
there is route-local debt but no accepted recovery. Therefore no wall claim is
opened. `local_pair_005` stays a boundary control. All 6 gates pass, and the
next gate must add independent direct-path evidence, accepted recovery, and
independent wall evidence.

The first-pass `local_pair_014` pathway-probe contract is materialized at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_g4_8_first_pass_014_pathway_probe_contract_gamma1e5_20260604/`.
It is a design contract, not evidence. It predeclares 16 route-plan rows:
8 positive `local_pair_014` rows across recovery-loop and direct-only
target-availability families, and 8 matched `local_pair_005` boundary-control
rows. The acceptance rules require independent direct-path availability,
accepted recovery after the objective-debt minimum, boundary no-leak, and
wall-claim closure. All 7 gates pass, and all 16 route rows are explicitly
marked `new_schedule_support_required`.

The executed first-pass `local_pair_014` pathway-probe trace is materialized at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_g4_8_first_pass_014_pathway_probe_trace_gamma1e5_20260604/`.
It executes exactly the 16 contract rows as 88 route-step configs and 704 trace
rows over 8 seeds. With endpoint identity read at the object level,
`local_pair_014` accepts 32/32 direct-only target-availability seed-routes and
32/32 recovery-loop seed-routes. `local_pair_005` stays closed as the boundary
control with 0/64 positive leaks and all 8 boundary guards closed. All 9 trace
gates pass. The important interpretive detail is that exact-anchor coincidence
between `original_source_anchor` and `drop_direct_guard_anchor` is treated as a
source-like endpoint object, not as a target/pathway failure. This is
pathway-probe evidence only; wall, method, full-replay, and quality/cost claims
remain closed.

The first-pass `local_pair_014` primitive wall-evidence audit is materialized at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_g4_8_first_pass_014_wall_evidence_audit_gamma1e5_20260604/`.
It joins the accepted direct-only and recovery-loop routes by identical
`start_condition` and seed. Under this paired unit, `local_pair_014` passes
32/32 wall seed units: direct-only routes expose the exclusive target object
from a source-like endpoint object, recovery-loop routes move source-like to
exclusive target to source-like after bridge support is restored, and objective
debt/recovery is accepted in every unit. The matched `local_pair_005` boundary
guard remains closed at 32/32 units. All 9 gates pass, and
`primitive_wall_evidence_ready_pairs == ["local_pair_014"]`. This opens only a
local primitive object-level wall-evidence claim. It does not establish
generality, exact wall-location localization beyond the coarse schedules,
quality/cost value, full-replay behavior, or method success.

The synthetic G4.9 primitive-wall mechanism demo is then materialized at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_variable_pair_synthetic_g4_9_primitive_wall_demo_v1_20260604/`.
It converts the `014` object-level read into a predeclared five-case small
Leiden+CPM panel: one balanced positive case and four boundary controls. The
positive case reproduces the source-like to exclusive-target to source-like
relation in 32/32 paired wall seed units across 4 starts and 8 seeds. The
controls all close with 0 wall-ready units and separate target saturation,
target absence/source lock, and nonrobust partial target opening. All 7 gates
pass. This is a synthetic explanation scaffold only, not NanoClustering
generality, exact wall localization, quality/cost, full-replay, method, or
algorithm evidence.

G4.9A then localizes that synthetic positive at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_variable_pair_synthetic_g4_9a_parameter_localization_v1_20260604/`.
It maps 75 predeclared cells across three 2D slices around the positive point:
direct/pair-bridge, pair-bridge/bridge-host, and direct/bridge-host. The center
cell reproduces G4.9 in all three plane duplicates at 32/32 ready units. The
map finds 6 full wall-ready cells, 20 partial/fragile wall-ready cells, and 49
nonready cells. The closed cells separate target-absent/source-locked,
target-saturated, and nonrobust or mixed boundary regimes. All 7 gates pass.
The conclusion is bounded: the synthetic primitive wall positive is not a
single-cell tuning artifact, but the ready surface is still a local mechanism
regime rather than general basin evidence.

The next real-data wall-localization contract is now materialized at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_g4_8_first_pass_014_wall_localization_contract_gamma1e5_20260604/`.
It translates the G4.9A `W/w/T/N/P` boundary vocabulary back to the accepted
`local_pair_014` real-data primitive wall candidate. The contract predeclares
16 route-plan rows and 192 fraction-step rows: `014` descent/ascent scans and
retained `005` boundary guards, with direct support retained and bridge support
scanned through `1.00,0.95,0.90,0.85,0.80,0.75,0.625,0.50,0.375,0.25,0.125,0.00`
plus the reverse sequence. All 7 contract gates pass. This is design-only; it
does not execute Leiden, locate the wall, retune a threshold, or open
generality, method, quality/cost, or full-replay claims.

The contract has now been executed at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_g4_8_first_pass_014_wall_localization_trace_gamma1e5_20260605/`.
The runner materializes 1,536 fraction-level rows, 128 route-scan rows, and 64
paired seed-start localization rows. The strict G4.9A vocabulary classifies
`local_pair_014` as partial/fragile: 1/32 positive seed-starts are strict `W`
and 31/32 are `P`; the retained `local_pair_005` boundary has 0 positive W-like
leaks. All 7 execution gates pass. This execution does not close wall
localization; it demonstrates that strict no-unknown endpoint-object acceptance
is not the right final readout for this real-data case.

The transition-band audit at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_g4_8_first_pass_014_wall_localization_transition_band_audit_gamma1e5_20260605/`
then re-reads the same trace without rerunning Leiden. It splits the 32
positive seed-start units into 1 strict interpretable wall interval, 30 monotone
intermediate transition bands, and 1 bounded nonmonotone transition band. All
32/32 positive seed-starts are bounded source-target transition bands, and the
`005` boundary has 0 positive-target routes and 0 positive-target steps. All 7
audit gates pass. The current design should therefore treat the wall as a local
transition interval with intermediate endpoint objects, not as a single clean
fraction point.

The signature-identity audit at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_g4_8_first_pass_014_wall_localization_signature_identity_audit_gamma1e5_20260605/`
then checks whether row-local endpoint assignments are stable object identities.
They are not. In `014`, 152 row-local unresolved rows split into 98 rows whose
signature is known elsewhere as a source-like/direct-guard object and 54 rows
that remain true signature-level unresolved intermediate objects across two
recurrent signatures. In `005`, all 204 row-local unresolved rows resolve to
signatures known elsewhere, leaving 0 signature-level unresolved boundary rows.
All 7 audit gates pass. This means the next method-design unit should be
signature identity and role-stability typing inside the transition band, not
more fraction thresholding.

The role-stability audit at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_g4_8_first_pass_014_intermediate_role_stability_audit_gamma1e5_20260605/`
then types those signatures by local node roles. The 12-node graph is expressed
as pair roles `L/R` plus bridge roles `B1`-`B10`. The six positive endpoint
signatures separate into a target anchor, two source-like anchors, one
hidden-known source/guard intermediate (`b7761471acbf`), one unresolved
pair-coassigned intermediate (`ca947e9fbe61`), and one unresolved pair-separated
bridge-reassignment intermediate (`531aa99db869`). The two unresolved
intermediate signatures appear in 44/64 positive seed-route rows; the
hidden-known source/guard signature appears in 50/64. All 7 gates pass. The
transition band is therefore typed and recurrent, but still local to `014`.

The transfer screen at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_g4_8_first_pass_014_role_pattern_transfer_screen_gamma1e5_20260605/`
then asks whether that role pattern appears in other already materialized
first-pass pairs. It screens 9 pairs and 38 endpoint signatures without reruns
or a new fraction sweep. `014` is the only clean first-pass scaffold and there
are 0 non-014 positive transfer candidates. The strongest next mechanism
question is `local_pair_016`, because it is strict-ready and has source/target
plus unresolved bridge-reassignment structure but still fails post-start
continuity. `005` remains a boundary guard, while `008`, `022`, and `002` are
closed-control analogs. This moves the next work from "find another positive"
to "explain why the closest strict-ready analog remains blocked."

The `016` continuity-block audit at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_g4_8_first_pass_016_continuity_block_audit_gamma1e5_20260605/`
then localizes that blockage. All 24 audited `016` routes pass source-start
support and final exclusive-target support, but every route has a single
post-start continuity block at step 2, bridge fraction 0.75. The recurrent
signature is `aeb59ab537e6`, typed as an unresolved pair-separated
bridge-reassignment: `L+B1` remains together while `R` is separated. Its support
distance is tied to original, drop-bridge, and drop-direct anchors at 0.0444.
Thus `016` is not evidence that the scaffold is absent; it is evidence that the
current readout treats a typed transient intermediate as a blocker.

The transition-evidence synthesis at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_g4_8_first_pass_transition_evidence_synthesis_gamma1e5_20260605/`
then collects the evidence ledger into pair, claim, and definition rows. The
evidence surface now supports five definition decisions: endpoint identity
should use endpoint signatures plus role typing rather than row-local labels;
real-data wall evidence should be allowed to be a bounded transition band rather
than a point wall; typed transient intermediate semantics must be decided before
more execution; role analogs are diagnostic until endpoint exclusivity,
pathway semantics, and guard closure all hold; and every broadened rule must
preserve `005` and closed controls as non-positive. This synthesis keeps `016`
diagnostic-only and makes the next gate a predicate-design task, not a new
search.

The typed-transient predicate screen at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_g4_8_first_pass_typed_transient_predicate_screen_gamma1e5_20260605/`
then materializes that predicate-design task. It compares four criteria on the
same existing data surface. P0, the strict all-positive baseline, accepts only
`014`. P1, the guarded single-step separated bridge-reassignment transient
candidate, accepts only `016` with 0 boundary/control/rare-ready guard leaks.
P2, endpoint-only broadening, accepts all 9 screened pairs and leaks `005`, all
4 controls, and both rare-ready analogs. P3, role-analog-only broadening,
accepts `016`, `007`, `008`, `002`, and `003`, leaking 2 controls and both
rare-ready analogs. Thus the current data supports a guarded typed-transient
definition candidate, not a relaxed endpoint-only or role-analog definition and
not a positive wall or method claim.

The `016` transient semantic-validation audit at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_g4_8_first_pass_016_transient_semantic_validation_gamma1e5_20260605/`
then checks what that candidate means. It finds a stable route shape: 24/24
routes pass through the same step-2 signature (`aeb59ab537e6`) at bridge
fraction 0.75, then persist at the drop-bridge target from steps 3-5. The
signature is a typed separated bridge-reassignment gateway, so it is not plain
seed noise. But it is equidistant to original, drop-bridge, and drop-direct
anchors in all 24 routes and objective debt accumulates into the target with no
recovery from the minimum. The classification is therefore
`recurrent_typed_transition_gateway_candidate_not_endpoint_or_positive_wall`.
The next valid gate is persistence/reversibility, not broad localization:
finer fractions around the step-2 saddle or a reverse target-to-source trace
under the same boundary/control guards.

## Immediate Next Work

The next work should use the G4.9 scaffold to prevent another narrow
single-pair loop. The current evidence has one real-data primitive wall object
(`014`), one synthetic mechanism reproduction, and one synthetic localization
map. The wall-localization contract has now been executed and audited. It still
lacks real-data generality and exact interpretation of the intermediate
signatures outside this single local graph.

### Thread 1: Generality

- audit why `local_pair_014` is clean while `local_pair_005` is partial, using
  anchor exclusivity, source/target signature collapse, and start/seed
  dependence as the next evidence fields;
- repeat the paired direct/recovery wall audit over additional clean
  object-level candidates if the endpoint-object atlas exposes them;
- use `local_pair_005` and the G4.9 boundary controls as false-positive guards:
  target saturation, target absence/source lock, and nonrobust partial target
  opening must stay non-positive.
- translate the G4.9A boundary vocabulary back to real-data readouts: do not
  collapse partial target opening, source lock, and target saturation into a
  single "failed" category.
- use the typed `014` transition-band signatures as the reference pattern for
  the next candidate-selection pass. The first transfer screen found no non-014
  positive transfer candidate, so the next gate should focus on `local_pair_016`
  as a strict-ready continuity-blocked analog before any new localization
  escalation. The first `016` audit localized the block to one step-2
  bridge-reassignment transient, so the next definition decision is whether
  typed transient intermediates are pathway evidence or continuity blockers.
  The semantic validation now classifies `016` as a recurrent typed
  transition-gateway candidate, not an endpoint or wall. The next gate should
  test persistence or reversibility of the step-2 saddle; do not run another
  broad fraction sweep unless it answers that mechanism question.

This tests whether current NanoClustering evidence can become a general
basin-candidate surface.

### Thread 2: Demo

- keep the variable-pair synthetic G4.9 runner fixed as the current primitive
  wall explanation scaffold;
- keep G4.9A as the current parameter-localization map; do not broaden it into
  policy or threshold search unless a new mechanism question is stated;
- treat the 014 wall-localization contract as the next execution surface: run
  the 16 route rows and 192 fraction steps, then audit transition intervals by
  start and seed before any wall-location language;
- keep the G4.1 strict-crossing interpretation separate from the earlier
  target-reconstruction trace;
- keep the frozen G4.3 `bridge_context_release_without_pair_merge` handle class
  unchanged;
- treat `pair_merge_plus_context_release` as a lower-priority diagnostic class
  because its strict crossings are lower-quality than the source endpoint in
  this surface;
- keep the G4.5 `neutral_release_with_direct_support_v1` selector frozen;
- keep the G4.6
  `restart_then_g4_5_selector_then_one_g4_3_handle_v1` schedule frozen;
- treat the G4.7 independent stress failure as a boundary result: controls stay
  suppressed, but shifted positives leave the bridge-release opportunity regime;
- keep the G4.8A opportunity-regime metrics frozen;
- treat G4.8B as a constructive boundary failure: the no-leak side holds, but
  fresh ready and pair-only cells collapse into target saturation;
- treat G4.8C as the first construction cartography result: ready opportunity
  exists, but it sits on a narrow pair-bridge/bridge-host balance surface;
- treat G4.8D as the first 2D balance result: ready opportunity is a sparse
  diagonal ridge, with target saturation on one side and nonrobust coexistence
  on the other;
- treat G4.8E as a diagonal-ridge refinement result: the ready surface is not
  a continuous ridge or finite-width band, but a stable centerline resonance
  lattice with alternating target-saturated and nonrobust centerline neighbors;
- treat G4.8F as the centerline signature audit: ready cells require the full
  8-source signature set, including single-side bridge sources, and
  source-neutral release; nonrobust cells have only two-side sources and are
  source-nonneutral; target cells expose no source;
- treat G4.8G as the fresh-context signature validation: the frozen
  construction-read rule holds across 4 fresh direct/host contexts and 52
  centerline cases;
- treat G4.8H as the bounded source-discovery smoke: the target-free rule
  recovers the exact release-source and ready-source sets over the materialized
  G4.8G endpoint pool without using target outcomes for decisions;
- treat G4.8I as the fresh discovered-source schedule panel: edge-mid
  direct/host contexts preserve the construction-read roles and the discovered
  sources drive schedule accounting without oracle source-signature reads;
- treat G4.8J as the first off-center failure-mode expansion: negative offsets
  correctly become nonrobust no-handle cases and positive offsets correctly
  become target-saturated no-source cases;
- treat the NanoClustering G4.8 source-condition analog screen as the first
  real-data local proxy check: strict-ready, rare-ready, target-saturated, and
  nonready controls exist under local ablation, but exact G4.8F source
  signatures are not materialized;
- treat the NanoClustering G4.8 frozen local analog validation panel as the
  fixed 23-pair execution surface: strict-ready, rare-ready, target-saturated,
  nonready, and coupled-failure controls are all included, but this is still
  design/materialization evidence only;
- treat the NanoClustering G4.8 local validation readout as the first
  held-out diagnostic over that surface: source-signature proxies are
  materialized, but 3 of 23 rows are seed/start fragile;
- treat the NanoClustering G4.8 seed/start validation contract as the current
  lane-classification boundary: stable lanes can feed primary local validation,
  conditional lanes can be used only under allowed start conditions, and
  boundary lanes remain diagnostic controls;
- treat the NanoClustering G4.8 local validation execution contract as the
  current execution-unit boundary: execute or simulate the 75 stable primary
  units first, and report the 16 conditional units plus 10 boundary units
  separately before any broad synthetic offset sweep or full NanoClustering
  replay;
- treat the NanoClustering G4.8 primary stable limitation readout as the
  current existing-Leiden limit map: the ready signal is 2 pairs and 10 units,
  while target saturation, latent release without original coassigned source,
  hard no-release, and coupled direct/bridge failure remain separate
  limitations;
- treat the NanoClustering G4.8 pathway/wall readiness audit as the current
  Stage 2A entry gate: only the two ready pairs can feed a tiny predeclared
  pathway-probe design, while wall claims remain closed until route traces,
  objective debt/recovery, polish reversion, support incompatibility, and
  measured post-route endpoint assignment exist;
- treat the NanoClustering G4.8 scoped pathway-probe contract as the current
  Stage 2A execution boundary: if executed, run only the 30 predeclared
  route-plan rows for `local_pair_009` and `local_pair_012`, keep the 65
  noncandidate units as false-positive guards, and require route trace,
  objective, endpoint, polish, and support fields before any wall language;
- treat the NanoClustering G4.8 scoped pathway-probe trace as the current
  Stage 2A evidence boundary: audit the 20 all-seed source-to-expected
  contracts, the 10 partial direct-dependency guard contracts, and the 27
  intermediate unknown seed-routes before any wall language; do not broaden to
  controls, quality/cost, full NanoClustering replay, or method claims;
- treat the NanoClustering G4.8 scoped pathway wall-evidence audit as the
  current wall-language boundary: no wall claim is open; the primary
  bridge-release pathway-shape audit has now separated seed-level direct-path
  candidates from intermediate unknown routes and objective debt/recovery
  behavior;
- treat the NanoClustering G4.8 primary bridge-release pathway-shape audit as
  the current pathway-shape boundary: physical direct-edge retention is
  confirmed, but accepted direct-path evidence remains 0 of 10 contracts;
- treat the NanoClustering G4.8 direct-path acceptance contract as the current
  direct-path boundary: D1-D9 rules are fixed, 53 seed-level candidates are
  preserved, and strict contract-level acceptance remains 0 of 10;
- treat the NanoClustering G4.8 cross-seed endpoint-atlas audit as the current
  topology boundary: same-seed unknown labels are not true novel endpoints in
  this trace. The next valid design work is to revise direct-path acceptance
  into two separate axes, same-seed anchor consistency and pair-level
  endpoint-atlas continuity, with objective recovery audited separately;
- report source availability, no-op mass, handle-run overhead, and
  restart-plus-handle unit ratios separately. Keep wall/pathway, quality/cost,
  and algorithm language closed until the regime conditions are characterized
  and then replayed on a fresh predeclared panel.

This tests whether the guiding premise can be demonstrated outside the large
data artifact chain.

## First Execution Notes

The first seed-anchor rotation audit is materialized at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_seed_anchor_rotation_audit_20260531/`.
It passes the count-level seed0-only check: all 18 non-seed0 Java/Rust anchors
have recurrent strong fragmentation candidates, with a minimum of 205 and a
median of 224 recurrent candidates per non-seed0 anchor. It does not pass the
stronger taxonomy check: seed0 T1 source-family recurrent recovery under
non-seed0 anchors ranges from 0.121951 to 0.341463. The next generality gate is
therefore symmetric endpoint objects, not wall/pathway or method claims.

The first symmetric endpoint-object audit is materialized at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_symmetric_endpoint_objects_20260531/`.
It builds all-seed endpoint overlap components from 84,216 endpoint-cluster
nodes and 513,667 retained overlap edges. The object surface is real: 8,053
objects have at least five-seed coverage and 7,588 have at least eight-seed
coverage. The strict seed0 T1 anchor-independent candidate share is only
0.341772, however, so the correct read is not seed-invariant taxonomy. It is a
stable-object versus anchor-local-fragment separation surface.

The first symmetric object decomposition is materialized at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_symmetric_object_decomposition_v1_20260531/`.
It decomposes 9,470 symmetric objects into 7,498 stable one-per-seed objects,
555 stable multi-cluster objects, 672 partial objects, and 745 anchor-local
fragments. Seed0 mapping non-promotion is now named: 52 source families are
anchor-independent candidates, 42 map to multi-cluster objects, 41 are partial,
27 are anchor-local fragments, and 4 are merged seed0-family objects. This
supports mechanism mining and stable-control selection, not taxonomy promotion.

The first tiny CPM demo seed sweep is materialized at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_tiny_cpm_demo_seed_sweep_20260531/`.
Plain Leiden + CPM over 100 seeds produces recurrent multi-endpoint behavior in
all four predeclared tiny graph families. The near-tie bridge case has two
equal-quality endpoints, while absorption, balanced split, and diffuse
fragmentation cases provide mechanism-specific recurring alternatives. This
supports baseline reproduction of the phenomenon in controlled form, but it is
not yet method improvement, wall/pathway, or quality/cost evidence.

The same tiny CPM demo now also freezes the controlled baseline universe: 17
baseline endpoints and 32 random-restart discovery-curve rows. The hard case is
`diffuse_fragment_star`: at budget 20, mean recurrent-endpoint recall is
0.854286 and all-recurrent hit rate is 0.275, whereas the near-tie bridge case
reaches full recurrent coverage by budget 20. Future method claims should be
judged against these fixed curves before returning to NanoClustering.

The first handle-conditioned tiny method probe is materialized at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_tiny_cpm_handle_method_v1_20260531/`.
It compares 15 mechanism-aware initial-membership handles against the frozen
restart baseline. It finds a real but narrow signal: maximum recurrent-recall
delta is 0.2575 and target-hit rate is 0.675. The hard case remains unresolved:
`diffuse_fragment_star` has budget-20 recurrent-recall delta -0.282857. The
right next method step is robustness and hard-case handle design, not a method
claim.

The endpoint replay diagnostic is materialized at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_tiny_cpm_endpoint_replay_v1_20260531/`.
It replays all 17 frozen endpoint signatures as direct Leiden initial
memberships over 10 replay seeds. All 15 recurrent endpoints replay to
themselves with same-endpoint rate 1.0. The six recurrent endpoints missed by
method v1 are also stable, including the three diffuse hard-case misses. The
diagnosis is therefore handle coverage rather than endpoint instability.

The replay-informed coverage-handle probe is materialized at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_tiny_cpm_handle_method_coverage_v2_20260531/`.
It appends mechanism-readable boundary-core and weak-pair tail-split handles to
the v1 registry. At budget 20, recurrent endpoint recall reaches 1.0 for all
four tiny graph families. The hard `diffuse_fragment_star` budget-20 delta
changes from -0.282857 in v1 to +0.145714 in coverage v2. This is evidence for
a small-demo candidate method surface, not an algorithm, pathway, cost,
quality, or NanoClustering claim.

## Next Stress Design: Coverage v2 Robustness

The next design should test whether `coverage_v2` is a robust mechanism signal
or a manually favorable endpoint-derived ordering. The result must stay inside
the small CPM demo surface until it passes these gates.

### Stress 1: Ordering Robustness

Question:

- Does `coverage_v2` still beat the frozen restart baseline when handle order
  is randomized, grouped differently, or adversarially delayed?

Protocol:

1. Reconstruct the frozen random-restart baseline distribution from
   `leiden_cpm_tiny_demo_seed_runs.csv`, not only the aggregate mean curve.
2. Execute the `coverage_v2` candidate registry under multiple order policies:
   canonical v2 order, within-family random permutations, v1-first then
   coverage handles, coverage-first then v1 handles, handle-type round robin,
   and adversarial delayed coverage handles.
3. Use the same budgets as the frozen baseline: 1, 2, 3, 5, 10, 20, 50, 100
   where the candidate count permits.
4. Report recurrent endpoint recall, all-recurrent hit rate, distinct frozen
   endpoint count, new endpoint count, and target-hit rate by family, budget,
   and order policy.

Pass:

- budget-20 recurrent recall remains at least the frozen restart mean for all
  four families under median order policy performance;
- `diffuse_fragment_star` remains positive against restart at budget 20 under
  most order policies;
- adversarial delayed coverage is recorded as a cost/ordering caveat rather
  than hidden.

Fail:

- the positive diffuse result only appears under the canonical hand-picked
  order. In that case the current method is an ordering heuristic, not a
  robust handle mechanism.

Primary artifact:

- `leiden_basin_tiny_cpm_coverage_order_robustness_v1_YYYYMMDD/`.

Execution:

- Materialized at
  `../../../research/consensus/results/adaptive_refinement/leiden_basin_tiny_cpm_coverage_order_robustness_v1_20260531/`.
- It reconstructs the restart baseline with 1,000 permutations and executes
  six order policies: canonical v2, v1-first, coverage-first, handle-type round
  robin, adversarial delayed coverage, and 200 random within-family orders.
- Budget-20 non-adversarial family-policy failures: 0.
- `diffuse_fragment_star` is positive against restart mean at budget 20 under
  all six policies; random order beat rate is 1.0.
- The caveat is early-budget ordering cost: delayed coverage is below restart
  for diffuse at budgets 5 and 10. This is a cost/order caveat, not endpoint
  instability and not an algorithm claim.

Detailed read:

- Budget-20 performance is not canonical-order luck: every tested policy
  reaches recurrent recall 1.0 for `diffuse_fragment_star`, and no
  non-adversarial family-policy row falls below the restart mean at budget 20.
- The order sensitivity lives earlier. `adversarial_delayed_coverage` falls
  below restart for absorption at budgets 5 and 10, for balanced split at
  budgets 2, 3, 5, and 10, and for diffuse at budgets 5 and 10.
- Canonical v2 still has early budget debt for balanced split at budgets 2, 3,
  and 5 and for diffuse at budget 5.
- Random within-family order is robust by budget 10 for balanced split and
  diffuse, but has nontrivial early variance: balanced split budget-2 beats the
  restart mean in only 0.405 of random orders, and diffuse budget-3 beats it in
  0.565 of random orders.
- First-hit diagnostics show that the replay-diagnosed missed endpoints are
  coverage-handle timing dependent: under adversarial delayed coverage,
  absorption endpoint03 first appears at attempt 20, balanced endpoints02/03 at
  attempts 19/20, and diffuse endpoints05/06/07 at attempts 18/19/20.

### Stress 2: Endpoint-Derived Dependency

Question:

- Can the coverage handles be generated from graph mechanism rules without
  using the frozen endpoint signatures as templates?

Protocol:

1. Build a blind mechanism-rule generator from the graph alone:
   boundary-core nodes are host nodes with direct high-weight contact to the
   ambiguous module; host-tail nodes are same-host nodes without that contact.
2. For absorption, generate small-module-to-host-boundary-core handles.
3. For balanced split, generate middle-module-to-host-boundary-core handles.
4. For diffuse fragmentation, generate weak-node pair handles by top host
   contact patterns plus boundary-core/tail split.
5. Compare blind-rule handles with `coverage_v2` hits, but do not let endpoint
   signatures choose the handles.

Pass:

- the blind-rule generator recovers the same qualitative miss classes as
  endpoint replay diagnosed: absorption-to-B boundary core, balanced
  A/B-boundary-core absorption, and diffuse weak-pair tail split;
- it reaches full or near-full recurrent endpoint recall on the tiny surface
  without copying endpoint signatures;
- failures are mechanism-readable, not unexplained candidate misses.

Fail:

- the only working handles are endpoint-template handles. Then `coverage_v2`
  remains a diagnostic oracle, not a plausible method.

Primary artifact:

- `leiden_basin_tiny_cpm_blind_rule_handle_probe_v1_YYYYMMDD/`.

Gate contract:

- **B1 construction independence:** the blind-rule candidate registry must be
  built before reading frozen endpoint manifests, replay diagnoses, endpoint
  signatures, endpoint ranks, or method-v1 missed-endpoint lists. Those files
  may be used only after candidate construction for evaluation.
- **B2 graph-evidence auditability:** every blind candidate must carry the
  graph-derived evidence that generated it: ambiguous module nodes, host
  candidate, boundary-core nodes, tail nodes if any, weak-node pair if any,
  and the edge/contact rule that selected them.
- **B3 qualitative miss-class recovery:** without endpoint templates, the
  generator must produce handles for the same mechanism classes Stress 1
  exposed as timing dependent: small-module-to-host boundary core, middle-
  module-to-host boundary core, and weak-pair tail split.
- **B4 discovery performance:** at budget 20, blind-rule handles must match or
  beat the frozen restart mean recurrent recall for all four tiny families.
  Strong pass requires `diffuse_fragment_star` recurrent recall 1.0; minimal
  pass requires positive diffuse delta over restart.
- **B5 first-hit sanity:** the replay-diagnosed hard endpoints should be hit by
  graph-rule candidates before the adversarial-delay attempts observed in
  Stress 1. This checks that the rule does not merely recreate the late
  coverage schedule.
- **B6 failure typing:** misses must be classified by mechanism-readable
  failure modes such as no boundary-core contrast, ambiguous host tie, weak-pair
  overgeneration, duplicate candidate, or polish collapse.
- **B7 claim gate:** even if B1-B6 pass, the result is still a tiny-demo
  candidate-method signal. It does not open optimizer-native wall/pathway,
  quality/cost, NanoClustering generality, or algorithm claims.

Execution:

- Materialized at
  `../../../research/consensus/results/adaptive_refinement/leiden_basin_tiny_cpm_blind_rule_handle_probe_v1_20260531/`.
- The script writes an 18-candidate blind-rule registry before reading frozen
  endpoint manifests, replay diagnoses, endpoint signatures, method-v1 misses,
  or coverage-v2 endpoint hits.
- Candidate rules include bridge-contact handles, small-module boundary-core
  handles, middle-module contact split and boundary-core handles, weak top-host
  handles, weak-pair tail-split handles, and control handles.
- All B1-B6 gates pass. At budget 20, every tiny family reaches recurrent
  endpoint recall 1.0. `diffuse_fragment_star` has delta 0.145714 over the
  frozen restart mean.
- All six replay-diagnosed hard endpoints are hit before their adversarial
  delayed-coverage first-hit attempts: absorption endpoint03 at attempt 3,
  balanced endpoints02/03 at attempts 2/3, and diffuse endpoints05/06/07 at
  attempts 4/6/7.
- This passes endpoint-derived dependency on the tiny surface. It does not
  imply algorithmic generality because the rule generator is still demo-family
  aware and has not been tested under handle-type ablation or mechanism
  variants.

### Stress 3: Handle-Type Ablation

Question:

- Which handle families actually carry the improvement?

Protocol:

Run fixed-budget comparisons for:

- v1 handles only;
- coverage handles only;
- v1 plus boundary-core only;
- v1 plus weak-pair tail-split only;
- blind-rule handles only;
- full coverage v2.

Metrics:

- recurrent endpoint recall by family and budget;
- first-hit budget for each recurrent endpoint;
- target-hit rate by handle type;
- handle-node count and group count as intervention-size diagnostics.

Pass:

- boundary-core handles explain the absorption and balanced missed endpoints;
- weak-pair tail-split handles explain the diffuse missed endpoints;
- the full v2 gain is decomposable rather than a black-box registry effect.

Fail:

- gains cannot be localized to mechanism-readable handle types. Then the method
  claim should stay closed.

Primary artifact:

- `leiden_basin_tiny_cpm_handle_ablation_v1_YYYYMMDD/`.

Next gate detail:

- Stress 3 should now decompose the successful blind-rule registry, not only
  the manually written `coverage_v2` registry.
- Required ablations are: bridge/contact controls only, boundary-core only,
  weak-pair tail-split only, weak top-host only, blind controls only, full
  blind-rule registry, and full blind-rule registry without controls.
- The key pass condition is explanatory localization: boundary-core handles
  should carry absorption and balanced gains, while weak-pair tail-split
  handles should carry the diffuse hard-endpoint gains.
- If full performance requires unrelated controls or exact registry ordering,
  then the method remains a registry effect rather than a mechanism-localized
  handle rule.

Gate contract:

- **A1 ablation registry integrity:** every ablation subset must be built from
  the already-materialized blind-rule candidate registry, not regenerated with
  endpoint feedback. Each subset must record included handle ids, excluded
  handle ids, handle types, and the reason for inclusion.
- **A2 baseline controls:** include at least these named subsets:
  `all_blind_rules`, `all_blind_rules_no_controls`, `boundary_core_only`,
  `weak_pair_tail_split_only`, `weak_top_host_only`, `bridge_contact_only`,
  `controls_only`, `boundary_core_plus_weak_pair`, and `drop_each_type_one_at_a_time`.
- **A3 mechanism-local positive attribution:** boundary-core subsets must hit
  the replay-diagnosed absorption and balanced hard endpoints earlier than
  adversarial delayed coverage, while weak-pair tail-split subsets must hit
  the diffuse hard endpoints earlier than adversarial delayed coverage.
- **A4 mechanism-local negative attribution:** removing boundary-core handles
  should specifically damage absorption and balanced hard-endpoint coverage;
  removing weak-pair tail-split handles should specifically damage diffuse
  hard-endpoint coverage. If removals do not have localized effects, the rule
  family is not causally attributed.
- **A5 control non-sufficiency:** controls-only, bridge-only, and weak-top-host-
  only subsets must not by themselves explain the hard endpoint gains. If they
  do, the interpretation shifts from boundary-core/weak-pair mechanisms to
  generic registry coverage.
- **A6 budget profile:** report first-hit attempt and recurrent recall at
  budgets 1, 2, 3, 5, 10, and 20. Strong pass requires the responsible subset
  to recover its targeted hard endpoints by budget 10; budget-20-only recovery
  is a cost caveat.
- **A7 interaction accounting:** if `boundary_core_plus_weak_pair` performs
  better than either family alone, record the interaction explicitly rather
  than assigning the gain to one handle type.
- **A8 claim gate:** passing ablation supports mechanism localization on the
  tiny demo only. It does not open optimizer-native wall/pathway, quality/cost,
  NanoClustering generality, or algorithm claims.

Gap review additions:

- **A9 schedule normalization:** ablations with fewer candidates must not gain
  merely because their candidates repeat more often within the same budget.
  Report both compacted schedules and slot-preserving schedules. In a
  slot-preserving schedule, removed handle types leave explicit empty/no-op
  slots so first-hit changes can be separated from schedule compression.
- **A10 target-scope scoring:** do not judge every subset by all-family recall.
  Score each subset on its declared target endpoint class: boundary-core on
  absorption and balanced hard endpoints, weak-pair tail-split on diffuse hard
  endpoints, bridge-contact on near-tie endpoints, and controls as negative
  controls. Global recall is secondary context only.
- **A11 exact endpoint attribution:** attribution must be endpoint-level, not
  family-level. For each hard endpoint, record which candidate ids hit it,
  whether the hit is exclusive to a handle type, first-hit attempt under each
  schedule, and whether it survives handle-type dropout.
- **A12 seed robustness of responsible handles:** for each responsible handle
  type, record candidate-by-seed outcome stability. A handle that hits the
  target endpoint only for one lucky method seed is a weaker mechanism signal
  than a handle that repeatedly polishes to the same endpoint.
- **A13 intervention-size caveat:** record handle-node count, initial group
  count, and touched ambiguous-node count for each candidate and subset. Do not
  claim cost advantage from this stress, but flag cases where the responsible
  subset works only by using much larger interventions.

Expected diagnostic outcomes:

| endpoint class | responsible subset | expected positive effect | expected negative effect |
| --- | --- | --- | --- |
| absorption endpoint03 | boundary-core only | first hit before adversarial attempt 20 | lost or delayed when boundary-core is removed |
| balanced endpoints02/03 | boundary-core only | first hit before adversarial attempts 19/20 | lost or delayed when boundary-core is removed |
| diffuse endpoints05/06/07 | weak-pair tail-split only | first hit before adversarial attempts 18/19/20 | lost or delayed when weak-pair tail-split is removed |
| near-tie endpoints01/02 | bridge-contact only | full recurrent recall by small budget | not part of boundary-core/weak-pair claim |

Primary outputs:

- `tiny_cpm_blind_rule_ablation_registry.csv`;
- `tiny_cpm_blind_rule_ablation_schedule.csv`;
- `tiny_cpm_blind_rule_ablation_attempts.csv`;
- `tiny_cpm_blind_rule_ablation_discovery.csv`;
- `tiny_cpm_blind_rule_ablation_first_hits.csv`;
- `tiny_cpm_blind_rule_ablation_attribution_matrix.csv`;
- `tiny_cpm_blind_rule_ablation_candidate_seed_stability.csv`;
- `tiny_cpm_blind_rule_ablation_gate_matrix.csv`;
- `tiny_cpm_blind_rule_ablation_report.md`.

Execution:

- Materialized at
  `../../../research/consensus/results/adaptive_refinement/leiden_basin_tiny_cpm_blind_rule_ablation_v1_20260531/`.
- It reuses the materialized blind-rule registry and evaluates 17 subsets under
  both compacted and slot-preserving schedules, yielding 2,720 scheduled
  attempts and 20 target-scoped attribution rows.
- All mechanism-localization gates pass, while the claim gate remains closed
  by design: 12 pass gates plus one closed claim gate.
- Positive attribution localizes to the intended handle classes. Boundary-core
  only hits all three absorption/balanced hard endpoints before adversarial
  delayed coverage, and weak-pair tail-split only hits all three diffuse hard
  endpoints before adversarial delayed coverage.
- Negative attribution also localizes. Dropping small-module boundary-core
  handles loses the absorption hard endpoint (0/1), dropping middle boundary-
  core handles loses the balanced hard endpoints (0/2), and dropping weak-pair
  tail-split handles loses the diffuse hard endpoints (0/3), under both
  schedule modes.
- Controls-only and weak-top-host-only do not explain hard-endpoint gains.
  Candidate-by-seed stability is deterministic for the responsible handle
  outcomes in this tiny panel, with minimum dominant endpoint rate 1.0.
- This supports tiny-demo mechanism localization of the blind-rule handles.
  It still does not promote route/pathway, wall, quality/cost,
  NanoClustering generality, or algorithm claims.

### Stress 4: Mechanism Variants, Not Parameter Sweeps

Question:

- Does the signal survive small predeclared mechanism variants, rather than
  only the exact demo graphs?

Purpose:

- separate mechanism generality from exact toy-graph overfitting;
- test both mechanism-preserving variants and mechanism-removed controls;
- keep failures interpretable instead of tuning them away.

Non-goals:

- no gamma sweep;
- no threshold sweep;
- no post-hoc graph editing after seeing endpoint or method results;
- no claim that the graph-rule generator is a general graph algorithm.

Panel construction rule:

- define all variants in a fixed manifest before any seed sweep;
- give each graph explicit role annotations such as host id, ambiguous module,
  boundary candidate, weak node, bridge node, and control node;
- record role annotations as evidence inputs, then also run role/name
  invariance checks with opaque node ids so graph rules do not depend on names
  such as `a`, `b`, `h`, `x`, or `m`;
- generate blind-rule handles from graph structure plus these declared demo
  roles, not from endpoint signatures, frozen manifests, or replay diagnoses;
- write the variant graph manifest and blind candidate registry before reading
  any endpoint-evaluation artifacts.

Variant catalog:

| mechanism family | variant id | mechanism state | expected baseline behavior | responsible rule | control read |
| --- | --- | --- | --- | --- | --- |
| near-tie bridge | `nt_symmetric_tie_anchor` | preserved anchor | bridge joins either side with near-equal quality | bridge-contact | calibration only, not generalization credit |
| near-tie bridge | `nt_light_bias_a` | perturbed preserved | side A dominates but side B may recur at lower rate | bridge-contact | success means calibrated dominance, not forced symmetry |
| near-tie bridge | `nt_hard_bias_a_control` | tie removed | one dominant bridge assignment | bridge-contact | failing to recover side B is correct |
| absorption triad | `ab_boundary_vs_diffuse` | preserved | compact boundary-core host competes with diffuse host contact | boundary-core | target hard endpoint should be reachable |
| absorption triad | `ab_symmetric_boundary` | preserved ambiguous | two compact host boundary cores compete | boundary-core | both host absorptions should be reachable if recurrent |
| absorption triad | `ab_diffuse_no_core_control` | mechanism removed | absorption, if any, lacks compact boundary core | boundary-core | boundary-core rule should not explain a hard endpoint |
| balanced split | `bs_equal_pull` | preserved | middle module has balanced host pull | boundary-core/contact split | split and collapse alternatives should recur |
| balanced split | `bs_light_asymmetry` | perturbed preserved | one host is favored but alternatives remain possible | boundary-core | success may be lower-rate but mechanism-readable |
| balanced split | `bs_single_host_dominant_control` | balance removed | middle module collapses to one host | boundary-core | missing the opposite collapse is correct |
| diffuse fragment | `df_one_pair` | preserved minimal | one weak-node pair creates a compact diffuse alternative | weak-pair tail-split | target pair endpoint should be reachable if recurrent |
| diffuse fragment | `df_two_pair` | preserved robust | two weak-node pairs create multiple diffuse alternatives | weak-pair tail-split | stronger version of the observed Stress 3 mechanism |
| diffuse fragment | `df_weak_module_separate_control` | fragmentation removed | weak module remains separate or coherent | weak-pair tail-split | weak-pair rule should not create a false diffuse claim |

Protocol:

1. Materialize the fixed variant graph manifest.
2. Compute mechanism-purity features for each graph: local contact mass, cut
   gap, host dominance, boundary-core concentration, weak-pair concentration,
   and decoy-control matching fields.
3. Build blind-rule candidate registries from graph structure and declared demo
   roles before endpoint-evaluation inputs are read.
4. Repeat candidate construction with opaque node ids and role/name
   permutations to test whether the same candidate classes are generated.
5. Write a phase-lock artifact that hashes the graph manifest, mechanism
   features, candidate registry, role-invariance rows, and runner config.
6. Run ordinary Leiden + CPM seed sweeps for each variant with the predeclared
   gamma and seed count.
7. Freeze recurrent endpoint manifests and restart discovery curves per
   variant using the predeclared recurrence threshold and endpoint inclusion
   rules.
8. Replay frozen recurrent endpoints as direct initial memberships to separate
   endpoint instability from handle failure.
9. Run fixed-budget handle attempts using the same budgets as Stress 3:
   1, 2, 3, 5, 10, and 20.
10. Score only target-scoped mechanism classes:
   bridge-contact on bridge alternatives, boundary-core on absorption/balanced
   hard endpoints, and weak-pair tail-split on diffuse hard endpoints.
11. Report compacted and slot-preserving schedules so smaller candidate sets do
   not gain hidden repetition budget.
12. Record shadow quality and cost fields for every run, but keep them out of
    pass/fail decisions in this stress.
13. Type every failure as one of: mechanism removed, tie broken, baseline did
   not expose recurrent alternatives, rule did not generate a candidate,
   candidate generated but polished to the wrong endpoint, or schedule/cost
   miss.

Gate contract:

- **V1 fixed panel integrity:** every variant must be declared in the manifest
  before execution, with gamma, seed count, role annotations, mechanism state,
  and expected control read.
- **V2 baseline reproduction:** mechanism-preserving variants must expose at
  least two recurrent Leiden + CPM endpoints or be marked
  `baseline_no_recurrent_alternative`. Mechanism-removed controls are allowed
  to collapse to one endpoint.
- **V3 construction independence:** blind-rule candidate registries must be
  written before frozen endpoint manifests, replay diagnoses, endpoint ranks,
  or method-hit rows are read.
- **V4 graph-evidence auditability:** every generated candidate must carry the
  graph evidence that selected it: role nodes, contact weights, boundary-core
  nodes, tail nodes, weak pairs, and the selection rule.
- **V5 mechanism-preserved positive attribution:** for each mechanism family,
  at least one non-anchor mechanism-preserving variant must recover its target
  hard endpoints by budget 20; strong pass requires budget 10.
- **V6 mechanism-removed specificity:** controls must not be counted as method
  failures merely because target endpoints disappear. They fail only if the
  method fabricates a positive target claim when the mechanism was removed.
- **V7 target-scope scoring:** global recurrent recall is context only.
  Variant claims are scored on their declared target mechanism class.
- **V8 schedule normalization:** compacted and slot-preserving schedules must
  both be reported for every subset and variant.
- **V9 endpoint-level attribution:** each target hit must name the endpoint id,
  candidate id, handle type, first-hit attempt, schedule mode, and whether the
  same target survives the relevant handle-type dropout.
- **V10 control non-sufficiency:** bridge-only, weak-top-host-only, and
  controls-only candidates must not explain boundary-core or weak-pair hard
  endpoints.
- **V11 seed robustness:** responsible candidate outcomes must be evaluated
  across method seeds; seed-fragile hits remain caveated.
- **V12 intervention-size caveat:** record touched node count, initial group
  count, and ambiguous-node count; do not claim cost advantage from this panel.
- **V13 family coverage:** all four mechanism families must have at least one
  interpretable non-anchor variant result before external-generalization
  wording is allowed.
- **V14 claim gate:** even a full pass supports only tiny-demo mechanism
  variant robustness. It does not open optimizer-native wall/pathway,
  quality/cost, NanoClustering generality, or algorithm claims.
- **V15 role/name invariance:** candidate generation must be stable under
  opaque node ids and role/name permutation. If a rule only works because the
  graph uses names such as `a`, `b`, `h`, `x`, or `m`, the variant panel is a
  demo-name artifact.
- **V16 endpoint replay stability:** recurrent variant endpoints must replay
  to themselves from direct initial membership. If target endpoints are
  unstable, method misses cannot be interpreted as handle failures.
- **V17 recurrence-definition lock:** the manifest must fix seed count,
  recurrence threshold, endpoint signature tolerance, top-quality handling, and
  rare-endpoint exclusion rules before execution.
- **V18 matched decoy specificity:** mechanism-removed controls should include
  size/contact-matched decoys, so specificity is not won by making controls
  structurally too easy. If a decoy cannot be built for a variant, the runner
  must emit a waiver row with the failed matching fields.
- **V19 mechanism-purity audit:** each variant must report mechanism features
  showing whether the intended axis is present without unintentionally
  activating another mechanism family as the easier explanation.
- **V20 baseline uncertainty:** method comparisons must be judged against the
  restart distribution, not only the restart mean. A tiny positive delta with
  broad restart uncertainty remains caveated. Strong pass requires target-
  scoped performance above the restart p75 at the same budget; mean-only
  improvement is diagnostic, not promotion evidence.
- **V21 revision lock:** any change to gamma, graph edges, role annotations, or
  candidate rules after seeing outcomes creates a new panel version. It cannot
  silently update `v1`.
- **V22 shadow quality/cost ledger:** CPM quality, wall time, actual Leiden
  evaluations, candidate count, touched-node count, and memory HWM should be
  recorded as shadow fields. They do not open quality or cost claims in Stress
  4.

Pass:

- ordinary Leiden + CPM reproduces recurrent alternatives in non-anchor
  mechanism-preserving variants for the relevant families;
- blind-rule handles beat the target-scoped restart p75 in preserved variants
  under budget 20, or are explicitly caveated as mean-only diagnostic signal;
- mechanism-removed controls do not produce false positive target claims;
- role/name invariance, endpoint replay stability, matched decoy specificity,
  and mechanism-purity audits do not expose leakage or control weakness;
- failures are typed by mechanism removal, tie breaking, or rule limitation.

Fail:

- positive results exist only on the original anchor graphs;
- preserved variants expose recurrent alternatives but the relevant blind-rule
  handle class cannot recover them;
- controls produce apparent target hits after the mechanism has been removed;
- candidate generation depends on demo names or role labels in a way that does
  not survive opaque-id checks;
- preserved variants expose recurrent alternatives but endpoint replay shows
  those alternatives are not stable;
- controls are not matched enough, and no waiver row explains the failed
  matching fields;
- results require changing gamma, edge weights, or candidate rules after
  looking at endpoint outcomes.

Primary artifact:

- `leiden_basin_tiny_cpm_mechanism_variant_panel_v1_YYYYMMDD/`.

Primary outputs:

- `tiny_cpm_variant_graph_manifest.csv`;
- `tiny_cpm_variant_graph_edges.csv`;
- `tiny_cpm_variant_graph_roles.csv`;
- `tiny_cpm_variant_config.json`;
- `tiny_cpm_variant_seed_runs.csv`;
- `tiny_cpm_variant_endpoint_summary.csv`;
- `tiny_cpm_variant_frozen_endpoint_manifest.csv`;
- `tiny_cpm_variant_restart_discovery.csv`;
- `tiny_cpm_variant_endpoint_replay.csv`;
- `tiny_cpm_variant_mechanism_features.csv`;
- `tiny_cpm_variant_role_invariance.csv`;
- `tiny_cpm_variant_phase_lock.json`;
- `tiny_cpm_variant_blind_candidate_registry.csv`;
- `tiny_cpm_variant_attempts.csv`;
- `tiny_cpm_variant_first_hits.csv`;
- `tiny_cpm_variant_target_attribution.csv`;
- `tiny_cpm_variant_dropout_attribution.csv`;
- `tiny_cpm_variant_baseline_uncertainty.csv`;
- `tiny_cpm_variant_shadow_quality_cost.csv`;
- `tiny_cpm_variant_failure_typing.csv`;
- `tiny_cpm_variant_gate_matrix.csv`;
- `tiny_cpm_variant_revision_lock.json`;
- `tiny_cpm_variant_summary.json`;
- `tiny_cpm_variant_p0_p4_report.md`;
- `tiny_cpm_variant_report.md`.

### Stress 4 Runner Implementation Design

Entry script:

- `research/consensus/scripts/leiden_basin/demo/analyze_leiden_cpm_tiny_mechanism_variant_panel.py`.

Default execution parameters:

- `--seeds 100`;
- `--n-iterations -1`;
- `--replay-seeds 10`;
- `--baseline-permutations 1000`;
- `--budgets 1,2,3,5,10,20`;
- `--recurrent-threshold-share 0.05`;
- `--recurrent-threshold-min 2`;
- `--strong-baseline-quantile 0.75`.

The runner should be implemented as explicit phases. Each phase writes its
own artifact and never mutates prior artifacts in place.

| phase | name | reads | writes | hard rule |
| --- | --- | --- | --- | --- |
| P0 | fixed manifest | variant builders only | graph manifest, graph sidecars, config | no Leiden runs and no endpoint artifacts |
| P1 | graph-only diagnostics | P0 graph manifest | mechanism features, decoy matching rows | no endpoint artifacts |
| P2 | blind candidate construction | P0/P1 only | blind candidate registry | must complete before endpoint manifest exists or is read |
| P3 | role/name invariance | P0/P1/P2 only | role invariance rows | opaque-id candidate classes must match canonical classes |
| P4 | phase lock | P0-P3 artifacts | phase lock JSON | hash P0-P3 before any seed outcome is evaluated |
| P5 | baseline seed sweep | P0 and phase lock | seed runs, endpoint summary, frozen manifest, restart distribution | no candidate-rule edits allowed |
| P6 | endpoint replay | P5 frozen endpoints | endpoint replay rows | unstable endpoints become attribution caveats |
| P7 | handle attempts | P2/P4/P5/P6 | schedules, attempts, first hits, target attribution | compacted and slot-preserving schedules required |
| P8 | attribution and gates | all prior outputs | dropout attribution, uncertainty, failure typing, gate matrix, report | claim gate remains closed |

Core dataclasses:

- `VariantGraphCase`: variant id, mechanism family, mechanism state, gamma,
  seed count, builder, expected baseline behavior, responsible rule, control
  read.
- `RoleAnnotation`: role id, node ids, role type, mechanism family, and whether
  the role is allowed for candidate generation.
- `GraphBundle`: canonical graph, opaque-id graph, role mapping, edge-weight
  table, and graph hash.
- `VariantCandidate`: variant id, handle candidate id, handle type, target
  mechanism class, initial groups, graph evidence, touched-node count, and
  candidate signature.

Implementation constraints:

- Do not import frozen endpoint manifests, replay rows, endpoint ranks, or
  method-hit rows in P0-P4.
- Do not derive candidate ids from endpoint signatures.
- Do not use raw node-name prefixes in candidate rules after P3. Candidate
  generation should use role annotations and edge evidence; canonical names are
  display labels only.
- Recurrent endpoints use the same default rule as the tiny CPM baseline:
  `max(recurrent_threshold_min, ceil(seeds * recurrent_threshold_share))`.
- Endpoint replay uses direct initial memberships from frozen endpoint
  signatures. Target attribution is blocked if a target endpoint replay rate is
  below 0.9 and caveated if it is below 1.0.
- Restart uncertainty should be stored at least as mean, median, p25, p75, p95,
  and all-target-hit rate for each variant, budget, and target scope.
- Strong method evidence requires target-scoped performance above restart p75
  at the same budget. Mean-only improvement is reported but cannot pass V20.
- Mechanism-removed controls require decoy matching or an explicit waiver row.
- Any graph, gamma, role, or candidate-rule change after P4 creates a new
  output directory, not an update to the current `v1` directory.

First implementation slice:

1. Implement P0-P4 only and run a smoke check.
2. Inspect the graph manifest, mechanism features, candidate registry,
   role/name invariance rows, and phase-lock hash.
3. Only if P0-P4 pass, implement P5-P8.

This split prevents the runner from learning from endpoint outcomes while
candidate generation is still being debugged.

The first P0-P4 implementation is materialized at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_tiny_cpm_mechanism_variant_panel_v1_20260531/`.
It writes 12 fixed variants, 513 graph-edge rows, 99 role rows, 34 blind
candidate rows, mechanism-feature rows, role/name invariance rows, config, and
a phase-lock artifact. Role/name invariance passes across canonical node names,
opaque node ids, and permuted role labels. The phase-lock hash is
`2d21ec3b65ac8859d56058fc9d84ccdc7a75fc2fc752171e89fe411e07a5805c`.
This only clears the construction-independence gate for P0-P4; it does not
open baseline reproduction, endpoint replay, target attribution, restart-p75,
wall/pathway, quality/cost, NanoClustering generality, or algorithm claims.

The P4.5 control-strength audit is materialized at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_tiny_cpm_mechanism_variant_panel_p4_5_control_audit_v1_20260601/`.
It verifies that the P0-P4 artifact hashes still match the phase-lock, confirms
candidate construction remains pre-endpoint, confirms role/name invariance, and
checks whether mechanism-removed controls include target-like decoys. Six gates
pass, but the control-decoy-strength gate is caveated: `ab_diffuse_no_core_control`
and `df_weak_module_separate_control` have zero target-like candidate rows.
Thus P5-P8 may be run as a diagnostic, but it should not be used as promotion
evidence until these controls are revised in a new phase-locked panel version.

### Stress 4 v1.1 Control-Matched Redesign

The P4.5 audit shows a design flaw, not an endpoint result. The weak controls
remove the intended mechanism so completely that the blind-rule generator has no
target-like decoy to try. That makes a future no-false-positive result too easy.
The next panel version should keep `v1` unchanged and materialize a new
phase-locked output directory, for example
`leiden_basin_tiny_cpm_mechanism_variant_panel_v1_1_YYYYMMDD/`.

Improved control principle:

- mechanism-removed controls must remove the responsible mechanism axis;
- they must still expose target-like decoy handles with matched size/contact
  pressure;
- decoy handles must use the same broad handle family as the positive rule
  class, but `target_claim_allowed=false`;
- decoy roles must be separated from responsible roles, so mechanism-purity
  features do not mistake a decoy for mechanism preservation.

Role-model changes:

- keep `boundary_core` and `weak_pair` as responsible mechanism roles;
- add `boundary_core_decoy` for target-like absorption/balanced controls;
- add `weak_pair_decoy` for target-like diffuse controls;
- include `decoy_target_rule_family`, `decoy_contact_mass`,
  `decoy_touched_node_count`, and `decoy_match_status` in mechanism features;
- P4.5 should count target-like decoys from handle type plus
  `target_claim_allowed=false`, not from positive target claims.

Absorption control redesign:

- replace `ab_diffuse_no_core_control` with a control that keeps no compact
  all-module boundary core;
- keep balanced diffuse module-to-host contact on both hosts;
- add scattered `boundary_core_decoy` roles with the same node-count scale as
  preserved boundary cores, but where no host core is jointly contacted by all
  small-module nodes;
- generate `blind_small_module_boundary_core_initialization` decoy candidates
  for those decoy roles, with `target_claim_allowed=false`;
- pass condition: responsible `boundary_core_concentration` remains near zero,
  while `target_like_decoy_candidate_count >= 1`.

Diffuse control redesign:

- replace `df_weak_module_separate_control` with a control that keeps the weak
  module coherent or separate through stronger internal weak-module cohesion;
- add superficial host contacts and `weak_pair_decoy` roles that look eligible
  to the weak-pair tail-split generator;
- keep responsible `weak_pair_count=0` or mark all weak-pair-like roles as
  decoys only;
- generate `blind_weak_pair_tail_split_initialization` decoy candidates with
  `target_claim_allowed=false`;
- pass condition: responsible weak-pair mechanism remains removed, while
  `target_like_decoy_candidate_count >= 1`.

Audit gate changes for `v1.1`:

- P4.5-G7 becomes a hard gate for promotion-oriented Stress 4 evidence, not a
  caveat;
- each mechanism-removed control must have at least one target-like decoy
  candidate for its responsible rule family;
- decoy touched-node count should be within the preserved family range, or the
  runner must emit a waiver row;
- decoy contact mass should be within a predeclared ratio band against the
  preserved family median, or the runner must emit a waiver row;
- controls still fail if any target-like decoy is counted as a positive target
  claim.

Execution sequence:

1. Materialize `v1.1` P0-P4 in a new output directory.
2. Run P4.5 immediately on `v1.1`.
3. If all P4.5 gates pass, then run P5-P8.
4. If P4.5 still has weak controls, do not run promotion-oriented P5-P8.

The existing `v1` panel remains useful as a diagnostic showing why target-like
decoy matching is necessary. It should not be overwritten or silently upgraded.

The revised `v1.1` P0-P4 panel is materialized at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_tiny_cpm_mechanism_variant_panel_v1_1_20260601/`.
It preserves the 12-variant Stress 4 family while adding explicit decoy roles
for the weak absorption and diffuse controls. The resulting graph-only panel has
542 edge rows, 107 role rows, 42 blind candidate rows, role/name invariance pass,
and phase-lock hash
`ee2be280a23a8aadb7c8f57e28ab654241f9e2d4d19118edbb84cc80da78177e`.

The revised P4.5 hard-gate audit is materialized at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_tiny_cpm_mechanism_variant_panel_p4_5_control_audit_v1_1_20260601/`.
With `--hard-control-decoy-gate`, all seven gates pass and readiness is
`ready_for_p5_p8_diagnostic_execution`. The two formerly weak controls now have
target-like decoys without positive target claims:

- `ab_diffuse_no_core_control`: two
  `blind_small_module_boundary_core_initialization` decoy candidates,
  responsible `boundary_core_concentration=0.0`, `decoy_contact_mass=9.0`;
- `df_weak_module_separate_control`: six
  `blind_weak_pair_tail_split_initialization` decoy candidates, responsible
  `weak_pair_count=0`, `weak_pair_concentration=0.0`, `decoy_contact_mass=7.0`.

This only clears the pre-endpoint control-strength gate. It still does not open
baseline reproduction, endpoint replay, target-scoped attribution, restart-p75,
wall/pathway, quality/cost, NanoClustering generality, or algorithm claims.

The downstream P5-P8 endpoint diagnostic is materialized at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_tiny_cpm_mechanism_variant_panel_p5_p8_v1_1_20260601/`.
It verifies the `v1.1` P0-P4 phase-lock before endpoint evaluation, requires
the P4.5 hard-gate audit, runs 1,200 ordinary Leiden + CPM seed runs, freezes 43
endpoint signatures including 33 recurrent endpoints, and executes 240
phase-locked candidate attempts. The result is useful but caveated:
`p5_p8_readiness=caveated_endpoint_diagnostic_only`.

Positive candidates do make target-compatible contact with recurrent endpoints:
15 of 26 preserved recurrent endpoints have target-class-compatible positive
hits, and 13 of those beat the random-restart p75 first-hit position. However,
coverage is incomplete. P8-G8 is caveated because target-compatible coverage is
15/26, with `df_one_pair` having zero target-class endpoint hits. Several
`df_two_pair` recurrent endpoints are also missed. Therefore `v1.1` supports a
diagnostic read that the mechanism-family candidate rules can reach some
alternative endpoints, but it is not promotion evidence for Stress 4 yet.

The next design step should focus on failure typing rather than another broad
threshold sweep:

- distinguish genuinely untargeted endpoints such as `small_module_separate`
  from target-family misses;
- decompose `df_one_pair` and missed `df_two_pair` endpoints by weak-pair slot,
  host target, and candidate schedule position;
- test whether missed diffuse endpoints require additional blind weak-pair
  candidate classes or only a different fixed schedule;
- keep control claims closed: all mechanism-removed controls still have zero
  positive target attempts.

The P8.1 structural failure-typing pass is materialized at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_tiny_cpm_mechanism_variant_panel_p8_1_failure_typing_v1_1_20260601/`.
It refines the P8-G8 caveat. The denominator of 26 preserved recurrent
endpoints includes endpoint types the current positive registry does not try to
target. Under structural target eligibility, 20 endpoints are targetable by the
frozen positive registry, 17 are hit, and 15 beat restart p75. The six
not-targeted endpoints are not method failures under the current registry: two
`small_module_separate` endpoints and four single weak-node split endpoints.

This changes the failure diagnosis:

- `df_one_pair` is not a target-eligible miss. Its pair-to-host-core endpoints
  are hit; its remaining recurrent endpoints are single-node splits outside the
  current positive weak-pair class.
- The true eligible misses are three `df_two_pair` joint endpoints where two
  weak pairs must be placed simultaneously into two host cores.
- The likely next mechanism is therefore a joint weak-pair candidate or
  schedule rule, not a broad parameter sweep.

The P8.2 downstream joint weak-pair probe is materialized at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_tiny_cpm_mechanism_variant_panel_p8_2_joint_weak_pair_probe_v1_1_20260601/`.
It reads the P8.1 true eligible misses and composes the already phase-locked
single weak-pair handles into joint initializations without editing the `v1.1`
P0-P4 registry. The result is decisive as a diagnosis: all three `df_two_pair`
joint endpoints recover, with 3 joint candidates, 30 attempts,
`recovered_endpoint_count=3`, and `all_joint_misses_recovered=true`. Each joint
endpoint is recovered for all 10 method seeds and hits on the first joint
attempt.

This does not promote `v1.1`, because the joint candidates were chosen after
reading P8.1 misses. The valid implication is narrower and more useful:
simultaneous weak-pair moves are sufficient for the remaining missed endpoints.
The next promotable version should be a `v1.2` pre-endpoint panel that derives
joint weak-pair candidates from role evidence alone, adds matched controls, and
then reruns P4.5/P5-P8/P8.1 without reading endpoint outcomes during candidate
construction.

### Stress 4 v1.2 Pre-Endpoint Joint Weak-Pair Panel

The `v1.2` P0-P4 panel is materialized at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_tiny_cpm_mechanism_variant_panel_v1_2_20260601/`.
It keeps the 12-variant Stress 4 graph universe from `v1.1`, keeps the stronger
control-decoy roles, and adds joint weak-pair candidates before any endpoint
manifest is read. The registry has 53 candidates: 42 inherited single/control
candidates, 4 positive `df_two_pair` joint weak-pair candidates, and 7 matched
joint decoy candidates in `df_weak_module_separate_control`. Role/name
invariance passes, and the phase-lock hash is
`293c79949d9d880ca141889604c6359a1bc6008514a0095d8a4b6b155eb8a48c`.

The `v1.2` P4.5 hard-control audit is materialized at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_tiny_cpm_mechanism_variant_panel_p4_5_control_audit_v1_2_20260601/`.
It verifies the phase lock, reports `weak_control_count=0` and
`blocked_control_count=0`, and keeps all mechanism-removed controls at
`target_claim_candidate_count=0` despite the new joint decoy candidates.

The `v1.2` P5-P8 endpoint diagnostic is materialized at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_tiny_cpm_mechanism_variant_panel_p5_p8_v1_2_20260601/`.
It reruns 1,200 ordinary Leiden + CPM seeds, freezes 43 endpoints including 33
recurrent endpoints, and executes 240 phase-locked candidate attempts. The
coarse P8-G8 gate remains caveated because it counts all 26 preserved recurrent
endpoints, including endpoints the current positive registry intentionally does
not target.

The sharper `v1.2` read is the P8.1 structural typing at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_tiny_cpm_mechanism_variant_panel_p8_1_failure_typing_v1_2_20260601/`:
20 preserved recurrent endpoints are structurally target eligible, all 20 are
hit, 16 beat random-restart p75, and 0 eligible misses remain. The three true
`v1.1` eligible misses in `df_two_pair` are now hit by pre-endpoint joint
weak-pair candidates at attempts 1, 2, and 3. Six preserved recurrent endpoints
remain non-targeted by the current positive registry: two small-module separate
endpoints and four single weak-node split endpoints.

The current valid claim is therefore narrow: role-derived simultaneous
weak-pair candidates fix the prior joint-endpoint coverage failure under the
Stress 4 tiny CPM mechanism-variant panel. This is still endpoint-initialization
evidence only; it does not execute route/pathway traces, promote walls, measure
quality/cost gains, or support NanoClustering generality or algorithm-level
claims.

### Stress 4 v1.2 Schedule Robustness

The first schedule/order robustness diagnostic for `v1.2` is materialized at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_tiny_cpm_mechanism_variant_panel_schedule_robustness_v1_2_20260601/`.
It keeps the phase-locked `v1.2` P0-P4 registry fixed and varies only the
candidate schedule. The panel has 105 schedules: canonical order,
positive-first, joint-first, joint-delayed, a joint-suppressed negative control,
and 100 random within-variant permutations. It writes 25,200 attempt rows and
2,730 endpoint-discovery rows.

All six schedule gates pass:

- phase-lock integrity;
- expected structural denominator of 20 target-eligible preserved endpoints;
- deterministic nonadversarial structural recall 1.0 and joint recall 1.0;
- random-permutation minimum structural recall 1.0 and joint recall 1.0 over
  100 sampled schedules;
- joint-suppressed negative control damages the joint endpoints
  (`joint_suppressed_joint_recall=0.0`);
- controls remain claim-disabled with `control_positive_attempt_count=0`.

The important mechanism read is that `df_two_pair` has 8 structurally eligible
endpoints under `v1.2`; all 8 are hit under nonadversarial schedules, but only
5 are hit when joint candidates are suppressed. The three dropped endpoints are
exactly the joint weak-pair endpoints. This closes the immediate concern that
the `v1.2` result was only canonical-order luck. It does not close external
generality, route/pathway, wall, quality/cost, or algorithm claims.

### NanoClustering Joint Weak-Pair Analog Screen

The first real-data analog screen for the `v1.2` joint weak-pair mechanism is
materialized at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_joint_weak_pair_analog_screen_20260601/`.
It reads the frozen NanoClustering v2.2 accepted-primitive measurement panel and
does not run clustering, construct method candidates, execute routes/pathways,
promote walls, or inspect quality/cost.

The screen finds 99 analog candidates among 223 accepted primitives:

- 17 `tier1_external_multi_fragment_host_competition_analog` primitives;
- 23 `tier2_recurrent_multi_fragment_analog` primitives;
- 59 `tier3_moderate_multi_fragment_analog` primitives;
- 89 candidate source families;
- 30 matched control-like contrast rows.

The immediate interpretation is narrow. NanoClustering contains enough
multi-fragment, host-competition structure to design a local panel around the
joint weak-pair mechanism. The screen does not show that a joint initialization
will work on NanoClustering. The next valid step is a frozen local panel with
pre-endpoint role extraction, matched source-preserved/non-analog controls, and
separate endpoint-replay readout.

### NanoClustering Local Panel Design

The first frozen local panel design is materialized at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_joint_weak_pair_local_panel_design_20260601/`.
It consumes the analog screen and freezes 30 candidate/control cases for future
endpoint replay:

- 17 core tier-1 joint weak-pair analog cases;
- 10 `strict_core_v0` primary cases and 7 full-core caveated sensitivity cases;
- 13 lower-tier reserve cases;
- 30 matched source-preserved non-analog controls;
- 60 candidate/control role rows;
- 386 event-role rows;
- 60 endpoint-family signature rows;
- 17 core-only one-to-one control sensitivity rows;
- 7 endpoint-replay contract rows.

The design is intentionally caveated rather than promotion-ready. Reusable
nearest controls are attached to every case, but only 8 unique control anchors
cover the 30 cases, with maximum reuse 6. The artifact therefore writes a
separate core-only one-to-one control sensitivity panel with 17 unique controls.
It also freezes endpoint-family signature rows because candidate top-1 endpoint
handles are diagnostic members, not standalone hit targets. The next
implementation should produce the contract artifacts, especially
`replay_config.json`, `endpoint_replay_attempt_rows.csv`,
`endpoint_signature_rows.csv`, `endpoint_replay_case_summary.csv`, and
`endpoint_replay_control_sensitivity_summary.csv`, against `strict_core_v0`
first. Execution remains closed by design.

### NanoClustering Endpoint Replay Readiness

The first endpoint-replay readiness runner is materialized at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_endpoint_replay_readiness_20260601/`.
It does not run replay. It checks whether replay is executable from the frozen
local panel. The result is:

- 10 `strict_core_v0` cases selected as the primary denominator;
- 20 candidate/control roles and 200 planned method-seed attempts frozen;
- 465/465 endpoint-family target handles resolved to local NanoClustering
  membership artifacts;
- the raw graph execution gate is now open through verified local mirrors of
  the Java/Rust sidecar candidate graphs; the original `/data/openalex_clusters`
  paths remain recorded as provenance, and the local active hierarchy graph is
  still not substituted because it is not row-identity compatible with the
  sidecar candidate graphs.

### NanoClustering Endpoint Replay Pilot Smoke

The first bounded replay pilot is materialized at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_endpoint_replay_pilot_smoke_20260601/`.
It executes the same NanoClustering endpoint sequence as the seed sweep:
`Leiden -> min_nano postprocess -> min_docs postprocess`. The first smoke uses
the first strict-core Java case, candidate/control roles, and method seed 0.

Result:

- 2 attempts executed;
- 12 endpoint-family target handles scored;
- branch graph load took 49.8s and the two attempts took 241.3s total;
- both roles start from the same seed0 full partition hash;
- both roles terminate in the same partition hash;
- `role_distinction_status=blocked_terminal_partition_identical_across_roles`.

This is a useful negative diagnostic. It shows that full-partition warm-start
replay is not a role-local basin/pathway probe on this surface. The next valid
implementation step is not to expand the 200-attempt replay grid. It is to
change the execution object: use a role-local boundary object, a fixed-node or
fixed-outside-mask constraint, or an explicit route intervention that can
distinguish candidate/control roles before any wall/pathway or method-success
claim is evaluated.

### NanoClustering Role-Local Boundary And Raw Route Smoke

The role-local boundary plan is materialized at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_role_local_boundary_plan_20260601/`.
It translates strict-core endpoint-family handles into concrete source/target
masks and fixed-outside route contracts:

- 20 strict-core roles;
- 10 candidate/control case contrasts;
- 139 target endpoint handles;
- 5,560 planned route rows across 4 route arms and 10 method seeds;
- all roles `ready_role_local_fixed_mask_contract`;
- all case contrasts `distinct_role_local_objects`;
- median pair free mask 74 nodes;
- maximum pair free mask share 0.00315.

Two route smokes clarify the mechanism boundary. The pair-free smoke at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_role_local_route_pilot_smoke_20260601/`
runs 4 raw attempts and has `route_arms_distinct_count=0`: target seeding alone
does not move the local raw terminal. The anchor smoke at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_role_local_route_pilot_anchor_smoke_20260601/`
runs 8 selected raw attempts, blocks 1 empty-free-mask arm before Rust, and has
`route_arms_distinct_count=4`: source-anchor and target-anchor arms produce
different raw fixed-mask terminals on all selected role-target pairs.

The bounded anchor-arm expansion at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_role_local_route_pilot_anchor_expand_seed0_20260601/`
runs the same two anchor arms across all 20 strict-core roles with one target
handle per role at method seed 0. It records 40 selected route attempts over 16
target handles, blocks 1 empty-free-mask arm before Rust, and has
`route_arms_distinct_count=20`. Among role-target pairs where both arms execute
Rust, the distinction rate is 19/19. This is a stronger local manipulability
signal: anchor choice can steer the tiny fixed-mask source/target boundary to
different raw terminal states across the strict-core panel.

Interpretation remains narrow. This is raw fixed-mask evidence only. It does
not yet show endpoint replay because postprocess is not fixed-mask aware in the
current wrapper. The next gate must define a fixed-mask-aware endpoint or
postprocess readout before any wall, quality/cost, real-data method-success, or
algorithm claim is reopened.

The fixed-mask endpoint readout pilot is materialized at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_fixed_mask_endpoint_readout_pilot_anchor_expand_seed0_20260601/`.
It keeps the route fixed-mask contract through a bounded postprocess
approximation: for the min-nano and min-doc stages, only small non-fixed nodes
may move under constrained Leiden. Across the same 40 anchor-arm attempts, it
executes 78 constrained postprocess rounds and preserves distinction through
the readout: `raw_route_arms_distinct_count=20`,
`post_nano_route_arms_distinct_count=20`, and
`post_doc_route_arms_distinct_count=20`. No local pair hash changes from raw
to post-doc, and no post-doc attempt changes nodes. The local alternatives are
therefore stable under this fixed-mask readout, but this is still not the
production NanoClustering endpoint wrapper and does not open wall/pathway or
method-success claims.

The deep-dive readout audit also narrows the interpretation. The 20 role-target
pairs contain 16 unique source/target/pair mask objects because two Rust
control-side objects are reused across case labels. More importantly, the
zero-change postprocess readout is partly expected: raw route optimization
already converges under the same CPM objective with a larger free set, while
the readout uses a subset of those free nodes. This makes the result a
consistency check, not a non-tautological endpoint-survival proof. The next
wall-facing gate should release both anchor-arm terminals into the same
fixed-outside pair mask and test whether they collapse or remain distinct under
a common feasible set.

That anchor-release gate is now materialized at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_anchor_release_pilot_anchor_expand_seed0_20260601/`.
The result is negative for the current anchor-arm basin interpretation. Across
20 role-target pairs and 16 unique local mask objects, the anchor terminals are
distinct before release but collapse after common release:
`anchor_pair_distinct_count=20`, `release_pair_distinct_count=0`, and
`release_pair_collapsed_count=20`. The common release terminal is neither
anchored endpoint (`release_equals_source_anchor_count=0`,
`release_equals_target_anchor_count=0`) and the anchored target identity largely
disappears: median released target doc-share is 0.02909. Therefore the current
anchor arms should be treated as boundary-condition manipulability probes, not
as wall evidence. Future gates must filter for common-release non-collapse
before reopening basin-wall language.

The follow-up policy comparison at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_anchor_release_policy_comparison_20260601/`
confirms that this is not only a first-target selection failure. Combining the
default anchor expansion with `lowest_target_overlap` and `largest_pair_free`
selections yields 37 unique role-target pairs across 10 panel cases. All 37 are
anchor-distinct, and all 37 collapse after common release
(`release_pair_distinct_count=0`, `release_pair_collapsed_count=37`). The largest
tested common-release free set has 246 movable nodes, and the median target
identity drop from target-anchor to common-release terminal is 0.97125. The
method gate therefore remains closed: the present source/target anchor-arm
construction is a boundary-condition probe, not a release-stable basin
distinction primitive.

The stricter common-mask multistart gate is also negative on the first bounded
slices. The comparison artifact at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_common_mask_multistart_comparison_20260601/`
combines `largest_pair_free` top-6 and `lowest_target_overlap` top-6 candidate
pairs. For each pair the runner uses source-state, target-seeded,
pair-singleton, source/target two-block, and four random pair-block
initializations under the same fixed-outside pair mask. Across 12 unique pairs
and 96 starts, every pair converges to one terminal pair hash
(`terminal_multiplicity_pair_count=0`,
`max_unique_terminal_pair_hash_count=1`). This is still not global evidence
against basin multiplicity; it specifically says the current local pair-mask
universe is not exposing release-stable multiplicity under the present
Leiden+CPM settings.

The basin-universe redesign artifact at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_basin_universe_redesign_20260601/`
then reframes the next executable gate. The failed route universe used a single
top1 endpoint handle even though the frozen readout contract is
endpoint-family signature distance. The redesign materializes 60
signature-level universes, 30 candidate/control case-union candidates, and 60
symmetric-object resolver rows. Signature universes are the right immediate
gate because they match the success unit, but they are not much larger than the
failed local masks: median 57 nodes, max 261, and median expansion 1.10169 over
available single-target pair baselines. A signature-level collapse should
therefore trigger a universe change to case-level or symmetric-object masks,
not another anchor-arm policy sweep.

The first executable signature-universe gate has now collapsed on the largest
strict-core candidate slice. The pilot at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_signature_universe_multistart_strict_candidate_top6_seed0_20260601/`
runs eight starts over each of six candidate signature universes
(`universe_node_count_median=155.5`, max 261). All six converge to one terminal
hash (`terminal_multiplicity_signature_count=0`,
`max_unique_terminal_universe_hash_count=1`), with median target-union doc-share
0.00247. This closes the "single top1 handle was too narrow" explanation for
the largest strict-core candidate slice. The next route universe must be
case-level candidate/control union or symmetric-object based.

The first case-universe materialization at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_case_universe_plan_20260601/`
turns those case-level candidates into actual fixed-outside mask contracts.
All 30 case universes are executable in principle, and all 10 strict-core cases
are route-contract ready. But the top-10 strict-core edge-interaction check
shows that the case-union route is weaker than it first looks: candidate and
control universes have zero node overlap, the actual node count equals the
upper-bound sum, median candidate/control cross-edge weight share is only
`5.489829496047739e-05`, and even the maximum is only
`0.0010298866522870241`. Therefore a case-union multistart runner should be
treated as an interaction-gated negative-control closure, not as the strongest
next basin-wall probe. The more promising redesign is now a symmetric-object
mask resolver, because it can define an anchor-independent object first and
then test optimizer dynamics inside that object rather than just OR-ing two
nearly disconnected role-side regions.

The symmetric-object universe materialization at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_symmetric_object_universe_plan_20260601/`
now resolves that stronger route into actual fixed-outside object-mask
contracts. All 60 role-level object universes are ready, with 36
anchor-independent rows, 18 anchor-independent P1 rows, and 5 strict-core
anchor-independent P1 rows. The object masks are still compact
(`object_node_count_median=26`, max 164) but no longer depend on OR-ing
candidate/control case sides. All 30 candidate/control case relations are
`disjoint_symmetric_objects` with median case-overlap share 0.0, so the case
relation should be treated as secondary diagnostics. The next executable gate
is therefore role-level symmetric-object multistart over the strict-core
anchor-independent P1 slice, not another case-union or anchor-arm sweep.

The first symmetric-object multistart execution then closes that immediate gate
negatively but usefully. The P1 run at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_symmetric_object_multistart_strict_anchor_independent_p1_seed0_20260601/`
tests 5 strict-core anchor-independent P1 role universes with 40 starts
(`seed0_source_state`, `seed0_object_seeded`, `object_singleton`,
`object_seed_component_blocks`, and four random object-block starts). It finds
zero terminal multiplicity (`max_unique_terminal_object_hash_count=1`). More
importantly, every role singleton-collapses inside the object:
`terminal_singleton_object_role_count=5`, median terminal object cluster count
76.0 against median object size 76.0, median component-reference ARI 0.0, and
median object best-cluster doc share `0.006055013194748`. The P2 unique-object
run at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_symmetric_object_multistart_strict_anchor_independent_p2_unique_seed0_20260601/`
repeats the same pattern over 2 unique P2 objects and 16 starts:
`terminal_singleton_object_role_count=2`, median terminal object cluster count
58.5 against median object size 58.5, and median component-reference ARI 0.0.
This should not be read as basin absence. It says the current object-only mask
does not supply a cohesive free subproblem; the next universe definition must
include support-neighborhood or attachment-aware context before another
terminal-multiplicity gate.

That support-neighborhood hypothesis has now been tested in its simplest form
and fails as a standalone fix. The support-top100 P1 unique-object run at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_symmetric_object_support_top100_p1_unique_seed0_20260602/`
adds 100 boundary-weight support nodes to each of 4 unique strict-core P1
objects and still reports zero terminal object/universe multiplicity, with all
universes singleton-collapsed. The top-1000 smoke at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_symmetric_object_support_top1000_smoke_one_20260602/`
expands one object to a 1,076-node free universe and still terminates as 1,076
singleton clusters.

The follow-up CPM merge-viability audit at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_symmetric_object_merge_viability_p1_unique_top0_100_1000_20260602/`
explains why. Across 4 P1 unique objects and support top-k values 0, 100, and
1000, none of the 12 audited universes has a positive internal free-free merge
candidate or a positive external free-to-fixed attachment node under
`gamma=0.7`. The best internal merge delta is still negative
(`-43749.985507246376`), and the best external attach delta is negative
(`-528324.9848484849`). A one-object exact-quality smoke at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_symmetric_object_merge_viability_quality_smoke_top0_1000_20260602/`
points the same way: object-only singleton is tied for best, and the top-1000
universe singleton is the best quality variant.

Therefore the next NanoClustering gate should not be another support top-k
widening or P1/P2 list expansion. The method-design requirement is now to define
a free universe whose CPM local mechanism is positive before testing terminal
multiplicity: lower-resolution/local-weight normalization checks, predeclared
weak-pair or bridge mechanisms, or another objective-positive support rule. This
keeps the guiding premise intact while rejecting the current universe
operationalization.

The first objective-mechanism audit makes that requirement concrete. The audit
at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_symmetric_object_objective_mechanisms_p1_unique_top0_100_1000_20260602/`
uses the same 4 P1 unique objects and support top-k values 0, 100, and 1000.
Under current doc-weighted CPM, all 12 universes remain objective-negative:
zero positive internal free-free merges and zero positive free-to-fixed
attachments. The best internal pair only becomes positive below
`gamma=4.763001162781675e-05`, and the best external attachment only below
`gamma=2.257078917502457e-05`; current `gamma=0.7` is therefore an objective
scale mismatch by four to five orders of magnitude. `sqrt_doc_weight` does not
fix this. `log1p_doc_weight` opens internal merges but not external attachments,
while unit and local-normalized doc-weight transforms open both. These are
candidate objective mechanisms, not claims that a method has succeeded.

The first critical-gamma terminal checks show why this matters. The
support-top1000 smoke at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_symmetric_object_multistart_support_top1000_gamma1e5_smoke_one_20260602/`
runs the representative object at `gamma=1e-5` and no longer singleton-collapses:
4 deterministic starts produce 3 object terminal hashes and 4 universe terminal
hashes. The bounded support-top100 P1 run at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_symmetric_object_multistart_support_top100_gamma1e5_p1_unique_20260602/`
extends this across 4 unique P1 objects: all 4 show terminal multiplicity, all 4
show universe-level multiplicity, and zero singleton-collapse remains. This is
not yet wall/pathway evidence. It is the first strong indication that the
previous failures were caused by an objective-negative universe, not by absence
of basin-like alternatives. The next design must now add controls against a
trivial low-resolution merge artifact before any promotion.

### Promotion Rule

Do not move from small-demo method signal to NanoClustering stress tests until
Stress 1, Stress 2, and Stress 3 pass. Stress 4 is required before any
external-generalization wording. None of these stresses open optimizer-native
wall/pathway, quality/cost, or algorithm claims by themselves.

## Stop Conditions

Stop generality escalation if:

- no non-seed0 anchor recovers T1-like structure;
- symmetric endpoint objects fail to reproduce anchor-local alternatives;
- alternatives are dominated by one data family or one graph scale.

Stop method escalation if:

- no small graph reproduces multi-endpoint behavior under Leiden + CPM;
- the proposed method only changes thresholds or ranking;
- wall/pathway evidence cannot be separated from final quality;
- intervention cost is not compared to restart cost.

## Claim Boundary

Until G1-G3 pass, the strongest claim is:

> Large empirical endpoint ensembles reveal candidate support-local structural
> alternatives that motivate a basin-candidate cartography protocol.

After G1-G3 pass, the claim can become:

> Basin-like alternatives recur across seeds and controlled CPM demo graphs,
> and their separation is explained by identifiable local graph mechanisms.

Only after G4-G5 pass can the claim become:

> A compact method can discover, diagnose, or navigate basin alternatives better
> than Leiden + CPM restart baselines.
