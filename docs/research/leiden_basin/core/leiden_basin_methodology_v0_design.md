# Leiden Basin Methodology v0 Design

Status: design draft
Date: 2026-05-29
Scope: Track C basin-definition and wall/pathway methodology only

This document turns the current evidence state into a next-step methodology.
It does not define a final basin theory, run a route batch, promote wall
claims, or inspect basin quality/cost.

The current claim vocabulary is refined by
`leiden_basin_surface_claim_schema.md`. That schema treats basin evidence as
surface-qualified: recurrent state signatures, local signature-objects,
endpoint objects, typed relations, wall-ready relations, and quality/method
claims are separate promotion levels. This v0 methodology keeps its
support-local basin objects, but future audits should report the surface schema
before using stronger basin or wall language.

## Current Evidence Basis

The active evidence supports two different conclusions:

- H1 has candidate support: multiple meaningful support-local basin candidates
  appear in current non-field34 artifacts.
- H2 is not operational: current pathway and wall evidence gates expose no
  executable route candidate.

The latest existence audit reports:

- `16` non-field34 cases;
- `5` strong and `4` moderate candidate multi-basin evidence cases;
- `87` strong and `75` moderate meaningful distinct pairs;
- `0` pathway candidates requiring manual review on the current 23-pair
  pathway surface.

Therefore the next research object is not a basin-transition operator. It is a
basin cartography protocol:

1. define primitive basin objects;
2. precommit a non-field34 validation panel;
3. define wall/pathway evidence requirements;
4. only then run small pathway probes.

## Research Question

Primary question:

> Do Leiden runs expose multiple meaningful observed basin candidates under a
> declared primitive definition, and can any pair be shown to have an
> optimizer-native wall or pathway relation?

This decomposes into four subquestions:

1. Which endpoint identities exist under fixed case and endpoint protocols?
2. Which endpoint identities form support-local basin candidates under declared
   distance and ambiguity rules?
3. Which candidate pairs are eligible for wall/pathway analysis before any
   route execution?
4. What evidence would be sufficient to label a pathway as `crosses`,
   `bounces`, `collapses`, or `unknown`?

Quality, final partition value, cost, and operator usefulness are downstream
questions. They must not decide basin existence or wall promotion.

## Primitive Basin Definition v0

The v0 design uses a two-level definition.

```text
endpoint_identity:
  Exact canonical final polished partition identity under a fixed case_id and
  fixed endpoint protocol.

global_observed_basin:
  Cluster of endpoint identities under a declared global partition-distance
  metric and threshold.

support_local_basin_candidate:
  Cluster of endpoint identities under a declared changed-support or boundary
  footprint metric and threshold.

basin_pair_relation:
  Pairwise relation between endpoint identities or support-local basin
  candidates: same, distinct, ambiguous, or hygiene-limited.

wall_pathway_relation:
  Route or optimizer evidence connecting two accepted basin candidates. This is
  not part of basin identity.
```

The operational object for the next pass is `support_local_basin_candidate`.
It is intentionally weaker than a global attraction basin. The design should
say this every time the result is reported.

After the `014`/`016`/`005` surface split, this definition should be read as a
surface-qualified claim. A support-local basin candidate may be endpoint-vector
evidence, signature-object evidence, or endpoint-object evidence depending on
the active surface. It becomes wall-ready only after a separate typed-relation
gate passes.

## Threshold And Ambiguity Rule

The v0 rule should use hard categories rather than a single continuous score:

- `same`: pair is inside the declared same-zone for the chosen relation metric;
- `distinct`: pair is outside the declared distinct-zone and passes support
  size and hygiene checks;
- `ambiguous`: pair lies in the boundary band or has conflicting global versus
  support-local evidence;
- `hygiene_limited`: pair is field34-like, tiny-support, duplicate, no-op, or
  support-source-limited.

The boundary band is not a nuisance to eliminate. It is a definition result.
Rows in the boundary band cannot be converted into wall evidence because a
route is stable or because a final endpoint looks interesting.

## NanoClustering Endpoint-Boundary Addendum

The external NanoClustering seed-ensemble evidence uses a different endpoint
universe from the local Leiden wall-route panel, so it must not be merged into
route or operator claims. It can, however, inform the primitive boundary
definition.

Current NanoClustering evidence supports a fragmentation-first endpoint
boundary rule:

- event axis: `top_split_share_ref_weight`, the largest surviving share of a
  seed0 reference cluster in another seed endpoint;
- strong fragmentation event: `top_split_share_ref_weight < 0.5`;
- recurrent strong fragmentation candidate: at least `2` strong events across
  comparison seeds;
- persistent strong fragmentation candidate: at least `5` strong events across
  comparison seeds;
- stable-like reference: no moderate fragmentation event under
  `top_split_share_ref_weight < 0.8`.

The stratified panel check currently favors recurrent strong fragmentation as
the primitive endpoint-boundary candidate rule. Recurrent strong candidates
mostly expand into severe-split or split-and-merge archetypes, while single
strong and moderate candidates are too mixed and stable-like controls remain
mild relabeling. Absorption must remain a separate archetype axis because
matched stable controls can show absorption without fragmentation.

The recurrent boundary-family registry makes this candidate rule operational
for the next definition pass. It identifies 478 recurrent families, then
separates 179 definition-core rows (`repeat_severe_core` plus
`persistent_mixed_core`), 163 stress-test rows (`multi_seed_mixed_recurrent`),
and 136 edge-control rows (`pair_only_*`). The pair-construction pass should
start from the definition-core rows, use stress-test rows to probe definition
fragility, and retain edge controls as counterexamples.

The first definition-core endpoint-pair expansion uses the pair-construction
panel, not the full 179-row core. It expands 20 panel-selected core families
into 147 strong endpoint-pair events with 0 mild relabeling events. The two
core tiers are not interchangeable: `repeat_severe_core` yields 60
split-and-merge, 13 severe-split, 3 moderate-split, and 2 merge-absorption
events, while `persistent_mixed_core` yields 42 merge-absorption, 19
moderate-split, 6 split-and-merge, and 2 severe-split events. The next
definition check should therefore test internal family coherence separately
for these two core tiers before reopening any optimizer-native pathway or wall
protocol.

The dataset-contrast diagnostic adds a separate density read. Current
NanoClustering full/candidate graphs are dense at the nano level, with
available summaries around 78k-80k nodes, 105M-107M edges, average degree near
2.7k, and undirected density near 0.033-0.035. However, the same data family's
top20-union projection is sparse, and prior Track C edge rows are not locally
available for direct density recomputation. Density should therefore be
treated as a plausible contributor to near-boundary alternatives, not as a
proven cause of the basin-fragmentation signal. Any promotion requires a
within-current-graph control over local degree, cut ratio, top-neighbor
concentration, and full versus top20-sparse behavior.

The primitive basin-distinction panel now separates endpoint handles and
source-target relations over the definition-core pair cases. It materializes 20
source reference endpoint handles, 147 comparison endpoint handles, and 147
relation rows. Under the current gate, 115 relations are accepted as primitive
distinct endpoint-pair relations. The important distinction is not quality:
`repeat_severe_core` mostly becomes fragmentation-dominant multi-endpoint
families, while `persistent_mixed_core` mostly becomes host-absorption or mixed
families. These are support-local basin candidates and relation archetypes, not
final global attraction basins.

The basin-vector panel is the next refinement of that definition. It treats an
event as a split-segment vector plus dominant host context, rather than as one
top comparison endpoint. This reveals that the strongest `repeat_severe_core`
families are diffuse multiway fragmentation families, and that some
`persistent_mixed_core` rows are not weak failures but source-host balanced
two-way splits or external-host absorption families. The current primitive
basin object should therefore be a support-local endpoint-vector family, not a
single top endpoint-pair handle.

The basin-vector coherence diagnostic adds the next gate over the same
definition-core events. A family is treated as definition-core stronger when its
split-vector class, shape core, dominant host context, and dominant host handle
repeat across comparison seeds. Under this diagnostic, 8 of 20 families are
coherent in both vector and host context, 1 is class-coherent with numeric
variation, and 11 are host-variable, split-mixed, or heterogeneous rule-edge
candidates. The resulting primitive is therefore not just an endpoint-vector
family, but an endpoint-vector family with an explicit coherence status.

The full definition-core expansion applies the same endpoint-vector and
coherence gates to all 179 definition-core families rather than the 20-family
pilot panel. It yields 1026 endpoint-pair events, 179 family-vector rows, and a
v1 family registry with 81 `definition_core_v1_coherent` accepted primitive
families. The remaining 98 rows are not negative evidence against basin
existence; they are definition-refinement queues: numeric-stress, split-coherent
host-variable, host-coherent split-mixed, or heterogeneous/rule-edge families.
This makes the current primitive basin definition operational but still
support-local and membership-derived.

The refinement-queue decomposition checks whether those 98 refinement rows are
true failures or overly coarse family units. Primary subfamily decomposition
recovers 134 coherent endpoint-vector subfamilies across 81 source families and
398 of 538 queue events, while 58 singleton/tiny events are deliberately not
promoted. The strongest v2 signal is not the initially expected fine
shape-core split. For split-mixed and heterogeneous queues, `split_vector_class`
is the better first split; for host-variable queues, `host_context_class` is the
better first split. Shape-core signatures should be retained as a secondary
coherence check rather than the first partition axis.

The resulting v2 primitive registry freezes the current operational basin
definition without opening the wall/pathway question. A v2 primitive is either
an accepted v1 coherent family or a primary-axis recovered coherent
endpoint-vector subfamily. This produces 215 coherent primitives covering 886
of 1026 definition-core events and 162 of 179 source families. The residual
140 primary-subfamily events, 17 source families without primary-axis v2
recovery, and alternative-axis-only recoveries remain definition-audit cases.

The v2 audit-surface review makes the definition less brittle by separating
support depth from the primitive definition. The inclusive v2 registry uses a
minimum recovered-subfamily support of 2 events, but many recovered rows are
thin: coverage falls to 766/1026, 658/1026, 550/1026, and 520/1026 if that
support floor is raised to 3, 4, 5, or 6. The primary decomposition rule remains
the default because it recovers most queue evidence, but 12 families have a
better alternative axis and 4 of those are strong exception candidates with
primary recovery zero and alternative recovery of at least 75% of source events.

V2.1 therefore does not change the primitive count. It retains the 215 v2
primitives and 886 covered events, but annotates them with confidence tiers and
separates exception handling from primitive promotion. Strong and weak
alternative-axis exceptions remain outside the primitive registry until their
event-level exception axis is materialized. Marginal secondary-axis gains stay
in the registry as primary-axis primitives with an explicit caveat.

The v2.1 detail review shows that the thin-support caveat is distributed, not
localized: 60 thin recovered primitives are spread across 51 source families.
The strong exception candidates are also concrete enough for the next
definition pass: recomputing their best-axis subfamilies from event-vector rows
recovers 24 coherent events, but those rows remain non-promoted until an
exception-axis rule is explicitly defined. The remaining non-tiny residuals
point to second-axis and joint-axis definition design rather than wall/pathway
search.

The axis-rule candidate materialization then tests these definition rules
without changing the registry. Strong exception-axis candidates are the
cleanest: all four recover most events under the preidentified best axis, for
24 recovered events out of 29 source events. Joint-axis candidates recover 12
of 18 residual events and are plausible for a narrower definition pass.
Second-axis candidates recover 22 of 53 residual events at best-per-target, but
no single axis cleanly resolves the host-coherent split-mixed queue.

The v2.2 exception-axis registry turns only the strong exception-axis signal
into primitive definition rows. It adds 8 exception-axis recovered coherent
subfamilies covering 24 events, raises primitive coverage from 886 to 910 of
1026 definition-core events, and reduces the residual queue from 140 to 116
events. The remaining 5 exception-axis events stay as singleton/tiny holdouts.
Second-axis and joint-axis candidates are deliberately left unpromoted because
their rule design is not yet clean enough.

The post-v2.2 option review checks whether that judgment should change before
freezing the definition. It does not. The remaining second/joint queues contain
15 targets and 44 events; current best axes recover 12 events, all as support-2
subfamilies, and no remaining target recovers most events. The operational
choice is therefore to freeze v2.2 as the current basin-definition surface and
carry the remaining second-axis, joint-axis, rule-edge, and tiny-support rows as
an explicit residual-debt ledger.

This addendum is still endpoint cartography. It does not define a final global
attraction basin and does not supply wall/pathway evidence.

The v2.2 instrumentation-surface audit checks whether the frozen definition can
serve as the next measurement surface. It preserves the 1026-event definition
universe as 910 accepted primitive events plus 116 residual-debt events, with
0 duplicate primitive-event rows. The accepted surface covers 166 source
families, of which 41 still carry residual debt; 13 definition-core families
are residual-only. Stress-test and edge-case control families remain
non-promoted, and matched stable controls contribute only context: they show
0 severe-like split or split-and-merge events, but they are not causal
validation. The next methodological unit is therefore an accepted-primitive
instrumentation panel with residual exclusions, not a v2.3 definition pass and
not a wall/pathway or quality/cost claim.

The accepted-primitive measurement panel then converts the frozen surface into
measurement rows. It has 223 primitive rows, 910 accepted event rows, and 166
accepted source families. The panel keeps residual debt as source-family
caveats: 41 accepted families carry residual debt, while residual-only families
are not converted into accepted measurement rows. The support distribution is
part of the measurement object: 82 deep-support, 72 moderate-support, and 69
thin-support primitives. Endpoint-vector fields are complete for all accepted
event rows, so the first distribution review can inspect split-vector,
host-context, shape-core, boundary-pattern, and host-handle concentration
without opening route, wall/pathway, quality, or cost claims.

The measurement distribution review turns that inspection into conservative
claim bands. The stable descriptive nucleus is 83 primitives over 452 accepted
events and 79 source families. The remaining accepted primitives require
caveats: 42 are thin but clean, 52 carry residual definition debt, and 46 have
host/shape/boundary concentration caveats. This means the first result should
claim a stable nucleus plus visible caveat classes, not a uniformly strong set
of 223 accepted primitives. Persistent mixed core supplies most of the stable
nucleus; repeat severe core remains the harder boundary class.

The claim-tier ladder makes the extension from 83 to 223 explicit. T1 is the
headline descriptive nucleus; T2 is thin-clean extension; T3 is thin with
concentration caveats; T4 is non-residual concentration caveat; T5 is standard
residual-debt caveat; and T6 is high-residual-debt audit priority. The
cumulative ladder is 83/452/79, 106/498/90, 125/536/99, 171/756/125,
218/899/161, and 223/910/166 for primitives/events/source families. This
changes result wording only and still excludes route execution, wall/pathway
promotion, basin-quality claims, cost claims, and directed-search claims.

## Precommitted Non-Field34 Panel

The next panel should be selected before route execution from the broader
calibration universe, not from current route success.

Panel input:

- start from the 206 wall-candidate calibration universe;
- exclude field34 unless the row passes a future explicit hygiene rule;
- exclude rows that are only same-control, duplicate, no-op, or tiny-support
  references;
- retain both positive-looking and negative/control cases.

Panel strata:

- strong H1 cases: high support and multiple distinct support-local pairs;
- moderate H1 cases: medium support or fewer distinct support-local pairs;
- ambiguous-definition cases: many endpoint identities but no accepted distinct
  support-local relation;
- negative/control cases: rows expected not to support wall/pathway claims.

Minimum panel shape for the next design pass:

- at least `2` fields;
- at least `3` graph or method families;
- at least `2` strong H1 cases;
- at least `1` moderate H1 case;
- at least `1` ambiguous-definition control;
- no c0-first or field34-first proof.

Recommended initial candidates from the current audit:

| role | case_id | reason |
| --- | --- | --- |
| strong H1 | `field26_gcc_emb_full_knn30_bc_cosine_budget12` | 42 strong meaningful pairs and broad endpoint support |
| strong H1 | `field26_gcc_emb_full_knn30_emb_knn_budget12` | 24 strong meaningful pairs under a different method |
| strong H1 | `field12_gcc_emb_full_knn30_emb_knn_budget12` | 14 strong meaningful pairs outside field26 |
| moderate H1 | `field30_gcc_emb_full_knn30_emb_knn_budget12` | 49 moderate meaningful pairs and useful contrast to field26 |
| moderate H1 | `field30_gcc_emb_full_knn30_bc_cosine_budget12` | 13 moderate meaningful pairs with smaller support |
| ambiguity control | `field26_gcc_emb_full_knn30_citation_embedding_budget12` | 12 endpoint identities but 0 accepted distinct support-local pairs |
| ambiguity control | `field12_gcc_emb_full_knn30_citation_embedding_budget12` | large support but 0 accepted distinct support-local pairs |

These rows are panel candidates, not final selections. A script should select
the panel from machine-readable audit tables and record why each row was kept
or rejected.

## Wall And Pathway Evidence Requirements

Wall/pathway analysis is allowed only for accepted `distinct` basin pairs from
the precommitted panel.

A wall/pathway row must carry:

- source basin candidate;
- target basin candidate;
- endpoint identity evidence grade;
- support-local relation status;
- route family;
- direct path availability;
- objective debt evidence;
- debt recovery evidence;
- polish reversion evidence;
- support incompatibility evidence;
- final endpoint assignment after route;
- route label and confidence.

Supported wall evidence requires at least one primary signal and one consistency
check:

Primary signals:

- failed direct transition between accepted distinct basin candidates;
- objective debt that is paid before reaching the target relation;
- polish reversion from target-like pre-polish state back to source or unknown;
- support incompatibility that prevents smooth interpolation.

Consistency checks:

- endpoint assignment after route is measured, not inferred from support alone;
- result is not field34 hygiene-limited;
- result is not explained by same-control behavior;
- route label does not depend on final quality.

Route-order stability alone is not wall evidence.

## Route Label v0

The route label vocabulary is fixed before execution:

| label | meaning | minimum evidence |
| --- | --- | --- |
| `crosses` | route starts in source basin candidate and ends in target basin candidate after paying/recovering wall debt | accepted source/target relation, endpoint assignment, debt/recovery or failed-direct contrast |
| `bounces` | route moves toward target relation but returns to source or ambiguous basin after polish | accepted source/target relation, target-like intermediate state, polish reversion |
| `collapses` | route leaves source but lands in neither target nor source accepted basin | accepted source relation, final endpoint assignment outside target/source |
| `unknown` | route evidence is incomplete or relation remains boundary-band limited | missing endpoint assignment, relation ambiguity, or hygiene limit |

Every label must cite the evidence fields that made it eligible. If evidence is
missing, the row remains `unknown`; it must not be upgraded by interpretation.

## Execution Sequence

Phase M0: freeze methodology

- write basin primitive v0;
- write panel-selection schema;
- write wall/pathway evidence schema;
- declare stop conditions.

Phase M1: build panel without new routes

- materialize candidate rows from the existence audit and calibration universe;
- assign inclusion/exclusion reasons;
- produce `precommitted_nonfield34_panel_v0`;
- verify it contains strong, moderate, ambiguous, and control rows.

Phase M2: enrich basin evidence without quality

- attach endpoint identity and membership evidence where cached;
- attach global and support-local distance summaries;
- preserve boundary-band rows as definition evidence;
- do not join quality or cost fields.

Phase M3: wall/pathway design review

- identify which selected pairs have enough evidence to test direct path or
  route-family hypotheses;
- mark missing evidence explicitly;
- do not execute probes unless a pair passes the eligibility schema.

Phase M4: limited pathway probe

- run only predeclared route families on predeclared pairs;
- stop after the small panel;
- label routes using v0 vocabulary;
- do not rank basins by quality.

Phase M5: downstream evaluation

- join quality and cost only for pairs with accepted basin relation and
  supported or explicitly unknown wall/pathway labels.

## Stop Conditions

Stop before pathway execution if:

- the panel cannot produce non-field34 accepted distinct basin pairs;
- selected rows are dominated by boundary-band ambiguity;
- endpoint assignment is unavailable for the intended route labels;
- evidence requirements collapse back to support-distance interpretation only.

Stop before algorithm claims if:

- no route can be labeled beyond `unknown`;
- wall evidence depends on final quality;
- only c0 or field34 produces interpretable transitions;
- a compact intervention does not beat broad restart under later evaluation.

## Expected Outputs

The next implementation should produce:

- `basin_methodology_v0_config.json`;
- `precommitted_nonfield34_panel_v0.csv`;
- `precommitted_nonfield34_panel_v0_summary.json`;
- `precommitted_nonfield34_panel_v0_report.md`;
- optional `wall_pathway_evidence_schema_v0.json`.

These artifacts should live under:

`research/consensus/results/adaptive_refinement/leiden_basin_methodology_v0_20260529/`

## M1 Materialization Status

The first M1 materialization has been generated at:

`research/consensus/results/adaptive_refinement/leiden_basin_methodology_v0_20260529/`

It passes the v0 panel-shape gates:

- `7` selected non-field34 cases;
- `3` fields: field12, field26, field30;
- `3` method families: bc_cosine, citation_embedding, emb_knn;
- `3` strong H1 cases;
- `2` moderate H1 cases;
- `2` ambiguous-definition controls;
- `142` accepted distinct pair candidates for M3 schema review.

This is still a panel artifact. It does not execute routes, promote wall
claims, or inspect basin quality/cost.

## M2 Evidence Enrichment Status

The first M2 enrichment has also been generated in the same result directory:

`research/consensus/results/adaptive_refinement/leiden_basin_methodology_v0_20260529/`

It adds:

- `methodology_v0_endpoint_evidence_rows.csv`;
- `methodology_v0_pair_evidence_rows.csv`;
- `methodology_v0_evidence_enrichment_summary.json`;
- `methodology_v0_evidence_enrichment_report.md`.

Current M2 counts:

- `53` endpoint evidence rows;
- `142` pair evidence rows;
- `15` endpoints with cached full-membership metadata;
- `38` endpoints with endpoint signature plus support hash evidence;
- `8` pairs with both full-membership caches available;
- `142` pairs ready for M3 schema review, all still missing wall/pathway
  evidence.

M2 remains evidence enrichment only. It does not load memberships, execute
routes, promote wall claims, or inspect basin quality/cost.

## M3 Wall/Pathway Schema Review Status

The first M3 schema review has been generated in the same result directory:

`research/consensus/results/adaptive_refinement/leiden_basin_methodology_v0_20260529/`

It adds:

- `methodology_v0_wall_pathway_schema_review_rows.csv`;
- `methodology_v0_wall_pathway_schema_review_summary.json`;
- `methodology_v0_wall_pathway_schema_review_report.md`.

Current M3 counts:

- `142` enriched pair rows reviewed;
- `5` rows joined to the existing 23-pair review surface;
- `137` rows remain M2-only with no existing pathway review surface;
- `2` existing partial-wall protocol references need trace audit against the
  v0 schema before any new probe;
- `3` existing route references remain blocked or insufficient for a supported
  wall/pathway label;
- `0` pairs are M4 probe-ready.

M3 remains schema review only. It does not execute routes, promote wall claims,
or inspect basin quality/cost. Support-distance distinctness is treated as
basin-pair relation evidence, not wall evidence.

## M4a Partial-Wall Trace Audit Status

The first M4a trace audit has been generated in the same result directory:

`research/consensus/results/adaptive_refinement/leiden_basin_methodology_v0_20260529/`

It adds:

- `methodology_v0_partial_wall_trace_audit_schedule_rows.csv`;
- `methodology_v0_partial_wall_trace_audit_pair_rows.csv`;
- `methodology_v0_partial_wall_trace_audit_summary.json`;
- `methodology_v0_partial_wall_trace_audit_report.md`.

Current M4a counts:

- `2` field26 partial-wall protocol references audited;
- `6` schedule rows audited;
- `2` pairs pass W1-W6 as
  `crosses_reference_schedule_stable_target_polish`;
- `2` pairs remain `not_promoted_constructed_pathway_reference_only`;
- `0` new route executions;
- `0` wall promotions.

The allowed claim is now narrower and more precise: schedule-stable constructed
pathway references exist for two field26 pairs under the v0 trace audit. The
forbidden claim remains a supported wall, basin-quality result, cost result, or
validated basin-tunneling operator. The reason is substantive: objective
debt/recovery and target endpoint assignment are present, but support
incompatibility is not observed and the post-polish support footprint remains
target-like rather than exact.

## Claim Ladder

| evidence state | allowed claim | forbidden claim |
| --- | --- | --- |
| H1 candidate evidence only | observed support-local basin candidates exist under declared thresholds | final global basin count |
| precommitted panel passes | the phenomenon is not selected from route success | pathway method works |
| accepted distinct pairs plus wall evidence | some candidate basin pairs have wall/pathway structure | better basin found |
| route labels assigned | route family can cross, bounce, collapse, or remain unknown for declared pairs | validated basin-tunneling algorithm |
| quality/cost joined after wall gates | downstream evaluation of basin usefulness | redefining basin by quality |

The strongest near-term paper direction is basin cartography: endpoint
identities, support-local basin candidates, boundary bands, and wall evidence
requirements. A directed basin-search method is a later claim.
