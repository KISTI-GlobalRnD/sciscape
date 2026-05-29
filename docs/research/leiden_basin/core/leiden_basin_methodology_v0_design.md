# Leiden Basin Methodology v0 Design

Status: design draft
Date: 2026-05-29
Scope: Track C basin-definition and wall/pathway methodology only

This document turns the current evidence state into a next-step methodology.
It does not define a final basin theory, run a route batch, promote wall
claims, or inspect basin quality/cost.

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
