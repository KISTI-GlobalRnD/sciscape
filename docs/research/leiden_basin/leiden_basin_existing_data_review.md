# Leiden Basin Existing Data Review

Status: basin-definition-only artifact review
Date: 2026-05-28
Scope: existing Track C artifacts only; no quality, materiality, operator, or
control-success claims

## Correction

This document intentionally ignores quality, materiality, regret, operator
success, and cost. Those are downstream questions. The only question here is:

> What does the existing data let us call a basin, if anything?

The answer is narrower than the previous draft:

- existing artifacts strongly show many distinct **endpoint identities**;
- existing artifacts show many **support-local separations** among endpoints;
- existing artifacts do not yet fix a final basin definition;
- exact partition hashes are too fine to be accepted as basin ids by themselves;
- support distance is a basin-relation signal, not basin identity, unless we
  explicitly define a support-local basin.

## Primitive Objects

The current primitive definition from
`docs/research/leiden_basin/leiden_basin_cartography_redesign.md` is:

`observed_basin = cluster(final_polished_endpoint_memberships | fixed case_id, fixed endpoint protocol)`

For the existing artifacts, this must be decomposed into smaller objects:

| Object | Meaning | Existing evidence |
| --- | --- | --- |
| `endpoint_identity` | exact final partition identity, usually a canonical membership hash | `p5_basin_signature` |
| `global_endpoint_distance` | sampled coassignment distance over the endpoint partition | `sample_coassignment_distance` |
| `support_relation` | distance between changed-support footprints | `changed_node_support_jaccard_distance`, `coarse_support_distance` |
| `coarse_grouping` | a thresholded grouping of endpoint identities | `same_coarse_basin`, `coarse_basin_id` |
| `route_trace` | path movement, wall, debt, recovery, target progress | transition artifacts only; not basin identity |

Only `coarse_grouping` can approximate an observed basin, and only after its
distance metric and threshold are declared. `endpoint_identity` is an endpoint,
not a basin. `support_relation` is a relation between endpoints, not a basin.

## Evidence Grade Reclassification

| Artifact family | What it can support now | What it cannot support now |
| --- | --- | --- |
| field30 signature review | exact endpoint identities and thresholded support-local coarse groups | final basin count |
| field26 citation-embedding signature review | exact endpoint identities and thresholded support-local coarse groups | final basin count |
| pairwise basin matrix | endpoint-distance/support-distance disagreement analysis | quality or material basin claims |
| threshold sensitivity | evidence that basin count depends on threshold choice | one canonical basin threshold |
| crossfield reachability audit | whether a target endpoint signature appears in available vanilla rows | full attraction-region reachability |
| c0 branch/tunneling artifacts | route traces and support-relation movement | basin identity |
| failed direction ledger | examples where support/route signals were misleading or insufficient | basin definition by itself |

## What Field30 Shows

Field30 has 48 p5-labeled endpoint rows across five graph/method cases.

At the exact endpoint-signature level:

- `p5_basin_signature` count sums to `48`;
- this means the current artifact sees 48 distinct endpoint identities;
- it does **not** mean 48 basins unless the basin definition is exact endpoint
  identity, which is probably too fine.

At the current moderate coarse threshold:

- `endpoint_tau = 0.02`;
- `support_tau = 0.5`;
- field30 has 44 coarse support-local groups across 48 p5-labeled endpoints;
- this is evidence of substantial support-local endpoint separation.

But threshold sensitivity is large:

- at looser support thresholds, some cases collapse strongly;
- at strict support thresholds, almost every endpoint remains separate;
- therefore the basin count is currently a function of threshold choice.

The important definition-level observation is not "field30 has N basins." The
safer observation is:

> field30 exposes many exact endpoint identities whose changed-support
> footprints are often far apart, while global sampled coassignment distances
> can remain very small.

That means a global partition-distance basin and a support-local basin may not
be the same object.

## What Field26 Shows

Field26 citation-embedding has 12 p5-labeled endpoint rows.

At the exact endpoint-signature level:

- `p5_basin_signature` count is `12`;
- this means 12 distinct endpoint identities.

At the current moderate coarse threshold:

- `endpoint_tau = 0.02`;
- `support_tau = 0.5`;
- all 12 endpoint identities remain separate.

But the threshold-sensitivity artifact shows the same warning:

- at `support_tau = 0.75`, the coarse count can collapse depending on
  endpoint-threshold settings;
- at looser endpoint/support rules, all endpoints can collapse into one group.

So field26 supports the same conclusion as field30:

> existing evidence is strong for support-local endpoint separation, but the
> final basin definition still depends on the metric and threshold.

## Metric Conflict Exposed By Existing Data

The existing pairwise matrices expose the core definition problem.

For field30:

- pairwise rows: `223`;
- mean support distances by case are roughly `0.588` to `0.786`;
- maximum sampled coassignment distances are tiny by comparison, roughly
  `0.000057` to `0.001329`.

For field26:

- pairwise rows: `66`;
- mean support distance is roughly `0.629`;
- maximum sampled coassignment distance is roughly `0.000013`.

This creates a critical choice:

1. If basin means **global partition basin**, the current support-local
   separations may be too small at whole-graph scale.
2. If basin means **support-local endpoint basin**, the current artifacts are
   much stronger, but the definition becomes local and must say so explicitly.

This is the central unresolved definition issue.

## Current Basin-Only Claim Status

| Possible claim | Status | Reason |
| --- | --- | --- |
| The artifacts contain multiple exact endpoint identities. | supported | `p5_basin_signature` differs across many p5 endpoints. |
| The artifacts contain support-local separated endpoint groups. | supported under declared thresholds | `coarse_support_distance` and `coarse_basin_id` separate many endpoints. |
| The artifacts establish a final number of basins. | not supported | threshold choice changes grouping. |
| The artifacts establish global attraction basins. | not supported | no attraction-region sampling; global endpoint distances are often tiny. |
| c0 branch/tunneling establishes basin transition. | not supported as basin identity | route/support movement is not final endpoint basin assignment. |

## Definition Options Now On The Table

Option 1: exact endpoint identity

- Definition: each unique canonical final partition signature is its own basin.
- Advantage: directly observable in existing data.
- Problem: too brittle; tiny endpoint changes become separate basins.
- Current judgment: useful as `endpoint_identity`, not basin.

Option 2: global partition basin

- Definition: final endpoints are clustered by label-invariant global partition
  distance.
- Advantage: closest to the usual partition-basin idea.
- Problem: existing sampled coassignment distances are very small, so this may
  collapse many support-distinct endpoints.
- Current judgment: theoretically clean, but may miss the phenomenon we are
  actually seeing.

Option 3: support-local observed basin

- Definition: final endpoints are clustered by changed-support or boundary
  footprint distance, optionally constrained by endpoint-distance sanity checks.
- Advantage: matches the strongest existing signal.
- Problem: this is not a global partition basin; it must be named as local.
- Current judgment: best primitive working definition if Track C is about local
  optimizer ambiguity.

Option 4: two-level basin definition

- Definition:
  - `global_observed_basin`: clustered by global partition distance;
  - `support_local_basin`: clustered by changed-support/boundary footprint.
- Advantage: prevents support-local evidence from being overclaimed as global
  basin evidence.
- Problem: more bookkeeping.
- Current judgment: safest for the next Phase 1 index.

## Recommended Primitive Definition For The Next Pass

Use the two-level definition:

```text
endpoint_identity:
  exact canonical final partition signature.

global_observed_basin:
  cluster of endpoint identities under a declared global partition-distance
  metric and threshold.

support_local_basin:
  cluster of endpoint identities under a declared changed-support or boundary
  footprint metric and threshold.

basin_relation:
  relation to a reference endpoint, such as vanilla-near or candidate-like.

route_trace:
  path evidence before final endpoint assignment; not a basin.
```

Under this definition, the existing data should be read as:

- field30/field26: strong evidence for many `endpoint_identity` values and many
  `support_local_basin` candidates;
- field30/field26: not yet evidence for many `global_observed_basin` values;
- c0 branch/tunneling: `basin_relation` and `route_trace`, not basin identity.

## Next Basin-Only Index Columns

The next artifact-only index should avoid quality fields and include only:

- `case_id`;
- `artifact_root`;
- `endpoint_protocol`;
- `endpoint_identity_count`;
- `global_distance_metric`;
- `global_distance_threshold`;
- `global_observed_basin_count`;
- `support_distance_metric`;
- `support_distance_threshold`;
- `support_local_basin_count`;
- `largest_global_basin_size`;
- `largest_support_local_basin_size`;
- `endpoint_identity_to_global_basin_method`;
- `endpoint_identity_to_support_local_basin_method`;
- `evidence_grade`;
- `ambiguity_flag`;
- `definition_notes`.

No quality, materiality, operator, cost, or control fields belong in this
basin-definition index.
