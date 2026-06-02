# SciScape Analysis Feature Priorities

Date: 2026-06-01

Scope: redesigned priority plan for growing SciScape around analysis
capabilities. This document assumes the feature contract in
`feature_definition.md` and the current implementation audit in
`feature_implementation_status.md`.

## Priority Rules

1. Analysis artifacts come before UI promises.
2. A result must be discoverable without knowing the internal folder layout.
3. Every exposed feature must have a readable artifact contract, validation path,
   and at least one small reproducible example.
4. Long-running analysis must expose progress, partial outputs, failure state,
   and resume/checkpoint metadata before it is treated as stable.
5. App UI must consume feature flags and exposure states; UI state must not
   become the source of truth.
6. Prefer strengthening the query-to-analysis-to-atlas spine before adding new
   product surfaces.
7. Avoid institutional or narrative claims until evidence references and entity
   normalization are reliable.

## Exposure States

Feature availability should be separated from implementation existence.

| State | Meaning | Required Evidence |
|---|---|---|
| `hidden` | feature should not appear in the app | missing artifact, invalid schema, or failed QA |
| `beta` | feature can appear with warnings | artifact exists, validator passes, but coverage or UX is incomplete |
| `stable` | feature can be treated as a normal product surface | artifact contract, validator, smoke fixture, and export path all pass |

The validator should decide the state. UI code should only display what the
loaded result says is available.

## Priority Bands

| Band | Role | Definition |
|---|---|---|
| P0 | Operating contract | Work required before more analysis features should be exposed |
| P1 | Query-to-Atlas core | Existing query, network, clustering, and map functions that define the product center |
| P1.5 | Interpretation reliability | Keyword cleaning, term evidence, and co-occurrence hardening needed for trustworthy cluster reading |
| P2 | General analyst workbench | Reusable matrix and cleaning workflows beyond the first cluster-reading path |
| P3 | Dynamic science map | Temporal, lifecycle, and cluster evolution analysis |
| P4 | Interpretation layer | Evidence-backed narratives and cluster reading aids |
| P5 | Domain expansion | Institutional, portfolio, and policy analytics |

## Executive Priority Order

| Rank | Priority | Feature Areas | Why Now | Done When |
|---:|---|---|---|---|
| 1 | P0. Operating contract and QA | F01, F03, F04, F13, F14 | Users already struggle to know what result folder to load, and long jobs need visible state | every result has manifest, feature states, exports, warnings, progress/checkpoint metadata, and validation status |
| 2 | P1. Query-to-Atlas core | F02, F04, F05, F07, F09, minimal F08/F10 | This is SciScape's main value: query or load papers, build a network, cluster, label, and read the map | one bounded sample can run or load end-to-end without manual file hunting |
| 3 | P1.5. Keyword and term interpretation reliability | F08, F10, minimal F06 | Cluster reading fails if labels, artifacts, abbreviations, and co-occurrence evidence are unreliable | cleaning rules, term evidence, and co-occurrence artifacts are replayable and validated |
| 4 | P2. General matrix and analyst workbench | F06, F08, F10 | This closes the gap with bibliometric tools after the first co-occurrence path is stable | matrix outputs are generic, replayable, exported, and independently validated |
| 5 | P3. Temporal and cluster evolution | F11 | Evolution map is high-value, but unsafe without stable lineage artifacts | cluster changes, growth, bursts, split/merge events, and stability are artifact-backed |
| 6 | P4. Evidence-backed narratives | F12 | Narrative is the differentiator only if it is auditable | every sentence-level claim links to terms, records, metrics, or cluster evidence |
| 7 | P5. Institutional analytics | F02, F05, F15 | Valuable later, but needs entity normalization and governance | institution/author/funder entities resolve consistently and metrics are controlled |

## P0. Operating Contract And QA

Goal: make every analysis output safe to find, load, inspect, compare, resume,
and export.

Primary work:

- Define a minimal `result_manifest.json` or equivalent manifest for result
  roots. It should include source files, run metadata, artifact paths, feature
  states, export paths, warnings, and provenance.
- Extend result manifests so `data.json`, result roots, and static reports share
  the same feature/version block.
- Add exposure states for atlas, keywords, term network, co-occurrence, matrix,
  temporal, evolution, narrative, quality, and export surfaces.
- Add export manifests for reports, graph files, tables, maps, and static viewer
  bundles.
- Add long-run status metadata: progress, heartbeat, shard state, checkpoint
  path, partial outputs, failure reason, and resume markers.
- Strengthen `sciscape_quality_gate.py` so incomplete features are hidden or
  marked beta rather than silently exposed.
- Keep demo, local, static, and small-query result loading as the canonical
  smoke paths.

Completion criteria:

- A user can open a result without knowing the internal folder layout.
- A loaded result shows what is available, what is missing, and what is unsafe.
- The app can disable unsupported lenses from artifacts alone.
- Long keyword or clustering jobs expose progress and recoverable partial state.
- Demo presets fail clearly when required artifacts are missing.
- Release checks catch keyword contamination, missing manifest fields, broken
  atlas/report payloads, and missing export references.

Do not do yet:

- Do not redesign the whole UI before feature flags and manifests are stable.
- Do not add narrative or evolution UI based on inferred or partial files.
- Do not treat a long-running job as stable if it can run for hours without
  visible intermediate artifacts.

## P1. Query-To-Atlas Core

Goal: make the current "query or load papers -> network -> clustering -> map ->
cluster reading" workflow coherent.

Primary work:

- Treat bounded OpenAlex query and local result loading as equal first-class
  entry points.
- Keep paper-level network construction as the default backbone: direct citation,
  bibliographic coupling, co-citation, and combined edges.
- Keep Rust CPM/Leiden and hierarchy generation as the main clustering path.
- Make the minimal keyword path produce interpretable cluster labels, common
  terms, cluster-specific terms, representative phrases, abbreviation evidence,
  and artifact flags.
- Make Atlas Map read cluster identity, size, hierarchy, representative works,
  neighbor relations, shared terms, and QA warnings from one payload.
- Stabilize minimal term co-occurrence output as a supporting analysis artifact
  for cluster interpretation.

Completion criteria:

- A bounded sample query can run to a saved result root.
- A local sample result can be loaded without manual file hunting.
- Required result artifacts are discoverable through the manifest.
- Each cluster has interpretable labels, representative works when joinable, and
  enough neighbor/term evidence to explain its position.
- Term co-occurrence is available as a validated table artifact, even before the
  full matrix workbench exists.
- One smoke fixture exercises network, clustering, hierarchy, keywords, atlas,
  co-occurrence, export manifest, and quality gate.

Do not do yet:

- Do not make advanced clustering controls a public default.
- Do not optimize every network family before the paper-level analysis spine is
  stable.
- Do not expose a polished map if core labels or artifact warnings are failing.

## P1.5. Keyword And Term Interpretation Reliability

Goal: make cluster interpretation trustworthy enough for real research use.

Primary work:

- Define replayable cleaning rule artifacts for stop terms, aliases, acronym
  expansion, spelling normalization, artifact blocking, metadata fragments, and
  family grouping.
- Persist before/after impact summaries for cleaning and normalization rules.
- Keep parenthetical abbreviation extraction and acronym evidence visible in the
  output.
- Separate common terms, cluster-specific terms, representative labels, and
  supporting abbreviations.
- Stabilize co-occurrence table/map artifacts separately from the general Matrix
  Builder.
- Add contamination fixtures for encoded HTML, publisher metadata, LaTeX
  preamble fragments, and ambiguous short tokens.
- Add large-run safety for keyword extraction: shard state, checkpoints, partial
  table writes, failure markers, and resumability.

Completion criteria:

- Cleaning changes are inspectable, replayable, and reversible at artifact
  level.
- A contaminated input fixture cannot surface blocked artifacts as top labels.
- Co-occurrence table/map artifacts validate independently of the UI.
- Large keyword runs produce intermediate artifacts and progress state.
- The app can show why a term is representative, common, cluster-specific,
  auxiliary, or blocked.

Do not do yet:

- Do not make manual cleaning edits that cannot be replayed.
- Do not merge co-occurrence hardening into the full generic Matrix Builder.
- Do not rely on free-form LLM cleanup without deterministic artifacts.

## P2. General Matrix And Analyst Workbench

Goal: turn lower-level matrix primitives into reusable analyst workflows.

Primary work:

- Define a general matrix artifact contract for occurrence, co-occurrence,
  proximity, similarity, and projection outputs.
- Add matrix writer and validator utilities after the first term co-occurrence
  artifact contract is stable.
- Add matrix-level provenance: source table, row entity, column entity,
  normalization, threshold, projection, and filter history.
- Make term network and co-occurrence views consume matrix artifacts rather than
  ad hoc in-memory tables.
- Add matrix exports that can support external tools and internal reproducible
  reports.

Completion criteria:

- A user can rebuild the same co-occurrence or similarity result from saved
  rules and matrix metadata.
- Matrix outputs can be exported and validated independently from the UI.
- Matrix views can be compared across cleaning rule sets or analysis runs.

Do not do yet:

- Do not expose a generic Matrix Builder UI until matrix schemas and QA checks
  are stable.
- Do not mix every entity-network feature into the first matrix milestone.

## P3. Temporal And Cluster Evolution

Goal: move from static maps to how fields change over time.

Primary work:

- Stabilize temporal trend artifacts for documents, keywords, clusters, and term
  families.
- Define cluster evolution artifacts: time slice, lineage, split, merge,
  emergence, decline, stability, and evidence rows.
- Add growth and burst summaries as analysis artifacts.
- Build a small evolution smoke example before adding the full UI lens.
- Keep the evolution map separate from generic temporal charts until lineage
  evidence is trustworthy.

Completion criteria:

- The same cluster can be followed across time slices with explicit evidence.
- Split, merge, emergence, and decline events have reproducible input rows.
- Evolution UI can be enabled or disabled by feature states.

Do not do yet:

- Do not present yearly keyword charts as "cluster evolution" unless cluster
  lineage has been computed.
- Do not generate narratives about change before evolution evidence exists.

## P4. Evidence-Backed Narratives

Goal: help users understand clusters without turning interpretation into
unsupported prose.

Primary work:

- Define a narrative artifact schema with claim IDs, evidence references,
  confidence, unsupported-claim flags, and source rows.
- Generate cluster summaries from terms, representative works, temporal signals,
  neighbor evidence, and quality warnings.
- Add a narrative QA gate that rejects claims without attached evidence.
- Expose narrative as a review layer, not as the source of cluster truth.

Completion criteria:

- Every cluster narrative claim links to concrete records, terms, metrics, or
  relation evidence.
- Missing or weak evidence is visible to the user.
- Narrative artifacts can be regenerated after cleaning, matrix, or evolution
  changes.

Do not do yet:

- Do not ship free-form LLM cluster descriptions as product output.
- Do not describe institutional strategy or field change without evidence-backed
  metrics.

## P5. Institutional And Portfolio Analytics

Goal: expand from science-map exploration into decision support.

Primary work:

- Define entity normalization for authors, affiliations, institutions, funders,
  venues, countries, patents, and grants.
- Add controlled institutional metrics only after entity resolution is auditable.
- Add collaboration, portfolio, benchmark, and opportunity views as derived
  analyses.
- Keep institution-specific dashboards separate from core cluster-analysis mode.

Completion criteria:

- Entity identity is stable enough for comparison.
- Metrics specify denominator, time window, source coverage, and uncertainty.
- Institutional reports can be reproduced from artifacts.

Do not do yet:

- Do not claim institutional benchmarking with paper-only cluster artifacts.
- Do not mix policy recommendations into the core analysis app.

## Next Concrete Work Queue

1. Define the minimal `result_manifest.json` schema.
2. Write and validate manifest generation for demo, static, local, and query
   outputs.
3. Add exposure states and export manifest fields to result-root validation.
4. Add long-run progress, shard, checkpoint, partial-output, and resume metadata
   contracts for live query and keyword extraction.
5. Add a smoke fixture that exercises query/local load, network, clustering,
   keywords, atlas, co-occurrence, export manifest, and quality gate.
6. Add replayable keyword cleaning rule artifacts and before/after impact
   summaries.
7. Add the stable co-occurrence table/map artifact contract.
8. Make Atlas evidence inspector consume only validated payload fields and
   exposure states.
9. Add general matrix writer and validator utilities after co-occurrence
   artifacts are stable.
10. Define temporal trend artifact schema.
11. Define cluster evolution artifact schema and one synthetic smoke example.
12. Define narrative evidence-reference schema without generation.
13. Revisit UI/UX redesign only after P0, P1, and P1.5 contracts are stable.

## Priority Review Cadence

After each milestone, update `feature_implementation_status.md` before adding
new user-facing scope. A feature can move upward only when its artifacts, tests,
feature states, and UI behavior agree.
