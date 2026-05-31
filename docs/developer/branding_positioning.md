# SciScape Branding And Positioning

Status: positioning draft
Date: 2026-06-01

This document defines the product-facing identity for SciScape. It is meant to
keep README copy, app mode design, demos, and release claims aligned.

## Brand Decision

Primary identity:

> SciScape is a local-first workbench for validated science landscapes.

Expanded description:

> SciScape turns research queries or bibliographic files into reproducible
> multi-layer science landscapes, interpretable keyword and co-occurrence
> views, hierarchy and terrain maps, cluster evolution views, evidence-backed
> narratives, quality-checked artifacts, and shareable static reports.

The brand should emphasize a validated workflow, not a single visualization
surface. SciScape's strongest public promise is that a user can move from
research records to an inspectable, reproducible, and shareable landscape
without treating data cleaning, clustering, labels, and QA as disconnected
steps.

## North Star

North-star sentence:

> Turn a research corpus into a defensible science atlas: structure, meaning,
> change, evidence, and shareable narrative.

A mature SciScape run should let an analyst answer these questions without
leaving the result bundle:

1. What corpus is being analyzed?
2. How were papers connected?
3. What clusters and hierarchy levels were found?
4. What terms, labels, and aliases explain each cluster?
5. What maps show the cluster geography and hierarchy?
6. How did each cluster grow, shift, split, merge, or fade over time?
7. Which representative papers, neighbor links, and taxonomy/context evidence
   support the interpretation?
8. What narrative should a human analyst read for this cluster?
9. Is the result clean enough to inspect, cite, publish, or share?
10. Can the same result be reopened from a static bundle without hidden state?

## Definition Of A Validated Science Landscape

A SciScape science landscape is not just an image, a graph layout, or a
clustering table. It is a validated result bundle with enough structure,
meaning, evidence, and QA to support analyst review.

Minimum conceptual layers:

| Layer | What it contains | Why it matters |
| --- | --- | --- |
| Corpus layer | source query/files, records, identifiers, years, metadata | defines the analysis universe |
| Network/matrix layer | edge families, matrix definitions, weights, normalization | explains why items are connected |
| Landscape layer | clustering, hierarchy, membership, sizes, parameters | defines topic structure |
| Meaning layer | keywords, labels, aliases, acronyms, common/specific terms | makes clusters interpretable |
| Map layer | cluster map, hierarchy map, terrain/graph/embedding views, term map | makes structure navigable |
| Evolution layer | activity timelines, cluster evolution maps, topic/child trajectories, milestones | explains change over time |
| Evidence layer | representative papers, snippets, taxonomy/context, neighbor evidence, lineage | grounds interpretation in data |
| Narrative layer | cluster summaries, lifecycle reading, boundary reading, QA caveats | turns evidence into analyst-readable understanding |
| Trust layer | feature flags, validation report, contamination checks, checksums, versions | says whether the result can be trusted |
| Sharing layer | report/data.json, static viewer, exports, reproducible paths | lets others reopen the same analysis state |

If a result lacks one of these layers, SciScape should say so explicitly. Missing
data should disable a lens with a reason instead of producing an empty or
misleading surface.

## Governance Levels

SciScape decisions should be discussed at the right level. This prevents a
feature acceptance rule from being treated as a project invariant, and prevents
project identity from being weakened by implementation convenience.

| Level | Name | Role |
| --- | --- | --- |
| 0 | North Star | Why SciScape exists and what ideal state it moves toward |
| 1 | Project Invariants | Identity-level rules that should not change unless the project identity changes |
| 2 | Product Principles | App and workflow design rules that shape user-facing behavior |
| 3 | Result Contract Rules | Artifact, schema, provenance, and validation requirements |
| 4 | Feature-Specific Rules | Acceptance criteria for keywords, maps, evolution, narratives, matrices, and other lenses |
| 5 | Release And Operational Gates | Checks required before demos, releases, scale claims, or public promises |

## Project Invariants

These are the Level 1 invariants. They are intentionally few. They define what
SciScape is, not how any single feature should be implemented.

1. The corpus boundary must always be explicit.

   A SciScape interpretation is only meaningful relative to a known corpus:
   source query or file list, filters, year range, included records, dropped
   records, and source timestamps.

2. The result bundle is the source of truth.

   The UI, README, demos, and reports must advertise only what artifacts and
   schemas can support. A feature is not real because a screen has a button for
   it.

3. Validation comes before interpretation.

   A contaminated, inconsistent, or under-specified result should not be
   promoted as a curated demo, release artifact, or trustworthy narrative.

4. Local-first and privacy-preserving operation is the default.

   Local files, private corpora, and precomputed result roots must remain usable
   without remote upload. External APIs and LLMs are optional, explicit, and
   auditable.

5. Interpretation must be evidence-backed and uncertainty-aware.

   Labels, maps, and cluster narratives must be traceable to terms,
   representative papers, lineage, neighbors, temporal signals, taxonomy
   evidence, QA caveats, or explicit human edits. When evidence is weak or
   partial, the product should say so.

6. Clustering is the engine, not the whole product.

   SciScape must keep improving clustering quality and speed, but the product
   promise is the full science-landscape workflow from corpus to narrative.

7. Claims must be bounded by contracts, tests, and artifacts.

   Research functions, Science Atlas-derived surfaces, LLM-assisted outputs,
   temporal evolution methods, and scale claims become public promises only when
   artifact contracts, validation checks, and smoke or benchmark evidence
   support them.

## Product Principles

These are Level 2 rules for app and workflow design. They may evolve as the
product matures, but they should remain subordinate to the project invariants.

1. Start from the user's analysis state, not from an implementation folder.

   The user should know whether they are opening a static result, validating a
   bundle, running a live query, cleaning terms, building a matrix, or reading a
   cluster narrative.

2. Separate mode from lens.

   A mode determines how data enters and whether computation runs. A lens is a
   view over an already loaded result bundle.

3. Missing or partial features must be visible.

   The app should disable unavailable lenses with a reason. It should not hide
   absence, silently synthesize unsupported views, or imply that a pipeline ran
   when it only loaded a static artifact.

4. Analyst changes must be inspectable and replayable.

   Cleaning rules, aliases, stop terms, acronym decisions, matrix definitions,
   label overrides, and narrative edits should leave a rule log or review trail.

5. Dense evidence belongs in progressive-disclosure surfaces.

   Maps and first screens should stay readable. Detailed evidence belongs in
   drawers, inspectors, drilldowns, QA panels, and exportable reports.

6. Cleaning, matrices, evolution, and narratives are product surfaces.

   They should not be treated as hidden preprocessing or decoration. They are
   part of the path from corpus to defensible science atlas.

## Result Contract Rules

These are Level 3 rules. They define what a SciScape result bundle must carry
when it claims a capability.

1. Every complete result should carry provenance fields.

   Required provenance includes source type, query or file list, source fetch or
   conversion timestamp, record counts, edge semantics, clustering parameters,
   keyword configuration, cleaning rules, package versions, and generated
   artifact list when available.

2. Feature flags must be inferred from artifacts.

   A result should advertise keywords, term networks, matrices, temporal data,
   evolution, narratives, quality, and exports only when the corresponding files
   and schema fields exist.

3. Disabled lenses must include a reason.

   Missing data should produce states such as `missing_keywords`,
   `missing_term_network`, `missing_evolution_data`, or
   `missing_narrative_evidence`, not a blank panel.

4. QA is a first-class artifact.

   Curated demos and release-quality outputs need validation summaries for
   schema consistency, count reconciliation, contamination, empty advertised
   lenses, and feature/artifact agreement.

5. Static bundles must be reopenable without hidden state.

   `report/data.json`, viewer bundles, and exported reports should carry enough
   version and feature information to be inspected after the original run
   environment is gone.

## Feature-Specific Rules

These are Level 4 rules. They should live in dedicated specs as the feature
contracts mature, but the high-level standards are:

- Keyword and cleaning features must preserve raw, normalized, display, score,
  frequency/count, n-gram, alias/family, acronym, common-vs-specific, and
  artifact evidence when available.
- Map and hierarchy features must preserve layer identity, selection state,
  labels, visible/hidden reasons, and enough geometry metadata to audit what is
  being drawn.
- Evolution features must carry yearly activity, lifecycle fields, topic or
  child-cluster trajectories, representative works, and versioned generation
  metadata.
- Narrative features must cite or reference available evidence: labels, terms,
  representative papers, lineage, neighbors, evolution, taxonomy/context, QA
  caveats, or explicit human edits.
- Matrix features must expose row/column labels, matrix type, weighting,
  sparsity, dropped rows, and exportable values.

## Release And Operational Gates

These are Level 5 checks. They are not project philosophy, but they protect the
public promise.

- Run the local release gate before release-oriented pushes.
- Validate curated demos and result roots with the artifact contract checker.
- Run web/static smoke tests for viewer and app surfaces that are being claimed.
- Record benchmark evidence before scale claims.
- Treat blocking contamination, empty advertised lenses, and schema mismatch as
  release blockers.

## Functional Promise

Use this promise when explaining what a "science landscape" contains. It keeps
the brand broad enough to cover clustering, labeling, hierarchy, maps, temporal
evolution, and narrative interpretation.

> From paper networks to cluster narratives.

The complete product promise is:

1. Build the research network.

   Convert queries or bibliographic files into paper records, edge layers,
   matrices, and provenance-backed source tables.

2. Discover the landscape structure.

   Run multi-layer clustering, produce hierarchy levels, preserve membership
   tables, and expose cluster-size and parameter evidence.

3. Name and explain clusters.

   Extract, clean, group, and score representative terms; generate labels; keep
   common-vs-specific terms, acronym evidence, aliases, and artifact flags
   visible.

4. Map the landscape.

   Show the cluster map, hierarchy map, terrain-style map, term co-occurrence
   map, and matrix views as different lenses over the same validated result
   bundle.

5. Read change over time.

   Provide temporal activity, cluster evolution maps, topic or child-cluster
   trajectories, milestone-style summaries, and representative works by period
   when the required temporal artifacts exist.

6. Read the cluster narrative.

   Connect labels, keywords, lineage, neighboring clusters, representative
   works, taxonomy/context evidence, evolution signals, and QA caveats into an
   evidence-backed cluster narrative.

7. Validate and share the result.

   Attach feature flags, QA reports, artifact contracts, source/version
   metadata, and static report/viewer outputs so a result can be reviewed or
   shared without rerunning the pipeline.

## Feature Improvement Test

Every new feature should answer at least one of these questions:

- Does it make the corpus boundary clearer?
- Does it make network or matrix construction more inspectable?
- Does it improve clustering, hierarchy, or membership validation?
- Does it make labels, keywords, aliases, or acronyms more interpretable?
- Does it make maps easier to navigate without losing evidence?
- Does it reveal temporal change or cluster evolution more clearly?
- Does it turn evidence into a better cluster narrative?
- Does it make QA, provenance, or sharing more reliable?

And it must pass these checks before becoming a public promise:

- What artifact or schema proves that the feature exists?
- How does the UI detect that the feature is unavailable?
- What QA failure should block this feature from being advertised?
- What source/version metadata will let another user audit it?
- Is it deterministic, optional, or LLM/API-assisted?
- Does it preserve local-first operation?

## Competitive Frame

SciScape should not be positioned as a clone of any single bibliometric tool.
It should be positioned as a workflow workbench that combines selected strengths
from several tool families while keeping a narrower, auditable scope.

| Tool family | Strong public association | SciScape response |
| --- | --- | --- |
| VOSviewer | Science maps, term co-occurrence, density/overlay views, thesaurus cleaning | Match core map and term-network expectations, but differentiate through artifact contracts, reproducible pipeline outputs, and local/static deployment |
| bibliometrix / Biblioshiny | End-to-end bibliometric workflow and standard metrics | Offer a smaller, engineering-oriented workflow first: ingest, network, landscape, keywords, report, validation |
| CiteSpace / SciMAT | Temporal knowledge mapping, burst analysis, thematic evolution | Support simple temporal lenses first; defer advanced burst/evolution claims until method contracts exist |
| KnowledgeMatrix / VantagePoint | Tech mining, matrices, cleaning, thesaurus, analyst refinement | Make matrix building and cleaning first-class follow-on modes instead of hidden preprocessing utilities |
| Gephi / Cytoscape | General graph editing and flexible graph exploration | Export to graph tools; do not compete as a full graph editor |
| Connected Papers / ResearchRabbit / Open Knowledge Maps | Query or seed driven exploration | Support query-to-landscape and curated static result bundles with stronger QA/provenance |

Reference links for scope checks:

- VOSviewer: https://www.vosviewer.com/features
- Biblioshiny: https://bibliometrix.org/home/index.php/layout/biblioshiny
- CiteSpace: https://citespace.podia.com/glossary-burstness
- SciMAT: https://sci2s.ugr.es/scimat/
- VantagePoint: https://www.thevantagepoint.com/vp.html

## Why This Is Competitive

SciScape's strongest differentiator is the combination of four capabilities
under one result contract:

1. Validated landscape generation.

   Multi-layer paper-network construction, Rust CPM/Leiden clustering, and
   hierarchy outputs are packaged as an auditable landscape rather than a
   one-off clustering result.

2. Interpretable meaning surfaces.

   Keywords, display labels, n-gram behavior, common-vs-cluster-specific terms,
   acronym evidence, and term co-occurrence are treated as product surfaces.
   They are not just labels attached after clustering.

3. Artifact QA and feature inference.

   SciScape should show whether a result is safe to inspect or publish. HTML,
   LaTeX, publisher metadata, missing files, mismatched feature flags, and
   empty advertised lenses are release-quality problems, not cosmetic warnings.

4. Local-first and shareable outputs.

   CLI, Python API, local web app, and static viewer outputs let analysts keep
   private data local while still sharing compact, reproducible result bundles
   such as hosted `data.json` reports.

This is attractive because many existing tools are excellent at one slice:
mapping, bibliometric metrics, temporal review, graph editing, or tech mining.
SciScape's opportunity is to make the full analysis state reproducible and
inspectable across those slices.

## Brand Pillars

Use these pillars in public docs and app copy.

| Pillar | Meaning | Proof points |
| --- | --- | --- |
| Validated landscapes | A science map should carry its provenance and QA state | artifact contract, feature flags, QA gates, count reconciliation |
| Multi-layer clustering | Research topics should be built from explicit edge semantics | DC/BC/CC/embedding layers, consensus weighting, Rust CPM/Leiden |
| Interpretable terms and matrices | Users need terms they can audit, clean, group, and compare | keyword QA, n-gram fields, aliases, co-occurrence, matrix builder roadmap |
| Evolution and narratives | Users need to understand how clusters form, shift, connect, and matter | cluster evolution map, topic/child trajectories, milestones, representative works, neighbor evidence |
| Local-first reproducibility | Analysis should run locally and export shareable bundles | CLI, Python API, local web, static viewer, report/data.json |

## Recommended Messaging

Short tagline:

> Validated science landscapes from research records.

Interpretation tagline:

> From paper networks to cluster narratives.

README first sentence:

> SciScape is a local-first SciSci workbench for building validated science
> landscapes from research queries or bibliographic files.

One-paragraph pitch:

> SciScape combines bibliographic ingestion, multi-layer paper-network
> construction, Rust CPM/Leiden clustering, interpretable keyword and
> co-occurrence analysis, hierarchy and terrain maps, cluster evolution views,
> evidence-backed narratives, artifact QA, and static report export. It is
> designed for analysts who need a reproducible research landscape, not only an
> interactive map.

Developer-facing summary:

> SciScape's product contract is the result bundle: records, edge semantics,
> membership, hierarchy, keywords, term networks or matrices, feature flags,
> QA, and exportable reports.

## Claim Boundaries

The brand should avoid claims that are not yet supported by the current product
contract.

- Do not claim to replace VOSviewer as a mature interactive map editor.
- Do not claim full Biblioshiny-equivalent metric coverage.
- Do not claim CiteSpace/SciMAT-style burst or thematic evolution until those
  methods have separate contracts and tests.
- Do not claim institutional benchmarking without normalized metric and data
  license contracts.
- Do not claim unrestricted full-corpus keyword scale beyond the benchmark
  evidence in `keyword_extraction_scaling.md`.
- Do not expose Dongdaemun research surfaces as the default public product.
- Do not claim Science Atlas Explorer-derived surfaces are part of the stable
  SciScape app until their artifact contracts, data payloads, and smoke tests
  have been ported.

## Science Atlas Absorption Track

The standalone Science Atlas Explorer project is a source of planned app and
interpretation surfaces. These should be absorbed as SciScape capabilities only
after the corresponding artifact contracts are stable.

Planned absorption targets:

- atlas-only first screen with explicit demo/query actions;
- result rail, result evidence drawer, and search-to-hierarchy summaries;
- map and hierarchy-map stage switch;
- terrain, embedding, and graph map bases;
- strict Domain/Macro/Meso/Micro layer toggles with Nano as detail/child scope;
- selected lineage path, child overlays, same-parent and cross-parent neighbor
  evidence;
- right inspector ordered for cluster reading: identity, connection reading,
  key terms, lineage, taxonomy, children, neighbors, representative papers,
  metrics, and QA;
- map lenses for scale, growth, impact, focus, boundary, and quality where
  payloads exist;
- cluster evolution map with yearly activity, phase topics, child/topic
  trajectories, citation genealogy, representative works, and narrative folds;
- package diagnostics that show versions, feature flags, render budgets, and
  payload coverage without overwhelming the primary map.

Absorption rule:

> Science Atlas features become SciScape promises only when the SciScape result
> bundle can validate the required artifacts and disable the corresponding lens
> with a visible reason when data is missing.

## Product Implications

The branding should drive near-term implementation choices:

- The first screen should start from mode selection: demo, static result, local
  result, live query, validation.
- Lenses should be enabled from artifact features, not hard-coded UI optimism.
- Term co-occurrence and matrix views are core to the brand, not optional
  leftovers.
- Cluster evolution and narrative views should become first-class lenses, but
  only when temporal/evolution artifacts are present.
- Cleaning audit and rule replay should become a visible analyst workflow.
- Every curated demo should show provenance, feature availability, and QA state.
- Static sharing should remain a first-class path because it makes the
  landscape portable without a server.

## Ideal Product State

The ideal SciScape experience is:

1. The user starts from a query, files, or a precomputed result bundle.
2. SciScape validates the input/result and explains what lenses are available.
3. The user sees an atlas-like map first, not an implementation folder.
4. The user can move between global map, hierarchy map, terrain/graph basis,
   term co-occurrence, matrix, and temporal/evolution views.
5. Selecting a cluster reveals identity, labels, terms, lineage, children,
   neighbors, representative papers, metrics, QA caveats, and evolution.
6. The cluster narrative is generated or displayed as an evidence-backed
   reading, with sources and uncertainty visible.
7. The entire result can be exported as a compact, static, reproducible bundle.

This ideal should guide implementation order. A feature is more valuable when
it moves the user closer to this complete loop instead of adding another
isolated chart, threshold, or internal option.

## Brand Architecture

Keep the public hierarchy simple:

- Product name: SciScape.
- Public identity: local-first workbench for validated science landscapes.
- Core engine: Rust CPM/Leiden and multi-layer landscape construction.
- Meaning layer: keyword, acronym, co-occurrence, and matrix mining.
- Narrative layer: hierarchy, terrain, evolution, neighbor evidence, and
  cluster narratives.
- Trust layer: artifact QA, feature inference, and reproducible bundles.
- Research-only names: Dongdaemun family names stay internal or explicitly
  opt-in until their claim boundaries are satisfied.
