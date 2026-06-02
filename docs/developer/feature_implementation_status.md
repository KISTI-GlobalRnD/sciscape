# SciScape Feature Implementation Status

Date: 2026-06-01

Scope: current code-surface audit against `docs/developer/feature_definition.md`.
This is not a benchmark, release certification, or product promise. It only
answers whether each feature area has real implementation in the repo today and
how complete that implementation appears to be.

## Legend

- `[x] Implemented`: usable code exists for the current contract, with tests or
  artifact checks covering the main path.
- `[~] Partial`: meaningful code exists, but a required artifact, workflow,
  schema, app surface, or QA gate is still missing.
- `[ ] Missing`: no production implementation beyond docs, stubs, or inferred
  artifact detection.
- `[d] Deferred`: intentionally outside the near-term implementation scope.

Completeness is a rough implementation estimate from repo evidence. It is not a
quality score and should be revised after each milestone.

## Executive Snapshot

The strongest implemented areas are the core analysis pipeline: input adapters,
OpenAlex query execution, link construction, Rust-backed clustering,
postprocess/landscape artifacts, keyword extraction and cleaning, static report
generation, local result loading, artifact validation, and graph export.

The weakest areas are product-level workspace management, first-class matrix
builder mode, cluster evolution maps, evidence-backed narrative generation, and
institutional analytics. These should not be presented as complete app modes
until their artifact contracts, UI surfaces, and validation checks are added.

## Summary Checklist

| ID | Feature Area | Status | Rough Completeness | Current Evidence | Main Missing Piece |
|---|---:|---:|---:|---|---|
| F01 | Workspace and project management | `[~]` | 35% | local result discovery, job store, demo presets | stable workspace/project/run manifest model |
| F02 | Ingest and normalize | `[~]` | 55% | WoS, Scopus, OpenAlex, BibTeX adapters; OpenAlex query pipeline | broader source coverage and normalized entity model |
| F03 | Demo, static viewer, local result loading | `[x]` | 80% | demo manifest, local result open, report/atlas attach, quality gate | workspace-level browsing and UX polish |
| F04 | Live query and job execution | `[~]` | 65% | `/api/query`, job status, SSE, OpenAlex pipeline output | cancellation, retry policy, partial artifact recovery UI |
| F05 | Network construction | `[x]` | 75% | DC/BC/CC builders, edge combination, filters, OpenAlex citation edges | first-class entity networks and richer evidence artifacts |
| F06 | Matrix builder | `[~]` | 25% | sparse matrix internals, co-occurrence helpers, artifact feature detection | explicit matrix-builder mode, schema, writer, validator |
| F07 | Clustering and hierarchy | `[x]` | 85% | Rust CPM/Leiden path, hierarchy, landscape, membership artifacts | app-level parameter workflow and expensive-run guardrails |
| F08 | Keyword extraction, labels, cleaning | `[~]` | 75% | pipeline, quality filters, abbreviation handling, term network, scaling docs | editable/replayable cleaning rules and full large-run benchmark |
| F09 | Atlas map, evidence, cluster reading | `[~]` | 70% | atlas payload builder, neighbors, representative works, web endpoints | UI rewrite and complete evidence inspector workflow |
| F10 | Term network and co-occurrence visualization | `[~]` | 70% | term network module, endpoint, co-occurrence evidence tables | map polish, threshold controls, export contract |
| F11 | Temporal and evolution | `[~]` | 30% | temporal keyword utilities, burst/trend helpers, feature detection | cluster evolution artifact contract and map UI |
| F12 | Evidence-backed narratives | `[ ]` | 15% | narrative feature detection and target definition | narrative generator, evidence references, QA contract |
| F13 | Validation and QA | `[x]` | 80% | artifact contract, result validation, quality gate, keyword artifact checks | strict checks for matrix/evolution/narrative features |
| F14 | Report, export, interoperability | `[~]` | 70% | HTML reports/viewer, dashboard export, GEXF, GraphML, static data | VOSviewer-style exports and export manifest completeness |
| F15 | Institutional analytics | `[d]` | 0% | target definition only | intentionally deferred after analyst workbench maturity |

## Code Surface Inventory

- CLI: `query`, `cluster`, `keywords`, `convert`, `landscape`, `visualize`,
  `viewer`, `export`, `web`, and `gui` commands exist in `sciscape/cli.py`.
- Web API: query jobs, job status, SSE stream, local data discovery/open,
  labels, temporal, bridge, term network, cluster details, label merges,
  treemap, abbreviations, consensus, what-if, quality, and export endpoints
  exist in `sciscape/web/app.py`.
- Artifact contract: result feature inference, local result validation,
  keyword artifact scanning, edge evidence samples, and atlas payload building
  exist in `sciscape/artifacts.py`.
- Keyword stack: extraction, normalization, quality annotation, abbreviation
  extraction, vocabulary cleansing, term networks, co-occurrence, diagnostics,
  depth, temporal, burst, and visualization helpers exist under
  `sciscape/keyword_extraction/`.
- Network stack: direct citation, bibliographic coupling, co-citation,
  filtering, combination, diagnostics, and OpenAlex edge conversion exist under
  `sciscape/linkage/` and `sciscape/openalex/`.
- Visualization/export: dashboard, report, viewer, hierarchy, network map,
  temporal charts, GEXF, and GraphML surfaces exist.
- Tests cover the main implemented surfaces, including artifacts, web app, demo
  gate, landscape, term networks, co-occurrence, temporal logic, adapters,
  OpenAlex edges, and network maps.

## Feature Detail

### F01. Workspace And Project Management

Status: `[~]` Partial. Rough completeness: 35%.

- `[x]` Local result roots can be discovered and opened from the web app.
- `[x]` Query jobs have persisted status and output directories.
- `[x]` Demo presets point users to prepared local result artifacts.
- `[~]` Result-root validation can infer available features from files.
- `[ ]` Workspace, project, dataset, run, rule-set, view, and export objects are
  not formalized as a single manifest model.
- `[ ]` There is no durable workspace browser with rename, archive, compare, or
  provenance editing semantics.

Review: this is enough for developer and demo operation, but not enough for the
"analyst workspace" product promise.

### F02. Ingest And Normalize

Status: `[~]` Partial. Rough completeness: 55%.

- `[x]` WoS, Scopus, OpenAlex, and BibTeX conversion paths exist.
- `[x]` OpenAlex live query can fetch works and reconstruct abstract text.
- `[x]` OpenAlex citation edges can be built from fetched works.
- `[~]` Title/abstract metadata cleaning exists mainly through keyword utilities
  and artifact filters.
- `[ ]` PubMed, Dimensions, Lens, Crossref, and generic CSV schema mapping are
  not complete first-class adapters.
- `[ ]` Institution, author, funder, venue, patent, and topic entities are not
  normalized as a unified entity model.

Review: usable for current paper-centric workflows, but not a complete
bibliometric ingest layer.

### F03. Demo, Static Viewer, And Local Result Loading

Status: `[x]` Implemented. Rough completeness: 80%.

- `[x]` Demo preset manifest support exists.
- `[x]` Missing expected demo artifacts can be detected.
- `[x]` Local result paths can be opened and inferred by the web app.
- `[x]` Report data can be attached to atlas payloads.
- `[x]` Static viewer/report export paths exist.
- `[~]` The local data workflow is still file/path oriented rather than
  workspace-object oriented.

Review: this is one of the safest near-term app surfaces to expose.

### F04. Live Query And Job Execution

Status: `[~]` Partial. Rough completeness: 65%.

- `[x]` Web query submission exists through `/api/query`.
- `[x]` Job status and job list endpoints exist.
- `[x]` SSE job streaming exists.
- `[x]` OpenAlex query execution writes result artifacts through the pipeline.
- `[~]` Download/view endpoints expose job outputs.
- `[ ]` Robust cancel, retry, quota handling, resumable partial artifacts, and
  shard-aware long-run controls are not complete app features.

Review: usable for small to medium live demos, but not enough for unattended
large-scale jobs without additional run controls.

### F05. Network Construction

Status: `[x]` Implemented. Rough completeness: 75%.

- `[x]` Direct citation, bibliographic coupling, and co-citation builders exist.
- `[x]` Edge filtering, normalization, and combination utilities exist.
- `[x]` OpenAlex fetched works can be converted into citation edges.
- `[x]` Network artifacts can feed clustering, landscape, and export surfaces.
- `[~]` Edge evidence sampling exists, but is not yet a complete user-facing
  evidence system for every network mode.
- `[ ]` Author, institution, venue, keyword, and patent networks are not complete
  first-class app modes.

Review: strong for paper-level citation/coupling networks; broader bibliometric
network families are still future work.

### F06. Matrix Builder

Status: `[~]` Partial. Rough completeness: 25%.

- `[x]` Sparse matrix internals are used by BC/CC and keyword extraction.
- `[x]` Co-occurrence table construction exists for keyword terms.
- `[~]` Artifact feature inference can recognize matrix-like outputs.
- `[ ]` There is no explicit matrix-builder command, app mode, schema, writer, or
  validator.
- `[ ]` Matrix rows/columns, normalization, thresholding, and projection metadata
  are not stored as a replayable artifact contract.

Review: implementation primitives exist, but the Matrix Builder should be marked
as not product-complete.

### F07. Clustering And Hierarchy

Status: `[x]` Implemented. Rough completeness: 85%.

- `[x]` Clustering command exists.
- `[x]` Rust-backed CPM/Leiden path exists in the core workflow.
- `[x]` Membership and hierarchy artifacts are validated and used by atlas
  payload construction.
- `[x]` Landscape/postprocess artifacts are first-class in result inference.
- `[x]` Tests cover key landscape and artifact behavior.
- `[~]` App-level parameter selection, cost-aware guardrails, and large-run
  cancellation/restart UX are still incomplete.

Review: this is a core implemented capability, with remaining work mainly around
operator controls and large-run safety.

### F08. Keyword Extraction, Labels, And Cleaning

Status: `[~]` Partial. Rough completeness: 75%.

- `[x]` Keyword extraction pipeline exists.
- `[x]` Quality annotation, artifact filtering, representative scoring,
  abbreviation extraction, and vocabulary cleansing exist.
- `[x]` Term network and co-occurrence helpers exist.
- `[x]` Scaling and sharding design has been documented.
- `[~]` Label merge and LLM labeling endpoints exist, but the full review loop is
  not yet a polished app workflow.
- `[ ]` Editable cleaning rule sets, replayable rule history, and rule-impact
  diffs are not complete.

Review: the algorithmic base is strong enough to keep improving, but the analyst
control surface is still incomplete.

### F09. Atlas Map, Evidence, And Cluster Reading

Status: `[~]` Partial. Rough completeness: 70%.

- `[x]` Atlas payload can be built from report and membership artifacts.
- `[x]` Hierarchy lineage, doc counts, labels, representative works, and neighbor
  evidence can be attached.
- `[x]` Cluster detail and atlas-related web endpoints exist.
- `[~]` Evidence is available in payloads, but not yet fully organized into a
  consistent inspector workflow.
- `[ ]` The UI/UX rewrite is still pending and should not be considered complete.

Review: the data contract is ahead of the interface. This should guide the next
UI implementation rather than be treated as a finished Atlas App.

### F10. Term Network And Co-Occurrence Visualization

Status: `[~]` Partial. Rough completeness: 70%.

- `[x]` Term network construction exists.
- `[x]` Co-occurrence collection exists.
- `[x]` `/api/jobs/{job_id}/term-network` endpoint exists.
- `[x]` Term/co-occurrence evidence can be surfaced from keyword artifacts.
- `[~]` Visualization controls for thresholding, layout, clustering, and export
  are not yet complete.
- `[ ]` A stable co-occurrence map/table artifact contract is not fully defined.

Review: the previously broken co-occurrence path has enough implementation
surface to stabilize, but needs an explicit artifact contract and UI QA.

### F11. Temporal And Evolution

Status: `[~]` Partial. Rough completeness: 30%.

- `[x]` Temporal keyword utilities and visualization helpers exist.
- `[x]` Burst and trend helpers exist.
- `[x]` Temporal tracking endpoints exist in the web app.
- `[~]` Artifact feature inference can detect some temporal/evolution payloads.
- `[ ]` Cluster evolution map artifacts, lineage transitions, split/merge
  events, and stability scores are not implemented as a stable contract.
- `[ ]` Evolution map UI is not product-ready.

Review: temporal keyword views exist, but the promised cluster evolution map is
still mostly design-level.

### F12. Evidence-Backed Narratives

Status: `[ ]` Missing. Rough completeness: 15%.

- `[~]` Narrative is named in the feature definition and artifact feature
  inference can detect narrative-like payloads.
- `[ ]` There is no stable narrative generation pipeline.
- `[ ]` There is no required evidence-reference schema for narrative claims.
- `[ ]` There is no narrative QA gate for unsupported claims, hallucination risk,
  or missing citations.
- `[ ]` There is no final app surface for cluster narrative review.

Review: this should remain a future feature until the evidence reference
contract and QA gate exist.

### F13. Validation And QA

Status: `[x]` Implemented. Rough completeness: 80%.

- `[x]` Result-root artifact validation exists.
- `[x]` Feature availability blocks can be inferred from artifacts.
- `[x]` Keyword artifact contamination checks exist.
- `[x]` Demo quality gate exists.
- `[x]` Tests cover major validation paths.
- `[~]` Matrix, evolution, narrative, and export-manifest validation are not yet
  strict enough for product claims.

Review: this is a strong foundation and should remain the gatekeeper before
features are exposed in the UI.

### F14. Report, Export, And Interoperability

Status: `[~]` Partial. Rough completeness: 70%.

- `[x]` Dashboard, report, and viewer export helpers exist.
- `[x]` CLI export supports GEXF and GraphML.
- `[x]` Static report data can be used by the web viewer.
- `[~]` Artifact contracts can summarize export availability.
- `[ ]` VOSviewer-style map/network exports are not clearly complete.
- `[ ]` Export manifests do not yet describe every exported view, filter, and
  data transformation.

Review: good enough for graph/report exports, but not complete as an
interoperability layer.

### F15. Institutional Analytics

Status: `[d]` Deferred. Rough completeness: 0%.

- `[ ]` Organization, department, collaboration, benchmark, portfolio, and
  strategy views are not implemented as product features.
- `[ ]` Institutional dashboards and decision reports are not implemented.
- `[ ]` Data governance, role-based visibility, and institutional entity
  resolution are not implemented.

Review: keep this deferred until the core analyst workbench, matrix builder,
evolution map, and narrative system are stable.

## Current Release Exposure Guidance

- Safe to expose now: local result loading, demo presets, static reports,
  OpenAlex query for bounded examples, paper-level networks, clustering,
  keyword extraction, term-network evidence, graph export, and artifact QA.
- Expose as beta: live query jobs, Atlas evidence reading, label review/merge,
  temporal keyword views, and co-occurrence visualization.
- Do not expose as complete: workspace projects, matrix builder mode, cluster
  evolution map, evidence-backed narratives, institutional analytics, and full
  VOSviewer-compatible interoperability.

## Next Implementation Targets

1. Define `workspace.json` or equivalent project/run manifest.
2. Define a first-class matrix artifact contract with writer and validator.
3. Define replayable keyword cleaning rule artifacts and before/after diffs.
4. Define cluster evolution artifact schema before building the map UI.
5. Define narrative evidence-reference schema before adding generation.
6. Add export manifests for every generated report, graph, table, and map.
