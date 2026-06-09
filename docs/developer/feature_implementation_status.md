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

Use the summary checklist as the canonical current status. Detail sections must
match the summary percentages after each implementation slice; when a mismatch
is found, prefer the current code evidence over older review text.

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
| F01 | Workspace and project management | `[~]` | 53% | local result discovery, job store, demo presets, workspace manifest design, writer/validator, legacy result registration, workspace-first local data API | durable browser and Home workspace UX |
| F02 | Ingest and normalize | `[~]` | 55% | WoS, Scopus, OpenAlex, BibTeX adapters; OpenAlex query pipeline | broader source coverage and normalized entity model |
| F03 | Demo, static viewer, local result loading | `[x]` | 80% | demo manifest, local result open, report/atlas attach, quality gate | workspace-level browsing and UX polish |
| F04 | Live query and job execution | `[~]` | 65% | `/api/query`, job status, SSE, OpenAlex pipeline output | cancellation, retry policy, partial artifact recovery UI |
| F05 | Network construction | `[x]` | 75% | DC/BC/CC builders, edge combination, filters, OpenAlex citation edges | first-class entity networks and richer evidence artifacts |
| F06 | Matrix builder | `[~]` | 43% | sparse matrix internals, co-occurrence helpers, artifact feature detection, matrix artifact design, general matrix writer/validator, term co-occurrence wrapper | explicit matrix-builder mode and exports |
| F07 | Clustering and hierarchy | `[x]` | 85% | Rust CPM/Leiden path, hierarchy, landscape, membership artifacts | app-level parameter workflow and expensive-run guardrails |
| F08 | Keyword extraction, labels, cleaning | `[~]` | 80% | pipeline, quality filters, abbreviation handling, term network, scaling docs, keyword rule artifacts | editable replay workflow, imported thesaurus adapters, and full large-run benchmark |
| F09 | Atlas map, evidence, cluster reading | `[~]` | 91% | atlas payload builder, neighbors, representative works, web endpoints, evidence inspector model, review checklist, review packet, filterable review queue, render payload adapter, deck.gl prototype, layer controls, render/perf/interaction/inspector smoke gates | complete evidence review workflow |
| F10 | Term network and co-occurrence visualization | `[~]` | 91% | term network module, endpoint, stable co-occurrence table/map artifacts, manifest-backed co-occurrence table export, VOSviewer-style term co-occurrence export, Term view export links, QA readouts, threshold presets | map polish and layout UX |
| F11 | Temporal and evolution | `[~]` | 64% | temporal keyword utilities, burst/trend helpers, feature detection, temporal/evolution artifact designs, temporal writer/validator, evolution writer/validator, synthetic evolution smoke, artifact-backed web Evolution lens | richer time-slice matching and full evolution map layout |
| F12 | Evidence-backed narratives | `[~]` | 24% | narrative feature detection, target definition, evidence-reference artifact design | writer, validator, review UI, and optional generator |
| F13 | Validation and QA | `[x]` | 83% | artifact contract, result validation, quality gate, keyword artifact checks, matrix/temporal/evolution artifact validators | strict checks for narrative/export features |
| F14 | Report, export, interoperability | `[~]` | 99% | HTML reports/viewer, dashboard export, GEXF, GraphML, VOSviewer-style map/network, VOSviewer thesaurus/rule-set export, VOSviewer-style term co-occurrence export, VOSviewer web bundle download, co-occurrence table export, CLI rule-export, static data, export manifest design, writer/validator, QA sidecars, result-manifest export inventories, normalized export selection/subset summaries, web subset-filtered graph exports | matrix builder export adapter |
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
  temporal charts, GEXF, GraphML, VOSviewer-style map/network, and
  manifest-backed export inventory surfaces exist.
- Tests cover the main implemented surfaces, including artifacts, web app, demo
  gate, landscape, term networks, co-occurrence, temporal logic, adapters,
  OpenAlex edges, and network maps.

## Feature Detail

### F01. Workspace And Project Management

Status: `[~]` Partial. Rough completeness: 53%.

- `[x]` Local result roots can be discovered and opened from the web app.
- `[x]` Query jobs have persisted status and output directories.
- `[x]` Demo presets point users to prepared local result artifacts.
- `[x]` Workspace/project/run manifest contract is defined in
  `docs/developer/workspace_manifest_design.md`.
- `[~]` Result-root validation can infer available features from files.
- `[x]` Workspace writer and validator are implemented.
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

Status: `[~]` Partial. Rough completeness: 43%.

- `[x]` Sparse matrix internals are used by BC/CC and keyword extraction.
- `[x]` Co-occurrence table construction exists for keyword terms.
- `[x]` A general matrix artifact contract is defined in
  `matrix_artifact_design.md`.
- `[x]` General sparse-triplet matrix writer, validator, QA sidecar, and
  term-co-occurrence wrapper exist.
- `[x]` Artifact feature inference distinguishes stable general matrix manifests
  from co-occurrence-only term-network evidence.
- `[ ]` There is no explicit matrix-builder command, app mode, or export flow.
- `[ ]` Matrix thresholding, projection choices, and compare-across-cleaning-rule
  metadata are not yet exposed as replayable user workflows.

Review: implementation primitives and the artifact contract now exist, but the
Matrix Builder should still be marked as not product-complete until a user-facing
builder mode, export path, and comparison workflow are implemented.

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

Status: `[~]` Partial. Rough completeness: 80%.

- `[x]` Keyword extraction pipeline exists.
- `[x]` Quality annotation, artifact filtering, representative scoring,
  abbreviation extraction, and vocabulary cleansing exist.
- `[x]` Term network and co-occurrence helpers exist.
- `[x]` Scaling and sharding design has been documented.
- `[x]` Replayable keyword rule artifact contract is defined in
  `docs/developer/keyword_rule_artifact_design.md`.
- `[x]` Keyword rule writer, validator, before/after impact summaries,
  result-manifest exposure, and keyword-pipeline auto-emission exist.
- `[x]` Rule applications preserve flag-level review evidence, cluster-sharded
  provenance remains unambiguous across split roots, and CLI root inference is
  explicit.
- `[~]` Label merge and LLM labeling endpoints exist, but the full review loop is
  not yet a polished app workflow.
- `[ ]` Editable cleaning workflow, workspace-level reusable rule registry,
  imported thesaurus adapters, and visual rule-impact diffs are not complete.

Review: the algorithmic base is strong enough to keep improving, but the analyst
control surface is still incomplete.

### F09. Atlas Map, Evidence, And Cluster Reading

Status: `[~]` Partial. Rough completeness: 91%.

- `[x]` Atlas payload can be built from report and membership artifacts.
- `[x]` Hierarchy lineage, doc counts, labels, representative works, and neighbor
  evidence can be attached.
- `[x]` Cluster detail and atlas-related web endpoints exist.
- `[x]` `/api/jobs/{job_id}/atlas-render` exposes renderer-oriented layer rows
  for deck.gl-style Atlas map engines.
- `[x]` A validated-payload evidence inspector contract is defined in
  `atlas_evidence_inspector_design.md`.
- `[x]` The web inspector builds a client-side
  `sciscape_inspector_evidence_view_v1` model and renders section states for
  identity, meaning, relations, hierarchy, works, and QA.
- `[x]` The P1 Atlas smoke gate asserts stable co-occurrence/evidence states and
  payload-backed neighbor rows with aggregate relation fields, shared-term
  fields, and sampled edge evidence.
- `[x]` `scripts/sciscape_quality_gate.py --atlas-inspector-smoke` optionally
  opens the static app in headless Chrome and verifies inspector node selection,
  sample-backed neighbor evidence, aggregate-only neighbor fallback, and
  representative-work rendering.
- `[x]` The inspector now includes a compact review checklist that summarizes
  term, work, relation, and QA readiness for cluster interpretation.
- `[x]` The inspector now includes a selected-cluster review packet that
  combines state, top terms, representative works, and active neighbor evidence
  before narrative generation.
- `[x]` The Evidence view now includes a filterable current-level review queue
  that counts ready/review/blocked clusters, links directly to review targets,
  offers next-target navigation, and syncs `atlas_review` URL state.
- `[x]` A deck.gl-oriented render-engine contract is defined in
  `atlas_render_engine_design.md`.
- `[x]` The current static web app has a guarded deck.gl Atlas Map prototype
  using `ScatterplotLayer`, `LineLayer`, `TextLayer`, and `OrthographicView`.
- `[x]` The deck.gl prototype exposes layer visibility controls, edge-weight
  thresholding, label-density control, URL-state persistence, and selected-node
  view centering.
- `[~]` Evidence is organized into a section-state inspector workflow, but fuller
  review affordances are still pending.
- `[x]` The P1 Atlas smoke validates the `/atlas-render` contract.
- `[x]` `scripts/sciscape_quality_gate.py --atlas-visual-smoke` renders a tiny
  deck.gl map in headless Chrome and checks for nonblank pixels when Chrome is
  available.
- `[x]` `scripts/sciscape_quality_gate.py --atlas-render-perf-smoke` validates
  a deterministic 100-node/500-edge render payload contract without requiring a
  browser.
- `[x]` `scripts/sciscape_quality_gate.py --atlas-render-scale-smoke` validates
  a deterministic 5k-node/25k-edge render payload contract without requiring a
  browser.
- `[x]` `scripts/sciscape_quality_gate.py --atlas-interaction-smoke` optionally
  validates 5k-node/25k-edge deck.gl browser rendering, selected-node camera
  update, center hit-test, and nonblank screenshot when Chrome is available.
- `[~]` Small-demo interaction smoke exists, but analyst-scale browser gates are
  not implemented yet.
- `[ ]` The UI/UX rewrite is still pending and should not be considered complete.

Review: the data contract is ahead of the interface. The new render payload
gives the deck.gl prototype a stable entry point, and the static viewer now has
a first GPU map surface with smoke coverage plus CI-scale and small-demo-scale
render payload performance gates. Optional browser smokes cover small-demo map
interaction, the inspector review checklist, the selected-cluster review packet,
and filterable current-level review queue behavior. Analyst-scale browser gates
and a fuller evidence review
workflow are still required before treating it as a finished Atlas App.

### F10. Term Network And Co-Occurrence Visualization

Status: `[~]` Partial. Rough completeness: 91%.

- `[x]` Term network construction exists.
- `[x]` Co-occurrence collection exists.
- `[x]` `/api/jobs/{job_id}/term-network` endpoint exists.
- `[x]` Term/co-occurrence evidence can be surfaced from keyword artifacts.
- `[x]` `term_cooccurrence.parquet` and `term_cooccurrence_map.json` are written
  by the landscape pipeline and validated as first-class artifacts.
- `[x]` Stable co-occurrence artifacts can be wrapped as a manifest-backed TSV
  table plus paired map JSON export, and local web result loading exposes that
  export through `result_manifest.exports`.
- `[x]` Term Co-occurrence view shows manifest-backed table, map, and export
  manifest download links when the co-occurrence export is present.
- `[x]` Term Co-occurrence view shows threshold/readability QA readouts for
  visible edges, hidden edges, visible labels, max edge weight, and reset layout.
- `[x]` Term Co-occurrence view includes data-driven density presets for all,
  core, and backbone views of the same term network.
- `[x]` Stable co-occurrence artifacts can be exported as VOSviewer-style term
  map/network files and included in the web VOSviewer bundle.
- `[~]` Visualization controls for thresholding, layout, clustering, and export
  are not yet complete.

Review: the previously broken co-occurrence path has enough implementation
surface to stabilize. The table/map artifact contract, table export, Term view
download affordance, threshold/readability readouts, and density presets now
exist; the remaining gap is map polish and more complete layout UX.

### F11. Temporal And Evolution

Status: `[~]` Partial. Rough completeness: 64%.

- `[x]` Temporal keyword utilities and visualization helpers exist.
- `[x]` Burst and trend helpers exist.
- `[x]` Temporal tracking endpoints exist in the web app.
- `[x]` Temporal trend artifact contract is defined in
  `docs/developer/temporal_artifact_design.md`.
- `[x]` Cluster evolution artifact contract and synthetic smoke example are
  defined in `docs/developer/evolution_artifact_design.md`.
- `[x]` Temporal trend writer and validator can emit artifact-backed temporal
  feature state.
- `[x]` Cluster evolution writer, validator, lineage transition artifacts, and
  synthetic split/merge/emergence/decline/continuation/ambiguous smoke are
  implemented.
- `[x]` Web app exposes an artifact-backed Evolution lens and bounded
  `/api/jobs/{job_id}/evolution` payload for slices, states, transitions,
  lineages, and events.
- `[~]` Membership-projection evolution writer intentionally emits only events
  that are supported by static membership continuity; split/merge are covered
  by the validator smoke and need richer time-slice matching for real data.
- `[ ]` Full evolution map layout is not product-ready.

Review: temporal/evolution artifacts are now inspectable and independently
validated, and the web app can inspect evolution rows. The promised cluster
evolution map remains gated on richer visual layout and matching beyond static
membership projection.

### F12. Evidence-Backed Narratives

Status: `[~]` Partial. Rough completeness: 24%.

- `[~]` Narrative is named in the feature definition and artifact feature
  inference can detect narrative-like payloads.
- `[x]` Narrative evidence-reference artifact contract is defined in
  `docs/developer/narrative_artifact_design.md`.
- `[ ]` There is no stable narrative evidence-reference writer.
- `[ ]` There is no narrative artifact validator or QA gate.
- `[ ]` There is no stable narrative generation pipeline.
- `[ ]` There is no final app surface for cluster narrative review.

Review: the evidence-reference contract now exists, but narrative should remain
hidden or beta until writer, validator, QA gate, and review UI exist.

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

Status: `[~]` Partial. Rough completeness: 99%.

- `[x]` Dashboard, report, and viewer export helpers exist.
- `[x]` CLI export supports GEXF, GraphML, and VOSviewer-style map/network
  files.
- `[x]` Static report data can be used by the web viewer.
- `[x]` Export manifest contract is defined in
  `docs/developer/export_manifest_design.md`.
- `[x]` Export manifest writer, validator, file/input/transform sidecars, and QA
  sidecar exist.
- `[x]` Result-root validation can expose manifest-backed `export=stable` while
  keeping legacy report/viewer exports at beta.
- `[x]` `result_manifest.exports` exposes manifest-backed primary export paths,
  export manifest refs, and compact output file inventories.
- `[x]` Web Download tab can render manifest-backed export manifests and export
  files from `result_manifest.exports`, including compact selection-summary
  chips for manifest-backed export rows.
- `[x]` Dashboard, report, static viewer, CLI GEXF/GraphML, and web network
  export paths write export manifests automatically.
- `[~]` Artifact contracts can summarize export availability.
- `[x]` VOSviewer-style map/network export writes tab-delimited map/network
  files and a manifest-backed export artifact.
- `[x]` VOSviewer keyword-rule thesaurus export writes `label` / `replace by`
  rows, a companion SciScape rule-set TSV, and a manifest-backed export
  artifact.
- `[x]` CLI `rule-export` supports VOSviewer thesaurus/rule-set export from a
  keyword rule artifact.
- `[x]` Web Download tab can build and download one `vosviewer_bundle.zip`
  from manifest-backed VOSviewer exports.
- `[x]` Graph, VOSviewer, dashboard, report, static viewer, CLI visualize/viewer,
  `run_landscape` report, GUI viewer, quality-gate smoke, rule-export, and web
  bundle export manifests preserve normalized `sciscape_export_selection_v1`
  view/filter/focus/layer metadata, and `result_manifest.exports` exposes
  compact `selection_summary` rows including focus and subset keys.
- `[x]` Web network export captures Atlas level, selected cluster, query,
  lens/view/focus mode, review filter, deck layers, edge threshold, label limit,
  and neighbor focus from the browser request into the export selection.
- `[x]` Web network export captures visible/focused subset mode, count, bounded
  cluster UID sample, truncation state, and pinned cluster IDs.
- `[x]` Web network export applies selected Atlas cluster subsets to GraphML and
  GEXF output files and records an `apply_selected_subset` transform.
- `[x]` Term co-occurrence table/map export writes a TSV table plus paired map
  JSON and records a manifest-backed export artifact.
- `[x]` VOSviewer-style term co-occurrence export writes term map/network files,
  records a manifest-backed export artifact, and is included in VOSviewer web
  bundles.
- `[ ]` Matrix-builder export mode remains future work.

Review: good enough for graph/report exports, and the manifest contract plus
writer/validator plus first command adapters now exist, but interoperability
should not be called complete until all user-facing view/filter state and
remaining matrix-builder interoperability is covered.

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
  evolution map, evidence-backed narratives, and institutional analytics.

## Next Implementation Targets

1. Harden inspector-driven review affordances and Cleaning-mode rule review.
2. Implement richer time-slice matching for evolution beyond static membership
   projection, then promote the Evolution lens into a true map layout.
3. Implement the narrative evidence-reference writer, validator, and unsupported
   claim gate from `narrative_artifact_design.md`.
4. Add a matrix-builder export adapter when its source contract is stable.
