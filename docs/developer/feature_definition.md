# SciScape Feature Definition

Status: product contract draft
Date: 2026-05-30

This document defines SciScape's intended product capabilities before the web
UI is redesigned. It is not a screen-level UI specification. It is the contract
for app modes, analysis lenses, inputs, outputs, and validation rules that the
UI, CLI, and static viewer should eventually share.

For the higher-level project identity, governance levels, project invariants,
and north-star definition, see `branding_positioning.md`. This feature
definition should remain subordinate to that document: a feature is in scope
only if it strengthens the validated science-landscape workflow without
violating the stated invariants.

## Positioning

SciScape should be framed as a local-first workbench for validated science
landscapes:

```text
query or files
-> bibliographic record preparation
-> network and matrix construction
-> clustering and hierarchy
-> keyword cleaning and labeling
-> cluster maps, hierarchy maps, evolution maps, and evidence review
-> cluster narratives
-> export, report, and validation artifacts
```

The primary differentiation should not be a single map view. It should be a
reproducible, QA-visible workflow that combines:

- VOSviewer-style science mapping and term co-occurrence visualization.
- Biblioshiny-style full-cycle bibliometric workflow.
- CiteSpace-style temporal and structural knowledge-domain review.
- KnowledgeMatrix and VantagePoint-style tech mining, matrix building, cleaning,
  thesaurus, and analyst refinement.
- SciScape's own Rust CPM/Leiden clustering, keyword QA, artifact filtering,
  and local/static deployment path.

## Benchmark Baseline

Use this table as the functional baseline. SciScape does not need to match every
feature in v1, but each feature should have an explicit decision: support,
defer, or exclude.

| Tool family | Reference tools | Baseline capability | SciScape response |
| --- | --- | --- | --- |
| Science mapping | VOSviewer | co-authorship, citation, bibliographic coupling, co-citation, term co-occurrence, density and overlay maps, thesaurus-based cleaning | Support core network and term-map equivalents, with explicit artifact contracts |
| Bibliometric workflow | bibliometrix / Biblioshiny | import, filtering, source/author/document metrics, conceptual/intellectual/social structures | Support a smaller workflow first: ingest, network, landscape, keywords, report |
| Temporal knowledge mapping | CiteSpace, SciMAT | timeline, burst, thematic evolution, strategic diagram | Defer advanced burst/thematic evolution, but reserve temporal lens contracts |
| Tech mining | KnowledgeMatrix, VantagePoint | user-defined lists, occurrence/co-occurrence/proximity matrices, clustering, visualization, preprocessing, string/thesaurus editors | Make Matrix Builder and Cleaning modes first-class app modes |
| Institutional analytics | SciVal, InCites, Dimensions Analytics | institutional benchmarking, normalized impact, portfolio analytics | Defer as out of scope unless an institutional data contract is added |
| General graph exploration | Gephi, Pajek, Cytoscape | flexible layout, filtering, graph export | Export GEXF/GraphML and avoid duplicating full graph-editor behavior |
| Literature exploration | Connected Papers, ResearchRabbit, Litmaps, Open Knowledge Maps | seed/query-based exploratory maps | Support query-to-map and curated static result bundles |

References to verify when expanding scope:

- VOSviewer features: https://www.vosviewer.com/features/highlights
- Biblioshiny features: https://bibliometrix.org/home/index.php/layout/biblioshiny
- CiteSpace overview: https://cluster.cis.drexel.edu/~cchen/citespace/
- SciMAT overview: https://sci2s.ugr.es/scimat/
- KnowledgeMatrix paper: https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART001223521
- VantagePoint overview: https://www.thevantagepoint.com/

## Feature Support Matrix

Use this matrix to prevent benchmark drift. Each capability must have one
product decision before it is exposed in the app.

Decision labels:

- `support-v1`: required for the first coherent app.
- `support-v1.5`: required for the KnowledgeMatrix/VantagePoint-facing
  refinement pass.
- `support-v2`: supported later after the core contracts stabilize.
- `defer`: promising, but no product promise yet.
- `exclude`: intentionally outside SciScape's scope.

| Capability | Benchmark pressure | Decision | Contract note |
| --- | --- | --- | --- |
| Curated demo result | VOSviewer, Open Knowledge Maps | `support-v1` | must load without knowing local folders |
| Hosted `data.json` static viewer | VOSviewer sharing, Atlas datapacks | `support-v1` | must work on GitHub Pages or any static host |
| OpenAlex query-to-result | literature exploration tools | `support-v1` | small jobs only, with job state and saved result root |
| Existing result-root loading | Atlas, local analysis workflows | `support-v1` | must validate before enabling lenses |
| WoS, Scopus, OpenAlex, BibTeX import | Biblioshiny, KnowledgeMatrix | `support-v1` for CLI, `support-v2` for app | app should not claim unsupported import adapters |
| PubMed, Dimensions, Lens, Crossref import | Biblioshiny, commercial platforms | `defer` | mention only as future adapter targets |
| Direct citation, bibliographic coupling, co-citation | VOSviewer, CiteSpace | `support-v1` | edge family and weight semantics must be recorded |
| Semantic or embedding KNN edges | SciScape differentiation | `support-v1` when embeddings exist | feature flag must expose missing embedding data |
| Co-authorship, institution, source, country networks | VOSviewer, Biblioshiny | `support-v1.5` | depends on normalized entity fields |
| Term co-occurrence graph/table/map | VOSviewer, KnowledgeMatrix | `support-v1` | must have edge counts and artifact validation |
| General matrix builder | KnowledgeMatrix, VantagePoint | `support-v1.5` | document-term, cluster-term, term-term first |
| Proximity/similarity matrices beyond co-occurrence | VantagePoint | `support-v1.5` | must distinguish occurrence, proximity, similarity |
| CPM/Leiden clustering | SciScape core | `support-v1` | backend, parameters, seed, and version must be preserved |
| User-configurable clustering app mode | VOSviewer/Gephi-like exploration | `support-v2` | avoid exposing research defaults prematurely |
| Keyword extraction and representative labels | all science mapping tools | `support-v1` | counts, score, n-gram, rank, and flags required |
| Artifact filtering and keyword QA | SciScape differentiation | `support-v1` | blocking artifacts must prevent release-quality status |
| Read-only cleaning audit | KnowledgeMatrix, VantagePoint | `support-v1` | show raw, normalized, display, family, flags |
| Editable thesaurus/alias/stop-term rules | VOSviewer, VantagePoint | `support-v1.5` | rule log must be replayable |
| Acronym expansion from local text | tech mining workflows | `support-v1.5` | preserve evidence source and ambiguity state |
| Temporal trend summary | Biblioshiny, CiteSpace | `support-v1` if `pubyear` exists | simple yearly counts before advanced evolution |
| Cluster evolution map | Science Atlas, CiteSpace/SciMAT | `support-v1.5` when evolution artifacts exist | yearly activity, phase topics, topic/child trajectories, representative works |
| Evidence-backed cluster narratives | Science Atlas, analyst review workflows | `support-v1.5` | must cite terms, representatives, lineage, neighbors, evolution, and QA caveats |
| Burst detection and thematic evolution | CiteSpace, SciMAT | `support-v2` | do not claim without separate method contract |
| Institutional benchmarking and normalized impact | SciVal, InCites | `defer` | requires controlled metric and data-license contracts |
| Full graph editor behavior | Gephi, Cytoscape | `exclude` | export to graph tools instead |
| LLM canonicalization | modern workflow expectation | `defer` | optional, auditable, never required for deterministic output |

## Product Principles

1. Start from the user's analysis state, not from an implementation folder.

   The app should make clear whether the user is opening a static result,
   running a live query, validating a bundle, cleaning terms, or building a
   matrix.

2. Separate app mode from analysis lens.

   An app mode decides how data enters and whether the pipeline can run. An
   analysis lens decides how an already-loaded result is inspected.

3. Advertise features from artifacts.

   The UI should not assume that keywords, term networks, temporal data,
   hierarchy, or representative documents exist. These should be inferred from
   files and schema columns.

4. Make cleaning and matrices product surfaces.

   Keyword cleaning, thesaurus rules, aliases, acronym expansion, and
   co-occurrence matrices are not internal utilities. They are core to
   analyst-grade SciSci work.

5. Keep validation visible.

   Every complete result should expose manifest, schema, QA, and warning
   information in a machine-readable form.

## App Modes

App modes are entry states. They determine what input is accepted, whether
SciScape runs computation, and what result contract must be produced.

| Mode ID | User goal | Primary inputs | Runs pipeline | Required output state | Priority |
| --- | --- | --- | --- | --- | --- |
| `demo` | Open a curated example immediately | bundled manifest or hosted `data.json` | no | loaded result, read-only provenance | v1 |
| `static_viewer` | View a GitHub Pages or shared static result | hosted `data.json` or report bundle | no | loaded result, read-only provenance | v1 |
| `local_result` | Inspect an existing SciScape output folder | result root, `data.json`, or report directory | no | loaded result plus validation report | v1 |
| `live_query` | Paste a query and build a small analysis | OpenAlex query and limits | yes | job state, result root, report bundle | v1 |
| `file_pipeline` | Run from user-provided bibliographic files | abstracts, references, edges, metadata | yes | result root with manifest | v2 |
| `matrix_builder` | Build occurrence, co-occurrence, proximity, and 1-mode/2-mode tables | records, term lists, fields, cluster labels | partial | matrix artifacts and matrix QA | v1.5 |
| `cleaning` | Review terms, aliases, stop terms, acronyms, and artifacts | keyword table, abstracts, thesaurus rules | partial | cleaned keyword table, rule log, QA | v1.5 |
| `clustering` | Configure and run CPM/Leiden landscape building | edge table, weights, levels, clustering config | yes | membership, hierarchy, clustering QA | v2 |
| `report` | Produce shareable report and exports | loaded result root | partial | HTML, JSON, graph exports, QA | v1 |
| `validation` | Check whether a result is safe to inspect or publish | result root or `data.json` | no | artifact contract report | v1 |

Mode rules:

- `demo`, `static_viewer`, and `local_result` must never pretend that a pipeline
  has run.
- `live_query`, `file_pipeline`, and `clustering` must expose job state and
  partial failure state.
- `matrix_builder` and `cleaning` may operate on partial results and should
  preserve rule/provenance logs.
- `validation` should be callable from CLI, web, and release gate.

### Mode Acceptance Criteria

Each mode must define success, partial success, and blocking conditions. The app
can then show a useful state instead of failing silently or enabling empty
lenses.

| Mode ID | Success criteria | Partial state | Blocking state |
| --- | --- | --- | --- |
| `demo` | curated result loads, feature block is present, at least one cluster and one keyword surface are non-empty | result loads but optional lenses are disabled | bundled result is missing, malformed, or fails validation |
| `static_viewer` | hosted `data.json` parses, schema/version is known or safely inferred, enabled lenses have data | `data.json` loads but advanced artifacts such as term network are absent | CORS/load failure, malformed JSON, empty result |
| `local_result` | result root or report directory validates and maps to feature flags | some files exist but count reconciliation or optional artifacts are missing | required files for all claimed features are missing or inconsistent |
| `live_query` | query job completes, result root is saved, report data and validation report exist | fetch or downstream step succeeds partially and exposes recoverable outputs | source fetch fails, no usable records, pipeline error without saved state |
| `file_pipeline` | records and edge/matrix inputs produce a manifest-backed result root | records parse but some network or keyword outputs are unavailable | input schema is unknown, IDs cannot be reconciled, or required files are unreadable |
| `matrix_builder` | selected fields produce matrix artifact with row/column metadata and QA | matrix exists but selected labels or weights are incomplete | no valid fields, empty matrix, or unsupported weighting request |
| `cleaning` | cleaned keyword table, rule log, and QA summary are produced | audit exists but no rule changes are applied | rule replay fails, cleaned output loses required keyword columns |
| `clustering` | membership, cluster sizes, backend metadata, and QA are produced | clustering completes but hierarchy, keywords, or report are missing | edge table invalid, backend fails, or membership cannot join to records |
| `report` | HTML/JSON/export artifacts are written with version and feature blocks | some export formats unavailable due to missing inputs | report data cannot be parsed or output directory cannot be written |
| `validation` | contract report classifies features, warnings, and blocking issues | warnings exist but at least one lens can load | no supported input artifact can be validated |

Minimum mode payload fields:

- `mode`
- `result_state`
- `result_root` or `source_uri`
- `features`
- `warnings`
- `versions`
- `created_at_utc`
- `source_summary`

## Analysis Lenses

Analysis lenses are inspectable surfaces over a loaded result. They should be
enabled or disabled by feature availability, not by hard-coded UI assumptions.

| Lens ID | Purpose | Required artifacts | v1 behavior |
| --- | --- | --- | --- |
| `overview` | Dataset size, year span, source counts, warning summary | abstracts or report data | show counts and missing-feature warnings |
| `cluster_map` | Explore clusters and hierarchy | membership, optional edges, report data | show cluster sizes, labels, parent/child levels when present |
| `keyword` | Review representative terms | keywords table | show common vs cluster-specific terms, n-gram mix, counts, flags |
| `term_network` | Inspect keyword co-occurrence | term network or co-occurrence matrix | show graph/table/map with counts and evidence links when present |
| `matrix` | Inspect occurrence, co-occurrence, proximity, 1-mode, and 2-mode matrices | matrix artifacts | show rows, columns, weighting, sparsity, top pairs |
| `evidence` | Inspect representative works and text evidence | abstracts, membership, representative docs | show titles, abstracts/snippets, term evidence |
| `temporal` | Inspect change over publication years or periods | pubyear, temporal summaries | show trend summaries; advanced evolution deferred |
| `evolution` | Inspect cluster lifecycle and composition shifts | evolution artifact, temporal summaries, representative works | show activity timeline, evolution map, topic/child trajectories, and milestones when present |
| `narrative` | Read an evidence-backed interpretation of a cluster | labels, terms, lineage, representatives, neighbors, optional evolution | show cluster narrative with citations to available evidence and QA caveats |
| `quality` | Inspect data contamination, duplicates, artifacts, imbalance, schema warnings | QA artifacts | show blocking and non-blocking issues |
| `export` | Prepare downstream files | report data, membership, edges, keywords | export JSON, CSV/TSV, GEXF/GraphML, HTML |

### Lens Artifact Requirements

This table is the bridge from product lens to result-root contract. The
dedicated artifact contract can later split these into stricter schema files.

| Lens ID | Required artifacts | Minimum fields | Empty behavior |
| --- | --- | --- | --- |
| `overview` | `report/data.json` or abstracts table | record count, optional year span, source summary | show validation summary only |
| `cluster_map` | membership table or report cluster payload | `uid`, one cluster column or cluster payload with IDs and sizes | disable lens with `missing_membership` |
| `keyword` | keyword table or report keyword payload | `cluster_id`, `term`, rank or score, frequency/count when available | disable lens with `missing_keywords` |
| `term_network` | term co-occurrence or term-network payload | source term, target term, count or weight | disable lens with `missing_term_network` |
| `matrix` | matrix artifact plus row/column labels | matrix type, row IDs, column IDs, values, weighting | disable lens with `missing_matrix` |
| `evidence` | abstracts plus membership, or representative docs | `uid`, `title`, optional `abstract`, joined cluster ID | show cluster-level evidence unavailable warning |
| `temporal` | abstracts or report payload with year data | `pubyear`, counts by year or period | disable lens with `missing_temporal_data` |
| `evolution` | cluster evolution payload | `cluster_id`, year counts, lifecycle or trajectory fields | disable lens with `missing_evolution_data` |
| `narrative` | narrative payload or enough evidence to generate a deterministic summary | cluster label, terms, lineage, representative works or evidence snippets | disable lens with `missing_narrative_evidence` |
| `quality` | QA artifact or validation output | severity, code, message, affected artifact | run lightweight validation if possible |
| `export` | loaded result data plus writable target | available artifact list and export formats | show only formats whose inputs exist |

Required count reconciliation:

- every `membership.uid` should exist in the abstract/record table when both
  are present.
- every keyword `cluster_id` should exist in membership or report cluster
  payload when both are present.
- term-network terms should resolve to keyword terms or carry an explicit
  `external_term=true` marker.
- report cluster counts should match membership counts or carry a documented
  sampling/filtering note.

## Capability Catalog

The following capabilities define what SciScape should support. They are grouped
by user-facing workflow rather than package directory.

### 1. Ingest And Normalize

Goal: load research records into a standard table.

Inputs:

- OpenAlex query results.
- Web of Science, Scopus, PubMed, Dimensions, Lens, Crossref, or CSV-like
  records when adapters exist.
- Local BibTeX or JSON/JSONL records.

Minimum standard columns:

- `uid`
- `title`
- `abstract`
- `pubyear`
- optional author, venue, institution, reference, citation, DOI, and URL fields

Required behaviors:

- Preserve source identifiers.
- Record source and conversion metadata.
- Flag blank title/abstract rows.
- Apply safe text normalization such as HTML unescape/tag removal and metadata
  artifact detection before keyword extraction.

### 2. Network And Matrix Construction

Goal: build graph and matrix representations that support both clustering and
tech-mining analysis.

Network families:

- direct citation
- bibliographic coupling
- co-citation
- semantic or embedding KNN
- keyword co-occurrence
- author, institution, source, and country co-occurrence where fields exist

Matrix families:

- document-term
- cluster-term
- term-term co-occurrence
- author-keyword
- institution-keyword
- source-keyword
- paper-cluster
- cluster-cluster proximity

Required behaviors:

- Expose weighting scheme and normalization.
- Distinguish occurrence, co-occurrence, proximity, and similarity.
- Preserve enough row/column labels for review.
- Report matrix shape, sparsity, top values, and dropped rows.

### 3. Clustering And Landscape

Goal: generate stable, interpretable research-topic landscapes.

Supported surfaces:

- Rust CPM/Leiden backend.
- Python compatibility surfaces under `sciscape.clustering`.
- optional hierarchy and postprocess outputs.

Required behaviors:

- Preserve clustering parameters and backend version.
- Report cluster sizes and singleton/small-cluster behavior.
- Distinguish production clustering from research-only Dongdaemun diagnostics.
- Keep membership-level outputs available for validation and re-labeling.

### 4. Keyword Extraction And Cleaning

Goal: produce interpretable labels that work across user domains.

Required surfaces:

- raw candidate terms
- normalized terms
- display labels
- n-gram type
- score and frequency
- cluster-specificity signal
- common-vs-cluster-specific flag
- alias or keyword-family group
- acronym expansion evidence when available
- artifact and metadata flags

Required behaviors:

- Keep unigram, bigram, and trigram behavior explicit.
- Prefer interpretable bigrams/trigrams when they are strong, but keep useful
  unigrams instead of suppressing them globally.
- Preserve lower-level derivative terms under a representative family when
  string or semantic similarity supports grouping.
- Extract acronym definitions from local text when patterns such as
  `long form (ACR)` or `ACR (long form)` appear.
- Treat HTML, LaTeX preamble fragments, publisher metadata, and boilerplate as
  blocking artifacts in release-quality results.

### 5. Visualization And Exploration

Goal: make results inspectable without requiring the user to know file paths.

Required surfaces:

- static viewer for hosted `data.json`
- local web app for query-to-result and result-folder inspection
- report/dashboard export for local sharing
- graph export for external tools

Required behaviors:

- Clearly show which mode is active.
- Disable unavailable lenses with a visible reason.
- Show result root, manifest, and source query when available.
- Avoid requiring users to know internal folder structure before loading a
  curated demo or recent result.

### 6. Report And Export

Goal: preserve outputs in forms useful for review, publication, and downstream
tools.

Outputs:

- `report/data.json`
- HTML report or dashboard
- `keywords.parquet` or CSV/TSV equivalent
- `membership.parquet`
- graph export such as GEXF or GraphML
- optional matrix artifacts
- QA artifacts

Required behaviors:

- Embed or sidecar a version block.
- Include feature availability.
- Include warnings that explain missing or partial outputs.
- Keep static viewer bundles small enough to host directly.

### 7. Validation And QA

Goal: prevent contaminated or incomplete outputs from becoming demos or release
artifacts.

Required checks:

- required files exist for the claimed mode
- required columns exist
- counts reconcile between abstracts, membership, keywords, and report data
- keyword artifacts are below release thresholds
- term network/co-occurrence payloads are non-empty when advertised
- feature flags match actual artifacts
- warnings distinguish blocking errors from informational gaps

## Feature Availability Contract

Every loaded result should be reducible to a compact feature block. This block
should be generated by validation, not hand-written in the UI.

```json
{
  "mode": "local_result",
  "result_state": "loaded",
  "features": {
    "overview": true,
    "cluster_map": true,
    "keyword": true,
    "term_network": true,
    "matrix": false,
    "evidence": true,
    "temporal": false,
    "evolution": false,
    "narrative": false,
    "quality": true,
    "export": true
  },
  "warnings": [
    {
      "code": "missing_temporal_summary",
      "severity": "info",
      "message": "Publication years exist, but no temporal summary artifact was found."
    }
  ]
}
```

Recommended feature inference:

| Feature | Enable when |
| --- | --- |
| `overview` | `data.json` or abstracts table has non-empty record counts |
| `cluster_map` | membership or report cluster payload exists |
| `keyword` | keyword table or report keyword payload exists |
| `term_network` | co-occurrence or term-network payload exists and has at least one edge |
| `matrix` | matrix artifact exists with row/column metadata |
| `evidence` | abstracts and membership can be joined, or representative docs exist |
| `temporal` | `pubyear` exists and temporal summaries or yearly grouping are available |
| `evolution` | cluster evolution artifact exists and has non-empty yearly or trajectory data |
| `narrative` | narrative artifact exists, or labels/terms/evidence meet the deterministic narrative contract |
| `quality` | QA report exists or validation can run |
| `export` | loaded result has enough data to write at least one supported export |

## Result States

Use these states consistently across CLI, web jobs, and viewer code.

| State | Meaning |
| --- | --- |
| `empty` | no data selected |
| `loading` | input is being parsed |
| `validating` | artifact contract is being checked |
| `running` | a pipeline job is active |
| `partial` | some outputs exist but required mode outputs are missing |
| `loaded` | result is usable for at least one analysis lens |
| `blocked` | validation found a blocking issue |
| `failed` | computation or load failed |

## Product Constraints

These constraints prevent the app from claiming capabilities that are only safe
in a CLI, batch, or precomputed setting.

### Data Size Classes

Size classes are product guidance, not performance guarantees. Claims must be
verified with release-gate artifacts before they become public limits.

| Class | Approximate size | Recommended surface | Notes |
| --- | --- | --- | --- |
| `demo_small` | up to about 1,500 records | web app, static viewer, docs demos | target for query-to-result examples |
| `analyst_medium` | about 1,500 to 50,000 records | CLI pipeline plus local result viewer | app may inspect precomputed outputs |
| `batch_large` | more than 50,000 records | CLI/batch only unless precomputed | do not run interactively in the browser |
| `atlas_scale` | hundreds of thousands or more | separate datapack-style architecture | outside normal SciScape app v1 |

### Web And Static Limits

- Static viewer mode can only inspect artifacts that are already hosted or
  bundled. It cannot fetch OpenAlex directly unless a separate backend exists.
- GitHub Pages-style deployments should assume read-only, precomputed data.
- Browser upload can inspect files, but long-running clustering and network
  building should go through the local web backend or CLI.
- The app should show missing capability reasons instead of hiding disabled
  lenses.

### Reproducibility Fields

Every complete result should preserve enough information to repeat or audit the
run:

- source type and source query or file list
- source fetch timestamp and record count
- SciScape package version
- schema/result manifest version
- clustering backend and backend version
- clustering parameters and random seed when applicable
- keyword extraction configuration
- cleaning rules and rule-log hash when applicable
- generated artifact list and checksums when available

### Privacy And Locality

SciScape should be safe for local bibliometric and tech-mining work:

- Local file and local result modes should not require remote upload.
- Optional LLM or external API features must be explicit and disabled by
  default for private data.
- Reports should not silently embed raw abstracts when a compact public bundle
  is requested.
- Exported demos should include enough provenance to be interpretable without
  leaking private local paths.

### Release Quality Gates

No output should be promoted to a curated demo or release artifact unless:

- validation has no blocking issues
- advertised feature flags match actual artifacts
- keyword artifact checks pass release thresholds
- report/static viewer smoke tests load a non-empty result
- count reconciliation is recorded or explicitly explained
- source and version metadata are visible to users

## V1 Product Scope

The first coherent Sciscape app should support:

1. `demo` mode with a curated result.
2. `static_viewer` mode for GitHub Pages-style `data.json`.
3. `local_result` mode for existing result roots and report folders.
4. `live_query` mode for small OpenAlex examples.
5. `validation` mode as a shared contract.
6. `overview`, `cluster_map`, `keyword`, `term_network`, `quality`, and
   `export` lenses.
7. Minimum matrix support for term co-occurrence tables and maps.
8. Read-only cleaning audit for raw, normalized, display, family, and artifact
   fields.
9. Lightweight temporal trend summaries when `pubyear` is available.

## V1.5 Product Scope

Next, add the features that make SciScape closer to KnowledgeMatrix and
VantagePoint:

1. `matrix_builder` mode.
2. `cleaning` mode.
3. editable or replayable thesaurus, stop-term, alias, and acronym rules.
4. matrix QA and matrix export.
5. keyword-family hierarchy display.
6. cluster evolution maps when temporal/evolution artifacts exist.
7. evidence-backed cluster narrative view.
8. co-authorship, institution, source, and country co-occurrence where
   normalized entity fields exist.

## V2 Product Scope

Later, add heavier or more research-dependent features:

1. `file_pipeline` mode for broader bibliographic import.
2. `clustering` mode with advanced configuration.
3. temporal evolution and burst-like analysis.
4. institutional benchmarking only if normalized metric contracts are added.
5. richer graph-editor behavior only through export or a clearly scoped viewer.

## Non-Goals

- Do not copy the NanoClustering Atlas UI directly.
- Do not copy the full Atlas sidecar architecture.
- Do not turn the Sciscape web app into a general graph editor.
- Do not claim institutional benchmarking without controlled data and metric
  contracts.
- Do not hard-code field-specific labels or cluster levels into the public
  contract.
- Do not make UI affordances the source of truth for feature availability.

## Implementation Order

1. Add artifact validation and feature inference for existing result roots.
2. Embed feature/version blocks into `report/data.json`.
3. Make the current web/static viewer consume the feature block.
4. Add matrix/co-occurrence artifact contracts.
5. Add cleaning rule artifacts for aliases, stop terms, acronyms, and artifacts.
6. Redesign the UI around modes and lenses after the contract stabilizes.
