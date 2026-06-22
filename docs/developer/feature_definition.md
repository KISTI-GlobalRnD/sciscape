# SciScape Feature Definition

Status: product contract draft
Date: 2026-06-01

This document defines SciScape features by product capability. It is not a
screen-level UI specification. It is the contract that future CLI, local web
app, static viewer, and report surfaces should share.

For project identity, immutable rules, north-star direction, and claim
boundaries, see `branding_positioning.md`. For Atlas Map-specific adoption from
the NanoClustering Science Atlas Explorer, see `atlas_app_benchmark.md`. For
result-root schema and feature inference details, see `artifact_contract.md`.

## Product Shape

SciScape is a local-first workbench for validated science landscapes:

```text
query or files
-> bibliographic record preparation
-> network and matrix construction
-> clustering and hierarchy
-> keyword cleaning and labeling
-> maps, evidence, temporal/evolution review, and narratives
-> export, report, and validation artifacts
```

The product promise is not a single map. It is a reproducible workflow that
combines:

- VOSviewer-style science maps, term co-occurrence, thesaurus cleaning, and
  shareable maps.
- bibliometrix/Biblioshiny-style full-cycle bibliometric workflow.
- CiteSpace/SciMAT-style temporal knowledge-domain review, with advanced claims
  deferred until method artifacts exist.
- KnowledgeMatrix/VantagePoint-style tech mining, matrix building, cleaning,
  thesaurus rules, and analyst refinement.
- SciScape's own Rust CPM/Leiden clustering, keyword QA, artifact filtering,
  local web app, static viewer, and release gates.

## Benchmark Baseline

Use these tool families as benchmark pressure, not as UI templates.

| Tool family | Reference tools | Baseline capability | SciScape response |
| --- | --- | --- | --- |
| Science mapping | VOSviewer | citation, bibliographic coupling, co-citation, co-authorship, term co-occurrence, density/overlay maps, thesaurus cleaning | support core network, term-map, cleaning, and export equivalents with artifact contracts |
| Bibliometric workflow | bibliometrix / Biblioshiny | import, filtering, source/author/document metrics, conceptual/intellectual/social structures | support a smaller complete workflow first: ingest, network, landscape, keywords, report |
| Temporal knowledge mapping | CiteSpace / SciMAT | timeline, burst detection, thematic evolution, strategic diagram | support simple temporal lenses first; defer burst/thematic evolution until method contracts exist |
| Tech mining | KnowledgeMatrix / VantagePoint | user lists, matrix generation, cleaning, thesaurus, analyst refinement | make Matrix Builder and Cleaning first-class modes |
| Literature exploration | Connected Papers / ResearchRabbit / Litmaps / Open Knowledge Maps | query or seed to exploratory map | support query-to-result and curated static bundles with stronger QA/provenance |
| Graph exploration | Gephi / Cytoscape / Pajek | flexible graph editing, filtering, graph export | export to graph tools; do not become a full graph editor |
| Institutional analytics | SciVal / InCites / Dimensions Analytics | normalized impact, portfolio analytics, institutional benchmarking | defer until metric and data-license contracts exist |

References to verify when expanding scope:

- VOSviewer features: https://www.vosviewer.com/features/highlights
- Biblioshiny features: https://bibliometrix.org/home/index.php/layout/biblioshiny
- CiteSpace overview: https://cluster.cis.drexel.edu/~cchen/citespace/
- SciMAT overview: https://sci2s.ugr.es/scimat/
- KnowledgeMatrix paper: https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART001223521
- VantagePoint overview: https://www.thevantagepoint.com/

## Feature Maturity Labels

| Label | Meaning |
| --- | --- |
| `support-v1` | required for the first coherent app or already part of the stable product contract |
| `support-v1.5` | required for the analyst-workbench pass after v1 contracts stabilize |
| `support-v2` | supported later after core artifacts and UI are stable |
| `defer` | promising but not a product promise yet |
| `exclude` | intentionally outside SciScape scope |

## Workspace Model

The workspace is the user's analysis state, not a folder browser. UI/UX should
open modes and lenses from workspace objects and artifact contracts, not from
guessed paths.

### Workspace Objects

| Object | Meaning | Required fields |
| --- | --- | --- |
| `workspace` | top-level container for projects, runs, reusable rules, settings | `workspace_id`, `name`, `root`, `created_at_utc`, `updated_at_utc` |
| `project` | research question or review topic | `project_id`, `title`, `description`, `status`, `tags` |
| `dataset` | imported or fetched records before analysis | `dataset_id`, `source_type`, `record_count`, `source_manifest`, `records_path` |
| `run` | pipeline execution from query/files/dataset to result artifacts | `run_id`, `mode`, `state`, `config`, `started_at_utc`, `completed_at_utc` |
| `result` | validated result root that can power lenses | `result_id`, `result_root`, `artifact_contract`, `features`, `versions` |
| `artifact` | typed file or payload used by a mode or lens | `artifact_id`, `role`, `path`, `schema_version`, `checksum` |
| `rule_set` | replayable cleaning, alias, thesaurus, acronym, or filter rules | `rule_set_id`, `rule_type`, `rules_path`, `source`, `version` |
| `matrix` | matrix artifact plus row/column metadata and QA | `matrix_id`, `matrix_type`, `shape`, `weighting`, `qa_path` |
| `export` | generated shareable or downstream package | `export_id`, `format`, `source_result_id`, `path`, `created_at_utc` |

### Workspace States

| State | Meaning | UI implication |
| --- | --- | --- |
| `no_workspace` | no readable or writable root is selected | offer create/open workspace and curated demo |
| `workspace_loaded` | workspace manifest is valid | show projects, demos, recent results, and runnable modes |
| `dataset_loaded` | records can be addressed by stable IDs | enable filtering, network construction, matrix builder, and pipeline start |
| `run_active` | a job has heartbeat, logs, or partial artifacts | show progress, partial outputs, retry/cancel state |
| `result_loaded` | validation enables at least one lens | show artifact-backed lenses and export |
| `refinement_active` | rule edits or matrix/cleaning comparisons are staged | show compare, replay, and commit controls |
| `publish_ready` | result passes release/demo gates | enable static viewer bundle, report, or demo packaging |

### Workspace Navigation Targets

| Target | Primary objects | Role |
| --- | --- | --- |
| `Home` | workspace, project, demo, recent result | start from analysis state, not folders |
| `Run` | run, dataset, partial artifacts | job state, logs, partial outputs, recoverable failures |
| `Atlas` | result, membership, keywords, edges, evidence | cluster map, hierarchy, evidence, temporal/evolution, narrative |
| `Matrices` | matrix, dataset, result, rule_set | matrix builder, matrix QA, matrix table, matrix export |
| `Cleaning` | rule_set, keyword/entity tables, abstracts | raw/normalized/display audit, rule replay, before/after QA |
| `Validation` | artifact contract, QA artifacts | blocking issues, warnings, release readiness |
| `Exports` | result, export manifests | static viewer, reports, graph/matrix exports |

Workspace acceptance rules:

- A UI screen may display only modes and lenses backed by loaded workspace
  objects or advertised as unavailable with a reason.
- A disabled mode or lens must name the missing artifact, schema field, or
  blocking validation rule.
- User edits create or update rule artifacts; raw source data is not silently
  mutated.
- A result can be published only from a validated result object.
- Workspace state must be serializable so CLI, local web, and future static
  viewers can agree on what exists.

## App Modes

Modes are entry states. They decide what inputs are accepted, whether SciScape
runs computation, and what output state must be produced.

| Mode ID | User goal | Primary inputs | Runs pipeline | Required output state | Priority |
| --- | --- | --- | --- | --- | --- |
| `demo` | open a curated example immediately | demo manifest or hosted `data.json` | no | loaded result, provenance, validation | v1 |
| `static_viewer` | view a shared static result | hosted `data.json` or report bundle | no | loaded result, read-only provenance | v1 |
| `local_result` | inspect an existing output | result root, `data.json`, report dir | no | loaded result plus artifact contract | v1 |
| `live_query` | paste a query and build a small analysis | OpenAlex query and limits | yes | job state, saved result root, report bundle | v1 |
| `validation` | check whether a result is safe to inspect or publish | result root or `data.json` | no | artifact contract report | v1 |
| `report` | produce shareable report and exports | loaded result root | partial | HTML, JSON, export manifests, QA | v1 |
| `matrix_builder` | build occurrence/co-occurrence/proximity/similarity matrices | records, terms, fields, clusters | partial | matrix artifacts and matrix QA | v1.5 |
| `cleaning` | review and refine terms/entities/rules | keywords/entities, abstracts, rules | partial | cleaned table, rule log, QA | v1.5 |
| `file_pipeline` | run from user-provided bibliographic files | records, references, edges, metadata | yes | result root with manifest | v2 |
| `clustering` | configure and run CPM/Leiden landscape building | edge table, weights, levels, clustering config | yes | membership, hierarchy, clustering QA | v2 |

## Analysis Lenses

Lenses inspect a loaded result. They are enabled by feature availability, not by
hard-coded UI assumptions.

| Lens ID | Purpose | Required artifacts | Empty behavior |
| --- | --- | --- | --- |
| `overview` | dataset size, year span, source counts, warning summary | report data or abstracts | show validation summary only |
| `cluster_map` | inspect clusters, hierarchy, layout, lineage | membership or report cluster payload | disable with `missing_membership` |
| `keyword` | review representative terms and labels | keyword table or report keywords | disable with `missing_keywords` |
| `term_network` | inspect keyword co-occurrence | term-network or co-occurrence payload | disable with `missing_term_network` |
| `matrix` | inspect occurrence/co-occurrence/proximity/similarity matrices | matrix artifact plus metadata | disable with `missing_matrix` |
| `evidence` | inspect representative works and text evidence | abstracts plus membership or representative docs | show evidence unavailable warning |
| `temporal` | inspect publication-year or period trends | `pubyear` or temporal summary | disable with `missing_temporal_data` |
| `evolution` | inspect cluster lifecycle and composition shifts | evolution artifact | disable with `missing_evolution_data` |
| `narrative` | read evidence-backed cluster interpretation | narrative artifact or enough evidence for deterministic narrative | disable with `missing_narrative_evidence` |
| `quality` | inspect contamination, duplicates, schema issues | QA artifact or validation output | run lightweight validation when possible |
| `export` | prepare downstream outputs | result data plus writable/target location | show only available formats |

## Feature Catalog

Each feature below states product intent, benchmark pressure, maturity, modes,
lenses, inputs, outputs, feature flag rules, blocking conditions, and UI/UX
placement. This is the source of truth for future UI/UX discussion.

### F01. Workspace And Project Management

Purpose:

- Let users start from projects, datasets, runs, results, rules, and exports
  instead of internal folders.
- Make every analysis state recoverable and inspectable.

Benchmark pressure:

- Biblioshiny no-code workflow.
- Open Knowledge Maps query-first start.
- Atlas datapack identity and package QA.

Maturity:

- `support-v1` for workspace-loaded demos, local results, recent runs.
- `support-v1.5` for reusable rule sets and matrix objects.

Modes and lenses:

- Modes: `demo`, `local_result`, `live_query`, `validation`, `report`.
- Lenses: `overview`, `quality`, `export`.

Inputs:

- workspace root
- demo manifest
- existing result root
- job store

Artifacts:

- `workspace.json`
- `projects/<project_id>/project_manifest.json`
- `datasets/<dataset_id>/dataset_manifest.json`
- `runs/<run_id>/run_manifest.json`
- `rules/<rule_set_id>/rule_set_manifest.json`
- `views/<view_id>/view_manifest.json`
- run status/logs
- result artifact contract
- export manifest

Feature flag and state rules:

- `workspace_loaded=true` when manifest is valid or a lightweight workspace can
  be inferred safely.
- `result_loaded=true` only after artifact validation enables at least one lens.
- `publish_ready=true` only after release/demo gates pass.

Success:

- User can see recent results and supported entry modes.
- Local result opening does not require knowing internal folder structure.
- Missing modules are visible with reasons.

Partial:

- workspace opens but some projects or runs are missing artifacts.
- old result root loads with inferred features and warnings.

Blocking:

- workspace manifest is malformed.
- selected result has no supported artifact.
- validation finds a blocking issue for the requested action.

UI/UX placement:

- `Home` is a neutral workspace dashboard, not a marketing page and not an
  automatic demo-query overlay.
- Dense details belong in bounded inspectors, drawers, or validation panels.

### F02. Ingest And Normalize

Purpose:

- Convert research records into a standard, provenance-preserving table.
- Keep input adapters separate from downstream analysis claims.

Benchmark pressure:

- Biblioshiny import/conversion.
- VOSviewer source support.
- KnowledgeMatrix/VantagePoint preprocessing.

Maturity:

- `support-v1` for OpenAlex and existing local tables in CLI/local workflows.
- `support-v2` for broader app-level file adapters.
- `defer` for commercial/source-specific adapters without contract coverage.

Modes and lenses:

- Modes: `live_query`, `file_pipeline`, `validation`.
- Lenses: `overview`, `quality`.

Inputs:

- OpenAlex query results
- Web of Science, Scopus, PubMed, Dimensions, Lens, Crossref, BibTeX, CSV-like
  records when adapters exist
- local JSON/JSONL/parquet record tables

Minimum standard columns:

- `uid`
- `title`
- `abstract`
- `pubyear`
- optional author, venue, institution, reference, citation, DOI, URL fields

Artifacts:

- `records/records.parquet`
- `records/source_manifest.json`
- optional `records/raw/`
- optional `filters/*.json`

Feature flag rule:

- `overview=true` when records or report data include non-empty count fields.
- `temporal=true` only when `pubyear` exists and temporal grouping can run.

Success:

- Stable IDs are preserved.
- Source metadata and conversion metadata are recorded.
- Blank title/abstract and unsafe text artifacts are flagged.

Partial:

- Records load but optional entity fields are missing.
- Records parse but references/citations are absent.

Blocking:

- IDs cannot be reconciled.
- required fields are absent for the requested downstream mode.
- source conversion loses provenance.

UI/UX placement:

- `Home` and `Run` show source summary, record count, and field coverage.
- Adapter limitations appear before network or matrix construction starts.

### F03. Demo, Static Viewer, And Local Result Loading

Purpose:

- Let users inspect precomputed results without understanding folder layout.
- Support GitHub Pages-style static sharing and local result reopening.

Benchmark pressure:

- VOSviewer sharing.
- Open Knowledge Maps immediate map.
- Atlas datapack loading.

Maturity:

- `support-v1`.

Modes and lenses:

- Modes: `demo`, `static_viewer`, `local_result`, `validation`.
- Lenses: all artifact-backed lenses.

Inputs:

- demo manifest
- hosted `data.json`
- local result root
- `landscape/report/data.json`
- report directory

Artifacts:

- `report/data.json`
- `landscape/report/index.html`
- `landscape/report/report.html`
- `landscape/qa/artifact_contract.json`
- expected demo artifacts from `examples/demo_presets.json`

Feature flag rule:

- Feature flags come from validation and embedded `_sciscape`, not from UI
  assumptions.

Success:

- Demo loads with non-empty cluster/keyword surfaces.
- Static viewer loads a hosted `data.json`.
- Local result validates and enables only supported lenses.

Partial:

- `data.json` loads but advanced sidecars are absent.
- old result root lacks optional manifest or version artifacts.

Blocking:

- malformed JSON
- no supported result artifact
- claimed features do not match files
- release-quality demo missing expected artifacts

UI/UX placement:

- `Home` should show curated demos and recent local results.
- `Validation` should explain missing or blocked features.

### F04. Live Query And Job Execution

Purpose:

- Turn a bounded query into a saved result root with job state and recoverable
  partial outputs.

Benchmark pressure:

- Open Knowledge Maps query-first experience.
- ResearchRabbit/Litmaps exploratory starts.

Maturity:

- `support-v1` for small OpenAlex jobs.
- `support-v2` for broader sources and larger interactive pipelines.

Modes and lenses:

- Modes: `live_query`, `validation`, `report`.
- Lenses: `overview`, `cluster_map`, `keyword`, `term_network`, `quality`,
  `export` when artifacts exist.

Inputs:

- OpenAlex query
- filters
- maximum record count
- edge family choices
- OpenAlex timeout/retry/backoff settings for operator use
- optional OpenAlex API attempt and retry-wait budgets
- optional in-flight OpenAlex request checkpoint polling
- clustering/keyword settings appropriate for small jobs

Artifacts:

- `runs/<run_id>/source_query.json`
- job status/logs
- result root under workspace output
- report bundle
- OpenAlex API telemetry in job status/run state for live queries
- artifact contract

Feature flag rule:

- job output becomes `result_loaded` only after validation.

Success:

- Query fetch completes.
- Result root is saved.
- Report data and validation report exist.
- Transient OpenAlex 429, timeout, and 5xx failures are retried within bounded
  limits and visible in progress logs.
- OpenAlex API attempts, retry counts, wait seconds, status codes, and
  exception classes are persisted for operator review.
- OpenAlex API attempt or retry-wait budget overruns stop the job with
  inspectable status instead of allowing unbounded retries.
- Web OpenAlex jobs check cancellation while external HTTP requests are in
  flight, and CLI users can opt into the same checkpoint polling.
- Supported cluster-sharded keyword failures can be resumed from the app as a new
  validated internal CLI job when `run_state.resume.command` is present.
- Failed cluster-sharded keyword shards can be rerun through a shard-scoped
  validated resume job when failed shard IDs are present.
- User-selected cluster-sharded keyword shards can be rerun by submitting
  explicit shard IDs to a validated resume endpoint.
- Cluster-sharded resume jobs can optionally override bounded worker count and
  parallel backend without accepting arbitrary CLI flags.
- Failed or partial long runs expose a compact recoverability summary for failed
  shards, checkpoints, partial output kinds, and resume readiness.
- The job run-state API exposes recoverable artifacts, available resume/retry/
  cancel actions, and a recommended operator action without requiring users to
  inspect manifest JSON manually.

Partial:

- fetch succeeds but downstream network, clustering, keyword, or report step
  fails with partial artifacts preserved.
- user cancellation records `cancelled` state, partial artifacts, and
  checkpoint metadata when available.
- in-flight request polling improves cancellation responsiveness, but does not
  promise that an already-open HTTP socket is killed at the transport layer
  before the configured request timeout.
- resume is restricted to known SciScape keyword-shard commands; arbitrary shell
  commands and full queue pause/prioritization controls remain outside this v1
  surface.

Blocking:

- source fetch fails
- no usable records
- pipeline error leaves no inspectable state

UI/UX placement:

- `Run` shows progress, logs, transient OpenAlex retry waits, API telemetry,
  partial artifacts, retry/cancel/resume state.
- Query limits should be visible before execution.

### F05. Network Construction

Purpose:

- Build graph representations that support clustering, maps, evidence, and
  export.

Benchmark pressure:

- VOSviewer citation/bibliographic-coupling/co-citation/co-authorship networks.
- CiteSpace structural knowledge-domain review.
- Gephi/Cytoscape export expectations.

Maturity:

- `support-v1` for direct citation, bibliographic coupling, co-citation,
  semantic/embedding KNN when embeddings exist, and keyword co-occurrence.
- `support-v1.5` for co-authorship, institution, source, country networks when
  normalized fields exist.

Modes and lenses:

- Modes: `live_query`, `file_pipeline`, `report`, `validation`.
- Lenses: `cluster_map`, `term_network`, `evidence`, `export`.

Inputs:

- records
- references/citations
- embeddings when applicable
- normalized entity fields
- term tables

Artifacts:

- `edges.parquet`
- optional `networks/<network_id>/edges.parquet`
- network metadata
- optional layout artifacts
- edge-evidence samples

Required metadata:

- network type
- source fields
- weight field
- counting method
- normalization method
- threshold/filtering rules
- generated timestamp and SciScape version

Feature flag rule:

- `cluster_map` may use membership/report data without edges.
- `evidence` for neighbor relations requires edges plus membership.
- raw neighbor samples require an edge-evidence sidecar.

Success:

- Edge endpoints resolve to stable record IDs.
- Weight semantics are recorded.
- Network family is explicit.

Partial:

- edges exist but no layout or raw sample sidecar exists.
- some entity networks are unavailable due to missing fields.

Blocking:

- edge endpoints cannot join to records.
- weight field is missing or non-numeric when required.
- network type is unknown.

UI/UX placement:

- `Run` shows network build state.
- `Atlas` shows aggregate neighbor evidence.
- `Exports` sends graph data to external tools.

### F06. Matrix Builder

Purpose:

- Build VantagePoint/KnowledgeMatrix-style occurrence, co-occurrence,
  proximity, similarity, and 1-mode/2-mode matrices.
- Make matrices reusable for visualization, clustering, export, and analyst
  review.

Benchmark pressure:

- KnowledgeMatrix matrix generation.
- VantagePoint tech-mining matrices and analyst refinement.
- VOSviewer term maps and thesaurus workflows.

Maturity:

- `support-v1` for minimum term co-occurrence tables/maps.
- `support-v1.5` for general Matrix Builder mode.

Modes and lenses:

- Modes: `matrix_builder`, `validation`, `report`.
- Lenses: `matrix`, `term_network`, `quality`, `export`.

Inputs:

- record table with stable `uid`
- row field or row list
- column field or column list
- optional cluster labels, year bins, entity fields, or term rules
- weighting and normalization configuration

Supported matrix types:

| Matrix type | Example | Minimum output |
| --- | --- | --- |
| occurrence | document-term, cluster-term, author-keyword | sparse matrix plus row/column metadata |
| co-occurrence | term-term, author-author, institution-keyword | pair table or sparse symmetric matrix |
| proximity | terms within sentence/window/field | pair table with window metadata |
| similarity | cosine/Jaccard/association strength | pair table with metric metadata |
| 2-mode | document-term, author-institution, paper-cluster | rectangular matrix with row/column roles |
| 1-mode projection | term-term from document-term, author-author from paper-author | projected edge table plus projection rule |

Artifacts:

- `matrices/<matrix_id>/matrix.parquet` or `matrix.npz`
- `matrices/<matrix_id>/rows.parquet`
- `matrices/<matrix_id>/columns.parquet`
- `matrices/<matrix_id>/metadata.json`
- `matrices/<matrix_id>/qa.json`

Required metadata:

- `matrix_id`
- `matrix_type`
- row and column roles
- source artifact paths
- source field names
- weighting method
- normalization method
- filtering thresholds
- shape, density, non-zero count
- dropped row/column counts
- created timestamp and SciScape version

Feature flag rule:

- `matrix=true` only when matrix values and row/column metadata exist.
- `term_network=true` can be inferred from co-occurrence rows only when at
  least one edge exists.

Success:

- Matrix shape and row/column labels reconcile with source tables.
- Empty rows, empty columns, and dropped records are counted.
- Weighting is visible before visualization.

Partial:

- matrix exists but selected labels, weights, or source coverage are incomplete.

Blocking:

- no stable row IDs
- no stable column IDs
- unsupported weighting request
- empty matrix after filtering
- matrix values cannot be traced to source fields

UI/UX placement:

- `Matrices` starts from a loaded dataset or result, not a file picker alone.
- Preview shows shape, sparsity, top rows/columns, and top pairs before a map
  renders.
- Matrix builder controls are saved as replayable configuration.

### F07. Clustering And Hierarchy

Purpose:

- Generate stable, interpretable topic landscapes while preserving membership
  evidence and backend configuration.

Benchmark pressure:

- VOSviewer/Gephi expectations for configurable map construction.
- SciScape differentiation through Rust CPM/Leiden.
- Atlas hierarchy and lineage reading.

Maturity:

- `support-v1` for backend-preserved landscape generation in CLI/query flows.
- `support-v2` for user-configurable app-level clustering mode.

Modes and lenses:

- Modes: `live_query`, `file_pipeline`, `clustering`, `validation`.
- Lenses: `cluster_map`, `evidence`, `quality`, `export`.

Inputs:

- edge table
- weight column
- clustering config
- hierarchy/postprocess config when used
- random seed

Artifacts:

- `landscape/membership.parquet`
- hierarchy metadata
- clustering QA
- backend/version metadata
- optional postprocess artifacts

Feature flag rule:

- `cluster_map=true` when membership or report cluster payload exists.
- lineage and child overlays require parent/child columns or parent UID fields.

Success:

- membership joins to records
- backend, objective, parameters, seed, and version are preserved
- cluster sizes and small-cluster behavior are reported

Partial:

- clustering completes but keywords, hierarchy, or report are missing.

Blocking:

- edge table invalid
- backend fails without useful partial output
- membership cannot join to records
- research-only Dongdaemun diagnostics are promoted as production clustering

UI/UX placement:

- `Run` reports clustering state.
- `Atlas` reads cluster map and hierarchy from result artifacts.
- Advanced clustering controls stay out of v1 UI.

### F08. Keyword Extraction, Labels, And Cleaning

Purpose:

- Produce interpretable labels across user domains and preserve the analyst
  cleaning path from raw terms to display labels.

Benchmark pressure:

- VOSviewer thesaurus cleaning.
- KnowledgeMatrix/VantagePoint cleaning and analyst refinement.
- SciScape keyword QA and artifact filtering.

Maturity:

- `support-v1` for extraction, representative labels, artifact filtering,
  read-only cleaning audit.
- `support-v1.5` for editable/replayable rules, aliasing, acronyms, and family
  hierarchy.

Modes and lenses:

- Modes: `live_query`, `cleaning`, `report`, `validation`.
- Lenses: `keyword`, `quality`, `narrative`, `export`.

Inputs:

- records/abstracts
- membership
- keyword extraction config
- optional rule sets
- optional thesaurus/alias/acronym files

Artifacts:

- `landscape/keywords.parquet`
- cleaned keyword table
- `rules/<rule_set_id>/rule_set_manifest.json`
- `rules/<rule_set_id>/rules.parquet`
- `rules/<rule_set_id>/rule_applications.parquet`
- `rules/<rule_set_id>/term_before_after.parquet`
- `rules/<rule_set_id>/impact_summary.json`
- `rules/<rule_set_id>/rule_set_qa.json`

Required cleaned keyword fields:

- `raw_term`
- `normalized_term`
- `display_label`
- `representative_label`
- `family_id`
- `cluster_id`
- `score`
- `frequency` or `count`
- `ngram`
- `quality_flags`
- `rule_ids`
- `review_status`

Feature flag rule:

- `keyword=true` when keyword table or report keywords exist.
- release-quality keyword display is blocked by top-ranked metadata, HTML, or
  LaTeX artifacts.

Success:

- unigram, bigram, and trigram behavior is explicit
- score/frequency/count are preserved
- common-vs-cluster-specific terms are distinguishable
- acronym evidence is preserved when extracted
- raw terms are preserved through cleaning

Partial:

- audit exists but no rules are applied
- acronym or alias match is ambiguous and flagged for review

Blocking:

- cleaned output loses required keyword columns
- rule replay is not reproducible
- raw terms cannot be recovered
- metadata/HTML/LaTeX artifacts appear in top-ranked display labels
- keyword rule QA is blocked or references unresolved rule ids

UI/UX placement:

- `Cleaning` shows raw, normalized, display, family, count, score, and flags in
  one audit table.
- Edits are staged, previewed, replayed, and then committed to rule artifacts.
- Dense review tables belong in bounded panels or drawers.

### F09. Atlas Map, Evidence, And Cluster Reading

Purpose:

- Make cluster landscapes readable through identity, hierarchy, lineage,
  neighbor relations, representative works, and QA badges.

Benchmark pressure:

- Science Atlas Explorer cluster-reading workflow.
- VOSviewer science maps.
- Open Knowledge Maps knowledge-map experience.

Maturity:

- `support-v1` for cluster map, hierarchy, representative works when joinable,
  neighbor aggregate evidence, module readiness, and URL state.
- `support-v1.5` for richer boundary, terrain, and evolution-backed map views.

Modes and lenses:

- Modes: `demo`, `static_viewer`, `local_result`, `live_query`.
- Lenses: `cluster_map`, `evidence`, `quality`, `temporal`, `evolution`,
  `narrative`.

Inputs:

- report data
- membership
- abstracts
- edges
- keywords
- optional edge-evidence sidecars
- optional layout/evolution/narrative artifacts

Artifacts:

- Atlas payload under `_sciscape.atlas`
- `landscape/membership.parquet`
- `landscape/report/data.json`
- `landscape/edge_evidence_samples.json`
- optional layout/evolution/narrative sidecars

Feature flag rule:

- `cluster_map=true` with membership or report cluster payload.
- representative works require abstracts plus membership.
- raw neighbor samples require edge-evidence sidecar.
- evolution/narrative panels require their own artifacts.

Success:

- nodes have stable `cluster_uid`, `level`, `cluster_id`, label, and keyword
  counts
- membership enriches doc counts, hierarchy, lineage, and child counts
- evidence surfaces distinguish aggregate neighbor facts from raw samples

Partial:

- report payload loads but doc counts or raw evidence are unavailable
- aggregate neighbor edges exist without raw sample sidecar

Blocking:

- identity fields are missing
- advertised lens lacks backing artifact
- hierarchy assumptions are hard-coded to one dataset family

UI/UX placement:

- `Atlas` is the primary loaded-result reading surface.
- Start from a neutral atlas default, not an automatic demo-query overlay.
- Detail-heavy evidence uses progressive-disclosure drawers or bounded panels.

### F10. Term Network And Co-Occurrence Visualization

Purpose:

- Let users inspect keyword co-occurrence as graph, table, and map evidence.

Benchmark pressure:

- VOSviewer term co-occurrence.
- KnowledgeMatrix matrix views.

Maturity:

- `support-v1` for term network graph/table/map from keyword/co-occurrence
  artifacts.

Modes and lenses:

- Modes: `report`, `validation`, `matrix_builder`.
- Lenses: `term_network`, `matrix`, `keyword`, `quality`, `export`.

Inputs:

- keyword table
- term co-occurrence rows
- optional matrix artifact

Artifacts:

- term-network payload
- co-occurrence table
- optional matrix artifact
- report/dashboard data

Feature flag rule:

- `term_network=true` only when at least one term edge exists.
- term-network terms should resolve to keyword terms or carry
  `external_term=true`.

Success:

- graph, table, and map agree on edge counts
- co-occurrence rows preserve counts/weights
- top terms remain inspectable by cluster

Partial:

- terms exist but no co-occurrence edges after filtering

Blocking:

- advertised term network has no edges
- terms cannot be reconciled with keyword table and have no external marker

UI/UX placement:

- `Atlas` may show compact term evidence.
- `Matrices` and `Exports` provide dense term-pair tables and downstream files.

### F11. Temporal And Evolution

Purpose:

- Support temporal reasoning while preventing unsupported burst/evolution
  claims.

Benchmark pressure:

- CiteSpace burst/timeline.
- SciMAT thematic evolution and strategic diagrams.
- Science Atlas cluster evolution maps.

Maturity:

- `support-v1` for simple temporal trend when `pubyear` exists.
- `support-v1.5` for cluster evolution map when evolution artifacts exist.
- `support-v2` for burst/thematic evolution after method contracts exist.

Modes and lenses:

- Modes: `report`, `validation`, future `file_pipeline`.
- Lenses: `temporal`, `evolution`, `narrative`, `quality`.

Inputs:

- records with `pubyear`
- membership or cluster payload
- period definitions
- transition/evolution artifacts for evolution

Artifacts:

- `temporal/temporal_manifest.json`
- `temporal/periods.parquet`
- `temporal/activity.parquet`
- `temporal/entity_series.parquet`
- `evolution/evolution_manifest.json`
- `evolution/time_slices.parquet`
- `evolution/cluster_states.parquet`
- `evolution/transitions.parquet`
- `evolution/lineages.parquet`
- `evolution/evolution_events.parquet`
- `evolution/evolution_qa.json`

Feature flag rule:

- `temporal=true` with `pubyear` and yearly grouping.
- `evolution=true` only with non-empty, validated evolution artifacts.
- burst/thematic evolution remains disabled without method-specific artifacts.

Success:

- period definitions are explicit
- cluster IDs reconcile with membership/report payload
- transition weights explain their denominator
- missing years or sparse periods are visible

Partial:

- yearly trends exist but no cluster evolution artifact exists

Blocking:

- publication years exist but cluster-year mapping is missing for evolution
- transition endpoints cannot resolve to cluster UIDs
- temporal metric lacks method metadata
- burst/thematic output lacks method parameters

UI/UX placement:

- `Temporal` can be a simple trend panel.
- `Evolution` is separate and appears only when evolution artifacts exist.

### F12. Evidence-Backed Narratives

Purpose:

- Convert cluster labels into auditable interpretations linked to evidence.

Benchmark pressure:

- Science Atlas Explorer narrative direction.
- Analyst review workflows.

Maturity:

- `support-v1.5` for evidence-backed cluster narrative view.
- LLM canonicalization remains `defer` unless fully auditable and optional.

Modes and lenses:

- Modes: `report`, `validation`, future `cleaning`.
- Lenses: `narrative`, `evidence`, `temporal`, `evolution`, `quality`.

Allowed evidence sources:

- representative keywords and keyword families
- common-vs-cluster-specific terms
- representative works
- abstracts or snippets when permitted
- lineage and child clusters
- neighbor relations and edge-evidence samples
- temporal or evolution artifacts
- matrix/co-occurrence evidence
- QA warnings and missing-feature caveats

Artifacts:

- `narrative/narrative_manifest.json`
- `narrative/narrative_targets.parquet`
- `narrative/claims.parquet`
- `narrative/evidence_sources.parquet`
- `narrative/evidence_refs.parquet`
- `narrative/claim_evidence_links.parquet`
- `narrative/narrative_sections.parquet`
- `narrative/review_decisions.parquet`
- `narrative/narrative_qa.json`
- `narrative/generation_metadata.json`
- `narrative/generation_prompts/prompt_batch_manifest.json`
- `narrative/generation_prompts/prompt_jobs.jsonl`
- `narrative/generation_outputs/generation_run_manifest.json`
- `narrative/generation_outputs/generated_claims.jsonl`
- `narrative/publication_summary.json`
- `narrative/publication_summary.md`
- `narrative/publication_summary.html`
- `narrative/publication_bundle.zip`

Feature flag rule:

- `narrative=true` only when validated narrative artifacts exist and evidence
  references can be resolved. Deterministic scaffolds without written
  artifacts do not enable the narrative lens.

Success:

- every claim links to evidence
- missing evidence appears as a caveat
- narrative claim, evidence, section, review, and QA artifacts are exposed through
  the result manifest and downloads surface when present
- reviewed narrative publication summaries render only accepted or not-required
  claims and list rejected, needs-revision, and pending claims as omitted rows
- reviewed narrative publication summaries include a cluster-level index with
  rendered, omitted, pending, and publication-state counts for multi-cluster
  report navigation
- Atlas narrative review surfaces link to reviewed publication Markdown, HTML,
  and JSON artifacts when they have been written, and can preview the reviewed
  publication JSON in the inspector
- reviewed narrative publication bundles include publication summaries,
  generation metadata, QA, review decisions, and claim/evidence tables
- reviewed narrative publication artifacts can be generated through the CLI
  after review decisions are present
- reviewed narrative publication artifacts can be refreshed through the web API
  and Atlas Narrative review panel without generating or promoting new claim
  text
- generated narratives include prompt/model metadata when LLMs are used
- generation prompt batches can be rendered from validated claim/evidence
  artifacts before any provider is called
- generation prompt batches can be executed through an explicit provider runner
  that writes generated claim-update JSONL before the safe apply step
- generation metadata, prompt batches, and generated-candidate outputs can be
  inspected through the web API and Atlas Narrative review panel without
  invoking a provider from the app
- model-assisted narrative claim updates preserve existing evidence links,
  reset generated claims to a review-required state, and refresh generation
  metadata plus QA before publication
- model-assisted narrative claim updates can be applied from JSON or JSONL
  batch outputs through the CLI, but prompt execution remains a separate
  auditable generation step
- model-assisted narrative claim candidates can be applied from existing
  manifest-backed generated JSONL outputs through the web API and Atlas
  Narrative review panel, but applied claims remain review-required until a
  reviewer accepts them
- Atlas review readiness and the review queue surface narrative review debt,
  including pending model-generated claims, before publication
- Atlas Narrative review panels summarize publication review debt and show
  blocker claims before already-ready claims
- Atlas Narrative review panels disclose when additional publication blockers
  remain outside the compact visible claim set
- Atlas Narrative review panels can switch claim scope between compact,
  review-required, and all loaded claims for the selected cluster
- reviewed publication summaries expose machine-readable readiness that
  distinguishes partial reviewed reports from full publication readiness
- job and narrative API summaries expose compact reviewed publication state,
  readiness, counts, and artifact path when publication artifacts exist
- Atlas Narrative review panels render compact reviewed publication readiness
  from job summary state before users open the full preview
- Atlas Narrative publish actions use reviewed publication readiness to
  distinguish first publish, partial report refresh, full report refresh, and
  blocked retry states
- Atlas review, generated-candidate apply, result load, and publication refresh
  actions invalidate or refresh stale reviewed publication previews
- deterministic narrative scaffolds include generation metadata that records
  source artifacts, transform steps, and scaffold parameters
- narrative edits are recorded as review decisions and the latest decision is
  readable from the cluster narrative view
- cluster narrative views expose reviewed and pending claim counts

Partial:

- deterministic scaffold exists but some optional evidence folds are missing

Blocking:

- narrative text has no evidence references
- evidence references cannot be resolved
- narrative contradicts QA flags or missing-feature state
- model-generated text is presented as deterministic analysis without metadata

UI/UX placement:

- `Narrative` is a reading surface with evidence folds, not a decorative card.
- Users can jump from a sentence to terms, works, neighbors, evolution rows, or
  QA caveats.

### F13. Validation And QA

Purpose:

- Prevent contaminated, incomplete, or unsupported outputs from becoming demos,
  release artifacts, or trustworthy narratives.

Benchmark pressure:

- Atlas package QA.
- SciScape artifact filtering and release gate needs.

Maturity:

- `support-v1`.

Modes and lenses:

- Modes: `validation`, all result-loading modes.
- Lenses: `quality`, all feature-gated lenses.

Inputs:

- result root
- report data
- abstracts
- membership
- keywords
- edges
- optional matrix/evolution/narrative/export artifacts

Artifacts:

- `landscape/qa/artifact_contract.json`
- validation JSON
- warnings and blocking issue list

Feature flag rule:

- all features are inferred from artifacts, embedded contracts, or validation.

Success:

- required files exist for claimed mode
- required columns exist
- counts reconcile or carry a documented caveat
- keyword artifact checks pass release thresholds
- feature flags match artifacts

Partial:

- warnings exist but at least one lens can load safely

Blocking:

- no supported artifact can be validated
- claimed features are missing artifacts
- top-ranked keyword artifacts leak into release-quality outputs
- schema inconsistencies prevent interpretation

UI/UX placement:

- `Validation` shows blocking issues first, then warnings, then feature
  availability.
- Empty panels are avoided; disabled lenses explain missing inputs.

### F14. Report, Export, And Interoperability

Purpose:

- Preserve outputs for review, publication, static sharing, and downstream
  graph/matrix tools.

Benchmark pressure:

- VOSviewer sharing and thesaurus interop.
- Gephi/Cytoscape/Pajek graph workflows.
- Biblioshiny report-style outputs.

Maturity:

- `support-v1` for report/static viewer, keywords, membership, JSON/HTML.
- `support-v1.5` for matrix export and richer graph/VOSviewer interop.

Modes and lenses:

- Modes: `report`, `demo`, `static_viewer`, `local_result`.
- Lenses: `export`, `quality`, all loaded result lenses.

Export families:

| Export | Source artifacts | Required metadata |
| --- | --- | --- |
| static viewer | `report/data.json`, optional compact assets | schema version and feature flags |
| HTML report | report payload and QA | source result and generated timestamp |
| keyword table | cleaned keywords | rule set and QA hash |
| graph export | nodes, edges, labels, layout | graph type, weight field, coordinate source |
| matrix export | matrix plus row/column metadata | matrix type and weighting |
| VOSviewer export | map/network/thesaurus-compatible tables | field mapping and counting method |
| package manifest | complete result or demo bundle | artifact list and checksums |

Artifacts:

- `report/data.json`
- HTML report/dashboard
- keyword CSV/TSV/parquet
- graph export files
- matrix export files
- static viewer bundle
- `exports/<export_id>/export_manifest.json`
- `exports/<export_id>/export_files.parquet`
- `exports/<export_id>/export_inputs.parquet`
- `exports/<export_id>/export_transforms.parquet`
- `exports/<export_id>/export_qa.json`

Feature flag rule:

- `export=true` when loaded result has enough data to write at least one
  supported export. Stable export exposure requires an export manifest and QA;
  legacy file-only exports are beta.

Success:

- exported files identify the source result
- stable IDs are preserved
- node/edge/matrix labels remain interpretable outside SciScape
- selected/visible subset exports record the selection and apply it where the
  target export format supports subset output
- export manifests record source artifacts, transforms, file inventory, and QA

Partial:

- some export formats are unavailable due to missing inputs

Blocking:

- export drops required IDs
- export uses labels without stable source keys
- graph export lacks required weight or relation type
- public bundle includes private paths or unintended raw text
- advertised export manifest has missing primary files or blocked QA

UI/UX placement:

- `Exports` shows available formats, missing requirements, and privacy caveats.
- Public static bundles should omit raw abstracts unless explicitly requested.

### F15. Institutional Analytics

Purpose:

- Reserve space for future portfolio, normalized impact, and institutional
  benchmarking without promising it prematurely.

Benchmark pressure:

- SciVal, InCites, Dimensions Analytics.

Maturity:

- `defer`.

Modes and lenses:

- none in v1 or v1.5.

Inputs:

- controlled citation metrics
- field-normalized baselines
- institutional/entity disambiguation
- data-license contract

Feature flag rule:

- disabled until metric definitions and data-license boundaries exist.

Blocking:

- no normalized metric contract
- no field normalization rule
- no data-license permission
- entity disambiguation not auditable

UI/UX placement:

- Do not show portfolio or normalized impact dashboards in the first app.

## Cross-Feature Contracts

### Feature Availability Block

Every loaded result should reduce to a compact block generated by validation,
not hand-written in the UI.

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
| `evolution` | validated evolution artifact exists with non-empty states, transitions, lineages, events, and QA |
| `narrative` | validated narrative artifact exists with supported claims and resolvable evidence refs |
| `quality` | QA report exists or validation can run |
| `export` | validated export manifest exists, or loaded result has enough data for beta legacy exports |

### Result States

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

### Required Count Reconciliation

- every `membership.uid` should exist in the abstract/record table when both
  are present
- every keyword `cluster_id` should exist in membership or report cluster
  payload when both are present
- term-network terms should resolve to keyword terms or carry an explicit
  `external_term=true` marker
- report cluster counts should match membership counts or carry a documented
  sampling/filtering note
- matrix rows/columns should resolve to source records, terms, entities, or
  documented external labels

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

- Local file and local result modes should not require remote upload.
- Optional LLM or external API features must be explicit and disabled by default
  for private data.
- Reports should not silently embed raw abstracts when a compact public bundle
  is requested.
- Exported demos should include enough provenance to be interpretable without
  leaking private local paths.

### Data Size Classes

Size classes are product guidance, not performance guarantees. Claims must be
verified with release-gate artifacts before they become public limits.

| Class | Approximate size | Recommended surface | Notes |
| --- | --- | --- | --- |
| `demo_small` | up to about 1,500 records | web app, static viewer, docs demos | target for query-to-result examples |
| `analyst_medium` | about 1,500 to 50,000 records | CLI pipeline plus local result viewer | app may inspect precomputed outputs |
| `batch_large` | more than 50,000 records | CLI/batch only unless precomputed | do not run interactively in the browser |
| `atlas_scale` | hundreds of thousands or more | separate datapack-style architecture | outside normal SciScape app v1 |

### Release Quality Gates

No output should be promoted to a curated demo or release artifact unless:

- validation has no blocking issues
- advertised feature flags match actual artifacts
- keyword artifact checks pass release thresholds
- report/static viewer smoke tests load a non-empty result
- count reconciliation is recorded or explicitly explained
- source and version metadata are visible to users

### Workspace-Ready Feature Checklist

A feature is ready for UI/UX design only when all of the following are defined:

1. user goal and mode or lens placement
2. required input objects
3. generated artifact names and schemas
4. feature flag inference rule
5. success, partial, and blocking states
6. QA checks and release gate behavior
7. export behavior if applicable
8. privacy/locality caveats
9. interaction entry point in the workspace navigation model

## Product Scope

### V1 Scope

The first coherent SciScape app should support:

1. `demo` mode with a curated result.
2. `static_viewer` mode for GitHub Pages-style `data.json`.
3. `local_result` mode for existing result roots and report folders.
4. `live_query` mode for small OpenAlex examples.
5. `validation` mode as a shared contract.
6. `overview`, `cluster_map`, `keyword`, `term_network`, `quality`, and
   `export` lenses.
7. minimum matrix support for term co-occurrence tables and maps.
8. read-only cleaning audit for raw, normalized, display, family, and artifact
   fields.
9. lightweight temporal trend summaries when `pubyear` is available.

### V1.5 Scope

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

### V2 Scope

Later, add heavier or more research-dependent features:

1. `file_pipeline` mode for broader bibliographic import.
2. `clustering` mode with advanced configuration.
3. temporal evolution and burst-like analysis.
4. institutional benchmarking only if normalized metric contracts are added.
5. richer graph-editor behavior only through export or a clearly scoped viewer.

## Non-Goals

- Do not copy the NanoClustering Atlas UI directly.
- Do not copy the full Atlas sidecar architecture.
- Do not turn the SciScape web app into a general graph editor.
- Do not claim institutional benchmarking without controlled data and metric
  contracts.
- Do not hard-code field-specific labels or cluster levels into the public
  contract.
- Do not make UI affordances the source of truth for feature availability.

## Implementation Order

1. Maintain result and workspace manifest contracts so users can load outputs
   without knowing the internal folder layout.
2. Stabilize artifact validation, feature inference, and exposure states for
   existing result roots.
3. Embed feature/version/export blocks into `report/data.json`, static reports,
   and result manifests.
4. Add long-run progress, shard, checkpoint, partial-output, and resume metadata
   for live query and keyword extraction.
5. Stabilize demo/static/local/query loading around manifest-backed result
   states.
6. Stabilize the query-to-atlas analysis spine: source records, networks,
   clustering, hierarchy, minimal keywords, atlas payload, co-occurrence table,
   exports, and QA.
7. Add cleaning rule artifacts for aliases, stop terms, acronyms, metadata
   fragments, and artifact filters.
8. Add stable co-occurrence table/map artifact contracts before the general
   Matrix Builder.
9. Add general matrix artifact contracts, writers, validators, and exports.
10. Add temporal/evolution artifact contracts before exposing evolution UI.
11. Add narrative artifacts only after evidence references are resolvable.
12. Redesign the UI around workspace, modes, and lenses after P0, P1, and P1.5
   contracts stabilize.
