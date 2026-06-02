# Atlas App Benchmark For SciScape

Status: active benchmark
Date: 2026-06-01

This note benchmarks the NanoClustering Science Atlas Explorer as a reference
for SciScape's Atlas Map direction. The goal is not to copy the Atlas app
wholesale. It is to preserve the parts that make cluster landscapes readable,
auditable, replaceable, and smoke-testable while rejecting assumptions that only
fit the NanoClustering release package.

Use this note together with `feature_definition.md`:

- this note says which Atlas Map behaviors SciScape should adopt, adapt, defer,
  or reject;
- `feature_definition.md` says how those choices fit into SciScape modes,
  lenses, inputs, outputs, and validation rules.

## Reference Surfaces

Benchmarked from:

- `/home/kimyoungjin06/Desktop/Workspace/1.1.5.science-atlas-explorer/`
- `/home/kimyoungjin06/Desktop/Workspace/1.1.4.KISTI_NanoClustering/apps/science-atlas-explorer/`
- `scripts/serve_science_atlas_api.py`
- `outputs/specter2/_current/science_atlas_explorer_paris_datapack_domain8_micro3773_desc_work_20260528/`

Useful Atlas files and roles:

| Atlas surface | Role |
| --- | --- |
| `MANIFEST.json` | top-level datapack identity, module map, counts, source paths, API attachment roles |
| `atlas_versions.json` | stable version block exposed by `/versions` and echoed in API payloads |
| `core/` | cluster spine: nodes, terms, representatives, neighbor edges, search docs |
| `layout/`, `terrain_layout/`, `graph_layout/` | replaceable layout modules with their own manifests and QA |
| `dashboard/` | derived mart for review, source health, search documents, progress |
| `work/` | document-level provenance and title search package |
| `descriptions/`, `excellence/` | optional enrichment modules with independent manifests |
| `qa/` | package QA, count reconciliation, and readiness proof |

## Decision Vocabulary

Use the following decisions for every Atlas Map feature before it becomes a
SciScape product promise.

| Decision | Meaning |
| --- | --- |
| `adopt` | The feature concept should move into SciScape with only naming/schema changes. |
| `adapt` | The feature is valuable, but the Atlas implementation or assumptions must be reshaped for generic SciScape results. |
| `defer` | The feature is in scope, but should remain disabled until the required artifact contract exists. |
| `reject` | The feature or implementation assumption should not become part of SciScape. |

## Atlas Map Absorption Matrix

### Stage, State, And Navigation

| Atlas surface | Atlas behavior | SciScape decision | Contract or implementation note |
| --- | --- | --- | --- |
| Neutral atlas first screen | Starts from the map without forcing a demo query or selected cluster. | `adopt` | Demo data should be a selectable mode, not the default mental model. |
| `StageModeSwitch` | Switches between Map and Hierarchy Map. | `adapt` | SciScape v1 implements lightweight Atlas view modes: `Map`, `Hierarchy`, and `Evidence`, restorable with `atlas_view`. Later `Evolution` should appear only when evolution artifacts exist. |
| URL/view-state restore | Stores selected cluster, query, lens, layer visibility, and view settings. | `adapt` | SciScape v1 restores lightweight `atlas_node`, `atlas_level`, `atlas_query`, `atlas_lens`, `atlas_view`, and `atlas_focus` query parameters for review links. |
| Result rail session | Recent searches, pinned clusters, selected result context, and mini drill state. | `adapt` | SciScape v1 keeps this browser-local: selected result context, pinned clusters, recent cluster selections, and grouped search rows over the loaded payload. |
| Runtime status and connection pills | Makes API/data-load state visible. | `adopt` | SciScape v1 recasts this as a compact Atlas orientation strip showing result source, contract state, and warning count from the loaded payload. |

### Map Reading And Layer Controls

| Atlas surface | Atlas behavior | SciScape decision | Contract or implementation note |
| --- | --- | --- | --- |
| Layer visibility controls | Independent Domain/Macro/Meso/Micro/Nano visibility, with Micro-only as a derived state. | `adopt` | SciScape should support generic ordered levels. Default labels can be Domain/Macro/Meso/Micro, but public code should accept any level names. |
| Selected lineage overlay | Highlights the selected cluster path through parent levels. | `adopt` | Requires stable `cluster_uid`, `level`, `parent_uid`, and `child_count`. |
| Child overlay | Reveals the selected child layer even when the child level is hidden. | `adopt` | SciScape v1 exposes selected-node children in the inspector as clickable child-cluster rows, using `parent_uid` and `child_count`. |
| Map orientation legend | Bounded, first-glance map explanation and warning summary. | `adopt` | SciScape v1 renders a payload-derived orientation strip for source, level path, visible coverage, evidence readiness, warnings, and selected lineage. |
| Map quick controls | Near/Fit/Global and compact camera controls. | `adapt` | SciScape v1 uses payload-derived focus controls instead: `Global`, `Family`, `Neighbors`, and `Pinned`, restorable with `atlas_focus`. |
| Map basis: Embedding/Terrain/Graph | Alternate coordinate bases over the same cluster landscape. | `adapt` then `defer` | Start with the existing layout. Enable Terrain/Graph only when layout artifacts and QA exist. |
| Boundary region geometry | Renders territories and boundary pressure. | `defer` | Needs explicit boundary polygons or reproducible region construction, plus cross-region evidence. |

### Search, Evidence, And Inspector

| Atlas surface | Atlas behavior | SciScape decision | Contract or implementation note |
| --- | --- | --- | --- |
| Grouped search results | Results grouped by hierarchy level with score context. | `adopt` | SciScape v1 provides client-side Atlas search over loaded cluster labels, representative terms, and representative works, grouped by generic level and restorable with `atlas_query`. |
| `ResultEvidenceDrawer` | Explains why search results matched by score, level, docs, children, and badges. | `adopt` | SciScape v1 renders this as a bounded selected-cluster evidence profile using existing payload fields: record join, terms, works, lineage, neighbors, and QA. |
| Inspector overview | Orders identity, interpretation, terms, lineage, children, neighbors, papers, and QC. | `adopt` | This should become the primary cluster-reading surface in SciScape Web. |
| Selected lineage strip | Human-readable breadcrumb for the current cluster. | `adopt` | Use generic level labels and stable cluster IDs. |
| Representative works | Shows documents supporting the selected cluster. | `adapt` | Must be disabled unless record-level provenance exists and joins to membership. |
| OpenAlex taxonomy block | Shows OpenAlex domain/field/subfield/topic profile. | `adapt` | Optional enrichment only. Do not make OpenAlex taxonomy part of the core contract. |
| Neighbor block | Shows adjacent clusters with relation label, strength, same-parent/cross-parent, and shared terms. | `adapt` | SciScape v1 shows neighbor rows plus a relation summary for same-parent, cross-cluster, shared-term coverage, and aggregate paper links. Do not require raw paper-pair samples in v1. |
| Neighbor evidence drawer | Shows raw work-pair or nano-pair evidence for a cluster relation. | `adapt` | SciScape v1 exposes an aggregate relation-evidence panel from neighbor weight, edge count, relation label, and shared terms; `run_landscape()` now writes bounded `edge_evidence_samples.json` sidecars when source artifacts exist, and the same panel shows those raw samples. |
| Narrative panel | Reads the cluster as an evidence-backed story. | `defer` | Must cite terms, representative works, lineage, neighbors, temporal/evolution evidence, and QA caveats. No unsupported automatic narrative. |

### Lenses, Metrics, And Time

| Atlas surface | Atlas behavior | SciScape decision | Contract or implementation note |
| --- | --- | --- | --- |
| Cluster lens | Colors clusters by identity. | `adopt` | SciScape v1 keeps this as the default identity view over labels and cluster IDs. |
| Scale lens | Colors by document count or size. | `adopt` | SciScape v1 normalizes document-count intensity within the visible level and exposes `atlas_lens=scale`. |
| Growth lens | Colors by rolling temporal growth. | `adapt` | Enable when `pubyear` or temporal summaries exist. |
| Impact/prominence lens | Colors by citation excellence. | `defer` | Requires controlled citation metrics and normalization contract. |
| Focus/concentration lens | Colors by concentration of excellence or topical focus. | `defer` | Requires a clear metric definition. Avoid vague "focus" scores. |
| Boundary lens | Colors or highlights cross-region pressure. | `defer` | Requires neighbor/cross-parent evidence and boundary geometry. |
| Quality lens | Colors by QC badges and artifact risk. | `adapt` | SciScape v1 ships a conservative `Evidence` lens instead, combining record join, terms, works, lineage, neighbors, and QA readiness with `atlas_lens=evidence`. |
| Metric lens scaling panel | Explains min/max, normalization scope, and missing metric coverage. | `adapt` | SciScape v1 shows a compact lens scale strip for scope, coverage, min/median/max range, and normalization over the visible level. |
| Evolution overlay | Shows lifecycle, yearly activity, topic/child trajectories, genealogy, and milestones. | `defer` | High-value v1.5 target, but must be artifact-backed before UI exposure. |

### Performance And Packaging

| Atlas surface | Atlas behavior | SciScape decision | Contract or implementation note |
| --- | --- | --- | --- |
| Feature flags from `/versions` | UI exposes only payload-backed features. | `adopt` | SciScape v1 infers feature flags from the result root and artifact contract, then exposes a compact Atlas module-readiness strip in the loaded result. |
| Lazy metric hydration | First paint omits heavy lens metrics and loads them on demand. | `adapt` | Useful after result roots become large. Do not add complexity for small demos first. |
| Code-split heavy panels | Hierarchy graph, evidence drawer, dashboard, and evolution load on first use. | `adapt` | Use only in the redesigned web app, not in the current static viewer. |
| Large sidecar package | Atlas uses many modules and marts. | `reject` for direct copy | SciScape should keep a smaller result root until a workflow proves it needs extra sidecars. |

## What SciScape Should Not Copy

| Atlas assumption | Why it should not move directly into SciScape |
| --- | --- |
| Fixed Domain/Macro/Meso/Micro/Nano hierarchy | SciScape should handle generic cluster levels and only use these as defaults. |
| NanoClustering/SPECTER2/Paris datapack identity | These are reference packages, not SciScape product assumptions. |
| UI claims for absent sidecars | Every lens must be disabled or caveated when its backing artifact is missing. |
| Raw neighbor sample promises without attached sample tables | The live package previously exposed neighbor edges while raw sample evidence was not always attached. SciScape must keep these as separate features. |
| Automatic cluster narrative without evidence links | Narrative is allowed only when every claim can point to terms, works, lineage, neighbors, temporal data, or QA caveats. |
| Full graph-editor behavior | SciScape should export to graph tools instead of becoming Gephi/Cytoscape. |
| Institutional ranking/impact dashboards by default | These need metric, license, and normalization contracts outside the core SciScape package. |
| Large Atlas-specific API/server topology | SciScape should support static `data.json`, local result roots, and the existing lightweight web server first. |

## Atlas Patterns Worth Copying

1. Module separation is explicit.

   Atlas separates the package into stable modules: `core`, `layout`,
   `dashboard`, `work`, enrichment modules, and `qa`. SciScape does not need the
   same scale, but it should stop treating `report/data.json` as the only
   contract. A small result root should have explicit artifacts and a manifest.

2. Versions are payload-level facts.

   Atlas has an explicit version block:

   - `datapack_version`
   - `atlas_version`
   - `hierarchy_version`
   - `layout_version`
   - module-specific versions
   - `created_at_utc`

   SciScape should introduce a small `sciscape_versions.json` or embed the same
   block inside `report/data.json`. Web/API/viewer code should be able to show
   and validate this without guessing from path names.

3. API responses echo data provenance.

   Atlas response types include `versions`, evidence source names, feature
   availability, and typed optional sections. SciScape should do the same in a
   lighter form: `data.json`, local web job results, and term network responses
   should expose the artifact version and available features.

4. Optional modules are advertised, not assumed.

   Atlas `/versions` returns a `features` block. This prevents docs and UI from
   claiming that a sidecar is live when it is not attached. SciScape needs the
   same idea for:

   - `has_keywords`
   - `has_membership`
   - `has_edges`
   - `has_report_data`
   - `has_term_network`
   - `has_cooccurrence_evidence`
   - `has_temporal`
   - `has_hierarchy`

5. QA is a first-class artifact.

   Atlas packages keep `qa/package_qa.json`, count reconciliation, and smoke
   records with the datapack. SciScape should generate a compact
   `qa/artifact_contract.json` for report roots and demo outputs. The release
   gate can then validate a real artifact rather than only checking functions.

6. Document-level provenance is preserved.

   Atlas treats work membership as a provenance anchor. SciScape's equivalent is
   the trio `abstracts.parquet`, `membership.parquet`, and `edges.parquet`.
   Demo/report bundles should not be considered complete if they only preserve
   labels and omit the data needed to interpret them.

7. Smoke tests target the contract, not the visuals.

   Atlas has Playwright smoke for UI behavior, but the stronger package pattern
   is that it tests stable payloads and route contracts. Since SciScape UI will
   be rewritten, current work should focus on contract-level smoke:

   - manifest loads
   - expected files exist
   - schemas contain required columns
   - counts reconcile
   - local demo can be opened
   - cluster network and term network endpoints return non-empty payloads

## What Not To Copy Directly

- Do not copy Atlas' large sidecar architecture into SciScape now.
- Do not add DuckDB/SQLite package marts unless a specific workflow needs them.
- Do not hard-code domain-specific levels like Domain/Macro/Meso/Micro/Nano into
  SciScape's public contract. SciScape should accept generic `cluster_*` levels
  and describe whichever levels exist.
- Do not depend on the current SciScape web UI. The UI is expected to be
  replaced.
- Do not add a large checked-in full demo datapack. Keep full OpenAlex demos
  reproducible and ignored; add only a small offline demo if needed.

## SciScape Adaptation

### Minimal Result Root

For a complete local SciScape result:

```text
<result_root>/
├── MANIFEST.json
├── abstracts.parquet
├── edges.parquet
└── landscape/
    ├── membership.parquet
    ├── keywords.parquet
    ├── sciscape_versions.json
    ├── qa/
    │   └── artifact_contract.json
    └── report/
        ├── data.json
        ├── index.html
        └── report.html
```

The manifest can be optional for old outputs but should be generated by new
`landscape`, `visualize`, and demo workflows.

### Version Block

Recommended minimal version block:

```json
{
  "schema_version": "sciscape_result_manifest_v1",
  "sciscape_version": "0.2.0",
  "result_version": "sciscape_result_<timestamp_or_slug>",
  "created_at_utc": "2026-05-30T00:00:00+00:00",
  "pipeline": {
    "source": "sciscape landscape",
    "cluster_backend": "rust_leiden",
    "keyword_backend": "python_or_rust_text"
  }
}
```

### Feature Block

Recommended feature block:

```json
{
  "features": {
    "keywords": true,
    "membership": true,
    "edges": true,
    "report_data": true,
    "term_network": true,
    "cooccurrence_evidence": true,
    "hierarchy": false,
    "temporal": false
  }
}
```

### Contract Validator

Add one validator entrypoint that can validate either a result root or a report
data file:

```bash
uv run --extra dev python scripts/sciscape_quality_gate.py \
  --artifact-root workspace/examples_output/openalex_live/perovskite_solar_cells_2020_2024
```

Initial validator checks:

- `abstracts.parquet`: `uid`, `title`, `abstract`, `pubyear`
- edge table: `uid1`, `uid2`, one numeric weight column
- `membership.parquet`: `uid`, at least one `cluster` or `cluster_*` column
- `keywords.parquet`: `cluster_id`, `term`, score/frequency columns when present
- `report/data.json`: parseable JSON, non-empty cluster list, schema/version block
- feature flags match available files and columns

### Release Gate Mapping

Current release gate already has:

- keyword artifact filtering smoke
- term co-occurrence smoke
- dashboard generation smoke
- web demo launcher smoke

Next release gate layer should add:

- synthetic result-root manifest generation
- artifact contract validation
- optional validation of existing live demo outputs with strict artifact schemas

## Recommended Absorption Sequence

1. Contract first.

   Maintain `docs/developer/artifact_contract.md` and `sciscape/artifacts.py`
   as the source of truth for required column definitions, feature inference,
   result-root validation, and compact manifest writing.

2. Atlas result payload.

   Update `export_report` and local web job outputs to embed a version block,
   feature block, and minimal cluster-node schema in `data.json`:

   - stable cluster UID
   - generic level name and display label
   - cluster ID
   - parent UID
   - document count
   - child count
   - optional x/y layout
   - keyword summary
   - QA badge summary

3. Contract validator.

   Keep `--artifact-root` in `scripts/sciscape_quality_gate.py` wired to the
   same validator, with tests for valid and invalid synthetic result roots. The
   validator should decide which lenses can be enabled rather than leaving that
   decision to UI code.

4. Web shell v1.

   Build the first Atlas Map shell around loaded result data only:

   - neutral loaded-result first screen
   - result-source status
   - layer visibility controls
   - map/cluster lens
   - selected-cluster inspector
   - lineage strip
   - grouped search/results rail when search payload exists

5. Evidence v1.

   Add result evidence and neighbor summaries using already available data:

   - result evidence drawer from search rows
   - representative terms
   - representative works only when membership joins to records
   - neighbor block only when cluster-neighbor edges exist

6. Advanced lenses.

   Add Quality, Growth, Boundary, Evolution, Terrain, Graph, and Narrative only
   when their artifacts and QA checks are present.

This gives SciScape the Atlas-style cluster-reading workflow without inheriting
the Atlas app's scale, fixed hierarchy, or dataset-specific assumptions.
