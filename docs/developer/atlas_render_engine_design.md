# Atlas Render Engine Design

This document defines the first renderer-oriented contract for the SciScape
Atlas App. It is the bridge between validated analysis artifacts and a future
deck.gl map surface.

## Decision

SciScape should treat deck.gl as the primary Atlas map rendering engine, not as
the analysis engine.

The backend and artifact layer remain responsible for:

- clustering, hierarchy, keywords, labels, and evidence;
- map/layout coordinates and layout QA;
- term co-occurrence and relation evidence;
- temporal and cluster evolution identity matching;
- narrative claim and evidence validation.

The renderer is responsible for:

- fast canvas/WebGL rendering;
- layer visibility, highlighting, picking, and hover state;
- camera, zoom, viewport, and minimap behavior;
- semantic label visibility and level-of-detail display.

## Engine Choice

| Engine | Role | Decision |
|---|---|---|
| deck.gl | Layered Atlas map rendering | Primary target |
| Cosmograph/cosmos.gl | Very large force graph exploration | Optional specialized graph mode |
| Sigma.js | Network-specific graph viewer | Optional export/viewer target |
| Cytoscape.js | Graph theory and biological/knowledge graph workflows | Not the main Atlas engine |
| ECharts | Charts and dashboards | Keep as auxiliary, not Atlas core |

deck.gl is the best fit for the Atlas map because the product needs layered
spatial analytics: cluster territories, node layers, edge layers, term
co-occurrence overlays, labels, temporal/evolution overlays, and evidence-linked
selection. Cosmograph may be stronger for browser-side force simulation of very
large raw graphs, but the Atlas should not depend on browser-side force layout
for its main scientific map.

## Payload Split

SciScape now distinguishes two payloads:

| Payload | Purpose |
|---|---|
| `sciscape_atlas_payload_v1` | Semantic cluster payload with evidence, lineage, neighbors, and works |
| `sciscape_atlas_render_payload_v1` | Renderer payload with coordinates, layer rows, camera hints, and deck.gl layer recommendations |

The semantic payload is allowed to be rich and evidence-heavy. The render payload
must stay narrow and stable enough for a GPU renderer to consume without carrying
the full evidence surface into every layer row.

## Render Payload Contract

Endpoint:

```text
GET /api/jobs/{job_id}/atlas-render
```

Top-level fields:

| Field | Meaning |
|---|---|
| `schema_version` | `sciscape_atlas_render_payload_v1` |
| `source_schema_version` | Atlas semantic payload schema |
| `engine_family` | Target engine family, default `deck.gl` |
| `view` | Orthographic 2D view hints and bounds |
| `levels` | Ordered hierarchy levels |
| `layers` | Renderer layer row groups |
| `node_count`, `edge_count`, `label_count` | Layer row counts |
| `warnings` | Semantic and render warnings |

Layer groups:

| Layer key | Recommended deck.gl layer | Meaning |
|---|---|---|
| `nodes` | `ScatterplotLayer` | Cluster points or territories |
| `edges` | `LineLayer` | Cluster relation and co-occurrence edges |
| `labels` | `TextLayer` | Semantic labels with priority and zoom hints |
| `hierarchy` | `LineLayer` | Parent-child relation scaffolding |

## Coordinate Policy

The render payload preserves `x` and `y` from the semantic Atlas payload when
present. If a node lacks coordinates, the adapter emits deterministic fallback
positions and marks:

```text
coordinate_source = generated | mixed | node_coordinates
```

Fallback positions are for smoke rendering only. A production Atlas map should
prefer a validated layout artifact with explicit coordinates, layout parameters,
and QA.

## Implementation Order

1. Keep existing D3/Plotly surfaces unchanged.
2. Add `AtlasRenderPayload` adapter and endpoint.
3. Build a deck.gl prototype against `/api/jobs/{job_id}/atlas-render`.
4. Add visual/performance smoke checks for sample result roots.
5. Promote deck.gl to the primary Atlas map only after the prototype passes
   node/edge count, label visibility, selection sync, and mobile rendering gates.

Current implementation status:

- Steps 1-4 are implemented for a tiny smoke surface.
- The prototype is intentionally guarded: if deck.gl or `/atlas-render` is not
  available, the existing card-based Atlas view remains visible.
- The static prototype includes layer visibility controls, edge-weight
  thresholding, label-density control, URL-state persistence, and selected-node
  view centering.
- `scripts/sciscape_quality_gate.py --p1-atlas-smoke` validates the
  `/atlas-render` contract while reopening a generated result through the web
  API.
- `scripts/sciscape_quality_gate.py --atlas-visual-smoke` uses headless Chrome
  to render a tiny deck.gl map and checks that the screenshot is not blank.
- `scripts/sciscape_quality_gate.py --atlas-render-perf-smoke` builds a
  deterministic 100-node/500-edge render payload and validates counts, layer
  recommendations, coordinate policy, payload size, and construction time
  without requiring a browser.
- `scripts/sciscape_quality_gate.py --atlas-render-scale-smoke` builds a
  deterministic 5k-node/25k-edge render payload and validates the same
  payload-level contract for small demo scale without requiring a browser.
- `scripts/sciscape_quality_gate.py --atlas-interaction-smoke` uses headless
  Chrome to render the 5k-node/25k-edge map, update the camera to a selected
  node, run a deck.gl hit-test at the viewport center, and verify a nonblank
  screenshot. This remains optional because it depends on a local browser and
  the deck.gl CDN.
- Browser interaction gates beyond small-demo scale remain future work.

## Performance Gates

The deck.gl prototype should be tested on at least:

| Gate | Nodes | Edges | Expected outcome |
|---|---:|---:|---|
| CI smoke | 100 | 500 | deterministic payload contract, bounded size/time |
| Small demo | 5,000 | 25,000 | deterministic payload contract plus optional browser interaction smoke |
| Analyst scale | 50,000 | 250,000 | usable LOD, edge filtering required |
| Stress | 200,000 | 1,000,000 | progressive loading or server-side thinning required |

The CI and small-demo smokes are intentionally payload-level so they can run in
local release gates and test suites without a browser, GPU, or CDN. The browser
visual and interaction smokes remain optional companion checks for deck.gl
rendering.

The app should prefer layer visibility toggles over layer removal for frequent
switching, and should not rebuild large layer data on every hover or filter
change.

## Open Questions

- Should layout coordinates become a persisted artifact separate from the Atlas
  semantic payload?
- Should evolution transitions use the same 2D coordinate basis as cluster
  nodes or a dedicated time-slice small-multiple view?
- Should term co-occurrence use deck.gl edge layers in the Atlas map or a
  Cosmograph-style specialized graph mode for dense term networks?
