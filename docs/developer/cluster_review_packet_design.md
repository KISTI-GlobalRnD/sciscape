# Cluster Review Packet Artifact Design

This document defines the compact review packet that sits between validated
analysis artifacts and future narrative claims.

The packet is not a narrative generator. It is a bounded, evidence-backed
cluster reading surface that can be validated without the web app and later
converted into narrative claim/evidence rows.

## Goal

A cluster review packet must answer:

- which cluster is being reviewed;
- which label, keywords, representative works, co-occurrence links, and QA
  caveats are available;
- which evidence refs support each row;
- whether the cluster is ready for narrative scaffolding or requires review.

## Non-Goals

- Do not generate free-form narrative text.
- Do not infer unsupported mechanisms or field causality.
- Do not hide quality caveats; preserve them as evidence rows.
- Do not replace the full narrative claim graph defined in
  `narrative_artifact_design.md`.

## Canonical Files

```text
<result_root>/review/
  cluster_review_packet.json
  cluster_review_packet_qa.json
```

For landscape-scoped outputs, the packet may also live under
`<result_root>/landscape/review/`.

## Schema Versions

- `sciscape_cluster_review_packet_v1`
- `sciscape_cluster_review_packet_qa_v1`

## Packet Shape

`cluster_review_packet.json` contains:

- `packet_id`, `title`, `created_at_utc`;
- `packet_scope` limits, such as max clusters, max keywords, and max evidence
  rows;
- `review_policy`, including `narrative_generation_allowed=false`;
- `source_artifacts`;
- `clusters`;
- `qa` summary.

Each cluster row contains:

- `cluster_uid`, `cluster_level`, `cluster_id`, `label`, `doc_count`;
- `review_status`: `clean` or `review_required`;
- `narrative_ready`: boolean;
- `keyword_evidence`;
- `representative_works`;
- `cooccurrence_evidence`;
- `quality_caveats`;
- `evidence_refs`.

Every row in `keyword_evidence`, `representative_works`,
`cooccurrence_evidence`, and `quality_caveats` must reference an id in the same
cluster's `evidence_refs`.

## Validation Rules

The validator blocks the packet when:

- the packet file is missing or has the wrong schema;
- no cluster rows exist;
- any cluster lacks `cluster_uid`;
- evidence row refs are missing, duplicated, or unresolved.

The validator warns when:

- QA sidecar is missing;
- a cluster has no display label;
- not every cluster is `narrative_ready`.

## Relationship To Narrative

The review packet is the deterministic input packet for future
`write_narrative_evidence_artifacts`.

Narrative artifacts may cite packet rows, but a packet alone does not make
`narrative=stable`. The narrative feature remains hidden until a
claim/evidence artifact exists and validates.

## Implementation Status

- `[x]` `write_cluster_review_packet_artifact` writes the packet and QA sidecar.
- `[x]` `validate_cluster_review_packet_artifact` validates evidence-ref
  resolution.
- `[x]` `result_manifest.artifacts` exposes packet and QA records.
- `[x]` The packet contributes to `evidence` and `quality` artifact refs.
- `[ ]` Narrative claim graph generation is not implemented.
- `[ ]` Web review UI integration is not implemented.
