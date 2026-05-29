# Consensus Research Script Structure

This note is the migration guardrail for `research/consensus/scripts/`.
The current directory is too large to navigate by filename alone, but many
research notes, tests, and saved artifact metadata still reference exact script
paths. Move scripts only after inventorying those references.

## Target Buckets

Use at most six top-level script buckets when this directory is split:

| Bucket | Intended content |
|---|---|
| `common/` | Shared helper modules such as `_common.py` |
| `consensus_core/` | Baseline multi-layer consensus experiments and core probes |
| `review_taxonomy/` | Boundary review, rank-shift review, taxonomy, and review scoring |
| `leiden_basin/` | Leiden basin, hysteresis, wall, route, and adaptive basin scripts |
| `dongdaemun_hierarchy/` | Dongdaemun and hierarchy-postprocess research scripts |
| `artifacts_reporting/` | Freeze, materialize, summarize, export, score, and figure utilities |

Do not create a seventh top-level bucket unless an existing bucket would become
semantically misleading. Prefer a second-level split inside the bucket instead.

## Leiden Basin Second-Level Buckets

`leiden_basin/` is the largest target bucket, so it must be split further:

| Bucket | Intended content |
|---|---|
| `leiden_basin/basin_signatures/` | Basin signatures, selector contracts, mode tradeoff, and general basin diagnostics |
| `leiden_basin/transition_routes/` | Transition, route, wall-route, pathway, tunneling, and direct-pair scripts |
| `leiden_basin/operator_probes/` | Attachment-margin, aligned-core, handle, selector, gate, recovery, and polish probes |
| `leiden_basin/evidence_panels/` | Reviews, audits, field eligibility, relation taxonomy, phase panels, and claim evidence |
| `leiden_basin/materialization/` | Cache, membership, prepare, join, and materialization scripts |
| `leiden_basin/hysteresis/` | Leiden hysteresis runs, monitors, and graph materialization |

The inventory command computes the target path for each script. Use that output
as the move manifest rather than hand-sorting files.

## Migration Rule

Before moving any script, run:

```bash
uv run --extra dev python scripts/inventory_research_scripts.py --fail-on-unclassified
```

Then inspect the most-referenced scripts:

```bash
uv run --extra dev python scripts/inventory_research_scripts.py --format markdown
```

Safe movement requires:

1. no `unclassified` scripts;
2. a replacement plan for every hard-coded `research/consensus/scripts/*.py`
   reference in docs, tests, and research notes;
3. compatibility wrappers or updated tests for scripts that are directly loaded
   by path;
4. no movement of uncommitted research work unless that work is intentionally
   included in the same commit.

## Recommended Phases

1. Add thin compatibility wrappers for high-reference scripts.
2. Move low-reference artifact/reporting utilities first.
3. Move domain-specific Leiden basin and Dongdaemun scripts after their active
   research notes are updated.
4. Move core E1-E7 consensus scripts last, because `research/consensus/README.md`
   documents their commands directly.
