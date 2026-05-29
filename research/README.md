# SciScape Research Index

This directory contains the active research workspace and archival evidence for
SciScape. Start here before reorganizing research directions, interpreting
adaptive-refinement results, or moving result artifacts.

## Current Orientation

- `PROJECT_TRACKS.md`: current three-track research map and claim boundaries.
- `DATA_RETENTION_PLAN.md`: keep, consolidate, archive, and drop-candidate
  policy for result artifacts.
- `FAILED_DIRECTIONS.md`: failed directions, negative controls, and revisit
  conditions.

## Active Track Entrypoints

- Track A, multi-layer consensus boundary signal:
  `consensus/README.md`
- Track B, Dongdaemun-post hierarchy repair:
  `../docs/research/dongdaemun/README.md`
- Track C, adaptive refinement and basin-tunneling R&D:
  `../docs/research/leiden_basin/README.md` for the Leiden basin research doc
  index, `../docs/research/leiden_basin/evidence/leiden_basin_data_inventory.md` for
  detailed artifact chronology, and
  `consensus/results/adaptive_refinement/leiden_basin_methodology_v0_20260529/`
  for the current methodology-v0 precommitted non-field34 panel. Also see
  `consensus/results/adaptive_refinement/leiden_basin_cycle_closure_writeup_20260529/`
  for the current closure state. Under the fixed current gates, Track C has
  basin-existence candidate evidence but 0 executable route candidates and 0
  wall-promotion candidates; reopen only through the gates in
  `PROJECT_TRACKS.md`.
- Deferred CPM-critical dendrogram/optimal-cut work:
  `dendrogram/README.md`

## Guardrail

Do not move or delete result artifacts directly from this index. Use
`DATA_RETENTION_PLAN.md` and `../scripts/research_retention_manifest.py` to
create a reviewed manifest first, and check `FAILED_DIRECTIONS.md` before
spending new compute on adaptive-refinement experiments.
