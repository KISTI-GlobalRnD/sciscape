# Materialization

Target location for cache, membership, prepare, join, and materialization
scripts.

- `materialize_leiden_basin_methodology_v0_panel.py`: builds the
  precommitted non-field34 methodology-v0 panel from existing basin-existence
  and calibration artifacts. It does not run routes or inspect quality/cost.
- `enrich_leiden_basin_methodology_v0_evidence.py`: joins endpoint identity,
  signature, distance, source, and cache-availability evidence onto the
  methodology-v0 panel. It does not load memberships, run routes, or inspect
  quality/cost.
- `review_leiden_basin_methodology_v0_wall_pathway_schema.py`: reviews the M2
  pair evidence against the v0 wall/pathway evidence schema using existing
  current-review and blocker ledgers. It does not execute routes, promote wall
  claims, or inspect quality/cost.
