# Consensus Scripts

This directory is intentionally treated as a migration target, not a stable
final layout. See `../SCRIPT_STRUCTURE.md` before moving files.

Inventory command:

```bash
uv run --extra dev python scripts/inventory_research_scripts.py --fail-on-unclassified
```

The target split is:

- `common/`
- `consensus_core/`
- `review_taxonomy/`
- `leiden_basin/`
- `dongdaemun_hierarchy/`
- `artifacts_reporting/`

`leiden_basin/` has its own second-level split because it is the dominant
script family:

- `leiden_basin/basin_signatures/`
- `leiden_basin/transition_routes/`
- `leiden_basin/operator_probes/`
- `leiden_basin/evidence_panels/`
- `leiden_basin/materialization/`
- `leiden_basin/hysteresis/`
