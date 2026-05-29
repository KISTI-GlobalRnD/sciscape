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
