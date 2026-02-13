# Module Architecture Overview

The Leiden module is organised into reusable building blocks so you can compose
single runs, exhaustive resolution scans, and hierarchical pipelines without
duplicating code. The key goals are:

1. **Speed optimisation** by reusing graph state and the Leiden optimiser across levels.
2. **Parameter tuning** utilities to scan or search resolution values with stability metrics.
3. **Post-processing** hooks to merge clusters that fall below minimum size/weight thresholds.
4. **Hierarchical pipeline** that mirrors the CWTS multi-level process (run → merge → contract).

## Components

- `runner.LeidenRunner`
  - Holds the igraph object, cached edge weights, and a `leidenalg.Optimiser` instance.
  - Exposes `run(resolution, seed, n_iterations, objective)` plus `contract(membership)` for multi-level runs.
  - Supports warm starts via `initial_membership` and returns the membership list, quality, and partition handle.

- `postprocess.merge_small_clusters`
  - Accepts a membership vector and merges clusters whose size/weight is below configurable thresholds into the
    neighbour with the strongest connection (max total edge weight).
  - Returns the remapped membership, merge log, and summary statistics.

- `hierarchy_builder.HierarchyBuilder`
  - Orchestrates multi-level clustering: run Leiden, post-process, store metadata, contract the graph, and repeat.
  - Produces `HierarchyLayer` records with membership arrays, cluster counts, resolution, quality, and thresholds.

- `tuning.scan_resolution_grid`
  - Evaluates grids of resolution/seed combinations (sequentially or in parallel) using a shared `LeidenRunner`.
  - Calculates quality, cluster counts, and optional NMI-based stability across seeds.

## Pipelines

- `pipeline.run_pipeline`
  - Delegates the core Leiden execution to `LeidenRunner`, records post-processing merges, and assembles
    hierarchical tables for the requested levels.

- `hierarchical_pipeline.run_hierarchy_pipeline`
  - Builds a CWTS-style hierarchy on top of `HierarchyBuilder`, yielding both metadata and ready-to-use tables.

Use these composable pieces to mix single-run diagnostics, grid searches, and hierarchy construction within the
same project.
