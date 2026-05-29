# SciScape Local Workspace

This directory is for local-only inputs, cache files, generated runs, and
temporary web jobs. It is ignored by git except for this README.

Recommended layout:

```text
workspace/
├── data/              # local input/cache tables
├── output/            # ad hoc landscape and clustering runs
├── examples_output/   # generated example/demo runs
├── reports/           # local generated reports
└── web_output/        # FastAPI query job outputs
```

Curated, shareable demo files should be promoted explicitly into `viewer/` or
documented release artifacts. Package code belongs under `sciscape/`.
