# SciScape Docs

SciScape documentation is organized for bounded navigation: every human-facing
documentation folder should stay at six visible entries or fewer, excluding
`README.md`.

## Map

| Path | Purpose |
|---|---|
| `user/` | Public workflows, modules, and IO schemas |
| `developer/` | Repo structure, decisions, and release readiness |
| `research/` | Research design notes and claim boundaries |
| `papers/` | Manuscript drafts, reports, and paper figures |
| `archive/` | Historical notes kept for traceability |
| `assets/` | Images and static documentation assets |

Run the fanout check before adding new documentation groups:

```bash
uv run --extra dev python scripts/check_doc_fanout.py
```
