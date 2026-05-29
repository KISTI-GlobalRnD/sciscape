# Agent Notes (SciScape)

## Source Of Truth
- Primary implementation lives under `sciscape/`.
- `sos/` is a compatibility shim only.

## Research Orientation
- Before triaging research directions or moving artifacts, read
  `research/PROJECT_TRACKS.md`, `research/DATA_RETENTION_PLAN.md`, and
  `research/FAILED_DIRECTIONS.md`.
- Treat `research/PROJECT_TRACKS.md` as the current map for the three active
  research tracks.
- Treat `research/FAILED_DIRECTIONS.md` as a guardrail: do not rerun archived
  negative-control directions unless their revisit condition is satisfied.
- Treat `research/DATA_RETENTION_PLAN.md` as documentation-only until a
  reviewed manifest exists; do not move or delete result artifacts just because
  the plan marks them as archive candidates.

## Testing
- Run `pytest -q` at repo root.

## Style
- Keep changes minimal and consistent with existing patterns.
- Avoid adding heavy dependencies; use optional extras when needed.

## Research Guardrails
- Treat `Dongdaemun` as a family name. For concrete research claims, follow
  `docs/research/dongdaemun/core/dongdaemun_naming_contract.md` and distinguish `Dongdaemun-post`,
  `Dongdaemun-refinement`, and diagnostic-only artifacts.
- For Leiden/Dongdaemun adaptive-refinement work, keep the two objectives
  separate: find a better partition, and find it with less cost.
- Do not treat positive `delta_q` alone as success. Report material gain and
  cost-adjusted gain with wall time, p5 evaluations, and memory HWM.
- Avoid more policy or threshold sweeps unless they answer a mechanism
  question, such as basin diversity, greedy failure, near-tie moves, or first
  trajectory divergence.
- Treat claims like "large or dense graphs have more basins" as hypotheses until
  supported by membership-level or signature-level basin evidence.
- Prefer instrumentation and reproducible artifacts before promoting a
  heuristic into a default policy.
