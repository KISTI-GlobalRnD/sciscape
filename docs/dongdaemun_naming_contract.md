# Dongdaemun Naming And Claim Contract

Status: naming and claim-boundary contract for Dongdaemun research artifacts,
scripts, reports, and manuscript text.

This document keeps the Dongdaemun brand useful without letting unsupported
claims drift into the validated method. Use `Dongdaemun` as the family name, but
qualify concrete claims with one of the roles below.

## Core Terms

### Dongdaemun

`Dongdaemun` is the umbrella name for objective-preserving macro-refinement
around Leiden-style CPM hierarchy construction.

Use the bare name only when referring to the whole family or to high-level
motivation. Do not use the bare name for a result row, benchmark claim, or
accepted output unless the specific variant is also clear from context.

### Dongdaemun-post

`Dongdaemun-post` is the current manuscript-backed method.

It runs after a current-level Leiden membership is available, starts from the
lower-tail repaired baseline, proposes upper-tail macro-refinements, and commits
only memberships whose exact original-graph CPM audit is non-regressing. A
rejected candidate may be stored for diagnostics, but it is not the effective
hierarchy output.

Claims allowed under this name:

- exact CPM audit is preserved;
- accepted quality-first outputs are non-regressing relative to the audited
  baseline;
- frozen manuscript evidence supports upper-tail improvement for the validated
  postprocess policy;
- fallback is part of the method.

Claims not allowed under this name:

- strict target satisfaction is guaranteed;
- integrated Leiden-loop acceleration has been validated;
- local-gamma or basin-probe policies are validated defaults.

### Dongdaemun-refinement

`Dongdaemun-refinement` is the integrated research target.

It allocates extra parent-internal refinement effort during the Leiden
refinement stage, using size or instability as search-budget signals while
leaving CPM unchanged. It must preserve the refinement subset invariant and
let reduced-graph Leiden decide cross-parent movement under the original CPM
objective.

Until cross-sample runtime-quality evidence exists, this name means
experimental or future-work behavior. It must stay opt-in in code and reports.

### Dongdaemun diagnostics

`Dongdaemun diagnostics` are instruments used to understand when a refinement
mechanism might help. This includes basin signatures, p5 candidate labels,
approximate polish labels, trajectory divergence traces, quotient diagnostics,
hard-cap rejected rows, and semantic sanity checks.

Diagnostic artifacts may motivate candidate generation or future policy design.
They do not establish an accepted Dongdaemun output by themselves.

## What Is Not The Main Dongdaemun Claim

The following may be useful, but should not be framed as the main Dongdaemun
contribution:

- raw Leiden optimization;
- lower-tail support repair alone;
- semantic coherence scoring;
- memory optimization of probe infrastructure;
- threshold, tau, or source sweeps without a mechanism question;
- positive but low-ROI `delta_q` changes;
- candidate ranking changes that do not alter the search transition rule or
  prove material basin coverage.

## Artifact Naming

Use names that disclose the role of the artifact.

- Use `dongdaemun_post_*` for validated postprocess evidence and reproductions.
- Use `dongdaemun_refinement_*` for opt-in integrated-loop prototypes.
- Use `dongdaemun_basin_*` only for basin evidence intended to support
  Dongdaemun-refinement mechanism claims.
- Use `leiden_multibasin_*` when the experiment is still a generic Leiden
  basin or p5 candidate-ranking study.
- Use `*_diagnostic_*` for artifacts that should not be read as accepted
  effective outputs.

Recommended metadata fields for new result tables:

- `dongdaemun_family`: `post`, `refinement`, `diagnostic`, or `none`;
- `dongdaemun_claim_level`: `supported`, `experimental`, `diagnostic`, or
  `hypothesis`;
- `effective_output`: boolean flag for memberships passed forward;
- `audit_delta_q`: exact original-graph CPM delta when an output is accepted;
- `material_gain`: boolean or threshold label, not just `delta_q > 0`;
- `cost_basis`: wall time, p5 count, memory HWM, or another explicit cost axis.

## Acceptance Language

Use these words consistently.

- `accepted`: an audited membership is allowed to become the effective output.
- `committed`: the effective hierarchy actually receives that accepted output.
- `selected`: a policy chose a candidate for evaluation or replay.
- `candidate`: a proposed membership or local move that may still be rejected.
- `diagnostic`: useful for explanation or future design, not an effective
  output.

Do not call a diagnostic row accepted because it has positive local, proxy, or
approximate score. Acceptance requires the variant-specific exact audit.

## Review Checklist

Before promoting a new result under the Dongdaemun name, answer these checks:

1. Is this `post`, `refinement`, or `diagnostic`?
2. Did the effective output pass exact original-graph CPM audit?
3. Is the gain material relative to graph scale and operating cost?
4. Does the result support a better partition, a faster search, or both?
5. Is there membership-level or signature-level evidence if the claim mentions
   basin diversity?
6. Is the result cross-sample evidence, a single fixture, or only a smoke test?

If any answer is unclear, keep the artifact diagnostic and avoid using it as a
Dongdaemun algorithm claim.
