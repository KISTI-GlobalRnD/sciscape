# Consensus as a Boundary Signal: Journal of Informetrics Manuscript Outline

## Target Journal

- Primary target: `Journal of Informetrics`
- Backup target: `Scientometrics`
- Default article type: original research article

## Working Title

Consensus as a Boundary Signal for Local Neighborhood Quality in Multi-Layer Scientific Paper Graphs

## Audience And Framing

This manuscript should read as an informetrics paper, not as a generic graph
mining paper.

- Primary audience:
  - researchers in informetrics, scientometrics, and science mapping
  - readers who care about bibliometric delineation quality and evaluation
- Main framing:
  - the evaluation target is `local neighborhood correctness`
  - the method claim is `boundary-sensitive combination`, not universal
    clustering superiority
- Supporting framing:
  - AMI/NMI and cluster-count summaries are structural context only
  - the main evidence comes from order-balanced local review, taxonomy, and
    uncertainty summaries

## Core Claim

`consensus_all` should be framed as a boundary-sensitive local neighborhood
rule, not as a universally superior clustering recipe.

For the revised fairness-aware package, `citation_consensus` should be treated
as the primary cross-field focal method. The existing `consensus_all` package
remains a legacy local-review package and should not be pooled with the new
protocol-aware results without an explicit bridge analysis.

The current corrected evidence supports three paper-level claims:

1. At fixed rank budget, `consensus_all` often produces more semantically
   correct target-centered local neighborhoods than `sum` baselines.
2. This advantage is regime-dependent: it is strongest when local neighborhoods
   differ sharply and overlap weakly.
3. In low-separation settings, many comparisons collapse into ties under
   order-balanced review, which is itself evidence of boundary ambiguity rather
   than clean method superiority.

## Current Canonical Evidence

Use only the corrected order-balanced `gemini_v3` review files under
`research/consensus/results/case_banks_corrected/`.

### Main slices

- `field_15 k=6`
  - review file:
    `field_15_k06_sum_minus_emb_vs_consensus_bank_n48_order_balanced_gemini_v3_rank_shift_review.json`
  - counts: `12 / 19 / 17` (`sum_minus_emb / consensus_all / tie`)
  - no-tie consensus win rate: `19 / 31 = 0.6129`
- `field_15 k=30`
  - review file:
    `field_15_k30_sum_minus_cc_vs_consensus_bank_n48_order_balanced_gemini_v3_rank_shift_review.json`
  - counts: `10 / 31 / 7`
  - no-tie consensus win rate: `31 / 41 = 0.7561`
- `field_12 k=6`
  - review file:
    `field_12_k06_sum_minus_emb_vs_consensus_bank_n48_order_balanced_gemini_v3_rank_shift_review.json`
  - counts: `3 / 33 / 12`
  - no-tie consensus win rate: `33 / 36 = 0.9167`

### Aggregated corrected summary

- uncertainty file: `results/taxonomy_corrected/review_uncertainty_gemini_v3.json`
- overall non-tie pool: `108`
- consensus wins: `83`
- baseline wins: `25`
- overall consensus win rate: `0.7685`
- Wilson 95% interval: `[0.6806, 0.8380]`

### Taxonomy summary

- taxonomy summary: `results/taxonomy_corrected/taxonomy_combined.json`
- taxonomy report: `results/taxonomy_corrected/taxonomy_report_gemini_v3.md`
- classified non-tie cases: `108`
- skipped ties: `36`
- dominant consensus-win labels:
  - `broad_context_noise`: `35`
  - `material_family_coherence`: `29`
  - `method_family_coherence`: `19`
- dominant baseline-win failure label:
  - `over_regularized_consensus`: `6`

### Regime summary

- regime file: `results/taxonomy_corrected/consensus_regime_model_gemini_v3.json`
- descriptive direction:
  - positive for consensus: `shift_score`, `mean_abs_rank_shift`,
    `cluster_size_ratio`
  - negative for consensus: `rank_jaccard`, `cluster_overlap_coeff`,
    `log_baseline_cluster_size`
- use this as descriptive support, not as the paper's main contribution

## Section Plan

### 1. Introduction

- Problem:
  multi-layer graph combination is usually justified with global clustering
  metrics, but those do not directly tell us whether a target paper gets the
  right local research neighborhood.
- Gap:
  AMI/NMI-style agreement is not semantic ground truth for local neighborhood
  quality.
- Thesis:
  `consensus_all` is useful because it suppresses broad bridge edges and
  improves boundary precision in target-centered local neighborhoods.
- Contributions:
  - an order-balanced local neighborhood review protocol
  - empirical evidence that consensus wins in high-shift regimes
  - a descriptive taxonomy of why consensus wins and when baseline wins remain
    legitimate

### 2. Method

- Graph inputs:
  - `bc_cosine`
  - `cc_cosine`
  - `dc_fractional`
  - `emb_knn`
- Combination methods:
  - `sum`
  - `consensus`
  - leave-one-out baselines `sum_minus_cc`, `sum_minus_emb`
- Baseline selection rule:
  each reviewed slice uses the strongest leave-one-out sum comparator retained
  for that field and `top-k` setting, so baseline labels differ across slices
  by design rather than by post-hoc cherry-picking.
- Evaluation protocol:
  - sample rank-shifted targets under fixed case banks
  - compare top-ranked local neighborhoods, not full partitions
  - use order-balanced dual-pass judging
  - preserve explicit `A/B/TIE` outputs
  - separate three fairness questions:
    - Protocol A: practical `top_k=30` per layer
    - Protocol B: `effective_k` as pre-fusion candidate-budget matching
    - Protocol C: edge-count-matched first-order clustering normalization

### 3. Results

#### 3.1 Structural Context

- Same-budget comparison and k-sweep show that consensus is not uniformly best,
  but behaves differently from single-layer and sum baselines.
- Protocol C should be described as edge-count-matched clustering comparison,
  not as complete clustering-load equivalence.
- Counter-regimes such as `field_30` should be stated explicitly:
  `cc_cosine_only` can remain better on global stability/AMI while
  `citation_consensus` still improves some local reranking decisions.
- The reviewed baseline is slice-specific because the most competitive
  non-consensus leave-one-out sum variant differs across field and rank-budget
  settings.
- Low-k settings should be presented as probing regimes, not as the universal
  deployment setting.
- Top-30 remains the practical structural default.

#### 3.2 Main Local Review Result

- `field_15 k=6` is ambiguity-heavy but still consensus-leaning after tie
  removal.
- `field_15 k=30` shows a clear consensus advantage.
- `field_12 k=6` shows a very strong consensus advantage.
- The main sentence to carry through the section:
  consensus is not universally dominant, but it is frequently superior once the
  comparison is restricted to stable non-tie cases.

#### 3.3 Mechanism Interpretation

- Consensus typically wins by removing broad context noise or preserving
  material-family or method-family coherence.
- Baseline wins are rarer and usually reflect over-regularized consensus or
  strong single-cue specificity.
- Tie-heavy slices should be described as ambiguity regimes, not as null
  results.

#### 3.4 Robustness And Regimes

- Uncertainty intervals keep the main effect conservative.
- Regime analysis should be described as descriptive support:
  consensus advantage grows when local neighborhoods diverge more strongly and
  overlap less.

### 4. Discussion

- Strong claim to make:
  consensus is a boundary-sensitive reranking rule for science mapping.
- Claim to avoid:
  consensus always improves clustering.
- Main limitation:
  current validation is order-balanced LLM review, not human annotation.
- Scope limitation:
  the corrected local-review evidence is concentrated in two fields and three
  canonical reviewed slices, so the paper should avoid claiming exhaustive
  domain coverage.
- Matching limitation:
  edge-count matching is only a first-order density normalization; higher-order
  topology, degree distribution, and weight concentration differences remain.
- Main defense:
  order balancing, tie preservation, corrected winner logic, and uncertainty
  intervals make the evaluation materially more conservative than the earlier
  single-pass setup.

## Figure And Table Map

### Main figures

- Figure 1: protocol overview
  - `figures/manuscript_joi_v1/figure1_protocol_overview.svg`
- Figure 2: local review outcomes by slice
  - `figures/manuscript_joi_v1/figure2_local_review_panels.png`
- Figure 3: uncertainty summary
  - `figures/manuscript_joi_v1/figure3_review_uncertainty.png`
- Figure 4: taxonomy summary
  - `figures/manuscript_joi_v1/figure4_taxonomy_summary.png`
- Figure 5: regime support
  - `figures/manuscript_joi_v1/figure5_regime_coefficients.png`

### Main tables

- Table 1: reviewed experimental slices
  - field, top-k, baseline, compared methods, reviewed cases, non-tie cases
- Table 2: main outcome summary
  - per-slice baseline / consensus / tie and no-tie win rate
- Table 3: taxonomy summary
  - label counts and dominant winner by label
- Table 4: representative qualitative cases
  - at least two consensus wins, two baseline wins, and one ambiguity case

## Writing Defaults

- Use `consensus_all` as the focal method name in the paper, but explain in the
  method section that it is the all-layer consensus combination.
- Use `local neighborhood correctness` rather than `cluster quality` as the
  main evaluation term.
- Mention AMI/NMI only as supporting structural context, not as the main
  validation target.
- Treat the manuscript as an informetrics contribution first and a graph
  combination paper second.
- Keep `dendrogram` work out of this manuscript.

## Non-Code Deliverables

- [ABSTRACT.md](./ABSTRACT.md)
- [RESULTS_NOTES.md](./RESULTS_NOTES.md)
- [CAPTIONS.md](./CAPTIONS.md)
- [SUBMISSION_STRATEGY.md](./SUBMISSION_STRATEGY.md)
- `figures/manuscript_joi_v1/`
