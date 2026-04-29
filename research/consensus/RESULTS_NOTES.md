# Results Notes

## Headline Numbers

- Canonical review slices use only corrected order-balanced `gemini_v3` files.
- Slice outcomes are reported as `baseline / consensus / tie`.
- `field_15 k=6`: `12 / 19 / 17`
- `field_15 k=30`: `10 / 31 / 7`
- `field_12 k=6`: `3 / 33 / 12`
- Overall non-tie pool: `108`
- Overall `consensus_all` wins: `83`
- Overall baseline wins: `25`
- Overall non-tie consensus win rate: `0.7685`
- Wilson 95% interval: `[0.6806, 0.8380]`

## Protocol D v1 Boundary-Accuracy Note

- `Protocol D v1` is a direct boundary-adjudication check, but it is not an
  overall boundary-accuracy estimate.
- Its sampler is intersection-heavy and disagreement-focused: cases enter only
  after both methods form reviewable groups and those groups differ enough to
  compare.
- High `NEITHER` rates mean the summary mixes true head-to-head boundary
  decisions with pathological cases where neither local neighborhood is good.
- Use `Protocol D v1` as a baseline diagnostic only; coverage-aware `Protocol D
  v2` is required before claiming overall boundary accuracy.

## Protocol D v2 Coverage-Aware Pilot Note

- Current aggregate:
  `results/cross_field_round2/boundary_coverage_v2_sample_summary.json`
- Completed pilot comparison:
  `cc-only vs citation_consensus` for `field_12`, `field_15`, and
  `field_30_textfilt` at `effective_k=30`.
- Positive net gaps favor `cc-only`; negative net gaps favor
  `citation_consensus`.

| Slice | Population coverage `cc/citation` | Population full net gap | Diagnostic full net gap | Diagnostic `NEITHER` rate |
|---|---:|---:|---:|---:|
| `field_12 ek30` | `0.5667 / 0.5333` | `+0.3333` | `+0.1935` | `0.5161` |
| `field_15 ek30` | `0.8000 / 0.9333` | `+0.0455` | `+0.1429` | `0.4000` |
| `field_30_textfilt ek30` | `0.7667 / 0.9000` | `-0.3000` | `-0.1579` | `0.3684` |

- Interpretation:
  D v2 confirms that boundary behavior is field/regime dependent. It supports
  coverage-aware evaluation, but it does not yet support a blanket claim that
  one method has generally more accurate boundaries.
- Remaining before primary manuscript claims:
  complete `cc-only vs all_consensus`, `bc-only vs cc-only`, and
  `emb-only vs cc-only` D v2 runs; then build a `Protocol E` micro-gold local
  partitioning pilot.

### Current-Code Sample Matrix

- After the Rust Leiden/postprocess optimization pass, fixed-gamma clustering
  can produce different cluster counts than the older reviewed D v2 payloads.
- Do not mix older reviewed `cc-only vs citation_consensus` payloads with newly
  generated `cc-only vs all_consensus` payloads as one final matrix.
- Current-code sample-only payloads are stored under
  `results/cross_field_round2/` and summarized in:
  `boundary_coverage_v2_current_sample_matrix_summary.json`.
- Current-code sample-only comparisons now prepared:
  - `cc-only vs citation_consensus`
  - `cc-only vs all_consensus`
  - for `field_12`, `field_15`, and `field_30_textfilt`
- These payloads still need live review before boundary-accuracy metrics can be
  reported from them.

## Figure Notes

### Figure 1: Protocol Overview

- Use this figure to introduce the evaluation logic before showing any result.
- The claim is procedural:
  the paper evaluates local neighborhood correctness rather than full-partition
  agreement.
- Caption should mention four inputs:
  graph layers, combination rules, rank-shift case bank, and order-balanced
  dual-pass review.

### Figure 2: Local Review Outcomes By Slice

- Main sentence:
  consensus is not universally dominant, but it is favored in every slice once
  the comparison is restricted to stable non-tie cases.
- Sub-points:
  - `field_15 k=6` is the ambiguity-heavy regime
  - `field_15 k=30` shows a clear consensus advantage
  - `field_12 k=6` shows a very strong consensus advantage
- Report no-tie rates in the text:
  - `19 / 31 = 0.6129`
  - `31 / 41 = 0.7561`
  - `33 / 36 = 0.9167`

### Figure 3: Uncertainty

- Use this as the main robustness figure.
- The claim is not that every slice is equally stable.
- The claim is that the corrected aggregate remains clearly consensus-leaning:
  `83 / 108` with Wilson 95% interval `[0.6806, 0.8380]`.
- Field-level note:
  - `field_12`: `33 / 36 = 0.9167`
  - `field_15`: `50 / 72 = 0.6944`

### Figure 4: Taxonomy Summary

- The mechanism claim should be framed positively:
  consensus tends to remove broad context noise and recover coherent local
  material or method families.
- Key counts:
  - `broad_context_noise`: `35`
  - `material_family_coherence`: `29`
  - `method_family_coherence`: `19`
- Baseline-win interpretation:
  baseline wins mostly cluster into `over_regularized_consensus` and
  `single_cue_specificity`, so they should be described as legitimate failure
  modes rather than exceptions to ignore.

### Figure 5: Regime Support

- Present this as descriptive support, not a predictive model contribution.
- Main interpretation:
  consensus advantage becomes more likely when local neighborhoods diverge more
  strongly and overlap less.
- Positive direction:
  `shift_score`, `mean_abs_rank_shift`, `cluster_size_ratio`
- Negative direction:
  `rank_jaccard`, `cluster_overlap_coeff`, `log_baseline_cluster_size`

## Table Notes

### Table 1: Reviewed Slices

- Include:
  - `field_15`, `k=6`, baseline `sum_minus_emb`, reviewed `48`, non-tie `31`
  - `field_15`, `k=30`, baseline `sum_minus_cc`, reviewed `48`, non-tie `41`
  - `field_12`, `k=6`, baseline `sum_minus_emb`, reviewed `48`, non-tie `36`
- Add one explanatory sentence in the table note or nearby text:
  the baseline label is slice-specific because each slice compares
  `consensus_all` against the strongest leave-one-out sum comparator used for
  that field and rank-budget setting, rather than forcing one fixed baseline
  across all regimes.
- Keep this table factual and compact.

### Table 2: Main Outcomes

- Include full counts and no-tie win rates for all three slices.
- Use the sentence:
  ties are treated as conservative ambiguity rather than as implicit wins for
  either method.

### Table 3: Taxonomy Summary

- Use the top eight labels already present in
  `results/taxonomy_corrected/taxonomy_combined.json`.
- Highlight that `34` of the `35` `broad_context_noise` cases favor
  `consensus_all`.

### Table 4: Representative Cases

- Recommended consensus-win cases:
  - `W2067783257`: carbon catalysts for oxidative dehydrogenation
  - `W2021514999`: ionic-liquid thermomorphism and extraction
  - `W3016217815`: corrective feedback preferences in L2 learning
- Recommended baseline-win cases:
  - `W2112002317`: zinc catalyst specificity for CO2 plus epoxide coupling
  - `W3088126537`: diboride-specific nitrogen reduction context
- Recommended ambiguity case:
  - `W2064036092`: dual-pass order-sensitive tie in `field_15 k=6`

## Discussion Notes

- Strongest safe sentence:
  consensus is a boundary-sensitive reranking rule for scientific paper graphs.
- Sentence to avoid:
  consensus is always the best clustering method.
- Limitation sentence:
  the current validation is based on order-balanced LLM review rather than
  human annotation, so the paper should claim conservative support for local
  neighborhood quality rather than absolute semantic ground truth.
- Scope sentence:
  the corrected evidence is concentrated in two fields and three canonical
  reviewed slices, so cross-field generalization should be framed as promising
  rather than exhaustive.
