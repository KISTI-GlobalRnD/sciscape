# Figure And Table Captions

## Figures

### Figure 1

Protocol overview for evaluating local neighborhood quality in multi-layer
scientific paper graphs. Multiple graph layers are combined into sum-based and
consensus-based runs under a matched rank budget, high-shift targets are
sampled into a fixed case bank, and each pair of local neighborhoods is judged
with an order-balanced dual-pass review protocol that preserves explicit
`A/B/TIE` outcomes.

### Figure 2

Corrected local review outcomes for the three canonical slices, reported as
`baseline / consensus / tie`. The `field_15 k=6` slice is ambiguity-heavy, but
`consensus_all` still leads after excluding ties; `field_15 k=30` and
`field_12 k=6` show progressively stronger consensus advantages. Ties are
treated conservatively as ambiguous cases rather than latent wins for either
method.

### Figure 3

Uncertainty summary for corrected order-balanced local review outcomes. The
aggregate non-tie result remains strongly consensus-leaning (`83/108`,
`76.9%`), while slice-level intervals show that the effect is strongest in
`field_12` and more moderate but still positive in `field_15`.

### Figure 4

Descriptive taxonomy of why one local neighborhood wins over the other.
Consensus wins are dominated by broad context noise removal and family-level
coherence recovery, whereas baseline wins concentrate in over-regularized
consensus and single-cue specificity cases.

### Figure 5

Descriptive regime support for when `consensus_all` is more likely to win a
local neighborhood comparison. Consensus advantage increases with larger rank
shifts and weaker top-neighbor overlap, while large baseline clusters and high
rank Jaccard agreement are associated with smaller consensus gains.

## Tables

### Table 1

Canonical reviewed slices used in the manuscript. Each row reports the field,
the rank budget, the baseline comparator, the total reviewed bank size, and the
non-tie subset used for win-rate interpretation. The baseline comparator is
slice-specific and corresponds to the strongest leave-one-out sum variant used
for that field and rank-budget setting.

### Table 2

Main corrected review outcomes by slice. Counts are reported as
`baseline / consensus / tie`, followed by the no-tie consensus win rate used in
the main text.

### Table 3

Taxonomy summary for the corrected non-tie pool (`n=108`). Labels are reported
with total counts and winner-specific counts to show which error modes dominate
consensus wins and baseline wins.

### Table 4

Representative qualitative cases illustrating three regimes: clear
consensus-win cases, clear baseline-win cases, and order-sensitive ambiguity
cases that collapse to ties under the corrected dual-pass protocol.
