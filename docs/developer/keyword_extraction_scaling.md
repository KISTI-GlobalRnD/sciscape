# Keyword Extraction Scaling Notes

Status: production-candidate for capped Atlas-scale extraction
Date: 2026-05-31

This note records the current large-scale behavior of SciScape keyword
extraction and the remaining work before claiming full-corpus scale support.

## What Is Now Supported

The keyword extraction pipeline now supports long-running clustered extraction
with operator-visible progress and restartable scoring shards.

Implemented safeguards:

- configurable cluster-task backend: `auto`, `loky`, `threading`, or
  `sequential`;
- progress JSON updates via `progress_path`;
- scoring shard output via `scoring_shard_dir`;
- shard resume using a fingerprint that includes scoring inputs, config, and
  `METADATA_ARTIFACT_FILTER_VERSION`;
- result-manifest detection of cluster-sharded `manifest.json`,
  `progress.json`, preflight/run summaries, candidate shard progress/done
  sidecars, and final shard done sidecars, including failed shard IDs, partial
  outputs, and resume markers;
- metadata artifact filtering for HTML, LaTeX preamble residue, publisher page
  fragments, Crossref/PubMed/Scopus/Google source residue, article metrics
  fragments, and suffix `abstract` fragments;
- materialized Atlas term deduplication by displayed term before rank
  assignment.

These changes are intended to prevent a long run from failing silently or
restarting without reusable intermediate output.

## Current Scale Evidence

Validated on the Atlas Domain8/Micro3773 representative text run:

| Level | Input rows | Clusters | Cap | Runtime | Output rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| Domain | 8,000 | 8 | 1,000 docs/cluster | 29 s | 400 |
| Macro | 35,000 | 35 | 1,000 docs/cluster | 124 s | 1,750 |
| Meso | 127,500 | 255 | 500 docs/cluster | 569 s | 12,746 |
| Micro | 375,256 | 3,773 | 100 docs/cluster | 2,294 s | 188,614 |

Final materialized Atlas terms:

| Level | Rows | Clusters |
| --- | ---: | ---: |
| Domain | 400 | 8 |
| Macro | 1,748 | 35 |
| Meso | 12,667 | 255 |
| Micro | 187,275 | 3,773 |

Final QC:

- legacy HTML/LaTeX/page metadata rows: 0;
- legacy HTML/LaTeX/page metadata top10 rows: 0;
- semantic metadata fragment rows: 0;
- semantic metadata fragment top10 rows: 0;
- display stop unigram rows: 0;
- blank rows: 0;
- comma/blob rows: 0;
- rows longer than 80 chars: 0;
- duplicate normalized terms within cluster: 0.

QC artifacts:

- `qa/semantic_keyword_filter_qc_20260531.md`
- `qa/semantic_keyword_filter_qc_20260531.json`

## Claim Boundary

SciScape keyword extraction is ready for capped, cluster-oriented Atlas term
generation at several thousand clusters.

It is not yet proven as an unrestricted full-corpus keyword engine for millions
or tens of millions of full-text rows without sampling or per-cluster caps.

## Cluster-Sharded V2 Direction

The opt-in `keyword_engine="cluster_sharded"` path is the intended scaling
direction for very large Nano-level runs such as tens of millions of documents
split across tens of thousands of clusters.

Design contract:

- source documents are first materialized into document shards by cluster
  shard, so candidate mining workers do not repeatedly scan the full source
  parquet;
- each cluster receives an adaptive candidate-pool cap rather than a fixed
  keyword count;
- candidate mining writes long-form `cluster_id, term, local_tf, local_doc_df`
  rows instead of materializing a dense or full sparse cluster x term matrix;
- candidate mining also writes per-shard parenthetical abbreviation evidence
  when abbreviation dictionaries are enabled;
- global corpus evidence is reduced into a single `global_term_stats.parquet`
  table containing term-level counts, cluster document frequency, entropy, and
  artifact risk;
- global abbreviation evidence is reduced into a cluster/global lookup and fed
  into the quality refinement stage;
- final c-TF-IDF-like scoring is performed by streaming candidate shards,
  applying the existing quality/representative-label refinement on the bounded
  candidate pool, and keeping per-cluster top-k results only;
- with `--quality-rerank`, final selection reserves the leading slice for
  representative phrase labels and fills the rest by quality score, so unigrams
  remain available as supporting evidence instead of being forced into rank-1
  labels;
- `review_*` rows are preserved in the flagged view but hidden from the
  default clean view when a cluster has cleaner alternatives;
- quality annotation also separates raw flags from structured QA fields:
  `quality_risk_family`, `quality_flag_basis`, `quality_flag_confidence`, and
  `clean_view_action`;
- representative quality marks stopword-compressed oxidation-state fragments
  such as `ii aqueous` as `review_fragment` only when a cleaner cluster-local
  replacement exists, preserving valid labels such as `pb ii`, `type ii ...`,
  and `ii vi semiconductor`;
- representative quality also demotes a small set of broad semantic heads such
  as `high performance` only when multiple longer local replacements exist,
  preserving concrete phrases such as `high performance computing`;
- final scoring writes default clean `keywords.parquet`, inclusive top-N
  `keywords_flagged.parquet`, and
  `qa/keyword_quality_residual_report.{json,md}` so users can inspect flags
  without losing a clean display surface;
- shard-level `.done.json` files make document sharding, candidate mining, and
  final scoring restartable.

False-positive guardrail:

- no keyword-quality rule should demote, hide, or relabel a term from shape
  alone when that shape can be valid in another domain;
- every new `review_*` or demotion rule must be replacement-aware or
  evidence-aware, and must preserve the candidate when no cleaner local
  alternative exists;
- every new rule must include paired negative-control tests that keep plausible
  valid terms in the primary/support tiers before the rule can be used in a
  large run;
- residual weak labels should be reported for review before broadening a rule;
  broad rule expansion is not an acceptable substitute for evidence.

Initial guardrails:

| Budget | Target | Warning | Hard stop |
| --- | ---: | ---: | ---: |
| Cluster-term candidate rows | 50,000,000 | 80,000,000 | 100,000,000 |
| Global unique terms | 5,000,000 | 8,000,000 | 10,000,000 |

The current V2 implementation is a Python fallback contract.  It is intended to
stabilize artifact schemas and ranking behavior before moving the hot kernels
to `rust-text`.  The first Rust target should be cluster-local candidate mining:
tokenization, n-gram counting, document coverage, year summaries, and
per-channel top-candidate retention.

Before a large Nano run, execute the membership-only preflight.  It writes
`manifest.json` and `preflight_summary.json` without scanning title/abstract
text, so it is the cheapest way to verify shard count and candidate-row upper
bound:

```bash
sciscape keywords abstracts.parquet membership.parquet \
  --keyword-engine cluster_sharded \
  --keyword-preflight-only \
  --cluster-level canonical_nano_id \
  --uid-col work_id \
  --title-col title \
  --abstract-col abstract \
  --year-col publication_year \
  --cluster-sharded-output-dir workspace/artifacts/keyword_cluster_sharded/preflight
```

Only start the full extraction when `preflight_summary.json` reports a status
below `hard_stop`.  A `warning` status does not automatically forbid execution,
but it should trigger cap or shard-size review before a multi-day job.

Production-scale runs should also set the shard and candidate-mining controls
explicitly so the command records the intended budget:

```bash
sciscape keywords abstracts.parquet membership.parquet \
  --keyword-engine cluster_sharded \
  --cluster-level canonical_nano_id \
  --uid-col work_id \
  --title-col title \
  --abstract-col abstract \
  --year-col publication_year \
  --include-title \
  --ngram-max 3 \
  --top-n 50 \
  --target-docs-per-shard 500000 \
  --max-clusters-per-shard 1024 \
  --candidate-pool-floor 256 \
  --candidate-pool-large 1024 \
  --candidate-pool-hard-max 1536 \
  --candidate-mining-progress-interval-docs 25000 \
  --candidate-mining-prune-interval-docs 50000 \
  --candidate-mining-prune-multiplier 8 \
  --quality-rerank \
  --cluster-sharded-output-dir workspace/artifacts/keyword_cluster_sharded/full_run \
  --progress-path workspace/artifacts/keyword_cluster_sharded/full_run/progress.json \
  -o workspace/artifacts/keyword_cluster_sharded/full_run/keywords.parquet
```

Each candidate shard writes `candidate_shard_XXXX.progress.json` while running
and records elapsed time, throughput, memory, tracked terms, and pruned term
counts in `candidate_shard_XXXX.done.json`.  Treat a stale progress file with
no matching `.done.json` as an interrupted shard that must be resumed or rerun.

## Remaining TODO

Before claiming full large-scale support, add a benchmark matrix that records:

- corpus size and cluster count;
- cap policy (`none`, fixed docs/cluster, representative-only);
- n-gram range and minimum frequency settings;
- enabled stages, especially vocab cleansing, quality rerank, temporal metrics,
  co-occurrence, and term network roles;
- wall time by stage;
- memory high-water mark;
- shard count, shard reuse rate, and restart behavior;
- output row count and QC failure count;
- top-k semantic spot review results.

Recommended benchmark slices:

| Slice | Purpose |
| --- | --- |
| 1k docs / 10 clusters | CI-scale correctness and artifact QC |
| 10k docs / 100 clusters | local developer smoke |
| 100k docs / 1k clusters | workstation-scale regression |
| 500k docs / 5k clusters with caps | Atlas-like production test |
| uncapped 500k+ docs | full-corpus stress test before public scale claims |

The benchmark output should be a machine-readable JSON plus a compact Markdown
summary that can be attached to release readiness checks.
