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
