# GPU Embedding Filter Runbook

This runbook is for rebuilding filtered embedding k-NN graphs for the next
cross-field consensus round on the GPU server.

## Scope

- Fields: `field_18`, `field_30`, `field_26`
- Target output: filtered `emb_full_knn30` graphs under
  `data/linktype_edges_gcc/field_{id}/`
- Filter policy:
  - `title + abstract >= 20`
  - `abstract >= 20`
  - `require_abstract = true`
  - unmatched metadata is kept by default
  - metadata join must pass `min_metadata_match >= 0.95`

## Important Caveat

Do **not** use the repo-local `data/openalex_metadata/field_*/works_text.parquet`
for these runs. The current local metadata does not match the historical
embedding artifact provenance and will fail the join-rate check.

You must point `ALIGNED_TEXT_ROOT` to a `works_text.parquet` snapshot that was
generated from the same OpenAlex snapshot as the embedding artifact.

Expected layout:

```text
${ALIGNED_TEXT_ROOT}/field_18/works_text.parquet
${ALIGNED_TEXT_ROOT}/field_30/works_text.parquet
${ALIGNED_TEXT_ROOT}/field_26/works_text.parquet
```

## One-shot Batch Run

```bash
cd /path/to/1.4.4.Sciscape
export ALIGNED_TEXT_ROOT=/path/to/aligned_openalex_text
export PYTHON_BIN=python
bash scripts/run_gpu_emb_knn_filtered_round2.sh
```

The batch script launches:

- `field_18` on GPU `0`
- `field_30` on GPU `1`
- `field_26` on GPU `2`

Logs are written to:

```text
research/consensus/results/logs_gpu_emb/
```

## Expected Outputs

Per field, the run should produce:

```text
data/linktype_edges_gcc/field_{id}/emb_full_knn30_textfilt_txt20_abs20_reqabs.parquet
data/linktype_edges_gcc/field_{id}/emb_full_knn30_textfilt_txt20_abs20_reqabs.metadata.json
```

The `.metadata.json` sidecar records:

- filter thresholds
- metadata source path
- join match rate
- kept/dropped node counts

## Manual Single-field Command

```bash
python scripts/gpu_build_emb_knn.py \
  --field 30 \
  --k 30 \
  --gpu 0 \
  --gcc-mapping data/linktype_edges_gcc/field_30/node_mapping.parquet \
  --out-dir data/linktype_edges_gcc/field_30 \
  --filter-text \
  --works-text-path /path/to/aligned_openalex_text/field_30/works_text.parquet \
  --min-text-len 20 \
  --min-abstract-len 20 \
  --require-abstract
```

## Quick Post-run Checks

```bash
ls data/linktype_edges_gcc/field_30 | rg 'textfilt|metadata'
cat data/linktype_edges_gcc/field_30/emb_full_knn30_textfilt_txt20_abs20_reqabs.metadata.json
```

If a run stops with `Metadata join rate too low`, the `works_text.parquet` path
is not aligned with the embedding artifact and should not be used.
