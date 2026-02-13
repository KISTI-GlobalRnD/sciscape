# Leiden Module (KRISS Pair Links)

## Overview
- End-to-end utilities for KRISS pair-link analytics: graph construction, multi-level Leiden clustering, hierarchy navigation, and cluster labelling.
- Includes a scalable keyword-extraction pipeline (uni/bi/tri-gram, c‑TF‑IDF, LLR, redundancy control) designed for million-document corpora.

## 개요
- KRISS 논문 네트워크 분석을 위한 그래프 구축·Leiden 클러스터링·계층 탐색·클러스터 라벨링 도구를 제공합니다.
- 최대 백만 문헌 규모에서 동작하는 유니그램/바이그램/트라이그램 키워드 추출 파이프라인(c‑TF‑IDF, LLR, 중복 억제)을 포함합니다.

---

## 1. Clustering Pipelines / 클러스터링 파이프라인

- `run_pipeline`, `run_hierarchy_pipeline`: single/multi-level Leiden clustering with automatic resolution search.
- `resolve_resolution_schedule`, `scan_resolution_grid`: resolution tuning, metric logging, multiprocessing support.
- `merge_small_clusters`, `PostprocessConfig`: size/weight based post-processing.
- `build_cluster_tables`, `get_cluster_hierarchy`, `HierarchyBuilder`: hierarchy construction and export helpers.
- `cluster_naming`, `build_core_documents`: LLM-based naming with optional DB fetching of representative papers.

See `notebooks/bio_ce_cx_demo.ipynb` for an interactive example and `demo_bio_ce_cx.py` for a command-line walkthrough.

---

## 2. Keyword Extraction Pipeline / 키워드 추출 파이프라인

### Highlights / 주요 특징
- **Document-level sparse aggregation**: CountVectorizer on raw abstracts (with optional title boost), followed by per-cluster group-sum.
- **Exact c‑TF‑IDF** with optional Log-Likelihood Ratio reweighting.
- **Phrase control**: uni-grams + configurable n-gram range (default bi/tri-gram), per-cluster frequency and document-coverage thresholds, subphrase suppression, Jaccard-MMR diversity.
- **Streaming-friendly**: row-group iteration via PyArrow or Polars fallback.
- **Time-series ready**: selected keywords are replayed to compute year-frequency (raw counts, easy to extend to normalised ratios).
- **Deterministic aliasing**: built-in unit rewrites, manual overrides, and forbidden-term guards keep “bq”/“sv” style abbreviations out of the canonical output.
- **Snapshot-friendly**: Stage 2 artefacts can be saved once and reused across many Stage 2.5 experiments (LLM, cache-only, or manual) without recomputing sparse matrices.

### Baseline Configuration / 권장 기본값

```python
from pathlib import Path
from leiden_module.keyword_extraction import KeywordExtractionConfig

cfg = KeywordExtractionConfig(
    abstract_path=Path("Data/FINAL_title_abstract_pubyear.parquet"),
    membership_path=Path("Output/total_membership_gamma16.parquet"),
    cluster_level="cluster_micro",

    include_title=True,
    title_weight=1.5,
    lowercase=True,
    strip_accents="unicode",

    min_df_unigram=5,
    max_df_unigram=0.98,
    min_df_phrase=5,
    max_df_phrase=0.98,
    use_phrase_vectorizer=True,
    ngram_min=2,
    ngram_max=3,
    phrase_min_count_per_cluster=10,

    min_cluster_doc_coverage=10,
    min_cluster_doc_coverage_ratio=0.01,  # ≥1% of cluster docs
    top_n_keywords=100,
    mmr_jaccard_lambda=0.25,
    mmr_pool_factor=3.0,

    w_ctfidf=1.0,
    w_llr=0.4,
    n_jobs=48,
)
```

### Quick Usage / 빠른 사용법

```python
from leiden_module.keyword_extraction import KeywordExtractionPipeline

pipeline = KeywordExtractionPipeline(cfg)
pipeline.run()  # returns DataFrame with cluster_id, term, score, frequency, pub_year_series
```

You can pause between stages:

```python
pipeline._fit_vectorizers()
pipeline._aggregate_counts()   # save sparse matrices if you need a checkpoint
raw_top_df = pipeline._stage_scores_and_topk()
term_year = pipeline._compute_year_series(raw_top_df)
```

#### Stage 2 Snapshots / Stage 2 세이브포인트

```python
snapshot_dir = Path("artifacts/stage2_snapshot_20251015")

# After Stage 2
pipeline.save_stage2_snapshot(snapshot_dir, raw_top_df)

# Later – reuse Stage 2 artefacts and re-run canonicalisation/time-series
final_df = pipeline.run_from_stage2_snapshot(snapshot_dir)
```

The snapshot includes vectorisers, sparse matrices, feature-name arrays, cluster statistics, and the Stage 2 dataframe. Use it to iterate on canonicalisation settings without recomputing Stage 0–2.

---

## Installation / 설치

```bash
pip install polars python-igraph leidenalg scipy scikit-learn joblib pyarrow
```

For keyword extraction, PyArrow accelerates streaming; Polars is an optional fallback.

---

## Quick Start (Clustering) / 빠른 시작 (클러스터링)

```python
from pathlib import Path
from leiden_module import LeidenConfig, run_pipeline

config = LeidenConfig(
    level_constraints=[(5, 100), (80, 500), (400, 5000)],
    resolution_bounds=(1e-3, 5.0),
    max_iterations=32,
    log_history=True,
)

tables = run_pipeline(
    zip_path=Path("Data/KRISS_pair_links/dc_bc_cc_total_pair.zip"),
    inner_name="dc_bc_cc_total_pair.txt",
    config=config,
)
```

`tables.membership` provides hierarchical cluster labels, `tables.description` summarises terminal clusters, and `tables.resolutions` records the discovered γ per level. Adjust `level_constraints` or supply explicit `resolutions={...}` for manual control.

---

## Keyword Extraction CLI Example / 키워드 추출 실행 예

```python
from pathlib import Path
from leiden_module.keyword_extraction import KeywordExtractionConfig, run_keyword_pipeline

cfg = KeywordExtractionConfig(
    abstract_path=Path("Data/FINAL_title_abstract_pubyear.parquet"),
    membership_path=Path("Output/total_membership_gamma16.parquet"),
    cluster_level="cluster_micro",
    include_title=True,
    title_weight=1.5,
    min_df_unigram=5,
    min_df_phrase=5,
    phrase_min_count_per_cluster=10,
    min_cluster_doc_coverage=10,
    min_cluster_doc_coverage_ratio=0.01,
    mmr_jaccard_lambda=0.25,
    w_llr=0.4,
    n_jobs=48,
)

keywords = run_keyword_pipeline(cfg)
keywords.to_parquet("Output/cluster_keywords.parquet", index=False)
```

The resulting columns:
- `cluster_id`
- `term`
- `score` (combined c‑TF‑IDF / LLR / MMR ranking)
- `frequency` (cluster term count)
- `pub_year_series` (dict: year → frequency)

---

## Advanced Controls / 추가 설정

| Parameter | Description (EN) | 설명 (KO) |
|-----------|-----------------|-----------|
| `min_df_*`, `max_df_*` | Drop extremely rare or ubiquitous tokens (int or proportion). | 너무 드물거나 자주 등장하는 토큰을 제거합니다. |
| `phrase_min_count_per_cluster` | Minimum per-cluster frequency for phrases after aggregation. | 클러스터 내 최소 등장 횟수(바이/트라이그램). |
| `min_cluster_doc_coverage[_ratio]` | Enforce representative terms by document count or percentage. | 문서 수/비율 조건으로 대표성 확보. |
| `mmr_jaccard_lambda`, `mmr_pool_factor` | Redundancy control via Jaccard MMR; pool factor determines candidate buffer size. | 자카드 기반 중복 억제, 후보 풀 크기 조절. |
| `w_ctfidf`, `w_llr` | Blend c‑TF‑IDF with LLR for between-cluster discriminativeness. | c‑TF‑IDF와 LLR 가중치 조합. |
| `alias_strategy` | `"llm"`, `"cache_only"`, `"load_only"`, `"prev_top_df"` canonicalisation modes. | 정규화 모드 선택 (LLM, 캐시, 이전 결과 재활용 등). |
| `manual_alias_path` | CSV/TSV/JSON file containing `cluster_id, original, action, canonical, …`. | 클러스터별 수동 정규화 규칙 파일. |
| `builtin_aliases`, `forbid_abbreviations` | Deterministic rewrites and hard bans for sensitive abbreviations. | 주요 약어 재매핑 및 금지어 설정. |

Prefer `save_stage2_snapshot` / `load_stage2_snapshot` / `run_from_stage2_snapshot` for quick iteration on Stage 2.5; the helpers package all Stage 2 artefacts and eliminate manual `joblib` juggling.

---

## Testing / 테스트

A smoke test is provided:

```bash
pytest leiden_module/tests/test_keyword_extraction.py
```

It verifies the keyword pipeline runs end-to-end with the new knobs (LLR, coverage ratios, MMR).

---

## Environment Configuration / 환경 설정 (.env)

```
OLLAMA_BASE_URL=http://172.16.2.42:11434/v1
OLLAMA_API_KEY=ollama
OLLAMA_MODEL=gpt-oss:20b
CLUSTER_DB_DRIVER=mysql        # or sqlite
CLUSTER_DB_HOST=...
CLUSTER_DB_PORT=3306
CLUSTER_DB_NAME=...
CLUSTER_DB_USER=...
CLUSTER_DB_PASSWORD=...
CLUSTER_TEMP_TABLE=tmp_cluster_uids
CLUSTER_META_TABLE=paper_metadata
CLUSTER_METRIC_TABLE=paper_metrics
```

Set `export OLLAMA_CONFIG=/path/to/.env` before running cluster naming utilities.

---

## Support / 문의

- Issues & feature requests: open a ticket in the repository.
- 업데이트나 콜라보 제안은 PR 또는 Issue로 남겨주세요. 빠르게 검토하겠습니다.

Happy clustering & keywording! 즐거운 분석 되세요!
