# SciScape

Leiden 기반 클러스터링 파이프라인과 클러스터 단위 키워드 추출 파이프라인을 하나의 Python 패키지(`sciscape`)로 제공합니다.

## 구성

- `sciscape.clustering`: 그래프 구성, Leiden 클러스터링, 해상도(γ) 탐색, 계층(hierarchy) 구성, 후처리(소규모 클러스터 병합) 등
- `sciscape.keyword_extraction`: 문서(초록/제목) 기반 n-gram 키워드 추출(c‑TF‑IDF, LLR, MMR 다양성, 연도별 시계열) 등

호환을 위해 `sos.*` import는 shim으로 유지합니다.

## 설치

이 저장소 루트에서:

```bash
python -m pip install -U pip
python -m pip install .
```

옵션 기능:

```bash
# Parquet row-group 스트리밍 가속
python -m pip install .[arrow]

# LLM 기반 정규화/요약 기능 사용 시
python -m pip install .[llm]
```

## 빠른 사용법

### 1) Clustering

```python
from pathlib import Path
from sciscape.clustering import LeidenConfig, run_pipeline

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

### 2) Keyword Extraction

```python
from pathlib import Path
from sciscape.keyword_extraction import KeywordExtractionConfig, run_keyword_pipeline

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

### (선택) Stage 2.5: LLM 기반 용어 정규화(alias merge/drop/translate)

`KeywordExtractionConfig`에서 `apply_alias_map=True`를 켜고 `alias_strategy`를 지정하면,
추출된 키워드를 canonical form으로 정규화할 수 있습니다.

- `alias_strategy="llm"`: LLM 호출로 `keep / merge_into / translate / drop` 결정(기본 chunking)
- `alias_strategy="llm_candidates"`: term별 `candidates`(후보 canonical allowlist)를 제공하는 방식  
  - `merge_into`는 해당 term의 후보 목록 안에서만 선택하도록 강제(대규모 vocabulary에서 안전성↑)

관련 옵션(일부):
- `alias_cache_path`: raw response + mapping 저장 경로
- `alias_max_terms_per_prompt`: 한 번에 보낼 term 개수
- `alias_candidate_column` / `alias_candidate_max` / `alias_candidate_enforce`: `llm_candidates`용 후보 컬럼/제약

## 개발/검증

```bash
# (권장) venv 환경에서
python -m pip install -e .[dev]
pytest -q
```

## 레거시

과거 코드/노트북 호환을 위해 `leiden_module/`이 남아있을 수 있습니다. 신규 개발/수정은 `sciscape/`를 기준으로 진행합니다.
