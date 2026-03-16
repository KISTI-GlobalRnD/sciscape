# SciScape

Leiden 기반 클러스터링 파이프라인과 클러스터 단위 키워드 추출 파이프라인을 하나의 Python 패키지(`sciscape`)로 제공합니다.

## 구성

```
sciscape/
├── clustering/          # 그래프 구성, Leiden 클러스터링, 해상도(γ) 탐색, 계층 구성
├── keyword_extraction/  # 9단계 키워드 추출 파이프라인
│   └── visualization/   # 대시보드, 네트워크맵, 계층/시계열 시각화
├── adapters/            # 외부 데이터 변환 (WoS, Scopus, OpenAlex)
├── landscape.py         # 엔드투엔드 파이프라인 (서브샘플 → 클러스터링 → 키워드 → 리포트)
└── cli.py               # CLI: sciscape cluster | keywords | convert | landscape
```

- `sciscape.clustering`: 그래프 구성, Leiden 클러스터링, 해상도(γ) 탐색, 계층(hierarchy) 구성, 후처리(소규모 클러스터 병합) 등
- `sciscape.keyword_extraction`: 문서(초록/제목) 기반 n-gram 키워드 추출(c‑TF‑IDF, LLR, MMR 다양성, 연도별 시계열) 등
- `sciscape.adapters`: Web of Science, Scopus, OpenAlex 데이터를 SciScape 입력 포맷으로 변환

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

# 시각화 (Plotly 차트, 대시보드)
python -m pip install .[viz]
```

## 키워드 추출 파이프라인 아키텍처

```
                           ┌─────────────────────────────────────────┐
  Parquet Inputs           │         9-Stage Pipeline                │
 ┌──────────────┐          │                                         │
 │  abstracts   │─────────►│  1. Vectorization  (CountVectorizer)    │
 │  membership  │─────────►│  2. Aggregation    (cluster counts)     │
 └──────────────┘          │  3. Vocab Cleansing                     │
                           │     3a. Notation/Spelling normalize     │
                           │     3b. Plural singularization          │
                           │     3c. Edit-distance merge (cluster-   │
                           │         aware safety)                   │
                           │     3d. Similarity graph → VocabSimGraph│
                           │  4. Scoring        (c-TF-IDF + LLR)    │
                           │  ─── Pass 2: Keyword Refinement ───     │
                           │  5. Cooccurrence   (PMI edges)          │
                           │  6. Term Network   (multi-layer graph)  │
                           │     → Bridge merge (sim graph 1-hop)    │
                           │  7. LLM Canonicalize (optional)         │
                           │  ─── Pass 3: Metadata ───               │
                           │  8. Depth          (broad/mid/specific) │
                           │  9. Temporal       (year series, trends) │
                           └────────────────┬────────────────────────┘
                                            │
                                            ▼
                                    ┌───────────────┐
                                    │  keywords.parquet
                                    │  + Dashboard HTML
                                    │  + Plotly charts
                                    └───────────────┘
```

**7가지 품질 필터** (P1–P7):
- P1: 학술 보일러플레이트 제거 (based, using, proposed method 등)
- P2: 복수형 자동 병합 (point clouds → point cloud)
- P3: Term network 기반 동의어 자동 병합
- P4: 짧은 약어 맥락 확장 (cooccurrence 기반)
- P5: 아티팩트 제거 (LaTeX 잔여물, 순수 숫자)
- P6: Cross-cluster 비특이적 용어 패널티
- P7: Fragment suppression (경계/브릿지 중복)

## CLI

```bash
# 클러스터링
sciscape cluster edges.zip edges.txt --levels 5,100 80,500 -o membership.parquet

# 키워드 추출
sciscape keywords abstracts.parquet membership.parquet \
  --cluster-level cluster_micro --top-n 100 --include-title --enable-all \
  -o keywords.parquet

# 외부 데이터 변환
sciscape convert wos savedrecs.txt -o abstracts.parquet
sciscape convert scopus scopus_export.csv -o abstracts.parquet
sciscape convert openalex works.jsonl -o abstracts.parquet

# 전체 파이프라인 (엣지 → 클러스터링 → 키워드 → 리포트)
sciscape landscape edges.parquet abstracts.parquet -o output/landscape --n-nodes 100000
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
    zip_path=Path("Data/edge_list.zip"),
    inner_name="edges.txt",
    config=config,
)
```

### 2) Keyword Extraction

```python
from pathlib import Path
from sciscape.keyword_extraction import KeywordExtractionConfig, run_keyword_pipeline

cfg = KeywordExtractionConfig(
    abstract_path=Path("abstracts.parquet"),
    membership_path=Path("membership.parquet"),
    # cluster_level=None → auto-detect finest level
    include_title=True,
    title_weight=1.5,
    min_df_unigram=5,
    min_df_phrase=5,
    n_jobs=-1,
)

keywords = run_keyword_pipeline(cfg)
keywords.to_parquet("keywords.parquet", index=False)
```

### 3) Input Adapters

```python
from sciscape.adapters import read_wos, read_scopus, read_openalex

# Web of Science
df = read_wos("savedrecs.txt")

# Scopus
df = read_scopus("scopus_export.csv")

# OpenAlex (supports .jsonl, .parquet, .csv — auto-reconstructs inverted index)
df = read_openalex("works.jsonl")

# 결과를 파이프라인 입력으로 저장
df.to_parquet("abstracts.parquet", index=False)
```

### 4) Visualization

```python
from sciscape.keyword_extraction.visualization import (
    export_dashboard,           # Self-contained HTML dashboard
    plot_cluster_map,           # 2D network map (MDS/spring layout)
    plot_cluster_map_with_keywords,  # Network map + keyword annotations
    plot_cluster_treemap,       # Hierarchical treemap drill-down
    plot_cluster_sunburst,      # Sunburst chart (cluster → depth → keyword)
    plot_temporal_heatmap,      # Keyword × year heatmap
    plot_cluster_trend_comparison,  # Cluster-level trend lines
    plot_cluster_keywords,      # Top-N keywords per cluster (bar chart)
    plot_score_distribution,    # Score box plots
    plot_cross_cluster_terms,   # Shared terms heatmap
)

# HTML 대시보드 (브라우저에서 열기)
export_dashboard(keywords, output_path="dashboard.html", open_browser=True)

# 개별 Plotly 차트
fig = plot_cluster_map(keywords, layout="mds")
fig.show()

fig = plot_cluster_sunburst(keywords)
fig.show()
```

### 5) Quality Filters

파이프라인에 내장된 7가지 품질 필터로 도메인 특화 키워드 품질을 높입니다.

```python
from sciscape.keyword_extraction.config import VocabMergeConfig
from sciscape.keyword_extraction.term_network import TermNetworkConfig
from sciscape.keyword_extraction.depth import DepthConfig

cfg = KeywordExtractionConfig(
    ...,
    # Stage 3: Vocab Cleansing (notation, plural, edit-distance, similarity graph)
    vocab_merge=VocabMergeConfig(enabled=True),
    # P1: 학술 보일러플레이트 제거
    academic_stopwords_enabled=True,
    # P5: 아티팩트 제거
    artifact_filter_enabled=True,
    # P6: Cross-cluster 패널티
    cross_cluster_penalty_enabled=True,
    # P7: Fragment suppression
    fragment_suppression_enabled=True,
    # Cooccurrence + Term Network
    cooccurrence_enabled=True,
    term_network=TermNetworkConfig(enabled=True, layers=["string", "token", "cooccurrence"]),
    auto_merge_enabled=True,
    short_term_expansion_enabled=True,
    # Depth classification
    depth=DepthConfig(enabled=True, n_levels=3),
)
```

British/American 철자 변이(disc→disk, colour→color 등 35종)는 정규화에서 자동 처리됩니다.

## I/O 스키마

자세한 입출력 스키마 문서는 [`docs/io_schema.md`](../docs/io_schema.md)를 참고하세요.

## 개발/검증

```bash
# (권장) venv 환경에서
python -m pip install -e .[dev]
pytest -q    # 420+ tests
```
