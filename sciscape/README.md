# SciScape

학술 논문 네트워크의 Leiden 클러스터링 + 키워드 추출 파이프라인.

## 구성

```
sciscape/
├── clustering/              # CPM Leiden 클러스터링
│   ├── prepartition.py        #   high-γ block → contraction → cascade hot start
│   ├── postprocess.py       #   split/merge refinement, γ search
│   ├── dendrogram.py        #   CPM density HAC
│   ├── constrained_cut.py   #   size-constrained optimal cut (DP)
│   ├── runner.py            #   LeidenRunner (leidenalg wrapper)
│   ├── ensemble.py          #   multi-seed ensemble
│   └── cluster_naming.py    #   LLM-based cluster naming (optional)
├── keyword_extraction/      # 10단계 키워드 추출
│   ├── visualization/       #   대시보드, 네트워크맵, 계층/시계열 시각화
│   └── ...                  #   vectorization → scoring → normalization → ...
├── adapters/                # WoS, Scopus, OpenAlex, BibTeX 입력 변환
├── landscape.py             # 엔드투엔드 파이프라인
├── cli.py                   # CLI: cluster | keywords | convert | landscape | viewer | gui
└── gui.py                   # Tkinter GUI
```

## 설치

```bash
pip install .                 # 기본
pip install ".[viz]"          # + 시각화 (Plotly)
pip install ".[arrow]"        # + Parquet 메타데이터 가속
pip install ".[llm]"          # + LLM 정규화/요약 (OpenAI)
pip install ".[dev]"          # + 개발/테스트
```

## CLI 전체 옵션

### `sciscape landscape` — 전체 파이프라인

```bash
sciscape landscape edges.parquet abstracts.parquet -o output/ [options]
```

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--n-nodes` | 100000 | BFS 서브샘플 대상 노드 수 |
| `--min-docs` | 1000 | 클러스터당 최소 문서 수 |
| `--gamma-block` | auto | Pre-partition γ (`auto`, `none`, 또는 float) |
| `--gamma-range` | 1e-6,1e-3 | γ 탐색 범위 (lo,hi) |
| `--seed` | 42 | 랜덤 시드 |
| `--top-n` | 80 | 클러스터당 키워드 수 |
| `--title` | SciScape Landscape | 리포트 제목 |
| `--force` | - | 캐시 무시, 전체 재실행 |
| `-v` | - | 상세 로그 출력 |

### `sciscape cluster` — 클러스터링만

```bash
sciscape cluster edges.zip edges.txt --levels 5,100 80,500 -o membership.parquet
```

### `sciscape keywords` — 키워드 추출만

```bash
sciscape keywords abstracts.parquet membership.parquet --enable-all --top-n 100 -o keywords.parquet
```

### `sciscape convert` — 데이터 변환

```bash
sciscape convert wos savedrecs.txt -o abstracts.parquet
sciscape convert scopus scopus_export.csv -o abstracts.parquet
sciscape convert openalex works.jsonl -o abstracts.parquet
sciscape convert bibtex references.bib -o abstracts.parquet
```

### `sciscape viewer` — 뷰어 HTML 생성

```bash
sciscape viewer -o viewer/index.html
```

### `sciscape gui` — GUI 실행

```bash
sciscape gui
```

## 출력 파일

`sciscape landscape` 실행 후 output 디렉토리 구조:

```
output/
├── membership.parquet       # uid + cluster_nano + cluster_micro
├── blocks.parquet           # pre-partition 캐시 (gamma_block, seed 등 메타데이터 포함)
├── abstracts_subset.parquet # 서브샘플된 초록 데이터
├── keywords.parquet         # 키워드 테이블 (term, score, frequency, temporal 등)
└── report/
    ├── data.json            # 대시보드 데이터 (viewer 업로드용)
    ├── index.html           # 메인 대시보드
    ├── network_map.html     # 클러스터 네트워크 맵 (MDS)
    ├── hierarchy_sunburst.html
    ├── hierarchy_treemap.html
    ├── temporal_heatmap.html
    ├── temporal_trends.html
    ├── landscape_detailed.html  # 키워드 annotation 클러스터 맵
    └── report.html          # 전체 리포트 인덱스
```

## Block-Init 모드

대규모 네트워크(수백만~수천만 노드)에서 효율적인 클러스터링을 위한 모드:

1. **Block formation**: 높은 γ에서 dense "Lego 블록" 생성
2. **Graph contraction**: 블록을 super-node로 축약 (10~100배 축소)
3. **γ search on contracted graph**: 축약 그래프에서 빠른 해상도 탐색
4. **Cascade hot start**: 인접 γ 결과를 초기값으로 전달, 축퇴 탈출
5. **Hot start refinement**: 원본 그래프에서 최종 정제

```python
from sciscape.landscape import LandscapeConfig, run_landscape

# auto: gamma_block = 10 × gamma_range 상한
cfg = LandscapeConfig(gamma_block="auto")

# 명시적 지정
cfg = LandscapeConfig(gamma_block=0.01)

# 비활성화 (기존 방식)
cfg = LandscapeConfig(gamma_block=None)
```

Block 캐시는 `blocks.parquet`에 저장되며, 동일 γ_block + 노드 수이면 재사용됩니다.

## 키워드 추출 파이프라인

```
                           ┌──────────────────────────────────────────┐
  Parquet Inputs           │         10-Stage Pipeline                │
 ┌──────────────┐          │                                          │
 │  abstracts   │─────────►│  1. Vectorization  (CountVectorizer)     │
 │  membership  │─────────►│  2. Vocab Merge    (edit-distance)       │
 └──────────────┘          │  3. Aggregation    (cluster counts)      │
                           │  4. Scoring        (c-TF-IDF + LLR)     │
                           │  5. Normalization  (P1–P7 filters)       │
                           │  6. Cooccurrence   (PMI edges)           │
                           │  7. Term Network   (multi-layer graph)   │
                           │  8. LLM Canonicalize (optional)          │
                           │  9. Depth          (broad/mid/specific)  │
                           │ 10. Temporal       (year series, trends) │
                           └────────────────┬─────────────────────────┘
                                            │
                                            ▼
                                    ┌───────────────┐
                                    │ keywords.parquet
                                    │ + Dashboard HTML
                                    │ + data.json
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

## Web Viewer

`report/data.json` 또는 `keywords.csv`를 GitHub Pages 뷰어에 업로드하면 브라우저에서 바로 탐색 가능:

**지원 포맷:**
- `data.json` — 전체 데이터 (네트워크, 시계열, 계층, 필터 이력)
- `keywords.csv` / `.tsv` — 범용 (`cluster_id`, `term`, `score` 필수열)

서버 없음, 데이터 전송 없음. 브라우저 로컬에서만 처리됩니다.

## Python API

### Landscape (전체 파이프라인)

```python
from sciscape.landscape import LandscapeConfig, run_landscape

result = run_landscape(
    "edges.parquet",
    "abstracts.parquet",
    "output/",
    config=LandscapeConfig(
        min_docs_per_cluster=500,
        gamma_block="auto",
        n_target_nodes=100_000,
    ),
)
# result["report_dir"], result["keywords_df"], ...
```

### Clustering

```python
from sciscape.clustering import LeidenConfig, run_pipeline

tables = run_pipeline(
    zip_path="edges.zip",
    inner_name="edges.txt",
    config=LeidenConfig(
        level_constraints=[(5, 100), (80, 500)],
        resolution_bounds=(1e-3, 5.0),
    ),
)
```

### Keyword Extraction

```python
from sciscape.keyword_extraction import KeywordExtractionConfig, run_keyword_pipeline

cfg = KeywordExtractionConfig(
    abstract_path="abstracts.parquet",
    membership_path="membership.parquet",
    include_title=True,
    top_n_keywords=100,
    normalization_enabled=True,
    cooccurrence_enabled=True,
)
keywords = run_keyword_pipeline(cfg)
```

### Input Adapters

```python
from sciscape.adapters import read_wos, read_scopus, read_openalex, read_bibtex

df = read_openalex("works.jsonl")
df = read_bibtex("references.bib")
df.to_parquet("abstracts.parquet", index=False)
```

### Visualization

```python
from sciscape.keyword_extraction.visualization import (
    export_dashboard, export_report, export_viewer,
    plot_cluster_map, plot_cluster_sunburst,
    plot_temporal_heatmap, plot_cluster_trend_comparison,
)

# HTML 대시보드
export_dashboard(keywords, "dashboard.html", open_browser=True)

# 전체 리포트 (대시보드 + 차트 + data.json)
export_report(keywords, "report/", viz_data=viz_data)

# 빈 뷰어 (GitHub Pages 배포용)
export_viewer("viewer.html")
```

## 개발/검증

```bash
python -m pip install -e ".[dev,viz,arrow]"
pytest -q    # 528 tests
```

## I/O 스키마

자세한 입출력 스키마 문서: [`docs/io_schema.md`](../docs/io_schema.md)
