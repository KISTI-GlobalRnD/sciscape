# SciScape

학술 논문 네트워크를 위한 SciSci 전주기 분석/시각화 패키지입니다. 데이터
변환, 네트워크 구성, multi-layer clustering, 키워드 추출, 네트워크 시각화,
리포트/뷰어 생성, 평가 유틸리티를 포함합니다.

## 구성

저장소 전체 구조와 산출물 위치는
[`../docs/project_structure.md`](../docs/project_structure.md)를 기준으로 봅니다.
패키지 구현의 source of truth는 이 `sciscape/` 디렉토리입니다.

```
sciscape/
├── clustering/              # CPM Leiden 클러스터링
│   ├── prepartition.py        #   high-γ block → contraction → cascade hot start
│   ├── postprocess.py       #   split/merge refinement, γ search
│   ├── hierarchy_oversize_postprocess.py # 계층별 oversize 후처리 (내부 dev opt-in)
│   ├── hierarchy_postprocess.py # 기존 import 호환 경로
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
├── cli.py                   # CLI: query | cluster | keywords | convert | landscape | visualize | viewer | export | web | gui
└── gui.py                   # Tkinter GUI
```

## 현재 지원 표면

기본 공개 표면은 end-to-end SciSci analysis/visualization workflow입니다.
데이터 변환, OpenAlex ingestion, multi-layer edge construction/combination,
Rust CPM/Leiden clustering, hierarchy construction, keyword extraction,
network visualization, report/web viewer, evaluation utilities를 포함합니다.
클러스터링이 핵심 엔진이지만, Sciscape 브랜딩은 전주기 분석과 시각화
패키지에 둡니다.

Dongdaemun은 개발/연구용 family name입니다. 구체적인 claim에는
`../docs/dongdaemun_naming_contract.md`를 따르고, `Dongdaemun-post`,
`Dongdaemun-refinement`, diagnostic-only artifact를 구분합니다. 기본
`sciscape landscape` 동작은 development-only Dongdaemun refinement를 켜지
않습니다.

## 설치

```bash
pip install .                 # 기본 Python package
pip install ./rust ./rust-text . # Rust backends 포함
pip install ".[viz]"          # + 시각화 (Plotly)
pip install ".[arrow]"        # + Parquet 메타데이터 가속
pip install ".[llm]"          # + LLM 정규화/요약 (OpenAI)
pip install ".[dev]"          # + 개발/테스트
```

## CLI 전체 옵션

### `sciscape landscape` — 전체 파이프라인

```bash
sciscape landscape abstracts.parquet edges.parquet -o workspace/output/landscape [options]
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

### `sciscape visualize` — 키워드 테이블 시각화

```bash
sciscape visualize keywords.parquet -o report/
sciscape visualize keywords.csv -o dashboard.html --dashboard-only
```

최소 입력 스키마는 `cluster_id`, `term`입니다. `score`, `frequency`,
`doc_coverage`가 없으면 샘플 확인용 기본값으로 채워 dashboard를 생성합니다.

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

`viewer/index.html`과 `viewer/data.json`을 같은 GitHub Pages 폴더에 두면
브라우저가 `data.json`을 자동으로 불러옵니다. 다른 위치의 데이터는
`viewer/index.html?data=path/to/data.json`처럼 지정할 수 있습니다.

### `sciscape web` — FastAPI 웹 인터페이스

```bash
sciscape web --host 127.0.0.1 --port 8000
```

브라우저에서 `http://127.0.0.1:8000`을 열고 OpenAlex query를 입력하면
fetch → edge construction → landscape clustering → keyword extraction →
report/dashboard 생성까지 백그라운드 job으로 실행됩니다. 완료 후 Web 탭에서
network, hierarchy, temporal, cluster, keyword view를 바로 확인하고,
Download 탭에서 생성된 HTML report/dashboard를 열 수 있습니다.

### `sciscape export` — 네트워크 내보내기

```bash
sciscape export edges.parquet membership.parquet --format gexf -o network.gexf
```

### `sciscape gui` — GUI 실행

```bash
sciscape gui
```

## 출력 파일

`sciscape landscape` 실행 후 output 디렉토리 구조:

```
workspace/output/landscape/
├── membership.parquet       # uid + cluster_nano + cluster_micro
├── nano/, micro/, ...       # 계층별 membership.parquet, meta.json
│   └── postprocess/         # 내부 opt-in 활성화 시 summary/move diagnostics
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

## 계층 후처리 자동화 (내부 개발 opt-in)

기본 `sciscape landscape` 및 CLI 동작은 변경하지 않습니다. 개발/실험용으로
명시적으로 켰을 때만 각 hierarchy level에서 다음 흐름을 수행합니다.

```
raw Leiden
  → small-cluster repair
  → oversize split-repair probes
  → oversize boundary trim
  → projection + membership/meta 저장
  → contraction for next level
```

목표는 작은 클러스터를 해결한 뒤에도 남는 oversized cluster가 다음 단계
contraction을 지배하지 않도록, 품질 보존 우선 정책으로 진단과 완화를
자동화하는 것입니다.

```python
from sciscape.clustering.hierarchy_oversize_postprocess import HierarchyPostprocessConfig
from sciscape.clustering.hierarchical import build_hierarchy

result = build_hierarchy(
    layer_paths={...},
    cache_dir="workspace/output/field_15",
    n_levels=4,
    hierarchy_postprocess=HierarchyPostprocessConfig(
        enabled=True,
        oversize_policy="quality_first",
        # use_rust_dongdaemun=True,  # 개발용 fast path
        # write_artifacts=False,     # required for the Rust fast path today
    ),
)
```

정책:
- `quality_first` (기본): 최종 exact CPM 품질 floor를 만족하면 accept합니다.
  `target_max_doc_weight` 달성 여부는 summary에 별도 기록합니다.
- `hard_cap`: 품질 floor와 max doc-weight cap을 모두 만족할 때만 accept합니다.
  실패 시 diagnostic artifact는 남기고 다음 level에는 small-repaired membership을
  사용합니다.

활성화 시 level별로 `postprocess/summary.json`,
`postprocess/oversize_boundary_trim_moves.csv`, 확장된 `meta.json`을 기록합니다.
캐시는 `postprocess_config_hash`가 일치할 때만 재사용됩니다.

Dongdaemun 관련 Rust fast path는 현재 개발/연구 전용입니다. 메인 공개 표면은
Rust CPM/Leiden 실행, projection, contraction, small-cluster postprocess와 SciSci
연구용 진단 모듈입니다.

방법론 리포트: [`docs/hierarchy_two_stage_postprocess_report.tex`](../docs/hierarchy_two_stage_postprocess_report.tex)

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

GitHub Pages 같은 정적 호스팅에서도 서버 없이 바로 탐색할 수 있습니다.

**배포 방식:**
- 단일 파일: `sciscape visualize ... --dashboard-only -o dashboard.html`로
  데이터가 포함된 HTML을 만들고 `dashboard.html`만 업로드
- 분리형 뷰어: `sciscape viewer -o viewer.html`로 만든 `viewer.html`과
  `report/data.json`을 같은 폴더에 `data.json` 이름으로 업로드
- 명시 경로: `viewer.html?data=path/to/data.json` 또는 `?data=path/to/keywords.csv`
- 수동 로드: 브라우저에서 `data.json`, `keywords.csv`, `keywords.tsv`를 drag/drop

**지원 포맷:**
- `data.json` — 전체 데이터 (네트워크, 시계열, 계층, 필터 이력)
- `keywords.csv` / `.tsv` — 범용 (`cluster_id`, `term` 필수열, `score` 선택)

서버 없음, 데이터 전송 없음. 브라우저 로컬에서만 처리됩니다.

## Python API

### Landscape (전체 파이프라인)

```python
from sciscape.landscape import LandscapeConfig, run_landscape

result = run_landscape(
    "edges.parquet",
    "abstracts.parquet",
    "workspace/output/landscape",
    config=LandscapeConfig(
        min_docs_per_cluster=500,
        gamma_block="auto",
        n_target_nodes=100_000,
    ),
)
# result["report_dir"], result["keywords_df"], ...
```

내부 계층 후처리를 `run_landscape` 경로에서 사용하려면 `LandscapeConfig`에
`hierarchy_postprocess=HierarchyPostprocessConfig(enabled=True)`를 전달합니다.
공개 CLI flag는 아직 제공하지 않습니다.

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

# 빈 뷰어 (GitHub Pages 배포용: 같은 폴더의 data.json 자동 로드)
export_viewer("viewer.html")
```

## 개발/검증

```bash
# 저장소 로컬 uv 환경을 사용합니다. 시스템 python에는 pytest/pip가 없을 수 있습니다.
uv run --extra dev python -m pytest -q

# Rust PyO3 binding 변경 후에는 editable native extension을 재빌드합니다.
uv run --extra dev maturin develop --manifest-path rust/Cargo.toml
uv run --extra dev python -m pytest -q

# push/release 전 전체 로컬 게이트
./scripts/release_check.sh
```

`research/**/results/**` 아래의 새 산출물은 기본적으로 git에서 무시합니다.
이미 추적 중인 curated evidence는 유지되지만, 새 산출물은
`../research/DATA_RETENTION_PLAN.md` 검토 후 필요한 경우에만 `git add -f`로
추가합니다. 릴리즈 체크리스트는 `../docs/release_readiness.md`를 참고합니다.

## I/O 스키마

자세한 입출력 스키마 문서: [`docs/io_schema.md`](../docs/io_schema.md)
