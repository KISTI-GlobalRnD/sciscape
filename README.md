# SciScape

학술 논문 네트워크의 Leiden 클러스터링 + 키워드 추출 파이프라인.

## 주요 기능

- **Hierarchical Leiden Clustering**: CPM 기반 다중 해상도 클러스터링, block-init + cascade hot start
- **10-Stage Keyword Extraction**: c-TF-IDF + LLR 스코어링, 7가지 품질 필터 (P1–P7), 시계열 분석
- **Input Adapters**: Web of Science, Scopus, OpenAlex, BibTeX 데이터 자동 변환
- **Interactive Visualization**: HTML 대시보드, 네트워크 맵, 계층 트리맵, 시계열 비교
- **Landscape Pipeline**: 엣지 → BFS 서브샘플 → 계층 클러스터링 → 키워드 → 대화형 리포트
- **Web Viewer**: GitHub Pages에 배포, CSV/JSON 드래그&드롭으로 결과 탐색 (서버 불필요)
- **GUI**: Tkinter 기반 최소 GUI (파라미터 설정 → 실행 → 진행률 → 리포트 열기)

## 설치

```bash
pip install .                # 기본
pip install ".[viz]"         # + 시각화 (Plotly)
pip install ".[arrow]"       # + Parquet 메타데이터 가속
pip install ".[dev]"         # + 개발/테스트
```

## 빠른 시작

### CLI

```bash
# 외부 데이터 → SciScape 포맷
sciscape convert openalex works.jsonl -o abstracts.parquet

# 전체 파이프라인 (엣지 → 클러스터링 → 키워드 → 리포트)
sciscape landscape edges.parquet abstracts.parquet -o output/

# 개별 단계
sciscape cluster edges.zip edges.txt --levels 5,100 80,500
sciscape keywords abstracts.parquet membership.parquet --enable-all -o keywords.parquet
```

### GUI

```bash
sciscape gui
```

### Web Viewer

```bash
# 뷰어 HTML 생성 (GitHub Pages 배포용)
sciscape viewer -o viewer/index.html

# 또는 기존 리포트의 data.json을 뷰어에 업로드
# → https://<user>.github.io/<repo>/ 에서 접속
```

## 패키지 구성

```
sciscape/
├── clustering/              # Leiden 클러스터링
│   ├── block_init.py        #   block-init + cascade hot start
│   ├── postprocess.py       #   split/merge refinement + γ search
│   ├── dendrogram.py        #   CPM density HAC
│   └── constrained_cut.py   #   size-constrained optimal cut
├── keyword_extraction/      # 10단계 키워드 추출
│   └── visualization/       #   대시보드, 네트워크맵, 시계열
├── adapters/                # WoS, Scopus, OpenAlex, BibTeX
├── landscape.py             # 엔드투엔드 파이프라인
├── cli.py                   # CLI (cluster, keywords, convert, landscape, viewer, gui)
└── gui.py                   # Tkinter GUI
```

## GitHub Pages Viewer

`viewer/index.html`은 정적 HTML 파일로, GitHub Pages에 배포하면 누구나 브라우저에서 결과를 탐색할 수 있습니다.

**지원 포맷:**
- `data.json` — `sciscape landscape`가 생성하는 전체 데이터 (네트워크, 시계열, 계층)
- `keywords.csv` / `keywords.tsv` — 범용 키워드 테이블 (`cluster_id`, `term`, `score` 필수)

**배포:**
1. GitHub repo Settings → Pages → Source: **GitHub Actions**
2. `viewer/index.html`이 main 브랜치에 푸시되면 자동 배포
3. 또는 Actions 탭에서 수동 트리거 (workflow_dispatch)

## 문서

- 상세 사용법: [`sciscape/README.md`](sciscape/README.md)
- I/O 스키마: [`docs/io_schema.md`](docs/io_schema.md)

## Python API

```python
from sciscape.landscape import LandscapeConfig, run_landscape

result = run_landscape(
    "edges.parquet",
    "abstracts.parquet",
    "output/",
    config=LandscapeConfig(min_docs_per_cluster=500),
)
```
