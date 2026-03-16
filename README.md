# SciScape

네트워크 기반 Leiden 클러스터링 + 키워드 추출 파이프라인 Python 패키지.

## 주요 기능

- **Leiden Clustering**: 그래프 기반 다중 해상도 클러스터링 (nano → micro → meso)
- **9-Stage Keyword Extraction**: c-TF-IDF + LLR 스코어링, 7가지 품질 필터, 시계열 분석
- **Input Adapters**: Web of Science, Scopus, OpenAlex 데이터 자동 변환
- **Interactive Visualization**: HTML 대시보드, 네트워크 맵, 계층 트리맵, 시계열 비교
- **Landscape Pipeline**: 엣지 → 계층적 클러스터링 → 키워드 추출 → 대화형 리포트 (원스텝)
- **CLI**: `sciscape cluster`, `sciscape keywords`, `sciscape convert`, `sciscape landscape`

## 설치

```bash
pip install .                # 기본
pip install .[viz]           # + 시각화 (Plotly)
pip install .[arrow]         # + Parquet 스트리밍 가속
pip install .[dev]           # + 개발/테스트
```

## 빠른 시작

```bash
# 외부 데이터 → SciScape 포맷
sciscape convert openalex works.jsonl -o abstracts.parquet

# 클러스터링
sciscape cluster edges.zip edges.txt --levels 5,100 80,500

# 키워드 추출 (모든 옵션 활성화)
sciscape keywords abstracts.parquet membership.parquet --enable-all -o keywords.parquet

# 전체 파이프라인 (엣지 → 클러스터링 → 키워드 → 리포트)
sciscape landscape edges.parquet abstracts.parquet -o output/landscape
```

## 구성

```
sciscape/
├── clustering/          # Leiden 클러스터링 파이프라인
├── keyword_extraction/  # 9단계 키워드 추출 파이프라인
│   └── visualization/   # 대시보드, 네트워크맵, 계층/시계열 시각화
├── adapters/            # WoS, Scopus, OpenAlex 입력 어댑터
├── landscape.py         # 엔드투엔드 파이프라인 (클러스터링 → 키워드 → 리포트)
└── cli.py               # 커맨드라인 인터페이스
```

## 문서

- 상세 사용법: [`sciscape/README.md`](sciscape/README.md)
- I/O 스키마: [`docs/io_schema.md`](docs/io_schema.md)

## Import

```python
import sciscape.clustering
import sciscape.keyword_extraction
import sciscape.adapters
```

호환을 위해 기존 `sos.*` import도 shim으로 유지합니다.
