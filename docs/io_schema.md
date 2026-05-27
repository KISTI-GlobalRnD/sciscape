# SciScape I/O Schema Reference

이 문서는 `sciscape.clustering`과 `sciscape.keyword_extraction`의
입력/출력 데이터 스키마를 정의합니다.

---

## 1. Clustering (`sciscape.clustering`)

### 입력: Edge Table

| 컬럼 | 타입 | 필수 | 설명 |
|------|------|------|------|
| `uid1` | string | Yes | 첫 번째 노드 ID |
| `uid2` | string | Yes | 두 번째 노드 ID |
| `rel_sum2` | float64 | Yes | 엣지 가중치 (정규화된 관련성 점수) |

**형식**: Parquet 또는 ZIP 안의 TSV
**비고**: 무방향 엣지 (uid1, uid2 순서 무관). Self-loop 미지원.

### 출력: ClusterTables

`run_pipeline()` → `ClusterTables` dataclass 반환.

#### membership (pl.DataFrame)

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `uid` | string | 원본 노드 ID |
| `{level_name}` | int | 계층 레벨별 인덱스 (1-indexed) |
| `total_index` | string | 점 구분 계층 인덱스 (예: `"1.2.3"`) |
| `cluster_{level_name}` | int | 해당 레벨의 Leiden 파티션 멤버십 |

#### description (pl.DataFrame)

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `{level_name}` | int | 계층 레벨 인덱스 |
| `total_index` | string | 점 구분 계층 인덱스 |
| `number_of_nodes` | int | 클러스터 내 문서 수 |

#### 기타 필드

| 필드 | 타입 | 설명 |
|------|------|------|
| `raw_membership` | pl.DataFrame | 계층화 전 원시 Leiden 파티션 |
| `levels` | Tuple[str, ...] | 계층 레벨 이름 (예: `("micro", "meso", "macro")`) |
| `resolutions` | Dict[str, float] | 레벨별 γ (해상도) 파라미터 |
| `qualities` | Dict[str, float] | 레벨별 modularity 품질 |

### 최소 설정

```python
from sciscape.clustering import LeidenConfig, run_pipeline

config = LeidenConfig(
    level_constraints=[(5, 100), (80, 500), (400, 5000)],
    resolution_bounds=(1e-3, 5.0),
)

tables = run_pipeline(
    zip_path=Path("edges.zip"),
    inner_name="edges.txt",
    config=config,
)
# tables.membership.to_parquet("membership.parquet")
```

---

## 2. Keyword Extraction (`sciscape.keyword_extraction`)

### 입력 1: Abstract Parquet

| 컬럼 | 타입 | 필수 | 설정 | 설명 |
|------|------|------|------|------|
| `uid` | string/int | Yes | `uid_col` | 문서 고유 ID |
| `abstract` | string | Yes | `abstract_col` | 초록 텍스트 |
| `pubyear` | int | Yes | `year_col` | 발행 연도 |
| `title` | string | No | `title_col` | 제목 (`include_title=True` 시 사용) |

**형식**: Parquet (PyArrow 스트리밍 지원)
**인코딩**: UTF-8
**비고**: UID는 membership 테이블과 일치해야 함. 중복 UID 시 양쪽 모두 사용됨.

### 입력 2: Membership Parquet

| 컬럼 | 타입 | 필수 | 설정 | 설명 |
|------|------|------|------|------|
| `uid` | string/int | Yes | `uid_col` | 문서 고유 ID (abstract와 동일) |
| `cluster_*` | int | Yes | `cluster_level` | 클러스터 ID (NaN 행은 제외됨) |

**비고**: `cluster_level` 기본값은 `None` (자동 감지: 가장 세분화된 `cluster_*` 컬럼 선택). 명시적 지정도 가능.

### 입력 3: Author Keywords (선택)

| 컬럼 | 타입 | 필수 | 설정 | 설명 |
|------|------|------|------|------|
| `uid` | string/int | Yes | `author_keyword_uid_col` | 문서 ID |
| `keyword` | string | Yes | `author_keyword_term_col` | 저자 키워드 |

**비고**: `author_keyword_path` 설정 시 활성화. 초록 텍스트에 병합됨.

### 출력: Keywords DataFrame (pd.DataFrame)

`run_keyword_pipeline()` 또는 `KeywordExtractionPipeline.run()` 반환.

#### CORE (항상 존재)

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `cluster_id` | int | 클러스터 ID |
| `term` | string | 추출된 키워드 |
| `score` | float | c-TF-IDF (또는 blended) 점수 |
| `frequency` | int | 클러스터 내 출현 빈도 |

#### TIER 2 (해당 스테이지 활성화 시)

| 컬럼 | 타입 | 조건 | 설명 |
|------|------|------|------|
| `doc_coverage` | int | Stage 4 | 해당 용어가 등장하는 문서 수 |
| `source_terms` | List[str] | Stage 7 (LLM) | 정규화 전 원래 용어 목록 |
| `pub_year_series` | Dict[int, int] | Stage 9 | 연도별 출현 빈도 `{2020: 15, 2021: 23}` |
| `ppm_series` | Dict[int, float] | Stage 9 | 연도별 PPM (백만분율) |
| `loglift_series` | Dict[int, float] | Stage 9 | 연도별 log-likelihood ratio |
| `bayesian_log_odds_series` | Dict[int, float] | Stage 9 | 연도별 Bayesian log-odds |
| `year_denominators` | Dict[int, int] | Stage 9 | 연도별 전체 문서 수 (PPM 분모) |

#### TIER 3 (고급 필터/스테이지 활성화 시)

| 컬럼 | 타입 | 조건 | 설명 |
|------|------|------|------|
| `depth_score` | float | Stage 8 (depth) | 개념 깊이 추정치 |
| `depth_level` | int | Stage 8 (depth) | 깊이 등급 (0=표면, 높을수록 심층) |
| `cross_cluster_count` | int | Stage 8 (depth) | 해당 용어가 등장하는 클러스터 수 |
| `candidates` | List[str] | Stage 6→7 | 동의어 후보 (term network + sim graph) |
| `expanded_from` | string | P4 | 확장 전 원래 약어 |
| `alias_actions` | List[str] | Stage 7 (LLM) | 정규화 결정 (keep/merge_into/drop) |
| `alias_notes` | string | Stage 7 (LLM) | 결정 사유 (사람 읽기용) |
| `alias_reason` | string | Stage 7 (LLM) | 결정 사유 (기술적) |
| `raw_term` | string | Quality diagnostics | 정제 전 term 보존값 |
| `normalized_term` | string | Quality diagnostics | 품질 판정용 정규화 term |
| `display_label` | string | Quality diagnostics | 보고서/시각화 표시용 label |
| `quality_score` | float | Quality diagnostics | 범용 품질 보정 점수 |
| `quality_multiplier` | float | Quality diagnostics | 원점수 대비 보정 배율 |
| `quality_flags` | string | Quality diagnostics | `too_global`, `phrase_preferred`, `acronym_like` 등 pipe-delimited reason code |

### 최소 설정

```python
from sciscape.keyword_extraction import KeywordExtractionConfig, run_keyword_pipeline

cfg = KeywordExtractionConfig(
    abstract_path=Path("abstracts.parquet"),
    membership_path=Path("membership.parquet"),
    # cluster_level=None → auto-detects finest level
)

keywords = run_keyword_pipeline(cfg)
```

### 전체 기능 설정

```python
from sciscape.keyword_extraction import (
    KeywordExtractionConfig,
    run_keyword_pipeline,
)
from sciscape.keyword_extraction.config import VocabMergeConfig
from sciscape.keyword_extraction.depth import DepthConfig
from sciscape.keyword_extraction.term_network import TermNetworkConfig

cfg = KeywordExtractionConfig(
    abstract_path=Path("abstracts.parquet"),
    membership_path=Path("membership.parquet"),
    # cluster_level auto-detected (finest level)
    include_title=True,
    title_weight=2.0,
    top_n_keywords=100,
    scoring_pool_factor=1.5,

    # Stage 3: Vocab Cleansing
    vocab_merge=VocabMergeConfig(enabled=True),

    # Post-scoring normalization
    normalization_enabled=True,
    norm_plural_merge_enabled=True,

    # Quality filters
    academic_stopwords_enabled=True,
    artifact_filter_enabled=True,
    cross_cluster_penalty_enabled=True,
    quality_diagnostics_enabled=True,
    quality_rerank_enabled=True,
    fragment_suppression_enabled=True,

    # Stage 5-6: Cooccurrence + Term Network
    cooccurrence_enabled=True,
    term_network=TermNetworkConfig(
        enabled=True,
        layers=["string", "token", "cooccurrence"],
    ),
    auto_merge_enabled=True,
    short_term_expansion_enabled=True,

    # Stage 8-9: Depth + Temporal
    depth=DepthConfig(enabled=True, n_levels=3),
    n_jobs=4,
)
```

---

## 3. 파이프라인 연결

Clustering 출력 → Keyword Extraction 입력의 연결:

```
clustering.run_pipeline()
    ↓
  tables.membership.to_parquet("membership.parquet")
    ↓  columns: uid, cluster_micro, cluster_meso, ...
    ↓
keyword_extraction.run_keyword_pipeline(cfg)
    cfg.membership_path = "membership.parquet"
    cfg.cluster_level = None  # auto-detect finest level (또는 명시적 지정)
```

**주의사항**:
- `uid` 컬럼이 양쪽 Parquet에서 동일한 타입/값이어야 함
- Membership에 없는 UID의 문서는 자동 제외됨
- 클러스터 ID가 NaN인 행은 자동 제외됨

---

## 4. Pipeline Stage 아키텍처

```
Pass 1: 문서 스캔 + 전수 정제
  Stage 1  Vectorization         — CountVectorizer 학습
  Stage 2  Aggregation           — 문서→클러스터 카운트 집계
  Stage 3  Vocab Cleansing       — 표기/철자/복수형/편집거리 정제
  Stage 4  Scoring + Top-K       — c-TF-IDF 점수화 + 풀링

Pass 2: 키워드 정제
  Quality Refinement       — 범용 quality flags/display label/reranking
  Stage 5  Cooccurrence          — 용어 공출현 행렬
  Stage 6  Term Network          — 유사도 네트워크 + 병합 그룹
  Stage 7  LLM Canonicalize      — LLM 기반 정규화 (선택)

Pass 3: 메타데이터
  Stage 8  Depth                 — 개념 깊이 추정
  Stage 9  Temporal              — 연도별 시계열 지표

Quality Filters (스테이지 내에서 적용):
  P1  Academic stopwords         — 학술 보일러플레이트 제거
  P2  Plural merge               — 복수형 → 단수형 병합
  P3  Auto-merge                 — 고신뢰도 동의어 자동 병합
  P4  Short-term expansion       — 짧은 약어 맥락 확장
  P5  Artifact filter            — LaTeX 잔여물/숫자/출판사 제거
  P6  Cross-cluster penalty      — 다수 클러스터 공통 용어 패널티
  P7  Fragment suppression       — 잘린 n-gram 억제
  P8  Quality rerank             — 공통어 하향, phrase 우선, 약어 display 확장
```
