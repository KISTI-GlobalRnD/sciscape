# Keyword Extraction Pipeline Guide (English / 한국어)

> This tutorial explains the redesigned keyword extraction pipeline in depth.  Every section contains
> both English and Korean text so that first-time contributors can follow along without switching
> references.

---

## Table of Contents

1. [Introduction / 소개](#introduction--소개)
2. [Problem Formulation & Notation / 문제 정의와 표기법](#problem-formulation--notation--문제-정의와-표기법)
3. [Pipeline Architecture / 파이프라인 구조](#pipeline-architecture--파이프라인-구조)
4. [Stage-by-Stage Theory / 스테이지별 이론 설명](#stage-by-stage-theory--스테이지별-이론-설명)
5. [Configuration Guidelines / 구성 가이드](#configuration-guidelines--구성-가이드)
6. [Savepoints & Restart Strategy / 저장 지점과 재시작 전략](#savepoints--restart-strategy--저장-지점과-재시작-전략)
7. [Canonicalisation & LLM Post-processing / 정규화 및 LLM 후처리](#canonicalisation--llm-post-processing--정규화-및-llm-후처리)
8. [Worked Example Notebook / 예제 노트북 안내](#worked-example-notebook--예제-노트북-안내)
9. [Diagnostics & Quality Checks / 진단과 품질 점검](#diagnostics--quality-checks--진단과-품질-점검)
10. [Appendix: Pseudocode & References / 부록: 의사코드와 참고자료](#appendix-pseudocode--references--부록-의사코드와-참고자료)

---

## Introduction / 소개

### English

The keyword extraction pipeline converts raw titles and abstracts into concise descriptors for each
cluster identified in the KRISS/Leiden workflow.  The pipeline is designed for analysts who need to:

* inspect and label clusters quickly;
* monitor temporal evolution of research topics;
* support dashboards and comparative analysis with reproducible metrics.

Design principles:

1. **Scalability** – stream Parquet row groups and operate entirely on sparse matrices.
2. **Explainability** – keep raw term frequencies, document coverage, and year-series for each keyword.
3. **Reproducibility** – provide stage-level savepoints so that long runs can resume without repeating
   previous work.
4. **Extensibility** – expose hooks for optional LLM-based canonicalisation without entangling core
   scoring logic.

### 한국어

키워드 추출 파이프라인은 KRISS/Leiden 프로세스에서 얻은 클러스터별 논문 제목·초록 정보를
대표 키워드로 요약합니다.  주요 활용 사례는 다음과 같습니다.

* 클러스터 라벨링 및 질적 해석
* 연도별 트렌드 분석 및 시각화
* 재현 가능한 지표를 활용한 비교 연구

설계 원칙은 다음과 같습니다.

1. **확장성** – Parquet 데이터를 스트리밍으로 읽어 희소 행렬 기반으로 처리합니다.
2. **설명 가능성** – 키워드별 빈도, 문서 커버리지, 연도별 추세를 함께 제공합니다.
3. **재현 가능성** – 스테이지별 저장 지점을 지원하여 중단 후 재개가 쉽습니다.
4. **확장성** – LLM 기반 정규화와 같은 고급 단계를 핵심 로직과 분리하여 연결할 수 있습니다.

---

## Problem Formulation & Notation / 문제 정의와 표기법

### English

Let

* \(D = \{d_1, \ldots, d_N\}\) be the set of documents (titles+abstracts).
* \(C = \{c_1, \ldots, c_K\}\) be the set of clusters.
* \(X \in \mathbb{R}^{N \times V}\) be the document–term count matrix
  produced by CountVectorizer.
* \(f_{c,j} = \sum_{d \in c} X_{d,j}\) be the total occurrences of term \(j\) in cluster \(c\).
* \(n_c = |\{d : d \in c\}|\) be the number of documents in cluster \(c\).
* \(df_{c,j} = |\{d \in c : X_{d,j} > 0\}|\) be the document coverage inside cluster \(c\).
* \(df_j = |\{c : f_{c,j} > 0\}|\) be the number of clusters where term \(j\) appears.

Class-based TF-IDF (c‑TF‑IDF) is defined by

\[
 tf_{c,j} = \frac{f_{c,j}}{\sum_{k} f_{c,k}}, \quad
 idf_j = \log \frac{1 + K}{1 + df_j} + 1, \quad
 score^{ctfidf}_{c,j} = tf_{c,j} \cdot idf_j.
\]

To quantify how distinctive a term is for cluster \(c\) relative to the rest of the corpus we compute the
log-likelihood ratio (LLR).  With contingency table entries

\[
 k_{11} = df_{c,j}, \quad k_{12} = n_c - df_{c,j}, \quad
 k_{21} = df_j - df_{c,j}, \quad k_{22} = N - n_c - k_{21},
\]

the LLR is

\[
 \mathrm{LLR}(k) = 2\sum_{i=1}^{4} k_i \log \frac{k_i}{e_i},\quad e_i \text{ is the expected count under independence}. 
\]

Scores are z-normalised within each cluster and combined via

\[
 score_{c,j} = w_{ctfidf} \cdot z( score^{ctfidf}_{c,j} ) + w_{llr} \cdot z( \mathrm{LLR}_{c,j} ).
\]

### 한국어

다음과 같이 기호를 정의합니다.

* \(D\) : 문서 집합, \(C\) : 클러스터 집합
* \(X\) : CountVectorizer로 얻은 문서–용어 희소 행렬
* \(f_{c,j}\) : 클러스터 \(c\)에서 용어 \(j\)의 등장 횟수
* \(n_c\) : 클러스터 \(c\)의 문서 수
* \(df_{c,j}\) : 클러스터 \(c\)에서 용어 \(j\)를 포함하는 문서 수
* \(df_j\) : 용어 \(j\)가 등장하는 클러스터 수

c‑TF‑IDF는 다음과 같이 계산합니다.

\(
 tf_{c,j} = \frac{f_{c,j}}{\sum_k f_{c,k}},\quad
 idf_j = \log \frac{1+K}{1+df_j} + 1,\quad
 score^{ctfidf}_{c,j} = tf_{c,j} \cdot idf_j.
\)

LLR은 클러스터 내·외부의 용어 출현을 비교하여 차별성을 측정하며, 위 공식과 동일한 방식으로
계산하고 z-정규화 후 가중합하여 최종 점수를 구합니다.

---

## Pipeline Architecture / 파이프라인 구조

### English

The pipeline contains four mandatory stages and one optional stage:

1. **Stage 0 – Vectoriser Fit**: stream documents, fit unigram & phrase CountVectorizers.
2. **Stage 1 – Aggregation**: transform batches, aggregate to cluster-term matrices, record coverage,
   apply minimum counts.
3. **Stage 2 – Scoring & Ranking**: compute c‑TF‑IDF, optional LLR, z-normalisation, diversity
   filtering (subphrase suppression + Jaccard-MMR).
4. **Stage 3 – Year Series**: re-vectorise selected terms only and build `{year: frequency}` histories.
5. **Stage 2.5 (Optional)**: canonicalise variants via LLM or dictionaries and recompute scores.

### 한국어

필수 스테이지 4개와 선택 스테이지 1개로 구성됩니다.

1. **Stage 0 – 벡터라이저 학습**
2. **Stage 1 – 집계** (문서 → 클러스터 희소행렬)
3. **Stage 2 – 스코어 계산 및 랭킹**
4. **Stage 3 – 연도별 추세 계산**
5. **Stage 2.5 – 정규화(선택)**

각 단계는 `KeywordExtractionPipeline` 메서드로 구현되어 있으며 필요 시 개별 호출이 가능합니다.

---

## Stage-by-Stage Theory / 스테이지별 이론 설명

### Stage 0 – Vectoriser Fit

*English* – CountVectorizer builds the vocabulary and analyser configuration.  We typically apply
`lowercase=True`, a custom `token_pattern`, and `strip_accents="unicode"` to normalise diacritics.  An
optional `title_weight` parameter repeats titles, emphasising concise phrases.

*한국어* – CountVectorizer로 어휘와 토큰화 규칙을 학습합니다.  `lowercase=True`, 사용자 정의
`token_pattern`, `strip_accents='unicode'`를 많이 사용하며 `title_weight`를 통해 제목을 반복하여 가중치를 부여합니다.

### Stage 1 – Aggregation

*English* – Instead of concatenating documents per cluster, we build sparse matrices by summing the
appropriate rows (`group_sum`).  This reduces memory usage and preserves token statistics.  Document
coverage (binary counts) is stored alongside raw frequencies for later filtering.

*한국어* – 문서를 하나로 붙이지 않고, 희소 행렬의 행을 클러스터별로 합산해 메모리 사용량을 크게 줄입니다.
동시에 문서 커버리지(해당 용어가 등장한 문서 수)를 저장하여 이후 필터링에 활용합니다.

### Stage 2 – Scoring & Ranking

*English* – We combine c‑TF‑IDF (cluster internal view) with LLR (global contrast).  MMR diversity
controls ensure the final list does not contain minor variants of the same phrase.  Coverage thresholds
remove keywords that appear in only a handful of documents.

*한국어* – c‑TF‑IDF로 클러스터 내부 중요도를 평가하고 LLR로 다른 클러스터와의 차이를 강화합니다.
MMR 기반 다양성 제어를 통해 유사 표현이 상위 리스트를 점유하는 것을 방지합니다.  커버리지 기준은
클러스터 내 문서 수에 따라 동적으로 조정할 수 있습니다.

### Stage 3 – Year Series

*English* – Only the selected top terms are re-vectorised, creating a compact `{year: count}` mapping per
cluster-term pair.  This enables plotting timelines or detecting emerging terms.

*한국어* – 최종 선택된 키워드만 재벡터화하여 `{연도: 빈도}` 정보를 계산합니다.  이를 이용해 트렌드 선
그래프를 그리거나 신흥 키워드를 탐지할 수 있습니다.

### Stage 2.5 – Canonicalisation (Optional)

See Section 7 for full details.  The core idea is to map surface forms to canonical equivalents and then
re-run Stage 2/3 with the merged statistics.

---

## Configuration Guidelines / 구성 가이드

| Parameter | Typical Range | Effect | 설명 |
|-----------|---------------|--------|------|
| `min_df_unigram` | 3–10 | Removes extremely rare noise | 드문 노이즈 제거 |
| `max_df_unigram` | 0.95–0.99 | Drops ubiquitous stopwords | 흔한 일반어 제거 |
| `min_df_phrase` | 3–10 | Ensures phrase robustness | 바이/트라이그램 안정성 확보 |
| `phrase_min_count_per_cluster` | 5–20 | Filters weak phrases post-aggregation | 집계 후 약한 구문 제거 |
| `min_cluster_doc_coverage` | 5–20 | Requires appearance across many docs | 여러 문서에 등장해야 유지 |
| `min_cluster_doc_coverage_ratio` | 0.005–0.02 | Scales coverage to cluster size | 클러스터 크기에 비례 |
| `mmr_jaccard_lambda` | 0.2–0.4 | Higher → more diversity | 값↑ → 중복 억제 강화 |
| `w_llr` | 0–0.5 | Emphasises between-cluster contrast | 클러스터 간 차별성 강조 |

Start with the baseline configuration in the README and adjust only when diagnostics in Section 9 suggest
it.

---

## Savepoints & Restart Strategy / 저장 지점과 재시작 전략

*English* – After each stage, persist artefacts (`joblib.dump` for vectorisers, `save_npz` for sparse
matrices, `to_parquet` for DataFrames).  When resuming, load the artefacts, assign them to the pipeline
attributes, and continue from the next stage.

*한국어* – 각 스테이지 이후 벡터라이저(`joblib.dump`), 희소행렬(`save_npz`), 결과 DataFrame(`to_parquet`)
을 저장합니다.  재실행 시 저장한 객체를 파이프라인 속성에 다시 연결하면 이어서 진행할 수 있습니다.

---

## Canonicalisation & LLM Post-processing / 정규화 및 LLM 후처리

1. Collect candidate metadata (term, score, frequency, coverage, optional context snippet).
2. Prompt the LLM to emit JSON mapping `original → canonical`, plus an action (`keep`, `merge`, `drop`).
3. Validate and build an alias table; log decisions for auditing.
4. Regenerate `C`, `DF`, and year-series matrices under canonical forms.
5. Re-run Stage 2/3 so the scores reflect merged variants.
6. Store provenance columns in the final table (`source_terms`, `correction_notes`).

LLM prompts should be deterministic (temperature 0) and you should keep fallback logic (use original term
if the response is malformed).

LLM 활용 시 위 순서를 따르며, 실패 시 원본 키워드를 사용하는 안전장치를 두는 것을 권장합니다.

---

## Worked Example Notebook / 예제 노트북 안내

Open `notebooks/toyexample.ipynb` to follow the entire pipeline on a 20-document synthetic corpus:

1. Build toy data and membership tables.
2. Execute each stage manually, saving artefacts after Stage 0 and Stage 1.
3. Inspect the Stage 2 output, simulate alias mappings, compute Stage 3 year-series.
4. Export final keywords and Top-3 per cluster.

해당 노트북을 실행하면 작은 데이터셋으로 전체 흐름과 저장/재시작 방법을 연습할 수 있습니다.

---

## Diagnostics & Quality Checks / 진단과 품질 점검

| Check | What to look for | 대응 |
|-------|------------------|------|
| Coverage histogram | Terms with very low coverage | Raise `min_cluster_doc_coverage` or ratio |
| LLR distribution | Many zeros? terms are generic | Increase `phrase_min_count_per_cluster`, tune `max_df_*` |
| Year-series sparsity | Terms appearing in one year only | Decide if seasonal behaviour is acceptable |
| Redundancy ratio | Many similar terms? | Increase `mmr_jaccard_lambda` or use canonicalisation |

### Debugging Tips / 디버깅 노하우

1. Inspect `pipeline.C_uni` and `pipeline.DF_uni` to ensure aggregation succeeded.
2. Use `term_year_to_long()` to plot timelines – a flat line may indicate generic terms.
3. Compare raw c‑TF‑IDF and LLR contributions to fine-tune weights.

---

## Appendix: Pseudocode & References / 부록: 의사코드와 참고자료

### Full Pipeline Pseudocode

```
pipe = KeywordExtractionPipeline(cfg)
pipe._fit_vectorizers()
pipe._aggregate_counts()
top_df = pipe._stage_scores_and_topk()
if cfg.enable_alias_stage:
    top_df = apply_alias_map(top_df, pipe)
term_year = pipe._compute_year_series(top_df)
final_df = attach_year_series(top_df, term_year)
```

### References / 참고 문헌

1. D. Angelov, “Top2Vec: Distributed Representations of Topics,” 2020.
2. P. Rayson and R. Garside, “Comparing Corpora Using Frequency Profiling,” 2000.
3. J. Carbonell and J. Goldstein, “The Use of MMR, Diversity-Based Reranking,” 1998.

---

Happy analysing! 즐거운 분석 되세요!
