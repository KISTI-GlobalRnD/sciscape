# Implementation Plan: CPM-Critical Dendrogram

## Implementation Strategy

### Core HAC: Rust + PyO3

100K 노드 × 3.28M 엣지에서 Python HAC는 1-2시간+ 소요.
반복 실험이 필요한 연구 단계에서 비현실적.

**Rust + PyO3 선택 이유:**
- C++ 동등 성능 (100K 노드 HAC: 1-3분)
- 메모리 안전 (컴파일 타임 보장, segfault 불가)
- cargo 빌드 (CMake 지옥 회피)
- maturin으로 Python wheel 원커맨드 빌드
- rayon으로 안전한 병렬화 가능
- cargo.lock으로 완전한 빌드 재현성

### 아키텍처: 분리된 crate + Python thin wrapper

```
sciscape/clustering/
├── dendrogram.py          # Python thin wrapper (igraph ↔ Rust 변환)
├── constrained_cut.py     # Pure Python O(n) DP (Rust 불필요)
└── ...

cpm-dendro/                 # 독립 Rust crate
├── Cargo.toml
├── pyproject.toml          # maturin 빌드 설정
├── src/
│   ├── lib.rs              # PyO3 모듈 진입점
│   ├── graph.rs            # CSR sparse graph
│   ├── hac.rs              # Core HAC (lazy max-heap)
│   ├── dendrogram.rs       # Dendrogram 구조체 + linkage matrix
│   └── triadic.rs          # Triadic closure 전처리
└── tests/
    ├── test_hac.rs
    └── test_triadic.rs
```

### Size-Constrained Cut: Pure Python

O(n) DP, 100K에서 밀리초. Rust 불필요. scipy linkage matrix 입력 → partition 출력.

---

## Phase 1: constrained_cut.py (Pure Python)

가장 간단하고 독립적. Dendrogram 없이도 테스트 가능 (linkage matrix 직접 생성).

### API

```python
def constrained_cut(
    linkage: np.ndarray,      # scipy-format (n-1, 4) linkage matrix
    min_size: int,            # minimum cluster size k
) -> CutResult:
    """Maximize #clusters s.t. all clusters ≥ min_size.
    Lexicographic: max(count, total_stability). O(n)."""
```

---

## Phase 2-5: cpm-dendro Rust crate

### Python에서 사용법

```python
from sciscape.clustering.dendrogram import build_dendrogram, constrained_cut

linkage = build_dendrogram(graph, triadic=True)  # Rust (1-3분)
result = constrained_cut(linkage, min_size=1000)  # Python (밀리초)
```

### Rust 핵심 알고리즘: Lazy Max-Heap Sparse HAC

```
1. 초기화: 각 노드 = 싱글톤 클러스터
   각 edge (u,v,w)에 대해 heap.push(ρ=w, u, v)

2. n-1번 반복:
   a. heap.pop() → (ρ, a, b)
      - a 또는 b가 dead → skip
      - ρ ≠ current ρ(a,b) → skip (stale)
   b. 새 클러스터 c = merge(a, b), height = ρ
   c. 모든 neighbor d에 대해:
      e_cd = e_ad + e_bd
      ρ_new = e_cd / (size[c] · size[d])
      heap.push(ρ_new, c, d)
   d. a, b를 dead로 표시, c를 alive에 추가

3. Return linkage matrix
```

### 빌드

```bash
cd cpm-dendro && maturin develop --release
```

---

## 구현 순서

| Phase | 내용 | 산출물 |
|-------|------|--------|
| **1** | constrained_cut.py + 테스트 | Pure Python DP |
| **2** | cpm-dendro crate 초기화 + graph.rs | Rust 프로젝트 구조 |
| **3** | hac.rs + dendrogram.rs + Rust 테스트 | Core 알고리즘 |
| **4** | PyO3 binding + dendrogram.py wrapper | Python 통합 |
| **5** | triadic.rs | 전처리 |
| **6** | KRISS 100K 실행 | **38개 ceiling 돌파 검증** |
| **7** | Paris 비교, 전체 실험 | 논문 실험 |

---

## 테스트 계획

### constrained_cut.py 단위 테스트
- test_trivial_cut: k=1 → all leaves
- test_k_equals_n: k=n → single cluster
- test_balanced_tree: known optimal
- test_stability_tiebreak: equal count → higher stability
- test_varying_k: multiple k on same tree
- test_four_node_counterexample

### Rust 단위 테스트 (cpm-dendro)
- test_two_nodes: 1 merge
- test_triangle: 3 nodes, verify merge order
- test_merge_heights_monotonic: dendrogram validity
- test_weighted_edges: non-unit weights
- test_disconnected: components merge last
- test_deterministic: same input → same output

### 통합 테스트
- test_dendrogram_pipeline: graph → dendrogram → cut → partition
- test_ceiling_breaking: Leiden 38개 vs dendrogram+cut

---

## 의존성

### sciscape (Python)
- igraph, numpy, scipy (기존)

### cpm-dendro (Rust)
- pyo3 (Python binding)
- numpy (PyO3 numpy interop)
- maturin (빌드)

### 비교 실험용 (선택)
- scikit-network (Paris baseline)
