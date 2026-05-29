# Code Review: CPM-Density Dendrogram Implementation

**Date**: 2026-03-16
**Reviewers**: Internal (Claude), External agent
**Status**: Phase 1~7 전부 완료

---

## Overview

전체 구조와 알고리즘 방향은 좋음. Deterministic tie-break, merge 기록 형식 깔끔.
초기 핵심 문제: 구현보다 주장이 더 앞서 있음 → 수정 완료.

---

## Phase 1 — 정확성 (Correctness) ✅

- **A1** `alive` 초기화: `vec![false; 2*n]` + leaf만 true + `debug_assert` ✅
- **A2** Shared mutable object: list comprehension ✅
- **A3** NaN heap corruption: `f64::total_cmp` + 입력 검증 ✅

## Phase 2 — 의미론 (Semantics) ✅

- **B1** similarity/distance 분리: `to_similarity_linkage_flat()` / `to_scipy_distance_flat()` ✅
- **B2** 명명: `CPM-critical` → `greedy CPM-density` ✅
- **B3** 복잡도: `O(n² worst-case)` 명시, TODO for union-by-size ✅
- **B4** Triadic semantics: doc 경고 + strategy enum (`mode="cpm"|"triadic_cpm"`) ✅
- **B5** Persistence: similarity linkage 전용 명시 ✅

## Phase 3 — 입력 검증 ✅

- **C1** vertex ID bounds → PyValueError ✅
- **C2** NaN/Inf/negative weight → PyValueError (Rust + Python 양쪽) ✅
- **C3** array length mismatch → PyValueError ✅
- **C4** n=0 → PyValueError ✅
- **C5** directed graph → ValueError ✅
- **C6** duplicate edge: accumulate (doc 명시) ✅

## Phase 4 — 설계 이슈 ✅

- **D1** GCC 전용: disconnected graph warning 추가, doc에 GCC 권장 명시 ✅
- **D2** `CutResult.feasible` 필드 추가 ✅
- **D3** strategy enum: `mode="cpm"|"triadic_cpm"` (lib.rs + dendrogram.py) ✅
- **D4** Cargo.toml `crate-type = ["cdylib", "lib"]` ✅

## Phase 5 — 테스트 보강 ✅

- **E1** counterexample 테스트 → `test_greedy_vs_cpm_optimum_divergence` (진짜 반례) ✅
- **E2** property-based test → `test_brute_force_agreement_small` (brute-force 비교) ✅
- **E3** `alive` 최종 count assertion ✅
- **E4** edge cases: single-node, isolated vertices, disconnected dendrogram, feasibility ✅
- **E5** disconnected + constrained_cut 통합 테스트 ✅

## Phase 6 — 성능 ✅

- **F1** Disconnected fallback: `BTreeSet<usize>` alive_set으로 O(1) fallback ✅
- **F2** parent_map: `Dict[int, float]` → `np.ndarray` 직접 인덱싱 ✅
- **F4** Triadic reverse-edge: linear scan → binary search ✅
- **F5** lib.rs: `Vec<Vec<f64>>` → `PyArray1::from_vec` + `.reshape([n, 4])` ✅

## Phase 7 — 정리 ✅

- Dead code 제거 (`_persistence()`, recursive `_collect_leaves()`) ✅
- `.gitignore`에 `research/auxiliary/cpm-dendro/target/` 추가 ✅
- `__init__.py` export 등록 ✅
- `// SAFETY:` 코멘트 추가 ✅
- Stale entry tolerance 단순화 (재계산 density 직접 사용) ✅

---

## 설계 결정 (2026-03-16)

1. **C6 duplicate edge**: 누적 (accumulate) — co-citation 등에서 자연스러움
2. **D1 disconnected**: GCC만 대상. Disconnected 입력 시 warning
3. **D3 mode**: strategy enum (`"cpm"`, `"triadic_cpm"`) — 향후 확장 대비

---

## 테스트 결과

- **Rust**: 17 tests passed, 0 warnings (hac 10, dendrogram 2, graph 2, triadic 3)
- **Python**: 21 tests passed (constrained_cut)
