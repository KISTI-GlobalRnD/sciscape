"""Hot start 테스트: contracted 결과를 initial_membership으로 사용.

비교:
  (A) Cold start: 원래 그래프에서 직접 Leiden (기본)
  (B) Contracted only: contracted graph 결과를 확장 (보정 없음)
  (C) Hot start: contracted 결과를 initial_membership으로 원래 그래프에서 재실행
"""

import logging
import time
from collections import Counter

import igraph as ig
import leidenalg as la
import numpy as np
import polars as pl
from sklearn.metrics import normalized_mutual_info_score

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("hot")

# ── 데이터 로드 ──────────────────────────────────────────────
DATA = "/home/kimyoungjin06/Desktop/Workspace/1.4.4.Sciscape/workspace/data/linktype_edges/field_34"
df = pl.read_parquet(f"{DATA}/bc_raw.parquet")

src_arr = df["src"].to_list()
dst_arr = df["dst"].to_list()
wt_arr = df["weight"].to_list()
all_nodes = sorted(set(src_arr) | set(dst_arr))
node_map = {n: i for i, n in enumerate(all_nodes)}
n_nodes = len(all_nodes)

g = ig.Graph(n=n_nodes, directed=False)
g.add_edges([(node_map[s], node_map[d]) for s, d in zip(src_arr, dst_arr)])
g.es["weight"] = [float(w) for w in wt_arr]

log.info(f"Graph: {g.vcount()} V, {g.ecount()} E")

SEED = 42
opt = la.Optimiser()


def run_leiden(graph, gamma, seed=SEED, initial_membership=None, node_sizes=None):
    """Leiden 실행 후 (membership, quality, time) 반환."""
    kwargs = dict(resolution_parameter=gamma, weights=graph.es["weight"])
    if initial_membership is not None:
        kwargs["initial_membership"] = list(initial_membership)
    if node_sizes is not None:
        kwargs["node_sizes"] = list(node_sizes)
    part = la.CPMVertexPartition(graph, **kwargs)
    opt.set_rng_seed(seed)
    t0 = time.time()
    opt.optimise_partition(part, n_iterations=-1)
    elapsed = time.time() - t0
    return list(part.membership), part.quality(), elapsed


# ── Step 1: 블록 형성 + Contraction ──────────────────────────
GAMMA_BLOCK = 1e-2

log.info(f"\n블록 형성 (γ_block = {GAMMA_BLOCK})...")
block_mem, q_block, t_block = run_leiden(g, GAMMA_BLOCK)
block_counts = Counter(block_mem)
n_blocks = len(block_counts)
singletons = sum(1 for s in block_counts.values() if s == 1)
log.info(f"  {n_blocks} blocks ({singletons} singletons), {t_block:.1f}s")

# Contraction
unique_blocks = sorted(block_counts.keys())
block_remap = {b: i for i, b in enumerate(unique_blocks)}
block_mem_remapped = [block_remap[b] for b in block_mem]
node_sizes = [0] * n_blocks
for b in block_mem:
    node_sizes[block_remap[b]] += 1

contracted = g.copy()
contracted.contract_vertices(block_mem_remapped)
contracted.simplify(combine_edges={"weight": "sum"}, multiple=True, loops=True)

log.info(f"  Contracted: {contracted.vcount()} V, {contracted.ecount()} E")

# ── Step 2: 비교 테스트 ──────────────────────────────────────
log.info(f"\n{'='*70}")
log.info(f"{'γ':>10s} | {'방법':>15s} | {'Quality':>14s} | {'#cl':>5s} | {'Time':>7s} | {'NMI_vs_A':>9s} | {'NMI_vs_B':>9s}")
log.info(f"{'-'*10}-+-{'-'*15}-+-{'-'*14}-+-{'-'*5}-+-{'-'*7}-+-{'-'*9}-+-{'-'*9}")

for gamma_test in [5e-3, 2e-3, 1e-3, 5e-4, 2e-4]:
    # (A) Cold start: 원래 그래프에서 직접
    mem_a, q_a, t_a = run_leiden(g, gamma_test)

    # (B) Contracted only
    mem_c, q_c, t_c = run_leiden(contracted, gamma_test, node_sizes=node_sizes)
    mem_b = [mem_c[block_remap[block_mem[i]]] for i in range(n_nodes)]
    # 원래 그래프 기준 quality
    part_b_check = la.CPMVertexPartition(g, resolution_parameter=gamma_test,
                                          weights=g.es["weight"], initial_membership=mem_b)
    q_b = part_b_check.quality()

    # (C) Hot start: contracted 결과를 초기값으로
    mem_hot, q_hot, t_hot = run_leiden(g, gamma_test, initial_membership=mem_b)

    # (D) Multi-seed cold start (5 seeds 중 best)
    best_q_d = -1e18
    best_mem_d = None
    t_d_total = 0
    for s in [42, 123, 456, 789, 1024]:
        mem_d, q_d, t_d = run_leiden(g, gamma_test, seed=s)
        t_d_total += t_d
        if q_d > best_q_d:
            best_q_d = q_d
            best_mem_d = mem_d

    # NMI 계산
    nmi_ba = normalized_mutual_info_score(mem_a, mem_b)
    nmi_ca = normalized_mutual_info_score(mem_a, mem_hot)
    nmi_cb = normalized_mutual_info_score(mem_b, mem_hot)
    nmi_da = normalized_mutual_info_score(mem_a, best_mem_d)

    log.info(f"  {gamma_test:>10.1e} | {'(A) Cold':>15s} | {q_a:>14.2f} | {len(set(mem_a)):>5d} | {t_a:>6.1f}s | {'  -':>9s} | {nmi_ba:>9.3f}")
    log.info(f"  {gamma_test:>10.1e} | {'(B) Contracted':>15s} | {q_b:>14.2f} | {len(set(mem_b)):>5d} | {t_c:>6.1f}s | {nmi_ba:>9.3f} | {'  -':>9s}")
    log.info(f"  {gamma_test:>10.1e} | {'(C) Hot start':>15s} | {q_hot:>14.2f} | {len(set(mem_hot)):>5d} | {t_hot:>6.1f}s | {nmi_ca:>9.3f} | {nmi_cb:>9.3f}")
    log.info(f"  {gamma_test:>10.1e} | {'(D) Best of 5':>15s} | {best_q_d:>14.2f} | {len(set(best_mem_d)):>5d} | {t_d_total:>6.1f}s | {nmi_da:>9.3f} | {'':>9s}")
    log.info(f"  {'-'*10}-+-{'-'*15}-+-{'-'*14}-+-{'-'*5}-+-{'-'*7}-+-{'-'*9}-+-{'-'*9}")

log.info(f"\n완료!")
