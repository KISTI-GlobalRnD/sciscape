"""γ 보존 테스트 v3: self-loop 제거 후 정확한 등식 검증.

수학:
  H_orig(P) = H_contracted(P) + C,  C = Σ_B e_B - γ·Σ_B C(s_B,2)

  But wait — leidenalg with node_sizes uses penalty = γ·C(N_c,2) where N_c = Σ s_i
  So C(N_c,2) = Σ_{i<j} s_i·s_j + Σ_i C(s_i,2)

  Without self-loops:
    H_contracted = Σ_c [E_inter_c - γ·C(N_c,2)]    # leidenalg
    H_orig       = Σ_c [(E_inter_c + Σ_B e_B) - γ·C(N_c,2)]
    → H_orig = H_contracted + Σ_B e_B               # constant is γ-independent!

검증: H_orig = H_contracted + Σ_B e_B (모든 γ에서 정확히 성립해야 함)
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
log = logging.getLogger("gamma_test")

# ── 데이터 로드 ──────────────────────────────────────────────
DATA = "/home/kimyoungjin06/Desktop/Workspace/1.4.4.Sciscape/data/linktype_edges/field_34"
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
total_weight = sum(g.es["weight"])

log.info(f"Graph: {g.vcount()} V, {g.ecount()} E, total_weight={total_weight:.2f}")

SEED = 42
opt = la.Optimiser()


def cpm_quality_manual(graph, membership, gamma):
    """수동 CPM 품질 계산."""
    clusters = {}
    for i, c in enumerate(membership):
        clusters.setdefault(c, []).append(i)
    total = 0.0
    weights = graph.es["weight"]
    for nodes in clusters.values():
        node_set = set(nodes)
        e_c = sum(
            weights[eid]
            for node in nodes
            for eid in graph.incident(node)
            if (edge := graph.es[eid]) is not None
            and (neighbor := edge.target if edge.source == node else edge.source) in node_set
            and node < neighbor
        )
        n_c = len(nodes)
        total += e_c - gamma * n_c * (n_c - 1) / 2
    return total


# ── Step 1: 블록 형성 ────────────────────────────────────────
GAMMA_BLOCK = 1e-2

log.info(f"\n{'='*60}")
log.info(f"Step 1: 블록 형성 (γ_block = {GAMMA_BLOCK})")
log.info(f"{'='*60}")

t0 = time.time()
part_block = la.CPMVertexPartition(g, resolution_parameter=GAMMA_BLOCK, weights=g.es["weight"])
opt.set_rng_seed(SEED)
opt.optimise_partition(part_block, n_iterations=-1)
block_mem = list(part_block.membership)
t_block = time.time() - t0

block_counts = Counter(block_mem)
n_blocks = len(block_counts)
singletons = sum(1 for s in block_counts.values() if s == 1)
log.info(f"  {n_blocks} blocks, {singletons} singletons, {t_block:.1f}s")
log.info(f"  Top 10: {sorted(block_counts.values(), reverse=True)[:10]}")

# ── Step 2: Contraction (self-loop 제거) ─────────────────────
log.info(f"\n{'='*60}")
log.info(f"Step 2: Contraction (self-loop 제거)")
log.info(f"{'='*60}")

unique_blocks = sorted(block_counts.keys())
block_remap = {b: i for i, b in enumerate(unique_blocks)}
block_mem_remapped = [block_remap[b] for b in block_mem]

node_sizes = [0] * n_blocks
for b in block_mem:
    node_sizes[block_remap[b]] += 1

# Contraction with self-loop REMOVAL
contracted = g.copy()
contracted.contract_vertices(block_mem_remapped)
contracted.simplify(combine_edges={"weight": "sum"}, multiple=True, loops=True)  # loops=True → 제거

inter_block_weight = sum(contracted.es["weight"])
intra_block_weight = total_weight - inter_block_weight  # = Σ_B e_B

log.info(f"  Contracted: {contracted.vcount()} V, {contracted.ecount()} E")
log.info(f"  Reduction: {g.vcount()}/{contracted.vcount()} = {g.vcount()/contracted.vcount():.1f}x")
log.info(f"  node_sizes: min={min(node_sizes)}, max={max(node_sizes)}")
log.info(f"")
log.info(f"  Total weight:       {total_weight:>14.2f}")
log.info(f"  Inter-block weight: {inter_block_weight:>14.2f}")
log.info(f"  Intra-block weight: {intra_block_weight:>14.2f}  ← 상수 C = Σ_B e_B")
log.info(f"  Intra-block 비율:   {intra_block_weight/total_weight*100:.1f}%")

# ── Step 3: 등식 검증 ─────────────────────────────────────────
# H_orig(P) = H_contracted(P) + C  where C = Σ_B e_B (γ-independent)

log.info(f"\n{'='*60}")
log.info(f"Step 3: 등식 검증  H_orig = H_contracted + C")
log.info(f"  C (상수) = {intra_block_weight:.4f}")
log.info(f"{'='*60}")

log.info(f"\n  {'γ':>10s} | {'H_orig':>12s} | {'H_contr':>12s} | {'H_c + C':>12s} | {'|차이|':>10s} | {'클러스터':>8s}")
log.info(f"  {'-'*10}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}-+-{'-'*8}")

for gamma_test in [5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5]:
    # Contracted + node_sizes → 최적 파티션
    p_c = la.CPMVertexPartition(
        contracted, resolution_parameter=gamma_test,
        weights=contracted.es["weight"], node_sizes=node_sizes,
    )
    opt.set_rng_seed(SEED)
    opt.optimise_partition(p_c, n_iterations=-1)
    q_c = p_c.quality()
    mem_c = list(p_c.membership)

    # 같은 파티션을 원래 그래프에서 수동 계산
    mem_expanded = [mem_c[block_remap[block_mem[i]]] for i in range(n_nodes)]
    q_orig = cpm_quality_manual(g, mem_expanded, gamma_test)

    # 등식 검증
    diff = abs(q_orig - (q_c + intra_block_weight))
    n_cl = len(set(mem_c))

    log.info(f"  {gamma_test:>10.1e} | {q_orig:>12.2f} | {q_c:>12.2f} | {q_c + intra_block_weight:>12.2f} | {diff:>10.6f} | {n_cl:>8d}")


# ── Step 4: 최적성 비교 ───────────────────────────────────────
log.info(f"\n{'='*60}")
log.info(f"Step 4: 최적성 비교 (contracted vs 직접)")
log.info(f"{'='*60}")

log.info(f"\n  {'γ':>10s} | {'Q_direct':>12s} | {'Q_via_C':>12s} | {'Gap':>10s} | {'Gap%':>7s} | {'NMI':>6s} | {'#D':>4s} | {'#C':>4s} | {'t_D':>6s} | {'t_C':>6s}")
log.info(f"  {'-'*10}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}-+-{'-'*7}-+-{'-'*6}-+-{'-'*4}-+-{'-'*4}-+-{'-'*6}-+-{'-'*6}")

for gamma_test in [5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5]:
    # (D) 원래 그래프 직접
    t0 = time.time()
    p_d = la.CPMVertexPartition(g, resolution_parameter=gamma_test, weights=g.es["weight"])
    opt.set_rng_seed(SEED)
    opt.optimise_partition(p_d, n_iterations=-1)
    q_d = p_d.quality()
    mem_d = list(p_d.membership)
    t_d = time.time() - t0

    # (C) Contracted + node_sizes
    t0 = time.time()
    p_c = la.CPMVertexPartition(
        contracted, resolution_parameter=gamma_test,
        weights=contracted.es["weight"], node_sizes=node_sizes,
    )
    opt.set_rng_seed(SEED)
    opt.optimise_partition(p_c, n_iterations=-1)
    q_c = p_c.quality()
    mem_c = list(p_c.membership)
    t_c = time.time() - t0

    # Contracted → 원래 기준 quality
    q_via = q_c + intra_block_weight
    gap = q_d - q_via
    gap_pct = gap / abs(q_d) * 100 if q_d != 0 else 0

    mem_exp = [mem_c[block_remap[block_mem[i]]] for i in range(n_nodes)]
    nmi = normalized_mutual_info_score(mem_d, mem_exp)

    log.info(f"  {gamma_test:>10.1e} | {q_d:>12.2f} | {q_via:>12.2f} | {gap:>10.2f} | {gap_pct:>6.2f}% | {nmi:>6.3f} | {len(set(mem_d)):>4d} | {len(set(mem_c)):>4d} | {t_d:>5.1f}s | {t_c:>5.1f}s")


# ── Step 5: node_sizes 대조군 ─────────────────────────────────
log.info(f"\n{'='*60}")
log.info(f"Step 5: node_sizes 유무 비교")
log.info(f"{'='*60}")

log.info(f"\n  {'γ':>10s} | {'Q_sizes':>12s} | {'Q_no_sizes':>12s} | {'NMI_s':>6s} | {'NMI_ns':>7s} | {'#cl_s':>5s} | {'#cl_ns':>6s}")
log.info(f"  {'-'*10}-+-{'-'*12}-+-{'-'*12}-+-{'-'*6}-+-{'-'*7}-+-{'-'*5}-+-{'-'*6}")

for gamma_test in [5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5]:
    # 원래 기준 (ground truth)
    p_d = la.CPMVertexPartition(g, resolution_parameter=gamma_test, weights=g.es["weight"])
    opt.set_rng_seed(SEED)
    opt.optimise_partition(p_d, n_iterations=-1)
    mem_d = list(p_d.membership)

    # With node_sizes
    p_w = la.CPMVertexPartition(
        contracted, resolution_parameter=gamma_test,
        weights=contracted.es["weight"], node_sizes=node_sizes,
    )
    opt.set_rng_seed(SEED)
    opt.optimise_partition(p_w, n_iterations=-1)
    mem_w_exp = [list(p_w.membership)[block_remap[block_mem[i]]] for i in range(n_nodes)]
    q_w = cpm_quality_manual(g, mem_w_exp, gamma_test)

    # Without node_sizes
    p_wo = la.CPMVertexPartition(
        contracted, resolution_parameter=gamma_test,
        weights=contracted.es["weight"],
    )
    opt.set_rng_seed(SEED)
    opt.optimise_partition(p_wo, n_iterations=-1)
    mem_wo_exp = [list(p_wo.membership)[block_remap[block_mem[i]]] for i in range(n_nodes)]
    q_wo = cpm_quality_manual(g, mem_wo_exp, gamma_test)

    nmi_w = normalized_mutual_info_score(mem_d, mem_w_exp)
    nmi_wo = normalized_mutual_info_score(mem_d, mem_wo_exp)

    log.info(f"  {gamma_test:>10.1e} | {q_w:>12.2f} | {q_wo:>12.2f} | {nmi_w:>6.3f} | {nmi_wo:>7.3f} | {len(set(p_w.membership)):>5d} | {len(set(p_wo.membership)):>6d}")


# ── Step 6: γ > γ_block 위반 테스트 ──────────────────────────
log.info(f"\n{'='*60}")
log.info(f"Step 6: γ > γ_block 위반 테스트 (γ_block = {GAMMA_BLOCK})")
log.info(f"  γ > γ_block이면 블록을 쪼개야 하지만 contraction이 막음")
log.info(f"{'='*60}")

log.info(f"\n  {'γ':>10s} | {'Q_direct':>12s} | {'Q_via_C':>12s} | {'Gap':>10s} | {'Gap%':>7s} | {'NMI':>6s} | {'#D':>4s} | {'#C':>4s}")
log.info(f"  {'-'*10}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}-+-{'-'*7}-+-{'-'*6}-+-{'-'*4}-+-{'-'*4}")

for gamma_test in [1e-2, 2e-2, 5e-2, 1e-1]:
    # 원래 그래프 직접
    p_d = la.CPMVertexPartition(g, resolution_parameter=gamma_test, weights=g.es["weight"])
    opt.set_rng_seed(SEED)
    opt.optimise_partition(p_d, n_iterations=-1)
    q_d = p_d.quality()
    mem_d = list(p_d.membership)

    # Contracted (블록 쪼개기 불가)
    p_c = la.CPMVertexPartition(
        contracted, resolution_parameter=gamma_test,
        weights=contracted.es["weight"], node_sizes=node_sizes,
    )
    opt.set_rng_seed(SEED)
    opt.optimise_partition(p_c, n_iterations=-1)
    q_c = p_c.quality()
    mem_c = list(p_c.membership)

    q_via = q_c + intra_block_weight
    gap = q_d - q_via
    gap_pct = gap / abs(q_d) * 100 if q_d != 0 else 0

    mem_exp = [mem_c[block_remap[block_mem[i]]] for i in range(n_nodes)]
    nmi = normalized_mutual_info_score(mem_d, mem_exp)

    log.info(f"  {gamma_test:>10.1e} | {q_d:>12.2f} | {q_via:>12.2f} | {gap:>10.2f} | {gap_pct:>6.2f}% | {nmi:>6.3f} | {len(set(mem_d)):>4d} | {len(set(mem_c)):>4d}")

log.info(f"\n{'='*60}")
log.info("완료!")
log.info(f"{'='*60}")
