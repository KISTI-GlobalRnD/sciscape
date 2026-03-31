"""축퇴 탈출 전략 비교 (γ=2e-4에서 hot start가 갇히는 문제).

전략:
  (A) Cold start (baseline)
  (B) Contracted only
  (C) Hot start (단일 시드) — 갇힘
  (D) Multi-seed hot start: contracted 결과 + 여러 시드
  (E) Cascade: γ=5e-4 결과를 initial로 γ=2e-4 실행
  (F) Multi-seed cascade
  (G) Contracted multi-seed → best를 hot start
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
log = logging.getLogger("escape")

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

log.info(f"Graph: {g.vcount()} V, {g.ecount()} E")

SEED = 42
SEEDS = [42, 123, 456, 789, 1024, 2048, 3333, 7777, 9999, 12345]
opt = la.Optimiser()
TARGET_GAMMA = 2e-4


def run_leiden(graph, gamma, seed=SEED, initial_membership=None, node_sizes=None):
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


# ── 블록 형성 + Contraction ───────────────────────────────────
GAMMA_BLOCK = 1e-2

log.info(f"블록 형성 (γ={GAMMA_BLOCK})...")
block_mem, _, t_block = run_leiden(g, GAMMA_BLOCK)
block_counts = Counter(block_mem)
n_blocks = len(block_counts)

unique_blocks = sorted(block_counts.keys())
block_remap = {b: i for i, b in enumerate(unique_blocks)}
block_mem_remapped = [block_remap[b] for b in block_mem]
node_sizes = [0] * n_blocks
for b in block_mem:
    node_sizes[block_remap[b]] += 1

contracted = g.copy()
contracted.contract_vertices(block_mem_remapped)
contracted.simplify(combine_edges={"weight": "sum"}, multiple=True, loops=True)

log.info(f"  {n_blocks} blocks → {contracted.vcount()} V, {contracted.ecount()} E")


def expand(mem_contracted):
    """Contracted membership → 원래 노드 membership."""
    return [mem_contracted[block_remap[block_mem[i]]] for i in range(n_nodes)]


def quality_on_orig(membership, gamma=TARGET_GAMMA):
    """원래 그래프에서 파티션의 quality."""
    p = la.CPMVertexPartition(g, resolution_parameter=gamma,
                               weights=g.es["weight"], initial_membership=membership)
    return p.quality()


# ── (A) Cold start baseline (multi-seed) ─────────────────────
log.info(f"\n{'='*70}")
log.info(f"Target γ = {TARGET_GAMMA}")
log.info(f"{'='*70}")

log.info(f"\n(A) Cold start — 10 seeds:")
cold_results = []
for s in SEEDS:
    mem, q, t = run_leiden(g, TARGET_GAMMA, seed=s)
    cold_results.append((mem, q, t, s))
    log.info(f"  seed={s:>5d}: Q={q:.2f}, #cl={len(set(mem)):>3d}, {t:.1f}s")

best_cold = max(cold_results, key=lambda x: x[1])
log.info(f"  ★ Best: seed={best_cold[3]}, Q={best_cold[1]:.2f}")


# ── (B) Contracted only (multi-seed) ─────────────────────────
log.info(f"\n(B) Contracted only — 10 seeds:")
contr_results = []
for s in SEEDS:
    mem_c, q_c, t_c = run_leiden(contracted, TARGET_GAMMA, seed=s, node_sizes=node_sizes)
    mem_exp = expand(mem_c)
    q_orig = quality_on_orig(mem_exp)
    contr_results.append((mem_exp, q_orig, t_c, s))
    log.info(f"  seed={s:>5d}: Q_orig={q_orig:.2f}, #cl={len(set(mem_c)):>3d}, {t_c:.3f}s")

best_contr = max(contr_results, key=lambda x: x[1])
log.info(f"  ★ Best: seed={best_contr[3]}, Q={best_contr[1]:.2f}")


# ── (C) Hot start 단일 ───────────────────────────────────────
log.info(f"\n(C) Hot start (contracted seed=42 → original seed=42):")
mem_c42 = contr_results[0][0]  # seed=42 contracted result
mem_hot, q_hot, t_hot = run_leiden(g, TARGET_GAMMA, initial_membership=mem_c42)
log.info(f"  Q={q_hot:.2f}, #cl={len(set(mem_hot)):>3d}, {t_hot:.1f}s")


# ── (D) Multi-seed hot start ─────────────────────────────────
log.info(f"\n(D) Multi-seed hot start (contracted seed=42 → 10 seeds on original):")
hot_results = []
for s in SEEDS:
    mem_h, q_h, t_h = run_leiden(g, TARGET_GAMMA, seed=s, initial_membership=mem_c42)
    hot_results.append((mem_h, q_h, t_h, s))
    log.info(f"  seed={s:>5d}: Q={q_h:.2f}, #cl={len(set(mem_h)):>3d}, {t_h:.1f}s")

best_hot = max(hot_results, key=lambda x: x[1])
log.info(f"  ★ Best: seed={best_hot[3]}, Q={best_hot[1]:.2f}")


# ── (E) Cascade: γ=5e-4 → γ=2e-4 ────────────────────────────
log.info(f"\n(E) Cascade: contracted@γ=5e-4 → hot start@γ=2e-4:")
# 먼저 contracted에서 γ=5e-4 해를 구함
mem_c_5e4, _, _ = run_leiden(contracted, 5e-4, node_sizes=node_sizes)
mem_5e4_exp = expand(mem_c_5e4)
# 이걸 γ=2e-4의 initial로
mem_cascade, q_cascade, t_cascade = run_leiden(g, TARGET_GAMMA, initial_membership=mem_5e4_exp)
log.info(f"  Q={q_cascade:.2f}, #cl={len(set(mem_cascade)):>3d}, {t_cascade:.1f}s")


# ── (F) Multi-seed cascade ───────────────────────────────────
log.info(f"\n(F) Multi-seed cascade (contracted@5e-4 → 10 seeds@2e-4):")
cascade_results = []
for s in SEEDS:
    # 각 시드로 contracted@5e-4
    mem_c5, _, _ = run_leiden(contracted, 5e-4, seed=s, node_sizes=node_sizes)
    mem_c5_exp = expand(mem_c5)
    # hot start@2e-4
    mem_cas, q_cas, t_cas = run_leiden(g, TARGET_GAMMA, seed=s, initial_membership=mem_c5_exp)
    cascade_results.append((mem_cas, q_cas, t_cas, s))
    log.info(f"  seed={s:>5d}: Q={q_cas:.2f}, #cl={len(set(mem_cas)):>3d}, {t_cas:.1f}s")

best_cascade = max(cascade_results, key=lambda x: x[1])
log.info(f"  ★ Best: seed={best_cascade[3]}, Q={best_cascade[1]:.2f}")


# ── (G) Contracted multi-seed → best hot start ───────────────
log.info(f"\n(G) Contracted multi-seed → best를 hot start:")
# 10개 contracted 결과 중 best를 hot start
mem_g, q_g, t_g = run_leiden(g, TARGET_GAMMA, initial_membership=best_contr[0])
log.info(f"  Q={q_g:.2f}, #cl={len(set(mem_g)):>3d}, {t_g:.1f}s")

# multi-seed로도
log.info(f"\n(G+) Best contracted → 10 seeds hot start:")
g_results = []
for s in SEEDS:
    mem_gp, q_gp, t_gp = run_leiden(g, TARGET_GAMMA, seed=s, initial_membership=best_contr[0])
    g_results.append((mem_gp, q_gp, t_gp, s))
g_best = max(g_results, key=lambda x: x[1])
log.info(f"  ★ Best: seed={g_best[3]}, Q={g_best[1]:.2f}")


# ── 최종 비교 ────────────────────────────────────────────────
log.info(f"\n{'='*70}")
log.info(f"최종 비교 @ γ = {TARGET_GAMMA}")
log.info(f"{'='*70}")

ref_q = best_cold[1]

results = [
    ("(A) Cold best/10", best_cold[1], sum(r[2] for r in cold_results)),
    ("(B) Contracted best/10", best_contr[1], sum(r[2] for r in contr_results)),
    ("(C) Hot start (1 seed)", q_hot, t_hot),
    ("(D) Hot start (10 seeds)", best_hot[1], sum(r[2] for r in hot_results)),
    ("(E) Cascade (1 seed)", q_cascade, t_cascade),
    ("(F) Cascade (10 seeds)", best_cascade[1], sum(r[2] for r in cascade_results)),
    ("(G) Best contr→hot (1)", q_g, t_g),
    ("(G+) Best contr→hot (10)", g_best[1], sum(r[2] for r in g_results)),
]

log.info(f"\n  {'방법':>30s} | {'Quality':>14s} | {'vs Best Cold':>12s} | {'Time':>8s}")
log.info(f"  {'-'*30}-+-{'-'*14}-+-{'-'*12}-+-{'-'*8}")

for name, q, t in results:
    diff = q - ref_q
    sign = "+" if diff >= 0 else ""
    log.info(f"  {name:>30s} | {q:>14.2f} | {sign}{diff:>11.2f} | {t:>7.1f}s")

# NMI matrix for best results
log.info(f"\n  NMI between best results:")
best_mems = {
    "A": best_cold[0],
    "B": best_contr[0],
    "D": best_hot[0],
    "F": best_cascade[0],
}
labels = list(best_mems.keys())
for i, l1 in enumerate(labels):
    for l2 in labels[i+1:]:
        nmi = normalized_mutual_info_score(best_mems[l1], best_mems[l2])
        log.info(f"    NMI({l1},{l2}) = {nmi:.3f}")
