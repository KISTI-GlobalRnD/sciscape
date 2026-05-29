"""Run the hierarchy postprocess field-expansion validation.

This runner extends the source-seed and next-level propagation validation from
the original field12/15/34 evidence set to additional GCC embedding fields.
New run artifacts are stored under ``field_expansion_runs`` while the published
tables are combined six-field outputs in the validation directory.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import sys

REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").exists()
)
SCRIPT_ROOT = REPO_ROOT / "research/consensus/scripts"
_SCRIPT_PATHS = [REPO_ROOT, SCRIPT_ROOT]
_SCRIPT_PATHS.extend(path for path in SCRIPT_ROOT.rglob("*") if path.is_dir())
for _script_path in reversed(_SCRIPT_PATHS):
    _script_path_str = str(_script_path)
    if _script_path_str not in sys.path:
        sys.path.insert(0, _script_path_str)


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import pyarrow.parquet as pq
import sciscape_leiden

from evaluate_hierarchy_postprocess import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RESULTS_DIR,
    _gini,
    _markdown_table,
    _normalized_entropy,
    _repo_path,
    _sample_configs,
    _write_table,
)
from run_hierarchy_postprocess_next_level import (  # noqa: E402
    _flatten_summary,
    _load_graph_arrays,
    _rel,
    _run_one,
    _safe_slug,
)
from run_hierarchy_postprocess_seed_sweep import _parse_int_list  # noqa: E402
from run_hierarchy_postprocess_source_seed_next_level import (  # noqa: E402
    _candidate_rows as _next_candidate_rows,
    _hard_cap_diagnostic_pairwise as _next_hard_cap_diagnostic_pairwise,
    _hard_cap_diagnostic_summary as _next_hard_cap_diagnostic_summary,
    _pairwise_summary as _next_pairwise_summary,
    _policy_summary as _next_policy_summary,
    _quality_first_pairwise as _next_quality_first_pairwise,
)
from run_hierarchy_postprocess_source_seed_sweep import (  # noqa: E402
    _diagnostic_summary as _source_diagnostic_summary,
    _hard_cap_diagnostic_pairwise as _source_hard_cap_diagnostic_pairwise,
    _metrics as _source_metrics,
    _policy_summary as _source_policy_summary,
    _quality_first_pairwise as _source_quality_first_pairwise,
    _run_source_seed,
)
from sciscape.clustering.integer_remap import integer_remap  # noqa: E402
from sciscape.clustering.leiden_rust import build_leiden_graph  # noqa: E402
from scripts.run_adaptive_split_merge_repair_probe import (  # noqa: E402
    _membership_weight_summary,
    _write_membership,
)

DEFAULT_FIELDS = (18, 26, 30)
DEFAULT_SEEDS = (11, 42, 73)
DEFAULT_POLICIES = ("two_stage_quality_first", "two_stage_hard_cap")
DEFAULT_EXISTING_SOURCE = DEFAULT_OUTPUT_DIR / "source_seed_sweep_effects.csv"
DEFAULT_EXISTING_NEXT = DEFAULT_OUTPUT_DIR / "source_seed_next_level_effects.csv"
DEFAULT_RUN_ROOT = DEFAULT_OUTPUT_DIR / "field_expansion_runs"
EXPECTED_FIELD_IDS = (12, 15, 18, 26, 30, 34)
UID1_COL = "uid1"
UID2_COL = "uid2"
WEIGHT_COL = "rel_sum2"

@dataclass(frozen=True)
class ExpansionConfig:
    field_id: int
    sample: str
    edge_path: Path
    sample_dir: Path
    graph_dir: Path
    prepare_summary_path: Path
    n_nodes: int
    n_edges: int
    target_min_doc_weight: float
    target_max_doc_weight: float
    resolution: float
    n_iterations: int

def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

def _field_sample(field_id: int) -> str:
    return f"field{int(field_id)}_gcc_emb_full_knn30"

def _field_edge_path(field_id: int) -> Path:
    return REPO_ROOT / "data" / "linktype_edges_gcc" / f"field_{int(field_id)}" / "emb_full_knn30.parquet"

def _target_max_doc_weight(n_nodes: int) -> float:
    return float(min(1500, max(750, round(0.0087 * int(n_nodes)))))

def _field_id_from_sample(sample: str) -> int | None:
    match = re.search(r"field_?(\d+)", str(sample))
    return int(match.group(1)) if match else None

def _scan_edge_n_nodes(edge_path: Path) -> int:
    edges = pl.scan_parquet(edge_path).select([UID1_COL, UID2_COL])
    nodes = pl.concat(
        [
            edges.select(pl.col(UID1_COL).alias("uid")),
            edges.select(pl.col(UID2_COL).alias("uid")),
        ],
        how="vertical",
    )
    return int(nodes.unique().select(pl.len()).collect().item())

def _edge_n_rows(edge_path: Path) -> int:
    return int(pq.ParquetFile(edge_path).metadata.num_rows)

def _write_node_weights(graph_dir: Path, n_nodes: int, *, force: bool) -> Path:
    path = graph_dir / "node_weights.f64.bin"
    expected_size = int(n_nodes) * np.dtype(np.float64).itemsize
    if path.exists() and path.stat().st_size == expected_size and not force:
        return path
    graph_dir.mkdir(parents=True, exist_ok=True)
    np.ones(int(n_nodes), dtype=np.float64).tofile(path)
    return path

def _cluster_weight_values(membership: np.ndarray, node_weights: np.ndarray) -> np.ndarray:
    membership_i64 = np.asarray(membership, dtype=np.int64)
    if membership_i64.size == 0:
        return np.asarray([], dtype=np.float64)
    weights = np.bincount(
        membership_i64,
        weights=np.asarray(node_weights, dtype=np.float64),
        minlength=int(membership_i64.max()) + 1,
    )
    counts = np.bincount(membership_i64, minlength=weights.shape[0])
    return weights[counts > 0]

def _write_weight_candidates(
    path: Path,
    membership: np.ndarray,
    node_weights: np.ndarray,
    *,
    policy: str,
    threshold: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    weights = np.bincount(np.asarray(membership, dtype=np.int64), weights=node_weights)
    counts = np.bincount(np.asarray(membership, dtype=np.int64), minlength=weights.shape[0])
    rows = [
        {
            "policy": policy,
            "cluster": int(cluster),
            "doc_weight": float(weight),
            "block_count": int(counts[cluster]),
        }
        for cluster, weight in enumerate(weights)
        if float(weight) > float(threshold)
    ]
    rows.sort(key=lambda row: (-row["doc_weight"], row["cluster"]))
    pd.DataFrame(rows, columns=["policy", "cluster", "doc_weight", "block_count"]).to_csv(
        path,
        index=False,
    )

def _config_from_summary(field_id: int, summary_path: Path) -> ExpansionConfig:
    summary = _read_json(summary_path)
    paths = summary.get("paths", {})
    graph_dir = _repo_path(paths.get("graph_dir")) or summary_path.parent / "graph"
    edge_path = _repo_path(summary.get("edge_path")) or _field_edge_path(field_id)
    assert graph_dir is not None and edge_path is not None
    return ExpansionConfig(
        field_id=int(field_id),
        sample=str(summary.get("sample") or _field_sample(field_id)),
        edge_path=edge_path,
        sample_dir=summary_path.parent,
        graph_dir=graph_dir,
        prepare_summary_path=summary_path,
        n_nodes=int(summary["n_nodes"]),
        n_edges=int(summary.get("n_edges") or 0),
        target_min_doc_weight=float(
            summary.get("target_min_doc_weight") or summary.get("min_size") or 50.0
        ),
        target_max_doc_weight=float(summary["target_max_doc_weight"]),
        resolution=float(summary.get("resolution") or 0.01),
        n_iterations=int(summary.get("n_iterations") or 5),
    )

def _prepare_field(
    *,
    field_id: int,
    run_root: Path,
    resolution: float,
    target_min_doc_weight: float,
    seed: int,
    n_iterations: int,
    force: bool,
) -> ExpansionConfig:
    edge_path = _field_edge_path(field_id)
    if not edge_path.exists():
        raise FileNotFoundError(f"Missing field edge parquet: {edge_path}")

    sample = _field_sample(field_id)
    sample_dir = run_root / sample
    graph_dir = sample_dir / "graph"
    summary_path = sample_dir / "prepare_summary.json"
    membership_path = sample_dir / "membership.parquet"
    node_weights_path = graph_dir / "node_weights.f64.bin"
    sidecars = (graph_dir / "src.u32.bin", graph_dir / "dst.u32.bin", graph_dir / "weight.f64.bin")
    if (
        summary_path.exists()
        and membership_path.exists()
        and node_weights_path.exists()
        and all(path.exists() for path in sidecars)
        and not force
    ):
        return _config_from_summary(field_id, summary_path)

    t0 = time.perf_counter()
    sample_dir.mkdir(parents=True, exist_ok=True)
    remap = integer_remap(
        edge_path,
        graph_dir,
        uid1_col=UID1_COL,
        uid2_col=UID2_COL,
        weight_col=WEIGHT_COL,
        overwrite=force,
        write_int_edges=True,
    )
    _write_node_weights(graph_dir, remap.n_nodes, force=force)
    target_max_doc_weight = _target_max_doc_weight(remap.n_nodes)

    src = np.memmap(graph_dir / "src.u32.bin", dtype=np.uint32, mode="r")
    dst = np.memmap(graph_dir / "dst.u32.bin", dtype=np.uint32, mode="r")
    weight = np.memmap(graph_dir / "weight.f64.bin", dtype=np.float64, mode="r")
    node_weights = np.memmap(node_weights_path, dtype=np.float64, mode="r")
    graph = build_leiden_graph(
        edges_src=src,
        edges_dst=dst,
        edges_weight=weight,
        n_nodes=int(remap.n_nodes),
        node_weights=np.asarray(node_weights, dtype=np.float64),
    )
    raw = graph.run_leiden(
        resolution=float(resolution),
        seed=int(seed),
        n_iterations=int(n_iterations),
    )
    post = graph.postprocess_small_clusters(
        resolution=float(resolution),
        min_size=int(target_min_doc_weight),
        membership=raw.membership,
        seed=int(seed),
        gamma_decay=0.5,
        max_rounds=5,
        use_greedy=True,
        use_component_merge=True,
    )
    membership = np.asarray(post.membership, dtype=np.uint64)
    _write_membership(membership_path, membership)
    large_candidates_path = sample_dir / "large_doc_weight_candidates.csv"
    oversize_candidates_path = sample_dir / "oversize_doc_weight_candidates.csv"
    node_weights_array = np.asarray(node_weights, dtype=np.float64)
    _write_weight_candidates(
        large_candidates_path,
        membership,
        node_weights_array,
        policy="large_doc_weight",
        threshold=float(target_min_doc_weight),
    )
    _write_weight_candidates(
        oversize_candidates_path,
        membership,
        node_weights_array,
        policy="oversize_doc_weight",
        threshold=float(target_max_doc_weight),
    )
    stats = _membership_weight_summary(
        membership,
        node_weights_array,
        min_weight=float(target_min_doc_weight),
        max_weight=float(target_max_doc_weight),
    )
    weights = _cluster_weight_values(membership, node_weights_array)
    summary = {
        "sample": sample,
        "field_id": int(field_id),
        "edge_path": _rel(edge_path),
        "n_nodes": int(remap.n_nodes),
        "n_edges": int(remap.n_edges),
        "resolution": float(resolution),
        "seed": int(seed),
        "n_iterations": int(n_iterations),
        "min_size": int(target_min_doc_weight),
        "target_min_doc_weight": float(target_min_doc_weight),
        "target_max_doc_weight": float(target_max_doc_weight),
        "raw_quality": float(raw.quality),
        "raw_n_clusters": int(raw.n_clusters),
        "post_quality": float(graph.cpm_quality(membership, resolution=float(resolution))),
        "post_n_clusters": int(stats["n_clusters"]),
        "post_max_cluster_size": float(stats["max_doc_weight"]),
        "post_n_clusters_gt_target_max": int(stats["n_above_max_doc_weight"]),
        "post_n_lt_min_size": int(stats["n_lt_min_doc_weight"]),
        "post_gini_doc_weight": _gini(weights),
        "post_entropy_doc_weight": _normalized_entropy(weights),
        "paths": {
            "graph_dir": _rel(graph_dir),
            "int_edges": _rel(remap.int_edges_path),
            "node_manifest": _rel(remap.node_manifest_path),
            "membership": _rel(membership_path),
            "large_candidates": _rel(large_candidates_path),
            "oversize_candidates": _rel(oversize_candidates_path),
        },
        "elapsed_sec": time.perf_counter() - t0,
    }
    _write_json(summary_path, summary)
    return _config_from_summary(field_id, summary_path)

def _dry_run_rows(fields: list[int], run_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for field_id in fields:
        edge_path = _field_edge_path(field_id)
        n_edges = _edge_n_rows(edge_path) if edge_path.exists() else 0
        n_nodes = _scan_edge_n_nodes(edge_path) if edge_path.exists() else 0
        rows.append(
            {
                "field_id": int(field_id),
                "sample": _field_sample(field_id),
                "edge_path": _rel(edge_path),
                "edge_exists": edge_path.exists(),
                "n_edges": n_edges,
                "n_nodes": n_nodes,
                "target_min_doc_weight": 50.0,
                "target_max_doc_weight": _target_max_doc_weight(n_nodes) if n_nodes else None,
                "sample_dir": _rel(run_root / _field_sample(field_id)),
                "graph_dir": _rel(run_root / _field_sample(field_id) / "graph"),
            }
        )
    return pd.DataFrame(rows)

def _run_new_source_seed_sweep(
    *,
    configs: list[ExpansionConfig],
    run_root: Path,
    seeds: list[int],
    policies: tuple[str, ...],
    apply_iterations: int,
    trim_max_moves_per_cluster: int,
    selection_singleton_budget: float,
    force: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total = len(configs) * len(seeds)
    index = 0
    for cfg in configs:
        for seed in seeds:
            index += 1
            print(
                f"[{index}/{total}] field-expansion source seed "
                f"{cfg.sample} seed={seed}",
                flush=True,
            )
            rows.extend(
                _run_source_seed(
                    sample=cfg.sample,
                    graph_dir=cfg.graph_dir,
                    output_dir=run_root,
                    resolution=cfg.resolution,
                    target_min_doc_weight=cfg.target_min_doc_weight,
                    target_max_doc_weight=cfg.target_max_doc_weight,
                    seed=int(seed),
                    n_iterations=cfg.n_iterations,
                    policies=policies,
                    apply_iterations=int(apply_iterations),
                    trim_max_moves_per_cluster=int(trim_max_moves_per_cluster),
                    selection_singleton_budget=float(selection_singleton_budget),
                    force=force,
                )
            )
    return pd.DataFrame(rows)

def _flatten_source_seed_next_summary(summary: dict[str, Any]) -> dict[str, Any]:
    row = _flatten_summary(summary)
    row["next_seed"] = row.pop("seed")
    return row

def _run_new_next_level(
    *,
    source_df: pd.DataFrame,
    configs: list[ExpansionConfig],
    run_root: Path,
    source_seeds: list[int],
    next_seeds: list[int],
    policies: set[str],
    include_diagnostics: bool,
    next_target_pct: float | None,
    next_target_min_pct: float,
    next_target_multiplier: float,
    next_min_doc_weight: float,
    force: bool,
) -> pd.DataFrame:
    source_seed_filter = set(source_seeds)
    rows = _next_candidate_rows(
        source_df,
        policies=policies,
        source_seeds=source_seed_filter,
        include_diagnostics=include_diagnostics,
    )
    config_by_sample = {cfg.sample: cfg for cfg in configs}
    graph_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    summaries: list[dict[str, Any]] = []
    total_runs = len(rows) * len(next_seeds)
    run_idx = 0
    for source_row in rows:
        sample = str(source_row["sample"])
        cfg = config_by_sample[sample]
        if sample not in graph_cache:
            print(f"loading field-expansion graph {sample}", flush=True)
            graph_cache[sample] = _load_graph_arrays(cfg.graph_dir)
        if next_target_pct is None:
            total_docs = float(cfg.n_nodes)
            current_target = float(source_row.get("target_max_doc_weight") or 0.0)
            current_target_pct = 100.0 * current_target / total_docs if total_docs else 0.0
            current_next_target_pct = max(
                float(next_target_min_pct),
                current_target_pct * float(next_target_multiplier),
            )
        else:
            current_next_target_pct = float(next_target_pct)

        for next_seed in next_seeds:
            run_idx += 1
            row = dict(source_row)
            row["rerun_run"] = f"{row['rerun_run']}_next_seed_{next_seed}"
            print(
                f"[{run_idx}/{total_runs}] field-expansion next-level "
                f"{sample} source_seed={row['source_seed']} "
                f"policy={row['rerun_policy']} next_seed={next_seed} "
                f"target_pct={current_next_target_pct:.4g}",
                flush=True,
            )
            summaries.append(
                _run_one(
                    row=row,
                    graph_arrays=graph_cache[sample],
                    graph_dir=cfg.graph_dir,
                    output_dir=run_root,
                    next_target_pct=current_next_target_pct,
                    next_min_doc_weight=float(next_min_doc_weight),
                    seed=int(next_seed),
                    force=force,
                    runs_dir_name="source_seed_next_level_runs",
                )
            )
    return pd.DataFrame([_flatten_source_seed_next_summary(summary) for summary in summaries])

def _load_csv(path: Path) -> pd.DataFrame:
    resolved = _repo_path(path)
    assert resolved is not None
    if not resolved.exists():
        raise FileNotFoundError(f"Missing required CSV: {resolved}")
    return pd.read_csv(resolved)

def _write_source_outputs(
    *,
    combined: pd.DataFrame,
    validation_dir: Path,
    prefix: str,
) -> dict[str, pd.DataFrame]:
    policy_summary = _source_policy_summary(combined)
    pairwise = _source_quality_first_pairwise(combined)
    hard_cap_pairwise = _source_hard_cap_diagnostic_pairwise(combined)
    hard_cap_summary = _source_diagnostic_summary(hard_cap_pairwise)
    _write_table(combined, validation_dir / f"{prefix}_source_seed_effects")
    _write_table(policy_summary, validation_dir / f"{prefix}_source_seed_policy_summary")
    _write_table(pairwise, validation_dir / f"{prefix}_source_seed_quality_first_vs_small_only")
    _write_table(hard_cap_pairwise, validation_dir / f"{prefix}_source_seed_hard_cap_diagnostics")
    _write_table(
        hard_cap_summary,
        validation_dir / f"{prefix}_source_seed_hard_cap_diagnostic_summary",
    )
    return {
        "source_effects": combined,
        "source_policy_summary": policy_summary,
        "source_pairwise": pairwise,
        "source_hard_cap_pairwise": hard_cap_pairwise,
        "source_hard_cap_summary": hard_cap_summary,
    }

def _write_next_outputs(
    *,
    combined: pd.DataFrame,
    validation_dir: Path,
    prefix: str,
) -> dict[str, pd.DataFrame]:
    policy_summary = _next_policy_summary(combined)
    pairwise = _next_quality_first_pairwise(combined)
    pairwise_summary = _next_pairwise_summary(pairwise)
    hard_cap_pairwise = _next_hard_cap_diagnostic_pairwise(combined)
    hard_cap_summary = _next_hard_cap_diagnostic_summary(hard_cap_pairwise)
    _write_table(combined, validation_dir / f"{prefix}_source_seed_next_level_effects")
    _write_table(
        policy_summary,
        validation_dir / f"{prefix}_source_seed_next_level_summary",
    )
    _write_table(
        pairwise,
        validation_dir / f"{prefix}_source_seed_next_level_quality_first_vs_small_only",
    )
    _write_table(
        pairwise_summary,
        validation_dir / f"{prefix}_source_seed_next_level_quality_first_vs_small_only_summary",
    )
    _write_table(
        hard_cap_pairwise,
        validation_dir / f"{prefix}_source_seed_next_level_hard_cap_diagnostics",
    )
    _write_table(
        hard_cap_summary,
        validation_dir / f"{prefix}_source_seed_next_level_hard_cap_diagnostic_summary",
    )
    return {
        "next_effects": combined,
        "next_policy_summary": policy_summary,
        "next_pairwise": pairwise,
        "next_pairwise_summary": pairwise_summary,
        "next_hard_cap_pairwise": hard_cap_pairwise,
        "next_hard_cap_summary": hard_cap_summary,
    }

def _plot_source_field_expansion(pairwise: pd.DataFrame, output_dir: Path) -> Path | None:
    if pairwise.empty:
        return None
    data = pairwise.copy()
    data["field_id"] = data["sample"].map(_field_id_from_sample)
    grouped = (
        data.groupby("field_id", as_index=False)[
            ["delta_max_ratio", "delta_oversize_count", "delta_gini"]
        ]
        .mean()
        .sort_values("field_id")
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    metrics = [
        ("delta_max_ratio", "source max/target ratio delta"),
        ("delta_oversize_count", "source oversize-count delta"),
        ("delta_gini", "source Gini delta"),
    ]
    colors = ["#4c78a8" if int(field) in {12, 15, 34} else "#54a24b" for field in grouped["field_id"]]
    for ax, (column, title) in zip(axes, metrics, strict=True):
        ax.bar(grouped["field_id"].astype(str), grouped[column], color=colors)
        ax.axhline(0.0, color="#222222", linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("field")
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("quality_first - small_only")
    fig.suptitle("Six-field source-seed expansion")
    fig.tight_layout()
    out_path = output_dir / "figure9_field_expansion_source_seed.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path

def _plot_next_field_expansion(pairwise: pd.DataFrame, output_dir: Path) -> Path | None:
    if pairwise.empty:
        return None
    data = pairwise.copy()
    data["field_id"] = data["sample"].map(_field_id_from_sample)
    grouped = (
        data.groupby("field_id", as_index=False)[
            [
                "delta_max_ratio",
                "delta_oversize_count",
                "delta_gini",
                "delta_parent_max_child_share",
            ]
        ]
        .mean()
        .sort_values("field_id")
    )
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    metrics = [
        ("delta_max_ratio", "next max/target ratio delta"),
        ("delta_oversize_count", "next oversize-count delta"),
        ("delta_gini", "next Gini delta"),
        ("delta_parent_max_child_share", "parent child-share delta"),
    ]
    colors = ["#4c78a8" if int(field) in {12, 15, 34} else "#54a24b" for field in grouped["field_id"]]
    for ax, (column, title) in zip(axes, metrics, strict=True):
        ax.bar(grouped["field_id"].astype(str), grouped[column], color=colors)
        ax.axhline(0.0, color="#222222", linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("field")
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("quality_first - small_only")
    fig.suptitle("Six-field next-level propagation expansion")
    fig.tight_layout()
    out_path = output_dir / "figure10_field_expansion_next_level.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path

def _field_breakdown(source_pairwise: pd.DataFrame, next_pairwise: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    source = source_pairwise.copy()
    next_df = next_pairwise.copy()
    if not source.empty:
        source["field_id"] = source["sample"].map(_field_id_from_sample)
    if not next_df.empty:
        next_df["field_id"] = next_df["sample"].map(_field_id_from_sample)
    field_ids = sorted(
        {
            int(field)
            for field in pd.concat(
                [
                    source.get("field_id", pd.Series(dtype=float)),
                    next_df.get("field_id", pd.Series(dtype=float)),
                ],
                ignore_index=True,
            ).dropna()
        }
    )
    for field_id in field_ids:
        s = source[source["field_id"] == field_id] if not source.empty else pd.DataFrame()
        n = next_df[next_df["field_id"] == field_id] if not next_df.empty else pd.DataFrame()
        rows.append(
            {
                "field_id": field_id,
                "sample": str(s["sample"].iloc[0] if not s.empty else n["sample"].iloc[0]),
                "evidence_group": "initial" if field_id in {12, 15, 34} else "expanded",
                "source_pairs": int(len(s)),
                "source_mean_delta_max_ratio": float(s["delta_max_ratio"].mean()) if not s.empty else np.nan,
                "source_pairs_lower_max_ratio": int((s["delta_max_ratio"] < 0).sum()) if not s.empty else 0,
                "source_mean_delta_oversize_count": float(s["delta_oversize_count"].mean()) if not s.empty else np.nan,
                "source_pairs_lower_oversize_count": int((s["delta_oversize_count"] < 0).sum()) if not s.empty else 0,
                "source_mean_delta_gini": float(s["delta_gini"].mean()) if not s.empty else np.nan,
                "source_pairs_lower_gini": int((s["delta_gini"] < 0).sum()) if not s.empty else 0,
                "next_pairs": int(len(n)),
                "next_mean_delta_max_ratio": float(n["delta_max_ratio"].mean()) if not n.empty else np.nan,
                "next_pairs_lower_max_ratio": int((n["delta_max_ratio"] < 0).sum()) if not n.empty else 0,
                "next_mean_delta_oversize_count": float(n["delta_oversize_count"].mean()) if not n.empty else np.nan,
                "next_pairs_lower_oversize_count": int((n["delta_oversize_count"] < 0).sum()) if not n.empty else 0,
                "next_mean_delta_gini": float(n["delta_gini"].mean()) if not n.empty else np.nan,
                "next_pairs_lower_gini": int((n["delta_gini"] < 0).sum()) if not n.empty else 0,
                "next_mean_delta_parent_max_child_share": (
                    float(n["delta_parent_max_child_share"].mean()) if not n.empty else np.nan
                ),
                "next_pairs_lower_parent_max_child_share": (
                    int((n["delta_parent_max_child_share"] < 0).sum()) if not n.empty else 0
                ),
            }
        )
    return pd.DataFrame(rows)

def _write_summary_json(
    *,
    validation_dir: Path,
    source_outputs: dict[str, pd.DataFrame],
    next_outputs: dict[str, pd.DataFrame],
    field_breakdown: pd.DataFrame,
    source_figure: Path | None,
    next_figure: Path | None,
) -> Path:
    source_pairwise = source_outputs["source_pairwise"]
    next_pairwise = next_outputs["next_pairwise"]
    next_summary = next_outputs["next_pairwise_summary"]
    source_effects = source_outputs["source_effects"]
    next_effects = next_outputs["next_effects"]
    summary = {
        "fields": sorted(
            int(field)
            for field in source_effects["sample"].map(_field_id_from_sample).dropna().unique()
        ),
        "source_seed_effect_rows": int(len(source_effects)),
        "source_quality_first_pairs": int(len(source_pairwise)),
        "source_quality_first_mean_delta_q": float(
            source_pairwise["delta_q_quality_first"].mean()
        )
        if not source_pairwise.empty
        else 0.0,
        "source_quality_first_pairs_lower_max_ratio": int(
            (source_pairwise["delta_max_ratio"] < 0).sum()
        )
        if not source_pairwise.empty
        else 0,
        "source_quality_first_pairs_lower_oversize_count": int(
            (source_pairwise["delta_oversize_count"] < 0).sum()
        )
        if not source_pairwise.empty
        else 0,
        "source_quality_first_pairs_lower_gini": int((source_pairwise["delta_gini"] < 0).sum())
        if not source_pairwise.empty
        else 0,
        "next_level_effect_rows": int(len(next_effects)),
        "next_quality_first_pairs": int(len(next_pairwise)),
        "next_quality_first_summary": (
            next_summary.iloc[0].to_dict() if not next_summary.empty else {}
        ),
        "field_breakdown": field_breakdown.to_dict("records"),
        "figures": {
            "source_seed": _rel(source_figure),
            "next_level": _rel(next_figure),
        },
    }
    path = validation_dir / "field_expansion_summary.json"
    _write_json(path, summary)
    return path

def _validate_outputs(validation_dir: Path) -> dict[str, Any]:
    source = pd.read_csv(validation_dir / "field_expansion_source_seed_effects.csv")
    source_pairwise = pd.read_csv(
        validation_dir / "field_expansion_source_seed_quality_first_vs_small_only.csv"
    )
    next_effects = pd.read_csv(validation_dir / "field_expansion_source_seed_next_level_effects.csv")
    next_pairwise = pd.read_csv(
        validation_dir / "field_expansion_source_seed_next_level_quality_first_vs_small_only.csv"
    )
    fields = sorted(
        int(field)
        for field in source["sample"].map(_field_id_from_sample).dropna().unique()
    )
    missing_fields = sorted(set(EXPECTED_FIELD_IDS).difference(fields))
    new_seed_counts = {
        int(field): int(
            source[
                (source["sample"].map(_field_id_from_sample) == field)
                & (source["membership_role"] == "effective")
            ]["seed"].nunique()
        )
        for field in DEFAULT_FIELDS
    }
    effective_next = next_effects[next_effects["membership_role"] == "effective"].copy()
    next_seed_counts = (
        effective_next.groupby(["sample", "source_seed", "source_policy"], dropna=False)[
            "next_seed"
        ]
        .nunique()
        .reset_index(name="n_next_seeds")
    )
    bad_next_seed_rows = next_seed_counts[next_seed_counts["n_next_seeds"] != 3]
    expected_source_pairs = len(EXPECTED_FIELD_IDS) * len(DEFAULT_SEEDS)
    expected_next_pairs = expected_source_pairs * len(DEFAULT_SEEDS)
    validation = {
        "fields": fields,
        "missing_fields": missing_fields,
        "new_field_source_seed_counts": new_seed_counts,
        "source_pairwise_rows": int(len(source_pairwise)),
        "expected_source_pairwise_rows": int(expected_source_pairs),
        "next_pairwise_rows": int(len(next_pairwise)),
        "expected_next_pairwise_rows": int(expected_next_pairs),
        "effective_next_groups_with_wrong_seed_count": int(len(bad_next_seed_rows)),
        "passed": (
            not missing_fields
            and all(count == len(DEFAULT_SEEDS) for count in new_seed_counts.values())
            and len(source_pairwise) == expected_source_pairs
            and len(next_pairwise) == expected_next_pairs
            and bad_next_seed_rows.empty
        ),
    }
    _write_json(validation_dir / "field_expansion_validation_summary.json", validation)
    return validation

def _write_field_expansion_report(
    *,
    validation_dir: Path,
    source_outputs: dict[str, pd.DataFrame],
    next_outputs: dict[str, pd.DataFrame],
    field_breakdown: pd.DataFrame,
    validation: dict[str, Any],
    source_figure: Path | None,
    next_figure: Path | None,
) -> Path:
    source_pairwise = source_outputs["source_pairwise"]
    next_summary = next_outputs["next_pairwise_summary"]
    hard_cap_source = source_outputs["source_hard_cap_summary"]
    hard_cap_next = next_outputs["next_hard_cap_summary"]
    lines = [
        "# Field Expansion Hierarchy Postprocess Validation",
        "",
        "This report combines the initial field12/15/34 evidence with expanded GCC embedding-field runs for field18, field26, and field30.",
        "",
        f"- Fields: {', '.join('field' + str(field) for field in validation['fields'])}",
        f"- Source quality_first pairs: {len(source_pairwise)}",
        f"- Next-level quality_first pairs: {int(next_summary['n_pairs'].iloc[0]) if not next_summary.empty else 0}",
        f"- Validation passed: {validation['passed']}",
        "",
    ]
    if not source_pairwise.empty:
        lines.extend(
            [
                "## Source-Level Quality-First vs Small-Only",
                "",
                f"- Mean exact delta Q: {source_pairwise['delta_q_quality_first'].mean():.6g}",
                f"- Mean max/target ratio delta: {source_pairwise['delta_max_ratio'].mean():.6g}",
                f"- Mean oversize-count delta: {source_pairwise['delta_oversize_count'].mean():.6g}",
                f"- Mean Gini delta: {source_pairwise['delta_gini'].mean():.6g}",
                f"- Lower max/target ratio rows: {int((source_pairwise['delta_max_ratio'] < 0).sum())} / {len(source_pairwise)}",
                f"- Lower oversize-count rows: {int((source_pairwise['delta_oversize_count'] < 0).sum())} / {len(source_pairwise)}",
                f"- Lower Gini rows: {int((source_pairwise['delta_gini'] < 0).sum())} / {len(source_pairwise)}",
                "",
            ]
        )
    if not next_summary.empty:
        lines.extend(
            [
                "## Next-Level Propagation",
                "",
                _markdown_table(next_summary),
                "",
            ]
        )
    if not hard_cap_source.empty or not hard_cap_next.empty:
        lines.extend(["## Hard-Cap Diagnostic", ""])
        if not hard_cap_source.empty:
            lines.extend(["Source level:", "", _markdown_table(hard_cap_source), ""])
        if not hard_cap_next.empty:
            lines.extend(["Next level:", "", _markdown_table(hard_cap_next), ""])
    if not field_breakdown.empty:
        lines.extend(["## Field Breakdown", "", _markdown_table(field_breakdown), ""])
    lines.extend(
        [
            "## Outputs",
            "",
            "- `field_expansion_source_seed_effects.csv` / `.parquet`",
            "- `field_expansion_source_seed_policy_summary.csv` / `.parquet`",
            "- `field_expansion_source_seed_quality_first_vs_small_only.csv` / `.parquet`",
            "- `field_expansion_source_seed_next_level_effects.csv` / `.parquet`",
            "- `field_expansion_source_seed_next_level_quality_first_vs_small_only.csv` / `.parquet`",
            "- `field_expansion_source_seed_next_level_summary.csv` / `.parquet`",
            "- `field_expansion_field_breakdown.csv` / `.parquet`",
            "",
        ]
    )
    if source_figure is not None or next_figure is not None:
        lines.extend(["## Figures", ""])
        if source_figure is not None:
            lines.append(f"- `{source_figure.name}`")
        if next_figure is not None:
            lines.append(f"- `{next_figure.name}`")
        lines.append("")
    path = validation_dir / "field_expansion_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--validation-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--fields", default=",".join(str(field) for field in DEFAULT_FIELDS))
    parser.add_argument("--source-seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--next-seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--policies", default=",".join(DEFAULT_POLICIES))
    parser.add_argument("--resolution", type=float, default=0.01)
    parser.add_argument("--target-min-doc-weight", type=float, default=50.0)
    parser.add_argument("--prepare-seed", type=int, default=42)
    parser.add_argument("--n-iterations", type=int, default=5)
    parser.add_argument("--apply-iterations", type=int, default=4)
    parser.add_argument("--trim-max-moves-per-cluster", type=int, default=100)
    parser.add_argument("--selection-singleton-budget", type=float, default=100.0)
    parser.add_argument("--next-target-pct", type=float, default=None)
    parser.add_argument("--next-target-min-pct", type=float, default=0.0)
    parser.add_argument("--next-target-multiplier", type=float, default=1.0)
    parser.add_argument("--next-min-doc-weight", type=float, default=100.0)
    parser.add_argument("--existing-source", type=Path, default=DEFAULT_EXISTING_SOURCE)
    parser.add_argument("--existing-next", type=Path, default=DEFAULT_EXISTING_NEXT)
    parser.add_argument(
        "--stage",
        choices=("all", "prepare", "source", "next", "aggregate", "validate"),
        default="all",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()

    fields = _parse_int_list(str(args.fields))
    source_seeds = _parse_int_list(str(args.source_seeds))
    next_seeds = _parse_int_list(str(args.next_seeds))
    policies = tuple(policy.strip() for policy in str(args.policies).split(",") if policy.strip())

    validation_dir = _repo_path(args.validation_dir)
    run_root = _repo_path(args.run_root)
    assert validation_dir is not None and run_root is not None
    if args.dry_run:
        dry = _dry_run_rows(fields, run_root)
        print(dry.to_csv(index=False), end="")
        return

    if args.stage == "validate":
        validation = _validate_outputs(validation_dir)
        print(json.dumps(validation, indent=2, sort_keys=True))
        return

    validation_dir.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)

    configs: list[ExpansionConfig] = []
    if args.stage in {"all", "prepare", "source", "next"}:
        for field_id in fields:
            print(f"preparing field{field_id}", flush=True)
            configs.append(
                _prepare_field(
                    field_id=int(field_id),
                    run_root=run_root,
                    resolution=float(args.resolution),
                    target_min_doc_weight=float(args.target_min_doc_weight),
                    seed=int(args.prepare_seed),
                    n_iterations=int(args.n_iterations),
                    force=bool(args.force),
                )
            )
    else:
        configs = [
            _config_from_summary(field_id, run_root / _field_sample(field_id) / "prepare_summary.json")
            for field_id in fields
        ]
    if args.stage == "prepare":
        print(
            pd.DataFrame(
                [
                    {
                        "field_id": cfg.field_id,
                        "sample": cfg.sample,
                        "n_nodes": cfg.n_nodes,
                        "n_edges": cfg.n_edges,
                        "target_min_doc_weight": cfg.target_min_doc_weight,
                        "target_max_doc_weight": cfg.target_max_doc_weight,
                        "prepare_summary_path": _rel(cfg.prepare_summary_path),
                    }
                    for cfg in configs
                ]
            ).to_csv(index=False),
            end="",
        )
        return

    new_source_path = run_root / "field_expansion_new_source_seed_effects.csv"
    if args.stage in {"all", "source"}:
        new_source = _run_new_source_seed_sweep(
            configs=configs,
            run_root=run_root,
            seeds=source_seeds,
            policies=policies,
            apply_iterations=int(args.apply_iterations),
            trim_max_moves_per_cluster=int(args.trim_max_moves_per_cluster),
            selection_singleton_budget=float(args.selection_singleton_budget),
            force=bool(args.force),
        )
        _write_table(new_source, run_root / "field_expansion_new_source_seed_effects")
    elif new_source_path.exists():
        new_source = pd.read_csv(new_source_path)
    else:
        new_source = pd.DataFrame()

    source_outputs: dict[str, pd.DataFrame] = {}
    if args.stage in {"all", "source", "aggregate", "validate"}:
        if new_source.empty:
            new_source = pd.read_csv(new_source_path)
        existing_source = _load_csv(args.existing_source)
        combined_source = pd.concat([existing_source, new_source], ignore_index=True)
        source_outputs = _write_source_outputs(
            combined=combined_source,
            validation_dir=validation_dir,
            prefix="field_expansion",
        )
    if args.stage == "source":
        print(f"Saved source expansion to {_rel(validation_dir / 'field_expansion_source_seed_effects.csv')}")
        return

    new_next_path = run_root / "field_expansion_new_source_seed_next_level_effects.csv"
    if args.stage in {"all", "next"}:
        if new_source.empty:
            new_source = pd.read_csv(new_source_path)
        new_next = _run_new_next_level(
            source_df=new_source,
            configs=configs,
            run_root=run_root,
            source_seeds=source_seeds,
            next_seeds=next_seeds,
            policies={"small_only", *policies},
            include_diagnostics=True,
            next_target_pct=args.next_target_pct,
            next_target_min_pct=float(args.next_target_min_pct),
            next_target_multiplier=float(args.next_target_multiplier),
            next_min_doc_weight=float(args.next_min_doc_weight),
            force=bool(args.force),
        )
        _write_table(new_next, run_root / "field_expansion_new_source_seed_next_level_effects")
    elif new_next_path.exists():
        new_next = pd.read_csv(new_next_path)
    else:
        new_next = pd.DataFrame()

    next_outputs: dict[str, pd.DataFrame] = {}
    source_figure = None
    next_figure = None
    field_breakdown = pd.DataFrame()
    if args.stage in {"all", "next", "aggregate", "validate"}:
        if not source_outputs:
            existing_source = _load_csv(args.existing_source)
            if new_source.empty:
                new_source = pd.read_csv(new_source_path)
            source_outputs = _write_source_outputs(
                combined=pd.concat([existing_source, new_source], ignore_index=True),
                validation_dir=validation_dir,
                prefix="field_expansion",
            )
        if new_next.empty:
            new_next = pd.read_csv(new_next_path)
        existing_next = _load_csv(args.existing_next)
        combined_next = pd.concat([existing_next, new_next], ignore_index=True)
        next_outputs = _write_next_outputs(
            combined=combined_next,
            validation_dir=validation_dir,
            prefix="field_expansion",
        )
        field_breakdown = _field_breakdown(
            source_outputs["source_pairwise"],
            next_outputs["next_pairwise"],
        )
        _write_table(field_breakdown, validation_dir / "field_expansion_field_breakdown")
        if not args.skip_figures:
            source_figure = _plot_source_field_expansion(
                source_outputs["source_pairwise"],
                validation_dir,
            )
            next_figure = _plot_next_field_expansion(
                next_outputs["next_pairwise"],
                validation_dir,
            )
        _write_summary_json(
            validation_dir=validation_dir,
            source_outputs=source_outputs,
            next_outputs=next_outputs,
            field_breakdown=field_breakdown,
            source_figure=source_figure,
            next_figure=next_figure,
        )

    validation = _validate_outputs(validation_dir)
    if next_outputs:
        report_path = _write_field_expansion_report(
            validation_dir=validation_dir,
            source_outputs=source_outputs,
            next_outputs=next_outputs,
            field_breakdown=field_breakdown,
            validation=validation,
            source_figure=source_figure,
            next_figure=next_figure,
        )
        print(f"Report: {_rel(report_path)}")
    print(json.dumps(validation, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
