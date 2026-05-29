"""Run actual next-level Leiden from hierarchy postprocess memberships.

The first validation pass measures the contraction input distribution.  This
script goes one step further: it contracts the original graph with each
candidate level-0 membership, runs a deterministic next-level gamma sweep and
Leiden pass, then records the next-level cluster weight distribution.
"""

from __future__ import annotations

import argparse
import json
import re
import time
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
import pyarrow as pa
import pyarrow.parquet as pq

from evaluate_hierarchy_postprocess import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RESULTS_DIR,
    _cluster_weights,
    _gini,
    _load_membership,
    _load_node_weights,
    _markdown_table,
    _normalized_entropy,
    _percentile,
    _repo_path,
    _sample_configs,
    _write_table,
)
from sciscape.clustering.hierarchical import (  # noqa: E402
    _contract_and_normalize,
    _sweep_gamma_direct,
)
from sciscape.clustering.leiden_rust import (  # noqa: E402
    build_leiden_graph,
    postprocess_small_clusters_rust,
)

DEFAULT_POLICIES = (
    "small_only",
    "oversize_split_only",
    "two_stage_quality_first",
    "two_stage_hard_cap",
)

def _rel(path: Path | str | None) -> str | None:
    if path is None:
        return None
    current = Path(path)
    try:
        return str(current.relative_to(REPO_ROOT))
    except ValueError:
        return str(current)

def _safe_slug(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9_.-]+", "_", value)
    return value.strip("_") or "run"

def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

def _write_membership(path: Path, membership: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "node_idx": np.arange(membership.shape[0], dtype=np.uint64),
            "cluster": np.asarray(membership, dtype=np.uint64),
        }
    )
    pq.write_table(table, path, compression="zstd")

def _load_graph_arrays(graph_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    src = np.memmap(graph_dir / "src.u32.bin", dtype=np.uint32, mode="r")
    dst = np.memmap(graph_dir / "dst.u32.bin", dtype=np.uint32, mode="r")
    weight = np.memmap(graph_dir / "weight.f64.bin", dtype=np.float64, mode="r")
    node_weights = np.memmap(graph_dir / "node_weights.f64.bin", dtype=np.float64, mode="r")
    return src, dst, weight, node_weights

def _compact_membership(membership: np.ndarray) -> np.ndarray:
    membership = np.asarray(membership, dtype=np.int64)
    unique = np.unique(membership)
    if unique.size and unique[0] == 0 and unique[-1] == unique.size - 1:
        return membership.astype(np.uint64, copy=False)
    remap = {int(old): idx for idx, old in enumerate(unique.tolist())}
    return np.asarray([remap[int(value)] for value in membership], dtype=np.uint64)

def _membership_metrics(
    membership: np.ndarray,
    node_weights: np.ndarray,
    *,
    target_max_doc_weight: float,
) -> dict[str, Any]:
    membership = np.asarray(membership, dtype=np.int64)
    weights = _cluster_weights(membership, node_weights)
    total = float(weights.sum()) if weights.size else 0.0
    sorted_desc = np.sort(weights)[::-1]
    return {
        "n_clusters": int(weights.size),
        "total_doc_weight": total,
        "target_max_doc_weight": float(target_max_doc_weight),
        "max_doc_weight": float(sorted_desc[0]) if sorted_desc.size else 0.0,
        "max_doc_weight_ratio": (
            float(sorted_desc[0]) / float(target_max_doc_weight)
            if sorted_desc.size and target_max_doc_weight > 0.0
            else 0.0
        ),
        "n_above_max_doc_weight": (
            int((weights > target_max_doc_weight).sum())
            if target_max_doc_weight > 0.0
            else 0
        ),
        "p50_doc_weight": _percentile(weights, 50),
        "p90_doc_weight": _percentile(weights, 90),
        "p95_doc_weight": _percentile(weights, 95),
        "p99_doc_weight": _percentile(weights, 99),
        "gini_doc_weight": _gini(weights),
        "entropy_doc_weight": _normalized_entropy(weights),
        "top1_doc_weight_share": float(sorted_desc[:1].sum() / total) if total else 0.0,
        "top5_doc_weight_share": float(sorted_desc[:5].sum() / total) if total else 0.0,
        "top10_doc_weights": [float(value) for value in sorted_desc[:10]],
    }

def _parent_child_metrics(
    parent_membership: np.ndarray,
    child_weights: np.ndarray,
) -> dict[str, Any]:
    parent = np.asarray(parent_membership, dtype=np.int64)
    child_weights = np.asarray(child_weights, dtype=np.float64)
    if parent.size == 0:
        return {
            "parent_max_child_share_max": 0.0,
            "parent_max_child_share_weighted_mean": 0.0,
            "parent_child_count_p50": 0.0,
            "parent_child_count_p95": 0.0,
        }
    n_parent = int(parent.max()) + 1
    totals = np.bincount(parent, weights=child_weights, minlength=n_parent)
    counts = np.bincount(parent, minlength=n_parent)
    max_child = np.zeros(n_parent, dtype=np.float64)
    np.maximum.at(max_child, parent, child_weights)
    active = totals > 0
    shares = max_child[active] / totals[active]
    active_totals = totals[active]
    active_counts = counts[active]
    return {
        "parent_max_child_share_max": float(shares.max()) if shares.size else 0.0,
        "parent_max_child_share_weighted_mean": (
            float(np.average(shares, weights=active_totals)) if shares.size else 0.0
        ),
        "parent_child_count_p50": _percentile(active_counts.astype(np.float64), 50),
        "parent_child_count_p95": _percentile(active_counts.astype(np.float64), 95),
    }

def _candidate_rows(
    eval_df: pd.DataFrame,
    *,
    policies: set[str],
    include_diagnostics: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in eval_df.to_dict("records"):
        policy = str(row.get("policy"))
        if policy not in policies:
            continue
        membership_path = row.get("membership_path")
        if isinstance(membership_path, str) and membership_path:
            rows.append(
                {
                    **row,
                    "membership_role": "effective",
                    "rerun_policy": policy,
                    "rerun_run": str(row.get("run")),
                    "rerun_membership_path": membership_path,
                    "diagnostic_for_policy": None,
                }
            )
        diagnostic_path = row.get("diagnostic_membership_path")
        if include_diagnostics and isinstance(diagnostic_path, str) and diagnostic_path:
            rows.append(
                {
                    **row,
                    "membership_role": "diagnostic",
                    "rerun_policy": f"{policy}_diagnostic",
                    "rerun_run": f"{row.get('run')}_diagnostic",
                    "rerun_membership_path": diagnostic_path,
                    "diagnostic_for_policy": policy,
                }
            )
    return rows

def _run_one(
    *,
    row: dict[str, Any],
    graph_arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    graph_dir: Path,
    output_dir: Path,
    next_target_pct: float,
    next_min_doc_weight: float,
    seed: int,
    force: bool,
    runs_dir_name: str = "actual_next_level_runs",
) -> dict[str, Any]:
    sample = str(row["sample"])
    policy = str(row["rerun_policy"])
    run = str(row["rerun_run"])
    role = str(row["membership_role"])
    run_dir = output_dir / runs_dir_name / _safe_slug(sample) / _safe_slug(run)
    summary_path = run_dir / "summary.json"
    if summary_path.exists() and not force:
        cached = _read_json(summary_path)
        if (
            float(cached.get("next_target_pct", -1.0)) == float(next_target_pct)
            and float(cached.get("next_min_doc_weight", -1.0)) == float(next_min_doc_weight)
            and int(cached.get("seed", -1)) == int(seed)
        ):
            return cached

    src, dst, weight, node_weights = graph_arrays
    membership_path = _repo_path(row["rerun_membership_path"])
    if membership_path is None or not membership_path.exists():
        raise FileNotFoundError(f"Missing membership: {membership_path}")

    t0 = time.perf_counter()
    membership = _compact_membership(_load_membership(membership_path))
    original_node_weights = _load_node_weights(graph_dir / "node_weights.f64.bin", int(membership.shape[0]))
    prev_node_sizes = None
    if not np.all(original_node_weights == 1.0):
        prev_node_sizes = original_node_weights.astype(np.int64)

    contract_t0 = time.perf_counter()
    contracted_src, contracted_dst, contracted_weight, contracted_n, child_doc_weights = (
        _contract_and_normalize(src, dst, weight, membership, prev_node_sizes)
    )
    contract_elapsed = time.perf_counter() - contract_t0
    child_doc_weights = np.asarray(child_doc_weights, dtype=np.float64)
    total_doc_weight = float(child_doc_weights.sum())
    next_target_max_doc_weight = total_doc_weight * float(next_target_pct) / 100.0
    contraction_metrics = _membership_metrics(
        np.arange(child_doc_weights.shape[0], dtype=np.uint64),
        child_doc_weights,
        target_max_doc_weight=float(row.get("target_max_doc_weight") or 0.0),
    )

    gamma_t0 = time.perf_counter()
    gamma = _sweep_gamma_direct(
        contracted_src,
        contracted_dst,
        contracted_weight,
        int(contracted_n),
        child_doc_weights,
        target_max_pct=float(next_target_pct),
        min_size=int(next_min_doc_weight),
        seed=int(seed),
        _log=None,
    )
    gamma_elapsed = time.perf_counter() - gamma_t0

    graph_t0 = time.perf_counter()
    graph = build_leiden_graph(
        edges_src=contracted_src,
        edges_dst=contracted_dst,
        edges_weight=contracted_weight,
        n_nodes=int(contracted_n),
        node_weights=child_doc_weights,
    )
    leiden = graph.run_leiden(resolution=float(gamma), seed=int(seed), n_iterations=10)
    leiden_elapsed = time.perf_counter() - graph_t0
    raw_next_membership = np.asarray(leiden.membership, dtype=np.uint64)
    raw_quality = float(leiden.quality)
    raw_metrics = _membership_metrics(
        raw_next_membership,
        child_doc_weights,
        target_max_doc_weight=next_target_max_doc_weight,
    )
    raw_parent_child = _parent_child_metrics(raw_next_membership, child_doc_weights)

    post_t0 = time.perf_counter()
    post = postprocess_small_clusters_rust(
        resolution=float(gamma),
        min_size=0,
        min_weight=float(next_min_doc_weight),
        membership=raw_next_membership,
        edges_src=contracted_src,
        edges_dst=contracted_dst,
        edges_weight=contracted_weight,
        node_weights=child_doc_weights,
        n_nodes=int(contracted_n),
        seed=int(seed),
        gamma_decay=0.5,
        max_rounds=3,
        use_greedy=True,
        use_component_merge=True,
    )
    post_membership = np.asarray(post.membership, dtype=np.uint64)
    post_quality = graph.cpm_quality(post_membership, resolution=float(gamma))
    post_elapsed = time.perf_counter() - post_t0
    post_metrics = _membership_metrics(
        post_membership,
        child_doc_weights,
        target_max_doc_weight=next_target_max_doc_weight,
    )
    post_parent_child = _parent_child_metrics(post_membership, child_doc_weights)

    raw_membership_path = run_dir / "next_level_raw_membership.parquet"
    post_membership_path = run_dir / "next_level_post_membership.parquet"
    _write_membership(raw_membership_path, raw_next_membership)
    _write_membership(post_membership_path, post_membership)

    summary = {
        "sample": sample,
        "policy": policy,
        "source_policy": str(row.get("policy")),
        "run": run,
        "source_run": str(row.get("run")),
        "source_seed": row.get("source_seed"),
        "membership_role": role,
        "diagnostic_for_policy": row.get("diagnostic_for_policy"),
        "source_status": str(row.get("status")),
        "source_accepted_for_contraction": bool(row.get("accepted_for_contraction")),
        "source_fallback_used": bool(row.get("fallback_used")),
        "source_delta_q": float(row.get("delta_q") or 0.0),
        "source_split_repair_exact_delta_q": float(
            row.get("split_repair_exact_delta_q") or 0.0
        ),
        "source_trim_exact_delta_q": float(row.get("trim_exact_delta_q") or 0.0),
        "source_membership_path": _rel(membership_path),
        "graph_dir": _rel(graph_dir),
        "seed": int(seed),
        "next_target_pct": float(next_target_pct),
        "next_min_doc_weight": float(next_min_doc_weight),
        "next_target_max_doc_weight": next_target_max_doc_weight,
        "contracted_n_nodes": int(contracted_n),
        "contracted_n_edges": int(contracted_weight.shape[0]),
        "contracted_top_k_rank_normalized": True,
        "contracted_child_doc_weight": contraction_metrics,
        "gamma": float(gamma),
        "raw_quality": raw_quality,
        "post_quality": float(post_quality),
        "post_delta_q": float(post_quality - raw_quality),
        "next_raw": {**raw_metrics, **raw_parent_child},
        "next_post": {**post_metrics, **post_parent_child},
        "paths": {
            "summary": _rel(summary_path),
            "next_level_raw_membership": _rel(raw_membership_path),
            "next_level_post_membership": _rel(post_membership_path),
        },
        "elapsed_sec": {
            "contract": contract_elapsed,
            "gamma_sweep": gamma_elapsed,
            "leiden": leiden_elapsed,
            "postprocess": post_elapsed,
            "total": time.perf_counter() - t0,
        },
    }
    _write_json(summary_path, summary)
    return summary

def _flatten_summary(summary: dict[str, Any]) -> dict[str, Any]:
    raw = summary["next_raw"]
    post = summary["next_post"]
    child = summary["contracted_child_doc_weight"]
    return {
        "sample": summary["sample"],
        "policy": summary["policy"],
        "source_policy": summary["source_policy"],
        "run": summary["run"],
        "source_seed": summary.get("source_seed"),
        "membership_role": summary["membership_role"],
        "diagnostic_for_policy": summary.get("diagnostic_for_policy"),
        "source_status": summary["source_status"],
        "source_accepted_for_contraction": summary["source_accepted_for_contraction"],
        "source_fallback_used": summary["source_fallback_used"],
        "source_delta_q": summary.get("source_delta_q", 0.0),
        "source_split_repair_exact_delta_q": summary.get(
            "source_split_repair_exact_delta_q",
            0.0,
        ),
        "source_trim_exact_delta_q": summary.get("source_trim_exact_delta_q", 0.0),
        "seed": summary["seed"],
        "next_target_pct": summary["next_target_pct"],
        "next_min_doc_weight": summary["next_min_doc_weight"],
        "next_target_max_doc_weight": summary["next_target_max_doc_weight"],
        "contracted_n_nodes": summary["contracted_n_nodes"],
        "contracted_n_edges": summary["contracted_n_edges"],
        "contracted_max_child_doc_weight": child["max_doc_weight"],
        "contracted_n_children_above_level0_target": child["n_above_max_doc_weight"],
        "gamma": summary["gamma"],
        "raw_quality": summary["raw_quality"],
        "post_quality": summary["post_quality"],
        "post_delta_q": summary["post_delta_q"],
        "raw_n_clusters": raw["n_clusters"],
        "raw_max_doc_weight": raw["max_doc_weight"],
        "raw_max_doc_weight_ratio": raw["max_doc_weight_ratio"],
        "raw_n_above_max_doc_weight": raw["n_above_max_doc_weight"],
        "raw_gini_doc_weight": raw["gini_doc_weight"],
        "raw_entropy_doc_weight": raw["entropy_doc_weight"],
        "raw_parent_max_child_share_max": raw["parent_max_child_share_max"],
        "raw_parent_max_child_share_weighted_mean": raw[
            "parent_max_child_share_weighted_mean"
        ],
        "post_n_clusters": post["n_clusters"],
        "post_max_doc_weight": post["max_doc_weight"],
        "post_max_doc_weight_ratio": post["max_doc_weight_ratio"],
        "post_n_above_max_doc_weight": post["n_above_max_doc_weight"],
        "post_p95_doc_weight": post["p95_doc_weight"],
        "post_p99_doc_weight": post["p99_doc_weight"],
        "post_gini_doc_weight": post["gini_doc_weight"],
        "post_entropy_doc_weight": post["entropy_doc_weight"],
        "post_top1_doc_weight_share": post["top1_doc_weight_share"],
        "post_parent_max_child_share_max": post["parent_max_child_share_max"],
        "post_parent_max_child_share_weighted_mean": post[
            "parent_max_child_share_weighted_mean"
        ],
        "elapsed_total_sec": summary["elapsed_sec"]["total"],
        "summary_path": summary["paths"]["summary"],
    }

def _policy_summary(df: pd.DataFrame) -> pd.DataFrame:
    effective = df[df["membership_role"] == "effective"].copy()
    if effective.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for policy, group in effective.groupby("source_policy", sort=False):
        rows.append(
            {
                "source_policy": policy,
                "n_runs": int(len(group)),
                "accepted_rate": float(group["source_accepted_for_contraction"].mean()),
                "fallback_rate": float(group["source_fallback_used"].mean()),
                "mean_next_post_max_ratio": float(group["post_max_doc_weight_ratio"].mean()),
                "mean_next_post_oversize_count": float(group["post_n_above_max_doc_weight"].mean()),
                "mean_next_post_gini": float(group["post_gini_doc_weight"].mean()),
                "mean_parent_max_child_share": float(
                    group["post_parent_max_child_share_weighted_mean"].mean()
                ),
                "mean_contracted_children": float(group["contracted_n_nodes"].mean()),
            }
        )
    return pd.DataFrame(rows)

def _plot_actual_next_level(df: pd.DataFrame, output_dir: Path) -> Path:
    effective = df[
        (df["membership_role"] == "effective")
        & df["source_policy"].isin(
            ["small_only", "oversize_split_only", "two_stage_quality_first", "two_stage_hard_cap"]
        )
    ].copy()
    effective["label"] = (
        effective["sample"].str.replace("_gcc_emb_full_knn30", "", regex=False)
        + "\n"
        + effective["source_policy"].str.replace("two_stage_", "", regex=False)
    )
    colors = effective["source_policy"].map(
        {
            "small_only": "#666666",
            "oversize_split_only": "#4c78a8",
            "two_stage_quality_first": "#54a24b",
            "two_stage_hard_cap": "#e45756",
        }
    )
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(np.arange(len(effective)), effective["post_max_doc_weight_ratio"], color=colors)
    ax.axhline(1.0, color="#222222", linestyle="--", linewidth=1)
    ax.set_xticks(np.arange(len(effective)))
    ax.set_xticklabels(effective["label"], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("next-level postprocess max doc weight / target")
    ax.set_title("Actual next-level imbalance after contraction and Leiden")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    out_path = output_dir / "figure4_actual_next_level_propagation.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path

def _write_report(
    *,
    effects: pd.DataFrame,
    comparison: pd.DataFrame,
    output_dir: Path,
    figure_path: Path | None,
) -> Path:
    qf = effects[
        (effects["membership_role"] == "effective")
        & (effects["source_policy"] == "two_stage_quality_first")
    ]
    small = effects[
        (effects["membership_role"] == "effective")
        & (effects["source_policy"] == "small_only")
    ]
    lines = [
        "# Actual Next-Level Hierarchy Postprocess Effects",
        "",
        "This rerun contracts each level-0 membership, performs a deterministic next-level gamma sweep, runs Leiden, then applies small-cluster repair on the contracted graph.",
        "By default, the next-level target is coarser than the source level: max(1.5%, current target percentage x 3).",
        "",
        f"- Rows: {len(effects)}",
        f"- Effective rows: {int((effects['membership_role'] == 'effective').sum())}",
        f"- Diagnostic rows: {int((effects['membership_role'] == 'diagnostic').sum())}",
        "",
        "## Main Readout",
        "",
    ]
    if not qf.empty and not small.empty:
        merged = qf.merge(
            small[["sample", "post_max_doc_weight_ratio", "post_n_above_max_doc_weight"]],
            on="sample",
            suffixes=("_quality_first", "_small_only"),
        )
        if not merged.empty:
            ratio_delta = (
                merged["post_max_doc_weight_ratio_quality_first"]
                - merged["post_max_doc_weight_ratio_small_only"]
            ).mean()
            oversize_delta = (
                merged["post_n_above_max_doc_weight_quality_first"]
                - merged["post_n_above_max_doc_weight_small_only"]
            ).mean()
            lines.append(
                f"- quality_first mean next-level max/target ratio delta vs small_only: {ratio_delta:.4g}"
            )
            lines.append(
                f"- quality_first mean next-level oversize-count delta vs small_only: {oversize_delta:.4g}"
            )
    lines.extend(
        [
            "- `membership_role=effective` is what the hierarchy would actually pass forward.",
            "- `membership_role=diagnostic` shows rejected hard-cap memberships for analysis only.",
            "",
            "## Tables",
            "",
            "- `actual_next_level_effects.csv` / `.parquet`",
            "- `actual_contraction_effects.csv` / `.parquet`",
            "- `actual_next_level_policy_comparison.csv` / `.parquet`",
            "",
        ]
    )
    if figure_path is not None:
        lines.extend(["## Figure", "", f"- `{figure_path.name}`", ""])
    if not comparison.empty:
        lines.extend(["## Policy Comparison", "", _markdown_table(comparison)])
    report_path = output_dir / "actual_next_level_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--validation-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--policies",
        default=",".join(DEFAULT_POLICIES),
        help="Comma-separated source policies from hierarchy_postprocess_eval.csv.",
    )
    parser.add_argument("--include-diagnostics", action="store_true", default=True)
    parser.add_argument("--no-diagnostics", dest="include_diagnostics", action="store_false")
    parser.add_argument(
        "--next-target-pct",
        type=float,
        default=None,
        help=(
            "Fixed next-level target percentage. If omitted, use "
            "max(--next-target-min-pct, current_target_pct * --next-target-multiplier)."
        ),
    )
    parser.add_argument("--next-target-min-pct", type=float, default=1.5)
    parser.add_argument("--next-target-multiplier", type=float, default=3.0)
    parser.add_argument("--next-min-doc-weight", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-figure", action="store_true")
    args = parser.parse_args()

    results_dir = _repo_path(args.results_dir)
    validation_dir = _repo_path(args.validation_dir)
    assert results_dir is not None and validation_dir is not None
    eval_path = validation_dir / "hierarchy_postprocess_eval.csv"
    if not eval_path.exists():
        raise FileNotFoundError(
            f"Missing {eval_path}. Run evaluate_hierarchy_postprocess.py first."
        )
    eval_df = pd.read_csv(eval_path)
    policies = {item.strip() for item in args.policies.split(",") if item.strip()}

    cross_path = results_dir / "postprocess_policy_matrix_cross_sample" / "cross_sample_summary.json"
    cross_summary = _read_json(cross_path) if cross_path.exists() else None
    configs = _sample_configs(results_dir, cross_summary)
    graph_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    summaries: list[dict[str, Any]] = []

    rows = _candidate_rows(
        eval_df,
        policies=policies,
        include_diagnostics=bool(args.include_diagnostics),
    )
    for idx, row in enumerate(rows, start=1):
        sample = str(row["sample"])
        cfg = configs[sample]
        graph_dir = cfg.node_weights_path.parent if cfg.node_weights_path else cfg.sample_dir / "graph"
        if sample not in graph_cache:
            print(f"[{idx}/{len(rows)}] loading graph {sample}", flush=True)
            graph_cache[sample] = _load_graph_arrays(graph_dir)
        if args.next_target_pct is None:
            total_docs = float(cfg.n_nodes or graph_cache[sample][3].shape[0])
            current_target = float(row.get("target_max_doc_weight") or 0.0)
            current_target_pct = 100.0 * current_target / total_docs if total_docs else 0.0
            next_target_pct = max(
                float(args.next_target_min_pct),
                current_target_pct * float(args.next_target_multiplier),
            )
        else:
            next_target_pct = float(args.next_target_pct)
        print(
            f"[{idx}/{len(rows)}] next-level {sample} {row['rerun_run']} "
            f"role={row['membership_role']} target_pct={next_target_pct:.3g}",
            flush=True,
        )
        summaries.append(
            _run_one(
                row=row,
                graph_arrays=graph_cache[sample],
                graph_dir=graph_dir,
                output_dir=validation_dir,
                next_target_pct=next_target_pct,
                next_min_doc_weight=float(args.next_min_doc_weight),
                seed=int(args.seed),
                force=bool(args.force),
            )
        )

    effects = pd.DataFrame([_flatten_summary(summary) for summary in summaries])
    comparison = _policy_summary(effects)
    _write_table(effects, validation_dir / "actual_next_level_effects")
    _write_table(effects, validation_dir / "actual_contraction_effects")
    _write_table(comparison, validation_dir / "actual_next_level_policy_comparison")
    figure_path = None if args.skip_figure else _plot_actual_next_level(effects, validation_dir)
    report_path = _write_report(
        effects=effects,
        comparison=comparison,
        output_dir=validation_dir,
        figure_path=figure_path,
    )
    print(f"Saved actual next-level effects to {_rel(validation_dir / 'actual_next_level_effects.csv')}")
    print(f"Report: {_rel(report_path)}")

if __name__ == "__main__":
    main()
