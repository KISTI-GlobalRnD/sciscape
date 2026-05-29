"""Run source-level hierarchy postprocess policies across Leiden seeds.

The next-level seed sweep holds the source membership fixed.  This script
varies the source-level Leiden seed itself, applies small-cluster repair, then
runs the two-stage oversize postprocess policies.  It is intentionally scoped
as a pilot runner over prepared graph artifacts.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace
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
from run_hierarchy_postprocess_next_level import _rel, _safe_slug  # noqa: E402
from run_hierarchy_postprocess_seed_sweep import _parse_int_list  # noqa: E402
from sciscape.clustering.leiden_rust import build_leiden_graph  # noqa: E402
from scripts.run_adaptive_split_merge_repair_probe import (  # noqa: E402
    _membership_weight_summary,
    _postprocess_policy_summary,
    _run_iterative_apply,
    _write_membership,
)

DEFAULT_SEEDS = (11, 42, 73)
DEFAULT_SAMPLE = "field34_combo_dc_bc_cc_sum"
DEFAULT_POLICIES = ("two_stage_quality_first", "two_stage_hard_cap")

def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def _load_graph_arrays(graph_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    src = np.memmap(graph_dir / "src.u32.bin", dtype=np.uint32, mode="r")
    dst = np.memmap(graph_dir / "dst.u32.bin", dtype=np.uint32, mode="r")
    weight = np.memmap(graph_dir / "weight.f64.bin", dtype=np.float64, mode="r")
    node_weights = np.memmap(graph_dir / "node_weights.f64.bin", dtype=np.float64, mode="r")
    return src, dst, weight, node_weights

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

def _write_candidates(path: Path, membership: np.ndarray, node_weights: np.ndarray, target: float) -> None:
    weights = _cluster_weight_values(membership, node_weights)
    rows = []
    for cluster, weight in enumerate(np.bincount(np.asarray(membership, dtype=np.int64), weights=node_weights)):
        if weight > target:
            rows.append(
                {
                    "policy": "oversize_doc_weight",
                    "cluster": int(cluster),
                    "doc_weight": float(weight),
                    "block_count": int(weight),
                }
            )
    rows.sort(key=lambda row: (-row["doc_weight"], row["cluster"]))
    pd.DataFrame(rows, columns=["policy", "cluster", "doc_weight", "block_count"]).to_csv(
        path,
        index=False,
    )

def _metrics(
    membership: np.ndarray,
    node_weights: np.ndarray,
    *,
    target_max_doc_weight: float,
) -> dict[str, Any]:
    stats = _membership_weight_summary(
        membership,
        node_weights,
        min_weight=0.0,
        max_weight=float(target_max_doc_weight),
    )
    weights = _cluster_weight_values(membership, node_weights)
    return {
        "n_clusters": int(stats["n_clusters"]),
        "max_doc_weight": float(stats["max_doc_weight"]),
        "n_above_max_doc_weight": int(stats["n_above_max_doc_weight"]),
        "target_max_doc_weight": float(target_max_doc_weight),
        "max_doc_weight_ratio": (
            float(stats["max_doc_weight"]) / float(target_max_doc_weight)
            if target_max_doc_weight > 0.0
            else 0.0
        ),
        "gini_doc_weight": _gini(weights),
        "entropy_doc_weight": _normalized_entropy(weights),
        "target_max_satisfied": bool(
            target_max_doc_weight <= 0.0 or int(stats["n_above_max_doc_weight"]) == 0
        ),
    }

def _policy_args(
    *,
    resolution: float,
    target_min_doc_weight: float,
    target_max_doc_weight: float,
    seed: int,
    policy: str,
    trim_max_moves_per_cluster: int,
    apply_iterations: int,
    selection_singleton_budget: float,
) -> SimpleNamespace:
    if policy == "two_stage_quality_first":
        acceptance_mode = "quality_first"
        trim_min_delta_q = 0.0
        trim_min_delta_q_source = "mode_default"
    elif policy == "two_stage_hard_cap":
        acceptance_mode = "hard_cap"
        trim_min_delta_q = -1.0
        trim_min_delta_q_source = "mode_default"
    else:
        raise ValueError(f"Unsupported policy: {policy}")
    args = SimpleNamespace(
        graph_dir=None,
        membership=None,
        candidates=None,
        output_dir=None,
        resolution=float(resolution),
        gamma_multipliers="1.02,1.05,1.10,1.15,1.20,1.25",
        min_core_weight=25.0,
        randomness=0.01,
        repair_epsilon=0.0,
        seed=int(seed),
        pair_seeded_probes=False,
        policy="",
        max_candidates=1000,
        target_min_doc_weight=float(target_min_doc_weight),
        target_max_doc_weight=float(target_max_doc_weight),
        oversize_acceptance_mode=acceptance_mode,
        selection_mode="oversize_first",
        selection_singleton_budget=float(selection_singleton_budget),
        selection_max_selected=0,
        apply_split_repair_candidates=True,
        apply_iterations=int(apply_iterations),
        applied_membership_output=None,
        apply_min_quality_delta=0.0,
        apply_oversize_boundary_trim=True,
        trim_min_delta_q=float(trim_min_delta_q),
        trim_min_delta_q_source=trim_min_delta_q_source,
        trim_max_moves_per_cluster=int(trim_max_moves_per_cluster),
    )
    return args

def _run_source_seed(
    *,
    sample: str,
    graph_dir: Path,
    output_dir: Path,
    resolution: float,
    target_min_doc_weight: float,
    target_max_doc_weight: float,
    seed: int,
    n_iterations: int,
    policies: tuple[str, ...],
    apply_iterations: int,
    trim_max_moves_per_cluster: int,
    selection_singleton_budget: float,
    force: bool,
) -> list[dict[str, Any]]:
    seed_dir = output_dir / "source_seed_sweep_runs" / _safe_slug(sample) / f"seed_{seed}"
    summary_path = seed_dir / "prepare_summary.json"
    src, dst, weight, node_weights = _load_graph_arrays(graph_dir)
    graph = build_leiden_graph(
        edges_src=src,
        edges_dst=dst,
        edges_weight=weight,
        n_nodes=int(node_weights.shape[0]),
        node_weights=np.asarray(node_weights, dtype=np.float64),
    )
    raw_graph = sciscape_leiden.load_graph_raw_files(
        int(node_weights.shape[0]),
        str(graph_dir / "src.u32.bin"),
        str(graph_dir / "dst.u32.bin"),
        str(graph_dir / "weight.f64.bin"),
        str(graph_dir / "node_weights.f64.bin"),
    )

    if summary_path.exists() and not force:
        prepare_summary = _read_json(summary_path)
        small_membership_path = _repo_path(prepare_summary["paths"]["membership"])
        assert small_membership_path is not None
        table = pd.read_parquet(small_membership_path)
        small_membership = table.sort_values("node_idx")["cluster"].to_numpy(dtype=np.uint64)
    else:
        t0 = time.perf_counter()
        seed_dir.mkdir(parents=True, exist_ok=True)
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
        small_membership = np.asarray(post.membership, dtype=np.uint64)
        membership_path = seed_dir / "membership.parquet"
        _write_membership(membership_path, small_membership)
        candidates_path = seed_dir / "oversize_doc_weight_candidates.csv"
        _write_candidates(
            candidates_path,
            small_membership,
            np.asarray(node_weights, dtype=np.float64),
            float(target_max_doc_weight),
        )
        small_metrics = _metrics(
            small_membership,
            np.asarray(node_weights, dtype=np.float64),
            target_max_doc_weight=float(target_max_doc_weight),
        )
        prepare_summary = {
            "sample": sample,
            "seed": int(seed),
            "resolution": float(resolution),
            "n_iterations": int(n_iterations),
            "target_min_doc_weight": float(target_min_doc_weight),
            "target_max_doc_weight": float(target_max_doc_weight),
            "raw_quality": float(raw.quality),
            "raw_n_clusters": int(raw.n_clusters),
            "post_quality": float(
                graph.cpm_quality(small_membership, resolution=float(resolution))
            ),
            "post_n_clusters": small_metrics["n_clusters"],
            "post_max_cluster_size": small_metrics["max_doc_weight"],
            "post_n_clusters_gt_target_max": small_metrics["n_above_max_doc_weight"],
            "post_n_lt_min_size": int(
                _membership_weight_summary(
                    small_membership,
                    np.asarray(node_weights, dtype=np.float64),
                    min_weight=float(target_min_doc_weight),
                    max_weight=float(target_max_doc_weight),
                )["n_lt_min_doc_weight"]
            ),
            "paths": {
                "graph_dir": _rel(graph_dir),
                "membership": _rel(membership_path),
                "oversize_candidates": _rel(candidates_path),
            },
            "elapsed_sec": time.perf_counter() - t0,
        }
        summary_path.write_text(
            json.dumps(prepare_summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    node_weights_array = np.asarray(node_weights, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    small_metrics = _metrics(
        small_membership,
        node_weights_array,
        target_max_doc_weight=float(target_max_doc_weight),
    )
    rows.append(
        {
            "sample": sample,
            "seed": int(seed),
            "policy": "small_only",
            "membership_role": "effective",
            "status": "committed",
            "accepted_for_contraction": True,
            "fallback_used": False,
            "delta_q": 0.0,
            "split_repair_exact_delta_q": 0.0,
            "trim_exact_delta_q": 0.0,
            "trim_moves": 0,
            "trim_moves_proposed": 0,
            "quality_floor_limited": False,
            "membership_path": prepare_summary["paths"]["membership"],
            "diagnostic_for_policy": None,
            **small_metrics,
        }
    )

    gamma_multipliers = np.asarray([1.02, 1.05, 1.10, 1.15, 1.20, 1.25], dtype=np.float64)
    for policy in policies:
        run_dir = seed_dir / policy
        policy_summary_path = run_dir / "iterative_split_repair_apply_summary.json"
        if policy_summary_path.exists() and not force:
            summary = _read_json(policy_summary_path)
        else:
            print(f"source_seed {sample} seed={seed} policy={policy}", flush=True)
            args = _policy_args(
                resolution=float(resolution),
                target_min_doc_weight=float(target_min_doc_weight),
                target_max_doc_weight=float(target_max_doc_weight),
                seed=int(seed),
                policy=policy,
                trim_max_moves_per_cluster=int(trim_max_moves_per_cluster),
                apply_iterations=int(apply_iterations),
                selection_singleton_budget=float(selection_singleton_budget),
            )
            summary = _run_iterative_apply(
                raw_graph,
                small_membership,
                node_weights_array,
                gamma_multipliers,
                run_dir,
                args,
                [],
            )
        status = str(summary.get("status"))
        accepted = status == "committed"
        final_metrics = summary.get("final_membership", {})
        role = "effective" if accepted else "diagnostic"
        paths = summary.get("paths", {})
        membership_path = paths.get("applied_membership") or paths.get("diagnostic_membership")
        row_metrics = {
            "n_clusters": int(final_metrics.get("n_clusters", 0)),
            "max_doc_weight": float(final_metrics.get("max_doc_weight", 0.0)),
            "n_above_max_doc_weight": int(
                final_metrics.get("n_above_max_doc_weight", 0)
            ),
            "target_max_doc_weight": float(target_max_doc_weight),
            "max_doc_weight_ratio": (
                float(final_metrics.get("max_doc_weight", 0.0))
                / float(target_max_doc_weight)
                if target_max_doc_weight > 0.0
                else 0.0
            ),
            "target_max_satisfied": bool(summary.get("target_max_satisfied", False)),
        }
        membership_for_weight = small_membership
        if membership_path:
            table = pd.read_parquet(_repo_path(membership_path))
            membership_for_weight = table.sort_values("node_idx")["cluster"].to_numpy(dtype=np.uint64)
        weights = _cluster_weight_values(membership_for_weight, node_weights_array)
        row_metrics["gini_doc_weight"] = _gini(weights)
        row_metrics["entropy_doc_weight"] = _normalized_entropy(weights)
        rows.append(
            {
                "sample": sample,
                "seed": int(seed),
                "policy": policy,
                "membership_role": role,
                "status": status,
                "accepted_for_contraction": accepted,
                "fallback_used": not accepted,
                "delta_q": float(summary.get("exact_delta_q_total", 0.0)),
                "split_repair_exact_delta_q": float(
                    summary.get("split_repair_exact_delta_q", 0.0)
                ),
                "trim_exact_delta_q": float(summary.get("trim_exact_delta_q", 0.0)),
                "trim_moves": int((summary.get("trim") or {}).get("n_moves", 0)),
                "trim_moves_proposed": int(
                    (summary.get("trim") or {}).get("n_moves_proposed", 0)
                ),
                "quality_floor_limited": bool(
                    (summary.get("trim") or {}).get("quality_floor_limited", False)
                ),
                "membership_path": membership_path,
                "diagnostic_for_policy": policy if not accepted else None,
                **row_metrics,
            }
        )
        if not accepted:
            rows.append(
                {
                    "sample": sample,
                    "seed": int(seed),
                    "policy": policy,
                    "membership_role": "effective",
                    "status": status,
                    "accepted_for_contraction": False,
                    "fallback_used": True,
                    "delta_q": 0.0,
                    "split_repair_exact_delta_q": 0.0,
                    "trim_exact_delta_q": 0.0,
                    "trim_moves": 0,
                    "trim_moves_proposed": 0,
                    "quality_floor_limited": False,
                    "membership_path": prepare_summary["paths"]["membership"],
                    "diagnostic_for_policy": None,
                    **small_metrics,
                }
            )
    return rows

def _policy_summary(df: pd.DataFrame) -> pd.DataFrame:
    effective = df[df["membership_role"] == "effective"].copy()
    rows: list[dict[str, Any]] = []
    for policy, group in effective.groupby("policy", sort=False):
        rows.append(
            {
                "policy": policy,
                "n_rows": int(len(group)),
                "accepted_rate": float(group["accepted_for_contraction"].mean()),
                "fallback_rate": float(group["fallback_used"].mean()),
                "mean_delta_q": float(group["delta_q"].mean()),
                "mean_max_ratio": float(group["max_doc_weight_ratio"].mean()),
                "mean_oversize_count": float(group["n_above_max_doc_weight"].mean()),
                "mean_gini": float(group["gini_doc_weight"].mean()),
                "target_satisfied_rate": float(group["target_max_satisfied"].mean()),
            }
        )
    return pd.DataFrame(rows)

def _quality_first_pairwise(df: pd.DataFrame) -> pd.DataFrame:
    effective = df[df["membership_role"] == "effective"].copy()
    qf = effective[effective["policy"] == "two_stage_quality_first"]
    small = effective[effective["policy"] == "small_only"]
    if qf.empty or small.empty:
        return pd.DataFrame()
    columns = [
        "sample",
        "seed",
        "max_doc_weight",
        "max_doc_weight_ratio",
        "n_above_max_doc_weight",
        "gini_doc_weight",
        "delta_q",
    ]
    merged = qf[columns].merge(
        small[columns],
        on=["sample", "seed"],
        suffixes=("_quality_first", "_small_only"),
    )
    merged["delta_max_ratio"] = (
        merged["max_doc_weight_ratio_quality_first"]
        - merged["max_doc_weight_ratio_small_only"]
    )
    merged["delta_oversize_count"] = (
        merged["n_above_max_doc_weight_quality_first"]
        - merged["n_above_max_doc_weight_small_only"]
    )
    merged["delta_gini"] = (
        merged["gini_doc_weight_quality_first"] - merged["gini_doc_weight_small_only"]
    )
    return merged

def _hard_cap_diagnostic_pairwise(df: pd.DataFrame) -> pd.DataFrame:
    diagnostics = df[
        (df["membership_role"] == "diagnostic")
        & (df["diagnostic_for_policy"] == "two_stage_hard_cap")
    ].copy()
    effective = df[
        (df["membership_role"] == "effective")
        & (df["policy"] == "two_stage_hard_cap")
    ].copy()
    if diagnostics.empty or effective.empty:
        return pd.DataFrame()
    columns = [
        "sample",
        "seed",
        "max_doc_weight",
        "max_doc_weight_ratio",
        "n_above_max_doc_weight",
        "gini_doc_weight",
        "delta_q",
        "status",
    ]
    merged = diagnostics[columns].merge(
        effective[columns],
        on=["sample", "seed"],
        suffixes=("_diagnostic", "_effective"),
    )
    merged["delta_max_ratio"] = (
        merged["max_doc_weight_ratio_diagnostic"]
        - merged["max_doc_weight_ratio_effective"]
    )
    merged["delta_oversize_count"] = (
        merged["n_above_max_doc_weight_diagnostic"]
        - merged["n_above_max_doc_weight_effective"]
    )
    merged["delta_gini"] = (
        merged["gini_doc_weight_diagnostic"] - merged["gini_doc_weight_effective"]
    )
    return merged

def _diagnostic_summary(pairwise: pd.DataFrame) -> pd.DataFrame:
    if pairwise.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "n_pairs": int(len(pairwise)),
                "mean_delta_q_diagnostic": float(pairwise["delta_q_diagnostic"].mean()),
                "mean_delta_max_ratio": float(pairwise["delta_max_ratio"].mean()),
                "mean_delta_oversize_count": float(
                    pairwise["delta_oversize_count"].mean()
                ),
                "mean_delta_gini": float(pairwise["delta_gini"].mean()),
                "pairs_with_lower_max_ratio": int(
                    (pairwise["delta_max_ratio"] < 0).sum()
                ),
                "pairs_with_lower_oversize_count": int(
                    (pairwise["delta_oversize_count"] < 0).sum()
                ),
                "pairs_with_lower_gini": int((pairwise["delta_gini"] < 0).sum()),
            }
        ]
    )

def _plot_source_seed(pairwise: pd.DataFrame, output_dir: Path) -> Path | None:
    if pairwise.empty:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    metrics = [
        ("delta_max_ratio", "max/target ratio delta"),
        ("delta_oversize_count", "oversize-count delta"),
        ("delta_gini", "Gini delta"),
    ]
    for ax, (column, title) in zip(axes, metrics, strict=True):
        ax.bar(pairwise["seed"].astype(str), pairwise[column], color="#54a24b")
        ax.axhline(0.0, color="#222222", linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("source Leiden seed")
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("quality_first - small_only")
    fig.suptitle("Source-level seed pilot: quality_first deltas")
    fig.tight_layout()
    out_path = output_dir / "figure7_source_seed_pilot.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path

def _write_report(
    *,
    effects: pd.DataFrame,
    policy_summary: pd.DataFrame,
    pairwise: pd.DataFrame,
    hard_cap_summary: pd.DataFrame,
    output_dir: Path,
    figure_path: Path | None,
) -> Path:
    lines = [
        "# Source-Level Seed Sweep Pilot",
        "",
        "This pilot varies the source-level Leiden seed before small repair and two-stage hierarchy postprocess. It currently targets prepared graph artifacts rather than rebuilding field samples from raw OpenAlex data.",
        "",
        f"- Rows: {len(effects)}",
        f"- Samples: {', '.join(sorted(effects['sample'].unique()))}",
        f"- Seeds: {', '.join(str(seed) for seed in sorted(effects['seed'].unique()))}",
        "",
    ]
    if not pairwise.empty:
        lines.extend(
            [
                "## Quality-First vs Small-Only",
                "",
                f"- Mean max/target ratio delta: {pairwise['delta_max_ratio'].mean():.6g}",
                f"- Mean oversize-count delta: {pairwise['delta_oversize_count'].mean():.6g}",
                f"- Mean Gini delta: {pairwise['delta_gini'].mean():.6g}",
                f"- Rows with lower max/target ratio: {int((pairwise['delta_max_ratio'] < 0).sum())} / {len(pairwise)}",
                f"- Rows with lower oversize count: {int((pairwise['delta_oversize_count'] < 0).sum())} / {len(pairwise)}",
                f"- Rows with lower Gini: {int((pairwise['delta_gini'] < 0).sum())} / {len(pairwise)}",
                "",
            ]
        )
    if not hard_cap_summary.empty:
        lines.extend(
            [
                "## Hard-Cap Diagnostic",
                "",
                _markdown_table(hard_cap_summary),
                "",
            ]
        )
    lines.extend(
        [
            "## Tables",
            "",
            "- `source_seed_sweep_effects.csv` / `.parquet`",
            "- `source_seed_sweep_policy_summary.csv` / `.parquet`",
            "- `source_seed_sweep_quality_first_vs_small_only.csv` / `.parquet`",
            "- `source_seed_sweep_hard_cap_diagnostics.csv` / `.parquet`",
            "- `source_seed_sweep_hard_cap_diagnostic_summary.csv` / `.parquet`",
            "",
        ]
    )
    if figure_path is not None:
        lines.extend(["## Figure", "", f"- `{figure_path.name}`", ""])
    if not policy_summary.empty:
        lines.extend(["## Policy Summary", "", _markdown_table(policy_summary)])
    report_path = output_dir / "source_seed_sweep_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--validation-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--samples", default=DEFAULT_SAMPLE)
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
    )
    parser.add_argument("--policies", default=",".join(DEFAULT_POLICIES))
    parser.add_argument("--apply-iterations", type=int, default=4)
    parser.add_argument("--trim-max-moves-per-cluster", type=int, default=100)
    parser.add_argument("--selection-singleton-budget", type=float, default=100.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-figure", action="store_true")
    args = parser.parse_args()

    results_dir = _repo_path(args.results_dir)
    validation_dir = _repo_path(args.validation_dir)
    assert results_dir is not None and validation_dir is not None
    cross_path = results_dir / "postprocess_policy_matrix_cross_sample" / "cross_sample_summary.json"
    cross_summary = _read_json(cross_path) if cross_path.exists() else None
    configs = _sample_configs(results_dir, cross_summary)
    samples = [sample.strip() for sample in str(args.samples).split(",") if sample.strip()]
    seeds = _parse_int_list(str(args.seeds))
    policies = tuple(policy.strip() for policy in str(args.policies).split(",") if policy.strip())

    rows: list[dict[str, Any]] = []
    for sample in samples:
        cfg = configs[sample]
        if cfg.node_weights_path is None:
            raise FileNotFoundError(f"No node weights path for sample {sample}")
        graph_dir = cfg.node_weights_path.parent
        prepare = _read_json(cfg.prepare_summary_path) if cfg.prepare_summary_path else {}
        resolution = float(prepare.get("resolution") or 0.01)
        target_min = float(cfg.target_min_doc_weight or prepare.get("min_size") or 50.0)
        target_max = float(cfg.target_max_doc_weight or prepare.get("target_max_doc_weight") or 0.0)
        for seed in seeds:
            print(f"source_seed_prepare {sample} seed={seed}", flush=True)
            rows.extend(
                _run_source_seed(
                    sample=sample,
                    graph_dir=graph_dir,
                    output_dir=validation_dir,
                    resolution=resolution,
                    target_min_doc_weight=target_min,
                    target_max_doc_weight=target_max,
                    seed=int(seed),
                    n_iterations=int(prepare.get("n_iterations") or 5),
                    policies=policies,
                    apply_iterations=int(args.apply_iterations),
                    trim_max_moves_per_cluster=int(args.trim_max_moves_per_cluster),
                    selection_singleton_budget=float(args.selection_singleton_budget),
                    force=bool(args.force),
                )
            )

    effects = pd.DataFrame(rows)
    policy_summary = _policy_summary(effects)
    pairwise = _quality_first_pairwise(effects)
    hard_cap_pairwise = _hard_cap_diagnostic_pairwise(effects)
    hard_cap_summary = _diagnostic_summary(hard_cap_pairwise)
    _write_table(effects, validation_dir / "source_seed_sweep_effects")
    _write_table(policy_summary, validation_dir / "source_seed_sweep_policy_summary")
    _write_table(pairwise, validation_dir / "source_seed_sweep_quality_first_vs_small_only")
    _write_table(
        hard_cap_pairwise,
        validation_dir / "source_seed_sweep_hard_cap_diagnostics",
    )
    _write_table(
        hard_cap_summary,
        validation_dir / "source_seed_sweep_hard_cap_diagnostic_summary",
    )
    figure_path = None if args.skip_figure else _plot_source_seed(pairwise, validation_dir)
    report_path = _write_report(
        effects=effects,
        policy_summary=policy_summary,
        pairwise=pairwise,
        hard_cap_summary=hard_cap_summary,
        output_dir=validation_dir,
        figure_path=figure_path,
    )
    print(f"Saved source seed sweep to {_rel(validation_dir / 'source_seed_sweep_effects.csv')}")
    print(f"Report: {_rel(report_path)}")

if __name__ == "__main__":
    main()
