"""Stress-test next-level hierarchy propagation across target multipliers.

The default next-level rerun uses a coarser adaptive target, which can hide
propagation differences because every policy satisfies the next-level cap. This
script reruns the same contraction -> gamma sweep -> Leiden -> small repair
pipeline for stricter source-level target multipliers.
"""

from __future__ import annotations

import argparse
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

from evaluate_hierarchy_postprocess import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RESULTS_DIR,
    _markdown_table,
    _repo_path,
    _sample_configs,
    _write_table,
)
from run_hierarchy_postprocess_next_level import (  # noqa: E402
    DEFAULT_POLICIES,
    _candidate_rows,
    _flatten_summary,
    _load_graph_arrays,
    _read_json,
    _rel,
    _run_one,
    _safe_slug,
)

DEFAULT_MULTIPLIERS = (1.0, 1.5, 2.0, 3.0)

def _parse_float_list(value: str) -> list[float]:
    parsed = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise ValueError("Expected at least one numeric multiplier.")
    if any(item <= 0.0 for item in parsed):
        raise ValueError("Target multipliers must be positive.")
    return parsed

def _target_slug(multiplier: float) -> str:
    return _safe_slug(f"target_x{multiplier:g}".replace(".", "p"))

def _policy_summary(df: pd.DataFrame) -> pd.DataFrame:
    effective = df[df["membership_role"] == "effective"].copy()
    if effective.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (multiplier, policy), group in effective.groupby(
        ["target_multiplier", "source_policy"], sort=True
    ):
        rows.append(
            {
                "target_multiplier": float(multiplier),
                "source_policy": policy,
                "n_runs": int(len(group)),
                "accepted_rate": float(group["source_accepted_for_contraction"].mean()),
                "fallback_rate": float(group["source_fallback_used"].mean()),
                "mean_post_max_ratio": float(group["post_max_doc_weight_ratio"].mean()),
                "mean_post_oversize_count": float(group["post_n_above_max_doc_weight"].mean()),
                "mean_post_gini": float(group["post_gini_doc_weight"].mean()),
                "mean_parent_max_child_share": float(
                    group["post_parent_max_child_share_weighted_mean"].mean()
                ),
                "mean_post_n_clusters": float(group["post_n_clusters"].mean()),
            }
        )
    return pd.DataFrame(rows)

def _quality_first_pairwise(df: pd.DataFrame) -> pd.DataFrame:
    effective = df[df["membership_role"] == "effective"].copy()
    small = effective[effective["source_policy"] == "small_only"]
    qf = effective[effective["source_policy"] == "two_stage_quality_first"]
    if small.empty or qf.empty:
        return pd.DataFrame()
    merge_keys = ["sample", "target_multiplier"]
    if "seed" in effective.columns:
        merge_keys.append("seed")
    columns = [
        "sample",
        "target_multiplier",
        *([] if "seed" not in effective.columns else ["seed"]),
        "next_target_max_doc_weight",
        "post_max_doc_weight",
        "post_max_doc_weight_ratio",
        "post_n_above_max_doc_weight",
        "post_gini_doc_weight",
        "post_parent_max_child_share_weighted_mean",
        "post_n_clusters",
        "post_quality",
    ]
    merged = qf[columns].merge(
        small[columns],
        on=merge_keys,
        suffixes=("_quality_first", "_small_only"),
    )
    if merged.empty:
        return merged
    merged["delta_post_max_doc_weight"] = (
        merged["post_max_doc_weight_quality_first"]
        - merged["post_max_doc_weight_small_only"]
    )
    merged["delta_post_max_ratio"] = (
        merged["post_max_doc_weight_ratio_quality_first"]
        - merged["post_max_doc_weight_ratio_small_only"]
    )
    merged["delta_post_oversize_count"] = (
        merged["post_n_above_max_doc_weight_quality_first"]
        - merged["post_n_above_max_doc_weight_small_only"]
    )
    merged["delta_post_gini"] = (
        merged["post_gini_doc_weight_quality_first"]
        - merged["post_gini_doc_weight_small_only"]
    )
    merged["delta_parent_max_child_share"] = (
        merged["post_parent_max_child_share_weighted_mean_quality_first"]
        - merged["post_parent_max_child_share_weighted_mean_small_only"]
    )
    merged["delta_post_n_clusters"] = (
        merged["post_n_clusters_quality_first"] - merged["post_n_clusters_small_only"]
    )
    merged["delta_post_quality"] = (
        merged["post_quality_quality_first"] - merged["post_quality_small_only"]
    )
    return merged

def _hard_cap_diagnostic_pairwise(df: pd.DataFrame) -> pd.DataFrame:
    diagnostics = df[
        (df["membership_role"] == "diagnostic")
        & (df["diagnostic_for_policy"] == "two_stage_hard_cap")
    ].copy()
    effective = df[
        (df["membership_role"] == "effective")
        & (df["source_policy"] == "two_stage_hard_cap")
    ].copy()
    if diagnostics.empty or effective.empty:
        return pd.DataFrame()
    merge_keys = ["sample", "target_multiplier"]
    if "seed" in df.columns:
        merge_keys.append("seed")
    columns = [
        "sample",
        "target_multiplier",
        *([] if "seed" not in df.columns else ["seed"]),
        "post_max_doc_weight",
        "post_max_doc_weight_ratio",
        "post_n_above_max_doc_weight",
        "post_gini_doc_weight",
        "post_parent_max_child_share_weighted_mean",
        "post_n_clusters",
        "post_quality",
        "source_status",
        "source_fallback_used",
    ]
    merged = diagnostics[columns].merge(
        effective[columns],
        on=merge_keys,
        suffixes=("_diagnostic", "_effective"),
    )
    if merged.empty:
        return merged
    merged["delta_post_max_doc_weight"] = (
        merged["post_max_doc_weight_diagnostic"]
        - merged["post_max_doc_weight_effective"]
    )
    merged["delta_post_max_ratio"] = (
        merged["post_max_doc_weight_ratio_diagnostic"]
        - merged["post_max_doc_weight_ratio_effective"]
    )
    merged["delta_post_oversize_count"] = (
        merged["post_n_above_max_doc_weight_diagnostic"]
        - merged["post_n_above_max_doc_weight_effective"]
    )
    merged["delta_post_gini"] = (
        merged["post_gini_doc_weight_diagnostic"]
        - merged["post_gini_doc_weight_effective"]
    )
    merged["delta_parent_max_child_share"] = (
        merged["post_parent_max_child_share_weighted_mean_diagnostic"]
        - merged["post_parent_max_child_share_weighted_mean_effective"]
    )
    merged["delta_post_n_clusters"] = (
        merged["post_n_clusters_diagnostic"] - merged["post_n_clusters_effective"]
    )
    merged["delta_post_quality"] = (
        merged["post_quality_diagnostic"] - merged["post_quality_effective"]
    )
    return merged

def _hard_cap_diagnostic_summary(pairwise: pd.DataFrame) -> pd.DataFrame:
    if pairwise.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for multiplier, group in pairwise.groupby("target_multiplier", sort=True):
        rows.append(
            {
                "target_multiplier": float(multiplier),
                "n_diagnostic_rows": int(len(group)),
                "mean_delta_max_ratio": float(group["delta_post_max_ratio"].mean()),
                "mean_delta_oversize_count": float(
                    group["delta_post_oversize_count"].mean()
                ),
                "mean_delta_gini": float(group["delta_post_gini"].mean()),
                "mean_delta_parent_max_child_share": float(
                    group["delta_parent_max_child_share"].mean()
                ),
                "mean_delta_post_quality": float(group["delta_post_quality"].mean()),
                "rows_with_lower_max_ratio": int(
                    (group["delta_post_max_ratio"] < 0).sum()
                ),
                "rows_with_lower_oversize_count": int(
                    (group["delta_post_oversize_count"] < 0).sum()
                ),
                "rows_with_lower_gini": int((group["delta_post_gini"] < 0).sum()),
            }
        )
    return pd.DataFrame(rows)

def _plot_sweep(comparison: pd.DataFrame, output_dir: Path) -> Path | None:
    if comparison.empty:
        return None
    policies = [
        "small_only",
        "oversize_split_only",
        "two_stage_quality_first",
        "two_stage_hard_cap",
    ]
    colors = {
        "small_only": "#666666",
        "oversize_split_only": "#4c78a8",
        "two_stage_quality_first": "#54a24b",
        "two_stage_hard_cap": "#e45756",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharex=True)
    for policy in policies:
        group = comparison[comparison["source_policy"] == policy].sort_values(
            "target_multiplier"
        )
        if group.empty:
            continue
        label = policy.replace("two_stage_", "")
        axes[0].plot(
            group["target_multiplier"],
            group["mean_post_oversize_count"],
            marker="o",
            color=colors[policy],
            label=label,
        )
        axes[1].plot(
            group["target_multiplier"],
            group["mean_post_max_ratio"],
            marker="o",
            color=colors[policy],
            label=label,
        )
    axes[0].set_ylabel("mean next-level oversize count")
    axes[1].set_ylabel("mean next-level max doc weight / target")
    for ax in axes:
        ax.set_xlabel("source-level target multiplier")
        ax.axhline(1.0, color="#222222", linestyle="--", linewidth=1, alpha=0.7)
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Next-level propagation under stricter target caps")
    fig.tight_layout()
    out_path = output_dir / "figure5_next_level_target_sweep.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path

def _write_report(
    *,
    effects: pd.DataFrame,
    comparison: pd.DataFrame,
    pairwise: pd.DataFrame,
    hard_cap_pairwise: pd.DataFrame,
    hard_cap_summary: pd.DataFrame,
    output_dir: Path,
    figure_path: Path | None,
) -> Path:
    lines = [
        "# Next-Level Target Sweep",
        "",
        "This stress test repeats the actual next-level contraction and Leiden rerun at source-level target multipliers. It is intended to expose hierarchy propagation effects that are hidden by the coarser adaptive next-level target.",
        "",
        "## Run Summary",
        "",
        f"- Rows: {len(effects)}",
        f"- Effective rows: {int((effects['membership_role'] == 'effective').sum())}",
        f"- Diagnostic rows: {int((effects['membership_role'] == 'diagnostic').sum())}",
        "- Target definition: next target max doc weight = source target max doc weight x multiplier.",
        "",
    ]
    if not pairwise.empty:
        strict_multiplier = float(pairwise["target_multiplier"].min())
        strict_pairwise = pairwise[pairwise["target_multiplier"] == strict_multiplier]
        strict_comparison = comparison[comparison["target_multiplier"] == strict_multiplier]
        strict_small = strict_comparison[
            strict_comparison["source_policy"] == "small_only"
        ]
        strict_qf = strict_comparison[
            strict_comparison["source_policy"] == "two_stage_quality_first"
        ]
        lines.extend(
            [
                "## Quality-First vs Small-Only",
                "",
                f"- Mean max/target ratio delta: {pairwise['delta_post_max_ratio'].mean():.6g}",
                f"- Mean oversize-count delta: {pairwise['delta_post_oversize_count'].mean():.6g}",
                f"- Mean Gini delta: {pairwise['delta_post_gini'].mean():.6g}",
                f"- Mean parent max-child share delta: {pairwise['delta_parent_max_child_share'].mean():.6g}",
                f"- Rows with lower max/target ratio: {int((pairwise['delta_post_max_ratio'] < 0).sum())} / {len(pairwise)}",
                f"- Rows with lower oversize count: {int((pairwise['delta_post_oversize_count'] < 0).sum())} / {len(pairwise)}",
                "",
            ]
        )
        if not strict_pairwise.empty and not strict_small.empty and not strict_qf.empty:
            lines.extend(
                [
                    "## Strictest Target Readout",
                    "",
                    f"- Strictest multiplier: {strict_multiplier:g}x source-level target.",
                    f"- small_only mean oversize count: {float(strict_small['mean_post_oversize_count'].iloc[0]):.6g}",
                    f"- quality_first mean oversize count: {float(strict_qf['mean_post_oversize_count'].iloc[0]):.6g}",
                    f"- small_only mean max/target ratio: {float(strict_small['mean_post_max_ratio'].iloc[0]):.6g}",
                    f"- quality_first mean max/target ratio: {float(strict_qf['mean_post_max_ratio'].iloc[0]):.6g}",
                    f"- quality_first mean Gini delta at strictest target: {strict_pairwise['delta_post_gini'].mean():.6g}",
                    f"- quality_first mean parent concentration delta at strictest target: {strict_pairwise['delta_parent_max_child_share'].mean():.6g}",
                    "",
                    "Interpretation: quality_first improves next-level concentration and strict-target oversize count, but it does not consistently minimize the single largest next-level parent. The H2 claim should therefore be framed as reduced propagation of concentration/oversize pressure rather than a blanket max-weight dominance claim.",
                    "",
                ]
            )
    if not hard_cap_pairwise.empty and not hard_cap_summary.empty:
        lines.extend(
            [
                "## Hard-Cap Diagnostic Readout",
                "",
                "Diagnostic hard-cap rows use rejected memberships and are not the hierarchy input. They show what the strict operational variant would have done if fallback had not protected the hierarchy.",
                f"- Diagnostic rows compared against effective hard_cap fallback: {len(hard_cap_pairwise)}",
                f"- Mean diagnostic max/target ratio delta: {hard_cap_pairwise['delta_post_max_ratio'].mean():.6g}",
                f"- Mean diagnostic oversize-count delta: {hard_cap_pairwise['delta_post_oversize_count'].mean():.6g}",
                f"- Mean diagnostic Gini delta: {hard_cap_pairwise['delta_post_gini'].mean():.6g}",
                f"- Mean diagnostic next-level quality delta: {hard_cap_pairwise['delta_post_quality'].mean():.6g}",
                f"- Rows with lower diagnostic max/target ratio: {int((hard_cap_pairwise['delta_post_max_ratio'] < 0).sum())} / {len(hard_cap_pairwise)}",
                f"- Rows with lower diagnostic oversize count: {int((hard_cap_pairwise['delta_post_oversize_count'] < 0).sum())} / {len(hard_cap_pairwise)}",
                "",
                "Interpretation: if diagnostic hard_cap improves balance only inconsistently, or only by relying on rejected/fallback-prone memberships, it should remain a strict operational variant rather than the paper default.",
                "",
                _markdown_table(hard_cap_summary),
                "",
            ]
        )
    lines.extend(
        [
            "## Tables",
            "",
            "- `next_level_target_sweep_effects.csv` / `.parquet`",
            "- `next_level_target_sweep_policy_comparison.csv` / `.parquet`",
            "- `next_level_target_sweep_quality_first_vs_small_only.csv` / `.parquet`",
            "- `next_level_target_sweep_hard_cap_diagnostics.csv` / `.parquet`",
            "- `next_level_target_sweep_hard_cap_diagnostic_summary.csv` / `.parquet`",
            "",
        ]
    )
    if figure_path is not None:
        lines.extend(["## Figure", "", f"- `{figure_path.name}`", ""])
    if not comparison.empty:
        lines.extend(["## Policy Comparison", "", _markdown_table(comparison)])
    report_path = output_dir / "next_level_target_sweep_report.md"
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
    parser.add_argument(
        "--multipliers",
        default=",".join(f"{value:g}" for value in DEFAULT_MULTIPLIERS),
        help="Comma-separated source-level target multipliers.",
    )
    parser.add_argument("--include-diagnostics", action="store_true")
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
    multipliers = _parse_float_list(str(args.multipliers))

    cross_path = results_dir / "postprocess_policy_matrix_cross_sample" / "cross_sample_summary.json"
    cross_summary = _read_json(cross_path) if cross_path.exists() else None
    configs = _sample_configs(results_dir, cross_summary)
    rows = _candidate_rows(
        eval_df,
        policies=policies,
        include_diagnostics=bool(args.include_diagnostics),
    )

    graph_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    total_weight_cache: dict[str, float] = {}
    flat_rows: list[dict[str, Any]] = []
    total_jobs = len(rows) * len(multipliers)
    job_idx = 0

    for multiplier in multipliers:
        multiplier_dir = validation_dir / "next_level_target_sweep_runs" / _target_slug(multiplier)
        for row in rows:
            job_idx += 1
            sample = str(row["sample"])
            cfg = configs[sample]
            graph_dir = (
                cfg.node_weights_path.parent
                if cfg.node_weights_path
                else cfg.sample_dir / "graph"
            )
            if sample not in graph_cache:
                print(f"[{job_idx}/{total_jobs}] loading graph {sample}", flush=True)
                graph_cache[sample] = _load_graph_arrays(graph_dir)
                total_weight_cache[sample] = float(np.asarray(graph_cache[sample][3]).sum())

            source_target = float(row.get("target_max_doc_weight") or 0.0)
            if source_target <= 0.0:
                continue
            total_doc_weight = total_weight_cache[sample]
            next_target_pct = 100.0 * source_target * float(multiplier) / total_doc_weight
            print(
                f"[{job_idx}/{total_jobs}] target x{multiplier:g} {sample} "
                f"{row['rerun_run']} role={row['membership_role']} "
                f"target_pct={next_target_pct:.4g}",
                flush=True,
            )
            summary = _run_one(
                row=row,
                graph_arrays=graph_cache[sample],
                graph_dir=graph_dir,
                output_dir=multiplier_dir,
                next_target_pct=next_target_pct,
                next_min_doc_weight=float(args.next_min_doc_weight),
                seed=int(args.seed),
                force=bool(args.force),
            )
            flat = _flatten_summary(summary)
            flat.update(
                {
                    "target_multiplier": float(multiplier),
                    "source_level_target_max_doc_weight": source_target,
                    "target_definition": "source_level_target_max_doc_weight_x_multiplier",
                    "sweep_run_dir": _rel(multiplier_dir),
                }
            )
            flat_rows.append(flat)

    effects = pd.DataFrame(flat_rows)
    comparison = _policy_summary(effects)
    pairwise = _quality_first_pairwise(effects)
    hard_cap_pairwise = _hard_cap_diagnostic_pairwise(effects)
    hard_cap_summary = _hard_cap_diagnostic_summary(hard_cap_pairwise)
    _write_table(effects, validation_dir / "next_level_target_sweep_effects")
    _write_table(comparison, validation_dir / "next_level_target_sweep_policy_comparison")
    _write_table(pairwise, validation_dir / "next_level_target_sweep_quality_first_vs_small_only")
    _write_table(
        hard_cap_pairwise,
        validation_dir / "next_level_target_sweep_hard_cap_diagnostics",
    )
    _write_table(
        hard_cap_summary,
        validation_dir / "next_level_target_sweep_hard_cap_diagnostic_summary",
    )
    figure_path = None if args.skip_figure else _plot_sweep(comparison, validation_dir)
    report_path = _write_report(
        effects=effects,
        comparison=comparison,
        pairwise=pairwise,
        hard_cap_pairwise=hard_cap_pairwise,
        hard_cap_summary=hard_cap_summary,
        output_dir=validation_dir,
        figure_path=figure_path,
    )
    print(
        "Saved target sweep effects to "
        f"{_rel(validation_dir / 'next_level_target_sweep_effects.csv')}"
    )
    print(f"Report: {_rel(report_path)}")

if __name__ == "__main__":
    main()
