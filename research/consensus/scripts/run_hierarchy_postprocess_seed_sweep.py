"""Evaluate next-level hierarchy postprocess stability across Leiden seeds.

This script reuses the same policy memberships as the target sweep but repeats
the contracted next-level gamma sweep and Leiden pass for multiple seeds.  It
is intended to check whether the H2/H3 conclusions survive seed variation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from evaluate_hierarchy_postprocess import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RESULTS_DIR,
    _markdown_table,
    _repo_path,
    _sample_configs,
    _write_table,
)
from run_hierarchy_postprocess_next_level import (  # noqa: E402
    _candidate_rows,
    _flatten_summary,
    _load_graph_arrays,
    _read_json,
    _rel,
    _run_one,
    _safe_slug,
)
from run_hierarchy_postprocess_target_sweep import (  # noqa: E402
    _hard_cap_diagnostic_pairwise,
    _hard_cap_diagnostic_summary,
    _parse_float_list,
    _quality_first_pairwise,
    _target_slug,
)


DEFAULT_SEEDS = (11, 42, 73)
DEFAULT_POLICIES = (
    "small_only",
    "two_stage_quality_first",
    "two_stage_hard_cap",
)


def _parse_int_list(value: str) -> list[int]:
    parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise ValueError("Expected at least one integer seed.")
    return parsed


def _policy_seed_summary(df: pd.DataFrame) -> pd.DataFrame:
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
                "n_rows": int(len(group)),
                "n_samples": int(group["sample"].nunique()),
                "n_seeds": int(group["seed"].nunique()),
                "accepted_rate": float(group["source_accepted_for_contraction"].mean()),
                "fallback_rate": float(group["source_fallback_used"].mean()),
                "mean_post_max_ratio": float(group["post_max_doc_weight_ratio"].mean()),
                "std_post_max_ratio": float(group["post_max_doc_weight_ratio"].std(ddof=0)),
                "mean_post_oversize_count": float(group["post_n_above_max_doc_weight"].mean()),
                "std_post_oversize_count": float(
                    group["post_n_above_max_doc_weight"].std(ddof=0)
                ),
                "mean_post_gini": float(group["post_gini_doc_weight"].mean()),
                "std_post_gini": float(group["post_gini_doc_weight"].std(ddof=0)),
                "mean_parent_max_child_share": float(
                    group["post_parent_max_child_share_weighted_mean"].mean()
                ),
                "std_parent_max_child_share": float(
                    group["post_parent_max_child_share_weighted_mean"].std(ddof=0)
                ),
            }
        )
    return pd.DataFrame(rows)


def _pairwise_seed_summary(pairwise: pd.DataFrame) -> pd.DataFrame:
    if pairwise.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for multiplier, group in pairwise.groupby("target_multiplier", sort=True):
        rows.append(
            {
                "target_multiplier": float(multiplier),
                "n_pairs": int(len(group)),
                "mean_delta_max_ratio": float(group["delta_post_max_ratio"].mean()),
                "std_delta_max_ratio": float(group["delta_post_max_ratio"].std(ddof=0)),
                "mean_delta_oversize_count": float(
                    group["delta_post_oversize_count"].mean()
                ),
                "std_delta_oversize_count": float(
                    group["delta_post_oversize_count"].std(ddof=0)
                ),
                "mean_delta_gini": float(group["delta_post_gini"].mean()),
                "std_delta_gini": float(group["delta_post_gini"].std(ddof=0)),
                "mean_delta_parent_max_child_share": float(
                    group["delta_parent_max_child_share"].mean()
                ),
                "std_delta_parent_max_child_share": float(
                    group["delta_parent_max_child_share"].std(ddof=0)
                ),
                "pairs_with_lower_max_ratio": int(
                    (group["delta_post_max_ratio"] < 0).sum()
                ),
                "pairs_with_lower_oversize_count": int(
                    (group["delta_post_oversize_count"] < 0).sum()
                ),
                "pairs_with_lower_gini": int((group["delta_post_gini"] < 0).sum()),
                "pairs_with_lower_parent_share": int(
                    (group["delta_parent_max_child_share"] < 0).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def _plot_seed_deltas(pairwise: pd.DataFrame, output_dir: Path) -> Path | None:
    if pairwise.empty:
        return None
    metrics = [
        ("delta_post_max_ratio", "max/target ratio delta"),
        ("delta_post_oversize_count", "oversize-count delta"),
        ("delta_post_gini", "Gini delta"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=False)
    for ax, (column, title) in zip(axes, metrics, strict=True):
        values = [group[column].to_numpy() for _, group in pairwise.groupby("target_multiplier")]
        labels = [f"{multiplier:g}x" for multiplier in sorted(pairwise["target_multiplier"].unique())]
        ax.boxplot(values, tick_labels=labels, showmeans=True)
        ax.axhline(0, color="#222222", linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("target multiplier")
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("quality_first - small_only")
    fig.suptitle("Seed stability of quality_first downstream deltas")
    fig.tight_layout()
    out_path = output_dir / "figure6_seed_stability_deltas.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def _write_report(
    *,
    effects: pd.DataFrame,
    policy_summary: pd.DataFrame,
    pairwise: pd.DataFrame,
    pairwise_summary: pd.DataFrame,
    hard_cap_summary: pd.DataFrame,
    output_dir: Path,
    figure_path: Path | None,
) -> Path:
    seeds = ", ".join(str(seed) for seed in sorted(effects["seed"].unique()))
    multipliers = ", ".join(
        f"{multiplier:g}x" for multiplier in sorted(effects["target_multiplier"].unique())
    )
    lines = [
        "# Next-Level Seed Stability Sweep",
        "",
        "This sweep repeats the contracted next-level gamma sweep, Leiden pass, and small-cluster repair across multiple seeds.",
        "",
        "## Run Summary",
        "",
        f"- Rows: {len(effects)}",
        f"- Effective rows: {int((effects['membership_role'] == 'effective').sum())}",
        f"- Diagnostic rows: {int((effects['membership_role'] == 'diagnostic').sum())}",
        f"- Seeds: {seeds}",
        f"- Target multipliers: {multipliers}",
        "",
    ]
    if not pairwise_summary.empty:
        first = pairwise_summary.iloc[0]
        lines.extend(
            [
                "## Quality-First vs Small-Only",
                "",
                f"- Mean max/target ratio delta: {float(first['mean_delta_max_ratio']):.6g}",
                f"- Mean oversize-count delta: {float(first['mean_delta_oversize_count']):.6g}",
                f"- Mean Gini delta: {float(first['mean_delta_gini']):.6g}",
                f"- Mean parent concentration delta: {float(first['mean_delta_parent_max_child_share']):.6g}",
                f"- Lower max/target ratio pairs: {int(first['pairs_with_lower_max_ratio'])} / {int(first['n_pairs'])}",
                f"- Lower oversize-count pairs: {int(first['pairs_with_lower_oversize_count'])} / {int(first['n_pairs'])}",
                f"- Lower Gini pairs: {int(first['pairs_with_lower_gini'])} / {int(first['n_pairs'])}",
                f"- Lower parent concentration pairs: {int(first['pairs_with_lower_parent_share'])} / {int(first['n_pairs'])}",
                "",
                "Interpretation: a stable negative Gini/parent-share delta supports the concentration-pressure claim even when max/target ratio is not uniformly improved.",
                "",
            ]
        )
    if not hard_cap_summary.empty:
        lines.extend(
            [
                "## Hard-Cap Diagnostic Stability",
                "",
                _markdown_table(hard_cap_summary),
                "",
            ]
        )
    lines.extend(
        [
            "## Tables",
            "",
            "- `next_level_seed_sweep_effects.csv` / `.parquet`",
            "- `next_level_seed_sweep_policy_summary.csv` / `.parquet`",
            "- `next_level_seed_sweep_quality_first_vs_small_only.csv` / `.parquet`",
            "- `next_level_seed_sweep_quality_first_vs_small_only_summary.csv` / `.parquet`",
            "- `next_level_seed_sweep_hard_cap_diagnostics.csv` / `.parquet`",
            "- `next_level_seed_sweep_hard_cap_diagnostic_summary.csv` / `.parquet`",
            "",
        ]
    )
    if figure_path is not None:
        lines.extend(["## Figure", "", f"- `{figure_path.name}`", ""])
    if not policy_summary.empty:
        lines.extend(["## Policy Summary", "", _markdown_table(policy_summary)])
    report_path = output_dir / "next_level_seed_sweep_report.md"
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
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="Comma-separated Leiden seeds.",
    )
    parser.add_argument(
        "--multipliers",
        default="1",
        help="Comma-separated source-level target multipliers.",
    )
    parser.add_argument("--include-diagnostics", action="store_true", default=True)
    parser.add_argument("--no-diagnostics", dest="include_diagnostics", action="store_false")
    parser.add_argument("--next-min-doc-weight", type=float, default=100.0)
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
    seeds = _parse_int_list(str(args.seeds))
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
    total_jobs = len(seeds) * len(multipliers) * len(rows)
    job_idx = 0

    for seed in seeds:
        for multiplier in multipliers:
            run_dir = (
                validation_dir
                / "next_level_seed_sweep_runs"
                / _safe_slug(f"seed_{seed}")
                / _target_slug(multiplier)
            )
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
                    f"[{job_idx}/{total_jobs}] seed={seed} target x{multiplier:g} "
                    f"{sample} {row['rerun_run']} role={row['membership_role']}",
                    flush=True,
                )
                summary = _run_one(
                    row=row,
                    graph_arrays=graph_cache[sample],
                    graph_dir=graph_dir,
                    output_dir=run_dir,
                    next_target_pct=next_target_pct,
                    next_min_doc_weight=float(args.next_min_doc_weight),
                    seed=int(seed),
                    force=bool(args.force),
                )
                flat = _flatten_summary(summary)
                flat.update(
                    {
                        "target_multiplier": float(multiplier),
                        "source_level_target_max_doc_weight": source_target,
                        "target_definition": "source_level_target_max_doc_weight_x_multiplier",
                        "seed_sweep_run_dir": _rel(run_dir),
                    }
                )
                flat_rows.append(flat)

    effects = pd.DataFrame(flat_rows)
    policy_summary = _policy_seed_summary(effects)
    pairwise = _quality_first_pairwise(effects)
    pairwise_summary = _pairwise_seed_summary(pairwise)
    hard_cap_pairwise = _hard_cap_diagnostic_pairwise(effects)
    hard_cap_summary = _hard_cap_diagnostic_summary(hard_cap_pairwise)

    _write_table(effects, validation_dir / "next_level_seed_sweep_effects")
    _write_table(policy_summary, validation_dir / "next_level_seed_sweep_policy_summary")
    _write_table(
        pairwise,
        validation_dir / "next_level_seed_sweep_quality_first_vs_small_only",
    )
    _write_table(
        pairwise_summary,
        validation_dir / "next_level_seed_sweep_quality_first_vs_small_only_summary",
    )
    _write_table(
        hard_cap_pairwise,
        validation_dir / "next_level_seed_sweep_hard_cap_diagnostics",
    )
    _write_table(
        hard_cap_summary,
        validation_dir / "next_level_seed_sweep_hard_cap_diagnostic_summary",
    )
    figure_path = None if args.skip_figure else _plot_seed_deltas(pairwise, validation_dir)
    report_path = _write_report(
        effects=effects,
        policy_summary=policy_summary,
        pairwise=pairwise,
        pairwise_summary=pairwise_summary,
        hard_cap_summary=hard_cap_summary,
        output_dir=validation_dir,
        figure_path=figure_path,
    )
    print(
        "Saved seed sweep effects to "
        f"{_rel(validation_dir / 'next_level_seed_sweep_effects.csv')}"
    )
    print(f"Report: {_rel(report_path)}")


if __name__ == "__main__":
    main()
