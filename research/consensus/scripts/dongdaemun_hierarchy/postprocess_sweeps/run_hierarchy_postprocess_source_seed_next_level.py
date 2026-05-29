"""Run next-level propagation from source-seed postprocess memberships.

The source-seed sweep measures level-0 policy behavior under different Leiden
seeds.  This runner takes those produced memberships, contracts the original
graph, runs next-level Leiden, and compares whether source-level changes reduce
strict-target imbalance propagation.
"""

from __future__ import annotations

import argparse
import json
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
    _flatten_summary,
    _load_graph_arrays,
    _rel,
    _run_one,
)
from run_hierarchy_postprocess_seed_sweep import _parse_int_list  # noqa: E402

DEFAULT_POLICIES = ("small_only", "two_stage_quality_first", "two_stage_hard_cap")

def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def _source_seed_values(value: str) -> set[int] | None:
    value = value.strip()
    if not value or value.lower() == "all":
        return None
    return set(_parse_int_list(value))

def _candidate_rows(
    source_df: pd.DataFrame,
    *,
    policies: set[str],
    source_seeds: set[int] | None,
    include_diagnostics: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in source_df.to_dict("records"):
        policy = str(row.get("policy"))
        role = str(row.get("membership_role"))
        source_seed = int(row.get("seed"))
        if source_seeds is not None and source_seed not in source_seeds:
            continue
        if role == "diagnostic":
            diagnostic_for = str(row.get("diagnostic_for_policy"))
            if not include_diagnostics or diagnostic_for not in policies:
                continue
            rerun_policy = f"{policy}_diagnostic"
            run_suffix = "diagnostic"
        else:
            if policy not in policies or role != "effective":
                continue
            rerun_policy = policy
            run_suffix = "effective"

        membership_path = row.get("membership_path")
        if not isinstance(membership_path, str) or not membership_path:
            continue

        run = f"source_seed_{source_seed}_{rerun_policy}_{run_suffix}"
        rows.append(
            {
                **row,
                "source_seed": source_seed,
                "run": run,
                "rerun_policy": rerun_policy,
                "rerun_run": run,
                "rerun_membership_path": membership_path,
            }
        )
    return rows

def _flatten_source_seed_summary(summary: dict[str, Any]) -> dict[str, Any]:
    row = _flatten_summary(summary)
    row["next_seed"] = row.pop("seed")
    return row

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
                "source_accepted_rate": float(
                    group["source_accepted_for_contraction"].mean()
                ),
                "source_fallback_rate": float(group["source_fallback_used"].mean()),
                "mean_source_delta_q": float(group["source_delta_q"].mean()),
                "mean_next_post_max_ratio": float(
                    group["post_max_doc_weight_ratio"].mean()
                ),
                "mean_next_post_oversize_count": float(
                    group["post_n_above_max_doc_weight"].mean()
                ),
                "mean_next_post_gini": float(group["post_gini_doc_weight"].mean()),
                "mean_parent_max_child_share": float(
                    group["post_parent_max_child_share_weighted_mean"].mean()
                ),
                "mean_contracted_children": float(group["contracted_n_nodes"].mean()),
            }
        )
    return pd.DataFrame(rows)

def _quality_first_pairwise(df: pd.DataFrame) -> pd.DataFrame:
    effective = df[df["membership_role"] == "effective"].copy()
    qf = effective[effective["source_policy"] == "two_stage_quality_first"]
    small = effective[effective["source_policy"] == "small_only"]
    if qf.empty or small.empty:
        return pd.DataFrame()

    keys = ["sample", "source_seed", "next_seed", "next_target_pct"]
    columns = [
        *keys,
        "source_delta_q",
        "post_max_doc_weight_ratio",
        "post_n_above_max_doc_weight",
        "post_gini_doc_weight",
        "post_parent_max_child_share_weighted_mean",
        "contracted_n_nodes",
    ]
    merged = qf[columns].merge(
        small[columns],
        on=keys,
        suffixes=("_quality_first", "_small_only"),
    )
    merged["delta_max_ratio"] = (
        merged["post_max_doc_weight_ratio_quality_first"]
        - merged["post_max_doc_weight_ratio_small_only"]
    )
    merged["delta_oversize_count"] = (
        merged["post_n_above_max_doc_weight_quality_first"]
        - merged["post_n_above_max_doc_weight_small_only"]
    )
    merged["delta_gini"] = (
        merged["post_gini_doc_weight_quality_first"]
        - merged["post_gini_doc_weight_small_only"]
    )
    merged["delta_parent_max_child_share"] = (
        merged["post_parent_max_child_share_weighted_mean_quality_first"]
        - merged["post_parent_max_child_share_weighted_mean_small_only"]
    )
    merged["delta_contracted_children"] = (
        merged["contracted_n_nodes_quality_first"] - merged["contracted_n_nodes_small_only"]
    )
    return merged

def _pairwise_summary(pairwise: pd.DataFrame) -> pd.DataFrame:
    if pairwise.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "n_pairs": int(len(pairwise)),
                "mean_source_delta_q_quality_first": float(
                    pairwise["source_delta_q_quality_first"].mean()
                ),
                "mean_delta_max_ratio": float(pairwise["delta_max_ratio"].mean()),
                "mean_delta_oversize_count": float(
                    pairwise["delta_oversize_count"].mean()
                ),
                "mean_delta_gini": float(pairwise["delta_gini"].mean()),
                "mean_delta_parent_max_child_share": float(
                    pairwise["delta_parent_max_child_share"].mean()
                ),
                "pairs_with_lower_max_ratio": int(
                    (pairwise["delta_max_ratio"] < 0).sum()
                ),
                "pairs_with_lower_oversize_count": int(
                    (pairwise["delta_oversize_count"] < 0).sum()
                ),
                "pairs_with_lower_gini": int((pairwise["delta_gini"] < 0).sum()),
                "pairs_with_lower_parent_max_child_share": int(
                    (pairwise["delta_parent_max_child_share"] < 0).sum()
                ),
            }
        ]
    )

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

    keys = ["sample", "source_seed", "next_seed", "next_target_pct"]
    columns = [
        *keys,
        "source_delta_q",
        "post_max_doc_weight_ratio",
        "post_n_above_max_doc_weight",
        "post_gini_doc_weight",
        "post_parent_max_child_share_weighted_mean",
    ]
    merged = diagnostics[columns].merge(
        effective[columns],
        on=keys,
        suffixes=("_diagnostic", "_effective"),
    )
    merged["delta_max_ratio"] = (
        merged["post_max_doc_weight_ratio_diagnostic"]
        - merged["post_max_doc_weight_ratio_effective"]
    )
    merged["delta_oversize_count"] = (
        merged["post_n_above_max_doc_weight_diagnostic"]
        - merged["post_n_above_max_doc_weight_effective"]
    )
    merged["delta_gini"] = (
        merged["post_gini_doc_weight_diagnostic"]
        - merged["post_gini_doc_weight_effective"]
    )
    merged["delta_parent_max_child_share"] = (
        merged["post_parent_max_child_share_weighted_mean_diagnostic"]
        - merged["post_parent_max_child_share_weighted_mean_effective"]
    )
    return merged

def _hard_cap_diagnostic_summary(pairwise: pd.DataFrame) -> pd.DataFrame:
    if pairwise.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "n_pairs": int(len(pairwise)),
                "mean_source_delta_q_diagnostic": float(
                    pairwise["source_delta_q_diagnostic"].mean()
                ),
                "mean_delta_max_ratio": float(pairwise["delta_max_ratio"].mean()),
                "mean_delta_oversize_count": float(
                    pairwise["delta_oversize_count"].mean()
                ),
                "mean_delta_gini": float(pairwise["delta_gini"].mean()),
                "mean_delta_parent_max_child_share": float(
                    pairwise["delta_parent_max_child_share"].mean()
                ),
                "pairs_with_lower_max_ratio": int(
                    (pairwise["delta_max_ratio"] < 0).sum()
                ),
                "pairs_with_lower_oversize_count": int(
                    (pairwise["delta_oversize_count"] < 0).sum()
                ),
                "pairs_with_lower_gini": int((pairwise["delta_gini"] < 0).sum()),
                "pairs_with_lower_parent_max_child_share": int(
                    (pairwise["delta_parent_max_child_share"] < 0).sum()
                ),
            }
        ]
    )

def _plot_pairwise(pairwise: pd.DataFrame, output_dir: Path) -> Path | None:
    if pairwise.empty:
        return None
    grouped = (
        pairwise.groupby("source_seed", as_index=False)[
            [
                "delta_max_ratio",
                "delta_oversize_count",
                "delta_gini",
                "delta_parent_max_child_share",
            ]
        ]
        .mean()
        .sort_values("source_seed")
    )
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    metrics = [
        ("delta_max_ratio", "max/target ratio delta"),
        ("delta_oversize_count", "oversize-count delta"),
        ("delta_gini", "Gini delta"),
        ("delta_parent_max_child_share", "parent child-share delta"),
    ]
    for ax, (column, title) in zip(axes, metrics, strict=True):
        ax.bar(grouped["source_seed"].astype(str), grouped[column], color="#54a24b")
        ax.axhline(0.0, color="#222222", linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("source Leiden seed")
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("quality_first - small_only")
    fig.suptitle("Source-seed next-level propagation under strict target")
    fig.tight_layout()
    out_path = output_dir / "figure8_source_seed_next_level_propagation.png"
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
    lines = [
        "# Source-Seed Next-Level Propagation",
        "",
        "This runner contracts source-seed memberships and reruns next-level Leiden. The default target is strict 1x of the source-level max-doc-weight percentage.",
        "",
        f"- Rows: {len(effects)}",
        f"- Effective rows: {int((effects['membership_role'] == 'effective').sum())}",
        f"- Diagnostic rows: {int((effects['membership_role'] == 'diagnostic').sum())}",
        f"- Source seeds: {', '.join(str(seed) for seed in sorted(effects['source_seed'].dropna().unique()))}",
        f"- Next-level seeds: {', '.join(str(seed) for seed in sorted(effects['next_seed'].unique()))}",
        "",
    ]
    if not pairwise_summary.empty:
        lines.extend(
            [
                "## Quality-First vs Small-Only",
                "",
                _markdown_table(pairwise_summary),
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
            "- `source_seed_next_level_effects.csv` / `.parquet`",
            "- `source_seed_next_level_policy_summary.csv` / `.parquet`",
            "- `source_seed_next_level_quality_first_vs_small_only.csv` / `.parquet`",
            "- `source_seed_next_level_quality_first_vs_small_only_summary.csv` / `.parquet`",
            "- `source_seed_next_level_hard_cap_diagnostics.csv` / `.parquet`",
            "- `source_seed_next_level_hard_cap_diagnostic_summary.csv` / `.parquet`",
            "",
        ]
    )
    if figure_path is not None:
        lines.extend(["## Figure", "", f"- `{figure_path.name}`", ""])
    if not policy_summary.empty:
        lines.extend(["## Policy Summary", "", _markdown_table(policy_summary)])
    report_path = output_dir / "source_seed_next_level_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--validation-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "source_seed_sweep_effects.csv",
    )
    parser.add_argument("--policies", default=",".join(DEFAULT_POLICIES))
    parser.add_argument("--source-seeds", default="all")
    parser.add_argument("--next-seeds", default="42")
    parser.add_argument("--include-diagnostics", action="store_true", default=True)
    parser.add_argument("--no-diagnostics", dest="include_diagnostics", action="store_false")
    parser.add_argument("--next-target-pct", type=float, default=None)
    parser.add_argument("--next-target-min-pct", type=float, default=0.0)
    parser.add_argument("--next-target-multiplier", type=float, default=1.0)
    parser.add_argument("--next-min-doc-weight", type=float, default=100.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-figure", action="store_true")
    args = parser.parse_args()

    results_dir = _repo_path(args.results_dir)
    validation_dir = _repo_path(args.validation_dir)
    input_path = _repo_path(args.input)
    assert results_dir is not None and validation_dir is not None and input_path is not None
    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing {input_path}. Run run_hierarchy_postprocess_source_seed_sweep.py first."
        )

    source_df = pd.read_csv(input_path)
    policies = {item.strip() for item in str(args.policies).split(",") if item.strip()}
    source_seeds = _source_seed_values(str(args.source_seeds))
    next_seeds = _parse_int_list(str(args.next_seeds))
    rows = _candidate_rows(
        source_df,
        policies=policies,
        source_seeds=source_seeds,
        include_diagnostics=bool(args.include_diagnostics),
    )

    cross_path = results_dir / "postprocess_policy_matrix_cross_sample" / "cross_sample_summary.json"
    cross_summary = _read_json(cross_path) if cross_path.exists() else None
    configs = _sample_configs(results_dir, cross_summary)
    graph_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    summaries: list[dict[str, Any]] = []
    total_runs = len(rows) * len(next_seeds)

    run_idx = 0
    for source_row in rows:
        sample = str(source_row["sample"])
        cfg = configs[sample]
        graph_dir = cfg.node_weights_path.parent if cfg.node_weights_path else cfg.sample_dir / "graph"
        if sample not in graph_cache:
            print(f"loading graph {sample}", flush=True)
            graph_cache[sample] = _load_graph_arrays(graph_dir)
        if args.next_target_pct is None:
            total_docs = float(cfg.n_nodes or graph_cache[sample][3].shape[0])
            current_target = float(source_row.get("target_max_doc_weight") or 0.0)
            current_target_pct = 100.0 * current_target / total_docs if total_docs else 0.0
            next_target_pct = max(
                float(args.next_target_min_pct),
                current_target_pct * float(args.next_target_multiplier),
            )
        else:
            next_target_pct = float(args.next_target_pct)
        for next_seed in next_seeds:
            run_idx += 1
            row = dict(source_row)
            row["rerun_run"] = f"{row['rerun_run']}_next_seed_{next_seed}"
            print(
                f"[{run_idx}/{total_runs}] source-seed next-level "
                f"{sample} source_seed={row['source_seed']} "
                f"policy={row['rerun_policy']} next_seed={next_seed} "
                f"target_pct={next_target_pct:.4g}",
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
                    seed=int(next_seed),
                    force=bool(args.force),
                    runs_dir_name="source_seed_next_level_runs",
                )
            )

    effects = pd.DataFrame([_flatten_source_seed_summary(summary) for summary in summaries])
    policy_summary = _policy_summary(effects)
    pairwise = _quality_first_pairwise(effects)
    pairwise_summary = _pairwise_summary(pairwise)
    hard_cap_pairwise = _hard_cap_diagnostic_pairwise(effects)
    hard_cap_summary = _hard_cap_diagnostic_summary(hard_cap_pairwise)

    _write_table(effects, validation_dir / "source_seed_next_level_effects")
    _write_table(policy_summary, validation_dir / "source_seed_next_level_policy_summary")
    _write_table(
        pairwise,
        validation_dir / "source_seed_next_level_quality_first_vs_small_only",
    )
    _write_table(
        pairwise_summary,
        validation_dir / "source_seed_next_level_quality_first_vs_small_only_summary",
    )
    _write_table(
        hard_cap_pairwise,
        validation_dir / "source_seed_next_level_hard_cap_diagnostics",
    )
    _write_table(
        hard_cap_summary,
        validation_dir / "source_seed_next_level_hard_cap_diagnostic_summary",
    )
    figure_path = None if args.skip_figure else _plot_pairwise(pairwise, validation_dir)
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
        "Saved source-seed next-level effects to "
        f"{_rel(validation_dir / 'source_seed_next_level_effects.csv')}"
    )
    print(f"Report: {_rel(report_path)}")

if __name__ == "__main__":
    main()
