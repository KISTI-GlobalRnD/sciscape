#!/usr/bin/env python3
"""Evaluate Leiden hysteresis probes as refinement-work accelerators.

This script is intentionally analysis-only. It consumes the qf/i/k_work traces
from the non-monotone group escape smoke run and rewrites the evidence around a
work-acceleration question:

    For the same qf target, does the perturbed branch use less refinement work?

Quality improvement is treated as a guard or secondary signal, not the primary
claim. The main x-axis is cumulative ``k_work`` rather than wall-clock time.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_DIR = (
    REPO_ROOT
    / "research/consensus/results/adaptive_refinement/"
    "leiden_hysteresis_shatter_smoke_20260512"
)


def _float_or_nan(value: Any) -> float:
    if value is None:
        return math.nan
    try:
        if pd.isna(value):
            return math.nan
    except TypeError:
        pass
    if value == "":
        return math.nan
    return float(value)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _reach_target(points: pd.DataFrame, target_ppm: float) -> dict[str, Any]:
    """Return interpolated k_work/t point where a branch first reaches target."""
    branch = points.sort_values(["t_k_work", "t_i", "t_k_phase"]).reset_index(drop=True)
    if branch.empty or not math.isfinite(target_ppm):
        return {
            "k_work": math.nan,
            "i": math.nan,
            "k_phase": math.nan,
            "kind": "missing",
            "segment": "",
        }

    first = branch.iloc[0]
    if float(first["qf_delta_ppm"]) >= target_ppm:
        return {
            "k_work": float(first["t_k_work"]),
            "i": float(first["t_i"]),
            "k_phase": float(first["t_k_phase"]),
            "kind": "first",
            "segment": str(first["t_label"]),
        }

    previous = first
    for _, current in branch.iloc[1:].iterrows():
        current_ppm = float(current["qf_delta_ppm"])
        if current_ppm < target_ppm:
            previous = current
            continue

        previous_ppm = float(previous["qf_delta_ppm"])
        span = current_ppm - previous_ppm
        frac = 1.0 if span == 0.0 else (target_ppm - previous_ppm) / span
        frac = min(1.0, max(0.0, frac))
        k_work = float(previous["t_k_work"]) + frac * (
            float(current["t_k_work"]) - float(previous["t_k_work"])
        )
        i_value = float(previous["t_i"]) + frac * (
            float(current["t_i"]) - float(previous["t_i"])
        )
        k_phase = float(previous["t_k_phase"]) + frac * (
            float(current["t_k_phase"]) - float(previous["t_k_phase"])
        )
        return {
            "k_work": k_work,
            "i": i_value,
            "k_phase": k_phase,
            "kind": "linear_interp",
            "segment": f"{previous['t_label']}->{current['t_label']}",
        }

    last = branch.iloc[-1]
    return {
        "k_work": math.nan,
        "i": math.nan,
        "k_phase": math.nan,
        "kind": "unreached",
        "segment": str(last["t_label"]),
    }


def _target_policy_specs(extra_final_ppm: float, perturb_final_ppm: float) -> list[dict[str, Any]]:
    matched_min = min(extra_final_ppm, perturb_final_ppm)
    return [
        {
            "target_policy": "matched_min",
            "target_ppm": matched_min,
            "target_note": "diagnostic_min_final",
        },
        {
            "target_policy": "baseline_plus_25ppm",
            "target_ppm": 25.0,
            "target_note": "branch_independent_fixed_gain",
        },
        {
            "target_policy": "extra_p5_final",
            "target_ppm": extra_final_ppm,
            "target_note": "ordinary_extra_polish_final",
        },
        {
            "target_policy": "inside_min_10ppm",
            "target_ppm": max(0.0, matched_min - 10.0),
            "target_note": "matched_min_minus_10ppm",
        },
    ]


def _target_reach_status(reach: dict[str, Any], target_ppm: float) -> str:
    if target_ppm <= 0.0 and math.isfinite(target_ppm):
        return "degenerate_zero_target"
    kind = str(reach.get("kind", ""))
    if kind in {"first", "linear_interp"}:
        return "reached"
    if kind == "unreached":
        return "did_not_reach_target"
    return "missing_curve"


def _score_target_policy(
    *,
    extra_points: pd.DataFrame,
    perturb_points: pd.DataFrame,
    target_policy: str,
    target_ppm: float,
    target_note: str = "",
) -> dict[str, Any]:
    extra_reach = _reach_target(extra_points, target_ppm)
    perturb_reach = _reach_target(perturb_points, target_ppm)
    extra_status = _target_reach_status(extra_reach, target_ppm)
    perturb_status = _target_reach_status(perturb_reach, target_ppm)
    extra_k = float(extra_reach["k_work"])
    perturb_k = float(perturb_reach["k_work"])
    if extra_status == perturb_status == "reached" and extra_k > 0.0:
        k_work_saving = extra_k - perturb_k
        saving_pct = k_work_saving / extra_k * 100.0
    elif extra_status == perturb_status == "degenerate_zero_target":
        k_work_saving = math.nan
        saving_pct = math.nan
    else:
        k_work_saving = math.nan
        saving_pct = math.nan
    return {
        "target_policy": target_policy,
        "target_ppm": target_ppm,
        "target_note": target_note,
        "common_target_ppm": target_ppm,
        "extra_k_work_to_target": extra_k,
        "perturb_k_work_to_target": perturb_k,
        "k_work_saving": k_work_saving,
        "k_work_saving_pct": saving_pct,
        "extra_i_to_target": float(extra_reach["i"]),
        "perturb_i_to_target": float(perturb_reach["i"]),
        "extra_k_phase_to_target": float(extra_reach["k_phase"]),
        "perturb_k_phase_to_target": float(perturb_reach["k_phase"]),
        "extra_reach_kind": extra_reach["kind"],
        "perturb_reach_kind": perturb_reach["kind"],
        "extra_tau_status": extra_status,
        "perturb_tau_status": perturb_status,
        "extra_reach_segment": extra_reach["segment"],
        "perturb_reach_segment": perturb_reach["segment"],
    }


def _classify_work_saving(saving_pct: float) -> str:
    if not math.isfinite(saving_pct):
        return "missing"
    if saving_pct >= 20.0:
        return "strong_work_acceleration"
    if saving_pct > 0.0:
        return "weak_work_acceleration"
    return "no_work_acceleration"


def _classify_quality_guard(same_work_adv_ppm: float, long_adv_ppm: float) -> str:
    if math.isfinite(long_adv_ppm) and long_adv_ppm < 0.0:
        return "long_regression_guard_fail"
    if same_work_adv_ppm < -10.0:
        return "same_work_qf_regression"
    if same_work_adv_ppm < 25.0:
        return "qf_neutral_or_tiny"
    if math.isfinite(long_adv_ppm) and long_adv_ppm >= 0.0:
        return "durable_qf_plus"
    return "short_run_qf_plus_needs_long"


def _classify_role(work_class: str, quality_class: str) -> str:
    if work_class == "no_work_acceleration":
        return "reject_no_work_acceleration"
    if quality_class in {"long_regression_guard_fail", "same_work_qf_regression"}:
        return "shortcut_only_quality_guard_fails"
    if work_class == "strong_work_acceleration" and quality_class == "durable_qf_plus":
        return "durable_work_acceleration_candidate"
    if work_class == "strong_work_acceleration" and quality_class == "short_run_qf_plus_needs_long":
        return "short_run_work_acceleration_candidate_needs_long"
    if work_class in {"strong_work_acceleration", "weak_work_acceleration"}:
        return "work_acceleration_quality_neutral"
    return "review"


def _load_membership_delta(input_dir: Path) -> dict[tuple[str, int], dict[str, Any]]:
    """Load currently available final-structure delta diagnostics.

    The membership-delta artifact is currently a focused bc73 review, so attach
    it conservatively only to that case.
    """
    path = input_dir / "membership_delta_review/bc73_membership_delta_summary.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    rows = frame[frame["comparison"] == "extra5_vs_perturb5"]
    if rows.empty:
        return {}
    row = rows.iloc[0]
    return {
        ("bc", 73): {
            "membership_delta_nodes_extra_vs_perturb": int(
                row["nodes_outside_a_dominant_blocks"]
            ),
            "membership_delta_pct_extra_vs_perturb": float(
                row["outside_a_dominant_pct"]
            ),
            "membership_delta_nmi_extra_vs_perturb": float(row["nmi"]),
            "membership_delta_ari_extra_vs_perturb": float(row["ari"]),
        }
    }


def build_scorecard(input_dir: Path) -> pd.DataFrame:
    points = _read_csv(input_dir / "qf_i_k_review/i_k_qf_points.csv")
    matched = _read_csv(input_dir / "qf_i_k_review/k_work_matched_summary.csv")
    final_cmp = _optional_csv(input_dir / "qf_curve_analysis/qf_curve_final_comparison.csv")
    long_cmp = _optional_csv(
        input_dir / "qf_convergence_speed_analysis_targeted/"
        "targeted_long_polish_final_comparison.csv"
    )
    membership_delta = _load_membership_delta(input_dir)

    long_by_key: dict[tuple[str, int], float] = {}
    if not long_cmp.empty:
        for _, row in long_cmp.iterrows():
            case = "bc" if "_bc_cosine" in str(row["case"]) else "cc"
            long_by_key[(case, int(row["seed"]))] = _float_or_nan(
                row.get("perturb20_minus_extra20_delta_ppm_of_baseline")
            )

    group_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    if not final_cmp.empty:
        for _, row in final_cmp.iterrows():
            case = "bc" if "_bc_cosine" in str(row["case"]) else "cc"
            group_by_key[(case, int(row["seed"]))] = {
                "source_cluster": int(row["source_cluster"]),
                "target_cluster": int(row["target_cluster"]),
                "group_kind": row["group_kind"],
                "initial_group_count": int(row["group_count"]),
                "initial_group_weight": float(row["group_weight"]),
            }

    rows: list[dict[str, Any]] = []
    for _, row in matched.sort_values(["case", "seed"]).iterrows():
        case = str(row["case"])
        seed = int(row["seed"])
        case_points = points[(points["case"] == case) & (points["seed"] == seed)]
        extra_points = case_points[case_points["branch"] == "extra"]
        perturb_points = case_points[case_points["branch"] == "perturb"]
        if extra_points.empty or perturb_points.empty:
            continue

        extra_final_ppm = float(row["extra_final_ppm"])
        perturb_final_ppm = float(row["perturb_final_ppm"])
        common_target_ppm = min(extra_final_ppm, perturb_final_ppm)
        extra_reach = _reach_target(extra_points, common_target_ppm)
        perturb_reach = _reach_target(perturb_points, common_target_ppm)
        extra_k = float(extra_reach["k_work"])
        perturb_k = float(perturb_reach["k_work"])
        work_saving = extra_k - perturb_k
        saving_pct = work_saving / extra_k * 100.0 if extra_k > 0.0 else math.nan

        same_work_adv_ppm = float(row["perturb_minus_extra_at_perturb_k_work_ppm"])
        long_adv_ppm = long_by_key.get((case, seed), math.nan)
        work_class = _classify_work_saving(saving_pct)
        quality_class = _classify_quality_guard(same_work_adv_ppm, long_adv_ppm)
        role = _classify_role(work_class, quality_class)

        out = {
            "case": case,
            "seed": seed,
            "case_full": row["case_full"],
            "common_target_ppm": common_target_ppm,
            "extra_k_work_to_target": extra_k,
            "perturb_k_work_to_target": perturb_k,
            "k_work_saving": work_saving,
            "k_work_saving_pct": saving_pct,
            "extra_i_to_target": float(extra_reach["i"]),
            "perturb_i_to_target": float(perturb_reach["i"]),
            "extra_k_phase_to_target": float(extra_reach["k_phase"]),
            "perturb_k_phase_to_target": float(perturb_reach["k_phase"]),
            "extra_reach_kind": extra_reach["kind"],
            "perturb_reach_kind": perturb_reach["kind"],
            "extra_reach_segment": extra_reach["segment"],
            "perturb_reach_segment": perturb_reach["segment"],
            "extra_final_ppm": extra_final_ppm,
            "perturb_final_ppm": perturb_final_ppm,
            "same_k_work_advantage_ppm": same_work_adv_ppm,
            "final_perturb_minus_extra_ppm": perturb_final_ppm - extra_final_ppm,
            "long_p20_advantage_ppm": long_adv_ppm,
            "work_speed_class": work_class,
            "quality_guard_class": quality_class,
            "acceleration_role": role,
        }
        out.update(group_by_key.get((case, seed), {}))
        out.update(membership_delta.get((case, seed), {}))
        rows.append(out)

    return pd.DataFrame(rows)


def _plot_qf_vs_k_work(points: pd.DataFrame, scorecard: pd.DataFrame, out_path: Path) -> None:
    cases = list(scorecard[["case", "seed"]].itertuples(index=False, name=None))
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharey=False)
    axes_flat = axes.ravel()
    for ax, (case, seed) in zip(axes_flat, cases, strict=False):
        subset = points[(points["case"] == case) & (points["seed"] == seed)]
        row = scorecard[(scorecard["case"] == case) & (scorecard["seed"] == seed)].iloc[0]
        for branch, color in (("extra", "#4C78A8"), ("perturb", "#F58518")):
            branch_points = subset[subset["branch"] == branch].sort_values("t_k_work")
            ax.plot(
                branch_points["t_k_work"],
                branch_points["qf_delta_ppm"],
                marker="o",
                label=branch,
                color=color,
                linewidth=1.8,
            )
            for _, point in branch_points.iterrows():
                ax.annotate(
                    str(int(point["t_i"])),
                    (point["t_k_work"], point["qf_delta_ppm"]),
                    fontsize=7,
                    xytext=(2, 2),
                    textcoords="offset points",
                )
        ax.axhline(float(row["common_target_ppm"]), color="#666666", linestyle="--", linewidth=0.8)
        ax.set_title(f"{case}{seed}: {row['acceleration_role']}", fontsize=9)
        ax.set_xlabel("k_work")
        ax.set_ylabel("qf delta ppm")
        ax.grid(alpha=0.25)
    for ax in axes_flat[len(cases) :]:
        ax.axis("off")
    axes_flat[0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_score_scatter(scorecard: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {
        "durable_work_acceleration_candidate": "#2F855A",
        "short_run_work_acceleration_candidate_needs_long": "#D69E2E",
        "work_acceleration_quality_neutral": "#3182CE",
        "shortcut_only_quality_guard_fails": "#C53030",
        "reject_no_work_acceleration": "#4A5568",
    }
    for role, rows in scorecard.groupby("acceleration_role", sort=False):
        ax.scatter(
            rows["k_work_saving_pct"],
            rows["same_k_work_advantage_ppm"],
            label=role,
            s=80,
            color=colors.get(role, "#718096"),
            alpha=0.9,
        )
        for _, row in rows.iterrows():
            ax.annotate(
                f"{row['case']}{int(row['seed'])}",
                (row["k_work_saving_pct"], row["same_k_work_advantage_ppm"]),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=8,
            )
    ax.axvline(0.0, color="#333333", linewidth=0.8)
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.axhline(25.0, color="#999999", linestyle="--", linewidth=0.8)
    ax.set_xlabel("k_work saving to common target (%)")
    ax.set_ylabel("qf advantage at same k_work (ppm)")
    ax.set_title("Work acceleration vs quality guard")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _format_float(value: Any, digits: int = 1) -> str:
    value = _float_or_nan(value)
    if not math.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


def write_outputs(input_dir: Path, out_dir: Path, scorecard: pd.DataFrame) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    points = _read_csv(input_dir / "qf_i_k_review/i_k_qf_points.csv")

    scorecard_path = out_dir / "work_acceleration_scorecard.csv"
    summary_path = out_dir / "work_acceleration_summary.json"
    report_path = out_dir / "work_acceleration_report.md"
    curve_path = out_dir / "qf_ppm_vs_k_work_acceleration_grid.png"
    scatter_path = out_dir / "work_saving_vs_quality_guard.png"

    scorecard.to_csv(scorecard_path, index=False)
    _plot_qf_vs_k_work(points, scorecard, curve_path)
    _plot_score_scatter(scorecard, scatter_path)

    class_counts = scorecard["acceleration_role"].value_counts().sort_index().to_dict()
    selected = scorecard[
        scorecard["acceleration_role"].isin(
            {
                "durable_work_acceleration_candidate",
                "short_run_work_acceleration_candidate_needs_long",
                "work_acceleration_quality_neutral",
            }
        )
    ].copy()
    selected = selected.sort_values(
        ["k_work_saving_pct", "same_k_work_advantage_ppm"], ascending=False
    )

    payload = {
        "schema": "leiden_hysteresis_work_acceleration_review.v1",
        "input_dir": str(input_dir.relative_to(REPO_ROOT)),
        "class_counts": class_counts,
        "n_rows": int(len(scorecard)),
        "paths": {
            "scorecard_csv": str(scorecard_path.relative_to(REPO_ROOT)),
            "report_md": str(report_path.relative_to(REPO_ROOT)),
            "qf_ppm_vs_k_work_grid": str(curve_path.relative_to(REPO_ROOT)),
            "work_saving_vs_quality_guard": str(scatter_path.relative_to(REPO_ROOT)),
        },
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# Leiden Hysteresis Work Acceleration Review",
        "",
        "This report treats the non-monotone group perturbation as a refinement-work accelerator, not as a basin-escape quality method.",
        "",
        "## Metrics",
        "",
        "- Primary: `k_work_saving_pct` to reach the common qf target `min(extra_final_ppm, perturb_final_ppm)`.",
        "- Quality guard: `same_k_work_advantage_ppm` and available `long_p20_advantage_ppm`.",
        "- Structural guard: membership delta is included where available; currently the focused delta is `bc73`.",
        "",
        "## Class Counts",
    ]
    for key, value in class_counts.items():
        lines.append(f"- {key}: {value}")

    lines.extend(
        [
            "",
            "## Scorecard",
            "",
            "| case | target ppm | k_work saving % | same-k_work qf adv ppm | long p20 adv ppm | initial group | membership delta | role |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for _, row in scorecard.sort_values(["case", "seed"]).iterrows():
        initial_group = row.get("initial_group_count", "")
        if initial_group != "" and not pd.isna(initial_group):
            initial_group = int(initial_group)
        membership_delta = _format_float(
            row.get("membership_delta_pct_extra_vs_perturb"), 3
        )
        if membership_delta:
            membership_delta = f"{membership_delta}%"
        lines.append(
            "| {case}{seed} | {target} | {saving} | {same} | {long} | {group} | {delta} | {role} |".format(
                case=row["case"],
                seed=int(row["seed"]),
                target=_format_float(row["common_target_ppm"], 1),
                saving=_format_float(row["k_work_saving_pct"], 1),
                same=_format_float(row["same_k_work_advantage_ppm"], 1),
                long=_format_float(row["long_p20_advantage_ppm"], 1),
                group=initial_group,
                delta=membership_delta,
                role=row["acceleration_role"],
            )
        )

    lines.extend(
        [
            "",
            "## Accelerator Candidates",
            "",
        ]
    )
    if selected.empty:
        lines.append("- None under the current guards.")
    else:
        for _, row in selected.iterrows():
            lines.append(
                "- {case}{seed}: save {saving}% k_work to target, same-k_work qf {same} ppm, role `{role}`.".format(
                    case=row["case"],
                    seed=int(row["seed"]),
                    saving=_format_float(row["k_work_saving_pct"], 1),
                    same=_format_float(row["same_k_work_advantage_ppm"], 1),
                    role=row["acceleration_role"],
                )
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The current evidence supports a work-acceleration framing better than a structural basin-escape framing.",
            "- Rows with long-polish regression are useful shortcut signals but should not be counted as durable quality improvements.",
            "- Tiny same-k_work qf advantages are acceptable for a speed claim only if the common-target work saving is stable across seeds.",
            "- Next validation should add membership-delta summaries for every promoted accelerator row, not only `bc73`.",
            "",
            f"- artifact: `{payload['paths']['scorecard_csv']}`",
            f"- artifact: `{payload['paths']['qf_ppm_vs_k_work_grid']}`",
            f"- artifact: `{payload['paths']['work_saving_vs_quality_guard']}`",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload["paths"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    out_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else input_dir / "work_acceleration_review"
    )
    scorecard = build_scorecard(input_dir)
    paths = write_outputs(input_dir, out_dir, scorecard)
    print(json.dumps(paths, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
