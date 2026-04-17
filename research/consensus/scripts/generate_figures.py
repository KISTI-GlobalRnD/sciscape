"""Generate paper figures from consensus research result JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _slug(label: str) -> str:
    return label.replace(" ", "_").replace("/", "_")


def plot_comparison(payload: dict, out_dir: Path) -> Path:
    field = payload["field"]
    results = payload.get("results", [])
    methods = [row["method"] for row in results]
    ami = [row["ami_mean"] for row in results]
    err = [row.get("ami_std", 0.0) for row in results]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(methods))
    ax.bar(x, ami, yerr=err, color="#4c78a8", alpha=0.9, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_ylabel("AMI")
    ax.set_title(f"{field}: single-layer vs consensus")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()

    out_path = out_dir / f"{field}_comparison_ami.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_leave_one_out(payload: dict, out_dir: Path) -> Path:
    field = payload["field"]
    ablations = payload.get("ablations", [])
    labels = [row["removed_layer"] for row in ablations]
    delta_ami = [row["delta_ami"] for row in ablations]
    colors = ["#54a24b" if value >= 0 else "#e45756" for value in delta_ami]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(labels))
    ax.bar(x, delta_ami, color=colors, alpha=0.9)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Δ AMI vs full model")
    ax.set_title(f"{field}: leave-one-out impact")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()

    out_path = out_dir / f"{field}_leave_one_out_delta_ami.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_consensus_tiers(payload: dict, out_dir: Path) -> Path:
    field = payload["field"]
    stats = payload.get("cluster_stats", {})
    levels = sorted(int(level) for level in stats)
    intra = [stats[str(level)]["intra_pct"] for level in levels]
    cross = [stats[str(level)]["cross_pct"] for level in levels]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(levels))
    ax.bar(x, intra, color="#54a24b", label="Intra-cluster")
    ax.bar(x, cross, bottom=intra, color="#e45756", label="Cross-cluster")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{level}L" for level in levels])
    ax.set_ylabel("% of edges")
    ax.set_title(f"{field}: intra/cross by consensus level")
    ax.set_ylim(0, 100)
    ax.legend()
    fig.tight_layout()

    out_path = out_dir / f"{field}_consensus_tiers.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_cross_field(payload: dict, out_dir: Path) -> Path | None:
    summary = payload.get("summary", [])
    if not summary:
        return None
    fields = [row["field"] for row in summary]
    gains = [row.get("ami_gain_vs_best_single", row.get("ami_gain", 0.0)) for row in summary]
    colors = ["#54a24b" if value >= 0 else "#e45756" for value in gains]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(fields))
    ax.bar(x, gains, color=colors)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(fields)
    ax.set_ylabel("AMI gain vs best single-layer")
    ax.set_title("Cross-field consensus gain")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()

    out_path = out_dir / "cross_field_gain.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_k_sweep(payload: dict, out_dir: Path) -> Path | None:
    rows = payload.get("summary", [])
    if not rows:
        return None

    field = payload["field"]
    ks = [row["effective_k"] for row in rows]
    best_single = [row["best_single"]["ami_mean"] if row.get("best_single") else np.nan for row in rows]
    citation = [row["citation_consensus"]["ami_mean"] if row.get("citation_consensus") else np.nan for row in rows]
    all_cons = [row["all_consensus"]["ami_mean"] if row.get("all_consensus") else np.nan for row in rows]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(ks, best_single, marker="o", label="Best single-layer", color="#4c78a8")
    if not np.all(np.isnan(citation)):
        ax.plot(ks, citation, marker="o", label="Citation consensus", color="#54a24b")
    if not np.all(np.isnan(all_cons)):
        ax.plot(ks, all_cons, marker="o", label="All-layer consensus", color="#f58518")
    ax.set_xlabel("effective_k")
    ax.set_ylabel("AMI")
    ax.set_title(f"{field}: effective-k sweep")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()

    out_path = out_dir / f"{field}_k_sweep_ami.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_cross_field_k_sweep(payload: dict, out_dir: Path) -> Path | None:
    runs = payload.get("runs", {})
    if not runs:
        return None

    fields = sorted(runs)
    k_values = sorted(
        {
            int(row["effective_k"])
            for run in runs.values()
            for row in run.get("summary", [])
        }
    )
    if not fields or not k_values:
        return None

    matrix = np.full((len(fields), len(k_values)), np.nan)
    for i, field in enumerate(fields):
        by_k = {
            int(row["effective_k"]): row.get("citation_gain_vs_best_single")
            for row in runs[field].get("summary", [])
        }
        for j, k in enumerate(k_values):
            value = by_k.get(k)
            if value is not None:
                matrix[i, j] = value

    fig, ax = plt.subplots(figsize=(max(8, len(k_values) * 0.35), max(4, len(fields) * 0.5)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=-0.1, vmax=0.1)
    ax.set_xticks(np.arange(len(k_values)))
    ax.set_xticklabels(k_values)
    ax.set_yticks(np.arange(len(fields)))
    ax.set_yticklabels(fields)
    ax.set_xlabel("effective_k")
    ax.set_title("Citation consensus AMI gain vs best single-layer")
    fig.colorbar(im, ax=ax, shrink=0.8, label="AMI gain")
    fig.tight_layout()

    out_path = out_dir / "cross_field_k_sweep_heatmap.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_boundary_review(payload: dict, out_dir: Path) -> Path | None:
    summary = payload.get("summary_by_level", {})
    if not summary:
        return None
    levels = sorted(int(level) for level in summary)
    assigned_share = []
    cohesion_mean = []
    for level in levels:
        row = summary[str(level)]
        n_cases = max(1, row["n_cases"])
        assigned_share.append(100.0 * row["belongs_assigned"] / n_cases)
        cohesion_mean.append(row["cohesion_mean"] or 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].bar([f"{level}L" for level in levels], assigned_share, color="#4c78a8")
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("% assigned-cluster wins")
    axes[0].set_title("Belonging preference")

    axes[1].bar([f"{level}L" for level in levels], cohesion_mean, color="#f58518")
    axes[1].set_ylim(0, 5)
    axes[1].set_ylabel("Mean cohesion score")
    axes[1].set_title("Cohesion by consensus level")

    field = payload["field"]
    fig.suptitle(f"{field}: boundary review")
    fig.tight_layout()
    out_path = out_dir / f"{field}_boundary_review.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate figures for consensus research")
    parser.add_argument("results_dir", type=Path, help="Directory containing result JSON files")
    parser.add_argument("-o", "--output", type=Path, default=Path("figures"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    for path in sorted(args.results_dir.glob("*_comparison.json")):
        generated.append(plot_comparison(_read_json(path), args.output))
    for path in sorted(args.results_dir.glob("*_k_sweep.json")):
        out_path = plot_k_sweep(_read_json(path), args.output)
        if out_path:
            generated.append(out_path)
    for path in sorted(args.results_dir.glob("*_leave_one_out.json")):
        generated.append(plot_leave_one_out(_read_json(path), args.output))
    for path in sorted(args.results_dir.glob("*_consensus_tiers.json")):
        generated.append(plot_consensus_tiers(_read_json(path), args.output))
    for path in sorted(args.results_dir.glob("*_boundary_review.json")):
        out_path = plot_boundary_review(_read_json(path), args.output)
        if out_path:
            generated.append(out_path)

    cross_field_path = args.results_dir / "cross_field_summary.json"
    if cross_field_path.exists():
        out_path = plot_cross_field(_read_json(cross_field_path), args.output)
        if out_path:
            generated.append(out_path)
    cross_field_k_path = args.results_dir / "cross_field_k_sweep_summary.json"
    if cross_field_k_path.exists():
        out_path = plot_cross_field_k_sweep(_read_json(cross_field_k_path), args.output)
        if out_path:
            generated.append(out_path)

    if generated:
        print("Generated:")
        for path in generated:
            print(f"  {path}")
    else:
        print(f"No compatible result files found in {args.results_dir}")


if __name__ == "__main__":
    main()
