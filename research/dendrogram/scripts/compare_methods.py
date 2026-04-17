"""Aggregate dendrogram method outputs into a comparison table and figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare dendrogram research methods")
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--field", type=str, required=True)
    parser.add_argument("-o", "--output", type=Path, default=None, help="Directory for comparison outputs")
    args = parser.parse_args()

    out_dir = args.output or args.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_path = args.results_dir / f"{args.field}_leiden_merge.json"
    hybrid_path = args.results_dir / f"{args.field}_hybrid.json"
    cut_path = args.results_dir / f"{args.field}_hybrid_cut.json"
    ablation_path = args.results_dir / f"{args.field}_cut_ablation.json"

    if not baseline_path.exists():
        raise FileNotFoundError(baseline_path)
    if not hybrid_path.exists():
        raise FileNotFoundError(hybrid_path)

    baseline = _read_json(baseline_path)
    hybrid = _read_json(hybrid_path)
    cut = _read_json(cut_path) if cut_path.exists() else None
    ablation = _read_json(ablation_path) if ablation_path.exists() else None

    deepest = hybrid.get("deepest_level") or {}
    levels = hybrid.get("levels", [])

    comparison = {
        "field": args.field,
        "baseline": {
            "method": baseline["method"],
            "n_clusters": baseline["n_clusters"],
            "max_pct": baseline["max_pct"],
            "ami_mean": baseline["ami_mean"],
            "singleton_pct": baseline["singleton_pct"],
        },
        "hybrid": {
            "method": hybrid["method"],
            "n_levels": hybrid["n_levels"],
            "deepest_level": deepest.get("name"),
            "n_clusters": deepest.get("n_clusters"),
            "max_pct": deepest.get("max_pct"),
            "ami_mean": (hybrid.get("stability") or {}).get("ami_mean"),
            "singleton_pct": deepest.get("singleton_pct"),
        },
        "hybrid_cut": (
            {
                "method": cut["method"],
                "n_clusters": cut["cut"]["n_clusters"],
                "max_pct": cut["cut"]["max_pct"],
                "singleton_pct": cut["cut"]["singleton_pct"],
                "total_stability": cut["cut"]["total_stability"],
                "feasible": cut["cut"]["feasible"],
            }
            if cut
            else None
        ),
        "threshold_cut": (
            ablation.get("best_threshold_cut")
            if ablation and ablation.get("best_threshold_cut")
            else None
        ),
        "delta": {
            "hierarchy_cluster_gain": (deepest.get("n_clusters") or 0) - baseline["n_clusters"],
            "hierarchy_max_pct_delta": (deepest.get("max_pct") or 0.0) - baseline["max_pct"],
            "hierarchy_ami_delta": ((hybrid.get("stability") or {}).get("ami_mean") or 0.0) - baseline["ami_mean"],
            "cut_cluster_gain": ((cut["cut"]["n_clusters"] if cut else 0) - baseline["n_clusters"]) if cut else None,
            "cut_max_pct_delta": ((cut["cut"]["max_pct"] if cut else 0.0) - baseline["max_pct"]) if cut else None,
            "threshold_cluster_gain": (
                (ablation["best_threshold_cut"]["n_clusters"] - baseline["n_clusters"])
                if ablation and ablation.get("best_threshold_cut")
                else None
            ),
            "threshold_max_pct_delta": (
                (ablation["best_threshold_cut"]["max_pct"] - baseline["max_pct"])
                if ablation and ablation.get("best_threshold_cut")
                else None
            ),
        },
    }

    md_lines = [
        f"# Dendrogram Method Comparison: {args.field}",
        "",
        "| Method | Cluster count | Max cluster % | AMI | Singleton % | Notes |",
        "|--------|---------------:|--------------:|----:|------------:|-------|",
        (
            f"| Leiden+merge | {baseline['n_clusters']} | {baseline['max_pct']:.1f} | "
            f"{baseline['ami_mean']:.3f} | {baseline['singleton_pct']:.1f} | standard postprocess |"
        ),
        (
            f"| Hybrid hierarchy ({deepest.get('name', '?')}) | {deepest.get('n_clusters', 0)} | "
            f"{deepest.get('max_pct', 0.0):.1f} | {((hybrid.get('stability') or {}).get('ami_mean') or 0.0):.3f} | "
            f"{deepest.get('singleton_pct', 0.0):.1f} | {hybrid['n_levels']} levels |"
        ),
    ]
    if cut:
        md_lines.append(
            f"| Hybrid + optimal cut | {cut['cut']['n_clusters']} | {cut['cut']['max_pct']:.1f} | "
            f"n/a | {cut['cut']['singleton_pct']:.1f} | feasible={cut['cut']['feasible']} stability={cut['cut']['total_stability']:.3f} |"
        )
    if ablation and ablation.get("best_threshold_cut"):
        best_threshold = ablation["best_threshold_cut"]
        md_lines.append(
            f"| Threshold cut (best) | {best_threshold['n_clusters']} | {best_threshold['max_pct']:.1f} | "
            f"n/a | {best_threshold['singleton_pct']:.1f} | threshold={best_threshold['threshold']:.6f} |"
        )
    md_lines.extend([
        "",
        "## Delta",
        "",
        f"- Hierarchy cluster gain: {comparison['delta']['hierarchy_cluster_gain']:+d}",
        f"- Hierarchy max cluster delta: {comparison['delta']['hierarchy_max_pct_delta']:+.1f}%",
        f"- Hierarchy AMI delta: {comparison['delta']['hierarchy_ami_delta']:+.3f}",
    ])
    if cut:
        md_lines.extend([
            f"- Optimal cut cluster gain: {comparison['delta']['cut_cluster_gain']:+d}",
            f"- Optimal cut max cluster delta: {comparison['delta']['cut_max_pct_delta']:+.1f}%",
        ])
    if ablation and ablation.get("best_threshold_cut"):
        md_lines.extend([
            f"- Threshold cut cluster gain: {comparison['delta']['threshold_cluster_gain']:+d}",
            f"- Threshold cut max cluster delta: {comparison['delta']['threshold_max_pct_delta']:+.1f}%",
        ])
    md_lines.extend([
        "",
        "## Hybrid Depth",
        "",
        "| Level | Clusters | Max cluster % | Singleton % |",
        "|------|---------:|--------------:|------------:|",
    ])
    for level in levels:
        md_lines.append(
            f"| {level['name']} | {level['n_clusters']} | {level['max_pct']:.1f} | {level['singleton_pct']:.1f} |"
        )
    if cut:
        md_lines.extend([
            "",
            "## Optimal Cut",
            "",
            f"- Contracted cut clusters: {cut['cut']['contracted_cut_clusters']}",
            f"- Total stability: {cut['cut']['total_stability']:.3f}",
            f"- Cut min size: {cut['cut_min_size']}",
        ])
    if ablation:
        md_lines.extend([
            "",
            "## Threshold Sweep",
            "",
            f"- Evaluated thresholds: {len(ablation['threshold_candidates'])}",
            f"- Feasible thresholds: {sum(1 for row in ablation['threshold_candidates'] if row['feasible'])}",
        ])

    md_path = out_dir / f"{args.field}_comparison.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    json_path = out_dir / f"{args.field}_comparison.json"
    json_path.write_text(json.dumps(comparison, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    bar_labels = ["Leiden+merge", f"Hybrid\n{deepest.get('name', '?')}"]
    bar_values = [baseline["n_clusters"], deepest.get("n_clusters", 0)]
    bar_colors = ["#4c78a8", "#f58518"]
    if cut:
        bar_labels.append("Hybrid\noptimal cut")
        bar_values.append(cut["cut"]["n_clusters"])
        bar_colors.append("#54a24b")
    if ablation and ablation.get("best_threshold_cut"):
        bar_labels.append("Threshold\nbest")
        bar_values.append(ablation["best_threshold_cut"]["n_clusters"])
        bar_colors.append("#e45756")
    axes[0].bar(bar_labels, bar_values, color=bar_colors)
    axes[0].set_ylabel("Cluster count")
    axes[0].set_title("Valid clusters")

    if levels:
        axes[1].plot(
            [level["name"] for level in levels],
            [level["n_clusters"] for level in levels],
            marker="o",
            color="#54a24b",
        )
    axes[1].set_ylabel("Cluster count")
    axes[1].set_title("Hybrid depth trajectory")
    axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle(f"{args.field}: dendrogram comparison")
    fig.tight_layout()
    fig_path = out_dir / f"{args.field}_comparison.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    print(md_path)
    print(json_path)
    print(fig_path)


if __name__ == "__main__":
    main()
