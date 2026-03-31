"""Demo script that runs Leiden hierarchy on the bio-CE-CX toy dataset."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Sequence

import polars as pl

from .config import HierarchyConfig, HierarchyLevelConfig, PostprocessConfig
from .graph import build_graph
from .hierarchy_builder import HierarchyBuilder
from .tuning import ResolutionScanEntry, ResolutionScanResult, scan_resolution_grid


DATA_PATH = Path(__file__).resolve().parents[1] / "Data" / "bio-CE-CX" / "bio-CE-CX.edges"


def load_edge_table(path: Path) -> pl.DataFrame:
    """Parse the whitespace-separated edge list into the expected schema."""

    rows = []
    with path.open("r", encoding="utf8") as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 3:
                continue
            uid1, uid2, weight = parts
            rows.append((uid1, uid2, float(weight)))

    return pl.DataFrame(
        rows,
        schema={"uid1": pl.Utf8, "uid2": pl.Utf8, "rel_sum2": pl.Float64},
    )


def summarise_scan(result: ResolutionScanResult) -> None:
    """Print a compact summary of the resolution scan."""

    grouped: dict[float, list[ResolutionScanEntry]] = defaultdict(list)
    for entry in result.entries:
        grouped[entry.resolution].append(entry)

    print("Resolution grid summary:")
    for gamma in sorted(grouped):
        entries = grouped[gamma]
        avg_clusters = mean(e.cluster_count for e in entries)
        best_quality = max(e.quality for e in entries)
        stability = result.stability.get(gamma, 1.0) if result.stability else None
        stability_txt = f", stability={stability:.3f}" if stability is not None else ""
        print(
            f"  gamma={gamma:.4f}: avg_clusters={avg_clusters:.1f}, "
            f"best_quality={best_quality:.4f}{stability_txt}"
        )


def pick_resolutions(result: ResolutionScanResult, n_levels: int = 2) -> Sequence[float]:
    """Select a set of resolutions spanning the cluster-count range."""

    grouped: dict[float, list[ResolutionScanEntry]] = defaultdict(list)
    for entry in result.entries:
        grouped[entry.resolution].append(entry)

    if not grouped:
        raise ValueError("Resolution scan returned no entries")

    sorted_resolutions = sorted(
        (
            (gamma, mean(e.cluster_count for e in entries))
            for gamma, entries in grouped.items()
        ),
        key=lambda item: item[1],
        reverse=True,
    )

    if n_levels >= len(sorted_resolutions):
        return [item[0] for item in sorted_resolutions]

    # Spread the selection across the range of cluster counts.
    step = max(1, len(sorted_resolutions) // n_levels)
    picks = [sorted_resolutions[i][0] for i in range(0, len(sorted_resolutions), step)]
    return picks[:n_levels]


def main() -> None:
    edges_path = DATA_PATH
    if not edges_path.exists():
        raise FileNotFoundError(f"Expected edge list at {edges_path}")

    edges = load_edge_table(edges_path)
    print(f"Loaded {edges.height} edges from {edges_path}")

    graph = build_graph(edges)
    print(f"Graph has {graph.vcount()} nodes and {graph.ecount()} edges")

    gamma_grid = [0.02, 0.05, 0.1, 0.2, 0.4, 0.8]
    seeds = [0, 1, 2]
    scan_result = scan_resolution_grid(
        graph,
        gamma_grid,
        seeds,
        objective="cpm",
        n_iterations=100,
        postprocess=PostprocessConfig(min_size=3),
        stability_metric="nmi",
        parallel=False,
    )

    summarise_scan(scan_result)

    selected = pick_resolutions(scan_result, n_levels=2)
    level_names = [f"level-{idx+1}" for idx in range(len(selected))]
    print("Selected resolutions for hierarchy:")
    for name, gamma in zip(level_names, selected):
        print(f"  {name}: gamma={gamma:.4f}")

    hierarchy_config = HierarchyConfig(
        levels=[
            HierarchyLevelConfig(
                name=name,
                resolution=gamma,
                seeds=seeds,
                iterations=150,
                postprocess=PostprocessConfig(min_size=3),
            )
            for name, gamma in zip(level_names, selected)
        ]
    )

    builder = HierarchyBuilder(
        graph,
        objective="cpm",
        default_iterations=150,
        default_seed=0,
    )
    hierarchy = builder.build(hierarchy_config)

    membership_columns = {"uid": graph.vs["uid"]}
    for layer in hierarchy.layers:
        membership_columns[f"cluster_{layer.name}"] = hierarchy.memberships_by_level[layer.name]
    membership_df = pl.DataFrame(membership_columns)
    membership_path = Path("bio_ce_cx_membership.parquet")
    membership_df.write_parquet(membership_path)
    print(f"Saved membership table to {membership_path.resolve()}")

    print("Hierarchy summary:")
    for layer in hierarchy.layers:
        print(
            f"  {layer.name}: clusters={layer.cluster_count}, "
            f"quality={layer.quality:.4f}, resolution={layer.resolution:.4f}"
        )


if __name__ == "__main__":
    main()
