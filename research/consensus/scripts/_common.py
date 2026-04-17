"""Shared helpers for consensus research scripts."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import polars as pl

# Add project root to path for direct script execution.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sciscape.clustering.auto_gamma import find_gamma
from sciscape.clustering.integer_remap import integer_remap_memory
from sciscape.evaluation.stability import compute_quality_report, evaluate_stability
from sciscape.linkage.combine import combine_edge_layers
from sciscape.linkage.filters import filter_top_k


LAYER_FILE_CANDIDATES = {
    "bc_cosine": ("bc_cosine",),
    "cc_cosine": ("cc_cosine",),
    "dc_fractional": ("dc_fractional",),
    "emb_knn": ("emb_knn", "emb_full_knn30"),
}


def allocate_effective_k(layer_names: list[str], effective_k: int) -> dict[str, int]:
    """Distribute a global effective-k budget across layers.

    The returned per-layer ``top_k`` values sum to ``effective_k`` whenever
    ``effective_k >= len(layer_names)``. For very small budgets, each layer
    still receives at least 1 neighbor so the combination remains defined.
    """
    if effective_k <= 0:
        raise ValueError(f"effective_k must be positive, got {effective_k}")
    if not layer_names:
        return {}

    ordered = sorted(layer_names)
    n_layers = len(ordered)
    base = effective_k // n_layers
    remainder = effective_k % n_layers
    allocation: dict[str, int] = {}
    for idx, name in enumerate(ordered):
        allocation[name] = max(1, base + (1 if idx < remainder else 0))
    return allocation


def _filter_layers_with_top_k(
    layers: dict[str, pl.DataFrame],
    top_k: int | dict[str, int] | str,
) -> dict[str, pl.DataFrame]:
    """Apply a top-k specification to each layer and return filtered tables."""
    if isinstance(top_k, dict):
        filtered: dict[str, pl.DataFrame] = {}
        for name, df in layers.items():
            if df.height == 0:
                continue
            current_k = int(top_k.get(name, 0))
            if current_k > 0:
                filtered[name] = filter_top_k(df, current_k, mode="symmetric")
            else:
                filtered[name] = df
        return filtered

    if isinstance(top_k, int) and top_k > 0:
        return {
            name: filter_top_k(df, top_k, mode="symmetric")
            for name, df in layers.items()
            if df.height > 0
        }

    return {name: df for name, df in layers.items() if df.height > 0}


def load_layer_paths(edge_dir: Path) -> dict[str, Path]:
    """Discover standard layer parquet files in *edge_dir*."""
    layer_paths: dict[str, Path] = {}
    for canonical_name, candidates in LAYER_FILE_CANDIDATES.items():
        for candidate in candidates:
            path = edge_dir / f"{candidate}.parquet"
            if path.exists():
                layer_paths[canonical_name] = path
                break
    return layer_paths


def load_layer_tables(edge_dir: Path) -> dict[str, pl.DataFrame]:
    """Load all standard layer parquet files from *edge_dir*."""
    return {name: pl.read_parquet(path) for name, path in load_layer_paths(edge_dir).items()}


def load_abstracts_table(abstract_path: Path) -> pl.DataFrame:
    """Load a standard abstract/metadata parquet file."""
    df = pl.read_parquet(abstract_path)
    if "uid" not in df.columns and "work_id" in df.columns:
        df = df.rename({"work_id": "uid"})
    wanted = [col for col in ("uid", "title", "abstract", "pubyear") if col in df.columns]
    if "uid" not in wanted:
        raise ValueError(f"Abstract table missing uid/work_id column: {abstract_path}")
    return df.select(wanted)


def abstracts_lookup(abstracts: pl.DataFrame) -> dict[str, dict[str, Any]]:
    """Return ``uid -> metadata`` lookup for abstract tables."""
    return {row["uid"]: row for row in abstracts.iter_rows(named=True)}


def membership_frame_from_edges(
    edges: pl.DataFrame,
    membership: Any,
    *,
    cluster_col: str = "cluster",
) -> pl.DataFrame:
    """Attach a membership vector to the authoritative UID order for *edges*."""
    _src, _dst, _w, _n_nodes, uids = integer_remap_memory(edges)
    return pl.DataFrame(
        {
            "uid": uids,
            cluster_col: list(membership),
        }
    )


def membership_map_from_edges(edges: pl.DataFrame, membership: Any) -> dict[str, int]:
    """Return UID -> cluster mapping aligned to *edges* remap order."""
    df = membership_frame_from_edges(edges, membership)
    return dict(zip(df["uid"].to_list(), df["cluster"].to_list()))


def layer_combo_label(layers: dict[str, Any]) -> str:
    """Stable label for a layer combination."""
    return "+".join(sorted(layers.keys()))


def run_combination(
    layers: dict[str, pl.DataFrame],
    *,
    strategy: str = "consensus",
    target_pct: float = 3.0,
    top_k: int | dict[str, int] | str = 30,
    min_size: int = 10,
    n_seeds: int = 5,
) -> dict[str, Any]:
    """Run one combined clustering experiment and return rich results."""
    metric_layers = _filter_layers_with_top_k(layers, top_k)
    combine_top_k: int | str = top_k
    combine_layers = layers
    if isinstance(top_k, dict):
        combine_layers = metric_layers
        combine_top_k = 0

    combined = combine_edge_layers(combine_layers, strategy=strategy, gcc=True, top_k=combine_top_k)
    gamma_result = find_gamma(
        combined,
        target_max_pct=target_pct,
        min_size=min_size,
        postprocess=True,
    )
    stability = evaluate_stability(
        combined,
        gamma=gamma_result.gamma,
        n_seeds=n_seeds,
        min_size=min_size,
        postprocess=True,
    )
    quality = compute_quality_report(
        combined,
        gamma_result.membership,
        gamma=gamma_result.gamma,
        target_pct=target_pct,
        layer_tables=metric_layers,
        stability=stability,
    )
    membership_df = membership_frame_from_edges(combined, gamma_result.membership)
    return {
        "combined": combined,
        "gamma_result": gamma_result,
        "stability": stability,
        "quality": quality,
        "membership_df": membership_df,
        "membership_map": dict(zip(membership_df["uid"].to_list(), membership_df["cluster"].to_list())),
        "layer_names": sorted(layers.keys()),
        "label": layer_combo_label(layers),
        "n_layers": len(layers),
        "top_k": top_k,
        "layer_top_k": top_k if isinstance(top_k, dict) else {name: top_k for name in sorted(layers)},
        "min_size": min_size,
    }


def serialize_run(
    run: dict[str, Any],
    *,
    method: str,
    strategy: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert a raw ``run_combination`` output to JSON-friendly metrics."""
    gamma_result = run["gamma_result"]
    stability = run["stability"]
    quality = run["quality"]
    payload = {
        "method": method,
        "label": run["label"],
        "layers": run["layer_names"],
        "n_layers": run["n_layers"],
        "strategy": strategy,
        "top_k": run["top_k"],
        "layer_top_k": run["layer_top_k"],
        "min_size": run["min_size"],
        "n_edges": run["combined"].height,
        "gamma": gamma_result.gamma,
        "n_clusters": gamma_result.n_clusters,
        "max_pct": gamma_result.max_pct,
        "top5": gamma_result.top5,
        "ami_mean": stability.ami_mean,
        "ami_std": stability.ami_std,
        "ari_mean": stability.ari_mean,
        "ari_std": stability.ari_std,
        "singleton_pct": quality.singleton_pct,
        "consensus_edges": quality.consensus_edge_pct,
    }
    if extra:
        payload.update(extra)
    return payload


def save_json(payload: Any, out_path: Path) -> None:
    """Write JSON with stable formatting."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
