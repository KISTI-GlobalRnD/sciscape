"""Shared helpers for consensus research scripts."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import polars as pl

# Add project root to path for direct script execution.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sciscape.clustering.auto_gamma import AutoGammaResult, find_gamma
from sciscape.clustering.integer_remap import integer_remap_memory
from sciscape.clustering.leiden_rust import (
    build_leiden_graph,
    postprocess_small_clusters_rust,
    run_leiden_rust,
)
from sciscape.evaluation.stability import compute_quality_report, evaluate_stability
from sciscape.linkage.combine import combine_edge_layers
from sciscape.linkage.filters import filter_top_k


LAYER_FILE_CANDIDATES = {
    "bc_cosine": ("bc_cosine",),
    "cc_cosine": ("cc_cosine",),
    "dc_fractional": ("dc_fractional",),
    "emb_knn": (
        "emb_knn_textfilt_txt20_abs20_reqabs",
        "emb_full_knn30_textfilt_txt20_abs20_reqabs",
        "emb_knn",
        "emb_full_knn30",
    ),
}


def infer_emb_mode(path: Path | None) -> str | None:
    """Return ``filtered`` / ``unfiltered`` / ``None`` for the embedding path."""
    if path is None:
        return None
    return "filtered" if "textfilt" in path.name else "unfiltered"


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
        if canonical_name == "emb_knn":
            exact_filtered = [edge_dir / f"{candidate}.parquet" for candidate in candidates if "textfilt" in candidate]
            exact_filtered = [path for path in exact_filtered if path.exists()]
            filtered_matches = sorted(edge_dir.glob("emb*_textfilt*.parquet"))
            if len(filtered_matches) > 1:
                names = ", ".join(path.name for path in filtered_matches)
                raise ValueError(f"Ambiguous filtered embedding candidates in {edge_dir}: {names}")
            if len(exact_filtered) == 1:
                layer_paths[canonical_name] = exact_filtered[0]
                continue
            if filtered_matches:
                layer_paths[canonical_name] = filtered_matches[0]
                continue
        for candidate in candidates:
            path = edge_dir / f"{candidate}.parquet"
            if path.exists():
                layer_paths[canonical_name] = path
                break
    return layer_paths


def layer_provenance(layer_paths: dict[str, Path]) -> dict[str, Any]:
    """Return JSON-friendly provenance metadata for resolved layer inputs."""
    ordered = {name: layer_paths[name] for name in sorted(layer_paths)}
    emb_path = ordered.get("emb_knn")
    return {
        "layer_paths": {name: str(path) for name, path in ordered.items()},
        "layer_file_names": {name: path.name for name, path in ordered.items()},
        "emb_path": str(emb_path) if emb_path is not None else None,
        "emb_mode": infer_emb_mode(emb_path),
    }


def load_layer_tables(edge_dir: Path) -> dict[str, pl.DataFrame]:
    """Load all standard layer parquet files from *edge_dir*."""
    return {name: pl.read_parquet(path) for name, path in load_layer_paths(edge_dir).items()}


def validate_field_embedding_contract(field: str, layer_paths: dict[str, Path]) -> None:
    """Enforce the naming convention that ``*_textfilt`` uses filtered embeddings."""
    if "_textfilt" not in field:
        return
    emb_path = layer_paths.get("emb_knn")
    emb_mode = infer_emb_mode(emb_path)
    if emb_mode != "filtered":
        raise RuntimeError(
            f"Field label '{field}' implies filtered embeddings, but resolved emb path is "
            f"{emb_path if emb_path is not None else '<missing>'}"
        )


def select_layers(
    layers: dict[str, pl.DataFrame],
    *,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> dict[str, pl.DataFrame]:
    """Return a layer subset while preserving canonical names."""
    selected = dict(layers)
    if include is not None:
        include_set = set(include)
        selected = {name: table for name, table in selected.items() if name in include_set}
    if exclude:
        exclude_set = set(exclude)
        selected = {name: table for name, table in selected.items() if name not in exclude_set}
    return selected


def _normalize_top_k_value(top_k: int | dict[str, int] | str) -> Any:
    """Return a JSON-serialisable stable representation of ``top_k``."""
    if isinstance(top_k, dict):
        return {key: int(top_k[key]) for key in sorted(top_k)}
    return top_k


def _layer_top_k_metadata(layer_names: list[str], top_k: int | dict[str, int] | str) -> dict[str, int | None]:
    """Return per-layer top-k metadata with a stable, JSON-friendly shape."""
    ordered = sorted(layer_names)
    if isinstance(top_k, dict):
        return {name: int(top_k.get(name, 0)) for name in ordered}
    if isinstance(top_k, int):
        return {name: int(top_k) for name in ordered}
    return {name: None for name in ordered}


def combine_layers(
    layers: dict[str, pl.DataFrame],
    *,
    strategy: str = "consensus",
    top_k: int | dict[str, int] | str = 30,
) -> tuple[pl.DataFrame, dict[str, pl.DataFrame]]:
    """Build the combined graph without running gamma search or stability."""
    metric_layers = _filter_layers_with_top_k(layers, top_k)
    combine_top_k: int | str = top_k
    combine_layers_input = layers
    if isinstance(top_k, dict):
        combine_layers_input = metric_layers
        combine_top_k = 0
    combined = combine_edge_layers(combine_layers_input, strategy=strategy, gcc=True, top_k=combine_top_k)
    return combined, metric_layers


def select_best_single_result(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Pick the strongest single-layer result with deterministic tie-breakers."""
    singles = [item for item in results if item.get("kind") == "single_layer"]
    if not singles:
        return None
    return max(
        singles,
        key=lambda item: (
            float(item.get("ami_mean", float("-inf"))),
            -float(item.get("ami_std", float("inf"))),
            -float(item.get("max_pct", float("inf"))),
        ),
    )


def gamma_cache_key(
    *,
    edge_dir: Path,
    strategy: str,
    layer_names: list[str],
    top_k: int | dict[str, int] | str,
    target_pct: float,
    min_size: int,
    protocol: str | None = None,
    cache_context: dict[str, Any] | None = None,
) -> str:
    """Stable cache key for a clustering configuration."""
    payload = {
        "edge_dir": str(Path(edge_dir)),
        "strategy": strategy,
        "layers": sorted(layer_names),
        "top_k": _normalize_top_k_value(top_k),
        "target_pct": float(target_pct),
        "min_size": int(min_size),
    }
    if protocol is not None:
        payload["protocol"] = protocol
    if cache_context:
        payload["cache_context"] = cache_context
    return json.dumps(payload, sort_keys=True, ensure_ascii=True)


def load_gamma_cache(cache_path: Path | None) -> dict[str, Any]:
    """Load a gamma cache JSON file if present."""
    if cache_path is None or not cache_path.exists():
        return {}
    return json.loads(cache_path.read_text(encoding="utf-8"))


def resolve_cached_gamma(
    cache_path: Path | None,
    *,
    edge_dir: Path,
    strategy: str,
    layer_names: list[str],
    top_k: int | dict[str, int] | str,
    target_pct: float,
    min_size: int,
    protocol: str | None = None,
    cache_context: dict[str, Any] | None = None,
) -> float | None:
    """Return cached gamma for a config, if available."""
    cache = load_gamma_cache(cache_path)
    key = gamma_cache_key(
        edge_dir=edge_dir,
        strategy=strategy,
        layer_names=layer_names,
        top_k=top_k,
        target_pct=target_pct,
        min_size=min_size,
        protocol=protocol,
        cache_context=cache_context,
    )
    record = cache.get(key)
    if not record:
        return None
    value = record.get("gamma")
    return float(value) if value is not None else None


def update_gamma_cache(
    cache_path: Path | None,
    *,
    edge_dir: Path,
    strategy: str,
    layer_names: list[str],
    top_k: int | dict[str, int] | str,
    target_pct: float,
    min_size: int,
    gamma_result: AutoGammaResult,
    extra: dict[str, Any] | None = None,
    protocol: str | None = None,
    cache_context: dict[str, Any] | None = None,
) -> None:
    """Store gamma metadata for a clustering configuration."""
    if cache_path is None:
        return
    cache = load_gamma_cache(cache_path)
    key = gamma_cache_key(
        edge_dir=edge_dir,
        strategy=strategy,
        layer_names=layer_names,
        top_k=top_k,
        target_pct=target_pct,
        min_size=min_size,
        protocol=protocol,
        cache_context=cache_context,
    )
    record = {
        "gamma": float(gamma_result.gamma),
        "n_clusters": int(gamma_result.n_clusters),
        "max_pct": float(gamma_result.max_pct),
        "top5": list(gamma_result.top5),
    }
    if extra:
        record.update(extra)
    cache[key] = record
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")


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
    gamma: float | None = None,
    compute_stability: bool = True,
    compute_quality: bool = True,
) -> dict[str, Any]:
    """Run one combined clustering experiment and return rich results."""
    combined, metric_layers = combine_layers(
        layers,
        strategy=strategy,
        top_k=top_k,
    )
    if gamma is None:
        gamma_result = find_gamma(
            combined,
            target_max_pct=target_pct,
            min_size=min_size,
            postprocess=True,
        )
    else:
        src, dst, weight, n_nodes, _uids = integer_remap_memory(combined)
        try:
            graph = build_leiden_graph(
                n_nodes=n_nodes,
                edges_src=src,
                edges_dst=dst,
                edges_weight=weight,
            )
        except AttributeError:
            graph = None
        if graph is not None:
            leiden = graph.run_leiden(
                resolution=float(gamma),
                seed=42,
                n_iterations=10,
            )
        else:
            leiden = run_leiden_rust(
                resolution=float(gamma),
                n_nodes=n_nodes,
                edges_src=src,
                edges_dst=dst,
                edges_weight=weight,
                seed=42,
                n_iterations=10,
            )
        membership = leiden.membership
        if min_size > 0:
            if graph is not None:
                post = graph.postprocess_small_clusters(
                    resolution=float(gamma),
                    min_size=min_size,
                    membership=membership,
                    seed=42,
                    n_iterations=10,
                )
            else:
                post = postprocess_small_clusters_rust(
                    resolution=float(gamma),
                    min_size=min_size,
                    membership=membership,
                    n_nodes=n_nodes,
                    edges_src=src,
                    edges_dst=dst,
                    edges_weight=weight,
                    seed=42,
                    n_iterations=10,
                )
            membership = post.membership
            n_clusters = post.n_clusters
        else:
            n_clusters = leiden.n_clusters

        cluster_sizes = (
            pl.DataFrame({"cluster": membership})
            .group_by("cluster")
            .len()
            .sort("len", descending=True)["len"]
            .to_list()
        )
        max_pct = round(cluster_sizes[0] / n_nodes * 100, 2) if cluster_sizes else 0.0
        gamma_result = AutoGammaResult(
            gamma=float(gamma),
            n_clusters=int(n_clusters),
            max_pct=max_pct,
            top5=cluster_sizes[:5],
            probes=[],
            membership=membership,
        )
    stability = None
    if compute_stability:
        stability = evaluate_stability(
            combined,
            gamma=gamma_result.gamma,
            n_seeds=n_seeds,
            min_size=min_size,
            postprocess=True,
        )
    quality = None
    if compute_quality:
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
        "layer_top_k": _layer_top_k_metadata(list(layers), top_k),
        "min_size": min_size,
        "compute_stability": compute_stability,
        "compute_quality": compute_quality,
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
    if stability is None or quality is None:
        raise ValueError("serialize_run requires run_combination(..., compute_stability=True, compute_quality=True)")
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
