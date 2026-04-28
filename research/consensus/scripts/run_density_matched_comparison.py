"""Protocol C: edge-count-matched consensus vs best single-layer baseline."""

from __future__ import annotations

import argparse
from math import isclose
import logging
from pathlib import Path

import polars as pl

from _common import (
    combine_layers,
    layer_provenance,
    load_layer_paths,
    resolve_cached_gamma,
    run_combination,
    save_json,
    select_best_single_result,
    serialize_run,
    update_gamma_cache,
    validate_field_embedding_contract,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

SINGLE_LAYER_METHODS = ("bc_cosine", "cc_cosine", "dc_fractional", "emb_knn")
CONSENSUS_METHODS = ("citation_consensus", "all_consensus")


def _resolve_output_path(output_arg: Path, field: str, consensus_method: str) -> Path:
    if output_arg.suffix == ".json":
        return output_arg
    return output_arg / f"{field}_{consensus_method}_density_matched_comparison.json"


def _single_layer_runs(
    layers: dict,
    *,
    target_pct: float,
    top_k: int,
    min_size: int,
    n_seeds: int,
) -> list[dict]:
    results: list[dict] = []
    for name in SINGLE_LAYER_METHODS:
        table = layers.get(name)
        if table is None:
            continue
        run = run_combination(
            {name: table},
            strategy="rank",
            target_pct=target_pct,
            top_k=top_k,
            min_size=min_size,
            n_seeds=n_seeds,
        )
        results.append(
            serialize_run(
                run,
                method=f"{name}_only",
                strategy="rank",
                extra={"kind": "single_layer", "protocol": "practical_top_k"},
            )
        )
    return results


def _consensus_layers(all_layers: dict, consensus_method: str) -> dict:
    if consensus_method == "citation_consensus":
        return {k: v for k, v in all_layers.items() if k in {"bc_cosine", "cc_cosine", "dc_fractional"}}
    if consensus_method == "all_consensus":
        return dict(all_layers)
    raise ValueError(f"Unsupported consensus method: {consensus_method}")


def _uniform_scale_top_k(original: dict[str, int], scale: float) -> dict[str, int]:
    return {name: max(1, int(round(value * scale))) for name, value in sorted(original.items())}


def _search_scales(scale_min: float, scale_max: float, scale_step: float) -> list[float]:
    scales: list[float] = []
    current = scale_min
    while current <= scale_max + 1e-12:
        scales.append(round(current, 6))
        current += scale_step
    if not any(isclose(scale, 1.0, rel_tol=0.0, abs_tol=1e-9) for scale in scales):
        scales.append(1.0)
    return sorted(set(scales))


def _search_density_match(
    layers: dict,
    *,
    strategy: str,
    original_layer_top_k: dict[str, int],
    target_edge_count: int,
    tolerance: float,
    scale_min: float,
    scale_max: float,
    scale_step: float,
) -> tuple[dict | None, dict]:
    seen: set[tuple[tuple[str, int], ...]] = set()
    candidates: list[dict] = []
    best_overall: dict | None = None

    for scale in _search_scales(scale_min, scale_max, scale_step):
        layer_top_k = _uniform_scale_top_k(original_layer_top_k, scale)
        key = tuple(sorted(layer_top_k.items()))
        if key in seen:
            continue
        seen.add(key)
        combined, _metric_layers = combine_layers(layers, strategy=strategy, top_k=layer_top_k)
        achieved = combined.height
        rel_err = abs(achieved - target_edge_count) / max(target_edge_count, 1)
        candidate = {
            "scale": scale,
            "scale_distance_from_one": abs(scale - 1.0),
            "layer_top_k": layer_top_k,
            "achieved_edge_count": achieved,
            "relative_edge_error": rel_err,
            "sum_top_k": sum(layer_top_k.values()),
        }
        candidates.append(candidate)
        if best_overall is None or (
            rel_err,
            candidate["scale_distance_from_one"],
            candidate["sum_top_k"],
        ) < (
            best_overall["relative_edge_error"],
            best_overall["scale_distance_from_one"],
            best_overall["sum_top_k"],
        ):
            best_overall = candidate

    feasible = [cand for cand in candidates if cand["relative_edge_error"] <= tolerance]
    if feasible:
        feasible.sort(
            key=lambda cand: (
                cand["relative_edge_error"],
                cand["scale_distance_from_one"],
                cand["sum_top_k"],
            )
        )
        return feasible[0], {
            "n_candidates_considered": len(candidates),
            "best_overall": best_overall,
            "closest_within_tolerance": feasible[0],
        }

    return None, {
        "n_candidates_considered": len(candidates),
        "best_overall": best_overall,
        "closest_within_tolerance": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Protocol C: edge-count-matched comparison")
    parser.add_argument("edge_dir", type=Path, help="Directory with edge parquet files")
    parser.add_argument("--field", type=str, required=True)
    parser.add_argument(
        "--consensus-method",
        type=str,
        default="citation_consensus",
        choices=CONSENSUS_METHODS,
        help="Consensus family to compare against the best Protocol A single layer",
    )
    parser.add_argument(
        "--baseline-method",
        type=str,
        default=None,
        choices=SINGLE_LAYER_METHODS,
        help="Optional single-layer override; default is Protocol A best single layer",
    )
    parser.add_argument("--target-pct", type=float, default=3.0)
    parser.add_argument("--top-k", type=int, default=30, help="Protocol A practical per-layer top-k")
    parser.add_argument("--min-size", type=int, default=10)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--edge-tolerance", type=float, default=0.05, help="Relative edge-count tolerance")
    parser.add_argument("--scale-min", type=float, default=0.05)
    parser.add_argument("--scale-max", type=float, default=2.0)
    parser.add_argument("--scale-step", type=float, default=0.02)
    parser.add_argument("--gamma-cache", type=Path, default=None, help="Optional gamma cache JSON")
    parser.add_argument("-o", "--output", type=Path, default=Path("results"))
    args = parser.parse_args()

    if args.output.suffix != ".json":
        args.output.mkdir(parents=True, exist_ok=True)
    layer_paths = load_layer_paths(args.edge_dir)
    validate_field_embedding_contract(args.field, layer_paths)
    layers = {name: pl.read_parquet(path) for name, path in layer_paths.items()}
    if not layers:
        raise FileNotFoundError(f"No standard edge parquet files found in {args.edge_dir}")

    log.info("Field: %s", args.field)
    log.info("Protocol C: %s", args.consensus_method)
    log.info("Protocol A anchor top_k=%d", args.top_k)

    single_results = _single_layer_runs(
        layers,
        target_pct=args.target_pct,
        top_k=args.top_k,
        min_size=args.min_size,
        n_seeds=args.n_seeds,
    )
    if not single_results:
        raise RuntimeError("No single-layer runs available to choose a baseline")

    if args.baseline_method is not None:
        baseline_method = f"{args.baseline_method}_only"
        baseline_result = next((row for row in single_results if row["method"] == baseline_method), None)
        if baseline_result is None:
            raise ValueError(f"Requested baseline method not available: {baseline_method}")
    else:
        baseline_result = select_best_single_result(single_results)
        if baseline_result is None:
            raise RuntimeError("Failed to select a best single-layer baseline")
        baseline_method = baseline_result["method"]

    consensus_layers = _consensus_layers(layers, args.consensus_method)
    if len(consensus_layers) < 2:
        raise RuntimeError(f"{args.consensus_method} requires at least two layers")

    original_layer_top_k = {name: args.top_k for name in sorted(consensus_layers)}
    practical_consensus = run_combination(
        consensus_layers,
        strategy="consensus",
        target_pct=args.target_pct,
        top_k=original_layer_top_k,
        min_size=args.min_size,
        n_seeds=args.n_seeds,
    )
    practical_consensus_result = serialize_run(
        practical_consensus,
        method=args.consensus_method,
        strategy="consensus",
        extra={"kind": "multi_layer", "protocol": "practical_top_k"},
    )

    matched_candidate, diagnostics = _search_density_match(
        consensus_layers,
        strategy="consensus",
        original_layer_top_k=original_layer_top_k,
        target_edge_count=baseline_result["n_edges"],
        tolerance=args.edge_tolerance,
        scale_min=args.scale_min,
        scale_max=args.scale_max,
        scale_step=args.scale_step,
    )

    payload = {
        "field": args.field,
        "edge_dir": str(args.edge_dir),
        "protocol": "edge_count_matched",
        "status": "running",
        "search_strategy": "uniform",
        "consensus_method": args.consensus_method,
        "baseline_method": baseline_method,
        "baseline_selection_rule": "best Protocol A single-layer by ami_mean, then lower ami_std, then lower max_pct",
        "target_pct": args.target_pct,
        "protocol_a_top_k": args.top_k,
        "min_size": args.min_size,
        "n_seeds": args.n_seeds,
        "edge_tolerance": args.edge_tolerance,
        "scale_min": args.scale_min,
        "scale_max": args.scale_max,
        "scale_step": args.scale_step,
        "protocol_a_single_layer_results": single_results,
        "protocol_a_consensus_reference": practical_consensus_result,
        "matching": {
            "target_edge_count": baseline_result["n_edges"],
            "original_layer_top_k": original_layer_top_k,
            **diagnostics,
        },
    }
    payload.update(layer_provenance(layer_paths))

    out_path = _resolve_output_path(args.output, args.field, args.consensus_method)

    if matched_candidate is None:
        payload["status"] = "failed"
        save_json(payload, out_path)
        raise RuntimeError(
            f"No edge-count-matched configuration within tolerance={args.edge_tolerance:.3f}; "
            f"closest rel_err={diagnostics['best_overall']['relative_edge_error']:.4f}"
        )

    matched_top_k = matched_candidate["layer_top_k"]
    protocol = "edge_count_matched"
    cache_context = {
        "search_strategy": "uniform",
        "target_edge_count": int(baseline_result["n_edges"]),
        "achieved_edge_count": int(matched_candidate["achieved_edge_count"]),
    }

    cached_gamma = resolve_cached_gamma(
        args.gamma_cache,
        edge_dir=args.edge_dir,
        strategy="consensus",
        layer_names=sorted(consensus_layers),
        top_k=matched_top_k,
        target_pct=args.target_pct,
        min_size=args.min_size,
        protocol=protocol,
        cache_context=cache_context,
    )
    if cached_gamma is not None:
        log.info("Reusing cached Protocol C gamma: %.9g", cached_gamma)

    matched_consensus = run_combination(
        consensus_layers,
        strategy="consensus",
        target_pct=args.target_pct,
        top_k=matched_top_k,
        min_size=args.min_size,
        n_seeds=args.n_seeds,
        gamma=cached_gamma,
    )
    update_gamma_cache(
        args.gamma_cache,
        edge_dir=args.edge_dir,
        strategy="consensus",
        layer_names=sorted(consensus_layers),
        top_k=matched_top_k,
        target_pct=args.target_pct,
        min_size=args.min_size,
        gamma_result=matched_consensus["gamma_result"],
        protocol=protocol,
        cache_context=cache_context,
        extra={
            "field": args.field,
            "label": args.consensus_method,
            "search_strategy": "uniform",
            "target_edge_count": int(baseline_result["n_edges"]),
            "achieved_edge_count": int(matched_candidate["achieved_edge_count"]),
        },
    )
    matched_consensus_result = serialize_run(
        matched_consensus,
        method=args.consensus_method,
        strategy="consensus",
        extra={"kind": "multi_layer", "protocol": protocol},
    )

    payload["baseline_run"] = baseline_result
    payload["matched_consensus_run"] = matched_consensus_result
    payload["matching"].update(
        {
            "matched_layer_top_k": matched_top_k,
            "achieved_edge_count": matched_candidate["achieved_edge_count"],
            "relative_edge_error": matched_candidate["relative_edge_error"],
            "chosen_scale": matched_candidate["scale"],
        }
    )
    payload["status"] = "ok"
    save_json(payload, out_path)
    log.info("Saved → %s", out_path)


if __name__ == "__main__":
    main()
