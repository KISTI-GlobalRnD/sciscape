"""E1: Same effective-k single-layer vs multi-layer comparison."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
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


import polars as pl

from _common import (
    allocate_effective_k,
    layer_provenance,
    load_layer_paths,
    run_combination,
    save_json,
    select_best_single_result,
    serialize_run,
    validate_field_embedding_contract,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

SINGLE_LAYER_METHODS = ("bc_cosine", "cc_cosine", "dc_fractional", "emb_knn")

def _run_single_layer(
    name: str,
    table,
    *,
    target_pct: float,
    top_k: int | dict[str, int],
    min_size: int,
    n_seeds: int,
) -> dict:
    run = run_combination(
        {name: table},
        strategy="rank",
        target_pct=target_pct,
        top_k=top_k,
        min_size=min_size,
        n_seeds=n_seeds,
    )
    return serialize_run(
        run,
        method=f"{name}_only",
        strategy="rank",
        extra={"kind": "single_layer"},
    )

def _run_multi_layer(
    layers: dict,
    *,
    method: str,
    target_pct: float,
    top_k: int | dict[str, int],
    min_size: int,
    n_seeds: int,
) -> dict:
    run = run_combination(
        layers,
        strategy="consensus",
        target_pct=target_pct,
        top_k=top_k,
        min_size=min_size,
        n_seeds=n_seeds,
    )
    return serialize_run(
        run,
        method=method,
        strategy="consensus",
        extra={"kind": "multi_layer"},
    )

def main() -> None:
    parser = argparse.ArgumentParser(description="E1: single-layer vs multi-layer consensus comparison")
    parser.add_argument("edge_dir", type=Path, help="Directory with edge parquet files")
    parser.add_argument("--field", type=str, required=True, help="Field identifier (e.g., field_15)")
    parser.add_argument("--target-pct", type=float, default=3.0)
    parser.add_argument(
        "--effective-k",
        type=int,
        default=30,
        help="Global neighbor budget distributed across layers for fair same-budget comparison",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Legacy fixed per-layer top-k. If set, disables effective-k budgeting.",
    )
    parser.add_argument("--min-size", type=int, default=10)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("-o", "--output", type=Path, default=Path("results"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    layer_paths = load_layer_paths(args.edge_dir)
    validate_field_embedding_contract(args.field, layer_paths)
    layers = {name: pl.read_parquet(path) for name, path in layer_paths.items()}
    if not layers:
        raise FileNotFoundError(f"No standard edge parquet files found in {args.edge_dir}")

    log.info("Field: %s", args.field)
    log.info("Layers: %s", ", ".join(sorted(layers)))
    if args.top_k is None:
        log.info(
            "Comparison budget: effective_k=%d, min_size=%d, target_pct=%.1f",
            args.effective_k,
            args.min_size,
            args.target_pct,
        )
        budget_mode = "effective_k"
    else:
        log.info(
            "Comparison budget: fixed per-layer top_k=%d, min_size=%d, target_pct=%.1f",
            args.top_k,
            args.min_size,
            args.target_pct,
        )
        budget_mode = "per_layer_top_k"

    results: list[dict] = []

    log.info("\n=== Single-layer baselines ===")
    for name in SINGLE_LAYER_METHODS:
        if name not in layers:
            continue
        result = _run_single_layer(
            name,
            layers[name],
            target_pct=args.target_pct,
            top_k=args.top_k if args.top_k is not None else args.effective_k,
            min_size=args.min_size,
            n_seeds=args.n_seeds,
        )
        results.append(result)
        log.info(
            "  %-14s edges=%7d clusters=%4d AMI=%.3f±%.3f",
            name,
            result["n_edges"],
            result["n_clusters"],
            result["ami_mean"],
            result["ami_std"],
        )

    citation_layers = {k: v for k, v in layers.items() if k in {"bc_cosine", "cc_cosine", "dc_fractional"}}
    if len(citation_layers) >= 2:
        citation_top_k = (
            args.top_k
            if args.top_k is not None
            else allocate_effective_k(sorted(citation_layers), args.effective_k)
        )
        log.info("\n=== Citation consensus ===")
        log.info("  layer_top_k=%s", citation_top_k)
        result = _run_multi_layer(
            citation_layers,
            method="citation_consensus",
            target_pct=args.target_pct,
            top_k=citation_top_k,
            min_size=args.min_size,
            n_seeds=args.n_seeds,
        )
        results.append(result)
        log.info(
            "  citation_consensus edges=%7d clusters=%4d AMI=%.3f±%.3f",
            result["n_edges"],
            result["n_clusters"],
            result["ami_mean"],
            result["ami_std"],
        )

    if len(layers) >= 2:
        all_top_k = (
            args.top_k
            if args.top_k is not None
            else allocate_effective_k(sorted(layers), args.effective_k)
        )
        log.info("\n=== All-layer consensus ===")
        log.info("  layer_top_k=%s", all_top_k)
        result = _run_multi_layer(
            layers,
            method="all_consensus",
            target_pct=args.target_pct,
            top_k=all_top_k,
            min_size=args.min_size,
            n_seeds=args.n_seeds,
        )
        results.append(result)
        log.info(
            "  all_consensus      edges=%7d clusters=%4d AMI=%.3f±%.3f",
            result["n_edges"],
            result["n_clusters"],
            result["ami_mean"],
            result["ami_std"],
        )

    payload = {
        "field": args.field,
        "edge_dir": str(args.edge_dir),
        "protocol": "candidate_budget_matched" if args.top_k is None else "practical_top_k",
        "target_pct": args.target_pct,
        "budget_mode": budget_mode,
        "effective_k": args.effective_k,
        "top_k": args.top_k,
        "min_size": args.min_size,
        "n_seeds": args.n_seeds,
        "results": results,
    }
    payload.update(layer_provenance(layer_paths))
    best_single = select_best_single_result(results)
    if best_single is not None:
        payload["best_single_method"] = best_single["method"]
    out_path = args.output / f"{args.field}_comparison.json"
    save_json(payload, out_path)
    log.info("\nSaved → %s", out_path)

if __name__ == "__main__":
    main()
