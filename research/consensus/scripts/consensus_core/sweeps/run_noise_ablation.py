"""Compare structural outcomes for sum/consensus noise-ablation variants."""

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
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from _common import (
    load_layer_tables,
    resolve_cached_gamma,
    run_combination,
    save_json,
    select_layers,
    serialize_run,
    update_gamma_cache,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

ABLATIONS = {
    "sum_all": {"strategy": "sum", "include": None, "exclude": None},
    "sum_minus_cc": {"strategy": "sum", "include": None, "exclude": ["cc_cosine"]},
    "sum_minus_emb": {"strategy": "sum", "include": None, "exclude": ["emb_knn"]},
    "cc_only": {"strategy": "rank", "include": ["cc_cosine"], "exclude": None},
    "emb_only": {"strategy": "rank", "include": ["emb_knn"], "exclude": None},
    "consensus_all": {"strategy": "consensus", "include": None, "exclude": None},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("edge_dir", type=Path, help="Directory with edge parquet files")
    parser.add_argument("--field", type=str, required=True)
    parser.add_argument("--target-pct", type=float, default=3.0)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--min-size", type=int, default=10)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument(
        "--configs",
        type=str,
        default="sum_all,sum_minus_cc,sum_minus_emb,cc_only,emb_only,consensus_all",
        help="Comma-separated subset of ablation configs to run",
    )
    parser.add_argument("--gamma-cache", type=Path, default=None, help="Optional gamma cache JSON")
    parser.add_argument("-o", "--output", type=Path, default=Path("results"))
    args = parser.parse_args()

    layers = load_layer_tables(args.edge_dir)
    if not layers:
        raise FileNotFoundError(f"No standard layers found in {args.edge_dir}")

    requested = [item.strip() for item in args.configs.split(",") if item.strip()]
    unknown = [item for item in requested if item not in ABLATIONS]
    if unknown:
        raise ValueError(f"Unknown ablation configs: {unknown}")

    results: list[dict] = []
    for name in requested:
        cfg = ABLATIONS[name]
        subset = select_layers(layers, include=cfg["include"], exclude=cfg["exclude"])
        if not subset:
            log.warning("Skipping %s: empty layer selection", name)
            continue
        gamma = resolve_cached_gamma(
            args.gamma_cache,
            edge_dir=args.edge_dir,
            strategy=cfg["strategy"],
            layer_names=sorted(subset),
            top_k=args.top_k,
            target_pct=args.target_pct,
            min_size=args.min_size,
        )
        if gamma is not None:
            log.info("%-15s using cached gamma=%.9g", name, gamma)
        run = run_combination(
            subset,
            strategy=cfg["strategy"],
            target_pct=args.target_pct,
            top_k=args.top_k,
            min_size=args.min_size,
            n_seeds=args.n_seeds,
            gamma=gamma,
        )
        update_gamma_cache(
            args.gamma_cache,
            edge_dir=args.edge_dir,
            strategy=cfg["strategy"],
            layer_names=sorted(subset),
            top_k=args.top_k,
            target_pct=args.target_pct,
            min_size=args.min_size,
            gamma_result=run["gamma_result"],
            extra={"field": args.field, "label": name},
        )
        result = serialize_run(
            run,
            method=name,
            strategy=cfg["strategy"],
            extra={"layers": sorted(subset)},
        )
        results.append(result)
        log.info(
            "%-15s strat=%-10s layers=%-40s clusters=%5d max_pct=%4.2f AMI=%.3f±%.3f",
            name,
            cfg["strategy"],
            "+".join(sorted(subset)),
            result["n_clusters"],
            result["max_pct"],
            result["ami_mean"],
            result["ami_std"],
        )

    payload = {
        "field": args.field,
        "edge_dir": str(args.edge_dir),
        "target_pct": args.target_pct,
        "top_k": args.top_k,
        "min_size": args.min_size,
        "n_seeds": args.n_seeds,
        "results": results,
    }
    out_path = args.output / f"{args.field}_noise_ablation.json"
    save_json(payload, out_path)
    log.info("\nSaved → %s", out_path)


if __name__ == "__main__":
    main()
