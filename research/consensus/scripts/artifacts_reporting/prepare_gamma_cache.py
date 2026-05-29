"""Precompute and persist gamma values for rank-shift review configurations."""

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


from _common import (
    gamma_cache_key,
    load_gamma_cache,
    load_layer_tables,
    resolve_cached_gamma,
    run_combination,
    save_json,
    select_layers,
    update_gamma_cache,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

VALID_METHODS = ("sum", "consensus", "rank", "max", "vote")

def _parse_layer_list(raw: str) -> list[str] | None:
    raw = raw.strip()
    if raw in {"", "*", "all", "-", "none"}:
        return None
    return [item.strip() for item in raw.split(",") if item.strip()] or None

def _parse_optional_gamma(raw: str | None) -> float | None:
    if raw is None:
        return None
    raw = raw.strip()
    if raw in {"", "-", "auto", "cache", "none"}:
        return None
    return float(raw)

def _parse_config(spec: str) -> dict:
    parts = [part.strip() for part in spec.split("|")]
    if len(parts) not in {4, 5}:
        raise ValueError(
            "Config spec must be 'label|strategy|include_layers|exclude_layers|gamma(optional)'"
        )
    label, strategy, include_raw, exclude_raw, *rest = parts
    if strategy not in VALID_METHODS:
        raise ValueError(f"Unknown strategy '{strategy}' in config '{spec}'")
    return {
        "label": label,
        "strategy": strategy,
        "include": _parse_layer_list(include_raw),
        "exclude": _parse_layer_list(exclude_raw),
        "gamma": _parse_optional_gamma(rest[0]) if rest else None,
    }

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("edge_dir", type=Path, help="Directory with edge parquet files")
    parser.add_argument("--field", type=str, required=True)
    parser.add_argument(
        "--config",
        action="append",
        required=True,
        help="label|strategy|include_layers|exclude_layers|gamma(optional)",
    )
    parser.add_argument("--target-pct", type=float, default=3.0)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--min-size", type=int, default=10)
    parser.add_argument("--gamma-cache", type=Path, required=True, help="Gamma cache JSON path")
    parser.add_argument("--force", action="store_true", help="Recompute even when cached gamma exists")
    parser.add_argument("-o", "--output", type=Path, default=Path("results"))
    args = parser.parse_args()

    layers = load_layer_tables(args.edge_dir)
    if not layers:
        raise FileNotFoundError(f"No standard layers found in {args.edge_dir}")

    configs = [_parse_config(spec) for spec in args.config]
    labels = [cfg["label"] for cfg in configs]
    if len(labels) != len(set(labels)):
        raise ValueError(f"Config labels must be unique: {labels}")

    cache_before = load_gamma_cache(args.gamma_cache)
    results: list[dict] = []
    for cfg in configs:
        subset = select_layers(layers, include=cfg["include"], exclude=cfg["exclude"])
        if not subset:
            raise ValueError(f"Config '{cfg['label']}' resolved to an empty layer selection")

        cached_gamma = resolve_cached_gamma(
            args.gamma_cache,
            edge_dir=args.edge_dir,
            strategy=cfg["strategy"],
            layer_names=sorted(subset),
            top_k=args.top_k,
            target_pct=args.target_pct,
            min_size=args.min_size,
        )
        cache_key = gamma_cache_key(
            edge_dir=args.edge_dir,
            strategy=cfg["strategy"],
            layer_names=sorted(subset),
            top_k=args.top_k,
            target_pct=args.target_pct,
            min_size=args.min_size,
        )
        if cached_gamma is not None and not args.force and cfg["gamma"] is None:
            record = cache_before.get(cache_key, {})
            log.info("%-20s using cached gamma=%.9g", cfg["label"], cached_gamma)
            results.append(
                {
                    "label": cfg["label"],
                    "strategy": cfg["strategy"],
                    "layers": sorted(subset),
                    "status": "cached",
                    "gamma": cached_gamma,
                    "n_clusters": record.get("n_clusters"),
                    "max_pct": record.get("max_pct"),
                }
            )
            continue

        gamma_override = cfg["gamma"] if cfg["gamma"] is not None else cached_gamma
        run = run_combination(
            subset,
            strategy=cfg["strategy"],
            target_pct=args.target_pct,
            top_k=args.top_k,
            min_size=args.min_size,
            n_seeds=1,
            gamma=gamma_override,
            compute_stability=False,
            compute_quality=False,
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
            extra={"field": args.field, "label": cfg["label"]},
        )
        log.info(
            "%-20s cached gamma=%.9g clusters=%5d max_pct=%4.2f",
            cfg["label"],
            run["gamma_result"].gamma,
            run["gamma_result"].n_clusters,
            run["gamma_result"].max_pct,
        )
        results.append(
            {
                "label": cfg["label"],
                "strategy": cfg["strategy"],
                "layers": sorted(subset),
                "status": "computed" if cfg["gamma"] is None else "computed_from_override",
                "gamma": run["gamma_result"].gamma,
                "n_clusters": run["gamma_result"].n_clusters,
                "max_pct": run["gamma_result"].max_pct,
            }
        )

    payload = {
        "field": args.field,
        "edge_dir": str(args.edge_dir),
        "gamma_cache": str(args.gamma_cache),
        "top_k": args.top_k,
        "target_pct": args.target_pct,
        "min_size": args.min_size,
        "force": args.force,
        "results": results,
    }
    out_path = args.output / f"{args.field}_k{args.top_k:02d}_gamma_cache_prep.json"
    save_json(payload, out_path)
    log.info("\nSaved → %s", out_path)

if __name__ == "__main__":
    main()
