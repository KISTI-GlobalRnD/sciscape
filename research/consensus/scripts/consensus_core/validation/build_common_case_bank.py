"""Build a common rank-shift case bank shared across multiple pairwise comparisons."""

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


import numpy as np

from _common import (
    abstracts_lookup,
    load_abstracts_table,
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


def _serialize_config(cfg: dict, subset: dict) -> dict:
    return {
        "label": cfg["label"],
        "strategy": cfg["strategy"],
        "include": cfg["include"],
        "exclude": cfg["exclude"],
        "gamma_override": cfg["gamma"],
        "layers": sorted(subset),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("edge_dir", type=Path, help="Directory with edge parquet files")
    parser.add_argument("abstract_path", type=Path, help="Metadata parquet with uid/title/abstract/pubyear")
    parser.add_argument("--field", type=str, required=True)
    parser.add_argument(
        "--config",
        action="append",
        required=True,
        help="label|strategy|include_layers|exclude_layers|gamma(optional)",
    )
    parser.add_argument("--reference", type=str, required=True, help="Reference config label")
    parser.add_argument(
        "--compare",
        type=str,
        default=None,
        help="Comma-separated comparison labels (default: all configs except reference)",
    )
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--target-pct", type=float, default=3.0)
    parser.add_argument("--min-size", type=int, default=10)
    parser.add_argument("--n-neighbors", type=int, default=8)
    parser.add_argument("--max-rank-jaccard", type=float, default=0.85)
    parser.add_argument("--min-cluster-overlap", type=float, default=0.5)
    parser.add_argument("--n-targets", type=int, default=128)
    parser.add_argument(
        "--policy",
        choices=("intersection", "union"),
        default="intersection",
        help="How to merge candidate UID sets across comparisons",
    )
    parser.add_argument("--gamma-cache", type=Path, default=None, help="Optional gamma cache JSON")
    parser.add_argument("-o", "--output", type=Path, default=Path("results"))
    args = parser.parse_args()

    layers = load_layer_tables(args.edge_dir)
    if not layers:
        raise FileNotFoundError(f"No standard layers found in {args.edge_dir}")
    abstracts = load_abstracts_table(args.abstract_path)
    meta = abstracts_lookup(abstracts)
    reviewable_uids = set(meta)

    configs = [_parse_config(spec) for spec in args.config]
    config_by_label = {cfg["label"]: cfg for cfg in configs}
    if len(configs) != len(config_by_label):
        raise ValueError("Config labels must be unique")
    if args.reference not in config_by_label:
        raise ValueError(f"Reference '{args.reference}' not found in configs")

    if args.compare:
        compare_labels = [label.strip() for label in args.compare.split(",") if label.strip()]
    else:
        compare_labels = [cfg["label"] for cfg in configs if cfg["label"] != args.reference]
    if not compare_labels:
        raise ValueError("No comparison labels selected")
    unknown = [label for label in compare_labels if label not in config_by_label]
    if unknown:
        raise ValueError(f"Unknown compare labels: {unknown}")

    needed_labels = [args.reference, *compare_labels]
    run_by_label: dict[str, dict] = {}
    config_payloads: list[dict] = []
    for label in needed_labels:
        cfg = config_by_label[label]
        subset = select_layers(layers, include=cfg["include"], exclude=cfg["exclude"])
        if not subset:
            raise ValueError(f"Config '{label}' resolved to an empty layer selection")
        gamma = cfg["gamma"]
        if gamma is None:
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
                log.info("%-20s using cached gamma=%.9g", label, gamma)
        run = run_combination(
            subset,
            strategy=cfg["strategy"],
            target_pct=args.target_pct,
            top_k=args.top_k,
            min_size=args.min_size,
            n_seeds=1,
            gamma=gamma,
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
            extra={"field": args.field, "label": label},
        )
        run_by_label[label] = run
        config_payloads.append(_serialize_config(cfg, subset))
        log.info(
            "%-20s strat=%-10s layers=%-40s gamma=%.9g clusters=%5d",
            label,
            cfg["strategy"],
            "+".join(sorted(subset)),
            run["gamma_result"].gamma,
            run["gamma_result"].n_clusters,
        )

    from sciscape.evaluation.sampler import collect_rank_shift_cases

    reference_label = args.reference
    reference_run = run_by_label[reference_label]
    case_maps: dict[str, dict] = {}
    comparison_stats: list[dict] = []
    uid_sets: list[set[str]] = []
    for compare_label in compare_labels:
        compare_run = run_by_label[compare_label]
        cases, n_eligible = collect_rank_shift_cases(
            reference_run["combined"],
            reference_run["membership_map"],
            compare_run["combined"],
            compare_run["membership_map"],
            method_a=reference_label,
            method_b=compare_label,
            abstracts=abstracts,
            n_neighbors=args.n_neighbors,
            min_cluster_size=args.min_size,
            max_rank_jaccard=args.max_rank_jaccard,
            min_cluster_overlap=args.min_cluster_overlap,
            allowed_uids=reviewable_uids,
        )
        case_map = {case.target_uid: case for case in cases}
        case_maps[compare_label] = case_map
        uid_sets.append(set(case_map))
        comparison_stats.append(
            {
                "label": compare_label,
                "n_eligible": n_eligible,
                "n_candidates": len(cases),
                "reference_clusters": reference_run["gamma_result"].n_clusters,
                "comparison_clusters": compare_run["gamma_result"].n_clusters,
            }
        )
        log.info(
            "Case pool %-14s candidates=%5d eligible=%6d",
            compare_label,
            len(cases),
            n_eligible,
        )

    if not uid_sets:
        raise ValueError("No case maps built")
    if args.policy == "intersection":
        selected_pool = set.intersection(*uid_sets)
    else:
        selected_pool = set.union(*uid_sets)

    ordered_targets: list[dict] = []
    for uid in selected_pool:
        per_compare = {}
        shift_scores = []
        rank_jaccards = []
        mean_rank_shifts = []
        cluster_overlaps = []
        cluster_changed = []
        for compare_label, case_map in case_maps.items():
            case = case_map.get(uid)
            if case is None:
                continue
            per_compare[compare_label] = {
                "shift_score": case.shift_score,
                "rank_jaccard": case.rank_jaccard,
                "mean_abs_rank_shift": case.mean_abs_rank_shift,
                "cluster_overlap_coeff": case.cluster_overlap_coeff,
                "cluster_changed": case.cluster_changed,
                "reference_cluster_size": case.method_a_cluster_size,
                "comparison_cluster_size": case.method_b_cluster_size,
            }
            shift_scores.append(case.shift_score)
            rank_jaccards.append(case.rank_jaccard)
            mean_rank_shifts.append(case.mean_abs_rank_shift)
            cluster_overlaps.append(case.cluster_overlap_coeff)
            cluster_changed.append(1 if case.cluster_changed else 0)
        ordered_targets.append(
            {
                "target_uid": uid,
                "target_title": meta.get(uid, {}).get("title", "") or "",
                "target_year": meta.get(uid, {}).get("pubyear"),
                "n_comparisons": len(per_compare),
                "avg_shift_score": round(float(np.mean(shift_scores)), 4) if shift_scores else None,
                "avg_rank_jaccard": round(float(np.mean(rank_jaccards)), 4) if rank_jaccards else None,
                "avg_mean_abs_rank_shift": (
                    round(float(np.mean(mean_rank_shifts)), 4) if mean_rank_shifts else None
                ),
                "avg_cluster_overlap_coeff": (
                    round(float(np.mean(cluster_overlaps)), 4) if cluster_overlaps else None
                ),
                "cluster_changed_rate": (
                    round(float(np.mean(cluster_changed)), 4) if cluster_changed else None
                ),
                "comparisons": per_compare,
            }
        )

    ordered_targets.sort(
        key=lambda row: (
            -row["n_comparisons"],
            -(row["avg_shift_score"] if row["avg_shift_score"] is not None else -1.0),
            row["avg_rank_jaccard"] if row["avg_rank_jaccard"] is not None else 1.0,
            row["target_uid"],
        )
    )
    ordered_targets = ordered_targets[: args.n_targets]

    payload = {
        "field": args.field,
        "edge_dir": str(args.edge_dir),
        "abstract_path": str(args.abstract_path),
        "reference": reference_label,
        "compare_labels": compare_labels,
        "policy": args.policy,
        "top_k": args.top_k,
        "target_pct": args.target_pct,
        "min_size": args.min_size,
        "n_neighbors": args.n_neighbors,
        "max_rank_jaccard": args.max_rank_jaccard,
        "min_cluster_overlap": args.min_cluster_overlap,
        "n_target_uids": len(ordered_targets),
        "target_uids": [row["target_uid"] for row in ordered_targets],
        "target_summaries": ordered_targets,
        "comparison_stats": comparison_stats,
        "configs": config_payloads,
        "gamma_cache": str(args.gamma_cache) if args.gamma_cache is not None else None,
    }
    out_path = args.output / f"{args.field}_k{args.top_k:02d}_{reference_label}_common_case_bank.json"
    save_json(payload, out_path)
    log.info("\nSaved → %s", out_path)


if __name__ == "__main__":
    main()
