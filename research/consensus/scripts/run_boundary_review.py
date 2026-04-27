"""E5: blind A/B review on disagreement boundary nodes."""

from __future__ import annotations

import argparse
import logging
import random
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np

from _common import (
    allocate_effective_k,
    abstracts_lookup,
    load_abstracts_table,
    load_layer_tables,
    run_combination,
    save_json,
    select_layers,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

VALID_METHODS = ("sum", "consensus", "rank", "max", "vote")


def _parse_layer_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def _docs_from_uids(uids: list[str], meta: dict[str, dict]) -> list[dict]:
    docs = []
    for uid in uids:
        record = meta.get(uid)
        if not record:
            continue
        docs.append(
            {
                "uid": uid,
                "title": record.get("title", "") or "",
                "abstract": record.get("abstract", "") or "",
                "pubyear": record.get("pubyear"),
            }
        )
    return docs


def _select_cases(cases: list[dict], n_cases: int, seed: int) -> list[dict]:
    rng = np.random.RandomState(seed)
    order = list(cases)
    rng.shuffle(order)
    return order[: min(n_cases, len(order))]


def _aggregate_counts(values: Iterable[str], *, positive: str, negative: str) -> dict[str, int]:
    counts = Counter(value for value in values if value in {positive, negative})
    total = counts.get(positive, 0) + counts.get(negative, 0)
    return {
        positive: counts.get(positive, 0),
        negative: counts.get(negative, 0),
        "total": total,
    }


def _mean(values: list[float | int]) -> float | None:
    if not values:
        return None
    return round(float(np.mean(values)), 4)


def _min_group_size(n_neighbors: int) -> int:
    return max(2, (n_neighbors + 1) // 2)


def _is_reviewable_case(case: dict, meta: dict[str, dict], *, min_group_size: int) -> bool:
    """Check whether a case has enough metadata to be sent to the reviewer."""
    target_docs = _docs_from_uids([case["target_uid"]], meta)
    if not target_docs:
        return False
    group_a_docs = _docs_from_uids(case["group_a_uids"], meta)
    group_b_docs = _docs_from_uids(case["group_b_uids"], meta)
    return len(group_a_docs) >= min_group_size and len(group_b_docs) >= min_group_size


def _resolve_top_k(layer_names: list[str], *, top_k: int, effective_k: int | None) -> int | dict[str, int]:
    if effective_k is None:
        return top_k
    if len(layer_names) <= 1:
        return effective_k
    return allocate_effective_k(layer_names, effective_k)


def _protocol_name(*, effective_k: int | None) -> str:
    return "candidate_budget_matched" if effective_k is not None else "practical_top_k"


def _summarize_reviews(
    reviewed_cases: list[dict],
    *,
    method_a: str,
    method_b: str,
    secondary_checks: bool,
) -> dict:
    comparison_votes = [case["comparison"]["winner"] for case in reviewed_cases]
    belonging_votes = [case["belonging"]["belongs_to"] for case in reviewed_cases]
    comparison_counts = _aggregate_counts(comparison_votes, positive="A", negative="B")
    belonging_counts = _aggregate_counts(belonging_votes, positive="A", negative="B")

    n_cases = len(reviewed_cases)
    comparison_valid = comparison_counts["total"]
    belonging_valid = belonging_counts["total"]
    comparison_score_a = [case["comparison"]["score_a"] for case in reviewed_cases]
    comparison_score_b = [case["comparison"]["score_b"] for case in reviewed_cases]
    belonging_conf = [case["belonging"]["confidence"] for case in reviewed_cases]

    summary = {
        "n_reviewed_cases": n_cases,
        "comparison": {
            "method_a": method_a,
            "method_b": method_b,
            "method_a_wins": comparison_counts["A"],
            "method_b_wins": comparison_counts["B"],
            "ties_or_invalid": n_cases - comparison_valid,
            "method_a_win_rate": round(comparison_counts["A"] / n_cases, 4) if n_cases else None,
            "method_b_win_rate": round(comparison_counts["B"] / n_cases, 4) if n_cases else None,
            "method_a_win_rate_no_ties": round(comparison_counts["A"] / comparison_valid, 4) if comparison_valid else None,
            "method_b_win_rate_no_ties": round(comparison_counts["B"] / comparison_valid, 4) if comparison_valid else None,
            "score_a_mean": _mean(comparison_score_a),
            "score_b_mean": _mean(comparison_score_b),
            "score_gap_mean": _mean([a - b for a, b in zip(comparison_score_a, comparison_score_b)]),
        },
        "belonging": {
            "method_a_prefers": belonging_counts["A"],
            "method_b_prefers": belonging_counts["B"],
            "ties_or_invalid": n_cases - belonging_valid,
            "method_a_preference_rate": round(belonging_counts["A"] / n_cases, 4) if n_cases else None,
            "method_b_preference_rate": round(belonging_counts["B"] / n_cases, 4) if n_cases else None,
            "method_a_preference_rate_no_ties": round(belonging_counts["A"] / belonging_valid, 4) if belonging_valid else None,
            "method_b_preference_rate_no_ties": round(belonging_counts["B"] / belonging_valid, 4) if belonging_valid else None,
            "confidence_mean": _mean(belonging_conf),
        },
    }

    if secondary_checks:
        cohesion_a = [case["secondary"]["group_a_cohesion"]["score"] for case in reviewed_cases if case.get("secondary")]
        cohesion_b = [case["secondary"]["group_b_cohesion"]["score"] for case in reviewed_cases if case.get("secondary")]
        outliers_a = [case["secondary"]["group_a_outliers"]["n_outliers"] for case in reviewed_cases if case.get("secondary")]
        outliers_b = [case["secondary"]["group_b_outliers"]["n_outliers"] for case in reviewed_cases if case.get("secondary")]
        summary["secondary"] = {
            "group_a_cohesion_mean": _mean(cohesion_a),
            "group_b_cohesion_mean": _mean(cohesion_b),
            "group_a_outliers_mean": _mean(outliers_a),
            "group_b_outliers_mean": _mean(outliers_b),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="E5: blind A/B review on disagreement boundary nodes")
    parser.add_argument("edge_dir", type=Path, help="Directory with edge parquet files")
    parser.add_argument("abstract_path", type=Path, help="Abstract parquet with uid/title/abstract/pubyear")
    parser.add_argument("--field", type=str, required=True)
    parser.add_argument("--method-a", type=str, default="sum", choices=VALID_METHODS)
    parser.add_argument("--method-b", type=str, default="consensus", choices=VALID_METHODS)
    parser.add_argument("--label-a", type=str, default=None, help="Display label for method A")
    parser.add_argument("--label-b", type=str, default=None, help="Display label for method B")
    parser.add_argument(
        "--layers-a",
        type=str,
        default=None,
        help="Comma-separated subset of layers for method A (default: all discovered layers)",
    )
    parser.add_argument(
        "--layers-b",
        type=str,
        default=None,
        help="Comma-separated subset of layers for method B (default: all discovered layers)",
    )
    parser.add_argument(
        "--exclude-layers-a",
        type=str,
        default=None,
        help="Comma-separated layers to exclude from method A",
    )
    parser.add_argument(
        "--exclude-layers-b",
        type=str,
        default=None,
        help="Comma-separated layers to exclude from method B",
    )
    parser.add_argument(
        "--effective-k",
        type=int,
        default=None,
        help="Global neighbor budget distributed across each method's selected layers",
    )
    parser.add_argument("--target-pct", type=float, default=3.0)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--min-size", type=int, default=10)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--n-cases", type=int, default=24)
    parser.add_argument("--n-neighbors", type=int, default=8)
    parser.add_argument("--boundary-quantile", type=float, default=0.9)
    parser.add_argument("--max-group-jaccard", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-only", action="store_true", help="Only create the disagreement review set")
    parser.add_argument("--secondary-checks", action="store_true", help="Also score cohesion and outliers per method")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("-o", "--output", type=Path, default=Path("results"))
    args = parser.parse_args()

    if args.method_a == args.method_b:
        raise ValueError("method_a and method_b must differ for blind A/B review")

    layers_a_spec = _parse_layer_list(args.layers_a)
    layers_b_spec = _parse_layer_list(args.layers_b)
    exclude_a_spec = _parse_layer_list(args.exclude_layers_a)
    exclude_b_spec = _parse_layer_list(args.exclude_layers_b)

    layers = load_layer_tables(args.edge_dir)
    layers_a = select_layers(layers, include=layers_a_spec, exclude=exclude_a_spec)
    layers_b = select_layers(layers, include=layers_b_spec, exclude=exclude_b_spec)
    if not layers_a:
        raise ValueError("Method A layer selection produced no layers")
    if not layers_b:
        raise ValueError("Method B layer selection produced no layers")
    abstracts = load_abstracts_table(args.abstract_path)
    meta = abstracts_lookup(abstracts)
    reviewable_uids = set(meta)
    label_a = args.label_a or args.method_a
    label_b = args.label_b or args.method_b

    log.info("Field: %s", args.field)
    log.info("Methods: %s vs %s", label_a, label_b)
    log.info("  A layers: %s", ", ".join(sorted(layers_a)))
    log.info("  B layers: %s", ", ".join(sorted(layers_b)))
    budget_mode = "effective_k" if args.effective_k is not None else "top_k"
    protocol = _protocol_name(effective_k=args.effective_k)
    top_k_a = _resolve_top_k(sorted(layers_a), top_k=args.top_k, effective_k=args.effective_k)
    top_k_b = _resolve_top_k(sorted(layers_b), top_k=args.top_k, effective_k=args.effective_k)
    if args.effective_k is not None:
        log.info(
            "Budget: effective_k=%d, min_size=%d, target_pct=%.1f",
            args.effective_k,
            args.min_size,
            args.target_pct,
        )
    else:
        log.info("Budget: top_k=%d, min_size=%d, target_pct=%.1f", args.top_k, args.min_size, args.target_pct)
    log.info("  A top_k: %s", top_k_a)
    log.info("  B top_k: %s", top_k_b)

    run_a = run_combination(
        layers_a,
        strategy=args.method_a,
        target_pct=args.target_pct,
        top_k=top_k_a,
        min_size=args.min_size,
        n_seeds=args.n_seeds,
        compute_stability=False,
        compute_quality=False,
    )
    run_b = run_combination(
        layers_b,
        strategy=args.method_b,
        target_pct=args.target_pct,
        top_k=top_k_b,
        min_size=args.min_size,
        n_seeds=args.n_seeds,
        compute_stability=False,
        compute_quality=False,
    )

    from sciscape.evaluation.sampler import sample_disagreement_cases

    used_boundary_quantile = args.boundary_quantile
    used_max_group_jaccard = args.max_group_jaccard
    candidate_set = None
    quantile_schedule = [args.boundary_quantile, 0.75, 0.5, 0.25, 0.0]
    jaccard_schedule = [args.max_group_jaccard, 0.75, 0.95, 1.0]

    for quantile in quantile_schedule:
        if quantile > args.boundary_quantile:
            continue
        for max_jaccard in jaccard_schedule:
            if max_jaccard < args.max_group_jaccard:
                continue
            candidate_set = sample_disagreement_cases(
                run_a["combined"],
                run_a["membership_map"],
                run_b["combined"],
                run_b["membership_map"],
                method_a=label_a,
                method_b=label_b,
                abstracts=abstracts,
                n_targets=max(args.n_cases * 4, 100),
                n_neighbors=args.n_neighbors,
                min_cluster_size=args.min_size,
                boundary_quantile=quantile,
                max_group_jaccard=max_jaccard,
                allowed_uids=reviewable_uids,
                seed=args.seed,
            )
            used_boundary_quantile = quantile
            used_max_group_jaccard = max_jaccard
            if candidate_set.cases:
                if quantile != args.boundary_quantile or max_jaccard != args.max_group_jaccard:
                    log.info(
                        "Fallback settings produced %d cases: boundary_quantile=%.2f, max_group_jaccard=%.2f",
                        len(candidate_set.cases),
                        quantile,
                        max_jaccard,
                    )
                break
        if candidate_set is not None and candidate_set.cases:
            break

    if candidate_set is None:
        raise RuntimeError("Failed to construct disagreement review candidate set")

    min_group_size = _min_group_size(args.n_neighbors)
    candidate_rows = [
        {
            "target_uid": case.target_uid,
            "target_title": case.target_title,
            "target_year": case.target_year,
            "method_a_cluster_id": case.method_a_cluster_id,
            "method_b_cluster_id": case.method_b_cluster_id,
            "method_a_cluster_size": case.method_a_cluster_size,
            "method_b_cluster_size": case.method_b_cluster_size,
            "method_a_cross_cluster_ratio": case.method_a_cross_cluster_ratio,
            "method_b_cross_cluster_ratio": case.method_b_cross_cluster_ratio,
            "group_a_uids": case.group_a_uids,
            "group_b_uids": case.group_b_uids,
            "overlap_size": case.overlap_size,
            "jaccard": case.jaccard,
        }
        for case in candidate_set.cases
    ]
    reviewable_rows = [
        case for case in candidate_rows
        if _is_reviewable_case(case, meta, min_group_size=min_group_size)
    ]
    log.info(
        "Reviewable disagreement cases: %d / %d",
        len(reviewable_rows),
        len(candidate_rows),
    )
    selected_cases = _select_cases(reviewable_rows, args.n_cases, args.seed)

    output_payload: dict = {
        "field": args.field,
        "edge_dir": str(args.edge_dir),
        "protocol": protocol,
        "abstract_path": str(args.abstract_path),
        "method_a": args.method_a,
        "method_b": args.method_b,
        "label_a": label_a,
        "label_b": label_b,
        "layers_a": sorted(layers_a),
        "layers_b": sorted(layers_b),
        "budget_mode": budget_mode,
        "effective_k": args.effective_k,
        "target_pct": args.target_pct,
        "top_k": args.top_k,
        "top_k_a": top_k_a,
        "top_k_b": top_k_b,
        "min_size": args.min_size,
        "n_seeds": args.n_seeds,
        "n_cases": args.n_cases,
        "n_neighbors": args.n_neighbors,
        "boundary_quantile": args.boundary_quantile,
        "used_boundary_quantile": used_boundary_quantile,
        "max_group_jaccard": args.max_group_jaccard,
        "used_max_group_jaccard": used_max_group_jaccard,
        "sample_only": args.sample_only,
        "secondary_checks": args.secondary_checks,
        "method_a_run": {
            "n_edges": run_a["combined"].height,
            "gamma": run_a["gamma_result"].gamma,
            "n_clusters": run_a["gamma_result"].n_clusters,
            "max_pct": run_a["gamma_result"].max_pct,
        },
        "method_b_run": {
            "n_edges": run_b["combined"].height,
            "gamma": run_b["gamma_result"].gamma,
            "n_clusters": run_b["gamma_result"].n_clusters,
            "max_pct": run_b["gamma_result"].max_pct,
        },
        "n_candidate_cases": candidate_set.n_candidates,
        "n_reviewable_cases": len(reviewable_rows),
        "selected_cases": selected_cases,
    }

    if args.sample_only:
        out_path = args.output / f"{args.field}_boundary_review.json"
        save_json(output_payload, out_path)
        log.info("\nSaved sample set → %s", out_path)
        return

    from sciscape.clustering.cluster_naming import create_client
    from sciscape.evaluation.reviewer import (
        review_belonging,
        review_comparison,
        review_group_cohesion,
        review_outliers,
    )

    random.seed(args.seed)
    client = create_client(model=args.model)
    reviewed_cases: list[dict] = []

    for idx, case in enumerate(selected_cases, start=1):
        target_docs = _docs_from_uids([case["target_uid"]], meta)
        group_a_docs = _docs_from_uids(case["group_a_uids"], meta)
        group_b_docs = _docs_from_uids(case["group_b_uids"], meta)
        if not target_docs:
            continue
        target_doc = target_docs[0]
        if len(group_a_docs) < min_group_size or len(group_b_docs) < min_group_size:
            continue

        log.info(
            "[%d/%d] uid=%s jaccard=%.2f %s_size=%d %s_size=%d",
            idx,
            len(selected_cases),
            case["target_uid"],
            case["jaccard"],
            args.method_a,
            case["method_a_cluster_size"],
            args.method_b,
            case["method_b_cluster_size"],
        )

        comparison = review_comparison(
            client,
            target_doc,
            group_a_docs,
            group_b_docs,
            method_a=label_a,
            method_b=label_b,
            model=args.model,
        )
        belonging = review_belonging(
            client,
            target_doc,
            group_a_docs,
            group_b_docs,
            method_a=label_a,
            method_b=label_b,
            model=args.model,
        )

        record = dict(case)
        record["comparison"] = {
            "winner": comparison.winner,
            "score_a": comparison.score_a,
            "score_b": comparison.score_b,
            "reasoning": comparison.reasoning,
            "presented_winner": comparison.presented_winner,
            "presented_score_a": comparison.presented_score_a,
            "presented_score_b": comparison.presented_score_b,
            "presented_method_a": comparison.presented_method_a,
            "presented_method_b": comparison.presented_method_b,
            "swapped": comparison.swapped,
        }
        record["belonging"] = {
            "belongs_to": belonging.belongs_to,
            "confidence": belonging.confidence,
            "reasoning": belonging.reasoning,
            "presented_belongs_to": belonging.presented_belongs_to,
            "presented_method_a": belonging.presented_method_a,
            "presented_method_b": belonging.presented_method_b,
            "swapped": belonging.swapped,
        }

        if args.secondary_checks:
            group_a_cohesion = review_group_cohesion(
                client,
                group_a_docs,
                method=label_a,
                model=args.model,
            )
            group_b_cohesion = review_group_cohesion(
                client,
                group_b_docs,
                method=label_b,
                model=args.model,
            )
            group_a_outliers = review_outliers(
                client,
                target_doc,
                group_a_docs,
                method=label_a,
                model=args.model,
            )
            group_b_outliers = review_outliers(
                client,
                target_doc,
                group_b_docs,
                method=label_b,
                model=args.model,
            )
            record["secondary"] = {
                "group_a_cohesion": {
                    "score": group_a_cohesion.cohesion_score,
                    "theme": group_a_cohesion.theme,
                    "n_outliers": group_a_cohesion.n_outliers,
                    "reasoning": group_a_cohesion.reasoning,
                },
                "group_b_cohesion": {
                    "score": group_b_cohesion.cohesion_score,
                    "theme": group_b_cohesion.theme,
                    "n_outliers": group_b_cohesion.n_outliers,
                    "reasoning": group_b_cohesion.reasoning,
                },
                "group_a_outliers": {
                    "n_outliers": group_a_outliers.n_outliers,
                    "cluster_theme": group_a_outliers.cluster_theme,
                    "reasoning": group_a_outliers.reasoning,
                },
                "group_b_outliers": {
                    "n_outliers": group_b_outliers.n_outliers,
                    "cluster_theme": group_b_outliers.cluster_theme,
                    "reasoning": group_b_outliers.reasoning,
                },
            }

        reviewed_cases.append(record)

    output_payload["reviewed_cases"] = reviewed_cases
    output_payload["summary"] = _summarize_reviews(
        reviewed_cases,
        method_a=label_a,
        method_b=label_b,
        secondary_checks=args.secondary_checks,
    )

    out_path = args.output / f"{args.field}_boundary_review.json"
    save_json(output_payload, out_path)
    log.info("\nSaved → %s", out_path)


if __name__ == "__main__":
    main()
