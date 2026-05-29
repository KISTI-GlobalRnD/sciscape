"""E6: blind A/B review on rank-shifted local neighborhoods."""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Iterable
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


import numpy as np

from _common import (
    allocate_effective_k,
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

def _parse_layer_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None

def _docs_from_ranked_neighbors(rows: list[dict], meta: dict[str, dict]) -> list[dict]:
    docs = []
    for row in rows:
        record = meta.get(row["uid"])
        if not record:
            continue
        docs.append(
            {
                "uid": row["uid"],
                "rank": row["rank"],
                "weight": row.get("weight"),
                "title": record.get("title", "") or "",
                "abstract": record.get("abstract", "") or "",
                "pubyear": record.get("pubyear"),
            }
        )
    return docs

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

def _serialize_case(case) -> dict:
    return {
        "target_uid": case.target_uid,
        "target_title": case.target_title,
        "target_year": case.target_year,
        "method_a_cluster_id": case.method_a_cluster_id,
        "method_b_cluster_id": case.method_b_cluster_id,
        "method_a_cluster_size": case.method_a_cluster_size,
        "method_b_cluster_size": case.method_b_cluster_size,
        "rank_jaccard": case.rank_jaccard,
        "overlap_size": case.overlap_size,
        "mean_abs_rank_shift": case.mean_abs_rank_shift,
        "max_abs_rank_shift": case.max_abs_rank_shift,
        "cluster_overlap_coeff": case.cluster_overlap_coeff,
        "cluster_changed": case.cluster_changed,
        "shift_score": case.shift_score,
        "neighbors_a": case.neighbors_a,
        "neighbors_b": case.neighbors_b,
        "shared_neighbors": case.shared_neighbors,
        "neighbors_only_a": case.neighbors_only_a,
        "neighbors_only_b": case.neighbors_only_b,
    }

def _serialize_comparison(comparison) -> dict:
    payload = {
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
    if getattr(comparison, "order_balance_mode", "single_pass") != "single_pass":
        payload["order_balance_mode"] = comparison.order_balance_mode
        payload["order_sensitive"] = getattr(comparison, "order_sensitive", False)
        payload["balanced_passes"] = getattr(comparison, "balanced_passes", [])
    return payload

def _selected_target_uids(cases: list[dict]) -> list[str]:
    return [case["target_uid"] for case in cases]

def _resume_compatible(existing_payload: dict, current_payload: dict) -> bool:
    keys = (
        "field",
        "edge_dir",
        "abstract_path",
        "method_a",
        "method_b",
        "label_a",
        "label_b",
        "layers_a",
        "layers_b",
        "budget_mode",
        "effective_k",
        "top_k",
        "top_k_a",
        "top_k_b",
        "n_cases",
        "n_neighbors",
        "case_bank",
        "strict_case_bank",
        "order_balanced",
    )
    if any(existing_payload.get(key) != current_payload.get(key) for key in keys):
        return False
    return _selected_target_uids(existing_payload.get("selected_cases", [])) == _selected_target_uids(
        current_payload.get("selected_cases", [])
    )

def _resolve_output_path(output_arg: Path, field: str) -> Path:
    if output_arg.suffix == ".json":
        return output_arg
    return output_arg / f"{field}_rank_shift_review.json"

def _resolve_top_k(layer_names: list[str], *, top_k: int, effective_k: int | None) -> int | dict[str, int]:
    if effective_k is None:
        return top_k
    if len(layer_names) <= 1:
        return effective_k
    return allocate_effective_k(layer_names, effective_k)

def _protocol_name(*, effective_k: int | None) -> str:
    return "candidate_budget_matched" if effective_k is not None else "practical_top_k"

def _summarize_reviews(reviewed_cases: list[dict], *, method_a: str, method_b: str) -> dict:
    votes = [case["comparison"]["winner"] for case in reviewed_cases]
    counts = _aggregate_counts(votes, positive="A", negative="B")
    n_cases = len(reviewed_cases)
    valid = counts["total"]
    scores_a = [case["comparison"]["score_a"] for case in reviewed_cases]
    scores_b = [case["comparison"]["score_b"] for case in reviewed_cases]
    shift_scores = [case["shift_score"] for case in reviewed_cases]
    rank_jaccards = [case["rank_jaccard"] for case in reviewed_cases]
    mean_shifts = [case["mean_abs_rank_shift"] for case in reviewed_cases]
    cluster_overlaps = [
        case["cluster_overlap_coeff"]
        for case in reviewed_cases
        if case.get("cluster_overlap_coeff") is not None
    ]
    cluster_changed = [1 if case["cluster_changed"] else 0 for case in reviewed_cases]
    return {
        "n_reviewed_cases": n_cases,
        "comparison": {
            "method_a": method_a,
            "method_b": method_b,
            "method_a_wins": counts["A"],
            "method_b_wins": counts["B"],
            "ties_or_invalid": n_cases - valid,
            "method_a_win_rate": round(counts["A"] / n_cases, 4) if n_cases else None,
            "method_b_win_rate": round(counts["B"] / n_cases, 4) if n_cases else None,
            "method_a_win_rate_no_ties": round(counts["A"] / valid, 4) if valid else None,
            "method_b_win_rate_no_ties": round(counts["B"] / valid, 4) if valid else None,
            "score_a_mean": _mean(scores_a),
            "score_b_mean": _mean(scores_b),
            "score_gap_mean": _mean([a - b for a, b in zip(scores_a, scores_b)]),
        },
        "sample_stats": {
            "shift_score_mean": _mean(shift_scores),
            "rank_jaccard_mean": _mean(rank_jaccards),
            "mean_abs_rank_shift_mean": _mean(mean_shifts),
            "cluster_overlap_coeff_mean": _mean(cluster_overlaps),
            "cluster_changed_rate": _mean(cluster_changed),
        },
    }

def main() -> None:
    parser = argparse.ArgumentParser(description="E6: blind A/B review on rank-shifted local neighborhoods")
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
    parser.add_argument("--max-rank-jaccard", type=float, default=0.85)
    parser.add_argument("--min-cluster-overlap", type=float, default=0.5)
    parser.add_argument("--gamma-a", type=float, default=None, help="Optional fixed gamma for method A")
    parser.add_argument("--gamma-b", type=float, default=None, help="Optional fixed gamma for method B")
    parser.add_argument("--gamma-cache", type=Path, default=None, help="Optional gamma cache JSON")
    parser.add_argument("--case-bank", type=Path, default=None, help="Optional common case bank JSON")
    parser.add_argument(
        "--strict-case-bank",
        action="store_true",
        help="Only review target UIDs present in the supplied case bank",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-only", action="store_true", help="Only create the rank-shift review set")
    parser.add_argument(
        "--order-balanced",
        action="store_true",
        help="Review each case in both A->B and B->A order, marking disagreements as TIE",
    )
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("-o", "--output", type=Path, default=Path("results"))
    args = parser.parse_args()

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

    label_a = args.label_a or args.method_a
    label_b = args.label_b or args.method_b
    if (
        args.method_a == args.method_b
        and sorted(layers_a) == sorted(layers_b)
        and args.gamma_a == args.gamma_b
        and label_a == label_b
    ):
        raise ValueError("method A and B configurations are identical")

    abstracts = load_abstracts_table(args.abstract_path)
    meta = abstracts_lookup(abstracts)
    reviewable_uids = set(meta)

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

    gamma_a = args.gamma_a
    gamma_b = args.gamma_b
    if gamma_a is None:
        gamma_a = resolve_cached_gamma(
            args.gamma_cache,
            edge_dir=args.edge_dir,
            strategy=args.method_a,
            layer_names=sorted(layers_a),
            top_k=top_k_a,
            target_pct=args.target_pct,
            min_size=args.min_size,
            protocol=protocol,
        )
        if gamma_a is not None:
            log.info("  cached gamma A: %.9g", gamma_a)
    if gamma_b is None:
        gamma_b = resolve_cached_gamma(
            args.gamma_cache,
            edge_dir=args.edge_dir,
            strategy=args.method_b,
            layer_names=sorted(layers_b),
            top_k=top_k_b,
            target_pct=args.target_pct,
            min_size=args.min_size,
            protocol=protocol,
        )
        if gamma_b is not None:
            log.info("  cached gamma B: %.9g", gamma_b)

    run_a = run_combination(
        layers_a,
        strategy=args.method_a,
        target_pct=args.target_pct,
        top_k=top_k_a,
        min_size=args.min_size,
        n_seeds=args.n_seeds,
        gamma=gamma_a,
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
        gamma=gamma_b,
        compute_stability=False,
        compute_quality=False,
    )

    update_gamma_cache(
        args.gamma_cache,
        edge_dir=args.edge_dir,
        strategy=args.method_a,
        layer_names=sorted(layers_a),
        top_k=top_k_a,
        target_pct=args.target_pct,
        min_size=args.min_size,
        gamma_result=run_a["gamma_result"],
        extra={"field": args.field, "label": label_a},
        protocol=protocol,
    )
    update_gamma_cache(
        args.gamma_cache,
        edge_dir=args.edge_dir,
        strategy=args.method_b,
        layer_names=sorted(layers_b),
        top_k=top_k_b,
        target_pct=args.target_pct,
        min_size=args.min_size,
        gamma_result=run_b["gamma_result"],
        extra={"field": args.field, "label": label_b},
        protocol=protocol,
    )

    from sciscape.evaluation.sampler import sample_rank_shift_cases

    case_bank_target_uids = None
    if args.case_bank is not None:
        bank_payload = json.loads(args.case_bank.read_text(encoding="utf-8"))
        case_bank_target_uids = list(bank_payload.get("target_uids", []))
        log.info("Using case bank: %s (%d target_uids)", args.case_bank, len(case_bank_target_uids))

    sample_set = sample_rank_shift_cases(
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
        max_rank_jaccard=args.max_rank_jaccard,
        min_cluster_overlap=args.min_cluster_overlap,
        allowed_uids=reviewable_uids,
        target_uids=case_bank_target_uids,
        strict_target_uids=args.strict_case_bank,
        seed=args.seed,
    )
    selected_cases = [_serialize_case(case) for case in sample_set.cases[: args.n_cases]]

    output_payload = {
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
        "max_rank_jaccard": args.max_rank_jaccard,
        "min_cluster_overlap": args.min_cluster_overlap,
        "gamma_a_override": args.gamma_a,
        "gamma_b_override": args.gamma_b,
        "gamma_cache": str(args.gamma_cache) if args.gamma_cache is not None else None,
        "case_bank": str(args.case_bank) if args.case_bank is not None else None,
        "strict_case_bank": args.strict_case_bank,
        "sample_only": args.sample_only,
        "order_balanced": args.order_balanced,
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
        "n_candidate_cases": sample_set.n_candidates,
        "selected_cases": selected_cases,
    }

    out_path = _resolve_output_path(args.output, args.field)

    if args.sample_only:
        save_json(output_payload, out_path)
        log.info("\nSaved sample set → %s", out_path)
        return

    from sciscape.clustering.cluster_naming import create_client
    from sciscape.evaluation.reviewer import (
        review_neighbor_rerank,
        review_neighbor_rerank_order_balanced,
    )

    client = create_client(model=args.model)
    reviewed_cases: list[dict] = []
    reviewed_case_uids: set[str] = set()
    if out_path.exists():
        existing_payload = json.loads(out_path.read_text(encoding="utf-8"))
        if _resume_compatible(existing_payload, output_payload):
            existing_by_uid = {
                record["target_uid"]: record
                for record in existing_payload.get("reviewed_cases", [])
                if record.get("target_uid")
            }
            reviewed_cases = [
                existing_by_uid[case["target_uid"]]
                for case in selected_cases
                if case["target_uid"] in existing_by_uid
            ]
            reviewed_case_uids = {record["target_uid"] for record in reviewed_cases}
            if reviewed_cases:
                log.info("Resuming from %s (%d completed cases)", out_path, len(reviewed_cases))
        else:
            log.warning("Existing output is incompatible with current settings; starting fresh: %s", out_path)

    output_payload["reviewed_cases"] = reviewed_cases
    output_payload["summary"] = _summarize_reviews(
        reviewed_cases,
        method_a=label_a,
        method_b=label_b,
    )
    save_json(output_payload, out_path)

    for idx, case in enumerate(selected_cases, start=1):
        if case["target_uid"] in reviewed_case_uids:
            log.info("[%d/%d] uid=%s already reviewed; skipping", idx, len(selected_cases), case["target_uid"])
            continue
        target_record = meta.get(case["target_uid"])
        if not target_record:
            continue
        target_doc = {
            "uid": case["target_uid"],
            "title": target_record.get("title", "") or "",
            "abstract": target_record.get("abstract", "") or "",
            "pubyear": target_record.get("pubyear"),
        }
        group_a_docs = _docs_from_ranked_neighbors(case["neighbors_a"], meta)
        group_b_docs = _docs_from_ranked_neighbors(case["neighbors_b"], meta)
        if len(group_a_docs) < args.n_neighbors or len(group_b_docs) < args.n_neighbors:
            continue

        log.info(
            "[%d/%d] uid=%s shift=%.3f jaccard=%.2f mean_rank_shift=%.2f",
            idx,
            len(selected_cases),
            case["target_uid"],
            case["shift_score"],
            case["rank_jaccard"],
            case["mean_abs_rank_shift"],
        )
        if args.order_balanced:
            comparison = review_neighbor_rerank_order_balanced(
                client,
                target_doc,
                group_a_docs,
                group_b_docs,
                method_a=label_a,
                method_b=label_b,
                model=args.model,
            )
        else:
            comparison = review_neighbor_rerank(
                client,
                target_doc,
                group_a_docs,
                group_b_docs,
                method_a=label_a,
                method_b=label_b,
                model=args.model,
            )
        record = dict(case)
        record["comparison"] = _serialize_comparison(comparison)
        reviewed_cases.append(record)
        reviewed_case_uids.add(case["target_uid"])
        output_payload["reviewed_cases"] = reviewed_cases
        output_payload["summary"] = _summarize_reviews(
            reviewed_cases,
            method_a=label_a,
            method_b=label_b,
        )
        save_json(output_payload, out_path)

    output_payload["reviewed_cases"] = reviewed_cases
    output_payload["summary"] = _summarize_reviews(
        reviewed_cases,
        method_a=label_a,
        method_b=label_b,
    )

    save_json(output_payload, out_path)
    log.info("\nSaved → %s", out_path)

if __name__ == "__main__":
    main()
