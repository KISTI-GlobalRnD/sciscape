"""Control analysis: review low-shift local neighborhoods where methods nearly agree."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any
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

def _mean(values: list[float | int]) -> float | None:
    if not values:
        return None
    return round(float(np.mean(values)), 4)

def _serialize_case(case: dict[str, Any]) -> dict[str, Any]:
    return dict(case)

def _summarize_reviews(reviewed_cases: list[dict], *, method_a: str, method_b: str) -> dict[str, Any]:
    wins_a = sum(1 for case in reviewed_cases if case["comparison"]["winner"] == "A")
    wins_b = sum(1 for case in reviewed_cases if case["comparison"]["winner"] == "B")
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
    n_cases = len(reviewed_cases)
    return {
        "n_reviewed_cases": n_cases,
        "comparison": {
            "method_a": method_a,
            "method_b": method_b,
            "method_a_wins": wins_a,
            "method_b_wins": wins_b,
            "ties_or_invalid": n_cases - wins_a - wins_b,
            "method_a_win_rate": round(wins_a / n_cases, 4) if n_cases else None,
            "method_b_win_rate": round(wins_b / n_cases, 4) if n_cases else None,
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

def _collect_null_cases(
    edges_a,
    membership_a,
    edges_b,
    membership_b,
    *,
    abstracts,
    n_neighbors: int,
    min_cluster_size: int,
    min_rank_jaccard: float,
    min_shift_score: float,
    max_shift_score: float,
    min_cluster_overlap: float,
    require_same_cluster: bool,
    allowed_uids: set[str] | None,
) -> tuple[list[dict[str, Any]], int]:
    from sciscape.evaluation.sampler import (
        _cluster_overlap_coefficient,
        _compute_all_neighbors,
        _compute_method_stats,
        _top_neighbors,
    )

    stats_a = _compute_method_stats(edges_a, membership_a)
    stats_b = _compute_method_stats(edges_b, membership_b)
    all_neighbors_a = _compute_all_neighbors(edges_a)
    all_neighbors_b = _compute_all_neighbors(edges_b)
    meta = abstracts_lookup(abstracts)

    ordered_candidates = sorted(set(membership_a) & set(membership_b))
    eligible = [
        uid
        for uid in ordered_candidates
        if allowed_uids is None or uid in allowed_uids
        if stats_a.cluster_sizes[membership_a[uid]] >= min_cluster_size
        and stats_b.cluster_sizes[membership_b[uid]] >= min_cluster_size
    ]
    candidate_rows: list[dict[str, Any]] = []
    for uid in eligible:
        neighbors_a = _top_neighbors(
            uid,
            all_neighbors_a,
            membership_a,
            n_neighbors=n_neighbors,
            allowed_uids=allowed_uids,
        )
        neighbors_b = _top_neighbors(
            uid,
            all_neighbors_b,
            membership_b,
            n_neighbors=n_neighbors,
            allowed_uids=allowed_uids,
        )
        if len(neighbors_a) < n_neighbors or len(neighbors_b) < n_neighbors:
            continue
        ranks_a = {row["uid"]: row["rank"] for row in neighbors_a}
        ranks_b = {row["uid"]: row["rank"] for row in neighbors_b}
        set_a = set(ranks_a)
        set_b = set(ranks_b)
        union = set_a | set_b
        overlap = set_a & set_b
        rank_jaccard = len(overlap) / len(union) if union else 1.0
        shared_neighbors: list[dict[str, Any]] = []
        abs_rank_shifts: list[int] = []
        for nbr in sorted(overlap):
            delta = int(ranks_b[nbr] - ranks_a[nbr])
            abs_rank_shifts.append(abs(delta))
            row_a = next(row for row in neighbors_a if row["uid"] == nbr)
            row_b = next(row for row in neighbors_b if row["uid"] == nbr)
            shared_neighbors.append(
                {
                    "uid": nbr,
                    "rank_a": row_a["rank"],
                    "rank_b": row_b["rank"],
                    "delta": delta,
                    "weight_a": row_a["weight"],
                    "weight_b": row_b["weight"],
                }
            )
        mean_abs_rank_shift = (
            round(float(np.mean(abs_rank_shifts)), 4) if abs_rank_shifts else float(n_neighbors)
        )
        max_abs_rank_shift = max(abs_rank_shifts) if abs_rank_shifts else n_neighbors
        cluster_overlap_coeff = round(
            _cluster_overlap_coefficient(uid, membership_a, stats_a, membership_b, stats_b), 4
        )
        cluster_changed = cluster_overlap_coeff < min_cluster_overlap
        shift_score = round(
            (1.0 - rank_jaccard)
            + (mean_abs_rank_shift / max(1, n_neighbors))
            + (0.25 if cluster_changed else 0.0),
            4,
        )

        if rank_jaccard < min_rank_jaccard:
            continue
        if shift_score < min_shift_score or shift_score > max_shift_score:
            continue
        if require_same_cluster and cluster_changed:
            continue

        record = meta.get(uid, {})
        candidate_rows.append(
            {
                "target_uid": uid,
                "target_title": record.get("title", ""),
                "target_year": record.get("pubyear"),
                "method_a_cluster_id": membership_a[uid],
                "method_b_cluster_id": membership_b[uid],
                "method_a_cluster_size": stats_a.cluster_sizes[membership_a[uid]],
                "method_b_cluster_size": stats_b.cluster_sizes[membership_b[uid]],
                "rank_jaccard": round(rank_jaccard, 4),
                "overlap_size": len(overlap),
                "mean_abs_rank_shift": mean_abs_rank_shift,
                "max_abs_rank_shift": max_abs_rank_shift,
                "cluster_overlap_coeff": cluster_overlap_coeff,
                "cluster_changed": cluster_changed,
                "shift_score": shift_score,
                "neighbors_a": neighbors_a,
                "neighbors_b": neighbors_b,
                "shared_neighbors": shared_neighbors,
                "neighbors_only_a": [row for row in neighbors_a if row["uid"] not in set_b],
                "neighbors_only_b": [row for row in neighbors_b if row["uid"] not in set_a],
            }
        )

    candidate_rows.sort(
        key=lambda row: (
            row["shift_score"],
            -row["rank_jaccard"],
            row["target_uid"],
        )
    )
    return candidate_rows, len(eligible)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("edge_dir", type=Path, help="Directory with edge parquet files")
    parser.add_argument("abstract_path", type=Path, help="Abstract parquet with uid/title/abstract/pubyear")
    parser.add_argument("--field", type=str, required=True)
    parser.add_argument("--method-a", type=str, default="sum", choices=VALID_METHODS)
    parser.add_argument("--method-b", type=str, default="consensus", choices=VALID_METHODS)
    parser.add_argument("--label-a", type=str, default=None)
    parser.add_argument("--label-b", type=str, default=None)
    parser.add_argument("--layers-a", type=str, default=None)
    parser.add_argument("--layers-b", type=str, default=None)
    parser.add_argument("--exclude-layers-a", type=str, default=None)
    parser.add_argument("--exclude-layers-b", type=str, default=None)
    parser.add_argument("--target-pct", type=float, default=3.0)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--min-size", type=int, default=10)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--n-cases", type=int, default=16)
    parser.add_argument("--n-neighbors", type=int, default=8)
    parser.add_argument("--min-rank-jaccard", type=float, default=0.85)
    parser.add_argument("--min-shift-score", type=float, default=0.05)
    parser.add_argument("--max-shift-score", type=float, default=0.45)
    parser.add_argument("--min-cluster-overlap", type=float, default=0.5)
    parser.add_argument("--allow-cluster-change", action="store_true")
    parser.add_argument("--gamma-a", type=float, default=None)
    parser.add_argument("--gamma-b", type=float, default=None)
    parser.add_argument("--gamma-cache", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--input-json",
        type=Path,
        default=None,
        help="Reuse selected_cases from an existing null-control JSON instead of recomputing them",
    )
    parser.add_argument("--sample-only", action="store_true")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("-o", "--output", type=Path, default=Path("research/consensus/results/controls"))
    args = parser.parse_args()

    layers_a_spec = _parse_layer_list(args.layers_a)
    layers_b_spec = _parse_layer_list(args.layers_b)
    exclude_a_spec = _parse_layer_list(args.exclude_layers_a)
    exclude_b_spec = _parse_layer_list(args.exclude_layers_b)

    label_a = args.label_a or args.method_a
    label_b = args.label_b or args.method_b
    abstracts = load_abstracts_table(args.abstract_path)
    reviewable_uids = set(abstracts["uid"].to_list())

    if args.input_json is not None:
        payload = json.loads(args.input_json.read_text(encoding="utf-8"))
        label_a = str(payload.get("label_a", label_a))
        label_b = str(payload.get("label_b", label_b))
        selected_cases = [_serialize_case(case) for case in payload.get("selected_cases", [])[: args.n_cases]]
        payload["field"] = payload.get("field", args.field)
        payload["edge_dir"] = payload.get("edge_dir", str(args.edge_dir))
        payload["abstract_path"] = str(args.abstract_path)
        payload["method_a"] = payload.get("method_a", args.method_a)
        payload["method_b"] = payload.get("method_b", args.method_b)
        payload["label_a"] = label_a
        payload["label_b"] = label_b
        payload["n_cases"] = args.n_cases
        payload["n_neighbors"] = args.n_neighbors
        payload["sample_only"] = args.sample_only
        payload["selected_cases"] = selected_cases
    else:
        layers = load_layer_tables(args.edge_dir)
        layers_a = select_layers(layers, include=layers_a_spec, exclude=exclude_a_spec)
        layers_b = select_layers(layers, include=layers_b_spec, exclude=exclude_b_spec)
        if not layers_a or not layers_b:
            raise ValueError("Layer selection produced no layers")

        gamma_a = args.gamma_a or resolve_cached_gamma(
            args.gamma_cache,
            edge_dir=args.edge_dir,
            strategy=args.method_a,
            layer_names=sorted(layers_a),
            top_k=args.top_k,
            target_pct=args.target_pct,
            min_size=args.min_size,
        )
        gamma_b = args.gamma_b or resolve_cached_gamma(
            args.gamma_cache,
            edge_dir=args.edge_dir,
            strategy=args.method_b,
            layer_names=sorted(layers_b),
            top_k=args.top_k,
            target_pct=args.target_pct,
            min_size=args.min_size,
        )

        run_a = run_combination(
            layers_a,
            strategy=args.method_a,
            target_pct=args.target_pct,
            top_k=args.top_k,
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
            top_k=args.top_k,
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
            top_k=args.top_k,
            target_pct=args.target_pct,
            min_size=args.min_size,
            gamma_result=run_a["gamma_result"],
            extra={"field": args.field, "label": label_a},
        )
        update_gamma_cache(
            args.gamma_cache,
            edge_dir=args.edge_dir,
            strategy=args.method_b,
            layer_names=sorted(layers_b),
            top_k=args.top_k,
            target_pct=args.target_pct,
            min_size=args.min_size,
            gamma_result=run_b["gamma_result"],
            extra={"field": args.field, "label": label_b},
        )

        candidate_rows, n_eligible = _collect_null_cases(
            run_a["combined"],
            run_a["membership_map"],
            run_b["combined"],
            run_b["membership_map"],
            abstracts=abstracts,
            n_neighbors=args.n_neighbors,
            min_cluster_size=args.min_size,
            min_rank_jaccard=args.min_rank_jaccard,
            min_shift_score=args.min_shift_score,
            max_shift_score=args.max_shift_score,
            min_cluster_overlap=args.min_cluster_overlap,
            require_same_cluster=not args.allow_cluster_change,
            allowed_uids=reviewable_uids,
        )
        selected_cases = [_serialize_case(case) for case in candidate_rows[: args.n_cases]]

        payload = {
            "field": args.field,
            "edge_dir": str(args.edge_dir),
            "abstract_path": str(args.abstract_path),
            "method_a": args.method_a,
            "method_b": args.method_b,
            "label_a": label_a,
            "label_b": label_b,
            "layers_a": sorted(layers_a),
            "layers_b": sorted(layers_b),
            "target_pct": args.target_pct,
            "top_k": args.top_k,
            "min_size": args.min_size,
            "n_seeds": args.n_seeds,
            "n_cases": args.n_cases,
            "n_neighbors": args.n_neighbors,
            "min_rank_jaccard": args.min_rank_jaccard,
            "min_shift_score": args.min_shift_score,
            "max_shift_score": args.max_shift_score,
            "min_cluster_overlap": args.min_cluster_overlap,
            "allow_cluster_change": args.allow_cluster_change,
            "sample_only": args.sample_only,
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
            "n_eligible_nodes": n_eligible,
            "n_candidate_cases": len(candidate_rows),
            "selected_cases": selected_cases,
        }

    if args.sample_only:
        out_path = args.output / f"{args.field}_null_control.json"
        save_json(payload, out_path)
        log.info("Saved → %s", out_path)
        return

    from sciscape.clustering.cluster_naming import create_client
    from sciscape.evaluation.reviewer import review_neighbor_rerank

    meta = abstracts_lookup(abstracts)
    client = create_client(model=args.model)
    reviewed_cases: list[dict[str, Any]] = []
    for idx, case in enumerate(selected_cases, start=1):
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
            "[%d/%d] uid=%s shift=%.3f jaccard=%.2f cluster_changed=%s",
            idx,
            len(selected_cases),
            case["target_uid"],
            case["shift_score"],
            case["rank_jaccard"],
            case["cluster_changed"],
        )
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
        reviewed_cases.append(record)

    payload["reviewed_cases"] = reviewed_cases
    payload["summary"] = _summarize_reviews(reviewed_cases, method_a=label_a, method_b=label_b)

    out_path = args.output / f"{args.field}_null_control.json"
    save_json(payload, out_path)
    log.info("Saved → %s", out_path)

if __name__ == "__main__":
    main()
