"""E5: Boundary-node blind review stratified by consensus level."""

from __future__ import annotations

import argparse
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import polars as pl

from _common import abstracts_lookup, load_abstracts_table, load_layer_tables, run_combination, save_json

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def _compute_node_consensus_levels(layers: dict[str, pl.DataFrame], *, top_k: int) -> dict[str, int]:
    from sciscape.linkage.filters import filter_top_k

    tagged_parts: list[pl.DataFrame] = []
    for df in layers.values():
        current = filter_top_k(df, top_k) if top_k > 0 else df
        tagged_parts.append(
            current.select(
                pl.min_horizontal("uid1", "uid2").alias("_lo"),
                pl.max_horizontal("uid1", "uid2").alias("_hi"),
            )
        )

    if not tagged_parts:
        return {}

    pair_counts = pl.concat(tagged_parts).group_by(["_lo", "_hi"]).len(name="n_layers")
    incident = pl.concat(
        [
            pair_counts.select(pl.col("_lo").alias("uid"), "n_layers"),
            pair_counts.select(pl.col("_hi").alias("uid"), "n_layers"),
        ]
    ).group_by("uid").agg(
        pl.col("n_layers").mean().alias("mean_consensus"),
        pl.col("n_layers").max().alias("max_consensus"),
    )

    n_layers_total = max(1, len(layers))
    levels: dict[str, int] = {}
    for row in incident.iter_rows(named=True):
        level = int(round(float(row["mean_consensus"])))
        levels[row["uid"]] = max(1, min(n_layers_total, level))
    return levels


def _build_adjacency(edges: pl.DataFrame) -> dict[str, list[tuple[str, float]]]:
    adjacency: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for row in edges.iter_rows(named=True):
        w = float(row.get("rel_sum2", 1.0))
        u1 = row["uid1"]
        u2 = row["uid2"]
        adjacency[u1].append((u2, w))
        adjacency[u2].append((u1, w))
    for uid in adjacency:
        adjacency[uid].sort(key=lambda item: item[1], reverse=True)
    return dict(adjacency)


def _unique_keep_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _build_same_cluster_group(
    uid: str,
    adjacency: dict[str, list[tuple[str, float]]],
    membership: dict[str, int],
    *,
    preferred: list[str],
    n_neighbors: int,
) -> list[str]:
    target_cluster = membership[uid]
    group = [nbr for nbr in preferred if membership.get(nbr) == target_cluster]
    if len(group) < n_neighbors:
        for nbr, _weight in adjacency.get(uid, []):
            if membership.get(nbr) == target_cluster and nbr not in group:
                group.append(nbr)
            if len(group) >= n_neighbors:
                break
    return group[:n_neighbors]


def _build_contrast_group(
    uid: str,
    adjacency: dict[str, list[tuple[str, float]]],
    membership: dict[str, int],
    *,
    n_neighbors: int,
) -> list[str]:
    target_cluster = membership[uid]
    cluster_scores: dict[int, float] = defaultdict(float)
    cluster_members: dict[int, list[tuple[str, float]]] = defaultdict(list)
    for nbr, weight in adjacency.get(uid, []):
        nbr_cluster = membership.get(nbr)
        if nbr_cluster is None or nbr_cluster == target_cluster:
            continue
        cluster_scores[nbr_cluster] += weight
        cluster_members[nbr_cluster].append((nbr, weight))

    if not cluster_scores:
        return []

    contrast_cluster = max(cluster_scores, key=cluster_scores.get)
    members = sorted(cluster_members[contrast_cluster], key=lambda item: item[1], reverse=True)
    return [nbr for nbr, _weight in members[:n_neighbors]]


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


def main() -> None:
    parser = argparse.ArgumentParser(description="E5: boundary-node blind review")
    parser.add_argument("edge_dir", type=Path, help="Directory with edge parquet files")
    parser.add_argument("abstract_path", type=Path, help="Abstract parquet with uid/title/abstract/pubyear")
    parser.add_argument("--field", type=str, required=True)
    parser.add_argument("--target-pct", type=float, default=3.0)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--min-size", type=int, default=10)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--samples-per-level", type=int, default=10)
    parser.add_argument("--n-neighbors", type=int, default=8)
    parser.add_argument("--boundary-quantile", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-only", action="store_true", help="Only create the stratified review set")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("-o", "--output", type=Path, default=Path("results"))
    args = parser.parse_args()

    layers = load_layer_tables(args.edge_dir)
    abstracts = load_abstracts_table(args.abstract_path)
    meta = abstracts_lookup(abstracts)

    run = run_combination(
        layers,
        strategy="consensus",
        target_pct=args.target_pct,
        top_k=args.top_k,
        min_size=args.min_size,
        n_seeds=args.n_seeds,
    )

    from sciscape.evaluation.sampler import sample_worst_case

    node_levels = _compute_node_consensus_levels(layers, top_k=args.top_k)
    candidate_set = None
    used_boundary_quantile = args.boundary_quantile
    quantile_schedule = [args.boundary_quantile, 0.75, 0.5, 0.25, 0.0]
    seen_quantiles: set[float] = set()
    for quantile in quantile_schedule:
        if quantile in seen_quantiles or quantile > args.boundary_quantile:
            continue
        seen_quantiles.add(quantile)
        candidate_set = sample_worst_case(
            run["combined"],
            run["membership_map"],
            abstracts=abstracts,
            n_targets=max(args.samples_per_level * max(1, len(layers)) * 4, 100),
            n_easy=max(2, args.n_neighbors // 2),
            n_hard=max(2, args.n_neighbors // 2),
            min_cluster_size=args.min_size,
            boundary_quantile=quantile,
            seed=args.seed,
        )
        used_boundary_quantile = quantile
        if candidate_set.cases:
            if quantile != args.boundary_quantile:
                log.info(
                    "Fallback boundary quantile %.2f produced %d candidate cases",
                    quantile,
                    len(candidate_set.cases),
                )
            break
    if candidate_set is None:
        raise RuntimeError("Failed to construct boundary review candidate set")
    adjacency = _build_adjacency(run["combined"])

    rng = np.random.RandomState(args.seed)
    buckets: dict[int, list] = defaultdict(list)
    for case in candidate_set.cases:
        level = node_levels.get(case.target_uid)
        if level is None:
            continue
        preferred = _unique_keep_order(case.easy_neighbors + case.hard_neighbors)
        group_a = _build_same_cluster_group(
            case.target_uid,
            adjacency,
            run["membership_map"],
            preferred=preferred,
            n_neighbors=args.n_neighbors,
        )
        group_b = _build_contrast_group(
            case.target_uid,
            adjacency,
            run["membership_map"],
            n_neighbors=args.n_neighbors,
        )
        if len(group_a) < max(3, args.n_neighbors // 2):
            continue
        target_doc = meta.get(case.target_uid)
        if not target_doc or not target_doc.get("abstract"):
            continue
        buckets[level].append(
            {
                "target_uid": case.target_uid,
                "cluster_id": case.cluster_id,
                "cluster_size": case.cluster_size,
                "cross_cluster_ratio": case.cross_cluster_ratio,
                "n_cross_edges": case.n_cross_edges,
                "consensus_level": level,
                "group_a_uids": group_a,
                "group_b_uids": group_b,
            }
        )

    selected_cases: list[dict] = []
    for level in sorted(buckets):
        cases = buckets[level]
        rng.shuffle(cases)
        selected = cases[: args.samples_per_level]
        selected_cases.extend(selected)
        log.info("Level %d: selected %d / %d cases", level, len(selected), len(cases))

    output_payload: dict = {
        "field": args.field,
        "edge_dir": str(args.edge_dir),
        "abstract_path": str(args.abstract_path),
        "target_pct": args.target_pct,
        "top_k": args.top_k,
        "min_size": args.min_size,
        "n_seeds": args.n_seeds,
        "samples_per_level": args.samples_per_level,
        "n_neighbors": args.n_neighbors,
        "boundary_quantile": args.boundary_quantile,
        "used_boundary_quantile": used_boundary_quantile,
        "sample_only": args.sample_only,
        "n_candidate_cases": len(candidate_set.cases),
        "selected_cases": selected_cases,
    }

    if args.sample_only:
        out_path = args.output / f"{args.field}_boundary_review.json"
        save_json(output_payload, out_path)
        log.info("\nSaved sample set → %s", out_path)
        return

    from sciscape.clustering.cluster_naming import create_client
    from sciscape.evaluation.reviewer import review_belonging, review_group_cohesion, review_outliers

    client = create_client(model=args.model)
    reviewed_cases: list[dict] = []
    level_summary: dict[int, dict] = {}

    for idx, case in enumerate(selected_cases, start=1):
        target_doc = _docs_from_uids([case["target_uid"]], meta)[0]
        group_a = _docs_from_uids(case["group_a_uids"], meta)
        group_b = _docs_from_uids(case["group_b_uids"], meta)
        if len(group_a) < max(3, args.n_neighbors // 2):
            continue

        log.info(
            "[%d/%d] uid=%s level=%d",
            idx,
            len(selected_cases),
            case["target_uid"],
            case["consensus_level"],
        )
        record = dict(case)
        record["target_title"] = target_doc.get("title", "")

        belonging = review_belonging(
            client,
            target_doc,
            group_a,
            group_b,
            method_a="assigned_cluster",
            method_b="contrast_cluster",
            model=args.model,
        )
        cohesion = review_group_cohesion(
            client,
            group_a,
            method="assigned_cluster",
            model=args.model,
        )
        outliers = review_outliers(
            client,
            target_doc,
            group_a,
            method="assigned_cluster",
            model=args.model,
        )
        record["belonging"] = {
            "belongs_to": belonging.belongs_to,
            "confidence": belonging.confidence,
            "reasoning": belonging.reasoning,
        }
        record["cohesion"] = {
            "score": cohesion.cohesion_score,
            "theme": cohesion.theme,
            "n_outliers": cohesion.n_outliers,
            "reasoning": cohesion.reasoning,
        }
        record["outliers"] = {
            "n_outliers": outliers.n_outliers,
            "cluster_theme": outliers.cluster_theme,
            "reasoning": outliers.reasoning,
        }
        reviewed_cases.append(record)

    for level in sorted({case["consensus_level"] for case in reviewed_cases}):
        level_cases = [case for case in reviewed_cases if case["consensus_level"] == level]
        belonging_counts = Counter(case["belonging"]["belongs_to"] for case in level_cases)
        cohesion_scores = [case["cohesion"]["score"] for case in level_cases]
        outlier_counts = [case["outliers"]["n_outliers"] for case in level_cases]
        level_summary[level] = {
            "n_cases": len(level_cases),
            "belongs_assigned": belonging_counts.get("A", 0),
            "belongs_contrast": belonging_counts.get("B", 0),
            "cohesion_mean": round(float(np.mean(cohesion_scores)), 3) if cohesion_scores else None,
            "outliers_mean": round(float(np.mean(outlier_counts)), 3) if outlier_counts else None,
        }

    output_payload["reviewed_cases"] = reviewed_cases
    output_payload["summary_by_level"] = {str(k): v for k, v in level_summary.items()}
    out_path = args.output / f"{args.field}_boundary_review.json"
    save_json(output_payload, out_path)
    log.info("\nSaved → %s", out_path)


if __name__ == "__main__":
    main()
