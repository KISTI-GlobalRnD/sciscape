"""Build and evaluate a taxonomy calibration set from reviewed local cases."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from _common import abstracts_lookup, load_abstracts_table, save_json

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sciscape.clustering.cluster_naming import create_client
from sciscape.evaluation.reviewer import classify_case_taxonomy


def _load_cases(summary_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for item in summary.get("per_file", []):
        taxonomy_path = item.get("taxonomy_json")
        if not taxonomy_path:
            continue
        payload = json.loads(Path(taxonomy_path).read_text(encoding="utf-8"))
        rows.extend(payload.get("classified_cases", []))
    return summary, rows


def _round_robin_sample(groups: dict[tuple[str, str], list[dict[str, Any]]], n_cases: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    for rows in groups.values():
        rng.shuffle(rows)

    ordered_keys = sorted(groups, key=lambda key: (-len(groups[key]), key[0], key[1]))
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    while len(selected) < n_cases:
        progressed = False
        for key in ordered_keys:
            bucket = groups[key]
            while bucket:
                candidate = bucket.pop()
                identity = (candidate["review_json"], candidate["target_uid"])
                if identity in seen:
                    continue
                selected.append(candidate)
                seen.add(identity)
                progressed = True
                break
            if len(selected) >= n_cases:
                break
        if not progressed:
            break
    return selected


def _build_sample(rows: list[dict[str, Any]], *, n_cases: int, seed: int) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["winner_method"], row["primary_label"])].append(dict(row))
    return _round_robin_sample(groups, n_cases=n_cases, seed=seed)


def _docs_from_ranked_neighbors(rows: list[dict[str, Any]], meta: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for row in rows:
        record = meta.get(row["uid"])
        if not record:
            continue
        docs.append(
            {
                "uid": row["uid"],
                "rank": row.get("rank"),
                "weight": row.get("weight"),
                "title": record.get("title", "") or "",
                "abstract": record.get("abstract", "") or "",
                "pubyear": record.get("pubyear"),
            }
        )
    return docs


def _find_review_case(review_payload: dict[str, Any], target_uid: str) -> dict[str, Any]:
    for case in review_payload.get("reviewed_cases", []):
        if case["target_uid"] == target_uid:
            return case
    raise KeyError(f"Target UID not found in review payload: {target_uid}")


def _evaluate_llm_labels(calibration_cases: list[dict[str, Any]], *, model: str | None) -> dict[str, Any]:
    by_review: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in calibration_cases:
        by_review[case["review_json"]].append(case)

    client = create_client(model=model)
    comparisons: list[dict[str, Any]] = []

    for review_path_str, cases in by_review.items():
        review_path = Path(review_path_str)
        review_payload = json.loads(review_path.read_text(encoding="utf-8"))
        abstracts = load_abstracts_table(Path(review_payload["abstract_path"]))
        meta = abstracts_lookup(abstracts)
        label_a = review_payload["label_a"]
        label_b = review_payload["label_b"]

        for case in cases:
            review_case = _find_review_case(review_payload, case["target_uid"])
            target_record = meta[case["target_uid"]]
            target_doc = {
                "uid": case["target_uid"],
                "title": target_record.get("title", "") or "",
                "abstract": target_record.get("abstract", "") or "",
                "pubyear": target_record.get("pubyear"),
            }
            group_a_docs = _docs_from_ranked_neighbors(review_case["neighbors_a"], meta)
            group_b_docs = _docs_from_ranked_neighbors(review_case["neighbors_b"], meta)
            taxonomy = classify_case_taxonomy(
                client,
                target_doc,
                group_a_docs,
                group_b_docs,
                winner=review_case["comparison"]["winner"],
                method_a=label_a,
                method_b=label_b,
                model=model,
            )
            comparisons.append(
                {
                    "review_json": review_path_str,
                    "field": case["field"],
                    "top_k": case["top_k"],
                    "target_uid": case["target_uid"],
                    "target_title": case["target_title"],
                    "winner_method": case["winner_method"],
                    "heuristic_label": case["primary_label"],
                    "llm_label": taxonomy.primary_label,
                    "label_match": case["primary_label"] == taxonomy.primary_label,
                    "heuristic_confidence": case.get("taxonomy_confidence"),
                    "llm_confidence": taxonomy.confidence,
                    "winner_advantage": taxonomy.winner_advantage,
                    "loser_failure_mode": taxonomy.loser_failure_mode,
                    "llm_raw_response": taxonomy.raw_response,
                }
            )

    exact_agreement = sum(1 for row in comparisons if row["label_match"])
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    for row in comparisons:
        confusion[row["heuristic_label"]][row["llm_label"]] += 1
    mismatches = [row for row in comparisons if not row["label_match"]]
    mismatches.sort(key=lambda row: (row["heuristic_label"], row["llm_label"], row["target_uid"]))
    return {
        "model_used": model or "env/default",
        "n_cases": len(comparisons),
        "exact_agreement": exact_agreement,
        "exact_agreement_rate": round(exact_agreement / len(comparisons), 4) if comparisons else None,
        "confusion": {key: dict(value) for key, value in sorted(confusion.items())},
        "comparisons": comparisons,
        "mismatches": mismatches[:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("taxonomy_summary", type=Path, help="Combined taxonomy summary JSON")
    parser.add_argument("--n-cases", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument(
        "--run-llm",
        action="store_true",
        help="Re-label sampled cases with the configured LLM and compare against the heuristic labels",
    )
    parser.add_argument("-o", "--output", type=Path, default=Path("research/consensus/results/taxonomy"))
    parser.add_argument("--stem", type=str, default="taxonomy_calibration")
    args = parser.parse_args()

    summary, rows = _load_cases(args.taxonomy_summary)
    sample = _build_sample(rows, n_cases=args.n_cases, seed=args.seed)
    sample_payload = {
        "source_summary": str(args.taxonomy_summary),
        "n_source_cases": len(rows),
        "n_sampled_cases": len(sample),
        "seed": args.seed,
        "sampled_cases": sample,
    }
    sample_path = args.output / f"{args.stem}_sample.json"
    save_json(sample_payload, sample_path)
    print(f"Saved → {sample_path}")

    summary_counts = {
        "winner_counts": dict(Counter(case["winner_method"] for case in sample)),
        "label_counts": dict(Counter(case["primary_label"] for case in sample)),
        "winner_by_k": {
            str(k): dict(Counter(case["winner_method"] for case in sample if int(case["top_k"]) == k))
            for k in sorted({int(case["top_k"]) for case in sample})
        },
    }
    summary_path = args.output / f"{args.stem}_sample_summary.json"
    save_json(summary_counts, summary_path)
    print(f"Saved → {summary_path}")

    if not args.run_llm:
        return

    comparison = _evaluate_llm_labels(sample, model=args.model)
    comparison["source_summary_model"] = summary.get("model_used")
    comparison_path = args.output / f"{args.stem}_llm_comparison.json"
    save_json(comparison, comparison_path)
    print(f"Saved → {comparison_path}")


if __name__ == "__main__":
    main()
