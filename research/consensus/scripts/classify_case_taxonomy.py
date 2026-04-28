"""Classify reviewed local neighborhood cases into a primary taxonomy."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from _common import abstracts_lookup, load_abstracts_table, save_json

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sciscape.clustering.cluster_naming import create_client
from sciscape.evaluation.reviewer import classify_case_taxonomy

ALLOWED_LABELS = {
    "single_cue_specificity",
    "broad_context_noise",
    "method_family_coherence",
    "material_family_coherence",
    "application_umbrella_noise",
    "semantic_drift",
    "coherent_refinement",
    "over_regularized_consensus",
}

HEURISTIC_KEYWORDS = {
    "single_cue_specificity": [
        "specific reaction",
        "specific material",
        "specific catalyst",
        "specific chemical",
        "specific problem setting",
        "exact",
        "decisive",
        "zinc",
        "silver",
        "lipase",
        "enzyme",
        "analyte",
        "named compound",
        "direct synthesis",
        "architecture",
    ],
    "broad_context_noise": [
        "too broad",
        "broad",
        "bridge",
        "dilute",
        "general context",
        "mixed",
        "noisy",
        "umbrella",
        "immediate research context",
    ],
    "method_family_coherence": [
        "method",
        "model",
        "simulation",
        "measurement",
        "spectroscopy",
        "rheology",
        "experimental",
        "computational",
        "synthesis route",
        "technique",
    ],
    "material_family_coherence": [
        "material family",
        "material system",
        "framework",
        "zeolite",
        "ionic liquid",
        "choline",
        "cobalt",
        "catalyst family",
        "compound family",
        "material context",
    ],
    "application_umbrella_noise": [
        "application",
        "applications",
        "solvent extraction",
        "recovery",
        "using ionic liquids",
        "broader use",
        "application umbrella",
    ],
    "semantic_drift": [
        "drift",
        "different topic",
        "different problem",
        "nearby but different",
        "shifts into",
        "unrelated",
        "different application",
    ],
    "coherent_refinement": [
        "narrower",
        "refinement",
        "more specific",
        "more focused",
        "preserves the target's immediate context",
        "target-centered",
        "coherent neighborhood",
    ],
    "over_regularized_consensus": [
        "omits obvious",
        "too aggressively",
        "too narrow",
        "suppresses",
        "drops obvious",
        "loses the specific signal",
        "over-regularized",
    ],
}


def _display_field_label(review_path: Path, review_payload: dict[str, Any]) -> str:
    label = review_path.stem.replace("_rank_shift_review", "")
    label = re.sub(r"_order_balanced_gemini_v\d+$", "", label)
    label = re.sub(r"_corrected$", "", label)
    if label:
        return label
    return str(review_payload.get("field", review_path.stem))


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


def _winner_method(case: dict[str, Any], *, label_a: str, label_b: str) -> tuple[str, str] | None:
    winner = case["comparison"]["winner"]
    if winner == "A":
        return label_a, label_b
    if winner == "B":
        return label_b, label_a
    return None


def _score_gap(case: dict[str, Any]) -> float:
    comparison = case["comparison"]
    if comparison["winner"] == "A":
        return float(comparison["score_a"]) - float(comparison["score_b"])
    return float(comparison["score_b"]) - float(comparison["score_a"])


def _normalize_label(label: str) -> str:
    return label if label in ALLOWED_LABELS else "unclassified"


def _representative_examples(classified_cases: list[dict[str, Any]], *, label_a: str, label_b: str) -> dict[str, list[dict[str, Any]]]:
    by_winner: dict[str, list[dict[str, Any]]] = {label_a: [], label_b: []}
    for winner_label in (label_a, label_b):
        cases = [case for case in classified_cases if case["winner_method"] == winner_label]
        cases.sort(
            key=lambda case: (
                -case["score_gap"],
                -case["taxonomy_confidence"],
                -case["shift_score"],
                case["target_uid"],
            )
        )
        by_winner[winner_label] = [
            {
                "target_uid": case["target_uid"],
                "target_title": case["target_title"],
                "primary_label": case["primary_label"],
                "score_gap": case["score_gap"],
                "shift_score": case["shift_score"],
                "rank_jaccard": case["rank_jaccard"],
                "winner_advantage": case["winner_advantage"],
                "loser_failure_mode": case["loser_failure_mode"],
                "review_reasoning": case["review_reasoning"],
            }
            for case in cases[:2]
        ]
    return by_winner


def _representative_by_label(classified_cases: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    labels = sorted({case["primary_label"] for case in classified_cases})
    by_label: dict[str, list[dict[str, Any]]] = {}
    for label in labels:
        cases = [case for case in classified_cases if case["primary_label"] == label]
        cases.sort(
            key=lambda case: (
                -case["score_gap"],
                -case["taxonomy_confidence"],
                case["target_uid"],
            )
        )
        by_label[label] = [
            {
                "field": case["field"],
                "top_k": case["top_k"],
                "winner_method": case["winner_method"],
                "target_uid": case["target_uid"],
                "target_title": case["target_title"],
                "score_gap": case["score_gap"],
                "winner_advantage": case["winner_advantage"],
                "loser_failure_mode": case["loser_failure_mode"],
                "review_reasoning": case["review_reasoning"],
            }
            for case in cases[:2]
        ]
    return by_label


def _summarize_cases(classified_cases: list[dict[str, Any]], *, top_k: int) -> dict[str, Any]:
    label_counts: Counter[str] = Counter()
    winner_counts: Counter[str] = Counter()
    label_by_winner: dict[str, Counter[str]] = defaultdict(Counter)
    for case in classified_cases:
        label_counts[case["primary_label"]] += 1
        winner_counts[case["winner_method"]] += 1
        label_by_winner[case["winner_method"]][case["primary_label"]] += 1
    return {
        "top_k": top_k,
        "n_cases": len(classified_cases),
        "label_counts": dict(label_counts),
        "winner_counts": dict(winner_counts),
        "label_by_winner": {
            winner: dict(counter)
            for winner, counter in sorted(label_by_winner.items())
        },
    }


def _split_reasoning(reasoning: str) -> tuple[str, str]:
    text = " ".join(reasoning.split())
    for marker in (" In contrast, ", " By contrast, ", " However, ", " Whereas "):
        if marker in text:
            left, right = text.split(marker, 1)
            return left.strip(), right.strip()
    return text, text


def _heuristic_taxonomy(
    case: dict[str, Any],
    *,
    target_doc: dict[str, Any],
    group_a_docs: Sequence[dict[str, Any]],
    group_b_docs: Sequence[dict[str, Any]],
    winner_method: str,
    loser_method: str,
) -> dict[str, Any]:
    reasoning = case["comparison"].get("reasoning", "") or ""
    target_text = f"{target_doc.get('title', '')} {target_doc.get('abstract', '')}"
    winner_docs = group_a_docs if case["comparison"]["winner"] == "A" else group_b_docs
    loser_docs = group_b_docs if case["comparison"]["winner"] == "A" else group_a_docs
    winner_titles = " ".join(doc.get("title", "") for doc in winner_docs[:4])
    loser_titles = " ".join(doc.get("title", "") for doc in loser_docs[:4])
    blob = " ".join([reasoning, target_text, winner_titles, loser_titles]).lower()

    scores: dict[str, int] = {}
    for label, keywords in HEURISTIC_KEYWORDS.items():
        scores[label] = sum(blob.count(keyword) for keyword in keywords)

    if winner_method.startswith("consensus") and scores["broad_context_noise"] > 0:
        scores["broad_context_noise"] += 2
    if winner_method.startswith("consensus") and (
        scores["material_family_coherence"] > 0 or scores["method_family_coherence"] > 0
    ):
        scores["coherent_refinement"] += 1
    if loser_method.startswith("consensus") and (
        scores["single_cue_specificity"] > 0 or "specific" in blob or "exact" in blob
    ):
        scores["over_regularized_consensus"] += 2

    priority = [
        "over_regularized_consensus",
        "single_cue_specificity",
        "broad_context_noise",
        "application_umbrella_noise",
        "semantic_drift",
        "method_family_coherence",
        "material_family_coherence",
        "coherent_refinement",
    ]
    best_label = max(priority, key=lambda label: (scores[label], -priority.index(label)))
    if scores[best_label] == 0:
        best_label = "semantic_drift" if winner_method.startswith("consensus") else "single_cue_specificity"

    winner_advantage, loser_failure_mode = _split_reasoning(reasoning)
    confidence = 4 if scores[best_label] >= 3 else 3 if scores[best_label] >= 1 else 2
    return {
        "primary_label": best_label,
        "winner_advantage": winner_advantage,
        "loser_failure_mode": loser_failure_mode,
        "confidence": confidence,
        "raw_response": "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review_json", nargs="+", type=Path, help="One or more rank_shift_review JSON files")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument(
        "--classifier",
        choices=("llm", "heuristic"),
        default="llm",
        help="Use live LLM classification or a deterministic reasoning-based heuristic fallback",
    )
    parser.add_argument("-o", "--output", type=Path, default=Path("results"))
    args = parser.parse_args()

    client = create_client(model=args.model) if args.classifier == "llm" else None
    combined_rows: list[dict[str, Any]] = []
    per_file_summaries: list[dict[str, Any]] = []

    for review_path in args.review_json:
        review = json.loads(review_path.read_text(encoding="utf-8"))
        field_label = _display_field_label(review_path, review)
        abstracts = load_abstracts_table(Path(review["abstract_path"]))
        meta = abstracts_lookup(abstracts)
        label_a = review["label_a"]
        label_b = review["label_b"]

        classified_cases: list[dict[str, Any]] = []
        skipped_ties = 0
        for case in review.get("reviewed_cases", []):
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
            if not group_a_docs or not group_b_docs:
                continue

            winner_info = _winner_method(case, label_a=label_a, label_b=label_b)
            if winner_info is None:
                skipped_ties += 1
                continue
            winner_method, loser_method = winner_info
            if args.classifier == "llm":
                taxonomy = classify_case_taxonomy(
                    client,
                    target_doc,
                    group_a_docs,
                    group_b_docs,
                    winner=case["comparison"]["winner"],
                    method_a=label_a,
                    method_b=label_b,
                    model=args.model,
                )
                primary_label = _normalize_label(taxonomy.primary_label)
                taxonomy_confidence = taxonomy.confidence
                winner_advantage = taxonomy.winner_advantage
                loser_failure_mode = taxonomy.loser_failure_mode
                raw_response = taxonomy.raw_response
            else:
                heuristic = _heuristic_taxonomy(
                    case,
                    target_doc=target_doc,
                    group_a_docs=group_a_docs,
                    group_b_docs=group_b_docs,
                    winner_method=winner_method,
                    loser_method=loser_method,
                )
                primary_label = _normalize_label(heuristic["primary_label"])
                taxonomy_confidence = int(heuristic["confidence"])
                winner_advantage = str(heuristic["winner_advantage"])
                loser_failure_mode = str(heuristic["loser_failure_mode"])
                raw_response = str(heuristic["raw_response"])
            row = {
                "review_json": str(review_path),
                "field": field_label,
                "top_k": int(review["top_k"]),
                "method_a": label_a,
                "method_b": label_b,
                "target_uid": case["target_uid"],
                "target_title": case.get("target_title", ""),
                "winner_letter": case["comparison"]["winner"],
                "winner_method": winner_method,
                "loser_method": loser_method,
                "primary_label": primary_label,
                "taxonomy_confidence": taxonomy_confidence,
                "winner_advantage": winner_advantage,
                "loser_failure_mode": loser_failure_mode,
                "score_gap": round(_score_gap(case), 4),
                "shift_score": case["shift_score"],
                "rank_jaccard": case["rank_jaccard"],
                "mean_abs_rank_shift": case["mean_abs_rank_shift"],
                "review_reasoning": case["comparison"].get("reasoning", ""),
                "classification_raw_response": raw_response,
            }
            classified_cases.append(row)
            combined_rows.append(row)

        file_summary = _summarize_cases(classified_cases, top_k=int(review["top_k"]))
        file_payload = {
            "model_used": args.model or "env/default",
            "classifier": args.classifier,
            "review_json": str(review_path),
            "field": field_label,
            "top_k": int(review["top_k"]),
            "label_a": label_a,
            "label_b": label_b,
            "skipped_ties": skipped_ties,
            "classified_cases": classified_cases,
            "summary": file_summary,
            "representative_examples": _representative_examples(
                classified_cases,
                label_a=label_a,
                label_b=label_b,
            ),
        }
        out_path = args.output / review_path.name.replace("_rank_shift_review.json", "_taxonomy.json")
        save_json(file_payload, out_path)
        per_file_summaries.append(
            {
                "review_json": str(review_path),
                "field": field_label,
                "top_k": int(review["top_k"]),
                "label_a": label_a,
                "label_b": label_b,
                "skipped_ties": skipped_ties,
                **file_summary,
            }
        )
        print(f"Saved → {out_path}")

    combined_label_counts: Counter[str] = Counter(row["primary_label"] for row in combined_rows)
    combined_winner_counts: Counter[str] = Counter(row["winner_method"] for row in combined_rows)
    label_by_winner: dict[str, Counter[str]] = defaultdict(Counter)
    winner_by_k: dict[int, Counter[str]] = defaultdict(Counter)
    for row in combined_rows:
        label_by_winner[row["winner_method"]][row["primary_label"]] += 1
        winner_by_k[row["top_k"]][row["winner_method"]] += 1

    combined_summary = {
        "model_used": args.model or "env/default",
        "classifier": args.classifier,
        "n_review_files": len(args.review_json),
        "n_classified_cases": len(combined_rows),
        "n_skipped_ties": sum(int(row.get("skipped_ties", 0)) for row in per_file_summaries),
        "per_file": per_file_summaries,
        "label_counts": dict(combined_label_counts),
        "winner_counts": dict(combined_winner_counts),
        "label_by_winner": {
            winner: dict(counter)
            for winner, counter in sorted(label_by_winner.items())
        },
        "winner_by_k": {
            str(k): dict(counter)
            for k, counter in sorted(winner_by_k.items())
        },
        "representative_by_label": _representative_by_label(combined_rows),
    }

    winner_methods = sorted({row["winner_method"] for row in combined_rows})
    combined_summary["representative_examples_by_winner"] = {
        winner_method: [
            {
                "field": case["field"],
                "top_k": case["top_k"],
                "primary_label": case["primary_label"],
                "target_uid": case["target_uid"],
                "target_title": case["target_title"],
                "score_gap": case["score_gap"],
                "winner_advantage": case["winner_advantage"],
                "loser_failure_mode": case["loser_failure_mode"],
                "review_reasoning": case["review_reasoning"],
            }
            for case in sorted(
                [row for row in combined_rows if row["winner_method"] == winner_method],
                key=lambda case: (
                    -case["score_gap"],
                    -case["taxonomy_confidence"],
                    case["target_uid"],
                ),
            )[:4]
        ]
        for winner_method in winner_methods
    }
    combined_summary.pop("representative_examples", None)

    stem = "taxonomy_combined" if len(args.review_json) > 1 else args.review_json[0].stem.replace("_rank_shift_review", "_taxonomy_summary")
    summary_json = args.output / f"{stem}.json"
    save_json(combined_summary, summary_json)
    print(f"Saved → {summary_json}")

    summary_csv = args.output / f"{stem}.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["field", "top_k", "winner_method", "primary_label", "count"],
        )
        writer.writeheader()
        counts: dict[tuple[str, int, str, str], int] = Counter(
            (row["field"], row["top_k"], row["winner_method"], row["primary_label"])
            for row in combined_rows
        )
        for (field, top_k, winner_method, primary_label), count in sorted(counts.items()):
            writer.writerow(
                {
                    "field": field,
                    "top_k": top_k,
                    "winner_method": winner_method,
                    "primary_label": primary_label,
                    "count": count,
                }
            )
    print(f"Saved → {summary_csv}")


if __name__ == "__main__":
    main()
