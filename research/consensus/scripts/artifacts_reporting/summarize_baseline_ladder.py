"""Summarize sum -> ablated sum -> consensus evidence in one ladder table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
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


from _common import save_json


def _infer_review_labels(payload: dict[str, Any]) -> tuple[str, str]:
    label_a = payload.get("label_a") or payload.get("method_a")
    label_b = payload.get("label_b") or payload.get("method_b")
    return str(label_a), str(label_b)


def _parse_slice_key(field: str) -> tuple[str, int]:
    if "_k" not in field:
        return field, 0
    prefix, suffix = field.split("_k", 1)
    digits = []
    for ch in suffix:
        if ch.isdigit():
            digits.append(ch)
        else:
            break
    return prefix, int("".join(digits)) if digits else 0


def _method_map_from_ablation(path: Path) -> tuple[tuple[str, int], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    field, top_k = _parse_slice_key(payload["field"])
    result_map = {row["method"]: row for row in payload["results"]}
    return (field, top_k), result_map


def _review_summary(path: Path) -> tuple[tuple[str, int], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    field, top_k = _parse_slice_key(payload["field"])
    label_a, label_b = _infer_review_labels(payload)
    return (field, top_k), {
        "review_json": str(path),
        "field": payload["field"],
        "top_k": top_k,
        "label_a": label_a,
        "label_b": label_b,
        "summary": payload.get("summary", {}),
    }


def _winner_string(summary: dict[str, Any]) -> str:
    comparison = summary.get("comparison", {})
    a = comparison.get("method_a_wins", 0)
    b = comparison.get("method_b_wins", 0)
    return f"{a}:{b}"


def _pairwise_lookup(reviews: list[dict[str, Any]], label_left: str, label_right: str) -> dict[str, Any] | None:
    for review in reviews:
        left = review["label_a"]
        right = review["label_b"]
        if (left, right) == (label_left, label_right):
            return review
    return None


def _pick_ablated_consensus_review(reviews: list[dict[str, Any]]) -> dict[str, Any] | None:
    preferred = [
        review
        for review in reviews
        if ("consensus" in review["label_a"] or "consensus" in review["label_b"])
        and ("sum_minus_" in review["label_a"] or "sum_minus_" in review["label_b"])
    ]
    if preferred:
        return preferred[0]
    fallback = [
        review
        for review in reviews
        if "consensus" in review["label_a"] or "consensus" in review["label_b"]
    ]
    return fallback[0] if fallback else None


def _render_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Baseline Ladder Summary",
        "",
        "| Slice | sum_all clusters/max% | ablated clusters/max% | consensus clusters/max% | sum_all vs ablated | sum_all vs consensus | ablated vs consensus |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {slice} | {sum_s} | {abl_s} | {con_s} | {sum_abl} | {sum_con} | {abl_con} |".format(
                slice=row["slice"],
                sum_s=row["sum_all_structure"],
                abl_s=row["ablated_structure"],
                con_s=row["consensus_structure"],
                sum_abl=row["sum_vs_ablated"],
                sum_con=row["sum_vs_consensus"],
                abl_con=row["ablated_vs_consensus"],
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation-json", nargs="+", type=Path, required=True)
    parser.add_argument("--review-json", nargs="+", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path, default=Path("research/consensus/results/ladder"))
    parser.add_argument("--stem", type=str, default="baseline_ladder")
    args = parser.parse_args()

    ablations = dict(_method_map_from_ablation(path) for path in args.ablation_json)
    reviews_by_slice: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for path in args.review_json:
        key, review = _review_summary(path)
        reviews_by_slice.setdefault(key, []).append(review)

    rows: list[dict[str, Any]] = []
    for key in sorted(ablations):
        field, top_k = key
        result_map = ablations[key]
        reviews = reviews_by_slice.get(key, [])

        consensus_review = _pick_ablated_consensus_review(reviews)
        if consensus_review is None:
            continue

        ablated_label = consensus_review["label_a"] if "consensus" not in consensus_review["label_a"] else consensus_review["label_b"]
        sum_all = result_map.get("sum_all")
        consensus_all = result_map.get("consensus_all")
        ablated = result_map.get(ablated_label)
        if sum_all is None or consensus_all is None or ablated is None:
            continue

        sum_vs_consensus = _pairwise_lookup(reviews, "sum", "consensus")
        if sum_vs_consensus is None:
            sum_vs_consensus = _pairwise_lookup(reviews, "sum_all", "consensus_all")
        sum_vs_ablated = _pairwise_lookup(reviews, "sum_all", ablated_label)
        ablated_vs_consensus = _pairwise_lookup(reviews, ablated_label, "consensus_all")
        if ablated_vs_consensus is None:
            ablated_vs_consensus = _pairwise_lookup(reviews, "consensus_all", ablated_label)

        row = {
            "field": field,
            "top_k": top_k,
            "slice": f"{field} k={top_k}",
            "ablated_label": ablated_label,
            "sum_all": sum_all,
            "ablated": ablated,
            "consensus_all": consensus_all,
            "sum_vs_ablated_review": sum_vs_ablated,
            "sum_vs_consensus_review": sum_vs_consensus,
            "ablated_vs_consensus_review": ablated_vs_consensus,
            "sum_all_structure": f"{sum_all['n_clusters']} / {sum_all['max_pct']:.2f}",
            "ablated_structure": f"{ablated['n_clusters']} / {ablated['max_pct']:.2f}",
            "consensus_structure": f"{consensus_all['n_clusters']} / {consensus_all['max_pct']:.2f}",
            "sum_vs_ablated": _winner_string(sum_vs_ablated["summary"]) if sum_vs_ablated else "NA",
            "sum_vs_consensus": _winner_string(sum_vs_consensus["summary"]) if sum_vs_consensus else "NA",
            "ablated_vs_consensus": _winner_string(ablated_vs_consensus["summary"]) if ablated_vs_consensus else "NA",
        }
        rows.append(row)

    payload = {"rows": rows}
    out_json = args.output / f"{args.stem}.json"
    save_json(payload, out_json)
    print(f"Saved → {out_json}")

    out_md = args.output / f"{args.stem}.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(_render_markdown(rows), encoding="utf-8")
    print(f"Saved → {out_md}")


if __name__ == "__main__":
    main()
