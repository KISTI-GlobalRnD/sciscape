"""Protocol D: gold-label boundary accuracy review on disagreement cases."""

from __future__ import annotations

import argparse
import json
import logging
import random
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
    allocate_effective_k,
    abstracts_lookup,
    load_abstracts_table,
    load_layer_tables,
    run_combination,
    save_json,
    select_layers,
)
from sciscape.evaluation.boundary_accuracy import summarize_boundary_accuracy

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


def _min_group_size(n_neighbors: int) -> int:
    return max(2, (n_neighbors + 1) // 2)


def _is_reviewable_case(case: dict, meta: dict[str, dict], *, min_group_size: int) -> bool:
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


def _resolve_output_path(output_arg: Path, field: str) -> Path:
    if output_arg.suffix == ".json":
        return output_arg
    return output_arg / f"{field}_boundary_accuracy_review.json"


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
        "boundary_quantile",
        "max_group_jaccard",
    )
    if any(existing_payload.get(key) != current_payload.get(key) for key in keys):
        return False
    old_uids = [case["target_uid"] for case in existing_payload.get("selected_cases", [])]
    new_uids = [case["target_uid"] for case in current_payload.get("selected_cases", [])]
    return old_uids == new_uids


def _serialize_gold(result) -> dict:
    return {
        "decision": result.decision,
        "belongs_with_a": result.belongs_with_a,
        "belongs_with_b": result.belongs_with_b,
        "confidence": result.confidence,
        "reasoning": result.reasoning,
        "presented_decision": result.presented_decision,
        "presented_belongs_with_a": result.presented_belongs_with_a,
        "presented_belongs_with_b": result.presented_belongs_with_b,
        "presented_method_a": result.presented_method_a,
        "presented_method_b": result.presented_method_b,
        "swapped": result.swapped,
    }


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
    parser.add_argument("--effective-k", type=int, default=None)
    parser.add_argument("--target-pct", type=float, default=3.0)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--min-size", type=int, default=10)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--n-cases", type=int, default=24)
    parser.add_argument("--n-neighbors", type=int, default=8)
    parser.add_argument("--boundary-quantile", type=float, default=0.9)
    parser.add_argument("--max-group-jaccard", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-only", action="store_true")
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

    abstracts = load_abstracts_table(args.abstract_path)
    meta = abstracts_lookup(abstracts)
    reviewable_uids = set(meta)
    label_a = args.label_a or args.method_a
    label_b = args.label_b or args.method_b
    budget_mode = "effective_k" if args.effective_k is not None else "top_k"
    protocol = _protocol_name(effective_k=args.effective_k)
    top_k_a = _resolve_top_k(sorted(layers_a), top_k=args.top_k, effective_k=args.effective_k)
    top_k_b = _resolve_top_k(sorted(layers_b), top_k=args.top_k, effective_k=args.effective_k)

    log.info("Field: %s", args.field)
    log.info("Boundary accuracy: %s vs %s", label_a, label_b)
    log.info("  A layers: %s", ", ".join(sorted(layers_a)))
    log.info("  B layers: %s", ", ".join(sorted(layers_b)))
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

    min_group_size = _min_group_size(args.n_neighbors)
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
        boundary_quantile=args.boundary_quantile,
        max_group_jaccard=args.max_group_jaccard,
        allowed_uids=reviewable_uids,
        seed=args.seed,
    )
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
        case for case in candidate_rows if _is_reviewable_case(case, meta, min_group_size=min_group_size)
    ]
    selected_cases = _select_cases(reviewable_rows, args.n_cases, args.seed)

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
        "boundary_quantile": args.boundary_quantile,
        "max_group_jaccard": args.max_group_jaccard,
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
        "n_candidate_cases": candidate_set.n_candidates,
        "n_reviewable_cases": len(reviewable_rows),
        "selected_cases": selected_cases,
    }

    out_path = _resolve_output_path(args.output, args.field)
    if args.sample_only:
        save_json(output_payload, out_path)
        log.info("Saved sample set → %s", out_path)
        return

    from sciscape.clustering.cluster_naming import create_client
    from sciscape.evaluation.reviewer import review_boundary_gold

    random.seed(args.seed)
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

    output_payload["reviewed_cases"] = reviewed_cases
    output_payload["summary"] = summarize_boundary_accuracy(reviewed_cases, method_a=label_a, method_b=label_b)
    save_json(output_payload, out_path)

    for idx, case in enumerate(selected_cases, start=1):
        if case["target_uid"] in reviewed_case_uids:
            continue
        target_docs = _docs_from_uids([case["target_uid"]], meta)
        group_a_docs = _docs_from_uids(case["group_a_uids"], meta)
        group_b_docs = _docs_from_uids(case["group_b_uids"], meta)
        if not target_docs or len(group_a_docs) < min_group_size or len(group_b_docs) < min_group_size:
            continue
        target_doc = target_docs[0]
        log.info(
            "[%d/%d] uid=%s jaccard=%.2f",
            idx,
            len(selected_cases),
            case["target_uid"],
            case["jaccard"],
        )
        result = review_boundary_gold(
            client,
            target_doc,
            group_a_docs,
            group_b_docs,
            method_a=label_a,
            method_b=label_b,
            model=args.model,
        )
        record = dict(case)
        record["gold"] = _serialize_gold(result)
        reviewed_cases.append(record)
        reviewed_case_uids.add(case["target_uid"])
        output_payload["reviewed_cases"] = reviewed_cases
        output_payload["summary"] = summarize_boundary_accuracy(reviewed_cases, method_a=label_a, method_b=label_b)
        save_json(output_payload, out_path)

    output_payload["reviewed_cases"] = reviewed_cases
    output_payload["summary"] = summarize_boundary_accuracy(reviewed_cases, method_a=label_a, method_b=label_b)
    save_json(output_payload, out_path)
    log.info("Saved review → %s", out_path)


if __name__ == "__main__":
    main()
