"""Protocol D v2: coverage-aware boundary utility review."""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import asdict
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


from _common import (
    allocate_effective_k,
    abstracts_lookup,
    load_abstracts_table,
    load_layer_tables,
    run_combination,
    save_json,
    select_layers,
)
from sciscape.evaluation.boundary_accuracy import summarize_boundary_coverage

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

VALID_METHODS = ("sum", "consensus", "rank", "max", "vote")


def _parse_layer_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def _parse_uid_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def _is_usable_record(record: dict[str, Any] | None) -> bool:
    if not record:
        return False
    return bool(str(record.get("title", "") or "").strip() and str(record.get("abstract", "") or "").strip())


def _docs_from_uids(uids: list[str], meta: dict[str, dict]) -> list[dict]:
    docs = []
    for uid in uids:
        record = meta.get(uid)
        if not _is_usable_record(record):
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
    return output_arg / f"{field}_boundary_coverage_v2_review.json"


def _sample_uid_order(payload: dict) -> list[str]:
    rows = payload.get("population_cases", []) + payload.get("diagnostic_cases", [])
    return list(dict.fromkeys(row["target_uid"] for row in rows))


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
        "gamma_a",
        "gamma_b",
        "n_neighbors",
        "sample_mode",
        "n_population_cases",
        "n_diagnostic_per_stratum",
        "max_group_jaccard",
    )
    if any(existing_payload.get(key) != current_payload.get(key) for key in keys):
        return False
    return _sample_uid_order(existing_payload) == _sample_uid_order(current_payload)


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


def _serialize_unary(result) -> dict:
    return {
        "decision": result.decision,
        "confidence": result.confidence,
        "reasoning": result.reasoning,
        "method": result.method,
    }


def _unique_selected_cases(payload: dict) -> list[dict]:
    by_uid: dict[str, dict] = {}
    for case in payload.get("population_cases", []) + payload.get("diagnostic_cases", []):
        by_uid.setdefault(case["target_uid"], case)
    return list(by_uid.values())


def _merge_reviews(cases: list[dict], reviewed_by_uid: dict[str, dict]) -> list[dict]:
    merged = []
    for case in cases:
        row = dict(case)
        reviewed = reviewed_by_uid.get(case["target_uid"], {})
        for key in ("gold", "unary_review_a", "unary_review_b", "review_error"):
            if key in reviewed:
                row[key] = reviewed[key]
        merged.append(row)
    return merged


def _review_complete(case: dict) -> bool:
    state = case.get("coverage_state")
    if state == "both_reviewable":
        return bool(case.get("gold") or case.get("review_error"))
    if state == "A_only_reviewable":
        return bool(case.get("unary_review_a") or case.get("review_error"))
    if state == "B_only_reviewable":
        return bool(case.get("unary_review_b") or case.get("review_error"))
    return True


def _refresh_summary(payload: dict, *, label_a: str, label_b: str) -> None:
    reviewed_by_uid = {case["target_uid"]: case for case in payload.get("reviewed_cases", [])}
    population = _merge_reviews(payload.get("population_cases", []), reviewed_by_uid)
    diagnostic = _merge_reviews(payload.get("diagnostic_cases", []), reviewed_by_uid)
    payload["review_progress"] = {
        "population_complete": sum(1 for case in population if _review_complete(case)),
        "population_total": len(population),
        "diagnostic_complete": sum(1 for case in diagnostic if _review_complete(case)),
        "diagnostic_total": len(diagnostic),
        "reviewed_unique_cases": len(reviewed_by_uid),
    }
    payload["summary"] = {
        "population": (
            summarize_boundary_coverage(population, method_a=label_a, method_b=label_b)
            if population and all(_review_complete(case) for case in population)
            else None
        ),
        "diagnostic": (
            summarize_boundary_coverage(diagnostic, method_a=label_a, method_b=label_b)
            if diagnostic and all(_review_complete(case) for case in diagnostic)
            else None
        ),
    }


def _review_payload(
    output_payload: dict,
    *,
    out_path: Path,
    meta: dict[str, dict],
    args: argparse.Namespace,
    label_a: str,
    label_b: str,
) -> None:
    """Run or resume LLM review for a prepared Protocol D v2 payload."""
    if args.sample_only:
        save_json(output_payload, out_path)
        log.info("Saved sample set -> %s", out_path)
        return

    from sciscape.clustering.cluster_naming import create_client
    from sciscape.evaluation.reviewer import review_boundary_gold, review_boundary_plausibility

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
                for case in _unique_selected_cases(output_payload)
                if case["target_uid"] in existing_by_uid
            ]
            reviewed_case_uids = {record["target_uid"] for record in reviewed_cases}
            if reviewed_cases:
                log.info("Resuming from %s (%d completed cases)", out_path, len(reviewed_cases))
        elif args.input_json is not None:
            log.info("Reviewing prepared input payload from %s", args.input_json)

    output_payload["reviewed_cases"] = reviewed_cases
    _refresh_summary(output_payload, label_a=label_a, label_b=label_b)
    save_json(output_payload, out_path)

    selected_cases = _unique_selected_cases(output_payload)
    for idx, case in enumerate(selected_cases, start=1):
        if case["target_uid"] in reviewed_case_uids:
            continue
        target_docs = _docs_from_uids([case["target_uid"]], meta)
        record = dict(case)
        if not target_docs:
            record["review_error"] = "missing target metadata"
            reviewed_cases.append(record)
            reviewed_case_uids.add(case["target_uid"])
            continue

        log.info("[%d/%d] uid=%s state=%s", idx, len(selected_cases), case["target_uid"], case["coverage_state"])
        try:
            if case["coverage_state"] == "both_reviewable":
                group_a_docs = _docs_from_uids(case["group_a_uids"], meta)
                group_b_docs = _docs_from_uids(case["group_b_uids"], meta)
                result = review_boundary_gold(
                    client,
                    target_docs[0],
                    group_a_docs,
                    group_b_docs,
                    method_a=label_a,
                    method_b=label_b,
                    model=args.model,
                )
                record["gold"] = _serialize_gold(result)
            elif case["coverage_state"] == "A_only_reviewable":
                group_a_docs = _docs_from_uids(case["group_a_uids"], meta)
                result = review_boundary_plausibility(
                    client,
                    target_docs[0],
                    group_a_docs,
                    method=label_a,
                    model=args.model,
                )
                record["unary_review_a"] = _serialize_unary(result)
            elif case["coverage_state"] == "B_only_reviewable":
                group_b_docs = _docs_from_uids(case["group_b_uids"], meta)
                result = review_boundary_plausibility(
                    client,
                    target_docs[0],
                    group_b_docs,
                    method=label_b,
                    model=args.model,
                )
                record["unary_review_b"] = _serialize_unary(result)
        except Exception as exc:  # pragma: no cover - protects resumable long runs
            record["review_error"] = f"{exc.__class__.__name__}: {exc}"
            output_payload["reviewed_cases"] = reviewed_cases + [record]
            _refresh_summary(output_payload, label_a=label_a, label_b=label_b)
            save_json(output_payload, out_path)
            raise

        reviewed_cases.append(record)
        reviewed_case_uids.add(case["target_uid"])
        output_payload["reviewed_cases"] = reviewed_cases
        _refresh_summary(output_payload, label_a=label_a, label_b=label_b)
        save_json(output_payload, out_path)

    output_payload["reviewed_cases"] = reviewed_cases
    _refresh_summary(output_payload, label_a=label_a, label_b=label_b)
    save_json(output_payload, out_path)
    log.info("Saved review -> %s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("edge_dir", type=Path, help="Directory with edge parquet files")
    parser.add_argument("abstract_path", type=Path, help="Abstract parquet with uid/title/abstract/pubyear")
    parser.add_argument("--field", type=str, required=True)
    parser.add_argument("--method-a", type=str, default="rank", choices=VALID_METHODS)
    parser.add_argument("--method-b", type=str, default="consensus", choices=VALID_METHODS)
    parser.add_argument("--label-a", type=str, default=None)
    parser.add_argument("--label-b", type=str, default=None)
    parser.add_argument("--layers-a", type=str, default="cc_cosine")
    parser.add_argument("--layers-b", type=str, default="bc_cosine,cc_cosine,dc_fractional")
    parser.add_argument("--exclude-layers-a", type=str, default=None)
    parser.add_argument("--exclude-layers-b", type=str, default=None)
    parser.add_argument("--effective-k", type=int, default=30)
    parser.add_argument("--gamma-a", type=float, default=None, help="Reuse a fixed gamma for method A")
    parser.add_argument("--gamma-b", type=float, default=None, help="Reuse a fixed gamma for method B")
    parser.add_argument("--target-pct", type=float, default=3.0)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--min-size", type=int, default=10)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--n-neighbors", type=int, default=8)
    parser.add_argument("--sample-mode", type=str, default="both", choices=("population", "diagnostic", "both"))
    parser.add_argument("--n-population-cases", type=int, default=30)
    parser.add_argument("--n-diagnostic-per-stratum", type=int, default=12)
    parser.add_argument("--max-group-jaccard", type=float, default=0.5)
    parser.add_argument("--target-uids", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-only", action="store_true")
    parser.add_argument("--input-json", type=Path, default=None, help="Review an existing sample-only payload")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("-o", "--output", type=Path, default=Path("results"))
    args = parser.parse_args()

    abstracts = load_abstracts_table(args.abstract_path)
    meta = abstracts_lookup(abstracts)
    if args.input_json is not None:
        output_payload = json.loads(args.input_json.read_text(encoding="utf-8"))
        output_payload["sample_only"] = bool(args.sample_only)
        label_a = output_payload.get("label_a", output_payload.get("method_a", "A"))
        label_b = output_payload.get("label_b", output_payload.get("method_b", "B"))
        out_path = args.output if args.output.suffix == ".json" else args.input_json
        _review_payload(output_payload, out_path=out_path, meta=meta, args=args, label_a=label_a, label_b=label_b)
        return

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
    budget_mode = "effective_k" if args.effective_k is not None else "top_k"
    protocol = _protocol_name(effective_k=args.effective_k)
    top_k_a = _resolve_top_k(sorted(layers_a), top_k=args.top_k, effective_k=args.effective_k)
    top_k_b = _resolve_top_k(sorted(layers_b), top_k=args.top_k, effective_k=args.effective_k)

    log.info("Field: %s", args.field)
    log.info("Boundary coverage v2: %s vs %s", label_a, label_b)
    log.info("  A layers: %s", ", ".join(sorted(layers_a)))
    log.info("  B layers: %s", ", ".join(sorted(layers_b)))

    run_a = run_combination(
        layers_a,
        strategy=args.method_a,
        target_pct=args.target_pct,
        top_k=top_k_a,
        min_size=args.min_size,
        n_seeds=args.n_seeds,
        gamma=args.gamma_a,
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
        gamma=args.gamma_b,
        compute_stability=False,
        compute_quality=False,
    )

    from sciscape.evaluation.sampler import sample_boundary_coverage_cases

    sample_set = sample_boundary_coverage_cases(
        run_a["combined"],
        run_a["membership_map"],
        run_b["combined"],
        run_b["membership_map"],
        method_a=label_a,
        method_b=label_b,
        abstracts=abstracts,
        n_neighbors=args.n_neighbors,
        min_cluster_size=args.min_size,
        n_population_cases=args.n_population_cases,
        n_diagnostic_per_stratum=args.n_diagnostic_per_stratum,
        sample_mode=args.sample_mode,
        target_uids=_parse_uid_list(args.target_uids),
        max_group_jaccard=args.max_group_jaccard,
        seed=args.seed,
    )

    output_payload = {
        "field": args.field,
        "edge_dir": str(args.edge_dir),
        "protocol": f"{protocol}_boundary_coverage_v2",
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
        "gamma_a": args.gamma_a,
        "gamma_b": args.gamma_b,
        "min_size": args.min_size,
        "n_seeds": args.n_seeds,
        "n_neighbors": args.n_neighbors,
        "sample_mode": args.sample_mode,
        "n_population_cases": args.n_population_cases,
        "n_diagnostic_per_stratum": args.n_diagnostic_per_stratum,
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
        "n_nodes": sample_set.n_nodes,
        "n_target_universe": sample_set.n_target_universe,
        "coverage_state_counts": sample_set.coverage_state_counts,
        "population_cases": [asdict(case) for case in sample_set.population_cases],
        "diagnostic_cases": [asdict(case) for case in sample_set.diagnostic_cases],
    }

    out_path = _resolve_output_path(args.output, args.field)
    _review_payload(output_payload, out_path=out_path, meta=meta, args=args, label_a=label_a, label_b=label_b)


if __name__ == "__main__":
    main()
