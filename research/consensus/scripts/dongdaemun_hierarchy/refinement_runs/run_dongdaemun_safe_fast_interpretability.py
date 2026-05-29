"""Interpret safe-fast Dongdaemun changes with cluster overlap and text terms."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
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
import pyarrow.parquet as pq

SCRIPT_DIR = Path(__file__).resolve().parent

import evaluate_dongdaemun_refinement_slice4 as pilot  # noqa: E402
from run_dongdaemun_safe_fast_validation import SAFE_FAST_PRESET  # noqa: E402

DEFAULT_OUTPUT_DIR = (
    Path("research/consensus/results/adaptive_refinement")
    / "dongdaemun_safe_fast_interpretability"
)
SCHEMA_VERSION = 1

TOKEN_RE = re.compile(r"[a-z][a-z0-9][a-z0-9-]+")
STOPWORDS = {
    "about",
    "after",
    "against",
    "among",
    "analysis",
    "and",
    "are",
    "based",
    "been",
    "between",
    "both",
    "can",
    "case",
    "data",
    "different",
    "during",
    "each",
    "effect",
    "effects",
    "for",
    "from",
    "has",
    "have",
    "here",
    "high",
    "into",
    "its",
    "may",
    "method",
    "methods",
    "model",
    "more",
    "not",
    "our",
    "paper",
    "patients",
    "process",
    "results",
    "show",
    "shown",
    "study",
    "such",
    "than",
    "that",
    "the",
    "their",
    "these",
    "this",
    "through",
    "using",
    "was",
    "were",
    "which",
    "with",
}

@dataclass(frozen=True)
class InterpretabilityConfig:
    n_iterations: int = 10
    randomness: float = 0.01
    max_parent_clusters: int = 4
    max_child_clusters: int = 5
    min_child_docs: int = 20
    top_terms: int = 12
    example_titles: int = 5

def _json_safe(value: Any) -> Any:
    return pilot._json_safe(value)

def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )

def _csv_value(value: Any) -> Any:
    safe = _json_safe(value)
    if safe is None:
        return ""
    if isinstance(safe, (list, dict)):
        return json.dumps(safe, sort_keys=True, separators=(",", ":"))
    return safe

def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})

def _repo_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    parsed = Path(path)
    if parsed.is_absolute():
        return parsed
    return REPO_ROOT / parsed

def _field_number(sample: str) -> int:
    match = re.search(r"field(\d+)", sample)
    if match is None:
        raise ValueError(f"Cannot infer field number from sample: {sample}")
    return int(match.group(1))

def _load_node_work_ids(input_cfg: pilot.Slice4Input) -> np.ndarray:
    manifest_path = input_cfg.graph_dir / "node_manifest.parquet"
    if not manifest_path.exists():
        src_path = input_cfg.graph_dir / "src.u32.bin"
        resolved_manifest = src_path.resolve().parent / "node_manifest.parquet"
        if resolved_manifest.exists():
            manifest_path = resolved_manifest
    if manifest_path.exists():
        schema_names = set(pq.read_schema(manifest_path).names)
        uid_col = "uid" if "uid" in schema_names else "work_id"
        table = pq.read_table(manifest_path, columns=["node_idx", uid_col])
        node_idx = table.column("node_idx").combine_chunks().to_numpy(zero_copy_only=False)
        uid = table.column(uid_col).combine_chunks().to_numpy(zero_copy_only=False)
    else:
        field = _field_number(input_cfg.sample)
        mapping_path = REPO_ROOT / f"data/linktype_edges_gcc/field_{field}/node_mapping.parquet"
        table = pq.read_table(mapping_path, columns=["idx", "work_id"])
        node_idx = table.column("idx").combine_chunks().to_numpy(zero_copy_only=False)
        uid = table.column("work_id").combine_chunks().to_numpy(zero_copy_only=False)
    order = np.argsort(np.asarray(node_idx, dtype=np.int64), kind="stable")
    return np.asarray(uid, dtype=object)[order]

def _load_text_rows(input_cfg: pilot.Slice4Input, work_ids: np.ndarray) -> list[dict[str, str]]:
    field = _field_number(input_cfg.sample)
    text_path = REPO_ROOT / f"data/openalex_metadata/field_{field}/works_text.parquet"
    table = pq.read_table(text_path, columns=["work_id", "title", "abstract"])
    payload = table.to_pydict()
    by_work = {
        str(work_id): {
            "title": str(title or ""),
            "abstract": str(abstract or ""),
        }
        for work_id, title, abstract in zip(
            payload["work_id"],
            payload["title"],
            payload["abstract"],
        )
    }
    rows: list[dict[str, str]] = []
    for work_id in work_ids:
        item = by_work.get(str(work_id), {"title": "", "abstract": ""})
        rows.append(
            {
                "work_id": str(work_id),
                "title": item["title"],
                "abstract": item["abstract"],
            }
        )
    return rows

def _tokens(text: str) -> list[str]:
    cleaned = html.unescape(re.sub(r"<[^>]+>", " ", text.lower()))
    return [
        token.strip("-")
        for token in TOKEN_RE.findall(cleaned)
        if token not in STOPWORDS and len(token.strip("-")) >= 3
    ]

def _doc_tokens(text_rows: list[dict[str, str]]) -> list[list[str]]:
    return [_tokens(f"{row['title']} {row['abstract']}") for row in text_rows]

def _idf(doc_terms: list[list[str]]) -> dict[str, float]:
    df: Counter[str] = Counter()
    for terms in doc_terms:
        df.update(set(terms))
    n_docs = max(1, len(doc_terms))
    return {
        term: math.log((1.0 + n_docs) / (1.0 + count)) + 1.0
        for term, count in df.items()
    }

def _cluster_weight_map(membership: np.ndarray, node_weights: np.ndarray) -> dict[int, float]:
    weights: dict[int, float] = defaultdict(float)
    for label, weight in zip(membership, node_weights):
        weights[int(label)] += float(weight)
    return dict(weights)

def _cluster_indices(membership: np.ndarray) -> dict[int, np.ndarray]:
    groups: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(membership):
        groups[int(label)].append(int(idx))
    return {
        label: np.asarray(indices, dtype=np.int64)
        for label, indices in groups.items()
    }

def _top_terms(
    indices: np.ndarray,
    doc_terms: list[list[str]],
    idf: dict[str, float],
    *,
    n: int,
) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    for idx in indices:
        counts.update(doc_terms[int(idx)])
    total = sum(counts.values())
    scored = [
        (term, count, count * idf.get(term, 1.0))
        for term, count in counts.items()
    ]
    scored.sort(key=lambda item: (-item[2], item[0]))
    return [
        {
            "term": term,
            "count": int(count),
            "share": 0.0 if total == 0 else float(count / total),
            "score": float(score),
        }
        for term, count, score in scored[:n]
    ]

def _example_titles(
    indices: np.ndarray,
    text_rows: list[dict[str, str]],
    doc_terms: list[list[str]],
    top_terms: list[dict[str, Any]],
    *,
    n: int,
) -> list[str]:
    term_set = {str(item["term"]) for item in top_terms[:5]}
    scored: list[tuple[int, int, str]] = []
    for idx in indices:
        title = text_rows[int(idx)]["title"].strip()
        if not title:
            continue
        score = sum(1 for term in doc_terms[int(idx)] if term in term_set)
        scored.append((-score, int(idx), title))
    scored.sort()
    return [title for _, _, title in scored[:n]]

def _overlap_rows(
    *,
    standard_membership: np.ndarray,
    safe_membership: np.ndarray,
    node_weights: np.ndarray,
) -> list[dict[str, Any]]:
    standard_groups = _cluster_indices(standard_membership)
    safe_groups = _cluster_indices(safe_membership)
    standard_weights = _cluster_weight_map(standard_membership, node_weights)
    safe_weights = _cluster_weight_map(safe_membership, node_weights)
    safe_lookup: dict[int, Counter[int]] = {}
    for std_label, indices in standard_groups.items():
        counts: Counter[int] = Counter(int(safe_membership[int(idx)]) for idx in indices)
        safe_lookup[std_label] = counts
    rows: list[dict[str, Any]] = []
    for std_label, counts in safe_lookup.items():
        size = int(sum(counts.values()))
        largest_safe, largest_overlap = counts.most_common(1)[0]
        child_items = [
            {
                "safe_cluster": int(label),
                "overlap_nodes": int(count),
                "overlap_fraction_of_standard": float(count / size),
                "safe_cluster_doc_weight": float(safe_weights.get(int(label), 0.0)),
            }
            for label, count in counts.most_common()
        ]
        rows.append(
            {
                "standard_cluster": int(std_label),
                "standard_nodes": size,
                "standard_doc_weight": float(standard_weights.get(int(std_label), 0.0)),
                "largest_safe_cluster": int(largest_safe),
                "largest_overlap_nodes": int(largest_overlap),
                "largest_overlap_fraction": float(largest_overlap / size),
                "fragment_count_ge_20": int(
                    sum(1 for count in counts.values() if int(count) >= 20)
                ),
                "fragment_count_ge_1pct": int(
                    sum(1 for count in counts.values() if float(count / size) >= 0.01)
                ),
                "children": child_items,
            }
        )
    rows.sort(
        key=lambda row: (
            -float(row["standard_doc_weight"]),
            float(row["largest_overlap_fraction"]),
            -int(row["fragment_count_ge_20"]),
        )
    )
    return rows

def _case_payload(
    *,
    input_cfg: pilot.Slice4Input,
    config: InterpretabilityConfig,
) -> dict[str, Any]:
    n_nodes = pilot._infer_n_nodes(input_cfg)
    node_weights = pilot._load_node_weights(input_cfg.node_weights_path, n_nodes)
    graph = pilot._load_graph(input_cfg, node_weights)
    start = time.perf_counter()
    result = graph.run_leiden_dongdaemun_safe_fast_refinement(
        target_max_weight=float(input_cfg.target_max_doc_weight),
        resolution=float(input_cfg.resolution),
        seed=int(input_cfg.seed),
        n_iterations=int(config.n_iterations),
        randomness=float(config.randomness),
    )
    elapsed_sec = time.perf_counter() - start
    standard_membership = np.asarray(result.standard.membership, dtype=np.uint64)
    safe_membership = np.asarray(result.membership, dtype=np.uint64)
    work_ids = _load_node_work_ids(input_cfg)
    if int(work_ids.shape[0]) != int(safe_membership.shape[0]):
        raise ValueError(
            f"node mapping length mismatch for {input_cfg.sample}: "
            f"{work_ids.shape[0]} vs {safe_membership.shape[0]}"
        )
    text_rows = _load_text_rows(input_cfg, work_ids)
    terms = _doc_tokens(text_rows)
    idf = _idf(terms)
    standard_groups = _cluster_indices(standard_membership)
    safe_groups = _cluster_indices(safe_membership)
    overlap_rows = _overlap_rows(
        standard_membership=standard_membership,
        safe_membership=safe_membership,
        node_weights=node_weights,
    )
    selected_parent_rows = [
        row
        for row in overlap_rows
        if int(row["fragment_count_ge_20"]) >= 2
        or float(row["largest_overlap_fraction"]) < 0.95
    ][: int(config.max_parent_clusters)]
    case_rows: list[dict[str, Any]] = []
    for row in selected_parent_rows:
        std_label = int(row["standard_cluster"])
        std_indices = standard_groups[std_label]
        std_terms = _top_terms(std_indices, terms, idf, n=int(config.top_terms))
        child_terms = []
        for child in row["children"][: int(config.max_child_clusters)]:
            safe_label = int(child["safe_cluster"])
            safe_indices = safe_groups[safe_label]
            if int(safe_indices.shape[0]) < int(config.min_child_docs):
                continue
            top = _top_terms(safe_indices, terms, idf, n=int(config.top_terms))
            child_terms.append(
                {
                    **child,
                    "safe_cluster_nodes": int(safe_indices.shape[0]),
                    "top_terms": top,
                    "example_titles": _example_titles(
                        safe_indices,
                        text_rows,
                        terms,
                        top,
                        n=int(config.example_titles),
                    ),
                }
            )
        case_rows.append(
            {
                **row,
                "standard_top_terms": std_terms,
                "standard_example_titles": _example_titles(
                    std_indices,
                    text_rows,
                    terms,
                    std_terms,
                    n=int(config.example_titles),
                ),
                "children": child_terms,
            }
        )
    return {
        "sample": input_cfg.sample,
        "seed": int(input_cfg.seed),
        "summary_path": pilot._rel(input_cfg.summary_path),
        "elapsed_sec": float(elapsed_sec),
        "selected_variant": str(result.selected_variant),
        "triggered": bool(result.triggered),
        "fallback_reason": str(result.fallback_reason),
        "standard_quality": float(result.standard.quality),
        "safe_quality": float(result.quality),
        "quality_delta_vs_standard": float(result.quality - result.standard.quality),
        "standard_max_doc_weight_ratio": float(result.standard_max_doc_weight_ratio),
        "selected_max_doc_weight_ratio": float(result.selected_max_doc_weight_ratio),
        "standard_n_above_max_doc_weight": int(result.standard_n_above_max_doc_weight),
        "selected_n_above_max_doc_weight": int(result.selected_n_above_max_doc_weight),
        "n_parent_rows": int(len(case_rows)),
        "parents": case_rows,
    }

def _case_rows_for_csv(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        for parent in case.get("parents", []):
            rows.append(
                {
                    "sample": case["sample"],
                    "seed": case["seed"],
                    "quality_delta_vs_standard": case["quality_delta_vs_standard"],
                    "standard_cluster": parent["standard_cluster"],
                    "standard_doc_weight": parent["standard_doc_weight"],
                    "standard_nodes": parent["standard_nodes"],
                    "largest_overlap_fraction": parent["largest_overlap_fraction"],
                    "fragment_count_ge_20": parent["fragment_count_ge_20"],
                    "fragment_count_ge_1pct": parent["fragment_count_ge_1pct"],
                    "standard_top_terms": [
                        item["term"] for item in parent["standard_top_terms"][:8]
                    ],
                    "child_summaries": [
                        {
                            "safe_cluster": child["safe_cluster"],
                            "overlap_fraction": child[
                                "overlap_fraction_of_standard"
                            ],
                            "top_terms": [
                                item["term"] for item in child["top_terms"][:8]
                            ],
                        }
                        for child in parent["children"][:5]
                    ],
                }
            )
    return rows

def _write_report(path: Path, *, cases: list[dict[str, Any]], config: InterpretabilityConfig) -> None:
    lines = [
        "# Dongdaemun Safe-Fast Interpretability Report",
        "",
        "This report compares standard Leiden and the safe-fast selected partition using cluster overlaps and OpenAlex title/abstract terms.",
        "",
        "## Safe-Fast Preset",
        "",
        f"`{SAFE_FAST_PRESET}`",
        "",
    ]
    for case in cases:
        lines.extend(
            [
                f"## {case['sample']} seed {case['seed']}",
                "",
                f"- selected_variant: {case['selected_variant']}",
                f"- quality_delta_vs_standard: {case['quality_delta_vs_standard']:.6f}",
                f"- max_doc_weight_ratio: {case['standard_max_doc_weight_ratio']:.6f} -> {case['selected_max_doc_weight_ratio']:.6f}",
                f"- n_above_max_doc_weight: {case['standard_n_above_max_doc_weight']} -> {case['selected_n_above_max_doc_weight']}",
                f"- elapsed_sec: {case['elapsed_sec']:.3f}",
                "",
            ]
        )
        for parent in case.get("parents", []):
            parent_terms = ", ".join(
                item["term"] for item in parent["standard_top_terms"][:8]
            )
            lines.extend(
                [
                    f"### Standard cluster {parent['standard_cluster']}",
                    "",
                    f"- nodes/doc_weight: {parent['standard_nodes']} / {parent['standard_doc_weight']:.3f}",
                    f"- largest_overlap_fraction: {parent['largest_overlap_fraction']:.3f}",
                    f"- fragments >=20 docs: {parent['fragment_count_ge_20']}",
                    f"- parent terms: {parent_terms}",
                    "",
                    "| safe cluster | overlap frac | nodes | top terms |",
                    "| ---: | ---: | ---: | --- |",
                ]
            )
            for child in parent["children"][: int(config.max_child_clusters)]:
                terms = ", ".join(item["term"] for item in child["top_terms"][:8])
                lines.append(
                    f"| {child['safe_cluster']} | {child['overlap_fraction_of_standard']:.3f} | {child['safe_cluster_nodes']} | {terms} |"
                )
            lines.extend(["", "Representative parent titles:", ""])
            for title in parent["standard_example_titles"][: int(config.example_titles)]:
                lines.append(f"- {title}")
            lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")

def run_interpretability(
    input_cfgs: list[pilot.Slice4Input],
    *,
    output_dir: Path,
    config: InterpretabilityConfig | None = None,
) -> dict[str, Any]:
    config = config or InterpretabilityConfig()
    cases = [_case_payload(input_cfg=input_cfg, config=config) for input_cfg in input_cfgs]
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "dongdaemun_safe_fast_interpretability_summary.json"
    csv_path = output_dir / "dongdaemun_safe_fast_changed_clusters.csv"
    report_path = output_dir / "dongdaemun_safe_fast_interpretability_report.md"
    payload = {
        "schema": f"dongdaemun_safe_fast_interpretability.v{SCHEMA_VERSION}",
        "config": asdict(config),
        "safe_fast_preset": SAFE_FAST_PRESET,
        "cases": cases,
        "paths": {
            "summary": summary_path,
            "csv": csv_path,
            "report": report_path,
        },
    }
    _write_json(summary_path, payload)
    _write_csv(
        csv_path,
        _case_rows_for_csv(cases),
        fields=[
            "sample",
            "seed",
            "quality_delta_vs_standard",
            "standard_cluster",
            "standard_doc_weight",
            "standard_nodes",
            "largest_overlap_fraction",
            "fragment_count_ge_20",
            "fragment_count_ge_1pct",
            "standard_top_terms",
            "child_summaries",
        ],
    )
    _write_report(report_path, cases=cases, config=config)
    return payload

def _summary_paths_from_args(args: argparse.Namespace) -> list[Path]:
    paths = [Path(path) for path in (args.summary or [])]
    if args.summary_list is not None:
        for line in Path(args.summary_list).read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                paths.append(Path(stripped))
    return paths

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", action="append", default=[])
    parser.add_argument("--summary-list", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-iterations", type=int, default=10)
    parser.add_argument("--randomness", type=float, default=0.01)
    parser.add_argument("--max-parent-clusters", type=int, default=4)
    parser.add_argument("--max-child-clusters", type=int, default=5)
    parser.add_argument("--min-child-docs", type=int, default=20)
    parser.add_argument("--top-terms", type=int, default=12)
    parser.add_argument("--example-titles", type=int, default=5)
    return parser

def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    summary_paths = _summary_paths_from_args(args)
    if not summary_paths:
        parser.error("at least one --summary or --summary-list entry is required")
    input_cfgs = [
        pilot._resolve_input_from_summary(_repo_path(path) or Path(path))
        for path in summary_paths
    ]
    config = InterpretabilityConfig(
        n_iterations=int(args.n_iterations),
        randomness=float(args.randomness),
        max_parent_clusters=int(args.max_parent_clusters),
        max_child_clusters=int(args.max_child_clusters),
        min_child_docs=int(args.min_child_docs),
        top_terms=int(args.top_terms),
        example_titles=int(args.example_titles),
    )
    result = run_interpretability(
        input_cfgs,
        output_dir=_repo_path(args.output_dir) or args.output_dir,
        config=config,
    )
    print(json.dumps(_json_safe(result["paths"]), indent=2, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
