#!/usr/bin/env python3
"""Validate opt-in non-monotone external-grain Leiden escape probes.

The probe starts from a standard Leiden membership, temporarily moves a ranked
external-grain group to its strongest neighboring cluster, then runs normal
Leiden polish at the original CPM resolution. Acceptance is monotone: the
returned proposal is used only when polish is non-loss versus the baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

REPO_ROOT = Path(__file__).resolve().parents[3]

from sciscape.clustering.leiden_rust import build_leiden_graph  # noqa: E402


DEFAULT_GRAPH_DIRS = (
    REPO_ROOT / "research/consensus/results/adaptive_refinement/"
    "dongdaemun_safe_fast_layer_comparison/bc_cosine_20260507/graphs/"
    "field15_gcc_emb_full_knn30/seed_42/bc_cosine",
    REPO_ROOT / "research/consensus/results/adaptive_refinement/"
    "dongdaemun_safe_fast_layer_comparison/cc_cosine_20260507/graphs/"
    "field15_gcc_emb_full_knn30/seed_42/cc_cosine",
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "research/consensus/results/adaptive_refinement/"
    "leiden_hysteresis_shatter_smoke_20260512"
)


@dataclass(frozen=True)
class GraphArrays:
    src: np.ndarray
    dst: np.ndarray
    weight: np.ndarray
    node_weights: np.ndarray


@dataclass(frozen=True)
class BaselineResult:
    seed: int
    quality: float
    n_clusters: int
    elapsed_sec: float
    membership: np.ndarray


def _load_graph_arrays(graph_dir: Path) -> GraphArrays:
    return GraphArrays(
        src=np.memmap(graph_dir / "src.u32.bin", dtype=np.uint32, mode="r"),
        dst=np.memmap(graph_dir / "dst.u32.bin", dtype=np.uint32, mode="r"),
        weight=np.memmap(graph_dir / "weight.f64.bin", dtype=np.float64, mode="r"),
        node_weights=np.memmap(
            graph_dir / "node_weights.f64.bin", dtype=np.float64, mode="r"
        ),
    )


def _case_name(graph_dir: Path) -> str:
    try:
        relative = graph_dir.relative_to(REPO_ROOT)
    except ValueError:
        return graph_dir.name
    parts = relative.parts
    if len(parts) >= 3:
        return "_".join(parts[-4:])
    return graph_dir.name


def _parse_int_list(value: str) -> list[int]:
    out = [int(part) for part in value.split(",") if part.strip()]
    if not out:
        raise ValueError("expected at least one integer")
    return out


def _parse_graph_dirs(value: str | None) -> list[Path]:
    if value is None:
        return [path.resolve() for path in DEFAULT_GRAPH_DIRS]
    return [
        Path(part).expanduser().resolve() for part in value.split(",") if part.strip()
    ]


def _cluster_weights(membership: np.ndarray, node_weights: np.ndarray) -> np.ndarray:
    labels = np.asarray(membership)
    if not np.issubdtype(labels.dtype, np.integer):
        labels = labels.astype(np.int64, copy=False)
    return np.bincount(
        labels,
        weights=np.asarray(node_weights, dtype=np.float64),
    )


def _membership_metrics(
    membership: np.ndarray,
    node_weights: np.ndarray,
    *,
    target_max_weight: float,
) -> dict[str, Any]:
    weights = _cluster_weights(membership, node_weights)
    max_weight = float(weights.max()) if weights.size else 0.0
    return {
        "n_clusters": int(weights.size),
        "max_doc_weight": max_weight,
        "max_doc_weight_ratio": (
            max_weight / target_max_weight if target_max_weight > 0.0 else 0.0
        ),
        "n_above_target": int(np.count_nonzero(weights > target_max_weight))
        if target_max_weight > 0.0
        else 0,
    }


def _top_weight_clusters(
    membership: np.ndarray,
    node_weights: np.ndarray,
    *,
    count: int,
    min_weight: float,
) -> list[int]:
    weights = _cluster_weights(membership, node_weights)
    order = np.argsort(weights)[::-1]
    selected: list[int] = []
    for cid in order:
        if len(selected) >= count:
            break
        if weights[cid] >= min_weight:
            selected.append(int(cid))
    return selected


def _external_priority_clusters(
    graph: Any,
    membership: np.ndarray,
    *,
    resolution: float,
    count: int,
    min_doc_weight: float,
    min_assigned_fraction: float,
    min_best_group_fraction: float,
) -> list[int]:
    if count <= 0:
        return []
    all_clusters = np.arange(int(np.max(membership)) + 1, dtype=np.uint64)
    priority_clusters = getattr(graph, "external_grain_priority_clusters", None)
    if priority_clusters is not None:
        return priority_clusters(
            membership,
            all_clusters,
            resolution=resolution,
            count=count,
            min_doc_weight=min_doc_weight,
            min_assigned_fraction=min_assigned_fraction,
            min_best_group_fraction=min_best_group_fraction,
        )
    probes = graph.external_grain_probes(
        membership,
        all_clusters,
        resolution=resolution,
        min_doc_weight=min_doc_weight,
        min_assigned_fraction=min_assigned_fraction,
        min_best_group_fraction=min_best_group_fraction,
    )
    order = sorted(
        range(probes.n_probes),
        key=lambda idx: (
            bool(probes.recommended_for_split_repair[idx]),
            float(probes.priority[idx]),
            float(probes.best_group_to_target_weight[idx]),
            float(probes.best_group_weight[idx]),
        ),
        reverse=True,
    )
    selected: list[int] = []
    for idx in order:
        if len(selected) >= count:
            break
        if int(probes.best_group_target[idx]) < 0:
            continue
        selected.append(int(probes.cluster[idx]))
    return selected


def _candidate_clusters(
    graph: Any,
    membership: np.ndarray,
    node_weights: np.ndarray,
    *,
    resolution: float,
    top_weight_count: int,
    external_priority_count: int,
    min_suspect_weight: float,
    min_doc_weight: float,
    min_assigned_fraction: float,
    min_best_group_fraction: float,
) -> list[int]:
    selected = _top_weight_clusters(
        membership,
        node_weights,
        count=top_weight_count,
        min_weight=min_suspect_weight,
    )
    selected.extend(
        _external_priority_clusters(
            graph,
            membership,
            resolution=resolution,
            count=external_priority_count,
            min_doc_weight=min_doc_weight,
            min_assigned_fraction=min_assigned_fraction,
            min_best_group_fraction=min_best_group_fraction,
        )
    )
    seen: set[int] = set()
    out: list[int] = []
    for cid in selected:
        if cid not in seen:
            seen.add(cid)
            out.append(cid)
    return out


def _run_baseline(
    graph: Any,
    *,
    seed: int,
    resolution: float,
    n_iterations: int,
    randomness: float,
) -> BaselineResult:
    t0 = time.perf_counter()
    result = graph.run_leiden(
        resolution=float(resolution),
        seed=int(seed),
        n_iterations=int(n_iterations),
        randomness=float(randomness),
    )
    return BaselineResult(
        seed=int(seed),
        quality=float(result.quality),
        n_clusters=int(result.n_clusters),
        elapsed_sec=time.perf_counter() - t0,
        membership=np.asarray(result.membership, dtype=np.uint64),
    )


def _best_candidate_row(candidate_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidate_rows:
        return {}
    return max(candidate_rows, key=lambda row: float(row["post_polish_delta_q"]))


def _row_for_membership(
    *,
    case: str,
    graph_dir: Path,
    policy: str,
    seed: int,
    max_group_candidates: int,
    baseline: BaselineResult,
    membership: np.ndarray,
    quality: float,
    accepted: bool,
    candidate_clusters: list[int],
    candidate_rows: list[dict[str, Any]],
    elapsed_sec: float,
    node_weights: np.ndarray,
    target_max_weight: float,
    pressure_guard_passed: bool | None = None,
) -> dict[str, Any]:
    best = _best_candidate_row(candidate_rows)
    metrics = _membership_metrics(
        membership,
        node_weights,
        target_max_weight=target_max_weight,
    )
    row: dict[str, Any] = {
        "case": case,
        "graph_dir": str(graph_dir.relative_to(REPO_ROOT)),
        "policy": policy,
        "seed": int(seed),
        "max_group_candidates": int(max_group_candidates),
        "baseline_quality": float(baseline.quality),
        "quality": float(quality),
        "quality_delta_vs_baseline": float(quality - baseline.quality),
        "post_polish_delta_q": float(best.get("post_polish_delta_q", 0.0)),
        "pre_polish_delta_q": float(best.get("pre_polish_delta_q", 0.0)),
        "accepted": bool(accepted),
        "elapsed_sec": float(elapsed_sec),
        "baseline_elapsed_sec": float(baseline.elapsed_sec),
        "baseline_n_clusters": int(baseline.n_clusters),
        "candidate_cluster_count": int(len(candidate_clusters)),
        "candidate_clusters": json.dumps(candidate_clusters),
        "evaluated_group_candidates": int(len(candidate_rows)),
        "nmi_vs_baseline": float(
            normalized_mutual_info_score(baseline.membership, membership)
        ),
        "ari_vs_baseline": float(adjusted_rand_score(baseline.membership, membership)),
    }
    row.update(metrics)
    for key in (
        "source_cluster",
        "target_cluster",
        "group_kind",
        "group_count",
        "group_weight",
        "group_to_target_weight",
        "group_move_delta_q",
        "group_split_delta_q",
        "recommended_for_split_repair",
        "priority",
    ):
        row[key] = best.get(key)
    if pressure_guard_passed is not None:
        row["pressure_guard_passed"] = bool(pressure_guard_passed)
    return row


def _write_outputs(
    out_dir: Path, rows: list[dict[str, Any]], summary: dict[str, Any]
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "hysteresis_shatter_smoke_rows.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    def display_path(path: Path) -> str:
        try:
            return str(path.relative_to(REPO_ROOT))
        except ValueError:
            return str(path)

    summary["paths"] = {
        "rows_csv": display_path(csv_path),
        "summary_json": display_path(out_dir / "hysteresis_shatter_smoke_summary.json"),
        "report_md": display_path(out_dir / "hysteresis_shatter_smoke_report.md"),
    }
    (out_dir / "hysteresis_shatter_smoke_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    lines = [
        "# Leiden Non-Monotone Group Escape Smoke",
        "",
        "| case | policy | seed | k | accepted | final dq | best post dq | NMI | ARI | max ratio |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {case} | {policy} | {seed} | {k} | {accepted} | {dq:.6f} | {post:.6f} | {nmi:.6f} | {ari:.6f} | {ratio:.6f} |".format(
                case=row["case"],
                policy=row["policy"],
                seed=int(row["seed"]),
                k=int(row["max_group_candidates"]),
                accepted=str(bool(row["accepted"])),
                dq=float(row["quality_delta_vs_baseline"]),
                post=float(row["post_polish_delta_q"]),
                nmi=float(row["nmi_vs_baseline"]),
                ari=float(row["ari_vs_baseline"]),
                ratio=float(row["max_doc_weight_ratio"]),
            )
        )
    (out_dir / "hysteresis_shatter_smoke_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dirs", type=str, default=None)
    parser.add_argument("--graph-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--resolution", type=float, default=0.01)
    parser.add_argument("--target-max-weight", type=float, default=1000.0)
    parser.add_argument("--seeds", type=str, default="11,42,73,101")
    parser.add_argument("--baseline-iterations", type=int, default=10)
    parser.add_argument("--polish-iterations", type=int, default=5)
    parser.add_argument("--randomness", type=float, default=0.01)
    parser.add_argument("--max-suspect-clusters", type=int, default=2)
    parser.add_argument("--max-external-priority-clusters", type=int, default=3)
    parser.add_argument("--min-suspect-weight", type=float, default=500.0)
    parser.add_argument("--max-group-candidates", type=str, default="1,3,5")
    parser.add_argument("--min-doc-weight", type=float, default=0.0)
    parser.add_argument("--min-assigned-fraction", type=float, default=0.0)
    parser.add_argument("--min-best-group-fraction", type=float, default=0.0)
    parser.add_argument("--quality-eps", type=float, default=0.0)
    parser.add_argument("--pressure-guard-margin", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph_dirs = (
        [args.graph_dir.expanduser().resolve()]
        if args.graph_dir is not None
        else _parse_graph_dirs(args.graph_dirs)
    )
    out_dir = args.output_dir.resolve()
    seeds = _parse_int_list(args.seeds)
    candidate_budgets = _parse_int_list(args.max_group_candidates)

    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "schema": "leiden_non_monotone_group_escape_smoke.v1",
        "graph_dirs": [str(path.relative_to(REPO_ROOT)) for path in graph_dirs],
        "resolution": float(args.resolution),
        "target_max_weight": float(args.target_max_weight),
        "seeds": seeds,
        "max_group_candidates": candidate_budgets,
        "acceptance": {
            "quality_eps": float(args.quality_eps),
            "rule": "accept polish result only when quality >= baseline + quality_eps",
        },
        "pressure_guard_margin": float(args.pressure_guard_margin),
    }

    for graph_dir in graph_dirs:
        arrays = _load_graph_arrays(graph_dir)
        graph = build_leiden_graph(
            edges_src=arrays.src,
            edges_dst=arrays.dst,
            edges_weight=arrays.weight,
            n_nodes=int(arrays.node_weights.shape[0]),
            node_weights=arrays.node_weights,
        )
        case = _case_name(graph_dir)

        for seed in seeds:
            baseline = _run_baseline(
                graph,
                seed=seed,
                resolution=args.resolution,
                n_iterations=args.baseline_iterations,
                randomness=args.randomness,
            )
            baseline_metrics = _membership_metrics(
                baseline.membership,
                arrays.node_weights,
                target_max_weight=args.target_max_weight,
            )
            candidate_clusters = _candidate_clusters(
                graph,
                baseline.membership,
                arrays.node_weights,
                resolution=args.resolution,
                top_weight_count=args.max_suspect_clusters,
                external_priority_count=args.max_external_priority_clusters,
                min_suspect_weight=args.min_suspect_weight,
                min_doc_weight=args.min_doc_weight,
                min_assigned_fraction=args.min_assigned_fraction,
                min_best_group_fraction=args.min_best_group_fraction,
            )
            candidate_clusters_array = np.asarray(candidate_clusters, dtype=np.uint64)

            for budget in candidate_budgets:
                t0 = time.perf_counter()
                probe = graph.non_monotone_group_escape_probe(
                    baseline.membership,
                    candidate_clusters_array,
                    resolution=args.resolution,
                    max_candidates=budget,
                    polish_iterations=args.polish_iterations,
                    randomness=args.randomness,
                    seed=seed + 5_000,
                    min_doc_weight=args.min_doc_weight,
                    min_assigned_fraction=args.min_assigned_fraction,
                    min_best_group_fraction=args.min_best_group_fraction,
                    quality_eps=args.quality_eps,
                )
                elapsed = time.perf_counter() - t0
                rows.append(
                    _row_for_membership(
                        case=case,
                        graph_dir=graph_dir,
                        policy="non_monotone_group_escape",
                        seed=seed,
                        max_group_candidates=budget,
                        baseline=baseline,
                        membership=probe.membership,
                        quality=probe.quality,
                        accepted=probe.accepted,
                        candidate_clusters=candidate_clusters,
                        candidate_rows=probe.candidate_rows,
                        elapsed_sec=elapsed,
                        node_weights=arrays.node_weights,
                        target_max_weight=args.target_max_weight,
                    )
                )

                guarded_metrics = _membership_metrics(
                    probe.membership,
                    arrays.node_weights,
                    target_max_weight=args.target_max_weight,
                )
                guard_passed = (
                    probe.accepted
                    and guarded_metrics["max_doc_weight_ratio"]
                    <= baseline_metrics["max_doc_weight_ratio"]
                    + args.pressure_guard_margin
                )
                guarded_membership = (
                    probe.membership if guard_passed else baseline.membership
                )
                guarded_quality = probe.quality if guard_passed else baseline.quality
                rows.append(
                    _row_for_membership(
                        case=case,
                        graph_dir=graph_dir,
                        policy="non_monotone_group_escape_pressure_guard",
                        seed=seed,
                        max_group_candidates=budget,
                        baseline=baseline,
                        membership=guarded_membership,
                        quality=guarded_quality,
                        accepted=guard_passed,
                        candidate_clusters=candidate_clusters,
                        candidate_rows=probe.candidate_rows,
                        elapsed_sec=elapsed,
                        node_weights=arrays.node_weights,
                        target_max_weight=args.target_max_weight,
                        pressure_guard_passed=guard_passed,
                    )
                )

    _write_outputs(out_dir, rows, summary)
    print(json.dumps(summary["paths"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
