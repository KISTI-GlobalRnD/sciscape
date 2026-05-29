"""Compare Python orchestration and Rust Dongdaemun hierarchy fast paths.

This runner consumes a prepared hierarchy-level graph plus the lower-tail
repaired membership used as Dongdaemun input.  It runs
``run_hierarchy_level_postprocess`` twice on the same cached Rust graph:

- existing Python orchestration backend
- opt-in Rust Dongdaemun fast path

The output is a compact validation bundle for deciding whether later fused
kernels are justified.  It intentionally does not reproduce artifact-writing
trim move CSVs; artifact-writing hierarchy runs remain on the Python backend.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
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

from sciscape.clustering.hierarchy_postprocess import (  # noqa: E402
    HierarchyPostprocessConfig,
    LevelPostprocessResult,
    run_hierarchy_level_postprocess,
)
from sciscape.clustering.leiden_rust import (  # noqa: E402
    RUST_DONGDAEMUN_AVAILABLE,
    build_leiden_graph,
)

DEFAULT_RESULTS_DIR = Path("research/consensus/results/adaptive_refinement")
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_DIR / "rust_dongdaemun_fast_path_validation"
SCHEMA_VERSION = 1

CSV_FIELDS = [
    "sample",
    "backend",
    "supported",
    "unsupported_reason",
    "elapsed_sec",
    "accepted",
    "status",
    "quality_delta",
    "split_repair_exact_delta_q",
    "trim_exact_delta_q",
    "n_oversize_before",
    "n_oversize_after",
    "max_doc_weight_before",
    "max_doc_weight_after",
    "target_max_satisfied",
    "changed_nodes",
    "n_clusters_final",
    "membership_equal_to_python",
    "membership_diff_nodes",
    "rust_audit_status",
    "rust_split_iterations",
    "rust_trim_moves_committed",
]

@dataclass(frozen=True)
class FastPathInput:
    sample: str
    graph_dir: Path
    membership_path: Path
    raw_membership_path: Path | None
    node_weights_path: Path | None
    resolution: float
    target_min_doc_weight: float
    target_max_doc_weight: float
    seed: int
    n_nodes: int | None = None

def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return _rel(value)
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        item = float(value)
        return None if math.isnan(item) else item
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value

def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )

def _repo_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    parsed = Path(path)
    if parsed.is_absolute():
        return parsed
    return REPO_ROOT / parsed

def _rel(path: Path | str | None) -> str | None:
    if path is None:
        return None
    parsed = Path(path)
    try:
        return str(parsed.relative_to(REPO_ROOT))
    except ValueError:
        return str(parsed)

def _safe_float(value: Any, *, name: str) -> float:
    if value is None:
        raise ValueError(f"Missing required float value: {name}")
    return float(value)

def _safe_int(value: Any, *, name: str, default: int | None = None) -> int:
    if value is None:
        if default is None:
            raise ValueError(f"Missing required integer value: {name}")
        return int(default)
    return int(value)

def _load_membership(path: Path) -> np.ndarray:
    table = pq.read_table(path, columns=["node_idx", "cluster"])
    node_idx = table.column("node_idx").combine_chunks().to_numpy(zero_copy_only=False)
    cluster = table.column("cluster").combine_chunks().to_numpy(zero_copy_only=False)
    if node_idx.size and not np.all(node_idx[:-1] <= node_idx[1:]):
        cluster = cluster[np.argsort(node_idx, kind="stable")]
    return np.asarray(cluster, dtype=np.uint64)

def _load_node_weights(path: Path | None, n_nodes: int) -> np.ndarray:
    if path is not None and path.exists():
        weights = np.fromfile(path, dtype=np.float64)
        if int(weights.shape[0]) != int(n_nodes):
            raise ValueError(
                f"node weight count mismatch: expected {n_nodes}, got {weights.shape[0]}"
            )
        return np.asarray(weights, dtype=np.float64)
    return np.ones(int(n_nodes), dtype=np.float64)

def _resolve_input_from_summary(
    summary_path: Path,
    *,
    sample: str | None = None,
    resolution: float | None = None,
    target_min_doc_weight: float | None = None,
    target_max_doc_weight: float | None = None,
    seed: int | None = None,
) -> FastPathInput:
    summary = _read_json(summary_path)
    paths = summary.get("paths", {})
    graph_dir = _repo_path(paths.get("graph_dir") or summary.get("graph_dir"))
    membership_path = _repo_path(paths.get("membership") or summary.get("membership"))
    raw_membership_path = _repo_path(
        paths.get("raw_membership")
        or paths.get("raw_membership_path")
        or summary.get("raw_membership")
        or summary.get("raw_membership_path")
    )
    if graph_dir is None:
        raise ValueError(f"Missing graph_dir in {summary_path}")
    if membership_path is None:
        raise ValueError(f"Missing membership path in {summary_path}")
    node_weights_path = _repo_path(
        paths.get("node_weights")
        or paths.get("node_weights_path")
        or summary.get("node_weights")
        or summary.get("node_weights_path")
    )
    if node_weights_path is None:
        node_weights_path = graph_dir / "node_weights.f64.bin"
    return FastPathInput(
        sample=str(sample or summary.get("sample") or summary_path.parent.name),
        graph_dir=graph_dir,
        membership_path=membership_path,
        raw_membership_path=raw_membership_path,
        node_weights_path=node_weights_path,
        resolution=_safe_float(
            resolution if resolution is not None else summary.get("resolution"),
            name="resolution",
        ),
        target_min_doc_weight=_safe_float(
            target_min_doc_weight
            if target_min_doc_weight is not None
            else summary.get("target_min_doc_weight") or summary.get("min_size"),
            name="target_min_doc_weight",
        ),
        target_max_doc_weight=_safe_float(
            target_max_doc_weight
            if target_max_doc_weight is not None
            else summary.get("target_max_doc_weight"),
            name="target_max_doc_weight",
        ),
        seed=_safe_int(seed if seed is not None else summary.get("seed"), name="seed", default=42),
        n_nodes=None if summary.get("n_nodes") is None else int(summary["n_nodes"]),
    )

def _resolve_explicit_input(args: argparse.Namespace) -> FastPathInput:
    graph_dir = _repo_path(args.graph_dir)
    membership_path = _repo_path(args.membership)
    if graph_dir is None or membership_path is None:
        raise ValueError("--graph-dir and --membership are required without --prepare-summary")
    node_weights_path = _repo_path(args.node_weights)
    if node_weights_path is None:
        node_weights_path = graph_dir / "node_weights.f64.bin"
    return FastPathInput(
        sample=str(args.sample or membership_path.parent.name),
        graph_dir=graph_dir,
        membership_path=membership_path,
        raw_membership_path=_repo_path(args.raw_membership),
        node_weights_path=node_weights_path,
        resolution=_safe_float(args.resolution, name="resolution"),
        target_min_doc_weight=_safe_float(
            args.target_min_doc_weight,
            name="target_min_doc_weight",
        ),
        target_max_doc_weight=_safe_float(
            args.target_max_doc_weight,
            name="target_max_doc_weight",
        ),
        seed=_safe_int(args.seed, name="seed", default=42),
        n_nodes=None if args.n_nodes is None else int(args.n_nodes),
    )

def _load_graph(input_cfg: FastPathInput, node_weights: np.ndarray) -> Any:
    graph_dir = input_cfg.graph_dir
    required = [
        graph_dir / "src.u32.bin",
        graph_dir / "dst.u32.bin",
        graph_dir / "weight.f64.bin",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing graph sidecar files: "
            + ", ".join(str(path) for path in missing)
        )
    src = np.memmap(graph_dir / "src.u32.bin", dtype=np.uint32, mode="r")
    dst = np.memmap(graph_dir / "dst.u32.bin", dtype=np.uint32, mode="r")
    weight = np.memmap(graph_dir / "weight.f64.bin", dtype=np.float64, mode="r")
    return build_leiden_graph(
        edges_src=src,
        edges_dst=dst,
        edges_weight=weight,
        n_nodes=int(node_weights.shape[0]),
        node_weights=np.asarray(node_weights, dtype=np.float64),
    )

def _graph_supports_rust_dongdaemun(graph: Any) -> bool:
    if not RUST_DONGDAEMUN_AVAILABLE:
        return False
    inner_graph = getattr(graph, "graph", None)
    return callable(getattr(graph, "dongdaemun_refine", None)) and callable(
        getattr(inner_graph, "dongdaemun_refine", None)
    )

def _nested_float(payload: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return float(default)
        current = current[key]
    return float(current)

def _nested_int(payload: dict[str, Any], *keys: str, default: int = 0) -> int:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return int(default)
        current = current[key]
    return int(current)

def _nested_bool(payload: dict[str, Any], *keys: str, default: bool = False) -> bool:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return bool(default)
        current = current[key]
    return bool(current)

def _membership_diff_summary(
    python_membership: np.ndarray | None,
    membership: np.ndarray | None,
) -> dict[str, Any]:
    if python_membership is None or membership is None:
        return {
            "membership_equal_to_python": None,
            "membership_diff_nodes": None,
        }
    left = np.asarray(python_membership, dtype=np.uint64)
    right = np.asarray(membership, dtype=np.uint64)
    if left.shape != right.shape:
        return {
            "membership_equal_to_python": False,
            "membership_diff_nodes": None,
        }
    diff_nodes = int(np.count_nonzero(left != right))
    return {
        "membership_equal_to_python": diff_nodes == 0,
        "membership_diff_nodes": diff_nodes,
    }

def _flatten_result(
    *,
    sample: str,
    backend: str,
    elapsed_sec: float,
    result: LevelPostprocessResult,
    python_membership: np.ndarray | None = None,
) -> dict[str, Any]:
    oversize = result.oversize_summary
    rust_audit = oversize.get("rust_audit", {})
    diff = _membership_diff_summary(python_membership, result.membership)
    is_rust = backend == "rust_dongdaemun" or result.backend == "rust_dongdaemun"
    return {
        "sample": sample,
        "backend": backend,
        "supported": True,
        "unsupported_reason": "",
        "elapsed_sec": float(elapsed_sec),
        "accepted": bool(result.accepted),
        "status": str(result.status),
        "quality_delta": float(oversize.get("final_exact_delta_q", 0.0)),
        "split_repair_exact_delta_q": float(
            oversize.get("split_repair_exact_delta_q", 0.0)
        ),
        "trim_exact_delta_q": float(oversize.get("trim_exact_delta_q", 0.0)),
        "n_oversize_before": _nested_int(oversize, "before", "n_above_max_doc_weight"),
        "n_oversize_after": _nested_int(oversize, "after", "n_above_max_doc_weight"),
        "max_doc_weight_before": _nested_float(oversize, "before", "max_doc_weight"),
        "max_doc_weight_after": _nested_float(oversize, "after", "max_doc_weight"),
        "target_max_satisfied": _nested_bool(oversize, "target_max_satisfied"),
        "changed_nodes": int(oversize.get("changed_nodes", 0)),
        "n_clusters_final": int(result.final_summary.get("n_clusters", 0)),
        "membership_equal_to_python": diff["membership_equal_to_python"],
        "membership_diff_nodes": diff["membership_diff_nodes"],
        "rust_audit_status": str(rust_audit.get("status", "")) if is_rust else "",
        "rust_split_iterations": len(oversize.get("iterations", [])) if is_rust else "",
        "rust_trim_moves_committed": (
            int(rust_audit.get("trim_moves_committed", 0) or 0) if is_rust else ""
        ),
    }

def _unsupported_row(*, sample: str, backend: str, reason: str) -> dict[str, Any]:
    row = {field: "" for field in CSV_FIELDS}
    row.update(
        {
            "sample": sample,
            "backend": backend,
            "supported": False,
            "unsupported_reason": reason,
        }
    )
    return row

def _run_backend(
    *,
    graph: Any,
    input_cfg: FastPathInput,
    raw_membership: np.ndarray,
    small_membership: np.ndarray,
    node_weights: np.ndarray,
    use_rust_dongdaemun: bool,
    oversize_policy: str,
    apply_iterations: int,
    trim_max_moves_per_cluster: int,
    quality_floor_delta: float,
    python_membership: np.ndarray | None = None,
) -> tuple[dict[str, Any], np.ndarray | None]:
    backend = "rust_dongdaemun" if use_rust_dongdaemun else "python"
    if use_rust_dongdaemun and not _graph_supports_rust_dongdaemun(graph):
        return (
            _unsupported_row(
                sample=input_cfg.sample,
                backend=backend,
                reason="installed Rust graph does not expose Graph.dongdaemun_refine",
            ),
            None,
        )
    config = HierarchyPostprocessConfig(
        enabled=True,
        use_rust_dongdaemun=bool(use_rust_dongdaemun),
        oversize_policy=oversize_policy,  # type: ignore[arg-type]
        apply_iterations=int(apply_iterations),
        quality_floor_delta=float(quality_floor_delta),
        trim_max_moves_per_cluster=int(trim_max_moves_per_cluster),
        write_artifacts=False,
    )
    start = time.perf_counter()
    result = run_hierarchy_level_postprocess(
        graph,
        raw_membership=raw_membership,
        small_membership=small_membership,
        node_weights=node_weights,
        resolution=float(input_cfg.resolution),
        min_doc_weight=float(input_cfg.target_min_doc_weight),
        target_max_doc_weight=float(input_cfg.target_max_doc_weight),
        config=config,
        seed=int(input_cfg.seed),
        output_dir=None,
    )
    elapsed = time.perf_counter() - start
    row = _flatten_result(
        sample=input_cfg.sample,
        backend=backend,
        elapsed_sec=elapsed,
        result=result,
        python_membership=python_membership,
    )
    row["backend"] = result.backend
    return row, np.asarray(result.membership, dtype=np.uint64)

def _comparison_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_backend = {str(row["backend"]): row for row in rows}
    python = by_backend.get("python")
    rust = by_backend.get("rust_dongdaemun")
    if python is None or rust is None or not rust.get("supported", False):
        return {
            "status": "incomplete",
            "reason": rust.get("unsupported_reason") if rust else "missing row",
        }
    return {
        "status": "compared",
        "same_status": python.get("status") == rust.get("status"),
        "same_acceptance": python.get("accepted") == rust.get("accepted"),
        "membership_equal": rust.get("membership_equal_to_python"),
        "membership_diff_nodes": rust.get("membership_diff_nodes"),
        "quality_delta_diff": float(rust["quality_delta"]) - float(python["quality_delta"]),
        "oversize_after_diff": int(rust["n_oversize_after"])
        - int(python["n_oversize_after"]),
        "max_doc_weight_after_diff": float(rust["max_doc_weight_after"])
        - float(python["max_doc_weight_after"]),
        "elapsed_sec_diff": float(rust["elapsed_sec"]) - float(python["elapsed_sec"]),
    }

def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _json_safe(row.get(field, "")) for field in CSV_FIELDS})

def _write_report(
    path: Path,
    *,
    input_cfg: FastPathInput,
    rows: list[dict[str, Any]],
    comparison: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Rust Dongdaemun Fast Path Validation",
        "",
        "This run compares the existing Python hierarchy postprocess orchestration with the opt-in Rust Dongdaemun fast path on the same cached Rust graph and lower-tail repaired membership.",
        "",
        "## Input",
        "",
        f"- Sample: {input_cfg.sample}",
        f"- Graph dir: `{_rel(input_cfg.graph_dir)}`",
        f"- Membership: `{_rel(input_cfg.membership_path)}`",
        f"- Raw membership: `{_rel(input_cfg.raw_membership_path)}`",
        f"- Resolution: {input_cfg.resolution:g}",
        f"- Target min doc weight: {input_cfg.target_min_doc_weight:g}",
        f"- Target max doc weight: {input_cfg.target_max_doc_weight:g}",
        f"- Seed: {input_cfg.seed}",
        "",
        "## Backend Rows",
        "",
        "| backend | supported | status | accepted | quality delta | oversize after | max weight after | elapsed sec |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {backend} | {supported} | {status} | {accepted} | {quality_delta:.6g} | {n_oversize_after} | {max_doc_weight_after:.6g} | {elapsed_sec:.3f} |".format(
                backend=row.get("backend", ""),
                supported=bool(row.get("supported", False)),
                status=row.get("status", "") or row.get("unsupported_reason", ""),
                accepted=bool(row.get("accepted", False)),
                quality_delta=float(row.get("quality_delta") or 0.0),
                n_oversize_after=int(row.get("n_oversize_after") or 0),
                max_doc_weight_after=float(row.get("max_doc_weight_after") or 0.0),
                elapsed_sec=float(row.get("elapsed_sec") or 0.0),
            )
        )
    lines.extend(
        [
            "",
            "## Comparison",
            "",
            f"- Status: {comparison.get('status')}",
            f"- Membership equal: {comparison.get('membership_equal')}",
            f"- Membership diff nodes: {comparison.get('membership_diff_nodes')}",
            f"- Quality delta diff: {comparison.get('quality_delta_diff')}",
            f"- Oversize-after diff: {comparison.get('oversize_after_diff')}",
            f"- Elapsed sec diff: {comparison.get('elapsed_sec_diff')}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")

def run_validation(
    input_cfg: FastPathInput,
    *,
    output_dir: Path,
    oversize_policy: str = "quality_first",
    apply_iterations: int = 4,
    trim_max_moves_per_cluster: int = 100,
    quality_floor_delta: float = 0.0,
) -> dict[str, Any]:
    small_membership = _load_membership(input_cfg.membership_path)
    raw_membership = (
        _load_membership(input_cfg.raw_membership_path)
        if input_cfg.raw_membership_path is not None
        else small_membership.copy()
    )
    n_nodes = int(input_cfg.n_nodes or small_membership.shape[0])
    if int(small_membership.shape[0]) != n_nodes:
        raise ValueError(
            f"membership length mismatch: expected {n_nodes}, got {small_membership.shape[0]}"
        )
    if int(raw_membership.shape[0]) != n_nodes:
        raise ValueError(
            f"raw membership length mismatch: expected {n_nodes}, got {raw_membership.shape[0]}"
        )
    node_weights = _load_node_weights(input_cfg.node_weights_path, n_nodes)
    graph = _load_graph(input_cfg, node_weights)

    python_row, python_membership = _run_backend(
        graph=graph,
        input_cfg=input_cfg,
        raw_membership=raw_membership,
        small_membership=small_membership,
        node_weights=node_weights,
        use_rust_dongdaemun=False,
        oversize_policy=oversize_policy,
        apply_iterations=apply_iterations,
        trim_max_moves_per_cluster=trim_max_moves_per_cluster,
        quality_floor_delta=quality_floor_delta,
        python_membership=None,
    )
    rust_row, rust_membership = _run_backend(
        graph=graph,
        input_cfg=input_cfg,
        raw_membership=raw_membership,
        small_membership=small_membership,
        node_weights=node_weights,
        use_rust_dongdaemun=True,
        oversize_policy=oversize_policy,
        apply_iterations=apply_iterations,
        trim_max_moves_per_cluster=trim_max_moves_per_cluster,
        quality_floor_delta=quality_floor_delta,
        python_membership=python_membership,
    )
    rows = [python_row, rust_row]
    if rust_membership is not None:
        diff = _membership_diff_summary(python_membership, rust_membership)
        rust_row.update(diff)
    comparison = _comparison_summary(rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "fast_path_comparison.csv"
    json_path = output_dir / "fast_path_comparison.json"
    report_path = output_dir / "README.md"
    _write_csv(csv_path, rows)
    payload = {
        "schema": f"rust_dongdaemun_fast_path_validation.v{SCHEMA_VERSION}",
        "input": asdict(input_cfg),
        "config": {
            "oversize_policy": oversize_policy,
            "apply_iterations": int(apply_iterations),
            "trim_max_moves_per_cluster": int(trim_max_moves_per_cluster),
            "quality_floor_delta": float(quality_floor_delta),
            "raw_membership_fallback_used": input_cfg.raw_membership_path is None,
        },
        "rows": rows,
        "comparison": comparison,
        "paths": {
            "csv": csv_path,
            "json": json_path,
            "report": report_path,
        },
    }
    _write_json(json_path, payload)
    _write_report(
        report_path,
        input_cfg=input_cfg,
        rows=rows,
        comparison=comparison,
    )
    return payload

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-summary", type=Path)
    parser.add_argument("--graph-dir", type=Path)
    parser.add_argument("--membership", type=Path)
    parser.add_argument("--raw-membership", type=Path)
    parser.add_argument("--node-weights", type=Path)
    parser.add_argument("--n-nodes", type=int)
    parser.add_argument("--sample")
    parser.add_argument("--resolution", type=float)
    parser.add_argument("--target-min-doc-weight", type=float)
    parser.add_argument("--target-max-doc-weight", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--oversize-policy",
        choices=("quality_first", "hard_cap"),
        default="quality_first",
    )
    parser.add_argument("--apply-iterations", type=int, default=4)
    parser.add_argument("--trim-max-moves-per-cluster", type=int, default=100)
    parser.add_argument("--quality-floor-delta", type=float, default=0.0)
    return parser

def _input_from_args(args: argparse.Namespace) -> FastPathInput:
    if args.prepare_summary is not None:
        return _resolve_input_from_summary(
            args.prepare_summary,
            sample=args.sample,
            resolution=args.resolution,
            target_min_doc_weight=args.target_min_doc_weight,
            target_max_doc_weight=args.target_max_doc_weight,
            seed=args.seed,
        )
    return _resolve_explicit_input(args)

def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    input_cfg = _input_from_args(args)
    payload = run_validation(
        input_cfg,
        output_dir=args.output_dir,
        oversize_policy=args.oversize_policy,
        apply_iterations=args.apply_iterations,
        trim_max_moves_per_cluster=args.trim_max_moves_per_cluster,
        quality_floor_delta=args.quality_floor_delta,
    )
    print(f"Saved fast-path validation outputs to {_rel(Path(payload['paths']['json']))}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
