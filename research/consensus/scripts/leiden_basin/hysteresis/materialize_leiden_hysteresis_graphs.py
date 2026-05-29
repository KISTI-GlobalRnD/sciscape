#!/usr/bin/env python3
"""Materialize edge-layer parquet inputs as Leiden hysteresis graph dirs.

The work-acceleration monitor consumes compact binary graph directories:
``src.u32.bin``, ``dst.u32.bin``, ``weight.f64.bin``, and
``node_weights.f64.bin``.  This script builds those directories from the
standard consensus edge parquet layout so multifidelity perturbation checks can
cover more fields and layer combinations than the prepared smoke graphs.
"""

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
_SCRIPT_PATHS = [REPO_ROOT, SCRIPT_ROOT]
_SCRIPT_PATHS.extend(path for path in SCRIPT_ROOT.rglob("*") if path.is_dir())
for _script_path in reversed(_SCRIPT_PATHS):
    _script_path_str = str(_script_path)
    if _script_path_str not in sys.path:
        sys.path.insert(0, _script_path_str)


import numpy as np
import polars as pl

SCRIPT_DIR = Path(__file__).resolve().parent
from _common import combine_layers, load_layer_paths, layer_provenance  # noqa: E402

DEFAULT_EDGE_ROOTS = (
    REPO_ROOT / "data/linktype_edges_gcc",
    REPO_ROOT / "data/linktype_edges",
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "research/consensus/results/adaptive_refinement/"
    "leiden_hysteresis_exception_detector_graphs_20260514"
)

METHOD_LAYER_SETS = {
    "bc_cosine": ("bc_cosine",),
    "cc_cosine": ("cc_cosine",),
    "emb_knn": ("emb_knn",),
    "citation_all": ("bc_cosine", "cc_cosine", "dc_fractional"),
    "citation_embedding": ("bc_cosine", "cc_cosine", "dc_fractional", "emb_knn"),
}

def _parse_csv(value: str) -> list[str]:
    out = [part.strip() for part in value.split(",") if part.strip()]
    if not out:
        raise ValueError("expected at least one comma-separated value")
    return out

def _parse_fields(value: str) -> list[int]:
    return [int(part) for part in _parse_csv(value)]

def _parse_edge_roots(value: str | None) -> list[Path]:
    if value is None:
        return [path.resolve() for path in DEFAULT_EDGE_ROOTS]
    return [Path(part).expanduser().resolve() for part in _parse_csv(value)]

def _field_edge_dir(field: int, edge_roots: list[Path]) -> Path:
    relative = Path(f"field_{int(field)}")
    for root in edge_roots:
        candidate = root / relative
        if candidate.exists():
            return candidate
    roots = ", ".join(str(root) for root in edge_roots)
    raise FileNotFoundError(f"missing edge dir for field {field}: searched {roots}")

def _preferred_embedding_path(edge_dir: Path) -> Path | None:
    candidates = [
        *sorted(edge_dir.glob("emb*_textfilt*.parquet")),
        edge_dir / "emb_full_knn30.parquet",
        edge_dir / "emb_prx_full_knn30.parquet",
        edge_dir / "emb_prx_knn30.parquet",
        edge_dir / "emb_mpnet_knn30.parquet",
        *sorted(edge_dir.glob("emb*_knn30.parquet")),
    ]
    seen: set[Path] = set()
    for path in candidates:
        path = path.resolve()
        if path in seen:
            continue
        seen.add(path)
        if path.exists():
            return path
    return None

def discover_layer_paths(edge_dir: Path) -> dict[str, Path]:
    """Resolve canonical layer paths, including older embedding file names."""
    layer_paths = load_layer_paths(edge_dir)
    if "emb_knn" not in layer_paths:
        emb = _preferred_embedding_path(edge_dir)
        if emb is not None:
            layer_paths["emb_knn"] = emb
    return layer_paths

def _read_edge_layer(path: Path) -> pl.DataFrame:
    frame = pl.read_parquet(path)
    if {"uid1", "uid2", "rel_sum2"} <= set(frame.columns):
        return frame.select("uid1", "uid2", "rel_sum2")
    if {"src", "dst", "weight"} <= set(frame.columns):
        return frame.rename({"src": "uid1", "dst": "uid2", "weight": "rel_sum2"}).select(
            "uid1",
            "uid2",
            "rel_sum2",
        )
    raise ValueError(
        f"{path} must contain uid1/uid2/rel_sum2 or src/dst/weight columns; "
        f"found {frame.columns}"
    )

def _load_method_edges(
    edge_dir: Path,
    method: str,
    *,
    strategy: str,
    top_k: int,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    if method not in METHOD_LAYER_SETS:
        raise ValueError(f"unknown method {method!r}; expected one of {sorted(METHOD_LAYER_SETS)}")
    layer_paths = discover_layer_paths(edge_dir)
    required = METHOD_LAYER_SETS[method]
    missing = [name for name in required if name not in layer_paths]
    if missing:
        raise FileNotFoundError(
            f"{edge_dir} missing layers for {method}: {', '.join(missing)}"
        )
    selected_paths = {name: layer_paths[name] for name in required}
    layers = {name: _read_edge_layer(path) for name, path in selected_paths.items()}
    if len(layers) == 1:
        edges = next(iter(layers.values())).select("uid1", "uid2", "rel_sum2")
        combine_metadata: dict[str, Any] = {"strategy": "single_layer", "top_k": None}
    else:
        edges, _metric_layers = combine_layers(layers, strategy=strategy, top_k=top_k)
        combine_metadata = {"strategy": strategy, "top_k": int(top_k)}
    return edges, {
        "method": method,
        "required_layers": list(required),
        "n_input_layers": len(layers),
        "layer_provenance": layer_provenance(selected_paths),
        **combine_metadata,
    }

def _write_graph_dir(edges: pl.DataFrame, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    uid1 = edges["uid1"].cast(pl.Utf8)
    uid2 = edges["uid2"].cast(pl.Utf8)
    uids = pl.concat([uid1, uid2]).unique().sort().to_list()
    uid_to_idx = {uid: idx for idx, uid in enumerate(uids)}
    src = np.fromiter((uid_to_idx[uid] for uid in uid1.to_list()), dtype=np.uint32)
    dst = np.fromiter((uid_to_idx[uid] for uid in uid2.to_list()), dtype=np.uint32)
    weight = np.ascontiguousarray(edges["rel_sum2"].to_numpy(), dtype=np.float64)
    n_nodes = len(uids)
    np.ascontiguousarray(src, dtype=np.uint32).tofile(output_dir / "src.u32.bin")
    np.ascontiguousarray(dst, dtype=np.uint32).tofile(output_dir / "dst.u32.bin")
    weight.tofile(output_dir / "weight.f64.bin")
    np.ones(int(n_nodes), dtype=np.float64).tofile(output_dir / "node_weights.f64.bin")
    pl.DataFrame(
        {
            "node_idx": np.arange(int(n_nodes), dtype=np.uint32),
            "uid": uids,
        }
    ).write_parquet(output_dir / "node_manifest.parquet")
    return {
        "graph_dir": str(output_dir),
        "n_nodes": int(n_nodes),
        "n_edges": int(len(weight)),
    }

def _csv_safe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    safe_rows: list[dict[str, Any]] = []
    for row in rows:
        safe: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, (dict, list, tuple)):
                safe[key] = json.dumps(value, sort_keys=True)
            else:
                safe[key] = value
        safe_rows.append(safe)
    return safe_rows

def _graph_dir_complete(graph_dir: Path) -> bool:
    required = (
        "src.u32.bin",
        "dst.u32.bin",
        "weight.f64.bin",
        "node_weights.f64.bin",
        "graph_metadata.json",
    )
    return all((graph_dir / name).exists() for name in required)

def materialize_graphs(
    *,
    fields: list[int],
    methods: list[str],
    edge_roots: list[Path],
    output_dir: Path,
    strategy: str = "consensus",
    top_k: int = 30,
    skip_missing: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for field in fields:
        edge_dir = _field_edge_dir(field, edge_roots)
        edge_root_label = edge_dir.parent.name
        sample = f"field{int(field)}_gcc_emb_full_knn30"
        if edge_root_label == "linktype_edges":
            sample = f"field{int(field)}_all_edges"
        for method in methods:
            graph_dir = output_dir / sample / method
            if _graph_dir_complete(graph_dir):
                rows.append(json.loads((graph_dir / "graph_metadata.json").read_text(encoding="utf-8")))
                continue
            try:
                edges, metadata = _load_method_edges(
                    edge_dir,
                    method,
                    strategy=strategy,
                    top_k=top_k,
                )
            except FileNotFoundError:
                if skip_missing:
                    continue
                raise
            graph_metadata = _write_graph_dir(edges, graph_dir)
            row = {
                "field": int(field),
                "sample": sample,
                "edge_dir": str(edge_dir),
                "edge_root": edge_root_label,
                **metadata,
                **graph_metadata,
            }
            rows.append(row)
            (graph_dir / "graph_metadata.json").write_text(
                json.dumps(row, indent=2, sort_keys=True),
                encoding="utf-8",
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "graph_manifest.csv"
    if rows:
        pl.DataFrame(_csv_safe_rows(rows)).write_csv(manifest)
    else:
        pl.DataFrame({"field": [], "method": [], "graph_dir": []}).write_csv(manifest)
    (output_dir / "graph_manifest.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return rows

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fields", type=str, default="12,15,18,26,30,34")
    parser.add_argument(
        "--methods",
        type=str,
        default="bc_cosine,cc_cosine,emb_knn,citation_all,citation_embedding",
    )
    parser.add_argument("--edge-roots", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--strategy", type=str, default="consensus")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--skip-missing", action="store_true")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    rows = materialize_graphs(
        fields=_parse_fields(args.fields),
        methods=_parse_csv(args.methods),
        edge_roots=_parse_edge_roots(args.edge_roots),
        output_dir=args.output_dir,
        strategy=args.strategy,
        top_k=int(args.top_k),
        skip_missing=bool(args.skip_missing),
    )
    print(json.dumps({"n_graphs": len(rows), "output_dir": str(args.output_dir)}, indent=2))

if __name__ == "__main__":
    main()
