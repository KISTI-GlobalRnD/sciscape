"""Tests for Leiden hysteresis graph materialization from edge parquet."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import polars as pl


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "research" / "consensus" / "scripts"
SCRIPT_PATH = Path(__file__).resolve().parents[1] / "research/consensus/scripts/leiden_basin/hysteresis/materialize_leiden_hysteresis_graphs.py"


def _load_script(module_name: str):
    if str(SCRIPT_PATH.parent) not in sys.path:
        sys.path.insert(0, str(SCRIPT_PATH.parent))
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_discovers_fallback_embedding_names(tmp_path):
    module = _load_script("materialize_leiden_hysteresis_graphs_discover")
    edge_dir = tmp_path / "field_34"
    edge_dir.mkdir()
    _write_edges(edge_dir / "bc_cosine.parquet", [("a", "b", 1.0)])
    _write_edges(edge_dir / "emb_prx_full_knn30.parquet", [("b", "c", 2.0)])

    paths = module.discover_layer_paths(edge_dir)

    assert paths["bc_cosine"].name == "bc_cosine.parquet"
    assert paths["emb_knn"].name == "emb_prx_full_knn30.parquet"


def test_materialize_graphs_writes_monitor_binary_contract(tmp_path):
    module = _load_script("materialize_leiden_hysteresis_graphs_contract")
    edge_root = tmp_path / "linktype_edges_gcc"
    edge_dir = edge_root / "field_12"
    edge_dir.mkdir(parents=True)
    _write_edges(edge_dir / "bc_cosine.parquet", [("a", "b", 1.0), ("b", "c", 0.5)])
    _write_edges(edge_dir / "cc_cosine.parquet", [("c", "d", 0.7)])
    _write_edges(edge_dir / "dc_fractional.parquet", [("a", "d", 0.4)])
    _write_edges(edge_dir / "emb_full_knn30.parquet", [("a", "c", 0.9), ("b", "d", 0.8)])

    rows = module.materialize_graphs(
        fields=[12],
        methods=["bc_cosine", "citation_embedding"],
        edge_roots=[edge_root],
        output_dir=tmp_path / "graphs",
        top_k=0,
    )

    assert len(rows) == 2
    graph_dir = Path(rows[0]["graph_dir"])
    for name in ("src.u32.bin", "dst.u32.bin", "weight.f64.bin", "node_weights.f64.bin"):
        assert (graph_dir / name).exists()
    src = np.memmap(graph_dir / "src.u32.bin", dtype=np.uint32, mode="r")
    weights = np.memmap(graph_dir / "node_weights.f64.bin", dtype=np.float64, mode="r")
    assert src.shape[0] == 2
    assert weights.shape[0] == 3
    manifest = pl.read_csv(tmp_path / "graphs" / "graph_manifest.csv")
    assert set(manifest["method"].to_list()) == {"bc_cosine", "citation_embedding"}


def test_materialize_graphs_accepts_src_dst_weight_schema(tmp_path):
    module = _load_script("materialize_leiden_hysteresis_graphs_src_dst")
    edge_root = tmp_path / "linktype_edges"
    edge_dir = edge_root / "field_34"
    edge_dir.mkdir(parents=True)
    _write_src_dst_edges(edge_dir / "bc_cosine.parquet", [(0, 1, 1.0), (1, 2, 0.5)])

    rows = module.materialize_graphs(
        fields=[34],
        methods=["bc_cosine"],
        edge_roots=[edge_root],
        output_dir=tmp_path / "graphs",
    )

    assert len(rows) == 1
    graph_dir = Path(rows[0]["graph_dir"])
    assert (graph_dir / "src.u32.bin").exists()
    assert rows[0]["n_nodes"] == 3


def _write_edges(path: Path, rows: list[tuple[str, str, float]]) -> None:
    pl.DataFrame(
        {
            "uid1": [row[0] for row in rows],
            "uid2": [row[1] for row in rows],
            "rel_sum2": [row[2] for row in rows],
        }
    ).write_parquet(path)


def _write_src_dst_edges(path: Path, rows: list[tuple[int, int, float]]) -> None:
    pl.DataFrame(
        {
            "src": [row[0] for row in rows],
            "dst": [row[1] for row in rows],
            "weight": [row[2] for row in rows],
        }
    ).write_parquet(path)
