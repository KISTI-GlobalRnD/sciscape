from __future__ import annotations

import importlib.util
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sciscape.clustering.leiden_rust import RUST_AVAILABLE


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "research"
    / "consensus"
    / "scripts"
    / "leiden_basin"
    / "basin_signatures"
    / "trajectory_failure"
    / "collect_leiden_vanilla_reachability_sweep.py"
)


def _load_module():
    script_dir = SCRIPT_PATH.parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    spec = importlib.util.spec_from_file_location(
        "collect_leiden_vanilla_reachability_sweep_for_test",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_tiny_graph(tmp_path: Path) -> Path:
    graph_dir = tmp_path / "graph"
    graph_dir.mkdir()
    np.asarray([0, 2, 0, 0, 1, 1], dtype=np.uint32).tofile(graph_dir / "src.u32.bin")
    np.asarray([1, 3, 2, 3, 2, 3], dtype=np.uint32).tofile(graph_dir / "dst.u32.bin")
    np.asarray([10.0, 10.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64).tofile(
        graph_dir / "weight.f64.bin"
    )
    np.ones(4, dtype=np.float64).tofile(graph_dir / "node_weights.f64.bin")
    return graph_dir


def test_canonical_partition_signature_ignores_label_names():
    module = _load_module()

    left = np.asarray([10, 10, 7, 7, 9], dtype=np.uint64)
    right = np.asarray([1, 1, 4, 4, 2], dtype=np.uint64)
    different = np.asarray([1, 4, 4, 4, 2], dtype=np.uint64)

    assert module.canonical_partition_signature(left) == module.canonical_partition_signature(
        right
    )
    assert module.canonical_partition_signature(left) != module.canonical_partition_signature(
        different
    )


def test_basin_sketch_stats_records_endpoint_and_support_sketches():
    module = _load_module()

    baseline = np.asarray([0, 0, 1, 1], dtype=np.uint64)
    membership = np.asarray([0, 1, 1, 1], dtype=np.uint64)
    sketch_nodes = np.asarray([0, 1, 2, 3], dtype=np.uint32)

    stats = module.basin_sketch_stats(
        baseline=baseline,
        membership=membership,
        sketch_nodes=sketch_nodes,
    )

    assert stats["sketch_status"] == "ok"
    assert stats["p5_changed_nodes_vs_baseline"] == 1
    assert stats["p5_changed_fraction_vs_baseline"] == 0.25
    assert stats[module.ALIGNMENT_ERROR_NODE_COUNT_COLUMN] == 1
    assert stats[module.ALIGNMENT_ERROR_FRACTION_COLUMN] == 0.25
    assert stats[module.SKETCH_HASH_COLUMN] == module.hash_u32_sequence(sketch_nodes)
    assert stats[module.SKETCH_BASELINE_COLUMN] == "0;0;1;1"
    assert stats[module.SKETCH_MEMBERSHIP_COLUMN] == "0;1;1;1"
    assert stats[module.ALIGNED_CHANGED_SUPPORT_NODE_COUNT_COLUMN] == 1
    assert stats[module.CHANGED_SUPPORT_COLUMN] == "1"
    assert stats[module.ALIGNED_CHANGED_SUPPORT_NODES_COLUMN] == "1"


def test_compatible_sketch_nodes_uses_candidate_group_neighborhood():
    module = _load_module()

    arrays = SimpleNamespace(
        src=np.asarray([0, 2, 0, 0, 1, 1], dtype=np.uint32),
        dst=np.asarray([1, 3, 2, 3, 2, 3], dtype=np.uint32),
        weight=np.asarray([10.0, 10.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64),
    )
    baseline = np.asarray([0, 0, 1, 1], dtype=np.uint64)
    candidate_rows = pd.DataFrame(
        [
            {
                "case": "graph",
                "candidate_index": 0,
                "source_cluster": 0,
                "target_cluster": 1,
                module.SKETCH_HASH_COLUMN: module.hash_u32_sequence(
                    np.asarray([0, 1, 2, 3], dtype=np.uint32)
                ),
            }
        ]
    )

    sketch_nodes, context = module.compatible_sketch_nodes(
        arrays=arrays,
        baseline_membership=baseline,
        node_weights=np.ones(4, dtype=np.float64),
        candidate_rows=candidate_rows,
    )

    np.testing.assert_array_equal(sketch_nodes, np.asarray([0, 1, 2, 3], dtype=np.uint32))
    assert context["sketch_context_status"] == "ok"
    assert context["sketch_context_reconstructed_candidates"] == 1
    assert context["sketch_context_hash_matches_candidate"] is True


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust backend required")
def test_collect_sweep_tiny_graph_writes_vanilla_basin_rows(tmp_path):
    module = _load_module()
    graph_dir = _write_tiny_graph(tmp_path)
    manifest = tmp_path / "portfolio_batch_cases.csv"
    pd.DataFrame(
        [
            {
                "case_slug": "tiny",
                "field": 1,
                "method": "tiny",
                "graph_dir": str(graph_dir),
                "status": "completed",
            }
        ]
    ).to_csv(manifest, index=False)
    targets = tmp_path / "target_rows.csv"
    pd.DataFrame(
        [
            {
                "case": graph_dir.name,
                "field": 1,
                "method": "tiny",
                "target_class": "material_winner",
            }
        ]
    ).to_csv(targets, index=False)

    payload = module.collect_sweep(
        case_manifest=manifest,
        target_rows_path=targets,
        output_dir=tmp_path / "out",
        seeds=(11,),
        randomness_values=(0.0,),
        n_iterations_values=module._parse_n_iterations_values("1,convergence"),
        target_classes={"material_winner"},
        fields=set(),
        methods=set(),
        resolution=0.000001,
    )

    rows = pd.read_csv(payload["paths"]["rows"])
    assert payload["run_count"] == 2
    assert set(rows["requested_n_iterations"]) == {"1", "convergence"}
    assert rows["p5_basin_signature"].fillna("").str.len().gt(0).all()
    assert rows["comparison_scope"].unique().tolist() == ["exact_signature_only"]
    assert Path(payload["paths"]["report"]).exists()
