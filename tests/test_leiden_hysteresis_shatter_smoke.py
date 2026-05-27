import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_smoke_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "research/consensus/scripts/run_leiden_hysteresis_shatter_smoke.py"
    )
    spec = importlib.util.spec_from_file_location(
        "leiden_hysteresis_shatter_smoke", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cluster_weights_accepts_uint32_membership():
    smoke = _load_smoke_module()
    weights = smoke._cluster_weights(
        np.asarray([0, 1, 1], dtype=np.uint32),
        np.asarray([1.0, 2.0, 3.0], dtype=np.float64),
    )

    np.testing.assert_array_equal(weights, np.asarray([1.0, 5.0]))


def test_hysteresis_smoke_output_contains_required_columns(tmp_path):
    smoke = _load_smoke_module()
    baseline = smoke.BaselineResult(
        seed=11,
        quality=1.0,
        n_clusters=2,
        elapsed_sec=0.01,
        membership=np.asarray([0, 0, 1], dtype=np.uint64),
    )
    candidate_rows = [
        {
            "source_cluster": 0,
            "target_cluster": 1,
            "group_kind": "best",
            "group_count": 1,
            "group_weight": 1.0,
            "group_to_target_weight": 2.0,
            "group_move_delta_q": -0.1,
            "group_split_delta_q": -0.2,
            "recommended_for_split_repair": True,
            "priority": 0.5,
            "pre_polish_delta_q": -0.1,
            "post_polish_delta_q": 0.2,
        }
    ]
    row = smoke._row_for_membership(
        case="synthetic",
        graph_dir=smoke.REPO_ROOT,
        policy="non_monotone_group_escape",
        seed=11,
        max_group_candidates=1,
        baseline=baseline,
        membership=np.asarray([0, 1, 1], dtype=np.uint64),
        quality=1.2,
        accepted=True,
        candidate_clusters=[0],
        candidate_rows=candidate_rows,
        elapsed_sec=0.02,
        node_weights=np.ones(3, dtype=np.float64),
        target_max_weight=2.0,
    )
    summary = {"schema": "test"}
    smoke._write_outputs(tmp_path, [row], summary)

    with (tmp_path / "hysteresis_shatter_smoke_rows.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    required = {
        "policy",
        "seed",
        "max_group_candidates",
        "pre_polish_delta_q",
        "post_polish_delta_q",
        "nmi_vs_baseline",
        "ari_vs_baseline",
        "max_doc_weight_ratio",
        "source_cluster",
        "target_cluster",
        "group_count",
        "group_weight",
        "group_to_target_weight",
        "elapsed_sec",
    }
    assert rows
    assert required.issubset(rows[0])
