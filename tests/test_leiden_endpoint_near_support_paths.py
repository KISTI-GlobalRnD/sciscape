from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "research"
    / "consensus"
    / "scripts"
    / "analyze_leiden_endpoint_near_support_paths.py"
)


def _load_module():
    script_dir = SCRIPT_PATH.parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    spec = importlib.util.spec_from_file_location(
        "analyze_leiden_endpoint_near_support_paths_for_test",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_match_vanilla_normalizes_numeric_iteration_tokens():
    module = _load_module()
    target = pd.Series(
        {
            "best_sketch_seed": 11.0,
            "best_sketch_randomness": 0.0,
            "best_sketch_requested_n_iterations": 10.0,
        }
    )
    vanilla = pd.DataFrame(
        [
            {
                "seed": 11,
                "randomness": 0.0,
                "requested_n_iterations": "10",
            }
        ]
    )

    matched = module._match_vanilla(target, vanilla)

    assert matched is not None
    assert int(matched["seed"]) == 11


def test_segment_cluster_rows_preserve_nodes_outside_endpoint_sketch():
    module = _load_module()
    rows = module._segment_cluster_rows(
        target=pd.Series(
            {
                "case": "case",
                "field": 34,
                "method": "cc_cosine",
                "candidate_index": 2,
                "target_class": "core_alternative",
            }
        ),
        segment="vanilla_only",
        nodes={10, 11, 99},
        node_to_index={10: 0, 11: 1},
        baseline_labels=np.asarray([1, 1], dtype=np.int64),
        dongdaemun_labels=np.asarray([1, 1], dtype=np.int64),
        vanilla_labels=np.asarray([2, 3], dtype=np.int64),
    )

    assert sum(int(row["node_count"]) for row in rows) == 2
    missing = [row for row in rows if int(row["missing_from_sketch"]) == 1]
    assert len(missing) == 1
