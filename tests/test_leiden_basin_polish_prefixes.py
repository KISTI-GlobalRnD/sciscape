from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "research"
    / "consensus"
    / "scripts"
    / "leiden_basin"
    / "operator_probes"
    / "polish_elbow"
    / "evaluate_leiden_basin_polish_prefixes.py"
)


def _load_module():
    script_dir = SCRIPT_PATH.parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    spec = importlib.util.spec_from_file_location(
        "evaluate_leiden_basin_polish_prefixes_for_test",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_select_prefix_rows_filters_and_takes_top_per_case():
    module = _load_module()
    rows = pd.DataFrame(
        [
            {
                "pair_id": "a",
                "barrier_aware_score": 0.1,
                "support_progress_fraction": 0.5,
                "peak_raw_barrier": 1.0,
            },
            {
                "pair_id": "a",
                "barrier_aware_score": 0.2,
                "support_progress_fraction": 0.1,
                "peak_raw_barrier": 2.0,
            },
            {
                "pair_id": "b",
                "barrier_aware_score": 0.3,
                "support_progress_fraction": 0.2,
                "peak_raw_barrier": 1.0,
            },
        ]
    )

    selected = module.select_prefix_rows(
        rows,
        pair_ids=("a",),
        top_prefixes_per_case=1,
    )

    assert selected["pair_id"].tolist() == ["a"]
    assert selected["barrier_aware_score"].tolist() == [0.2]
