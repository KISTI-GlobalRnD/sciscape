from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "research"
    / "consensus"
    / "scripts"
    / "analyze_leiden_basin_barrier_aware_pathways.py"
)


def _load_module():
    script_dir = SCRIPT_PATH.parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    spec = importlib.util.spec_from_file_location(
        "analyze_leiden_basin_barrier_aware_pathways_for_test",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_barrier_aware_analysis_reads_profile_dirs(tmp_path: Path):
    module = _load_module()
    input_dir = tmp_path / "profiles"
    profile_dir = input_dir / "p"
    profile_dir.mkdir(parents=True)
    (profile_dir / module.SINGLE_SUMMARY_FILENAME).write_text(
        json.dumps(
            {
                "case": "case",
                "field": 34,
                "method": "cc",
                "pair_id": "p",
                "candidate_index": 2,
                "vanilla_seed": 11,
                "vanilla_randomness": 0.0,
                "vanilla_requested_n_iterations": "10",
                "candidate_support_size": 1,
                "vanilla_support_size": 5,
                "v_only_support_size": 4,
                "vanilla_minus_candidate_quality": 1.0,
            }
        )
    )
    pd.DataFrame(
        [
            {
                "parent_state_id": "q_first:0:root",
                "scoring_policy": "q_first",
                "step_index": 1,
                "unit_id": "q",
                "unit_node_count": 1,
                "candidate_label_closure_extra_count": 0,
                "candidate_progress_fraction": 0.25,
                "incremental_progress_fraction": 0.25,
                "delta_q_immediate": 1.0,
                "raw_barrier_if_chosen": 0.0,
                "q_first_score": 1.0,
                "progress_first_score": 0.25,
                "balanced_score": 0.25,
            },
            {
                "parent_state_id": "q_first:0:root",
                "scoring_policy": "q_first",
                "step_index": 1,
                "unit_id": "hidden",
                "unit_node_count": 2,
                "candidate_label_closure_extra_count": 10,
                "candidate_progress_fraction": 0.5,
                "incremental_progress_fraction": 0.5,
                "delta_q_immediate": -2.0,
                "raw_barrier_if_chosen": 2.0,
                "q_first_score": -2.0,
                "progress_first_score": 0.5,
                "balanced_score": 0.25,
            },
        ]
    ).to_csv(profile_dir / module.FRONTIER_ROWS_FILENAME, index=False)
    pd.DataFrame(
        columns=[
            "state_id",
            "selected_unit_ids",
            "selected_unit_count",
            "flipped_node_count",
        ]
    ).to_csv(profile_dir / module.BEAM_ROWS_FILENAME, index=False)

    summary = module.run_analysis(
        input_dir=input_dir,
        output_dir=tmp_path / "out",
        pair_ids=(),
        max_prefix_rows_per_case=10,
        min_support_progress=0.0,
        barrier_floor=1.0,
    )
    prefix_rows = pd.read_csv(tmp_path / "out" / module.PREFIX_ROWS_FILENAME)
    case_rows = pd.read_csv(tmp_path / "out" / module.CASE_ROWS_FILENAME)

    assert summary["profile_count"] == 1
    assert "hidden" in set(prefix_rows["unit_id"])
    assert case_rows.loc[0, "q_greedy_miss_rows"] == 1
