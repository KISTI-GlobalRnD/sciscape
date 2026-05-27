"""Tests for the resumable Leiden portfolio batch runner."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "research" / "consensus" / "scripts"
BATCH_PATH = SCRIPT_DIR / "run_leiden_hysteresis_portfolio_batch.py"


def _load_script(module_name: str):
    if str(BATCH_PATH.parent) not in sys.path:
        sys.path.insert(0, str(BATCH_PATH.parent))
    spec = importlib.util.spec_from_file_location(module_name, BATCH_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_filter_manifest_honors_fields_methods_and_limit():
    module = _load_script("portfolio_batch_filter")
    manifest = pd.DataFrame(
        [
            _manifest_row(12, "bc_cosine"),
            _manifest_row(12, "emb_knn"),
            _manifest_row(30, "cc_cosine"),
        ]
    )

    filtered = module._filter_manifest(
        manifest,
        fields=[12],
        methods=["emb_knn", "bc_cosine"],
        limit=1,
    )

    assert len(filtered) == 1
    assert int(filtered.iloc[0]["field"]) == 12
    assert filtered.iloc[0]["method"] == "bc_cosine"


def test_run_batch_writes_completion_marker_and_aggregate_cases(tmp_path, monkeypatch):
    module = _load_script("portfolio_batch_run")
    graph_dir = tmp_path / "graph"
    graph_dir.mkdir()
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame([{**_manifest_row(12, "emb_knn"), "graph_dir": str(graph_dir)}]).to_csv(
        manifest_path,
        index=False,
    )
    out_dir = tmp_path / "out"
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        output_index = command.index("--output-dir") + 1
        case_dir = Path(command[output_index])
        pd.DataFrame(
            [
                {
                    "case": "case",
                    "seed": 11,
                    "candidate_budget": 3,
                    "branch": "perturb",
                }
            ]
        ).to_csv(case_dir / "monitor_run_rows.csv", index=False)
        pd.DataFrame(
            [
                {
                    "case": "case",
                    "seed": 11,
                    "candidate_budget": 3,
                    "target_policy": "extra_p5_final",
                    "k_work_saving_pct": 10.0,
                    "net_elapsed_saving_pct": 20.0,
                    "acceleration_role": "work_acceleration_quality_neutral",
                }
            ]
        ).to_csv(case_dir / "work_acceleration_monitor_scorecard.csv", index=False)
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    args = SimpleNamespace(
        graph_manifest=manifest_path,
        output_dir=out_dir,
        fields="12",
        methods="emb_knn",
        candidate_eval_modes="parallel_full_p5_portfolio",
        seeds="11",
        candidate_budgets="3",
        baseline_iterations=10,
        polish_iterations=5,
        prescreen_iterations=1,
        final_iterations=5,
        multifidelity_finalists=1,
        local_merge_summary_mode="compact",
        keep_raw_trajectory=False,
        probe_only=False,
        basin_signatures=False,
        resume=True,
        limit=None,
    )

    module.run_batch(args)
    module.run_batch(args)

    assert len(calls) == 1
    marker_paths = list(out_dir.glob("*/portfolio_batch_case_complete.json"))
    assert len(marker_paths) == 1
    marker = json.loads(marker_paths[0].read_text(encoding="utf-8"))
    assert marker["status"] == "completed"
    cases = pd.read_csv(out_dir / "portfolio_batch_cases.csv")
    assert set(cases["status"]) == {"skipped"}
    scorecard = pd.read_csv(out_dir / "portfolio_batch_scorecard.csv")
    assert set(scorecard["candidate_eval_mode"]) == {"parallel_full_p5_portfolio"}


def test_monitor_command_forwards_probe_only_flag(tmp_path):
    module = _load_script("portfolio_batch_probe_only_command")
    args = SimpleNamespace(
        baseline_iterations=10,
        polish_iterations=5,
        prescreen_iterations=1,
        final_iterations=5,
        multifidelity_finalists=1,
        local_merge_summary_mode="compact",
        keep_raw_trajectory=False,
        probe_only=True,
        basin_signatures=True,
    )

    command = module._monitor_command(
        graph_dir=tmp_path / "graph",
        output_dir=tmp_path / "out",
        mode="parallel_full_p5_portfolio",
        seed=11,
        budget=3,
        args=args,
    )

    assert "--probe-only" in command
    assert "--basin-signatures" in command


def test_parallel_worker_limit_sets_rayon_env(monkeypatch):
    module = _load_script("portfolio_batch_parallel_worker_env")
    monkeypatch.delenv("RAYON_NUM_THREADS", raising=False)
    args = SimpleNamespace(
        max_parallel_candidate_workers=2,
        memory_budget_gb=None,
        estimated_candidate_worker_gb=0.0,
        memory_reserve_gb=16.0,
    )

    env, limit = module._subprocess_env_for_mode("parallel_full_p5_portfolio", args)

    assert limit == 2
    assert env["RAYON_NUM_THREADS"] == "2"
    assert env["SCISCAPE_PARALLEL_CANDIDATE_WORKER_LIMIT"] == "2"


def test_memory_budget_can_cap_parallel_worker_limit():
    module = _load_script("portfolio_batch_memory_worker_limit")
    args = SimpleNamespace(
        max_parallel_candidate_workers=8,
        memory_budget_gb=130.0,
        estimated_candidate_worker_gb=60.0,
        memory_reserve_gb=16.0,
    )

    assert module._parallel_worker_limit(args) == 2


def test_read_csv_if_exists_treats_empty_csv_as_empty_frame(tmp_path):
    module = _load_script("portfolio_batch_empty_csv")
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")

    frame = module._read_csv_if_exists(path)

    assert frame.empty


def _manifest_row(field: int, method: str) -> dict[str, object]:
    return {
        "field": field,
        "method": method,
        "graph_dir": f"/tmp/field{field}/{method}",
        "sample": f"field{field}_gcc_emb_full_knn30",
    }
