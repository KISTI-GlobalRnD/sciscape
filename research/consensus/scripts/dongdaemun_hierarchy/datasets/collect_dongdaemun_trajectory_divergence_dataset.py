"""Collect pilot Dongdaemun trajectory traces for boundary divergence analysis.

The collector is intentionally orchestration-only. It runs a supplied command
once per case/policy with trajectory and candidate trace environment variables
set, so existing experiment entry points can be reused without embedding data
loading assumptions here.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence
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


@dataclass(frozen=True)
class PolicySpec:
    name: str
    env: dict[str, str]

DEFAULT_POLICIES = (
    PolicySpec("online", {}),
    PolicySpec(
        "local_shake_trace",
        {
            "SCISCAPE_ADAPTIVE_LOCAL_SHAKE_MODE": "trace_only",
            "SCISCAPE_ADAPTIVE_LOCAL_SHAKE_ARMS": "near_tie_refinement,resolution_up,resolution_down,seed_local_refinement",
            "SCISCAPE_ADAPTIVE_LOCAL_SHAKE_RESOLUTION_UP_MULTIPLIERS": "1.02",
            "SCISCAPE_ADAPTIVE_LOCAL_SHAKE_RESOLUTION_DOWN_MULTIPLIERS": "0.98",
            "SCISCAPE_ADAPTIVE_LOCAL_SHAKE_MAX_ARMS_PER_PARENT": "2",
            "SCISCAPE_ADAPTIVE_LOCAL_SHAKE_MAX_CANDIDATES_PER_PARENT": "4",
            "SCISCAPE_ADAPTIVE_LOCAL_SHAKE_NEAR_TIE_MARGIN_PARENT_WEIGHT": "1e-4",
            "SCISCAPE_ADAPTIVE_LOCAL_SHAKE_NEAR_TIE_RANDOMNESS": "0.05",
            "SCISCAPE_ADAPTIVE_LOCAL_SHAKE_SEED_PERTURBATIONS": "1",
        },
    ),
    PolicySpec(
        "local_shake_qf_replace",
        {
            "SCISCAPE_ADAPTIVE_LOCAL_SHAKE_MODE": "qf_replace",
            "SCISCAPE_ADAPTIVE_LOCAL_SHAKE_ARMS": "near_tie_refinement,resolution_up,resolution_down,seed_local_refinement",
            "SCISCAPE_ADAPTIVE_LOCAL_SHAKE_RESOLUTION_UP_MULTIPLIERS": "1.02",
            "SCISCAPE_ADAPTIVE_LOCAL_SHAKE_RESOLUTION_DOWN_MULTIPLIERS": "0.98",
            "SCISCAPE_ADAPTIVE_LOCAL_SHAKE_MAX_ARMS_PER_PARENT": "2",
            "SCISCAPE_ADAPTIVE_LOCAL_SHAKE_MAX_CANDIDATES_PER_PARENT": "4",
            "SCISCAPE_ADAPTIVE_LOCAL_SHAKE_NEAR_TIE_MARGIN_PARENT_WEIGHT": "1e-4",
            "SCISCAPE_ADAPTIVE_LOCAL_SHAKE_NEAR_TIE_RANDOMNESS": "0.05",
            "SCISCAPE_ADAPTIVE_LOCAL_SHAKE_SEED_PERTURBATIONS": "1",
        },
    ),
    PolicySpec(
        "local_shake_pressure_guarded",
        {
            "SCISCAPE_ADAPTIVE_LOCAL_SHAKE_MODE": "pressure_guarded",
            "SCISCAPE_ADAPTIVE_LOCAL_SHAKE_ARMS": "near_tie_refinement,resolution_up,resolution_down,seed_local_refinement",
            "SCISCAPE_ADAPTIVE_LOCAL_SHAKE_RESOLUTION_UP_MULTIPLIERS": "1.02",
            "SCISCAPE_ADAPTIVE_LOCAL_SHAKE_RESOLUTION_DOWN_MULTIPLIERS": "0.98",
            "SCISCAPE_ADAPTIVE_LOCAL_SHAKE_MAX_ARMS_PER_PARENT": "2",
            "SCISCAPE_ADAPTIVE_LOCAL_SHAKE_MAX_CANDIDATES_PER_PARENT": "4",
            "SCISCAPE_ADAPTIVE_LOCAL_SHAKE_NEAR_TIE_MARGIN_PARENT_WEIGHT": "1e-4",
            "SCISCAPE_ADAPTIVE_LOCAL_SHAKE_NEAR_TIE_RANDOMNESS": "0.05",
            "SCISCAPE_ADAPTIVE_LOCAL_SHAKE_SEED_PERTURBATIONS": "1",
        },
    ),
    PolicySpec("r05_total1_all", {"SCISCAPE_LEIDEN_RANDOMNESS": "0.05"}),
    PolicySpec("r10_total1_node", {"SCISCAPE_LEIDEN_RANDOMNESS": "0.10"}),
    PolicySpec(
        "near_tie_probe",
        {
            "SCISCAPE_ADAPTIVE_NEAR_TIE_PROBE_MODE": "candidate",
            "SCISCAPE_ADAPTIVE_NEAR_TIE_MARGIN_PARENT_WEIGHT": "1e-4",
            "SCISCAPE_ADAPTIVE_NEAR_TIE_RANDOMNESS": "0.05",
            "SCISCAPE_ADAPTIVE_NEAR_TIE_MAX_DECISIONS_PER_PARENT": "8",
        },
    ),
)

def load_cases(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))

def build_trace_env(
    *,
    base_env: Mapping[str, str],
    output_dir: Path,
    case_id: str,
    policy: PolicySpec,
    epoch: str,
) -> dict[str, str]:
    run_id = f"{case_id}:{policy.name}"
    env = dict(base_env)
    env.update(policy.env)
    env.update(
        {
            "SCISCAPE_DDM_TRAJECTORY_TRACE_PATH": str(output_dir / "trajectory_trace.jsonl"),
            "SCISCAPE_DDM_TRAJECTORY_TRACE_RUN_ID": run_id,
            "SCISCAPE_DDM_TRAJECTORY_TRACE_EPOCH": epoch,
            "SCISCAPE_DDM_CANDIDATE_TRACE_PATH": str(output_dir / "candidate_trace.jsonl"),
            "SCISCAPE_DDM_CANDIDATE_TRACE_RUN_ID": run_id,
            "SCISCAPE_DDM_CANDIDATE_TRACE_EPOCH": epoch,
        }
    )
    return env

def collect_dongdaemun_trajectory_divergence_dataset_for_test(
    *,
    cases: Sequence[Mapping[str, str]],
    output_dir: Path,
    command: Sequence[str],
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> list[dict[str, str]]:
    runner = runner or subprocess.run
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, str]] = []
    for case in cases:
        case_id = str(case.get("case_id") or case.get("id") or len(manifest_rows))
        for policy in DEFAULT_POLICIES:
            epoch = f"{case_id}-{policy.name}"
            env = build_trace_env(
                base_env=os.environ,
                output_dir=output_dir,
                case_id=case_id,
                policy=policy,
                epoch=epoch,
            )
            completed = runner(
                list(command),
                env=env,
                cwd=case.get("cwd") or None,
                text=True,
                capture_output=True,
                check=False,
            )
            manifest_rows.append(
                {
                    "case_id": case_id,
                    "policy": policy.name,
                    "run_id": f"{case_id}:{policy.name}",
                    "returncode": str(completed.returncode),
                }
            )
    manifest_path = output_dir / "trajectory_collection_manifest.csv"
    with manifest_path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["case_id", "policy", "run_id", "returncode"]
        )
        writer.writeheader()
        writer.writerows(manifest_rows)
    return manifest_rows

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if not args.command:
        parser.error("command is required after --")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    rows = collect_dongdaemun_trajectory_divergence_dataset_for_test(
        cases=load_cases(args.cases_csv),
        output_dir=args.output_dir,
        command=command,
    )
    failed = sum(1 for row in rows if row["returncode"] != "0")
    return 1 if failed else 0

if __name__ == "__main__":
    raise SystemExit(main())
