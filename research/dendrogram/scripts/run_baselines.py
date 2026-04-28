"""Run the available dendrogram research baselines for one field."""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run dendrogram research baselines")
    parser.add_argument("edge_path", type=Path)
    parser.add_argument("--field", type=str, required=True)
    parser.add_argument("--min-size", type=int, default=30)
    parser.add_argument("--target-pct", type=float, default=3.0)
    parser.add_argument("--n-levels", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-o", "--output", type=Path, default=Path("results"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    script_dir = Path(__file__).parent

    jobs = [
        {
            "method": "leiden_merge",
            "cmd": [
                sys.executable,
                str(script_dir / "run_leiden_merge.py"),
                str(args.edge_path),
                "--field",
                args.field,
                "--min-size",
                str(args.min_size),
                "--target-pct",
                str(args.target_pct),
                "-o",
                str(args.output),
            ],
        },
        {
            "method": "hybrid_hierarchy",
            "cmd": [
                sys.executable,
                str(script_dir / "run_hybrid.py"),
                str(args.edge_path),
                "--field",
                args.field,
                "--n-levels",
                str(args.n_levels),
                "--min-size",
                str(args.min_size),
                "--target-pct",
                str(args.target_pct),
                "--seed",
                str(args.seed),
                "-o",
                str(args.output),
            ],
        },
        {
            "method": "hybrid_optimal_cut",
            "cmd": [
                sys.executable,
                str(script_dir / "run_optimal_cut.py"),
                str(args.edge_path),
                "--field",
                args.field,
                "--nano-min-size",
                str(args.min_size),
                "--target-pct",
                str(args.target_pct),
                "--seed",
                str(args.seed),
                "-o",
                str(args.output),
            ],
        },
        {
            "method": "cut_ablation",
            "cmd": [
                sys.executable,
                str(script_dir / "run_cut_ablation.py"),
                str(args.edge_path),
                "--field",
                args.field,
                "--nano-min-size",
                str(args.min_size),
                "--target-pct",
                str(args.target_pct),
                "--seed",
                str(args.seed),
                "-o",
                str(args.output),
            ],
        },
    ]

    executed: list[dict] = []
    for job in jobs:
        log.info("\nRunning %s", job["method"])
        subprocess.run(job["cmd"], check=True)
        executed.append({"method": job["method"], "status": "completed"})

    skipped = []
    if importlib.util.find_spec("sknetwork") is None:
        skipped.append({"method": "paris_dp", "reason": "scikit-network not installed"})
    if importlib.util.find_spec("graph_tool") is None:
        skipped.append({"method": "nested_sbm", "reason": "graph-tool not installed"})
    skipped.append({"method": "recursive_split", "reason": "runner not implemented in research script set"})

    payload = {
        "field": args.field,
        "edge_path": str(args.edge_path),
        "min_size": args.min_size,
        "target_pct": args.target_pct,
        "n_levels": args.n_levels,
        "seed": args.seed,
        "executed": executed,
        "skipped": skipped,
    }
    out_path = args.output / f"{args.field}_baseline_manifest.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log.info("\nSaved → %s", out_path)


if __name__ == "__main__":
    main()
