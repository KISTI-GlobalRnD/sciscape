"""E4: Cross-field generalization.

Run the same consensus pipeline on all 7 fields and compare patterns.
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

FIELDS = ["field_15", "field_12", "field_18", "field_26", "field_29", "field_30", "field_34"]


def main():
    parser = argparse.ArgumentParser(description="E4: Cross-field generalization")
    parser.add_argument("data_root", type=Path, help="Root directory with field_XX/edges/ subdirs")
    parser.add_argument("-o", "--output", type=Path, default=Path("results"))
    parser.add_argument("--target-pct", type=float, default=3.0)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).parent / "run_comparison.py"

    for field in FIELDS:
        edge_dir = args.data_root / field / "edges"
        if not edge_dir.exists():
            log.warning(f"Skipping {field}: {edge_dir} not found")
            continue

        log.info(f"\n{'='*60}")
        log.info(f"Field: {field}")
        log.info(f"{'='*60}")

        cmd = [
            sys.executable, str(script),
            str(edge_dir),
            "--field", field,
            "--target-pct", str(args.target_pct),
            "-o", str(args.output),
        ]
        subprocess.run(cmd, check=True)

    # Aggregate results
    all_results = {}
    for field in FIELDS:
        result_path = args.output / f"{field}_comparison.json"
        if result_path.exists():
            with open(result_path) as f:
                all_results[field] = json.load(f)

    summary_path = args.output / "cross_field_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info(f"\nSummary → {summary_path}")

    # Print comparison table
    log.info(f"\n{'Field':<12} {'BC AMI':<10} {'Consensus AMI':<15} {'Gain':<8}")
    log.info("-" * 50)
    for field, results in all_results.items():
        bc = next((r for r in results if r.get("layer") == "bc_cosine"), None)
        cons = next((r for r in results if "consensus" in r.get("strategy", "")), None)
        if bc and cons:
            gain = cons["ami_mean"] - bc["ami_mean"]
            log.info(f"{field:<12} {bc['ami_mean']:.3f}     {cons['ami_mean']:.3f}          {gain:+.3f}")


if __name__ == "__main__":
    main()
