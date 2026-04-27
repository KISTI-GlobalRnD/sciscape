"""E4: Cross-field generalization for consensus clustering."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

from _common import save_json, select_best_single_result

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

FIELDS = ["field_15", "field_12", "field_18", "field_26", "field_29", "field_30", "field_34"]


def _select_result(results: list[dict], method: str) -> dict | None:
    return next((item for item in results if item.get("method") == method), None)


def main() -> None:
    parser = argparse.ArgumentParser(description="E4: cross-field generalization")
    parser.add_argument("data_root", type=Path, help="Root directory with field_XX/edges/ subdirs")
    parser.add_argument("-o", "--output", type=Path, default=Path("results"))
    parser.add_argument("--target-pct", type=float, default=3.0)
    parser.add_argument("--effective-k", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=None, help="Legacy fixed per-layer top-k override")
    parser.add_argument("--min-size", type=int, default=10)
    parser.add_argument("--n-seeds", type=int, default=5)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_name("run_comparison.py")

    for field in FIELDS:
        edge_dir = args.data_root / field / "edges"
        if not edge_dir.exists():
            log.warning("Skipping %s: %s not found", field, edge_dir)
            continue

        log.info("\n%s", "=" * 60)
        log.info("Field: %s", field)
        log.info("%s", "=" * 60)
        cmd = [
            sys.executable,
            str(script),
            str(edge_dir),
            "--field",
            field,
            "--target-pct",
            str(args.target_pct),
            "--min-size",
            str(args.min_size),
            "--n-seeds",
            str(args.n_seeds),
            "-o",
            str(args.output),
        ]
        if args.top_k is None:
            cmd.extend(["--effective-k", str(args.effective_k)])
        else:
            cmd.extend(["--top-k", str(args.top_k)])
        subprocess.run(cmd, check=True)

    aggregate: dict[str, dict] = {}
    summary_rows: list[dict] = []
    for field in FIELDS:
        result_path = args.output / f"{field}_comparison.json"
        if not result_path.exists():
            continue
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        aggregate[field] = payload
        results = payload.get("results", [])
        best_single = select_best_single_result(results)
        bc = _select_result(results, "bc_cosine_only")
        all_cons = _select_result(results, "all_consensus")
        citation_cons = _select_result(results, "citation_consensus")
        chosen = all_cons or citation_cons
        if best_single and chosen:
            summary_rows.append(
                {
                    "field": field,
                    "baseline_method": best_single["method"],
                    "consensus_method": chosen["method"],
                    "best_single_method": best_single["method"],
                    "best_single_ami": best_single["ami_mean"],
                    "consensus_ami": chosen["ami_mean"],
                    "ami_gain_vs_best_single": chosen["ami_mean"] - best_single["ami_mean"],
                    "best_single_clusters": best_single["n_clusters"],
                    "consensus_clusters": chosen["n_clusters"],
                    "cluster_gain_vs_best_single": chosen["n_clusters"] - best_single["n_clusters"],
                    "bc_ami": bc["ami_mean"] if bc else None,
                    "ami_gain": (chosen["ami_mean"] - bc["ami_mean"]) if bc else None,
                    "bc_clusters": bc["n_clusters"] if bc else None,
                    "cluster_gain": (chosen["n_clusters"] - bc["n_clusters"]) if bc else None,
                }
            )

    summary_path = args.output / "cross_field_summary.json"
    save_json(
        {
            "fields": FIELDS,
            "target_pct": args.target_pct,
            "effective_k": args.effective_k,
            "top_k": args.top_k,
            "min_size": args.min_size,
            "n_seeds": args.n_seeds,
            "summary": summary_rows,
            "runs": aggregate,
        },
        summary_path,
    )
    log.info("\nSummary → %s", summary_path)

    if summary_rows:
        log.info("\n%-12s %-16s %-14s %-8s", "Field", "Best single", "Consensus AMI", "Gain")
        log.info("%s", "-" * 60)
        for row in summary_rows:
            log.info(
                "%-12s %-16s %.3f          %+0.3f",
                row["field"],
                f"{row['best_single_method']}={row['best_single_ami']:.3f}",
                row["consensus_ami"],
                row["ami_gain_vs_best_single"],
            )


if __name__ == "__main__":
    main()
