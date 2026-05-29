"""Sweep effective-k budgets for one field and aggregate comparison results."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from pathlib import Path
import sys

REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").exists()
)
SCRIPT_ROOT = REPO_ROOT / "research/consensus/scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from _common import layer_provenance, load_layer_paths, save_json, select_best_single_result, validate_field_embedding_contract

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def parse_k_values(spec: str) -> list[int]:
    """Parse comma-separated integers and inclusive ranges like ``1-30``."""
    values: set[int] = set()
    for part in spec.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            lo_s, hi_s = token.split("-", 1)
            lo = int(lo_s)
            hi = int(hi_s)
            if lo > hi:
                lo, hi = hi, lo
            values.update(range(lo, hi + 1))
        else:
            values.add(int(token))
    ordered = sorted(v for v in values if v > 0)
    if not ordered:
        raise ValueError(f"No positive k values parsed from {spec!r}")
    return ordered


def _select_result(results: list[dict], method: str) -> dict | None:
    return next((item for item in results if item.get("method") == method), None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep effective-k budgets for one field")
    parser.add_argument("edge_dir", type=Path, help="Directory with edge parquet files")
    parser.add_argument("--field", type=str, required=True)
    parser.add_argument("--k-values", type=str, default="1-30", help="Comma-separated integers and ranges")
    parser.add_argument("--target-pct", type=float, default=3.0)
    parser.add_argument("--min-size", type=int, default=10)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true", help="Recompute existing per-k runs")
    parser.add_argument("-o", "--output", type=Path, default=Path("results"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    layer_paths = load_layer_paths(args.edge_dir)
    validate_field_embedding_contract(args.field, layer_paths)
    raw_dir = args.output / f"{args.field}_k_sweep_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    script = Path(__file__).with_name("run_comparison.py")
    k_values = parse_k_values(args.k_values)
    sweep_rows: list[dict] = []
    raw_runs: dict[str, dict] = {}

    for effective_k in k_values:
        run_field = f"{args.field}_k{effective_k:02d}"
        run_path = raw_dir / f"{run_field}_comparison.json"
        if run_path.exists() and not args.overwrite:
            log.info("\n[%s] effective_k=%d (resume existing)", args.field, effective_k)
            payload = json.loads(run_path.read_text(encoding="utf-8"))
        else:
            cmd = [
                sys.executable,
                str(script),
                str(args.edge_dir),
                "--field",
                run_field,
                "--effective-k",
                str(effective_k),
                "--target-pct",
                str(args.target_pct),
                "--min-size",
                str(args.min_size),
                "--n-seeds",
                str(args.n_seeds),
                "-o",
                str(raw_dir),
            ]
            log.info("\n[%s] effective_k=%d", args.field, effective_k)
            subprocess.run(cmd, check=True)
            payload = json.loads(run_path.read_text(encoding="utf-8"))
        raw_runs[str(effective_k)] = payload
        results = payload.get("results", [])
        best_single = select_best_single_result(results)
        citation = _select_result(results, "citation_consensus")
        all_cons = _select_result(results, "all_consensus")

        row = {
            "field": args.field,
            "effective_k": effective_k,
            "best_single": best_single,
            "citation_consensus": citation,
            "all_consensus": all_cons,
            "best_method": None,
            "citation_gain_vs_best_single": None,
            "all_gain_vs_best_single": None,
        }
        if best_single:
            row["best_method"] = best_single["method"]
            if citation:
                row["citation_gain_vs_best_single"] = citation["ami_mean"] - best_single["ami_mean"]
            if all_cons:
                row["all_gain_vs_best_single"] = all_cons["ami_mean"] - best_single["ami_mean"]
        sweep_rows.append(row)

    out_path = args.output / f"{args.field}_k_sweep.json"
    save_json(
        {
            "field": args.field,
            "edge_dir": str(args.edge_dir),
            "protocol": "candidate_budget_matched",
            "k_values": k_values,
            "target_pct": args.target_pct,
            "min_size": args.min_size,
            "n_seeds": args.n_seeds,
            "overwrite": args.overwrite,
            "raw_dir": str(raw_dir),
            "summary": sweep_rows,
            "runs": raw_runs,
            **layer_provenance(layer_paths),
        },
        out_path,
    )
    log.info("\nSaved → %s", out_path)


if __name__ == "__main__":
    main()
