"""Run blind A/B boundary review across fields and k values."""

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
_SCRIPT_PATHS = [REPO_ROOT, SCRIPT_ROOT]
_SCRIPT_PATHS.extend(path for path in SCRIPT_ROOT.rglob("*") if path.is_dir())
for _script_path in reversed(_SCRIPT_PATHS):
    _script_path_str = str(_script_path)
    if _script_path_str not in sys.path:
        sys.path.insert(0, _script_path_str)


from _common import save_json

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

DEFAULT_FIELDS = ("field_15", "field_12")

def parse_k_values(spec: str) -> list[int]:
    """Parse comma-separated integers and inclusive ranges like ``6,12-30``."""
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

def _resolve_edge_dir(data_root: Path, field: str) -> Path | None:
    direct = data_root / field
    nested = direct / "edges"
    if nested.exists():
        return nested
    if direct.exists():
        return direct
    return None

def _resolve_abstract_path(abstract_root: Path, field: str) -> Path | None:
    candidate = abstract_root / field / "works_text.parquet"
    return candidate if candidate.exists() else None

def main() -> None:
    parser = argparse.ArgumentParser(description="Run boundary-review A/B grid")
    parser.add_argument("edge_root", type=Path, help="Root directory with field edge subdirectories")
    parser.add_argument("abstract_root", type=Path, help="Root directory with field metadata subdirectories")
    parser.add_argument("--fields", type=str, default=",".join(DEFAULT_FIELDS))
    parser.add_argument("--k-values", type=str, default="6,30")
    parser.add_argument("--method-a", type=str, default="sum")
    parser.add_argument("--method-b", type=str, default="consensus")
    parser.add_argument("--target-pct", type=float, default=3.0)
    parser.add_argument("--min-size", type=int, default=10)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--n-cases", type=int, default=24)
    parser.add_argument("--n-neighbors", type=int, default=8)
    parser.add_argument("--boundary-quantile", type=float, default=0.9)
    parser.add_argument("--max-group-jaccard", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-only", action="store_true")
    parser.add_argument("--secondary-checks", action="store_true")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("-o", "--output", type=Path, default=Path("results"))
    args = parser.parse_args()

    fields = [field.strip() for field in args.fields.split(",") if field.strip()]
    k_values = parse_k_values(args.k_values)
    args.output.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output / "boundary_review_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    script = Path(__file__).with_name("run_boundary_review.py")
    summary_rows: list[dict] = []
    runs: dict[str, dict[str, dict]] = {}

    for field in fields:
        edge_dir = _resolve_edge_dir(args.edge_root, field)
        abstract_path = _resolve_abstract_path(args.abstract_root, field)
        if edge_dir is None or abstract_path is None:
            log.warning("Skipping %s: missing edge or abstract input", field)
            continue

        runs[field] = {}
        for effective_k in k_values:
            run_field = f"{field}_k{effective_k:02d}_{args.method_a}_vs_{args.method_b}"
            out_path = raw_dir / f"{run_field}_boundary_review.json"
            if out_path.exists() and not args.overwrite:
                log.info("\n[%s] effective_k=%d (resume existing)", field, effective_k)
            else:
                cmd = [
                    sys.executable,
                    str(script),
                    str(edge_dir),
                    str(abstract_path),
                    "--field",
                    run_field,
                    "--method-a",
                    args.method_a,
                    "--method-b",
                    args.method_b,
                    "--top-k",
                    str(effective_k),
                    "--target-pct",
                    str(args.target_pct),
                    "--min-size",
                    str(args.min_size),
                    "--n-seeds",
                    str(args.n_seeds),
                    "--n-cases",
                    str(args.n_cases),
                    "--n-neighbors",
                    str(args.n_neighbors),
                    "--boundary-quantile",
                    str(args.boundary_quantile),
                    "--max-group-jaccard",
                    str(args.max_group_jaccard),
                    "--seed",
                    str(args.seed),
                    "-o",
                    str(raw_dir),
                ]
                if args.sample_only:
                    cmd.append("--sample-only")
                if args.secondary_checks:
                    cmd.append("--secondary-checks")
                if args.model:
                    cmd.extend(["--model", args.model])
                log.info("\n[%s] effective_k=%d", field, effective_k)
                subprocess.run(cmd, check=True)

            payload = json.loads(out_path.read_text(encoding="utf-8"))
            runs[field][str(effective_k)] = payload
            summary = payload.get("summary", {})
            comparison = summary.get("comparison", {})
            row = {
                "field": field,
                "effective_k": effective_k,
                "method_a": args.method_a,
                "method_b": args.method_b,
                "n_candidate_cases": payload.get("n_candidate_cases", 0),
                "n_reviewed_cases": summary.get("n_reviewed_cases", 0),
                "method_a_wins": comparison.get("method_a_wins"),
                "method_b_wins": comparison.get("method_b_wins"),
                "ties_or_invalid": comparison.get("ties_or_invalid"),
                "method_a_win_rate": comparison.get("method_a_win_rate"),
                "method_b_win_rate": comparison.get("method_b_win_rate"),
                "method_a_win_rate_no_ties": comparison.get("method_a_win_rate_no_ties"),
                "method_b_win_rate_no_ties": comparison.get("method_b_win_rate_no_ties"),
            }
            summary_rows.append(row)

    out_path = args.output / "boundary_review_summary.json"
    save_json(
        {
            "fields": fields,
            "k_values": k_values,
            "method_a": args.method_a,
            "method_b": args.method_b,
            "target_pct": args.target_pct,
            "min_size": args.min_size,
            "n_seeds": args.n_seeds,
            "n_cases": args.n_cases,
            "sample_only": args.sample_only,
            "secondary_checks": args.secondary_checks,
            "raw_dir": str(raw_dir),
            "summary": summary_rows,
            "runs": runs,
        },
        out_path,
    )
    log.info("\nSaved → %s", out_path)

if __name__ == "__main__":
    main()
