"""Run effective-k sweeps across multiple fields and aggregate differences."""

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

FIELDS = ["field_15", "field_12", "field_18", "field_26", "field_29", "field_30", "field_34"]

def _best_row(rows: list[dict], key: str) -> dict | None:
    usable = [row for row in rows if row.get(key) is not None]
    if not usable:
        return None
    return max(usable, key=lambda row: row[key])

def _resolve_edge_dir(data_root: Path, field: str) -> Path | None:
    direct = data_root / field
    nested = direct / "edges"
    if nested.exists():
        return nested
    if direct.exists():
        return direct
    return None

def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-field effective-k sweep")
    parser.add_argument("data_root", type=Path, help="Root directory with field_XX/edges/ subdirs")
    parser.add_argument("--fields", type=str, default=",".join(FIELDS), help="Comma-separated field list")
    parser.add_argument("--k-values", type=str, default="1-30")
    parser.add_argument("--target-pct", type=float, default=3.0)
    parser.add_argument("--min-size", type=int, default=10)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true", help="Recompute existing per-field sweeps")
    parser.add_argument("-o", "--output", type=Path, default=Path("results"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    fields = [item.strip() for item in args.fields.split(",") if item.strip()]
    script = Path(__file__).with_name("run_effective_k_sweep.py")

    aggregate_runs: dict[str, dict] = {}
    field_summary: list[dict] = []

    for field in fields:
        edge_dir = _resolve_edge_dir(args.data_root, field)
        if edge_dir is None:
            log.warning("Skipping %s: no edge directory under %s", field, args.data_root)
            continue

        cmd = [
            sys.executable,
            str(script),
            str(edge_dir),
            "--field",
            field,
            "--k-values",
            args.k_values,
            "--target-pct",
            str(args.target_pct),
            "--min-size",
            str(args.min_size),
            "--n-seeds",
            str(args.n_seeds),
            *(["--overwrite"] if args.overwrite else []),
            "-o",
            str(args.output),
        ]
        log.info("\n%s", "=" * 60)
        log.info("Field sweep: %s", field)
        log.info("%s", "=" * 60)
        subprocess.run(cmd, check=True)

        run_path = args.output / f"{field}_k_sweep.json"
        payload = json.loads(run_path.read_text(encoding="utf-8"))
        aggregate_runs[field] = payload
        rows = payload.get("summary", [])

        best_citation = _best_row(rows, "citation_gain_vs_best_single")
        best_all = _best_row(rows, "all_gain_vs_best_single")
        best_single = _best_row(
            [
                {
                    "effective_k": row["effective_k"],
                    "best_single_ami": row["best_single"]["ami_mean"],
                    "best_single_method": row["best_single"]["method"],
                }
                for row in rows
                if row.get("best_single")
            ],
            "best_single_ami",
        )

        field_summary.append(
            {
                "field": field,
                "best_single_peak": best_single,
                "best_citation_gain": best_citation,
                "best_all_gain": best_all,
            }
        )

    out_path = args.output / "cross_field_k_sweep_summary.json"
    save_json(
        {
            "fields": fields,
            "k_values": args.k_values,
            "target_pct": args.target_pct,
            "min_size": args.min_size,
            "n_seeds": args.n_seeds,
            "overwrite": args.overwrite,
            "field_summary": field_summary,
            "runs": aggregate_runs,
        },
        out_path,
    )
    log.info("\nSaved → %s", out_path)

if __name__ == "__main__":
    main()
