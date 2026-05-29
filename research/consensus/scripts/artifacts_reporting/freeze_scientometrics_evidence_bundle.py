"""Freeze curated Scientometrics evidence artifacts into one bundle.

The hierarchy-postprocess validation directory contains both paper-facing
evidence and large reproducibility scratch outputs.  This script copies the
curated paper/supplementary artifacts into a dated bundle and writes archive
indices for everything that was intentionally left out of the paper bundle.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable
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


from evaluate_hierarchy_postprocess import DEFAULT_OUTPUT_DIR  # noqa: E402

DEFAULT_BUNDLE_DIR = (
    Path("research/consensus/results")
    / f"scientometrics_evidence_freeze_{date.today():%Y%m%d}"
)

@dataclass(frozen=True)

class ArtifactSpec:
    source_root: str
    source_relative_path: str
    output_relative_path: str
    category: str
    role: str
    required: bool = True

def _paired_table_specs(
    stem: str,
    *,
    output_dir: str,
    category: str,
    role: str,
    source_prefix: str = "",
) -> list[ArtifactSpec]:
    specs: list[ArtifactSpec] = []
    for suffix in (".csv", ".parquet"):
        source_rel = f"{source_prefix}{stem}{suffix}"
        specs.append(
            ArtifactSpec(
                source_root="validation",
                source_relative_path=source_rel,
                output_relative_path=f"{output_dir}/{stem}{suffix}",
                category=category,
                role=role,
            )
        )
    return specs

def _validation_artifact_specs() -> list[ArtifactSpec]:
    specs: list[ArtifactSpec] = [
        ArtifactSpec(
            "validation",
            "README.md",
            "main/reports/source_validation_readme.md",
            "main",
            "source validation overview",
        ),
        ArtifactSpec(
            "validation",
            "paper_evidence_brief.md",
            "main/reports/paper_evidence_brief.md",
            "main",
            "evidence brief",
        ),
        ArtifactSpec(
            "validation",
            "field_expansion_report.md",
            "main/reports/field_expansion_report.md",
            "main",
            "six-field evidence report",
        ),
        ArtifactSpec(
            "validation",
            "actual_next_level_report.md",
            "main/reports/actual_next_level_report.md",
            "main",
            "next-level propagation report",
        ),
        ArtifactSpec(
            "validation",
            "table1_policy_comparison.md",
            "main/tables/table1_policy_comparison.md",
            "main",
            "manuscript table",
        ),
        ArtifactSpec(
            "validation",
            "table2_failure_taxonomy.md",
            "main/tables/table2_failure_taxonomy.md",
            "main",
            "manuscript table",
        ),
    ]

    for figure_name in (
        "figure1_two_stage_pipeline.png",
        "figure2_size_quality_tradeoff.png",
        "figure4_actual_next_level_propagation.png",
        "figure9_field_expansion_source_seed.png",
        "figure10_field_expansion_next_level.png",
        "figure11_semantic_coherence.png",
    ):
        specs.append(
            ArtifactSpec(
                "validation",
                figure_name,
                f"main/figures/{figure_name}",
                "main",
                "manuscript figure",
            )
        )

    for figure_name in (
        "figure3_contraction_precondition.png",
        "figure5_next_level_target_sweep.png",
        "figure6_seed_stability_deltas.png",
        "figure7_source_seed_pilot.png",
        "figure8_source_seed_next_level_propagation.png",
    ):
        specs.append(
            ArtifactSpec(
                "validation",
                figure_name,
                f"supplementary/figures/{figure_name}",
                "supplementary",
                "supporting figure",
            )
        )

    specs.extend(
        [
            ArtifactSpec(
                "validation",
                "next_level_target_sweep_report.md",
                "supplementary/reports/next_level_target_sweep_report.md",
                "supplementary",
                "target sweep report",
            ),
            ArtifactSpec(
                "validation",
                "next_level_seed_sweep_report.md",
                "supplementary/reports/next_level_seed_sweep_report.md",
                "supplementary",
                "seed sweep report",
            ),
            ArtifactSpec(
                "validation",
                "source_seed_sweep_report.md",
                "supplementary/reports/source_seed_sweep_report.md",
                "supplementary",
                "source-seed sweep report",
            ),
            ArtifactSpec(
                "validation",
                "source_seed_next_level_report.md",
                "supplementary/reports/source_seed_next_level_report.md",
                "supplementary",
                "source-seed next-level report",
            ),
            ArtifactSpec(
                "validation",
                "semantic_coherence_report.md",
                "supplementary/reports/semantic_coherence_report.md",
                "supplementary",
                "semantic sanity-check report",
            ),
        ]
    )

    for stem in (
        "policy_comparison",
        "field_expansion_source_seed_quality_first_vs_small_only",
        "field_expansion_source_seed_next_level_quality_first_vs_small_only_summary",
        "field_expansion_field_breakdown",
        "failure_taxonomy_summary",
        "semantic_coherence_quality_first_vs_small_only",
    ):
        specs.extend(
            _paired_table_specs(
                stem,
                output_dir="main/tables",
                category="main",
                role="main analysis table",
            )
        )

    for stem in (
        "hierarchy_postprocess_eval",
        "contraction_effects",
        "actual_contraction_effects",
        "actual_next_level_effects",
        "actual_next_level_policy_comparison",
        "field_expansion_source_seed_effects",
        "field_expansion_source_seed_policy_summary",
        "field_expansion_source_seed_next_level_effects",
        "field_expansion_source_seed_next_level_summary",
        "field_expansion_source_seed_next_level_quality_first_vs_small_only",
        "field_expansion_source_seed_hard_cap_diagnostic_summary",
        "field_expansion_source_seed_hard_cap_diagnostics",
        "field_expansion_source_seed_next_level_hard_cap_diagnostic_summary",
        "field_expansion_source_seed_next_level_hard_cap_diagnostics",
        "source_seed_sweep_effects",
        "source_seed_sweep_policy_summary",
        "source_seed_sweep_quality_first_vs_small_only",
        "source_seed_sweep_hard_cap_diagnostic_summary",
        "source_seed_sweep_hard_cap_diagnostics",
        "source_seed_next_level_effects",
        "source_seed_next_level_policy_summary",
        "source_seed_next_level_quality_first_vs_small_only",
        "source_seed_next_level_quality_first_vs_small_only_summary",
        "source_seed_next_level_hard_cap_diagnostic_summary",
        "source_seed_next_level_hard_cap_diagnostics",
        "next_level_target_sweep_effects",
        "next_level_target_sweep_policy_comparison",
        "next_level_target_sweep_quality_first_vs_small_only",
        "next_level_target_sweep_hard_cap_diagnostic_summary",
        "next_level_target_sweep_hard_cap_diagnostics",
        "next_level_seed_sweep_effects",
        "next_level_seed_sweep_policy_summary",
        "next_level_seed_sweep_quality_first_vs_small_only",
        "next_level_seed_sweep_quality_first_vs_small_only_summary",
        "next_level_seed_sweep_hard_cap_diagnostic_summary",
        "next_level_seed_sweep_hard_cap_diagnostics",
        "failure_taxonomy",
        "semantic_coherence_effects",
        "semantic_coherence_field_breakdown",
        "semantic_coherence_policy_summary",
        "semantic_coherence_cluster_metrics",
    ):
        specs.extend(
            _paired_table_specs(
                stem,
                output_dir="supplementary/tables",
                category="supplementary",
                role="supporting analysis table",
            )
        )

    for json_name in (
        "field_expansion_summary.json",
        "field_expansion_validation_summary.json",
        "semantic_coherence_summary.json",
    ):
        specs.append(
            ArtifactSpec(
                "validation",
                json_name,
                f"supplementary/metadata/{json_name}",
                "supplementary",
                "run metadata",
            )
        )

    same_gamma_prefix = "same_gamma_oversize_extension/"
    specs.append(
        ArtifactSpec(
            "validation",
            f"{same_gamma_prefix}iterative_quality_first_report.md",
            "supplementary/same_gamma_extension/iterative_quality_first_report.md",
            "supplementary",
            "same-gamma extension report",
        )
    )
    for stem in (
        "iterative_quality_first_effects",
        "iterative_quality_first_passes",
        "iterative_quality_first_candidates",
        "iterative_quality_first_vs_current",
        "iterative_quality_first_policy_summary",
    ):
        specs.extend(
            _paired_table_specs(
                stem,
                output_dir="supplementary/same_gamma_extension/tables",
                category="supplementary",
                role="same-gamma extension table",
                source_prefix=same_gamma_prefix,
            )
        )
    specs.append(
        ArtifactSpec(
            "validation",
            f"{same_gamma_prefix}iterative_quality_first_compute_summary.json",
            "supplementary/same_gamma_extension/iterative_quality_first_compute_summary.json",
            "supplementary",
            "same-gamma extension run metadata",
        )
    )

    branch_prefix = "branch_adaptive_quality_first_pilot_clean/"
    specs.extend(
        [
            ArtifactSpec(
                "validation",
                f"{branch_prefix}branch_adaptive_diagnostics_report.md",
                "supplementary/branch_adaptive/branch_adaptive_diagnostics_report.md",
                "supplementary",
                "branch-adaptive diagnostic report",
            ),
            ArtifactSpec(
                "validation",
                f"{branch_prefix}branch_adaptive_compute_summary.json",
                "supplementary/branch_adaptive/branch_adaptive_compute_summary.json",
                "supplementary",
                "branch-adaptive run metadata",
            ),
            ArtifactSpec(
                "validation",
                f"{branch_prefix}figure12_branch_adaptive_tau_sensitivity.png",
                "supplementary/branch_adaptive/figure12_branch_adaptive_tau_sensitivity.png",
                "supplementary",
                "branch-adaptive figure",
            ),
        ]
    )
    for stem in (
        "branch_adaptive_split_candidates",
        "branch_adaptive_parent_summary",
        "branch_adaptive_tau_sensitivity",
        "branch_adaptive_tau_candidate_selection",
        "branch_adaptive_candidate_stability",
        "branch_adaptive_policy_effects",
        "branch_adaptive_quality_first_vs_current",
    ):
        specs.extend(
            _paired_table_specs(
                stem,
                output_dir="supplementary/branch_adaptive/tables",
                category="supplementary",
                role="branch-adaptive diagnostic table",
                source_prefix=branch_prefix,
            )
        )

    return specs

def _doc_artifact_specs() -> list[ArtifactSpec]:
    return [
        ArtifactSpec(
            "docs",
            "scientometrics_manuscript_outline.md",
            "docs/papers/scientometrics_manuscript_outline.md",
            "docs",
            "manuscript skeleton",
        ),
        ArtifactSpec(
            "docs",
            "branch_adaptive_case_study_notes.md",
            "docs/research/leiden_basin/branch_adaptive_case_study_notes.md",
            "docs",
            "branch-adaptive case-study notes",
        ),
        ArtifactSpec(
            "docs",
            "branch_adaptive_quality_first_research_note.md",
            "docs/research/leiden_basin/branch_adaptive_quality_first_research_note.md",
            "docs",
            "branch-adaptive research note",
        ),
        ArtifactSpec(
            "docs",
            "hierarchy_postprocess_research_roadmap.md",
            "docs/research/hierarchy/hierarchy_postprocess_research_roadmap.md",
            "docs",
            "research roadmap",
        ),
        ArtifactSpec(
            "docs",
            "methodology_final_design.md",
            "docs/research/methodology/methodology_final_design.md",
            "docs",
            "methodology note",
            required=False,
        ),
        ArtifactSpec(
            "docs",
            "research_problem_statement.md",
            "docs/research/methodology/research_problem_statement.md",
            "docs",
            "research problem note",
            required=False,
        ),
        ArtifactSpec(
            "docs",
            "code_review_cpm_dendro.md",
            "docs/research/dendrogram/code_review_cpm_dendro.md",
            "docs",
            "CPM/dendrogram technical note",
            required=False,
        ),
    ]

def _artifact_specs() -> list[ArtifactSpec]:
    return _validation_artifact_specs() + _doc_artifact_specs()

def _source_base(spec: ArtifactSpec, validation_dir: Path, docs_dir: Path) -> Path:
    if spec.source_root == "validation":
        return validation_dir
    if spec.source_root == "docs":
        return docs_dir
    raise ValueError(f"Unknown source root: {spec.source_root}")

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

def _copy_artifacts(
    specs: list[ArtifactSpec],
    *,
    validation_dir: Path,
    docs_dir: Path,
    output_dir: Path,
    dry_run: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen_outputs: set[str] = set()
    for spec in specs:
        source = _source_base(spec, validation_dir, docs_dir) / spec.source_relative_path
        dest = output_dir / spec.output_relative_path
        status = "copied"
        size = ""
        digest = ""
        if spec.output_relative_path in seen_outputs:
            raise ValueError(f"Duplicate bundle output path: {spec.output_relative_path}")
        seen_outputs.add(spec.output_relative_path)
        if source.exists():
            stat = source.stat()
            size = int(stat.st_size)
            digest = _sha256(source)
            if not dry_run:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, dest)
        elif spec.required:
            status = "missing_required"
        else:
            status = "missing_optional"
        rows.append(
            {
                "source_root": spec.source_root,
                "source_relative_path": spec.source_relative_path,
                "output_relative_path": spec.output_relative_path,
                "category": spec.category,
                "role": spec.role,
                "status": status,
                "size_bytes": size,
                "sha256": digest,
            }
        )
    return rows

def _all_validation_files(validation_dir: Path) -> list[Path]:
    return sorted(path for path in validation_dir.rglob("*") if path.is_file())

def _archive_rows(
    *,
    validation_dir: Path,
    copied_validation_sources: set[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    all_rows: list[dict[str, object]] = []
    unselected_rows: list[dict[str, object]] = []
    for path in _all_validation_files(validation_dir):
        rel = path.relative_to(validation_dir).as_posix()
        stat = path.stat()
        row = {
            "relative_path": rel,
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "in_frozen_bundle": rel in copied_validation_sources,
        }
        all_rows.append(row)
        if rel not in copied_validation_sources:
            unselected = dict(row)
            unselected["archive_reason"] = "not_in_curated_paper_or_supplementary_set"
            unselected_rows.append(unselected)
    return all_rows, unselected_rows

def _directory_rows(validation_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for directory in sorted(path for path in validation_dir.rglob("*") if path.is_dir()):
        files = [path for path in directory.rglob("*") if path.is_file()]
        if not files:
            continue
        rows.append(
            {
                "relative_dir": directory.relative_to(validation_dir).as_posix(),
                "file_count": len(files),
                "size_bytes": sum(path.stat().st_size for path in files),
            }
        )
    return rows

def _write_readme(
    *,
    output_dir: Path,
    validation_dir: Path,
    docs_dir: Path,
    manifest_rows: list[dict[str, object]],
    unselected_count: int,
    dry_run: bool,
) -> None:
    if dry_run:
        return
    copied = [row for row in manifest_rows if row["status"] == "copied"]
    missing_required = [row for row in manifest_rows if row["status"] == "missing_required"]
    by_category: dict[str, int] = {}
    for row in copied:
        by_category[str(row["category"])] = by_category.get(str(row["category"]), 0) + 1

    payload = {
        "bundle": output_dir.name,
        "validation_dir": validation_dir.as_posix(),
        "docs_dir": docs_dir.as_posix(),
        "copied_file_count": len(copied),
        "missing_required_count": len(missing_required),
        "unselected_validation_file_count": unselected_count,
        "files_by_category": by_category,
    }
    (output_dir / "bundle_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    readme = f"""# Scientometrics Evidence Freeze

This directory is a curated freeze of the hierarchy-postprocess evidence for the
Scientometrics manuscript track.

## Layout

- `main/`: paper-facing reports, tables, and figures.
- `supplementary/`: detailed supporting analyses, diagnostics, and robustness tables.
- `docs/`: manuscript skeleton and research notes.
- `archive/`: indices for validation artifacts that were not copied into the curated
  paper/supplementary set.

## Source Directories

- validation results: `{validation_dir.as_posix()}`
- document notes: `{docs_dir.as_posix()}`

## Counts

- copied files: {len(copied)}
- missing required files: {len(missing_required)}
- unselected validation files indexed in archive: {unselected_count}

See `MANIFEST.csv` for checksums and `bundle_summary.json` for machine-readable
summary metadata.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")

    archive_readme = """# Archive Index

The large validation directory is not physically moved by this bundle script.
Instead, unselected artifacts are indexed here so the paper freeze stays compact
and reproducible while the original run directories remain usable by existing
scripts.

- `original_artifact_index.csv`: every file under the validation directory.
- `unselected_validation_artifacts.csv`: files not copied into `main/`,
  `supplementary/`, or `docs/`.
- `run_directory_index.csv`: aggregate file counts and byte sizes by validation
  subdirectory.
"""
    archive_dir = output_dir / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    (archive_dir / "README.md").write_text(archive_readme, encoding="utf-8")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze curated Scientometrics evidence artifacts into one bundle."
    )
    parser.add_argument(
        "--validation-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Hierarchy-postprocess validation artifact directory.",
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=Path("docs"),
        help="Documentation directory containing manuscript notes.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=DEFAULT_BUNDLE_DIR,
        help="Destination freeze bundle directory.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove an existing output directory before writing the bundle.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the manifest in memory without copying files.",
    )
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    validation_dir = args.validation_dir
    docs_dir = args.docs_dir
    output_dir = args.output_dir

    if not validation_dir.exists():
        raise FileNotFoundError(f"Validation directory does not exist: {validation_dir}")
    if not docs_dir.exists():
        raise FileNotFoundError(f"Docs directory does not exist: {docs_dir}")
    if output_dir.exists() and args.force and not args.dry_run:
        shutil.rmtree(output_dir)
    elif output_dir.exists() and not args.dry_run:
        raise FileExistsError(f"Output directory already exists: {output_dir}")

    specs = _artifact_specs()
    manifest_rows = _copy_artifacts(
        specs,
        validation_dir=validation_dir,
        docs_dir=docs_dir,
        output_dir=output_dir,
        dry_run=args.dry_run,
    )
    copied_validation_sources = {
        str(row["source_relative_path"])
        for row in manifest_rows
        if row["source_root"] == "validation" and row["status"] == "copied"
    }
    all_rows, unselected_rows = _archive_rows(
        validation_dir=validation_dir,
        copied_validation_sources=copied_validation_sources,
    )
    directory_rows = _directory_rows(validation_dir)

    if not args.dry_run:
        _write_csv(
            output_dir / "MANIFEST.csv",
            manifest_rows,
            [
                "source_root",
                "source_relative_path",
                "output_relative_path",
                "category",
                "role",
                "status",
                "size_bytes",
                "sha256",
            ],
        )
        _write_csv(
            output_dir / "archive" / "original_artifact_index.csv",
            all_rows,
            ["relative_path", "size_bytes", "mtime_ns", "in_frozen_bundle"],
        )
        _write_csv(
            output_dir / "archive" / "unselected_validation_artifacts.csv",
            unselected_rows,
            [
                "relative_path",
                "size_bytes",
                "mtime_ns",
                "in_frozen_bundle",
                "archive_reason",
            ],
        )
        _write_csv(
            output_dir / "archive" / "run_directory_index.csv",
            directory_rows,
            ["relative_dir", "file_count", "size_bytes"],
        )
        _write_readme(
            output_dir=output_dir,
            validation_dir=validation_dir,
            docs_dir=docs_dir,
            manifest_rows=manifest_rows,
            unselected_count=len(unselected_rows),
            dry_run=args.dry_run,
        )

    copied_count = sum(1 for row in manifest_rows if row["status"] == "copied")
    missing_required = [
        row for row in manifest_rows if row["status"] == "missing_required"
    ]
    print(f"Bundle directory: {output_dir}")
    print(f"Copied artifacts: {copied_count}")
    print(f"Unselected validation artifacts indexed: {len(unselected_rows)}")
    if missing_required:
        print("Missing required artifacts:")
        for row in missing_required:
            print(f"  - {row['source_root']}:{row['source_relative_path']}")
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
