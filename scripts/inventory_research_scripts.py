#!/usr/bin/env python3
"""Inventory research scripts before reorganizing script directories."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path


DEFAULT_ROOT = Path("research/consensus/scripts")
SCAN_ROOTS = [
    Path("AGENTS.md"),
    Path("CHANGELOG.md"),
    Path("README.md"),
    Path("docs"),
    Path("research"),
    Path("sciscape"),
    Path("scripts"),
    Path("tests"),
]
IGNORED_DIR_NAMES = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".svn",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "target",
    "workspace",
}
IGNORED_PATH_PREFIXES = {
    "docs/api",
    "research/consensus/results",
    "research/dendrogram/results",
    "research/experiments/results",
    "workspace",
}
TEXT_EXTENSIONS = {
    ".csv",
    ".json",
    ".md",
    ".py",
    ".rst",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
TEXT_FILENAMES = {
    "AGENTS.md",
    "CHANGELOG.md",
    "Dockerfile",
    "Makefile",
    "README.md",
}
MAX_SCAN_BYTES = 2_000_000


@dataclass(frozen=True)
class ScriptRecord:
    path: str
    name: str
    bucket: str
    sub_bucket: str
    detail_bucket: str
    target_path: str
    action: str
    git_state: str
    reference_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classify research scripts and count path references before moving "
            "them into subdirectories."
        )
    )
    parser.add_argument(
        "--root",
        default=str(DEFAULT_ROOT),
        help=f"Script directory to inventory. Defaults to {DEFAULT_ROOT}.",
    )
    parser.add_argument(
        "--format",
        choices=["text", "markdown", "json"],
        default="text",
        help="Output format. Defaults to text.",
    )
    parser.add_argument(
        "--fail-on-unclassified",
        action="store_true",
        help="Exit nonzero if any script falls into the unclassified bucket.",
    )
    parser.add_argument(
        "--top-references",
        type=int,
        default=12,
        help="Number of most-referenced scripts to print in text/markdown mode.",
    )
    return parser.parse_args()


def display_path(path: Path) -> str:
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.as_posix()


def is_ignored(path: Path) -> bool:
    rel = display_path(path)
    if any(part in IGNORED_DIR_NAMES for part in path.parts):
        return True
    return any(rel == prefix or rel.startswith(f"{prefix}/") for prefix in IGNORED_PATH_PREFIXES)


def git_paths(args: list[str]) -> set[str]:
    try:
        proc = subprocess.run(
            ["git", *args],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return set()
    if proc.returncode not in (0, 1):
        return set()
    return {line.strip() for line in proc.stdout.splitlines() if line.strip()}


def git_states(root: Path) -> dict[str, str]:
    root_text = display_path(root)
    tracked = git_paths(["ls-files", "--", root_text])
    untracked = git_paths(["ls-files", "--others", "--exclude-standard", "--", root_text])
    states: dict[str, str] = {}
    states.update({path: "tracked" for path in tracked})
    states.update({path: "untracked" for path in untracked})
    return states


def script_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*.py")
        if path.is_file() and not is_ignored(path)
    )


def action_name(script_name: str) -> str:
    stem = Path(script_name).stem
    if stem.startswith("_"):
        return "helper"
    return stem.split("_", 1)[0]


def classify(script_name: str) -> str:
    stem = Path(script_name).stem
    if stem.startswith("_"):
        return "common"

    if "leiden" in stem or "branch_adaptive" in stem:
        return "leiden_basin"

    if "dongdaemun" in stem or "hierarchy_postprocess" in stem or "cyclic_lookahead" in stem:
        return "dongdaemun_hierarchy"

    if any(token in stem for token in ("rank_shift", "boundary", "taxonomy", "review")):
        return "review_taxonomy"

    if any(
        stem.startswith(prefix)
        for prefix in (
            "aggregate_",
            "export_",
            "fit_",
            "freeze_",
            "generate_",
            "materialize_",
            "prepare_",
            "repair_",
            "score_",
            "summarize_",
        )
    ):
        return "artifacts_reporting"

    if any(
        stem.startswith(prefix)
        for prefix in (
            "analyze_",
            "build_",
            "classify_",
            "collect_",
            "estimate_",
            "evaluate_",
            "probe_",
            "profile_",
            "rank_",
            "run_",
            "screen_",
            "search_",
        )
    ):
        return "consensus_core"

    return "unclassified"


def classify_leiden_sub_bucket(script_name: str) -> str:
    stem = Path(script_name).stem

    if "hysteresis" in stem:
        return "hysteresis"

    if any(
        token in stem
        for token in (
            "direct_pair",
            "pathway",
            "route",
            "transition",
            "tunneling",
            "wall_route",
        )
    ):
        return "transition_routes"

    if any(
        token in stem
        for token in (
            "attachment_margin",
            "branch_candidate",
            "elbow",
            "gate",
            "handle",
            "joint_bundle",
            "polish",
            "post_gate",
            "recovery",
            "selector",
            "target_unit",
        )
    ):
        return "operator_probes"

    if any(
        token in stem
        for token in (
            "cache",
            "join_",
            "materialize",
            "pending_membership",
            "prepare_",
        )
    ):
        return "materialization"

    if any(
        token in stem
        for token in (
            "audit",
            "calibrate",
            "current_results",
            "definition",
            "evidence",
            "field34",
            "freeze",
            "phase1",
            "relation",
            "review",
            "taxonomy",
            "triage",
            "wall",
        )
    ):
        return "evidence_panels"

    return "basin_signatures"


def classify_operator_detail(script_name: str) -> str:
    stem = Path(script_name).stem

    if any(token in stem for token in ("gate_release", "gate_attachment")):
        return "gate_release"

    if any(token in stem for token in ("post_gate", "recovery")):
        return "post_gate_recovery"

    if "joint_bundle" in stem:
        return "joint_bundle"

    if any(token in stem for token in ("aligned_core", "local_handle")):
        return "aligned_core"

    if "attachment" in stem:
        return "attachment_margin"

    if any(token in stem for token in ("polish", "elbow", "target_unit")):
        return "polish_elbow"

    if any(token in stem for token in ("source_screen", "selector_source", "screen_leiden_basin_selector")):
        return "selector_sources"

    return "selector_signals"


def classify_transition_detail(script_name: str) -> str:
    stem = Path(script_name).stem

    if any(token in stem for token in ("review", "triage", "freeze")):
        return "route_reviews"

    if any(token in stem for token in ("tunneling", "barrier", "pathway_debt")):
        return "tunneling_pathways"

    if any(token in stem for token in ("combined_route_gate", "route_gate", "route_wall")):
        return "route_gate_panels"

    if any(token in stem for token in ("direct_pair", "route", "wall")):
        return "route_wall"

    if any(token in stem for token in ("operator", "minimal_pathway")):
        return "transition_operators"

    if any(token in stem for token in ("boundary", "landscape")):
        return "transition_diagnostics"

    return "closure_context"


def classify_basin_signature_detail(script_name: str) -> str:
    stem = Path(script_name).stem

    if any(token in stem for token in ("branch", "random_refinement")):
        return "branch_growth"

    if any(token in stem for token in ("multifidelity", "greedy_failures", "vanilla_reachability")):
        return "trajectory_failure"

    if any(token in stem for token in ("portfolio", "contract")):
        return "portfolio_contracts"

    if any(token in stem for token in ("multibasin", "signature", "threshold", "decision_rules")):
        return "signature_detection"

    if any(token in stem for token in ("endpoint", "ordered_flips", "recomputed")):
        return "endpoint_flips"

    return "local_modes"


def classify_evidence_panel_detail(script_name: str) -> str:
    stem = Path(script_name).stem

    if "audit" in stem:
        return "audits"

    if any(token in stem for token in ("field34", "definition")):
        return "field_eligibility"

    if any(token in stem for token in ("relation", "taxonomy", "stable_ambiguous")):
        return "relation_taxonomy"

    if any(token in stem for token in ("phase1", "wall_protocol", "wall_panel")):
        return "phase_panels"

    if "portfolio_evidence" in stem:
        return "portfolio_evidence"

    return "review_panels"


def classify_dongdaemun_detail(script_name: str) -> str:
    stem = Path(script_name).stem

    if stem.startswith("collect_"):
        return "datasets"

    if any(token in stem for token in ("branch_lookahead", "cyclic", "adaptive_stochastic")):
        return "prototype_runs"

    if "hierarchy_postprocess" in stem and any(token in stem for token in ("seed", "sweep", "expansion")):
        return "postprocess_sweeps"

    if "hierarchy_postprocess" in stem:
        return "postprocess_evaluation"

    if any(token in stem for token in ("refinement", "rust_dongdaemun", "safe_fast")):
        return "refinement_runs"

    if stem.startswith("summarize_"):
        return "trace_summaries"

    return "trajectory_analysis"


def classify_review_detail(script_name: str) -> str:
    stem = Path(script_name).stem

    if "rank_shift" in stem:
        return "rank_shift"

    if "taxonomy" in stem:
        return "taxonomy"

    if "uncertainty" in stem or "reproducibility" in stem or "order_balanced" in stem:
        return "review_uncertainty"

    return "boundary_reviews"


def classify_consensus_core_detail(script_name: str) -> str:
    stem = Path(script_name).stem

    if any(token in stem for token in ("sweep", "cross_field", "noise_ablation")):
        return "sweeps"

    if any(token in stem for token in ("semantic", "same_gamma", "common_case", "sum_noise")):
        return "validation"

    return "baseline_comparisons"


def target_sub_bucket(script_name: str, bucket: str) -> str:
    if bucket == "leiden_basin":
        return classify_leiden_sub_bucket(script_name)
    return ""


def target_detail_bucket(script_name: str, bucket: str, sub_bucket: str) -> str:
    if bucket == "leiden_basin":
        if sub_bucket == "operator_probes":
            return classify_operator_detail(script_name)
        if sub_bucket == "transition_routes":
            return classify_transition_detail(script_name)
        if sub_bucket == "basin_signatures":
            return classify_basin_signature_detail(script_name)
        if sub_bucket == "evidence_panels":
            return classify_evidence_panel_detail(script_name)
        return ""

    if bucket == "dongdaemun_hierarchy":
        return classify_dongdaemun_detail(script_name)

    if bucket == "review_taxonomy":
        return classify_review_detail(script_name)

    if bucket == "consensus_core":
        return classify_consensus_core_detail(script_name)

    return ""


def target_path(root: Path, script_name: str, bucket: str, sub_bucket: str, detail_bucket: str) -> str:
    if bucket == "unclassified":
        return display_path(root / script_name)
    parts = [root, Path(bucket)]
    if sub_bucket:
        parts.append(Path(sub_bucket))
    if detail_bucket:
        parts.append(Path(detail_bucket))
    parts.append(Path(script_name))
    return display_path(Path(*parts))


def iter_scan_files() -> list[Path]:
    files: list[Path] = []
    for root in SCAN_ROOTS:
        if not root.exists() or is_ignored(root):
            continue
        if root.is_file():
            files.append(root)
            continue
        for path in root.rglob("*"):
            if is_ignored(path) or not path.is_file():
                continue
            if path.name not in TEXT_FILENAMES and path.suffix.lower() not in TEXT_EXTENSIONS:
                continue
            try:
                if path.stat().st_size > MAX_SCAN_BYTES:
                    continue
            except OSError:
                continue
            files.append(path)
    return files


def reference_counts(root: Path, scripts: list[Path]) -> dict[str, int]:
    counts = dict.fromkeys((display_path(script) for script in scripts), 0)
    repo_prefix = f"{display_path(root)}/"
    needles = {
        display_path(script): (
            display_path(script),
            f"{repo_prefix}{script.name}",
            f"scripts/{script.name}",
        )
        for script in scripts
    }
    for path in iter_scan_files():
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for rel, patterns in needles.items():
            if any(pattern in text for pattern in patterns):
                counts[rel] += 1
    return counts


def build_records(root: Path) -> list[ScriptRecord]:
    scripts = script_files(root)
    states = git_states(root)
    refs = reference_counts(root, scripts)
    records: list[ScriptRecord] = []
    for script in scripts:
        rel = display_path(script)
        bucket = classify(script.name)
        sub_bucket = target_sub_bucket(script.name, bucket)
        detail_bucket = target_detail_bucket(script.name, bucket, sub_bucket)
        records.append(
            ScriptRecord(
                path=rel,
                name=script.name,
                bucket=bucket,
                sub_bucket=sub_bucket,
                detail_bucket=detail_bucket,
                target_path=target_path(root, script.name, bucket, sub_bucket, detail_bucket),
                action=action_name(script.name),
                git_state=states.get(rel, "unknown"),
                reference_count=refs.get(rel, 0),
            )
        )
    return records


def bucket_counts(records: list[ScriptRecord]) -> Counter[str]:
    return Counter(record.bucket for record in records)


def git_state_counts(records: list[ScriptRecord]) -> Counter[str]:
    return Counter(record.git_state for record in records)


def action_counts(records: list[ScriptRecord]) -> Counter[str]:
    return Counter(record.action for record in records)


def sub_bucket_counts(records: list[ScriptRecord]) -> Counter[str]:
    return Counter(
        f"{record.bucket}/{record.sub_bucket}" if record.sub_bucket else record.bucket
        for record in records
    )


def target_bucket_counts(records: list[ScriptRecord]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in records:
        parts = [record.bucket]
        if record.sub_bucket:
            parts.append(record.sub_bucket)
        if record.detail_bucket:
            parts.append(record.detail_bucket)
        counts["/".join(parts)] += 1
    return counts


def grouped_records(records: list[ScriptRecord]) -> dict[str, list[ScriptRecord]]:
    grouped: dict[str, list[ScriptRecord]] = defaultdict(list)
    for record in records:
        grouped[record.bucket].append(record)
    return dict(sorted(grouped.items()))


def print_text(records: list[ScriptRecord], top_references: int) -> None:
    print(f"Research script inventory: {len(records)} scripts")
    print()
    print("Bucket counts:")
    for bucket, count in bucket_counts(records).most_common():
        print(f"- {bucket}: {count}")
    print()
    print("Target sub-bucket counts:")
    for bucket, count in sub_bucket_counts(records).most_common():
        print(f"- {bucket}: {count}")
    print()
    print("Target detail-bucket counts:")
    for bucket, count in target_bucket_counts(records).most_common():
        print(f"- {bucket}: {count}")
    print()
    print("Git state counts:")
    for state, count in git_state_counts(records).most_common():
        print(f"- {state}: {count}")
    print()
    print("Action prefix counts:")
    for action, count in action_counts(records).most_common():
        print(f"- {action}: {count}")
    print()
    print(f"Top referenced scripts (top {top_references}):")
    for record in sorted(records, key=lambda item: (-item.reference_count, item.name))[:top_references]:
        print(
            f"- {record.reference_count:>3} refs | {record.bucket:<20} | "
            f"{record.path} -> {record.target_path}"
        )


def print_markdown(records: list[ScriptRecord], top_references: int) -> None:
    print("# Research Script Inventory")
    print()
    print(f"Total scripts: **{len(records)}**")
    print()
    print("## Bucket Counts")
    print()
    print("| Bucket | Count |")
    print("|---|---:|")
    for bucket, count in bucket_counts(records).most_common():
        print(f"| `{bucket}` | {count} |")
    print()
    print("## Target Sub-Bucket Counts")
    print()
    print("| Target | Count |")
    print("|---|---:|")
    for bucket, count in sub_bucket_counts(records).most_common():
        print(f"| `{bucket}` | {count} |")
    print()
    print("## Target Detail-Bucket Counts")
    print()
    print("| Target | Count |")
    print("|---|---:|")
    for bucket, count in target_bucket_counts(records).most_common():
        print(f"| `{bucket}` | {count} |")
    print()
    print("## Git State Counts")
    print()
    print("| State | Count |")
    print("|---|---:|")
    for state, count in git_state_counts(records).most_common():
        print(f"| `{state}` | {count} |")
    print()
    print(f"## Top Referenced Scripts")
    print()
    print("| References | Bucket | Current path | Target path |")
    print("|---:|---|---|---|")
    for record in sorted(records, key=lambda item: (-item.reference_count, item.name))[:top_references]:
        print(
            f"| {record.reference_count} | `{record.bucket}` | "
            f"`{record.path}` | `{record.target_path}` |"
        )


def print_json(records: list[ScriptRecord]) -> None:
    payload = {
        "total": len(records),
        "bucket_counts": dict(bucket_counts(records)),
        "sub_bucket_counts": dict(sub_bucket_counts(records)),
        "target_bucket_counts": dict(target_bucket_counts(records)),
        "git_state_counts": dict(git_state_counts(records)),
        "action_counts": dict(action_counts(records)),
        "records": [asdict(record) for record in records],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    records = build_records(root)

    if args.format == "json":
        print_json(records)
    elif args.format == "markdown":
        print_markdown(records, args.top_references)
    else:
        print_text(records, args.top_references)

    if args.fail_on_unclassified and any(record.bucket == "unclassified" for record in records):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
