#!/usr/bin/env python3
"""Build a review manifest for local research result artifacts.

The manifest is deliberately read-only. It inventories result artifact roots,
matches them against ``research/DATA_RETENTION_PLAN.md`` entries, and writes a
CSV that can be reviewed before any archive, compression, or deletion step.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import io
import re
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_PLAN = Path("research/DATA_RETENTION_PLAN.md")
DEFAULT_OUTPUT = Path("research/retention_manifest.csv")
DEFAULT_ARTIFACT_ROOTS = [
    Path("research/consensus/results"),
    Path("research/consensus/results/adaptive_refinement"),
    Path("research/dendrogram/results"),
    Path("research/experiments/combination/results"),
]
EXPANDED_RESULT_ROOTS = {
    "research/consensus/results/adaptive_refinement",
}
RETENTION_LABELS = {
    "KEEP-LIVE",
    "KEEP-SUMMARY",
    "CONSOLIDATE",
    "ARCHIVE",
    "DROP-CANDIDATE",
}
MANIFEST_COLUMNS = [
    "path",
    "size",
    "size_bytes",
    "track",
    "label",
    "reason",
    "representative_summary",
    "rerun_command",
    "failure_id",
    "exists",
    "source",
]
TEXT_SUMMARY_SUFFIXES = {".csv", ".json", ".md", ".txt"}


@dataclass(frozen=True)
class RetentionRule:
    pattern: str
    track: str
    label: str
    reason: str
    source_line: int


def display_path(path: Path, *, root: Path | None = None) -> str:
    base = root or Path.cwd()
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()


def normalize_pattern(path: str) -> str:
    return path.strip().rstrip("/")


def extract_paths(line: str) -> list[str]:
    paths: list[str] = []
    for value in re.findall(r"`([^`]+)`", line):
        if value.startswith("research/"):
            paths.append(normalize_pattern(value))
    return paths


def load_rules(plan_path: Path) -> list[RetentionRule]:
    rules: list[RetentionRule] = []
    track = ""
    label = ""
    section_rule_indexes: list[int] = []
    reason_targets: list[int] = []
    reason_parts: list[str] = []

    def flush_reason() -> None:
        nonlocal reason_parts, reason_targets
        if not reason_targets:
            return
        reason = " ".join(part for part in reason_parts if part).strip()
        if reason:
            for index in reason_targets:
                previous = rules[index]
                rules[index] = RetentionRule(
                    pattern=previous.pattern,
                    track=previous.track,
                    label=previous.label,
                    reason=reason,
                    source_line=previous.source_line,
                )
        reason_parts = []
        reason_targets = []

    for line_no, raw_line in enumerate(plan_path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        is_heading = line.startswith("##")
        is_bullet = line.startswith("- ")
        if is_heading or is_bullet:
            flush_reason()

        if line.startswith("## "):
            track = line.removeprefix("## ").strip()
            label = ""
            section_rule_indexes = []
            continue

        if line.startswith("### "):
            heading = line.removeprefix("### ").strip()
            label = heading if heading in RETENTION_LABELS else ""
            section_rule_indexes = []
            continue

        if label and is_bullet:
            for pattern in extract_paths(line):
                rules.append(
                    RetentionRule(
                        pattern=pattern,
                        track=track,
                        label=label,
                        reason="",
                        source_line=line_no,
                    )
                )
                section_rule_indexes.append(len(rules) - 1)
            continue

        if line.startswith("Reason:") and section_rule_indexes:
            reason_targets = list(section_rule_indexes)
            reason_parts = [line.removeprefix("Reason:").strip()]
            continue

        if reason_targets:
            if line:
                reason_parts.append(line)
            else:
                flush_reason()

    flush_reason()
    return rules


def iter_artifacts(artifact_roots: list[Path], *, repo_root: Path) -> list[Path]:
    artifacts: list[Path] = []
    seen: set[str] = set()
    for root in artifact_roots:
        resolved = root if root.is_absolute() else repo_root / root
        if not resolved.exists() or not resolved.is_dir():
            continue
        for child in sorted(resolved.iterdir()):
            if child.name.startswith(".") or child.name == "__pycache__":
                continue
            rel = display_path(child, root=repo_root)
            if rel in EXPANDED_RESULT_ROOTS:
                continue
            if rel not in seen:
                artifacts.append(child)
                seen.add(rel)
    return artifacts


def match_rule(path: Path, rules: list[RetentionRule], *, repo_root: Path) -> RetentionRule | None:
    rel = display_path(path, root=repo_root)
    exact = [rule for rule in rules if "*" not in rule.pattern and rel == rule.pattern]
    if exact:
        return exact[0]
    wildcard = [rule for rule in rules if "*" in rule.pattern and fnmatch.fnmatch(rel, rule.pattern)]
    if wildcard:
        return sorted(wildcard, key=lambda rule: len(rule.pattern), reverse=True)[0]
    return None


def directory_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        if child.is_file() and not child.is_symlink():
            try:
                total += child.stat().st_size
            except FileNotFoundError:
                continue
    return total


def human_size(size_bytes: int) -> str:
    value = float(size_bytes)
    for unit in ("B", "K", "M", "G", "T"):
        if value < 1024 or unit == "T":
            if unit == "B":
                return f"{int(value)}B"
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{size_bytes}B"


def representative_summary(path: Path, *, repo_root: Path, max_files: int = 3000) -> str:
    if path.is_file():
        return display_path(path, root=repo_root)

    best: tuple[int, str] | None = None
    scanned = 0
    for child in path.rglob("*"):
        if not child.is_file() or child.suffix.lower() not in TEXT_SUMMARY_SUFFIXES:
            continue
        scanned += 1
        name = child.name.lower()
        priority = 4
        if "manifest" in name:
            priority = 0
        elif "summary" in name:
            priority = 1
        elif "report" in name:
            priority = 2
        elif child.suffix.lower() == ".md":
            priority = 3
        rel = display_path(child, root=repo_root)
        candidate = (priority, rel)
        if best is None or candidate < best:
            best = candidate
        if scanned >= max_files:
            break

    return best[1] if best else ""


def manifest_rows(
    *,
    artifact_roots: list[Path],
    rules: list[RetentionRule],
    repo_root: Path,
    include_missing_plan_entries: bool = False,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    artifacts = iter_artifacts(artifact_roots, repo_root=repo_root)
    seen_paths: set[str] = set()

    for artifact in artifacts:
        rel = display_path(artifact, root=repo_root)
        seen_paths.add(rel)
        rule = match_rule(artifact, rules, repo_root=repo_root)
        size_bytes = directory_size(artifact)
        rows.append(
            {
                "path": rel,
                "size": human_size(size_bytes),
                "size_bytes": str(size_bytes),
                "track": rule.track if rule else "",
                "label": rule.label if rule else "UNCLASSIFIED",
                "reason": rule.reason if rule else "No matching DATA_RETENTION_PLAN entry.",
                "representative_summary": representative_summary(artifact, repo_root=repo_root),
                "rerun_command": "",
                "failure_id": "",
                "exists": "true",
                "source": f"{DEFAULT_PLAN.as_posix()}:{rule.source_line}" if rule else "",
            }
        )

    if include_missing_plan_entries:
        for rule in rules:
            if "*" in rule.pattern or rule.pattern in seen_paths:
                continue
            path = repo_root / rule.pattern
            if path.exists():
                continue
            rows.append(
                {
                    "path": rule.pattern,
                    "size": "0B",
                    "size_bytes": "0",
                    "track": rule.track,
                    "label": rule.label,
                    "reason": rule.reason,
                    "representative_summary": "",
                    "rerun_command": "",
                    "failure_id": "",
                    "exists": "false",
                    "source": f"{DEFAULT_PLAN.as_posix()}:{rule.source_line}",
                }
            )

    return sorted(rows, key=lambda row: (row["label"], row["track"], row["path"]))


def write_csv(rows: list[dict[str, str]], output: Path | None) -> None:
    handle: io.TextIOBase
    if output is None:
        handle = sys.stdout
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        handle = output.open("w", encoding="utf-8", newline="")
    try:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if output is not None:
            handle.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Write the manifest CSV to stdout instead of --output.",
    )
    parser.add_argument(
        "--root",
        action="append",
        type=Path,
        default=None,
        help="Result artifact root to scan. May be passed more than once.",
    )
    parser.add_argument(
        "--include-missing-plan-entries",
        action="store_true",
        help="Include exact DATA_RETENTION_PLAN entries that no longer exist.",
    )
    parser.add_argument(
        "--fail-on-unclassified",
        action="store_true",
        help="Exit nonzero if any scanned artifact has no retention-plan rule.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repo_root = Path.cwd()
    rules = load_rules(args.plan)
    roots = args.root if args.root is not None else DEFAULT_ARTIFACT_ROOTS
    rows = manifest_rows(
        artifact_roots=roots,
        rules=rules,
        repo_root=repo_root,
        include_missing_plan_entries=args.include_missing_plan_entries,
    )
    write_csv(rows, None if args.stdout else args.output)
    unclassified = sum(1 for row in rows if row["label"] == "UNCLASSIFIED")
    print(
        f"research retention manifest rows={len(rows)} unclassified={unclassified}",
        file=sys.stderr,
    )
    return 1 if args.fail_on_unclassified and unclassified else 0


if __name__ == "__main__":
    raise SystemExit(main())
