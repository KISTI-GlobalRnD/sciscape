#!/usr/bin/env python3
"""Check that documentation folders stay small enough to scan."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


DOC_EXTENSIONS = {".md", ".rst", ".txt", ".tex", ".ipynb"}
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
    "env",
    "node_modules",
    "target",
    "venv",
    "workspace",
}
IGNORED_PATHS = {
    "docs/api",
    "research/consensus/results",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fail if a documentation parent has too many visible children. "
            "Visible children are subdirectories plus direct document files, "
            "excluding README.md and generated paths."
        )
    )
    parser.add_argument(
        "--root",
        action="append",
        default=None,
        help="Root to scan. May be passed more than once. Defaults to docs.",
    )
    parser.add_argument(
        "--max-entries",
        type=int,
        default=6,
        help="Maximum visible entries per parent directory. Defaults to 6.",
    )
    return parser.parse_args()


def display_path(path: Path) -> str:
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.as_posix()


def is_ignored_dir(path: Path) -> bool:
    rel = display_path(path)
    if path.name.startswith("."):
        return True
    if any(part in IGNORED_DIR_NAMES for part in path.parts):
        return True
    return rel in IGNORED_PATHS or any(rel.startswith(f"{prefix}/") for prefix in IGNORED_PATHS)


def iter_doc_dirs(root: Path) -> list[Path]:
    dirs: list[Path] = []
    stack = [root]
    while stack:
        current = stack.pop()
        if not current.exists() or not current.is_dir() or is_ignored_dir(current):
            continue
        dirs.append(current)
        children = [child for child in current.iterdir() if child.is_dir() and not is_ignored_dir(child)]
        stack.extend(sorted(children, reverse=True))
    return dirs


def visible_entries(path: Path) -> list[Path]:
    entries: list[Path] = []
    for child in sorted(path.iterdir()):
        if child.is_dir():
            if not is_ignored_dir(child):
                entries.append(child)
            continue
        if child.name.lower() == "readme.md":
            continue
        if child.suffix.lower() in DOC_EXTENSIONS:
            entries.append(child)
    return entries


def main() -> int:
    args = parse_args()
    roots = [Path(root) for root in (args.root or ["docs"])]
    violations: list[tuple[Path, list[Path]]] = []

    for root in roots:
        for directory in iter_doc_dirs(root):
            entries = visible_entries(directory)
            if len(entries) > args.max_entries:
                violations.append((directory, entries))

    if violations:
        print(f"Documentation fanout violations (max {args.max_entries} visible entries):")
        for directory, entries in violations:
            names = ", ".join(entry.name for entry in entries)
            print(f"- {display_path(directory)}: {len(entries)} entries")
            print(f"  entries: {names}")
        return 1

    scanned = ", ".join(display_path(root) for root in roots)
    print(f"Documentation fanout OK: {scanned} <= {args.max_entries} visible entries per parent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
