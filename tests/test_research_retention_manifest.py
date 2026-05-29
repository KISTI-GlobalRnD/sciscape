"""Tests for research retention manifest generation."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "research_retention_manifest.py"
_SCRIPT_SPEC = spec_from_file_location("research_retention_manifest_test_module", _SCRIPT_PATH)
assert _SCRIPT_SPEC is not None and _SCRIPT_SPEC.loader is not None
_SCRIPT_MODULE = module_from_spec(_SCRIPT_SPEC)
sys.modules[_SCRIPT_SPEC.name] = _SCRIPT_MODULE
_SCRIPT_SPEC.loader.exec_module(_SCRIPT_MODULE)

load_rules = _SCRIPT_MODULE.load_rules
manifest_rows = _SCRIPT_MODULE.manifest_rows


def test_load_rules_captures_label_track_and_reason(tmp_path):
    plan = tmp_path / "DATA_RETENTION_PLAN.md"
    plan.write_text(
        "\n".join(
            [
                "## Track A: Example",
                "",
                "### KEEP-LIVE",
                "",
                "- `research/consensus/results/example_keep/`",
                "",
                "Reason: paper-facing artifact.",
                "",
                "### ARCHIVE",
                "",
                "- `research/consensus/results/example_archive_*`",
                "",
                "Reason: predecessor runs.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    rules = load_rules(plan)

    assert [rule.pattern for rule in rules] == [
        "research/consensus/results/example_keep",
        "research/consensus/results/example_archive_*",
    ]
    assert rules[0].track == "Track A: Example"
    assert rules[0].label == "KEEP-LIVE"
    assert rules[0].reason == "paper-facing artifact."
    assert rules[1].label == "ARCHIVE"
    assert rules[1].reason == "predecessor runs."


def test_manifest_rows_classifies_exact_wildcard_and_unclassified(tmp_path):
    plan = tmp_path / "DATA_RETENTION_PLAN.md"
    plan.write_text(
        "\n".join(
            [
                "## Track A: Example",
                "### KEEP-LIVE",
                "- `research/consensus/results/example_keep/`",
                "Reason: current evidence.",
                "### ARCHIVE",
                "- `research/consensus/results/example_archive_*`",
                "Reason: superseded evidence.",
            ]
        ),
        encoding="utf-8",
    )
    result_root = tmp_path / "research" / "consensus" / "results"
    keep_dir = result_root / "example_keep"
    archive_dir = result_root / "example_archive_20260529"
    unknown_dir = result_root / "example_unknown"
    keep_dir.mkdir(parents=True)
    archive_dir.mkdir()
    unknown_dir.mkdir()
    (keep_dir / "summary.json").write_text("{}", encoding="utf-8")
    (archive_dir / "report.md").write_text("# report\n", encoding="utf-8")
    (unknown_dir / "raw.bin").write_bytes(b"abc")

    rows = manifest_rows(
        artifact_roots=[Path("research/consensus/results")],
        rules=load_rules(plan),
        repo_root=tmp_path,
    )
    by_path = {row["path"]: row for row in rows}

    assert by_path["research/consensus/results/example_keep"]["label"] == "KEEP-LIVE"
    assert by_path["research/consensus/results/example_keep"]["representative_summary"].endswith(
        "summary.json"
    )
    assert by_path["research/consensus/results/example_archive_20260529"]["label"] == "ARCHIVE"
    assert by_path["research/consensus/results/example_unknown"]["label"] == "UNCLASSIFIED"
