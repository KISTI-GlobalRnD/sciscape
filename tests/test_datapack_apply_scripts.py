from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(script_name: str) -> ModuleType:
    path = REPO_ROOT / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def _write_checksum(path: Path, relpaths: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{'0' * 64}  {relpath}\n" for relpath in relpaths))


def test_clean_keyword_export_apply_updates_datapack_checksums(tmp_path):
    module = _load_script_module("export_clean_keywords_to_atlas_datapack.py")
    datapack = tmp_path / "datapack"
    output_dir = tmp_path / "export"
    (datapack / "core").mkdir(parents=True)
    (datapack / "dashboard" / "tables").mkdir(parents=True)
    (datapack / "qa").mkdir()
    output_dir.mkdir()

    pd.DataFrame(
        {
            "cluster_uid": ["domain:0", "nano:0"],
            "term": ["legacy domain", "legacy nano"],
            "rank": [1, 1],
            "score": [0.4, 0.1],
            "evidence_channel": ["legacy", "legacy"],
        }
    ).to_parquet(datapack / "core" / "atlas_cluster_terms.parquet", index=False)
    pd.DataFrame({"cluster_uid": ["nano:0"], "term": ["legacy nano"]}).to_parquet(
        datapack / "dashboard" / "tables" / "nano_terms_topk.parquet",
        index=False,
    )
    _write_checksum(
        datapack / "CHECKSUMS.sha256",
        ["core/atlas_cluster_terms.parquet", "dashboard/tables/nano_terms_topk.parquet"],
    )
    _write_checksum(datapack / "core" / "CHECKSUMS.sha256", ["atlas_cluster_terms.parquet"])
    _write_checksum(datapack / "dashboard" / "CHECKSUMS.sha256", ["tables/nano_terms_topk.parquet"])
    _write_checksum(datapack / "dashboard" / "tables" / "CHECKSUMS.sha256", ["nano_terms_topk.parquet"])

    core_nano = pd.DataFrame(
        {
            "cluster_uid": ["nano:0"],
            "term": ["clean nano"],
            "rank": [1],
            "score": [0.9],
            "evidence_channel": ["sciscape_clean_v10"],
        }
    )
    dashboard_nano = pd.DataFrame(
        {
            "cluster_uid": ["nano:0"],
            "level": ["nano"],
            "cluster_id": [0],
            "ngram_n": [2],
            "term": ["clean nano"],
            "term_count": [5],
            "term_doc_count": [4],
            "representative_doc_count": [4],
            "score": [0.9],
            "rank": [1],
        }
    )

    marker = module.apply_to_datapack(datapack, core_nano, dashboard_nano, output_dir)

    core_digest = module.sha256_file(datapack / "core" / "atlas_cluster_terms.parquet")
    dashboard_digest = module.sha256_file(datapack / "dashboard" / "tables" / "nano_terms_topk.parquet")
    assert marker["core_sha256"] == core_digest
    assert marker["dashboard_sha256"] == dashboard_digest
    assert f"{core_digest}  core/atlas_cluster_terms.parquet" in (datapack / "CHECKSUMS.sha256").read_text()
    assert f"{dashboard_digest}  dashboard/tables/nano_terms_topk.parquet" in (
        datapack / "CHECKSUMS.sha256"
    ).read_text()
    assert f"{core_digest}  atlas_cluster_terms.parquet" in (datapack / "core" / "CHECKSUMS.sha256").read_text()
    assert f"{dashboard_digest}  tables/nano_terms_topk.parquet" in (
        datapack / "dashboard" / "CHECKSUMS.sha256"
    ).read_text()
    assert f"{dashboard_digest}  nano_terms_topk.parquet" in (
        datapack / "dashboard" / "tables" / "CHECKSUMS.sha256"
    ).read_text()


def test_rollup_lineage_validation_detects_stale_node_parent_chain(tmp_path):
    module = _load_script_module("rollup_clean_keywords_to_atlas_datapack.py")
    node_path = tmp_path / "atlas_cluster_nodes.parquet"
    lineage = pd.DataFrame(
        {
            "hierarchy_version": ["v1"],
            "nano_id": [1],
            "nano_docs": [10],
            "micro_id": [10],
            "meso_id": [20],
            "macro_id": [30],
            "domain_id": [0],
        }
    )
    pd.DataFrame(
        {
            "cluster_uid": ["domain:0", "macro:30", "meso:20", "micro:10", "nano:1"],
            "level": ["domain", "macro", "meso", "micro", "nano"],
            "cluster_id": [0, 30, 20, 10, 1],
            "parent_uid": ["", "domain:0", "macro:30", "meso:999", "micro:10"],
        }
    ).to_parquet(node_path, index=False)

    validation = module.validate_lineage_against_nodes(lineage, node_path)

    assert validation["status"] == "failed"
    assert validation["parent_mismatch_count"] == 1
    assert validation["parent_mismatch_examples"][0]["child_uid"] == "micro:10"
    with pytest.raises(ValueError, match="node hierarchy does not match"):
        module.assert_node_lineage_validation_passed(validation)


def test_rollup_apply_blocks_when_node_lineage_validation_fails(tmp_path):
    module = _load_script_module("rollup_clean_keywords_to_atlas_datapack.py")
    datapack = tmp_path / "datapack"
    (datapack / "core").mkdir(parents=True)
    (datapack / "dashboard" / "tables").mkdir(parents=True)
    (datapack / "qa").mkdir()
    pd.DataFrame(
        {
            "cluster_uid": ["nano:1"],
            "level": ["nano"],
            "cluster_id": [1],
            "term": ["clean term"],
            "term_count": [3],
            "term_doc_count": [2],
            "representative_doc_count": [10],
            "score": [0.9],
            "rank": [1],
        }
    ).to_parquet(datapack / "dashboard" / "tables" / "nano_terms_topk.parquet", index=False)
    pd.DataFrame(
        {
            "hierarchy_version": ["v1"],
            "nano_id": [1],
            "nano_docs": [10],
            "micro_id": [10],
            "meso_id": [20],
            "macro_id": [30],
            "domain_id": [0],
        }
    ).to_parquet(datapack / "dashboard" / "tables" / "cluster_lineage.parquet", index=False)
    pd.DataFrame(
        {
            "cluster_uid": ["domain:0", "macro:30", "meso:20", "micro:10", "nano:1"],
            "level": ["domain", "macro", "meso", "micro", "nano"],
            "cluster_id": [0, 30, 20, 10, 1],
            "parent_uid": ["", "domain:0", "macro:30", "meso:999", "micro:10"],
        }
    ).to_parquet(datapack / "core" / "atlas_cluster_nodes.parquet", index=False)
    pd.DataFrame(
        {
            "cluster_uid": ["nano:1"],
            "term": ["old term"],
            "rank": [1],
            "score": [0.1],
            "evidence_channel": ["old"],
        }
    ).to_parquet(datapack / "core" / "atlas_cluster_terms.parquet", index=False)

    with pytest.raises(ValueError, match="node hierarchy does not match"):
        module.run(SimpleNamespace(datapack_dir=datapack, nano_terms=None, top_n_upper=5, write_preview=False, apply=True))
