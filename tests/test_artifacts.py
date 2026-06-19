from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd

from sciscape.artifacts import (
    COOCCURRENCE_ARTIFACT_SCHEMA_VERSION,
    EVOLUTION_EVENTS_SCHEMA_VERSION,
    EVOLUTION_MANIFEST_SCHEMA_VERSION,
    EVOLUTION_QA_SCHEMA_VERSION,
    EVOLUTION_STATE_MEMBERSHIP_SCHEMA_VERSION,
    EXPORT_FILES_SCHEMA_VERSION,
    EXPORT_INPUTS_SCHEMA_VERSION,
    EXPORT_MANIFEST_SCHEMA_VERSION,
    EXPORT_QA_SCHEMA_VERSION,
    EXPORT_TRANSFORMS_SCHEMA_VERSION,
    KEYWORD_RULE_MANIFEST_SCHEMA_VERSION,
    KEYWORD_RULE_QA_SCHEMA_VERSION,
    MATRIX_ENTITIES_SCHEMA_VERSION,
    MATRIX_MANIFEST_SCHEMA_VERSION,
    MATRIX_QA_SCHEMA_VERSION,
    MATRIX_VALUES_SCHEMA_VERSION,
    NARRATIVE_CLAIMS_SCHEMA_VERSION,
    NARRATIVE_GENERATION_METADATA_SCHEMA_VERSION,
    NARRATIVE_MANIFEST_SCHEMA_VERSION,
    NARRATIVE_PUBLICATION_SCHEMA_VERSION,
    NARRATIVE_QA_SCHEMA_VERSION,
    RESULT_MANIFEST_SCHEMA_VERSION,
    TEMPORAL_ACTIVITY_SCHEMA_VERSION,
    TEMPORAL_MANIFEST_SCHEMA_VERSION,
    TEMPORAL_QA_SCHEMA_VERSION,
    WORKSPACE_MANIFEST_SCHEMA_VERSION,
    WORKSPACE_QA_SCHEMA_VERSION,
    build_atlas_payload_from_report_data,
    build_atlas_render_payload,
    build_result_manifest,
    build_report_data_contract,
    CLUSTER_REVIEW_PACKET_QA_SCHEMA_VERSION,
    CLUSTER_REVIEW_PACKET_SCHEMA_VERSION,
    infer_result_artifacts,
    load_result_manifest,
    register_result_in_workspace,
    validate_cluster_review_packet_artifact,
    validate_evolution_artifact,
    validate_export_manifest,
    validate_keyword_rule_artifact,
    validate_matrix_artifact,
    validate_narrative_artifact,
    validate_result_root,
    validate_temporal_artifact,
    validate_workspace,
    write_cooccurrence_artifacts,
    write_document_overlap_evolution_artifacts,
    write_edge_evidence_samples,
    write_evidence_backed_evolution_artifacts,
    write_evolution_artifacts,
    write_evolution_synthetic_smoke_artifact,
    write_export_manifest,
    write_keyword_rule_artifacts,
    write_matrix_artifact,
    write_matrix_from_term_cooccurrence,
    write_narrative_evidence_artifacts,
    write_narrative_publication_artifacts,
    write_slice_local_membership_evolution_artifacts,
    write_slice_reclustering_evolution_artifacts,
    write_slice_membership_evolution_artifacts,
    write_temporal_artifacts,
    write_artifact_contract,
    write_cluster_review_packet_artifact,
    write_result_manifest,
    write_workspace_manifest,
)
from sciscape.export import (
    export_cooccurrence_table,
    export_graphml,
    export_vosviewer_bundle,
    export_matrix_artifact,
    export_vosviewer_network,
    export_vosviewer_term_cooccurrence,
    export_vosviewer_thesaurus,
)
from sciscape.keyword_extraction.rule_artifact import write_keyword_cleaning_rule_artifacts
from sciscape.keyword_extraction.visualization import export_dashboard, export_report, export_viewer


def _write_valid_result_root(root: Path) -> Path:
    landscape = root / "landscape"
    report = landscape / "report"
    report.mkdir(parents=True)

    pd.DataFrame(
        {
            "uid": ["D0", "D1", "D2", "D3"],
            "title": [
                "Perovskite interface passivation",
                "Perovskite device stability",
                "Graph neural traffic forecasting",
                "Graph neural anomaly detection",
            ],
            "abstract": [
                "Interface passivation improves perovskite solar cell stability.",
                "Stable perovskite devices use passivation layers.",
                "Graph neural networks forecast traffic over sensor graphs.",
                "Graph neural networks detect anomalies in dynamic graphs.",
            ],
            "pubyear": [2021, 2022, 2021, 2022],
            "cited_by_count": [5, 8, 3, 10],
        }
    ).to_parquet(root / "abstracts.parquet", index=False)
    pd.DataFrame(
        {
            "uid1": ["D0", "D1", "D2"],
            "uid2": ["D1", "D2", "D3"],
            "rel_sum2": [2.0, 1.0, 2.0],
        }
    ).to_parquet(root / "edges.parquet", index=False)
    pd.DataFrame(
        {
            "uid": ["D0", "D1", "D2", "D3"],
            "cluster": [0, 0, 1, 1],
        }
    ).to_parquet(landscape / "membership.parquet", index=False)
    pd.DataFrame(
        {
            "cluster_id": [0, 0, 1, 1],
            "term": [
                "perovskite solar cells",
                "interface passivation",
                "graph neural networks",
                "traffic forecasting",
            ],
            "score": [0.9, 0.8, 0.95, 0.75],
            "frequency": [2, 1, 2, 1],
        }
    ).to_parquet(landscape / "keywords.parquet", index=False)

    report_data = {
        "0": {
            "label": "perovskite solar cells",
            "keywords": [{"term": "perovskite solar cells"}, {"term": "interface passivation"}],
            "network_edges": [{"source": "perovskite solar cells", "target": "interface passivation", "weight": 1}],
            "cooccurrence_table": [{"source": "perovskite solar cells", "target": "interface passivation", "count": 1}],
        },
        "1": {
            "label": "graph neural networks",
            "keywords": [{"term": "graph neural networks"}, {"term": "traffic forecasting"}],
            "network_edges": [{"source": "graph neural networks", "target": "traffic forecasting", "weight": 1}],
            "cooccurrence_table": [{"source": "graph neural networks", "target": "traffic forecasting", "count": 1}],
        },
        "_trend_scores": {"perovskite solar cells": {"2021": 1, "2022": 2}},
    }
    report_data["_sciscape"] = build_report_data_contract(report_data)
    (report / "data.json").write_text(json.dumps(report_data), encoding="utf-8")
    return root


def _keyword_before_after_row(
    *,
    cluster_id: int,
    raw_term: str,
    rule_ids: str = "",
    term_after: str | None = None,
    display_label: str | None = None,
    quality_flags: str = "",
    blocked: bool = False,
    block_reason: str = "",
) -> dict[str, object]:
    final_term = raw_term if term_after is None else term_after
    return {
        "cluster_id": cluster_id,
        "raw_term": raw_term,
        "term_before": raw_term,
        "term_after": final_term,
        "display_label": final_term if display_label is None else display_label,
        "family_id": final_term.lower(),
        "parent_term": "",
        "variant_count": 1,
        "rule_ids": rule_ids,
        "quality_flags": quality_flags,
        "review_status": "blocked" if blocked else "accepted",
        "tier_before": "candidate",
        "tier_after": "drop" if blocked else "primary",
        "blocked": blocked,
        "block_reason": block_reason,
    }


def test_validate_result_root_infers_features_and_counts(tmp_path):
    root = _write_valid_result_root(tmp_path / "result")

    result = validate_result_root(root)
    payload = result.to_dict()

    assert payload["ok"] is True
    assert payload["result_state"] == "loaded"
    assert payload["features"]["overview"] is True
    assert payload["features"]["cluster_map"] is True
    assert payload["features"]["keyword"] is True
    assert payload["features"]["term_network"] is True
    assert payload["features"]["matrix"] is False
    assert payload["features"]["evidence"] is True
    assert payload["features"]["temporal"] is True
    assert payload["features"]["evolution"] is False
    assert payload["features"]["narrative"] is False
    assert payload["features"]["quality"] is True
    assert payload["features"]["export"] is True
    assert payload["counts"]["abstract_rows"] == 4
    assert payload["counts"]["membership_rows"] == 4
    assert payload["counts"]["keyword_rows"] == 4
    assert payload["counts"]["report_clusters"] == 2
    assert not [w for w in payload["warnings"] if w["severity"] == "error"]


def test_write_cooccurrence_artifacts_promotes_stable_manifest_feature(tmp_path):
    root = _write_valid_result_root(tmp_path / "result")

    written = write_cooccurrence_artifacts(root)

    assert written is not None
    table_path = written["table_path"]
    map_path = written["map_path"]
    assert table_path == root / "landscape" / "term_cooccurrence.parquet"
    assert map_path == root / "landscape" / "term_cooccurrence_map.json"
    table = pd.read_parquet(table_path)
    assert len(table) == 2
    assert set(
        [
            "schema_version",
            "cluster_uid",
            "cluster_level",
            "cluster_id",
            "source",
            "target",
            "weight",
            "relation",
        ]
    ).issubset(table.columns)
    assert set(table["schema_version"]) == {COOCCURRENCE_ARTIFACT_SCHEMA_VERSION}

    cooc_map = json.loads(map_path.read_text(encoding="utf-8"))
    assert cooc_map["schema_version"] == COOCCURRENCE_ARTIFACT_SCHEMA_VERSION
    assert cooc_map["edge_count"] == 2
    assert "perovskite solar cells" in cooc_map["terms"]

    contract = validate_result_root(root).to_dict()
    assert contract["counts"]["cooccurrence_artifacts"] == 2
    assert contract["counts"]["cooccurrence_rows"] == 2
    assert contract["features"]["term_network"] is True
    assert contract["features"]["matrix"] is False

    manifest = build_result_manifest(root).to_dict()
    assert manifest["features"]["cooccurrence"]["state"] == "stable"
    assert manifest["features"]["cooccurrence"]["reason"] == "feature validated"
    assert "cooccurrence" in manifest["features"]["cooccurrence"]["artifact_refs"]
    assert manifest["features"]["matrix"]["state"] == "hidden"
    assert manifest["features"]["matrix"]["artifact_refs"] == []
    assert manifest["artifacts"]["cooccurrence"]["schema_version"] == COOCCURRENCE_ARTIFACT_SCHEMA_VERSION
    assert manifest["artifacts"]["cooccurrence"]["rows"] == 2


def test_export_cooccurrence_table_writes_manifest_backed_table(tmp_path):
    root = _write_valid_result_root(tmp_path / "result")
    write_cooccurrence_artifacts(root)

    written = export_cooccurrence_table(root)

    table_path = root / "exports" / "term_cooccurrence_table" / "term_cooccurrence.tsv"
    map_path = root / "exports" / "term_cooccurrence_table" / "term_cooccurrence_map.json"
    assert written["table_path"] == table_path
    assert written["map_path"] == map_path
    assert written["manifest_path"] == root / "exports" / "term_cooccurrence_table" / "export_manifest.json"
    assert table_path.exists()
    assert map_path.exists()
    table = pd.read_csv(table_path, sep="\t")
    assert len(table) == 2
    assert {"source", "target", "weight", "relation"}.issubset(table.columns)
    assert set(table["schema_version"]) == {COOCCURRENCE_ARTIFACT_SCHEMA_VERSION}

    cooc_map = json.loads(map_path.read_text(encoding="utf-8"))
    assert cooc_map["schema_version"] == COOCCURRENCE_ARTIFACT_SCHEMA_VERSION
    assert cooc_map["edge_count"] == 2

    validation = validate_export_manifest(written["manifest_path"]).to_dict()
    assert validation["status"] == "passed"
    assert validation["export_family"] == "table"
    assert validation["export_kind"] == "term_cooccurrence_table"
    assert validation["counts"]["files"] == 2
    assert validation["counts"]["inputs"] == 2

    manifest = build_result_manifest(root).to_dict()
    exports = [export for export in manifest["exports"] if export["export_id"] == "term_cooccurrence_table"]
    assert len(exports) == 1
    assert exports[0]["path"] == "exports/term_cooccurrence_table/term_cooccurrence.tsv"
    assert exports[0]["export_manifest_ref"] == "exports/term_cooccurrence_table/export_manifest.json"
    assert exports[0]["selection_summary"] == {
        "scope": "cooccurrence_artifact",
        "view_mode": "term_cooccurrence_table",
        "view_family": "table",
        "cluster_level": None,
        "filter_count": 0,
        "threshold_keys": [],
        "layer_state_keys": ["map_file", "row_count", "source_table", "table_format"],
        "focus_keys": [],
        "subset_mode": None,
        "subset_count": None,
        "subset_keys": [],
    }
    assert [row["path"] for row in exports[0]["files"]] == [
        "exports/term_cooccurrence_table/term_cooccurrence.tsv",
        "exports/term_cooccurrence_table/term_cooccurrence_map.json",
    ]


def test_export_vosviewer_term_cooccurrence_writes_map_network_and_manifest(tmp_path):
    root = _write_valid_result_root(tmp_path / "result")
    write_cooccurrence_artifacts(root)

    written = export_vosviewer_term_cooccurrence(root)

    map_path = root / "vosviewer" / "vosviewer_term_map.txt"
    network_path = root / "vosviewer" / "vosviewer_term_network.txt"
    manifest_path = root / "exports" / "vosviewer_term_cooccurrence" / "export_manifest.json"
    assert written["map_path"] == map_path
    assert written["network_path"] == network_path
    assert written["manifest_path"] == manifest_path
    assert map_path.exists()
    assert network_path.exists()

    map_lines = map_path.read_text(encoding="utf-8").splitlines()
    assert map_lines[0].split("\t") == [
        "id",
        "label",
        "description",
        "cluster",
        "weight<Links>",
        "weight<Total link strength>",
        "score<Cluster breadth>",
    ]
    map_rows = [line.split("\t") for line in map_lines[1:]]
    assert {row[1]: row[4:7] for row in map_rows} == {
        "graph neural networks": ["1", "1.000000", "1"],
        "interface passivation": ["1", "1.000000", "1"],
        "perovskite solar cells": ["1", "1.000000", "1"],
        "traffic forecasting": ["1", "1.000000", "1"],
    }
    assert network_path.read_text(encoding="utf-8").splitlines() == [
        "1\t4\t1.000000",
        "2\t3\t1.000000",
    ]

    validation = validate_export_manifest(manifest_path).to_dict()
    assert validation["status"] == "passed"
    assert validation["export_family"] == "vosviewer"
    assert validation["export_kind"] == "vosviewer_term_cooccurrence"
    assert validation["counts"]["files"] == 2
    assert validation["counts"]["inputs"] == 2
    export_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert export_manifest["selection"]["scope"] == "cooccurrence_artifact"
    assert export_manifest["selection"]["view"]["mode"] == "vosviewer_term_cooccurrence"
    assert export_manifest["selection"]["thresholds"] == {"min_link_strength": 0}
    assert export_manifest["selection"]["layer_state"] == {
        "map_file": "vosviewer_term_map.txt",
        "network_file": "vosviewer_term_network.txt",
        "source_table": "term_cooccurrence.parquet",
        "term_count": 4,
        "link_count": 2,
        "cluster_count": 2,
        "counting_method": "summed_cooccurrence_weight",
    }

    result_manifest = build_result_manifest(root).to_dict()
    exports = [export for export in result_manifest["exports"] if export["export_id"] == "vosviewer_term_cooccurrence"]
    assert len(exports) == 1
    assert exports[0]["path"] == "vosviewer/vosviewer_term_map.txt"
    assert exports[0]["export_manifest_ref"] == "exports/vosviewer_term_cooccurrence/export_manifest.json"
    assert exports[0]["selection_summary"] == {
        "scope": "cooccurrence_artifact",
        "view_mode": "vosviewer_term_cooccurrence",
        "view_family": "vosviewer",
        "cluster_level": None,
        "filter_count": 1,
        "threshold_keys": ["min_link_strength"],
        "layer_state_keys": [
            "cluster_count",
            "counting_method",
            "link_count",
            "map_file",
            "network_file",
            "source_table",
            "term_count",
        ],
        "focus_keys": [],
        "subset_mode": None,
        "subset_count": None,
        "subset_keys": [],
    }
    assert {row["role"]: row["path"] for row in exports[0]["files"]} == {
        "map": "vosviewer/vosviewer_term_map.txt",
        "network": "vosviewer/vosviewer_term_network.txt",
    }


def test_export_vosviewer_term_cooccurrence_collapses_more_than_1000_clusters(tmp_path):
    root = tmp_path / "result"
    landscape = root / "landscape"
    landscape.mkdir(parents=True)
    pd.DataFrame({"cluster_id": [0], "term": ["seed"], "score": [1.0]}).to_parquet(landscape / "keywords.parquet")
    pd.DataFrame(
        [
            {
                "source": f"term_{index}",
                "target": f"term_x_{index}",
                "weight": 1.0,
                "cluster_uid": f"cluster:{index}",
            }
            for index in range(1001)
        ]
    ).to_parquet(landscape / "term_cooccurrence.parquet")

    written = export_vosviewer_term_cooccurrence(root)

    map_lines = written["map_path"].read_text(encoding="utf-8").splitlines()
    cluster_ids = [int(line.split("\t")[3]) for line in map_lines[1:]]
    assert max(cluster_ids) == 1000
    manifest = json.loads(written["manifest_path"].read_text(encoding="utf-8"))
    layer_state = manifest["selection"]["layer_state"]
    assert layer_state["cluster_count"] == 1000
    assert layer_state["source_cluster_count"] == 1001
    assert layer_state["vosviewer_cluster_count"] == 1000
    assert layer_state["cluster_assignment"] == "top_999_plus_overflow"
    assert validate_export_manifest(written["manifest_path"]).to_dict()["status"] == "passed"


def test_write_cluster_review_packet_artifact_promotes_evidence_and_quality_refs(tmp_path):
    root = _write_valid_result_root(tmp_path / "result")
    write_cooccurrence_artifacts(root)

    written = write_cluster_review_packet_artifact(root)

    packet_path = root / "review" / "cluster_review_packet.json"
    qa_path = root / "review" / "cluster_review_packet_qa.json"
    assert written is not None
    assert written["packet_path"] == packet_path
    assert written["qa_path"] == qa_path
    assert written["clusters"] == 2
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    assert packet["schema_version"] == CLUSTER_REVIEW_PACKET_SCHEMA_VERSION
    assert packet["review_policy"]["narrative_generation_allowed"] is False
    assert packet["qa"]["path"] == "review/cluster_review_packet_qa.json"
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    assert qa["schema_version"] == CLUSTER_REVIEW_PACKET_QA_SCHEMA_VERSION
    assert qa["counts"]["clusters"] == 2
    assert qa["counts"]["narrative_ready_clusters"] == 2

    by_uid = {cluster["cluster_uid"]: cluster for cluster in packet["clusters"]}
    cluster = by_uid["cluster:0"]
    assert cluster["review_status"] == "clean"
    assert cluster["narrative_ready"] is True
    assert [row["term"] for row in cluster["keyword_evidence"]] == [
        "perovskite solar cells",
        "interface passivation",
    ]
    assert cluster["representative_works"][0]["title"] == "Perovskite device stability"
    assert cluster["cooccurrence_evidence"] == [
        {
            "evidence_ref_id": cluster["cooccurrence_evidence"][0]["evidence_ref_id"],
            "rank": 1,
            "source": "perovskite solar cells",
            "target": "interface passivation",
            "weight": 1.0,
            "count": 1,
            "relation": "cooccurrence",
        }
    ]
    ref_ids = {row["evidence_ref_id"] for row in cluster["evidence_refs"]}
    assert {row["evidence_ref_id"] for row in cluster["keyword_evidence"]} <= ref_ids
    assert {row["evidence_ref_id"] for row in cluster["representative_works"]} <= ref_ids
    assert {row["evidence_ref_id"] for row in cluster["cooccurrence_evidence"]} <= ref_ids

    validation = validate_cluster_review_packet_artifact(packet_path).to_dict()
    assert validation["status"] == "passed"
    assert validation["counts"]["clusters"] == 2
    assert validation["counts"]["evidence_refs"] == 10
    assert validation["checks"]["evidence_refs_resolvable"]["status"] == "passed"

    manifest = build_result_manifest(root).to_dict()
    assert manifest["artifacts"]["cluster_review_packet"]["path"] == "review/cluster_review_packet.json"
    assert manifest["artifacts"]["cluster_review_packet"]["schema_version"] == CLUSTER_REVIEW_PACKET_SCHEMA_VERSION
    assert manifest["artifacts"]["cluster_review_packet_qa"]["path"] == "review/cluster_review_packet_qa.json"
    assert "cluster_review_packet" in manifest["features"]["evidence"]["artifact_refs"]
    assert "cluster_review_packet_qa" in manifest["features"]["quality"]["artifact_refs"]
    assert manifest["features"]["narrative"]["state"] == "hidden"
    assert manifest["quality"]["gate_paths"] == [
        "landscape/qa/artifact_contract.json",
        "review/cluster_review_packet_qa.json",
    ]


def test_validate_cluster_review_packet_accepts_result_root(tmp_path):
    root = _write_valid_result_root(tmp_path / "result")
    write_cooccurrence_artifacts(root)
    write_cluster_review_packet_artifact(root)

    validation = validate_cluster_review_packet_artifact(root).to_dict()

    assert validation["status"] == "passed"
    assert validation["checks"]["source_artifacts"]["status"] == "passed"
    assert validation["counts"]["clusters"] == 2


def test_validate_result_root_does_not_promote_stale_review_packet_evidence(tmp_path):
    root = _write_valid_result_root(tmp_path / "result")
    write_cluster_review_packet_artifact(root)
    stale = tmp_path / "stale"
    (stale / "review").mkdir(parents=True)
    (stale / "review" / "cluster_review_packet.json").write_text(
        (root / "review" / "cluster_review_packet.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (stale / "review" / "cluster_review_packet_qa.json").write_text(
        (root / "review" / "cluster_review_packet_qa.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    validation = validate_result_root(stale).to_dict()

    assert validation["features"]["evidence"] is False
    assert validation["counts"]["review_packet_artifacts"] == 1
    assert validation["counts"]["stable_review_packet_artifacts"] == 0
    assert any(warning["code"] == "missing_review_packet_source_artifact" for warning in validation["warnings"])


def test_write_cluster_review_packet_external_output_uses_portable_qa_path(tmp_path):
    root = _write_valid_result_root(tmp_path / "result")
    external = tmp_path / "handoff_review"

    written = write_cluster_review_packet_artifact(root, output_dir=external)

    assert written is not None
    packet = json.loads((external / "cluster_review_packet.json").read_text(encoding="utf-8"))
    assert packet["qa"]["path"] == "cluster_review_packet_qa.json"
    assert not Path(packet["qa"]["path"]).is_absolute()


def test_validate_cluster_review_packet_blocks_unresolved_evidence_refs(tmp_path):
    root = _write_valid_result_root(tmp_path / "result")
    written = write_cluster_review_packet_artifact(root)
    assert written is not None
    packet_path = written["packet_path"]
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    packet["clusters"][0]["keyword_evidence"][0]["evidence_ref_id"] = "missing-ref"
    packet_path.write_text(json.dumps(packet, indent=2, sort_keys=True), encoding="utf-8")

    validation = validate_cluster_review_packet_artifact(packet_path).to_dict()

    assert validation["status"] == "blocked"
    assert validation["checks"]["evidence_refs_resolvable"]["status"] == "blocked"
    assert any(issue["code"] == "unresolved_review_evidence_refs" for issue in validation["blocking_issues"])


def test_write_narrative_evidence_artifacts_creates_claim_graph_from_review_packet(tmp_path):
    root = _write_valid_result_root(tmp_path / "result")
    write_cooccurrence_artifacts(root)
    write_cluster_review_packet_artifact(root)

    written = write_narrative_evidence_artifacts(root)

    assert written is not None
    manifest_path = root / "narrative" / "narrative_manifest.json"
    qa_path = root / "narrative" / "narrative_qa.json"
    assert written["manifest_path"] == manifest_path
    assert written["qa_path"] == qa_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == NARRATIVE_MANIFEST_SCHEMA_VERSION
    assert manifest["text_policy"]["llm_generation_allowed"] is False
    assert manifest["outputs"]["generation_metadata"] == "generation_metadata.json"
    generation_metadata = json.loads((root / "narrative" / "generation_metadata.json").read_text(encoding="utf-8"))
    assert generation_metadata["schema_version"] == NARRATIVE_GENERATION_METADATA_SCHEMA_VERSION
    assert generation_metadata["generation_mode"] == "deterministic_scaffold"
    assert generation_metadata["llm_generation_used"] is False
    assert generation_metadata["model_generation"] is None
    assert generation_metadata["parameters"]["max_targets"] == 500
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    assert qa["schema_version"] == NARRATIVE_QA_SCHEMA_VERSION
    assert qa["feature_state"] == "beta"

    claims = pd.read_parquet(root / "narrative" / "claims.parquet")
    links = pd.read_parquet(root / "narrative" / "claim_evidence_links.parquet")
    refs = pd.read_parquet(root / "narrative" / "evidence_refs.parquet")
    assert set(claims["schema_version"]) == {NARRATIVE_CLAIMS_SCHEMA_VERSION}
    assert {"identity", "keyword_meaning", "representative_work", "relation"}.issubset(set(claims["claim_type"]))
    assert claims["evidence_ref_count"].min() >= 1
    assert set(links["claim_id"]) <= set(claims["claim_id"])
    assert set(links["evidence_ref_id"]) <= set(refs["evidence_ref_id"])

    validation = validate_narrative_artifact(manifest_path).to_dict()
    assert validation["status"] == "warning"
    assert validation["feature_state"] == "beta"
    assert validation["checks"]["refs_resolvable"]["status"] == "passed"
    assert validation["counts"]["targets"] == 2
    assert validation["counts"]["aggregate_only_refs"] > 0

    result_manifest = build_result_manifest(root).to_dict()
    assert result_manifest["features"]["narrative"]["state"] == "beta"
    assert "narrative" in result_manifest["features"]["narrative"]["artifact_refs"]
    assert result_manifest["features"]["keyword"]["state"] == "stable"
    assert "narrative_qa" in result_manifest["features"]["quality"]["artifact_refs"]
    artifacts = result_manifest["artifacts"]
    assert artifacts["narrative"]["path"] == "narrative/narrative_manifest.json"
    assert artifacts["narrative_targets"]["path"] == "narrative/narrative_targets.parquet"
    assert artifacts["narrative_claims"]["path"] == "narrative/claims.parquet"
    assert artifacts["narrative_claims"]["role"] == "narrative_table"
    assert artifacts["narrative_claims"]["rows"] == validation["counts"]["claims"]
    assert artifacts["narrative_generation_metadata"]["path"] == "narrative/generation_metadata.json"
    assert artifacts["narrative_generation_metadata"]["schema_version"] == NARRATIVE_GENERATION_METADATA_SCHEMA_VERSION
    assert artifacts["narrative_evidence_refs"]["path"] == "narrative/evidence_refs.parquet"
    assert artifacts["narrative_claim_evidence_links"]["path"] == "narrative/claim_evidence_links.parquet"
    assert artifacts["narrative_sections"]["path"] == "narrative/narrative_sections.parquet"
    assert "narrative_review_decisions" not in artifacts
    assert "narrative/narrative_qa.json" in result_manifest["quality"]["gate_paths"]


def test_validate_narrative_artifact_blocks_unsupported_normal_claim(tmp_path):
    root = _write_valid_result_root(tmp_path / "result")
    write_cooccurrence_artifacts(root)
    write_cluster_review_packet_artifact(root)
    written = write_narrative_evidence_artifacts(root)
    assert written is not None
    claims_path = root / "narrative" / "claims.parquet"
    claims = pd.read_parquet(claims_path)
    claims.loc[0, "support_state"] = "unsupported"
    claims.loc[0, "claim_type"] = "identity"
    claims.to_parquet(claims_path, index=False)

    validation = validate_narrative_artifact(root).to_dict()

    assert validation["status"] == "blocked"
    assert any(issue["code"] == "narrative_unsupported_normal_claims" for issue in validation["blocking_issues"])
    contract = validate_result_root(root).to_dict()
    assert contract["result_state"] == "blocked"
    assert any(warning["code"] == "narrative_unsupported_normal_claims" for warning in contract["warnings"])


def test_validate_narrative_artifact_accepts_model_metadata_sidecar(tmp_path):
    root = _write_valid_result_root(tmp_path / "result")
    write_cooccurrence_artifacts(root)
    write_cluster_review_packet_artifact(root)
    written = write_narrative_evidence_artifacts(root)
    assert written is not None
    claims_path = root / "narrative" / "claims.parquet"
    metadata_path = root / "narrative" / "generation_metadata.json"
    claims = pd.read_parquet(claims_path)
    claims.loc[0, "text_origin"] = "model_generated"
    claims.to_parquet(claims_path, index=False)

    blocked = validate_narrative_artifact(root).to_dict()

    assert blocked["status"] == "blocked"
    assert any(issue["code"] == "narrative_model_metadata_missing" for issue in blocked["blocking_issues"])

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["llm_generation_used"] = True
    metadata["text_origins"] = ["deterministic_template", "model_generated"]
    metadata["model_generation"] = {
        "provider": "test",
        "model": "test-model",
        "model_run_id": "run-test",
        "prompt_ref": "prompt:test",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    validation = validate_narrative_artifact(root).to_dict()

    assert validation["status"] != "blocked"
    assert not any(issue["code"] == "narrative_model_metadata_missing" for issue in validation["blocking_issues"])


def test_write_narrative_publication_artifacts_render_reviewed_claims(tmp_path):
    root = _write_valid_result_root(tmp_path / "result")
    write_cooccurrence_artifacts(root)
    write_cluster_review_packet_artifact(root)
    written = write_narrative_evidence_artifacts(root)
    assert written is not None
    claims_path = root / "narrative" / "claims.parquet"
    reviews_path = root / "narrative" / "review_decisions.parquet"
    manifest_path = root / "narrative" / "narrative_manifest.json"
    claims = pd.read_parquet(claims_path)
    accepted_claim_id = str(claims.iloc[0]["claim_id"])
    rejected_claim_id = str(claims.iloc[1]["claim_id"])
    target_id = str(claims.iloc[0]["target_id"])
    claims.loc[claims["claim_id"].map(str) == accepted_claim_id, "review_state"] = "accepted"
    claims.loc[claims["claim_id"].map(str) == rejected_claim_id, "review_state"] = "rejected"
    claims.to_parquet(claims_path, index=False)
    pd.DataFrame(
        [
            {
                "schema_version": "sciscape_narrative_review_decisions_v1",
                "narrative_id": written["narrative_id"],
                "decision_id": "decision_accept",
                "claim_id": accepted_claim_id,
                "decision_type": "accepted",
                "reviewer": "tester",
                "decided_at_utc": "2026-06-18T00:00:00+00:00",
                "reason": "ready for publication",
                "target_id": target_id,
                "cluster_uid": "cluster:0",
            },
            {
                "schema_version": "sciscape_narrative_review_decisions_v1",
                "narrative_id": written["narrative_id"],
                "decision_id": "decision_reject",
                "claim_id": rejected_claim_id,
                "decision_type": "rejected",
                "reviewer": "tester",
                "decided_at_utc": "2026-06-18T00:01:00+00:00",
                "reason": "do not publish",
                "target_id": target_id,
                "cluster_uid": "cluster:0",
            },
        ]
    ).to_parquet(reviews_path, index=False)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    outputs = dict(manifest.get("outputs") or {})
    outputs["reviews"] = "review_decisions.parquet"
    manifest["outputs"] = outputs
    manifest["review_state_advertised"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    publication = write_narrative_publication_artifacts(root)

    assert publication is not None
    assert publication["available"] is True
    assert publication["schema_version"] == NARRATIVE_PUBLICATION_SCHEMA_VERSION
    assert publication["publication_state"] == "partial_review"
    assert publication["paths"]["bundle"] == "narrative/publication_bundle.zip"
    assert "narrative/publication_summary.html" in publication["bundle_members"]
    payload = json.loads((root / "narrative" / "publication_summary.json").read_text(encoding="utf-8"))
    markdown = (root / "narrative" / "publication_summary.md").read_text(encoding="utf-8")
    html_report = (root / "narrative" / "publication_summary.html").read_text(encoding="utf-8")
    assert payload["schema_version"] == NARRATIVE_PUBLICATION_SCHEMA_VERSION
    assert payload["counts"]["rendered_claims"] == 1
    assert payload["counts"]["rejected_claims"] == 1
    assert accepted_claim_id in markdown
    assert "Omitted Claims" in markdown
    assert rejected_claim_id in markdown
    assert "<!doctype html>" in html_report
    assert accepted_claim_id in html_report
    assert "Omitted Claims" in html_report
    assert rejected_claim_id in html_report
    with zipfile.ZipFile(root / "narrative" / "publication_bundle.zip") as archive:
        names = set(archive.namelist())
    assert {
        "narrative/narrative_manifest.json",
        "narrative/narrative_qa.json",
        "narrative/generation_metadata.json",
        "narrative/publication_summary.json",
        "narrative/publication_summary.md",
        "narrative/publication_summary.html",
        "narrative/review_decisions.parquet",
        "narrative/claims.parquet",
        "narrative/evidence_refs.parquet",
    }.issubset(names)
    manifest_after = build_result_manifest(root).to_dict()
    assert manifest_after["artifacts"]["narrative_publication_json"]["path"] == "narrative/publication_summary.json"
    assert manifest_after["artifacts"]["narrative_publication_markdown"]["path"] == "narrative/publication_summary.md"
    assert manifest_after["artifacts"]["narrative_publication_html"]["path"] == "narrative/publication_summary.html"
    assert manifest_after["artifacts"]["narrative_publication_html"]["format"] == "html"
    assert manifest_after["artifacts"]["narrative_publication_bundle"]["path"] == "narrative/publication_bundle.zip"
    assert manifest_after["artifacts"]["narrative_publication_bundle"]["format"] == "zip"


def test_write_keyword_rule_artifacts_promotes_cleaning_manifest_and_quality_refs(tmp_path):
    root = _write_valid_result_root(tmp_path / "result")
    rules = pd.DataFrame(
        [
            {
                "rule_id": "html_fragment_block",
                "rule_family": "html_fragment",
                "match_type": "regex",
                "pattern": r"class\\s+htmlview",
                "replacement": "",
                "action": "block",
                "confidence_policy": "high_precision_artifact",
                "destructive": True,
                "enabled": True,
                "created_by": "test",
                "reason": "encoded publisher HTML fragment",
            }
        ]
    )
    before_after = pd.DataFrame(
        [
            _keyword_before_after_row(
                cluster_id=0,
                raw_term="class htmlview paragraph",
                rule_ids="html_fragment_block",
                term_after="",
                display_label="",
                quality_flags="metadata_fragment",
                blocked=True,
                block_reason="encoded HTML metadata fragment",
            ),
            _keyword_before_after_row(
                cluster_id=0,
                raw_term="perovskite solar cells",
            ),
        ]
    )

    written = write_keyword_rule_artifacts(
        root,
        rule_set_id="keyword_cleaning_test_v1",
        rules=rules,
        before_after=before_after,
    )

    assert written["manifest_path"] == root / "rules" / "keyword_cleaning_test_v1" / "rule_set_manifest.json"
    assert written["qa"]["schema_version"] == KEYWORD_RULE_QA_SCHEMA_VERSION
    assert written["qa"]["status"] == "passed"
    assert written["qa"]["counts"]["blocked_rows"] == 1
    assert written["qa"]["contamination_counts"]["top_artifact_rows_after"] == 0

    validation = validate_keyword_rule_artifact(written["manifest_path"]).to_dict()
    assert validation["status"] == "passed"
    assert validation["rule_family_counts"]["html_fragment"] == 1

    contract = validate_result_root(root).to_dict()
    assert contract["ok"] is True
    assert contract["counts"]["keyword_rule_artifacts"] == 1
    assert contract["counts"]["stable_keyword_rule_artifacts"] == 1
    assert contract["counts"]["keyword_rule_blocked_rows"] == 1
    assert contract["counts"]["keyword_rule_top_artifact_rows_after"] == 0

    manifest = build_result_manifest(root).to_dict()
    assert manifest["artifacts"]["keyword_rules"]["schema_version"] == KEYWORD_RULE_MANIFEST_SCHEMA_VERSION
    assert manifest["artifacts"]["keyword_rule_qa"]["schema_version"] == KEYWORD_RULE_QA_SCHEMA_VERSION
    assert "keyword_rules" in manifest["features"]["keyword"]["artifact_refs"]
    assert "keyword_rule_qa" in manifest["features"]["quality"]["artifact_refs"]
    assert "rules/keyword_cleaning_test_v1/rule_set_qa.json" in manifest["quality"]["gate_paths"]


def test_validate_keyword_rule_artifact_blocks_unsafe_destructive_stop_rule(tmp_path):
    root = _write_valid_result_root(tmp_path / "result")
    rules = pd.DataFrame(
        [
            {
                "rule_id": "date_stop_block",
                "rule_family": "stop_term",
                "match_type": "literal",
                "pattern": "date",
                "replacement": "",
                "action": "block",
                "confidence_policy": "unsafe_global_stop",
                "destructive": True,
                "enabled": True,
                "created_by": "test",
                "reason": "stop terms must not be destructive blocks by default",
            }
        ]
    )
    before_after = pd.DataFrame(
        [
            _keyword_before_after_row(
                cluster_id=0,
                raw_term="date",
                rule_ids="date_stop_block",
                term_after="",
                display_label="",
                blocked=True,
                block_reason="unsafe stop-term block fixture",
            )
        ]
    )

    written = write_keyword_rule_artifacts(
        root,
        rule_set_id="keyword_cleaning_unsafe_v1",
        rules=rules,
        before_after=before_after,
    )

    assert written["qa"]["status"] == "blocked"
    assert any(issue["code"] == "unsafe_keyword_rule_block_action" for issue in written["qa"]["blocking_issues"])

    contract = validate_result_root(root).to_dict()
    assert contract["ok"] is False
    assert contract["result_state"] == "blocked"
    assert any(w["code"] == "unsafe_keyword_rule_block_action" for w in contract["warnings"])


def test_validate_result_root_blocks_malformed_cooccurrence_table(tmp_path):
    root = _write_valid_result_root(tmp_path / "result")
    pd.DataFrame({"source": ["a"], "target": ["b"]}).to_parquet(
        root / "landscape" / "term_cooccurrence.parquet",
        index=False,
    )

    payload = validate_result_root(root).to_dict()

    assert payload["ok"] is False
    assert payload["result_state"] == "blocked"
    assert any(w["code"] == "missing_columns" and w["artifact"] == "cooccurrence" for w in payload["warnings"])


def test_write_matrix_artifact_promotes_stable_matrix_feature(tmp_path):
    root = _write_valid_result_root(tmp_path / "result")
    entities = pd.DataFrame(
        {
            "entity_key": ["term:perovskite", "term:passivation"],
            "entity_index": [0, 1],
            "entity_type": ["term", "term"],
            "label": ["perovskite", "passivation"],
        }
    )
    values = pd.DataFrame(
        {
            "row_key": ["term:perovskite"],
            "column_key": ["term:passivation"],
            "row_index": [0],
            "column_index": [1],
            "value": [1.0],
            "raw_value": [3.0],
            "support_count": [3],
            "relation": ["cooccurrence"],
        }
    )

    written = write_matrix_artifact(
        root,
        "term_matrix",
        "cooccurrence",
        values,
        entities,
        entities.copy(),
        value_spec={"name": "cooccurrence_weight", "type": "float", "range": [0.0, 1.0]},
        weighting={"raw_metric": "test_pair", "normalization": "max", "symmetric": True, "storage": "upper_triangle"},
        source_artifacts=[{"role": "keywords", "path": "landscape/keywords.parquet"}],
        transforms=[{"step": "synthetic_triplet"}],
    )

    assert written["manifest_path"] == root / "matrices" / "term_matrix" / "matrix_manifest.json"
    assert written["qa"]["schema_version"] == MATRIX_QA_SCHEMA_VERSION
    assert written["qa"]["status"] == "passed"

    table = pd.read_parquet(written["values_path"])
    assert set(table["schema_version"]) == {MATRIX_VALUES_SCHEMA_VERSION}
    assert table["matrix_id"].iloc[0] == "term_matrix"

    validation = validate_matrix_artifact(written["manifest_path"]).to_dict()
    assert validation["status"] == "passed"
    assert validation["counts"]["rows"] == 2
    assert validation["counts"]["columns"] == 2
    assert validation["counts"]["nnz"] == 1

    contract = validate_result_root(root).to_dict()
    assert contract["features"]["matrix"] is True
    assert contract["counts"]["general_matrix_artifacts"] == 1
    assert contract["counts"]["stable_matrix_artifacts"] == 1
    assert contract["counts"]["matrix_nnz"] == 1

    manifest = build_result_manifest(root).to_dict()
    assert manifest["features"]["matrix"]["state"] == "stable"
    assert "matrix" in manifest["features"]["matrix"]["artifact_refs"]
    assert manifest["artifacts"]["matrix"]["schema_version"] == MATRIX_MANIFEST_SCHEMA_VERSION
    assert manifest["artifacts"]["matrix_values"]["path"] == "matrices/term_matrix/matrix_values.parquet"
    assert manifest["artifacts"]["matrix_values"]["schema_version"] == MATRIX_VALUES_SCHEMA_VERSION
    assert manifest["artifacts"]["matrix_rows"]["path"] == "matrices/term_matrix/row_entities.parquet"
    assert manifest["artifacts"]["matrix_rows"]["schema_version"] == MATRIX_ENTITIES_SCHEMA_VERSION
    assert manifest["artifacts"]["matrix_columns"]["path"] == "matrices/term_matrix/column_entities.parquet"
    assert manifest["artifacts"]["matrix_columns"]["schema_version"] == MATRIX_ENTITIES_SCHEMA_VERSION
    assert manifest["artifacts"]["matrix_qa"]["path"] == "matrices/term_matrix/matrix_qa.json"
    assert manifest["artifacts"]["matrix_qa"]["schema_version"] == MATRIX_QA_SCHEMA_VERSION


def test_validate_matrix_artifact_blocks_missing_entity_refs(tmp_path):
    root = _write_valid_result_root(tmp_path / "result")
    rows = pd.DataFrame(
        {
            "entity_key": ["term:perovskite"],
            "entity_index": [0],
            "entity_type": ["term"],
            "label": ["perovskite"],
        }
    )
    columns = pd.DataFrame(
        {
            "entity_key": ["term:passivation"],
            "entity_index": [0],
            "entity_type": ["term"],
            "label": ["passivation"],
        }
    )
    values = pd.DataFrame(
        {
            "row_key": ["term:missing"],
            "column_key": ["term:passivation"],
            "row_index": [0],
            "column_index": [0],
            "value": [1.0],
            "relation": ["cooccurrence"],
        }
    )

    written = write_matrix_artifact(
        root,
        "bad_matrix",
        "cooccurrence",
        values,
        rows,
        columns,
        value_spec={"name": "cooccurrence_weight", "type": "float"},
        weighting={"raw_metric": "test_pair", "normalization": "none"},
        source_artifacts=[{"role": "keywords", "path": "landscape/keywords.parquet"}],
    )

    assert written["qa"]["status"] == "blocked"
    assert any(issue["code"] == "missing_matrix_row_refs" for issue in written["qa"]["blocking_issues"])

    contract = validate_result_root(root).to_dict()
    assert contract["ok"] is False
    assert contract["result_state"] == "blocked"
    assert any(w["code"] == "missing_matrix_row_refs" for w in contract["warnings"])


def test_write_matrix_from_term_cooccurrence_wraps_existing_sidecar(tmp_path):
    root = _write_valid_result_root(tmp_path / "result")
    write_cooccurrence_artifacts(root)

    written = write_matrix_from_term_cooccurrence(root)

    assert written is not None
    assert written["matrix_id"] == "term_cooccurrence_default"
    assert written["qa"]["status"] == "passed"
    values = pd.read_parquet(written["values_path"])
    rows = pd.read_parquet(written["row_entities_path"])
    assert len(values) == 2
    assert len(rows) == 4
    assert set(values["relation"]) == {"term_cooccurrence"}

    contract = validate_result_root(root).to_dict()
    assert contract["counts"]["cooccurrence_artifacts"] == 2
    assert contract["counts"]["general_matrix_artifacts"] == 1
    assert contract["counts"]["stable_matrix_artifacts"] == 1
    assert contract["features"]["matrix"] is True

    manifest = build_result_manifest(root).to_dict()
    assert manifest["features"]["cooccurrence"]["state"] == "stable"
    assert manifest["features"]["matrix"]["state"] == "stable"
    assert manifest["artifacts"]["matrix_values"]["path"] == "matrices/term_cooccurrence_default/matrix_values.parquet"


def test_export_matrix_artifact_writes_manifest_backed_csv_triplets(tmp_path):
    root = _write_valid_result_root(tmp_path / "result")
    write_cooccurrence_artifacts(root)
    write_matrix_from_term_cooccurrence(root)

    written = export_matrix_artifact(root, matrix_id="term_cooccurrence_default", export_format="csv-triplets")

    assert written["primary_path"] == (
        root / "exports" / "matrix_term_cooccurrence_default_csv_triplets" / "matrix_triplets.csv"
    )
    table = pd.read_csv(written["primary_path"])
    assert {"row_key", "column_key", "row_label", "column_label", "value", "relation"}.issubset(table.columns)
    assert set(table["relation"]) == {"term_cooccurrence"}
    validation = validate_export_manifest(written["manifest_path"]).to_dict()
    assert validation["status"] == "passed"
    assert validation["export_family"] == "matrix"
    assert validation["export_kind"] == "matrix_csv_triplets"
    assert validation["counts"]["files"] == 1
    assert validation["counts"]["inputs"] == 5

    export_manifest = json.loads(written["manifest_path"].read_text(encoding="utf-8"))
    assert export_manifest["selection"]["scope"] == "matrix_artifact"
    assert export_manifest["selection"]["view"]["mode"] == "matrix_csv_triplets"
    assert export_manifest["selection"]["layer_state"]["matrix_id"] == "term_cooccurrence_default"
    assert export_manifest["selection"]["layer_state"]["nnz"] == 2

    manifest = build_result_manifest(root).to_dict()
    exports = [
        export
        for export in manifest["exports"]
        if export["export_id"] == "matrix_term_cooccurrence_default_csv_triplets"
    ]
    assert len(exports) == 1
    assert exports[0]["export_family"] == "matrix"
    assert exports[0]["path"] == "exports/matrix_term_cooccurrence_default_csv_triplets/matrix_triplets.csv"


def test_export_matrix_artifact_writes_json_summary(tmp_path):
    root = _write_valid_result_root(tmp_path / "result")
    write_cooccurrence_artifacts(root)
    matrix = write_matrix_from_term_cooccurrence(root)

    written = export_matrix_artifact(matrix["manifest_path"], export_format="json-summary")

    payload = json.loads(written["primary_path"].read_text(encoding="utf-8"))
    assert payload["schema_version"] == "sciscape_matrix_export_summary_v1"
    assert payload["matrix_id"] == "term_cooccurrence_default"
    assert payload["counts"]["nnz"] == 2
    validation = validate_export_manifest(written["manifest_path"]).to_dict()
    assert validation["status"] == "passed"
    assert validation["export_kind"] == "matrix_json_summary"


def test_export_matrix_artifact_writes_vosviewer_network(tmp_path):
    root = _write_valid_result_root(tmp_path / "result")
    write_cooccurrence_artifacts(root)
    write_matrix_from_term_cooccurrence(root)

    written = export_matrix_artifact(
        root,
        matrix_id="term_cooccurrence_default",
        export_format="vosviewer-network",
    )

    export_dir = root / "exports" / "matrix_term_cooccurrence_default_vosviewer_network"
    map_path = export_dir / "vosviewer_matrix_map.txt"
    network_path = export_dir / "vosviewer_matrix_network.txt"
    assert written["primary_path"] == map_path
    assert written["map_path"] == map_path
    assert written["network_path"] == network_path
    assert map_path.exists()
    assert network_path.exists()

    map_lines = map_path.read_text(encoding="utf-8").splitlines()
    assert map_lines[0].split("\t") == [
        "id",
        "label",
        "description",
        "cluster",
        "weight<Links>",
        "weight<Total link strength>",
        "score<Entity index>",
    ]
    assert network_path.read_text(encoding="utf-8").splitlines() == [
        "1\t4\t1.000000",
        "2\t3\t1.000000",
    ]

    validation = validate_export_manifest(written["manifest_path"]).to_dict()
    assert validation["status"] == "passed"
    assert validation["export_family"] == "vosviewer"
    assert validation["export_kind"] == "matrix_vosviewer_network"
    assert validation["counts"]["files"] == 2
    assert validation["counts"]["inputs"] == 5

    export_manifest = json.loads(written["manifest_path"].read_text(encoding="utf-8"))
    assert export_manifest["selection"]["scope"] == "matrix_artifact"
    assert export_manifest["selection"]["view"]["mode"] == "matrix_vosviewer_network"
    assert export_manifest["selection"]["thresholds"] == {"min_link_strength": 0}
    assert export_manifest["selection"]["layer_state"] == {
        "matrix_id": "term_cooccurrence_default",
        "matrix_family": "cooccurrence",
        "matrix_format": "sparse_triplet",
        "export_format": "vosviewer-network",
        "row_count": 4,
        "column_count": 4,
        "nnz": 2,
        "map_file": "vosviewer_matrix_map.txt",
        "network_file": "vosviewer_matrix_network.txt",
        "term_count": 4,
        "link_count": 2,
        "counting_method": "matrix_value_sum",
    }

    manifest = build_result_manifest(root).to_dict()
    exports = [
        export
        for export in manifest["exports"]
        if export["export_id"] == "matrix_term_cooccurrence_default_vosviewer_network"
    ]
    assert len(exports) == 1
    assert exports[0]["export_family"] == "vosviewer"
    assert (
        exports[0]["path"]
        == "exports/matrix_term_cooccurrence_default_vosviewer_network/vosviewer_matrix_map.txt"
    )
    assert {row["role"]: row["path"] for row in exports[0]["files"]} == {
        "map": "exports/matrix_term_cooccurrence_default_vosviewer_network/vosviewer_matrix_map.txt",
        "network": "exports/matrix_term_cooccurrence_default_vosviewer_network/vosviewer_matrix_network.txt",
    }


def test_write_temporal_artifacts_promotes_stable_temporal_feature(tmp_path):
    root = _write_valid_result_root(tmp_path / "result")
    records = pd.read_parquet(root / "abstracts.parquet")
    membership = pd.read_parquet(root / "landscape" / "membership.parquet")
    keywords = pd.read_parquet(root / "landscape" / "keywords.parquet")
    keywords["pub_year_series"] = [
        {"2021": 1, "2022": 2},
        {"2021": 1},
        {"2021": 2, "2022": 1},
        {"2022": 1},
    ]

    written = write_temporal_artifacts(
        root,
        temporal_id="yearly_trends",
        records_df=records,
        membership_df=membership,
        keywords_df=keywords,
        event_methods=["growth_rate"],
    )

    assert written["manifest_path"] == root / "temporal" / "temporal_manifest.json"
    assert written["qa"]["schema_version"] == TEMPORAL_QA_SCHEMA_VERSION
    assert written["qa"]["status"] == "passed"

    activity = pd.read_parquet(written["activity_path"])
    series = pd.read_parquet(written["series_path"])
    events = pd.read_parquet(written["events_path"])
    assert set(activity["schema_version"]) == {TEMPORAL_ACTIVITY_SCHEMA_VERSION}
    assert activity["doc_count"].sum() == 4
    assert set(series["entity_type"]) == {"result", "cluster", "term"}
    assert set(series["metric"]) == {"doc_count", "pub_year_series"}
    assert not events.empty

    validation = validate_temporal_artifact(written["manifest_path"]).to_dict()
    assert validation["status"] == "passed"
    assert validation["counts"]["periods"] == 2
    assert validation["counts"]["activity_rows"] == 2
    assert validation["counts"]["event_rows"] == len(events)

    contract = validate_result_root(root).to_dict()
    assert contract["features"]["temporal"] is True
    assert contract["counts"]["temporal_artifacts"] == 1
    assert contract["counts"]["stable_temporal_artifacts"] == 1
    assert contract["counts"]["temporal_periods"] == 2
    assert contract["counts"]["temporal_series_rows"] == len(series)

    manifest = build_result_manifest(root).to_dict()
    assert manifest["features"]["temporal"]["state"] == "stable"
    assert "temporal" in manifest["features"]["temporal"]["artifact_refs"]
    assert manifest["artifacts"]["temporal"]["schema_version"] == TEMPORAL_MANIFEST_SCHEMA_VERSION


def test_validate_temporal_artifact_blocks_unknown_series_period(tmp_path):
    root = _write_valid_result_root(tmp_path / "result")
    records = pd.read_parquet(root / "abstracts.parquet")
    membership = pd.read_parquet(root / "landscape" / "membership.parquet")
    written = write_temporal_artifacts(
        root,
        temporal_id="yearly_trends",
        records_df=records,
        membership_df=membership,
    )
    series_path = written["series_path"]
    series = pd.read_parquet(series_path)
    series.loc[0, "period_id"] = "year:2099"
    series.to_parquet(series_path, index=False)

    validation = validate_temporal_artifact(written["manifest_path"]).to_dict()

    assert validation["status"] == "blocked"
    assert any(issue["code"] == "unknown_temporal_series_periods" for issue in validation["blocking_issues"])

    contract = validate_result_root(root).to_dict()
    assert contract["ok"] is False
    assert contract["result_state"] == "blocked"
    assert any(w["code"] == "unknown_temporal_series_periods" for w in contract["warnings"])


def test_result_manifest_marks_pubyear_only_temporal_as_beta(tmp_path):
    root = _write_valid_result_root(tmp_path / "result")

    manifest = build_result_manifest(root).to_dict()

    assert manifest["features"]["temporal"]["state"] == "beta"
    assert manifest["features"]["temporal"]["reason"] == "pubyear exists but no temporal artifact has been written yet"


def test_write_evolution_synthetic_smoke_covers_all_event_types(tmp_path):
    root = _write_valid_result_root(tmp_path / "result")

    written = write_evolution_synthetic_smoke_artifact(root)

    assert written["manifest_path"] == root / "evolution" / "evolution_manifest.json"
    assert written["qa"]["schema_version"] == EVOLUTION_QA_SCHEMA_VERSION
    assert written["qa"]["status"] == "passed"

    events = pd.read_parquet(written["events_path"])
    assert set(events["schema_version"]) == {EVOLUTION_EVENTS_SCHEMA_VERSION}
    assert events["event_type"].value_counts().to_dict() == {
        "continuation": 3,
        "split": 1,
        "merge": 1,
        "emergence": 1,
        "decline": 1,
        "ambiguous": 1,
    }

    validation = validate_evolution_artifact(written["manifest_path"]).to_dict()
    assert validation["status"] == "passed"
    assert validation["event_counts"] == {
        "ambiguous": 1,
        "continuation": 3,
        "decline": 1,
        "emergence": 1,
        "merge": 1,
        "split": 1,
    }

    contract = validate_result_root(root).to_dict()
    assert contract["features"]["evolution"] is True
    assert contract["counts"]["evolution_artifacts"] == 1
    assert contract["counts"]["stable_evolution_artifacts"] == 1
    assert contract["counts"]["evolution_event_rows"] == len(events)

    manifest = build_result_manifest(root).to_dict()
    assert manifest["features"]["evolution"]["state"] == "stable"
    assert "evolution" in manifest["features"]["evolution"]["artifact_refs"]
    assert manifest["artifacts"]["evolution"]["schema_version"] == EVOLUTION_MANIFEST_SCHEMA_VERSION


def test_validate_evolution_artifact_blocks_invalid_matching_diagnostics(tmp_path):
    root = _write_valid_result_root(tmp_path / "result")
    written = write_evolution_synthetic_smoke_artifact(root)
    manifest_path = written["manifest_path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["matching_method"]["diagnostics"] = {
        "source": "",
        "state_count": 15,
        "candidate_transition_rows": "bad",
        "retained_transition_rows": 9,
        "dropped_transition_rows": 0,
        "min_transition_score": 0.5,
        "min_support_count": 1,
        "slice_pair_count": 1,
        "slice_pairs": [
            {
                "source_slice_id": "year:2020",
                "target_slice_id": "year:2021",
                "source_slice_index": 0,
                "target_slice_index": 1,
                "candidate_count": 9,
                "retained_count": 9,
                "dropped_count": 0,
            }
        ],
        "relation_counts": {"continuation": 3},
        "warning_flag_counts": {},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    validation = validate_evolution_artifact(manifest_path).to_dict()

    assert validation["status"] == "blocked"
    assert validation["checks"]["matching_diagnostics"]["status"] == "blocked"
    assert any(issue["code"] == "invalid_evolution_matching_diagnostics" for issue in validation["blocking_issues"])


def test_write_evolution_artifacts_promotes_stable_evolution_feature(tmp_path):
    root = _write_valid_result_root(tmp_path / "result")
    records = pd.read_parquet(root / "abstracts.parquet")
    membership = pd.read_parquet(root / "landscape" / "membership.parquet")
    keywords = pd.read_parquet(root / "landscape" / "keywords.parquet")

    written = write_evolution_artifacts(
        root,
        evolution_id="yearly_cluster_evolution",
        records_df=records,
        membership_df=membership,
        keywords_df=keywords,
    )

    assert written["manifest_path"] == root / "evolution" / "evolution_manifest.json"
    assert written["qa"]["schema_version"] == EVOLUTION_QA_SCHEMA_VERSION
    assert written["qa"]["status"] == "passed"

    states = pd.read_parquet(written["cluster_states_path"])
    state_membership = pd.read_parquet(written["state_membership_path"])
    transitions = pd.read_parquet(written["transitions_path"])
    events = pd.read_parquet(written["events_path"])
    assert len(states) == 4
    assert len(state_membership) == 4
    assert set(state_membership["schema_version"]) == {EVOLUTION_STATE_MEMBERSHIP_SCHEMA_VERSION}
    assert len(transitions) == 2
    assert set(events["event_type"]) == {"continuation"}

    validation = validate_evolution_artifact(written["manifest_path"]).to_dict()
    assert validation["status"] == "passed"
    assert validation["counts"]["slices"] == 2
    assert validation["counts"]["states"] == 4
    assert validation["counts"]["state_membership_rows"] == 4
    assert validation["counts"]["transitions"] == 2

    contract = validate_result_root(root).to_dict()
    assert contract["features"]["evolution"] is True
    assert contract["counts"]["stable_evolution_artifacts"] == 1
    assert contract["counts"]["evolution_slices"] == 2
    assert contract["counts"]["evolution_transitions"] == 2

    manifest = build_result_manifest(root).to_dict()
    assert manifest["features"]["evolution"]["state"] == "stable"
    assert manifest["artifacts"]["evolution_state_membership"]["path"] == "evolution/state_membership.parquet"
    assert manifest["artifacts"]["evolution_state_membership"]["rows"] == 4


def test_write_evidence_backed_evolution_artifacts_promotes_stable_feature(tmp_path):
    root = _write_valid_result_root(tmp_path / "result")
    slices = pd.DataFrame(
        [
            {"slice_id": "year:2020", "slice_index": 0, "start_year": 2020, "end_year": 2020},
            {"slice_id": "year:2021", "slice_index": 1, "start_year": 2021, "end_year": 2021},
        ]
    )
    state_evidence = pd.DataFrame(
        [
            {"slice_id": "year:2020", "cluster_id": "A", "doc_count": 6, "top_terms": ["alpha"]},
            {"slice_id": "year:2021", "cluster_id": "B1", "doc_count": 3, "top_terms": ["beta"]},
            {"slice_id": "year:2021", "cluster_id": "B2", "doc_count": 3, "top_terms": ["gamma"]},
            {"slice_id": "year:2020", "cluster_id": "C", "doc_count": 4},
            {"slice_id": "year:2021", "cluster_id": "C", "doc_count": 4},
        ]
    )
    transition_evidence = pd.DataFrame(
        [
            {"source_state_id": "year:2020_cluster:A", "target_state_id": "year:2021_cluster:B1", "score": 0.8, "support_count": 3},
            {"source_state_id": "year:2020_cluster:A", "target_state_id": "year:2021_cluster:B2", "score": 0.7, "support_count": 3},
            {"source_state_id": "year:2020_cluster:C", "target_state_id": "year:2021_cluster:C", "score": 0.9, "support_count": 4},
        ]
    )

    written = write_evidence_backed_evolution_artifacts(
        root,
        evolution_id="slice_local_evolution",
        slices_df=slices,
        state_evidence_df=state_evidence,
        transition_evidence_df=transition_evidence,
        metric="term_overlap",
    )

    assert written["manifest_path"] == root / "evolution" / "evolution_manifest.json"
    assert written["qa"]["schema_version"] == EVOLUTION_QA_SCHEMA_VERSION
    assert written["qa"]["status"] == "passed"

    states = pd.read_parquet(written["cluster_states_path"])
    transitions = pd.read_parquet(written["transitions_path"])
    events = pd.read_parquet(written["events_path"])
    assert len(states) == 5
    assert len(transitions) == 3
    assert {"split", "continuation"} <= set(events["event_type"])
    assert set(transitions["metric"]) == {"term_overlap"}

    validation = validate_evolution_artifact(written["manifest_path"]).to_dict()
    assert validation["status"] == "passed"
    assert validation["counts"]["slices"] == 2
    assert validation["counts"]["states"] == 5
    assert validation["counts"]["transitions"] == 3

    contract = validate_result_root(root).to_dict()
    assert contract["features"]["evolution"] is True
    assert contract["counts"]["stable_evolution_artifacts"] == 1
    assert contract["counts"]["evolution_event_rows"] == len(events)

    manifest = build_result_manifest(root).to_dict()
    assert manifest["features"]["evolution"]["state"] == "stable"
    assert manifest["artifacts"]["evolution"]["path"] == "evolution/evolution_manifest.json"
    assert manifest["artifacts"]["evolution_cluster_states"]["path"] == "evolution/cluster_states.parquet"
    assert manifest["artifacts"]["evolution_cluster_states"]["rows"] == 5
    assert manifest["artifacts"]["evolution_events"]["path"] == "evolution/evolution_events.parquet"
    assert manifest["artifacts"]["evolution_events"]["rows"] == len(events)
    assert manifest["artifacts"]["evolution_qa"]["path"] == "evolution/evolution_qa.json"
    assert manifest["artifacts"]["evolution_qa"]["role"] == "qa"
    assert "evolution_cluster_states" in manifest["features"]["evolution"]["artifact_refs"]
    assert "evolution_qa" in manifest["features"]["evolution"]["artifact_refs"]
    evolution_manifest = json.loads(written["manifest_path"].read_text(encoding="utf-8"))
    assert evolution_manifest["matching_method"]["metric"] == "term_overlap"
    assert evolution_manifest["matching_method"]["diagnostics"]["source"] == "explicit_transition_evidence"
    assert evolution_manifest["matching_method"]["diagnostics"]["retained_transition_rows"] == 3


def test_write_document_overlap_evolution_artifacts_promotes_stable_feature(tmp_path):
    root = _write_valid_result_root(tmp_path / "result")
    slices = pd.DataFrame(
        [
            {"slice_id": "year:2020", "slice_index": 0, "start_year": 2020, "end_year": 2020},
            {"slice_id": "year:2021", "slice_index": 1, "start_year": 2021, "end_year": 2021},
        ]
    )
    state_evidence = pd.DataFrame(
        [
            {"slice_id": "year:2020", "cluster_id": "A", "doc_count": 4, "top_terms": ["alpha"]},
            {"slice_id": "year:2021", "cluster_id": "B1", "doc_count": 2, "top_terms": ["beta"]},
            {"slice_id": "year:2021", "cluster_id": "B2", "doc_count": 2, "top_terms": ["gamma"]},
            {"slice_id": "year:2020", "cluster_id": "C", "doc_count": 3},
            {"slice_id": "year:2021", "cluster_id": "C", "doc_count": 3},
        ]
    )
    state_membership = pd.DataFrame(
        [
            *[{"slice_id": "year:2020", "cluster_id": "A", "uid": f"A{i}"} for i in range(4)],
            *[{"slice_id": "year:2021", "cluster_id": "B1", "uid": f"A{i}"} for i in range(2)],
            *[{"slice_id": "year:2021", "cluster_id": "B2", "uid": f"A{i}"} for i in range(2, 4)],
            *[{"slice_id": "year:2020", "cluster_id": "C", "uid": f"C{i}"} for i in range(3)],
            *[{"slice_id": "year:2021", "cluster_id": "C", "uid": f"C{i}"} for i in range(3)],
        ]
    )

    written = write_document_overlap_evolution_artifacts(
        root,
        evolution_id="overlap_evolution",
        slices_df=slices,
        state_evidence_df=state_evidence,
        state_membership_df=state_membership,
        matching_method={"min_transition_score": 0.5, "min_support_count": 2},
    )

    assert written["manifest_path"] == root / "evolution" / "evolution_manifest.json"
    assert written["qa"]["status"] == "passed"

    transitions = pd.read_parquet(written["transitions_path"])
    state_membership = pd.read_parquet(written["state_membership_path"])
    events = pd.read_parquet(written["events_path"])
    assert len(transitions) == 3
    assert len(state_membership) == 14
    assert set(state_membership["schema_version"]) == {EVOLUTION_STATE_MEMBERSHIP_SCHEMA_VERSION}
    assert set(transitions["metric"]) == {"jaccard_doc_overlap"}
    assert "split" in set(events["event_type"])

    validation = validate_evolution_artifact(written["manifest_path"]).to_dict()
    assert validation["status"] == "passed"
    assert validation["counts"]["transitions"] == 3
    assert validation["counts"]["state_membership_rows"] == 14

    manifest = build_result_manifest(root).to_dict()
    assert manifest["features"]["evolution"]["state"] == "stable"
    assert manifest["artifacts"]["evolution_state_membership"]["path"] == "evolution/state_membership.parquet"
    evolution_manifest = json.loads(written["manifest_path"].read_text(encoding="utf-8"))
    assert evolution_manifest["matching_method"]["normalization"] == "state_document_membership_overlap"
    diagnostics = evolution_manifest["matching_method"]["diagnostics"]
    assert diagnostics["source"] == "state_document_membership_overlap"
    assert diagnostics["candidate_transition_rows"] == 3
    assert diagnostics["retained_transition_rows"] == 3
    assert diagnostics["slice_pairs"][0]["candidate_count"] == 3


def test_write_slice_membership_evolution_artifacts_promotes_stable_feature(tmp_path):
    root = _write_valid_result_root(tmp_path / "result")
    records = pd.DataFrame(
        {
            "uid": ["A20", "A21", "A22", "B21", "B22", "C20"],
            "pubyear": [2020, 2021, 2022, 2021, 2022, 2020],
        }
    )
    membership = pd.DataFrame(
        {
            "uid": ["A20", "A21", "A22", "B21", "B22", "C20"],
            "cluster": ["A", "A", "A", "B", "B", "C"],
        }
    )
    keywords = pd.DataFrame(
        {
            "cluster_id": ["A", "B", "C"],
            "term": ["alpha", "beta", "carbon"],
        }
    )

    written = write_slice_membership_evolution_artifacts(
        root,
        evolution_id="membership_overlap",
        records_df=records,
        membership_df=membership,
        keywords_df=keywords,
        metric="overlap_min",
        periodization={"window_years": 2, "step_years": 1},
        matching_method={"min_transition_score": 0.5, "min_support_count": 1},
        source_artifacts=[
            {"role": "records", "path": "abstracts.parquet"},
            {"role": "membership", "path": "landscape/membership.parquet"},
        ],
    )

    assert written["manifest_path"] == root / "evolution" / "evolution_manifest.json"
    assert written["qa"]["status"] == "passed"

    transitions = pd.read_parquet(written["transitions_path"])
    states = pd.read_parquet(written["cluster_states_path"])
    state_membership = pd.read_parquet(written["state_membership_path"])
    assert set(states["cluster_label"]) >= {"alpha", "beta", "carbon"}
    assert len(state_membership) == 8
    assert set(transitions["metric"]) == {"overlap_min"}
    assert len(transitions) == 2

    validation = validate_evolution_artifact(written["manifest_path"]).to_dict()
    assert validation["status"] == "passed"
    assert validation["counts"]["state_membership_rows"] == 8

    manifest = build_result_manifest(root).to_dict()
    assert manifest["features"]["evolution"]["state"] == "stable"
    assert manifest["artifacts"]["evolution_state_membership"]["path"] == "evolution/state_membership.parquet"
    evolution_manifest = json.loads(written["manifest_path"].read_text(encoding="utf-8"))
    assert evolution_manifest["matching_method"]["normalization"] == "periodized_slice_membership_document_overlap"
    assert "project_membership_to_time_slices" in [row["step"] for row in evolution_manifest["transforms"]]


def test_write_slice_local_membership_evolution_artifacts_promotes_stable_feature(tmp_path):
    root = _write_valid_result_root(tmp_path / "result")
    slice_membership = pd.DataFrame(
        [
            *[
                {"slice_id": "year:2020", "slice_index": 0, "start_year": 2020, "end_year": 2020, "uid": f"A{i}", "cluster_id": "A"}
                for i in range(4)
            ],
            *[
                {"slice_id": "year:2021", "slice_index": 1, "start_year": 2021, "end_year": 2021, "uid": f"A{i}", "cluster_id": "B1"}
                for i in range(2)
            ],
            *[
                {"slice_id": "year:2021", "slice_index": 1, "start_year": 2021, "end_year": 2021, "uid": f"A{i}", "cluster_id": "B2"}
                for i in range(2, 4)
            ],
            *[
                {"slice_id": "year:2020", "slice_index": 0, "start_year": 2020, "end_year": 2020, "uid": f"C{i}", "cluster_id": "C"}
                for i in range(3)
            ],
            *[
                {"slice_id": "year:2021", "slice_index": 1, "start_year": 2021, "end_year": 2021, "uid": f"C{i}", "cluster_id": "C"}
                for i in range(3)
            ],
        ]
    )
    source_path = root / "evolution_inputs" / "slice_membership.parquet"
    source_path.parent.mkdir(parents=True)
    slice_membership.to_parquet(source_path, index=False)

    written = write_slice_local_membership_evolution_artifacts(
        root,
        evolution_id="slice_local_overlap",
        slice_membership_df=slice_membership,
        metric="overlap_min",
        matching_method={"min_transition_score": 0.5, "min_support_count": 2},
        source_artifacts=[{"role": "slice_membership", "path": "evolution_inputs/slice_membership.parquet"}],
        default_level="micro",
    )

    assert written["qa"]["status"] == "passed"
    transitions = pd.read_parquet(written["transitions_path"])
    states = pd.read_parquet(written["cluster_states_path"])
    state_membership = pd.read_parquet(written["state_membership_path"])
    assert len(states) == 5
    assert len(state_membership) == 14
    assert len(transitions) == 3

    validation = validate_evolution_artifact(written["manifest_path"]).to_dict()
    assert validation["status"] == "passed"
    assert validation["event_counts"]["split"] == 1
    assert validation["counts"]["state_membership_rows"] == 14

    manifest = build_result_manifest(root).to_dict()
    assert manifest["features"]["evolution"]["state"] == "stable"
    evolution_manifest = json.loads(written["manifest_path"].read_text(encoding="utf-8"))
    assert evolution_manifest["matching_method"]["normalization"] == "slice_local_membership_document_overlap"
    assert evolution_manifest["slice_method"]["state_method"] == "slice_local_membership"
    assert "derive_slice_local_cluster_states" in [row["step"] for row in evolution_manifest["transforms"]]


def test_write_slice_reclustering_evolution_artifacts_promotes_stable_feature(tmp_path):
    root = _write_valid_result_root(tmp_path / "result")
    records = pd.DataFrame(
        {
            "uid": ["A20", "A21", "A22", "C20", "C21", "C22"],
            "pubyear": [2020, 2021, 2022, 2020, 2021, 2022],
        }
    )
    edges = pd.DataFrame(
        {
            "uid1": ["A20", "A21", "C20", "C21"],
            "uid2": ["A21", "A22", "C21", "C22"],
            "rel_sum2": [2.0, 2.0, 2.0, 2.0],
        }
    )
    source_dir = root / "evolution_inputs"
    source_dir.mkdir(parents=True)
    records.to_parquet(source_dir / "records.parquet", index=False)
    edges.to_parquet(source_dir / "edges.parquet", index=False)
    slice_membership_output = root / "evolution_work" / "slice_reclustering_membership.parquet"
    slice_membership_parts_dir = root / "evolution_work" / "slice_reclustering_membership_parts"
    progress_path = root / "evolution_work" / "slice_reclustering_progress.json"

    written = write_slice_reclustering_evolution_artifacts(
        root,
        evolution_id="slice_recluster_overlap",
        records_df=records,
        edges_df=edges,
        metric="overlap_min",
        periodization={"window_years": 2, "step_years": 1},
        matching_method={"min_transition_score": 0.5, "min_support_count": 1},
        source_artifacts=[
            {"role": "records", "path": "evolution_inputs/records.parquet"},
            {"role": "edges", "path": "evolution_inputs/edges.parquet"},
        ],
        resolution=0.01,
        backend="igraph",
        max_workers=2,
        slice_membership_output=slice_membership_output,
        slice_membership_parts_dir=slice_membership_parts_dir,
        progress_path=progress_path,
    )

    assert written["qa"]["status"] == "passed"
    assert written["slice_membership_path"] == slice_membership_output.resolve()
    assert written["slice_membership_parts_dir"] == slice_membership_parts_dir.resolve()
    assert written["progress_path"] == progress_path.resolve()
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert progress["status"] == "completed"
    assert progress["membership_rows"] == 8
    assert progress["params"]["max_workers"] == 2
    assert progress["membership_part_count"] == 2
    part_files = sorted(slice_membership_parts_dir.glob("*.parquet"))
    assert len(part_files) == 2
    assert sum(len(pd.read_parquet(path)) for path in part_files) == 8
    generated_membership = pd.read_parquet(slice_membership_output)
    assert len(generated_membership) == 8
    assert set(generated_membership["backend"]) == {"igraph"}
    validation = validate_evolution_artifact(written["manifest_path"]).to_dict()
    assert validation["status"] == "passed"
    assert validation["counts"]["slices"] == 2
    assert validation["counts"]["states"] == 4
    assert validation["counts"]["transitions"] == 2
    assert validation["event_counts"]["continuation"] == 2

    evolution_manifest = json.loads(written["manifest_path"].read_text(encoding="utf-8"))
    assert evolution_manifest["matching_method"]["normalization"] == "slice_reclustering_document_overlap"
    transforms = evolution_manifest["transforms"]
    assert "run_slice_local_reclustering" in [row["step"] for row in transforms]
    recluster_transform = next(row for row in transforms if row["step"] == "run_slice_local_reclustering")
    assert recluster_transform["max_workers"] == 2
    assert recluster_transform["slice_membership_output"] == "evolution_work/slice_reclustering_membership.parquet"
    assert recluster_transform["slice_membership_parts_dir"] == "evolution_work/slice_reclustering_membership_parts"
    assert recluster_transform["progress_path"] == "evolution_work/slice_reclustering_progress.json"
    assert recluster_transform["slice_membership_rows"] == 8
    manifest = build_result_manifest(root).to_dict()
    assert manifest["features"]["evolution"]["state"] == "stable"


def test_result_artifact_inference_accepts_evolution_manifest_path(tmp_path):
    root = _write_valid_result_root(tmp_path / "result")
    written = write_evolution_synthetic_smoke_artifact(root)

    artifacts = infer_result_artifacts(written["manifest_path"])
    assert artifacts.result_root == root
    assert artifacts.evolution_manifest_paths == (written["manifest_path"],)

    contract = validate_result_root(written["manifest_path"]).to_dict()
    assert contract["result_root"] == str(root)
    assert contract["features"]["evolution"] is True
    assert contract["counts"]["stable_evolution_artifacts"] == 1


def test_validate_evolution_artifact_blocks_unknown_transition_state_ref(tmp_path):
    root = _write_valid_result_root(tmp_path / "result")
    written = write_evolution_synthetic_smoke_artifact(root)
    transitions_path = written["transitions_path"]
    transitions = pd.read_parquet(transitions_path)
    transitions.loc[0, "target_state_id"] = "missing_state"
    transitions.to_parquet(transitions_path, index=False)

    validation = validate_evolution_artifact(written["manifest_path"]).to_dict()

    assert validation["status"] == "blocked"
    assert any(issue["code"] == "missing_evolution_transition_state_refs" for issue in validation["blocking_issues"])

    contract = validate_result_root(root).to_dict()
    assert contract["ok"] is False
    assert contract["result_state"] == "blocked"
    assert any(w["code"] == "missing_evolution_transition_state_refs" for w in contract["warnings"])


def test_validate_result_root_blocks_advertised_missing_feature(tmp_path):
    report = tmp_path / "result" / "landscape" / "report"
    report.mkdir(parents=True)
    (report / "data.json").write_text(
        json.dumps(
            {
                "0": {"label": "empty", "keywords": []},
                "_sciscape": {
                    "features": {"term_network": True},
                    "schema_version": "sciscape_report_data_contract_v1",
                },
            }
        ),
        encoding="utf-8",
    )

    result = validate_result_root(report / "data.json")
    payload = result.to_dict()

    assert payload["ok"] is False
    assert payload["result_state"] == "blocked"
    assert any(w["code"] == "advertised_feature_missing" for w in payload["warnings"])


def test_validate_result_root_detects_evolution_and_narrative_artifacts(tmp_path):
    root = _write_valid_result_root(tmp_path / "result")
    landscape = root / "landscape"
    (landscape / "cluster_evolution.json").write_text(
        json.dumps({"clusters": [{"cluster_uid": "cluster:0", "yearly_activity": [{"year": 2022, "count": 2}]}]}),
        encoding="utf-8",
    )
    (landscape / "cluster_narratives.json").write_text(
        json.dumps({"clusters": [{"cluster_uid": "cluster:0", "summary": "Evidence-backed summary."}]}),
        encoding="utf-8",
    )

    result = validate_result_root(root)
    payload = result.to_dict()

    assert payload["ok"] is True
    assert payload["features"]["evolution"] is True
    assert payload["features"]["narrative"] is True
    assert payload["counts"]["evolution_artifacts"] == 1
    assert payload["counts"]["narrative_artifacts"] == 1
    assert payload["artifacts"]["evolution_artifacts"] == ["landscape/cluster_evolution.json"]
    assert payload["artifacts"]["narrative_artifacts"] == ["landscape/cluster_narratives.json"]


def test_report_data_contract_detects_embedded_evolution_and_narrative():
    report_data = {
        "clusters": [
            {
                "cluster_uid": "cluster:0",
                "label": "perovskite solar cells",
                "keywords": [{"term": "perovskite solar cells"}],
                "evolution": {"yearly_activity": [{"year": 2022, "count": 2}]},
                "narrative": {"summary": "Evidence-backed summary."},
            }
        ]
    }

    contract = build_report_data_contract(report_data)

    assert contract["features"]["evolution"] is True
    assert contract["features"]["narrative"] is True


def test_report_data_contract_embeds_minimal_atlas_nodes():
    report_data = {
        "7": {
            "label": "perovskite solar cells, interface passivation",
            "doc_count": 12,
            "x": 0.25,
            "y": -0.5,
            "keywords": [
                {
                    "term": "perovskite solar cells",
                    "score": 0.91,
                    "frequency": 7,
                    "keyword_label_tier": "primary_phrase",
                }
            ],
        }
    }

    atlas = build_atlas_payload_from_report_data(report_data)
    contract = build_report_data_contract(report_data)

    assert atlas["schema_version"] == "sciscape_atlas_payload_v1"
    assert atlas["levels"] == ["cluster"]
    assert atlas["node_count"] == 1
    node = atlas["nodes"][0]
    assert node["cluster_uid"] == "cluster:7"
    assert node["level"] == "cluster"
    assert node["cluster_id"] == 7
    assert node["short_label"] == "perovskite solar cells"
    assert node["doc_count"] == 12
    assert node["doc_count_source"] == "doc_count"
    assert node["keyword_count"] == 1
    assert node["child_count"] == 0
    assert node["x"] == 0.25
    assert node["y"] == -0.5
    assert node["keywords"][0]["rank"] == 1
    assert node["keywords"][0]["term"] == "perovskite solar cells"
    assert contract["atlas"]["nodes"][0]["cluster_uid"] == "cluster:7"


def test_atlas_render_payload_builds_deck_layer_rows_from_atlas_payload():
    atlas = {
        "schema_version": "sciscape_atlas_payload_v1",
        "levels": ["macro", "micro"],
        "nodes": [
            {
                "cluster_uid": "macro:10",
                "level": "macro",
                "cluster_id": 10,
                "label": "Energy systems",
                "short_label": "Energy systems",
                "doc_count": 12,
                "child_count": 1,
                "x": 2.0,
                "y": -1.0,
            },
            {
                "cluster_uid": "micro:100",
                "level": "micro",
                "cluster_id": 100,
                "parent_uid": "macro:10",
                "label": "Solar cells",
                "short_label": "Solar cells",
                "doc_count": 4,
                "keyword_count": 2,
            },
        ],
        "edges": [
            {
                "source_uid": "macro:10",
                "target_uid": "micro:100",
                "level": "micro",
                "weight": 3.0,
                "edge_count": 2,
                "relation_label": "parent evidence",
                "shared_terms": ["solar cells"],
            }
        ],
        "warnings": [{"code": "source_warning", "severity": "info", "message": "kept"}],
    }

    payload = build_atlas_render_payload(atlas)

    assert payload["schema_version"] == "sciscape_atlas_render_payload_v1"
    assert payload["source_schema_version"] == "sciscape_atlas_payload_v1"
    assert payload["engine_family"] == "deck.gl"
    assert payload["view"]["type"] == "OrthographicView"
    assert payload["view"]["coordinate_source"] == "mixed"
    assert payload["node_count"] == 2
    assert payload["edge_count"] == 1
    assert payload["label_count"] == 2
    assert payload["hierarchy_edge_count"] == 1

    nodes = {row["cluster_uid"]: row for row in payload["layers"]["nodes"]["rows"]}
    assert nodes["macro:10"]["position"] == [2.0, -1.0]
    assert nodes["macro:10"]["coordinate_source"] == "node_coordinates"
    assert nodes["micro:100"]["coordinate_source"] == "generated_parent_radial"
    assert nodes["micro:100"]["render_radius"] > nodes["macro:10"]["render_radius"] / 2

    edge = payload["layers"]["edges"]["rows"][0]
    assert edge["source_uid"] == "macro:10"
    assert edge["target_uid"] == "micro:100"
    assert edge["source_position"] == [2.0, -1.0]
    assert payload["layers"]["nodes"]["recommended_deck_layer"] == "ScatterplotLayer"
    assert payload["layers"]["labels"]["recommended_deck_layer"] == "TextLayer"
    assert any(w["code"] == "generated_atlas_render_coordinates" for w in payload["warnings"])


def test_atlas_payload_enriches_doc_counts_and_cluster_edges_from_artifacts(tmp_path):
    root = _write_valid_result_root(tmp_path / "result")
    report_data = json.loads((root / "landscape" / "report" / "data.json").read_text(encoding="utf-8"))

    atlas = build_atlas_payload_from_report_data(
        report_data,
        membership_path=root / "landscape" / "membership.parquet",
        edges_path=root / "edges.parquet",
        abstracts_path=root / "abstracts.parquet",
    )
    nodes = {node["cluster_uid"]: node for node in atlas["nodes"]}

    assert atlas["edge_count"] == 1
    assert nodes["cluster:0"]["doc_count"] == 2
    assert nodes["cluster:0"]["doc_count_source"] == "membership:cluster"
    assert nodes["cluster:1"]["doc_count"] == 2
    edge = atlas["edges"][0]
    assert edge["source_uid"] == "cluster:0"
    assert edge["target_uid"] == "cluster:1"
    assert edge["weight"] == 1.0
    assert edge["edge_count"] == 1
    assert nodes["cluster:0"]["neighbor_count"] == 1
    assert nodes["cluster:0"]["neighbors"][0]["cluster_uid"] == "cluster:1"
    assert nodes["cluster:0"]["representative_work_count"] == 2
    assert nodes["cluster:0"]["representative_works"][0]["uid"] == "D1"
    assert nodes["cluster:0"]["representative_works"][0]["title"] == "Perovskite device stability"
    assert nodes["cluster:0"]["representative_works"][0]["cited_by_count"] == 8


def test_atlas_payload_attaches_edge_evidence_samples_from_sidecar(tmp_path):
    root = _write_valid_result_root(tmp_path / "result")
    sidecar = root / "landscape" / "edge_evidence_samples.json"
    sidecar.write_text(
        json.dumps(
            {
                "relations": [
                    {
                        "source_uid": "cluster:0",
                        "target_uid": "cluster:1",
                        "samples": [
                            {
                                "uid1": "D1",
                                "uid2": "D2",
                                "title1": "Perovskite device stability",
                                "title2": "Graph neural traffic forecasting",
                                "layer": "bc",
                                "rel_sum2": 1.0,
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    report_data = json.loads((root / "landscape" / "report" / "data.json").read_text(encoding="utf-8"))

    atlas = build_atlas_payload_from_report_data(
        report_data,
        membership_path=root / "landscape" / "membership.parquet",
        edges_path=root / "edges.parquet",
        abstracts_path=root / "abstracts.parquet",
        edge_evidence_paths=(sidecar,),
    )
    nodes = {node["cluster_uid"]: node for node in atlas["nodes"]}

    assert atlas["edges"][0]["sample_count"] == 1
    assert atlas["edges"][0]["samples"][0]["source_work_uid"] == "D1"
    assert atlas["edges"][0]["samples"][0]["target_work_uid"] == "D2"
    assert atlas["edges"][0]["samples"][0]["edge_type"] == "bc"
    assert nodes["cluster:0"]["neighbors"][0]["sample_count"] == 1
    assert nodes["cluster:0"]["neighbors"][0]["samples"][0]["source_title"] == "Perovskite device stability"

    contract = validate_result_root(root).to_dict()
    assert contract["counts"]["edge_evidence_artifacts"] == 1
    assert contract["artifacts"]["edge_evidence_artifacts"] == ["landscape/edge_evidence_samples.json"]


def test_write_edge_evidence_samples_from_standard_artifacts(tmp_path):
    root = _write_valid_result_root(tmp_path / "result")
    output_path = root / "landscape" / "edge_evidence_samples.json"

    written = write_edge_evidence_samples(
        edges_path=root / "edges.parquet",
        membership_path=root / "landscape" / "membership.parquet",
        abstracts_path=root / "abstracts.parquet",
        output_path=output_path,
        max_relations=2,
        max_samples_per_relation=1,
    )

    assert written == output_path
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    relation = payload["relations"][0]
    assert payload["schema_version"] == "sciscape_edge_evidence_samples_v1"
    assert relation["source_uid"] == "cluster:0"
    assert relation["target_uid"] == "cluster:1"
    assert relation["edge_count"] == 1
    assert relation["samples"][0]["source_work_uid"] == "D1"
    assert relation["samples"][0]["target_work_uid"] == "D2"
    assert relation["samples"][0]["source_title"] == "Perovskite device stability"


def test_write_edge_evidence_samples_matches_single_level_atlas_uids(tmp_path):
    root = _write_valid_result_root(tmp_path / "result")
    membership_path = root / "landscape" / "membership.parquet"
    membership = pd.read_parquet(membership_path).rename(columns={"cluster": "cluster_nano"})
    membership.to_parquet(membership_path, index=False)
    output_path = root / "landscape" / "edge_evidence_samples.json"

    write_edge_evidence_samples(
        edges_path=root / "edges.parquet",
        membership_path=membership_path,
        abstracts_path=root / "abstracts.parquet",
        output_path=output_path,
        max_relations=2,
        max_samples_per_relation=1,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    relation = payload["relations"][0]
    assert relation["source_uid"] == "cluster:0"
    assert relation["target_uid"] == "cluster:1"
    assert relation["level"] == "cluster"


def test_atlas_payload_infers_membership_hierarchy(tmp_path):
    report_data = {
        "clusters": [
            {"cluster_uid": "macro:10", "level": "macro", "cluster_id": 10, "label": "Energy systems"},
            {"cluster_uid": "micro:100", "level": "micro", "cluster_id": 100, "label": "Solar cells"},
            {"cluster_uid": "micro:101", "level": "micro", "cluster_id": 101, "label": "Battery materials"},
        ]
    }
    membership = pd.DataFrame(
        {
            "uid": ["D0", "D1", "D2", "D3", "D4"],
            "cluster_macro": [10, 10, 10, 10, 10],
            "cluster_micro": [100, 100, 101, 101, 101],
        }
    )

    path = tmp_path / "membership.parquet"
    membership.to_parquet(path, index=False)
    try:
        atlas = build_atlas_payload_from_report_data(report_data, membership_path=path)
    finally:
        path.unlink(missing_ok=True)

    nodes = {node["cluster_uid"]: node for node in atlas["nodes"]}
    assert nodes["macro:10"]["doc_count"] == 5
    assert nodes["macro:10"]["child_count"] == 2
    assert nodes["micro:100"]["doc_count"] == 2
    assert nodes["micro:100"]["parent_uid"] == "macro:10"
    assert [row["cluster_uid"] for row in nodes["micro:100"]["lineage"]] == ["macro:10", "micro:100"]


def test_atlas_payload_promotes_legacy_report_clusters_to_leaf_level(tmp_path):
    report_data = {
        "0": {"label": "traffic forecasting", "keywords": [{"term": "traffic forecasting"}]},
        "1": {"label": "drug interaction", "keywords": [{"term": "drug interaction"}]},
    }
    membership = pd.DataFrame(
        {
            "uid": ["D0", "D1", "D2", "D3", "D4"],
            "cluster_micro": [10, 10, 10, 10, 11],
            "cluster_nano": [0, 0, 1, 1, 2],
        }
    )
    path = tmp_path / "membership.parquet"
    membership.to_parquet(path, index=False)

    atlas = build_atlas_payload_from_report_data(report_data, membership_path=path)
    nodes = {node["cluster_uid"]: node for node in atlas["nodes"]}

    assert atlas["levels"] == ["micro", "nano"]
    assert nodes["nano:0"]["level"] == "nano"
    assert nodes["nano:0"]["doc_count"] == 2
    assert nodes["nano:0"]["parent_uid"] == "micro:10"
    assert nodes["nano:1"]["doc_count"] == 2
    assert nodes["micro:10"]["node_source"] == "membership_parent"
    assert nodes["micro:10"]["child_count"] == 2
    assert [row["cluster_uid"] for row in nodes["nano:0"]["lineage"]] == ["micro:10", "nano:0"]


def test_validate_result_root_blocks_top_metadata_artifacts(tmp_path):
    root = _write_valid_result_root(tmp_path / "result")
    keyword_path = root / "landscape" / "keywords.parquet"
    keywords = pd.read_parquet(keyword_path)
    contaminated = pd.concat(
        [
            pd.DataFrame(
                {
                    "cluster_id": [0],
                    "term": ["class htmlview paragraph"],
                    "score": [999.0],
                    "frequency": [10],
                    "quality_flags": ["metadata_fragment"],
                    "representative_rank": [1],
                }
            ),
            keywords.assign(representative_rank=range(2, len(keywords) + 2)),
        ],
        ignore_index=True,
    )
    contaminated.to_parquet(keyword_path, index=False)

    data_path = root / "landscape" / "report" / "data.json"
    data = json.loads(data_path.read_text(encoding="utf-8"))
    data["0"]["keywords"].insert(
        0,
        {
            "term": "usepackage",
            "display_label": "usepackage",
            "quality_flags": "metadata_fragment",
        },
    )
    data_path.write_text(json.dumps(data), encoding="utf-8")

    result = validate_result_root(root)
    payload = result.to_dict()

    assert payload["ok"] is False
    assert payload["result_state"] == "blocked"
    assert payload["counts"]["keyword_top_artifact_rows"] == 1
    assert payload["counts"]["report_keyword_top_artifact_rows"] == 1
    codes = {warning["code"] for warning in payload["warnings"]}
    assert "top_keyword_artifact" in codes
    assert "top_report_keyword_artifact" in codes


def test_write_artifact_contract_uses_landscape_qa_dir(tmp_path):
    root = _write_valid_result_root(tmp_path / "result")

    result = write_artifact_contract(root)
    contract = root / "landscape" / "qa" / "artifact_contract.json"

    assert result.ok is True
    assert contract.exists()
    payload = json.loads(contract.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "sciscape_artifact_contract_v1"
    assert payload["features"]["term_network"] is True


def test_write_export_manifest_promotes_stable_export_feature(tmp_path):
    root = _write_valid_result_root(tmp_path / "result")
    write_artifact_contract(root)
    report_data = root / "landscape" / "report" / "data.json"

    written = write_export_manifest(
        root,
        export_id="json_report_data",
        export_family="report",
        export_kind="json_report_data",
        primary_file=report_data,
        source_artifacts=[
            {
                "role": "report_data",
                "artifact_ref": "report_data",
                "path": "landscape/report/data.json",
                "feature_ref": "overview",
            }
        ],
        feature_refs=["overview", "cluster_map", "export"],
        compatibility={"target_tools": ["SciScape"], "limitations": []},
    )

    assert written["manifest_path"] == root / "exports" / "json_report_data" / "export_manifest.json"
    assert written["qa"]["schema_version"] == EXPORT_QA_SCHEMA_VERSION
    assert written["qa"]["status"] == "passed"

    files = pd.read_parquet(written["files_path"])
    assert set(files["schema_version"]) == {EXPORT_FILES_SCHEMA_VERSION}
    assert files["path"].iloc[0] == "landscape/report/data.json"

    validation = validate_export_manifest(written["manifest_path"]).to_dict()
    assert validation["status"] == "passed"
    assert validation["counts"]["files"] == 1
    assert validation["counts"]["inputs"] == 1
    export_manifest = json.loads(written["manifest_path"].read_text(encoding="utf-8"))
    assert export_manifest["selection"]["schema_version"] == "sciscape_export_selection_v1"
    assert export_manifest["selection"]["scope"] == "full_result"
    assert export_manifest["selection"]["view"] == {
        "mode": "json_report_data",
        "family": "report",
    }
    assert export_manifest["selection"]["filters"] == []

    contract = validate_result_root(root).to_dict()
    assert contract["counts"]["stable_export_artifacts"] == 1
    assert contract["counts"]["export_file_rows"] == 1
    assert contract["features"]["export"] is True

    manifest = build_result_manifest(root).to_dict()
    assert manifest["features"]["export"]["state"] == "stable"
    assert "export" in manifest["features"]["export"]["artifact_refs"]
    assert manifest["artifacts"]["export"]["schema_version"] == EXPORT_MANIFEST_SCHEMA_VERSION
    report_exports = [export for export in manifest["exports"] if export["export_id"] == "json_report_data"]
    assert len(report_exports) == 1
    assert report_exports[0]["status"] == "passed"
    assert report_exports[0]["selection"]["view"]["mode"] == "json_report_data"
    assert report_exports[0]["selection_summary"] == {
        "scope": "full_result",
        "view_mode": "json_report_data",
        "view_family": "report",
        "cluster_level": None,
        "filter_count": 0,
        "threshold_keys": [],
        "layer_state_keys": [],
        "focus_keys": [],
        "subset_mode": None,
        "subset_count": None,
        "subset_keys": [],
    }


def test_validate_export_manifest_blocks_missing_export_file(tmp_path):
    root = _write_valid_result_root(tmp_path / "result")
    export_dir = root / "exports" / "bad_export"
    export_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "schema_version": [EXPORT_FILES_SCHEMA_VERSION],
            "export_id": ["bad_export"],
            "file_id": ["primary"],
            "path": ["missing/report.html"],
            "role": ["primary"],
            "format": ["html"],
            "public_share_state": ["local"],
        }
    ).to_parquet(export_dir / "export_files.parquet", index=False)
    pd.DataFrame(
        {
            "schema_version": [EXPORT_INPUTS_SCHEMA_VERSION],
            "export_id": ["bad_export"],
            "input_id": ["input_1"],
            "artifact_ref": ["report_data"],
            "artifact_role": ["report_data"],
            "artifact_path": ["landscape/report/data.json"],
            "feature_state": ["stable"],
            "required": [True],
        }
    ).to_parquet(export_dir / "export_inputs.parquet", index=False)
    pd.DataFrame(
        {
            "schema_version": [EXPORT_TRANSFORMS_SCHEMA_VERSION],
            "export_id": ["bad_export"],
            "transform_id": ["transform_1"],
            "step_index": [0],
            "transform_type": ["wrap_existing_export"],
            "description": ["wrap"],
            "parameters": ["{}"],
        }
    ).to_parquet(export_dir / "export_transforms.parquet", index=False)
    (export_dir / "export_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": EXPORT_MANIFEST_SCHEMA_VERSION,
                "export_id": "bad_export",
                "title": "Bad export",
                "result_id": None,
                "export_family": "report",
                "export_kind": "html_report",
                "format": "html",
                "status": "passed",
                "feature_refs": ["export"],
                "source_artifacts": [{"role": "report_data", "path": "landscape/report/data.json"}],
                "selection": {"scope": "full_result"},
                "transform_summary": {"transform_count": 1},
                "compatibility": {"target_tools": ["SciScape"], "limitations": []},
                "outputs": {
                    "files": "export_files.parquet",
                    "inputs": "export_inputs.parquet",
                    "transforms": "export_transforms.parquet",
                    "qa": "export_qa.json",
                },
                "created_at_utc": "2026-06-05T00:00:00+00:00",
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )

    validation = validate_export_manifest(export_dir).to_dict()

    assert validation["status"] == "blocked"
    assert any(issue["code"] == "missing_export_files" for issue in validation["blocking_issues"])


def test_build_result_manifest_wraps_artifact_contract(tmp_path):
    root = _write_valid_result_root(tmp_path / "result")
    write_artifact_contract(root)

    manifest = build_result_manifest(root).to_dict()

    assert manifest["schema_version"] == RESULT_MANIFEST_SCHEMA_VERSION
    assert manifest["result_kind"] == "imported_result"
    assert manifest["result_root"] == "."
    assert manifest["quality"]["validation_state"] == "passed"
    assert manifest["artifacts"]["records"]["path"] == "abstracts.parquet"
    assert manifest["artifacts"]["membership"]["path"] == "landscape/membership.parquet"
    assert manifest["artifacts"]["keywords"]["rows"] == 4
    assert manifest["artifacts"]["artifact_contract"]["status"] == "present"
    assert manifest["features"]["cluster_map"]["state"] == "stable"
    assert manifest["features"]["keyword"]["state"] == "stable"
    assert manifest["features"]["cooccurrence"]["state"] == "beta"
    assert "keywords" in manifest["features"]["cooccurrence"]["artifact_refs"]
    assert manifest["features"]["export"]["state"] == "beta"
    assert manifest["run_state"]["status"] == "complete"
    assert any(export["kind"] == "json_report_data" for export in manifest["exports"])


def test_result_manifest_run_state_accepts_override_and_status_sidecar(tmp_path):
    root = _write_valid_result_root(tmp_path / "result")
    status_payload = {
        "schema_version": "sciscape_live_job_status_v1",
        "job_id": "job-1",
        "status": "running",
        "updated_at_utc": "2026-06-02T00:00:00+00:00",
        "progress": ["fetching records", "building graph"],
        "run_state": {
            "status": "running",
            "heartbeat_at_utc": "2026-06-02T00:00:00+00:00",
            "progress": {"current": 2, "total": None, "unit": "messages"},
            "checkpoints": [{"path": "job_status.json", "kind": "job_status", "status": "present"}],
            "partial_outputs": [{"path": "abstracts.parquet", "kind": "abstracts", "status": "present"}],
            "resume": {"supported": False, "command": None},
        },
    }
    (root / "job_status.json").write_text(json.dumps(status_payload), encoding="utf-8")

    manifest = build_result_manifest(root).to_dict()

    assert manifest["artifacts"]["job_status"]["status"] == "present"
    assert manifest["run_state"]["status"] == "running"
    assert manifest["run_state"]["progress"]["current"] == 2
    assert manifest["run_state"]["failure"] is None
    assert manifest["run_state"]["checkpoints"] == [
        {"path": "job_status.json", "kind": "job_status", "status": "present"}
    ]

    written = write_result_manifest(
        root,
        run_state_overrides={
            "status": "complete",
            "finished_at_utc": "2026-06-02T00:05:00+00:00",
            "progress": {"current": 100, "total": 100, "unit": "percent"},
        },
    ).to_dict()

    assert written["run_state"]["status"] == "complete"
    assert written["run_state"]["finished_at_utc"] == "2026-06-02T00:05:00+00:00"
    assert written["run_state"]["progress"]["unit"] == "percent"


def test_result_manifest_detects_keyword_progress_and_scoring_shards(tmp_path):
    root = _write_valid_result_root(tmp_path / "result")
    landscape = root / "landscape"
    (landscape / "progress.json").write_text(
        json.dumps(
            {
                "updated_at_utc": "2026-06-02T01:00:00+00:00",
                "stage": "scoring_topk",
                "processed": 64,
                "total": 256,
                "percent": 25.0,
            }
        ),
        encoding="utf-8",
    )
    shard_dir = landscape / "scoring_shards"
    shard_dir.mkdir()
    (shard_dir / "scoring_topk_shard_0000.parquet").write_bytes(b"placeholder")
    (shard_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "sciscape_scoring_shard_manifest_v1",
                "status": "running",
                "updated_at_utc": "2026-06-02T01:01:00+00:00",
                "shard_count": 4,
                "resume": True,
                "completed_shards": [
                    {
                        "shard_index": 0,
                        "row_start": 0,
                        "row_end": 64,
                        "path": "scoring_topk_shard_0000.parquet",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    manifest = build_result_manifest(root).to_dict()

    assert manifest["artifacts"]["pipeline_progress"]["path"] == "landscape/progress.json"
    assert manifest["artifacts"]["scoring_shard_manifest"]["path"] == "landscape/scoring_shards/manifest.json"
    assert manifest["run_state"]["status"] == "running"
    assert manifest["run_state"]["progress"]["stage"] == "scoring_topk"
    assert manifest["run_state"]["shards"] == {"total": 4, "complete": 1, "failed": 0, "running": 3}
    assert manifest["run_state"]["resume"]["supported"] is True
    assert manifest["run_state"]["partial_outputs"][0]["kind"] == "scoring_shard"


def test_result_manifest_detects_cluster_sharded_keyword_run_state(tmp_path):
    root = _write_valid_result_root(tmp_path / "result")
    output_dir = root / "landscape" / "keyword_cluster_sharded" / "full_run"
    candidate_dir = output_dir / "candidates"
    candidate_dir.mkdir(parents=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "sciscape_keyword_cluster_shards_v1",
                "created_at_utc": "2026-06-02T01:00:00+00:00",
                "total_clusters": 30,
                "total_docs": 3000,
                "shards": [
                    {"shard_id": 0, "cluster_ids": [0, 1], "doc_count": 1000},
                    {"shard_id": 1, "cluster_ids": [2, 3], "doc_count": 1000},
                    {"shard_id": 2, "cluster_ids": [4, 5], "doc_count": 1000},
                ],
            }
        ),
        encoding="utf-8",
    )
    (output_dir / "progress.json").write_text(
        json.dumps(
            {
                "updated_at_utc": "2026-06-02T01:05:00+00:00",
                "stage": "candidate_mining",
                "processed": 1,
                "total": 3,
                "percent": 33.3,
            }
        ),
        encoding="utf-8",
    )
    (output_dir / "preflight_summary.json").write_text(
        json.dumps(
            {
                "schema_version": "sciscape_keyword_cluster_sharded_preflight_v1",
                "status": "ok",
                "shard_count": 3,
                "abstract_path": str(root / "abstracts.parquet"),
                "membership_path": str(root / "landscape" / "membership.parquet"),
                "cluster_level": "cluster",
            }
        ),
        encoding="utf-8",
    )
    candidate_path = candidate_dir / "candidate_shard_0000.parquet"
    candidate_path.write_bytes(b"placeholder")
    (candidate_dir / "candidate_shard_0000.done.json").write_text(
        json.dumps(
            {
                "schema_version": "sciscape_keyword_candidate_shard_done_v1",
                "status": "complete",
                "shard_id": 0,
                "rows": 17,
                "source_rows": 1000,
                "elapsed_sec": 12.5,
                "peak_rss_mb": 256.0,
                "path": str(candidate_path),
            }
        ),
        encoding="utf-8",
    )
    (candidate_dir / "candidate_shard_0001.progress.json").write_text(
        json.dumps(
            {
                "schema_version": "sciscape_keyword_candidate_shard_progress_v1",
                "status": "running",
                "shard_id": 1,
                "rows_processed": 200,
                "rows_total": 1000,
                "output_path": str(candidate_dir / "candidate_shard_0001.parquet"),
            }
        ),
        encoding="utf-8",
    )
    (candidate_dir / "candidate_shard_0002.progress.json").write_text(
        json.dumps(
            {
                "schema_version": "sciscape_keyword_candidate_shard_progress_v1",
                "status": "failed",
                "shard_id": 2,
                "rows_processed": 75,
                "rows_total": 1000,
                "error": "RuntimeError('worker lost')",
                "output_path": str(candidate_dir / "candidate_shard_0002.parquet"),
            }
        ),
        encoding="utf-8",
    )

    manifest = build_result_manifest(root).to_dict()

    assert manifest["artifacts"]["keyword_cluster_shard_manifest"]["path"] == (
        "landscape/keyword_cluster_sharded/full_run/manifest.json"
    )
    assert manifest["artifacts"]["keyword_cluster_sharded_progress"]["path"] == (
        "landscape/keyword_cluster_sharded/full_run/progress.json"
    )
    assert manifest["artifacts"]["keyword_cluster_sharded_preflight"]["path"] == (
        "landscape/keyword_cluster_sharded/full_run/preflight_summary.json"
    )
    assert manifest["run_state"]["status"] == "failed"
    assert manifest["run_state"]["progress"]["stage"] == "candidate_mining"
    assert manifest["run_state"]["shards"] == {"total": 3, "complete": 1, "failed": 1, "running": 1}
    assert manifest["run_state"]["failure"]["failed_shards"] == [2]
    assert manifest["run_state"]["resume"]["supported"] is True
    assert manifest["run_state"]["resume"]["artifact_dir"] == "landscape/keyword_cluster_sharded/full_run"
    assert "--keyword-engine cluster_sharded" in manifest["run_state"]["resume"]["command"]
    assert "--cluster-level cluster" in manifest["run_state"]["resume"]["command"]
    assert f"-o {output_dir / 'keywords.parquet'}" in manifest["run_state"]["resume"]["command"]
    assert "--scoring-shard-resume" in manifest["run_state"]["resume"]["command"]
    assert manifest["run_state"]["partial_outputs"][0]["path"] == (
        "landscape/keyword_cluster_sharded/full_run/candidates/candidate_shard_0000.parquet"
    )
    assert manifest["run_state"]["partial_outputs"][0]["kind"] == "candidate_shard"


def test_write_result_manifest_uses_result_root(tmp_path):
    root = _write_valid_result_root(tmp_path / "result")

    manifest = write_result_manifest(root)
    manifest_path = root / "result_manifest.json"

    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == RESULT_MANIFEST_SCHEMA_VERSION
    assert payload["result_id"] == manifest.result_id
    assert payload["features"]["quality"]["state"] == "stable"


def test_load_result_manifest_preserves_metadata_but_recomputes_features(tmp_path):
    root = _write_valid_result_root(tmp_path / "result")
    (root / "result_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": RESULT_MANIFEST_SCHEMA_VERSION,
                "result_id": "curated-result",
                "title": "Curated Result",
                "features": {
                    "narrative": {
                        "state": "stable",
                        "reason": "stale advertised state",
                        "artifact_refs": [],
                        "warnings": [],
                    }
                },
                "run_state": {
                    "status": "stopped_by_qc",
                    "finished_at_utc": "2026-06-02T02:00:00+00:00",
                    "progress": {"current": 23, "total": 40, "unit": "clusters"},
                    "failure": {"reason": "QC stopped the run"},
                },
                "provenance": {"commands": ["sciscape query --query curated"]},
            }
        ),
        encoding="utf-8",
    )

    manifest = load_result_manifest(root)

    assert manifest["manifest_state"] == "present"
    assert manifest["manifest_path"] == "result_manifest.json"
    assert manifest["result_id"] == "curated-result"
    assert manifest["title"] == "Curated Result"
    assert manifest["features"]["narrative"]["state"] == "hidden"
    assert manifest["features"]["cluster_map"]["state"] == "stable"
    assert manifest["run_state"]["status"] == "stopped_by_qc"
    assert manifest["run_state"]["progress"] == {"current": 23, "total": 40, "unit": "clusters"}
    assert manifest["run_state"]["failure"] == {"reason": "QC stopped the run"}
    assert manifest["provenance"]["commands"] == ["sciscape query --query curated"]


def test_load_result_manifest_accepts_legacy_manifest_alias(tmp_path):
    root = _write_valid_result_root(tmp_path / "result")
    (root / "MANIFEST.json").write_text(
        json.dumps(
            {
                "schema_version": RESULT_MANIFEST_SCHEMA_VERSION,
                "result_id": "legacy-result",
                "title": "Legacy Result",
            }
        ),
        encoding="utf-8",
    )

    manifest = load_result_manifest(root)

    assert manifest["manifest_state"] == "legacy"
    assert manifest["manifest_path"] == "MANIFEST.json"
    assert manifest["result_id"] == "legacy-result"
    assert manifest["title"] == "Legacy Result"
    assert manifest["features"]["keyword"]["state"] == "stable"


def test_write_workspace_manifest_creates_registry_and_qa(tmp_path):
    workspace = tmp_path / "workspace"

    written = write_workspace_manifest(
        workspace,
        workspace_id="workspace_test",
        name="Workspace Test",
    )

    manifest_path = workspace / "workspace.json"
    qa_path = workspace / "workspace_qa.json"
    assert written["manifest_path"] == manifest_path
    assert written["qa_path"] == qa_path
    assert manifest_path.exists()
    assert qa_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == WORKSPACE_MANIFEST_SCHEMA_VERSION
    assert manifest["workspace_id"] == "workspace_test"
    assert manifest["objects"]["results"] == []
    assert qa["schema_version"] == WORKSPACE_QA_SCHEMA_VERSION
    assert qa["status"] == "passed"
    assert qa["state"] == "stable"
    assert qa["counts"]["results"] == 0

    validation = validate_workspace(workspace).to_dict()
    assert validation["ok"] is True
    assert validation["state"] == "stable"


def test_register_result_in_workspace_adds_relative_result_ref(tmp_path):
    workspace = tmp_path / "workspace"
    result_root = _write_valid_result_root(workspace / "results" / "perovskite")
    write_workspace_manifest(
        workspace,
        workspace_id="workspace_test",
        name="Workspace Test",
    )

    written = register_result_in_workspace(workspace, result_root, project_id="project_pending")

    result_ref = written["registered_result"]
    assert result_ref["path"] == "results/perovskite/result_manifest.json"
    assert result_ref["state"] == "validated"
    assert result_ref["project_id"] == "project_pending"
    assert (result_root / "result_manifest.json").exists()

    manifest = json.loads((workspace / "workspace.json").read_text(encoding="utf-8"))
    assert manifest["objects"]["results"][0]["path"] == "results/perovskite/result_manifest.json"
    assert manifest["recent"]["results"] == [result_ref["result_id"]]
    assert manifest["defaults"]["result_id"] == result_ref["result_id"]
    assert "project_id" not in manifest["defaults"]

    qa = json.loads((workspace / "workspace_qa.json").read_text(encoding="utf-8"))
    assert qa["status"] == "passed"
    assert qa["objects"]["results"][0]["validation_state"] == "passed"


def test_validate_workspace_blocks_absolute_object_paths(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "workspace.json").write_text(
        json.dumps(
            {
                "schema_version": WORKSPACE_MANIFEST_SCHEMA_VERSION,
                "workspace_id": "workspace_test",
                "name": "Workspace Test",
                "root": ".",
                "created_at_utc": "2026-06-03T00:00:00+00:00",
                "updated_at_utc": "2026-06-03T00:00:00+00:00",
                "objects": {
                    "projects": [],
                    "datasets": [],
                    "runs": [],
                    "results": [
                        {
                            "result_id": "absolute",
                            "path": str(tmp_path / "outside" / "result_manifest.json"),
                            "state": "validated",
                        }
                    ],
                    "rule_sets": [],
                    "views": [],
                    "exports": [],
                },
                "recent": {"results": ["absolute"]},
                "defaults": {"result_id": "absolute"},
                "settings": {},
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )

    validation = validate_workspace(workspace).to_dict()

    assert validation["ok"] is False
    assert validation["state"] == "blocked"
    assert any(issue["code"] == "absolute_workspace_path" for issue in validation["blocking_issues"])


def test_quality_gate_validates_artifact_root_and_writes_contract(tmp_path):
    root = _write_valid_result_root(tmp_path / "result")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/sciscape_quality_gate.py",
            "--artifact-root",
            str(root),
            "--write-artifact-contract",
            "--write-result-manifest",
            "--json",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(completed.stdout)

    gate = payload["gates"]["artifact_contract"]
    assert payload["status"] == "passed"
    assert gate["ok"] is True
    assert gate["features"]["term_network"] is True
    assert Path(gate["artifact_contract_path"]).exists()
    manifest_gate = payload["gates"]["result_manifest"]
    assert manifest_gate["schema_version"] == RESULT_MANIFEST_SCHEMA_VERSION
    assert Path(manifest_gate["result_manifest_path"]).exists()
    assert manifest_gate["features"]["keyword"]["state"] == "stable"


def test_dashboard_export_embeds_report_data_contract(tmp_path):
    keywords = pd.DataFrame(
        {
            "cluster_id": [0, 0],
            "term": ["perovskite solar cells", "interface passivation"],
            "score": [0.9, 0.8],
            "frequency": [2, 1],
        }
    )
    dashboard = tmp_path / "dashboard.html"

    export_dashboard(keywords, output_path=str(dashboard))
    html = dashboard.read_text(encoding="utf-8")

    assert "sciscape_report_data_contract_v1" in html
    assert '"sciscape_version"' in html
    assert "sciscape_atlas_payload_v1" in html
    assert "SCISCAPE_CONTRACT" in html
    assert "Result contract" in html
    assert "TAB_FEATURES" in html


def test_dashboard_export_writes_export_manifest(tmp_path):
    keywords = pd.DataFrame(
        {
            "cluster_id": [0, 0],
            "term": ["perovskite solar cells", "interface passivation"],
            "score": [0.9, 0.8],
            "frequency": [2, 1],
        }
    )
    dashboard = tmp_path / "dashboard.html"

    export_dashboard(keywords, output_path=str(dashboard))

    manifest_path = tmp_path / "exports" / "keyword_dashboard" / "export_manifest.json"
    assert manifest_path.exists()
    validation = validate_export_manifest(manifest_path).to_dict()
    assert validation["status"] == "passed"
    assert validation["export_kind"] == "keyword_dashboard_html"
    export_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert export_manifest["selection"]["scope"] == "keyword_table"
    assert export_manifest["selection"]["view"] == {
        "mode": "keyword_dashboard",
        "surface": "dashboard_export",
        "family": "viewer",
    }
    assert export_manifest["selection"]["layer_state"]["keyword_count"] == 2
    assert export_manifest["selection"]["layer_state"]["cluster_count"] == 1
    result_manifest = build_result_manifest(tmp_path).to_dict()
    dashboard_exports = [row for row in result_manifest["exports"] if row["export_id"] == "keyword_dashboard"]
    assert len(dashboard_exports) == 1
    assert dashboard_exports[0]["selection_summary"]["view_mode"] == "keyword_dashboard"
    assert dashboard_exports[0]["selection_summary"]["view_family"] == "viewer"


def test_viewer_export_writes_export_manifest(tmp_path):
    viewer = tmp_path / "viewer.html"

    export_viewer(output_path=str(viewer), title="SciScape Static Viewer")

    manifest_path = tmp_path / "exports" / "static_viewer" / "export_manifest.json"
    assert manifest_path.exists()
    validation = validate_export_manifest(manifest_path).to_dict()
    assert validation["status"] == "passed"
    assert validation["export_kind"] == "static_viewer_html"
    export_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert export_manifest["selection"]["scope"] == "hosted_or_uploaded_data"
    assert export_manifest["selection"]["view"]["mode"] == "static_viewer"
    assert export_manifest["selection"]["layer_state"] == {
        "data_mode": "hosted_data_json_or_upload",
        "default_data_url": "data.json",
        "supports_query_data_url": True,
    }


def test_report_export_writes_atlas_payload_to_data_json(tmp_path):
    keywords = pd.DataFrame(
        {
            "cluster_id": [0, 0, 1],
            "term": ["perovskite solar cells", "interface passivation", "graph neural networks"],
            "score": [0.9, 0.8, 0.95],
            "frequency": [2, 1, 2],
        }
    )

    export_report(keywords, output_dir=str(tmp_path / "report"))
    payload = json.loads((tmp_path / "report" / "data.json").read_text(encoding="utf-8"))
    contract = payload["_sciscape"]

    assert contract["atlas"]["schema_version"] == "sciscape_atlas_payload_v1"
    assert contract["atlas"]["node_count"] == 2
    assert [node["cluster_uid"] for node in contract["atlas"]["nodes"]] == ["cluster:0", "cluster:1"]
    assert contract["atlas"]["nodes"][0]["keyword_count"] == 2


def test_report_export_writes_bundle_export_manifest(tmp_path):
    keywords = pd.DataFrame(
        {
            "cluster_id": [0, 0, 1],
            "term": ["perovskite solar cells", "interface passivation", "graph neural networks"],
            "score": [0.9, 0.8, 0.95],
            "frequency": [2, 1, 2],
        }
    )

    export_report(keywords, output_dir=str(tmp_path / "report"))

    manifest_path = tmp_path / "exports" / "html_report" / "export_manifest.json"
    assert manifest_path.exists()
    validation = validate_export_manifest(manifest_path).to_dict()
    assert validation["status"] == "passed"
    assert validation["export_kind"] == "html_report"
    assert validation["counts"]["files"] >= 5
    export_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert export_manifest["selection"]["scope"] == "keyword_table"
    assert export_manifest["selection"]["view"] == {
        "mode": "html_report",
        "surface": "report_export",
        "family": "report",
    }
    assert export_manifest["selection"]["layer_state"]["keyword_count"] == 3
    assert export_manifest["selection"]["layer_state"]["cluster_count"] == 2
    assert export_manifest["selection"]["layer_state"]["generated_file_count"] >= 5
    result_manifest = build_result_manifest(tmp_path).to_dict()
    report_exports = [row for row in result_manifest["exports"] if row["export_id"] == "html_report"]
    assert len(report_exports) == 1
    assert report_exports[0]["selection_summary"]["view_mode"] == "html_report"
    assert report_exports[0]["selection_summary"]["layer_state_keys"] == [
        "cluster_count",
        "data_mode",
        "generated_file_count",
        "keyword_count",
        "open_browser",
    ]


def test_graphml_export_can_write_export_manifest(tmp_path):
    import polars as pl

    result_root = tmp_path / "result"
    result_root.mkdir()
    edges_path = result_root / "edges.parquet"
    membership_path = result_root / "membership.parquet"
    abstracts_path = result_root / "abstracts.parquet"
    pl.DataFrame({"uid1": ["D0"], "uid2": ["D1"], "rel_sum2": [1.0]}).write_parquet(edges_path)
    pl.DataFrame({"uid": ["D0", "D1"], "cluster": [0, 0]}).write_parquet(membership_path)
    pl.DataFrame(
        {
            "uid": ["D0", "D1"],
            "title": ["Paper A", "Paper B"],
            "abstract": ["A", "B"],
            "pubyear": [2021, 2022],
        }
    ).write_parquet(abstracts_path)

    export_graphml(
        pl.read_parquet(edges_path),
        pl.read_parquet(membership_path),
        result_root / "network.graphml",
        abstracts=pl.read_parquet(abstracts_path),
        write_manifest=True,
        result_root=result_root,
        source_paths={
            "edges": edges_path,
            "membership": membership_path,
            "abstracts": abstracts_path,
        },
    )

    manifest_path = result_root / "exports" / "network_graphml" / "export_manifest.json"
    assert manifest_path.exists()
    validation = validate_export_manifest(manifest_path).to_dict()
    assert validation["status"] == "passed"
    assert validation["export_kind"] == "graphml_graph"
    assert validation["counts"]["inputs"] == 3
    export_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert export_manifest["selection"]["view"] == {
        "mode": "cluster_network_export",
        "surface": "graph_export",
        "family": "graph",
    }
    assert export_manifest["selection"]["layer_state"] == {
        "network_format": "graphml",
        "edge_table": "edges",
    }


def test_vosviewer_export_writes_map_network_and_manifest(tmp_path):
    import polars as pl

    result_root = _write_valid_result_root(tmp_path / "result")
    edges_path = result_root / "edges.parquet"
    membership_path = result_root / "landscape" / "membership.parquet"
    abstracts_path = result_root / "abstracts.parquet"
    pl.DataFrame(
        {
            "uid1": ["D0", "D1", "D0"],
            "uid2": ["D1", "D2", "D1"],
            "rel_sum2": [1.0, 2.0, 3.0],
        }
    ).write_parquet(edges_path)
    pl.DataFrame({"uid": ["D0", "D1", "D2"], "cluster": [0, 0, 1]}).write_parquet(membership_path)

    written = export_vosviewer_network(
        pl.read_parquet(edges_path),
        pl.read_parquet(membership_path),
        result_root / "vosviewer",
        abstracts=pl.read_parquet(abstracts_path),
        result_root=result_root,
        source_paths={
            "edges": edges_path,
            "membership": membership_path,
            "abstracts": abstracts_path,
        },
    )

    map_lines = written["map_path"].read_text(encoding="utf-8").splitlines()
    network_lines = written["network_path"].read_text(encoding="utf-8").splitlines()
    assert map_lines[0].split("\t")[:4] == ["id", "label", "description", "cluster"]
    assert len(map_lines) == 4
    map_rows = [line.split("\t") for line in map_lines[1:]]
    assert {row[1]: row[4:6] for row in map_rows} == {
        "D0": ["1", "4.000000"],
        "D1": ["2", "6.000000"],
        "D2": ["1", "2.000000"],
    }
    assert network_lines == ["1\t2\t4.000000", "2\t3\t2.000000"]

    manifest_path = result_root / "exports" / "vosviewer_map_network" / "export_manifest.json"
    assert written["manifest_path"] == manifest_path
    validation = validate_export_manifest(manifest_path).to_dict()
    assert validation["status"] == "passed"
    assert validation["export_family"] == "vosviewer"
    assert validation["export_kind"] == "vosviewer_map_network"
    assert validation["counts"]["files"] == 2
    export_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert export_manifest["selection"]["scope"] == "full_result"
    assert export_manifest["selection"]["view"]["mode"] == "vosviewer_map_network"
    assert export_manifest["selection"]["cluster_level"] == "cluster"
    assert export_manifest["selection"]["thresholds"] == {"min_link_strength": 0}
    assert export_manifest["selection"]["layer_state"] == {
        "map_file": "vosviewer_map.txt",
        "network_file": "vosviewer_network.txt",
        "edge_weight_column": "rel_sum2",
    }

    result_manifest = write_result_manifest(result_root).to_dict()
    vos_exports = [export for export in result_manifest["exports"] if export["export_id"] == "vosviewer_map_network"]
    assert len(vos_exports) == 1
    vos_export = vos_exports[0]
    assert vos_export["path"] == "vosviewer/vosviewer_map.txt"
    assert vos_export["export_manifest_ref"] == "exports/vosviewer_map_network/export_manifest.json"
    assert vos_export["export_family"] == "vosviewer"
    assert {row["role"]: row["path"] for row in vos_export["files"]} == {
        "map": "vosviewer/vosviewer_map.txt",
        "network": "vosviewer/vosviewer_network.txt",
    }
    assert vos_export["selection_summary"]["view_mode"] == "vosviewer_map_network"
    assert vos_export["selection_summary"]["cluster_level"] == "cluster"
    assert vos_export["selection_summary"]["threshold_keys"] == ["min_link_strength"]


def test_vosviewer_thesaurus_export_writes_rule_set_and_manifest(tmp_path):
    result_root = _write_valid_result_root(tmp_path / "result")
    keywords = pd.DataFrame(
        {
            "cluster_id": [0, 0, 1, 1],
            "term": [
                "class htmlview paragraph",
                "graph neural networks",
                "gnn",
                "color behavior",
            ],
            "raw_term": [
                "class htmlview paragraph",
                "graph neural networks",
                "gnn",
                "colour behaviour",
            ],
            "quality_flags": ["metadata_fragment", "", "", ""],
            "keyword_label_tier": ["review_artifact", "primary_phrase", "supporting", "primary_phrase"],
            "norm_merged_from": ["", '["graph neural network"]', "", ""],
            "abbreviation_target": ["", "", "graph neural networks", ""],
            "abbreviation_status": ["", "", "confirmed", ""],
            "score": [0.9, 1.2, 0.7, 0.8],
            "frequency": [10, 12, 8, 7],
            "representative_rank": [1, 1, 2, 3],
        }
    )
    rule_artifact = write_keyword_cleaning_rule_artifacts(result_root, keywords=keywords)

    written = export_vosviewer_thesaurus(
        rule_artifact["manifest_path"],
        result_root / "vosviewer",
        result_root=result_root,
    )

    thesaurus_lines = written["thesaurus_path"].read_text(encoding="utf-8").splitlines()
    assert thesaurus_lines[0] == "label\treplace by"
    assert set(thesaurus_lines[1:]) == {
        "class htmlview paragraph\t",
        "colour behaviour\tcolor behavior",
        "gnn\tgraph neural networks",
        "graph neural network\tgraph neural networks",
    }
    rule_set_lines = written["rule_set_path"].read_text(encoding="utf-8").splitlines()
    assert rule_set_lines[0].split("\t")[:3] == ["rule_id", "rule_family", "match_type"]
    assert any("html_fragment_block" in line for line in rule_set_lines)
    assert any("abbreviation_evidence_expand" in line for line in rule_set_lines)

    manifest_path = result_root / "exports" / "vosviewer_thesaurus" / "export_manifest.json"
    assert written["manifest_path"] == manifest_path
    validation = validate_export_manifest(manifest_path).to_dict()
    assert validation["status"] == "passed"
    assert validation["export_family"] == "vosviewer"
    assert validation["export_kind"] == "vosviewer_thesaurus"
    assert validation["counts"]["files"] == 2
    export_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert export_manifest["selection"]["scope"] == "keyword_rule_artifact"
    assert export_manifest["selection"]["view"]["mode"] == "cleaning_rules"
    assert export_manifest["selection"]["filters"] == [
        {"field": "action", "op": "exclude", "value": "keep_with_flag"}
    ]
    assert export_manifest["selection"]["layer_state"]["thesaurus_columns"] == ["label", "replace by"]

    result_manifest = write_result_manifest(result_root).to_dict()
    vos_exports = [export for export in result_manifest["exports"] if export["export_id"] == "vosviewer_thesaurus"]
    assert len(vos_exports) == 1
    assert vos_exports[0]["path"] == "vosviewer/vosviewer_thesaurus.txt"
    assert {row["role"]: row["path"] for row in vos_exports[0]["files"]} == {
        "thesaurus": "vosviewer/vosviewer_thesaurus.txt",
        "rule_set": "vosviewer/sciscape_keyword_rules.tsv",
    }
    assert vos_exports[0]["selection_summary"]["scope"] == "keyword_rule_artifact"
    assert vos_exports[0]["selection_summary"]["view_mode"] == "cleaning_rules"


def test_vosviewer_bundle_export_uses_manifest_backed_vosviewer_files(tmp_path):
    import polars as pl

    result_root = _write_valid_result_root(tmp_path / "result")
    edges_path = result_root / "edges.parquet"
    membership_path = result_root / "landscape" / "membership.parquet"
    abstracts_path = result_root / "abstracts.parquet"
    pl.DataFrame({"uid1": ["D0"], "uid2": ["D1"], "rel_sum2": [1.0]}).write_parquet(edges_path)
    pl.DataFrame({"uid": ["D0", "D1"], "cluster": [0, 0]}).write_parquet(membership_path)
    export_vosviewer_network(
        pl.read_parquet(edges_path),
        pl.read_parquet(membership_path),
        result_root / "vosviewer",
        abstracts=pl.read_parquet(abstracts_path),
        result_root=result_root,
        source_paths={"edges": edges_path, "membership": membership_path, "abstracts": abstracts_path},
    )
    keywords = pd.DataFrame(
        {
            "cluster_id": [0, 0],
            "term": ["class htmlview paragraph", "gnn"],
            "raw_term": ["class htmlview paragraph", "gnn"],
            "quality_flags": ["metadata_fragment", ""],
            "keyword_label_tier": ["review_artifact", "supporting"],
            "abbreviation_target": ["", "graph neural networks"],
            "abbreviation_status": ["", "confirmed"],
        }
    )
    rule_artifact = write_keyword_cleaning_rule_artifacts(result_root, keywords=keywords)
    export_vosviewer_thesaurus(rule_artifact["manifest_path"], result_root / "vosviewer", result_root=result_root)
    write_cooccurrence_artifacts(result_root)
    export_vosviewer_term_cooccurrence(result_root)
    write_matrix_from_term_cooccurrence(result_root)
    export_matrix_artifact(result_root, export_format="vosviewer-network")

    written = export_vosviewer_bundle(result_root)

    assert written["bundle_path"] == result_root / "exports" / "vosviewer_bundle" / "vosviewer_bundle.zip"
    validation = validate_export_manifest(written["manifest_path"]).to_dict()
    assert validation["status"] == "passed"
    assert validation["export_family"] == "bundle"
    assert validation["export_kind"] == "vosviewer_bundle"
    export_manifest = json.loads(written["manifest_path"].read_text(encoding="utf-8"))
    assert export_manifest["selection"]["scope"] == "manifest_backed_exports"
    assert export_manifest["selection"]["view"]["mode"] == "download_bundle"
    assert export_manifest["selection"]["filters"] == [
        {"field": "export_family", "op": "eq", "value": "vosviewer"}
    ]
    assert export_manifest["selection"]["layer_state"] == {
        "source_inventory": "result_manifest.exports",
        "bundle_file_count": 16,
    }

    with zipfile.ZipFile(written["bundle_path"]) as archive:
        names = set(archive.namelist())
        assert {
            "vosviewer/vosviewer_map.txt",
            "vosviewer/vosviewer_network.txt",
            "vosviewer/vosviewer_term_map.txt",
            "vosviewer/vosviewer_term_network.txt",
            "vosviewer/vosviewer_thesaurus.txt",
            "vosviewer/sciscape_keyword_rules.tsv",
            "exports/matrix_term_cooccurrence_default_vosviewer_network/vosviewer_matrix_map.txt",
            "exports/matrix_term_cooccurrence_default_vosviewer_network/vosviewer_matrix_network.txt",
            "exports/vosviewer_map_network/export_manifest.json",
            "exports/vosviewer_map_network/export_qa.json",
            "exports/vosviewer_term_cooccurrence/export_manifest.json",
            "exports/vosviewer_term_cooccurrence/export_qa.json",
            "exports/vosviewer_thesaurus/export_manifest.json",
            "exports/vosviewer_thesaurus/export_qa.json",
            "exports/matrix_term_cooccurrence_default_vosviewer_network/export_manifest.json",
            "exports/matrix_term_cooccurrence_default_vosviewer_network/export_qa.json",
            "vosviewer_bundle_inventory.json",
        }.issubset(names)
        inventory = json.loads(archive.read("vosviewer_bundle_inventory.json").decode("utf-8"))
    assert inventory["source"] == "result_manifest.exports"
    assert inventory["file_count"] == 16

    result_manifest = write_result_manifest(result_root).to_dict()
    bundle_exports = [export for export in result_manifest["exports"] if export["export_id"] == "vosviewer_bundle"]
    assert len(bundle_exports) == 1
    assert bundle_exports[0]["path"] == "exports/vosviewer_bundle/vosviewer_bundle.zip"
    assert bundle_exports[0]["selection_summary"]["view_mode"] == "download_bundle"
    assert bundle_exports[0]["selection_summary"]["filter_count"] == 1


def test_cli_rule_export_supports_vosviewer_thesaurus(tmp_path):
    result_root = _write_valid_result_root(tmp_path / "result")
    keywords = pd.DataFrame(
        {
            "cluster_id": [0, 0],
            "term": ["class htmlview paragraph", "gnn"],
            "raw_term": ["class htmlview paragraph", "gnn"],
            "quality_flags": ["metadata_fragment", ""],
            "keyword_label_tier": ["review_artifact", "supporting"],
            "abbreviation_target": ["", "graph neural networks"],
            "abbreviation_status": ["", "confirmed"],
        }
    )
    rule_artifact = write_keyword_cleaning_rule_artifacts(result_root, keywords=keywords)
    output_dir = result_root / "vosviewer_cli"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "sciscape.cli",
            "rule-export",
            str(rule_artifact["manifest_path"]),
            "--format",
            "vosviewer-thesaurus",
            "-o",
            str(output_dir),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "vosviewer_thesaurus.txt" in completed.stdout
    assert "sciscape_keyword_rules.tsv" in completed.stdout
    assert (output_dir / "vosviewer_thesaurus.txt").exists()
    assert (output_dir / "sciscape_keyword_rules.tsv").exists()
    manifest_path = result_root / "exports" / "vosviewer_thesaurus" / "export_manifest.json"
    assert validate_export_manifest(manifest_path).ok is True


def test_cli_export_supports_vosviewer_format(tmp_path):
    import polars as pl

    edges_path = tmp_path / "edges.parquet"
    membership_path = tmp_path / "membership.parquet"
    output_dir = tmp_path / "vosviewer_cli"
    pl.DataFrame({"uid1": ["D0"], "uid2": ["D1"], "rel_sum2": [1.0]}).write_parquet(edges_path)
    pl.DataFrame({"uid": ["D0", "D1"], "cluster": [0, 0]}).write_parquet(membership_path)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "sciscape.cli",
            "export",
            str(edges_path),
            str(membership_path),
            "--format",
            "vosviewer",
            "-o",
            str(output_dir),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "vosviewer_map.txt" in completed.stdout
    assert (output_dir / "vosviewer_map.txt").exists()
    assert (output_dir / "vosviewer_network.txt").exists()
    manifest_path = output_dir / "exports" / "vosviewer_map_network" / "export_manifest.json"
    assert validate_export_manifest(manifest_path).ok is True
