from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from sciscape.artifacts import (
    COOCCURRENCE_ARTIFACT_SCHEMA_VERSION,
    EVOLUTION_EVENTS_SCHEMA_VERSION,
    EVOLUTION_MANIFEST_SCHEMA_VERSION,
    EVOLUTION_QA_SCHEMA_VERSION,
    MATRIX_MANIFEST_SCHEMA_VERSION,
    MATRIX_QA_SCHEMA_VERSION,
    MATRIX_VALUES_SCHEMA_VERSION,
    RESULT_MANIFEST_SCHEMA_VERSION,
    TEMPORAL_ACTIVITY_SCHEMA_VERSION,
    TEMPORAL_MANIFEST_SCHEMA_VERSION,
    TEMPORAL_QA_SCHEMA_VERSION,
    WORKSPACE_MANIFEST_SCHEMA_VERSION,
    WORKSPACE_QA_SCHEMA_VERSION,
    build_atlas_payload_from_report_data,
    build_result_manifest,
    build_report_data_contract,
    load_result_manifest,
    register_result_in_workspace,
    validate_evolution_artifact,
    validate_matrix_artifact,
    validate_result_root,
    validate_temporal_artifact,
    validate_workspace,
    write_cooccurrence_artifacts,
    write_edge_evidence_samples,
    write_evolution_artifacts,
    write_evolution_synthetic_smoke_artifact,
    write_matrix_artifact,
    write_matrix_from_term_cooccurrence,
    write_temporal_artifacts,
    write_artifact_contract,
    write_result_manifest,
    write_workspace_manifest,
)
from sciscape.keyword_extraction.visualization import export_dashboard, export_report


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
    assert payload["features"]["matrix"] is True
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
    assert contract["features"]["matrix"] is True

    manifest = build_result_manifest(root).to_dict()
    assert manifest["features"]["cooccurrence"]["state"] == "stable"
    assert manifest["features"]["cooccurrence"]["reason"] == "feature validated"
    assert "cooccurrence" in manifest["features"]["cooccurrence"]["artifact_refs"]
    assert manifest["artifacts"]["cooccurrence"]["schema_version"] == COOCCURRENCE_ARTIFACT_SCHEMA_VERSION
    assert manifest["artifacts"]["cooccurrence"]["rows"] == 2


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
    transitions = pd.read_parquet(written["transitions_path"])
    events = pd.read_parquet(written["events_path"])
    assert len(states) == 4
    assert len(transitions) == 2
    assert set(events["event_type"]) == {"continuation"}

    validation = validate_evolution_artifact(written["manifest_path"]).to_dict()
    assert validation["status"] == "passed"
    assert validation["counts"]["slices"] == 2
    assert validation["counts"]["states"] == 4
    assert validation["counts"]["transitions"] == 2

    contract = validate_result_root(root).to_dict()
    assert contract["features"]["evolution"] is True
    assert contract["counts"]["stable_evolution_artifacts"] == 1
    assert contract["counts"]["evolution_slices"] == 2
    assert contract["counts"]["evolution_transitions"] == 2

    manifest = build_result_manifest(root).to_dict()
    assert manifest["features"]["evolution"]["state"] == "stable"


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
