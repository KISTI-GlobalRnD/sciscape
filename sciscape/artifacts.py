"""Result artifact validation and feature inference for SciScape outputs."""

from __future__ import annotations

import html
import json
import math
import re
import shlex
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import __version__ as SCISCAPE_VERSION
from .evolution import (
    build_document_overlap_evolution,
    build_evidence_backed_evolution,
    build_membership_projection_evolution,
    build_slice_local_membership_evidence,
    build_slice_reclustering_membership,
    build_slice_membership_evidence,
)


ARTIFACT_CONTRACT_SCHEMA_VERSION = "sciscape_artifact_contract_v1"
RESULT_MANIFEST_SCHEMA_VERSION = "sciscape_result_manifest_v1"
WORKSPACE_MANIFEST_SCHEMA_VERSION = "sciscape_workspace_manifest_v1"
WORKSPACE_QA_SCHEMA_VERSION = "sciscape_workspace_qa_v1"
REPORT_DATA_CONTRACT_SCHEMA_VERSION = "sciscape_report_data_contract_v1"
ATLAS_PAYLOAD_SCHEMA_VERSION = "sciscape_atlas_payload_v1"
ATLAS_RENDER_PAYLOAD_SCHEMA_VERSION = "sciscape_atlas_render_payload_v1"
EDGE_EVIDENCE_SCHEMA_VERSION = "sciscape_edge_evidence_samples_v1"
COOCCURRENCE_ARTIFACT_SCHEMA_VERSION = "sciscape_cooccurrence_artifact_v1"
CLUSTER_REVIEW_PACKET_SCHEMA_VERSION = "sciscape_cluster_review_packet_v1"
CLUSTER_REVIEW_PACKET_QA_SCHEMA_VERSION = "sciscape_cluster_review_packet_qa_v1"
NARRATIVE_MANIFEST_SCHEMA_VERSION = "sciscape_narrative_manifest_v1"
NARRATIVE_TARGETS_SCHEMA_VERSION = "sciscape_narrative_targets_v1"
NARRATIVE_CLAIMS_SCHEMA_VERSION = "sciscape_narrative_claims_v1"
NARRATIVE_EVIDENCE_SOURCES_SCHEMA_VERSION = "sciscape_narrative_evidence_sources_v1"
NARRATIVE_EVIDENCE_REFS_SCHEMA_VERSION = "sciscape_narrative_evidence_refs_v1"
NARRATIVE_CLAIM_EVIDENCE_LINKS_SCHEMA_VERSION = "sciscape_narrative_claim_evidence_links_v1"
NARRATIVE_SECTIONS_SCHEMA_VERSION = "sciscape_narrative_sections_v1"
NARRATIVE_REVIEW_DECISIONS_SCHEMA_VERSION = "sciscape_narrative_review_decisions_v1"
NARRATIVE_QA_SCHEMA_VERSION = "sciscape_narrative_qa_v1"
MATRIX_MANIFEST_SCHEMA_VERSION = "sciscape_matrix_manifest_v1"
MATRIX_VALUES_SCHEMA_VERSION = "sciscape_matrix_values_sparse_triplet_v1"
MATRIX_ENTITIES_SCHEMA_VERSION = "sciscape_matrix_entities_v1"
MATRIX_QA_SCHEMA_VERSION = "sciscape_matrix_qa_v1"
TEMPORAL_MANIFEST_SCHEMA_VERSION = "sciscape_temporal_manifest_v1"
TEMPORAL_PERIODS_SCHEMA_VERSION = "sciscape_temporal_periods_v1"
TEMPORAL_ACTIVITY_SCHEMA_VERSION = "sciscape_temporal_activity_v1"
TEMPORAL_ENTITY_SERIES_SCHEMA_VERSION = "sciscape_temporal_entity_series_v1"
TEMPORAL_EVENTS_SCHEMA_VERSION = "sciscape_temporal_events_v1"
TEMPORAL_QA_SCHEMA_VERSION = "sciscape_temporal_qa_v1"
EVOLUTION_MANIFEST_SCHEMA_VERSION = "sciscape_evolution_manifest_v1"
EVOLUTION_TIME_SLICES_SCHEMA_VERSION = "sciscape_evolution_time_slices_v1"
EVOLUTION_CLUSTER_STATES_SCHEMA_VERSION = "sciscape_evolution_cluster_states_v1"
EVOLUTION_TRANSITIONS_SCHEMA_VERSION = "sciscape_evolution_transitions_v1"
EVOLUTION_LINEAGES_SCHEMA_VERSION = "sciscape_evolution_lineages_v1"
EVOLUTION_EVENTS_SCHEMA_VERSION = "sciscape_evolution_events_v1"
EVOLUTION_STATE_MEMBERSHIP_SCHEMA_VERSION = "sciscape_evolution_state_membership_v1"
EVOLUTION_QA_SCHEMA_VERSION = "sciscape_evolution_qa_v1"
EVOLUTION_SYNTHETIC_SMOKE_SCHEMA_VERSION = "sciscape_evolution_synthetic_smoke_v1"
EXPORT_MANIFEST_SCHEMA_VERSION = "sciscape_export_manifest_v1"
EXPORT_FILES_SCHEMA_VERSION = "sciscape_export_files_v1"
EXPORT_INPUTS_SCHEMA_VERSION = "sciscape_export_inputs_v1"
EXPORT_TRANSFORMS_SCHEMA_VERSION = "sciscape_export_transforms_v1"
EXPORT_QA_SCHEMA_VERSION = "sciscape_export_qa_v1"
KEYWORD_RULE_MANIFEST_SCHEMA_VERSION = "sciscape_keyword_rule_set_manifest_v1"
KEYWORD_RULES_SCHEMA_VERSION = "sciscape_keyword_rules_v1"
KEYWORD_RULE_APPLICATIONS_SCHEMA_VERSION = "sciscape_keyword_rule_applications_v1"
KEYWORD_TERM_BEFORE_AFTER_SCHEMA_VERSION = "sciscape_keyword_term_before_after_v1"
KEYWORD_RULE_IMPACT_SUMMARY_SCHEMA_VERSION = "sciscape_keyword_rule_impact_summary_v1"
KEYWORD_RULE_QA_SCHEMA_VERSION = "sciscape_keyword_rule_qa_v1"
FEATURE_KEYS = (
    "overview",
    "cluster_map",
    "keyword",
    "term_network",
    "matrix",
    "evidence",
    "temporal",
    "evolution",
    "narrative",
    "quality",
    "export",
)
RESULT_MANIFEST_FEATURE_KEYS = (
    "overview",
    "cluster_map",
    "keyword",
    "term_network",
    "cooccurrence",
    "matrix",
    "evidence",
    "temporal",
    "evolution",
    "narrative",
    "quality",
    "export",
)
WORKSPACE_OBJECT_FAMILIES = (
    "projects",
    "datasets",
    "runs",
    "results",
    "rule_sets",
    "views",
    "exports",
)
WORKSPACE_OBJECT_ID_KEYS = {
    "projects": "project_id",
    "datasets": "dataset_id",
    "runs": "run_id",
    "results": "result_id",
    "rule_sets": "rule_set_id",
    "views": "view_id",
    "exports": "export_id",
}
REQUIRED_ABSTRACT_COLUMNS = {"uid", "title", "abstract", "pubyear"}
REQUIRED_MEMBERSHIP_COLUMNS = {"uid"}
REQUIRED_KEYWORD_COLUMNS = {"cluster_id", "term"}
REQUIRED_EDGE_COLUMNS = {"uid1", "uid2"}
REQUIRED_COOCCURRENCE_COLUMNS = {
    "schema_version",
    "cluster_uid",
    "cluster_level",
    "cluster_id",
    "source",
    "target",
    "weight",
    "relation",
}
REQUIRED_MATRIX_VALUES_COLUMNS = {
    "schema_version",
    "matrix_id",
    "row_key",
    "column_key",
    "row_index",
    "column_index",
    "value",
    "relation",
}
REQUIRED_MATRIX_ENTITY_COLUMNS = {
    "schema_version",
    "matrix_id",
    "entity_key",
    "entity_index",
    "entity_type",
    "label",
}
SUPPORTED_MATRIX_FAMILIES = frozenset(
    {"occurrence", "cooccurrence", "proximity", "similarity", "projection", "temporal"}
)
SUPPORTED_EXPORT_FAMILIES = frozenset({"report", "viewer", "graph", "table", "matrix", "map", "vosviewer", "bundle"})
REQUIRED_EXPORT_FILE_COLUMNS = {
    "schema_version",
    "export_id",
    "file_id",
    "path",
    "role",
    "format",
    "public_share_state",
}
REQUIRED_EXPORT_INPUT_COLUMNS = {
    "schema_version",
    "export_id",
    "input_id",
    "artifact_ref",
    "artifact_role",
    "artifact_path",
    "feature_state",
    "required",
}
REQUIRED_EXPORT_TRANSFORM_COLUMNS = {
    "schema_version",
    "export_id",
    "transform_id",
    "step_index",
    "transform_type",
    "description",
    "parameters",
}
REQUIRED_NARRATIVE_TARGET_COLUMNS = {
    "schema_version",
    "narrative_id",
    "target_id",
    "target_type",
    "target_key",
    "target_label",
    "feature_state",
}
REQUIRED_NARRATIVE_CLAIM_COLUMNS = {
    "schema_version",
    "narrative_id",
    "claim_id",
    "target_id",
    "section_id",
    "claim_type",
    "claim_text",
    "support_state",
    "confidence",
    "evidence_ref_count",
    "text_origin",
    "review_state",
}
REQUIRED_NARRATIVE_EVIDENCE_SOURCE_COLUMNS = {
    "schema_version",
    "narrative_id",
    "evidence_source_id",
    "artifact_ref",
    "artifact_role",
    "artifact_path",
    "resolver",
    "source_state",
}
REQUIRED_NARRATIVE_EVIDENCE_REF_COLUMNS = {
    "schema_version",
    "narrative_id",
    "evidence_ref_id",
    "evidence_source_id",
    "evidence_type",
    "entity_type",
    "entity_key",
    "locator_type",
    "locator",
    "evidence_label",
}
REQUIRED_NARRATIVE_CLAIM_LINK_COLUMNS = {
    "schema_version",
    "narrative_id",
    "claim_id",
    "evidence_ref_id",
    "evidence_role",
    "link_strength",
    "required",
}
REQUIRED_NARRATIVE_SECTION_COLUMNS = {
    "schema_version",
    "narrative_id",
    "section_id",
    "target_id",
    "section_type",
    "section_title",
    "section_state",
    "claim_count",
}
REQUIRED_NARRATIVE_REVIEW_COLUMNS = {
    "schema_version",
    "narrative_id",
    "decision_id",
    "claim_id",
    "decision_type",
    "reviewer",
    "decided_at_utc",
    "reason",
}
REQUIRED_TEMPORAL_PERIOD_COLUMNS = {
    "schema_version",
    "temporal_id",
    "period_id",
    "period_index",
    "period_label",
    "start_year",
    "end_year",
    "unit",
}
REQUIRED_TEMPORAL_ACTIVITY_COLUMNS = {
    "schema_version",
    "temporal_id",
    "period_id",
    "start_year",
    "end_year",
    "doc_count",
    "edge_count",
    "active_cluster_count",
    "unknown_year_count",
}
REQUIRED_TEMPORAL_SERIES_COLUMNS = {
    "schema_version",
    "temporal_id",
    "entity_type",
    "entity_key",
    "entity_label",
    "period_id",
    "metric",
    "value",
    "raw_value",
    "denominator",
    "support_count",
}
REQUIRED_TEMPORAL_EVENTS_COLUMNS = {
    "schema_version",
    "temporal_id",
    "event_id",
    "event_type",
    "entity_type",
    "entity_key",
    "entity_label",
    "start_period_id",
    "end_period_id",
    "metric",
    "score",
    "method",
    "support_count",
}
REQUIRED_EVOLUTION_TIME_SLICE_COLUMNS = {
    "schema_version",
    "evolution_id",
    "slice_id",
    "slice_index",
    "slice_label",
    "start_year",
    "end_year",
    "unit",
    "doc_count",
}
REQUIRED_EVOLUTION_STATE_COLUMNS = {
    "schema_version",
    "evolution_id",
    "state_id",
    "slice_id",
    "slice_index",
    "cluster_key",
    "cluster_label",
    "doc_count",
    "term_count",
    "top_terms",
}
REQUIRED_EVOLUTION_TRANSITION_COLUMNS = {
    "schema_version",
    "evolution_id",
    "transition_id",
    "source_state_id",
    "target_state_id",
    "source_slice_id",
    "target_slice_id",
    "metric",
    "score",
    "support_count",
    "source_doc_count",
    "target_doc_count",
    "relation",
}
REQUIRED_EVOLUTION_LINEAGE_COLUMNS = {
    "schema_version",
    "evolution_id",
    "lineage_id",
    "state_id",
    "slice_id",
    "slice_index",
    "role",
    "stability_score",
}
REQUIRED_EVOLUTION_EVENT_COLUMNS = {
    "schema_version",
    "evolution_id",
    "event_id",
    "event_type",
    "slice_id",
    "state_id",
    "lineage_id",
    "transition_refs",
    "score",
    "support_count",
    "method",
}
EVOLUTION_EVENT_TYPES = frozenset({"continuation", "split", "merge", "emergence", "decline", "ambiguous"})
REQUIRED_KEYWORD_RULE_COLUMNS = {
    "schema_version",
    "rule_set_id",
    "rule_id",
    "rule_family",
    "match_type",
    "pattern",
    "replacement",
    "action",
    "confidence_policy",
    "destructive",
    "enabled",
    "created_by",
    "reason",
}
REQUIRED_KEYWORD_RULE_APPLICATION_COLUMNS = {
    "schema_version",
    "rule_set_id",
    "application_id",
    "rule_id",
    "cluster_id",
    "raw_term",
    "normalized_term_before",
    "display_label_before",
    "normalized_term_after",
    "display_label_after",
    "action",
    "decision",
    "evidence_type",
    "evidence_value",
    "score_before",
    "score_after",
    "frequency",
    "rank_before",
    "rank_after",
}
REQUIRED_KEYWORD_TERM_BEFORE_AFTER_COLUMNS = {
    "schema_version",
    "rule_set_id",
    "cluster_id",
    "raw_term",
    "term_before",
    "term_after",
    "display_label",
    "family_id",
    "parent_term",
    "variant_count",
    "rule_ids",
    "quality_flags",
    "review_status",
    "tier_before",
    "tier_after",
    "blocked",
    "block_reason",
}
KEYWORD_RULE_FAMILIES = frozenset(
    {
        "artifact_block",
        "metadata_block",
        "latex_fragment",
        "html_fragment",
        "stop_term",
        "alias",
        "acronym_expand",
        "subphrase_group",
        "spelling_normalize",
        "plural_singular",
        "tier_adjust",
        "review_flag",
    }
)
KEYWORD_RULE_ACTIONS = frozenset(
    {
        "block",
        "flag",
        "normalize",
        "alias_to",
        "expand_to",
        "group_under",
        "tier_down",
        "keep_with_flag",
    }
)
KEYWORD_RULE_BLOCK_FAMILIES = frozenset({"artifact_block", "metadata_block", "latex_fragment", "html_fragment"})
WEIGHT_COLUMN_CANDIDATES = (
    "rel_sum2",
    "weight",
    "score",
    "similarity",
    "cosine",
    "edge_weight",
)
KEYWORD_ARTIFACT_TOP_K = 10
BLOCKING_ARTIFACT_FLAGS = frozenset({"metadata_fragment"})
REVIEW_ARTIFACT_FLAGS = frozenset(
    {
        "artifact_like",
        "artifact_formula",
        "compact_formula_fragment",
        "dimension_fragment",
        "mixed_formula_fragment",
        "unresolved_compact_short_form",
    }
)


@dataclass(frozen=True)
class ArtifactIssue:
    code: str
    severity: str
    message: str
    artifact: str | None = None

    @property
    def is_blocking(self) -> bool:
        return self.severity in {"error", "blocking"}


@dataclass(frozen=True)
class ArtifactTableInfo:
    path: str
    exists: bool
    rows: int | None = None
    columns: list[str] | None = None


@dataclass(frozen=True)
class ResultArtifacts:
    input_path: Path
    result_root: Path
    landscape_dir: Path | None = None
    report_data_path: Path | None = None
    abstracts_path: Path | None = None
    edges_path: Path | None = None
    membership_path: Path | None = None
    keywords_path: Path | None = None
    matrix_paths: tuple[Path, ...] = ()
    matrix_manifest_paths: tuple[Path, ...] = ()
    keyword_rule_manifest_paths: tuple[Path, ...] = ()
    temporal_manifest_paths: tuple[Path, ...] = ()
    evolution_manifest_paths: tuple[Path, ...] = ()
    export_manifest_paths: tuple[Path, ...] = ()
    edge_evidence_paths: tuple[Path, ...] = ()
    evolution_paths: tuple[Path, ...] = ()
    narrative_manifest_paths: tuple[Path, ...] = ()
    narrative_paths: tuple[Path, ...] = ()
    review_packet_paths: tuple[Path, ...] = ()
    qa_path: Path | None = None


@dataclass(frozen=True)
class ArtifactValidationResult:
    schema_version: str
    mode: str
    result_state: str
    result_root: str
    source_uri: str
    features: dict[str, bool]
    warnings: list[dict[str, Any]]
    versions: dict[str, Any]
    artifacts: dict[str, Any]
    counts: dict[str, int]
    created_at_utc: str

    @property
    def ok(self) -> bool:
        return self.result_state != "blocked"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["ok"] = self.ok
        return data


@dataclass(frozen=True)
class ArtifactRecord:
    role: str
    path: str
    format: str
    status: str
    required_for: list[str]
    schema_version: str | None = None
    rows: int | None = None
    columns: list[str] | None = None
    size_bytes: int | None = None
    checksum: str | None = None
    created_at_utc: str | None = None
    description: str | None = None
    warnings: list[dict[str, Any]] | None = None


@dataclass(frozen=True)
class FeatureExposure:
    state: str
    reason: str
    artifact_refs: list[str]
    warnings: list[dict[str, Any]]


@dataclass(frozen=True)
class RunState:
    status: str
    started_at_utc: str | None = None
    finished_at_utc: str | None = None
    heartbeat_at_utc: str | None = None
    progress: dict[str, Any] | None = None
    shards: dict[str, int] | None = None
    checkpoints: list[dict[str, Any]] | None = None
    partial_outputs: list[dict[str, Any]] | None = None
    failure: dict[str, Any] | None = None
    resume: dict[str, Any] | None = None


@dataclass(frozen=True)
class ResultManifest:
    schema_version: str
    result_id: str
    title: str
    result_kind: str
    created_at_utc: str
    sciscape_version: str
    result_root: str
    source: dict[str, Any]
    run_state: dict[str, Any]
    artifacts: dict[str, dict[str, Any]]
    features: dict[str, dict[str, Any]]
    quality: dict[str, Any]
    exports: list[dict[str, Any]]
    provenance: dict[str, Any]
    updated_at_utc: str | None = None
    description: str | None = None
    tags: list[str] | None = None
    ui: dict[str, Any] | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WorkspaceValidationResult:
    schema_version: str
    workspace_id: str | None
    state: str
    status: str
    workspace_root: str
    manifest_path: str | None
    qa_path: str
    counts: dict[str, int]
    checks: dict[str, dict[str, Any]]
    objects: dict[str, list[dict[str, Any]]]
    defaults: dict[str, Any]
    recent: dict[str, list[str]]
    warnings: list[dict[str, Any]]
    blocking_issues: list[dict[str, Any]]
    created_at_utc: str

    @property
    def ok(self) -> bool:
        return self.status != "blocked"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["ok"] = self.ok
        return data


@dataclass(frozen=True)
class MatrixArtifactValidationResult:
    schema_version: str
    matrix_id: str | None
    matrix_family: str | None
    status: str
    matrix_dir: str
    manifest_path: str
    paths: dict[str, str | None]
    counts: dict[str, int]
    checks: dict[str, dict[str, Any]]
    warnings: list[dict[str, Any]]
    blocking_issues: list[dict[str, Any]]
    created_at_utc: str

    @property
    def ok(self) -> bool:
        return self.status != "blocked"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["ok"] = self.ok
        return data


@dataclass(frozen=True)
class ExportManifestValidationResult:
    schema_version: str
    export_id: str | None
    export_family: str | None
    export_kind: str | None
    status: str
    export_dir: str
    manifest_path: str
    paths: dict[str, str | None]
    counts: dict[str, int]
    checks: dict[str, dict[str, Any]]
    compatibility: dict[str, Any]
    warnings: list[dict[str, Any]]
    blocking_issues: list[dict[str, Any]]
    created_at_utc: str

    @property
    def ok(self) -> bool:
        return self.status != "blocked"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["ok"] = self.ok
        return data


@dataclass(frozen=True)
class KeywordRuleValidationResult:
    schema_version: str
    rule_set_id: str | None
    status: str
    rule_dir: str
    manifest_path: str
    paths: dict[str, str | None]
    counts: dict[str, int]
    rule_family_counts: dict[str, int]
    action_counts: dict[str, int]
    contamination_counts: dict[str, int]
    checks: dict[str, dict[str, Any]]
    warnings: list[dict[str, Any]]
    blocking_issues: list[dict[str, Any]]
    created_at_utc: str

    @property
    def ok(self) -> bool:
        return self.status != "blocked"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["ok"] = self.ok
        return data


@dataclass(frozen=True)
class TemporalArtifactValidationResult:
    schema_version: str
    temporal_id: str | None
    status: str
    temporal_dir: str
    manifest_path: str
    paths: dict[str, str | None]
    counts: dict[str, int]
    event_counts: dict[str, int]
    checks: dict[str, dict[str, Any]]
    warnings: list[dict[str, Any]]
    blocking_issues: list[dict[str, Any]]
    created_at_utc: str

    @property
    def ok(self) -> bool:
        return self.status != "blocked"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["ok"] = self.ok
        return data


@dataclass(frozen=True)
class EvolutionArtifactValidationResult:
    schema_version: str
    evolution_id: str | None
    status: str
    evolution_dir: str
    manifest_path: str
    paths: dict[str, str | None]
    counts: dict[str, int]
    event_counts: dict[str, int]
    checks: dict[str, dict[str, Any]]
    warnings: list[dict[str, Any]]
    blocking_issues: list[dict[str, Any]]
    created_at_utc: str

    @property
    def ok(self) -> bool:
        return self.status != "blocked"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["ok"] = self.ok
        return data


@dataclass(frozen=True)
class ClusterReviewPacketValidationResult:
    schema_version: str
    packet_id: str | None
    status: str
    packet_path: str
    qa_path: str | None
    counts: dict[str, int]
    checks: dict[str, dict[str, Any]]
    warnings: list[dict[str, Any]]
    blocking_issues: list[dict[str, Any]]
    created_at_utc: str

    @property
    def ok(self) -> bool:
        return self.status != "blocked"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["ok"] = self.ok
        return data


@dataclass(frozen=True)
class NarrativeArtifactValidationResult:
    schema_version: str
    narrative_id: str | None
    status: str
    narrative_dir: str
    manifest_path: str
    paths: dict[str, str | None]
    counts: dict[str, int]
    claim_counts: dict[str, int]
    checks: dict[str, dict[str, Any]]
    warnings: list[dict[str, Any]]
    blocking_issues: list[dict[str, Any]]
    feature_state: str
    created_at_utc: str

    @property
    def ok(self) -> bool:
        return self.status != "blocked"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["ok"] = self.ok
        return data


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _rel(path: Path | None, root: Path) -> str | None:
    if path is None:
        return None
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _looks_like_landscape_dir(path: Path) -> bool:
    return (
        (path / "membership.parquet").exists()
        or (path / "keywords.parquet").exists()
        or (path / "report" / "data.json").exists()
    )


def _find_landscape_dir(root: Path) -> Path | None:
    direct = root / "landscape"
    if _looks_like_landscape_dir(direct):
        return direct
    candidates = [
        child
        for child in root.iterdir()
        if child.is_dir() and child.name.startswith("landscape") and _looks_like_landscape_dir(child)
    ] if root.exists() and root.is_dir() else []
    if candidates:
        return sorted(candidates, key=lambda p: (p.name != "landscape", p.name))[0]
    return None


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _collect_optional_artifacts(
    bases: list[Path | None],
    patterns: tuple[str, ...],
) -> tuple[Path, ...]:
    paths: list[Path] = []
    for base in bases:
        if base is None or not base.exists() or not base.is_dir():
            continue
        for pattern in patterns:
            paths.extend(path for path in base.glob(pattern) if path.is_file())
    return tuple(sorted(set(paths)))


def infer_result_artifacts(path: str | Path) -> ResultArtifacts:
    """Infer standard SciScape artifact paths from a result root or data file."""

    input_path = Path(path).expanduser()
    if input_path.exists():
        input_path = input_path.resolve()

    if input_path.is_file() and input_path.name == "data.json":
        report_data = input_path
        if input_path.parent.name == "report":
            landscape_dir = input_path.parent.parent
            result_root = landscape_dir.parent
        else:
            landscape_dir = None
            result_root = input_path.parent
    elif input_path.is_file() and input_path.name == "evolution_manifest.json" and input_path.parent.name == "evolution":
        report_data = None
        landscape_dir = None
        result_root = input_path.parent.parent
    elif input_path.is_file():
        report_data = input_path if input_path.name.endswith(".json") else None
        landscape_dir = input_path.parent if _looks_like_landscape_dir(input_path.parent) else None
        result_root = landscape_dir.parent if landscape_dir else input_path.parent
    elif input_path.name == "report" and input_path.is_dir():
        report_data = input_path / "data.json"
        landscape_dir = input_path.parent if _looks_like_landscape_dir(input_path.parent) else None
        result_root = landscape_dir.parent if landscape_dir else input_path.parent
    elif _looks_like_landscape_dir(input_path):
        landscape_dir = input_path
        result_root = input_path.parent
        report_data = input_path / "report" / "data.json"
    else:
        result_root = input_path
        landscape_dir = _find_landscape_dir(result_root)
        report_data = (
            landscape_dir / "report" / "data.json"
            if landscape_dir is not None
            else result_root / "data.json"
        )

    if report_data is not None and not report_data.exists():
        report_data = None

    abstracts = _first_existing([
        result_root / "abstracts.parquet",
        result_root / "abstracts_subset.parquet",
        landscape_dir / "abstracts.parquet" if landscape_dir else result_root / "_missing_abstracts",
    ])
    edges = _first_existing([
        result_root / "edges.parquet",
        result_root / "combined_edges.parquet",
        landscape_dir / "edges.parquet" if landscape_dir else result_root / "_missing_edges",
    ])
    membership = _first_existing([
        landscape_dir / "membership.parquet" if landscape_dir else result_root / "_missing_membership",
        result_root / "membership.parquet",
    ])
    keywords = _first_existing([
        landscape_dir / "keywords.parquet" if landscape_dir else result_root / "_missing_keywords",
        result_root / "keywords.parquet",
    ])

    matrix_paths: list[Path] = []
    matrix_manifest_paths: list[Path] = []
    keyword_rule_manifest_paths: list[Path] = []
    temporal_manifest_paths: list[Path] = []
    evolution_manifest_paths: list[Path] = []
    narrative_manifest_paths: list[Path] = []
    export_manifest_paths: list[Path] = []
    for base in [landscape_dir, result_root]:
        if base is None or not base.exists() or not base.is_dir():
            continue
        for pattern in ("*matrix*.parquet", "*cooccurrence*.parquet", "*cooccurrence*.json"):
            matrix_paths.extend(path for path in base.glob(pattern) if path.is_file())
        matrix_manifest_paths.extend(
            path for path in (base / "matrices").glob("*/matrix_manifest.json") if path.is_file()
        )
        keyword_rule_manifest_paths.extend(
            path for path in (base / "rules").glob("*/rule_set_manifest.json") if path.is_file()
        )
        export_manifest_paths.extend(
            path for path in (base / "exports").glob("*/export_manifest.json") if path.is_file()
        )
        temporal_manifest = base / "temporal" / "temporal_manifest.json"
        if temporal_manifest.exists() and temporal_manifest.is_file():
            temporal_manifest_paths.append(temporal_manifest)
        evolution_manifest = base / "evolution" / "evolution_manifest.json"
        if evolution_manifest.exists() and evolution_manifest.is_file():
            evolution_manifest_paths.append(evolution_manifest)
        narrative_manifest = base / "narrative" / "narrative_manifest.json"
        if narrative_manifest.exists() and narrative_manifest.is_file():
            narrative_manifest_paths.append(narrative_manifest)

    evolution_paths = _collect_optional_artifacts(
        [
            landscape_dir,
            landscape_dir / "report" if landscape_dir else None,
            landscape_dir / "evolution" if landscape_dir else None,
            result_root,
        ],
        ("*evolution*.json", "*evolution*.parquet", "*trajectory*.json", "*trajectory*.parquet"),
    )
    evolution_manifest_set = set(evolution_manifest_paths)
    evolution_paths = tuple(
        path
        for path in evolution_paths
        if path not in evolution_manifest_set
        and not (path.parent.name == "evolution" and (path.parent / "evolution_manifest.json").exists())
    )
    edge_evidence_paths = _collect_optional_artifacts(
        [
            landscape_dir,
            landscape_dir / "report" if landscape_dir else None,
            landscape_dir / "evidence" if landscape_dir else None,
            result_root,
        ],
        (
            "*edge*evidence*.json",
            "*edge*evidence*.parquet",
            "*neighbor*evidence*.json",
            "*neighbor*evidence*.parquet",
            "*relation*evidence*.json",
            "*relation*evidence*.parquet",
        ),
    )
    narrative_paths = _collect_optional_artifacts(
        [
            landscape_dir,
            landscape_dir / "report" if landscape_dir else None,
            landscape_dir / "narrative" if landscape_dir else None,
            result_root,
        ],
        ("*narrative*.json", "*narrative*.parquet"),
    )
    narrative_manifest_set = set(narrative_manifest_paths)
    narrative_paths = tuple(
        path
        for path in narrative_paths
        if path not in narrative_manifest_set
        and not (path.parent.name == "narrative" and (path.parent / "narrative_manifest.json").exists())
    )
    review_packet_paths = _collect_optional_artifacts(
        [
            landscape_dir / "review" if landscape_dir else None,
            result_root / "review",
        ],
        ("cluster_review_packet.json", "cluster_review_packet_*.json"),
    )
    review_packet_paths = tuple(path for path in review_packet_paths if not path.name.endswith("_qa.json"))

    qa_path = None
    for candidate in [
        landscape_dir / "qa" / "artifact_contract.json" if landscape_dir else None,
        result_root / "qa" / "artifact_contract.json",
    ]:
        if candidate and candidate.exists():
            qa_path = candidate
            break

    return ResultArtifacts(
        input_path=input_path,
        result_root=result_root,
        landscape_dir=landscape_dir,
        report_data_path=report_data,
        abstracts_path=abstracts,
        edges_path=edges,
        membership_path=membership,
        keywords_path=keywords,
        matrix_paths=tuple(sorted(set(matrix_paths))),
        matrix_manifest_paths=tuple(sorted(set(matrix_manifest_paths))),
        keyword_rule_manifest_paths=tuple(sorted(set(keyword_rule_manifest_paths))),
        temporal_manifest_paths=tuple(sorted(set(temporal_manifest_paths))),
        evolution_manifest_paths=tuple(sorted(set(evolution_manifest_paths))),
        export_manifest_paths=tuple(sorted(set(export_manifest_paths))),
        edge_evidence_paths=edge_evidence_paths,
        evolution_paths=evolution_paths,
        narrative_manifest_paths=tuple(sorted(set(narrative_manifest_paths))),
        narrative_paths=narrative_paths,
        review_packet_paths=review_packet_paths,
        qa_path=qa_path,
    )


def _parquet_info(path: Path, root: Path, issues: list[ArtifactIssue], role: str) -> ArtifactTableInfo:
    try:
        parquet_file = pq.ParquetFile(path)
    except Exception as exc:  # pragma: no cover - exact parquet errors vary
        issues.append(ArtifactIssue("invalid_parquet", "error", f"Could not read parquet: {exc}", role))
        return ArtifactTableInfo(path=_rel(path, root) or str(path), exists=True)
    return ArtifactTableInfo(
        path=_rel(path, root) or str(path),
        exists=True,
        rows=int(parquet_file.metadata.num_rows),
        columns=list(parquet_file.schema_arrow.names),
    )


def _json_payload(path: Path, issues: list[ArtifactIssue], role: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        issues.append(ArtifactIssue("invalid_json", "error", f"Could not read JSON: {exc}", role))
        return None
    if not isinstance(payload, dict):
        issues.append(ArtifactIssue("invalid_json_shape", "error", "JSON payload must be an object.", role))
        return None
    return payload


def _missing_columns(
    *,
    info: ArtifactTableInfo | None,
    required: set[str],
    issues: list[ArtifactIssue],
    role: str,
) -> set[str]:
    columns = set(info.columns or []) if info else set()
    missing = required - columns
    if missing:
        issues.append(
            ArtifactIssue(
                "missing_columns",
                "error",
                f"Missing required columns: {sorted(missing)}",
                role,
            )
        )
    return missing


def _cluster_columns(columns: list[str] | None) -> list[str]:
    if not columns:
        return []
    return [col for col in columns if col == "cluster" or col.startswith("cluster_")]


def _has_numeric_weight(path: Path, columns: list[str] | None) -> bool:
    if not columns:
        return False
    try:
        schema = pq.ParquetFile(path).schema_arrow
    except Exception:
        return False
    for column in columns:
        if column in REQUIRED_EDGE_COLUMNS:
            continue
        if column in WEIGHT_COLUMN_CANDIDATES or column.endswith("_weight"):
            field_type = schema.field(column).type
            if pa.types.is_integer(field_type) or pa.types.is_floating(field_type):
                return True
    return False


def _report_clusters(report_data: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not report_data:
        return []
    clusters = report_data.get("clusters")
    if isinstance(clusters, list):
        rows = []
        for index, item in enumerate(clusters):
            if not isinstance(item, dict):
                continue
            row = dict(item)
            row.setdefault("_cluster_index", index)
            rows.append(row)
        return rows
    rows = []
    for key, value in report_data.items():
        if str(key).startswith("_") or not isinstance(value, dict):
            continue
        row = dict(value)
        row.setdefault("_cluster_key", str(key))
        rows.append(row)
    return rows


def _report_metadata(report_data: dict[str, Any] | None) -> dict[str, Any]:
    if not report_data:
        return {}
    meta = report_data.get("_sciscape")
    if isinstance(meta, dict):
        return meta
    return {}


def _coerce_int(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(out):
        return None
    return out


def _cluster_level(cluster: Mapping[str, Any]) -> str:
    for key in ("level", "cluster_level", "level_id"):
        value = cluster.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return "cluster"


def _cluster_id(cluster: Mapping[str, Any], fallback_index: int) -> int | str:
    for key in ("cluster_id", "id"):
        value = cluster.get(key)
        if value is None:
            continue
        int_value = _coerce_int(value)
        return int_value if int_value is not None else str(value)
    key_value = cluster.get("_cluster_key")
    if key_value is not None:
        int_value = _coerce_int(key_value)
        return int_value if int_value is not None else str(key_value)
    index_value = _coerce_int(cluster.get("_cluster_index"))
    return index_value if index_value is not None else int(fallback_index)


def _short_label(label: str, *, max_chars: int = 56) -> str:
    parts = [part.strip() for part in label.split(",") if part.strip()]
    short = parts[0] if parts else label.strip()
    if len(short) <= max_chars:
        return short
    return short[: max_chars - 3].rstrip() + "..."


def _plain_atlas_text(value: Any) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _cluster_label(cluster: Mapping[str, Any], cluster_id: int | str) -> str:
    for key in ("label", "short_label", "overview_title", "name"):
        value = cluster.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return f"Cluster {cluster_id}"


def _cluster_doc_count(cluster: Mapping[str, Any]) -> tuple[int | None, str]:
    for key in ("doc_count", "work_count", "n_docs", "n_works", "size"):
        count = _coerce_int(cluster.get(key))
        if count is not None:
            return count, key
    return None, "unavailable"


def _cluster_child_count(cluster: Mapping[str, Any]) -> int:
    count = _coerce_int(cluster.get("child_count"))
    if count is not None:
        return count
    for key in ("children", "children_preview"):
        value = cluster.get(key)
        if isinstance(value, list):
            return len(value)
    return 0


def _cluster_parent_uid(cluster: Mapping[str, Any]) -> str | None:
    for key in ("parent_uid", "parent_cluster_uid", "parent"):
        value = cluster.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _cluster_keywords(cluster: Mapping[str, Any], *, limit: int = 8) -> list[dict[str, Any]]:
    keywords = cluster.get("keywords")
    if not isinstance(keywords, list):
        return []
    rows: list[dict[str, Any]] = []
    for rank, keyword in enumerate(keywords, start=1):
        if not isinstance(keyword, Mapping):
            continue
        term_values = _keyword_dict_term_values(keyword)
        term = term_values[0] if term_values else ""
        if not term:
            continue
        row: dict[str, Any] = {"term": term, "rank": rank}
        for key in ("score", "frequency", "doc_coverage", "keyword_label_tier", "keyword_scope"):
            if key in keyword:
                row[key] = keyword[key]
        rows.append(row)
        if len(rows) >= limit:
            break
    return rows


def _cluster_badges(cluster: Mapping[str, Any]) -> list[dict[str, Any]]:
    existing = cluster.get("badges")
    if isinstance(existing, list):
        return [dict(badge) for badge in existing if isinstance(badge, Mapping)]
    keywords = cluster.get("keywords")
    if not isinstance(keywords, list):
        return []
    blocking = 0
    review = 0
    for keyword in keywords:
        if not isinstance(keyword, Mapping):
            continue
        if _keyword_dict_has_blocking_artifact(keyword):
            blocking += 1
        elif _keyword_dict_has_review_artifact(keyword):
            review += 1
    badges: list[dict[str, Any]] = []
    if blocking:
        badges.append(
            {
                "badge_id": "keyword_artifact",
                "label": "Keyword artifact",
                "severity": "critical",
                "tooltip": f"{blocking} keyword rows look like metadata, HTML, or LaTeX artifacts.",
            }
        )
    if review:
        badges.append(
            {
                "badge_id": "keyword_review_artifact",
                "label": "Review keyword",
                "severity": "warning",
                "tooltip": f"{review} keyword rows require review before release-quality use.",
            }
        )
    return badges


def _cluster_representative_works(cluster: Mapping[str, Any], *, limit: int = 3) -> list[dict[str, Any]]:
    works = None
    for key in ("representative_works", "representative_papers", "papers", "documents"):
        value = cluster.get(key)
        if isinstance(value, list):
            works = value
            break
    if not works:
        return []

    rows: list[dict[str, Any]] = []
    for rank, work in enumerate(works, start=1):
        if not isinstance(work, Mapping):
            continue
        title = _plain_atlas_text(work.get("title") or work.get("display_title") or "")
        uid = str(work.get("uid") or work.get("id") or work.get("work_id") or "").strip()
        if not title and not uid:
            continue
        row: dict[str, Any] = {"rank": rank}
        if uid:
            row["uid"] = uid
        if title:
            row["title"] = title
        year = _coerce_int(work.get("pubyear") or work.get("year") or work.get("publication_year"))
        if year is not None:
            row["year"] = year
        citations = _coerce_int(work.get("cited_by_count") or work.get("citation_count") or work.get("citations"))
        if citations is not None:
            row["cited_by_count"] = citations
        for key in ("doi", "source", "source_display_name"):
            value = work.get(key)
            if value is not None and str(value).strip():
                row[key] = str(value).strip()
        rows.append(row)
        if len(rows) >= limit:
            break
    return rows


def _atlas_node_from_cluster(cluster: Mapping[str, Any], fallback_index: int) -> dict[str, Any]:
    level = _cluster_level(cluster)
    cluster_id = _cluster_id(cluster, fallback_index)
    cluster_uid = str(cluster.get("cluster_uid") or f"{level}:{cluster_id}")
    label = _cluster_label(cluster, cluster_id)
    doc_count, doc_count_source = _cluster_doc_count(cluster)
    node: dict[str, Any] = {
        "cluster_uid": cluster_uid,
        "level": level,
        "cluster_id": cluster_id,
        "label": label,
        "short_label": str(cluster.get("short_label") or _short_label(label)),
        "parent_uid": _cluster_parent_uid(cluster),
        "doc_count": doc_count,
        "doc_count_source": doc_count_source,
        "keyword_count": _coerce_int(cluster.get("n_keywords")) or len(cluster.get("keywords") or []),
        "child_count": _cluster_child_count(cluster),
        "keywords": _cluster_keywords(cluster),
        "badges": _cluster_badges(cluster),
        "representative_works": _cluster_representative_works(cluster),
        "representative_work_count": _coerce_int(cluster.get("representative_work_count")) or 0,
    }
    for coord in ("x", "y"):
        value = _coerce_float(cluster.get(coord))
        if value is not None:
            node[coord] = value
    return node


_LEVEL_ORDER = {
    "domain": 0,
    "macro": 1,
    "meso": 2,
    "micro": 3,
    "nano": 4,
    "cluster": 5,
}


def _atlas_cluster_key(value: Any) -> str:
    int_value = _coerce_int(value)
    if int_value is not None:
        return str(int_value)
    return str(value).strip()


def _membership_level_from_column(column: str) -> str:
    if column == "cluster":
        return "cluster"
    if column.startswith("cluster_"):
        return column.removeprefix("cluster_")
    return column


def _ordered_membership_cluster_columns(columns: list[str]) -> list[str]:
    indexed = list(enumerate(columns))
    return [
        column
        for _, column in sorted(
            indexed,
            key=lambda item: (
                _LEVEL_ORDER.get(_membership_level_from_column(item[1]).lower(), 100),
                item[0],
            ),
        )
    ]


def _artifact_warning(code: str, message: str, artifact: str, severity: str = "warning") -> dict[str, Any]:
    return asdict(ArtifactIssue(code, severity, message, artifact))


def _read_membership_for_atlas(
    membership_path: str | Path | None,
    warnings: list[dict[str, Any]],
) -> tuple[pd.DataFrame | None, list[str]]:
    if membership_path is None:
        return None, []
    path = Path(membership_path)
    if not path.exists():
        return None, []
    try:
        columns = list(pq.ParquetFile(path).schema_arrow.names)
    except Exception as exc:
        warnings.append(_artifact_warning("atlas_membership_schema_failed", str(exc), "membership"))
        return None, []
    cluster_cols = _ordered_membership_cluster_columns(_cluster_columns(columns))
    if "uid" not in columns or not cluster_cols:
        return None, []
    try:
        df = pd.read_parquet(path, columns=["uid", *cluster_cols])
    except Exception as exc:
        warnings.append(_artifact_warning("atlas_membership_read_failed", str(exc), "membership"))
        return None, []
    return df, cluster_cols


def _read_abstracts_for_atlas(
    abstracts_path: str | Path | None,
    warnings: list[dict[str, Any]],
) -> pd.DataFrame | None:
    if abstracts_path is None:
        return None
    path = Path(abstracts_path)
    if not path.exists():
        return None
    try:
        columns = list(pq.ParquetFile(path).schema_arrow.names)
    except Exception as exc:
        warnings.append(_artifact_warning("atlas_abstract_schema_failed", str(exc), "abstracts"))
        return None
    if "uid" not in columns:
        return None
    wanted = [
        "uid",
        "title",
        "display_title",
        "pubyear",
        "year",
        "publication_year",
        "cited_by_count",
        "citation_count",
        "citations",
        "doi",
        "source",
        "source_display_name",
    ]
    read_columns = [column for column in wanted if column in columns]
    try:
        return pd.read_parquet(path, columns=read_columns)
    except Exception as exc:
        warnings.append(_artifact_warning("atlas_abstract_read_failed", str(exc), "abstracts"))
        return None


def _atlas_level_columns(nodes: list[dict[str, Any]], cluster_cols: list[str]) -> dict[str, str]:
    if not cluster_cols:
        return {}
    level_by_col = {_membership_level_from_column(column): column for column in cluster_cols}
    node_levels = []
    for node in nodes:
        level = str(node.get("level") or "cluster")
        if level not in node_levels:
            node_levels.append(level)

    level_columns: dict[str, str] = {}
    for level in node_levels:
        if level in level_by_col:
            level_columns[level] = level_by_col[level]
        elif f"cluster_{level}" in cluster_cols:
            level_columns[level] = f"cluster_{level}"
        elif len(cluster_cols) == 1:
            level_columns[level] = cluster_cols[0]
    return level_columns


def _apply_membership_doc_counts(
    nodes: list[dict[str, Any]],
    membership_df: pd.DataFrame | None,
    level_columns: dict[str, str],
) -> None:
    if membership_df is None or not level_columns:
        return
    count_by_level: dict[str, dict[str, int]] = {}
    for level, column in level_columns.items():
        if column not in membership_df.columns:
            continue
        counts = membership_df[column].dropna().map(_atlas_cluster_key).value_counts()
        count_by_level[level] = {str(key): int(value) for key, value in counts.items()}

    for node in nodes:
        if node.get("doc_count") is not None:
            continue
        level = str(node.get("level") or "cluster")
        cluster_key = _atlas_cluster_key(node.get("cluster_id"))
        count = count_by_level.get(level, {}).get(cluster_key)
        if count is None:
            continue
        node["doc_count"] = count
        node["doc_count_source"] = f"membership:{level_columns[level]}"


def _normalize_legacy_report_nodes_to_membership_leaf(
    nodes: list[dict[str, Any]],
    membership_df: pd.DataFrame | None,
    cluster_cols: list[str],
) -> None:
    """Map legacy report `cluster:*` nodes to the finest membership level."""
    if membership_df is None or len(cluster_cols) < 2 or not nodes:
        return
    levels = {str(node.get("level") or "cluster") for node in nodes}
    if levels != {"cluster"}:
        return

    leaf_col = cluster_cols[-1]
    leaf_level = _membership_level_from_column(leaf_col)
    if leaf_level == "cluster" or leaf_col not in membership_df.columns:
        return

    report_keys = {_atlas_cluster_key(node.get("cluster_id")) for node in nodes}
    leaf_keys = set(membership_df[leaf_col].dropna().map(_atlas_cluster_key).tolist())
    if not report_keys or not report_keys <= leaf_keys:
        return

    for node in nodes:
        cluster_key = _atlas_cluster_key(node.get("cluster_id"))
        old_uid = str(node.get("cluster_uid") or "")
        node["level"] = leaf_level
        if not old_uid or old_uid == f"cluster:{cluster_key}":
            node["cluster_uid"] = f"{leaf_level}:{cluster_key}"
        node["level_source"] = f"membership:{leaf_col}"


def _add_membership_parent_nodes(
    nodes: list[dict[str, Any]],
    membership_df: pd.DataFrame | None,
    cluster_cols: list[str],
) -> None:
    if membership_df is None or len(cluster_cols) < 2:
        return
    existing = _node_uid_by_level_key(nodes)
    new_nodes: list[dict[str, Any]] = []
    for column in cluster_cols[:-1]:
        if column not in membership_df.columns:
            continue
        level = _membership_level_from_column(column)
        for raw_value in sorted(
            membership_df[column].dropna().map(_atlas_cluster_key).unique(),
            key=lambda value: (_coerce_int(value) is None, _coerce_int(value) if _coerce_int(value) is not None else str(value)),
        ):
            cluster_key = str(raw_value)
            if (level, cluster_key) in existing:
                continue
            cluster_id = _coerce_int(cluster_key)
            display_id: int | str = cluster_id if cluster_id is not None else cluster_key
            node = {
                "cluster_uid": f"{level}:{cluster_key}",
                "level": level,
                "cluster_id": display_id,
                "label": f"{level.title()} {cluster_key}",
                "short_label": f"{level.title()} {cluster_key}",
                "parent_uid": None,
                "doc_count": None,
                "doc_count_source": "unavailable",
                "keyword_count": 0,
                "child_count": 0,
                "keywords": [],
                "badges": [],
                "node_source": "membership_parent",
            }
            existing[(level, cluster_key)] = str(node["cluster_uid"])
            new_nodes.append(node)
    nodes.extend(new_nodes)


def _node_uid_by_level_key(nodes: list[dict[str, Any]]) -> dict[tuple[str, str], str]:
    return {
        (str(node.get("level") or "cluster"), _atlas_cluster_key(node.get("cluster_id"))): str(node["cluster_uid"])
        for node in nodes
    }


def _apply_membership_hierarchy(
    nodes: list[dict[str, Any]],
    membership_df: pd.DataFrame | None,
    cluster_cols: list[str],
) -> None:
    if membership_df is None or len(cluster_cols) < 2:
        return
    uid_by_key = _node_uid_by_level_key(nodes)
    node_by_uid = {str(node["cluster_uid"]): node for node in nodes}
    children_by_parent: dict[str, set[str]] = {}

    for parent_col, child_col in zip(cluster_cols, cluster_cols[1:]):
        if parent_col not in membership_df.columns or child_col not in membership_df.columns:
            continue
        parent_level = _membership_level_from_column(parent_col)
        child_level = _membership_level_from_column(child_col)
        pairs = membership_df[[parent_col, child_col]].dropna()
        if pairs.empty:
            continue
        pairs = pairs.assign(
            _parent_key=pairs[parent_col].map(_atlas_cluster_key),
            _child_key=pairs[child_col].map(_atlas_cluster_key),
        )
        for child_key, group in pairs.groupby("_child_key", sort=False):
            parent_key = str(group["_parent_key"].mode(dropna=True).iloc[0])
            child_uid = uid_by_key.get((child_level, str(child_key)))
            if child_uid is None:
                continue
            parent_uid = uid_by_key.get((parent_level, parent_key), f"{parent_level}:{parent_key}")
            child_node = node_by_uid.get(child_uid)
            if child_node is not None and not child_node.get("parent_uid"):
                child_node["parent_uid"] = parent_uid
            children_by_parent.setdefault(parent_uid, set()).add(child_uid)

    for parent_uid, children in children_by_parent.items():
        parent_node = node_by_uid.get(parent_uid)
        if parent_node is not None and not parent_node.get("child_count"):
            parent_node["child_count"] = len(children)


def _apply_atlas_lineage(nodes: list[dict[str, Any]]) -> None:
    node_by_uid = {str(node["cluster_uid"]): node for node in nodes}
    for node in nodes:
        current = node
        ancestors: list[dict[str, Any]] = []
        seen = {str(node["cluster_uid"])}
        while current.get("parent_uid"):
            parent_uid = str(current["parent_uid"])
            if parent_uid in seen:
                break
            seen.add(parent_uid)
            parent = node_by_uid.get(parent_uid)
            if parent is None:
                ancestors.append({"cluster_uid": parent_uid})
                break
            ancestors.append(
                {
                    "cluster_uid": str(parent["cluster_uid"]),
                    "level": parent.get("level"),
                    "cluster_id": parent.get("cluster_id"),
                    "label": parent.get("label"),
                    "short_label": parent.get("short_label"),
                }
            )
            current = parent
        node["lineage"] = list(reversed(ancestors)) + [
            {
                "cluster_uid": str(node["cluster_uid"]),
                "level": node.get("level"),
                "cluster_id": node.get("cluster_id"),
                "label": node.get("label"),
                "short_label": node.get("short_label"),
            }
        ]


def _edge_weight_column(columns: list[str]) -> str | None:
    for column in WEIGHT_COLUMN_CANDIDATES:
        if column in columns:
            return column
    for column in columns:
        if column.endswith("_weight"):
            return column
    return None


def _node_keyword_terms(node: Mapping[str, Any]) -> list[str]:
    terms: list[str] = []
    for keyword in node.get("keywords") or []:
        term = keyword.get("term") if isinstance(keyword, Mapping) else str(keyword)
        if term and str(term).strip():
            terms.append(str(term).strip())
    return terms


def _shared_keyword_terms(source: Mapping[str, Any], target: Mapping[str, Any], *, limit: int = 5) -> list[str]:
    target_terms = {term.lower(): term for term in _node_keyword_terms(target)}
    shared: list[str] = []
    for term in _node_keyword_terms(source):
        if term.lower() in target_terms and term not in shared:
            shared.append(term)
        if len(shared) >= limit:
            break
    return shared


def _cluster_edges_from_membership_edges(
    nodes: list[dict[str, Any]],
    membership_df: pd.DataFrame | None,
    edges_path: str | Path | None,
    level_columns: dict[str, str],
    warnings: list[dict[str, Any]],
    *,
    max_cluster_edges: int,
) -> list[dict[str, Any]]:
    if membership_df is None or edges_path is None or not level_columns:
        return []
    path = Path(edges_path)
    if not path.exists():
        return []
    try:
        edge_columns = list(pq.ParquetFile(path).schema_arrow.names)
    except Exception as exc:
        warnings.append(_artifact_warning("atlas_edge_schema_failed", str(exc), "edges"))
        return []
    if not REQUIRED_EDGE_COLUMNS <= set(edge_columns):
        return []
    weight_col = _edge_weight_column(edge_columns)
    read_columns = ["uid1", "uid2", *( [weight_col] if weight_col else [] )]
    try:
        edge_df = pd.read_parquet(path, columns=read_columns)
    except Exception as exc:
        warnings.append(_artifact_warning("atlas_edge_read_failed", str(exc), "edges"))
        return []
    if edge_df.empty:
        return []

    uid_by_level_key = _node_uid_by_level_key(nodes)
    node_by_uid = {str(node["cluster_uid"]): node for node in nodes}
    atlas_edges: list[dict[str, Any]] = []

    for level, column in level_columns.items():
        if column not in membership_df.columns:
            continue
        known_keys = {
            cluster_key: uid
            for (node_level, cluster_key), uid in uid_by_level_key.items()
            if node_level == level
        }
        if not known_keys:
            continue
        uid_to_cluster = dict(
            zip(
                membership_df["uid"].astype(str),
                membership_df[column].map(_atlas_cluster_key),
                strict=False,
            )
        )
        mapped = pd.DataFrame(
            {
                "_source_key": edge_df["uid1"].astype(str).map(uid_to_cluster),
                "_target_key": edge_df["uid2"].astype(str).map(uid_to_cluster),
            }
        )
        mapped = mapped.dropna()
        mapped = mapped[
            (mapped["_source_key"] != mapped["_target_key"])
            & mapped["_source_key"].isin(known_keys)
            & mapped["_target_key"].isin(known_keys)
        ]
        if mapped.empty:
            continue
        source_key = mapped["_source_key"].where(mapped["_source_key"] <= mapped["_target_key"], mapped["_target_key"])
        target_key = mapped["_target_key"].where(mapped["_source_key"] <= mapped["_target_key"], mapped["_source_key"])
        work = pd.DataFrame(
            {
                "source_uid": source_key.map(known_keys),
                "target_uid": target_key.map(known_keys),
                "weight": (
                    pd.to_numeric(edge_df.loc[mapped.index, weight_col], errors="coerce").fillna(1.0)
                    if weight_col
                    else 1.0
                ),
            }
        )
        grouped = (
            work.groupby(["source_uid", "target_uid"], sort=False)
            .agg(weight=("weight", "sum"), edge_count=("weight", "size"))
            .reset_index()
            .sort_values(["weight", "edge_count"], ascending=[False, False], kind="stable")
        )
        for row in grouped.head(max_cluster_edges).itertuples(index=False):
            source = node_by_uid.get(str(row.source_uid), {})
            target = node_by_uid.get(str(row.target_uid), {})
            same_parent = bool(source.get("parent_uid") and source.get("parent_uid") == target.get("parent_uid"))
            atlas_edges.append(
                {
                    "source_uid": str(row.source_uid),
                    "target_uid": str(row.target_uid),
                    "level": level,
                    "weight": float(row.weight),
                    "edge_count": int(row.edge_count),
                    "shared_terms": _shared_keyword_terms(source, target),
                    "same_parent": same_parent,
                    "relation_label": "same-parent" if same_parent else "cross-cluster",
                }
            )

    atlas_edges.sort(key=lambda edge: (float(edge["weight"]), int(edge["edge_count"])), reverse=True)
    return atlas_edges[:max_cluster_edges]


def _edge_relation_key(source_uid: Any, target_uid: Any) -> tuple[str, str] | None:
    source = str(source_uid or "").strip()
    target = str(target_uid or "").strip()
    if not source or not target:
        return None
    return tuple(sorted((source, target)))


def _first_mapping_value(row: Mapping[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        value = row.get(name)
        if value is None:
            continue
        if isinstance(value, float) and pd.isna(value):
            continue
        if str(value).strip():
            return value
    return None


def _edge_evidence_endpoint_pair(row: Mapping[str, Any]) -> tuple[str, str] | None:
    source = _first_mapping_value(
        row,
        (
            "source_uid",
            "source_cluster_uid",
            "source_cluster",
            "from_uid",
            "cluster_uid",
            "cluster",
        ),
    )
    target = _first_mapping_value(
        row,
        (
            "target_uid",
            "target_cluster_uid",
            "target_cluster",
            "to_uid",
            "neighbor_uid",
            "neighbor_cluster_uid",
            "neighbor_cluster",
        ),
    )
    if source is None or target is None:
        return None
    return str(source), str(target)


def _edge_evidence_rows_from_json(payload: Any) -> list[Mapping[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, Mapping)]
    if not isinstance(payload, Mapping):
        return []
    for key in ("samples", "edge_evidence", "relations", "edges", "neighbors"):
        rows = payload.get(key)
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, Mapping)]
    return []


def _normalize_edge_evidence_sample(row: Mapping[str, Any], rank: int) -> dict[str, Any]:
    sample: dict[str, Any] = {"rank": rank}
    field_map = {
        "source_work_uid": ("source_work_uid", "source_paper_uid", "source_document_uid", "uid1", "work_uid1", "paper_uid1"),
        "target_work_uid": ("target_work_uid", "target_paper_uid", "target_document_uid", "uid2", "work_uid2", "paper_uid2"),
        "source_title": ("source_title", "title1", "source_work_title", "source_paper_title"),
        "target_title": ("target_title", "title2", "target_work_title", "target_paper_title"),
        "edge_type": ("edge_type", "relation_type", "type", "layer"),
        "weight": ("weight", "score", "edge_weight", "rel_sum2"),
    }
    for output_key, names in field_map.items():
        value = _first_mapping_value(row, names)
        if value is not None:
            if output_key == "weight":
                numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
                if not pd.isna(numeric):
                    sample[output_key] = float(numeric)
                else:
                    sample[output_key] = str(value)
            else:
                sample[output_key] = str(value)
    return sample


def _read_edge_evidence_samples(
    edge_evidence_paths: tuple[str | Path, ...],
    warnings: list[dict[str, Any]],
    *,
    max_samples_per_edge: int,
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    samples_by_edge: dict[tuple[str, str], list[dict[str, Any]]] = {}
    if max_samples_per_edge <= 0:
        return samples_by_edge
    for raw_path in edge_evidence_paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        try:
            if path.suffix.lower() == ".json":
                rows = _edge_evidence_rows_from_json(json.loads(path.read_text(encoding="utf-8")))
            elif path.suffix.lower() == ".parquet":
                rows = pd.read_parquet(path).to_dict(orient="records")
            else:
                continue
        except Exception as exc:
            warnings.append(_artifact_warning("atlas_edge_evidence_read_failed", str(exc), "edge_evidence"))
            continue
        rank_by_edge: dict[tuple[str, str], int] = {}
        for row in rows:
            pair = _edge_evidence_endpoint_pair(row)
            if pair is None:
                continue
            key = _edge_relation_key(*pair)
            if key is None:
                continue
            raw_samples = row.get("samples") or row.get("sample_edges") or row.get("sample_works")
            if isinstance(raw_samples, list) and raw_samples:
                candidate_samples = [sample for sample in raw_samples if isinstance(sample, Mapping)]
            else:
                candidate_samples = [row]
            bucket = samples_by_edge.setdefault(key, [])
            for sample_row in candidate_samples:
                if len(bucket) >= max_samples_per_edge:
                    break
                rank_by_edge[key] = rank_by_edge.get(key, 0) + 1
                bucket.append(_normalize_edge_evidence_sample(sample_row, rank_by_edge[key]))
    return samples_by_edge


def _apply_edge_evidence_samples_to_edges(
    edges: list[dict[str, Any]],
    edge_evidence_paths: tuple[str | Path, ...],
    warnings: list[dict[str, Any]],
    *,
    max_samples_per_edge: int,
) -> None:
    samples_by_edge = _read_edge_evidence_samples(
        edge_evidence_paths,
        warnings,
        max_samples_per_edge=max_samples_per_edge,
    )
    if not samples_by_edge:
        return
    for edge in edges:
        key = _edge_relation_key(edge.get("source_uid"), edge.get("target_uid"))
        if key is None:
            continue
        samples = samples_by_edge.get(key, [])
        if not samples:
            continue
        edge["samples"] = samples[:max_samples_per_edge]
        edge["sample_count"] = len(samples)


def _title_lookup_from_abstracts(abstracts_df: pd.DataFrame | None) -> dict[str, dict[str, Any]]:
    if abstracts_df is None or "uid" not in abstracts_df.columns:
        return {}
    lookup: dict[str, dict[str, Any]] = {}
    for row in abstracts_df.to_dict(orient="records"):
        uid = str(row.get("uid") or "")
        if not uid:
            continue
        lookup[uid] = row
    return lookup


def _edge_evidence_work_sample(
    row: pd.Series,
    *,
    source_uid: str,
    target_uid: str,
    source_work: Mapping[str, Any] | None,
    target_work: Mapping[str, Any] | None,
    weight_col: str | None,
    rank: int,
) -> dict[str, Any]:
    sample: dict[str, Any] = {
        "rank": rank,
        "source_work_uid": source_uid,
        "target_work_uid": target_uid,
    }
    if source_work:
        title = _first_mapping_value(source_work, ("title", "display_title"))
        if title is not None:
            sample["source_title"] = _plain_atlas_text(title)
    if target_work:
        title = _first_mapping_value(target_work, ("title", "display_title"))
        if title is not None:
            sample["target_title"] = _plain_atlas_text(title)
    if weight_col and weight_col in row:
        weight = pd.to_numeric(pd.Series([row[weight_col]]), errors="coerce").iloc[0]
        if not pd.isna(weight):
            sample["weight"] = float(weight)
    return sample


def write_edge_evidence_samples(
    *,
    edges_path: str | Path,
    membership_path: str | Path,
    abstracts_path: str | Path | None,
    output_path: str | Path,
    max_relations: int = 300,
    max_samples_per_relation: int = 3,
) -> Path | None:
    """Write bounded work-pair samples for aggregate Atlas neighbor relations.

    The output is intentionally a review-scale sidecar. It samples the strongest
    paper edges for each aggregate cluster relation and keeps enough provenance
    for the web inspector without copying the full edge table.
    """

    output = Path(output_path)
    if max_relations <= 0 or max_samples_per_relation <= 0:
        return None
    warnings: list[dict[str, Any]] = []
    membership_df, cluster_cols = _read_membership_for_atlas(membership_path, warnings)
    if membership_df is None or not cluster_cols:
        return None
    edge_path = Path(edges_path)
    if not edge_path.exists():
        return None
    try:
        edge_columns = list(pq.ParquetFile(edge_path).schema_arrow.names)
    except Exception:
        return None
    if not REQUIRED_EDGE_COLUMNS <= set(edge_columns):
        return None
    weight_col = _edge_weight_column(edge_columns)
    read_columns = ["uid1", "uid2", *( [weight_col] if weight_col else [] )]
    try:
        edge_df = pd.read_parquet(edge_path, columns=read_columns)
    except Exception:
        return None
    if edge_df.empty:
        return None

    abstracts_df = _read_abstracts_for_atlas(abstracts_path, warnings)
    work_lookup = _title_lookup_from_abstracts(abstracts_df)
    relations: list[dict[str, Any]] = []

    for column in cluster_cols:
        if column not in membership_df.columns:
            continue
        level = "cluster" if len(cluster_cols) == 1 else _membership_level_from_column(column)
        uid_to_cluster = dict(
            zip(
                membership_df["uid"].astype(str),
                membership_df[column].map(_atlas_cluster_key),
                strict=False,
            )
        )
        mapped = pd.DataFrame(
            {
                "_source_key": edge_df["uid1"].astype(str).map(uid_to_cluster),
                "_target_key": edge_df["uid2"].astype(str).map(uid_to_cluster),
                "_uid1": edge_df["uid1"].astype(str),
                "_uid2": edge_df["uid2"].astype(str),
                "_weight": (
                    pd.to_numeric(edge_df[weight_col], errors="coerce").fillna(1.0)
                    if weight_col
                    else 1.0
                ),
            }
        )
        mapped = mapped.dropna(subset=["_source_key", "_target_key"])
        mapped = mapped[mapped["_source_key"] != mapped["_target_key"]]
        if mapped.empty:
            continue
        mapped["_a_key"] = mapped["_source_key"].where(
            mapped["_source_key"] <= mapped["_target_key"],
            mapped["_target_key"],
        )
        mapped["_b_key"] = mapped["_target_key"].where(
            mapped["_source_key"] <= mapped["_target_key"],
            mapped["_source_key"],
        )
        mapped["_source_uid"] = level + ":" + mapped["_a_key"].astype(str)
        mapped["_target_uid"] = level + ":" + mapped["_b_key"].astype(str)

        grouped = (
            mapped.groupby(["_source_uid", "_target_uid"], sort=False)
            .agg(weight=("_weight", "sum"), edge_count=("_weight", "size"))
            .reset_index()
            .sort_values(["weight", "edge_count"], ascending=[False, False], kind="stable")
        )
        for _, rel in grouped.head(max_relations).iterrows():
            rows = mapped[
                (mapped["_source_uid"] == rel["_source_uid"])
                & (mapped["_target_uid"] == rel["_target_uid"])
            ].sort_values("_weight", ascending=False, kind="stable")
            samples = []
            for rank, (_, edge_row) in enumerate(rows.head(max_samples_per_relation).iterrows(), start=1):
                samples.append(
                    _edge_evidence_work_sample(
                        edge_row,
                        source_uid=str(edge_row["_uid1"]),
                        target_uid=str(edge_row["_uid2"]),
                        source_work=work_lookup.get(str(edge_row["_uid1"])),
                        target_work=work_lookup.get(str(edge_row["_uid2"])),
                        weight_col="_weight",
                        rank=rank,
                    )
                )
            if samples:
                relations.append(
                    {
                        "source_uid": str(rel["_source_uid"]),
                        "target_uid": str(rel["_target_uid"]),
                        "level": level,
                        "weight": float(rel["weight"]),
                        "edge_count": int(rel["edge_count"]),
                        "samples": samples,
                    }
                )

    if not relations:
        return None
    relations.sort(key=lambda row: (float(row.get("weight") or 0), int(row.get("edge_count") or 0)), reverse=True)
    payload = {
        "schema_version": EDGE_EVIDENCE_SCHEMA_VERSION,
        "created_at_utc": _utc_now(),
        "max_samples_per_relation": int(max_samples_per_relation),
        "relations": relations[:max_relations],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output


def _apply_atlas_neighbors(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    *,
    max_neighbors_per_node: int,
) -> None:
    node_by_uid = {str(node["cluster_uid"]): node for node in nodes}
    neighbors: dict[str, list[dict[str, Any]]] = {uid: [] for uid in node_by_uid}
    for edge in edges:
        source_uid = str(edge["source_uid"])
        target_uid = str(edge["target_uid"])
        for uid, other_uid in ((source_uid, target_uid), (target_uid, source_uid)):
            other = node_by_uid.get(other_uid, {})
            neighbors.setdefault(uid, []).append(
                {
                    "cluster_uid": other_uid,
                    "label": other.get("label"),
                    "short_label": other.get("short_label"),
                    "level": other.get("level"),
                    "weight": edge.get("weight"),
                    "edge_count": edge.get("edge_count"),
                    "shared_terms": edge.get("shared_terms", []),
                    "same_parent": edge.get("same_parent", False),
                    "relation_label": edge.get("relation_label", "cross-cluster"),
                    "sample_count": edge.get("sample_count", 0),
                    "samples": edge.get("samples", []),
                }
            )
    for uid, node in node_by_uid.items():
        rows = sorted(
            neighbors.get(uid, []),
            key=lambda row: (float(row.get("weight") or 0), int(row.get("edge_count") or 0)),
            reverse=True,
        )
        node["neighbor_count"] = len(rows)
        node["neighbors"] = rows[:max_neighbors_per_node]


def _first_nonempty(row: pd.Series, columns: tuple[str, ...]) -> Any:
    for column in columns:
        if column not in row:
            continue
        value = row[column]
        if value is not None and not pd.isna(value) and str(value).strip():
            return value
    return None


def _atlas_work_from_row(row: pd.Series, rank: int) -> dict[str, Any] | None:
    uid = _first_nonempty(row, ("uid",))
    title = _first_nonempty(row, ("title", "display_title"))
    if title is None and uid is None:
        return None
    work: dict[str, Any] = {"rank": rank}
    if uid is not None:
        work["uid"] = str(uid)
    if title is not None:
        work["title"] = _plain_atlas_text(title)
    year = _coerce_int(_first_nonempty(row, ("pubyear", "year", "publication_year")))
    if year is not None:
        work["year"] = year
    citations = _coerce_int(_first_nonempty(row, ("cited_by_count", "citation_count", "citations")))
    if citations is not None:
        work["cited_by_count"] = citations
    for source_key in ("doi", "source", "source_display_name"):
        value = _first_nonempty(row, (source_key,))
        if value is not None:
            work[source_key] = str(value)
    return work


def _apply_atlas_representative_works(
    nodes: list[dict[str, Any]],
    membership_df: pd.DataFrame | None,
    abstracts_df: pd.DataFrame | None,
    level_columns: dict[str, str],
    *,
    max_representative_works: int,
) -> None:
    if membership_df is None or abstracts_df is None or not level_columns or max_representative_works <= 0:
        return
    if "uid" not in membership_df.columns or "uid" not in abstracts_df.columns:
        return

    node_by_level_key = {
        (str(node.get("level") or "cluster"), _atlas_cluster_key(node.get("cluster_id"))): node
        for node in nodes
    }
    if not node_by_level_key:
        return

    abstracts = abstracts_df.copy()
    abstracts["uid"] = abstracts["uid"].astype(str)
    membership_uids = membership_df[["uid", *[column for column in level_columns.values() if column in membership_df.columns]]].copy()
    membership_uids["uid"] = membership_uids["uid"].astype(str)
    joined = membership_uids.merge(abstracts, on="uid", how="inner")
    if joined.empty:
        return

    sort_columns: list[str] = []
    ascending: list[bool] = []
    for column in ("cited_by_count", "citation_count", "citations"):
        if column in joined.columns:
            joined[f"_{column}_sort"] = pd.to_numeric(joined[column], errors="coerce").fillna(-1)
            sort_columns.append(f"_{column}_sort")
            ascending.append(False)
            break
    for column in ("pubyear", "year", "publication_year"):
        if column in joined.columns:
            joined[f"_{column}_sort"] = pd.to_numeric(joined[column], errors="coerce").fillna(-1)
            sort_columns.append(f"_{column}_sort")
            ascending.append(False)
            break
    if "title" in joined.columns:
        sort_columns.append("title")
        ascending.append(True)

    for level, column in level_columns.items():
        if column not in joined.columns:
            continue
        work = joined.copy()
        work["_cluster_key"] = work[column].map(_atlas_cluster_key)
        if sort_columns:
            work = work.sort_values(sort_columns, ascending=ascending, kind="stable")
        for cluster_key, group in work.groupby("_cluster_key", sort=False):
            node = node_by_level_key.get((level, str(cluster_key)))
            if node is None:
                continue
            node["representative_work_count"] = int(len(group))
            if node.get("representative_works"):
                continue
            rows: list[dict[str, Any]] = []
            for rank, (_, row) in enumerate(group.head(max_representative_works).iterrows(), start=1):
                item = _atlas_work_from_row(row, rank)
                if item is not None:
                    rows.append(item)
            node["representative_works"] = rows
            if rows:
                node["representative_works_source"] = f"membership:{column}+abstracts"


def build_atlas_payload_from_report_data(
    report_data: dict[str, Any],
    *,
    membership_path: str | Path | None = None,
    edges_path: str | Path | None = None,
    abstracts_path: str | Path | None = None,
    edge_evidence_paths: tuple[str | Path, ...] = (),
    max_cluster_edges: int = 300,
    max_neighbors_per_node: int = 8,
    max_representative_works: int = 3,
    max_edge_evidence_samples: int = 3,
) -> dict[str, Any]:
    """Build the minimal Atlas Map payload embedded under ``_sciscape``."""

    clusters = _report_clusters(report_data)
    nodes = [_atlas_node_from_cluster(cluster, index) for index, cluster in enumerate(clusters)]
    warnings: list[dict[str, Any]] = []
    membership_df, cluster_cols = _read_membership_for_atlas(membership_path, warnings)
    abstracts_df = _read_abstracts_for_atlas(abstracts_path, warnings)
    _normalize_legacy_report_nodes_to_membership_leaf(nodes, membership_df, cluster_cols)
    _add_membership_parent_nodes(nodes, membership_df, cluster_cols)
    level_columns = _atlas_level_columns(nodes, cluster_cols)
    _apply_membership_doc_counts(nodes, membership_df, level_columns)
    _apply_membership_hierarchy(nodes, membership_df, cluster_cols)
    _apply_atlas_lineage(nodes)
    _apply_atlas_representative_works(
        nodes,
        membership_df,
        abstracts_df,
        level_columns,
        max_representative_works=max(0, max_representative_works),
    )
    edges = _cluster_edges_from_membership_edges(
        nodes,
        membership_df,
        edges_path,
        level_columns,
        warnings,
        max_cluster_edges=max(0, max_cluster_edges),
    )
    _apply_edge_evidence_samples_to_edges(
        edges,
        tuple(edge_evidence_paths),
        warnings,
        max_samples_per_edge=max(0, max_edge_evidence_samples),
    )
    _apply_atlas_neighbors(nodes, edges, max_neighbors_per_node=max(0, max_neighbors_per_node))
    levels: list[str] = []
    for node in nodes:
        level = str(node["level"])
        if level not in levels:
            levels.append(level)
    levels.sort(key=lambda level: _LEVEL_ORDER.get(level.lower(), 100))
    missing_doc_counts = sum(1 for node in nodes if node.get("doc_count") is None)
    if missing_doc_counts:
        warnings.append(
            {
                "code": "missing_doc_count",
                "severity": "info",
                "message": f"{missing_doc_counts} cluster nodes do not have document counts.",
            }
        )
    return {
        "schema_version": ATLAS_PAYLOAD_SCHEMA_VERSION,
        "levels": levels,
        "nodes": nodes,
        "node_count": len(nodes),
        "edges": edges,
        "edge_count": len(edges),
        "warnings": warnings,
    }


def _atlas_render_level_index(level: Any, levels: list[str]) -> int:
    level_name = str(level or "cluster")
    if level_name in levels:
        return levels.index(level_name)
    return _LEVEL_ORDER.get(level_name.lower(), len(levels))


def _atlas_render_node_radius(node: Mapping[str, Any]) -> float:
    doc_count = _coerce_float(node.get("doc_count"))
    if doc_count is None or doc_count < 0:
        return 7.0
    return round(max(5.0, min(36.0, 5.0 + math.sqrt(doc_count) * 2.2)), 3)


def _atlas_render_label_priority(node: Mapping[str, Any]) -> float:
    doc_count = _coerce_float(node.get("doc_count")) or 0.0
    child_count = _coerce_float(node.get("child_count")) or 0.0
    keyword_count = _coerce_float(node.get("keyword_count")) or 0.0
    return round(math.log1p(max(0.0, doc_count)) * 10.0 + child_count * 2.0 + keyword_count, 3)


def _atlas_render_bounds(positions: Mapping[str, list[float]]) -> dict[str, float | None]:
    if not positions:
        return {"min_x": None, "max_x": None, "min_y": None, "max_y": None}
    xs = [pos[0] for pos in positions.values()]
    ys = [pos[1] for pos in positions.values()]
    return {
        "min_x": round(min(xs), 6),
        "max_x": round(max(xs), 6),
        "min_y": round(min(ys), 6),
        "max_y": round(max(ys), 6),
    }


def _atlas_render_positions(
    nodes: list[Mapping[str, Any]],
    levels: list[str],
) -> tuple[dict[str, list[float]], dict[str, str], list[dict[str, Any]]]:
    positions: dict[str, list[float]] = {}
    sources: dict[str, str] = {}
    warnings: list[dict[str, Any]] = []
    nodes_by_level: dict[str, list[Mapping[str, Any]]] = {}
    for node in nodes:
        uid = str(node.get("cluster_uid") or "").strip()
        if not uid:
            continue
        x = _coerce_float(node.get("x"))
        y = _coerce_float(node.get("y"))
        if x is not None and y is not None:
            positions[uid] = [round(x, 6), round(y, 6)]
            sources[uid] = "node_coordinates"
            continue
        nodes_by_level.setdefault(str(node.get("level") or "cluster"), []).append(node)

    generated = 0
    for level in sorted(nodes_by_level, key=lambda item: _atlas_render_level_index(item, levels)):
        level_nodes = nodes_by_level[level]
        level_index = _atlas_render_level_index(level, levels)
        total = max(1, len(level_nodes))
        for local_index, node in enumerate(level_nodes):
            uid = str(node.get("cluster_uid") or "").strip()
            if not uid or uid in positions:
                continue
            parent_uid = str(node.get("parent_uid") or "").strip()
            parent_position = positions.get(parent_uid)
            angle = (local_index / total) * math.tau + level_index * 0.41
            if parent_position:
                radius = 26.0 + level_index * 8.0 + (local_index % 5) * 2.5
                x = parent_position[0] + math.cos(angle) * radius
                y = parent_position[1] + math.sin(angle) * radius
                source = "generated_parent_radial"
            else:
                radius = 90.0 * (level_index + 1) + (local_index % 7) * 4.0
                x = math.cos(angle) * radius
                y = math.sin(angle) * radius
                source = "generated_radial"
            positions[uid] = [round(x, 6), round(y, 6)]
            sources[uid] = source
            generated += 1

    if generated:
        warnings.append(
            {
                "code": "generated_atlas_render_coordinates",
                "severity": "info",
                "message": (
                    f"{generated} atlas nodes lacked x/y coordinates and received deterministic "
                    "fallback positions for renderer smoke use."
                ),
            }
        )
    return positions, sources, warnings


def build_atlas_render_payload(
    atlas_payload: Mapping[str, Any],
    *,
    engine_family: str = "deck.gl",
) -> dict[str, Any]:
    """Build a renderer-oriented Atlas payload from a semantic Atlas payload.

    The semantic Atlas payload keeps cluster evidence and lineage. This render
    payload is intentionally narrower: stable ids, positions, layer rows, and
    deck.gl-friendly metadata that can also be adapted by other engines.
    """

    nodes = [node for node in atlas_payload.get("nodes", []) if isinstance(node, Mapping)]
    edges = [edge for edge in atlas_payload.get("edges", []) if isinstance(edge, Mapping)]
    levels = [str(level) for level in atlas_payload.get("levels", []) if str(level).strip()]
    if not levels:
        for node in nodes:
            level = str(node.get("level") or "cluster")
            if level not in levels:
                levels.append(level)
        levels.sort(key=lambda level: _LEVEL_ORDER.get(level.lower(), 100))

    positions, coordinate_sources, render_warnings = _atlas_render_positions(nodes, levels)
    node_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    hierarchy_rows: list[dict[str, Any]] = []

    for node in nodes:
        uid = str(node.get("cluster_uid") or "").strip()
        position = positions.get(uid)
        if not uid or not position:
            continue
        level = str(node.get("level") or "cluster")
        level_index = _atlas_render_level_index(level, levels)
        doc_count = _coerce_int(node.get("doc_count"))
        label = str(node.get("label") or node.get("short_label") or uid)
        short_label = str(node.get("short_label") or _short_label(label))
        parent_uid = str(node.get("parent_uid") or "").strip() or None
        row = {
            "id": uid,
            "cluster_uid": uid,
            "cluster_id": node.get("cluster_id"),
            "level": level,
            "level_index": level_index,
            "parent_uid": parent_uid,
            "label": label,
            "short_label": short_label,
            "position": position,
            "x": position[0],
            "y": position[1],
            "coordinate_source": coordinate_sources.get(uid, "unknown"),
            "doc_count": doc_count,
            "doc_count_log": round(math.log1p(max(0, doc_count or 0)), 6),
            "keyword_count": _coerce_int(node.get("keyword_count")) or 0,
            "child_count": _coerce_int(node.get("child_count")) or 0,
            "neighbor_count": _coerce_int(node.get("neighbor_count")) or 0,
            "representative_work_count": _coerce_int(node.get("representative_work_count")) or 0,
            "badge_count": len(node.get("badges") or []),
            "render_radius": _atlas_render_node_radius(node),
            "label_priority": _atlas_render_label_priority(node),
            "color_key": parent_uid or level,
            "pickable": True,
        }
        node_rows.append(row)
        label_rows.append(
            {
                "id": f"label:{uid}",
                "cluster_uid": uid,
                "text": short_label,
                "position": position,
                "level": level,
                "level_index": level_index,
                "priority": row["label_priority"],
                "min_zoom": max(-4, level_index - 2),
            }
        )
        if parent_uid and parent_uid in positions:
            hierarchy_rows.append(
                {
                    "id": f"hierarchy:{parent_uid}->{uid}",
                    "source_uid": parent_uid,
                    "target_uid": uid,
                    "source_position": positions[parent_uid],
                    "target_position": position,
                    "relation": "parent-child",
                    "level": level,
                }
            )

    edge_rows: list[dict[str, Any]] = []
    for index, edge in enumerate(edges):
        source_uid = str(edge.get("source_uid") or edge.get("source") or "").strip()
        target_uid = str(edge.get("target_uid") or edge.get("target") or "").strip()
        source_position = positions.get(source_uid)
        target_position = positions.get(target_uid)
        if not source_uid or not target_uid or source_position is None or target_position is None:
            continue
        weight = _coerce_float(edge.get("weight")) or 0.0
        edge_rows.append(
            {
                "id": str(edge.get("edge_uid") or f"edge:{source_uid}->{target_uid}:{index}"),
                "source_uid": source_uid,
                "target_uid": target_uid,
                "source_position": source_position,
                "target_position": target_position,
                "level": str(edge.get("level") or "cluster"),
                "weight": weight,
                "edge_count": _coerce_int(edge.get("edge_count")) or 0,
                "render_width": round(max(1.0, min(10.0, 1.0 + math.log1p(max(0.0, weight)))), 3),
                "relation_label": str(edge.get("relation_label") or "relation"),
                "same_parent": bool(edge.get("same_parent")),
                "shared_terms": list(edge.get("shared_terms") or [])[:8],
                "sample_count": _coerce_int(edge.get("sample_count")) or 0,
                "pickable": True,
            }
        )

    coordinate_source_values = set(coordinate_sources.values())
    if not coordinate_source_values:
        coordinate_source = "none"
    elif coordinate_source_values == {"node_coordinates"}:
        coordinate_source = "node_coordinates"
    elif "node_coordinates" in coordinate_source_values:
        coordinate_source = "mixed"
    else:
        coordinate_source = "generated"

    atlas_warnings = list(atlas_payload.get("warnings") or [])
    return {
        "schema_version": ATLAS_RENDER_PAYLOAD_SCHEMA_VERSION,
        "source_schema_version": atlas_payload.get("schema_version"),
        "engine_family": engine_family,
        "view": {
            "type": "OrthographicView",
            "coordinate_system": "cartesian_2d",
            "coordinate_source": coordinate_source,
            "bounds": _atlas_render_bounds(positions),
        },
        "levels": levels,
        "layers": {
            "nodes": {
                "layer_id": "atlas-clusters",
                "recommended_deck_layer": "ScatterplotLayer",
                "rows": node_rows,
            },
            "edges": {
                "layer_id": "atlas-relations",
                "recommended_deck_layer": "LineLayer",
                "rows": edge_rows,
            },
            "labels": {
                "layer_id": "atlas-labels",
                "recommended_deck_layer": "TextLayer",
                "rows": label_rows,
            },
            "hierarchy": {
                "layer_id": "atlas-hierarchy",
                "recommended_deck_layer": "LineLayer",
                "rows": hierarchy_rows,
            },
        },
        "node_count": len(node_rows),
        "edge_count": len(edge_rows),
        "label_count": len(label_rows),
        "hierarchy_edge_count": len(hierarchy_rows),
        "warnings": [*atlas_warnings, *render_warnings],
    }


def _report_has_terms(clusters: list[dict[str, Any]]) -> bool:
    return any(bool(cluster.get("keywords")) for cluster in clusters)


def _report_term_edge_count(clusters: list[dict[str, Any]]) -> int:
    total = 0
    for cluster in clusters:
        network_edges = cluster.get("network_edges")
        cooc_rows = cluster.get("cooccurrence_table")
        if isinstance(network_edges, list):
            total += len(network_edges)
        if isinstance(cooc_rows, list):
            total += len(cooc_rows)
    return total


def _cooccurrence_weight(row: Mapping[str, Any]) -> float:
    for key in ("weight", "cooccurrence_weight", "count", "cooccurrence_count"):
        value = _coerce_float(row.get(key))
        if value is not None:
            return value
    return 1.0


def _cooccurrence_count(row: Mapping[str, Any]) -> int | None:
    for key in ("count", "cooccurrence_count"):
        value = _coerce_int(row.get(key))
        if value is not None:
            return value
    return None


def _cooccurrence_rows_from_report_data(report_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract a stable row-level co-occurrence table from report data."""

    rows: list[dict[str, Any]] = []
    for cluster_index, cluster in enumerate(_report_clusters(report_data)):
        table = cluster.get("cooccurrence_table")
        if not isinstance(table, list):
            continue
        cluster_level = _cluster_level(cluster)
        cluster_id = str(_cluster_id(cluster, cluster_index))
        cluster_uid = str(cluster.get("cluster_uid") or f"{cluster_level}:{cluster_id}")
        for row_index, item in enumerate(table, start=1):
            if not isinstance(item, Mapping):
                continue
            source = str(item.get("source") or "").strip()
            target = str(item.get("target") or "").strip()
            if not source or not target:
                continue
            rows.append(
                {
                    "schema_version": COOCCURRENCE_ARTIFACT_SCHEMA_VERSION,
                    "cluster_uid": cluster_uid,
                    "cluster_level": cluster_level,
                    "cluster_id": cluster_id,
                    "row_rank": int(row_index),
                    "source": source,
                    "target": target,
                    "weight": float(_cooccurrence_weight(item)),
                    "count": _cooccurrence_count(item),
                    "source_tier": str(item.get("source_tier") or ""),
                    "target_tier": str(item.get("target_tier") or ""),
                    "relation": str(item.get("relation") or "cooccurrence"),
                    "primary_term": str(item.get("primary_term") or ""),
                    "support_term": str(item.get("support_term") or ""),
                    "support_tier": str(item.get("support_tier") or ""),
                }
            )

    rows.sort(
        key=lambda row: (
            str(row["cluster_level"]),
            str(row["cluster_id"]),
            -float(row["weight"]),
            str(row["source"]),
            str(row["target"]),
        )
    )
    return rows


def _keyword_label_column(columns: list[str]) -> str:
    for column in ("display_label", "label", "term", "keyword"):
        if column in columns:
            return column
    return columns[0]


def _keyword_score_column(columns: list[str]) -> str | None:
    for column in ("representative_score", "quality_score", "score", "tfidf", "frequency"):
        if column in columns:
            return column
    return None


def _cooccurrence_rows_from_keywords(
    keywords_path: Path,
    *,
    top_k_per_cluster: int = 10,
) -> list[dict[str, Any]]:
    """Build a bounded co-occurrence table from top keywords per cluster."""

    keywords = pd.read_parquet(keywords_path)
    if keywords.empty:
        return []
    columns = list(keywords.columns)
    cluster_col = next((column for column in columns if "cluster" in column.lower()), columns[0])
    label_col = _keyword_label_column(columns)
    score_col = _keyword_score_column(columns)
    tier_col = "keyword_label_tier" if "keyword_label_tier" in columns else None

    rows: list[dict[str, Any]] = []
    for cluster_id, group in keywords.groupby(cluster_col, sort=True):
        work = group.copy()
        if score_col is not None:
            work = work.sort_values(score_col, ascending=False, kind="mergesort")
        terms: list[dict[str, Any]] = []
        seen: set[str] = set()
        for _, item in work.iterrows():
            term = str(item[label_col] or "").strip()
            if not term or term in seen:
                continue
            seen.add(term)
            score = _coerce_float(item.get(score_col)) if score_col is not None else None
            terms.append(
                {
                    "term": term,
                    "score": score if score is not None else 1.0,
                    "tier": str(item.get(tier_col) or "") if tier_col is not None else "",
                }
            )
            if len(terms) >= max(1, int(top_k_per_cluster)):
                break
        cluster_id_str = str(cluster_id)
        cluster_uid = f"cluster:{cluster_id_str}"
        for row_rank, left_index in enumerate(range(len(terms)), start=1):
            left = terms[left_index]
            for right in terms[left_index + 1 :]:
                weight = (float(left["score"]) + float(right["score"])) / 2.0
                rows.append(
                    {
                        "schema_version": COOCCURRENCE_ARTIFACT_SCHEMA_VERSION,
                        "cluster_uid": cluster_uid,
                        "cluster_level": "cluster",
                        "cluster_id": cluster_id_str,
                        "row_rank": int(row_rank),
                        "source": left["term"],
                        "target": right["term"],
                        "weight": float(weight),
                        "count": 1,
                        "source_tier": left["tier"],
                        "target_tier": right["tier"],
                        "relation": "within_cluster_keyword_pair",
                        "primary_term": left["term"],
                        "support_term": right["term"],
                        "support_tier": right["tier"],
                    }
                )

    rows.sort(
        key=lambda row: (
            str(row["cluster_level"]),
            str(row["cluster_id"]),
            -float(row["weight"]),
            str(row["source"]),
            str(row["target"]),
        )
    )
    return rows


def _cooccurrence_map_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    term_map: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        source = str(row["source"])
        target = str(row["target"])
        for term, other in ((source, target), (target, source)):
            term_map.setdefault(term, []).append(
                {
                    "term": other,
                    "cluster_uid": row["cluster_uid"],
                    "cluster_id": row["cluster_id"],
                    "cluster_level": row["cluster_level"],
                    "weight": row["weight"],
                    "count": row.get("count"),
                    "relation": row["relation"],
                    "source": source,
                    "target": target,
                    "primary_term": row.get("primary_term", ""),
                    "support_term": row.get("support_term", ""),
                    "support_tier": row.get("support_tier", ""),
                }
            )

    for term, values in term_map.items():
        term_map[term] = sorted(
            values,
            key=lambda item: (
                -float(item.get("weight") or 0.0),
                str(item.get("cluster_uid") or ""),
                str(item.get("term") or ""),
            ),
        )
    return {
        "schema_version": COOCCURRENCE_ARTIFACT_SCHEMA_VERSION,
        "edge_count": int(len(rows)),
        "term_count": int(len(term_map)),
        "cluster_count": int(len({row["cluster_uid"] for row in rows})),
        "terms": term_map,
    }


def write_cooccurrence_artifacts(
    path: str | Path,
    *,
    output_dir: str | Path | None = None,
    table_filename: str = "term_cooccurrence.parquet",
    map_filename: str = "term_cooccurrence_map.json",
) -> dict[str, Any] | None:
    """Write stable co-occurrence table/map sidecars from report or keywords.

    Returns ``None`` when neither report data nor keyword rows can produce
    co-occurrence rows.
    """

    artifacts = infer_result_artifacts(path)
    rows: list[dict[str, Any]] = []
    if artifacts.report_data_path is not None:
        report_data = json.loads(artifacts.report_data_path.read_text(encoding="utf-8"))
        if isinstance(report_data, dict):
            rows = _cooccurrence_rows_from_report_data(report_data)
    if not rows and artifacts.keywords_path is not None:
        rows = _cooccurrence_rows_from_keywords(artifacts.keywords_path)
    if not rows:
        return None

    target_dir = Path(output_dir) if output_dir is not None else artifacts.landscape_dir or artifacts.result_root
    target_dir.mkdir(parents=True, exist_ok=True)
    table_path = target_dir / table_filename
    map_path = target_dir / map_filename
    pd.DataFrame(rows).to_parquet(table_path, index=False)
    map_payload = _cooccurrence_map_from_rows(rows)
    map_payload["source_report_data"] = _rel(artifacts.report_data_path, artifacts.result_root)
    map_payload["source_table"] = _rel(table_path, artifacts.result_root)
    map_path.write_text(json.dumps(map_payload, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "schema_version": COOCCURRENCE_ARTIFACT_SCHEMA_VERSION,
        "table_path": table_path,
        "map_path": map_path,
        "rows": int(len(rows)),
        "terms": int(map_payload["term_count"]),
        "clusters": int(map_payload["cluster_count"]),
    }


def _review_packet_issue(
    code: str,
    severity: str,
    message: str,
    *,
    artifact: str | None = None,
) -> dict[str, Any]:
    issue = {"code": code, "severity": severity, "message": message}
    if artifact:
        issue["artifact"] = artifact
    return issue


def _cluster_review_packet_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.exists():
        candidate = candidate.resolve()
    if candidate.is_file():
        return candidate
    direct = candidate / "cluster_review_packet.json"
    if direct.exists() or candidate.name == "review":
        return direct
    review_path = candidate / "review" / "cluster_review_packet.json"
    if review_path.exists() or (candidate / "review").is_dir():
        return review_path
    return direct


def _review_packet_qa_path(packet_path: Path) -> Path:
    return packet_path.parent / "cluster_review_packet_qa.json"


def _review_packet_result_root(packet_path: Path) -> Path:
    if packet_path.parent.name == "review":
        return packet_path.parent.parent
    return packet_path.parent


def _review_packet_portable_rel(path: Path, *, result_root: Path, packet_path: Path) -> str:
    rel = _rel(path, result_root)
    if rel and not Path(rel).is_absolute():
        return rel
    try:
        return path.relative_to(packet_path.parent).as_posix()
    except ValueError:
        return path.name


def _review_packet_declared_qa_path(packet_path: Path, payload: Mapping[str, Any]) -> tuple[Path, str | None, bool]:
    qa_ref = None
    qa = payload.get("qa")
    if isinstance(qa, Mapping):
        raw = str(qa.get("path") or "").strip()
        qa_ref = raw or None
    if not qa_ref:
        return _review_packet_qa_path(packet_path), None, False
    ref_path = Path(qa_ref)
    if ref_path.is_absolute():
        return ref_path, qa_ref, True
    result_root = _review_packet_result_root(packet_path)
    result_relative = result_root / ref_path
    if result_relative.exists():
        return result_relative, qa_ref, False
    return packet_path.parent / ref_path, qa_ref, False


def _review_packet_source_artifacts(artifacts: ResultArtifacts) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    root = artifacts.result_root
    for role, path, required in [
        ("report_data", artifacts.report_data_path, False),
        ("records", artifacts.abstracts_path, False),
        ("membership", artifacts.membership_path, False),
        ("keywords", artifacts.keywords_path, False),
    ]:
        if path is None:
            continue
        rows.append(
            {
                "role": role,
                "path": _rel(path, root),
                "required": required,
            }
        )
    landscape_dir = artifacts.landscape_dir
    if landscape_dir is not None:
        cooc_table = landscape_dir / "term_cooccurrence.parquet"
        if cooc_table.exists():
            rows.append({"role": "cooccurrence", "path": _rel(cooc_table, root), "required": False})
        cooc_map = landscape_dir / "term_cooccurrence_map.json"
        if cooc_map.exists():
            rows.append({"role": "cooccurrence_map", "path": _rel(cooc_map, root), "required": False})
    for index, path in enumerate(artifacts.edge_evidence_paths, start=1):
        rows.append({"role": f"edge_evidence_{index}", "path": _rel(path, root), "required": False})
    return rows


def _review_packet_nodes_from_keywords(
    artifacts: ResultArtifacts,
    *,
    max_keywords_per_cluster: int,
) -> list[dict[str, Any]]:
    if artifacts.keywords_path is None or not artifacts.keywords_path.exists():
        return []
    try:
        keywords = pd.read_parquet(artifacts.keywords_path)
    except Exception:
        return []
    if keywords.empty:
        return []
    columns = list(keywords.columns)
    cluster_col = next((column for column in columns if "cluster" in column.lower()), columns[0])
    label_col = _keyword_label_column(columns)
    score_col = _keyword_score_column(columns)
    nodes: list[dict[str, Any]] = []
    for cluster_id, group in keywords.groupby(cluster_col, sort=True):
        work = group.copy()
        if score_col:
            work = work.sort_values(score_col, ascending=False, kind="stable")
        keyword_rows = []
        for rank, row in enumerate(work.head(max(1, max_keywords_per_cluster)).to_dict("records"), start=1):
            term = str(row.get(label_col) or "").strip()
            if not term:
                continue
            keyword = {"term": term, "rank": rank}
            score = _coerce_float(row.get(score_col)) if score_col else None
            if score is not None:
                keyword["score"] = score
            for key in ("frequency", "keyword_label_tier", "quality_flags", "review_status"):
                if key in row and row.get(key) not in (None, ""):
                    keyword[key] = row.get(key)
            keyword_rows.append(keyword)
        cluster_key = str(cluster_id)
        label = keyword_rows[0]["term"] if keyword_rows else f"Cluster {cluster_key}"
        nodes.append(
            {
                "cluster_uid": f"cluster:{cluster_key}",
                "level": "cluster",
                "cluster_id": cluster_key,
                "label": label,
                "short_label": _short_label(label),
                "doc_count": None,
                "keywords": keyword_rows,
                "badges": _cluster_badges({"keywords": keyword_rows}),
                "representative_works": [],
                "representative_work_count": 0,
            }
        )
    return nodes


def _review_packet_nodes(
    artifacts: ResultArtifacts,
    *,
    max_keywords_per_cluster: int,
    max_representative_works: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    warnings: list[dict[str, Any]] = []
    if artifacts.report_data_path is not None:
        try:
            report_data = json.loads(artifacts.report_data_path.read_text(encoding="utf-8"))
        except Exception as exc:
            warnings.append(_review_packet_issue("review_report_read_failed", "warning", str(exc), artifact="report_data"))
            report_data = None
        if isinstance(report_data, dict):
            atlas = build_atlas_payload_from_report_data(
                report_data,
                membership_path=artifacts.membership_path,
                edges_path=artifacts.edges_path,
                abstracts_path=artifacts.abstracts_path,
                edge_evidence_paths=artifacts.edge_evidence_paths,
                max_representative_works=max_representative_works,
            )
            nodes = [dict(node) for node in atlas.get("nodes", []) if isinstance(node, Mapping)]
            return nodes, [*warnings, *list(atlas.get("warnings") or [])]
    return _review_packet_nodes_from_keywords(
        artifacts,
        max_keywords_per_cluster=max_keywords_per_cluster,
    ), warnings


def _review_packet_cooccurrence_by_cluster(artifacts: ResultArtifacts) -> dict[str, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    landscape_dir = artifacts.landscape_dir
    cooc_path = landscape_dir / "term_cooccurrence.parquet" if landscape_dir is not None else None
    if cooc_path is not None and cooc_path.exists():
        try:
            rows = pd.read_parquet(cooc_path).to_dict("records")
        except Exception:
            rows = []
    if not rows and artifacts.report_data_path is not None:
        try:
            payload = json.loads(artifacts.report_data_path.read_text(encoding="utf-8"))
        except Exception:
            payload = None
        if isinstance(payload, dict):
            rows = _cooccurrence_rows_from_report_data(payload)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        cluster_uid = str(row.get("cluster_uid") or "").strip()
        if not cluster_uid:
            cluster_level = str(row.get("cluster_level") or "cluster")
            cluster_id = str(row.get("cluster_id") or "unknown")
            cluster_uid = f"{cluster_level}:{cluster_id}"
        grouped.setdefault(cluster_uid, []).append(row)
    for cluster_uid, values in grouped.items():
        grouped[cluster_uid] = sorted(
            values,
            key=lambda item: (
                -float(_coerce_float(item.get("weight")) or 0.0),
                str(item.get("source") or ""),
                str(item.get("target") or ""),
            ),
        )
    return grouped


def _packet_ref_id(cluster_uid: str, evidence_type: str, value: str | int) -> str:
    return _safe_id(f"{cluster_uid}_{evidence_type}_{value}", fallback=f"{evidence_type}_ref")


def _cluster_review_packet_row(
    node: Mapping[str, Any],
    *,
    cooccurrence_rows: list[Mapping[str, Any]],
    max_keywords_per_cluster: int,
    max_representative_works: int,
    max_cooccurrence_links: int,
) -> dict[str, Any]:
    cluster_uid = str(node.get("cluster_uid") or "").strip()
    level = str(node.get("level") or "cluster")
    cluster_id = str(node.get("cluster_id") or cluster_uid.split(":", 1)[-1])
    label = str(node.get("label") or node.get("short_label") or cluster_uid)
    evidence_refs: list[dict[str, Any]] = []
    keyword_evidence: list[dict[str, Any]] = []
    representative_works: list[dict[str, Any]] = []
    cooccurrence_evidence: list[dict[str, Any]] = []
    quality_caveats: list[dict[str, Any]] = []

    for rank, keyword in enumerate(list(node.get("keywords") or [])[:max_keywords_per_cluster], start=1):
        if not isinstance(keyword, Mapping):
            continue
        term = str(keyword.get("term") or "").strip()
        if not term:
            continue
        ref_id = _packet_ref_id(cluster_uid, "term", rank)
        evidence_refs.append(
            {
                "evidence_ref_id": ref_id,
                "evidence_type": "term",
                "source_role": "keywords",
                "evidence_label": term,
                "aggregate_only": True,
            }
        )
        row = {
            "evidence_ref_id": ref_id,
            "rank": rank,
            "term": term,
            "score": _coerce_float(keyword.get("score")),
            "frequency": _coerce_int(keyword.get("frequency")),
            "tier": str(keyword.get("keyword_label_tier") or keyword.get("keyword_scope") or ""),
            "quality_flags": str(keyword.get("quality_flags") or ""),
        }
        keyword_evidence.append({key: value for key, value in row.items() if value not in (None, "")})

    for rank, work in enumerate(list(node.get("representative_works") or [])[:max_representative_works], start=1):
        if not isinstance(work, Mapping):
            continue
        uid = str(work.get("uid") or work.get("id") or "").strip()
        title = _plain_atlas_text(work.get("title") or "")
        if not uid and not title:
            continue
        ref_id = _packet_ref_id(cluster_uid, "work", uid or rank)
        evidence_refs.append(
            {
                "evidence_ref_id": ref_id,
                "evidence_type": "work",
                "source_role": "records",
                "evidence_label": title or uid,
                "entity_key": uid,
                "aggregate_only": False,
            }
        )
        row = {
            "evidence_ref_id": ref_id,
            "rank": rank,
            "uid": uid,
            "title": title,
            "year": _coerce_int(work.get("year") or work.get("pubyear")),
            "cited_by_count": _coerce_int(work.get("cited_by_count")),
        }
        representative_works.append({key: value for key, value in row.items() if value not in (None, "")})

    for rank, cooc in enumerate(cooccurrence_rows[:max_cooccurrence_links], start=1):
        source = str(cooc.get("source") or "").strip()
        target = str(cooc.get("target") or "").strip()
        if not source or not target:
            continue
        ref_id = _packet_ref_id(cluster_uid, "cooccurrence", rank)
        label_text = f"{source} - {target}"
        evidence_refs.append(
            {
                "evidence_ref_id": ref_id,
                "evidence_type": "cooccurrence",
                "source_role": "cooccurrence",
                "evidence_label": label_text,
                "aggregate_only": True,
            }
        )
        row = {
            "evidence_ref_id": ref_id,
            "rank": rank,
            "source": source,
            "target": target,
            "weight": _coerce_float(cooc.get("weight")) or 1.0,
            "count": _coerce_int(cooc.get("count")),
            "relation": str(cooc.get("relation") or "cooccurrence"),
        }
        cooccurrence_evidence.append({key: value for key, value in row.items() if value not in (None, "")})

    for rank, badge in enumerate(list(node.get("badges") or []), start=1):
        if not isinstance(badge, Mapping):
            continue
        ref_id = _packet_ref_id(cluster_uid, "qa_caveat", rank)
        label_text = str(badge.get("label") or badge.get("badge_id") or "Review caveat")
        evidence_refs.append(
            {
                "evidence_ref_id": ref_id,
                "evidence_type": "qa_caveat",
                "source_role": "quality",
                "evidence_label": label_text,
                "aggregate_only": True,
            }
        )
        quality_caveats.append(
            {
                "evidence_ref_id": ref_id,
                "code": str(badge.get("badge_id") or "quality_caveat"),
                "severity": str(badge.get("severity") or "warning"),
                "message": str(badge.get("tooltip") or label_text),
            }
        )

    critical_caveat = any(str(row.get("severity")) == "critical" for row in quality_caveats)
    narrative_ready = bool(keyword_evidence and (representative_works or cooccurrence_evidence) and not critical_caveat)
    review_status = "review_required" if quality_caveats or not narrative_ready else "clean"
    return {
        "cluster_uid": cluster_uid,
        "cluster_level": level,
        "cluster_id": cluster_id,
        "label": label,
        "short_label": str(node.get("short_label") or _short_label(label)),
        "doc_count": _coerce_int(node.get("doc_count")),
        "review_status": review_status,
        "narrative_ready": narrative_ready,
        "counts": {
            "keywords": len(keyword_evidence),
            "representative_works": len(representative_works),
            "cooccurrence_links": len(cooccurrence_evidence),
            "quality_caveats": len(quality_caveats),
            "evidence_refs": len(evidence_refs),
        },
        "keyword_evidence": keyword_evidence,
        "representative_works": representative_works,
        "cooccurrence_evidence": cooccurrence_evidence,
        "quality_caveats": quality_caveats,
        "evidence_refs": evidence_refs,
    }


def _cluster_review_packet_qa(packet: Mapping[str, Any], warnings: list[dict[str, Any]]) -> dict[str, Any]:
    clusters = [row for row in packet.get("clusters", []) if isinstance(row, Mapping)]
    counts = {
        "clusters": len(clusters),
        "narrative_ready_clusters": sum(1 for row in clusters if row.get("narrative_ready") is True),
        "review_required_clusters": sum(1 for row in clusters if row.get("review_status") == "review_required"),
        "keyword_evidence_rows": sum(len(row.get("keyword_evidence") or []) for row in clusters),
        "representative_work_rows": sum(len(row.get("representative_works") or []) for row in clusters),
        "cooccurrence_evidence_rows": sum(len(row.get("cooccurrence_evidence") or []) for row in clusters),
        "quality_caveat_rows": sum(len(row.get("quality_caveats") or []) for row in clusters),
        "evidence_refs": sum(len(row.get("evidence_refs") or []) for row in clusters),
    }
    checks = {
        "evidence_refs_resolvable": {"status": "passed"},
        "narrative_ready": {
            "status": "passed" if counts["narrative_ready_clusters"] == counts["clusters"] else "warning",
            "ready": counts["narrative_ready_clusters"],
            "total": counts["clusters"],
        },
    }
    status = "warning" if warnings or any(check.get("status") != "passed" for check in checks.values()) else "passed"
    return {
        "schema_version": CLUSTER_REVIEW_PACKET_QA_SCHEMA_VERSION,
        "packet_id": packet.get("packet_id"),
        "status": status,
        "counts": counts,
        "checks": checks,
        "warnings": warnings,
        "blocking_issues": [],
        "created_at_utc": _utc_now(),
    }


def write_cluster_review_packet_artifact(
    path: str | Path,
    *,
    output_dir: str | Path | None = None,
    packet_id: str = "cluster_review_packet_default",
    max_clusters: int = 500,
    max_keywords_per_cluster: int = 8,
    max_representative_works: int = 3,
    max_cooccurrence_links: int = 8,
) -> dict[str, Any] | None:
    """Write a compact evidence packet for cluster review and future narratives."""

    artifacts = infer_result_artifacts(path)
    nodes, warnings = _review_packet_nodes(
        artifacts,
        max_keywords_per_cluster=max_keywords_per_cluster,
        max_representative_works=max_representative_works,
    )
    if not nodes:
        return None
    cooccurrence_by_cluster = _review_packet_cooccurrence_by_cluster(artifacts)
    clusters = [
        _cluster_review_packet_row(
            node,
            cooccurrence_rows=cooccurrence_by_cluster.get(str(node.get("cluster_uid") or ""), []),
            max_keywords_per_cluster=max_keywords_per_cluster,
            max_representative_works=max_representative_works,
            max_cooccurrence_links=max_cooccurrence_links,
        )
        for node in nodes[: max(1, int(max_clusters))]
        if str(node.get("cluster_uid") or "").strip()
    ]
    if not clusters:
        return None

    target_dir = Path(output_dir) if output_dir is not None else artifacts.result_root / "review"
    target_dir.mkdir(parents=True, exist_ok=True)
    packet_path = target_dir / "cluster_review_packet.json"
    qa_path = target_dir / "cluster_review_packet_qa.json"
    packet = {
        "schema_version": CLUSTER_REVIEW_PACKET_SCHEMA_VERSION,
        "packet_id": packet_id,
        "title": "Cluster review packet",
        "result_id": None,
        "created_at_utc": _utc_now(),
        "packet_scope": {
            "target_type": "cluster",
            "max_clusters": int(max_clusters),
            "max_keywords_per_cluster": int(max_keywords_per_cluster),
            "max_representative_works": int(max_representative_works),
            "max_cooccurrence_links": int(max_cooccurrence_links),
        },
        "review_policy": {
            "allowed_evidence_types": ["term", "work", "cooccurrence", "qa_caveat"],
            "narrative_generation_allowed": False,
            "unsupported_claim_action": "block_in_narrative_validator",
            "review_status_values": ["clean", "review_required"],
        },
        "source_artifacts": _review_packet_source_artifacts(artifacts),
        "clusters": clusters,
        "warnings": warnings,
    }
    qa = _cluster_review_packet_qa(packet, warnings)
    packet["qa"] = {
        "path": _review_packet_portable_rel(qa_path, result_root=artifacts.result_root, packet_path=packet_path),
        "status": qa["status"],
        "counts": qa["counts"],
    }
    packet_path.write_text(json.dumps(packet, indent=2, sort_keys=True), encoding="utf-8")
    qa_path.write_text(json.dumps(qa, indent=2, sort_keys=True), encoding="utf-8")
    validation = validate_cluster_review_packet_artifact(packet_path)
    return {
        "schema_version": CLUSTER_REVIEW_PACKET_SCHEMA_VERSION,
        "packet_path": packet_path,
        "qa_path": qa_path,
        "packet_id": packet_id,
        "qa": qa,
        "validation": validation.to_dict(),
        "clusters": int(len(clusters)),
        "narrative_ready_clusters": int(qa["counts"]["narrative_ready_clusters"]),
    }


def validate_cluster_review_packet_artifact(path: str | Path) -> ClusterReviewPacketValidationResult:
    """Validate a compact cluster review packet without loading the web app."""

    packet_path = _cluster_review_packet_path(path)
    warnings: list[dict[str, Any]] = []
    blocking_issues: list[dict[str, Any]] = []
    checks: dict[str, dict[str, Any]] = {}
    payload: dict[str, Any] = {}
    if not packet_path.exists():
        blocking_issues.append(
            _review_packet_issue("missing_review_packet", "blocking", "Cluster review packet is missing.", artifact="packet")
        )
    else:
        try:
            loaded = json.loads(packet_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                payload = loaded
            else:
                blocking_issues.append(
                    _review_packet_issue("invalid_review_packet_shape", "blocking", "Packet JSON must be an object.", artifact="packet")
                )
        except Exception as exc:
            blocking_issues.append(
                _review_packet_issue("invalid_review_packet_json", "blocking", f"Could not read packet JSON: {exc}", artifact="packet")
            )

    packet_id = str(payload.get("packet_id") or "") if payload else None
    if payload:
        if payload.get("schema_version") != CLUSTER_REVIEW_PACKET_SCHEMA_VERSION:
            blocking_issues.append(
                _review_packet_issue(
                    "unsupported_review_packet_schema",
                    "blocking",
                    f"Unsupported cluster review packet schema: {payload.get('schema_version')}",
                    artifact="packet",
                )
            )
        clusters = payload.get("clusters")
        if not isinstance(clusters, list) or not clusters:
            blocking_issues.append(
                _review_packet_issue("missing_review_clusters", "blocking", "Packet must contain at least one cluster.", artifact="packet")
            )
            clusters = []
    else:
        clusters = []

    cluster_count = 0
    ready_count = 0
    review_required_count = 0
    evidence_ref_count = 0
    keyword_count = 0
    work_count = 0
    cooccurrence_count = 0
    caveat_count = 0
    unresolved_refs = 0
    duplicate_refs = 0
    for index, cluster in enumerate(clusters):
        if not isinstance(cluster, Mapping):
            blocking_issues.append(
                _review_packet_issue("invalid_review_cluster_shape", "blocking", "Cluster rows must be objects.", artifact="clusters")
            )
            continue
        cluster_count += 1
        cluster_uid = str(cluster.get("cluster_uid") or "").strip()
        if not cluster_uid:
            blocking_issues.append(
                _review_packet_issue("missing_review_cluster_uid", "blocking", f"Cluster row {index} has no cluster_uid.", artifact="clusters")
            )
        if not str(cluster.get("label") or "").strip():
            warnings.append(
                _review_packet_issue("missing_review_cluster_label", "warning", f"Cluster {cluster_uid or index} has no label.", artifact="clusters")
            )
        if cluster.get("narrative_ready") is True:
            ready_count += 1
        if cluster.get("review_status") == "review_required":
            review_required_count += 1
        refs = cluster.get("evidence_refs")
        if not isinstance(refs, list):
            blocking_issues.append(
                _review_packet_issue("missing_review_evidence_refs", "blocking", f"Cluster {cluster_uid or index} has no evidence refs.", artifact="clusters")
            )
            refs = []
        ref_ids: set[str] = set()
        for ref in refs:
            if not isinstance(ref, Mapping):
                continue
            ref_id = str(ref.get("evidence_ref_id") or "").strip()
            if not ref_id:
                blocking_issues.append(
                    _review_packet_issue("missing_review_evidence_ref_id", "blocking", f"Cluster {cluster_uid or index} has an unnamed evidence ref.", artifact="clusters")
                )
                continue
            if ref_id in ref_ids:
                duplicate_refs += 1
            ref_ids.add(ref_id)
        evidence_ref_count += len(ref_ids)
        for field_name, counter_name in [
            ("keyword_evidence", "keywords"),
            ("representative_works", "works"),
            ("cooccurrence_evidence", "cooccurrence"),
            ("quality_caveats", "caveats"),
        ]:
            rows = cluster.get(field_name)
            if not isinstance(rows, list):
                rows = []
            for row in rows:
                if not isinstance(row, Mapping):
                    continue
                ref_id = str(row.get("evidence_ref_id") or "").strip()
                if not ref_id or ref_id not in ref_ids:
                    unresolved_refs += 1
            if counter_name == "keywords":
                keyword_count += len(rows)
            elif counter_name == "works":
                work_count += len(rows)
            elif counter_name == "cooccurrence":
                cooccurrence_count += len(rows)
            elif counter_name == "caveats":
                caveat_count += len(rows)

    if duplicate_refs:
        blocking_issues.append(
            _review_packet_issue("duplicate_review_evidence_refs", "blocking", f"{duplicate_refs} duplicate evidence refs were found.", artifact="clusters")
        )
    if unresolved_refs:
        blocking_issues.append(
            _review_packet_issue("unresolved_review_evidence_refs", "blocking", f"{unresolved_refs} evidence rows do not resolve to cluster evidence_refs.", artifact="clusters")
        )

    result_root = _review_packet_result_root(packet_path)
    source_artifacts = payload.get("source_artifacts") if payload else None
    source_count = 0
    missing_sources = 0
    absolute_source_paths = 0
    outside_source_paths = 0
    invalid_source_rows = 0
    if payload:
        if not isinstance(source_artifacts, list) or not source_artifacts:
            warnings.append(
                _review_packet_issue(
                    "missing_review_packet_source_artifacts",
                    "warning",
                    "Cluster review packet should record source artifacts.",
                    artifact="source_artifacts",
                )
            )
        else:
            source_count = len(source_artifacts)
            root_resolved = result_root.resolve()
            for source in source_artifacts:
                if not isinstance(source, Mapping):
                    invalid_source_rows += 1
                    continue
                raw_path = str(source.get("path") or "").strip()
                if not raw_path:
                    invalid_source_rows += 1
                    continue
                source_path = Path(raw_path)
                if source_path.is_absolute():
                    absolute_source_paths += 1
                    continue
                resolved = (result_root / source_path).resolve()
                try:
                    resolved.relative_to(root_resolved)
                except ValueError:
                    outside_source_paths += 1
                    continue
                if not resolved.exists():
                    missing_sources += 1
            if invalid_source_rows:
                warnings.append(
                    _review_packet_issue(
                        "invalid_review_packet_source_artifact",
                        "warning",
                        f"{invalid_source_rows} source artifact refs are invalid.",
                        artifact="source_artifacts",
                    )
                )
            if absolute_source_paths:
                warnings.append(
                    _review_packet_issue(
                        "absolute_review_packet_source_path",
                        "warning",
                        f"{absolute_source_paths} source artifact refs use absolute paths.",
                        artifact="source_artifacts",
                    )
                )
            if outside_source_paths:
                warnings.append(
                    _review_packet_issue(
                        "outside_review_packet_source_path",
                        "warning",
                        f"{outside_source_paths} source artifact refs escape the result root.",
                        artifact="source_artifacts",
                    )
                )
            if missing_sources:
                warnings.append(
                    _review_packet_issue(
                        "missing_review_packet_source_artifact",
                        "warning",
                        f"{missing_sources} source artifact refs do not exist.",
                        artifact="source_artifacts",
                    )
                )
    checks["source_artifacts"] = {
        "status": "warning" if missing_sources or absolute_source_paths or outside_source_paths or invalid_source_rows or source_count == 0 else "passed",
        "count": int(source_count),
        "missing": int(missing_sources),
        "absolute_paths": int(absolute_source_paths),
        "outside_paths": int(outside_source_paths),
        "invalid_rows": int(invalid_source_rows),
    }

    qa_path, qa_ref, qa_absolute = _review_packet_declared_qa_path(packet_path, payload)
    qa_rel = qa_ref or (qa_path.name if qa_path.exists() else None)
    if qa_absolute:
        blocking_issues.append(
            _review_packet_issue(
                "absolute_review_packet_qa_path",
                "blocking",
                "Cluster review packet QA path must be portable, not absolute.",
                artifact="qa",
            )
        )
    if not qa_path.exists():
        warnings.append(
            _review_packet_issue("missing_review_packet_qa", "warning", "Cluster review packet QA sidecar is missing.", artifact="qa")
        )
    else:
        try:
            qa_payload = json.loads(qa_path.read_text(encoding="utf-8"))
            if qa_payload.get("schema_version") != CLUSTER_REVIEW_PACKET_QA_SCHEMA_VERSION:
                warnings.append(
                    _review_packet_issue("unsupported_review_packet_qa_schema", "warning", "Unsupported review packet QA schema.", artifact="qa")
                )
        except Exception as exc:
            warnings.append(
                _review_packet_issue("invalid_review_packet_qa_json", "warning", f"Could not read review packet QA: {exc}", artifact="qa")
            )

    checks["evidence_refs_resolvable"] = {
        "status": "blocked" if unresolved_refs or duplicate_refs else "passed",
        "unresolved_refs": int(unresolved_refs),
        "duplicate_refs": int(duplicate_refs),
    }
    checks["narrative_ready"] = {
        "status": "passed" if ready_count == cluster_count else "warning",
        "ready": int(ready_count),
        "total": int(cluster_count),
    }
    counts = {
        "clusters": int(cluster_count),
        "narrative_ready_clusters": int(ready_count),
        "review_required_clusters": int(review_required_count),
        "keyword_evidence_rows": int(keyword_count),
        "representative_work_rows": int(work_count),
        "cooccurrence_evidence_rows": int(cooccurrence_count),
        "quality_caveat_rows": int(caveat_count),
        "evidence_refs": int(evidence_ref_count),
    }
    return ClusterReviewPacketValidationResult(
        schema_version=CLUSTER_REVIEW_PACKET_SCHEMA_VERSION,
        packet_id=packet_id or None,
        status="blocked" if blocking_issues else "passed",
        packet_path=str(packet_path),
        qa_path=qa_rel,
        counts=counts,
        checks=checks,
        warnings=warnings,
        blocking_issues=blocking_issues,
        created_at_utc=_utc_now(),
    )

def _narrative_issue(code: str, severity: str, message: str, *, artifact: str | None = None) -> dict[str, Any]:
    issue = {"code": code, "severity": severity, "message": message}
    if artifact:
        issue["artifact"] = artifact
    return issue


def _narrative_dir_and_manifest(path: str | Path) -> tuple[Path, Path]:
    candidate = Path(path).expanduser()
    if candidate.exists():
        candidate = candidate.resolve()
    if candidate.is_file():
        return candidate.parent, candidate
    manifest = candidate / "narrative_manifest.json"
    if manifest.exists() or candidate.name == "narrative":
        return candidate, manifest
    nested = candidate / "narrative" / "narrative_manifest.json"
    if nested.exists() or (candidate / "narrative").is_dir():
        return nested.parent, nested
    return candidate, manifest


def _narrative_result_root(narrative_dir: Path) -> Path:
    if narrative_dir.name == "narrative":
        if narrative_dir.parent.name == "landscape":
            return narrative_dir.parent.parent
        return narrative_dir.parent
    return narrative_dir


def _narrative_output_path(narrative_dir: Path, manifest: Mapping[str, Any], key: str, default_name: str) -> Path:
    outputs = manifest.get("outputs") if isinstance(manifest.get("outputs"), Mapping) else {}
    raw = str(outputs.get(key) or default_name)
    path = Path(raw)
    return path if path.is_absolute() else narrative_dir / path


def _read_narrative_table(
    path: Path,
    *,
    required: set[str],
    schema_version: str,
    artifact: str,
    warnings: list[dict[str, Any]],
    blocking_issues: list[dict[str, Any]],
) -> pd.DataFrame | None:
    if not path.exists():
        blocking_issues.append(_narrative_issue(f"missing_narrative_{artifact}", "blocking", f"Missing narrative {artifact} table.", artifact=artifact))
        return None
    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        blocking_issues.append(_narrative_issue(f"invalid_narrative_{artifact}_table", "blocking", f"Could not read narrative {artifact} table: {exc}", artifact=artifact))
        return None
    missing = required - set(df.columns)
    if missing:
        blocking_issues.append(_narrative_issue(f"missing_narrative_{artifact}_columns", "blocking", f"Narrative {artifact} table is missing columns: {sorted(missing)}", artifact=artifact))
        return df
    schema_values = set(df["schema_version"].dropna().map(str).unique().tolist()) if "schema_version" in df.columns else set()
    if schema_values and schema_values != {schema_version}:
        blocking_issues.append(_narrative_issue(f"unsupported_narrative_{artifact}_schema", "blocking", f"Unsupported narrative {artifact} schema values: {sorted(schema_values)}", artifact=artifact))
    if df.empty and artifact != "reviews":
        warnings.append(_narrative_issue(f"empty_narrative_{artifact}", "warning", f"Narrative {artifact} table is empty.", artifact=artifact))
    return df


def _narrative_claim_counts(claims: pd.DataFrame | None) -> dict[str, int]:
    counts = {"supported": 0, "weak": 0, "contradicted": 0, "unsupported": 0, "caveat": 0, "model_generated": 0}
    if claims is None or claims.empty:
        return counts
    if "support_state" in claims.columns:
        values = claims["support_state"].fillna("").map(str).str.lower()
        for key in ("supported", "weak", "contradicted", "unsupported", "caveat"):
            counts[key] = int((values == key).sum())
    if "text_origin" in claims.columns:
        counts["model_generated"] = int((claims["text_origin"].fillna("").map(str) == "model_generated").sum())
    return counts


def _bounded_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) and 0.0 <= numeric <= 1.0 else None


def validate_narrative_artifact(path: str | Path) -> NarrativeArtifactValidationResult:
    """Validate a narrative claim/evidence artifact without loading the web app."""

    narrative_dir, manifest_path = _narrative_dir_and_manifest(path)
    result_root = _narrative_result_root(narrative_dir)
    warnings: list[dict[str, Any]] = []
    blocking_issues: list[dict[str, Any]] = []
    checks: dict[str, dict[str, Any]] = {}
    manifest: dict[str, Any] = {}
    if not manifest_path.exists():
        blocking_issues.append(_narrative_issue("missing_narrative_manifest", "blocking", "Missing narrative_manifest.json.", artifact="manifest"))
    else:
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                manifest = loaded
            else:
                blocking_issues.append(_narrative_issue("invalid_narrative_manifest_shape", "blocking", "Narrative manifest JSON must be an object.", artifact="manifest"))
        except Exception as exc:
            blocking_issues.append(_narrative_issue("invalid_narrative_manifest_json", "blocking", f"Could not read narrative manifest JSON: {exc}", artifact="manifest"))
    narrative_id = str(manifest.get("narrative_id") or "") if manifest else ""
    if manifest and manifest.get("schema_version") != NARRATIVE_MANIFEST_SCHEMA_VERSION:
        blocking_issues.append(_narrative_issue("unsupported_narrative_manifest_schema", "blocking", f"Unsupported narrative manifest schema: {manifest.get('schema_version')}", artifact="manifest"))

    paths = {
        "targets": _narrative_output_path(narrative_dir, manifest, "targets", "narrative_targets.parquet"),
        "claims": _narrative_output_path(narrative_dir, manifest, "claims", "claims.parquet"),
        "evidence_sources": _narrative_output_path(narrative_dir, manifest, "evidence_sources", "evidence_sources.parquet"),
        "evidence_refs": _narrative_output_path(narrative_dir, manifest, "evidence_refs", "evidence_refs.parquet"),
        "claim_evidence_links": _narrative_output_path(narrative_dir, manifest, "claim_evidence_links", "claim_evidence_links.parquet"),
        "sections": _narrative_output_path(narrative_dir, manifest, "sections", "narrative_sections.parquet"),
        "reviews": _narrative_output_path(narrative_dir, manifest, "reviews", "review_decisions.parquet"),
        "qa": _narrative_output_path(narrative_dir, manifest, "qa", "narrative_qa.json"),
    }
    targets = _read_narrative_table(paths["targets"], required=REQUIRED_NARRATIVE_TARGET_COLUMNS, schema_version=NARRATIVE_TARGETS_SCHEMA_VERSION, artifact="targets", warnings=warnings, blocking_issues=blocking_issues)
    claims = _read_narrative_table(paths["claims"], required=REQUIRED_NARRATIVE_CLAIM_COLUMNS, schema_version=NARRATIVE_CLAIMS_SCHEMA_VERSION, artifact="claims", warnings=warnings, blocking_issues=blocking_issues)
    sources = _read_narrative_table(paths["evidence_sources"], required=REQUIRED_NARRATIVE_EVIDENCE_SOURCE_COLUMNS, schema_version=NARRATIVE_EVIDENCE_SOURCES_SCHEMA_VERSION, artifact="evidence_sources", warnings=warnings, blocking_issues=blocking_issues)
    refs = _read_narrative_table(paths["evidence_refs"], required=REQUIRED_NARRATIVE_EVIDENCE_REF_COLUMNS, schema_version=NARRATIVE_EVIDENCE_REFS_SCHEMA_VERSION, artifact="evidence_refs", warnings=warnings, blocking_issues=blocking_issues)
    links = _read_narrative_table(paths["claim_evidence_links"], required=REQUIRED_NARRATIVE_CLAIM_LINK_COLUMNS, schema_version=NARRATIVE_CLAIM_EVIDENCE_LINKS_SCHEMA_VERSION, artifact="claim_evidence_links", warnings=warnings, blocking_issues=blocking_issues)
    sections = _read_narrative_table(paths["sections"], required=REQUIRED_NARRATIVE_SECTION_COLUMNS, schema_version=NARRATIVE_SECTIONS_SCHEMA_VERSION, artifact="sections", warnings=warnings, blocking_issues=blocking_issues)
    reviews = None
    if bool(manifest.get("review_state_advertised")) or paths["reviews"].exists():
        reviews = _read_narrative_table(paths["reviews"], required=REQUIRED_NARRATIVE_REVIEW_COLUMNS, schema_version=NARRATIVE_REVIEW_DECISIONS_SCHEMA_VERSION, artifact="reviews", warnings=warnings, blocking_issues=blocking_issues)

    def _id_set(df: pd.DataFrame | None, column: str) -> set[str]:
        if df is None or column not in df.columns:
            return set()
        return set(df[column].dropna().map(str).tolist())

    target_ids = _id_set(targets, "target_id")
    section_ids = _id_set(sections, "section_id")
    claim_ids = _id_set(claims, "claim_id")
    source_ids = _id_set(sources, "evidence_source_id")
    evidence_ids = _id_set(refs, "evidence_ref_id")
    for name, df, column in [
        ("targets", targets, "target_id"),
        ("sections", sections, "section_id"),
        ("claims", claims, "claim_id"),
        ("evidence_sources", sources, "evidence_source_id"),
        ("evidence_refs", refs, "evidence_ref_id"),
    ]:
        if df is not None and column in df.columns:
            duplicates = int(df[column].map(str).duplicated().sum())
            if duplicates:
                blocking_issues.append(_narrative_issue(f"duplicate_narrative_{name}", "blocking", f"{duplicates} duplicate narrative {column} values were found.", artifact=name))

    unresolved_target_refs = unresolved_section_refs = invalid_confidence = 0
    unsupported_normal_claims = model_metadata_missing = supported_claims_without_required_links = 0
    if claims is not None:
        for row in claims.to_dict("records"):
            claim_id = str(row.get("claim_id") or "")
            if str(row.get("target_id") or "") not in target_ids:
                unresolved_target_refs += 1
            if str(row.get("section_id") or "") not in section_ids:
                unresolved_section_refs += 1
            if _bounded_float(row.get("confidence")) is None:
                invalid_confidence += 1
            support_state = str(row.get("support_state") or "").lower()
            claim_type = str(row.get("claim_type") or "").lower()
            if support_state in {"unsupported", "contradicted"} and claim_type not in {"quality_caveat", "limitation"}:
                unsupported_normal_claims += 1
            if str(row.get("text_origin") or "") == "model_generated" and not manifest.get("model_generation"):
                model_metadata_missing += 1
            if support_state == "supported":
                claim_links = links[links["claim_id"].map(str) == claim_id] if links is not None and "claim_id" in links.columns else pd.DataFrame()
                has_required = any(
                    bool(link.get("required")) or str(link.get("evidence_role") or "") in {"primary", "supporting"}
                    for link in claim_links.to_dict("records")
                )
                if not has_required:
                    supported_claims_without_required_links += 1

    unresolved_source_refs = aggregate_only_refs = 0
    if refs is not None:
        for row in refs.to_dict("records"):
            if str(row.get("evidence_source_id") or "") not in source_ids:
                unresolved_source_refs += 1
            if str(row.get("locator_type") or "") == "aggregate" or bool(row.get("aggregate_only", False)):
                aggregate_only_refs += 1

    unresolved_claim_links = unresolved_evidence_links = duplicate_links = invalid_link_strength = 0
    if links is not None:
        seen_links: set[tuple[str, str, str]] = set()
        for row in links.to_dict("records"):
            claim_id = str(row.get("claim_id") or "")
            evidence_ref_id = str(row.get("evidence_ref_id") or "")
            role = str(row.get("evidence_role") or "")
            key = (claim_id, evidence_ref_id, role)
            if key in seen_links:
                duplicate_links += 1
            seen_links.add(key)
            if claim_id not in claim_ids:
                unresolved_claim_links += 1
            if evidence_ref_id not in evidence_ids:
                unresolved_evidence_links += 1
            if _bounded_float(row.get("link_strength")) is None:
                invalid_link_strength += 1

    section_target_refs = 0
    if sections is not None and "target_id" in sections.columns:
        section_target_refs = int(sum(1 for value in sections["target_id"].dropna().map(str).tolist() if value not in target_ids))
    source_missing = source_absolute = source_blocked = 0
    if sources is not None:
        root_resolved = result_root.resolve()
        for row in sources.to_dict("records"):
            artifact_path = str(row.get("artifact_path") or "")
            if str(row.get("source_state") or "") == "blocked":
                source_blocked += 1
            if not artifact_path:
                source_missing += 1
                continue
            path_obj = Path(artifact_path)
            if path_obj.is_absolute():
                source_absolute += 1
                continue
            resolved = (result_root / path_obj).resolve()
            try:
                resolved.relative_to(root_resolved)
            except ValueError:
                source_absolute += 1
                continue
            if not resolved.exists():
                source_missing += 1

    blocking_checks = {
        "unresolved_target_refs": unresolved_target_refs + section_target_refs,
        "unresolved_section_refs": unresolved_section_refs,
        "unresolved_source_refs": unresolved_source_refs,
        "unresolved_claim_links": unresolved_claim_links,
        "unresolved_evidence_links": unresolved_evidence_links,
        "duplicate_links": duplicate_links,
        "invalid_confidence": invalid_confidence,
        "invalid_link_strength": invalid_link_strength,
        "unsupported_normal_claims": unsupported_normal_claims,
        "model_metadata_missing": model_metadata_missing,
        "supported_claims_without_required_links": supported_claims_without_required_links,
        "missing_source_artifacts": source_missing,
        "absolute_source_paths": source_absolute,
        "blocked_sources": source_blocked,
    }
    for code, count in blocking_checks.items():
        if count:
            blocking_issues.append(_narrative_issue(f"narrative_{code}", "blocking", f"{count} narrative {code.replace('_', ' ')} were found.", artifact="narrative"))

    claim_counts = _narrative_claim_counts(claims)
    checks["refs_resolvable"] = {"status": "blocked" if any(blocking_checks.values()) else "passed", **{key: int(value) for key, value in blocking_checks.items()}}
    checks["claim_support"] = {
        "status": "blocked" if supported_claims_without_required_links else ("warning" if claim_counts["weak"] or aggregate_only_refs else "passed"),
        "weak_claims": int(claim_counts["weak"]),
        "aggregate_only_refs": int(aggregate_only_refs),
    }
    if not paths["qa"].exists():
        warnings.append(_narrative_issue("missing_narrative_qa", "warning", "Narrative QA sidecar is missing.", artifact="qa"))
    else:
        try:
            qa_payload = json.loads(paths["qa"].read_text(encoding="utf-8"))
            if qa_payload.get("schema_version") != NARRATIVE_QA_SCHEMA_VERSION:
                warnings.append(_narrative_issue("unsupported_narrative_qa_schema", "warning", "Unsupported narrative QA schema.", artifact="qa"))
        except Exception as exc:
            warnings.append(_narrative_issue("invalid_narrative_qa_json", "warning", f"Could not read narrative QA: {exc}", artifact="qa"))

    counts = {
        "targets": int(len(targets) if targets is not None else 0),
        "sections": int(len(sections) if sections is not None else 0),
        "claims": int(len(claims) if claims is not None else 0),
        "evidence_sources": int(len(sources) if sources is not None else 0),
        "evidence_refs": int(len(refs) if refs is not None else 0),
        "claim_evidence_links": int(len(links) if links is not None else 0),
        "reviews": int(len(reviews) if reviews is not None else 0),
        "aggregate_only_refs": int(aggregate_only_refs),
    }
    status = "blocked" if blocking_issues else ("warning" if warnings or claim_counts["weak"] or aggregate_only_refs else "passed")
    feature_state = "blocked" if status == "blocked" else ("beta" if status == "warning" else "stable")
    return NarrativeArtifactValidationResult(
        schema_version=NARRATIVE_MANIFEST_SCHEMA_VERSION,
        narrative_id=narrative_id or None,
        status=status,
        narrative_dir=str(narrative_dir),
        manifest_path=str(manifest_path),
        paths={key: _rel(value, result_root) if value.exists() else None for key, value in paths.items()},
        counts=counts,
        claim_counts=claim_counts,
        checks=checks,
        warnings=warnings,
        blocking_issues=blocking_issues,
        feature_state=feature_state,
        created_at_utc=_utc_now(),
    )


def _narrative_empty_frame(columns: set[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=sorted(columns))


def _narrative_source_id(role: str) -> str:
    return _safe_id(f"source_{role}", fallback="source")


def _review_packet_for_narrative(path: str | Path, artifacts: ResultArtifacts) -> tuple[Path | None, dict[str, Any] | None, list[dict[str, Any]]]:
    warnings: list[dict[str, Any]] = []
    packet_path = artifacts.review_packet_paths[0] if artifacts.review_packet_paths else None
    if packet_path is None:
        written = write_cluster_review_packet_artifact(path)
        if written is None:
            warnings.append(_narrative_issue("missing_cluster_review_packet", "warning", "No cluster review packet could be created for narrative scaffolding.", artifact="cluster_review_packet"))
            return None, None, warnings
        packet_path = Path(written["packet_path"])
    validation = validate_cluster_review_packet_artifact(packet_path)
    if validation.status == "blocked":
        warnings.extend(dict(issue) for issue in validation.blocking_issues)
        return packet_path, None, warnings
    try:
        loaded = json.loads(packet_path.read_text(encoding="utf-8"))
    except Exception as exc:
        warnings.append(_narrative_issue("cluster_review_packet_read_failed", "warning", f"Could not read cluster review packet: {exc}", artifact="cluster_review_packet"))
        return packet_path, None, warnings
    return packet_path, loaded if isinstance(loaded, dict) else None, warnings


def _narrative_claim_row(
    *,
    narrative_id: str,
    claim_id: str,
    target_id: str,
    section_id: str,
    claim_type: str,
    claim_text: str,
    support_state: str,
    confidence: float,
    evidence_ref_ids: list[str],
    sort_order: int,
    warning_flags: str = "",
) -> dict[str, Any]:
    return {
        "schema_version": NARRATIVE_CLAIMS_SCHEMA_VERSION,
        "narrative_id": narrative_id,
        "claim_id": claim_id,
        "target_id": target_id,
        "section_id": section_id,
        "claim_type": claim_type,
        "claim_text": claim_text,
        "support_state": support_state,
        "confidence": confidence,
        "evidence_ref_count": len(evidence_ref_ids),
        "text_origin": "deterministic_template",
        "review_state": "not_required",
        "claim_template_id": claim_type,
        "language": "en",
        "sort_order": sort_order,
        "warning_flags": warning_flags,
    }


def write_narrative_evidence_artifacts(
    path: str | Path,
    *,
    output_dir: str | Path | None = None,
    narrative_id: str = "cluster_narrative_evidence_default",
    max_targets: int = 500,
    max_terms_per_claim: int = 3,
    max_works_per_claim: int = 2,
    max_relations_per_claim: int = 2,
) -> dict[str, Any] | None:
    """Write deterministic narrative claim/evidence scaffolds from a cluster review packet."""

    artifacts = infer_result_artifacts(path)
    packet_path, packet, warnings = _review_packet_for_narrative(path, artifacts)
    if not packet:
        return None
    clusters = [row for row in packet.get("clusters", []) if isinstance(row, Mapping)]
    if not clusters:
        return None
    target_dir = Path(output_dir).expanduser().resolve() if output_dir is not None else artifacts.result_root / "narrative"
    target_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "targets": "narrative_targets.parquet",
        "claims": "claims.parquet",
        "evidence_sources": "evidence_sources.parquet",
        "evidence_refs": "evidence_refs.parquet",
        "claim_evidence_links": "claim_evidence_links.parquet",
        "sections": "narrative_sections.parquet",
        "qa": "narrative_qa.json",
    }
    packet_rel = _review_packet_portable_rel(
        packet_path or artifacts.result_root / "review" / "cluster_review_packet.json",
        result_root=artifacts.result_root,
        packet_path=target_dir / "narrative_manifest.json",
    )
    source_by_role: dict[str, dict[str, Any]] = {}
    for row in packet.get("source_artifacts") or []:
        if not isinstance(row, Mapping):
            continue
        role = str(row.get("role") or "").strip()
        source_path = str(row.get("path") or "").strip()
        if role and source_path and not Path(source_path).is_absolute():
            source_by_role.setdefault(role, {"role": role, "path": source_path})
    used_source_roles = {"cluster_review_packet"}
    for cluster in clusters[: max(1, int(max_targets))]:
        for ref in cluster.get("evidence_refs") or []:
            if isinstance(ref, Mapping):
                used_source_roles.add(str(ref.get("source_role") or "cluster_review_packet"))
    source_rows = []
    for role in sorted(used_source_roles):
        source = source_by_role.get(role)
        artifact_path = str(source.get("path")) if source else packet_rel
        source_rows.append(
            {
                "schema_version": NARRATIVE_EVIDENCE_SOURCES_SCHEMA_VERSION,
                "narrative_id": narrative_id,
                "evidence_source_id": _narrative_source_id(role),
                "artifact_ref": role,
                "artifact_role": role,
                "artifact_path": artifact_path,
                "schema_version_ref": None,
                "resolver": "cluster_review_packet_ref" if artifact_path == packet_rel else "result_relative_artifact",
                "source_state": "stable",
            }
        )

    target_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    section_rows: list[dict[str, Any]] = []
    claim_rows: list[dict[str, Any]] = []
    link_rows: list[dict[str, Any]] = []
    for target_index, cluster in enumerate(clusters[: max(1, int(max_targets))], start=1):
        cluster_uid = str(cluster.get("cluster_uid") or f"cluster:{target_index}")
        target_id = _safe_id(f"target_{cluster_uid}", fallback=f"target_{target_index}")
        label = str(cluster.get("label") or cluster.get("short_label") or cluster_uid)
        keywords = [row for row in cluster.get("keyword_evidence") or [] if isinstance(row, Mapping)]
        works = [row for row in cluster.get("representative_works") or [] if isinstance(row, Mapping)]
        relations = [row for row in cluster.get("cooccurrence_evidence") or [] if isinstance(row, Mapping)]
        caveats = [row for row in cluster.get("quality_caveats") or [] if isinstance(row, Mapping)]
        all_refs = [row for row in cluster.get("evidence_refs") or [] if isinstance(row, Mapping)]
        target_rows.append(
            {
                "schema_version": NARRATIVE_TARGETS_SCHEMA_VERSION,
                "narrative_id": narrative_id,
                "target_id": target_id,
                "target_type": "cluster",
                "target_key": cluster_uid,
                "target_label": label,
                "feature_state": "beta" if cluster.get("review_status") == "review_required" else "stable",
                "cluster_uid": cluster_uid,
                "cluster_id": str(cluster.get("cluster_id") or ""),
                "level": str(cluster.get("cluster_level") or ""),
                "doc_count": _coerce_int(cluster.get("doc_count")),
                "keyword_count": len(keywords),
                "evidence_count": len(all_refs),
                "warning_flags": str(cluster.get("review_status") or ""),
            }
        )
        ref_lookup = {str(row.get("evidence_ref_id")): row for row in all_refs if row.get("evidence_ref_id")}
        for ref_id, ref in ref_lookup.items():
            source_role = str(ref.get("source_role") or "cluster_review_packet")
            evidence_type = str(ref.get("evidence_type") or "metric")
            evidence_rows.append(
                {
                    "schema_version": NARRATIVE_EVIDENCE_REFS_SCHEMA_VERSION,
                    "narrative_id": narrative_id,
                    "evidence_ref_id": ref_id,
                    "evidence_source_id": _narrative_source_id(source_role),
                    "evidence_type": "relation" if evidence_type == "cooccurrence" else evidence_type,
                    "entity_type": "cluster",
                    "entity_key": cluster_uid,
                    "locator_type": "aggregate" if bool(ref.get("aggregate_only", True)) else "entity_key",
                    "locator": str(ref.get("entity_key") or ref.get("evidence_label") or ref_id),
                    "evidence_label": str(ref.get("evidence_label") or ref_id),
                    "support_count": None,
                    "cluster_uid": cluster_uid,
                    "aggregate_only": bool(ref.get("aggregate_only", True)),
                    "warning_flags": "",
                }
            )

        def add_claim(section_type: str, title: str, claim_type: str, text: str, ref_ids: list[str], *, state: str, confidence: float, role: str = "primary", flags: str = "") -> None:
            if not ref_ids and state == "supported":
                return
            section_id = _safe_id(f"{target_id}_{section_type}", fallback=f"{target_id}_section")
            claim_id = _safe_id(f"{target_id}_{claim_type}_{len(claim_rows) + 1}", fallback=f"claim_{len(claim_rows) + 1}")
            claim_rows.append(
                _narrative_claim_row(
                    narrative_id=narrative_id,
                    claim_id=claim_id,
                    target_id=target_id,
                    section_id=section_id,
                    claim_type=claim_type,
                    claim_text=text,
                    support_state=state,
                    confidence=confidence,
                    evidence_ref_ids=ref_ids,
                    sort_order=len(claim_rows) + 1,
                    warning_flags=flags,
                )
            )
            for link_index, ref_id in enumerate(ref_ids, start=1):
                link_rows.append(
                    {
                        "schema_version": NARRATIVE_CLAIM_EVIDENCE_LINKS_SCHEMA_VERSION,
                        "narrative_id": narrative_id,
                        "claim_id": claim_id,
                        "evidence_ref_id": ref_id,
                        "evidence_role": role,
                        "link_strength": max(0.0, min(1.0, confidence)),
                        "required": role in {"primary", "caveat"},
                        "sort_order": link_index,
                        "link_reason": claim_type,
                    }
                )
            section = next((row for row in section_rows if row["section_id"] == section_id), None)
            if section is None:
                section_rows.append(
                    {
                        "schema_version": NARRATIVE_SECTIONS_SCHEMA_VERSION,
                        "narrative_id": narrative_id,
                        "section_id": section_id,
                        "target_id": target_id,
                        "section_type": section_type,
                        "section_title": title,
                        "section_state": "beta" if state == "weak" else "stable",
                        "claim_count": 1,
                        "sort_order": len(section_rows) + 1,
                        "artifact_refs": "cluster_review_packet",
                        "warning_flags": flags,
                    }
                )
            else:
                section["claim_count"] = int(section.get("claim_count") or 0) + 1
                if state == "weak":
                    section["section_state"] = "beta"

        term_refs = [str(row.get("evidence_ref_id")) for row in keywords[: max(1, int(max_terms_per_claim))] if row.get("evidence_ref_id")]
        term_labels = [str(row.get("term") or "") for row in keywords[: max(1, int(max_terms_per_claim))] if row.get("term")]
        if term_refs:
            add_claim("identity", "Identity", "identity", f"{label} is represented by top terms: {', '.join(term_labels)}.", term_refs, state="weak", confidence=0.65, flags="aggregate_only")
            add_claim("meaning", "Meaning", "keyword_meaning", f"Representative keyword evidence for {label} includes {', '.join(term_labels)}.", term_refs, state="weak", confidence=0.6, role="supporting", flags="aggregate_only")
        work_refs = [str(row.get("evidence_ref_id")) for row in works[: max(1, int(max_works_per_claim))] if row.get("evidence_ref_id")]
        work_labels = [str(row.get("title") or row.get("uid") or "") for row in works[: max(1, int(max_works_per_claim))] if row.get("title") or row.get("uid")]
        if work_refs:
            add_claim("works", "Representative Works", "representative_work", f"Representative works for {label} include {', '.join(work_labels)}.", work_refs, state="supported", confidence=0.82)
        relation_refs = [str(row.get("evidence_ref_id")) for row in relations[: max(1, int(max_relations_per_claim))] if row.get("evidence_ref_id")]
        relation_labels = [f"{row.get('source')} - {row.get('target')}" for row in relations[: max(1, int(max_relations_per_claim))]]
        if relation_refs:
            add_claim("relations", "Relations", "relation", f"Term co-occurrence evidence for {label} includes {', '.join(relation_labels)}.", relation_refs, state="weak", confidence=0.55, role="supporting", flags="aggregate_only")
        caveat_refs = [str(row.get("evidence_ref_id")) for row in caveats if row.get("evidence_ref_id")]
        if caveat_refs:
            add_claim("limitations", "Limitations", "quality_caveat", f"{label} has quality caveats that should be reviewed before narrative use.", caveat_refs, state="caveat", confidence=0.9, role="caveat")
        if cluster.get("review_status") == "review_required" or not cluster.get("narrative_ready"):
            fallback_refs = caveat_refs or term_refs[:1] or list(ref_lookup.keys())[:1]
            add_claim("limitations", "Limitations", "limitation", f"{label} requires review before it can be treated as a complete narrative.", fallback_refs, state="caveat", confidence=0.75, role="caveat", flags=str(cluster.get("review_status") or "review_required"))

    tables = {
        "targets": pd.DataFrame(target_rows) if target_rows else _narrative_empty_frame(REQUIRED_NARRATIVE_TARGET_COLUMNS),
        "claims": pd.DataFrame(claim_rows) if claim_rows else _narrative_empty_frame(REQUIRED_NARRATIVE_CLAIM_COLUMNS),
        "evidence_sources": pd.DataFrame(source_rows) if source_rows else _narrative_empty_frame(REQUIRED_NARRATIVE_EVIDENCE_SOURCE_COLUMNS),
        "evidence_refs": pd.DataFrame(evidence_rows) if evidence_rows else _narrative_empty_frame(REQUIRED_NARRATIVE_EVIDENCE_REF_COLUMNS),
        "claim_evidence_links": pd.DataFrame(link_rows) if link_rows else _narrative_empty_frame(REQUIRED_NARRATIVE_CLAIM_LINK_COLUMNS),
        "sections": pd.DataFrame(section_rows) if section_rows else _narrative_empty_frame(REQUIRED_NARRATIVE_SECTION_COLUMNS),
    }
    for key, df in tables.items():
        table_path = target_dir / outputs[key]
        table_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(table_path, index=False)

    manifest = {
        "schema_version": NARRATIVE_MANIFEST_SCHEMA_VERSION,
        "narrative_id": narrative_id,
        "title": "Cluster narrative evidence references",
        "result_id": None,
        "narrative_scope": {"target_types": ["cluster"], "cluster_level": "mixed", "max_targets": int(max_targets), "source_packet": packet_rel},
        "claim_policy": {
            "allowed_claim_types": ["identity", "keyword_meaning", "representative_work", "relation", "quality_caveat", "limitation"],
            "minimum_evidence_refs": 1,
            "unsupported_claim_action": "block",
            "contradiction_action": "block",
            "weak_evidence_action": "mark_beta",
        },
        "evidence_policy": {"allowed_source_roles": sorted(used_source_roles), "require_resolvable_refs": True, "allow_aggregate_only": True, "allow_quotes": False},
        "text_policy": {"allowed_origins": ["deterministic_template", "human_review"], "llm_generation_allowed": False, "require_generation_metadata_when_model_generated": True},
        "source_artifacts": [
            {"role": "cluster_review_packet", "path": packet_rel, "required": True},
            *[dict(row) for row in packet.get("source_artifacts") or [] if isinstance(row, Mapping)],
        ],
        "rule_sets": [],
        "transforms": [
            {"step": "load_cluster_review_packet"},
            {"step": "collect_narrative_targets"},
            {"step": "collect_evidence_refs"},
            {"step": "build_deterministic_claim_scaffold"},
            {"step": "link_claim_evidence"},
            {"step": "validate_claim_support"},
        ],
        "outputs": outputs,
        "created_at_utc": _utc_now(),
        "warnings": warnings,
    }
    manifest_path = target_dir / "narrative_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    validation = validate_narrative_artifact(manifest_path)
    qa = {
        "schema_version": NARRATIVE_QA_SCHEMA_VERSION,
        "narrative_id": narrative_id,
        "status": validation.status,
        "checks": validation.checks,
        "counts": validation.counts,
        "claim_counts": validation.claim_counts,
        "unsupported_claims": [],
        "warnings": validation.warnings,
        "blocking_issues": validation.blocking_issues,
        "feature_state": validation.feature_state,
        "created_at_utc": _utc_now(),
    }
    qa_path = target_dir / outputs["qa"]
    qa_path.write_text(json.dumps(qa, indent=2, sort_keys=True), encoding="utf-8")
    validation = validate_narrative_artifact(manifest_path)
    return {
        "schema_version": NARRATIVE_MANIFEST_SCHEMA_VERSION,
        "narrative_id": narrative_id,
        "manifest_path": manifest_path,
        "qa_path": qa_path,
        "validation": validation.to_dict(),
        "counts": validation.counts,
        "feature_state": validation.feature_state,
    }


def _matrix_issue(
    code: str,
    severity: str,
    message: str,
    *,
    artifact: str | None = None,
) -> dict[str, Any]:
    issue = {"code": code, "severity": severity, "message": message}
    if artifact:
        issue["artifact"] = artifact
    return issue


def _matrix_dir_and_manifest(path: str | Path) -> tuple[Path, Path]:
    candidate = Path(path).expanduser()
    if candidate.exists():
        candidate = candidate.resolve()
    if candidate.is_file():
        return candidate.parent, candidate
    return candidate, candidate / "matrix_manifest.json"


def _matrix_result_root(matrix_dir: Path) -> Path:
    if matrix_dir.parent.name == "matrices":
        scope_root = matrix_dir.parent.parent
        if scope_root.name.startswith("landscape") and scope_root.parent.exists():
            return scope_root.parent
        return scope_root
    return matrix_dir.parent


def _matrix_output_paths(matrix_dir: Path, manifest: Mapping[str, Any]) -> dict[str, Path]:
    outputs = manifest.get("outputs") if isinstance(manifest.get("outputs"), Mapping) else {}
    return {
        "values": matrix_dir / str(outputs.get("values") or "matrix_values.parquet"),
        "rows": matrix_dir / str(outputs.get("rows") or "row_entities.parquet"),
        "columns": matrix_dir / str(outputs.get("columns") or "column_entities.parquet"),
        "qa": matrix_dir / str(outputs.get("qa") or "matrix_qa.json"),
    }


def _matrix_path_payload(paths: Mapping[str, Path], matrix_dir: Path) -> dict[str, str | None]:
    return {key: _rel(path, matrix_dir) for key, path in paths.items()}


def _matrix_required_fields(manifest: Mapping[str, Any]) -> set[str]:
    required = {
        "schema_version",
        "matrix_id",
        "title",
        "matrix_family",
        "format",
        "row_entity_type",
        "column_entity_type",
        "shape",
        "value",
        "weighting",
        "source_artifacts",
        "outputs",
        "created_at_utc",
    }
    return {key for key in required if manifest.get(key) in (None, "")}


def _matrix_read_parquet(
    path: Path,
    *,
    artifact: str,
    warnings: list[dict[str, Any]],
    blocking_issues: list[dict[str, Any]],
) -> pd.DataFrame | None:
    if not path.exists():
        blocking_issues.append(
            _matrix_issue(
                "missing_matrix_table",
                "blocking",
                f"Missing matrix {artifact} table.",
                artifact=artifact,
            )
        )
        return None
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        blocking_issues.append(
            _matrix_issue(
                "invalid_matrix_parquet",
                "blocking",
                f"Could not read matrix {artifact} parquet: {exc}",
                artifact=artifact,
            )
        )
        return None


def _matrix_missing_columns(
    df: pd.DataFrame | None,
    required: set[str],
    *,
    artifact: str,
    blocking_issues: list[dict[str, Any]],
) -> set[str]:
    columns = set(df.columns) if df is not None else set()
    missing = required - columns
    if missing:
        blocking_issues.append(
            _matrix_issue(
                "missing_matrix_columns",
                "blocking",
                f"Missing required matrix columns: {sorted(missing)}",
                artifact=artifact,
            )
        )
    return missing


def _matrix_entity_checks(
    df: pd.DataFrame | None,
    *,
    manifest: Mapping[str, Any],
    axis: str,
    warnings: list[dict[str, Any]],
    blocking_issues: list[dict[str, Any]],
) -> dict[str, Any]:
    if df is None or _matrix_missing_columns(df, REQUIRED_MATRIX_ENTITY_COLUMNS, artifact=axis, blocking_issues=blocking_issues):
        return {"status": "blocked", "count": 0}
    entity_type = str(manifest.get(f"{axis}_entity_type") or "")
    if not set(df["schema_version"]) <= {MATRIX_ENTITIES_SCHEMA_VERSION}:
        blocking_issues.append(
            _matrix_issue(
                "unsupported_matrix_entities_schema",
                "blocking",
                f"Unsupported {axis} entity schema.",
                artifact=axis,
            )
        )
    if not set(df["matrix_id"].map(str)) <= {str(manifest.get("matrix_id"))}:
        blocking_issues.append(
            _matrix_issue(
                "matrix_entity_id_mismatch",
                "blocking",
                f"{axis} entity matrix_id does not match manifest.",
                artifact=axis,
            )
        )
    if entity_type and not set(df["entity_type"].map(str)) <= {entity_type}:
        warnings.append(
            _matrix_issue(
                "matrix_entity_type_mismatch",
                "warning",
                f"{axis} entity_type values do not all match the manifest.",
                artifact=axis,
            )
        )
    duplicate_keys = int(df["entity_key"].duplicated().sum())
    duplicate_indices = int(df["entity_index"].duplicated().sum())
    if duplicate_keys:
        blocking_issues.append(
            _matrix_issue("duplicate_matrix_entity_keys", "blocking", f"{axis} entity keys are duplicated.", artifact=axis)
        )
    if duplicate_indices:
        blocking_issues.append(
            _matrix_issue(
                "duplicate_matrix_entity_indices",
                "blocking",
                f"{axis} entity indices are duplicated.",
                artifact=axis,
            )
        )
    try:
        indices = sorted(int(value) for value in df["entity_index"].tolist())
    except Exception:
        blocking_issues.append(
            _matrix_issue("invalid_matrix_entity_index", "blocking", f"{axis} entity_index must be integer-like.", artifact=axis)
        )
        indices = []
    if indices and indices != list(range(len(indices))):
        warnings.append(
            _matrix_issue(
                "non_contiguous_matrix_entity_index",
                "warning",
                f"{axis} entity_index is not contiguous from zero.",
                artifact=axis,
            )
        )
    return {"status": "passed", "count": int(len(df))}


def _matrix_values_checks(
    values: pd.DataFrame | None,
    rows: pd.DataFrame | None,
    columns: pd.DataFrame | None,
    *,
    manifest: Mapping[str, Any],
    warnings: list[dict[str, Any]],
    blocking_issues: list[dict[str, Any]],
) -> dict[str, Any]:
    if values is None or _matrix_missing_columns(
        values,
        REQUIRED_MATRIX_VALUES_COLUMNS,
        artifact="values",
        blocking_issues=blocking_issues,
    ):
        return {"status": "blocked", "count": 0}
    if not set(values["schema_version"]) <= {MATRIX_VALUES_SCHEMA_VERSION}:
        blocking_issues.append(
            _matrix_issue(
                "unsupported_matrix_values_schema",
                "blocking",
                "Unsupported matrix values schema.",
                artifact="values",
            )
        )
    matrix_id = str(manifest.get("matrix_id"))
    if not set(values["matrix_id"].map(str)) <= {matrix_id}:
        blocking_issues.append(
            _matrix_issue(
                "matrix_values_id_mismatch",
                "blocking",
                "Values matrix_id does not match manifest.",
                artifact="values",
            )
        )
    numeric = pd.to_numeric(values["value"], errors="coerce")
    if numeric.isna().any() or (~numeric.map(lambda value: math.isfinite(float(value)))).any():
        blocking_issues.append(
            _matrix_issue(
                "invalid_matrix_values",
                "blocking",
                "Matrix values must be finite numeric values.",
                artifact="values",
            )
        )
    if rows is not None and "entity_key" in rows.columns:
        missing_rows = set(values["row_key"].map(str)) - set(rows["entity_key"].map(str))
        if missing_rows:
            blocking_issues.append(
                _matrix_issue(
                    "missing_matrix_row_refs",
                    "blocking",
                    f"{len(missing_rows)} value row_key refs are missing from row entities.",
                    artifact="values",
                )
            )
    if columns is not None and "entity_key" in columns.columns:
        missing_columns = set(values["column_key"].map(str)) - set(columns["entity_key"].map(str))
        if missing_columns:
            blocking_issues.append(
                _matrix_issue(
                    "missing_matrix_column_refs",
                    "blocking",
                    f"{len(missing_columns)} value column_key refs are missing from column entities.",
                    artifact="values",
                )
            )
    duplicate_keys = ["row_key", "column_key"]
    if "period" in values.columns:
        duplicate_keys.append("period")
    if not manifest.get("allow_duplicate_cells"):
        duplicates = int(values.duplicated(subset=duplicate_keys).sum())
        if duplicates:
            blocking_issues.append(
                _matrix_issue(
                    "duplicate_matrix_cells",
                    "blocking",
                    f"{duplicates} duplicate matrix cells were found.",
                    artifact="values",
                )
            )
    return {"status": "passed", "count": int(len(values))}


def _matrix_shape_checks(
    *,
    manifest: Mapping[str, Any],
    values: pd.DataFrame | None,
    rows: pd.DataFrame | None,
    columns: pd.DataFrame | None,
    warnings: list[dict[str, Any]],
    blocking_issues: list[dict[str, Any]],
) -> dict[str, Any]:
    shape = manifest.get("shape") if isinstance(manifest.get("shape"), Mapping) else {}
    expected = {
        "rows": _coerce_int(shape.get("rows")),
        "columns": _coerce_int(shape.get("columns")),
        "nnz": _coerce_int(shape.get("nnz")),
    }
    actual = {
        "rows": 0 if rows is None else int(len(rows)),
        "columns": 0 if columns is None else int(len(columns)),
        "nnz": 0 if values is None else int(len(values)),
    }
    mismatches = [key for key, value in expected.items() if value is not None and value != actual[key]]
    if mismatches:
        blocking_issues.append(
            _matrix_issue(
                "matrix_shape_mismatch",
                "blocking",
                f"Manifest shape does not match table counts for: {mismatches}.",
                artifact="manifest",
            )
        )
    if expected["rows"] is None or expected["columns"] is None or expected["nnz"] is None:
        warnings.append(
            _matrix_issue(
                "incomplete_matrix_shape",
                "warning",
                "Manifest shape should record rows, columns, and nnz.",
                artifact="manifest",
            )
        )
    return {"status": "passed" if not mismatches else "blocked", "expected": expected, "actual": actual}


def _matrix_symmetry_checks(
    *,
    manifest: Mapping[str, Any],
    values: pd.DataFrame | None,
    warnings: list[dict[str, Any]],
    blocking_issues: list[dict[str, Any]],
) -> dict[str, Any]:
    weighting = manifest.get("weighting") if isinstance(manifest.get("weighting"), Mapping) else {}
    if not weighting.get("symmetric"):
        return {"status": "skipped", "reason": "matrix is not declared symmetric"}
    if manifest.get("row_entity_type") != manifest.get("column_entity_type"):
        blocking_issues.append(
            _matrix_issue(
                "symmetric_matrix_entity_type_mismatch",
                "blocking",
                "Symmetric matrices must use the same row and column entity type.",
                artifact="manifest",
            )
        )
    shape = manifest.get("shape") if isinstance(manifest.get("shape"), Mapping) else {}
    if shape.get("rows") != shape.get("columns"):
        blocking_issues.append(
            _matrix_issue(
                "symmetric_matrix_shape_mismatch",
                "blocking",
                "Symmetric matrices must be square.",
                artifact="manifest",
            )
        )
    if values is None or not {"row_index", "column_index", "value"}.issubset(values.columns):
        return {"status": "blocked"}
    storage = str(weighting.get("storage") or "upper_triangle")
    row_index = pd.to_numeric(values["row_index"], errors="coerce")
    column_index = pd.to_numeric(values["column_index"], errors="coerce")
    if storage == "upper_triangle":
        lower_rows = int((row_index > column_index).sum())
        if lower_rows:
            blocking_issues.append(
                _matrix_issue(
                    "invalid_upper_triangle_storage",
                    "blocking",
                    "Upper-triangle symmetric matrices cannot contain row_index > column_index cells.",
                    artifact="values",
                )
            )
        return {"status": "passed", "storage": storage}
    if storage in {"both_directions", "full"} and not values.empty:
        lookup = {
            (str(row.row_key), str(row.column_key)): float(row.value)
            for row in values.itertuples(index=False)
        }
        missing_reverse = 0
        mismatched_reverse = 0
        for (row_key, column_key), value in lookup.items():
            if row_key == column_key:
                continue
            reverse = lookup.get((column_key, row_key))
            if reverse is None:
                missing_reverse += 1
            elif not math.isclose(value, reverse, rel_tol=1e-9, abs_tol=1e-12):
                mismatched_reverse += 1
        if missing_reverse or mismatched_reverse:
            blocking_issues.append(
                _matrix_issue(
                    "symmetric_matrix_reverse_mismatch",
                    "blocking",
                    "Both-direction symmetric matrix storage has missing or mismatched reverse cells.",
                    artifact="values",
                )
            )
        return {
            "status": "passed" if not missing_reverse and not mismatched_reverse else "blocked",
            "storage": storage,
            "missing_reverse": missing_reverse,
            "mismatched_reverse": mismatched_reverse,
        }
    warnings.append(
        _matrix_issue(
            "unknown_symmetric_matrix_storage",
            "warning",
            f"Unknown symmetric matrix storage mode: {storage}",
            artifact="manifest",
        )
    )
    return {"status": "warning", "storage": storage}


def _matrix_source_checks(
    *,
    result_root: Path,
    manifest: Mapping[str, Any],
    warnings: list[dict[str, Any]],
    blocking_issues: list[dict[str, Any]],
) -> dict[str, Any]:
    source_artifacts = manifest.get("source_artifacts")
    if not isinstance(source_artifacts, list) or not source_artifacts:
        warnings.append(
            _matrix_issue(
                "missing_matrix_source_artifacts",
                "warning",
                "Matrix manifest should record at least one source artifact.",
                artifact="manifest",
            )
        )
        return {"status": "warning", "count": 0}
    missing = 0
    for source in source_artifacts:
        if not isinstance(source, Mapping):
            warnings.append(
                _matrix_issue(
                    "invalid_matrix_source_artifact",
                    "warning",
                    "Matrix source artifact refs should be objects.",
                    artifact="manifest",
                )
            )
            continue
        path = source.get("path")
        if not path:
            warnings.append(
                _matrix_issue(
                    "missing_matrix_source_path",
                    "warning",
                    "Matrix source artifact ref has no path.",
                    artifact="manifest",
                )
            )
            continue
        source_path = Path(str(path))
        resolved = source_path if source_path.is_absolute() else result_root / source_path
        if not resolved.exists():
            missing += 1
    if missing:
        warnings.append(
            _matrix_issue(
                "missing_matrix_source_artifact",
                "warning",
                f"{missing} matrix source artifact refs do not exist.",
                artifact="manifest",
            )
        )
    return {"status": "warning" if missing else "passed", "count": len(source_artifacts), "missing": missing}


def validate_matrix_artifact(path: str | Path) -> MatrixArtifactValidationResult:
    """Validate a general sparse-triplet matrix artifact."""

    matrix_dir, manifest_path = _matrix_dir_and_manifest(path)
    result_root = _matrix_result_root(matrix_dir)
    warnings: list[dict[str, Any]] = []
    blocking_issues: list[dict[str, Any]] = []
    checks: dict[str, dict[str, Any]] = {}
    manifest: dict[str, Any] = {}

    if not manifest_path.exists():
        blocking_issues.append(
            _matrix_issue(
                "missing_matrix_manifest",
                "blocking",
                "Missing matrix_manifest.json.",
                artifact="manifest",
            )
        )
    else:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            blocking_issues.append(
                _matrix_issue(
                    "invalid_matrix_manifest_json",
                    "blocking",
                    f"Could not read matrix manifest: {exc}",
                    artifact="manifest",
                )
            )
        if not isinstance(manifest, dict):
            blocking_issues.append(
                _matrix_issue(
                    "invalid_matrix_manifest_shape",
                    "blocking",
                    "Matrix manifest must be a JSON object.",
                    artifact="manifest",
                )
            )
            manifest = {}

    if manifest:
        if manifest.get("schema_version") != MATRIX_MANIFEST_SCHEMA_VERSION:
            blocking_issues.append(
                _matrix_issue(
                    "unsupported_matrix_manifest_schema",
                    "blocking",
                    f"Unsupported matrix manifest schema: {manifest.get('schema_version')}",
                    artifact="manifest",
                )
            )
        if manifest.get("format") != "sparse_triplet":
            blocking_issues.append(
                _matrix_issue(
                    "unsupported_matrix_format",
                    "blocking",
                    f"Unsupported matrix format: {manifest.get('format')}",
                    artifact="manifest",
                )
            )
        if manifest.get("matrix_family") not in SUPPORTED_MATRIX_FAMILIES:
            blocking_issues.append(
                _matrix_issue(
                    "unsupported_matrix_family",
                    "blocking",
                    f"Unsupported matrix family: {manifest.get('matrix_family')}",
                    artifact="manifest",
                )
            )
        missing_fields = _matrix_required_fields(manifest)
        if missing_fields:
            blocking_issues.append(
                _matrix_issue(
                    "missing_matrix_manifest_fields",
                    "blocking",
                    f"Missing matrix manifest fields: {sorted(missing_fields)}",
                    artifact="manifest",
                )
            )
    checks["manifest"] = {
        "status": "blocked" if blocking_issues else "passed",
        "schema_version": manifest.get("schema_version"),
    }

    paths = _matrix_output_paths(matrix_dir, manifest)
    rows = _matrix_read_parquet(paths["rows"], artifact="rows", warnings=warnings, blocking_issues=blocking_issues)
    columns = _matrix_read_parquet(
        paths["columns"],
        artifact="columns",
        warnings=warnings,
        blocking_issues=blocking_issues,
    )
    values = _matrix_read_parquet(
        paths["values"],
        artifact="values",
        warnings=warnings,
        blocking_issues=blocking_issues,
    )
    checks["rows"] = _matrix_entity_checks(
        rows,
        manifest=manifest,
        axis="row",
        warnings=warnings,
        blocking_issues=blocking_issues,
    )
    checks["columns"] = _matrix_entity_checks(
        columns,
        manifest=manifest,
        axis="column",
        warnings=warnings,
        blocking_issues=blocking_issues,
    )
    checks["values"] = _matrix_values_checks(
        values,
        rows,
        columns,
        manifest=manifest,
        warnings=warnings,
        blocking_issues=blocking_issues,
    )
    checks["shape"] = _matrix_shape_checks(
        manifest=manifest,
        values=values,
        rows=rows,
        columns=columns,
        warnings=warnings,
        blocking_issues=blocking_issues,
    )
    checks["symmetry"] = _matrix_symmetry_checks(
        manifest=manifest,
        values=values,
        warnings=warnings,
        blocking_issues=blocking_issues,
    )
    checks["sources"] = _matrix_source_checks(
        result_root=result_root,
        manifest=manifest,
        warnings=warnings,
        blocking_issues=blocking_issues,
    )

    qa_path = paths["qa"]
    if not qa_path.exists():
        warnings.append(
            _matrix_issue(
                "missing_matrix_qa_sidecar",
                "warning",
                "matrix_qa.json is missing; writers should persist the validation result.",
                artifact="qa",
            )
        )

    counts = {
        "rows": int(0 if rows is None else len(rows)),
        "columns": int(0 if columns is None else len(columns)),
        "nnz": int(0 if values is None else len(values)),
        "warnings": len(warnings),
        "blocking_issues": len(blocking_issues),
    }
    if blocking_issues:
        status = "blocked"
    elif warnings:
        status = "warning"
    else:
        status = "passed"

    return MatrixArtifactValidationResult(
        schema_version=MATRIX_QA_SCHEMA_VERSION,
        matrix_id=str(manifest.get("matrix_id")) if manifest.get("matrix_id") else None,
        matrix_family=str(manifest.get("matrix_family")) if manifest.get("matrix_family") else None,
        status=status,
        matrix_dir=str(matrix_dir),
        manifest_path=_rel(manifest_path, matrix_dir) or str(manifest_path),
        paths=_matrix_path_payload(paths, matrix_dir),
        counts=counts,
        checks=checks,
        warnings=warnings,
        blocking_issues=blocking_issues,
        created_at_utc=_utc_now(),
    )


def _matrix_entity_type(df: pd.DataFrame, fallback: str) -> str:
    if "entity_type" not in df.columns or df.empty:
        return fallback
    values = [str(value) for value in df["entity_type"].dropna().unique().tolist()]
    return values[0] if values else fallback


def _matrix_existing_result_id(result_root: Path) -> str | None:
    manifest_path = find_result_manifest_path(result_root)
    if manifest_path is None:
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    result_id = manifest.get("result_id")
    return str(result_id) if result_id else None


def write_matrix_artifact(
    result_root: str | Path,
    matrix_id: str,
    matrix_family: str,
    values_df: pd.DataFrame,
    row_entities_df: pd.DataFrame,
    column_entities_df: pd.DataFrame,
    *,
    value_spec: Mapping[str, Any],
    weighting: Mapping[str, Any],
    source_artifacts: list[Mapping[str, Any]],
    rule_sets: list[Mapping[str, Any]] | None = None,
    transforms: list[Mapping[str, Any]] | None = None,
    title: str | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Write a general sparse-triplet matrix artifact and QA sidecar."""

    root = Path(result_root).expanduser().resolve()
    matrix_id = _safe_id(matrix_id, fallback="matrix")
    if matrix_family not in SUPPORTED_MATRIX_FAMILIES:
        raise ValueError(f"unsupported matrix family: {matrix_family}")
    matrix_dir = Path(output_dir).expanduser().resolve() if output_dir else root / "matrices" / matrix_id
    matrix_dir.mkdir(parents=True, exist_ok=True)

    values = values_df.copy()
    rows = row_entities_df.copy()
    columns = column_entities_df.copy()
    values["schema_version"] = MATRIX_VALUES_SCHEMA_VERSION
    values["matrix_id"] = matrix_id
    rows["schema_version"] = MATRIX_ENTITIES_SCHEMA_VERSION
    rows["matrix_id"] = matrix_id
    columns["schema_version"] = MATRIX_ENTITIES_SCHEMA_VERSION
    columns["matrix_id"] = matrix_id

    row_entity_type = _matrix_entity_type(rows, "row")
    column_entity_type = _matrix_entity_type(columns, "column")
    shape = {"rows": int(len(rows)), "columns": int(len(columns)), "nnz": int(len(values))}
    outputs = {
        "values": "matrix_values.parquet",
        "rows": "row_entities.parquet",
        "columns": "column_entities.parquet",
        "qa": "matrix_qa.json",
    }
    manifest = {
        "schema_version": MATRIX_MANIFEST_SCHEMA_VERSION,
        "matrix_id": matrix_id,
        "title": title or matrix_id.replace("_", " ").title(),
        "matrix_family": matrix_family,
        "format": "sparse_triplet",
        "result_id": _matrix_existing_result_id(root),
        "row_entity_type": row_entity_type,
        "column_entity_type": column_entity_type,
        "shape": shape,
        "value": dict(value_spec),
        "weighting": dict(weighting),
        "source_artifacts": [dict(item) for item in source_artifacts],
        "rule_sets": [dict(item) for item in (rule_sets or [])],
        "transforms": [dict(item) for item in (transforms or [])],
        "outputs": outputs,
        "created_at_utc": _utc_now(),
        "warnings": [],
    }

    values.to_parquet(matrix_dir / outputs["values"], index=False)
    rows.to_parquet(matrix_dir / outputs["rows"], index=False)
    columns.to_parquet(matrix_dir / outputs["columns"], index=False)
    (matrix_dir / "matrix_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    validation = validate_matrix_artifact(matrix_dir)
    qa_payload = validation.to_dict()
    qa_payload["warnings"] = [
        warning for warning in qa_payload["warnings"] if warning.get("code") != "missing_matrix_qa_sidecar"
    ]
    qa_payload["counts"]["warnings"] = len(qa_payload["warnings"])
    if qa_payload["status"] == "warning" and not qa_payload["warnings"]:
        qa_payload["status"] = "passed"
    (matrix_dir / outputs["qa"]).write_text(
        json.dumps(qa_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    validation = validate_matrix_artifact(matrix_dir)
    return {
        "schema_version": MATRIX_MANIFEST_SCHEMA_VERSION,
        "matrix_id": matrix_id,
        "matrix_dir": matrix_dir,
        "manifest_path": matrix_dir / "matrix_manifest.json",
        "values_path": matrix_dir / outputs["values"],
        "row_entities_path": matrix_dir / outputs["rows"],
        "column_entities_path": matrix_dir / outputs["columns"],
        "qa_path": matrix_dir / outputs["qa"],
        "qa": validation.to_dict(),
    }


def write_matrix_from_term_cooccurrence(
    path: str | Path,
    *,
    matrix_id: str = "term_cooccurrence_default",
) -> dict[str, Any] | None:
    """Wrap P1.5 term co-occurrence sidecars as a general matrix artifact."""

    artifacts = infer_result_artifacts(path)
    landscape = artifacts.landscape_dir
    if landscape is None:
        return None
    cooc_path = landscape / "term_cooccurrence.parquet"
    map_path = landscape / "term_cooccurrence_map.json"
    if not cooc_path.exists():
        written = write_cooccurrence_artifacts(path)
        if written is None:
            return None
        cooc_path = Path(written["table_path"])
        map_path = Path(written["map_path"])
    cooc = pd.read_parquet(cooc_path)
    if cooc.empty:
        return None
    if not {"source", "target", "weight"}.issubset(cooc.columns):
        raise ValueError("term co-occurrence table must include source, target, and weight columns")

    terms = sorted(set(cooc["source"].map(str)) | set(cooc["target"].map(str)))
    term_to_index = {term: index for index, term in enumerate(terms)}
    pair_rows: dict[tuple[str, str], dict[str, Any]] = {}
    for item in cooc.itertuples(index=False):
        source = str(getattr(item, "source"))
        target = str(getattr(item, "target"))
        if source == target:
            left, right = source, target
        else:
            left, right = sorted((source, target), key=lambda term: term_to_index[term])
        key = (left, right)
        raw_weight = float(getattr(item, "weight"))
        support = _coerce_int(getattr(item, "count", None)) or 1
        row = pair_rows.setdefault(
            key,
            {
                "row_key": left,
                "column_key": right,
                "raw_value": 0.0,
                "support_count": 0,
                "relation": "term_cooccurrence",
            },
        )
        row["raw_value"] += raw_weight
        row["support_count"] += support

    max_raw = max((float(row["raw_value"]) for row in pair_rows.values()), default=1.0) or 1.0
    matrix_rows = []
    for row in pair_rows.values():
        row_key = str(row["row_key"])
        column_key = str(row["column_key"])
        matrix_rows.append(
            {
                "row_key": row_key,
                "column_key": column_key,
                "row_index": int(term_to_index[row_key]),
                "column_index": int(term_to_index[column_key]),
                "value": float(row["raw_value"]) / float(max_raw),
                "raw_value": float(row["raw_value"]),
                "support_count": int(row["support_count"]),
                "relation": row["relation"],
            }
        )
    matrix_rows.sort(key=lambda row: (int(row["row_index"]), -float(row["value"]), int(row["column_index"])))
    row_ranks: dict[str, int] = {}
    for row in matrix_rows:
        row_key = str(row["row_key"])
        row_ranks[row_key] = row_ranks.get(row_key, 0) + 1
        row["rank"] = row_ranks[row_key]

    entities = pd.DataFrame(
        {
            "entity_key": terms,
            "entity_index": list(range(len(terms))),
            "entity_type": ["term"] * len(terms),
            "label": terms,
            "term": terms,
        }
    )
    values = pd.DataFrame(matrix_rows)
    source_artifacts = [{"role": "cooccurrence", "path": _rel(cooc_path, artifacts.result_root)}]
    if map_path.exists():
        source_artifacts.append({"role": "cooccurrence_map", "path": _rel(map_path, artifacts.result_root)})
    if artifacts.keywords_path:
        source_artifacts.append({"role": "keywords", "path": _rel(artifacts.keywords_path, artifacts.result_root)})
    return write_matrix_artifact(
        artifacts.result_root,
        matrix_id,
        "cooccurrence",
        values,
        entities,
        entities.copy(),
        value_spec={
            "name": "cooccurrence_weight",
            "type": "float",
            "range": [0.0, 1.0],
            "interpretation": "normalized term co-occurrence strength",
        },
        weighting={
            "raw_metric": "term_cooccurrence",
            "normalization": "max",
            "threshold": 0.0,
            "symmetric": True,
            "storage": "upper_triangle",
        },
        source_artifacts=source_artifacts,
        transforms=[
            {"step": "load_term_cooccurrence"},
            {"step": "aggregate_term_pairs"},
            {"step": "normalize_by_max_raw_value"},
            {"step": "build_sparse_triplets"},
        ],
        title="Term co-occurrence matrix",
    )


def _keyword_rule_issue(
    code: str,
    severity: str,
    message: str,
    artifact: str = "keyword_rules",
) -> dict[str, Any]:
    return {"code": code, "severity": severity, "message": message, "artifact": artifact}


def _keyword_rule_dir_and_manifest(path: str | Path) -> tuple[Path, Path]:
    raw_path = Path(path).expanduser().resolve()
    if raw_path.is_dir():
        return raw_path, raw_path / "rule_set_manifest.json"
    return raw_path.parent, raw_path


def _keyword_rule_output_paths(manifest: Mapping[str, Any], rule_dir: Path) -> dict[str, Path | None]:
    outputs = manifest.get("outputs")
    outputs = outputs if isinstance(outputs, Mapping) else {}
    defaults = {
        "rules": "rules.parquet",
        "applications": "rule_applications.parquet",
        "before_after": "term_before_after.parquet",
        "impact_summary": "rule_impact_summary.json",
        "qa": "rule_set_qa.json",
    }
    paths: dict[str, Path | None] = {}
    for key, default in defaults.items():
        name = outputs.get(key, default)
        paths[key] = rule_dir / str(name) if name else None
    return paths


def _keyword_rule_read_parquet(
    path: Path | None,
    table_name: str,
    blocking_issues: list[dict[str, Any]],
) -> pd.DataFrame:
    if path is None or not path.exists():
        blocking_issues.append(
            _keyword_rule_issue(
                f"missing_keyword_rule_{table_name}",
                "blocking",
                f"Keyword rule artifact is missing `{table_name}` table.",
            )
        )
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        blocking_issues.append(
            _keyword_rule_issue(
                f"unreadable_keyword_rule_{table_name}",
                "blocking",
                f"Could not read keyword rule `{table_name}` table: {exc}",
            )
        )
        return pd.DataFrame()


def _keyword_rule_missing_columns(df: pd.DataFrame, required: set[str]) -> list[str]:
    return sorted(required - set(df.columns))


def _keyword_rule_table_check(
    *,
    df: pd.DataFrame,
    table_name: str,
    required: set[str],
    expected_schema: str,
    blocking_issues: list[dict[str, Any]],
) -> dict[str, Any]:
    missing = _keyword_rule_missing_columns(df, required)
    if missing:
        blocking_issues.append(
            _keyword_rule_issue(
                f"missing_keyword_rule_{table_name}_columns",
                "blocking",
                f"Keyword rule `{table_name}` table is missing columns: {', '.join(missing)}.",
            )
        )
    invalid_schema_rows = 0
    if "schema_version" in df.columns:
        invalid_schema_rows = int((df["schema_version"].dropna().map(str) != expected_schema).sum())
        if invalid_schema_rows:
            blocking_issues.append(
                _keyword_rule_issue(
                    f"unsupported_keyword_rule_{table_name}_schema",
                    "blocking",
                    f"{invalid_schema_rows} `{table_name}` rows use an unsupported schema version.",
                )
            )
    return {
        "rows": int(len(df)),
        "required_columns": sorted(required),
        "missing_columns": missing,
        "invalid_schema_rows": invalid_schema_rows,
    }


def _keyword_rule_truthy(value: Any) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y", "block", "blocked"}


def _keyword_rule_split_ids(value: Any) -> set[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return set()
    if isinstance(value, (list, tuple, set)):
        return {str(item).strip() for item in value if str(item).strip()}
    text = str(value).strip()
    if not text:
        return set()
    return {part.strip() for part in re.split(r"[|,;]", text) if part.strip()}


def _keyword_rule_term_values(row: pd.Series) -> list[str]:
    values: list[str] = []
    for column in ("display_label", "term_after", "normalized_term_after", "term_before", "raw_term"):
        if column not in row.index:
            continue
        value = row.get(column)
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        text = str(value).strip()
        if text and text not in values:
            values.append(text)
    return values


def _keyword_rule_policy_checks(
    *,
    rules: pd.DataFrame,
    applications: pd.DataFrame,
    before_after: pd.DataFrame,
    warnings: list[dict[str, Any]],
    blocking_issues: list[dict[str, Any]],
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    known_rule_ids: set[str] = set()

    if "rule_id" in rules.columns:
        rule_ids = rules["rule_id"].dropna().map(str).map(str.strip)
        blank_rule_rows = int((rule_ids == "").sum())
        if blank_rule_rows:
            blocking_issues.append(
                _keyword_rule_issue(
                    "blank_keyword_rule_id",
                    "blocking",
                    f"{blank_rule_rows} keyword rule rows have blank rule IDs.",
                )
            )
        rule_ids = rule_ids[rule_ids != ""]
        known_rule_ids = set(rule_ids)
        duplicate_rule_ids = sorted(rule_ids[rule_ids.duplicated()].unique().tolist())
        if duplicate_rule_ids:
            blocking_issues.append(
                _keyword_rule_issue(
                    "duplicate_keyword_rule_ids",
                    "blocking",
                    f"Duplicate keyword rule IDs found: {', '.join(duplicate_rule_ids[:10])}.",
                )
            )
        checks["duplicate_rule_ids"] = duplicate_rule_ids
        checks["blank_rule_id_rows"] = blank_rule_rows

    if "rule_family" in rules.columns:
        invalid_families = sorted(set(rules["rule_family"].dropna().map(str)) - KEYWORD_RULE_FAMILIES)
        if invalid_families:
            blocking_issues.append(
                _keyword_rule_issue(
                    "unsupported_keyword_rule_family",
                    "blocking",
                    f"Unsupported keyword rule families: {', '.join(invalid_families)}.",
                )
            )
        checks["invalid_rule_families"] = invalid_families

    if "action" in rules.columns:
        invalid_actions = sorted(set(rules["action"].dropna().map(str)) - KEYWORD_RULE_ACTIONS)
        if invalid_actions:
            blocking_issues.append(
                _keyword_rule_issue(
                    "unsupported_keyword_rule_action",
                    "blocking",
                    f"Unsupported keyword rule actions: {', '.join(invalid_actions)}.",
                )
            )
        checks["invalid_actions"] = invalid_actions

    unsafe_ids: list[str] = []
    if len(rules) and {"rule_id", "rule_family", "action", "destructive"} <= set(rules.columns):
        unsafe = rules[
            rules.apply(
                lambda row: (
                    (str(row.get("action")) == "block" or _keyword_rule_truthy(row.get("destructive")))
                    and str(row.get("rule_family")) not in KEYWORD_RULE_BLOCK_FAMILIES
                ),
                axis=1,
            )
        ]
        unsafe_ids = unsafe["rule_id"].dropna().map(str).head(20).tolist()
        if unsafe_ids:
            blocking_issues.append(
                _keyword_rule_issue(
                    "unsafe_keyword_rule_block_action",
                    "blocking",
                    (
                        "Destructive keyword block rules are limited to artifact, metadata, "
                        f"HTML, and LaTeX families. Unsafe rule IDs: {', '.join(unsafe_ids)}."
                    ),
                )
            )
    checks["unsafe_block_rule_ids"] = unsafe_ids

    if known_rule_ids and "rule_id" in applications.columns:
        application_rule_ids = {
            rule_id
            for rule_id in applications["rule_id"].dropna().map(str).map(str.strip).tolist()
            if rule_id
        }
        unknown_application_rules = sorted(application_rule_ids - known_rule_ids)
        if unknown_application_rules:
            blocking_issues.append(
                _keyword_rule_issue(
                    "unknown_keyword_rule_application_rule",
                    "blocking",
                    f"Rule applications reference unknown rule IDs: {', '.join(unknown_application_rules[:20])}.",
                )
            )
        checks["unknown_application_rule_ids"] = unknown_application_rules

    unknown_before_after_rules: set[str] = set()
    if known_rule_ids and "rule_ids" in before_after.columns:
        for value in before_after["rule_ids"].tolist():
            unknown_before_after_rules.update(_keyword_rule_split_ids(value) - known_rule_ids)
        if unknown_before_after_rules:
            warnings.append(
                _keyword_rule_issue(
                    "unknown_keyword_rule_before_after_rule",
                    "warning",
                    (
                        "Before/after rows reference rule IDs not present in the rule table: "
                        f"{', '.join(sorted(unknown_before_after_rules)[:20])}."
                    ),
                )
            )
    checks["unknown_before_after_rule_ids"] = sorted(unknown_before_after_rules)

    if not len(rules) and not len(before_after):
        warnings.append(
            _keyword_rule_issue(
                "empty_keyword_rule_table",
                "warning",
                "Keyword rule artifact has no rule rows.",
            )
        )

    return checks


def _keyword_rule_contamination_counts(before_after: pd.DataFrame) -> dict[str, int]:
    if before_after.empty:
        return {
            "artifact_rows_after": 0,
            "top_artifact_rows_after": 0,
            "review_artifact_rows_after": 0,
        }
    working = before_after.copy()
    if "blocked" in working.columns:
        blocked = working["blocked"].map(_keyword_rule_truthy)
        working = working.loc[~blocked].copy()
    if working.empty:
        return {
            "artifact_rows_after": 0,
            "top_artifact_rows_after": 0,
            "review_artifact_rows_after": 0,
        }
    if "rank_after" in working.columns:
        working["_rank_after"] = pd.to_numeric(working["rank_after"], errors="coerce").fillna(float("inf"))
        working = working.sort_values(["cluster_id", "_rank_after"], kind="stable")

    artifact_rows = 0
    top_artifact_rows = 0
    review_artifact_rows = 0
    for _, group in working.groupby("cluster_id", dropna=False, sort=False):
        for rank, (_, row) in enumerate(group.iterrows(), start=1):
            terms = _keyword_rule_term_values(row)
            has_blocking = any(_looks_like_metadata_artifact_term_lazy(term) for term in terms)
            flags = _flag_set(row.get("quality_flags") if "quality_flags" in row.index else None)
            if has_blocking or (flags & BLOCKING_ARTIFACT_FLAGS):
                artifact_rows += 1
                if rank <= KEYWORD_ARTIFACT_TOP_K:
                    top_artifact_rows += 1
            elif flags & REVIEW_ARTIFACT_FLAGS:
                review_artifact_rows += 1
    return {
        "artifact_rows_after": int(artifact_rows),
        "top_artifact_rows_after": int(top_artifact_rows),
        "review_artifact_rows_after": int(review_artifact_rows),
    }


def validate_keyword_rule_artifact(path: str | Path) -> KeywordRuleValidationResult:
    """Validate keyword-cleaning rule artifacts and their before/after QA surface."""

    rule_dir, manifest_path = _keyword_rule_dir_and_manifest(path)
    warnings: list[dict[str, Any]] = []
    blocking_issues: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {}
    if not manifest_path.exists():
        blocking_issues.append(
            _keyword_rule_issue(
                "missing_keyword_rule_manifest",
                "blocking",
                "Keyword rule artifact manifest is missing.",
            )
        )
    else:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            blocking_issues.append(
                _keyword_rule_issue(
                    "unreadable_keyword_rule_manifest",
                    "blocking",
                    f"Could not read keyword rule artifact manifest: {exc}",
                )
            )

    if manifest and manifest.get("schema_version") != KEYWORD_RULE_MANIFEST_SCHEMA_VERSION:
        blocking_issues.append(
            _keyword_rule_issue(
                "unsupported_keyword_rule_manifest_schema",
                "blocking",
                f"Unsupported keyword rule manifest schema: {manifest.get('schema_version')}.",
            )
        )

    paths = _keyword_rule_output_paths(manifest, rule_dir)
    rules = _keyword_rule_read_parquet(paths["rules"], "rules", blocking_issues)
    applications = _keyword_rule_read_parquet(paths["applications"], "applications", blocking_issues)
    before_after = _keyword_rule_read_parquet(paths["before_after"], "before_after", blocking_issues)

    checks = {
        "rules": _keyword_rule_table_check(
            df=rules,
            table_name="rules",
            required=REQUIRED_KEYWORD_RULE_COLUMNS,
            expected_schema=KEYWORD_RULES_SCHEMA_VERSION,
            blocking_issues=blocking_issues,
        ),
        "applications": _keyword_rule_table_check(
            df=applications,
            table_name="applications",
            required=REQUIRED_KEYWORD_RULE_APPLICATION_COLUMNS,
            expected_schema=KEYWORD_RULE_APPLICATIONS_SCHEMA_VERSION,
            blocking_issues=blocking_issues,
        ),
        "before_after": _keyword_rule_table_check(
            df=before_after,
            table_name="before_after",
            required=REQUIRED_KEYWORD_TERM_BEFORE_AFTER_COLUMNS,
            expected_schema=KEYWORD_TERM_BEFORE_AFTER_SCHEMA_VERSION,
            blocking_issues=blocking_issues,
        ),
    }
    checks["policy"] = _keyword_rule_policy_checks(
        rules=rules,
        applications=applications,
        before_after=before_after,
        warnings=warnings,
        blocking_issues=blocking_issues,
    )

    contamination_counts = _keyword_rule_contamination_counts(before_after)
    if contamination_counts["top_artifact_rows_after"]:
        blocking_issues.append(
            _keyword_rule_issue(
                "top_keyword_artifact_after_cleaning",
                "blocking",
                (
                    f"{contamination_counts['top_artifact_rows_after']} artifact-like keyword rows remain "
                    f"in the top {KEYWORD_ARTIFACT_TOP_K} after cleaning."
                ),
            )
        )
    elif contamination_counts["artifact_rows_after"]:
        warnings.append(
            _keyword_rule_issue(
                "keyword_artifact_rows_after_cleaning",
                "warning",
                f"{contamination_counts['artifact_rows_after']} artifact-like keyword rows remain after cleaning.",
            )
        )

    if paths.get("qa") and not paths["qa"].exists():
        warnings.append(
            _keyword_rule_issue(
                "missing_keyword_rule_qa_sidecar",
                "warning",
                "Keyword rule QA sidecar is missing.",
            )
        )

    rule_family_counts = (
        {str(key): int(value) for key, value in rules["rule_family"].value_counts(dropna=False).items()}
        if "rule_family" in rules.columns
        else {}
    )
    action_counts = (
        {str(key): int(value) for key, value in rules["action"].value_counts(dropna=False).items()}
        if "action" in rules.columns
        else {}
    )
    blocked_rows = int(before_after["blocked"].map(_keyword_rule_truthy).sum()) if "blocked" in before_after.columns else 0
    changed_rows = 0
    if {"term_before", "term_after"} <= set(before_after.columns):
        changed_rows = int((before_after["term_before"].map(str) != before_after["term_after"].map(str)).sum())

    counts = {
        "rules": int(len(rules)),
        "enabled_rules": int(rules["enabled"].map(_keyword_rule_truthy).sum()) if "enabled" in rules.columns else 0,
        "applications": int(len(applications)),
        "before_after_rows": int(len(before_after)),
        "blocked_rows": blocked_rows,
        "changed_rows": changed_rows,
        "warnings": int(len(warnings)),
        "blocking_issues": int(len(blocking_issues)),
    }

    if blocking_issues:
        status = "blocked"
    elif warnings:
        status = "warning"
    else:
        status = "passed"

    return KeywordRuleValidationResult(
        schema_version=KEYWORD_RULE_QA_SCHEMA_VERSION,
        rule_set_id=manifest.get("rule_set_id") if isinstance(manifest, Mapping) else None,
        status=status,
        rule_dir=str(rule_dir),
        manifest_path=str(manifest_path),
        paths={key: str(value) if value is not None else None for key, value in paths.items()},
        counts=counts,
        rule_family_counts=rule_family_counts,
        action_counts=action_counts,
        contamination_counts=contamination_counts,
        checks=checks,
        warnings=warnings,
        blocking_issues=blocking_issues,
        created_at_utc=_utc_now(),
    )


def _keyword_rule_normalize_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _keyword_rule_prepare_rules(rules: pd.DataFrame | None, *, rule_set_id: str) -> pd.DataFrame:
    df = rules.copy() if rules is not None else pd.DataFrame()
    row_count = len(df)
    defaults: dict[str, Any] = {
        "schema_version": KEYWORD_RULES_SCHEMA_VERSION,
        "rule_set_id": rule_set_id,
        "rule_family": "review_flag",
        "match_type": "literal",
        "pattern": "",
        "replacement": "",
        "action": "flag",
        "confidence_policy": "review",
        "destructive": False,
        "enabled": True,
        "created_by": "sciscape",
        "reason": "",
    }
    if "rule_id" not in df.columns:
        df["rule_id"] = [f"rule_{index + 1:06d}" for index in range(row_count)]
    for column, default in defaults.items():
        if column not in df.columns:
            df[column] = default
    for column in REQUIRED_KEYWORD_RULE_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    df["schema_version"] = KEYWORD_RULES_SCHEMA_VERSION
    df["rule_set_id"] = rule_set_id
    for column in (
        "rule_id",
        "rule_family",
        "match_type",
        "pattern",
        "replacement",
        "action",
        "confidence_policy",
        "created_by",
        "reason",
    ):
        df[column] = df[column].map(_keyword_rule_normalize_text)
    df["destructive"] = df["destructive"].map(_keyword_rule_truthy)
    df["enabled"] = df["enabled"].map(lambda value: True if value is None else _keyword_rule_truthy(value))
    return df[sorted(set(df.columns))]


def _keyword_rule_source_artifacts(root: Path, artifacts: ResultArtifacts) -> list[dict[str, Any]]:
    source_artifacts: list[dict[str, Any]] = []
    for role, source_path in (
        ("keywords", artifacts.keywords_path),
        ("report_data", artifacts.report_data_path),
        ("membership", artifacts.membership_path),
    ):
        rel_path = _rel(source_path, root)
        if rel_path:
            source_artifacts.append({"role": role, "path": rel_path})
    return source_artifacts


def _keyword_rule_default_before_after(
    keywords: pd.DataFrame | None,
    *,
    rule_set_id: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if keywords is not None and not keywords.empty and {"cluster_id", "term"} <= set(keywords.columns):
        for item in keywords.itertuples(index=False):
            cluster_id = getattr(item, "cluster_id")
            term = _keyword_rule_normalize_text(getattr(item, "term"))
            flags: list[str] = []
            if _looks_like_metadata_artifact_term_lazy(term):
                flags.append("metadata_fragment")
            rows.append(
                {
                    "schema_version": KEYWORD_TERM_BEFORE_AFTER_SCHEMA_VERSION,
                    "rule_set_id": rule_set_id,
                    "cluster_id": cluster_id,
                    "raw_term": term,
                    "term_before": term,
                    "term_after": term,
                    "display_label": term,
                    "family_id": term.lower(),
                    "parent_term": "",
                    "variant_count": 1,
                    "rule_ids": "",
                    "quality_flags": "|".join(flags),
                    "review_status": "needs_review" if flags else "accepted",
                    "tier_before": "",
                    "tier_after": "",
                    "blocked": False,
                    "block_reason": "",
                }
            )
    return pd.DataFrame(rows, columns=sorted(REQUIRED_KEYWORD_TERM_BEFORE_AFTER_COLUMNS | {"quality_flags"}))


def _keyword_rule_prepare_before_after(
    before_after: pd.DataFrame | None,
    *,
    rule_set_id: str,
    keywords: pd.DataFrame | None,
) -> pd.DataFrame:
    df = before_after.copy() if before_after is not None else _keyword_rule_default_before_after(keywords, rule_set_id=rule_set_id)
    defaults: dict[str, Any] = {
        "schema_version": KEYWORD_TERM_BEFORE_AFTER_SCHEMA_VERSION,
        "rule_set_id": rule_set_id,
        "cluster_id": "",
        "raw_term": "",
        "term_before": "",
        "term_after": "",
        "display_label": "",
        "family_id": "",
        "parent_term": "",
        "variant_count": 1,
        "rule_ids": "",
        "quality_flags": "",
        "review_status": "accepted",
        "tier_before": "",
        "tier_after": "",
        "blocked": False,
        "block_reason": "",
    }
    for column, default in defaults.items():
        if column not in df.columns:
            df[column] = default
    df["schema_version"] = KEYWORD_TERM_BEFORE_AFTER_SCHEMA_VERSION
    df["rule_set_id"] = rule_set_id
    df["blocked"] = df["blocked"].map(_keyword_rule_truthy)
    for column in (
        "raw_term",
        "term_before",
        "term_after",
        "display_label",
        "family_id",
        "parent_term",
        "rule_ids",
        "quality_flags",
        "review_status",
        "tier_before",
        "tier_after",
        "block_reason",
    ):
        df[column] = df[column].map(_keyword_rule_normalize_text)
    return df[sorted(set(df.columns))]


def _keyword_rule_default_applications(
    before_after: pd.DataFrame,
    *,
    rule_set_id: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, row in before_after.iterrows():
        rule_ids = sorted(_keyword_rule_split_ids(row.get("rule_ids")))
        changed = str(row.get("term_before", "")) != str(row.get("term_after", ""))
        blocked = _keyword_rule_truthy(row.get("blocked"))
        if not rule_ids and not changed and not blocked:
            continue
        for rule_id in rule_ids or [""]:
            rows.append(
                {
                    "schema_version": KEYWORD_RULE_APPLICATIONS_SCHEMA_VERSION,
                    "rule_set_id": rule_set_id,
                    "application_id": f"app_{index + 1:08d}_{len(rows) + 1:04d}",
                    "rule_id": rule_id,
                    "cluster_id": row.get("cluster_id", ""),
                    "raw_term": row.get("raw_term", ""),
                    "normalized_term_before": row.get("term_before", ""),
                    "display_label_before": row.get("term_before", ""),
                    "normalized_term_after": row.get("term_after", ""),
                    "display_label_after": row.get("display_label", row.get("term_after", "")),
                    "action": "block" if blocked else "normalize",
                    "decision": "blocked" if blocked else "applied",
                    "evidence_type": "before_after",
                    "evidence_value": row.get("block_reason", ""),
                    "score_before": None,
                    "score_after": None,
                    "frequency": None,
                    "rank_before": None,
                    "rank_after": None,
                }
            )
    return pd.DataFrame(rows, columns=sorted(REQUIRED_KEYWORD_RULE_APPLICATION_COLUMNS))


def _keyword_rule_prepare_applications(
    applications: pd.DataFrame | None,
    *,
    rule_set_id: str,
    before_after: pd.DataFrame,
) -> pd.DataFrame:
    df = applications.copy() if applications is not None else _keyword_rule_default_applications(before_after, rule_set_id=rule_set_id)
    defaults: dict[str, Any] = {
        "schema_version": KEYWORD_RULE_APPLICATIONS_SCHEMA_VERSION,
        "rule_set_id": rule_set_id,
        "application_id": "",
        "rule_id": "",
        "cluster_id": "",
        "raw_term": "",
        "normalized_term_before": "",
        "display_label_before": "",
        "normalized_term_after": "",
        "display_label_after": "",
        "action": "",
        "decision": "applied",
        "evidence_type": "",
        "evidence_value": "",
        "score_before": None,
        "score_after": None,
        "frequency": None,
        "rank_before": None,
        "rank_after": None,
    }
    for column, default in defaults.items():
        if column not in df.columns:
            df[column] = default
    if len(df) and not df["application_id"].map(_keyword_rule_normalize_text).any():
        df["application_id"] = [f"app_{index + 1:08d}" for index in range(len(df))]
    df["schema_version"] = KEYWORD_RULE_APPLICATIONS_SCHEMA_VERSION
    df["rule_set_id"] = rule_set_id
    for column in (
        "application_id",
        "rule_id",
        "raw_term",
        "normalized_term_before",
        "display_label_before",
        "normalized_term_after",
        "display_label_after",
        "action",
        "decision",
        "evidence_type",
        "evidence_value",
    ):
        df[column] = df[column].map(_keyword_rule_normalize_text)
    return df[sorted(set(df.columns))]


def _keyword_rule_impact_summary(
    *,
    rule_set_id: str,
    rules: pd.DataFrame,
    applications: pd.DataFrame,
    before_after: pd.DataFrame,
    validation: KeywordRuleValidationResult | None = None,
) -> dict[str, Any]:
    blocked_rows = int(before_after["blocked"].map(_keyword_rule_truthy).sum()) if "blocked" in before_after.columns else 0
    changed_rows = 0
    if {"term_before", "term_after"} <= set(before_after.columns):
        changed_rows = int((before_after["term_before"].map(str) != before_after["term_after"].map(str)).sum())
    return {
        "schema_version": KEYWORD_RULE_IMPACT_SUMMARY_SCHEMA_VERSION,
        "rule_set_id": rule_set_id,
        "counts": {
            "rules": int(len(rules)),
            "enabled_rules": int(rules["enabled"].map(_keyword_rule_truthy).sum()) if "enabled" in rules.columns else 0,
            "applications": int(len(applications)),
            "before_after_rows": int(len(before_after)),
            "blocked_rows": blocked_rows,
            "changed_rows": changed_rows,
        },
        "rule_family_counts": (
            {str(key): int(value) for key, value in rules["rule_family"].value_counts(dropna=False).items()}
            if "rule_family" in rules.columns
            else {}
        ),
        "action_counts": (
            {str(key): int(value) for key, value in rules["action"].value_counts(dropna=False).items()}
            if "action" in rules.columns
            else {}
        ),
        "contamination_counts": validation.contamination_counts if validation else {},
        "created_at_utc": _utc_now(),
    }


def write_keyword_rule_artifacts(
    path: str | Path,
    *,
    rule_set_id: str = "keyword_cleaning_default_v1",
    title: str | None = None,
    rules: pd.DataFrame | None = None,
    applications: pd.DataFrame | None = None,
    before_after: pd.DataFrame | None = None,
    keywords: pd.DataFrame | None = None,
    output_dir: str | Path | None = None,
    source_artifacts: list[dict[str, Any]] | None = None,
    transforms: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Write a manifest-backed keyword-cleaning rule artifact."""

    artifacts = infer_result_artifacts(path)
    root = artifacts.result_root
    rule_dir = Path(output_dir).expanduser().resolve() if output_dir else root / "rules" / rule_set_id
    rule_dir.mkdir(parents=True, exist_ok=True)

    if keywords is None and artifacts.keywords_path and artifacts.keywords_path.exists():
        keywords = pd.read_parquet(artifacts.keywords_path)

    rules_df = _keyword_rule_prepare_rules(rules, rule_set_id=rule_set_id)
    before_after_df = _keyword_rule_prepare_before_after(before_after, rule_set_id=rule_set_id, keywords=keywords)
    applications_df = _keyword_rule_prepare_applications(
        applications,
        rule_set_id=rule_set_id,
        before_after=before_after_df,
    )

    outputs = {
        "rules": "rules.parquet",
        "applications": "rule_applications.parquet",
        "before_after": "term_before_after.parquet",
        "impact_summary": "rule_impact_summary.json",
        "qa": "rule_set_qa.json",
    }
    manifest = {
        "schema_version": KEYWORD_RULE_MANIFEST_SCHEMA_VERSION,
        "rule_set_id": rule_set_id,
        "title": title or rule_set_id.replace("_", " ").title(),
        "result_id": _matrix_existing_result_id(root),
        "rule_scope": "keyword_cleaning",
        "policy": {
            "destructive_block_families": sorted(KEYWORD_RULE_BLOCK_FAMILIES),
            "allowed_actions": sorted(KEYWORD_RULE_ACTIONS),
            "allowed_rule_families": sorted(KEYWORD_RULE_FAMILIES),
            "top_artifact_rows_after_cleaning": "blocking",
        },
        "source_artifacts": source_artifacts if source_artifacts is not None else _keyword_rule_source_artifacts(root, artifacts),
        "transforms": [dict(item) for item in (transforms or [])],
        "outputs": outputs,
        "created_at_utc": _utc_now(),
        "warnings": [],
    }

    rules_df.to_parquet(rule_dir / outputs["rules"], index=False)
    applications_df.to_parquet(rule_dir / outputs["applications"], index=False)
    before_after_df.to_parquet(rule_dir / outputs["before_after"], index=False)
    (rule_dir / "rule_set_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    validation = validate_keyword_rule_artifact(rule_dir)
    qa_payload = validation.to_dict()
    qa_payload["warnings"] = [
        warning for warning in qa_payload["warnings"] if warning.get("code") != "missing_keyword_rule_qa_sidecar"
    ]
    qa_payload["counts"]["warnings"] = len(qa_payload["warnings"])
    if qa_payload["status"] == "warning" and not qa_payload["warnings"]:
        qa_payload["status"] = "passed"
    (rule_dir / outputs["impact_summary"]).write_text(
        json.dumps(
            _keyword_rule_impact_summary(
                rule_set_id=rule_set_id,
                rules=rules_df,
                applications=applications_df,
                before_after=before_after_df,
                validation=validation,
            ),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (rule_dir / outputs["qa"]).write_text(json.dumps(qa_payload, indent=2, sort_keys=True), encoding="utf-8")
    validation = validate_keyword_rule_artifact(rule_dir)
    return {
        "schema_version": KEYWORD_RULE_MANIFEST_SCHEMA_VERSION,
        "rule_set_id": rule_set_id,
        "rule_dir": rule_dir,
        "manifest_path": rule_dir / "rule_set_manifest.json",
        "rules_path": rule_dir / outputs["rules"],
        "applications_path": rule_dir / outputs["applications"],
        "before_after_path": rule_dir / outputs["before_after"],
        "impact_summary_path": rule_dir / outputs["impact_summary"],
        "qa_path": rule_dir / outputs["qa"],
        "qa": validation.to_dict(),
    }


def _export_issue(
    code: str,
    severity: str,
    message: str,
    *,
    artifact: str | None = None,
) -> dict[str, Any]:
    return {"code": code, "severity": severity, "message": message, "artifact": artifact}


def _export_dir_and_manifest(path: str | Path) -> tuple[Path, Path]:
    candidate = Path(path).expanduser().resolve()
    if candidate.is_file():
        return candidate.parent, candidate
    if candidate.name == "export_manifest.json":
        return candidate.parent, candidate
    return candidate, candidate / "export_manifest.json"


def _export_result_root(export_dir: Path) -> Path:
    if export_dir.parent.name == "exports":
        return export_dir.parent.parent
    return export_dir.parent


def _export_output_paths(export_dir: Path, manifest: Mapping[str, Any]) -> dict[str, Path]:
    outputs = manifest.get("outputs") if isinstance(manifest.get("outputs"), Mapping) else {}
    return {
        "files": export_dir / str(outputs.get("files") or "export_files.parquet"),
        "inputs": export_dir / str(outputs.get("inputs") or "export_inputs.parquet"),
        "transforms": export_dir / str(outputs.get("transforms") or "export_transforms.parquet"),
        "qa": export_dir / str(outputs.get("qa") or "export_qa.json"),
    }


def _export_required_fields(manifest: Mapping[str, Any]) -> set[str]:
    required = {
        "schema_version",
        "export_id",
        "title",
        "export_family",
        "export_kind",
        "format",
        "status",
        "feature_refs",
        "source_artifacts",
        "selection",
        "transform_summary",
        "compatibility",
        "outputs",
        "created_at_utc",
        "warnings",
    }
    return {field for field in required if field not in manifest}


def _export_read_parquet(
    path: Path,
    *,
    artifact: str,
    warnings: list[dict[str, Any]],
    blocking_issues: list[dict[str, Any]],
) -> pd.DataFrame | None:
    if not path.exists():
        blocking_issues.append(
            _export_issue(
                f"missing_export_{artifact}",
                "blocking",
                f"Missing export {artifact} table.",
                artifact=artifact,
            )
        )
        return None
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        blocking_issues.append(
            _export_issue(
                f"invalid_export_{artifact}_parquet",
                "blocking",
                f"Could not read export {artifact} parquet: {exc}",
                artifact=artifact,
            )
        )
        return None


def _export_missing_columns(
    df: pd.DataFrame | None,
    required: set[str],
    *,
    artifact: str,
    blocking_issues: list[dict[str, Any]],
) -> bool:
    if df is None:
        return True
    missing = required - set(df.columns)
    if missing:
        blocking_issues.append(
            _export_issue(
                "missing_export_columns",
                "blocking",
                f"Missing required export {artifact} columns: {sorted(missing)}",
                artifact=artifact,
            )
        )
        return True
    return False


def _export_table_schema_check(
    df: pd.DataFrame | None,
    *,
    artifact: str,
    expected_schema: str,
    required: set[str],
    warnings: list[dict[str, Any]],
    blocking_issues: list[dict[str, Any]],
) -> dict[str, Any]:
    if df is None or _export_missing_columns(df, required, artifact=artifact, blocking_issues=blocking_issues):
        return {"status": "blocked", "rows": 0}
    schema_values = set(df["schema_version"].dropna().map(str).unique().tolist())
    if schema_values and schema_values != {expected_schema}:
        blocking_issues.append(
            _export_issue(
                "unsupported_export_table_schema",
                "blocking",
                f"Unsupported export {artifact} schema values: {sorted(schema_values)}",
                artifact=artifact,
            )
        )
        return {"status": "blocked", "rows": int(len(df)), "schema_values": sorted(schema_values)}
    if df.empty:
        warnings.append(
            _export_issue(
                "empty_export_table",
                "warning",
                f"Export {artifact} table is empty.",
                artifact=artifact,
            )
        )
        return {"status": "warning", "rows": 0, "schema_values": sorted(schema_values)}
    return {"status": "passed", "rows": int(len(df)), "schema_values": sorted(schema_values)}


def _path_from_relative(root: Path, rel_path: object) -> Path | None:
    if rel_path is None:
        return None
    text = str(rel_path)
    if not text:
        return None
    path = Path(text)
    if path.is_absolute():
        return path
    return root / path


def _export_file_checks(
    files: pd.DataFrame | None,
    *,
    result_root: Path,
    blocking_issues: list[dict[str, Any]],
) -> dict[str, Any]:
    if files is None or "path" not in files.columns:
        return {"status": "blocked", "missing": 0, "absolute_paths": 0}
    missing = 0
    absolute_paths = 0
    for value in files["path"].dropna().map(str).tolist():
        if Path(value).is_absolute():
            absolute_paths += 1
            continue
        if not (result_root / value).exists():
            missing += 1
    if absolute_paths:
        blocking_issues.append(
            _export_issue(
                "absolute_export_file_path",
                "blocking",
                f"{absolute_paths} export file paths are absolute; manifests must use result-relative paths.",
                artifact="files",
            )
        )
    if missing:
        blocking_issues.append(
            _export_issue(
                "missing_export_files",
                "blocking",
                f"{missing} exported files are missing from the result root.",
                artifact="files",
            )
        )
    return {
        "status": "blocked" if missing or absolute_paths else "passed",
        "missing": int(missing),
        "absolute_paths": int(absolute_paths),
    }


def _export_input_checks(
    inputs: pd.DataFrame | None,
    *,
    result_root: Path,
    blocking_issues: list[dict[str, Any]],
) -> dict[str, Any]:
    if inputs is None or "artifact_path" not in inputs.columns:
        return {"status": "blocked", "missing_required": 0, "absolute_paths": 0}
    missing_required = 0
    absolute_paths = 0
    for row in inputs.to_dict("records"):
        artifact_path = str(row.get("artifact_path") or "")
        if not artifact_path:
            continue
        if Path(artifact_path).is_absolute():
            absolute_paths += 1
            continue
        required = bool(row.get("required"))
        if required and not (result_root / artifact_path).exists():
            missing_required += 1
    if absolute_paths:
        blocking_issues.append(
            _export_issue(
                "absolute_export_input_path",
                "blocking",
                f"{absolute_paths} export input paths are absolute; manifests must use result-relative paths.",
                artifact="inputs",
            )
        )
    if missing_required:
        blocking_issues.append(
            _export_issue(
                "missing_export_source_artifacts",
                "blocking",
                f"{missing_required} required export source artifacts are missing.",
                artifact="inputs",
            )
        )
    return {
        "status": "blocked" if missing_required or absolute_paths else "passed",
        "missing_required": int(missing_required),
        "absolute_paths": int(absolute_paths),
    }


def validate_export_manifest(path: str | Path) -> ExportManifestValidationResult:
    """Validate one export manifest and its file/input/transform sidecars."""

    export_dir, manifest_path = _export_dir_and_manifest(path)
    result_root = _export_result_root(export_dir)
    warnings: list[dict[str, Any]] = []
    blocking_issues: list[dict[str, Any]] = []
    checks: dict[str, dict[str, Any]] = {}
    manifest: dict[str, Any] = {}

    if not manifest_path.exists():
        blocking_issues.append(
            _export_issue("missing_export_manifest", "blocking", "Missing export_manifest.json.", artifact="manifest")
        )
    else:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            blocking_issues.append(
                _export_issue(
                    "invalid_export_manifest_json",
                    "blocking",
                    f"Could not read export manifest: {exc}",
                    artifact="manifest",
                )
            )
        if not isinstance(manifest, dict):
            blocking_issues.append(
                _export_issue(
                    "invalid_export_manifest_shape",
                    "blocking",
                    "Export manifest must be a JSON object.",
                    artifact="manifest",
                )
            )
            manifest = {}

    if manifest:
        if manifest.get("schema_version") != EXPORT_MANIFEST_SCHEMA_VERSION:
            blocking_issues.append(
                _export_issue(
                    "unsupported_export_manifest_schema",
                    "blocking",
                    f"Unsupported export manifest schema: {manifest.get('schema_version')}",
                    artifact="manifest",
                )
            )
        if manifest.get("export_family") not in SUPPORTED_EXPORT_FAMILIES:
            blocking_issues.append(
                _export_issue(
                    "unsupported_export_family",
                    "blocking",
                    f"Unsupported export family: {manifest.get('export_family')}",
                    artifact="manifest",
                )
            )
        missing_fields = _export_required_fields(manifest)
        if missing_fields:
            blocking_issues.append(
                _export_issue(
                    "missing_export_manifest_fields",
                    "blocking",
                    f"Missing export manifest fields: {sorted(missing_fields)}",
                    artifact="manifest",
                )
            )
    checks["manifest"] = {
        "status": "blocked" if blocking_issues else "passed",
        "schema_version": manifest.get("schema_version"),
    }

    paths = _export_output_paths(export_dir, manifest)
    files = _export_read_parquet(paths["files"], artifact="files", warnings=warnings, blocking_issues=blocking_issues)
    inputs = _export_read_parquet(
        paths["inputs"],
        artifact="inputs",
        warnings=warnings,
        blocking_issues=blocking_issues,
    )
    transforms = _export_read_parquet(
        paths["transforms"],
        artifact="transforms",
        warnings=warnings,
        blocking_issues=blocking_issues,
    )
    checks["files"] = _export_table_schema_check(
        files,
        artifact="files",
        expected_schema=EXPORT_FILES_SCHEMA_VERSION,
        required=REQUIRED_EXPORT_FILE_COLUMNS,
        warnings=warnings,
        blocking_issues=blocking_issues,
    )
    checks["inputs"] = _export_table_schema_check(
        inputs,
        artifact="inputs",
        expected_schema=EXPORT_INPUTS_SCHEMA_VERSION,
        required=REQUIRED_EXPORT_INPUT_COLUMNS,
        warnings=warnings,
        blocking_issues=blocking_issues,
    )
    checks["transforms"] = _export_table_schema_check(
        transforms,
        artifact="transforms",
        expected_schema=EXPORT_TRANSFORMS_SCHEMA_VERSION,
        required=REQUIRED_EXPORT_TRANSFORM_COLUMNS,
        warnings=warnings,
        blocking_issues=blocking_issues,
    )
    checks["file_inventory"] = _export_file_checks(files, result_root=result_root, blocking_issues=blocking_issues)
    checks["source_artifacts"] = _export_input_checks(inputs, result_root=result_root, blocking_issues=blocking_issues)

    qa_path = paths["qa"]
    if not qa_path.exists():
        warnings.append(
            _export_issue(
                "missing_export_qa_sidecar",
                "warning",
                "export_qa.json is missing; writers should persist the validation result.",
                artifact="qa",
            )
        )

    counts = {
        "files": int(0 if files is None else len(files)),
        "inputs": int(0 if inputs is None else len(inputs)),
        "transforms": int(0 if transforms is None else len(transforms)),
        "warnings": len(warnings),
        "blocking_issues": len(blocking_issues),
    }
    if blocking_issues:
        status = "blocked"
    elif warnings:
        status = "warning"
    else:
        status = "passed"

    return ExportManifestValidationResult(
        schema_version=EXPORT_QA_SCHEMA_VERSION,
        export_id=str(manifest.get("export_id")) if manifest.get("export_id") else None,
        export_family=str(manifest.get("export_family")) if manifest.get("export_family") else None,
        export_kind=str(manifest.get("export_kind")) if manifest.get("export_kind") else None,
        status=status,
        export_dir=str(export_dir),
        manifest_path=_rel(manifest_path, export_dir) or str(manifest_path),
        paths={key: _rel(value, export_dir) for key, value in paths.items()},
        counts=counts,
        checks=checks,
        compatibility=dict(manifest.get("compatibility") or {}),
        warnings=warnings,
        blocking_issues=blocking_issues,
        created_at_utc=_utc_now(),
    )


def _export_existing_result_id(result_root: Path) -> str | None:
    return _matrix_existing_result_id(result_root)


def _export_rel_to_root(path: str | Path, root: Path) -> str:
    if path in (None, ""):
        return ""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        rel = _rel(candidate.resolve(), root)
        return rel or str(candidate)
    return str(candidate)


def _export_file_format(path: str | Path, fallback: str = "file") -> str:
    suffix = Path(path).suffix.lower().lstrip(".")
    return suffix or fallback


def _export_feature_states(root: Path, feature_refs: list[str]) -> dict[str, str]:
    manifest = build_result_manifest(root).to_dict()
    features = manifest.get("features", {})
    states: dict[str, str] = {}
    for feature in feature_refs:
        payload = features.get(feature)
        if isinstance(payload, Mapping):
            states[feature] = str(payload.get("state") or "hidden")
        else:
            states[feature] = "hidden"
    return states


def _export_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _export_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_export_jsonable(item) for item in value]
    return value


def _export_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _normalize_export_selection(
    selection: Mapping[str, Any] | None,
    *,
    export_family: str,
    export_kind: str,
    feature_refs: list[str],
) -> dict[str, Any]:
    raw = _export_mapping(_export_jsonable(selection or {}))
    view_raw = raw.pop("view", {})
    if isinstance(view_raw, Mapping):
        view = dict(view_raw)
    elif view_raw:
        view = {"mode": str(view_raw)}
    else:
        view = {}
    view.setdefault("mode", export_kind)
    view.setdefault("family", export_family)

    filters = raw.pop("filters", [])
    if filters in (None, ""):
        filters = []
    elif not isinstance(filters, list):
        filters = [filters]

    payload = {
        "schema_version": "sciscape_export_selection_v1",
        "scope": str(raw.pop("scope", "full_result") or "full_result"),
        "view": _export_jsonable(view),
        "cluster_level": raw.pop("cluster_level", None),
        "filters": _export_jsonable(filters),
        "thresholds": _export_jsonable(_export_mapping(raw.pop("thresholds", {}))),
        "layer_state": _export_jsonable(_export_mapping(raw.pop("layer_state", {}))),
        "focus": _export_jsonable(_export_mapping(raw.pop("focus", {}))),
        "subset": _export_jsonable(_export_mapping(raw.pop("subset", {}))),
        "feature_refs": [str(feature) for feature in feature_refs],
    }
    for key, value in raw.items():
        payload[str(key)] = _export_jsonable(value)
    return payload


def _manifest_export_selection(root: Path, export_manifest_path: str | None) -> dict[str, Any]:
    if not export_manifest_path:
        return {}
    try:
        manifest = json.loads((root / export_manifest_path).read_text(encoding="utf-8"))
    except Exception:
        return {}
    selection = manifest.get("selection")
    return dict(selection) if isinstance(selection, Mapping) else {}


def _manifest_selection_summary(selection: Mapping[str, Any]) -> dict[str, Any]:
    view = selection.get("view") if isinstance(selection.get("view"), Mapping) else {}
    filters = selection.get("filters") if isinstance(selection.get("filters"), list) else []
    thresholds = selection.get("thresholds") if isinstance(selection.get("thresholds"), Mapping) else {}
    layer_state = selection.get("layer_state") if isinstance(selection.get("layer_state"), Mapping) else {}
    focus = selection.get("focus") if isinstance(selection.get("focus"), Mapping) else {}
    subset = selection.get("subset") if isinstance(selection.get("subset"), Mapping) else {}
    return {
        "scope": selection.get("scope"),
        "view_mode": view.get("mode"),
        "view_family": view.get("family"),
        "cluster_level": selection.get("cluster_level"),
        "filter_count": len(filters),
        "threshold_keys": sorted(str(key) for key in thresholds),
        "layer_state_keys": sorted(str(key) for key in layer_state),
        "focus_keys": sorted(str(key) for key in focus),
        "subset_mode": subset.get("mode"),
        "subset_count": subset.get("count"),
        "subset_keys": sorted(str(key) for key in subset),
    }


def write_export_manifest(
    result_root: str | Path,
    *,
    export_id: str,
    export_family: str,
    export_kind: str,
    primary_file: str | Path,
    source_artifacts: list[Mapping[str, Any]],
    feature_refs: list[str],
    files: list[Mapping[str, Any]] | None = None,
    selection: Mapping[str, Any] | None = None,
    transforms: list[Mapping[str, Any]] | None = None,
    compatibility: Mapping[str, Any] | None = None,
    title: str | None = None,
    format: str | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Write a manifest-backed export wrapper for existing output files."""

    root = Path(result_root).expanduser().resolve()
    export_id = _safe_id(export_id, fallback="export")
    if export_family not in SUPPORTED_EXPORT_FAMILIES:
        raise ValueError(f"unsupported export family: {export_family}")
    export_dir = Path(output_dir).expanduser().resolve() if output_dir else root / "exports" / export_id
    export_dir.mkdir(parents=True, exist_ok=True)

    primary_rel = _export_rel_to_root(primary_file, root)
    primary_format = format or _export_file_format(primary_rel)
    file_rows = files or [
        {
            "file_id": "primary",
            "path": primary_rel,
            "role": "primary",
            "format": primary_format,
            "public_share_state": "local",
        }
    ]
    normalized_files: list[dict[str, Any]] = []
    for index, row in enumerate(file_rows, start=1):
        rel_path = _export_rel_to_root(row.get("path") or primary_rel, root)
        resolved = _path_from_relative(root, rel_path)
        normalized_files.append(
            {
                "schema_version": EXPORT_FILES_SCHEMA_VERSION,
                "export_id": export_id,
                "file_id": str(row.get("file_id") or f"file_{index}"),
                "path": rel_path,
                "role": str(row.get("role") or ("primary" if index == 1 else "support")),
                "format": str(row.get("format") or _export_file_format(rel_path, primary_format)),
                "bytes": int(resolved.stat().st_size) if resolved is not None and resolved.exists() else None,
                "exists": bool(resolved is not None and resolved.exists() and not Path(rel_path).is_absolute()),
                "public_share_state": str(row.get("public_share_state") or "local"),
            }
        )

    feature_states = _export_feature_states(root, feature_refs)
    normalized_inputs: list[dict[str, Any]] = []
    for index, row in enumerate(source_artifacts, start=1):
        feature = str(row.get("feature_ref") or (feature_refs[0] if feature_refs else "export"))
        normalized_inputs.append(
            {
                "schema_version": EXPORT_INPUTS_SCHEMA_VERSION,
                "export_id": export_id,
                "input_id": str(row.get("input_id") or f"input_{index}"),
                "artifact_ref": str(row.get("artifact_ref") or row.get("role") or f"source_{index}"),
                "artifact_role": str(row.get("artifact_role") or row.get("role") or "source"),
                "artifact_path": _export_rel_to_root(row.get("path") or "", root),
                "feature_state": str(row.get("feature_state") or feature_states.get(feature, "hidden")),
                "required": bool(row.get("required", True)),
            }
        )

    normalized_transforms: list[dict[str, Any]] = []
    for index, row in enumerate(transforms or [{"transform_type": "wrap_existing_export", "description": "Wrap existing export files."}], start=1):
        normalized_transforms.append(
            {
                "schema_version": EXPORT_TRANSFORMS_SCHEMA_VERSION,
                "export_id": export_id,
                "transform_id": str(row.get("transform_id") or f"transform_{index}"),
                "step_index": int(row.get("step_index") if row.get("step_index") is not None else index - 1),
                "transform_type": str(row.get("transform_type") or row.get("step") or "transform"),
                "description": str(row.get("description") or row.get("transform_type") or row.get("step") or "transform"),
                "parameters": json.dumps(row.get("parameters") or {}, sort_keys=True),
            }
        )

    outputs = {
        "files": "export_files.parquet",
        "inputs": "export_inputs.parquet",
        "transforms": "export_transforms.parquet",
        "qa": "export_qa.json",
    }
    manifest_source_artifacts: list[dict[str, Any]] = []
    for row in source_artifacts:
        source_row = dict(row)
        if "path" in source_row:
            source_row["path"] = _export_rel_to_root(source_row.get("path") or "", root)
        if "artifact_path" in source_row:
            source_row["artifact_path"] = _export_rel_to_root(source_row.get("artifact_path") or "", root)
        manifest_source_artifacts.append(source_row)
    manifest = {
        "schema_version": EXPORT_MANIFEST_SCHEMA_VERSION,
        "export_id": export_id,
        "title": title or export_id.replace("_", " ").title(),
        "result_id": _export_existing_result_id(root),
        "export_family": export_family,
        "export_kind": export_kind,
        "format": primary_format,
        "status": "pending",
        "feature_refs": [str(feature) for feature in feature_refs],
        "source_artifacts": manifest_source_artifacts,
        "selection": _normalize_export_selection(
            selection,
            export_family=export_family,
            export_kind=export_kind,
            feature_refs=feature_refs,
        ),
        "transform_summary": {
            "transform_count": len(normalized_transforms),
            "primary_transform": normalized_transforms[0]["transform_type"] if normalized_transforms else None,
        },
        "compatibility": dict(compatibility or {"target_tools": ["SciScape"], "limitations": []}),
        "outputs": outputs,
        "created_at_utc": _utc_now(),
        "warnings": [],
    }

    pd.DataFrame(normalized_files).to_parquet(export_dir / outputs["files"], index=False)
    pd.DataFrame(normalized_inputs).to_parquet(export_dir / outputs["inputs"], index=False)
    pd.DataFrame(normalized_transforms).to_parquet(export_dir / outputs["transforms"], index=False)
    manifest_path = export_dir / "export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    validation = validate_export_manifest(export_dir)
    qa_payload = validation.to_dict()
    qa_payload["warnings"] = [
        warning for warning in qa_payload["warnings"] if warning.get("code") != "missing_export_qa_sidecar"
    ]
    qa_payload["counts"]["warnings"] = len(qa_payload["warnings"])
    if qa_payload["status"] == "warning" and not qa_payload["warnings"]:
        qa_payload["status"] = "passed"
    manifest["status"] = qa_payload["status"]
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (export_dir / outputs["qa"]).write_text(json.dumps(qa_payload, indent=2, sort_keys=True), encoding="utf-8")
    validation = validate_export_manifest(export_dir)
    return {
        "schema_version": EXPORT_MANIFEST_SCHEMA_VERSION,
        "export_id": export_id,
        "export_dir": export_dir,
        "manifest_path": manifest_path,
        "files_path": export_dir / outputs["files"],
        "inputs_path": export_dir / outputs["inputs"],
        "transforms_path": export_dir / outputs["transforms"],
        "qa_path": export_dir / outputs["qa"],
        "qa": validation.to_dict(),
    }


def _temporal_issue(
    code: str,
    severity: str,
    message: str,
    *,
    artifact: str | None = None,
) -> dict[str, Any]:
    issue = {"code": code, "severity": severity, "message": message}
    if artifact:
        issue["artifact"] = artifact
    return issue


def _temporal_dir_and_manifest(path: str | Path) -> tuple[Path, Path]:
    candidate = Path(path).expanduser()
    if candidate.exists():
        candidate = candidate.resolve()
    if candidate.is_file():
        return candidate.parent, candidate
    return candidate, candidate / "temporal_manifest.json"


def _temporal_result_root(temporal_dir: Path) -> Path:
    if temporal_dir.name == "temporal":
        scope_root = temporal_dir.parent
        if scope_root.name.startswith("landscape") and scope_root.parent.exists():
            return scope_root.parent
        return scope_root
    return temporal_dir.parent


def _temporal_output_paths(temporal_dir: Path, manifest: Mapping[str, Any]) -> dict[str, Path]:
    outputs = manifest.get("outputs") if isinstance(manifest.get("outputs"), Mapping) else {}
    paths = {
        "periods": temporal_dir / str(outputs.get("periods") or "periods.parquet"),
        "activity": temporal_dir / str(outputs.get("activity") or "activity.parquet"),
        "series": temporal_dir / str(outputs.get("series") or "entity_series.parquet"),
        "qa": temporal_dir / str(outputs.get("qa") or "temporal_qa.json"),
    }
    events = outputs.get("events")
    paths["events"] = temporal_dir / str(events or "temporal_events.parquet")
    return paths


def _temporal_path_payload(paths: Mapping[str, Path], temporal_dir: Path) -> dict[str, str | None]:
    return {key: _rel(path, temporal_dir) for key, path in paths.items()}


def _temporal_required_fields(manifest: Mapping[str, Any]) -> set[str]:
    required = {
        "schema_version",
        "temporal_id",
        "title",
        "periodization",
        "entity_types",
        "metrics",
        "event_types",
        "source_artifacts",
        "transforms",
        "outputs",
        "created_at_utc",
    }
    return {key for key in required if manifest.get(key) in (None, "")}


def _temporal_read_parquet(
    path: Path,
    *,
    artifact: str,
    required: bool,
    blocking_issues: list[dict[str, Any]],
) -> pd.DataFrame | None:
    if not path.exists():
        if required:
            blocking_issues.append(
                _temporal_issue(
                    "missing_temporal_table",
                    "blocking",
                    f"Missing temporal {artifact} table.",
                    artifact=artifact,
                )
            )
        return None
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        blocking_issues.append(
            _temporal_issue(
                "invalid_temporal_parquet",
                "blocking",
                f"Could not read temporal {artifact} parquet: {exc}",
                artifact=artifact,
            )
        )
        return None


def _temporal_missing_columns(
    df: pd.DataFrame | None,
    required: set[str],
    *,
    artifact: str,
    blocking_issues: list[dict[str, Any]],
) -> set[str]:
    columns = set(df.columns) if df is not None else set()
    missing = required - columns
    if missing:
        blocking_issues.append(
            _temporal_issue(
                "missing_temporal_columns",
                "blocking",
                f"Missing required temporal columns: {sorted(missing)}",
                artifact=artifact,
            )
        )
    return missing


def _temporal_numeric_finite(
    df: pd.DataFrame | None,
    columns: list[str],
    *,
    artifact: str,
    blocking_issues: list[dict[str, Any]],
) -> None:
    if df is None:
        return
    for column in columns:
        if column not in df.columns:
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")
        mask = df[column].notna()
        bad = numeric[mask].isna()
        if bad.any() or (~numeric[mask].map(lambda value: math.isfinite(float(value)))).any():
            blocking_issues.append(
                _temporal_issue(
                    "invalid_temporal_numeric_values",
                    "blocking",
                    f"Temporal {column} values must be finite numeric values.",
                    artifact=artifact,
                )
            )


def _temporal_period_checks(
    periods: pd.DataFrame | None,
    *,
    manifest: Mapping[str, Any],
    blocking_issues: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
) -> dict[str, Any]:
    if periods is None or _temporal_missing_columns(
        periods,
        REQUIRED_TEMPORAL_PERIOD_COLUMNS,
        artifact="periods",
        blocking_issues=blocking_issues,
    ):
        return {"status": "blocked", "count": 0}
    temporal_id = str(manifest.get("temporal_id"))
    if not set(periods["schema_version"]) <= {TEMPORAL_PERIODS_SCHEMA_VERSION}:
        blocking_issues.append(
            _temporal_issue("unsupported_temporal_periods_schema", "blocking", "Unsupported periods schema.", artifact="periods")
        )
    if not set(periods["temporal_id"].map(str)) <= {temporal_id}:
        blocking_issues.append(
            _temporal_issue("temporal_period_id_mismatch", "blocking", "Period temporal_id does not match manifest.", artifact="periods")
        )
    if periods["period_id"].duplicated().any():
        blocking_issues.append(
            _temporal_issue("duplicate_temporal_periods", "blocking", "Temporal period_id values are duplicated.", artifact="periods")
        )
    try:
        indices = sorted(int(value) for value in periods["period_index"].tolist())
    except Exception:
        indices = []
        blocking_issues.append(
            _temporal_issue("invalid_temporal_period_index", "blocking", "period_index must be integer-like.", artifact="periods")
        )
    if indices and indices != list(range(len(indices))):
        warnings.append(
            _temporal_issue(
                "non_contiguous_temporal_period_index",
                "warning",
                "period_index is not contiguous from zero.",
                artifact="periods",
            )
        )
    return {"status": "passed", "count": int(len(periods))}


def _temporal_activity_checks(
    activity: pd.DataFrame | None,
    periods: pd.DataFrame | None,
    *,
    manifest: Mapping[str, Any],
    blocking_issues: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
) -> dict[str, Any]:
    if activity is None or _temporal_missing_columns(
        activity,
        REQUIRED_TEMPORAL_ACTIVITY_COLUMNS,
        artifact="activity",
        blocking_issues=blocking_issues,
    ):
        return {"status": "blocked", "count": 0}
    temporal_id = str(manifest.get("temporal_id"))
    if not set(activity["schema_version"]) <= {TEMPORAL_ACTIVITY_SCHEMA_VERSION}:
        blocking_issues.append(
            _temporal_issue("unsupported_temporal_activity_schema", "blocking", "Unsupported activity schema.", artifact="activity")
        )
    if not set(activity["temporal_id"].map(str)) <= {temporal_id}:
        blocking_issues.append(
            _temporal_issue("temporal_activity_id_mismatch", "blocking", "Activity temporal_id does not match manifest.", artifact="activity")
        )
    period_ids = set(periods["period_id"].map(str)) if periods is not None and "period_id" in periods.columns else set()
    activity_periods = set(activity["period_id"].map(str))
    missing_periods = period_ids - activity_periods
    unknown_periods = activity_periods - period_ids
    if missing_periods:
        blocking_issues.append(
            _temporal_issue(
                "missing_temporal_activity_periods",
                "blocking",
                f"Activity table is missing {len(missing_periods)} periods.",
                artifact="activity",
            )
        )
    if unknown_periods:
        blocking_issues.append(
            _temporal_issue(
                "unknown_temporal_activity_periods",
                "blocking",
                f"Activity table references {len(unknown_periods)} unknown periods.",
                artifact="activity",
            )
        )
    for column in ("doc_count", "edge_count", "active_cluster_count", "unknown_year_count"):
        numeric = pd.to_numeric(activity[column], errors="coerce")
        if numeric.dropna().lt(0).any():
            blocking_issues.append(
                _temporal_issue(
                    "negative_temporal_activity_count",
                    "blocking",
                    f"{column} must be non-negative.",
                    artifact="activity",
                )
            )
    if "unknown_year_count" in activity.columns and pd.to_numeric(activity["unknown_year_count"], errors="coerce").max() > 0:
        warnings.append(
            _temporal_issue(
                "temporal_unknown_years",
                "warning",
                "Some records have missing or invalid publication years.",
                artifact="activity",
            )
        )
    return {"status": "passed", "count": int(len(activity))}


def _temporal_declared_metrics(manifest: Mapping[str, Any]) -> set[str]:
    metrics = manifest.get("metrics")
    if not isinstance(metrics, list):
        return set()
    return {str(item.get("name")) for item in metrics if isinstance(item, Mapping) and item.get("name")}


def _temporal_series_checks(
    series: pd.DataFrame | None,
    periods: pd.DataFrame | None,
    *,
    manifest: Mapping[str, Any],
    blocking_issues: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
) -> dict[str, Any]:
    if series is None or _temporal_missing_columns(
        series,
        REQUIRED_TEMPORAL_SERIES_COLUMNS,
        artifact="series",
        blocking_issues=blocking_issues,
    ):
        return {"status": "blocked", "count": 0}
    temporal_id = str(manifest.get("temporal_id"))
    if not set(series["schema_version"]) <= {TEMPORAL_ENTITY_SERIES_SCHEMA_VERSION}:
        blocking_issues.append(
            _temporal_issue("unsupported_temporal_series_schema", "blocking", "Unsupported entity series schema.", artifact="series")
        )
    if not set(series["temporal_id"].map(str)) <= {temporal_id}:
        blocking_issues.append(
            _temporal_issue("temporal_series_id_mismatch", "blocking", "Series temporal_id does not match manifest.", artifact="series")
        )
    period_ids = set(periods["period_id"].map(str)) if periods is not None and "period_id" in periods.columns else set()
    unknown_periods = set(series["period_id"].map(str)) - period_ids
    if unknown_periods:
        blocking_issues.append(
            _temporal_issue(
                "unknown_temporal_series_periods",
                "blocking",
                f"Series table references {len(unknown_periods)} unknown periods.",
                artifact="series",
            )
        )
    metrics = _temporal_declared_metrics(manifest)
    unknown_metrics = set(series["metric"].map(str)) - metrics
    if unknown_metrics:
        blocking_issues.append(
            _temporal_issue(
                "undeclared_temporal_series_metrics",
                "blocking",
                f"Series table uses undeclared metrics: {sorted(unknown_metrics)}",
                artifact="series",
            )
        )
    if series.duplicated(subset=["entity_type", "entity_key", "period_id", "metric"]).any():
        blocking_issues.append(
            _temporal_issue("duplicate_temporal_series_rows", "blocking", "Duplicate entity/period/metric rows found.", artifact="series")
        )
    _temporal_numeric_finite(series, ["value", "raw_value", "denominator"], artifact="series", blocking_issues=blocking_issues)
    return {"status": "passed", "count": int(len(series))}


def _temporal_events_checks(
    events: pd.DataFrame | None,
    periods: pd.DataFrame | None,
    series: pd.DataFrame | None,
    *,
    manifest: Mapping[str, Any],
    blocking_issues: list[dict[str, Any]],
) -> dict[str, Any]:
    event_types = manifest.get("event_types") if isinstance(manifest.get("event_types"), list) else []
    if events is None:
        if event_types:
            blocking_issues.append(
                _temporal_issue(
                    "missing_temporal_events_table",
                    "blocking",
                    "Manifest advertises event types but temporal_events.parquet is missing.",
                    artifact="events",
                )
            )
            return {"status": "blocked", "count": 0}
        return {"status": "skipped", "count": 0}
    if _temporal_missing_columns(events, REQUIRED_TEMPORAL_EVENTS_COLUMNS, artifact="events", blocking_issues=blocking_issues):
        return {"status": "blocked", "count": int(len(events))}
    temporal_id = str(manifest.get("temporal_id"))
    if not events.empty and not set(events["schema_version"]) <= {TEMPORAL_EVENTS_SCHEMA_VERSION}:
        blocking_issues.append(
            _temporal_issue("unsupported_temporal_events_schema", "blocking", "Unsupported events schema.", artifact="events")
        )
    if not events.empty and not set(events["temporal_id"].map(str)) <= {temporal_id}:
        blocking_issues.append(
            _temporal_issue("temporal_events_id_mismatch", "blocking", "Events temporal_id does not match manifest.", artifact="events")
        )
    period_ids = set(periods["period_id"].map(str)) if periods is not None and "period_id" in periods.columns else set()
    unknown_periods = (set(events["start_period_id"].map(str)) | set(events["end_period_id"].map(str))) - period_ids
    if unknown_periods:
        blocking_issues.append(
            _temporal_issue(
                "unknown_temporal_event_periods",
                "blocking",
                f"Events reference {len(unknown_periods)} unknown periods.",
                artifact="events",
            )
        )
    if series is not None and not series.empty:
        series_keys = set(
            zip(
                series["entity_type"].map(str),
                series["entity_key"].map(str),
                series["metric"].map(str),
            )
        )
        event_keys = set(
            zip(
                events["entity_type"].map(str),
                events["entity_key"].map(str),
                events["metric"].map(str),
            )
        )
        missing = event_keys - series_keys
        if missing:
            blocking_issues.append(
                _temporal_issue(
                    "temporal_event_series_ref_missing",
                    "blocking",
                    f"{len(missing)} event entity/metric refs are absent from series rows.",
                    artifact="events",
                )
            )
    _temporal_numeric_finite(events, ["score"], artifact="events", blocking_issues=blocking_issues)
    return {"status": "passed", "count": int(len(events))}


def _temporal_source_checks(
    *,
    result_root: Path,
    manifest: Mapping[str, Any],
    warnings: list[dict[str, Any]],
) -> dict[str, Any]:
    source_artifacts = manifest.get("source_artifacts")
    if not isinstance(source_artifacts, list) or not source_artifacts:
        warnings.append(
            _temporal_issue(
                "missing_temporal_source_artifacts",
                "warning",
                "Temporal manifest should record at least one source artifact.",
                artifact="manifest",
            )
        )
        return {"status": "warning", "count": 0}
    missing = 0
    for source in source_artifacts:
        if not isinstance(source, Mapping):
            warnings.append(
                _temporal_issue("invalid_temporal_source_artifact", "warning", "Source refs should be objects.", artifact="manifest")
            )
            continue
        path = source.get("path")
        if not path:
            warnings.append(
                _temporal_issue("missing_temporal_source_path", "warning", "Source ref has no path.", artifact="manifest")
            )
            continue
        source_path = Path(str(path))
        resolved = source_path if source_path.is_absolute() else result_root / source_path
        if not resolved.exists():
            missing += 1
    if missing:
        warnings.append(
            _temporal_issue(
                "missing_temporal_source_artifact",
                "warning",
                f"{missing} temporal source artifact refs do not exist.",
                artifact="manifest",
            )
        )
    return {"status": "warning" if missing else "passed", "count": len(source_artifacts), "missing": missing}


def validate_temporal_artifact(path: str | Path) -> TemporalArtifactValidationResult:
    """Validate an artifact-backed temporal trend directory."""

    temporal_dir, manifest_path = _temporal_dir_and_manifest(path)
    result_root = _temporal_result_root(temporal_dir)
    warnings: list[dict[str, Any]] = []
    blocking_issues: list[dict[str, Any]] = []
    checks: dict[str, dict[str, Any]] = {}
    manifest: dict[str, Any] = {}

    if not manifest_path.exists():
        blocking_issues.append(
            _temporal_issue("missing_temporal_manifest", "blocking", "Missing temporal_manifest.json.", artifact="manifest")
        )
    else:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            blocking_issues.append(
                _temporal_issue(
                    "invalid_temporal_manifest_json",
                    "blocking",
                    f"Could not read temporal manifest: {exc}",
                    artifact="manifest",
                )
            )
        if not isinstance(manifest, dict):
            blocking_issues.append(
                _temporal_issue("invalid_temporal_manifest_shape", "blocking", "Temporal manifest must be an object.", artifact="manifest")
            )
            manifest = {}

    if manifest:
        if manifest.get("schema_version") != TEMPORAL_MANIFEST_SCHEMA_VERSION:
            blocking_issues.append(
                _temporal_issue(
                    "unsupported_temporal_manifest_schema",
                    "blocking",
                    f"Unsupported temporal manifest schema: {manifest.get('schema_version')}",
                    artifact="manifest",
                )
            )
        missing_fields = _temporal_required_fields(manifest)
        if missing_fields:
            blocking_issues.append(
                _temporal_issue(
                    "missing_temporal_manifest_fields",
                    "blocking",
                    f"Missing temporal manifest fields: {sorted(missing_fields)}",
                    artifact="manifest",
                )
            )
        periodization = manifest.get("periodization")
        if isinstance(periodization, Mapping) and periodization.get("unit") != "year":
            warnings.append(
                _temporal_issue(
                    "unsupported_temporal_periodization_unit",
                    "warning",
                    "Only yearly periodization is supported by the first validator.",
                    artifact="manifest",
                )
            )
    checks["manifest"] = {
        "status": "blocked" if blocking_issues else "passed",
        "schema_version": manifest.get("schema_version"),
    }

    paths = _temporal_output_paths(temporal_dir, manifest)
    periods = _temporal_read_parquet(paths["periods"], artifact="periods", required=True, blocking_issues=blocking_issues)
    activity = _temporal_read_parquet(paths["activity"], artifact="activity", required=True, blocking_issues=blocking_issues)
    series = _temporal_read_parquet(paths["series"], artifact="series", required=True, blocking_issues=blocking_issues)
    event_required = bool(manifest.get("event_types")) if isinstance(manifest.get("event_types"), list) else False
    events = _temporal_read_parquet(paths["events"], artifact="events", required=event_required, blocking_issues=blocking_issues)

    checks["periods"] = _temporal_period_checks(
        periods,
        manifest=manifest,
        blocking_issues=blocking_issues,
        warnings=warnings,
    )
    checks["activity"] = _temporal_activity_checks(
        activity,
        periods,
        manifest=manifest,
        blocking_issues=blocking_issues,
        warnings=warnings,
    )
    checks["series"] = _temporal_series_checks(
        series,
        periods,
        manifest=manifest,
        blocking_issues=blocking_issues,
        warnings=warnings,
    )
    checks["events"] = _temporal_events_checks(
        events,
        periods,
        series,
        manifest=manifest,
        blocking_issues=blocking_issues,
    )
    checks["sources"] = _temporal_source_checks(result_root=result_root, manifest=manifest, warnings=warnings)

    qa_path = paths["qa"]
    if not qa_path.exists():
        warnings.append(
            _temporal_issue(
                "missing_temporal_qa_sidecar",
                "warning",
                "temporal_qa.json is missing; writers should persist the validation result.",
                artifact="qa",
            )
        )

    event_counts = (
        {str(key): int(value) for key, value in events["event_type"].value_counts().to_dict().items()}
        if events is not None and "event_type" in events.columns
        else {}
    )
    counts = {
        "periods": int(0 if periods is None else len(periods)),
        "activity_rows": int(0 if activity is None else len(activity)),
        "series_rows": int(0 if series is None else len(series)),
        "event_rows": int(0 if events is None else len(events)),
        "missing_years": int(
            0
            if activity is None or "unknown_year_count" not in activity.columns
            else pd.to_numeric(activity["unknown_year_count"], errors="coerce").max() or 0
        ),
        "warnings": len(warnings),
        "blocking_issues": len(blocking_issues),
    }
    if blocking_issues:
        status = "blocked"
    elif warnings:
        status = "warning"
    else:
        status = "passed"

    return TemporalArtifactValidationResult(
        schema_version=TEMPORAL_QA_SCHEMA_VERSION,
        temporal_id=str(manifest.get("temporal_id")) if manifest.get("temporal_id") else None,
        status=status,
        temporal_dir=str(temporal_dir),
        manifest_path=_rel(manifest_path, temporal_dir) or str(manifest_path),
        paths=_temporal_path_payload(paths, temporal_dir),
        counts=counts,
        event_counts=event_counts,
        checks=checks,
        warnings=warnings,
        blocking_issues=blocking_issues,
        created_at_utc=_utc_now(),
    )


def _temporal_year_column(records: pd.DataFrame) -> str | None:
    for column in ("pubyear", "year", "publication_year"):
        if column in records.columns:
            return column
    return None


def _temporal_uid_column(records: pd.DataFrame) -> str | None:
    for column in ("uid", "work_id", "id"):
        if column in records.columns:
            return column
    return None


def _temporal_metric_defs(metric_names: list[str]) -> list[dict[str, Any]]:
    definitions: list[dict[str, Any]] = []
    for metric in metric_names:
        if metric == "doc_count":
            definitions.append(
                {
                    "name": "doc_count",
                    "value_type": "integer",
                    "denominator": None,
                    "normalization": "none",
                    "interpretation": "documents assigned to the entity during the period",
                }
            )
        else:
            definitions.append(
                {
                    "name": metric,
                    "value_type": "float",
                    "denominator": None,
                    "normalization": "as_recorded",
                    "interpretation": f"temporal series value recorded in `{metric}`",
                }
            )
    return definitions


def _temporal_series_dict(value: Any) -> dict[int, float]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return {}
    payload = value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return {}
    if not isinstance(payload, Mapping):
        return {}
    out: dict[int, float] = {}
    for key, raw in payload.items():
        year = _coerce_int(key)
        amount = _coerce_float(raw)
        if year is not None and amount is not None:
            out[year] = float(amount)
    return out


def _temporal_cluster_columns(membership: pd.DataFrame | None) -> list[str]:
    if membership is None:
        return []
    return [column for column in membership.columns if column == "cluster" or column.startswith("cluster_")]


def _temporal_level_from_cluster_column(column: str) -> str:
    if column == "cluster":
        return "cluster"
    return column.removeprefix("cluster_") or "cluster"


def _temporal_default_sources(root: Path, artifacts: ResultArtifacts) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for role, path in [
        ("records", artifacts.abstracts_path),
        ("membership", artifacts.membership_path),
        ("keywords", artifacts.keywords_path),
        ("edges", artifacts.edges_path),
    ]:
        rel_path = _rel(path, root)
        if rel_path:
            sources.append({"role": role, "path": rel_path})
    return sources


def _temporal_build_periods(temporal_id: str, years: list[int], periodization: Mapping[str, Any] | None) -> pd.DataFrame:
    if not years:
        raise ValueError("temporal artifacts require at least one valid publication year")
    start_year = int(periodization.get("start_year", min(years))) if periodization else min(years)
    end_year = int(periodization.get("end_year", max(years))) if periodization else max(years)
    if end_year < start_year:
        raise ValueError("temporal end_year must be greater than or equal to start_year")
    rows = []
    for index, year in enumerate(range(start_year, end_year + 1)):
        rows.append(
            {
                "schema_version": TEMPORAL_PERIODS_SCHEMA_VERSION,
                "temporal_id": temporal_id,
                "period_id": f"year:{year}",
                "period_index": int(index),
                "period_label": str(year),
                "start_year": int(year),
                "end_year": int(year),
                "unit": "year",
            }
        )
    return pd.DataFrame(rows)


def _temporal_activity_rows(
    temporal_id: str,
    records: pd.DataFrame,
    periods: pd.DataFrame,
    *,
    year_column: str,
    uid_column: str | None,
    membership: pd.DataFrame | None,
    edges: pd.DataFrame | None,
    unknown_year_count: int,
) -> pd.DataFrame:
    year_values = pd.to_numeric(records[year_column], errors="coerce")
    work = records.copy()
    work["_temporal_year"] = year_values
    if uid_column:
        work["_temporal_uid"] = work[uid_column].map(str)

    joined_membership = None
    cluster_columns = _temporal_cluster_columns(membership)
    if membership is not None and uid_column and "uid" in membership.columns and cluster_columns:
        mem = membership[["uid", *cluster_columns]].copy()
        mem["uid"] = mem["uid"].map(str)
        joined_membership = work[["_temporal_uid", "_temporal_year"]].merge(
            mem,
            left_on="_temporal_uid",
            right_on="uid",
            how="left",
        )

    edge_years: pd.DataFrame | None = None
    if edges is not None and uid_column and {"uid1", "uid2"}.issubset(edges.columns):
        year_lookup = dict(zip(work["_temporal_uid"], work["_temporal_year"]))
        edge_years = edges[["uid1", "uid2"]].copy()
        edge_years["_year1"] = edge_years["uid1"].map(lambda value: year_lookup.get(str(value)))
        edge_years["_year2"] = edge_years["uid2"].map(lambda value: year_lookup.get(str(value)))

    rows = []
    for period in periods.itertuples(index=False):
        year = int(period.start_year)
        year_records = work[work["_temporal_year"] == year]
        active_cluster_count = None
        if joined_membership is not None and cluster_columns:
            active: set[str] = set()
            period_membership = joined_membership[joined_membership["_temporal_year"] == year]
            for column in cluster_columns:
                active.update(period_membership[column].dropna().map(str).tolist())
            active_cluster_count = len(active)
        edge_count = None
        if edge_years is not None:
            edge_count = int(((edge_years["_year1"] == year) & (edge_years["_year2"] == year)).sum())
        rows.append(
            {
                "schema_version": TEMPORAL_ACTIVITY_SCHEMA_VERSION,
                "temporal_id": temporal_id,
                "period_id": period.period_id,
                "start_year": int(period.start_year),
                "end_year": int(period.end_year),
                "doc_count": int(len(year_records)),
                "edge_count": edge_count,
                "active_cluster_count": active_cluster_count,
                "unknown_year_count": int(unknown_year_count),
            }
        )
    return pd.DataFrame(rows)


def _temporal_result_series(temporal_id: str, activity: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for row in activity.itertuples(index=False):
        value = int(row.doc_count)
        rows.append(
            {
                "schema_version": TEMPORAL_ENTITY_SERIES_SCHEMA_VERSION,
                "temporal_id": temporal_id,
                "entity_type": "result",
                "entity_key": "result",
                "entity_label": "Result",
                "period_id": row.period_id,
                "metric": "doc_count",
                "value": float(value),
                "raw_value": float(value),
                "denominator": None,
                "support_count": value,
            }
        )
    return rows


def _temporal_cluster_series(
    temporal_id: str,
    records: pd.DataFrame,
    periods: pd.DataFrame,
    *,
    year_column: str,
    uid_column: str | None,
    membership: pd.DataFrame | None,
) -> list[dict[str, Any]]:
    if membership is None or not uid_column or "uid" not in membership.columns:
        return []
    cluster_columns = _temporal_cluster_columns(membership)
    if not cluster_columns:
        return []
    work = records[[uid_column, year_column]].copy()
    work[uid_column] = work[uid_column].map(str)
    work["_temporal_year"] = pd.to_numeric(work[year_column], errors="coerce")
    mem = membership[["uid", *cluster_columns]].copy()
    mem["uid"] = mem["uid"].map(str)
    joined = work.merge(mem, left_on=uid_column, right_on="uid", how="inner")
    year_to_period = {int(row.start_year): str(row.period_id) for row in periods.itertuples(index=False)}
    rows = []
    for column in cluster_columns:
        level = _temporal_level_from_cluster_column(column)
        grouped = (
            joined.dropna(subset=[column, "_temporal_year"])
            .groupby([column, "_temporal_year"], sort=True)
            .size()
            .reset_index(name="doc_count")
        )
        for _, item in grouped.iterrows():
            year = _coerce_int(item["_temporal_year"])
            if year is None or year not in year_to_period:
                continue
            cluster_id = str(item[column])
            value = int(item["doc_count"])
            rows.append(
                {
                    "schema_version": TEMPORAL_ENTITY_SERIES_SCHEMA_VERSION,
                    "temporal_id": temporal_id,
                    "entity_type": "cluster",
                    "entity_key": f"{level}:{cluster_id}",
                    "entity_label": f"{level}:{cluster_id}",
                    "period_id": year_to_period[year],
                    "metric": "doc_count",
                    "value": float(value),
                    "raw_value": float(value),
                    "denominator": None,
                    "support_count": value,
                    "cluster_id": cluster_id,
                    "cluster_uid": f"{level}:{cluster_id}",
                    "level": level,
                }
            )
    return rows


def _temporal_keyword_series(
    temporal_id: str,
    periods: pd.DataFrame,
    keywords: pd.DataFrame | None,
) -> tuple[list[dict[str, Any]], list[str]]:
    if keywords is None or keywords.empty:
        return [], []
    label_col = _keyword_label_column(list(keywords.columns))
    metric_cols = [column for column in ("pub_year_series", "ppm_series", "loglift_series", "temporal") if column in keywords.columns]
    if not metric_cols:
        return [], []
    year_to_period = {int(row.start_year): str(row.period_id) for row in periods.itertuples(index=False)}
    rows = []
    for _, item in keywords.iterrows():
        label = str(item.get(label_col) or "").strip()
        if not label:
            continue
        cluster_id = str(item.get("cluster_id")) if "cluster_id" in keywords.columns and item.get("cluster_id") is not None else None
        for metric in metric_cols:
            series = _temporal_series_dict(item.get(metric))
            for year, amount in sorted(series.items()):
                if year not in year_to_period:
                    continue
                row = {
                    "schema_version": TEMPORAL_ENTITY_SERIES_SCHEMA_VERSION,
                    "temporal_id": temporal_id,
                    "entity_type": "term",
                    "entity_key": f"term:{label}",
                    "entity_label": label,
                    "period_id": year_to_period[year],
                    "metric": metric,
                    "value": float(amount),
                    "raw_value": float(amount),
                    "denominator": None,
                    "support_count": None,
                    "term": label,
                }
                if cluster_id is not None:
                    row["cluster_id"] = cluster_id
                    row["cluster_uid"] = f"cluster:{cluster_id}"
                rows.append(row)
    return rows, metric_cols


def _temporal_event_rows(
    temporal_id: str,
    series: pd.DataFrame,
    *,
    event_methods: list[str],
) -> pd.DataFrame | None:
    if not event_methods or series.empty:
        return None
    if "growth_rate" not in event_methods:
        return None
    rows = []
    work = series.sort_values(["entity_type", "entity_key", "metric", "period_id"], kind="stable")
    for (entity_type, entity_key, metric), group in work.groupby(["entity_type", "entity_key", "metric"], sort=True):
        if len(group) < 2:
            continue
        first = group.iloc[0]
        last = group.iloc[-1]
        first_value = float(first["value"])
        final_value = float(last["value"])
        denom = abs(first_value) if abs(first_value) > 0 else 1.0
        growth_rate = (final_value - first_value) / denom
        if growth_rate == 0:
            continue
        event_type = "growth" if growth_rate > 0 else "decline"
        event_id = _safe_id(f"{event_type}_{entity_type}_{entity_key}_{metric}", fallback=f"{event_type}_event")
        rows.append(
            {
                "schema_version": TEMPORAL_EVENTS_SCHEMA_VERSION,
                "temporal_id": temporal_id,
                "event_id": event_id,
                "event_type": event_type,
                "entity_type": str(entity_type),
                "entity_key": str(entity_key),
                "entity_label": str(last.get("entity_label") or entity_key),
                "start_period_id": str(first["period_id"]),
                "end_period_id": str(last["period_id"]),
                "metric": str(metric),
                "score": float(abs(growth_rate)),
                "method": "growth_rate",
                "support_count": int(len(group)),
                "baseline_value": first_value,
                "final_value": final_value,
                "growth_rate": float(growth_rate),
                "duration_periods": int(len(group)),
            }
        )
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=sorted(REQUIRED_TEMPORAL_EVENTS_COLUMNS))


def write_temporal_artifacts(
    result_root: str | Path,
    *,
    temporal_id: str,
    records_df: pd.DataFrame,
    membership_df: pd.DataFrame | None = None,
    keywords_df: pd.DataFrame | None = None,
    edge_df: pd.DataFrame | None = None,
    periodization: Mapping[str, Any] | None = None,
    metrics: list[Mapping[str, Any]] | None = None,
    event_methods: list[str] | None = None,
    source_artifacts: list[Mapping[str, Any]] | None = None,
    rule_sets: list[Mapping[str, Any]] | None = None,
    transforms: list[Mapping[str, Any]] | None = None,
    output_dir: str | Path | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    """Write yearly temporal trend artifacts for a result root."""

    root = Path(result_root).expanduser().resolve()
    temporal_id = _safe_id(temporal_id, fallback="yearly_trends")
    if records_df.empty:
        raise ValueError("records_df must not be empty")
    year_column = _temporal_year_column(records_df)
    if year_column is None:
        raise ValueError("records_df must include pubyear, year, or publication_year")
    years_raw = pd.to_numeric(records_df[year_column], errors="coerce")
    valid_years = sorted({int(year) for year in years_raw.dropna().tolist() if int(year) > 0})
    if not valid_years:
        raise ValueError("records_df has no valid publication years")
    unknown_year_count = int(len(records_df) - len(years_raw.dropna()))
    periods = _temporal_build_periods(temporal_id, valid_years, periodization)
    uid_column = _temporal_uid_column(records_df)
    activity = _temporal_activity_rows(
        temporal_id,
        records_df,
        periods,
        year_column=year_column,
        uid_column=uid_column,
        membership=membership_df,
        edges=edge_df,
        unknown_year_count=unknown_year_count,
    )
    series_rows = _temporal_result_series(temporal_id, activity)
    series_rows.extend(
        _temporal_cluster_series(
            temporal_id,
            records_df,
            periods,
            year_column=year_column,
            uid_column=uid_column,
            membership=membership_df,
        )
    )
    keyword_rows, keyword_metrics = _temporal_keyword_series(temporal_id, periods, keywords_df)
    series_rows.extend(keyword_rows)
    series = pd.DataFrame(series_rows)
    if series.empty:
        series = pd.DataFrame(columns=sorted(REQUIRED_TEMPORAL_SERIES_COLUMNS))

    metric_names = ["doc_count", *[metric for metric in keyword_metrics if metric != "doc_count"]]
    metric_defs = [dict(item) for item in metrics] if metrics is not None else _temporal_metric_defs(metric_names)
    event_methods = [str(method) for method in (event_methods or [])]
    events = _temporal_event_rows(temporal_id, series, event_methods=event_methods)
    event_types = sorted(set(events["event_type"].map(str))) if events is not None and not events.empty else []

    temporal_dir = Path(output_dir).expanduser().resolve() if output_dir else root / "temporal"
    temporal_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "periods": "periods.parquet",
        "activity": "activity.parquet",
        "series": "entity_series.parquet",
        "events": "temporal_events.parquet",
        "qa": "temporal_qa.json",
    }
    sources = [dict(item) for item in source_artifacts] if source_artifacts is not None else _temporal_default_sources(root, infer_result_artifacts(root))
    periodization_payload = {
        "unit": "year",
        "window_years": 1,
        "step_years": 1,
        "start_year": int(periods["start_year"].min()),
        "end_year": int(periods["start_year"].max()),
        "closed": "point",
        "include_unknown_year": False,
    }
    if periodization:
        periodization_payload.update(dict(periodization))
        periodization_payload["unit"] = "year"
    entity_types = sorted(set(series["entity_type"].map(str))) if not series.empty and "entity_type" in series.columns else ["result"]
    manifest = {
        "schema_version": TEMPORAL_MANIFEST_SCHEMA_VERSION,
        "temporal_id": temporal_id,
        "title": title or temporal_id.replace("_", " ").title(),
        "result_id": _matrix_existing_result_id(root),
        "periodization": periodization_payload,
        "entity_types": entity_types,
        "metrics": metric_defs,
        "event_types": event_types,
        "source_artifacts": sources,
        "rule_sets": [dict(item) for item in (rule_sets or [])],
        "transforms": [
            {"step": "parse_publication_years"},
            {"step": "build_yearly_periods"},
            {"step": "aggregate_activity"},
            {"step": "aggregate_entity_series"},
            *([{"step": "detect_temporal_events", "methods": event_methods}] if event_methods else []),
            *[dict(item) for item in (transforms or [])],
        ],
        "outputs": outputs,
        "created_at_utc": _utc_now(),
        "warnings": [],
    }
    periods.to_parquet(temporal_dir / outputs["periods"], index=False)
    activity.to_parquet(temporal_dir / outputs["activity"], index=False)
    series.to_parquet(temporal_dir / outputs["series"], index=False)
    if events is not None:
        events.to_parquet(temporal_dir / outputs["events"], index=False)
    (temporal_dir / "temporal_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    validation = validate_temporal_artifact(temporal_dir)
    qa_payload = validation.to_dict()
    qa_payload["warnings"] = [
        warning for warning in qa_payload["warnings"] if warning.get("code") != "missing_temporal_qa_sidecar"
    ]
    qa_payload["counts"]["warnings"] = len(qa_payload["warnings"])
    if qa_payload["status"] == "warning" and not qa_payload["warnings"]:
        qa_payload["status"] = "passed"
    (temporal_dir / outputs["qa"]).write_text(json.dumps(qa_payload, indent=2, sort_keys=True), encoding="utf-8")
    validation = validate_temporal_artifact(temporal_dir)
    return {
        "schema_version": TEMPORAL_MANIFEST_SCHEMA_VERSION,
        "temporal_id": temporal_id,
        "temporal_dir": temporal_dir,
        "manifest_path": temporal_dir / "temporal_manifest.json",
        "periods_path": temporal_dir / outputs["periods"],
        "activity_path": temporal_dir / outputs["activity"],
        "series_path": temporal_dir / outputs["series"],
        "events_path": temporal_dir / outputs["events"] if events is not None else None,
        "qa_path": temporal_dir / outputs["qa"],
        "qa": validation.to_dict(),
    }


def _evolution_issue(
    code: str,
    severity: str,
    message: str,
    *,
    artifact: str | None = None,
) -> dict[str, Any]:
    issue = {"code": code, "severity": severity, "message": message}
    if artifact:
        issue["artifact"] = artifact
    return issue


def _evolution_dir_and_manifest(path: str | Path) -> tuple[Path, Path]:
    candidate = Path(path).expanduser()
    if candidate.exists():
        candidate = candidate.resolve()
    if candidate.is_file():
        return candidate.parent, candidate
    return candidate, candidate / "evolution_manifest.json"


def _evolution_result_root(evolution_dir: Path) -> Path:
    if evolution_dir.name == "evolution":
        scope_root = evolution_dir.parent
        if scope_root.name.startswith("landscape") and scope_root.parent.exists():
            return scope_root.parent
        return scope_root
    return evolution_dir.parent


def _evolution_output_paths(evolution_dir: Path, manifest: Mapping[str, Any]) -> dict[str, Path]:
    outputs = manifest.get("outputs") if isinstance(manifest.get("outputs"), Mapping) else {}
    paths = {
        "time_slices": evolution_dir / str(outputs.get("time_slices") or "time_slices.parquet"),
        "cluster_states": evolution_dir / str(outputs.get("cluster_states") or "cluster_states.parquet"),
        "transitions": evolution_dir / str(outputs.get("transitions") or "transitions.parquet"),
        "lineages": evolution_dir / str(outputs.get("lineages") or "lineages.parquet"),
        "events": evolution_dir / str(outputs.get("events") or "evolution_events.parquet"),
        "qa": evolution_dir / str(outputs.get("qa") or "evolution_qa.json"),
    }
    if outputs.get("state_membership"):
        paths["state_membership"] = evolution_dir / str(outputs.get("state_membership"))
    if outputs.get("synthetic_smoke"):
        paths["synthetic_smoke"] = evolution_dir / str(outputs.get("synthetic_smoke"))
    return paths


def _evolution_path_payload(paths: Mapping[str, Path], evolution_dir: Path) -> dict[str, str | None]:
    return {key: _rel(path, evolution_dir) for key, path in paths.items()}


def _evolution_required_fields(manifest: Mapping[str, Any]) -> set[str]:
    required = {
        "schema_version",
        "evolution_id",
        "title",
        "slice_method",
        "matching_method",
        "event_rules",
        "entity_scope",
        "metrics",
        "source_artifacts",
        "rule_sets",
        "transforms",
        "outputs",
        "created_at_utc",
    }
    return {key for key in required if manifest.get(key) in (None, "")}


def _evolution_read_parquet(
    path: Path,
    *,
    artifact: str,
    required: bool,
    blocking_issues: list[dict[str, Any]],
) -> pd.DataFrame | None:
    if not path.exists():
        if required:
            blocking_issues.append(
                _evolution_issue(
                    "missing_evolution_table",
                    "blocking",
                    f"Missing evolution {artifact} table.",
                    artifact=artifact,
                )
            )
        return None
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        blocking_issues.append(
            _evolution_issue(
                "invalid_evolution_parquet",
                "blocking",
                f"Could not read evolution {artifact} parquet: {exc}",
                artifact=artifact,
            )
        )
        return None


def _evolution_missing_columns(
    df: pd.DataFrame | None,
    required: set[str],
    *,
    artifact: str,
    blocking_issues: list[dict[str, Any]],
) -> set[str]:
    columns = set(df.columns) if df is not None else set()
    missing = required - columns
    if missing:
        blocking_issues.append(
            _evolution_issue(
                "missing_evolution_columns",
                "blocking",
                f"Missing required evolution columns: {sorted(missing)}",
                artifact=artifact,
            )
        )
    return missing


def _evolution_numeric_finite(
    df: pd.DataFrame | None,
    columns: list[str],
    *,
    artifact: str,
    blocking_issues: list[dict[str, Any]],
) -> None:
    if df is None:
        return
    for column in columns:
        if column not in df.columns:
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")
        mask = df[column].notna()
        bad = numeric[mask].isna()
        if bad.any() or (~numeric[mask].map(lambda value: math.isfinite(float(value)))).any():
            blocking_issues.append(
                _evolution_issue(
                    "invalid_evolution_numeric_values",
                    "blocking",
                    f"Evolution {column} values must be finite numeric values.",
                    artifact=artifact,
                )
            )


def _evolution_metric_ranges(manifest: Mapping[str, Any]) -> dict[str, tuple[float, float]]:
    ranges: dict[str, tuple[float, float]] = {}
    metrics = manifest.get("metrics")
    if not isinstance(metrics, list):
        return ranges
    for metric in metrics:
        if not isinstance(metric, Mapping) or not metric.get("name"):
            continue
        raw_range = metric.get("range")
        if isinstance(raw_range, (list, tuple)) and len(raw_range) == 2:
            low = _coerce_float(raw_range[0])
            high = _coerce_float(raw_range[1])
            if low is not None and high is not None:
                ranges[str(metric["name"])] = (float(low), float(high))
    return ranges


def _evolution_refs(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    payload = value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = [part.strip() for part in re.split(r"[|,]", text) if part.strip()]
    if isinstance(payload, (list, tuple, set)):
        return [str(item) for item in payload if str(item).strip()]
    return [str(payload)] if str(payload).strip() else []


def _evolution_time_slice_checks(
    slices: pd.DataFrame | None,
    *,
    manifest: Mapping[str, Any],
    blocking_issues: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
) -> dict[str, Any]:
    if slices is None or _evolution_missing_columns(
        slices,
        REQUIRED_EVOLUTION_TIME_SLICE_COLUMNS,
        artifact="time_slices",
        blocking_issues=blocking_issues,
    ):
        return {"status": "blocked", "count": 0}
    evolution_id = str(manifest.get("evolution_id"))
    if slices.empty:
        blocking_issues.append(
            _evolution_issue("empty_evolution_slices", "blocking", "Evolution must contain at least one time slice.", artifact="time_slices")
        )
    if not set(slices["schema_version"]) <= {EVOLUTION_TIME_SLICES_SCHEMA_VERSION}:
        blocking_issues.append(
            _evolution_issue("unsupported_evolution_slices_schema", "blocking", "Unsupported time_slices schema.", artifact="time_slices")
        )
    if not set(slices["evolution_id"].map(str)) <= {evolution_id}:
        blocking_issues.append(
            _evolution_issue("evolution_slices_id_mismatch", "blocking", "Slice evolution_id does not match manifest.", artifact="time_slices")
        )
    if slices["slice_id"].duplicated().any():
        blocking_issues.append(
            _evolution_issue("duplicate_evolution_slices", "blocking", "slice_id values are duplicated.", artifact="time_slices")
        )
    try:
        indices = sorted(int(value) for value in slices["slice_index"].tolist())
    except Exception:
        indices = []
        blocking_issues.append(
            _evolution_issue("invalid_evolution_slice_index", "blocking", "slice_index must be integer-like.", artifact="time_slices")
        )
    if indices and indices != list(range(len(indices))):
        warnings.append(
            _evolution_issue("non_contiguous_evolution_slice_index", "warning", "slice_index is not contiguous from zero.", artifact="time_slices")
        )
    if len(slices) < 2:
        warnings.append(
            _evolution_issue("sparse_evolution_slices", "warning", "Evolution artifact has fewer than two time slices.", artifact="time_slices")
        )
    for column in ("doc_count",):
        numeric = pd.to_numeric(slices[column], errors="coerce")
        if numeric.dropna().lt(0).any():
            blocking_issues.append(
                _evolution_issue("negative_evolution_slice_count", "blocking", f"{column} must be non-negative.", artifact="time_slices")
            )
    return {"status": "passed", "count": int(len(slices))}


def _evolution_state_checks(
    states: pd.DataFrame | None,
    slices: pd.DataFrame | None,
    *,
    manifest: Mapping[str, Any],
    blocking_issues: list[dict[str, Any]],
) -> dict[str, Any]:
    if states is None or _evolution_missing_columns(
        states,
        REQUIRED_EVOLUTION_STATE_COLUMNS,
        artifact="cluster_states",
        blocking_issues=blocking_issues,
    ):
        return {"status": "blocked", "count": 0}
    evolution_id = str(manifest.get("evolution_id"))
    if states.empty:
        blocking_issues.append(
            _evolution_issue("empty_evolution_states", "blocking", "Evolution must contain at least one cluster state.", artifact="cluster_states")
        )
    if not set(states["schema_version"]) <= {EVOLUTION_CLUSTER_STATES_SCHEMA_VERSION}:
        blocking_issues.append(
            _evolution_issue("unsupported_evolution_states_schema", "blocking", "Unsupported cluster_states schema.", artifact="cluster_states")
        )
    if not set(states["evolution_id"].map(str)) <= {evolution_id}:
        blocking_issues.append(
            _evolution_issue("evolution_states_id_mismatch", "blocking", "State evolution_id does not match manifest.", artifact="cluster_states")
        )
    if states["state_id"].duplicated().any():
        blocking_issues.append(
            _evolution_issue("duplicate_evolution_states", "blocking", "state_id values are duplicated.", artifact="cluster_states")
        )
    slice_ids = set(slices["slice_id"].map(str)) if slices is not None and "slice_id" in slices.columns else set()
    unknown_slices = set(states["slice_id"].map(str)) - slice_ids
    if unknown_slices:
        blocking_issues.append(
            _evolution_issue(
                "unknown_evolution_state_slices",
                "blocking",
                f"Cluster states reference {len(unknown_slices)} unknown slices.",
                artifact="cluster_states",
            )
        )
    doc_count = pd.to_numeric(states["doc_count"], errors="coerce")
    if doc_count.isna().any() or doc_count.le(0).any():
        blocking_issues.append(
            _evolution_issue("invalid_evolution_state_doc_count", "blocking", "State doc_count must be positive.", artifact="cluster_states")
        )
    return {"status": "passed", "count": int(len(states))}


def _evolution_transition_checks(
    transitions: pd.DataFrame | None,
    slices: pd.DataFrame | None,
    states: pd.DataFrame | None,
    *,
    manifest: Mapping[str, Any],
    blocking_issues: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
) -> dict[str, Any]:
    if transitions is None or _evolution_missing_columns(
        transitions,
        REQUIRED_EVOLUTION_TRANSITION_COLUMNS,
        artifact="transitions",
        blocking_issues=blocking_issues,
    ):
        return {"status": "blocked", "count": 0}
    evolution_id = str(manifest.get("evolution_id"))
    if not transitions.empty and not set(transitions["schema_version"]) <= {EVOLUTION_TRANSITIONS_SCHEMA_VERSION}:
        blocking_issues.append(
            _evolution_issue("unsupported_evolution_transitions_schema", "blocking", "Unsupported transitions schema.", artifact="transitions")
        )
    if not transitions.empty and not set(transitions["evolution_id"].map(str)) <= {evolution_id}:
        blocking_issues.append(
            _evolution_issue("evolution_transitions_id_mismatch", "blocking", "Transition evolution_id does not match manifest.", artifact="transitions")
        )
    if transitions.duplicated(subset=["source_state_id", "target_state_id", "metric"]).any():
        blocking_issues.append(
            _evolution_issue("duplicate_evolution_transitions", "blocking", "Duplicate source/target/metric transitions found.", artifact="transitions")
        )
    state_ids = set(states["state_id"].map(str)) if states is not None and "state_id" in states.columns else set()
    missing_sources = set(transitions["source_state_id"].map(str)) - state_ids
    missing_targets = set(transitions["target_state_id"].map(str)) - state_ids
    if missing_sources or missing_targets:
        blocking_issues.append(
            _evolution_issue(
                "missing_evolution_transition_state_refs",
                "blocking",
                f"Transitions reference {len(missing_sources) + len(missing_targets)} unknown states.",
                artifact="transitions",
            )
        )
    slice_index = (
        {str(row.slice_id): int(row.slice_index) for row in slices.itertuples(index=False)}
        if slices is not None and {"slice_id", "slice_index"}.issubset(slices.columns)
        else {}
    )
    unknown_slices = (set(transitions["source_slice_id"].map(str)) | set(transitions["target_slice_id"].map(str))) - set(slice_index)
    if unknown_slices:
        blocking_issues.append(
            _evolution_issue("unknown_evolution_transition_slices", "blocking", "Transitions reference unknown slices.", artifact="transitions")
        )
    allow_skip = bool((manifest.get("matching_method") or {}).get("allow_skip_slices")) if isinstance(manifest.get("matching_method"), Mapping) else False
    for row in transitions.itertuples(index=False):
        source_index = slice_index.get(str(row.source_slice_id))
        target_index = slice_index.get(str(row.target_slice_id))
        if source_index is None or target_index is None:
            continue
        if not allow_skip and target_index != source_index + 1:
            blocking_issues.append(
                _evolution_issue(
                    "non_adjacent_evolution_transition",
                    "blocking",
                    "Transitions must connect adjacent slices unless allow_skip_slices is declared.",
                    artifact="transitions",
                )
            )
            break
    _evolution_numeric_finite(
        transitions,
        ["score", "support_count", "source_doc_count", "target_doc_count"],
        artifact="transitions",
        blocking_issues=blocking_issues,
    )
    metric_ranges = _evolution_metric_ranges(manifest)
    for row in transitions.itertuples(index=False):
        score = _coerce_float(row.score)
        if score is None:
            continue
        low, high = metric_ranges.get(str(row.metric), (0.0, 1.0))
        if score < low or score > high:
            blocking_issues.append(
                _evolution_issue(
                    "evolution_transition_score_out_of_range",
                    "blocking",
                    f"Transition score for metric `{row.metric}` is outside [{low}, {high}].",
                    artifact="transitions",
                )
            )
            break
    relations = {"candidate", "continuation", "split_child", "merge_parent", "ambiguous"}
    unknown_relations = set(transitions["relation"].dropna().map(str)) - relations
    if unknown_relations:
        blocking_issues.append(
            _evolution_issue("unknown_evolution_transition_relation", "blocking", f"Unknown transition relations: {sorted(unknown_relations)}", artifact="transitions")
        )
    if slices is not None and len(slices) > 1 and transitions.empty:
        warnings.append(
            _evolution_issue("empty_evolution_transitions", "warning", "No adjacent-slice transition evidence was recorded.", artifact="transitions")
        )
    return {"status": "passed", "count": int(len(transitions))}


def _evolution_lineage_checks(
    lineages: pd.DataFrame | None,
    slices: pd.DataFrame | None,
    states: pd.DataFrame | None,
    *,
    manifest: Mapping[str, Any],
    blocking_issues: list[dict[str, Any]],
) -> dict[str, Any]:
    if lineages is None or _evolution_missing_columns(
        lineages,
        REQUIRED_EVOLUTION_LINEAGE_COLUMNS,
        artifact="lineages",
        blocking_issues=blocking_issues,
    ):
        return {"status": "blocked", "count": 0}
    evolution_id = str(manifest.get("evolution_id"))
    if lineages.empty:
        blocking_issues.append(
            _evolution_issue("empty_evolution_lineages", "blocking", "Evolution must contain at least one lineage row.", artifact="lineages")
        )
    if not set(lineages["schema_version"]) <= {EVOLUTION_LINEAGES_SCHEMA_VERSION}:
        blocking_issues.append(
            _evolution_issue("unsupported_evolution_lineages_schema", "blocking", "Unsupported lineages schema.", artifact="lineages")
        )
    if not set(lineages["evolution_id"].map(str)) <= {evolution_id}:
        blocking_issues.append(
            _evolution_issue("evolution_lineages_id_mismatch", "blocking", "Lineage evolution_id does not match manifest.", artifact="lineages")
        )
    state_ids = set(states["state_id"].map(str)) if states is not None and "state_id" in states.columns else set()
    unknown_states = set(lineages["state_id"].map(str)) - state_ids
    if unknown_states:
        blocking_issues.append(
            _evolution_issue("missing_evolution_lineage_state_refs", "blocking", "Lineages reference unknown states.", artifact="lineages")
        )
    slice_ids = set(slices["slice_id"].map(str)) if slices is not None and "slice_id" in slices.columns else set()
    unknown_slices = set(lineages["slice_id"].map(str)) - slice_ids
    if unknown_slices:
        blocking_issues.append(
            _evolution_issue("missing_evolution_lineage_slice_refs", "blocking", "Lineages reference unknown slices.", artifact="lineages")
        )
    allow_multi = bool((manifest.get("matching_method") or {}).get("allow_multi_lineage_state")) if isinstance(manifest.get("matching_method"), Mapping) else False
    if not allow_multi and lineages["state_id"].duplicated().any():
        blocking_issues.append(
            _evolution_issue("duplicate_evolution_lineage_state_refs", "blocking", "A state appears in multiple lineages.", artifact="lineages")
        )
    _evolution_numeric_finite(lineages, ["stability_score"], artifact="lineages", blocking_issues=blocking_issues)
    score = pd.to_numeric(lineages["stability_score"], errors="coerce")
    if score.dropna().lt(0).any() or score.dropna().gt(1).any():
        blocking_issues.append(
            _evolution_issue("evolution_lineage_stability_out_of_range", "blocking", "Lineage stability_score must be in [0, 1].", artifact="lineages")
        )
    return {"status": "passed", "count": int(len(lineages))}


def _evolution_transition_lookup(transitions: pd.DataFrame | None) -> dict[str, dict[str, str]]:
    if transitions is None or transitions.empty or "transition_id" not in transitions.columns:
        return {}
    lookup: dict[str, dict[str, str]] = {}
    for row in transitions.itertuples(index=False):
        lookup[str(row.transition_id)] = {
            "source": str(row.source_state_id),
            "target": str(row.target_state_id),
            "source_slice": str(row.source_slice_id),
            "target_slice": str(row.target_slice_id),
        }
    return lookup


def _evolution_event_checks(
    events: pd.DataFrame | None,
    states: pd.DataFrame | None,
    lineages: pd.DataFrame | None,
    transitions: pd.DataFrame | None,
    *,
    manifest: Mapping[str, Any],
    blocking_issues: list[dict[str, Any]],
) -> dict[str, Any]:
    if events is None or _evolution_missing_columns(
        events,
        REQUIRED_EVOLUTION_EVENT_COLUMNS,
        artifact="events",
        blocking_issues=blocking_issues,
    ):
        return {"status": "blocked", "count": 0}
    evolution_id = str(manifest.get("evolution_id"))
    if not events.empty and not set(events["schema_version"]) <= {EVOLUTION_EVENTS_SCHEMA_VERSION}:
        blocking_issues.append(
            _evolution_issue("unsupported_evolution_events_schema", "blocking", "Unsupported evolution_events schema.", artifact="events")
        )
    if not events.empty and not set(events["evolution_id"].map(str)) <= {evolution_id}:
        blocking_issues.append(
            _evolution_issue("evolution_events_id_mismatch", "blocking", "Event evolution_id does not match manifest.", artifact="events")
        )
    unknown_event_types = set(events["event_type"].dropna().map(str)) - EVOLUTION_EVENT_TYPES
    if unknown_event_types:
        blocking_issues.append(
            _evolution_issue("unknown_evolution_event_type", "blocking", f"Unknown event types: {sorted(unknown_event_types)}", artifact="events")
        )
    state_ids = set(states["state_id"].map(str)) if states is not None and "state_id" in states.columns else set()
    unknown_states = set(events["state_id"].dropna().map(str)) - state_ids
    if unknown_states:
        blocking_issues.append(
            _evolution_issue("missing_evolution_event_state_refs", "blocking", "Events reference unknown states.", artifact="events")
        )
    lineage_ids = set(lineages["lineage_id"].map(str)) if lineages is not None and "lineage_id" in lineages.columns else set()
    event_lineages = {value for value in events["lineage_id"].dropna().map(str) if value and value.lower() != "none"}
    unknown_lineages = event_lineages - lineage_ids
    if unknown_lineages:
        blocking_issues.append(
            _evolution_issue("missing_evolution_event_lineage_refs", "blocking", "Events reference unknown lineages.", artifact="events")
        )
    transition_lookup = _evolution_transition_lookup(transitions)
    transition_ids = set(transition_lookup)
    for row in events.itertuples(index=False):
        event_type = str(row.event_type)
        refs = _evolution_refs(row.transition_refs)
        unknown_refs = set(refs) - transition_ids
        if unknown_refs:
            blocking_issues.append(
                _evolution_issue("missing_evolution_event_transition_refs", "blocking", "Events reference unknown transitions.", artifact="events")
            )
            break
        if event_type in {"continuation", "split", "merge", "ambiguous"} and not refs:
            blocking_issues.append(
                _evolution_issue(
                    "missing_evolution_event_support",
                    "blocking",
                    f"{event_type} events require transition_refs.",
                    artifact="events",
                )
            )
            break
        if event_type == "split":
            targets = set(_evolution_refs(getattr(row, "target_state_ids", None)))
            if not targets:
                targets = {transition_lookup[ref]["target"] for ref in refs if ref in transition_lookup}
            if len(targets) < 2:
                blocking_issues.append(
                    _evolution_issue("invalid_evolution_split_event", "blocking", "Split events require at least two target states.", artifact="events")
                )
                break
        if event_type == "merge":
            sources = set(_evolution_refs(getattr(row, "source_state_ids", None)))
            if not sources:
                sources = {transition_lookup[ref]["source"] for ref in refs if ref in transition_lookup}
            if len(sources) < 2:
                blocking_issues.append(
                    _evolution_issue("invalid_evolution_merge_event", "blocking", "Merge events require at least two source states.", artifact="events")
                )
                break
        if event_type == "ambiguous" and len(refs) < 2:
            blocking_issues.append(
                _evolution_issue("invalid_evolution_ambiguous_event", "blocking", "Ambiguous events require at least two transition refs.", artifact="events")
            )
            break
    _evolution_numeric_finite(events, ["score", "support_count"], artifact="events", blocking_issues=blocking_issues)
    score = pd.to_numeric(events["score"], errors="coerce")
    if score.dropna().lt(0).any() or score.dropna().gt(1).any():
        blocking_issues.append(
            _evolution_issue("evolution_event_score_out_of_range", "blocking", "Event score must be in [0, 1].", artifact="events")
        )
    return {"status": "passed", "count": int(len(events))}


def _evolution_source_checks(
    *,
    result_root: Path,
    manifest: Mapping[str, Any],
    warnings: list[dict[str, Any]],
) -> dict[str, Any]:
    source_artifacts = manifest.get("source_artifacts")
    if not isinstance(source_artifacts, list) or not source_artifacts:
        warnings.append(
            _evolution_issue(
                "missing_evolution_source_artifacts",
                "warning",
                "Evolution manifest should record at least one source artifact.",
                artifact="manifest",
            )
        )
        return {"status": "warning", "count": 0, "missing": 0}
    missing = 0
    for source in source_artifacts:
        if not isinstance(source, Mapping):
            warnings.append(_evolution_issue("invalid_evolution_source_artifact", "warning", "Source refs should be objects.", artifact="manifest"))
            continue
        path = source.get("path")
        if not path:
            warnings.append(_evolution_issue("missing_evolution_source_path", "warning", "Source ref has no path.", artifact="manifest"))
            continue
        source_path = Path(str(path))
        resolved = source_path if source_path.is_absolute() else result_root / source_path
        if not resolved.exists():
            missing += 1
    if missing:
        warnings.append(
            _evolution_issue(
                "missing_evolution_source_artifact",
                "warning",
                f"{missing} evolution source artifact refs do not exist.",
                artifact="manifest",
            )
        )
    return {"status": "warning" if missing else "passed", "count": len(source_artifacts), "missing": missing}


def _evolution_qa_checks(
    qa_path: Path,
    *,
    warnings: list[dict[str, Any]],
    blocking_issues: list[dict[str, Any]],
) -> dict[str, Any]:
    if not qa_path.exists():
        warnings.append(
            _evolution_issue("missing_evolution_qa_sidecar", "warning", "evolution_qa.json has not been written yet.", artifact="qa")
        )
        return {"status": "warning"}
    try:
        payload = json.loads(qa_path.read_text(encoding="utf-8"))
    except Exception as exc:
        blocking_issues.append(
            _evolution_issue("invalid_evolution_qa_json", "blocking", f"Could not read evolution QA sidecar: {exc}", artifact="qa")
        )
        return {"status": "blocked"}
    if not isinstance(payload, dict):
        blocking_issues.append(_evolution_issue("invalid_evolution_qa_shape", "blocking", "Evolution QA sidecar must be an object.", artifact="qa"))
        return {"status": "blocked"}
    if payload.get("schema_version") != EVOLUTION_QA_SCHEMA_VERSION:
        blocking_issues.append(_evolution_issue("unsupported_evolution_qa_schema", "blocking", "Unsupported evolution QA schema.", artifact="qa"))
        return {"status": "blocked"}
    if payload.get("status") == "blocked":
        blocking_issues.append(_evolution_issue("evolution_qa_blocked", "blocking", "Evolution QA sidecar records blocked status.", artifact="qa"))
        return {"status": "blocked"}
    return {"status": "passed" if payload.get("status") == "passed" else "warning"}


def validate_evolution_artifact(path: str | Path) -> EvolutionArtifactValidationResult:
    """Validate a lineage-backed cluster evolution artifact directory."""

    evolution_dir, manifest_path = _evolution_dir_and_manifest(path)
    result_root = _evolution_result_root(evolution_dir)
    warnings: list[dict[str, Any]] = []
    blocking_issues: list[dict[str, Any]] = []
    checks: dict[str, dict[str, Any]] = {}
    manifest: dict[str, Any] = {}

    if not manifest_path.exists():
        blocking_issues.append(
            _evolution_issue("missing_evolution_manifest", "blocking", "Missing evolution_manifest.json.", artifact="manifest")
        )
    else:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            blocking_issues.append(
                _evolution_issue("invalid_evolution_manifest_json", "blocking", f"Could not read evolution manifest: {exc}", artifact="manifest")
            )
        if not isinstance(manifest, dict):
            blocking_issues.append(_evolution_issue("invalid_evolution_manifest_shape", "blocking", "Evolution manifest must be an object.", artifact="manifest"))
            manifest = {}

    if manifest:
        if manifest.get("schema_version") != EVOLUTION_MANIFEST_SCHEMA_VERSION:
            blocking_issues.append(
                _evolution_issue(
                    "unsupported_evolution_manifest_schema",
                    "blocking",
                    f"Unsupported evolution manifest schema: {manifest.get('schema_version')}",
                    artifact="manifest",
                )
            )
        missing_fields = _evolution_required_fields(manifest)
        if missing_fields:
            blocking_issues.append(
                _evolution_issue(
                    "missing_evolution_manifest_fields",
                    "blocking",
                    f"Missing evolution manifest fields: {sorted(missing_fields)}",
                    artifact="manifest",
                )
            )

    paths = _evolution_output_paths(evolution_dir, manifest)
    slices = _evolution_read_parquet(paths["time_slices"], artifact="time_slices", required=True, blocking_issues=blocking_issues)
    states = _evolution_read_parquet(paths["cluster_states"], artifact="cluster_states", required=True, blocking_issues=blocking_issues)
    transitions = _evolution_read_parquet(paths["transitions"], artifact="transitions", required=True, blocking_issues=blocking_issues)
    lineages = _evolution_read_parquet(paths["lineages"], artifact="lineages", required=True, blocking_issues=blocking_issues)
    events = _evolution_read_parquet(paths["events"], artifact="events", required=True, blocking_issues=blocking_issues)
    state_membership = (
        _evolution_read_parquet(paths["state_membership"], artifact="state_membership", required=False, blocking_issues=blocking_issues)
        if "state_membership" in paths
        else None
    )

    checks["time_slices"] = _evolution_time_slice_checks(
        slices,
        manifest=manifest,
        blocking_issues=blocking_issues,
        warnings=warnings,
    )
    checks["cluster_states"] = _evolution_state_checks(
        states,
        slices,
        manifest=manifest,
        blocking_issues=blocking_issues,
    )
    checks["transitions"] = _evolution_transition_checks(
        transitions,
        slices,
        states,
        manifest=manifest,
        blocking_issues=blocking_issues,
        warnings=warnings,
    )
    checks["lineages"] = _evolution_lineage_checks(
        lineages,
        slices,
        states,
        manifest=manifest,
        blocking_issues=blocking_issues,
    )
    checks["events"] = _evolution_event_checks(
        events,
        states,
        lineages,
        transitions,
        manifest=manifest,
        blocking_issues=blocking_issues,
    )
    checks["source_artifacts"] = _evolution_source_checks(result_root=result_root, manifest=manifest, warnings=warnings)
    checks["qa"] = _evolution_qa_checks(paths["qa"], warnings=warnings, blocking_issues=blocking_issues)

    event_counts = (
        {str(key): int(value) for key, value in events["event_type"].value_counts().sort_index().items()}
        if events is not None and "event_type" in events.columns
        else {}
    )
    missing_refs = sum(
        1
        for issue in blocking_issues
        if "ref" in str(issue.get("code", "")) or "unknown" in str(issue.get("code", ""))
    )
    counts = {
        "slices": int(len(slices)) if slices is not None else 0,
        "states": int(len(states)) if states is not None else 0,
        "transitions": int(len(transitions)) if transitions is not None else 0,
        "state_membership_rows": int(len(state_membership)) if state_membership is not None else 0,
        "lineages": int(len(lineages)) if lineages is not None else 0,
        "events": int(len(events)) if events is not None else 0,
        "event_rows": int(len(events)) if events is not None else 0,
        "missing_refs": int(missing_refs),
        "warnings": len(warnings),
        "blocking_issues": len(blocking_issues),
    }
    status = "blocked" if blocking_issues else "warning" if warnings else "passed"
    return EvolutionArtifactValidationResult(
        schema_version=EVOLUTION_QA_SCHEMA_VERSION,
        evolution_id=str(manifest.get("evolution_id")) if manifest.get("evolution_id") else None,
        status=status,
        evolution_dir=str(evolution_dir),
        manifest_path=str(manifest_path),
        paths=_evolution_path_payload(paths, evolution_dir),
        counts=counts,
        event_counts=event_counts,
        checks=checks,
        warnings=warnings,
        blocking_issues=blocking_issues,
        created_at_utc=_utc_now(),
    )


def _evolution_default_sources(root: Path, artifacts: ResultArtifacts, temporal_manifest: str | Path | None = None) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for role, path in [
        ("records", artifacts.abstracts_path),
        ("membership", artifacts.membership_path),
        ("keywords", artifacts.keywords_path),
        ("edges", artifacts.edges_path),
    ]:
        rel_path = _rel(path, root)
        if rel_path:
            sources.append({"role": role, "path": rel_path})
    if temporal_manifest:
        temporal_path = Path(temporal_manifest)
        sources.append({"role": "temporal", "path": _rel(temporal_path, root) or str(temporal_manifest)})
    return sources


def _write_evolution_payload(
    root: Path,
    evolution_dir: Path,
    *,
    manifest: dict[str, Any],
    slices: pd.DataFrame,
    states: pd.DataFrame,
    transitions: pd.DataFrame,
    lineages: pd.DataFrame,
    events: pd.DataFrame,
    state_membership: pd.DataFrame | None = None,
) -> dict[str, Any]:
    evolution_dir.mkdir(parents=True, exist_ok=True)
    outputs = manifest["outputs"]
    slices.to_parquet(evolution_dir / outputs["time_slices"], index=False)
    states.to_parquet(evolution_dir / outputs["cluster_states"], index=False)
    transitions.to_parquet(evolution_dir / outputs["transitions"], index=False)
    lineages.to_parquet(evolution_dir / outputs["lineages"], index=False)
    events.to_parquet(evolution_dir / outputs["events"], index=False)
    if state_membership is not None and outputs.get("state_membership"):
        state_membership.to_parquet(evolution_dir / outputs["state_membership"], index=False)
    (evolution_dir / "evolution_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    validation = validate_evolution_artifact(evolution_dir)
    qa_payload = validation.to_dict()
    qa_payload["warnings"] = [
        warning for warning in qa_payload["warnings"] if warning.get("code") != "missing_evolution_qa_sidecar"
    ]
    qa_payload["counts"]["warnings"] = len(qa_payload["warnings"])
    if qa_payload["status"] == "warning" and not qa_payload["warnings"]:
        qa_payload["status"] = "passed"
    (evolution_dir / outputs["qa"]).write_text(json.dumps(qa_payload, indent=2, sort_keys=True), encoding="utf-8")
    validation = validate_evolution_artifact(evolution_dir)
    return {
        "schema_version": EVOLUTION_MANIFEST_SCHEMA_VERSION,
        "evolution_id": manifest["evolution_id"],
        "evolution_dir": evolution_dir,
        "manifest_path": evolution_dir / "evolution_manifest.json",
        "time_slices_path": evolution_dir / outputs["time_slices"],
        "cluster_states_path": evolution_dir / outputs["cluster_states"],
        "transitions_path": evolution_dir / outputs["transitions"],
        "lineages_path": evolution_dir / outputs["lineages"],
        "events_path": evolution_dir / outputs["events"],
        "state_membership_path": evolution_dir / outputs["state_membership"] if outputs.get("state_membership") else None,
        "qa_path": evolution_dir / outputs["qa"],
        "qa": validation.to_dict(),
    }


def write_evolution_artifacts(
    result_root: str | Path,
    *,
    evolution_id: str,
    records_df: pd.DataFrame,
    membership_df: pd.DataFrame,
    keywords_df: pd.DataFrame | None = None,
    temporal_manifest: str | Path | None = None,
    periodization: Mapping[str, Any] | None = None,
    matching_method: Mapping[str, Any] | None = None,
    event_rules: Mapping[str, Any] | None = None,
    source_artifacts: list[Mapping[str, Any]] | None = None,
    rule_sets: list[Mapping[str, Any]] | None = None,
    transforms: list[Mapping[str, Any]] | None = None,
    output_dir: str | Path | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    """Write v1 membership-projection cluster evolution artifacts."""

    root = Path(result_root).expanduser().resolve()
    analysis = build_membership_projection_evolution(
        evolution_id=evolution_id,
        records_df=records_df,
        membership_df=membership_df,
        keywords_df=keywords_df,
        periodization=periodization,
        matching_method=matching_method,
        event_rules=event_rules,
        transforms=transforms,
    )
    evolution_dir = Path(output_dir).expanduser().resolve() if output_dir else root / "evolution"
    outputs = {
        "time_slices": "time_slices.parquet",
        "cluster_states": "cluster_states.parquet",
        "state_membership": "state_membership.parquet",
        "transitions": "transitions.parquet",
        "lineages": "lineages.parquet",
        "events": "evolution_events.parquet",
        "qa": "evolution_qa.json",
    }
    sources = (
        [dict(item) for item in source_artifacts]
        if source_artifacts is not None
        else _evolution_default_sources(root, infer_result_artifacts(root), temporal_manifest=temporal_manifest)
    )
    manifest = {
        "schema_version": EVOLUTION_MANIFEST_SCHEMA_VERSION,
        "evolution_id": analysis.evolution_id,
        "title": title or analysis.evolution_id.replace("_", " ").title(),
        "result_id": _matrix_existing_result_id(root),
        "slice_method": analysis.periodization,
        "matching_method": analysis.matching_method,
        "event_rules": analysis.event_rules,
        "entity_scope": analysis.entity_scope,
        "metrics": analysis.metrics,
        "source_artifacts": sources,
        "rule_sets": [dict(item) for item in (rule_sets or [])],
        "transforms": analysis.transforms,
        "outputs": outputs,
        "created_at_utc": _utc_now(),
        "warnings": [],
    }
    return _write_evolution_payload(
        root,
        evolution_dir,
        manifest=manifest,
        slices=analysis.slices,
        states=analysis.states,
        transitions=analysis.transitions,
        lineages=analysis.lineages,
        events=analysis.events,
        state_membership=analysis.state_membership,
    )


def write_evidence_backed_evolution_artifacts(
    result_root: str | Path,
    *,
    evolution_id: str,
    slices_df: pd.DataFrame,
    state_evidence_df: pd.DataFrame,
    transition_evidence_df: pd.DataFrame,
    metric: str,
    temporal_manifest: str | Path | None = None,
    periodization: Mapping[str, Any] | None = None,
    matching_method: Mapping[str, Any] | None = None,
    event_rules: Mapping[str, Any] | None = None,
    entity_scope: Mapping[str, Any] | None = None,
    source_artifacts: list[Mapping[str, Any]] | None = None,
    rule_sets: list[Mapping[str, Any]] | None = None,
    transforms: list[Mapping[str, Any]] | None = None,
    output_dir: str | Path | None = None,
    title: str | None = None,
    default_level: str = "cluster",
    allow_skip_slices: bool = False,
) -> dict[str, Any]:
    """Write v1 evolution artifacts from explicit state and transition evidence."""

    root = Path(result_root).expanduser().resolve()
    analysis = build_evidence_backed_evolution(
        evolution_id=evolution_id,
        slices=slices_df,
        state_evidence=state_evidence_df,
        transition_evidence=transition_evidence_df,
        metric=metric,
        matching_method=matching_method,
        event_rules=event_rules,
        periodization=periodization,
        entity_scope=entity_scope,
        transforms=transforms,
        default_level=default_level,
        allow_skip_slices=allow_skip_slices,
    )
    evolution_dir = Path(output_dir).expanduser().resolve() if output_dir else root / "evolution"
    outputs = {
        "time_slices": "time_slices.parquet",
        "cluster_states": "cluster_states.parquet",
        "transitions": "transitions.parquet",
        "lineages": "lineages.parquet",
        "events": "evolution_events.parquet",
        "qa": "evolution_qa.json",
    }
    sources = (
        [dict(item) for item in source_artifacts]
        if source_artifacts is not None
        else _evolution_default_sources(root, infer_result_artifacts(root), temporal_manifest=temporal_manifest)
    )
    manifest = {
        "schema_version": EVOLUTION_MANIFEST_SCHEMA_VERSION,
        "evolution_id": analysis.evolution_id,
        "title": title or analysis.evolution_id.replace("_", " ").title(),
        "result_id": _matrix_existing_result_id(root),
        "slice_method": analysis.periodization,
        "matching_method": analysis.matching_method,
        "event_rules": analysis.event_rules,
        "entity_scope": analysis.entity_scope,
        "metrics": analysis.metrics,
        "source_artifacts": sources,
        "rule_sets": [dict(item) for item in (rule_sets or [])],
        "transforms": analysis.transforms,
        "outputs": outputs,
        "created_at_utc": _utc_now(),
        "warnings": [],
    }
    return _write_evolution_payload(
        root,
        evolution_dir,
        manifest=manifest,
        slices=analysis.slices,
        states=analysis.states,
        transitions=analysis.transitions,
        lineages=analysis.lineages,
        events=analysis.events,
        state_membership=analysis.state_membership,
    )


def write_document_overlap_evolution_artifacts(
    result_root: str | Path,
    *,
    evolution_id: str,
    slices_df: pd.DataFrame,
    state_evidence_df: pd.DataFrame,
    state_membership_df: pd.DataFrame,
    metric: str = "jaccard_doc_overlap",
    temporal_manifest: str | Path | None = None,
    uid_column: str | None = None,
    state_id_column: str = "state_id",
    periodization: Mapping[str, Any] | None = None,
    matching_method: Mapping[str, Any] | None = None,
    event_rules: Mapping[str, Any] | None = None,
    entity_scope: Mapping[str, Any] | None = None,
    source_artifacts: list[Mapping[str, Any]] | None = None,
    rule_sets: list[Mapping[str, Any]] | None = None,
    transforms: list[Mapping[str, Any]] | None = None,
    output_dir: str | Path | None = None,
    title: str | None = None,
    default_level: str = "cluster",
    require_complete_membership: bool = True,
) -> dict[str, Any]:
    """Write v1 evolution artifacts from state-document overlap evidence."""

    root = Path(result_root).expanduser().resolve()
    analysis = build_document_overlap_evolution(
        evolution_id=evolution_id,
        slices=slices_df,
        state_evidence=state_evidence_df,
        state_membership=state_membership_df,
        metric=metric,
        uid_column=uid_column,
        state_id_column=state_id_column,
        matching_method=matching_method,
        event_rules=event_rules,
        periodization=periodization,
        entity_scope=entity_scope,
        transforms=transforms,
        default_level=default_level,
        require_complete_membership=require_complete_membership,
    )
    evolution_dir = Path(output_dir).expanduser().resolve() if output_dir else root / "evolution"
    outputs = {
        "time_slices": "time_slices.parquet",
        "cluster_states": "cluster_states.parquet",
        "state_membership": "state_membership.parquet",
        "transitions": "transitions.parquet",
        "lineages": "lineages.parquet",
        "events": "evolution_events.parquet",
        "qa": "evolution_qa.json",
    }
    sources = (
        [dict(item) for item in source_artifacts]
        if source_artifacts is not None
        else _evolution_default_sources(root, infer_result_artifacts(root), temporal_manifest=temporal_manifest)
    )
    manifest = {
        "schema_version": EVOLUTION_MANIFEST_SCHEMA_VERSION,
        "evolution_id": analysis.evolution_id,
        "title": title or analysis.evolution_id.replace("_", " ").title(),
        "result_id": _matrix_existing_result_id(root),
        "slice_method": analysis.periodization,
        "matching_method": analysis.matching_method,
        "event_rules": analysis.event_rules,
        "entity_scope": analysis.entity_scope,
        "metrics": analysis.metrics,
        "source_artifacts": sources,
        "rule_sets": [dict(item) for item in (rule_sets or [])],
        "transforms": analysis.transforms,
        "outputs": outputs,
        "created_at_utc": _utc_now(),
        "warnings": [],
    }
    return _write_evolution_payload(
        root,
        evolution_dir,
        manifest=manifest,
        slices=analysis.slices,
        states=analysis.states,
        transitions=analysis.transitions,
        lineages=analysis.lineages,
        events=analysis.events,
        state_membership=analysis.state_membership,
    )


def write_slice_membership_evolution_artifacts(
    result_root: str | Path,
    *,
    evolution_id: str,
    records_df: pd.DataFrame,
    membership_df: pd.DataFrame,
    keywords_df: pd.DataFrame | None = None,
    metric: str = "overlap_min",
    temporal_manifest: str | Path | None = None,
    periodization: Mapping[str, Any] | None = None,
    matching_method: Mapping[str, Any] | None = None,
    event_rules: Mapping[str, Any] | None = None,
    entity_scope: Mapping[str, Any] | None = None,
    source_artifacts: list[Mapping[str, Any]] | None = None,
    rule_sets: list[Mapping[str, Any]] | None = None,
    transforms: list[Mapping[str, Any]] | None = None,
    output_dir: str | Path | None = None,
    title: str | None = None,
    cluster_column: str | None = None,
    uid_column: str | None = None,
    membership_uid_column: str | None = None,
    representative_work_limit: int = 50,
    require_complete_membership: bool = True,
) -> dict[str, Any]:
    """Write v1 document-overlap evolution artifacts from records and membership."""

    root = Path(result_root).expanduser().resolve()
    evidence = build_slice_membership_evidence(
        evolution_id=evolution_id,
        records_df=records_df,
        membership_df=membership_df,
        keywords_df=keywords_df,
        periodization=periodization,
        cluster_column=cluster_column,
        uid_column=uid_column,
        membership_uid_column=membership_uid_column,
        representative_work_limit=representative_work_limit,
    )
    merged_periodization = dict(evidence.periodization)
    if periodization:
        merged_periodization.update(dict(periodization))
    merged_scope = dict(evidence.entity_scope)
    if entity_scope:
        merged_scope.update(dict(entity_scope))
    matching = {
        "metric": metric,
        "min_transition_score": 0.5,
        "min_support_count": 1,
        "tie_policy": "keep_all_above_threshold",
        "normalization": "periodized_slice_membership_document_overlap",
    }
    if matching_method:
        matching.update(dict(matching_method))
    matching["metric"] = metric
    sources = (
        [dict(item) for item in source_artifacts]
        if source_artifacts is not None
        else _evolution_default_sources(root, infer_result_artifacts(root), temporal_manifest=temporal_manifest)
    )
    return write_document_overlap_evolution_artifacts(
        root,
        evolution_id=evidence.evolution_id,
        slices_df=evidence.slices,
        state_evidence_df=evidence.state_evidence,
        state_membership_df=evidence.state_membership,
        metric=metric,
        temporal_manifest=temporal_manifest,
        uid_column="uid",
        state_id_column="state_id",
        periodization=merged_periodization,
        matching_method=matching,
        event_rules=event_rules,
        entity_scope=merged_scope,
        source_artifacts=sources,
        rule_sets=rule_sets,
        transforms=[dict(item) for item in (transforms or [])] + evidence.transforms,
        output_dir=output_dir,
        title=title,
        default_level=str(evidence.entity_scope.get("cluster_level") or "cluster"),
        require_complete_membership=require_complete_membership,
    )


def write_slice_local_membership_evolution_artifacts(
    result_root: str | Path,
    *,
    evolution_id: str,
    slice_membership_df: pd.DataFrame,
    slices_df: pd.DataFrame | None = None,
    keywords_df: pd.DataFrame | None = None,
    metric: str = "overlap_min",
    temporal_manifest: str | Path | None = None,
    matching_method: Mapping[str, Any] | None = None,
    event_rules: Mapping[str, Any] | None = None,
    entity_scope: Mapping[str, Any] | None = None,
    source_artifacts: list[Mapping[str, Any]] | None = None,
    rule_sets: list[Mapping[str, Any]] | None = None,
    transforms: list[Mapping[str, Any]] | None = None,
    output_dir: str | Path | None = None,
    title: str | None = None,
    cluster_column: str | None = None,
    uid_column: str | None = None,
    slice_id_column: str = "slice_id",
    representative_work_limit: int = 50,
    default_level: str = "cluster",
    require_complete_membership: bool = True,
) -> dict[str, Any]:
    """Write v1 document-overlap evolution artifacts from slice-local membership."""

    root = Path(result_root).expanduser().resolve()
    evidence = build_slice_local_membership_evidence(
        evolution_id=evolution_id,
        slice_membership_df=slice_membership_df,
        slices_df=slices_df,
        keywords_df=keywords_df,
        cluster_column=cluster_column,
        uid_column=uid_column,
        slice_id_column=slice_id_column,
        representative_work_limit=representative_work_limit,
        default_level=default_level,
    )
    merged_scope = dict(evidence.entity_scope)
    if entity_scope:
        merged_scope.update(dict(entity_scope))
    matching = {
        "metric": metric,
        "min_transition_score": 0.5,
        "min_support_count": 1,
        "tie_policy": "keep_all_above_threshold",
        "normalization": "slice_local_membership_document_overlap",
    }
    if matching_method:
        matching.update(dict(matching_method))
    matching["metric"] = metric
    sources = (
        [dict(item) for item in source_artifacts]
        if source_artifacts is not None
        else _evolution_default_sources(root, infer_result_artifacts(root), temporal_manifest=temporal_manifest)
    )
    return write_document_overlap_evolution_artifacts(
        root,
        evolution_id=evidence.evolution_id,
        slices_df=evidence.slices,
        state_evidence_df=evidence.state_evidence,
        state_membership_df=evidence.state_membership,
        metric=metric,
        temporal_manifest=temporal_manifest,
        uid_column="uid",
        state_id_column="state_id",
        periodization=evidence.periodization,
        matching_method=matching,
        event_rules=event_rules,
        entity_scope=merged_scope,
        source_artifacts=sources,
        rule_sets=rule_sets,
        transforms=[dict(item) for item in (transforms or [])] + evidence.transforms,
        output_dir=output_dir,
        title=title,
        default_level=str(evidence.entity_scope.get("cluster_level") or default_level or "cluster"),
        require_complete_membership=require_complete_membership,
    )


def write_slice_reclustering_evolution_artifacts(
    result_root: str | Path,
    *,
    evolution_id: str,
    records_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    keywords_df: pd.DataFrame | None = None,
    metric: str = "overlap_min",
    temporal_manifest: str | Path | None = None,
    periodization: Mapping[str, Any] | None = None,
    matching_method: Mapping[str, Any] | None = None,
    event_rules: Mapping[str, Any] | None = None,
    entity_scope: Mapping[str, Any] | None = None,
    source_artifacts: list[Mapping[str, Any]] | None = None,
    rule_sets: list[Mapping[str, Any]] | None = None,
    output_dir: str | Path | None = None,
    title: str | None = None,
    uid_column: str | None = None,
    edge_source_column: str | None = None,
    edge_target_column: str | None = None,
    edge_weight_column: str | None = None,
    resolution: float = 1.0,
    objective: str = "cpm",
    seed: int = 0,
    n_iterations: int = 10,
    backend: str = "auto",
    min_docs_per_slice: int = 1,
    max_workers: int = 1,
    slice_membership_output: str | Path | None = None,
    slice_membership_parts_dir: str | Path | None = None,
    progress_path: str | Path | None = None,
    representative_work_limit: int = 50,
    require_complete_membership: bool = True,
) -> dict[str, Any]:
    """Write v1 evolution artifacts by reclustering induced slice graphs."""

    root = Path(result_root).expanduser().resolve()
    progress_output_path: Path | None = None
    progress_ref: str | None = None
    if progress_path is not None:
        progress_output_path = Path(progress_path).expanduser()
        if not progress_output_path.is_absolute():
            progress_output_path = root / progress_output_path
        progress_output_path = progress_output_path.resolve()
        progress_ref = _rel(progress_output_path, root) or str(progress_output_path)
    membership_parts_path: Path | None = None
    membership_parts_ref: str | None = None
    if slice_membership_parts_dir is not None:
        membership_parts_path = Path(slice_membership_parts_dir).expanduser()
        if not membership_parts_path.is_absolute():
            membership_parts_path = root / membership_parts_path
        membership_parts_path = membership_parts_path.resolve()
        membership_parts_ref = _rel(membership_parts_path, root) or str(membership_parts_path)
    slice_membership = build_slice_reclustering_membership(
        evolution_id=evolution_id,
        records_df=records_df,
        edges_df=edges_df,
        periodization=periodization,
        uid_column=uid_column,
        edge_source_column=edge_source_column,
        edge_target_column=edge_target_column,
        edge_weight_column=edge_weight_column,
        resolution=resolution,
        objective=objective,
        seed=seed,
        n_iterations=n_iterations,
        backend=backend,
        min_docs_per_slice=min_docs_per_slice,
        max_workers=max_workers,
        progress_path=progress_output_path,
        membership_parts_dir=membership_parts_path,
    )
    slice_membership_path: Path | None = None
    slice_membership_ref: str | None = None
    if slice_membership_output is not None:
        slice_membership_path = Path(slice_membership_output).expanduser()
        if not slice_membership_path.is_absolute():
            slice_membership_path = root / slice_membership_path
        slice_membership_path = slice_membership_path.resolve()
        slice_membership_path.parent.mkdir(parents=True, exist_ok=True)
        slice_membership.to_parquet(slice_membership_path, index=False)
        slice_membership_ref = _rel(slice_membership_path, root) or str(slice_membership_path)
    matching = {
        "metric": metric,
        "min_transition_score": 0.5,
        "min_support_count": 1,
        "tie_policy": "keep_all_above_threshold",
        "normalization": "slice_reclustering_document_overlap",
    }
    if matching_method:
        matching.update(dict(matching_method))
    matching["metric"] = metric
    resolved_backends = sorted(set(slice_membership["backend"].map(str))) if "backend" in slice_membership.columns else []
    sources = (
        [dict(item) for item in source_artifacts]
        if source_artifacts is not None
        else _evolution_default_sources(root, infer_result_artifacts(root), temporal_manifest=temporal_manifest)
    )
    recluster_transform = {
        "step": "run_slice_local_reclustering",
        "requested_backend": str(backend),
        "resolved_backends": resolved_backends,
        "objective": str(objective),
        "resolution": float(resolution),
        "seed": int(seed),
        "n_iterations": int(n_iterations),
        "min_docs_per_slice": int(min_docs_per_slice),
        "max_workers": int(max_workers),
        "slice_membership_rows": int(len(slice_membership)),
    }
    if slice_membership_ref is not None:
        recluster_transform["slice_membership_output"] = slice_membership_ref
    if membership_parts_ref is not None:
        recluster_transform["slice_membership_parts_dir"] = membership_parts_ref
    if progress_ref is not None:
        recluster_transform["progress_path"] = progress_ref
    default_level = str((entity_scope or {}).get("cluster_level") or "cluster")
    written = write_slice_local_membership_evolution_artifacts(
        root,
        evolution_id=evolution_id,
        slice_membership_df=slice_membership,
        keywords_df=keywords_df,
        metric=metric,
        temporal_manifest=temporal_manifest,
        matching_method=matching,
        event_rules=event_rules,
        entity_scope=entity_scope,
        source_artifacts=sources,
        rule_sets=rule_sets,
        transforms=[recluster_transform],
        output_dir=output_dir,
        title=title,
        cluster_column="cluster_id",
        uid_column="uid",
        slice_id_column="slice_id",
        representative_work_limit=representative_work_limit,
        default_level=default_level,
        require_complete_membership=require_complete_membership,
    )
    if slice_membership_path is not None:
        written["slice_membership_path"] = slice_membership_path
    if membership_parts_path is not None:
        written["slice_membership_parts_dir"] = membership_parts_path
    if progress_output_path is not None:
        written["progress_path"] = progress_output_path
    return written


def _synthetic_evolution_state(
    evolution_id: str,
    state_id: str,
    slice_id: str,
    slice_index: int,
    cluster_key: str,
    doc_count: int = 3,
) -> dict[str, Any]:
    return {
        "schema_version": EVOLUTION_CLUSTER_STATES_SCHEMA_VERSION,
        "evolution_id": evolution_id,
        "state_id": state_id,
        "slice_id": slice_id,
        "slice_index": int(slice_index),
        "cluster_key": cluster_key,
        "cluster_label": cluster_key,
        "doc_count": int(doc_count),
        "term_count": 1,
        "top_terms": json.dumps([cluster_key], ensure_ascii=True),
        "cluster_uid": cluster_key,
        "cluster_id": cluster_key.split(":")[-1],
        "level": "synthetic",
        "representative_work_ids": "[]",
        "source_cluster_key": cluster_key,
        "warning_flags": "",
    }


def _synthetic_evolution_transition(
    evolution_id: str,
    transition_id: str,
    source: str,
    target: str,
    source_slice: str,
    target_slice: str,
    *,
    relation: str,
    score: float,
    support_count: int = 3,
) -> dict[str, Any]:
    return {
        "schema_version": EVOLUTION_TRANSITIONS_SCHEMA_VERSION,
        "evolution_id": evolution_id,
        "transition_id": transition_id,
        "source_state_id": source,
        "target_state_id": target,
        "source_slice_id": source_slice,
        "target_slice_id": target_slice,
        "metric": "synthetic_overlap",
        "score": float(score),
        "support_count": int(support_count),
        "source_doc_count": int(support_count),
        "target_doc_count": int(support_count),
        "relation": relation,
        "shared_doc_count": int(support_count),
        "rank_from_source": 1,
        "rank_to_target": 1,
        "warning_flags": "",
    }


def _synthetic_evolution_event(
    evolution_id: str,
    event_id: str,
    event_type: str,
    slice_id: str,
    state_id: str,
    lineage_id: str,
    refs: list[str],
    *,
    source_state_ids: list[str] | None = None,
    target_state_ids: list[str] | None = None,
    score: float = 1.0,
    support_count: int = 3,
) -> dict[str, Any]:
    return {
        "schema_version": EVOLUTION_EVENTS_SCHEMA_VERSION,
        "evolution_id": evolution_id,
        "event_id": event_id,
        "event_type": event_type,
        "slice_id": slice_id,
        "state_id": state_id,
        "lineage_id": lineage_id,
        "transition_refs": json.dumps(refs, ensure_ascii=True),
        "score": float(score),
        "support_count": int(support_count),
        "method": "synthetic_smoke_fixture",
        "source_state_ids": json.dumps(source_state_ids or [], ensure_ascii=True),
        "target_state_ids": json.dumps(target_state_ids or [], ensure_ascii=True),
        "event_label": event_type.replace("_", " ").title(),
        "warning_flags": "",
    }


def write_evolution_synthetic_smoke_artifact(
    result_root: str | Path,
    *,
    evolution_id: str = "synthetic_all_events",
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Write a deterministic evolution smoke artifact that covers every event type."""

    root = Path(result_root).expanduser().resolve()
    evolution_id = _safe_id(evolution_id, fallback="synthetic_all_events")
    evolution_dir = Path(output_dir).expanduser().resolve() if output_dir else root / "evolution"
    outputs = {
        "time_slices": "time_slices.parquet",
        "cluster_states": "cluster_states.parquet",
        "transitions": "transitions.parquet",
        "lineages": "lineages.parquet",
        "events": "evolution_events.parquet",
        "qa": "evolution_qa.json",
        "synthetic_smoke": "synthetic_smoke_example.json",
    }
    slices = pd.DataFrame(
        [
            {
                "schema_version": EVOLUTION_TIME_SLICES_SCHEMA_VERSION,
                "evolution_id": evolution_id,
                "slice_id": "year:2020",
                "slice_index": 0,
                "slice_label": "2020",
                "start_year": 2020,
                "end_year": 2020,
                "unit": "year",
                "doc_count": 18,
                "edge_count": None,
                "active_cluster_count": 6,
                "unknown_year_count": 0,
                "warning_flags": "",
            },
            {
                "schema_version": EVOLUTION_TIME_SLICES_SCHEMA_VERSION,
                "evolution_id": evolution_id,
                "slice_id": "year:2021",
                "slice_index": 1,
                "slice_label": "2021",
                "start_year": 2021,
                "end_year": 2021,
                "unit": "year",
                "doc_count": 18,
                "edge_count": None,
                "active_cluster_count": 7,
                "unknown_year_count": 0,
                "warning_flags": "",
            },
            {
                "schema_version": EVOLUTION_TIME_SLICES_SCHEMA_VERSION,
                "evolution_id": evolution_id,
                "slice_id": "year:2022",
                "slice_index": 2,
                "slice_label": "2022",
                "start_year": 2022,
                "end_year": 2022,
                "unit": "year",
                "doc_count": 6,
                "edge_count": None,
                "active_cluster_count": 2,
                "unknown_year_count": 0,
                "warning_flags": "",
            },
        ]
    )
    state_specs = [
        ("A20", "year:2020", 0, "synthetic:A"),
        ("A21", "year:2021", 1, "synthetic:A"),
        ("A22", "year:2022", 2, "synthetic:A"),
        ("B20", "year:2020", 0, "synthetic:B"),
        ("B21a", "year:2021", 1, "synthetic:B_a"),
        ("B21b", "year:2021", 1, "synthetic:B_b"),
        ("C20a", "year:2020", 0, "synthetic:C_a"),
        ("C20b", "year:2020", 0, "synthetic:C_b"),
        ("C21", "year:2021", 1, "synthetic:C"),
        ("D21", "year:2021", 1, "synthetic:D"),
        ("D22", "year:2022", 2, "synthetic:D"),
        ("E20", "year:2020", 0, "synthetic:E"),
        ("X20", "year:2020", 0, "synthetic:X"),
        ("Y21", "year:2021", 1, "synthetic:Y"),
        ("Z21", "year:2021", 1, "synthetic:Z"),
    ]
    states = pd.DataFrame([_synthetic_evolution_state(evolution_id, *spec) for spec in state_specs])
    transitions = pd.DataFrame(
        [
            _synthetic_evolution_transition(evolution_id, "t_A20_A21", "A20", "A21", "year:2020", "year:2021", relation="continuation", score=0.95),
            _synthetic_evolution_transition(evolution_id, "t_A21_A22", "A21", "A22", "year:2021", "year:2022", relation="continuation", score=0.96),
            _synthetic_evolution_transition(evolution_id, "t_B20_B21a", "B20", "B21a", "year:2020", "year:2021", relation="split_child", score=0.76),
            _synthetic_evolution_transition(evolution_id, "t_B20_B21b", "B20", "B21b", "year:2020", "year:2021", relation="split_child", score=0.74),
            _synthetic_evolution_transition(evolution_id, "t_C20a_C21", "C20a", "C21", "year:2020", "year:2021", relation="merge_parent", score=0.81),
            _synthetic_evolution_transition(evolution_id, "t_C20b_C21", "C20b", "C21", "year:2020", "year:2021", relation="merge_parent", score=0.79),
            _synthetic_evolution_transition(evolution_id, "t_D21_D22", "D21", "D22", "year:2021", "year:2022", relation="continuation", score=0.88),
            _synthetic_evolution_transition(evolution_id, "t_X20_Y21", "X20", "Y21", "year:2020", "year:2021", relation="ambiguous", score=0.61),
            _synthetic_evolution_transition(evolution_id, "t_X20_Z21", "X20", "Z21", "year:2020", "year:2021", relation="ambiguous", score=0.60),
        ]
    )
    lineage_map = {
        "A20": "lineage_A",
        "A21": "lineage_A",
        "A22": "lineage_A",
        "B20": "lineage_B",
        "B21a": "lineage_B_a",
        "B21b": "lineage_B_b",
        "C20a": "lineage_C_a",
        "C20b": "lineage_C_b",
        "C21": "lineage_C",
        "D21": "lineage_D",
        "D22": "lineage_D",
        "E20": "lineage_E",
        "X20": "lineage_X",
        "Y21": "lineage_Y",
        "Z21": "lineage_Z",
    }
    lineages = pd.DataFrame(
        [
            {
                "schema_version": EVOLUTION_LINEAGES_SCHEMA_VERSION,
                "evolution_id": evolution_id,
                "lineage_id": lineage_map[state.state_id],
                "state_id": state.state_id,
                "slice_id": state.slice_id,
                "slice_index": int(state.slice_index),
                "role": "continuation" if state.state_id in {"A21", "D22"} else "terminal" if state.state_id in {"A22"} else "root",
                "stability_score": 1.0,
                "root_state_id": state.state_id,
                "lineage_label": state.cluster_label,
                "event_refs": "[]",
                "warning_flags": "",
            }
            for state in states.itertuples(index=False)
        ]
    )
    events = pd.DataFrame(
        [
            _synthetic_evolution_event(evolution_id, "e_cont_A20_A21", "continuation", "year:2021", "A21", "lineage_A", ["t_A20_A21"], source_state_ids=["A20"], target_state_ids=["A21"], score=0.95),
            _synthetic_evolution_event(evolution_id, "e_cont_A21_A22", "continuation", "year:2022", "A22", "lineage_A", ["t_A21_A22"], source_state_ids=["A21"], target_state_ids=["A22"], score=0.96),
            _synthetic_evolution_event(evolution_id, "e_cont_D21_D22", "continuation", "year:2022", "D22", "lineage_D", ["t_D21_D22"], source_state_ids=["D21"], target_state_ids=["D22"], score=0.88),
            _synthetic_evolution_event(evolution_id, "e_split_B20", "split", "year:2020", "B20", "lineage_B", ["t_B20_B21a", "t_B20_B21b"], source_state_ids=["B20"], target_state_ids=["B21a", "B21b"], score=0.76, support_count=6),
            _synthetic_evolution_event(evolution_id, "e_merge_C21", "merge", "year:2021", "C21", "lineage_C", ["t_C20a_C21", "t_C20b_C21"], source_state_ids=["C20a", "C20b"], target_state_ids=["C21"], score=0.81, support_count=6),
            _synthetic_evolution_event(evolution_id, "e_emergence_D21", "emergence", "year:2021", "D21", "lineage_D", [], target_state_ids=["D21"], score=1.0),
            _synthetic_evolution_event(evolution_id, "e_decline_E20", "decline", "year:2020", "E20", "lineage_E", [], source_state_ids=["E20"], score=1.0),
            _synthetic_evolution_event(evolution_id, "e_ambiguous_X20", "ambiguous", "year:2020", "X20", "lineage_X", ["t_X20_Y21", "t_X20_Z21"], source_state_ids=["X20"], target_state_ids=["Y21", "Z21"], score=0.61, support_count=6),
        ]
    )
    smoke = {
        "schema_version": EVOLUTION_SYNTHETIC_SMOKE_SCHEMA_VERSION,
        "evolution_id": evolution_id,
        "time_slices": [
            {"slice_id": "year:2020", "slice_index": 0, "start_year": 2020, "end_year": 2020},
            {"slice_id": "year:2021", "slice_index": 1, "start_year": 2021, "end_year": 2021},
            {"slice_id": "year:2022", "slice_index": 2, "start_year": 2022, "end_year": 2022},
        ],
        "expected_event_counts": {
            "continuation": 3,
            "split": 1,
            "merge": 1,
            "emergence": 1,
            "decline": 1,
            "ambiguous": 1,
        },
        "expected_blocking_issues": [],
    }
    evolution_dir.mkdir(parents=True, exist_ok=True)
    (evolution_dir / outputs["synthetic_smoke"]).write_text(json.dumps(smoke, indent=2, sort_keys=True), encoding="utf-8")
    manifest = {
        "schema_version": EVOLUTION_MANIFEST_SCHEMA_VERSION,
        "evolution_id": evolution_id,
        "title": "Synthetic All-Events Evolution Smoke",
        "result_id": _matrix_existing_result_id(root),
        "slice_method": {
            "unit": "year",
            "window_years": 1,
            "step_years": 1,
            "start_year": 2020,
            "end_year": 2022,
            "state_method": "synthetic_fixture",
            "include_unknown_year": False,
        },
        "matching_method": {
            "metric": "synthetic_overlap",
            "min_transition_score": 0.5,
            "min_support_count": 1,
            "tie_policy": "fixture_declared",
            "normalization": "fixture",
        },
        "event_rules": {
            "continuation_min_score": 0.5,
            "split_min_children": 2,
            "merge_min_parents": 2,
            "emergence_max_incoming_score": 0.0,
            "decline_max_outgoing_score": 0.0,
            "ambiguous_score_margin": 0.05,
        },
        "entity_scope": {
            "cluster_level": "synthetic",
            "cluster_id_namespace": "fixture_state_ids",
            "document_universe": "synthetic_fixture",
            "filter_refs": [],
        },
        "metrics": [
            {"name": "synthetic_overlap", "value_type": "float", "range": [0.0, 1.0], "interpretation": "fixture transition confidence"},
            {"name": "lineage_stability", "value_type": "float", "range": [0.0, 1.0], "interpretation": "fixture lineage stability"},
        ],
        "source_artifacts": [{"role": "synthetic_fixture", "path": _rel(evolution_dir / outputs["synthetic_smoke"], root)}],
        "rule_sets": [],
        "transforms": [{"step": "write_synthetic_all_event_fixture"}],
        "outputs": outputs,
        "created_at_utc": _utc_now(),
        "warnings": [],
    }
    return _write_evolution_payload(
        root,
        evolution_dir,
        manifest=manifest,
        slices=slices,
        states=states,
        transitions=transitions,
        lineages=lineages,
        events=events,
    )


def _non_empty_payload(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _report_has_evolution(report_data: dict[str, Any] | None, clusters: list[dict[str, Any]]) -> bool:
    if not report_data:
        return False
    for key in ("_evolution", "evolution", "cluster_evolution"):
        if _non_empty_payload(report_data.get(key)):
            return True
    for cluster in clusters:
        if _non_empty_payload(cluster.get("evolution")):
            return True
        if _non_empty_payload(cluster.get("yearly_activity")):
            return True
        if _non_empty_payload(cluster.get("topic_trajectory")):
            return True
        if _non_empty_payload(cluster.get("child_trajectory")):
            return True
    return False


def _report_has_narrative(report_data: dict[str, Any] | None, clusters: list[dict[str, Any]]) -> bool:
    if not report_data:
        return False
    for key in ("_narratives", "narratives", "cluster_narratives"):
        if _non_empty_payload(report_data.get(key)):
            return True
    for cluster in clusters:
        if _non_empty_payload(cluster.get("narrative")):
            return True
        if _non_empty_payload(cluster.get("overview_markdown")):
            return True
        overview = cluster.get("overview")
        if isinstance(overview, Mapping) and _non_empty_payload(overview.get("overview_markdown")):
            return True
    return False


def _keywords_term_edge_count(path: Path, issues: list[ArtifactIssue]) -> int:
    try:
        cols = pd.read_parquet(path, columns=["cluster_id", "term"])
    except Exception as exc:
        issues.append(
            ArtifactIssue(
                "keyword_network_scan_failed",
                "warning",
                f"Could not scan keyword table for term-network capability: {exc}",
                "keywords",
            )
        )
        return 0
    if cols.empty:
        return 0
    total = 0
    for size in cols.groupby("cluster_id", dropna=False).size():
        if int(size) >= 2:
            total += int(size) * (int(size) - 1) // 2
    return total


def _flag_set(value: object) -> set[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return set()
    if isinstance(value, (list, tuple, set)):
        return {str(item).strip().lower() for item in value if str(item).strip()}
    text = str(value).strip().lower()
    if not text:
        return set()
    return {part.strip() for part in text.replace(",", "|").split("|") if part.strip()}


def _looks_like_metadata_artifact_term_lazy(term: object) -> bool:
    # Lazy import avoids a circular import with dashboard export helpers.
    from .keyword_extraction.utils import _looks_like_metadata_artifact_term

    return _looks_like_metadata_artifact_term(term)


def _row_term_values(row: pd.Series) -> list[str]:
    values: list[str] = []
    for column in ("display_label", "term", "raw_term"):
        if column not in row.index:
            continue
        value = row.get(column)
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        text = str(value).strip()
        if text and text not in values:
            values.append(text)
    return values


def _row_has_blocking_artifact(row: pd.Series) -> bool:
    flags = _flag_set(row.get("quality_flags") if "quality_flags" in row.index else None)
    if flags & BLOCKING_ARTIFACT_FLAGS:
        return True
    return any(_looks_like_metadata_artifact_term_lazy(term) for term in _row_term_values(row))


def _row_has_review_artifact(row: pd.Series) -> bool:
    flags = _flag_set(row.get("quality_flags") if "quality_flags" in row.index else None)
    tier = str(row.get("keyword_label_tier", "")).strip().lower() if "keyword_label_tier" in row.index else ""
    return bool(flags & REVIEW_ARTIFACT_FLAGS) or tier == "review_artifact"


def _sort_keywords_for_scan(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    if "representative_rank" in working.columns:
        working["_scan_rank"] = pd.to_numeric(working["representative_rank"], errors="coerce")
        working["_scan_rank"] = working["_scan_rank"].fillna(float("inf"))
        return working.sort_values(["cluster_id", "_scan_rank"], kind="stable")
    if "quality_score" in working.columns:
        working["_scan_score"] = pd.to_numeric(working["quality_score"], errors="coerce").fillna(float("-inf"))
        return working.sort_values(["cluster_id", "_scan_score"], ascending=[True, False], kind="stable")
    if "score" in working.columns:
        working["_scan_score"] = pd.to_numeric(working["score"], errors="coerce").fillna(float("-inf"))
        return working.sort_values(["cluster_id", "_scan_score"], ascending=[True, False], kind="stable")
    return working


def _sample_artifact_rows(rows: list[dict[str, Any]], *, limit: int = 5) -> str:
    samples = []
    for row in rows[:limit]:
        cluster = row.get("cluster_id")
        term = row.get("term")
        rank = row.get("rank")
        samples.append(f"C{cluster}:{rank}:{term}")
    return ", ".join(samples)


def _scan_keyword_artifacts(
    path: Path,
    info: ArtifactTableInfo,
    issues: list[ArtifactIssue],
) -> tuple[int, int, int]:
    columns = set(info.columns or [])
    if not {"cluster_id", "term"} <= columns:
        return 0, 0, 0
    scan_columns = [
        column
        for column in (
            "cluster_id",
            "term",
            "raw_term",
            "display_label",
            "quality_flags",
            "keyword_label_tier",
            "representative_rank",
            "quality_score",
            "score",
        )
        if column in columns
    ]
    try:
        df = pd.read_parquet(path, columns=scan_columns)
    except Exception as exc:
        issues.append(
            ArtifactIssue(
                "keyword_artifact_scan_failed",
                "warning",
                f"Could not scan keyword artifacts: {exc}",
                "keywords",
            )
        )
        return 0, 0, 0
    if df.empty:
        return 0, 0, 0

    df = _sort_keywords_for_scan(df)
    blocking_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    top_blocking_rows: list[dict[str, Any]] = []

    for cluster_id, group in df.groupby("cluster_id", dropna=False, sort=False):
        for rank, (_, row) in enumerate(group.iterrows(), start=1):
            term_values = _row_term_values(row)
            term = term_values[0] if term_values else ""
            record = {"cluster_id": cluster_id, "rank": rank, "term": term}
            if _row_has_blocking_artifact(row):
                blocking_rows.append(record)
                if rank <= KEYWORD_ARTIFACT_TOP_K:
                    top_blocking_rows.append(record)
            elif _row_has_review_artifact(row):
                review_rows.append(record)

    if blocking_rows:
        issues.append(
            ArtifactIssue(
                "keyword_artifact_rows",
                "warning",
                (
                    f"{len(blocking_rows)} metadata/HTML/LaTeX artifact keyword rows detected"
                    f" ({_sample_artifact_rows(blocking_rows)})."
                ),
                "keywords",
            )
        )
    if review_rows:
        issues.append(
            ArtifactIssue(
                "keyword_review_artifact_rows",
                "info",
                f"{len(review_rows)} review-artifact keyword rows are present outside hard metadata rules.",
                "keywords",
            )
        )
    if top_blocking_rows:
        issues.append(
            ArtifactIssue(
                "top_keyword_artifact",
                "error",
                (
                    f"{len(top_blocking_rows)} metadata/HTML/LaTeX artifacts appear in top "
                    f"{KEYWORD_ARTIFACT_TOP_K} keyword rows ({_sample_artifact_rows(top_blocking_rows)})."
                ),
                "keywords",
            )
        )
    return len(blocking_rows), len(top_blocking_rows), len(review_rows)


def _keyword_dict_term_values(keyword: Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("display_label", "term", "raw_term"):
        value = keyword.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text and text not in values:
            values.append(text)
    return values


def _keyword_dict_has_blocking_artifact(keyword: Mapping[str, Any]) -> bool:
    if _flag_set(keyword.get("quality_flags")) & BLOCKING_ARTIFACT_FLAGS:
        return True
    return any(_looks_like_metadata_artifact_term_lazy(term) for term in _keyword_dict_term_values(keyword))


def _keyword_dict_has_review_artifact(keyword: Mapping[str, Any]) -> bool:
    flags = _flag_set(keyword.get("quality_flags"))
    tier = str(keyword.get("keyword_label_tier", "")).strip().lower()
    return bool(flags & REVIEW_ARTIFACT_FLAGS) or tier == "review_artifact"


def _scan_report_keyword_artifacts(
    clusters: list[dict[str, Any]],
    issues: list[ArtifactIssue],
) -> tuple[int, int, int]:
    blocking_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    top_blocking_rows: list[dict[str, Any]] = []

    for cluster_idx, cluster in enumerate(clusters):
        cluster_id = cluster.get("cluster_id", cluster.get("id", cluster_idx))
        keywords = cluster.get("keywords")
        if not isinstance(keywords, list):
            continue
        for rank, keyword in enumerate(keywords, start=1):
            if not isinstance(keyword, dict):
                continue
            term_values = _keyword_dict_term_values(keyword)
            term = term_values[0] if term_values else ""
            record = {"cluster_id": cluster_id, "rank": rank, "term": term}
            if _keyword_dict_has_blocking_artifact(keyword):
                blocking_rows.append(record)
                if rank <= KEYWORD_ARTIFACT_TOP_K:
                    top_blocking_rows.append(record)
            elif _keyword_dict_has_review_artifact(keyword):
                review_rows.append(record)

    if blocking_rows:
        issues.append(
            ArtifactIssue(
                "report_keyword_artifact_rows",
                "warning",
                (
                    f"{len(blocking_rows)} metadata/HTML/LaTeX artifact report terms detected"
                    f" ({_sample_artifact_rows(blocking_rows)})."
                ),
                "report_data",
            )
        )
    if review_rows:
        issues.append(
            ArtifactIssue(
                "report_keyword_review_artifact_rows",
                "info",
                f"{len(review_rows)} review-artifact report terms are present outside hard metadata rules.",
                "report_data",
            )
        )
    if top_blocking_rows:
        issues.append(
            ArtifactIssue(
                "top_report_keyword_artifact",
                "error",
                (
                    f"{len(top_blocking_rows)} metadata/HTML/LaTeX artifacts appear in top "
                    f"{KEYWORD_ARTIFACT_TOP_K} report keyword rows ({_sample_artifact_rows(top_blocking_rows)})."
                ),
                "report_data",
            )
        )
    return len(blocking_rows), len(top_blocking_rows), len(review_rows)


def _read_column_set(path: Path, column: str) -> set[Any]:
    values = pd.read_parquet(path, columns=[column])[column]
    return set(values.dropna().map(str).tolist())


def _reconcile_counts(
    *,
    artifacts: ResultArtifacts,
    membership_info: ArtifactTableInfo | None,
    keyword_info: ArtifactTableInfo | None,
    issues: list[ArtifactIssue],
) -> None:
    if artifacts.abstracts_path and artifacts.membership_path:
        try:
            abstract_uids = _read_column_set(artifacts.abstracts_path, "uid")
            membership_uids = _read_column_set(artifacts.membership_path, "uid")
        except Exception as exc:
            issues.append(ArtifactIssue("uid_reconciliation_failed", "warning", str(exc), "membership"))
        else:
            missing = membership_uids - abstract_uids
            if missing:
                issues.append(
                    ArtifactIssue(
                        "membership_uid_missing_from_abstracts",
                        "error",
                        f"{len(missing)} membership uids are absent from abstracts.",
                        "membership",
                    )
                )

    if artifacts.membership_path and artifacts.keywords_path and membership_info and keyword_info:
        cluster_cols = _cluster_columns(membership_info.columns)
        if not cluster_cols:
            return
        try:
            membership_clusters: set[Any] = set()
            mem = pd.read_parquet(artifacts.membership_path, columns=cluster_cols)
            for column in cluster_cols:
                membership_clusters.update(mem[column].dropna().map(str).tolist())
            keyword_clusters = _read_column_set(artifacts.keywords_path, "cluster_id")
        except Exception as exc:
            issues.append(ArtifactIssue("cluster_reconciliation_failed", "warning", str(exc), "keywords"))
            return
        missing_clusters = keyword_clusters - membership_clusters
        if missing_clusters:
            issues.append(
                ArtifactIssue(
                    "keyword_cluster_missing_from_membership",
                    "error",
                    f"{len(missing_clusters)} keyword cluster IDs are absent from membership.",
                    "keywords",
                )
            )


def _severity_dicts(issues: list[ArtifactIssue]) -> list[dict[str, Any]]:
    return [asdict(issue) for issue in issues]


def _versions() -> dict[str, Any]:
    return {
        "sciscape_version": SCISCAPE_VERSION,
        "artifact_contract_schema_version": ARTIFACT_CONTRACT_SCHEMA_VERSION,
    }


def validate_result_root(path: str | Path, *, mode: str = "local_result") -> ArtifactValidationResult:
    """Validate a SciScape result root, report directory, or ``data.json`` file."""

    artifacts = infer_result_artifacts(path)
    root = artifacts.result_root
    issues: list[ArtifactIssue] = []
    artifact_info: dict[str, Any] = {
        "input_path": str(artifacts.input_path),
        "result_root": str(root),
        "landscape_dir": _rel(artifacts.landscape_dir, root),
        "report_data": _rel(artifacts.report_data_path, root),
        "abstracts": _rel(artifacts.abstracts_path, root),
        "edges": _rel(artifacts.edges_path, root),
        "membership": _rel(artifacts.membership_path, root),
        "keywords": _rel(artifacts.keywords_path, root),
        "matrix_artifacts": [_rel(path, root) for path in artifacts.matrix_paths],
        "matrix_manifest_artifacts": [_rel(path, root) for path in artifacts.matrix_manifest_paths],
        "matrix_summaries": [],
        "keyword_rule_manifest_artifacts": [_rel(path, root) for path in artifacts.keyword_rule_manifest_paths],
        "keyword_rule_summaries": [],
        "temporal_manifest_artifacts": [_rel(path, root) for path in artifacts.temporal_manifest_paths],
        "temporal_summaries": [],
        "evolution_manifest_artifacts": [_rel(path, root) for path in artifacts.evolution_manifest_paths],
        "evolution_summaries": [],
        "export_manifest_artifacts": [_rel(path, root) for path in artifacts.export_manifest_paths],
        "export_summaries": [],
        "edge_evidence_artifacts": [_rel(path, root) for path in artifacts.edge_evidence_paths],
        "evolution_artifacts": [_rel(path, root) for path in artifacts.evolution_paths],
        "narrative_manifest_artifacts": [_rel(path, root) for path in artifacts.narrative_manifest_paths],
        "narrative_summaries": [],
        "narrative_artifacts": [_rel(path, root) for path in artifacts.narrative_paths],
        "review_packet_artifacts": [_rel(path, root) for path in artifacts.review_packet_paths],
        "review_packet_summaries": [],
        "qa": _rel(artifacts.qa_path, root),
        "tables": {},
    }

    report_data = _json_payload(artifacts.report_data_path, issues, "report_data") if artifacts.report_data_path else None
    report_clusters = _report_clusters(report_data)
    report_edge_count = _report_term_edge_count(report_clusters)
    report_has_evolution = _report_has_evolution(report_data, report_clusters)
    report_has_narrative = _report_has_narrative(report_data, report_clusters)
    report_artifact_rows, report_top_artifact_rows, report_review_artifact_rows = _scan_report_keyword_artifacts(
        report_clusters,
        issues,
    )

    abstract_info = None
    if artifacts.abstracts_path:
        abstract_info = _parquet_info(artifacts.abstracts_path, root, issues, "abstracts")
        artifact_info["tables"]["abstracts"] = asdict(abstract_info)
        _missing_columns(info=abstract_info, required=REQUIRED_ABSTRACT_COLUMNS, issues=issues, role="abstracts")

    edge_info = None
    if artifacts.edges_path:
        edge_info = _parquet_info(artifacts.edges_path, root, issues, "edges")
        artifact_info["tables"]["edges"] = asdict(edge_info)
        _missing_columns(info=edge_info, required=REQUIRED_EDGE_COLUMNS, issues=issues, role="edges")
        if not _has_numeric_weight(artifacts.edges_path, edge_info.columns):
            issues.append(
                ArtifactIssue(
                    "missing_edge_weight",
                    "warning",
                    "No recognized numeric edge weight column was found.",
                    "edges",
                )
            )

    membership_info = None
    if artifacts.membership_path:
        membership_info = _parquet_info(artifacts.membership_path, root, issues, "membership")
        artifact_info["tables"]["membership"] = asdict(membership_info)
        _missing_columns(info=membership_info, required=REQUIRED_MEMBERSHIP_COLUMNS, issues=issues, role="membership")
        if not _cluster_columns(membership_info.columns):
            issues.append(
                ArtifactIssue(
                    "missing_cluster_column",
                    "error",
                    "Membership must include `cluster` or at least one `cluster_*` column.",
                    "membership",
                )
            )

    keyword_info = None
    keyword_edge_count = 0
    keyword_artifact_rows = 0
    keyword_top_artifact_rows = 0
    keyword_review_artifact_rows = 0
    if artifacts.keywords_path:
        keyword_info = _parquet_info(artifacts.keywords_path, root, issues, "keywords")
        artifact_info["tables"]["keywords"] = asdict(keyword_info)
        if not _missing_columns(info=keyword_info, required=REQUIRED_KEYWORD_COLUMNS, issues=issues, role="keywords"):
            keyword_edge_count = _keywords_term_edge_count(artifacts.keywords_path, issues)
            (
                keyword_artifact_rows,
                keyword_top_artifact_rows,
                keyword_review_artifact_rows,
            ) = _scan_keyword_artifacts(artifacts.keywords_path, keyword_info, issues)

    cooccurrence_artifacts = 0
    cooccurrence_rows = 0
    cooccurrence_json_rows = 0
    for matrix_path in artifacts.matrix_paths:
        rel_path = _rel(matrix_path, root) or str(matrix_path)
        role = "cooccurrence" if "cooccurrence" in str(matrix_path).lower() else "matrix"
        table_key = f"{role}_artifact:{rel_path}"
        if role == "cooccurrence":
            cooccurrence_artifacts += 1
        if matrix_path.suffix.lower() == ".parquet":
            matrix_info = _parquet_info(matrix_path, root, issues, role)
            artifact_info["tables"][table_key] = asdict(matrix_info)
            if role == "cooccurrence":
                if not _missing_columns(
                    info=matrix_info,
                    required=REQUIRED_COOCCURRENCE_COLUMNS,
                    issues=issues,
                    role=role,
                ):
                    cooccurrence_rows += int(matrix_info.rows or 0)
                    if not matrix_info.rows:
                        issues.append(
                            ArtifactIssue(
                                "empty_cooccurrence_table",
                                "warning",
                                "Co-occurrence artifact table has no rows.",
                                role,
                            )
                        )
        elif role == "cooccurrence" and matrix_path.suffix.lower() == ".json":
            payload = _json_payload(matrix_path, issues, role)
            if payload is None:
                continue
            if payload.get("schema_version") != COOCCURRENCE_ARTIFACT_SCHEMA_VERSION:
                issues.append(
                    ArtifactIssue(
                        "unsupported_cooccurrence_schema",
                        "warning",
                        f"Unsupported co-occurrence artifact schema: {payload.get('schema_version')}",
                        role,
                    )
                )
            else:
                cooccurrence_json_rows = max(
                    cooccurrence_json_rows,
                    int(_coerce_int(payload.get("edge_count")) or 0),
                )
    cooccurrence_rows = max(cooccurrence_rows, cooccurrence_json_rows)

    general_matrix_artifacts = 0
    stable_matrix_artifacts = 0
    matrix_nnz = 0
    for manifest_path in artifacts.matrix_manifest_paths:
        matrix_validation = validate_matrix_artifact(manifest_path)
        matrix_payload = matrix_validation.to_dict()
        artifact_info["matrix_summaries"].append(
            {
                "matrix_id": matrix_payload.get("matrix_id"),
                "matrix_family": matrix_payload.get("matrix_family"),
                "status": matrix_payload.get("status"),
                "path": _rel(manifest_path, root),
                "counts": matrix_payload.get("counts", {}),
            }
        )
        general_matrix_artifacts += 1
        matrix_nnz += int(matrix_validation.counts.get("nnz", 0))
        if matrix_validation.status == "passed":
            stable_matrix_artifacts += 1
        for issue in matrix_validation.blocking_issues:
            issues.append(
                ArtifactIssue(
                    str(issue.get("code") or "matrix_artifact_blocked"),
                    "error",
                    str(issue.get("message") or "Matrix artifact validation failed."),
                    "matrix",
                )
            )
        for warning in matrix_validation.warnings:
            issues.append(
                ArtifactIssue(
                    str(warning.get("code") or "matrix_artifact_warning"),
                    "warning",
                    str(warning.get("message") or "Matrix artifact has validation warnings."),
                    "matrix",
                )
            )

    keyword_rule_artifacts = 0
    stable_keyword_rule_artifacts = 0
    keyword_rule_application_rows = 0
    keyword_rule_before_after_rows = 0
    keyword_rule_blocked_rows = 0
    keyword_rule_artifact_rows_after = 0
    keyword_rule_top_artifact_rows_after = 0
    for manifest_path in artifacts.keyword_rule_manifest_paths:
        keyword_rule_validation = validate_keyword_rule_artifact(manifest_path)
        keyword_rule_payload = keyword_rule_validation.to_dict()
        artifact_info["keyword_rule_summaries"].append(
            {
                "rule_set_id": keyword_rule_payload.get("rule_set_id"),
                "status": keyword_rule_payload.get("status"),
                "path": _rel(manifest_path, root),
                "counts": keyword_rule_payload.get("counts", {}),
                "contamination_counts": keyword_rule_payload.get("contamination_counts", {}),
            }
        )
        keyword_rule_artifacts += 1
        keyword_rule_application_rows += int(keyword_rule_validation.counts.get("applications", 0))
        keyword_rule_before_after_rows += int(keyword_rule_validation.counts.get("before_after_rows", 0))
        keyword_rule_blocked_rows += int(keyword_rule_validation.counts.get("blocked_rows", 0))
        keyword_rule_artifact_rows_after += int(
            keyword_rule_validation.contamination_counts.get("artifact_rows_after", 0)
        )
        keyword_rule_top_artifact_rows_after += int(
            keyword_rule_validation.contamination_counts.get("top_artifact_rows_after", 0)
        )
        if keyword_rule_validation.status == "passed":
            stable_keyword_rule_artifacts += 1
        for issue in keyword_rule_validation.blocking_issues:
            issues.append(
                ArtifactIssue(
                    str(issue.get("code") or "keyword_rule_artifact_blocked"),
                    "error",
                    str(issue.get("message") or "Keyword rule artifact validation failed."),
                    "keyword_rules",
                )
            )
        for warning in keyword_rule_validation.warnings:
            issues.append(
                ArtifactIssue(
                    str(warning.get("code") or "keyword_rule_artifact_warning"),
                    "warning",
                    str(warning.get("message") or "Keyword rule artifact has validation warnings."),
                    "keyword_rules",
                )
            )

    temporal_artifacts = 0
    stable_temporal_artifacts = 0
    temporal_periods = 0
    temporal_series_rows = 0
    temporal_event_rows = 0
    temporal_missing_years = 0
    for manifest_path in artifacts.temporal_manifest_paths:
        temporal_validation = validate_temporal_artifact(manifest_path)
        temporal_payload = temporal_validation.to_dict()
        artifact_info["temporal_summaries"].append(
            {
                "temporal_id": temporal_payload.get("temporal_id"),
                "status": temporal_payload.get("status"),
                "path": _rel(manifest_path, root),
                "counts": temporal_payload.get("counts", {}),
                "event_counts": temporal_payload.get("event_counts", {}),
            }
        )
        temporal_artifacts += 1
        temporal_periods += int(temporal_validation.counts.get("periods", 0))
        temporal_series_rows += int(temporal_validation.counts.get("series_rows", 0))
        temporal_event_rows += int(temporal_validation.counts.get("event_rows", 0))
        temporal_missing_years = max(temporal_missing_years, int(temporal_validation.counts.get("missing_years", 0)))
        if temporal_validation.status == "passed":
            stable_temporal_artifacts += 1
        for issue in temporal_validation.blocking_issues:
            issues.append(
                ArtifactIssue(
                    str(issue.get("code") or "temporal_artifact_blocked"),
                    "error",
                    str(issue.get("message") or "Temporal artifact validation failed."),
                    "temporal",
                )
            )
        for warning in temporal_validation.warnings:
            issues.append(
                ArtifactIssue(
                    str(warning.get("code") or "temporal_artifact_warning"),
                    "warning",
                    str(warning.get("message") or "Temporal artifact has validation warnings."),
                    "temporal",
                )
            )

    evolution_artifacts = 0
    stable_evolution_artifacts = 0
    evolution_slices = 0
    evolution_states = 0
    evolution_transitions = 0
    evolution_lineages = 0
    evolution_event_rows = 0
    for manifest_path in artifacts.evolution_manifest_paths:
        evolution_validation = validate_evolution_artifact(manifest_path)
        evolution_payload = evolution_validation.to_dict()
        artifact_info["evolution_summaries"].append(
            {
                "evolution_id": evolution_payload.get("evolution_id"),
                "status": evolution_payload.get("status"),
                "path": _rel(manifest_path, root),
                "counts": evolution_payload.get("counts", {}),
                "event_counts": evolution_payload.get("event_counts", {}),
            }
        )
        evolution_artifacts += 1
        evolution_slices += int(evolution_validation.counts.get("slices", 0))
        evolution_states += int(evolution_validation.counts.get("states", 0))
        evolution_transitions += int(evolution_validation.counts.get("transitions", 0))
        evolution_lineages += int(evolution_validation.counts.get("lineages", 0))
        evolution_event_rows += int(evolution_validation.counts.get("event_rows", 0))
        if evolution_validation.status == "passed":
            stable_evolution_artifacts += 1
        for issue in evolution_validation.blocking_issues:
            issues.append(
                ArtifactIssue(
                    str(issue.get("code") or "evolution_artifact_blocked"),
                    "error",
                    str(issue.get("message") or "Evolution artifact validation failed."),
                    "evolution",
                )
            )
        for warning in evolution_validation.warnings:
            issues.append(
                ArtifactIssue(
                    str(warning.get("code") or "evolution_artifact_warning"),
                    "warning",
                    str(warning.get("message") or "Evolution artifact has validation warnings."),
                    "evolution",
                )
            )

    export_artifacts = 0
    stable_export_artifacts = 0
    export_file_rows = 0
    for manifest_path in artifacts.export_manifest_paths:
        export_validation = validate_export_manifest(manifest_path)
        export_payload = export_validation.to_dict()
        artifact_info["export_summaries"].append(
            {
                "export_id": export_payload.get("export_id"),
                "export_family": export_payload.get("export_family"),
                "export_kind": export_payload.get("export_kind"),
                "status": export_payload.get("status"),
                "path": _rel(manifest_path, root),
                "counts": export_payload.get("counts", {}),
            }
        )
        export_artifacts += 1
        export_file_rows += int(export_validation.counts.get("files", 0))
        if export_validation.status == "passed":
            stable_export_artifacts += 1
        for issue in export_validation.blocking_issues:
            issues.append(
                ArtifactIssue(
                    str(issue.get("code") or "export_artifact_blocked"),
                    "error",
                    str(issue.get("message") or "Export artifact validation failed."),
                    "export",
                )
            )
        for warning in export_validation.warnings:
            issues.append(
                ArtifactIssue(
                    str(warning.get("code") or "export_artifact_warning"),
                    "warning",
                    str(warning.get("message") or "Export artifact has validation warnings."),
                    "export",
                )
            )

    narrative_artifacts = 0
    stable_narrative_artifacts = 0
    narrative_targets = 0
    narrative_claims = 0
    narrative_evidence_refs = 0
    narrative_aggregate_only_refs = 0
    for manifest_path in artifacts.narrative_manifest_paths:
        narrative_validation = validate_narrative_artifact(manifest_path)
        narrative_payload = narrative_validation.to_dict()
        artifact_info["narrative_summaries"].append(
            {
                "narrative_id": narrative_payload.get("narrative_id"),
                "status": narrative_payload.get("status"),
                "feature_state": narrative_payload.get("feature_state"),
                "path": _rel(manifest_path, root),
                "counts": narrative_payload.get("counts", {}),
                "claim_counts": narrative_payload.get("claim_counts", {}),
            }
        )
        narrative_artifacts += 1
        narrative_targets += int(narrative_validation.counts.get("targets", 0))
        narrative_claims += int(narrative_validation.counts.get("claims", 0))
        narrative_evidence_refs += int(narrative_validation.counts.get("evidence_refs", 0))
        narrative_aggregate_only_refs += int(narrative_validation.counts.get("aggregate_only_refs", 0))
        if narrative_validation.status == "passed":
            stable_narrative_artifacts += 1
        elif narrative_validation.status == "warning":
            issues.append(
                ArtifactIssue(
                    "narrative_artifact_warning_state",
                    "warning",
                    "Narrative artifact is available but should be treated as beta.",
                    "narrative",
                )
            )
        for issue in narrative_validation.blocking_issues:
            issues.append(
                ArtifactIssue(
                    str(issue.get("code") or "narrative_artifact_blocked"),
                    "error",
                    str(issue.get("message") or "Narrative artifact validation failed."),
                    "narrative",
                )
            )
        for warning in narrative_validation.warnings:
            issues.append(
                ArtifactIssue(
                    str(warning.get("code") or "narrative_artifact_warning"),
                    "warning",
                    str(warning.get("message") or "Narrative artifact has validation warnings."),
                    "narrative",
                )
            )

    review_packet_artifacts = 0
    stable_review_packet_artifacts = 0
    review_packet_clusters = 0
    review_packet_narrative_ready_clusters = 0
    review_packet_review_required_clusters = 0
    review_packet_evidence_refs = 0
    for packet_path in artifacts.review_packet_paths:
        packet_validation = validate_cluster_review_packet_artifact(packet_path)
        packet_payload = packet_validation.to_dict()
        artifact_info["review_packet_summaries"].append(
            {
                "packet_id": packet_payload.get("packet_id"),
                "status": packet_payload.get("status"),
                "path": _rel(packet_path, root),
                "counts": packet_payload.get("counts", {}),
            }
        )
        review_packet_artifacts += 1
        review_packet_clusters += int(packet_validation.counts.get("clusters", 0))
        review_packet_narrative_ready_clusters += int(packet_validation.counts.get("narrative_ready_clusters", 0))
        review_packet_review_required_clusters += int(packet_validation.counts.get("review_required_clusters", 0))
        review_packet_evidence_refs += int(packet_validation.counts.get("evidence_refs", 0))
        source_check = packet_validation.checks.get("source_artifacts", {})
        if packet_validation.status == "passed" and source_check.get("status") == "passed":
            stable_review_packet_artifacts += 1
        for issue in packet_validation.blocking_issues:
            issues.append(
                ArtifactIssue(
                    str(issue.get("code") or "review_packet_blocked"),
                    "error",
                    str(issue.get("message") or "Cluster review packet validation failed."),
                    "cluster_review_packet",
                )
            )
        for warning in packet_validation.warnings:
            issues.append(
                ArtifactIssue(
                    str(warning.get("code") or "review_packet_warning"),
                    "warning",
                    str(warning.get("message") or "Cluster review packet has validation warnings."),
                    "cluster_review_packet",
                )
            )

    _reconcile_counts(
        artifacts=artifacts,
        membership_info=membership_info,
        keyword_info=keyword_info,
        issues=issues,
    )

    if not any(
        [
            artifacts.report_data_path,
            artifacts.abstracts_path,
            artifacts.membership_path,
            artifacts.keywords_path,
            artifacts.edges_path,
            artifacts.matrix_manifest_paths,
            artifacts.keyword_rule_manifest_paths,
            artifacts.temporal_manifest_paths,
            artifacts.evolution_manifest_paths,
            artifacts.export_manifest_paths,
            artifacts.narrative_manifest_paths,
            artifacts.review_packet_paths,
        ]
    ):
        issues.append(
            ArtifactIssue(
                "no_supported_artifacts",
                "error",
                "No supported SciScape artifacts were found.",
                None,
            )
        )

    features = {key: False for key in FEATURE_KEYS}
    counts = {
        "abstract_rows": int(abstract_info.rows or 0) if abstract_info else 0,
        "edge_rows": int(edge_info.rows or 0) if edge_info else 0,
        "membership_rows": int(membership_info.rows or 0) if membership_info else 0,
        "keyword_rows": int(keyword_info.rows or 0) if keyword_info else 0,
        "report_clusters": len(report_clusters),
        "report_term_edges": int(report_edge_count),
        "derived_keyword_term_edges": int(keyword_edge_count),
        "matrix_artifacts": len(artifacts.matrix_paths),
        "general_matrix_artifacts": int(general_matrix_artifacts),
        "stable_matrix_artifacts": int(stable_matrix_artifacts),
        "matrix_nnz": int(matrix_nnz),
        "keyword_rule_artifacts": int(keyword_rule_artifacts),
        "stable_keyword_rule_artifacts": int(stable_keyword_rule_artifacts),
        "keyword_rule_application_rows": int(keyword_rule_application_rows),
        "keyword_rule_before_after_rows": int(keyword_rule_before_after_rows),
        "keyword_rule_blocked_rows": int(keyword_rule_blocked_rows),
        "keyword_rule_artifact_rows_after": int(keyword_rule_artifact_rows_after),
        "keyword_rule_top_artifact_rows_after": int(keyword_rule_top_artifact_rows_after),
        "cooccurrence_artifacts": int(cooccurrence_artifacts),
        "cooccurrence_rows": int(cooccurrence_rows),
        "edge_evidence_artifacts": len(artifacts.edge_evidence_paths),
        "review_packet_artifacts": int(review_packet_artifacts),
        "stable_review_packet_artifacts": int(stable_review_packet_artifacts),
        "review_packet_clusters": int(review_packet_clusters),
        "review_packet_narrative_ready_clusters": int(review_packet_narrative_ready_clusters),
        "review_packet_review_required_clusters": int(review_packet_review_required_clusters),
        "review_packet_evidence_refs": int(review_packet_evidence_refs),
        "temporal_artifacts": int(temporal_artifacts),
        "stable_temporal_artifacts": int(stable_temporal_artifacts),
        "temporal_periods": int(temporal_periods),
        "temporal_series_rows": int(temporal_series_rows),
        "temporal_event_rows": int(temporal_event_rows),
        "temporal_missing_years": int(temporal_missing_years),
        "evolution_artifacts": int(evolution_artifacts + len(artifacts.evolution_paths)),
        "stable_evolution_artifacts": int(stable_evolution_artifacts),
        "evolution_slices": int(evolution_slices),
        "evolution_states": int(evolution_states),
        "evolution_transitions": int(evolution_transitions),
        "evolution_lineages": int(evolution_lineages),
        "evolution_event_rows": int(evolution_event_rows),
        "legacy_evolution_artifacts": len(artifacts.evolution_paths),
        "narrative_artifacts": int(narrative_artifacts + len(artifacts.narrative_paths)),
        "stable_narrative_artifacts": int(stable_narrative_artifacts),
        "legacy_narrative_artifacts": len(artifacts.narrative_paths),
        "narrative_targets": int(narrative_targets),
        "narrative_claims": int(narrative_claims),
        "narrative_evidence_refs": int(narrative_evidence_refs),
        "narrative_aggregate_only_refs": int(narrative_aggregate_only_refs),
        "export_artifacts": int(export_artifacts),
        "stable_export_artifacts": int(stable_export_artifacts),
        "export_file_rows": int(export_file_rows),
        "keyword_artifact_rows": int(keyword_artifact_rows),
        "keyword_top_artifact_rows": int(keyword_top_artifact_rows),
        "keyword_review_artifact_rows": int(keyword_review_artifact_rows),
        "report_keyword_artifact_rows": int(report_artifact_rows),
        "report_keyword_top_artifact_rows": int(report_top_artifact_rows),
        "report_keyword_review_artifact_rows": int(report_review_artifact_rows),
    }

    features["overview"] = counts["abstract_rows"] > 0 or counts["report_clusters"] > 0
    features["cluster_map"] = counts["membership_rows"] > 0 or counts["report_clusters"] > 0
    features["keyword"] = counts["keyword_rows"] > 0 or _report_has_terms(report_clusters)
    legacy_matrix_artifacts = max(0, len(artifacts.matrix_paths) - cooccurrence_artifacts)
    counts["legacy_matrix_artifacts"] = int(legacy_matrix_artifacts)

    features["term_network"] = report_edge_count > 0 or keyword_edge_count > 0 or cooccurrence_rows > 0
    features["matrix"] = bool(general_matrix_artifacts or legacy_matrix_artifacts)
    features["evidence"] = (
        (counts["abstract_rows"] > 0 and counts["membership_rows"] > 0)
        or counts["edge_evidence_artifacts"] > 0
        or counts["stable_review_packet_artifacts"] > 0
    )
    has_pubyear = bool(abstract_info and abstract_info.columns and "pubyear" in abstract_info.columns)
    features["temporal"] = bool(temporal_artifacts or has_pubyear)
    features["evolution"] = bool(evolution_artifacts or artifacts.evolution_paths) or report_has_evolution
    features["narrative"] = bool(narrative_artifacts or artifacts.narrative_paths) or report_has_narrative
    features["quality"] = True
    features["export"] = any(
        [
            stable_export_artifacts > 0,
            features["keyword"],
            features["cluster_map"],
            artifacts.report_data_path is not None,
        ]
    )

    advertised = _report_metadata(report_data).get("features", {})
    if isinstance(advertised, dict):
        for feature, advertised_value in advertised.items():
            if feature in features and advertised_value is True and not features[feature]:
                issues.append(
                    ArtifactIssue(
                        "advertised_feature_missing",
                        "error",
                        f"Report advertises unavailable feature `{feature}`.",
                        "report_data",
                    )
                )

    if artifacts.report_data_path and not _report_metadata(report_data):
        issues.append(
            ArtifactIssue(
                "missing_report_contract",
                "info",
                "Report data has no `_sciscape` feature/version block.",
                "report_data",
            )
        )

    blocking = any(issue.is_blocking for issue in issues)
    if blocking:
        state = "blocked"
    elif any(features.values()):
        state = "loaded" if features["overview"] or features["keyword"] or features["cluster_map"] else "partial"
    else:
        state = "empty"

    return ArtifactValidationResult(
        schema_version=ARTIFACT_CONTRACT_SCHEMA_VERSION,
        mode=mode,
        result_state=state,
        result_root=str(root),
        source_uri=str(artifacts.input_path),
        features=features,
        warnings=_severity_dicts(issues),
        versions=_versions(),
        artifacts=artifact_info,
        counts=counts,
        created_at_utc=_utc_now(),
    )


def default_artifact_contract_path(result: ArtifactValidationResult) -> Path:
    root = Path(result.result_root)
    landscape = result.artifacts.get("landscape_dir")
    if landscape:
        return root / str(landscape) / "qa" / "artifact_contract.json"
    return root / "qa" / "artifact_contract.json"


def write_artifact_contract(
    path: str | Path,
    *,
    output_path: str | Path | None = None,
    mode: str = "local_result",
) -> ArtifactValidationResult:
    """Validate and write an artifact contract JSON report."""

    result = validate_result_root(path, mode=mode)
    target = Path(output_path) if output_path is not None else default_artifact_contract_path(result)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return result


def _safe_id(value: object, *, fallback: str = "result") -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "")).strip("._-")
    return text or fallback


def _artifact_format(path: str | Path | None) -> str:
    if path is None:
        return "unknown"
    suffix = Path(path).suffix.lower().lstrip(".")
    if suffix in {"parquet", "json", "html", "csv", "graphml", "gexf"}:
        return suffix
    return suffix or "directory"


def _path_size_bytes(root: Path, rel_path: str | None) -> int | None:
    if not rel_path:
        return None
    path = root / rel_path
    if not path.exists() or not path.is_file():
        return None
    return int(path.stat().st_size)


def _artifact_record(
    *,
    root: Path,
    role: str,
    path: str | None,
    required_for: list[str],
    table_info: Mapping[str, Any] | None = None,
    schema_version: str | None = None,
    description: str | None = None,
) -> dict[str, Any]:
    status = "missing"
    if path:
        candidate = root / path
        if candidate.exists():
            status = "present"
    record = ArtifactRecord(
        role=role,
        path=path or "",
        format=_artifact_format(path),
        status=status,
        required_for=required_for,
        schema_version=schema_version,
        rows=(
            int(table_info["rows"])
            if table_info and table_info.get("rows") is not None
            else None
        ),
        columns=list(table_info.get("columns") or []) if table_info else None,
        size_bytes=_path_size_bytes(root, path),
        description=description,
        warnings=[],
    )
    return asdict(record)


def _add_optional_artifact(
    records: dict[str, dict[str, Any]],
    *,
    key: str,
    root: Path,
    role: str,
    path: str | None,
    required_for: list[str],
    table_info: Mapping[str, Any] | None = None,
    schema_version: str | None = None,
    description: str | None = None,
) -> None:
    if not path:
        return
    records[key] = _artifact_record(
        root=root,
        role=role,
        path=path,
        required_for=required_for,
        table_info=table_info,
        schema_version=schema_version,
        description=description,
    )


def _unique_artifact_key(records: Mapping[str, Any], base_key: str) -> str:
    if base_key not in records:
        return base_key
    suffix = 2
    while f"{base_key}_{suffix}" in records:
        suffix += 1
    return f"{base_key}_{suffix}"


def _add_evolution_output_artifacts(
    records: dict[str, dict[str, Any]],
    *,
    root: Path,
    manifest_rel_path: str,
    suffix: str,
) -> None:
    try:
        validation = validate_evolution_artifact(root / manifest_rel_path).to_dict()
    except Exception:
        return
    paths = validation.get("paths") if isinstance(validation.get("paths"), Mapping) else {}
    counts = validation.get("counts") if isinstance(validation.get("counts"), Mapping) else {}
    evolution_dir = Path(manifest_rel_path).parent
    specs = [
        (
            "time_slices",
            "time_slices",
            "evolution_table",
            EVOLUTION_TIME_SLICES_SCHEMA_VERSION,
            "Evolution time-slice table.",
            counts.get("slices"),
        ),
        (
            "cluster_states",
            "cluster_states",
            "evolution_table",
            EVOLUTION_CLUSTER_STATES_SCHEMA_VERSION,
            "Evolution slice-local cluster state table.",
            counts.get("states"),
        ),
        (
            "state_membership",
            "state_membership",
            "evolution_table",
            EVOLUTION_STATE_MEMBERSHIP_SCHEMA_VERSION,
            "Optional evolution state-document membership table.",
            counts.get("state_membership_rows"),
        ),
        (
            "transitions",
            "transitions",
            "evolution_table",
            EVOLUTION_TRANSITIONS_SCHEMA_VERSION,
            "Evolution state-transition evidence table.",
            counts.get("transitions"),
        ),
        (
            "lineages",
            "lineages",
            "evolution_table",
            EVOLUTION_LINEAGES_SCHEMA_VERSION,
            "Evolution lineage table.",
            counts.get("lineages"),
        ),
        (
            "events",
            "events",
            "evolution_table",
            EVOLUTION_EVENTS_SCHEMA_VERSION,
            "Evolution event table.",
            counts.get("events", counts.get("event_rows")),
        ),
        ("qa", "qa", "qa", EVOLUTION_QA_SCHEMA_VERSION, "Evolution artifact QA report.", None),
    ]
    for output_key, base_key, role, schema_version, description, rows in specs:
        rel_output = paths.get(output_key)
        if not rel_output:
            continue
        key = f"evolution_{base_key}" if not suffix else f"evolution_{suffix}_{base_key}"
        records[_unique_artifact_key(records, key)] = _artifact_record(
            root=root,
            role=role,
            path=(evolution_dir / str(rel_output)).as_posix(),
            required_for=["quality", "evolution"] if role == "qa" else ["evolution"],
            table_info={"rows": rows} if rows is not None else None,
            schema_version=schema_version,
            description=description,
        )


def _is_cluster_sharded_keyword_manifest(payload: Mapping[str, Any] | None) -> bool:
    return bool(payload and payload.get("schema_version") == "sciscape_keyword_cluster_shards_v1")


def _cluster_sharded_keyword_dirs(root: Path, landscape_dir: Path | None) -> list[Path]:
    candidates: list[Path] = []
    seen: set[Path] = set()
    bases = [root]
    if landscape_dir is not None:
        bases.append(landscape_dir)
    for base in bases:
        if not base.exists() or not base.is_dir():
            continue
        manifest_candidates = [base / "manifest.json"]
        manifest_candidates.extend(base.glob("keyword*/manifest.json"))
        manifest_candidates.extend(base.glob("keyword*/*/manifest.json"))
        for manifest_path in manifest_candidates:
            payload = _read_run_json(manifest_path)
            if not _is_cluster_sharded_keyword_manifest(payload):
                continue
            output_dir = manifest_path.parent
            if output_dir not in seen:
                candidates.append(output_dir)
                seen.add(output_dir)
    return candidates


def _run_metadata_artifact_candidates(root: Path, landscape_dir: Path | None) -> list[tuple[str, str, str, str]]:
    bases = [root]
    if landscape_dir is not None:
        bases.append(landscape_dir)
    candidates: list[tuple[str, str, str, str]] = []
    for base in bases:
        for filename, key, role, description in [
            ("job_status.json", "job_status", "job_status", "Background query job status and progress messages."),
            ("keyword_progress.json", "keyword_progress", "progress", "Keyword extraction stage progress."),
            ("progress.json", "pipeline_progress", "progress", "Pipeline stage progress."),
            ("checkpoint_meta.json", "keyword_checkpoint", "checkpoint", "Keyword extraction checkpoint metadata."),
        ]:
            path = base / filename
            if path.exists() and path.is_file():
                rel_path = _rel(path, root)
                if rel_path:
                    candidates.append((key, role, rel_path, description))
        shard_manifest = base / "scoring_shards" / "manifest.json"
        if shard_manifest.exists() and shard_manifest.is_file():
            rel_path = _rel(shard_manifest, root)
            if rel_path:
                candidates.append(
                    (
                        "scoring_shard_manifest",
                        "shard_manifest",
                        rel_path,
                        "Keyword scoring shard manifest for resumable large-cluster extraction.",
                    )
                )
    for output_dir in _cluster_sharded_keyword_dirs(root, landscape_dir):
        for filename, key, role, description in [
            (
                "manifest.json",
                "keyword_cluster_shard_manifest",
                "shard_manifest",
                "Cluster-sharded keyword extraction shard and budget manifest.",
            ),
            (
                "progress.json",
                "keyword_cluster_sharded_progress",
                "progress",
                "Cluster-sharded keyword extraction stage progress.",
            ),
            (
                "preflight_summary.json",
                "keyword_cluster_sharded_preflight",
                "preflight",
                "Cluster-sharded keyword extraction preflight and budget summary.",
            ),
            (
                "run_summary.json",
                "keyword_cluster_sharded_run_summary",
                "run_summary",
                "Cluster-sharded keyword extraction completion summary.",
            ),
        ]:
            path = output_dir / filename
            if path.exists() and path.is_file():
                rel_path = _rel(path, root)
                if rel_path:
                    candidates.append((key, role, rel_path, description))
    return candidates


def _build_manifest_artifacts(validation: ArtifactValidationResult) -> dict[str, dict[str, Any]]:
    root = Path(validation.result_root)
    artifact_info = validation.artifacts
    tables = artifact_info.get("tables", {})
    landscape_dir = root / artifact_info["landscape_dir"] if artifact_info.get("landscape_dir") else None
    records: dict[str, dict[str, Any]] = {}

    _add_optional_artifact(
        records,
        key="records",
        root=root,
        role="records",
        path=artifact_info.get("abstracts"),
        required_for=["overview", "evidence", "temporal"],
        table_info=tables.get("abstracts"),
    )
    _add_optional_artifact(
        records,
        key="edges",
        root=root,
        role="edges",
        path=artifact_info.get("edges"),
        required_for=["cluster_map", "evidence"],
        table_info=tables.get("edges"),
    )
    _add_optional_artifact(
        records,
        key="membership",
        root=root,
        role="membership",
        path=artifact_info.get("membership"),
        required_for=["cluster_map", "evidence"],
        table_info=tables.get("membership"),
    )
    _add_optional_artifact(
        records,
        key="keywords",
        root=root,
        role="keywords",
        path=artifact_info.get("keywords"),
        required_for=["keyword", "term_network", "cooccurrence"],
        table_info=tables.get("keywords"),
    )
    _add_optional_artifact(
        records,
        key="report_data",
        root=root,
        role="report_data",
        path=artifact_info.get("report_data"),
        required_for=["overview", "cluster_map", "keyword", "export"],
        schema_version=REPORT_DATA_CONTRACT_SCHEMA_VERSION,
    )

    default_contract = _rel(default_artifact_contract_path(validation), root)
    records["artifact_contract"] = _artifact_record(
        root=root,
        role="qa",
        path=default_contract,
        required_for=["quality"],
        schema_version=ARTIFACT_CONTRACT_SCHEMA_VERSION,
    )

    for i, rel_path in enumerate(artifact_info.get("edge_evidence_artifacts", []), start=1):
        key = "edge_evidence" if i == 1 else f"edge_evidence_{i}"
        records[key] = _artifact_record(
            root=root,
            role="edge_evidence",
            path=rel_path,
            required_for=["evidence"],
            schema_version=EDGE_EVIDENCE_SCHEMA_VERSION,
        )

    for i, rel_path in enumerate(artifact_info.get("review_packet_artifacts", []), start=1):
        key = "cluster_review_packet" if i == 1 else f"cluster_review_packet_{i}"
        records[key] = _artifact_record(
            root=root,
            role="cluster_review_packet",
            path=rel_path,
            required_for=["evidence", "quality"],
            schema_version=CLUSTER_REVIEW_PACKET_SCHEMA_VERSION,
            description="Cluster review packet for evidence-backed label and narrative review.",
        )
        qa_path = Path(str(rel_path)).parent / "cluster_review_packet_qa.json"
        if (root / qa_path).exists():
            qa_key = "cluster_review_packet_qa" if "cluster_review_packet_qa" not in records else f"cluster_review_packet_qa_{i}"
            qa_suffix = 2
            while qa_key in records:
                qa_key = f"cluster_review_packet_qa_{qa_suffix}"
                qa_suffix += 1
            records[qa_key] = _artifact_record(
                root=root,
                role="qa",
                path=qa_path.as_posix(),
                required_for=["quality"],
                schema_version=CLUSTER_REVIEW_PACKET_QA_SCHEMA_VERSION,
                description="Cluster review packet QA report.",
            )

    for i, rel_path in enumerate(artifact_info.get("matrix_artifacts", []), start=1):
        role = "cooccurrence" if "cooccurrence" in str(rel_path).lower() else "matrix"
        key = role
        suffix = 2
        while key in records:
            key = f"{role}_{suffix}"
            suffix += 1
        records[key] = _artifact_record(
            root=root,
            role=role,
            path=rel_path,
            required_for=["cooccurrence"] if role == "cooccurrence" else ["matrix"],
            table_info=tables.get(f"{role}_artifact:{rel_path}"),
            schema_version=COOCCURRENCE_ARTIFACT_SCHEMA_VERSION if role == "cooccurrence" else None,
            description=(
                "Term co-occurrence table/map artifact."
                if role == "cooccurrence"
                else "Matrix artifact."
            ),
        )

    for i, rel_path in enumerate(artifact_info.get("matrix_manifest_artifacts", []), start=1):
        key = "matrix" if "matrix" not in records else f"matrix_{i}"
        suffix = 2
        while key in records:
            key = f"matrix_{suffix}"
            suffix += 1
        records[key] = _artifact_record(
            root=root,
            role="matrix",
            path=rel_path,
            required_for=["matrix"],
            schema_version=MATRIX_MANIFEST_SCHEMA_VERSION,
            description="General sparse-triplet matrix artifact manifest.",
        )
        matrix_dir = Path(str(rel_path)).parent
        manifest_path = root / rel_path
        outputs: Mapping[str, Any] = {}
        if manifest_path.exists():
            try:
                matrix_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if isinstance(matrix_manifest.get("outputs"), Mapping):
                    outputs = matrix_manifest["outputs"]
            except Exception:
                outputs = {}
        companion_specs = [
            (
                "values",
                "matrix_values",
                "matrix_values.parquet",
                MATRIX_VALUES_SCHEMA_VERSION,
                "Matrix sparse-triplet values.",
            ),
            (
                "rows",
                "matrix_rows",
                "row_entities.parquet",
                MATRIX_ENTITIES_SCHEMA_VERSION,
                "Matrix row entity metadata.",
            ),
            (
                "columns",
                "matrix_columns",
                "column_entities.parquet",
                MATRIX_ENTITIES_SCHEMA_VERSION,
                "Matrix column entity metadata.",
            ),
            (
                "qa",
                "qa",
                "matrix_qa.json",
                MATRIX_QA_SCHEMA_VERSION,
                "Matrix artifact QA report.",
            ),
        ]
        for output_key, role, default_name, schema_version, description in companion_specs:
            rel_output = Path(str(outputs.get(output_key) or default_name))
            rel_output_path = rel_output if rel_output.is_absolute() else matrix_dir / rel_output
            output_key_name = f"{key}_{output_key}"
            output_suffix = 2
            while output_key_name in records:
                output_key_name = f"{key}_{output_key}_{output_suffix}"
                output_suffix += 1
            records[output_key_name] = _artifact_record(
                root=root,
                role=role,
                path=rel_output_path.as_posix(),
                required_for=["matrix"] if role != "qa" else ["matrix", "quality"],
                schema_version=schema_version,
                description=description,
            )

    for i, rel_path in enumerate(artifact_info.get("keyword_rule_manifest_artifacts", []), start=1):
        key = "keyword_rules" if i == 1 else f"keyword_rules_{i}"
        suffix = 2
        while key in records:
            key = f"keyword_rules_{suffix}"
            suffix += 1
        records[key] = _artifact_record(
            root=root,
            role="keyword_rules",
            path=rel_path,
            required_for=["keyword", "quality"],
            schema_version=KEYWORD_RULE_MANIFEST_SCHEMA_VERSION,
            description="Keyword cleaning rule-set manifest.",
        )
        qa_path = Path(str(rel_path)).parent / "rule_set_qa.json"
        if (root / qa_path).exists():
            qa_key = "keyword_rule_qa" if "keyword_rule_qa" not in records else f"keyword_rule_qa_{i}"
            qa_suffix = 2
            while qa_key in records:
                qa_key = f"keyword_rule_qa_{qa_suffix}"
                qa_suffix += 1
            records[qa_key] = _artifact_record(
                root=root,
                role="qa",
                path=qa_path.as_posix(),
                required_for=["quality"],
                schema_version=KEYWORD_RULE_QA_SCHEMA_VERSION,
                description="Keyword cleaning rule-set QA report.",
            )

    for i, rel_path in enumerate(artifact_info.get("temporal_manifest_artifacts", []), start=1):
        key = "temporal" if i == 1 else f"temporal_{i}"
        records[key] = _artifact_record(
            root=root,
            role="temporal",
            path=rel_path,
            required_for=["temporal"],
            schema_version=TEMPORAL_MANIFEST_SCHEMA_VERSION,
            description="Artifact-backed temporal trend manifest.",
        )

    for i, rel_path in enumerate(artifact_info.get("evolution_manifest_artifacts", []), start=1):
        key = "evolution" if i == 1 else f"evolution_{i}"
        records[key] = _artifact_record(
            root=root,
            role="evolution",
            path=rel_path,
            required_for=["evolution"],
            schema_version=EVOLUTION_MANIFEST_SCHEMA_VERSION,
            description="Lineage-backed cluster evolution manifest.",
        )
        _add_evolution_output_artifacts(
            records,
            root=root,
            manifest_rel_path=str(rel_path),
            suffix="" if i == 1 else str(i),
        )

    for i, rel_path in enumerate(artifact_info.get("evolution_artifacts", []), start=1):
        key = "evolution" if "evolution" not in records else f"evolution_legacy_{i}"
        records[key] = _artifact_record(root=root, role="evolution", path=rel_path, required_for=["evolution"])

    for i, rel_path in enumerate(artifact_info.get("narrative_manifest_artifacts", []), start=1):
        key = "narrative" if "narrative" not in records else f"narrative_{i}"
        suffix = 2
        while key in records:
            key = f"narrative_{suffix}"
            suffix += 1
        records[key] = _artifact_record(
            root=root,
            role="narrative",
            path=rel_path,
            required_for=["narrative"],
            schema_version=NARRATIVE_MANIFEST_SCHEMA_VERSION,
            description="Evidence-backed narrative claim graph manifest.",
        )
        qa_path = Path(str(rel_path)).parent / "narrative_qa.json"
        if (root / qa_path).exists():
            qa_key = "narrative_qa" if "narrative_qa" not in records else f"narrative_qa_{i}"
            qa_suffix = 2
            while qa_key in records:
                qa_key = f"narrative_qa_{qa_suffix}"
                qa_suffix += 1
            records[qa_key] = _artifact_record(
                root=root,
                role="qa",
                path=qa_path.as_posix(),
                required_for=["quality"],
                schema_version=NARRATIVE_QA_SCHEMA_VERSION,
                description="Narrative claim graph QA report.",
            )

    for i, rel_path in enumerate(artifact_info.get("export_manifest_artifacts", []), start=1):
        key = "export" if "export" not in records else f"export_{i}"
        suffix = 2
        while key in records:
            key = f"export_{suffix}"
            suffix += 1
        records[key] = _artifact_record(
            root=root,
            role="export",
            path=rel_path,
            required_for=["export"],
            schema_version=EXPORT_MANIFEST_SCHEMA_VERSION,
            description="Manifest-backed export artifact.",
        )

    for i, rel_path in enumerate(artifact_info.get("narrative_artifacts", []), start=1):
        key = "narrative" if "narrative" not in records and i == 1 else f"narrative_legacy_{i}"
        records[key] = _artifact_record(root=root, role="narrative", path=rel_path, required_for=["narrative"])

    for key, role, rel_path, description in _run_metadata_artifact_candidates(root, landscape_dir):
        record_key = key
        suffix = 2
        while record_key in records:
            record_key = f"{key}_{suffix}"
            suffix += 1
        records[record_key] = _artifact_record(
            root=root,
            role=role,
            path=rel_path,
            required_for=["quality", "run_state"],
            description=description,
        )

    return records


def _present_artifact_refs(artifacts: Mapping[str, Mapping[str, Any]], candidates: list[str]) -> list[str]:
    refs: list[str] = []
    for artifact_key, artifact in artifacts.items():
        if artifact.get("status") not in {"present", "generated", "partial"}:
            continue
        if any(artifact_key == candidate or artifact_key.startswith(f"{candidate}_") for candidate in candidates):
            refs.append(artifact_key)
    return refs


def _feature_artifact_candidates(feature: str) -> list[str]:
    mapping = {
        "overview": ["records", "report_data"],
        "cluster_map": ["membership", "report_data"],
        "keyword": ["keywords", "keyword_rules", "report_data"],
        "term_network": ["term_network", "keywords", "report_data"],
        "cooccurrence": ["cooccurrence", "keywords", "report_data"],
        "matrix": ["matrix"],
        "evidence": ["records", "membership", "edge_evidence", "cluster_review_packet"],
        "temporal": ["records", "temporal"],
        "evolution": ["evolution"],
        "narrative": ["narrative"],
        "quality": ["artifact_contract", "keyword_rule_qa", "cluster_review_packet_qa"],
        "export": ["export", "keyword_rules", "report_data", "report_html", "viewer_html"],
    }
    return mapping.get(feature, [])


def _has_manifest_cooccurrence(validation: ArtifactValidationResult, artifacts: Mapping[str, Mapping[str, Any]]) -> bool:
    if any(record.get("role") == "cooccurrence" and record.get("status") == "present" for record in artifacts.values()):
        return True
    return bool(
        validation.counts.get("report_term_edges", 0) > 0
        or validation.counts.get("derived_keyword_term_edges", 0) > 0
    )


def _has_general_matrix_artifact(artifacts: Mapping[str, Mapping[str, Any]]) -> bool:
    return any(
        record.get("role") == "matrix"
        and record.get("status") == "present"
        and record.get("schema_version") == MATRIX_MANIFEST_SCHEMA_VERSION
        for record in artifacts.values()
    )


def _has_temporal_artifact(artifacts: Mapping[str, Mapping[str, Any]]) -> bool:
    return any(
        record.get("role") == "temporal"
        and record.get("status") == "present"
        and record.get("schema_version") == TEMPORAL_MANIFEST_SCHEMA_VERSION
        for record in artifacts.values()
    )


def _has_evolution_artifact(artifacts: Mapping[str, Mapping[str, Any]]) -> bool:
    return any(
        record.get("role") == "evolution"
        and record.get("status") == "present"
        and record.get("schema_version") == EVOLUTION_MANIFEST_SCHEMA_VERSION
        for record in artifacts.values()
    )


def _has_narrative_artifact(artifacts: Mapping[str, Mapping[str, Any]]) -> bool:
    return any(
        record.get("role") == "narrative"
        and record.get("status") == "present"
        and record.get("schema_version") == NARRATIVE_MANIFEST_SCHEMA_VERSION
        for record in artifacts.values()
    )


def _has_export_manifest_artifact(artifacts: Mapping[str, Mapping[str, Any]]) -> bool:
    return any(
        record.get("role") == "export"
        and record.get("status") == "present"
        and record.get("schema_version") == EXPORT_MANIFEST_SCHEMA_VERSION
        for record in artifacts.values()
    )


def _manifest_feature_available(feature: str, validation: ArtifactValidationResult, artifacts: Mapping[str, Mapping[str, Any]]) -> bool:
    if feature == "cooccurrence":
        return _has_manifest_cooccurrence(validation, artifacts)
    return bool(validation.features.get(feature, False))


def _feature_reason(feature: str, state: str, validation: ArtifactValidationResult, artifacts: Mapping[str, Mapping[str, Any]]) -> str:
    if validation.result_state == "blocked" and feature != "quality":
        return "result validation is blocked"
    if state == "hidden":
        return "feature is not backed by available artifacts"
    if feature == "quality" and state == "beta":
        return "result validation has warnings"
    if feature == "cooccurrence" and not any(record.get("role") == "cooccurrence" for record in artifacts.values()):
        return "derived from keyword/report term edges; stable co-occurrence artifact not written yet"
    if feature == "matrix" and not _has_general_matrix_artifact(artifacts):
        return "matrix-like artifact exists but stable general matrix artifact is not written yet"
    if feature == "temporal" and not _has_temporal_artifact(artifacts):
        return "pubyear exists but no temporal artifact has been written yet"
    if feature == "evolution" and not _has_evolution_artifact(artifacts):
        return "legacy/report evolution payload exists but stable evolution artifact has not been written yet"
    if feature == "narrative" and not _has_narrative_artifact(artifacts):
        return "legacy/report narrative payload exists but stable narrative claim graph has not been written yet"
    if feature == "export" and not _has_export_manifest_artifact(artifacts):
        return "legacy report/viewer export files are available but no stable export manifest has been written yet"
    if state == "beta":
        return "feature inferred with validation warnings or partial artifact coverage"
    return "feature validated"


def _feature_warning_payload(feature: str, warnings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    relevant: list[dict[str, Any]] = []
    candidates = set(_feature_artifact_candidates(feature))
    candidates.add(feature)
    aliases = {
        "keyword_rule": "keyword_rules",
        "keyword_rules": "keyword_rules",
        "keywords": "keywords",
        "report": "report_data",
        "report_data": "report_data",
        "qa": "quality",
        "artifact_contract": "quality",
    }
    for warning in warnings:
        if warning.get("severity") == "info":
            continue
        if feature == "quality":
            relevant.append(warning)
            continue
        artifact = str(warning.get("artifact") or "").strip()
        normalized = aliases.get(artifact, artifact)
        code = str(warning.get("code") or "").strip()
        if (
            normalized in candidates
            or any(normalized.startswith(f"{candidate}_") for candidate in candidates)
            or code.startswith(f"{feature}_")
        ):
            relevant.append(warning)
    return relevant


def _feature_exposures(
    validation: ArtifactValidationResult,
    artifacts: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    blocking = validation.result_state == "blocked"
    warning_payload = validation.warnings
    exposures: dict[str, dict[str, Any]] = {}
    for feature in RESULT_MANIFEST_FEATURE_KEYS:
        available = _manifest_feature_available(feature, validation, artifacts)
        feature_warnings = _feature_warning_payload(feature, warning_payload)
        if feature == "quality":
            state = "stable" if validation.ok and not feature_warnings else "beta"
        elif blocking:
            state = "hidden"
        elif not available:
            state = "hidden"
        elif feature == "cooccurrence" and not any(record.get("role") == "cooccurrence" for record in artifacts.values()):
            state = "beta"
        elif feature == "matrix" and not _has_general_matrix_artifact(artifacts):
            state = "beta"
        elif feature == "temporal" and not _has_temporal_artifact(artifacts):
            state = "beta"
        elif feature == "evolution" and not _has_evolution_artifact(artifacts):
            state = "beta"
        elif feature == "narrative" and not _has_narrative_artifact(artifacts):
            state = "beta"
        elif feature == "export" and not _has_export_manifest_artifact(artifacts):
            state = "beta"
        elif feature_warnings:
            state = "beta"
        else:
            state = "stable"
        refs = _present_artifact_refs(artifacts, _feature_artifact_candidates(feature))
        exposures[feature] = asdict(
            FeatureExposure(
                state=state,
                reason=_feature_reason(feature, state, validation, artifacts),
                artifact_refs=refs,
                warnings=feature_warnings if state == "beta" else [],
            )
        )
    return exposures


def _manifest_source(validation: ArtifactValidationResult, mode: str) -> dict[str, Any]:
    source_type = "unknown"
    if mode == "demo":
        source_type = "demo_fixture"
    elif mode in {"static_viewer", "report"}:
        source_type = "static_bundle"
    elif mode in {"live_query", "query_result"}:
        source_type = "openalex_query"
    elif validation.counts.get("abstract_rows", 0) > 0:
        source_type = "parquet_records"
    return {
        "source_type": source_type,
        "query": None,
        "source_files": [],
        "record_count": int(validation.counts.get("abstract_rows", 0)),
        "retrieved_at_utc": None,
        "filters": {},
        "source_uri": validation.source_uri,
    }


def _read_run_json(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _merge_run_entries(
    base_entries: list[dict[str, Any]] | None,
    override_entries: Any,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in list(base_entries or []) + list(override_entries or []):
        if not isinstance(item, Mapping):
            continue
        row = {key: value for key, value in dict(item).items() if value is not None}
        key = (str(row.get("path", "")), str(row.get("kind", row.get("role", ""))))
        if key in seen:
            entries = [existing for existing in entries if (str(existing.get("path", "")), str(existing.get("kind", existing.get("role", "")))) != key]
        seen.add(key)
        entries.append(row)
    return entries


def _merge_run_state(base: Mapping[str, Any], overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    if not overrides:
        return dict(base)
    merged = dict(base)
    for key, value in overrides.items():
        if key in {"progress", "shards", "resume", "failure"}:
            if value is None:
                merged[key] = None
            elif isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
                merged[key] = {**dict(merged[key]), **dict(value)}
            else:
                merged[key] = value
        elif key in {"checkpoints", "partial_outputs"}:
            merged[key] = _merge_run_entries(merged.get(key), value)
        else:
            merged[key] = value
    if merged.get("status") in {"queued", "running", "complete", "imported", "partial"} and "failure" not in overrides:
        merged["failure"] = None
    return merged


def _normalize_job_status(status: Any) -> str:
    mapping = {
        "pending": "queued",
        "queued": "queued",
        "running": "running",
        "done": "complete",
        "complete": "complete",
        "completed": "complete",
        "error": "failed",
        "failed": "failed",
    }
    return mapping.get(str(status or "").lower(), str(status or "imported"))


def _run_state_from_job_status(root: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    existing = payload.get("run_state")
    if isinstance(existing, Mapping):
        state = dict(existing)
    else:
        progress_messages = payload.get("progress")
        progress_count = len(progress_messages) if isinstance(progress_messages, list) else payload.get("progress_messages_count")
        state = {
            "status": _normalize_job_status(payload.get("status")),
            "started_at_utc": payload.get("started_at_utc"),
            "finished_at_utc": payload.get("finished_at_utc"),
            "heartbeat_at_utc": payload.get("updated_at_utc") or payload.get("heartbeat_at_utc"),
            "progress": {
                "current": int(progress_count or 0),
                "total": None,
                "unit": "messages",
            },
            "failure": {"reason": payload.get("error")} if payload.get("error") else None,
        }
    checkpoints = _merge_run_entries(
        state.get("checkpoints") if isinstance(state, Mapping) else [],
        [{"path": "job_status.json", "kind": "job_status", "status": "present"}],
    )
    state["checkpoints"] = checkpoints
    partial_outputs = state.get("partial_outputs")
    if not partial_outputs and isinstance(payload.get("partial_outputs"), list):
        state["partial_outputs"] = payload["partial_outputs"]
    return state


def _run_state_path_row(path: Path, root: Path, kind: str, status: str = "present", **extra: Any) -> dict[str, Any]:
    row = {
        "path": _rel(path, root) or str(path),
        "kind": kind,
        "status": status,
    }
    row.update({key: value for key, value in extra.items() if value is not None})
    return row


def _run_state_from_progress(root: Path, payload: Mapping[str, Any], path: Path) -> dict[str, Any]:
    processed = payload.get("processed")
    total = payload.get("total")
    stage = payload.get("stage")
    status = "complete" if stage == "complete" else "running"
    progress: dict[str, Any] = {
        "current": int(processed or 0),
        "total": int(total or 0),
        "unit": "clusters" if stage == "scoring_topk" else "stage_units",
        "stage": stage,
    }
    if payload.get("percent") is not None:
        progress["percent"] = payload.get("percent")
    return {
        "status": status,
        "heartbeat_at_utc": payload.get("updated_at_utc"),
        "progress": progress,
        "checkpoints": [_run_state_path_row(path, root, "progress")],
    }


def _run_state_from_shard_manifest(root: Path, payload: Mapping[str, Any], path: Path) -> dict[str, Any]:
    completed = payload.get("completed_shards")
    completed_count = len(completed) if isinstance(completed, list) else 0
    shard_count = int(payload.get("shard_count") or 0)
    partial_outputs = []
    if isinstance(completed, list):
        for row in completed:
            if not isinstance(row, Mapping) or not row.get("path"):
                continue
            shard_path = Path(str(row["path"]))
            if not shard_path.is_absolute():
                shard_path = path.parent / shard_path.name
            partial_outputs.append(
                _run_state_path_row(
                    shard_path,
                    root,
                    "scoring_shard",
                    row_start=row.get("row_start"),
                    row_end=row.get("row_end"),
                    loaded=row.get("loaded"),
                )
            )
    return {
        "status": _normalize_job_status(payload.get("status")),
        "heartbeat_at_utc": payload.get("updated_at_utc") or payload.get("created_at_utc"),
        "shards": {
            "total": shard_count,
            "complete": completed_count,
            "failed": 0,
            "running": 0 if payload.get("status") == "complete" else max(0, shard_count - completed_count),
        },
        "checkpoints": [_run_state_path_row(path, root, "shard_manifest")],
        "partial_outputs": partial_outputs,
        "resume": {"supported": bool(payload.get("resume")), "command": None},
    }


def _read_run_json_files(paths: Iterable[Path]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for path in paths:
        payload = _read_run_json(path)
        if payload:
            payload["_sidecar_path"] = path
            payloads.append(payload)
    return payloads


def _shard_ids_from_payloads(payloads: Iterable[Mapping[str, Any]]) -> set[int]:
    shard_ids: set[int] = set()
    for payload in payloads:
        shard_id = _coerce_int(payload.get("shard_id"))
        if shard_id is not None:
            shard_ids.add(shard_id)
    return shard_ids


def _run_state_partial_output_from_done(
    payload: Mapping[str, Any],
    *,
    root: Path,
    kind: str,
) -> dict[str, Any] | None:
    raw_path = payload.get("path") or payload.get("shard_path") or payload.get("output_path")
    if not raw_path:
        return None
    row = _run_state_path_row(
        Path(str(raw_path)),
        root,
        kind,
        status=str(payload.get("status") or "present"),
        rows=_coerce_int(payload.get("rows")),
        shard_id=_coerce_int(payload.get("shard_id")),
        source_rows=_coerce_int(payload.get("source_rows")),
        elapsed_sec=payload.get("elapsed_sec"),
        peak_rss_mb=payload.get("peak_rss_mb"),
    )
    flagged_path = payload.get("flagged_path")
    if flagged_path:
        row["flagged_path"] = _rel(Path(str(flagged_path)), root)
    return row


def _cluster_sharded_resume_command(
    *,
    output_dir: Path,
    manifest_payload: Mapping[str, Any],
    preflight_payload: Mapping[str, Any],
) -> str | None:
    abstract_path = preflight_payload.get("abstract_path") or manifest_payload.get("abstract_path")
    membership_path = preflight_payload.get("membership_path") or manifest_payload.get("membership_path")
    if not abstract_path or not membership_path:
        return None

    cluster_level = preflight_payload.get("cluster_level") or manifest_payload.get("cluster_level")
    artifact_dir = str(output_dir)
    output_path = str(output_dir / "keywords.parquet")
    progress_path = str(output_dir / "progress.json")
    parts: list[str] = [
        "sciscape",
        "keywords",
        str(abstract_path),
        str(membership_path),
        "--keyword-engine",
        "cluster_sharded",
    ]
    if cluster_level:
        parts.extend(["--cluster-level", str(cluster_level)])
    parts.extend(
        [
            "--cluster-sharded-output-dir",
            artifact_dir,
            "--progress-path",
            progress_path,
            "-o",
            output_path,
            "--scoring-shard-resume",
        ]
    )
    return " ".join(shlex.quote(part) for part in parts)


def _run_state_from_cluster_sharded_keyword_dir(
    root: Path,
    output_dir: Path,
    manifest_payload: Mapping[str, Any],
) -> dict[str, Any]:
    shard_rows = manifest_payload.get("shards") if isinstance(manifest_payload.get("shards"), list) else []
    shard_count = _coerce_int(manifest_payload.get("shard_count")) or len(shard_rows)
    run_summary = _read_run_json(output_dir / "run_summary.json") or {}
    progress = _read_run_json(output_dir / "progress.json") or {}
    preflight = _read_run_json(output_dir / "preflight_summary.json") or {}
    candidate_done = _read_run_json_files(sorted((output_dir / "candidates").glob("candidate_shard_*.done.json")))
    candidate_progress = _read_run_json_files(sorted((output_dir / "candidates").glob("candidate_shard_*.progress.json")))
    final_done = _read_run_json_files(sorted((output_dir / "final").glob("keyword_shard_*.done.json")))

    candidate_done_ids = _shard_ids_from_payloads(candidate_done)
    final_done_ids = _shard_ids_from_payloads(final_done)
    completed_ids = final_done_ids or candidate_done_ids
    incomplete_progress = [payload for payload in candidate_progress if _coerce_int(payload.get("shard_id")) not in candidate_done_ids]
    failed_ids = _shard_ids_from_payloads(
        payload for payload in incomplete_progress if str(payload.get("status") or "").lower() == "failed"
    )
    running_ids = _shard_ids_from_payloads(
        payload
        for payload in incomplete_progress
        if str(payload.get("status") or "").lower() in {"running", "selecting", "writing"}
    )
    complete_count = len(completed_ids)
    failed_count = len(failed_ids)
    running_count = len(running_ids) if running_ids else max(0, int(shard_count) - complete_count - failed_count)

    if str(run_summary.get("status") or "").lower() == "complete":
        status = "complete"
        complete_count = int(shard_count)
        failed_count = 0
        running_count = 0
    elif failed_count:
        status = "failed"
    elif complete_count or running_count:
        status = "running"
    else:
        status = "partial"

    checkpoints = [_run_state_path_row(output_dir / "manifest.json", root, "cluster_sharded_manifest")]
    for filename, kind in [
        ("progress.json", "progress"),
        ("preflight_summary.json", "preflight"),
        ("run_summary.json", "run_summary"),
    ]:
        path = output_dir / filename
        if path.exists() and path.is_file():
            checkpoints.append(_run_state_path_row(path, root, kind))

    partial_outputs = [
        row
        for row in (
            _run_state_partial_output_from_done(payload, root=root, kind="candidate_shard")
            for payload in candidate_done
        )
        if row is not None
    ]
    partial_outputs.extend(
        row
        for row in (
            _run_state_partial_output_from_done(payload, root=root, kind="keyword_shard")
            for payload in final_done
        )
        if row is not None
    )
    for filename, kind in [("keywords.parquet", "keywords"), ("keywords_flagged.parquet", "keywords_flagged")]:
        path = output_dir / filename
        if path.exists() and path.is_file():
            partial_outputs.append(_run_state_path_row(path, root, kind, rows=_coerce_int(run_summary.get("final_rows"))))

    progress_payload: dict[str, Any]
    if progress:
        processed = _coerce_int(progress.get("processed")) or _coerce_int(progress.get("current")) or complete_count
        total = _coerce_int(progress.get("total")) or shard_count
        progress_payload = {
            "current": int(processed),
            "total": int(total),
            "unit": progress.get("unit") or ("shards" if progress.get("stage") in {"candidate_mining", "final_scoring"} else "stage_units"),
            "stage": progress.get("stage"),
        }
        if progress.get("percent") is not None:
            progress_payload["percent"] = progress.get("percent")
    else:
        progress_payload = {
            "current": complete_count,
            "total": int(shard_count),
            "unit": "shards",
            "stage": "complete" if status == "complete" else "candidate_mining",
        }

    failure = None
    if failed_count:
        failure = {
            "reason": f"{failed_count} cluster-sharded keyword shard(s) failed",
            "failed_shards": sorted(failed_ids),
        }

    heartbeat = (
        run_summary.get("created_at_utc")
        or progress.get("updated_at_utc")
        or manifest_payload.get("updated_at_utc")
        or manifest_payload.get("created_at_utc")
    )
    resume_command = _cluster_sharded_resume_command(
        output_dir=output_dir,
        manifest_payload=manifest_payload,
        preflight_payload=preflight,
    )
    resume = {
        "supported": status != "complete",
        "command": resume_command if status != "complete" else None,
        "mode": "rerun_with_same_cluster_sharded_output_dir",
        "artifact_dir": _rel(output_dir, root),
    }
    if resume_command and status != "complete":
        resume["command_kind"] = "cli_resume"

    return {
        "status": status,
        "heartbeat_at_utc": heartbeat,
        "progress": progress_payload,
        "shards": {
            "total": int(shard_count),
            "complete": int(complete_count),
            "failed": int(failed_count),
            "running": int(running_count),
        },
        "checkpoints": checkpoints,
        "partial_outputs": partial_outputs,
        "failure": failure,
        "resume": resume,
    }


def _detected_run_state_overrides(validation: ArtifactValidationResult) -> dict[str, Any]:
    root = Path(validation.result_root)
    artifact_info = validation.artifacts
    landscape_dir = root / artifact_info["landscape_dir"] if artifact_info.get("landscape_dir") else None
    detected: dict[str, Any] = {}

    progress_candidates = [
        root / "keyword_progress.json",
        root / "progress.json",
    ]
    if landscape_dir is not None:
        progress_candidates.extend([landscape_dir / "keyword_progress.json", landscape_dir / "progress.json"])
    for path in progress_candidates:
        payload = _read_run_json(path)
        if payload:
            detected = _merge_run_state(detected, _run_state_from_progress(root, payload, path))

    shard_candidates = [root / "scoring_shards" / "manifest.json"]
    if landscape_dir is not None:
        shard_candidates.append(landscape_dir / "scoring_shards" / "manifest.json")
    for path in shard_candidates:
        payload = _read_run_json(path)
        if payload:
            detected = _merge_run_state(detected, _run_state_from_shard_manifest(root, payload, path))

    for output_dir in _cluster_sharded_keyword_dirs(root, landscape_dir):
        payload = _read_run_json(output_dir / "manifest.json")
        if payload:
            detected = _merge_run_state(
                detected,
                _run_state_from_cluster_sharded_keyword_dir(root, output_dir, payload),
            )

    job_payload = _read_run_json(root / "job_status.json")
    if job_payload:
        detected = _merge_run_state(detected, _run_state_from_job_status(root, job_payload))

    return detected


def _manifest_run_state(
    validation: ArtifactValidationResult,
    *,
    run_state_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if validation.result_state == "blocked":
        status = "failed"
        failure: dict[str, Any] | None = {
            "reason": "artifact validation blocked result",
            "warnings": validation.warnings,
        }
    elif validation.result_state in {"loaded", "partial"}:
        status = "complete"
        failure = None
    elif validation.result_state == "empty":
        status = "partial"
        failure = None
    else:
        status = "imported"
        failure = None
    base = asdict(
        RunState(
            status=status,
            heartbeat_at_utc=validation.created_at_utc,
            progress={"current": 100 if status == "complete" else 0, "total": 100, "unit": "percent"},
            shards={"total": 0, "complete": 0, "failed": 0, "running": 0},
            checkpoints=[],
            partial_outputs=[],
            failure=failure,
            resume={"supported": False, "command": None},
        )
    )
    detected = _detected_run_state_overrides(validation)
    return _merge_run_state(_merge_run_state(base, detected), run_state_overrides)


def _manifest_quality(validation: ArtifactValidationResult) -> dict[str, Any]:
    root = Path(validation.result_root)
    contract_path = _rel(default_artifact_contract_path(validation), root)
    gate_paths = [contract_path] if contract_path else []
    for rel_path in validation.artifacts.get("keyword_rule_manifest_artifacts", []):
        qa_path = Path(str(rel_path)).parent / "rule_set_qa.json"
        if (root / qa_path).exists():
            gate_paths.append(qa_path.as_posix())
    for rel_path in validation.artifacts.get("review_packet_artifacts", []):
        qa_path = Path(str(rel_path)).parent / "cluster_review_packet_qa.json"
        if (root / qa_path).exists():
            gate_paths.append(qa_path.as_posix())
    for rel_path in validation.artifacts.get("narrative_manifest_artifacts", []):
        qa_path = Path(str(rel_path)).parent / "narrative_qa.json"
        if (root / qa_path).exists():
            gate_paths.append(qa_path.as_posix())
    blocking_count = sum(1 for warning in validation.warnings if warning.get("severity") in {"error", "blocking"})
    if blocking_count:
        state = "blocked"
    elif validation.warnings:
        state = "passed_with_warnings"
    else:
        state = "passed"
    return {
        "validation_state": state,
        "artifact_contract_path": contract_path,
        "warning_count": len(validation.warnings),
        "blocking_count": blocking_count,
        "gate_paths": gate_paths,
        "last_validated_at_utc": validation.created_at_utc,
    }


def _manifest_optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        if pd.isna(value):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _manifest_export_file_inventory(root: Path, export_manifest_path: str | None) -> list[dict[str, Any]]:
    if not export_manifest_path:
        return []
    manifest_path = root / export_manifest_path
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        export_dir = manifest_path.parent
        files_path = _export_output_paths(export_dir, manifest)["files"]
        files = pd.read_parquet(files_path)
    except Exception:
        return []
    if files.empty:
        return []

    inventory: list[dict[str, Any]] = []
    for row in files.to_dict("records"):
        path = str(row.get("path") or "")
        resolved = root / path if path and not Path(path).is_absolute() else Path(path)
        exists = bool(path and not Path(path).is_absolute() and resolved.exists())
        inventory.append(
            {
                "file_id": str(row.get("file_id") or ""),
                "role": str(row.get("role") or "support"),
                "path": path,
                "format": str(row.get("format") or _export_file_format(path)),
                "public_share_state": str(row.get("public_share_state") or "local"),
                "bytes": _manifest_optional_int(row.get("bytes")),
                "exists": exists,
            }
        )
    return inventory


def _manifest_primary_export_path(files: list[dict[str, Any]], fallback: str | None) -> str | None:
    if not files:
        return fallback
    for preferred_role in ("primary", "map", "viewer", "report", "html", "network"):
        for row in files:
            if row.get("role") == preferred_role and row.get("path"):
                return str(row["path"])
    return str(files[0]["path"]) if files[0].get("path") else fallback


def _manifest_exports(validation: ArtifactValidationResult, artifacts: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    root = Path(validation.result_root)
    exports: list[dict[str, Any]] = []
    seen_export_ids: set[str] = set()
    seen_export_paths: set[str] = set()

    for summary in validation.artifacts.get("export_summaries", []):
        export_manifest_ref = summary.get("path")
        files = _manifest_export_file_inventory(root, export_manifest_ref)
        primary_path = _manifest_primary_export_path(files, export_manifest_ref)
        selection = _manifest_export_selection(root, export_manifest_ref)
        export_id = str(
            summary.get("export_id")
            or _safe_id(Path(str(export_manifest_ref or "export")).parent.name, fallback="export")
        )
        exports.append(
            {
                "export_id": export_id,
                "kind": summary.get("export_kind") or "manifest_export",
                "path": primary_path,
                "format": _export_file_format(primary_path or "export_manifest.json", "json"),
                "export_family": summary.get("export_family"),
                "export_manifest_ref": export_manifest_ref,
                "feature_refs": ["export"],
                "source_artifact_refs": ["export"],
                "status": summary.get("status") or "present",
                "counts": summary.get("counts", {}),
                "files": files,
                "selection": selection,
                "selection_summary": _manifest_selection_summary(selection),
            }
        )
        seen_export_ids.add(export_id)
        if primary_path:
            seen_export_paths.add(str(primary_path))

    def add_export(export_id: str, kind: str, path: str | None, fmt: str, features: list[str], source_refs: list[str]) -> None:
        if not path:
            return
        if export_id in seen_export_ids or path in seen_export_paths:
            return
        status = "present" if (root / path).exists() else "missing"
        exports.append(
            {
                "export_id": export_id,
                "kind": kind,
                "path": path,
                "format": fmt,
                "feature_refs": features,
                "source_artifact_refs": source_refs,
                "status": status,
            }
        )
        seen_export_ids.add(export_id)
        seen_export_paths.add(path)

    report_data = artifacts.get("report_data", {}).get("path")
    add_export("json_report_data", "json_report_data", report_data, "json", ["overview", "cluster_map"], ["report_data"])

    artifact_info = validation.artifacts
    landscape_dir = artifact_info.get("landscape_dir")
    if landscape_dir:
        add_export(
            "report_html",
            "html_report",
            f"{landscape_dir}/report/report.html",
            "html",
            ["overview", "keyword", "cluster_map"],
            ["report_data"],
        )
        add_export(
            "static_viewer",
            "static_viewer",
            f"{landscape_dir}/report/index.html",
            "html",
            ["overview", "keyword", "cluster_map"],
            ["report_data"],
        )

    for filename, kind, fmt in [
        ("network.gexf", "gexf_graph", "gexf"),
        ("network.graphml", "graphml_graph", "graphml"),
    ]:
        path = root / filename
        if path.exists():
            add_export(filename.replace(".", "_"), kind, filename, fmt, ["cluster_map"], ["edges", "membership"])

    return exports


def _manifest_title(root: Path, mode: str) -> str:
    name = root.name if root.name else "SciScape result"
    if mode and mode != "local_result":
        return f"{name} ({mode})"
    return name


def _manifest_result_kind(mode: str) -> str:
    mapping = {
        "demo": "demo_result",
        "static_viewer": "static_bundle",
        "report": "static_bundle",
        "live_query": "query_result",
        "query_result": "query_result",
        "file_pipeline": "file_pipeline",
        "local_result": "imported_result",
    }
    return mapping.get(mode, "imported_result")


def build_result_manifest(
    path: str | Path,
    *,
    mode: str = "local_result",
    source_overrides: Mapping[str, Any] | None = None,
    run_state_overrides: Mapping[str, Any] | None = None,
) -> ResultManifest:
    """Build a result-root manifest from the current artifact validator output."""

    validation = validate_result_root(path, mode=mode)
    root = Path(validation.result_root)
    artifacts = _build_manifest_artifacts(validation)
    now = _utc_now()
    result_id = _safe_id(f"{mode}_{root.name}", fallback="sciscape_result")
    source = _manifest_source(validation, mode)
    if source_overrides:
        source.update({key: value for key, value in source_overrides.items() if value is not None})
    return ResultManifest(
        schema_version=RESULT_MANIFEST_SCHEMA_VERSION,
        result_id=result_id,
        title=_manifest_title(root, mode),
        result_kind=_manifest_result_kind(mode),
        created_at_utc=now,
        updated_at_utc=now,
        sciscape_version=SCISCAPE_VERSION,
        result_root=".",
        source=source,
        run_state=_manifest_run_state(validation, run_state_overrides=run_state_overrides),
        artifacts=artifacts,
        features=_feature_exposures(validation, artifacts),
        quality=_manifest_quality(validation),
        exports=_manifest_exports(validation, artifacts),
        provenance={
            "commands": [],
            "config_paths": [],
            "config_hash": None,
            "git_commit": None,
            "git_dirty": None,
            "random_seed": None,
            "environment": {},
        },
    )


def default_result_manifest_path(result: ArtifactValidationResult | str | Path) -> Path:
    if isinstance(result, ArtifactValidationResult):
        root = Path(result.result_root)
    else:
        root = Path(validate_result_root(result).result_root)
    return root / "result_manifest.json"


def find_result_manifest_path(path: str | Path) -> Path | None:
    """Find a canonical or legacy result manifest next to a result root."""

    root = infer_result_artifacts(path).result_root
    for candidate in (root / "result_manifest.json", root / "MANIFEST.json"):
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def read_result_manifest(path: str | Path) -> dict[str, Any] | None:
    """Read an existing result manifest, returning ``None`` when absent."""

    manifest_path = find_result_manifest_path(path)
    if manifest_path is None:
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _merge_existing_manifest_metadata(
    generated: dict[str, Any],
    existing: Mapping[str, Any],
    *,
    preserve_existing_run_state: bool = False,
) -> dict[str, Any]:
    merged = dict(generated)
    for key in ("result_id", "title", "description", "tags", "ui", "notes", "created_at_utc"):
        value = existing.get(key)
        if value not in (None, ""):
            merged[key] = value
    if preserve_existing_run_state and isinstance(existing.get("run_state"), Mapping):
        merged["run_state"] = _merge_run_state(
            dict(generated.get("run_state") or {}),
            existing["run_state"],
        )
    if isinstance(existing.get("provenance"), Mapping):
        merged["provenance"] = {
            **dict(generated.get("provenance") or {}),
            **dict(existing["provenance"]),
        }
    return merged


def _attach_manifest_load_state(
    manifest: dict[str, Any],
    *,
    root: Path,
    manifest_path: Path | None,
    state: str,
    warning: str | None = None,
) -> dict[str, Any]:
    payload = dict(manifest)
    payload["manifest_state"] = state
    payload["manifest_path"] = _rel(manifest_path, root) if manifest_path is not None else None
    quality = dict(payload.get("quality") or {})
    quality["manifest_state"] = state
    quality["manifest_path"] = payload["manifest_path"]
    if warning:
        quality.setdefault("manifest_warnings", []).append(warning)
    payload["quality"] = quality
    return payload


def load_result_manifest(
    path: str | Path,
    *,
    mode: str = "local_result",
    source_overrides: Mapping[str, Any] | None = None,
    run_state_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load an existing manifest when available, with validator-backed state.

    Existing manifest metadata such as title, result_id, notes, and provenance is
    preserved, but artifact paths, feature exposure states, quality, and exports
    are regenerated from the current validator output.
    """

    validation = validate_result_root(path, mode=mode)
    root = Path(validation.result_root)
    generated = build_result_manifest(
        path,
        mode=mode,
        source_overrides=source_overrides,
        run_state_overrides=run_state_overrides,
    ).to_dict()
    manifest_path = find_result_manifest_path(path)
    if manifest_path is None:
        return _attach_manifest_load_state(
            generated,
            root=root,
            manifest_path=None,
            state="inferred",
        )
    state = "legacy" if manifest_path.name == "MANIFEST.json" else "present"
    try:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return _attach_manifest_load_state(
            generated,
            root=root,
            manifest_path=manifest_path,
            state="invalid",
            warning=f"Could not read result manifest: {exc}",
        )
    if existing.get("schema_version") != RESULT_MANIFEST_SCHEMA_VERSION:
        return _attach_manifest_load_state(
            generated,
            root=root,
            manifest_path=manifest_path,
            state="invalid",
            warning=f"Unsupported result manifest schema: {existing.get('schema_version')}",
        )
    has_run_state_sidecar = bool(_detected_run_state_overrides(validation)) or run_state_overrides is not None
    merged = _merge_existing_manifest_metadata(
        generated,
        existing,
        preserve_existing_run_state=not has_run_state_sidecar,
    )
    return _attach_manifest_load_state(
        merged,
        root=root,
        manifest_path=manifest_path,
        state=state,
    )


def write_result_manifest(
    path: str | Path,
    *,
    output_path: str | Path | None = None,
    mode: str = "local_result",
    source_overrides: Mapping[str, Any] | None = None,
    run_state_overrides: Mapping[str, Any] | None = None,
) -> ResultManifest:
    """Validate and write ``result_manifest.json`` for a SciScape result root."""

    manifest = build_result_manifest(
        path,
        mode=mode,
        source_overrides=source_overrides,
        run_state_overrides=run_state_overrides,
    )
    validation = validate_result_root(path, mode=mode)
    target = Path(output_path) if output_path is not None else default_result_manifest_path(validation)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def default_workspace_manifest_path(workspace_root: str | Path) -> Path:
    """Return the canonical ``workspace.json`` path for a workspace root."""

    return Path(workspace_root).expanduser().resolve() / "workspace.json"


def default_workspace_qa_path(workspace_root: str | Path) -> Path:
    """Return the canonical ``workspace_qa.json`` path for a workspace root."""

    return Path(workspace_root).expanduser().resolve() / "workspace_qa.json"


def read_workspace_manifest(workspace_root: str | Path) -> dict[str, Any] | None:
    """Read ``workspace.json`` when present."""

    manifest_path = default_workspace_manifest_path(workspace_root)
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _workspace_empty_objects() -> dict[str, list[dict[str, Any]]]:
    return {family: [] for family in WORKSPACE_OBJECT_FAMILIES}


def _workspace_empty_recent() -> dict[str, list[str]]:
    return {family: [] for family in ("projects", "runs", "results", "views", "exports")}


def _workspace_issue(
    code: str,
    severity: str,
    message: str,
    *,
    family: str | None = None,
    object_id: str | None = None,
    path: str | None = None,
) -> dict[str, Any]:
    issue: dict[str, Any] = {"code": code, "severity": severity, "message": message}
    if family:
        issue["family"] = family
    if object_id:
        issue["object_id"] = object_id
    if path:
        issue["path"] = path
    return issue


def _workspace_has_local_result_candidates(root: Path, *, max_dirs: int = 300) -> bool:
    names = {"result_manifest.json", "MANIFEST.json", "data.json", "membership.parquet"}
    for rel in ("workspace/output", "workspace/examples_output", "workspace/web_output", "viewer"):
        base = root / rel
        if not base.exists() or not base.is_dir():
            continue
        stack = [base]
        seen = 0
        while stack and seen < max_dirs:
            current = stack.pop()
            seen += 1
            try:
                children = list(current.iterdir())
            except OSError:
                continue
            if any(child.is_file() and child.name in names for child in children):
                return True
            stack.extend(child for child in children if child.is_dir())
    return False


def _workspace_normalize_objects(
    objects: Mapping[str, Any] | None = None,
    **overrides: list[Mapping[str, Any]] | None,
) -> dict[str, list[dict[str, Any]]]:
    normalized = _workspace_empty_objects()
    if isinstance(objects, Mapping):
        for family in WORKSPACE_OBJECT_FAMILIES:
            rows = objects.get(family)
            if isinstance(rows, list):
                normalized[family] = [dict(row) for row in rows if isinstance(row, Mapping)]
    for family, rows in overrides.items():
        if rows is not None:
            normalized[family] = [dict(row) for row in rows]
    return normalized


def _workspace_object_ids(objects: Mapping[str, list[Mapping[str, Any]]]) -> dict[str, set[str]]:
    ids: dict[str, set[str]] = {}
    for family, rows in objects.items():
        id_key = WORKSPACE_OBJECT_ID_KEYS.get(family, f"{family.rstrip('s')}_id")
        ids[family] = {str(row.get(id_key)) for row in rows if row.get(id_key)}
    return ids


def _workspace_ref_path(root: Path, rel_path: object) -> Path | None:
    if not rel_path:
        return None
    path = Path(str(rel_path))
    if path.is_absolute():
        return path
    return root / path


def _workspace_result_root_from_ref(root: Path, rel_path: object) -> Path | None:
    path = _workspace_ref_path(root, rel_path)
    if path is None:
        return None
    if path.name in {"result_manifest.json", "MANIFEST.json"}:
        return path.parent
    return infer_result_artifacts(path).result_root


def _workspace_ref_status(
    *,
    root: Path,
    family: str,
    row: Mapping[str, Any],
    object_id: str,
    warnings: list[dict[str, Any]],
    blocking_issues: list[dict[str, Any]],
) -> dict[str, Any]:
    ref = dict(row)
    raw_path = ref.get("path")
    if raw_path in (None, ""):
        warnings.append(
            _workspace_issue(
                "missing_object_path",
                "warning",
                f"Workspace {family} entry has no manifest path.",
                family=family,
                object_id=object_id,
            )
        )
        ref["path_state"] = "missing"
        return ref

    path_text = str(raw_path)
    path = Path(path_text)
    external = bool(ref.get("external"))
    if path.is_absolute() and not external:
        blocking_issues.append(
            _workspace_issue(
                "absolute_workspace_path",
                "blocking",
                "Workspace object paths must be relative unless marked external.",
                family=family,
                object_id=object_id,
                path=path_text,
            )
        )
        ref["path_state"] = "invalid"
        return ref

    if external:
        warnings.append(
            _workspace_issue(
                "external_workspace_ref",
                "warning",
                "Workspace object is marked as external and may not be shareable.",
                family=family,
                object_id=object_id,
                path=path_text,
            )
        )
        ref["path_state"] = "external"
        return ref

    resolved = root / path_text
    if resolved.exists():
        ref["path_state"] = "present"
    else:
        ref["path_state"] = "missing"
        if ref.get("state") not in {"missing", "stale", "archived", "external"}:
            warnings.append(
                _workspace_issue(
                    "missing_object_manifest",
                    "warning",
                    "Workspace object manifest path does not exist.",
                    family=family,
                    object_id=object_id,
                    path=path_text,
                )
            )

    if family == "results" and ref["path_state"] == "present":
        result_root = _workspace_result_root_from_ref(root, path_text)
        if result_root is not None:
            try:
                result_manifest = load_result_manifest(result_root)
            except Exception as exc:
                warnings.append(
                    _workspace_issue(
                        "result_ref_validation_failed",
                        "warning",
                        f"Could not validate registered result: {exc}",
                        family=family,
                        object_id=object_id,
                        path=path_text,
                    )
                )
            else:
                quality = dict(result_manifest.get("quality") or {})
                ref["manifest_state"] = result_manifest.get("manifest_state")
                ref["result_kind"] = result_manifest.get("result_kind")
                ref["run_status"] = dict(result_manifest.get("run_state") or {}).get("status")
                ref["validation_state"] = quality.get("validation_state")
                ref["feature_states"] = {
                    key: value.get("state")
                    for key, value in dict(result_manifest.get("features") or {}).items()
                    if isinstance(value, Mapping)
                }
                if quality.get("validation_state") == "blocked":
                    warnings.append(
                        _workspace_issue(
                            "registered_result_blocked",
                            "warning",
                            "Registered result validates as blocked.",
                            family=family,
                            object_id=object_id,
                            path=path_text,
                        )
                    )
    return ref


def _validate_workspace_payload(
    root: Path,
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
    qa_path: Path,
) -> WorkspaceValidationResult:
    warnings: list[dict[str, Any]] = []
    blocking_issues: list[dict[str, Any]] = []
    checks: dict[str, dict[str, Any]] = {}

    if manifest.get("schema_version") != WORKSPACE_MANIFEST_SCHEMA_VERSION:
        blocking_issues.append(
            _workspace_issue(
                "unsupported_workspace_schema",
                "blocking",
                f"Unsupported workspace manifest schema: {manifest.get('schema_version')}",
            )
        )
    checks["schema_supported"] = {
        "status": "blocked" if blocking_issues else "passed",
        "expected": WORKSPACE_MANIFEST_SCHEMA_VERSION,
        "actual": manifest.get("schema_version"),
    }

    raw_objects = manifest.get("objects")
    if not isinstance(raw_objects, Mapping):
        blocking_issues.append(
            _workspace_issue(
                "missing_workspace_objects",
                "blocking",
                "Workspace manifest must include an objects mapping.",
            )
        )
        raw_objects = {}

    objects = _workspace_empty_objects()
    counts: dict[str, int] = {}
    missing_refs = 0
    for family in WORKSPACE_OBJECT_FAMILIES:
        rows = raw_objects.get(family, [])
        if not isinstance(rows, list):
            blocking_issues.append(
                _workspace_issue(
                    "invalid_workspace_object_family",
                    "blocking",
                    "Workspace object family must be a list.",
                    family=family,
                )
            )
            rows = []
        id_key = WORKSPACE_OBJECT_ID_KEYS[family]
        seen: set[str] = set()
        normalized_rows: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                blocking_issues.append(
                    _workspace_issue(
                        "invalid_workspace_object_ref",
                        "blocking",
                        "Workspace object ref must be an object.",
                        family=family,
                    )
                )
                continue
            object_id = row.get(id_key)
            if object_id in (None, ""):
                blocking_issues.append(
                    _workspace_issue(
                        "missing_workspace_object_id",
                        "blocking",
                        f"Workspace {family} entry is missing {id_key}.",
                        family=family,
                    )
                )
                object_id = f"missing_{family}_{index}"
            object_id_text = str(object_id)
            if object_id_text in seen:
                blocking_issues.append(
                    _workspace_issue(
                        "duplicate_workspace_object_id",
                        "blocking",
                        f"Workspace {family} contains a duplicate {id_key}.",
                        family=family,
                        object_id=object_id_text,
                    )
                )
            seen.add(object_id_text)
            ref = _workspace_ref_status(
                root=root,
                family=family,
                row=row,
                object_id=object_id_text,
                warnings=warnings,
                blocking_issues=blocking_issues,
            )
            if ref.get("path_state") == "missing":
                missing_refs += 1
            normalized_rows.append(ref)
        objects[family] = normalized_rows
        counts[family] = len(normalized_rows)

    ids = _workspace_object_ids(objects)
    default_map = {
        "project_id": "projects",
        "result_id": "results",
        "view_id": "views",
    }
    defaults = dict(manifest.get("defaults") or {})
    for key, family in default_map.items():
        value = defaults.get(key)
        if value and str(value) not in ids.get(family, set()):
            blocking_issues.append(
                _workspace_issue(
                    "unresolved_workspace_default",
                    "blocking",
                    f"Workspace default {key} does not resolve to a registered {family} object.",
                    family=family,
                    object_id=str(value),
                )
            )
    for path in defaults.get("output_roots") or []:
        if Path(str(path)).is_absolute():
            blocking_issues.append(
                _workspace_issue(
                    "absolute_workspace_output_root",
                    "blocking",
                    "Workspace default output roots must be relative.",
                    path=str(path),
                )
            )

    recent = _workspace_empty_recent()
    raw_recent = manifest.get("recent") or {}
    if isinstance(raw_recent, Mapping):
        for family in recent:
            values = raw_recent.get(family, [])
            if isinstance(values, list):
                recent[family] = [str(value) for value in values]
            for value in recent[family]:
                if value not in ids.get(family, set()):
                    warnings.append(
                        _workspace_issue(
                            "unresolved_recent_workspace_ref",
                            "warning",
                            "Workspace recent ref does not resolve to a registered object.",
                            family=family,
                            object_id=value,
                        )
                    )
    elif raw_recent:
        warnings.append(
            _workspace_issue(
                "invalid_workspace_recent",
                "warning",
                "Workspace recent field should be an object.",
            )
        )

    for warning in manifest.get("warnings") or []:
        if isinstance(warning, Mapping):
            warnings.append(dict(warning))
        elif warning:
            warnings.append(_workspace_issue("workspace_manifest_warning", "warning", str(warning)))

    counts["missing_refs"] = missing_refs
    counts["warnings"] = len(warnings)
    counts["blocking_issues"] = len(blocking_issues)
    checks["object_refs"] = {
        "status": "blocked" if any(issue["code"].startswith("invalid_workspace_object") for issue in blocking_issues) else "passed",
        "counts": {family: counts[family] for family in WORKSPACE_OBJECT_FAMILIES},
        "missing_refs": missing_refs,
    }
    checks["unique_ids"] = {
        "status": "blocked" if any(issue["code"] == "duplicate_workspace_object_id" for issue in blocking_issues) else "passed",
    }
    checks["default_refs"] = {
        "status": "blocked" if any(issue["code"].startswith("unresolved_workspace_default") for issue in blocking_issues) else "passed",
    }
    checks["recent_refs"] = {
        "status": "warning" if any(issue["code"] == "unresolved_recent_workspace_ref" for issue in warnings) else "passed",
    }
    checks["result_refs"] = {
        "status": "warning" if any(issue["code"].startswith("registered_result") or issue["code"].startswith("result_ref") for issue in warnings) else "passed",
        "count": counts["results"],
    }

    if blocking_issues:
        state = "blocked"
        status = "blocked"
    elif warnings:
        state = "beta"
        status = "warning"
    else:
        state = "stable"
        status = "passed"

    return WorkspaceValidationResult(
        schema_version=WORKSPACE_QA_SCHEMA_VERSION,
        workspace_id=str(manifest.get("workspace_id")) if manifest.get("workspace_id") else None,
        state=state,
        status=status,
        workspace_root=str(root),
        manifest_path=_rel(manifest_path, root),
        qa_path=_rel(qa_path, root) or "workspace_qa.json",
        counts=counts,
        checks=checks,
        objects=objects,
        defaults=defaults,
        recent=recent,
        warnings=warnings,
        blocking_issues=blocking_issues,
        created_at_utc=_utc_now(),
    )


def validate_workspace(workspace_root: str | Path) -> WorkspaceValidationResult:
    """Validate a SciScape workspace registry."""

    root = Path(workspace_root).expanduser().resolve()
    manifest_path = default_workspace_manifest_path(root)
    qa_path = default_workspace_qa_path(root)
    if not manifest_path.exists():
        warnings = [
            _workspace_issue(
                "missing_workspace_manifest",
                "warning",
                "No workspace.json exists at the workspace root.",
                path="workspace.json",
            )
        ]
        has_candidates = _workspace_has_local_result_candidates(root)
        state = "inferred" if has_candidates else "hidden"
        return WorkspaceValidationResult(
            schema_version=WORKSPACE_QA_SCHEMA_VERSION,
            workspace_id=None,
            state=state,
            status="warning" if has_candidates else "passed",
            workspace_root=str(root),
            manifest_path=None,
            qa_path=_rel(qa_path, root) or "workspace_qa.json",
            counts={**{family: 0 for family in WORKSPACE_OBJECT_FAMILIES}, "missing_refs": 0, "warnings": len(warnings), "blocking_issues": 0},
            checks={
                "schema_supported": {"status": "missing"},
                "object_refs": {"status": "missing", "counts": {family: 0 for family in WORKSPACE_OBJECT_FAMILIES}},
            },
            objects=_workspace_empty_objects(),
            defaults={},
            recent=_workspace_empty_recent(),
            warnings=warnings,
            blocking_issues=[],
            created_at_utc=_utc_now(),
        )

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        blocking_issues = [
            _workspace_issue(
                "malformed_workspace_manifest",
                "blocking",
                f"Could not read workspace manifest: {exc}",
                path="workspace.json",
            )
        ]
        return WorkspaceValidationResult(
            schema_version=WORKSPACE_QA_SCHEMA_VERSION,
            workspace_id=None,
            state="blocked",
            status="blocked",
            workspace_root=str(root),
            manifest_path=_rel(manifest_path, root),
            qa_path=_rel(qa_path, root) or "workspace_qa.json",
            counts={**{family: 0 for family in WORKSPACE_OBJECT_FAMILIES}, "missing_refs": 0, "warnings": 0, "blocking_issues": len(blocking_issues)},
            checks={"schema_supported": {"status": "blocked"}},
            objects=_workspace_empty_objects(),
            defaults={},
            recent=_workspace_empty_recent(),
            warnings=[],
            blocking_issues=blocking_issues,
            created_at_utc=_utc_now(),
        )
    if not isinstance(manifest, Mapping):
        manifest = {"schema_version": None, "objects": {}}
    return _validate_workspace_payload(root, manifest, manifest_path=manifest_path, qa_path=qa_path)


def _write_workspace_payload(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = default_workspace_manifest_path(root)
    qa_path = default_workspace_qa_path(root)
    payload = dict(manifest)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    validation = validate_workspace(root)
    qa = validation.to_dict()
    qa_path.write_text(json.dumps(qa, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "manifest_path": manifest_path,
        "qa_path": qa_path,
        "manifest": payload,
        "qa": qa,
        "validation": validation,
    }


def write_workspace_manifest(
    workspace_root: str | Path,
    *,
    workspace_id: str,
    name: str,
    projects: list[Mapping[str, Any]] | None = None,
    datasets: list[Mapping[str, Any]] | None = None,
    runs: list[Mapping[str, Any]] | None = None,
    results: list[Mapping[str, Any]] | None = None,
    rule_sets: list[Mapping[str, Any]] | None = None,
    views: list[Mapping[str, Any]] | None = None,
    exports: list[Mapping[str, Any]] | None = None,
    defaults: Mapping[str, Any] | None = None,
    settings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a workspace registry and its QA sidecar."""

    root = Path(workspace_root).expanduser().resolve()
    existing = read_workspace_manifest(root) or {}
    now = _utc_now()
    objects = _workspace_normalize_objects(
        existing.get("objects") if isinstance(existing, Mapping) else None,
        projects=projects,
        datasets=datasets,
        runs=runs,
        results=results,
        rule_sets=rule_sets,
        views=views,
        exports=exports,
    )
    merged_defaults = {
        "mode": "local_result",
        "output_roots": ["workspace/output", "workspace/examples_output", "workspace/web_output"],
    }
    if isinstance(existing.get("defaults"), Mapping):
        merged_defaults.update(dict(existing["defaults"]))
    if defaults:
        merged_defaults.update(dict(defaults))
    merged_settings = {
        "auto_register_completed_runs": True,
        "show_legacy_results": True,
    }
    if isinstance(existing.get("settings"), Mapping):
        merged_settings.update(dict(existing["settings"]))
    if settings:
        merged_settings.update(dict(settings))
    recent = _workspace_empty_recent()
    if isinstance(existing.get("recent"), Mapping):
        for family in recent:
            values = existing["recent"].get(family, [])
            if isinstance(values, list):
                recent[family] = [str(value) for value in values]

    manifest = {
        "schema_version": WORKSPACE_MANIFEST_SCHEMA_VERSION,
        "workspace_id": workspace_id,
        "name": name,
        "root": ".",
        "created_at_utc": existing.get("created_at_utc") or now,
        "updated_at_utc": now,
        "objects": objects,
        "recent": recent,
        "defaults": merged_defaults,
        "settings": merged_settings,
        "warnings": list(existing.get("warnings") or []),
    }
    return _write_workspace_payload(root, manifest)


def register_result_in_workspace(
    workspace_root: str | Path,
    result_root: str | Path,
    *,
    project_id: str | None = None,
) -> dict[str, Any]:
    """Register an existing result root in ``workspace.json`` without moving it."""

    root = Path(workspace_root).expanduser().resolve()
    existing = read_workspace_manifest(root)
    if existing is None:
        existing = write_workspace_manifest(
            root,
            workspace_id=_safe_id(root.name, fallback="workspace_local_default"),
            name=root.name or "SciScape Local Workspace",
        )["manifest"]

    result_validation = validate_result_root(result_root)
    written_manifest = write_result_manifest(result_validation.result_root)
    result_manifest = written_manifest.to_dict()
    result_manifest_path = Path(result_validation.result_root) / "result_manifest.json"
    result_path = _rel(result_manifest_path, root)
    external = Path(result_path).is_absolute()
    result_id = str(result_manifest.get("result_id") or _safe_id(Path(result_validation.result_root).name))
    result_ref: dict[str, Any] = {
        "result_id": result_id,
        "path": result_path,
        "state": "validated" if result_validation.ok else "blocked",
        "title": result_manifest.get("title"),
        "result_kind": result_manifest.get("result_kind"),
        "validation_state": result_manifest.get("quality", {}).get("validation_state"),
        "updated_at_utc": _utc_now(),
    }
    if external:
        result_ref["external"] = True
    if project_id:
        result_ref["project_id"] = project_id

    objects = _workspace_normalize_objects(existing.get("objects") if isinstance(existing, Mapping) else None)
    results = objects["results"]
    replaced = False
    for index, row in enumerate(results):
        if row.get("result_id") == result_id or row.get("path") == result_path:
            results[index] = {**row, **result_ref}
            replaced = True
            break
    if not replaced:
        results.append(result_ref)
    objects["results"] = results

    recent = _workspace_empty_recent()
    raw_recent = existing.get("recent") if isinstance(existing, Mapping) else None
    if isinstance(raw_recent, Mapping):
        for family in recent:
            values = raw_recent.get(family, [])
            if isinstance(values, list):
                recent[family] = [str(value) for value in values]
    recent["results"] = [result_id] + [value for value in recent["results"] if value != result_id]
    recent["results"] = recent["results"][:10]

    defaults = dict(existing.get("defaults") or {}) if isinstance(existing, Mapping) else {}
    defaults.setdefault("result_id", result_id)
    if project_id and any(row.get("project_id") == project_id for row in objects["projects"]):
        defaults.setdefault("project_id", project_id)

    manifest = {
        "schema_version": WORKSPACE_MANIFEST_SCHEMA_VERSION,
        "workspace_id": existing.get("workspace_id") or _safe_id(root.name, fallback="workspace_local_default"),
        "name": existing.get("name") or root.name or "SciScape Local Workspace",
        "root": ".",
        "created_at_utc": existing.get("created_at_utc") or _utc_now(),
        "updated_at_utc": _utc_now(),
        "objects": objects,
        "recent": recent,
        "defaults": defaults,
        "settings": dict(existing.get("settings") or {}),
        "warnings": list(existing.get("warnings") or []),
    }
    written = _write_workspace_payload(root, manifest)
    written["registered_result"] = result_ref
    return written


def _feature_block_from_report_data(report_data: dict[str, Any]) -> dict[str, bool]:
    clusters = _report_clusters(report_data)
    term_edges = _report_term_edge_count(clusters)
    has_terms = _report_has_terms(clusters)
    has_evolution = _report_has_evolution(report_data, clusters)
    has_narrative = _report_has_narrative(report_data, clusters)
    features = {key: False for key in FEATURE_KEYS}
    features["overview"] = bool(clusters)
    features["cluster_map"] = bool(clusters)
    features["keyword"] = has_terms
    features["term_network"] = term_edges > 0
    features["matrix"] = False
    features["evidence"] = False
    features["temporal"] = bool(report_data.get("_trend_scores"))
    features["evolution"] = has_evolution
    features["narrative"] = has_narrative
    features["quality"] = False
    features["export"] = bool(clusters or has_terms)
    return features


def build_report_data_contract(report_data: dict[str, Any], *, mode: str = "static_viewer") -> dict[str, Any]:
    """Build the lightweight `_sciscape` block embedded in report ``data.json``."""

    features = _feature_block_from_report_data(report_data)
    atlas_payload = build_atlas_payload_from_report_data(report_data)
    result_state = "loaded" if any(features.values()) else "empty"
    warnings: list[dict[str, Any]] = []
    if not features["term_network"]:
        warnings.append(
            asdict(
                ArtifactIssue(
                    "missing_term_network",
                    "info",
                    "No term-network or co-occurrence rows are embedded in report data.",
                    "report_data",
                )
            )
        )
    warnings.extend(atlas_payload.get("warnings", []))
    return {
        "schema_version": REPORT_DATA_CONTRACT_SCHEMA_VERSION,
        "mode": mode,
        "result_state": result_state,
        "features": features,
        "warnings": warnings,
        "atlas": atlas_payload,
        "versions": {
            "sciscape_version": SCISCAPE_VERSION,
            "report_data_contract_schema_version": REPORT_DATA_CONTRACT_SCHEMA_VERSION,
            "atlas_payload_schema_version": ATLAS_PAYLOAD_SCHEMA_VERSION,
        },
        "created_at_utc": _utc_now(),
    }


__all__ = [
    "ARTIFACT_CONTRACT_SCHEMA_VERSION",
    "ATLAS_PAYLOAD_SCHEMA_VERSION",
    "CLUSTER_REVIEW_PACKET_QA_SCHEMA_VERSION",
    "CLUSTER_REVIEW_PACKET_SCHEMA_VERSION",
    "COOCCURRENCE_ARTIFACT_SCHEMA_VERSION",
    "EDGE_EVIDENCE_SCHEMA_VERSION",
    "EVOLUTION_CLUSTER_STATES_SCHEMA_VERSION",
    "EVOLUTION_EVENTS_SCHEMA_VERSION",
    "EVOLUTION_LINEAGES_SCHEMA_VERSION",
    "EVOLUTION_MANIFEST_SCHEMA_VERSION",
    "EVOLUTION_QA_SCHEMA_VERSION",
    "EVOLUTION_SYNTHETIC_SMOKE_SCHEMA_VERSION",
    "EVOLUTION_TIME_SLICES_SCHEMA_VERSION",
    "EVOLUTION_TRANSITIONS_SCHEMA_VERSION",
    "EXPORT_FILES_SCHEMA_VERSION",
    "EXPORT_INPUTS_SCHEMA_VERSION",
    "EXPORT_MANIFEST_SCHEMA_VERSION",
    "EXPORT_QA_SCHEMA_VERSION",
    "EXPORT_TRANSFORMS_SCHEMA_VERSION",
    "KEYWORD_RULE_APPLICATIONS_SCHEMA_VERSION",
    "KEYWORD_RULE_IMPACT_SUMMARY_SCHEMA_VERSION",
    "KEYWORD_RULE_MANIFEST_SCHEMA_VERSION",
    "KEYWORD_RULE_QA_SCHEMA_VERSION",
    "KEYWORD_RULES_SCHEMA_VERSION",
    "KEYWORD_TERM_BEFORE_AFTER_SCHEMA_VERSION",
    "MATRIX_ENTITIES_SCHEMA_VERSION",
    "MATRIX_MANIFEST_SCHEMA_VERSION",
    "MATRIX_QA_SCHEMA_VERSION",
    "MATRIX_VALUES_SCHEMA_VERSION",
    "NARRATIVE_CLAIM_EVIDENCE_LINKS_SCHEMA_VERSION",
    "NARRATIVE_CLAIMS_SCHEMA_VERSION",
    "NARRATIVE_EVIDENCE_REFS_SCHEMA_VERSION",
    "NARRATIVE_EVIDENCE_SOURCES_SCHEMA_VERSION",
    "NARRATIVE_MANIFEST_SCHEMA_VERSION",
    "NARRATIVE_QA_SCHEMA_VERSION",
    "NARRATIVE_REVIEW_DECISIONS_SCHEMA_VERSION",
    "NARRATIVE_SECTIONS_SCHEMA_VERSION",
    "NARRATIVE_TARGETS_SCHEMA_VERSION",
    "REPORT_DATA_CONTRACT_SCHEMA_VERSION",
    "RESULT_MANIFEST_SCHEMA_VERSION",
    "RESULT_MANIFEST_FEATURE_KEYS",
    "TEMPORAL_ACTIVITY_SCHEMA_VERSION",
    "TEMPORAL_ENTITY_SERIES_SCHEMA_VERSION",
    "TEMPORAL_EVENTS_SCHEMA_VERSION",
    "TEMPORAL_MANIFEST_SCHEMA_VERSION",
    "TEMPORAL_PERIODS_SCHEMA_VERSION",
    "TEMPORAL_QA_SCHEMA_VERSION",
    "WORKSPACE_MANIFEST_SCHEMA_VERSION",
    "WORKSPACE_QA_SCHEMA_VERSION",
    "ArtifactRecord",
    "ArtifactIssue",
    "ArtifactValidationResult",
    "ClusterReviewPacketValidationResult",
    "EvolutionArtifactValidationResult",
    "ExportManifestValidationResult",
    "FeatureExposure",
    "KeywordRuleValidationResult",
    "MatrixArtifactValidationResult",
    "NarrativeArtifactValidationResult",
    "ResultManifest",
    "ResultArtifacts",
    "RunState",
    "TemporalArtifactValidationResult",
    "WorkspaceValidationResult",
    "build_atlas_payload_from_report_data",
    "build_atlas_render_payload",
    "build_report_data_contract",
    "build_result_manifest",
    "default_artifact_contract_path",
    "default_result_manifest_path",
    "default_workspace_manifest_path",
    "default_workspace_qa_path",
    "find_result_manifest_path",
    "infer_result_artifacts",
    "load_result_manifest",
    "read_result_manifest",
    "read_workspace_manifest",
    "register_result_in_workspace",
    "validate_keyword_rule_artifact",
    "validate_cluster_review_packet_artifact",
    "validate_matrix_artifact",
    "validate_narrative_artifact",
    "validate_evolution_artifact",
    "validate_export_manifest",
    "validate_result_root",
    "validate_temporal_artifact",
    "validate_workspace",
    "write_edge_evidence_samples",
    "write_cluster_review_packet_artifact",
    "write_narrative_evidence_artifacts",
    "write_cooccurrence_artifacts",
    "write_keyword_rule_artifacts",
    "write_matrix_artifact",
    "write_matrix_from_term_cooccurrence",
    "write_evolution_artifacts",
    "write_document_overlap_evolution_artifacts",
    "write_evidence_backed_evolution_artifacts",
    "write_slice_local_membership_evolution_artifacts",
    "write_slice_reclustering_evolution_artifacts",
    "write_slice_membership_evolution_artifacts",
    "write_evolution_synthetic_smoke_artifact",
    "write_export_manifest",
    "write_temporal_artifacts",
    "write_artifact_contract",
    "write_result_manifest",
    "write_workspace_manifest",
]
