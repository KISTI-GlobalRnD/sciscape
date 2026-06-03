"""Result artifact validation and feature inference for SciScape outputs."""

from __future__ import annotations

import html
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import __version__ as SCISCAPE_VERSION


ARTIFACT_CONTRACT_SCHEMA_VERSION = "sciscape_artifact_contract_v1"
RESULT_MANIFEST_SCHEMA_VERSION = "sciscape_result_manifest_v1"
WORKSPACE_MANIFEST_SCHEMA_VERSION = "sciscape_workspace_manifest_v1"
WORKSPACE_QA_SCHEMA_VERSION = "sciscape_workspace_qa_v1"
REPORT_DATA_CONTRACT_SCHEMA_VERSION = "sciscape_report_data_contract_v1"
ATLAS_PAYLOAD_SCHEMA_VERSION = "sciscape_atlas_payload_v1"
EDGE_EVIDENCE_SCHEMA_VERSION = "sciscape_edge_evidence_samples_v1"
COOCCURRENCE_ARTIFACT_SCHEMA_VERSION = "sciscape_cooccurrence_artifact_v1"
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
    edge_evidence_paths: tuple[Path, ...] = ()
    evolution_paths: tuple[Path, ...] = ()
    narrative_paths: tuple[Path, ...] = ()
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
    for base in [landscape_dir, result_root]:
        if base is None or not base.exists() or not base.is_dir():
            continue
        for pattern in ("*matrix*.parquet", "*cooccurrence*.parquet", "*cooccurrence*.json"):
            matrix_paths.extend(path for path in base.glob(pattern) if path.is_file())

    evolution_paths = _collect_optional_artifacts(
        [
            landscape_dir,
            landscape_dir / "report" if landscape_dir else None,
            landscape_dir / "evolution" if landscape_dir else None,
            result_root,
        ],
        ("*evolution*.json", "*evolution*.parquet", "*trajectory*.json", "*trajectory*.parquet"),
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
        edge_evidence_paths=edge_evidence_paths,
        evolution_paths=evolution_paths,
        narrative_paths=narrative_paths,
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
        "edge_evidence_artifacts": [_rel(path, root) for path in artifacts.edge_evidence_paths],
        "evolution_artifacts": [_rel(path, root) for path in artifacts.evolution_paths],
        "narrative_artifacts": [_rel(path, root) for path in artifacts.narrative_paths],
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
        "cooccurrence_artifacts": int(cooccurrence_artifacts),
        "cooccurrence_rows": int(cooccurrence_rows),
        "edge_evidence_artifacts": len(artifacts.edge_evidence_paths),
        "evolution_artifacts": len(artifacts.evolution_paths),
        "narrative_artifacts": len(artifacts.narrative_paths),
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
    features["term_network"] = report_edge_count > 0 or keyword_edge_count > 0 or cooccurrence_rows > 0
    features["matrix"] = bool(artifacts.matrix_paths) or report_edge_count > 0
    features["evidence"] = counts["abstract_rows"] > 0 and counts["membership_rows"] > 0
    features["temporal"] = bool(abstract_info and abstract_info.columns and "pubyear" in abstract_info.columns)
    features["evolution"] = bool(artifacts.evolution_paths) or report_has_evolution
    features["narrative"] = bool(artifacts.narrative_paths) or report_has_narrative
    features["quality"] = True
    features["export"] = any([features["keyword"], features["cluster_map"], artifacts.report_data_path is not None])

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

    for i, rel_path in enumerate(artifact_info.get("matrix_artifacts", []), start=1):
        role = "cooccurrence" if "cooccurrence" in str(rel_path).lower() else "matrix"
        key = role if i == 1 else f"{role}_{i}"
        records[key] = _artifact_record(
            root=root,
            role=role,
            path=rel_path,
            required_for=["cooccurrence", "matrix"] if role == "cooccurrence" else ["matrix"],
            table_info=tables.get(f"{role}_artifact:{rel_path}"),
            schema_version=COOCCURRENCE_ARTIFACT_SCHEMA_VERSION if role == "cooccurrence" else None,
            description=(
                "Term co-occurrence table/map artifact."
                if role == "cooccurrence"
                else "Matrix artifact."
            ),
        )

    for i, rel_path in enumerate(artifact_info.get("evolution_artifacts", []), start=1):
        key = "evolution" if i == 1 else f"evolution_{i}"
        records[key] = _artifact_record(root=root, role="evolution", path=rel_path, required_for=["evolution"])

    for i, rel_path in enumerate(artifact_info.get("narrative_artifacts", []), start=1):
        key = "narrative" if i == 1 else f"narrative_{i}"
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
    return [
        key
        for key in candidates
        if key in artifacts and artifacts[key].get("status") in {"present", "generated", "partial"}
    ]


def _feature_artifact_candidates(feature: str) -> list[str]:
    mapping = {
        "overview": ["records", "report_data"],
        "cluster_map": ["membership", "report_data"],
        "keyword": ["keywords", "report_data"],
        "term_network": ["term_network", "keywords", "report_data"],
        "cooccurrence": ["cooccurrence", "matrix", "keywords", "report_data"],
        "matrix": ["matrix", "cooccurrence", "report_data"],
        "evidence": ["records", "membership", "edge_evidence"],
        "temporal": ["records", "temporal"],
        "evolution": ["evolution"],
        "narrative": ["narrative"],
        "quality": ["artifact_contract"],
        "export": ["report_data", "report_html", "viewer_html", "export_manifest"],
    }
    return mapping.get(feature, [])


def _has_manifest_cooccurrence(validation: ArtifactValidationResult, artifacts: Mapping[str, Mapping[str, Any]]) -> bool:
    if any(record.get("role") == "cooccurrence" and record.get("status") == "present" for record in artifacts.values()):
        return True
    return bool(
        validation.counts.get("report_term_edges", 0) > 0
        or validation.counts.get("derived_keyword_term_edges", 0) > 0
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
    if feature == "cooccurrence" and not any(record.get("role") == "cooccurrence" for record in artifacts.values()):
        return "derived from keyword/report term edges; stable co-occurrence artifact not written yet"
    if state == "beta":
        return "feature inferred with validation warnings or partial artifact coverage"
    return "feature validated"


def _feature_exposures(
    validation: ArtifactValidationResult,
    artifacts: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    blocking = validation.result_state == "blocked"
    warning_payload = validation.warnings
    warning_count = sum(1 for warning in warning_payload if warning.get("severity") not in {"info"})
    exposures: dict[str, dict[str, Any]] = {}
    for feature in RESULT_MANIFEST_FEATURE_KEYS:
        available = _manifest_feature_available(feature, validation, artifacts)
        if feature == "quality":
            state = "stable" if validation.ok else "beta"
        elif blocking:
            state = "hidden"
        elif not available:
            state = "hidden"
        elif feature == "cooccurrence" and not any(record.get("role") == "cooccurrence" for record in artifacts.values()):
            state = "beta"
        elif warning_count:
            state = "beta"
        else:
            state = "stable"
        refs = _present_artifact_refs(artifacts, _feature_artifact_candidates(feature))
        exposures[feature] = asdict(
            FeatureExposure(
                state=state,
                reason=_feature_reason(feature, state, validation, artifacts),
                artifact_refs=refs,
                warnings=warning_payload if state == "beta" else [],
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
        "gate_paths": [contract_path] if contract_path else [],
        "last_validated_at_utc": validation.created_at_utc,
    }


def _manifest_exports(validation: ArtifactValidationResult, artifacts: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    root = Path(validation.result_root)
    exports: list[dict[str, Any]] = []

    def add_export(export_id: str, kind: str, path: str | None, fmt: str, features: list[str], source_refs: list[str]) -> None:
        if not path:
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
    features["matrix"] = term_edges > 0
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
    "COOCCURRENCE_ARTIFACT_SCHEMA_VERSION",
    "EDGE_EVIDENCE_SCHEMA_VERSION",
    "REPORT_DATA_CONTRACT_SCHEMA_VERSION",
    "RESULT_MANIFEST_SCHEMA_VERSION",
    "RESULT_MANIFEST_FEATURE_KEYS",
    "WORKSPACE_MANIFEST_SCHEMA_VERSION",
    "WORKSPACE_QA_SCHEMA_VERSION",
    "ArtifactRecord",
    "ArtifactIssue",
    "ArtifactValidationResult",
    "FeatureExposure",
    "ResultManifest",
    "ResultArtifacts",
    "RunState",
    "WorkspaceValidationResult",
    "build_atlas_payload_from_report_data",
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
    "validate_result_root",
    "validate_workspace",
    "write_edge_evidence_samples",
    "write_cooccurrence_artifacts",
    "write_artifact_contract",
    "write_result_manifest",
    "write_workspace_manifest",
]
