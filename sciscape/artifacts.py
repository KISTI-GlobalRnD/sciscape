"""Result artifact validation and feature inference for SciScape outputs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import __version__ as SCISCAPE_VERSION


ARTIFACT_CONTRACT_SCHEMA_VERSION = "sciscape_artifact_contract_v1"
REPORT_DATA_CONTRACT_SCHEMA_VERSION = "sciscape_report_data_contract_v1"
FEATURE_KEYS = (
    "overview",
    "cluster_map",
    "keyword",
    "term_network",
    "matrix",
    "evidence",
    "temporal",
    "quality",
    "export",
)
REQUIRED_ABSTRACT_COLUMNS = {"uid", "title", "abstract", "pubyear"}
REQUIRED_MEMBERSHIP_COLUMNS = {"uid"}
REQUIRED_KEYWORD_COLUMNS = {"cluster_id", "term"}
REQUIRED_EDGE_COLUMNS = {"uid1", "uid2"}
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
        return [item for item in clusters if isinstance(item, dict)]
    return [
        value
        for key, value in report_data.items()
        if not str(key).startswith("_") and isinstance(value, dict)
    ]


def _report_metadata(report_data: dict[str, Any] | None) -> dict[str, Any]:
    if not report_data:
        return {}
    meta = report_data.get("_sciscape")
    if isinstance(meta, dict):
        return meta
    return {}


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
        "qa": _rel(artifacts.qa_path, root),
        "tables": {},
    }

    report_data = _json_payload(artifacts.report_data_path, issues, "report_data") if artifacts.report_data_path else None
    report_clusters = _report_clusters(report_data)
    report_edge_count = _report_term_edge_count(report_clusters)
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
    features["term_network"] = report_edge_count > 0 or keyword_edge_count > 0
    features["matrix"] = bool(artifacts.matrix_paths) or report_edge_count > 0
    features["evidence"] = counts["abstract_rows"] > 0 and counts["membership_rows"] > 0
    features["temporal"] = bool(abstract_info and abstract_info.columns and "pubyear" in abstract_info.columns)
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


def _feature_block_from_report_data(report_data: dict[str, Any]) -> dict[str, bool]:
    clusters = _report_clusters(report_data)
    term_edges = _report_term_edge_count(clusters)
    has_terms = _report_has_terms(clusters)
    features = {key: False for key in FEATURE_KEYS}
    features["overview"] = bool(clusters)
    features["cluster_map"] = bool(clusters)
    features["keyword"] = has_terms
    features["term_network"] = term_edges > 0
    features["matrix"] = term_edges > 0
    features["evidence"] = False
    features["temporal"] = bool(report_data.get("_trend_scores"))
    features["quality"] = False
    features["export"] = bool(clusters or has_terms)
    return features


def build_report_data_contract(report_data: dict[str, Any], *, mode: str = "static_viewer") -> dict[str, Any]:
    """Build the lightweight `_sciscape` block embedded in report ``data.json``."""

    features = _feature_block_from_report_data(report_data)
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
    return {
        "schema_version": REPORT_DATA_CONTRACT_SCHEMA_VERSION,
        "mode": mode,
        "result_state": result_state,
        "features": features,
        "warnings": warnings,
        "versions": {
            "sciscape_version": SCISCAPE_VERSION,
            "report_data_contract_schema_version": REPORT_DATA_CONTRACT_SCHEMA_VERSION,
        },
        "created_at_utc": _utc_now(),
    }


__all__ = [
    "ARTIFACT_CONTRACT_SCHEMA_VERSION",
    "REPORT_DATA_CONTRACT_SCHEMA_VERSION",
    "ArtifactIssue",
    "ArtifactValidationResult",
    "ResultArtifacts",
    "build_report_data_contract",
    "default_artifact_contract_path",
    "infer_result_artifacts",
    "validate_result_root",
    "write_artifact_contract",
]
