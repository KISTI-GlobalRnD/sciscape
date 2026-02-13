"""Utilities for selecting representative documents from the database."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING, Union

try:  # optional dependency for SQLite
    import sqlite3
except ImportError:  # pragma: no cover - optional dependency
    sqlite3 = None

if TYPE_CHECKING:  # pragma: no cover - optional dependency typing
    import pandas as pd

try:  # optional dependency for MySQL
    import pymysql
except ImportError:  # pragma: no cover - optional dependency
    pymysql = None

DEFAULT_UID_COL = "uid"
DEFAULT_TITLE_COL = "title"
DEFAULT_ABSTRACT_COL = "abstract"
DEFAULT_YEAR_COL = "pubyear"
DEFAULT_CITATION_COL = "citation_count"

ENV_CONFIG_PATH = "OLLAMA_CONFIG"
ENV_DB_PATH = "CLUSTER_DB_PATH"
ENV_DB_NAME = "CLUSTER_DB_NAME"
ENV_DB_DRIVER = "CLUSTER_DB_DRIVER"
ENV_DB_HOST = "CLUSTER_DB_HOST"
ENV_DB_PORT = "CLUSTER_DB_PORT"
ENV_DB_USER = "CLUSTER_DB_USER"
ENV_DB_PASSWORD = "CLUSTER_DB_PASSWORD"
ENV_META_TABLE = "CLUSTER_META_TABLE"
ENV_METRIC_TABLE = "CLUSTER_METRIC_TABLE"
ENV_TITLE_TABLE = "CLUSTER_TITLE_TABLE"
ENV_ABSTRACT_TABLE = "CLUSTER_ABSTRACT_TABLE"
ENV_YEAR_TABLE = "CLUSTER_YEAR_TABLE"
ENV_CITATION_TABLE = "CLUSTER_CITATION_TABLE"
ENV_TEMP_TABLE = "CLUSTER_TEMP_TABLE"
ENV_UID_COL = "CLUSTER_UID_COL"
ENV_TITLE_COL = "CLUSTER_TITLE_COL"
ENV_ABSTRACT_COL = "CLUSTER_ABSTRACT_COL"
ENV_YEAR_COL = "CLUSTER_YEAR_COL"
ENV_CITATION_COL = "CLUSTER_CITATION_COL"


@dataclass
class ClusterDocument:
    """Representative document used for cluster naming."""

    uid: str
    title: str
    abstract: str
    pubyear: Optional[int] = None
    citation_count: Optional[float] = None


@dataclass
class ClusterDBConfig:
    """Database configuration for extracting documents."""

    db_path: Optional[str] = None
    db_name: Optional[str] = None
    driver: str = "mysql"
    host: Optional[str] = None
    port: Optional[int] = None
    user: Optional[str] = None
    password: Optional[str] = None
    meta_table: str = "paper_metadata"
    metric_table: Optional[str] = None
    title_table: Optional[str] = None
    abstract_table: Optional[str] = None
    year_table: Optional[str] = None
    citation_table: Optional[str] = None
    temp_table: Optional[str] = None
    uid_col: str = DEFAULT_UID_COL
    title_col: str = DEFAULT_TITLE_COL
    abstract_col: str = DEFAULT_ABSTRACT_COL
    year_col: str = DEFAULT_YEAR_COL
    citation_col: str = DEFAULT_CITATION_COL


FieldSpec = Tuple[Optional[str], str]
FieldMap = OrderedDict[str, FieldSpec]
ExtraFieldSpec = Union[FieldSpec, Mapping[str, str]]


def _load_env_file(env_path: Path) -> Dict[str, str]:
    data: Dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip()
    return data


def load_db_config() -> ClusterDBConfig:
    from os import environ

    config = ClusterDBConfig()
    env_path = environ.get(ENV_CONFIG_PATH)
    overlay: Dict[str, str] = {}
    env_file_path: Optional[Path] = None
    if env_path:
        candidate = Path(env_path)
        if candidate.exists():
            env_file_path = candidate
    else:
        candidates = [
            Path.cwd() / ".env",
            Path(__file__).resolve().parent / ".env",
            Path(__file__).resolve().parent.parent / ".env",
        ]
        for candidate in candidates:
            if candidate.exists():
                env_file_path = candidate
                break
    if env_file_path:
        overlay = _load_env_file(env_file_path)

    def get_value(key: str, default: Optional[str]) -> Optional[str]:
        if key in overlay:
            return overlay[key]
        return environ.get(key, default)

    config.driver = get_value(ENV_DB_DRIVER, config.driver) or config.driver
    config.db_path = get_value(ENV_DB_PATH, config.db_path)
    config.db_name = get_value(ENV_DB_NAME, config.db_name)
    config.host = get_value(ENV_DB_HOST, config.host)
    port_value = get_value(ENV_DB_PORT, None)
    if port_value:
        try:
            config.port = int(port_value)
        except ValueError:
            pass
    config.user = get_value(ENV_DB_USER, config.user)
    config.password = get_value(ENV_DB_PASSWORD, config.password)
    config.meta_table = get_value(ENV_META_TABLE, config.meta_table) or config.meta_table
    config.metric_table = get_value(ENV_METRIC_TABLE, config.metric_table)
    config.title_table = get_value(ENV_TITLE_TABLE, config.title_table) or config.title_table
    config.abstract_table = get_value(ENV_ABSTRACT_TABLE, config.abstract_table) or config.abstract_table
    config.year_table = get_value(ENV_YEAR_TABLE, config.year_table) or config.year_table
    config.citation_table = get_value(ENV_CITATION_TABLE, config.citation_table) or config.citation_table
    config.temp_table = get_value(ENV_TEMP_TABLE, config.temp_table) or config.temp_table
    config.uid_col = get_value(ENV_UID_COL, config.uid_col) or config.uid_col
    config.title_col = get_value(ENV_TITLE_COL, config.title_col) or config.title_col
    config.abstract_col = get_value(ENV_ABSTRACT_COL, config.abstract_col) or config.abstract_col
    config.year_col = get_value(ENV_YEAR_COL, config.year_col) or config.year_col
    config.citation_col = get_value(ENV_CITATION_COL, config.citation_col) or config.citation_col

    return config


def _determine_base_table(
    config: ClusterDBConfig,
    fields: Optional[Mapping[str, FieldSpec]] = None,
) -> str:
    if fields:
        uid_spec = fields.get("uid")
        if uid_spec and uid_spec[0]:
            return uid_spec[0]
        for table, _ in fields.values():
            if table:
                return table
    for candidate in (
        config.meta_table,
        config.title_table,
        config.abstract_table,
        config.year_table,
        config.citation_table,
        config.metric_table,
    ):
        if candidate:
            return candidate
    raise ValueError(
        "Unable to determine base table; set meta_table or provide a field map with table hints."
    )


def _normalize_extra_fields(
    extra_fields: Mapping[str, ExtraFieldSpec],
    default_table: Optional[str],
) -> FieldMap:
    result: FieldMap = OrderedDict()
    for name, spec in extra_fields.items():
        table: Optional[str]
        column: Optional[str]
        if isinstance(spec, (tuple, list)):
            if len(spec) != 2:
                raise ValueError(
                    f"Extra field '{name}' must be a (table, column) pair; received {spec!r}."
                )
            table, column = spec  # type: ignore[assignment]
        elif isinstance(spec, Mapping):
            table = spec.get("table") or spec.get("table_name")
            column = spec.get("column") or spec.get("column_name")
        else:
            raise TypeError(
                f"Unsupported specification {spec!r} for extra field '{name}'."
            )
        if not column:
            raise ValueError(
                f"Column name must be provided for extra field '{name}'."
            )
        table_name = table or default_table
        if not table_name:
            raise ValueError(
                f"Unable to determine table for extra field '{name}'. "
                "Specify a table explicitly or configure meta_table."
            )
        result[name] = (table_name, column)
    return result


def build_field_map(
    config: ClusterDBConfig,
    *,
    include_defaults: bool = True,
    extra_fields: Optional[Mapping[str, ExtraFieldSpec]] = None,
) -> FieldMap:
    field_map: FieldMap = OrderedDict()
    base_table = _determine_base_table(config)

    def add_field(name: str, column: Optional[str], table: Optional[str]) -> None:
        if not column:
            return
        actual_table = table or base_table
        if not actual_table:
            raise ValueError(
                f"Cannot determine table for field '{name}'. "
                "Set meta_table or provide explicit table names."
            )
        field_map[name] = (actual_table, column)

    if include_defaults:
        add_field("uid", config.uid_col, config.meta_table)
        add_field("title", config.title_col, config.title_table)
        add_field("abstract", config.abstract_col, config.abstract_table or config.title_table)
        add_field("pubyear", config.year_col, config.year_table)
        add_field(
            "citation_count",
            config.citation_col,
            config.citation_table or config.metric_table,
        )

    if extra_fields:
        field_map.update(_normalize_extra_fields(extra_fields, base_table))

    if "uid" not in field_map:
        add_field("uid", config.uid_col, config.meta_table)

    return field_map


def _build_select_query(
    config: ClusterDBConfig,
    fields: FieldMap,
    placeholders: Optional[str],
    *,
    use_temp_table: bool = False,
    orderings: Optional[Sequence[Tuple[str, bool]]] = None,
    limit: Optional[int] = None,
) -> Tuple[str, List[str]]:
    if not fields:
        raise ValueError("Field map must include at least one column.")
    if "uid" not in fields:
        raise ValueError("Field map must include the 'uid' column.")

    base_table = fields["uid"][0] or _determine_base_table(config, fields)
    if not base_table:
        raise ValueError("Unable to determine base table for SELECT.")

    alias_map: Dict[str, str] = {base_table: "base"}
    joins: List[str] = []
    column_order: List[str] = []
    alias_counter = 1

    def ensure_alias(table: str, prefix: str) -> str:
        nonlocal alias_counter
        if table == base_table:
            return "base"
        if table in alias_map or (
            use_temp_table and table == (config.temp_table or "")
        ):
            return alias_map[table]
        alias = f"{prefix}{alias_counter}"
        alias_counter += 1
        alias_map[table] = alias
        joins.append(
            f"LEFT JOIN {table} {alias} ON {alias}.{config.uid_col} = base.{config.uid_col}"
        )
        return alias

    select_clauses: List[str] = []
    for name, (table, column) in fields.items():
        actual_table = table or base_table
        if not actual_table:
            continue
        alias = "base" if actual_table == base_table else ensure_alias(actual_table, "t")
        select_clauses.append(f"  {alias}.{column} AS {name}")
        column_order.append(name)

    if not select_clauses:
        raise ValueError("No valid columns available for SELECT statement.")
    query = ["SELECT", ",\n".join(select_clauses)]
    if use_temp_table and config.temp_table:
        if "." in config.temp_table:
            schema, table = config.temp_table.split(".", 1)
            temp_from = f"`{schema}`.`{table}`"
        else:
            temp_from = config.temp_table
        query.append(
            f"FROM {temp_from} temp JOIN {base_table} base ON temp.{config.uid_col} = base.{config.uid_col}"
        )
    else:
        query.append(f"FROM {base_table} base")
    query.extend(joins)
    if not use_temp_table:
        if placeholders is None:
            raise ValueError("Placeholders must be provided when not using a temp table")
        query.append(f"WHERE base.{config.uid_col} IN ({placeholders})")
    if orderings:
        order_clauses = [
            f"{column} {'DESC' if descending else 'ASC'}"
            for column, descending in orderings
        ]
        query.append("ORDER BY " + ", ".join(order_clauses))
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be a positive integer")
        query.append(f"LIMIT {int(limit)}")
    return "\n".join(query), column_order


def _normalize_order_by(
    order_by: Optional[
        Union[
            str,
            Sequence[str],
            Sequence[Tuple[str, Union[bool, str]]],
        ]
    ],
    descending: bool,
    valid_columns: Sequence[str],
) -> Optional[List[Tuple[str, bool]]]:
    if order_by is None:
        return None

    valid = set(valid_columns)
    if isinstance(order_by, str):
        candidates: Sequence[Union[str, Tuple[str, Union[bool, str]]]] = [order_by]
    else:
        candidates = list(order_by)

    normalised: List[Tuple[str, bool]] = []
    for item in candidates:
        if isinstance(item, str):
            column = item
            direction = descending
        else:
            if len(item) != 2:
                raise ValueError(
                    "Each order_by entry must be a (column, direction) pair."
                )
            column, direction_value = item
            if isinstance(direction_value, bool):
                direction = direction_value
            elif isinstance(direction_value, str):
                lowered = direction_value.lower()
                if lowered in {"desc", "descending"}:
                    direction = True
                elif lowered in {"asc", "ascending"}:
                    direction = False
                else:
                    raise ValueError(
                        f"Unrecognised sort direction '{direction_value}' for column '{column}'."
                    )
            else:
                raise TypeError(
                    f"Sort direction for column '{column}' must be a bool or string."
                )

        if column not in valid:
            raise ValueError(
                f"Cannot order by '{column}'. Available columns: {sorted(valid)}."
            )
        normalised.append((column, direction))

    return normalised


def _fetch_records_sqlite(
    config: ClusterDBConfig,
    uids: Sequence[str],
    fields: FieldMap,
    orderings: Optional[Sequence[Tuple[str, bool]]] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Optional[str]]]:
    if sqlite3 is None:
        raise ImportError("sqlite3 is not available in this environment.")
    if not config.db_path:
        raise ValueError("CLUSTER_DB_PATH must be set for SQLite driver")
    if not uids:
        return []

    placeholders = ",".join("?" for _ in uids)
    sql, columns = _build_select_query(
        config,
        fields,
        placeholders,
        orderings=orderings,
        limit=limit,
    )

    with sqlite3.connect(config.db_path) as conn:
        rows = conn.execute(sql, list(uids)).fetchall()

    return [dict(zip(columns, row)) for row in rows]


def fetch_records(
    config: ClusterDBConfig,
    uids: Sequence[str],
    *,
    fields: Optional[Mapping[str, FieldSpec]] = None,
    extra_fields: Optional[Mapping[str, ExtraFieldSpec]] = None,
    include_defaults: bool = True,
    order_by: Optional[
        Union[
            str,
            Sequence[str],
            Sequence[Tuple[str, Union[bool, str]]],
        ]
    ] = None,
    descending: bool = True,
    limit: Optional[int] = None,
    chunk_size: int = 1000,
) -> List[Dict[str, Optional[str]]]:
    """Fetch raw metadata for the given UIDs.

    Currently supports SQLite via ``db_path``. Extend or override as needed for
    other backends (e.g., PostgreSQL via SQLAlchemy).
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if not uids:
        return []

    if fields is None and include_defaults:
        field_map = build_field_map(
            config,
            include_defaults=True,
            extra_fields=extra_fields,
        )
    else:
        field_map = OrderedDict(fields.items()) if fields else OrderedDict()
        if include_defaults:
            defaults = build_field_map(config, include_defaults=True)
            for name, spec in defaults.items():
                field_map.setdefault(name, spec)
        if extra_fields:
            default_table = (
                field_map.get("uid", (None, None))[0]
                or _determine_base_table(config, field_map)
            )
            field_map.update(_normalize_extra_fields(extra_fields, default_table))
    uid_spec = field_map.get("uid")
    if uid_spec is None:
        uid_table = _determine_base_table(config, field_map)
        uid_spec = (uid_table, config.uid_col)
    elif not uid_spec[0]:
        uid_table = _determine_base_table(config, field_map)
        uid_spec = (uid_table, uid_spec[1])
    field_map = OrderedDict([("uid", uid_spec), *[(k, v) for k, v in field_map.items() if k != "uid"]])

    orderings = _normalize_order_by(order_by, descending, list(field_map.keys()))

    driver = (config.driver or "mysql").lower()

    if driver == "sqlite":
        return _fetch_records_sqlite(
            config,
            uids,
            field_map,
            orderings=orderings,
            limit=limit,
        )

    if driver in {"mysql", "mariadb"}:
        if pymysql is None:
            raise ImportError(
                "pymysql is required for MySQL connections; install it or switch driver"
            )
        use_temp = bool(config.temp_table)
        placeholders = ",".join(["%s"] * len(uids)) if not use_temp else None
        sql, columns = _build_select_query(
            config,
            field_map,
            placeholders,
            use_temp_table=use_temp,
            orderings=orderings,
            limit=limit,
        )

        conn = pymysql.connect(
            host=config.host,
            user=config.user,
            password=config.password,
            port=config.port or 3306,
            db=config.db_name,
            charset="utf8mb4",
        )
        try:
            with conn.cursor() as cursor:
                temp_qualified = None
                temp_is_persistent = False
                try:
                    if use_temp and config.temp_table:
                        if "." in config.temp_table:
                            schema, table = config.temp_table.split(".", 1)
                            temp_qualified = f"`{schema}`.`{table}`"
                            temp_is_persistent = True
                        else:
                            temp_qualified = f"`{config.temp_table}`"

                        drop_stmt = (
                            f"DROP TABLE IF EXISTS {temp_qualified}"
                            if temp_is_persistent
                            else f"DROP TEMPORARY TABLE IF EXISTS {temp_qualified}"
                        )
                        cursor.execute(drop_stmt)

                        create_stmt = (
                            f"CREATE TABLE {temp_qualified} ("
                            f"  {config.uid_col} VARCHAR(255) PRIMARY KEY"
                            ") ENGINE=MEMORY"
                            if temp_is_persistent
                            else f"CREATE TEMPORARY TABLE {temp_qualified} ("
                            f"  {config.uid_col} VARCHAR(255) PRIMARY KEY"
                            ") ENGINE=MEMORY"
                        )
                        cursor.execute(create_stmt)

                        values = [(uid,) for uid in uids]
                        for idx in range(0, len(values), chunk_size):
                            cursor.executemany(
                                f"INSERT INTO {temp_qualified} ({config.uid_col}) VALUES (%s)",
                                values[idx : idx + chunk_size],
                            )

                    if use_temp:
                        cursor.execute(sql)
                    else:
                        cursor.execute(sql, list(uids))
                    rows = cursor.fetchall()
                finally:
                    if use_temp and temp_qualified:
                        drop_stmt = (
                            f"DROP TABLE IF EXISTS {temp_qualified}"
                            if temp_is_persistent
                            else f"DROP TEMPORARY TABLE IF EXISTS {temp_qualified}"
                        )
                        cursor.execute(drop_stmt)
        finally:
            conn.close()

        return [dict(zip(columns, row)) for row in rows]

    raise NotImplementedError(f"Unsupported database driver: {config.driver}")


class UIDDataExtractor:
    """High-level helper for fetching publication metadata by UID."""

    def __init__(self, config: Optional[ClusterDBConfig] = None) -> None:
        self.config = config or load_db_config()

    def build_field_map(
        self,
        *,
        include_defaults: bool = True,
        extra_fields: Optional[Mapping[str, ExtraFieldSpec]] = None,
    ) -> FieldMap:
        return build_field_map(
            self.config,
            include_defaults=include_defaults,
            extra_fields=extra_fields,
        )

    def fetch(
        self,
        uids: Sequence[str],
        *,
        fields: Optional[Mapping[str, FieldSpec]] = None,
        extra_fields: Optional[Mapping[str, ExtraFieldSpec]] = None,
        include_defaults: bool = True,
        order_by: Optional[
            Union[
                str,
                Sequence[str],
                Sequence[Tuple[str, Union[bool, str]]],
            ]
        ] = None,
        descending: bool = True,
        limit: Optional[int] = None,
        chunk_size: int = 1000,
        as_dataframe: bool = False,
    ) -> Union[List[Dict[str, Optional[str]]], "pd.DataFrame"]:
        records = fetch_records(
            self.config,
            uids,
            fields=fields,
            extra_fields=extra_fields,
            include_defaults=include_defaults,
            order_by=order_by,
            descending=descending,
            limit=limit,
            chunk_size=chunk_size,
        )
        if not as_dataframe:
            return records

        try:
            import pandas as pd  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "pandas is required when as_dataframe=True. "
                "Install pandas or set as_dataframe=False."
            ) from exc
        return pd.DataFrame(records)


def _to_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value: Optional[str]) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def select_core_documents(
    records: Sequence[Dict[str, Optional[str]]],
    *,
    top_k: int = 5,
    top_n_per_year: int = 3,
    recent_years: int = 10,
    max_documents: Optional[int] = None,
) -> List[ClusterDocument]:
    """Select representative documents using top-K and per-year sampling."""

    cleaned = []
    for rec in records:
        title = (rec.get("title") or "").strip()
        abstract = (rec.get("abstract") or "").strip()
        if not title or not abstract:
            continue
        cleaned.append(
            {
                "uid": rec.get("uid"),
                "title": title,
                "abstract": abstract,
                "pubyear": _to_int(rec.get("pubyear")),
                "citation": _to_float(rec.get("citation_count")),
            }
        )

    if not cleaned:
        return []

    key_citation = lambda r: (r["citation"], r["pubyear"] or 0)
    sorted_by_citation = sorted(cleaned, key=key_citation, reverse=True)

    selected: List[Dict[str, Optional[str]]] = []
    seen = set()

    for rec in sorted_by_citation:
        uid = rec.get("uid")
        if not uid or uid in seen:
            continue
        selected.append(rec)
        seen.add(uid)
        if len(selected) >= top_k:
            break

    valid_years = [r["pubyear"] for r in cleaned if r["pubyear"]]
    latest_year = max(valid_years) if valid_years else datetime.utcnow().year
    start_year = latest_year - recent_years + 1

    for year in range(latest_year, start_year - 1, -1):
        year_records = [r for r in cleaned if r["pubyear"] == year]
        year_records.sort(key=key_citation, reverse=True)
        for rec in year_records[:top_n_per_year]:
            uid = rec.get("uid")
            if not uid or uid in seen:
                continue
            selected.append(rec)
            seen.add(uid)
            if max_documents and len(selected) >= max_documents:
                break
        if max_documents and len(selected) >= max_documents:
            break

    if max_documents:
        selected = selected[:max_documents]

    return [
        ClusterDocument(
            uid=str(rec.get("uid")),
            title=str(rec.get("title")),
            abstract=str(rec.get("abstract")),
            pubyear=rec.get("pubyear"),
            citation_count=rec.get("citation"),
        )
        for rec in selected
    ]


def build_core_documents(
    uids: Sequence[str],
    *,
    top_k: int = 5,
    top_n_per_year: int = 3,
    recent_years: int = 10,
    max_documents: Optional[int] = None,
    config: Optional[ClusterDBConfig] = None,
) -> List[ClusterDocument]:
    """Fetch and select core documents for the given UID list."""

    if config is None:
        config = load_db_config()

    records = fetch_records(config, uids)
    return select_core_documents(
        records,
        top_k=top_k,
        top_n_per_year=top_n_per_year,
        recent_years=recent_years,
        max_documents=max_documents,
    )


__all__ = [
    "ClusterDocument",
    "ClusterDBConfig",
    "load_db_config",
    "build_field_map",
    "fetch_records",
    "UIDDataExtractor",
    "select_core_documents",
    "build_core_documents",
]
