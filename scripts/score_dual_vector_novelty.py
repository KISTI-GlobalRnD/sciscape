#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _require_h5py():
    try:
        import h5py  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "Missing dependency: h5py.\n"
            "Recommended: `uv pip install --python .venv/bin/python h5py`\n"
        ) from exc
    return h5py


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected top-level object in {path}")
    return payload


def _normalize_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _normalize_group(value: object) -> str:
    return _normalize_text(value)


def _normalize_year(value: object) -> int | None:
    text = _normalize_text(value)
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _iter_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                text = line.strip()
                if not text:
                    continue
                obj = json.loads(text)
                if not isinstance(obj, dict):
                    raise SystemExit(f"Expected object per line in {path} at line {line_no}")
                rows.append(obj)
        return rows
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise SystemExit(f"Expected top-level list in {path}")
        return [dict(row) for row in payload]
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open("r", encoding="utf-8", newline="") as f:
            return [dict(row) for row in csv.DictReader(f, delimiter=delimiter)]
    raise SystemExit(f"Unsupported input format: {path.suffix} (use .jsonl, .json, .csv, or .tsv)")


def _join_nonempty(parts: list[str]) -> str:
    return " ".join(part for part in parts if part)


def _is_finite_matrix(matrix: np.ndarray) -> bool:
    return bool(np.isfinite(matrix).all())


@dataclass(frozen=True)
class RoleRecord:
    work_id: str
    title: str
    abstract: str
    background_summary: str
    novelty_claim: str
    method_core: str
    main_findings: str
    contribution_type: str
    year: int | None
    group: str
    background_text: str
    novelty_text: str
    background_source: str
    novelty_source: str


def _build_role_record(row: dict[str, Any], args: argparse.Namespace) -> RoleRecord:
    work_id = _normalize_text(row.get(args.id_field))
    if not work_id:
        raise SystemExit(f"Missing `{args.id_field}` in input row: {row}")

    title = _normalize_text(row.get(args.title_field))
    abstract = _normalize_text(row.get(args.abstract_field))
    background_summary = _normalize_text(row.get(args.background_field))
    novelty_claim = _normalize_text(row.get(args.novelty_field))
    method_core = _normalize_text(row.get(args.method_field))
    main_findings = _normalize_text(row.get(args.findings_field))
    contribution_type = _normalize_text(row.get(args.contribution_type_field))
    year = _normalize_year(row.get(args.year_field)) if args.year_field else None
    group = _normalize_group(row.get(args.group_field)) if args.group_field else ""

    background_source = "structured"
    novelty_source = "structured"

    if background_summary:
        background_text = _join_nonempty([title, background_summary])
    else:
        background_text = _join_nonempty([title, abstract])
        background_source = "abstract_fallback" if abstract else "title_only"

    novelty_parts = [novelty_claim, method_core, main_findings]
    if any(novelty_parts):
        novelty_text = _join_nonempty([title] + novelty_parts)
    else:
        novelty_text = _join_nonempty([title, abstract])
        novelty_source = "abstract_fallback" if abstract else "title_only"

    return RoleRecord(
        work_id=work_id,
        title=title,
        abstract=abstract,
        background_summary=background_summary,
        novelty_claim=novelty_claim,
        method_core=method_core,
        main_findings=main_findings,
        contribution_type=contribution_type,
        year=year,
        group=group,
        background_text=background_text,
        novelty_text=novelty_text,
        background_source=background_source,
        novelty_source=novelty_source,
    )


def _write_jsonl(path: Path, rows: list[RoleRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(asdict(row), ensure_ascii=True) + "\n")


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim != 2:
        raise SystemExit(f"Expected 2D embedding matrix, got shape {matrix.shape}")
    if not _is_finite_matrix(matrix):
        raise SystemExit("Embedding matrix contains NaN or inf values")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    return matrix / norms


def _write_h5(path: Path, work_ids: list[str], matrix: np.ndarray) -> None:
    h5py = _require_h5py()
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("work_ids", data=np.asarray(work_ids, dtype="S"))
        f.create_dataset("embeddings", data=matrix.astype(np.float32))


def _load_h5_aligned(path: Path, work_ids: list[str]) -> np.ndarray:
    h5py = _require_h5py()
    with h5py.File(path, "r") as f:
        raw_ids = f["work_ids"][:]
        emb = np.asarray(f["embeddings"][:], dtype=np.float32)
    if emb.ndim != 2:
        raise SystemExit(f"Expected 2D `embeddings` dataset in {path}, got shape {emb.shape}")
    if int(emb.shape[0]) != len(raw_ids):
        raise SystemExit(
            f"Mismatched H5 row counts in {path}: work_ids={len(raw_ids)} embeddings={int(emb.shape[0])}"
        )
    if not _is_finite_matrix(emb):
        raise SystemExit(f"Embedding matrix in {path} contains NaN or inf values")

    id_map: dict[str, int] = {}
    duplicate_ids: list[str] = []
    for idx, value in enumerate(raw_ids):
        work_id = value.decode("utf-8") if isinstance(value, bytes) else str(value)
        if work_id in id_map:
            duplicate_ids.append(work_id)
            continue
        id_map[work_id] = idx
    if duplicate_ids:
        preview = ", ".join(sorted(set(duplicate_ids))[:10])
        raise SystemExit(f"Duplicate work_ids found in {path}: {preview}")

    missing = [work_id for work_id in work_ids if work_id not in id_map]
    if missing:
        preview = ", ".join(missing[:10])
        raise SystemExit(f"Missing {len(missing)} work_ids in {path}: {preview}")
    rows = [emb[id_map[work_id]] for work_id in work_ids]
    return _l2_normalize(np.asarray(rows, dtype=np.float32))


def _encode_role_texts(records: list[RoleRecord], args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    from nanoclustering.specter2.adapter_embeddings import add_ctrl_token
    from nanoclustering.specter2.adapter_embeddings import encode_texts
    from nanoclustering.specter2.adapter_embeddings import load_adapter_model
    from nanoclustering.specter2.adapter_embeddings import parse_pooling

    loaded = load_adapter_model(
        args.base_model,
        adapter=args.adapter,
        adapter_name=str(args.adapter_name),
        adapter_source=str(args.adapter_source),
        ctrl_tokens=[str(args.ctrl_token)] if args.ctrl_token else None,
        postprocess_artifact=args.postprocess_artifact,
        device=str(args.device),
    )
    background_texts = add_ctrl_token([row.background_text for row in records], args.ctrl_token)
    novelty_texts = add_ctrl_token([row.novelty_text for row in records], args.ctrl_token)
    background = encode_texts(
        loaded,
        background_texts,
        max_length=int(args.max_length),
        batch_size=int(args.batch_size),
        pooling=parse_pooling(str(args.pooling)),
        token_idx=int(args.token_idx),
        normalize=True,
    ).numpy()
    novelty = encode_texts(
        loaded,
        novelty_texts,
        max_length=int(args.max_length),
        batch_size=int(args.batch_size),
        pooling=parse_pooling(str(args.pooling)),
        token_idx=int(args.token_idx),
        normalize=True,
    ).numpy()
    return background.astype(np.float32), novelty.astype(np.float32)


def _build_group_lookup(records: list[RoleRecord]) -> dict[str, np.ndarray]:
    groups: dict[str, list[int]] = {}
    for idx, row in enumerate(records):
        if not row.group:
            continue
        groups.setdefault(row.group, []).append(idx)
    return {group: np.asarray(indices, dtype=np.int32) for group, indices in groups.items()}


def _candidate_indices(
    idx: int,
    query: RoleRecord,
    all_indices: np.ndarray,
    group_lookup: dict[str, np.ndarray],
    year_values: np.ndarray,
    year_present: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, str]:
    if args.group_field:
        if not query.group:
            return np.empty(0, dtype=np.int32), "missing_query_group"
        candidate_idx = group_lookup.get(query.group, np.empty(0, dtype=np.int32))
    else:
        candidate_idx = all_indices

    if candidate_idx.size:
        candidate_idx = candidate_idx[candidate_idx != idx]

    if args.year_window is not None:
        if query.year is None:
            return np.empty(0, dtype=np.int32), "missing_query_year"
        if candidate_idx.size:
            valid_idx = candidate_idx[year_present[candidate_idx]]
            if valid_idx.size:
                year_delta = np.abs(year_values[valid_idx] - int(query.year))
                candidate_idx = valid_idx[year_delta <= int(args.year_window)]
            else:
                candidate_idx = valid_idx

    return candidate_idx, "ok"


def _summarize_records(records: list[RoleRecord]) -> dict[str, int]:
    return {
        "rows": len(records),
        "background_structured": sum(1 for row in records if row.background_source == "structured"),
        "background_abstract_fallback": sum(1 for row in records if row.background_source == "abstract_fallback"),
        "background_title_only": sum(1 for row in records if row.background_source == "title_only"),
        "novelty_structured": sum(1 for row in records if row.novelty_source == "structured"),
        "novelty_abstract_fallback": sum(1 for row in records if row.novelty_source == "abstract_fallback"),
        "novelty_title_only": sum(1 for row in records if row.novelty_source == "title_only"),
    }


def _safe_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.mean(values))


def _resolve_config_path(config_arg: str) -> Path | None:
    text = str(config_arg).strip()
    if not text:
        return None
    return Path(text).resolve()


def _apply_config_defaults(args: argparse.Namespace) -> dict[str, Any]:
    config_path = _resolve_config_path(str(args.config))
    if config_path is None:
        return {}

    config = _read_json(config_path)
    cohort_cfg = config.get("cohort", {})
    embedding_cfg = config.get("embedding", {})
    shared_encoder_defaults = embedding_cfg.get("shared_encoder_defaults", {})

    if int(args.cohort_k) == 20 and "cohort_k" in cohort_cfg:
        args.cohort_k = int(cohort_cfg["cohort_k"])
    if int(args.score_top_m) == 5 and "score_top_m" in cohort_cfg:
        args.score_top_m = int(cohort_cfg["score_top_m"])
    if str(args.pooling) == "cls" and "pooling" in shared_encoder_defaults:
        args.pooling = str(shared_encoder_defaults["pooling"])

    use_external = bool(args.background_h5) or bool(args.novelty_h5)
    if not use_external and not str(args.base_model).strip():
        base_model = _normalize_text(shared_encoder_defaults.get("base_model"))
        if base_model:
            args.base_model = base_model

    return config


def _build_scores(
    records: list[RoleRecord],
    background: np.ndarray,
    novelty: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    background = _l2_normalize(background.astype(np.float32))
    novelty = _l2_normalize(novelty.astype(np.float32))

    cohort_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    requested_top_m = int(args.score_top_m)
    all_indices = np.arange(len(records), dtype=np.int32)
    group_lookup = _build_group_lookup(records) if args.group_field else {}
    year_values = np.asarray([int(row.year) if row.year is not None else 0 for row in records], dtype=np.int32)
    year_present = np.asarray([row.year is not None for row in records], dtype=bool)

    for idx, query in enumerate(records):
        candidate_idx, cohort_reason = _candidate_indices(
            idx,
            query,
            all_indices,
            group_lookup,
            year_values,
            year_present,
            args,
        )
        if candidate_idx.size == 0:
            score_rows.append(
                {
                    "work_id": query.work_id,
                    "title": query.title,
                    "year": query.year,
                    "group": query.group,
                    "cohort_size": 0,
                    "score_top_m_requested": requested_top_m,
                    "score_top_m_used": 0,
                    "background_affinity": 0.0,
                    "novelty_redundancy": 0.0,
                    "novelty_nn_similarity": 0.0,
                    "novelty_score": 0.0,
                    "cohort_reason": cohort_reason if cohort_reason != "ok" else "no_candidates_after_filters",
                    "background_source": query.background_source,
                    "novelty_source": query.novelty_source,
                }
            )
            continue

        query_background = background[idx]
        candidate_bg = background[candidate_idx] @ query_background
        order = np.argsort(-candidate_bg)
        top_pos = order[: int(args.cohort_k)]
        top_k = candidate_idx[top_pos]
        top_bg = candidate_bg[top_pos]
        query_novelty = novelty[idx]
        top_nov = novelty[top_k] @ query_novelty
        novelty_order = np.argsort(-top_nov)
        top_m = min(requested_top_m, int(top_k.size))
        novelty_top_m = top_nov[novelty_order[:top_m]]
        background_top_m = top_bg[:top_m]

        for rank, neighbor_idx in enumerate(top_k, start=1):
            neighbor = records[int(neighbor_idx)]
            year_gap = None
            if query.year is not None and neighbor.year is not None:
                year_gap = abs(int(query.year) - int(neighbor.year))
            cohort_rows.append(
                {
                    "query_work_id": query.work_id,
                    "query_title": query.title,
                    "query_year": query.year,
                    "query_group": query.group,
                    "neighbor_rank_by_background": rank,
                    "neighbor_work_id": neighbor.work_id,
                    "neighbor_title": neighbor.title,
                    "neighbor_year": neighbor.year,
                    "neighbor_group": neighbor.group,
                    "background_similarity": float(top_bg[rank - 1]),
                    "novelty_similarity": float(top_nov[rank - 1]),
                    "year_gap": year_gap,
                }
            )

        background_affinity = _safe_mean(background_top_m)
        novelty_redundancy = _safe_mean(novelty_top_m)
        novelty_nn_similarity = float(np.max(top_nov)) if top_nov.size else 0.0
        novelty_score = background_affinity * (1.0 - novelty_redundancy)

        score_rows.append(
            {
                "work_id": query.work_id,
                "title": query.title,
                    "year": query.year,
                    "group": query.group,
                    "cohort_size": int(top_k.size),
                    "score_top_m_requested": requested_top_m,
                    "score_top_m_used": top_m,
                    "background_affinity": background_affinity,
                    "novelty_redundancy": novelty_redundancy,
                    "novelty_nn_similarity": novelty_nn_similarity,
                    "novelty_score": novelty_score,
                    "cohort_reason": "ok",
                    "background_source": query.background_source,
                    "novelty_source": query.novelty_source,
                }
            )

    score_rows.sort(key=lambda row: (-float(row["novelty_score"]), -float(row["background_affinity"]), str(row["work_id"])))
    return cohort_rows, score_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_review_packet(
    path: Path,
    records: list[RoleRecord],
    cohort_rows: list[dict[str, Any]],
    score_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    role_counts = _summarize_records(records)
    top_rows = score_rows[: min(10, len(score_rows))]
    bottom_rows = list(reversed(score_rows[-min(5, len(score_rows)) :])) if score_rows else []
    cohort_map: dict[str, list[dict[str, Any]]] = {}
    for row in cohort_rows:
        cohort_map.setdefault(str(row["query_work_id"]), []).append(row)

    lines = [
        "# Dual-Vector Novelty Review Packet",
        "",
        f"- rows: `{len(records)}`",
        f"- cohort_k: `{int(args.cohort_k)}`",
        f"- score_top_m_requested: `{int(args.score_top_m)}`",
        f"- group_field: `{args.group_field or 'none'}`",
        f"- year_field: `{args.year_field or 'none'}`",
        f"- year_window: `{args.year_window if args.year_window is not None else 'none'}`",
        "- note: `novelty_score` is a ranking signal and is not guaranteed to stay in the `[0, 1]` range.",
        "",
        "## Coverage",
        "",
        f"- background structured: `{role_counts['background_structured']}`",
        f"- background abstract fallback: `{role_counts['background_abstract_fallback']}`",
        f"- background title only: `{role_counts['background_title_only']}`",
        f"- novelty structured: `{role_counts['novelty_structured']}`",
        f"- novelty abstract fallback: `{role_counts['novelty_abstract_fallback']}`",
        f"- novelty title only: `{role_counts['novelty_title_only']}`",
        "",
        "## Highest novelty scores",
        "",
    ]

    for row in top_rows:
        lines.append(
            f"- `{row['work_id']}`: novelty_score=`{float(row['novelty_score']):.4f}`, "
            f"background_affinity=`{float(row['background_affinity']):.4f}`, "
            f"novelty_redundancy=`{float(row['novelty_redundancy']):.4f}`, "
            f"cohort_reason=`{row['cohort_reason']}`"
        )
        if row["title"]:
            lines.append(f"  title: {row['title']}")
        for neighbor in cohort_map.get(str(row["work_id"]), [])[:3]:
            lines.append(
                f"  cohort#{neighbor['neighbor_rank_by_background']}: "
                f"`{neighbor['neighbor_work_id']}` "
                f"(bg=`{float(neighbor['background_similarity']):.4f}`, "
                f"nov=`{float(neighbor['novelty_similarity']):.4f}`)"
            )

    if bottom_rows:
        lines.extend(["", "## Lowest novelty scores", ""])
        for row in bottom_rows:
            lines.append(
                f"- `{row['work_id']}`: novelty_score=`{float(row['novelty_score']):.4f}`, "
                f"background_affinity=`{float(row['background_affinity']):.4f}`, "
                f"novelty_redundancy=`{float(row['novelty_redundancy']):.4f}`, "
                f"cohort_reason=`{row['cohort_reason']}`"
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Pilot scorer for cohort-relative dual-vector novelty.")
    ap.add_argument("--input", required=True, help="Input role-field file (.jsonl/.json/.csv/.tsv).")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--config", default="")

    ap.add_argument("--id-field", default="work_id")
    ap.add_argument("--title-field", default="title")
    ap.add_argument("--abstract-field", default="abstract")
    ap.add_argument("--background-field", default="background_summary")
    ap.add_argument("--novelty-field", default="novelty_claim")
    ap.add_argument("--method-field", default="method_core")
    ap.add_argument("--findings-field", default="main_findings")
    ap.add_argument("--contribution-type-field", default="contribution_type")
    ap.add_argument("--year-field", default="year")
    ap.add_argument("--group-field", default="")

    ap.add_argument("--background-h5", default="")
    ap.add_argument("--novelty-h5", default="")
    ap.add_argument("--base-model", default="")
    ap.add_argument("--adapter", default="")
    ap.add_argument("--adapter-name", default="[PRX]")
    ap.add_argument("--adapter-source", default="auto")
    ap.add_argument("--ctrl-token", default="")
    ap.add_argument("--postprocess-artifact", default="")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--pooling", default="cls")
    ap.add_argument("--token-idx", type=int, default=0)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=32)

    ap.add_argument("--cohort-k", type=int, default=20)
    ap.add_argument("--score-top-m", type=int, default=5)
    ap.add_argument("--year-window", type=int, default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    config = _apply_config_defaults(args)
    if int(args.cohort_k) <= 0:
        raise SystemExit("--cohort-k must be > 0")
    if int(args.score_top_m) <= 0:
        raise SystemExit("--score-top-m must be > 0")

    input_path = Path(args.input).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = _iter_rows(input_path)
    records = [_build_role_record(row, args) for row in raw_rows]
    if not records:
        raise SystemExit(f"No rows found in {input_path}")

    seen: set[str] = set()
    for row in records:
        if row.work_id in seen:
            raise SystemExit(f"Duplicate work_id found: {row.work_id}")
        seen.add(row.work_id)

    normalized_jsonl = out_dir / "paper_role_fields.jsonl"
    _write_jsonl(normalized_jsonl, records)

    work_ids = [row.work_id for row in records]
    use_external = bool(args.background_h5) or bool(args.novelty_h5)
    use_encoder = bool(args.base_model)
    if use_external and use_encoder:
        raise SystemExit("Choose either external H5 inputs or live encoding, not both")
    if use_external:
        if not args.background_h5 or not args.novelty_h5:
            raise SystemExit("Both --background-h5 and --novelty-h5 are required together")
        background = _load_h5_aligned(Path(args.background_h5).resolve(), work_ids)
        novelty = _load_h5_aligned(Path(args.novelty_h5).resolve(), work_ids)
    elif use_encoder:
        background, novelty = _encode_role_texts(records, args)
        _write_h5(out_dir / "background_embeddings.h5", work_ids, background)
        _write_h5(out_dir / "novelty_embeddings.h5", work_ids, novelty)
    else:
        raise SystemExit("Provide either --background-h5/--novelty-h5 or --base-model")

    if not use_encoder:
        _write_h5(out_dir / "background_embeddings.h5", work_ids, background)
        _write_h5(out_dir / "novelty_embeddings.h5", work_ids, novelty)

    cohort_rows, score_rows = _build_scores(records, background, novelty, args)
    _write_csv(out_dir / "background_cohorts.csv", cohort_rows)
    _write_csv(out_dir / "novelty_scores.csv", score_rows)
    _write_review_packet(out_dir / "novelty_review_packet.md", records, cohort_rows, score_rows, args)

    summary = {
        "ok": True,
        "inputs": {
            "input": str(input_path),
            "config": str(_resolve_config_path(str(args.config))) if str(args.config).strip() else None,
            "rows": len(records),
            "group_field": args.group_field or None,
            "year_field": args.year_field or None,
            "year_window": args.year_window,
        },
        "embedding_mode": "external_h5" if use_external else "shared_encoder",
        "contract_version": config.get("version") if isinstance(config, dict) else None,
        "outputs": {
            "paper_role_fields_jsonl": str(normalized_jsonl),
            "background_embeddings_h5": str(out_dir / "background_embeddings.h5"),
            "novelty_embeddings_h5": str(out_dir / "novelty_embeddings.h5"),
            "background_cohorts_csv": str(out_dir / "background_cohorts.csv"),
            "novelty_scores_csv": str(out_dir / "novelty_scores.csv"),
            "novelty_review_packet_md": str(out_dir / "novelty_review_packet.md"),
        },
        "coverage": _summarize_records(records),
    }
    (out_dir / "report.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
