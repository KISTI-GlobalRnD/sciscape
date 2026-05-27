"""Evaluate external-grain probes as pre-screen predictors for split-repair.

The evaluator treats external-grain output as pre-apply diagnostics and joins it
to full split-repair probe output to build cluster-level labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


STRONG_SUCCESS_THRESHOLD = 1.0
PRACTICAL_SUCCESS_THRESHOLD = 1e-6

DEFAULT_GROUP_FRACTION_THRESHOLDS = (0.001, 0.005, 0.01)
DEFAULT_INCIDENT_BUDGETS = (1000, 2500, 5000, 10000, 25000, 50000, 100000)
DEFAULT_CASCADE_MAX_INCIDENT_EDGES = 50000

SUMMARY_CSV_FIELDS = [
    "scope",
    "sample",
    "regime",
    "gamma",
    "rule",
    "label",
    "total_count",
    "labeled_count",
    "positive_count",
    "selected_count",
    "selected_fraction",
    "tp",
    "fp",
    "tn",
    "fn",
    "precision",
    "recall",
    "f1",
    "estimated_full_repair_cost_units",
    "selected_full_repair_cost_units",
    "estimated_saved_full_repair_cost_units",
    "estimated_saved_full_repair_cost_fraction",
    "external_probe_elapsed_sec",
    "split_repair_probe_elapsed_sec",
    "split_to_external_elapsed_ratio",
]

EXAMPLE_FIELDS = [
    "sample",
    "cluster",
    "doc_weight",
    "block_count",
    "incident_directed_edges",
    "best_group_delta_q",
    "best_group_fraction",
    "assigned_fraction",
    "positive_group_count",
    "best_group_action",
    "best_net_delta_q",
    "best_gamma_multiplier",
    "escaped_source_weight",
    "n_parts",
    "core_part_count",
    "singleton_frac",
    "split_debt_per_doc_weight",
    "strong_success",
    "practical_success",
]


@dataclass(frozen=True)
class SampleSpec:
    name: str
    external_grain_csv: Path
    split_repair_csv: Path | None = None
    cost_only: bool = False
    gamma: float | None = None
    regime: str = ""
    candidate_policy: str = ""
    external_summary_json: Path | None = None
    split_repair_summary_json: Path | None = None


@dataclass(frozen=True)
class Rule:
    name: str
    selector: Callable[[dict[str, Any]], bool]


@dataclass(frozen=True)
class CoverageSpec:
    name: str
    external_grain_csv: Path | None = None
    split_repair_csv: Path | None = None
    cost_only: bool = False
    full_repair_forbidden: bool = False
    gamma: float | None = None
    regime: str = ""
    candidate_policy: str = ""


def _normalize_key(key: str) -> str:
    return key.strip().lower()


def _normalize_row(row: dict[str, str]) -> dict[str, str]:
    return {_normalize_key(key): value for key, value in row.items()}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return [_normalize_row(row) for row in csv.DictReader(fh)]


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(result):
        return default
    return result


def _to_int(value: Any, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _to_bool(value: Any, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _record_cost(record: dict[str, Any]) -> float:
    return max(0.0, float(record.get("incident_directed_edges", 0.0) or 0.0))


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _format_threshold(value: float) -> str:
    text = f"{value:g}"
    return text.replace(".", "p").replace("-", "m")


def _resolve_path(value: str | None, base_dir: Path) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return base_dir / path


def _load_yaml_or_json(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "YAML metadata requires PyYAML; use JSON metadata or install PyYAML."
            ) from exc
        return yaml.safe_load(text)
    return json.loads(text)


def _sample_specs_from_raw(raw_samples: Any, base_dir: Path) -> list[SampleSpec]:
    if not isinstance(raw_samples, list):
        raise ValueError("metadata must be a list or an object with a 'samples' list")

    samples: list[SampleSpec] = []
    for index, raw in enumerate(raw_samples):
        if not isinstance(raw, dict):
            raise ValueError(f"sample entry {index} must be an object")
        name = str(raw.get("name") or "").strip()
        if not name:
            raise ValueError(f"sample entry {index} is missing 'name'")
        external_csv = _resolve_path(
            raw.get("external_grain_csv") or raw.get("external_csv"),
            base_dir,
        )
        if external_csv is None:
            raise ValueError(f"sample {name!r} is missing 'external_grain_csv'")
        split_csv = _resolve_path(
            raw.get("split_repair_csv") or raw.get("split_merge_repair_csv"),
            base_dir,
        )
        samples.append(
            SampleSpec(
                name=name,
                external_grain_csv=external_csv,
                split_repair_csv=split_csv,
                cost_only=_to_bool(raw.get("cost_only"), False),
                gamma=None if raw.get("gamma") in (None, "") else _to_float(raw["gamma"]),
                regime=str(raw.get("regime") or ""),
                candidate_policy=str(raw.get("candidate_policy") or raw.get("policy") or ""),
                external_summary_json=_resolve_path(
                    raw.get("external_summary_json") or raw.get("external_summary"),
                    base_dir,
                ),
                split_repair_summary_json=_resolve_path(
                    raw.get("split_repair_summary_json") or raw.get("split_repair_summary"),
                    base_dir,
                ),
            )
        )
    return samples


def _coverage_specs_from_raw(raw_specs: Any, base_dir: Path) -> list[CoverageSpec]:
    if raw_specs is None:
        return []
    if not isinstance(raw_specs, list):
        raise ValueError("'coverage_matrix' must be a list")

    specs: list[CoverageSpec] = []
    for index, raw in enumerate(raw_specs):
        if not isinstance(raw, dict):
            raise ValueError(f"coverage entry {index} must be an object")
        name = str(raw.get("name") or "").strip()
        if not name:
            raise ValueError(f"coverage entry {index} is missing 'name'")
        specs.append(
            CoverageSpec(
                name=name,
                external_grain_csv=_resolve_path(
                    raw.get("external_grain_csv") or raw.get("external_csv"),
                    base_dir,
                ),
                split_repair_csv=_resolve_path(
                    raw.get("split_repair_csv") or raw.get("split_merge_repair_csv"),
                    base_dir,
                ),
                cost_only=_to_bool(raw.get("cost_only"), False),
                full_repair_forbidden=_to_bool(raw.get("full_repair_forbidden"), False),
                gamma=None if raw.get("gamma") in (None, "") else _to_float(raw["gamma"]),
                regime=str(raw.get("regime") or ""),
                candidate_policy=str(raw.get("candidate_policy") or raw.get("policy") or ""),
            )
        )
    return specs


def load_metadata(path: Path) -> tuple[list[SampleSpec], list[CoverageSpec]]:
    """Load validation samples and optional expected coverage matrix."""

    data = _load_yaml_or_json(path)
    raw_samples = data.get("samples", data) if isinstance(data, dict) else data
    raw_coverage = data.get("coverage_matrix") if isinstance(data, dict) else None
    samples = _sample_specs_from_raw(raw_samples, path.parent)
    coverage = _coverage_specs_from_raw(raw_coverage, path.parent)
    if not coverage:
        coverage = [
            CoverageSpec(
                name=sample.name,
                external_grain_csv=sample.external_grain_csv,
                split_repair_csv=sample.split_repair_csv,
                cost_only=sample.cost_only,
                gamma=sample.gamma,
                regime=sample.regime,
                candidate_policy=sample.candidate_policy,
            )
            for sample in samples
        ]
    return samples, coverage


def load_sample_specs(path: Path) -> list[SampleSpec]:
    """Load sample metadata from JSON or YAML."""

    samples, _ = load_metadata(path)
    return samples


def _phase_elapsed(summary_path: Path | None, names: Iterable[str]) -> float | None:
    if summary_path is None or not summary_path.exists():
        return None
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    wanted = set(names)
    for phase in data.get("phases", []):
        if phase.get("name") in wanted:
            return _to_float(phase.get("elapsed_sec"), 0.0)
    return None


def _external_record(sample: SampleSpec, row: dict[str, str]) -> dict[str, Any]:
    return {
        "sample": sample.name,
        "gamma": sample.gamma,
        "regime": sample.regime,
        "candidate_policy": sample.candidate_policy,
        "cluster": _to_int(row.get("cluster")),
        "rank": _to_int(row.get("rank")),
        "block_count": _to_int(row.get("block_count")),
        "doc_weight": _to_float(row.get("doc_weight")),
        "incident_directed_edges": _to_int(row.get("incident_directed_edges")),
        "source_directed_edges": _to_int(row.get("source_directed_edges")),
        "external_directed_edges": _to_int(row.get("external_directed_edges")),
        "assigned_fraction": _to_float(row.get("assigned_fraction")),
        "best_group_delta_q": _to_float(row.get("best_group_delta_q")),
        "best_group_fraction": _to_float(row.get("best_group_fraction")),
        "best_group_action": _to_int(row.get("best_group_action")),
        "positive_group_count": _to_int(row.get("positive_group_count")),
        "recommended_for_split_repair": _to_bool(
            row.get("recommended_for_split_repair"), False
        ),
    }


def _split_record(row: dict[str, str]) -> dict[str, Any]:
    doc_weight = _to_float(row.get("doc_weight"))
    singleton_weight = _to_float(row.get("singleton_weight"))
    split_delta_q_base = _to_float(row.get("split_delta_q_base"))
    split_debt = max(0.0, -split_delta_q_base)
    return {
        "best_gamma_multiplier": _to_float(row.get("gamma_multiplier")),
        "best_probe_resolution": _to_float(row.get("probe_resolution")),
        "best_net_delta_q": _to_float(row.get("net_delta_q")),
        "repair_delta_q": _to_float(row.get("repair_delta_q")),
        "escaped_source_weight": _to_float(row.get("escaped_source_weight")),
        "escaped_source_units": _to_int(row.get("escaped_source_units")),
        "n_parts": _to_int(row.get("n_parts")),
        "core_part_count": _to_int(row.get("core_part_count")),
        "singleton_weight": singleton_weight,
        "cut_weight": _to_float(row.get("cut_weight")),
        "split_delta_q_base": split_delta_q_base,
        "split_delta_q_probe": _to_float(row.get("split_delta_q_probe")),
        "split_debt": split_debt,
        "split_debt_per_doc_weight": split_debt / doc_weight if doc_weight else 0.0,
        "singleton_frac": singleton_weight / doc_weight if doc_weight else 0.0,
        "restored_source_cluster": _to_bool(row.get("restored_source_cluster"), False),
    }


def best_split_rows_by_cluster(rows: list[dict[str, str]]) -> dict[int, dict[str, Any]]:
    """Pick the best full split-repair row per cluster by net delta Q."""

    best: dict[int, tuple[tuple[float, float, float], dict[str, Any]]] = {}
    for row in rows:
        cluster = _to_int(row.get("cluster"))
        net_delta_q = _to_float(row.get("net_delta_q"))
        escaped = _to_float(row.get("escaped_source_weight"))
        rank = _to_float(row.get("rank"), 1e12)
        score = (net_delta_q, escaped, -rank)
        record = _split_record(row)
        current = best.get(cluster)
        if current is None or score > current[0]:
            best[cluster] = (score, record)
    return {cluster: record for cluster, (_, record) in best.items()}


def join_sample_records(sample: SampleSpec) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Join one sample's external-grain rows to optional best split-repair labels."""

    external_rows = _read_csv(sample.external_grain_csv)
    records = [_external_record(sample, row) for row in external_rows]

    split_by_cluster: dict[int, dict[str, Any]] = {}
    if sample.split_repair_csv is not None and sample.split_repair_csv.exists():
        split_by_cluster = best_split_rows_by_cluster(_read_csv(sample.split_repair_csv))
    elif not sample.cost_only:
        raise FileNotFoundError(
            f"sample {sample.name!r} requires split-repair CSV: {sample.split_repair_csv}"
        )

    missing_labels: list[int] = []
    for record in records:
        split = split_by_cluster.get(int(record["cluster"]))
        if split is None:
            record["strong_success"] = None
            record["practical_success"] = None
            missing_labels.append(int(record["cluster"]))
            continue
        record.update(split)
        record["strong_success"] = record["best_net_delta_q"] > STRONG_SUCCESS_THRESHOLD
        record["practical_success"] = (
            record["best_net_delta_q"] > PRACTICAL_SUCCESS_THRESHOLD
            and record["escaped_source_weight"] > 0
        )

    if missing_labels and not sample.cost_only:
        preview = ", ".join(str(cluster) for cluster in missing_labels[:10])
        suffix = "" if len(missing_labels) <= 10 else ", ..."
        raise ValueError(
            f"sample {sample.name!r} has {len(missing_labels)} external rows without "
            f"split-repair labels: {preview}{suffix}"
        )

    external_elapsed = _phase_elapsed(
        sample.external_summary_json,
        names=("external_grain_probes",),
    )
    split_elapsed = _phase_elapsed(
        sample.split_repair_summary_json,
        names=("split_merge_repair_probes",),
    )
    summary = {
        "sample": sample.name,
        "gamma": sample.gamma,
        "regime": sample.regime,
        "candidate_policy": sample.candidate_policy,
        "cost_only": sample.cost_only,
        "n_external_rows": len(records),
        "n_labeled_rows": sum(
            1 for record in records if record["strong_success"] is not None
        ),
        "n_missing_labels": sum(
            1 for record in records if record["strong_success"] is None
        ),
        "external_grain_csv": str(sample.external_grain_csv),
        "split_repair_csv": str(sample.split_repair_csv) if sample.split_repair_csv else None,
        "external_probe_elapsed_sec": external_elapsed,
        "split_repair_probe_elapsed_sec": split_elapsed,
        "split_to_external_elapsed_ratio": _safe_ratio(split_elapsed, external_elapsed)
        if external_elapsed is not None and split_elapsed is not None
        else None,
    }
    return records, summary


def _split_only_fallback(record: dict[str, Any]) -> bool:
    return (
        int(record.get("n_parts", 0) or 0) >= 3
        and int(record.get("core_part_count", 0) or 0) >= 2
        and float(record.get("singleton_frac", 1.0) or 0.0) <= 0.5
        and float(record.get("split_debt_per_doc_weight", 1e12) or 0.0) <= 2.5
    )


def build_rules(
    incident_budgets: Iterable[int] = DEFAULT_INCIDENT_BUDGETS,
    group_fraction_thresholds: Iterable[float] = DEFAULT_GROUP_FRACTION_THRESHOLDS,
    cascade_max_incident_edges: int = DEFAULT_CASCADE_MAX_INCIDENT_EDGES,
) -> list[Rule]:
    rules = [
        Rule(
            "doc_weight_ge_1500",
            lambda record: float(record.get("doc_weight", 0.0) or 0.0) >= 1500.0,
        ),
        Rule(
            "external_positive",
            lambda record: float(record.get("best_group_delta_q", 0.0) or 0.0) > 0.0,
        ),
    ]
    for threshold in group_fraction_thresholds:
        rules.append(
            Rule(
                f"external_positive_group_ge_{_format_threshold(threshold)}",
                lambda record, threshold=threshold: (
                    float(record.get("best_group_delta_q", 0.0) or 0.0) > 0.0
                    and float(record.get("best_group_fraction", 0.0) or 0.0)
                    >= threshold
                ),
            )
        )
    rules.extend(
        [
            Rule(
                "external_positive_or_doc_weight_ge_1500",
                lambda record: (
                    float(record.get("best_group_delta_q", 0.0) or 0.0) > 0.0
                    or float(record.get("doc_weight", 0.0) or 0.0) >= 1500.0
                ),
            ),
            Rule(
                "cascade_high_precision",
                lambda record: (
                    float(record.get("doc_weight", 0.0) or 0.0) >= 1500.0
                    and float(record.get("best_group_delta_q", 0.0) or 0.0) > 0.0
                    and float(record.get("best_group_fraction", 0.0) or 0.0) >= 0.01
                    and float(record.get("assigned_fraction", 0.0) or 0.0) >= 0.5
                ),
            ),
            Rule(
                "cascade_high_recall",
                lambda record: (
                    float(record.get("best_group_delta_q", 0.0) or 0.0) > 0.0
                    or (
                        float(record.get("doc_weight", 0.0) or 0.0) >= 1500.0
                        and _split_only_fallback(record)
                    )
                ),
            ),
            Rule(
                "cascade_size_cost_external_split_fallback",
                lambda record: (
                    float(record.get("doc_weight", 0.0) or 0.0) >= 750.0
                    and (
                        cascade_max_incident_edges <= 0
                        or _record_cost(record) <= cascade_max_incident_edges
                    )
                    and (
                        float(record.get("best_group_delta_q", 0.0) or 0.0) > 0.0
                        or _split_only_fallback(record)
                    )
                ),
            ),
        ]
    )
    for budget in incident_budgets:
        rules.append(
            Rule(
                f"external_positive_incident_le_{int(budget)}",
                lambda record, budget=int(budget): (
                    float(record.get("best_group_delta_q", 0.0) or 0.0) > 0.0
                    and _record_cost(record) <= budget
                ),
            )
        )
    return rules


def confusion_for_rule(
    records: list[dict[str, Any]],
    rule: Rule,
    label: str,
) -> dict[str, Any]:
    total = len(records)
    total_cost = sum(_record_cost(record) for record in records)
    selected = [record for record in records if rule.selector(record)]
    selected_cost = sum(_record_cost(record) for record in selected)

    tp = fp = tn = fn = 0
    positive_count = 0
    labeled_count = 0
    for record in records:
        value = record.get(label)
        if value is None:
            continue
        labeled_count += 1
        is_positive = bool(value)
        is_selected = rule.selector(record)
        if is_positive:
            positive_count += 1
        if is_selected and is_positive:
            tp += 1
        elif is_selected and not is_positive:
            fp += 1
        elif not is_selected and is_positive:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    saved_cost = total_cost - selected_cost
    return {
        "rule": rule.name,
        "label": label,
        "total_count": total,
        "labeled_count": labeled_count,
        "positive_count": positive_count,
        "selected_count": len(selected),
        "selected_fraction": len(selected) / total if total else 0.0,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "estimated_full_repair_cost_units": total_cost,
        "selected_full_repair_cost_units": selected_cost,
        "estimated_saved_full_repair_cost_units": saved_cost,
        "estimated_saved_full_repair_cost_fraction": saved_cost / total_cost
        if total_cost
        else 0.0,
    }


def _example_payload(record: dict[str, Any]) -> dict[str, Any]:
    payload = {field: record.get(field) for field in EXAMPLE_FIELDS}
    return payload


def _examples_for_rule(
    records: list[dict[str, Any]],
    rule: Rule,
    label: str,
    example_limit: int,
) -> dict[str, list[dict[str, Any]]]:
    false_positives: list[dict[str, Any]] = []
    false_negatives: list[dict[str, Any]] = []
    for record in records:
        value = record.get(label)
        if value is None:
            continue
        is_selected = rule.selector(record)
        if is_selected and not bool(value):
            false_positives.append(record)
        elif not is_selected and bool(value):
            false_negatives.append(record)

    false_positives.sort(
        key=lambda record: (
            float(record.get("best_group_delta_q", 0.0) or 0.0),
            float(record.get("doc_weight", 0.0) or 0.0),
        ),
        reverse=True,
    )
    false_negatives.sort(
        key=lambda record: (
            float(record.get("best_net_delta_q", 0.0) or 0.0),
            float(record.get("doc_weight", 0.0) or 0.0),
        ),
        reverse=True,
    )
    return {
        "false_positives": [
            _example_payload(record) for record in false_positives[:example_limit]
        ],
        "false_negatives": [
            _example_payload(record) for record in false_negatives[:example_limit]
        ],
    }


def _sample_rule_rows(
    sample_summary: dict[str, Any],
    metrics: list[dict[str, Any]],
    scope: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        row = {
            "scope": scope,
            "sample": sample_summary.get("sample", "__global__"),
            "regime": sample_summary.get("regime", ""),
            "gamma": sample_summary.get("gamma"),
            **metric,
            "external_probe_elapsed_sec": sample_summary.get("external_probe_elapsed_sec"),
            "split_repair_probe_elapsed_sec": sample_summary.get(
                "split_repair_probe_elapsed_sec"
            ),
            "split_to_external_elapsed_ratio": sample_summary.get(
                "split_to_external_elapsed_ratio"
            ),
        }
        rows.append(row)
    return rows


def evaluate_records(
    records: list[dict[str, Any]],
    rules: list[Rule],
    sample_summary: dict[str, Any],
    example_limit: int = 10,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    labels = ("strong_success", "practical_success")
    metrics: list[dict[str, Any]] = []
    examples: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = {}
    for rule in rules:
        examples[rule.name] = {}
        for label in labels:
            metric = confusion_for_rule(records, rule, label)
            metrics.append(metric)
            examples[rule.name][label] = _examples_for_rule(
                records, rule, label, example_limit
            )

    cost_curves = [
        metric
        for metric in metrics
        if metric["rule"].startswith("external_positive_incident_le_")
    ]
    detail = {
        **sample_summary,
        "metrics": metrics,
        "cost_curves": cost_curves,
        "examples": examples,
    }
    return metrics, detail


def _metric_by_rule(
    metrics: list[dict[str, Any]],
    rule: str,
    label: str,
) -> dict[str, Any] | None:
    for metric in metrics:
        if metric["rule"] == rule and metric["label"] == label:
            return metric
    return None


def _check(name: str, passed: bool | None, **details: Any) -> dict[str, Any]:
    return {"name": name, "pass": passed, **details}


def summarize_acceptance(
    result: dict[str, Any],
    labeled_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize validation acceptance gates from the current metrics."""

    useful_checks: list[dict[str, Any]] = []
    for sample in result["per_sample"]:
        metric = _metric_by_rule(sample["metrics"], "external_positive", "strong_success")
        if metric is None:
            continue
        if sample.get("regime") == "giant":
            useful_checks.append(
                _check(
                    f"{sample['sample']}: giant external_positive strong recall >= 0.90",
                    metric["recall"] >= 0.90,
                    observed=metric["recall"],
                    threshold=0.90,
                    selected_count=metric["selected_count"],
                    positive_count=metric["positive_count"],
                )
            )
        if sample.get("regime") in {"band", "weak"}:
            useful_checks.append(
                _check(
                    f"{sample['sample']}: band/weak external_positive selected fraction <= 0.05",
                    metric["selected_fraction"] <= 0.05,
                    observed=metric["selected_fraction"],
                    threshold=0.05,
                    selected_count=metric["selected_count"],
                    total_count=metric["total_count"],
                )
            )
        ratio = sample.get("split_to_external_elapsed_ratio")
        if ratio is not None:
            useful_checks.append(
                _check(
                    f"{sample['sample']}: external-grain runtime at least 10x cheaper",
                    ratio >= 10.0,
                    observed=ratio,
                    threshold=10.0,
                )
            )

    global_metrics = result["global_weighted_summary"]["metrics"]
    predictor_rules = [
        metric
        for metric in global_metrics
        if metric["label"] == "strong_success"
        and metric["selected_count"] > 0
        and (
            metric["rule"].startswith("external_")
            or metric["rule"].startswith("cascade_")
        )
    ]
    high_precision = max(
        predictor_rules,
        key=lambda metric: (
            metric["precision"],
            metric["recall"],
            metric["selected_count"],
        ),
        default=None,
    )
    high_recall = max(
        predictor_rules,
        key=lambda metric: (
            metric["recall"],
            metric["precision"],
            metric["selected_count"],
        ),
        default=None,
    )
    cost_fields_available = all(
        record.get("incident_directed_edges") is not None for record in labeled_records
    )
    cascade_checks = [
        _check(
            "global high-precision subset precision >= 0.95",
            high_precision is not None and high_precision["precision"] >= 0.95,
            rule_scope="external_or_cascade",
            rule=high_precision["rule"] if high_precision else None,
            observed=high_precision["precision"] if high_precision else None,
            recall=high_precision["recall"] if high_precision else None,
            selected_count=high_precision["selected_count"] if high_precision else 0,
            threshold=0.95,
        ),
        _check(
            "global high-recall mode recall >= 0.85",
            high_recall is not None and high_recall["recall"] >= 0.85,
            rule_scope="external_or_cascade",
            rule=high_recall["rule"] if high_recall else None,
            observed=high_recall["recall"] if high_recall else None,
            precision=high_recall["precision"] if high_recall else None,
            selected_count=high_recall["selected_count"] if high_recall else 0,
            threshold=0.85,
        ),
        _check(
            "selected-candidate cost fields are available",
            cost_fields_available,
            field="incident_directed_edges",
            labeled_count=len(labeled_records),
        ),
    ]

    useful_pass = all(check["pass"] for check in useful_checks) if useful_checks else None
    cascade_pass = all(check["pass"] for check in cascade_checks)
    return {
        "external_grain_predictor_useful": {
            "pass": useful_pass,
            "checks": useful_checks,
        },
        "cascade_ready_for_apply_mode_design": {
            "pass": cascade_pass,
            "checks": cascade_checks,
        },
    }


def summarize_coverage(specs: list[CoverageSpec]) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for spec in specs:
        has_external = spec.external_grain_csv is not None and spec.external_grain_csv.exists()
        has_split = spec.split_repair_csv is not None and spec.split_repair_csv.exists()
        if has_external and has_split:
            status = "matched_labeled"
        elif has_external and spec.cost_only:
            status = "cost_only_probe_available"
        elif has_external:
            status = "missing_split_label"
        elif has_split:
            status = "missing_external_grain"
        elif spec.full_repair_forbidden:
            status = "missing_cost_probe"
        else:
            status = "missing_artifacts"
        counts[status] = counts.get(status, 0) + 1
        entries.append(
            {
                "name": spec.name,
                "gamma": spec.gamma,
                "regime": spec.regime,
                "candidate_policy": spec.candidate_policy,
                "status": status,
                "cost_only": spec.cost_only,
                "full_repair_forbidden": spec.full_repair_forbidden,
                "has_external_grain_csv": has_external,
                "has_split_repair_csv": has_split,
                "external_grain_csv": str(spec.external_grain_csv)
                if spec.external_grain_csv
                else None,
                "split_repair_csv": str(spec.split_repair_csv)
                if spec.split_repair_csv
                else None,
            }
        )
    return {
        "counts": counts,
        "entries": entries,
    }


def evaluate_samples(
    samples: list[SampleSpec],
    coverage_specs: list[CoverageSpec] | None = None,
    incident_budgets: Iterable[int] = DEFAULT_INCIDENT_BUDGETS,
    group_fraction_thresholds: Iterable[float] = DEFAULT_GROUP_FRACTION_THRESHOLDS,
    cascade_max_incident_edges: int = DEFAULT_CASCADE_MAX_INCIDENT_EDGES,
    example_limit: int = 10,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rules = build_rules(
        incident_budgets=incident_budgets,
        group_fraction_thresholds=group_fraction_thresholds,
        cascade_max_incident_edges=cascade_max_incident_edges,
    )

    per_sample: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    all_records: list[dict[str, Any]] = []
    cost_only_records: list[dict[str, Any]] = []

    for sample in samples:
        records, sample_summary = join_sample_records(sample)
        metrics, detail = evaluate_records(
            records,
            rules,
            sample_summary,
            example_limit=example_limit,
        )
        per_sample.append(detail)
        csv_rows.extend(_sample_rule_rows(sample_summary, metrics, scope="sample"))
        if sample.cost_only:
            cost_only_records.extend(records)
        else:
            all_records.extend(records)

    global_summary = {
        "sample": "__global__",
        "gamma": None,
        "regime": "",
        "external_probe_elapsed_sec": None,
        "split_repair_probe_elapsed_sec": None,
        "split_to_external_elapsed_ratio": None,
    }
    global_metrics, global_detail = evaluate_records(
        all_records,
        rules,
        global_summary,
        example_limit=example_limit,
    )
    csv_rows.extend(_sample_rule_rows(global_summary, global_metrics, scope="global"))

    cost_only_summary = None
    if cost_only_records:
        sample_summary = {
            "sample": "__cost_only__",
            "gamma": None,
            "regime": "",
            "external_probe_elapsed_sec": None,
            "split_repair_probe_elapsed_sec": None,
            "split_to_external_elapsed_ratio": None,
        }
        cost_only_metrics, cost_only_summary = evaluate_records(
            cost_only_records,
            rules,
            sample_summary,
            example_limit=example_limit,
        )
        csv_rows.extend(
            _sample_rule_rows(sample_summary, cost_only_metrics, scope="cost_only")
        )

    result = {
        "description": (
            "External-grain pre-screen predictor validation against best "
            "cluster-level full split-repair labels."
        ),
        "label_definitions": {
            "strong_success": f"best_net_delta_q > {STRONG_SUCCESS_THRESHOLD:g}",
            "practical_success": (
                f"best_net_delta_q > {PRACTICAL_SUCCESS_THRESHOLD:g} "
                "and escaped_source_weight > 0"
            ),
        },
        "rule_names": [rule.name for rule in rules],
        "per_sample": per_sample,
        "global_weighted_summary": global_detail,
        "cost_only_summary": cost_only_summary,
    }
    result["acceptance"] = summarize_acceptance(result, all_records)
    if coverage_specs is not None:
        result["coverage_summary"] = summarize_coverage(coverage_specs)
    return result, csv_rows


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _parse_int_list(text: str) -> tuple[int, ...]:
    if not text.strip():
        return ()
    return tuple(int(item.strip()) for item in text.split(",") if item.strip())


def _parse_float_list(text: str) -> tuple[float, ...]:
    if not text.strip():
        return ()
    return tuple(float(item.strip()) for item in text.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate external-grain probes as split-repair pre-screen predictors."
    )
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--incident-budgets",
        default=",".join(str(value) for value in DEFAULT_INCIDENT_BUDGETS),
        help="Comma-separated incident_directed_edges budgets for cost curves.",
    )
    parser.add_argument(
        "--group-fraction-thresholds",
        default=",".join(str(value) for value in DEFAULT_GROUP_FRACTION_THRESHOLDS),
        help="Comma-separated best_group_fraction thresholds for external-positive rules.",
    )
    parser.add_argument(
        "--cascade-max-incident-edges",
        type=int,
        default=DEFAULT_CASCADE_MAX_INCIDENT_EDGES,
        help="Incident-edge budget used by the default cascade rule; <=0 disables it.",
    )
    parser.add_argument("--example-limit", type=int, default=10)
    args = parser.parse_args()

    samples, coverage_specs = load_metadata(args.metadata)
    result, rows = evaluate_samples(
        samples,
        coverage_specs=coverage_specs,
        incident_budgets=_parse_int_list(args.incident_budgets),
        group_fraction_thresholds=_parse_float_list(args.group_fraction_thresholds),
        cascade_max_incident_edges=args.cascade_max_incident_edges,
        example_limit=args.example_limit,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "external_grain_predictor_validation.json"
    csv_path = args.output_dir / "external_grain_predictor_validation.csv"
    json_path.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_csv(rows, csv_path)
    print(json.dumps({"json": str(json_path), "csv": str(csv_path)}, indent=2))


if __name__ == "__main__":
    main()
