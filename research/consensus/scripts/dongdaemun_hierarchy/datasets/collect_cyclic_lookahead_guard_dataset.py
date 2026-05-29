"""Collect trigger-level data for cheap cyclic postprocess guard design.

The expensive ``local_qf_beam_cyclic_lookahead`` variant is treated as an
oracle.  This script turns one or more cyclic pilot row CSVs into a compact
dataset where each row is a cyclic postprocess trigger, with cheap features
available before running lookahead and labels from schedule-replay lookahead.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable
import sys

REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").exists()
)
SCRIPT_ROOT = REPO_ROOT / "research/consensus/scripts"
_SCRIPT_PATHS = [REPO_ROOT, SCRIPT_ROOT]
_SCRIPT_PATHS.extend(path for path in SCRIPT_ROOT.rglob("*") if path.is_dir())
for _script_path in reversed(_SCRIPT_PATHS):
    _script_path_str = str(_script_path)
    if _script_path_str not in sys.path:
        sys.path.insert(0, _script_path_str)


DEFAULT_INPUT_ROOT = (
    Path("research/consensus/results/adaptive_refinement")
    / "dongdaemun_cyclic_postprocess_pilot_20260511"
)
DEFAULT_OUTPUT_DIR = (
    Path("research/consensus/results/adaptive_refinement")
    / "dongdaemun_cyclic_guard_dataset_20260511"
)

SCHEMA_VERSION = 1
ROWS_FILENAME = "cyclic_guard_training_rows.csv"
FEATURE_SUMMARY_FILENAME = "cyclic_guard_feature_summary.csv"
RULE_SCREEN_FILENAME = "cyclic_guard_rule_screen.csv"
REPORT_FILENAME = "cyclic_guard_report.md"
SUMMARY_FILENAME = "cyclic_guard_summary.json"

NUMERIC_FEATURES = (
    "immediate_delta_q",
    "immediate_delta_q_abs",
    "refinement_quality",
    "refinement_quality_delta_vs_start",
    "refinement_max_doc_weight_ratio",
    "refinement_n_above_max_doc_weight",
    "refinement_n_clusters",
    "refinement_n_singletons",
    "refinement_selected_parent_count_total",
    "refinement_applied_parent_count_total",
    "refinement_same_gamma_candidates_total",
    "refinement_high_gamma_candidates_total",
    "refinement_candidate_quality_delta_sum",
    "postprocess_quality_before",
    "postprocess_quality_after",
    "lookahead_iterations",
)

def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None

def _float_value(value: Any, default: float = 0.0) -> float:
    number = _finite_float(value)
    return default if number is None else number

def _int_value(value: Any, default: int = 0) -> int:
    number = _finite_float(value)
    return default if number is None else int(number)

def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}

def _parse_json_list(value: Any) -> list[str]:
    if value is None or not str(value).strip():
        return []
    text = str(value).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return [part.strip() for part in text.split(",") if part.strip()]
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return [str(parsed)]

def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))

def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value

def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fieldnames})

def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

def discover_row_paths(input_root: Path) -> list[Path]:
    return sorted(Path(input_root).glob("**/cyclic_postprocess_pilot_rows.csv"))

def _row_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("sample", "")),
        str(row.get("variant", "")),
        str(row.get("step", "")),
        str(row.get("chunk_index", "")),
    )

def _prefixed_refinement_fields(row: dict[str, Any] | None) -> dict[str, Any]:
    row = row or {}
    fields = {
        "quality": _finite_float(row.get("quality")),
        "quality_delta_vs_start": _finite_float(row.get("quality_delta_vs_start")),
        "max_doc_weight_ratio": _finite_float(row.get("max_doc_weight_ratio")),
        "n_above_max_doc_weight": _int_value(row.get("n_above_max_doc_weight")),
        "n_clusters": _int_value(row.get("n_clusters")),
        "n_singletons": _int_value(row.get("n_singletons")),
        "selected_parent_count_total": _int_value(
            row.get("selected_parent_count_total")
        ),
        "applied_parent_count_total": _int_value(row.get("applied_parent_count_total")),
        "same_gamma_candidates_total": _int_value(row.get("same_gamma_candidates_total")),
        "high_gamma_candidates_total": _int_value(row.get("high_gamma_candidates_total")),
        "candidate_quality_delta_sum": _finite_float(
            row.get("candidate_quality_delta_sum")
        ),
    }
    return {f"refinement_{key}": value for key, value in fields.items()}

def build_trigger_rows(row_paths: Iterable[Path]) -> list[dict[str, Any]]:
    trigger_rows: list[dict[str, Any]] = []
    for path in row_paths:
        rows = _read_csv(Path(path))
        refinement_by_key = {
            _row_key(row): row for row in rows if row.get("phase") == "refinement_chunk"
        }
        for row in rows:
            if row.get("phase") != "cyclic_postprocess":
                continue
            before = _finite_float(row.get("postprocess_quality_before"))
            after = _finite_float(row.get("postprocess_quality_after"))
            immediate_delta = None if before is None or after is None else after - before
            reasons = _parse_json_list(row.get("postprocess_reasons"))
            lookahead_used = _bool_value(row.get("lookahead_guard_used"))
            lookahead_accepted = (
                _bool_value(row.get("lookahead_guard_accepted"))
                if lookahead_used
                else None
            )
            refinement = refinement_by_key.get(_row_key(row))
            out = {
                "source_rows_path": str(path),
                "run_name": Path(path).parent.name,
                "sample": row.get("sample", ""),
                "variant": row.get("variant", ""),
                "step": _int_value(row.get("step")),
                "chunk_index": _int_value(row.get("chunk_index")),
                "postprocess_status": row.get("postprocess_status", ""),
                "postprocess_reasons": reasons,
                "trigger_no_applied_parents": "no_applied_parents" in reasons,
                "trigger_interval": "interval" in reasons,
                "trigger_quality_plateau": "quality_plateau" in reasons,
                "postprocess_accepted_after_guard": _bool_value(
                    row.get("postprocess_accepted")
                ),
                "immediate_postprocess_accepted": bool(
                    lookahead_used or _bool_value(row.get("postprocess_accepted"))
                ),
                "postprocess_quality_before": before,
                "postprocess_quality_after": after,
                "immediate_delta_q": immediate_delta,
                "immediate_delta_q_abs": (
                    None if immediate_delta is None else abs(immediate_delta)
                ),
                "lookahead_guard_used": lookahead_used,
                "lookahead_guard_accepted": lookahead_accepted,
                "lookahead_label": lookahead_accepted,
                "lookahead_iterations": _int_value(row.get("lookahead_iterations")),
                "lookahead_baseline_quality": _finite_float(
                    row.get("lookahead_baseline_quality")
                ),
                "lookahead_candidate_quality": _finite_float(
                    row.get("lookahead_candidate_quality")
                ),
                "lookahead_delta_q": _finite_float(row.get("lookahead_delta_q")),
                "lookahead_min_delta_q": _finite_float(row.get("lookahead_min_delta_q")),
                "lookahead_elapsed_sec": _finite_float(row.get("lookahead_elapsed_sec")),
            }
            out.update(_prefixed_refinement_fields(refinement))
            trigger_rows.append(out)
    return trigger_rows

def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    dx = [x - mean_x for x in xs]
    dy = [y - mean_y for y in ys]
    denom_x = math.sqrt(sum(x * x for x in dx))
    denom_y = math.sqrt(sum(y * y for y in dy))
    if denom_x == 0.0 or denom_y == 0.0:
        return None
    return sum(x * y for x, y in zip(dx, dy)) / (denom_x * denom_y)

def summarize_features(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    labeled = [row for row in rows if row.get("lookahead_label") is not None]
    labels = [1.0 if row.get("lookahead_label") else 0.0 for row in labeled]
    summaries: list[dict[str, Any]] = []
    for feature in NUMERIC_FEATURES:
        pairs = [
            (_finite_float(row.get(feature)), 1.0 if row.get("lookahead_label") else 0.0)
            for row in labeled
        ]
        pairs = [(x, y) for x, y in pairs if x is not None]
        accepted = [x for x, y in pairs if y == 1.0]
        rejected = [x for x, y in pairs if y == 0.0]
        values = [x for x, _ in pairs]
        summaries.append(
            {
                "feature": feature,
                "n": len(values),
                "n_accept": len(accepted),
                "n_reject": len(rejected),
                "mean": None if not values else sum(values) / len(values),
                "mean_accept": None
                if not accepted
                else sum(accepted) / len(accepted),
                "mean_reject": None
                if not rejected
                else sum(rejected) / len(rejected),
                "accept_minus_reject": None
                if not accepted or not rejected
                else (sum(accepted) / len(accepted))
                - (sum(rejected) / len(rejected)),
                "pearson_with_accept": _pearson(
                    [x for x, _ in pairs],
                    [y for _, y in pairs],
                )
                if len(set(labels)) > 1
                else None,
            }
        )
    return summaries

def _thresholds(values: list[float]) -> list[float]:
    finite = sorted({value for value in values if math.isfinite(value)})
    if not finite:
        return []
    candidates = {finite[0], finite[-1], 0.0, 0.1, 1.0, 5.0, 10.0}
    if len(finite) > 1:
        candidates.update((a + b) / 2.0 for a, b in zip(finite, finite[1:]))
    return sorted(candidates)

def _metrics(name: str, predictions: list[bool], labels: list[bool]) -> dict[str, Any]:
    tp = sum(pred and label for pred, label in zip(predictions, labels))
    fp = sum(pred and not label for pred, label in zip(predictions, labels))
    tn = sum((not pred) and (not label) for pred, label in zip(predictions, labels))
    fn = sum((not pred) and label for pred, label in zip(predictions, labels))
    total = len(labels)
    return {
        "rule": name,
        "n": total,
        "tp": tp,
        "fp_false_accept": fp,
        "tn": tn,
        "fn_false_reject": fn,
        "accuracy": None if total == 0 else (tp + tn) / total,
        "precision": None if tp + fp == 0 else tp / (tp + fp),
        "recall": None if tp + fn == 0 else tp / (tp + fn),
    }

def screen_rules(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    labeled = [row for row in rows if row.get("lookahead_label") is not None]
    labels = [bool(row["lookahead_label"]) for row in labeled]
    if not labeled:
        return []
    rule_rows: list[dict[str, Any]] = []
    for feature in (
        "immediate_delta_q",
        "refinement_quality_delta_vs_start",
        "refinement_candidate_quality_delta_sum",
    ):
        values = [_finite_float(row.get(feature)) for row in labeled]
        thresholds = _thresholds([value for value in values if value is not None])
        for threshold in thresholds:
            predictions = [
                (_finite_float(row.get(feature)) or -math.inf) >= threshold
                for row in labeled
            ]
            rule_rows.append(
                _metrics(f"{feature} >= {threshold:.6g}", predictions, labels)
            )
    rule_rows.append(
        _metrics(
            "immediate_delta_q >= 1 and refinement_applied_parent_count_total == 0",
            [
                (_finite_float(row.get("immediate_delta_q")) or -math.inf) >= 1.0
                and _int_value(row.get("refinement_applied_parent_count_total")) == 0
                for row in labeled
            ],
            labels,
        )
    )
    return sorted(
        rule_rows,
        key=lambda row: (
            row["fp_false_accept"],
            -(row["accuracy"] if row["accuracy"] is not None else -1.0),
            row["fn_false_reject"],
            row["rule"],
        ),
    )

def _build_report(
    *,
    trigger_rows: list[dict[str, Any]],
    feature_rows: list[dict[str, Any]],
    rule_rows: list[dict[str, Any]],
) -> str:
    labeled = [row for row in trigger_rows if row.get("lookahead_label") is not None]
    accepted = sum(bool(row.get("lookahead_label")) for row in labeled)
    rejected = len(labeled) - accepted
    lines = [
        "# Cyclic Lookahead Guard Dataset",
        "",
        f"- Trigger rows: {len(trigger_rows)}",
        f"- Labeled lookahead rows: {len(labeled)}",
        f"- Oracle accept/reject: {accepted}/{rejected}",
        "",
        "## Top Feature Signals",
        "",
        "| feature | n | mean accept | mean reject | diff | corr |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    ranked_features = sorted(
        feature_rows,
        key=lambda row: abs(row.get("pearson_with_accept") or 0.0),
        reverse=True,
    )
    for row in ranked_features[:10]:
        lines.append(
            "| {feature} | {n} | {ma} | {mr} | {diff} | {corr} |".format(
                feature=row["feature"],
                n=row["n"],
                ma=_fmt(row.get("mean_accept")),
                mr=_fmt(row.get("mean_reject")),
                diff=_fmt(row.get("accept_minus_reject")),
                corr=_fmt(row.get("pearson_with_accept")),
            )
        )
    lines.extend(
        [
            "",
            "## Rule Screen",
            "",
            "| rule | fp false accept | fn false reject | accuracy | precision | recall |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rule_rows[:10]:
        lines.append(
            "| {rule} | {fp} | {fn} | {acc} | {prec} | {rec} |".format(
                rule=row["rule"],
                fp=row["fp_false_accept"],
                fn=row["fn_false_reject"],
                acc=_fmt(row.get("accuracy")),
                prec=_fmt(row.get("precision")),
                rec=_fmt(row.get("recall")),
            )
        )
    lines.append("")
    return "\n".join(lines)

def _fmt(value: Any) -> str:
    number = _finite_float(value)
    return "" if number is None else f"{number:.4g}"

def collect_guard_dataset(
    *,
    row_paths: Iterable[Path],
    output_dir: Path,
) -> dict[str, Any]:
    row_paths = [Path(path) for path in row_paths]
    trigger_rows = build_trigger_rows(row_paths)
    feature_rows = summarize_features(trigger_rows)
    rule_rows = screen_rules(trigger_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / ROWS_FILENAME
    features_path = output_dir / FEATURE_SUMMARY_FILENAME
    rules_path = output_dir / RULE_SCREEN_FILENAME
    report_path = output_dir / REPORT_FILENAME
    summary_path = output_dir / SUMMARY_FILENAME
    _write_csv(rows_path, trigger_rows)
    _write_csv(features_path, feature_rows)
    _write_csv(rules_path, rule_rows)
    report_path.write_text(
        _build_report(
            trigger_rows=trigger_rows,
            feature_rows=feature_rows,
            rule_rows=rule_rows,
        ),
        encoding="utf-8",
    )
    labeled = [row for row in trigger_rows if row.get("lookahead_label") is not None]
    payload = {
        "schema": "cyclic_lookahead_guard_dataset.v1",
        "schema_version": SCHEMA_VERSION,
        "n_input_files": len(row_paths),
        "n_trigger_rows": len(trigger_rows),
        "n_labeled_rows": len(labeled),
        "n_oracle_accept": sum(bool(row.get("lookahead_label")) for row in labeled),
        "n_oracle_reject": sum(not bool(row.get("lookahead_label")) for row in labeled),
        "input_paths": [str(path) for path in row_paths],
        "paths": {
            "training_rows": str(rows_path),
            "feature_summary": str(features_path),
            "rule_screen": str(rules_path),
            "report": str(report_path),
            "summary": str(summary_path),
        },
    }
    _write_json(summary_path, payload)
    return payload

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--input", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser

def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    row_paths = list(args.input) if args.input else discover_row_paths(args.input_root)
    payload = collect_guard_dataset(row_paths=row_paths, output_dir=args.output_dir)
    print(f"Saved cyclic guard dataset to {payload['paths']['summary']}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
