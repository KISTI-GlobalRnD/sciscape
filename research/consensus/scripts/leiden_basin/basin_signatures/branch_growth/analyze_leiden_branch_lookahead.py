"""Analyze delayed-commit branch policies for Leiden iteration profiles.

This is an offline analysis layer over ``run_leiden_random_refinement_profile``
output.  Each branch candidate is the observed ``(seed, randomness)`` pair for a
sample/layer, and each iteration budget is treated as a lookahead checkpoint.
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


DEFAULT_INPUT_CSV = (
    Path("research/consensus/results/adaptive_refinement")
    / "leiden_iteration_budget_profile_20260511"
    / "leiden_random_refinement_profile_rows.csv"
)
DEFAULT_OUTPUT_DIR = (
    Path("research/consensus/results/adaptive_refinement")
    / "leiden_branch_lookahead_analysis_20260511"
)

BUDGET_LABELS = ("1", "2", "3", "5", "10", "convergence")
EARLY_SURVIVAL_BUDGETS = ("1", "2", "3", "5")
POLICY_STAGE1_BUDGETS = ("1", "2")
FINAL_BUDGETS = ("10", "convergence")
POLICY_NAMES = (
    "greedy_top1",
    "quality_top5",
    "mixed_beam_v1",
    "lookahead_v1",
    "mixed_beam_v2",
    "iter5_screen_all_top3",
    "iter5_screen_all_top5",
    "margin_polish_top2",
)
V2_POLICY_NAMES = (
    "mixed_beam_v2",
    "iter5_screen_all_top3",
    "iter5_screen_all_top5",
    "margin_polish_top2",
)
MARGIN_POLISH_RELATIVE_GAP = 1.0e-4
MARGIN_POLISH_ABSOLUTE_GAP = 25.0
SCHEMA_VERSION = 1

RANK_FLIP_FILENAME = "branch_rank_flip_by_candidate.csv"
SURVIVAL_FILENAME = "branch_survival_by_budget.csv"
LATE_FLIP_FILENAME = "branch_late_riser_faller_summary.csv"
POLICY_FILENAME = "branch_policy_simulation.csv"
REPORT_FILENAME = "branch_lookahead_report.md"
SUMMARY_FILENAME = "branch_lookahead_summary.json"

RANK_FLIP_FIELDS = [
    "sample",
    "source_sample",
    "edge_layer",
    "candidate_id",
    "seed",
    "randomness",
    "final_budget",
    "final_rank",
    "final_quality",
    "final_max_doc_weight_ratio",
    "final_n_above_max_doc_weight",
    "iter1_to_final_rank_delta",
    "iter1_to_final_quality_delta",
    "rank_path",
    "quality_path",
    "pressure_path",
]
for _label in BUDGET_LABELS:
    _prefix = "convergence" if _label == "convergence" else f"iter{_label}"
    RANK_FLIP_FIELDS.extend(
        [
            f"{_prefix}_rank",
            f"{_prefix}_quality",
            f"{_prefix}_max_doc_weight_ratio",
            f"{_prefix}_n_above_max_doc_weight",
            f"{_prefix}_elapsed_sec",
            f"{_prefix}_n_clusters",
        ]
    )

SURVIVAL_FIELDS = [
    "sample",
    "source_sample",
    "edge_layer",
    "early_budget",
    "top_k",
    "final_budget",
    "n_candidates",
    "n_samples",
    "final_top1_in_early_topk",
    "top1_survival_rate",
    "final_top3_hit_count",
    "final_top3_size",
    "final_top3_hit_rate",
    "final_top3_all_in_early_topk",
    "top3_full_survival_rate",
    "mean_final_top3_hit_rate",
    "final_top1_candidate_id",
    "final_top1_early_rank",
    "final_top3_candidate_ids",
    "early_topk_candidate_ids",
]

LATE_FLIP_FIELDS = [
    "sample",
    "source_sample",
    "edge_layer",
    "candidate_id",
    "seed",
    "randomness",
    "early_budget",
    "final_budget",
    "classification",
    "early_rank",
    "final_rank",
    "rank_delta",
    "rank_improvement",
    "early_quality",
    "final_quality",
    "quality_delta",
    "early_max_doc_weight_ratio",
    "final_max_doc_weight_ratio",
    "pressure_delta",
    "early_n_above_max_doc_weight",
    "final_n_above_max_doc_weight",
]

POLICY_FIELDS = [
    "sample",
    "source_sample",
    "edge_layer",
    "policy_name",
    "stage1_budget",
    "stage2_budget",
    "stage2_promote_k",
    "final_budget",
    "polish_top1_convergence",
    "convergence_polish_k",
    "n_convergence_polished",
    "polished_candidate_ids",
    "selected_before_polish_candidate_id",
    "selection_stage",
    "n_stage1_candidates",
    "n_stage1_retained",
    "n_stage2_evaluated",
    "n_final_evaluated",
    "stage1_retained_candidate_ids",
    "stage1_retention_reasons",
    "stage2_promoted_candidate_ids",
    "selected_candidate_id",
    "selected_seed",
    "selected_randomness",
    "selected_budget",
    "selected_quality",
    "selected_max_doc_weight_ratio",
    "selected_n_above_max_doc_weight",
    "selected_pressure_penalty_score",
    "best10_candidate_id",
    "best10_quality",
    "best10_max_doc_weight_ratio",
    "best_convergence_candidate_id",
    "best_convergence_quality",
    "best_convergence_max_doc_weight_ratio",
    "quality_gap_to_best10",
    "quality_gap_to_best_convergence",
    "quality_recovery_ratio_vs_best10",
    "quality_recovery_ratio_vs_best_convergence",
    "pressure_delta_vs_best10",
    "pressure_delta_vs_best_convergence",
    "retained_contains_best10",
    "promoted_contains_best10",
    "selected_matches_best10_candidate",
    "selected_matches_best_convergence_candidate",
    "stage1_elapsed_sec",
    "stage2_elapsed_sec",
    "final_elapsed_sec",
    "polish_elapsed_sec",
    "estimated_elapsed_proxy_sec",
    "full10_elapsed_sec",
    "full_convergence_elapsed_sec",
    "estimated_elapsed_saving_vs_full10_ratio",
    "estimated_elapsed_saving_vs_full_convergence_ratio",
]

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

def _int_value(value: Any) -> int | None:
    number = _finite_float(value)
    return None if number is None else int(number)

def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}

def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value), separators=(",", ":"))
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value

def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    seen = set(fieldnames)
    all_fields = list(fieldnames)
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                all_fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=all_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in all_fields})

def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

def _budget_sort_key(label: str) -> tuple[int, int]:
    return (1, 0) if label == "convergence" else (0, int(label))

def _budget_prefix(label: str) -> str:
    return "convergence" if label == "convergence" else f"iter{label}"

def _budget_label(row: dict[str, Any]) -> str:
    requested = row.get("requested_n_iterations")
    if requested is not None and str(requested).strip():
        return str(requested).strip()
    n_iterations = _int_value(row.get("n_iterations")) or 0
    return "convergence" if n_iterations == 0 else str(n_iterations)

def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    normalized["sample"] = str(row.get("sample") or "")
    normalized["source_sample"] = row.get("source_sample") or ""
    normalized["edge_layer"] = row.get("edge_layer") or ""
    normalized["seed"] = _int_value(row.get("seed"))
    normalized["randomness"] = _finite_float(row.get("randomness"))
    normalized["requested_n_iterations"] = _budget_label(row)
    normalized["supported"] = _bool_value(row.get("supported", True))
    normalized["quality"] = _finite_float(row.get("quality"))
    normalized["max_doc_weight_ratio"] = _finite_float(row.get("max_doc_weight_ratio"))
    normalized["n_above_max_doc_weight"] = _int_value(row.get("n_above_max_doc_weight"))
    normalized["elapsed_sec"] = _finite_float(row.get("elapsed_sec"))
    normalized["n_clusters"] = _int_value(row.get("n_clusters"))
    normalized["quality_gain_per_sec"] = _finite_float(row.get("quality_gain_per_sec"))
    return normalized

def load_profile_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return [_normalize_row(row) for row in csv.DictReader(fh)]

def _candidate_key(row: dict[str, Any]) -> tuple[int, float]:
    return (int(row["seed"]), float(row["randomness"]))

def _candidate_id_from_key(key: tuple[int, float]) -> str:
    seed, randomness = key
    return f"seed={seed}|randomness={randomness:g}"

def _candidate_id(row: dict[str, Any]) -> str:
    return _candidate_id_from_key(_candidate_key(row))

def _selection_sort_key(row: dict[str, Any]) -> tuple[float, float, int, float, int, float]:
    quality = _finite_float(row.get("quality"))
    pressure = _finite_float(row.get("max_doc_weight_ratio"))
    n_above = _int_value(row.get("n_above_max_doc_weight"))
    elapsed = _finite_float(row.get("elapsed_sec"))
    return (
        -(quality if quality is not None else -math.inf),
        pressure if pressure is not None else math.inf,
        n_above if n_above is not None else 10**12,
        elapsed if elapsed is not None else math.inf,
        int(row.get("seed") or 0),
        float(row.get("randomness") or 0.0),
    )

def _pressure_sort_key(row: dict[str, Any]) -> tuple[float, int, float, float, int, float]:
    quality = _finite_float(row.get("quality"))
    pressure = _finite_float(row.get("max_doc_weight_ratio"))
    n_above = _int_value(row.get("n_above_max_doc_weight"))
    elapsed = _finite_float(row.get("elapsed_sec"))
    return (
        pressure if pressure is not None else math.inf,
        n_above if n_above is not None else 10**12,
        -(quality if quality is not None else -math.inf),
        elapsed if elapsed is not None else math.inf,
        int(row.get("seed") or 0),
        float(row.get("randomness") or 0.0),
    )

def _sum_elapsed(rows: Iterable[dict[str, Any]]) -> float:
    return float(
        sum(
            elapsed
            for elapsed in (_finite_float(row.get("elapsed_sec")) for row in rows)
            if elapsed is not None
        )
    )

def _best_row(rows: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [row for row in rows if row.get("supported") and row.get("quality") is not None]
    if not eligible:
        return None
    return min(eligible, key=_selection_sort_key)

def _rows_by_sample_budget(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        if not row.get("supported"):
            continue
        grouped.setdefault(str(row["sample"]), {}).setdefault(_budget_label(row), []).append(row)
    return grouped

def _rows_by_candidate(rows: Iterable[dict[str, Any]]) -> dict[tuple[int, float], dict[str, Any]]:
    return {_candidate_key(row): row for row in rows if row.get("supported")}

def _rank_by_candidate(rows: list[dict[str, Any]]) -> dict[tuple[int, float], int]:
    return {
        _candidate_key(row): rank
        for rank, row in enumerate(sorted(rows, key=_selection_sort_key), start=1)
    }

def _available_final_budget(sample_budgets: dict[str, list[dict[str, Any]]]) -> str:
    if sample_budgets.get("convergence"):
        return "convergence"
    if sample_budgets.get("10"):
        return "10"
    return sorted(sample_budgets, key=_budget_sort_key)[-1]

def build_rank_flip_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for sample, by_budget in sorted(_rows_by_sample_budget(rows).items()):
        ranks_by_budget = {
            label: _rank_by_candidate(budget_rows)
            for label, budget_rows in by_budget.items()
        }
        rows_by_budget_candidate = {
            label: _rows_by_candidate(budget_rows)
            for label, budget_rows in by_budget.items()
        }
        candidate_keys = sorted(
            {
                key
                for rows_by_candidate in rows_by_budget_candidate.values()
                for key in rows_by_candidate
            }
        )
        final_budget = _available_final_budget(by_budget)
        final_candidates = rows_by_budget_candidate.get(final_budget, {})
        final_ranks = ranks_by_budget.get(final_budget, {})
        for key in candidate_keys:
            first_row = next(
                rows_by_candidate[key]
                for rows_by_candidate in rows_by_budget_candidate.values()
                if key in rows_by_candidate
            )
            final_row = final_candidates.get(key)
            iter1_row = rows_by_budget_candidate.get("1", {}).get(key)
            final_rank = final_ranks.get(key)
            iter1_rank = ranks_by_budget.get("1", {}).get(key)
            row: dict[str, Any] = {
                "sample": sample,
                "source_sample": first_row.get("source_sample"),
                "edge_layer": first_row.get("edge_layer"),
                "candidate_id": _candidate_id_from_key(key),
                "seed": key[0],
                "randomness": key[1],
                "final_budget": final_budget,
                "final_rank": final_rank,
                "final_quality": None if final_row is None else final_row.get("quality"),
                "final_max_doc_weight_ratio": (
                    None if final_row is None else final_row.get("max_doc_weight_ratio")
                ),
                "final_n_above_max_doc_weight": (
                    None if final_row is None else final_row.get("n_above_max_doc_weight")
                ),
                "iter1_to_final_rank_delta": (
                    None
                    if iter1_rank is None or final_rank is None
                    else final_rank - iter1_rank
                ),
                "iter1_to_final_quality_delta": (
                    None
                    if iter1_row is None or final_row is None
                    else final_row.get("quality") - iter1_row.get("quality")
                ),
            }
            rank_path: list[str] = []
            quality_path: list[str] = []
            pressure_path: list[str] = []
            for label in BUDGET_LABELS:
                budget_row = rows_by_budget_candidate.get(label, {}).get(key)
                prefix = _budget_prefix(label)
                rank = ranks_by_budget.get(label, {}).get(key)
                row[f"{prefix}_rank"] = rank
                row[f"{prefix}_quality"] = None if budget_row is None else budget_row.get("quality")
                row[f"{prefix}_max_doc_weight_ratio"] = (
                    None if budget_row is None else budget_row.get("max_doc_weight_ratio")
                )
                row[f"{prefix}_n_above_max_doc_weight"] = (
                    None if budget_row is None else budget_row.get("n_above_max_doc_weight")
                )
                row[f"{prefix}_elapsed_sec"] = (
                    None if budget_row is None else budget_row.get("elapsed_sec")
                )
                row[f"{prefix}_n_clusters"] = (
                    None if budget_row is None else budget_row.get("n_clusters")
                )
                if budget_row is not None:
                    rank_path.append(f"{label}:{rank}")
                    quality_path.append(f"{label}:{budget_row.get('quality'):.6f}")
                    pressure = budget_row.get("max_doc_weight_ratio")
                    pressure_path.append(
                        f"{label}:{'' if pressure is None else f'{pressure:.6f}'}"
                    )
            row["rank_path"] = " > ".join(rank_path)
            row["quality_path"] = " > ".join(quality_path)
            row["pressure_path"] = " > ".join(pressure_path)
            output.append(row)
    return output

def _sample_meta(sample_rows: list[dict[str, Any]]) -> dict[str, Any]:
    first = sample_rows[0] if sample_rows else {}
    return {
        "source_sample": first.get("source_sample", ""),
        "edge_layer": first.get("edge_layer", ""),
    }

def build_branch_survival_rows(
    rows: list[dict[str, Any]],
    *,
    top_k_values: tuple[int, ...] = (1, 2, 3, 5),
) -> list[dict[str, Any]]:
    sample_rows: list[dict[str, Any]] = []
    for sample, by_budget in sorted(_rows_by_sample_budget(rows).items()):
        meta = _sample_meta(next(iter(by_budget.values()), []))
        for early_budget in EARLY_SURVIVAL_BUDGETS:
            early_rows = by_budget.get(early_budget, [])
            if not early_rows:
                continue
            early_ranked = sorted(early_rows, key=_selection_sort_key)
            early_ranks = _rank_by_candidate(early_rows)
            for final_budget in FINAL_BUDGETS:
                final_rows = by_budget.get(final_budget, [])
                if not final_rows or final_budget == early_budget:
                    continue
                final_ranked = sorted(final_rows, key=_selection_sort_key)
                final_top1_key = _candidate_key(final_ranked[0])
                final_top3_keys = [_candidate_key(row) for row in final_ranked[:3]]
                for top_k in top_k_values:
                    early_top_keys = {
                        _candidate_key(row) for row in early_ranked[: min(top_k, len(early_ranked))]
                    }
                    hit_count = sum(1 for key in final_top3_keys if key in early_top_keys)
                    sample_rows.append(
                        {
                            "sample": sample,
                            **meta,
                            "early_budget": early_budget,
                            "top_k": top_k,
                            "final_budget": final_budget,
                            "n_candidates": len(early_ranked),
                            "n_samples": 1,
                            "final_top1_in_early_topk": final_top1_key in early_top_keys,
                            "top1_survival_rate": 1.0
                            if final_top1_key in early_top_keys
                            else 0.0,
                            "final_top3_hit_count": hit_count,
                            "final_top3_size": len(final_top3_keys),
                            "final_top3_hit_rate": (
                                hit_count / len(final_top3_keys) if final_top3_keys else None
                            ),
                            "final_top3_all_in_early_topk": hit_count == len(final_top3_keys),
                            "top3_full_survival_rate": 1.0
                            if hit_count == len(final_top3_keys)
                            else 0.0,
                            "mean_final_top3_hit_rate": (
                                hit_count / len(final_top3_keys) if final_top3_keys else None
                            ),
                            "final_top1_candidate_id": _candidate_id_from_key(final_top1_key),
                            "final_top1_early_rank": early_ranks.get(final_top1_key),
                            "final_top3_candidate_ids": [
                                _candidate_id_from_key(key) for key in final_top3_keys
                            ],
                            "early_topk_candidate_ids": [
                                _candidate_id(row)
                                for row in early_ranked[: min(top_k, len(early_ranked))]
                            ],
                        }
                    )

    aggregate_rows: list[dict[str, Any]] = []
    aggregate_groups: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in sample_rows:
        aggregate_groups.setdefault(
            (str(row["early_budget"]), int(row["top_k"]), str(row["final_budget"])),
            [],
        ).append(row)
    for (early_budget, top_k, final_budget), group in sorted(
        aggregate_groups.items(), key=lambda item: (_budget_sort_key(item[0][0]), item[0][1], _budget_sort_key(item[0][2]))
    ):
        top1_values = [
            _finite_float(row.get("top1_survival_rate"))
            for row in group
            if _finite_float(row.get("top1_survival_rate")) is not None
        ]
        top3_full_values = [
            _finite_float(row.get("top3_full_survival_rate"))
            for row in group
            if _finite_float(row.get("top3_full_survival_rate")) is not None
        ]
        top3_hit_values = [
            _finite_float(row.get("final_top3_hit_rate"))
            for row in group
            if _finite_float(row.get("final_top3_hit_rate")) is not None
        ]
        aggregate_rows.append(
            {
                "sample": "__all__",
                "source_sample": "",
                "edge_layer": "",
                "early_budget": early_budget,
                "top_k": top_k,
                "final_budget": final_budget,
                "n_candidates": "",
                "n_samples": len(group),
                "final_top1_in_early_topk": "",
                "top1_survival_rate": (
                    sum(top1_values) / len(top1_values) if top1_values else None
                ),
                "final_top3_hit_count": "",
                "final_top3_size": "",
                "final_top3_hit_rate": "",
                "final_top3_all_in_early_topk": "",
                "top3_full_survival_rate": (
                    sum(top3_full_values) / len(top3_full_values)
                    if top3_full_values
                    else None
                ),
                "mean_final_top3_hit_rate": (
                    sum(top3_hit_values) / len(top3_hit_values) if top3_hit_values else None
                ),
                "final_top1_candidate_id": "",
                "final_top1_early_rank": "",
                "final_top3_candidate_ids": "",
                "early_topk_candidate_ids": "",
            }
        )
    return sample_rows + aggregate_rows

def build_late_riser_faller_rows(
    rows: list[dict[str, Any]],
    *,
    final_top_rank: int = 3,
    low_rank_threshold: int = 5,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for sample, by_budget in sorted(_rows_by_sample_budget(rows).items()):
        for early_budget in EARLY_SURVIVAL_BUDGETS:
            early_rows = by_budget.get(early_budget, [])
            if not early_rows:
                continue
            early_by_candidate = _rows_by_candidate(early_rows)
            early_ranks = _rank_by_candidate(early_rows)
            for final_budget in FINAL_BUDGETS:
                final_rows = by_budget.get(final_budget, [])
                if not final_rows or final_budget == early_budget:
                    continue
                final_by_candidate = _rows_by_candidate(final_rows)
                final_ranks = _rank_by_candidate(final_rows)
                for key, early_rank in early_ranks.items():
                    final_rank = final_ranks.get(key)
                    if final_rank is None:
                        continue
                    classification: str | None = None
                    if early_rank > low_rank_threshold and final_rank <= final_top_rank:
                        classification = "late_riser"
                    elif early_rank <= final_top_rank and final_rank > low_rank_threshold:
                        classification = "late_faller"
                    if classification is None:
                        continue
                    early_row = early_by_candidate[key]
                    final_row = final_by_candidate[key]
                    early_pressure = _finite_float(early_row.get("max_doc_weight_ratio"))
                    final_pressure = _finite_float(final_row.get("max_doc_weight_ratio"))
                    early_quality = _finite_float(early_row.get("quality"))
                    final_quality = _finite_float(final_row.get("quality"))
                    output.append(
                        {
                            "sample": sample,
                            "source_sample": early_row.get("source_sample"),
                            "edge_layer": early_row.get("edge_layer"),
                            "candidate_id": _candidate_id_from_key(key),
                            "seed": key[0],
                            "randomness": key[1],
                            "early_budget": early_budget,
                            "final_budget": final_budget,
                            "classification": classification,
                            "early_rank": early_rank,
                            "final_rank": final_rank,
                            "rank_delta": final_rank - early_rank,
                            "rank_improvement": early_rank - final_rank,
                            "early_quality": early_quality,
                            "final_quality": final_quality,
                            "quality_delta": (
                                None
                                if early_quality is None or final_quality is None
                                else final_quality - early_quality
                            ),
                            "early_max_doc_weight_ratio": early_pressure,
                            "final_max_doc_weight_ratio": final_pressure,
                            "pressure_delta": (
                                None
                                if early_pressure is None or final_pressure is None
                                else final_pressure - early_pressure
                            ),
                            "early_n_above_max_doc_weight": early_row.get(
                                "n_above_max_doc_weight"
                            ),
                            "final_n_above_max_doc_weight": final_row.get(
                                "n_above_max_doc_weight"
                            ),
                        }
                    )
    return sorted(
        output,
        key=lambda row: (
            str(row["sample"]),
            _budget_sort_key(str(row["early_budget"])),
            _budget_sort_key(str(row["final_budget"])),
            str(row["classification"]),
            int(row["final_rank"]),
        ),
    )

def _add_selection(
    selected: list[tuple[int, float]],
    reasons: dict[tuple[int, float], list[str]],
    row: dict[str, Any] | None,
    reason: str,
) -> None:
    if row is None:
        return
    key = _candidate_key(row)
    reasons.setdefault(key, []).append(reason)
    if key not in selected:
        selected.append(key)

def _select_quality_top_k(stage1_rows: list[dict[str, Any]], k: int) -> tuple[list[tuple[int, float]], dict[tuple[int, float], list[str]]]:
    selected: list[tuple[int, float]] = []
    reasons: dict[tuple[int, float], list[str]] = {}
    for row in sorted(stage1_rows, key=_selection_sort_key)[:k]:
        _add_selection(selected, reasons, row, f"quality_top{k}")
    return selected, reasons

def _select_mixed_beam_v1(
    stage1_rows: list[dict[str, Any]],
    *,
    quality_k: int = 3,
) -> tuple[list[tuple[int, float]], dict[tuple[int, float], list[str]]]:
    selected, reasons = _select_quality_top_k(stage1_rows, quality_k)
    pressure_safe = min(stage1_rows, key=_pressure_sort_key) if stage1_rows else None
    _add_selection(selected, reasons, pressure_safe, "pressure_safe_top1")

    represented_randomness = {key[1] for key in selected}
    represented_seeds = {key[0] for key in selected}
    rescue_rows = sorted(
        stage1_rows,
        key=lambda row: (
            _candidate_key(row) in selected,
            float(row.get("randomness") or 0.0) in represented_randomness,
            int(row.get("seed") or 0) in represented_seeds,
            _selection_sort_key(row),
        ),
    )
    rescue = next((row for row in rescue_rows if _candidate_key(row) not in selected), None)
    _add_selection(selected, reasons, rescue, "diversity_rescue_top1")
    return selected, reasons

def _add_seed_family_rescue(
    selected: list[tuple[int, float]],
    reasons: dict[tuple[int, float], list[str]],
    stage1_rows: list[dict[str, Any]],
) -> None:
    represented_seeds = sorted({seed for seed, _randomness in selected})
    for seed in represented_seeds:
        rescue = next(
            (
                row
                for row in sorted(stage1_rows, key=_selection_sort_key)
                if int(row.get("seed") or 0) == seed
                and _candidate_key(row) not in selected
            ),
            None,
        )
        _add_selection(selected, reasons, rescue, "seed_family_rescue")

def _select_mixed_beam_v2(
    stage1_rows: list[dict[str, Any]],
) -> tuple[list[tuple[int, float]], dict[tuple[int, float], list[str]]]:
    selected, reasons = _select_mixed_beam_v1(stage1_rows, quality_k=5)
    _add_seed_family_rescue(selected, reasons, stage1_rows)
    return selected, reasons

def select_policy_candidates(
    policy_name: str,
    stage1_rows: list[dict[str, Any]],
) -> tuple[list[tuple[int, float]], dict[tuple[int, float], list[str]]]:
    if policy_name == "greedy_top1":
        return _select_quality_top_k(stage1_rows, 1)
    if policy_name == "quality_top5":
        return _select_quality_top_k(stage1_rows, 5)
    if policy_name == "mixed_beam_v1":
        return _select_mixed_beam_v1(stage1_rows)
    if policy_name == "mixed_beam_v2":
        return _select_mixed_beam_v2(stage1_rows)
    if policy_name == "lookahead_v1":
        return _select_quality_top_k(stage1_rows, 5)
    raise ValueError(f"Unknown policy: {policy_name}")

def _pressure_penalty_score(row: dict[str, Any] | None) -> float | None:
    if row is None:
        return None
    quality = _finite_float(row.get("quality"))
    if quality is None:
        return None
    pressure = _finite_float(row.get("max_doc_weight_ratio")) or 0.0
    n_above = _int_value(row.get("n_above_max_doc_weight")) or 0
    return quality - 1000.0 * max(0.0, pressure - 1.0) - 100.0 * n_above

def _candidate_ids(keys: Iterable[tuple[int, float]]) -> list[str]:
    return [_candidate_id_from_key(key) for key in keys]

def _reason_payload(reasons: dict[tuple[int, float], list[str]]) -> dict[str, str]:
    return {
        _candidate_id_from_key(key): "+".join(dict.fromkeys(value))
        for key, value in reasons.items()
    }

def _rows_for_keys(
    rows_by_candidate: dict[tuple[int, float], dict[str, Any]],
    keys: Iterable[tuple[int, float]],
) -> list[dict[str, Any]]:
    return [rows_by_candidate[key] for key in keys if key in rows_by_candidate]

def _quality_ratio(selected_quality: float | None, reference_quality: float | None) -> float | None:
    if selected_quality is None or reference_quality in (None, 0.0):
        return None
    return selected_quality / reference_quality

def _gap(reference_quality: float | None, selected_quality: float | None) -> float | None:
    if reference_quality is None or selected_quality is None:
        return None
    return reference_quality - selected_quality

def _delta(selected_value: float | None, reference_value: float | None) -> float | None:
    if selected_value is None or reference_value is None:
        return None
    return selected_value - reference_value

def _margin_convergence_polish_k(final_rows: list[dict[str, Any]]) -> int:
    ranked = sorted(final_rows, key=_selection_sort_key)
    if len(ranked) < 2:
        return 1
    top_quality = _finite_float(ranked[0].get("quality"))
    second_quality = _finite_float(ranked[1].get("quality"))
    if top_quality is None or second_quality is None:
        return 1
    absolute_gap = abs(top_quality - second_quality)
    relative_gap = absolute_gap / max(abs(top_quality), 1.0)
    if (
        absolute_gap <= MARGIN_POLISH_ABSOLUTE_GAP
        or relative_gap <= MARGIN_POLISH_RELATIVE_GAP
    ):
        return 2
    return 1

def _select_after_convergence_polish(
    *,
    final_rows: list[dict[str, Any]],
    convergence_by_candidate: dict[tuple[int, float], dict[str, Any]],
    convergence_polish_k: int,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], str]:
    ranked_final = sorted(final_rows, key=_selection_sort_key)
    selected_before_polish = ranked_final[0]
    if convergence_polish_k <= 0:
        return selected_before_polish, selected_before_polish, [], "iter10"
    polish_keys = [_candidate_key(row) for row in ranked_final[:convergence_polish_k]]
    polished_rows = [
        convergence_by_candidate[key]
        for key in polish_keys
        if key in convergence_by_candidate
    ]
    if not polished_rows:
        return selected_before_polish, selected_before_polish, [], "iter10"
    selected = _best_row(polished_rows) or selected_before_polish
    return selected, selected_before_polish, polished_rows, "convergence_polish"

def _branch_policy_row(
    *,
    sample: str,
    meta: dict[str, Any],
    policy_name: str,
    stage1_budget: str,
    stage2_budget: str,
    stage2_promote_k: int | str,
    final_budget: str,
    convergence_polish_k: int,
    retained_keys: list[tuple[int, float]],
    reasons: dict[tuple[int, float], list[str]],
    promoted_keys: list[tuple[int, float]],
    stage1_rows: list[dict[str, Any]],
    stage2_rows: list[dict[str, Any]],
    final_rows: list[dict[str, Any]],
    convergence_by_candidate: dict[tuple[int, float], dict[str, Any]],
    best10: dict[str, Any] | None,
    best_convergence: dict[str, Any] | None,
    full10_elapsed: float,
    full_convergence_elapsed: float,
) -> dict[str, Any]:
    selected, selected_before_polish, polished_rows, selection_stage = (
        _select_after_convergence_polish(
            final_rows=final_rows,
            convergence_by_candidate=convergence_by_candidate,
            convergence_polish_k=convergence_polish_k,
        )
    )
    selected_key = _candidate_key(selected)
    selected_before_key = _candidate_key(selected_before_polish)
    best10_key = None if best10 is None else _candidate_key(best10)
    best_convergence_key = (
        None if best_convergence is None else _candidate_key(best_convergence)
    )
    selected_quality = _finite_float(selected.get("quality"))
    selected_pressure = _finite_float(selected.get("max_doc_weight_ratio"))
    selected_above = _int_value(selected.get("n_above_max_doc_weight"))
    best10_quality = None if best10 is None else _finite_float(best10.get("quality"))
    best10_pressure = (
        None if best10 is None else _finite_float(best10.get("max_doc_weight_ratio"))
    )
    best_convergence_quality = (
        None
        if best_convergence is None
        else _finite_float(best_convergence.get("quality"))
    )
    best_convergence_pressure = (
        None
        if best_convergence is None
        else _finite_float(best_convergence.get("max_doc_weight_ratio"))
    )
    stage1_elapsed = _sum_elapsed(stage1_rows)
    stage2_elapsed = _sum_elapsed(stage2_rows)
    final_elapsed = _sum_elapsed(final_rows)
    polish_elapsed = _sum_elapsed(polished_rows)
    estimated_elapsed = stage1_elapsed + stage2_elapsed + final_elapsed + polish_elapsed
    polished_keys = [_candidate_key(row) for row in polished_rows]
    selected_budget = "convergence" if selection_stage == "convergence_polish" else "10"
    return {
        "sample": sample,
        **meta,
        "policy_name": policy_name,
        "stage1_budget": stage1_budget,
        "stage2_budget": stage2_budget,
        "stage2_promote_k": stage2_promote_k,
        "final_budget": final_budget,
        "polish_top1_convergence": convergence_polish_k >= 1,
        "convergence_polish_k": convergence_polish_k,
        "n_convergence_polished": len(polished_rows),
        "polished_candidate_ids": _candidate_ids(polished_keys),
        "selected_before_polish_candidate_id": _candidate_id_from_key(
            selected_before_key
        ),
        "selection_stage": selection_stage,
        "n_stage1_candidates": len(stage1_rows),
        "n_stage1_retained": len(retained_keys),
        "n_stage2_evaluated": len(stage2_rows),
        "n_final_evaluated": len(final_rows),
        "stage1_retained_candidate_ids": _candidate_ids(retained_keys),
        "stage1_retention_reasons": _reason_payload(reasons),
        "stage2_promoted_candidate_ids": _candidate_ids(promoted_keys),
        "selected_candidate_id": _candidate_id_from_key(selected_key),
        "selected_seed": selected_key[0],
        "selected_randomness": selected_key[1],
        "selected_budget": selected_budget,
        "selected_quality": selected_quality,
        "selected_max_doc_weight_ratio": selected_pressure,
        "selected_n_above_max_doc_weight": selected_above,
        "selected_pressure_penalty_score": _pressure_penalty_score(selected),
        "best10_candidate_id": (
            "" if best10_key is None else _candidate_id_from_key(best10_key)
        ),
        "best10_quality": best10_quality,
        "best10_max_doc_weight_ratio": best10_pressure,
        "best_convergence_candidate_id": (
            ""
            if best_convergence_key is None
            else _candidate_id_from_key(best_convergence_key)
        ),
        "best_convergence_quality": best_convergence_quality,
        "best_convergence_max_doc_weight_ratio": best_convergence_pressure,
        "quality_gap_to_best10": _gap(best10_quality, selected_quality),
        "quality_gap_to_best_convergence": _gap(
            best_convergence_quality, selected_quality
        ),
        "quality_recovery_ratio_vs_best10": _quality_ratio(
            selected_quality, best10_quality
        ),
        "quality_recovery_ratio_vs_best_convergence": _quality_ratio(
            selected_quality, best_convergence_quality
        ),
        "pressure_delta_vs_best10": _delta(selected_pressure, best10_pressure),
        "pressure_delta_vs_best_convergence": _delta(
            selected_pressure, best_convergence_pressure
        ),
        "retained_contains_best10": (
            best10_key in set(retained_keys) if best10_key is not None else None
        ),
        "promoted_contains_best10": (
            best10_key in set(promoted_keys) if best10_key is not None else None
        ),
        "selected_matches_best10_candidate": (
            selected_key == best10_key if best10_key is not None else None
        ),
        "selected_matches_best_convergence_candidate": (
            selected_key == best_convergence_key
            if best_convergence_key is not None
            else None
        ),
        "stage1_elapsed_sec": stage1_elapsed,
        "stage2_elapsed_sec": stage2_elapsed,
        "final_elapsed_sec": final_elapsed,
        "polish_elapsed_sec": polish_elapsed,
        "estimated_elapsed_proxy_sec": estimated_elapsed,
        "full10_elapsed_sec": full10_elapsed,
        "full_convergence_elapsed_sec": full_convergence_elapsed,
        "estimated_elapsed_saving_vs_full10_ratio": (
            1.0 - estimated_elapsed / full10_elapsed
            if full10_elapsed > 0.0
            else None
        ),
        "estimated_elapsed_saving_vs_full_convergence_ratio": (
            1.0 - estimated_elapsed / full_convergence_elapsed
            if full_convergence_elapsed > 0.0
            else None
        ),
    }

def simulate_branch_policies(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    early_policy_names = (
        "greedy_top1",
        "quality_top5",
        "mixed_beam_v1",
        "lookahead_v1",
        "mixed_beam_v2",
    )
    for sample, by_budget in sorted(_rows_by_sample_budget(rows).items()):
        iter5_rows = by_budget.get("5", [])
        iter10_rows = by_budget.get("10", [])
        convergence_rows = by_budget.get("convergence", [])
        if not iter10_rows:
            continue
        best10 = _best_row(iter10_rows)
        best_convergence = _best_row(convergence_rows) if convergence_rows else None
        iter5_by_candidate = _rows_by_candidate(iter5_rows)
        iter10_by_candidate = _rows_by_candidate(iter10_rows)
        convergence_by_candidate = _rows_by_candidate(convergence_rows)
        full10_elapsed = _sum_elapsed(iter10_rows)
        full_convergence_elapsed = _sum_elapsed(convergence_rows)
        meta = _sample_meta(iter10_rows)

        for stage1_budget in POLICY_STAGE1_BUDGETS:
            stage1_rows = by_budget.get(stage1_budget, [])
            if not stage1_rows:
                continue
            for policy_name in early_policy_names:
                retained_keys, reasons = select_policy_candidates(policy_name, stage1_rows)
                stage2_budget = ""
                stage2_promote_k: int | str = ""
                stage2_rows: list[dict[str, Any]] = []
                promoted_keys = list(retained_keys)
                if (
                    policy_name in {"mixed_beam_v1", "lookahead_v1", "mixed_beam_v2"}
                    and iter5_by_candidate
                ):
                    stage2_budget = "5"
                    stage2_rows = _rows_for_keys(iter5_by_candidate, retained_keys)
                    stage2_promote_k = min(3, len(stage2_rows))
                    promoted_keys = [
                        _candidate_key(row)
                        for row in sorted(stage2_rows, key=_selection_sort_key)[
                            : int(stage2_promote_k)
                        ]
                    ]

                final_rows = _rows_for_keys(iter10_by_candidate, promoted_keys)
                if not final_rows:
                    continue
                for convergence_polish_k in (0, 1):
                    output.append(
                        _branch_policy_row(
                            sample=sample,
                            meta=meta,
                            policy_name=policy_name,
                            stage1_budget=stage1_budget,
                            stage2_budget=stage2_budget,
                            stage2_promote_k=stage2_promote_k,
                            final_budget="10",
                            convergence_polish_k=convergence_polish_k,
                            retained_keys=retained_keys,
                            reasons=reasons,
                            promoted_keys=promoted_keys,
                            stage1_rows=stage1_rows,
                            stage2_rows=stage2_rows,
                            final_rows=final_rows,
                            convergence_by_candidate=convergence_by_candidate,
                            best10=best10,
                            best_convergence=best_convergence,
                            full10_elapsed=full10_elapsed,
                            full_convergence_elapsed=full_convergence_elapsed,
                        )
                    )

        if iter5_rows:
            iter5_ranked = sorted(iter5_rows, key=_selection_sort_key)
            retained_keys = [_candidate_key(row) for row in iter5_ranked]
            reasons = {
                key: ["iter5_screen_all"]
                for key in retained_keys
            }
            for policy_name, promote_k in (
                ("iter5_screen_all_top3", 3),
                ("iter5_screen_all_top5", 5),
                ("margin_polish_top2", 3),
            ):
                promoted_keys = [
                    _candidate_key(row)
                    for row in iter5_ranked[: min(promote_k, len(iter5_ranked))]
                ]
                final_rows = _rows_for_keys(iter10_by_candidate, promoted_keys)
                if not final_rows:
                    continue
                convergence_polish_k = (
                    _margin_convergence_polish_k(final_rows)
                    if policy_name == "margin_polish_top2"
                    else 1
                )
                output.append(
                    _branch_policy_row(
                        sample=sample,
                        meta=meta,
                        policy_name=policy_name,
                        stage1_budget="5",
                        stage2_budget="",
                        stage2_promote_k=promote_k,
                        final_budget="10",
                        convergence_polish_k=convergence_polish_k,
                        retained_keys=retained_keys,
                        reasons=reasons,
                        promoted_keys=promoted_keys,
                        stage1_rows=iter5_rows,
                        stage2_rows=[],
                        final_rows=final_rows,
                        convergence_by_candidate=convergence_by_candidate,
                        best10=best10,
                        best_convergence=best_convergence,
                        full10_elapsed=full10_elapsed,
                        full_convergence_elapsed=full_convergence_elapsed,
                    )
                )
    return output

def _best_policy_rows(policy_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_sample: dict[str, list[dict[str, Any]]] = {}
    for row in policy_rows:
        by_sample.setdefault(str(row.get("sample")), []).append(row)
    best_rows: list[dict[str, Any]] = []
    for sample, rows_for_sample in sorted(by_sample.items()):
        best_rows.append(
            min(
                rows_for_sample,
                key=lambda row: (
                    _sort_number(row.get("quality_gap_to_best10"), math.inf),
                    -(
                        _sort_number(
                            row.get("estimated_elapsed_saving_vs_full10_ratio"),
                            -math.inf,
                        )
                    ),
                    _sort_number(row.get("pressure_delta_vs_best10"), math.inf),
                    str(row.get("policy_name")),
                    str(row.get("stage1_budget")),
                ),
            )
        )
    return best_rows

def _best_v2_policy_rows(policy_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_sample: dict[str, list[dict[str, Any]]] = {}
    for row in policy_rows:
        if row.get("policy_name") in V2_POLICY_NAMES:
            by_sample.setdefault(str(row.get("sample")), []).append(row)
    best_rows: list[dict[str, Any]] = []
    for sample, rows_for_sample in sorted(by_sample.items()):
        best_rows.append(
            min(
                rows_for_sample,
                key=lambda row: (
                    _sort_number(row.get("quality_gap_to_best_convergence"), math.inf),
                    -(
                        _sort_number(
                            row.get("estimated_elapsed_saving_vs_full_convergence_ratio"),
                            -math.inf,
                        )
                    ),
                    _sort_number(row.get("pressure_delta_vs_best_convergence"), math.inf),
                    int(row.get("n_final_evaluated") or 0)
                    + int(row.get("n_convergence_polished") or 0),
                    str(row.get("policy_name")),
                ),
            )
        )
    return best_rows

def _sort_number(value: Any, default: float) -> float:
    number = _finite_float(value)
    return default if number is None else number

def _format_float(value: Any, digits: int = 6) -> str:
    number = _finite_float(value)
    return "" if number is None else f"{number:.{digits}f}"

def _first_top3_budget(rank_row: dict[str, Any]) -> str:
    for label in BUDGET_LABELS:
        prefix = _budget_prefix(label)
        rank = _int_value(rank_row.get(f"{prefix}_rank"))
        if rank is not None and rank <= 3:
            return label
    return ""

def _write_report(
    path: Path,
    *,
    input_csv: Path,
    rows: list[dict[str, Any]],
    rank_rows: list[dict[str, Any]],
    survival_rows: list[dict[str, Any]],
    late_rows: list[dict[str, Any]],
    policy_rows: list[dict[str, Any]],
) -> None:
    sample_count = len({row["sample"] for row in rows if row.get("supported")})
    candidate_count = len(
        {
            (row["sample"], row["seed"], row["randomness"])
            for row in rows
            if row.get("supported")
        }
    )
    lines = [
        "# Leiden Branch Lookahead Analysis",
        "",
        f"- Input rows: `{input_csv}`",
        f"- Supported profile rows: {sum(1 for row in rows if row.get('supported'))}",
        f"- Samples/layers: {sample_count}",
        f"- Branch candidates: {candidate_count}",
        "",
        "## Rank Flip Evidence",
        "",
        "| sample | layer | candidate | iter1_rank | iter5_rank | iter10_rank | convergence_rank | iter1_to_final_rank_delta | iter1_to_final_quality_delta |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    rank_by_sample: dict[str, list[dict[str, Any]]] = {}
    for row in rank_rows:
        rank_by_sample.setdefault(str(row.get("sample")), []).append(row)
    for sample, sample_rank_rows in sorted(rank_by_sample.items()):
        for row in sorted(
            sample_rank_rows,
            key=lambda item: (
                -abs(int(item.get("iter1_to_final_rank_delta") or 0)),
                _sort_number(item.get("final_rank"), math.inf),
                str(item.get("candidate_id")),
            ),
        )[:6]:
            lines.append(
                "| {sample} | {layer} | {candidate} | {r1} | {r5} | {r10} | {rc} | {rd} | {qd} |".format(
                    sample=sample,
                    layer=row.get("edge_layer", ""),
                    candidate=row.get("candidate_id", ""),
                    r1=row.get("iter1_rank", ""),
                    r5=row.get("iter5_rank", ""),
                    r10=row.get("iter10_rank", ""),
                    rc=row.get("convergence_rank", ""),
                    rd=row.get("iter1_to_final_rank_delta", ""),
                    qd=_format_float(row.get("iter1_to_final_quality_delta"), 3),
                )
            )

    lines.extend(
        [
            "",
            "## Greedy Failure Diagnosis",
            "",
            "| sample | layer | final_top1_candidate | iter1_rank | iter2_rank | iter3_rank | iter5_rank | iter10_rank | convergence_rank | first_top3_budget |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for sample, sample_rank_rows in sorted(rank_by_sample.items()):
        final_top1 = next(
            (
                row
                for row in sorted(sample_rank_rows, key=lambda item: _sort_number(item.get("final_rank"), math.inf))
                if _int_value(row.get("final_rank")) == 1
            ),
            None,
        )
        if final_top1 is None:
            continue
        lines.append(
            "| {sample} | {layer} | {candidate} | {r1} | {r2} | {r3} | {r5} | {r10} | {rc} | {first} |".format(
                sample=sample,
                layer=final_top1.get("edge_layer", ""),
                candidate=final_top1.get("candidate_id", ""),
                r1=final_top1.get("iter1_rank", ""),
                r2=final_top1.get("iter2_rank", ""),
                r3=final_top1.get("iter3_rank", ""),
                r5=final_top1.get("iter5_rank", ""),
                r10=final_top1.get("iter10_rank", ""),
                rc=final_top1.get("convergence_rank", ""),
                first=_first_top3_budget(final_top1),
            )
        )

    lines.extend(
        [
            "",
            "## Early Survival",
            "",
            "| early_budget | top_k | final_budget | top1_survival_rate | top3_full_survival_rate | mean_top3_hit_rate |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    aggregate_survival = [row for row in survival_rows if row.get("sample") == "__all__"]
    for row in aggregate_survival:
        lines.append(
            "| {early} | {top_k} | {final} | {top1} | {top3_full} | {top3_hit} |".format(
                early=row.get("early_budget", ""),
                top_k=row.get("top_k", ""),
                final=row.get("final_budget", ""),
                top1=_format_float(row.get("top1_survival_rate"), 3),
                top3_full=_format_float(row.get("top3_full_survival_rate"), 3),
                top3_hit=_format_float(row.get("mean_final_top3_hit_rate"), 3),
            )
        )

    lines.extend(
        [
            "",
            "## Late Risers And Fallers",
            "",
            "| sample | layer | class | early_budget | final_budget | candidate | early_rank | final_rank | quality_delta | pressure_delta |",
            "|---|---|---|---:|---:|---|---:|---:|---:|---:|",
        ]
    )
    late_by_sample: dict[str, list[dict[str, Any]]] = {}
    for row in late_rows:
        late_by_sample.setdefault(str(row.get("sample")), []).append(row)
    for sample, sample_late_rows in sorted(late_by_sample.items()):
        for row in sorted(
            sample_late_rows,
            key=lambda item: (
                _budget_sort_key(str(item.get("early_budget"))),
                _budget_sort_key(str(item.get("final_budget"))),
                str(item.get("classification")),
                -abs(int(item.get("rank_improvement") or 0)),
            ),
        )[:8]:
            lines.append(
                "| {sample} | {layer} | {classification} | {early} | {final} | {candidate} | {er} | {fr} | {qd} | {pd} |".format(
                    sample=sample,
                    layer=row.get("edge_layer", ""),
                    classification=row.get("classification", ""),
                    early=row.get("early_budget", ""),
                    final=row.get("final_budget", ""),
                    candidate=row.get("candidate_id", ""),
                    er=row.get("early_rank", ""),
                    fr=row.get("final_rank", ""),
                    qd=_format_float(row.get("quality_delta"), 3),
                    pd=_format_float(row.get("pressure_delta"), 6),
                )
            )

    lines.extend(
        [
            "",
            "## Policy Simulation",
            "",
            "| sample | layer | policy | stage1 | polish | selected | gap_to_best10 | pressure_delta_vs_best10 | elapsed_saving_vs_full10 |",
            "|---|---|---|---:|---:|---|---:|---:|---:|",
        ]
    )
    for row in _best_policy_rows(policy_rows):
        lines.append(
            "| {sample} | {layer} | {policy} | {stage1} | {polish} | {selected} | {gap} | {pressure} | {saving} |".format(
                sample=row.get("sample", ""),
                layer=row.get("edge_layer", ""),
                policy=row.get("policy_name", ""),
                stage1=row.get("stage1_budget", ""),
                polish=row.get("polish_top1_convergence", ""),
                selected=row.get("selected_candidate_id", ""),
                gap=_format_float(row.get("quality_gap_to_best10"), 3),
                pressure=_format_float(row.get("pressure_delta_vs_best10"), 6),
                saving=_format_float(row.get("estimated_elapsed_saving_vs_full10_ratio"), 3),
            )
        )

    lines.extend(
        [
            "",
            "## Best v2 Policies",
            "",
            "| sample | layer | policy | stage1 | promoted | polish_k | selected | gap_to_best_convergence | saving_vs_full_convergence | pressure_delta_vs_best_convergence |",
            "|---|---|---|---:|---:|---:|---|---:|---:|---:|",
        ]
    )
    for row in _best_v2_policy_rows(policy_rows):
        lines.append(
            "| {sample} | {layer} | {policy} | {stage1} | {promoted} | {polish_k} | {selected} | {gap} | {saving} | {pressure} |".format(
                sample=row.get("sample", ""),
                layer=row.get("edge_layer", ""),
                policy=row.get("policy_name", ""),
                stage1=row.get("stage1_budget", ""),
                promoted=row.get("stage2_promote_k", ""),
                polish_k=row.get("convergence_polish_k", ""),
                selected=row.get("selected_candidate_id", ""),
                gap=_format_float(row.get("quality_gap_to_best_convergence"), 3),
                saving=_format_float(
                    row.get("estimated_elapsed_saving_vs_full_convergence_ratio"),
                    3,
                ),
                pressure=_format_float(row.get("pressure_delta_vs_best_convergence"), 6),
            )
        )

    default_v2_rows = [
        row
        for row in policy_rows
        if row.get("policy_name") == "iter5_screen_all_top3"
    ]
    default_v2_by_sample: dict[str, list[dict[str, Any]]] = {}
    for row in default_v2_rows:
        default_v2_by_sample.setdefault(str(row.get("sample")), []).append(row)
    viable_default_samples = []
    for sample, rows_for_sample in sorted(default_v2_by_sample.items()):
        if any(
            _sort_number(row.get("quality_gap_to_best_convergence"), math.inf)
            <= 1.0e-9
            and _sort_number(
                row.get("estimated_elapsed_saving_vs_full_convergence_ratio"),
                -math.inf,
            )
            > 0.0
            for row in rows_for_sample
        ):
            viable_default_samples.append(sample)
    lines.extend(["", "## Recommendation", ""])
    if (
        len(viable_default_samples) == len(default_v2_by_sample)
        and default_v2_by_sample
    ):
        lines.append(
            "The default v2 recommendation is `iter5_screen_all_top3`: screen all candidates at iter5, promote the top3 to iter10, then polish the selected top1 at convergence."
        )
    elif viable_default_samples:
        lines.append(
            "`iter5_screen_all_top3` is the default v2 candidate policy, but this run only fully recovered best convergence quality with positive elapsed savings for: "
            + ", ".join(viable_default_samples)
            + "."
        )
    else:
        lines.append(
            "`iter5_screen_all_top3` is the default v2 candidate policy, but this run should be reviewed with `iter5_screen_all_top5` and `margin_polish_top2` fallbacks because it did not recover best convergence quality with positive elapsed savings in every case."
        )
    lines.append(
        "Selection uses quality first, with max-doc pressure, number of over-target clusters, and elapsed time only as tie-breakers."
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")

def analyze_branch_lookahead(
    *,
    input_csv: Path = DEFAULT_INPUT_CSV,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    top_k_values: tuple[int, ...] = (1, 2, 3, 5),
) -> dict[str, Any]:
    rows = load_profile_rows(input_csv)
    supported_rows = [row for row in rows if row.get("supported")]
    output_dir.mkdir(parents=True, exist_ok=True)

    rank_rows = build_rank_flip_rows(supported_rows)
    survival_rows = build_branch_survival_rows(
        supported_rows,
        top_k_values=top_k_values,
    )
    late_rows = build_late_riser_faller_rows(supported_rows)
    policy_rows = simulate_branch_policies(supported_rows)

    rank_path = output_dir / RANK_FLIP_FILENAME
    survival_path = output_dir / SURVIVAL_FILENAME
    late_path = output_dir / LATE_FLIP_FILENAME
    policy_path = output_dir / POLICY_FILENAME
    report_path = output_dir / REPORT_FILENAME
    summary_path = output_dir / SUMMARY_FILENAME

    _write_csv(rank_path, rank_rows, RANK_FLIP_FIELDS)
    _write_csv(survival_path, survival_rows, SURVIVAL_FIELDS)
    _write_csv(late_path, late_rows, LATE_FLIP_FIELDS)
    _write_csv(policy_path, policy_rows, POLICY_FIELDS)
    _write_report(
        report_path,
        input_csv=input_csv,
        rows=supported_rows,
        rank_rows=rank_rows,
        survival_rows=survival_rows,
        late_rows=late_rows,
        policy_rows=policy_rows,
    )

    payload = {
        "schema": f"leiden_branch_lookahead_analysis.v{SCHEMA_VERSION}",
        "input_csv": str(input_csv),
        "output_dir": str(output_dir),
        "n_supported_rows": len(supported_rows),
        "n_rank_flip_rows": len(rank_rows),
        "n_survival_rows": len(survival_rows),
        "n_late_flip_rows": len(late_rows),
        "n_policy_rows": len(policy_rows),
        "paths": {
            "rank_flip_by_candidate": str(rank_path),
            "survival_by_budget": str(survival_path),
            "late_riser_faller_summary": str(late_path),
            "policy_simulation": str(policy_path),
            "report": str(report_path),
            "summary": str(summary_path),
        },
    }
    _write_json(summary_path, payload)
    return payload

def _parse_top_k_values(value: str) -> tuple[int, ...]:
    items = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not items:
        raise ValueError("--top-k-values must contain at least one integer")
    if any(item <= 0 for item in items):
        raise ValueError("--top-k-values must be positive integers")
    return tuple(dict.fromkeys(items))

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze branch-local Leiden lookahead policies from iteration-budget profile rows.",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help=f"Profile rows CSV (default: {DEFAULT_INPUT_CSV})",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--top-k-values",
        default="1,2,3,5",
        help="Comma-separated early top-k values for survival analysis.",
    )
    return parser

def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    payload = analyze_branch_lookahead(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        top_k_values=_parse_top_k_values(args.top_k_values),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
