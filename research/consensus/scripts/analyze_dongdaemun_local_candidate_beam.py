"""Analyze local Dongdaemun refinement candidate beam policies.

This script works on opt-in Dongdaemun ``candidate_trace.jsonl`` files.  The
unit of analysis is not a full Leiden run.  It is one refinement parent inside
one run/depth, where the Rust implementation generates several local
candidates and currently collapses them to one applied choice.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


DEFAULT_TRACE_PATH = (
    Path("research/consensus/results/adaptive_refinement")
    / "dongdaemun_refinement_qs_profile"
    / "selective_benchmark_20260508"
    / "representative_sources_p4_c16_current_sp0_1_2_policy_core"
    / "candidate_trace.jsonl"
)
DEFAULT_RUNS_PATH = DEFAULT_TRACE_PATH.with_name("candidate_trace_runs.jsonl")
DEFAULT_OUTPUT_DIR = (
    Path("research/consensus/results/adaptive_refinement")
    / "dongdaemun_local_candidate_beam_analysis_20260511"
)

SCHEMA_VERSION = 1
POLICY_NAMES = (
    "current_greedy",
    "quality_top1",
    "pressure_first",
    "pressure_within_quality_band",
    "quality_top3_pressure",
    "mixed_local_beam_v1",
    "balanced_norm_v1",
    "seed_consensus_lite",
)
DEFAULT_GROUP_FIELDS = (
    "sample",
    "variant",
    "config_id",
    "gamma_preset",
    "seed_perturbations",
    "parent_selection_policy",
    "candidate_quality_policy",
    "adaptive_plateau_quality_band",
)

PARENT_POLICY_FILENAME = "local_candidate_beam_by_parent.csv"
POLICY_SUMMARY_FILENAME = "local_candidate_beam_policy_summary.csv"
MISSED_CASES_FILENAME = "local_candidate_beam_missed_cases.csv"
REPORT_FILENAME = "local_candidate_beam_report.md"
SUMMARY_FILENAME = "local_candidate_beam_summary.json"


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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


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


def _candidate_id(candidate: dict[str, Any]) -> int:
    return _int_value(candidate.get("candidate_id"))


def _candidate_label(candidate: dict[str, Any] | None) -> str | None:
    if candidate is None:
        return None
    return str(candidate.get("candidate_id"))


def _is_selectable(candidate: dict[str, Any]) -> bool:
    return _bool_value(candidate.get("valid")) and _bool_value(
        candidate.get("quality_passes")
    )


def _quality(candidate: dict[str, Any]) -> float:
    return _float_value(candidate.get("candidate_delta_q"), -math.inf)


def _pressure(candidate: dict[str, Any]) -> float:
    value = _finite_float(candidate.get("pressure_reduction"))
    if value is not None:
        return value
    return _float_value(candidate.get("largest_child_fraction_improvement"), -math.inf)


def _child_ratio(candidate: dict[str, Any]) -> float:
    value = _finite_float(candidate.get("candidate_max_child_weight_ratio"))
    if value is not None:
        return value
    return _largest_fraction(candidate)


def _largest_fraction(candidate: dict[str, Any]) -> float:
    return _float_value(candidate.get("largest_child_fraction"), math.inf)


def _singleton_fraction(candidate: dict[str, Any]) -> float:
    return _float_value(candidate.get("singleton_weight_fraction"), math.inf)


def _diagnostic_score(candidate: dict[str, Any]) -> float:
    return _float_value(candidate.get("adaptive_diagnostic_score"), -math.inf)


def _quality_sort_key(candidate: dict[str, Any]) -> tuple[float, ...]:
    return (
        -_quality(candidate),
        -_pressure(candidate),
        _child_ratio(candidate),
        _largest_fraction(candidate),
        _singleton_fraction(candidate),
        float(_candidate_id(candidate)),
    )


def _pressure_sort_key(candidate: dict[str, Any]) -> tuple[float, ...]:
    return (
        -_pressure(candidate),
        -_quality(candidate),
        _child_ratio(candidate),
        _largest_fraction(candidate),
        _singleton_fraction(candidate),
        float(_candidate_id(candidate)),
    )


def _diagnostic_sort_key(candidate: dict[str, Any]) -> tuple[float, ...]:
    return (
        -_diagnostic_score(candidate),
        -_quality(candidate),
        -_pressure(candidate),
        float(_candidate_id(candidate)),
    )


def _normalizer(values: list[float]):
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return lambda _value: 0.0
    minimum = min(finite_values)
    maximum = max(finite_values)
    if math.isclose(minimum, maximum):
        return lambda _value: 1.0
    return lambda value: (value - minimum) / (maximum - minimum)


def _balanced_norm_sort_key(
    candidates: list[dict[str, Any]], pressure_weight: float
):
    quality_norm = _normalizer([_quality(candidate) for candidate in candidates])
    pressure_norm = _normalizer([_pressure(candidate) for candidate in candidates])

    def sort_key(candidate: dict[str, Any]) -> tuple[float, ...]:
        score = quality_norm(_quality(candidate)) + pressure_weight * pressure_norm(
            _pressure(candidate)
        )
        return (
            -score,
            -_quality(candidate),
            -_pressure(candidate),
            _child_ratio(candidate),
            _singleton_fraction(candidate),
            float(_candidate_id(candidate)),
        )

    return sort_key


def _top_quality(candidates: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    return sorted(candidates, key=_quality_sort_key)[: max(0, k)]


def _top_pressure(candidates: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    return sorted(candidates, key=_pressure_sort_key)[: max(0, k)]


def _within_quality_band(
    candidates: list[dict[str, Any]], quality_band: float
) -> list[dict[str, Any]]:
    if not candidates:
        return []
    best_quality = max(_quality(candidate) for candidate in candidates)
    band = max(0.0, quality_band)
    return [
        candidate
        for candidate in candidates
        if _quality(candidate) >= best_quality - band
    ]


def _select_pressure_within_quality_band(
    candidates: list[dict[str, Any]], quality_band: float
) -> dict[str, Any] | None:
    in_band = _within_quality_band(candidates, quality_band)
    if not in_band:
        return None
    return sorted(in_band, key=_pressure_sort_key)[0]


@dataclass
class BeamChoice:
    selected: dict[str, Any] | None
    retained: list[dict[str, Any]] = field(default_factory=list)
    reasons: dict[int, list[str]] = field(default_factory=dict)

    def retained_ids(self) -> list[int]:
        return [_candidate_id(candidate) for candidate in self.retained]

    def reason_columns(self) -> dict[str, str]:
        return {
            str(candidate_id): "|".join(reasons)
            for candidate_id, reasons in sorted(self.reasons.items())
        }


@dataclass
class CandidateTraceGroup:
    run_id: str
    depth: int
    parent_id: int
    parent_visit_index: int
    profiles: list[dict[str, Any]]
    applied_candidate_id: int | None = None


def _choice_from_selected(selected: dict[str, Any] | None) -> BeamChoice:
    return BeamChoice(
        selected=selected,
        retained=[] if selected is None else [selected],
        reasons={} if selected is None else {_candidate_id(selected): ["selected"]},
    )


def _select_mixed_local_beam(
    candidates: list[dict[str, Any]],
    *,
    beam_width: int,
    quality_band: float,
) -> BeamChoice:
    beam_width = max(1, beam_width)
    reasons: dict[int, list[str]] = defaultdict(list)
    retained_by_id: dict[int, dict[str, Any]] = {}

    def add(candidate: dict[str, Any], reason: str) -> None:
        candidate_id = _candidate_id(candidate)
        retained_by_id.setdefault(candidate_id, candidate)
        if reason not in reasons[candidate_id]:
            reasons[candidate_id].append(reason)

    for candidate in _top_quality(candidates, 2):
        add(candidate, "quality_top2")
    for candidate in _top_pressure(candidates, 2):
        add(candidate, "pressure_top2")
    for candidate in sorted(candidates, key=_diagnostic_sort_key)[:1]:
        add(candidate, "diagnostic_top1")

    if len(retained_by_id) < beam_width:
        best_by_source: dict[str, dict[str, Any]] = {}
        for candidate in sorted(candidates, key=_quality_sort_key):
            source = str(candidate.get("source") or "")
            best_by_source.setdefault(source, candidate)
        for candidate in best_by_source.values():
            add(candidate, "source_rescue")
            if len(retained_by_id) >= beam_width:
                break

    retained = list(retained_by_id.values())
    if len(retained) > beam_width:
        retained = sorted(retained, key=_pressure_sort_key)[:beam_width]
        retained_ids = {_candidate_id(candidate) for candidate in retained}
        reasons = defaultdict(
            list,
            {
                candidate_id: value
                for candidate_id, value in reasons.items()
                if candidate_id in retained_ids
            },
        )

    selected = _select_pressure_within_quality_band(retained, quality_band)
    return BeamChoice(selected=selected, retained=retained, reasons=dict(reasons))


def _signature(candidate: dict[str, Any], precision: int) -> tuple[Any, ...]:
    return (
        _int_value(candidate.get("candidate_n_clusters")),
        round(_largest_fraction(candidate), precision),
        round(_singleton_fraction(candidate), precision),
    )


def _select_seed_consensus_lite(
    candidates: list[dict[str, Any]],
    *,
    quality_band: float,
    signature_precision: int,
) -> BeamChoice:
    retained = _within_quality_band(candidates, quality_band)
    if not retained:
        return BeamChoice(selected=None)

    by_signature: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for candidate in retained:
        by_signature[_signature(candidate, signature_precision)].append(candidate)

    def group_key(item: tuple[tuple[Any, ...], list[dict[str, Any]]]) -> tuple[float, ...]:
        _, group = item
        return (
            -float(len(group)),
            -max(_quality(candidate) for candidate in group),
            -max(_pressure(candidate) for candidate in group),
        )

    _, best_group = sorted(by_signature.items(), key=group_key)[0]
    selected = _select_pressure_within_quality_band(best_group, quality_band)
    reasons = {_candidate_id(candidate): ["consensus_band"] for candidate in retained}
    if selected is not None:
        reasons[_candidate_id(selected)].append("selected_signature")
    return BeamChoice(selected=selected, retained=retained, reasons=reasons)


def select_policy_candidate(
    policy_name: str,
    candidates: list[dict[str, Any]],
    *,
    current: dict[str, Any] | None,
    beam_width: int,
    quality_band: float,
    pressure_weight: float,
    signature_precision: int,
) -> BeamChoice:
    selectable = [candidate for candidate in candidates if _is_selectable(candidate)]
    if policy_name == "current_greedy":
        return _choice_from_selected(current)
    if not selectable:
        return BeamChoice(selected=None)
    if policy_name == "quality_top1":
        return _choice_from_selected(_top_quality(selectable, 1)[0])
    if policy_name == "pressure_first":
        return _choice_from_selected(_top_pressure(selectable, 1)[0])
    if policy_name == "pressure_within_quality_band":
        selected = _select_pressure_within_quality_band(selectable, quality_band)
        retained = _within_quality_band(selectable, quality_band)
        return BeamChoice(
            selected=selected,
            retained=retained,
            reasons={_candidate_id(candidate): ["quality_band"] for candidate in retained},
        )
    if policy_name == "quality_top3_pressure":
        retained = _top_quality(selectable, min(3, max(1, beam_width)))
        selected = _select_pressure_within_quality_band(retained, quality_band)
        return BeamChoice(
            selected=selected,
            retained=retained,
            reasons={_candidate_id(candidate): ["quality_top3"] for candidate in retained},
        )
    if policy_name == "mixed_local_beam_v1":
        return _select_mixed_local_beam(
            selectable,
            beam_width=beam_width,
            quality_band=quality_band,
        )
    if policy_name == "balanced_norm_v1":
        retained = sorted(
            selectable,
            key=_balanced_norm_sort_key(selectable, pressure_weight),
        )[: max(1, beam_width)]
        selected = retained[0]
        return BeamChoice(
            selected=selected,
            retained=retained,
            reasons={_candidate_id(candidate): ["balanced_norm"] for candidate in retained},
        )
    if policy_name == "seed_consensus_lite":
        return _select_seed_consensus_lite(
            selectable,
            quality_band=quality_band,
            signature_precision=signature_precision,
        )
    raise ValueError(f"Unknown policy: {policy_name}")


def _group_sort_key(group_key: tuple[str, int, int]) -> tuple[str, int, int]:
    run_id, depth, parent_id = group_key
    return (run_id, depth, parent_id)


def _load_run_metadata(runs_path: Path | None) -> dict[str, dict[str, Any]]:
    if runs_path is None:
        return {}
    return {
        str(row.get("run_id")): row
        for row in _read_jsonl(runs_path)
        if row.get("run_id")
    }


def _candidate_groups_from_events(events: list[dict[str, Any]]) -> list[CandidateTraceGroup]:
    groups: list[CandidateTraceGroup] = []
    visit_counts: dict[tuple[str, int, int], int] = defaultdict(int)
    current_key: tuple[str, int, int] | None = None
    current_visit_index = 0
    current_profiles: list[dict[str, Any]] = []
    current_applied_id: int | None = None
    last_candidate_id: int | None = None

    def finalize_current() -> None:
        nonlocal current_key
        nonlocal current_visit_index
        nonlocal current_profiles
        nonlocal current_applied_id
        nonlocal last_candidate_id
        if current_key is not None and current_profiles:
            run_id, depth, parent_id = current_key
            groups.append(
                CandidateTraceGroup(
                    run_id=run_id,
                    depth=depth,
                    parent_id=parent_id,
                    parent_visit_index=current_visit_index,
                    profiles=current_profiles,
                    applied_candidate_id=current_applied_id,
                )
            )
        current_key = None
        current_visit_index = 0
        current_profiles = []
        current_applied_id = None
        last_candidate_id = None

    def start_group(key: tuple[str, int, int]) -> None:
        nonlocal current_key
        nonlocal current_visit_index
        nonlocal current_profiles
        nonlocal current_applied_id
        nonlocal last_candidate_id
        visit_counts[key] += 1
        current_key = key
        current_visit_index = visit_counts[key]
        current_profiles = []
        current_applied_id = None
        last_candidate_id = None

    for event in events:
        event_name = event.get("event")
        if event_name not in {"candidate_profile", "candidate_decision"}:
            continue
        key = (
            str(event.get("run_id")),
            _int_value(event.get("depth")),
            _int_value(event.get("parent_id")),
        )
        if event_name == "candidate_profile":
            candidate_id = _int_value(event.get("candidate_id"))
            starts_new_block = (
                current_key is None
                or key != current_key
                or (
                    last_candidate_id is not None
                    and candidate_id <= last_candidate_id
                    and current_profiles
                )
            )
            if starts_new_block:
                finalize_current()
                start_group(key)
            current_profiles.append(event)
            last_candidate_id = candidate_id
            continue

        if event.get("decision") != "selected_applied":
            continue
        if current_key != key:
            finalize_current()
            start_group(key)
        current_applied_id = _int_value(event.get("candidate_id"))
        finalize_current()

    finalize_current()
    return groups


def _candidate_by_id(
    candidates: list[dict[str, Any]], candidate_id: int | None
) -> dict[str, Any] | None:
    if candidate_id is None:
        return None
    for candidate in candidates:
        if _candidate_id(candidate) == candidate_id:
            return candidate
    return None


def _fallback_current_candidate(
    candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    selected_profiles = [
        candidate
        for candidate in candidates
        if str(candidate.get("decision") or "") == "selected_by_policy"
    ]
    if not selected_profiles:
        return None
    return sorted(selected_profiles, key=lambda candidate: _candidate_id(candidate))[-1]


def _candidate_delta(
    selected: dict[str, Any] | None,
    current: dict[str, Any] | None,
    field: str,
) -> float | None:
    if selected is None or current is None:
        return None
    selected_value = _finite_float(selected.get(field))
    current_value = _finite_float(current.get(field))
    if selected_value is None or current_value is None:
        return None
    return selected_value - current_value


def _metric_delta(
    selected: dict[str, Any] | None,
    current: dict[str, Any] | None,
    metric,
) -> float | None:
    if selected is None or current is None:
        return None
    selected_value = metric(selected)
    current_value = metric(current)
    if not math.isfinite(selected_value) or not math.isfinite(current_value):
        return None
    return selected_value - current_value


def _selected_columns(prefix: str, candidate: dict[str, Any] | None) -> dict[str, Any]:
    if candidate is None:
        return {
            f"{prefix}_candidate_id": None,
            f"{prefix}_source": None,
            f"{prefix}_source_index": None,
            f"{prefix}_gamma_multiplier": None,
            f"{prefix}_repaired": None,
        f"{prefix}_candidate_delta_q": None,
        f"{prefix}_pressure_reduction": None,
        f"{prefix}_pressure_reduction_effective": None,
        f"{prefix}_candidate_max_child_weight_ratio": None,
        f"{prefix}_candidate_max_child_weight_ratio_effective": None,
        f"{prefix}_largest_child_fraction": None,
            f"{prefix}_singleton_weight_fraction": None,
            f"{prefix}_quotient_score": None,
            f"{prefix}_adaptive_diagnostic_score": None,
        }
    return {
        f"{prefix}_candidate_id": _candidate_id(candidate),
        f"{prefix}_source": candidate.get("source"),
        f"{prefix}_source_index": candidate.get("source_index"),
        f"{prefix}_gamma_multiplier": candidate.get("gamma_multiplier"),
        f"{prefix}_repaired": candidate.get("repaired"),
        f"{prefix}_candidate_delta_q": candidate.get("candidate_delta_q"),
        f"{prefix}_pressure_reduction": candidate.get("pressure_reduction"),
        f"{prefix}_pressure_reduction_effective": _pressure(candidate),
        f"{prefix}_candidate_max_child_weight_ratio": candidate.get(
            "candidate_max_child_weight_ratio"
        ),
        f"{prefix}_candidate_max_child_weight_ratio_effective": _child_ratio(candidate),
        f"{prefix}_largest_child_fraction": candidate.get("largest_child_fraction"),
        f"{prefix}_singleton_weight_fraction": candidate.get("singleton_weight_fraction"),
        f"{prefix}_quotient_score": candidate.get("quotient_score"),
        f"{prefix}_adaptive_diagnostic_score": candidate.get(
            "adaptive_diagnostic_score"
        ),
    }


def build_parent_policy_rows(
    *,
    events: list[dict[str, Any]],
    run_metadata: dict[str, dict[str, Any]],
    policy_names: Iterable[str] = POLICY_NAMES,
    beam_width: int = 5,
    quality_band: float = 1.0,
    pressure_weight: float = 0.35,
    signature_precision: int = 3,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups = _candidate_groups_from_events(events)
    groups = sorted(
        groups,
        key=lambda group: (
            _group_sort_key((group.run_id, group.depth, group.parent_id)),
            group.parent_visit_index,
        ),
    )
    for group in groups:
        run_id = group.run_id
        depth = group.depth
        parent_id = group.parent_id
        candidates = sorted(group.profiles, key=_candidate_id)
        selectable = [candidate for candidate in candidates if _is_selectable(candidate)]
        current = _candidate_by_id(candidates, group.applied_candidate_id)
        if current is None:
            current = _fallback_current_candidate(candidates)
        metadata = run_metadata.get(run_id, {})
        base_row: dict[str, Any] = {
            "run_id": run_id,
            "depth": depth,
            "parent_id": parent_id,
            "parent_visit_index": group.parent_visit_index,
            "parent_size": candidates[0].get("parent_size") if candidates else None,
            "parent_weight": candidates[0].get("parent_weight") if candidates else None,
            "standard_n_clusters": candidates[0].get("standard_n_clusters")
            if candidates
            else None,
            "n_candidates": len(candidates),
            "n_selectable_candidates": len(selectable),
            "selectable_candidate_ids": [_candidate_id(candidate) for candidate in selectable],
            "beam_width": beam_width,
            "quality_band": quality_band,
            "pressure_weight": pressure_weight,
            "signature_precision": signature_precision,
        }
        base_row.update(metadata)
        base_row.update(_selected_columns("current", current))

        for policy_name in policy_names:
            choice = select_policy_candidate(
                policy_name,
                candidates,
                current=current,
                beam_width=beam_width,
                quality_band=quality_band,
                pressure_weight=pressure_weight,
                signature_precision=signature_precision,
            )
            selected = choice.selected
            selected_id = _candidate_label(selected)
            current_id = _candidate_label(current)
            selected_differs = (
                selected_id is not None and current_id is not None and selected_id != current_id
            )
            row = {
                **base_row,
                "policy_name": policy_name,
                "n_retained_candidates": len(choice.retained),
                "retained_candidate_ids": choice.retained_ids(),
                "retention_reasons": choice.reason_columns(),
                "selected_differs_from_current": selected_differs,
                "candidate_delta_q_vs_current": _candidate_delta(
                    selected, current, "candidate_delta_q"
                ),
                "pressure_reduction_delta_vs_current": _metric_delta(
                    selected, current, _pressure
                ),
                "candidate_max_child_weight_ratio_delta_vs_current": _metric_delta(
                    selected, current, _child_ratio
                ),
                "largest_child_fraction_delta_vs_current": _candidate_delta(
                    selected, current, "largest_child_fraction"
                ),
                "singleton_weight_fraction_delta_vs_current": _candidate_delta(
                    selected, current, "singleton_weight_fraction"
                ),
            }
            row.update(_selected_columns("selected", selected))
            rows.append(row)
    return rows


@dataclass
class PolicyAggregate:
    n_parent_decisions: int = 0
    n_with_current: int = 0
    n_with_selected: int = 0
    n_differs_from_current: int = 0
    total_retained: int = 0
    quality_delta_sum: float = 0.0
    quality_delta_count: int = 0
    pressure_delta_sum: float = 0.0
    pressure_delta_count: int = 0
    ratio_delta_sum: float = 0.0
    ratio_delta_count: int = 0
    quality_improved: int = 0
    quality_lost: int = 0
    pressure_improved: int = 0
    pressure_worse: int = 0
    max_child_ratio_improved: int = 0
    max_child_ratio_worse: int = 0

    def add(self, row: dict[str, Any]) -> None:
        self.n_parent_decisions += 1
        self.total_retained += _int_value(row.get("n_retained_candidates"))
        if row.get("current_candidate_id") not in (None, ""):
            self.n_with_current += 1
        if row.get("selected_candidate_id") not in (None, ""):
            self.n_with_selected += 1
        if _bool_value(row.get("selected_differs_from_current")):
            self.n_differs_from_current += 1
        quality_delta = _finite_float(row.get("candidate_delta_q_vs_current"))
        if quality_delta is not None:
            self.quality_delta_sum += quality_delta
            self.quality_delta_count += 1
            if quality_delta > 0:
                self.quality_improved += 1
            elif quality_delta < 0:
                self.quality_lost += 1
        pressure_delta = _finite_float(row.get("pressure_reduction_delta_vs_current"))
        if pressure_delta is not None:
            self.pressure_delta_sum += pressure_delta
            self.pressure_delta_count += 1
            if pressure_delta > 0:
                self.pressure_improved += 1
            elif pressure_delta < 0:
                self.pressure_worse += 1
        ratio_delta = _finite_float(
            row.get("candidate_max_child_weight_ratio_delta_vs_current")
        )
        if ratio_delta is not None:
            self.ratio_delta_sum += ratio_delta
            self.ratio_delta_count += 1
            if ratio_delta < 0:
                self.max_child_ratio_improved += 1
            elif ratio_delta > 0:
                self.max_child_ratio_worse += 1

    def as_row(self) -> dict[str, Any]:
        return {
            "n_parent_decisions": self.n_parent_decisions,
            "n_with_current": self.n_with_current,
            "n_with_selected": self.n_with_selected,
            "n_differs_from_current": self.n_differs_from_current,
            "differs_from_current_rate": (
                self.n_differs_from_current / self.n_with_current
                if self.n_with_current
                else None
            ),
            "mean_retained_candidates": (
                self.total_retained / self.n_parent_decisions
                if self.n_parent_decisions
                else None
            ),
            "mean_candidate_delta_q_vs_current": (
                self.quality_delta_sum / self.quality_delta_count
                if self.quality_delta_count
                else None
            ),
            "mean_pressure_reduction_delta_vs_current": (
                self.pressure_delta_sum / self.pressure_delta_count
                if self.pressure_delta_count
                else None
            ),
            "mean_candidate_max_child_weight_ratio_delta_vs_current": (
                self.ratio_delta_sum / self.ratio_delta_count
                if self.ratio_delta_count
                else None
            ),
            "quality_improved_count": self.quality_improved,
            "quality_lost_count": self.quality_lost,
            "pressure_improved_count": self.pressure_improved,
            "pressure_worse_count": self.pressure_worse,
            "max_child_ratio_improved_count": self.max_child_ratio_improved,
            "max_child_ratio_worse_count": self.max_child_ratio_worse,
        }


def _metadata_group_key(
    row: dict[str, Any], group_fields: tuple[str, ...]
) -> tuple[Any, ...]:
    return tuple(row.get(field) for field in group_fields)


def build_policy_summary_rows(
    parent_policy_rows: list[dict[str, Any]],
    *,
    group_fields: tuple[str, ...] = DEFAULT_GROUP_FIELDS,
) -> list[dict[str, Any]]:
    aggregates: dict[tuple[Any, ...], PolicyAggregate] = defaultdict(PolicyAggregate)
    for row in parent_policy_rows:
        key = _metadata_group_key(row, group_fields) + (row.get("policy_name"),)
        aggregates[key].add(row)

    rows: list[dict[str, Any]] = []
    for key, aggregate in sorted(
        aggregates.items(),
        key=lambda item: tuple("" if value is None else str(value) for value in item[0]),
    ):
        row = dict(zip((*group_fields, "policy_name"), key))
        row.update(aggregate.as_row())
        rows.append(row)
    return rows


def build_overall_policy_summary(
    parent_policy_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    aggregates: dict[str, PolicyAggregate] = defaultdict(PolicyAggregate)
    for row in parent_policy_rows:
        aggregates[str(row.get("policy_name"))].add(row)
    rows: list[dict[str, Any]] = []
    for policy_name in POLICY_NAMES:
        aggregate = aggregates.get(policy_name)
        if aggregate is None:
            continue
        row = {"policy_name": policy_name}
        row.update(aggregate.as_row())
        rows.append(row)
    return rows


def build_missed_case_rows(parent_policy_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [
        row
        for row in parent_policy_rows
        if _bool_value(row.get("selected_differs_from_current"))
        and (
            _float_value(row.get("pressure_reduction_delta_vs_current"), 0.0) > 0.0
            or _float_value(row.get("candidate_delta_q_vs_current"), 0.0) > 0.0
            or _float_value(
                row.get("candidate_max_child_weight_ratio_delta_vs_current"), 0.0
            )
            < 0.0
        )
    ]
    return sorted(
        rows,
        key=lambda row: (
            -_float_value(row.get("candidate_delta_q_vs_current"), 0.0),
            -_float_value(row.get("pressure_reduction_delta_vs_current"), 0.0),
            _float_value(
                row.get("candidate_max_child_weight_ratio_delta_vs_current"), 0.0
            ),
        ),
    )


def _format_float(value: Any, digits: int = 4) -> str:
    number = _finite_float(value)
    return "" if number is None else f"{number:.{digits}f}"


def _build_report(
    *,
    trace_path: Path,
    runs_path: Path | None,
    parent_rows: list[dict[str, Any]],
    overall_summary_rows: list[dict[str, Any]],
    missed_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# Dongdaemun Local Candidate Beam Analysis",
        "",
        f"- Trace: `{trace_path}`",
        f"- Runs: `{runs_path}`" if runs_path is not None else "- Runs: none",
        f"- Parent-policy rows: {len(parent_rows)}",
        f"- Missed/alternative cases: {len(missed_rows)}",
        "- Primary objective: maximize parent-local CPM `candidate_delta_q`; pressure columns are diagnostics.",
        "",
        "## Overall Policy Summary",
        "",
        "| policy | parents | differs | mean dQ | mean pressure d | mean ratio d |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in overall_summary_rows:
        lines.append(
            "| {policy} | {parents} | {differs} | {dq} | {pressure} | {ratio} |".format(
                policy=row["policy_name"],
                parents=row["n_parent_decisions"],
                differs=_format_float(row.get("differs_from_current_rate"), 3),
                dq=_format_float(row.get("mean_candidate_delta_q_vs_current"), 3),
                pressure=_format_float(
                    row.get("mean_pressure_reduction_delta_vs_current"), 4
                ),
                ratio=_format_float(
                    row.get("mean_candidate_max_child_weight_ratio_delta_vs_current"),
                    4,
                ),
            )
        )

    lines.extend(
        [
            "",
            "## Top Alternative Cases",
            "",
            "| policy | run | depth | parent | current | selected | dQ | pressure d | ratio d |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in missed_rows[:20]:
        lines.append(
            "| {policy} | {run} | {depth} | {parent} | {current} | {selected} | {dq} | {pressure} | {ratio} |".format(
                policy=row["policy_name"],
                run=row["run_id"],
                depth=row["depth"],
                parent=row["parent_id"],
                current=row.get("current_candidate_id"),
                selected=row.get("selected_candidate_id"),
                dq=_format_float(row.get("candidate_delta_q_vs_current"), 3),
                pressure=_format_float(
                    row.get("pressure_reduction_delta_vs_current"), 4
                ),
                ratio=_format_float(
                    row.get("candidate_max_child_weight_ratio_delta_vs_current"), 4
                ),
            )
        )
    lines.append("")
    return "\n".join(lines)


def analyze_candidate_trace(
    *,
    trace_path: Path,
    runs_path: Path | None,
    output_dir: Path,
    policy_names: tuple[str, ...] = POLICY_NAMES,
    group_fields: tuple[str, ...] = DEFAULT_GROUP_FIELDS,
    beam_width: int = 5,
    quality_band: float = 1.0,
    pressure_weight: float = 0.35,
    signature_precision: int = 3,
) -> dict[str, Any]:
    events = _read_jsonl(trace_path)
    run_metadata = _load_run_metadata(runs_path)
    parent_rows = build_parent_policy_rows(
        events=events,
        run_metadata=run_metadata,
        policy_names=policy_names,
        beam_width=beam_width,
        quality_band=quality_band,
        pressure_weight=pressure_weight,
        signature_precision=signature_precision,
    )
    policy_summary_rows = build_policy_summary_rows(
        parent_rows,
        group_fields=group_fields,
    )
    overall_summary_rows = build_overall_policy_summary(parent_rows)
    missed_rows = build_missed_case_rows(parent_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    parent_path = output_dir / PARENT_POLICY_FILENAME
    policy_summary_path = output_dir / POLICY_SUMMARY_FILENAME
    missed_path = output_dir / MISSED_CASES_FILENAME
    report_path = output_dir / REPORT_FILENAME
    summary_path = output_dir / SUMMARY_FILENAME

    _write_csv(parent_path, parent_rows)
    _write_csv(policy_summary_path, policy_summary_rows)
    _write_csv(missed_path, missed_rows)
    report_path.write_text(
        _build_report(
            trace_path=trace_path,
            runs_path=runs_path,
            parent_rows=parent_rows,
            overall_summary_rows=overall_summary_rows,
            missed_rows=missed_rows,
        ),
        encoding="utf-8",
    )

    payload = {
        "schema": "dongdaemun_local_candidate_beam_analysis.v1",
        "schema_version": SCHEMA_VERSION,
        "trace_path": str(trace_path),
        "runs_path": None if runs_path is None else str(runs_path),
        "beam_width": beam_width,
        "quality_band": quality_band,
        "pressure_weight": pressure_weight,
        "signature_precision": signature_precision,
        "policy_names": list(policy_names),
        "group_fields": list(group_fields),
        "n_events": len(events),
        "n_parent_policy_rows": len(parent_rows),
        "n_policy_summary_rows": len(policy_summary_rows),
        "n_missed_case_rows": len(missed_rows),
        "overall_policy_summary": overall_summary_rows,
        "paths": {
            "parent_policy": str(parent_path),
            "policy_summary": str(policy_summary_path),
            "missed_cases": str(missed_path),
            "report": str(report_path),
            "summary": str(summary_path),
        },
    }
    _write_json(summary_path, payload)
    return payload


def _parse_policy_names(value: str | None) -> tuple[str, ...]:
    if value is None:
        return POLICY_NAMES
    names = tuple(item.strip() for item in value.split(",") if item.strip())
    if not names:
        raise ValueError("--policies must contain at least one policy")
    unknown = [name for name in names if name not in POLICY_NAMES]
    if unknown:
        raise ValueError(f"Unknown policies: {', '.join(unknown)}")
    return names


def _parse_group_fields(value: str | None) -> tuple[str, ...]:
    if value is None:
        return DEFAULT_GROUP_FIELDS
    fields = tuple(item.strip() for item in value.split(",") if item.strip())
    if not fields:
        raise ValueError("--group-fields must contain at least one field")
    return fields


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE_PATH)
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--quality-band", type=float, default=1.0)
    parser.add_argument("--pressure-weight", type=float, default=0.35)
    parser.add_argument("--signature-precision", type=int, default=3)
    parser.add_argument(
        "--policies",
        help=f"Comma-separated subset of policies. Default: {', '.join(POLICY_NAMES)}.",
    )
    parser.add_argument(
        "--group-fields",
        help=(
            "Comma-separated metadata fields for policy summary grouping. "
            f"Default: {', '.join(DEFAULT_GROUP_FIELDS)}."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    runs_path = args.runs if args.runs and args.runs.exists() else None
    payload = analyze_candidate_trace(
        trace_path=args.trace,
        runs_path=runs_path,
        output_dir=args.output_dir,
        policy_names=_parse_policy_names(args.policies),
        group_fields=_parse_group_fields(args.group_fields),
        beam_width=args.beam_width,
        quality_band=args.quality_band,
        pressure_weight=args.pressure_weight,
        signature_precision=args.signature_precision,
    )
    print(f"Saved local candidate beam analysis to {payload['paths']['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
