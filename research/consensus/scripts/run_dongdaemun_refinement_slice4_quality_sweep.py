"""Quality-aware Slice 4 sweep for integrated Dongdaemun refinement.

The baseline Slice 4 pilot validates that the integrated refinement path runs
on prepared source-level graphs.  This runner keeps max-weight diagnostics, but
uses CPM objective quality as the primary signal:

- recompute every returned membership with ``graph.cpm_quality`` as a sanity
  check against the reported Rust quality;
- run standard Leiden once per prepared summary;
- sweep refinement settings over gamma-multiplier presets,
  seed perturbation counts, and repair off/on variants;
- report whether any refinement row reaches a better local optimum than the
  standard row from the same graph/gamma/seed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT))

import evaluate_dongdaemun_refinement_slice4 as pilot  # noqa: E402


DEFAULT_OUTPUT_DIR = (
    Path("research/consensus/results/adaptive_refinement")
    / "dongdaemun_refinement_slice4_pilot"
    / "field34_quality_sweep"
)
QUALITY_RECOMPUTE_ABS_TOL = 1.0e-6
SCHEMA_VERSION = 1
CHECKPOINT_ROWS_FILENAME = "slice4_quality_sweep_rows.jsonl"
PROGRESS_FILENAME = "slice4_quality_sweep_progress.json"
CANDIDATE_TRACE_FILENAME = "candidate_trace.jsonl"
CANDIDATE_TRACE_RUNS_FILENAME = "candidate_trace_runs.jsonl"
CANDIDATE_TRACE_PATH_ENV = "SCISCAPE_DDM_CANDIDATE_TRACE_PATH"
CANDIDATE_TRACE_RUN_ID_ENV = "SCISCAPE_DDM_CANDIDATE_TRACE_RUN_ID"
CANDIDATE_TRACE_EPOCH_ENV = "SCISCAPE_DDM_CANDIDATE_TRACE_EPOCH"
QUALITY_TRACE_FILENAME = "quality_trace.jsonl"
QUALITY_TRACE_RUNS_FILENAME = "quality_trace_runs.jsonl"
QUALITY_TRACE_PATH_ENV = "SCISCAPE_DDM_QUALITY_TRACE_PATH"
QUALITY_TRACE_RUN_ID_ENV = "SCISCAPE_DDM_QUALITY_TRACE_RUN_ID"
QUALITY_TRACE_EPOCH_ENV = "SCISCAPE_DDM_QUALITY_TRACE_EPOCH"

GAMMA_PRESETS: dict[str, tuple[float, ...]] = {
    "mild": (1.02, 1.05),
    "current": (1.02, 1.05, 1.10, 1.15, 1.20, 1.25),
    "aggressive": (1.10, 1.25, 1.50, 2.00),
}
DEFAULT_SEED_PERTURBATIONS = (0, 1, 2, 4)
DEFAULT_CANDIDATE_QUALITY_POLICIES = ("structural",)
DEFAULT_PARENT_SELECTION_POLICIES = ("weight",)

SWEEP_FIELDS = [
    "sample",
    "seed",
    "summary_path",
    "config_id",
    "gamma_preset",
    "gamma_multipliers",
    "seed_perturbations",
    "max_extra_parents_per_iteration",
    "max_extra_children_per_parent",
    "parent_selection_policy",
    "candidate_quality_policy",
    "min_candidate_delta_q",
    "adaptive_plateau_quality_band",
    "use_final_quality_guard",
    "min_final_quality_delta",
    "use_baseline_repair",
    "candidate_trace_run_id",
    "quality_trace_run_id",
    "baseline_repair_policy",
    "baseline_repair_replace_min_parent_ratio",
    "auto_fast_trigger_max_doc_weight_ratio",
    "auto_fast_trigger_min_above_max_doc_weight",
    "auto_fast_accept_max_doc_weight_ratio",
    "auto_fast_accept_min_quality_delta",
    "auto_fast_accept_min_quality_delta_ratio",
    "auto_fast_triggered",
    "auto_fast_fallback_triggered",
    "auto_fast_fallback_reason",
    *pilot.CSV_FIELDS,
    "quality_recomputed",
    "quality_recompute_delta",
    "quality_recompute_abs_delta",
    "quality_recompute_ok",
]


@dataclass(frozen=True)
class SweepConfig:
    config_id: str
    gamma_preset: str
    gamma_multipliers: tuple[float, ...]
    seed_perturbations: int
    max_extra_parents_per_iteration: int = 16
    max_extra_children_per_parent: int = 64
    parent_selection_policy: str = "weight"
    candidate_quality_policy: str = "structural"
    min_candidate_delta_q: float = 0.0
    adaptive_plateau_quality_band: float = 0.0
    use_final_quality_guard: bool = False
    min_final_quality_delta: float = 0.0
    baseline_repair_policy: str = "replace"
    baseline_repair_replace_min_parent_ratio: float = 1.05
    auto_fast_trigger_max_doc_weight_ratio: float | None = None
    auto_fast_trigger_min_above_max_doc_weight: int | None = None
    auto_fast_accept_max_doc_weight_ratio: float | None = None
    auto_fast_accept_min_quality_delta: float | None = None
    auto_fast_accept_min_quality_delta_ratio: float | None = None


def _json_safe(value: Any) -> Any:
    return pilot._json_safe(value)


def _csv_value(value: Any) -> Any:
    safe = _json_safe(value)
    if safe is None:
        return ""
    if isinstance(safe, (list, dict)):
        return json.dumps(safe, sort_keys=True, separators=(",", ":"))
    return safe


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _summary_ref(input_cfg: pilot.Slice4Input) -> str | None:
    value = pilot._rel(input_cfg.summary_path)
    return None if value is None else str(value)


def _row_key_from_parts(
    *,
    summary_path: str | None,
    seed: int,
    config_id: str,
    variant: str,
    use_baseline_repair: bool | None,
) -> str:
    return json.dumps(
        [
            summary_path,
            int(seed),
            str(config_id),
            str(variant),
            use_baseline_repair,
        ],
        separators=(",", ":"),
    )


def _row_key(row: dict[str, Any]) -> str:
    return _row_key_from_parts(
        summary_path=None
        if row.get("summary_path") is None
        else str(row.get("summary_path")),
        seed=int(row.get("seed")),
        config_id=str(row.get("config_id")),
        variant=str(row.get("variant")),
        use_baseline_repair=row.get("use_baseline_repair"),
    )


def _input_row_key(
    *,
    input_cfg: pilot.Slice4Input,
    config_id: str,
    variant: str,
    use_baseline_repair: bool | None,
) -> str:
    return _row_key_from_parts(
        summary_path=_summary_ref(input_cfg),
        seed=int(input_cfg.seed),
        config_id=config_id,
        variant=variant,
        use_baseline_repair=use_baseline_repair,
    )


def _candidate_trace_run_id(row_key: str) -> str:
    return hashlib.sha1(row_key.encode("utf-8")).hexdigest()[:16]


def _candidate_trace_run_id_for_input(
    *,
    input_cfg: pilot.Slice4Input,
    config_id: str,
    variant: str,
    use_baseline_repair: bool,
) -> str:
    return _candidate_trace_run_id(
        _input_row_key(
            input_cfg=input_cfg,
            config_id=config_id,
            variant=variant,
            use_baseline_repair=use_baseline_repair,
        )
    )


def _candidate_trace_run_metadata(
    *,
    input_cfg: pilot.Slice4Input,
    sweep_config: SweepConfig,
    variant: str,
    use_baseline_repair: bool,
    run_id: str,
) -> dict[str, Any]:
    row_key = _input_row_key(
        input_cfg=input_cfg,
        config_id=sweep_config.config_id,
        variant=variant,
        use_baseline_repair=use_baseline_repair,
    )
    return {
        "schema": f"dongdaemun_refinement_candidate_trace_run.v{SCHEMA_VERSION}",
        "run_id": run_id,
        "row_key": row_key,
        "sample": input_cfg.sample,
        "seed": int(input_cfg.seed),
        "summary_path": _summary_ref(input_cfg),
        "variant": variant,
        "use_baseline_repair": bool(use_baseline_repair),
        "config_id": sweep_config.config_id,
        "gamma_preset": sweep_config.gamma_preset,
        "gamma_multipliers": [float(x) for x in sweep_config.gamma_multipliers],
        "seed_perturbations": int(sweep_config.seed_perturbations),
        "max_extra_parents_per_iteration": int(
            sweep_config.max_extra_parents_per_iteration
        ),
        "max_extra_children_per_parent": int(
            sweep_config.max_extra_children_per_parent
        ),
        "parent_selection_policy": sweep_config.parent_selection_policy,
        "candidate_quality_policy": sweep_config.candidate_quality_policy,
        "min_candidate_delta_q": float(sweep_config.min_candidate_delta_q),
        "adaptive_plateau_quality_band": float(
            sweep_config.adaptive_plateau_quality_band
        ),
        "use_final_quality_guard": bool(sweep_config.use_final_quality_guard),
        "min_final_quality_delta": float(sweep_config.min_final_quality_delta),
        "baseline_repair_policy": sweep_config.baseline_repair_policy,
        "baseline_repair_replace_min_parent_ratio": float(
            sweep_config.baseline_repair_replace_min_parent_ratio
        ),
    }


def _quality_trace_run_metadata(
    *,
    input_cfg: pilot.Slice4Input,
    sweep_config: SweepConfig,
    variant: str,
    use_baseline_repair: bool,
    run_id: str,
) -> dict[str, Any]:
    payload = _candidate_trace_run_metadata(
        input_cfg=input_cfg,
        sweep_config=sweep_config,
        variant=variant,
        use_baseline_repair=use_baseline_repair,
        run_id=run_id,
    )
    payload["schema"] = f"dongdaemun_refinement_quality_trace_run.v{SCHEMA_VERSION}"
    return payload


def _append_candidate_trace_run_metadata(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":")))
        fh.write("\n")


def _append_quality_trace_run_metadata(path: Path, payload: dict[str, Any]) -> None:
    _append_candidate_trace_run_metadata(path, payload)


@contextmanager
def _candidate_trace_context(run_id: str | None):
    previous = os.environ.get(CANDIDATE_TRACE_RUN_ID_ENV)
    if run_id is None:
        os.environ.pop(CANDIDATE_TRACE_RUN_ID_ENV, None)
    else:
        os.environ[CANDIDATE_TRACE_RUN_ID_ENV] = run_id
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(CANDIDATE_TRACE_RUN_ID_ENV, None)
        else:
            os.environ[CANDIDATE_TRACE_RUN_ID_ENV] = previous


@contextmanager
def _quality_trace_context(run_id: str | None):
    previous = os.environ.get(QUALITY_TRACE_RUN_ID_ENV)
    if run_id is None:
        os.environ.pop(QUALITY_TRACE_RUN_ID_ENV, None)
    else:
        os.environ[QUALITY_TRACE_RUN_ID_ENV] = run_id
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(QUALITY_TRACE_RUN_ID_ENV, None)
        else:
            os.environ[QUALITY_TRACE_RUN_ID_ENV] = previous


@contextmanager
def _candidate_trace_path_context(
    path: Path | None,
    *,
    explicit: bool,
    resume: bool,
):
    previous_path = os.environ.get(CANDIDATE_TRACE_PATH_ENV)
    previous_epoch = os.environ.get(CANDIDATE_TRACE_EPOCH_ENV)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
    if path is not None and explicit:
        if not resume and path.exists():
            path.unlink()
        os.environ[CANDIDATE_TRACE_PATH_ENV] = str(path)
        os.environ[CANDIDATE_TRACE_EPOCH_ENV] = uuid.uuid4().hex
    try:
        yield
    finally:
        if not explicit:
            return
        if previous_path is None:
            os.environ.pop(CANDIDATE_TRACE_PATH_ENV, None)
        else:
            os.environ[CANDIDATE_TRACE_PATH_ENV] = previous_path
        if previous_epoch is None:
            os.environ.pop(CANDIDATE_TRACE_EPOCH_ENV, None)
        else:
            os.environ[CANDIDATE_TRACE_EPOCH_ENV] = previous_epoch


@contextmanager
def _quality_trace_path_context(
    path: Path | None,
    *,
    explicit: bool,
    resume: bool,
):
    previous_path = os.environ.get(QUALITY_TRACE_PATH_ENV)
    previous_epoch = os.environ.get(QUALITY_TRACE_EPOCH_ENV)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
    if path is not None and explicit:
        if not resume and path.exists():
            path.unlink()
        os.environ[QUALITY_TRACE_PATH_ENV] = str(path)
        os.environ[QUALITY_TRACE_EPOCH_ENV] = uuid.uuid4().hex
    try:
        yield
    finally:
        if not explicit:
            return
        if previous_path is None:
            os.environ.pop(QUALITY_TRACE_PATH_ENV, None)
        else:
            os.environ[QUALITY_TRACE_PATH_ENV] = previous_path
        if previous_epoch is None:
            os.environ.pop(QUALITY_TRACE_EPOCH_ENV, None)
        else:
            os.environ[QUALITY_TRACE_EPOCH_ENV] = previous_epoch


def _checkpoint_path(output_dir: Path) -> Path:
    return output_dir / CHECKPOINT_ROWS_FILENAME


def _progress_path(output_dir: Path) -> Path:
    return output_dir / PROGRESS_FILENAME


def _standard_membership_cache_path(
    output_dir: Path,
    input_cfg: pilot.Slice4Input,
    run_config: pilot.Slice4RunConfig,
) -> Path:
    fingerprint = json.dumps(
        {
            "summary_path": _summary_ref(input_cfg),
            "seed": int(input_cfg.seed),
            "resolution": float(input_cfg.resolution),
            "n_iterations": int(run_config.n_iterations),
            "randomness": float(run_config.randomness),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()[:16]
    return output_dir / "membership_cache" / f"{digest}_standard.npy"


def _load_checkpoint_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows_by_key: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows_by_key
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows_by_key[_row_key(row)] = row
    return rows_by_key


def _append_checkpoint_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(
                json.dumps(
                    _json_safe(row),
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            fh.write("\n")


def _planned_row_keys(
    *,
    input_cfg: pilot.Slice4Input,
    sweep_configs: list[SweepConfig],
) -> list[str]:
    keys = [
        _input_row_key(
            input_cfg=input_cfg,
            config_id="standard",
            variant=pilot.VARIANT_STANDARD,
            use_baseline_repair=None,
        )
    ]
    for sweep_config in sweep_configs:
        keys.extend(
            [
                _input_row_key(
                    input_cfg=input_cfg,
                    config_id=sweep_config.config_id,
                    variant=pilot.VARIANT_REPAIR_OFF,
                    use_baseline_repair=False,
                ),
                _input_row_key(
                    input_cfg=input_cfg,
                    config_id=sweep_config.config_id,
                    variant=pilot.VARIANT_REPAIR_ON,
                    use_baseline_repair=True,
                ),
            ]
        )
    return keys


def _ordered_rows_from_checkpoint(
    *,
    summary_paths: list[Path],
    sweep_configs: list[SweepConfig],
    rows_by_key: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for summary_path in summary_paths:
        input_cfg = pilot._resolve_input_from_summary(summary_path)
        for key in _planned_row_keys(input_cfg=input_cfg, sweep_configs=sweep_configs):
            row = rows_by_key.get(key)
            if row is not None:
                rows.append(row)
                seen.add(key)
    for key, row in rows_by_key.items():
        if key not in seen:
            rows.append(row)
    return rows


def _parse_int_tuple(value: str | None) -> tuple[int, ...]:
    if value is None:
        return DEFAULT_SEED_PERTURBATIONS
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _parse_gamma_presets(value: str | None) -> tuple[str, ...]:
    if value is None:
        return tuple(GAMMA_PRESETS)
    names = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = [name for name in names if name not in GAMMA_PRESETS]
    if unknown:
        raise ValueError(
            "unknown gamma preset(s): "
            + ", ".join(unknown)
            + f"; available: {', '.join(GAMMA_PRESETS)}"
        )
    return names


def _parse_candidate_quality_policies(value: str | None) -> tuple[str, ...]:
    if value is None:
        return DEFAULT_CANDIDATE_QUALITY_POLICIES
    policies = tuple(part.strip() for part in value.split(",") if part.strip())
    allowed = {
        "structural",
        "quality_guarded_structural",
        "quality_floor",
        "quality_first",
        "selective",
        "pressure_aware",
        "adaptive_plateau",
    }
    unknown = [policy for policy in policies if policy not in allowed]
    if unknown:
        raise ValueError(
            "unknown candidate quality policy/policies: "
            + ", ".join(unknown)
            + f"; available: {', '.join(sorted(allowed))}"
        )
    return policies


def _parse_baseline_repair_policies(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ("replace",)
    policies = tuple(part.strip() for part in value.split(",") if part.strip())
    allowed = {"replace", "augment", "adaptive"}
    unknown = [policy for policy in policies if policy not in allowed]
    if unknown:
        raise ValueError(
            "unknown baseline repair policy/policies: "
            + ", ".join(unknown)
            + f"; available: {', '.join(sorted(allowed))}"
        )
    return policies


def _parse_parent_selection_policies(value: str | None) -> tuple[str, ...]:
    if value is None:
        return DEFAULT_PARENT_SELECTION_POLICIES
    policies = tuple(part.strip() for part in value.split(",") if part.strip())
    allowed = {"weight", "pressure_boundary"}
    unknown = [policy for policy in policies if policy not in allowed]
    if unknown:
        raise ValueError(
            "unknown parent selection policy/policies: "
            + ", ".join(unknown)
            + f"; available: {', '.join(sorted(allowed))}"
        )
    return policies


def _parse_float_tuple(value: str | None, default: tuple[float, ...]) -> tuple[float, ...]:
    if value is None:
        return default
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def _format_float_suffix(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def _build_sweep_configs(
    *,
    gamma_presets: tuple[str, ...] = tuple(GAMMA_PRESETS),
    seed_perturbations: tuple[int, ...] = DEFAULT_SEED_PERTURBATIONS,
    candidate_quality_policies: tuple[str, ...] = DEFAULT_CANDIDATE_QUALITY_POLICIES,
    parent_selection_policies: tuple[str, ...] = DEFAULT_PARENT_SELECTION_POLICIES,
    min_candidate_delta_q: float = 0.0,
    adaptive_plateau_quality_band: float | None = None,
    adaptive_plateau_quality_bands: tuple[float, ...] | None = None,
    use_final_quality_guard: bool = False,
    min_final_quality_delta: float = 0.0,
    baseline_repair_policies: tuple[str, ...] = ("replace",),
    baseline_repair_replace_min_parent_ratio: float = 1.05,
    max_extra_parents_per_iteration: int = 16,
    max_extra_children_per_parent: int = 64,
    auto_fast_trigger_max_doc_weight_ratio: float | None = None,
    auto_fast_trigger_min_above_max_doc_weight: int | None = None,
    auto_fast_accept_max_doc_weight_ratio: float | None = None,
    auto_fast_accept_min_quality_delta: float | None = None,
    auto_fast_accept_min_quality_delta_ratio: float | None = None,
) -> list[SweepConfig]:
    configs: list[SweepConfig] = []
    use_auto_fast = (
        auto_fast_trigger_max_doc_weight_ratio is not None
        or auto_fast_trigger_min_above_max_doc_weight is not None
        or auto_fast_accept_max_doc_weight_ratio is not None
        or auto_fast_accept_min_quality_delta is not None
        or auto_fast_accept_min_quality_delta_ratio is not None
    )
    adaptive_bands = (
        adaptive_plateau_quality_bands
        if adaptive_plateau_quality_bands is not None
        else (
            0.0
            if adaptive_plateau_quality_band is None
            else float(adaptive_plateau_quality_band),
        )
    )
    for preset in gamma_presets:
        for perturbations in seed_perturbations:
            for parent_policy in parent_selection_policies:
                for policy in candidate_quality_policies:
                    policy_bands = (
                        adaptive_bands if policy == "adaptive_plateau" else (0.0,)
                    )
                    for adaptive_band in policy_bands:
                        if not math.isfinite(float(adaptive_band)) or float(adaptive_band) < 0.0:
                            raise ValueError(
                                "adaptive_plateau_quality_band must be finite and non-negative"
                            )
                        for repair_policy in baseline_repair_policies:
                            suffix = "" if policy == "structural" else f"_{policy}"
                            if policy == "adaptive_plateau" and float(adaptive_band) != 0.0:
                                suffix += f"_band{_format_float_suffix(float(adaptive_band))}"
                            if parent_policy != "weight":
                                suffix += f"_parent_{parent_policy}"
                            if repair_policy != "replace":
                                suffix += f"_repair_{repair_policy}"
                            if int(max_extra_parents_per_iteration) != 16:
                                suffix += f"_p{int(max_extra_parents_per_iteration)}"
                            if int(max_extra_children_per_parent) != 64:
                                suffix += f"_c{int(max_extra_children_per_parent)}"
                            if use_final_quality_guard:
                                suffix += "_final_guard"
                            if use_auto_fast:
                                suffix += "_auto_fast"
                            if auto_fast_accept_min_quality_delta is not None:
                                suffix += "_quality_accept"
                            if auto_fast_accept_min_quality_delta_ratio is not None:
                                suffix += "_quality_accept_ratio"
                            configs.append(
                                SweepConfig(
                                    config_id=f"{preset}_sp{perturbations}{suffix}",
                                    gamma_preset=preset,
                                    gamma_multipliers=GAMMA_PRESETS[preset],
                                    seed_perturbations=int(perturbations),
                                    max_extra_parents_per_iteration=int(
                                        max_extra_parents_per_iteration
                                    ),
                                    max_extra_children_per_parent=int(
                                        max_extra_children_per_parent
                                    ),
                                    parent_selection_policy=parent_policy,
                                    candidate_quality_policy=policy,
                                    min_candidate_delta_q=float(min_candidate_delta_q),
                                    adaptive_plateau_quality_band=float(adaptive_band),
                                    use_final_quality_guard=bool(use_final_quality_guard),
                                    min_final_quality_delta=float(min_final_quality_delta),
                                    baseline_repair_policy=repair_policy,
                                    baseline_repair_replace_min_parent_ratio=float(
                                        baseline_repair_replace_min_parent_ratio
                                    ),
                                    auto_fast_trigger_max_doc_weight_ratio=(
                                        None
                                        if auto_fast_trigger_max_doc_weight_ratio is None
                                        else float(auto_fast_trigger_max_doc_weight_ratio)
                                    ),
                                    auto_fast_trigger_min_above_max_doc_weight=(
                                        None
                                        if auto_fast_trigger_min_above_max_doc_weight is None
                                        else int(auto_fast_trigger_min_above_max_doc_weight)
                                    ),
                                    auto_fast_accept_max_doc_weight_ratio=(
                                        None
                                        if auto_fast_accept_max_doc_weight_ratio is None
                                        else float(auto_fast_accept_max_doc_weight_ratio)
                                    ),
                                    auto_fast_accept_min_quality_delta=(
                                        None
                                        if auto_fast_accept_min_quality_delta is None
                                        else float(auto_fast_accept_min_quality_delta)
                                    ),
                                    auto_fast_accept_min_quality_delta_ratio=(
                                        None
                                        if auto_fast_accept_min_quality_delta_ratio is None
                                        else float(auto_fast_accept_min_quality_delta_ratio)
                                    ),
                                )
                            )
    return configs


def _recompute_quality(
    graph: Any,
    membership: np.ndarray | None,
    resolution: float,
    reported_quality: Any,
) -> dict[str, Any]:
    if membership is None or reported_quality is None:
        return {
            "quality_recomputed": None,
            "quality_recompute_delta": None,
            "quality_recompute_abs_delta": None,
            "quality_recompute_ok": None,
        }
    quality = float(
        graph.cpm_quality(np.asarray(membership, dtype=np.uint64), resolution=resolution)
    )
    delta = quality - float(reported_quality)
    return {
        "quality_recomputed": quality,
        "quality_recompute_delta": delta,
        "quality_recompute_abs_delta": abs(delta),
        "quality_recompute_ok": abs(delta) <= QUALITY_RECOMPUTE_ABS_TOL,
    }


def _add_sweep_fields(
    *,
    row: dict[str, Any],
    input_cfg: pilot.Slice4Input,
    sweep_config: SweepConfig | None,
    use_baseline_repair: bool | None,
    quality_check: dict[str, Any],
) -> dict[str, Any]:
    enriched = {
        "sample": input_cfg.sample,
        "seed": int(input_cfg.seed),
        "summary_path": pilot._rel(input_cfg.summary_path),
        "config_id": "standard" if sweep_config is None else sweep_config.config_id,
        "gamma_preset": "standard" if sweep_config is None else sweep_config.gamma_preset,
        "gamma_multipliers": []
        if sweep_config is None
        else [float(x) for x in sweep_config.gamma_multipliers],
        "seed_perturbations": None
        if sweep_config is None
        else int(sweep_config.seed_perturbations),
        "max_extra_parents_per_iteration": None
        if sweep_config is None
        else int(sweep_config.max_extra_parents_per_iteration),
        "max_extra_children_per_parent": None
        if sweep_config is None
        else int(sweep_config.max_extra_children_per_parent),
        "parent_selection_policy": "standard"
        if sweep_config is None
        else sweep_config.parent_selection_policy,
        "candidate_quality_policy": "standard"
        if sweep_config is None
        else sweep_config.candidate_quality_policy,
        "min_candidate_delta_q": None
        if sweep_config is None
        else float(sweep_config.min_candidate_delta_q),
        "adaptive_plateau_quality_band": None
        if sweep_config is None
        else float(sweep_config.adaptive_plateau_quality_band),
        "use_final_quality_guard": None
        if sweep_config is None
        else bool(sweep_config.use_final_quality_guard),
        "min_final_quality_delta": None
        if sweep_config is None
        else float(sweep_config.min_final_quality_delta),
        "use_baseline_repair": use_baseline_repair,
        "candidate_trace_run_id": None,
        "quality_trace_run_id": None,
        "baseline_repair_policy": "replace"
        if sweep_config is None
        else sweep_config.baseline_repair_policy,
        "baseline_repair_replace_min_parent_ratio": None
        if sweep_config is None
        else float(sweep_config.baseline_repair_replace_min_parent_ratio),
        "auto_fast_trigger_max_doc_weight_ratio": None
        if sweep_config is None
        else sweep_config.auto_fast_trigger_max_doc_weight_ratio,
        "auto_fast_trigger_min_above_max_doc_weight": None
        if sweep_config is None
        else sweep_config.auto_fast_trigger_min_above_max_doc_weight,
        "auto_fast_accept_max_doc_weight_ratio": None
        if sweep_config is None
        else sweep_config.auto_fast_accept_max_doc_weight_ratio,
        "auto_fast_accept_min_quality_delta": None
        if sweep_config is None
        else sweep_config.auto_fast_accept_min_quality_delta,
        "auto_fast_accept_min_quality_delta_ratio": None
        if sweep_config is None
        else sweep_config.auto_fast_accept_min_quality_delta_ratio,
        "auto_fast_triggered": None,
        "auto_fast_fallback_triggered": None,
        "auto_fast_fallback_reason": "",
    }
    enriched.update(row)
    enriched.update(quality_check)
    return {field: enriched.get(field) for field in SWEEP_FIELDS}


def _set_repair_pair_comparison(
    *,
    repair_off_row: dict[str, Any],
    repair_on_row: dict[str, Any],
    repair_off_membership: np.ndarray | None,
    repair_on_membership: np.ndarray | None,
) -> None:
    if (
        repair_off_membership is None
        or repair_on_membership is None
        or repair_off_membership.shape != repair_on_membership.shape
    ):
        equal: bool | None = (
            None
            if repair_off_membership is None or repair_on_membership is None
            else False
        )
    else:
        equal = bool(np.array_equal(repair_off_membership, repair_on_membership))
    repair_off_row["membership_equal_repair_off_on"] = equal
    repair_on_row["membership_equal_repair_off_on"] = equal
    if repair_off_row.get("quality") is None:
        return
    repair_off_quality = float(repair_off_row["quality"])
    for row in (repair_off_row, repair_on_row):
        if row.get("quality") is None:
            continue
        delta = float(row["quality"]) - repair_off_quality
        row["quality_delta_vs_repair_off"] = delta
        row["quality_improved_vs_repair_off"] = delta > 0.0


def _auto_fast_enabled(sweep_config: SweepConfig) -> bool:
    return (
        sweep_config.auto_fast_trigger_max_doc_weight_ratio is not None
        or sweep_config.auto_fast_trigger_min_above_max_doc_weight is not None
        or sweep_config.auto_fast_accept_max_doc_weight_ratio is not None
        or sweep_config.auto_fast_accept_min_quality_delta is not None
        or sweep_config.auto_fast_accept_min_quality_delta_ratio is not None
    )


def _auto_fast_should_run(standard_row: dict[str, Any], sweep_config: SweepConfig) -> bool:
    if not _auto_fast_enabled(sweep_config):
        return True
    checks: list[bool] = []
    if sweep_config.auto_fast_trigger_max_doc_weight_ratio is not None:
        checks.append(
            float(standard_row.get("max_doc_weight_ratio") or 0.0)
            > float(sweep_config.auto_fast_trigger_max_doc_weight_ratio)
        )
    if sweep_config.auto_fast_trigger_min_above_max_doc_weight is not None:
        checks.append(
            int(standard_row.get("n_above_max_doc_weight") or 0)
            >= int(sweep_config.auto_fast_trigger_min_above_max_doc_weight)
        )
    return any(checks) if checks else True


def _standard_effective_row(
    *,
    standard_row: dict[str, Any],
    variant: str,
    use_baseline_repair: bool,
    elapsed_sec: float,
    triggered: bool,
    fallback_reason: str,
) -> dict[str, Any]:
    row = {field: standard_row.get(field) for field in pilot.CSV_FIELDS}
    row.update(
        {
            "variant": variant,
            "elapsed_sec": float(elapsed_sec),
            "quality_delta_vs_standard": 0.0,
            "quality_delta_vs_repair_off": None,
            "quality_improved_vs_standard": False,
            "quality_improved_vs_repair_off": None,
            "membership_equal_to_standard": True,
            "membership_diff_nodes_vs_standard": 0,
            "membership_equal_repair_off_on": None,
            "auto_fast_triggered": bool(triggered),
            "auto_fast_fallback_triggered": True,
            "auto_fast_fallback_reason": fallback_reason,
        }
    )
    if use_baseline_repair:
        for field in pilot.BASELINE_REPAIR_AUDIT_FIELDS:
            row[field] = 0 if field.endswith("_total") else 0.0
    return row


def _apply_auto_fast_acceptance(
    *,
    row: dict[str, Any],
    membership: np.ndarray | None,
    standard_row: dict[str, Any],
    standard_membership: np.ndarray | None,
    sweep_config: SweepConfig,
    variant: str,
    use_baseline_repair: bool,
) -> tuple[dict[str, Any], np.ndarray | None]:
    if not _auto_fast_enabled(sweep_config):
        row["auto_fast_triggered"] = True
        row["auto_fast_fallback_triggered"] = False
        row["auto_fast_fallback_reason"] = ""
        return row, membership
    row["auto_fast_triggered"] = True
    row["auto_fast_fallback_triggered"] = False
    row["auto_fast_fallback_reason"] = ""
    if sweep_config.auto_fast_accept_max_doc_weight_ratio is None:
        accepted_max = math.inf
    else:
        accepted_max = float(standard_row.get("max_doc_weight") or 0.0) * float(
            sweep_config.auto_fast_accept_max_doc_weight_ratio
        )
    if float(row.get("max_doc_weight") or math.inf) > accepted_max:
        return (
            _standard_effective_row(
                standard_row=standard_row,
                variant=variant,
                use_baseline_repair=use_baseline_repair,
                elapsed_sec=float(row.get("elapsed_sec") or 0.0),
                triggered=True,
                fallback_reason="max_doc_weight_guard",
            ),
            standard_membership,
        )
    if (
        sweep_config.auto_fast_accept_min_quality_delta is not None
        or sweep_config.auto_fast_accept_min_quality_delta_ratio is not None
    ):
        quality_delta = 0.0
        if sweep_config.auto_fast_accept_min_quality_delta is not None:
            quality_delta += float(sweep_config.auto_fast_accept_min_quality_delta)
        if sweep_config.auto_fast_accept_min_quality_delta_ratio is not None:
            quality_delta += abs(float(standard_row.get("quality") or 0.0)) * float(
                sweep_config.auto_fast_accept_min_quality_delta_ratio
            )
        min_quality = float(standard_row.get("quality") or 0.0) + quality_delta
        if float(row.get("quality") or -math.inf) < min_quality:
            return (
                _standard_effective_row(
                    standard_row=standard_row,
                    variant=variant,
                    use_baseline_repair=use_baseline_repair,
                    elapsed_sec=float(row.get("elapsed_sec") or 0.0),
                    triggered=True,
                    fallback_reason="quality_guard",
                ),
                standard_membership,
            )
    return row, membership


def _run_one_summary(
    *,
    input_cfg: pilot.Slice4Input,
    sweep_configs: list[SweepConfig],
    output_dir: Path | None = None,
    rows_by_key: dict[str, dict[str, Any]] | None = None,
    checkpoint_path: Path | None = None,
    progress_callback: Callable[[dict[str, dict[str, Any]]], None] | None = None,
    candidate_trace_runs_path: Path | None = None,
    quality_trace_runs_path: Path | None = None,
) -> list[dict[str, Any]]:
    completed = rows_by_key if rows_by_key is not None else {}
    append_enabled = rows_by_key is not None and checkpoint_path is not None
    base_config = pilot.Slice4RunConfig()

    def emit(batch: list[dict[str, Any]]) -> None:
        if not batch:
            return
        if append_enabled:
            _append_checkpoint_rows(checkpoint_path, batch)
        for item in batch:
            completed[_row_key(item)] = item
        rows.extend(batch)
        if progress_callback is not None:
            progress_callback(completed)

    n_nodes = pilot._infer_n_nodes(input_cfg)
    node_weights = pilot._load_node_weights(input_cfg.node_weights_path, n_nodes)
    graph = pilot._load_graph(input_cfg, node_weights)

    rows: list[dict[str, Any]] = []
    standard_key = _input_row_key(
        input_cfg=input_cfg,
        config_id="standard",
        variant=pilot.VARIANT_STANDARD,
        use_baseline_repair=None,
    )
    standard_cache_path = (
        None
        if output_dir is None
        else _standard_membership_cache_path(output_dir, input_cfg, base_config)
    )
    standard_row = completed.get(standard_key)
    standard_membership: np.ndarray | None = None
    if standard_row is not None and standard_cache_path is not None:
        try:
            standard_membership = np.load(standard_cache_path)
        except FileNotFoundError:
            standard_membership = None
    if standard_row is None or standard_membership is None:
        standard_raw_row, standard_membership = pilot._run_variant(
            graph=graph,
            input_cfg=input_cfg,
            run_config=base_config,
            node_weights=node_weights,
            variant=pilot.VARIANT_STANDARD,
            standard_membership=None,
            standard_quality=None,
        )
        if standard_cache_path is not None and standard_membership is not None:
            standard_cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(standard_cache_path, standard_membership)
        standard_row = _add_sweep_fields(
            row=standard_raw_row,
            input_cfg=input_cfg,
            sweep_config=None,
            use_baseline_repair=None,
            quality_check=_recompute_quality(
                graph,
                standard_membership,
                input_cfg.resolution,
                standard_raw_row.get("quality"),
            ),
        )
        emit([standard_row])
    standard_quality = float(standard_row["quality"])

    for sweep_config in sweep_configs:
        repair_off_key = _input_row_key(
            input_cfg=input_cfg,
            config_id=sweep_config.config_id,
            variant=pilot.VARIANT_REPAIR_OFF,
            use_baseline_repair=False,
        )
        repair_on_key = _input_row_key(
            input_cfg=input_cfg,
            config_id=sweep_config.config_id,
            variant=pilot.VARIANT_REPAIR_ON,
            use_baseline_repair=True,
        )
        if rows_by_key is not None and repair_off_key in completed and repair_on_key in completed:
            continue
        repair_off_trace_run_id = _candidate_trace_run_id_for_input(
            input_cfg=input_cfg,
            config_id=sweep_config.config_id,
            variant=pilot.VARIANT_REPAIR_OFF,
            use_baseline_repair=False,
        )
        repair_on_trace_run_id = _candidate_trace_run_id_for_input(
            input_cfg=input_cfg,
            config_id=sweep_config.config_id,
            variant=pilot.VARIANT_REPAIR_ON,
            use_baseline_repair=True,
        )
        run_config = pilot.Slice4RunConfig(
            gamma_multipliers=sweep_config.gamma_multipliers,
            seed_perturbations=sweep_config.seed_perturbations,
            max_extra_parents_per_iteration=(
                sweep_config.max_extra_parents_per_iteration
            ),
            max_extra_children_per_parent=(
                sweep_config.max_extra_children_per_parent
            ),
            parent_selection_policy=sweep_config.parent_selection_policy,
            candidate_quality_policy=sweep_config.candidate_quality_policy,
            min_candidate_delta_q=sweep_config.min_candidate_delta_q,
            adaptive_plateau_quality_band=sweep_config.adaptive_plateau_quality_band,
            use_final_quality_guard=sweep_config.use_final_quality_guard,
            min_final_quality_delta=sweep_config.min_final_quality_delta,
            baseline_repair_policy=sweep_config.baseline_repair_policy,
            baseline_repair_replace_min_parent_ratio=(
                sweep_config.baseline_repair_replace_min_parent_ratio
            ),
        )
        if not _auto_fast_should_run(standard_row, sweep_config):
            fallback_elapsed = float(standard_row.get("elapsed_sec") or 0.0)
            repair_off_row = _standard_effective_row(
                standard_row=standard_row,
                variant=pilot.VARIANT_REPAIR_OFF,
                use_baseline_repair=False,
                elapsed_sec=fallback_elapsed,
                triggered=False,
                fallback_reason="trigger_not_met",
            )
            repair_on_row = _standard_effective_row(
                standard_row=standard_row,
                variant=pilot.VARIANT_REPAIR_ON,
                use_baseline_repair=True,
                elapsed_sec=fallback_elapsed,
                triggered=False,
                fallback_reason="trigger_not_met",
            )
            repair_off_membership = standard_membership
            repair_on_membership = standard_membership
            _set_repair_pair_comparison(
                repair_off_row=repair_off_row,
                repair_on_row=repair_on_row,
                repair_off_membership=repair_off_membership,
                repair_on_membership=repair_on_membership,
            )
            emit(
                [
                    _add_sweep_fields(
                        row=repair_off_row,
                        input_cfg=input_cfg,
                        sweep_config=sweep_config,
                        use_baseline_repair=False,
                        quality_check=_recompute_quality(
                            graph,
                            repair_off_membership,
                            input_cfg.resolution,
                            repair_off_row.get("quality"),
                        ),
                    ),
                    _add_sweep_fields(
                        row=repair_on_row,
                        input_cfg=input_cfg,
                        sweep_config=sweep_config,
                        use_baseline_repair=True,
                        quality_check=_recompute_quality(
                            graph,
                            repair_on_membership,
                            input_cfg.resolution,
                            repair_on_row.get("quality"),
                        ),
                    ),
                ]
            )
            continue
        if candidate_trace_runs_path is not None:
            _append_candidate_trace_run_metadata(
                candidate_trace_runs_path,
                _candidate_trace_run_metadata(
                    input_cfg=input_cfg,
                    sweep_config=sweep_config,
                    variant=pilot.VARIANT_REPAIR_OFF,
                    use_baseline_repair=False,
                    run_id=repair_off_trace_run_id,
                ),
            )
        if quality_trace_runs_path is not None:
            _append_quality_trace_run_metadata(
                quality_trace_runs_path,
                _quality_trace_run_metadata(
                    input_cfg=input_cfg,
                    sweep_config=sweep_config,
                    variant=pilot.VARIANT_REPAIR_OFF,
                    use_baseline_repair=False,
                    run_id=repair_off_trace_run_id,
                ),
            )
        with _candidate_trace_context(
            repair_off_trace_run_id if candidate_trace_runs_path is not None else None
        ):
            with _quality_trace_context(
                repair_off_trace_run_id if quality_trace_runs_path is not None else None
            ):
                repair_off_row, repair_off_membership = pilot._run_variant(
                    graph=graph,
                    input_cfg=input_cfg,
                    run_config=run_config,
                    node_weights=node_weights,
                    variant=pilot.VARIANT_REPAIR_OFF,
                    standard_membership=standard_membership,
                    standard_quality=standard_quality,
                )
        if candidate_trace_runs_path is not None:
            _append_candidate_trace_run_metadata(
                candidate_trace_runs_path,
                _candidate_trace_run_metadata(
                    input_cfg=input_cfg,
                    sweep_config=sweep_config,
                    variant=pilot.VARIANT_REPAIR_ON,
                    use_baseline_repair=True,
                    run_id=repair_on_trace_run_id,
                ),
            )
        if quality_trace_runs_path is not None:
            _append_quality_trace_run_metadata(
                quality_trace_runs_path,
                _quality_trace_run_metadata(
                    input_cfg=input_cfg,
                    sweep_config=sweep_config,
                    variant=pilot.VARIANT_REPAIR_ON,
                    use_baseline_repair=True,
                    run_id=repair_on_trace_run_id,
                ),
            )
        with _candidate_trace_context(
            repair_on_trace_run_id if candidate_trace_runs_path is not None else None
        ):
            with _quality_trace_context(
                repair_on_trace_run_id if quality_trace_runs_path is not None else None
            ):
                repair_on_row, repair_on_membership = pilot._run_variant(
                    graph=graph,
                    input_cfg=input_cfg,
                    run_config=run_config,
                    node_weights=node_weights,
                    variant=pilot.VARIANT_REPAIR_ON,
                    standard_membership=standard_membership,
                    standard_quality=standard_quality,
                )
        repair_off_row, repair_off_membership = _apply_auto_fast_acceptance(
            row=repair_off_row,
            membership=repair_off_membership,
            standard_row=standard_row,
            standard_membership=standard_membership,
            sweep_config=sweep_config,
            variant=pilot.VARIANT_REPAIR_OFF,
            use_baseline_repair=False,
        )
        repair_off_row["candidate_trace_run_id"] = repair_off_trace_run_id
        repair_off_row["quality_trace_run_id"] = repair_off_trace_run_id
        repair_on_row, repair_on_membership = _apply_auto_fast_acceptance(
            row=repair_on_row,
            membership=repair_on_membership,
            standard_row=standard_row,
            standard_membership=standard_membership,
            sweep_config=sweep_config,
            variant=pilot.VARIANT_REPAIR_ON,
            use_baseline_repair=True,
        )
        repair_on_row["candidate_trace_run_id"] = repair_on_trace_run_id
        repair_on_row["quality_trace_run_id"] = repair_on_trace_run_id
        _set_repair_pair_comparison(
            repair_off_row=repair_off_row,
            repair_on_row=repair_on_row,
            repair_off_membership=repair_off_membership,
            repair_on_membership=repair_on_membership,
        )
        emit(
            [
                _add_sweep_fields(
                    row=repair_off_row,
                    input_cfg=input_cfg,
                    sweep_config=sweep_config,
                    use_baseline_repair=False,
                    quality_check=_recompute_quality(
                        graph,
                        repair_off_membership,
                        input_cfg.resolution,
                        repair_off_row.get("quality"),
                    ),
                ),
                _add_sweep_fields(
                    row=repair_on_row,
                    input_cfg=input_cfg,
                    sweep_config=sweep_config,
                    use_baseline_repair=True,
                    quality_check=_recompute_quality(
                        graph,
                        repair_on_membership,
                        input_cfg.resolution,
                        repair_on_row.get("quality"),
                    ),
                ),
            ]
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SWEEP_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in SWEEP_FIELDS})


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(
        [{field: _json_safe(row.get(field)) for field in SWEEP_FIELDS} for row in rows]
    )
    pq.write_table(table, path)


def _finite_values(rows: list[dict[str, Any]], field: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(field)
        if value is None:
            continue
        value = float(value)
        if math.isfinite(value):
            values.append(value)
    return values


def _finite_or_neg_inf(value: Any) -> float:
    if value is None:
        return -math.inf
    value = float(value)
    return value if math.isfinite(value) else -math.inf


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    refinement_rows = [
        row for row in rows if row.get("variant") != pilot.VARIANT_STANDARD
    ]
    repair_on_rows = [
        row for row in rows if row.get("variant") == pilot.VARIANT_REPAIR_ON
    ]
    best_refinement = max(
        refinement_rows,
        key=lambda row: _finite_or_neg_inf(row.get("quality_delta_vs_standard")),
        default=None,
    )
    best_repair_gain = max(
        repair_on_rows,
        key=lambda row: _finite_or_neg_inf(row.get("quality_delta_vs_repair_off")),
        default=None,
    )
    recompute_deltas = _finite_values(rows, "quality_recompute_abs_delta")
    by_config: dict[str, dict[str, Any]] = {}
    for row in refinement_rows:
        config_id = str(row.get("config_id"))
        group = by_config.setdefault(
            config_id,
            {
                "config_id": config_id,
                "gamma_preset": row.get("gamma_preset"),
                "seed_perturbations": row.get("seed_perturbations"),
                "max_extra_parents_per_iteration": row.get(
                    "max_extra_parents_per_iteration"
                ),
                "max_extra_children_per_parent": row.get(
                    "max_extra_children_per_parent"
                ),
                "parent_selection_policy": row.get("parent_selection_policy"),
                "candidate_quality_policy": row.get("candidate_quality_policy"),
                "min_candidate_delta_q": row.get("min_candidate_delta_q"),
                "adaptive_plateau_quality_band": row.get(
                    "adaptive_plateau_quality_band"
                ),
                "use_final_quality_guard": row.get("use_final_quality_guard"),
                "min_final_quality_delta": row.get("min_final_quality_delta"),
                "baseline_repair_policy": row.get("baseline_repair_policy"),
                "baseline_repair_replace_min_parent_ratio": row.get(
                    "baseline_repair_replace_min_parent_ratio"
                ),
                "auto_fast_trigger_max_doc_weight_ratio": row.get(
                    "auto_fast_trigger_max_doc_weight_ratio"
                ),
                "auto_fast_trigger_min_above_max_doc_weight": row.get(
                    "auto_fast_trigger_min_above_max_doc_weight"
                ),
                "auto_fast_accept_max_doc_weight_ratio": row.get(
                    "auto_fast_accept_max_doc_weight_ratio"
                ),
                "auto_fast_accept_min_quality_delta": row.get(
                    "auto_fast_accept_min_quality_delta"
                ),
                "auto_fast_accept_min_quality_delta_ratio": row.get(
                    "auto_fast_accept_min_quality_delta_ratio"
                ),
                "n_rows": 0,
                "n_improved_vs_standard": 0,
                "repair_on_improved_vs_repair_off": 0,
                "final_quality_guard_enabled_count": 0,
                "final_quality_guard_triggered_count": 0,
                "auto_fast_triggered_count": 0,
                "auto_fast_fallback_triggered_count": 0,
                "best_quality_delta_vs_standard": -math.inf,
                "best_quality_delta_vs_repair_off": -math.inf,
            },
        )
        group["n_rows"] += 1
        if bool(row.get("quality_improved_vs_standard")):
            group["n_improved_vs_standard"] += 1
        if row.get("variant") == pilot.VARIANT_REPAIR_ON and bool(
            row.get("quality_improved_vs_repair_off")
        ):
            group["repair_on_improved_vs_repair_off"] += 1
        if bool(row.get("final_quality_guard_enabled")):
            group["final_quality_guard_enabled_count"] += 1
        if bool(row.get("final_quality_guard_triggered")):
            group["final_quality_guard_triggered_count"] += 1
        if bool(row.get("auto_fast_triggered")):
            group["auto_fast_triggered_count"] += 1
        if bool(row.get("auto_fast_fallback_triggered")):
            group["auto_fast_fallback_triggered_count"] += 1
        group["best_quality_delta_vs_standard"] = max(
            float(group["best_quality_delta_vs_standard"]),
            _finite_or_neg_inf(row.get("quality_delta_vs_standard")),
        )
        if row.get("quality_delta_vs_repair_off") is not None:
            group["best_quality_delta_vs_repair_off"] = max(
                float(group["best_quality_delta_vs_repair_off"]),
                _finite_or_neg_inf(row.get("quality_delta_vs_repair_off")),
            )
    return {
        "n_rows": int(len(rows)),
        "n_refinement_rows": int(len(refinement_rows)),
        "n_refinement_improved_vs_standard": int(
            sum(1 for row in refinement_rows if bool(row.get("quality_improved_vs_standard")))
        ),
        "n_repair_on_improved_vs_repair_off": int(
            sum(
                1
                for row in repair_on_rows
                if bool(row.get("quality_improved_vs_repair_off"))
            )
        ),
        "n_final_quality_guard_enabled": int(
            sum(1 for row in refinement_rows if bool(row.get("final_quality_guard_enabled")))
        ),
        "n_final_quality_guard_triggered": int(
            sum(1 for row in refinement_rows if bool(row.get("final_quality_guard_triggered")))
        ),
        "n_auto_fast_triggered": int(
            sum(1 for row in refinement_rows if bool(row.get("auto_fast_triggered")))
        ),
        "n_auto_fast_fallback_triggered": int(
            sum(
                1
                for row in refinement_rows
                if bool(row.get("auto_fast_fallback_triggered"))
            )
        ),
        "best_refinement_row": best_refinement,
        "best_repair_gain_row": best_repair_gain,
        "quality_recompute_max_abs_delta": max(recompute_deltas or [0.0]),
        "quality_recompute_all_ok": all(
            bool(row.get("quality_recompute_ok"))
            for row in rows
            if row.get("quality_recompute_ok") is not None
        ),
        "repair_off_baseline_repair_audit_zero_all": all(
            float(row.get(field) or 0.0) == 0.0
            for row in rows
            if row.get("variant") == pilot.VARIANT_REPAIR_OFF
            for field in pilot.BASELINE_REPAIR_AUDIT_FIELDS
        ),
        "baseline_repair_audit_semantics": (
            "baseline_repair_merge_count_total sums internal merges across "
            "evaluated repair candidates; baseline_repair_selected_total counts "
            "final selected parent candidates whose selected candidate included "
            "repair merges."
        ),
        "candidate_profile": pilot._audit_profile_summary(refinement_rows),
        "by_config": sorted(
            by_config.values(),
            key=lambda item: (
                -float(item["best_quality_delta_vs_standard"]),
                str(item["config_id"]),
            ),
        ),
    }


def _format_optional_float(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.6g}"


def _write_report(
    path: Path,
    *,
    summaries: list[Path],
    sweep_configs: list[SweepConfig],
    rows: list[dict[str, Any]],
    aggregate: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    best = aggregate.get("best_refinement_row") or {}
    best_repair = aggregate.get("best_repair_gain_row") or {}
    profile = aggregate.get("candidate_profile") or {}
    quadrants = profile.get("quadrants") or {}
    confusion = profile.get("decision_confusion") or {}
    same_gamma = profile.get("same_gamma") or {}
    high_gamma = profile.get("high_gamma") or {}
    lines = [
        "# Dongdaemun Refinement Slice 4 Quality-Aware Sweep",
        "",
        "This quality-aware sweep keeps max-weight pressure as a diagnostic but judges the experiment by CPM objective quality on the same graph, gamma, and seed.",
        "",
        "## Scope",
        "",
        f"- Summaries: {len(summaries)}",
        f"- Sweep configs: {len(sweep_configs)}",
        f"- Rows: {len(rows)}",
        f"- Refinement rows improved vs standard: {aggregate.get('n_refinement_improved_vs_standard')} / {aggregate.get('n_refinement_rows')}",
        f"- Repair-on rows improved vs repair-off: {aggregate.get('n_repair_on_improved_vs_repair_off')}",
        f"- Final quality guard triggered: {aggregate.get('n_final_quality_guard_triggered')} / {aggregate.get('n_final_quality_guard_enabled')}",
        f"- Auto-fast triggered rows: {aggregate.get('n_auto_fast_triggered')} / {aggregate.get('n_refinement_rows')}",
        f"- Auto-fast fallback rows: {aggregate.get('n_auto_fast_fallback_triggered')} / {aggregate.get('n_refinement_rows')}",
        f"- Quality recompute max abs delta: {aggregate.get('quality_recompute_max_abs_delta'):.6g}",
        f"- Quality recompute all ok: {aggregate.get('quality_recompute_all_ok')}",
        f"- Repair-off baseline repair audit zero all: {aggregate.get('repair_off_baseline_repair_audit_zero_all')}",
        "",
        "## Best Rows",
        "",
        "- Best refinement vs standard: "
        + (
            "none"
            if not best
            else (
                f"seed {best.get('seed')}, {best.get('config_id')}, "
                f"{best.get('variant')}, delta {float(best.get('quality_delta_vs_standard') or 0.0):.6g}"
            )
        ),
        "- Best repair-on gain vs repair-off: "
        + (
            "none"
            if not best_repair
            else (
                f"seed {best_repair.get('seed')}, {best_repair.get('config_id')}, "
                f"delta {float(best_repair.get('quality_delta_vs_repair_off') or 0.0):.6g}"
            )
        ),
        "",
        "## Audit Semantics",
        "",
        f"- {aggregate.get('baseline_repair_audit_semantics')}",
        "- Q/S quadrants: "
        f"Q+/S+={quadrants.get('qpos_spos', 0)}, "
        f"Q+/S-={quadrants.get('qpos_sneg', 0)}, "
        f"Q-/S+={quadrants.get('qneg_spos', 0)}, "
        f"Q-/S-={quadrants.get('qneg_sneg', 0)}.",
        "- Decision confusion: "
        f"TP={confusion.get('true_positive', 0)}, "
        f"FP={confusion.get('false_positive', 0)}, "
        f"FN={confusion.get('false_negative', 0)}, "
        f"TN={confusion.get('true_negative', 0)}.",
        "- Source success rates: "
        f"same-gamma Q+/S+={same_gamma.get('qpos_spos_rate')}, "
        f"high-gamma Q+/S+={high_gamma.get('qpos_spos_rate')}; "
        f"same-gamma applied={same_gamma.get('applied_rate')}, "
        f"high-gamma applied={high_gamma.get('applied_rate')}.",
        "",
        "## Config Summary",
        "",
        "| config | gamma preset | seed perturbations | parents | children | parent policy | quality policy | repair policy | repair ratio | final guard | guard triggered | auto trigger | auto fallback | improved vs standard | repair-on improved vs off | best delta vs standard | best repair-on delta |",
        "| --- | --- | ---: | ---: | ---: | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in aggregate.get("by_config", []):
        lines.append(
            "| {config} | {preset} | {sp} | {parents} | {children} | {parent_policy} | {policy} | {repair_policy} | {repair_ratio:.6g} | {guard} | {triggered} | {auto_triggered} | {auto_fallback} | {improved} | {repair_improved} | {best_std:.6g} | {best_repair:.6g} |".format(
                config=item.get("config_id"),
                preset=item.get("gamma_preset"),
                sp=item.get("seed_perturbations"),
                parents=item.get("max_extra_parents_per_iteration"),
                children=item.get("max_extra_children_per_parent"),
                parent_policy=item.get("parent_selection_policy"),
                policy=item.get("candidate_quality_policy"),
                repair_policy=item.get("baseline_repair_policy"),
                repair_ratio=float(
                    item.get("baseline_repair_replace_min_parent_ratio") or 0.0
                ),
                guard=item.get("use_final_quality_guard"),
                triggered=item.get("final_quality_guard_triggered_count"),
                auto_triggered=item.get("auto_fast_triggered_count"),
                auto_fallback=item.get("auto_fast_fallback_triggered_count"),
                improved=item.get("n_improved_vs_standard"),
                repair_improved=item.get("repair_on_improved_vs_repair_off"),
                best_std=float(item.get("best_quality_delta_vs_standard") or 0.0),
                best_repair=float(item.get("best_quality_delta_vs_repair_off") or 0.0),
            )
        )
    lines.extend(
        [
            "",
            "## Top Refinement Rows",
            "",
            "| seed | config | variant | delta vs standard | candidate delta sum | quality rejects | max doc weight | repair candidates | repair merges |",
            "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    top_rows = sorted(
        [row for row in rows if row.get("variant") != pilot.VARIANT_STANDARD],
        key=lambda row: _finite_or_neg_inf(row.get("quality_delta_vs_standard")),
        reverse=True,
    )[:12]
    for row in top_rows:
        lines.append(
            "| {seed} | {config} | {variant} | {delta_std:.6g} | {candidate_delta:.6g} | {quality_rejects} | {max_weight:.6g} | {candidates} | {merges} |".format(
                seed=row.get("seed"),
                config=row.get("config_id"),
                variant=row.get("variant"),
                delta_std=float(row.get("quality_delta_vs_standard") or 0.0),
                candidate_delta=float(row.get("candidate_quality_delta_sum") or 0.0),
                quality_rejects=int(row.get("candidate_rejected_by_quality_total") or 0),
                max_weight=float(row.get("max_doc_weight") or 0.0),
                candidates=int(row.get("baseline_repair_candidates_total") or 0),
                merges=int(row.get("baseline_repair_merge_count_total") or 0),
            )
        )
    lines.extend([""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _expected_row_count(n_summaries: int, n_configs: int) -> int:
    return int(n_summaries) * (1 + 2 * int(n_configs))


def _write_progress(
    *,
    output_dir: Path,
    summary_paths: list[Path],
    sweep_configs: list[SweepConfig],
    rows_by_key: dict[str, dict[str, Any]],
) -> None:
    rows = _ordered_rows_from_checkpoint(
        summary_paths=summary_paths,
        sweep_configs=sweep_configs,
        rows_by_key=rows_by_key,
    )
    csv_path = output_dir / "slice4_quality_sweep.csv"
    _write_csv(csv_path, rows)
    _write_json(
        _progress_path(output_dir),
        {
            "schema": f"dongdaemun_refinement_slice4_quality_sweep_progress.v{SCHEMA_VERSION}",
            "checkpoint_rows": _checkpoint_path(output_dir),
            "csv": csv_path,
            "n_completed_rows": len(rows),
            "n_expected_rows": _expected_row_count(len(summary_paths), len(sweep_configs)),
            "complete": len(rows)
            >= _expected_row_count(len(summary_paths), len(sweep_configs)),
        },
    )


def run_sweep(
    summary_paths: list[Path],
    *,
    output_dir: Path,
    sweep_configs: list[SweepConfig] | None = None,
    resume: bool = False,
    candidate_trace_path: Path | None = None,
    quality_trace_path: Path | None = None,
) -> dict[str, Any]:
    sweep_configs = sweep_configs or _build_sweep_configs()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = _checkpoint_path(output_dir)
    if not resume and checkpoint_path.exists():
        checkpoint_path.unlink()
    rows_by_key = _load_checkpoint_rows(checkpoint_path) if resume else {}
    explicit_candidate_trace_path = candidate_trace_path is not None
    if candidate_trace_path is None:
        env_trace_path = os.environ.get(CANDIDATE_TRACE_PATH_ENV)
        candidate_trace_path = Path(env_trace_path) if env_trace_path else None
    if candidate_trace_path is not None:
        candidate_trace_runs_path = output_dir / CANDIDATE_TRACE_RUNS_FILENAME
        if not resume and candidate_trace_runs_path.exists():
            candidate_trace_runs_path.unlink()
    else:
        candidate_trace_runs_path = None
    explicit_quality_trace_path = quality_trace_path is not None
    if quality_trace_path is None:
        env_trace_path = os.environ.get(QUALITY_TRACE_PATH_ENV)
        quality_trace_path = Path(env_trace_path) if env_trace_path else None
    if quality_trace_path is not None:
        quality_trace_runs_path = output_dir / QUALITY_TRACE_RUNS_FILENAME
        if not resume and quality_trace_runs_path.exists():
            quality_trace_runs_path.unlink()
    else:
        quality_trace_runs_path = None

    def progress_callback(current_rows: dict[str, dict[str, Any]]) -> None:
        _write_progress(
            output_dir=output_dir,
            summary_paths=summary_paths,
            sweep_configs=sweep_configs,
            rows_by_key=current_rows,
        )

    with _candidate_trace_path_context(
        candidate_trace_path,
        explicit=explicit_candidate_trace_path,
        resume=resume,
    ):
        with _quality_trace_path_context(
            quality_trace_path,
            explicit=explicit_quality_trace_path,
            resume=resume,
        ):
            for summary_path in summary_paths:
                input_cfg = pilot._resolve_input_from_summary(summary_path)
                planned = _planned_row_keys(input_cfg=input_cfg, sweep_configs=sweep_configs)
                if all(key in rows_by_key for key in planned):
                    continue
                _run_one_summary(
                    input_cfg=input_cfg,
                    sweep_configs=sweep_configs,
                    output_dir=output_dir,
                    rows_by_key=rows_by_key,
                    checkpoint_path=checkpoint_path,
                    progress_callback=progress_callback,
                    candidate_trace_runs_path=candidate_trace_runs_path,
                    quality_trace_runs_path=quality_trace_runs_path,
                )
            rows = _ordered_rows_from_checkpoint(
                summary_paths=summary_paths,
                sweep_configs=sweep_configs,
                rows_by_key=rows_by_key,
            )
            aggregate = _aggregate_rows(rows)

            csv_path = output_dir / "slice4_quality_sweep.csv"
            parquet_path = output_dir / "slice4_quality_sweep.parquet"
            summary_path = output_dir / "slice4_quality_sweep_summary.json"
            report_path = output_dir / "slice4_quality_sweep_report.md"
            _write_csv(csv_path, rows)
            _write_parquet(parquet_path, rows)
            payload = {
                "schema": f"dongdaemun_refinement_slice4_quality_sweep.v{SCHEMA_VERSION}",
                "summary_paths": summary_paths,
                "sweep_configs": [asdict(config) for config in sweep_configs],
                "aggregate": aggregate,
                "rows": rows,
                "paths": {
                    "checkpoint_rows": checkpoint_path,
                    "progress": _progress_path(output_dir),
                    "csv": csv_path,
                    "parquet": parquet_path,
                    "summary": summary_path,
                    "report": report_path,
                },
            }
            if candidate_trace_path is not None and candidate_trace_runs_path is not None:
                payload["paths"]["candidate_trace"] = candidate_trace_path
                payload["paths"]["candidate_trace_runs"] = candidate_trace_runs_path
            if quality_trace_path is not None and quality_trace_runs_path is not None:
                payload["paths"]["quality_trace"] = quality_trace_path
                payload["paths"]["quality_trace_runs"] = quality_trace_runs_path
            _write_json(summary_path, payload)
            _write_report(
                report_path,
                summaries=summary_paths,
                sweep_configs=sweep_configs,
                rows=rows,
                aggregate=aggregate,
            )
            return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        action="append",
        type=Path,
        required=True,
        help="Prepared source-level prepare_summary.json. Repeat for multiple seeds.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--gamma-presets",
        help=f"Comma-separated subset of presets: {', '.join(GAMMA_PRESETS)}",
    )
    parser.add_argument(
        "--seed-perturbations",
        help="Comma-separated seed perturbation counts; default 0,1,2,4.",
    )
    parser.add_argument(
        "--candidate-quality-policies",
        help=(
            "Comma-separated policies: structural, quality_guarded_structural, "
            "quality_floor, quality_first, selective, pressure_aware, "
            "adaptive_plateau."
        ),
    )
    parser.add_argument("--min-candidate-delta-q", type=float, default=0.0)
    parser.add_argument(
        "--adaptive-plateau-quality-band",
        help=(
            "Comma-separated adaptive_plateau quality bands. Only adaptive_plateau "
            "configs are expanded across these values. Default: 0."
        ),
    )
    parser.add_argument("--max-extra-parents-per-iteration", type=int, default=16)
    parser.add_argument("--max-extra-children-per-parent", type=int, default=64)
    parser.add_argument(
        "--parent-selection-policies",
        help="Comma-separated parent queue policies: weight, pressure_boundary.",
    )
    parser.add_argument("--use-final-quality-guard", action="store_true")
    parser.add_argument("--min-final-quality-delta", type=float, default=0.0)
    parser.add_argument(
        "--baseline-repair-policies",
        help="Comma-separated repair policies for repair-on rows: replace, augment, adaptive.",
    )
    parser.add_argument(
        "--baseline-repair-replace-min-parent-ratio",
        type=float,
        default=1.05,
    )
    parser.add_argument(
        "--auto-fast-trigger-max-doc-weight-ratio",
        type=float,
        help=(
            "Only run refinement when the standard row max_doc_weight_ratio is "
            "above this threshold. Disabled when omitted."
        ),
    )
    parser.add_argument(
        "--auto-fast-trigger-min-above-max-doc-weight",
        type=int,
        help=(
            "Only run refinement when the standard row has at least this many "
            "clusters above target max doc weight. Disabled when omitted."
        ),
    )
    parser.add_argument(
        "--auto-fast-accept-max-doc-weight-ratio",
        type=float,
        help=(
            "Fallback to the standard row when refined max_doc_weight exceeds "
            "standard max_doc_weight times this ratio. Disabled when omitted."
        ),
    )
    parser.add_argument(
        "--auto-fast-accept-min-quality-delta",
        type=float,
        help=(
            "Fallback to the standard row when refined quality is below standard "
            "quality plus this delta. Disabled when omitted."
        ),
    )
    parser.add_argument(
        "--auto-fast-accept-min-quality-delta-ratio",
        type=float,
        help=(
            "Fallback when refined quality is below standard quality plus "
            "abs(standard quality) times this ratio. Disabled when omitted."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse slice4_quality_sweep_rows.jsonl rows already present in output-dir.",
    )
    parser.add_argument(
        "--candidate-trace",
        action="store_true",
        help=(
            "Write opt-in candidate scalar trace JSONL to "
            "output-dir/candidate_trace.jsonl."
        ),
    )
    parser.add_argument(
        "--candidate-trace-path",
        type=Path,
        help=(
            "Append opt-in candidate scalar trace JSONL to this path. "
            "Each event includes run_id for joining to candidate_trace_run_id in the CSV."
        ),
    )
    parser.add_argument(
        "--quality-trace",
        action="store_true",
        help=(
            "Write opt-in quality checkpoint trace JSONL to "
            "output-dir/quality_trace.jsonl."
        ),
    )
    parser.add_argument(
        "--quality-trace-path",
        type=Path,
        help=(
            "Append opt-in quality checkpoint trace JSONL to this path. "
            "Each event includes run_id for joining to quality_trace_run_id in the CSV."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    sweep_configs = _build_sweep_configs(
        gamma_presets=_parse_gamma_presets(args.gamma_presets),
        seed_perturbations=_parse_int_tuple(args.seed_perturbations),
        candidate_quality_policies=_parse_candidate_quality_policies(
            args.candidate_quality_policies
        ),
        parent_selection_policies=_parse_parent_selection_policies(
            args.parent_selection_policies
        ),
        min_candidate_delta_q=float(args.min_candidate_delta_q),
        adaptive_plateau_quality_bands=_parse_float_tuple(
            args.adaptive_plateau_quality_band,
            (0.0,),
        ),
        use_final_quality_guard=bool(args.use_final_quality_guard),
        min_final_quality_delta=float(args.min_final_quality_delta),
        baseline_repair_policies=_parse_baseline_repair_policies(
            args.baseline_repair_policies
        ),
        baseline_repair_replace_min_parent_ratio=float(
            args.baseline_repair_replace_min_parent_ratio
        ),
        max_extra_parents_per_iteration=int(args.max_extra_parents_per_iteration),
        max_extra_children_per_parent=int(args.max_extra_children_per_parent),
        auto_fast_trigger_max_doc_weight_ratio=(
            args.auto_fast_trigger_max_doc_weight_ratio
        ),
        auto_fast_trigger_min_above_max_doc_weight=(
            args.auto_fast_trigger_min_above_max_doc_weight
        ),
        auto_fast_accept_max_doc_weight_ratio=(
            args.auto_fast_accept_max_doc_weight_ratio
        ),
        auto_fast_accept_min_quality_delta=(
            args.auto_fast_accept_min_quality_delta
        ),
        auto_fast_accept_min_quality_delta_ratio=(
            args.auto_fast_accept_min_quality_delta_ratio
        ),
    )
    candidate_trace_path = args.candidate_trace_path
    if args.candidate_trace and candidate_trace_path is None:
        candidate_trace_path = args.output_dir / CANDIDATE_TRACE_FILENAME
    quality_trace_path = args.quality_trace_path
    if args.quality_trace and quality_trace_path is None:
        quality_trace_path = args.output_dir / QUALITY_TRACE_FILENAME
    payload = run_sweep(
        args.summary,
        output_dir=args.output_dir,
        sweep_configs=sweep_configs,
        resume=bool(args.resume),
        candidate_trace_path=candidate_trace_path,
        quality_trace_path=quality_trace_path,
    )
    print(f"Saved quality sweep outputs to {pilot._rel(payload['paths']['summary'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
