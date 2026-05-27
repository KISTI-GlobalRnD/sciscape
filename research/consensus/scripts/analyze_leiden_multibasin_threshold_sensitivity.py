#!/usr/bin/env python3
"""Sweep coarse-basin thresholds for Dongdaemun p5 basin sketches."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import pandas as pd

from analyze_leiden_multibasin_signatures import (
    _finite_float,
    _group_columns,
    _mark_material_gain,
    _read_csvs,
    _signature_frame,
    build_coarse_basin_rows,
    build_pairwise_basin_matrix,
)


DEFAULT_ENDPOINT_TAUS = "0,0.000001,0.000005,0.00001,0.00002,0.02"
DEFAULT_SUPPORT_TAUS = "0,0.25,0.5,0.75,1.0"


def _parse_float_csv(value: str) -> list[float]:
    out = [float(part) for part in value.split(",") if part.strip()]
    if not out:
        raise ValueError("expected at least one float")
    return out


def _group_key_map(frame: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    group_cols = _group_columns(frame)
    if group_cols:
        return group_cols, frame
    out = frame.copy()
    out["_all"] = "all"
    return ["_all"], out


def _summarize_threshold(
    signature_rows: pd.DataFrame,
    *,
    endpoint_tau: float,
    support_tau: float,
    iso_q_delta: float,
    iso_q_relative_ppm: float,
) -> pd.DataFrame:
    pairwise = build_pairwise_basin_matrix(
        signature_rows,
        coarse_endpoint_tau=endpoint_tau,
        coarse_support_tau=support_tau,
        iso_q_delta=iso_q_delta,
        iso_q_relative_ppm=iso_q_relative_ppm,
    )
    coarse = build_coarse_basin_rows(signature_rows, pairwise)
    group_cols, grouped_rows = _group_key_map(signature_rows)
    rows: list[dict[str, Any]] = []
    for group_key, group in grouped_rows.groupby(group_cols, dropna=False):
        group_key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        base = dict(zip(group_cols, group_key_values, strict=False))
        group_pairwise = pairwise
        group_coarse = coarse
        for column, value in base.items():
            if column in group_pairwise.columns:
                group_pairwise = group_pairwise[group_pairwise[column] == value]
            if column in group_coarse.columns:
                group_coarse = group_coarse[group_coarse[column] == value]
        p5 = pd.to_numeric(group.get("p5_delta_q"), errors="coerce")
        labeled = group[p5.notna()]
        coarse_count = int(len(group_coarse))
        largest_coarse = (
            int(pd.to_numeric(group_coarse["candidate_count"], errors="coerce").max())
            if not group_coarse.empty
            else 0
        )
        rows.append(
            {
                **base,
                "endpoint_tau": endpoint_tau,
                "support_tau": support_tau,
                "iso_q_delta": iso_q_delta,
                "iso_q_relative_ppm": iso_q_relative_ppm,
                "candidate_count": int(len(labeled)),
                "exact_basin_count": int(labeled["p5_basin_signature"].nunique()),
                "coarse_basin_count": coarse_count,
                "largest_coarse_basin_size": largest_coarse,
                "same_coarse_pair_count": int(
                    group_pairwise.get("same_coarse_basin", False).map(bool).sum()
                )
                if not group_pairwise.empty
                else 0,
                "iso_q_pair_count": int(
                    group_pairwise.get("iso_q_pair", False).map(bool).sum()
                )
                if not group_pairwise.empty
                else 0,
                "partition_distinct_iso_q_pair_count": int(
                    group_pairwise.get("partition_distinct_iso_q_pair", False)
                    .map(bool)
                    .sum()
                )
                if not group_pairwise.empty
                else 0,
                "mean_sample_coassignment_distance": _finite_float(
                    pd.to_numeric(
                        group_pairwise.get("sample_coassignment_distance"),
                        errors="coerce",
                    ).mean()
                )
                if not group_pairwise.empty
                else math.nan,
                "max_sample_coassignment_distance": _finite_float(
                    pd.to_numeric(
                        group_pairwise.get("sample_coassignment_distance"),
                        errors="coerce",
                    ).max()
                )
                if not group_pairwise.empty
                else math.nan,
                "mean_support_jaccard_distance": _finite_float(
                    pd.to_numeric(
                        group_pairwise.get("coarse_support_distance"),
                        errors="coerce",
                    ).mean()
                )
                if not group_pairwise.empty
                else math.nan,
                "max_support_jaccard_distance": _finite_float(
                    pd.to_numeric(
                        group_pairwise.get("coarse_support_distance"),
                        errors="coerce",
                    ).max()
                )
                if not group_pairwise.empty
                else math.nan,
                "support_distance_source": ";".join(
                    sorted(
                        str(value)
                        for value in group_pairwise.get(
                            "coarse_support_distance_source", pd.Series(dtype=object)
                        ).dropna().unique()
                    )
                )
                if not group_pairwise.empty
                else "",
            }
        )
    return pd.DataFrame(rows)


def build_threshold_sensitivity(
    candidates: pd.DataFrame,
    *,
    endpoint_taus: list[float],
    support_taus: list[float],
    material_delta_q: float,
    material_relative_ppm: float,
    iso_q_delta: float,
    iso_q_relative_ppm: float,
) -> pd.DataFrame:
    signature_rows = _signature_frame(candidates)
    signature_rows = _mark_material_gain(
        signature_rows,
        material_delta_q=material_delta_q,
        material_relative_ppm=material_relative_ppm,
    )
    frames: list[pd.DataFrame] = []
    for endpoint_tau in endpoint_taus:
        for support_tau in support_taus:
            frames.append(
                _summarize_threshold(
                    signature_rows,
                    endpoint_tau=endpoint_tau,
                    support_tau=support_tau,
                    iso_q_delta=iso_q_delta,
                    iso_q_relative_ppm=iso_q_relative_ppm,
                )
            )
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def write_report(output_dir: Path, sensitivity: pd.DataFrame) -> None:
    lines = [
        "# Leiden Multi-Basin Threshold Sensitivity",
        "",
        "This is a Dongdaemun diagnostic artifact for coarse-basin threshold calibration.",
        "",
    ]
    if sensitivity.empty:
        lines.append("- No threshold sensitivity rows were available.")
    else:
        display_cols = [
            column
            for column in [
                "candidate_eval_mode",
                "case",
                "seed",
                "candidate_budget",
                "endpoint_tau",
                "support_tau",
                "exact_basin_count",
                "coarse_basin_count",
                "largest_coarse_basin_size",
                "same_coarse_pair_count",
                "partition_distinct_iso_q_pair_count",
            ]
            if column in sensitivity.columns
        ]
        lines.extend(_markdown_table(sensitivity[display_cols]).splitlines())
    (output_dir / "leiden_multibasin_threshold_sensitivity_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    columns = list(frame.columns)
    out = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append("" if math.isnan(value) else f"{value:.6g}")
            else:
                values.append(str(value))
        out.append("| " + " | ".join(values) + " |")
    return "\n".join(out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--endpoint-taus", type=str, default=DEFAULT_ENDPOINT_TAUS)
    parser.add_argument("--support-taus", type=str, default=DEFAULT_SUPPORT_TAUS)
    parser.add_argument("--material-delta-q", type=float, default=1.0)
    parser.add_argument("--material-relative-ppm", type=float, default=10.0)
    parser.add_argument("--iso-q-delta", type=float, default=10.0)
    parser.add_argument("--iso-q-relative-ppm", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = _read_csvs(input_dir, "candidate_level_rows.csv")
    sensitivity = build_threshold_sensitivity(
        candidates,
        endpoint_taus=_parse_float_csv(args.endpoint_taus),
        support_taus=_parse_float_csv(args.support_taus),
        material_delta_q=args.material_delta_q,
        material_relative_ppm=args.material_relative_ppm,
        iso_q_delta=args.iso_q_delta,
        iso_q_relative_ppm=args.iso_q_relative_ppm,
    )
    sensitivity.to_csv(
        output_dir / "leiden_multibasin_threshold_sensitivity.csv", index=False
    )
    write_report(output_dir, sensitivity)
    print(
        {
            "candidate_rows": int(len(candidates)),
            "sensitivity_rows": int(len(sensitivity)),
            "output_dir": str(output_dir),
        }
    )


if __name__ == "__main__":
    main()
