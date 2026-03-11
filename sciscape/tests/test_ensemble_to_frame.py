from __future__ import annotations

from pathlib import Path

import polars as pl

from sciscape.clustering.ensemble import EnsembleResult


def test_ensemble_to_frame_reads_all_gamma_dirs(tmp_path: Path) -> None:
    uids = ["u1", "u2", "u3"]
    gamma_values = [1.0, 2.0]
    seeds = [0, 1]

    # Write per-gamma aggregated membership tables that `to_frame()` expects.
    for gamma in gamma_values:
        gamma_dir = tmp_path / f"gamma_{gamma:.6g}"
        gamma_dir.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(
            {
                "uid": uids,
                "membership_seed_0": [0, 0, 1],
                "membership_seed_1": [1, 1, 0],
            }
        ).write_parquet(gamma_dir / "memberships.parquet")

    # Provide summary rows so `to_frame()` can attach cluster_count/quality.
    summary_rows = []
    for gamma in gamma_values:
        for seed in seeds:
            summary_rows.append(
                {
                    "gamma": gamma,
                    "seed": seed,
                    "cluster_count": 2,
                    "quality": 0.123 + 0.01 * seed,
                    "largest_size": 2,
                    "largest_ratio": 2 / 3,
                    "tiny_total": 0,
                    "tiny_ratio": 0.0,
                    "non_tiny_total": 3,
                    "non_tiny_ratio": 1.0,
                    "num_clusters": 2,
                    "num_non_tiny_clusters": 2,
                }
            )

    result = EnsembleResult(
        uids=uids,
        memberships=[],
        gamma_values=gamma_values,
        seeds=seeds,
        output_dir=tmp_path,
        summary_rows=summary_rows,
    )

    frame = result.to_frame()
    assert frame.height == len(uids) * len(gamma_values) * len(seeds)
    assert set(frame.columns) >= {"uid", "gamma", "seed", "cluster", "cluster_count", "quality"}

    # Sanity: we should see rows for both gammas.
    assert set(frame["gamma"].unique().to_list()) == set(gamma_values)
