"""Ensemble clustering utilities for Leiden runs across gamma grids and seeds."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import math
import json
import multiprocessing as mp

import igraph as ig
import polars as pl
import leidenalg as la

from .config import EnsembleConfig, LeidenConfig
from .graph import build_graph, giant_component
from .io import load_edge_table
from .partitioning import partition_class
from .logging import PROGRESS_LOG_FILE, resolve_log_path, write_progress_event


# Globals used by worker processes
_GLOBAL_GRAPH: Optional[ig.Graph] = None
_GLOBAL_WEIGHTS = None
_GLOBAL_OBJECTIVE: Optional[str] = None
_GLOBAL_ITERATIONS: Optional[int] = None
_GLOBAL_NORMALIZE: Optional[bool] = None


def _init_worker(graph: ig.Graph, weights, objective: str, iterations: Optional[int], normalize: bool) -> None:
    global _GLOBAL_GRAPH, _GLOBAL_WEIGHTS, _GLOBAL_OBJECTIVE, _GLOBAL_ITERATIONS, _GLOBAL_NORMALIZE
    _GLOBAL_GRAPH = graph
    _GLOBAL_WEIGHTS = weights
    _GLOBAL_OBJECTIVE = objective
    _GLOBAL_ITERATIONS = iterations
    _GLOBAL_NORMALIZE = normalize


def _worker_task(args: Tuple[float, int]) -> EnsembleMembership:
    gamma, seed = args
    return _compute_membership(
        _GLOBAL_GRAPH,
        _GLOBAL_WEIGHTS,
        _GLOBAL_OBJECTIVE,
        gamma,
        seed,
        _GLOBAL_ITERATIONS,
        _GLOBAL_NORMALIZE,
        progress_cb=None,
    )


def _handle_membership(
    entry: EnsembleMembership,
    uids: Sequence[str],
    config: EnsembleConfig,
    memberships: List[EnsembleMembership],
    summary_rows: List[Dict[str, object]],
    gamma_seed_files: Dict[float, List[Tuple[int, Path]]],
    gamma_stats: Dict[float, Dict[str, Dict[str, float]]],
    total_nodes: int,
) -> None:
    counts = Counter(entry.membership)
    largest_size = max(counts.values()) if counts else 0
    tiny_threshold = max(
        config.min_cluster_size or 0,
        config.min_cluster_ratio * total_nodes,
    )
    tiny_total = sum(size for size in counts.values() if size < tiny_threshold)
    num_clusters = len(counts)
    num_non_tiny = sum(1 for size in counts.values() if size >= tiny_threshold)

    summary_rows.append(
        {
            "gamma": entry.gamma,
            "seed": entry.seed,
            "cluster_count": entry.cluster_count,
            "quality": entry.quality,
            "largest_size": float(largest_size),
            "largest_ratio": float(largest_size / total_nodes) if total_nodes else 0.0,
            "tiny_total": float(tiny_total),
            "tiny_ratio": float(tiny_total / total_nodes) if total_nodes else 0.0,
            "non_tiny_total": float(total_nodes - tiny_total),
            "non_tiny_ratio": float((total_nodes - tiny_total) / total_nodes) if total_nodes else 0.0,
            "num_clusters": float(num_clusters),
            "num_non_tiny_clusters": float(num_non_tiny),
        }
    )

    if config.retain_memberships:
        memberships.append(entry)

    if config.output_dir:
        gamma_dir = Path(config.output_dir) / f"gamma_{entry.gamma:.6g}"
        gamma_dir.mkdir(parents=True, exist_ok=True)
        seed_path = gamma_dir / f"membership_seed_{entry.seed}.parquet"
        pl.DataFrame({
            "uid": list(uids),
            "membership": entry.membership,
        }).write_parquet(seed_path)

        gamma_seed_files.setdefault(entry.gamma, []).append((entry.seed, seed_path))
        info = gamma_stats.setdefault(
            entry.gamma,
            {
                "cluster_counts": {},
                "qualities": {},
                "largest_sizes": {},
                "tiny_totals": {},
                "non_tiny_totals": {},
                "non_tiny_counts": {},
                "total_nodes": total_nodes,
            },
        )
        info["cluster_counts"][str(entry.seed)] = float(entry.cluster_count)
        info["qualities"][str(entry.seed)] = float(entry.quality)
        info["largest_sizes"][str(entry.seed)] = float(largest_size)
        info["tiny_totals"][str(entry.seed)] = float(tiny_total)
        info["non_tiny_totals"][str(entry.seed)] = float(total_nodes - tiny_total)
        info["non_tiny_counts"][str(entry.seed)] = float(num_non_tiny)


@dataclass
class EnsembleMembership:
    gamma: float
    seed: int
    cluster_count: int
    quality: float
    membership: List[int]
    label_mapping: Dict[int, int]


@dataclass
class EnsembleResult:
    uids: List[str]
    memberships: List[EnsembleMembership]
    gamma_values: List[float]
    seeds: List[int]
    output_dir: Optional[Path] = None
    summary_rows: List[Dict[str, object]] | None = None

    def to_frame(self) -> pl.DataFrame:
        """Return a long-format Polars DataFrame of all memberships."""

        if self.memberships:
            records: List[Dict[str, object]] = []
            for entry in self.memberships:
                for uid, label in zip(self.uids, entry.membership):
                    records.append(
                        {
                            "uid": uid,
                            "gamma": entry.gamma,
                            "seed": entry.seed,
                            "cluster": label,
                            "cluster_count": entry.cluster_count,
                            "quality": entry.quality,
                        }
                    )
            return pl.DataFrame(records)

        if self.output_dir is None:
            raise ValueError("No memberships retained; set retain_memberships=True or provide output_dir")

        frames: List[pl.DataFrame] = []
        summary = self.summary()
        if summary.is_empty():
            return summary

        summary_lookup = summary.to_dict(as_series=False)
        gamma_map = {}
        for gamma, seed, cluster_count, quality in zip(
            summary_lookup["gamma"],
            summary_lookup["seed"],
            summary_lookup["cluster_count"],
            summary_lookup["quality"],
        ):
            gamma_map.setdefault(gamma, {})[seed] = (cluster_count, quality)

        for gamma in self.gamma_values:
            gamma_dir = self.output_dir / f"gamma_{gamma:.6g}"
            aggregated = gamma_dir / "memberships.parquet"
            if not aggregated.exists():
                continue
            df = pl.read_parquet(aggregated)
            seed_cols = [c for c in df.columns if c.startswith("membership_seed_")]
            if not seed_cols:
                continue
            long = df.unpivot(index=["uid"], on=seed_cols, variable_name="seed_col", value_name="cluster")
            long = long.with_columns(
                pl.lit(gamma).alias("gamma"),
                pl.col("seed_col")
                .str.replace("membership_seed_", "")
                .cast(pl.Int64)
                .alias("seed"),
            ).drop("seed_col")
            if gamma in gamma_map:
                rows = []
                for seed, (cluster_count, quality) in gamma_map[gamma].items():
                    rows.append(
                        {
                            "seed": int(seed),
                            "cluster_count": int(cluster_count) if cluster_count is not None else None,
                            "quality": float(quality) if quality is not None else None,
                        }
                    )
                if rows:
                    metrics = pl.DataFrame(rows)
                    long = long.join(metrics, on="seed", how="left")
            frames.append(long)

        return pl.concat(frames) if frames else pl.DataFrame()

    def summary(self) -> pl.DataFrame:
        """Return a summary table aggregating cluster counts/quality per gamma and seed."""

        if self.summary_rows:
            df = pl.DataFrame(self.summary_rows)
        elif self.output_dir and (self.output_dir / "ensemble_summary.parquet").exists():
            df = pl.read_parquet(self.output_dir / "ensemble_summary.parquet")
        else:
            rows = [
                {
                    "gamma": entry.gamma,
                    "seed": entry.seed,
                    "cluster_count": entry.cluster_count,
                    "quality": entry.quality,
                    "largest_size": len(entry.membership) if entry.membership else None,
                    "largest_ratio": (len(entry.membership) / len(self.uids)) if entry.membership else None,
                    "tiny_total": None,
                    "tiny_ratio": None,
                    "num_clusters": entry.cluster_count,
                }
                for entry in self.memberships
            ]
            df = pl.DataFrame(rows)

        expected = {
            "cluster_count": None,
            "quality": None,
            "largest_size": None,
            "largest_ratio": None,
            "tiny_total": None,
            "tiny_ratio": None,
            "non_tiny_total": None,
            "non_tiny_ratio": None,
            "num_clusters": None,
            "num_non_tiny_clusters": None,
        }
        for col in expected:
            if col not in df.columns:
                df = df.with_columns(pl.lit(expected[col]).alias(col))
        return df

    def gamma_metrics(self) -> pl.DataFrame:
        """Aggregate cluster-size and quality metrics per gamma."""

        df = self.summary()
        return (
            df.group_by("gamma")
            .agg([
                pl.mean("largest_ratio").alias("avg_largest_ratio"),
                pl.max("largest_ratio").alias("max_largest_ratio"),
                pl.mean("tiny_ratio").alias("avg_tiny_ratio"),
                pl.mean("non_tiny_ratio").alias("avg_non_tiny_ratio"),
                pl.mean("cluster_count").alias("avg_cluster_count"),
                pl.mean("num_clusters").alias("avg_num_clusters"),
                pl.mean("num_non_tiny_clusters").alias("avg_num_non_tiny_clusters"),
                pl.mean("quality").alias("avg_quality"),
            ])
            .sort("gamma")
        )


def _generate_gamma_values(config: EnsembleConfig) -> List[float]:
    if config.gamma_values:
        unique = sorted({float(g) for g in config.gamma_values if g > 0})
        if not unique:
            raise ValueError("gamma_values must contain positive numbers")
        return unique

    if config.gamma_count < 2:
        return [config.gamma_min]

    if config.gamma_scale.lower() == "log":
        if config.gamma_min <= 0 or config.gamma_max <= 0:
            raise ValueError("gamma_min and gamma_max must be positive for log scale")
        start = math.log10(config.gamma_min)
        end = math.log10(config.gamma_max)
        step = (end - start) / (config.gamma_count - 1)
        return [10 ** (start + i * step) for i in range(config.gamma_count)]

    # linear scale
    step = (config.gamma_max - config.gamma_min) / (config.gamma_count - 1)
    return [config.gamma_min + i * step for i in range(config.gamma_count)]


def _normalize_membership(membership: Sequence[int]) -> Tuple[List[int], Dict[int, int]]:
    counts = Counter(membership)
    ordering = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    mapping = {old: idx for idx, (old, _) in enumerate(ordering)}
    normalized = [mapping[label] for label in membership]
    return normalized, mapping


def _prepare_progress_hook(
    progress: Optional[Callable[[str], None]],
    *,
    log: bool,
    progress_log_path: Path,
) -> Callable[[str], None]:
    def emit(message: str) -> None:
        if log:
            write_progress_event(message, path=progress_log_path)
        if progress:
            progress(message)

    return emit


def _compute_membership(
    graph: ig.Graph,
    weights,
    objective: str,
    gamma: float,
    seed: int,
    iterations: Optional[int],
    normalize: bool,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> EnsembleMembership:
    partition_cls = partition_class(objective)
    kwargs = {"seed": seed}
    if iterations is not None:
        kwargs["n_iterations"] = iterations

    partition = la.find_partition(
        graph,
        partition_type=partition_cls,
        weights=weights,
        resolution_parameter=gamma,
        **kwargs,
    )
    membership = list(partition.membership)
    mapping: Dict[int, int] = {}
    if normalize:
        membership, mapping = _normalize_membership(membership)

    cluster_count = len(set(membership))
    quality = float(partition.quality())

    if progress_cb:
        progress_cb(
            f"ensemble: gamma={gamma:.6g}, seed={seed} -> {cluster_count} clusters (quality={quality:.6f})"
        )

    return EnsembleMembership(
        gamma=gamma,
        seed=seed,
        cluster_count=cluster_count,
        quality=quality,
        membership=membership,
        label_mapping=mapping,
    )


def run_ensemble_pipeline(
    zip_path: Path,
    inner_name: str,
    leiden_config: LeidenConfig,
    ensemble_config: EnsembleConfig,
) -> EnsembleResult:
    """Run an ensemble of Leiden partitions across gamma values and seeds."""

    edges = load_edge_table(zip_path, inner_name)
    graph = build_graph(edges)
    giant = giant_component(graph)
    uids = list(giant.vs["uid"])
    total_nodes = len(uids)

    gamma_values = _generate_gamma_values(ensemble_config)
    seeds = list(ensemble_config.seeds)
    resolved_progress_log_path = resolve_log_path(
        default_path=PROGRESS_LOG_FILE,
        explicit_path=ensemble_config.progress_log_path or leiden_config.progress_log_path,
        log_dir=ensemble_config.log_dir or leiden_config.log_dir or ensemble_config.output_dir,
        run_id=ensemble_config.run_id or leiden_config.run_id,
    )
    progress_cb = _prepare_progress_hook(
        ensemble_config.progress,
        log=bool(leiden_config.log_history or ensemble_config.progress),
        progress_log_path=resolved_progress_log_path,
    )

    weights = giant.es["weight"] if "weight" in giant.es.attributes() else None
    iterations = (
        ensemble_config.n_iterations
        if ensemble_config.n_iterations is not None
        else leiden_config.leiden_iterations
    )

    memberships: List[EnsembleMembership] = []
    summary_rows: List[Dict[str, object]] = []
    gamma_seed_files: Dict[float, List[Tuple[int, Path]]] = {}
    gamma_stats: Dict[float, Dict[str, Dict[str, float]]] = {}

    output_dir = Path(ensemble_config.output_dir) if ensemble_config.output_dir else None
    existing_metrics: Dict[Tuple[float, int], Dict[str, object]] = {}
    existing_metadata: Dict[float, Dict[str, object]] = {}
    if output_dir and output_dir.exists():
        summary_path = output_dir / "ensemble_summary.parquet"
        if summary_path.exists():
            existing_summary_df = pl.read_parquet(summary_path)
            for row in existing_summary_df.iter_rows(named=True):
                key = (float(row["gamma"]), int(row["seed"]))
                existing_metrics[key] = dict(row)
        for gamma in gamma_values:
            metadata_path = output_dir / f"gamma_{gamma:.6g}" / "metadata.json"
            if metadata_path.exists():
                with metadata_path.open() as fh:
                    existing_metadata[gamma] = json.load(fh)

    tasks_to_run: List[Tuple[float, int]] = []
    for gamma in gamma_values:
        gamma_dir = output_dir / f"gamma_{gamma:.6g}" if output_dir else None
        metadata = existing_metadata.get(gamma)
        for seed in seeds:
            key = (gamma, seed)
            membership_path = None
            if gamma_dir:
                membership_path = gamma_dir / f"membership_seed_{seed}.parquet"
            if (
                key in existing_metrics
                and membership_path is not None
                and membership_path.exists()
            ):
                row = existing_metrics[key]
                summary_rows.append(row)

                if metadata:
                    total_nodes_meta = metadata.get("total_nodes", total_nodes)
                else:
                    total_nodes_meta = total_nodes

                stats = gamma_stats.setdefault(
                    gamma,
                    {
                        "cluster_counts": {},
                        "qualities": {},
                        "largest_sizes": {},
                        "tiny_totals": {},
                        "non_tiny_totals": {},
                        "non_tiny_counts": {},
                        "total_nodes": total_nodes_meta,
                    },
                )
                seed_key = str(seed)
                cluster_val = (
                    (metadata.get("cluster_counts", {}) or {}).get(seed_key)
                    if metadata
                    else row.get("cluster_count")
                )
                quality_val = (
                    (metadata.get("qualities", {}) or {}).get(seed_key)
                    if metadata
                    else row.get("quality")
                )
                largest_val = (
                    (metadata.get("largest_sizes", {}) or {}).get(seed_key)
                    if metadata
                    else row.get("largest_size")
                )
                tiny_val = (
                    (metadata.get("tiny_totals", {}) or {}).get(seed_key)
                    if metadata
                    else row.get("tiny_total")
                )
                non_tiny_val = (
                    (metadata.get("non_tiny_totals", {}) or {}).get(seed_key)
                    if metadata and metadata.get("non_tiny_totals")
                    else row.get("non_tiny_total")
                )
                non_tiny_count_val = (
                    (metadata.get("non_tiny_counts", {}) or {}).get(seed_key)
                    if metadata and metadata.get("non_tiny_counts")
                    else row.get("num_non_tiny_clusters")
                )

                if cluster_val is None:
                    cluster_val = row.get("cluster_count")
                if quality_val is None:
                    quality_val = row.get("quality")
                if largest_val is None:
                    largest_val = row.get("largest_size")
                if tiny_val is None:
                    tiny_val = row.get("tiny_total")
                if non_tiny_val is None and row.get("non_tiny_total") is not None:
                    non_tiny_val = row.get("non_tiny_total")
                if non_tiny_count_val is None and row.get("num_non_tiny_clusters") is not None:
                    non_tiny_count_val = row.get("num_non_tiny_clusters")

                if cluster_val is not None:
                    stats["cluster_counts"][seed_key] = float(cluster_val)
                if quality_val is not None:
                    stats["qualities"][seed_key] = float(quality_val)
                if largest_val is not None:
                    stats["largest_sizes"][seed_key] = float(largest_val)
                if tiny_val is not None:
                    stats["tiny_totals"][seed_key] = float(tiny_val)
                if non_tiny_val is not None:
                    stats["non_tiny_totals"][seed_key] = float(non_tiny_val)
                if non_tiny_count_val is not None:
                    stats["non_tiny_counts"][seed_key] = float(non_tiny_count_val)

                if membership_path is not None:
                    gamma_seed_files.setdefault(gamma, []).append((seed, membership_path))

                if progress_cb:
                    progress_cb(
                        f"ensemble: gamma={gamma:.6g}, seed={seed} -> using cached result"
                    )
                continue

            tasks_to_run.append((gamma, seed))

    if ensemble_config.parallel and len(tasks_to_run) > 1:
        from concurrent.futures import ProcessPoolExecutor

        start_method = ensemble_config.start_method.lower()
        if start_method not in {"spawn", "fork"}:
            raise ValueError("start_method must be 'spawn' or 'fork'")

        ctx = mp.get_context(start_method)
        pool_kwargs = {"max_workers": ensemble_config.workers, "mp_context": ctx}

        with ProcessPoolExecutor(
            initializer=_init_worker,
            initargs=(giant, weights, leiden_config.objective, iterations, ensemble_config.normalize_labels),
            **pool_kwargs,
        ) as executor:
            for entry in executor.map(_worker_task, tasks_to_run):
                _handle_membership(
                    entry,
                    uids,
                    ensemble_config,
                    memberships,
                    summary_rows,
                    gamma_seed_files,
                    gamma_stats,
                    total_nodes,
                )
                progress_cb(
                    f"ensemble: gamma={entry.gamma:.6g}, seed={entry.seed} -> {entry.cluster_count} clusters (quality={entry.quality:.6f})"
                )
    elif tasks_to_run:
        for gamma, seed in tasks_to_run:
            entry = _compute_membership(
                giant,
                weights,
                leiden_config.objective,
                gamma,
                seed,
                iterations,
                ensemble_config.normalize_labels,
                progress_cb,
            )
            _handle_membership(
                entry,
                uids,
                ensemble_config,
                memberships,
                summary_rows,
                gamma_seed_files,
                gamma_stats,
                total_nodes,
            )
    else:
        progress_cb("ensemble: no new tasks to run (all seeds cached)")

    if ensemble_config.retain_memberships:
        memberships.sort(key=lambda m: (m.gamma, m.seed))

    if ensemble_config.output_dir:
        _finalize_outputs(
            Path(ensemble_config.output_dir),
            uids,
            gamma_seed_files,
            gamma_stats,
            summary_rows,
        )

    return EnsembleResult(
        uids=uids,
        memberships=memberships if ensemble_config.retain_memberships else [],
        gamma_values=gamma_values,
        seeds=seeds,
        output_dir=Path(ensemble_config.output_dir) if ensemble_config.output_dir else None,
        summary_rows=summary_rows,
    )


def _finalize_outputs(
    output_dir: Path,
    uids: Sequence[str],
    gamma_seed_files: Dict[float, List[Tuple[int, Path]]],
    gamma_stats: Dict[float, Dict[str, Dict[str, float]]],
    summary_rows: Sequence[Dict[str, object]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if summary_rows:
        pl.DataFrame(summary_rows).write_parquet(output_dir / "ensemble_summary.parquet")

    for gamma, files in gamma_seed_files.items():
        gamma_dir = output_dir / f"gamma_{gamma:.6g}"
        gamma_dir.mkdir(parents=True, exist_ok=True)

        df = pl.DataFrame({"uid": list(uids)})
        for seed, path in sorted(files, key=lambda x: x[0]):
            membership_series = pl.read_parquet(path)["membership"]
            df = df.with_columns(membership_series.alias(f"membership_seed_{seed}"))
        df.write_parquet(gamma_dir / "memberships.parquet")

        stats = gamma_stats.get(
            gamma,
            {
                "cluster_counts": {},
                "qualities": {},
                "largest_sizes": {},
                "tiny_totals": {},
                "non_tiny_totals": {},
                "non_tiny_counts": {},
                "total_nodes": len(uids),
            },
        )
        metadata = {
            "gamma": gamma,
            "seeds": sorted(stats["cluster_counts"], key=lambda s: int(s)) if stats["cluster_counts"] else [],
            "cluster_counts": stats["cluster_counts"],
            "qualities": stats["qualities"],
            "largest_sizes": stats.get("largest_sizes", {}),
            "tiny_totals": stats.get("tiny_totals", {}),
            "non_tiny_totals": stats.get("non_tiny_totals", {}),
            "non_tiny_counts": stats.get("non_tiny_counts", {}),
            "total_nodes": stats.get("total_nodes", len(uids)),
        }
        metadata_path = gamma_dir / "metadata.json"
        with (gamma_dir / "metadata.json").open("w", encoding="utf-8") as fh:
            json.dump(metadata, fh, ensure_ascii=False, indent=2)


__all__ = [
    "EnsembleConfig",
    "EnsembleMembership",
    "EnsembleResult",
    "run_ensemble_pipeline",
]
