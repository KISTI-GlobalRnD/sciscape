"""Subprocess wrapper for the CWTS networkanalysis Java Leiden backend.

Runs Leiden community detection via the CWTS ``RunNetworkClustering`` CLI,
which accepts integer edge lists and writes membership files.  This backend
bypasses igraph entirely, enabling clustering on 50M+ node graphs.

This module is **optional** — the rest of sciscape works with Python
igraph + leidenalg alone. Install the ``java`` extra for large-scale
workloads::

    pip install sciscape[java]   # marks intent; actual setup below

Setup
-----
1. Install JDK >= 11 (e.g. ``sudo apt install openjdk-11-jdk``).
2. Build or download the CWTS networkanalysis JAR::

       git clone https://github.com/CWTSLeiden/networkanalysis.git
       cd networkanalysis && ./gradlew build
       # JAR at build/libs/networkanalysis-*.jar

   For ``--fixed-nodes`` support (constrained postprocess), use the
   KISTI fork with the ``feature/fixed-nodes`` branch.

3. Set the JAR path via environment variable or function argument::

       export LEIDEN_JAR=/path/to/networkanalysis.jar

   Or pass ``jar_path=`` to each function call.

4. (Optional) For multi-level classification, also build
   ``publicationclassification``::

       export CLASSIFICATION_JAR=/path/to/publicationclassification.jar
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class JavaLeidenResult:
    """Result of a Java Leiden clustering run."""

    membership: np.ndarray       # int64, length = n_nodes
    n_clusters: int
    resolution: float
    output_path: Path | None     # path to membership TSV (None if temp)


def _resolve_jar(jar_path: Path | None) -> Path:
    """Resolve the JAR path: explicit arg > $LEIDEN_JAR env > error."""
    if jar_path is not None:
        jar = Path(jar_path)
        if not jar.is_file():
            raise FileNotFoundError(f"Leiden JAR not found: {jar}")
        return jar

    env = os.environ.get("LEIDEN_JAR")
    if env:
        jar = Path(env)
        if not jar.is_file():
            raise FileNotFoundError(
                f"$LEIDEN_JAR points to missing file: {jar}"
            )
        return jar

    raise FileNotFoundError(
        "Leiden JAR path not specified. "
        "Pass jar_path= or set $LEIDEN_JAR environment variable. "
        "Download from https://github.com/CWTSLeiden/networkanalysis"
    )


def _prepare_edge_tsv(
    edge_path: Path,
    tsv_path: Path,
    *,
    weighted: bool = True,
    coalesce_undirected: bool = True,
) -> int:
    """Convert int_edges.parquet → TSV edge list for the Java CLI.

    Format: ``src<TAB>dst<TAB>weight`` (or ``src<TAB>dst`` if unweighted).
    Uses streaming scan to avoid loading the full edge table into memory.
    Returns the number of edges written.

    CWTS networkanalysis only supports undirected networks and rejects
    duplicate neighbors. When ``coalesce_undirected`` is true, parallel
    edges and reversed duplicate pairs are collapsed by summing weights.
    """
    lf = pl.scan_parquet(edge_path)

    if coalesce_undirected:
        lf = lf.select(
            pl.min_horizontal("src", "dst").alias("src"),
            pl.max_horizontal("src", "dst").alias("dst"),
            pl.col("weight").cast(pl.Float64).alias("weight"),
        ).filter(pl.col("src") != pl.col("dst"))

        if weighted:
            lf = lf.group_by("src", "dst").agg(pl.col("weight").sum())
        else:
            lf = lf.group_by("src", "dst").agg().select("src", "dst")
    else:
        if weighted:
            lf = lf.select("src", "dst", "weight")
        else:
            lf = lf.select("src", "dst")

    lf.sink_csv(tsv_path, separator="\t", include_header=False)
    return _count_lines(tsv_path)


def _count_lines(path: Path) -> int:
    """Count lines in a text file without loading it into memory."""
    n = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            n += chunk.count(b"\n")
    return n


def _parse_membership_tsv(path: Path, n_nodes: int) -> np.ndarray:
    """Read the Java Leiden output membership file.

    The CWTS CLI writes either:
    - one line per node with a single cluster column, or
    - two tab-separated columns: ``node_idx`` and ``cluster``.
    """
    mem = np.loadtxt(path, dtype=np.int64)
    if mem.ndim == 0:
        mem = mem.reshape(1)
    if mem.ndim == 2:
        if mem.shape[1] != 2:
            raise ValueError(
                f"Expected 2 columns in membership TSV, got shape {mem.shape}"
            )
        nodes = mem[:, 0]
        clusters = mem[:, 1]
        expected = np.arange(n_nodes, dtype=np.int64)
        if nodes.shape[0] != n_nodes or not np.array_equal(nodes, expected):
            raise ValueError(
                "Membership TSV node column does not match expected 0-indexed order"
            )
        return clusters
    if mem.shape[0] != n_nodes:
        raise ValueError(
            f"Membership length ({mem.shape[0]}) != n_nodes ({n_nodes})"
        )
    return mem


def run_leiden_java(
    edge_path: Path,
    *,
    resolution: float,
    n_nodes: int,
    jar_path: Path | None = None,
    output_path: Path | None = None,
    input_clustering_path: Path | None = None,
    fixed_nodes: set[int] | None = None,
    edge_tsv_path: Path | None = None,
    seed: int = 0,
    iterations: int = 10,
    randomness: float = 0.01,
    weighted: bool = True,
    coalesce_undirected_edges: bool = True,
    java_cmd: str = "java",
    java_heap: str = "8g",
) -> JavaLeidenResult:
    """Run Leiden clustering via the CWTS Java backend.

    Parameters
    ----------
    edge_path : Path
        Path to ``int_edges.parquet`` (columns: src, dst, weight).
    resolution : float
        CPM resolution parameter (gamma).
    n_nodes : int
        Total number of nodes (needed to validate output).
    jar_path : Path, optional
        Path to ``networkanalysis-X.Y.Z.jar``. Falls back to ``$LEIDEN_JAR``.
    output_path : Path, optional
        Where to write the membership TSV. If None, uses a temp file.
    fixed_nodes : set[int], optional
        Node indices that should not change cluster assignment.
        Requires the patched CWTS networkanalysis JAR with
        ``--fixed-nodes`` support.
    seed : int
        Random seed for the Leiden algorithm.
    iterations : int
        Number of Leiden iterations.
    randomness : float
        Randomness parameter of the Leiden refinement phase.
    weighted : bool
        Whether to pass edge weights to the algorithm.
    coalesce_undirected_edges : bool
        Collapse duplicate undirected pairs before invoking Java. The CWTS
        backend rejects duplicate neighbors, while the Rust backend treats
        parallel edges as additive weights.
    java_cmd : str
        Java executable (default ``"java"``).
    java_heap : str
        Max Java heap size (e.g. ``"8g"``, ``"32g"``).

    Returns
    -------
    JavaLeidenResult
    """
    jar = _resolve_jar(jar_path)
    if iterations <= 0:
        raise ValueError(
            "Java Leiden CLI requires iterations to be a positive integer; "
            "iterations=0 convergence mode is only supported by the Rust backend."
        )
    edge_path = Path(edge_path)

    use_temp_output = output_path is None
    use_temp_tsv = edge_tsv_path is None

    try:
        # Prepare TSV edge list
        if edge_tsv_path is None:
            with tempfile.NamedTemporaryFile(
                suffix=".tsv", prefix="leiden_edges_", delete=False
            ) as f:
                tsv_path = Path(f.name)
            n_edges = _prepare_edge_tsv(
                edge_path,
                tsv_path,
                weighted=weighted,
                coalesce_undirected=coalesce_undirected_edges,
            )
            log.info(
                "leiden_java: prepared %d edges → %s", n_edges, tsv_path
            )
        else:
            tsv_path = Path(edge_tsv_path)
            if tsv_path.exists():
                n_edges = _count_lines(tsv_path)
                log.info(
                    "leiden_java: reusing %d-edge TSV %s", n_edges, tsv_path
                )
            else:
                tsv_path.parent.mkdir(parents=True, exist_ok=True)
                n_edges = _prepare_edge_tsv(
                    edge_path,
                    tsv_path,
                    weighted=weighted,
                    coalesce_undirected=coalesce_undirected_edges,
                )
                log.info(
                    "leiden_java: prepared %d edges → %s", n_edges, tsv_path
                )

        # Prepare output path
        if use_temp_output:
            with tempfile.NamedTemporaryFile(
                suffix=".tsv", prefix="leiden_out_", delete=False
            ) as f:
                output_path = Path(f.name)

        # Build Java command.
        #
        # The networkanalysis artifact is a library JAR rather than an
        # executable fat jar, so we call the main class explicitly:
        #
        #   java -cp networkanalysis.jar nl.cwts.networkanalysis.run.RunNetworkClustering
        #     -q CPM -r <resolution> -a Leiden
        #     -s 1 --seed <seed> -i <iterations>
        #     [-w] -o <output> <input>
        cmd = [
            java_cmd,
            f"-Xmx{java_heap}",
            "-cp", str(jar),
            "nl.cwts.networkanalysis.run.RunNetworkClustering",
            "-q", "CPM",
            "-r", str(resolution),
            "-a", "Leiden",
            "-s", "1",
            "--seed", str(seed),
            "-i", str(iterations),
            "--randomness", str(randomness),
        ]
        if weighted:
            cmd.append("-w")
        if input_clustering_path is not None:
            cmd.extend(["--input-clustering", str(input_clustering_path)])

        # Write fixed nodes file if provided
        fixed_nodes_path = None
        if fixed_nodes:
            with tempfile.NamedTemporaryFile(
                suffix=".txt", prefix="leiden_fixed_", delete=False, mode="w",
            ) as f:
                fixed_nodes_path = Path(f.name)
                for node in sorted(fixed_nodes):
                    f.write(f"{node}\n")
            cmd.extend(["--fixed-nodes", str(fixed_nodes_path)])
            log.info("leiden_java: %d fixed nodes", len(fixed_nodes))

        cmd.extend(["-o", str(output_path), str(tsv_path)])

        log.info("leiden_java: running %s", " ".join(cmd))

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Java Leiden failed (exit {result.returncode}):\n"
                f"stdout: {result.stdout[:2000]}\n"
                f"stderr: {result.stderr[:2000]}"
            )

        if result.stderr:
            log.debug("leiden_java stderr: %s", result.stderr[:500])

        # Parse output
        membership = _parse_membership_tsv(output_path, n_nodes)
        n_clusters = int(np.max(membership)) + 1

        log.info(
            "leiden_java: %d nodes → %d clusters (gamma=%.6g)",
            n_nodes, n_clusters, resolution,
        )

        return JavaLeidenResult(
            membership=membership,
            n_clusters=n_clusters,
            resolution=resolution,
            output_path=output_path if not use_temp_output else None,
        )

    finally:
        # Clean up temp TSV edge file
        if use_temp_tsv and tsv_path.exists():
            tsv_path.unlink()
        # Clean up temp output if we created it
        if use_temp_output and output_path is not None and output_path.exists():
            output_path.unlink()
        # Clean up temp fixed nodes file
        if fixed_nodes_path is not None and fixed_nodes_path.exists():
            fixed_nodes_path.unlink()


# ── Constrained postprocess ──────────────────────────────────────


def postprocess_small_clusters_java(
    edge_path: Path,
    membership: np.ndarray,
    *,
    resolution: float,
    min_size: int,
    jar_path: Path | None = None,
    seed: int = 0,
    iterations: int = 10,
    weighted: bool = True,
    java_cmd: str = "java",
    java_heap: str = "8g",
) -> JavaLeidenResult:
    """Reassign small-cluster nodes using constrained Leiden.

    Nodes in clusters with size >= ``min_size`` are fixed; all other
    nodes are free to move. Re-runs Leiden at the **same** resolution
    so that unfixed nodes find their optimal assignment while large
    clusters remain intact.

    Parameters
    ----------
    edge_path : Path
        Path to ``int_edges.parquet`` (columns: src, dst, weight).
    membership : numpy.ndarray
        Cluster assignment from the initial Leiden run (int, length n_nodes).
    resolution : float
        Same CPM resolution parameter used in the initial run.
    min_size : int
        Clusters smaller than this are eligible for reassignment.
    jar_path, seed, iterations, weighted, java_cmd, java_heap
        Forwarded to :func:`run_leiden_java`.

    Returns
    -------
    JavaLeidenResult
        Updated clustering with small clusters reassigned.
    """
    from collections import Counter

    n_nodes = len(membership)
    counts = Counter(membership.tolist())

    # Identify large clusters and their node indices
    large_clusters = {cid for cid, cnt in counts.items() if cnt >= min_size}
    fixed = {i for i, cid in enumerate(membership) if cid in large_clusters}

    n_small_nodes = n_nodes - len(fixed)
    n_small_clusters = sum(1 for cid, cnt in counts.items() if cnt < min_size)
    log.info(
        "postprocess: %d small clusters (%d nodes) to reassign, "
        "%d large clusters (%d nodes) fixed",
        n_small_clusters, n_small_nodes,
        len(large_clusters), len(fixed),
    )

    if n_small_nodes == 0:
        log.info("postprocess: no small clusters, skipping")
        return JavaLeidenResult(
            membership=membership,
            n_clusters=len(large_clusters),
            resolution=resolution,
            output_path=None,
        )

    # Write initial clustering to temp file
    with tempfile.NamedTemporaryFile(
        suffix=".tsv", prefix="leiden_init_", delete=False, mode="w",
    ) as f:
        init_path = Path(f.name)
        for i in range(n_nodes):
            f.write(f"{i}\t{membership[i]}\n")

    try:
        result = run_leiden_java(
            edge_path,
            resolution=resolution,
            n_nodes=n_nodes,
            jar_path=jar_path,
            input_clustering_path=init_path,
            fixed_nodes=fixed,
            seed=seed,
            iterations=iterations,
            weighted=weighted,
            java_cmd=java_cmd,
            java_heap=java_heap,
        )

        # Log changes
        changed = int(np.sum(result.membership != membership))
        log.info(
            "postprocess: %d nodes changed cluster (%d → %d clusters)",
            changed, len(counts), result.n_clusters,
        )
        return result

    finally:
        if init_path.exists():
            init_path.unlink()


# ── Multi-level classification (publicationclassification backend) ──


@dataclass(frozen=True)
class LevelConfig:
    """Configuration for one hierarchy level."""

    resolution: float
    min_cluster_size: int  # minimum publications per cluster


@dataclass(frozen=True)
class JavaMultiLevelResult:
    """Result of a multi-level Java classification run."""

    memberships: dict[str, np.ndarray]  # level_name → int64 membership
    n_nodes: int
    level_configs: dict[str, LevelConfig]
    output_path: Path | None


def _resolve_classification_jar(jar_path: Path | None) -> Path:
    """Resolve the publicationclassification JAR path."""
    if jar_path is not None:
        jar = Path(jar_path)
        if not jar.is_file():
            raise FileNotFoundError(f"Classification JAR not found: {jar}")
        return jar

    env = os.environ.get("CLASSIFICATION_JAR")
    if env:
        jar = Path(env)
        if not jar.is_file():
            raise FileNotFoundError(
                f"$CLASSIFICATION_JAR points to missing file: {jar}"
            )
        return jar

    raise FileNotFoundError(
        "Classification JAR path not specified. "
        "Pass jar_path= or set $CLASSIFICATION_JAR environment variable. "
        "Build from https://github.com/CWTSLeiden/publicationclassification"
    )


def _prepare_pub_tsv(n_nodes: int, pub_path: Path, *, core_nodes: set[int] | None = None) -> None:
    """Write publication file: node_idx<TAB>is_core."""
    with open(pub_path, "w") as f:
        for i in range(n_nodes):
            is_core = "true" if (core_nodes is None or i in core_nodes) else "false"
            f.write(f"{i}\t{is_core}\n")


def _prepare_cit_link_tsv(
    edge_path: Path,
    tsv_path: Path,
) -> int:
    """Convert int_edges.parquet → citation link TSV (src, dst, weight)."""
    lf = pl.scan_parquet(edge_path)
    lf = lf.select("src", "dst", "weight")
    lf.sink_csv(tsv_path, separator="\t", include_header=False)
    n_edges = pl.scan_parquet(edge_path).select(pl.len()).collect().item()
    return n_edges


def _parse_classification_tsv(
    path: Path,
    n_nodes: int,
    level_names: list[str],
) -> dict[str, np.ndarray]:
    """Parse multi-level classification output.

    Format: pub_id<TAB>cluster_level1<TAB>cluster_level2<TAB>...
    """
    data = np.loadtxt(path, dtype=np.int64)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    n_cols = data.shape[1]
    # First column is publication ID, rest are cluster assignments
    expected_cols = 1 + len(level_names)
    if n_cols != expected_cols:
        raise ValueError(
            f"Expected {expected_cols} columns (id + {len(level_names)} levels), "
            f"got {n_cols}"
        )

    pub_ids = data[:, 0]
    if pub_ids.shape[0] != n_nodes:
        raise ValueError(
            f"Output has {pub_ids.shape[0]} rows, expected {n_nodes}"
        )

    memberships = {}
    for i, name in enumerate(level_names):
        memberships[name] = data[:, i + 1]

    return memberships


def run_multilevel_java(
    edge_path: Path,
    *,
    n_nodes: int,
    levels: dict[str, LevelConfig],
    jar_path: Path | None = None,
    output_path: Path | None = None,
    largest_component: bool = True,
    iterations: int = 10,
    core_nodes: set[int] | None = None,
    java_cmd: str = "java",
    java_heap: str = "16g",
) -> JavaMultiLevelResult:
    """Run multi-level Leiden classification via CWTS publicationclassification.

    This uses graph contraction between levels with proper node_sizes
    handling internally in the Java library.

    Parameters
    ----------
    edge_path : Path
        Path to ``int_edges.parquet`` (columns: src, dst, weight).
    n_nodes : int
        Total number of nodes.
    levels : dict[str, LevelConfig]
        Ordered dict of level_name → LevelConfig. Up to 3 levels
        (micro, meso, macro). Resolution must be descending.
    jar_path : Path, optional
        Path to publicationclassification fat JAR. Falls back to
        ``$CLASSIFICATION_JAR``.
    output_path : Path, optional
        Where to write the classification TSV. If None, uses a temp file.
    largest_component : bool
        Whether to restrict to the largest connected component.
    iterations : int
        Number of Leiden iterations per level.
    core_nodes : set[int], optional
        Nodes to mark as "core" in the publication file. If None, all
        nodes are core.
    java_cmd : str
        Java executable.
    java_heap : str
        Max Java heap size.

    Returns
    -------
    JavaMultiLevelResult
    """
    jar = _resolve_classification_jar(jar_path)
    edge_path = Path(edge_path)

    level_names = list(levels.keys())
    if len(level_names) > 3:
        raise ValueError("publicationclassification supports up to 3 levels")

    # Pad to exactly 3 levels (Java CLI requires 3)
    # Use resolution=0, threshold=1 for unused levels
    padded_names = level_names + [f"_unused_{i}" for i in range(3 - len(level_names))]
    padded_configs = []
    for name in padded_names:
        if name in levels:
            padded_configs.append(levels[name])
        else:
            padded_configs.append(LevelConfig(resolution=0.0, min_cluster_size=1))

    use_temp_output = output_path is None

    with tempfile.TemporaryDirectory(prefix="leiden_ml_") as tmpdir:
        tmpdir = Path(tmpdir)
        pub_path = tmpdir / "publications.tsv"
        cit_path = tmpdir / "citations.tsv"

        # Prepare input files
        _prepare_pub_tsv(n_nodes, pub_path, core_nodes=core_nodes)
        n_edges = _prepare_cit_link_tsv(edge_path, cit_path)
        log.info(
            "multilevel_java: %d nodes, %d edges, %d levels",
            n_nodes, n_edges, len(level_names),
        )

        if use_temp_output:
            output_path = tmpdir / "classification.tsv"

        # Build command
        # PublicationClassificationCreator args (file mode, 11 args):
        # pub_file cit_link_file classification_file
        # largest_component n_iterations
        # resolution_micro pub_threshold_micro
        # resolution_meso pub_threshold_meso
        # resolution_macro pub_threshold_macro
        cmd = [
            java_cmd,
            f"-Xmx{java_heap}",
            "-cp", str(jar),
            "nl.cwts.publicationclassification.run.PublicationClassificationCreator",
            str(pub_path),
            str(cit_path),
            str(output_path),
            str(largest_component).lower(),
            str(iterations),
        ]
        for cfg in padded_configs:
            cmd.extend([str(cfg.resolution), str(cfg.min_cluster_size)])

        log.info("multilevel_java: running %s", " ".join(cmd[:8]) + " ...")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Java multi-level clustering failed (exit {result.returncode}):\n"
                f"stdout: {result.stdout[:2000]}\n"
                f"stderr: {result.stderr[:2000]}"
            )

        if result.stdout:
            log.info("multilevel_java stdout: %s", result.stdout[:500])

        # Parse output
        memberships = _parse_classification_tsv(
            output_path, n_nodes, level_names,
        )

        for name, mem in memberships.items():
            n_cl = int(np.max(mem)) + 1
            log.info(
                "  %s: %d clusters (γ=%.2e, min_size=%d)",
                name, n_cl, levels[name].resolution, levels[name].min_cluster_size,
            )

        return JavaMultiLevelResult(
            memberships=memberships,
            n_nodes=n_nodes,
            level_configs=dict(levels),
            output_path=output_path if not use_temp_output else None,
        )


__all__ = [
    "JavaLeidenResult",
    "run_leiden_java",
    "LevelConfig",
    "JavaMultiLevelResult",
    "run_multilevel_java",
]
