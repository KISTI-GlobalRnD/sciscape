"""Tests for sciscape.clustering.leiden_java — unit tests without Java."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest

from sciscape.clustering.leiden_java import (
    JavaLeidenResult,
    _parse_membership_tsv,
    _prepare_edge_tsv,
    _resolve_jar,
    run_leiden_java,
)


# ── _resolve_jar ────────────────────────────────────────────


class TestResolveJar:
    def test_explicit_path(self, tmp_path):
        jar = tmp_path / "leiden.jar"
        jar.touch()
        assert _resolve_jar(jar) == jar

    def test_explicit_path_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            _resolve_jar(tmp_path / "missing.jar")

    def test_env_var(self, tmp_path, monkeypatch):
        jar = tmp_path / "leiden.jar"
        jar.touch()
        monkeypatch.setenv("LEIDEN_JAR", str(jar))
        assert _resolve_jar(None) == jar

    def test_env_var_missing_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LEIDEN_JAR", str(tmp_path / "gone.jar"))
        with pytest.raises(FileNotFoundError, match="LEIDEN_JAR"):
            _resolve_jar(None)

    def test_no_path_no_env(self, monkeypatch):
        monkeypatch.delenv("LEIDEN_JAR", raising=False)
        with pytest.raises(FileNotFoundError, match="not specified"):
            _resolve_jar(None)


# ── _prepare_edge_tsv ──────────────────────────────────────


class TestPrepareEdgeTsv:
    @pytest.fixture
    def int_edges_parquet(self, tmp_path):
        edges = pl.DataFrame({
            "src": [0, 1, 2],
            "dst": [1, 2, 3],
            "weight": [0.5, 1.0, 1.5],
        })
        path = tmp_path / "int_edges.parquet"
        edges.write_parquet(path)
        return path

    def test_writes_tsv(self, int_edges_parquet, tmp_path):
        tsv = tmp_path / "edges.tsv"
        n = _prepare_edge_tsv(int_edges_parquet, tsv, weighted=True)
        assert n == 3
        assert tsv.exists()
        lines = tsv.read_text().strip().split("\n")
        assert len(lines) == 3
        # Check tab-separated format
        parts = lines[0].split("\t")
        assert len(parts) == 3  # src, dst, weight

    def test_unweighted(self, int_edges_parquet, tmp_path):
        tsv = tmp_path / "edges.tsv"
        _prepare_edge_tsv(int_edges_parquet, tsv, weighted=False)
        lines = tsv.read_text().strip().split("\n")
        parts = lines[0].split("\t")
        assert len(parts) == 2  # src, dst only


# ── _parse_membership_tsv ──────────────────────────────────


class TestParseMembershipTsv:
    def test_basic(self, tmp_path):
        path = tmp_path / "membership.tsv"
        path.write_text("0\n0\n1\n1\n2\n")
        mem = _parse_membership_tsv(path, n_nodes=5)
        assert mem.dtype == np.int64
        assert len(mem) == 5
        np.testing.assert_array_equal(mem, [0, 0, 1, 1, 2])

    def test_length_mismatch(self, tmp_path):
        path = tmp_path / "membership.tsv"
        path.write_text("0\n1\n")
        with pytest.raises(ValueError, match="n_nodes"):
            _parse_membership_tsv(path, n_nodes=5)

    def test_single_node(self, tmp_path):
        path = tmp_path / "membership.tsv"
        path.write_text("0\n")
        mem = _parse_membership_tsv(path, n_nodes=1)
        assert len(mem) == 1

    def test_two_column_output(self, tmp_path):
        path = tmp_path / "membership.tsv"
        path.write_text("0\t0\n1\t0\n2\t1\n3\t1\n")
        mem = _parse_membership_tsv(path, n_nodes=4)
        np.testing.assert_array_equal(mem, [0, 0, 1, 1])


# ── run_leiden_java (mocked subprocess) ────────────────────


class TestRunLeidenJava:
    @pytest.fixture
    def setup(self, tmp_path):
        """Create minimal int_edges.parquet and fake JAR."""
        edges = pl.DataFrame({
            "src": [0, 1, 2],
            "dst": [1, 2, 3],
            "weight": [0.5, 1.0, 1.5],
        })
        edge_path = tmp_path / "int_edges.parquet"
        edges.write_parquet(edge_path)

        jar = tmp_path / "leiden.jar"
        jar.touch()

        return edge_path, jar, tmp_path

    def test_success_mocked(self, setup):
        edge_path, jar, tmp_path = setup

        def fake_run(cmd, **kwargs):
            # Write fake membership output
            output_idx = cmd.index("-o") + 1
            output_path = Path(cmd[output_idx])
            output_path.write_text("0\n0\n1\n1\n")

            class FakeResult:
                returncode = 0
                stdout = ""
                stderr = ""
            return FakeResult()

        with patch("sciscape.clustering.leiden_java.subprocess.run", side_effect=fake_run):
            result = run_leiden_java(
                edge_path,
                resolution=0.001,
                n_nodes=4,
                jar_path=jar,
                output_path=tmp_path / "out.tsv",
            )

        assert isinstance(result, JavaLeidenResult)
        assert len(result.membership) == 4
        assert result.n_clusters == 2
        assert result.resolution == 0.001

    def test_java_failure_raises(self, setup):
        edge_path, jar, _ = setup

        def fake_run(cmd, **kwargs):
            class FakeResult:
                returncode = 1
                stdout = ""
                stderr = "OutOfMemoryError"
            return FakeResult()

        with patch("sciscape.clustering.leiden_java.subprocess.run", side_effect=fake_run):
            with pytest.raises(RuntimeError, match="Java Leiden failed"):
                run_leiden_java(
                    edge_path,
                    resolution=0.001,
                    n_nodes=4,
                    jar_path=jar,
                )

    def test_no_jar_raises(self, setup, monkeypatch):
        edge_path, _, _ = setup
        monkeypatch.delenv("LEIDEN_JAR", raising=False)
        with pytest.raises(FileNotFoundError):
            run_leiden_java(
                edge_path,
                resolution=0.001,
                n_nodes=4,
            )
