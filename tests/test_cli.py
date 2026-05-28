"""Tests for sciscape CLI parser construction and argument parsing."""

from pathlib import Path

import pandas as pd
import pytest

from sciscape.cli import _build_parser, _run_visualize


@pytest.fixture
def parser():
    return _build_parser()


# ---------------------------------------------------------------------------
# Parser construction
# ---------------------------------------------------------------------------

class TestBuildParser:
    def test_returns_parser(self, parser):
        assert parser.prog == "sciscape"

    @pytest.mark.parametrize("cmd", ["cluster", "keywords", "convert", "landscape", "visualize", "viewer", "gui"])
    def test_subcommands_exist(self, parser, cmd):
        # Should not raise
        parser.parse_args([cmd] if cmd in ("gui",) else self._minimal_args(cmd))

    def test_missing_subcommand_exits(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args([])

    @staticmethod
    def _minimal_args(cmd):
        """Return minimal valid args for each subcommand."""
        if cmd == "cluster":
            return ["cluster", "edges.zip", "edges.txt"]
        if cmd == "keywords":
            return ["keywords", "abs.parquet", "mem.parquet"]
        if cmd == "convert":
            return ["convert", "wos", "data.txt"]
        if cmd == "landscape":
            return ["landscape", "abs.parquet", "edges.parquet"]
        if cmd == "visualize":
            return ["visualize", "keywords.parquet"]
        if cmd == "viewer":
            return ["viewer"]
        if cmd == "gui":
            return ["gui"]
        raise ValueError(cmd)


# ---------------------------------------------------------------------------
# Landscape subcommand
# ---------------------------------------------------------------------------

class TestLandscapeArgs:
    def test_gamma_pre_default(self, parser):
        args = parser.parse_args(["landscape", "a.parquet", "e.parquet"])
        assert args.gamma_pre == "auto"

    def test_gamma_pre_none(self, parser):
        args = parser.parse_args(["landscape", "a.parquet", "e.parquet", "--gamma-pre", "none"])
        assert args.gamma_pre == "none"

    def test_gamma_pre_float(self, parser):
        args = parser.parse_args(["landscape", "a.parquet", "e.parquet", "--gamma-pre", "0.01"])
        assert args.gamma_pre == "0.01"

    def test_gamma_range_default_none(self, parser):
        args = parser.parse_args(["landscape", "a.parquet", "e.parquet"])
        assert args.gamma_range is None

    def test_gamma_range_valid(self, parser):
        args = parser.parse_args(["landscape", "a.parquet", "e.parquet", "--gamma-range", "1e-5,1e-2"])
        assert args.gamma_range == "1e-5,1e-2"

    def test_seed_default(self, parser):
        args = parser.parse_args(["landscape", "a.parquet", "e.parquet"])
        assert args.seed == 42

    def test_seed_explicit(self, parser):
        args = parser.parse_args(["landscape", "a.parquet", "e.parquet", "--seed", "123"])
        assert args.seed == 123

    def test_n_nodes_default(self, parser):
        args = parser.parse_args(["landscape", "a.parquet", "e.parquet"])
        assert args.n_nodes == 100_000

    def test_top_n_default(self, parser):
        args = parser.parse_args(["landscape", "a.parquet", "e.parquet"])
        assert args.top_n == 80

    def test_force_flag(self, parser):
        args = parser.parse_args(["landscape", "a.parquet", "e.parquet", "--force"])
        assert args.force is True

    def test_verbose_flag(self, parser):
        args = parser.parse_args(["landscape", "a.parquet", "e.parquet", "-v"])
        assert args.verbose is True

    def test_output_dir_default(self, parser):
        args = parser.parse_args(["landscape", "a.parquet", "e.parquet"])
        assert args.output_dir == Path("landscape_output")

    def test_output_dir_explicit(self, parser):
        args = parser.parse_args(["landscape", "a.parquet", "e.parquet", "-o", "/tmp/out"])
        assert args.output_dir == Path("/tmp/out")


# ---------------------------------------------------------------------------
# Landscape validation (post-parse, in _run_landscape)
# ---------------------------------------------------------------------------

class TestLandscapeValidation:
    """Test the gamma-range / gamma-pre parsing logic from _run_landscape."""

    @staticmethod
    def _parse_gamma_range(raw: str) -> tuple[float, float]:
        """Replicate the gamma-range validation from _run_landscape."""
        lo, hi = raw.split(",")
        return (float(lo), float(hi))

    @staticmethod
    def _parse_gamma_pre(raw: str):
        """Replicate gamma_pre parsing from _run_landscape."""
        gb = raw.strip().lower()
        if gb == "none":
            return None
        elif gb == "auto":
            return "auto"
        else:
            return float(gb)

    def test_gamma_range_parse_valid(self):
        lo, hi = self._parse_gamma_range("1e-5,1e-2")
        assert lo == pytest.approx(1e-5)
        assert hi == pytest.approx(1e-2)

    def test_gamma_range_parse_invalid_no_comma(self):
        with pytest.raises(ValueError):
            self._parse_gamma_range("bad_value")

    def test_gamma_range_parse_invalid_non_numeric(self):
        with pytest.raises(ValueError):
            self._parse_gamma_range("abc,def")

    def test_gamma_range_parse_invalid_three_parts(self):
        with pytest.raises(ValueError):
            self._parse_gamma_range("1,2,3")

    def test_gamma_pre_auto(self):
        assert self._parse_gamma_pre("auto") == "auto"

    def test_gamma_pre_none(self):
        assert self._parse_gamma_pre("none") is None

    def test_gamma_pre_float(self):
        assert self._parse_gamma_pre("0.01") == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# Convert subcommand
# ---------------------------------------------------------------------------

class TestConvertArgs:
    @pytest.mark.parametrize("source", ["wos", "scopus", "openalex", "bibtex"])
    def test_valid_sources(self, parser, source):
        args = parser.parse_args(["convert", source, "input.txt"])
        assert args.source == source
        assert args.command == "convert"

    def test_invalid_source_exits(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["convert", "invalid_source", "input.txt"])

    def test_output_default(self, parser):
        args = parser.parse_args(["convert", "wos", "input.txt"])
        assert args.output == Path("abstracts.parquet")

    def test_output_explicit(self, parser):
        args = parser.parse_args(["convert", "wos", "input.txt", "-o", "out.parquet"])
        assert args.output == Path("out.parquet")

    def test_encoding_default_none(self, parser):
        args = parser.parse_args(["convert", "wos", "input.txt"])
        assert args.encoding is None

    def test_encoding_explicit(self, parser):
        args = parser.parse_args(["convert", "wos", "input.txt", "--encoding", "utf-8"])
        assert args.encoding == "utf-8"

    def test_keep_no_abstract_flag(self, parser):
        args = parser.parse_args(["convert", "wos", "input.txt", "--keep-no-abstract"])
        assert args.keep_no_abstract is True

    def test_keep_no_abstract_default(self, parser):
        args = parser.parse_args(["convert", "wos", "input.txt"])
        assert args.keep_no_abstract is False


# ---------------------------------------------------------------------------
# Keywords subcommand
# ---------------------------------------------------------------------------

class TestKeywordsArgs:
    def test_basic_parse(self, parser):
        args = parser.parse_args(["keywords", "abs.parquet", "mem.parquet"])
        assert args.command == "keywords"
        assert args.abstract_path == Path("abs.parquet")
        assert args.membership_path == Path("mem.parquet")

    def test_defaults(self, parser):
        args = parser.parse_args(["keywords", "abs.parquet", "mem.parquet"])
        assert args.cluster_level is None
        assert args.top_n == 100
        assert args.include_title is False
        assert args.min_df == 5
        assert args.ngram_max == 3
        assert args.n_jobs == -1
        assert args.enable_all is False
        assert args.output == Path("keywords.parquet")
        assert args.verbose is False

    def test_explicit_options(self, parser):
        args = parser.parse_args([
            "keywords", "abs.parquet", "mem.parquet",
            "--cluster-level", "level_2",
            "--top-n", "50",
            "--include-title",
            "--min-df", "10",
            "--ngram-max", "4",
            "--n-jobs", "4",
            "--enable-all",
            "-o", "kw.parquet",
            "-v",
        ])
        assert args.cluster_level == "level_2"
        assert args.top_n == 50
        assert args.include_title is True
        assert args.min_df == 10
        assert args.ngram_max == 4
        assert args.n_jobs == 4
        assert args.enable_all is True
        assert args.output == Path("kw.parquet")
        assert args.verbose is True


# ---------------------------------------------------------------------------
# Visualize subcommand
# ---------------------------------------------------------------------------

class TestVisualizeArgs:
    def test_basic_parse(self, parser):
        args = parser.parse_args(["visualize", "keywords.parquet"])
        assert args.command == "visualize"
        assert args.keyword_table == Path("keywords.parquet")

    def test_defaults(self, parser):
        args = parser.parse_args(["visualize", "keywords.parquet"])
        assert args.output == Path("sciscape_report")
        assert args.title == "SciScape Keyword Report"
        assert args.dashboard_only is False
        assert args.open is False

    def test_explicit_options(self, parser):
        args = parser.parse_args([
            "visualize",
            "keywords.csv",
            "-o",
            "dashboard.html",
            "--title",
            "Sample Dashboard",
            "--dashboard-only",
            "--open",
        ])
        assert args.output == Path("dashboard.html")
        assert args.title == "Sample Dashboard"
        assert args.dashboard_only is True
        assert args.open is True

    def test_dashboard_only_generates_html_from_minimal_csv(self, parser, tmp_path):
        keyword_path = tmp_path / "keywords.csv"
        pd.DataFrame(
            {
                "cluster_id": [0, 0, 1],
                "term": ["perovskite solar cell", "halide perovskite", "graph neural network"],
            }
        ).to_csv(keyword_path, index=False)

        output_path = tmp_path / "dashboard.html"
        args = parser.parse_args([
            "visualize",
            str(keyword_path),
            "-o",
            str(output_path),
            "--title",
            "Sample Dashboard",
            "--dashboard-only",
        ])

        _run_visualize(args)

        html = output_path.read_text(encoding="utf-8")
        assert "Sample Dashboard" in html
        assert "perovskite solar cell" in html


# ---------------------------------------------------------------------------
# Cluster subcommand
# ---------------------------------------------------------------------------

class TestClusterArgs:
    def test_basic_parse(self, parser):
        args = parser.parse_args(["cluster", "edges.zip", "edges.txt"])
        assert args.command == "cluster"
        assert args.zip_path == Path("edges.zip")
        assert args.inner_name == "edges.txt"

    def test_defaults(self, parser):
        args = parser.parse_args(["cluster", "edges.zip", "edges.txt"])
        assert args.levels is None
        assert args.resolution_bounds == "1e-3,5.0"
        assert args.max_iterations == 32
        assert args.seed is None
        assert args.output == Path("membership.parquet")
        assert args.verbose is False

    def test_levels_single_pair(self, parser):
        args = parser.parse_args(["cluster", "e.zip", "e.txt", "--levels", "5,100"])
        assert args.levels == ["5,100"]

    def test_levels_multiple_pairs(self, parser):
        args = parser.parse_args(["cluster", "e.zip", "e.txt", "--levels", "5,100", "80,500", "400,5000"])
        assert args.levels == ["5,100", "80,500", "400,5000"]

    def test_seed_explicit(self, parser):
        args = parser.parse_args(["cluster", "e.zip", "e.txt", "--seed", "7"])
        assert args.seed == 7

    def test_resolution_bounds_explicit(self, parser):
        args = parser.parse_args(["cluster", "e.zip", "e.txt", "--resolution-bounds", "0.1,10.0"])
        assert args.resolution_bounds == "0.1,10.0"


# ---------------------------------------------------------------------------
# Cluster validation (post-parse, in _run_cluster)
# ---------------------------------------------------------------------------

class TestClusterLevelsValidation:
    """Test the --levels parsing logic from _run_cluster."""

    @staticmethod
    def _parse_levels(pairs: list[str]) -> list[tuple[int, int]]:
        """Replicate --levels validation from _run_cluster."""
        result = []
        for pair in pairs:
            lo, hi = pair.split(",")
            result.append((int(lo), int(hi)))
        return result

    def test_valid_single(self):
        assert self._parse_levels(["5,100"]) == [(5, 100)]

    def test_valid_multiple(self):
        assert self._parse_levels(["5,100", "80,500"]) == [(5, 100), (80, 500)]

    def test_invalid_no_comma(self):
        with pytest.raises(ValueError):
            self._parse_levels(["bad"])

    def test_invalid_non_numeric(self):
        with pytest.raises(ValueError):
            self._parse_levels(["abc,def"])
