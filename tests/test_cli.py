"""Tests for sciscape CLI parser construction and argument parsing."""

import json
from pathlib import Path

import pandas as pd
import pytest

from sciscape.artifacts import validate_evolution_artifact
from sciscape.cli import _build_parser, _run_evolution, _run_query, _run_visualize


@pytest.fixture
def parser():
    return _build_parser()


# ---------------------------------------------------------------------------
# Parser construction
# ---------------------------------------------------------------------------

class TestBuildParser:
    def test_returns_parser(self, parser):
        assert parser.prog == "sciscape"

    @pytest.mark.parametrize("cmd", ["query", "cluster", "keywords", "convert", "landscape", "visualize", "viewer", "evolution", "gui"])
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
        if cmd == "query":
            return ["query", "graph neural networks"]
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
        if cmd == "evolution":
            return ["evolution", "result", "slices.parquet", "states.parquet", "transitions.parquet"]
        if cmd == "gui":
            return ["gui"]
        raise ValueError(cmd)


# ---------------------------------------------------------------------------
# Query subcommand
# ---------------------------------------------------------------------------

class TestQueryArgs:
    def test_openalex_retry_defaults(self, parser):
        args = parser.parse_args(["query", "graph neural networks"])
        assert args.request_timeout == 30.0
        assert args.max_retries == 3
        assert args.backoff_base == 1.0
        assert args.backoff_max == 30.0

    def test_openalex_retry_explicit(self, parser):
        args = parser.parse_args([
            "query",
            "graph neural networks",
            "--request-timeout", "12",
            "--max-retries", "5",
            "--backoff-base", "0.25",
            "--backoff-max", "8",
        ])
        assert args.request_timeout == 12.0
        assert args.max_retries == 5
        assert args.backoff_base == 0.25
        assert args.backoff_max == 8.0

    def test_run_query_passes_retry_config(self, parser, monkeypatch, tmp_path, capsys):
        captured = {}

        class FakeResult:
            n_works = 0
            n_edges = {}
            abstracts_path = None
            edges_path = None
            landscape_dir = None

        def fake_run(config):
            captured["config"] = config
            return FakeResult()

        monkeypatch.setattr("sciscape.openalex.run_openalex_pipeline", fake_run)
        args = parser.parse_args([
            "query",
            "graph neural networks",
            "--request-timeout", "12",
            "--max-retries", "5",
            "--backoff-base", "0.25",
            "--backoff-max", "8",
            "-o", str(tmp_path / "out"),
        ])

        _run_query(args)

        config = captured["config"]
        assert config.request_timeout == 12.0
        assert config.max_retries == 5
        assert config.backoff_base == 0.25
        assert config.backoff_max == 8.0
        assert "Done: 0 works" in capsys.readouterr().out


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
        assert args.keyword_engine == "legacy"
        assert args.cluster_sharded_output_dir is None
        assert args.keyword_preflight_only is False
        assert args.uid_col == "uid"
        assert args.title_col == "title"
        assert args.abstract_col == "abstract"
        assert args.year_col == "pubyear"
        assert args.target_docs_per_shard == 500_000
        assert args.max_clusters_per_shard == 1024
        assert args.candidate_pool_floor == 256
        assert args.candidate_pool_large == 1024
        assert args.candidate_pool_hard_max == 1536
        assert args.global_candidate_row_warning == 80_000_000
        assert args.global_candidate_row_hard_stop == 100_000_000
        assert args.candidate_mining_progress_interval_docs == 25_000
        assert args.candidate_mining_prune_interval_docs == 50_000
        assert args.candidate_mining_prune_multiplier == 8
        assert args.include_title is False
        assert args.min_df == 5
        assert args.ngram_max == 3
        assert args.n_jobs == -1
        assert args.quality_rerank is False
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
            "--quality-rerank",
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
        assert args.quality_rerank is True
        assert args.enable_all is True
        assert args.output == Path("kw.parquet")
        assert args.verbose is True

    def test_cluster_sharded_preflight_options(self, parser):
        args = parser.parse_args([
            "keywords",
            "abs.parquet",
            "mem.parquet",
            "--keyword-engine",
            "cluster_sharded",
            "--keyword-preflight-only",
            "--cluster-sharded-output-dir",
            "keyword_v2",
            "--uid-col",
            "work_id",
            "--year-col",
            "publication_year",
            "--target-docs-per-shard",
            "250000",
            "--max-clusters-per-shard",
            "512",
            "--candidate-pool-floor",
            "128",
            "--candidate-pool-large",
            "768",
            "--candidate-pool-hard-max",
            "1024",
            "--candidate-mining-progress-interval-docs",
            "10000",
            "--candidate-mining-prune-interval-docs",
            "20000",
            "--candidate-mining-prune-multiplier",
            "6",
        ])
        assert args.keyword_engine == "cluster_sharded"
        assert args.keyword_preflight_only is True
        assert args.cluster_sharded_output_dir == Path("keyword_v2")
        assert args.uid_col == "work_id"
        assert args.year_col == "publication_year"
        assert args.target_docs_per_shard == 250_000
        assert args.max_clusters_per_shard == 512
        assert args.candidate_pool_floor == 128
        assert args.candidate_pool_large == 768
        assert args.candidate_pool_hard_max == 1024
        assert args.candidate_mining_progress_interval_docs == 10_000
        assert args.candidate_mining_prune_interval_docs == 20_000
        assert args.candidate_mining_prune_multiplier == 6


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
        assert args.output == Path("workspace/reports/sciscape_report")
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
        manifest_path = tmp_path / "exports" / "keyword_dashboard" / "export_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["selection"]["view"]["surface"] == "cli_visualize"
        assert manifest["selection"]["layer_state"]["command"] == "visualize"
        assert manifest["selection"]["layer_state"]["dashboard_only"] is True
        assert manifest["selection"]["layer_state"]["keyword_table"] == str(keyword_path)


# ---------------------------------------------------------------------------
# Evolution subcommand
# ---------------------------------------------------------------------------

class TestEvolutionArgs:
    def test_basic_parse(self, parser):
        args = parser.parse_args([
            "evolution",
            "result",
            "slices.parquet",
            "states.parquet",
            "transitions.parquet",
        ])
        assert args.command == "evolution"
        assert args.result_root == Path("result")
        assert args.slices_table == Path("slices.parquet")
        assert args.state_evidence_table == Path("states.parquet")
        assert args.transition_evidence_table == Path("transitions.parquet")

    def test_defaults(self, parser):
        args = parser.parse_args([
            "evolution",
            "result",
            "slices.parquet",
            "states.parquet",
            "transitions.parquet",
        ])
        assert args.evolution_id == "cluster_evolution"
        assert args.metric == "transition_score"
        assert args.output_dir is None
        assert args.temporal_manifest is None
        assert args.default_level == "cluster"
        assert args.allow_skip_slices is False
        assert args.min_transition_score == 0.5
        assert args.min_support_count == 1

    def test_explicit_options(self, parser):
        args = parser.parse_args([
            "evolution",
            "result",
            "slices.csv",
            "states.csv",
            "transitions.csv",
            "--evolution-id",
            "term_evolution",
            "--metric",
            "term_overlap",
            "--title",
            "Term Evolution",
            "--output-dir",
            "custom_evolution",
            "--temporal-manifest",
            "temporal/temporal_manifest.json",
            "--default-level",
            "micro",
            "--allow-skip-slices",
            "--min-transition-score",
            "0.25",
            "--min-support-count",
            "2",
            "--matching-method",
            '{"tie_policy":"keep_all"}',
            "--event-rules",
            '{"ambiguous_score_margin":0.1}',
            "--periodization",
            '{"unit":"year"}',
            "--entity-scope",
            '{"cluster_level":"micro"}',
        ])
        assert args.evolution_id == "term_evolution"
        assert args.metric == "term_overlap"
        assert args.title == "Term Evolution"
        assert args.output_dir == Path("custom_evolution")
        assert args.temporal_manifest == Path("temporal/temporal_manifest.json")
        assert args.default_level == "micro"
        assert args.allow_skip_slices is True
        assert args.min_transition_score == pytest.approx(0.25)
        assert args.min_support_count == 2
        assert args.matching_method == '{"tie_policy":"keep_all"}'

    def test_run_evolution_writes_valid_artifact_from_csv(self, parser, tmp_path):
        root = tmp_path / "result"
        root.mkdir()
        pd.DataFrame({"uid": ["D0"]}).to_parquet(root / "abstracts.parquet", index=False)

        slices_path = tmp_path / "slices.csv"
        states_path = tmp_path / "states.csv"
        transitions_path = tmp_path / "transitions.csv"
        pd.DataFrame(
            [
                {"slice_id": "year:2020", "slice_index": 0, "start_year": 2020, "end_year": 2020},
                {"slice_id": "year:2021", "slice_index": 1, "start_year": 2021, "end_year": 2021},
            ]
        ).to_csv(slices_path, index=False)
        pd.DataFrame(
            [
                {"slice_id": "year:2020", "cluster_id": "A", "doc_count": 3, "top_terms": '["alpha"]'},
                {"slice_id": "year:2021", "cluster_id": "A", "doc_count": 3, "top_terms": '["alpha"]'},
            ]
        ).to_csv(states_path, index=False)
        pd.DataFrame(
            [
                {
                    "source_state_id": "year:2020_cluster:A",
                    "target_state_id": "year:2021_cluster:A",
                    "score": 0.9,
                    "support_count": 3,
                }
            ]
        ).to_csv(transitions_path, index=False)

        args = parser.parse_args([
            "evolution",
            str(root),
            str(slices_path),
            str(states_path),
            str(transitions_path),
            "--evolution-id",
            "csv_evolution",
            "--metric",
            "term_overlap",
        ])
        _run_evolution(args)

        manifest_path = root / "evolution" / "evolution_manifest.json"
        validation = validate_evolution_artifact(manifest_path).to_dict()
        assert validation["status"] == "passed"
        assert validation["counts"]["slices"] == 2
        assert validation["counts"]["states"] == 2
        assert validation["counts"]["transitions"] == 1
        assert validation["event_counts"]["continuation"] == 1


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
