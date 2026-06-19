"""Tests for sciscape CLI parser construction and argument parsing."""

import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from sciscape.artifacts import (
    build_report_data_contract,
    validate_evolution_artifact,
    validate_export_manifest,
    validate_matrix_artifact,
    validate_narrative_artifact,
    write_cluster_review_packet_artifact,
    write_cooccurrence_artifacts,
    write_narrative_evidence_artifacts,
)
from sciscape.cli import (
    _build_parser,
    _run_bundle,
    _run_evolution,
    _run_evolution_evidence,
    _run_evolution_from_membership,
    _run_evolution_from_slice_membership,
    _run_evolution_from_slice_reclustering,
    _run_matrix,
    _run_narrative,
    _run_query,
    _run_visualize,
)


@pytest.fixture
def parser():
    return _build_parser()


def _write_cli_matrix_result_root(root: Path) -> Path:
    landscape = root / "landscape"
    landscape.mkdir(parents=True)
    pd.DataFrame(
        [
            {"source": "perovskite solar cells", "target": "interface passivation", "weight": 2.0, "count": 3},
            {"source": "graph neural networks", "target": "traffic forecasting", "weight": 1.0, "count": 1},
        ]
    ).to_parquet(landscape / "term_cooccurrence.parquet", index=False)
    pd.DataFrame(
        [
            {"cluster_id": 0, "term": "perovskite solar cells", "score": 0.9},
            {"cluster_id": 0, "term": "interface passivation", "score": 0.8},
            {"cluster_id": 1, "term": "graph neural networks", "score": 0.95},
            {"cluster_id": 1, "term": "traffic forecasting", "score": 0.75},
        ]
    ).to_parquet(landscape / "keywords.parquet", index=False)
    return root


def _write_cli_narrative_result_root(root: Path) -> Path:
    landscape = root / "landscape"
    report = landscape / "report"
    report.mkdir(parents=True)
    pd.DataFrame(
        {
            "uid": ["D0", "D1"],
            "title": ["Perovskite passivation", "Stable perovskite devices"],
            "abstract": [
                "Interface passivation improves perovskite stability.",
                "Passivation layers improve device durability.",
            ],
            "pubyear": [2021, 2022],
        }
    ).to_parquet(root / "abstracts.parquet", index=False)
    pd.DataFrame({"uid1": ["D0"], "uid2": ["D1"], "rel_sum2": [2.0]}).to_parquet(root / "edges.parquet", index=False)
    pd.DataFrame({"uid": ["D0", "D1"], "cluster": [0, 0]}).to_parquet(landscape / "membership.parquet", index=False)
    pd.DataFrame(
        {
            "cluster_id": [0, 0],
            "term": ["perovskite solar cells", "interface passivation"],
            "score": [0.9, 0.8],
            "frequency": [2, 1],
        }
    ).to_parquet(landscape / "keywords.parquet", index=False)
    report_data = {
        "0": {
            "label": "perovskite solar cells",
            "keywords": [{"term": "perovskite solar cells"}, {"term": "interface passivation"}],
            "network_edges": [{"source": "perovskite solar cells", "target": "interface passivation", "weight": 1}],
            "cooccurrence_table": [{"source": "perovskite solar cells", "target": "interface passivation", "count": 1}],
        },
    }
    report_data["_sciscape"] = build_report_data_contract(report_data)
    (report / "data.json").write_text(json.dumps(report_data), encoding="utf-8")
    write_cooccurrence_artifacts(root)
    write_cluster_review_packet_artifact(root)
    write_narrative_evidence_artifacts(root)
    return root


# ---------------------------------------------------------------------------
# Parser construction
# ---------------------------------------------------------------------------

class TestBuildParser:
    def test_returns_parser(self, parser):
        assert parser.prog == "sciscape"

    @pytest.mark.parametrize(
        "cmd",
        [
            "query",
            "cluster",
            "keywords",
            "convert",
            "landscape",
            "visualize",
            "viewer",
            "evolution-evidence",
            "evolution-from-membership",
            "evolution-from-slice-membership",
            "evolution-from-slice-reclustering",
            "evolution",
            "matrix",
            "bundle",
            "narrative",
            "gui",
        ],
    )
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
        if cmd == "evolution-evidence":
            return ["evolution-evidence", "records.parquet", "membership.parquet"]
        if cmd == "evolution-from-membership":
            return ["evolution-from-membership", "result", "records.parquet", "membership.parquet"]
        if cmd == "evolution-from-slice-membership":
            return ["evolution-from-slice-membership", "result", "slice_membership.parquet"]
        if cmd == "evolution-from-slice-reclustering":
            return ["evolution-from-slice-reclustering", "result", "records.parquet", "edges.parquet"]
        if cmd == "matrix":
            return ["matrix", "wrap-term-cooccurrence", "result"]
        if cmd == "bundle":
            return ["bundle", "vosviewer", "result"]
        if cmd == "narrative":
            return [
                "narrative",
                "render-prompts",
                "result",
                "--prompt-ref",
                "prompt:test",
            ]
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
        assert args.api_attempt_budget is None
        assert args.retry_wait_budget_seconds is None
        assert args.interruptible_requests is False
        assert args.request_poll_interval == 0.25

    def test_openalex_retry_explicit(self, parser):
        args = parser.parse_args([
            "query",
            "graph neural networks",
            "--request-timeout", "12",
            "--max-retries", "5",
            "--backoff-base", "0.25",
            "--backoff-max", "8",
            "--api-attempt-budget", "11",
            "--retry-wait-budget-seconds", "6",
            "--interruptible-requests",
            "--request-poll-interval", "0.1",
        ])
        assert args.request_timeout == 12.0
        assert args.max_retries == 5
        assert args.backoff_base == 0.25
        assert args.backoff_max == 8.0
        assert args.api_attempt_budget == 11
        assert args.retry_wait_budget_seconds == 6.0
        assert args.interruptible_requests is True
        assert args.request_poll_interval == 0.1

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
            "--api-attempt-budget", "11",
            "--retry-wait-budget-seconds", "6",
            "--interruptible-requests",
            "--request-poll-interval", "0.1",
            "-o", str(tmp_path / "out"),
        ])

        _run_query(args)

        config = captured["config"]
        assert config.request_timeout == 12.0
        assert config.max_retries == 5
        assert config.backoff_base == 0.25
        assert config.backoff_max == 8.0
        assert config.api_attempt_budget == 11
        assert config.retry_wait_budget_seconds == 6.0
        assert config.interruptible_requests is True
        assert config.request_poll_interval == 0.1
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
        assert args.cluster_sharded_shard_ids is None
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
            "--cluster-sharded-shard-ids",
            "3,1,3",
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
        assert args.cluster_sharded_shard_ids == (1, 3)
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
# Matrix subcommand
# ---------------------------------------------------------------------------

class TestMatrixArgs:
    def test_basic_parse(self, parser):
        args = parser.parse_args(["matrix", "wrap-term-cooccurrence", "result"])
        assert args.command == "matrix"
        assert args.matrix_command == "wrap-term-cooccurrence"
        assert args.result_root == Path("result")

    def test_defaults(self, parser):
        args = parser.parse_args(["matrix", "wrap-term-cooccurrence", "result"])
        assert args.matrix_id == "term_cooccurrence_default"
        assert args.json is False

    def test_explicit_options(self, parser):
        args = parser.parse_args([
            "matrix",
            "wrap-term-cooccurrence",
            "result",
            "--matrix-id",
            "custom_terms",
            "--json",
        ])
        assert args.matrix_id == "custom_terms"
        assert args.json is True

    def test_export_parse(self, parser):
        args = parser.parse_args([
            "matrix",
            "export",
            "result",
            "--matrix-id",
            "term_matrix",
            "--format",
            "vosviewer-network",
            "-o",
            "result/exports/term_matrix_vosviewer",
            "--json",
        ])
        assert args.command == "matrix"
        assert args.matrix_command == "export"
        assert args.matrix == Path("result")
        assert args.matrix_id == "term_matrix"
        assert args.export_format == "vosviewer-network"
        assert args.output_dir == Path("result/exports/term_matrix_vosviewer")
        assert args.json is True

    def test_run_matrix_wraps_term_cooccurrence(self, parser, tmp_path, capsys):
        root = _write_cli_matrix_result_root(tmp_path / "result")
        args = parser.parse_args(["matrix", "wrap-term-cooccurrence", str(root)])

        _run_matrix(args)

        out = capsys.readouterr().out
        assert "Matrix artifact saved" in out
        assert "status=passed, nnz=2, rows=4, columns=4" in out
        matrix_dir = root / "matrices" / "term_cooccurrence_default"
        validation = validate_matrix_artifact(matrix_dir).to_dict()
        assert validation["status"] == "passed"
        values = pd.read_parquet(matrix_dir / "matrix_values.parquet")
        assert len(values) == 2
        assert set(values["relation"]) == {"term_cooccurrence"}

    def test_run_matrix_json_output(self, parser, tmp_path, capsys):
        root = _write_cli_matrix_result_root(tmp_path / "result")
        args = parser.parse_args([
            "matrix",
            "wrap-term-cooccurrence",
            str(root),
            "--matrix-id",
            "custom_terms",
            "--json",
        ])

        _run_matrix(args)

        payload = json.loads(capsys.readouterr().out)
        assert payload["matrix_id"] == "custom_terms"
        assert payload["qa"]["status"] == "passed"
        assert payload["manifest_path"].endswith("matrices/custom_terms/matrix_manifest.json")

    def test_run_matrix_export_writes_manifest_backed_export(self, parser, tmp_path, capsys):
        root = _write_cli_matrix_result_root(tmp_path / "result")
        _run_matrix(parser.parse_args(["matrix", "wrap-term-cooccurrence", str(root)]))
        capsys.readouterr()
        args = parser.parse_args(["matrix", "export", str(root), "--format", "json-summary"])

        _run_matrix(args)

        out = capsys.readouterr().out
        assert "Matrix export saved" in out
        manifest_path = root / "exports" / "matrix_term_cooccurrence_default_json_summary" / "export_manifest.json"
        validation = validate_export_manifest(manifest_path).to_dict()
        assert validation["status"] == "passed"
        assert validation["export_kind"] == "matrix_json_summary"

    def test_run_matrix_export_writes_vosviewer_network(self, parser, tmp_path, capsys):
        root = _write_cli_matrix_result_root(tmp_path / "result")
        _run_matrix(parser.parse_args(["matrix", "wrap-term-cooccurrence", str(root)]))
        capsys.readouterr()
        args = parser.parse_args(["matrix", "export", str(root), "--format", "vosviewer-network"])

        _run_matrix(args)

        out = capsys.readouterr().out
        assert "Matrix export saved" in out
        manifest_path = (
            root
            / "exports"
            / "matrix_term_cooccurrence_default_vosviewer_network"
            / "export_manifest.json"
        )
        validation = validate_export_manifest(manifest_path).to_dict()
        assert validation["status"] == "passed"
        assert validation["export_family"] == "vosviewer"
        assert validation["export_kind"] == "matrix_vosviewer_network"
        network_path = (
            root
            / "exports"
            / "matrix_term_cooccurrence_default_vosviewer_network"
            / "vosviewer_matrix_network.txt"
        )
        assert network_path.exists()


# ---------------------------------------------------------------------------
# Bundle subcommand
# ---------------------------------------------------------------------------

class TestBundleArgs:
    def test_vosviewer_parse(self, parser):
        args = parser.parse_args([
            "bundle",
            "vosviewer",
            "result",
            "--ensure-term-exports",
            "--json",
        ])
        assert args.command == "bundle"
        assert args.bundle_command == "vosviewer"
        assert args.result_root == Path("result")
        assert args.ensure_term_exports is True
        assert args.json is True

    def test_run_bundle_vosviewer_ensures_term_exports(self, parser, tmp_path, capsys):
        root = _write_cli_matrix_result_root(tmp_path / "result")
        args = parser.parse_args(["bundle", "vosviewer", str(root), "--ensure-term-exports"])

        _run_bundle(args)

        out = capsys.readouterr().out
        assert "VOSviewer bundle saved" in out
        bundle_path = root / "exports" / "vosviewer_bundle" / "vosviewer_bundle.zip"
        manifest_path = root / "exports" / "vosviewer_bundle" / "export_manifest.json"
        validation = validate_export_manifest(manifest_path).to_dict()
        assert validation["status"] == "passed"
        assert validation["export_kind"] == "vosviewer_bundle"
        with zipfile.ZipFile(bundle_path) as archive:
            names = set(archive.namelist())
        assert {
            "vosviewer/vosviewer_term_map.txt",
            "vosviewer/vosviewer_term_network.txt",
            "exports/matrix_term_cooccurrence_default_vosviewer_network/vosviewer_matrix_map.txt",
            "exports/matrix_term_cooccurrence_default_vosviewer_network/vosviewer_matrix_network.txt",
            "exports/vosviewer_term_cooccurrence/export_manifest.json",
            "exports/matrix_term_cooccurrence_default_vosviewer_network/export_manifest.json",
            "vosviewer_bundle_inventory.json",
        }.issubset(names)

    def test_run_bundle_vosviewer_json_output(self, parser, tmp_path, capsys):
        root = _write_cli_matrix_result_root(tmp_path / "result")
        args = parser.parse_args(["bundle", "vosviewer", str(root), "--ensure-term-exports", "--json"])

        _run_bundle(args)

        payload = json.loads(capsys.readouterr().out)
        assert payload["bundle_path"].endswith("exports/vosviewer_bundle/vosviewer_bundle.zip")
        assert payload["inventory_path"].endswith("exports/vosviewer_bundle/vosviewer_bundle_inventory.json")
        assert payload["manifest_path"].endswith("exports/vosviewer_bundle/export_manifest.json")


# ---------------------------------------------------------------------------
# Narrative subcommand
# ---------------------------------------------------------------------------

class TestNarrativeArgs:
    def test_render_prompts_parse(self, parser):
        args = parser.parse_args([
            "narrative",
            "render-prompts",
            "result",
            "--prompt-batch-id",
            "batch-001",
            "--prompt-ref",
            "prompt:narrative:v1",
            "--prompt-version",
            "v1",
            "--include-review-state",
            "not_required",
            "--max-claims",
            "12",
            "--max-evidence-refs",
            "4",
            "--json",
        ])

        assert args.command == "narrative"
        assert args.narrative_command == "render-prompts"
        assert args.result_root == Path("result")
        assert args.prompt_batch_id == "batch-001"
        assert args.prompt_ref == "prompt:narrative:v1"
        assert args.prompt_version == "v1"
        assert args.include_review_state == ["not_required"]
        assert args.max_claims == 12
        assert args.max_evidence_refs == 4
        assert args.json is True

    def test_apply_generated_parse(self, parser):
        args = parser.parse_args([
            "narrative",
            "apply-generated",
            "result",
            "updates.json",
            "--provider",
            "test-provider",
            "--model",
            "test-model",
            "--model-run-id",
            "run-001",
            "--prompt-ref",
            "prompt:narrative:v1",
            "--prompt-digest",
            "sha256:test",
            "--reset-review-state",
            "needs_revision",
            "--json",
        ])

        assert args.command == "narrative"
        assert args.narrative_command == "apply-generated"
        assert args.result_root == Path("result")
        assert args.updates_file == Path("updates.json")
        assert args.provider == "test-provider"
        assert args.model == "test-model"
        assert args.model_run_id == "run-001"
        assert args.prompt_ref == "prompt:narrative:v1"
        assert args.prompt_digest == "sha256:test"
        assert args.reset_review_state == "needs_revision"
        assert args.json is True

    def test_run_prompts_parse(self, parser):
        args = parser.parse_args([
            "narrative",
            "run-prompts",
            "result",
            "--provider",
            "echo",
            "--model",
            "echo-model",
            "--model-run-id",
            "run-001",
            "--max-jobs",
            "3",
            "--apply",
            "--reset-review-state",
            "needs_revision",
            "--json",
        ])

        assert args.command == "narrative"
        assert args.narrative_command == "run-prompts"
        assert args.prompt_batch == Path("result")
        assert args.provider == "echo"
        assert args.model == "echo-model"
        assert args.model_run_id == "run-001"
        assert args.max_jobs == 3
        assert args.apply is True
        assert args.reset_review_state == "needs_revision"
        assert args.json is True

    def test_run_narrative_render_prompts_json(self, parser, tmp_path, capsys):
        root = _write_cli_narrative_result_root(tmp_path / "result")
        args = parser.parse_args([
            "narrative",
            "render-prompts",
            str(root),
            "--prompt-batch-id",
            "cli_prompt_batch",
            "--prompt-ref",
            "prompt:narrative:v1",
            "--max-claims",
            "1",
            "--json",
        ])

        _run_narrative(args)

        payload = json.loads(capsys.readouterr().out)
        assert payload["available"] is True
        assert payload["jobs"] == 1
        jobs_path = root / payload["jobs_path"]
        jobs = [json.loads(line) for line in jobs_path.read_text(encoding="utf-8").splitlines()]
        assert len(jobs) == 1
        assert jobs[0]["prompt_batch_id"] == "cli_prompt_batch"
        assert jobs[0]["prompt_ref"] == "prompt:narrative:v1"
        assert jobs[0]["claim_id"]

    def test_run_narrative_run_prompts_json(self, parser, tmp_path, capsys):
        root = _write_cli_narrative_result_root(tmp_path / "result")
        render_args = parser.parse_args([
            "narrative",
            "render-prompts",
            str(root),
            "--max-claims",
            "1",
        ])
        _run_narrative(render_args)
        capsys.readouterr()
        args = parser.parse_args([
            "narrative",
            "run-prompts",
            str(root),
            "--provider",
            "echo",
            "--model",
            "echo-model",
            "--model-run-id",
            "run-001",
            "--max-jobs",
            "1",
            "--apply",
            "--json",
        ])

        _run_narrative(args)

        payload = json.loads(capsys.readouterr().out)
        assert payload["available"] is True
        assert payload["generated_count"] == 1
        assert payload["failed_count"] == 0
        assert payload["applied"] is True
        assert (root / payload["updates_path"]).exists()
        assert (root / payload["run_manifest_path"]).exists()

    def test_run_narrative_apply_generated_json(self, parser, tmp_path, capsys):
        root = _write_cli_narrative_result_root(tmp_path / "result")
        claims = pd.read_parquet(root / "narrative" / "claims.parquet")
        claim_id = str(claims.iloc[0]["claim_id"])
        updates_path = tmp_path / "updates.json"
        updates_path.write_text(
            json.dumps(
                {
                    "claim_updates": [
                        {
                            "claim_id": claim_id,
                            "claim_text": "Generated claim text grounded in existing evidence.",
                        }
                    ],
                    "parameters": {"temperature": 0.0},
                }
            ),
            encoding="utf-8",
        )
        args = parser.parse_args([
            "narrative",
            "apply-generated",
            str(root),
            str(updates_path),
            "--provider",
            "test-provider",
            "--model",
            "test-model",
            "--model-run-id",
            "run-001",
            "--prompt-ref",
            "prompt:narrative:v1",
            "--json",
        ])

        _run_narrative(args)

        payload = json.loads(capsys.readouterr().out)
        assert payload["available"] is True
        assert payload["applied_count"] == 1
        updated_claims = pd.read_parquet(root / "narrative" / "claims.parquet")
        updated = updated_claims[updated_claims["claim_id"].map(str) == claim_id].iloc[0]
        assert updated["text_origin"] == "model_generated"
        assert updated["review_state"] == "not_reviewed"
        validation = validate_narrative_artifact(root).to_dict()
        assert validation["status"] != "blocked"
        assert validation["claim_counts"]["model_generated"] == 1

    def test_run_narrative_apply_generated_jsonl(self, parser, tmp_path, capsys):
        root = _write_cli_narrative_result_root(tmp_path / "result")
        claim_id = str(pd.read_parquet(root / "narrative" / "claims.parquet").iloc[0]["claim_id"])
        updates_path = tmp_path / "updates.jsonl"
        updates_path.write_text(
            json.dumps({"claim_id": claim_id, "claim_text": "JSONL generated claim text."}) + "\n",
            encoding="utf-8",
        )
        args = parser.parse_args([
            "narrative",
            "apply-generated",
            str(root),
            str(updates_path),
            "--provider",
            "test-provider",
            "--model",
            "test-model",
            "--model-run-id",
            "run-002",
            "--prompt-ref",
            "prompt:narrative:v1",
        ])

        _run_narrative(args)

        out = capsys.readouterr().out
        assert "Narrative generation updates applied: 1 claims" in out
        assert "status=" in out


# ---------------------------------------------------------------------------
# Evolution evidence subcommand
# ---------------------------------------------------------------------------

class TestEvolutionEvidenceArgs:
    def test_basic_parse(self, parser):
        args = parser.parse_args([
            "evolution-evidence",
            "records.parquet",
            "membership.parquet",
        ])
        assert args.command == "evolution-evidence"
        assert args.records_table == Path("records.parquet")
        assert args.membership_table == Path("membership.parquet")
        assert args.output_dir == Path("evolution_evidence")
        assert args.output_format == "parquet"

    def test_explicit_options(self, parser):
        args = parser.parse_args([
            "evolution-evidence",
            "records.csv",
            "membership.csv",
            "--keywords-table",
            "keywords.csv",
            "--evolution-id",
            "rolling_terms",
            "--cluster-column",
            "cluster_micro",
            "--uid-column",
            "work_id",
            "--membership-uid-column",
            "paper_id",
            "--representative-work-limit",
            "5",
            "--periodization",
            '{"window_years":2,"step_years":1}',
            "--output-format",
            "csv",
            "-o",
            "evidence",
            "--json",
        ])
        assert args.keywords_table == Path("keywords.csv")
        assert args.evolution_id == "rolling_terms"
        assert args.cluster_column == "cluster_micro"
        assert args.uid_column == "work_id"
        assert args.membership_uid_column == "paper_id"
        assert args.representative_work_limit == 5
        assert args.periodization == '{"window_years":2,"step_years":1}'
        assert args.output_format == "csv"
        assert args.output_dir == Path("evidence")
        assert args.json is True

    def test_run_evolution_evidence_writes_tables_from_csv(self, parser, tmp_path):
        records_path = tmp_path / "records.csv"
        membership_path = tmp_path / "membership.csv"
        keywords_path = tmp_path / "keywords.csv"
        output_dir = tmp_path / "evidence"
        pd.DataFrame(
            {
                "uid": ["D0", "D1", "D2", "D3"],
                "pubyear": [2020, 2021, 2021, 2022],
            }
        ).to_csv(records_path, index=False)
        pd.DataFrame(
            {
                "uid": ["D0", "D1", "D2", "D3"],
                "cluster_micro": ["A", "A", "B", "A"],
            }
        ).to_csv(membership_path, index=False)
        pd.DataFrame({"cluster_id": ["A", "B"], "term": ["alpha", "beta"]}).to_csv(keywords_path, index=False)

        args = parser.parse_args([
            "evolution-evidence",
            str(records_path),
            str(membership_path),
            "--keywords-table",
            str(keywords_path),
            "--cluster-column",
            "cluster_micro",
            "--periodization",
            '{"window_years":2}',
            "-o",
            str(output_dir),
        ])
        _run_evolution_evidence(args)

        slices = pd.read_parquet(output_dir / "time_slices.parquet")
        states = pd.read_parquet(output_dir / "state_evidence.parquet")
        state_membership = pd.read_parquet(output_dir / "state_membership.parquet")
        manifest = json.loads((output_dir / "evolution_evidence_manifest.json").read_text(encoding="utf-8"))
        assert slices["slice_id"].tolist() == ["year:2020-2021", "year:2021-2022"]
        assert set(states["cluster_key"]) == {"micro:A", "micro:B"}
        assert set(state_membership["schema_version"]) == {"sciscape_evolution_state_membership_v1"}
        assert manifest["schema_version"] == "sciscape_evolution_evidence_pack_v1"
        assert manifest["counts"]["state_membership_rows"] == len(state_membership)


# ---------------------------------------------------------------------------
# Evolution from membership subcommand
# ---------------------------------------------------------------------------

class TestEvolutionFromMembershipArgs:
    def test_basic_parse(self, parser):
        args = parser.parse_args([
            "evolution-from-membership",
            "result",
            "records.parquet",
            "membership.parquet",
        ])
        assert args.command == "evolution-from-membership"
        assert args.result_root == Path("result")
        assert args.records_table == Path("records.parquet")
        assert args.membership_table == Path("membership.parquet")
        assert args.metric == "overlap_min"
        assert args.output_dir is None

    def test_explicit_options(self, parser):
        args = parser.parse_args([
            "evolution-from-membership",
            "result",
            "records.csv",
            "membership.csv",
            "--keywords-table",
            "keywords.csv",
            "--evolution-id",
            "rolling_terms",
            "--metric",
            "jaccard_doc_overlap",
            "--title",
            "Rolling Terms",
            "--output-dir",
            "custom_evolution",
            "--temporal-manifest",
            "temporal/temporal_manifest.json",
            "--cluster-column",
            "cluster_micro",
            "--uid-column",
            "work_id",
            "--membership-uid-column",
            "paper_id",
            "--representative-work-limit",
            "5",
            "--min-transition-score",
            "0.25",
            "--min-support-count",
            "2",
            "--matching-method",
            '{"tie_policy":"keep_all"}',
            "--event-rules",
            '{"ambiguous_score_margin":0.1}',
            "--periodization",
            '{"window_years":2}',
            "--entity-scope",
            '{"cluster_level":"micro"}',
            "--allow-incomplete-state-membership",
            "--json",
        ])
        assert args.keywords_table == Path("keywords.csv")
        assert args.evolution_id == "rolling_terms"
        assert args.metric == "jaccard_doc_overlap"
        assert args.title == "Rolling Terms"
        assert args.output_dir == Path("custom_evolution")
        assert args.temporal_manifest == Path("temporal/temporal_manifest.json")
        assert args.cluster_column == "cluster_micro"
        assert args.uid_column == "work_id"
        assert args.membership_uid_column == "paper_id"
        assert args.representative_work_limit == 5
        assert args.min_transition_score == pytest.approx(0.25)
        assert args.min_support_count == 2
        assert args.allow_incomplete_state_membership is True
        assert args.json is True

    def test_run_evolution_from_membership_writes_valid_artifact(self, parser, tmp_path):
        root = tmp_path / "result"
        root.mkdir()
        records_path = tmp_path / "records.csv"
        membership_path = tmp_path / "membership.csv"
        keywords_path = tmp_path / "keywords.csv"
        pd.DataFrame(
            {
                "uid": ["A20", "A21", "A22", "B21", "B22", "C20"],
                "pubyear": [2020, 2021, 2022, 2021, 2022, 2020],
            }
        ).to_csv(records_path, index=False)
        pd.DataFrame(
            {
                "uid": ["A20", "A21", "A22", "B21", "B22", "C20"],
                "cluster_micro": ["A", "A", "A", "B", "B", "C"],
            }
        ).to_csv(membership_path, index=False)
        pd.DataFrame({"cluster_id": ["A", "B", "C"], "term": ["alpha", "beta", "carbon"]}).to_csv(keywords_path, index=False)

        args = parser.parse_args([
            "evolution-from-membership",
            str(root),
            str(records_path),
            str(membership_path),
            "--keywords-table",
            str(keywords_path),
            "--cluster-column",
            "cluster_micro",
            "--periodization",
            '{"window_years":2}',
            "--evolution-id",
            "membership_cli_evolution",
        ])
        _run_evolution_from_membership(args)

        validation = validate_evolution_artifact(root / "evolution" / "evolution_manifest.json").to_dict()
        assert validation["status"] == "passed"
        assert validation["counts"]["slices"] == 2
        assert validation["counts"]["transitions"] == 2
        assert validation["counts"]["state_membership_rows"] == 8
        assert validation["event_counts"]["continuation"] == 2


# ---------------------------------------------------------------------------
# Evolution from slice-local membership subcommand
# ---------------------------------------------------------------------------

class TestEvolutionFromSliceMembershipArgs:
    def test_basic_parse(self, parser):
        args = parser.parse_args([
            "evolution-from-slice-membership",
            "result",
            "slice_membership.parquet",
        ])
        assert args.command == "evolution-from-slice-membership"
        assert args.result_root == Path("result")
        assert args.slice_membership_table == Path("slice_membership.parquet")
        assert args.metric == "overlap_min"
        assert args.slice_id_column == "slice_id"
        assert args.default_level == "cluster"

    def test_explicit_options(self, parser):
        args = parser.parse_args([
            "evolution-from-slice-membership",
            "result",
            "slice_membership.csv",
            "--slices-table",
            "slices.csv",
            "--keywords-table",
            "keywords.csv",
            "--evolution-id",
            "slice_terms",
            "--metric",
            "jaccard_doc_overlap",
            "--title",
            "Slice Terms",
            "--output-dir",
            "custom_evolution",
            "--temporal-manifest",
            "temporal/temporal_manifest.json",
            "--cluster-column",
            "cluster_id",
            "--uid-column",
            "work_id",
            "--slice-id-column",
            "period_id",
            "--default-level",
            "micro",
            "--representative-work-limit",
            "5",
            "--min-transition-score",
            "0.25",
            "--min-support-count",
            "2",
            "--matching-method",
            '{"tie_policy":"keep_all"}',
            "--event-rules",
            '{"ambiguous_score_margin":0.1}',
            "--entity-scope",
            '{"cluster_level":"micro"}',
            "--allow-incomplete-state-membership",
            "--json",
        ])
        assert args.slices_table == Path("slices.csv")
        assert args.keywords_table == Path("keywords.csv")
        assert args.evolution_id == "slice_terms"
        assert args.metric == "jaccard_doc_overlap"
        assert args.title == "Slice Terms"
        assert args.output_dir == Path("custom_evolution")
        assert args.temporal_manifest == Path("temporal/temporal_manifest.json")
        assert args.cluster_column == "cluster_id"
        assert args.uid_column == "work_id"
        assert args.slice_id_column == "period_id"
        assert args.default_level == "micro"
        assert args.representative_work_limit == 5
        assert args.min_transition_score == pytest.approx(0.25)
        assert args.min_support_count == 2
        assert args.allow_incomplete_state_membership is True
        assert args.json is True

    def test_run_evolution_from_slice_membership_writes_valid_artifact(self, parser, tmp_path):
        root = tmp_path / "result"
        root.mkdir()
        slice_membership_path = tmp_path / "slice_membership.csv"
        pd.DataFrame(
            [
                *[
                    {"slice_id": "year:2020", "slice_index": 0, "start_year": 2020, "end_year": 2020, "uid": f"A{i}", "cluster_id": "A"}
                    for i in range(4)
                ],
                *[
                    {"slice_id": "year:2021", "slice_index": 1, "start_year": 2021, "end_year": 2021, "uid": f"A{i}", "cluster_id": "B1"}
                    for i in range(2)
                ],
                *[
                    {"slice_id": "year:2021", "slice_index": 1, "start_year": 2021, "end_year": 2021, "uid": f"A{i}", "cluster_id": "B2"}
                    for i in range(2, 4)
                ],
                *[
                    {"slice_id": "year:2020", "slice_index": 0, "start_year": 2020, "end_year": 2020, "uid": f"C{i}", "cluster_id": "C"}
                    for i in range(3)
                ],
                *[
                    {"slice_id": "year:2021", "slice_index": 1, "start_year": 2021, "end_year": 2021, "uid": f"C{i}", "cluster_id": "C"}
                    for i in range(3)
                ],
            ]
        ).to_csv(slice_membership_path, index=False)

        args = parser.parse_args([
            "evolution-from-slice-membership",
            str(root),
            str(slice_membership_path),
            "--cluster-column",
            "cluster_id",
            "--default-level",
            "micro",
            "--min-transition-score",
            "0.5",
            "--min-support-count",
            "2",
            "--evolution-id",
            "slice_membership_cli_evolution",
        ])
        _run_evolution_from_slice_membership(args)

        validation = validate_evolution_artifact(root / "evolution" / "evolution_manifest.json").to_dict()
        assert validation["status"] == "passed"
        assert validation["counts"]["slices"] == 2
        assert validation["counts"]["transitions"] == 3
        assert validation["counts"]["state_membership_rows"] == 14
        assert validation["event_counts"]["split"] == 1


# ---------------------------------------------------------------------------
# Evolution from slice-local reclustering subcommand
# ---------------------------------------------------------------------------

class TestEvolutionFromSliceReclusteringArgs:
    def test_basic_parse(self, parser):
        args = parser.parse_args([
            "evolution-from-slice-reclustering",
            "result",
            "records.parquet",
            "edges.parquet",
        ])
        assert args.command == "evolution-from-slice-reclustering"
        assert args.result_root == Path("result")
        assert args.records_table == Path("records.parquet")
        assert args.edge_table == Path("edges.parquet")
        assert args.metric == "overlap_min"
        assert args.backend == "auto"
        assert args.resolution == pytest.approx(1.0)

    def test_explicit_options(self, parser):
        args = parser.parse_args([
            "evolution-from-slice-reclustering",
            "result",
            "records.csv",
            "edges.csv",
            "--keywords-table",
            "keywords.csv",
            "--evolution-id",
            "slice_recluster",
            "--metric",
            "jaccard_doc_overlap",
            "--title",
            "Slice Recluster",
            "--output-dir",
            "custom_evolution",
            "--temporal-manifest",
            "temporal/temporal_manifest.json",
            "--uid-column",
            "work_id",
            "--edge-source-column",
            "source",
            "--edge-target-column",
            "target",
            "--edge-weight-column",
            "weight",
            "--resolution",
            "0.01",
            "--objective",
            "cpm",
            "--backend",
            "igraph",
            "--seed",
            "7",
            "--n-iterations",
            "3",
            "--min-docs-per-slice",
            "2",
            "--slice-reclustering-workers",
            "2",
            "--slice-membership-output",
            "evolution_work/slice_membership.parquet",
            "--slice-membership-parts-dir",
            "evolution_work/slice_membership_parts",
            "--progress-path",
            "evolution_work/progress.json",
            "--representative-work-limit",
            "5",
            "--min-transition-score",
            "0.25",
            "--min-support-count",
            "2",
            "--matching-method",
            '{"tie_policy":"keep_all"}',
            "--event-rules",
            '{"ambiguous_score_margin":0.1}',
            "--periodization",
            '{"window_years":2}',
            "--entity-scope",
            '{"cluster_level":"micro"}',
            "--allow-incomplete-state-membership",
            "--json",
        ])
        assert args.keywords_table == Path("keywords.csv")
        assert args.evolution_id == "slice_recluster"
        assert args.metric == "jaccard_doc_overlap"
        assert args.title == "Slice Recluster"
        assert args.output_dir == Path("custom_evolution")
        assert args.temporal_manifest == Path("temporal/temporal_manifest.json")
        assert args.uid_column == "work_id"
        assert args.edge_source_column == "source"
        assert args.edge_target_column == "target"
        assert args.edge_weight_column == "weight"
        assert args.resolution == pytest.approx(0.01)
        assert args.backend == "igraph"
        assert args.seed == 7
        assert args.n_iterations == 3
        assert args.min_docs_per_slice == 2
        assert args.slice_reclustering_workers == 2
        assert args.slice_membership_output == Path("evolution_work/slice_membership.parquet")
        assert args.slice_membership_parts_dir == Path("evolution_work/slice_membership_parts")
        assert args.progress_path == Path("evolution_work/progress.json")
        assert args.representative_work_limit == 5
        assert args.min_transition_score == pytest.approx(0.25)
        assert args.min_support_count == 2
        assert args.allow_incomplete_state_membership is True
        assert args.json is True

    def test_run_evolution_from_slice_reclustering_writes_valid_artifact(self, parser, tmp_path):
        root = tmp_path / "result"
        root.mkdir()
        records_path = tmp_path / "records.csv"
        edges_path = tmp_path / "edges.csv"
        pd.DataFrame(
            {
                "uid": ["A20", "A21", "A22", "C20", "C21", "C22"],
                "pubyear": [2020, 2021, 2022, 2020, 2021, 2022],
            }
        ).to_csv(records_path, index=False)
        pd.DataFrame(
            {
                "uid1": ["A20", "A21", "C20", "C21"],
                "uid2": ["A21", "A22", "C21", "C22"],
                "rel_sum2": [2.0, 2.0, 2.0, 2.0],
            }
        ).to_csv(edges_path, index=False)
        slice_membership_output = root / "evolution_work" / "slice_membership.parquet"
        slice_membership_parts_dir = root / "evolution_work" / "slice_membership_parts"

        args = parser.parse_args([
            "evolution-from-slice-reclustering",
            str(root),
            str(records_path),
            str(edges_path),
            "--periodization",
            '{"window_years":2,"step_years":1}',
            "--resolution",
            "0.01",
            "--backend",
            "igraph",
            "--slice-reclustering-workers",
            "2",
            "--slice-membership-output",
            str(slice_membership_output),
            "--slice-membership-parts-dir",
            str(slice_membership_parts_dir),
            "--evolution-id",
            "slice_recluster_cli_evolution",
        ])
        _run_evolution_from_slice_reclustering(args)

        generated_membership = pd.read_parquet(slice_membership_output)
        assert len(generated_membership) == 8
        assert set(generated_membership["slice_id"]) == {"year:2020-2021", "year:2021-2022"}
        part_files = sorted(slice_membership_parts_dir.glob("*.parquet"))
        assert len(part_files) == 2
        assert sum(len(pd.read_parquet(path)) for path in part_files) == 8
        progress = json.loads((root / "evolution_work" / "slice_reclustering_progress.json").read_text(encoding="utf-8"))
        assert progress["status"] == "completed"
        assert progress["completed_slices"] == 2
        assert progress["membership_rows"] == 8
        assert progress["membership_part_count"] == 2
        assert progress["params"]["max_workers"] == 2
        validation = validate_evolution_artifact(root / "evolution" / "evolution_manifest.json").to_dict()
        assert validation["status"] == "passed"
        assert validation["counts"]["slices"] == 2
        assert validation["counts"]["states"] == 4
        assert validation["counts"]["transitions"] == 2
        assert validation["event_counts"]["continuation"] == 2


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
        assert args.derive_transitions == "explicit"

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
        assert args.state_membership_table is None
        assert args.state_membership_uid_column is None
        assert args.state_membership_state_id_column == "state_id"
        assert args.allow_incomplete_state_membership is False

    def test_document_overlap_parse_without_transition_table(self, parser):
        args = parser.parse_args([
            "evolution",
            "result",
            "slices.parquet",
            "states.parquet",
            "--derive-transitions",
            "document-overlap",
            "--state-membership-table",
            "state_membership.parquet",
        ])
        assert args.transition_evidence_table is None
        assert args.derive_transitions == "document-overlap"
        assert args.state_membership_table == Path("state_membership.parquet")

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
            "--state-membership-uid-column",
            "work_id",
            "--state-membership-state-id-column",
            "state",
            "--allow-incomplete-state-membership",
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
        assert args.state_membership_uid_column == "work_id"
        assert args.state_membership_state_id_column == "state"
        assert args.allow_incomplete_state_membership is True

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

    def test_run_evolution_derives_document_overlap_transitions_from_csv(self, parser, tmp_path):
        root = tmp_path / "result"
        root.mkdir()
        pd.DataFrame({"uid": ["D0"]}).to_parquet(root / "abstracts.parquet", index=False)

        slices_path = tmp_path / "slices.csv"
        states_path = tmp_path / "states.csv"
        membership_path = tmp_path / "state_membership.csv"
        pd.DataFrame(
            [
                {"slice_id": "year:2020", "slice_index": 0, "start_year": 2020, "end_year": 2020},
                {"slice_id": "year:2021", "slice_index": 1, "start_year": 2021, "end_year": 2021},
            ]
        ).to_csv(slices_path, index=False)
        pd.DataFrame(
            [
                {"slice_id": "year:2020", "cluster_id": "A", "doc_count": 4, "top_terms": '["alpha"]'},
                {"slice_id": "year:2021", "cluster_id": "B1", "doc_count": 2, "top_terms": '["beta"]'},
                {"slice_id": "year:2021", "cluster_id": "B2", "doc_count": 2, "top_terms": '["gamma"]'},
            ]
        ).to_csv(states_path, index=False)
        pd.DataFrame(
            [
                *[{"slice_id": "year:2020", "cluster_id": "A", "uid": f"A{i}"} for i in range(4)],
                *[{"slice_id": "year:2021", "cluster_id": "B1", "uid": f"A{i}"} for i in range(2)],
                *[{"slice_id": "year:2021", "cluster_id": "B2", "uid": f"A{i}"} for i in range(2, 4)],
            ]
        ).to_csv(membership_path, index=False)

        args = parser.parse_args([
            "evolution",
            str(root),
            str(slices_path),
            str(states_path),
            "--derive-transitions",
            "document-overlap",
            "--state-membership-table",
            str(membership_path),
            "--evolution-id",
            "overlap_cli_evolution",
            "--min-transition-score",
            "0.5",
            "--min-support-count",
            "2",
        ])
        _run_evolution(args)

        manifest_path = root / "evolution" / "evolution_manifest.json"
        validation = validate_evolution_artifact(manifest_path).to_dict()
        assert validation["status"] == "passed"
        assert validation["counts"]["states"] == 3
        assert validation["counts"]["transitions"] == 2
        assert validation["event_counts"]["split"] == 1


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
