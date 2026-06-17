import json
from pathlib import Path

import pandas as pd

from sciscape.artifacts import validate_keyword_rule_artifact
from sciscape.cli import _infer_keyword_rule_result_root
from sciscape.keyword_extraction import KeywordExtractionConfig, run_keyword_pipeline
from sciscape.keyword_extraction.cluster_sharded import (
    _keyword_rule_source_artifact,
    adaptive_candidate_cap,
    build_cluster_shard_manifest,
    run_cluster_sharded_preflight,
    score_candidate_shard,
)
from sciscape.keyword_extraction.extraction import _DataSource
from sciscape.keyword_extraction.pipeline import KeywordExtractionPipeline
from sciscape.keyword_extraction.rule_artifact import build_keyword_rule_artifact_inputs


def test_keyword_pipeline_smoke(tmp_path):
    abstracts = pd.DataFrame(
        {
            "uid": ["D1", "D2", "D3", "D4"],
            "title": [
                "Advances in quantum sensors",
                "Resilient grids with machine learning",
                "Battery lifecycle modelling",
                "Solar materials innovation",
            ],
            "abstract": [
                "Quantum sensing enables precise magnetic field measurements.",
                "Machine learning improves grid resilience and forecasting accuracy.",
                "Lifecycle models for batteries utilise physics informed learning.",
                "New perovskite solar materials deliver higher efficiency cells.",
            ],
            "pubyear": [2018, 2019, 2020, 2021],
        }
    )
    membership = pd.DataFrame(
        {
            "uid": ["D1", "D2", "D3", "D4"],
            "cluster": [0, 1, 1, 0],
        }
    )

    abstract_path = Path(tmp_path) / "abstracts.parquet"
    membership_path = Path(tmp_path) / "membership.parquet"
    result_root = Path(tmp_path) / "result"
    abstracts.to_parquet(abstract_path, index=False)
    membership.to_parquet(membership_path, index=False)

    cfg = KeywordExtractionConfig(
        abstract_path=abstract_path,
        membership_path=membership_path,
        cluster_level="cluster",
        include_title=True,
        title_weight=1.0,
        min_df_unigram=1,
        min_df_phrase=1,
        max_df_unigram=1.0,
        max_df_phrase=1.0,
        phrase_min_count_per_cluster=1,
        min_cluster_doc_coverage=1,
        min_cluster_doc_coverage_ratio=0.1,
        top_n_keywords=3,
        ngram_min=2,
        ngram_max=3,
        use_phrase_vectorizer=True,
        mmr_jaccard_lambda=0.3,
        mmr_pool_factor=3.0,
        w_llr=0.5,
        n_jobs=1,
        keyword_rule_result_root=result_root,
    )

    keywords = run_keyword_pipeline(cfg)
    assert not keywords.empty
    required = {"cluster_id", "term", "score", "frequency", "pub_year_series"}
    assert required.issubset(set(keywords.columns))
    rule_manifest = result_root / "rules" / "keyword_cleaning_default_v1" / "rule_set_manifest.json"
    assert rule_manifest.exists()
    rule_validation = validate_keyword_rule_artifact(rule_manifest).to_dict()
    assert rule_validation["status"] == "passed"
    assert rule_validation["counts"]["before_after_rows"] == len(keywords)


def test_keyword_rule_artifact_inputs_block_only_structural_artifacts():
    keywords = pd.DataFrame(
        {
            "cluster_id": [0, 0, 1],
            "term": ["class htmlview paragraph", "date", "graph neural networks"],
            "raw_term": ["class htmlview paragraph", "date", "graph neural networks"],
            "score": [0.9, 0.5, 1.2],
            "frequency": [10, 4, 12],
            "representative_rank": [1, 2, 1],
            "quality_flags": ["metadata_fragment", "short_form", ""],
            "keyword_label_tier": ["review_artifact", "review_short_form", "primary_phrase"],
        }
    )

    rules, applications, before_after = build_keyword_rule_artifact_inputs(keywords)

    rule_by_id = {row["rule_id"]: row for row in rules.to_dict("records")}
    assert rule_by_id["html_fragment_block"]["action"] == "block"
    assert rule_by_id["html_fragment_block"]["destructive"] is True
    assert rule_by_id["quality_review_short_form"]["action"] == "keep_with_flag"
    assert rule_by_id["quality_review_review_short_form"]["pattern"] == "review_short_form"
    blocked = before_after[before_after["raw_term"] == "class htmlview paragraph"].iloc[0]
    assert bool(blocked["blocked"]) is True
    reviewed = before_after[before_after["raw_term"] == "date"].iloc[0]
    assert bool(reviewed["blocked"]) is False
    assert reviewed["review_status"] == "needs_review"
    review_apps = applications[applications["rule_id"] == "quality_review_short_form"]
    assert set(review_apps["evidence_value"]) == {"short_form"}
    assert set(applications["decision"]) == {"applied", "blocked"}


def test_keyword_rule_source_artifact_uses_absolute_path_when_outside_root(tmp_path):
    result_root = Path(tmp_path) / "result"
    output_dir = Path(tmp_path) / "keyword_v2"
    result_root.mkdir()
    output_dir.mkdir()

    inside = _keyword_rule_source_artifact("keywords", result_root / "keywords.parquet", result_root)
    outside = _keyword_rule_source_artifact("keywords", output_dir / "keywords.parquet", result_root)

    assert inside == {"role": "keywords", "path": "keywords.parquet"}
    assert outside == {"role": "keywords", "path": str((output_dir / "keywords.parquet").resolve())}


def test_cli_infers_keyword_rule_result_root_from_landscape_output(tmp_path):
    result_root = Path(tmp_path) / "result"

    inferred = _infer_keyword_rule_result_root(result_root / "landscape" / "keywords.parquet", None)
    explicit = _infer_keyword_rule_result_root(Path("keywords.parquet"), result_root)
    missing = _infer_keyword_rule_result_root(Path("keywords.parquet"), None)

    assert inferred == result_root
    assert explicit == result_root
    assert missing is None


def test_cluster_task_progress_file(tmp_path):
    abstracts = pd.DataFrame(
        {
            "uid": ["D1", "D2"],
            "title": ["Quantum sensors", "Solar cells"],
            "abstract": ["Quantum sensing text.", "Solar cell text."],
            "pubyear": [2020, 2021],
        }
    )
    membership = pd.DataFrame({"uid": ["D1", "D2"], "cluster": [0, 1]})

    abstract_path = Path(tmp_path) / "abstracts.parquet"
    membership_path = Path(tmp_path) / "membership.parquet"
    progress_path = Path(tmp_path) / "keyword_progress.json"
    abstracts.to_parquet(abstract_path, index=False)
    membership.to_parquet(membership_path, index=False)

    cfg = KeywordExtractionConfig(
        abstract_path=abstract_path,
        membership_path=membership_path,
        cluster_level="cluster",
        parallel_backend="sequential",
        progress_path=progress_path,
        progress_interval_clusters=1,
    )
    pipeline = KeywordExtractionPipeline(cfg)

    assert pipeline._run_cluster_tasks("unit_stage", 3, lambda idx: idx + 10) == [10, 11, 12]
    payload = json.loads(progress_path.read_text())
    assert payload["stage"] == "unit_stage"
    assert payload["processed"] == 3
    assert payload["total"] == 3
    assert payload["parallel_backend"] == "sequential"


def test_keyword_pipeline_scoring_shards_match_unsharded(tmp_path):
    abstracts = pd.DataFrame(
        {
            "uid": ["D1", "D2", "D3", "D4", "D5", "D6"],
            "title": [
                "Quantum sensor devices",
                "Quantum magnetic sensing",
                "Solar cell materials",
                "Perovskite solar devices",
                "Graph neural networks",
                "Traffic graph forecasting",
            ],
            "abstract": [
                "Quantum sensing improves magnetic field measurement.",
                "Magnetic quantum sensors support precision metrology.",
                "Solar cell materials improve photovoltaic efficiency.",
                "Perovskite solar cells improve device efficiency.",
                "Graph neural networks learn molecular representations.",
                "Traffic forecasting uses graph neural network models.",
            ],
            "pubyear": [2018, 2019, 2020, 2021, 2022, 2023],
        }
    )
    membership = pd.DataFrame(
        {"uid": ["D1", "D2", "D3", "D4", "D5", "D6"], "cluster": [0, 0, 1, 1, 2, 2]}
    )
    abstract_path = Path(tmp_path) / "abstracts.parquet"
    membership_path = Path(tmp_path) / "membership.parquet"
    abstracts.to_parquet(abstract_path, index=False)
    membership.to_parquet(membership_path, index=False)

    common = dict(
        abstract_path=abstract_path,
        membership_path=membership_path,
        cluster_level="cluster",
        include_title=True,
        title_weight=1.0,
        min_df_unigram=1,
        min_df_phrase=1,
        phrase_min_count_per_cluster=1,
        top_n_keywords=4,
        scoring_pool_factor=1.0,
        ngram_min=2,
        ngram_max=3,
        use_phrase_vectorizer=True,
        normalization_enabled=False,
        quality_diagnostics_enabled=False,
        abbreviation_dictionary_enabled=False,
        n_jobs=1,
        parallel_backend="sequential",
    )
    unsharded = run_keyword_pipeline(KeywordExtractionConfig(**common))
    shard_dir = Path(tmp_path) / "scoring_shards"
    progress_path = Path(tmp_path) / "progress.json"
    sharded = run_keyword_pipeline(
        KeywordExtractionConfig(
            **common,
            scoring_shard_dir=shard_dir,
            scoring_shard_size_clusters=1,
            progress_path=progress_path,
            progress_interval_clusters=1,
        )
    )

    cols = ["cluster_id", "term", "score", "frequency", "doc_coverage"]
    left = unsharded[cols].sort_values(cols[:2]).reset_index(drop=True)
    right = sharded[cols].sort_values(cols[:2]).reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right, check_exact=False, atol=1e-12)

    manifest = json.loads((shard_dir / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["shard_count"] == 3
    assert len(list(shard_dir.glob("*.done.json"))) == 3
    progress = json.loads(progress_path.read_text())
    assert progress["stage"] == "complete"


def test_cluster_sharded_cap_policy_scales_with_doc_count(tmp_path):
    dummy_abstracts = Path(tmp_path) / "abstracts.parquet"
    dummy_membership = Path(tmp_path) / "membership.parquet"
    pd.DataFrame({"uid": ["D1"], "abstract": ["x"], "pubyear": [2020]}).to_parquet(dummy_abstracts, index=False)
    pd.DataFrame({"uid": ["D1"], "cluster": [0]}).to_parquet(dummy_membership, index=False)
    cfg = KeywordExtractionConfig(
        abstract_path=dummy_abstracts,
        membership_path=dummy_membership,
        cluster_level="cluster",
        candidate_pool_floor=256,
        candidate_pool_large=1024,
        candidate_pool_hard_max=1536,
    )

    assert adaptive_candidate_cap(250, cfg) == 256
    assert adaptive_candidate_cap(1_000, cfg) >= 512
    assert adaptive_candidate_cap(10_000, cfg) <= 1536
    assert adaptive_candidate_cap(10_000, cfg) > adaptive_candidate_cap(1_000, cfg)


def test_cluster_sharded_keyword_engine_writes_resume_artifacts(tmp_path):
    abstracts = pd.DataFrame(
        {
            "uid": ["D1", "D2", "D3", "D4", "D5", "D6"],
            "title": [
                "Quantum sensor devices",
                "Quantum magnetic sensing",
                "Solar cell materials",
                "Perovskite solar devices",
                "Graph neural networks",
                "Traffic graph forecasting",
            ],
            "abstract": [
                "Quantum sensing improves magnetic field measurement.",
                "Magnetic quantum sensors support precision metrology.",
                "Solar cell materials improve photovoltaic efficiency.",
                "Perovskite solar cells improve device efficiency.",
                "Graph neural networks learn molecular representations.",
                "Traffic forecasting uses graph neural network models.",
            ],
            "pubyear": [2018, 2019, 2020, 2021, 2022, 2023],
        }
    )
    membership = pd.DataFrame(
        {"uid": ["D1", "D2", "D3", "D4", "D5", "D6"], "cluster": [0, 0, 1, 1, 2, 2]}
    )
    abstract_path = Path(tmp_path) / "abstracts.parquet"
    membership_path = Path(tmp_path) / "membership.parquet"
    output_dir = Path(tmp_path) / "keyword_v2"
    abstracts.to_parquet(abstract_path, index=False)
    membership.to_parquet(membership_path, index=False)

    cfg = KeywordExtractionConfig(
        abstract_path=abstract_path,
        membership_path=membership_path,
        cluster_level="cluster",
        keyword_engine="cluster_sharded",
        cluster_sharded_output_dir=output_dir,
        include_title=True,
        ngram_min=2,
        ngram_max=3,
        top_n_keywords=4,
        candidate_pool_floor=8,
        candidate_pool_target=12,
        candidate_pool_large=16,
        candidate_pool_hard_max=24,
        target_docs_per_shard=3,
        max_clusters_per_shard=2,
        min_df_unigram=1,
        min_df_phrase=1,
        phrase_min_count_per_cluster=1,
        n_jobs=1,
        parallel_backend="sequential",
        candidate_mining_progress_interval_docs=1,
        candidate_mining_prune_interval_docs=1,
        candidate_mining_prune_multiplier=2,
        progress_path=Path(tmp_path) / "progress.json",
    )

    manifest = build_cluster_shard_manifest(cfg, _DataSource(cfg))
    assert manifest["total_clusters"] == 3
    assert len(manifest["shards"]) >= 2

    keywords = run_keyword_pipeline(cfg)
    assert not keywords.empty
    assert set(["cluster_id", "term", "score", "rank", "tier", "keyword_engine"]).issubset(keywords.columns)
    assert set(keywords["keyword_engine"]) == {"cluster_sharded"}
    assert keywords.groupby("cluster_id")["rank"].max().max() <= 4
    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "global" / "global_term_stats.parquet").exists()
    assert (output_dir / "keywords.parquet").exists()
    assert (output_dir / "keywords_flagged.parquet").exists()
    assert (output_dir / "qa" / "keyword_quality_residual_report.json").exists()
    assert (output_dir / "qa" / "keyword_quality_residual_report.md").exists()
    rule_manifest = output_dir / "rules" / "keyword_cleaning_default_v1" / "rule_set_manifest.json"
    assert rule_manifest.exists()
    rule_validation = validate_keyword_rule_artifact(rule_manifest).to_dict()
    assert rule_validation["status"] == "passed"
    assert rule_validation["counts"]["before_after_rows"] == len(keywords)
    run_summary = json.loads((output_dir / "run_summary.json").read_text())
    assert run_summary["keyword_rule_manifest_path"] == str(rule_manifest)
    candidate_done = list((output_dir / "candidates").glob("*.done.json"))
    final_done = list((output_dir / "final").glob("*.done.json"))
    assert candidate_done
    assert final_done
    candidate_done_payload = json.loads(candidate_done[0].read_text())
    assert candidate_done_payload["fingerprint"]
    assert candidate_done_payload["elapsed_sec"] >= 0
    assert candidate_done_payload["source_rows"] > 0
    assert "peak_rss_mb" in candidate_done_payload
    progress_files = list((output_dir / "candidates").glob("*.progress.json"))
    assert progress_files
    candidate_progress = json.loads(progress_files[0].read_text())
    assert candidate_progress["status"] == "complete"
    assert candidate_progress["rows_processed"] == candidate_progress["rows_total"]
    assert "terms_tracked" in candidate_progress
    final_done_payload = json.loads(final_done[0].read_text())
    assert final_done_payload["fingerprint"]
    assert final_done_payload["flagged_path"]
    assert Path(final_done_payload["flagged_path"]).exists()
    rerun = run_keyword_pipeline(cfg)
    pd.testing.assert_frame_equal(
        keywords.sort_values(["cluster_id", "rank"]).reset_index(drop=True),
        rerun.sort_values(["cluster_id", "rank"]).reset_index(drop=True),
        check_exact=False,
    )
    progress = json.loads((Path(tmp_path) / "progress.json").read_text())
    assert progress["stage"] == "complete"


def test_cluster_sharded_keyword_engine_reruns_selected_shards_without_partial_overwrite(tmp_path):
    abstracts = pd.DataFrame(
        {
            "uid": ["D1", "D2", "D3", "D4", "D5", "D6"],
            "title": [
                "Quantum sensor devices",
                "Quantum magnetic sensing",
                "Solar cell materials",
                "Perovskite solar devices",
                "Graph neural networks",
                "Traffic graph forecasting",
            ],
            "abstract": [
                "Quantum sensing improves magnetic field measurement.",
                "Magnetic quantum sensors support precision metrology.",
                "Solar cell materials improve photovoltaic efficiency.",
                "Perovskite solar cells improve device efficiency.",
                "Graph neural networks learn molecular representations.",
                "Traffic forecasting uses graph neural network models.",
            ],
            "pubyear": [2018, 2019, 2020, 2021, 2022, 2023],
        }
    )
    membership = pd.DataFrame(
        {"uid": ["D1", "D2", "D3", "D4", "D5", "D6"], "cluster": [0, 0, 1, 1, 2, 2]}
    )
    abstract_path = Path(tmp_path) / "abstracts.parquet"
    membership_path = Path(tmp_path) / "membership.parquet"
    output_dir = Path(tmp_path) / "keyword_v2"
    abstracts.to_parquet(abstract_path, index=False)
    membership.to_parquet(membership_path, index=False)

    common = dict(
        abstract_path=abstract_path,
        membership_path=membership_path,
        cluster_level="cluster",
        keyword_engine="cluster_sharded",
        cluster_sharded_output_dir=output_dir,
        include_title=True,
        ngram_min=2,
        ngram_max=3,
        top_n_keywords=4,
        candidate_pool_floor=8,
        candidate_pool_target=12,
        candidate_pool_large=16,
        candidate_pool_hard_max=24,
        target_docs_per_shard=3,
        max_clusters_per_shard=2,
        min_df_unigram=1,
        min_df_phrase=1,
        phrase_min_count_per_cluster=1,
        n_jobs=1,
        parallel_backend="sequential",
        candidate_mining_progress_interval_docs=1,
        candidate_mining_prune_interval_docs=1,
        candidate_mining_prune_multiplier=2,
    )
    full = run_keyword_pipeline(KeywordExtractionConfig(**common))
    assert set(full["cluster_id"]) == {0, 1, 2}

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    target_shard = int(manifest["shards"][1]["shard_id"])
    (output_dir / "candidates" / f"candidate_shard_{target_shard:04d}.parquet").unlink()
    (output_dir / "candidates" / f"candidate_shard_{target_shard:04d}.done.json").unlink()
    (output_dir / "final" / f"keyword_shard_{target_shard:04d}.parquet").unlink()
    (output_dir / "final" / f"keyword_shard_{target_shard:04d}.done.json").unlink()

    rerun = run_keyword_pipeline(
        KeywordExtractionConfig(
            **common,
            cluster_sharded_shard_ids=(target_shard,),
            scoring_shard_resume=True,
        )
    )

    assert set(rerun["cluster_id"]) == {0, 1, 2}
    assert len(rerun) == len(full)
    run_summary = json.loads((output_dir / "run_summary.json").read_text(encoding="utf-8"))
    assert run_summary["active_shard_ids"] == [target_shard]
    assert run_summary["aggregate_candidate_shards"] == len(manifest["shards"])


def test_cluster_sharded_final_scoring_uses_quality_rerank(tmp_path):
    candidate_path = Path(tmp_path) / "candidate_shard_0000.parquet"
    pd.DataFrame(
        {
            "cluster_id": [0, 0],
            "term": ["session", "session based recommendation"],
            "local_tf": [120, 30],
            "local_doc_df": [40, 15],
            "title_tf": [0, 5],
            "abstract_tf": [120, 25],
            "first_year": [2020, 2020],
            "last_year": [2024, 2024],
            "channel_flags": ["frequency", "phrase_ngram|title_weighted"],
            "artifact_risk": [0.0, 0.0],
            "candidate_score_hint": [10.0, 5.0],
        }
    ).to_parquet(candidate_path, index=False)
    global_stats = pd.DataFrame(
        {
            "term_id": [0, 1],
            "term": ["session", "session based recommendation"],
            "total_tf": [120, 30],
            "total_doc_df": [40, 15],
            "cluster_df": [1, 1],
            "max_cluster_tf": [120, 30],
            "first_year": [2020, 2020],
            "last_year": [2024, 2024],
            "artifact_risk": [0.0, 0.0],
            "cluster_entropy": [0.0, 0.0],
            "term_keep_score": [10.0, 5.0],
        }
    )
    cfg = KeywordExtractionConfig(
        abstract_path=Path(tmp_path) / "abstracts.parquet",
        membership_path=Path(tmp_path) / "membership.parquet",
        cluster_level="cluster",
        top_n_keywords=1,
        quality_rerank_enabled=True,
    )

    out_path = score_candidate_shard(cfg, candidate_path, global_stats, 100, Path(tmp_path))
    result = pd.read_parquet(out_path)

    assert result["term"].tolist() == ["session based recommendation"]
    row = result.iloc[0]
    assert row["keyword_engine"] == "cluster_sharded"
    assert row["keyword_label_tier"] == "primary_phrase"
    assert row["representative_score"] > 0
    assert row["display_label"] == "session based recommendation"


def test_cluster_sharded_final_scoring_drops_oxidation_state_gap_fragment_when_clean_terms_exist(tmp_path):
    candidate_path = Path(tmp_path) / "candidate_shard_0000.parquet"
    terms = [
        "ii aqueous",
        "aqueous solution",
        "pb ii",
        "removal heavy metals",
        "activated carbon",
    ]
    pd.DataFrame(
        {
            "cluster_id": [0] * len(terms),
            "term": terms,
            "local_tf": [164, 514, 277, 81, 284],
            "local_doc_df": [164, 514, 277, 81, 284],
            "title_tf": [164, 100, 20, 30, 25],
            "abstract_tf": [0, 414, 257, 51, 259],
            "first_year": [2001, 2001, 2001, 2001, 2001],
            "last_year": [2025, 2025, 2025, 2025, 2025],
            "channel_flags": [
                "frequency|phrase_ngram|title_weighted",
                "frequency|phrase_ngram|title_weighted",
                "frequency|phrase_ngram|title_weighted",
                "frequency|phrase_ngram|title_weighted",
                "frequency|phrase_ngram|title_weighted",
            ],
            "artifact_risk": [0.0] * len(terms),
            "candidate_score_hint": [12.0, 10.0, 9.0, 8.0, 7.0],
        }
    ).to_parquet(candidate_path, index=False)
    global_stats = pd.DataFrame(
        {
            "term_id": list(range(len(terms))),
            "term": terms,
            "total_tf": [164, 514, 277, 81, 284],
            "total_doc_df": [164, 514, 277, 81, 284],
            "cluster_df": [1, 9, 3, 1, 7],
            "max_cluster_tf": [164, 514, 277, 81, 284],
            "first_year": [2001] * len(terms),
            "last_year": [2025] * len(terms),
            "artifact_risk": [0.0] * len(terms),
            "cluster_entropy": [0.0] * len(terms),
            "term_keep_score": [6.0, 5.0, 4.0, 3.0, 2.0],
        }
    )
    cfg = KeywordExtractionConfig(
        abstract_path=Path(tmp_path) / "abstracts.parquet",
        membership_path=Path(tmp_path) / "membership.parquet",
        cluster_level="cluster",
        top_n_keywords=3,
        quality_rerank_enabled=True,
    )

    out_path = score_candidate_shard(cfg, candidate_path, global_stats, 10, Path(tmp_path))
    result = pd.read_parquet(out_path)
    flagged = pd.read_parquet(out_path.with_name(f"{out_path.stem}.flagged{out_path.suffix}"))

    assert "ii aqueous" not in result["term"].tolist()
    assert "ii aqueous" in flagged["term"].tolist()
    flagged_row = flagged[flagged["term"].eq("ii aqueous")].iloc[0]
    assert flagged_row["keyword_label_tier"] == "review_fragment"
    assert "oxidation_state_gap_fragment" in flagged_row["quality_flags"]
    assert flagged_row["quality_risk_family"] == "phrase_fragment"
    assert "cluster_replacement" in flagged_row["quality_flag_basis"]
    assert flagged_row["quality_flag_confidence"] == "medium"
    assert flagged_row["clean_view_action"] == "hide_from_clean"
    assert set(result["keyword_label_tier"]) == {"primary_phrase"}
    assert {"aqueous solution", "pb ii", "removal heavy metals"} & set(result["term"])


def test_cluster_sharded_clean_view_hides_review_short_forms_without_deleting_flagged_rows(tmp_path):
    candidate_path = Path(tmp_path) / "candidate_shard_0000.parquet"
    terms = [
        "dna bsa",
        "graph neural network",
        "graph learning",
        "network analysis",
    ]
    pd.DataFrame(
        {
            "cluster_id": [0] * len(terms),
            "term": terms,
            "local_tf": [200, 35, 20, 15],
            "local_doc_df": [80, 20, 10, 8],
            "title_tf": [5, 8, 2, 1],
            "abstract_tf": [195, 27, 18, 14],
            "first_year": [2020] * len(terms),
            "last_year": [2024] * len(terms),
            "channel_flags": [
                "frequency|phrase_ngram|title_weighted",
                "phrase_ngram|title_weighted",
                "phrase_ngram",
                "phrase_ngram",
            ],
            "artifact_risk": [0.0] * len(terms),
            "candidate_score_hint": [12.0, 6.0, 5.0, 4.0],
        }
    ).to_parquet(candidate_path, index=False)
    global_stats = pd.DataFrame(
        {
            "term_id": list(range(len(terms))),
            "term": terms,
            "total_tf": [200, 35, 20, 15],
            "total_doc_df": [80, 20, 10, 8],
            "cluster_df": [1, 1, 1, 1],
            "max_cluster_tf": [200, 35, 20, 15],
            "first_year": [2020] * len(terms),
            "last_year": [2024] * len(terms),
            "artifact_risk": [0.0] * len(terms),
            "cluster_entropy": [0.0] * len(terms),
            "term_keep_score": [12.0, 6.0, 5.0, 4.0],
        }
    )
    cfg = KeywordExtractionConfig(
        abstract_path=Path(tmp_path) / "abstracts.parquet",
        membership_path=Path(tmp_path) / "membership.parquet",
        cluster_level="cluster",
        top_n_keywords=3,
        quality_rerank_enabled=True,
    )

    out_path = score_candidate_shard(cfg, candidate_path, global_stats, 10, Path(tmp_path))
    result = pd.read_parquet(out_path)
    flagged = pd.read_parquet(out_path.with_name(f"{out_path.stem}.flagged{out_path.suffix}"))

    assert "dna bsa" not in result["term"].tolist()
    assert not result["keyword_label_tier"].astype(str).str.startswith("review_").any()
    assert "dna bsa" in flagged["term"].tolist()
    flagged_row = flagged[flagged["term"].eq("dna bsa")].iloc[0]
    assert flagged_row["keyword_label_tier"] == "review_short_form"
    assert "candidate_short_form" in flagged_row["quality_flags"]
    assert flagged_row["quality_risk_family"] == "short_form_unresolved"
    assert "shape_only" in flagged_row["quality_flag_basis"]
    assert flagged_row["quality_flag_confidence"] == "low"
    assert flagged_row["clean_view_action"] == "hide_from_clean"


def test_cluster_sharded_final_scoring_uses_abbreviation_lookup(tmp_path):
    candidate_path = Path(tmp_path) / "candidate_shard_0000.parquet"
    pd.DataFrame(
        {
            "cluster_id": [0, 0],
            "term": ["gnn", "graph neural network"],
            "local_tf": [200, 35],
            "local_doc_df": [80, 20],
            "title_tf": [5, 8],
            "abstract_tf": [195, 27],
            "first_year": [2020, 2020],
            "last_year": [2024, 2024],
            "channel_flags": ["frequency|acronym_like", "phrase_ngram|title_weighted"],
            "artifact_risk": [0.0, 0.0],
            "candidate_score_hint": [12.0, 6.0],
        }
    ).to_parquet(candidate_path, index=False)
    global_stats = pd.DataFrame(
        {
            "term_id": [0, 1],
            "term": ["gnn", "graph neural network"],
            "total_tf": [200, 35],
            "total_doc_df": [80, 20],
            "cluster_df": [1, 1],
            "max_cluster_tf": [200, 35],
            "first_year": [2020, 2020],
            "last_year": [2024, 2024],
            "artifact_risk": [0.0, 0.0],
            "cluster_entropy": [0.0, 0.0],
            "term_keep_score": [12.0, 6.0],
        }
    )
    lookup = {
        "global": {
            "gnn": {
                "long_form": "graph neural network",
                "support_docs": 2,
                "cluster_support_docs": 0,
                "confidence": 0.9,
                "is_ambiguous": False,
                "ambiguity_type": "none",
                "top_support_ratio": 1.0,
                "status": "corpus_expanded",
                "usable": True,
            }
        },
        "cluster": {
            (0, "gnn"): {
                "long_form": "graph neural network",
                "support_docs": 2,
                "cluster_support_docs": 2,
                "confidence": 0.95,
                "is_ambiguous": False,
                "ambiguity_type": "none",
                "top_support_ratio": 1.0,
                "status": "cluster_expanded",
                "usable": True,
            }
        },
    }
    cfg = KeywordExtractionConfig(
        abstract_path=Path(tmp_path) / "abstracts.parquet",
        membership_path=Path(tmp_path) / "membership.parquet",
        cluster_level="cluster",
        top_n_keywords=1,
        quality_rerank_enabled=True,
    )

    out_path = score_candidate_shard(
        cfg,
        candidate_path,
        global_stats,
        100,
        Path(tmp_path),
        abbreviation_lookup=lookup,
        abbreviation_lookup_digest="test-abbreviation-lookup",
    )
    result = pd.read_parquet(out_path)

    assert result["term"].tolist() == ["graph neural network"]
    assert result.iloc[0]["keyword_label_tier"] == "primary_phrase"


def test_cluster_sharded_preflight_writes_budget_summary(tmp_path):
    abstracts = pd.DataFrame(
        {
            "uid": ["D1", "D2", "D3", "D4", "D5"],
            "title": ["A", "B", "C", "D", "E"],
            "abstract": ["a", "b", "c", "d", "e"],
            "pubyear": [2020, 2020, 2021, 2021, 2022],
        }
    )
    membership = pd.DataFrame(
        {"uid": ["D1", "D2", "D3", "D4", "D5"], "cluster": [0, 0, 0, 1, 1]}
    )
    abstract_path = Path(tmp_path) / "abstracts.parquet"
    membership_path = Path(tmp_path) / "membership.parquet"
    output_dir = Path(tmp_path) / "preflight"
    abstracts.to_parquet(abstract_path, index=False)
    membership.to_parquet(membership_path, index=False)

    cfg = KeywordExtractionConfig(
        abstract_path=abstract_path,
        membership_path=membership_path,
        cluster_level="cluster",
        keyword_engine="cluster_sharded",
        cluster_sharded_output_dir=output_dir,
        candidate_pool_floor=8,
        candidate_pool_target=12,
        candidate_pool_large=16,
        candidate_pool_hard_max=24,
        target_docs_per_shard=3,
        max_clusters_per_shard=2,
    )

    summary = run_cluster_sharded_preflight(cfg)

    assert summary["schema_version"] == "sciscape_keyword_cluster_sharded_preflight_v1"
    assert summary["status"] == "ok"
    assert summary["total_clusters"] == 2
    assert summary["total_docs"] == 5
    assert summary["shard_count"] == 2
    assert summary["expected_candidate_rows_upper_bound"] == 16
    assert summary["candidate_cap_stats"]["max"] == 8
    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "preflight_summary.json").exists()
    saved = json.loads((output_dir / "preflight_summary.json").read_text(encoding="utf-8"))
    assert saved["expected_candidate_rows_upper_bound"] == summary["expected_candidate_rows_upper_bound"]


def test_sos_shim_imports():
    import sos

    from sos.keyword_extraction import KeywordExtractionConfig as ShimCfg
    from sos.keyword_extraction import run_keyword_pipeline as shim_run

    assert sos.__version__
    assert ShimCfg is KeywordExtractionConfig
    assert shim_run is run_keyword_pipeline
