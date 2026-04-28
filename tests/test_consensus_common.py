"""Tests for shared consensus research helpers."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


_COMMON_PATH = Path(__file__).resolve().parents[1] / "research" / "consensus" / "scripts" / "_common.py"
_COMMON_SPEC = spec_from_file_location("consensus_common_test_module", _COMMON_PATH)
assert _COMMON_SPEC is not None and _COMMON_SPEC.loader is not None
_COMMON_MODULE = module_from_spec(_COMMON_SPEC)
_COMMON_SPEC.loader.exec_module(_COMMON_MODULE)
_layer_top_k_metadata = _COMMON_MODULE._layer_top_k_metadata
combine_layers = _COMMON_MODULE.combine_layers
gamma_cache_key = _COMMON_MODULE.gamma_cache_key
infer_emb_mode = _COMMON_MODULE.infer_emb_mode
layer_provenance = _COMMON_MODULE.layer_provenance
load_layer_paths = _COMMON_MODULE.load_layer_paths
select_best_single_result = _COMMON_MODULE.select_best_single_result
validate_field_embedding_contract = _COMMON_MODULE.validate_field_embedding_contract


class TestLayerTopKMetadata:

    def test_keeps_per_layer_integer_budget(self):
        metadata = _layer_top_k_metadata(["cc", "bc"], 30)
        assert metadata == {"bc": 30, "cc": 30}

    def test_keeps_per_layer_dict_budget(self):
        metadata = _layer_top_k_metadata(["cc", "bc"], {"cc": 12, "bc": 8})
        assert metadata == {"bc": 8, "cc": 12}

    def test_uses_none_for_unfiltered_layers(self):
        metadata = _layer_top_k_metadata(["cc", "bc"], "none")
        assert metadata == {"bc": None, "cc": None}


class TestConsensusSelectionHelpers:

    def test_select_best_single_uses_tie_breakers(self):
        results = [
            {"method": "bc_cosine_only", "kind": "single_layer", "ami_mean": 0.9, "ami_std": 0.02, "max_pct": 40.0},
            {"method": "cc_cosine_only", "kind": "single_layer", "ami_mean": 0.9, "ami_std": 0.01, "max_pct": 45.0},
            {"method": "dc_fractional_only", "kind": "single_layer", "ami_mean": 0.89, "ami_std": 0.0, "max_pct": 10.0},
        ]
        best = select_best_single_result(results)
        assert best is not None
        assert best["method"] == "cc_cosine_only"

    def test_gamma_cache_key_changes_with_protocol_context(self, tmp_path):
        common = dict(
            edge_dir=tmp_path,
            strategy="consensus",
            layer_names=["bc_cosine", "cc_cosine"],
            top_k={"bc_cosine": 10, "cc_cosine": 10},
            target_pct=3.0,
            min_size=10,
        )
        key_a = gamma_cache_key(**common)
        key_b = gamma_cache_key(
            **common,
            protocol="edge_count_matched",
            cache_context={"search_strategy": "uniform", "target_edge_count": 1000},
        )
        assert key_a != key_b

    def test_combine_layers_int_and_uniform_dict_top_k_match(self):
        import polars as pl

        layer_a = pl.DataFrame(
            {
                "uid1": ["u1", "u1", "u2", "u2", "u3", "u3"],
                "uid2": ["u2", "u3", "u1", "u3", "u1", "u2"],
                "rel_sum2": [0.9, 0.8, 0.9, 0.7, 0.8, 0.7],
            }
        )
        layer_b = pl.DataFrame(
            {
                "uid1": ["u1", "u1", "u2", "u2", "u3", "u3"],
                "uid2": ["u2", "u3", "u1", "u3", "u1", "u2"],
                "rel_sum2": [0.5, 0.4, 0.5, 0.3, 0.4, 0.3],
            }
        )
        layers = {"bc_cosine": layer_a, "cc_cosine": layer_b}
        combined_int, _ = combine_layers(layers, strategy="consensus", top_k=2)
        combined_dict, _ = combine_layers(layers, strategy="consensus", top_k={"bc_cosine": 2, "cc_cosine": 2})
        assert combined_int.sort(["uid1", "uid2"]).equals(combined_dict.sort(["uid1", "uid2"]))

    def test_load_layer_paths_prefers_filtered_emb(self, tmp_path):
        (tmp_path / "bc_cosine.parquet").write_text("", encoding="utf-8")
        (tmp_path / "emb_full_knn30.parquet").write_text("", encoding="utf-8")
        (tmp_path / "emb_full_knn30_textfilt_txt20_abs20_reqabs.parquet").write_text("", encoding="utf-8")
        paths = load_layer_paths(tmp_path)
        assert paths["emb_knn"].name == "emb_full_knn30_textfilt_txt20_abs20_reqabs.parquet"

    def test_load_layer_paths_fails_on_multiple_filtered_emb_candidates(self, tmp_path):
        (tmp_path / "bc_cosine.parquet").write_text("", encoding="utf-8")
        (tmp_path / "emb_full_knn30_textfilt_txt20_abs20_reqabs.parquet").write_text("", encoding="utf-8")
        (tmp_path / "emb_alt_textfilt_variant.parquet").write_text("", encoding="utf-8")
        try:
            load_layer_paths(tmp_path)
        except ValueError as exc:
            assert "Ambiguous filtered embedding candidates" in str(exc)
        else:
            raise AssertionError("Expected ambiguity failure for multiple filtered embedding candidates")

    def test_validate_field_embedding_contract_requires_filtered_emb_for_textfilt_field(self, tmp_path):
        emb_path = tmp_path / "emb_full_knn30.parquet"
        emb_path.write_text("", encoding="utf-8")
        try:
            validate_field_embedding_contract("field_12_textfilt", {"emb_knn": emb_path})
        except RuntimeError as exc:
            assert "implies filtered embeddings" in str(exc)
        else:
            raise AssertionError("Expected RuntimeError for textfilt field without filtered embedding")

    def test_layer_provenance_records_embedding_mode(self, tmp_path):
        emb_path = tmp_path / "emb_full_knn30_textfilt_txt20_abs20_reqabs.parquet"
        emb_path.write_text("", encoding="utf-8")
        prov = layer_provenance({"emb_knn": emb_path})
        assert prov["emb_mode"] == "filtered"
        assert infer_emb_mode(emb_path) == "filtered"
