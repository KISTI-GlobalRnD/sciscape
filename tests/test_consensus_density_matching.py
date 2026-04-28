"""Tests for Protocol C density-matched helpers."""

from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import sys


_DENSITY_PATH = (
    Path(__file__).resolve().parents[1]
    / "research"
    / "consensus"
    / "scripts"
    / "run_density_matched_comparison.py"
)
sys.path.insert(0, str(_DENSITY_PATH.parent))
_DENSITY_SPEC = spec_from_file_location("consensus_density_match_test_module", _DENSITY_PATH)
assert _DENSITY_SPEC is not None and _DENSITY_SPEC.loader is not None
_DENSITY_MODULE = module_from_spec(_DENSITY_SPEC)
_DENSITY_SPEC.loader.exec_module(_DENSITY_MODULE)

_BRIDGE_PATH = (
    Path(__file__).resolve().parents[1]
    / "research"
    / "consensus"
    / "scripts"
    / "summarize_consensus_bridge.py"
)
_BRIDGE_SPEC = spec_from_file_location("consensus_bridge_test_module", _BRIDGE_PATH)
assert _BRIDGE_SPEC is not None and _BRIDGE_SPEC.loader is not None
_BRIDGE_MODULE = module_from_spec(_BRIDGE_SPEC)
_BRIDGE_SPEC.loader.exec_module(_BRIDGE_MODULE)

_uniform_scale_top_k = _DENSITY_MODULE._uniform_scale_top_k
_search_density_match = _DENSITY_MODULE._search_density_match
_search_scales = _DENSITY_MODULE._search_scales
_consensus_layers = _DENSITY_MODULE._consensus_layers
_is_current_payload = _BRIDGE_MODULE._is_current_payload
_load_payload = _BRIDGE_MODULE._load_payload
_index_results = _BRIDGE_MODULE._index_results


class TestDensityMatchedHelpers:

    def test_uniform_scaling_keeps_minimum_one(self):
        scaled = _uniform_scale_top_k({"bc_cosine": 30, "cc_cosine": 30, "dc_fractional": 30}, 0.01)
        assert scaled == {"bc_cosine": 1, "cc_cosine": 1, "dc_fractional": 1}

    def test_search_density_match_returns_within_tolerance(self, monkeypatch):
        def fake_combine_layers(_layers, *, strategy, top_k):
            achieved = sum(top_k.values()) * 100

            class FakeCombined:
                height = achieved

            return FakeCombined(), {}

        monkeypatch.setattr(_DENSITY_MODULE, "combine_layers", fake_combine_layers)
        selected, diagnostics = _search_density_match(
            {"bc_cosine": object(), "cc_cosine": object(), "dc_fractional": object()},
            strategy="consensus",
            original_layer_top_k={"bc_cosine": 30, "cc_cosine": 30, "dc_fractional": 30},
            target_edge_count=3000,
            tolerance=0.05,
            scale_min=0.05,
            scale_max=2.0,
            scale_step=0.05,
        )
        assert selected is not None
        assert selected["layer_top_k"] == {"bc_cosine": 10, "cc_cosine": 10, "dc_fractional": 10}
        assert selected["relative_edge_error"] == 0.0
        assert diagnostics["closest_within_tolerance"] == selected

    def test_search_density_match_prefers_scale_closest_to_one(self, monkeypatch):
        def fake_combine_layers(_layers, *, strategy, top_k):
            total = sum(top_k.values())
            if total == 90:
                achieved = 1100
            elif total == 60:
                achieved = 900
            else:
                achieved = total * 100

            class FakeCombined:
                height = achieved

            return FakeCombined(), {}

        monkeypatch.setattr(_DENSITY_MODULE, "combine_layers", fake_combine_layers)
        selected, _diagnostics = _search_density_match(
            {"bc_cosine": object(), "cc_cosine": object(), "dc_fractional": object()},
            strategy="consensus",
            original_layer_top_k={"bc_cosine": 30, "cc_cosine": 30, "dc_fractional": 30},
            target_edge_count=1000,
            tolerance=0.11,
            scale_min=0.66,
            scale_max=1.0,
            scale_step=0.34,
        )
        assert selected is not None
        assert selected["scale"] == 1.0

    def test_search_density_match_fails_cleanly_when_no_match(self, monkeypatch):
        def fake_combine_layers(_layers, *, strategy, top_k):
            achieved = 100000 + sum(top_k.values())

            class FakeCombined:
                height = achieved

            return FakeCombined(), {}

        monkeypatch.setattr(_DENSITY_MODULE, "combine_layers", fake_combine_layers)
        selected, diagnostics = _search_density_match(
            {"bc_cosine": object(), "cc_cosine": object(), "dc_fractional": object()},
            strategy="consensus",
            original_layer_top_k={"bc_cosine": 30, "cc_cosine": 30, "dc_fractional": 30},
            target_edge_count=3000,
            tolerance=0.05,
            scale_min=0.05,
            scale_max=0.5,
            scale_step=0.05,
        )
        assert selected is None
        assert diagnostics["best_overall"] is not None
        assert diagnostics["best_overall"]["relative_edge_error"] > 0.05

    def test_search_scales_always_includes_one(self):
        scales = _search_scales(0.1, 0.9, 0.2)
        assert 1.0 in scales

    def test_consensus_layers_rejects_unknown_method(self):
        try:
            _consensus_layers({"bc_cosine": object(), "cc_cosine": object()}, "bad_method")
        except ValueError as exc:
            assert "Unsupported consensus method" in str(exc)
        else:
            raise AssertionError("Expected ValueError for invalid consensus method")


class TestConsensusBridgeHelpers:

    def test_bridge_helpers_index_methods(self, tmp_path):
        payload = {
            "results": [
                {"method": "cc_cosine_only", "ami_mean": 0.9},
                {"method": "citation_consensus", "ami_mean": 0.85},
                {"method": "all_consensus", "ami_mean": 0.8},
            ]
        }
        path = tmp_path / "comparison.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        loaded = _load_payload(path)
        indexed = _index_results(loaded["results"])
        assert indexed["cc_cosine_only"]["ami_mean"] == 0.9
        assert indexed["citation_consensus"]["ami_mean"] == 0.85

    def test_current_payload_requires_protocol_and_layer_paths(self):
        assert not _is_current_payload({"protocol": "candidate_budget_matched"})
        assert not _is_current_payload({"layer_paths": {"emb_knn": "x.parquet"}})
        assert _is_current_payload(
            {
                "protocol": "candidate_budget_matched",
                "layer_paths": {"emb_knn": "x.parquet"},
            }
        )
