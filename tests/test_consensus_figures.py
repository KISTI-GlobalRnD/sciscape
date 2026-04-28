"""Tests for consensus paper-figure generation helpers."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "research" / "consensus" / "scripts" / "generate_figures.py"
_SCRIPT_SPEC = spec_from_file_location("consensus_generate_figures_test_module", _SCRIPT_PATH)
assert _SCRIPT_SPEC is not None and _SCRIPT_SPEC.loader is not None
_SCRIPT_MODULE = module_from_spec(_SCRIPT_SPEC)
_SCRIPT_SPEC.loader.exec_module(_SCRIPT_MODULE)

plot_rank_shift_review = _SCRIPT_MODULE.plot_rank_shift_review
plot_review_uncertainty = _SCRIPT_MODULE.plot_review_uncertainty
plot_taxonomy_summary = _SCRIPT_MODULE.plot_taxonomy_summary
plot_regime_model = _SCRIPT_MODULE.plot_regime_model
_matching_files = _SCRIPT_MODULE._matching_files


class TestConsensusFigureHelpers:

    def test_plot_rank_shift_review(self, tmp_path):
        payload = {
            "field": "field_15_k30_sum_minus_cc_vs_consensus_bank_n48_order_balanced_gemini_v3",
            "summary": {
                "comparison": {
                    "method_a": "sum_minus_cc",
                    "method_b": "consensus_all",
                    "method_a_wins": 10,
                    "method_b_wins": 31,
                    "ties_or_invalid": 7,
                }
            },
        }
        out_path = plot_rank_shift_review(payload, tmp_path)
        assert out_path is not None
        assert out_path.exists()

    def test_plot_review_uncertainty(self, tmp_path):
        payload = {
            "per_review": [
                {
                    "review_json": "research/consensus/results/case_banks_corrected/field_15_k06_sum_minus_emb_vs_consensus_bank_n48_order_balanced_gemini_v3_rank_shift_review.json",
                    "focal_win_rate": 0.6129,
                    "wilson95": [0.4382, 0.7627],
                },
                {
                    "review_json": "research/consensus/results/case_banks_corrected/field_15_k30_sum_minus_cc_vs_consensus_bank_n48_order_balanced_gemini_v3_rank_shift_review.json",
                    "focal_win_rate": 0.7561,
                    "wilson95": [0.6066, 0.8617],
                },
            ]
        }
        out_path = plot_review_uncertainty(payload, tmp_path)
        assert out_path is not None
        assert out_path.exists()

    def test_plot_taxonomy_summary(self, tmp_path):
        payload = {
            "label_counts": {
                "broad_context_noise": 10,
                "over_regularized_consensus": 4,
            },
            "label_by_winner": {
                "consensus_all": {"broad_context_noise": 8},
                "sum_minus_emb": {"broad_context_noise": 2, "over_regularized_consensus": 4},
            },
        }
        out_path = plot_taxonomy_summary(payload, tmp_path)
        assert out_path is not None
        assert out_path.exists()

    def test_plot_regime_model(self, tmp_path):
        payload = {
            "top_positive_coefficients": [
                {"feature": "num__shift_score", "coefficient": 0.3},
                {"feature": "num__mean_abs_rank_shift", "coefficient": 0.2},
            ],
            "top_negative_coefficients": [
                {"feature": "num__rank_jaccard", "coefficient": -0.4},
                {"feature": "num__cluster_overlap_coeff", "coefficient": -0.2},
            ],
        }
        out_path = plot_regime_model(payload, tmp_path)
        assert out_path is not None
        assert out_path.exists()

    def test_matching_files_skips_directories(self, tmp_path):
        (tmp_path / "good_rank_shift_review.json").write_text("{}", encoding="utf-8")
        (tmp_path / "bad_rank_shift_review.json").mkdir()
        matches = _matching_files(tmp_path, "*_rank_shift_review.json")
        assert matches == [tmp_path / "good_rank_shift_review.json"]
