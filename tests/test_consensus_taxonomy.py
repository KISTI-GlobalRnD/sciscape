"""Tests for consensus taxonomy helpers."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "research"
    / "consensus"
    / "scripts"
    / "review_taxonomy"
    / "taxonomy"
    / "classify_case_taxonomy.py"
)
sys.path.insert(0, str(_SCRIPT_PATH.parent))
_SCRIPT_SPEC = spec_from_file_location("consensus_taxonomy_test_module", _SCRIPT_PATH)
assert _SCRIPT_SPEC is not None and _SCRIPT_SPEC.loader is not None
_SCRIPT_MODULE = module_from_spec(_SCRIPT_SPEC)
_SCRIPT_SPEC.loader.exec_module(_SCRIPT_MODULE)
_winner_method = _SCRIPT_MODULE._winner_method
_display_field_label = _SCRIPT_MODULE._display_field_label


class TestWinnerMethod:

    def test_returns_label_pair_for_a_winner(self):
        result = _winner_method({"comparison": {"winner": "A"}}, label_a="sum_minus_emb", label_b="consensus_all")
        assert result == ("sum_minus_emb", "consensus_all")

    def test_returns_label_pair_for_b_winner(self):
        result = _winner_method({"comparison": {"winner": "B"}}, label_a="sum_minus_emb", label_b="consensus_all")
        assert result == ("consensus_all", "sum_minus_emb")

    def test_returns_none_for_tie(self):
        result = _winner_method({"comparison": {"winner": "TIE"}}, label_a="sum_minus_emb", label_b="consensus_all")
        assert result is None

    def test_display_field_label_prefers_normalized_filename(self):
        review_path = Path("field_15_k06_sum_minus_emb_vs_consensus_bank_n48_order_balanced_gemini_v3_rank_shift_review.json")
        label = _display_field_label(review_path, {"field": "legacy_name"})
        assert label == "field_15_k06_sum_minus_emb_vs_consensus_bank_n48"
