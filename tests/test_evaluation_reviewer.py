"""Tests for evaluation reviewer prompt adaptation."""

import pytest

from sciscape.evaluation.reviewer import (
    BOUNDARY_GOLD_SYSTEM_PROMPT,
    BOUNDARY_PLAUSIBILITY_SYSTEM_PROMPT,
    BELONGING_SYSTEM_PROMPT,
    CASE_TAXONOMY_SYSTEM_PROMPT,
    COMPARISON_SYSTEM_PROMPT,
    GEMINI_PROMPT_ADDON,
    RERANK_SYSTEM_PROMPT,
    _chat_completion_with_retry,
    _prompt_for_model,
    _resolve_plausibility_decision,
    _safe_json,
    _unswap_binary_winner,
    review_boundary_gold,
    review_boundary_plausibility,
    review_neighbor_rerank,
    review_neighbor_rerank_order_balanced,
)


class TestPromptForModel:

    def test_adds_gemini_addon_for_gemini_models(self):
        prompt = _prompt_for_model("BASE PROMPT", "gemini-2.5-pro")
        assert "BASE PROMPT" in prompt
        assert GEMINI_PROMPT_ADDON.strip() in prompt

    def test_leaves_non_gemini_prompt_unchanged(self):
        prompt = _prompt_for_model("BASE PROMPT", "gpt-4o-mini")
        assert prompt == "BASE PROMPT"

    def test_comparison_prompt_mentions_merge_split_tradeoff(self):
        assert "over-merged umbrella" in COMPARISON_SYSTEM_PROMPT
        assert "over-split fragment" in COMPARISON_SYSTEM_PROMPT
        assert "Group A and Group B independently" in COMPARISON_SYSTEM_PROMPT
        assert "TIE" in COMPARISON_SYSTEM_PROMPT

    def test_belonging_prompt_mentions_right_granularity(self):
        assert "narrowest coherent level" in BELONGING_SYSTEM_PROMPT
        assert "immediate research context" in BELONGING_SYSTEM_PROMPT

    def test_boundary_gold_prompt_requests_independent_judgments(self):
        assert "judge independently whether the target naturally belongs with Group A" in BOUNDARY_GOLD_SYSTEM_PROMPT
        assert "A_ONLY" in BOUNDARY_GOLD_SYSTEM_PROMPT
        assert "UNCLEAR" in BOUNDARY_GOLD_SYSTEM_PROMPT

    def test_boundary_plausibility_prompt_requests_unary_decision(self):
        assert "one candidate group" in BOUNDARY_PLAUSIBILITY_SYSTEM_PROMPT
        assert "PLAUSIBLE" in BOUNDARY_PLAUSIBILITY_SYSTEM_PROMPT
        assert "NOT_PLAUSIBLE" in BOUNDARY_PLAUSIBILITY_SYSTEM_PROMPT

    def test_rerank_prompt_mentions_local_neighborhood(self):
        assert "immediate research context" in RERANK_SYSTEM_PROMPT
        assert "broad bridge papers" in RERANK_SYSTEM_PROMPT
        assert "Judge only the local neighborhood shown here" in RERANK_SYSTEM_PROMPT
        assert "Group A and Group B independently" in RERANK_SYSTEM_PROMPT
        assert "TIE" in RERANK_SYSTEM_PROMPT

    def test_taxonomy_prompt_lists_allowed_labels(self):
        assert "single_cue_specificity" in CASE_TAXONOMY_SYSTEM_PROMPT
        assert "broad_context_noise" in CASE_TAXONOMY_SYSTEM_PROMPT
        assert "over_regularized_consensus" in CASE_TAXONOMY_SYSTEM_PROMPT
        assert '"primary_label"' in CASE_TAXONOMY_SYSTEM_PROMPT


class TestSafeJson:

    def test_parses_markdown_fenced_json(self):
        parsed = _safe_json("```json\n{\"winner\":\"A\"}\n```")
        assert parsed == {"winner": "A"}


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, content):
        self._content = content

    def create(self, **kwargs):
        return _FakeResponse(self._content)


class _SequentialFakeCompletions:
    def __init__(self, contents):
        self._contents = list(contents)
        self._index = 0

    def create(self, **kwargs):
        content = self._contents[self._index]
        self._index += 1
        return _FakeResponse(content)


class _FlakyCompletions:
    def __init__(self, failures, content, exc_type):
        self._failures = failures
        self._content = content
        self._exc_type = exc_type
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if self.calls <= self._failures:
            raise self._exc_type("temporary failure")
        return _FakeResponse(self._content)


class _FakeChat:
    def __init__(self, content):
        self.completions = _FakeCompletions(content)


class _SequentialFakeChat:
    def __init__(self, contents):
        self.completions = _SequentialFakeCompletions(contents)


class _FlakyFakeChat:
    def __init__(self, failures, content, exc_type):
        self.completions = _FlakyCompletions(failures, content, exc_type)


class _FakeClient:
    def __init__(self, content):
        self.chat = _FakeChat(content)
        self._sciscape_model = "gemini-2.5-pro"


class _SequentialFakeClient:
    def __init__(self, contents):
        self.chat = _SequentialFakeChat(contents)
        self._sciscape_model = "gemini-2.5-pro"


class _FlakyFakeClient:
    def __init__(self, failures, content, exc_type):
        self.chat = _FlakyFakeChat(failures, content, exc_type)
        self._sciscape_model = "gemini-2.5-pro"


class APITimeoutError(Exception):
    pass


class BadRequestError(Exception):
    pass


class TestTieHandling:

    def test_review_neighbor_rerank_preserves_explicit_winner_on_equal_scores(self):
        client = _FakeClient('{"winner":"A","score_a":5,"score_b":5,"reasoning":"Identical groups."}')
        result = review_neighbor_rerank(
            client,
            {"uid": "T", "title": "Target", "abstract": "Target abstract"},
            [{"uid": "A1", "title": "Neighbor 1", "abstract": "Abstract 1", "rank": 1}],
            [{"uid": "B1", "title": "Neighbor 1", "abstract": "Abstract 1", "rank": 1}],
            method_a="sum_minus_emb",
            method_b="consensus_all",
            model="gemini-2.5-pro",
            randomize=False,
        )
        assert result.winner == "A"
        assert result.score_a == 5
        assert result.score_b == 5

    def test_review_neighbor_rerank_falls_back_to_tie_when_winner_missing(self):
        client = _FakeClient('{"score_a":5,"score_b":5,"reasoning":"Identical groups."}')
        result = review_neighbor_rerank(
            client,
            {"uid": "T", "title": "Target", "abstract": "Target abstract"},
            [{"uid": "A1", "title": "Neighbor 1", "abstract": "Abstract 1", "rank": 1}],
            [{"uid": "B1", "title": "Neighbor 1", "abstract": "Abstract 1", "rank": 1}],
            method_a="sum_minus_emb",
            method_b="consensus_all",
            model="gemini-2.5-pro",
            randomize=False,
        )
        assert result.winner == "TIE"
        assert result.score_a == 5
        assert result.score_b == 5

    def test_unswap_binary_winner_preserves_tie(self):
        assert _unswap_binary_winner("TIE", swapped=True) == "TIE"
        assert _unswap_binary_winner("A", swapped=True) == "B"
        assert _unswap_binary_winner("B", swapped=True) == "A"

    def test_review_neighbor_rerank_preserves_tie_when_swapped(self, monkeypatch):
        monkeypatch.setattr("random.random", lambda: 0.0)
        client = _FakeClient('{"winner":"TIE","score_a":4,"score_b":4,"reasoning":"Both groups are comparable."}')
        result = review_neighbor_rerank(
            client,
            {"uid": "T", "title": "Target", "abstract": "Target abstract"},
            [{"uid": "A1", "title": "Neighbor 1", "abstract": "Abstract 1", "rank": 1}],
            [{"uid": "B1", "title": "Neighbor 2", "abstract": "Abstract 2", "rank": 1}],
            method_a="sum_minus_emb",
            method_b="consensus_all",
            model="gemini-2.5-pro",
            randomize=True,
        )
        assert result.swapped is True
        assert result.presented_winner == "TIE"
        assert result.winner == "TIE"
        assert result.score_a == 4
        assert result.score_b == 4

    def test_review_neighbor_rerank_records_presented_method_order(self, monkeypatch):
        monkeypatch.setattr("random.random", lambda: 0.0)
        client = _FakeClient('{"winner":"A","score_a":5,"score_b":4,"reasoning":"Presented A is better."}')
        result = review_neighbor_rerank(
            client,
            {"uid": "T", "title": "Target", "abstract": "Target abstract"},
            [{"uid": "A1", "title": "Neighbor 1", "abstract": "Abstract 1", "rank": 1}],
            [{"uid": "B1", "title": "Neighbor 2", "abstract": "Abstract 2", "rank": 1}],
            method_a="sum_minus_emb",
            method_b="consensus_all",
            model="gemini-2.5-pro",
            randomize=True,
        )
        assert result.swapped is True
        assert result.presented_method_a == "consensus_all"
        assert result.presented_method_b == "sum_minus_emb"
        assert result.method_a == "sum_minus_emb"
        assert result.method_b == "consensus_all"

    def test_review_neighbor_rerank_order_balanced_marks_disagreement_as_tie(self):
        client = _SequentialFakeClient(
            [
                '{"winner":"A","score_a":5,"score_b":4,"reasoning":"First order prefers A."}',
                '{"winner":"A","score_a":5,"score_b":4,"reasoning":"Reversed order still prefers presented A."}',
            ]
        )
        result = review_neighbor_rerank_order_balanced(
            client,
            {"uid": "T", "title": "Target", "abstract": "Target abstract"},
            [{"uid": "A1", "title": "Neighbor 1", "abstract": "Abstract 1", "rank": 1}],
            [{"uid": "B1", "title": "Neighbor 2", "abstract": "Abstract 2", "rank": 1}],
            method_a="sum_minus_emb",
            method_b="consensus_all",
            model="gemini-2.5-pro",
        )
        assert result.winner == "TIE"
        assert result.order_sensitive is True
        assert result.order_balance_mode == "dual_pass"
        assert len(result.balanced_passes) == 2
        assert result.score_a == 4.5
        assert result.score_b == 4.5

    def test_review_neighbor_rerank_order_balanced_keeps_stable_winner(self):
        client = _SequentialFakeClient(
            [
                '{"winner":"A","score_a":5,"score_b":4,"reasoning":"Original order prefers A."}',
                '{"winner":"B","score_a":4,"score_b":5,"reasoning":"Reversed order still prefers original A."}',
            ]
        )
        result = review_neighbor_rerank_order_balanced(
            client,
            {"uid": "T", "title": "Target", "abstract": "Target abstract"},
            [{"uid": "A1", "title": "Neighbor 1", "abstract": "Abstract 1", "rank": 1}],
            [{"uid": "B1", "title": "Neighbor 2", "abstract": "Abstract 2", "rank": 1}],
            method_a="sum_minus_emb",
            method_b="consensus_all",
            model="gemini-2.5-pro",
        )
        assert result.winner == "A"
        assert result.order_sensitive is False
        assert result.score_a == 5.0
        assert result.score_b == 4.0

    def test_review_boundary_gold_unswaps_presented_membership(self, monkeypatch):
        monkeypatch.setattr("random.random", lambda: 0.0)
        client = _FakeClient(
            '{"belongs_with_a":"YES","belongs_with_b":"NO","decision":"A_ONLY","confidence":4,"reasoning":"Presented A fits."}'
        )
        result = review_boundary_gold(
            client,
            {"uid": "T", "title": "Target", "abstract": "Target abstract"},
            [{"uid": "A1", "title": "Neighbor 1", "abstract": "Abstract 1"}],
            [{"uid": "B1", "title": "Neighbor 2", "abstract": "Abstract 2"}],
            method_a="sum_minus_emb",
            method_b="consensus_all",
            model="gemini-2.5-pro",
            randomize=True,
        )
        assert result.swapped is True
        assert result.presented_method_a == "consensus_all"
        assert result.presented_method_b == "sum_minus_emb"
        assert result.belongs_with_a is False
        assert result.belongs_with_b is True
        assert result.decision == "B_ONLY"

    def test_review_boundary_plausibility_parses_decision(self):
        client = _FakeClient(
            '{"decision":"PLAUSIBLE","confidence":4,"reasoning":"The group fits the target."}'
        )
        result = review_boundary_plausibility(
            client,
            {"uid": "T", "title": "Target", "abstract": "Target abstract"},
            [{"uid": "A1", "title": "Neighbor 1", "abstract": "Abstract 1"}],
            method="cc_only",
            model="gemini-2.5-pro",
        )
        assert result.decision == "PLAUSIBLE"
        assert result.confidence == 4
        assert result.method == "cc_only"

    def test_resolve_plausibility_decision_accepts_boolish_values(self):
        assert _resolve_plausibility_decision("yes") == "PLAUSIBLE"
        assert _resolve_plausibility_decision("no") == "NOT_PLAUSIBLE"
        assert _resolve_plausibility_decision("other") == "UNCLEAR"


class TestRetryHandling:

    def test_chat_completion_with_retry_recovers_from_timeout(self, monkeypatch):
        client = _FlakyFakeClient(1, '{"winner":"A"}', APITimeoutError)
        sleeps = []
        monkeypatch.setattr("sciscape.evaluation.reviewer.time.sleep", lambda seconds: sleeps.append(seconds))

        response = _chat_completion_with_retry(
            client,
            model="gemini-2.5-pro",
            messages=[{"role": "user", "content": "hello"}],
            max_attempts=3,
            timeout=1.0,
        )

        assert response.choices[0].message.content == '{"winner":"A"}'
        assert client.chat.completions.calls == 2
        assert sleeps == [1.0]

    def test_chat_completion_with_retry_raises_non_retryable_error(self, monkeypatch):
        client = _FlakyFakeClient(1, '{"winner":"A"}', BadRequestError)
        monkeypatch.setattr("sciscape.evaluation.reviewer.time.sleep", lambda seconds: None)

        with pytest.raises(BadRequestError):
            _chat_completion_with_retry(
                client,
                model="gemini-2.5-pro",
                messages=[{"role": "user", "content": "hello"}],
                max_attempts=3,
                timeout=1.0,
            )
