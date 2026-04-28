"""LLM-based blind cluster quality reviewer.

Given a target paper and its cluster neighbors, asks the LLM:
"Are these papers a cohesive research group?"

Supports blind comparison: show neighbors from method A and B
without revealing which is which.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

log = logging.getLogger(__name__)

_RETRYABLE_LLM_ERROR_NAMES = {
    "APITimeoutError",
    "APIConnectionError",
    "InternalServerError",
    "RateLimitError",
    "ReadTimeout",
    "TimeoutError",
}

PROMPT_JSON_RULES = """Output rules:
- Return ONLY one JSON object.
- Do not use markdown fences.
- Do not add commentary before or after the JSON.
- Use only the requested keys.
- Base the judgment only on the provided titles and abstracts.
"""

REVIEW_SYSTEM_PROMPT = f"""You are a scientific research evaluator.

You will be given one target paper and a set of papers claimed to be in the same research cluster.

Task:
- infer the most specific shared research theme, if any
- judge whether the neighbors form a coherent topic around the target
- flag papers that do not fit the main theme

Evaluation criteria:
- topical alignment to the target paper
- specificity of the shared theme
- internal consistency among neighbors
- presence of clear outliers or mixed subtopics

Return ONLY valid JSON:
{{
  "cohesion_score": <1-5>,
  "theme": "<one-line description of the shared theme, or 'no clear theme'>",
  "outliers": [<indices of papers that don't fit, 0-indexed>],
  "reasoning": "<1-2 sentences>"
}}

Scoring:
5 = Tight, specific topic with near-complete thematic agreement
4 = Clear shared area with minor variation or one small outlier
3 = Related but broad; plausible umbrella cluster
2 = Weak connection; mixed subtopics or loose relation
1 = No coherent shared theme

{PROMPT_JSON_RULES}
"""

COMPARISON_SYSTEM_PROMPT = f"""You are a scientific research evaluator performing a blind A/B comparison.

You will be given:
- one target paper
- Group A: papers proposed as the target's cluster neighbors by one method
- Group B: papers proposed as the target's cluster neighbors by another method

Task:
- first infer the narrowest target-centered research theme supported by the target title and abstract
- compare Group A and Group B as candidate neighborhoods for that target-centered theme
- decide which group is the better research cluster for the target paper
- score each group on thematic fit, internal coherence, and the correctness of its semantic granularity

Important:
- Ignore group labels A/B as arbitrary presentation order.
- Your judgment should remain the same if the group labels were swapped.
- Ignore cluster size, method names, and any imagined metadata not shown.
- Score Group A and Group B independently before deciding a winner.
- Distinguish these cases carefully:
  - coherent refinement: narrower but still complete and on-topic
  - over-merged umbrella: broad group that mixes adjacent subtopics, methods, or application domains
  - over-split fragment: small or specific group that drops obvious close neighbors needed to define the target's actual subtopic
  - wrong-topic group: poor thematic match to the target
- A broader umbrella group is not automatically better just because it looks safer.
- A narrower group is better only if it is still coherent and still captures the target's immediate research context.
- A single exact-match paper does not automatically win if the rest of that group is noisier, broader, or worse ranked.
- Prefer the group that best balances:
  - direct fit to the target
  - internal coherence
  - specificity
  - avoiding over-merge
  - avoiding over-split
- If the groups are substantively tied after independent scoring, return equal scores and set winner to TIE.

Return ONLY valid JSON:
{{
  "winner": "<A or B or TIE>",
  "score_a": <1-5>,
  "score_b": <1-5>,
  "reasoning": "<2-3 sentences explaining why one is better>"
}}

Score rubric:
5 = tight, target-centered, coherent, and at the right granularity
4 = clearly good fit with only minor breadth or omission
3 = plausible but either somewhat over-merged or somewhat over-split
2 = clearly over-merged, over-split, or thematically mixed
1 = poor fit to the target

{PROMPT_JSON_RULES}
"""


BELONGING_SYSTEM_PROMPT = f"""You are a scientific research evaluator.

You will be given one target paper and two candidate research groups.

Task:
- decide which group the target paper belongs to more naturally
- base the decision on thematic fit, terminology overlap, problem setting, method family, application domain, and semantic granularity
- prefer the group that places the target at the narrowest coherent level that still preserves its immediate research context

Important:
- Ignore A/B ordering bias.
- A broad umbrella group should lose if the other group captures a coherent, more target-specific subtopic.
- A narrow group should lose if it seems brittle, omits obvious close neighbors, or isolates the target too aggressively.
- Think in terms of "where does this paper most naturally belong as a research neighborhood?" not merely "which group is less wrong?"
- You must choose A or B. If both are imperfect, choose the less-wrong group.

Return ONLY valid JSON:
{{
  "belongs_to": "<A or B>",
  "confidence": <1-5>,
  "reasoning": "<2-3 sentences explaining why the target fits better in that group>"
}}

Confidence:
5 = one group clearly matches both topic and granularity
4 = clearly better fit with limited doubt
3 = ambiguous but one side still preserves the target's context better
2 = very ambiguous and weak fit
1 = target fits neither group well

{PROMPT_JSON_RULES}
"""

BOUNDARY_GOLD_SYSTEM_PROMPT = f"""You are a scientific research evaluator labeling a boundary case.

You will be given:
- one target paper
- Group A: papers that one clustering method places around the target
- Group B: papers that another clustering method places around the target

Task:
- judge independently whether the target naturally belongs with Group A
- judge independently whether the target naturally belongs with Group B
- convert those two yes/no judgments into one boundary decision label

Decision labels:
- A_ONLY = target belongs with Group A but not Group B
- B_ONLY = target belongs with Group B but not Group A
- BOTH = both groups are valid local neighborhoods for the target
- NEITHER = neither group is a good local neighborhood for the target
- UNCLEAR = evidence is too ambiguous to decide confidently

Important:
- Ignore A/B ordering bias.
- Judge Group A and Group B independently before deciding.
- A broad umbrella group should not receive credit if it loses the target's narrow coherent context.
- A narrow group should not receive credit if it omits obvious close neighbors needed for the target's immediate context.
- Return UNCLEAR only when the evidence is genuinely ambiguous after independent judging.

Return ONLY valid JSON:
{{
  "belongs_with_a": "<YES or NO or UNCLEAR>",
  "belongs_with_b": "<YES or NO or UNCLEAR>",
  "decision": "<A_ONLY or B_ONLY or BOTH or NEITHER or UNCLEAR>",
  "confidence": <1-5>,
  "reasoning": "<2-3 sentences explaining the boundary judgment>"
}}

Confidence:
5 = boundary decision is clear
4 = clearly preferred with limited ambiguity
3 = meaningful ambiguity remains but one interpretation is still better
2 = very ambiguous
1 = almost no confidence

{PROMPT_JSON_RULES}
"""

BOUNDARY_PLAUSIBILITY_SYSTEM_PROMPT = f"""You are a scientific research evaluator judging one local neighborhood.

You will be given:
- one target paper
- one candidate group of papers that a clustering method places around the target

Task:
- decide whether this group plausibly represents the target paper's immediate research neighborhood
- judge topical fit, internal coherence, semantic granularity, and whether the group is too broad or too narrow

Decision labels:
- PLAUSIBLE = the group is a valid local neighborhood for the target
- NOT_PLAUSIBLE = the group is too broad, too narrow, incoherent, or off-topic
- UNCLEAR = the evidence is too ambiguous to decide confidently

Important:
- A broad umbrella group should not receive credit if it loses the target's narrow coherent context.
- A narrow group should not receive credit if it omits obvious close neighbors needed for the target's immediate context.
- Return UNCLEAR only when the evidence is genuinely ambiguous.

Return ONLY valid JSON:
{{
  "decision": "<PLAUSIBLE or NOT_PLAUSIBLE or UNCLEAR>",
  "confidence": <1-5>,
  "reasoning": "<2-3 sentences explaining the plausibility judgment>"
}}

Confidence:
5 = plausibility decision is clear
4 = clear decision with limited ambiguity
3 = meaningful ambiguity remains but one interpretation is still better
2 = very ambiguous
1 = almost no confidence

{PROMPT_JSON_RULES}
"""

RERANK_SYSTEM_PROMPT = f"""You are a scientific research evaluator performing a blind A/B comparison of local graph neighborhoods.

You will be given:
- one target paper
- Group A: the top-ranked local neighbors proposed by one graph construction
- Group B: the top-ranked local neighbors proposed by another graph construction

Task:
- decide which ranked neighborhood better captures the target paper's immediate research context
- focus on the quality of the local neighborhood around the target, not on global cluster counts
- prefer the group whose higher-ranked papers are more directly connected to the target's specific topic, method family, and problem setting

Important:
- Ignore group labels A/B as arbitrary presentation order.
- Your judgment should remain the same if the group labels were swapped.
- Score Group A and Group B independently before deciding a winner.
- Treat the order inside each group as a salience hint: higher-ranked neighbors are claimed to be more central to the target.
- Penalize broad bridge papers that are ranked too highly if they dilute the target's immediate subtopic.
- Penalize brittle over-splitting if a group omits obvious close neighbors needed to define the target's local research neighborhood.
- Reward coherent refinement when a narrower group preserves the target's immediate context better than a broad umbrella group.
- A single exact-match paper does not automatically win if the rest of that group is noisier, broader, or worse ranked.
- Judge only the local neighborhood shown here, not the whole partition.
- If the two neighborhoods are equally good or equally flawed, return equal scores and set winner to TIE rather than inventing a difference.

Return ONLY valid JSON:
{{
  "winner": "<A or B or TIE>",
  "score_a": <1-5>,
  "score_b": <1-5>,
  "reasoning": "<2-3 sentences explaining which local neighborhood is better>"
}}

Score rubric:
5 = excellent immediate neighborhood; top-ranked neighbors are direct, coherent matches
4 = clearly good local neighborhood with only minor noise or omission
3 = plausible but somewhat broad, noisy, or incomplete
2 = weak local neighborhood with clear bridges, omissions, or mismatches
1 = poor local neighborhood for the target

{PROMPT_JSON_RULES}
"""

GROUP_COHESION_SYSTEM_PROMPT = f"""You are a scientific research evaluator.

You will be given a set of papers claimed to be in the same research cluster. There is no target paper in this task.

Task:
- infer the most specific common theme
- judge whether the set is a coherent research group
- count and identify outliers

Return ONLY valid JSON:
{{
  "cohesion_score": <1-5>,
  "theme": "<one-line description of the shared theme, or 'no clear theme'>",
  "n_outliers": <number of papers that don't fit the main theme>,
  "outlier_indices": [<0-indexed>],
  "reasoning": "<2-3 sentences>"
}}

Scoring:
5 = tight, specific topic with strong internal agreement
4 = clear shared area with at most one minor outlier
3 = broad but still plausible umbrella cluster
2 = mixed subtopics with weak coherence
1 = no coherent research theme

{PROMPT_JSON_RULES}
"""

OUTLIER_SYSTEM_PROMPT = f"""You are a scientific research evaluator.

You will be given a target paper and its cluster neighbors.

Task:
- identify which neighbor papers do not belong with the target
- state the core theme that connects the target and the non-outlier neighbors

Return ONLY valid JSON:
{{
  "n_outliers": <number of papers that don't belong>,
  "outlier_indices": [<0-indexed list of outlier papers>],
  "cluster_theme": "<the theme that connects the target and the good neighbors>",
  "reasoning": "<1-2 sentences about why the outliers don't fit>"
}}

{PROMPT_JSON_RULES}
"""

CASE_TAXONOMY_SYSTEM_PROMPT = f"""You are a scientific research evaluator assigning a primary case taxonomy to a local neighborhood comparison.

You will be given:
- one target paper
- Group A: one candidate local neighborhood
- Group B: another candidate local neighborhood
- the winner of a prior blind review between A and B

Task:
- infer the main reason the winning group beat the losing group
- assign exactly one primary taxonomy label from the allowed list
- explain the winning group's main advantage and the losing group's main failure mode

Allowed labels:
- single_cue_specificity
  The winner preserves a decisive narrow cue such as a catalyst metal, analyte, enzyme, named compound, or exact architecture.
- broad_context_noise
  The loser is too broad because it pulls in bridge papers or general context that dilutes the target's immediate subtopic.
- method_family_coherence
  The winner better preserves a coherent method, measurement, modeling, or synthesis family.
- material_family_coherence
  The winner better preserves the relevant material, compound, framework, or catalyst family.
- application_umbrella_noise
  The loser drifts toward a broad application umbrella instead of the target's more specific problem setting.
- semantic_drift
  The loser shifts into a nearby but different topic, meaning, or problem despite surface similarity.
- coherent_refinement
  The winner is a narrower but still complete and coherent refinement of a broader neighborhood.
- over_regularized_consensus
  The loser appears over-regularized: it suppresses a decisive cue or omits obvious close neighbors needed for the target.

Rules:
- Choose exactly one label.
- Prefer the most specific label that explains why the winner is better.
- Use only the provided titles and abstracts.
- Treat the stated winner as fixed; do not relitigate the winner.

Return ONLY valid JSON:
{{
  "primary_label": "<one allowed label>",
  "winner_advantage": "<1-2 sentences>",
  "loser_failure_mode": "<1-2 sentences>",
  "confidence": <1-5>
}}

Confidence:
5 = label clearly explains the case
4 = strong explanation with minor ambiguity
3 = plausible but not unique
2 = weak fit
1 = highly ambiguous

{PROMPT_JSON_RULES}
"""

GEMINI_PROMPT_ADDON = """Gemini-specific calibration:
- Be conservative about claiming a shared theme when the evidence is weak.
- Prefer lexical/topic evidence from the abstracts over stylistic similarity.
- Do not reward broad umbrella topics when one group is more specifically aligned to the target.
- Distinguish coherent subtopic refinement from brittle over-splitting.
- Prefer the narrowest coherent neighborhood that still preserves the target's immediate research context.
- Do not prefer the first-presented group by default; score both groups independently before selecting a winner.
- If both groups are noisy, still choose the one with fewer topical mismatches.
"""


@dataclass
class BelongingResult:
    """Result of 'which group does the target belong to?'"""
    target_uid: str
    belongs_to: str  # "A" or "B"
    confidence: int
    reasoning: str
    method_a: str
    method_b: str
    raw_response: str
    presented_belongs_to: str = ""
    presented_method_a: str = ""
    presented_method_b: str = ""
    swapped: bool = False


@dataclass
class BoundaryGoldResult:
    """Gold-label adjudication for a local boundary case."""
    target_uid: str
    decision: str
    belongs_with_a: bool | None
    belongs_with_b: bool | None
    confidence: int
    reasoning: str
    method_a: str
    method_b: str
    raw_response: str
    presented_decision: str = ""
    presented_belongs_with_a: bool | None = None
    presented_belongs_with_b: bool | None = None
    presented_method_a: str = ""
    presented_method_b: str = ""
    swapped: bool = False


@dataclass
class BoundaryPlausibilityResult:
    """Unary plausibility judgment for one local boundary neighborhood."""
    target_uid: str
    decision: str
    confidence: int
    reasoning: str
    method: str
    raw_response: str


@dataclass
class GroupCohesionResult:
    """Result of group-only cohesion evaluation (no target)."""
    cohesion_score: int
    theme: str
    n_outliers: int
    outlier_indices: List[int]
    reasoning: str
    method: str
    raw_response: str


@dataclass
class OutlierResult:
    """Result of outlier detection in a cluster."""
    target_uid: str
    n_outliers: int
    outlier_indices: List[int]
    cluster_theme: str
    reasoning: str
    method: str
    raw_response: str


@dataclass
class ReviewResult:
    """Result of a single cluster review."""
    target_uid: str
    cohesion_score: int
    theme: str
    outliers: List[int]
    reasoning: str
    raw_response: str
    method: str = ""


@dataclass
class ComparisonResult:
    """Result of a blind A vs B comparison."""
    target_uid: str
    winner: str  # "A", "B", or "TIE"
    score_a: int
    score_b: int
    reasoning: str
    method_a: str
    method_b: str
    raw_response: str
    presented_winner: str = ""
    presented_score_a: int = 0
    presented_score_b: int = 0
    presented_method_a: str = ""
    presented_method_b: str = ""
    swapped: bool = False
    order_sensitive: bool = False
    order_balance_mode: str = "single_pass"
    balanced_passes: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class TaxonomyResult:
    """Primary case taxonomy for a reviewed local neighborhood comparison."""
    target_uid: str
    primary_label: str
    winner_advantage: str
    loser_failure_mode: str
    confidence: int
    winner: str
    method_a: str
    method_b: str
    raw_response: str


def _format_paper(uid: str, title: str, abstract: str, index: int) -> str:
    abs_trunc = abstract[:500] + "..." if len(abstract) > 500 else abstract
    return f"[{index}] {title}\n    {abs_trunc}"


def _format_ranked_paper(paper: dict, index: int) -> str:
    abs_trunc = paper.get("abstract", "")[:500]
    if len(paper.get("abstract", "")) > 500:
        abs_trunc += "..."
    rank = paper.get("rank", index + 1)
    return f"[rank {rank}] {paper.get('title', '')}\n    {abs_trunc}"


def _prompt_for_model(base_prompt: str, model: str) -> str:
    """Adapt system prompts for provider/model-specific quirks."""
    if "gemini" in model.lower():
        return f"{base_prompt.rstrip()}\n\n{GEMINI_PROMPT_ADDON}\n"
    return base_prompt


def _unswap_binary_winner(winner: str, *, swapped: bool) -> str:
    """Map a presented A/B winner back to the original order.

    Non-binary outputs such as ``TIE`` are preserved.
    """
    if not swapped:
        return winner
    if winner == "A":
        return "B"
    if winner == "B":
        return "A"
    return winner


def _resolve_comparison_winner(winner: str, *, score_a: int, score_b: int) -> str:
    """Resolve the final comparison winner.

    Prefer an explicit valid winner returned by the model. If the winner field
    is missing or invalid, fall back to the score difference and finally to
    ``TIE`` when the scores are equal.
    """
    if winner in {"A", "B", "TIE"}:
        return winner
    if score_a > score_b:
        return "A"
    if score_b > score_a:
        return "B"
    return "TIE"


def _parse_boolish(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    raw = str(value).strip().upper()
    if raw in {"YES", "Y", "TRUE", "T", "1"}:
        return True
    if raw in {"NO", "N", "FALSE", "F", "0"}:
        return False
    return None


def _resolve_boundary_decision(
    decision: str,
    *,
    belongs_with_a: bool | None,
    belongs_with_b: bool | None,
) -> str:
    raw = str(decision).strip().upper()
    if raw in {"A_ONLY", "B_ONLY", "BOTH", "NEITHER", "UNCLEAR"}:
        return raw
    if belongs_with_a is True and belongs_with_b is False:
        return "A_ONLY"
    if belongs_with_a is False and belongs_with_b is True:
        return "B_ONLY"
    if belongs_with_a is True and belongs_with_b is True:
        return "BOTH"
    if belongs_with_a is False and belongs_with_b is False:
        return "NEITHER"
    return "UNCLEAR"


def _resolve_plausibility_decision(decision: Any) -> str:
    raw = str(decision).strip().upper() if decision is not None else ""
    if raw in {"PLAUSIBLE", "NOT_PLAUSIBLE", "UNCLEAR"}:
        return raw
    if raw in {"YES", "Y", "TRUE", "T", "1"}:
        return "PLAUSIBLE"
    if raw in {"NO", "N", "FALSE", "F", "0"}:
        return "NOT_PLAUSIBLE"
    return "UNCLEAR"


def _is_retryable_llm_exception(exc: Exception) -> bool:
    if exc.__class__.__name__ in _RETRYABLE_LLM_ERROR_NAMES:
        return True
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "timed out",
            "timeout",
            "temporarily unavailable",
            "connection reset",
            "connection aborted",
            "rate limit",
            "429",
            "502",
            "503",
            "504",
        )
    )


def _chat_completion_with_retry(
    client,
    *,
    model: str,
    messages: Sequence[dict[str, str]],
    max_attempts: int = 6,
    timeout: float = 180.0,
):
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return client.chat.completions.create(
                model=model,
                messages=list(messages),
                timeout=timeout,
            )
        except Exception as exc:  # pragma: no cover - exercised via fake client tests
            last_exc = exc
            if not _is_retryable_llm_exception(exc) or attempt == max_attempts:
                raise
            sleep_s = min(30.0, float(2 ** (attempt - 1)))
            log.warning(
                "LLM request failed with %s on attempt %d/%d; retrying in %.1fs",
                exc.__class__.__name__,
                attempt,
                max_attempts,
                sleep_s,
            )
            time.sleep(sleep_s)
    raise RuntimeError("LLM retry loop exited without returning or raising") from last_exc


def _comparison_pass_payload(result: "ComparisonResult", *, pass_id: str) -> Dict[str, Any]:
    return {
        "pass_id": pass_id,
        "winner": result.winner,
        "score_a": result.score_a,
        "score_b": result.score_b,
        "reasoning": result.reasoning,
        "method_a": result.method_a,
        "method_b": result.method_b,
        "presented_winner": result.presented_winner,
        "presented_score_a": result.presented_score_a,
        "presented_score_b": result.presented_score_b,
        "presented_method_a": result.presented_method_a,
        "presented_method_b": result.presented_method_b,
        "swapped": result.swapped,
    }


def _normalize_to_original_order(
    result: "ComparisonResult",
    *,
    original_method_a: str,
    original_method_b: str,
) -> "ComparisonResult":
    if result.method_a == original_method_a and result.method_b == original_method_b:
        return result
    if result.method_a == original_method_b and result.method_b == original_method_a:
        return ComparisonResult(
            target_uid=result.target_uid,
            winner=_unswap_binary_winner(result.winner, swapped=True),
            score_a=result.score_b,
            score_b=result.score_a,
            reasoning=result.reasoning,
            method_a=original_method_a,
            method_b=original_method_b,
            raw_response=result.raw_response,
            presented_winner=result.presented_winner,
            presented_score_a=result.presented_score_a,
            presented_score_b=result.presented_score_b,
            presented_method_a=result.presented_method_a,
            presented_method_b=result.presented_method_b,
            swapped=result.swapped,
            order_sensitive=result.order_sensitive,
            order_balance_mode=result.order_balance_mode,
            balanced_passes=list(result.balanced_passes),
        )
    raise ValueError(
        "Comparison result methods do not match the expected original labels: "
        f"{result.method_a}, {result.method_b} vs {original_method_a}, {original_method_b}"
    )


def review_cluster(
    client,
    target: dict,
    neighbors: Sequence[dict],
    *,
    model: str | None = None,
) -> ReviewResult:
    """Review cluster cohesion for one target + neighbors.

    Parameters
    ----------
    client : OpenAI-compatible client
    target : dict with uid, title, abstract
    neighbors : list of dicts with uid, title, abstract
    """
    model = model or getattr(client, "_sciscape_model", "gpt-oss:20b")

    user_parts = [
        "TARGET PAPER:",
        _format_paper(target["uid"], target.get("title", ""), target.get("abstract", ""), -1),
        "\nCLUSTER NEIGHBORS:",
    ]
    for i, n in enumerate(neighbors):
        user_parts.append(_format_paper(n["uid"], n.get("title", ""), n.get("abstract", ""), i))

    user_content = "\n".join(user_parts)

    response = _chat_completion_with_retry(
        client,
        model=model,
        messages=[
            {"role": "system", "content": _prompt_for_model(REVIEW_SYSTEM_PROMPT, model)},
            {"role": "user", "content": user_content},
        ],
    )
    raw = response.choices[0].message.content.strip()
    parsed = _safe_json(raw)

    return ReviewResult(
        target_uid=target["uid"],
        cohesion_score=int(parsed.get("cohesion_score", 0)),
        theme=str(parsed.get("theme", "")),
        outliers=parsed.get("outliers", []),
        reasoning=str(parsed.get("reasoning", "")),
        raw_response=raw,
    )


def review_comparison(
    client,
    target: dict,
    neighbors_a: Sequence[dict],
    neighbors_b: Sequence[dict],
    *,
    method_a: str = "A",
    method_b: str = "B",
    model: str | None = None,
    randomize: bool = True,
) -> ComparisonResult:
    """Blind A/B comparison of two clustering results.

    Parameters
    ----------
    randomize : bool
        If True, randomly swap A/B order to prevent position bias.
    """
    import random
    model = model or getattr(client, "_sciscape_model", "gpt-oss:20b")

    # Optionally swap to prevent position bias
    original_method_a = method_a
    original_method_b = method_b
    swapped = False
    if randomize and random.random() < 0.5:
        neighbors_a, neighbors_b = neighbors_b, neighbors_a
        method_a, method_b = method_b, method_a
        swapped = True
    presented_method_a = method_a
    presented_method_b = method_b

    user_parts = [
        "TARGET PAPER:",
        _format_paper(target["uid"], target.get("title", ""), target.get("abstract", ""), -1),
        "\nGROUP A:",
    ]
    for i, n in enumerate(neighbors_a):
        user_parts.append(_format_paper(n["uid"], n.get("title", ""), n.get("abstract", ""), i))
    user_parts.append("\nGROUP B:")
    for i, n in enumerate(neighbors_b):
        user_parts.append(_format_paper(n["uid"], n.get("title", ""), n.get("abstract", ""), i))

    user_content = "\n".join(user_parts)

    response = _chat_completion_with_retry(
        client,
        model=model,
        messages=[
            {"role": "system", "content": _prompt_for_model(COMPARISON_SYSTEM_PROMPT, model)},
            {"role": "user", "content": user_content},
        ],
    )
    raw = response.choices[0].message.content.strip()
    parsed = _safe_json(raw)

    winner_presented = str(parsed.get("winner", "")).upper()
    score_a_presented = int(parsed.get("score_a", 0))
    score_b_presented = int(parsed.get("score_b", 0))
    winner = winner_presented
    score_a = score_a_presented
    score_b = score_b_presented

    # Unswap if needed
    if swapped:
        winner = _unswap_binary_winner(winner, swapped=True)
        score_a, score_b = score_b, score_a
        method_a, method_b = original_method_a, original_method_b

    winner = _resolve_comparison_winner(winner, score_a=score_a, score_b=score_b)

    return ComparisonResult(
        target_uid=target["uid"],
        winner=winner,
        score_a=score_a,
        score_b=score_b,
        reasoning=str(parsed.get("reasoning", "")),
        method_a=method_a,
        method_b=method_b,
        raw_response=raw,
        presented_winner=winner_presented,
        presented_score_a=score_a_presented,
        presented_score_b=score_b_presented,
        presented_method_a=presented_method_a,
        presented_method_b=presented_method_b,
        swapped=swapped,
    )


def review_neighbor_rerank(
    client,
    target: dict,
    neighbors_a: Sequence[dict],
    neighbors_b: Sequence[dict],
    *,
    method_a: str = "A",
    method_b: str = "B",
    model: str | None = None,
    randomize: bool = True,
) -> ComparisonResult:
    """Blind A/B review of two ranked local neighborhoods around a target."""
    import random

    model = model or getattr(client, "_sciscape_model", "gpt-oss:20b")
    original_method_a = method_a
    original_method_b = method_b
    swapped = False
    if randomize and random.random() < 0.5:
        neighbors_a, neighbors_b = neighbors_b, neighbors_a
        method_a, method_b = method_b, method_a
        swapped = True
    presented_method_a = method_a
    presented_method_b = method_b

    user_parts = [
        "TARGET PAPER:",
        _format_paper(target["uid"], target.get("title", ""), target.get("abstract", ""), -1),
        "\nGROUP A: ranked local neighbors (smaller rank = stronger local connection)",
    ]
    for i, n in enumerate(neighbors_a):
        user_parts.append(_format_ranked_paper(n, i))
    user_parts.append("\nGROUP B: ranked local neighbors (smaller rank = stronger local connection)")
    for i, n in enumerate(neighbors_b):
        user_parts.append(_format_ranked_paper(n, i))

    response = _chat_completion_with_retry(
        client,
        model=model,
        messages=[
            {"role": "system", "content": _prompt_for_model(RERANK_SYSTEM_PROMPT, model)},
            {"role": "user", "content": "\n".join(user_parts)},
        ],
    )
    raw = response.choices[0].message.content.strip()
    parsed = _safe_json(raw)

    winner_presented = str(parsed.get("winner", "")).upper()
    score_a_presented = int(parsed.get("score_a", 0))
    score_b_presented = int(parsed.get("score_b", 0))
    winner = winner_presented
    score_a = score_a_presented
    score_b = score_b_presented

    if swapped:
        winner = _unswap_binary_winner(winner, swapped=True)
        score_a, score_b = score_b, score_a
        method_a, method_b = original_method_a, original_method_b

    winner = _resolve_comparison_winner(winner, score_a=score_a, score_b=score_b)

    return ComparisonResult(
        target_uid=target["uid"],
        winner=winner,
        score_a=score_a,
        score_b=score_b,
        reasoning=str(parsed.get("reasoning", "")),
        method_a=method_a,
        method_b=method_b,
        raw_response=raw,
        presented_winner=winner_presented,
        presented_score_a=score_a_presented,
        presented_score_b=score_b_presented,
        presented_method_a=presented_method_a,
        presented_method_b=presented_method_b,
        swapped=swapped,
    )


def review_neighbor_rerank_order_balanced(
    client,
    target: dict,
    neighbors_a: Sequence[dict],
    neighbors_b: Sequence[dict],
    *,
    method_a: str = "A",
    method_b: str = "B",
    model: str | None = None,
) -> ComparisonResult:
    """Two-pass rerank review that neutralizes presentation order.

    The same comparison is evaluated twice:
    1. original presentation order (A then B)
    2. reversed presentation order (B then A)

    If the normalized winners disagree across passes, the final outcome is
    conservatively marked as ``TIE`` and flagged as order-sensitive.
    """

    first = review_neighbor_rerank(
        client,
        target,
        neighbors_a,
        neighbors_b,
        method_a=method_a,
        method_b=method_b,
        model=model,
        randomize=False,
    )
    second_raw = review_neighbor_rerank(
        client,
        target,
        neighbors_b,
        neighbors_a,
        method_a=method_b,
        method_b=method_a,
        model=model,
        randomize=False,
    )
    second = _normalize_to_original_order(
        second_raw,
        original_method_a=method_a,
        original_method_b=method_b,
    )

    stable = first.winner == second.winner
    if stable:
        winner = first.winner
        reasoning = first.reasoning
    else:
        winner = "TIE"
        reasoning = (
            "Order-balanced passes disagreed across presentation order; "
            "the comparison is treated conservatively as a tie."
        )

    return ComparisonResult(
        target_uid=target["uid"],
        winner=winner,
        score_a=round((first.score_a + second.score_a) / 2.0, 4),
        score_b=round((first.score_b + second.score_b) / 2.0, 4),
        reasoning=reasoning,
        method_a=method_a,
        method_b=method_b,
        raw_response=json.dumps(
            {
                "pass_a_then_b": first.raw_response,
                "pass_b_then_a": second.raw_response,
            },
            ensure_ascii=False,
        ),
        presented_winner="",
        presented_score_a=0,
        presented_score_b=0,
        presented_method_a="",
        presented_method_b="",
        swapped=False,
        order_sensitive=not stable,
        order_balance_mode="dual_pass",
        balanced_passes=[
            _comparison_pass_payload(first, pass_id="A_then_B"),
            _comparison_pass_payload(second, pass_id="B_then_A"),
        ],
    )


def review_belonging(
    client,
    target: dict,
    group_a: Sequence[dict],
    group_b: Sequence[dict],
    *,
    method_a: str = "A",
    method_b: str = "B",
    model: str | None = None,
    randomize: bool = True,
) -> BelongingResult:
    """Eval 1: Which group does the target naturally belong to?"""
    import random
    model = model or getattr(client, "_sciscape_model", "gpt-oss:20b")

    original_method_a = method_a
    original_method_b = method_b
    swapped = False
    if randomize and random.random() < 0.5:
        group_a, group_b = group_b, group_a
        method_a, method_b = method_b, method_a
        swapped = True
    presented_method_a = method_a
    presented_method_b = method_b

    user_parts = [
        "TARGET PAPER:",
        _format_paper(target["uid"], target.get("title", ""), target.get("abstract", ""), -1),
        "\nGROUP A:",
    ]
    for i, n in enumerate(group_a):
        user_parts.append(_format_paper(n["uid"], n.get("title", ""), n.get("abstract", ""), i))
    user_parts.append("\nGROUP B:")
    for i, n in enumerate(group_b):
        user_parts.append(_format_paper(n["uid"], n.get("title", ""), n.get("abstract", ""), i))

    response = _chat_completion_with_retry(
        client,
        model=model,
        messages=[
            {"role": "system", "content": _prompt_for_model(BELONGING_SYSTEM_PROMPT, model)},
            {"role": "user", "content": "\n".join(user_parts)},
        ],
    )
    raw = response.choices[0].message.content.strip()
    parsed = _safe_json(raw)

    belongs_presented = str(parsed.get("belongs_to", "")).upper()
    belongs = belongs_presented
    if swapped:
        belongs = _unswap_binary_winner(belongs, swapped=True)
        method_a, method_b = original_method_a, original_method_b

    return BelongingResult(
        target_uid=target["uid"],
        belongs_to=belongs,
        confidence=int(parsed.get("confidence", 0)),
        reasoning=str(parsed.get("reasoning", "")),
        method_a=method_a,
        method_b=method_b,
        raw_response=raw,
        presented_belongs_to=belongs_presented,
        presented_method_a=presented_method_a,
        presented_method_b=presented_method_b,
        swapped=swapped,
    )


def review_boundary_gold(
    client,
    target: dict,
    group_a: Sequence[dict],
    group_b: Sequence[dict],
    *,
    method_a: str = "A",
    method_b: str = "B",
    model: str | None = None,
    randomize: bool = True,
) -> BoundaryGoldResult:
    """Blind gold-label adjudication for a disagreement boundary case."""
    import random

    model = model or getattr(client, "_sciscape_model", "gpt-oss:20b")
    original_method_a = method_a
    original_method_b = method_b
    swapped = False
    if randomize and random.random() < 0.5:
        group_a, group_b = group_b, group_a
        method_a, method_b = method_b, method_a
        swapped = True
    presented_method_a = method_a
    presented_method_b = method_b

    user_parts = [
        "TARGET PAPER:",
        _format_paper(target["uid"], target.get("title", ""), target.get("abstract", ""), -1),
        "\nGROUP A:",
    ]
    for i, n in enumerate(group_a):
        user_parts.append(_format_paper(n["uid"], n.get("title", ""), n.get("abstract", ""), i))
    user_parts.append("\nGROUP B:")
    for i, n in enumerate(group_b):
        user_parts.append(_format_paper(n["uid"], n.get("title", ""), n.get("abstract", ""), i))

    response = _chat_completion_with_retry(
        client,
        model=model,
        messages=[
            {"role": "system", "content": _prompt_for_model(BOUNDARY_GOLD_SYSTEM_PROMPT, model)},
            {"role": "user", "content": "\n".join(user_parts)},
        ],
    )
    raw = response.choices[0].message.content.strip()
    parsed = _safe_json(raw)

    presented_a = _parse_boolish(parsed.get("belongs_with_a"))
    presented_b = _parse_boolish(parsed.get("belongs_with_b"))
    presented_decision = _resolve_boundary_decision(
        parsed.get("decision", ""),
        belongs_with_a=presented_a,
        belongs_with_b=presented_b,
    )

    belongs_with_a = presented_a
    belongs_with_b = presented_b
    if swapped:
        belongs_with_a, belongs_with_b = presented_b, presented_a
        method_a, method_b = original_method_a, original_method_b

    decision = _resolve_boundary_decision(
        "" if swapped else presented_decision,
        belongs_with_a=belongs_with_a,
        belongs_with_b=belongs_with_b,
    )

    return BoundaryGoldResult(
        target_uid=target["uid"],
        decision=decision,
        belongs_with_a=belongs_with_a,
        belongs_with_b=belongs_with_b,
        confidence=int(parsed.get("confidence", 0)),
        reasoning=str(parsed.get("reasoning", "")),
        method_a=method_a,
        method_b=method_b,
        raw_response=raw,
        presented_decision=presented_decision,
        presented_belongs_with_a=presented_a,
        presented_belongs_with_b=presented_b,
        presented_method_a=presented_method_a,
        presented_method_b=presented_method_b,
        swapped=swapped,
    )


def review_boundary_plausibility(
    client,
    target: dict,
    group: Sequence[dict],
    *,
    method: str = "",
    model: str | None = None,
) -> BoundaryPlausibilityResult:
    """Unary plausibility review for a single target-centered group."""
    model = model or getattr(client, "_sciscape_model", "gpt-oss:20b")
    user_parts = [
        "TARGET PAPER:",
        _format_paper(target["uid"], target.get("title", ""), target.get("abstract", ""), -1),
        "\nCANDIDATE LOCAL NEIGHBORHOOD:",
    ]
    for i, n in enumerate(group):
        user_parts.append(_format_paper(n["uid"], n.get("title", ""), n.get("abstract", ""), i))

    response = _chat_completion_with_retry(
        client,
        model=model,
        messages=[
            {"role": "system", "content": _prompt_for_model(BOUNDARY_PLAUSIBILITY_SYSTEM_PROMPT, model)},
            {"role": "user", "content": "\n".join(user_parts)},
        ],
    )
    raw = response.choices[0].message.content.strip()
    parsed = _safe_json(raw)
    return BoundaryPlausibilityResult(
        target_uid=target["uid"],
        decision=_resolve_plausibility_decision(parsed.get("decision", parsed.get("plausibility", ""))),
        confidence=int(parsed.get("confidence", 0)),
        reasoning=str(parsed.get("reasoning", "")),
        method=method,
        raw_response=raw,
    )


def review_group_cohesion(
    client,
    papers: Sequence[dict],
    *,
    method: str = "",
    model: str | None = None,
) -> GroupCohesionResult:
    """Eval 2: How cohesive is this group? (no target paper)"""
    model = model or getattr(client, "_sciscape_model", "gpt-oss:20b")

    user_parts = ["RESEARCH GROUP:"]
    for i, p in enumerate(papers):
        user_parts.append(_format_paper(p["uid"], p.get("title", ""), p.get("abstract", ""), i))

    response = _chat_completion_with_retry(
        client,
        model=model,
        messages=[
            {"role": "system", "content": _prompt_for_model(GROUP_COHESION_SYSTEM_PROMPT, model)},
            {"role": "user", "content": "\n".join(user_parts)},
        ],
    )
    raw = response.choices[0].message.content.strip()
    parsed = _safe_json(raw)

    return GroupCohesionResult(
        cohesion_score=int(parsed.get("cohesion_score", 0)),
        theme=str(parsed.get("theme", "")),
        n_outliers=int(parsed.get("n_outliers", 0)),
        outlier_indices=parsed.get("outlier_indices", []),
        reasoning=str(parsed.get("reasoning", "")),
        method=method,
        raw_response=raw,
    )


def review_outliers(
    client,
    target: dict,
    neighbors: Sequence[dict],
    *,
    method: str = "",
    model: str | None = None,
) -> OutlierResult:
    """Eval 3: How many outliers in this cluster?"""
    model = model or getattr(client, "_sciscape_model", "gpt-oss:20b")

    user_parts = [
        "TARGET PAPER:",
        _format_paper(target["uid"], target.get("title", ""), target.get("abstract", ""), -1),
        "\nCLUSTER NEIGHBORS:",
    ]
    for i, n in enumerate(neighbors):
        user_parts.append(_format_paper(n["uid"], n.get("title", ""), n.get("abstract", ""), i))

    response = _chat_completion_with_retry(
        client,
        model=model,
        messages=[
            {"role": "system", "content": _prompt_for_model(OUTLIER_SYSTEM_PROMPT, model)},
            {"role": "user", "content": "\n".join(user_parts)},
        ],
    )
    raw = response.choices[0].message.content.strip()
    parsed = _safe_json(raw)

    return OutlierResult(
        target_uid=target["uid"],
        n_outliers=int(parsed.get("n_outliers", 0)),
        outlier_indices=parsed.get("outlier_indices", []),
        cluster_theme=str(parsed.get("cluster_theme", "")),
        reasoning=str(parsed.get("reasoning", "")),
        method=method,
        raw_response=raw,
    )


def classify_case_taxonomy(
    client,
    target: dict,
    group_a: Sequence[dict],
    group_b: Sequence[dict],
    *,
    winner: str,
    method_a: str = "A",
    method_b: str = "B",
    model: str | None = None,
) -> TaxonomyResult:
    """Assign one primary taxonomy label to a reviewed A/B local neighborhood case."""
    model = model or getattr(client, "_sciscape_model", "gpt-oss:20b")

    user_parts = [
        "TARGET PAPER:",
        _format_paper(target["uid"], target.get("title", ""), target.get("abstract", ""), -1),
        "\nGROUP A:",
    ]
    for i, n in enumerate(group_a):
        user_parts.append(_format_ranked_paper(n, i))
    user_parts.append("\nGROUP B:")
    for i, n in enumerate(group_b):
        user_parts.append(_format_ranked_paper(n, i))
    user_parts.append(f"\nBLIND REVIEW WINNER: Group {winner}")
    user_parts.append(f"METHOD A LABEL: {method_a}")
    user_parts.append(f"METHOD B LABEL: {method_b}")

    response = _chat_completion_with_retry(
        client,
        model=model,
        messages=[
            {"role": "system", "content": _prompt_for_model(CASE_TAXONOMY_SYSTEM_PROMPT, model)},
            {"role": "user", "content": "\n".join(user_parts)},
        ],
    )
    raw = response.choices[0].message.content.strip()
    parsed = _safe_json(raw)

    return TaxonomyResult(
        target_uid=target["uid"],
        primary_label=str(parsed.get("primary_label", "")),
        winner_advantage=str(parsed.get("winner_advantage", "")),
        loser_failure_mode=str(parsed.get("loser_failure_mode", "")),
        confidence=int(parsed.get("confidence", 0)),
        winner=winner,
        method_a=method_a,
        method_b=method_b,
        raw_response=raw,
    )


def _safe_json(text: str) -> dict:
    """Parse JSON from LLM response, handling markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[\w-]*\s*", "", text, count=1)
        text = re.sub(r"\s*```$", "", text, count=1).strip()
    if "{" in text:
        text = text[text.find("{"):]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


__all__ = [
    "review_cluster", "review_comparison", "review_belonging", "review_boundary_gold",
    "review_boundary_plausibility",
    "review_group_cohesion", "review_outliers", "classify_case_taxonomy",
    "ReviewResult", "ComparisonResult", "BelongingResult", "BoundaryGoldResult",
    "BoundaryPlausibilityResult",
    "GroupCohesionResult", "OutlierResult", "TaxonomyResult",
]
