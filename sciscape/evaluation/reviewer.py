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
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

log = logging.getLogger(__name__)

REVIEW_SYSTEM_PROMPT = """You are a scientific research evaluator. You will be given a target paper and a set of papers claimed to be in the same research cluster.

Evaluate whether the cluster is cohesive — do these papers share a clear, specific research theme?

Return ONLY valid JSON:
{
  "cohesion_score": <1-5>,
  "theme": "<one-line description of the shared theme, or 'no clear theme'>",
  "outliers": [<indices of papers that don't fit, 0-indexed>],
  "reasoning": "<1-2 sentences>"
}

Scoring:
5 = Tight, specific topic (e.g., "graph neural networks for drug discovery")
4 = Clear shared area with minor variation
3 = Related but broad (e.g., "machine learning" as a catch-all)
2 = Weak connection, multiple unrelated sub-topics
1 = No coherent theme
"""

COMPARISON_SYSTEM_PROMPT = """You are a scientific research evaluator. You will be given a target paper and TWO sets of cluster neighbors (Group A and Group B). Each group claims to be the target's research cluster.

Evaluate which group is more cohesive with the target paper.

Return ONLY valid JSON:
{
  "winner": "<A or B>",
  "score_a": <1-5>,
  "score_b": <1-5>,
  "reasoning": "<2-3 sentences explaining why one is better>"
}
"""


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
    winner: str  # "A" or "B"
    score_a: int
    score_b: int
    reasoning: str
    method_a: str
    method_b: str
    raw_response: str


def _format_paper(uid: str, title: str, abstract: str, index: int) -> str:
    abs_trunc = abstract[:500] + "..." if len(abstract) > 500 else abstract
    return f"[{index}] {title}\n    {abs_trunc}"


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

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
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
    swapped = False
    if randomize and random.random() < 0.5:
        neighbors_a, neighbors_b = neighbors_b, neighbors_a
        method_a, method_b = method_b, method_a
        swapped = True

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

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": COMPARISON_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
    raw = response.choices[0].message.content.strip()
    parsed = _safe_json(raw)

    winner = str(parsed.get("winner", "")).upper()
    score_a = int(parsed.get("score_a", 0))
    score_b = int(parsed.get("score_b", 0))

    # Unswap if needed
    if swapped:
        winner = "B" if winner == "A" else "A"
        score_a, score_b = score_b, score_a
        method_a, method_b = method_b, method_a

    return ComparisonResult(
        target_uid=target["uid"],
        winner=winner,
        score_a=score_a,
        score_b=score_b,
        reasoning=str(parsed.get("reasoning", "")),
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


__all__ = ["review_cluster", "review_comparison", "ReviewResult", "ComparisonResult"]
