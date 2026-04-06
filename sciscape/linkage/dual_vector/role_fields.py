"""Decompose a paper abstract into background / novelty role fields.

Given a raw abstract string, produces structured fields:
  - background_summary: sentences describing prior work, motivation, gap
  - novelty_claim: the paper's main contribution claim
  - method_core: methodology sentence(s)
  - main_findings: key result sentence(s)

These are then separately encoded by SPECTER2 to produce background vs
novelty embedding vectors for dual-vector link construction.

Decomposition strategy (in priority order):
  1. **Structured headers** — detect explicit "Background:", "Methods:", etc.
  2. **Cue-word matching** — keyword patterns (e.g., "we propose", "results show")
  3. **Positional fallback** — first sentence → background, middle → method, last → result
"""
from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Any


# ── Text utilities ──────────────────────────────────────────────────

def sql_quote(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def read_parquet_sql(path: str | Path) -> str:
    p = Path(path).resolve()
    s = str(p)
    if any(ch in s for ch in "*?[]"):
        return f"read_parquet({sql_quote(s)})"
    if p.is_dir():
        return f"read_parquet({sql_quote(str(p / '*.parquet'))})"
    if p.suffix.lower() == ".parquet":
        return f"read_parquet({sql_quote(s)})"
    raise SystemExit(f"Unsupported parquet input: {p}")


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def clean_abstract(text: str) -> str:
    cleaned = html.unescape(normalize_text(text))
    if not cleaned:
        return ""
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = cleaned.replace("\\n", " ").replace("\n", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


# Common abbreviations that should NOT be treated as sentence endings
_ABBREV_RE = re.compile(
    r"\b("
    r"et al|Fig|Figs|Tab|Eq|Eqs|Ref|Refs|Vol|No|vs|Dr|Mr|Mrs|Ms|Prof|Jr|Sr"
    r"|Inc|Ltd|Corp|Dept|Univ|etc|approx|ca|cf|ed|eds|trans|rev"
    r"|i\.e|e\.g|viz"
    r")\.\s+",
    re.IGNORECASE,
)

# Initials pattern: single uppercase letter followed by period (e.g., "J. K. Rowling")
_INITIALS_RE = re.compile(r"\b([A-Z])\.\s+(?=[A-Z][.\s])")


def split_sentences(text: str) -> list[str]:
    cleaned = clean_abstract(text)
    if not cleaned:
        return []

    _S = "\x00"  # sentinel character to protect non-sentence periods

    # Step 1: Protect abbreviations BEFORE any manipulation
    # "et al. showed" → "et al.\x00showed"
    def _protect_abbrev(m: re.Match) -> str:
        return m.group(0).replace(". ", "." + _S)
    cleaned = _ABBREV_RE.sub(_protect_abbrev, cleaned)

    # Step 2: Protect initials: "J.K." or "J. K." patterns
    # Match single uppercase letter + period + optional space + uppercase letter
    cleaned = re.sub(r"\b([A-Z])\.\s*(?=[A-Z])", lambda m: m.group(1) + "." + _S, cleaned)

    # Step 3: Handle missing space after period (e.g., "studied.It was" → split)
    # Only split on period + uppercase if NOT protected by sentinel
    cleaned = re.sub(r"\.([A-Z])", r". \1", cleaned)

    # Step 4: Split on sentence-ending punctuation followed by space
    parts = re.split(r"(?<=[.!?])\s+", cleaned)

    # Restore sentinel → space
    return [normalize_text(part.replace(_S, " ")) for part in parts
            if normalize_text(part.replace(_S, " "))]


def join_nonempty(parts: list[str]) -> str:
    return " ".join(part for part in parts if part)


# ── Language detection (lightweight) ────────────────────────────────

_ENGLISH_COMMON = frozenset({
    "the", "of", "and", "in", "to", "a", "is", "that", "for", "was",
    "on", "are", "with", "as", "this", "by", "from", "an", "be", "were",
    "or", "at", "which", "have", "has", "it", "not", "but", "we", "our",
    "can", "been", "its", "these", "their", "also", "about", "between",
    "more", "than", "other", "such", "into", "when", "both", "some",
    "may", "will", "each", "all", "would", "there", "do", "no", "if",
})


def _is_likely_english(text: str, threshold: float = 0.10) -> bool:
    """Quick heuristic: fraction of words that are common English words.

    Also checks that the text has enough Latin-alphabet content — texts
    in Arabic, CJK, etc. that contain few ASCII words are classified as
    non-English.
    """
    # Check Latin-alphabet ratio first
    latin_chars = len(re.findall(r"[a-zA-Z]", text))
    total_alpha = len(re.findall(r"\w", text))  # any word char (incl. unicode)
    if total_alpha > 10 and latin_chars / total_alpha < 0.5:
        return False  # predominantly non-Latin script

    words = re.findall(r"[a-zA-Z]+", text.lower())
    if len(words) < 5:
        return len(words) > 0  # no Latin words at all → non-English
    hits = sum(1 for w in words if w in _ENGLISH_COMMON)
    return (hits / len(words)) >= threshold


# ── Structured abstract detection ──────────────────────────────────

# Header patterns found in structured abstracts (case-insensitive)
_HEADER_RE = re.compile(
    r"^\s*(?:"
    r"(?P<background>background|introduction|context|purpose|objective|aim|motivation|rationale)"
    r"|(?P<method>method(?:s|ology)?|design|approach|materials?\s*(?:and|&)\s*methods?|procedure|experiment(?:al)?(?:\s+design)?|study\s+design)"
    r"|(?P<result>result(?:s)?|finding(?:s)?|outcome(?:s)?)"
    r"|(?P<conclusion>conclusion(?:s)?|discussion|implication(?:s)?|significance|summary)"
    r")\s*[:.\-]\s*",
    re.IGNORECASE,
)


def _try_structured_parse(text: str) -> dict[str, list[str]] | None:
    """Attempt to parse an abstract with explicit section headers.

    Returns a dict of {role: [sentences...]} if headers are found, else None.
    Requires at least 2 distinct header types to count as structured.
    """
    cleaned = clean_abstract(text)
    if not cleaned:
        return None

    # Split on header boundaries
    segments: list[tuple[str, str]] = []  # (role, text)
    current_role = ""
    current_text: list[str] = []

    for line in re.split(r"(?<=[.!?])\s+|\n+", cleaned):
        line = line.strip()
        if not line:
            continue
        m = _HEADER_RE.match(line)
        if m:
            if current_role and current_text:
                segments.append((current_role, " ".join(current_text)))
            # Determine which role matched
            for role in ("background", "method", "result", "conclusion"):
                if m.group(role):
                    current_role = role
                    break
            # Strip the header from the text
            remainder = line[m.end():].strip()
            current_text = [remainder] if remainder else []
        else:
            current_text.append(line)

    if current_role and current_text:
        segments.append((current_role, " ".join(current_text)))

    # Need at least 2 distinct header types
    roles_found = {role for role, _ in segments}
    if len(roles_found) < 2:
        return None

    result: dict[str, list[str]] = {"background": [], "method": [], "result": [], "conclusion": []}
    for role, text in segments:
        result[role].append(text)
    return result


# ── Cue-word sentence classification ───────────────────────────────

BACKGROUND_CUES = (
    "background",
    "however",
    "despite",
    "although",
    "while",
    "whereas",
    "challenge",
    "problem",
    "remains",
    "little is known",
    "not well understood",
    "poorly understood",
    "gap",
    "objective",
    "aim",
    "motivation",
    "purpose of this",
    "has received",
    "has attracted",
    "is essential",
    "is important",
    "is crucial",
    "is critical",
    "plays a key role",
    "have been widely",
    "has been extensively",
    "in recent years",
    "recently",
    "traditional",
    "conventional",
)

METHOD_CUES = (
    "we propose",
    "we present",
    "we report",
    "we develop",
    "we introduce",
    "we describe",
    "we design",
    "we employ",
    "we apply",
    "this paper",
    "this study",
    "this work",
    "this article",
    "this research",
    "in this article",
    "in the present",
    "the present study",
    "the present paper",
    "we investigate",
    "we evaluate",
    "we analyze",
    "we examine",
    "we explore",
    "we discuss",
    "we consider",
    "was investigated",
    "was examined",
    "was studied",
    "was assessed",
    "was proposed",
    "was developed",
    "were investigated",
    "were examined",
    "were studied",
    "extensively investigated",
    "by using",
    "by means of",
    "by employing",
    "method",
    "methodology",
    "approach",
    "framework",
    "algorithm",
    "simulation",
    "experiment",
    "was examined",
    "were examined",
    "was assessed",
    "were assessed",
    "was conducted",
    "was measured",
    "were measured",
    "was prepared",
    "were prepared",
    "was synthesized",
    "were synthesized",
    "was fabricated",
    "was characterized",
    "were characterized",
    "aims at",
    "aims to",
    "aimed at",
    "aimed to",
    "focus on",
    "focused on",
)

RESULT_CUES = (
    "results show",
    "results indicate",
    "results suggest",
    "results reveal",
    "our results",
    "the results",
    "demonstrate that",
    "demonstrated that",
    "we found",
    "we found that",
    "found that the",
    "it was found",
    "it was shown",
    "it is shown",
    "it was observed",
    "we observe that",
    "we conclude",
    "conclude that",
    "in conclusion",
    "these findings",
    "our findings",
    "the findings",
    "outperform",
    "these results",
    "our analysis show",
    "our analysis reveal",
    "confirmed that",
    "validated by",
)


def sentence_role(sentence: str, index: int, total: int) -> str:
    """Classify a sentence by role using cue words + position.

    Priority rules:
    - First sentence: background cues checked first (introductory context),
      then treat as "background" even if method cues match — the opening
      sentence of an abstract sets context, not methodology.
    - Middle sentences: result > method > background > positional.
    - Last sentence (if ≥4 total): lean toward result.
    """
    text = sentence.lower()

    has_result = any(cue in text for cue in RESULT_CUES)
    has_method = any(cue in text for cue in METHOD_CUES)
    has_background = any(cue in text for cue in BACKGROUND_CUES)

    # First sentence: favor background role (sets context)
    if index == 0:
        if has_background:
            return "background"
        # Even with method cues, first sentence is contextual → background
        # unless it's clearly a result
        if has_result and not has_method:
            return "result"
        return "background"

    # Last sentence in longer abstracts: lean toward result
    if total >= 4 and index >= total - 1:
        if has_result:
            return "result"
        if has_background:
            return "background"
        if has_method:
            return "method"
        return "result"

    # Middle sentences: standard priority
    if has_result:
        return "result"
    if has_method:
        return "method"
    if has_background:
        return "background"
    return "other"


# ── Main decomposition ─────────────────────────────────────────────

def build_role_fields(abstract: str) -> dict[str, Any]:
    """Decompose abstract into role-based fields.

    Returns a dict with keys:
      abstract, abstract_sentence_count,
      background_summary, novelty_claim, method_core, main_findings,
      contribution_type, role_heuristic_note, decomposition_method,
      bg_sentence_count, nov_sentence_count
    """
    sentences = split_sentences(abstract)

    # ── Detect non-abstract text (acknowledgements, ToC, truncated) ──
    full_text = " ".join(sentences).lower()
    is_non_abstract = (
        # Single-sentence non-abstracts
        (len(sentences) == 1 and (
            full_text.startswith("this paper was supported")
            or full_text.startswith("this work was supported")
            or full_text.startswith("this research was supported")
            or full_text.startswith("acknowledgement")
            or "chapter 1:" in full_text
            or "chapter 2:" in full_text
            or full_text.startswith("contents:")
            or full_text.startswith("table of contents")
            or re.match(r"^abstract\s+\w{1,10}$", full_text)
        ))
        # Multi-sentence non-abstracts (service descriptions, metadata)
        or "cheminform is a weekly" in full_text
        or "abstracting service" in full_text
    )
    if is_non_abstract:
        sentences = []  # treat as no-abstract

    # ── No abstract ──
    if not sentences:
        return {
            "abstract": "",
            "abstract_sentence_count": 0,
            "background_summary": "",
            "novelty_claim": "",
            "method_core": "",
            "main_findings": "",
            "contribution_type": "",
            "role_heuristic_note": "non_abstract_text" if is_non_abstract else "no_abstract",
            "decomposition_method": "none",
            "quality": "unusable",
            "bg_sentence_count": 0,
            "nov_sentence_count": 0,
        }

    # ── Non-English filter (before any decomposition attempt) ──
    if not _is_likely_english(" ".join(sentences)):
        return {
            "abstract": clean_abstract(abstract),
            "abstract_sentence_count": len(sentences),
            "background_summary": "",
            "novelty_claim": "",
            "method_core": "",
            "main_findings": "",
            "contribution_type": "",
            "role_heuristic_note": "non_english_filtered",
            "decomposition_method": "none",
            "bg_sentence_count": 0,
            "nov_sentence_count": 0,
            "quality": "filtered",
        }

    # ── Single sentence ──
    if len(sentences) == 1:
        sentence = sentences[0]
        return {
            "abstract": sentence,
            "abstract_sentence_count": 1,
            "background_summary": sentence,
            "novelty_claim": sentence,
            "method_core": sentence,
            "main_findings": sentence,
            "contribution_type": "single_sentence",
            "role_heuristic_note": "single_sentence",
            "decomposition_method": "degenerate",
            "bg_sentence_count": 1,
            "nov_sentence_count": 1,
            "quality": "unusable",
        }

    # ── Try structured parse first ──
    structured = _try_structured_parse(abstract)
    if structured:
        # bg = background section; nov = method + result + conclusion
        bg_text = join_nonempty(structured["background"]) or sentences[0]
        nov_text = join_nonempty(
            structured["method"] + structured["result"] + structured["conclusion"]
        ) or sentences[-1]

        method_text = join_nonempty(structured["method"])
        result_text = join_nonempty(structured["result"])
        conclusion_text = join_nonempty(structured["conclusion"])

        method_core = method_text or nov_text.split(". ")[0]
        main_findings = result_text or conclusion_text or sentences[-1]
        novelty_claim = method_text or main_findings

        roles_found = [r for r in ("background", "method", "result", "conclusion")
                       if structured[r]]
        note = f"structured_headers:{'+'.join(roles_found)}"

        if method_text and result_text:
            contribution_type = "method_and_result"
        elif method_text:
            contribution_type = "method"
        elif result_text:
            contribution_type = "result"
        else:
            contribution_type = "structured_partial"

        bg_sents = len(split_sentences(bg_text))
        nov_sents = len(split_sentences(nov_text))
        note += f"|bg_sents={bg_sents},nov_sents={nov_sents}"

        return {
            "abstract": clean_abstract(abstract),
            "abstract_sentence_count": len(sentences),
            "background_summary": bg_text,
            "novelty_claim": novelty_claim,
            "method_core": method_core,
            "main_findings": main_findings,
            "contribution_type": contribution_type,
            "role_heuristic_note": note,
            "decomposition_method": "structured",
            "bg_sentence_count": bg_sents,
            "nov_sentence_count": nov_sents,
            "quality": "high",
        }

    # ── Cue-word classification ──
    total = len(sentences)
    labels = [sentence_role(s, i, total) for i, s in enumerate(sentences)]

    n_cue_matched = sum(1 for l in labels if l != "other")
    cue_ratio = n_cue_matched / total

    # ── Map ALL sentences to bg or nov (100% coverage) ──
    # Step 1: Find the boundary — first method or result sentence
    first_nov_idx = next(
        (i for i, l in enumerate(labels) if l in ("method", "result")),
        total,  # no method/result found → everything is bg
    )

    # Step 2: Assign every sentence to bg or nov
    bg_parts: list[str] = []
    nov_parts: list[str] = []
    for i, (sent, label) in enumerate(zip(sentences, labels)):
        if label == "background":
            bg_parts.append(sent)
        elif label in ("method", "result"):
            nov_parts.append(sent)
        else:  # "other" → position-based assignment
            if i < first_nov_idx:
                bg_parts.append(sent)  # before first method/result → bg
            else:
                nov_parts.append(sent)  # after → nov

    # Step 3: Ensure neither side is empty
    if not bg_parts:
        bg_parts = [sentences[0]]
    if not nov_parts:
        nov_parts = [sentences[-1]]

    # Step 4: Balance bg/nov ratio
    # 4a: If nov has only 1 sentence and bg has 3+, move last bg to nov
    #     unless it has strong background cues
    if len(nov_parts) == 1 and len(bg_parts) >= 3:
        candidate = bg_parts[-1].lower()
        has_strong_bg = any(cue in candidate for cue in (
            "however", "despite", "although", "challenge", "problem",
            "remains", "little is known", "poorly understood", "gap",
        ))
        if not has_strong_bg:
            nov_parts.insert(0, bg_parts.pop())

    # 4b: Cap bg at ~2/3 of total — if bg is excessively dominant,
    #     move trailing bg sentences to nov (they're likely transitional)
    max_bg = max(2, (total * 2) // 3)
    while len(bg_parts) > max_bg and len(bg_parts) > 2:
        nov_parts.insert(0, bg_parts.pop())

    background_summary = join_nonempty(bg_parts)
    nov_text = join_nonempty(nov_parts)

    # Extract specific role fields for downstream compatibility
    method_sents = [s for s, l in zip(sentences, labels) if l == "method"]
    result_sents = [s for s, l in zip(sentences, labels) if l == "result"]

    method_core = method_sents[0] if method_sents else nov_parts[0]
    main_findings = result_sents[0] if result_sents else nov_parts[-1]
    novelty_claim = method_sents[0] if method_sents else main_findings

    # Determine contribution type and note
    if method_sents and result_sents:
        contribution_type = "method_and_result"
        note = "cue_method_result"
    elif method_sents:
        contribution_type = "method"
        note = "cue_method_only"
    elif result_sents:
        contribution_type = "result"
        note = "cue_result_only"
    else:
        contribution_type = "positional"
        note = "positional_fallback"

    # Enrich note with coverage info
    role_dist = {r: labels.count(r) for r in ("background", "method", "result", "other")}
    note += f"|bg={role_dist['background']},m={role_dist['method']},r={role_dist['result']},o={role_dist['other']}"
    note += f"|cue_ratio={cue_ratio:.2f}"
    note += f"|bg_sents={len(bg_parts)},nov_sents={len(nov_parts)}"

    # Quality tier
    if cue_ratio >= 0.6 and len(bg_parts) >= 2 and len(nov_parts) >= 2:
        quality = "high"
    elif len(bg_parts) >= 1 and len(nov_parts) >= 1 and contribution_type != "positional":
        quality = "medium"
    else:
        quality = "low"
    note += f"|quality={quality}"

    return {
        "abstract": clean_abstract(abstract),
        "abstract_sentence_count": total,
        "background_summary": background_summary,
        "novelty_claim": novelty_claim,
        "method_core": method_core,
        "main_findings": main_findings,
        "contribution_type": contribution_type,
        "role_heuristic_note": note,
        "decomposition_method": "cue_word",
        "bg_sentence_count": len(bg_parts),
        "nov_sentence_count": len(nov_parts),
        "quality": quality,
    }
