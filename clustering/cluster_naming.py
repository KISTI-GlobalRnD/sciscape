"""Cluster naming utilities powered by a local LLM."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence
import json

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore

from .core_documents import ClusterDocument

DEFAULT_MODEL = "gpt-oss:20b"
DEFAULT_BASE_URL = "http://172.16.2.42:11434/v1"
DEFAULT_API_KEY = "ollama"
DEFAULT_MAX_DOCUMENTS = 8

ENV_BASE_URL = "OLLAMA_BASE_URL"
ENV_API_KEY = "OLLAMA_API_KEY"
ENV_MODEL = "OLLAMA_MODEL"
ENV_CONFIG_PATH = "OLLAMA_CONFIG"

LANGUAGE_PROMPT = """
You are a language detection and translation assistant.

Rules:
1. Detect the input language and return its ISO 639-1 code.
2. If the text is already in English:
   - Return {"lang": "en", "text": ""} (do not copy the original text).
3. If the text is not English:
   - Translate it into English.
   - Return {"lang": "<ISO-639-1 code>", "text": "<English translation>"}.
4. Output ONLY valid JSON.
""".strip()

SUMMARISER_SYSTEM_PROMPT = """
You are a scientific topic summariser. You will be given representative documents (title and abstract) for a community of papers. Produce a concise cluster name and description that would help a researcher understand the theme.

Return ONLY valid JSON with the following schema:
{
  "name": string,        # <= 120 characters, title case, no quotes
  "description": string, # 2-3 sentences (< 400 characters total)
  "keywords": [string],  # 3-6 distinctive keywords or phrases
  "sources": [string],   # UIDs of the documents you actually used
  "notes": string        # optional short remark about translation or coverage (may be empty)
}
""".strip()


@dataclass
class ClusterSummary:
    """Structured output for a cluster name/description."""

    cluster_label: str
    name: str
    description: str
    keywords: List[str]
    sources: List[str]
    notes: str
    raw_response: str


def _load_env_file(env_path: Path) -> Dict[str, str]:
    data: Dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip()
    return data


def _determine_settings(
    *,
    base_url: Optional[str],
    api_key: Optional[str],
    model: Optional[str],
) -> Dict[str, str]:
    from os import environ

    config: Dict[str, str] = {}

    env_path = environ.get(ENV_CONFIG_PATH)
    if env_path:
        path = Path(env_path)
        if path.exists():
            config.update(_load_env_file(path))

    config.setdefault("base_url", environ.get(ENV_BASE_URL, DEFAULT_BASE_URL))
    config.setdefault("api_key", environ.get(ENV_API_KEY, DEFAULT_API_KEY))
    config.setdefault("model", environ.get(ENV_MODEL, DEFAULT_MODEL))

    if base_url:
        config["base_url"] = base_url
    if api_key:
        config["api_key"] = api_key
    if model:
        config["model"] = model

    return config


@dataclass
class ClusterDBConfig:
    """Database configuration for extracting representative documents."""

    db_path: Optional[str] = None
    db_name: Optional[str] = None  # reserved for future use (e.g., DSN)
    meta_table: str = "paper_metadata"
    metric_table: Optional[str] = None
    uid_col: str = "uid"
    year_col: str = "pubyear"
    citation_col: str = "citation_count"


def create_client(
    *,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> "OpenAI":
    """Instantiate an OpenAI client targeting the local LLM endpoint.

    Configuration precedence:
        1. Keyword arguments
        2. Variables from an optional `.env` file (path via ``OLLAMA_CONFIG``)
        3. Environment variables ``OLLAMA_BASE_URL`` and ``OLLAMA_API_KEY``
        4. Built-in defaults
    """

    if OpenAI is None:  # pragma: no cover - optional dependency path
        raise ImportError("openai package is required for cluster naming")

    settings = _determine_settings(base_url=base_url, api_key=api_key, model=model)
    client = OpenAI(base_url=settings["base_url"], api_key=settings["api_key"])
    client._leiden_module_model = settings["model"]  # type: ignore[attr-defined]
    return client


def _call_chat_completion(
    client: "OpenAI",
    *,
    model: str,
    system_prompt: str,
    user_content: str,
) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )
    return response.choices[0].message.content.strip()


def detect_and_translate(
    client: "OpenAI",
    text: str,
    *,
    model: Optional[str] = None,
) -> Dict[str, str]:
    """Detect the language of *text* and translate to English if needed."""

    if not text.strip():
        return {"lang": "", "text": ""}

    effective_model = model or getattr(client, "_leiden_module_model", DEFAULT_MODEL)

    raw = _call_chat_completion(
        client,
        model=effective_model,
        system_prompt=LANGUAGE_PROMPT,
        user_content=text,
    )

    for _ in range(3):
        try:
            result = json.loads(raw)
            break
        except json.JSONDecodeError:
            raw = raw.strip()
    else:
        return {"lang": "Invalid JSON", "text": raw}

    if result.get("lang") == "en" and not result.get("text"):
        result["text"] = text

    return result


def _prepare_document_text(
    client: "OpenAI",
    document: ClusterDocument,
    *,
    model: Optional[str],
) -> Dict[str, str]:
    merged = f"Title: {document.title}\nAbstract: {document.abstract}".strip()
    translation = detect_and_translate(client, merged, model=model)
    return {
        "uid": document.uid,
        "lang": translation.get("lang", ""),
        "text": translation.get("text", merged),
    }


def _build_cluster_user_prompt(
    cluster_label: str,
    documents: Sequence[Dict[str, str]],
) -> str:
    lines = [f"Cluster: {cluster_label}", "Representative documents:"]
    for idx, doc in enumerate(documents, start=1):
        text = doc.get("text", "").strip()
        lines.append(
            f"\nDocument {idx} (UID: {doc['uid']}, lang: {doc.get('lang', '')}):\n{text}"
        )
    lines.append(
        "\nProvide a concise name, description, keywords, and list the UIDs you relied on."
    )
    return "\n".join(lines)


def summarise_cluster(
    client: "OpenAI",
    cluster_label: str,
    documents: Sequence[ClusterDocument],
    *,
    model: str = DEFAULT_MODEL,
    max_documents: int = DEFAULT_MAX_DOCUMENTS,
) -> ClusterSummary:
    """Generate a cluster summary using the local LLM."""

    if not documents:
        raise ValueError("documents must contain at least one entry")

    selected = list(documents)[:max_documents]
    translated_docs = [
        _prepare_document_text(client, doc, model=model)
        for doc in selected
    ]

    user_prompt = _build_cluster_user_prompt(cluster_label, translated_docs)
    raw = _call_chat_completion(
        client,
        model=model or getattr(client, "_leiden_module_model", DEFAULT_MODEL),
        system_prompt=SUMMARISER_SYSTEM_PROMPT,
        user_content=user_prompt,
    )

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        raise ValueError(
            f"Model returned invalid JSON for cluster {cluster_label}: {raw}"
        ) from None

    keywords = parsed.get("keywords") or []
    if isinstance(keywords, str):
        keywords = [kw.strip() for kw in keywords.split(",") if kw.strip()]

    sources = parsed.get("sources") or []
    if isinstance(sources, str):
        sources = [uid.strip() for uid in sources.split(",") if uid.strip()]

    return ClusterSummary(
        cluster_label=cluster_label,
        name=str(parsed.get("name", "")).strip(),
        description=str(parsed.get("description", "")).strip(),
        keywords=[str(kw).strip() for kw in keywords],
        sources=[str(uid).strip() for uid in sources],
        notes=str(parsed.get("notes", "")).strip(),
        raw_response=raw,
    )


def summarise_clusters(
    client: "OpenAI",
    cluster_documents: Mapping[str, Iterable[ClusterDocument]],
    *,
    model: Optional[str] = None,
    max_documents: int = DEFAULT_MAX_DOCUMENTS,
) -> List[ClusterSummary]:
    """Generate summaries for multiple clusters.

    Parameters
    ----------
    client:
        OpenAI client configured for the local model.
    cluster_documents:
        Mapping from cluster label to an iterable of :class:`ClusterDocument`.
    model:
        Model identifier, defaults to ``gpt-oss:20b``.
    max_documents:
        Cap on how many representative documents to feed into the prompt per cluster.
    """

    summaries: List[ClusterSummary] = []
    for cluster_label, docs in cluster_documents.items():
        documents = list(docs)
        if not documents:
            continue
        summary = summarise_cluster(
            client,
            cluster_label,
            documents,
            model=model,
            max_documents=max_documents,
        )
        summaries.append(summary)
    return summaries


__all__ = [
    "ClusterDocument",
    "ClusterSummary",
    "create_client",
    "detect_and_translate",
    "summarise_cluster",
    "summarise_clusters",
]
