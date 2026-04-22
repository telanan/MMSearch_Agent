"""
Text search tool for MMSearch-R1.

Implements call_text_search() with the exact signature expected by the
MMSearch-R1 rollout engine (workers/multimodal/rollout/vllm_rollout_spmd.py).

Pipeline: SerpAPI web search → JINA Reader (full page text) → LLM summarization.

Required env vars:
    SERPAPI_KEY      — from https://serpapi.com
    JINA_API_KEY     — from https://jina.ai  (optional; free tier works without key)
    OPENAI_API_KEY   — any OpenAI-compatible key (optional; falls back to truncation)
    OPENAI_BASE_URL  — API base URL, e.g. https://xh.v1api.cc/v1 (optional)
    OPENAI_MODEL     — model name, e.g. deepseek-chat (optional, default: gpt-4o-mini)
"""

import os
import requests
from openai import OpenAI

SERPAPI_KEY    = os.environ.get("SERPAPI_KEY", "")
JINA_API_KEY   = os.environ.get("JINA_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "")
OPENAI_MODEL   = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

_JINA_HEADERS = {"Authorization": f"Bearer {JINA_API_KEY}"} if JINA_API_KEY else {}
_TIMEOUT = 20
_SUMMARIZER: OpenAI | None = None


def _get_summarizer() -> OpenAI | None:
    global _SUMMARIZER
    if _SUMMARIZER is None and OPENAI_API_KEY:
        kwargs = {"api_key": OPENAI_API_KEY}
        if OPENAI_BASE_URL:
            kwargs["base_url"] = OPENAI_BASE_URL
        _SUMMARIZER = OpenAI(**kwargs)
    return _SUMMARIZER


def _fetch_page(url: str) -> str:
    """Fetch full page text via JINA Reader."""
    try:
        r = requests.get(
            f"https://r.jina.ai/{url}",
            headers=_JINA_HEADERS,
            timeout=_TIMEOUT,
        )
        if r.status_code == 200:
            return r.text
    except Exception:
        pass
    return ""


def _summarize(query: str, text: str) -> str:
    """Summarize page text with Qwen, or truncate if no API key."""
    client = _get_summarizer()
    if client is None:
        return text[:800]
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Summarize the following web page content to answer the user query. "
                        "Be concise (under 200 words) and focus on factual information."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Query: {query}\n\nContent:\n{text[:10000]}",
                },
            ],
            max_tokens=300,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return text[:800]


def call_text_search(
    text_query: str,
    top_k: int = 3,
) -> tuple[str, dict]:
    """
    Web text search via SerpAPI + JINA Reader + Qwen summarization.

    Args:
        text_query: The search query string.
        top_k:      Max number of web pages to retrieve and summarize.

    Returns:
        formatted_text: String injected into <information>...</information>.
        meta:           {"success": bool, "num_results": int}
    """
    if not SERPAPI_KEY:
        return (
            "[Text Search] Error: SERPAPI_KEY not set.",
            {"success": False, "num_results": 0},
        )

    try:
        resp = requests.get(
            "https://serpapi.com/search",
            params={"engine": "google", "q": text_query, "num": top_k, "api_key": SERPAPI_KEY},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return (
            f"[Text Search] API error: {e}",
            {"success": False, "num_results": 0},
        )

    organic = data.get("organic_results", [])[:top_k]
    if not organic:
        return (
            "[Text Search] No results found.",
            {"success": False, "num_results": 0},
        )

    lines = ["[Text Search Results]"]
    for i, item in enumerate(organic, 1):
        title   = item.get("title", "No title")
        link    = item.get("link", "")
        snippet = item.get("snippet", "")

        page_text = _fetch_page(link) if link else ""
        summary   = _summarize(text_query, page_text or snippet)

        lines.append(f"\n{i}. {title}")
        if link:
            lines.append(f"   URL: {link}")
        lines.append(f"   {summary}")

    return (
        "\n".join(lines),
        {"success": True, "num_results": len(organic)},
    )
