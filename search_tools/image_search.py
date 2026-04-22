"""
Image search tool for MMSearch-R1.

Implements call_image_search() with the exact signature expected by the
MMSearch-R1 rollout engine (workers/multimodal/rollout/vllm_rollout_spmd.py).

Required env var:
    SERPAPI_KEY  — from https://serpapi.com (free tier: 100 searches/month)
"""

import os
import requests
from io import BytesIO
from PIL import Image

SERPAPI_KEY = os.environ.get("SERPAPI_KEY", "")
_TIMEOUT = 15


def _fetch_image(url: str) -> Image.Image | None:
    try:
        r = requests.get(url, timeout=_TIMEOUT, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        return Image.open(BytesIO(r.content)).convert("RGB")
    except Exception:
        return None


def call_image_search(
    image_url: str,
    top_k: int = 5,
) -> tuple[str, list[Image.Image], dict]:
    """
    Reverse image search via SerpAPI Google Lens.

    Args:
        image_url: URL of the query image.
        top_k:     Max number of visual matches to return.

    Returns:
        formatted_text: String injected into <information>...</information>.
        thumbnails:     List of PIL Images (thumbnail of each result).
        meta:           {"success": bool, "num_images": int}
    """
    if not SERPAPI_KEY:
        return (
            "[Image Search] Error: SERPAPI_KEY not set.",
            [],
            {"success": False, "num_images": 0},
        )

    try:
        resp = requests.get(
            "https://serpapi.com/search",
            params={"engine": "google_lens", "url": image_url, "api_key": SERPAPI_KEY},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return (
            f"[Image Search] API error: {e}",
            [],
            {"success": False, "num_images": 0},
        )

    matches = data.get("visual_matches", [])[:top_k]
    if not matches:
        return (
            "[Image Search] No visual matches found.",
            [],
            {"success": False, "num_images": 0},
        )

    thumbnails: list[Image.Image] = []
    lines = ["[Image Search Results]"]

    for i, item in enumerate(matches, 1):
        title = item.get("title", "No title")
        link = item.get("link", "")
        thumb_url = item.get("thumbnail", "")

        img = _fetch_image(thumb_url) if thumb_url else None
        if img:
            thumbnails.append(img)
            lines.append(f"{i}. <image_token> {title}")
        else:
            lines.append(f"{i}. {title}")

        if link:
            lines.append(f"   Source: {link}")

    return (
        "\n".join(lines),
        thumbnails,
        {"success": True, "num_images": len(thumbnails)},
    )
