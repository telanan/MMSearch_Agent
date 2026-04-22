"""
FactualVQA dataset loading and reward computation for MMSearch-R1 training.

Dataset: lmms-lab/FactualVQA
Each example has:
  - question: str
  - image: PIL.Image (or image_url: str)
  - answers: list[str]  (accepted answer strings)
  - search_results: dict  (cached web search results — used during training rollout)

Reward:
  - Exact match (EM) or substring match (SubEM) on final <answer>: +1.0
  - Per search tool call: -0.1 (search_penalty)
  - Format violation (no <answer> tag): -0.1 (format_penalty)
  - Final reward = answer_reward - search_penalty * num_searches - format_penalty
"""

import re
import string
from typing import Optional

from datasets import load_dataset, Dataset


# ─── Answer normalization ────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    return text


def exact_match(prediction: str, gold_answers: list[str]) -> bool:
    pred = _normalize(prediction)
    return any(_normalize(g) == pred for g in gold_answers)


def substring_match(prediction: str, gold_answers: list[str]) -> bool:
    pred = _normalize(prediction)
    return any(_normalize(g) in pred or pred in _normalize(g) for g in gold_answers)


# ─── Reward function ─────────────────────────────────────────────────────────

_PAT_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_PAT_IMAGE_SEARCH = re.compile(r"<search>", re.DOTALL)
_PAT_TEXT_SEARCH = re.compile(r"<text_search>", re.DOTALL)

SEARCH_PENALTY = 0.1
FORMAT_PENALTY = 0.1


def compute_reward(
    response: str,
    gold_answers: list[str],
    use_substring_match: bool = True,
) -> dict:
    """
    Compute MMSearch-R1 reward for a single response.

    Args:
        response: Full model response (may contain tool calls + final answer).
        gold_answers: List of acceptable answer strings.
        use_substring_match: Fall back to SubEM if EM fails.

    Returns:
        {
            "reward": float,
            "answer_reward": float,
            "search_penalty": float,
            "format_penalty": float,
            "predicted_answer": str,
            "em": bool,
            "subem": bool,
        }
    """
    answer_match = _PAT_ANSWER.search(response)
    format_penalty = 0.0 if answer_match else FORMAT_PENALTY

    predicted = answer_match.group(1).strip() if answer_match else ""

    em = exact_match(predicted, gold_answers)
    subem = substring_match(predicted, gold_answers) if not em else True
    answer_reward = 1.0 if (em or (use_substring_match and subem)) else 0.0

    num_image_searches = len(_PAT_IMAGE_SEARCH.findall(response))
    num_text_searches = len(_PAT_TEXT_SEARCH.findall(response))
    total_searches = num_image_searches + num_text_searches
    search_penalty = SEARCH_PENALTY * total_searches

    reward = answer_reward - search_penalty - format_penalty

    return {
        "reward": reward,
        "answer_reward": answer_reward,
        "search_penalty": search_penalty,
        "format_penalty": format_penalty,
        "predicted_answer": predicted,
        "em": em,
        "subem": subem,
    }


# ─── Dataset loading ─────────────────────────────────────────────────────────

def load_factualvqa(
    split: str = "train",
    cache_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> Dataset:
    """
    Load FactualVQA from HuggingFace.

    Args:
        split: "train" or "test"
        cache_dir: Local cache path (set to /root/autodl-tmp/datasets on AutoDL)
        max_samples: Truncate dataset for debugging

    Returns:
        HuggingFace Dataset with columns:
          question, image, answers, [cached_search_results]
    """
    ds = load_dataset(
        "lmms-lab/FactualVQA",
        split=split,
        cache_dir=cache_dir,
    )
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds


def collate_batch(examples: list[dict]) -> dict:
    """
    Collate a list of FactualVQA examples into training batch format
    expected by the veRL rollout engine.

    Returns dict with:
      - input_ids, attention_mask, pixel_values (model inputs)
      - gold_answers (list of list of str, for reward computation)
      - question_ids (for deduplication)
    """
    # veRL handles tokenization internally via the rollout worker;
    # here we just organize the raw fields.
    return {
        "questions": [ex["question"] for ex in examples],
        "images": [ex.get("image") for ex in examples],
        "gold_answers": [ex["answers"] for ex in examples],
        "cached_search_results": [ex.get("cached_search_results", {}) for ex in examples],
    }
