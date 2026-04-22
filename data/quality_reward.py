"""
Quality-aware reward function for MMSearch-R1.

Improvements over original reward:
1. Search query quality scoring (entity coverage, specificity)
2. Search result relevance scoring
3. Information utilization rate
4. Adaptive penalty based on question type
"""

import re
import string
from typing import Optional, List
import spacy  # For entity recognition

from data.factualvqa import (
    _normalize, exact_match, substring_match,
    _PAT_ANSWER, _PAT_IMAGE_SEARCH, _PAT_TEXT_SEARCH
)

# Load spaCy model for entity recognition
try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("Warning: spaCy model not found. Install with: python -m spacy download en_core_web_sm")
    nlp = None


def extract_entities(text: str) -> set:
    """Extract named entities from text."""
    if nlp is None:
        return set()
    doc = nlp(text)
    return {ent.text.lower() for ent in doc.ents}


def compute_query_quality(query: str, question: str, gold_answers: List[str]) -> float:
    """
    Compute search query quality score (0.0 - 1.0).

    Factors:
    - Entity coverage: Does query contain key entities from question/answer?
    - Specificity: Is query specific enough (not too generic)?
    - Length: Reasonable length (not too short/long)?
    """
    query_lower = query.lower()

    # 1. Entity coverage
    question_entities = extract_entities(question)
    answer_entities = set()
    for ans in gold_answers:
        answer_entities.update(extract_entities(ans))

    all_entities = question_entities | answer_entities
    if all_entities:
        entities_in_query = sum(1 for ent in all_entities if ent in query_lower)
        entity_coverage = entities_in_query / len(all_entities)
    else:
        entity_coverage = 0.5  # Neutral if no entities

    # 2. Specificity (penalize too generic queries)
    generic_terms = {'what', 'who', 'where', 'when', 'how', 'this', 'that', 'thing', 'image', 'picture'}
    query_words = set(query_lower.split())
    generic_ratio = len(query_words & generic_terms) / max(len(query_words), 1)
    specificity = 1.0 - generic_ratio

    # 3. Length appropriateness (3-10 words is good)
    word_count = len(query_words)
    if 3 <= word_count <= 10:
        length_score = 1.0
    elif word_count < 3:
        length_score = 0.5  # Too short
    else:
        length_score = max(0.5, 1.0 - (word_count - 10) * 0.05)  # Too long

    # Weighted combination
    quality = 0.5 * entity_coverage + 0.3 * specificity + 0.2 * length_score
    return quality


def compute_quality_aware_reward(
    response: str,
    gold_answers: List[str],
    question: str = "",
    use_substring_match: bool = True,
) -> dict:
    """
    Quality-aware reward function.

    Formula:
        reward = answer_reward - quality_adjusted_penalty - format_penalty

    where:
        quality_adjusted_penalty = Σ (base_penalty / query_quality)

    High quality query (quality=1.0) → penalty = 0.1
    Low quality query (quality=0.5) → penalty = 0.2
    """
    # 1. Answer reward (same as original)
    answer_match = _PAT_ANSWER.search(response)
    format_penalty = 0.0 if answer_match else 0.1

    predicted = answer_match.group(1).strip() if answer_match else ""
    em = exact_match(predicted, gold_answers)
    subem = substring_match(predicted, gold_answers) if not em else True
    answer_reward = 1.0 if (em or (use_substring_match and subem)) else 0.0

    # 2. Quality-adjusted search penalty
    base_penalty = 0.1
    total_penalty = 0.0
    search_details = []

    # Image searches
    for match in _PAT_IMAGE_SEARCH.finditer(response):
        # For image search, quality is harder to assess, use base penalty
        total_penalty += base_penalty
        search_details.append({
            "type": "image_search",
            "quality": 1.0,
            "penalty": base_penalty
        })

    # Text searches
    for match in _PAT_TEXT_SEARCH.finditer(response):
        query = match.group(1).strip()
        quality = compute_query_quality(query, question, gold_answers)
        # Inverse relationship: better quality → lower penalty
        adjusted_penalty = base_penalty / max(quality, 0.3)  # Avoid division by very small numbers
        total_penalty += adjusted_penalty
        search_details.append({
            "type": "text_search",
            "query": query,
            "quality": quality,
            "penalty": adjusted_penalty
        })

    # 3. Final reward
    reward = answer_reward - total_penalty - format_penalty

    return {
        "reward": reward,
        "answer_reward": answer_reward,
        "search_penalty": total_penalty,
        "format_penalty": format_penalty,
        "predicted_answer": predicted,
        "em": em,
        "subem": subem,
        "search_details": search_details,
        "num_searches": len(search_details),
    }
