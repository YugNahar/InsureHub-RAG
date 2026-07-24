"""
select_agent() — the single entry point for agent routing. Two-stage:
fast embedding-similarity match first, LLM fallback only for the
ambiguous band in between two thresholds.

Two thresholds, not one — deliberately. A single threshold would mean
every query that isn't confidently a specialist agent (i.e. nearly every
ordinary Layla/RAG query) pays an LLM round-trip on every turn. The low
floor lets the large majority of ordinary traffic skip the LLM call
entirely, matching the old regex's "free, instant, no-match-for-most-
queries" cost profile while still catching genuine ambiguous cases.

_HIGH_CONFIDENCE_THRESHOLD / _LOW_CONFIDENCE_FLOOR are placeholders —
calibrate from real data (score the known-agent example sets and a clean
control set spanning every existing policy type, see
contamination_corpus_runner.py for the established methodology this
project uses for exactly this kind of calibration) before relying on
these values, same discipline as this codebase's other thresholds
(_POINT_CONFIDENCE_FLOOR, insurer_confidence_threshold). Do NOT seed
these from multi_source_rag.py's 0.94/0.97 (near-duplicate query
matching) or metadata_tagger.py's 0.65 (keyword-hit confidence) — those
calibrate a different score distribution than short-query-vs-short-
description semantic similarity.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from . import embeddings, llm_fallback, registry

logger = logging.getLogger(__name__)

# Recalibrated 2026-07-24 (2nd pass) after narrowing Ava's scope from
# "travel insurance topic" to "travel insurance QUOTE/PURCHASE intent"
# (see agent_router/agents/ava.py's docstring — the topic-level scoping
# was routing plain informational questions like "What is travel
# insurance?" into the Ava handoff, which was wrong since only the
# quote/bind/issue flow is implemented). That narrowing broke the old
# 0.78/0.68 calibration: embeddings encode TOPIC, not speech-act, so
# "What is travel insurance?" (informational) and "get me a quote for my
# trip" (purchase) score nearly identically on pure topic similarity —
# real observed overlap: quote-intent phrases scored as low as 0.780,
# while informational travel-insurance questions scored up to 0.861,
# well inside the old high-confidence band.
#
# Refit against the same real-query discipline: quote-intent min 0.780
# still sets the floor a genuine quote request needs to even enter
# consideration, but the embedding-only HIGH threshold had to move up to
# 0.90 (clears every observed informational-question score, only lets
# through unambiguous quote phrasing like "sign me up for travel
# insurance" / "can I speak with Ava"). Everything between the two
# thresholds — which now includes EVERY informational travel-insurance
# question, not just adversarial edge cases — leans entirely on the LLM
# fallback's purchase-vs-informational-intent instruction (see
# llm_fallback.py) to do the real discrimination work. Confirmed live:
# 12/12 on a quote-vs-informational-vs-adversarial discrimination set
# after sharpening that prompt.
_HIGH_CONFIDENCE_THRESHOLD = 0.90  # clears every observed informational-question score
_LOW_CONFIDENCE_FLOOR = 0.68        # unchanged — still above ordinary clean-control's ceiling


@dataclass
class RoutingDecision:
    agent_name: Optional[str]
    confidence: float
    method: str  # "embedding" | "llm" | "none"


async def select_agent(query: str) -> RoutingDecision:
    scores = await embeddings.best_agent_similarity(query)
    if not scores:
        return RoutingDecision(None, 0.0, "none")

    best_name, best_score = scores[0]

    if best_score >= _HIGH_CONFIDENCE_THRESHOLD:
        logger.debug("[agent_router] embedding match: %r -> %r (%.3f)", query, best_name, best_score)
        return RoutingDecision(best_name, best_score, "embedding")

    if best_score < _LOW_CONFIDENCE_FLOOR:
        return RoutingDecision(None, best_score, "none")

    # Ambiguous band — only candidates that at least cleared the floor get
    # offered to the LLM, so it isn't asked to consider agents nothing
    # about the query resembled at all.
    candidate_names = [name for name, score in scores if score >= _LOW_CONFIDENCE_FLOOR]
    candidates = [a for a in (registry.get_agent(n) for n in candidate_names) if a is not None]
    llm_pick = await llm_fallback.classify_agent_llm(query, candidates)
    if llm_pick:
        logger.info("[agent_router] LLM fallback routed: %r -> %r (embedding best=%.3f)", query, llm_pick, best_score)
        return RoutingDecision(llm_pick, best_score, "llm")
    return RoutingDecision(None, best_score, "llm")
