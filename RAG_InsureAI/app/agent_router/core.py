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

# Calibrated 2026-07-24 against 25 real queries (8 known-travel phrases,
# 14 clean-control queries spanning every other existing policy type,
# 3 adversarial travel-adjacent-but-not-travel-insurance queries) — see
# agent_router's calibration notes / commit message for the raw numbers.
# Confirmed BGE-base similarity for short insurance-domain queries runs
# much higher across the board than multi_source_rag.py's near-duplicate
# thresholds (0.94/0.97) would suggest — even unrelated queries like
# "What is motor insurance?" scored 0.673 against Ava's description,
# since all insurance-domain text shares heavy vocabulary overlap. The
# real separation found: known-travel min 0.780, clean-control max
# 0.674 — a clean gap with no overlap in this sample. Adversarial
# queries (0.681-0.776) correctly fall in the ambiguous band between the
# two thresholds, which is exactly where they belong.
_HIGH_CONFIDENCE_THRESHOLD = 0.78  # at/above known-travel's observed floor
_LOW_CONFIDENCE_FLOOR = 0.68        # just above clean-control's observed ceiling (0.674)


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
