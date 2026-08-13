"""
Detects whether a NEW message, arriving while a session is already mid-
conversation with a specialist agent (active_agent != "layla") or waiting
on a pending handoff confirmation, should interrupt/skip that and be
answered by Layla instead.

Confirmed live, repeatedly, across several completely different phrasings
of the same underlying failure: a user mid-Ava-flow (or mid a pending
Ava-offer) asks something that has nothing to do with Ava, and gets
Ava's own canned reply or the same "connect you with Ava? (yes/no)"
re-ask instead of an actual answer — "What is motor insurance?", "I am
going to travel to europe so should i take the health insurance",
"What is the difference between premium and deductible?", "Can you
explain it in detail" (a follow-up to an unrelated crop-insurance
question). Each of those was originally "fixed" by adding ONE MORE
regex pattern for that exact phrasing (_GENERAL_QUESTION_RE, then a
dropped start-anchor, then _COMPARISON_DEFINITION_RE) — a whack-a-mole
pattern that predictably kept finding new gaps, because a fixed keyword/
phrase list can never anticipate every way of asking an unrelated
question. Rebuilt (2026-08-13) on the SAME description+examples routing
signal agent_router.core.select_agent() already uses for the OPPOSITE
decision (should a NEW message route TO an agent) — an agent's own
registry.AgentDefinition already carries a description and example
queries specifically so a generic classifier can compare a message
against "what this agent actually handles" instead of guessing from
scratch each time. This module reuses that same signal, and the same
LLM-fallback framing (name + real description, not a bare display-name
string), for the "should this message LEAVE the agent" decision instead.

Critically, an interruption must NOT touch active_agent or
awaiting_agent_confirmation or tell the specialist bot anything happened
— the specialist conversation (its own state, stored in its own DB
keyed by session_id, or the pending-offer flag) is simply skipped for
this one turn, so the NEXT message that looks like a continuation (a
plan number, a name, an email, a literal yes/no) resumes exactly where
it left off, with no special "resume" logic needed.

Same two-stage discipline as agent_router.core's own routing decision and
multi_source_rag.py's _resolve_modifier_intent: a fast, free regex catches
the overwhelming majority of real specialist-flow replies (numbers, yes/
no, contact info) with zero cost; embedding similarity against the active
agent's OWN description/examples resolves the large majority of what's
left, for free; only a genuinely ambiguous middle band pays for a cheap
LLM call.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from . import embeddings, registry

logger = logging.getLogger(__name__)

# Numbers, yes/no, and the specific quote-flow commands Ava's own reply
# text advertises ("reply with a number", "sort by price/insurer", "show
# add-ons for option N", "compare 1 and 2") — all near-certain continuations
# of an in-progress specialist conversation, never a fresh general question.
# Deliberately kept as a regex, not folded into the embedding signal below:
# these are near-zero-ambiguity STRUCTURAL shapes (a bare number, a bare
# yes/no) that don't carry enough semantic content for an embedding
# comparison to be meaningful, not a topic-guessing problem.
_LIKELY_CONTINUATION_RE = re.compile(
    r"^\s*(?:\d{1,2}|yes|yeah|yep|sure|ok(?:ay)?|no|nope|"
    r"compare\s+\d+\s+(?:and|&)\s+\d+|sort\s+by\s+\S+|"
    r"show\s+add-?ons?\s+for\s+(?:option\s+)?\d+)\s*[.!]?\s*$",
    re.IGNORECASE,
)
# Contact details (name/email/phone) are what a quote flow asks for next —
# never route these to an LLM call, and never treat them as an interruption.
_CONTACT_INFO_RE = re.compile(r"[\w.+-]+@[\w-]+\.[a-z]{2,}|\+?\d[\d\s-]{7,}\d", re.IGNORECASE)

# Calibrated live (2026-08-13) against the agent's real registered
# description + example_queries (see agents/ava.py) — the same embedder
# and the same _AGENT_DOC_VECTORS cache agent_router.core.select_agent()
# already warms, just scored against ONE agent instead of the max over
# all of them (embeddings.agent_similarity). Measured scores told a
# different story than expected: there is NO clean low-vs-high split.
# Genuine should-CONTINUE in-flow follow-ups ("what does group cover
# mean" 0.554, "what's the difference between individual and family
# cover" 0.583, "how much will it cost for a family of 4" 0.599) land
# in the SAME 0.54-0.67 band as confirmed should-INTERRUPT cases ("Can
# you explain it in detail" after crop insurance 0.543, "What is the
# difference between premium and deductible?" 0.632, "What is motor
# insurance?" 0.633) — same root cause agent_router.core's own
# calibration comment already documents for the opposite decision:
# embeddings encode TOPIC/domain ("this is generically insurance-
# related"), not the specific purchase-intent-vs-informational
# distinction that actually matters here. Only a genuinely high score
# reliably means "confidently still in Ava's own domain" — an explicit
# quote/purchase restatement scored 0.80-1.0 in every case tested.
# Below that, score alone can't be trusted either way, so everything
# else goes to the LLM fallback rather than an unsafe low-floor auto-
# interrupt that would have wrongly caught "what does group cover mean"
# (0.554) right alongside genuine interruptions at a similar score.
_HIGH_CONTINUATION_THRESHOLD = 0.80  # >= this: confidently still relevant, skip the LLM call


async def is_interruption(message: str, agent_name: str) -> bool:
    """True if `message` should be answered by Layla instead of forwarded
    to `agent_name` (the active agent's REGISTRY key, e.g. "ava" — matches
    ChatSession.active_agent / AgentHubSession.awaiting_agent_confirmation
    values, NOT the human-readable display_name). Safe default is False
    (stay with the specialist) on any ambiguity or failure — least
    disruptive, matches agent_router.core's own default-to-Layla-only-on-
    confidence philosophy, just inverted (here the "safe" choice is to NOT
    interrupt an in-progress task on a guess)."""
    text = message.strip()
    if not text:
        return False
    if _LIKELY_CONTINUATION_RE.match(text) or _CONTACT_INFO_RE.search(text):
        return False

    score = await embeddings.agent_similarity(text, agent_name)
    if score is not None and score >= _HIGH_CONTINUATION_THRESHOLD:
        logger.debug("[agent_router.interrupt] embedding: %r -> continue (%.3f)", text, score)
        return False

    # A short, question-mark-free phrase (<=4 words) reads far more like a
    # name ("John Smith") or a brief reply than a fresh question — skip the
    # LLM call for these too.
    if len(text.split()) <= 4 and "?" not in text:
        return False
    return await _classify_interruption_llm(text, agent_name)


async def _classify_interruption_llm(message: str, agent_name: str) -> bool:
    agent = registry.get_agent(agent_name)
    # Same framing as agent_router.llm_fallback.classify_agent_llm — name +
    # real registered description, not a bare display-name string, so this
    # LLM call gets the actual "what this agent does" signal instead of
    # having to infer it from a human-readable label alone.
    label = f"{agent.name}: {agent.description}" if agent else (agent_name or "the specialist agent")
    display = agent.display_name if agent else agent_name

    try:
        from multi_source_rag import _backend_completion
    except Exception as exc:
        logger.debug("[agent_router.interrupt] could not import _backend_completion: %s", exc)
        return False

    prompt = f"""A user is in the middle of a conversation with a specialist agent:

  {label}

They just sent a NEW message. Decide: is this message CONTINUING that task — answering a
question {display} asked (their name, contact details, a plan selection), a follow-up
about that SAME quote/task, or a question about the agent's own process (a field it's
asking for, a term specific to filling in its form) — or is it a completely different,
general question, or a request about a DIFFERENT insurance type/topic, that the main
assistant should answer instead?

MESSAGE: {message}

Reply with ONLY "continue" or "interrupt" — nothing else."""

    try:
        raw = await _backend_completion(prompt, max_tokens=5, timeout=5, temperature=0)
    except Exception as exc:
        logger.debug("[agent_router.interrupt] classification call failed: %s", exc)
        return False
    if not raw:
        return False
    result = raw.strip().lower().startswith("interrupt")
    logger.debug("[agent_router.interrupt] %r -> %s", message, "interrupt" if result else "continue")
    return result
