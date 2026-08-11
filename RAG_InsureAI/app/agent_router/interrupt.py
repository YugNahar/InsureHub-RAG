"""
Detects whether a NEW message, arriving while a session is already mid-
conversation with a specialist agent (active_agent != "layla"), should
interrupt that conversation and be answered by Layla instead.

Confirmed live: a user deep in Ava's travel-quote flow asked "What is
motor insurance?" and got Ava's own canned menu reply ("Please reply
with a number to select a plan...") instead of an actual answer — sticky
routing (agent_hub.py, unchanged) forwards every message to the active
agent with no way out except a manual super-admin override. Layla should
answer a genuine off-topic question instead. Critically, this must NOT
touch active_agent or tell the specialist bot anything happened — the
specialist conversation (its own state, stored in its own DB keyed by
session_id) is simply skipped for this one turn, so the NEXT message
that looks like a continuation (a plan number, a name, an email) resumes
exactly where it left off, with no special "resume" logic needed.

Same two-stage discipline as agent_router.core's own routing decision and
multi_source_rag.py's _resolve_modifier_intent: a fast, free regex catches
the overwhelming majority of real specialist-flow replies (numbers, yes/
no, contact info, short non-question phrases) with zero LLM cost; only
genuinely ambiguous longer/question-shaped input pays for a cheap LLM call.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Numbers, yes/no, and the specific quote-flow commands Ava's own reply
# text advertises ("reply with a number", "sort by price/insurer", "show
# add-ons for option N", "compare 1 and 2") — all near-certain continuations
# of an in-progress specialist conversation, never a fresh general question.
_LIKELY_CONTINUATION_RE = re.compile(
    r"^\s*(?:\d{1,2}|yes|yeah|yep|sure|ok(?:ay)?|no|nope|"
    r"compare\s+\d+\s+(?:and|&)\s+\d+|sort\s+by\s+\S+|"
    r"show\s+add-?ons?\s+for\s+(?:option\s+)?\d+)\s*[.!]?\s*$",
    re.IGNORECASE,
)
# Contact details (name/email/phone) are what a quote flow asks for next —
# never route these to an LLM call, and never treat them as an interruption.
_CONTACT_INFO_RE = re.compile(r"[\w.+-]+@[\w-]+\.[a-z]{2,}|\+?\d[\d\s-]{7,}\d", re.IGNORECASE)

# A WH-question mentioning insurance/policy/claim/cover in general terms is
# a strong, deterministic interrupt signal — confirmed live the LLM
# fallback alone is inconsistent on this exact shape ("What is motor
# insurance?" -> interrupt, "Does home insurance cover flood damage?" ->
# continue, same session, same classifier, no code change in between).
# Deliberately excludes "premium"/"quote"/"cost" — those can legitimately
# refer to the ACTIVE specialist conversation (e.g. "what's my premium
# going to be" mid-quote), so those stay on the LLM fallback path rather
# than a blanket deterministic interrupt.
#
# WH-word is NOT anchored to the start of the message — confirmed live: a
# user mid-Ava-offer (pending confirmation to connect to Ava for a Europe
# travel quote) sent "I am going to travel to europe so should i take the
# health insurance" and got stuck re-asking the same yes/no offer twice in
# a row. The anchored version only matches when the WH-word is literally
# the first word, so a real preamble before the actual question ("I am
# going to Europe, so should I...") fell through to the LLM fallback,
# which — for this exact case, at temperature=0 so consistently wrong both
# times — read it as continuing the Europe-travel offer rather than a
# distinct general question, likely because the topic (Europe travel)
# genuinely overlaps with what the pending offer itself is about, unlike
# the clearly-unrelated "What is motor insurance?" case this regex was
# originally built for. _LIKELY_CONTINUATION_RE and _CONTACT_INFO_RE above
# already short-circuit every known genuine continuation shape (numbers,
# yes/no, quote-flow commands, contact info) before this check ever runs,
# so dropping the start-anchor doesn't reopen that risk — it only widens
# which phrasing of an actual WH-question this deterministic path catches.
_GENERAL_QUESTION_RE = re.compile(
    r"\b(?:what|how|does|do|is|are|can|could|would|should|why|when|where|who|which)\b"
    r"[^.!]*\b(?:insurance|polic(?:y|ies)|claims?|cover(?:s|age|ed)?)\b",
    re.IGNORECASE,
)


async def is_interruption(message: str, active_agent_name: str) -> bool:
    """True if `message` should be answered by Layla instead of forwarded
    to the currently-active specialist agent. Safe default is False (stay
    with the specialist) on any ambiguity or failure — least disruptive,
    matches agent_router.core's own default-to-Layla-only-on-confidence
    philosophy, just inverted (here the "safe" choice is to NOT interrupt
    an in-progress task on a guess)."""
    text = message.strip()
    if not text:
        return False
    if _LIKELY_CONTINUATION_RE.match(text) or _CONTACT_INFO_RE.search(text):
        return False
    if _GENERAL_QUESTION_RE.search(text):
        return True
    # A short, question-mark-free phrase (<=4 words) reads far more like a
    # name ("John Smith") or a brief reply than a fresh question — skip the
    # LLM call for these too.
    if len(text.split()) <= 4 and "?" not in text:
        return False
    return await _classify_interruption_llm(text, active_agent_name)


async def _classify_interruption_llm(message: str, active_agent_name: str) -> bool:
    try:
        from multi_source_rag import _backend_completion
    except Exception as exc:
        logger.debug("[agent_router.interrupt] could not import _backend_completion: %s", exc)
        return False

    prompt = f"""A user is in the middle of a conversation with {active_agent_name}, a
specialist agent handling one specific task (e.g. quoting or purchasing a policy). They
just sent a NEW message. Decide: is this message CONTINUING that task — answering a
question {active_agent_name} asked (their name, contact details, a plan selection) or a
follow-up about that SAME quote/task — or is it a completely different, general question
they want the main assistant to answer instead?

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
