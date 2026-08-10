"""
Registers "ava" (the travel bot, app/travel_bot/) as a routable agent.

Scope is deliberately narrow: QUOTE/PURCHASE intent only ("get me a quote
for my trip to India", "what will the premium be", "I want to buy travel
insurance"), not general "what is travel insurance / what does it cover"
questions — those are ordinary RAG knowledge-base questions the KB
already answers directly (confirmed live: the KB has real travel-
insurance content). Only the quote/bind/issue flow is implemented today
(app/travel_bot/), so only quote-intent queries should hand off to it;
routing every travel-*topic* mention to Ava would hijack legitimate RAG
questions the moment "travel insurance" appears in them. See
agent_routing_corpus.json's "travel_info_not_ava" set — these must NEVER
trigger the offer, exactly the failure this scoping fixes.
"""
from __future__ import annotations

import asyncio

from travel_bot.core.database import SessionLocal as _AvaSessionLocal
from travel_bot.routers.chat import (
    chat_endpoint as _ava_chat_endpoint,
    get_or_create_session as _ava_get_or_create_session,
    FIELD_OPTIONS_MAP as _AVA_FIELD_OPTIONS_MAP,
)
from travel_bot.schemas.chat import ChatRequest as _AvaChatRequest
from travel_bot.schemas.travel import COVERAGE_TYPE_LOCKS as _AVA_COVERAGE_TYPE_LOCKS
from travel_bot.models.message import Message as _AvaMessage

from ..registry import AgentDefinition, register_agent

_INTAKE_FORM_WELCOME = (
    "Sure — let's get you a quote. Fill in your details below, or upload a "
    "document and I'll pre-fill what I can."
)


def _call_sync(session_id: str, message: str) -> dict:
    db = _AvaSessionLocal()
    try:
        # api.py calls invoke(session_id, "") exactly once, right when a
        # handoff is accepted, to produce Ava's opening turn — the
        # established convention for "this is the start of the
        # conversation" (see api.py's _handoff_yes_gen). Short-circuit
        # chat_endpoint's own QUOTING-phase question-asking here rather
        # than teaching the phase machine itself about a "first turn"
        # special case: this is the one feature this whole change exists
        # to replace, so it's scoped to this one bridge function, not
        # chat_endpoint (which every other turn — including a user who
        # free-types instead of using the form — still goes through
        # completely unmodified).
        if message == "":
            db_session = _ava_get_or_create_session(db, session_id)
            state = db_session.extracted_data or {}
            db.add(_AvaMessage(session_id=session_id, role="assistant", content=_INTAKE_FORM_WELCOME))
            db.commit()
            return {
                "text": _INTAKE_FORM_WELCOME,
                "ui": {
                    "type": "ava_form",
                    "field_options": _AVA_FIELD_OPTIONS_MAP,
                    "coverage_type_locks": _AVA_COVERAGE_TYPE_LOCKS,
                },
                "collected_fields": state,
            }

        resp = _ava_chat_endpoint(_AvaChatRequest(session_id=session_id, message=message), db)
        return {"text": resp.response, "ui": resp.ui, "collected_fields": resp.collected_fields}
    finally:
        db.close()


async def _invoke(session_id: str, message: str) -> dict:
    # chat_endpoint is sync (def, not async def) — offload it so an Ava
    # turn's DB + LLM latency doesn't block the whole event loop for
    # every other concurrent user, unlike the old inline call it replaces.
    return await asyncio.to_thread(_call_sync, session_id, message)


register_agent(AgentDefinition(
    name="ava",
    display_name="Ava, our travel insurance specialist",
    intent_phrase="a travel insurance quote",
    description=(
        "Handles GETTING A QUOTE, BUYING, or PURCHASING travel insurance for a "
        "specific trip — quoting, binding, and issuing UAE travel-insurance "
        "policies (trips abroad, Schengen visas, Hajj/Umrah, backpacking, "
        "honeymoons). Only for quote/purchase requests, e.g. 'get me a quote "
        "for my trip to India', 'what will the premium be if I'm going to "
        "Dubai', 'I want to buy travel insurance'. Does NOT cover general "
        "informational questions about what travel insurance is or what it "
        "covers — those are answered directly from the knowledge base, not by "
        "this agent."
    ),
    example_queries=[
        "can you get me a quote for my travel to India",
        "can you help me with a travel quote",
        "what will be the premium if I am going to India",
        "what will be the premium if I am going to Dubai",
        "I want to buy travel insurance for my trip",
        "get me a travel insurance quote for my trip to the UK",
        "how much will travel insurance cost for my trip to Thailand",
        "I need to purchase travel insurance for my upcoming trip",
        "quote me travel insurance for my Schengen visa",
        "I want to buy Hajj insurance",
        "I want to buy Umrah insurance",
        "give me a price for travel insurance to Europe",
        "book travel insurance for my honeymoon",
        "sign me up for travel insurance",
        "I'd like to purchase a travel insurance policy",
        "issue me a travel insurance policy for my backpacking trip",
        "can I talk to a travel specialist",
        "connect me with the travel agent",
        "can I speak with Ava",
    ],
    invoke=_invoke,
))
