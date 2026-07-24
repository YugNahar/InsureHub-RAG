"""
Registers "ava" (the travel bot, app/travel_bot/) as a routable agent.

example_queries are unpacked from the regex alternatives the old
_TRAVEL_INTENT_PATTERN (formerly api.py) matched, so this at minimum
preserves today's recall — plus the whole point of embedding+LLM routing
is that new phrasings the regex never saw (e.g. "I'm flying to Bali next
week, do I need any insurance?") can still route correctly via the LLM
fallback without anyone hand-writing a new regex branch.
"""
from __future__ import annotations

import asyncio

from travel_bot.core.database import SessionLocal as _AvaSessionLocal
from travel_bot.routers.chat import chat_endpoint as _ava_chat_endpoint
from travel_bot.schemas.chat import ChatRequest as _AvaChatRequest

from ..registry import AgentDefinition, register_agent


def _call_sync(session_id: str, message: str) -> str:
    db = _AvaSessionLocal()
    try:
        resp = _ava_chat_endpoint(_AvaChatRequest(session_id=session_id, message=message), db)
        return resp.response
    finally:
        db.close()


async def _invoke(session_id: str, message: str) -> str:
    # chat_endpoint is sync (def, not async def) — offload it so an Ava
    # turn's DB + LLM latency doesn't block the whole event loop for
    # every other concurrent user, unlike the old inline call it replaces.
    return await asyncio.to_thread(_call_sync, session_id, message)


register_agent(AgentDefinition(
    name="ava",
    display_name="Ava, our travel insurance specialist",
    intent_phrase="travel insurance",
    description=(
        "Handles travel/trip insurance: quoting, binding, and issuing UAE "
        "travel-insurance policies for trips abroad, Schengen visas, Hajj/Umrah, "
        "backpacking, honeymoons, and similar travel scenarios."
    ),
    example_queries=[
        "travel insurance",
        "trip insurance",
        "travel cover",
        "insure my trip",
        "I need insurance for my trip",
        "insurance for my vacation",
        "insurance for a holiday",
        "going abroad",
        "travelling abroad next month",
        "Schengen visa insurance",
        "Hajj insurance",
        "Umrah insurance",
        "flight insurance",
        "visa insurance",
        "backpacking trip insurance",
        "honeymoon trip insurance",
        "holiday insurance",
        "can I talk to a travel specialist",
        "connect me with the travel agent",
        "can I speak with Ava",
    ],
    invoke=_invoke,
))
