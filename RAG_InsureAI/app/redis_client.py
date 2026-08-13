"""
Shared async Redis client + small per-key JSON helpers, used by every
session/conversation store this app persists — agent_hub.py's ChatSession
history, the /ask-stream multi-turn conversation memory, and
ConversationAgent's discovery/refinement state. Replaces those stores'
previous file-based persistence (sessions_data.json, conversations.json,
conversation_agent_sessions.json) — full-file rewrites on every single
chat turn didn't scale (conversations.json alone was 5.5MB across ~875
sessions before this), where a per-key Redis write touches only the one
session that actually changed.

One client, created lazily on first use so importing this module never
blocks or requires Redis to already be reachable — actual connection
errors surface from the first real command instead, matching how the
travel_bot SQLAlchemy engine (also created at import time, also lazy)
already behaves in this codebase.
"""
import json
import os
from typing import Any, List, Optional

import redis.asyncio as redis

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")

_client: Optional[redis.Redis] = None


def get_redis() -> redis.Redis:
    global _client
    if _client is None:
        _client = redis.from_url(REDIS_URL, decode_responses=True)
    return _client


async def get_json(key: str) -> Any:
    raw = await get_redis().get(key)
    return json.loads(raw) if raw is not None else None


async def set_json(key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
    payload = json.dumps(value, ensure_ascii=False)
    if ttl_seconds:
        await get_redis().set(key, payload, ex=ttl_seconds)
    else:
        await get_redis().set(key, payload)


async def delete_key(key: str) -> None:
    await get_redis().delete(key)


async def scan_keys(pattern: str) -> List[str]:
    """Every key matching *pattern* — used only for the handful of places
    that need to enumerate every session (startup load, dashboard
    listings), never on a hot per-request path."""
    client = get_redis()
    return [key async for key in client.scan_iter(match=pattern)]
