"""
Agent registry — the extensibility point. Each specialist agent registers
itself once (see agents/ava.py for the pattern); nothing else in this
package or in api.py needs to change to add a new one.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class AgentDefinition:
    name: str
    """Canonical key, e.g. "ava" — matches ChatSession.active_agent values."""

    display_name: str
    """Shown to the user in the handoff offer, e.g. "Ava, our travel insurance specialist"."""

    intent_phrase: str
    """Short noun phrase for the generic offer template, e.g. "travel insurance"."""

    description: str
    """1-3 sentences. Shown to the LLM fallback classifier and embedded for
    similarity matching — this is the agent's "description" half of the
    description+examples routing signal."""

    example_queries: list[str] = field(default_factory=list)
    """Seed phrases a user might plausibly send this agent — the "examples"
    half of the routing signal. Each is embedded individually (not
    averaged into one centroid) so a short, distinctive example doesn't
    get diluted — see embeddings.py."""

    invoke: Optional[Callable[[str, str], Awaitable[str]]] = None
    """async (session_id, message) -> reply text. The actual dispatch
    target — today an in-process call into the agent's own handler
    (e.g. travel_bot's chat_endpoint); a future out-of-process agent's
    invoke would be a small async function making a real MCP client call
    instead, with the exact same signature."""

    enabled: bool = True


_REGISTRY: dict[str, AgentDefinition] = {}


def register_agent(defn: AgentDefinition) -> None:
    if defn.name in _REGISTRY:
        logger.warning("[agent_router] re-registering agent %r (was already registered)", defn.name)
    _REGISTRY[defn.name] = defn
    logger.info(
        "[agent_router] registered agent %r (%d example queries)",
        defn.name, len(defn.example_queries),
    )


def get_agent(name: str) -> Optional[AgentDefinition]:
    return _REGISTRY.get(name)


def all_agents() -> list[AgentDefinition]:
    """Only enabled agents — this is what routing/embedding logic should
    iterate over, not _REGISTRY directly."""
    return [a for a in _REGISTRY.values() if a.enabled]
