"""
LLM fallback for agent routing — only called for queries that land in the
ambiguous confidence band (see core.select_agent). Modeled directly on
multi_source_rag.py's _classify_query_policy_type_llm: same
_backend_completion() call, same cheap max_tokens/timeout budget, same
"first token, validated against a known set, safe default on anything
else" parsing discipline.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from .registry import AgentDefinition

logger = logging.getLogger(__name__)


async def classify_agent_llm(query: str, candidates: list[AgentDefinition]) -> Optional[str]:
    """
    Ask a fast/cheap LLM call to pick one of `candidates` (or "none") for
    this query. Returns the agent name, or None (= stay with Layla, the
    least disruptive default) on "none", any parse failure, timeout, or
    an answer that doesn't match a known candidate name.
    """
    if not candidates:
        return None
    try:
        from multi_source_rag import _backend_completion
    except Exception as exc:
        logger.debug("[agent_router] could not import _backend_completion: %s", exc)
        return None

    label_list = "\n".join(f"  - {a.name}: {a.description}" for a in candidates)
    prompt = f"""A user is chatting with Layla, a general insurance assistant. Decide whether
this message should be handed off to ONE specialist agent below, or Layla should keep it.

{label_list}
  - none: Layla should keep handling this herself

MESSAGE: {query}

Reply with ONLY the agent name or the word "none" — nothing else."""

    try:
        raw = await _backend_completion(prompt, max_tokens=10, timeout=3)
    except Exception as exc:
        logger.debug("[agent_router] LLM fallback call failed: %s", exc)
        return None
    if not raw:
        return None

    label = re.split(r"[\s\n,.:;()]", raw.strip().lower())[0]
    valid = {a.name for a in candidates}
    result = label if label in valid else None
    logger.debug("[agent_router] LLM fallback: %r -> %r (candidates=%s)", query, result, sorted(valid))
    return result
