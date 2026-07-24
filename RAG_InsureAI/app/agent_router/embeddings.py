"""
Embedding-similarity routing signal — the fast first stage of
core.select_agent(). Reuses the same process-wide embedder every other
part of this app uses for retrieval (turbovec_store._get_shared_embed_model),
not a second model, and the exact encode(...) call shape already
established at multi_source_rag.py's semantic-cache lookup (normalized
embeddings, so cosine similarity is a plain dot product).
"""
from __future__ import annotations

import asyncio
import logging
import threading

import numpy as np

from turbovec_store import EMBED_MODEL_NAME, _get_shared_embed_model
from . import registry

logger = logging.getLogger(__name__)

# agent_name -> (doc_texts: list[str], doc_vectors: np.ndarray of shape (n_docs, dim))
_AGENT_DOC_VECTORS: dict[str, tuple[list[str], np.ndarray]] = {}
_WARM_LOCK = threading.Lock()
_WARMED = False


async def warm_agent_embeddings(force: bool = False) -> None:
    """
    Precompute embeddings for every registered agent's description + each
    example query. Idempotent by default (force=False skips work if
    already warmed) — safe to call from api.py's own startup handler AND
    as a lazy safety net inside best_agent_similarity() for any process
    that imports this module without running that handler (a standalone
    script, the stdio MCP server, a future test harness).
    """
    global _WARMED
    if _WARMED and not force:
        return
    with _WARM_LOCK:
        if _WARMED and not force:
            return
        agents = registry.all_agents()
        if not agents:
            logger.warning("[agent_router] warm_agent_embeddings called with zero registered agents")
            _WARMED = True
            return
        model = _get_shared_embed_model(EMBED_MODEL_NAME)
        new_vectors: dict[str, tuple[list[str], np.ndarray]] = {}
        for agent in agents:
            docs = [agent.description] + list(agent.example_queries)
            docs = [d for d in docs if d and d.strip()]
            if not docs:
                logger.warning("[agent_router] agent %r has no description/examples to embed", agent.name)
                continue
            vectors = await asyncio.to_thread(
                lambda docs=docs: model.encode(docs, normalize_embeddings=True, show_progress_bar=False)
            )
            new_vectors[agent.name] = (docs, np.asarray(vectors))
        _AGENT_DOC_VECTORS.clear()
        _AGENT_DOC_VECTORS.update(new_vectors)
        _WARMED = True
        logger.info(
            "[agent_router] warmed embeddings for %d agent(s): %s",
            len(new_vectors), {name: len(docs) for name, (docs, _) in new_vectors.items()},
        )


async def best_agent_similarity(query: str) -> list[tuple[str, float]]:
    """
    [(agent_name, max_cosine_similarity_over_its_docs), ...] sorted
    best-first across all registered agents. Max-over-documents, not an
    averaged centroid — a short, distinctive example query shouldn't get
    diluted by averaging it against a longer description, and this is the
    closer analog to what the old regex's OR-list of literal phrases
    already did (any one match was enough).
    """
    if not _WARMED:
        await warm_agent_embeddings()
    if not _AGENT_DOC_VECTORS:
        return []
    model = _get_shared_embed_model(EMBED_MODEL_NAME)
    query_vec = await asyncio.to_thread(
        lambda: model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
    )
    scores: list[tuple[str, float]] = []
    for agent_name, (_docs, doc_vectors) in _AGENT_DOC_VECTORS.items():
        sims = doc_vectors @ query_vec  # normalized vectors -> dot product == cosine similarity
        scores.append((agent_name, float(sims.max())))
    scores.sort(key=lambda pair: pair[1], reverse=True)
    return scores
