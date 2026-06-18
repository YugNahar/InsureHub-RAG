"""
Summary-level vector store: one entry per ingested document / video / webpage.

Used for two-stage retrieval:
  Stage 1 — query summaries  → identify the most relevant source documents
  Stage 2 — query chunks     → retrieve detailed text only from those sources

Summaries also act as a human-readable index: the frontend can display them
so users can see what's in the knowledge base at a glance.
"""
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List

from langchain_core.documents import Document
from turbovec_store import TurboVecStore

logger = logging.getLogger(__name__)

COLLECTION_NAME = os.getenv("TVEC_SUMMARIES_COLLECTION", "insurance_summaries")


class SummaryStore:
    """One summary Document per ingested source (file / video / webpage)."""

    def __init__(self):
        self._store = TurboVecStore(
            collection_name=COLLECTION_NAME,
            persist_subdir="summaries",
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def upsert(
        self,
        source: str,
        summary_text: str,
        metadata: Dict[str, Any],
    ) -> str:
        """Store (or replace) the summary for *source*. Returns assigned doc ID."""
        self.delete(source)
        doc = Document(
            page_content=summary_text,
            metadata={
                "source": source,
                "ingested_at": datetime.now(timezone.utc).isoformat(),
                **{k: v for k, v in metadata.items() if v is not None},
            },
        )
        ids = self._store.add_documents([doc])
        logger.info("[SummaryStore] upserted summary for source=%s", source)
        return ids[0] if ids else ""

    def delete(self, source: str) -> None:
        """Remove the summary for *source* (no-op if not found)."""
        self._store.delete_by_field("source", source)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> List[Document]:
        """Semantic search over all summaries."""
        return self._store.search(query, top_k=top_k, use_hybrid=True, use_reranker=False)

    def get_top_sources(self, query: str, top_k: int = 3) -> List[str]:
        """Return the *source* values of the top-k matching summaries."""
        docs = self.search(query, top_k=top_k)
        return [d.metadata.get("source", "") for d in docs if d.metadata.get("source")]

    def list_all(self) -> List[Dict[str, Any]]:
        """Return all summaries sorted newest-first."""
        results = []
        for doc_id, text in self._store._docs.items():
            meta = dict(self._store._metadatas.get(doc_id, {}))
            results.append({"text": text, "metadata": meta})
        results.sort(key=lambda x: x["metadata"].get("ingested_at", ""), reverse=True)
        return results

    def source_exists(self, source: str) -> bool:
        return source in self._store.list_values("source")

    def count(self) -> int:
        return self._store.count()
