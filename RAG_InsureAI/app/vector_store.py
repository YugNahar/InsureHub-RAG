"""
TurboVec-backed Vector Store — replaces ChromaDB with TurboQuantIndex for
ultra-fast ANN retrieval (4-bit quantization, ~4GB RAM, no GPU).

Falls back to ChromaDB when turbovec is not installed.
"""
import os
import logging
from typing import List, Optional, Dict, Any

from langchain_core.documents import Document

from turbovec_store import TurboVecStore, _ensure_turbovec

logger = logging.getLogger(__name__)

COLLECTION_NAME = os.getenv("TVEC_DOCS_COLLECTION", "insurance_docs")


class ChromaVectorStore:
    """
    Persistent vector store with:
      - Dense retrieval via TurboVec (TurboQuantIndex with 4-bit quantization)
      - Keyword retrieval (BM25)
      - Cross-encoder reranking

    This class retains the original name for backward compatibility with
    existing imports in rag.py, but is now backed by TurboVec rather than
    ChromaDB when available.
    """

    def __init__(self):
        self._store = TurboVecStore(
            collection_name=COLLECTION_NAME,
            persist_subdir="documents",
        )
        # Expose collection-like API for code that accessed `self.collection` directly
        self.collection = CollectionCompat(self._store)
        self.embed_model = self._store.embed_model  # keep accessible

        logger.info(
            "ChromaVectorStore (TurboVec-backed) ready — collection=%s, chunks=%d",
            COLLECTION_NAME, self._store.count(),
        )

    # ── Delegate everything to TurboVecStore ───────────────────────────────────

    def add_documents(self, docs: List[Document]) -> List[str]:
        return self._store.add_documents(docs)

    def delete_by_source(self, source: str):
        self._store.delete_by_field("source", source)

    def delete_all(self):
        self._store.delete_all()

    def list_sources(self) -> List[str]:
        return self._store.list_values("source")

    def list_filenames(self) -> List[str]:
        return self._store.list_values("filename")

    def list_values(self, field: str) -> List[str]:
        return self._store.list_values(field)

    def warmup(self) -> None:
        self._store.warmup()

    def rerank_documents(self, query: str, docs, top_k: int):
        return self._store.rerank_documents(query, docs, top_k)

    def count(self) -> int:
        return self._store.count()

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        use_hybrid: bool = True,
        use_reranker: bool = False,
    ) -> List[Document]:
        return self._store.search(query, top_k, filter_metadata, use_hybrid, use_reranker)

    def get_full_content(self, source: str) -> str:
        """
        Return all chunk text for a given source, ordered by page.
        Uses _CollectionCompat.get() instead of accessing private internals.
        """
        results = self.collection.get(
            where={"source": source},
            include=["documents", "metadatas"],
        )
        documents = results.get("documents") or []
        metadatas = results.get("metadatas") or [{} for _ in documents]

        def page_key(item: tuple[str, dict]) -> tuple[int, object]:
            page = item[1].get("page")
            if isinstance(page, (int, float)):
                return (0, page)
            return (1, str(page or ""))

        ordered = sorted(zip(documents, metadatas), key=page_key)
        return "\n\n".join(text for text, _ in ordered)


class CollectionCompat:
    """
    Thin compatibility shim so that existing code accessing
    `vector_store.collection.get(...)` continues to work.

    Only the methods actually used by the codebase are implemented:
      - get(include, where, limit)
      - count()
    """

    def __init__(self, store: TurboVecStore):
        self._store = store

    def get(self, include=None, where=None, limit=None):
        """Emulate Chroma's collection.get() by scanning in-memory data.
        Delegates $or/$and filter logic to TurboVecStore._metadata_matches_filter()."""
        ids = []
        documents = []
        metadatas = []

        for doc_id, meta in self._store._metadatas.items():
            if where:
                if not self._store._metadata_matches_filter(meta, where):
                    continue
            ids.append(doc_id)
            documents.append(self._store._docs.get(doc_id, ""))
            metadatas.append(meta)

            if limit and len(ids) >= limit:
                break

        result = {"ids": ids}
        if include is None or "documents" in include:
            result["documents"] = documents
        if include is None or "metadatas" in include:
            result["metadatas"] = metadatas
        return result

    def count(self) -> int:
        return self._store.count()