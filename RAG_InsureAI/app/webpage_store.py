"""
TurboVec-backed Vector Store for permanent webpage storage — separate from
documents and videos. Provides hybrid search and reranking.
"""
import os
import logging
from typing import List, Optional, Dict, Any

from langchain_core.documents import Document

from turbovec_store import TurboVecStore

logger = logging.getLogger(__name__)

COLLECTION_NAME = os.getenv("TVEC_WEBPAGES_COLLECTION", "insurance_webpages")


class WebpageVectorStore:
    """Persistent vector store for webpage content."""

    def __init__(self):
        self._store = TurboVecStore(
            collection_name=COLLECTION_NAME,
            persist_subdir="webpages",
        )

    def add_webpage_chunks(self, url: str, chunks: List[Document]) -> List[str]:
        if not chunks:
            return []
        self.delete_by_url(url)
        for chunk in chunks:
            chunk.metadata["source_url"] = url
            chunk.metadata["source_type"] = "webpage"
        return self._store.add_documents(chunks)

    def delete_by_url(self, url: str):
        self._store.delete_by_field("source_url", url)

    def url_exists(self, url: str) -> bool:
        urls = self._store.list_values("source_url")
        return url in urls

    def list_urls(self) -> List[str]:
        return self._store.list_values("source_url")

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict] = None,
        use_hybrid: bool = True,
        use_reranker: bool = False,
    ) -> List[Document]:
        if filter_metadata is None:
            filter_metadata = {"source_type": "webpage"}
        return self._store.search(query, top_k, filter_metadata, use_hybrid, use_reranker)

    def count(self) -> int:
        return self._store.count()

    @property
    def collection(self):
        """Compatibility shim for direct ChromaDB-like access."""
        from vector_store import CollectionCompat
        return CollectionCompat(self._store)
