"""
TurboVec-backed Vector Store for video transcripts — permanent storage,
deduplication, hybrid search, and reranking.
"""
import os
import logging
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse, urlencode, parse_qs, urlunparse

from langchain_core.documents import Document

from turbovec_store import TurboVecStore

logger = logging.getLogger(__name__)

COLLECTION_NAME = os.getenv("TVEC_VIDEOS_COLLECTION", "insurance_videos")


def _normalize_video_url(url: str) -> str:
    """Strip tracking params (si, utm_*, etc.) from video URLs."""
    parsed = urlparse(url)
    params = parse_qs(parsed.query, keep_blank_values=True)
    # Remove known tracking-only params
    for key in ["si", "utm_source", "utm_medium", "utm_campaign", "feature"]:
        params.pop(key, None)
    clean_query = urlencode({k: v[0] for k, v in params.items()})
    return urlunparse((
        parsed.scheme, parsed.netloc, parsed.path,
        parsed.params, clean_query, ""
    ))


class VideoVectorStore:
    """Persistent vector store for video transcripts (YouTube)."""

    def __init__(self):
        self._store = TurboVecStore(
            collection_name=COLLECTION_NAME,
            persist_subdir="videos",
        )

    def add_video_chunks(self, url: str, chunks: List[Document], title: str = "") -> List[str]:
        url = _normalize_video_url(url)
        if not chunks:
            return []
        self.delete_by_url(url)
        for chunk in chunks:
            chunk.metadata["source_url"] = url
            chunk.metadata["source_type"] = "video"
            # Ensure title is always stored so list_videos_with_titles() can retrieve it
            if title:
                chunk.metadata["video_title"] = title
            elif "title" in chunk.metadata and not chunk.metadata.get("video_title"):
                chunk.metadata["video_title"] = chunk.metadata["title"]
        return self._store.add_documents(chunks)

    def delete_by_url(self, url: str):
        url = _normalize_video_url(url)
        self._store.delete_by_field("source_url", url)

    def url_exists(self, url: str) -> bool:
        url = _normalize_video_url(url)
        urls = self._store.list_values("source_url")
        return url in urls

    def list_urls(self) -> List[str]:
        return self._store.list_values("source_url")

    def list_videos_with_titles(self) -> List[Dict[str, str]]:
        """Return [{url, title}] for every stored video."""
        urls = self._store.list_values("source_url")
        if not urls:
            return []
        # Get title for each URL from the first matching chunk's metadata
        results = []
        seen = set()
        try:
            url_to_title: Dict[str, str] = {}
            for meta in self._store._metadatas.values():
                u = meta.get("source_url", "")
                if u and u not in url_to_title:
                    url_to_title[u] = meta.get("video_title") or meta.get("title") or u
            for u in urls:
                if u not in seen:
                    seen.add(u)
                    results.append({"url": u, "title": url_to_title.get(u, u)})
        except Exception:
            for u in urls:
                if u not in seen:
                    seen.add(u)
                    results.append({"url": u, "title": u})
        return results

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict] = None,
        use_hybrid: bool = True,
        use_reranker: bool = False,
    ) -> List[Document]:
        if filter_metadata is None:
            filter_metadata = {"source_type": "video"}
        return self._store.search(query, top_k, filter_metadata, use_hybrid, use_reranker)

    def count(self) -> int:
        return self._store.count()

    @property
    def collection(self):
        """Compatibility shim for direct ChromaDB-like access."""
        from vector_store import CollectionCompat
        return CollectionCompat(self._store)
