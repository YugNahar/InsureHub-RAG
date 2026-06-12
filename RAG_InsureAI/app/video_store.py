"""
Vector store for video transcripts (YouTube) – permanent storage, deduplication.
"""
import os
import uuid
import logging
from typing import List, Optional, Dict, Any

import chromadb
from chromadb.config import Settings
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import numpy as np

logger = logging.getLogger(__name__)

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "insurance_videos"

class VideoVectorStore:
    """Persistent vector store for video transcripts (YouTube)."""
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self.embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        self.embed_model.max_seq_length = 512
        self.reranker = None
        self._bm25_index = None
        self._bm25_corpus = []
        self._rebuild_bm25_flag = True
        logger.info("VideoVectorStore ready — collection=%s, chunks=%d", COLLECTION_NAME, self.collection.count())

    def _embed(self, texts: List[str]) -> List[List[float]]:
        return self.embed_model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False).tolist()

    def _rebuild_bm25(self):
        all_data = self.collection.get(include=["documents", "metadatas"])
        if not all_data["ids"]:
            self._bm25_index = None
            self._bm25_corpus = []
            return
        tokenized_corpus = []
        corpus = []
        for doc_id, text, meta in zip(all_data["ids"], all_data["documents"], all_data["metadatas"]):
            tokens = text.lower().split()
            tokenized_corpus.append(tokens)
            corpus.append((doc_id, text, meta))
        self._bm25_index = BM25Okapi(tokenized_corpus)
        self._bm25_corpus = corpus
        logger.info("Video BM25 rebuilt with %d docs", len(corpus))

    def _get_bm25(self):
        if self._rebuild_bm25_flag:
            self._rebuild_bm25()
            self._rebuild_bm25_flag = False
        return self._bm25_index

    def add_video_chunks(self, url: str, chunks: List[Document]) -> List[str]:
        if not chunks:
            return []
        self.delete_by_url(url)
        ids = [str(uuid.uuid4()) for _ in chunks]
        texts = [c.page_content for c in chunks]
        metadatas = []
        for chunk, iid in zip(chunks, ids):
            meta = dict(chunk.metadata)
            meta["id"] = iid
            meta["source_url"] = url
            meta["source_type"] = "video"
            for k, v in list(meta.items()):
                if isinstance(v, list):
                    meta[k] = ", ".join(str(x) for x in v)
                elif v is None:
                    meta[k] = ""
            metadatas.append(meta)
        embeddings = self._embed(texts)
        self.collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=texts)
        self._rebuild_bm25_flag = True
        logger.info("Added %d chunks for video %s", len(ids), url)
        return ids

    def delete_by_url(self, url: str):
        results = self.collection.get(where={"source_url": url}, include=[])
        ids = results["ids"]
        if ids:
            self.collection.delete(ids=ids)
            self._rebuild_bm25()   # rebuild immediately so stale chunks aren't returned
            self._rebuild_bm25_flag = False
            logger.info("Deleted %d chunks for video %s", len(ids), url)

    def url_exists(self, url: str) -> bool:
        results = self.collection.get(where={"source_url": url}, limit=1, include=[])
        return len(results["ids"]) > 0

    def list_urls(self) -> List[str]:
        all_meta = self.collection.get(include=["metadatas"])
        urls = set()
        for meta in all_meta["metadatas"]:
            if meta.get("source_url"):
                urls.add(meta["source_url"])
        return sorted(urls)

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict] = None,
        use_hybrid: bool = True,
        use_reranker: bool = False
    ) -> List[Document]:
        """Search with optional hybrid and reranking."""
        count = self.collection.count()
        if count == 0:
            return []

        # Dense search
        safe_k = min(2 * top_k, count)
        dense = self._dense_search(query, safe_k, filter_metadata)
        
        # Hybrid: add BM25
        if use_hybrid:
            bm25 = self._bm25_search(query, top_k)
            merged = {}
            for doc_id, text, meta, score in dense + bm25:
                if doc_id not in merged:
                    merged[doc_id] = (doc_id, text, meta, score)
            candidates = list(merged.values())
        else:
            candidates = dense
        
        # Rerank
        if use_reranker and len(candidates) > 1:
            candidates = self._rerank(query, candidates, top_k)
        else:
            candidates = candidates[:top_k]
        
        docs = []
        for doc_id, text, meta, orig_score in candidates:
            doc = Document(page_content=text, metadata=dict(meta))
            doc.metadata["similarity"] = orig_score
            docs.append(doc)
        return docs

    def _dense_search(self, query: str, k: int, filter_meta: Optional[Dict] = None) -> List[tuple]:
        q_emb = self._embed([query])[0]
        kwargs = {"query_embeddings": [q_emb], "n_results": k, "include": ["documents", "metadatas", "distances"]}
        if filter_meta:
            kwargs["where"] = filter_meta
        res = self.collection.query(**kwargs)
        results = []
        for doc_id, text, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
            results.append((doc_id, text, meta, 1 - dist))
        return results

    def _bm25_search(self, query: str, k: int) -> List[tuple]:
        bm25 = self._get_bm25()
        if bm25 is None or not self._bm25_corpus:
            return []
        tokens = query.lower().split()
        scores = bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[-k:][::-1]
        results = []
        for idx in top_idx:
            if scores[idx] > 0:
                doc_id, text, meta = self._bm25_corpus[idx]
                results.append((doc_id, text, meta, scores[idx]))
        return results

    def _rerank(self, query: str, candidates: List[tuple], top_k: int) -> List[tuple]:
        if not candidates:
            return []
        if self.reranker is None:
            self.reranker = CrossEncoder(RERANKER_MODEL_NAME)
        pairs = [(query, text) for (_, text, _, _) in candidates]
        scores = self.reranker.predict(pairs)
        combined = list(zip(candidates, scores))
        combined.sort(key=lambda x: x[1], reverse=True)
        return [item[0] for item in combined[:top_k]]

    def count(self) -> int:
        return self.collection.count()