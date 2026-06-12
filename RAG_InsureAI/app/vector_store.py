"""
ChromaDB Vector Store with hybrid search (dense + BM25) and cross‑encoder reranking.
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
COLLECTION_NAME = "insurance_docs"


class ChromaVectorStore:
    """
    Persistent vector store with:
      - Dense retrieval (BGE)
      - Keyword retrieval (BM25)
      - Cross‑encoder reranking
    """

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

        # Cross‑encoder for reranking (lazy load)
        self.reranker = None
        self._bm25_index = None
        self._bm25_corpus = []
        self._rebuild_bm25_flag = True

        logger.info(
            "ChromaVectorStore ready — collection=%s, embed=%s, chunks=%d",
            COLLECTION_NAME, EMBED_MODEL_NAME, self.collection.count(),
        )

    # ------------------------------------------------------------------
    # Embedding (dense)
    # ------------------------------------------------------------------
    def _embed(self, texts: List[str]) -> List[List[float]]:
        return self.embed_model.encode(
            texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False
        ).tolist()

    # ------------------------------------------------------------------
    # BM25 index management
    # ------------------------------------------------------------------
    def _rebuild_bm25(self):
        """Fetch all chunks and build BM25 index."""
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
        logger.info("BM25 index rebuilt with %d documents", len(corpus))

    def _get_bm25(self):
        if self._rebuild_bm25_flag:
            self._rebuild_bm25()
            self._rebuild_bm25_flag = False
        return self._bm25_index

    # ------------------------------------------------------------------
    # Add / Delete / Update
    # ------------------------------------------------------------------
    def add_documents(self, docs: List[Document]) -> List[str]:
        if not docs:
            return []

        ids = [str(uuid.uuid4()) for _ in docs]
        texts = [doc.page_content for doc in docs]
        metadatas = []
        for doc, iid in zip(docs, ids):
            meta = dict(doc.metadata)
            meta["id"] = iid
            for k, v in list(meta.items()):
                if isinstance(v, list):
                    meta[k] = ", ".join(str(x) for x in v)
                elif v is None:
                    meta[k] = ""
            metadatas.append(meta)

        embeddings = self._embed(texts)

        batch = 5000
        for i in range(0, len(ids), batch):
            self.collection.add(
                ids=ids[i:i+batch],
                embeddings=embeddings[i:i+batch],
                metadatas=metadatas[i:i+batch],
                documents=texts[i:i+batch],
            )

        self._rebuild_bm25_flag = True
        logger.info("Added %d chunks, BM25 will be rebuilt.", len(ids))
        return ids

    def delete_by_source(self, source: str):
        results = self.collection.get(where={"source": source}, include=[])
        ids = results["ids"]
        if ids:
            self.collection.delete(ids=ids)
            self._rebuild_bm25_flag = True
            logger.info("Deleted %d chunks for source=%s", len(ids), source)

    def delete_all(self):
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._rebuild_bm25_flag = True
        logger.info("Cleared entire vector store.")

    def list_sources(self) -> List[str]:
        all_meta = self.collection.get(include=["metadatas"])
        sources = {meta.get("source") for meta in all_meta["metadatas"] if meta.get("source")}
        return sorted(sources)

    # ------------------------------------------------------------------
    # Hybrid search + reranking
    # ------------------------------------------------------------------
    def _dense_search(self, query: str, k: int, filter_meta: Optional[Dict] = None) -> List[tuple]:
        query_embedding = self._embed([query])[0]
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": k,
            "include": ["documents", "metadatas", "distances"],
        }
        if filter_meta:
            kwargs["where"] = filter_meta
        try:
            res = self.collection.query(**kwargs)
        except Exception as e:
            logger.error("Dense query failed (filter=%s): %s", filter_meta, e)
            raise

        results = []
        for doc_id, text, meta, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            similarity = 1 - dist
            results.append((doc_id, text, meta, similarity))
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
        rerank_scores = self.reranker.predict(pairs)
        combined = list(zip(candidates, rerank_scores))
        combined.sort(key=lambda x: x[1], reverse=True)
        reranked = [item[0] for item in combined[:top_k]]
        for (doc_id, text, meta, _), score in zip(reranked, [s for _, s in combined[:top_k]]):
            meta["rerank_score"] = float(score)
        return reranked

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        use_hybrid: bool = True,
        use_reranker: bool = False,
    ) -> List[Document]:
        count = self.collection.count()
        if count == 0:
            return []

        safe_k = min(2 * top_k, count)
        dense_candidates = self._dense_search(query, k=safe_k, filter_meta=filter_metadata)

        if use_hybrid:
            bm25_candidates = self._bm25_search(query, k=top_k)
            merged = {}
            for doc_id, text, meta, score in dense_candidates + bm25_candidates:
                if doc_id not in merged:
                    merged[doc_id] = (doc_id, text, meta, score)
            candidates = list(merged.values())
        else:
            candidates = dense_candidates

        if use_reranker and len(candidates) > 1:
            candidates = self._rerank(query, candidates, top_k)
        else:
            candidates = candidates[:top_k]

        docs = []
        for doc_id, text, meta, orig_score in candidates:
            doc = Document(page_content=text, metadata=dict(meta))
            doc.metadata["similarity"] = orig_score
            doc.metadata["retrieval_method"] = "hybrid+rerank" if use_hybrid and use_reranker else "dense"
            docs.append(doc)
        return docs

    def count(self) -> int:
        return self.collection.count()