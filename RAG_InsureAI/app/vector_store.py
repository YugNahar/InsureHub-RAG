"""
ChromaDB Vector Store with hybrid search (dense + BM25) and cross‑encoder reranking.
"""
import os
import pickle
import uuid
import logging
from typing import List, Optional, Dict, Any

import chromadb
from chromadb.config import Settings
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "insurance_docs"
BM25_CACHE_VERSION = 1
BM25_CACHE_PATH = os.path.join(CHROMA_PERSIST_DIR, f"{COLLECTION_NAME}_bm25_cache.pkl")


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
        self._rebuild_bm25_flag = not self._load_bm25()

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
    def _save_bm25(self):
        """Persist the BM25 corpus atomically so startup can skip a Chroma scan."""
        collection_count = self.collection.count()
        if len(self._bm25_corpus) != collection_count:
            logger.warning(
                "BM25 cache not saved because corpus size (%d) differs from Chroma count (%d).",
                len(self._bm25_corpus),
                collection_count,
            )
            return

        payload = {
            "version": BM25_CACHE_VERSION,
            "collection": COLLECTION_NAME,
            "document_count": collection_count,
            "corpus": self._bm25_corpus,
        }
        os.makedirs(os.path.dirname(BM25_CACHE_PATH), exist_ok=True)
        temp_path = f"{BM25_CACHE_PATH}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        try:
            with open(temp_path, "wb") as cache_file:
                pickle.dump(payload, cache_file, protocol=pickle.HIGHEST_PROTOCOL)
                cache_file.flush()
                os.fsync(cache_file.fileno())
            os.replace(temp_path, BM25_CACHE_PATH)
            logger.info("BM25 cache saved (%d docs).", collection_count)
        except Exception as exc:
            logger.warning("BM25 cache save failed: %s", exc)
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    def _load_bm25(self) -> bool:
        """Load a current BM25 corpus from disk without scanning Chroma."""
        try:
            os.makedirs(os.path.dirname(BM25_CACHE_PATH), exist_ok=True)
        except OSError as exc:
            logger.warning("BM25 cache directory unavailable, will rebuild: %s", exc)
            return False

        if not os.path.exists(BM25_CACHE_PATH):
            return False

        try:
            with open(BM25_CACHE_PATH, "rb") as cache_file:
                payload = pickle.load(cache_file)

            if not isinstance(payload, dict):
                raise ValueError("unsupported legacy cache format")
            if payload.get("version") != BM25_CACHE_VERSION:
                raise ValueError("cache version mismatch")
            if payload.get("collection") != COLLECTION_NAME:
                raise ValueError("cache collection mismatch")

            corpus = payload.get("corpus")
            cached_count = payload.get("document_count")
            collection_count = self.collection.count()
            if not isinstance(corpus, list) or cached_count != collection_count:
                raise ValueError(
                    f"cache count {cached_count!r} does not match Chroma count {collection_count}"
                )
            if len(corpus) != collection_count:
                raise ValueError(
                    f"corpus size {len(corpus)} does not match Chroma count {collection_count}"
                )
            if any(
                not isinstance(item, tuple)
                or len(item) != 3
                or not isinstance(item[1], str)
                or not isinstance(item[2], dict)
                for item in corpus
            ):
                raise ValueError("cache corpus contains invalid entries")

            self._bm25_corpus = corpus
            tokenized_corpus = [text.lower().split() for _, text, _ in corpus]
            self._bm25_index = BM25Okapi(tokenized_corpus) if tokenized_corpus else None
            logger.info("BM25 cache loaded (%d docs).", len(corpus))
            return True
        except Exception as exc:
            logger.warning("BM25 cache load failed, will rebuild: %s", exc)
            self._bm25_index = None
            self._bm25_corpus = []
            return False

    def _invalidate_bm25_cache(self):
        """Discard stale in-memory and on-disk BM25 data before a mutation."""
        self._bm25_index = None
        self._bm25_corpus = []
        self._rebuild_bm25_flag = True
        try:
            if os.path.exists(BM25_CACHE_PATH):
                os.remove(BM25_CACHE_PATH)
        except OSError as exc:
            logger.warning("BM25 cache invalidation failed: %s", exc)

    def _rebuild_bm25(self):
        """Fetch all chunks and build BM25 index."""
        all_data = self.collection.get(include=["documents", "metadatas"])
        if not all_data["ids"]:
            self._bm25_index = None
            self._bm25_corpus = []
            self._save_bm25()
            logger.info("BM25 index rebuilt with 0 documents")
            return

        tokenized_corpus = []
        corpus = []
        for doc_id, text, meta in zip(all_data["ids"], all_data["documents"], all_data["metadatas"]):
            tokens = text.lower().split()
            tokenized_corpus.append(tokens)
            corpus.append((doc_id, text, meta))

        self._bm25_index = BM25Okapi(tokenized_corpus)
        self._bm25_corpus = corpus
        self._save_bm25()
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
        self._invalidate_bm25_cache()
        for i in range(0, len(ids), batch):
            self.collection.add(
                ids=ids[i:i+batch],
                embeddings=embeddings[i:i+batch],
                metadatas=metadatas[i:i+batch],
                documents=texts[i:i+batch],
            )

        logger.info("Added %d chunks, BM25 will be rebuilt.", len(ids))
        return ids

    def delete_by_source(self, source: str):
        results = self.collection.get(where={"source": source}, include=[])
        ids = results["ids"]
        if ids:
            self._invalidate_bm25_cache()
            self.collection.delete(ids=ids)
            logger.info("Deleted %d chunks for source=%s", len(ids), source)

    def delete_all(self):
        self._invalidate_bm25_cache()
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._bm25_index = None
        self._bm25_corpus = []
        self._rebuild_bm25_flag = False
        self._save_bm25()
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

    @staticmethod
    def _metadata_value_matches(value: Any, condition: Any) -> bool:
        """Evaluate the Chroma metadata operators used by this application."""
        if not isinstance(condition, dict):
            return value == condition

        for operator, expected in condition.items():
            try:
                if operator == "$eq" and value != expected:
                    return False
                if operator == "$ne" and value == expected:
                    return False
                if operator == "$in" and (
                    not isinstance(expected, (list, tuple, set, frozenset))
                    or value not in expected
                ):
                    return False
                if operator == "$nin" and (
                    not isinstance(expected, (list, tuple, set, frozenset))
                    or value in expected
                ):
                    return False
                if operator == "$contains" and (
                    not isinstance(value, str)
                    or not isinstance(expected, str)
                    or expected not in value
                ):
                    return False
                if operator == "$gt" and not (value is not None and value > expected):
                    return False
                if operator == "$gte" and not (value is not None and value >= expected):
                    return False
                if operator == "$lt" and not (value is not None and value < expected):
                    return False
                if operator == "$lte" and not (value is not None and value <= expected):
                    return False
                if operator not in {
                    "$eq", "$ne", "$in", "$nin", "$contains",
                    "$gt", "$gte", "$lt", "$lte",
                }:
                    logger.warning("Unsupported BM25 metadata filter operator: %s", operator)
                    return False
            except (TypeError, ValueError):
                return False
        return True

    @classmethod
    def _metadata_matches_filter(cls, metadata: Dict[str, Any], filter_meta: Optional[Dict]) -> bool:
        """Apply a Chroma-style metadata filter to one BM25 corpus entry."""
        if not filter_meta:
            return True
        if not isinstance(filter_meta, dict):
            return False

        for key, condition in filter_meta.items():
            if key == "$and":
                if not isinstance(condition, list) or not condition or not all(
                    cls._metadata_matches_filter(metadata, item) for item in condition
                ):
                    return False
            elif key == "$or":
                if not isinstance(condition, list) or not condition or not any(
                    cls._metadata_matches_filter(metadata, item) for item in condition
                ):
                    return False
            elif not cls._metadata_value_matches(metadata.get(key), condition):
                return False
        return True

    def _bm25_search(
        self,
        query: str,
        k: int,
        filter_meta: Optional[Dict] = None,
    ) -> List[tuple]:
        bm25 = self._get_bm25()
        if bm25 is None or not self._bm25_corpus:
            return []

        tokens = query.lower().split()
        scores = bm25.get_scores(tokens)
        filtered_with_scores = [
            (scores[index], doc_id, text, meta)
            for index, (doc_id, text, meta) in enumerate(self._bm25_corpus)
            if self._metadata_matches_filter(meta, filter_meta)
        ]
        filtered_with_scores.sort(key=lambda item: item[0], reverse=True)
        return [
            (doc_id, text, meta, score)
            for score, doc_id, text, meta in filtered_with_scores[:k]
            if score > 0
        ]

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

        hybrid_used = use_hybrid
        if hybrid_used:
            bm25_candidates = self._bm25_search(
                query,
                k=top_k,
                filter_meta=filter_metadata,
            )
            merged = {}
            for doc_id, text, meta, score in dense_candidates + bm25_candidates:
                if doc_id not in merged:
                    merged[doc_id] = (doc_id, text, meta, score)
            candidates = list(merged.values())
        else:
            candidates = dense_candidates

        reranker_used = use_reranker and len(candidates) > 1
        if reranker_used:
            candidates = self._rerank(query, candidates, top_k)
        else:
            candidates = candidates[:top_k]

        docs = []
        for doc_id, text, meta, orig_score in candidates:
            doc = Document(page_content=text, metadata=dict(meta))
            doc.metadata["similarity"] = orig_score
            method = "hybrid" if hybrid_used else "dense"
            if reranker_used:
                method += "+rerank"
            doc.metadata["retrieval_method"] = method
            docs.append(doc)
        return docs

    def count(self) -> int:
        return self.collection.count()
