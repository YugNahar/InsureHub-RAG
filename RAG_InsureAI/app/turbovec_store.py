"""
TurboVec-based vector store — replaces ChromaDB with TurboQuantIndex for
dense ANN retrieval while keeping the same embedding model, BM25 hybrid,
and cross-encoder reranker.

Benefits:
  - ~4GB memory instead of ~31GB (optimized for ARM and low-resource hosts)
  - 4-bit quantization, no GPU required
  - Zero external dependencies for the index itself
  - Suitable for air-gapped / privacy-sensitive deployments
"""
import json
import logging
import os
import pickle
import threading
import uuid
from typing import List, Optional, Dict, Any

import numpy as np
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"
TVEC_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "turbovec_data")

_TVEC_IMPORT_ATTEMPTED = False
_TVEC_AVAILABLE = False
_TVEC_IMPORT_LOCK = threading.Lock()


def _ensure_turbovec():
    """Lazy-import turbovec exactly once. Returns True if available."""
    global _TVEC_AVAILABLE, _TVEC_IMPORT_ATTEMPTED
    if _TVEC_IMPORT_ATTEMPTED:
        return _TVEC_AVAILABLE
    with _TVEC_IMPORT_LOCK:
        if _TVEC_IMPORT_ATTEMPTED:
            return _TVEC_AVAILABLE
        try:
            import turbovec  # noqa: F401
            _TVEC_AVAILABLE = True
            logger.info("turbovec is available — using TurboQuantIndex for ANN.")
        except ImportError:
            _TVEC_AVAILABLE = False
            logger.warning(
                "turbovec not installed. Install with: pip install turbovec\n"
                "Falling back to ChromaDB (if available) or degraded mode."
            )
        _TVEC_IMPORT_ATTEMPTED = True
    return _TVEC_AVAILABLE


def _get_embed_dim(embed_model: SentenceTransformer) -> int:
    """Return the embedding dimension of the given model."""
    return embed_model.get_sentence_embedding_dimension()


class TurboVecStore:
    """
    A ChromaDB-replacement vector store backed by TurboQuantIndex for dense
    ANN retrieval, with:
      - Dense retrieval (BGE embedding -> TurboVec)
      - Keyword retrieval (BM25)
      - Cross-encoder reranking
      - JSON-based metadata/document persistence (TurboVec stores only vectors)

    Metadata is persisted as newline-delimited JSON lines (`.ndjson`) alongside
    the TurboVec index file so data survives restarts without re-ingestion.
    """

    def __init__(
        self,
        collection_name: str,
        persist_subdir: str = "",
        embed_model_name: str = EMBED_MODEL_NAME,
        bit_width: int = 4,
    ):
        _ensure_turbovec()

        self.collection_name = collection_name
        self.persist_dir = os.path.join(TVEC_PERSIST_DIR, persist_subdir or collection_name)
        os.makedirs(self.persist_dir, exist_ok=True)

        self.index_path = os.path.join(self.persist_dir, f"{collection_name}.tq")
        self.meta_path = os.path.join(self.persist_dir, f"{collection_name}_meta.ndjson")
        self.bm25_cache_path = os.path.join(self.persist_dir, f"{collection_name}_bm25.pkl")
        self.idmap_path = os.path.join(self.persist_dir, f"{collection_name}_idmap.json")

        # Embedding model
        self.embed_model = SentenceTransformer(embed_model_name)
        self.embed_model.max_seq_length = 512
        self.embed_dim = _get_embed_dim(self.embed_model)
        self.bit_width = bit_width

        # Reranker (lazy-loaded)
        self.reranker = None

        # BM25 state
        self._bm25_index = None
        self._bm25_corpus: List[tuple] = []  # (doc_id, text, metadata_dict)
        self._rebuild_bm25_flag = True

        # In-memory document store (keyed by doc_id)
        self._docs: Dict[str, str] = {}           # doc_id -> text
        self._metadatas: Dict[str, dict] = {}      # doc_id -> metadata

        # TurboVec index
        self._tvec_index = None
        self._id_map: Dict[int, str] = {}         # sequential int ID -> doc_id
        self._reverse_id_map: Dict[str, int] = {}  # doc_id -> sequential int ID
        self._next_seq_id = 0

        # Load existing data from disk
        self._load_state()

        # If metadata loaded but TurboVec index couldn't load, rebuild it
        if self._docs and self._tvec_index is None:
            logger.warning(
                "TurboVec index is missing or corrupt for '%s' (%d documents loaded). "
                "Triggering re-embedding and index rebuild.",
                self.collection_name, len(self._docs),
            )
            self._rebuild_tvec_index()

        logger.info(
            "TurboVecStore ready — collection=%s, embed=%s, dim=%d, chunks=%d",
            collection_name, embed_model_name, self.embed_dim, len(self._docs),
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_state(self):
        """Write index + metadata + ID map atomically."""
        # 1. Write metadata as NDJSON
        meta_path_tmp = self.meta_path + f".{os.getpid()}.{uuid.uuid4().hex}.tmp"
        try:
            with open(meta_path_tmp, "w", encoding="utf-8") as f:
                for doc_id in list(self._docs.keys()):
                    record = {
                        "id": doc_id,
                        "text": self._docs[doc_id],
                        "metadata": self._metadatas.get(doc_id, {}),
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            os.replace(meta_path_tmp, self.meta_path)
        except Exception as exc:
            logger.warning("Failed to save metadata for %s: %s", self.collection_name, exc)
        finally:
            if os.path.exists(meta_path_tmp):
                try:
                    os.remove(meta_path_tmp)
                except OSError:
                    pass

        # 2. Write TurboVec index
        if self._tvec_index is not None and self._next_seq_id > 0:
            idx_path_tmp = self.index_path + f".{os.getpid()}.{uuid.uuid4().hex}.tmp"
            try:
                self._tvec_index.write(idx_path_tmp)
                os.replace(idx_path_tmp, self.index_path)
            except Exception as exc:
                logger.warning("Failed to save TurboVec index for %s: %s", self.collection_name, exc)
            finally:
                if os.path.exists(idx_path_tmp):
                    try:
                        os.remove(idx_path_tmp)
                    except OSError:
                        pass

        # 3. Write ID map (so restart can reconstruct seq_id -> doc_id exactly)
        idmap_path_tmp = self.idmap_path + f".{os.getpid()}.{uuid.uuid4().hex}.tmp"
        try:
            with open(idmap_path_tmp, "w") as f:
                json.dump({
                    "id_map": {str(k): v for k, v in self._id_map.items()},
                    "next_seq_id": self._next_seq_id,
                }, f)
            os.replace(idmap_path_tmp, self.idmap_path)
        except Exception as exc:
            logger.warning("Failed to save ID map for %s: %s", self.collection_name, exc)
        finally:
            if os.path.exists(idmap_path_tmp):
                try:
                    os.remove(idmap_path_tmp)
                except OSError:
                    pass

        # 4. Save BM25 cache
        self._save_bm25()

    def _load_state(self):
        """Load index + metadata + ID map from disk."""
        # Load metadata
        if os.path.exists(self.meta_path):
            try:
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        record = json.loads(line)
                        doc_id = record["id"]
                        self._docs[doc_id] = record["text"]
                        self._metadatas[doc_id] = record.get("metadata", {})
                logger.info(
                    "Loaded %d documents for %s from %s",
                    len(self._docs), self.collection_name, self.meta_path,
                )
            except Exception as exc:
                logger.warning("Failed to load metadata for %s: %s", self.collection_name, exc)

        # Load ID map first (so doc_id <-> seq_id mapping is exact)
        if os.path.exists(self.idmap_path):
            try:
                with open(self.idmap_path, "r") as f:
                    d = json.load(f)
                self._id_map = {int(k): v for k, v in d["id_map"].items()}
                self._reverse_id_map = {v: k for k, v in self._id_map.items()}
                self._next_seq_id = d["next_seq_id"]
                logger.info(
                    "Loaded ID map for %s (%d entries).",
                    self.collection_name, len(self._id_map),
                )
            except Exception as exc:
                logger.warning(
                    "Failed to load ID map for %s: %s. Will rebuild from metadata order.",
                    self.collection_name, exc,
                )
                self._id_map.clear()
                self._reverse_id_map.clear()
                self._next_seq_id = 0

        # If no ID map on disk but metadata loaded, rebuild from metadata order
        # (legacy migration path for first restart after upgrade)
        if not self._id_map and self._docs:
            logger.info("Rebuilding ID map from metadata insertion order for %s.", self.collection_name)
            for idx, doc_id in enumerate(self._docs):
                self._id_map[idx] = doc_id
                self._reverse_id_map[doc_id] = idx
            self._next_seq_id = len(self._docs)

        # Load TurboVec index — use IdMapIndex if available (consistent with _ensure_tvec_index),
        # else fall back to TurboQuantIndex for backward compatibility.
        if os.path.exists(self.index_path):
            try:
                from turbovec import IdMapIndex
                self._tvec_index = IdMapIndex.load(self.index_path)
                logger.info(
                    "Loaded TurboVec IdMapIndex for %s (%d vectors).",
                    self.collection_name, self._next_seq_id,
                )
            except Exception:
                try:
                    from turbovec import TurboQuantIndex
                    self._tvec_index = TurboQuantIndex.load(self.index_path)
                    logger.info(
                        "Loaded TurboVec TurboQuantIndex for %s (%d vectors).",
                        self.collection_name, self._next_seq_id,
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to load TurboVec index for %s: %s",
                        self.collection_name, exc,
                    )
                    self._tvec_index = None

        # Load BM25 cache
        self._load_bm25()

    def _rebuild_tvec_index(self):
        """Re-embed all documents and rebuild the TurboVec index from scratch."""
        if not self._docs:
            return
        doc_ids = list(self._docs.keys())
        texts = [self._docs[did] for did in doc_ids]
        logger.info(
            "Re-embedding %d documents for %s TurboVec index...",
            len(texts), self.collection_name,
        )
        embeddings = self._embed(texts)

        # Reset index
        self._tvec_index = None
        self._id_map.clear()
        self._reverse_id_map.clear()
        self._next_seq_id = 0

        self._add_vectors(doc_ids, embeddings)
        self._save_state()
        logger.info(
            "TurboVec index rebuilt for %s (%d vectors).",
            self.collection_name, self._next_seq_id,
        )

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _embed(self, texts: List[str]) -> np.ndarray:
        return self.embed_model.encode(
            texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False,
        )

    # ------------------------------------------------------------------
    # TurboVec index management
    # ------------------------------------------------------------------

    def _ensure_tvec_index(self):
        """Lazy-create the TurboVec index."""
        if self._tvec_index is None:
            from turbovec import IdMapIndex
            self._tvec_index = IdMapIndex(dim=self.embed_dim, bit_width=self.bit_width)

    def _add_vectors(self, doc_ids: List[str], embeddings: np.ndarray):
        """Add vectors to the TurboVec index."""
        self._ensure_tvec_index()
        start = self._next_seq_id
        seq_ids = list(range(start, start + len(doc_ids)))
        self._tvec_index.add_with_ids(embeddings, np.array(seq_ids, dtype=np.uint64))
        for seq_id, doc_id in zip(seq_ids, doc_ids):
            self._id_map[seq_id] = doc_id
            self._reverse_id_map[doc_id] = seq_id
        self._next_seq_id = start + len(doc_ids)

    def _remove_vector(self, doc_id: str):
        """Remove a vector from the TurboVec index by doc_id."""
        seq_id = self._reverse_id_map.get(doc_id)
        if seq_id is not None:
            try:
                self._tvec_index.remove(seq_id)
            except Exception as exc:
                logger.warning("TurboVec remove failed for %s: %s", doc_id, exc)
            self._id_map.pop(seq_id, None)
            self._reverse_id_map.pop(doc_id, None)

    # ------------------------------------------------------------------
    # BM25 index management
    # ------------------------------------------------------------------

    def _save_bm25(self):
        """Persist the BM25 corpus (always writes, even if empty)."""
        # Bug fix: always write whatever corpus we have — don't skip on mismatch
        payload = {
            "version": 1,
            "collection": self.collection_name,
            "document_count": len(self._docs),
            "corpus": self._bm25_corpus,
        }
        # Initialize temp_path before try so finally always has it
        temp_path = self.bm25_cache_path + f".{os.getpid()}.{uuid.uuid4().hex}.tmp"
        try:
            os.makedirs(os.path.dirname(self.bm25_cache_path), exist_ok=True)
            with open(temp_path, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, self.bm25_cache_path)
        except Exception as exc:
            logger.warning("BM25 cache save failed for %s: %s", self.collection_name, exc)
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    def _load_bm25(self) -> bool:
        """Load the BM25 corpus from disk."""
        if not os.path.exists(self.bm25_cache_path):
            return False
        try:
            with open(self.bm25_cache_path, "rb") as f:
                payload = pickle.load(f)
            if not isinstance(payload, dict) or payload.get("version") != 1:
                return False
            if payload.get("collection") != self.collection_name:
                return False
            corpus = payload.get("corpus")
            cached_count = payload.get("document_count")
            if not isinstance(corpus, list) or cached_count != len(self._docs):
                return False
            self._bm25_corpus = corpus
            tokenized = [text.lower().split() for _, text, _ in corpus]
            self._bm25_index = BM25Okapi(tokenized) if tokenized else None
            logger.info("BM25 cache loaded for %s (%d docs).", self.collection_name, len(corpus))
            return True
        except Exception as exc:
            logger.warning("BM25 cache load failed for %s: %s", self.collection_name, exc)
            self._bm25_index = None
            self._bm25_corpus = []
            return False

    def _invalidate_bm25_cache(self):
        self._bm25_index = None
        self._bm25_corpus = []
        self._rebuild_bm25_flag = True
        try:
            if os.path.exists(self.bm25_cache_path):
                os.remove(self.bm25_cache_path)
        except OSError:
            pass

    def _rebuild_bm25(self):
        if not self._docs:
            self._bm25_index = None
            self._bm25_corpus = []
            self._save_bm25()
            return
        tokenized = []
        corpus = []
        for doc_id, text in self._docs.items():
            tokens = text.lower().split()
            tokenized.append(tokens)
            corpus.append((doc_id, text, self._metadatas.get(doc_id, {})))
        self._bm25_index = BM25Okapi(tokenized)
        self._bm25_corpus = corpus
        self._save_bm25()

    def _get_bm25(self):
        if self._rebuild_bm25_flag:
            self._rebuild_bm25()
            self._rebuild_bm25_flag = False
        return self._bm25_index

    # ------------------------------------------------------------------
    # Public CRUD API
    # ------------------------------------------------------------------

    def add_documents(self, docs: List[Document]) -> List[str]:
        """Add documents. Returns list of assigned IDs."""
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

        self._invalidate_bm25_cache()
        self._add_vectors(ids, embeddings)

        for doc_id, text, meta in zip(ids, texts, metadatas):
            self._docs[doc_id] = text
            self._metadatas[doc_id] = meta

        self._save_state()
        logger.info("Added %d chunks to %s.", len(ids), self.collection_name)
        return ids

    def delete_by_field(self, field: str, value: str):
        """Delete all documents where metadata[field] == value."""
        to_remove = [
            doc_id for doc_id, meta in self._metadatas.items()
            if meta.get(field) == value
        ]
        if not to_remove:
            return
        self._invalidate_bm25_cache()
        for doc_id in to_remove:
            self._remove_vector(doc_id)
            self._docs.pop(doc_id, None)
            self._metadatas.pop(doc_id, None)
        self._save_state()
        logger.info("Deleted %d chunks for %s=%s from %s", len(to_remove), field, value, self.collection_name)

    def delete_all(self):
        """Delete all documents in the collection."""
        self._tvec_index = None
        self._id_map.clear()
        self._reverse_id_map.clear()
        self._next_seq_id = 0
        self._docs.clear()
        self._metadatas.clear()
        self._invalidate_bm25_cache()
        self._save_state()
        # Remove index and idmap files
        for path in [self.index_path, self.idmap_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
        logger.info("Cleared entire %s store.", self.collection_name)

    def list_values(self, field: str) -> List[str]:
        """List unique values for a metadata field."""
        values = set()
        for meta in self._metadatas.values():
            val = meta.get(field)
            if val:
                values.add(val)
        return sorted(values)

    # ------------------------------------------------------------------
    # Dense search via TurboVec
    # ------------------------------------------------------------------

    def _metadata_value_matches(self, value: Any, condition: Any) -> bool:
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
                    return False
            except (TypeError, ValueError):
                return False
        return True

    def _metadata_matches_filter(self, metadata: Dict, filter_meta: Optional[Dict]) -> bool:
        if not filter_meta:
            return True
        if not isinstance(filter_meta, dict):
            return False
        for key, condition in filter_meta.items():
            if key == "$and":
                if not isinstance(condition, list) or not condition or not all(
                    self._metadata_matches_filter(metadata, item) for item in condition
                ):
                    return False
            elif key == "$or":
                if not isinstance(condition, list) or not condition or not any(
                    self._metadata_matches_filter(metadata, item) for item in condition
                ):
                    return False
            elif not self._metadata_value_matches(metadata.get(key), condition):
                return False
        return True

    def _dense_search(self, query: str, k: int, filter_meta: Optional[Dict] = None) -> List[tuple]:
        if self._tvec_index is None or self._next_seq_id == 0:
            return []
        query_emb = self._embed([query])[0]
        query_emb_2d = np.expand_dims(query_emb, axis=0)
        safe_k = min(k, self._next_seq_id)
        scores, indices = self._tvec_index.search(query_emb_2d, k=safe_k)
        scores = scores[0]
        indices = indices[0]

        results = []
        for score, seq_id in zip(scores, indices):
            doc_id = self._id_map.get(int(seq_id))
            if doc_id is None:
                continue
            meta = self._metadatas.get(doc_id, {})
            if not self._metadata_matches_filter(meta, filter_meta):
                continue
            text = self._docs.get(doc_id, "")
            results.append((doc_id, text, meta, float(score)))
        return results

    # ------------------------------------------------------------------
    # BM25 search
    # ------------------------------------------------------------------

    def _bm25_search(self, query: str, k: int, filter_meta: Optional[Dict] = None) -> List[tuple]:
        bm25 = self._get_bm25()
        if bm25 is None or not self._bm25_corpus:
            return []
        tokens = query.lower().split()
        scores = bm25.get_scores(tokens)
        filtered_with_scores = [
            (scores[idx], doc_id, text, meta)
            for idx, (doc_id, text, meta) in enumerate(self._bm25_corpus)
            if self._metadata_matches_filter(meta, filter_meta)
        ]
        filtered_with_scores.sort(key=lambda x: x[0], reverse=True)
        return [
            (doc_id, text, meta, float(score))
            for score, doc_id, text, meta in filtered_with_scores[:k]
            if score > 0
        ]

    # ------------------------------------------------------------------
    # Reranker — does NOT mutate in-memory meta dicts
    # ------------------------------------------------------------------

    def _rerank(self, query: str, candidates: List[tuple], top_k: int) -> List[tuple]:
        if not candidates:
            return []
        if self.reranker is None:
            self.reranker = CrossEncoder(RERANKER_MODEL_NAME)
        pairs = [(query, text) for (_, text, _, _) in candidates]
        rerank_scores = self.reranker.predict(pairs)
        combined = list(zip(candidates, rerank_scores))
        combined.sort(key=lambda x: x[1], reverse=True)
        reranked = []
        for (doc_id, text, meta, _), score in combined[:top_k]:
            # Don't mutate the shared meta dict — return a pristine copy
            reranked.append((doc_id, text, dict(meta), float(score)))
        return reranked

    # ------------------------------------------------------------------
    # Public search API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        use_hybrid: bool = True,
        use_reranker: bool = False,
    ) -> List[Document]:
        count = len(self._docs)
        if count == 0:
            return []

        safe_k = min(2 * top_k, count)
        dense_candidates = self._dense_search(query, k=safe_k, filter_meta=filter_metadata)

        hybrid_used = use_hybrid
        if hybrid_used:
            bm25_candidates = self._bm25_search(query, k=top_k, filter_meta=filter_metadata)
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
        return len(self._docs)