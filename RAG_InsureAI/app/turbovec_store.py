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
import re
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


# ──────────────────────────────────────────────────────────────────────────
# Process-wide shared models
# ──────────────────────────────────────────────────────────────────────────
# Every TurboVecStore (documents / videos / webpages) used to instantiate its
# own SentenceTransformer and its own CrossEncoder, so a single process ended
# up holding 3 redundant copies of each model in memory. Both models are
# stateless w.r.t. any one collection's data, so they're hoisted here as
# lazily-loaded, process-wide singletons and shared by reference across every
# TurboVecStore instance. Each model is still loaded at most once for the
# life of the process; `.encode()` / `.predict()` still run fresh on every
# call exactly as before — only the "create the model object" step is
# deduplicated.
_SHARED_EMBED_MODELS: Dict[str, SentenceTransformer] = {}
_SHARED_EMBED_LOCK = threading.Lock()

_shared_reranker: Optional[CrossEncoder] = None
_SHARED_RERANKER_LOCK = threading.Lock()


def _get_shared_embed_model(embed_model_name: str) -> SentenceTransformer:
    """
    Return the process-wide SentenceTransformer for *embed_model_name*,
    loading it on first use. Keyed by model name so a future caller that
    passes a different embed_model_name still gets correct (if separate)
    caching, while all current callers — which all use the same default
    model — share one instance.
    """
    model = _SHARED_EMBED_MODELS.get(embed_model_name)
    if model is not None:
        return model
    with _SHARED_EMBED_LOCK:
        model = _SHARED_EMBED_MODELS.get(embed_model_name)
        if model is None:
            model = SentenceTransformer(embed_model_name)
            model.max_seq_length = 512
            _SHARED_EMBED_MODELS[embed_model_name] = model
            logger.info("[SharedModels] embedding model loaded: %s", embed_model_name)
    return model


def _get_shared_reranker() -> CrossEncoder:
    """Return the process-wide CrossEncoder reranker, loading it on first use."""
    global _shared_reranker
    if _shared_reranker is not None:
        return _shared_reranker
    with _SHARED_RERANKER_LOCK:
        if _shared_reranker is None:
            _shared_reranker = CrossEncoder(RERANKER_MODEL_NAME)
            logger.info("[SharedModels] reranker loaded: %s", RERANKER_MODEL_NAME)
    return _shared_reranker


# CrossEncoder already caps at 512 tokens (BAAI/bge-reranker-base's
# model_max_length), but that cap is enforced AFTER tokenizing the full
# input — so a 4000-char chunk still pays full tokenization cost, and if it
# tokenizes to more than ~512 tokens the forward pass runs at the max
# sequence length every time. Measured in production: real KB chunks average
# 2000-4400 characters (500-1100+ tokens), consistently at or past the cap,
# while reranking calls on chunks that size took 5-10s for as few as 8-14
# candidates — the dominant cost in the whole request, dwarfing the actual
# LLM generation call (~0.4s to Groq). A short synthetic-text benchmark
# (150 chars) completed the same candidate counts in under a second,
# confirming chunk length — not candidate count — drives the latency.
# Truncate the text used for rerank SCORING only; the returned Document
# still carries the untruncated page_content, so the final LLM context is
# unaffected. 700 chars (~150-180 tokens) keeps a full topic-establishing
# paragraph — more than enough for a relevance judgment — while cutting
# forward-pass cost well below the 512-token cap.
_RERANK_TEXT_CHARS = 700

# Minimum word length counted as a "distinctive" query term when picking a
# rerank window — filters out short connective words (how, can, the, are)
# that appear everywhere and would swamp a genuinely rare match.
_MIN_KEYWORD_LEN = 4
# Stride for the sliding window scan — smaller than _RERANK_TEXT_CHARS so
# windows overlap and a match sitting near a boundary is never split across
# two candidate windows and missed by both.
_RERANK_WINDOW_STRIDE = _RERANK_TEXT_CHARS // 3


def _truncate_for_rerank(text: str, query: str = "") -> str:
    """Return up to _RERANK_TEXT_CHARS of *text* for the cross-encoder's
    scoring pass — same fixed budget as always (see the latency measurement
    above _RERANK_TEXT_CHARS), but choosing WHICH _RERANK_TEXT_CHARS window
    to send instead of blindly always taking the first one.

    Always slicing text[:_RERANK_TEXT_CHARS] silently hides everything past
    that point — for a real KB chunk (section chunker allows up to 2000
    chars, some run past that) the ONLY sentence that actually answers a
    given question can sit well beyond the cutoff. Confirmed live: "the
    policy... needs to be kept in safe custody and in the knowledge of the
    close relatives" — a direct, on-topic answer to "how can relatives be
    informed about the policy?" — sat at character 1426 of a 2097-char
    chunk. The reranker scored that (query, truncated-passage) pair 0.005-
    0.013 (near-zero) purely because it never saw the one relevant sentence,
    even though BM25 independently ranked this exact chunk #1 by keyword
    match on the same query — proof the content match was there, just
    invisible to the reranker.

    Fix: scan overlapping _RERANK_TEXT_CHARS windows and keep the one with
    the highest keyword-match weight — cheap pure-Python string scanning
    done once per candidate before tokenization, negligible next to the
    model forward pass that actually drives the measured latency. Each
    keyword is weighted by 1/(its count in the full chunk): a term that
    appears once (like "relatives" above) is far more distinctive — and a
    far stronger signal for WHERE the relevant content actually is — than
    one repeated throughout (like "policy", which would otherwise score
    every window roughly equally and drown out the rare match's signal).
    This is the same intuition as BM25's IDF weighting, approximated
    locally within the chunk instead of needing corpus-wide statistics.
    Ties prefer the later window over the default opening slice, since the
    scan order means a later window can only tie (not just lose to) the
    default if it independently contains an equally strong, different
    match — worth surfacing rather than defaulting back to position 0.
    Falls back to the plain start-of-text slice when the query is empty or
    carries no usable keywords (keeps existing well-behaved cases, e.g. a
    topic-establishing opening paragraph, exactly as before).
    """
    if not text:
        return text
    if len(text) <= _RERANK_TEXT_CHARS or not query:
        return text[:_RERANK_TEXT_CHARS]

    keywords = {w for w in re.findall(r"\w+", query.lower()) if len(w) >= _MIN_KEYWORD_LEN}
    if not keywords:
        return text[:_RERANK_TEXT_CHARS]

    text_lower = text.lower()
    weights = {w: 1.0 / text_lower.count(w) for w in keywords if w in text_lower}
    if not weights:
        return text[:_RERANK_TEXT_CHARS]

    def _window_score(window: str) -> float:
        return sum(wt for w, wt in weights.items() if w in window)

    best_start = 0
    best_score = _window_score(text_lower[:_RERANK_TEXT_CHARS])
    last_start = len(text) - _RERANK_TEXT_CHARS
    for start in list(range(0, last_start, _RERANK_WINDOW_STRIDE)) + [last_start]:
        score = _window_score(text_lower[start:start + _RERANK_TEXT_CHARS])
        if score >= best_score:
            best_score, best_start = score, start

    return text[best_start:best_start + _RERANK_TEXT_CHARS]


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

        # Embedding model — process-wide shared instance (see
        # _get_shared_embed_model). Every collection (documents/videos/
        # webpages) uses the same model name by default, so they all resolve
        # to the same underlying object instead of each loading their own copy.
        self.embed_model = _get_shared_embed_model(embed_model_name)
        self.embed_dim = _get_embed_dim(self.embed_model)
        self.bit_width = bit_width

        # Reranker — also process-wide shared (see _ensure_reranker below).
        # Kept as an instance attribute for backward compatibility with any
        # code that reads store.reranker directly, but it always points at
        # the same shared object once loaded.
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
            texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False,
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
            tokenized = [self._tokenize(text) for _, text, _ in corpus]
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
            tokens = self._tokenize(text)
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

    def get_metadata_summary(self, match_field: str, match_value: str) -> Dict[str, Any]:
        """Summarize metadata for all chunks where metadata[match_field] == match_value.

        Returns the chunk count plus unique values seen for 'policy_type' and
        'section', so callers (e.g. the eval API) can inspect what's stored
        without scanning chunks themselves.
        """
        policy_types = set()
        sections = set()
        chunk_count = 0
        for meta in self._metadatas.values():
            if meta.get(match_field) != match_value:
                continue
            chunk_count += 1
            policy_type = meta.get("policy_type")
            if policy_type:
                policy_types.add(policy_type)
            section = meta.get("section")
            if section:
                sections.add(section)
        return {
            match_field: match_value,
            "chunk_count": chunk_count,
            "policy_types": sorted(policy_types),
            "sections": sorted(sections),
        }

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
        # The ANN index is unaware of metadata filters — it returns the global
        # top-k by cosine similarity, and filters are applied AFTER.  When a
        # restrictive filter is active (e.g. doc_type=youtube) the target docs
        # may not appear in a small top-k because other source types rank higher.
        # Pre-fetch a large candidate pool so that filtered subsets still have
        # enough candidates to fill the requested k slots.
        if filter_meta:
            fetch_k = min(max(k * 15, 100), self._next_seq_id)
        else:
            fetch_k = min(k * 2, self._next_seq_id)
        safe_k = max(fetch_k, 1)
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

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        Tokenize text for BM25: lowercase + strip punctuation so that
        query tokens like 'ulip?' match document tokens like 'ulip'.
        Without this, every query ending in '?' gets zero BM25 score for
        the last keyword (e.g. 'What is a ULIP?' → token 'ulip?' never
        matches document token 'ulip').
        """
        return [re.sub(r'[^\w]', '', t) for t in text.lower().split()
                if re.sub(r'[^\w]', '', t)]

    def _bm25_search(self, query: str, k: int, filter_meta: Optional[Dict] = None) -> List[tuple]:
        bm25 = self._get_bm25()
        if bm25 is None or not self._bm25_corpus:
            return []
        tokens = self._tokenize(query)

        # When a metadata filter is active, score only the matching sub-corpus.
        # This prevents IDF weights from being diluted by unrelated documents
        # and gives more accurate BM25 scores within the filtered slice.
        if filter_meta:
            filtered_corpus = [
                (doc_id, text, meta)
                for doc_id, text, meta in self._bm25_corpus
                if self._metadata_matches_filter(meta, filter_meta)
            ]
            if not filtered_corpus:
                return []
            # Build a temporary BM25 index over just the filtered docs.
            tokenized = [self._tokenize(text) for _, text, _ in filtered_corpus]
            from rank_bm25 import BM25Okapi
            sub_bm25 = BM25Okapi(tokenized)
            scores = sub_bm25.get_scores(tokens)
            scored = [
                (float(scores[idx]), doc_id, text, meta)
                for idx, (doc_id, text, meta) in enumerate(filtered_corpus)
            ]
        else:
            scores = bm25.get_scores(tokens)
            scored = [
                (float(scores[idx]), doc_id, text, meta)
                for idx, (doc_id, text, meta) in enumerate(self._bm25_corpus)
            ]

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            (doc_id, text, meta, score)
            for score, doc_id, text, meta in scored[:k]
            if score > 0
        ]

    # ------------------------------------------------------------------
    # Reranker — does NOT mutate in-memory meta dicts
    # ------------------------------------------------------------------

    def _ensure_reranker(self):
        if self.reranker is None:
            self.reranker = _get_shared_reranker()

    def warmup(self):
        """
        Force-load all models and run a dummy inference pass so that JIT
        compilation and CUDA graph capture happen at startup, not on the
        first user query.
        """
        logger.info("[Warmup] starting embedding model warmup ...")
        self.embed_model.encode(
            ["warmup insurance policy claim coverage"],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        logger.info("[Warmup] embedding model ready")
        logger.info("[Warmup] loading cross-encoder reranker ...")
        self._ensure_reranker()
        self.reranker.predict([("warmup query", "warmup document text")])
        logger.info("[Warmup] reranker ready")

    def _rerank(self, query: str, candidates: List[tuple], top_k: int) -> List[tuple]:
        if not candidates:
            return []
        self._ensure_reranker()
        pairs = [(query, _truncate_for_rerank(c[1], query)) for c in candidates]
        rerank_scores = self.reranker.predict(pairs)
        combined = list(zip(candidates, rerank_scores))
        combined.sort(key=lambda x: x[1], reverse=True)
        reranked = []
        for c, score in combined[:top_k]:
            dense_score = c[4] if len(c) > 4 else c[3]
            reranked.append((c[0], c[1], dict(c[2]), float(score), dense_score))
        return reranked

    def rerank_documents(self, query: str, docs: List[Document], top_k: int) -> List[Document]:
        """
        Rerank a list of Documents in a single CrossEncoder.predict() call.

        Use this instead of per-search reranking when multiple searches have
        been merged and deduplicated — one call is far cheaper than N calls.
        """
        if not docs:
            return []
        self._ensure_reranker()
        pairs = [(query, _truncate_for_rerank(doc.page_content, query)) for doc in docs]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        result = []
        for doc, score in ranked[:top_k]:
            new_doc = Document(page_content=doc.page_content, metadata=dict(doc.metadata))
            new_doc.metadata["rerank_score"] = float(score)
            base_method = doc.metadata.get("retrieval_method", "dense")
            if "+rerank" not in base_method:
                new_doc.metadata["retrieval_method"] = base_method + "+rerank"
            result.append(new_doc)
        return result

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
            # Reciprocal Rank Fusion — combines dense and BM25 rank positions so a
            # chunk that scores high in BM25 (exact keyword match) but low in dense
            # is not penalised by keeping only the weak dense score.
            _RRF_K = 60
            dense_map = {
                doc_id: (rank, text, meta, score)
                for rank, (doc_id, text, meta, score) in enumerate(dense_candidates)
            }
            # Normalize BM25 scores to [0,1] so BM25-only hits get a
            # meaningful similarity value instead of 0.0
            _bm25_max = max((s for _, _, _, s in bm25_candidates), default=1.0) or 1.0
            bm25_map = {
                doc_id: (rank, text, meta, score / _bm25_max)
                for rank, (doc_id, text, meta, score) in enumerate(bm25_candidates)
            }
            n_dense = len(dense_candidates)
            n_bm25 = len(bm25_candidates)
            all_ids = set(dense_map) | set(bm25_map)
            merged = {}
            for doc_id in all_ids:
                d_rank = dense_map[doc_id][0] if doc_id in dense_map else n_dense
                b_rank = bm25_map[doc_id][0] if doc_id in bm25_map else n_bm25
                rrf = 1.0 / (_RRF_K + d_rank + 1) + 1.0 / (_RRF_K + b_rank + 1)
                if doc_id in dense_map:
                    _, text, meta, dense_score = dense_map[doc_id]
                else:
                    _, text, meta, bm25_norm = bm25_map[doc_id]
                    dense_score = bm25_norm  # normalized BM25 score as proxy
                merged[doc_id] = (doc_id, text, meta, rrf, dense_score)
            candidates = sorted(merged.values(), key=lambda x: x[3], reverse=True)
        else:
            candidates = dense_candidates

        reranker_used = use_reranker and len(candidates) > 1
        if reranker_used:
            candidates = self._rerank(query, candidates, top_k)
        else:
            candidates = candidates[:top_k]

        docs = []
        for candidate in candidates:
            doc_id, text, meta = candidate[0], candidate[1], candidate[2]
            orig_score = candidate[3]
            # For hybrid results, candidate[4] holds the original dense cosine score
            dense_score = candidate[4] if len(candidate) > 4 else orig_score
            doc = Document(page_content=text, metadata=dict(meta))
            doc.metadata["similarity"] = dense_score
            # After _rerank(), orig_score is the cross-encoder's relevance
            # score, not an RRF rank score — store it under the same
            # "rerank_score" key rerank_documents() uses for doc chunks, so
            # every downstream relevance gate sees it regardless of which
            # of the two reranking code paths produced it.
            if reranker_used:
                doc.metadata["rerank_score"] = orig_score
            else:
                doc.metadata["rrf_score"] = orig_score
            method = "hybrid" if hybrid_used else "dense"
            if reranker_used:
                method += "+rerank"
            doc.metadata["retrieval_method"] = method
            docs.append(doc)
        return docs

    def get_all_by_filter(self, filter_meta: Dict[str, Any]) -> List[Document]:
        """
        Return ALL documents that match *filter_meta* by scanning in-memory metadata.
        Unlike search(), this does NOT use the ANN index — it is guaranteed to find
        every document that satisfies the filter regardless of its embedding distance
        from any query.  Use it when you need completeness rather than relevance ranking
        (e.g. fetching all YouTube chunks by doc_type).
        """
        results: List[Document] = []
        for doc_id, text in self._docs.items():
            meta = self._metadatas.get(doc_id, {})
            if self._metadata_matches_filter(meta, filter_meta):
                results.append(Document(page_content=text, metadata={**meta}))
        return results

    def count(self) -> int:
        return len(self._docs)