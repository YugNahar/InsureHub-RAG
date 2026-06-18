"""
Disk-persisted KV cache for RAG query results with semantic similarity lookup.

Two lookup modes
----------------
1. Exact  — SHA-256 of (query + config flags + sorted source list).
            Instant O(1) hit for identical re-queries.
2. Semantic — cosine similarity between the incoming query embedding and all
              stored query embeddings.  Returns the best match above a
              configurable threshold (default 0.92).  Catches rephrased queries
              that mean the same thing ("What is the grace period?" vs "How
              many days is the grace period?").

Cache key still includes the sorted source list so adding a new document
automatically invalidates all cached answers for the same query.

TTL: configurable, default 3600 s (1 hour).
Eviction: lazy on read + LRU when max_entries is reached.
"""
import hashlib
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_CACHE_VERSION = 2          # bumped because entry schema changed (added query_embedding)
_SEMANTIC_THRESHOLD_DEFAULT = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.78"))


class QueryKVCache:
    """
    Disk-persisted semantic KV cache.

    Parameters
    ----------
    cache_path    : absolute path to the JSON file.
    ttl_seconds   : entry lifetime (default 3600 s).
    max_entries   : LRU eviction threshold (default 500).
    sem_threshold : cosine similarity threshold for semantic hits (0–1).
    """

    def __init__(
        self,
        cache_path: str,
        ttl_seconds: int = 3600,
        max_entries: int = 500,
        sem_threshold: float = _SEMANTIC_THRESHOLD_DEFAULT,
    ):
        self._path = cache_path
        self._ttl  = ttl_seconds
        self._max  = max_entries
        self._sem_threshold = sem_threshold

        # key -> {value, ts, hits, ts_last_hit, query_text, query_embedding}
        self._data: Dict[str, Dict] = {}

        # In-memory embedding matrix for fast semantic search (rebuilt lazily)
        self._emb_matrix: Optional[np.ndarray] = None   # shape (N, D)
        self._emb_keys:   List[str] = []                # parallel list of keys
        self._emb_dirty = True                          # rebuild needed flag

        self._load()

    # ──────────────────────────────────────────────────────────────────────────
    # Key construction
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def make_key(
        query: str,
        top_k: int,
        use_hybrid: bool,
        use_reranker: bool,
        generate_answer: bool,
        run_ragas: bool,
        sources: List[str],
    ) -> str:
        """Deterministic exact-match key.  Changing any parameter gives a new key."""
        payload = json.dumps(
            {
                "q": query.strip().lower(),
                "k": top_k,
                "h": use_hybrid,
                "r": use_reranker,
                "a": generate_answer,
                "g": run_ragas,
                "s": sorted(sources),
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    # ──────────────────────────────────────────────────────────────────────────
    # Public API — exact
    # ──────────────────────────────────────────────────────────────────────────

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Return cached value or None if missing / expired."""
        entry = self._data.get(key)
        if entry is None:
            return None
        if time.time() - entry["ts"] > self._ttl:
            del self._data[key]
            self._emb_dirty = True
            return None
        entry["hits"] += 1
        entry["ts_last_hit"] = time.time()
        return entry["value"]

    def put(
        self,
        key: str,
        value: Dict[str, Any],
        query_embedding: Optional[np.ndarray] = None,
        query_text: str = "",
    ) -> None:
        """Store *value* under *key*, optionally with an embedding for semantic lookup."""
        self._evict_if_needed()
        entry: Dict[str, Any] = {
            "value":       value,
            "ts":          time.time(),
            "hits":        0,
            "ts_last_hit": time.time(),
            "query_text":  query_text,
        }
        if query_embedding is not None:
            entry["query_embedding"] = query_embedding.tolist()
        self._data[key] = entry
        self._emb_dirty = True
        self._save()

    # ──────────────────────────────────────────────────────────────────────────
    # Public API — semantic
    # ──────────────────────────────────────────────────────────────────────────

    def semantic_get(
        self,
        query_embedding: np.ndarray,
        threshold: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Find the cached result whose stored query embedding is most similar
        to *query_embedding* (cosine similarity).

        Returns the cached value if best similarity ≥ threshold, else None.
        Skips entries that have no stored embedding or are expired.
        """
        if not self._data:
            return None

        thr = threshold if threshold is not None else self._sem_threshold
        self._rebuild_emb_matrix()

        if self._emb_matrix is None or self._emb_matrix.shape[0] == 0:
            return None

        # query_embedding must be unit-normalised (BGE always returns normalised)
        qe = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        sims: np.ndarray = self._emb_matrix @ qe

        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        if best_sim < thr:
            return None

        best_key = self._emb_keys[best_idx]
        logger.info(
            "[KVCache] semantic hit  sim=%.3f  cached_query=%r",
            best_sim,
            self._data.get(best_key, {}).get("query_text", "")[:80],
        )
        return self.get(best_key)   # goes through TTL check + hit counter

    # ──────────────────────────────────────────────────────────────────────────
    # Public API — maintenance
    # ──────────────────────────────────────────────────────────────────────────

    def invalidate(self, key: str) -> None:
        self._data.pop(key, None)
        self._emb_dirty = True
        self._save()

    def flush(self) -> int:
        now = time.time()
        before = len(self._data)
        self._data = {k: v for k, v in self._data.items() if now - v["ts"] <= self._ttl}
        removed = before - len(self._data)
        if removed:
            self._emb_dirty = True
            self._save()
        return removed

    def clear(self) -> None:
        self._data.clear()
        self._emb_matrix = None
        self._emb_keys   = []
        self._emb_dirty  = False
        self._save()

    def stats(self) -> Dict[str, Any]:
        now = time.time()
        live    = sum(1 for v in self._data.values() if now - v["ts"] <= self._ttl)
        expired = len(self._data) - live
        total_hits = sum(v.get("hits", 0) for v in self._data.values())
        sem_entries = sum(1 for v in self._data.values() if "query_embedding" in v)
        return {
            "total_entries":    len(self._data),
            "live_entries":     live,
            "expired_entries":  expired,
            "semantic_entries": sem_entries,
            "total_hits":       total_hits,
            "ttl_seconds":      self._ttl,
            "max_entries":      self._max,
            "sem_threshold":    self._sem_threshold,
            "cache_path":       self._path,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────────────────────────────────────

    def _rebuild_emb_matrix(self) -> None:
        if not self._emb_dirty:
            return
        now = time.time()
        keys, vecs = [], []
        for k, entry in self._data.items():
            if now - entry["ts"] > self._ttl:
                continue
            emb = entry.get("query_embedding")
            if emb is None:
                continue
            keys.append(k)
            vecs.append(emb)

        if vecs:
            mat = np.array(vecs, dtype=np.float32)
            norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-10
            self._emb_matrix = mat / norms
        else:
            self._emb_matrix = None
        self._emb_keys  = keys
        self._emb_dirty = False

    def _evict_if_needed(self) -> None:
        if len(self._data) < self._max:
            return
        now = time.time()
        expired = [k for k, v in self._data.items() if now - v["ts"] > self._ttl]
        for k in expired:
            del self._data[k]
        if len(self._data) < self._max:
            self._emb_dirty = True
            return
        oldest = min(self._data, key=lambda k: self._data[k].get("ts_last_hit", self._data[k]["ts"]))
        del self._data[oldest]
        self._emb_dirty = True

    def _save(self) -> None:
        tmp = self._path + ".tmp"
        try:
            os.makedirs(os.path.dirname(self._path), exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"version": _CACHE_VERSION, "entries": self._data}, f)
            os.replace(tmp, self._path)
        except Exception as exc:
            logger.warning("[KVCache] save failed: %s", exc)
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass

    def _load(self) -> None:
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                d = json.load(f)
            if d.get("version") != _CACHE_VERSION:
                logger.info("[KVCache] schema version changed — starting fresh")
                return
            self._data = d.get("entries", {})
            self._emb_dirty = True
            logger.info("[KVCache] loaded %d entries from %s", len(self._data), self._path)
        except Exception as exc:
            logger.warning("[KVCache] load failed (%s) — starting fresh", exc)
