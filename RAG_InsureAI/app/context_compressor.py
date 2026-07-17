"""
Context Compressor — sentence-level semantic compression for RAG context.

Problem: retrieved chunks may contain a lot of surrounding text irrelevant to
the specific query.  Sending full chunks wastes the LLM's token budget and can
push key information out of the context window.

Solution: embed the query and every sentence in each chunk using the same BGE
model already loaded by TurboVec.  Keep only the sentences whose cosine
similarity to the query exceeds a threshold.  Sentences are kept in their
original document order so the compressed output remains coherent prose.

Properties:
- Zero extra model downloads — reuses the in-process SentenceTransformer.
- Zero extra LLM calls — pure embedding arithmetic.
- Graceful fallback — if compression leaves < MIN_CHARS chars, the original
  chunk is returned untouched.
- Per-chunk stats stored in metadata so the UI can show compression ratio.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

import numpy as np
from langchain_core.documents import Document

from turbovec_store import _rerank_metadata_prefix

logger = logging.getLogger(__name__)

# Sentence must have at least this many chars to be embedded / kept.
_MIN_SENT_CHARS = 25

# If compressed result is shorter than this we fall back to the original.
_MIN_COMPRESSED_CHARS = 60


def _split_sentences(text: str, for_youtube: bool = False) -> List[str]:
    """
    Split text into sentences.

    For normal text: uses punctuation boundaries with abbreviation protection.
    For YouTube transcripts (for_youtube=True): auto-generated captions have no
    punctuation, so we fall back to fixed word-window chunks (25 words each).
    If punctuation splitting produces fewer than 2 sentences, the word-window
    fallback is always tried regardless of for_youtube.
    """
    # ── Punctuation-based splitting (PDFs, webpages, hand-typed docs) ────────
    abbrev = re.sub(
        r'\b(Mr|Mrs|Ms|Dr|Prof|vs|etc|e\.g|i\.e|fig|no|pg|pp|vol|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\.',
        r'\1<DOT>',
        text,
        flags=re.IGNORECASE,
    )
    abbrev = re.sub(r'(\d+)\.(\d)', r'\1<DOT>\2', abbrev)
    raw = re.split(r'(?<=[.!?])\s+(?=[A-Z\d\"\'\(])', abbrev)
    punct_sentences = [s.replace('<DOT>', '.').strip() for s in raw if len(s.replace('<DOT>', '').strip()) >= _MIN_SENT_CHARS]

    # If we got real sentence boundaries and this isn't a YouTube chunk, done.
    if len(punct_sentences) >= 2 and not for_youtube:
        return punct_sentences

    # ── Word-window fallback (YouTube / un-punctuated transcripts) ────────────
    # Group words into ~25-word pseudo-sentences so the compressor can
    # compare individual idea units rather than the whole blob at once.
    words = text.split()
    if len(words) >= 10:
        window_sentences = []
        for i in range(0, len(words), 25):
            chunk = " ".join(words[i:i + 25])
            if len(chunk) >= _MIN_SENT_CHARS:
                window_sentences.append(chunk)
        if len(window_sentences) >= 2:
            return window_sentences

    # Last resort: return whatever punctuation splitting gave us (even 1 item)
    return punct_sentences if punct_sentences else [text]


class ContextCompressor:
    """
    Compress retrieved chunks to query-relevant sentences.

    Parameters
    ----------
    embed_model : SentenceTransformer
        The embedding model shared with TurboVec (BAAI/bge-base-en-v1.5).
    similarity_threshold : float
        Sentences with cosine similarity ≥ this value are kept.
        Lower = more aggressive inclusion (more context, less filtering).
    min_sentences : int
        Always keep at least this many top-scoring sentences per chunk even
        if they fall below the threshold — prevents empty compression.
    max_sentences : int
        Hard cap on sentences kept per chunk to control token usage.
    max_chars_per_chunk : int
        If the original chunk is ≤ this many chars, skip compression (already
        small enough to fit the context window comfortably).
    """

    def __init__(
        self,
        embed_model: Any,
        similarity_threshold: float = 0.38,
        min_sentences: int = 2,
        max_sentences: int = 10,
        max_chars_per_chunk: int = 600,
    ):
        self._model = embed_model
        self._threshold = similarity_threshold
        self._min_sents = min_sentences
        self._max_sents = max_sentences
        self._skip_below = max_chars_per_chunk
        # KB chunk text is static (sourced from ingested documents), so the
        # same chunk's sentences were being re-split and re-embedded from
        # scratch on every request that retrieved it — measured live at
        # ~8s for a single request's worth of sentences on this
        # deployment's CPU (transformer inference here is generally slow,
        # see the reranker-serialization comment in multi_source_rag.py's
        # ask_stream for a related measurement). Only the QUERY changes
        # between requests, not the chunk content, so caching each chunk's
        # (sentences, embeddings) by its own text means that cost is paid
        # once ever per chunk instead of once per request. Bounded FIFO,
        # not LRU: simpler, and good enough since the KB itself is a
        # bounded, mostly-static set of chunks.
        self._sent_cache: Dict[tuple, tuple] = {}
        self._sent_cache_max = 3000

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compress(self, query: str, chunks: List[Document]) -> List[Document]:
        """
        Return a new list of Documents where each chunk's text is compressed
        to the sentences most relevant to *query*.

        Chunks shorter than `max_chars_per_chunk` are returned as-is.
        Chunks where compression removes too much text are returned as-is.
        """
        if not chunks or not query.strip():
            return chunks

        query_emb: np.ndarray = self._model.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        )[0]

        results: List[Document] = []
        total_saved = 0

        for chunk in chunks:
            text = chunk.page_content
            original_len = len(text)

            # Skip small chunks — no benefit from compressing them.
            if original_len <= self._skip_below:
                results.append(chunk)
                continue

            compressed_doc = self._compress_one(chunk, query_emb)
            saved = original_len - len(compressed_doc.page_content)
            total_saved += max(saved, 0)
            results.append(compressed_doc)

        if total_saved > 0:
            logger.info(
                "[Compressor] compressed %d chunks — saved ~%d chars total",
                len(chunks), total_saved,
            )

        return results

    def compress_to_budget(
        self,
        query: str,
        chunks: List[Document],
        max_total_chars: int,
    ) -> List[Document]:
        """
        Trim chunks so their combined character count fits within
        *max_total_chars*.

        This method does NOT compress individual chunks that are already
        smaller than the budget — it only trims when the aggregate total
        exceeds the LLM's context window.

        Steps:
          1. Give every chunk a fair-share allocation (max_total_chars / N)
             up front; any share left unused by a chunk smaller than its
             allocation rolls over to chunks that need more, highest-
             relevance-first (caller's pre-sort order).
          2. For a chunk that fits within its final allocation, include it
             as-is.
          3. For a chunk that exceeds its allocation, keep only the most
             query-relevant sentences from it that still fit.
          4. As a last resort, hard-truncate at a sentence boundary.

        Fair-share replaces a strict "fill in rank order until the budget
        runs out" pass — confirmed live: 10 chunks competing for a
        6000-char budget left the 4th chunk with 20 characters and chunks
        5-10 with none at all, discarding entire retrieved sources outright
        even though they'd been relevant enough to be retrieved in the
        first place. Every chunk now gets at least an equal cut up front,
        so a wide candidate pool (detailed mode retrieves more chunks
        specifically to have more material to draw from) actually gets
        used instead of being crushed down to whichever 2-3 chunks
        happened to sort first. This gracefully degrades back to the old
        behavior whenever there's enough room for everyone — the
        redistribution step still lets top-ranked chunks grow to their
        full size first, exactly as before, once every chunk's minimum is
        covered.
        """
        total = sum(len(d.page_content) for d in chunks)
        if total <= max_total_chars:
            # Everything already fits — return untouched, zero embedding cost.
            return chunks

        logger.info(
            "[Compressor] budget trim: %d chars across %d chunks → target %d chars",
            total, len(chunks), max_total_chars,
        )

        query_emb: np.ndarray = self._model.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        )[0]

        sizes = [len(d.page_content) for d in chunks]
        n = len(chunks)
        fair_share = max_total_chars // n
        allocations = [min(s, fair_share) for s in sizes]
        leftover = max_total_chars - sum(allocations)
        for i, s in enumerate(sizes):
            if leftover <= 0:
                break
            if s > allocations[i]:
                extra = min(leftover, s - allocations[i])
                allocations[i] += extra
                leftover -= extra

        # Pass 1: classify each chunk without touching the model yet. For a
        # chunk needing sentence-level ranking, reuse its cached
        # (sentences, embeddings) if this exact chunk text was compressed
        # before — the KB itself doesn't change between requests, only the
        # query does, so cache hits are the common case after warm-up.
        # Only genuine cache misses get queued for the model; those are
        # still batched into one encode() call rather than one per chunk.
        _slots: List[Optional[Dict[str, Any]]] = [None] * n
        _all_sentences: List[str] = []
        _offsets: List[tuple] = []  # (slot_index, start, end) into _all_sentences

        for i, (doc, alloc) in enumerate(zip(chunks, allocations)):
            if alloc <= 0:
                continue

            text = doc.page_content

            # Chunk fits in its allocation — include it as-is, no compression.
            if len(text) <= alloc:
                _slots[i] = {"kind": "asis", "doc": doc}
                continue

            # Chunk is too large for its allocation — keep only the most
            # query-relevant sentences that fit within `alloc` chars.
            is_yt = (
                doc.metadata.get("doc_type") == "youtube"
                or "youtube" in str(doc.metadata.get("source_type", "")).lower()
                or "whisper" in str(doc.metadata.get("source_type", "")).lower()
            )
            _cache_key = (text, is_yt)
            _cached = self._sent_cache.get(_cache_key)
            if _cached is not None:
                sentences, sent_embs = _cached
                if len(sentences) <= 1:
                    truncated = text[:alloc].rsplit('. ', 1)[0] + '…'
                    _slots[i] = {"kind": "hard_truncate", "doc": doc, "text": truncated}
                    continue
                _slots[i] = {
                    "kind": "rank", "doc": doc, "alloc": alloc,
                    "sentences": sentences, "sent_embs": sent_embs,
                }
                continue

            sentences = _split_sentences(text, for_youtube=is_yt)

            if len(sentences) <= 1:
                # Single atomic sentence — hard-truncate at the nearest
                # sentence boundary rather than cutting mid-word.
                truncated = text[:alloc].rsplit('. ', 1)[0] + '…'
                _slots[i] = {
                    "kind": "hard_truncate",
                    "doc": doc,
                    "text": truncated,
                }
                self._sent_cache[_cache_key] = (sentences, None)
                continue

            # Embed an enriched version of each sentence (metadata prefix
            # only, never returned) so the embedding model has a chance to
            # know this sentence's policy_type/section — the same fix as
            # _rerank_metadata_prefix's docstring describes for the
            # cross-encoder reranker, applied here to the compressor's own
            # separate embedding-similarity sentence selection. Confirmed
            # live this same "broken hand" query dropped the one sentence
            # that actually answered it (a pre-existing-injury exclusion
            # clause) during compression even though the chunk containing
            # it had already survived retrieval and reranking — the raw
            # sentence's wording alone doesn't say "this is an exclusion,"
            # so it lost out to other sentences in the same chunk that
            # happened to share more surface vocabulary with the query.
            # `sentences` (unenriched) is what actually gets returned in
            # the final compressed text — only the embedding INPUT changes.
            _prefix = _rerank_metadata_prefix(doc.metadata)
            start = len(_all_sentences)
            _all_sentences.extend(_prefix + s for s in sentences)
            end = len(_all_sentences)
            _offsets.append((i, start, end))
            _slots[i] = {
                "kind": "rank",
                "doc": doc,
                "alloc": alloc,
                "sentences": sentences,
                "cache_key": _cache_key,
            }

        # Pass 2: one batched encode() call for every cache-miss sentence
        # collected above.
        if _all_sentences:
            _all_sent_embs = self._model.encode(
                _all_sentences, normalize_embeddings=True, batch_size=32, show_progress_bar=False
            )
        else:
            _all_sent_embs = None

        for slot_idx, start, end in _offsets:
            slot = _slots[slot_idx]
            sent_embs = _all_sent_embs[start:end]
            slot["sent_embs"] = sent_embs
            _cache_key = slot.pop("cache_key")
            if len(self._sent_cache) >= self._sent_cache_max:
                self._sent_cache.pop(next(iter(self._sent_cache)))
            self._sent_cache[_cache_key] = (slot["sentences"], sent_embs)

        for i, slot in enumerate(_slots):
            if slot is None or slot["kind"] != "rank":
                continue
            doc = slot["doc"]
            alloc = slot["alloc"]
            sentences = slot["sentences"]
            sent_embs = slot["sent_embs"]
            scores = np.dot(sent_embs, query_emb)
            ranked = np.argsort(scores)[::-1]

            # Guarantee the single highest-scoring sentence a slot before
            # the greedy walk below runs — a plain greedy walk in score
            # order skips any sentence that doesn't fit the REMAINING
            # budget and moves on, so an oversized #1 sentence can lose out
            # to several shorter, lower-relevance ones that happen to fit,
            # even though it individually outscores every one of them.
            # Confirmed live: a 760-char sentence containing the only
            # clause that actually answered "can I get insurance for my
            # broken hand" (a pre-existing-injury exclusion) scored highest
            # among its chunk's sentences but was silently dropped this
            # way, while several shorter, less relevant sentences from the
            # same chunk filled the budget instead — the generated answer
            # was never grounded in the one fact that mattered. If the top
            # sentence alone exceeds the whole allocation, it still wins:
            # hard-truncated, it's used as this chunk's entire compressed
            # result rather than being dropped for lesser complete ones.
            top_idx = int(ranked[0])
            top_sentence = sentences[top_idx]
            if len(top_sentence) + 2 > alloc:
                truncated = top_sentence[:alloc].rsplit(' ', 1)[0] + '…'
                _slots[i] = {"kind": "hard_truncate", "doc": doc, "text": truncated}
                continue

            kept_indices: set = {top_idx}
            used = len(top_sentence) + 2
            for idx in ranked[1:]:
                s = sentences[int(idx)]
                if used + len(s) + 2 <= alloc:
                    kept_indices.add(int(idx))
                    used += len(s) + 2
                if used >= alloc:
                    break

            # Re-assemble in original document order (not relevance order)
            # so the LLM reads coherent prose, not a jumbled ranking.
            kept_text = ' '.join(sentences[j] for j in sorted(kept_indices))
            _slots[i] = {
                "kind": "ranked_result",
                "doc": doc,
                "text": kept_text,
            }

        final: List[Document] = []
        for slot in _slots:
            if slot is None:
                continue
            if slot["kind"] == "asis":
                final.append(slot["doc"])
            elif slot["kind"] == "hard_truncate":
                final.append(Document(
                    page_content=slot["text"],
                    metadata={**slot["doc"].metadata, "hard_truncated": True},
                ))
            elif slot["kind"] == "ranked_result":
                final.append(Document(
                    page_content=slot["text"],
                    metadata={**slot["doc"].metadata, "budget_trimmed": True},
                ))

        return final

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compress_one(self, chunk: Document, query_emb: np.ndarray) -> Document:
        text = chunk.page_content
        is_yt = (
            chunk.metadata.get("doc_type") == "youtube"
            or "youtube" in str(chunk.metadata.get("source_type", "")).lower()
            or "whisper" in str(chunk.metadata.get("source_type", "")).lower()
        )
        sentences = _split_sentences(text, for_youtube=is_yt)

        if len(sentences) < 2:
            return chunk

        sent_embs = self._model.encode(
            sentences,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        )
        scores: np.ndarray = np.dot(sent_embs, query_emb)

        # Always keep the top min_sentences regardless of threshold
        ranked = list(np.argsort(scores)[::-1])
        keep: set[int] = set(int(i) for i in ranked[: self._min_sents])

        # Add any sentence that clears the threshold (up to max_sentences)
        for i, sc in enumerate(scores):
            if len(keep) >= self._max_sents:
                break
            if float(sc) >= self._threshold:
                keep.add(i)

        # Preserve original sentence order
        kept_text = ' '.join(sentences[i] for i in sorted(keep))

        if len(kept_text) < _MIN_COMPRESSED_CHARS:
            return chunk  # too aggressively compressed — use original

        compression_ratio = round(len(kept_text) / len(text), 2)
        return Document(
            page_content=kept_text,
            metadata={
                **chunk.metadata,
                "compressed": True,
                "compression_ratio": compression_ratio,
                "original_chars": len(text),
                "compressed_chars": len(kept_text),
            },
        )