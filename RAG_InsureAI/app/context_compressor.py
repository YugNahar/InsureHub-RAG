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
from typing import Any, List

import numpy as np
from langchain_core.documents import Document

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

        final: List[Document] = []

        for doc, alloc in zip(chunks, allocations):
            if alloc <= 0:
                continue

            text = doc.page_content

            # Chunk fits in its allocation — include it as-is, no compression.
            if len(text) <= alloc:
                final.append(doc)
                continue

            # Chunk is too large for its allocation — keep only the most
            # query-relevant sentences that fit within `alloc` chars.
            is_yt = (
                doc.metadata.get("doc_type") == "youtube"
                or "youtube" in str(doc.metadata.get("source_type", "")).lower()
                or "whisper" in str(doc.metadata.get("source_type", "")).lower()
            )
            sentences = _split_sentences(text, for_youtube=is_yt)

            if len(sentences) <= 1:
                # Single atomic sentence — hard-truncate at the nearest
                # sentence boundary rather than cutting mid-word.
                truncated = text[:alloc].rsplit('. ', 1)[0] + '…'
                final.append(Document(
                    page_content=truncated,
                    metadata={**doc.metadata, "hard_truncated": True},
                ))
                continue

            sent_embs = self._model.encode(
                sentences, normalize_embeddings=True, batch_size=32, show_progress_bar=False
            )
            scores = np.dot(sent_embs, query_emb)
            ranked = np.argsort(scores)[::-1]

            kept_indices: set = set()
            used = 0
            for idx in ranked:
                s = sentences[int(idx)]
                if used + len(s) + 2 <= alloc:
                    kept_indices.add(int(idx))
                    used += len(s) + 2
                if used >= alloc:
                    break

            if not kept_indices:
                # Even the single best sentence is too long — hard-truncate it.
                best = sentences[int(ranked[0])]
                truncated = best[:alloc] + '…'
                final.append(Document(
                    page_content=truncated,
                    metadata={**doc.metadata, "hard_truncated": True},
                ))
                continue

            # Re-assemble in original document order (not relevance order)
            # so the LLM reads coherent prose, not a jumbled ranking.
            kept_text = ' '.join(sentences[i] for i in sorted(kept_indices))
            final.append(Document(
                page_content=kept_text,
                metadata={**doc.metadata, "budget_trimmed": True},
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