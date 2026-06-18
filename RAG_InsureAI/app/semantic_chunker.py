"""
semantic_chunker.py — Embedding-based semantic chunking for InsureHub RAG.

Replaces fixed-size character/word splitting with boundaries that follow
the actual meaning of the text.

How it works
------------
1. Split the input into sentences (or, for unpunctuated YouTube/Whisper
   transcripts, fixed word-window pseudo-sentences — same approach already
   used by context_compressor.py).
2. Embed each sentence blended with a small window of its neighbors (using
   the same BGE model already loaded by TurboVec), to smooth out noise from
   very short sentences.
3. Compute cosine distance between every pair of consecutive sentence
   embeddings.
4. Wherever that distance spikes above a percentile threshold (a topic
   shift), cut a new chunk.
5. Hard min/max character guards prevent pathological output: chunks below
   `min_chunk_chars` get merged into a neighbor, and a chunk is force-cut at
   `max_chunk_chars` even if no semantic breakpoint occurred (protects the
   LLM context window on long, topically-uniform stretches of text).
6. Sliding-window overlap: the last `overlap_chars` of each chunk are
   prepended to the next chunk so retrieval never misses facts that straddle
   a boundary (same mechanic as LangChain RecursiveCharacterTextSplitter).

Target sizes (recommended caller config)
-----------------------------------------
  chunk_size    = 3000  chars  ≈ 500 words  — enough context per chunk
  overlap_chars = 600   chars  ≈ 100 words  — bridges cross-boundary facts
  min_chunk_chars = 900 chars  ≈ 150 words  — prevents tiny orphan chunks

YouTube pseudo-sentence window
--------------------------------
  Auto-generated captions have no punctuation, so we split on fixed word
  windows (40 words each) as pseudo-sentences for the embedding step.
  40-word windows give ~12-13 pseudo-sentences per 500-word chunk, which
  is enough variety for the cosine-distance boundary detector to work well.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any, List

import numpy as np
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Sentence must have at least this many chars to be embedded / kept.
_MIN_SENT_CHARS = 25

# Word-window size for YouTube/Whisper transcripts (no punctuation).
# 40 words ≈ 240 chars — large enough for meaningful embeddings,
# small enough to give the boundary detector ~12 windows per 500-word chunk.
_YT_WORD_WINDOW = 40

_DEFAULT_EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")
_default_model: Any = None


def _get_default_embed_model() -> Any:
    """
    Lazy-loaded fallback embedding model, used only if a caller doesn't pass
    one in. Prefer passing the shared TurboVec embed_model instead — this
    fallback loads a second copy of the model into memory.
    """
    global _default_model
    if _default_model is None:
        from sentence_transformers import SentenceTransformer
        logger.warning(
            "[SemanticChunker] No embed_model provided — lazy-loading a "
            "second copy of '%s'. Pass the shared TurboVec embed_model to "
            "avoid the extra memory cost.",
            _DEFAULT_EMBED_MODEL_NAME,
        )
        _default_model = SentenceTransformer(_DEFAULT_EMBED_MODEL_NAME)
    return _default_model


def _split_sentences(text: str, for_youtube: bool = False) -> List[str]:
    """
    Split text into sentences.

    For normal text: uses punctuation boundaries with abbreviation protection.
    For YouTube transcripts (for_youtube=True): auto-generated captions have
    no punctuation, so we fall back to fixed word-window chunks (_YT_WORD_WINDOW
    words each) which act as pseudo-sentences for the embedding step. If
    punctuation splitting produces fewer than 2 sentences, the word-window
    fallback is always tried regardless of for_youtube.
    """
    abbrev = re.sub(
        r'\b(Mr|Mrs|Ms|Dr|Prof|vs|etc|e\.g|i\.e|fig|no|pg|pp|vol|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\.',
        r'\1<DOT>',
        text,
        flags=re.IGNORECASE,
    )
    abbrev = re.sub(r'(\d+)\.(\d)', r'\1<DOT>\2', abbrev)
    raw = re.split(r'(?<=[.!?])\s+(?=[A-Z\d\"\'\(])', abbrev)
    punct_sentences = [
        s.replace('<DOT>', '.').strip()
        for s in raw
        if len(s.replace('<DOT>', '').strip()) >= _MIN_SENT_CHARS
    ]

    if len(punct_sentences) >= 2 and not for_youtube:
        return punct_sentences

    words = text.split()
    if len(words) >= 10:
        window_sentences = []
        for i in range(0, len(words), _YT_WORD_WINDOW):
            chunk = " ".join(words[i:i + _YT_WORD_WINDOW])
            if len(chunk) >= _MIN_SENT_CHARS:
                window_sentences.append(chunk)
        if len(window_sentences) >= 2:
            return window_sentences

    return punct_sentences if punct_sentences else [text]


class SemanticChunker:
    """
    Embedding-based semantic chunker with sliding-window overlap.

    Parameters
    ----------
    embed_model : SentenceTransformer
        Shared embedding model (pass the same BGE model used elsewhere —
        e.g. ChromaVectorStore.embed_model — to avoid loading a duplicate).
    breakpoint_percentile : float
        A sentence-to-sentence distance must be in the top X percentile of
        all distances within the document to be treated as a chunk boundary.
        Higher = fewer, larger chunks. Default 90.
    buffer_size : int
        Number of neighboring sentences blended with each sentence before
        embedding, to stabilize the embedding of very short sentences.
    min_chunk_chars : int
        Chunks below this size are merged into a neighbor.
        Recommended: 900 chars ≈ 150 words.
    max_chunk_chars : int
        Hard ceiling — force a cut even without a semantic breakpoint once a
        chunk reaches this size, so one topically-uniform stretch of text
        can't grow unbounded.
        Recommended: 3000 chars ≈ 500 words.
    overlap_chars : int
        Number of characters from the END of the previous chunk to prepend
        to the START of the next chunk. This sliding-window overlap ensures
        facts that straddle a chunk boundary are retrievable from either
        side. Set to 0 to disable overlap.
        Recommended: 600 chars ≈ 100 words.
    """

    def __init__(
        self,
        embed_model: Any = None,
        breakpoint_percentile: float = 90.0,
        buffer_size: int = 1,
        min_chunk_chars: int = 900,
        max_chunk_chars: int = 3000,
        overlap_chars: int = 600,
    ):
        self._model = embed_model
        self._pct = breakpoint_percentile
        self._buffer = buffer_size
        self._min_chars = min_chunk_chars
        self._max_chars = max_chunk_chars
        self._overlap = overlap_chars

    def _model_or_default(self) -> Any:
        return self._model if self._model is not None else _get_default_embed_model()

    # ------------------------------------------------------------------
    # Core algorithm
    # ------------------------------------------------------------------

    def split_text(self, text: str, for_youtube: bool = False) -> List[str]:
        """
        Split text into semantically coherent chunks with overlap.

        Pass 1 — semantic boundary detection:
          Embeds sentences, computes cosine distances, cuts at distance spikes.
          Hard min/max char guards prevent pathological chunk sizes.

        Pass 2 — sliding-window overlap:
          Prepends the last `overlap_chars` of each chunk onto the start of
          the next chunk. This is done on the *text* level (after Pass 1
          assembles chunks from sentences) so the overlap is always a clean
          word boundary rather than a mid-sentence cut.
        """
        sentences = _split_sentences(text, for_youtube=for_youtube)
        if len(sentences) < 2:
            return [text] if text.strip() else []

        # Blend each sentence with a small window of neighbors for a more
        # stable embedding (reduces noise from very short sentences).
        combined = []
        for i in range(len(sentences)):
            lo = max(0, i - self._buffer)
            hi = min(len(sentences), i + self._buffer + 1)
            combined.append(" ".join(sentences[lo:hi]))

        model = self._model_or_default()
        embeddings = model.encode(
            combined, normalize_embeddings=True, batch_size=32, show_progress_bar=False
        )

        sims = np.array([
            float(np.dot(embeddings[i], embeddings[i + 1]))
            for i in range(len(embeddings) - 1)
        ])
        distances = 1 - sims

        if len(distances) == 0:
            return [text]

        threshold = float(np.percentile(distances, self._pct))
        # Strict ">" — if every distance is identical (uniform topic), no
        # breakpoint should fire (using ">=" here would shatter every
        # sentence into its own chunk in that degenerate case).
        breakpoints = {i for i, d in enumerate(distances) if d > threshold}

        # ── Pass 1: build chunks by semantic boundaries + size guards ─────
        chunks: List[str] = []
        current: List[str] = [sentences[0]]
        current_len = len(sentences[0])

        for i in range(1, len(sentences)):
            sent = sentences[i]
            semantic_break = (i - 1) in breakpoints
            size_break = current_len + len(sent) + 1 > self._max_chars
            if (semantic_break or size_break) and current_len >= self._min_chars:
                chunks.append(" ".join(current))
                current = [sent]
                current_len = len(sent)
            else:
                current.append(sent)
                current_len += len(sent) + 1

        if current:
            chunks.append(" ".join(current))

        # Merge a too-small trailing chunk into its predecessor.
        if len(chunks) > 1 and len(chunks[-1]) < self._min_chars:
            chunks[-2] = chunks[-2] + " " + chunks[-1]
            chunks.pop()

        # ── Pass 2: apply sliding-window overlap ──────────────────────────
        # Skip if overlap disabled or only one chunk (nothing to overlap).
        if self._overlap > 0 and len(chunks) > 1:
            chunks = self._apply_overlap(chunks, self._overlap)

        return chunks

    @staticmethod
    def _apply_overlap(chunks: List[str], overlap_chars: int) -> List[str]:
        """
        Prepend the last `overlap_chars` of chunk[i-1] to the start of
        chunk[i] for every i > 0.

        Overlap is trimmed to a word boundary so the prepended text never
        starts mid-word. The separator between the overlap tail and the
        chunk body is a single space (consistent with how chunks were built
        from sentence lists).

        The first chunk is never modified — it has no predecessor.
        """
        result: List[str] = [chunks[0]]

        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            tail = prev[-overlap_chars:] if len(prev) > overlap_chars else prev

            # Trim to the nearest word boundary so we don't start mid-word.
            # Find the first space in tail (scan left-to-right) and drop
            # everything before it — that guarantees a clean word start.
            first_space = tail.find(" ")
            if first_space != -1:
                tail = tail[first_space + 1:]

            # Avoid duplicating content when a short chunk is entirely
            # covered by the overlap window.
            current = chunks[i]
            if tail and not current.startswith(tail[:30]):
                result.append(tail + " " + current)
            else:
                result.append(current)

        return result

    # ------------------------------------------------------------------
    # LangChain-compatible interface
    # ------------------------------------------------------------------

    def split_documents(self, docs: List[Document]) -> List[Document]:
        """
        Drop-in replacement for RecursiveCharacterTextSplitter.split_documents().
        Detects YouTube/Whisper content automatically and switches to the
        word-window sentence fallback for it.
        """
        result: List[Document] = []
        for doc in docs:
            is_youtube = (
                doc.metadata.get("doc_type") == "youtube"
                or "youtube" in str(doc.metadata.get("source_type", "")).lower()
                or "whisper" in str(doc.metadata.get("source_type", "")).lower()
            )
            pieces = self.split_text(doc.page_content, for_youtube=is_youtube)
            for idx, piece in enumerate(pieces):
                result.append(Document(
                    page_content=piece,
                    metadata={**doc.metadata, "chunk_index": idx, "chunking_method": "semantic"},
                ))
        return result