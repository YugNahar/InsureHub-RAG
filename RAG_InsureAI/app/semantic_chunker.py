"""
semantic_chunker.py — Paragraph-based semantic chunker for InsureHub RAG.

Embedding model : BAAI/bge-base-en-v1.5  (same model used by TurboVec for retrieval)

Strategy (same for ALL content types — PDFs, YouTube, web pages):
1. Split text into paragraphs (blank lines → single newlines → 50-word windows).
2. Embed every paragraph with BAAI/bge-base-en-v1.5.
3. Greedy grouping — for each new paragraph compute its cosine similarity to
   the MEAN embedding of all paragraphs already in the current group:
     similarity >= 0.4  AND  group still fits in 500 words
         → same topic → add to current group
     similarity <  0.4  OR   group would exceed 500 words
         → topic shifted or too big → flush group as a chunk, start new group
   Using the group mean (centroid) means the decision considers ALL paragraphs
   in the group, not just the last one — so paragraphs 1-2-3-4-5 that all
   discuss the same concept get grouped into ONE chunk even though only
   consecutive pairs are directly compared by other methods.
4. Cap chunks at 500 words. Force a new chunk even without a topic shift.
5. Prepend the last 60 words of each chunk to the next chunk (overlap).
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any, List

import numpy as np
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# ── Tuning ──────────────────────────────────────────────────────────────────────
MAX_CHUNK_WORDS   = 500   # hard word ceiling per chunk
OVERLAP_WORDS     = 60    # words from previous chunk prepended to next
SIM_THRESHOLD     = 0.4   # cosine similarity floor — below this = topic shift
_MIN_PARA_CHARS   = 20    # drop blank / very short fragments

# ── Embedding model ─────────────────────────────────────────────────────────────
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")
_default_model: Any = None


def _get_default_embed_model() -> Any:
    global _default_model
    if _default_model is None:
        from sentence_transformers import SentenceTransformer
        logger.warning(
            "[SemanticChunker] No embed_model passed — lazy-loading '%s'. "
            "Pass the shared TurboVec embed_model to avoid a duplicate copy in memory.",
            EMBED_MODEL_NAME,
        )
        _default_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _default_model


# ── Step 1: paragraph splitting ─────────────────────────────────────────────────

def _split_paragraphs(text: str) -> List[str]:
    """
    Split text into paragraphs — the atomic units that get embedded.

    Tier 1 — blank-line splitting  (standard PDFs, handbooks, web pages)
    Tier 2 — single-newline splitting  (PDFs where blank lines are stripped)
    Tier 3 — fixed 50-word windows  (YouTube transcripts — no newlines at all)
    """
    # Tier 1: blank lines
    paras = [p.strip() for p in re.split(r'\n{2,}', text) if len(p.strip()) >= _MIN_PARA_CHARS]
    if len(paras) >= 2:
        return paras

    # Tier 2: single newlines
    lines = [l.strip() for l in text.split('\n') if len(l.strip()) >= _MIN_PARA_CHARS]
    if len(lines) >= 2:
        return lines

    # Tier 3: 50-word windows (YouTube / no-newline transcripts)
    words = text.split()
    if len(words) >= 20:
        windows: List[str] = []
        for i in range(0, len(words), 50):
            w = " ".join(words[i: i + 50])
            if len(w) >= _MIN_PARA_CHARS:
                windows.append(w)
        if len(windows) >= 2:
            return windows

    return [text.strip()] if text.strip() else []


# ── Page-merge helper ────────────────────────────────────────────────────────────
_PAGE_MARKER_RE = re.compile(r"<<<PAGE:([^>]+)>>>")


def _merge_pages(docs: List[Document]) -> List[Document]:
    """
    Merge per-page Documents from the same PDF/DOCX into one Document so
    topic boundaries are detected across page breaks (not forced at them).
    <<<PAGE:N>>> markers are embedded so page numbers can be recovered later.
    """
    if len(docs) <= 1:
        return docs

    groups: dict[str, List[Document]] = {}
    order: List[str] = []
    for doc in docs:
        src = doc.metadata.get("source") or doc.metadata.get("filename") or str(id(doc))
        if src not in groups:
            groups[src] = []
            order.append(src)
        groups[src].append(doc)

    merged: List[Document] = []
    for src in order:
        group     = groups[src]
        has_pages = any("page" in d.metadata for d in group)
        if len(group) == 1 or not has_pages:
            merged.extend(group)
            continue

        group_sorted = sorted(group, key=lambda d: int(d.metadata.get("page", 0)))
        parts: List[str] = []
        for d in group_sorted:
            pg = d.metadata.get("page", "?")
            parts.append(f"<<<PAGE:{pg}>>>\n{d.page_content}")
        full_text = "\n\n".join(parts)

        base_meta = dict(group_sorted[0].metadata)
        base_meta["page"]        = group_sorted[0].metadata.get("page", 1)
        base_meta["total_pages"] = group_sorted[-1].metadata.get("total_pages", len(group_sorted))
        merged.append(Document(page_content=full_text, metadata=base_meta))
        logger.info("[SemanticChunker] Merged %d pages of '%s'", len(group_sorted), src)

    return merged


# ── Core chunker ─────────────────────────────────────────────────────────────────

class SemanticChunker:
    """
    Paragraph-based semantic chunker using BAAI/bge-base-en-v1.5.

    Groups consecutive paragraphs into one chunk as long as:
      (a) the new paragraph's cosine similarity to the group's mean embedding
          is >= sim_threshold  (default 0.4), AND
      (b) adding the paragraph would not exceed max_chunk_words (default 500).

    Using the group mean (centroid) means all N paragraphs currently in the
    group influence the decision — not just the last one.  So a group of 5
    ULIP paragraphs keeps its ULIP character and correctly absorbs a 6th ULIP
    paragraph even if it phrased things differently.

    Parameters
    ----------
    embed_model     : SentenceTransformer — pass the shared TurboVec model.
    max_chunk_words : int   — word ceiling per chunk (default 500).
    overlap_words   : int   — words prepended to next chunk (default 60).
    sim_threshold   : float — min cosine similarity to stay in same group (default 0.4).
    """

    def __init__(
        self,
        embed_model: Any = None,
        max_chunk_words: int = MAX_CHUNK_WORDS,
        overlap_words: int = OVERLAP_WORDS,
        sim_threshold: float = SIM_THRESHOLD,
        # Backward-compat kwargs — accepted but ignored:
        breakpoint_percentile: float = None,
        breakpoint_pct: float = None,
        buffer_size: int = None,
        min_chunk_chars: int = None,
        max_chunk_chars: int = None,
        overlap_chars: int = None,
    ):
        self._model         = embed_model
        self._max_words     = max_chunk_words
        self._overlap_words = overlap_words
        self._sim_threshold = sim_threshold

    def _model_or_default(self) -> Any:
        return self._model if self._model is not None else _get_default_embed_model()

    def split_text(
        self,
        text: str,
        for_youtube: bool = False,  # accepted for backward compat, ignored
    ) -> tuple:
        """
        Split *text* into final chunks.

        Step 1 — paragraph splitting (blank lines / newlines / word windows)
        Step 2 — embed every paragraph with BGE
        Step 3 — greedy grouping:
                   for each paragraph, compute cosine similarity to the
                   MEAN embedding of all paragraphs already in the current group.
                   If similar enough AND fits 500 words → add to group.
                   Else → flush group as chunk, start fresh group.
        Step 4 — 60-word overlap between consecutive chunks
        """
        paragraphs = _split_paragraphs(text)
        if len(paragraphs) < 2:
            stripped = text.strip()
            return ([stripped], [0]) if stripped else ([], [])

        # ── Step 2: embed all paragraphs at once ────────────────────────────
        model = self._model_or_default()
        embeddings = model.encode(
            paragraphs,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        )  # shape: (n_paragraphs, embedding_dim)

        # ── Step 3: greedy grouping ──────────────────────────────────────────
        chunks: List[str] = []

        # State of the current group being built
        group_paras:   List[str] = [paragraphs[0]]
        group_emb_sum: np.ndarray = embeddings[0].copy()   # running sum for fast mean
        group_words:   int = len(paragraphs[0].split())

        for i in range(1, len(paragraphs)):
            para       = paragraphs[i]
            para_emb   = embeddings[i]          # already L2-normalised
            para_words = len(para.split())

            # Mean embedding of current group (re-normalise for cosine dot product)
            group_mean = group_emb_sum / len(group_paras)
            norm = float(np.linalg.norm(group_mean))
            if norm > 1e-9:
                group_mean = group_mean / norm

            # Cosine similarity: new paragraph vs current group mean
            sim = float(np.dot(group_mean, para_emb))

            same_topic    = sim >= self._sim_threshold
            fits_in_limit = group_words + para_words <= self._max_words

            logger.debug(
                "[SemanticChunker] para[%d] sim=%.3f threshold=%.2f "
                "words=%d/%d same_topic=%s fits=%s",
                i, sim, self._sim_threshold,
                group_words + para_words, self._max_words,
                same_topic, fits_in_limit,
            )

            if same_topic and fits_in_limit:
                # Similar topic + fits within 500 words → add to current group
                group_paras.append(para)
                group_emb_sum += para_emb
                group_words   += para_words
            else:
                # Topic shifted or too big → flush current group as one chunk
                chunks.append("\n\n".join(group_paras))
                logger.debug(
                    "[SemanticChunker] flushed chunk with %d paragraphs (%d words)",
                    len(group_paras), group_words,
                )
                group_paras   = [para]
                group_emb_sum = para_emb.copy()
                group_words   = para_words

        # Flush the last group
        if group_paras:
            chunks.append("\n\n".join(group_paras))

        logger.info(
            "[SemanticChunker] %d paragraphs → %d chunks (sim_threshold=%.2f, max_words=%d)",
            len(paragraphs), len(chunks), self._sim_threshold, self._max_words,
        )

        # ── Step 4: sliding-window overlap ───────────────────────────────────
        overlap_sizes: List[int] = [0] * len(chunks)
        if self._overlap_words > 0 and len(chunks) > 1:
            chunks, overlap_sizes = self._apply_overlap(chunks, self._overlap_words)

        return chunks, overlap_sizes  # (List[str], List[int])

    @staticmethod
    def _apply_overlap(
        chunks: List[str], overlap_words: int
    ) -> tuple:
        """
        Prepend the last N words of chunk[i-1] to the start of chunk[i].
        Returns (new_chunks, overlap_word_counts) where overlap_word_counts[i]
        is the number of words prepended to chunk[i] (0 for chunk[0]).
        """
        result: List[str] = [chunks[0]]
        sizes: List[int]  = [0]
        for i in range(1, len(chunks)):
            prev_words = chunks[i - 1].split()
            tail_words = prev_words[-overlap_words:] if len(prev_words) > overlap_words else prev_words
            tail = " ".join(tail_words)
            current = chunks[i]
            # Skip overlap if the next chunk already starts with the same content
            # (happens when consecutive PDF pages repeat the same boundary text).
            if tail and not current.startswith(tail[:60]):
                result.append(tail + "\n\n" + current)
                sizes.append(len(tail_words))
            else:
                result.append(current)
                sizes.append(0)
        return result, sizes

    def split_documents(
        self,
        docs: List[Document],
        doc_type: str = "policy_document",  # backward compat, ignored
        llm: Any = None,                    # backward compat, ignored
    ) -> List[Document]:
        """
        Split Documents into semantically coherent chunks.
        Multi-page PDFs: pages merged first so topic detection spans page breaks.
        Page numbers: recovered from <<<PAGE:N>>> markers after re-splitting.
        """
        docs = _merge_pages(docs)

        result: List[Document] = []
        for doc in docs:
            page_value = (
                doc.metadata.get("page")
                or doc.metadata.get("page_number")
                or doc.metadata.get("page_num")
                or 0
            )
            pieces_raw, overlap_sizes = self.split_text(doc.page_content)
            for idx, (piece, ov_size) in enumerate(zip(pieces_raw, overlap_sizes)):
                markers = _PAGE_MARKER_RE.findall(piece)
                recovered_page = page_value
                if markers:
                    try:
                        nums = [int(m) for m in markers]
                        recovered_page = min(nums)
                    except (ValueError, TypeError):
                        recovered_page = markers[0]
                    piece = _PAGE_MARKER_RE.sub("", piece).strip()

                # Count overlap words after marker stripping so the value is
                # accurate for the stored (marker-free) text.
                result.append(Document(
                    page_content=piece,
                    metadata={
                        **doc.metadata,
                        "page":                recovered_page,
                        "chunk_index":         idx,
                        "chunking_method":     "semantic",
                        "overlap_prefix_words": ov_size,
                    },
                ))
        return result
