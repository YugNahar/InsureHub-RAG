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

from metadata_tagger import _regex_policy_score

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
        logger.warning(
            "[SemanticChunker] No embed_model passed — falling back to the shared "
            "TurboVec model for '%s'. Pass embed_model explicitly to avoid this path.",
            EMBED_MODEL_NAME,
        )
        # Go through TurboVec's shared getter rather than constructing our
        # own SentenceTransformer. Building one directly bypassed both the
        # process-wide cache (a second full copy of the model in memory,
        # which the warning above already complained about) AND the device
        # resolution — no device= argument means it always lands on CPU,
        # even on a GPU host where every other model load is on cuda.
        from turbovec_store import _get_shared_embed_model
        _default_model = _get_shared_embed_model(EMBED_MODEL_NAME)
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


# ── Section-boundary detection ───────────────────────────────────────────────────
# Cosine-similarity grouping alone under-detects a genuine topic-type shift
# within one broad domain (see _cross_type_merge_conflict above and
# project_live_upload_metadata_pipeline_test) — different insurance-type
# paragraphs still measure 0.58-0.68 similarity due to shared domain
# vocabulary. Real structured documents (PDFs, handbooks) already mark
# topic shifts explicitly via headings, which is a far more reliable
# signal than embeddings when it's available. Chunking WITHIN detected
# section boundaries (never across them) fixes multi-topic chunk blending
# structurally, rather than trying to patch it after the fact with
# reranking/classification workarounds.
#
# Heading candidates are short, ALL-CAPS, multi-word lines. The hard part
# is separating genuine section titles from repeating page furniture
# (running headers/footers, "Learning Objectives" boilerplate that
# appears on every lesson page) — confirmed empirically against this
# project's real KB: boilerplate lines like "LEARNING OBJECTIVES" repeat
# 12+ times across one source document, while genuine headings like
# "MOTOR INSURANCE" or "THIRD PARTY ADMINISTRATORS-HEALTH" appear exactly
# once. A frequency filter (appears <=2 times in the document) reliably
# separates the two without needing a fixed boilerplate word list that
# would only work for this one KB's specific documents.
_HEADING_BOILERPLATE = {
    "learning objectives", "lesson outline", "lesson round-up", "lesson round up",
    "self-test questions", "self test questions", "professional programme",
    "study material", "list of recommended books", "arrangement of study lessons",
    "practice test paper",
}
# Repeating running-footer page markers like "PP-IL&P 208" — short
# alpha/punctuation code followed by a bare number.
_HEADING_PAGE_MARKER_RE = re.compile(r"^[A-Z][A-Z0-9&.\-]{1,12}\s+\d+$")
# Table-of-contents entries: "LESSON ROUND UP ... 220", "TOPIC … 51", or
# any line ending in a bare page number after real words.
_HEADING_TOC_RE = re.compile(r"(\.{2,}|…)|\s\d+$")


def _is_heading_candidate(line: str) -> bool:
    line = line.strip()
    if not (3 <= len(line) < 70):
        return False
    if not line.isupper():
        return False
    if len(line.split()) < 2:
        return False
    if line.lower() in _HEADING_BOILERPLATE:
        return False
    if _HEADING_PAGE_MARKER_RE.match(line):
        return False
    if _HEADING_TOC_RE.search(line):
        return False
    return True


def _extract_sections(text: str) -> List[tuple]:
    """
    Split *text* into (section_heading, section_text) tuples using detected
    headings as boundaries.

    Falls back to a single ("", text) section when fewer than 2 genuine
    heading breaks are found — short documents or content with no
    heading-like structure at all (plain prose, YouTube transcripts)
    shouldn't be forced into artificial section boundaries; the existing
    embedding-similarity grouping is the right tool for those.
    """
    lines = text.split("\n")
    candidates = [l.strip() for l in lines if _is_heading_candidate(l)]
    freq: dict[str, int] = {}
    for c in candidates:
        freq[c] = freq.get(c, 0) + 1
    genuine_headings = {h for h, c in freq.items() if c <= 2}

    sections: List[tuple] = []
    current_heading = ""
    current_lines: List[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped in genuine_headings and _is_heading_candidate(stripped):
            if current_lines:
                sections.append((current_heading, "\n".join(current_lines)))
            current_heading = stripped
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_heading, "\n".join(current_lines)))

    if len([s for s in sections if s[0]]) < 2:
        return [("", text)]
    return sections


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


# ── Cross-topic merge veto ───────────────────────────────────────────────────────
# Cosine similarity alone under-detects a topic-TYPE shift within a single
# broad domain: two paragraphs about completely different insurance types
# (e.g. motor vs. crop) both use heavy shared "insurance domain" vocabulary
# (policy, cover, claim, damage, premium), so their embeddings can still
# clear SIM_THRESHOLD even though a human — or the existing policy_type
# classifier — would never call them the same topic. Confirmed live via a
# controlled 6-section test document (motor/travel/marine/crop/fire/
# fidelity, each in plain natural language): 6 clearly distinct topics
# collapsed into just 2 chunks, each spanning 3-4 unrelated insurance
# types, and the resulting multi-topic chunks then got mistagged with
# whichever type happened to have the most incidental keyword hits
# (both chunks landed on "travel" — one of them containing ZERO travel
# content at all).
#
# This adds a second, independent signal alongside cosine similarity: the
# SAME regex confidence bar classify_chunk_policy_type() already uses
# (>=2 keyword hits AND 2x the runner-up) applied separately to the
# accumulated group so far and to the candidate paragraph. If BOTH sides
# clear that bar and land on DIFFERENT types, the merge is vetoed — forcing
# a new chunk boundary — even if cosine similarity says they're related
# enough. A weak/ambiguous regex signal on either side (the common case)
# never blocks a merge; this only fires when regex is confident on both
# sides and they genuinely disagree, keeping the false-positive veto rate
# low while catching the clear-cut cross-type merges that caused this bug.
def _regex_confident_type(text: str) -> str | None:
    scores = _regex_policy_score(text)
    positive = {k: v for k, v in scores.items() if v > 0}
    if not positive:
        return None
    best_type = max(positive, key=positive.__getitem__)
    best = positive[best_type]
    sorted_vals = sorted(positive.values(), reverse=True)
    runner_up = sorted_vals[1] if len(sorted_vals) > 1 else 0
    if best >= 2 and best >= (runner_up * 2 + 1):
        return best_type
    return None


def _cross_type_merge_conflict(group_text: str, para_text: str) -> bool:
    group_type = _regex_confident_type(group_text)
    para_type = _regex_confident_type(para_text)
    return group_type is not None and para_type is not None and group_type != para_type


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

    def _group_paragraphs_into_chunks(self, text: str) -> tuple:
        """
        The original greedy paragraph-grouping algorithm, scoped to a single
        contiguous span of text (one detected section, or the whole document
        when no section boundaries were found). Never sees text from a
        different section — that boundary is now enforced structurally by
        split_text() below, not just by the embedding-similarity/type-
        conflict checks within this method.

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
            # See _cross_type_merge_conflict above — cosine similarity alone
            # under-detects a topic-TYPE shift within one broad domain
            # (shared "insurance" vocabulary keeps unrelated types looking
            # similar enough). Only checked when similarity/word-limit would
            # otherwise allow the merge, since it's an extra veto, not an
            # independent merge trigger.
            type_conflict = (
                same_topic and fits_in_limit
                and _cross_type_merge_conflict("\n\n".join(group_paras), para)
            )

            logger.debug(
                "[SemanticChunker] para[%d] sim=%.3f threshold=%.2f "
                "words=%d/%d same_topic=%s fits=%s type_conflict=%s",
                i, sim, self._sim_threshold,
                group_words + para_words, self._max_words,
                same_topic, fits_in_limit, type_conflict,
            )

            if same_topic and fits_in_limit and not type_conflict:
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

    def split_text(
        self,
        text: str,
        for_youtube: bool = False,  # accepted for backward compat, ignored
    ) -> tuple:
        """
        Split *text* into final chunks, never letting a chunk cross a
        detected section boundary (see _extract_sections above). Each
        detected section is chunked independently via
        _group_paragraphs_into_chunks — overlap is applied within a
        section, never carried across into a different section's first
        chunk, since that would reintroduce the exact topic-blending this
        exists to prevent.

        Returns (chunks, overlap_sizes, section_headings) — a 3-tuple, the
        heading each chunk belongs to ("" when no section structure was
        detected) so callers can classify once per section and apply that
        result to every chunk sharing the same heading, instead of
        classifying each chunk independently.
        """
        sections = _extract_sections(text)
        all_chunks: List[str] = []
        all_overlap_sizes: List[int] = []
        all_headings: List[str] = []
        for heading, section_text in sections:
            section_chunks, section_overlaps = self._group_paragraphs_into_chunks(section_text)
            all_chunks.extend(section_chunks)
            all_overlap_sizes.extend(section_overlaps)
            all_headings.extend([heading] * len(section_chunks))

        if len(sections) > 1:
            logger.info(
                "[SemanticChunker] %d sections detected → %d total chunks",
                len(sections), len(all_chunks),
            )

        return all_chunks, all_overlap_sizes, all_headings

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
            pieces_raw, overlap_sizes, headings = self.split_text(doc.page_content)
            doc_source = doc.metadata.get("source") or doc.metadata.get("filename") or "doc"
            for idx, (piece, ov_size, heading) in enumerate(zip(pieces_raw, overlap_sizes, headings)):
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
                # accurate for the stored (marker-free) text. section_id
                # groups every chunk produced from the same detected section
                # (heading == "" when no section structure was found, in
                # which case each chunk is its own section as before) — lets
                # a caller classify policy_type once per section instead of
                # once per chunk (see project_live_upload_metadata_
                # pipeline_test: a single 500-word chunk substantively
                # discussing 3-4 different insurance types can only carry
                # one label, so tagging at the SECTION level, where a
                # heading marks a genuine single-topic boundary, is the
                # actual fix rather than a better guess at the chunk level).
                result.append(Document(
                    page_content=piece,
                    metadata={
                        **doc.metadata,
                        "page":                recovered_page,
                        "chunk_index":         idx,
                        "chunking_method":     "semantic",
                        "overlap_prefix_words": ov_size,
                        "section_heading":     heading,
                        "section_id":          f"{doc_source}::{heading}" if heading else f"{doc_source}::chunk{idx}",
                    },
                ))
        return result
