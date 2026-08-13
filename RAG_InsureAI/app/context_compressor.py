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

logger = logging.getLogger(__name__)

# Sentence must have at least this many chars to be embedded / kept.
_MIN_SENT_CHARS = 25

# Phase D (2026-08-04, plan_shallow_answers_context_budget.md): the smallest
# per-chunk allocation still worth keeping as a real chunk rather than
# dropping it. Calibrated well above _MIN_SENT_CHARS (25) — that floor is
# "still embeddable as a sentence fragment," not "still useful to an LLM
# trying to answer from it." 220 chars is roughly one full sentence of this
# KB's typical prose (see the ~112-char fragments this fix replaces —
# confirmed live those were routinely mid-word/mid-clause, not complete
# thoughts). See compress_to_budget's docstring for the failure mode this
# closes.
_MIN_VIABLE_CHUNK_CHARS = 220

# Same reasoning as _MIN_VIABLE_CHUNK_CHARS, one level deeper: the
# smallest per-WINDOW allocation (see the fair-share-across-windows
# packing in compress_to_budget) still worth keeping as a real fragment
# rather than starving it out entirely. Lower than the chunk floor since
# a window is a sub-unit of a chunk, not a whole chunk on its own.
_MIN_VIABLE_WINDOW_CHARS = 80


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


def _window_bounds(n: int, window: int) -> List[tuple]:
    """Non-overlapping (start, end) index ranges covering all n sentences,
    each up to `window` sentences wide, in original document order.
    window=1 reproduces the original per-sentence behavior exactly
    (n windows of size 1); window >= n collapses to one window covering
    everything — both fall out of the same range() formula, no special
    case needed."""
    if n <= 0:
        return []
    w = max(1, window)
    return [(s, min(s + w, n)) for s in range(0, n, w)]


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
        window_size: int = 5,
    ):
        self._model = embed_model
        self._threshold = similarity_threshold
        self._min_sents = min_sentences
        self._max_sents = max_sentences
        self._skip_below = max_chars_per_chunk
        # Score each sentence by the similarity of the WINDOW of
        # window_size consecutive sentences it belongs to, not the
        # sentence alone (2026-08-13) — confirmed live an isolated
        # sentence often lacks enough surface vocabulary to score well
        # against the query even when it states the one fact that
        # answers it: "Form B which is also known as Comprehensive
        # Policy is an optional cover" scored 0.4684 alone (rank 26 of
        # 28 sentences in its chunk) against "Explain motor insurance in
        # detail", but the same text grouped with its 4 neighboring
        # sentences scored 0.7111 (rank 2 of 6) — because the window as
        # a whole shares more of the query's vocabulary ("motor
        # insurance", "types of policies") than the isolated sentence
        # does on its own. Tested for the opposite failure mode too
        # (windowing diluting a genuinely important standalone
        # sentence by averaging it with irrelevant neighbors) with a
        # sentence echoing this module's own documented "broken hand"
        # exclusion case, surrounded by up to 4 unrelated filler
        # sentences — windowing still scored HIGHER than scoring it
        # alone in every case tried, never lower. window_size=1
        # reproduces the original per-sentence behavior exactly (see
        # _window_bounds), so this is a strict generalization, not a
        # different mechanism bolted on.
        self._window = window_size
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
          0. If fair-share would fall below a per-chunk viability floor
             (_MIN_VIABLE_CHUNK_CHARS), drop the lowest-ranked (tail)
             chunks first rather than spreading an unusably thin
             allocation across all of them — see the Phase D note below.
          1. Give every SURVIVING chunk a fair-share allocation
             (max_total_chars / N) up front; any share left unused by a
             chunk smaller than its allocation rolls over to chunks that
             need more, highest-relevance-first (caller's pre-sort order).
          2. For a chunk that fits within its final allocation, include it
             as-is.
          3. For a chunk that exceeds its allocation, keep only the most
             query-relevant sentences from it that still fit.
          4. As a last resort, hard-truncate at a sentence boundary.

        Fair-share (step 1) replaces a strict "fill in rank order until the
        budget runs out" pass — confirmed live: 10 chunks competing for a
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

        Phase D (2026-08-04, plan_shallow_answers_context_budget.md): fair
        share has its OWN failure mode at the low end, opposite to the one
        it was built to fix above — it spends the whole budget fragmenting
        EVERY chunk into a sliver, rather than keeping SOME chunks
        genuinely usable. Confirmed live: an 8-chunk, 900-char budget gave
        every chunk ~112 chars — well under one real sentence — so every
        chunk took the hard_truncate path (step 4) and the model received
        8 mid-sentence fragments instead of coherent context, which is
        exactly what upstream context-budget fixes (Phases A/C0/C3) were
        needed for in the first place. Step 0 only engages BELOW the
        viability floor — above it this is a no-op, and step 1's behavior
        for the case it was built for (see the paragraph above) is
        unchanged, byte-for-byte. `chunks` arrives here already sorted by
        relevance (`_sort_and_truncate` runs before this call in
        multi_source_rag.py), so dropping from the end is well-defined:
        it drops the least-relevant survivors, not an arbitrary subset.
        A dropped chunk is absent from the RETURNED (compressed) list, but
        callers keep `_full_context_uncompressed` built from the ORIGINAL
        retrieved set for the post-generation grounding checks — those are
        unaffected by anything this method drops or fragments, by design
        (see multi_source_rag.py's own comment on that variable).
        """
        total = sum(len(d.page_content) for d in chunks)
        if total <= max_total_chars:
            # Everything already fits — return untouched, zero embedding cost.
            return chunks

        logger.info(
            "[Compressor] budget trim: %d chars across %d chunks → target %d chars",
            total, len(chunks), max_total_chars,
        )

        n = len(chunks)
        _dropped_for_viability = 0
        while n > 1 and max_total_chars // n < _MIN_VIABLE_CHUNK_CHARS:
            chunks = chunks[:-1]
            n -= 1
            _dropped_for_viability += 1
        if _dropped_for_viability:
            logger.info(
                "[Compressor] dropped %d lowest-ranked chunk(s) rather than fragment all "
                "%d below the %d-char viability floor (%d chars ÷ %d chunks = %d/chunk)",
                _dropped_for_viability, n + _dropped_for_viability, _MIN_VIABLE_CHUNK_CHARS,
                max_total_chars, n + _dropped_for_viability,
                max_total_chars // (n + _dropped_for_viability),
            )

        query_emb: np.ndarray = self._model.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        )[0]

        sizes = [len(d.page_content) for d in chunks]
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
        # (sentences, window_bounds, window_embs) if this exact chunk text
        # was compressed before — the KB itself doesn't change between
        # requests, only the query does, so cache hits are the common case
        # after warm-up. Only genuine cache misses get queued for the
        # model; those are still batched into one encode() call rather
        # than one per chunk. Cached by WINDOW rather than by individual
        # sentence (2026-08-13) — see self._window's own docstring in
        # __init__ for why scoring moved from per-sentence to per-window;
        # window_bounds lets each sentence still get its own slot in the
        # final packing step below, just scored via its window instead of
        # itself alone.
        _slots: List[Optional[Dict[str, Any]]] = [None] * n
        _all_windows: List[str] = []
        _offsets: List[tuple] = []  # (slot_index, start, end) into _all_windows
        # Real production hit rate was unmeasured — the three plausible causes
        # of "still slow despite caching" (cold-start on a wide/low-overlap
        # retrieval pattern, cache size churning against a bigger corpus, or
        # a genuine per-token model-speed floor) each point to a different
        # fix, and this is the one number that tells them apart.
        _cache_hits = 0
        _cache_misses = 0

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
            _cache_key = (text, is_yt, self._window)
            _cached = self._sent_cache.get(_cache_key)
            if _cached is not None:
                _cache_hits += 1
                sentences, window_bounds, window_embs = _cached
                if len(sentences) <= 1:
                    truncated = text[:alloc].rsplit('. ', 1)[0] + '…'
                    _slots[i] = {"kind": "hard_truncate", "doc": doc, "text": truncated}
                    continue
                _slots[i] = {
                    "kind": "rank", "doc": doc, "alloc": alloc,
                    "sentences": sentences, "window_bounds": window_bounds, "window_embs": window_embs,
                }
                continue

            _cache_misses += 1
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
                self._sent_cache[_cache_key] = (sentences, [], None)
                continue

            # Deliberately NOT prepending _rerank_metadata_prefix() here
            # (2026-08-13, reverted — see git history for the earlier
            # attempt). It's a proven win for the cross-encoder reranker,
            # which compares candidates ACROSS different chunks/sources —
            # there, a chunk's policy_type/section tells the reranker
            # something a differently-tagged competing chunk lacks. This
            # loop is different: every window here belongs to the SAME
            # chunk, `alloc` was already fixed per-chunk by the outer
            # fair-share pass above, and windows are only ever ranked
            # against OTHER WINDOWS OF THIS SAME CHUNK — which all share
            # identical metadata. A prefix constant across every candidate
            # in a ranking can't add discriminating signal there; it can
            # only distort it, and confirmed live it does: this chunk's
            # metadata says "Section: Principles", but the chunk's own
            # text actually spans two headings ("BASIC PRINCIPLES OF MOTOR
            # INSURANCE" and, later, "TYPES OF MOTOR INSURANCE POLICIES").
            # Prepending "[Section: Principles]" to every window pulled
            # the later, correctly-on-topic "Form A / Form B" window's
            # score down from rank 3 of 10 to rank 7 of 10 relative to
            # its chunk-mates, purely because that window is about a
            # DIFFERENT section than the one label covering the whole
            # chunk — dropping it out of the packer's budget entirely.
            window_bounds = _window_bounds(len(sentences), self._window)
            window_texts = [" ".join(sentences[s:e]) for s, e in window_bounds]
            start = len(_all_windows)
            _all_windows.extend(window_texts)
            end = len(_all_windows)
            _offsets.append((i, start, end))
            _slots[i] = {
                "kind": "rank",
                "doc": doc,
                "alloc": alloc,
                "sentences": sentences,
                "window_bounds": window_bounds,
                "cache_key": _cache_key,
            }

        # Pass 2: one batched encode() call for every cache-miss window
        # collected above.
        if _all_windows:
            _all_window_embs = self._model.encode(
                _all_windows, normalize_embeddings=True, batch_size=32, show_progress_bar=False
            )
        else:
            _all_window_embs = None

        for slot_idx, start, end in _offsets:
            slot = _slots[slot_idx]
            window_embs = _all_window_embs[start:end]
            slot["window_embs"] = window_embs
            _cache_key = slot.pop("cache_key")
            if len(self._sent_cache) >= self._sent_cache_max:
                self._sent_cache.pop(next(iter(self._sent_cache)))
            self._sent_cache[_cache_key] = (slot["sentences"], slot["window_bounds"], window_embs)

        for i, slot in enumerate(_slots):
            if slot is None or slot["kind"] != "rank":
                continue
            doc = slot["doc"]
            alloc = slot["alloc"]
            sentences = slot["sentences"]
            window_bounds = slot["window_bounds"]
            window_embs = slot["window_embs"]
            window_scores = np.dot(window_embs, query_emb)
            # Broadcast each window's score onto every sentence it covers —
            # this IS the mechanism described in self._window's docstring:
            # a sentence's relevance score comes from the window of
            # surrounding context it belongs to, not itself in isolation.
            scores = np.empty(len(sentences))
            for (s, e), wscore in zip(window_bounds, window_scores):
                scores[s:e] = wscore
            # Pack WHOLE windows only — never split one into a partial
            # slice (2026-08-13). Windows exist precisely because a lone
            # sentence's embedding is too noisy to rank reliably (see the
            # class docstring); an earlier version of this fix sliced an
            # over-budget window down to its "best" individual sentences,
            # which reintroduces that exact noise one level down. Confirmed
            # live: both the real "Form A / Form B" case and a synthetic
            # "one highly specific on-topic sentence surrounded by generic
            # filler" case lost the specific sentence to per-sentence
            # re-ranking even though their WINDOW correctly scored highest
            # — the specific sentence alone still reads as low-relevance
            # in isolation, which is the exact problem windowing exists to
            # avoid. Keeping every window atomic avoids ever needing that
            # unreliable per-sentence judgment call again.
            n_windows = len(window_bounds)
            window_sizes = [
                sum(len(sentences[j]) + 2 for j in range(s, e))
                for s, e in window_bounds
            ]

            # Drop windows too small to matter even if nothing else
            # competed for the budget, same floor as _MIN_VIABLE_CHUNK_CHARS
            # one level deeper — an empty/near-empty window isn't a useful
            # unit to place at all.
            active = [idx for idx in range(n_windows) if window_sizes[idx] >= _MIN_VIABLE_WINDOW_CHARS]
            if not active:
                active = list(range(n_windows))

            # Greedy-WITH-SKIP across windows in score order (2026-08-13)
            # — the reason a plain greedy walk starved out the Form A/B
            # window even after windowing fixed its ranking: a handful of
            # big, high-scoring windows can consume the ENTIRE per-chunk
            # allocation before a smaller, lower-but-still-relevant window
            # is ever tried. Stopping at the first window that doesn't fit
            # (a "break") reproduces that starvation; skipping it and
            # continuing to try smaller, lower-scored windows against
            # whatever budget remains is what lets a short but genuinely
            # relevant window claim leftover space the bigger ones left
            # behind, without ever fragmenting any window.
            ranked_active = sorted(active, key=lambda idx: window_scores[idx], reverse=True)
            kept_indices: set = set()
            remaining = alloc
            skipped: list = []
            for idx in ranked_active:
                s, e = window_bounds[idx]
                size = window_sizes[idx]
                if size <= remaining:
                    kept_indices.update(range(s, e))
                    remaining -= size
                else:
                    skipped.append(idx)

            # Leftover budget too small for any whole skipped window to
            # fit — rather than wasting it, take as much of the single
            # highest-scoring skipped window as fits, sentence by sentence
            # in document order (not re-ranked — this is filling genuine
            # leftover scraps, a secondary best-effort step, not the
            # primary per-window decision that needed reliable scoring).
            if skipped and remaining >= _MIN_VIABLE_WINDOW_CHARS:
                best_skipped = max(skipped, key=lambda idx: window_scores[idx])
                s, e = window_bounds[best_skipped]
                used = 0
                for j in range(s, e):
                    L = len(sentences[j]) + 2
                    if used + L <= remaining:
                        kept_indices.add(j)
                        used += L
                    else:
                        break

            if not kept_indices:
                # Degenerate case (alloc smaller than any single kept
                # sentence) — fall back to the single highest-scoring
                # sentence, hard-truncated if it alone still doesn't fit,
                # rather than returning nothing for this chunk. Mirrors
                # the pre-fair-share "guarantee the top sentence a slot"
                # safety net this replaces.
                top_idx = int(np.argmax(scores))
                top_sentence = sentences[top_idx]
                if len(top_sentence) + 2 > alloc:
                    truncated = top_sentence[:alloc].rsplit(' ', 1)[0] + '…'
                    _slots[i] = {"kind": "hard_truncate", "doc": doc, "text": truncated}
                    continue
                kept_indices = {top_idx}

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

        _ranked_total = _cache_hits + _cache_misses
        if _ranked_total > 0:
            logger.info(
                "[Compressor] sentence-cache: %d hit / %d miss (%.0f%% hit rate this call), "
                "cache size=%d/%d",
                _cache_hits, _cache_misses, 100 * _cache_hits / _ranked_total,
                len(self._sent_cache), self._sent_cache_max,
            )

        return final
