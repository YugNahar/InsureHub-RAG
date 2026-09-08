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
5. Prepend the LAST SENTENCE of each chunk to the next chunk (overlap) —
   not a fixed word count. A fixed-word overlap can dominate a short
   next chunk (60 words is a small fraction of a 500-word chunk but
   over half of a 100-word one) and routinely starts the next chunk
   mid-sentence. A whole-sentence overlap scales with the actual unit
   of meaning being carried forward and never starts mid-sentence.
   Falls back to a word-count tail (capped at OVERLAP_WORDS) only when
   the previous chunk has no real internal sentence boundary at all.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any, List, Optional

import numpy as np
from langchain_core.documents import Document

from metadata_tagger import _regex_policy_score, classify_query_policy_type

logger = logging.getLogger(__name__)

# ── Tuning ──────────────────────────────────────────────────────────────────────
MAX_CHUNK_WORDS   = 500   # hard word ceiling per chunk
OVERLAP_WORDS     = 60    # overlap unit is now the previous chunk's LAST SENTENCE;
                          # this is only the fallback/cap for when no real sentence
                          # boundary exists, or the real last sentence runs unusually long
SIM_THRESHOLD     = 0.4   # cosine similarity floor — below this = topic shift
_MIN_PARA_CHARS   = 20    # drop blank / very short fragments

# Reuses the exact abbreviation-guard pattern already proven in
# multi_source_rag.py's _pgf_split_sentences (PGF's prose-sentence path) —
# same regex, same "merge back after an abbreviation" behavior, so a period
# after "Rs." or "Ltd." (both common in this KB) doesn't falsely end a
# sentence here either.
_OVERLAP_ABBREV_RE = re.compile(
    r'\b(?:Rs|Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|e\.g|i\.e|no|vol|'
    r'pp|approx|Inc|Ltd|Co|St|Ave|Fig)\.$',
    re.IGNORECASE,
)


def _split_into_sentences(text: str) -> List[str]:
    """Split on [.!?] followed by whitespace, then merge a split-off piece
    back onto the previous one whenever that previous piece ends in a
    known abbreviation — so "Rs. 5,000" or "M/s. ABC Ltd." never counts
    as a sentence boundary on their own.
    """
    raw = re.split(r'(?<=[.!?])\s+', text)
    merged: List[str] = []
    for piece in raw:
        if merged and _OVERLAP_ABBREV_RE.search(merged[-1]):
            merged[-1] = f"{merged[-1]} {piece}"
        else:
            merged.append(piece)
    return merged

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


# ── Bulleted-list detection ──────────────────────────────────────────────────────
# A section built from several DISTINCT bulleted tips/points (e.g. "■ Buy term
# insurance early... ■ Disclose all material facts... ■ Choose a sum assured...
# ■ Keep nominee details updated...") shares enough surface vocabulary (all
# about "buying life insurance") that the similarity-grouping below could
# still merge them back into one chunk even once paragraph-splitting
# correctly separates them — SIM_THRESHOLD=0.4 is a low bar, and several
# tips on the same broad topic can clear it. Confirmed live 2026-08-31: a
# real "Practical Tips" section with 4 distinct bullet tips stayed ONE
# ~700-char chunk end to end (never reached the similarity check at all —
# _split_paragraphs' three tiers don't recognize a bullet glyph as a
# boundary, so this whole section looked like a single paragraph), diluting
# its embedding match for any query about just ONE of the four tips enough
# that a genuinely grounded, almost-verbatim-present fact ("premium is
# generally locked in based on age at purchase") got dropped by the
# post-generation grounding check simply because that one tip's sentence
# never made it into the small set of chunks the checker actually saw —
# same root shape as the webpage FAQ-bundling bug this session already
# fixed (see project_webpage_chunking_heading_detection_gap), just via a
# different mechanism (bullet markers instead of missing HTML heading
# structure).
#
# Same "structural marker beats embedding similarity" principle used
# throughout this module for section headings (see the comment above
# _extract_sections): an explicit, repeated bullet GLYPH is unambiguous,
# ground-truth list structure straight from the source document's own
# formatting, not an inferred shape. Once a genuine list is detected, each
# item becomes its own chunk directly — bypassing the greedy similarity
# merge entirely, never re-joined even if two adjacent tips happen to
# score high similarity, the same "never cross a detected boundary"
# guarantee split_text already gives section headings. Deliberately a
# small, closed set of unambiguous bullet GLYPHS (not a plain "-" or "*",
# both of which appear constantly inside ordinary prose — hyphenated
# words, math, emphasis — and would false-positive constantly).
_BULLET_MARKER_RE = re.compile(r'(?:^|\s)([■•●▪‣◦])\s+')
_MIN_BULLET_ITEMS = 3


def _split_bullet_items(text: str) -> "List[str] | None":
    """Returns one string per bulleted item when *text* contains a genuine
    bulleted list (>= _MIN_BULLET_ITEMS occurrences of a recognized bullet
    glyph), collapsing each item's own internal line-wrap whitespace/
    newlines into single spaces first — PDF extraction routinely wraps one
    bullet's sentence across several physical lines with blank-line-shaped
    gaps between them, which would otherwise look like several separate
    short paragraphs rather than one coherent item. Any text before the
    first marker (a lead-in sentence, if the section has one) is prepended
    to the first item rather than dropped. Returns None when fewer than
    _MIN_BULLET_ITEMS markers are found — a single stray bullet character
    isn't a real list, and callers should fall through to the normal
    paragraph-based splitting.
    """
    markers = list(_BULLET_MARKER_RE.finditer(text))
    if len(markers) < _MIN_BULLET_ITEMS:
        return None
    items: List[str] = []
    for i, m in enumerate(markers):
        start = m.start(1)
        end = markers[i + 1].start(1) if i + 1 < len(markers) else len(text)
        item = " ".join(text[start:end].split())
        if item:
            items.append(item)
    if len(items) < _MIN_BULLET_ITEMS:
        return None
    lead_in = text[:markers[0].start(1)].strip()
    if lead_in:
        items[0] = f"{lead_in} {items[0]}"
    return items


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
# Heading candidates are short, ALL-CAPS OR Title Case, multi-word lines.
# The hard part is separating genuine section titles from repeating page
# furniture (running headers/footers, "Learning Objectives" boilerplate
# that appears on every lesson page) — confirmed empirically against this
# project's real KB: boilerplate lines like "LEARNING OBJECTIVES" repeat
# 12+ times across one source document, while genuine headings like
# "MOTOR INSURANCE" or "THIRD PARTY ADMINISTRATORS-HEALTH" appear exactly
# once. A frequency filter (appears <=2 times in the document) reliably
# separates the two without needing a fixed boilerplate word list that
# would only work for this one KB's specific documents.
#
# ALL-CAPS-only was too narrow: confirmed live 2026-08-10 (off-vocab test,
# a synthetic drone-insurance PDF) that a real, common heading style this
# never matched at all -- Title Case ("1. What Drone Insurance Covers",
# "1.1 Hull Cover") -- meaning heading detection (and everything that
# depends on it: doc_prior, regex_first_pass_policy_type's strongest
# branch, Phase 3's page-header fallback) had zero signal to work with
# for a document that literally states its own topic in its own title.
# ALL-CAPS is this KB's scanned-textbook convention, not a general
# property of real-world PDFs (Word exports, typeset guides, etc.
# overwhelmingly use Title Case instead) -- see _is_title_case below.
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

# Small connector words don't count toward the Title Case capitalization
# check below -- a genuine heading like "Third-Party Liability Cover"
# or "Who Can Buy This Policy" is still Title Case even though a real
# style guide would lowercase "of"/"the"/"and" inside it.
_TITLE_CASE_SKIP_WORDS = {
    "a", "an", "the", "of", "in", "on", "to", "for", "and", "or", "is",
    "are", "with", "by", "at", "from", "as", "vs", "vs.",
}


def _is_title_case(line: str) -> bool:
    """Heading-shaped Title Case: almost every significant word starts
    with a capital letter, and the line doesn't end in sentence-
    terminating punctuation -- a genuine heading is a short label, not a
    sentence. Confirmed against real prose (this module's own docstrings
    and this KB's chunk text): a sentence starting with a capitalized
    phrase drops to mostly-lowercase within a few words, so the >=80%
    bar cleanly separates "Third-Party Liability Cover" (heading) from
    "Third-Party Liability Cover responds when the drone causes bodily
    injury..." (sentence, same opening words) without needing NLP."""
    if line[-1:] in ".?!,;:":
        return False
    words = line.split()
    # Numeric section prefixes ("1.1", "2.") and bare punctuation ("—")
    # have no case at all -- counting them as "not capitalized" against
    # the ratio wrongly penalizes numbered headings like "4. Premium
    # Factors" (confirmed live 2026-08-10: dropped the ratio to 0.67 on
    # short numbered headings, just under the 0.8 bar). Excluded from
    # the denominator entirely rather than counted as a miss.
    significant = [
        w for w in words
        if any(c.isalpha() for c in w) and w.lower().strip(".,()") not in _TITLE_CASE_SKIP_WORDS
    ]
    if not significant:
        return False
    capitalized = sum(1 for w in significant if w[:1].isupper())
    return capitalized / len(significant) >= 0.8


# A "#"/"##"/... prefix means document_loader._parse_html_to_text found a
# real HTML <h1>-<h6> tag at this line -- ground-truth structure straight
# from the source markup, not an inferred shape. That's strictly stronger
# evidence than the Title-Case/punctuation heuristics below, which are
# tuned for un-marked plain text and would otherwise reject a real FAQ-style
# heading like "How do I determine the right life insurance coverage for
# me?" on BOTH counts at once: it ends in "?" (an immediate reject in
# _is_title_case) and its content words are mostly lowercase, natural-
# question phrasing rather than Title Case (confirmed live 2026-08-31: a
# policybazaar FAQ page's <h2> questions all failed silently this way,
# leaving every FAQ bundled into one oversized, unbounded chunk despite
# the extraction layer correctly marking each heading).
_MARKDOWN_HEADING_RE = re.compile(r"^#{1,6}\s+")

# Strips markdown emphasis/underline decoration a heading line can carry
# after the "#" prefix itself is removed — confirmed live 2026-09-01 with
# pymupdf4llm markdown output: "## 7. **<u>LIFE INSURANCE</u>**" leaves
# "7. **<u>LIFE INSURANCE</u>**" as the raw heading text once the "##" is
# stripped, decoration and all, unless removed separately.
_HEADING_DECORATION_RE = re.compile(r"\*\*|__|<u>|</u>|<b>|</b>|<sup>|</sup>|<sub>|</sub>")


def _clean_heading_text(text: str) -> str:
    return _HEADING_DECORATION_RE.sub("", text).strip()

# ── Sub-headings within a section's own body ─────────────────────────────────────
# A real markdown "#" heading marks a genuine SECTION boundary — but a
# section can itself enumerate several distinct sub-items, each with its
# own numbered/lettered/Roman-numeral lead-in immediately followed by its
# own definition. FOUR real markdown shapes cover this, confirmed against
# real documents:
#
#   1. BOLD_MARKER — a bold numbered/lettered lead-in immediately inside
#      its own body text (confirmed live 2026-09-01, pymupdf4llm markdown
#      output of a real insurance textbook): "- **1) Term insurance**
#      \n\n A term insurance product provides..." / "- **2) Whole life
#      insurance** \n\n Whole life insurance product provides...".
#   2. BOLD_BULLET — same shape as (1) but with a bullet glyph
#      (■•●▪‣◦) instead of a number/letter, e.g. "■ Term insurance".
#   3. BULLET_MARKER — a markdown BULLET-LIST item whose own text starts
#      with a numbered/lettered/Roman-numeral marker, e.g. "- a) The
#      collateral shall be..." / "   - i. the LC shall be...", nested
#      sub-bullets at deeper indentation for nested clauses. Added
#      2026-09-04, ported back from rag_site_1 (a downstream fork) after
#      confirming live against a real, independently-downloaded IRDAI
#      regulatory-circular PDF that pymupdf4llm renders lettered/Roman-
#      numeral CLAUSE lists (as opposed to this file's own originally-
#      observed named-item lists) as plain bulleted list items, NOT bold
#      spans — the original version of this regex (shapes 1-2 only) was
#      a complete no-op against that document shape. Confirmed via the
#      SAME real, independent PDF this codebase's own pymupdf4llm output
#      was tested against, not a self-constructed case built to pass this
#      regex.
#   4. BARE_BULLET — a markdown bullet-list item with NO embedded
#      number/letter marker at all, just a bullet and its own lead text
#      (e.g. "- Term insurance ... " with no "1)"/"a)" prefix) — some
#      source documents enumerate sub-items this way instead of
#      numbering them. The bullet's own first few words stand in for the
#      marker, since there's no separate label to pull.
#
# Bold/bullet + a short marker immediately inside it is a deliberately
# narrow, high-precision pattern (not "any bold text") — ordinary
# emphasis on a word or two mid-sentence never matches this because it
# lacks the leading marker.
#
# Covers every enumeration style actually seen across this KB's PDFs, not
# just plain digits — confirmed live 2026-09-01 in this SAME document
# ("LIFE INSURANCE PRODUCTS" section uses Roman numerals: "I. Term
# insurance & Health Insurance plans - II. Endowment & Money-back
# plans..."), plus the full alphabet (not just a-h — there was never a
# real reason to cap there) and the same bullet-glyph set already
# established and tuned for list-splitting elsewhere in this file
# (_BULLET_MARKER_RE) reused here for consistency rather than inventing a
# second glyph list. The Roman-numeral fragment requires a leading valid
# Roman character before trying to match (avoids a zero-length match),
# and single letters/numerals still need trailing "." or ")" — a bare
# bullet glyph doesn't, since that's its own complete, self-punctuating
# marker in normal usage (e.g. "■ Term insurance").
_ROMAN_NUMERAL_FRAGMENT = r"(?=[MDCLXVI])M{0,4}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})"
_CLAUSE_MARKER_FRAGMENT = r"\(?(?:" + _ROMAN_NUMERAL_FRAGMENT + r"|\d{1,3}|[a-zA-Z])[.)]"
_BULLET_GLYPH_FRAGMENT = r"[-*■•●▪‣◦]"
_SUBHEADING_RE = re.compile(
    r"\*\*" + _CLAUSE_MARKER_FRAGMENT + r"\s*(?P<bold_marker>[^*\n]{2,80}?)\*\*"
    r"|\*\*" + _BULLET_GLYPH_FRAGMENT + r"\s*(?P<bold_bullet>[^*\n]{2,80}?)\*\*"
    r"|^[ \t]*" + _BULLET_GLYPH_FRAGMENT + r"[ \t]+(?P<bullet_marker>" + _CLAUSE_MARKER_FRAGMENT + r")\s"
    r"|^[ \t]*" + _BULLET_GLYPH_FRAGMENT + r"[ \t]+(?P<bare_bullet>(?:[^\s,\n]+[ \t]+){0,6}[^\s,\n]+)",
    re.MULTILINE | re.IGNORECASE,
)


# Bare-minimum length floor kept ONLY to skip spaCy entirely on trivially
# short bodies — NOT the real gate anymore. Confirmed live 2026-09-04 (on
# a downstream fork's corpus, then verified the same principle applies
# here): word count alone can't separate a bare label from a real short
# clause — "Present Residential Address:" (4 words) and "whether the
# applicant has ever been declared bankrupt" (8 words) can be nearly the
# same length, so no length threshold reliably tells them apart.
_MIN_SUBHEADING_WORDS_FLOOR = 3

# spaCy's dependency parser gives a real linguistic signal a length
# threshold can't: does this body text have a finite verb at its root (a
# real predicate), or is it just a noun phrase with nothing asserted
# about it? The first is genuine elaboration worth a sub-heading split;
# the second is a bare enumerated name with nothing behind it — exactly
# the "Term insurance & Health Insurance plans" case this gate was
# originally built for, just detected by what the text actually IS
# rather than how long it happens to be.
try:
    import spacy
    _SUBHEADING_NLP: Optional["spacy.language.Language"] = spacy.load(
        "en_core_web_sm", disable=["ner", "lemmatizer"],
    )
except Exception as _spacy_import_exc:  # pragma: no cover - defensive only
    logger.warning(
        "[semantic_chunker] spaCy unavailable (%s) — sub-heading bodies "
        "fall back to the bare word-count floor only",
        _spacy_import_exc,
    )
    _SUBHEADING_NLP = None


def _is_meaningful_clause(text: str) -> bool:
    """True if *text* has a finite verb at the root of its dependency
    parse — i.e. is a real clause with something asserted, not just a
    bare label/noun phrase. Fails open (True) if spaCy is unavailable.
    """
    if _SUBHEADING_NLP is None:
        return True
    doc = _SUBHEADING_NLP(text[:300])
    return any(tok.dep_ == "ROOT" and tok.pos_ in ("VERB", "AUX") for tok in doc)


def _split_by_subheadings(text: str) -> List[tuple]:
    """
    Split *text* at bold numbered/lettered sub-item markers (see
    _SUBHEADING_RE) into (sub_heading, piece_text) tuples. sub_heading is
    "" for any text BEFORE the first marker (may be the whole text, if no
    markers are found at all — always at least one tuple is returned).
    The marker itself is stripped from the piece text; the piece runs
    from just after one marker to just before the next (or end of text).

    A matched marker only counts as a genuine sub-heading boundary if its
    own body clears _MIN_SUBHEADING_WORDS_FLOOR (cheap pre-filter) AND is
    a real clause per _is_meaningful_clause — confirmed live 2026-09-01: a
    BARE enumerated name list, where every item is bold and numbered but
    has ZERO elaboration before the next item starts (e.g. "- **I. Term
    insurance & Health Insurance plans** - **II. Endowment & Money-back
    plans** - **III. Whole life plans**..."), is structurally just an
    index/summary list, not a set of self-contained definitions — the
    real definitions of these types live in a COMPLETELY DIFFERENT
    section of the same document. Splitting a bare list like this would
    only produce empty, useless chunks (just a name, nothing else) —
    worse than not splitting at all, since real content that already
    existed as one coherent unit would get fragmented for no benefit.
    A marker that fails this bar is simply not treated as a boundary —
    its own text flows through as ordinary body content of whichever
    section/sub-section it falls inside, exactly as if it had never
    matched _SUBHEADING_RE at all.

    A text that's already a genuine FLAT bulleted list (>= _MIN_BULLET_
    ITEMS plain bullet-glyph markers, the same signal _split_bullet_items
    downstream uses to detect one) is left untouched by the BOLD_BULLET/
    BARE_BULLET shapes specifically — those key off the exact same glyph
    set, and have no way to tell "a bullet acting as a short inline
    label" (their intended target) apart from "an ordinary bullet in a
    flat list whose own first few words simply don't contain a comma
    yet." Confirmed live 2026-09-05: "■ Losses arising from participation
    in adventure sports, unless a specific add-on has been purchased."
    had its first 7 words cut off as a bogus sub-heading — losing the
    bullet glyph and lead-in entirely — purely because no comma appears
    within them, even though the whole thing is one ordinary, complete
    descriptive sentence with no heading/body structure at all.
    _split_bullet_items already gives every item in a genuine flat list
    its own complete, correctly-formed chunk; nothing here needs to (or
    should) pre-split it first. BOLD_MARKER/BULLET_MARKER (the two
    numbered/lettered-clause shapes) are unaffected — they name a
    genuinely different structure (nested regulatory sub-clauses) that a
    plain glyph-only bullet list never matches anyway.
    """
    _is_genuine_bullet_list = len(_BULLET_MARKER_RE.findall(text)) >= _MIN_BULLET_ITEMS
    all_matches = list(_SUBHEADING_RE.finditer(text))
    if _is_genuine_bullet_list:
        all_matches = [
            m for m in all_matches
            if m.group("bare_bullet") is None and m.group("bold_bullet") is None
        ]
    if not all_matches:
        return [("", text)]

    matches = []
    for i, m in enumerate(all_matches):
        body_end = all_matches[i + 1].start() if i + 1 < len(all_matches) else len(text)
        body_text = text[m.end():body_end]
        body_word_count = len(body_text.split())
        if body_word_count >= _MIN_SUBHEADING_WORDS_FLOOR and _is_meaningful_clause(body_text):
            matches.append(m)
    if not matches:
        return [("", text)]

    pieces: List[tuple] = []
    lead_in = text[: matches[0].start()].strip()
    if lead_in:
        pieces.append(("", lead_in))
    for i, m in enumerate(matches):
        # Exactly one named group is non-None depending on which
        # alternative in _SUBHEADING_RE matched (see its own comment for
        # what each of the four shapes covers).
        raw_sub_heading = m.group("bold_marker") or m.group("bold_bullet") or \
            m.group("bullet_marker") or m.group("bare_bullet")
        sub_heading = _clean_heading_text(raw_sub_heading).rstrip(":;,.")
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()
        pieces.append((sub_heading, f"{sub_heading}\n\n{body}" if body else sub_heading))
    return pieces


def _is_heading_candidate(line: str) -> bool:
    line = line.strip()
    _md_match = _MARKDOWN_HEADING_RE.match(line)
    if _md_match:
        heading_text = line[_md_match.end():].strip()
        if not (1 <= len(heading_text) < 200):
            return False
        if heading_text.lower() in _HEADING_BOILERPLATE:
            return False
        return True
    if not (3 <= len(line) < 70):
        return False
    if not (line.isupper() or _is_title_case(line)):
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
    Split *text* into (section_heading, sub_heading, section_text) triples
    using detected headings as top-level boundaries, then further split
    each section at any detected bold numbered/lettered sub-item markers
    (see _split_by_subheadings) — a document like "## Whole Life vs
    Endowment" containing "- **1) Term insurance** ... - **2) Whole life
    insurance** ..." now produces separate pieces for each numbered item,
    not one blob only the first item's content reliably survives
    retrieval from (confirmed live 2026-09-01). sub_heading is "" for a
    section with no such markers — fully backward compatible.

    Falls back to a single ("", "", text) section when fewer than 2
    genuine heading breaks are found — short documents or content with no
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
            _md_match = _MARKDOWN_HEADING_RE.match(stripped)
            current_heading = _clean_heading_text(
                stripped[_md_match.end():] if _md_match else stripped
            )
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_heading, "\n".join(current_lines)))

    if len([s for s in sections if s[0]]) < 2:
        sections = [("", text)]

    expanded: List[tuple] = []
    for heading, section_text in sections:
        for sub_heading, piece_text in _split_by_subheadings(section_text):
            if piece_text.strip():
                expanded.append((heading, sub_heading, piece_text))
    return expanded


# ── Page-header/footer topic mining (plan_policy_type_tagging.md RC-4) ──────────
# _extract_sections above deliberately EXCLUDES a frequently-repeating line
# (freq > 2) from genuine-heading detection, treating it as boilerplate noise
# ("LEARNING OBJECTIVES" repeats 12+ times on purpose). But when that
# repeating line itself NAMES a specific insurance type -- "Practice of Life
# Insurance", "Liability Insurance & Documents in General Insurance", "Health
# Insurance" -- it is real, printed, ground-truth per-document topic signal
# sitting unused in chunk text, not noise (RC-4). This matters most for
# exactly the documents Phase 1's doc_prior correctly refuses to anchor: a
# genuinely multi-topic reference textbook (e.g. "9.3 INSURANCE LAW AND
# PRACTICE.pdf") has doc_prior="general" by design, but its own chapters
# still print a distinct running header per chapter -- this is the signal
# that resolves THOSE sections correctly, complementary to doc_prior rather
# than redundant with it.
def _mine_page_header_topic(text: str) -> str:
    """Find a line that repeats >=3 times in *text* (a stricter bar than
    _extract_sections' >2-excludes cutoff, to avoid mining a two-page
    coincidence) AND itself resolves to a specific policy type via
    classify_query_policy_type() -- a repeating line that's just page
    furniture (a bare page code, "LEARNING OBJECTIVES") classifies
    "general" and is correctly skipped rather than mined as a false label.
    Returns "" when no such line exists."""
    freq: dict[str, int] = {}
    for line in text.split("\n"):
        stripped = line.strip()
        if 3 <= len(stripped) < 90:
            freq[stripped] = freq.get(stripped, 0) + 1
    repeating = [line for line, count in freq.items() if count >= 3]
    for line in sorted(repeating, key=lambda l: freq[l], reverse=True):
        if classify_query_policy_type(line) != "general":
            return line
    return ""


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

        A genuine bulleted list (see _split_bullet_items above) is handled
        BEFORE any of this — each item becomes its own chunk directly,
        never subject to the similarity/word-limit grouping decision at
        all, since that grouping is exactly what let several distinct
        bulleted tips collapse back into one diluted chunk (confirmed
        live, see that function's own comment for the full incident).
        """
        bullet_items = _split_bullet_items(text)
        if bullet_items is not None:
            # No overlap applied here, unlike the paragraph path below.
            # _apply_overlap prepends the last _overlap_words (60) words of
            # the PREVIOUS chunk to bridge context across a boundary that
            # cuts through continuous flowing prose — necessary there
            # because a paragraph chunk can be truncated mid-thought. A
            # bulleted item is already a complete, self-contained fact by
            # construction (that's the whole reason it's being kept as its
            # own chunk in the first place), so there's no mid-thought
            # continuity to bridge. Confirmed live 2026-08-31: applying the
            # same word-count overlap here backfired badly on SHORT items —
            # each bullet tip in this KB runs well under 60 words, so "the
            # last 60 words of the previous chunk" was the ENTIRE previous
            # item, producing a sliding 2-item window (chunk N = item[N-1]
            # + item[N]) instead of 4 clean, distinct chunks. Zero overlap
            # keeps each chunk exactly one item, which is the entire point.
            logger.info(
                "[SemanticChunker] detected a %d-item bulleted list -> %d chunks "
                "(one per item, bypassing similarity grouping and overlap)",
                len(bullet_items), len(bullet_items),
            )
            return bullet_items, [0] * len(bullet_items)

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

        Returns (chunks, overlap_sizes, section_headings, sub_headings) — a
        4-tuple. section_headings is the top-level heading each chunk
        belongs to ("" when no section structure was detected); sub_headings
        is the bold numbered/lettered sub-item heading within that section,
        if one was detected ("" otherwise — most chunks) — see
        _extract_sections/_split_by_subheadings for how it's found. Lets
        callers classify once per section and apply that result to every
        chunk sharing the same heading, instead of classifying each chunk
        independently, while still keeping each sub-item's own identity
        available in metadata.
        """
        sections = _extract_sections(text)
        all_chunks: List[str] = []
        all_overlap_sizes: List[int] = []
        all_headings: List[str] = []
        all_sub_headings: List[str] = []
        for heading, sub_heading, section_text in sections:
            section_chunks, section_overlaps = self._group_paragraphs_into_chunks(section_text)
            all_chunks.extend(section_chunks)
            all_overlap_sizes.extend(section_overlaps)
            all_headings.extend([heading] * len(section_chunks))
            all_sub_headings.extend([sub_heading] * len(section_chunks))

        if len(sections) > 1:
            logger.info(
                "[SemanticChunker] %d sections detected → %d total chunks",
                len(sections), len(all_chunks),
            )

        return all_chunks, all_overlap_sizes, all_headings, all_sub_headings

    @staticmethod
    def _apply_overlap(
        chunks: List[str], overlap_words: int
    ) -> tuple:
        """
        Prepend the LAST SENTENCE of chunk[i-1] — not a fixed word count —
        to the start of chunk[i]. Returns (new_chunks, overlap_word_counts)
        where overlap_word_counts[i] is the number of words prepended to
        chunk[i] (0 for chunk[0]).

        Confirmed live (2026-09-01): the old fixed-N-words tail cut a
        disproportionate share of a SHORT next chunk — 60 words is a small
        fraction of a normal ~500-word chunk but over half of a ~100-word
        one (a real case: a document's tail-end chunk after its bulk of
        content already went into the previous chunk). It also routinely
        started the next chunk mid-sentence (e.g. "investments. As the
        name of the plan specifies..."), which isn't broken but is
        visibly a raw slice, not a real unit of meaning. A whole-sentence
        overlap scales with the actual content being carried forward
        instead of an arbitrary cut, and never starts mid-sentence.

        overlap_words is kept as a fallback AND a safety cap for two
        cases a pure last-sentence approach can't cover on its own:
        - The previous chunk has no real internal sentence boundary at
          all (ends on a heading, a list marker, a fragment with no
          terminal punctuation) — _split_into_sentences then returns the
          WHOLE chunk as "one sentence", which would overlap the entire
          previous chunk rather than a bounded unit. Falls back to the
          old word-count tail in that case.
        - The genuine last sentence is unusually long (some regulatory/
          legal sentences in this KB run 100+ words) — capped to the
          last overlap_words words of it, so one pathological long
          sentence can't balloon the overlap unboundedly.
        """
        result: List[str] = [chunks[0]]
        sizes: List[int]  = [0]
        for i in range(1, len(chunks)):
            prev_text = chunks[i - 1].strip()
            prev_words = prev_text.split()
            prev_sentences = [s for s in _split_into_sentences(prev_text) if s.strip()]
            tail = prev_sentences[-1].strip() if prev_sentences else ""
            tail_words = tail.split()

            if not tail or (len(prev_sentences) <= 1 and len(tail_words) > overlap_words):
                # No real sentence boundary found — fall back to the old
                # fixed-word-count tail rather than overlapping the
                # entire previous chunk.
                tail_words = prev_words[-overlap_words:] if len(prev_words) > overlap_words else prev_words
                tail = " ".join(tail_words)
            elif len(tail_words) > overlap_words:
                # A genuine but unusually long final sentence — cap it.
                tail_words = tail_words[-overlap_words:]
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
        # section_id disambiguation: a heading's own TEXT recurring later in
        # the same document (e.g. "General Exclusions" under both a Health
        # chapter and a Motor chapter of one combined-lines PDF) must NOT
        # collapse into the same section_id — confirmed by inspection: the
        # bare f"{doc_source}::{heading}" key below has no way to tell two
        # such occurrences apart, so both would merge into one group and
        # get classified together as if they were one contiguous section.
        # Tracked per doc_source (a single split_documents() call can cover
        # multiple different source files) and carried across the outer
        # `for doc in docs` loop below, since one physical document's pages
        # can span more than one `doc` object after _merge_pages. Consecutive
        # chunks that keep the SAME heading value still share one occurrence
        # (that's the whole point of section_id — grouping a section's own
        # sibling chunks together) -- only a genuine transition INTO a
        # heading bumps the count, and only when that heading text has
        # already been used by an earlier, non-adjacent section.
        _prev_key_by_source: dict[str, tuple] = {}
        _key_occurrence_by_source: dict[str, dict[tuple, int]] = {}
        _current_occurrence_by_source: dict[str, int] = {}
        for doc in docs:
            page_value = (
                doc.metadata.get("page")
                or doc.metadata.get("page_number")
                or doc.metadata.get("page_num")
                or 0
            )
            pieces_raw, overlap_sizes, headings, sub_headings = self.split_text(doc.page_content)
            doc_source = doc.metadata.get("source") or doc.metadata.get("filename") or "doc"
            # Once per document (see _mine_page_header_topic above) -- a
            # fallback signal for chunks with no genuine detected heading,
            # not a replacement for one that already exists.
            page_header_topic = _mine_page_header_topic(doc.page_content)
            for idx, (piece, ov_size, heading, sub_heading) in enumerate(
                zip(pieces_raw, overlap_sizes, headings, sub_headings)
            ):
                markers = _PAGE_MARKER_RE.findall(piece)
                recovered_page = page_value
                if markers:
                    try:
                        nums = [int(m) for m in markers]
                        recovered_page = min(nums)
                    except (ValueError, TypeError):
                        recovered_page = markers[0]
                    piece = _PAGE_MARKER_RE.sub("", piece).strip()

                # Strip a verbatim occurrence of the mined running
                # header/footer line from this chunk's own text before it's
                # stored — otherwise it competes with the chunk's real
                # content as ordinary prose during body-keyword scoring
                # instead of acting as the label it actually is (Phase 3,
                # plan_policy_type_tagging.md RC-4). Applies whenever the
                # line is present, not only when it's about to stand in as
                # the fallback heading below — it's boilerplate either way.
                if page_header_topic and page_header_topic in piece:
                    piece = "\n".join(
                        line for line in piece.split("\n")
                        if line.strip() != page_header_topic
                    ).strip()

                # A mined page-header topic (see _mine_page_header_topic
                # above) stands in for section_heading ONLY when no genuine
                # heading was detected for this chunk — it never overrides
                # a real one, and deliberately does NOT affect section_id
                # grouping below, which stays keyed on the genuine heading
                # exactly as before. This just revives
                # regex_first_pass_policy_type()'s heading-check branch
                # (the strongest one, RC-1) for documents whose own
                # printed page furniture already states the topic, without
                # changing how chunks get grouped into sections.
                effective_heading = heading or page_header_topic

                # New section boundary (heading OR sub_heading changed from
                # the immediately preceding piece, tracked per doc_source)
                # — bump the occurrence count for this (heading, sub_heading)
                # pair so a later, non-adjacent recurrence of the same text
                # gets a distinct section_id instead of merging with an
                # earlier section. Keying on the PAIR (not heading alone)
                # means two different sub-items under the same parent
                # heading — "1) Term insurance" then "2) Whole life
                # insurance" — are correctly treated as separate sections,
                # not silently merged just because their parent heading
                # didn't change; a sub-item long enough to span multiple
                # chunks still shares one section_id, same as before.
                _key = (heading, sub_heading)
                if _key != _prev_key_by_source.get(doc_source):
                    if heading or sub_heading:
                        occ_map = _key_occurrence_by_source.setdefault(doc_source, {})
                        occ_map[_key] = occ_map.get(_key, 0) + 1
                        _current_occurrence_by_source[doc_source] = occ_map[_key]
                    _prev_key_by_source[doc_source] = _key

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
                        "section_heading":     effective_heading,
                        "sub_heading":         sub_heading,
                        "section_id":          (
                            f"{doc_source}::{heading}::{sub_heading}::{_current_occurrence_by_source[doc_source]}"
                            if heading or sub_heading else f"{doc_source}::chunk{idx}"
                        ),
                    },
                ))
        return result
