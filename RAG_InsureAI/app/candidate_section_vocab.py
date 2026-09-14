"""
Open-vocabulary SECTION-category discovery — the section-side sibling of
candidate_vocab.py (which does the same thing for policy_type).

_detect_section() (rag.py) is a fixed, closed list of ~13 categories
(benefits/exclusions/claims/definitions/eligibility/flight_delay/medical/
baggage/legislation/types_of_insurance/principles/history/case_law/
chapter) with no fallback — content that doesn't match any of them
always lands on "general", with no way to ever recognize it again later
even if the exact same kind of content shows up repeatedly across many
documents. This module is that missing fallback: a free-text,
self-growing label store, populated from chunk text at ingestion time
(classify_candidate_section() in metadata_tagger.py) and matched
cheaply at query time (match_candidate_section_vocab(), no LLM call)
against whatever's already been discovered.

2026-09-14 update: promotion WAS added after all, mirroring
candidate_vocab.py's mechanism — not because section needs a closed
`$in` retrieval filter the way policy_type does (it still doesn't,
see multi_source_rag.py's guaranteed-inclusion step), but because the
query-side embedding classifier (_get_query_section_prototype_embeddings
in multi_source_rag.py) only ever compared a query against the fixed
12 hand-written prototype sentences — a section category discovered
repeatedly at ingestion had no way to become recognizable to THAT
tier at query time, only to the much weaker literal-2-keyword-overlap
fast path and an LLM call gated behind it. Promotion here means "add
this label's own generated prototype sentence to the embedding
comparison set", not "add a hard retrieval filter" — see
get_active_section_vocab_extra() and multi_source_rag.py's
get_active_section_vocab().
"""
import json
import logging
import os
import re
import threading
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DATA_DIR = os.path.join(
    os.getenv("INSUREHUB_DATA_DIR", os.path.expanduser("~/.insurehub")),
    "candidate_vocab",
)
_CANDIDATE_SECTION_PATH = os.path.join(_DATA_DIR, "candidate_section_vocab.json")
_ACTIVE_SECTION_VOCAB_EXTRA_PATH = os.path.join(_DATA_DIR, "active_section_vocab_extra.json")

_lock = threading.Lock()

# Same two-guard bar candidate_vocab.py uses for policy_type promotion
# (see that module's own maybe_promote() for the full rationale: raw
# guess_count alone is gameable by one document re-chunked/re-queried
# many times, only independently-confirmed source diversity tells real
# repeat evidence apart from that). Kept at the identical values for
# consistency, not because section's stakes independently calibrate to
# the same numbers.
_PROMOTION_MIN_GUESS_COUNT = 5
_PROMOTION_MIN_DISTINCT_SOURCES = 2

# Same degenerate-answer guard as candidate_vocab.py's own list, plus a
# couple of section-specific non-answers an open-ended classifier might
# produce for content that just restates "this is a general passage."
_DEGENERATE_LABELS = {
    "other", "others", "unclear", "unknown", "none", "na", "general",
    "various", "misc", "miscellaneous", "mixed", "unsure",
    "not_applicable", "not_sure", "no_specific_type", "no_specific",
    "n_a", "section", "content", "text", "passage", "information",
}


def _atomic_write_json(path: str, data: Any) -> None:
    tmp = path + ".tmp"
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
    except Exception as exc:
        logger.warning("[candidate_section_vocab] write failed for %s: %s", path, exc)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _load_json(path: str, default: Any) -> Any:
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("[candidate_section_vocab] load failed for %s (%s) — using default", path, exc)
        return default


def normalize_section_label(raw: str) -> Optional[str]:
    if not raw:
        return None
    label = re.sub(r"[^a-z0-9\s_-]", "", raw.strip().lower())
    label = re.sub(r"[\s-]+", "_", label.strip()).strip("_")
    if not label or label in _DEGENERATE_LABELS:
        return None
    return label


def get_candidate_section_vocab() -> Dict[str, Dict]:
    with _lock:
        return _load_json(_CANDIDATE_SECTION_PATH, {})


def match_candidate_section_vocab(text: str) -> Optional[str]:
    """Cheap keyword-overlap check against already-discovered candidate
    section labels — same 2-keyword-minimum bar as candidate_vocab.py's
    match_candidate_vocab(), for the identical reason: a single generic
    shared word ("policy", "insured") would otherwise cheap-match almost
    anything."""
    candidates = get_candidate_section_vocab()
    if not candidates:
        return None
    t = (text or "").lower()
    for label, info in candidates.items():
        keywords = info.get("keywords", [])
        hits = sum(1 for kw in keywords if kw and kw.lower() in t)
        if hits >= 2:
            return label
    return None


def get_active_section_vocab_extra() -> Dict[str, Dict]:
    """Promoted section labels only — multi_source_rag.py's
    get_active_section_vocab() unions this on top of the hardcoded
    _QUERY_SECTION_PROTOTYPES categories, the section-side mirror of
    candidate_vocab.get_active_vocab_extra()."""
    return _load_json(_ACTIVE_SECTION_VOCAB_EXTRA_PATH, {})


def promote_to_active_section_vocab(label: str, desc: str, keywords: List[str]) -> None:
    with _lock:
        extra = _load_json(_ACTIVE_SECTION_VOCAB_EXTRA_PATH, {})
        extra[label] = {"desc": desc, "keywords": keywords}
        _atomic_write_json(_ACTIVE_SECTION_VOCAB_EXTRA_PATH, extra)
        candidates = _load_json(_CANDIDATE_SECTION_PATH, {})
        candidates.pop(label, None)
        _atomic_write_json(_CANDIDATE_SECTION_PATH, candidates)
    logger.info("[candidate_section_vocab] promoted %r into active section vocabulary", label)


def maybe_promote_section(label: str) -> bool:
    """Auto-promote `label` into the active/embedded section vocabulary
    once it crosses the evidence bar above — called after every
    upsert_candidate_section() so promotion fires the moment a candidate
    qualifies. Idempotent — an already-promoted label is skipped via the
    active-vocab check.

    Unlike candidate_vocab.py's maybe_promote(), distinct-source count is
    read directly off the candidate's own already-deduplicated `sources`
    list rather than scanning a separate append-only log — this module
    never needed that log, since a section label's cheap-match field
    never had policy_type's hard-retrieval-filter stakes to justify the
    extra audit trail (see the module docstring's 2026-09-14 update)."""
    if label in get_active_section_vocab_extra():
        return False
    entry = get_candidate_section_vocab().get(label)
    if not entry or entry.get("guess_count", 0) < _PROMOTION_MIN_GUESS_COUNT:
        return False
    distinct = len(entry.get("sources", []))
    if distinct < _PROMOTION_MIN_DISTINCT_SOURCES:
        return False
    keywords = entry.get("keywords") or []
    natural_phrase = label.replace("_", " ")
    if natural_phrase not in keywords:
        keywords = keywords + [natural_phrase]
    desc = (
        f"Auto-promoted open-vocabulary section: {natural_phrase}. "
        f"Seen {entry.get('guess_count')} times across {distinct} distinct sources."
    )
    promote_to_active_section_vocab(label, desc, keywords)
    logger.info(
        "[candidate_section_vocab] AUTO-PROMOTED %r into active section vocabulary "
        "(guess_count=%d, distinct_sources=%d)",
        label, entry.get("guess_count"), distinct,
    )
    return True


def upsert_candidate_section(label: str, keywords: List[str], source: str) -> None:
    with _lock:
        candidates = _load_json(_CANDIDATE_SECTION_PATH, {})
        entry = candidates.get(label, {"keywords": [], "sources": [], "guess_count": 0})
        existing_kw = set(entry.get("keywords", []))
        for kw in keywords:
            existing_kw.add(kw)
        entry["keywords"] = sorted(existing_kw)[:20]
        sources = set(entry.get("sources", []))
        if source:
            sources.add(source)
        entry["sources"] = sorted(sources)[:50]
        entry["guess_count"] = entry.get("guess_count", 0) + 1
        candidates[label] = entry
        _atomic_write_json(_CANDIDATE_SECTION_PATH, candidates)
    # Outside the lock above — maybe_promote_section() takes its own lock
    # via promote_to_active_section_vocab(); nesting would deadlock (plain
    # threading.Lock, not reentrant) — same discipline as
    # candidate_vocab.py's upsert_candidate().
    try:
        maybe_promote_section(label)
    except Exception as exc:
        logger.warning("[candidate_section_vocab] promotion check failed for %r: %s", label, exc)
