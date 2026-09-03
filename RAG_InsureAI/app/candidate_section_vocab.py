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

Deliberately simpler than candidate_vocab.py — no "promotion to an
active/closed vocabulary" step. A policy_type candidate gets promoted
because retrieval filtering needs a genuine closed list to build
`$in` clauses from. A section candidate never touches retrieval
filtering at all (see multi_source_rag.py's guaranteed-inclusion step) —
it only ever needs to be MATCHABLE via its own candidate_section
metadata field, which is already true the moment it's first
discovered. One file, one job: label -> keyword hints, grows
automatically.
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

_lock = threading.Lock()

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
