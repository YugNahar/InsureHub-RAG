"""
Persistent store for the open-vocabulary policy-type classification system.

Three files under $INSUREHUB_DATA_DIR/candidate_vocab/:
  active_vocab_extra.json  — promoted types, unioned with the hardcoded
                              _POLICY_TYPE_HINTS at read time
                              (see metadata_tagger.get_active_vocab()).
  candidate_vocab.json     — free-text label -> keyword hints, grows
                              automatically from open-ended guesses. Never
                              touches the retrieval filter — a wrong entry
                              here can only mis-tag a candidate field.
  candidate_log.jsonl      — append-only record of every open-ended guess,
                              the substrate for a future manual promotion
                              review (not built in this pass).

Writes are guarded by a plain threading.Lock rather than asyncio.Lock:
callers include both asyncio.to_thread ingestion worker threads and the
event loop thread directly (query-time classification), and only a real
OS-level lock correctly serializes across that mix — none of the existing
lock patterns elsewhere in this codebase cover that combination.
"""
import json
import logging
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DATA_DIR = os.path.join(
    os.getenv("INSUREHUB_DATA_DIR", os.path.expanduser("~/.insurehub")),
    "candidate_vocab",
)
_ACTIVE_VOCAB_EXTRA_PATH = os.path.join(_DATA_DIR, "active_vocab_extra.json")
_CANDIDATE_VOCAB_PATH = os.path.join(_DATA_DIR, "candidate_vocab.json")
_CANDIDATE_LOG_PATH = os.path.join(_DATA_DIR, "candidate_log.jsonl")

_lock = threading.Lock()

# Degenerate guesses the open-ended classifier sometimes returns instead of
# genuinely naming a topic ("I don't know" phrased as a label). Normalized
# away so two unrelated null guesses can't collide and falsely trigger the
# reranking candidate-match bypass (both would otherwise say "other").
_DEGENERATE_LABELS = {
    "other", "others", "unclear", "unknown", "none", "na",
    "general", "various", "misc", "miscellaneous", "mixed", "unsure",
    "not_applicable", "not_sure", "no_specific_type", "no_specific",
    "n_a", "insurance", "policy", "insurance_policy",
}


def _atomic_write_json(path: str, data: Any) -> None:
    tmp = path + ".tmp"
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
    except Exception as exc:
        logger.warning("[candidate_vocab] write failed for %s: %s", path, exc)
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
        logger.warning("[candidate_vocab] load failed for %s (%s) — using default", path, exc)
        return default


def normalize_candidate_label(raw: str) -> Optional[str]:
    """
    Lowercase, strip, collapse to underscores, drop degenerate non-answers.
    Returns None when the guess isn't a real label at all — callers must
    treat that as "no candidate" rather than storing an empty string, so a
    genuinely non-specific guess can never collide with another one.
    """
    if not raw:
        return None
    label = re.sub(r"[^a-z0-9\s_-]", "", raw.strip().lower())
    label = re.sub(r"[\s-]+", "_", label.strip()).strip("_")
    if not label or label in _DEGENERATE_LABELS:
        return None
    return label


def get_active_vocab_extra() -> Dict[str, Dict]:
    """
    Promoted types only — the part _POLICY_TYPE_HINTS doesn't already cover.
    metadata_tagger.get_active_vocab() unions this on top of the hardcoded 12.
    """
    return _load_json(_ACTIVE_VOCAB_EXTRA_PATH, {})


def promote_to_active_vocab(
    label: str, desc: str, keywords: List[str], regex: Optional[List[str]] = None
) -> None:
    """
    Move a candidate label into the active vocabulary. Intended for a
    phase-5 manual-review promotion endpoint — not wired to any endpoint in
    this pass, but the mechanism is here so promotion is a data write, not
    a code change or redeploy.
    """
    with _lock:
        extra = _load_json(_ACTIVE_VOCAB_EXTRA_PATH, {})
        extra[label] = {
            "desc": desc,
            "keywords": keywords,
            "regex": regex or [re.escape(kw) for kw in keywords if kw],
        }
        _atomic_write_json(_ACTIVE_VOCAB_EXTRA_PATH, extra)
        candidates = _load_json(_CANDIDATE_VOCAB_PATH, {})
        candidates.pop(label, None)
        _atomic_write_json(_CANDIDATE_VOCAB_PATH, candidates)
    logger.info("[candidate_vocab] promoted %r into active vocabulary", label)


def get_candidate_vocab() -> Dict[str, Dict]:
    return _load_json(_CANDIDATE_VOCAB_PATH, {})


def match_candidate_vocab(text: str) -> Optional[str]:
    """
    Cheap keyword-overlap check against already-discovered candidate
    labels, tried before paying for another open-ended LLM call. Requires
    at least 2 of a label's stored keywords present, not just 1 — confirmed
    live in a 203-chunk backfill run that a single-keyword match against
    generic insurance-domain vocabulary ("risk", "types", "loss",
    "business", "companies") let one broadly-worded label cheap-match
    ~200 genuinely unrelated chunks across 5 different source documents,
    all without ever re-verifying via the LLM. First label clearing the
    2-match bar wins — this is only ever a hint for a field that never
    touches the retrieval filter, so it doesn't need the active
    vocabulary's stricter confidence-contest scoring.
    """
    candidates = get_candidate_vocab()
    if not candidates:
        return None
    t = text.lower()
    for label, info in candidates.items():
        keywords = info.get("keywords", [])
        hits = sum(1 for kw in keywords if kw and kw.lower() in t)
        if hits >= 2:
            return label
    return None


# 2026-07-23: promotion wired up (previously built but never called — see
# module history). Threshold requires BOTH a real repeat count AND source
# diversity, not guess_count alone: candidate_vocab.json had accumulated
# entries like "pet_insurance" (guess_count=16) that were mostly this
# session's own repeated test-corpus runs hitting the same fixed query
# set, not independent real-world confirmation. Two guards against
# promoting noise:
#   - _PROMOTION_MIN_GUESS_COUNT: a one-off LLM guess (most candidates
#     never exceed guess_count=1-2, see live data) shouldn't become
#     permanent retrieval-filter vocabulary.
#   - _PROMOTION_MIN_DISTINCT_SOURCES: guess_count alone can't tell "16
#     independent confirmations" from "1 document re-chunked 16 times" or
#     "the same test query repeated 16 times" — distinct sources can.
_PROMOTION_MIN_GUESS_COUNT = 5
_PROMOTION_MIN_DISTINCT_SOURCES = 2


def _count_distinct_sources(label: str) -> int:
    """Distinct source signatures logged for `label` in candidate_log.jsonl
    — the real diversity signal guess_count alone can't provide."""
    sources = set()
    if not os.path.exists(_CANDIDATE_LOG_PATH):
        return 0
    try:
        with open(_CANDIDATE_LOG_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if rec.get("label") == label:
                    sources.add((rec.get("source") or "")[:120])
    except Exception as exc:
        logger.warning("[candidate_vocab] distinct-source count failed for %r: %s", label, exc)
    return len(sources)


def maybe_promote(label: str) -> bool:
    """
    Auto-promote `label` into the active vocabulary once it has crossed
    the evidence thresholds above. Called after every upsert_candidate()
    so promotion fires the moment a candidate qualifies, not on a
    separate schedule. Idempotent — already-promoted labels are skipped
    via the active-vocab check, not re-promoted or re-logged.
    """
    if label in get_active_vocab_extra():
        return False
    entry = get_candidate_vocab().get(label)
    if not entry or entry.get("guess_count", 0) < _PROMOTION_MIN_GUESS_COUNT:
        return False
    distinct = _count_distinct_sources(label)
    if distinct < _PROMOTION_MIN_DISTINCT_SOURCES:
        return False
    keywords = entry.get("keywords") or []
    # Query-side confirmations always pass keywords=[] (a short question
    # isn't a good source of distinguishing vocabulary — see
    # upsert_candidate's callers) — a type promoted mostly from query-side
    # evidence would otherwise get an EMPTY regex, silently defeating the
    # whole point of promotion: the fast/cheap regex classifier that runs
    # before any LLM call would have nothing to match on. Confirmed live
    # 2026-07-23: "pet_insurance" (16 guesses, all source_type="query")
    # promoted with keywords=[] before this fallback existed. Guarantee at
    # least the label's own natural-language phrase is always matchable.
    natural_phrase = label.replace("_", " ")
    if natural_phrase not in keywords:
        keywords = keywords + [natural_phrase]
    desc = (
        f"Auto-promoted open-vocabulary type: {natural_phrase}. "
        f"Seen {entry.get('guess_count')} times across {distinct} distinct sources."
    )
    promote_to_active_vocab(label, desc, keywords)
    logger.info(
        "[candidate_vocab] AUTO-PROMOTED %r into active vocabulary "
        "(guess_count=%d, distinct_sources=%d)",
        label, entry.get("guess_count"), distinct,
    )
    return True


def upsert_candidate(label: str, keywords: List[str], source: str, source_type: str) -> None:
    """
    Record one open-ended guess: grow the candidate vocabulary and append to
    the guess log. `label` must already be normalized (see
    normalize_candidate_label) and non-None — callers drop degenerate
    guesses before reaching this point, so nothing here needs to re-check.
    `source_type` is "chunk" or "query".
    """
    now = time.time()
    with _lock:
        candidates = _load_json(_CANDIDATE_VOCAB_PATH, {})
        entry = candidates.get(label)
        if entry is None:
            candidates[label] = {
                "keywords": keywords[:10],
                "first_seen": now,
                "last_seen": now,
                "guess_count": 1,
            }
        else:
            entry["last_seen"] = now
            entry["guess_count"] = entry.get("guess_count", 0) + 1
            existing_kw = set(entry.get("keywords", []))
            for kw in keywords:
                if kw not in existing_kw and len(entry["keywords"]) < 20:
                    entry["keywords"].append(kw)
                    existing_kw.add(kw)
        _atomic_write_json(_CANDIDATE_VOCAB_PATH, candidates)

        try:
            os.makedirs(_DATA_DIR, exist_ok=True)
            with open(_CANDIDATE_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "label": label,
                    "source": source[:300],
                    "source_type": source_type,
                    "timestamp": now,
                }) + "\n")
        except Exception as exc:
            logger.warning("[candidate_vocab] log append failed: %s", exc)

    logger.info("[candidate_vocab] guess %r from %s %r", label, source_type, source[:80])
    # Outside the lock above — maybe_promote() takes its own lock via
    # promote_to_active_vocab(); nesting them would deadlock (plain
    # threading.Lock, not reentrant).
    try:
        maybe_promote(label)
    except Exception as exc:
        logger.warning("[candidate_vocab] promotion check failed for %r: %s", label, exc)
