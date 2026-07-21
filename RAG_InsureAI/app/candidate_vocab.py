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
    "n_a",
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
    Cheap keyword-substring check against already-discovered candidate
    labels, tried before paying for another open-ended LLM call. First hit
    wins — this is only ever a hint for a field that never touches the
    retrieval filter, so it doesn't need the active vocabulary's stricter
    confidence-contest scoring.
    """
    candidates = get_candidate_vocab()
    if not candidates:
        return None
    t = text.lower()
    for label, info in candidates.items():
        for kw in info.get("keywords", []):
            if kw and kw.lower() in t:
                return label
    return None


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
