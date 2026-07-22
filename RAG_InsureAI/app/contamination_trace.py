"""
Diagnostics for cross-topic contamination in point-based answers.
Phase 0 of the contamination root-cause plan (see plan.md at the repo
root) — every contamination fix before this was diagnosed from a single
live repro at a time, with no way to see in bulk which chunk tag or
score band actually produces leakage. This module is the instrument
that makes it visible and measurable instead.

Opt-in via CONTAMINATION_TRACE=1 (mirrors multi_source_rag.py's own
TIMING log convention — cheap to leave off in normal operation, since a
disabled trace skips the extra cross-encoder scoring call entirely, not
just the write). Writes one JSON line per detailed/point answer to
$INSUREHUB_DATA_DIR/contamination_trace/trace.jsonl containing:
  - the query and its classified policy type(s)
  - every retrieved chunk: source, policy_type, candidate_policy_type,
    rerank_score
  - every surviving point, its best-matching source chunk (by word
    overlap, a generous approximation used for attribution only — NOT
    the enforcement signal), that chunk's policy_type, and a per-point
    query-relevance score from the shared cross-encoder

Read the trace file with `jq` or the corpus runner
(contamination_corpus_runner.py) rather than eyeballing raw logs.
"""
import json
import logging
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

TRACE_ENABLED = os.getenv("CONTAMINATION_TRACE", "").strip().lower() in ("1", "true", "yes")

_DATA_DIR = os.path.join(
    os.getenv("INSUREHUB_DATA_DIR", os.path.expanduser("~/.insurehub")),
    "contamination_trace",
)
_TRACE_PATH = os.path.join(_DATA_DIR, "trace.jsonl")
_lock = threading.Lock()

_WORD_RE = re.compile(r"\w+")


def _norm_word(w: str) -> str:
    # Same light plural stemming as multi_source_rag.py's grounding
    # check — duplicated deliberately rather than imported, since this
    # module is a leaf diagnostic utility multi_source_rag.py imports
    # FROM, not the reverse; keeping it dependency-free avoids a circular
    # import.
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def _norm_words(text: str) -> List[str]:
    return [_norm_word(w) for w in _WORD_RE.findall(text.lower())]


def best_matching_chunk_index(point_text: str, chunk_texts: List[str]) -> Optional[int]:
    """Return the index of the chunk in *chunk_texts* with the highest
    word-overlap against *point_text*, or None if no chunk shares any
    significant (4+ letter) word with it.

    This is diagnostic attribution only, deliberately generous/approximate
    (plain word overlap, no threshold) — it exists to answer "which chunk
    did this point most likely come from" for a human reading the trace,
    not to gate anything. The actual enforcement signal is the per-point
    cross-encoder relevance score, computed separately.
    """
    point_words = {w for w in _norm_words(point_text) if len(w) >= 4}
    if not point_words:
        return None
    best_idx, best_score = None, 0
    for i, text in enumerate(chunk_texts):
        chunk_words = set(_norm_words(text or ""))
        score = sum(1 for w in point_words if w in chunk_words)
        if score > best_score:
            best_idx, best_score = i, score
    return best_idx


def write_trace(record: Dict[str, Any]) -> None:
    """Append one JSON line to the trace file. Never raises — a broken
    trace write must never affect the live answer being streamed to the
    user; the caller already wraps this in its own try/except as a
    second layer of protection, but this function is defensive on its
    own too.
    """
    if not TRACE_ENABLED:
        return
    record = dict(record)
    record.setdefault("timestamp", time.time())
    try:
        with _lock:
            os.makedirs(_DATA_DIR, exist_ok=True)
            with open(_TRACE_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
    except Exception as exc:
        logger.warning("[contamination_trace] write failed: %s", exc)
